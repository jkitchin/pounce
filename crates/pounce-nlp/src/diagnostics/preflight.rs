//! Starting-point preflight: evaluate a model once at `x0`, before any
//! solve, and report what the initializer and the first iteration will see.
//!
//! # Why this exists
//!
//! A local NLP solver's fate is largely decided at iteration 0, but the
//! solver only reports starting-point trouble *after* it has tripped over
//! it (`Invalid_Number_Detected` mid-solve, immediate restoration, a slow
//! crawl caused by scaling). This runs before the solve and reports:
//!
//! * **Non-finite evaluations** — NaN/inf in `f`, `∇f`, `g`, the Jacobian,
//!   or the Hessian at x0. These are fatal: the solve would abort.
//! * **Bound violations of x0** and components sitting exactly on a bound
//!   (the interior clamp will move both).
//! * **Interior-clamp displacement** — the `bound_push` / `bound_frac`
//!   clamp (`DefaultIterateInitializer`) applied to x0, so "the solver
//!   silently moved my point" is visible up front.
//! * **Initial constraint violation**, **derivative scale spread**, and the
//!   factors automatic (gradient-based) scaling will pick here.
//!
//! [`check_tnlp`] is the whole check and takes any `&mut dyn TNLP`, so a
//! library embedder gets the same report the `pounce check-x0` subcommand
//! renders. The CLI owns argument parsing, `.nl` loading and formatting;
//! this module owns the evaluation.

use super::{RowReport, box_violation, name_at};
use crate::orig_ipopt_nlp::{gradient_obj_scale, gradient_row_scale, gradient_scaling_fires};
use crate::tnlp::{BoundsInfo, SparsityRequest, StartingPoint, TNLP};
use pounce_common::types::{Number, lower_bound_present, upper_bound_present};

/// An explicit starting point supplied by the caller, replacing the
/// model's own `get_starting_point`.
///
/// `source` is a free-form label for the report (the CLI passes the
/// `--x0-file` path); the check itself only reads `x`.
#[derive(Debug, Clone)]
pub struct X0Override {
    pub x: Vec<Number>,
    pub source: String,
}

/// Numerical knobs for [`check_tnlp`].
///
/// These are the parts of the CLI's `check-x0` arguments that change what is
/// *measured*, separated from the parts that change where input comes from
/// or how output is rendered — those stay in the CLI.
#[derive(Debug, Clone)]
pub struct PreflightOptions {
    /// Violations above this are counted in `n_violated` (default 1e-6).
    pub feas_tol: Number,
    /// `bound_push` used for the clamp preview (default 1e-2).
    pub bound_push: Number,
    /// `bound_frac` used for the clamp preview (default 1e-2).
    pub bound_frac: Number,
    /// Max offenders listed per category (default 5).
    pub max_list: usize,
    /// `nlp_scaling_max_gradient` for the scaling preview (default 100).
    pub scaling_max_gradient: Number,
    /// Replace the model's own starting point with this one.
    pub x0: Option<X0Override>,
}

impl Default for PreflightOptions {
    fn default() -> Self {
        PreflightOptions {
            feas_tol: 1e-6,
            bound_push: 1e-2,
            bound_frac: 1e-2,
            max_list: 5,
            scaling_max_gradient: NLP_SCALING_MAX_GRADIENT,
            x0: None,
        }
    }
}

/// One non-finite evaluation entry.
#[derive(Debug, Clone)]
pub struct NonFinite {
    pub index: usize,
    pub name: String,
    pub value: Number,
}

/// One Jacobian/Hessian non-finite entry (row/col in matrix coordinates).
#[derive(Debug, Clone)]
pub struct NonFiniteEntry {
    pub row: usize,
    pub col: usize,
    pub row_name: String,
    pub col_name: String,
    pub value: Number,
}

/// One interior-clamp displacement entry.
#[derive(Debug, Clone)]
pub struct ClampMove {
    pub index: usize,
    pub name: String,
    pub from: Number,
    pub to: Number,
    pub distance: Number,
}

/// Max/min-nonzero magnitude summary of a derivative array at x0.
#[derive(Debug, Clone, Default)]
pub struct ScaleSpread {
    pub max_abs: Number,
    pub min_abs_nonzero: Number,
    /// `max_abs / min_abs_nonzero`, or 0 when there are no nonzeros.
    pub ratio: Number,
}

/// `nlp_scaling_max_gradient`'s default — the cutoff above which
/// gradient-based scaling fires. Overridable with `--scaling-max-gradient`
/// so a preflight can preview the same cutoff the solve will run under.
pub const NLP_SCALING_MAX_GRADIENT: Number = 100.0;

/// `nlp_scaling_min_value`'s default — the floor on a computed scale.
pub const NLP_SCALING_MIN_VALUE: Number = 1e-8;

/// One row block's share of the gradient-based scaling preview.
///
/// The equalities (`c`) and the inequalities (`d`) are gated *separately*
/// upstream: unless some row of the block exceeds the cutoff, the whole
/// block is left unscaled. So `fires` is not redundant with `n_scaled`.
#[derive(Debug, Clone, Default)]
pub struct RowScaleBlock {
    pub n_rows: usize,
    /// Whether the block gets a scale vector at all.
    pub fires: bool,
    /// Rows the block would scale down (factor < 1). Zero when `!fires`.
    pub n_scaled: usize,
    /// Smallest factor assigned, or 1.0 when nothing is scaled.
    pub min_scale: Number,
    /// Rows driven all the way to `nlp_scaling_min_value`.
    pub n_at_floor: usize,
    /// Rows whose Jacobian is **entirely zero** at x0. These are the rows
    /// the sample cannot see; they come out at 1.0 whatever their
    /// coefficients are.
    pub n_zero_jac: usize,
}

/// What `nlp_scaling_method=gradient-based` will do to this model at this
/// x0 — the solver's own arithmetic
/// ([`gradient_obj_scale`] / [`gradient_row_scale`]), not a copy of it.
///
/// # Why a preflight reports this
///
/// Gradient-based scaling is a **point sample**: it reads `∇f` and the
/// Jacobian once, at x0, and never looks again. That is a good estimator
/// of a row's magnitude when the row's derivative at x0 is representative
/// of its derivative elsewhere, and no estimator at all when it is not.
/// The extreme case is a row whose Jacobian *vanishes* at x0 — a
/// `½xᵀQx ≤ b` written about the origin, started from `x = 0`. The sample
/// reads zero, the row is left at factor 1.0, and however badly `Q` and `b`
/// disagree in magnitude the scaler has no way to know it. That is how
/// AMPL emits `qcqp1000-2c` — every variable free, no initial guess, and
/// `k` rows of pure `½xᵀQᵢx ≤ bᵢ`.
///
/// So the two halves of this report are complementary: the block below
/// says what the sample decided, and [`Self::quad_rows`] says what the
/// sample could not see. See `dev-notes/quadratic-structure-exploitation.md`
/// §8 (gh #703).
#[derive(Debug, Clone, Default)]
pub struct ScalingPreview {
    /// The `nlp_scaling_max_gradient` cutoff this preview assumed.
    pub max_gradient: Number,
    /// `‖∇f(x0)‖_∞`.
    pub max_grad_f: Number,
    /// The objective factor `df` the scaler will pick.
    pub obj_scale: Number,
    /// Equality rows (`g_l == g_u`).
    pub c: RowScaleBlock,
    /// Everything else — the inequality and range rows.
    pub d: RowScaleBlock,
    /// Rows recognized as quadratic, worst mismatch first. `.nl` models
    /// only; empty for a builtin or a model with no quadratic row.
    pub quad_rows: Vec<QuadRowScale>,
    /// Total quadratic rows found (`quad_rows` is capped at `--max-list`).
    pub n_quad_rows: usize,
    /// Of those, how many the sample leaves at factor 1.0.
    pub n_quad_unscaled: usize,
    /// Of those, how many have an identically-zero Jacobian row at x0.
    pub n_quad_zero_jac: usize,
    /// Largest `rhs / curvature` over the quadratic rows, or 0 when there
    /// are none.
    pub max_quad_mismatch: Number,
}

/// The coefficient magnitudes of one quadratic constraint row, read off
/// the `.nl` without reference to any point, paired with what the
/// gradient sample at x0 made of it.
///
/// `curvature` is `‖Q‖_∞` (the largest absolute row sum of the row's
/// Hessian), which is Gershgorin's bound on `λ_max(Q)` and is the exact
/// quantity §8's second-stage row scale `eᵢ = 1/max(‖Qᵢ‖_∞, ‖aᵢ‖_∞, |bᵢ|)`
/// is built from. It is an upper bound on the curvature, not the curvature.
#[derive(Debug, Clone)]
pub struct QuadRowScale {
    pub index: usize,
    pub name: String,
    /// `‖Q‖_∞` — see the type docs.
    pub curvature: Number,
    /// `‖a‖_∞` over the `.nl` linear section plus the degree-1 terms the
    /// writer folded into the nonlinear tree.
    pub linear: Number,
    /// `|b|` — the finite bound the row is written against, shifted by the
    /// folded constant. A range row reports the larger magnitude.
    pub rhs: Number,
    /// `‖∇g(x0)‖_∞` — what gradient-based scaling actually samples.
    pub jac_at_x0: Number,
    /// The factor that sample assigns the row.
    pub scale: Number,
    /// `rhs / curvature`. §8's statistic: the `qcqp1500-1c` right-hand
    /// sides are 1.58e5–1.80e5 against `λ_max(Qᵢ) ≈ 1.6e3`, a 100×
    /// mismatch that biases `sᵢ = −gᵢ(x)` and hence the `−sᵢ/λᵢ` KKT
    /// diagonal. Zero when the curvature is zero.
    pub mismatch: Number,
}

// `QuadRowCoef` / `quad_row_coefs` live in `pounce_nl::nl_scaling`, where
// `nlp_scaling_method=curvature-based` also reads them (gh #703). One
// implementation: a preflight that reported different magnitudes from the
// ones the scaler acts on would be worse than no preflight.
/// One constraint row's quadratic coefficient census, as the scaling
/// preview reads it.
///
/// The *producer* of this census is `.nl`-specific (it reads coefficients
/// off the model's linear section and nonlinear tree, which a bare TNLP
/// cannot expose), and stays in `pounce_nl::nl_scaling::quad_row_coefs`.
/// The record itself is plain numbers, so it lives here with the check that
/// consumes it — a frontend that can supply the census gets the fuller
/// report, and one that cannot passes `&[]`.
#[derive(Debug, Clone, Copy)]
pub struct QuadRowCoef {
    pub index: usize,
    /// `‖Q‖_∞` — see the type docs.
    pub curvature: Number,
    /// `‖a‖_∞` over the `.nl` linear section plus the degree-1 terms the
    /// writer folded into the nonlinear tree.
    pub linear: Number,
    /// `|b|` — the finite bound the row is written against, shifted by the
    /// folded constant. A range row reports the larger magnitude.
    pub rhs: Number,
}

/// Reproduce gradient-based scaling's decision at x0.
///
/// `jac_row_max[i]` is row `i`'s Jacobian ∞-norm at x0, seeded the way
/// upstream seeds it (`f64::MIN_POSITIVE`) so an all-zero row is
/// distinguishable from a row of zeros that never appeared — both come out
/// at 1.0, which is the point.
fn scaling_preview(
    jac_row_max: &[Number],
    g_l: &[Number],
    g_u: &[Number],
    max_grad_f: Number,
    quad_coefs: &[QuadRowCoef],
    con_names: &[String],
    opts: &PreflightOptions,
) -> ScalingPreview {
    let max_gradient = opts.scaling_max_gradient;
    let max_list = opts.max_list;
    let m = jac_row_max.len();
    let is_equality =
        |i: usize| lower_bound_present(g_l[i]) && upper_bound_present(g_u[i]) && g_l[i] == g_u[i];
    let c_rows: Vec<Number> = (0..m)
        .filter(|&i| is_equality(i))
        .map(|i| jac_row_max[i])
        .collect();
    let d_rows: Vec<Number> = (0..m)
        .filter(|&i| !is_equality(i))
        .map(|i| jac_row_max[i])
        .collect();

    let block = |rows: &[Number]| -> RowScaleBlock {
        let fires = gradient_scaling_fires(rows, max_gradient, 0.0);
        let mut b = RowScaleBlock {
            n_rows: rows.len(),
            fires,
            min_scale: 1.0,
            ..Default::default()
        };
        for &r in rows {
            if r <= 0.0 || r == Number::MIN_POSITIVE {
                b.n_zero_jac += 1;
            }
            if !fires {
                continue;
            }
            let s = gradient_row_scale(r, max_gradient, NLP_SCALING_MIN_VALUE, 0.0);
            if s < 1.0 {
                b.n_scaled += 1;
            }
            if s <= NLP_SCALING_MIN_VALUE {
                b.n_at_floor += 1;
            }
            b.min_scale = b.min_scale.min(s);
        }
        b
    };
    let c = block(&c_rows);
    let d = block(&d_rows);

    // Each quadratic row's factor comes from whichever block it is in.
    let mut quad_rows: Vec<QuadRowScale> = quad_coefs
        .iter()
        .map(|q| {
            let i = q.index;
            let fires = if is_equality(i) { c.fires } else { d.fires };
            let scale = if fires {
                gradient_row_scale(jac_row_max[i], max_gradient, NLP_SCALING_MIN_VALUE, 0.0)
            } else {
                1.0
            };
            let raw = jac_row_max[i];
            QuadRowScale {
                index: i,
                name: name_at(con_names, i, 'c'),
                curvature: q.curvature,
                linear: q.linear,
                rhs: q.rhs,
                jac_at_x0: if raw == Number::MIN_POSITIVE {
                    0.0
                } else {
                    raw
                },
                scale,
                mismatch: if q.curvature > 0.0 {
                    q.rhs / q.curvature
                } else {
                    0.0
                },
            }
        })
        .collect();

    let n_quad_rows = quad_rows.len();
    let n_quad_unscaled = quad_rows.iter().filter(|q| q.scale >= 1.0).count();
    let n_quad_zero_jac = quad_rows.iter().filter(|q| q.jac_at_x0 == 0.0).count();
    let max_quad_mismatch = quad_rows
        .iter()
        .fold(0.0_f64, |m: f64, q| m.max(q.mismatch));

    quad_rows.sort_by(|a, b| {
        b.mismatch
            .partial_cmp(&a.mismatch)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.index.cmp(&b.index))
    });
    quad_rows.truncate(max_list);

    ScalingPreview {
        max_gradient,
        max_grad_f,
        obj_scale: gradient_obj_scale(max_grad_f, max_gradient, NLP_SCALING_MIN_VALUE, 0.0),
        c,
        d,
        quad_rows,
        n_quad_rows,
        n_quad_unscaled,
        n_quad_zero_jac,
        max_quad_mismatch,
    }
}

/// The fully-evaluated preflight result.
#[derive(Debug)]
pub struct PreflightOutcome {
    pub n_vars: usize,
    pub n_cons: usize,
    pub nl_sha256: Option<String>,
    pub source: String,
    pub x0_source: String,
    pub x0_all_zero: bool,
    pub objective: Option<Number>,
    // non-finite scans (counts are totals; lists are capped at max_list)
    pub grad_nonfinite: Vec<NonFinite>,
    pub grad_nonfinite_count: usize,
    pub g_nonfinite: Vec<NonFinite>,
    pub g_nonfinite_count: usize,
    pub jac_nonfinite: Vec<NonFiniteEntry>,
    pub jac_nonfinite_count: usize,
    /// `None` when the TNLP declines exact Hessians (quasi-Newton).
    pub hess_nonfinite_count: Option<usize>,
    // x0 vs bounds
    pub bound_violations: Vec<RowReport>,
    pub n_bound_violations: usize,
    pub max_bound_violation: Number,
    pub n_on_bounds: usize,
    // interior-clamp preview
    pub clamp_moves: Vec<ClampMove>,
    pub n_clamp_moved: usize,
    pub max_clamp_move: Number,
    // initial constraint violation
    pub con_violations: Vec<RowReport>,
    pub n_con_violations: usize,
    pub max_con_violation: Number,
    // derivative scale spread
    pub grad_spread: ScaleSpread,
    pub jac_spread: ScaleSpread,
    // what the automatic scaler will make of this x0
    pub scaling: ScalingPreview,
    // rollup
    pub warnings: Vec<String>,
    pub fatal: bool,
    pub verdict: &'static str,
}

/// The core preflight over any TNLP. Public so the debugger / tests can
/// reuse it without going through a file.
///
/// The scaling preview's quadratic-row census is empty here: a bare TNLP
/// exposes derivatives, not coefficients, so there is nothing to read it
/// from. [`check_tnlp_with_quadratics`] takes the census a `.nl` model can
/// supply.
pub fn check_tnlp(
    tnlp: &mut dyn TNLP,
    var_names: &[String],
    con_names: &[String],
    nl_sha256: Option<String>,
    source: String,
    opts: &PreflightOptions,
) -> Result<PreflightOutcome, String> {
    check_tnlp_with_quadratics(tnlp, var_names, con_names, nl_sha256, source, &[], opts)
}

/// [`check_tnlp`] plus the quadratic-row coefficients read off the model's
/// `.nl` (see [`quad_row_coefs`]), which is the half of the scaling report
/// no evaluation at x0 can produce.
#[allow(clippy::too_many_arguments)]
pub fn check_tnlp_with_quadratics(
    tnlp: &mut dyn TNLP,
    var_names: &[String],
    con_names: &[String],
    nl_sha256: Option<String>,
    source: String,
    quad_coefs: &[QuadRowCoef],
    opts: &PreflightOptions,
) -> Result<PreflightOutcome, String> {
    let info = tnlp.get_nlp_info().ok_or("get_nlp_info failed")?;
    let n = info.n.max(0) as usize;
    let m = info.m.max(0) as usize;
    let nnz = info.nnz_jac_g.max(0) as usize;
    let nnz_h = info.nnz_h_lag.max(0) as usize;
    let fortran = matches!(info.index_style, crate::tnlp::IndexStyle::Fortran);
    let off = if fortran { 1usize } else { 0usize };

    // --- bounds ---
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    if !tnlp.get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) {
        return Err("get_bounds_info failed".to_string());
    }

    // --- starting point ---
    let mut x = vec![0.0; n];
    let (mut zl_buf, mut zu_buf, mut lam_buf) = (vec![0.0; n], vec![0.0; n], vec![0.0; m]);
    let x0_source = if let Some(over) = &opts.x0 {
        if over.x.len() != n {
            // Reported against `source`, not a generic phrase: the caller's
            // label is the only thing that says *which* starting point is
            // the wrong length (the CLI passes the `--x0-file` path).
            return Err(format!(
                "{} has {} values but the problem has {n} variables",
                over.source,
                over.x.len()
            ));
        }
        x.copy_from_slice(&over.x);
        over.source.clone()
    } else {
        if !tnlp.get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x,
            init_z: false,
            z_l: &mut zl_buf,
            z_u: &mut zu_buf,
            init_lambda: false,
            lambda: &mut lam_buf,
        }) {
            return Err("get_starting_point failed".to_string());
        }
        "model".to_string()
    };
    let x0_all_zero = n > 0 && x.iter().all(|v| *v == 0.0);

    // --- evaluations at x0 ---
    let objective = tnlp.eval_f(&x, true);
    let obj_finite = objective.map(|v| v.is_finite()).unwrap_or(false);

    let mut grad_f = vec![0.0; n];
    let grad_ok = tnlp.eval_grad_f(&x, false, &mut grad_f);
    let (grad_nonfinite, grad_nonfinite_count) =
        scan_nonfinite(&grad_f, var_names, 'x', opts.max_list, grad_ok);

    let mut g = vec![0.0; m];
    let g_ok = m == 0 || tnlp.eval_g(&x, false, &mut g);
    let (g_nonfinite, g_nonfinite_count) = scan_nonfinite(&g, con_names, 'c', opts.max_list, g_ok);

    // Jacobian: structure then values.
    let mut irow = vec![0i32; nnz];
    let mut jcol = vec![0i32; nnz];
    let mut jval = vec![0.0; nnz];
    let mut jac_ok = nnz == 0;
    if nnz > 0 {
        jac_ok = tnlp.eval_jac_g(
            Some(&x),
            false,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        ) && tnlp.eval_jac_g(
            Some(&x),
            false,
            SparsityRequest::Values { values: &mut jval },
        );
    }
    let mut jac_nonfinite = Vec::new();
    let mut jac_nonfinite_count = 0usize;
    if jac_ok {
        for k in 0..nnz {
            if !jval[k].is_finite() {
                jac_nonfinite_count += 1;
                if jac_nonfinite.len() < opts.max_list {
                    let row = (irow[k] as usize).wrapping_sub(off);
                    let col = (jcol[k] as usize).wrapping_sub(off);
                    jac_nonfinite.push(NonFiniteEntry {
                        row,
                        col,
                        row_name: name_at(con_names, row, 'c'),
                        col_name: name_at(var_names, col, 'x'),
                        value: jval[k],
                    });
                }
            }
        }
    } else if nnz > 0 {
        jac_nonfinite_count = usize::MAX; // "evaluation itself failed"
    }

    // Hessian of the Lagrangian at (x0, lambda=0, obj_factor=1) — catches
    // second-derivative domain errors. Optional: quasi-Newton TNLPs decline.
    let hess_nonfinite_count = if nnz_h > 0 {
        let mut hrow = vec![0i32; nnz_h];
        let mut hcol = vec![0i32; nnz_h];
        let mut hval = vec![0.0; nnz_h];
        let lambda0 = vec![0.0; m];
        let ok = tnlp.eval_h(
            None,
            false,
            1.0,
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut hrow,
                jcol: &mut hcol,
            },
        ) && tnlp.eval_h(
            Some(&x),
            false,
            1.0,
            Some(&lambda0),
            true,
            SparsityRequest::Values { values: &mut hval },
        );
        if ok {
            Some(hval.iter().filter(|v| !v.is_finite()).count())
        } else {
            None
        }
    } else {
        None
    };

    // --- x0 vs bounds ---
    let mut bound_violations: Vec<RowReport> = Vec::new();
    let mut n_bound_violations = 0usize;
    let mut max_bound_violation = 0.0_f64;
    let mut n_on_bounds = 0usize;
    for j in 0..n {
        let viol = box_violation(x[j], x_l[j], x_u[j]);
        if viol > opts.feas_tol {
            n_bound_violations += 1;
            max_bound_violation = max_bound_violation.max(viol);
            push_worst(
                &mut bound_violations,
                RowReport {
                    index: j,
                    name: name_at(var_names, j, 'x'),
                    value: x[j],
                    lo: x_l[j],
                    hi: x_u[j],
                    violation: viol,
                },
                opts.max_list,
            );
        }
        if x[j].is_finite() {
            let at_lo =
                lower_bound_present(x_l[j]) && (x[j] - x_l[j]).abs() <= 1e-8 * (1.0 + x_l[j].abs());
            let at_hi =
                upper_bound_present(x_u[j]) && (x_u[j] - x[j]).abs() <= 1e-8 * (1.0 + x_u[j].abs());
            if at_lo || at_hi {
                n_on_bounds += 1;
            }
        }
    }

    // --- interior-clamp preview (DefaultIterateInitializer::push_to_interior) ---
    let mut clamp_moves: Vec<ClampMove> = Vec::new();
    let mut n_clamp_moved = 0usize;
    let mut max_clamp_move = 0.0_f64;
    for j in 0..n {
        if !x[j].is_finite() {
            continue;
        }
        let to = clamp_to_interior(x[j], x_l[j], x_u[j], opts.bound_push, opts.bound_frac);
        let d = (to - x[j]).abs();
        if d > 0.0 {
            n_clamp_moved += 1;
            max_clamp_move = max_clamp_move.max(d);
            if clamp_moves.len() < opts.max_list
                || clamp_moves.last().map(|w| d > w.distance).unwrap_or(false)
            {
                clamp_moves.push(ClampMove {
                    index: j,
                    name: name_at(var_names, j, 'x'),
                    from: x[j],
                    to,
                    distance: d,
                });
                clamp_moves.sort_by(|a, b| {
                    b.distance
                        .partial_cmp(&a.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                clamp_moves.truncate(opts.max_list);
            }
        }
    }

    // --- initial constraint violation ---
    let mut con_violations: Vec<RowReport> = Vec::new();
    let mut n_con_violations = 0usize;
    let mut max_con_violation = 0.0_f64;
    if g_ok {
        for i in 0..m {
            let viol = box_violation(g[i], g_l[i], g_u[i]);
            if viol > opts.feas_tol {
                n_con_violations += 1;
                if viol.is_finite() {
                    max_con_violation = max_con_violation.max(viol);
                }
                push_worst(
                    &mut con_violations,
                    RowReport {
                        index: i,
                        name: name_at(con_names, i, 'c'),
                        value: g[i],
                        lo: g_l[i],
                        hi: g_u[i],
                        violation: viol,
                    },
                    opts.max_list,
                );
            }
        }
    }

    // --- derivative scale spread ---
    let grad_spread = scale_spread(grad_f.iter().copied());
    let jac_spread = scale_spread(jval.iter().copied());

    // --- what gradient-based scaling will make of this x0 ---
    // Row maxima seeded the way upstream seeds them, so a row with no
    // Jacobian entry at all and a row whose entries are all zero reach
    // `gradient_row_scale` identically — which is how the solver sees them.
    let mut jac_row_max = vec![Number::MIN_POSITIVE; m];
    // The solver samples at the point `TNLPAdapter::GetStartingPoint`
    // hands over, which pins every fixed variable (`x_l == x_u`) to its
    // value; and it takes the objective's ∞-norm over the *non-fixed*
    // variables only, because fixed ones are not in the algorithm's x.
    // Both matter: on `flosp2hm` an unpinned x0 moved ‖∇f‖∞ by five orders
    // of magnitude and decided whether the objective got scaled at all
    // (see `scale_gradient_based`). So the preview lifts first, and
    // re-evaluates when the lift actually moved something.
    let fixed: Vec<bool> = (0..n)
        .map(|j| lower_bound_present(x_l[j]) && upper_bound_present(x_u[j]) && x_l[j] == x_u[j])
        .collect();
    let mut lifted = x.clone();
    let mut moved = false;
    for j in 0..n {
        if fixed[j] && lifted[j] != x_l[j] {
            lifted[j] = x_l[j];
            moved = true;
        }
    }
    let (scale_grad, scale_jval) = if moved {
        let mut gf = vec![0.0; n];
        let mut jv = vec![0.0; nnz];
        let gok = tnlp.eval_grad_f(&lifted, true, &mut gf);
        let jok = nnz == 0
            || tnlp.eval_jac_g(
                Some(&lifted),
                true,
                SparsityRequest::Values { values: &mut jv },
            );
        (
            if gok { gf } else { grad_f.clone() },
            if jok { jv } else { jval.clone() },
        )
    } else {
        (grad_f.clone(), jval.clone())
    };
    let max_grad_f = if grad_ok {
        (0..n)
            .filter(|&j| !fixed[j])
            .fold(0.0_f64, |acc, j| acc.max(scale_grad[j].abs()))
    } else {
        0.0
    };
    if jac_ok {
        for k in 0..nnz {
            let row = (irow[k] as usize).wrapping_sub(off);
            if row < m {
                let v = scale_jval[k].abs();
                if v > jac_row_max[row] {
                    jac_row_max[row] = v;
                }
            }
        }
    }
    let scaling = scaling_preview(
        &jac_row_max,
        &g_l,
        &g_u,
        max_grad_f,
        quad_coefs,
        con_names,
        opts,
    );

    // --- warnings + verdict ---
    let mut warnings = Vec::new();
    let eval_failed = !grad_ok || !g_ok || (!jac_ok && nnz > 0) || objective.is_none();
    let nonfinite_total = grad_nonfinite_count.min(usize::MAX - 1)
        + g_nonfinite_count.min(usize::MAX - 1)
        + if jac_nonfinite_count == usize::MAX {
            0
        } else {
            jac_nonfinite_count
        }
        + hess_nonfinite_count.unwrap_or(0)
        + usize::from(!obj_finite && objective.is_some());
    let fatal = eval_failed || nonfinite_total > 0;
    if eval_failed {
        warnings.push(
            "an evaluation callback failed outright at the starting point; \
             the solver cannot start from this x0"
                .to_string(),
        );
    }
    if nonfinite_total > 0 {
        warnings.push(format!(
            "{nonfinite_total} non-finite value(s) at the starting point; a solve \
             would abort with Invalid_Number_Detected. The interior clamp only \
             repairs bound violations, not domain errors — move x0 into the \
             domain or add bounds that keep it there"
        ));
    }
    if x0_all_zero {
        warnings.push(
            "the starting point is all zeros: the model supplies no initial \
             guess (or an explicitly zero one)"
                .to_string(),
        );
    }
    if n_bound_violations > 0 {
        warnings.push(format!(
            "x0 violates {n_bound_violations} variable bound(s) (max {max_bound_violation:.3e}); \
             the initializer will clamp them inside"
        ));
    }
    if n_on_bounds > 0 {
        warnings.push(format!(
            "{n_on_bounds} component(s) of x0 sit exactly on a bound and will be \
             pushed into the interior (bound_push={:.1e}); if x0 is a previous \
             solution, use the warm-start recipe (warm_start_init_point=yes with \
             tightened warm_start_bound_push/_frac)",
            opts.bound_push
        ));
    }
    if max_con_violation > 1e4 {
        warnings.push(format!(
            "very large initial infeasibility (max constraint violation \
             {max_con_violation:.3e}); consider a better starting point or \
             least_square_init_primal=yes"
        ));
    }
    // A quadratic row the sample cannot see is not a curiosity when its
    // coefficients disagree by orders of magnitude: the row keeps factor
    // 1.0, its slack `s = −g(x)` inherits the right-hand side's scale, and
    // the `−s/λ` KKT diagonal inherits it too (gh #703).
    if scaling.n_quad_zero_jac > 0 && scaling.max_quad_mismatch > 1e2 {
        warnings.push(format!(
            "{} quadratic row(s) have an identically-zero Jacobian at x0, so \
             gradient-based scaling leaves them at factor 1.0 — and their \
             right-hand sides run up to {:.3e}x their curvature ‖Q‖_∞. The \
             automatic scaler samples derivatives at x0 and cannot see this; \
             set per-row factors with nlp_scaling_method=user-scaling, or \
             rewrite the rows about a point where their gradient is nonzero",
            scaling.n_quad_zero_jac, scaling.max_quad_mismatch
        ));
    }
    for (label, s) in [("gradient", &grad_spread), ("Jacobian", &jac_spread)] {
        if s.ratio > 1e8 || s.max_abs > 1e8 {
            warnings.push(format!(
                "{label} magnitudes at x0 span a large range (max {:.3e}, min \
                 nonzero {:.3e}); see the scaling reference page",
                s.max_abs, s.min_abs_nonzero
            ));
        }
    }

    let verdict = if fatal {
        "FATAL"
    } else if warnings.is_empty() {
        "CLEAN"
    } else {
        "WARNINGS"
    };

    Ok(PreflightOutcome {
        n_vars: n,
        n_cons: m,
        nl_sha256,
        source,
        x0_source,
        x0_all_zero,
        objective,
        grad_nonfinite,
        grad_nonfinite_count,
        g_nonfinite,
        g_nonfinite_count,
        jac_nonfinite,
        jac_nonfinite_count: if jac_nonfinite_count == usize::MAX {
            0
        } else {
            jac_nonfinite_count
        },
        hess_nonfinite_count,
        bound_violations,
        n_bound_violations,
        max_bound_violation,
        n_on_bounds,
        clamp_moves,
        n_clamp_moved,
        max_clamp_move,
        con_violations,
        n_con_violations,
        max_con_violation,
        grad_spread,
        jac_spread,
        scaling,
        warnings,
        fatal,
        verdict,
    })
}

/// The per-component interior clamp from
/// `DefaultIterateInitializer::push_to_interior` (see
/// `crates/pounce-algorithm/src/init/default.rs` and
/// `docs/src/initialization.md`).
pub fn clamp_to_interior(
    x: Number,
    lo: Number,
    hi: Number,
    bound_push: Number,
    bound_frac: Number,
) -> Number {
    match (lower_bound_present(lo), upper_bound_present(hi)) {
        (true, true) => {
            let span = hi - lo;
            let p_l = (bound_push * lo.abs().max(1.0)).min(bound_frac * span);
            let p_u = (bound_push * hi.abs().max(1.0)).min(bound_frac * span);
            x.max(lo + p_l).min(hi - p_u)
        }
        (true, false) => x.max(lo + bound_push * lo.abs().max(1.0)),
        (false, true) => x.min(hi - bound_push * hi.abs().max(1.0)),
        (false, false) => x,
    }
}

fn scan_nonfinite(
    values: &[Number],
    names: &[String],
    kind: char,
    cap: usize,
    eval_ok: bool,
) -> (Vec<NonFinite>, usize) {
    if !eval_ok {
        return (Vec::new(), 0);
    }
    let mut out = Vec::new();
    let mut count = 0usize;
    for (i, v) in values.iter().enumerate() {
        if !v.is_finite() {
            count += 1;
            if out.len() < cap {
                out.push(NonFinite {
                    index: i,
                    name: name_at(names, i, kind),
                    value: *v,
                });
            }
        }
    }
    (out, count)
}

/// Keep the `cap` worst entries by violation, descending.
fn push_worst(list: &mut Vec<RowReport>, r: RowReport, cap: usize) {
    list.push(r);
    list.sort_by(|a, b| {
        b.violation
            .partial_cmp(&a.violation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    list.truncate(cap);
}

fn scale_spread(values: impl Iterator<Item = Number>) -> ScaleSpread {
    let mut max_abs = 0.0_f64;
    let mut min_abs = Number::INFINITY;
    for v in values {
        let a = v.abs();
        if a.is_finite() && a > 0.0 {
            max_abs = max_abs.max(a);
            min_abs = min_abs.min(a);
        }
    }
    if max_abs == 0.0 {
        ScaleSpread::default()
    } else {
        ScaleSpread {
            max_abs,
            min_abs_nonzero: min_abs,
            ratio: max_abs / min_abs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tnlp::{IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution};
    use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};

    /// min 1/x0 + x1  s.t. x0 + x1 = 1, with x0 starting AT zero — the
    /// canonical Invalid_Number_Detected trap.
    struct DomainTrap {
        x0: Vec<Number>,
    }

    impl TNLP for DomainTrap {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 2,
                m: 1,
                nnz_jac_g: 2,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[0.0, NLP_LOWER_BOUND_INF]);
            b.x_u
                .copy_from_slice(&[NLP_UPPER_BOUND_INF, NLP_UPPER_BOUND_INF]);
            b.g_l[0] = 1.0;
            b.g_u[0] = 1.0;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            if sp.init_x {
                sp.x.copy_from_slice(&self.x0);
            }
            true
        }
        fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
            Some(1.0 / x[0] + x[1])
        }
        fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
            grad_f[0] = -1.0 / (x[0] * x[0]);
            grad_f[1] = 1.0;
            true
        }
        fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g[0] = x[0] + x[1];
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow.copy_from_slice(&[0, 0]);
                    jcol.copy_from_slice(&[0, 1]);
                }
                SparsityRequest::Values { values } => {
                    values.copy_from_slice(&[1.0, 1.0]);
                }
            }
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {}
    }

    fn check(x0: Vec<Number>) -> PreflightOutcome {
        let mut t = DomainTrap { x0 };
        check_tnlp(
            &mut t,
            &[],
            &[],
            None,
            "test".into(),
            &PreflightOptions::default(),
        )
        .expect("check")
    }

    #[test]
    fn nan_at_x0_is_fatal() {
        // x0[0] = 0 → f = 1/0 = inf, grad[0] = -inf.
        let o = check(vec![0.0, 0.0]);
        assert!(o.fatal);
        assert_eq!(o.verdict, "FATAL");
        assert!(o.grad_nonfinite_count >= 1);
        assert!(o.x0_all_zero);
    }

    #[test]
    fn clean_interior_point_passes() {
        let o = check(vec![0.5, 0.5]);
        assert!(!o.fatal);
        assert_eq!(o.n_bound_violations, 0);
        // x0 + x1 = 1 exactly: feasible.
        assert_eq!(o.n_con_violations, 0);
        assert_eq!(o.verdict, "CLEAN");
        assert!((o.objective.unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn on_bound_component_is_flagged_and_clamped() {
        // x0[0] = 1e-12 is (numerically) on its lower bound 0; the clamp
        // moves it to ~bound_push = 1e-2 (span is infinite: one-sided).
        let o = check(vec![1e-12, 1.0]);
        assert!(o.n_on_bounds >= 1);
        assert!(o.n_clamp_moved >= 1);
        assert!((o.max_clamp_move - 1e-2).abs() < 1e-9);
        assert!(
            o.warnings
                .iter()
                .any(|w| w.contains("warm_start_bound_push"))
        );
    }

    #[test]
    fn bound_violation_reported() {
        let o = check(vec![-3.0, 1.0]);
        assert_eq!(o.n_bound_violations, 1);
        assert!((o.max_bound_violation - 3.0).abs() < 1e-12);
        // clamp brings it inside: from -3 to lo + push
        assert!(o.n_clamp_moved >= 1);
    }

    #[test]
    fn infeasible_start_is_not_fatal() {
        let o = check(vec![5.0, 5.0]);
        assert!(!o.fatal);
        assert_eq!(o.n_con_violations, 1);
        assert!((o.max_con_violation - 9.0).abs() < 1e-12);
    }

    #[test]
    fn clamp_formula_matches_default_initializer() {
        // Two-sided [1, 5], bound_push=bound_frac=1e-2:
        // p_l = min(1e-2*1, 1e-2*4) = 0.01 → 1.0 clamps to 1.01.
        assert!((clamp_to_interior(1.0, 1.0, 5.0, 1e-2, 1e-2) - 1.01).abs() < 1e-15);
        // Interior stays put.
        assert_eq!(clamp_to_interior(3.0, 1.0, 5.0, 1e-2, 1e-2), 3.0);
        // Free variable untouched.
        assert_eq!(
            clamp_to_interior(-7.0, NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF, 1e-2, 1e-2),
            -7.0
        );
        // Upper one-sided: hi=100 → push = 1e-2*100 = 1 → 100 → 99.
        assert!(
            (clamp_to_interior(100.0, NLP_LOWER_BOUND_INF, 100.0, 1e-2, 1e-2) - 99.0).abs() < 1e-12
        );
    }

    #[test]
    fn scale_spread_ignores_zeros_and_nonfinite() {
        let s = scale_spread(vec![0.0, 1e-6, 1e3, Number::NAN].into_iter());
        assert!((s.max_abs - 1e3).abs() < 1e-9);
        assert!((s.min_abs_nonzero - 1e-6).abs() < 1e-18);
        assert!((s.ratio - 1e9).abs() / 1e9 < 1e-9);
    }

    // ---------------------------------------------------------------
    // gh #703 — the scaling preview
    // ---------------------------------------------------------------

    /// `min 10·x  s.t.  1000·x ≥ 4e6`, started at `x = 5000`.
    ///
    /// The same fixture as `orig_ipopt_nlp`'s
    /// `gradient_based_scaling_scales_d_l_and_d_u` / `..._obj_target_gradient`
    /// pair, restated here so the preview is pinned against a case whose
    /// factors that module already asserts the *solver* produces: row max
    /// 1000 against the cutoff 100 gives `d_scale = 0.1`, and
    /// `‖∇f‖ = 10 < 100` leaves the objective at 1.0.
    struct OneIneqLargeOffset;

    impl TNLP for OneIneqLargeOffset {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 1,
                m: 1,
                nnz_jac_g: 1,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = NLP_LOWER_BOUND_INF;
            b.x_u[0] = NLP_UPPER_BOUND_INF;
            b.g_l[0] = 4.0e6;
            b.g_u[0] = NLP_UPPER_BOUND_INF;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 5000.0;
            true
        }
        fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
            Some(10.0 * x[0])
        }
        fn eval_grad_f(&mut self, _: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = 10.0;
            true
        }
        fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = 1000.0 * x[0];
            true
        }
        fn eval_jac_g(&mut self, _: Option<&[Number]>, _: bool, req: SparsityRequest<'_>) -> bool {
            match req {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0;
                    jcol[0] = 0;
                }
                SparsityRequest::Values { values } => values[0] = 1000.0,
            }
            true
        }
        fn eval_h(
            &mut self,
            _: Option<&[Number]>,
            _: bool,
            _: Number,
            _: Option<&[Number]>,
            _: bool,
            _: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
    }

    #[test]
    fn scaling_preview_reproduces_the_solvers_factors() {
        let mut t = OneIneqLargeOffset;
        let o = check_tnlp(
            &mut t,
            &[],
            &[],
            None,
            "test".to_string(),
            &PreflightOptions::default(),
        )
        .unwrap();
        let s = &o.scaling;
        assert_eq!(s.max_gradient, 100.0);
        // ‖∇f‖ = 10 is below the cutoff, so the objective is unscaled.
        assert_eq!(s.max_grad_f, 10.0);
        assert_eq!(s.obj_scale, 1.0);
        // The single row is an inequality; row max 1000 > 100, so the
        // block fires and the factor is 100/1000.
        assert_eq!(s.c.n_rows, 0);
        assert_eq!(s.d.n_rows, 1);
        assert!(s.d.fires);
        assert_eq!(s.d.n_scaled, 1);
        assert!((s.d.min_scale - 0.1).abs() < 1e-15);
        assert_eq!(s.d.n_zero_jac, 0);
        // No `.nl`, so no coefficient census.
        assert_eq!(s.n_quad_rows, 0);
    }

    #[test]
    fn a_row_below_the_cutoff_leaves_the_whole_block_unscaled() {
        // `gradient_scaling_fires` is a per-block gate, not per-row: the
        // DomainTrap equality's Jacobian is [1, 1], far below 100.
        let mut t = DomainTrap { x0: vec![1.0, 0.0] };
        let o = check_tnlp(
            &mut t,
            &[],
            &[],
            None,
            "test".to_string(),
            &PreflightOptions::default(),
        )
        .unwrap();
        assert_eq!(o.scaling.c.n_rows, 1);
        assert!(!o.scaling.c.fires);
        assert_eq!(o.scaling.c.n_scaled, 0);
        assert_eq!(o.scaling.c.min_scale, 1.0);
    }

    /// The objective factor the preview predicts must be the one
    /// `OrigIpoptNlp` actually installs. Both call
    /// [`gradient_obj_scale`], so the arithmetic cannot drift; what this
    /// pins is everything *around* it — which point is sampled, and which
    /// variables the ∞-norm is taken over.
    #[test]
    fn preview_objective_factor_matches_the_installed_one() {
        use crate::orig_ipopt_nlp::{NoScaling, OrigIpoptNlp, ScalingMethod};
        use crate::tnlp_adapter::TNLPAdapter;
        use std::cell::RefCell;
        use std::rc::Rc;

        for tnlp in [
            Rc::new(RefCell::new(BigObjGradient)) as Rc<RefCell<dyn TNLP>>,
            Rc::new(RefCell::new(OneIneqLargeOffset)) as Rc<RefCell<dyn TNLP>>,
        ] {
            let preview = {
                let mut t = tnlp.borrow_mut();
                check_tnlp(
                    &mut *t,
                    &[],
                    &[],
                    None,
                    "test".to_string(),
                    &PreflightOptions::default(),
                )
                .unwrap()
                .scaling
                .obj_scale
            };
            let adapter = Rc::new(RefCell::new(TNLPAdapter::new(Rc::clone(&tnlp)).unwrap()));
            let mut nlp = OrigIpoptNlp::new(adapter, Rc::new(NoScaling)).unwrap();
            nlp.determine_scaling_from_starting_point(
                ScalingMethod::GradientBased,
                NLP_SCALING_MAX_GRADIENT,
                NLP_SCALING_MIN_VALUE,
                0.0,
                0.0,
            );
            assert_eq!(
                preview,
                nlp.obj_scale_factor(),
                "preview and installed objective factor disagree"
            );
        }
    }

    /// `min x₀ + 1e6·x₁` where `x₁` is **fixed** at 3.
    ///
    /// The big gradient component belongs to the fixed variable, which is
    /// not in the algorithm's `x`, so the scaler's ∞-norm is 1 and the
    /// objective comes out unscaled. A preview that took the norm over all
    /// `n` components would read 1e6 and predict a factor of 1e-4 — which
    /// is what this fixture is here to catch.
    struct BigObjGradient;

    impl TNLP for BigObjGradient {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 2,
                m: 1,
                nnz_jac_g: 2,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = NLP_LOWER_BOUND_INF;
            b.x_u[0] = NLP_UPPER_BOUND_INF;
            b.x_l[1] = 3.0;
            b.x_u[1] = 3.0;
            b.g_l[0] = NLP_LOWER_BOUND_INF;
            b.g_u[0] = 10.0;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 0.0;
            sp.x[1] = 0.0;
            true
        }
        fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
            Some(x[0] + 1.0e6 * x[1])
        }
        fn eval_grad_f(&mut self, _: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = 1.0;
            g[1] = 1.0e6;
            true
        }
        fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = x[0] + x[1];
            true
        }
        fn eval_jac_g(&mut self, _: Option<&[Number]>, _: bool, req: SparsityRequest<'_>) -> bool {
            match req {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0;
                    jcol[0] = 0;
                    irow[1] = 0;
                    jcol[1] = 1;
                }
                SparsityRequest::Values { values } => {
                    values[0] = 1.0;
                    values[1] = 1.0;
                }
            }
            true
        }
        fn eval_h(
            &mut self,
            _: Option<&[Number]>,
            _: bool,
            _: Number,
            _: Option<&[Number]>,
            _: bool,
            _: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
    }
}
