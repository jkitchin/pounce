//! `SqpAlgorithm` — active-set SQP outer loop. Consumes an
//! `SqpProblemSpec` for evaluation; delegates the QP subproblem
//! solve to `pounce_qp::ParametricActiveSetSolver`.
//!
//! Outer loop (Nocedal-Wright §18 standard SQP):
//! 1. Evaluate `f, ∇f, c, ∇c, ∇²L` at `x_k`.
//! 2. Build the QP via `SqpQpData::build`.
//! 3. Solve the QP via `pounce-qp` (warm-started by the previous
//!    `WorkingSet` when available).
//! 4. KKT-error check on `x_k` (before stepping) — if all
//!    component tolerances are met, declare optimal.
//! 5. Globalization step acceptance via either the Fletcher-
//!    Leyffer 2002 filter (`SqpGlobalization::Filter`, default)
//!    or the Han-Powell l1-merit (`SqpGlobalization::L1Elastic`),
//!    both backtracking on α.
//! 6. Take `α·p`; promote `(x_k + α p, λ_g, λ_x)` to the next
//!    iterate and carry the QP's `WorkingSet` for the next solve.

use crate::sqp::bfgs::DampedBfgs;
use crate::sqp::filter::{SqpFilter, filter_line_search};
use crate::sqp::iterates::SqpIterates;
use crate::sqp::lbfgs::LBfgs;
use crate::sqp::line_search::l1_merit_line_search;
use crate::sqp::options::{SqpGlobalization, SqpHessianSource, SqpOptions};
use crate::sqp::problem::SqpProblemSpec;
use crate::sqp::qp_assembly::{SqpQpData, Triplet};
use crate::sqp::result::{SqpError, SqpResult, SqpStatus};
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF, Number};
use pounce_linalg::triplet::GenTMatrix;
use pounce_qp::{
    BoundStatus, ConsStatus, HessianInertia, ParametricActiveSetSolver, QpOptions, QpSolver,
    QpStatus, WorkingSet,
};

/// Header for the `sqp_print_level >= 1` iteration table.
///
/// Column widths for `iter` / `objective` / `inf_pr` / `inf_du` are the
/// interior-point table's exactly (`crate::output::orig`), so the two arms'
/// logs line up when a reader puts them side by side on one model. The
/// remaining columns are the ones this arm actually has: an active-set SQP
/// has no barrier parameter and no dual step length, so `lg(mu)`, `lg(rg)`
/// and `alpha_du` would be permanently blank, and it does have a working-set
/// size that the IPM does not.
const SQP_ITER_HEADER: &str = "iter      objective   inf_pr   inf_du    ||p||    alpha     ws";

/// What the step into the current iterate cost — the trailing columns of a
/// row. `None` on row 0, where no step has been taken yet.
struct SqpStepRecord {
    /// `‖p‖_inf` of the QP step accepted into this iterate.
    p_inf: Number,
    /// Line-search step length.
    alpha: Number,
    /// Whether the accepted step was a second-order correction. Rendered as
    /// a trailing `c`, the same position Ipopt puts its `alpha_char` in.
    soc: bool,
    /// Active-set changes (adds + drops) in the QP that produced the step.
    ws_changes: u32,
}

/// One row of the `sqp_print_level >= 1` table.
///
/// Row `k` reports the residuals **at** `x_k`; the step columns describe the
/// step that arrived there, so they are blank on row 0. That is Ipopt's
/// convention and `crate::output::orig::OrigIterationOutput::format_row`'s.
fn format_sqp_row(
    iter: u32,
    obj: Number,
    inf_pr: Number,
    inf_du: Number,
    step: Option<&SqpStepRecord>,
) -> String {
    use crate::output::orig::format_e;
    match step {
        Some(st) => format!(
            "{:>4} {:>14} {:>8} {:>8} {:>8} {:>8}{} {:>5}",
            iter,
            format_e(obj, 7),
            format_e(inf_pr, 2),
            format_e(inf_du, 2),
            format_e(st.p_inf, 2),
            format_e(st.alpha, 2),
            if st.soc { 'c' } else { ' ' },
            st.ws_changes,
        ),
        None => format!(
            "{:>4} {:>14} {:>8} {:>8} {:>8} {:>8}  {:>5}",
            iter,
            format_e(obj, 7),
            format_e(inf_pr, 2),
            format_e(inf_du, 2),
            "-",
            "-",
            "-",
        ),
    }
}

/// What the per-iteration hook sees: the iterate `x_k` and the residuals
/// measured **at** it (the same numbers the `sqp_print_level >= 1` row
/// prints), plus the step that arrived there.
pub struct SqpIterationInfo<'a> {
    pub iter: u32,
    pub obj: Number,
    pub inf_pr: Number,
    pub inf_du: Number,
    /// `‖p‖_inf` of the accepted step into this iterate (0 on iteration 0).
    pub p_inf: Number,
    /// Line-search step length of that step (0 on iteration 0).
    pub alpha: Number,
    pub x: &'a [Number],
    /// `[λ_c; λ_d]` in algorithm space.
    pub lambda_g: &'a [Number],
    /// Signed bound multiplier `z_L - z_U` per variable.
    pub lambda_x: &'a [Number],
    /// `[c(x); d(x)]` in algorithm space.
    pub c_vals: &'a [Number],
    /// `∇f + Jᵀλ_g − λ_x` (the stationarity residual `check_kkt` measures)
    /// in algorithm space, from this iterate's `∇f` and `J`.
    pub grad_lag_x: &'a [Number],
}

/// Per-iteration hook. Returning `false` stops the solve.
pub type SqpIterationHook = Box<dyn FnMut(&SqpIterationInfo<'_>) -> bool>;

/// SQP-side algorithm driver.
pub struct SqpAlgorithm {
    iteration_hook: Option<SqpIterationHook>,
    qp_solver: ParametricActiveSetSolver,
    qp_opts: QpOptions,
    opts: SqpOptions,
    /// `opts.hessian` resolved for the current solve's `n` (see
    /// [`SqpHessianSource::resolve`]); set at the top of every solve.
    hessian: SqpHessianSource,
    iterates: Option<SqpIterates>,
    /// Filter for Fletcher-Leyffer globalization; reset at the
    /// top of each `optimize` call. Unused when
    /// `opts.globalization = L1Elastic`.
    filter: SqpFilter,
    /// Whether this solve's cold step QPs are seeded by the convex interior
    /// point; set per solve from `opts.qp_ipm_cold_start_min_dim`.
    ipm_cold_start: bool,
    /// The last interior-point seed's iteration count, which sets a warm
    /// step QP's budget (see [`solve_subproblem_qp`]).
    ipm_seed: IpmSeed,
}

/// State the IPM-seeded subproblem solves share across one SQP solve.
#[derive(Default)]
pub(crate) struct IpmSeed {
    /// Interior-point iterations of the most recent successful seed.
    pub(crate) last_iters: std::cell::Cell<Option<usize>>,
}

impl SqpAlgorithm {
    pub fn new(qp_solver: ParametricActiveSetSolver, opts: SqpOptions) -> Self {
        Self {
            qp_solver,
            qp_opts: QpOptions::sqp_subproblem(),
            hessian: opts.hessian,
            opts,
            iterates: None,
            filter: SqpFilter::new(),
            iteration_hook: None,
            ipm_cold_start: false,
            ipm_seed: IpmSeed::default(),
        }
    }

    /// Install a hook fired once per iteration, at the iterate the loop is
    /// about to test for convergence (iteration 0 and the final iterate
    /// included, as on the interior-point arm). Returning `false` ends the
    /// solve with [`SqpStatus::UserRequestedStop`].
    pub fn set_iteration_hook(&mut self, hook: SqpIterationHook) {
        self.iteration_hook = Some(hook);
    }

    /// Override the per-call QP-solver options. Defaults are
    /// `pounce_qp::QpOptions::sqp_subproblem()` — `QpOptions::default()`
    /// with second-order certification off, for the reason given there
    /// (which include the
    /// `use_schur_updates = false` and `anti_cycling = Expand`
    /// from Phase 5a.2). Callers can pin tighter tolerances or
    /// flip `use_schur_updates = true` for warm-started workloads.
    pub fn with_qp_options(mut self, qp_opts: QpOptions) -> Self {
        self.qp_opts = qp_opts;
        self
    }

    pub fn options(&self) -> &SqpOptions {
        &self.opts
    }

    pub fn iterates(&self) -> Option<&SqpIterates> {
        self.iterates.as_ref()
    }

    /// Run the SQP loop to convergence (or `max_iter`). Cold-starts
    /// the iterate from `nlp.x_init()` and an empty working set.
    pub fn optimize<N: SqpProblemSpec>(&mut self, nlp: &mut N) -> Result<SqpResult, SqpError> {
        self.optimize_with_warm_start(nlp, None)
    }

    /// Warm-start variant. `warm = Some(prev)` seeds the iterate
    /// from `prev.{x, lambda_g, lambda_x, working}` instead of the
    /// NLP's cold defaults. Dimensions are validated against the
    /// problem; any mismatch is fatal. The QP solver consumes
    /// `warm.working` (when present) via `solve_with_working_set`.
    ///
    /// `warm = None` is equivalent to [`Self::optimize`].
    ///
    /// Implements the §6 design-note warm-start contract: the
    /// tuple `(x, λ_g, λ_x, 𝒲)`. The Hessian carry-forward
    /// (damped-BFGS / L-BFGS state) is *not* part of the warm-start
    /// payload — each `optimize` call rebuilds its own Hessian
    /// approximation from scratch.
    pub fn optimize_with_warm_start<N: SqpProblemSpec>(
        &mut self,
        nlp: &mut N,
        warm: Option<SqpIterates>,
    ) -> Result<SqpResult, SqpError> {
        // The step-QP iteration budget scales with the problem unless the
        // caller set one (gh#1054). Restored afterwards, so a reused driver
        // re-derives it for the next problem's size.
        // Schur updates likewise follow the size, and the Hessian source,
        // when the caller left them to the driver
        // (`SqpOptions::step_qp_schur_updates`).
        let saved = (self.qp_opts.max_iter, self.qp_opts.use_schur_updates);
        self.qp_opts.max_iter = step_qp_iter_budget(saved.0, nlp.n(), nlp.m());
        if let Some(on) = self
            .opts
            .step_qp_schur_updates(nlp.n() + nlp.m(), self.opts.hessian)
        {
            self.qp_opts.use_schur_updates = on;
        }
        self.ipm_cold_start = self
            .opts
            .qp_ipm_cold_start_min_dim
            .is_some_and(|d| nlp.n() + nlp.m() >= d);
        self.ipm_seed.last_iters.set(None);
        let res = self.optimize_inner(nlp, warm);
        (self.qp_opts.max_iter, self.qp_opts.use_schur_updates) = saved;
        res
    }

    fn optimize_inner<N: SqpProblemSpec>(
        &mut self,
        nlp: &mut N,
        warm: Option<SqpIterates>,
    ) -> Result<SqpResult, SqpError> {
        let n = nlp.n();
        let m = nlp.m();
        let (xl, xu) = nlp.variable_bounds();
        let (bl_c, bu_c) = nlp.constraint_bounds();
        if xl.len() != n || xu.len() != n {
            return Err(SqpError::DimensionMismatch(format!(
                "variable_bounds length must be n = {n}"
            )));
        }
        if bl_c.len() != m || bu_c.len() != m {
            return Err(SqpError::DimensionMismatch(format!(
                "constraint_bounds length must be m = {m}"
            )));
        }

        let mut iter = match warm {
            Some(w) => {
                if w.x.len() != n {
                    return Err(SqpError::DimensionMismatch(format!(
                        "warm.x length {} must equal n = {n}",
                        w.x.len()
                    )));
                }
                if w.lambda_g.len() != m {
                    return Err(SqpError::DimensionMismatch(format!(
                        "warm.lambda_g length {} must equal m = {m}",
                        w.lambda_g.len()
                    )));
                }
                if w.lambda_x.len() != n {
                    return Err(SqpError::DimensionMismatch(format!(
                        "warm.lambda_x length {} must equal n = {n}",
                        w.lambda_x.len()
                    )));
                }
                if let Some(ws) = w.working.as_ref() {
                    ws.validate_dims(n, m).map_err(SqpError::QpFailure)?;
                }
                w
            }
            None => {
                let mut cold = SqpIterates::cold(n, m);
                let x_init = nlp.x_init();
                if x_init.len() != n {
                    return Err(SqpError::DimensionMismatch(format!(
                        "x_init length must be n = {n}"
                    )));
                }
                cold.x = x_init;
                cold
            }
        };

        let mut n_qp_solves: u32 = 0;
        // Inner active-set work: adds + drops summed over every step QP
        // solved in this call. This — not the outer iteration count — is
        // what a warm start is trying to reduce, and on a QP-shaped NLP
        // (one outer iteration by construction) it is the *only* thing
        // that moves. Second-order-correction QPs are not counted: they
        // are solved inside the line search, which does not surface its
        // subproblem stats.
        let mut n_qp_working_set_changes: u32 = 0;
        let mut final_stationarity = 0.0;
        let mut final_constr_viol = 0.0;
        // l1-merit penalty parameter ν, adapted across iterations
        // by `l1_merit_line_search`. Initialized from
        // `SqpOptions::l1_penalty`.
        let mut nu = self.opts.l1_penalty;
        // Trailing columns of the `sqp_print_level >= 1` row: what the step
        // into the CURRENT iterate cost. Written at the end of each
        // iteration, read at the top of the next, `None` on row 0.
        let mut prev_step: Option<SqpStepRecord> = None;
        // Reset filter state at the top of each optimize call.
        self.filter = SqpFilter::new();
        // Cache the most recent f(x) and c(x) so we don't
        // re-evaluate them after a successful line search (the
        // LS already computed them at the new iterate).
        let mut f_cached: Option<Number> = None;
        let mut c_cached: Option<Vec<Number>> = None;
        // Previous iterate's `(x, ∇f, ∇c)`, kept so the quasi-Newton
        // curvature pair can difference `∇L` at a single fixed multiplier
        // (see [`curvature_pair`]). Storing `∇L` directly — as the older
        // `DampedBfgs::update(x, ∇L)` form did — bakes in the multiplier
        // that was current at the time, which is precisely the bug.
        let mut prev_point: Option<(Vec<Number>, Vec<Number>, Triplet)> = None;

        // Damped-BFGS state, allocated only if needed. The
        // matrix is updated at the END of each iteration (after
        // we have x_new and the next ∇L), then queried at the
        // TOP of the next iteration to populate the QP Hessian.
        self.hessian = self.opts.hessian.resolve(n, self.opts.dense_hessian_max_n);
        // L-BFGS above the dense ceiling goes to the QP in product form,
        // lifted (gh#1054); below it, as the dense matrix it always was.
        let lift_lbfgs =
            self.hessian == SqpHessianSource::Lbfgs && n > self.opts.dense_hessian_max_n;
        let mut bfgs: Option<DampedBfgs> = if matches!(self.hessian, SqpHessianSource::DampedBfgs) {
            Some(DampedBfgs::new(n))
        } else {
            None
        };
        let mut lbfgs: Option<LBfgs> = if matches!(self.hessian, SqpHessianSource::Lbfgs) {
            Some(LBfgs::new(n, self.opts.lbfgs_max_history.max(1) as usize))
        } else {
            None
        };

        // Iteration-0 curvature probe (issue #358 tail).
        //
        // `DampedBfgs::update` sizes the identity seed from the first
        // `(s, y)` pair — but that pair only exists at iteration 1, and
        // iteration **0** already solves a QP against `B`. With `B = I`
        // on a problem where `‖∇²L‖ ≫ 1`, that first step overshoots the
        // Newton step by `~cond(∇²L)`; the filter (empty, and `θ` tiny at
        // a near-feasible start) accepts the objective-blowing step, the
        // iterate is flung to `‖x‖ ~ 1e3`, and the huge `(s, y)` pairs
        // that follow drive `B` so ill-conditioned that the QP subproblem
        // itself fails a few iterations later (`QpStepFailed`, surfacing
        // as `Search_Direction_Becomes_Too_Small`).
        //
        // Fix the scale *before* that first QP with one extra gradient
        // evaluation: step a short distance along the steepest-descent
        // direction, difference the gradients, and seed `B = γI` with the
        // resulting Rayleigh quotient `γ = sᵀy / sᵀs`. For a quadratic
        // this is exactly the curvature along the probe direction, and it
        // lies in `[λ_min(∇²L), λ_max(∇²L)]`.
        //
        // The probe differences the *objective* gradient, so it estimates
        // `∇²f` — which equals the Lagrangian Hessian `∇²L = ∇²f + Σλᵢ∇²cᵢ`
        // only when the constraint-curvature term vanishes, i.e. when every
        // constraint is linear (or there are none). That condition is
        // exactly the #358 family, and it is *not* cosmetic: on the Maratos
        // problem (`min 2(x₁²+x₂²−1) − x₁ s.t. x₁²+x₂²=1`) `∇²f = 4I` while
        // `∇²L ≈ I` at the solution, so seeding the objective curvature
        // would over-scale `B` fourfold and cost that solve its convergence.
        //
        // Detect linearity directly rather than trusting a declaration:
        // compare the constraint Jacobian at the probe point with the one
        // at `x`. Identical ⇒ `∇c` is constant ⇒ constraints are linear ⇒
        // the objective Hessian *is* the Lagrangian Hessian and the probe
        // is exact. Otherwise leave the identity seed alone and let the
        // rank-2 updates (which see the true `∇L`) do the work.
        if bfgs.is_some() || lbfgs.is_some() {
            let g0 = nlp.eval_grad_f(&iter.x);
            let g_norm = g0.iter().map(|v| v * v).sum::<Number>().sqrt();
            if g_norm.is_finite() && g_norm > 0.0 {
                // Absolute probe length, scaled by the iterate so the step
                // is meaningful in the problem's own units but always tiny
                // relative to it. `1e-7` keeps the gradient difference well
                // above f64 roundoff without leaving the local model.
                let x_scale = iter.x.iter().map(|v| v.abs()).fold(1.0, f64::max);
                let eps = 1e-7 * x_scale;
                let step: Vec<Number> = g0.iter().map(|gi| -eps * gi / g_norm).collect();
                let x_probe: Vec<Number> =
                    iter.x.iter().zip(step.iter()).map(|(a, d)| a + d).collect();
                // Constant-Jacobian (linear-constraint) check, per above.
                let linear_constraints = m == 0 || {
                    let j0 = nlp.eval_jac_c(&iter.x);
                    let j1 = nlp.eval_jac_c(&x_probe);
                    j0.vals.len() == j1.vals.len()
                        && j0.vals.iter().zip(j1.vals.iter()).all(|(a, c)| {
                            // Relative comparison: a linear constraint
                            // reproduces its Jacobian bit-for-bit, so this
                            // only tolerates evaluation noise.
                            let scale = a.abs().max(c.abs()).max(1.0);
                            (a - c).abs() <= 1e-12 * scale
                        })
                };
                if linear_constraints {
                    let g1 = nlp.eval_grad_f(&x_probe);
                    let s_y: Number = step
                        .iter()
                        .zip(g1.iter().zip(g0.iter()))
                        .map(|(si, (a, bg))| si * (a - bg))
                        .sum();
                    let s_s: Number = step.iter().map(|v| v * v).sum();
                    if s_s > 0.0 && s_y.is_finite() {
                        // A non-positive quotient means the probe direction
                        // has non-positive curvature (nonconvex or
                        // numerically flat); `seed_scale` ignores it, leaving
                        // the identity seed rather than a meaningless or
                        // negative scale.
                        // L-BFGS takes the same seed (gh#1054): it used to
                        // start from `B = I` with no probe at all, which is
                        // the #358 overshoot this block exists to prevent.
                        if let Some(b) = bfgs.as_mut() {
                            b.seed_scale(s_y / s_s);
                        }
                        if let Some(b) = lbfgs.as_mut() {
                            b.seed_scale(s_y / s_s);
                        }
                    }
                }
            }
        }

        // Bounded so a curvature escape that keeps returning to the same
        // neighbourhood cannot spin: each one costs a fresh descent, and a
        // handful is far more than any of the measured models needs (one).
        const MAX_SECOND_ORDER_ESCAPES: u32 = 8;
        let mut escapes: u32 = 0;

        for outer in 0..self.opts.max_iter {
            let grad_f = nlp.eval_grad_f(&iter.x);
            let c_vals = c_cached.take().unwrap_or_else(|| nlp.eval_c(&iter.x));
            let f_curr = f_cached.take().unwrap_or_else(|| nlp.eval_f(&iter.x));
            let jac_c = nlp.eval_jac_c(&iter.x);
            // KKT check uses the current iterate's evaluations.
            let kkt = check_kkt(
                n, m, &iter, &grad_f, &c_vals, &bl_c, &bu_c, &xl, &xu, &jac_c,
            );
            final_stationarity = kkt.stationarity;
            final_constr_viol = kkt.constr_viol;

            // Fired BEFORE this iteration's Lagrangian Hessian is evaluated, as
            // on the interior-point arm (callback after acceptance, before the
            // next Hessian): a caller that builds `eval_h` from the iterate it
            // read in the callback (gh#1015) must see THIS iterate, not the
            // previous one.
            if let Some(hook) = self.iteration_hook.as_mut() {
                let (p_inf, alpha) = prev_step
                    .as_ref()
                    .map_or((0.0, 0.0), |s| (s.p_inf, s.alpha));
                // `check_kkt`'s stationarity vector, from the same `∇f`, `J`.
                let mut grad_lag_x = grad_f.clone();
                for k in 0..jac_c.irow.len() {
                    let (i, j) = ((jac_c.irow[k] - 1) as usize, (jac_c.jcol[k] - 1) as usize);
                    if let (Some(g), Some(&l)) = (grad_lag_x.get_mut(j), iter.lambda_g.get(i)) {
                        *g += jac_c.vals[k] * l;
                    }
                }
                for (g, &lx) in grad_lag_x.iter_mut().zip(iter.lambda_x.iter()) {
                    *g -= lx;
                }
                let keep_going = hook(&SqpIterationInfo {
                    iter: outer,
                    obj: f_curr,
                    inf_pr: kkt.constr_viol,
                    inf_du: kkt.stationarity,
                    p_inf,
                    alpha,
                    x: &iter.x,
                    lambda_g: &iter.lambda_g,
                    lambda_x: &iter.lambda_x,
                    c_vals: &c_vals,
                    grad_lag_x: &grad_lag_x,
                });
                if !keep_going {
                    self.iterates = Some(iter.clone());
                    return Ok(SqpResult {
                        x: iter.x,
                        lambda_g: iter.lambda_g,
                        lambda_x: iter.lambda_x,
                        obj: f_curr,
                        status: SqpStatus::UserRequestedStop,
                        n_iter: outer,
                        n_qp_solves,
                        n_qp_working_set_changes,
                        final_stationarity,
                        final_constr_viol,
                        working_set: iter.working,
                    });
                }
            }

            let mut low_rank = None;
            let hess_lag = match self.hessian {
                SqpHessianSource::Exact => nlp.eval_hess_lag(&iter.x, &iter.lambda_g),
                SqpHessianSource::DampedBfgs => {
                    let bfgs = bfgs.as_mut().expect("DampedBfgs state initialized above");
                    if let Some((s, y)) =
                        curvature_pair(prev_point.as_ref(), &iter, &grad_f, &jac_c, n)
                    {
                        bfgs.update_sy(&s, &y);
                    }
                    bfgs.as_triplet()
                }
                SqpHessianSource::Lbfgs => {
                    let lb = lbfgs.as_mut().expect("LBfgs state initialized above");
                    if let Some((s, y)) =
                        curvature_pair(prev_point.as_ref(), &iter, &grad_f, &jac_c, n)
                    {
                        lb.update_sy(&s, &y);
                    }
                    if lift_lbfgs {
                        low_rank = Some(lb.low_rank());
                        // Never read: the step comes from `low_rank`, and the
                        // only other consumer (the second-order check at a
                        // KKT point) takes the exact Hessian on this source.
                        Triplet {
                            n_rows: n,
                            n_cols: n,
                            irow: Vec::new(),
                            jcol: Vec::new(),
                            vals: Vec::new(),
                        }
                    } else {
                        lb.as_triplet()
                    }
                }
                SqpHessianSource::QuasiNewtonAuto => {
                    unreachable!("resolved at the top of the solve")
                }
            };

            // Remember this iterate's `(x, ∇f, ∇c)` so the next
            // iteration can build its curvature pair at a fixed
            // multiplier. See `curvature_pair`.
            prev_point = Some((iter.x.clone(), grad_f.clone(), jac_c.clone()));

            // Non-finite residuals mean the iterate itself is garbage, and
            // every gate below this point — the convergence test, the
            // gh #856 negative-curvature escape installed on it, the filter,
            // the merit function — is a comparison against a tolerance that
            // a `NaN` cannot inform. Stop here and say so, mirroring the
            // interior-point arm's `if !nlp_err.is_finite()` screen
            // (`ipopt_alg.rs`) so the two arms give the same verdict on the
            // same condition (gh #876).
            //
            // The objective is screened too. Without it an iterate whose
            // objective has overflowed to `-inf`, with residuals still finite,
            // went on to a line search whose trial point evaluated to `NaN`
            // and was reported as a step-size stall
            // (`Search_Direction_Becomes_Too_Small`) — on `unbounded_cubic`'s
            // limited-memory leg, once the step QPs stopped stalling at their
            // iteration cap (gh#1061) and the SQP followed the cubic down.
            if !kkt.stationarity.is_finite() || !kkt.constr_viol.is_finite() || !f_curr.is_finite()
            {
                let obj = nlp.eval_f(&iter.x);
                self.iterates = Some(iter.clone());
                return Ok(SqpResult {
                    x: iter.x,
                    lambda_g: iter.lambda_g,
                    lambda_x: iter.lambda_x,
                    obj,
                    status: SqpStatus::InvalidNumber,
                    n_iter: outer,
                    n_qp_solves,
                    n_qp_working_set_changes,
                    final_stationarity,
                    final_constr_viol,
                    working_set: iter.working,
                });
            }

            // Per-iteration summary row (`sqp_print_level >= 1`). Emitted at
            // the TOP of the iteration, on Ipopt's convention: row `k`
            // reports the residuals AT `x_k`, and its step columns describe
            // the step that arrived there — so they are blank on row 0. The
            // interior-point arm's table reads the same way
            // (`crate::output::orig::OrigIterationOutput::format_row`), which
            // is the whole point: a reader should not have to learn a second
            // convention to compare the two arms on one model.
            //
            // Written with `println!` rather than through a journalist because
            // `SqpAlgorithm` has no handle on one — the same reason
            // `pounce_solve_report::console` prints the problem statistics and
            // the end-of-run summary that bracket this table directly.
            //
            // This block used to be `#[cfg(test)]` around a `tracing::debug!`,
            // which meant the option was registered, documented in
            // `docs/src/active-set-sqp.md` as "1=per-iter summary", threaded
            // all the way to `opts.print_level` — and produced nothing
            // whatsoever in any binary a user can run, at any level, with or
            // without `RUST_LOG`. Same shape as gh#677
            // (`limited_memory_initialization` registered and never read) and
            // the `sqp_qp_use_homotopy` no-op: a switch nothing reads is worse
            // than no switch, because its documentation describes behaviour
            // that does not exist.
            if self.opts.print_level >= 1 {
                if outer % 10 == 0 {
                    println!("{SQP_ITER_HEADER}");
                }
                println!(
                    "{}",
                    format_sqp_row(
                        outer,
                        f_curr,
                        kkt.constr_viol,
                        kkt.stationarity,
                        prev_step.as_ref(),
                    )
                );
            }
            if self.opts.print_level >= 2 {
                println!(
                    "         x={:?} ν={:.2e}",
                    iter.x.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(),
                    nu,
                );
            }

            // `sqp_tol` and `sqp_dual_inf_tol` are both registered and both
            // documented as a max-norm tolerance on the stationarity
            // residual, but only `dual_inf_tol` was ever read — `opts.tol`
            // (default 1e-8) was dead, so the loose 1e-4 governed alone and
            // silently capped attainable accuracy (max x-error `7e-5` on the
            // #358 sweep). Honor both by requiring the tighter, which is the
            // only reading under which neither option is a no-op. Restores
            // `~5e-9` worst-case accuracy for ~10% more iterations. Same
            // registered-but-inert defect family as gh #360.
            let stationarity_tol = self.opts.tol.min(self.opts.dual_inf_tol);
            if kkt.stationarity <= stationarity_tol && kkt.constr_viol <= self.opts.constr_viol_tol
            {
                // First-order KKT is necessary and not sufficient. Before
                // reporting success on an indefinite Lagrangian, look for a
                // feasible direction of negative curvature *at the converged
                // multipliers* and, where one is found, step along it and keep
                // going (gh #856).
                //
                // `nonconvex_qp.nl` under `algorithm=active-set-sqp` is the
                // case: `min x₀x₁ s.t. x₀+x₁ = 2, 0 ≤ x ≤ 4`, on whose
                // feasible segment `f(x₀) = x₀(2−x₀)` is concave, so the
                // `(1, 1)` this converges to at `f = 1` is the constrained
                // **maximum** and the minimum is `0` at either endpoint. It was
                // reported `Solve_Succeeded`. The escape below moves to
                // `(2, 0)`, where a bound joins the active set, the null space
                // closes and `f = 0` certifies.
                //
                // Refuted **by exhibition**, exactly as gh #848 does one layer
                // down: the direction is only acted on after stepping along it
                // and finding the true objective lower, so the curvature search
                // is free to be approximate and a direction it gets wrong
                // costs an evaluation rather than a wrong answer.
                //
                // The Hessian is the **exact** `∇²L`, taken here even when the
                // steps were driven by a quasi-Newton one. That is not an
                // optimization detail, it is what makes the check exist at all
                // under `limited-memory`: a damped-BFGS or L-BFGS matrix is
                // positive definite by construction, so searching it for
                // negative curvature can only ever find none, and gating the
                // check on `SqpHessianSource::Exact` left the L-BFGS leg
                // certifying the same constrained maximum. `eval_hess_lag` is
                // a required method of `SqpProblemSpec`, so it is always
                // callable; this costs one Hessian evaluation per converged
                // solve, and an implementation that has none to give returns
                // an empty triplet, which finds no curvature and changes
                // nothing.
                //
                // The L-BFGS leg is not exotic coverage: the Python frontend
                // and the CasADi plugin both select `limited-memory` on their
                // own whenever no exact Lagrangian Hessian is available.
                let exact_hess = if matches!(self.hessian, SqpHessianSource::Exact) {
                    None
                } else {
                    Some(nlp.eval_hess_lag(&iter.x, &iter.lambda_g))
                };
                if escapes < MAX_SECOND_ORDER_ESCAPES
                    && let Some(d) = negative_curvature_at_kkt_point(
                        n,
                        m,
                        &iter.x,
                        exact_hess.as_ref().unwrap_or(&hess_lag),
                        &jac_c,
                        &c_vals,
                        &bl_c,
                        &bu_c,
                        &xl,
                        &xu,
                        self.opts.constr_viol_tol,
                    )
                    && let Some(next) = exhibit_better_point(
                        nlp,
                        &iter.x,
                        &d,
                        f_curr,
                        &xl,
                        &xu,
                        &bl_c,
                        &bu_c,
                        self.opts.constr_viol_tol,
                    )
                {
                    tracing::debug!(target: "pounce::sqp",
                        "first-order KKT point refuted at second order; stepping \
                         along negative curvature to a strictly better feasible \
                         point (gh #856)");
                    escapes += 1;
                    iter.x = next;
                    iter.working = None;
                    f_cached = None;
                    c_cached = None;
                    prev_point = None;
                    continue;
                }
                self.iterates = Some(iter.clone());
                return Ok(SqpResult {
                    x: iter.x,
                    lambda_g: iter.lambda_g,
                    lambda_x: iter.lambda_x,
                    obj: f_curr,
                    status: SqpStatus::Optimal,
                    n_iter: outer,
                    n_qp_solves,
                    n_qp_working_set_changes,
                    final_stationarity,
                    final_constr_viol,
                    working_set: iter.working,
                });
            }

            // Time budget (gh#1054). Tested after the convergence check, so
            // an iterate that is already optimal is reported as such however
            // late it arrived, and before the step QP, which is where an
            // active-set solve at scale spends its time.
            if let Some(kind) = self.opts.deadline.as_ref().and_then(|d| d.exceeded()) {
                self.iterates = Some(iter.clone());
                return Ok(SqpResult {
                    x: iter.x,
                    lambda_g: iter.lambda_g,
                    lambda_x: iter.lambda_x,
                    obj: f_curr,
                    status: kind.into(),
                    n_iter: outer,
                    n_qp_solves,
                    n_qp_working_set_changes,
                    final_stationarity,
                    final_constr_viol,
                    working_set: iter.working,
                });
            }
            // What is left of it, handed to every step QP this iteration as
            // its `time_limit`, so one subproblem cannot outlive the solve.
            // `qp_budget` remembers which budget set the limit, so a QP that
            // stops on it is reported as that budget and not as the
            // subproblem's own iteration limit.
            let qp_budget = self.opts.deadline.as_ref().map(deadline_remaining);
            let qp_time_limit =
                qp_time_limit_now(self.opts.deadline.as_ref(), self.qp_opts.time_limit);

            let qp_data = match low_rank.as_ref() {
                Some(b) => SqpQpData::build_low_rank(
                    &iter.x, &grad_f, &c_vals, &bl_c, &bu_c, &xl, &xu, jac_c, b,
                ),
                None => SqpQpData::build(
                    &iter.x,
                    &grad_f,
                    &c_vals,
                    &bl_c,
                    &bu_c,
                    &xl,
                    &xu,
                    jac_c,
                    hess_lag,
                    self.hessian_inertia(),
                ),
            };

            // Scale-relative inner-QP tolerances (issue #358 tail).
            //
            // `QpOptions::{feas_tol, opt_tol}` are **absolute** (1e-9 each).
            // That is a sane default for a standalone `solve_qp` on
            // well-scaled data, but this QP is an *inner* subproblem whose
            // data inherits the NLP's scale: with `‖∇f‖ ~ 1e3` and
            // `‖B‖ ~ 1e3`, an absolute 1e-9 is ~1e-12 *relative* — at the
            // f64 noise floor. The active-set solver then cannot certify
            // its own optimality, burns its whole iteration budget, and
            // returns `MaxIter`; the driver reports `QpStepFailed`, which
            // surfaces to the user as `Search_Direction_Becomes_Too_Small`
            // on a QP that is trivially solvable. This is what stalled the
            // ill-conditioned tail of #358 even once the Hessian scale was
            // fixed by the probe above.
            //
            // Scale both tolerances by the QP data magnitude, so the inner
            // solve is asked for a *relative* accuracy it can actually
            // reach. Nothing is lost in the answer: the SQP outer loop
            // still gates optimality on the true, unscaled NLP KKT
            // residuals (`dual_inf_tol` / `constr_viol_tol`) at the top of
            // each iteration, so a sloppier inner step can only cost an
            // extra outer iteration — never a false `Optimal`. Measured on
            // a 500-instance convex-QP sweep this converts 34 failures into
            // successes with a *bit-for-bit identical* error distribution
            // (median 6e-11, max true constraint violation 4e-11).
            //
            // The `SCALE_MAX` clamp bounds the relaxation on pathological
            // data (a quasi-Newton `B` that has blown up); it does not bind
            // on any problem in the sweep.
            const SCALE_MAX: Number = 1e6;
            let g_inf = grad_f.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let b_inf = qp_data.hess_inf;
            let base = self.qp_opts.clone();
            self.qp_opts.time_limit = qp_time_limit;
            let scale = g_inf.max(b_inf).clamp(1.0, SCALE_MAX);
            if scale > 1.0 {
                self.qp_opts.opt_tol = base.opt_tol * scale;
                self.qp_opts.feas_tol = base.feas_tol * scale;
            }

            // Warm-start from the previous QP's working set when
            // available. Pounce-qp's `solve_with_working_set`
            // internally computes a feasible primal compatible
            // with the supplied set (it satisfies every active
            // row exactly) — necessary because each SQP
            // linearization shifts the QP's constraint RHS by
            // `-c(x_k)`, so the previous QP's *primal* doesn't
            // carry over even when the active set does.
            let warm_started = iter.working.is_some();
            let mut sol = if let Some(prev_w) = iter.working.as_ref() {
                // A warm solve that *errors* falls back to cold rather than
                // aborting the SQP (gh #855). The cold-start fallback below
                // already exists for a warm solve that comes back `MaxIter` /
                // `NumericalError`, on the reasoning that the carried-over
                // working set can be a poor guess; a hard `Err` from the same
                // call is the **stronger** form of that signal and was the one
                // case the fallback could not see, because `?` propagated out
                // of the whole algorithm first.
                //
                // The error that motivates this says so itself: `eigena2`
                // under `algorithm=active-set-sqp` reaches outer iteration 17
                // and the warm solve fails with "pinned KKT constraint block
                // is rank-deficient (inertia shift masked a singular
                // constraint block); prune to a linearly-independent subset".
                // That is a statement about the *pinned set*, which is exactly
                // what a warm start supplies and a cold start rebuilds. The
                // SQP exited `Internal_Error` / `solve_result_num=500` -- "the
                // solver broke, retry" -- on a model whose objective it can
                // report; with the cold re-solve it ends
                // `Maximum_Iterations_Exceeded` at `obj = 82.5177`, against
                // the NLP arm's 82.5 on the same file.
                //
                // The cold error is still propagated: if a clean start fails
                // too, the failure is not about the working set and there is
                // nothing further to try here.
                match solve_step_qp(
                    &mut self.qp_solver,
                    &qp_data,
                    Some(prev_w),
                    &self.qp_opts,
                    self.ipm_cold_start.then_some(&self.ipm_seed),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::debug!(target: "pounce::sqp",
                            "warm-started step QP failed hard ({e:?}); re-solving \
                             from cold, since the carried working set is what a \
                             cold start rebuilds (gh #855)");
                        n_qp_solves += 1;
                        solve_step_qp(
                            &mut self.qp_solver,
                            &qp_data,
                            None,
                            &self.qp_opts,
                            self.ipm_cold_start.then_some(&self.ipm_seed),
                        )?
                    }
                }
            } else {
                solve_step_qp(
                    &mut self.qp_solver,
                    &qp_data,
                    None,
                    &self.qp_opts,
                    self.ipm_cold_start.then_some(&self.ipm_seed),
                )?
            };
            n_qp_solves += 1;
            n_qp_working_set_changes += sol.stats.n_working_set_changes;

            // Cold-start fallback: a warm start seeds the QP with the
            // previous iterate's working set, which is usually a big
            // win but can occasionally strand the active-set solver at
            // its iteration limit (or a numerical breakdown) on a QP
            // that is perfectly solvable from a clean start — e.g.
            // when a quasi-Newton Hessian has drifted enough that the
            // carried-over active set is a poor guess. Rather than
            // give up with `QpStepFailed`, re-solve once from cold;
            // this is what rescues the curved-constraint SQP runs of
            // issue #349 that previously reported
            // `Search_Direction_Becomes_Too_Small`.
            if warm_started && matches!(sol.status, QpStatus::MaxIter | QpStatus::NumericalError) {
                let cold = solve_step_qp(
                    &mut self.qp_solver,
                    &qp_data,
                    None,
                    &self.qp_opts,
                    self.ipm_cold_start.then_some(&self.ipm_seed),
                )?;
                n_qp_solves += 1;
                n_qp_working_set_changes += cold.stats.n_working_set_changes;
                // `Unbounded` is accepted as well as `Optimal`, and it is
                // the point of gh #855. A cold re-solve that comes back
                // `Unbounded` has *found something* — an unblocked direction
                // of negative curvature in the null space of its working set
                // — and taking only `Optimal` threw that away, leaving `sol`
                // on the original `MaxIter`. The unbounded-model fallback
                // below is gated on `sol.status == Unbounded`, so the
                // δ-shifted proximal step written for exactly this situation
                // was unreachable from a retry.
                //
                // Accepting it is safe because the fallback does not trust it
                // either: it re-tests the ray against the true NLP (gh #388)
                // and only takes the proximal branch when the certificate
                // does *not* survive. A spurious `Unbounded` therefore costs
                // one re-test, not a wrong verdict.
                //
                // Nothing else can be accepted here: `Infeasible` from a cold
                // solve would contradict the warm one on the same subproblem
                // without a tie-breaker, and `MaxIter` / `NumericalError` are
                // what we already have.
                //
                // COVERAGE, stated because it matters: no fixture in the CLI
                // corpus reaches this branch. Swept under
                // `algorithm=active-set-sqp`, the retries fire on three
                // fixtures (`cresc4`, `eigena2`, `jit1_boxed`) and return only
                // `MaxIter` or `Optimal`, including under `sqp_qp_max_iter`
                // forced down to 2. gh #855 observed the `Unbounded` return on
                // `eigena2` in a build carrying second-order certification for
                // the step subproblem, which is gh #856's subject and does not
                // exist here yet. It is kept rather than dropped because it is
                // not redundant -- nothing else makes the unbounded-model
                // fallback reachable from a retry -- which is the opposite of
                // the gh #846 case, where a second arm already rejected
                // everything the removed one would have.
                if matches!(cold.status, QpStatus::Optimal | QpStatus::Unbounded) {
                    sol = cold;
                }
            }

            // Quasi-Newton reset fallback (issue #358 tail). If the QP
            // still cannot be solved, the usual culprit is not the
            // linearization but the *approximated* Hessian: a damped-BFGS
            // matrix that has accumulated enough drift (typically after a
            // large early step on an ill-conditioned problem) to make the
            // step subproblem numerically unsolvable. That is recoverable
            // — throwing away the accumulated curvature and retrying from
            // a scaled identity almost always yields a usable step —
            // whereas the alternative is aborting an otherwise healthy
            // solve with `QpStepFailed`, which the user sees as
            // `Search_Direction_Becomes_Too_Small` on a trivially solvable
            // problem. Rebuild the subproblem around the reset Hessian and
            // re-solve once, from cold (the carried working set belongs to
            // the discarded model).
            let mut qp_data = qp_data;
            if matches!(sol.status, QpStatus::MaxIter | QpStatus::NumericalError)
                && let Some(b) = bfgs.as_mut()
            {
                b.reset_to_scale();
                qp_data = SqpQpData::build(
                    &iter.x,
                    &grad_f,
                    &c_vals,
                    &bl_c,
                    &bu_c,
                    &xl,
                    &xu,
                    nlp.eval_jac_c(&iter.x),
                    b.as_triplet(),
                    self.hessian_inertia(),
                );
                let retry = solve_step_qp(
                    &mut self.qp_solver,
                    &qp_data,
                    None,
                    &self.qp_opts,
                    self.ipm_cold_start.then_some(&self.ipm_seed),
                )?;
                n_qp_solves += 1;
                n_qp_working_set_changes += retry.stats.n_working_set_changes;
                // Same as the cold retry above (gh #855): an `Unbounded`
                // verdict is a finding, not a failure, and discarding it hid
                // the proximal fallback from this branch too.
                //
                // The subproblem stays consistent: this branch runs only
                // while `sol.status` is `MaxIter`/`NumericalError`, so it
                // cannot fire after the cold retry has been accepted, and
                // `qp_data` is rebound above — so the fallback below re-solves
                // the *reset-Hessian* subproblem that produced this verdict,
                // not the discarded one.
                if matches!(retry.status, QpStatus::Optimal | QpStatus::Unbounded) {
                    sol = retry;
                }
            }

            // Unbounded-model fallback (gh #423). The step QP being
            // unbounded below is a statement about the *linearization*, so
            // re-test the ray against the true NLP (gh #388) — and when it
            // does not survive, do not stop there. An unbounded model on a
            // bounded NLP is not a dead end; it is the textbook signal that
            // the model needs regularizing (Nocedal-Wright §18.4), and δ
            // from §4.5 inertia control already *is* that regularization.
            // So re-solve declining the certificate: the same subproblem,
            // the same shift, but the unblocked direction takes the
            // δ-shifted proximal step instead of certifying recession.
            //
            // This is not a corner case. A nonconvex NLP with `m = 0` and
            // no finite bounds has *nothing that can ever block* a
            // negative-curvature direction, so every indefinite iterate
            // produces this certificate. gh #419 gave those iterates a real
            // step where a bound exists and left them with none where one
            // does not: a chain of coupled double wells (`n = 12`, `m = 0`)
            // that converged to f = 0.027424 in 24 iterations died at
            // iteration 1 with `QpStepFailed` at f = 26.03. The proximal
            // step is slow — that slowness is what #416 was about — but it
            // is a step, and it is only reached here where the alternative
            // is no step at all.
            let mut ray_certified = false;
            if sol.status == QpStatus::Unbounded {
                ray_certified = sol.unbounded_ray.as_ref().is_some_and(|d| {
                    ray_certifies_unbounded(
                        nlp,
                        &iter.x,
                        d,
                        f_curr,
                        &grad_f,
                        &bl_c,
                        &bu_c,
                        &xl,
                        &xu,
                        self.opts.constr_viol_tol,
                    )
                });
                if !ray_certified {
                    tracing::debug!(target: "pounce::sqp",
                        "unbounded step QP whose recession ray does not survive \
                         re-testing against the NLP — re-solving for the δ-shifted \
                         proximal step (gh #423)");
                    let prox_opts = QpOptions {
                        certify_recession_ray: false,
                        ..self.qp_opts.clone()
                    };
                    let prox = solve_step_qp(
                        &mut self.qp_solver,
                        &qp_data,
                        None,
                        &prox_opts,
                        self.ipm_cold_start.then_some(&self.ipm_seed),
                    )?;
                    n_qp_solves += 1;
                    if prox.status == QpStatus::Optimal {
                        sol = prox;
                    }
                }
            }
            self.qp_opts = base;

            // An infeasible step QP is a statement about the linearization at
            // `x_k`, not about the NLP (Fletcher's Sℓ1QP; Nocedal-Wright
            // §18.5). The QP's elastic phase hands back the point minimizing
            // its ℓ1 violation, which is a step toward feasibility; when it
            // reduces the linearized violation, take it and let the ℓ1 merit
            // line search judge it. The multipliers it returns are a Farkas
            // certificate, not a step's multipliers, so the current ones are
            // kept. Only a step that cannot reduce the linearized violation —
            // `x_k` stationary for it — is the local-infeasibility signal.
            //
            // Stopping on the first infeasible QP declared `deb7` (813
            // variables, feasible: the interior point solves it at 97.56)
            // `Infeasible_Problem_Detected` at iteration 0, and the
            // second-opinion ladder agreed three times, since every re-solve
            // starts from the same linearization (gh#1061 sweep).
            let elastic_step = sol.status == QpStatus::Infeasible && {
                let zero = vec![0.0; n];
                let before = linearized_l1_violation(&qp_data, &zero);
                let after = linearized_l1_violation(&qp_data, &sol.x);
                tracing::debug!(target: "pounce::sqp",
                    "infeasible step QP: linearized ℓ1 violation {before:.3e} → {after:.3e} \
                     along its elastic step");
                after < (1.0 - ELASTIC_STEP_MIN_GAIN) * before
            };
            if elastic_step {
                sol.lambda_g.clone_from(&iter.lambda_g);
                sol.lambda_x.clone_from(&iter.lambda_x);
            }

            match sol.status {
                QpStatus::Optimal => {}
                QpStatus::Infeasible if elastic_step => {}
                QpStatus::Infeasible => {
                    let obj = nlp.eval_f(&iter.x);
                    self.iterates = Some(iter.clone());
                    return Ok(SqpResult {
                        x: iter.x,
                        lambda_g: iter.lambda_g,
                        lambda_x: iter.lambda_x,
                        obj,
                        status: SqpStatus::InfeasibleSubproblem,
                        n_iter: outer,
                        n_qp_solves,
                        n_qp_working_set_changes,
                        final_stationarity,
                        final_constr_viol,
                        working_set: iter.working,
                    });
                }
                // The QP subproblem neither solved nor certified
                // infeasibility. `MaxIter` / `NumericalError` mean the
                // active-set QP could not resolve the (typically extremely
                // degenerate) step subproblem — the m/n ≫ 1 collapsed-cone
                // geometry of #282. Terminate the SQP with an HONEST
                // non-committal status rather than a hard error, and — the
                // point of #282 — WITHOUT ever asserting infeasibility on a
                // problem we have not certified infeasible.
                QpStatus::MaxIter | QpStatus::TimeLimit | QpStatus::NumericalError => {
                    let obj = nlp.eval_f(&iter.x);
                    self.iterates = Some(iter.clone());
                    // Report *which* of the two it was. Both are honest
                    // non-committal failures making no infeasibility claim
                    // (#282), but only the budget one is actionable by the
                    // user, and merging them hid the dominant Maros-Mészáros
                    // failure mode behind a step-size verdict. See
                    // `SqpStatus::QpIterationLimit`.
                    // `TimeLimit` is the solve's own `max_wall_time` /
                    // `max_cpu_time` when the deadline set the QP's limit
                    // (gh#1054), and is reported as that. A limit the caller
                    // put on the subproblem directly rides with `MaxIter`: a
                    // budget ran out, no claim about the problem.
                    let status = match sol.status {
                        QpStatus::TimeLimit => match qp_budget {
                            Some((left, budget))
                                if self.qp_opts.time_limit.is_none_or(|own| left <= own) =>
                            {
                                budget.into()
                            }
                            _ => SqpStatus::QpIterationLimit,
                        },
                        QpStatus::MaxIter => SqpStatus::QpIterationLimit,
                        _ => SqpStatus::QpStepFailed,
                    };
                    return Ok(SqpResult {
                        x: iter.x,
                        lambda_g: iter.lambda_g,
                        lambda_x: iter.lambda_x,
                        obj,
                        status,
                        n_iter: outer,
                        n_qp_solves,
                        n_qp_working_set_changes,
                        final_stationarity,
                        final_constr_viol,
                        working_set: iter.working,
                    });
                }
                // The step QP is unbounded below along a certified
                // recession ray of the LOCAL model (zero curvature,
                // feasible for every step length, strict descent). That
                // is a statement about the linearization, not yet about
                // the NLP — so it was re-tested against the true
                // objective and constraints above. If it survived, the
                // NLP itself is unbounded and we say so with the same
                // `Diverging_Iterates` verdict every other POUNCE path
                // returns. If it did not, the proximal re-solve above
                // already had its chance to produce a step, and reaching
                // here means even that failed: the QP simply could not
                // produce a usable step, which is `QpStepFailed`.
                //
                // Neither outcome is a hard error. This used to return
                // `QpFailure(LinearSolverFailure("QP subproblem returned
                // status unbounded"))`, which surfaced to AMPL / Pyomo /
                // GAMS consumers as `Internal_Error` /
                // `solve_result_num=500` — "the solver broke" — on a
                // model that is merely unbounded (`300`), and named a
                // linear-solver failure when no linear solver had failed
                // (gh #388).
                QpStatus::Unbounded => {
                    let certified = ray_certified;
                    let obj = nlp.eval_f(&iter.x);
                    self.iterates = Some(iter.clone());
                    return Ok(SqpResult {
                        x: iter.x,
                        lambda_g: iter.lambda_g,
                        lambda_x: iter.lambda_x,
                        obj,
                        status: if certified {
                            SqpStatus::Unbounded
                        } else {
                            SqpStatus::QpStepFailed
                        },
                        n_iter: outer,
                        n_qp_solves,
                        n_qp_working_set_changes,
                        final_stationarity,
                        final_constr_viol,
                        working_set: iter.working,
                    });
                }
            }

            let p_inf = sol.x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            if self.opts.print_level >= 2 {
                println!(
                    "         qp: ‖p‖_inf={:.3e} ‖λ_g_qp‖_inf={:.3e}",
                    p_inf,
                    sol.lambda_g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
                );
            }
            // Globalization: l1-merit backtracking (Han-Powell)
            // or filter (Fletcher-Leyffer 2002). The two share
            // the same backtracking shell + acceptance API; the
            // filter keeps state across iterations on
            // `self.filter`.
            //
            // Both are handed a second-order-correction (SOC)
            // provider (the Maratos remedy). When the full step
            // (α = 1) is rejected because it increased the
            // constraint violation, the line search calls this
            // closure with `c(x_k + p)` to obtain a corrected full
            // step. We build that step by re-solving the SAME QP
            // with the general-constraint RHS re-centered on the
            // trial-point constraint values: the original QP models
            // `c(x_k) + A p`, and the SOC replaces `c(x_k)` by
            // `c(x_k + p) − A p`, so the correction subproblem
            // targets the true (curved) violation at the trial
            // point (Nocedal-Wright §18.11). The just-solved working
            // set warm-starts the correction. Only meaningful with
            // general constraints (`m > 0`).
            //
            // Pre-computed `A p` for the RHS re-centering:
            let a_p = if m > 0 {
                mat_vec_gen(&qp_data.a, &sol.x, m)
            } else {
                Vec::new()
            };
            let mut n_soc_solves: u32 = 0;
            // Working set from the SOC subproblem, kept so that a
            // taken SOC step warm-starts the next iteration from the
            // active set that actually describes `x + p_soc` (not the
            // original QP's set, which belongs to the rejected step).
            let mut soc_working: Option<WorkingSet> = None;
            let ls = {
                let qp_solver = &mut self.qp_solver;
                let qp_opts = &self.qp_opts;
                let deadline = self.opts.deadline.as_ref();
                let seed = self.ipm_cold_start.then_some(&self.ipm_seed);
                let qp_data_ref = &qp_data;
                let c_curr_ref = &c_vals;
                let a_p_ref = &a_p;
                let sol_working = &sol.working;
                let n_soc = &mut n_soc_solves;
                let soc_working_slot = &mut soc_working;
                let mut soc = |c_trial: &[Number]| -> Option<crate::sqp::line_search::SocStep> {
                    let mm = qp_data_ref.m;
                    // Re-center the general-constraint RHS on the
                    // trial-point violation, preserving ±∞ sentinels.
                    let mut bl_soc = qp_data_ref.bl.clone();
                    let mut bu_soc = qp_data_ref.bu.clone();
                    for i in 0..mm {
                        let delta = c_curr_ref[i] - c_trial[i] + a_p_ref[i];
                        if qp_data_ref.bl[i] > NLP_LOWER_BOUND_INF {
                            bl_soc[i] = qp_data_ref.bl[i] + delta;
                        }
                        if qp_data_ref.bu[i] < NLP_UPPER_BOUND_INF {
                            bu_soc[i] = qp_data_ref.bu[i] + delta;
                        }
                    }
                    // `bl` / `bu` carry the lifted rows (if any) past `mm`;
                    // only the NLP's own rows are re-centred.
                    let qp_soc = qp_data_ref.as_qp_with_row_bounds(&bl_soc, &bu_soc);
                    // What is left of the solve's budget now, not at the top of
                    // the iteration: the step QP and the trial points have
                    // spent part of it since (gh#1061: the correction QP ran
                    // 40–142 s past a wall cap on that stale figure).
                    let soc_opts = QpOptions {
                        time_limit: qp_time_limit_now(deadline, qp_opts.time_limit),
                        ..qp_opts.clone()
                    };
                    let sol_soc = solve_subproblem_qp(
                        qp_solver,
                        qp_data_ref,
                        qp_soc,
                        Some(sol_working),
                        &soc_opts,
                        seed,
                        "SOC QP",
                    )
                    .ok()?;
                    *n_soc += 1;
                    if sol_soc.status == QpStatus::Optimal {
                        *soc_working_slot = Some(sol_soc.working);
                        Some(crate::sqp::line_search::SocStep {
                            p: sol_soc.x,
                            lambda_g: sol_soc.lambda_g,
                            lambda_x: sol_soc.lambda_x,
                        })
                    } else {
                        None
                    }
                };
                let soc_ref: Option<crate::sqp::line_search::SocProvider<'_>> =
                    if m > 0 { Some(&mut soc) } else { None };
                match self.opts.globalization {
                    SqpGlobalization::L1Elastic => l1_merit_line_search(
                        nlp,
                        &iter.x,
                        &sol.x,
                        &sol.lambda_g,
                        &grad_f,
                        f_curr,
                        &c_vals,
                        &bl_c,
                        &bu_c,
                        &xl,
                        &xu,
                        nu,
                        &self.opts,
                        soc_ref,
                    ),
                    SqpGlobalization::Filter => filter_line_search(
                        nlp,
                        &mut self.filter,
                        &iter.x,
                        &sol.x,
                        f_curr,
                        &c_vals,
                        &bl_c,
                        &bu_c,
                        &xl,
                        &xu,
                        nu,
                        &self.opts,
                        soc_ref,
                    ),
                }
            };
            n_qp_solves += n_soc_solves;
            if self.opts.print_level >= 2 {
                println!(
                    "         ls: α={:.3e} ν={:.3e} ok={} f_new={:.3e}",
                    ls.alpha, ls.nu, ls.success, ls.f_new
                );
            }
            if !ls.success {
                self.iterates = Some(iter.clone());
                return Ok(SqpResult {
                    x: iter.x,
                    lambda_g: iter.lambda_g,
                    lambda_x: iter.lambda_x,
                    obj: f_curr,
                    status: SqpStatus::LineSearchFailed,
                    n_iter: outer,
                    n_qp_solves,
                    n_qp_working_set_changes,
                    final_stationarity,
                    final_constr_viol,
                    working_set: Some(sol.working),
                });
            }
            iter.x = ls.x_new;
            // Captured before the match below moves `sol.working` and
            // `ls.soc_duals` out from under them. Both are cheap Copy reads;
            // taking them here rather than after is what keeps the row's
            // `ws` column this QP's own count instead of the running total.
            let qp_ws_changes = sol.stats.n_working_set_changes;
            let soc_taken = ls.soc_duals.is_some();
            match ls.soc_duals {
                Some((soc_lg, soc_lx)) => {
                    // A second-order-correction step was taken (α = 1
                    // on the SOC subproblem). Adopt the SOC
                    // subproblem's own multipliers and working set so
                    // `(step, multipliers, active set)` stay a
                    // consistent triple — required for the quasi-
                    // Newton Hessian update to stay well-conditioned
                    // and for the next QP to warm-start correctly.
                    iter.lambda_g = soc_lg;
                    iter.lambda_x = soc_lx;
                    iter.working = soc_working.take().or(Some(sol.working));
                }
                None => {
                    for (l, &lq) in iter.lambda_g.iter_mut().zip(sol.lambda_g.iter()) {
                        *l = (1.0 - ls.alpha) * *l + ls.alpha * lq;
                    }
                    for (l, &lq) in iter.lambda_x.iter_mut().zip(sol.lambda_x.iter()) {
                        *l = (1.0 - ls.alpha) * *l + ls.alpha * lq;
                    }
                    iter.working = Some(sol.working);
                }
            }
            nu = ls.nu;
            f_cached = Some(ls.f_new);
            c_cached = Some(ls.c_new);
            // Recorded after acceptance, so the row describes the step that
            // was actually taken rather than the last one tried. `ws_changes`
            // is this QP's contribution to the running total, which is the
            // number the working-set warm start is judged on.
            prev_step = Some(SqpStepRecord {
                p_inf,
                alpha: ls.alpha,
                soc: soc_taken,
                ws_changes: qp_ws_changes,
            });
        }

        let obj = nlp.eval_f(&iter.x);
        self.iterates = Some(iter.clone());
        Ok(SqpResult {
            x: iter.x,
            lambda_g: iter.lambda_g,
            lambda_x: iter.lambda_x,
            obj,
            status: SqpStatus::MaxIter,
            n_iter: self.opts.max_iter,
            n_qp_solves,
            n_qp_working_set_changes,
            final_stationarity,
            final_constr_viol,
            working_set: iter.working,
        })
    }

    fn hessian_inertia(&self) -> HessianInertia {
        if self.hessian.is_quasi_newton() {
            // Damped BFGS and L-BFGS are PSD by construction. (A lifted
            // L-BFGS subproblem sets its own, see `build_low_rank`.)
            HessianInertia::Psd
        } else {
            // Exact ∇²L is indefinite on nonconvex NLPs; let the
            // QP solver's §4.5 inertia control handle it.
            HessianInertia::Indefinite
        }
    }
}

/// gh #388: does the step QP's certified recession ray certify the **NLP**
/// unbounded below?
///
/// The inner QP hands back a direction `d` that is a recession ray *of the
/// linearization at `x`*: `∇²L d ≈ 0`, `d` feasible for the linearized
/// constraints at every step length, `∇q(x)ᵀd < 0`. On an LP or a QP that
/// linearization is exact and `d` is a recession ray of the original
/// problem; on a general NLP it need not be — the constraints curve back
/// and the objective can turn around. The two cases must not share a
/// status, so we settle it by evaluation rather than by faith: walk the
/// ray and check, at the **true** `f` and `c`, that
///
///  1. every probe point is *feasible* (variable bounds and constraint
///     bounds, the latter with a roundoff allowance that grows with the
///     row scale so a linear row evaluated at `‖x‖ ~ 1e12` is not failed
///     on cancellation noise), and
///  2. the objective keeps falling at **at least half** the initial linear
///     rate `∇f(x)ᵀd` — not merely falling. A ray that decelerates is
///     settling onto a finite optimum, the same distinction the IPM's
///     divergence guard draws (#248/#252/#285).
///
/// Probes span twelve decades of step length, so a "pass" is a family of
/// genuinely feasible points whose objective marches to `−∞` at a linear
/// rate over `1e12`. Anything short of that — one infeasible probe, one
/// decelerating decade, a NaN — returns `false` and the caller reports the
/// non-committal `QpStepFailed` instead. False negatives cost an honest
/// "no step" status; a false positive would tell a modeler their bounded
/// model is unbounded, so the asymmetry is deliberate.
///
/// `dir` need not be normalized (it is rescaled to unit max-norm here, so
/// the probe lengths are in the iterate's own units).
#[allow(clippy::too_many_arguments)]
fn ray_certifies_unbounded<N: SqpProblemSpec>(
    nlp: &mut N,
    x: &[Number],
    dir: &[Number],
    f_x: Number,
    grad_f: &[Number],
    bl_c: &[Number],
    bu_c: &[Number],
    xl: &[Number],
    xu: &[Number],
    constr_viol_tol: Number,
) -> bool {
    /// Step lengths along the unit-max-norm ray, spanning twelve decades.
    const PROBES: [Number; 7] = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12];
    /// Roundoff allowance per unit of `row_scale · ‖x‖∞` when checking a
    /// constraint at a far-out probe: comfortably above f64 epsilon
    /// (`2.2e-16`) to absorb accumulation over a row, far below anything
    /// a real violation would produce.
    const ROUNDOFF_REL: Number = 1e-12;

    let n = x.len();
    if dir.len() != n || grad_f.len() != n || !f_x.is_finite() {
        return false;
    }
    let scale = dir.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if !scale.is_finite() || scale <= 0.0 {
        return false;
    }
    let d: Vec<Number> = dir.iter().map(|v| v / scale).collect();

    // Descent of the TRUE objective along the ray. The QP certified this
    // for its own (possibly quasi-Newton) model gradient; re-derive it
    // from `∇f(x)` so the rate we hold the probes to is the real one.
    let slope: Number = grad_f.iter().zip(d.iter()).map(|(g, di)| g * di).sum();
    let g_norm = grad_f.iter().map(|v| v * v).sum::<Number>().sqrt();
    // Numerically meaningful (not roundoff-scale) descent; a NaN slope
    // fails the `is_finite` guard rather than sneaking past the comparison.
    let descent_bar = -1e-9 * g_norm.max(1.0);
    if !slope.is_finite() || slope >= descent_bar {
        return false;
    }

    // Per-row `max_j |∂c_i/∂x_j|`, the scale a linear row's value grows
    // with along the ray — the basis for the roundoff allowance in (1).
    let m = bl_c.len();
    let mut row_scale: Vec<Number> = vec![0.0; m];
    {
        let jac = nlp.eval_jac_c(x);
        for k in 0..jac.vals.len() {
            let i = (jac.irow[k] - 1) as usize;
            row_scale[i] = row_scale[i].max(jac.vals[k].abs());
        }
    }

    for &t in PROBES.iter() {
        let xt: Vec<Number> = x.iter().zip(d.iter()).map(|(xi, di)| xi + t * di).collect();
        if xt.iter().any(|v| !v.is_finite()) {
            return false;
        }

        // (1a) Variable bounds. These are exact linear rows in the probe's
        // own arithmetic, so the tolerance stays tight.
        for i in 0..n {
            let tol = 1e-9 * (1.0 + xt[i].abs());
            if xl[i] > NLP_LOWER_BOUND_INF && xt[i] < xl[i] - tol {
                return false;
            }
            if xu[i] < NLP_UPPER_BOUND_INF && xt[i] > xu[i] + tol {
                return false;
            }
        }

        // (1b) Constraint bounds, at the true (possibly nonlinear) `c`.
        let x_inf = xt.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let c = nlp.eval_c(&xt);
        if c.len() != m {
            return false;
        }
        for i in 0..m {
            if c[i].is_nan() {
                return false;
            }
            let tol =
                constr_viol_tol.max(0.0) * (1.0 + c[i].abs()) + ROUNDOFF_REL * row_scale[i] * x_inf;
            if bl_c[i] > NLP_LOWER_BOUND_INF && c[i] < bl_c[i] - tol {
                return false;
            }
            if bu_c[i] < NLP_UPPER_BOUND_INF && c[i] > bu_c[i] + tol {
                return false;
            }
        }

        // (2) Sustained (non-decelerating) descent. `-inf` passes: an
        // objective that has already overflowed downward is not evidence
        // against unboundedness.
        let f_t = nlp.eval_f(&xt);
        if f_t.is_nan() || f_t > f_x + 0.5 * slope * t {
            return false;
        }
    }
    true
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct KktError {
    pub stationarity: Number,
    pub constr_viol: Number,
}

/// Fraction of the linearized ℓ1 violation an elastic step must remove to be
/// taken (see `elastic_step` in the outer loop). Any real reduction is a
/// direction toward feasibility; the margin only keeps round-off on a
/// stationary point of the violation from passing for one.
const ELASTIC_STEP_MIN_GAIN: Number = 1e-6;

/// ℓ1 violation of the step QP's own rows and bounds at step `d`: the
/// linearization `c(x_k) + J d` against the row bounds, and `x_k + d` against
/// the variable bounds, both as the QP poses them (shifted to step space).
/// Only the NLP's rows and variables count; a lifted subproblem's extra
/// rows and columns are bookkeeping, not constraints.
fn linearized_l1_violation(data: &SqpQpData, d: &[Number]) -> Number {
    let (m, n) = (data.m, data.n);
    let ad = mat_vec_gen(&data.a, d, m);
    crate::sqp::line_search::l1_violation(
        &d[..n],
        &ad,
        &data.bl[..m],
        &data.bu[..m],
        &data.xl[..n],
        &data.xu[..n],
    )
}

/// Sparse `A · p` for an `m × n` general-constraint Jacobian stored
/// as a `GenTMatrix` (1-based triplet indices). Used to re-center
/// the second-order-correction QP's RHS on the trial point.
fn mat_vec_gen(a: &GenTMatrix, p: &[Number], m: usize) -> Vec<Number> {
    let mut out = vec![0.0; m];
    let irows = a.irows();
    let jcols = a.jcols();
    let vals = a.values();
    for k in 0..vals.len() {
        let i = (irows[k] - 1) as usize;
        let j = (jcols[k] - 1) as usize;
        out[i] += vals[k] * p[j];
    }
    out
}

/// Build the quasi-Newton curvature pair `(s, y)` for the step from the
/// previous iterate to the current one, differencing `∇L` at a **single,
/// fixed multiplier** (Nocedal-Wright §18.3):
///
/// ```text
///     s = x_k − x_{k−1}
///     y = ∇L(x_k, λ_k) − ∇L(x_{k−1}, λ_k)        ← the SAME λ_k twice
/// ```
///
/// Returns `None` on the first iteration (no previous point yet).
///
/// **Why the fixed multiplier matters (gh #361).** The previous code held
/// `∇L(x_{k−1}, λ_{k−1})` inside the Hessian object and differenced against
/// `∇L(x_k, λ_k)`, giving
///
/// ```text
///     y = (∇f_k − ∇f_{k−1}) + (J_kᵀλ_k − J_{k−1}ᵀλ_{k−1})
/// ```
///
/// For **linear** constraints `J` is constant, so that second group collapses
/// to `Aᵀ(λ_k − λ_{k−1})` — pure *multiplier* difference, carrying no
/// curvature information at all. Since the true `∇²L` equals `∇²f` there, the
/// whole term is spurious, and it feeds a divergent loop: a perturbed `B`
/// yields a worse QP multiplier, which injects a larger error into the next
/// `y`, which corrupts `B` further. On equality-constrained QPs (where `λ` is
/// sign-free and can swing hard) the multiplier was observed oscillating and
/// growing exponentially — `−13, 19, −69, 104, −145, 581, −1320, 3176, …` —
/// while `x` itself sat on the exact optimum. The solve then burned its whole
/// iteration budget and exited `Maximum_Iterations_Exceeded` *at the right
/// answer*, because the stationarity residual is computed from that garbage
/// multiplier.
///
/// Using one multiplier at both points makes the term telescope to
/// `Σλᵏᵢ(∇cᵢ(x_k) − ∇cᵢ(x_{k−1}))`, which is the genuine constraint-curvature
/// contribution: it vanishes identically for linear constraints (as it must)
/// and is retained for nonlinear ones.
fn curvature_pair(
    prev: Option<&(Vec<Number>, Vec<Number>, Triplet)>,
    iter: &SqpIterates,
    grad_f: &[Number],
    jac_c: &Triplet,
    n: usize,
) -> Option<(Vec<Number>, Vec<Number>)> {
    let (prev_x, prev_grad_f, prev_jac) = prev?;
    let s: Vec<Number> = iter
        .x
        .iter()
        .zip(prev_x.iter())
        .map(|(a, b)| a - b)
        .collect();
    // Both evaluated at the *current* multiplier `iter.lambda_g`.
    let lag_curr = compute_grad_lag(grad_f, jac_c, &iter.lambda_g, n);
    let lag_prev = compute_grad_lag(prev_grad_f, prev_jac, &iter.lambda_g, n);
    let y: Vec<Number> = lag_curr
        .iter()
        .zip(lag_prev.iter())
        .map(|(a, b)| a - b)
        .collect();
    Some((s, y))
}

/// Lagrangian gradient `∇L(x, λ_g) = ∇f(x) + J_c(x)ᵀ λ_g` at the
/// current iterate. Used by the damped-BFGS update.
fn compute_grad_lag(
    grad_f: &[Number],
    jac_c: &Triplet,
    lambda_g: &[Number],
    n: usize,
) -> Vec<Number> {
    let mut out = grad_f.to_vec();
    debug_assert_eq!(out.len(), n);
    for k in 0..jac_c.irow.len() {
        let row_i = (jac_c.irow[k] - 1) as usize;
        let col_j = (jac_c.jcol[k] - 1) as usize;
        out[col_j] += jac_c.vals[k] * lambda_g[row_i];
    }
    out
}
/// Walk `d` from `x` and return a strictly better feasible point, or `None`.
///
/// This is what turns a curvature *direction* into a refutation. gh #848
/// established the shape one layer down: a feasible point with a strictly
/// lower objective is proof needing no theory, so the search that produced
/// `d` may be as approximate as it likes — a direction it gets wrong costs
/// the evaluations below and nothing else.
///
/// The step length is the distance to the first blocking variable bound,
/// halved back until the *nonlinear* constraints are satisfied to
/// `constr_viol_tol`. That backtrack is the difference between this and the
/// QP-level version: `d` lies in the null space of the *linearized* active
/// constraints, which holds them exactly only where they are linear.
/// `nonconvex_qp`'s equality is linear, so the first trial is accepted there;
/// a curved constraint gives up its step rather than trading feasibility for
/// objective, which the SQP's own merit function would then have to undo.
#[allow(clippy::too_many_arguments)]
fn exhibit_better_point<N: SqpProblemSpec>(
    nlp: &mut N,
    x: &[Number],
    d: &[Number],
    f_curr: Number,
    xl: &[Number],
    xu: &[Number],
    bl_c: &[Number],
    bu_c: &[Number],
    constr_viol_tol: Number,
) -> Option<Vec<Number>> {
    let n = x.len();
    let dn = d.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if !(dn > 0.0) || !dn.is_finite() {
        return None;
    }
    // Both signs: curvature is even, so `-d` descends wherever `d` does, and
    // only one of them may have room before a bound.
    //
    // Every feasible trial is *scored* and the best one is returned, rather
    // than the first that clears the bar (gh#873). Two reasons, and the second
    // is a defect the first exposed:
    //
    // 1. The profile along `d` is not monotone. `f(x + αd) ≈ f(x) + ½α²·dᵀ∇²f d`
    //    only near `x`; on `nonconvex_two_escapes` the quartic term takes `f`
    //    back to `+1.8` at the wall while the interior of the same ray reaches
    //    `−0.225`. So the step length has to be searched, not guessed —
    //    which is why the halving below now runs on *either* rejection and not
    //    only on infeasibility, the mechanism that had the SQP arm certifying
    //    that fixture's documented maximum at every `neg_curv_escapes`.
    // 2. `sign` is scanned in a fixed order, so "first acceptable" made the
    //    answer depend on it. On `min −x₀² + x₁²` with `x₀ ∈ [−2, g]` the `+d`
    //    wall is worth `−g²` and the `−d` wall is worth `−4`; returning the
    //    first meant a `g` small enough to clear the bar handed back `−g²` and
    //    threw away the global minimum sitting in the other sign.
    //
    // Scoring costs nothing extra — the evaluations already happened — and the
    // bound stays the 24 halvings per sign the loop always carried, spent once
    // at convergence on a point already known to have negative curvature.
    // The feasibility a refutation is held to is the *incumbent's*, not the
    // convergence tolerance (gh#873, found by the fixture sweep).
    //
    // `constr_viol_tol` is the bar for calling a solve converged; using it here
    // let the exhibition buy objective with infeasibility. Measured on
    // `cresc4.nl` (lbfgs leg): the KKT point is feasible to 2.2e-16, and the
    // trials this accepted violated the rows by 6e-7 — nine orders worse, but
    // legal under `constr_viol_tol = 1e-6` — to gain 4.4e-7 of objective. The
    // arm then restored feasibility and returned to the same point, eight
    // times over, until `MAX_SECOND_ORDER_ESCAPES` capped it: 15 iterations
    // became 45 for an answer identical to 15 significant figures. That is
    // gh#544's shape exactly — the right answer, slowly — and it is the reason
    // CLAUDE.md requires the sweep on a trajectory change.
    //
    // A point that proves the incumbent is not a local minimum has to be at
    // least as feasible as the incumbent is. The slack is `1e-12` *relative to
    // the trial's own row magnitudes*, so it is roundoff and not a distance in
    // the units of `c`; and it is clamped at `constr_viol_tol`, so this bar is
    // never looser than the one it replaces.
    //
    // The cost is honest and worth naming: against a *curved* active
    // constraint a straight tangent probe leaves the feasible set at order
    // `α²`, so on such a model the exhibition now declines rather than
    // accepting a point that is only tolerance-feasible. That is the same
    // structural limit as gh#873 D3 — this walks a straight line and tests the
    // objective — made visible instead of paid for in iterations.
    let viol_at = |nlp: &mut N, v: &[Number]| -> (Number, Number) {
        let c = nlp.eval_c(v);
        let mut viol = 0.0_f64;
        let mut scale = 0.0_f64;
        for (j, &cj) in c.iter().enumerate() {
            viol = viol
                .max((bl_c[j] - cj).max(0.0))
                .max((cj - bu_c[j]).max(0.0));
            scale = scale.max(cj.abs());
        }
        (viol, scale)
    };
    let (viol_curr, _) = viol_at(nlp, x);

    let mut best: Option<(Number, Vec<Number>)> = None;
    for sign in [1.0_f64, -1.0] {
        let mut alpha = f64::INFINITY;
        for i in 0..n {
            let di = sign * d[i];
            if di > 1e-12 * dn && xu[i] < f64::INFINITY {
                alpha = alpha.min((xu[i] - x[i]) / di);
            }
            if di < -1e-12 * dn && xl[i] > f64::NEG_INFINITY {
                alpha = alpha.min((x[i] - xl[i]) / -di);
            }
        }
        // Unbounded in this direction is not this function's business -- the
        // unbounded-model fallback owns that -- so cap and keep going.
        if !alpha.is_finite() {
            alpha = 1.0 / dn;
        }
        for _ in 0..24 {
            if !(alpha > 0.0) {
                break;
            }
            let trial: Vec<Number> = (0..n)
                .map(|i| (x[i] + sign * alpha * d[i]).clamp(xl[i], xu[i]))
                .collect();
            let (viol, c_scale) = viol_at(nlp, &trial);
            let feas_bar = viol_curr.max(1e-12 * c_scale).min(constr_viol_tol);
            if viol <= feas_bar {
                let f_trial = nlp.eval_f(&trial);
                if f_trial.is_finite() && best.as_ref().is_none_or(|(b, _)| f_trial < *b) {
                    best = Some((f_trial, trial));
                }
            }
            alpha *= 0.5;
        }
    }

    // Strictly better by more than the objective's own scale can round, so
    // this cannot fire on noise at a genuine optimum.
    //
    // The `1.0 +` is what made that "the objective's own scale" rather than an
    // absolute `1e-10`, and it is itself scaled here (gh#873 D2, the same class
    // as gh#872). On `min k·x₀x₁ s.t. x₀ + x₁ = 2` the true improvement is
    // `64·k` (the corner `(9, −7)` against the stationary `(1, 1)`), so from
    // `k ≈ 1e-12` down the whole model lived below the additive
    // `1e-10` and every genuine refutation was rejected as noise — while the
    // reduced Hessian is `−k`, as indefinite at `k = 1e-30` as at `k = 1`.
    //
    // Lowered only: `.min(1.0)` leaves the bar exactly as it was for any
    // objective at or above unit scale, so this cannot make an existing solve
    // newly refutable.
    let (f_best, x_best) = best?;
    let f_scale = f_curr.abs().max(f_best.abs());
    let bar = 1e-10 * (f_scale.min(1.0) + f_curr.abs());
    (f_best < f_curr - bar).then_some(x_best)
}

/// The activity tolerance for one bound or row, in that quantity's own units.
///
/// A raw `constr_viol_tol` is a distance in the units of `x` (for a bound) or
/// of `c` (for a row), so using it directly is an absolute threshold on a
/// scale-dependent quantity — gh#873 D2, the same class as gh#872. Confirmed
/// as pure scaling by an exact change of variables `u = S·x` with the gap held
/// fixed in `x` units: at `S = 1e-2` a `5e-7` gap crosses `tol`, and a bound
/// with multiplier zero is frozen into the working set, closing the null space
/// and the second-order verdict with it.
///
/// `scale` is the quantity and its own bounds; infinite entries are skipped,
/// and everything left scales together under a change of units, so the ratio
/// the test really wants is invariant.
///
/// Scaled **only downwards**, as in gh#872's `psd_band`: widening the test
/// above unit scale would freeze bounds that are free today, which loses
/// refutations rather than gaining them, and every refutation still has to
/// exhibit a strictly better feasible point before it changes any answer.
fn active_tol(tol: Number, scale: &[Number]) -> Number {
    let s = scale
        .iter()
        .filter(|v| v.is_finite())
        .fold(0.0_f64, |a, v| a.max(v.abs()));
    tol * s.min(1.0)
}

/// A feasible direction of negative curvature at a *converged* first-order
/// point, or `None` when the point survives the second-order test (gh #856).
///
/// # Why this runs at convergence and not at each step
///
/// gh #848 gave standalone QP solves a second-order screen. Applying the same
/// screen to the SQP's **step** subproblem is wrong, and gh #856 has the
/// counterexample: that QP is a local model built from the *current*
/// multiplier estimates, and its second-order verdict is not the NLP's. At
/// SQP iteration 0 the multipliers are still zero, so the exact Lagrangian
/// Hessian is `∇²f`; started at HS071's own `x*` the step QP's working set
/// leaves a one-dimensional null space on which `dᵀHd = -4.05e-2`, and the
/// point is refuted — correctly for that model, wrongly for the NLP, whose
/// reduced Hessian at `x*` is positive once the multipliers have converged.
///
/// At *convergence* that objection disappears: the multipliers are the
/// converged ones, so `∇²L` is the Hessian the second-order condition is
/// actually about. This is the same distinction gh #856 draws when it says
/// "with the converged multipliers the reduced Hessian is positive" — the
/// check is meaningful exactly where it is run.
///
/// # What it computes
///
/// `Z` is an orthonormal basis for the null space of the active constraint
/// normals — equality rows, active inequality rows and active bounds — taken
/// from the eigenvectors of `BᵀB` whose eigenvalue is negligible against the
/// largest. The reduced Hessian `ZᵀHZ` is then eigen-decomposed, and a
/// sufficiently negative eigenvalue yields `d = Z v`: a direction that holds
/// every active constraint to first order and along which the objective
/// curves down.
///
/// Returns `None` when there is no negative curvature, when the active set
/// leaves no degrees of freedom, or when either eigensolve fails to converge
/// — never a direction it is unsure of, since the caller acts on it.
#[allow(clippy::too_many_arguments)]
fn negative_curvature_at_kkt_point(
    n: usize,
    m: usize,
    x: &[Number],
    hess_lag: &Triplet,
    jac_c: &Triplet,
    c_vals: &[Number],
    bl_c: &[Number],
    bu_c: &[Number],
    xl: &[Number],
    xu: &[Number],
    tol: Number,
) -> Option<Vec<Number>> {
    // Dense and `O(n³)`, so it is bounded rather than let loose on a large
    // model. It runs once, at convergence, on the way to reporting success —
    // and skipping it returns exactly today's answer, so the ceiling costs
    // coverage and never correctness. (Contrast gh #849, where the analogous
    // ceiling silently withdrew a *guarantee*.)
    const MAX_N: usize = 512;
    if n == 0 || n > MAX_N {
        return None;
    }

    // The active set. A bound or row counts as active when the iterate sits
    // on it to within the same tolerance the convergence test just used, so
    // this is the set the verdict was issued about.
    let mut rows: Vec<Vec<Number>> = Vec::new();
    for j in 0..m {
        let t = active_tol(tol, &[c_vals[j], bl_c[j], bu_c[j]]);
        let lo_active = bl_c[j] > f64::NEG_INFINITY && (c_vals[j] - bl_c[j]).abs() <= t;
        let hi_active = bu_c[j] < f64::INFINITY && (bu_c[j] - c_vals[j]).abs() <= t;
        if lo_active || hi_active {
            let mut r = vec![0.0; n];
            // `pounce_linalg` triplets are **1-based** (see `triplet.rs`).
            for k in 0..jac_c.vals.len() {
                if jac_c.irow[k] as usize == j + 1 {
                    r[jac_c.jcol[k] as usize - 1] += jac_c.vals[k];
                }
            }
            rows.push(r);
        }
    }
    for i in 0..n {
        let t = active_tol(tol, &[x[i], xl[i], xu[i]]);
        let on_lo = xl[i] > f64::NEG_INFINITY && (x[i] - xl[i]).abs() <= t;
        let on_hi = xu[i] < f64::INFINITY && (xu[i] - x[i]).abs() <= t;
        if on_lo || on_hi {
            let mut r = vec![0.0; n];
            r[i] = 1.0;
            rows.push(r);
        }
    }

    // Null space of the active normals, from the eigenvectors of `BᵀB`.
    let mut z: Vec<Number> = Vec::new();
    let n_dof;
    if rows.is_empty() {
        n_dof = n;
        z = vec![0.0; n * n];
        for i in 0..n {
            z[i * n + i] = 1.0;
        }
    } else {
        let mut btb = vec![0.0; n * n];
        for r in &rows {
            for a in 0..n {
                if r[a] == 0.0 {
                    continue;
                }
                for c in 0..n {
                    btb[c * n + a] += r[a] * r[c];
                }
            }
        }
        let (mut ev, mut evec) = (vec![0.0; n], vec![0.0; n * n]);
        if !pounce_linalg::symmetric_eigen(&btb, n, &mut ev, &mut evec) {
            return None;
        }
        let lam_max = ev.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        // Relative to `BᵀB`'s own spectral scale, with no absolute floor under
        // it (gh#873 D2). `lam_max` is already that scale, so the `.max(1.0)`
        // this carried made the rank cut absolute for small constraint
        // normals: a row scaled down by a change of units then read as part of
        // the null space, and the direction built from it does not respect the
        // constraint it came from.
        //
        // `lam_max == 0` means every active row has a zero gradient, so none
        // of them restricts anything and the whole space is free. The old
        // `.max(1.0)` produced that answer as a side effect of the floor;
        // stated here so removing the floor does not quietly change it into
        // "no degrees of freedom", which would decline the refutation.
        let cut = if lam_max > 0.0 { 1e-9 * lam_max } else { 0.0 };
        if lam_max == 0.0 {
            z.extend_from_slice(&evec[..n * n]);
            n_dof = n;
        } else {
            for (j, &lam) in ev.iter().enumerate() {
                if lam.abs() <= cut {
                    z.extend_from_slice(&evec[j * n..(j + 1) * n]);
                }
            }
            n_dof = z.len() / n;
        }
    }
    if n_dof == 0 {
        return None;
    }

    // `H Z`, then `Zᵀ (H Z)`.
    let mut hz = vec![0.0; n * n_dof];
    for k in 0..n_dof {
        let (zc, out) = (&z[k * n..(k + 1) * n], &mut hz[k * n..(k + 1) * n]);
        // Stored as one triangle, so an off-diagonal entry contributes to
        // both of its rows.
        for e in 0..hess_lag.vals.len() {
            let (r, c, v) = (
                hess_lag.irow[e] as usize - 1,
                hess_lag.jcol[e] as usize - 1,
                hess_lag.vals[e],
            );
            out[r] += v * zc[c];
            if r != c {
                out[c] += v * zc[r];
            }
        }
    }
    let mut rh = vec![0.0; n_dof * n_dof];
    for a in 0..n_dof {
        for b in 0..n_dof {
            rh[b * n_dof + a] = (0..n).map(|i| z[a * n + i] * hz[b * n + i]).sum();
        }
    }

    let (mut ev, mut evec) = (vec![0.0; n_dof], vec![0.0; n_dof * n_dof]);
    if !pounce_linalg::symmetric_eigen(&rh, n_dof, &mut ev, &mut evec) {
        return None;
    }
    // Relative to the Hessian's own scale, so this cannot fire on the
    // rounding noise of a genuinely positive-semidefinite reduced Hessian --
    // and with no absolute floor under it (gh#873 D2, gh#872's sibling). The
    // `.max(1.0)` this carried made the test absolute below unit Hessian
    // scale, and `min k·x₀x₁ s.t. x₀ + x₁ = 2` then returned the constrained
    // *maximum* `f = k` for every `k` from `1e-8` down -- a family whose
    // reduced Hessian is `−k`, i.e. as indefinite at `k = 1e-30` as at `k = 1`.
    let h_scale = hess_lag.vals.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if ev[0] >= -1e-8 * h_scale {
        return None;
    }
    // `d = Z v` with `v` the eigenvector of the most negative eigenvalue --
    // column 0 of the column-major `evec`, since the eigensolver returns them
    // in ascending order.
    let mut d = vec![0.0; n];
    for (i, di) in d.iter_mut().enumerate() {
        *di = (0..n_dof).map(|k| z[k * n + i] * evec[k]).sum();
    }
    Some(d)
}

/// Iteration budget for one step QP.
///
/// `pounce-qp`'s default is a flat 200, the interior-point budget, and the
/// wrong shape for an active-set method, which needs about one iteration per
/// working-set change: a cold first QP on a model with thousands of active
/// bounds cannot finish inside it, and reports `QpIterationLimit` on a
/// subproblem it was solving correctly. Scaled exactly as
/// `pounce_convex::active_set` scales the direct driver's (10 per variable
/// and row, floor 200, ceiling 200 000), and like there an explicitly set
/// `sqp_qp_max_iter` — anything but the default — wins.
fn step_qp_iter_budget(current: u32, n: usize, m: usize) -> u32 {
    const DEFAULT_MAX_ITER: u32 = 200;
    const PER_DIM: usize = 10;
    if current != DEFAULT_MAX_ITER {
        return current;
    }
    n.saturating_add(m)
        .saturating_mul(PER_DIM)
        .clamp(200, 200_000) as u32
}

/// Solve the step QP `data` poses, warm-started from `warm` (in the NLP's
/// dimensions) when given, and return the solution in the NLP's dimensions.
///
/// The one place the lifted low-rank form (gh#1054) meets the solver: the
/// working set goes in lifted and the solution comes out unlifted, so the
/// rest of the loop never sees the extra variables. On an unlifted
/// subproblem both maps are the identity.
///
/// Each solve is logged at debug level under `pounce::sqp::qp` with its
/// outcome, working-set changes, refactorizations, Schur updates, rank tests
/// and wall time (gh#1061). A QP that takes minutes is otherwise one opaque
/// number, and the remedies differ: many working-set changes say the working
/// set moved a lot, a refactor count says the Schur path kept resetting, and
/// a rank-test count says it spent its time pruning dependent rows.
///
/// The wall time is split as well: factorization, back-solves (with their
/// count), rank tests, and `other` — everything outside the linear solver,
/// which is KKT assembly, the ratio and drop tests and the SMW correction's
/// dense arithmetic. Counts say what the QP did; the split says which of it
/// cost the time. It was a large `other` with a modest `factor` that would
/// have named the Schur path's `O(m·nnz(A))` reset assembly at once (gh#1061:
/// 2.2 s per Schur-path refactorization against 0.57 s off it).
fn solve_step_qp(
    solver: &mut ParametricActiveSetSolver,
    data: &SqpQpData,
    warm: Option<&WorkingSet>,
    opts: &QpOptions,
    seed: Option<&IpmSeed>,
) -> Result<pounce_qp::QpSolution, pounce_qp::QpError> {
    solve_subproblem_qp(solver, data, data.as_qp(), warm, opts, seed, "step QP")
}

/// [`solve_step_qp`] for a QP built on `data` with other row bounds — the
/// second-order-correction QP, which re-centres them on the trial point —
/// logged as `kind`.
///
/// The correction QP used to call the solver directly, so it had none of
/// the step QP's handling: no seeding for a warm start that pins infeasible,
/// and no log line. gh#1061: on the reporter's model the SQP reached
/// iteration 2, rejected the full step, and the correction QP sat in the
/// elastic phase for the rest of the wall budget (585 pivots in 345 s,
/// objective frozen) without a line in the log. Solved alone, the same QP
/// is optimal from the interior point in 32 s.
///
/// `seed` is `Some` where step QPs are seeded from the interior point (see
/// [`SqpOptions::qp_ipm_cold_start_min_dim`]). There a warm start that pins
/// feasibly is also given a budget: [`WARM_ITERS_PER_SEED_ITER`] active-set
/// iterations per iteration the last seed took. A warm solve that would need
/// more is costing more than seeding it, and is seeded instead.
pub(crate) fn solve_subproblem_qp(
    solver: &mut ParametricActiveSetSolver,
    data: &SqpQpData,
    qp: pounce_qp::QpProblem<'_>,
    warm: Option<&WorkingSet>,
    opts: &QpOptions,
    seed: Option<&IpmSeed>,
    kind: &str,
) -> Result<pounce_qp::QpSolution, pounce_qp::QpError> {
    let started = std::time::Instant::now();
    let guards_before = solver.rank_guard_calls();
    let split_before = solver.linsol_time_split();
    let seeded = match (warm, seed) {
        (None, Some(seed)) => ipm_seeded_step_qp(solver, &qp, data.hess_inf, opts, seed),
        _ => None,
    };
    // gh#1061: warm step QPs on the reporter's model needed 68–350
    // working-set changes and 51–263 s each, at about one factorization per
    // change, where the seed takes 19–22 interior-point iterations and
    // 15–20 s and finishes in 0–2 changes. Counted in iterations rather than
    // seconds, so the same model takes the same path on every machine.
    let budget = seed
        .and_then(|s| s.last_iters.get())
        .map(|k| (k * WARM_ITERS_PER_SEED_ITER) as u32)
        .filter(|&b| b < opts.max_iter);
    let budget_opts = budget.map(|b| QpOptions {
        max_iter: b,
        ..opts.clone()
    });
    let sol = match (seeded, warm) {
        (Some(sol), _) => Ok(sol),
        // A warm start that pins to an infeasible point would go to the
        // ℓ1-elastic phase 1, which at this size is the cold start's cost
        // again. gh#1061's second step QP (warm from the seeded first one,
        // feasible: IPOPT 128.990447) reached the 300 s cap there after 564
        // working-set changes; seeded like a cold QP it is optimal in 12.5 s
        // with 2. The elastic path stays for when the seed declines.
        (None, Some(w)) if seed.is_some() => {
            let seed = seed.expect("guarded");
            let lifted = data.lift_working_set(w);
            let warm_opts = budget_opts.as_ref().unwrap_or(opts);
            match solver.solve_with_working_set_if_feasible(&qp, &lifted, warm_opts) {
                // Over budget: seed, or if the seed declines, carry on from
                // where the warm solve stopped with the full budget.
                Ok(Some(sol)) if budget.is_some() && sol.status == QpStatus::MaxIter => {
                    tracing::debug!(target: "pounce::sqp::qp",
                        "warm start over its budget of {} iterations (n={}, m={}, {} working-set \
                         changes in {:.3?}); seeding from the interior point",
                        warm_opts.max_iter, qp.n, qp.m, sol.stats.n_working_set_changes,
                        started.elapsed());
                    match ipm_seeded_step_qp(solver, &qp, data.hess_inf, opts, seed) {
                        Some(seeded) => Ok(seeded),
                        None => solver.solve_with_working_set(&qp, &sol.working, opts),
                    }
                }
                Ok(Some(sol)) => Ok(sol),
                Ok(None) => {
                    tracing::debug!(target: "pounce::sqp::qp",
                        "warm start pins infeasible (n={}, m={}); seeding from the interior point",
                        qp.n, qp.m);
                    match ipm_seeded_step_qp(solver, &qp, data.hess_inf, opts, seed) {
                        Some(sol) => Ok(sol),
                        None => solver.solve_with_working_set(&qp, &lifted, opts),
                    }
                }
                Err(e) => Err(e),
            }
        }
        (None, Some(w)) => solver.solve_with_working_set(&qp, &data.lift_working_set(w), opts),
        (None, None) => solver.solve(&qp, None, opts),
    };
    let elapsed = started.elapsed();
    let rank_guard_calls = solver.rank_guard_calls() - guards_before;
    let t = solver.linsol_time_split().since(&split_before);
    let other = elapsed.saturating_sub(t.factor + t.resolve + t.rank_test);
    match &sol {
        Ok(s) => tracing::debug!(target: "pounce::sqp::qp",
            "{kind} (n={}, m={}, warm={}): {} in {:.3?}, {} working-set changes, \
             {} refactors, {} Schur updates, {} rank tests, phase1={}; \
             time: {} factorizations {:.3?}, {} back-solves {:.3?}, rank tests {:.3?}, other {:.3?}",
            qp.n, qp.m, warm.is_some(), s.status, elapsed,
            s.stats.n_working_set_changes, s.stats.n_refactor, s.stats.n_schur_updates,
            rank_guard_calls, s.stats.used_phase1,
            t.n_factor, t.factor, t.n_resolve, t.resolve, t.rank_test, other),
        Err(e) => tracing::debug!(target: "pounce::sqp::qp",
            "{kind} (n={}, m={}, warm={}): error {e:?} after {:.3?}, {} rank tests; \
             time: {} factorizations {:.3?}, {} back-solves {:.3?}, rank tests {:.3?}, other {:.3?}",
            qp.n, qp.m, warm.is_some(), elapsed, rank_guard_calls,
            t.n_factor, t.factor, t.n_resolve, t.resolve, t.rank_test, other),
    }
    Ok(data.unlift_solution(sol?))
}

/// A step QP solved by the convex interior point and finished by the
/// active-set solver from the working set that solution identifies, or `None`
/// to fall back to the active-set path as before (see
/// [`SqpOptions::qp_ipm_cold_start_min_dim`]). Used for a cold QP, and for a
/// warm one whose working set pins to an infeasible point.
///
/// `None` unless every stage succeeds: the Hessian is certified PSD (an
/// interior point is not a method for an indefinite QP), the interior point
/// reaches `Optimal` (an infeasible QP goes to the active set's elastic
/// phase, whose step the SQP takes — see `elastic_step`), and the warm
/// active-set solve reaches `Optimal` too. The active-set solution is what
/// comes back, so the step, its multipliers and its working set are the
/// active-set method's own; the interior point only chose where it starts.
fn ipm_seeded_step_qp(
    solver: &mut ParametricActiveSetSolver,
    qp: &pounce_qp::QpProblem,
    hess_inf: Number,
    opts: &QpOptions,
    seed: &IpmSeed,
) -> Option<pounce_qp::QpSolution> {
    use pounce_convex::Triplet as T;
    let started = std::time::Instant::now();
    let (n, m) = (qp.n, qp.m);
    let feral = || {
        Box::new(pounce_feral::FeralSolverInterface::new())
            as Box<dyn pounce_linsol::SparseSymLinearSolverInterface>
    };
    let p_lower: Vec<T> = (0..qp.h.irows().len())
        .map(|k| {
            let (i, j) = (qp.h.irows()[k] as usize - 1, qp.h.jcols()[k] as usize - 1);
            T::new(i.max(j), i.min(j), qp.h.values()[k])
        })
        .collect();
    if qp.hessian_inertia != HessianInertia::Psd {
        // Absolute, as the certificate's contract is, at the scale of `H`.
        let tol = 1e-9 * (1.0 + hess_inf);
        if pounce_convex::certify_psd_lower_triangle(n, &p_lower, tol, feral) != Ok(true) {
            return None;
        }
    }
    let mut rows: Vec<Vec<(usize, Number)>> = vec![Vec::new(); m];
    for k in 0..qp.a.irows().len() {
        let (i, j) = (qp.a.irows()[k] as usize - 1, qp.a.jcols()[k] as usize - 1);
        rows[i].push((j, qp.a.values()[k]));
    }
    let finite_lo = |v: Number| v > NLP_LOWER_BOUND_INF;
    let finite_hi = |v: Number| v < NLP_UPPER_BOUND_INF;
    let (mut a, mut b, mut g, mut h) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (i, row) in rows.iter().enumerate() {
        if qp.bl[i] == qp.bu[i] {
            a.extend(row.iter().map(|&(j, v)| T::new(b.len(), j, v)));
            b.push(qp.bl[i]);
            continue;
        }
        if finite_hi(qp.bu[i]) {
            g.extend(row.iter().map(|&(j, v)| T::new(h.len(), j, v)));
            h.push(qp.bu[i]);
        }
        if finite_lo(qp.bl[i]) {
            g.extend(row.iter().map(|&(j, v)| T::new(h.len(), j, -v)));
            h.push(-qp.bl[i]);
        }
    }
    let prob = pounce_convex::QpProblem {
        n,
        p_lower,
        c: qp.g.to_vec(),
        a,
        b,
        g,
        h,
        lb: qp
            .xl
            .iter()
            .map(|&v| {
                if finite_lo(v) {
                    v
                } else {
                    pounce_convex::NEG_INF
                }
            })
            .collect(),
        ub: qp
            .xu
            .iter()
            .map(|&v| {
                if finite_hi(v) {
                    v
                } else {
                    pounce_convex::POS_INF
                }
            })
            .collect(),
    };
    let ipm_opts = pounce_convex::QpOptions {
        time_limit: opts.time_limit,
        max_iter: IPM_SEED_MAX_ITER,
        ..pounce_convex::QpOptions::default()
    };
    let ipm = pounce_convex::solve_qp_ipm(&prob, &ipm_opts, feral);
    let ipm_time = started.elapsed();
    if ipm.status == pounce_convex::QpStatus::Optimal {
        seed.last_iters.set(Some(ipm.iters));
    }
    if ipm.status != pounce_convex::QpStatus::Optimal {
        tracing::debug!(target: "pounce::sqp::qp",
            "IPM cold start (n={n}, m={m}): {:?} after {} iterations in {ipm_time:.3?}; \
             solving from cold instead", ipm.status, ipm.iters);
        return None;
    }
    // The working set the interior point's answer sits on: every row and
    // bound within a scaled `IPM_ACTIVE_TOL` of a side it can be active at.
    // A wrong guess either way costs the warm solve pivots, not correctness.
    let near = |v: Number, bound: Number| (v - bound).abs() <= IPM_ACTIVE_TOL * (1.0 + bound.abs());
    let ax = mat_vec_gen(qp.a, &ipm.x, m);
    let mut working = WorkingSet::cold(n, m);
    for i in 0..m {
        working.constraints[i] = if qp.bl[i] == qp.bu[i] {
            ConsStatus::Equality
        } else if finite_lo(qp.bl[i]) && near(ax[i], qp.bl[i]) {
            ConsStatus::AtLower
        } else if finite_hi(qp.bu[i]) && near(ax[i], qp.bu[i]) {
            ConsStatus::AtUpper
        } else {
            ConsStatus::Inactive
        };
    }
    for j in 0..n {
        working.bounds[j] = if qp.xl[j] == qp.xu[j] {
            BoundStatus::Fixed
        } else if finite_lo(qp.xl[j]) && near(ipm.x[j], qp.xl[j]) {
            BoundStatus::AtLower
        } else if finite_hi(qp.xu[j]) && near(ipm.x[j], qp.xu[j]) {
            BoundStatus::AtUpper
        } else {
            BoundStatus::Inactive
        };
    }
    let warm_opts = QpOptions {
        time_limit: opts.time_limit.map(|t| t.saturating_sub(started.elapsed())),
        ..opts.clone()
    };
    let sol = solver
        .solve_with_working_set(qp, &working, &warm_opts)
        .ok()?;
    tracing::debug!(target: "pounce::sqp::qp",
        "IPM cold start (n={n}, m={m}): optimal after {} iterations in {ipm_time:.3?}, \
         {} rows and bounds active; warm active-set solve {} in {:.3?} with {} \
         working-set changes", ipm.iters, working.active_count(), sol.status,
        sol.stats.time, sol.stats.n_working_set_changes);
    (sol.status == QpStatus::Optimal).then_some(sol)
}

/// Iteration cap on the seeding interior point, against its own default of
/// 200. gh#1061's step QPs took 16 and 19; a QP the interior point cannot
/// solve in a few times that goes back to the active-set path rather than
/// spending the budget here. Measured on a borderline-feasible emulation of
/// the second step QP, the uncapped interior point ran 57 iterations to a
/// 300 s limit.
const IPM_SEED_MAX_ITER: usize = 50;

/// Active-set iterations a warm-started subproblem QP may take, per
/// interior-point iteration the last seed took, before it is seeded instead
/// (see [`solve_subproblem_qp`]). An active-set iteration on gh#1061's model
/// is one factorization or one back-solve, and a working-set change about
/// two of them; an interior-point iteration, with the certificate and the
/// warm finish spread over it, about 0.8 s against 0.44 s per active-set
/// factorization. Four per seed iteration therefore lets a warm solve spend
/// about what a seed costs (80 iterations, ~40 changes, at the seed's 20)
/// before it gives up, so a warm start that is near its answer, as most are,
/// keeps it, and one that is not costs at most about twice the seed.
const WARM_ITERS_PER_SEED_ITER: usize = 4;

/// Relative activity tolerance for reading a working set off an
/// interior-point solution in [`ipm_seeded_step_qp`]. The interior point stops
/// at a complementarity of order its `tol`, so an active row sits within a few
/// orders of that and an inactive one, at a nondegenerate solution, well
/// outside it.
const IPM_ACTIVE_TOL: Number = 1e-6;

/// A subproblem QP's time limit: what is left of the solve's `deadline` at
/// this moment, or the QP's `own` limit if that is tighter.
fn qp_time_limit_now(
    deadline: Option<&pounce_common::timing::Deadline>,
    own: Option<std::time::Duration>,
) -> Option<std::time::Duration> {
    match (deadline.map(deadline_remaining), own) {
        (Some((left, _)), Some(own)) => Some(left.min(own)),
        (Some((left, _)), None) => Some(left),
        (None, own) => own,
    }
}

/// What is left of `deadline`, as the wall-clock duration a step QP can take
/// as its `time_limit`, with the budget that set it (gh#1054).
///
/// The QP reads only a wall clock, so the CPU half is converted one-for-one:
/// a busy solve accrues CPU time at about the wall rate, so its remaining CPU
/// seconds are the best wall estimate the QP can be given. The tighter of the
/// two wins.
pub(crate) fn deadline_remaining(
    deadline: &pounce_common::timing::Deadline,
) -> (std::time::Duration, pounce_common::timing::DeadlineKind) {
    use pounce_common::timing::DeadlineKind;
    let wall = deadline.remaining_wall();
    let cpu = deadline.remaining_cpu();
    let (left, kind) = if cpu < wall {
        (cpu, DeadlineKind::Cpu)
    } else {
        (wall, DeadlineKind::Wall)
    };
    // `from_secs_f64` panics on a negative, non-finite or out-of-range
    // argument, and the registered "unlimited" default is `1e20` s, past
    // `Duration`'s range. A century is unlimited for this purpose.
    const UNLIMITED_S: Number = 3.2e9;
    let left = if left.is_finite() {
        left.clamp(0.0, UNLIMITED_S)
    } else {
        UNLIMITED_S
    };
    (std::time::Duration::from_secs_f64(left), kind)
}

pub(crate) fn check_kkt(
    n: usize,
    m: usize,
    iter: &SqpIterates,
    grad_f: &[Number],
    c_vals: &[Number],
    bl_c: &[Number],
    bu_c: &[Number],
    xl: &[Number],
    xu: &[Number],
    jac_c: &crate::sqp::qp_assembly::Triplet,
) -> KktError {
    // Constraint violation: max(0, bl - c, c - bu) on every row,
    // plus bound violation on every variable.
    //
    // `NaN` propagates rather than reducing away (gh #876). `f64::max`
    // ignores `NaN`, so `(bl - c).max(0.0)` on a non-finite `c` is `0.0` and
    // a diverged iterate scores as perfectly feasible. The same reduction on
    // the stationarity rows below is the site the issue reports; both are
    // fixed, because either one alone still lets `check_kkt` return a clean
    // `KktError` from garbage.
    let mut viol = 0.0_f64;
    let mut nonfinite = false;
    for i in 0..m {
        let lo = if bl_c[i] > NLP_LOWER_BOUND_INF {
            (bl_c[i] - c_vals[i]).max(0.0)
        } else {
            0.0
        };
        let hi = if bu_c[i] < NLP_UPPER_BOUND_INF {
            (c_vals[i] - bu_c[i]).max(0.0)
        } else {
            0.0
        };
        nonfinite |= !c_vals[i].is_finite();
        viol = viol.max(lo).max(hi);
    }
    for i in 0..n {
        nonfinite |= !iter.x[i].is_finite();
        let lo = if xl[i] > NLP_LOWER_BOUND_INF {
            (xl[i] - iter.x[i]).max(0.0)
        } else {
            0.0
        };
        let hi = if xu[i] < NLP_UPPER_BOUND_INF {
            (iter.x[i] - xu[i]).max(0.0)
        } else {
            0.0
        };
        viol = viol.max(lo).max(hi);
    }

    // Stationarity: ∇f + Jᵀ λ_g − λ_x. pounce-qp's KKT is
    // `Hx + Aᵀλ_qp + (lower-bound multiplier) e_i − (upper-bound
    // multiplier) e_i = -g`. Since `λ_x = z_l − z_u` packs the
    // bound-multiplier sign, the variable-bound term enters the
    // stationarity check with a negative sign — i.e. at the
    // optimum `∇f + Jᵀ λ_g = λ_x`.
    let mut stat = vec![0.0; n];
    for (s, &g) in stat.iter_mut().zip(grad_f.iter()) {
        *s = g;
    }
    // Add Jᵀ λ_g
    for k in 0..jac_c.irow.len() {
        let i = (jac_c.irow[k] - 1) as usize; // 0-based row in c
        let j = (jac_c.jcol[k] - 1) as usize; // 0-based col in x
        stat[j] += jac_c.vals[k] * iter.lambda_g[i];
    }
    // Subtract λ_x
    for (s, &lx) in stat.iter_mut().zip(iter.lambda_x.iter()) {
        *s -= lx;
    }
    let stat_max = crate::sqp::line_search::inf_norm(&stat);

    // An iterate or a constraint value that is not finite makes every residual
    // computed from it meaningless, whichever way the arithmetic happens to
    // reduce: `+inf - inf` is `NaN`, but `(bl - inf).max(0.0)` is a tidy `0.0`
    // and `inf.abs()` is a large number that at least fails `<= tol`. Reporting
    // `NaN` for both is the honest answer and the one the caller's `<= tol`
    // gates already handle correctly.
    if nonfinite {
        return KktError {
            stationarity: Number::NAN,
            constr_viol: Number::NAN,
        };
    }

    KktError {
        stationarity: stat_max,
        constr_viol: viol,
    }
}

#[cfg(test)]
mod kkt_nan_tests {
    //! gh #876 — `check_kkt` must not launder a non-finite iterate into a
    //! clean `KktError`.
    //!
    //! The caller's convergence gate is
    //! `kkt.stationarity <= stationarity_tol && kkt.constr_viol <= constr_viol_tol`,
    //! so anything `check_kkt` reports as `0.0` is reported as converged. Both
    //! of its reductions used to swallow `NaN`:
    //!
    //! * `stat.iter().map(|s| s.abs()).fold(0.0_f64, f64::max)` — `f64::max`
    //!   is defined to *ignore* `NaN`, so an all-`NaN` stationarity vector
    //!   reduced to `0.0`. This is the site gh #876 reports, and the third
    //!   instance of the same shape in this workspace: gh #222 fixed it in
    //!   `pounce-convex`, gh #845 in `pounce-sensitivity`.
    //! * `viol.max(lo).max(hi)` where `lo = (bl - c).max(0.0)` — the same
    //!   definition one layer earlier, so a `NaN` constraint value scored as
    //!   *perfectly feasible* before the outer `max` ever saw it. The issue
    //!   does not name this one; fixing only the reported site would still
    //!   have left `constr_viol` reading `0.0` on a diverged iterate.
    //!
    //! ## Which branch each test reaches
    //!
    //! Per this repo's gh #756 lesson, a guard is only evidence about the
    //! branch its fixture reaches, so the four tests below are chosen to land
    //! in four different places: a `NaN` arriving through `c_vals`, one
    //! arriving through `x`, one arriving through the *duals* (where `x` and
    //! `c` are both finite and only the stationarity row is poisoned — the
    //! `inf_norm` fix, not the `nonfinite` flag), and an ordinary finite
    //! iterate that must be completely unaffected.
    //!
    //! ## Mutation table
    //!
    //! | revert | red |
    //! |---|---|
    //! | `inf_norm` back to `fold(0.0, f64::max)` | `a_nan_dual_poisons_only_stationarity`, `inf_norm_tests::a_nan_entry_propagates_rather_than_reducing_away` |
    //! | drop the `nonfinite` flag | `a_nan_constraint_value_is_not_perfect_feasibility`, `a_nan_iterate_is_not_perfect_feasibility` |
    //! | both | all four of the above |
    //!
    //! Measured, both directions: each mutation turns exactly the listed
    //! tests red and leaves the others — including
    //! `a_finite_iterate_is_measured_exactly_as_before` — green. Neither half
    //! subsumes the other, which is why both are here.

    use super::*;
    use crate::sqp::qp_assembly::Triplet;
    use pounce_common::types::Index;

    /// `min x0` s.t. one row `c(x) = x0`, no bounds — the smallest shape that
    /// exercises every accumulation in `check_kkt`.
    fn setup(x: Vec<Number>, c: Vec<Number>, lam_g: Vec<Number>) -> KktError {
        let n = x.len();
        let m = c.len();
        let iter = SqpIterates {
            x,
            lambda_g: lam_g,
            lambda_x: vec![0.0; n],
            working: None,
        };
        let jac = Triplet {
            n_rows: m,
            n_cols: n,
            irow: (1..=m as i32).map(|i| i as Index).collect(),
            jcol: vec![1 as Index; m],
            vals: vec![1.0; m],
        };
        check_kkt(
            n,
            m,
            &iter,
            &vec![1.0; n],
            &c,
            &vec![0.0; m],
            &vec![0.0; m],
            &vec![NLP_LOWER_BOUND_INF; n],
            &vec![NLP_UPPER_BOUND_INF; n],
            &jac,
        )
    }

    #[test]
    fn a_finite_iterate_is_measured_exactly_as_before() {
        // x0 = 2 against the equality row c = x0 = 0: violation 2, and
        // stationarity ∇f + Jᵀλ = 1 + 1·(-1) = 0.
        let k = setup(vec![2.0], vec![2.0], vec![-1.0]);
        assert_eq!(k.constr_viol, 2.0);
        assert_eq!(k.stationarity, 0.0);
    }

    #[test]
    fn a_nan_constraint_value_is_not_perfect_feasibility() {
        let k = setup(vec![0.0], vec![Number::NAN], vec![-1.0]);
        assert!(
            k.constr_viol.is_nan(),
            "a NaN constraint value reported constr_viol = {}, which clears \
             every tolerance the caller compares it against",
            k.constr_viol
        );
        assert!(k.stationarity.is_nan());
    }

    #[test]
    fn a_nan_iterate_is_not_perfect_feasibility() {
        let k = setup(vec![Number::NAN], vec![0.0], vec![-1.0]);
        assert!(k.constr_viol.is_nan());
        assert!(k.stationarity.is_nan());
    }

    /// The dual-side branch: `x` and `c` are both finite, so the `nonfinite`
    /// flag stays false and nothing but `inf_norm` stands between a `NaN`
    /// multiplier and a stationarity residual of `0.0`. This is the test that
    /// the `nonfinite` flag alone does **not** cover.
    #[test]
    fn a_nan_dual_poisons_only_stationarity() {
        let k = setup(vec![0.0], vec![0.0], vec![Number::NAN]);
        assert!(
            k.stationarity.is_nan(),
            "a NaN multiplier reduced to stationarity = {}",
            k.stationarity
        );
        // Feasibility genuinely holds at this x, and saying so is correct —
        // the flag is about the *primal* iterate, not about the duals.
        assert_eq!(k.constr_viol, 0.0);
    }
}

#[cfg(test)]
mod inf_norm_tests {
    use crate::sqp::line_search::inf_norm;

    #[test]
    fn a_nan_entry_propagates_rather_than_reducing_away() {
        assert!(inf_norm(&[1.0, f64::NAN, 2.0]).is_nan());
        assert!(inf_norm(&[f64::NAN]).is_nan());
    }

    #[test]
    fn finite_vectors_are_unchanged() {
        assert_eq!(inf_norm(&[]), 0.0);
        assert_eq!(inf_norm(&[-3.0, 1.0, 2.0]), 3.0);
        assert_eq!(inf_norm(&[f64::INFINITY, 1.0]), f64::INFINITY);
    }
}
