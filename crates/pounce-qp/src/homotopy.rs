//! §4.2 parametric homotopy — the qpOASES-lineage path this crate is named for.
//!
//! # Why this exists
//!
//! `ParametricActiveSetSolver` advertises "true parametric warm starting" and
//! cites the qpOASES lineage, but `solve_parametric` was a stub that discarded
//! its arguments and cold-solved. The homotopy was never implemented, and its
//! absence is *structural* rather than a missing convenience:
//!
//! The design note (`docs/src/active-set-sqp-warm-start.md` §4.3) specifies that
//! l1-elastic phase-1 works by driving the elastic slacks to zero **as the
//! homotopy proceeds**. With no homotopy, phase-1 degenerated into solving a
//! standalone, maximally-degenerate QP from cold — the hardest case for an
//! active-set method — which is exactly where it stalls and fails to terminate
//! on the Maros-Mészáros set.
//!
//! # The path
//!
//! §4.2 specifies tracing `(1-t)·QP₀ + t·QP₁` for `t ∈ [0,1]`, jumping the
//! working set wherever a multiplier reaches zero or a constraint reaches its
//! bound. Along a segment with a fixed working set `W` the KKT system
//!
//! ```text
//!   [ H   A_Wᵀ ] [ x ]   [ -g    ]
//!   [ A_W   0  ] [ λ ] = [  b_W(t) ]
//! ```
//!
//! is *affine in `t`*, so `(x(t), λ(t))` moves linearly and the next event is
//! found by two ratio tests in `t` rather than by a line search in step space.
//!
//! # Choosing `QP₀` (the part the textbook version gets wrong here)
//!
//! The canonical cold start takes `W₀ = ∅`, which makes `x(t) = −H⁻¹g(t)` and
//! therefore **requires `H` nonsingular**. Most of the Maros-Mészáros set is
//! LP-like with singular or zero `H`, where an empty working set leaves a
//! null-space direction and the KKT is singular. Repairing that needs `n` active
//! constraints — i.e. a vertex — which is the phase-1 problem again. Circular.
//!
//! This module sidesteps it: `QP₀` is the **box-only relaxation** — the target
//! QP with every general row dropped. That is solvable on an existing, tested
//! fast path ([`super::ParametricActiveSetSolver::solve_box_constrained`]), and
//! its solution comes with a working set of active bounds that makes the reduced
//! Hessian nonsingular whenever the box does. Only the **row bounds** are then
//! homotopied in, from a relaxation that `x₀` strictly satisfies to the target;
//! `H` and `g` are held fixed, so the `-g` block above never moves and the
//! direction solve has a zero primal right-hand side.
//!
//! The consequence worth stating plainly: there is **no phase-1**. `x(t)` starts
//! feasible for the `t`-problem, and the primal ratio test is what keeps it that
//! way, so feasibility is never *searched* for.
//!
//! It can still be *lost*, and this module used to claim otherwise — that
//! feasibility held "at every point on the path by construction". That claim was
//! wrong and it was load-bearing: `pounce-convex`'s driver switched off its
//! simplex phase-1 seed on the strength of it, leaving nothing to fall back on
//! when the path's prediction turned out unusable (#413). The ratio test only
//! ever *prevents* a violation and cannot repair one, so a row it fails to cap
//! stays violated for the rest of the path and drifts further out; see the
//! path-feasibility report in [`ParametricActiveSetSolver::trace_path`] for the
//! two measured ways that happens.
//!
//! The path is still only a **predictor** — the corrector re-derives the primal
//! from the working set and often recovers from a path that drifted — so losing
//! feasibility degrades the prediction rather than invalidating it, and the path
//! reports the loss instead of abandoning itself over it.
//!
//! # References
//!
//! - Ferreau, Kirches, Potschka, Bock, Diehl, "qpOASES: a parametric active-set
//!   algorithm for quadratic programming", *Math. Prog. Comp.* **6** (2014) —
//!   the dense reference algorithm and the homotopy's ratio tests.
//! - Kirches, *Fast Numerical Methods for Mixed-Integer Nonlinear
//!   Model-Predictive Control* (2011), Ch. 5–7 — the sparse Schur extension.

use crate::error::{QpError, QpStatus};
use crate::kkt::{a_times_x, assemble_active_set_kkt};
use crate::options::QpOptions;
use crate::problem::{QpProblem, QpSolution, QpStats};
use crate::solver::ParametricActiveSetSolver;
use crate::solver::QpSolver as _;
use crate::working_set::{BoundStatus, ConsStatus, WorkingSet};
use pounce_common::Number;
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use std::time::Instant;

/// How far the relaxed `t = 0` row bounds sit outside `A x₀`, relative to the
/// row's own scale. Strictly positive so every row starts *inactive* with real
/// slack — a row that started exactly at its bound would be degenerate at `t=0`
/// and the first ratio test would see a zero-length step.
const RELAX_MARGIN: Number = 1.0;

/// `t` is clamped into `[0, 1]`; an event closer than this to the current `t` is
/// treated as coincident (a degenerate tie) rather than as forward progress, so
/// the loop cannot spin on a zero-length advance.
const T_EPS: Number = 1e-12;

/// Outcome of the two ratio tests: what happens first as `t` increases.
#[derive(Debug, Clone, Copy)]
enum Event {
    /// Inactive row `i` reaches its lower bound.
    AddRowLower(usize),
    /// Inactive row `i` reaches its upper bound.
    AddRowUpper(usize),
    /// Active row `i`'s multiplier reaches zero and must leave.
    DropRow(usize),
    /// Active bound on variable `j` has its multiplier reach zero.
    DropBound(usize),
}

/// Primal regularization `δ` for the path, derived from the problem's own scale.
///
/// Needed because the path starts from the box relaxation, and that relaxation is
/// **unbounded** whenever `H` has no curvature in a box-unbounded direction —
/// which is most LP-like instances (`QAFIRO` returns `obj = -inf`). Running the
/// path on `H + δI` bounds it.
///
/// This is sound *specifically because the path is only a predictor*: the working
/// set it discovers is handed to a corrector that solves the true QP, so `δ`
/// never enters the reported answer, only the prediction of which constraints
/// end up active.
///
/// `δ` is derived, not guessed. With `H = 0` the box relaxation's solution is
/// `x₀ = clamp(−g/δ, box)`, so `‖g‖∞ / X` — for `X` a representative variable
/// magnitude (median finite box width, else 1) — is the `δ` that places `‖x₀‖`
/// at roughly `X`. That is the *scale* reference; the regularization itself is a
/// small relative fraction of it, [`DELTA_REL`].
///
/// Sizing `δ` at the box scale was the original choice and it was wrong: at
/// `O(1)` relative size, `H + δI` is not a perturbation of the problem but a
/// different problem, and the path faithfully predicts *its* active set. On
/// `QRECIPE` (published optimum −266.616) that produced −104.83; at `1e-6`
/// relative it produces the exact optimum. Measured, both directions:
///
/// | δ relative size | QRECIPE result | path | time |
/// |---|---|---|---|
/// | `1` (box scale) | −104.83, wrong | 99 changes | 34.8 s |
/// | `1e-6` | **−266.616, exact** | 125 changes | **0.053 s** |
///
/// The small `δ` is *both* more accurate and far faster: a bad prediction costs
/// the corrector far more than the slightly longer path costs the predictor. The
/// worry that a small `δ` would blow up `‖x₀‖` and so the path length is real but
/// mild — 99 → 125 changes here, 29 → 34 on `QAFIRO`.
///
/// Returns `None` when `g` is zero — there is nothing to scale against, and with
/// `H = 0` and `g = 0` every feasible point is optimal anyway.
fn path_regularization_delta(qp: &QpProblem<'_>) -> Option<Number> {
    /// `δ` as a fraction of the problem's own objective scale. Small enough that
    /// `H + δI` is a *perturbation* rather than a different problem, large enough
    /// to bound the box relaxation in double precision. See the table above.
    const DELTA_REL: Number = 1e-6;

    let g_inf = qp.g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if !(g_inf > 0.0) || !g_inf.is_finite() {
        return None;
    }
    let mut widths: Vec<Number> = (0..qp.n)
        .filter_map(|i| {
            let (l, u) = (qp.xl[i], qp.xu[i]);
            (l > NLP_LOWER_BOUND_INF && u < NLP_UPPER_BOUND_INF).then(|| (u - l).abs())
        })
        .filter(|w| w.is_finite() && *w > 0.0)
        .collect();
    let x_scale = if widths.is_empty() {
        1.0
    } else {
        widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        widths[widths.len() / 2]
    };
    let delta = DELTA_REL * g_inf / x_scale.max(1e-12);
    delta.is_finite().then_some(delta.clamp(1e-12, 1e12))
}

/// `H + δI`, as a fresh symmetric triplet matrix.
///
/// `H` is stored lower-triangle 1-based with each pair listed once, so `δ` is
/// added to an existing diagonal entry where one is present and appended
/// otherwise — appending unconditionally would double-count a diagonal that `H`
/// already carries.
fn regularized_hessian(qp: &QpProblem<'_>, delta: Number) -> SymTMatrix {
    let (irows, jcols, vals) = (qp.h.irows(), qp.h.jcols(), qp.h.values());
    let mut ir: Vec<i32> = irows.to_vec();
    let mut jc: Vec<i32> = jcols.to_vec();
    let mut vl: Vec<Number> = vals.to_vec();

    let mut has_diag = vec![false; qp.n];
    for k in 0..ir.len() {
        if ir[k] == jc[k] {
            let idx = (ir[k] - 1) as usize;
            if idx < qp.n {
                vl[k] += delta;
                has_diag[idx] = true;
            }
        }
    }
    for (i, seen) in has_diag.iter().enumerate() {
        if !seen {
            ir.push((i + 1) as i32);
            jc.push((i + 1) as i32);
            vl.push(delta);
        }
    }

    let space = SymTMatrixSpace::new(qp.n as i32, ir, jc);
    let mut h = SymTMatrix::new(space);
    h.set_values(&vl);
    h
}

/// How far outside its `t`-interpolated bound a row may sit before the path is
/// reported as having lost feasibility, relative to the row's own magnitude.
///
/// Has to straddle two regimes measured on the same QSHARE2B path: benign
/// accumulated round-off in the direction solves, which ran at `2e-8` for a
/// hundred steps and never grew, versus a genuine uncapped crossing, which
/// entered at `8e-2` and compounded to `22` by `t = 1`. Two orders of magnitude
/// above the drift and six below the break.
const PATH_FEAS_TOL_REL: Number = 1e-6;

/// The worst `t`-problem row violation at `x`, or `None` when every row is
/// within [`PATH_FEAS_TOL_REL`] of its interpolated bound.
///
/// This is the invariant the module header used to assert held by construction;
/// see the report in [`ParametricActiveSetSolver::trace_path`] for why it does
/// not. Row bounds interpolate `bl0 -> qp.bl` and `bu0 -> qp.bu` in `t`; variable
/// bounds do not move along this path, so they are not checked here.
fn worst_path_violation(
    qp: &QpProblem<'_>,
    x: &[Number],
    bl0: &[Number],
    bu0: &[Number],
    t: Number,
    m: usize,
) -> Option<(usize, Number)> {
    let ax = a_times_x(qp.a, x, m);
    let mut worst: Option<(usize, Number)> = None;
    for i in 0..m {
        let blt = bl0[i] + t * bound_rate(bl0[i], qp.bl[i], true);
        let but = bu0[i] + t * bound_rate(bu0[i], qp.bu[i], false);
        let v = (blt - ax[i]).max(ax[i] - but);
        if v > PATH_FEAS_TOL_REL * (1.0 + ax[i].abs()) && worst.is_none_or(|(_, prev)| v > prev) {
            worst = Some((i, v));
        }
    }
    worst
}

/// `rows` reordered so the entries of `first` lead, in `first`'s own order.
///
/// [`crate::solver::independent_active_subset`] is a greedy scan that keeps the
/// earliest independent rows it meets, so presenting a row earlier is what makes
/// it survive the selection. That is the whole mechanism behind the exchange
/// pivot in [`ParametricActiveSetSolver::trace_path`]: promoting a row costs one
/// reordering, and the scan works out for itself which row must be evicted to
/// make space.
fn promote_first(rows: &[usize], first: &[usize]) -> Vec<usize> {
    if first.is_empty() {
        return rows.to_vec();
    }
    let mut out: Vec<usize> = Vec::with_capacity(rows.len());
    out.extend(first.iter().copied().filter(|i| rows.contains(i)));
    out.extend(rows.iter().copied().filter(|i| !first.contains(i)));
    out
}

/// Rate of change of a row bound per unit `t` (0 when the bound is infinite).
fn bound_rate(relaxed: Number, target: Number, is_lower: bool) -> Number {
    let infinite = if is_lower {
        target <= NLP_LOWER_BOUND_INF
    } else {
        target >= NLP_UPPER_BOUND_INF
    };
    if infinite { 0.0 } else { target - relaxed }
}

impl ParametricActiveSetSolver {
    /// Solve `qp` from cold by tracing the row-bound homotopy described in this
    /// module's docs.
    ///
    /// `Ok(None)` is a signal for the caller to fall back to the conventional
    /// cold path — not a verdict about `qp`.
    pub(crate) fn solve_homotopy(
        &mut self,
        qp: &QpProblem<'_>,
        warm: Option<(&QpProblem<'_>, &QpSolution)>,
        opts: &QpOptions,
    ) -> Result<Option<QpSolution>, QpError> {
        let started = Instant::now();
        let n = qp.n;
        let m = qp.m;
        if m == 0 {
            // No rows to bring in: the box relaxation *is* the problem, and the
            // existing fast path already handles it.
            return Ok(None);
        }

        // ---- Warm start: QP₀ *is* the previous problem ----
        //
        // The prior solution is optimal for the prior QP, so the path starts on
        // the solution manifold with no `QP₀` to construct and no box relaxation
        // to bound — the whole reason the warm case is easier than the cold one.
        // `g` moves too here (the cold path holds it fixed), which is the `dg`
        // term in the direction solve below.
        if let Some((prev, sol_prev)) = warm {
            let mut x = sol_prev.x.clone();
            x.resize(n, 0.0);
            let mut working = sol_prev.working.clone();
            working.constraints.resize(m, ConsStatus::Inactive);
            working.bounds.resize(n, BoundStatus::Inactive);
            let mut lambda_g = sol_prev.lambda_g.clone();
            lambda_g.resize(m, 0.0);
            let mut lambda_x = sol_prev.lambda_x.clone();
            lambda_x.resize(n, 0.0);
            let bl0 = prev.bl.to_vec();
            let bu0 = prev.bu.to_vec();
            let dg: Vec<Number> = (0..n).map(|j| qp.g[j] - prev.g[j]).collect();
            return self.trace_path(
                qp, qp.h, x, working, lambda_g, lambda_x, bl0, bu0, dg, opts, started,
            );
        }

        // ---- QP₀: the box-only relaxation ----
        //
        // `m = 0` demands a genuinely 0-row Jacobian: `QpProblem::validate`
        // cross-checks `A`'s row count against `m`, so handing it the target's
        // `A` with `m = 0` is rejected outright.
        let empty_a = GenTMatrix::new(GenTMatrixSpace::new(0, n as i32, Vec::new(), Vec::new()));
        let trace = std::env::var("POUNCE_HOMOTOPY_DEBUG").is_ok();

        // Built inline rather than by a closure: a closure returning
        // `QpProblem<'_>` cannot tie the borrow of `h` to the returned value's
        // lifetime, so it fails to compile.
        macro_rules! box_qp {
            ($h:expr) => {
                QpProblem {
                    n,
                    m: 0,
                    h: $h,
                    g: qp.g,
                    a: &empty_a,
                    bl: &[],
                    bu: &[],
                    xl: qp.xl,
                    xu: qp.xu,
                    hessian_inertia: qp.hessian_inertia,
                }
            };
        }

        // Try the true `H` first, and regularize only if that fails. An
        // unbounded box relaxation means `H` has no curvature in a
        // box-unbounded direction; the *target* may still be bounded (a row
        // constraint can cut the ray off), so it is not an unboundedness verdict
        // — just a statement that the path cannot start from here.
        //
        // Regularizing only on failure keeps the problems that already work
        // bit-identical: `HS21` and `QPTEST` have positive-definite `H` and never
        // reach the retry.
        let mut h_reg_holder: Option<SymTMatrix> = None;
        let box_sol = {
            let first = self.solve(&box_qp!(qp.h), None, opts);
            if trace {
                match &first {
                    Ok(s) => eprintln!("[hom] box relaxation: {:?} obj={:.6e}", s.status, s.obj),
                    Err(e) => eprintln!("[hom] box relaxation ERROR: {e}"),
                }
            }
            match first {
                Ok(s) if s.status == QpStatus::Optimal => s,
                _ => {
                    let Some(delta) = path_regularization_delta(qp) else {
                        return Ok(None);
                    };
                    let h_reg = regularized_hessian(qp, delta);
                    let retry = self.solve(&box_qp!(&h_reg), None, opts);
                    if trace {
                        match &retry {
                            Ok(s) => eprintln!(
                                "[hom] box relaxation (delta={delta:.3e}): {:?} obj={:.6e}",
                                s.status, s.obj
                            ),
                            Err(e) => eprintln!("[hom] box relaxation (regularized) ERROR: {e}"),
                        }
                    }
                    match retry {
                        Ok(s) if s.status == QpStatus::Optimal => {
                            h_reg_holder = Some(h_reg);
                            s
                        }
                        _ => return Ok(None),
                    }
                }
            }
        };

        // Everything on the path is traced against this Hessian; the corrector at
        // the end uses the caller's `qp` and therefore the true `H`.
        let path_h: &SymTMatrix = h_reg_holder.as_ref().unwrap_or(qp.h);
        let path_qp = QpProblem {
            n,
            m,
            h: path_h,
            g: qp.g,
            a: qp.a,
            bl: qp.bl,
            bu: qp.bu,
            xl: qp.xl,
            xu: qp.xu,
            hessian_inertia: qp.hessian_inertia,
        };

        let mut x = box_sol.x.clone();
        let mut working = WorkingSet::cold(n, m);
        for (i, st) in working.bounds.iter_mut().enumerate() {
            *st = box_sol.working.bounds[i];
        }
        // Every row starts inactive, with genuine slack.
        let ax0 = a_times_x(qp.a, &x, m);
        let mut bl0 = vec![0.0; m];
        let mut bu0 = vec![0.0; m];
        for i in 0..m {
            let scale = RELAX_MARGIN * (1.0 + ax0[i].abs());
            // Relax outward from `A x₀` far enough that `x₀` is strictly
            // interior; where the target is already looser, keep the target.
            bl0[i] = if qp.bl[i] <= NLP_LOWER_BOUND_INF {
                qp.bl[i]
            } else {
                (ax0[i] - scale).min(qp.bl[i])
            };
            bu0[i] = if qp.bu[i] >= NLP_UPPER_BOUND_INF {
                qp.bu[i]
            } else {
                (ax0[i] + scale).max(qp.bu[i])
            };
        }

        let mut lambda_g = vec![0.0; m];
        let mut lambda_x = box_sol.lambda_x.clone();
        lambda_x.resize(n, 0.0);

        // Cold arm: hand the relaxed start to the shared tracer with a zero
        // `dg` — the cold path holds `g` fixed and moves only the row bounds.
        let dg = vec![0.0; n];
        self.trace_path(
            qp, path_h, x, working, lambda_g, lambda_x, bl0, bu0, dg, opts, started,
        )
    }

    /// Trace the homotopy from a start state to `t = 1`, returning the corrected
    /// solution, or `None` when the path cannot be completed.
    ///
    /// Shared by the cold and warm entry points: both differ only in where the
    /// path *starts* (a box relaxation with widened row bounds, versus the
    /// previous problem and its solution) and in whether `g` moves. Everything
    /// after that — the two ratio tests in `t`, the working-set jumps, the rank
    /// repair, and the final corrector — is identical.
    #[allow(clippy::too_many_arguments)]
    fn trace_path(
        &mut self,
        qp: &QpProblem<'_>,
        path_h: &SymTMatrix,
        mut x: Vec<Number>,
        mut working: WorkingSet,
        mut lambda_g: Vec<Number>,
        mut lambda_x: Vec<Number>,
        bl0: Vec<Number>,
        bu0: Vec<Number>,
        dg: Vec<Number>,
        opts: &QpOptions,
        started: Instant,
    ) -> Result<Option<QpSolution>, QpError> {
        let n = qp.n;
        let m = qp.m;
        let trace = std::env::var("POUNCE_HOMOTOPY_DEBUG").is_ok();
        let path_qp = QpProblem {
            n,
            m,
            h: path_h,
            g: qp.g,
            a: qp.a,
            bl: qp.bl,
            bu: qp.bu,
            xl: qp.xl,
            xu: qp.xu,
            hessian_inertia: qp.hessian_inertia,
        };
        let mut t: Number = 0.0;
        let mut n_changes: u32 = 0;
        let mut n_refactor: u32 = 0;
        // Rank repairs allowed on this path. Each strictly shrinks the KKT
        // working set so the retry loop terminates on its own; the cap bounds
        // the pathological case.
        let mut rank_repairs: u32 = (n + m).min(1000) as u32;

        // ---- Active set vs KKT working set ----
        //
        // These are *different sets* at a degenerate point, and conflating them
        // is what broke this path (#413).
        //
        // `working` is the **active set**: the rows and bounds sitting on their
        // boundary at the current `t`. At a degenerate point more of them can be
        // binding than there are independent directions, and the active-set KKT
        // is then singular — no H-block shift repairs a rank-deficient
        // *constraint* block. So the KKT is assembled from a maximal linearly
        // independent subset, the **working set**, tracked here.
        //
        // The previous code expressed the prune by marking the surplus rows
        // `Inactive`, which threw away the fact that they are still *on their
        // bound*, and then had to hide them from the primal ratio test (the
        // `tabu_cons` list) to stop the repair and the ratio test from fighting.
        // A hidden row is not capped by the ratio test, so the step walked
        // straight through it, and since the ratio test can only prevent a
        // violation and never repair one (`dt < 0` is discarded), that row
        // stayed violated for the rest of the path — 10 of the 14 crossings
        // measured on QSHARE2B.
        //
        // Keeping the distinction lets a pruned row stay in the active set,
        // where the step logic can still see it and hold it.
        let mut in_kkt_c = vec![true; m];
        let mut in_kkt_b = vec![true; n];

        // Rows the exchange pivot has promoted into the working set at the
        // current `t`, most recent first, and the budget for doing so. Cleared
        // whenever `t` actually advances, because the dependency that forced the
        // exchange is a property of the current vertex.
        let mut promoted: Vec<usize> = Vec::new();
        let mut exchanges: u32 = (n + m).min(500) as u32;

        // Each iteration either advances `t` or changes the working set, and the
        // budget bounds the total.
        for _step in 0..opts.max_iter {
            if t >= 1.0 - T_EPS {
                break;
            }
            if trace && _step % 50 == 0 {
                eprintln!("[hom] step={_step} t={t:.6e}");
            }

            // The active set: everything currently on its boundary.
            let active_cons: Vec<usize> = (0..m)
                .filter(|&i| working.constraints[i].is_active())
                .collect();
            let active_bounds: Vec<usize> =
                (0..n).filter(|&i| working.bounds[i].is_active()).collect();
            // The working set: the independent subset the KKT is built from.
            // Equal to the active set except at a degenerate vertex.
            let kkt_cons: Vec<usize> = active_cons
                .iter()
                .copied()
                .filter(|&i| in_kkt_c[i])
                .collect();
            let kkt_bounds: Vec<usize> = active_bounds
                .iter()
                .copied()
                .filter(|&j| in_kkt_b[j])
                .collect();
            let (k_c, k_b) = (kkt_cons.len(), kkt_bounds.len());

            // ---- Direction: d/dt of (x, λ) along this segment ----
            //
            // `H` is fixed along the path; `g` moves only on the warm path (see
            // `dg` below). The active rows' bounds advance at `bound_rate`.
            let kkt = assemble_active_set_kkt(&path_qp, &kkt_cons, &kkt_bounds);
            let mut rhs = vec![0.0; n + k_c + k_b];
            // Stationarity block: `H x(t) + g(t) + A_Wᵀ λ(t) = 0` differentiated
            // in `t` gives `H dx + dg + A_Wᵀ dλ = 0`, so the primal right-hand
            // side is `−dg`. Zero on the cold path, where `g` is held fixed and
            // only the row bounds move; non-zero on the warm path, where the
            // objective moves from the previous problem's to the new one's.
            for (j, r) in rhs[..n].iter_mut().enumerate() {
                *r = -dg[j];
            }
            for (slot, &i) in kkt_cons.iter().enumerate() {
                let is_lower = matches!(working.constraints[i], ConsStatus::AtLower);
                rhs[n + slot] = match working.constraints[i] {
                    ConsStatus::Equality => bound_rate(bu0[i], qp.bu[i], false),
                    _ => bound_rate(
                        if is_lower { bl0[i] } else { bu0[i] },
                        if is_lower { qp.bl[i] } else { qp.bu[i] },
                        is_lower,
                    ),
                };
            }
            // Variable bounds do not move along this path, so their rows are 0.
            match self.factorize_with_inertia_control(kkt, &mut rhs, (k_c + k_b) as i32, n, opts) {
                Ok(_) => {}
                // A rank-deficient active set on the path is the same situation
                // `solve_general` handles by pruning; rather than duplicate that
                // logic here, hand the problem back to the conventional path.
                // A rank-deficient active set: at a degenerate point more rows
                // are binding than there are variables, and the surplus is
                // linearly dependent. No H-block shift repairs a rank-deficient
                // *constraint* block, so inertia control exhausts its shifts and
                // reports "reduced Hessian remains non-PD on null(A_W)".
                //
                // Prune to a maximal independent subset and retry at the same
                // `t`, exactly as `solve_general` does. A dropped row is a linear
                // combination of the kept ones, so it stays satisfied at the
                // current iterate and the feasible set is unchanged — only the
                // rank deficiency goes away. `x` and `t` do not move, so nothing
                // about the path's position is lost.
                //
                // This matters late rather than early: measured on the 24
                // smallest problems, 4 hit this, and they hit it near the end of
                // the path (`QSHARE2B` at t = 0.99999, `QSCAGR7` at t = 0.99986).
                // Bailing there discards a path that was one step from done.
                Err(e) if e.is_recoverable_factorization_failure() && rank_repairs > 0 => {
                    // Re-select the working set out of the *whole* active set,
                    // with any rows the exchange pivot has promoted offered
                    // first — `independent_active_subset` is a greedy scan that
                    // keeps the earliest independent rows, so the ordering is
                    // how a promotion is expressed.
                    let ordered = promote_first(&active_cons, &promoted);
                    let (kc, kb) = crate::solver::independent_active_subset(
                        &mut self.linsol,
                        &path_qp,
                        &ordered,
                        &active_bounds,
                    );
                    if kc.len() == kkt_cons.len()
                        && kb.len() == kkt_bounds.len()
                        && promoted.is_empty()
                    {
                        // Already full rank ⇒ not a deficiency this can repair.
                        if trace {
                            eprintln!("[hom] KKT failure at t={t:.6e}, full rank already: {e}");
                        }
                        return Ok(None);
                    }
                    rank_repairs -= 1;
                    // The pruned rows stay in the *active set* — they are still
                    // on their bounds — and leave only the KKT. The step logic
                    // below is what keeps them there.
                    let mut keep_c = vec![false; m];
                    for &i in &kc {
                        keep_c[i] = true;
                    }
                    let mut keep_b = vec![false; n];
                    for &j in &kb {
                        keep_b[j] = true;
                    }
                    for &i in &active_cons {
                        if !keep_c[i] && in_kkt_c[i] {
                            in_kkt_c[i] = false;
                            lambda_g[i] = 0.0;
                        }
                    }
                    for &j in &active_bounds {
                        if !keep_b[j] && in_kkt_b[j] {
                            in_kkt_b[j] = false;
                            lambda_x[j] = 0.0;
                        }
                    }
                    for &i in &kc {
                        in_kkt_c[i] = true;
                    }
                    for &j in &kb {
                        in_kkt_b[j] = true;
                    }
                    if trace {
                        eprintln!(
                            "[hom] rank repair at t={t:.6e}: {} active cons -> {} in KKT, \
                             {} active bounds -> {} in KKT",
                            active_cons.len(),
                            kc.len(),
                            active_bounds.len(),
                            kb.len()
                        );
                    }
                    continue;
                }
                Err(e) => {
                    if trace {
                        eprintln!("[hom] KKT factorization failed at t={t:.6e}: {e}");
                    }
                    return Ok(None);
                }
            }
            n_refactor += 1;
            let dx: Vec<Number> = rhs[..n].to_vec();
            let dlam_c: Vec<Number> = (0..k_c).map(|s| rhs[n + s]).collect();
            let dlam_b: Vec<Number> = (0..k_b).map(|s| rhs[n + k_c + s]).collect();

            let a_dx = a_times_x(qp.a, &dx, m);
            let ax = a_times_x(qp.a, &x, m);

            // ---- Exchange pivot at a degenerate vertex ----
            //
            // Rows in the active set but not the KKT are on their bounds, but
            // the direction solve was never told to hold them: `dx` satisfies
            // `a_j·dx = rate_j` only for `j` in the working set. A pruned row
            // `i` is a combination of those, `a_i = Σ c_j a_j`, so its residual
            // moves at `a_i·dx = Σ c_j rate_j` — a quantity fixed by the working
            // set, with no reason to match row `i`'s *own* bound rate. When it
            // exceeds that rate the row is about to be pushed off its bound, and
            // stepping would violate it permanently.
            //
            // (The old comment here asserted the opposite — "a pruned row is a
            // linear combination of the kept ones, so it stays satisfied". That
            // is true of a *static* point and false along a path where the
            // bounds themselves are moving, which is the whole situation.)
            //
            // The fix is the textbook degenerate-vertex exchange: `a_i·dx` can
            // only be changed by changing the working set, so promote row `i`
            // into it and let the greedy independence scan evict whichever row
            // it now depends on. The new direction holds row `i` exactly.
            //
            // Promotions accumulate at this `t` and are cleared as soon as the
            // path advances, so the exchange never permanently reorders the
            // problem. If a row that is already promoted still gets pushed off,
            // no working set drawn from this vertex can hold every binding row
            // and the path genuinely cannot continue — reported and abandoned
            // rather than papered over, which is safe because `pounce-convex`
            // now keeps a seeded fallback for exactly this case.
            let mut offender: Option<usize> = None;
            for &i in &active_cons {
                if in_kkt_c[i] {
                    continue;
                }
                let (rate, own) = match working.constraints[i] {
                    ConsStatus::AtUpper | ConsStatus::Equality => {
                        (a_dx[i], bound_rate(bu0[i], qp.bu[i], false))
                    }
                    ConsStatus::AtLower => (-a_dx[i], -bound_rate(bl0[i], qp.bl[i], true)),
                    ConsStatus::Inactive => unreachable!("filtered by active_cons"),
                };
                // Outward motion, scaled by the row's own magnitude so the test
                // is not fooled by a badly scaled row.
                if rate - own > PATH_FEAS_TOL_REL * (1.0 + a_dx[i].abs()) {
                    offender = Some(i);
                    break;
                }
            }
            // An already-promoted row that is *still* being pushed off means the
            // independence scan could not keep it even when offered first — it
            // depends on rows promoted before it, so no working set drawn from
            // this vertex holds them all. Take the step anyway rather than
            // abandoning the path: the capture below pins the row back if the
            // step really does violate it, which bounds the damage, and the path
            // is only a predictor. (Abandoning here was tried and cost far more
            // than it saved — it fired on every degenerate vertex and threw away
            // paths that were about to complete.)
            let stuck = offender.is_some_and(|i| promoted.contains(&i) || exchanges == 0);
            if let Some(i) = offender.filter(|_| !stuck) {
                exchanges -= 1;
                promoted.insert(0, i);
                let ordered = promote_first(&active_cons, &promoted);
                let (kc, kb) = crate::solver::independent_active_subset(
                    &mut self.linsol,
                    &path_qp,
                    &ordered,
                    &active_bounds,
                );
                for &c in &active_cons {
                    in_kkt_c[c] = false;
                }
                for &bnd in &active_bounds {
                    in_kkt_b[bnd] = false;
                }
                for &c in &kc {
                    in_kkt_c[c] = true;
                }
                for &bnd in &kb {
                    in_kkt_b[bnd] = true;
                }
                n_changes += 1;
                if trace {
                    eprintln!("[hom] exchange at t={t:.6e}: promoted row {i} into the KKT");
                }
                continue;
            }

            // ---- Ratio test 1 (primal): when does an inactive row bind? ----
            let mut t_next: Number = 1.0;
            let mut event: Option<Event> = None;

            for i in 0..m {
                if working.constraints[i].is_active() {
                    continue;
                }
                // Upper: a_i·x(t) − bu_i(t) = 0.
                if qp.bu[i] < NLP_UPPER_BOUND_INF {
                    let gap = bu0[i] + t * (qp.bu[i] - bu0[i]) - ax[i];
                    let rate = a_dx[i] - bound_rate(bu0[i], qp.bu[i], false);
                    if rate > 0.0 {
                        let dt = gap / rate;
                        if dt >= -T_EPS && t + dt < t_next - T_EPS {
                            t_next = (t + dt).clamp(t, 1.0);
                            event = Some(Event::AddRowUpper(i));
                        }
                    }
                }
                // Lower: bl_i(t) − a_i·x(t) = 0.
                if qp.bl[i] > NLP_LOWER_BOUND_INF {
                    let gap = ax[i] - (bl0[i] + t * (qp.bl[i] - bl0[i]));
                    let rate = bound_rate(bl0[i], qp.bl[i], true) - a_dx[i];
                    if rate > 0.0 {
                        let dt = gap / rate;
                        if dt >= -T_EPS && t + dt < t_next - T_EPS {
                            t_next = (t + dt).clamp(t, 1.0);
                            event = Some(Event::AddRowLower(i));
                        }
                    }
                }
            }

            // ---- Ratio test 2 (dual): when does an active multiplier vanish? ----
            //
            // An inequality's multiplier must keep its sign; reaching zero means
            // the row stops binding and has to leave the working set. Equality
            // rows are exempt — their multipliers are unrestricted.
            for (slot, &i) in kkt_cons.iter().enumerate() {
                if matches!(working.constraints[i], ConsStatus::Equality) {
                    continue;
                }
                let lam = lambda_g[i];
                let rate = dlam_c[slot];
                // Sign convention: `AtUpper` multipliers are ≥ 0, `AtLower` ≤ 0
                // in this engine's packing (see `solve_general`'s drop test).
                let heading_to_zero = (lam > 0.0 && rate < 0.0) || (lam < 0.0 && rate > 0.0);
                if heading_to_zero {
                    let dt = -lam / rate;
                    if dt >= -T_EPS && t + dt < t_next - T_EPS {
                        t_next = (t + dt).clamp(t, 1.0);
                        event = Some(Event::DropRow(i));
                    }
                }
            }
            for (slot, &j) in kkt_bounds.iter().enumerate() {
                if matches!(working.bounds[j], BoundStatus::Fixed) {
                    continue;
                }
                let lam = lambda_x[j];
                let rate = dlam_b[slot];
                let heading_to_zero = (lam > 0.0 && rate < 0.0) || (lam < 0.0 && rate > 0.0);
                if heading_to_zero {
                    let dt = -lam / rate;
                    if dt >= -T_EPS && t + dt < t_next - T_EPS {
                        t_next = (t + dt).clamp(t, 1.0);
                        event = Some(Event::DropBound(j));
                    }
                }
            }

            // ---- Advance to the event (or to t = 1) ----
            let dt = t_next - t;
            if dt > T_EPS {
                // Real progress along the path. The exchange pivot's promotions
                // were a statement about the dependency structure *at the vertex
                // just left*, which no longer necessarily holds, so release them.
                promoted.clear();
            }
            for (xi, &d) in x.iter_mut().zip(dx.iter()) {
                *xi += dt * d;
            }
            for (slot, &i) in kkt_cons.iter().enumerate() {
                lambda_g[i] += dt * dlam_c[slot];
            }
            for (slot, &j) in kkt_bounds.iter().enumerate() {
                lambda_x[j] += dt * dlam_b[slot];
            }
            t = t_next;

            // ---- Post-step capture: the ratio test's missing repair ----
            //
            // The ratio test can only ever *prevent* a violation. It picks the
            // earliest crossing and caps `dt` there; a row whose gap has already
            // gone negative yields `dt < 0`, which the `dt >= -T_EPS` filter
            // discards, so nothing ever pulls it back. Violation is absorbing,
            // and the direction solve keeps pushing — on QSHARE2B row 7 entered
            // at `8e-2` and reached `22` by `t = 1`.
            //
            // Rows get past the cap because `t` has finite resolution and row
            // rates do not. Two crossings within `T_EPS` of each other in `t`
            // lose the strict `t + dt < t_next - T_EPS` comparison, and a row
            // moving at `a_i·dx ≈ 1e8` turns that `1e-12` of unresolved `t` into
            // a real `1e-4` of constraint violation. Shrinking `T_EPS` cannot
            // fix this — the rate is unbounded, and at `t ≈ 1 - 1e-5` there is
            // no resolution left to give.
            //
            // So repair instead of preventing: a row the step pushed past its
            // bound is *added to the active set*, which is what the ratio test
            // would have done had it resolved the crossing. `x` keeps whatever
            // small overshoot it picked up, but the row is now held by the
            // direction solve, so the error stops compounding — and the working
            // set, which is the only thing this path actually hands downstream,
            // says the truth about which rows are binding.
            for i in 0..m {
                if working.constraints[i].is_active() {
                    continue;
                }
                let axi = ax[i] + dt * a_dx[i];
                let tol = PATH_FEAS_TOL_REL * (1.0 + axi.abs());
                if qp.bu[i] < NLP_UPPER_BOUND_INF
                    && axi - (bu0[i] + t * bound_rate(bu0[i], qp.bu[i], false)) > tol
                {
                    working.constraints[i] = ConsStatus::AtUpper;
                    in_kkt_c[i] = true;
                    n_changes += 1;
                } else if qp.bl[i] > NLP_LOWER_BOUND_INF
                    && (bl0[i] + t * bound_rate(bl0[i], qp.bl[i], true)) - axi > tol
                {
                    working.constraints[i] = ConsStatus::AtLower;
                    in_kkt_c[i] = true;
                    n_changes += 1;
                }
            }

            // The invariant the exchange pivot and the capture above exist to
            // hold. Kept as an assertion-shaped trace rather than a bail-out: the
            // path is only a predictor, the corrector re-derives the primal from
            // the working set, and a small late drift is not worth discarding a
            // path over.
            if trace && let Some((i, v)) = worst_path_violation(qp, &x, &bl0, &bu0, t, m) {
                eprintln!("[hom] path infeasible at t={t:.9e}: row {i} by {v:.3e}");
            }

            match event {
                None => {
                    // Nothing binds before t = 1: the path is complete.
                    t = 1.0;
                    break;
                }
                Some(Event::AddRowUpper(i)) => {
                    working.constraints[i] = ConsStatus::AtUpper;
                    in_kkt_c[i] = true;
                    n_changes += 1;
                }
                Some(Event::AddRowLower(i)) => {
                    working.constraints[i] = ConsStatus::AtLower;
                    in_kkt_c[i] = true;
                    n_changes += 1;
                }
                Some(Event::DropRow(i)) => {
                    working.constraints[i] = ConsStatus::Inactive;
                    lambda_g[i] = 0.0;
                    in_kkt_c[i] = true;
                    n_changes += 1;
                }
                Some(Event::DropBound(j)) => {
                    working.bounds[j] = BoundStatus::Inactive;
                    lambda_x[j] = 0.0;
                    in_kkt_b[j] = true;
                    n_changes += 1;
                }
            }
        }

        // The path is only a *predictor* for the final active set: `t` may have
        // stopped short of 1, and the linear algebra along the way accumulates
        // error. Hand the discovered working set to the conventional solver,
        // which corrects the iterate and applies the usual feasibility audit and
        // status logic. That keeps every existing guarantee — nothing here is
        // reported as optimal on the homotopy's own authority.
        if t < 1.0 - T_EPS {
            if trace {
                eprintln!("[hom] path did NOT reach t=1 (stopped at {t:.6e}); falling back");
            }
            return Ok(None);
        }
        if trace {
            eprintln!(
                "[hom] reached t=1 after {n_changes} working-set changes; handoff x has \
                 max target violation {:.3e}",
                crate::solver::max_violation(qp, &x)
            );
        }
        let mut sol =
            <Self as crate::solver::QpSolver>::solve_with_working_set(self, qp, &working, opts)?;
        sol.stats = QpStats {
            n_working_set_changes: sol.stats.n_working_set_changes + n_changes,
            n_refactor: sol.stats.n_refactor + n_refactor,
            n_schur_updates: sol.stats.n_schur_updates,
            used_phase1: false,
            time: started.elapsed(),
        };
        Ok(Some(sol))
    }
}
