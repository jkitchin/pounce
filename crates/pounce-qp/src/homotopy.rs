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

/// How close to `1` counts as having reached the end of the path.
const T_EPS: Number = 1e-12;

/// Two ratio-test events are **coincident** — the same degenerate vertex, hit at
/// the same parameter value — when their crossing points differ by no more than
/// this. `t` lives in `[0, 1]`, so this is a rounding-scale window (~50 ulp),
/// not a tolerance with an algorithmic opinion in it.
///
/// It exists because the winning event has to bring *every* row that binds
/// there into the working set, not just the first one the scan happened to
/// find. A row left inactive at a bound it is exactly on gets crossed by the
/// next direction, and a crossing is unrecoverable: the primal ratio test can
/// only *prevent* a violation, never repair one, so from there the row drifts
/// out for the rest of the path (`QSHARE2B` row 7 went 8e-2 -> 22 that way).
const T_TIE: Number = 1e-14;

/// Outcome of the two ratio tests: what happens first as `t` increases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Event {
    /// Inactive row `i` reaches its lower bound.
    AddRowLower(usize),
    /// Inactive row `i` reaches its upper bound.
    AddRowUpper(usize),
    /// Inactive variable `j` reaches its lower bound.
    AddBoundLower(usize),
    /// Inactive variable `j` reaches its upper bound.
    AddBoundUpper(usize),
    /// Active row `i`'s multiplier reaches zero and must leave.
    DropRow(usize),
    /// Active bound on variable `j` has its multiplier reach zero.
    DropBound(usize),
}

/// The winner set of one step's two ratio tests: the earliest crossing, and
/// every other crossing coincident with it.
///
/// Both tests feed the same instance, because they compete for the same step —
/// whichever of "a row binds" and "a multiplier vanishes" happens first sets
/// `t_next`, and anything within [`T_TIE`] of it happens there too.
///
/// Separated out from the tracer because the selection rule is where gh #434's
/// uncapped crossings came from and is worth testing on its own: it is pure,
/// and its inputs are two numbers, while the loop it lives in needs a
/// factorization per step to reach.
pub(crate) struct RatioTest {
    /// Parameter value the step will advance to.
    pub(crate) t_next: Number,
    /// Current parameter value; crossings are reported relative to it.
    t: Number,
    /// The winner and its coincident set, as `(parameter value, event)`.
    pub(crate) winners: Vec<(Number, Event)>,
}

impl RatioTest {
    /// A fresh test at `t`, with no crossing found yet — so the incumbent is the
    /// end of the path.
    pub(crate) fn new(t: Number) -> Self {
        RatioTest {
            t_next: 1.0,
            t,
            winners: Vec::new(),
        }
    }

    /// Reset to `t` for the next step, keeping the allocation.
    pub(crate) fn restart(&mut self, t: Number) {
        self.t_next = 1.0;
        self.t = t;
        self.winners.clear();
    }

    /// Offer a crossing `dt` ahead of the current `t`.
    ///
    /// A slightly negative `dt` is a row a hair past its bound already; it is
    /// admitted and clamped to `t`, so it binds now with a zero-length step.
    /// Further out than that is a row the ratio test can no longer recover (see
    /// the path-feasibility report in [`ParametricActiveSetSolver::trace_path`]),
    /// and past `t = 1` is off the end of the path — neither is an event.
    ///
    /// The comparison against the incumbent is exact. It used to carry a
    /// `- T_EPS` margin, which reads as a don't-bother-for-a-hair guard but is
    /// not one: it made a crossing that happens *earlier* than the incumbent, by
    /// less than `T_EPS`, lose to it — so the step knowingly overshot the
    /// earlier one. Measured on `QSHARE2B`, row 132 crossed at `dt = 2.9e-16`
    /// and lost to a step of `1.1e-14`; the row was left inactive and violated,
    /// and since the ratio test can only prevent a violation and never repair
    /// one, it drifted out for the rest of the path (gh #434).
    ///
    /// Anything strictly worse than the incumbent is dropped as it arrives, so
    /// the winner set stays the size of one coincident group rather than growing
    /// with the row count.
    pub(crate) fn admit(&mut self, dt: Number, ev: Event) {
        if dt < -T_EPS {
            return;
        }
        let tc = (self.t + dt).max(self.t);
        if tc > self.t_next + T_TIE {
            return;
        }
        if tc < self.t_next - T_TIE {
            self.winners.clear();
        }
        self.t_next = self.t_next.min(tc);
        self.winners.push((tc, ev));
    }

    /// The events that fire at `t_next`: the winner plus everything coincident.
    ///
    /// Firing only one of a coincident set is what leaves a row sitting exactly
    /// on a bound it is not in the working set for, which the next direction
    /// then pushes it across.
    pub(crate) fn firing(&self) -> impl Iterator<Item = Event> + '_ {
        let t_next = self.t_next;
        self.winners
            .iter()
            .filter(move |&&(tc, _)| tc <= t_next + T_TIE)
            .map(|&(_, ev)| ev)
    }
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
/// bounds do not move along this path, so they are checked against `qp` directly.
///
/// The box used to be skipped here entirely, on the reasoning that it does not
/// move. Not moving is not the same as not being violated: until gh #602 there
/// was no bound-adding ratio test, so `x(t)` could walk straight out of a
/// stationary box and this report — the only instrument pointed at path
/// feasibility — was blind to it by construction. Variables are reported with
/// index `n_rows + j` so one return type covers both.
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
    for j in 0..qp.n {
        let v = (qp.xl[j] - x[j]).max(x[j] - qp.xu[j]);
        if v > PATH_FEAS_TOL_REL * (1.0 + x[j].abs()) && worst.is_none_or(|(_, prev)| v > prev) {
            worst = Some((m + j, v));
        }
    }
    worst
}

/// Machine-readable one-line summary of a completed path, emitted on every
/// exit from [`ParametricActiveSetSolver::trace_path`] when
/// `POUNCE_HOMOTOPY_DEBUG` is set.
///
/// The line is a stable, parseable contract rather than prose because the
/// measurement gh #434 asks for — path steps and final `t`, per problem, for
/// both arms — is a benchmark sweep that has to read it back out. `exit` is one
/// of `complete` (the path reached `t = 1`), `budget` (the step budget ran out
/// short of it), `stalled` (the loop ended without either), `kkt` (an
/// unrecoverable factorization failure), or `rank` (a rank repair that had
/// nothing left to prune).
fn trace_summary(
    exit: &str,
    steps: u32,
    t: Number,
    n_changes: u32,
    n_refactor: u32,
    longest_stall: u32,
) {
    eprintln!(
        "[hom] summary exit={exit} steps={steps} t={t:.17} changes={n_changes} \
         refactor={n_refactor} stall={longest_stall}"
    );
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
                Ok(mut s) if s.status == QpStatus::TimeLimit => {
                    s.lambda_g.resize(m, 0.0);
                    s.working.constraints.resize(m, ConsStatus::Inactive);
                    return Ok(Some(s));
                }
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
                        Ok(mut s) if s.status == QpStatus::TimeLimit => {
                            s.lambda_g.resize(m, 0.0);
                            s.working.constraints.resize(m, ConsStatus::Inactive);
                            return Ok(Some(s));
                        }
                        _ => return Ok(None),
                    }
                }
            }
        };

        // Everything on the path is traced against this Hessian; the corrector at
        // the end uses the caller's `qp` and therefore the true `H`.
        let path_h: &SymTMatrix = h_reg_holder.as_ref().unwrap_or(qp.h);

        let x = box_sol.x.clone();
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

        let lambda_g = vec![0.0; m];
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
        // Rank repairs allowed on this path. Each strictly shrinks the active
        // set so the retry loop terminates on its own; the cap bounds the case
        // where the ratio test keeps re-adding a pruned row.
        let mut rank_repairs: u32 = (n + m).min(1000) as u32;
        // Rows pruned by a rank repair *at the current* `t`. Without this the
        // repair and the primal ratio test fight each other: the repair drops a
        // linearly dependent row, the ratio test immediately re-adds it because
        // it is still exactly at its bound at this `t`, and the pair loops
        // forever without advancing. Measured on QSHARE2B, which repaired
        // 79 -> 77 constraints at t = 0.9999973 and then repeated that same
        // repair until the budget ran out.
        //
        // A pruned row is a linear combination of the kept ones, so it stays
        // satisfied and excluding it changes nothing about feasibility. The tabu
        // is cleared the moment `t` actually advances, because at a new `t` the
        // dependency that justified the prune no longer necessarily holds — this
        // suppresses the cycle without permanently blinding the ratio test.
        let mut tabu_cons = vec![false; m];
        // The same tabu for bounds, and it only became necessary with the
        // bound-adding ratio test above: before that a bound the rank repair
        // pruned could never come back, so the prune -> re-add cycle `tabu_cons`
        // exists to break had no bound analogue. It does now.
        let mut tabu_bounds = vec![false; n];
        // Iterations actually executed. Distinct from `n_changes`: a rank repair
        // and a degenerate zero-length advance both consume a step (and a
        // factorization) without necessarily moving the working set or `t`.
        let mut steps: u32 = 0;
        // Hoisted out of the loop and restarted per step so the ratio tests do
        // not allocate once per path step.
        let mut ratio = RatioTest::new(0.0);
        // Steps since `t` last advanced, and the worst such run on this path.
        //
        // A step that does not move `t` is a working-set exchange at a fixed
        // parameter value — a degenerate pivot. A few are normal. An unbounded
        // number of them is the path cycling, which is how the homotopy's
        // measured losses actually fail: they are not slow paths, they are
        // stopped ones (gh #434).
        let mut stalled: u32 = 0;
        let mut longest_stall: u32 = 0;

        // Each iteration either advances `t` or changes the working set, and the
        // budget bounds the total.
        for _step in 0..opts.max_iter {
            if crate::deadline::expired() {
                return Ok(Some(QpSolution {
                    obj: crate::solver::quad_objective(qp, &x),
                    x,
                    lambda_g,
                    lambda_x,
                    working,
                    status: QpStatus::TimeLimit,
                    stats: QpStats {
                        n_working_set_changes: n_changes,
                        n_refactor,
                        n_schur_updates: 0,
                        used_phase1: false,
                        time: started.elapsed(),
                    },
                    unbounded_ray: None,
                }));
            }
            if t >= 1.0 - T_EPS {
                break;
            }
            steps += 1;
            if trace && _step % 50 == 0 {
                eprintln!("[hom] step={_step} t={t:.17} stall={stalled}");
            }

            let active_cons: Vec<usize> = (0..m)
                .filter(|&i| working.constraints[i].is_active())
                .collect();
            let active_bounds: Vec<usize> =
                (0..n).filter(|&i| working.bounds[i].is_active()).collect();
            let (k_c, k_b) = (active_cons.len(), active_bounds.len());

            // ---- Direction: d/dt of (x, λ) along this segment ----
            //
            // `H` is fixed along the path; `g` moves only on the warm path (see
            // `dg` below). The active rows' bounds advance at `bound_rate`.
            let kkt = assemble_active_set_kkt(&path_qp, &active_cons, &active_bounds);
            let mut rhs = vec![0.0; n + k_c + k_b];
            // Stationarity block: `H x(t) + g(t) + A_Wᵀ λ(t) = 0` differentiated
            // in `t` gives `H dx + dg + A_Wᵀ dλ = 0`, so the primal right-hand
            // side is `−dg`. Zero on the cold path, where `g` is held fixed and
            // only the row bounds move; non-zero on the warm path, where the
            // objective moves from the previous problem's to the new one's.
            for (j, r) in rhs[..n].iter_mut().enumerate() {
                *r = -dg[j];
            }
            for (slot, &i) in active_cons.iter().enumerate() {
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
                    let (kc, kb) = crate::solver::independent_active_subset(
                        &mut self.linsol,
                        &path_qp,
                        &active_cons,
                        &active_bounds,
                    );
                    if kc.len() == active_cons.len() && kb.len() == active_bounds.len() {
                        // Already full rank ⇒ not a deficiency this can repair.
                        if trace {
                            eprintln!("[hom] KKT failure at t={t:.6e}, full rank already: {e}");
                            trace_summary("rank", steps, t, n_changes, n_refactor, longest_stall);
                        }
                        return Ok(None);
                    }
                    rank_repairs -= 1;
                    let mut keep_c = vec![false; m];
                    for &i in &kc {
                        keep_c[i] = true;
                    }
                    let mut keep_b = vec![false; n];
                    for &j in &kb {
                        keep_b[j] = true;
                    }
                    for &i in &active_cons {
                        if !keep_c[i] {
                            working.constraints[i] = ConsStatus::Inactive;
                            lambda_g[i] = 0.0;
                            tabu_cons[i] = true;
                            n_changes += 1;
                        }
                    }
                    for &j in &active_bounds {
                        if !keep_b[j] {
                            working.bounds[j] = BoundStatus::Inactive;
                            lambda_x[j] = 0.0;
                            tabu_bounds[j] = true;
                            n_changes += 1;
                        }
                    }
                    if trace {
                        eprintln!(
                            "[hom] rank repair at t={t:.6e}: {} -> {} cons, {} -> {} bounds",
                            active_cons.len(),
                            kc.len(),
                            active_bounds.len(),
                            kb.len()
                        );
                    }
                    continue;
                }
                // A cancelled factorization is not a path breakdown. Falling
                // through to `Ok(None)` would report "path could not be
                // started" and send the caller off to begin a *fresh*
                // conventional solve with the budget already spent; propagate
                // so the entry point turns it into `TimeLimit` directly.
                Err(QpError::DeadlineExpired) => return Err(QpError::DeadlineExpired),
                Err(e) => {
                    if trace {
                        eprintln!("[hom] KKT factorization failed at t={t:.6e}: {e}");
                        trace_summary("kkt", steps, t, n_changes, n_refactor, longest_stall);
                    }
                    return Ok(None);
                }
            }
            n_refactor += 1;
            let dx: Vec<Number> = rhs[..n].to_vec();
            let dlam_c: Vec<Number> = (0..k_c).map(|s| rhs[n + s]).collect();
            let dlam_b: Vec<Number> = (0..k_b).map(|s| rhs[n + k_c + s]).collect();

            // ---- Ratio test 1 (primal): when does an inactive row bind? ----
            let a_dx = a_times_x(qp.a, &dx, m);
            let ax = a_times_x(qp.a, &x, m);
            ratio.restart(t);

            for i in 0..m {
                if working.constraints[i].is_active() || tabu_cons[i] {
                    continue;
                }
                // Upper: a_i·x(t) − bu_i(t) = 0.
                if qp.bu[i] < NLP_UPPER_BOUND_INF {
                    let gap = bu0[i] + t * (qp.bu[i] - bu0[i]) - ax[i];
                    let rate = a_dx[i] - bound_rate(bu0[i], qp.bu[i], false);
                    if rate > 0.0 {
                        ratio.admit(gap / rate, Event::AddRowUpper(i));
                    }
                }
                // Lower: bl_i(t) − a_i·x(t) = 0.
                if qp.bl[i] > NLP_LOWER_BOUND_INF {
                    let gap = ax[i] - (bl0[i] + t * (qp.bl[i] - bl0[i]));
                    let rate = bound_rate(bl0[i], qp.bl[i], true) - a_dx[i];
                    if rate > 0.0 {
                        ratio.admit(gap / rate, Event::AddRowLower(i));
                    }
                }
            }

            // ---- Ratio test 1b (primal): when does an inactive *bound* bind? ----
            //
            // The same test as the rows above, on the identity rows of the box.
            // It is simpler because variable bounds do not move along this path:
            // the bound's own rate is zero, so the crossing is governed by `dx`
            // alone.
            //
            // This did not exist before gh #602. Without it the path could drop
            // a bound but never add one, so `x(t)` crossed inactive bounds with
            // nothing to cap the step — silently, since `worst_path_violation`
            // skipped the box too. That is a defect on the *cold* arm as much as
            // the warm one: nothing about the box relaxation start prevents the
            // direction from leaving the box on the way to `t = 1`.
            //
            // Same absorbing-violation logic as the rows: the ratio test can only
            // *prevent* a crossing, never repair one, so a bound crossed once
            // stays crossed for the rest of the path.
            for j in 0..n {
                if working.bounds[j].is_active() || tabu_bounds[j] {
                    continue;
                }
                if qp.xu[j] < NLP_UPPER_BOUND_INF && dx[j] > 0.0 {
                    ratio.admit((qp.xu[j] - x[j]) / dx[j], Event::AddBoundUpper(j));
                }
                if qp.xl[j] > NLP_LOWER_BOUND_INF && dx[j] < 0.0 {
                    ratio.admit((x[j] - qp.xl[j]) / -dx[j], Event::AddBoundLower(j));
                }
            }

            // ---- Ratio test 2 (dual): when does an active multiplier vanish? ----
            //
            // An inequality's multiplier must keep its sign; reaching zero means
            // the row stops binding and has to leave the working set. Equality
            // rows are exempt — their multipliers are unrestricted.
            for (slot, &i) in active_cons.iter().enumerate() {
                if matches!(working.constraints[i], ConsStatus::Equality) {
                    continue;
                }
                let lam = lambda_g[i];
                let rate = dlam_c[slot];
                // Sign convention: `AtUpper` multipliers are ≥ 0, `AtLower` ≤ 0
                // in this engine's packing (see `solve_general`'s drop test).
                let heading_to_zero = (lam > 0.0 && rate < 0.0) || (lam < 0.0 && rate > 0.0);
                if heading_to_zero {
                    ratio.admit(-lam / rate, Event::DropRow(i));
                }
            }
            for (slot, &j) in active_bounds.iter().enumerate() {
                if matches!(working.bounds[j], BoundStatus::Fixed) {
                    continue;
                }
                let lam = lambda_x[j];
                let rate = dlam_b[slot];
                let heading_to_zero = (lam > 0.0 && rate < 0.0) || (lam < 0.0 && rate > 0.0);
                if heading_to_zero {
                    ratio.admit(-lam / rate, Event::DropBound(j));
                }
            }

            // ---- Advance to the event (or to t = 1) ----
            //
            // Firing a whole coincident set can leave the active set
            // rank-deficient. That is not a new failure mode — the direction
            // solve above already repairs it by pruning to a maximal independent
            // subset — and it is the same trade the textbook degenerate-vertex
            // rule makes.
            let dt = ratio.t_next - t;
            if dt > T_EPS {
                // Real progress along the path: the rank-repair tabu was scoped
                // to the parameter value it was raised at, so release it.
                tabu_cons.iter_mut().for_each(|f| *f = false);
                tabu_bounds.iter_mut().for_each(|f| *f = false);
                stalled = 0;
            } else {
                stalled += 1;
                longest_stall = longest_stall.max(stalled);
            }
            for (xi, &d) in x.iter_mut().zip(dx.iter()) {
                *xi += dt * d;
            }
            for (slot, &i) in active_cons.iter().enumerate() {
                lambda_g[i] += dt * dlam_c[slot];
            }
            for (slot, &j) in active_bounds.iter().enumerate() {
                lambda_x[j] += dt * dlam_b[slot];
            }
            t = ratio.t_next;

            // ---- Path-feasibility report ----
            //
            // The module header used to claim `x(t)` is feasible for the
            // `t`-problem at every point "by construction". It is not. The primal
            // ratio test only ever *prevents* a violation and cannot repair one:
            // a row whose gap has gone negative yields `dt < 0`, which the
            // `dt >= -T_EPS` filter rejects, so the row stays inactive and
            // violated for the rest of the path while the direction solve pushes
            // it further out. Violation is absorbing.
            //
            // Two measured ways in (QSHARE2B, 14 crossings on one path):
            //
            //  * **Rank-repair tabu (10 of 14).** `tabu_cons` exists to break a
            //    prune -> re-add cycle, but it skips the row in the *primal ratio
            //    test*, not just in the add decision. The pruned row sits on its
            //    bound with a large rate (`a_i·dx = 2.6e5` on row 19), the step is
            //    computed as if it were absent, and it is crossed. The comment
            //    justifying the tabu — "a pruned row is a linear combination of
            //    the kept ones, so it stays satisfied" — is false: dependence
            //    gives `a_i·dx = Σ c_j (a_j·dx)`, and nothing makes that
            //    combination track row `i`'s *own* bound rate.
            //
            //  * **Coincident / sub-resolution events (4 of 14).** ~~Two
            //    crossings at the same `t_next`, or one below the `T_EPS` floor
            //    (row 132: crossing at `dt = 2.9e-16`, step taken `1.1e-14`),
            //    lose the strict `t + dt < t_next - T_EPS` comparison.~~ Fixed
            //    in gh #434: [`RatioTest`] compares crossings exactly and fires
            //    the whole coincident set, so neither an earlier-by-a-hair
            //    crossing nor a tied one is stepped over. That alone recovered
            //    five of the seven instances the homotopy had been losing.
            //
            // Repairing the *first* mechanism properly needs an exchange pivot at
            // the degenerate vertex (add the violated row, drop a dependent one),
            // which is a real algorithmic addition and is left to follow-up work.
            //
            // Deliberately a *report* and not a bail-out. Abandoning the path here
            // was tried and measured worse: the corrector is a genuine corrector
            // and often recovers from a path that drifted off (`QSHIP04S` reaches
            // a violation of 7.1e4 at `t = 0.5` and still solves to the published
            // optimum). What the caller needs is not for this path to give up, but
            // a *fallback that exists* when its prediction turns out unusable —
            // which is what `pounce-convex`'s seeded last-resort retry restores.
            if trace && let Some((i, v)) = worst_path_violation(qp, &x, &bl0, &bu0, t, m) {
                // Indices at or past `m` are variables, not rows — decode them
                // here or a box violation prints as a row that does not exist.
                let what = if i < m {
                    format!("row {i}")
                } else {
                    format!("bound on x[{}]", i - m)
                };
                eprintln!("[hom] path infeasible at t={t:.9e}: {what} by {v:.3e}");
            }

            if ratio.winners.is_empty() {
                // Nothing binds before t = 1: the path is complete.
                t = 1.0;
                break;
            }
            for ev in ratio.firing() {
                match ev {
                    Event::AddRowUpper(i) => {
                        working.constraints[i] = ConsStatus::AtUpper;
                    }
                    Event::AddRowLower(i) => {
                        working.constraints[i] = ConsStatus::AtLower;
                    }
                    Event::AddBoundUpper(j) => {
                        working.bounds[j] = BoundStatus::AtUpper;
                    }
                    Event::AddBoundLower(j) => {
                        working.bounds[j] = BoundStatus::AtLower;
                    }
                    Event::DropRow(i) => {
                        working.constraints[i] = ConsStatus::Inactive;
                        lambda_g[i] = 0.0;
                    }
                    Event::DropBound(j) => {
                        working.bounds[j] = BoundStatus::Inactive;
                        lambda_x[j] = 0.0;
                    }
                }
                n_changes += 1;
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
                let exit = if steps >= opts.max_iter {
                    "budget"
                } else {
                    "stalled"
                };
                trace_summary(exit, steps, t, n_changes, n_refactor, longest_stall);
            }
            return Ok(None);
        }
        if trace {
            trace_summary("complete", steps, t, n_changes, n_refactor, longest_stall);
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
