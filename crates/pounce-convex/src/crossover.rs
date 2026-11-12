//! LP crossover: purify a near-optimal interior-point iterate to an *exact*
//! optimal vertex using the active-set QP engine ([`pounce_qp`]).
//!
//! # Why
//!
//! A pure interior-point method cannot certify a **degenerate** LP vertex to
//! `tol`: where strict complementarity fails the fraction-to-boundary step
//! collapses (`α → 1e-4`), `μ` freezes, and the primal residual plateaus above
//! the tolerance, so the solve grinds to its iteration cap with the *correct*
//! objective but no convergence certificate (NETLIB GEN family). This is a
//! fundamental limit — it is exactly why every commercial LP solver pairs an
//! IPM with a **crossover** that pivots the near-optimal interior point to an
//! exact optimal vertex basis (Andersen & Ye 1996, "Combining interior-point
//! and pivoting algorithms for linear programming"; Megiddo 1991).
//!
//! # How
//!
//! Two engines, tried in order, each judged by the never-regress gate below.
//!
//! 1. **Revised simplex** (`crate::simplex`): pivots the interior iterate
//!    to an optimal basis on an unsymmetric LU factor, one variable at a time,
//!    with Bland's rule through the degeneracy. It accepts or declines in
//!    milliseconds on the LPs crossover targets.
//! 2. **Active-set bridge** ([`pounce_qp::ParametricActiveSetSolver`]): the
//!    convex LP translated into pounce-qp's two-sided form
//!    ([`crate::active_set::ActiveSetQp`]), purified there, and the
//!    multipliers sign-transformed back. It runs when the simplex breaks down
//!    or its vertex is declined. After a decline it starts from the simplex
//!    basis, pinned (the basis is optimal; what was declined is the point and
//!    duals read off its factor). If that is not accepted either, or the
//!    simplex broke down, it starts from the interior iterate's active set,
//!    and may go through pounce-qp's l1-elastic phase 1 to another optimal
//!    vertex. Each bridge solve is capped at [`CROSSOVER_MAX_ITER`] (1000)
//!    working-set changes in all (the pivot under way when the cap is reached
//!    completes, so a rank repair inside it can pass the cap by a few).
//!
//! # Representation bridge
//!
//! Convex standard form (this crate, [`crate::qp::QpProblem`]):
//! `min ½xᵀPx + cᵀx  s.t.  Ax = b,  Gx ≤ h,  lb ≤ x ≤ ub`, with equality dual
//! `y` (free), inequality dual `z ≥ 0`, and bound duals `z_lb, z_ub ≥ 0`;
//! stationarity `Px + c + Aᵀy + Gᵀz − z_lb + z_ub = 0`.
//!
//! pounce-qp form ([`pounce_qp::QpProblem`]): `min ½xᵀHx + gᵀx`
//! `s.t.  bl ≤ A_qp x ≤ bu,  xl ≤ x ≤ xu`, Lagrangian
//! `L = ½xᵀHx + gᵀx + λ_gᵀ(A_qp x − β) + λ_satᵀ(x − β_bnd)` ⇒ stationarity
//! `Hx + g + A_qpᵀλ_g + λ_sat = 0`, with the user-facing bound multiplier
//! `lambda_x = z_l − z_u = −λ_sat` (`solver.rs` §5/§6 convention).
//!
//! Mapping `A_qp = [A_eq ; G]`, `g = c`, `H = P`:
//! - **eq rows** `bl = bu = b`            → `y[k]   = lambda_g[k]`
//! - **ineq rows** `bl = −∞, bu = h`      → `z[i]   = lambda_g[m_eq + i]`
//!   (`≥ 0` at the optimum: an active `≤` row is `AtUpper`, whose `lambda_g`
//!   the drop test in `solver.rs:810` keeps non-negative — *no* sign flip)
//! - **native variable bounds** `xl=lb, xu=ub` → `z_lb[i] = max(0,  lambda_x[i])`,
//!   `z_ub[i] = max(0, −lambda_x[i])` (from `−z_lb + z_ub = −lambda_x`).
//!
//! Mapping the convex variable bounds to pounce-qp's *native* `xl/xu` (rather
//! than expanding them into general rows) keeps the bridge correct for library
//! callers that carry bounds in `lb/ub`; the CLI `.nl` path already expands
//! bounds into `G` rows, so for the GEN target `lb/ub` are empty and the bound
//! mapping is a no-op.
//!
//! # Never-regress
//!
//! Crossover is a strict refinement: the purified vertex replaces the interior
//! iterate only when it is at least as good a KKT point of the *original*
//! problem (its `kkt_error()` does not exceed the interior iterate's, within a
//! tiny slack). Because an LP KKT point is globally optimal, a pounce-qp
//! `Optimal` vertex has the correct objective by construction; the KKT-error
//! gate is what guarantees we never return something worse — on any
//! `Err`/non-`Optimal`/regressing outcome we return the original solution
//! unchanged.

use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_qp::{
    AntiCyclingChoice, ConsStatus, ParametricActiveSetSolver, QpOptions as ActiveSetOptions,
    QpSolution as ActiveSetSolution, QpSolver, QpStatus as ActiveSetStatus, WorkingSet,
};

use crate::active_set::{ActiveSetQp, back_translate};
use crate::ipm::QpOptions;
use crate::qp::{QpProblem, QpSolution, QpStatus};

/// Inequality dual above which the IPM iterate is taken to mark a row active
/// (warm-start hint only; the active-set engine pivots to fix any over- or
/// under-identification, so this threshold need not be sharp).
const ACTIVE_Z_TOL: f64 = 1e-7;
/// Slack below which the IPM iterate is taken to mark a row binding.
const ACTIVE_S_TOL: f64 = 1e-7;
/// Slack on the never-regress KKT-error comparison.
const REGRESS_SLACK: f64 = 1e-9;
/// Working-set changes one bridge solve may make: the cap on each of its
/// passes (`max_iter`) and on all of them together (`max_working_set_changes`).
/// The whole-solve cap is the bound: an l1-elastic phase 1 makes several passes
/// — reduced-row and full-row, penalty escalations, a recovery solve — and
/// with only the per-pass cap NETLIB SHARE1B spent 5 535 changes, 18.7 s over
/// two solves, on a vertex it then declined. Public so callers and tests that
/// bound crossover's cost read the cap rather than restate it; at most two
/// bridge solves run per crossover.
pub const CROSSOVER_MAX_ITER: u32 = 1000;

/// What crossover did on the last [`maybe_crossover`] call that got past the
/// gate (gh#990 item 12). Carried out-of-band — the CLI prints a one-line
/// summary from it — because `QpSolution` has over 200 construction sites and
/// no `Default`; see [`crate::sigma_verdict`] for the same trade.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossoverReport {
    /// The engine whose vertex the never-regress gate judged last: `simplex`
    /// (the revised-simplex LU-basis engine), `active-set` (the pounce-qp
    /// bridge, after the simplex broke down or its vertex was declined), or
    /// `none` when no engine produced a vertex: the simplex broke down and
    /// the bridge did not reach `Optimal`, or the time budget ran out before
    /// either was judged. `simplex` with `accepted == false` is a declined
    /// simplex vertex the bridge then found nothing better than.
    pub engine: &'static str,
    /// The purified vertex replaced the interior iterate.
    pub accepted: bool,
    /// Structurals the interior iterate left strictly inside their bounds.
    /// This and the pivot and flip counts below are the simplex's, recorded
    /// whenever the simplex returned a vertex (accepted or declined), and `0`
    /// after a simplex breakdown; the bridge's work is
    /// `bridge_working_set_changes`.
    pub superbasics: usize,
    /// Basis changes spent resolving them.
    pub pivots_push: usize,
    pub pivots_phase1: usize,
    pub pivots_phase2: usize,
    /// Bound flips.
    pub flips: usize,
    /// Working-set changes the active-set bridge made, summed over its solves
    /// (`0` when it did not run). Each solve is capped at
    /// [`CROSSOVER_MAX_ITER`] changes in all, and at most two run, so this is
    /// at most twice the cap (plus the overshoot of a final rank repair).
    pub bridge_working_set_changes: u32,
    /// KKT error of the interior iterate and of the point returned.
    pub kkt_error_before: f64,
    pub kkt_error_after: f64,
}

#[derive(Default)]
struct Attempt {
    engine: Option<&'static str>,
    accepted: bool,
    stats: Option<crate::simplex::SimplexStats>,
    bridge_working_set_changes: u32,
}

thread_local! {
    static ATTEMPT: std::cell::RefCell<Option<Attempt>> = const { std::cell::RefCell::new(None) };
    static REPORT: std::cell::RefCell<Option<CrossoverReport>> = const { std::cell::RefCell::new(None) };
}

/// Forget any recorded report. Call before a solve whose report you will read.
pub fn clear_report() {
    REPORT.with(|r| *r.borrow_mut() = None);
}

/// Take the report of the last crossover attempt on this thread, if one ran.
pub fn take_report() -> Option<CrossoverReport> {
    REPORT.with(|r| r.borrow_mut().take())
}

/// Run the LP-crossover phase. Returns a purified exact-vertex solution when it
/// is a strict improvement (never-regress), otherwise the original `sol`.
///
/// `make_backend` supplies the sparse symmetric linear-solver backend for the
/// active-set engine — the same factory the IPM uses, so crossover inherits the
/// caller's solver choice (FERAL in tree).
pub fn maybe_crossover<F>(
    prob: &QpProblem,
    sol: QpSolution,
    opts: &QpOptions,
    make_backend: &mut F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if !opts.crossover || !prob.p_lower.is_empty() {
        return sol;
    }
    ATTEMPT.with(|a| *a.borrow_mut() = None);
    let orig = sol.clone();
    let out = maybe_crossover_inner(prob, sol, opts, make_backend);
    if let Some(a) = ATTEMPT.with(|a| a.borrow_mut().take()) {
        let st = a.stats.unwrap_or_default();
        let report = CrossoverReport {
            engine: a.engine.unwrap_or("none"),
            accepted: a.accepted,
            superbasics: st.superbasics,
            pivots_push: st.pivots_push,
            pivots_phase1: st.pivots_phase1,
            pivots_phase2: st.pivots_phase2,
            flips: st.flips,
            bridge_working_set_changes: a.bridge_working_set_changes,
            kkt_error_before: orig.kkt_residuals(prob).kkt_error(),
            kkt_error_after: out.kkt_residuals(prob).kkt_error(),
        };
        REPORT.with(|r| *r.borrow_mut() = Some(report));
    }
    out
}

fn maybe_crossover_inner<F>(
    prob: &QpProblem,
    sol: QpSolution,
    opts: &QpOptions,
    make_backend: &mut F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    // Out of budget: skip the refinement, exactly as every other abandonment
    // path here does (`_ => return sol`). Crossover only ever *improves* the
    // IPM's answer — restamping its status would turn "we declined to polish
    // this optimum" into "this problem was not solved", discarding a result the
    // IPM already produced. The entry point re-labels afterwards, and only when
    // there is no verdict to lose (see [`crate::ipm::mark_timed_out`]).
    if crate::deadline::expired() {
        return sol;
    }
    // ---- Gate: pure LP, plausibly-optimal status, at least one constraint ----
    if !opts.crossover || !prob.p_lower.is_empty() {
        return sol;
    }
    if !matches!(
        sol.status,
        QpStatus::Optimal | QpStatus::OptimalInaccurate | QpStatus::IterationLimit
    ) {
        return sol;
    }
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();
    let m = m_eq + m_ineq;
    if m == 0 {
        // No constraints: the IPM already returns the (bound-clamped)
        // unconstrained optimum; there is no vertex to pivot to.
        return sol;
    }

    // ---- Primary path: revised-simplex LU-basis crossover ----
    // The architecturally-correct engine ([`crate::simplex`]) pivots one
    // variable at a time on an unsymmetric LU basis, walking straight through
    // degeneracy with Bland's rule — so it resolves the highly-degenerate NETLIB
    // GEN vertices (issue #133) that the active-set bridge stalls on from the
    // interior iterate. On any breakdown it returns `None`.
    ATTEMPT.with(|a| *a.borrow_mut() = Some(Attempt::default()));
    let simplex = crate::simplex::crossover_simplex(prob, &sol, opts);
    if crate::deadline::expired() {
        return sol;
    }
    let basis = match simplex {
        Some(v) => {
            let candidate = QpSolution {
                status: QpStatus::Optimal,
                x: v.solution.x,
                y: v.solution.y,
                z: v.solution.z,
                z_lb: v.solution.z_lb,
                z_ub: v.solution.z_ub,
                obj: v.solution.obj,
                iters: sol.iters,
                iterates: sol.iterates.clone(),
            };
            let accepted = never_regresses(prob, &candidate, &sol);
            ATTEMPT.with(|a| {
                if let Some(at) = a.borrow_mut().as_mut() {
                    at.engine = Some("simplex");
                    at.accepted = accepted;
                    at.stats = Some(v.solution.stats);
                }
            });
            if accepted {
                return candidate;
            }
            v.basis
        }
        None => None,
    };

    let native = ActiveSetQp::from_convex(prob);
    let qp = native.problem();
    // ---- Declined vertex, first bridge: the simplex basis, pinned ----
    // The simplex priced out, so its basis is optimal; what the gate declined
    // is the point and duals read off its LU factor. The active-set engine
    // pins the constraints the basis makes active (its nonbasic bounds and
    // rows, plus every equality row, so the set can exceed `n` and be
    // dependent), solves its own KKT system for both, and pivots from there.
    // A pinned KKT that does not factor, or a pinned point outside the
    // feasibility tolerance, is refused (`None`) rather than sent to
    // l1-elastic phase 1.
    if let Some(basis) = basis {
        let working = native.basis_working_set(&basis, m_eq);
        let mut solver = ParametricActiveSetSolver::new(make_backend());
        if let Ok(Some(q)) =
            solver.solve_with_working_set_if_feasible(&qp, &working, &bridge_options())
            && let Some(c) = accept_bridge(prob, &q, &sol)
        {
            return c;
        }
        if crate::deadline::expired() {
            return sol;
        }
        // Not accepted: an optimal basis can still be badly conditioned. The
        // interior iterate's active set often leads elsewhere — to another
        // optimal vertex whose multipliers pass — so it is tried next, as
        // after a breakdown.
    }

    // ---- Bridge from the interior iterate ----
    // Working-set hint: slacks s_i = h_i − (Gx)_i; a row is hinted active when
    // its IPM dual is positive and its slack is ~0. Equality rows are always
    // active; bounds are left Inactive and the engine pivots them in as needed.
    let mut gx = vec![0.0; m_ineq];
    prob.g_mul(&sol.x, &mut gx);
    let mut constraints = Vec::with_capacity(m);
    for _ in 0..m_eq {
        constraints.push(ConsStatus::Equality);
    }
    for i in 0..m_ineq {
        let slack = prob.h[i] - gx[i];
        if sol.z[i] > ACTIVE_Z_TOL && slack.abs() <= ACTIVE_S_TOL {
            constraints.push(ConsStatus::AtUpper);
        } else {
            constraints.push(ConsStatus::Inactive);
        }
    }
    let working = WorkingSet {
        bounds: vec![pounce_qp::BoundStatus::Inactive; n],
        constraints,
    };
    // `solve_with_working_set` factorizes the hinted set for a primal. At a
    // degenerate optimum the hint names more rows than columns; the engine
    // prunes it to an independent subset, and when the pinned point is
    // infeasible it recovers through l1-elastic phase 1. That phase is what
    // finds the better-conditioned vertices, and also what is expensive: its
    // passes — reduced-row and full-row phase 1, the penalty escalations, the
    // recovery solve — each get their own `max_iter`. The solve-wide
    // `max_working_set_changes` in [`bridge_options`] is what bounds the
    // whole. One solve: a cold retry from the origin goes through the same
    // phase 1 and, on the declined vertices measured, rescued none this one
    // missed. After a simplex breakdown the retry is dropped on the same
    // reasoning, not on a measurement: one fixture reaches a breakdown.
    let mut solver = ParametricActiveSetSolver::new(make_backend());
    match solver.solve_with_working_set(&qp, &working, &bridge_options()) {
        Ok(q) => accept_bridge(prob, &q, &sol).unwrap_or(sol),
        Err(_) => sol,
    }
}

/// The settings of every bridge solve. Bland's rule keeps the degenerate
/// pivot loop finite; [`CROSSOVER_MAX_ITER`] caps both each pass and the
/// working-set changes of the whole solve; the time limit is what is left of
/// the caller's.
fn bridge_options() -> ActiveSetOptions {
    ActiveSetOptions {
        time_limit: crate::deadline::remaining(),
        anti_cycling: AntiCyclingChoice::Bland,
        max_iter: CROSSOVER_MAX_ITER,
        max_working_set_changes: Some(CROSSOVER_MAX_ITER),
        ..ActiveSetOptions::default()
    }
}

/// Put a bridge solve to the never-regress gate: `Some` with the vertex in
/// convex coordinates (the sign transform in the module docs) when it is
/// `Optimal` and passes, `None` otherwise. Records its work, and for an
/// `Optimal` one the verdict, either way.
///
/// The candidate is an exact KKT point of the LP (pounce-qp `Optimal`), so by
/// LP convexity its objective is globally optimal. The KKT-error gate is the
/// robust realization of "never return something worse": if the purified
/// vertex's residuals against the ORIGINAL problem exceed the interior
/// iterate's (a sign translation / feasibility surprise), keep the original.
fn accept_bridge(
    prob: &QpProblem,
    qsol: &ActiveSetSolution,
    sol: &QpSolution,
) -> Option<QpSolution> {
    ATTEMPT.with(|a| {
        if let Some(at) = a.borrow_mut().as_mut() {
            at.bridge_working_set_changes += qsol.stats.n_working_set_changes;
        }
    });
    if qsol.status != ActiveSetStatus::Optimal {
        return None;
    }
    let candidate = QpSolution {
        iters: sol.iters,
        iterates: sol.iterates.clone(),
        ..back_translate(prob, qsol)
    };
    let accepted = never_regresses(prob, &candidate, sol);
    ATTEMPT.with(|a| {
        if let Some(at) = a.borrow_mut().as_mut() {
            at.engine = Some("active-set");
            at.accepted = accepted;
        }
    });
    accepted.then_some(candidate)
}

/// The never-regress gate (module docs): `candidate` may replace the interior
/// iterate `sol` only when every component of it is finite and its KKT error
/// against the original problem does not exceed `sol`'s by more than
/// [`REGRESS_SLACK`]. The finiteness test comes first because the error is a
/// max of residuals and `f64::max` drops NaN: an all-NaN candidate reads as
/// zero error. This gate is the verification crossover applies in place of
/// `verify_status` to what [`back_translate`] returns.
fn never_regresses(prob: &QpProblem, candidate: &QpSolution, sol: &QpSolution) -> bool {
    let finite = [
        &candidate.x,
        &candidate.y,
        &candidate.z,
        &candidate.z_lb,
        &candidate.z_ub,
    ]
    .into_iter()
    .all(|v| v.iter().all(|c| c.is_finite()));
    if !finite {
        return false;
    }
    let cand_err = candidate.kkt_residuals(prob).kkt_error();
    let orig_err = sol.kkt_residuals(prob).kkt_error();
    cand_err.is_finite() && cand_err <= orig_err.max(0.0) + REGRESS_SLACK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipm::{QpOptions, solve_qp_ipm};
    use crate::qp::{QpProblem, Triplet};
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    /// An expired budget makes crossover *decline to refine*; it must not cost
    /// the caller the optimum the IPM already computed.
    ///
    /// Deterministic: the solve runs to `Optimal` with no limit, and only then
    /// is `maybe_crossover` entered inside an already-expired deadline scope —
    /// reproducing on demand the race the guard exists for (the deadline
    /// crossing between the IPM converging and crossover starting).
    #[test]
    fn expired_budget_declines_crossover_without_losing_the_verdict() {
        let prob = unique_vertex_lp();
        let sol = solve_qp_ipm(&prob, &opts_on(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "precondition");
        let x = sol.x.clone();
        let obj = sol.obj;

        let out = crate::deadline::with_deadline(Some(std::time::Duration::ZERO), || {
            maybe_crossover(&prob, sol, &opts_on(), &mut backend)
        });
        assert_eq!(
            out.status,
            QpStatus::Optimal,
            "an expired budget restamped a solved LP as unsolved"
        );
        assert_eq!(out.x, x, "declining to refine must not disturb the iterate");
        assert_eq!(out.obj, obj);
    }

    /// LP used across the sign/round-trip tests:
    ///   min −2x₁ − x₂
    ///   s.t. x₁+x₂≤6 (r0), x₁≤4 (r1), x₂≤4 (r2), −x₁≤0 (r3), −x₂≤0 (r4)
    /// Unique optimum x* = (4, 2), f* = −10. Rows 0 and 1 bind. Stationarity
    /// c + z₀(1,1) + z₁(1,0) = 0 ⇒ z₀ = z₁ = 1 (pins the dual sign exactly).
    fn unique_vertex_lp() -> QpProblem {
        QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![-2.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 0, 1.0),
                Triplet::new(2, 1, 1.0),
                Triplet::new(3, 0, -1.0),
                Triplet::new(4, 1, -1.0),
            ],
            h: vec![6.0, 4.0, 4.0, 0.0, 0.0],
            lb: vec![],
            ub: vec![],
        }
    }

    fn opts_on() -> QpOptions {
        // Crossover is off by default (opt-in); these tests exercise the
        // crossover path, so enable it explicitly.
        QpOptions {
            crossover: true,
            ..QpOptions::default()
        }
    }

    fn opts_off() -> QpOptions {
        QpOptions {
            crossover: false,
            ..QpOptions::default()
        }
    }

    /// Translation round-trip and the **sign transform** (R3): purifying the
    /// interior iterate lands the exact vertex with vanishing KKT error,
    /// `z ≥ 0`, and the analytically-pinned duals `z₀ = z₁ = 1`. A flipped
    /// sign would clamp these duals to 0, blow up the stationarity residual,
    /// and the never-regress gate would return the (non-vertex) interior point
    /// instead — so a pass here proves the sign is right.
    #[test]
    fn sign_transform_and_round_trip() {
        let prob = unique_vertex_lp();
        // Interior iterate (crossover disabled), then purify explicitly.
        let interior = solve_qp_ipm(&prob, &opts_off(), backend);
        let mut mk = backend;
        let cand = maybe_crossover(&prob, interior, &opts_on(), &mut mk);

        assert_eq!(cand.status, QpStatus::Optimal);
        assert!((cand.x[0] - 4.0).abs() < 1e-8, "x0 = {}", cand.x[0]);
        assert!((cand.x[1] - 2.0).abs() < 1e-8, "x1 = {}", cand.x[1]);
        assert!((cand.obj + 10.0).abs() < 1e-8, "obj = {}", cand.obj);
        assert!(
            cand.z.iter().all(|&zi| zi >= -1e-12),
            "z must be ≥ 0: {:?}",
            cand.z
        );
        assert!((cand.z[0] - 1.0).abs() < 1e-7, "z0 = {} (sign!)", cand.z[0]);
        assert!((cand.z[1] - 1.0).abs() < 1e-7, "z1 = {} (sign!)", cand.z[1]);
        for i in 2..5 {
            assert!(cand.z[i].abs() < 1e-7, "z{i} = {} should be 0", cand.z[i]);
        }
        assert!(
            cand.kkt_residuals(&prob).kkt_error() < 1e-9,
            "kkt_error = {}",
            cand.kkt_residuals(&prob).kkt_error()
        );
    }

    /// Degenerate LP: the optimum x* = (4, 2) is pinned by THREE binding rows
    /// (more than the 2 variables) — the loss-of-unique-basis case that defeats
    /// a pure IPM. Crossover must still reach the exact vertex.
    #[test]
    fn degenerate_lp_purifies_to_exact_vertex() {
        let mut prob = unique_vertex_lp();
        // Add 2x₁ + x₂ ≤ 10, binding & redundant at (4, 2).
        prob.g.push(Triplet::new(5, 0, 2.0));
        prob.g.push(Triplet::new(5, 1, 1.0));
        prob.h.push(10.0);

        let sol = solve_qp_ipm(&prob, &opts_on(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 4.0).abs() < 1e-8, "x0 = {}", sol.x[0]);
        assert!((sol.x[1] - 2.0).abs() < 1e-8, "x1 = {}", sol.x[1]);
        assert!((sol.obj + 10.0).abs() < 1e-8, "obj = {}", sol.obj);
        assert!(
            sol.z.iter().all(|&zi| zi >= -1e-12),
            "z must be ≥ 0: {:?}",
            sol.z
        );
        assert!(
            sol.kkt_residuals(&prob).kkt_error() < 1e-9,
            "kkt_error = {}",
            sol.kkt_residuals(&prob).kkt_error()
        );
    }

    /// Never-regress: crossover on is never *worse* than crossover off — the
    /// purified result's KKT error and objective do not regress. Checked on the
    /// degenerate LP, where crossover is supposed to help most.
    #[test]
    fn never_regress_vs_crossover_off() {
        let mut prob = unique_vertex_lp();
        prob.g.push(Triplet::new(5, 0, 2.0));
        prob.g.push(Triplet::new(5, 1, 1.0));
        prob.h.push(10.0);

        let off = solve_qp_ipm(&prob, &opts_off(), backend);
        let on = solve_qp_ipm(&prob, &opts_on(), backend);
        let off_err = off.kkt_residuals(&prob).kkt_error();
        let on_err = on.kkt_residuals(&prob).kkt_error();
        assert!(
            on_err <= off_err + REGRESS_SLACK,
            "crossover regressed KKT error: on {on_err} > off {off_err}"
        );
        // Objective must not get worse (this is a minimization).
        assert!(
            on.obj <= off.obj + 1e-7 * (1.0 + off.obj.abs()),
            "crossover regressed objective: on {} > off {}",
            on.obj,
            off.obj
        );
    }

    /// Gate: a genuine QP (`P ≠ 0`) is left untouched — crossover is a no-op,
    /// so the interior optimum (which is generally NOT a vertex) is returned as
    /// is. `min ½‖x‖² s.t. x₁+x₂ ≥ 1` ⇒ x* = (0.5, 0.5).
    #[test]
    fn gate_skips_genuine_qp() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![0.0, 0.0],
            a: vec![],
            b: vec![],
            // −x₁ − x₂ ≤ −1  (i.e. x₁ + x₂ ≥ 1)
            g: vec![Triplet::new(0, 0, -1.0), Triplet::new(0, 1, -1.0)],
            h: vec![-1.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &opts_on(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 0.5).abs() < 1e-6, "x0 = {}", sol.x[0]);
        assert!((sol.x[1] - 0.5).abs() < 1e-6, "x1 = {}", sol.x[1]);
    }

    /// Crossover handles native variable bounds (`lb/ub`) correctly: the bound
    /// duals come back through `lambda_x`. `min −x₁ − x₂ s.t. x₁+x₂≤3, 0≤x≤2`
    /// ⇒ x* = (1, 2) or (2, 1) — pick the unique vertex with c = (−2, −1):
    /// x* = (2, 1) (x₁ at its upper bound, x₁+x₂≤3 binding), f* = −5.
    #[test]
    fn handles_native_variable_bounds() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![-2.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![3.0],
            lb: vec![0.0, 0.0],
            ub: vec![2.0, 2.0],
        };
        let sol = solve_qp_ipm(&prob, &opts_on(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 2.0).abs() < 1e-8, "x0 = {}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-8, "x1 = {}", sol.x[1]);
        assert!((sol.obj + 5.0).abs() < 1e-8, "obj = {}", sol.obj);
        assert!(
            sol.kkt_residuals(&prob).kkt_error() < 1e-8,
            "kkt_error = {}",
            sol.kkt_residuals(&prob).kkt_error()
        );
    }
}
