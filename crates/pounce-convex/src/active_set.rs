//! Active-set QP driver — the [`pounce_qp`] parametric active-set engine
//! wired directly into the convex path, as a peer of [`crate::ipm`].
//!
//! # Why this module exists
//!
//! `solver_selection=qp-active-set` used to reach the active-set engine the
//! long way round: the CLI rewrote it to `algorithm=active-set-sqp` and ran
//! the **full SQP outer loop** over the QP, treating it as a general NLP.
//!
//! That is not mathematically wrong. With an exact Hessian and constraints
//! that are already linear, the first SQP subproblem *is* the original QP, so
//! a healthy run converges in one outer iteration (and did — successful solves
//! reported `n_iter = 1`). But it is an architectural mismatch that forfeits
//! everything the convex path already has, and pays for machinery a QP does
//! not need:
//!
//! * **No presolve.** The SQP path never touches [`crate::presolve`], so the
//!   engine faced every redundant row, fixed variable, and singleton the
//!   IPM never sees. This matters far more for an active-set method than for
//!   an IPM: interior-point iteration counts are nearly problem-size
//!   independent, whereas an active-set pivot count is *combinatorial* in the
//!   size of the active set. Presolve is the difference between a shrunk
//!   problem and one that exhausts its iteration budget.
//! * **No scaling.** No Ruiz equilibration, and `nlp_scaling` is not threaded
//!   through the SQP residuals at all.
//! * **No timing, and misleading statuses**, both of which routed through the
//!   NLP report path rather than the convex one.
//! * **Pointless overhead**: AMPL evaluation callbacks, BFGS storage, and a
//!   filter/line-search globalization, none of which do anything for a
//!   problem whose linearization is exact.
//!
//! This module is the direct route. It hands the QP straight to
//! [`ParametricActiveSetSolver`], inside the same presolve → solve → postsolve
//! wrapper the IPM runs under, so the engine inherits the convex driver's
//! problem reduction, reporting, and status vocabulary.
//!
//! The SQP route is untouched and remains correct for genuine NLPs
//! (`algorithm=active-set-sqp`), where the outer loop is doing real work.
//!
//! # Translation
//!
//! The convex → `pounce-qp` translation (and, critically, the **dual sign
//! transform** on the way back) is the same one [`crate::crossover`] performs;
//! see that module's docs for the derivation. The differences here are:
//!
//! * The Hessian is carried through. Crossover is gated to pure LPs and builds
//!   an empty `H`; a QP driver must translate `p_lower`. Both sides store the
//!   **lower triangle once, 1-based for `pounce-qp` and 0-based for the convex
//!   form**, mirroring off-diagonals implicitly — see
//!   [`QpProblem::p_mul_add`](crate::qp::QpProblem::p_mul_add) against
//!   `pounce_qp`'s `SymTMatrixSpace`, which agree exactly, so the map is a
//!   straight `+1` on both indices with no triangle fixup.
//! * The starting point comes from a **simplex phase-1 feasible vertex**
//!   rather than an IPM iterate (there is no prior iterate here to hint from).
//!   That seed is what makes this path work at all: `pounce-qp`'s own
//!   l1-elastic phase-1 does not terminate on the degenerate netlib-derived
//!   QPs in the Maros-Mészáros set, and handing `solve` a feasible primal
//!   routes it straight to phase-2, which is the part of the method that is
//!   strong. See `crate::simplex::simplex_feasible_vertex`.
//! * Every terminal status is mapped, not just `Optimal` — this is the primary
//!   driver, so it must report `MaxIter` / `NumericalError` / `Infeasible` /
//!   `Unbounded` honestly rather than falling back to a previous solution.
//!
//! The translation itself is [`ActiveSetQp`], owned rather than built in
//! locals, and the read-back is `back_translate`. Both are shared with
//! [`ActiveSetSession`](crate::active_set_session::ActiveSetSession), the
//! persistent form of this driver: it keeps the `(problem, solution)` pair in
//! the engine's coordinates so a *family* of QPs can be traced parametrically
//! instead of solved cold one at a time (gh #769). Everything below is reached
//! by that session rather than restated in it, so a session solve that reuses
//! nothing is this driver, unchanged.

use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_linsol::SparseSymLinearSolverInterface;
// `ActiveSetOverrides` is `pounce-qp`'s: it overlays the `sqp_qp_*` family
// onto a `QpOptions` base, which the SQP subproblem path in
// `pounce-algorithm` needs too, and `pounce-qp` is the only crate both of
// them depend on. Re-exported from this crate's root, so the public path
// callers already use is unchanged.
use pounce_qp::{
    ActiveSetOverrides, BoundStatus, ConsStatus, HessianInertia, ParametricActiveSetSolver,
    QpOptions as ActiveSetOptions, QpProblem as ActiveSetProblem, QpSolver,
    QpStatus as ActiveSetStatus, QpWarmStart, SecondOrderVerdict, WorkingSet,
};

use crate::ipm::{FARKAS_RESID_TOL, QpOptions, dot, finite_or_failed, inf_norm};
use crate::qp::{BoxScreen, QpProblem, QpSolution, QpStatus, screen_variable_box};

/// Clamp a convex lower-bound value to pounce-qp's `±1e19` free convention.
fn to_qp_lower(lb: f64) -> f64 {
    if lb <= NLP_LOWER_BOUND_INF {
        NLP_LOWER_BOUND_INF
    } else {
        lb
    }
}

/// Clamp a convex upper-bound value to pounce-qp's `±1e19` free convention.
fn to_qp_upper(ub: f64) -> f64 {
    if ub >= NLP_UPPER_BOUND_INF {
        NLP_UPPER_BOUND_INF
    } else {
        ub
    }
}

/// A solution carrying no information, returned when the engine reports a
/// status for which no iterate is meaningful. `x` is zero-filled rather than
/// left empty so downstream residual/objective code has the right lengths.
pub(crate) fn empty_solution(n: usize, m_eq: usize, m_ineq: usize, status: QpStatus) -> QpSolution {
    QpSolution {
        status,
        x: vec![0.0; n],
        y: vec![0.0; m_eq],
        z: vec![0.0; m_ineq],
        z_lb: vec![0.0; n],
        z_ub: vec![0.0; n],
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    }
}

/// Solve a **convex** QP with the [`pounce_qp`] parametric active-set engine.
///
/// Signature-compatible with [`crate::ipm::solve_qp_ipm`] so the CLI driver
/// can select between them without restructuring, including the
/// `make_backend` factory (the active-set engine may need more than one
/// backend instance over a solve).
///
/// The caller vouches that `prob.p_lower` is positive semidefinite — that is
/// what [`HessianInertia::Psd`] *claims* to the engine. Use
/// [`solve_qp_active_set_inertia`] for a Hessian that is (or may be)
/// indefinite; claiming PSD for one understates the curvature the engine has
/// to control, and two of the settings this driver picks are chosen on that
/// claim (see there for which).
pub fn solve_qp_active_set<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    engine: &ActiveSetOverrides,
    make_backend: &mut F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    solve_qp_active_set_inertia(prob, opts, engine, HessianInertia::Psd, make_backend)
}

/// [`solve_qp_active_set`], with the caller's claim about the inertia of `P`
/// made explicit — the entry point for a **nonconvex** (indefinite-Hessian) QP.
///
/// The active-set engine handles indefinite Hessians by construction: §4.5
/// inertia control shifts the H block until the reduced KKT factor has the
/// right inertia, so phase-2 descends on a locally convex model and the point
/// it reaches satisfies the first-order conditions of the *original* QP.
///
/// **First-order conditions are not local optimality**, and this doc comment
/// used to say they were — "what it returns for an indefinite `P` is therefore
/// a local solution, exactly as the NLP filter-IPM's `optimal` is local on a
/// nonconvex NLP". The analogy does not hold, and the gap is the whole of
/// gh #848: on an indefinite `P` a **saddle point** satisfies the first-order
/// conditions exactly — vanishing projected gradient, sign-admissible
/// multipliers — and the inertia shift is what walks *to* one rather than what
/// prevents it. Shifting `H` to `H + δI` makes the model locally convex; it
/// does not make the model's stationary point a minimizer of `H`. The engine
/// reported `Optimal` at objective `0` on `min ½xᵀ[[1,5],[5,1]]x` over
/// `[−1,1]²`, whose minimum is `−4`. Begun at `x0 = [0.99, −0.99]`
/// (`f = −3.92`) it still returned `f ≈ 0`, so it moved **uphill** and
/// certified the result.
///
/// Two guards now stand between that and an `Optimal`, and they are not
/// redundant — each covers a class the other cannot see.
///
/// * **Certification, in the engine** (`pounce-qp`'s `negcurv`). At a
///   first-order point it asks whether `ZᵀHZ` is positive definite on
///   `null(A_W)`, and answers with a witness: `d` with `A_W d = 0` checked
///   explicitly and `dᵀHd < 0` evaluated against `H`. It is the guard that can
///   *improve* the answer — the engine escapes along the witness, re-solves,
///   and returns the better point. A witness it cannot get off downgrades the
///   status; a probe that cannot conclude leaves the first-order verdict
///   standing, never claiming a certificate it does not hold.
///   [`verify_status`] is handed the finding through `QpStats::second_order`,
///   because it cannot re-derive it: the returned point is first-order clean
///   by construction. `certify_second_order` turns it off, and it costs
///   nothing under [`HessianInertia::Psd`], where it never runs.
///
/// * **Refutation by exhibition, here** ([`refute_indefinite_optimum`]). This
///   one searches the free set and then unrestrictedly, so it is *not* limited
///   to the working set's null space — and it demotes only after walking the
///   direction and evaluating a strictly better feasible point, so no false
///   demotion is available to it. Where the direction is a feasible recession
///   one it returns the unboundedness certificate instead.
///
/// The second exists because the first has a blind spot with a name: negative
/// curvature behind a **degenerate active bound**. A bound whose multiplier is
/// exactly zero stays in the working set — the drop rule needs a multiplier
/// that violates its sign condition by more than `opt_tol` — and the null space
/// the certification searches is then too small. `pounce-qp`'s `negcurv`
/// module carries the worked example.
///
/// What neither gives is a **global** solution, and seeing past every working
/// set is the NP-hard part of nonconvex QP, which a status check must not
/// pretend to do.
///
/// Two things this driver does for the convex case change under
/// [`HessianInertia::Indefinite`]:
///
/// * **Schur updates are turned off** (as a default — an explicit
///   `sqp_qp_use_schur_updates=yes` still wins). The rank-2 SMW update in
///   `SchurState::apply_change` does not re-check inertia, so a DROP can
///   enlarge the active-set null space and expose negative curvature that the
///   cached factor will not regularize until the next reset. That gap is
///   documented on `pounce_qp::solver` as latent *for indefinite inputs*, and
///   it is latent precisely because nothing had ever fed this driver one. The
///   refactor path runs `factorize_with_inertia_control` every iteration and
///   has no such window. [`engine_options`] applies this, so an external
///   driver assembling the solve out of [`ActiveSetQp`] gets it too.
/// * **`hessian_inertia` stops claiming PSD**, so the l1-elastic
///   reformulation marks its augmented problem indefinite rather than
///   collapsing it to PSD.
///
/// The unboundedness certificate ([`ray_certifies_unbounded`]) needed no
/// switch: it accepts a feasible recession direction of *negative curvature*,
/// which is unreachable for a PSD `P` and is the way a nonconvex QP runs off
/// to `−∞`. Until the second-order test existed, though, nothing ever handed
/// it such a ray — every `Unbounded` the engine produced came from the
/// zero-curvature branch, so the negative-curvature one was written and
/// unreachable (gh #791). The escape is its producer:
/// `crates/pounce-convex/tests/issue848_saddle_not_optimal.rs` reaches it on
/// `min ½(x₀² − x₁²)` with both variables free.
///
/// Not offered on [`ActiveSetSession`](crate::active_set_session::ActiveSetSession):
/// that surface warm-starts one problem from the last, and the homotopy it
/// traces is a predictor built for the convex case. A nonconvex sequence wants
/// its own measurement before it gets an entry point.
pub fn solve_qp_active_set_inertia<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    engine: &ActiveSetOverrides,
    inertia: HessianInertia,
    make_backend: &mut F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    solve_qp_active_set_attempt(prob, opts, engine, inertia, make_backend).sol
}

/// [`solve_qp_active_set`], keeping the `pounce-qp`-side pair the winning
/// attempt produced.
///
/// Exists for [`ActiveSetSession`](crate::active_set_session::ActiveSetSession),
/// which needs `(qp_prev, sol_prev)` in the engine's own coordinates to trace
/// the next problem parametrically from this one (gh #769). The free function
/// above discards it, so a caller that does not warm-start pays nothing beyond
/// the translated problem it was going to build anyway.
pub(crate) fn solve_qp_active_set_attempt<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    engine: &ActiveSetOverrides,
    inertia: HessianInertia,
    make_backend: &mut F,
) -> Attempt
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    crate::deadline::with_deadline(opts.time_limit, || {
        if crate::deadline::expired() {
            return Attempt::opaque(empty_solution(
                prob.n,
                prob.m_eq(),
                prob.m_ineq(),
                QpStatus::TimeLimit,
            ));
        }
        let mut att = solve_qp_active_set_inner(prob, opts, engine, inertia, make_backend);
        // One gate over every exit of the body below — see [`finite_or_failed`].
        att.sol = finite_or_failed(prob, att.sol);
        if crate::deadline::expired() {
            // Shared policy with the IPM path: a deadline crossing observed
            // *after* the inner solve returned relabels a give-up status, but
            // never overwrites a verdict. See [`crate::ipm::mark_timed_out`].
            att.sol = crate::ipm::mark_timed_out(att.sol);
        }
        att
    })
}

/// One trip through the engine: the convex-space answer, and — when the caller
/// is a session that may warm-start from it — the `pounce-qp`-space pair it
/// came from.
///
/// `native` is `None` whenever the pair would be a *lie* about the reported
/// answer: a hard engine error (there is no solution), and the equilibrated
/// retry (whose pair describes the scaled problem, not the one the caller
/// posed). It is deliberately **not** cleared when the solve merely failed —
/// deciding which verdicts are worth tracing from belongs to the session, and
/// it applies that rule in one place
/// ([`ActiveSetSession::remember`](crate::active_set_session::ActiveSetSession)),
/// where the *reported* status is also in hand. The `finite_or_failed` gate
/// above can replace `sol` outright without touching `native`; that same rule
/// is what keeps the mismatch from ever being warm-started from, because a
/// replaced solution never carries a solved status.
pub(crate) struct Attempt {
    pub(crate) sol: QpSolution,
    pub(crate) native: Option<NativeSolve>,
}

impl Attempt {
    /// An answer with no engine pair behind it.
    pub(crate) fn opaque(sol: QpSolution) -> Self {
        Attempt { sol, native: None }
    }
}

/// A `pounce-qp` problem and the solution the engine returned for it — the two
/// arguments [`QpSolver::solve_parametric`] takes as `(qp_prev, sol_prev)`.
pub(crate) struct NativeSolve {
    pub(crate) qp: ActiveSetQp,
    pub(crate) sol: pounce_qp::QpSolution,
}

fn solve_qp_active_set_inner<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    engine: &ActiveSetOverrides,
    inertia: HessianInertia,
    make_backend: &mut F,
) -> Attempt
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    // ---- Screen the variable box before anything else (gh #295, gh #491) ----
    //
    // Peer to the screen [`crate::ipm`] runs at each of its own entry points;
    // see [`screen_variable_box`] for why an empty box must not be left to the
    // engine (on this path it panicked) and why a hairline crossing is repaired
    // rather than rejected.
    let snapped;
    let prob = match screen_variable_box(prob) {
        BoxScreen::Feasible => prob,
        BoxScreen::Empty => {
            return Attempt::opaque(empty_solution(
                prob.n,
                prob.m_eq(),
                prob.m_ineq(),
                QpStatus::PrimalInfeasible,
            ));
        }
        BoxScreen::Snapped(p) => {
            snapped = p;
            &snapped
        }
    };

    // Ruiz equilibration is applied as a **retry after failure**, not
    // unconditionally, because for this engine it is a genuine trade rather
    // than an improvement. Measured on Maros-Mészáros:
    //
    // | problem  | unscaled            | Ruiz-equilibrated   |
    // |----------|---------------------|---------------------|
    // | LOTSCHD  | MaxIter (2.36)      | **Optimal (2398.4)**|
    // | CVXQP1_S | **Optimal (11590.7)**| MaxIter (17220)    |
    // | CVXQP2_S | **Optimal (8120.9)** | MaxIter (9965)     |
    //
    // Scaling rescues the badly-scaled netlib-derived instances and breaks the
    // well-scaled CVXQP family, so *neither* fixed choice dominates. Solving
    // unscaled first and equilibrating only when that fails takes both columns'
    // wins and cannot regress: we reach the retry only after the first attempt
    // already failed to produce a verified KKT point. This mirrors the IPM's
    // own HSDE-then-Ruiz fallback (`solve_qp_ipm_core`) and its reasoning
    // — "there is nothing left to regress".
    let unscaled_opts = QpOptions {
        equilibrate: false,
        ..*opts
    };
    let att = solve_translated(
        prob,
        &unscaled_opts,
        engine,
        inertia,
        make_backend,
        FeasibilityProbe::Allowed,
    );
    if is_conclusive(att.sol.status) {
        return att;
    }
    if crate::deadline::expired() {
        return timed_out(att);
    }

    let mut best = att;
    if opts.equilibrate {
        let (scaled, scaling) = crate::equilibrate::equilibrate(prob);
        let mut retry = solve_translated(
            &scaled,
            &unscaled_opts,
            engine,
            inertia,
            make_backend,
            FeasibilityProbe::Allowed,
        );
        // The pair belongs to the *scaled* problem, so it is not a `(qp_prev,
        // sol_prev)` for anything the caller will pose next; drop it rather
        // than let a session trace a path from coordinates it never asked for.
        retry.native = None;
        scaling.unscale_solution(prob, &mut retry.sol);
        // Re-verify against the ORIGINAL problem: the verdict reached inside the
        // scaled solve certifies a KKT point of the *scaled* QP, and unscaling
        // moves the residuals, so it has to be re-earned here rather than carried
        // over on the strength of the wrong problem's numbers.
        retry.sol.status = reverify_after_unscale(retry.sol.status, &retry.sol, prob, opts);
        // Keep the original failure when the retry also fails, so the reported
        // status describes the attempt made on the problem as the user posed it.
        // `PrimalInfeasible` counts as a win here because `reverify_after_unscale`
        // just re-earned its Farkas certificate against the *original* problem;
        // `DualInfeasible` does not, because its witness (the engine's ray) is not
        // available out here to re-check.
        if is_solved(retry.sol.status) || retry.sol.status == QpStatus::PrimalInfeasible {
            best = retry;
        }
        if is_conclusive(best.sol.status) {
            return best;
        }
        if crate::deadline::expired() {
            return timed_out(best);
        }
    }

    // ---- Last resort: the simplex seed the homotopy displaced (#413) ----
    //
    // Turning the homotopy on also turned the simplex phase-1 vertex seed *off*
    // (see the `seed` binding in `solve_translated`), on the stated premise that
    // the homotopy "is feasible along the whole path" and so is itself the
    // cold-start mechanism. That premise is false — see the path-feasibility
    // guard in `pounce_qp::homotopy` — and when it fails the engine is left with
    // *neither* mechanism: no usable path, and no seed, so it cold-starts the
    // l1-elastic phase-1 that this module's own header records as not
    // terminating on the degenerate netlib-derived QPs in Maros-Mészáros.
    //
    // That is the #413 timeout mode, and it is not a slow corrector. On
    // `QSHARE2B` the seedless route spends its entire iteration budget to return
    // `4854` against a published `11703.7`, still carrying a constraint
    // violation of 20; the seeded route solves it in 52 iterations / 0.03 s.
    //
    // Ordering is load-bearing. This runs *after* the Ruiz retry, not before:
    // both earlier attempts are then bit-identical to what they were, so nothing
    // that used to solve can be displaced — including by the clock, which is the
    // way an added stage can regress a benchmark even when its logic cannot.
    // (Measured: with this stage first, `QPCBOEI1` went from 0.68 s to a 60 s
    // timeout, because the seeded attempt consumed the budget the Ruiz retry
    // needed.) When the caller already asked for `use_homotopy=no`, the first
    // attempt was seeded and there is nothing new to try.
    if engine.use_homotopy != Some(false) {
        if crate::deadline::expired() {
            return timed_out(best);
        }
        let seeded_engine = ActiveSetOverrides {
            use_homotopy: Some(false),
            ..*engine
        };
        let seeded = solve_translated(
            prob,
            &unscaled_opts,
            &seeded_engine,
            inertia,
            make_backend,
            FeasibilityProbe::Allowed,
        );
        if is_solved(seeded.sol.status) || seeded.sol.status == QpStatus::PrimalInfeasible {
            return seeded;
        }
    }

    best
}

/// [`crate::ipm::mark_timed_out`] over an [`Attempt`].
fn timed_out(mut att: Attempt) -> Attempt {
    att.sol = crate::ipm::mark_timed_out(att.sol);
    att
}

/// Did the solve produce a usable, verified KKT point?
pub(crate) fn is_solved(s: QpStatus) -> bool {
    matches!(s, QpStatus::Optimal | QpStatus::OptimalInaccurate)
}

/// Is this verdict final — nothing a second attempt could improve on?
///
/// Either a verified KKT point or a *verified certificate*. Both are earned
/// against the original problem (`verify_status` / `reverify_after_unscale`
/// re-derive every certificate there), so there is no weaker claim here than
/// [`is_solved`] makes: an equilibrated retry cannot overturn a proof, and
/// running one anyway would only burn a second solve before falling back to
/// this same answer.
///
/// `DualInfeasible` is deliberately absent from the retry-acceptance side of
/// this test (see the call site): the unboundedness certificate is re-derived
/// from a ray that only exists inside `solve_translated`, so a retry's claim
/// has no witness left to re-check after unscaling.
pub(crate) fn is_conclusive(s: QpStatus) -> bool {
    is_solved(s)
        || matches!(
            s,
            QpStatus::PrimalInfeasible | QpStatus::DualInfeasible | QpStatus::TimeLimit
        )
}

/// May this attempt spend a second solve on the objective-free feasibility twin
/// to turn an uncertified infeasibility claim into a certified one?
///
/// Exists only to make the recursion finite: the probe re-enters
/// [`solve_translated`] on a problem whose own probe would be the identical
/// solve, so the inner call is handed [`Self::Forbidden`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeasibilityProbe {
    Allowed,
    Forbidden,
}

/// The `pounce-qp` form of a convex [`QpProblem`], owning its storage.
///
/// [`pounce_qp::QpProblem`] borrows every array it is handed — the engine
/// never copies a matrix — so a translation cannot simply *return* one:
/// something has to hold the Hessian, the Jacobian and the four bound vectors
/// for as long as the solve runs. That something used to be a run of locals
/// inside `solve_translated`, which is why the translation was unreachable
/// from outside this module and every frontend wanting the active-set engine
/// over a convex `QpProblem` had to restate it (gh #769). It is also what a
/// *parametric* reuse needs to keep: [`QpSolver::solve_parametric`] takes the
/// previous problem as well as the previous solution, and the previous problem
/// is exactly this.
///
/// The map itself is documented in this module's header — the `+1` on both
/// index arrays with no triangle fixup, the `[A_eq ; G]` row stacking, and the
/// `±1e19` free-bound convention. [`ActiveSetSession`] is the supported way to
/// reach the engine; this type is public so that a caller doing something the
/// session does not cover translates rather than re-derives.
///
/// Translating is only half of it, and shipping only that half was the first
/// review finding on gh #769: a caller who can build the native problem but
/// cannot read its answer back still restates the dual sign transform, the
/// objective reconstruction and the verification gate — the three parts of
/// this that go wrong silently. So the return leg is public too:
/// [`engine_options`] for the settings this path was measured under, and
/// [`back_translate_verified`] (or [`back_translate`] plus [`verify_status`])
/// for the answer.
///
/// [`ActiveSetSession`]: crate::active_set_session::ActiveSetSession
pub struct ActiveSetQp {
    n: usize,
    m: usize,
    h: SymTMatrix,
    g: Vec<f64>,
    a: GenTMatrix,
    bl: Vec<f64>,
    bu: Vec<f64>,
    xl: Vec<f64>,
    xu: Vec<f64>,
    hessian_inertia: HessianInertia,
}

impl ActiveSetQp {
    /// Translate a convex QP into the engine's form.
    ///
    /// Total: every convex problem has a `pounce-qp` image, so this cannot
    /// fail. What it does *not* do is screen the problem — see
    /// [`screen_variable_box`], which the drivers run first because an empty
    /// variable box panicked the engine (gh #295).
    ///
    /// **An external caller runs it too.** The full recipe, which is what
    /// [`solve_qp_active_set`] and [`ActiveSetSession`] do internally:
    ///
    /// 1. [`screen_variable_box`] — `Empty` is a certified `PrimalInfeasible`
    ///    with no solve; `Snapped` replaces the problem with the repaired copy;
    ///    `Feasible` passes through. Skipping this turns an empty box into a
    ///    hard `Err`, and an *impossible* bound into a wrong `Optimal`.
    /// 2. `from_convex` on whatever step 1 handed back, then
    ///    [`engine_options`] and a solve of [`Self::problem`].
    /// 3. [`back_translate_verified`] for the answer.
    ///
    /// [`screen_variable_box`]: crate::screen_variable_box
    /// [`ActiveSetSession`]: crate::active_set_session::ActiveSetSession
    pub fn from_convex(prob: &QpProblem) -> Self {
        let n = prob.n;
        let m_eq = prob.m_eq();
        let m_ineq = prob.m_ineq();
        let m = m_eq + m_ineq;

        // ---- Hessian: lower triangle, 0-based -> 1-based (no triangle fixup) ----
        let mut h_irow = Vec::with_capacity(prob.p_lower.len());
        let mut h_jcol = Vec::with_capacity(prob.p_lower.len());
        let mut h_val = Vec::with_capacity(prob.p_lower.len());
        for t in &prob.p_lower {
            // Defensive: the convex form documents `row >= col`, but a caller
            // that supplied the upper triangle would otherwise be silently
            // transposed into a different matrix. Normalizing costs nothing and
            // makes the translation total.
            let (r, c) = if t.row >= t.col {
                (t.row, t.col)
            } else {
                (t.col, t.row)
            };
            h_irow.push((r + 1) as i32);
            h_jcol.push((c + 1) as i32);
            h_val.push(t.val);
        }
        let h_space = SymTMatrixSpace::new(n as i32, h_irow, h_jcol);
        let mut h = SymTMatrix::new(h_space);
        h.set_values(&h_val);

        // ---- Jacobian A_qp = [A_eq ; G], 1-based ----
        let nnz = prob.a.len() + prob.g.len();
        let mut irows = Vec::with_capacity(nnz);
        let mut jcols = Vec::with_capacity(nnz);
        let mut vals = Vec::with_capacity(nnz);
        for t in &prob.a {
            irows.push((t.row + 1) as i32);
            jcols.push((t.col + 1) as i32);
            vals.push(t.val);
        }
        for t in &prob.g {
            irows.push((m_eq + t.row + 1) as i32);
            jcols.push((t.col + 1) as i32);
            vals.push(t.val);
        }
        let mut a_qp = GenTMatrix::new(GenTMatrixSpace::new(m as i32, n as i32, irows, jcols));
        a_qp.set_values(&vals);

        // ---- Row bounds: eq rows bl=bu=b; ineq rows bl=-inf, bu=h ----
        let mut bl = Vec::with_capacity(m);
        let mut bu = Vec::with_capacity(m);
        for &bk in &prob.b {
            bl.push(bk);
            bu.push(bk);
        }
        for &hi in &prob.h {
            bl.push(NLP_LOWER_BOUND_INF);
            bu.push(to_qp_upper(hi));
        }

        // ---- Variable bounds ----
        let mut xl = Vec::with_capacity(n);
        let mut xu = Vec::with_capacity(n);
        for i in 0..n {
            xl.push(to_qp_lower(prob.lb_of(i)));
            xu.push(to_qp_upper(prob.ub_of(i)));
        }

        Self {
            n,
            m,
            h,
            g: prob.c.clone(),
            a: a_qp,
            bl,
            bu,
            xl,
            xu,
            // The convex claim, which is what every caller of this translation
            // made implicitly before there was a way to say otherwise.
            // [`Self::with_hessian_inertia`] is how a nonconvex QP says so.
            hessian_inertia: HessianInertia::Psd,
        }
    }

    /// Replace the PSD claim [`Self::from_convex`] attaches.
    ///
    /// The claim is not decoration: [`engine_options`] reads it too, and
    /// [`HessianInertia::Indefinite`] turns off the Schur updates whose SMW
    /// path does not re-check inertia. An external driver assembling the solve
    /// out of these pieces must therefore set it here *and* pass the same
    /// value to `engine_options` — which is why that function takes it rather
    /// than defaulting (gh #786). See [`solve_qp_active_set_inertia`] for the
    /// whole argument.
    pub fn with_hessian_inertia(mut self, inertia: HessianInertia) -> Self {
        self.hessian_inertia = inertia;
        self
    }

    /// Borrow the translated data as the engine's problem type.
    ///
    /// Cheap — no copying — so a caller re-borrows per solve rather than
    /// holding one of these across calls and fighting the lifetime.
    pub fn problem(&self) -> ActiveSetProblem<'_> {
        ActiveSetProblem {
            n: self.n,
            m: self.m,
            h: &self.h,
            g: &self.g,
            a: &self.a,
            bl: &self.bl,
            bu: &self.bu,
            xl: &self.xl,
            xu: &self.xu,
            // The caller's claim, carried through rather than asserted here.
            // It was a hard-coded `Psd` on the reasoning that "the convex path
            // only accepts problems the class detector has already ruled
            // convex" — true until `solve_qp_active_set_inertia` gave a
            // nonconvex QP a way in (gh #786), and a false PSD claim is exactly
            // what makes the elastic reformulation solve a differently-shaped
            // problem than the one it was handed.
            hessian_inertia: self.hessian_inertia,
        }
    }

    /// Variables in the translated problem.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Rows in the translated problem — `m_eq + m_ineq` of the convex form,
    /// stacked in that order.
    pub fn m(&self) -> usize {
        self.m
    }

    /// The translated variable bounds, in `pounce-qp`'s `±1e19` convention.
    ///
    /// Crate-visible only: a public caller reads the same two slices off
    /// [`Self::problem`], and one accessor for a thing is enough.
    pub(crate) fn xl(&self) -> &[f64] {
        &self.xl
    }

    /// See [`Self::xl`].
    pub(crate) fn xu(&self) -> &[f64] {
        &self.xu
    }
}

/// The engine settings this driver runs under, before and after the caller's
/// own `sqp_qp_*` overrides.
///
/// Split out of `solve_translated` so the warm parametric path in
/// [`ActiveSetSession`](crate::active_set_session::ActiveSetSession) runs on
/// the *same* configuration as the cold one (gh #769) — a session whose warm
/// and cold legs disagreed about `max_iter` or the Schur updates would be
/// reporting on two different solvers. Public for the third caller the issue
/// is about: an external driver solving [`ActiveSetQp::problem`] directly gets
/// the iteration budget, the Schur-update choice and the homotopy setting this
/// path was measured under, rather than `ActiveSetOptions::default()`.
///
/// `inertia` is the same claim [`ActiveSetQp::with_hessian_inertia`] carries,
/// and it is a parameter rather than a default because one of the settings
/// below turns on it (gh #786). Pass [`HessianInertia::Psd`] for a convex QP —
/// what every caller meant before there was anything else to say.
pub fn engine_options(
    opts: &QpOptions,
    engine: &ActiveSetOverrides,
    n: usize,
    m: usize,
    inertia: HessianInertia,
) -> ActiveSetOptions {
    let qopts = ActiveSetOptions {
        time_limit: crate::deadline::remaining(),
        max_iter: active_set_iter_budget(opts, n, m),
        // Absorb working-set changes as rank-2 Schur updates against a cached
        // factor instead of assembling and factoring a fresh active-set KKT
        // every iteration. This is the lever the wall-clock problem needs: a
        // cold convex-QP solve here runs thousands of iterations (budget
        // `10·(n+m)`), and a per-iteration refactorization is what turns that
        // into a timeout — `QSCTAP1` goes from a 60s timeout to ~2s.
        //
        // Enabling it required fixing two real defects in that path first
        // (`pounce-qp`: singular-Schur recovery, and iterative refinement in
        // `SchurState::solve`); before those it silently lost ~1e-6 of
        // accuracy and aborted outright on singular Schur blocks. See
        // `pounce-qp/tests/schur_vs_refactor.rs`.
        // Off for an indefinite Hessian: `SchurState::apply_change`'s rank-2
        // SMW update does not re-check inertia, so a DROP can enlarge the
        // active-set null space and expose negative curvature the cached
        // factor does not regularize until the next reset. `pounce_qp::solver`
        // documents that gap as latent for indefinite inputs; it was latent
        // because nothing fed this driver one (gh #786). The refactor path
        // runs `factorize_with_inertia_control` every iteration and has no
        // such window.
        use_schur_updates: inertia != HessianInertia::Indefinite,
        // Trace the §4.2 parametric homotopy rather than the conventional
        // phase-1/phase-2 scheme. Measured on the full Maros-Mészáros set at a
        // 120 s cap, same binary: 71/138 correct against 58/138 for the
        // conventional path, with zero solved-but-wrong in both.
        //
        // The trade is not uniform — 20 problems gained, 7 lost, and six of the
        // losses are large instances that previously solved and now hit the
        // cap (AUG2D, AUG2DC, CONT-050, CONT-100, DTOC3, STADAT3): the homotopy
        // is slower there, not wrong. `sqp_qp_use_homotopy=no` restores the old
        // behaviour for such a workload.
        //
        // Set here rather than in `QpOptions::default()` on purpose: that
        // default is read by every `pounce-qp` consumer including the SQP outer
        // loop's inner subproblems, which this benchmark does not cover.
        use_homotopy: true,
        ..ActiveSetOptions::default()
    };
    // Applied last: the two settings this driver picks for itself (the
    // size-scaled `max_iter` and `use_schur_updates`) are defaults, not
    // mandates, so an explicit user request overrides them.
    let mut qopts = qopts;
    engine.apply(&mut qopts);
    qopts
}

/// Translate to `pounce-qp` form, solve, and verify — one attempt, no scaling
/// decisions. `opts.equilibrate` is ignored here; the caller owns that choice.
fn solve_translated<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    engine: &ActiveSetOverrides,
    inertia: HessianInertia,
    make_backend: &mut F,
    probe: FeasibilityProbe,
) -> Attempt
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();
    let m = m_eq + m_ineq;

    let native = ActiveSetQp::from_convex(prob).with_hessian_inertia(inertia);
    let qopts = engine_options(opts, engine, n, m, inertia);

    // Seed from a simplex phase-1 feasible **vertex**, when one is available.
    //
    // `QpSolver::solve` routes a warm start whose primal is feasible directly
    // into phase-2 and never enters l1-elastic mode — which is the point, since
    // that phase-1 does not terminate on the degenerate netlib-derived QPs here
    // (see `simplex::simplex_feasible_vertex`). Phase-2 from a feasible vertex
    // is the part of this method that works well.
    //
    // The working set must come from the simplex **basis**, not from
    // tolerance-snapping the point. `solve_general` trusts the working set it
    // is handed and steps with a zero-RHS active-set system, so:
    //   * a *cold* working set claims nothing is active, and on these low-rank
    //     `P` instances the reduced Hessian is then singular and the engine
    //     certifies a spurious zero-curvature ray (measured: `QSHARE2B`
    //     `-1.2e10`, `QSCAGR7` `-9.9e14`, both `DivergingIterates`);
    //   * snapping every tight row instead names *more than* `n` binding rows
    //     at a degenerate vertex, and no H-block shift repairs a rank-deficient
    //     constraint block (the failure `crossover` documents on the GEN family).
    // The basis avoids both: exactly `n` nonbasic variables, each on a bound,
    // linearly independent because the basis matrix is nonsingular.
    // Skipped entirely when the homotopy is on: the seed exists *because* the
    // parametric homotopy was missing, and the homotopy is itself the cold-start
    // mechanism (it starts from the box relaxation and is feasible along the
    // whole path). Computing one anyway would both pay for a phase-1 solve that
    // is no longer needed and suppress the homotopy, whose hook in
    // `QpSolver::solve` only fires on a genuinely cold call.
    let seed = if qopts.use_homotopy {
        None
    } else {
        crate::simplex::simplex_feasible_vertex(prob).map(|v| {
            use crate::simplex::AtBound;
            let mut working = WorkingSet::cold(n, m);
            for (i, st) in working.bounds.iter_mut().enumerate() {
                // A variable fixed by equal bounds is `Fixed` regardless of which
                // side the basis parked it on.
                let (l, u) = (native.xl()[i], native.xu()[i]);
                let fixed = l > NLP_LOWER_BOUND_INF && u < NLP_UPPER_BOUND_INF && l == u;
                *st = match (fixed, v.struct_at[i]) {
                    (true, _) => BoundStatus::Fixed,
                    (false, AtBound::Lower) => BoundStatus::AtLower,
                    (false, AtBound::Upper) => BoundStatus::AtUpper,
                    (false, AtBound::Free) => BoundStatus::Inactive,
                };
            }
            // Row order matches: the translation above stacks `[A_eq ; G]` in the
            // same order the simplex builds its rows, so index `i` is the same row.
            for (i, st) in working.constraints.iter_mut().enumerate() {
                // Equality rows are ALWAYS active — they are equations, not
                // one-sided constraints that may or may not bind.
                //
                // Deriving their activity from the basis instead (leaving a row
                // inactive when its logical is basic) is tempting: it makes the
                // active count come out at exactly `n` and so avoids the
                // `KKT inertia mismatch: expected 105 …, got 102` that `CVXQP3_S`
                // otherwise hits. But the premise is false. An equality's logical
                // is boxed to `[0, 0]`; a *basic* one is simply a degenerate basic
                // variable sitting at zero, which says nothing about whether the row
                // is a linear combination of the others. Dropping such a row drops a
                // real constraint — the ratio test skips `bl == bu` rows, so it
                // never comes back — and the engine is then free to move in a
                // direction that violates it. On NETLIB `afiro` that produced a
                // phantom `Unbounded` verdict on an LP with a finite optimum of
                // −464.75 (issue #133's regression test caught it).
                //
                // More active rows than variables *is* legitimate at a degenerate
                // vertex; the dependence among them is a rank problem, and
                // `schur_reset_rank_repaired` in `pounce-qp` is what resolves it —
                // by pruning to a maximal independent subset with the algebra to
                // justify each drop, rather than guessing from basis status here.
                *st = if i < m_eq {
                    ConsStatus::Equality
                } else {
                    match v.row_at[i] {
                        // Slack `s = h − Gx` nonbasic at its lower bound 0 means
                        // `Gx = h`: the row sits at its upper bound `bu = h`.
                        AtBound::Lower | AtBound::Upper => ConsStatus::AtUpper,
                        AtBound::Free => ConsStatus::Inactive,
                    }
                };
            }
            QpWarmStart {
                x: v.x,
                lambda_g: vec![0.0; m],
                lambda_x: vec![0.0; n],
                working,
            }
        })
    };

    debug_trace(|| {
        format!(
            "n={n} m_eq={m_eq} m_ineq={m_ineq} seed={} budget={}",
            if seed.is_some() {
                "SIMPLEX-VERTEX"
            } else {
                "NONE (cold)"
            },
            qopts.max_iter,
        )
    });
    let mut solver = ParametricActiveSetSolver::new(make_backend());
    let qsol = match solver.solve(&native.problem(), seed.as_ref(), &qopts) {
        Ok(q) => q,
        // A hard `QpError` (singular factor, dimension mismatch) is a
        // numerical failure, not an infeasibility claim — never assert
        // infeasibility without a certificate.
        //
        // The error text is only surfaced under the debug env var, but it is
        // the *only* record of why the solve died: this arm discards the
        // engine's message and returns a zero-filled solution, so a caller
        // otherwise sees `NumericalFailure` with no cause. Two distinct causes
        // hide here on Maros-Mészáros — `KKT matrix is singular (LICQ
        // violation …)`, the documented rank-detection limitation, and
        // `KKT inertia mismatch`, which was a rank-deficient seed working set.
        Err(e) => {
            debug_trace(|| format!("solver.solve HARD ERROR: {e}"));
            return Attempt::opaque(empty_solution(n, m_eq, m_ineq, QpStatus::NumericalFailure));
        }
    };

    let mut sol = back_translate(prob, &qsol);
    sol.status = verify_status(
        qsol.status,
        qsol.unbounded_ray.as_deref(),
        qsol.stats.second_order,
        &sol,
        prob,
        opts,
    );
    // Two second-order guards, and they cover different classes (gh #848).
    //
    // The engine's own certification is upstream of here and is the one that
    // can *improve* the answer: it works on `null(A_W)` for the working set it
    // stopped at, and where it finds a witness it escapes along it and returns
    // the better point rather than a verdict about the worse one. Its finding
    // arrives through `qsol.stats.second_order`, which `verify_status` cannot
    // re-derive -- the point it is handed is first-order clean by construction.
    //
    // What that check cannot see is negative curvature hidden behind a
    // degenerate active bound: a bound whose multiplier is exactly zero stays
    // in the working set, and the null space searched is then too small
    // (`pounce-qp`'s `negcurv` module carries the worked example). This screen
    // is not restricted to that null space -- it searches the free set and then
    // unrestrictedly -- so that class is exactly what it reaches.
    //
    // It is safe to stack because it never demotes on a *direction*: it walks
    // the direction and evaluates, and demotes only having exhibited a strictly
    // better feasible point. A `Certified` verdict it disagrees with is one
    // where it holds the counterexample.
    if inertia == HessianInertia::Indefinite && sol.status == QpStatus::Optimal {
        if let Some(demoted) = refute_indefinite_optimum(prob, &sol, opts) {
            debug_trace(|| format!("indefinite optimum refuted by exhibition -> {demoted:?}"));
            sol.status = demoted;
        }
    }
    // The engine says infeasible and its own multipliers could not prove it.
    // Before demoting that to "the solver broke", spend one more solve on the
    // objective-free twin, whose multipliers *can* — see [`feasibility_probe`].
    if qsol.status == ActiveSetStatus::Infeasible
        && sol.status == QpStatus::NumericalFailure
        && probe == FeasibilityProbe::Allowed
        && let Some((y, z)) = feasibility_probe(prob, opts, engine, make_backend)
    {
        // Carry the *certifying* multipliers out, replacing the
        // objective-carrying ones that proved nothing. This keeps the invariant
        // every consumer of a `PrimalInfeasible` here relies on — the returned
        // `(y, z)` verify the status attached to them — which is what lets
        // `reverify_after_unscale` re-earn the verdict against the original
        // problem after an equilibrated retry. Without it that re-check tested
        // the wrong vectors and threw away a proof the driver had just made.
        // The bound duals are dropped rather than left mismatched: the
        // certificate is a statement about `(y, z)` and the box, and stale
        // `z_lb`/`z_ub` from a different multiplier set say nothing about it.
        sol.y = y;
        sol.z = z;
        sol.z_lb = vec![0.0; n];
        sol.z_ub = vec![0.0; n];
        sol.status = QpStatus::PrimalInfeasible;
    }
    debug_trace(|| {
        format!(
            "engine={:?} -> reported={:?} kkt_err={:.3e} obj={:.6e}",
            qsol.status,
            sol.status,
            sol.kkt_residuals(prob).kkt_error(),
            sol.obj,
        )
    });
    Attempt {
        sol,
        native: Some(NativeSolve {
            qp: native,
            sol: qsol,
        }),
    }
}

/// Read a `pounce-qp` solution back into convex coordinates — the **dual sign
/// transform** included.
///
/// Split out of `solve_translated` so the warm parametric path reads its
/// answer with the identical code (gh #769), and public for the same reason
/// the forward translation is: the sign transform is the part of this
/// translation that is easy to get subtly wrong and impossible to see
/// afterwards, since a flipped multiplier still looks like a multiplier. An
/// external caller that can build [`ActiveSetQp`] but not read its answer back
/// has to restate exactly that (raised in review of gh #769 by @GermanHeim).
///
/// The returned status is a placeholder — [`verify_status`] decides the
/// verdict, and no caller of this function may skip it.
/// [`back_translate_verified`] is the composition that cannot be
/// half-applied, and is what most callers want.
///
/// The engine's own `obj` is deliberately **not** carried over: the objective
/// is recomputed here in convex coordinates (`½xᵀPx + cᵀx`) so the two forms
/// cannot silently drift apart.
pub fn back_translate(prob: &QpProblem, qsol: &pounce_qp::QpSolution) -> QpSolution {
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();

    // ---- Back-translate (sign transform — see crate::crossover docs) ----
    let mut y = vec![0.0; m_eq];
    if qsol.lambda_g.len() >= m_eq {
        y.copy_from_slice(&qsol.lambda_g[..m_eq]);
    }
    let mut z = vec![0.0; m_ineq];
    for i in 0..m_ineq {
        if let Some(&l) = qsol.lambda_g.get(m_eq + i) {
            z[i] = l.max(0.0);
        }
    }
    let mut z_lb = vec![0.0; n];
    let mut z_ub = vec![0.0; n];
    for i in 0..n {
        if let Some(&l) = qsol.lambda_x.get(i) {
            z_lb[i] = l.max(0.0);
            z_ub[i] = (-l).max(0.0);
        }
    }

    // Objective recomputed in convex coordinates (½xᵀPx + cᵀx) rather than
    // taken from the engine, so the two forms cannot silently drift apart.
    let mut px = vec![0.0; n];
    prob.p_mul(&qsol.x, &mut px);
    let obj = (0..n).map(|i| (0.5 * px[i] + prob.c[i]) * qsol.x[i]).sum();

    QpSolution {
        // Provisional — `verify_status` decides the final verdict.
        status: QpStatus::Optimal,
        x: qsol.x.clone(),
        y,
        z,
        z_lb,
        z_ub,
        obj,
        // The active-set engine counts active-set changes rather than
        // interior-point iterations; that is its analogue of "iterations"
        // and the quantity a user tuning `max_iter` is actually spending.
        iters: qsol.stats.n_working_set_changes as usize,
        iterates: Vec::new(),
    }
}

/// Read a `pounce-qp` solution back into convex coordinates **and decide the
/// verdict** — [`back_translate`] followed by [`verify_status`], plus the two
/// gates every driver here applies afterwards.
///
/// This is the whole of what a caller outside this crate needs after solving
/// [`ActiveSetQp::problem`] itself, and the reason it exists as one call is
/// that the ordering is not optional. `back_translate` returns a *provisional*
/// `Optimal`; a caller who stops there has propagated the engine's claim
/// unchecked, which is the failure `verify_status` documents at length
/// (`QSC205` returns `Optimal` at the wrong objective, `DUALC1` certifies a
/// feasible QP infeasible). Two composed calls with a rule about their order
/// is an API that reads as complete when it is half-applied — so the composition
/// is the supported entry point and the pieces are exported for callers doing
/// something in between.
///
/// The two gates after verification are the driver's, not the engine's:
/// non-finite fields are replaced by an honest failure rather than shipped as
/// numbers, and a deadline crossing observed after the solve returned relabels
/// a give-up status (never a verdict).
///
/// What this does **not** include is the cold driver's second rung on a
/// rejected infeasibility claim — the objective-free `feasibility_probe`, which
/// needs a backend and another solve. A caller wanting that behaviour wants
/// [`solve_qp_active_set`] or [`ActiveSetSession`], which run it.
///
/// [`ActiveSetSession`]: crate::active_set_session::ActiveSetSession
pub fn back_translate_verified(
    prob: &QpProblem,
    qsol: &pounce_qp::QpSolution,
    opts: &QpOptions,
) -> QpSolution {
    let mut sol = back_translate(prob, qsol);
    sol.status = verify_status(
        qsol.status,
        qsol.unbounded_ray.as_deref(),
        qsol.stats.second_order,
        &sol,
        prob,
        opts,
    );
    let sol = finite_or_failed(prob, sol);
    if crate::deadline::expired() {
        crate::ipm::mark_timed_out(sol)
    } else {
        sol
    }
}

/// Re-solve the **objective-free twin** of `prob` — same `A, b, G, h` and the
/// same box, but `P = 0` and `c = 0` — and return *its* multipliers `(y, z)`
/// when they prove the constraint system has no solution.
///
/// # Why a second solve is the right answer here
///
/// [`certifies_primal_infeasible`] fails on an unbounded variable for a
/// structural reason, not a tolerance one: the elastic phase-1 multipliers carry
/// a residual `Aᵀy + Gᵀz = −(Px + c)` left behind by the original objective, and
/// with no finite bound on that variable there is nothing to bound its
/// contribution with. Deleting the objective deletes the residual at the source:
/// the twin's phase-1 minimizes violation *alone*, so its stationarity is
/// `Aᵀy + Gᵀz = 0` to machine precision — a textbook Farkas pair. Measured on
/// the gh #415 LP with its bounds removed: `q = (0, 0)` exactly against
/// `q = (−1, −1)` from the objective-carrying solve.
///
/// The twin has the *same feasible set* as `prob`, so a certificate for it is a
/// certificate for `prob` — which is why the check below is run against `prob`
/// itself and needs no translation back.
///
/// This costs a whole extra solve, so it is deliberately last: only after the
/// engine has already claimed infeasibility *and* the free check on its own
/// multipliers came up empty. A feasible problem reaches it only via a false
/// `Infeasible` claim (the `DUALC1` hazard), and then the twin solve finds the
/// feasible set is non-empty and certifies nothing — the claim stays demoted.
fn feasibility_probe<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    engine: &ActiveSetOverrides,
    make_backend: &mut F,
) -> Option<(Vec<f64>, Vec<f64>)>
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let twin = QpProblem {
        p_lower: Vec::new(),
        c: vec![0.0; prob.n],
        ..prob.clone()
    };
    // The twin drops `P` entirely, so its Hessian is the empty (trivially
    // PSD) one whatever the caller claimed about the original's — `Psd` here
    // is a fact about the problem being solved, not a claim carried over.
    let sol = solve_translated(
        &twin,
        opts,
        engine,
        HessianInertia::Psd,
        make_backend,
        FeasibilityProbe::Forbidden,
    )
    .sol;
    certifies_primal_infeasible(prob, &sol.y, &sol.z, opts).then_some((sol.y, sol.z))
}

/// Emit a diagnostic line when `POUNCE_AS_DEBUG` is set in the environment.
///
/// This path makes three decisions that are invisible in the reported status —
/// whether a simplex seed was obtained, what iteration budget was chosen, and
/// what the engine's own verdict was before verification either accepted or
/// demoted it. Reconstructing those from the outside is guesswork, and the
/// guesses were wrong twice while this driver was being built. The closure
/// defers formatting so an unset var costs one `env::var` and no allocation.
fn debug_trace(msg: impl FnOnce() -> String) {
    if std::env::var("POUNCE_AS_DEBUG").is_ok() {
        eprintln!("[as] {}", msg());
    }
}

/// Re-earn a status against the original problem after unscaling.
///
/// A success carried out of the scaled solve is only evidence about the scaled
/// QP. Rather than trust it, re-measure: keep a clean verdict only if the
/// unscaled point still earns it, and let a solve that lands in the acceptable
/// band say so. Non-success statuses pass through — they made no claim to
/// re-check.
///
/// A `PrimalInfeasible` carried out of the scaled solve is a claim too, and one
/// with the same problem: equilibration is a diagonal congruence, so it
/// preserves *whether* the QP is infeasible but not the certificate's numbers,
/// which `unscale_solution` has moved. So the Farkas certificate is re-derived
/// here against the original data rather than inherited on the strength of the
/// scaled problem's arithmetic.
fn reverify_after_unscale(
    scaled_status: QpStatus,
    sol: &QpSolution,
    prob: &QpProblem,
    opts: &QpOptions,
) -> QpStatus {
    if scaled_status == QpStatus::PrimalInfeasible {
        return if certifies_primal_infeasible(prob, &sol.y, &sol.z, opts) {
            QpStatus::PrimalInfeasible
        } else {
            QpStatus::NumericalFailure
        };
    }
    if !matches!(
        scaled_status,
        QpStatus::Optimal | QpStatus::OptimalInaccurate
    ) {
        return scaled_status;
    }
    solved_band(
        adjudicated_kkt_error(prob, sol, opts.tol, opts.obj_constant),
        opts.tol,
    )
    .unwrap_or(QpStatus::NumericalFailure)
}

/// Multiple of `tol` inside which a solve is downgraded to "acceptable" rather
/// than rejected — the IPM's `1e3·tol` band.
const ACCEPTABLE_FACTOR: f64 = 1e3;

/// Band a KKT error into the reported success statuses: within `tol` is a clean
/// [`QpStatus::Optimal`], within [`ACCEPTABLE_FACTOR`]`·tol` the "solved to
/// acceptable level" tier, anything worse (or non-finite) is not a solve and the
/// caller decides which failure status to report.
fn solved_band(err: f64, tol: f64) -> Option<QpStatus> {
    if !err.is_finite() {
        None
    } else if err <= tol {
        Some(QpStatus::Optimal)
    } else if err <= ACCEPTABLE_FACTOR * tol {
        Some(QpStatus::OptimalInaccurate)
    } else {
        None
    }
}

/// The natural scale of `prob` at the point `sol` — the largest of the term
/// magnitudes that compose the three KKT residuals, and hence the size the
/// finite-precision floor on an *absolute* residual is proportional to.
fn natural_scale(prob: &QpProblem, sol: &QpSolution) -> f64 {
    let mut px = vec![0.0; prob.n];
    prob.p_mul(&sol.x, &mut px);
    // `at_mul`/`gt_mul` accumulate, so a zeroed target yields the product itself.
    let mut aty = vec![0.0; prob.n];
    prob.at_mul(&sol.y, &mut aty);
    let mut gtz = vec![0.0; prob.n];
    prob.gt_mul(&sol.z, &mut gtz);
    inf_norm(&px)
        .max(inf_norm(&prob.c))
        .max(inf_norm(&aty))
        .max(inf_norm(&gtz))
        .max(inf_norm(&sol.z_lb))
        .max(inf_norm(&sol.z_ub))
        .max(inf_norm(&prob.b))
        .max(inf_norm(&prob.h))
        .max(sol.obj.abs())
}

/// The KKT error the status bands are measured against: the **absolute**
/// residual in the original coordinates, with the *stationarity* and
/// *complementarity* terms relaxed to their scale-relative counterparts where
/// the absolute test is unreachable in double precision (gh #641).
///
/// # The defect
///
/// [`QpResiduals::kkt_error`](crate::qp::QpResiduals::kkt_error) is unnormalized
/// — in particular its complementarity term `max|zᵢsᵢ|` carries the magnitude of
/// the problem data twice over. On a QP that is merely *large* rather than
/// ill-conditioned, `sᵢ = hᵢ − gᵢᵀx` cannot be computed to better than
/// `‖data‖·ε` even at an exactly optimal point, so `zᵢsᵢ` floors at
/// `‖z‖·‖data‖·ε`; stationarity floors at `‖P‖·‖x‖·ε` the same way. For
/// `‖data‖ ≳ 1e9` both sit above the default `tol = 1e-8`. Comparing that floor
/// to `tol` labelled a machine-precision-exact solution `OptimalInaccurate` (or
/// `NumericalFailure` at a tightened `tol`) while the convex IPM — the *less*
/// accurate engine on the same instance — reported a clean `Optimal`. That is
/// the active-set analogue of gh #336, whose fix (#337) made the non-symmetric
/// HSDE driver's post-loop adjudication scale-relative and never reached here.
///
/// # Why primal feasibility stays absolute
///
/// The relaxation is deliberately confined to the two dual-side terms. A primal
/// residual is a violation of the user's own constraints, in the user's own
/// units, and it is what the solve report prints as "Constraint violation" —
/// there is no metric in which reporting `Optimal` beside a visible violation is
/// honest. It also does not have the same finite-precision floor: on the
/// reported instance primal infeasibility is `2.2e−16` while complementarity is
/// `1.1e−7`, five decades of daylight, because `‖G‖` is `O(1)` even when `‖P‖`
/// is `1e9`.
///
/// This is not hypothetical. Relaxing the primal term as well — via the same
/// equilibrated normalizer — regressed the `scaled_feasible_a` CLI fixture from
/// an exact solve to `Optimal` at a constraint violation of `7.8e−3`: a row with
/// huge coefficients makes that violation look like `1e−9` *relative to the
/// row*, and the driver then accepted the unscaled attempt instead of falling
/// through to the equilibrated retry that actually solves it. Keeping the primal
/// term absolute leaves that fixture bit-identical.
///
/// # Why it cannot loosen anything else
///
/// * The `err <= tol` short-circuit means every solve that already passes the
///   tight absolute test is untouched, and pays nothing for this check.
/// * [`relative_stop_permitted`](crate::hsde::relative_stop_permitted) — the
///   same gate, and the same crossover `max_scale·ε > tol`, the HSDE stopping
///   test uses — keeps the relative arm shut for well- and moderately-scaled
///   problems, where a `tol`-accurate answer is reachable and demanding it is
///   right. At the default `tol` the crossover is `‖data‖ ≈ 4.5e7`, matching the
///   reported threshold (`K = 1e8` still certifies absolutely, `K = 1e9` does not).
/// * `err.min(..)` means the relative arm can only lower the error, never raise
///   it, so no solve that used to be reported as solved can start failing.
///
/// The relative terms come from
/// [`equilibrated_kkt_rel_parts`](crate::ipm::equilibrated_kkt_rel_parts): each
/// residual over the magnitude of its own terms, measured in the
/// Ruiz-equilibrated metric. The metric is load-bearing rather than incidental —
/// normalizing by *global* ∞-norms in the original coordinates is blind to a
/// spread in the **variable** scales, and gh #414 is a family of QPs where that
/// blindness certifies a badly wrong point as optimal. Equilibration is the
/// diagonal change of variables that removes the spread, so no column can mask
/// another's violation.
fn adjudicated_kkt_error(prob: &QpProblem, sol: &QpSolution, tol: f64, obj_constant: f64) -> f64 {
    let res = sol.kkt_residuals(prob);
    let err = res.kkt_error();
    if !err.is_finite() || err <= tol {
        return err;
    }
    if !crate::hsde::relative_stop_permitted(natural_scale(prob, sol), tol) {
        return err;
    }
    let rel = crate::ipm::equilibrated_kkt_rel_parts(prob, sol, obj_constant);
    err.min(
        res.primal_infeasibility
            .max(rel.dual_infeasibility)
            .max(rel.complementarity),
    )
}

/// Decide the reported status from the engine's verdict **and** the measured
/// KKT residuals of the point it returned.
///
/// The engine's own status is never propagated unchecked, for two reasons the
/// Maros-Mészáros set demonstrates directly:
///
/// * **A claimed `Optimal` can be wrong.** `QSC205` returns `Optimal` with
///   objective `0.0` against a true optimum of `−5.81e−3`. Routed through the
///   SQP outer loop this was caught downstream — the outer loop re-tested the
///   NLP KKT conditions and refused to converge, reporting an iteration limit.
///   Solving directly removes that accidental safety net, so the check has to
///   be made deliberately here. A *silently wrong* answer is the one failure
///   mode worse than not solving: the baseline active-set column had zero
///   solved-but-wrong across all 138 problems, and that property must survive
///   this refactor.
/// * **A claimed `Infeasible` can be wrong.** `DUALC1` is feasible with a
///   published optimum of `6155.25`, and the phase-1 elastic mode certifies it
///   infeasible. An infeasibility verdict is a *proof obligation*: the IPM
///   only reports one behind a verified Farkas certificate, and the SQP path
///   deliberately refuses to assert infeasibility it cannot back (#282). So the
///   claim is not propagated on the engine's word — it is re-derived here from
///   the phase-1 multipliers ([`certifies_primal_infeasible`], with
///   [`feasibility_probe`] as a second attempt) and only reported when it holds.
///   A claim that cannot be backed is downgraded to an honest "did not solve".
/// * **A KKT-clean point can still not be a minimum.** The residual this
///   function bands is a *first-order* one, and on an indefinite `P` a saddle
///   point satisfies the first-order conditions exactly — vanishing projected
///   gradient, sign-admissible multipliers, zero complementarity. So the
///   verification above, applied on its own to
///   [`solve_qp_active_set_inertia`]'s inputs, does not merely fail to catch
///   the saddle: it *promotes* it, overriding whatever non-`Optimal` status
///   the engine assigned after refuting it. That is gh #848, and the reason
///   this function needs `second_order` — the engine's finding cannot be
///   re-derived from the returned point, because the returned point is
///   first-order clean by construction.
///
/// The `Optimal` / `OptimalInaccurate` / fail banding mirrors the IPM's
/// post-loop verdict (`hsde.rs`): within `tol` is clean, within `1e3·tol` is
/// the "acceptable level" tier, anything worse is not a solve. What is banded is
/// [`adjudicated_kkt_error`] rather than the raw `kkt_error()` — see there for
/// why an unnormalized residual cannot be compared to an absolute `tol` on a
/// large-data QP (gh #641).
pub fn verify_status(
    engine: ActiveSetStatus,
    ray: Option<&[f64]>,
    second_order: SecondOrderVerdict,
    sol: &QpSolution,
    prob: &QpProblem,
    opts: &QpOptions,
) -> QpStatus {
    let err = adjudicated_kkt_error(prob, sol, opts.tol, opts.obj_constant);
    // A refuted point is never salvaged by its residual, on any arm.
    //
    // Every promotion in this function runs through `solved_to`, and every one
    // of them reads the *first-order* KKT residual. That residual is exactly
    // what a saddle point of an indefinite `P` satisfies — the engine's own
    // `QpStatus::Optimal` at the saddle was the gh #848 defect, and re-deriving
    // the same conditions here reproduces it rather than catching it. So when
    // the engine hands back a witness direction of negative curvature, the band
    // is closed: `solved_to` returns `None` and each arm falls to its own
    // honest non-verdict (`MaxIter` -> `IterationLimit`, `TimeLimit` ->
    // `TimeLimit`, the rest -> `NumericalFailure`).
    //
    // The `Unbounded` arm's certificate check is deliberately *upstream* of
    // this: a ray that stands up is a proof about the model, and negative
    // curvature along a recession direction is the very thing it proves
    // (`ray_certifies_unbounded`'s `dᵀPd < 0` branch, gh #791). Refusing that
    // one would discard the strongest verdict available.
    let refuted = second_order == SecondOrderVerdict::NegativeCurvature;
    let solved_to = |e: f64| {
        if refuted {
            None
        } else {
            solved_band(e, opts.tol)
        }
    };

    match engine {
        // Trust, but verify.
        ActiveSetStatus::Optimal => solved_to(err).unwrap_or(QpStatus::NumericalFailure),
        // Unboundedness is a proof obligation too, and the engine's claim does
        // not survive contact with this benchmark: `QSCSD1` (published optimum
        // 8.667, plainly bounded) is reported `Unbounded`. Propagating that
        // unchecked produced "Problem is unbounded (dual infeasible)" on a
        // bounded QP — a *wrong verdict about the user's model*, which is worse
        // than any failure status. So re-derive the certificate here from the
        // returned ray, exactly as the `Infeasible` case is re-derived.
        ActiveSetStatus::Unbounded => match ray {
            Some(d) if ray_certifies_unbounded(prob, d) => QpStatus::DualInfeasible,
            // No ray, or a ray that does not stand up: fall back on the point.
            _ => solved_to(err).unwrap_or(QpStatus::NumericalFailure),
        },
        // Infeasibility is a proof obligation, so the claim is re-derived from
        // the phase-1 multipliers exactly as the `Unbounded` claim is re-derived
        // from the ray. A certificate that stands up is the *right* verdict about
        // the user's model and must be reported as such (gh #415: without this,
        // a two-row LP whose contradiction is visible by inspection came back as
        // `InternalError` / `solve_result_num=500` — "the solver broke, retry" —
        // where every other engine said `200`, "your model is infeasible").
        // A claim that does not stand up is still downgraded: if the returned
        // point satisfies the KKT conditions anyway, report that; otherwise say
        // honestly that we did not solve it.
        ActiveSetStatus::Infeasible => {
            if certifies_primal_infeasible(prob, &sol.y, &sol.z, opts) {
                QpStatus::PrimalInfeasible
            } else {
                solved_to(err).unwrap_or(QpStatus::NumericalFailure)
            }
        }
        // Budget or breakdown: the point may still be good enough to report at
        // the acceptable tier (the IPM salvages solves this way), but it is
        // never promoted to a clean `Optimal`.
        ActiveSetStatus::MaxIter => solved_to(err).unwrap_or(QpStatus::IterationLimit),
        // The wall clock is a budget like the iteration cap, and is salvaged the
        // same way: `solved_to` re-derives the KKT error against the original
        // problem, so a cancelled solve that had in fact already reached the
        // answer reports it. Discarding the point because of *when* the engine
        // stopped — while keeping the identical point when it stopped for
        // running out of iterations — is the same verdict erasure
        // [`crate::ipm::mark_timed_out`] exists to prevent, one layer down.
        ActiveSetStatus::TimeLimit => solved_to(err).unwrap_or(QpStatus::TimeLimit),
        ActiveSetStatus::NumericalError => solved_to(err).unwrap_or(QpStatus::NumericalFailure),
    }
}

/// Does `d` actually certify that `prob` is unbounded below?
///
/// `d` must first be a direction the feasible set recedes along — all three of:
///
/// * **equalities preserved**, `Ad ≈ 0`;
/// * **inequalities non-increasing**, `Gd ≤ 0`, so no row is eventually violated;
/// * **box respected directionally** — a component may only move toward an
///   *infinite* bound;
///
/// so that `x + td` stays feasible for every `t ≥ 0`. Along such a `d`,
/// `f(x + td) = f(x) + t·(Px + c)ᵀd + ½t²·dᵀPd`, and the objective decreases
/// without bound in exactly two cases:
///
/// * **negative curvature**, `dᵀPd < 0` — the `t²` term dominates whatever the
///   slope is. Unreachable for a PSD `P`, so this branch belongs to the
///   indefinite Hessians [`solve_qp_active_set_inertia`] admits (gh #786);
/// * **zero curvature and strict descent**, `Pd ≈ 0` with `(Px + c)ᵀd < 0`,
///   which reduces to `cᵀd < 0` independently of `x`. Zero curvature is a
///   requirement of *this* branch, not of the test as a whole: a direction the
///   objective curves up along turns back up however steep its slope.
///
/// Anything less is not a certificate.
///
/// Tolerances are relative to `‖d‖∞` because the ray is not normalized; the
/// curvature test normalizes by `‖d‖∞²` instead, so it can be compared against
/// the scale of `P`.
/// Second-order screen on a claimed optimum of an **indefinite** QP, by
/// **exhibiting a strictly better feasible point** (gh #848).
///
/// # What the engine actually proves
///
/// [`verify_status`] re-derives *first-order* KKT residuals. For a convex QP
/// first-order KKT is equivalent to global optimality, so that guard was sound
/// for every input the engine had ever been given. gh #786 then admitted
/// indefinite Hessians on the premise that the engine returns "a local
/// optimum, the same guarantee the NLP filter-IPM gives on a nonconvex NLP".
/// For an indefinite `P` first-order KKT is necessary and **not sufficient**,
/// no second-order test was added, and the result was a confident wrong
/// answer where `v0.10.0` had refused the class outright:
///
/// ```text
///   P = [[1, 5], [5, 1]], c = 0, box [-1, 1]^2,  eig(P) = [-4, 6]
///   qp-active-set -> Optimal, x = [0, 0], f = 0
///   but x = [1, -1] is feasible with f = -4
/// ```
///
/// The start point was ignored entirely — begun at `x0 = [0.99, -0.99]`
/// (`f = -3.92`) the engine still returned `f ≈ 0` and certified it, so it
/// moved **uphill** and reported success. Across 40 random indefinite box QPs,
/// 30 returned a point beaten by an explicitly exhibited feasible point and 23
/// were not even local minima.
///
/// The engine's inertia control is not the missing test and must not be read
/// as it: it shifts the KKT diagonal so the *factorization* has the right
/// inertia, which makes the linear algebra work at each iteration. It says
/// nothing about the curvature of `P` on the feasible directions at the point
/// finally returned.
///
/// # Why this refutes by exhibition rather than by a cone argument
///
/// The second-order necessary condition lives on the *critical cone*, and
/// deciding copositivity on a cone is not something to attempt inside a status
/// check. So this does not try. It looks for a direction of negative curvature,
/// then **walks along it and evaluates the objective**: a feasible point with a
/// strictly lower objective is a refutation that needs no theory at all, and is
/// the first of the four oracles the issue offers.
///
/// The consequence that makes this design safe is the important one: the search
/// for a direction can be as heuristic as it likes. A direction it misses
/// leaves the verdict where it already was; a direction it finds is only ever
/// acted on after the walk has *proved* the point beatable. **There is no false
/// demotion available to it.** That is why the free-set restriction below is
/// allowed to ignore the general rows — the step length accounts for every
/// bound and every row, so a direction that leaves the feasible region
/// immediately yields a zero step and no refutation.
///
/// # The two verdicts
///
/// A negative-curvature direction that is also a feasible **recession**
/// direction is not merely a better point, it is a certificate of
/// unboundedness, and [`ray_certifies_unbounded`] already knows how to say so.
/// That branch existed and could never fire: its only call site is the
/// `Unbounded` arm of [`verify_status`], and on `min −x₀² + ½x₁², x₀ ≥ 0` the
/// engine claims `Optimal` (`obj = 0`, `iters = 0`), so the arm that would
/// consult it is never reached. Its unit tests call it directly and stayed
/// green throughout. Reaching it from the `Optimal` side is what turns that
/// model from `Solve_Succeeded` into the `Diverging_Iterates` the NLP arm
/// reports on the same binary.
///
/// Anything else beatable is demoted to [`QpStatus::NumericalFailure`] — the
/// same "we did not solve it" this function's neighbours already use when a
/// claim does not survive re-derivation. It is not a satisfying verdict, but a
/// point the solver can itself prove is not optimal must not go out labelled
/// `Optimal`.
///
/// Returns `None` when nothing was refuted, which leaves the first-order
/// verdict standing.
fn refute_indefinite_optimum(
    prob: &QpProblem,
    sol: &QpSolution,
    opts: &QpOptions,
) -> Option<QpStatus> {
    let n = prob.n;
    if n == 0 || sol.x.len() != n || !sol.x.iter().all(|v| v.is_finite()) {
        return None;
    }
    let p_scale = prob
        .p_lower
        .iter()
        .fold(0.0_f64, |a, t| a.max(t.val.abs()))
        .max(1.0);

    // Two restrictions of the curvature search, and both are needed.
    //
    // The *interior* one -- coordinates strictly inside their bounds -- finds
    // the curvature a saddle in the middle of the box carries, which is the
    // reported `P = [[1, 5], [5, 1]]` case.
    //
    // The *unrestricted* one is what reaches a coordinate sitting **on** a
    // bound that it may still leave. On `min −x₀² + ½x₁², x₀ ≥ 0` the engine
    // returns `x = 0` with `iters = 0`; `x₀` is at its bound, so the interior
    // search sees only `P₁₁ = 1 > 0` and finds nothing, while the whole defect
    // lives along `+x₀`. The bound is *weakly* active there (the gradient
    // vanishes, so its multiplier is zero) and the direction leaves it into
    // the feasible region.
    //
    // Widening the search cannot widen what is *accepted*: a direction that
    // points out of an active bound gets a zero step from
    // [`max_feasible_step`] and refutes nothing, and both signs are tried, so
    // the inward half of a one-sided cone is always among the candidates.
    let margin = |v: f64| 1e-9 * (1.0 + v.abs());
    let free: Vec<bool> = (0..n)
        .map(|i| {
            let (lo, hi) = (prob.lb_of(i), prob.ub_of(i));
            sol.x[i] > lo + margin(lo) && sol.x[i] < hi - margin(hi)
        })
        .collect();
    let all = vec![true; n];
    let mut candidates: Vec<Vec<f64>> = Vec::with_capacity(2);
    if free.iter().any(|&f| f) {
        candidates.extend(most_negative_curvature_direction(prob, &free, p_scale));
    }
    if !free.iter().all(|&f| f) {
        candidates.extend(most_negative_curvature_direction(prob, &all, p_scale));
    }
    if candidates.is_empty() {
        return None;
    }

    let f_at = |x: &[f64]| {
        let mut px = vec![0.0; n];
        prob.p_mul(x, &mut px);
        0.5 * (0..n).map(|i| x[i] * px[i]).sum::<f64>()
            + (0..n).map(|i| prob.c[i] * x[i]).sum::<f64>()
    };
    let f_cur = f_at(&sol.x);
    if !f_cur.is_finite() {
        return None;
    }

    for (d, sign) in candidates.iter().flat_map(|d| [(d, 1.0_f64), (d, -1.0)]) {
        let dir: Vec<f64> = d.iter().map(|v| sign * v).collect();
        // A feasible recession direction of negative curvature is the
        // unboundedness certificate, not merely a better point.
        if ray_certifies_unbounded(prob, &dir) {
            return Some(QpStatus::DualInfeasible);
        }
        let Some(alpha) = max_feasible_step(prob, &sol.x, &dir) else {
            continue;
        };
        if !(alpha > 0.0) {
            continue;
        }
        let trial: Vec<f64> = (0..n).map(|i| sol.x[i] + alpha * dir[i]).collect();
        let f_new = f_at(&trial);
        // Strictly better by more than the solve's own tolerance, measured
        // relative to the objective's own scale so this cannot fire on
        // rounding at a genuine optimum.
        if f_new.is_finite() && f_new < f_cur - opts.tol * (1.0 + f_cur.abs()) {
            return Some(QpStatus::NumericalFailure);
        }
    }
    None
}

/// A direction of negative curvature for `P` restricted to the free
/// coordinates, or `None` if none was found.
///
/// Shifted power iteration: the dominant eigenvector of `(σI − P_F)` with
/// `σ` a Gershgorin upper bound on `λ_max(P_F)` is the eigenvector of
/// `λ_min(P_F)`. Matrix-free — one `p_mul` per iteration — so this costs
/// `O(nnz)` per step and carries no dimension ceiling, which matters:
/// a ceiling that silently skips the check is the shape of the companion
/// defect on `check_psd`.
///
/// Heuristic on purpose. It can miss a direction, and missing leaves the
/// verdict exactly where it was; it cannot cause a wrong demotion, because
/// [`refute_indefinite_optimum`] acts only after walking the direction and
/// finding a strictly better feasible point.
fn most_negative_curvature_direction(
    prob: &QpProblem,
    free: &[bool],
    p_scale: f64,
) -> Option<Vec<f64>> {
    let n = prob.n;
    // Gershgorin bound over the free block. `p_lower` stores the lower
    // triangle, so an off-diagonal entry contributes to both of its rows.
    let mut diag = vec![0.0; n];
    let mut off = vec![0.0; n];
    for t in &prob.p_lower {
        if !free[t.row] || !free[t.col] {
            continue;
        }
        if t.row == t.col {
            diag[t.row] += t.val;
        } else {
            off[t.row] += t.val.abs();
            off[t.col] += t.val.abs();
        }
    }
    let shift = (0..n)
        .filter(|&i| free[i])
        .fold(0.0_f64, |a, i| a.max(diag[i] + off[i]))
        .max(p_scale);

    // Deterministic start: no `rand` dependency, and a fixed vector makes the
    // verdict reproducible run to run. Mixed signs so it is not orthogonal to
    // the eigenvector on a symmetric fixture -- `[1, 1, ...]` is exactly the
    // wrong choice for `P = [[1, 5], [5, 1]]`, whose negative eigenvector is
    // `[1, -1]`.
    let mut v: Vec<f64> = (0..n)
        .map(|i| {
            if free[i] {
                let k = (i % 7) as f64;
                (1.0 + k) * if i % 2 == 0 { 1.0 } else { -1.0 }
            } else {
                0.0
            }
        })
        .collect();
    let mut pv = vec![0.0; n];
    let mut best: Option<(f64, Vec<f64>)> = None;
    for _ in 0..64 {
        let nrm = v.iter().fold(0.0_f64, |a, x| a + x * x).sqrt();
        if !(nrm > 0.0) || !nrm.is_finite() {
            break;
        }
        for x in v.iter_mut() {
            *x /= nrm;
        }
        pv.iter_mut().for_each(|x| *x = 0.0);
        prob.p_mul(&v, &mut pv);
        let q: f64 = (0..n).filter(|&i| free[i]).map(|i| v[i] * pv[i]).sum();
        if best.as_ref().is_none_or(|(bq, _)| q < *bq) {
            best = Some((q, v.clone()));
        }
        // v <- (shift I - P_F) v, kept inside the free block.
        for i in 0..n {
            v[i] = if free[i] { shift * v[i] - pv[i] } else { 0.0 };
        }
    }
    let (q, vec) = best?;
    // Comparable to `P`'s own scale, and far above the rounding floor of a
    // genuinely zero curvature -- the same shape `ray_certifies_unbounded`
    // uses for its own curvature test.
    (q < -1e-8 * p_scale).then_some(vec)
}

/// How far `x` can move along `d` before leaving the feasible region, or
/// `None` when it cannot move at all (an equality row the direction does not
/// respect, or a row already tight that the direction points out of).
///
/// `Some(f64::INFINITY)` means nothing blocks the direction.
fn max_feasible_step(prob: &QpProblem, x: &[f64], d: &[f64]) -> Option<f64> {
    let n = prob.n;
    let dn = d.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if !(dn > 0.0) || !dn.is_finite() {
        return None;
    }
    let slack = 1e-9 * dn;
    // Equalities must be respected exactly; there is no step along a direction
    // that moves off the affine set.
    let mut ad = vec![0.0; prob.m_eq()];
    for t in &prob.a {
        ad[t.row] += t.val * d[t.col];
    }
    if ad.iter().any(|v| v.abs() > slack) {
        return None;
    }

    let mut alpha = f64::INFINITY;
    for i in 0..n {
        let (lo, hi) = (prob.lb_of(i), prob.ub_of(i));
        if d[i] > slack && hi < crate::qp::POS_INF {
            alpha = alpha.min(((hi - x[i]) / d[i]).max(0.0));
        }
        if d[i] < -slack && lo > crate::qp::NEG_INF {
            alpha = alpha.min(((lo - x[i]) / d[i]).max(0.0));
        }
    }
    let mut gx = vec![0.0; prob.m_ineq()];
    let mut gd = vec![0.0; prob.m_ineq()];
    prob.g_mul(x, &mut gx);
    for t in &prob.g {
        gd[t.row] += t.val * d[t.col];
    }
    for j in 0..prob.m_ineq() {
        if gd[j] > slack {
            alpha = alpha.min(((prob.h[j] - gx[j]) / gd[j]).max(0.0));
        }
    }
    Some(alpha)
}

fn ray_certifies_unbounded(prob: &QpProblem, d: &[f64]) -> bool {
    if d.len() != prob.n {
        return false;
    }
    let dn = d.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if !dn.is_finite() || dn == 0.0 {
        return false;
    }
    // Scale-relative slack. Deliberately looser than `tol`: the ray comes out of
    // a finite-precision null-space computation, so demanding `tol` here would
    // reject genuine certificates.
    let slack = 1e-7 * dn;

    let mut pd = vec![0.0; prob.n];
    prob.p_mul(d, &mut pd);

    let mut ad = vec![0.0; prob.m_eq()];
    for t in &prob.a {
        ad[t.row] += t.val * d[t.col];
    }
    if ad.iter().any(|v| v.abs() > slack) {
        return false;
    }

    let mut gd = vec![0.0; prob.m_ineq()];
    for t in &prob.g {
        gd[t.row] += t.val * d[t.col];
    }
    if gd.iter().any(|&v| v > slack) {
        return false;
    }

    for (i, &di) in d.iter().enumerate() {
        if di < -slack && prob.lb_of(i) > crate::qp::NEG_INF {
            return false;
        }
        if di > slack && prob.ub_of(i) < crate::qp::POS_INF {
            return false;
        }
    }

    // `d` is a feasible recession direction. Along it,
    // `f(x + td) = f(x) + t·(Px + c)ᵀd + ½t²·dᵀPd`, so it certifies
    // unboundedness in exactly two ways.
    //
    // **Negative curvature** — `dᵀPd < 0` — sends the `t²` term to `−∞` on its
    // own, whatever the slope does. This is unreachable for a PSD `P` (and the
    // threshold below is far above the rounding floor of one), so it fires only
    // on the indefinite Hessians `solve_qp_active_set_inertia` admits; it is
    // how a nonconvex QP runs off to `−∞`, and the branch is why that entry
    // point needed no separate certificate. Measured on the *normalized* `u =
    // d/‖d‖∞` so the quantity carries the units of `P` and can be compared
    // against `P`'s own scale — the same shape as the frontend's `check_psd`
    // tolerance.
    let curvature: f64 = (0..prob.n).map(|i| d[i] * pd[i]).sum::<f64>() / (dn * dn);
    let p_scale = prob
        .p_lower
        .iter()
        .fold(0.0_f64, |a, t| a.max(t.val.abs()))
        .max(1.0);
    if curvature < -1e-8 * p_scale {
        return true;
    }

    // **Strict descent along a zero-curvature direction.** Here the `t²` term
    // must vanish — a direction the objective curves *up* along turns back up
    // however steep its slope — so `Pd ≈ 0` is a requirement of this branch,
    // not a precondition of the whole test. With it, the directional derivative
    // `(Px + c)ᵀd` reduces to `cᵀd` independently of `x`.
    if pd.iter().any(|v| v.abs() > slack) {
        return false;
    }
    let slope: f64 = (0..prob.n).map(|i| prob.c[i] * d[i]).sum();
    slope < -slack
}

/// Do the multipliers `(y, z ≥ 0)` prove that `prob` has **no** feasible point?
///
/// # Why not the textbook Farkas test
///
/// The classic certificate for `{x : Ax = b, Gx ≤ h}` is `(y, z ≥ 0)` with
/// `Aᵀy + Gᵀz = 0` and `bᵀy + hᵀz < 0`, and that is what the IPM checks
/// ([`crate::ipm::detect_infeasibility`]). Applied to *these* multipliers it
/// rejects every genuine certificate the active-set engine produces, because of
/// where they come from: the l1-elastic phase-1 minimizes
/// `½xᵀPx + cᵀx + γ·(violation)`, so its stationarity condition is
/// `Px + c + Aᵀy + Gᵀz = 0` — the original objective is still in there, and
/// `q := Aᵀy + Gᵀz` settles at `−(Px + c)`, not at `0`. That residual does not
/// shrink as phase-1 converges; it only shrinks *relative to* `‖(y,z)‖ ∝ γ`.
/// On the two-row LP of gh #415 the relative residual is `1e-6` against a
/// `FARKAS_RESID_TOL` of `1e-10`, so the certificate — which is real — reads as
/// noise, and the engine's correct `Infeasible` verdict was thrown away as a
/// numerical failure.
///
/// # The test used instead
///
/// For **any** feasible `x` and any `(y, z ≥ 0)`, `yᵀ(Ax − b) = 0` and
/// `zᵀ(Gx − h) ≤ 0`, so
///
/// ```text
///     qᵀx  ≤  bᵀy + hᵀz  =:  v          where q := Aᵀy + Gᵀz
/// ```
///
/// A feasible `x` also lies in the box, so `qᵀx ≥ L := min_{lb ≤ x ≤ ub} qᵀx`,
/// which is separable: `L = Σᵢ min(qᵢ·lbᵢ, qᵢ·ubᵢ)`. Therefore
///
/// ```text
///     L > v   ⟹   no feasible point exists.
/// ```
///
/// This is a strict generalization of the Farkas test — with no finite bounds
/// it *is* the Farkas test (`L` is finite only if `q ≡ 0`, and then `L = 0 > v`
/// is exactly `v < 0`) — but on a boxed problem it does not merely *tolerate*
/// the `−(Px + c)` residual, it accounts for it exactly. The bound multipliers
/// `z_lb`/`z_ub` are deliberately not used: minimizing over the box is the same
/// deduction done optimally, and it cannot be thrown off by a phase-1 iterate's
/// noisy bound duals.
///
/// A variable whose *binding* side is infinite makes `L = −∞` and there is
/// nothing to deduce — unless its `qᵢ` is negligible, which is the one place
/// this falls back on [`FARKAS_RESID_TOL`]'s tolerance argument rather than an
/// exact bound.
///
/// Soundness is the property that matters here: a false positive is a wrong
/// verdict about the user's model, which is worse than any failure status (the
/// `DUALC1` hazard this module's docs describe). Every step above is an
/// inequality that holds for *every* feasible point, so a pass is a proof up to
/// floating point, and the `ctol` margin covers that.
fn certifies_primal_infeasible(prob: &QpProblem, y: &[f64], z: &[f64], opts: &QpOptions) -> bool {
    if y.len() != prob.m_eq() || z.len() != prob.m_ineq() {
        return false;
    }
    let dual_norm = inf_norm(y).max(inf_norm(z));
    if !dual_norm.is_finite() || dual_norm == 0.0 {
        return false;
    }
    // `z ≥ 0` is what makes `zᵀ(Gx − h) ≤ 0`; without it there is no deduction.
    let ctol = opts.infeas_tol;
    if z.iter().any(|&zi| zi < -ctol * dual_norm) {
        return false;
    }

    let mut q = vec![0.0; prob.n]; // q = Aᵀy + Gᵀz
    prob.at_mul(y, &mut q);
    prob.gt_mul(z, &mut q);
    let v = dot(&prob.b, y) + dot(&prob.h, z); // v = bᵀy + hᵀz
    if !v.is_finite() {
        return false;
    }

    // L = min over the box of qᵀx, separably.
    let resid_slack = FARKAS_RESID_TOL * dual_norm;
    let mut l = 0.0_f64;
    for (i, &qi) in q.iter().enumerate() {
        if !qi.is_finite() {
            return false;
        }
        if qi == 0.0 {
            continue;
        }
        // The minimizing corner: push `xᵢ` to its lower bound when `qᵢ > 0`, to
        // its upper bound when `qᵢ < 0`.
        let bound = if qi > 0.0 {
            prob.lb_of(i)
        } else {
            prob.ub_of(i)
        };
        if bound > crate::qp::NEG_INF && bound < crate::qp::POS_INF {
            l += qi * bound;
        } else if qi.abs() > resid_slack {
            return false; // qᵀx is unbounded below on the box — no deduction
        }
    }
    if !l.is_finite() {
        return false;
    }

    // Margin against floating-point noise, relative to the magnitudes actually
    // compared. Scaling by `dual_norm` as well keeps a certificate whose two
    // sides are both ~0 next to huge multipliers from passing on rounding.
    let mag = dual_norm.max(v.abs()).max(l.abs());
    l - v > ctol * mag
}

/// Iteration budget for the active-set engine.
///
/// The engine's own default is a flat 200, which is the IPM's budget and the
/// wrong shape for this method: an interior-point iteration count is nearly
/// independent of problem size, whereas an active-set method needs roughly one
/// iteration per active-set change, so its count grows with `n + m`. On the
/// Maros-Mészáros set the flat cap was the single largest failure class —
/// `DUALC1` (n = 9, m = 215) exhausts it and then solves *exactly*, in one
/// outer iteration, once the budget is raised.
///
/// An explicit user `max_iter` always wins; otherwise scale with the problem
/// and keep 200 as a floor for tiny instances.
fn active_set_iter_budget(opts: &QpOptions, n: usize, m: usize) -> u32 {
    const DEFAULT_IPM_MAX_ITER: usize = 200;
    const PER_DIM: usize = 10;
    if opts.max_iter != DEFAULT_IPM_MAX_ITER {
        // User set it explicitly — respect it.
        return opts.max_iter.min(u32::MAX as usize) as u32;
    }
    let scaled = n.saturating_add(m).saturating_mul(PER_DIM);
    scaled.clamp(200, 200_000) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipm::solve_qp_ipm;
    use crate::qp::CROSSED_BOX_TOL;
    use crate::qp::{POS_INF, Triplet};
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    /// `min (x₀−3)² + (x₁−2)²  s.t.  x₀ + x₁ ≤ 4`, written in `½xᵀPx + cᵀx`
    /// form (`P = 2I`, `c = (−6, −4)`, constant 13 dropped).
    ///
    /// Unconstrained optimum `(3, 2)` violates the row, so it binds and the
    /// solution is the projection onto `x₀ + x₁ = 4`: `x* = (2.5, 1.5)`,
    /// objective `−12.5`. Stationarity `Px + c + Gᵀz = 0` gives
    /// `(−1, −1) + z(1, 1) = 0 ⇒ z = 1`, which pins the **dual sign**: a
    /// flipped transform would clamp `z` to 0.
    fn projection_qp() -> QpProblem {
        QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-6.0, -4.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![4.0],
            lb: vec![],
            ub: vec![],
        }
    }

    // ---- gh #786: indefinite Hessians on `solve_qp_active_set_inertia` ----

    /// `min ½xᵀPx + cᵀx` over `[−1, 1]²` with `P = diag(−2, 1)`,
    /// `c = (0.5, −0.5)` — separable, so the exact answer is arithmetic.
    ///
    /// * `x₀`: `−x₀² + 0.5x₀` is **concave**, so its minimum over `[−1, 1]` is
    ///   at an endpoint — `−1.5` at `x₀ = −1`, against `−0.5` at `x₀ = +1`.
    ///   Both are local minima; only the first is global.
    /// * `x₁`: `0.5x₁² − 0.5x₁` is convex with an interior minimum
    ///   `−0.125` at `x₁ = 0.5`.
    ///
    /// So `f* = −1.625` at `(−1, 0.5)`, and the *other* local minimum is
    /// `−0.625` at `(1, 0.5)`. Asserting the global value is what makes this a
    /// test rather than a smoke check: a solver that treats the concave
    /// coordinate's interior stationary point `x₀ = 0.25` as an answer reports
    /// `−0.0625 − 0.125`, and one that simply lands on the nearer bound
    /// reports `−0.625`.
    fn indefinite_box_qp() -> QpProblem {
        QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, -2.0), Triplet::new(1, 1, 1.0)],
            c: vec![0.5, -0.5],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![-1.0, -1.0],
            ub: vec![1.0, 1.0],
        }
    }

    #[test]
    fn indefinite_hessian_solves_on_the_inertia_entry_point() {
        let prob = indefinite_box_qp();
        let mut mk = backend;
        let sol = solve_qp_active_set_inertia(
            &prob,
            &QpOptions::default(),
            &ActiveSetOverrides::default(),
            HessianInertia::Indefinite,
            &mut mk,
        );
        assert_eq!(sol.status, QpStatus::Optimal, "x = {:?}", sol.x);
        assert!(
            (sol.obj - (-1.625)).abs() < 1e-8,
            "expected the global optimum −1.625, got {} at {:?}",
            sol.obj,
            sol.x
        );
        assert!((sol.x[0] - (-1.0)).abs() < 1e-8, "x = {:?}", sol.x);
        assert!((sol.x[1] - 0.5).abs() < 1e-8, "x = {:?}", sol.x);
    }

    /// The claim the caller makes is the *only* difference between the two
    /// entry points on the same data, and `solve_qp_active_set` is documented
    /// to be the convex one. Pinned so the delegation cannot quietly become a
    /// pass-through of something else.
    #[test]
    fn the_plain_entry_point_is_the_psd_claim() {
        let prob = projection_qp();
        let (mut mk_a, mut mk_b) = (backend, backend);
        let plain = solve_qp_active_set(
            &prob,
            &QpOptions::default(),
            &ActiveSetOverrides::default(),
            &mut mk_a,
        );
        let claimed = solve_qp_active_set_inertia(
            &prob,
            &QpOptions::default(),
            &ActiveSetOverrides::default(),
            HessianInertia::Psd,
            &mut mk_b,
        );
        assert_eq!(plain.status, claimed.status);
        assert_eq!(plain.iters, claimed.iters);
        assert_eq!(plain.x, claimed.x);
    }

    /// A nonconvex QP that really is unbounded below, and the *only* reason it
    /// is: `min −x₀²` over `x₀ ≥ 0` has no descent at the origin (`c = 0`, so
    /// `cᵀd = 0`), it has negative curvature. The pre-gh#786 certificate
    /// required `Pd ≈ 0` before it would look at anything else, so this ray was
    /// rejected and the verdict fell back to the point.
    #[test]
    fn negative_curvature_certifies_unboundedness() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![Triplet::new(0, 0, -2.0)],
            c: vec![0.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![0.0],
            ub: vec![POS_INF],
        };
        assert!(
            ray_certifies_unbounded(&prob, &[1.0]),
            "a feasible recession direction of negative curvature is a proof"
        );
        // …and the branch does not fire without the recession half: `d = −1`
        // walks straight out of the box.
        assert!(!ray_certifies_unbounded(&prob, &[-1.0]));
    }

    /// The negative-curvature branch must stay unreachable on a PSD `P`, or it
    /// would hand a *convex* QP a false unboundedness verdict — the failure
    /// mode `verify_status` re-derives every certificate to avoid. `d = (1, 0)`
    /// here is a genuine recession direction of the feasible set (`x₀` is
    /// unbounded above) with zero slope and **positive** curvature: the
    /// objective turns back up along it.
    #[test]
    fn positive_curvature_along_a_recession_direction_is_not_a_proof() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![0.0, 0.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![0.0, 0.0],
            ub: vec![POS_INF, POS_INF],
        };
        assert!(!ray_certifies_unbounded(&prob, &[1.0, 0.0]));
        // The zero-curvature/strict-descent branch is untouched: drop the
        // curvature in `x₀` and give the objective a downhill slope there.
        let flat = QpProblem {
            p_lower: vec![Triplet::new(1, 1, 2.0)],
            c: vec![-1.0, 0.0],
            ..prob.clone()
        };
        assert!(ray_certifies_unbounded(&flat, &[1.0, 0.0]));
    }

    #[test]
    fn analytic_qp_primal_dual_and_sign() {
        let prob = projection_qp();
        let mut mk = backend;
        let sol = solve_qp_active_set(
            &prob,
            &QpOptions::default(),
            &ActiveSetOverrides::default(),
            &mut mk,
        );

        assert_eq!(sol.status, QpStatus::Optimal, "status");
        assert!((sol.x[0] - 2.5).abs() < 1e-8, "x0 = {}", sol.x[0]);
        assert!((sol.x[1] - 1.5).abs() < 1e-8, "x1 = {}", sol.x[1]);
        assert!((sol.obj + 12.5).abs() < 1e-8, "obj = {}", sol.obj);
        // The sign check: z must be +1, not 0 and not −1.
        assert!(sol.z[0] >= -1e-12, "z must be >= 0: {}", sol.z[0]);
        assert!((sol.z[0] - 1.0).abs() < 1e-7, "z0 = {} (sign!)", sol.z[0]);
    }

    /// The Hessian **off-diagonal** path: `p_lower` stores each pair once and
    /// both forms mirror it implicitly, so a translation that dropped or
    /// double-counted the mirror would change the matrix. Cross-checked
    /// against the IPM on the same problem — the two engines must agree.
    ///
    /// `P = [[2, 1], [1, 2]]`, `c = (−4, −5)`, `x₀ + x₁ ≤ 3`. Stationarity
    /// `Px = −c` gives `x* = (1, 2)` exactly, with `xᵀPx = 14` and
    /// `cᵀx = −14`, so `obj = 7 − 14 = −7`. Getting the mirror wrong changes
    /// `P` and moves `x*` well outside these tolerances.
    ///
    /// Note the optimum sits *exactly on* `x₀ + x₁ = 3`. That is deliberate —
    /// it is the degenerate boundary case — but it means the IPM, which
    /// approaches from the interior, stops a few `1e−5` short while the
    /// active-set engine lands on the vertex exactly. So the analytic values
    /// are asserted tightly and the IPM cross-check is only asked to agree to
    /// interior-point accuracy.
    #[test]
    fn off_diagonal_hessian_matches_ipm() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![
                Triplet::new(0, 0, 2.0),
                Triplet::new(1, 0, 1.0), // the off-diagonal, stored once
                Triplet::new(1, 1, 2.0),
            ],
            c: vec![-4.0, -5.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![3.0],
            lb: vec![],
            ub: vec![],
        };
        let mut mk = backend;
        let asol = solve_qp_active_set(
            &prob,
            &QpOptions::default(),
            &ActiveSetOverrides::default(),
            &mut mk,
        );
        let isol = solve_qp_ipm(&prob, &QpOptions::default(), backend);

        assert_eq!(asol.status, QpStatus::Optimal);
        assert_eq!(isol.status, QpStatus::Optimal);

        // Analytic optimum — tight, and the real test of the mirror.
        assert!((asol.x[0] - 1.0).abs() < 1e-9, "x0 = {}", asol.x[0]);
        assert!((asol.x[1] - 2.0).abs() < 1e-9, "x1 = {}", asol.x[1]);
        assert!((asol.obj + 7.0).abs() < 1e-9, "obj = {}", asol.obj);

        // Cross-check: the IPM must land on the same point, to its own accuracy.
        for i in 0..2 {
            assert!(
                (asol.x[i] - isol.x[i]).abs() < 1e-4,
                "x{i}: active-set {} vs ipm {}",
                asol.x[i],
                isol.x[i]
            );
        }
        assert!(
            (asol.obj - isol.obj).abs() < 1e-6,
            "obj: active-set {} vs ipm {}",
            asol.obj,
            isol.obj
        );
    }

    /// Equality rows and variable bounds both translate, and the bound
    /// multipliers land in the right sign slot.
    /// `min ½(x₀² + x₁²)  s.t.  x₀ + x₁ = 2,  x₀ ≥ 1.5`.
    /// Optimum `(1.5, 0.5)` — the bound binds.
    #[test]
    fn equality_and_bounds_match_ipm() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![2.0],
            g: vec![],
            h: vec![],
            lb: vec![1.5, crate::qp::NEG_INF],
            ub: vec![],
        };
        let mut mk = backend;
        let asol = solve_qp_active_set(
            &prob,
            &QpOptions::default(),
            &ActiveSetOverrides::default(),
            &mut mk,
        );

        assert_eq!(asol.status, QpStatus::Optimal, "status");
        assert!((asol.x[0] - 1.5).abs() < 1e-7, "x0 = {}", asol.x[0]);
        assert!((asol.x[1] - 0.5).abs() < 1e-7, "x1 = {}", asol.x[1]);
        // Lower bound binds ⇒ its multiplier is the positive one.
        assert!(asol.z_lb[0] >= -1e-12, "z_lb must be >= 0");
        assert!(asol.z_ub[0] >= -1e-12, "z_ub must be >= 0");

        let isol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert!(
            (asol.obj - isol.obj).abs() < 1e-6,
            "obj: active-set {} vs ipm {}",
            asol.obj,
            isol.obj
        );
    }

    /// One row `−x ≤ −rhs` (i.e. `x ≥ rhs`) on a single variable in `[0, 10]`.
    /// Infeasible exactly when `rhs > 10`, and infeasible *because of the box* —
    /// the row alone is satisfiable, so the classic zero-residual Farkas test
    /// has nothing to say here and only the box minimization can decide it.
    fn one_row_vs_box(rhs: f64) -> QpProblem {
        QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0)],
            h: vec![-rhs],
            lb: vec![0.0],
            ub: vec![10.0],
        }
    }

    /// The box minimization must be evaluated at the corner that *minimizes*
    /// `qᵀx`, and getting that backwards certifies feasible problems as
    /// infeasible. `x ≥ 5` with `x ≤ 10` is plainly satisfiable; the multiplier
    /// `z = 1` gives `q = (−1)` and `v = −5`, so the true bound
    /// `L = q·ub = −10` correctly fails `L > v`, whereas the wrong corner
    /// (`q·lb = 0`) would "prove" a feasible problem empty.
    #[test]
    fn box_minimum_is_taken_at_the_minimizing_corner() {
        assert!(
            !certifies_primal_infeasible(&one_row_vs_box(5.0), &[], &[1.0], &QpOptions::default()),
            "x ≥ 5, x ≤ 10 is feasible — no certificate may be accepted"
        );
    }

    /// The same shape pushed past the box: `x ≥ 15` with `x ≤ 10` is empty, and
    /// `L = −10 > v = −15` proves it.
    #[test]
    fn box_only_infeasibility_is_certified() {
        assert!(certifies_primal_infeasible(
            &one_row_vs_box(15.0),
            &[],
            &[1.0],
            &QpOptions::default()
        ));
    }

    /// A feasible QP whose feasible set is the single point `{0}` — every row
    /// active, no interior, multipliers wildly non-unique (the #282 hazard in
    /// miniature). Elastic-scale multipliers on it produce `q = 0` and `v = 0`:
    /// no strict descent, so no certificate. A false positive here would be a
    /// wrong statement about the user's model.
    #[test]
    fn spurious_certificate_on_feasible_qp_is_rejected() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![-1.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(1, 0, -1.0),
                Triplet::new(2, 1, 1.0),
                Triplet::new(3, 1, -1.0),
            ],
            h: vec![0.0; 4],
            lb: vec![],
            ub: vec![],
        };
        assert!(!certifies_primal_infeasible(
            &prob,
            &[],
            &[1e6, 1e6, 1e6, 1e6],
            &QpOptions::default()
        ));
    }

    /// gh #415's multipliers, verbatim from the engine: `z = (999999, 1e6)` on
    /// `x₀ + x₁ ≤ 1` / `x₀ + x₁ ≥ 3`. They leave `q = (−1, −1) = −c`, a
    /// *relative* residual of `1e−6` against a `FARKAS_RESID_TOL` of `1e−10`,
    /// so the textbook Farkas test rejects them — which is how a certified
    /// infeasibility became `NumericalFailure`. With the box `[0, 10]²` the
    /// residual is accounted for exactly (`L = −20 > v = −2000001`).
    ///
    /// Delete the box and the same multipliers must stop being a proof: that is
    /// not a regression but the reason [`feasibility_probe`] exists.
    #[test]
    fn box_accounts_for_the_objective_residual_the_farkas_test_rejects() {
        let boxed = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![1.0, 1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 0, -1.0),
                Triplet::new(1, 1, -1.0),
            ],
            h: vec![1.0, -3.0],
            lb: vec![0.0, 0.0],
            ub: vec![10.0, 10.0],
        };
        let z = [999_999.0, 1_000_000.0];
        assert!(
            certifies_primal_infeasible(&boxed, &[], &z, &QpOptions::default()),
            "the box makes these multipliers a proof"
        );

        let free = QpProblem {
            lb: vec![],
            ub: vec![],
            ..boxed
        };
        assert!(
            !certifies_primal_infeasible(&free, &[], &z, &QpOptions::default()),
            "without a box the residual is unaccounted for — must not be trusted"
        );
    }

    /// The budget scales with `n + m` instead of sitting at the IPM's flat
    /// 200, and an explicit user setting still wins.
    #[test]
    fn iteration_budget_scales_and_respects_user() {
        let d = QpOptions::default();
        assert_eq!(d.max_iter, 200, "test assumes the IPM default is 200");
        // Tiny problem keeps the floor.
        assert_eq!(active_set_iter_budget(&d, 2, 1), 200);
        // DUALC1's shape (n=9, m=215) must clear the flat 200 that broke it.
        assert!(
            active_set_iter_budget(&d, 9, 215) > 200,
            "budget must scale past the flat cap for DUALC1's shape"
        );
        // Explicit user value wins.
        let u = QpOptions {
            max_iter: 37,
            ..QpOptions::default()
        };
        assert_eq!(active_set_iter_budget(&u, 9, 215), 37);
    }

    /// **Singular Hessian with general inequality rows** — the geometry that
    /// exposed the seeding contract. `P = diag(0, 1)` has no curvature along
    /// `x₀`, so the reduced Hessian is singular unless the working set pins
    /// that direction. Handed a feasible point with a *cold* working set the
    /// engine sees no active rows, finds zero curvature along a feasible
    /// descent direction, and certifies a spurious unbounded ray; the working
    /// set has to come from the simplex basis for this to solve.
    ///
    /// `min ½x₁² − 2x₀ − x₁  s.t.  x₀ + x₁ ≤ 2,  x₀ ≤ 1.5,  x ≥ 0`.
    /// Both rows bind at `x* = (1.5, 0.5)`: stationarity
    /// `(−2, x₁−1) + z₀(1,1) + z₁(1,0) = 0` gives `z₀ = 0.5, z₁ = 1.5 ≥ 0`,
    /// and `obj = 0.125 − 3 − 0.5 = −3.375`.
    #[test]
    fn singular_hessian_with_inequalities_solves() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(1, 1, 1.0)], // P = diag(0, 1)
            c: vec![-2.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 0, 1.0),
            ],
            h: vec![2.0, 1.5],
            lb: vec![0.0, 0.0],
            ub: vec![],
        };
        let mut mk = backend;
        let sol = solve_qp_active_set(
            &prob,
            &QpOptions::default(),
            &ActiveSetOverrides::default(),
            &mut mk,
        );

        assert_eq!(sol.status, QpStatus::Optimal, "status");
        assert!((sol.x[0] - 1.5).abs() < 1e-7, "x0 = {}", sol.x[0]);
        assert!((sol.x[1] - 0.5).abs() < 1e-7, "x1 = {}", sol.x[1]);
        assert!((sol.obj + 3.375).abs() < 1e-7, "obj = {}", sol.obj);

        let isol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert!(
            (sol.obj - isol.obj).abs() < 1e-6,
            "obj: active-set {} vs ipm {}",
            sol.obj,
            isol.obj
        );
    }

    /// `min ½x² s.t. lb ≤ x ≤ ub`, solved by both engines. Returns
    /// `(active-set, ipm)` so the two can be held to the same answer.
    fn boxed_scalar(lb: f64, ub: f64) -> (QpSolution, QpSolution) {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![Triplet::new(0, 0, 1.0)],
            c: vec![0.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![lb],
            ub: vec![ub],
        };
        let mut mk = backend;
        let asol = solve_qp_active_set(
            &prob,
            &QpOptions::default(),
            &ActiveSetOverrides::default(),
            &mut mk,
        );
        let isol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        (asol, isol)
    }

    /// gh #491: a **reversed** box (`lb > ub`) is `PrimalInfeasible`, not a
    /// panic — and the two engines must agree on it.
    ///
    /// Before the fix the engine's `validate` rejected `xl > xu` outright, so
    /// both the first attempt and the Ruiz retry returned `NumericalFailure`
    /// and the last-resort simplex-seeded attempt ran — where the seed was
    /// clamped into the inverted interval and `f64::clamp` panicked with
    /// `min > max`. Through the Python bindings that became an uncatchable
    /// `pyo3_runtime.PanicException`, so this test would have *aborted* the
    /// test binary rather than failed an assertion.
    #[test]
    fn reversed_box_is_primal_infeasible_not_a_panic() {
        for (lb, ub) in [
            (1.0, 0.0),
            (5.0, 3.0),
            (0.0, -1e-6),
            (-1.0, -1e3),
            // The band the IPM used to resolve neither way, returning
            // `NumericalFailure` at a `NaN` iterate: wider than the tolerated
            // crossing, narrower than what its arithmetic could certify.
            (0.0, -1e-8),
            (0.0, -1e-7),
        ] {
            let (asol, isol) = boxed_scalar(lb, ub);
            assert_eq!(
                asol.status,
                QpStatus::PrimalInfeasible,
                "active-set on lb={lb} ub={ub}"
            );
            assert_eq!(
                isol.status,
                QpStatus::PrimalInfeasible,
                "ipm on lb={lb} ub={ub}"
            );
        }
    }

    /// The screen must not swallow the adjacent well-formed boxes: `lb == ub`
    /// is a fixed variable, and `±∞` on the absent side is the ordinary
    /// one-sided encoding. Both are feasible and must still solve.
    #[test]
    fn well_formed_boxes_are_untouched_by_the_screen() {
        let (asol, isol) = boxed_scalar(2.0, 2.0);
        assert_eq!(asol.status, QpStatus::Optimal, "active-set on lb == ub");
        assert_eq!(isol.status, QpStatus::Optimal, "ipm on lb == ub");
        assert!((asol.x[0] - 2.0).abs() < 1e-8, "x = {}", asol.x[0]);

        let (asol, isol) = boxed_scalar(f64::NEG_INFINITY, f64::INFINITY);
        assert_eq!(asol.status, QpStatus::Optimal, "active-set on a free var");
        assert_eq!(isol.status, QpStatus::Optimal, "ipm on a free var");
        assert!(asol.x[0].abs() < 1e-8, "x = {}", asol.x[0]);
    }

    /// A crossing no wider than [`crate::qp::CROSSED_BOX_TOL`] is a tolerance
    /// artifact — presolve's bound tightening can hand a driver a reduced
    /// problem carrying one, having already ruled it tolerable — so it is
    /// repaired to a fixed variable at the box midpoint rather than called
    /// infeasible.
    ///
    /// That is also where the IPM's own iteration landed on this input before
    /// the screen existed (it converged to the midpoint and reported
    /// `Optimal`), so the repair preserves its answer rather than replacing
    /// it. Both engines are checked: the property a user comparing `method=`
    /// observes is that they agree.
    #[test]
    fn hairline_crossing_is_repaired_not_rejected() {
        for gap in [1e-14, 1e-12, 1e-10, CROSSED_BOX_TOL] {
            let (asol, isol) = boxed_scalar(0.0, -gap);
            assert_eq!(asol.status, QpStatus::Optimal, "active-set on gap={gap:e}");
            assert_eq!(isol.status, QpStatus::Optimal, "ipm on gap={gap:e}");
            // Midpoint of the crossed box, to well inside the crossing itself.
            assert!(
                (asol.x[0] + 0.5 * gap).abs() <= gap,
                "x = {} for gap={gap:e}",
                asol.x[0]
            );
        }
    }

    /// The reversed box reached through an `n > 1` problem with real
    /// constraints, so the screen is exercised where the engine would otherwise
    /// have work to do. Only one variable is reversed; the rest of the box is
    /// ordinary.
    #[test]
    fn reversed_box_on_one_variable_of_many() {
        let prob = QpProblem {
            lb: vec![0.0, 1.0],
            ub: vec![10.0, 0.0], // variable 1 is reversed
            ..projection_qp()
        };
        let mut mk = backend;
        let sol = solve_qp_active_set(
            &prob,
            &QpOptions::default(),
            &ActiveSetOverrides::default(),
            &mut mk,
        );
        assert_eq!(sol.status, QpStatus::PrimalInfeasible);
        assert_eq!(sol.x.len(), prob.n, "x is returned at full length");
    }

    /// The impossible-bound class (gh #295) reaches the same verdict on this
    /// path. The IPM screens it at every entry point; before this the
    /// active-set driver screened it nowhere, so the raw Rust caller and the
    /// CLI's `qp-active-set` route got whatever the engine happened to do.
    #[test]
    fn impossible_bounds_are_primal_infeasible_on_the_active_set_path() {
        for (lb, ub) in [
            (f64::INFINITY, f64::INFINITY),
            (f64::NEG_INFINITY, f64::NEG_INFINITY),
        ] {
            let (asol, isol) = boxed_scalar(lb, ub);
            assert_eq!(
                asol.status,
                QpStatus::PrimalInfeasible,
                "active-set on lb={lb} ub={ub}"
            );
            assert_eq!(isol.status, QpStatus::PrimalInfeasible, "ipm");
        }
    }

    /// The exact optimum of [`projection_qp`], as a solution the engine could
    /// have been holding when it stopped.
    fn projection_optimum() -> QpSolution {
        QpSolution {
            status: QpStatus::Optimal,
            x: vec![2.5, 1.5],
            y: vec![],
            z: vec![1.0],
            z_lb: vec![0.0; 2],
            z_ub: vec![0.0; 2],
            obj: -12.5,
            iters: 0,
            iterates: Vec::new(),
        }
    }

    /// A cancelled solve is verified, not discarded.
    ///
    /// `verify_status` re-derives the KKT error against the original problem
    /// for every engine status, so the point an engine happens to be holding
    /// when it stops decides the verdict. Before this, `TimeLimit` alone
    /// skipped that check: the identical point was reported `Optimal` when the
    /// engine ran out of *iterations* and thrown away when it ran out of
    /// *seconds*. Which budget expired is not a fact about the user's problem.
    #[test]
    fn a_cancelled_solve_still_gets_its_kkt_point_verified() {
        let prob = projection_qp();
        let opts = QpOptions::default();
        let sol = projection_optimum();

        assert_eq!(
            verify_status(
                ActiveSetStatus::TimeLimit,
                None,
                SecondOrderVerdict::NotChecked,
                &sol,
                &prob,
                &opts
            ),
            QpStatus::Optimal,
            "an optimal point does not stop being optimal because the clock ran out"
        );
        // The iteration cap has always been salvaged this way; the two budgets
        // must now reach the same verdict on the same point.
        assert_eq!(
            verify_status(
                ActiveSetStatus::MaxIter,
                None,
                SecondOrderVerdict::NotChecked,
                &sol,
                &prob,
                &opts
            ),
            verify_status(
                ActiveSetStatus::TimeLimit,
                None,
                SecondOrderVerdict::NotChecked,
                &sol,
                &prob,
                &opts
            ),
            "the two budgets must agree on an identical point"
        );
    }

    /// The other half: salvage is earned, not assumed. A cancelled solve that
    /// really was nowhere near the answer still reports `TimeLimit`.
    #[test]
    fn a_cancelled_solve_far_from_optimal_is_still_a_time_limit() {
        let prob = projection_qp();
        let opts = QpOptions::default();
        let sol = QpSolution {
            x: vec![0.0, 0.0],
            z: vec![0.0],
            obj: 0.0,
            ..projection_optimum()
        };

        assert_eq!(
            verify_status(
                ActiveSetStatus::TimeLimit,
                None,
                SecondOrderVerdict::NotChecked,
                &sol,
                &prob,
                &opts
            ),
            QpStatus::TimeLimit
        );
    }

    /// `min ½K‖x‖² − K(x₀+x₁)  s.t.  x₀ + x₁ ≤ 1` — the gh #641 minimal case.
    /// The row binds, so by symmetry `x* = (½, ½)`, `obj* = −0.75K`, and
    /// stationarity `Kx − K + z(1,1) = 0` gives `z = K/2`.
    fn scaled_projection_qp(k: f64) -> QpProblem {
        QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, k), Triplet::new(1, 1, k)],
            c: vec![-k, -k],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![1.0],
            lb: vec![],
            ub: vec![],
        }
    }

    fn scaled_projection_point(k: f64, x: [f64; 2]) -> QpSolution {
        let obj = 0.5 * k * (x[0] * x[0] + x[1] * x[1]) - k * (x[0] + x[1]);
        QpSolution {
            status: QpStatus::Optimal,
            x: x.to_vec(),
            y: vec![],
            z: vec![k / 2.0],
            z_lb: vec![0.0; 2],
            z_ub: vec![0.0; 2],
            obj,
            iters: 0,
            iterates: Vec::new(),
        }
    }

    /// gh #641: the scale-relative arm only ever *relaxes*, and only where the
    /// absolute test is unreachable. Below the crossover the tight absolute test
    /// must still govern, so a point whose residual sits between `tol` and
    /// `1e3·tol` on a moderately-scaled QP stays `OptimalInaccurate` — the arm
    /// must not quietly promote every near-miss to a clean `Optimal`.
    #[test]
    fn moderate_scale_keeps_the_absolute_test() {
        let prob = projection_qp();
        let opts = QpOptions::default();
        // Perturb `x` off the optimum by enough to put the KKT error inside the
        // reduced-accuracy band (‖P‖ = 2, so the stationarity residual is 2δ).
        let sol = QpSolution {
            x: vec![2.5 + 5e-8, 1.5],
            ..projection_optimum()
        };
        let err = sol.kkt_residuals(&prob).kkt_error();
        assert!(
            err > opts.tol && err <= ACCEPTABLE_FACTOR * opts.tol,
            "test setup: {err:.3e} must land in the acceptable band"
        );
        assert_eq!(
            adjudicated_kkt_error(&prob, &sol, opts.tol, opts.obj_constant),
            err,
            "below the crossover the absolute residual must be used unchanged"
        );
        assert_eq!(
            verify_status(
                ActiveSetStatus::Optimal,
                None,
                SecondOrderVerdict::NotChecked,
                &sol,
                &prob,
                &opts
            ),
            QpStatus::OptimalInaccurate
        );
    }

    /// gh #641: above the crossover the arm opens, and an *exactly* optimal
    /// point — whose absolute residual cannot go below the `‖z‖·‖data‖·ε`
    /// complementarity floor — is certified `Optimal` rather than demoted.
    ///
    /// The point is the exact optimum's nearest representable neighbour (one ulp
    /// out in `x₀`), which is what the engine actually returns: `0.5` itself is
    /// representable and lands the residual at a clean zero, so testing that
    /// would test nothing. One ulp of `x` is `K·ε` of stationarity — `1.1e−4`
    /// here — which is the whole defect: no iterate can do better, and the
    /// absolute test rejects all of them.
    #[test]
    fn large_scale_exact_optimum_is_certified_optimal() {
        let k = 1e12;
        let prob = scaled_projection_qp(k);
        let opts = QpOptions::default();
        let sol = scaled_projection_point(k, [0.5 + f64::EPSILON / 2.0, 0.5]);
        assert!(
            sol.kkt_residuals(&prob).kkt_error() > ACCEPTABLE_FACTOR * opts.tol,
            "test setup: the absolute residual must be outside every band"
        );
        assert_eq!(
            verify_status(
                ActiveSetStatus::Optimal,
                None,
                SecondOrderVerdict::NotChecked,
                &sol,
                &prob,
                &opts
            ),
            QpStatus::Optimal
        );
    }

    /// The safety half of gh #641: relaxing the *measure* must not relax the
    /// *verdict*. A point that is genuinely not optimal — here one that violates
    /// the binding row outright — is rejected at the same scale where the arm is
    /// open, so the fix cannot manufacture a success out of a wrong answer.
    #[test]
    fn large_scale_wrong_point_is_still_rejected() {
        let k = 1e10;
        let prob = scaled_projection_qp(k);
        let opts = QpOptions::default();
        // Both components at 1: primal-infeasible (x₀+x₁ = 2 > 1) and far from
        // stationary. Its *relative* residual is O(1), not merely above `tol`.
        let sol = scaled_projection_point(k, [1.0, 1.0]);
        assert!(
            adjudicated_kkt_error(&prob, &sol, opts.tol, opts.obj_constant)
                > ACCEPTABLE_FACTOR * opts.tol,
            "a non-optimal point must not be rescued by the scale-relative arm"
        );
        assert_eq!(
            verify_status(
                ActiveSetStatus::Optimal,
                None,
                SecondOrderVerdict::NotChecked,
                &sol,
                &prob,
                &opts
            ),
            QpStatus::NumericalFailure
        );
    }

    /// gh #641, the sharp edge of the fix: primal feasibility is **not** relaxed
    /// to the relative measure, however large the problem.
    ///
    /// The point below is stationary and exactly complementary — its only defect
    /// is that it violates a constraint row by `7.8e−3`. Because that row's
    /// coefficients are `1e6`, the violation is a mere `7.8e−9` *relative to the
    /// row*, so an equilibrated relative primal residual waves it through. That
    /// is not a hypothetical: an earlier draft of this fix normalized all three
    /// terms and turned the `scaled_feasible_a` CLI fixture from an exact solve
    /// into `Optimal` printed beside a visible constraint violation, because
    /// accepting the unscaled attempt skipped the equilibrated retry that
    /// actually solves it.
    #[test]
    fn primal_feasibility_is_never_relaxed() {
        const K: f64 = 1e10; // objective scale — opens the relative arm
        const R: f64 = 1e6; // row scale — hides the violation if normalized
        const GAP: f64 = 7.8e-9; // x₀'s true distance from the candidate point

        // `min ½K‖x‖² − K(x₀+x₁)  s.t.  R·x₀ ≤ R(1 − GAP)`. The unconstrained
        // minimizer is `(1, 1)`; the row binds just short of it.
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, K), Triplet::new(1, 1, K)],
            c: vec![-K, -K],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, R)],
            h: vec![R * (1.0 - GAP)],
            lb: vec![],
            ub: vec![],
        };
        // The *unconstrained* minimizer, offered with a zero multiplier: exactly
        // stationary, exactly complementary, and infeasible.
        let sol = QpSolution {
            status: QpStatus::Optimal,
            x: vec![1.0, 1.0],
            y: vec![],
            z: vec![0.0],
            z_lb: vec![0.0; 2],
            z_ub: vec![0.0; 2],
            obj: -K,
            iters: 0,
            iterates: Vec::new(),
        };

        let res = sol.kkt_residuals(&prob);
        let opts = QpOptions::default();
        assert!(
            res.dual_infeasibility <= opts.tol && res.complementarity <= opts.tol,
            "test setup: only the primal term may be at fault ({res:?})"
        );
        assert!(
            res.primal_infeasibility > ACCEPTABLE_FACTOR * opts.tol,
            "test setup: the violation must be plainly outside every band"
        );
        // The trap: measured relatively, this violation looks converged.
        assert!(
            crate::ipm::equilibrated_kkt_rel_parts(&prob, &sol, opts.obj_constant)
                .primal_infeasibility
                <= opts.tol,
            "test setup: a relative primal residual would accept this point"
        );

        assert_eq!(
            verify_status(
                ActiveSetStatus::Optimal,
                None,
                SecondOrderVerdict::NotChecked,
                &sol,
                &prob,
                &opts
            ),
            QpStatus::NumericalFailure,
            "a constraint violation is a constraint violation at any scale"
        );
    }
}
