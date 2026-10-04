//! Primal-dual interior-point driver for convex QP.
//!
//! Infeasible-start primal-dual path-following with **Mehrotra
//! predictor-corrector** (adaptive centering σ = (μ_aff/μ)³ plus the
//! second-order `Δs∘Δz` term) and fraction-to-boundary step control.
//! Predictor and corrector share one factorization per iteration. The
//! homogeneous self-dual embedding (for clean infeasibility detection
//! and a self-starting iterate) is the remaining Phase 3 piece and slots
//! into this same scaffolding.
//!
//! On bound/inequality-constrained convex QPs this reaches the solution
//! in materially fewer interior-point iterations than routing the same
//! problem through the NLP filter-IPM — see
//! `crates/pounce-cli/tests/qp_vs_nlp_iterations.rs` (≈41% fewer at
//! n=50), the check behind the plan's 30–50% claim.
//!
//! ## Method
//!
//! For the standard-form QP (see [`crate::qp`]) with slacks `s ≥ 0` on
//! the inequalities (`Gx + s = h`) and multipliers `y` (equality),
//! `z ≥ 0` (inequality), the KKT conditions are
//!
//! ```text
//!   P x + c + Aᵀ y + Gᵀ z = 0      (stationarity, r_d)
//!   A x − b              = 0       (r_p)
//!   G x + s − h          = 0       (r_g)
//!   s ∘ z                = 0       (complementarity)
//! ```
//!
//! Each iteration solves the symmetric indefinite Newton system
//!
//! ```text
//!   ⎡ P+δI   Aᵀ      Gᵀ        ⎤ ⎡dx⎤   ⎡ −r_d            ⎤
//!   ⎢ A      −δI     0         ⎥ ⎢dy⎥ = ⎢ −r_p            ⎥
//!   ⎣ G      0    −(S⊘Z)−δI    ⎦ ⎣dz⎦   ⎣ −r_g + r_c ⊘ z  ⎦
//! ```
//!
//! (with `ds` recovered from `dz`) through the shared
//! [`pounce_linsol::Factorization`]. The tiny static regularization `δ`
//! makes the system quasi-definite so the LDLᵀ has a well-defined
//! inertia; because convergence is tested on the *unregularized*
//! residuals, the fixed point is the true QP solution — `δ` only
//! perturbs the search direction.
//!
//! The cone-specific pieces (`μ`, the `S⊘Z` scaling diagonal, the
//! complementarity residual, `ds` recovery, and the fraction-to-boundary
//! step) all route through the [`Cone`](crate::cones::Cone) trait so
//! that Phases 4–6 extend rather than rewrite this driver.

use crate::cones::{CompositeCone, Cone, ConeBlock, ConeSpec};
use crate::correctors;
use crate::debug::{ConvexDebugState, fire};
use crate::qp::{
    BoxScreen, QpIterate, QpProblem, QpResiduals, QpSolution, QpStatus, screen_variable_box,
};
use pounce_common::debug::{Checkpoint, DebugAction, DebugHook};
use pounce_common::types::{Index, Number};
use pounce_linsol::{Factorization, SparseSymLinearSolverInterface};
use std::collections::BTreeMap;
use std::time::Duration;

/// Tolerance on the **residual** of an infeasibility/unboundedness
/// certificate's defining equation (`‖Aᵀy+Gᵀz‖` for a Farkas pair,
/// `‖Px‖,‖Ax‖,‖Gx‖` for a recession ray), relative to the certificate's own
/// magnitude. Deliberately far tighter than [`QpOptions::infeas_tol`] (the
/// certificate-*value*/cone-membership tolerance): a genuine certificate
/// drives this residual to ~machine precision, whereas a *feasible* problem's
/// best approximate certificate floors at `∝ 1/‖x*‖` and must be rejected.
/// See [`detect_infeasibility_with`] for the full derivation (regression: a
/// feasible large-`‖x*‖` QP — POWELL20 — was declared primal-infeasible when
/// this shared `infeas_tol`).
pub(crate) const FARKAS_RESID_TOL: f64 = 1e-10;

/// Tolerance on the **normalized directional curvature** `dᵀPd / ‖d‖²` of a
/// candidate recession ray `d`. A convex QP recedes along `d` (objective
/// `−∞`) iff the curvature along `d` is exactly zero *and* `cᵀd < 0`; the
/// dual-infeasibility certificate accepts `d` only when the per-unit curvature
/// `dᵀPd/‖d‖²` (an eigenvalue-scale, `‖d‖`-invariant quantity — a diverging
/// iterate cannot inflate it) is below this floor.
///
/// The floor separates two regimes that a genuine unbounded solve and a bounded
/// tiny-curvature solve fall cleanly on either side of. A **bounded** problem
/// floors the normalized curvature at its smallest genuine directional
/// eigenvalue: `1e-12` for `P = diag(1e6, 1e-12)` (gh #293), `1e-16` for the
/// gh #273 unit case `P = 1e-16`. A **genuine recession** drives it toward zero
/// — exactly `0` for an LP or an axis-aligned null block, and, for a singular
/// `P` whose curved variable is pinned to a bound as the null variable
/// diverges, `~1e-140` and shrinking (the curved component decays like the
/// barrier parameter while `‖d‖` grows). The threshold sits many orders below
/// every real eigenvalue that must be rejected (`< 1e-16`) yet enormously above
/// the vanishing curvature of a true recession, so the two never collide.
/// Deliberately below machine epsilon: any direction this flat is
/// indistinguishable from `null(P)` at double precision, and — per gh #293 P0 —
/// a missed certification degrades to a safe `IterationLimit`, never a wrong
/// `DualInfeasible` on a bounded problem. See [`detect_infeasibility_with`].
const RECESSION_CURV_TOL: f64 = 1e-20;

/// Hard ceiling on any fraction-to-boundary parameter the adaptive rule
/// produces (see [`QpOptions::tau_max`]), and the default of that option.
/// Strictly below 1 so an accepted step always leaves the iterate in the
/// *open* cone: at τ = 1 exactly, a blocking component lands on `sᵢ = 0` /
/// `zᵢ = 0`, and the next iteration's `sᵢ/zᵢ` scaling and `ds` recovery
/// divide by it. The gap is far below any tolerance the solve converges to,
/// so this costs nothing in progress.
const TAU_CEIL: f64 = 1.0 - 1e-12;

/// The corrector's fraction-to-boundary parameter for **orthant** blocks:
/// the Mehrotra tail `τ = clamp(1 − μ, tau, tau_max)`.
///
/// As μ → 0 this approaches 1 and the corrector takes essentially the full
/// Newton step, which is what makes a warm start pay off in Newton steps
/// rather than in a logarithm of the perturbation (gh #417). Far from the
/// solution (μ ≥ 1 − `tau`, and on badly-scaled data where μ is large) it
/// reduces to the static `opts.tau`, so early iterations are unchanged.
fn adaptive_tau(mu: f64, opts: &QpOptions) -> f64 {
    // `tau` wins if a caller sets an inverted pair (`tau_max < tau`), which is
    // how the static behaviour is requested (`tau_max == tau`).
    let hi = opts.tau_max.min(TAU_CEIL).max(opts.tau);
    (1.0 - mu).clamp(opts.tau, hi)
}

/// Options for the QP interior-point solve.
#[derive(Debug, Clone, Copy)]
pub struct QpOptions {
    /// Solve-wide wall-clock budget. Retries, fallback engines, and crossover
    /// share one monotonic deadline. An in-flight backend factorization is not
    /// interrupted, so expiration may overshoot by one such operation.
    pub time_limit: Option<Duration>,
    /// Convergence tolerance on the max KKT residual and duality measure.
    pub tol: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Fraction-to-boundary parameter τ ∈ (0, 1) — the **floor** of the
    /// adaptive rule described on [`Self::tau_max`], and the flat value used
    /// everywhere that rule does not apply (the predictor step, every
    /// non-orthant cone block, and the HSDE driver). (The centering parameter
    /// σ is computed adaptively by the Mehrotra predictor; it is not an
    /// option.)
    pub tau: f64,
    /// Ceiling of the **adaptive** fraction-to-boundary rule
    /// `τ = clamp(1 − μ, tau, tau_max)`, applied by the direct (non-HSDE)
    /// driver to the corrector step on nonnegative-orthant blocks only.
    ///
    /// A static τ caps every step at a fixed fraction of the distance to the
    /// boundary, so μ and the residuals fall by a fixed factor per iteration
    /// (~20× at τ = 0.95) *regardless of how good the starting point is*. The
    /// iteration count is then `log₁/₍₁₋τ₎(μ₀/tol)` and a warm start can only
    /// lower μ₀ — it buys a logarithm of the perturbation rather than the one
    /// or two Newton steps a nearby problem deserves. Letting τ → 1 as μ → 0
    /// (the standard Mehrotra tail) restores the near-full step: on the
    /// warm-start QP families this cuts warm iterations 35–60% (gh #417) with
    /// cold counts untouched, since cold solves run HSDE.
    ///
    /// Scoped deliberately:
    /// * **orthant blocks only** — τ → 1 on a second-order or PSD block drives
    ///   the iterate onto a curved boundary its NT scaling cannot survive, and
    ///   costs the direct driver ~60% of the SOC instances it solves. See
    ///   [`CompositeCone::max_step_split`].
    /// * **corrector only** — the predictor's step lengths feed Mehrotra's
    ///   σ = (μ_aff/μ)³ heuristic, which is calibrated against a static τ.
    /// * **direct driver only** — the HSDE loop's step is also limited by the
    ///   τ/κ ray, so the same idea needs its own study there.
    ///
    /// Default `1 − 1e-12`: effectively "τ → 1" while keeping the iterate
    /// strictly inside the cone, so a block can never land exactly on the
    /// boundary and produce a division by a zero `zᵢ`. Set `tau_max == tau` to
    /// restore the old static-τ behaviour exactly.
    pub tau_max: f64,
    /// Static KKT regularization δ. Added on the (block) diagonal to make
    /// the reduced KKT system quasi-definite, so the LDLᵀ has a stable,
    /// well-defined inertia. Because convergence is tested on the
    /// *unregularized* residuals, δ only perturbs the search direction — but
    /// with a full Newton step it also floors the achievable primal residual
    /// at `δ·‖dy‖`. On badly-scaled NETLIB LPs the equality multipliers grow
    /// large (`adlittle`: `‖dy‖ ≈ 4e8`), so a too-large δ freezes `inf_pr`
    /// above the tolerance and the IPM stalls to its iteration cap. The
    /// default is sized small enough to clear that floor on such instances
    /// while still keeping the factorization quasi-definite (see [`Default`]).
    pub reg: f64,
    /// Relative tolerance for the *value* and cone-membership parts of an
    /// infeasibility/unboundedness certificate (`bᵀy+hᵀz < 0`, `z ∈ K*`),
    /// taken relative to the certificate's own magnitude. The certificate's
    /// *residual* (its defining equation `Aᵀy+Gᵀz = 0`, or `Px=Ax=Gx=0` for a
    /// recession ray) is held to the far tighter [`FARKAS_RESID_TOL`] instead:
    /// a genuine certificate drives the residual to ~machine precision, while
    /// a feasible problem's best approximate certificate only reaches a floor
    /// `∝ 1/‖x*‖`. Splitting the two is what keeps a status backed by a real
    /// proof — `IterationLimit` is the fallback when no certificate verifies.
    pub infeas_tol: f64,
    /// Use the homogeneous self-dual embedding driver ([`crate::hsde`]) rather
    /// than the infeasible-start primal–dual method. HSDE self-starts, produces
    /// infeasibility/unboundedness certificates natively, and stays stable on
    /// badly-conditioned problems where the infeasible-start method diverges
    /// (its duality measure blows up — e.g. NETLIB `nl`, where the direct path
    /// runs `mu` to ~1e11 and trips a spurious `NumericalFailure`, while HSDE
    /// converges). It is also the substrate for the non-symmetric cones
    /// (exp/power). This matches Clarabel/ECOS/SCS, which embed precisely for
    /// that robustness. **Default `true`.**
    ///
    /// HSDE does not (yet) exploit warm starts or reuse an external
    /// factorization, so the advanced performance paths — [`QpWarmStart`] and
    /// the build-once [`QpFactorization`] handle — set this `false` to opt back
    /// into the direct solver, which they require. Their callers are doing
    /// *nearby reoptimization* (a known-solvable neighborhood), where the
    /// direct path's fragility is not a concern.
    ///
    /// With `false` on a PSD-carrying problem, [`solve_socp_ipm`] retries a
    /// direct solve that ends without a full answer through HSDE once (gh
    /// #226) — the direct driver is known-weak on boundary-degenerate PSD
    /// optima, where the embedding stays well-conditioned.
    pub use_hsde: bool,
    /// Collect a per-iteration convergence trace into
    /// [`crate::QpSolution::iterates`]. Off by default so a normal solve has
    /// no recording overhead; turn on when a solve report or benchmark
    /// harness wants the per-iteration history. Default `false`.
    pub collect_iterates: bool,
    /// Ruiz-equilibrate the problem data before solving (see
    /// [`crate::equilibrate`]). A conditioning aid for the **direct**
    /// infeasible-start IPM, which factorizes the raw KKT system and is fragile
    /// on badly-scaled data. It is applied only when [`Self::use_hsde`] is
    /// `false` (the direct one-shot path and the warm-start path); the default
    /// HSDE driver skips it, conditioning the system internally through its
    /// per-cone NT scaling. Applied only on the LP/QP orthant entry points
    /// ([`solve_qp_ipm`] / [`solve_qp_ipm_warm`]), where per-row scaling
    /// preserves the cone; the SOCP/conic driver never equilibrates, since
    /// per-row scaling is unsound for non-orthant cones. Default `true`.
    pub equilibrate: bool,
    /// A constant added to `½xᵀPx + cᵀx` to obtain the objective the **caller**
    /// reports. Default `0.0`.
    ///
    /// [`QpProblem`] models the quadratic form only, so a model whose objective
    /// carries a degree-0 term hands the solver an objective displaced by that
    /// term. Least-squares objectives — `Σ(xᵢ − aᵢ)²`, constant `Σaᵢ²` — are the
    /// common case, and the displacement is unbounded: on the
    /// `scaled_feasible_a` fixture the caller's optimum is `0` while the
    /// solver's is `−5.0e11`.
    ///
    /// That matters because the scale-relative stopping test
    /// ([`crate::hsde::relative_stop_permitted`]) normalizes the duality gap by
    /// the objective *magnitude*, the standard convention. Under a large
    /// displacement that magnitude is a property of the constant and not of the
    /// solution, so `tol`-relative becomes a blanket `tol·|constant|` absolute
    /// slack on the gap: HSDE certified `Optimal` on `scaled_feasible_a` at a
    /// caller-visible objective of `236.85` — `4.7e-10` relative in *its* metric
    /// and 100% wrong in the caller's (gh #689). Told the constant, the same
    /// solve normalizes by the objective the caller actually reads and runs on
    /// to the true optimum.
    ///
    /// **Purely a convergence-test normalizer.** It never enters the KKT
    /// system, the search direction, the duals, or [`QpSolution::obj`] (which
    /// stays the quadratic form's own value, as before — the caller adds the
    /// constant back exactly as it always did). A wrong or missing value can
    /// therefore only make the gap test tighter or looser, never unsound; `0.0`
    /// — the default, and what every caller that does not set it gets — is the
    /// tightest choice and reproduces the historical test whenever the true
    /// constant is small next to the objective.
    pub obj_constant: f64,
    /// Run the LP-crossover phase ([`crate::crossover`]) after the interior-
    /// point solve. For a **pure LP** (`P = 0`), crossover hands the near-
    /// optimal interior iterate to the active-set engine ([`pounce_qp`]),
    /// which pivots it to an *exact* optimal vertex basis. This closes the
    /// gap on degenerate LPs (NETLIB GEN family), where strict
    /// complementarity fails, the fraction-to-boundary step collapses, and a
    /// pure IPM cannot certify the vertex to `tol` — exactly the
    /// IPM-then-crossover pairing every commercial LP solver uses
    /// (Andersen & Ye 1996). It is a strict, **never-regress** refinement: the
    /// purified vertex is returned only when it is feasible and its KKT error
    /// does not exceed the interior iterate's. A no-op for genuine QPs
    /// (`P ≠ 0`) and for the warm-start / debug entry points.
    ///
    /// **Default `false` — opt-in.** Crossover is correct (never-regress) but
    /// the active-set purification is currently *slow* on the degenerate /
    /// large NETLIB LPs it most targets: on the LP suite it regressed solve
    /// times 3×–800× versus the pure IPM (dozens of sub-second LPs pushed past
    /// the 300 s cap) while still **not** reaching an exact `Optimal` vertex on
    /// the GEN family it was built for (see issue #133). Until the purification
    /// is made fast and robust (the deferred LU-basis engine), it ships off by
    /// default and is enabled explicitly — CLI `qp_crossover=yes`, or this
    /// field — for callers who want exact-vertex refinement on small,
    /// well-behaved LPs and can absorb the cost.
    pub crossover: bool,
    /// Maximum Gondzio multiple centrality correctors per iteration, on
    /// **nonnegative-orthant** blocks only (see [`crate::correctors`]). `0`
    /// disables them.
    ///
    /// Each corrector is one extra back-solve through the factorization the
    /// iteration already paid for — never a refactorization — and is kept only
    /// if it lengthens the fraction-to-boundary step. It is the standard answer
    /// to an iterate whose steps are *accepted but short*: the products `sᵢzᵢ`
    /// have spread out, the blocking component stops the step far from the
    /// boundary, and re-centering the spread-out products buys back the step
    /// length that poor centrality took away.
    ///
    /// Both symmetric drivers honour this: the HSDE loop, which has had the
    /// scheme since the NETLIB GEN degenerate-face work, and the direct
    /// `run_ipm`, which gained it in gh #588. Default 3 — Gondzio's own
    /// recommendation and the value the HSDE driver has always used, so the
    /// default leaves that driver bit-for-bit unchanged.
    pub gondzio_max_corr: usize,
}

impl Default for QpOptions {
    fn default() -> Self {
        QpOptions {
            time_limit: None,
            tol: 1e-8,
            max_iter: 200,
            tau: 0.95,
            tau_max: TAU_CEIL,
            // δ = 1e-10: small enough that the primal-residual floor δ·‖dy‖
            // clears `tol` even when the equality duals are large (badly
            // scaled NETLIB LPs such as `adlittle`, which stalls at the cap
            // with δ = 1e-8 but converges in ~57 iters here), yet still
            // strictly positive so the reduced KKT stays quasi-definite for a
            // stable LDLᵀ inertia. The whole 1e-9‥1e-11 band converges the
            // LP/QP benchmark suites; 1e-10 is centered in it.
            reg: 1e-10,
            infeas_tol: 1e-7,
            use_hsde: true,
            collect_iterates: false,
            equilibrate: true,
            obj_constant: 0.0,
            // Opt-in: off by default. See the field doc — correct but slow on
            // the LPs it targets, and does not yet reach Optimal on GEN (#133).
            crossover: false,
            gondzio_max_corr: crate::correctors::MAX_CORR,
        }
    }
}

/// Solve a convex QP, honoring any per-variable bounds (`lb`/`ub`).
///
/// Variable bounds are a first-class part of [`QpProblem`] so presolve
/// can reason about boxes; the solver itself expands the *finite* bounds
/// into internal inequality rows, runs the bounds-agnostic Mehrotra core
/// ([`solve_qp_core`]), and splits the returned inequality multipliers
/// back into the original `z` and the bound multipliers `z_lb`/`z_ub`.
/// The iteration math is unchanged by the presence of bounds.
pub fn solve_qp_ipm<F>(prob: &QpProblem, opts: &QpOptions, make_backend: F) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    crate::deadline::with_deadline(opts.time_limit, || {
        solve_qp_ipm_scoped(prob, opts, make_backend, None)
    })
}

/// The body of [`solve_qp_ipm`], with an optional [`DebugHook`] threaded to
/// the *primary* solve.
///
/// `hook` is what makes [`solve_qp_ipm_debug`] the same function as
/// [`solve_qp_ipm`] rather than a parallel one (gh #892). It is handed to the
/// first driver invocation only; the recovery re-solves below (the
/// equilibrated retry, the reverify, the infeasibility twin) run unhooked, so
/// the debugger observes the solve the caller asked for and every fallback
/// still runs exactly as it does without a debugger attached. The *answer* is
/// therefore identical with and without the hook.
fn solve_qp_ipm_scoped<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    make_backend: F,
    hook: Option<&mut dyn DebugHook>,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let mut make_backend = make_backend;
    if crate::deadline::expired() {
        return timed_out_solution(prob);
    }
    // Screen the variable box before bound expansion (gh #295, gh #491):
    // `expand_bounds` is sign-agnostic, so a *present* `+∞` lower / `−∞` upper
    // bound would be mishandled as an *absent* one and a violating point
    // reported `Optimal`; and a box crossed by more than a tolerance is empty,
    // which the iteration resolved only for wide crossings — in between it
    // returned `NumericalFailure` at a `NaN` iterate. A hairline crossing is
    // repaired to the midpoint the iteration converged to anyway. See
    // [`screen_variable_box`].
    let snapped;
    let prob = match screen_variable_box(prob) {
        BoxScreen::Feasible => prob,
        BoxScreen::Empty => return trivial_primal_infeasible_solution(prob),
        BoxScreen::Snapped(p) => {
            snapped = p;
            &snapped
        }
    };
    // Interior-point solve in the original problem's coordinates (the core
    // already unscales any internal Ruiz equilibration before returning).
    let (sol, sigma_uncertified) =
        crate::sigma_verdict::tracking(|| solve_qp_ipm_core(prob, opts, &mut make_backend, hook));
    if crate::deadline::expired() {
        return mark_timed_out(sol);
    }
    // LP-crossover refinement: for a pure LP, purify the interior iterate to an
    // exact optimal vertex via the active-set engine. Gated to pure LPs and
    // never-regressing — a no-op for QPs and whenever the vertex is not a
    // strict improvement. Runs against the same un-equilibrated `prob` so the
    // `z`/`s` conventions line up. See [`crate::crossover`].
    // gh#880 follow-up: the `σ` verdict below is about the point the cascade
    // returned. Crossover can replace that point without re-entering the
    // solver, and `sigma_verdict`'s own module doc says a consumer sensitive
    // to *which* of the two points the bit describes "needs to re-record
    // after crossover rather than assume". gh#888 created exactly such a
    // consumer one merge earlier: the CLI reroutes `ProblemClass::Lp` on
    // `OptimalInaccurate`, so a stale demotion no longer just labels the
    // answer, it throws it away and re-solves on the NLP arm — discarding a
    // crossover-purified *exact vertex* on a verdict about the interior
    // iterate that vertex replaced.
    //
    // Kept as a re-record rather than a blanket clear. "Crossover fired, so
    // the demotion cannot apply" would be an argument from crossover's
    // never-regressing gate; asking the estimator again is a measurement of
    // the point actually being returned, and it is the same estimator with
    // the same cut, so the two sites cannot drift.
    //
    // The clone is taken only when there is a verdict to re-record, so the
    // ordinary path pays nothing. See `sigma_verdict_after_crossover` for the
    // measured reachability of each half — the crossover half is the majority
    // case once `qp_crossover=yes`, the uncertified half was not reproducible.
    let pre_crossover_x = sigma_uncertified.then(|| sol.x.clone());
    let sol = if crate::debug_stop::requested() {
        sol
    } else {
        crate::crossover::maybe_crossover(prob, sol, opts, &mut make_backend)
    };
    let sigma_uncertified = sigma_verdict_after_crossover(
        prob,
        &sol,
        pre_crossover_x.as_deref(),
        opts.tol,
        sigma_uncertified,
    );
    // Crossover declines rather than restamps when the budget runs out mid-
    // refinement, so account for a deadline crossed in there here — where
    // `mark_timed_out`'s verdict rule applies and an `Optimal` cannot be lost.
    let sol = if crate::deadline::expired() {
        mark_timed_out(sol)
    } else {
        sol
    };
    let sol = demote_uncertified_sigma_optimum(sol, sigma_uncertified);
    demote_optimum_above_tol(prob, finite_or_failed(prob, sol), opts)
}

/// gh#984 item 3: never hand back `Optimal` beside a KKT error above `tol`.
///
/// The verdict is judged on [`QpSolution::kkt_residuals_above_floor`] -- the
/// returned point's own residuals, each read above the rounding error of
/// evaluating it -- which is also what the Python `residuals["kkt_error"]` and
/// the CLI report, so the status and the number printed beside it are one
/// measurement. A well-scaled solve is untouched (its floors are `~1e-14`); a
/// point the scale-relative arm stopped on whose residual is *genuinely* above
/// both `tol` and its floor becomes [`QpStatus::OptimalInaccurate`], a usable
/// answer at reduced accuracy that does not claim what it did not reach.
///
/// Last in the pipeline, so no retry or repair path reads the demoted status.
/// Orthant/box problems only: the conic entry points judge cone membership
/// with their own residuals.
fn demote_optimum_above_tol(prob: &QpProblem, sol: QpSolution, opts: &QpOptions) -> QpSolution {
    let tol = opts.tol;
    if std::env::var("DBG984").is_ok() {
        eprintln!(
            "DBG984 {:?} it {} raw {:?} adj {:?} tol {tol:e}",
            sol.status,
            sol.iters,
            sol.kkt_residuals(prob),
            sol.kkt_residuals_above_floor(prob)
        );
    }
    if !opts.use_hsde
        || sol.status != QpStatus::Optimal
        || sol.kkt_residuals_above_floor(prob).kkt_error() <= tol
    {
        return sol;
    }
    QpSolution {
        status: QpStatus::OptimalInaccurate,
        ..sol
    }
}

/// Strip `Optimal` from a `σ`-path answer the cascade could not certify.
///
/// gh #880. When [`hsde_cost_scale`] rescales the objective, the cascade tries
/// three drivers and asks [`normalized_optimum_is_genuine`] of each; if none
/// passes it returns the closest to optimality in the caller's own
/// coordinates. That point is the right one to keep — on the 72-instance
/// coupled census it beats the un-normalized re-solve on seven of nine
/// failures, by 6x to 15416x — but it went back under a bare `Optimal`, and on
/// three instances that `Optimal` sat on an `x` further from the optimum than
/// f64 permits at that conditioning (`3.06e-04` against a `kappa*eps` floor of
/// `2.20e-04`).
///
/// The census numbers this PR quotes — instances above their floor 3 -> 0,
/// worst forward error `3.06e-04` -> `5.62e-05`, iterations 443 -> 404 — are
/// the *change's*, not this function's. It is a pure relabel at the outermost
/// layer, after every retry has run; it cannot move a forward error and
/// certainly cannot move an iteration count. Those belong to the compensated
/// estimator and the cut, which reroute which candidate the cascade accepts,
/// and to the CLI reroute the new status enables.
///
/// **Applied at the outermost layer, not where the pick is made.** Demoting
/// inside the cascade looks equivalent and is not: `OptimalInaccurate` is one
/// of the statuses [`solve_qp_ipm_core`] reads as the badly-scaled pathology
/// worth a Ruiz-equilibrated retry, and that retry is accepted whenever it
/// converges to a clean `Optimal` — its status outranking the better answer.
/// Measured, demoting there takes the census's worst forward error from
/// `3.06e-04` to `3.93e+01`. By this point every retry has run, so the verdict
/// cannot re-enter the solver.
///
/// What is recorded is the **scale-free half** of the cascade's verdict and a
/// margin, not "nothing was certified" — see the recording site in
/// [`solve_qp_core`] for why the mixture is not reportable and the margin is
/// the estimator's own uncertainty rather than a knob.
fn demote_uncertified_sigma_optimum(sol: QpSolution, sigma_uncertified: bool) -> QpSolution {
    if !sigma_uncertified || sol.status != QpStatus::Optimal {
        return sol;
    }
    QpSolution {
        status: QpStatus::OptimalInaccurate,
        ..sol
    }
}

/// The interior-point solve (the historical [`solve_qp_ipm`] body): bounds-aware
/// orthant solve with optional Ruiz equilibration, returning a solution in the
/// original problem's coordinates. Factored out so [`solve_qp_ipm`] can layer
/// the LP-crossover refinement on top.
fn solve_qp_ipm_core<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    make_backend: F,
    hook: Option<&mut dyn DebugHook>,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    // Ruiz-equilibrate the data first — but only for the *direct* driver.
    // Solving the scaled problem and unscaling the result keeps the direct
    // infeasible-start IPM well-conditioned without changing the recovered KKT
    // point. The HSDE driver does NOT need (and must not get) this: the
    // self-dual embedding conditions the system internally through its per-cone
    // NT scaling — exactly as Clarabel/ECOS do, neither of which Ruiz-pre-scales
    // — so it solves even badly-scaled data (NETLIB `nl`, ‖c‖~1e6) directly.
    // Layering Ruiz on top is not only redundant for HSDE, it composes badly
    // with presolve: presolve's reductions plus Ruiz's σ=1/‖c‖ cost scaling
    // over-condition the reduced KKT system and trip the factorization near the
    // boundary (a `NumericalFailure` that neither transform produces alone).
    // See `crate::equilibrate`.
    let mut make_backend = make_backend;
    if opts.equilibrate && !opts.use_hsde {
        return equilibrated_solve(
            prob,
            opts,
            /* use_hsde */ false,
            &mut make_backend,
            hook,
        );
    }
    let sol = solve_qp_ipm_unscaled(prob, opts, &mut make_backend, hook);
    // HSDE robustness fallback. The self-dual driver normally conditions itself
    // through its per-cone NT scaling and so deliberately skips Ruiz pre-scaling
    // (see the comment above). But on a *severely* ill-scaled system — e.g. the
    // spatial-B&B relaxation LPs whose McCormick/division columns and ln/√
    // envelope tangents span `|G| ∈ [1e-7, 1e6]` — the embedded KKT
    // factorization can still break down (`NumericalFailure`), discarding an
    // otherwise-correct iterate and leaving the B&B node with no lower bound.
    // When that happens, retry once *with* Ruiz equilibration. This is sound and
    // does not contradict the "Ruiz composes badly with HSDE" note: we only get
    // here because the un-equilibrated solve already failed, so there is nothing
    // left to regress — equilibration can only recover a usable solve or fail
    // the same way (in which case we keep the original result).
    //
    // gh #293: the same retry also rescues an HSDE solve that failed to reach a
    // clean `Optimal` on a badly-scaled QP whose Hessian curvature is tiny. The
    // canonical case is a uniformly tiny Hessian — `P = diag(1e-12, 1e-12)`,
    // optimum at `‖x*‖ ≈ 1e12` — where HSDE's per-cone NT scaling never sees the
    // objective curvature (the `P` block is 12 orders below O(1)), so the
    // iterates crawl and the budget is exhausted short of the optimum. Ruiz
    // pre-scaling lifts `P̂` to O(1) and the same driver then converges. The same
    // pathology surfaces through either non-converged status depending on the
    // geometry: as `IterationLimit` when the optimum is unconstrained (the
    // iterates never arrive), and as `OptimalInaccurate` when a constraint binds
    // near it (a usable-but-loose iterate is returned at the cap). Both are keyed
    // on the same retry, so the fix covers the *regime*, not one status symbol.
    // This does not contradict the "Ruiz composes badly with HSDE" note either:
    // we only reach here because the un-equilibrated solve did not cleanly
    // converge, so there is nothing left to regress. Unlike the
    // `NumericalFailure` case (where any non-failing status is an improvement),
    // `IterationLimit` and `OptimalInaccurate` are *honest* statuses, so the
    // equilibrated retry is accepted **only when it converges to a clean
    // `Optimal`** — never when it merely returns a different non-converged or
    // certificate status. A genuinely hard problem thus keeps its truthful
    // status, and no infeasibility/unboundedness verdict can be introduced by
    // the retry. (Cone-carrying problems never reach this branch: exp/power/SOC
    // solve through `solve_socp_ipm`, and Ruiz — an orthant-only row scaling —
    // is confined to this LP/QP entry point.)
    let retry_on = matches!(
        sol.status,
        QpStatus::NumericalFailure | QpStatus::IterationLimit | QpStatus::OptimalInaccurate
    );
    // `!debug_stop::requested()` guards every recovery re-solve in this
    // function and its neighbours (gh #892): each of them re-solves unhooked,
    // so after a debugger `quit` they would run to completion and hand back a
    // verdict for a solve the user deliberately halted. Declining to start one
    // leaves the halted run's own status standing, which is the honest answer
    // and the one the debug path returned before it shared this code. The
    // predicate is `false` whenever no debugger is attached, so the ordinary
    // path is bit-for-bit unchanged.
    if opts.use_hsde && opts.equilibrate && retry_on && !crate::debug_stop::requested() {
        if crate::deadline::expired() {
            return mark_timed_out(sol);
        }
        // Budget the retry on a *pure LP*, where its premise does not apply.
        //
        // Everything above justifies this retry by Hessian curvature that NT
        // scaling cannot see: `P = diag(1e-12)` with the optimum at `‖x*‖ ~
        // 1e12`, where Ruiz lifts `P̂` to O(1) and the same driver then
        // converges. That is a QP story. A pure LP has `P = 0` — there is no
        // curvature term for equilibration to surface — so an LP retry is only
        // ever fixing row/column scaling, and when that works, it works fast.
        //
        // Measured over the LP, QP and lpopt corpora (513 problems, 24 retries):
        // every *accepted* LP retry converged by iteration 78 (24, 30, 32, 32,
        // 33, 34, 34, 34, 46, 78), while every LP retry that ran to the cap was
        // discarded — `gen`, `gen1`, `complex`, `df2177`, `dsbmip`, `pilot.ja`,
        // `de063155`, `irish-electricity`, all 199 iterations, all rejected, all
        // paying a second full budget to learn nothing. `gen` and `gen1` alone
        // are 347 s of that corpus and end up on the NLP arm regardless, which
        // solves them in ~1 s. The `P = 0` gate is why this is not applied
        // globally: `QSCFXM1/2/3` and `Q25FV47` are accepted at 131–168
        // iterations, and a flat cap would demote four clean `Optimal` results.
        //
        // Half the first solve's budget clears the longest observed LP success
        // by 22 iterations and scales with a user-set `max_iter`. If an unseen
        // LP needs more, the failure mode is soft: the retry is rejected, the
        // first solve's honest non-converged status stands, and under
        // `solver_selection=auto` gh #535 routes it to the NLP arm — where
        // these LPs were already going.
        //
        // Gated to the two statuses whose acceptance below demands a *certified*
        // `Optimal`. That restriction is load-bearing, not tidiness: a
        // first-solve `NumericalFailure` accepts any non-failing retry status as
        // an improvement on a breakdown, so capping the budget there lets the
        // cap *manufacture* an `IterationLimit` and have it accepted — reporting
        // "Maximum iterations exceeded", with the capped retry's iterate, where
        // the honest answer is "Numerical failure". `lp_afiro` at `qp_tau=0.99`
        // does exactly this; `issue_535_lp_falls_back_to_nlp` pins it. Under
        // `IterationLimit`/`OptimalInaccurate` a capped retry can only fail to
        // certify and be discarded — which is the outcome the cap wants sooner.
        let mut retry_opts = opts.clone();
        retry_opts.max_iter = equilibrated_retry_budget(prob, sol.status, opts.max_iter);
        let retry = equilibrated_solve(
            prob,
            &retry_opts,
            /* use_hsde */ true,
            &mut make_backend,
            None,
        );
        // An `Optimal` from this retry has to earn the same way the one in
        // [`verify_or_repair_optimum`] does (gh #712). This retry runs *inside*
        // the equilibrated metric, so its own absolute convergence test is
        // applied to the Ruiz-scaled problem and says nothing about the point's
        // accuracy in the caller's coordinates — exactly the gap gh #414 opened
        // this check for. Until gh #712 this was the one `Optimal` in this
        // function that reached a caller unchecked, and on `scaled_feasible_a`
        // it returned a point whose absolute KKT error is `2.3e3` as
        // `SolveSucceeded`. A retry that cannot certify leaves the original
        // status standing, which is the honest answer: the loop really did run
        // out of iterations.
        let retry_optimal_genuine = retry.status == QpStatus::Optimal
            && optimum_is_genuine(prob, &retry, opts.tol, opts.obj_constant);
        let accept = match sol.status {
            // Any non-failing status is an improvement on a breakdown — except
            // a false `Optimal`, which is worse than an honest failure.
            QpStatus::NumericalFailure => {
                retry.status != QpStatus::NumericalFailure
                    && (retry.status != QpStatus::Optimal || retry_optimal_genuine)
            }
            QpStatus::IterationLimit | QpStatus::OptimalInaccurate => retry_optimal_genuine,
            _ => false,
        };
        if accept {
            return retry;
        }
    }
    // gh #414: the mirror image of the retry above — an HSDE solve that believes
    // it converged but did not. See [`verify_or_repair_optimum`]. Touches only
    // an `Optimal` verdict, so it composes with (and cannot disturb) the
    // certificate handling below.
    let sol = if crate::debug_stop::requested() {
        sol
    } else {
        verify_or_repair_optimum(prob, opts, sol, &mut make_backend)
    };
    // gh #293 (extreme tail): refute a *spurious* unboundedness certificate on a
    // QP with genuine curvature. A `DualInfeasible` verdict rests on finding a
    // recession ray whose normalized curvature `dᵀPd/‖d‖²` is below
    // [`RECESSION_CURV_TOL`]. When the Hessian is so tiny that a bounded descent
    // direction's curvature sinks to that floor (`P ≈ 1e-20`, at the edge of
    // double precision), the raw HSDE solve can read a bounded ray as a
    // recession and wrongly certify the problem unbounded — the exact failure
    // #290/#309 fixed for `1e-12`, resurfacing only in the machine-epsilon tail.
    // A *genuine* recession lies in `null(P)` and survives equilibration, so
    // re-solving the Ruiz-scaled problem with the direct driver (which lifts
    // `P̂` to O(1), making the true curvature visible) is a decisive cross-check:
    // if it returns a clean, finite `Optimal`, the problem was bounded and the
    // certificate was a scaling artifact, so return the verified optimum;
    // otherwise keep the original verdict. Gated to `P ≠ 0` — a pure LP's
    // unboundedness is exact, never a curvature artifact — so genuine unbounded
    // LPs never pay for the reverify.
    if opts.use_hsde
        && opts.equilibrate
        && sol.status == QpStatus::DualInfeasible
        && prob.p_lower.iter().any(|t| t.val != 0.0)
        && !crate::debug_stop::requested()
    {
        if crate::deadline::expired() {
            return mark_timed_out(sol);
        }
        let verify = equilibrated_solve(
            prob,
            opts,
            /* use_hsde */ false,
            &mut make_backend,
            None,
        );
        if verify.status == QpStatus::Optimal {
            return verify;
        }
    }
    // An unboundedness verdict on a problem that has no feasible point at all.
    //
    // `DualInfeasible` rests on a recession direction `d` with `Pd ≈ 0, Ad ≈ 0,
    // −Gd ∈ K, cᵀd < 0`. That certificate is about the *dual*, and it is
    // perfectly valid on an infeasible primal — the recession direction of an
    // empty feasible set still exists — so a problem can be, and often is, both
    // primal- and dual-infeasible at once. When it is, both verdicts are true
    // and the choice between them is a reporting decision. `PrimalInfeasible`
    // is the one to give: it is what pounce's own active-set engine and every
    // external oracle (HiGHS, Gurobi) report on such a model, and the one a
    // caller can act on — AMPL `solve_result_num=200`, "the model is
    // infeasible, fix it", rather than `300` (`DivergingIterates`).
    //
    // The two certificates cannot be separated inside the iteration: they are
    // residual races against the same iterate, and which gate clears first is
    // arbitrary. Measured on `w·x ≤ 1` with `w·x ≥ 3` (HiGHS: infeasible), the
    // Farkas value held at `−1.72` with `z ∈ K*` while its residual fell
    // `1.9e-3 → 9.5e-5 → 4.7e-6 → 2.4e-7` toward an `8.6e-11` gate — and the
    // recession gate opened with three orders still to go. Deciding it *inside*
    // the loop means picking a tolerance: tried, and a rule loose enough to
    // catch this case also suppressed 11 of 200 genuine unbounded verdicts,
    // trading one wrong answer for more missing ones.
    //
    // So decide it out here, where the question can be *asked directly* instead
    // of inferred: re-solve the objective-free twin (`P = 0, c = 0`), which has
    // the same feasible set and, having no objective, cannot be unbounded — its
    // only possible answers are "here is a feasible point" and "there is none".
    // A `PrimalInfeasible` twin means the recession direction was never about
    // unboundedness, and the verdict is corrected. Anything else leaves the
    // original verdict untouched, so a genuinely unbounded problem keeps it.
    //
    // Costs one extra solve, and only on a `DualInfeasible` verdict. The twin
    // cannot re-enter this branch: with `c = 0` no direction has `cᵀd < 0`, so
    // `DualInfeasible` is unreachable for it.
    if sol.status == QpStatus::DualInfeasible && !crate::debug_stop::requested() {
        if crate::deadline::expired() {
            return mark_timed_out(sol);
        }
        let twin = QpProblem {
            p_lower: Vec::new(),
            c: vec![0.0; prob.n],
            ..prob.clone()
        };
        if solve_qp_ipm_unscaled(&twin, opts, &mut make_backend, None).status
            == QpStatus::PrimalInfeasible
        {
            let mut infeasible = sol;
            infeasible.status = QpStatus::PrimalInfeasible;
            return infeasible;
        }
        return sol;
    }
    sol
}

/// Iteration budget for the gh #293 equilibrated retry — `max_iter` in
/// general, half that for a pure LP whose first solve merely failed to
/// converge. See the call site in [`solve_qp_ipm_core`] for the full rationale
/// and the corpus measurements behind the halving.
fn equilibrated_retry_budget(prob: &QpProblem, first: QpStatus, max_iter: usize) -> usize {
    let is_lp = prob.p_lower.iter().all(|t| t.val == 0.0);
    let certify_or_reject = matches!(
        first,
        QpStatus::IterationLimit | QpStatus::OptimalInaccurate
    );
    if is_lp && certify_or_reject {
        max_iter / 2
    } else {
        max_iter
    }
}

/// Run an equilibrated solve: Ruiz-scale `prob`, solve the scaled problem with
/// the driver selected by `use_hsde`, and unscale the result back to the
/// original problem's coordinates. Shared by the HSDE convergence fallback
/// (`use_hsde = true`) and the dual-infeasibility reverify guard
/// (`use_hsde = false`, the direct driver, which exposes the true curvature of
/// a tiny Hessian to the recession test).
fn equilibrated_solve<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    use_hsde: bool,
    make_backend: &mut F,
    hook: Option<&mut dyn DebugHook>,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let (scaled, scaling) = crate::equilibrate::equilibrate(prob);
    let inner = QpOptions {
        equilibrate: false,
        use_hsde,
        // The equilibrated objective is `σ` times the original's, so the
        // objective constant travels with it (`σ = 1` for a QP; only a pure
        // LP's cost normalization makes this anything else).
        obj_constant: opts.obj_constant * scaling.sigma(),
        ..*opts
    };
    // A hook rides into the *scaled* problem: this is the driver the caller's
    // solve actually runs, so it is the one a debugger has to watch, and the
    // blocks it sees are the Ruiz-scaled ones (the returned solution is
    // unscaled below, as always).
    let mut sol = solve_qp_ipm_unscaled(&scaled, &inner, make_backend, hook);
    scaling.unscale_solution(prob, &mut sol);
    if use_hsde {
        sol
    } else {
        demote_false_equilibrated_optimum(prob, sol, opts.tol)
    }
}

/// Re-check a direct-driver `Optimal` **in the caller's own coordinates**, and
/// demote it when it does not survive the trip back out of the Ruiz metric.
///
/// The direct driver's convergence test — absolute or scale-relative — is
/// applied to the *equilibrated* problem, and that is not the same statement as
/// optimality of the point the caller receives. Ruiz is a diagonal change of
/// variables `x = Dc x̂` whose dual map divides by `Dc`, so a `Dc` spanning many
/// decades multiplies the recovered dual residual by up to `1/min Dc`. On
/// `feasible_x0_sentinel_bound` (coefficients from `1e-320` to `1e30`, so
/// `min Dc ≈ 6e-16`) the returned iterate reads `‖r_d‖ = 2.3e-9` in the scaled
/// metric — comfortably converged — and `2.3` in the user's, at an objective of
/// `1.30` against a true `0`. That is the same class of false success gh #414
/// caught on the HSDE side, arriving through the opposite door: there the
/// unscaled test was blind and the equilibrated one decisive, here it is the
/// equilibrated test that is blind. So neither metric is trusted alone — a
/// point has to look optimal in *both* to keep the verdict.
///
/// Costs nothing on a solve that converged outright: a point whose *absolute*
/// KKT error in the user's coordinates is already within `tol` needs no
/// argument at all and short-circuits before any extra work.
///
/// Demotes to [`QpStatus::NumericalFailure`] rather than
/// [`QpStatus::OptimalInaccurate`], for the reason [`verify_or_repair_optimum`]
/// gives: "usable at reduced accuracy" still reports `ok` / exit 0 through the
/// CLI, which a point this far out is not.
fn demote_false_equilibrated_optimum(prob: &QpProblem, sol: QpSolution, tol: f64) -> QpSolution {
    if sol.status != QpStatus::Optimal
        || sol.kkt_residuals(prob).kkt_error() <= tol
        || normalized_optimum_is_genuine_relative(prob, &sol)
    {
        return sol;
    }
    QpSolution {
        status: QpStatus::NumericalFailure,
        ..sol
    }
}

/// The relative-KKT cut separating a genuine optimum from a scaling artifact,
/// shared by [`equilibrated_kkt_rel`] (gh #414) and
/// [`normalized_optimum_is_genuine_relative`] (gh #324). It is also the ceiling
/// on [`sigma_path_rel_tol`], so the `σ` path is never *looser* than this.
///
/// Measured on the #414 family, the false optima land in `2e-2‥1.2e2` and the
/// repaired optima of the very same problems in `1e-12‥1e-9`; the #286
/// huge-magnitude solves — genuine optima that only the *relative* arm can
/// certify, the regime most at risk of being rejected here — sit at `4e-10` and
/// `1.5e-8`. The cut therefore has better than an order of margin above every
/// genuine solve and four below every observed failure.
const FALSE_OPTIMUM_REL_TOL: f64 = 1e-3;

/// `sol`'s KKT residual relative to the scale of its own terms, measured **in
/// the Ruiz-equilibrated metric** — the scale-invariant answer to "is this
/// actually a KKT point of `prob`?".
///
/// Each residual is normalized by the natural magnitude of its own terms, the
/// same shape as HSDE's in-loop relative test (`crate::hsde`): stationarity by
/// the gradient scale `‖P̂x̂‖ ∨ ‖ĉ‖ ∨ ‖Ĝᵀẑ‖ ∨ ‖Âᵀŷ‖ ∨ ‖ẑ_lb‖ ∨ ‖ẑ_ub‖`, primal
/// by the rhs scale, and complementarity by the **objective** magnitude (its
/// terms `ŝᵢẑᵢ` are the duality gap's, and the gap's scale is the objective's,
/// not the gradient's — normalizing complementarity by a gradient scale that
/// Ruiz has pulled down to `O(1)` while `ŝᵀẑ` stays invariant would reject the
/// #286 huge-magnitude optima).
///
/// What makes it work where the *unscaled* relative test
/// ([`normalized_optimum_is_genuine`]) does not is the metric, not the formula.
/// Normalizing by a *global* ∞-norm in the original coordinates is blind to a
/// spread in the **variable** scales: one badly-scaled column makes `‖Px‖∞`
/// enormous, and dividing every component's residual by it grants a blanket
/// relaxation to the well-scaled components, where the real violation lives. On
/// the #414 family — `x_i ~ 1e-6‥1e6`, `cond(P) ~ 1e24`, trivially
/// well-conditioned after `z = x/s` — the returned point's unscaled relative
/// residual reads `2e-4`, inside any sane cut, while its true error is `O(1e3)`.
/// Ruiz equilibration is exactly the diagonal change of variables that removes
/// that spread, so in the scaled problem every variable and every row carries an
/// `O(1)` scale and no column can mask another: measured there, the same point
/// reads `1.2e2` against `2.9e-10` for the true optimum.
///
/// Orthant/box only: Ruiz is a per-row scaling, which is unsound for a
/// non-orthant cone (see [`crate::equilibrate`]), so callers must gate on the
/// cones being nonnegative.
fn equilibrated_kkt_rel(prob: &QpProblem, sol: &QpSolution, obj_constant: f64) -> f64 {
    equilibrated_kkt_rel_parts(prob, sol, obj_constant).kkt_error()
}

/// The three components [`equilibrated_kkt_rel`] takes the max of, each already
/// divided by its own normalizer — a [`QpResiduals`] holding *relative* numbers
/// rather than absolute ones.
///
/// Exposed because the active-set driver's post-loop adjudication (gh #641)
/// needs them separately: it relaxes only the stationarity and complementarity
/// terms to the relative measure and keeps primal feasibility absolute, in the
/// user's own coordinates. See `crate::active_set::adjudicated_kkt_error` for
/// why that split is the whole safety property. That path is orthant/box by
/// construction, satisfying the cone restriction above.
pub(crate) fn equilibrated_kkt_rel_parts(
    prob: &QpProblem,
    sol: &QpSolution,
    obj_constant: f64,
) -> QpResiduals {
    let (scaled, scaling) = crate::equilibrate::equilibrate(prob);
    let ssol = scaling.scale_solution(sol);
    let res = ssol.kkt_residuals(&scaled);
    let mut px = vec![0.0; scaled.n];
    scaled.p_mul(&ssol.x, &mut px);
    let mut gtz = vec![0.0; scaled.n];
    scaled.gt_mul(&ssol.z, &mut gtz);
    let mut aty = vec![0.0; scaled.n];
    scaled.at_mul(&ssol.y, &mut aty);
    let mut gx = vec![0.0; scaled.m_ineq()];
    scaled.g_mul(&ssol.x, &mut gx);
    let mut ax = vec![0.0; scaled.m_eq()];
    scaled.a_mul(&ssol.x, &mut ax);
    let gscale = inf_norm(&px)
        .max(inf_norm(&scaled.c))
        .max(inf_norm(&gtz))
        .max(inf_norm(&aty))
        .max(inf_norm(&ssol.z_lb))
        .max(inf_norm(&ssol.z_ub))
        .max(1.0);
    let pscale = inf_norm(&scaled.b)
        .max(inf_norm(&scaled.h))
        .max(inf_norm(&gx))
        .max(inf_norm(&ax))
        .max(1.0);
    // The objective of the *scaled* problem, not `sol.obj`. Every other
    // quantity here is measured in the equilibrated metric, and for a pure LP
    // the equilibration carries a cost scaling σ = 1/max|ĉ| that multiplies
    // `ĉ`, `ẑ` and hence both the dual residual and `ŝᵀẑ`. Dividing a
    // σ-scaled complementarity by an unscaled objective would leave the ratio
    // off by σ — up to `1e8` either way, a false accept on a large-cost LP and
    // a false *reject* on a tiny-cost one. Recomputing here keeps numerator
    // and denominator in one metric, and `σ` cancels exactly. (A QP keeps
    // σ = 1, so this is a no-op there.)
    // ...plus the caller's degree-0 objective term (`QpOptions::obj_constant`,
    // gh #689), in that same metric — the equilibration multiplies the
    // objective by `σ`, so the constant does too. `QpProblem` models
    // `½xᵀPx + cᵀx` only, so on a model whose objective carries a constant the
    // sum above is the caller's objective *displaced* by it, and normalizing by
    // the displaced value is what gh #712 was: `scaled_feasible_a` minimizes
    // `Σ(xᵢ−aᵢ)²` with `Σaᵢ² ≈ 5e11`, so a point whose absolute KKT error is
    // `2.3e3` read `4.6e-9` here and was certified. Told the constant, the
    // normalizer measures the objective the caller actually reads (`~0` at that
    // point, so the `max(1.0)` floor governs) and the same point reads `2.3e3`.
    // `0.0` — the default, and every library caller that does not set it — is
    // the tightest choice and leaves this bit-for-bit unchanged, which is what
    // keeps the gh #286 huge-magnitude optima (genuine large objectives, no
    // constant) certified by the only arm that can certify them.
    //
    // Note what the correction does on a least-squares model *at* its optimum:
    // the quadratic form and the constant are equal and opposite, their sum is
    // `~0`, the `max(1.0)` floor governs, and this arm silently becomes an
    // absolute test. That is right — the caller's objective really is `O(1)`
    // there — but it means the numerator can no longer be a product that only
    // large data made large, which is why the complementarity it divides is
    // [`resolvable_complementarity`] and not the raw residual.
    let cscale = (ssol
        .x
        .iter()
        .zip(&px)
        .zip(&scaled.c)
        .map(|((&xi, &pxi), &ci)| 0.5 * xi * pxi + ci * xi)
        .sum::<f64>()
        + obj_constant * scaling.sigma())
    .abs()
    .max(1.0);
    QpResiduals {
        primal_infeasibility: res.primal_infeasibility / pscale,
        // The box term rides the same primal normalizer as the aggregate it is
        // part of; no caller of this function reads it (the adjudication looks
        // at the three aggregates), but leaving it unnormalized would make it
        // incomparable with the `primal_infeasibility` beside it.
        bound_violation: res.bound_violation / pscale,
        dual_infeasibility: res.dual_infeasibility / gscale,
        complementarity: resolvable_complementarity(&scaled, &ssol) / cscale,
    }
}

/// The slack `a − b` is a difference of two computed quantities, so it is
/// quantised in units of `ε · max(|a|, |b|)`: no iterate can place it strictly
/// between `0` and that quantum, and which side of the quantum it lands on is
/// arithmetic luck rather than a statement about the point. `κ` covers the
/// accumulation over a row's nonzeros and the linear solve's conditioning on
/// top of the single subtraction — the same reading of "numerically zero", and
/// the same constant, the NLP-side primal residual uses
/// (`pounce_algorithm`'s `ROW_NOISE_KAPPA` / `primal_noise_floor_kappa`,
/// gh #446, gh #528).
const SLACK_NOISE_KAPPA: f64 = 64.0;

/// Whether a slack of `slack` between two quantities of size `magnitude` is
/// distinguishable from zero at all. See [`SLACK_NOISE_KAPPA`].
fn slack_is_resolvable(slack: f64, magnitude: f64) -> bool {
    slack.abs() > SLACK_NOISE_KAPPA * f64::EPSILON * magnitude
}

/// `max_i |sᵢ zᵢ|` over the complementarity pairs whose **slack is resolvable**
/// — the pairs where a nonzero product is evidence of anything (gh #712).
///
/// A pair whose slack sits under its own rounding quantum is complementary as
/// far as double precision can tell: the iterate is *at* that bound, and the
/// product it forms with a large multiplier measures the quantum, not a
/// violation. Counting it turns the scale-relative test into a floor on the
/// **data** scale — on `feasible_x0_wide_scale` the bound presolve derives for
/// `x₁` is `1.4e-8` wide next to `|x₁| ≈ 7.1e5` (`46` ulps), the converged
/// iterate sits `7e-9` inside it, and against a multiplier of `1.8e7` that is
/// a product of `0.13` on a point that matches the NLP oracle to 13 digits.
///
/// It does not soften a real violation: on `scaled_feasible_a`, the model this
/// measure exists to reject, the offending slack is `5e-6` against a quantum of
/// `8.5e-12` — six orders resolvable, and counted.
///
/// Only the *relative* measure abstains. The absolute residual
/// ([`QpSolution::kkt_residuals`]) is untouched, and it is what
/// [`optimum_is_genuine`] consults first.
fn resolvable_complementarity(prob: &QpProblem, sol: &QpSolution) -> f64 {
    let mut gx = vec![0.0; prob.m_ineq()];
    prob.g_mul(&sol.x, &mut gx);
    let mut comp = 0.0_f64;
    for ((&hi, &gxi), &zi) in prob.h.iter().zip(&gx).zip(&sol.z) {
        let s = hi - gxi;
        if slack_is_resolvable(s, hi.abs().max(gxi.abs())) {
            comp = comp.max((s * zi).abs());
        }
    }
    for i in 0..prob.n {
        let (lb, ub, xi) = (prob.lb_of(i), prob.ub_of(i), sol.x[i]);
        if lb > -1e19 && slack_is_resolvable(xi - lb, xi.abs().max(lb.abs())) {
            comp = comp.max(((xi - lb) * sol.z_lb[i]).abs());
        }
        if ub < 1e19 && slack_is_resolvable(ub - xi, xi.abs().max(ub.abs())) {
            comp = comp.max(((ub - xi) * sol.z_ub[i]).abs());
        }
    }
    comp
}

/// Whether an `Optimal` verdict is backed by a point that really is one.
///
/// Short-circuits on the *absolute* KKT residual: a point already accurate to
/// `tol` in the original coordinates is unimpeachable, needs no metric
/// argument, and pays nothing for this check — which is every well- and
/// moderately-scaled solve. Only a solve that reached `Optimal` through HSDE's
/// scale-*relative* convergence arm (`crate::hsde::relative_stop_permitted`,
/// the arm that opens once absolute `tol` accuracy is below the
/// finite-precision floor) is measured in the equilibrated metric.
fn optimum_is_genuine(prob: &QpProblem, sol: &QpSolution, tol: f64, obj_constant: f64) -> bool {
    sol.kkt_residuals(prob).kkt_error() <= tol
        || equilibrated_kkt_rel(prob, sol, obj_constant) <= FALSE_OPTIMUM_REL_TOL
}

/// Re-check an HSDE `Optimal` and, when it is a scaling artifact, repair or
/// demote it — never let a false success out (gh #414).
///
/// HSDE certifies convergence on *scale-relative* residuals once the problem's
/// natural scale puts absolute `tol` accuracy below the finite-precision floor
/// (`hsde::relative_stop_permitted`). Those normalizers are global ∞-norms, so
/// on a QP whose **variables** span many decades they are dominated by the
/// worst-scaled column and the test stops bounding the error in every other
/// direction: the embedding reports `Optimal` at a point whose own
/// `kkt_error` is `8.3e3` and whose objective is `67.13` against a true
/// `-3.96`. Downstream this is a success everywhere — `success=True` from
/// `solve_qp`, `SolveSucceeded` / `solve_result_num=0` / exit 0 from the
/// AMPL/Pyomo/GAMS drivers — so the wrong point is consumed as an answer.
///
/// The instance is not hard: it is well-conditioned after a diagonal rescaling,
/// which is precisely what Ruiz equilibration finds. So when the verdict fails
/// [`optimum_is_genuine`], re-solve equilibrated — the same repair gh #293
/// already applies to a *non*-converged HSDE solve, extended to the case where
/// the driver wrongly believes it converged. On the reported instance that
/// retry returns the oracle's `-3.958501808`.
///
/// If the retry cannot certify a genuine optimum either, the original verdict
/// is demoted to [`QpStatus::NumericalFailure`] rather than upgraded: the
/// solver has no certified answer, and saying so is the floor this function
/// guarantees. `OptimalInaccurate` would be the wrong demotion — it means
/// "usable at reduced accuracy" and still reports `ok` / exit 0 through the CLI,
/// which a relative residual of `1e-3` or worse is not.
///
/// A no-op unless the solve is HSDE-with-equilibration-allowed and ended
/// `Optimal`; the direct driver already runs *inside* the equilibrated metric
/// (its absolute test is applied to the Ruiz-scaled problem), so it cannot
/// reach this failure.
fn verify_or_repair_optimum<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    sol: QpSolution,
    make_backend: &mut F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if !(opts.use_hsde && opts.equilibrate)
        || sol.status != QpStatus::Optimal
        || optimum_is_genuine(prob, &sol, opts.tol, opts.obj_constant)
    {
        return sol;
    }
    let retry = equilibrated_solve(prob, opts, /* use_hsde */ true, make_backend, None);
    let genuine = |c: &QpSolution| {
        c.status == QpStatus::Optimal && optimum_is_genuine(prob, c, opts.tol, opts.obj_constant)
    };
    let err = |c: &QpSolution| c.kkt_residuals(prob).kkt_error();
    // A retry accurate to `tol` in the caller's own coordinates is
    // unimpeachable -- this function's own first test says so -- so stop here
    // and pay nothing more. This is the common repair.
    if genuine(&retry) && err(&retry) <= opts.tol {
        return retry;
    }
    // gh #846: otherwise the equilibrated *embedding* is not obviously the
    // best answer available, and on an ill-conditioned box QP it measurably is
    // not. Measured on a 9-variable diagonal box QP with `eig = [1e7 .. 1e13]`,
    // where the closed form is `clamp(t, -1, 1)` and needs no solver:
    //
    // | candidate                  | kkt_error | equil. rel | genuine | ‖x−x*‖∞ |
    // |----------------------------|-----------|------------|---------|---------|
    // | the point handed in        | 1.30e2    | 4.77e-3    | no      | 9.5e-7  |
    // | equilibrated HSDE retry    | 1.26e2    | 1.96e-10   | yes     | 1.1e-5  |
    // | equilibrated DIRECT driver | 1.22e-4   | 7.58e-24   | yes     | 1.1e-16 |
    //
    // The retry is genuine, so it used to be returned unconditionally -- and
    // it is *twelve times worse in x* than the point it replaced, while a
    // third candidate that is better on every measure at once (six orders of
    // absolute KKT, fourteen of equilibrated relative, eleven of `x`) was
    // never asked for. The embedding's stopping test normalizes its gap by the
    // objective's magnitude, which on data of this scale buys slack the direct
    // driver's absolute test does not.
    //
    // So both are asked, and the choice is made on **absolute `kkt_error` in
    // the caller's own coordinates** -- the caller's own definition of the
    // thing, and a ranking rather than another threshold to calibrate. Only
    // genuine candidates are eligible, so this cannot promote a point gh #414
    // exists to reject; it only stops that guard from settling for the first
    // acceptable answer when a better one is one solve away.
    //
    // The extra solve is on the repair path only, which is reached solely when
    // a claimed optimum has already failed [`optimum_is_genuine`].
    // `equilibrated_solve(.., use_hsde = false, ..)` is the same call this
    // function's neighbour already makes to refute a spurious unboundedness
    // certificate, on the same LP/QP-only entry point, so it carries no new
    // exposure to cone-carrying problems.
    let direct = equilibrated_solve(prob, opts, /* use_hsde */ false, make_backend, None);
    let best = [retry, direct]
        .into_iter()
        .filter(genuine)
        .min_by(|a, b| err(a).total_cmp(&err(b)));
    match best {
        Some(c) => c,
        // Neither could certify. The original verdict is demoted rather than
        // upgraded: the solver has no certified answer, and saying so is the
        // floor this function guarantees.
        None => QpSolution {
            status: QpStatus::NumericalFailure,
            ..sol
        },
    }
}

/// The bounds-aware orthant solve without equilibration (the historical
/// [`solve_qp_ipm`] body). Factored out so [`solve_qp_ipm`] can wrap it with
/// Ruiz scaling.
fn solve_qp_ipm_unscaled<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    make_backend: F,
    hook: Option<&mut dyn DebugHook>,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if !prob.has_bounds() {
        let cone = CompositeCone::single_nonneg(prob.m_ineq());
        return solve_qp_core(prob, &cone, opts, None, make_backend, hook);
    }
    let (expanded, bound_rows) = expand_bounds(prob);
    let cone = CompositeCone::single_nonneg(expanded.m_ineq());
    let sol = solve_qp_core(&expanded, &cone, opts, None, make_backend, hook);
    split_bound_duals(prob, &bound_rows, sol)
}

/// Solve a convex LP / QP with an interactive [`DebugHook`] attached: the
/// hook is fired at each interior-point checkpoint (iteration start, after
/// the Newton step, after the step is applied, and at termination) so a
/// debugger can step, inspect, and break on the solve.
///
/// **This is [`solve_qp_ipm`] with a hook, not a parallel implementation.**
/// It is the same function body, reached with `hook = Some(..)`, so driver
/// selection (`use_hsde`), Ruiz equilibration, the `σ` cost normalization,
/// every verify-and-retry guard and the LP crossover all run exactly as they
/// do without a debugger attached, and the returned solution is unchanged.
/// That identity is the point (gh #892): a debugger that substitutes a
/// different algorithm is debugging a run the user never shipped.
///
/// The hook rides the *primary* solve. The recovery re-solves — the
/// equilibrated retry, the `σ` re-solve, the reverify, the infeasibility twin
/// — run unhooked, so what the debugger observes is the solve proper. Which
/// blocks it sees follows from the driver that solve actually uses: the HSDE
/// drivers expose the embedding (with `tau`/`kappa`), the direct driver the
/// user's variables; finite bounds are expanded into a trailing nonnegative
/// block either way and surface in `s`/`z`.
pub fn solve_qp_ipm_debug<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    hook: &mut dyn DebugHook,
    make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    crate::deadline::with_deadline(opts.time_limit, || {
        // gh #880: see `solve_socp_ipm`. Outside `debug_stop::with_scope` so
        // the frame spans everything the scoped body may retry.
        let (sol, sigma_uncertified) = crate::sigma_verdict::tracking(|| {
            crate::debug_stop::with_scope(|| {
                solve_qp_ipm_scoped(prob, opts, make_backend, Some(hook))
            })
        });
        demote_uncertified_sigma_optimum(sol, sigma_uncertified)
    })
}

/// Solve a convex QP starting from a warm point (typically a previous
/// solution of a nearby problem). See [`QpWarmStart`] for the centering
/// strategy and when warm starting helps.
///
/// Identical to [`solve_qp_ipm`] except the interior-point iteration is
/// seeded from `warm` instead of the cold default. The *solution* is
/// independent of the start (the IPM converges to the same KKT point); a
/// good warm start only reduces the iteration count.
pub fn solve_qp_ipm_warm<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    warm: &QpWarmStart,
    make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    crate::deadline::with_deadline(opts.time_limit, || {
        if crate::deadline::expired() {
            timed_out_solution(prob)
        } else {
            // gh #880: see `solve_socp_ipm`. This entry forces a non-HSDE
            // options struct today, so it cannot reach `record` — but the
            // invariant the `record` assertion states is "every entry installs
            // the frame", and leaving the one exception to be rediscovered is
            // how F8 happened.
            let mut make_backend = make_backend;
            // gh #988: columns with `lb == ub` are removed by the cold path's
            // fixed-variable handling, so a warm point off the pin only
            // pushes the direct method away from the only feasible value.
            let pinned;
            let warm = if (0..prob.n).any(|i| {
                prob.lb_of(i) == prob.ub_of(i) && warm.x.get(i).is_some_and(|&x| x != prob.lb_of(i))
            }) {
                let mut w = warm.clone();
                for i in 0..prob.n {
                    if prob.lb_of(i) == prob.ub_of(i) && i < w.x.len() {
                        w.x[i] = prob.lb_of(i);
                    }
                }
                pinned = w;
                &pinned
            } else {
                warm
            };
            let (inner, sigma_uncertified) = crate::sigma_verdict::tracking(|| {
                solve_qp_ipm_warm_inner(prob, opts, warm, &mut make_backend)
            });
            // One gate over every exit of the body below — see [`finite_or_failed`].
            let mut sol = finite_or_failed(
                prob,
                demote_uncertified_sigma_optimum(inner, sigma_uncertified),
            );
            // gh #988: a warm start that cannot help must degrade to the cold
            // path. The warm leg runs the direct infeasible-start method,
            // which on an infeasible neighbour or a stiff fixed-column model
            // ends in `NumericalFailure` / `IterationLimit` where the cold
            // HSDE solve certifies or converges. Only a clean verdict (optimal
            // or a certified infeasibility) is kept.
            if !crate::deadline::expired()
                && matches!(
                    sol.status,
                    QpStatus::NumericalFailure
                        | QpStatus::IterationLimit
                        | QpStatus::OptimalInaccurate
                )
            {
                let cold = solve_qp_ipm(prob, opts, &mut make_backend);
                let cold_clean = matches!(
                    cold.status,
                    QpStatus::Optimal | QpStatus::PrimalInfeasible | QpStatus::DualInfeasible
                );
                let keep_inaccurate_warm = sol.status == QpStatus::OptimalInaccurate;
                if cold_clean || !keep_inaccurate_warm {
                    let warm_iters = sol.iters;
                    sol = cold;
                    sol.iters += warm_iters;
                }
            }
            if crate::deadline::expired() {
                mark_timed_out(sol)
            } else {
                sol
            }
        }
    })
}

fn solve_qp_ipm_warm_inner<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    warm: &QpWarmStart,
    make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    // Warm-starting requires the direct infeasible-start solver: HSDE
    // self-starts and ignores a warm point (see `QpOptions::use_hsde`). So this
    // path always runs the direct method, independent of the (HSDE) default —
    // otherwise the warm start would silently do nothing. A caller that
    // warm-starts is doing nearby reoptimization (a known-solvable
    // neighborhood), where the direct path's fragility is not a concern.
    // Screen the variable box before equilibration and bound expansion
    // (gh #295, gh #491). Equilibration is a diagonal congruence and scales
    // the bounds with the variables, so a crossing survives it — but only the
    // *unscaled* widths are comparable to `CROSSED_BOX_TOL`, which is why the
    // screen runs here and not after.
    let snapped;
    let prob = match screen_variable_box(prob) {
        BoxScreen::Feasible => prob,
        BoxScreen::Empty => return trivial_primal_infeasible_solution(prob),
        BoxScreen::Snapped(p) => {
            snapped = p;
            &snapped
        }
    };
    let direct = QpOptions {
        use_hsde: false,
        equilibrate: false,
        ..*opts
    };
    // (`direct.obj_constant` is fixed up below for the equilibrated branch,
    // whose scaled objective carries the cost scaling `σ`.)
    // Equilibrate (default on) just as the cold path does, mapping the
    // warm-start point into the scaled coordinates so the warm benefit is
    // preserved and the two paths run on identically-conditioned data.
    if opts.equilibrate {
        let (scaled, scaling) = crate::equilibrate::equilibrate(prob);
        let scaled_warm = scaling.scale_warm_start(warm);
        let direct = QpOptions {
            obj_constant: direct.obj_constant * scaling.sigma(),
            ..direct
        };
        let mut sol = solve_qp_ipm_warm_inner(&scaled, &direct, &scaled_warm, make_backend);
        scaling.unscale_solution(prob, &mut sol);
        // Same re-check the cold equilibrated path applies: a verdict reached
        // inside the Ruiz metric is not yet a statement about the point the
        // caller receives. See [`demote_false_equilibrated_optimum`].
        return demote_false_equilibrated_optimum(prob, sol, opts.tol);
    }
    if !prob.has_bounds() {
        let w = WarmStart {
            x: warm.x.clone(),
            y: warm.y.clone(),
            z: warm.z.clone(),
        };
        let cone = CompositeCone::single_nonneg(prob.m_ineq());
        return solve_qp_core(prob, &cone, &direct, Some(&w), make_backend, None);
    }
    let (expanded, bound_rows) = expand_bounds(prob);
    let w = WarmStart {
        x: warm.x.clone(),
        y: warm.y.clone(),
        z: merge_bound_duals(prob, &bound_rows, warm),
    };
    let cone = CompositeCone::single_nonneg(expanded.m_ineq());
    let sol = solve_qp_core(&expanded, &cone, &direct, Some(&w), make_backend, None);
    split_bound_duals(prob, &bound_rows, sol)
}

/// Solve a standard-form **SOCP** (or mixed LP/QP + second-order cones):
/// `min ½xᵀPx+cᵀx s.t. Ax=b, Gx ⪯_K h`, where the inequality block `Gx ≤ h`
/// is partitioned into the cones `K` described by `cones` (in row order;
/// each `s = h − Gx` block must lie in its cone). `cones` must cover the
/// `m_ineq` rows. Variable bounds (`lb`/`ub`) are appended as a trailing
/// nonnegative block.
pub fn solve_socp_ipm<F>(
    prob: &QpProblem,
    cones: &[ConeSpec],
    opts: &QpOptions,
    make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    crate::deadline::with_deadline(opts.time_limit, || {
        // gh #880: install the verdict frame at every entry that can reach the
        // cascade, not only the ones a corpus happens to exercise.
        // `solve_socp_ipm` passes the caller's `opts` through
        // `solve_socp_symmetric` unchanged, and `use_hsde` defaults to true,
        // so an all-nonneg (or empty) cone reaches `record` here — unlike
        // `solve_qp_ipm_warm_inner`, which forces a non-HSDE struct and
        // cannot. Before this, `solve_socp_ipm` returned the uncertified pick
        // as a clean `Optimal` on the same model `solve_qp_ipm` demotes.
        let (sol, sigma_uncertified) = crate::sigma_verdict::tracking(|| {
            solve_socp_ipm_scoped(prob, cones, opts, make_backend, None)
        });
        demote_uncertified_sigma_optimum(sol, sigma_uncertified)
    })
}

/// The body of [`solve_socp_ipm`], with an optional [`DebugHook`] threaded to
/// the primary solve. See [`solve_socp_ipm_debug`] for why the debugger enters
/// here rather than through a path of its own (gh #892).
fn solve_socp_ipm_scoped<F>(
    prob: &QpProblem,
    cones: &[ConeSpec],
    opts: &QpOptions,
    make_backend: F,
    hook: Option<&mut dyn DebugHook>,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if crate::deadline::expired() {
        return timed_out_solution(prob);
    }
    // One gate over every exit of the body below — see [`finite_or_failed`].
    let sol = finite_or_failed(
        prob,
        solve_socp_ipm_inner(prob, cones, opts, make_backend, hook),
    );
    if crate::deadline::expired() {
        mark_timed_out(sol)
    } else {
        sol
    }
}

fn solve_socp_ipm_inner<F>(
    prob: &QpProblem,
    cones: &[ConeSpec],
    opts: &QpOptions,
    make_backend: F,
    hook: Option<&mut dyn DebugHook>,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    // Screen the variable box before bound expansion (gh #295, gh #491);
    // `expand_bounds` is sign-agnostic. The screen reads only `lb`/`ub`, which
    // this path appends as a trailing nonnegative block exactly as the QP path
    // does, so the cones are unaffected by it either way.
    let snapped;
    let prob = match screen_variable_box(prob) {
        BoxScreen::Feasible => prob,
        BoxScreen::Empty => return trivial_primal_infeasible_solution(prob),
        BoxScreen::Snapped(p) => {
            snapped = p;
            &snapped
        }
    };
    // The cones must partition the inequality rows exactly; otherwise the
    // cone vectors and the `m_ineq` slack disagree and the driver would read
    // out of bounds (an exp/power cone is always 3 rows). Fail cleanly here.
    if !cone_dims_cover(cones, prob.m_ineq()) {
        return failed_solution(
            prob,
            vec![0.0; prob.n],
            vec![0.0; prob.m_eq()],
            vec![0.0; prob.m_ineq()],
            0,
        );
    }
    // Non-symmetric cones (exponential / power) route to the dedicated HSDE
    // driver; self-scaled cones (orthant / SOC / PSD) stay on the symmetric
    // path below. Mixing the two families in one problem is not supported.
    let has_nonsym = cones
        .iter()
        .any(|c| matches!(c, ConeSpec::Exponential | ConeSpec::Power(_)));
    let has_psd = cones.iter().any(|c| matches!(c, ConeSpec::Psd(_)));
    if has_nonsym && has_psd {
        return failed_solution(
            prob,
            vec![0.0; prob.n],
            vec![0.0; prob.m_eq()],
            vec![0.0; prob.m_ineq()],
            0,
        );
    }
    if has_nonsym {
        return solve_nonsym(prob, cones, opts, make_backend, hook);
    }
    // Sparsity: split any block-diagonal PSD cone into independent smaller
    // cones (one dense O(m²) KKT block → several small ones, exploited by the
    // sparse factorization). The transform is solution-equivalent; the dual
    // `z` is scattered back to the original row layout afterward.
    if has_psd {
        // First the cheap block-diagonal split (disjoint blocks → no new
        // variables); then chordal range-space decomposition of any still
        // connected-but-sparse PSD cone (introduces clique blocks + overlap
        // consistency equalities). Reconstruct the dual through both layers.
        let mut make_backend = make_backend;
        let (prob1, cones1, row_map) = decompose_psd(prob, cones);
        let (prob2, cones2, recon) = chordal_decompose(&prob1, &cones1);
        let run = |o: &QpOptions, mk: &mut F, hook: Option<&mut dyn DebugHook>| {
            let sol2 = solve_socp_symmetric(&prob2, &cones2, o, mk, hook);
            let sol1 = chordal_reconstruct(sol2, &recon, &prob1);
            remap_decomposed_z(sol1, &row_map, prob.m_ineq())
        };
        let sol = run(opts, &mut make_backend, hook);
        // gh #226: the direct symmetric driver is known-weak on PSD programs
        // whose optimum sits on the cone boundary (a rank-deficient slack,
        // where the NT scaling's condition number blows up) — a small
        // fraction of well-posed instances stall or break down there while
        // the HSDE embedding solves them cleanly. When a caller opted out of
        // HSDE and the direct solve ended without a full answer, retry once
        // with the embedding, mirroring the reverse-direction fallback in
        // `solve_qp_ipm_core`. Sound for the same reason: the direct solve
        // already failed, so there is nothing left to regress — the retry is
        // kept only when it is a strict upgrade. Verified infeasibility /
        // unboundedness certificates are proofs, not failures, and are never
        // second-guessed.
        if !opts.use_hsde
            && matches!(
                sol.status,
                QpStatus::NumericalFailure | QpStatus::IterationLimit | QpStatus::OptimalInaccurate
            )
            && !crate::debug_stop::requested()
        {
            let hsde_opts = QpOptions {
                use_hsde: true,
                ..*opts
            };
            let retry = run(&hsde_opts, &mut make_backend, None);
            if hsde_retry_is_upgrade(sol.status, retry.status) {
                return retry;
            }
        }
        return sol;
    }
    let mut make_backend = make_backend;
    let sol = solve_socp_symmetric(prob, cones, opts, &mut make_backend, hook);
    // gh #414: a cone program whose cones are *all* nonnegative is an LP/QP
    // wearing the conic entry point's clothes (`solver_selection=socp` on a
    // box-constrained QP lands here), and it inherits the same false `Optimal`
    // under a variable-scale spread. Ruiz equilibration — which the repair
    // rests on — is a per-row scaling and stays sound exactly on the orthant,
    // so the check is gated on that and every genuine cone program is
    // untouched. See [`verify_or_repair_optimum`].
    if cones.iter().all(|c| matches!(c, ConeSpec::Nonneg(_))) && !crate::debug_stop::requested() {
        return verify_or_repair_optimum(prob, opts, sol, &mut make_backend);
    }
    sol
}

/// Debug-enabled [`solve_socp_ipm`]: fires the interactive [`DebugHook`] at
/// each interior-point checkpoint.
///
/// **This is [`solve_socp_ipm`] with a hook, not a parallel implementation.**
/// It enters the same body with `hook = Some(..)`, so cone routing, driver
/// selection (`use_hsde`), the `σ` cost normalization, the PSD chordal
/// decomposition and every verify-and-retry guard run exactly as they do
/// without a debugger, and the returned solution is unchanged.
///
/// gh #892 is why that identity is spelled out. This entry point used to
/// build its own factorization and call the core loop directly for symmetric
/// cones, which never consulted `use_hsde` — so attaching the debugger
/// silently substituted the *direct* IPM for the default HSDE embedding, and
/// with it dropped the `σ` normalization and the equilibrate-and-verify
/// guards. On a 5-variable convex QCQP that turned an `Optimal` agreeing with
/// Clarabel to `1.3e-10` into `NumericalFailure`; the iteration count moved on
/// every instance tried. A debugger that changes the trajectory is debugging a
/// run the user never shipped.
///
/// The hook rides the *primary* solve; the recovery re-solves (the HSDE retry
/// on a failed PSD solve, the `σ` re-solves, [`verify_or_repair_optimum`]) run
/// unhooked. Which blocks the hook sees therefore follows from the driver that
/// solve actually uses — the HSDE drivers expose the embedding, with
/// `tau`/`kappa`; the direct driver (`qp_hsde=no`) exposes the user's
/// variables — and a decomposed PSD cone is debugged in its clique blocks.
pub fn solve_socp_ipm_debug<F>(
    prob: &QpProblem,
    cones: &[ConeSpec],
    opts: &QpOptions,
    hook: &mut dyn DebugHook,
    make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    crate::deadline::with_deadline(opts.time_limit, || {
        // gh #880: see `solve_socp_ipm`.
        let (sol, sigma_uncertified) = crate::sigma_verdict::tracking(|| {
            crate::debug_stop::with_scope(|| {
                solve_socp_ipm_scoped(prob, cones, opts, make_backend, Some(hook))
            })
        });
        demote_uncertified_sigma_optimum(sol, sigma_uncertified)
    })
}

/// The symmetric-cone solve (orthant / SOC / PSD): expand finite bounds into
/// a trailing orthant block, run the Mehrotra core, and split the bound
/// duals back out. Shared by [`solve_socp_ipm`] and the PSD-decomposed path.
fn solve_socp_symmetric<F>(
    prob: &QpProblem,
    cones: &[ConeSpec],
    opts: &QpOptions,
    make_backend: F,
    hook: Option<&mut dyn DebugHook>,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if !prob.has_bounds() {
        let cone = CompositeCone::from_specs(cones);
        return solve_qp_core(prob, &cone, opts, None, make_backend, hook);
    }
    // Bounds expand into a trailing nonnegative block after the user cones.
    let (expanded, bound_rows) = expand_bounds(prob);
    let mut specs = cones.to_vec();
    specs.push(ConeSpec::Nonneg(bound_rows.len()));
    let cone = CompositeCone::from_specs(&specs);
    let sol = solve_qp_core(&expanded, &cone, opts, None, make_backend, hook);
    split_bound_duals(prob, &bound_rows, sol)
}

/// Scatter the inequality dual `z` of a PSD-decomposed solve back to the
/// original inequality-row layout: new row `r` maps to `row_map[r]`, and the
/// dropped cross-block rows (structurally zero; their `G` rows are empty so
/// they carry no stationarity term) take dual `0`. Everything else
/// (`x`/`y`/bound duals/objective) is unchanged by the decomposition.
fn remap_decomposed_z(sol: QpSolution, row_map: &[usize], orig_m_ineq: usize) -> QpSolution {
    let mut z = vec![0.0; orig_m_ineq];
    for (new_r, &orig_r) in row_map.iter().enumerate() {
        z[orig_r] = sol.z[new_r];
    }
    QpSolution { z, ..sol }
}

/// Split each block-diagonal `Psd(n)` cone into independent PSD cones over
/// the connected components of its aggregate sparsity graph.
///
/// A `Psd(n)` cone occupies `n(n+1)/2` `svec` rows of `(G, h)`. Treating the
/// matrix indices `0..n` as graph vertices and adding an edge `(i,j)` for
/// every *structurally present* off-diagonal `svec` row (nonzero `h` or a
/// non-empty `G` row), the connected components partition the matrix into
/// diagonal blocks: cross-component entries are structurally zero, so
/// `smat(s)` is block-diagonal and `⪰ 0` iff each block is. The cone is then
/// replaced by one `Psd(|C|)` per component `C` (its lower triangle pulled
/// from the original rows, in `svec` order), and the cross-component rows are
/// dropped. Non-PSD cones and undecomposable PSD cones pass through unchanged.
///
/// Returns `(transformed problem, transformed cones, new→original ineq-row
/// map)`. This turns one dense `O((n(n+1)/2)²)` KKT block into several small
/// ones — the first (non-overlapping) rung of chordal sparsity for SDPs.
pub(crate) fn decompose_psd(
    prob: &QpProblem,
    cones: &[ConeSpec],
) -> (QpProblem, Vec<ConeSpec>, Vec<usize>) {
    use crate::qp::Triplet;
    let m_ineq = prob.m_ineq();
    let mut rows_of_g: Vec<Vec<Triplet>> = vec![Vec::new(); m_ineq];
    for t in &prob.g {
        rows_of_g[t.row].push(*t);
    }

    let mut new_g: Vec<Triplet> = Vec::new();
    let mut new_h: Vec<f64> = Vec::new();
    let mut new_cones: Vec<ConeSpec> = Vec::new();
    let mut row_map: Vec<usize> = Vec::new();

    // Copy original ineq row `r` to a fresh row at the end of `new_g`/`new_h`.
    let emit =
        |r: usize, new_g: &mut Vec<Triplet>, new_h: &mut Vec<f64>, row_map: &mut Vec<usize>| {
            let nr = new_h.len();
            for t in &rows_of_g[r] {
                new_g.push(Triplet::new(nr, t.col, t.val));
            }
            new_h.push(prob.h[r]);
            row_map.push(r);
        };

    let mut off = 0usize;
    for c in cones {
        let d = c.dim();
        match c {
            ConeSpec::Psd(n) => {
                let n = *n;
                // svec local order: (i,j) for j in 0..n, i in j..n.
                let mut kij: Vec<(usize, usize)> = Vec::with_capacity(d);
                for j in 0..n {
                    for i in j..n {
                        kij.push((i, j));
                    }
                }
                // Union-find over the matrix indices.
                let mut parent: Vec<usize> = (0..n).collect();
                fn find(parent: &mut [usize], x: usize) -> usize {
                    let mut r = x;
                    while parent[r] != r {
                        r = parent[r];
                    }
                    let mut cur = x;
                    while parent[cur] != r {
                        let nxt = parent[cur];
                        parent[cur] = r;
                        cur = nxt;
                    }
                    r
                }
                for (k, &(i, j)) in kij.iter().enumerate() {
                    if i != j {
                        let r = off + k;
                        let present = prob.h[r] != 0.0 || !rows_of_g[r].is_empty();
                        if present {
                            let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                            if ri != rj {
                                parent[ri] = rj;
                            }
                        }
                    }
                }
                // Components, in ascending-vertex order.
                let mut comps: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
                for v in 0..n {
                    let root = find(&mut parent, v);
                    comps.entry(root).or_default().push(v);
                }
                if comps.len() <= 1 {
                    // Nothing to split: copy the cone's rows through unchanged.
                    for k in 0..d {
                        emit(off + k, &mut new_g, &mut new_h, &mut row_map);
                    }
                    new_cones.push(ConeSpec::Psd(n));
                } else {
                    // Global (i,j) → local svec index `k`.
                    let mut idx = std::collections::HashMap::with_capacity(d);
                    for (k, &(i, j)) in kij.iter().enumerate() {
                        idx.insert((i, j), k);
                    }
                    for comp in comps.values() {
                        let cn = comp.len();
                        // Each component's own lower triangle, in svec order.
                        for jj in 0..cn {
                            for ii in jj..cn {
                                // comp is ascending, so comp[ii] ≥ comp[jj].
                                let k = idx[&(comp[ii], comp[jj])];
                                emit(off + k, &mut new_g, &mut new_h, &mut row_map);
                            }
                        }
                        new_cones.push(ConeSpec::Psd(cn));
                    }
                    // Cross-component rows are structurally zero → dropped.
                }
            }
            _ => {
                for k in 0..d {
                    emit(off + k, &mut new_g, &mut new_h, &mut row_map);
                }
                new_cones.push(*c);
            }
        }
        off += d;
    }

    let new_prob = QpProblem {
        g: new_g,
        h: new_h,
        ..prob.clone()
    };
    (new_prob, new_cones, row_map)
}

/// Where a (post-block-split) inequality row's dual comes from after the
/// chordal range-space reformulation.
enum ZSrc {
    /// A row copied verbatim — its dual is `z[aug_ineq_row]`.
    Ineq(usize),
    /// A PSD entry that became a consistency equality — its dual is the
    /// equality multiplier `y[aug_eq_row]`.
    Eq(usize),
    /// A dropped (out-of-pattern) entry — dual `0`.
    Zero,
}

/// Bookkeeping to map an augmented solve back to the pre-chordal layout.
pub(crate) struct ChordalRecon {
    orig_n: usize,
    orig_m_eq: usize,
    orig_m_ineq: usize,
    z_src: Vec<ZSrc>,
}

/// Range-space chordal decomposition of any connected-but-sparse PSD cone.
///
/// For a `Psd(n)` cone whose sparsity pattern is chordal with overlapping
/// maximal cliques `C₁…C_p`, the slack `s ⪰ 0` is rewritten as
/// `s = Σ_k Tᵀ_{C_k} S_k T_{C_k}` with each `S_k ⪰ 0` (Agler et al.). This
/// introduces clique matrix variables `w_k = svec(S_k)` (appended to `x`,
/// each constrained `⪰ 0` by a small `Psd(|C_k|)` cone), and one **consistency
/// equality** per clique-covered entry — `(h − Gx)ᵢⱼ = Σ_{k∋(i,j)} (S_k)ᵢⱼ` —
/// replacing the one dense `O(m²)` block with several small ones. Entries
/// outside every clique are structurally zero and dropped.
///
/// Dense or already-decomposed PSD cones (and all non-PSD cones) pass through
/// unchanged. Returns `(augmented problem, augmented cones, reconstruction)`.
pub(crate) fn chordal_decompose(
    prob: &QpProblem,
    cones: &[ConeSpec],
) -> (QpProblem, Vec<ConeSpec>, ChordalRecon) {
    use crate::cones::chordal;
    use crate::cones::psd::svec_index;
    use crate::qp::Triplet;
    use std::collections::HashMap;

    let orig_n = prob.n;
    let orig_m_eq = prob.m_eq();
    let orig_m_ineq = prob.m_ineq();

    let mut rows_of_g: Vec<Vec<Triplet>> = vec![Vec::new(); orig_m_ineq];
    for t in &prob.g {
        rows_of_g[t.row].push(*t);
    }

    let mut aug_g: Vec<Triplet> = Vec::new();
    let mut aug_h: Vec<f64> = Vec::new();
    let mut aug_cones: Vec<ConeSpec> = Vec::new();
    let mut aug_a: Vec<Triplet> = prob.a.clone();
    let mut aug_b: Vec<f64> = prob.b.clone();
    let mut z_src: Vec<ZSrc> = (0..orig_m_ineq).map(|_| ZSrc::Zero).collect();
    let mut aug_n = orig_n;
    let mut eq_row = orig_m_eq; // next augmented equality row index

    let mut off = 0usize;
    for c in cones {
        let d = c.dim();
        let decompose = match c {
            ConeSpec::Psd(n) if *n >= 2 => Some(*n),
            _ => None,
        };
        let cliques = decompose.and_then(|n| {
            let mut edges = Vec::new();
            for j in 0..n {
                for i in (j + 1)..n {
                    let r = off + svec_index(n, i, j);
                    if prob.h[r] != 0.0 || !rows_of_g[r].is_empty() {
                        edges.push((i, j));
                    }
                }
            }
            let ch = chordal::analyze(n, &edges);
            // Only worth it when it genuinely splits into >1 clique.
            (ch.cliques.len() > 1).then_some((n, ch.cliques))
        });

        match cliques {
            None => {
                // Copy this cone's rows verbatim.
                for k in 0..d {
                    let nr = aug_h.len();
                    for t in &rows_of_g[off + k] {
                        aug_g.push(Triplet::new(nr, t.col, t.val));
                    }
                    aug_h.push(prob.h[off + k]);
                    z_src[off + k] = ZSrc::Ineq(nr);
                }
                aug_cones.push(*c);
            }
            Some((n, cl_list)) => {
                // Allocate a clique block per maximal clique and a Psd cone
                // (s = w_k via G = −I) enforcing S_k ⪰ 0.
                let mut clique_cols: Vec<(Vec<usize>, usize)> = Vec::new();
                for cl in &cl_list {
                    let cn = cl.len();
                    let wbase = aug_n;
                    aug_n += cn * (cn + 1) / 2;
                    for jj in 0..cn {
                        for ii in jj..cn {
                            let nr = aug_h.len();
                            aug_g.push(Triplet::new(nr, wbase + svec_index(cn, ii, jj), -1.0));
                            aug_h.push(0.0);
                        }
                    }
                    aug_cones.push(ConeSpec::Psd(cn));
                    clique_cols.push((cl.clone(), wbase));
                }
                // Position of each vertex within each clique.
                let pos: Vec<HashMap<usize, usize>> = cl_list
                    .iter()
                    .map(|cl| cl.iter().enumerate().map(|(p, &v)| (v, p)).collect())
                    .collect();
                // One consistency equality per clique-covered entry.
                for j in 0..n {
                    for i in j..n {
                        let k = svec_index(n, i, j);
                        let r = off + k;
                        // Cliques containing both i and j contribute (S_k)ᵢⱼ.
                        let mut w_terms: Vec<usize> = Vec::new();
                        for (ci, (cl, wbase)) in clique_cols.iter().enumerate() {
                            if let (Some(&pi), Some(&pj)) = (pos[ci].get(&i), pos[ci].get(&j)) {
                                let (a, b) = if pi >= pj { (pi, pj) } else { (pj, pi) };
                                let _ = cl;
                                w_terms.push(wbase + svec_index(cl.len(), a, b));
                            }
                        }
                        if w_terms.is_empty() {
                            continue; // out-of-pattern entry: dropped (s = 0)
                        }
                        // (h − Gx)_r = Σ w  ⇔  Gx + Σ w = h_r  (equality `eq_row`).
                        for t in &rows_of_g[r] {
                            aug_a.push(Triplet::new(eq_row, t.col, t.val));
                        }
                        for &wc in &w_terms {
                            aug_a.push(Triplet::new(eq_row, wc, 1.0));
                        }
                        aug_b.push(prob.h[r]);
                        z_src[r] = ZSrc::Eq(eq_row);
                        eq_row += 1;
                    }
                }
            }
        }
        off += d;
    }

    // Augmented variable vector x' = (x, w): objective and Hessian carry no
    // `w` terms, bounds (if any) extend as free.
    let mut c_aug = prob.c.clone();
    c_aug.resize(aug_n, 0.0);
    let (lb, ub) = if prob.has_bounds() {
        let mut lb = prob.lb.clone();
        let mut ub = prob.ub.clone();
        lb.resize(aug_n, crate::qp::NEG_INF);
        ub.resize(aug_n, crate::qp::POS_INF);
        (lb, ub)
    } else {
        (Vec::new(), Vec::new())
    };
    let aug_prob = QpProblem {
        n: aug_n,
        p_lower: prob.p_lower.clone(),
        c: c_aug,
        a: aug_a,
        b: aug_b,
        g: aug_g,
        h: aug_h,
        lb,
        ub,
    };
    let recon = ChordalRecon {
        orig_n,
        orig_m_eq,
        orig_m_ineq,
        z_src,
    };
    (aug_prob, aug_cones, recon)
}

/// Map a solve of the chordal-augmented problem back to the pre-chordal
/// layout: the primal/objective are unchanged on the original variables, and
/// each PSD dual entry is recovered from its consistency-equality multiplier
/// (a clique-covered entry), a copied row's dual, or `0` (dropped entry).
fn chordal_reconstruct(sol: QpSolution, recon: &ChordalRecon, _prob1: &QpProblem) -> QpSolution {
    let mut z = vec![0.0; recon.orig_m_ineq];
    for (r, src) in recon.z_src.iter().enumerate() {
        z[r] = match *src {
            ZSrc::Ineq(ar) => sol.z[ar],
            ZSrc::Eq(er) => sol.y[er],
            ZSrc::Zero => 0.0,
        };
    }
    QpSolution {
        status: sol.status,
        x: sol.x[..recon.orig_n].to_vec(),
        y: sol.y[..recon.orig_m_eq].to_vec(),
        z,
        z_lb: sol.z_lb[..recon.orig_n].to_vec(),
        z_ub: sol.z_ub[..recon.orig_n].to_vec(),
        obj: sol.obj,
        iters: sol.iters,
        iterates: sol.iterates,
    }
}

/// Warm-started [`solve_socp_ipm`]: seed the iteration from `warm` (a nearby
/// SOCP's solution). The warm `(s, z)` are projected into each cone's
/// interior (orthant positivity / SOC `λ_min` floor); the solution is
/// start-independent, so warm starting is intended to reduce iterations when
/// compared with the same direct driver. The default cold solve uses HSDE,
/// however, while symmetric warm solves are forced onto the direct driver;
/// SOC-heavy problems can therefore take more iterations than cold HSDE and
/// may return `OptimalInaccurate`, a truthful reduced-accuracy KKT result with
/// the same objective contract.
/// Finite variable bounds are first-class and they are expanded into a trailing
/// nonnegative cone block and the returned bound multipliers are restored to
/// `z_lb`/`z_ub`.
///
/// Warm starts for symmetric cones always use the direct (non-HSDE) driver.
/// Non-symmetric exponential/power cones use their dedicated cold HSDE route
/// because that driver has no warm-start plumbing. When `opts.use_hsde` is
/// true, a cold HSDE solve is retried if a symmetric direct warm attempt fails
/// without producing a usable answer. `OptimalInaccurate` is usable and is
/// returned directly, preserving the benefit of the warm start.
pub fn solve_socp_ipm_warm<F>(
    prob: &QpProblem,
    cones: &[ConeSpec],
    warm: &QpWarmStart,
    opts: &QpOptions,
    make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    crate::deadline::with_deadline(opts.time_limit, || {
        if crate::deadline::expired() {
            timed_out_solution(prob)
        } else {
            // gh #880: see `solve_socp_ipm`. This entry also reaches the
            // cascade through its own HSDE fallback, which calls
            // `solve_socp_ipm_inner` directly rather than `solve_socp_ipm`,
            // so the frame has to be here rather than one layer in.
            let (inner, sigma_uncertified) = crate::sigma_verdict::tracking(|| {
                solve_socp_ipm_warm_scoped(prob, cones, warm, opts, make_backend)
            });
            let sol = finite_or_failed(
                prob,
                demote_uncertified_sigma_optimum(inner, sigma_uncertified),
            );
            if crate::deadline::expired() {
                mark_timed_out(sol)
            } else {
                sol
            }
        }
    })
}

fn solve_socp_ipm_warm_scoped<F>(
    prob: &QpProblem,
    cones: &[ConeSpec],
    warm: &QpWarmStart,
    opts: &QpOptions,
    mut make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if crate::deadline::expired() {
        return timed_out_solution(prob);
    }
    let snapped;
    let prob = match screen_variable_box(prob) {
        BoxScreen::Feasible => prob,
        BoxScreen::Empty => return trivial_primal_infeasible_solution(prob),
        BoxScreen::Snapped(p) => {
            snapped = p;
            &snapped
        }
    };
    if !cone_dims_cover(cones, prob.m_ineq()) {
        return failed_solution(
            prob,
            vec![0.0; prob.n],
            vec![0.0; prob.m_eq()],
            vec![0.0; prob.m_ineq()],
            0,
        );
    }
    let has_nonsym = cones
        .iter()
        .any(|c| matches!(c, ConeSpec::Exponential | ConeSpec::Power(_)));

    // Non-symmetric cones have no warm-start plumbing yet, so use the same
    // cold HSDE route as `solve_socp_ipm` (the `use_hsde` flag is immaterial
    // to this dedicated driver). Mixed non-symmetric/PSD products remain an
    // unsupported combination, matching the cold entry point.
    let direct = if has_nonsym {
        if cones.iter().any(|c| matches!(c, ConeSpec::Psd(_))) {
            failed_solution(
                prob,
                vec![0.0; prob.n],
                vec![0.0; prob.m_eq()],
                vec![0.0; prob.m_ineq()],
                0,
            )
        } else {
            return solve_nonsym(prob, cones, opts, &mut make_backend, None);
        }
    } else {
        let direct_opts = QpOptions {
            use_hsde: false,
            ..*opts
        };
        let (expanded, bound_rows) = expand_bounds(prob);
        let mut specs = cones.to_vec();
        if !bound_rows.is_empty() {
            specs.push(ConeSpec::Nonneg(bound_rows.len()));
        }
        let cone = CompositeCone::from_specs(&specs);
        let w = WarmStart {
            x: warm.x.clone(),
            y: warm.y.clone(),
            z: merge_bound_duals(prob, &bound_rows, warm),
        };
        let sol = solve_qp_core(
            &expanded,
            &cone,
            &direct_opts,
            Some(&w),
            &mut make_backend,
            None,
        );
        split_bound_duals(prob, &bound_rows, sol)
    };

    // `use_hsde` is the fallback permission here, not the initial-driver
    // selector.
    if opts.use_hsde && warm_hsde_retry_needed(direct.status) && !crate::deadline::expired() {
        let hsde_opts = QpOptions {
            use_hsde: true,
            ..*opts
        };
        let retry = solve_socp_ipm_inner(prob, cones, &hsde_opts, &mut make_backend, None);
        if hsde_retry_is_upgrade(direct.status, retry.status) {
            return retry;
        }
    }
    direct
}

/// Whether a direct warm result has no usable answer and therefore warrants a
/// cold HSDE retry. `OptimalInaccurate` deliberately stays out of this set:
/// it is a usable, certified-to-reduced-accuracy result, and retrying would
/// discard the warm solve's iteration savings.
fn warm_hsde_retry_needed(status: QpStatus) -> bool {
    matches!(
        status,
        QpStatus::NumericalFailure | QpStatus::IterationLimit
    )
}

/// Whether a retry has strictly more useful status information than the
/// original solve. A clean optimum always wins; a failed solve can be
/// replaced by any usable verdict, while an inaccurate result is replaced
/// only by a clean optimum.
fn hsde_retry_is_upgrade(original: QpStatus, retry: QpStatus) -> bool {
    match (original, retry) {
        (_, QpStatus::Optimal) => true,
        (
            QpStatus::NumericalFailure | QpStatus::IterationLimit,
            QpStatus::OptimalInaccurate | QpStatus::PrimalInfeasible | QpStatus::DualInfeasible,
        ) => true,
        _ => false,
    }
}

#[cfg(test)]
mod warm_hsde_fallback_tests {
    use super::{hsde_retry_is_upgrade, warm_hsde_retry_needed};
    use crate::qp::QpStatus;

    #[test]
    fn reduced_accuracy_warm_result_is_not_retried() {
        assert!(!warm_hsde_retry_needed(QpStatus::OptimalInaccurate));
        assert!(warm_hsde_retry_needed(QpStatus::NumericalFailure));
        assert!(warm_hsde_retry_needed(QpStatus::IterationLimit));
    }

    #[test]
    fn retry_replaces_only_with_strictly_better_status() {
        assert!(hsde_retry_is_upgrade(
            QpStatus::NumericalFailure,
            QpStatus::OptimalInaccurate
        ));
        assert!(hsde_retry_is_upgrade(
            QpStatus::IterationLimit,
            QpStatus::PrimalInfeasible
        ));
        assert!(hsde_retry_is_upgrade(
            QpStatus::OptimalInaccurate,
            QpStatus::Optimal
        ));
        assert!(!hsde_retry_is_upgrade(
            QpStatus::OptimalInaccurate,
            QpStatus::PrimalInfeasible
        ));
    }
}

/// Route a problem whose cone product contains an **exponential** cone to the
/// non-symmetric HSDE driver ([`crate::hsde_nonsym`]). Orthant, second-order,
/// exponential, and power blocks are all supported (a second-order cone may be
/// mixed with a non-symmetric one). Variable bounds expand into a trailing
/// orthant block exactly as in the symmetric path.
fn solve_nonsym<F>(
    prob: &QpProblem,
    cones: &[ConeSpec],
    opts: &QpOptions,
    make_backend: F,
    hook: Option<&mut dyn DebugHook>,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    use crate::hsde_nonsym::{NsBlock, solve_conic_hsde_nonsym, solve_conic_hsde_nonsym_debug};

    fn blocks_of(cones: &[ConeSpec], extra_orthant: usize) -> Vec<NsBlock> {
        let mut blocks = Vec::with_capacity(cones.len() + 1);
        for c in cones {
            match c {
                ConeSpec::Nonneg(n) => blocks.push(NsBlock::Orthant(*n)),
                ConeSpec::SecondOrder(m) => blocks.push(NsBlock::SecondOrder(*m)),
                ConeSpec::Exponential => blocks.push(NsBlock::exp()),
                ConeSpec::Power(a) => blocks.push(NsBlock::power(*a)),
                // PSD is self-scaled and runs on the symmetric driver; the
                // PSD-with-exp/power mix is rejected upstream in
                // `solve_socp_ipm`, so this arm is never reached.
                ConeSpec::Psd(_) => {
                    unreachable!("PSD cone routes to the symmetric driver, not hsde_nonsym")
                }
            }
        }
        if extra_orthant > 0 {
            blocks.push(NsBlock::Orthant(extra_orthant));
        }
        blocks
    }

    if !prob.has_bounds() {
        let blocks = blocks_of(cones, 0);
        // Exact cone-domain infeasibility screen (gh #283): a power/exp cone
        // coordinate pinned strictly outside its `≥ 0` domain proves primal
        // infeasibility, which the HSDE's residual-gated Farkas detector misses.
        if crate::hsde_nonsym::detect_cone_domain_infeasible(prob, &blocks) {
            return trivial_primal_infeasible_solution(prob);
        }
        return match hook {
            Some(h) => solve_conic_hsde_nonsym_debug(prob, &blocks, opts, h, make_backend),
            None => solve_conic_hsde_nonsym(prob, &blocks, opts, make_backend),
        };
    }
    let (expanded, bound_rows) = expand_bounds(prob);
    let blocks = blocks_of(cones, bound_rows.len());
    if crate::hsde_nonsym::detect_cone_domain_infeasible(&expanded, &blocks) {
        return trivial_primal_infeasible_solution(prob);
    }
    let sol = match hook {
        Some(h) => solve_conic_hsde_nonsym_debug(&expanded, &blocks, opts, h, make_backend),
        None => solve_conic_hsde_nonsym(&expanded, &blocks, opts, make_backend),
    };
    split_bound_duals(prob, &bound_rows, sol)
}

/// Expand a problem's finite variable bounds into extra `G` rows
/// (`x_i ≤ ub_i` and `−x_i ≤ −lb_i`), returning the bounds-free expanded
/// problem and the `(row, var, is_upper)` provenance of each appended row
/// so the bound multipliers can be split back out.
fn expand_bounds(prob: &QpProblem) -> (QpProblem, Vec<(usize, usize, bool)>) {
    let mut g = prob.g.clone();
    let mut h = prob.h.clone();
    let mut bound_rows: Vec<(usize, usize, bool)> = Vec::new();
    for i in 0..prob.n {
        let ub = prob.ub_of(i);
        if ub < crate::qp::BOUND_INF {
            let r = h.len();
            g.push(crate::qp::Triplet::new(r, i, 1.0));
            h.push(ub);
            bound_rows.push((r, i, true));
        }
        let lb = prob.lb_of(i);
        if lb > -crate::qp::BOUND_INF {
            let r = h.len();
            g.push(crate::qp::Triplet::new(r, i, -1.0));
            h.push(-lb);
            bound_rows.push((r, i, false));
        }
    }
    let expanded = QpProblem {
        n: prob.n,
        p_lower: prob.p_lower.clone(),
        c: prob.c.clone(),
        a: prob.a.clone(),
        b: prob.b.clone(),
        g,
        h,
        lb: Vec::new(),
        ub: Vec::new(),
    };
    (expanded, bound_rows)
}

/// A warm-start iterate: a previous primal/dual solution to seed the
/// interior-point iteration for a *nearby* problem (same structure, mildly
/// perturbed `c`/`b`/`h`/bounds). Its fields mirror [`QpSolution`], so the
/// idiomatic use is to feed back the prior solve's solution.
///
/// ## Why warm starting an IPM needs care
///
/// Unlike active-set/simplex methods, a primal-dual interior-point method
/// converges *to* the complementarity boundary (`s∘z → 0`). A converged
/// warm point therefore lies essentially **on** that boundary — the worst
/// place to restart, since the IPM needs a well-centered interior iterate.
/// Seeding `(x, s, z)` verbatim typically stalls.
///
/// [`solve_qp_ipm_warm`] handles this with a Mehrotra-style recentering
/// ([`init_iterate`]): it keeps the warm primal `x` (whose slack pattern
/// `h − Gx` encodes the active set) but pushes the slacks `s` and
/// multipliers `z` back into the interior with a **scale-aware floor**, so
/// the start is genuinely interior and centered while still benefiting
/// from the warm `x`. The benefit is real but bounded — it is largest when
/// the active set is stable across the perturbation, and modest or absent
/// when it changes substantially (a known property of IPM warm starts).
#[derive(Debug, Clone)]
pub struct QpWarmStart {
    /// Primal iterate (length `n`).
    pub x: Vec<f64>,
    /// Equality multipliers (length `m_eq`).
    pub y: Vec<f64>,
    /// Inequality multipliers for the original `G` rows (length `m_ineq`).
    pub z: Vec<f64>,
    /// Lower-bound multipliers (length `n`).
    pub z_lb: Vec<f64>,
    /// Upper-bound multipliers (length `n`).
    pub z_ub: Vec<f64>,
}

impl QpWarmStart {
    /// Build a warm start from a previous [`QpSolution`].
    pub fn from_solution(sol: &QpSolution) -> Self {
        QpWarmStart {
            x: sol.x.clone(),
            y: sol.y.clone(),
            z: sol.z.clone(),
            z_lb: sol.z_lb.clone(),
            z_ub: sol.z_ub.clone(),
        }
    }
}

/// Internal warm start expressed in the *expanded* space (variable bounds
/// already folded into the inequality block, so `z` covers `G`-rows then
/// the appended bound rows).
struct WarmStart {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

/// Build the expanded-space `z` for a warm start: the original `G`-row
/// multipliers followed by each appended bound row's `z_lb`/`z_ub` value,
/// in the same append order as [`expand_bounds`]. Inverse of
/// [`split_bound_duals`]'s `z` handling.
fn merge_bound_duals(
    prob: &QpProblem,
    bound_rows: &[(usize, usize, bool)],
    warm: &QpWarmStart,
) -> Vec<f64> {
    let base_m = prob.m_ineq();
    let mut z = vec![0.0; base_m + bound_rows.len()];
    let copy = base_m.min(warm.z.len());
    z[..copy].copy_from_slice(&warm.z[..copy]);
    for &(r, var, is_upper) in bound_rows {
        let v = if is_upper {
            warm.z_ub.get(var).copied().unwrap_or(0.0)
        } else {
            warm.z_lb.get(var).copied().unwrap_or(0.0)
        };
        if r < z.len() {
            z[r] = v;
        }
    }
    z
}

/// Move the appended bound rows' multipliers from the expanded solution's
/// `z` into `z_lb`/`z_ub`, and trim `z` back to the original rows.
fn split_bound_duals(
    prob: &QpProblem,
    bound_rows: &[(usize, usize, bool)],
    mut sol: QpSolution,
) -> QpSolution {
    let base_m = prob.m_ineq();
    let mut z = vec![0.0; base_m];
    z.copy_from_slice(&sol.z[..base_m]);
    let mut z_lb = vec![0.0; prob.n];
    let mut z_ub = vec![0.0; prob.n];
    for &(r, var, is_upper) in bound_rows {
        if is_upper {
            z_ub[var] = sol.z[r];
        } else {
            z_lb[var] = sol.z[r];
        }
    }
    sol.z = z;
    sol.z_lb = z_lb;
    sol.z_ub = z_ub;
    sol
}

/// The cost-normalized embedding solve, with the recovered duals and objective
/// mapped back out of the `σ` metric — the `σ ≠ 1` branch of [`solve_qp_core`]
/// with its verdict check removed.
///
/// Split out so the check can be tested against the point it actually judges.
/// The `σ` guard is now strict enough that no public entry point returns a
/// `σ`-manufactured false optimum (gh #414 reopened), which is the fix — but it
/// also means the guarantee tests behind that guard can no longer construct
/// their subject through a public door. They call this instead, so they keep
/// measuring a real escaped point rather than a hand-written one.
fn cost_normalized_hsde_solve<F>(
    scaled: &QpProblem,
    cone: &CompositeCone,
    inner: &QpOptions,
    sigma: f64,
    make_backend: &mut F,
    hook: Option<&mut dyn DebugHook>,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let mut sol = crate::hsde::solve_conic_hsde(scaled, cone, inner, make_backend, hook);
    for v in sol.y.iter_mut().chain(sol.z.iter_mut()) {
        *v *= sigma;
    }
    sol.obj *= sigma;
    sol
}

/// Objective-normalization factor `σ ≥ 1` for the HSDE driver (see the call
/// site in [`solve_qp_core`]). Returns the magnitude of the objective data
/// `max(‖P‖∞, ‖c‖∞)`, rounded **up to a power of two** so that dividing the
/// data by `σ` and multiplying the recovered duals/objective back by `σ` is
/// exact in floating point — but only once that magnitude is large enough to
/// genuinely destabilize the embedding's `τ`. The threshold is the same
/// crossover the scale-relative stop uses (`σ·ε > tol`): below it, `tol`-level
/// *absolute* KKT accuracy is still reachable and the embedding is healthy, so
/// the wrapper returns `1.0` and the solve is byte-for-byte the historical one.
///
/// Crucially this keys on the objective **coefficient** magnitude, not the
/// objective *value* at the solution: the large-data QP cluster
/// (POWELL20/BOYD/QSHELL) owes its large objective to a large `‖x*‖` with
/// modest `(P, c)` coefficients, so `σ = 1` there and its finely-tuned
/// `τ`/`κ` iterates are untouched. Only data whose coefficients themselves are
/// astronomically large (gh #286: `‖P‖ ~ 1e21`) is rescaled.
fn hsde_cost_scale(prob: &QpProblem, tol: f64) -> f64 {
    let mag = prob
        .p_lower
        .iter()
        .map(|t| t.val.abs())
        .chain(prob.c.iter().map(|v| v.abs()))
        .fold(0.0_f64, f64::max);
    // Only normalize once the coefficient magnitude is large enough that a
    // `tol`-level absolute residual is below the finite-precision floor
    // (`mag·ε > tol`) — the regime where the embedding's `τ` collapses. Below
    // it the historical (un-normalized) solve is preserved exactly.
    if !(mag.is_finite() && mag * f64::EPSILON > tol) {
        return 1.0;
    }
    // Round up to a power of two: the scale/unscale round-trip is then exact.
    let e = mag.log2().ceil();
    let sigma = 2.0_f64.powf(e);
    if sigma.is_finite() && sigma >= 1.0 {
        sigma
    } else {
        1.0
    }
}

/// Whether a solution the cost-normalized HSDE path certified `Optimal` is a
/// *genuine* optimum of `prob` (the original, un-normalized problem), rather
/// than a false certificate manufactured by the objective scaling (gh #324, and
/// gh #414 reopened).
///
/// # What `σ` actually promises, and what it delivers
///
/// The embedding solves `(P/σ, c/σ)` and applies its **absolute** stopping test
/// there, so what it certifies is `‖r‖ ≤ tol` on the scaled data — `‖r‖ ≤ σ·tol`
/// in the caller's coordinates. That is a *relative* test wearing an absolute
/// one's clothes, and two things about it are wrong unless this function
/// corrects them:
///
/// 1. **It never passes the gate.** The driver's own relative arm is admissible
///    only once absolute `tol` accuracy is below the finite-precision floor
///    ([`crate::hsde::relative_stop_permitted`]). The `σ` route reaches the same
///    relaxation without ever asking that question, because inside the scaled
///    metric the data looks `O(1)` and the loop believes it is running the
///    strict test.
/// 2. **It is relative to the wrong thing.** `σ` is sized by the objective
///    **coefficient** magnitude `max(‖P‖∞, ‖c‖∞)`
///    ([`hsde_cost_scale`]), while a stationarity residual has to be small
///    against the **gradient** scale `‖Px*‖∞ ∨ ‖c‖∞`. The two differ by `‖x*‖`,
///    unboundedly: on `min (x₀−1)² + (10⁴x₁−1)²`, `σ = 2²⁸ ≈ 2.7e8` and the
///    gradient scale is `2e4`, so the embedding stopped at `‖Px+c‖∞ = 2.499` and
///    called it `Optimal` — `x` wrong by `2.5e-4` relative, on a problem
///    clarabel solves to `1.4e-16`. That is gh #414 reopened, and no amount of
///    conditioning explains it: the un-normalized embedding solves the same
///    instance in **one** iteration.
///
/// So the test is asked in two arms, in this order:
///
/// - **Absolute.** A point accurate to `tol` in the caller's own coordinates is
///   optimal by the definition the caller was given, whatever `σ` was. Every
///   well- and moderately-scaled solve leaves here for the price of one
///   residual evaluation.
/// - **Relative**, and only where [`crate::hsde::relative_stop_permitted`] says
///   a relative test is admissible **at the gradient scale that governs this
///   point** — the correction to (1) and (2) together — against
///   [`sigma_path_rel_tol`], which is the correction to a flat cut that did not
///   track `tol` (see there for the measured populations).
///
/// A `false` costs one un-normalized re-solve, which then faces this same test;
/// it never costs an answer. That is why this door can be strict where
/// [`normalized_optimum_is_genuine_relative`]'s cannot.
fn normalized_optimum_is_genuine(
    prob: &QpProblem,
    cone: &CompositeCone,
    sol: &QpSolution,
    tol: f64,
) -> bool {
    // Absolute arm, asked first and unconditionally: a point already accurate
    // to `tol` in the caller's own coordinates is optimal by the definition the
    // caller was given, whatever `σ` was. Every well- and moderately-scaled
    // solve leaves here, paying one residual evaluation.
    if sol.kkt_residuals(prob).kkt_error() <= tol {
        return true;
    }
    // Relative arm, admissible only where the embedding's own relative stop
    // would be — and asked at the scale that actually governs the caller's
    // residual. This gate is the gh #414-reopened fix; see the doc comment.
    let (gscale, pscale) = unscaled_residual_scales(prob, sol);
    let cut = sigma_path_rel_tol(tol);
    let (rstat, dscale) = stationarity_rows(prob, sol);
    crate::hsde::relative_stop_permitted(gscale.max(pscale), tol)
        && unscaled_relative_kkt(prob, sol) <= cut
        // ... and the same question asked one row at a time, because
        // `gscale` and `pscale` are aggregates and an aggregate cannot see a
        // flat direction (gh #846, gh #875). Strictly two further conjuncts:
        // they can only turn an accept into a reject, and a reject costs one
        // un-normalized re-solve.
        //
        // Both halves of the KKT system are asked, because an aggregate is
        // blind in both. Stationarity comes first: it is the only one of the
        // two that has rows to test on a problem with no inequality rows and
        // no bounds, which is the shape that left gh #875 open.
        && sigma_stationarity_is_genuine(&rstat, &dscale, tol, cut)
        && sigma_complementarity_is_genuine(prob, cone, sol, tol, cut, &dscale)
        // ... and then the same question asked as a *distance*, because both
        // of the above are per-row ratios and a per-row ratio cannot see a
        // coupled spectrum (gh #880). Also strictly a further conjunct.
        && sigma_forward_error_is_small(prob, sol, cut)
}

/// Row `i` of the stationarity equation, decomposed into the residual
/// `rᵢ = (Px + c + Aᵀy + Gᵀz − z_lb + z_ub)ᵢ` and `dᵢ`, the largest single
/// term that built it — the scale anything landing in that row has to be
/// significant against.
///
/// Both σ-path componentwise guards ask about these same six terms —
/// [`sigma_stationarity_is_genuine`] measures the row's own residual against
/// `dᵢ`, [`sigma_complementarity_is_genuine`] measures a multiplier's
/// contribution to the row against it — so the decomposition lives in one
/// place. Two copies that drifted apart would disagree about the same row.
fn stationarity_rows(prob: &QpProblem, sol: &QpSolution) -> (Vec<f64>, Vec<f64>) {
    let n = prob.n;
    let mut px = vec![0.0; n];
    prob.p_mul(&sol.x, &mut px);
    let mut aty = vec![0.0; n];
    prob.at_mul(&sol.y, &mut aty);
    let mut gtz = vec![0.0; n];
    prob.gt_mul(&sol.z, &mut gtz);
    (0..n)
        .map(|i| {
            let terms = [px[i], prob.c[i], -sol.z_lb[i], sol.z_ub[i], aty[i], gtz[i]];
            let r = terms.iter().sum::<f64>();
            let d = terms.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            (r, d)
        })
        .unzip()
}

/// The `σ` path's genuineness test asked of the **stationarity** rows, one row
/// at a time (gh #875) — the sibling of [`sigma_complementarity_is_genuine`],
/// and the arm that guard's doc comment used to say was unnecessary.
///
/// # Why it is not optional
///
/// [`sigma_complementarity_is_genuine`] loops over `prob.m_ineq()` and over
/// the finite entries of `lb`/`ub`. On a problem with **neither** — an
/// unconstrained QP — it returns `true` having executed no test at all, and
/// [`normalized_optimum_is_genuine`] then rests on the aggregate
/// `unscaled_relative_kkt <= cut` arm alone, which is the exact aggregate the
/// componentwise guards exist to stop trusting.
///
/// The reported instance is two variables and nothing else:
/// `min ½(x₀−3)² + ½·10¹²(x₁−½)²`, so `x* = (3, ½)` by identity. The `σ` path
/// returned `x₀ = 0.027` — wrong by `2.97` on a coordinate whose optimum is
/// `3` — in one iteration, and printed `Dual infeasibility 5.50e+01` on the
/// line above `EXIT: Optimal Solution Found.`; `qp_hsde=no` and Ipopt both
/// return `(3, ½)` exactly. Across the family the error is a clean function of
/// condition number and independent of coefficient magnitude — `3e-4` at
/// `cond = 1e6`, `1.5` at `1e10`, `3.0` (i.e. `x₀ ≈ 0`, 100% wrong) at `1e12`
/// — while the *relative objective* error stays at `1e-11 ‥ 1e-16` throughout,
/// which is the objective-parity blind spot CLAUDE.md names.
///
/// The aggregate cannot see it, and the printed norm does not even point at
/// the guilty row. At that returned point row 1 carries `|r₁| = 55.0` against
/// its own scale `5e11`, a relative `1.1e-10` and genuinely converged; row 0
/// carries `|r₀| = 2.97` against its own scale `3`, a relative `0.99` and
/// completely unconverged. The reported `‖r‖∞` is `55.0` — row **1**'s — and
/// over `gscale = 5e11` it reads `1.1e-10` and sails through. The row that is
/// wrong is invisible in the aggregate twice over: once because its residual
/// is not the largest, and once because its scale is not the largest.
///
/// # The test
///
/// Row `i`'s residual against the largest single term that built it
/// ([`stationarity_rows`]), after the same absolute arm the complementarity
/// guard asks first: a residual already `tol`-small in the caller's own
/// coordinates is stationary by the definition the caller was given. A
/// non-finite residual compares false at both gates and so rejects, rather
/// than being waved past as "small" — the gh #845 shape.
///
/// # What the old doc comment claimed
///
/// That the same spectrum "unconstrained, in a wide box it never reaches, and
/// under an equality row all come back exact to `3e-16`", so that "the failure
/// needs an **active bound**". Re-measured on that same `EIG`/`TGT` pair, two
/// of the three shapes it named are wrong: unconstrained returns
/// `|x−x*|∞ = 2.03e-02` via the `σ` path in one iteration, and only the
/// wide-box shape comes back at `2.22e-16` — and it does so by taking the
/// *direct* path, not the `σ` one. The claim that the arm below rejects
/// nothing this one does not was true of the fixtures it was measured on, and
/// those fixtures all had an active bound.
fn sigma_stationarity_is_genuine(rstat: &[f64], dscale: &[f64], tol: f64, cut: f64) -> bool {
    rstat
        .iter()
        .zip(dscale)
        .all(|(&r, &d)| r.abs() <= tol || r.abs() <= cut * d)
}

/// `a + b` and the exact rounding error it dropped, for `|a| ≥ |b|` or not.
///
/// Knuth's `two_sum`. The pair `(s, e)` satisfies `s + e == a + b` exactly in
/// real arithmetic, so carrying `e` alongside `s` costs six flops and buys the
/// error term back.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `−(Px + c)` with the rounding error of the sum carried, not dropped.
///
/// Each product `P_ij·x_j` is split exactly into `p + e` by one FMA
/// (`f64::mul_add` is fused by contract, so `mul_add(a, b, -p)` is the exact
/// product error), and each accumulation into the running total is split by
/// [`two_sum`]. The dropped parts are summed separately and folded back at the
/// end, which is a two-term (double-double) accumulator: enough to make the
/// computed residual the exact residual to working precision.
///
/// Why it matters here rather than being a general nicety:
/// [`sigma_forward_error_is_small`] divides this residual by the Newton
/// operator, so any error in it is amplified by `‖M⁻¹‖`. In the ordinary sum
/// that error is `ε‖P‖‖x‖` and the amplification turns it into `ε·cond·‖x‖`,
/// which *is* the quantity being tested against. The guard was then reading
/// its own arithmetic noise. See gh #880 and
/// `tests/issue880_coupled_sigma_forward_error.rs`.
///
/// Mirrors [`QpProblem::p_mul_add`]'s traversal exactly, including the
/// off-diagonal mirroring of the stored lower triangle, so the two cannot
/// disagree about what `P` is.
fn residual_compensated(prob: &QpProblem, x: &[f64]) -> Vec<f64> {
    let n = prob.n;
    // `hi` carries the running sum, `lo` the accumulated dropped bits.
    let mut hi = prob.c.clone();
    hi.resize(n, 0.0);
    let mut lo = vec![0.0_f64; n];
    let accumulate = |i: usize, a: f64, b: f64, hi: &mut [f64], lo: &mut [f64]| {
        let prod = a * b;
        // Exact product error; zero when the product is exact.
        let perr = a.mul_add(b, -prod);
        let (s, serr) = two_sum(hi[i], prod);
        hi[i] = s;
        lo[i] += serr + perr;
    };
    for t in &prob.p_lower {
        accumulate(t.row, t.val, x[t.col], &mut hi, &mut lo);
        if t.row != t.col {
            accumulate(t.col, t.val, x[t.row], &mut hi, &mut lo);
        }
    }
    hi.iter().zip(&lo).map(|(h, l)| -(h + l)).collect()
}

/// The `σ` path's genuineness test asked as a **forward error bound** — how far
/// the returned `x` can be from the true minimizer — instead of as a ratio
/// between a residual and a term of its own row (gh #880).
///
/// # Why the two componentwise arms are not enough
///
/// [`sigma_stationarity_is_genuine`] and [`sigma_complementarity_is_genuine`]
/// both hold row `i`'s residual against `dᵢ`, the largest single term that
/// built that row ([`stationarity_rows`]). That is a **directional** scale, not
/// a **reduced** one — the same distinction CLAUDE.md draws for the sensitivity
/// classifier (`reduced/diagonal`, gh #763) and for its constraint rows
/// (`reduced/directional`, gh #804), one crate over.
///
/// On a **diagonal** `P` the distinction does not bite: row `i`'s stationarity
/// residual is `eᵢ(xᵢ − x*ᵢ)` and `dᵢ` is of order `eᵢ|x*ᵢ|`, so the ratio *is*
/// the relative error in `xᵢ` and the test is exact. That is the entire
/// separable half of gh #875, and it is why that fix measured as total there.
/// Rotate the same spectrum and the stiff mode appears in **every** row: every
/// denominator becomes the stiff mode's contribution, all `n` of them collapse
/// back to one number, and the componentwise refinement buys nothing over the
/// aggregate it was introduced to replace. The guard is present, is evaluated,
/// and rejects nothing.
///
/// Measured on gh #880's 72-instance census (`cond 1e2 ‥ 1e12` × `mag
/// 1e-3 ‥ 1e3` × `n ∈ {2,5}` × rotated or not, with `P = Q diag(e) Qᵀ` and
/// `c = −P t`, so `x* = t` by construction): after gh #875 the separable half
/// is 0/36 wrong and the coupled half is **17/36 wrong, every one bit-identical
/// to the pre-#875 baseline** — same `x`, same iteration count, same reported
/// dual infeasibility. Over those 17 the componentwise stationarity ratio tops
/// out at `6.9e-9`, three orders *below* the `cut` it is compared against.
///
/// # The test
///
/// `x − x*` is not a per-row quantity, so no per-row denominator can bound it.
/// Solve for it instead. `Δ` is the **affine-scaling** Newton step — the step
/// the iteration itself would take toward `μ = 0` from the returned point:
///
/// ```text
///     (P + Gᵀ Σ_row G + Σ_bnd) Δ = −(Px + c)
/// ```
///
/// where `Σ` is the barrier diagonal implied by the returned multipliers and
/// their slacks (`zⱼ/sⱼ`), the same operator the interior-point iteration's
/// own Newton system carries, reducing to exactly `P` when nothing is active.
/// `‖Δ‖∞` over `max(1, ‖x‖∞)` is then a *relative distance to the optimum*,
/// which is basis-free by construction: it is a norm of a vector, not a ratio
/// of two coordinates, so rotating the problem rotates `Δ` and leaves `‖Δ‖`
/// alone. It is compared against the same `cut` the other arms use.
///
/// ## The right-hand side is `−(Px + c)`, not the stationarity residual
///
/// Eliminating `Δz` and `Δs` from the Newton system leaves
/// `(P + GᵀΣG)Δ = −r_d + Gᵀz − GᵀΣ r_p`, and `r_d = Px + c + Gᵀz`, so the
/// multiplier term cancels exactly: the right-hand side is `−(Px + c)`, with
/// `r_p ≡ 0` because the slack is recovered as `h − Gx` rather than read back.
/// The cancellation is not a simplification, it is the point. **The returned
/// multipliers are not trusted to be complementary, only to say how stiff each
/// row is.** Using `r_d` instead — the obvious choice, and the one this arm
/// shipped with first — makes the estimate agree with a point that is wrong in
/// `x` but self-consistent in its own multipliers, which is precisely the
/// shape the `σ` cascade's second candidate takes: measured on gh #880's
/// box-constrained coupled instance the un-normalized re-solve returns
/// `x = (2.064, −0.748)`, displaced 1.56 along the soft eigenvector, with
/// `O(1)` bound multipliers on bounds whose slack is `~10`. Those multipliers
/// absorb the whole gradient, `‖r_d‖ = 2e-6`, and an `r_d`-based estimate
/// reads `9.9e-8` and accepts an answer that is 31% wrong. The affine-scaling
/// form reads `1.1` and rejects.
///
/// The corollary is that an *active* bound must read as **stiff**: there
/// `−(Px + c)` is legitimately large and only `Σ → ∞` holds `Δ` down. See
/// [`barrier_ratio`], and `an_active_bound_is_stiff_not_free` in
/// `tests/issue880_coupled_sigma_forward_error.rs`.
///
/// On the census the estimate tracks the true error to a few percent across
/// nine orders of magnitude, and the two populations separate by **22×**:
/// the largest correct instance reads `6.4e-8` and the smallest wrong one
/// `1.4e-6`, with `cut = 1e-6` at the default `tol` sitting between them. The
/// threshold is therefore not tuned — `cut` is what the other two arms already
/// use, and the quantity it now gates is the one the census calls "wrong".
///
/// # Solved matrix-free, and why an under-solve is the safe direction
///
/// Conjugate gradients on `p_mul`/`g_mul` — the operator is positive definite
/// wherever `P` is PD, and PSD otherwise — capped at
/// [`FORWARD_ERROR_CG_ITERS`], with no factorization and no dimension ceiling
/// (the same shape as gh #871's null-space search). CG starts at `Δ = 0` and
/// builds `‖Δ_k‖` upward, so a cap that stops early, a `pᵀMp ≤ 0` breakout on a
/// singular `P`, and an LP's `P = 0` all **under**-estimate the error. An
/// under-estimate accepts, which is the status quo: this conjunct can only ever
/// turn an accept into a reject, never the reverse. What that costs is stated
/// in [`normalized_optimum_is_genuine`] — one un-normalized re-solve — and what
/// it buys is bounded by the same fact: on a large, viciously conditioned
/// problem CG will not have converged inside the cap and the guard quietly
/// stops working rather than misfiring.
///
/// # What it does not cover
///
/// **Equality rows.** `A` restricts motion exactly rather than through a
/// diagonal, so bounding `Δ` under equalities is the saddle system and not this
/// operator; using this operator anyway would over-estimate the error and
/// reject points that are fine. The guard declines on `m_eq() > 0` and the
/// other two arms stand there exactly as they did.
///
/// **`cond ≥ 1e12`.** Rejecting is only useful if there is somewhere correct to
/// route to, and at `cond = 1e12` the un-normalized path this rejects *into* is
/// itself wrong (gh #880 sub-problem 2, still open). Six of the census's 17
/// stay wrong after this fix, with a different wrong answer. That is not a
/// regression and not a fix; it is the guard doing its job into a destination
/// that cannot yet serve it.
fn sigma_forward_error_is_small(prob: &QpProblem, sol: &QpSolution, cut: f64) -> bool {
    // Equalities need the saddle system, not a diagonal; see the doc comment.
    if prob.m_eq() > 0 {
        return true;
    }
    let n = prob.n;
    if n == 0 {
        return true;
    }
    // `−(Px + c)`: the affine-scaling right-hand side, which is the
    // stationarity residual with the inequality and bound multipliers taken
    // back out. See the doc comment — trusting them is what let a
    // non-complementary point read as converged.
    //
    // Accumulated in COMPENSATED arithmetic, and that is what makes the
    // verdict mean anything at high conditioning (gh #880). Formed the
    // ordinary way this sum carries an absolute error of order
    // `ε‖P‖‖x‖`, so `Δ = M⁻¹r` carries `‖P⁻¹‖ε‖P‖‖x‖ = ε·cond·‖x‖` — the
    // estimator cannot resolve a relative error below `ε·cond` and cannot
    // reject what it cannot see. That floor, not the cut, is what left nine
    // census instances returning a wrong `x` under `Optimal` at
    // `cond ≥ 1e10`, eight of them wrong by *less* than their own instance's
    // floor.
    //
    // `residual_compensated` removes the term: each product is split exactly
    // by FMA and each partial sum by `two_sum`, so the computed residual is
    // the exact one to working precision and `Δ` is limited by the solve
    // rather than by the right-hand side. Measured against a
    // `fractions.Fraction` reference on the census's own rotated spectra, it
    // agrees to every digit printed at `cond = 1e10` and `1e12` and at
    // injected errors from `5×` down to `0.02×` of the old floor, where the
    // plain sum reads between `0.29×` and `5.4×` of the truth — in both
    // directions, so the old estimator both accepted wrong answers and would
    // have rejected right ones.
    //
    // The gh #880 census does *not* discriminate this: swapped back to the
    // plain sum it demotes the same nine instances with the same errors,
    // because its forward errors sit orders away from the cutoff. The
    // estimator only decides anything within a few multiples of `tol`, which
    // that corpus does not populate. `compensated_residual_tests` is what
    // holds the claim instead, on rows whose exact value is known by
    // construction: `the_compensated_residual_survives_a_cancelling_row`
    // fails, reading `0.0` where the truth is `1.0`, if the plain sum is
    // restored.
    let rhs = residual_compensated(prob, &sol.x);
    if rhs.iter().any(|v| !v.is_finite()) {
        // Not "small", but it is the gh #845 shape and the stationarity arm
        // already rejects on it. Nothing to add here.
        return true;
    }
    // A `None` anywhere is "no usable stiffness", which declines; see
    // [`barrier_ratio`] for why that is the accepting direction here.
    let (Some(sigma_bnd), Some(sigma_row)) = (
        barrier_diagonal_bounds(prob, sol),
        barrier_diagonal_rows(prob, sol),
    ) else {
        return true;
    };
    let delta = match forward_error_step(prob, &rhs, &sigma_bnd, &sigma_row) {
        Some(d) => d,
        None => return true,
    };
    let dnorm = delta.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let xnorm = sol.x.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    dnorm.is_finite() && dnorm <= cut * xnorm
}

/// Iterations of conjugate gradients [`sigma_forward_error_is_small`] spends
/// estimating the forward error. Sized so the guard is free on the small,
/// badly-scaled models the `σ` path actually sees (the census tops out at
/// `n = 5`, and CG is exact in `n` steps there) while staying bounded on a
/// large one. Stopping early under-estimates, which accepts; see the doc
/// comment on the caller.
const FORWARD_ERROR_CG_ITERS: usize = 64;

/// `zⱼ/sⱼ` for one block — the barrier stiffness the returned multiplier and
/// its slack imply — or `None` where they imply nothing usable.
///
/// **`None` is not zero, and the difference is the whole of the safe
/// direction.** [`sigma_forward_error_is_small`]'s right-hand side is
/// `−(Px + c)`, which at an *active* bound is large and is held down only by
/// that bound's `Σ`. So a `Σ` that reads too **small** inflates `‖Δ‖` and
/// rejects a correct answer, which is the one direction this guard may not err
/// in; a `Σ` that reads too large only shrinks `‖Δ‖`, accepts, and leaves the
/// status quo. Substituting `0` for an unusable ratio — the obvious
/// defensive-looking choice, and what the first draft of this arm did — is
/// therefore exactly backwards: it declares an active bound *free*. The arm
/// declines on `None` instead, which accepts.
///
/// Unusable means a non-positive or non-finite slack (an iterate at or past
/// its bound, where the diagonal model does not apply at all) or a ratio that
/// is not a finite non-negative number. A missing bound is not unusable: it
/// contributes `0` because it genuinely contributes nothing.
fn barrier_ratio(z: f64, slack: f64) -> Option<f64> {
    if !(slack > 0.0) || !slack.is_finite() {
        return None;
    }
    let s = z / slack;
    (s.is_finite() && s >= 0.0).then_some(s)
}

/// `Σ_bnd`, the barrier diagonal the returned bound multipliers imply:
/// `z_lb ᵢ/(xᵢ − lbᵢ) + z_ub ᵢ/(ubᵢ − xᵢ)`, with an absent bound contributing
/// nothing and any *present* bound whose ratio is unusable aborting the whole
/// estimate — see [`barrier_ratio`].
fn barrier_diagonal_bounds(prob: &QpProblem, sol: &QpSolution) -> Option<Vec<f64>> {
    (0..prob.n)
        .map(|i| {
            let mut s = 0.0;
            let lb = prob.lb_of(i);
            if lb > -crate::qp::BOUND_INF {
                s += barrier_ratio(sol.z_lb[i], sol.x[i] - lb)?;
            }
            let ub = prob.ub_of(i);
            if ub < crate::qp::BOUND_INF {
                s += barrier_ratio(sol.z_ub[i], ub - sol.x[i])?;
            }
            s.is_finite().then_some(s)
        })
        .collect()
}

/// `Σ_row`, the same quantity for the inequality rows: `zⱼ/sⱼ` with the slack
/// recovered as `sⱼ = hⱼ − (Gx)ⱼ` rather than read back, so the primal
/// residual of the eliminated Newton system is zero by construction.
fn barrier_diagonal_rows(prob: &QpProblem, sol: &QpSolution) -> Option<Vec<f64>> {
    let m = prob.m_ineq();
    if m == 0 {
        return Some(Vec::new());
    }
    let mut gx = vec![0.0; m];
    prob.g_mul(&sol.x, &mut gx);
    (0..m)
        .map(|j| barrier_ratio(sol.z[j], prob.h[j] - gx[j]))
        .collect()
}

/// Conjugate gradients on `(P + Gᵀ Σ_row G + Σ_bnd) Δ = r`, from `Δ = 0`.
///
/// Returns `None` when the operator is not usable as a positive-definite one
/// (a non-positive curvature sample, a non-finite intermediate) — the caller
/// reads that as "no evidence", which accepts. Every early exit is on the
/// under-estimating side; see [`sigma_forward_error_is_small`].
fn forward_error_step(
    prob: &QpProblem,
    r: &[f64],
    sigma_bnd: &[f64],
    sigma_row: &[f64],
) -> Option<Vec<f64>> {
    let n = prob.n;
    let m = prob.m_ineq();
    let mut scratch_row = vec![0.0; m];
    let mut scratch_col = vec![0.0; n];
    // `y ← M v`, with `M = P + Gᵀ Σ_row G + diag(Σ_bnd)`.
    let apply =
        |v: &[f64], y: &mut [f64], scratch_row: &mut Vec<f64>, scratch_col: &mut Vec<f64>| {
            // `p_mul` is `y += P v`, not `y = P v`, and `y` is reused across CG
            // iterations — zero it first or the operator accumulates.
            y.iter_mut().for_each(|s| *s = 0.0);
            prob.p_mul(v, y);
            for (yi, (vi, si)) in y.iter_mut().zip(v.iter().zip(sigma_bnd)) {
                *yi += si * vi;
            }
            if m > 0 {
                scratch_row.iter_mut().for_each(|s| *s = 0.0);
                prob.g_mul(v, scratch_row);
                for (s, sig) in scratch_row.iter_mut().zip(sigma_row) {
                    *s *= sig;
                }
                scratch_col.iter_mut().for_each(|s| *s = 0.0);
                prob.gt_mul(scratch_row, scratch_col);
                for (yi, ci) in y.iter_mut().zip(scratch_col.iter()) {
                    *yi += ci;
                }
            }
        };

    let mut x = vec![0.0; n];
    let mut res = r.to_vec();
    let mut p = res.clone();
    let mut rs = res.iter().map(|v| v * v).sum::<f64>();
    if !rs.is_finite() {
        return None;
    }
    if rs == 0.0 {
        return Some(x);
    }
    let mut mp = vec![0.0; n];
    for _ in 0..FORWARD_ERROR_CG_ITERS.min(n.saturating_mul(4).max(1)) {
        apply(&p, &mut mp, &mut scratch_row, &mut scratch_col);
        let denom = p.iter().zip(&mp).map(|(a, b)| a * b).sum::<f64>();
        if !denom.is_finite() || denom <= 0.0 {
            // Singular or indefinite along `p`: stop with what we have, which
            // is an under-estimate. `x` is still a valid Krylov iterate.
            break;
        }
        let alpha = rs / denom;
        if !alpha.is_finite() {
            break;
        }
        for (xi, pi) in x.iter_mut().zip(&p) {
            *xi += alpha * pi;
        }
        for (ri, mpi) in res.iter_mut().zip(&mp) {
            *ri -= alpha * mpi;
        }
        let rs2 = res.iter().map(|v| v * v).sum::<f64>();
        if !rs2.is_finite() {
            break;
        }
        if rs2 <= 0.0 {
            break;
        }
        let beta = rs2 / rs;
        for (pi, ri) in p.iter_mut().zip(&res) {
            *pi = ri + beta * *pi;
        }
        rs = rs2;
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

/// The `σ` path's genuineness test, asked **one orthant row at a time**
/// instead of over an aggregate scale (gh #846).
///
/// # Why an aggregate scale is not enough
///
/// [`normalized_optimum_is_genuine`]'s relative arm divides each residual by
/// one number for the whole problem — `gscale = ‖Px‖∞ ∨ ‖c‖∞` for
/// stationarity and complementarity, `pscale` for feasibility. On an
/// ill-conditioned separable QP that number belongs to the *stiffest*
/// coordinate, and it is then the denominator for every other one. The flat
/// directions — the ones whose optimum a solver is most likely to get wrong,
/// because moving them barely changes the objective — are measured against a
/// scale that has nothing to do with them.
///
/// The reported instance is a 6-variable diagonal box QP with
/// `eig = [1e3 ‥ 1e11]` on `[-1, 1]`, separable, so `x* = clamp(t, -1, 1)`
/// with no solver in the loop. At the default `tol = 1e-8` the `σ` path
/// returned `x₀ = 0.837` against a true `1.0` — off by `0.17` on a unit box.
/// The objective could not see it either: `-1.17834000816e10` against
/// `-1.17834002580e10`, a **relative objective error of 1.5e-8 for a 17%
/// error in x₀**, which is the objective-parity blind spot CLAUDE.md names,
/// one level down from the fixture corpus.
///
/// # This is the bound half, and it is not the whole guard
///
/// On gh #846's family what buys the slack is the embedding's
/// objective-relative gap test (`gap / (1 + |obj|)`), and a gap is spent on
/// the bound multipliers — so an active bound is where *that* failure lives,
/// and this is the arm that catches it. Removing this test turns four tests
/// red.
///
/// It is **not** sufficient on its own, and the version of this comment that
/// said so was wrong (gh #875). The loops below run over `prob.m_ineq()` and
/// over the finite entries of `lb`/`ub`; a problem with neither reaches the
/// end and returns `true` having tested nothing, which put an unconstrained
/// ill-conditioned QP back on the bare aggregate and returned `x₀ = 0.027`
/// for a true `3`. The stationarity arm this comment used to call redundant
/// is [`sigma_stationarity_is_genuine`], it runs as a conjunct ahead of this
/// one, and its doc comment carries the re-measurement.
///
/// # The test
///
/// Complementarity says one of the two factors is at zero, so it is asked as
/// exactly that, each factor against the scale it lives in:
///
/// - the slack is negligible — `|sⱼ| ≤ cut·max(|hⱼ|, |(Gx)ⱼ|)`, the largest
///   term that built it; **or**
/// - the multiplier is negligible — `|Gⱼᵢ·zⱼ| ≤ cut·dᵢ` at *every* variable
///   row `i` this row feeds, `dᵢ` being the largest term in that row's
///   stationarity equation. A multiplier that changes no stationarity row it
///   touches is a bound out of the active set, and its slack may then be
///   anything.
///
/// Neither factor needs a floor and neither is a product of unlike units,
/// which is what the aggregate `|zⱼsⱼ| / (gscale ∨ pscale)` was. On the
/// reported instance the un-normalized re-solve — the point a `σ` reject
/// routes to — comes back with `‖Px+c+Aᵀy+Gᵀz‖∞ = 7.6e-6`, genuinely small,
/// and `max |zⱼsⱼ| = 18.2`, which over `gscale = 4.0e10` reads `4.5e-10` and
/// sails through. Componentwise the two ratios are `2.9e-3` and `1.0`, so the
/// row is rejected — and `2.9e-3` is not an abstraction, it *is* the returned
/// `x`'s error, because `zs/z = s` is the distance from the bound.
///
/// **Nonnegative-orthant rows only, on purpose.** An orthant row is
/// complementary one row at a time; an SOC or PSD block is complementary as a
/// *block*, and reading its rows individually is pounce#209 — a feasible,
/// optimal QCQP made to look badly infeasible. Non-orthant blocks are skipped
/// and keep the aggregate test, which
/// [`QpSolution::kkt_residuals_conic`] already measures per block.
///
/// The variable-bound arrays are covered as well as `G`, because both shapes
/// reach here: [`solve_qp_ipm`] expands `lb`/`ub` into trailing orthant rows
/// before [`solve_qp_core`] ever sees them, but a caller that hands the core a
/// problem with bounds still in place gets the same test.
fn sigma_complementarity_is_genuine(
    prob: &QpProblem,
    cone: &CompositeCone,
    sol: &QpSolution,
    tol: f64,
    cut: f64,
    // Each variable row's stationarity denominator, the scale a multiplier
    // landing in it has to be significant against, from [`stationarity_rows`].
    dscale: &[f64],
) -> bool {
    let n = prob.n;
    // Which inequality rows are orthant rows, and what each one's `Gx` is.
    let mut orthant = vec![false; prob.m_ineq()];
    for (off, kind) in cone.blocks() {
        if let crate::cones::ConeKind::Nonneg(c) = kind {
            let dim = crate::cones::Cone::dim(c);
            for f in orthant.iter_mut().skip(*off).take(dim) {
                *f = true;
            }
        }
    }
    let mut gx = vec![0.0; prob.m_ineq()];
    prob.g_mul(&sol.x, &mut gx);
    // `max |Gⱼᵢ|`-weighted view of each row, built once: for row `j` the test
    // needs every `(i, Gⱼᵢ)` it touches.
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); prob.m_ineq()];
    for t in &prob.g {
        rows[t.row].push((t.col, t.val));
    }

    // The `max(1, ·)` floor is load-bearing, and its absence is a defect with
    // no lower bound on how wrong it gets. Without it the scale **collapses to
    // zero** exactly where the answer is most obviously right — a variable
    // resting on a bound of `0`, where `bound` and `x` are both `0` — and then
    // `|slack| <= cut·0` is false for every nonzero slack, however tiny. A
    // converged `1e-16` is rejected on that arithmetic. It is the same
    // convention `sigma_forward_error_is_small` uses for `‖x‖`, and for the
    // same reason: below a scale of 1 the only meaningful comparison is
    // absolute.
    //
    // gh #880 found this rather than fixing it: the cascade rejected an LP
    // whose relative KKT error was `9.5e-17` and whose `x` was the exact
    // vertex, and the `Optimal` it returned hid the rejection.
    //
    // **Measured inert on both corpora, not structurally inert** — the
    // distinction matters, and "it cannot produce a wrong answer" is the
    // argument CLAUDE.md names as the one that shipped gh #544. This guard
    // *is* the per-candidate verdict: accepting where it used to reject makes
    // `solve_qp_core` return the normalized solve rather than falling through
    // to the un-normalized re-solve, the direct driver and the `kkt_error`
    // pick, so a different candidate can come back and a demotion that would
    // have been recorded can be suppressed. It simply does not happen on
    // anything either corpus contains: identical census errors and iteration
    // counts, and no pre-existing fixture moves in the sweep.
    // `a_converged_slack_on_a_zero_bound_is_negligible` pins the guard
    // directly, which is the only evidence a corpus cannot give.
    let negligible = |v: f64, scale: f64| v.abs() <= cut * scale.max(1.0);
    for j in 0..prob.m_ineq() {
        if !orthant[j] {
            continue;
        }
        let (z, slack) = (sol.z[j], prob.h[j] - gx[j]);
        // Absolute, in the caller's own coordinates: a `tol`-small product is
        // complementary by the definition the caller was given. A non-finite
        // product compares false here and so falls through to the two ratio
        // tests rather than being waved past as "small" -- the gh #845 shape,
        // one crate over.
        let complementary_absolutely = (z * slack).abs() <= tol;
        if complementary_absolutely {
            continue;
        }
        if negligible(slack, prob.h[j].abs().max(gx[j].abs())) {
            continue;
        }
        if rows[j]
            .iter()
            .all(|&(i, gji)| negligible(gji * z, dscale[i]))
        {
            continue;
        }
        return false;
    }

    // The same two questions for bounds still carried on `prob` itself.
    for i in 0..n {
        let (lb, ub) = (prob.lb_of(i), prob.ub_of(i));
        for (z, slack, bound, finite) in [
            (sol.z_lb[i], sol.x[i] - lb, lb, lb > -1e19),
            (sol.z_ub[i], ub - sol.x[i], ub, ub < 1e19),
        ] {
            let complementary_absolutely = (z * slack).abs() <= tol;
            if !finite || complementary_absolutely {
                continue;
            }
            if negligible(slack, bound.abs().max(sol.x[i].abs())) || negligible(z, dscale[i]) {
                continue;
            }
            return false;
        }
    }
    true
}

/// The relative-KKT cut the `σ` path holds a claimed optimum to: `tol`-level
/// accuracy in the relative metric, with two orders of slack for the digits a
/// finite-precision solve cannot control — never looser than the flat
/// [`FALSE_OPTIMUM_REL_TOL`] a caller got before, so loosening `tol` cannot buy
/// more slack than the historical cut.
///
/// The flat cut alone is what left gh #414 open after the equilibrated repair.
/// It was calibrated against gh #324's *cold-start* certificate, which is
/// `O(1)`, so it separates that failure by three orders — and says nothing at
/// all about a point four orders inside it.
///
/// Measured on this crate's fixtures, as multiples of `tol` (which is what a
/// `tol`-tracking cut has to separate — the absolute numbers below are all at
/// the default `tol = 1e-8`):
///
/// | population | relative KKT | × `tol` |
/// |---|---|---|
/// | `issue414_cost_normalized_false_optimal`, the un-normalized re-solves this fix routes to | `5e-11 ‥ 1e-10` | `0.005 ‥ 0.01` |
/// | gh #324 cold-start family, after its re-solve | `6.9e-10 ‥ 2.0e-9` | `0.069 ‥ 0.20` |
/// | gh #286 huge-magnitude optima — genuine solves that only a relative arm can certify, so the population most at risk of a wrong reject | `6.9e-10 ‥ 3.2e-9` | `0.069 ‥ 0.32` |
/// | **worst genuine** | | **`0.32`** |
/// | **mildest false** | | **`625`** |
/// | `issue414_cost_normalized_false_optimal`, the `σ` points (`span` 3.7 ‥ 6.0) | `6.2e-6 ‥ 2.5e-3` | `625 ‥ 2.5e5` |
/// | `qcqp_columns_illcond`'s `σ` point (the one CLI fixture that reaches this path) | `9.6e-2` | `9.6e6` |
///
/// `100·tol` sits **313× above** the worst genuine solve and **6.25× below**
/// the mildest false one. The margin is deliberately lopsided toward the
/// genuine side: a wrong reject costs one un-normalized re-solve, a wrong
/// accept ships a wrong answer under `success=True`, and the genuine
/// population above is measured while the next one is not.
///
/// Two figures here correct gh #418's notes, which put the gh #286 optima at
/// `4e-10` and `1.5e-8` (`1.5·tol`) and the gh #414 false optima at
/// `2e-2 ‥ 1.2e2`. Re-measured on current `main` the gh #286 family reads
/// `6.9e-10 ‥ 3.2e-9`; the gh #414 original's `σ` point is not in the table at
/// all because it is rejected by the **absolute** arm, its relative residual
/// being inside even the flat cut — which is the fact
/// `the_false_optimum_is_invisible_unscaled_and_obvious_equilibrated` pins.
///
/// The tightening is affordable *here* and nowhere else in this file, for the
/// reason [`normalized_optimum_is_genuine_relative`] gives: a reject on this
/// path costs one un-normalized re-solve, which faces the same test again.
fn sigma_path_rel_tol(tol: f64) -> f64 {
    (100.0 * tol).min(FALSE_OPTIMUM_REL_TOL)
}

/// How much slack the `σ` demotion allows before calling a pick "far from
/// optimal" — see the table at the recording site in `solve_qp_core` for the
/// window this sits in, which is pinned from both sides by existing tests.
///
/// Module-scope rather than function-local because two sites ask the same
/// question of the same estimator: the cascade, about the pick it is
/// returning, and `solve_qp_ipm`, about the point `maybe_crossover` may have
/// replaced it with.
const SIGMA_DEMOTION_MARGIN: f64 = 10.0;

/// The `σ` demotion verdict, re-asked if [`crate::crossover::maybe_crossover`]
/// replaced the point it was recorded about.
///
/// gh#880 records the verdict inside `solve_qp_core`, about the candidate the
/// cascade returns. On a pure LP the crossover then purifies that interior
/// iterate to an **exact vertex** without re-entering the solver, so the
/// recorded bit describes a point that is no longer the answer.
/// `sigma_verdict`'s module doc names this and says a consumer sensitive to
/// which of the two points the bit describes "needs to re-record after
/// crossover rather than assume".
///
/// gh#888 created that consumer one merge earlier, and it is not a label: the
/// CLI reroutes `ProblemClass::Lp` on `OptimalInaccurate`, so a stale demotion
/// **discards the purified vertex** and re-solves on the NLP arm. The point
/// thrown away is the exact one.
///
/// Re-asking rather than clearing, deliberately. "Crossover fired, so the
/// demotion cannot apply" would be an argument from crossover's
/// never-regressing gate; this is a measurement of the point actually being
/// returned, through the same estimator with the same cut, so the two sites
/// cannot drift apart.
///
/// `before` is `None` when there was no verdict to re-record — the ordinary
/// path, which pays nothing, since the clone that produces it is taken only
/// when the cascade demoted something.
///
/// **Reachability, measured rather than assumed** (instrumented build, probes
/// on `hsde_cost_scale`, `sigma_verdict::record` and the crossover gate):
///
/// | | |
/// |---|---|
/// | crossover is on by default | **no** — `qp_crossover=yes` opts in |
/// | crossover *moves* `x` when enabled | **331 of 550** pure LPs |
/// | `σ` engages (`max(‖P‖∞,‖c‖∞)·ε > tol`) | 336 of those 550; 2 of 96 CLI fixtures |
/// | cascade records `uncertified` on a pure LP | **0 of 550** |
///
/// So the two halves have very different frequencies: the crossover half is
/// the *majority* case once the option is on, and the uncertified half is what
/// could not be produced. Note also that
/// [`sigma_forward_error_is_small`] returns early on `m_eq > 0`, so only a
/// model with **no equality rows** can be demoted at all — which is most of
/// why a netlib-shaped LP never is.
///
/// With the precondition forced, the effect is total and it is not cosmetic:
/// on the 182 LPs where crossover moves `x`, the stale verdict reroutes
/// **182 of 182** to the NLP arm, and scored against HiGHS on the 167 that
/// solve, keeping the crossover vertex is exact (median relative error `0`,
/// 167/167 within `1e-8`) while the reroute lands >10× further out on **43**
/// of them. The point being discarded is the exact one, which is the whole
/// argument for re-recording.
fn sigma_verdict_after_crossover(
    prob: &QpProblem,
    sol: &QpSolution,
    before: Option<&[f64]>,
    tol: f64,
    recorded: bool,
) -> bool {
    match before {
        // The status conjunct is part of the bit's definition at the
        // recording site — "this answer should be demoted", i.e. currently
        // labelled `Optimal` AND far from optimal — and that site's comment
        // says to widen the recording rather than the reader. Repeating it
        // here keeps the two writers of the same bit agreeing on what it
        // means. Unreachable today, since `demote_uncertified_sigma_optimum`
        // re-checks the status and crossover is never-regressing; it is
        // consistency, not behaviour (R5).
        Some(before) if before != sol.x.as_slice() => {
            sol.status == QpStatus::Optimal
                && !sigma_forward_error_is_small(
                    prob,
                    sol,
                    SIGMA_DEMOTION_MARGIN * sigma_path_rel_tol(tol),
                )
        }
        _ => recorded,
    }
}

/// Each un-normalized KKT residual of `sol` over the natural magnitude of its
/// own terms — the ratio both `σ`-path cuts are applied to.
fn unscaled_relative_kkt(prob: &QpProblem, sol: &QpSolution) -> f64 {
    let res = sol.kkt_residuals(prob);
    let (gscale, pscale) = unscaled_residual_scales(prob, sol);
    (res.dual_infeasibility / gscale)
        .max(res.primal_infeasibility / pscale)
        .max(res.complementarity / gscale.max(pscale))
}

/// The natural magnitudes the un-normalized KKT residuals of `sol` are measured
/// against: the objective-gradient scale `‖Px‖∞ ∨ ‖c‖∞` for stationarity and
/// the rhs scale `‖b‖∞ ∨ ‖h‖∞` for primal feasibility, each floored at 1 so a
/// zero-scale block cannot divide by zero.
///
/// Both are evaluated **at the returned point**, not on the data alone: that is
/// the whole difference between this and `σ`, and the reason a residual `σ` was
/// willing to license can still be enormous relative to the gradient it has to
/// be small against.
fn unscaled_residual_scales(prob: &QpProblem, sol: &QpSolution) -> (f64, f64) {
    let mut px = vec![0.0; prob.n];
    prob.p_mul(&sol.x, &mut px);
    let gscale = inf_norm(&px).max(inf_norm(&prob.c)).max(1.0);
    let pscale = inf_norm(&prob.b).max(inf_norm(&prob.h)).max(1.0);
    (gscale, pscale)
}

/// The *relative* half of [`normalized_optimum_is_genuine`], ungated: each
/// residual over the natural magnitude of its own terms, against
/// [`FALSE_OPTIMUM_REL_TOL`].
///
/// Kept separate because the two callers can afford different strictness, and
/// the difference is what their failure branch does:
///
/// - [`normalized_optimum_is_genuine`] (the `σ` path) answers a question whose
///   "no" costs one un-normalized re-solve, which then faces the same test
///   again. It can afford the gate: a wrong "no" loses an iteration, never an
///   answer.
/// - [`demote_false_equilibrated_optimum`] answers a question whose "no" is a
///   demotion to [`QpStatus::NumericalFailure`], with no repair behind it. A
///   wrong "no" there destroys a correct answer, so it must reject only on
///   positive evidence that the point is bad — and "the stopping rule was not
///   entitled to a relative test at this scale" is not that evidence. It stays
///   ungated: `equilibrated_trace_objective_is_in_original_coordinates` is a
///   well-scaled LP (`‖c‖ = 1e3`) whose direct-driver point is correct at a
///   relative `1e-10` and an absolute residual just past `tol`; gating this
///   caller demotes it.
fn normalized_optimum_is_genuine_relative(prob: &QpProblem, sol: &QpSolution) -> bool {
    unscaled_relative_kkt(prob, sol) <= FALSE_OPTIMUM_REL_TOL
}

/// Solve `prob` the way `qp_hsde=no` would — the direct driver, Ruiz-
/// equilibrated — as the `σ` path's last resort (gh #846).
///
/// **Not generic, on purpose.** [`solve_qp_core`] cannot simply call itself or
/// [`equilibrated_solve`] with `use_hsde: false`: those are generic in the
/// backend factory, so a self-call monomorphizes `F`, `&mut F`, `&mut &mut F`,
/// … without end, and the compiler says so (`overflow evaluating the
/// requirement`). Taking `&mut dyn FnMut()` erases the parameter and the
/// instantiation graph closes.
///
/// It enters at [`solve_qp_ipm_core`] rather than [`solve_qp_direct`] because
/// the equilibration is the point. Measured on gh #846's family across
/// `mag = 1e7 ‥ 1e14`, the raw direct driver returns `NumericalFailure` at
/// `1e12` and `IterationLimit` at `1e13`, while the same driver behind Ruiz
/// returns `‖x − x*‖∞ ≤ 1e-6` at every magnitude. `prob` arrives with its
/// bounds already expanded into orthant rows, so `solve_qp_ipm_core` re-enters
/// [`solve_qp_core`] with `use_hsde` off and this branch is not reached again.
fn direct_driver_fallback(
    prob: &QpProblem,
    opts: &QpOptions,
    make_backend: &mut dyn FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
) -> QpSolution {
    let direct = QpOptions {
        use_hsde: false,
        ..*opts
    };
    solve_qp_ipm_core(prob, &direct, make_backend, None)
}

/// Bounds-agnostic Mehrotra predictor-corrector core. `prob.lb`/`ub` are
/// ignored here; the public [`solve_qp_ipm`] handles bound expansion.
fn solve_qp_core<F>(
    prob: &QpProblem,
    cone: &CompositeCone,
    opts: &QpOptions,
    warm: Option<&WarmStart>,
    mut make_backend: F,
    hook: Option<&mut dyn DebugHook>,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if crate::deadline::expired() {
        return timed_out_solution(prob);
    }
    // gh #880: this attempt owns the `σ` verdict from here on. A retry that
    // re-enters must not inherit a verdict about the answer it is replacing.
    crate::sigma_verdict::clear();
    // Opt-in homogeneous self-dual embedding driver. It builds its own
    // factorization and self-starts, so it bypasses the warm-start /
    // factor-reuse plumbing below (warm is ignored — it cannot change the
    // solution, only the iteration count, which HSDE does not exploit yet).
    if opts.use_hsde {
        // Objective (cost) normalization for the embedding. HSDE deliberately
        // skips Ruiz row/column equilibration (its per-cone NT scaling
        // conditions the *constraint* system internally), but the NT scaling
        // does nothing about the sheer *magnitude* of the objective data
        // `(P, c)`. When those coefficients are enormous — e.g. a badly-scaled
        // QP with `‖P‖ ~ 1e21` (gh #286) — the homogeneous embedding's `τ`
        // collapses toward the `τ → 0` certificate boundary: the dual residual
        // scale swamps the `τ`-row, primal feasibility then crawls, and the
        // solve grinds to its iteration cap at a box-violating iterate even
        // though the dual/gap converged in a few dozen steps. Dividing the
        // objective by a scalar `σ ≥ 1` (argmin-invariant: the minimizer of
        // `½xᵀPx+cᵀx` and of `½xᵀ(P/σ)x+(c/σ)ᵀx` coincide) restores an O(1)
        // objective so `τ` stays healthy and the embedding converges in a
        // handful of iterations — the cost scaling Clarabel/OSQP apply as a
        // matter of course. The recovered dual multipliers and objective are
        // in the scaled metric and are mapped back below (`y,z ← σ·y,σ·z`,
        // `obj ← σ·obj`); the primal `x` needs no correction.
        //
        // Gated on `σ` being large enough to actually threaten the embedding
        // (see [`hsde_cost_scale`]) and rounded to a power of two, so ordinary
        // and moderately-scaled data — including the large-data QP cluster
        // whose magnitude lives in `‖x*‖`, not the coefficients — are left
        // **bit-for-bit unchanged** (`σ = 1`, the wrapper is a no-op).
        let sigma = hsde_cost_scale(prob, opts.tol);
        if sigma != 1.0 {
            let scaled = prob.scaled_objective(1.0 / sigma);
            // The normalized objective is the original divided by `σ`; the
            // caller's objective constant is in the original metric, so it is
            // divided too (see [`QpOptions::obj_constant`]).
            let inner = QpOptions {
                obj_constant: opts.obj_constant / sigma,
                ..*opts
            };
            let sol =
                cost_normalized_hsde_solve(&scaled, cone, &inner, sigma, &mut make_backend, hook);
            // gh #324: the cost normalization divides the objective by σ (sized
            // to ‖P‖∞). When ‖c‖ ≪ σ — a huge Hessian coefficient paired with a
            // modest gradient, e.g. `P = diag(1e-10, 1e10)`, `c = [-1, -1]` — the
            // *scaled* cold-start dual residual ‖c/σ‖ underflows below `tol`, so
            // the embedding certifies `Optimal` at the untouched start (x = 0),
            // nowhere near stationary in the original metric (kkt_error ≈ ‖c‖).
            // A normalized `Optimal` must therefore be re-checked against the
            // true, un-normalized *relative* KKT residual. When it is spurious
            // the un-normalized solve — well-conditioned whenever σ has not
            // actually collapsed τ, which is exactly the ‖c‖ ≪ σ regime — reaches
            // the real optimum; if that solve cannot converge either, its honest
            // non-`Optimal` status stands (never a false `Optimal`).
            // A debugger stop lands here as a non-`Optimal` status, and the
            // re-solves below run unhooked — so the halted run would be
            // replaced by one the debugger never saw (gh #892). Take the
            // stopped result as it stands.
            if sol.status != QpStatus::Optimal
                || crate::debug_stop::requested()
                || normalized_optimum_is_genuine(prob, cone, &sol, opts.tol)
            {
                tracing::debug!(
                    sigma,
                    status = ?sol.status,
                    "convex sigma: normalized solve accepted"
                );
                return sol;
            }
            tracing::debug!(
                sigma,
                kkt_error = sol.kkt_residuals(prob).kkt_error(),
                "convex sigma: normalized optimum rejected, re-solving un-normalized"
            );
            let plain = crate::hsde::solve_conic_hsde(prob, cone, opts, &mut make_backend, None);
            if plain.status == QpStatus::Optimal
                && normalized_optimum_is_genuine(prob, cone, &plain, opts.tol)
            {
                tracing::debug!("convex sigma: un-normalized re-solve accepted");
                return plain;
            }
            // An infeasibility / unboundedness certificate is not this guard's
            // to overturn — it is positive evidence about the problem rather
            // than a claimed optimum, and the historical code returned it here
            // unconditionally. Only a *claimed optimum* (or a non-convergence,
            // which claims nothing) goes on to the third driver.
            if matches!(
                plain.status,
                QpStatus::PrimalInfeasible | QpStatus::DualInfeasible
            ) {
                return plain;
            }
            // gh #846: the un-normalized re-solve is still the *embedding*, and
            // the embedding's own stopping test normalizes the duality gap by
            // the objective's magnitude (`gap / (1 + |obj|)`, see
            // `hsde::solve_conic_hsde`). On data whose coefficients reach `1e11`
            // that licenses an absolute gap of `tol·|obj|`, which on the
            // flattest curvature in the spectrum is a large distance in `x`:
            // measured on the reported 6-variable box QP, `|obj| = 1.18e10` and
            // the returned `x₀` sits `2.9e-3` off its bound. So `σ` is an
            // amplifier here, not the origin, and rejecting its certificate
            // only moves the caller from `1.6e-1` wrong to `2.9e-3` wrong.
            //
            // The direct driver below applies its stopping test in the caller's
            // own coordinates and has no objective-relative gap arm, so it is
            // untouched by that. Measured across `mag = 1e7 ‥ 1e14` on the
            // reported family it returns `‖x − x*‖∞ ≤ 1e-6` at every magnitude
            // while the embedding degrades from `1e9` up. Its answer is taken
            // only when it passes the same test the two embedding answers just
            // failed, so this can substitute a *certified* point for an
            // uncertified one and nothing else.
            //
            // Reached only after two `Optimal` answers have both been judged
            // false, which on the corpora is never: 1 of 79 CLI fixtures
            // reaches `σ` at all and 0 of 138 Maros-Meszaros problems do.
            // The direct driver is an **orthant-only** entry point: Ruiz is a
            // row scaling and `solve_qp_ipm_core`'s own comment says
            // cone-carrying problems never reach it, since SOC/exp/power solve
            // through `solve_socp_ipm`. Handing it a QCQP silently drops the
            // cone structure and returns the answer to a different problem —
            // measured on `qcqp_columns_illcond`, `-210.53` against the
            // `-364.2102` that `solver_selection=nlp` and `qp_hsde=no` both
            // agree on. So on anything but a pure orthant this fallback does
            // not exist, and the un-normalized re-solve stands exactly as it
            // did before gh #846.
            if !cone
                .blocks()
                .iter()
                .all(|(_, k)| matches!(k, crate::cones::ConeKind::Nonneg(_)))
            {
                tracing::debug!(
                    "convex sigma: non-orthant cone, keeping the un-normalized \
                     re-solve"
                );
                return plain;
            }
            let direct = direct_driver_fallback(prob, opts, &mut make_backend);
            if direct.status == QpStatus::Optimal
                && normalized_optimum_is_genuine(prob, cone, &direct, opts.tol)
            {
                tracing::debug!("convex sigma: direct-driver fallback accepted");
                return direct;
            }
            // Nothing was certified. Rather than default to any one driver,
            // hand back whichever claimed optimum is closest to optimality in
            // the **caller's own coordinates** — `kkt_error` is absolute and
            // un-normalized, so this is the caller's own definition of the
            // thing, and it is a ranking rather than another threshold to
            // calibrate. It cannot promote a non-converged iterate over a
            // converged one: only `Optimal` candidates are eligible, and if
            // none is, the un-normalized re-solve's honest status stands as it
            // always did.
            //
            // Measured on gh #846's family this is what closes the last two
            // gaps. At `‖P‖ ~ 1e23` the embedding reaches its iteration cap
            // both times while the direct driver converges in 22 iterations to
            // `1e-17`; on a spectrum reaching down to `1e-2` no candidate is
            // certifiable at all, and the direct driver's `1.3e-5` is returned
            // instead of the embedding's `9.9e-1`.
            // Index 1 is the un-normalized re-solve, whose honest status is
            // what a caller got before this fallback existed and is therefore
            // the default when nothing claims an optimum at all.
            let mut candidates = vec![sol, plain, direct];
            let pick = candidates
                .iter()
                .enumerate()
                .filter(|(_, c)| c.status == QpStatus::Optimal)
                .min_by(|(_, a), (_, b)| {
                    a.kkt_residuals(prob)
                        .kkt_error()
                        .total_cmp(&b.kkt_residuals(prob).kkt_error())
                })
                .map_or(1, |(i, _)| i);
            tracing::debug!(
                pick,
                kkt_error = candidates[pick].kkt_residuals(prob).kkt_error(),
                status = ?candidates[pick].status,
                "convex sigma: nothing certified, returning the closest \
                 claimed optimum"
            );
            // gh #880: the point is the best available and stays, but the
            // cascade has now declined to certify it three times over and the
            // caller must not be handed it under a bare `Optimal`. Recorded
            // rather than applied here: `OptimalInaccurate` is one of the
            // statuses `solve_qp_ipm_core` reads as the badly-scaled pathology
            // worth a Ruiz-equilibrated retry, and that retry is accepted
            // whenever it converges to a clean `Optimal` — its status
            // outranking the better answer. Measured, demoting at this line
            // takes the gh #880 census's worst forward error from `3.06e-04`
            // to `3.93e+01`. `solve_qp_ipm` applies it once every retry has
            // run.
            // What gets recorded is **not** "the cascade certified nothing".
            // That verdict mixes two different questions: whether the residual
            // is small in *absolute* terms, which legitimately depends on the
            // objective's scale, and whether `x` is far from the optimum,
            // which does not. Recording the mixture makes the status flap
            // under an objective rescaling that leaves the argmin unchanged by
            // identity — measured over `k = 1e-2 ‥ 1e8` on a `cond = 1e10`
            // coupled model it oscillates Optimal/Inaccurate six times, which
            // is not a contract anyone can use, and is the same
            // scale-dependence gh #880 is about in the first place.
            //
            // The forward-error arm is the scale-free half. Unconstrained,
            // `Δ = −P⁻¹(Px + c)` is exactly invariant under
            // `(P, c) → k(P, c)`, and `‖Δ‖/max(1, ‖x‖)` is the basis-free
            // distance this issue added it to measure. It is the arm that
            // answers the question a status should answer.
            // ... and asked with a margin, because a status must not flip on
            // rounding noise. At `cond = 1e10` the realized forward error is
            // `1.47e-6` against a cut of `1e-6`: the model sits *on* the
            // decision boundary, and sweeping an inert objective rescaling
            // `k = 1e-2 ‥ 1e8` across it flips the verdict six times, since
            // each `k` reaches a slightly different iterate. That is
            // solve-to-solve variation, not summation noise. `cond = 1e12`
            // (`3.06e-04`, 300x the cut) is decisive at every `k`.
            //
            // **The constant is pinned from both sides by tests that already
            // exist**, which is a checkable claim rather than an asserted one:
            //
            // | margin | outcome |
            // |---|---|
            // | `<= 2` | `an_inert_objective_rescaling_stays_inert_on_the_coupled_arm` fails — the rescaling flaps |
            // | `5`, `10` | green |
            // | `>= 50` | `every_tolerance_is_correct_on_the_coupled_arm` fails — stops demoting where it must |
            // | `>= 200` | `a_bound_constrained_coupled_instance_is_solved` fails too |
            //
            // So the admissible window is `(2, 50)` and `10` is roughly its
            // geometric centre. (An earlier draft justified this by the plain
            // summation's `0.29x`–`5.4x` spread against a `Fraction`
            // reference; that is the noise of an estimator this same change
            // replaces with `residual_compensated`, so it cannot be the
            // reason.)
            let far_from_optimal = !sigma_forward_error_is_small(
                prob,
                &candidates[pick],
                SIGMA_DEMOTION_MARGIN * sigma_path_rel_tol(opts.tol),
            );
            // NOTE the bit recorded is narrower than the module's parameter
            // name suggests: it is "this answer should be demoted", i.e. the
            // pick is *currently* labelled `Optimal` **and** it is far from
            // optimal. An uncertified pick that already carries a non-`Optimal`
            // status records `false`, because there is nothing to demote. That
            // is exactly what `demote_uncertified_sigma_optimum` needs and the
            // two checks are deliberately redundant, but a second consumer —
            // a diagnostic, a log line, a sensitivity gate — would
            // under-report uncertified points if it read this bit as "the
            // cascade certified nothing". Widen the recording, not the reader.
            crate::sigma_verdict::record(
                candidates[pick].status == QpStatus::Optimal && far_from_optimal,
            );
            return candidates.swap_remove(pick);
        }
        return crate::hsde::solve_conic_hsde(prob, cone, opts, make_backend, hook);
    }

    // Build the fixed KKT pattern and an initial factorization, then run
    // the iteration. The pattern is constant across iterations (only the
    // cone scaling block changes), so the loop `refactor`s rather than
    // re-analyzing. Build-once / solve-many across *instances* with the
    // same pattern is exposed via [`QpFactorization`].
    let (kkt, mut fact) = match build_factorization(prob, cone, opts, &mut make_backend) {
        Ok(pair) => pair,
        Err(()) => {
            let n = prob.n;
            return failed_solution(
                prob,
                vec![0.0; n],
                vec![0.0; prob.m_eq()],
                vec![0.0; prob.m_ineq()],
                0,
            );
        }
    };
    if crate::deadline::expired() {
        return timed_out_solution(prob);
    }
    run_ipm(prob, cone, opts, &kkt, &mut fact, warm, hook)
}

/// Build the constant KKT pattern for `prob` and a `Factorization` over
/// it (seeded with the initial scaling). Shared by the single-shot path
/// and the reusable [`QpFactorization`] handle. `Err(())` ⇒ the initial
/// factorization failed.
pub(crate) fn build_factorization<F>(
    prob: &QpProblem,
    cone: &CompositeCone,
    opts: &QpOptions,
    make_backend: &mut F,
) -> Result<(KktStructure, Factorization), ()>
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    // Seed the scaling at the cone identity (s = z = e ⇒ block = I).
    let mut e = vec![0.0; prob.m_ineq()];
    cone.identity(&mut e);

    let kkt = KktStructure::build(prob, cone, opts.reg);
    let dim = kkt.dim; // base rows + per-SOC auxiliary variables
    let mut kkt_vals = kkt.values.clone();
    kkt.update_blocks(cone, &e, &e, opts.reg, &mut kkt_vals);
    // gh#986 item 3: the seed factorization is numeric as well as symbolic,
    // and at `qp_reg = 0` a consistent duplicated equality row makes it
    // singular (a zero pivot on a variable only the dependent rows touch) —
    // the solve then died at iteration 0 as a `NumericalFailure`. The seed
    // values only feed the symbolic analysis (every iteration refactors with
    // its own regularization), so with *no* static regularization requested
    // the seed carries a token floor on the `(x, x)` and equality blocks.
    // Any `reg > 0` — the default included — seeds exactly as before.
    if opts.reg <= 0.0 {
        const SEED_REG_FLOOR: f64 = 1e-8;
        kkt.update_primal_reg(SEED_REG_FLOOR, &mut kkt_vals);
        kkt.update_eq_reg(SEED_REG_FLOOR, &mut kkt_vals);
    }
    let fact = Factorization::new(
        dim as Index,
        kkt.airn.clone(),
        kkt.ajcn.clone(),
        kkt_vals,
        make_backend(),
    )
    .map_err(|_| ())?;
    Ok((kkt, fact))
}

/// The **scale-relative** convergence arm of the direct driver: the primal and
/// dual residuals measured against the natural magnitude of their own terms,
/// and *permitted to conclude only* once `tol`-level absolute accuracy is below
/// the finite-precision floor ([`crate::hsde::relative_stop_permitted`], the
/// same gate and the same normalizers the HSDE loop already applies).
///
/// Without it the direct driver has no way to finish a solve whose data puts
/// the absolute test out of reach. `scaled_feasible_a` (gh #689) is the
/// canonical case: the Ruiz-equilibrated problem's optimum sits at `‖x̂‖ ≈ 5e9`
/// against `‖ĥ‖ ≈ 5e9`, so forming `Gx + s − h` cancels two `5e9` quantities
/// and the primal residual floors at `5e-6 ≈ 4 ulp` — a thousand times `tol` —
/// while the iterate is the *exact* optimum (its true KKT error reads `4e-25`).
/// The absolute test can never pass there, so the loop ran on past its own
/// answer until `s` and `z` underflowed into the denormals and the
/// factorization broke down: 175 iterations to a `NumericalFailure` sitting on
/// the optimum. With this arm the same solve stops at 27.
///
/// **Complementarity is deliberately left absolute.** The relaxation is
/// justified by *cancellation*, not by size: `Gx + s − h` and
/// `Px + c + Aᵀy + Gᵀz` are differences of like-magnitude terms, so their
/// achievable accuracy is `≈ scale·ε` and below that floor only a relative
/// statement is meaningful. `μ = ⟨s,z⟩/deg` is a **sum of products of
/// nonnegatives** — nothing cancels, and it converges to zero at any problem
/// scale — so there is no floor to excuse relaxing it, and relaxing it costs
/// real accuracy: normalizing `μ` by the objective magnitude (the shape
/// [`equilibrated_kkt_rel_parts`] and HSDE's `gap_rel` use) hands a QP whose
/// objective is dominated by a constant offset a blanket `|obj|`-sized
/// tolerance on the gap. On `scaled_feasible_a` that offset is `5e11`, so the
/// relative-gap form stops `~5e3` of objective early — the very
/// objective instability gh #689 reports on this fixture pair. Holding `μ`
/// absolute lands the same solve on the exact optimum.
///
/// Two gates, cheap-first. The outer one uses only quantities already to hand
/// (`‖c‖, ‖b‖, ‖h‖, ‖s‖` — each a term of `scale_d`/`scale_p`, hence a lower
/// bound on the natural scale, so it can only ever open *later* than the real
/// gate), which keeps an ordinarily-scaled solve — every solve where this arm
/// could not fire anyway — at one comparison and no matvecs.
fn scale_relative_stop(
    prob: &QpProblem,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    s: &[f64],
    pinf: f64,
    dinf: f64,
    mu: f64,
    tol: f64,
) -> bool {
    if !(mu < tol) {
        return false;
    }
    let norm_s = inf_norm(s);
    let cheap = inf_norm(&prob.c)
        .max(inf_norm(&prob.b))
        .max(inf_norm(&prob.h))
        .max(norm_s);
    if !crate::hsde::relative_stop_permitted(cheap, tol) {
        return false;
    }
    let (n, m_eq, m_ineq) = (prob.n, prob.m_eq(), prob.m_ineq());
    let mut px = vec![0.0; n];
    prob.p_mul(x, &mut px);
    let mut aty = vec![0.0; n];
    prob.at_mul(y, &mut aty);
    let mut gtz = vec![0.0; n];
    prob.gt_mul(z, &mut gtz);
    let mut ax = vec![0.0; m_eq];
    prob.a_mul(x, &mut ax);
    let mut gx = vec![0.0; m_ineq];
    prob.g_mul(x, &mut gx);

    let scale_d = inf_norm(&px)
        .max(inf_norm(&aty))
        .max(inf_norm(&gtz))
        .max(inf_norm(&prob.c));
    let scale_p = inf_norm(&ax)
        .max(inf_norm(&gx))
        .max(norm_s)
        .max(inf_norm(&prob.b))
        .max(inf_norm(&prob.h));
    if !crate::hsde::relative_stop_permitted(scale_d.max(scale_p), tol) {
        return false;
    }
    pinf / (1.0 + scale_p) < tol && dinf / (1.0 + scale_d) < tol
}

/// Build the starting iterate `(x, y, z, s)` for [`run_ipm`] by **Mehrotra-style
/// recentering** (Mehrotra 1992, §7) of a seed point — the warm start when one
/// is supplied, otherwise the origin `x = 0, y = 0, z = e`.
///
/// The cold seed goes through the same recentering as a warm one, which is what
/// sizes it to the problem's own data (gh #689). The historical cold start,
/// `s = z = e` regardless of the data, is *not* a starting point: it is a fixed
/// point of unit scale asserted over a problem whose slacks may live anywhere.
/// On `scaled_feasible_a` the Ruiz-equilibrated feasible set sits at
/// `‖h‖ ≈ 5e9`, so from `s = e` the very first Newton direction — a perfectly
/// good `‖dx‖ ≈ 2.9e9`, pointed at the optimum — was cut by
/// fraction-to-boundary to `α ≈ 8e-9`. The iterate could not move; the
/// corrector, dividing `σμ` by slacks pinned at `1`, then returned directions
/// of `1e18`, `z` blew up to `7e21`, and the solve diverged to the iteration
/// cap at `kkt_error 8e45`. Seeding `s` from the implied slacks instead makes
/// the same solve converge in 27 iterations. (This is the failure mode
/// [`QpOptions::use_hsde`] documents the direct driver for — NETLIB `nl`,
/// "`mu` to ~1e11" — with a measurement attached.)
///
/// The recentering, for either seed:
///
/// 1. Keep the seed primal `x` and equality multipliers `y`.
/// 2. Take the implied slacks `s̃ = h − Gx` (their signs encode which
///    inequalities the seed `x` makes active/violated) and the seed `z`.
///    From the origin this is `s̃ = h`, so the primal scale of the start is
///    the problem's own.
/// 3. Shift both into the strict interior by `δ = max(−1.5·min(·), floor)`.
///    The `floor` is **adaptive**: it is the seed point's KKT residual `ρ`
///    on *this* problem, clamped to `[1e-9·scale, 0.1·scale]` with
///    `scale = max(1, ‖s̃‖∞, ‖z‖∞)`. A converged warm point sits on the
///    complementarity boundary (`s̃ᵢ` or `zᵢ ≈ 0`), so a floor is required
///    to keep the restart interior — but a *fixed* floor overwrites the
///    warm dual structure and degrades to a primal-only warm start.
///    Sizing the floor to `ρ` keeps `s`/`z` near their warm (correctly
///    structured) values when the problem is nearby (small `ρ`), so the
///    IPM exploits the warm duals — and softens toward the conservative
///    `0.1·scale` when the active set has moved (large `ρ`). This both
///    deepens the benefit on nearby problems and keeps it from ever doing
///    worse than a centered start. From the cold seed the same rule reads
///    as "size the interior floor to the residual the origin leaves", which
///    is the scale the first Newton step has to work in.
/// 4. A final centering shift `½(s·z)/Σz`, `½(s·z)/Σs` balances `s` and
///    `z` (Mehrotra's second step).
///
/// The returned iterate always satisfies `s > 0, z > 0`. If `warm`'s
/// dimensions don't match the (expanded) problem it is ignored and the
/// cold seed is used, so a stale warm start can never corrupt a solve.
fn init_iterate(
    prob: &QpProblem,
    cone: &CompositeCone,
    n: usize,
    m_eq: usize,
    m_ineq: usize,
    warm: Option<&WarmStart>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // The seed point. A matching warm primal `x` is enough to warm start;
    // `y`/`z` fall back to the cold values when they don't match (so a
    // primal-only warm start — e.g. feeding back just the previous primal —
    // is supported). With no warm start the seed is the origin with the cone
    // identity duals — the historical cold start's `(x, y, z)`.
    //
    // Both seeds then go through the *same* Mehrotra recentering below, which
    // is what sizes the cold start to the problem's own data (gh #689). See
    // this function's doc comment for why `s = z = e` is not a starting point.
    let mut ident = vec![0.0; m_ineq];
    cone.identity(&mut ident);
    let (x, y, mut z) = match warm {
        Some(w) if w.x.len() == n => (
            w.x.clone(),
            if w.y.len() == m_eq {
                w.y.clone()
            } else {
                vec![0.0; m_eq]
            },
            if w.z.len() == m_ineq {
                w.z.clone()
            } else {
                ident.clone()
            },
        ),
        _ => (vec![0.0; n], vec![0.0; m_eq], ident.clone()),
    };

    // No cone: x/y are the whole iterate, s/z are empty.
    if m_ineq == 0 {
        return (x, y, z, Vec::new());
    }

    // Implied slacks s̃ = h − Gx.
    let mut gx = vec![0.0; m_ineq];
    prob.g_mul(&x, &mut gx);
    let mut s: Vec<f64> = (0..m_ineq).map(|i| prob.h[i] - gx[i]).collect();

    let scale = 1.0_f64.max(inf_norm(&s)).max(inf_norm(&z));

    // Adaptive interior floor sized to the seed point's KKT residual ρ on
    // *this* problem. ρ measures how far the seed is from satisfying the
    // KKT system: a small ρ (nearby problem, stable active set) lets the
    // slacks/multipliers stay near their warm — correctly structured —
    // values, so the IPM exploits the warm duals and needs few steps; a
    // large ρ (the active set moved, so the warm point is badly infeasible)
    // softens the floor toward the conservative `0.1·scale`. This
    // self-corrects: warm starting never does worse than a centered start,
    // and gains the most when it can. From the cold seed ρ is just the
    // residual the origin leaves, which is the scale the first Newton step
    // has to work in.
    let floor = {
        let mut rd = prob.c.clone();
        prob.p_mul_add(&x, &mut rd);
        prob.at_mul_add(&y, &mut rd);
        prob.gt_mul_add(&z, &mut rd);
        let mut rp: Vec<f64> = prob.b.iter().map(|b| -b).collect();
        prob.a_mul_add(&x, &mut rp);
        // Inequality infeasibility of the seed point: max(0, Gx − h) = −s̃.
        let viol = s.iter().fold(0.0_f64, |m, &si| m.max((-si).max(0.0)));
        let rho = inf_norm(&rd).max(inf_norm(&rp)).max(viol);
        rho.clamp(1e-9 * scale, 0.1 * scale)
    };
    // Project (s, z) into the strict interior of each cone block and
    // rebalance (orthant: positivity + Mehrotra; SOC: lift λ_min).
    cone.recenter_warm(&mut s, &mut z, floor);
    (x, y, z, s)
}

/// Run the Mehrotra predictor-corrector iteration for `prob` given an
/// already-built KKT pattern (`kkt`) and a live `Factorization` (`fact`)
/// over that pattern. The factorization is re-numeric-factored each
/// iteration (symbolic reuse); when `fact` is reused across instances
/// with the *same pattern*, the AMD ordering / symbolic factor is reused
/// across instances too.
fn run_ipm(
    prob: &QpProblem,
    cone: &CompositeCone,
    opts: &QpOptions,
    kkt: &KktStructure,
    fact: &mut Factorization,
    warm: Option<&WarmStart>,
    mut hook: Option<&mut dyn DebugHook>,
) -> QpSolution {
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();

    let (mut x, mut y, mut z, mut s) = init_iterate(prob, cone, n, m_eq, m_ineq, warm);

    let mut r_d = vec![0.0; n];
    let mut r_p = vec![0.0; m_eq];
    let mut r_g = vec![0.0; m_ineq];
    let mut r_c = vec![0.0; m_ineq];
    let mut rhs_term = vec![0.0; m_ineq];
    // The KKT system carries one auxiliary variable per second-order cone;
    // the rhs is sized to it (auxiliary rows are zero).
    let mut rhs = vec![0.0; kkt.dim];
    let mut dx = vec![0.0; n];
    let mut dy = vec![0.0; m_eq];
    let mut dz = vec![0.0; m_ineq];
    let mut ds = vec![0.0; m_ineq];
    let mut ds_aff = vec![0.0; m_ineq];
    let mut dz_aff = vec![0.0; m_ineq];
    let mut kkt_vals = kkt.values.clone();
    // Set when the factorization rescue raised δ_w this iteration.
    let mut primal_reg_raised = false;

    // Gondzio centrality-corrector scratch: one extra direction plus the
    // trial combined step, and the zero linear residual a corrector solve
    // takes. Allocated only when correctors can actually fire — the scheme is
    // orthant-only (`crate::correctors`), so a SOCP/PSD solve pays nothing.
    let correcting = opts.gondzio_max_corr > 0 && m_ineq != 0 && cone.is_orthant();
    let scratch = |k: usize| vec![0.0; if correcting { k } else { 0 }];
    let (mut cdx, mut cdy, mut cdz, mut cds) =
        (scratch(n), scratch(m_eq), scratch(m_ineq), scratch(m_ineq));
    let (mut step_s, mut step_z) = (scratch(m_ineq), scratch(m_ineq));
    let (zeros_n, zeros_meq, zeros_m) = (scratch(n), scratch(m_eq), scratch(m_ineq));
    let mut tally = correctors::Tally::default();

    let mut iters = 0;
    let mut status = QpStatus::IterationLimit;
    let mut iterates: Vec<QpIterate> = Vec::new();
    let (dir_unit_d, dir_unit_g) = crate::qp::objective_units(prob);

    for it in 0..opts.max_iter {
        iters = it;
        if crate::deadline::expired() {
            status = QpStatus::TimeLimit;
            break;
        }

        // --- residuals (unregularized; this is the convergence test) ---
        // r_d = P x + c + Aᵀ y + Gᵀ z
        r_d.iter_mut().zip(&prob.c).for_each(|(r, c)| *r = *c);
        prob.p_mul_add(&x, &mut r_d);
        prob.at_mul_add(&y, &mut r_d);
        prob.gt_mul_add(&z, &mut r_d);
        // r_p = A x − b
        r_p.iter_mut().zip(&prob.b).for_each(|(r, b)| *r = -*b);
        prob.a_mul_add(&x, &mut r_p);
        // r_g = G x + s − h
        for i in 0..m_ineq {
            r_g[i] = s[i] - prob.h[i];
        }
        prob.g_mul_add(&x, &mut r_g);

        let mu = cone.mu(&s, &z);
        let pinf = inf_norm(&r_p).max(inf_norm(&r_g));
        let dinf = inf_norm(&r_d);
        // gh#984: stationarity and `μ` are in the units of `(P, c)`, so an
        // absolute `tol` on them depends on the caller's choice of units
        // (`P·1e-9` stopped with the weights off by `1.6e-2`). Read against the
        // objective's own unit when it is below 1 -- the same rule the HSDE loop
        // applies (`hsde::solve_conic_hsde`), downward only: the direct driver
        // is already unit-invariant for an LP (equilibration normalizes `c`)
        // and a large unit is the scale-relative arm's.
        let res = (dinf / dir_unit_d).max(pinf).max(mu / dir_unit_g);
        // Per-iteration objective, needed for the trace and for the
        // debugger's `objective()` accessor.
        let obj_it = if opts.collect_iterates || hook.is_some() {
            let mut px = vec![0.0; n];
            prob.p_mul_add(&x, &mut px);
            (0..n).map(|i| 0.5 * x[i] * px[i] + prob.c[i] * x[i]).sum()
        } else {
            0.0
        };

        // Debugger checkpoint: top of iteration — residuals and the
        // accepted iterate from the previous step are in place; the
        // search direction (`dx`/…`) is the previous iteration's (zero on
        // the first), as on the NLP path.
        if hook.is_some() {
            let mut st = ConvexDebugState {
                cp: Checkpoint::IterStart,
                iter: it as i32,
                mu,
                pinf,
                dinf,
                res,
                obj: obj_it,
                alpha: (0.0, 0.0),
                x: &mut x,
                s: &mut s,
                y: &mut y,
                z: &mut z,
                dx: &dx,
                dy: &dy,
                dz: &dz,
                ds: &ds,
                tau: None,
                kappa: None,
                status: None,
            };
            if fire(&mut hook, &mut st) == DebugAction::Stop {
                break;
            }
        }

        // Breakdown: a non-finite iterate carries no information, and every
        // test below is a comparison against it. Stop and say so (gh #222).
        if !all_finite(&[&x, &s, &y, &z]) {
            status = QpStatus::NumericalFailure;
            break;
        }

        // Breakdown: the cones are self-dual, so `⟨s,z⟩ ≥ 0` for any iterate
        // genuinely inside them — a clearly negative μ means the iterate has
        // left the cone (a fraction-to-boundary failure) and every Newton
        // step from here is computed on meaningless data. Fail fast instead
        // of diverging to a non-finite iterate (gh #226). The threshold sits
        // orders of magnitude above the tiny negative values ordinary
        // round-off can produce as μ → 0 near convergence (|μ| ≲ ε·‖s‖‖z‖).
        if mu < -1e-10 * (1.0 + inf_norm(&s) * inf_norm(&z)) {
            status = QpStatus::NumericalFailure;
            break;
        }

        if res < opts.tol || scale_relative_stop(prob, &x, &y, &z, &s, pinf, dinf, mu, opts.tol) {
            status = QpStatus::Optimal;
            // Record the converged iterate so the trace *ends* at the
            // optimum, matching the NLP path's N+1 convention (a problem
            // solved in N steps logs N+1 records: the cold start through the
            // converged point). Every other record is pushed at the bottom of
            // the loop with the step that was taken *from* it; the converged
            // iterate takes no step, so its `alpha`s are zero. Without this a
            // solve that converges immediately (e.g. a tiny well-conditioned
            // QP in one step) would leave only the pre-step cold start in the
            // trace, and the trace's final objective would not be the optimum.
            if opts.collect_iterates {
                iterates.push(QpIterate {
                    iter: it,
                    objective: obj_it,
                    primal_infeasibility: pinf,
                    dual_infeasibility: dinf,
                    mu,
                    alpha_primal: 0.0,
                    alpha_dual: 0.0,
                });
            }
            break;
        }

        // Verified infeasibility / unboundedness detection. Checked
        // (not assumed), so a positive result is a proof and a false
        // positive is impossible; this is the HSDE benefit without the
        // homogeneous-embedding rewrite. Cheap (a few matvecs).
        if let Some(infeas) = detect_infeasibility_cone(prob, &x, &y, &z, opts, cone) {
            status = infeas;
            break;
        }

        // --- update the cone scaling block(s) and refactor (numeric-only;
        // the symbolic factor / ordering is reused). The one factorization
        // then backs both the predictor and corrector solves. ---
        // Undo a previous iteration's rescue of the (x, x) block (below), so a
        // single hard iterate never inflates δ_w for the rest of the solve.
        if primal_reg_raised {
            kkt.update_primal_reg(opts.reg, &mut kkt_vals);
            primal_reg_raised = false;
        }
        kkt.update_blocks(cone, &s, &z, opts.reg, &mut kkt_vals);
        // Adaptive μ-scaled regularization on the equality block: bounds the
        // duals of a rank-deficient equality Jacobian so the primal residual
        // converges below `tol` (see `adaptive_eq_reg`). Reduces to the static
        // `opts.reg` at the tolerance, leaving already-converging LPs/QPs
        // unchanged at the optimum.
        let mut delta_c = adaptive_eq_reg(mu, opts.reg);
        kkt.update_eq_reg(delta_c, &mut kkt_vals);
        // gh#986 item 3: with `qp_reg = 0` (or any reg below the roundoff
        // floor) a *consistent duplicated equality row* leaves a zero pivot in
        // the quasi-definite factorization — `(x, x)` carries `P_ii + reg = 0`
        // on a variable only the dependent rows touch — and the solve died at
        // iteration 0 as a `NumericalFailure`, 602 variables or 5. Only the
        // factorization *failure* path changes: raise δ_w (then δ_c) in the
        // HSDE driver's staged ladder, bounded, and refactor. A factorization
        // that succeeds first time — every solve that converged before — takes
        // exactly the path it always did.
        if fact.refactor(&kkt_vals).is_err() {
            use crate::hsde::{
                DELTA_C_FACTOR, DELTA_C_INIT, DELTA_C_MAX, DELTA_W_FACTOR, DELTA_W_INIT,
                DELTA_W_MAX,
            };
            let mut delta_w = opts.reg;
            let mut rescued = false;
            for _ in 0..20 {
                if delta_w < DELTA_W_MAX {
                    delta_w = (delta_w.max(DELTA_W_INIT) * DELTA_W_FACTOR).min(DELTA_W_MAX);
                } else if delta_c < DELTA_C_MAX {
                    delta_c = (delta_c.max(DELTA_C_INIT) * DELTA_C_FACTOR).min(DELTA_C_MAX);
                } else {
                    break;
                }
                kkt.update_primal_reg(delta_w, &mut kkt_vals);
                primal_reg_raised = true;
                kkt.update_eq_reg(delta_c, &mut kkt_vals);
                if fact.refactor(&kkt_vals).is_ok() {
                    rescued = true;
                    break;
                }
            }
            if !rescued {
                status = QpStatus::NumericalFailure;
                break;
            }
        }

        // === Predictor (affine-scaling) step: σ = 0 ===
        // r_c = s∘z (affine target).
        cone.comp_residual(&s, &z, 0.0, &mut r_c);
        cone.rhs_comp_term(&s, &z, &r_c, &mut rhs_term);
        build_rhs(&r_d, &r_p, &r_g, &rhs_term, n, m_eq, m_ineq, &mut rhs);
        if fact.solve_one(&mut rhs).is_err() {
            status = QpStatus::NumericalFailure;
            break;
        }
        split_step(&rhs, n, m_eq, m_ineq, &mut dx, &mut dy, &mut dz);
        cone.recover_ds(&s, &z, &r_c, &dz, &mut ds_aff);
        dz_aff.copy_from_slice(&dz);

        // Affine step lengths and the predicted duality measure μ_aff. Held at
        // the static τ: μ_aff feeds Mehrotra's σ = (μ_aff/μ)³ heuristic, whose
        // calibration assumes the predictor's own damping.
        let (alpha_p_aff, alpha_d_aff) =
            step_lengths(cone, &s, &ds_aff, &z, &dz_aff, (opts.tau, opts.tau), m_ineq);
        let sigma = if m_ineq == 0 {
            0.0
        } else {
            // μ_aff = ⟨s + αp ds_aff, z + αd dz_aff⟩ / m
            let mut dot = 0.0;
            for i in 0..m_ineq {
                dot += (s[i] + alpha_p_aff * ds_aff[i]) * (z[i] + alpha_d_aff * dz_aff[i]);
            }
            let mu_aff = dot / m_ineq as f64;
            // Mehrotra's heuristic centering parameter σ = (μ_aff/μ)³.
            (mu_aff / mu).powi(3)
        };

        // === Corrector step: centered target + second-order term ===
        // Compute the step direction (`dx`/`dy`/`dz`/`ds`) and the step
        // lengths taken this iteration, but defer *applying* it until after
        // the `AfterSearchDirection` checkpoint. With no cone the predictor
        // is already the full Newton step (`dz`/`ds` empty, full step).
        let (mut step_p, mut step_d) = (1.0_f64, 1.0_f64);
        if m_ineq != 0 {
            let sigma_mu = sigma * mu;
            cone.comp_residual_corrector(&s, &z, &ds_aff, &dz_aff, sigma_mu, &mut r_c);
            cone.rhs_comp_term(&s, &z, &r_c, &mut rhs_term);
            build_rhs(&r_d, &r_p, &r_g, &rhs_term, n, m_eq, m_ineq, &mut rhs);
            if fact.solve_one(&mut rhs).is_err() {
                status = QpStatus::NumericalFailure;
                break;
            }
            split_step(&rhs, n, m_eq, m_ineq, &mut dx, &mut dy, &mut dz);
            cone.recover_ds(&s, &z, &r_c, &dz, &mut ds);

            // The corrector step is the one that gets the Mehrotra tail
            // `τ → 1` on orthant blocks; non-orthant blocks keep `opts.tau`.
            let (alpha_p, alpha_d) = step_lengths(
                cone,
                &s,
                &ds,
                &z,
                &dz,
                (adaptive_tau(mu, opts), opts.tau),
                m_ineq,
            );
            step_p = alpha_p;
            step_d = alpha_d;

            // Breakdown: an exactly zero step (both lengths — the PSD
            // fraction-to-boundary returns 0 when the block has numerically
            // left the cone, gh #226) leaves the iterate bit-for-bit
            // unchanged, so every later pass recomputes the same direction
            // and the same zero step until the iteration cap. Stop now
            // instead; the final verdict below still salvages a near-optimal
            // iterate, and the PSD entry point falls back to HSDE on this
            // status. A *tiny but nonzero* step is deliberately not treated
            // as a stall: near a breakdown the direction can be huge, so
            // even a ~1e-15 step moves the iterate materially and some such
            // solves do recover.
            if step_p.max(step_d) <= 0.0 {
                status = QpStatus::NumericalFailure;
                break;
            }

            // === Gondzio multiple centrality correctors ===
            // The same scheme the HSDE driver runs (`crate::correctors` holds
            // the shared half), fitted to this driver's *split* primal/dual
            // step. Each pass enlarges each length by δ, projects the
            // complementarity products that trial step would produce into the
            // band `[β_lo·μ, β_hi·μ]`, and solves for the correction through
            // the factor already in hand — zero linear residual, complementarity
            // right-hand side only, so it costs one back-solve and no
            // refactorization.
            //
            // Accepted only when **both** lengths grow by at least γδ. Gondzio's
            // rule is stated per-length and a split step has two of them; taking
            // a corrector that lengthens one while shortening the other trades a
            // known gain for an unknown loss, and this driver's residuals are
            // not symmetric in the two, so the conservative conjunction is what
            // ships.
            // Gated on the Mehrotra step still being short: correcting an
            // already-long step trades the superlinear tail for at most 0.1 of
            // a step. See `correctors::ALPHA_MAX` for the measurement.
            if correcting && mu > 0.0 && correctors::worth_correcting((step_p, step_d)) {
                let band = correctors::Band::around(mu);
                let taus = (adaptive_tau(mu, opts), opts.tau);
                tally.iters += 1;
                for _ in 0..opts.gondzio_max_corr {
                    let trial = (
                        correctors::trial_step(step_p),
                        correctors::trial_step(step_d),
                    );
                    // r_c holds the deviation ṽ − t, so `recover_ds` yields a
                    // correction with z∘cds + s∘cdz = t − ṽ.
                    if !correctors::project_products(band, (&s, &ds), (&z, &dz), trial, &mut r_c) {
                        // Every product already centered: nothing to correct,
                        // and no back-solve spent finding that out.
                        break;
                    }
                    cone.rhs_comp_term(&s, &z, &r_c, &mut rhs_term);
                    build_rhs(
                        &zeros_n, &zeros_meq, &zeros_m, &rhs_term, n, m_eq, m_ineq, &mut rhs,
                    );
                    if fact.solve_one(&mut rhs).is_err() {
                        // A failed corrector is not a failed iteration: the
                        // Mehrotra direction in hand is still usable, so drop
                        // the correction and step with it.
                        break;
                    }
                    split_step(&rhs, n, m_eq, m_ineq, &mut cdx, &mut cdy, &mut cdz);
                    cone.recover_ds(&s, &z, &r_c, &cdz, &mut cds);
                    for i in 0..m_ineq {
                        step_s[i] = ds[i] + cds[i];
                        step_z[i] = dz[i] + cdz[i];
                    }
                    let (a_p, a_d) = step_lengths(cone, &s, &step_s, &z, &step_z, taus, m_ineq);
                    let keep = correctors::accepts(a_p, step_p) && correctors::accepts(a_d, step_d);
                    tally.record(keep, (a_p - step_p).min(a_d - step_d));
                    if !keep {
                        break;
                    }
                    for i in 0..n {
                        dx[i] += cdx[i];
                    }
                    for i in 0..m_eq {
                        dy[i] += cdy[i];
                    }
                    for i in 0..m_ineq {
                        dz[i] += cdz[i];
                        ds[i] += cds[i];
                    }
                    step_p = a_p;
                    step_d = a_d;
                }
            }
        }

        // Debugger checkpoint: the Newton step and its fraction-to-boundary
        // lengths are known but not yet applied.
        if hook.is_some() {
            let mut st = ConvexDebugState {
                cp: Checkpoint::AfterSearchDirection,
                iter: it as i32,
                mu,
                pinf,
                dinf,
                res,
                obj: obj_it,
                alpha: (step_p, step_d),
                x: &mut x,
                s: &mut s,
                y: &mut y,
                z: &mut z,
                dx: &dx,
                dy: &dy,
                dz: &dz,
                ds: &ds,
                tau: None,
                kappa: None,
                status: None,
            };
            if fire(&mut hook, &mut st) == DebugAction::Stop {
                break;
            }
        }

        // Apply the step (the no-cone full step is `step_p = step_d = 1`).
        for i in 0..n {
            x[i] += step_p * dx[i];
        }
        for i in 0..m_eq {
            y[i] += step_d * dy[i];
        }
        for i in 0..m_ineq {
            s[i] += step_p * ds[i];
            z[i] += step_d * dz[i];
        }
        if crate::deadline::expired() {
            status = QpStatus::TimeLimit;
            break;
        }

        // Debugger checkpoint: the new iterate is in place.
        if hook.is_some() {
            let mut st = ConvexDebugState {
                cp: Checkpoint::AfterStep,
                iter: it as i32,
                mu,
                pinf,
                dinf,
                res,
                obj: obj_it,
                alpha: (step_p, step_d),
                x: &mut x,
                s: &mut s,
                y: &mut y,
                z: &mut z,
                dx: &dx,
                dy: &dy,
                dz: &dz,
                ds: &ds,
                tau: None,
                kappa: None,
                status: None,
            };
            if fire(&mut hook, &mut st) == DebugAction::Stop {
                break;
            }
        }

        if opts.collect_iterates {
            iterates.push(QpIterate {
                iter: it,
                objective: obj_it,
                primal_infeasibility: pinf,
                dual_infeasibility: dinf,
                mu,
                alpha_primal: step_p,
                alpha_dual: step_d,
            });
        }

        // This iteration took its step; count it, so a run that exhausts
        // `max_iter` reports `max_iter` rather than `max_iter - 1` (gh#987).
        // Early exits (convergence, a verdict) break above, before here, and
        // keep `iters = it`: the number of steps taken before the test passed.
        iters = it + 1;
    }

    // `!is_verdict`: the loop breaks with `Optimal` the moment its convergence
    // test passes, and the deadline can cross in the residual/objective work
    // that follows. Stamping `TimeLimit` over that conclusion would throw away
    // an answer this solve *did* reach — see [`mark_timed_out`].
    if crate::deadline::expired() && !is_verdict(status) {
        status = QpStatus::TimeLimit;
    }

    // Final verdict from the true KKT error of the point being returned — the
    // same rule the HSDE driver applies (see `VERDICT` in `hsde.rs`), so the two
    // drivers cannot drift apart on whether a solve that ended without its own
    // verdict actually produced an answer. Strictly an upgrade.
    //
    // `TimeLimit` belongs in this set for the same reason `IterationLimit`
    // does: all three are "stopped without concluding", and a cancelled solve
    // whose last iterate happens to satisfy the KKT conditions to `tol` has an
    // answer sitting right there. Excluding it would report `TimeLimit` on a
    // point this very block is about to certify as optimal.
    if matches!(
        status,
        QpStatus::NumericalFailure | QpStatus::IterationLimit | QpStatus::TimeLimit
    ) {
        let candidate = QpSolution {
            status,
            x: x.clone(),
            y: y.clone(),
            z: z.clone(),
            z_lb: vec![0.0; n],
            z_ub: vec![0.0; n],
            obj: 0.0,
            iters,
            iterates: Vec::new(),
        };
        let in_dual_cone = cone.in_dual_cone(&z, 1e-9);
        let true_res = candidate
            .kkt_residuals_conic(prob, &cone.specs())
            .kkt_error();
        status = match true_res {
            e if in_dual_cone && e < opts.tol => QpStatus::Optimal,
            e if in_dual_cone && e < 1e3 * opts.tol => QpStatus::OptimalInaccurate,
            _ => status,
        };
    }

    // Objective ½ xᵀP x + cᵀx.
    let mut px = vec![0.0; n];
    prob.p_mul_add(&x, &mut px);
    let mut obj = 0.0;
    for i in 0..n {
        obj += 0.5 * x[i] * px[i] + prob.c[i] * x[i];
    }

    // Debugger post-mortem at the final iterate (the returned action is
    // ignored — the solve is over).
    if hook.is_some() {
        let status_str = format!("{status:?}");
        let mut st = ConvexDebugState {
            cp: Checkpoint::Terminated,
            iter: iters as i32,
            mu: cone.mu(&s, &z),
            pinf: inf_norm(&r_p).max(inf_norm(&r_g)),
            dinf: inf_norm(&r_d),
            res: 0.0,
            obj,
            alpha: (0.0, 0.0),
            x: &mut x,
            s: &mut s,
            y: &mut y,
            z: &mut z,
            dx: &dx,
            dy: &dy,
            dz: &dz,
            ds: &ds,
            tau: None,
            kappa: None,
            status: Some(&status_str),
        };
        let _ = fire(&mut hook, &mut st);
    }

    let nn = n;
    tally.report("direct", iters);
    // Never hand back a success verdict without a usable solution (gh #222).
    let status = demote_unusable(status, &x, obj);
    QpSolution {
        status,
        x,
        y,
        z,
        z_lb: vec![0.0; nn],
        z_ub: vec![0.0; nn],
        obj,
        iters,
        iterates,
    }
}

/// A reusable convex-QP factorization: build the KKT symbolic factor
/// (AMD ordering) **once** for a fixed problem *structure*, then solve
/// many instances that share that structure, paying the symbolic
/// analysis only on construction. This is the build-once / solve-many
/// handle (cf. the JAX `JaxProblem` from pounce#75) at the convex-QP
/// level.
///
/// "Same structure" means: same `n`, same `A`/`G`/`P` sparsity pattern,
/// and the same *set* of finite variable bounds (so the bound-expanded
/// KKT pattern is identical). Only the numeric data — `c`, `b`, `h`, and
/// the bound *values* — may change between solves. A solve whose problem
/// does not match the captured structure returns
/// [`QpStatus::NumericalFailure`] rather than silently producing a wrong
/// answer; use the one-shot [`solve_qp_ipm`] for heterogeneous problems.
pub struct QpFactorization {
    fact: Factorization,
    opts: QpOptions,
    /// The (orthant) inequality cone of the expanded problem; reused for
    /// the KKT pattern check and the per-solve scaling.
    cone: CompositeCone,
    /// Captured structure fingerprint for the per-solve compatibility
    /// check (same `n` and same expanded KKT pattern).
    n: usize,
    airn: Vec<Index>,
    ajcn: Vec<Index>,
}

impl QpFactorization {
    /// Build the reusable factor from a representative `base` problem.
    /// Returns `None` if the initial factorization fails (e.g. a
    /// structurally singular KKT system).
    pub fn build<F>(base: &QpProblem, opts: &QpOptions, mut make_backend: F) -> Option<Self>
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    {
        let expanded = if base.has_bounds() {
            expand_bounds(base).0
        } else {
            base.clone()
        };
        let cone = CompositeCone::single_nonneg(expanded.m_ineq());
        let (kkt, fact) = build_factorization(&expanded, &cone, opts, &mut make_backend).ok()?;
        Some(QpFactorization {
            airn: kkt.airn,
            ajcn: kkt.ajcn,
            n: base.n,
            fact,
            cone,
            opts: *opts,
        })
    }

    /// Solve `prob`, reusing the captured symbolic factor. `prob` must
    /// share the captured structure (see the type docs); otherwise a
    /// `NumericalFailure` solution is returned.
    pub fn solve(&mut self, prob: &QpProblem) -> QpSolution {
        crate::deadline::with_deadline(self.opts.time_limit, || {
            let sol = self.solve_inner(prob, None);
            if crate::deadline::expired() {
                mark_timed_out(sol)
            } else {
                sol
            }
        })
    }

    /// Solve `prob` reusing the captured symbolic factor **and** warm
    /// starting from `warm` (a nearby problem's solution). Combines the
    /// two reuse axes: the symbolic factorization is paid once at `build`,
    /// and the interior-point iteration is seeded from the warm point (see
    /// [`QpWarmStart`]). Same structure requirement as [`Self::solve`].
    pub fn solve_warm(&mut self, prob: &QpProblem, warm: &QpWarmStart) -> QpSolution {
        crate::deadline::with_deadline(self.opts.time_limit, || self.solve_warm_scoped(prob, warm))
    }

    fn solve_warm_scoped(&mut self, prob: &QpProblem, warm: &QpWarmStart) -> QpSolution {
        let (expanded_z, _) = if prob.has_bounds() {
            // `merge_bound_duals` needs the bound-row provenance.
            let (_, bound_rows) = expand_bounds(prob);
            (merge_bound_duals(prob, &bound_rows, warm), ())
        } else {
            (warm.z.clone(), ())
        };
        let w = WarmStart {
            x: warm.x.clone(),
            y: warm.y.clone(),
            z: expanded_z,
        };
        let sol = self.solve_inner(prob, Some(&w));
        if crate::deadline::expired() {
            mark_timed_out(sol)
        } else {
            sol
        }
    }

    fn solve_inner(&mut self, prob: &QpProblem, warm: Option<&WarmStart>) -> QpSolution {
        let (expanded, bound_rows) = if prob.has_bounds() {
            expand_bounds(prob)
        } else {
            (prob.clone(), Vec::new())
        };
        // Rebuild this instance's pattern and require it to match the
        // captured one exactly (same nnz, same row/col indices).
        let kkt = KktStructure::build(&expanded, &self.cone, self.opts.reg);
        if prob.n != self.n || kkt.airn != self.airn || kkt.ajcn != self.ajcn {
            return failed_solution(
                prob,
                vec![0.0; prob.n],
                vec![0.0; prob.m_eq()],
                vec![0.0; prob.m_ineq()],
                0,
            );
        }
        // Reuse the live factorization (it carries the symbolic analysis;
        // `run_ipm` refactors numerically per iteration). The same factor
        // object is reused across solves, so the AMD ordering / symbolic
        // factor is paid once at `build`.
        let sol = run_ipm(
            &expanded,
            &self.cone,
            &self.opts,
            &kkt,
            &mut self.fact,
            warm,
            None,
        );
        split_bound_duals(prob, &bound_rows, sol)
    }
}

/// Whether the cone specs partition exactly `m_ineq` inequality rows — the
/// invariant the conic drivers assume (each `s = h − Gx` block sits in one
/// cone, with an exp/power cone occupying exactly 3 rows). A mismatch is a
/// caller error that would otherwise index past the slack vector.
fn cone_dims_cover(cones: &[ConeSpec], m_ineq: usize) -> bool {
    cones.iter().map(|c| c.dim()).sum::<usize>() == m_ineq
}

/// Build a `NumericalFailure` solution from the current iterate (used
/// when the *initial* factorization fails before the loop starts).
///
/// All failure call sites pass the trivial point `x = 0, y = 0, z = 0`.
/// The inequality dual `z` is **0**, not the cold-start identity `e`: a
/// failure carries no usable iterate, and `z = 0` (the cone apex) is the
/// one value valid in *every* dual cone — the orthant, but also SOC / PSD /
/// exponential / power, where the all-ones vector used previously is not
/// even a member (e.g. `(1,…,1)` violates an SOC of dimension ≥ 3). This
/// keeps the reported dual cone-feasible and consistent across all drivers
/// (cf. `hsde::failed`, `hsde_nonsym::failed`).
/// Last gate before a solution leaves the crate: a non-finite entry anywhere
/// in it is replaced by an honest zero-filled `NumericalFailure` (or a
/// zero-filled `TimeLimit` when the deadline is the authoritative verdict).
///
/// A `NaN` in the returned iterate is never information. It cannot be checked
/// against a bound, printed into a `.sol`, or fed to a warm start, and every
/// arithmetic it touches downstream returns `NaN` in turn — so it converts one
/// solver's failure into the caller's, several steps removed from the cause.
/// The status was already `NumericalFailure` in the case this was written for
/// (a `Gx ≤ h` system infeasible by about `1e-8`, where the iteration neither
/// converged nor certified), and `obj = NaN` still reached the CLI's own
/// summary line.
///
/// Deliberately a *guard*, not a repair: it does not attempt to salvage a
/// point, and it fires only on data no consumer could have used. A solve that
/// returns finite numbers passes through untouched, including one that failed.
///
/// Applied at the entry points a caller reaches — [`solve_qp_ipm`],
/// [`solve_qp_ipm_warm`], [`solve_socp_ipm`], and the active-set driver — and
/// deliberately **not** on the `*_debug` ones, where the raw iterate is the
/// thing being inspected and replacing it would hide what the hook was
/// attached to see.
pub(crate) fn finite_or_failed(prob: &QpProblem, sol: QpSolution) -> QpSolution {
    let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
    if finite(&sol.x)
        && finite(&sol.y)
        && finite(&sol.z)
        && finite(&sol.z_lb)
        && finite(&sol.z_ub)
        && sol.obj.is_finite()
    {
        return sol;
    }
    let iters = sol.iters;
    if sol.status == QpStatus::TimeLimit {
        let mut timed_out = timed_out_solution(prob);
        timed_out.iters = iters;
        return timed_out;
    }
    failed_solution(
        prob,
        vec![0.0; prob.n],
        vec![0.0; prob.m_eq()],
        vec![0.0; prob.m_ineq()],
        iters,
    )
}

fn failed_solution(
    prob: &QpProblem,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    iters: usize,
) -> QpSolution {
    let mut px = vec![0.0; prob.n];
    prob.p_mul_add(&x, &mut px);
    let mut obj = 0.0;
    for i in 0..prob.n {
        obj += 0.5 * x[i] * px[i] + prob.c[i] * x[i];
    }
    QpSolution {
        status: QpStatus::NumericalFailure,
        x,
        y,
        z,
        z_lb: vec![0.0; prob.n],
        z_ub: vec![0.0; prob.n],
        obj,
        iters,
        iterates: Vec::new(),
    }
}

fn timed_out_solution(prob: &QpProblem) -> QpSolution {
    QpSolution {
        status: QpStatus::TimeLimit,
        x: vec![0.0; prob.n],
        y: vec![0.0; prob.m_eq()],
        z: vec![0.0; prob.m_ineq()],
        z_lb: vec![0.0; prob.n],
        z_ub: vec![0.0; prob.n],
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    }
}

/// Label a *finished* solve as cancelled — but never at the cost of a verdict.
///
/// Every caller runs this after an inner solve has already returned, so `sol`
/// may well carry a real answer: the deadline crossing that brought us here can
/// land in the instant *between* convergence and the check. Overwriting an
/// `Optimal` (or a verified infeasible/unbounded certificate) with `TimeLimit`
/// there does not report a timeout, it discards a correct result — the user
/// whose problem solves in 1.001 s under `time_limit = 1` loses the optimum they
/// in fact computed, and every consumer downstream (the CLI's status mapping,
/// the SQP fallback) reads a non-answer.
///
/// So only a *non-verdict* status is relabelled. `IterationLimit` and
/// `NumericalFailure` describe a solve that stopped without concluding
/// anything, and for those "the clock ran out" is the more useful account of
/// why; `Optimal`, `OptimalInaccurate`, and the two certificates are
/// conclusions, and they stand.
pub(crate) fn mark_timed_out(mut sol: QpSolution) -> QpSolution {
    if !is_verdict(sol.status) {
        sol.status = QpStatus::TimeLimit;
    }
    sol
}

/// True when the status is a *conclusion about the problem* rather than a
/// record of the solver giving up — i.e. something a deadline crossing must not
/// erase. See [`mark_timed_out`].
pub(crate) fn is_verdict(status: QpStatus) -> bool {
    matches!(
        status,
        QpStatus::Optimal
            | QpStatus::OptimalInaccurate
            | QpStatus::PrimalInfeasible
            | QpStatus::DualInfeasible
    )
}

/// Build a `PrimalInfeasible` solution reported by a **setup-time** screen —
/// the cone-domain screen (gh #283) and the impossible-bound screen (gh #295,
/// a *present* `+∞` lower / `−∞` upper bound). Carries the trivial iterate; the
/// status is the certified result. `z = 0` (the cone apex) is dual-cone-feasible
/// in every cone.
fn trivial_primal_infeasible_solution(prob: &QpProblem) -> QpSolution {
    QpSolution {
        status: QpStatus::PrimalInfeasible,
        x: vec![0.0; prob.n],
        y: vec![0.0; prob.m_eq()],
        z: vec![0.0; prob.m_ineq()],
        z_lb: vec![0.0; prob.n],
        z_ub: vec![0.0; prob.n],
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    }
}

/// Build the Newton RHS `[−r_d; −r_p; −r_g + r_c ⊘ z]` for a given
/// complementarity residual `r_c` (predictor or corrector).
#[allow(clippy::too_many_arguments)]
/// Assemble the reduced KKT right-hand side `[-r_d; -r_p; -r_g + comp_term]`.
/// `comp_term` is the cone's contribution at the `(z)` rows (the orthant's
/// is `r_c ⊘ z`), computed by the caller via [`Cone::rhs_comp_term`] so the
/// block is cone-specific rather than baked in here.
pub(crate) fn build_rhs(
    r_d: &[f64],
    r_p: &[f64],
    r_g: &[f64],
    comp_term: &[f64],
    n: usize,
    m_eq: usize,
    m_ineq: usize,
    rhs: &mut [f64],
) {
    for i in 0..n {
        rhs[i] = -r_d[i];
    }
    for i in 0..m_eq {
        rhs[n + i] = -r_p[i];
    }
    for i in 0..m_ineq {
        rhs[n + m_eq + i] = -r_g[i] + comp_term[i];
    }
    // Auxiliary-variable rows (per second-order cone, appended after the
    // base rows) have zero right-hand side; re-zero them since `solve_one`
    // overwrote the buffer with the previous step.
    for v in rhs.iter_mut().skip(n + m_eq + m_ineq) {
        *v = 0.0;
    }
}

/// Copy the solved RHS into the (dx, dy, dz) step components.
pub(crate) fn split_step(
    rhs: &[f64],
    n: usize,
    m_eq: usize,
    m_ineq: usize,
    dx: &mut [f64],
    dy: &mut [f64],
    dz: &mut [f64],
) {
    dx.copy_from_slice(&rhs[0..n]);
    dy.copy_from_slice(&rhs[n..n + m_eq]);
    dz.copy_from_slice(&rhs[n + m_eq..n + m_eq + m_ineq]);
}

/// Separate fraction-to-boundary step lengths for the primal slack `s`
/// (via `ds`) and dual `z` (via `dz`). Returns `(alpha_primal,
/// alpha_dual)`; both are 1 when there is no cone.
///
/// `taus` is `(orthant, other)`: the first damps the nonnegative-orthant
/// blocks, the second every remaining cone kind. Passing the same value for
/// both is the plain [`Cone::max_step`]; only the corrector splits them — see
/// [`QpOptions::tau_max`].
fn step_lengths(
    cone: &CompositeCone,
    s: &[f64],
    ds: &[f64],
    z: &[f64],
    dz: &[f64],
    taus: (f64, f64),
    m_ineq: usize,
) -> (f64, f64) {
    if m_ineq == 0 {
        return (1.0, 1.0);
    }
    let (tau_orthant, tau_other) = taus;
    (
        cone.max_step_split(s, ds, tau_orthant, tau_other),
        cone.max_step_split(z, dz, tau_orthant, tau_other),
    )
}

/// Bench-only re-export of the KKT assembly so the `scaling` example can
/// time it in isolation. Not part of the public solving API.
#[doc(hidden)]
pub fn assemble_kkt_for_bench(
    prob: &QpProblem,
    scaling: &[f64],
    reg: f64,
    _dim: usize,
) -> (Vec<Index>, Vec<Index>, Vec<Number>) {
    let cone = CompositeCone::single_nonneg(prob.m_ineq());
    let kkt = KktStructure::build(prob, &cone, reg);
    let mut vals = kkt.values.clone();
    // Orthant block s/z = scaling at z = 1.
    let ones = vec![1.0; prob.m_ineq()];
    kkt.update_blocks(&cone, scaling, &ones, reg, &mut vals);
    (kkt.airn, kkt.ajcn, vals)
}

/// Fixed-pattern KKT structure for the QP augmented system.
///
/// The KKT *sparsity pattern* is identical across all IPM iterations —
/// only the `(z, z)` diagonal (the cone scaling block) changes from step
/// to step. This struct captures the pattern (`airn`/`ajcn`, 1-based
/// lower triangle) and the constant part of the values once, plus the
/// positions of the scaling-dependent diagonal entries, so each
/// iteration recomputes only `O(m_ineq)` values and the solver can
/// `refactor` (numeric-only, reusing the symbolic factor / fill-reducing
/// ordering) instead of rebuilding the factorization from scratch. This
/// is the constant-pattern symbolic reuse called for in
/// `dev-notes/performance-engineering.md`; without it the per-iteration
/// cost is dominated by repeated symbolic analysis on large sparse QPs.
/// Value-array positions of one cone's `(z, z)` scaling block, aligned with
/// the cone's [`CompositeCone::blocks`] order.
enum ZBlockPos {
    /// One value position per row (orthant diagonal).
    Diagonal(Vec<usize>),
    /// A second-order cone in **diagonal + rank-1** form, represented with
    /// one auxiliary variable `ξ`: the `(z,z)` diagonal entries, the
    /// coupling column `(z_i, ξ) = u_i`, and the `(ξ,ξ) = +1` entry. Its
    /// Schur complement reproduces the dense block `diag(d) + uuᵀ`, keeping
    /// the factorization sparse (ECOS/Clarabel sparse-SOC trick).
    DiagRank1 {
        diag_pos: Vec<usize>,
        u_pos: Vec<usize>,
        aux_pos: usize,
    },
    /// A fully dense symmetric block (the PSD cone's `W ⊗ₛ W`): the
    /// value-array positions of its lower triangle, row-major
    /// `[(0,0),(1,0),(1,1),…]`, aligned with [`ConeBlock::DenseLower`].
    Dense { pos: Vec<usize> },
}

/// How a cone block enters the `(z,z)` position of the KKT system.
#[derive(Clone, Copy, PartialEq)]
enum BlockShape {
    /// Orthant: one diagonal entry per row.
    Diagonal,
    /// Second-order cone: diagonal + rank-1 via an auxiliary variable.
    DiagRank1,
    /// PSD cone: a fully dense symmetric lower-triangle block.
    Dense,
}

pub(crate) struct KktStructure {
    pub(crate) airn: Vec<Index>,
    pub(crate) ajcn: Vec<Index>,
    /// Constant values (everything except the scaling block; the `(z, z)`
    /// diagonal entries hold their `-reg` term here).
    pub(crate) values: Vec<Number>,
    /// Total KKT dimension, including the per-SOC auxiliary variables.
    pub(crate) dim: usize,
    /// Per-cone `(z, z)` block positions, in `cone.blocks()` order.
    z_blocks: Vec<ZBlockPos>,
    /// Value-array positions of the `(y, y)` equality-multiplier diagonal,
    /// one per equality row. Seeded with `-reg` in [`Self::build`] and
    /// overwritten each iteration with the adaptive, μ-scaled `-δ_c` by
    /// [`Self::update_eq_reg`] — the Jacobian regularization that lets a
    /// rank-deficient equality system (redundant rows, non-unique duals)
    /// converge below `tol` instead of flooring the primal residual at
    /// `δ·‖dy‖`. Empty when there are no equality rows.
    y_diag_pos: Vec<usize>,
    /// Value-array positions of the `(x, x)` diagonal, one per column, and
    /// the `P` diagonal that sits under the regularization. Seeded with
    /// `P + reg` in [`Self::build`] and overwritten each iteration with
    /// `P + δ_w` by [`Self::update_primal_reg`].
    ///
    /// For an LP `P = 0`, so this block *is* the regularization: at the
    /// 1e-10 static default the x-pivots sit at the roundoff floor and LDLᵀ
    /// loses their signs, which reads out as a wrong-inertia deficit that no
    /// amount of `δ_c` / `(z, z)` escalation can repair — those bumps are on
    /// the wrong blocks. Ipopt's Algorithm IC escalates exactly this δ_w for
    /// wrong inertia; `δ_c` answers a rank-deficient equality Jacobian.
    x_diag_pos: Vec<usize>,
    x_diag_base: Vec<f64>,
}

impl KktStructure {
    /// Build the pattern and constant values once for `prob`'s inequality
    /// cone `cone`. Each cone block contributes either a diagonal entry per
    /// row (orthant) or a dense lower-triangle block (SOC) at its `(z, z)`
    /// position; all seeded with `-reg` on the diagonal. The pattern is
    /// constant across iterations — only the scaling values change — so the
    /// solver `refactor`s rather than re-analyzing.
    pub(crate) fn build(prob: &QpProblem, cone: &CompositeCone, reg: f64) -> Self {
        let n = prob.n;
        let m_eq = prob.m_eq();
        let mut entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut add = |r: usize, c: usize, v: f64| {
            let (r, c) = if r >= c { (r, c) } else { (c, r) };
            *entries.entry((r, c)).or_insert(0.0) += v;
        };

        // (x,x): P + δ_w I. The P diagonal is captured before the
        // regularization is folded in so `update_primal_reg` can rewrite δ_w
        // each iteration without losing P.
        let mut x_diag_base = vec![0.0f64; n];
        for t in &prob.p_lower {
            add(t.row, t.col, t.val);
            if t.row == t.col {
                x_diag_base[t.row] += t.val;
            }
        }
        for i in 0..n {
            add(i, i, reg);
        }
        // (y,x): A; (y,y): −δI.
        for t in &prob.a {
            add(n + t.row, t.col, t.val);
        }
        for i in 0..m_eq {
            add(n + i, n + i, -reg);
        }
        // (z,x): G.
        for t in &prob.g {
            add(n + m_eq + t.row, t.col, t.val);
        }
        // (z,z): per cone block, seeded with −δI. SOC blocks get an
        // auxiliary variable (appended after the base rows) carrying the
        // rank-1 term. The scaling values are written by `update_blocks`.
        let base_dim = n + m_eq + prob.m_ineq();
        let shapes = block_shapes(cone);
        let mut aux = base_dim; // next auxiliary-variable index
        for ((off, k), shape) in cone.blocks().iter().zip(&shapes) {
            let d = k.dim();
            let zbase = n + m_eq + off;
            for i in 0..d {
                add(zbase + i, zbase + i, -reg); // diagonal (filled per iter)
            }
            match shape {
                BlockShape::Diagonal => {}
                BlockShape::DiagRank1 => {
                    // Aux: coupling (z_i, ξ) = u_i and (ξ, ξ) = +1.
                    for i in 0..d {
                        add(aux, zbase + i, 0.0);
                    }
                    add(aux, aux, 1.0);
                    aux += 1;
                }
                BlockShape::Dense => {
                    // Reserve the strict lower triangle of the (z,z) block;
                    // the diagonal was already added above.
                    for i in 0..d {
                        for j in 0..i {
                            add(zbase + i, zbase + j, 0.0);
                        }
                    }
                }
            }
        }
        let dim = aux;

        let nnz = entries.len();
        let mut airn = Vec::with_capacity(nnz);
        let mut ajcn = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        let mut coord_to_pos: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (pos, ((r, c), v)) in entries.into_iter().enumerate() {
            airn.push((r + 1) as Index);
            ajcn.push((c + 1) as Index);
            values.push(v);
            coord_to_pos.insert((r, c), pos);
        }

        // Record each cone block's positions in `blocks()` order.
        let mut z_blocks = Vec::with_capacity(cone.blocks().len());
        let mut aux = base_dim;
        for ((off, k), shape) in cone.blocks().iter().zip(&shapes) {
            let d = k.dim();
            let zbase = n + m_eq + off;
            match shape {
                BlockShape::Diagonal => {
                    let diag_pos = (0..d)
                        .map(|i| coord_to_pos[&(zbase + i, zbase + i)])
                        .collect();
                    z_blocks.push(ZBlockPos::Diagonal(diag_pos));
                }
                BlockShape::DiagRank1 => {
                    let diag_pos = (0..d)
                        .map(|i| coord_to_pos[&(zbase + i, zbase + i)])
                        .collect();
                    let u_pos = (0..d).map(|i| coord_to_pos[&(aux, zbase + i)]).collect();
                    let aux_pos = coord_to_pos[&(aux, aux)];
                    z_blocks.push(ZBlockPos::DiagRank1 {
                        diag_pos,
                        u_pos,
                        aux_pos,
                    });
                    aux += 1;
                }
                BlockShape::Dense => {
                    // Lower triangle, row-major — matching ConeBlock::DenseLower.
                    let mut pos = Vec::with_capacity(d * (d + 1) / 2);
                    for i in 0..d {
                        for j in 0..=i {
                            pos.push(coord_to_pos[&(zbase + i, zbase + j)]);
                        }
                    }
                    z_blocks.push(ZBlockPos::Dense { pos });
                }
            }
        }

        // Positions of the (y,y) equality-multiplier diagonal, for the
        // per-iteration adaptive regularization. Built unconditionally; the
        // `-reg` seed is already in `values` from the loop above.
        let y_diag_pos: Vec<usize> = (0..m_eq).map(|i| coord_to_pos[&(n + i, n + i)]).collect();
        let x_diag_pos: Vec<usize> = (0..n).map(|i| coord_to_pos[&(i, i)]).collect();

        KktStructure {
            airn,
            ajcn,
            values,
            dim,
            z_blocks,
            y_diag_pos,
            x_diag_pos,
            x_diag_base,
        }
    }

    /// Write the per-iteration cone scaling into `out` (a copy of
    /// `self.values`): each block's `(z, z)` entries become `-(block) -
    /// reg·I`, from the cone's [`Cone::kkt_block`].
    pub(crate) fn update_blocks(
        &self,
        cone: &CompositeCone,
        s: &[f64],
        z: &[f64],
        reg: f64,
        out: &mut [Number],
    ) {
        for ((off, k), zb) in cone.blocks().iter().zip(&self.z_blocks) {
            let d = k.dim();
            let block = k.kkt_block(&s[*off..off + d], &z[*off..off + d]);
            match (zb, block) {
                (ZBlockPos::Diagonal(pos), ConeBlock::Diagonal(vals)) => {
                    for (i, &p) in pos.iter().enumerate() {
                        out[p] = -vals[i] - reg;
                    }
                }
                (
                    ZBlockPos::DiagRank1 {
                        diag_pos,
                        u_pos,
                        aux_pos,
                    },
                    ConeBlock::DiagPlusRank1 { diag, u },
                ) => {
                    // (z,z) block = −(diag(d) + uuᵀ) − reg, with the rank-1
                    // carried by the aux variable ξ: diagonal −dᵢ − reg, the
                    // coupling (z_i, ξ) = uᵢ, and (ξ, ξ) = +1. Its Schur
                    // complement is −diag(d) − reg − uuᵀ = −(W²) − reg.
                    for i in 0..d {
                        out[diag_pos[i]] = -diag[i] - reg;
                        out[u_pos[i]] = u[i];
                    }
                    out[*aux_pos] = 1.0;
                }
                (ZBlockPos::Dense { pos }, ConeBlock::DenseLower { dim: _, lower }) => {
                    // (z,z) block = −H − reg·I, H = W⊗ₛW dense. Lower triangle
                    // row-major; reg only on the diagonal (i == j).
                    let mut idx = 0;
                    for i in 0..d {
                        for j in 0..=i {
                            out[pos[idx]] = -lower[idx] - if i == j { reg } else { 0.0 };
                            idx += 1;
                        }
                    }
                }
                _ => unreachable!("cone block shape changed between build and update"),
            }
        }
    }

    /// Overwrite the `(y, y)` equality-multiplier diagonal with the adaptive
    /// regularization `-δ_c` for the current barrier parameter. Call once per
    /// iteration, after [`Self::update_blocks`], on the same `out` buffer.
    ///
    /// A no-op when there are no equality rows.
    pub(crate) fn update_eq_reg(&self, delta_c: f64, out: &mut [Number]) {
        for &p in &self.y_diag_pos {
            out[p] = -delta_c;
        }
    }

    /// Overwrite the `(x, x)` diagonal with `P + δ_w` for the current primal
    /// regularization. Call once per iteration, on the same `out` buffer as
    /// [`Self::update_blocks`] / [`Self::update_eq_reg`].
    pub(crate) fn update_primal_reg(&self, delta_w: f64, out: &mut [Number]) {
        for (&p, &base) in self.x_diag_pos.iter().zip(&self.x_diag_base) {
            out[p] = base + delta_w;
        }
    }
}

/// Adaptive equality-Jacobian regularization `δ_c(μ)`, mirroring the NLP
/// path's primal-dual perturbation handler (`δ_cd_val · μ^δ_cd_exp`, Ipopt
/// defaults `1e-8 · μ^0.25`).
///
/// Floored at `reg` so it never drops below the static value the LP/QP
/// suites already converge with — at `μ = tol = 1e-8` the μ-term equals
/// exactly `1e-8 · (1e-8)^0.25 = 1e-10 = reg`, so a problem that already
/// reaches the optimum sees the *same* regularization there; the only change
/// is *extra* regularization in the earlier, larger-μ iterations. That extra
/// damping keeps the duals of a rank-deficient equality system (gen/gen1's
/// redundant rows) bounded, so the primal residual `δ·‖dy‖` clears `tol`
/// instead of flooring at ~9e-5. Capped at `1e-2` to stay well-conditioned.
pub(crate) fn adaptive_eq_reg(mu: f64, reg: f64) -> f64 {
    const DELTA_CD_VAL: f64 = 1e-8;
    const DELTA_CD_EXP: f64 = 0.25;
    const DELTA_CD_MAX: f64 = 1e-2;
    (DELTA_CD_VAL * mu.max(0.0).powf(DELTA_CD_EXP))
        .max(reg)
        .min(DELTA_CD_MAX)
}

/// How each cone block enters the `(z,z)` position — diagonal (orthant),
/// diag-plus-rank-1 (SOC), or fully dense (PSD) — probed via `kkt_block` at
/// the cone identity.
fn block_shapes(cone: &CompositeCone) -> Vec<BlockShape> {
    cone.blocks()
        .iter()
        .map(|(_, k)| {
            let d = k.dim();
            let mut e = vec![0.0; d];
            k.identity(&mut e);
            match k.kkt_block(&e, &e) {
                ConeBlock::Diagonal(_) => BlockShape::Diagonal,
                ConeBlock::DiagPlusRank1 { .. } => BlockShape::DiagRank1,
                ConeBlock::DenseLower { .. } => BlockShape::Dense,
            }
        })
        .collect()
}

/// Whether every entry of every block of an iterate is finite.
///
/// A single non-finite entry means the iteration has broken down: there is no
/// point continuing, and — more importantly — no verdict but a failure is
/// honest, since the "solution" carries no information.
pub(crate) fn all_finite(blocks: &[&[f64]]) -> bool {
    blocks.iter().all(|b| b.iter().all(|v: &f64| v.is_finite()))
}

/// Demote a success verdict that is not backed by a usable solution (gh #222).
///
/// The last line of defence before a solution leaves either driver: reporting
/// `Optimal` is a *claim*, and a caller that checks the status — the documented
/// way to know an answer is usable — must never be handed `NaN` alongside it.
/// Whatever went wrong upstream, `NumericalFailure` is the honest verdict.
///
/// Deliberately a separate final pass rather than a fix at one breakdown site:
/// the guarantee wanted is about what comes *out*, so it belongs where the
/// result is assembled, and it then holds no matter which internal path
/// produced the iterate.
pub(crate) fn demote_unusable(status: QpStatus, x: &[f64], obj: f64) -> QpStatus {
    let claims_success = matches!(status, QpStatus::Optimal | QpStatus::OptimalInaccurate);
    if claims_success && !(all_finite(&[x]) && obj.is_finite()) {
        return QpStatus::NumericalFailure;
    }
    status
}

/// `‖v‖∞`, propagating `NaN` rather than swallowing it (gh #222).
///
/// The obvious `fold(0.0, |m, x| m.max(x.abs()))` is **wrong on a `NaN`
/// input**, and silently so: `f64::max` is defined to *ignore* `NaN`, so
/// `0.0f64.max(NaN) == 0.0` and the ∞-norm of an all-`NaN` vector comes back as
/// a perfect `0.0`.
///
/// Every convergence test in both drivers is a comparison of `inf_norm`-derived
/// residuals against `tol`, so that turned a fully diverged iterate into a
/// declaration of optimality. On the gh #222 instance the direct driver's
/// iterate went entirely non-finite at iteration 31 and the residuals it
/// computed from that iterate read `pinf = dinf = res = 0`, so `res < tol`
/// passed and the solve returned `Optimal` with `x = [NaN, NaN]`.
///
/// `NaN` short-circuits here so the norm is genuinely `NaN`; every `< tol`
/// test against it is then false, which is the correct answer.
pub(crate) fn inf_norm(v: &[f64]) -> f64 {
    let mut m = 0.0_f64;
    for &x in v {
        if x.is_nan() {
            return f64::NAN;
        }
        m = m.max(x.abs());
    }
    m
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Check the current iterate for a *verified* infeasibility certificate.
///
/// Returns `Some(PrimalInfeasible | DualInfeasible)` **only** when the
/// certificate's defining (in)equalities hold to `opts.infeas_tol`
/// relative to the certificate's own magnitude. Because the certificate
/// is checked, not assumed, a positive result is a genuine proof and a
/// false positive is impossible; an unverifiable iterate returns `None`
/// and the solve keeps going (ultimately `IterationLimit`).
///
/// This recovers HSDE's headline benefit — clean infeasible/unbounded
/// status instead of silently exhausting the iteration budget — without
/// the homogeneous embedding's full rewrite of the iteration. When the
/// problem is primal-infeasible the IPM's dual iterate `(y, z)` diverges
/// along a Farkas ray, so its normalization satisfies the primal
/// certificate; when the problem is unbounded the primal iterate `x`
/// diverges along a recession direction satisfying the dual certificate.
///
/// Certificates (for `min ½xᵀPx + cᵀx s.t. Ax = b, Gx ≤ h`):
/// - **Primal infeasible:** `(y, z ≥ 0)` with `Aᵀy + Gᵀz ≈ 0` and
///   `bᵀy + hᵀz < 0` (Farkas). `z ≥ 0` is maintained by the IPM.
/// - **Dual infeasible / unbounded:** direction `d` (= `x`) with
///   `Pd ≈ 0, Ad ≈ 0, −Gd ∈ K, cᵀd < 0` (orthant: `−Gd ∈ K ⟺ Gd ≤ 0`).
///
/// This orthant-exact entry point is the documented baseline that the
/// cone-aware variants ([`detect_infeasibility_cone`] for the symmetric
/// composite cone, `detect_infeasibility_nscone` for the non-symmetric
/// driver) generalize. Both production drivers now route through a
/// cone-aware path, so this plain version is retained for documentation
/// and as a contrast oracle in tests.
#[allow(dead_code)]
pub(crate) fn detect_infeasibility(
    prob: &QpProblem,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    opts: &QpOptions,
) -> Option<QpStatus> {
    // Default dual-cone test: componentwise `zᵢ ≥ −tol`, exact for the
    // nonnegative orthant (LP/QP) and the non-symmetric Farkas paths. The
    // cone-aware entry point is [`detect_infeasibility_cone`].
    //
    // Default primal-recession test: `−Gd ∈ R₊ᵐ`, i.e. `(Gd)ᵢ ≤ tol`
    // componentwise — exact for the orthant.
    detect_infeasibility_with(
        prob,
        x,
        y,
        z,
        opts,
        |z, tol| z.iter().all(|&zi| zi >= -tol),
        |gd, tol| gd.iter().all(|&v| v <= tol),
    )
}

/// Cone-aware variant of [`detect_infeasibility`]: validates **both**
/// certificates against the **actual** cone instead of componentwise.
///
/// - *Primal infeasibility* — the Farkas dual multiplier `z` must lie in the
///   dual cone `K*` (orthant: `z ≥ 0`; SOC: `z₀ ≥ ‖z₁‖`; PSD: `smat(z) ⪰ 0`).
/// - *Dual infeasibility / unboundedness* — for a cone constraint
///   `Gx ⪯_K h`, the recession direction `d` must satisfy `−Gd ∈ K`, not the
///   componentwise `Gd ≤ 0`. E.g. `−Gd = (0.1, 0.5)` passes componentwise but
///   is **not** in the SOC, so the componentwise test would emit a false
///   `DualInfeasible`.
///
/// The componentwise default ([`detect_infeasibility`]) is correct only for
/// the orthant. Every cone reaching `CompositeCone` is symmetric (self-dual:
/// orthant/SOC/PSD; exp/power route to `hsde_nonsym`), so `−Gd ∈ K` is tested
/// as `cone.in_dual_cone(−Gd)`.
pub(crate) fn detect_infeasibility_cone(
    prob: &QpProblem,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    opts: &QpOptions,
    cone: &CompositeCone,
) -> Option<QpStatus> {
    detect_infeasibility_with(
        prob,
        x,
        y,
        z,
        opts,
        |z, tol| cone.in_dual_cone(z, tol),
        |gd, tol| {
            // `−Gd ∈ K`; K self-dual here ⇒ test via `in_dual_cone`.
            let neg: Vec<f64> = gd.iter().map(|&v| -v).collect();
            cone.in_dual_cone(&neg, tol)
        },
    )
}

pub(crate) fn detect_infeasibility_with(
    prob: &QpProblem,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    opts: &QpOptions,
    dual_cone_ok: impl Fn(&[f64], f64) -> bool,
    primal_recession_ok: impl Fn(&[f64], f64) -> bool,
) -> Option<QpStatus> {
    let n = prob.n;
    // Certificate *value* threshold and cone-membership slack: a modest
    // tolerance (`infeas_tol`, 1e-7) is right for "is `bᵀy+hᵀz` meaningfully
    // negative" and "is `z` in the dual cone".
    let ctol = opts.infeas_tol;
    // Certificate *residual* tolerance: far tighter (`FARKAS_RESID_TOL`,
    // ~1e-10). A finite-precision Farkas pair `(y,z)` only proves
    // infeasibility in the limit `‖Aᵀy+Gᵀz‖ → 0`. A FEASIBLE problem still
    // admits an approximate certificate, but its residual cannot fall below a
    // floor `∝ 1/‖x*‖` (the bound `bᵀy+hᵀz ≥ -‖x*‖₁·‖Aᵀy+Gᵀz‖∞` means a
    // large-norm feasible point leaves only a small residual to "explain").
    // POWELL20 (`‖x*‖ ~ 1e7`) floors at `~7.5e-8` — which the loose `ctol`
    // (1e-7) wrongly accepted, declaring a feasible QP primal-infeasible at
    // iteration 2. A *genuine* certificate drives the residual to ~machine
    // precision (`~1e-15`). `FARKAS_RESID_TOL` sits ~5 orders above the latter
    // and ~3 below the former, so it rejects the spurious floor while still
    // accepting real certificates. (Symmetric reasoning applies to the
    // recession residuals `Px,Ax,Gx` in the dual-infeasibility test below.)
    let rtol = FARKAS_RESID_TOL;

    // --- Primal infeasibility (Farkas certificate) ---
    let dual_norm = inf_norm(y).max(inf_norm(z));
    if dual_norm > 0.0 {
        let mut resid = vec![0.0; n]; // Aᵀy + Gᵀz
        prob.at_mul(y, &mut resid);
        prob.gt_mul(z, &mut resid);
        let cert = dot(&prob.b, y) + dot(&prob.h, z); // bᵀy + hᵀz
        let z_ok = dual_cone_ok(z, ctol * dual_norm);
        if cert < -ctol * dual_norm && inf_norm(&resid) <= rtol * dual_norm && z_ok {
            return Some(QpStatus::PrimalInfeasible);
        }
    }

    // --- Dual infeasibility / unboundedness (recession direction d = x) ---
    let x_norm = inf_norm(x);
    if x_norm > 0.0 {
        let mut pd = vec![0.0; n];
        prob.p_mul(x, &mut pd);
        let mut ad = vec![0.0; prob.m_eq()];
        prob.a_mul(x, &mut ad);
        let mut gd = vec![0.0; prob.m_ineq()];
        prob.g_mul(x, &mut gd);
        let cd = dot(&prob.c, x);
        // Recession condition `−Gd ∈ K` (orthant ⇒ componentwise `Gd ≤ 0`;
        // SOC/PSD ⇒ true cone membership). Checked, not componentwise, so a
        // direction that merely has `Gd ≤ 0` but `−Gd ∉ K` is rejected.
        let gd_ok = primal_recession_ok(&gd, ctol * x_norm);
        // `d` is a recession direction of the *quadratic* iff the objective
        // stays downhill along it forever: `f(x+td) = f + t·cᵀd + ½t²·dᵀPd →
        // −∞`. Since a convex QP has `P ⪰ 0`, that requires **zero directional
        // curvature** `dᵀPd = 0` (any `dᵀPd > 0` makes the quadratic term
        // dominate, so the objective has a finite minimum along `d` and the
        // problem is bounded there) together with `cᵀd < 0`.
        //
        // The quantity to test is the *normalized* directional curvature
        // `dᵀPd/‖d‖²` — the curvature per unit length along `d`, an
        // eigenvalue-scale number that a diverging iterate (`‖d‖ → ∞`) cannot
        // inflate. Two earlier residual tests were both wrong on a mixed-scale
        // Hessian:
        //   * `‖Pd‖ ≤ rtol·‖d‖` collapses to `‖P‖ ≤ rtol` (‖d‖ cancels), so any
        //     strictly-convex QP with `‖P‖ < rtol` read as unbounded (gh #273).
        //   * `‖Pd‖ ≤ rtol·‖d‖·‖P‖` (gh #290) fixes the *uniform* small case but
        //     still fails when `P`'s eigenvalues span many orders: normalizing
        //     by the single global scale `‖P‖ = max|P|` cannot express
        //     `d ∈ null(P)`. For `P = diag(1e6, 1e-12)` the descent ray `d = e₁`
        //     has genuine per-unit curvature `dᵀPd/‖d‖² = 1e-12 > 0` (bounded,
        //     `f* = −5e11`), yet `‖Pd‖ = 1e-12 ≪ rtol·‖P‖ = 1e-16·1e6`, so it
        //     was falsely certified `DualInfeasible` — a wrong unboundedness
        //     certificate on a bounded problem. See gh #293.
        //
        // Testing `dᵀPd/‖d‖²` against an absolute floor separates the two
        // regimes cleanly: a *bounded* problem floors the normalized curvature
        // at its smallest real directional eigenvalue (`1e-12` here, `1e-16` for
        // the #273 `P = 1e-16` case), while a *genuine* recession drives it to
        // zero — exactly `0` for an LP or an axis-aligned null block, and, for a
        // singular `P` whose curved variable is pinned to a bound while the null
        // variable diverges, `~1e-140` and shrinking. `RECESSION_CURV_TOL` sits
        // far below every eigenvalue that must be rejected yet vastly above a
        // true recession's vanishing curvature. See gh #293 (P0/P1/P2).
        let curv = dot(x, &pd); // dᵀPd (pd = P·d)
        let d_norm_sq = dot(x, x); // ‖d‖² > 0 (guarded by x_norm > 0)
        let curv_ok = curv <= RECESSION_CURV_TOL * d_norm_sq;
        if cd < -ctol * x_norm && curv_ok && inf_norm(&ad) <= rtol * x_norm && gd_ok {
            return Some(QpStatus::DualInfeasible);
        }
    }

    None
}

#[cfg(test)]
mod adaptive_tau_tests {
    //! The Mehrotra tail of gh #417: `τ = clamp(1 − μ, tau, tau_max)`.
    use super::{QpOptions, TAU_CEIL, adaptive_tau};

    #[test]
    fn tau_rises_toward_one_as_mu_falls() {
        let opts = QpOptions::default();
        // Far out (large μ) the static τ still governs: no change to the
        // early iterations, which is where the damping earns its keep.
        assert_eq!(adaptive_tau(1.0, &opts), opts.tau);
        assert_eq!(adaptive_tau(0.5, &opts), opts.tau);
        // The rule engages once 1 − μ clears the floor, and is monotone.
        assert!(adaptive_tau(1e-2, &opts) > opts.tau);
        assert!(adaptive_tau(1e-6, &opts) > adaptive_tau(1e-2, &opts));
        // Always strictly inside (0, 1): a step landing exactly on the
        // boundary would divide by a zero `zᵢ` next iteration.
        for mu in [1e-9, 1e-14, 0.0] {
            let t = adaptive_tau(mu, &opts);
            assert!(t < 1.0 && t >= opts.tau, "τ({mu:e}) = {t}");
        }
        assert_eq!(adaptive_tau(0.0, &opts), TAU_CEIL);
    }

    #[test]
    fn tau_max_equal_to_tau_restores_the_static_rule() {
        let opts = QpOptions {
            tau_max: 0.95,
            ..QpOptions::default()
        };
        assert_eq!(opts.tau, 0.95);
        for mu in [10.0, 1.0, 1e-3, 1e-12, 0.0] {
            assert_eq!(adaptive_tau(mu, &opts), 0.95, "μ = {mu:e}");
        }
        // An intermediate ceiling caps the tail where the caller asks it to.
        let capped = QpOptions {
            tau_max: 0.999,
            ..QpOptions::default()
        };
        assert_eq!(adaptive_tau(1e-12, &capped), 0.999);
        // Below the ceiling the rule is untouched: τ = 1 − μ.
        assert!((adaptive_tau(0.005, &capped) - 0.995).abs() < 1e-15);
    }

    /// An inverted pair is not a panic and not a τ below the floor: the
    /// floor wins. (`f64::clamp` panics outright when `min > max`.)
    #[test]
    fn an_inverted_tau_pair_falls_back_to_the_floor() {
        let opts = QpOptions {
            tau: 0.99,
            tau_max: 0.5,
            ..QpOptions::default()
        };
        for mu in [1.0, 1e-3, 1e-12] {
            assert_eq!(adaptive_tau(mu, &opts), 0.99, "μ = {mu:e}");
        }
    }
}

#[cfg(test)]
mod detect_infeasibility_tests {
    //! H7 regression: the dual-infeasibility recession test must validate
    //! `−Gd ∈ K`, not componentwise `Gd ≤ 0`. These call the `pub(crate)`
    //! detectors directly with crafted recession directions.
    use super::{detect_infeasibility, detect_infeasibility_cone};
    use crate::QpOptions;
    use crate::cones::{CompositeCone, ConeSpec};
    use crate::qp::{QpProblem, QpStatus, Triplet};

    /// `min −x₀` with the single SOC row block `Gx ⪯_{SOC} h`,
    /// `G = [[−0.1], [−0.5]]`. Recession direction `d = (1)` gives
    /// `Gd = (−0.1, −0.5)`: componentwise `≤ 0` (the OLD test passes) but
    /// `−Gd = (0.1, 0.5)` has `0.1 < ‖0.5‖`, so `−Gd ∉ SOC` — the direction
    /// is NOT a genuine recession ray. The cone-aware detector must return
    /// `None`; the orthant default (wrongly) returns `DualInfeasible`,
    /// demonstrating the bug.
    fn soc_false_recession_problem() -> QpProblem {
        QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![-1.0], // cᵀd = −1 < 0
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, -0.1), // (Gd)₀ = −0.1
                Triplet::new(1, 0, -0.5), // (Gd)₁ = −0.5
            ],
            h: vec![0.0, 0.0],
            lb: vec![],
            ub: vec![],
        }
    }

    /// gh #273 — a strictly convex QP must never be certified unbounded just
    /// because its Hessian is numerically small.
    ///
    /// `min -x + x²/(2M)  s.t.  x ≥ 0` has the unique minimum `x* = M`,
    /// `f* = -M/2`, for every finite `M > 0`. The old recession test compared
    /// `‖Pd‖ ≤ rtol·‖d‖`; since `‖Pd‖ = ‖P‖·‖d‖` for a scalar `P`, `‖d‖`
    /// cancelled and the test reduced to `‖P‖ ≤ rtol`. So every `M ≥ 1/rtol`
    /// (i.e. `P ≤ 1e-10`) read as unbounded. The bound is now scaled by `‖P‖`,
    /// making it a genuine relative nullspace test.
    #[test]
    fn tiny_hessian_is_not_a_recession_direction() {
        let opts = QpOptions::default();
        let y: [f64; 0] = [];
        let z: [f64; 0] = [];
        // P far below FARKAS_RESID_TOL (1e-10) in every case.
        for p_val in [1e-10, 1e-12, 1e-16] {
            let prob = QpProblem {
                n: 1,
                p_lower: vec![Triplet::new(0, 0, p_val)],
                c: vec![-1.0],
                a: vec![],
                b: vec![],
                g: vec![],
                h: vec![],
                lb: vec![0.0],
                ub: vec![f64::INFINITY],
            };
            let x = [1.0]; // candidate recession direction
            assert_eq!(
                detect_infeasibility(&prob, &x, &y, &z, &opts),
                None,
                "P = {p_val:e} is strictly positive, so d = 1 is NOT a recession \
                 direction and the QP is bounded below; certifying unboundedness \
                 here returns a wrong answer for a problem with a finite optimum"
            );
        }
    }

    /// The complement of the test above: a genuinely singular `P` with the
    /// direction lying in its nullspace must still certify unboundedness, so
    /// the #273 fix introduces no false negative.
    ///
    /// `min ½x₀² - x₁  s.t.  x ≥ 0` with `P = diag(1, 0)`: `d = (0, 1)` has
    /// `Pd = 0` exactly and `cᵀd = -1 < 0`.
    #[test]
    fn singular_hessian_nullspace_direction_is_still_dual_infeasible() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0)], // P = diag(1, 0)
            c: vec![0.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![0.0, 0.0],
            ub: vec![f64::INFINITY, f64::INFINITY],
        };
        let opts = QpOptions::default();
        let x = [0.0, 1.0]; // in null(P)
        let y: [f64; 0] = [];
        let z: [f64; 0] = [];
        assert_eq!(
            detect_infeasibility(&prob, &x, &y, &z, &opts),
            Some(QpStatus::DualInfeasible),
            "d = (0,1) is exactly in null(P) with c'd < 0 — a genuine recession \
             ray that must still be detected"
        );
    }

    /// gh #293 — the mixed-scale regression the normalized-curvature test
    /// exists for. `P = diag(1e6, 1e-12)`, and the tiny-curvature descent ray
    /// `d = (0, 1)` has genuine curvature `dᵀPd = 1e-12 > 0`, so per-unit
    /// curvature `dᵀPd/‖d‖² = 1e-12` — a bounded direction (`f* = −5e11`), NOT a
    /// recession. The pre-#293 `‖Pd‖ ≤ rtol·‖d‖·max|P|` test read `1e-12 ≤
    /// 1e-16·1e6 = 1e-10` and falsely certified `DualInfeasible`; the
    /// normalized-curvature test rejects it because `1e-12 ≫ RECESSION_CURV_TOL`.
    #[test]
    fn mixed_scale_tiny_curvature_direction_is_not_a_recession() {
        let opts = QpOptions::default();
        let y: [f64; 0] = [];
        let z: [f64; 0] = [];
        // Vary the *small* eigenvalue across the whole "looks tiny relative to
        // ‖P‖ = 1e6" band. Every one has positive curvature along d, so none is
        // a recession ray; all must return None. The genuine null block (0.0)
        // is covered by `singular_hessian_nullspace_direction_is_still_dual…`.
        for small in [1e-8, 1e-12, 1e-16, 1e-19] {
            let prob = QpProblem {
                n: 2,
                p_lower: vec![Triplet::new(0, 0, 1e6), Triplet::new(1, 1, small)],
                c: vec![0.0, -1.0],
                a: vec![],
                b: vec![],
                g: vec![],
                h: vec![],
                lb: vec![0.0, 0.0],
                ub: vec![f64::INFINITY, f64::INFINITY],
            };
            let x = [0.0, 1.0]; // descent ray; dᵀPd = small > 0
            assert_eq!(
                detect_infeasibility(&prob, &x, &y, &z, &opts),
                None,
                "P = diag(1e6, {small:e}): d = (0,1) has positive curvature \
                 dᵀPd = {small:e} (bounded below), certifying it unbounded is a \
                 wrong answer regardless of how small that curvature is next to \
                 max|P| = 1e6"
            );
        }
    }

    /// An LP (`P` empty) must be unaffected: `dᵀPd` is exactly zero, so the
    /// normalized curvature is `0 ≤ RECESSION_CURV_TOL` and genuine LP
    /// unboundedness is still certified.
    #[test]
    fn empty_hessian_lp_unboundedness_unaffected() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![-1.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![0.0],
            ub: vec![f64::INFINITY],
        };
        let opts = QpOptions::default();
        let x = [1.0];
        let y: [f64; 0] = [];
        let z: [f64; 0] = [];
        assert_eq!(
            detect_infeasibility(&prob, &x, &y, &z, &opts),
            Some(QpStatus::DualInfeasible),
            "an LP with no Hessian is unbounded along d = 1; dᵀPd = 0 must \
             certify (normalized curvature 0 ≤ RECESSION_CURV_TOL)"
        );
    }

    #[test]
    fn soc_recession_not_in_cone_is_not_dual_infeasible() {
        let prob = soc_false_recession_problem();
        let opts = QpOptions::default();
        let x = [1.0]; // recession direction d
        let y: [f64; 0] = [];
        let z = [0.0, 0.0];

        // The bug: orthant/componentwise test accepts the bogus direction.
        let componentwise = detect_infeasibility(&prob, &x, &y, &z, &opts);
        assert_eq!(
            componentwise,
            Some(QpStatus::DualInfeasible),
            "componentwise test should (wrongly) accept −Gd=(0.1,0.5) as recession"
        );

        // The fix: cone-aware test rejects it (−Gd ∉ SOC).
        let cone = CompositeCone::from_specs(&[ConeSpec::SecondOrder(2)]);
        let cone_aware = detect_infeasibility_cone(&prob, &x, &y, &z, &opts, &cone);
        assert_eq!(
            cone_aware, None,
            "cone-aware test must reject −Gd=(0.1,0.5): not in SOC, so no \
             verified unboundedness certificate"
        );
    }

    /// A genuine SOC recession: `G = [[−1.0], [0.0]]`, `d = (1)` gives
    /// `Gd = (−1, 0)`, `−Gd = (1, 0)` with `1 ≥ ‖0‖` ⇒ `−Gd ∈ SOC`. The
    /// cone-aware detector must still report `DualInfeasible` (no false
    /// negative from the fix).
    #[test]
    fn soc_genuine_recession_still_dual_infeasible() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![-1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0), Triplet::new(1, 0, 0.0)],
            h: vec![0.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let opts = QpOptions::default();
        let x = [1.0];
        let y: [f64; 0] = [];
        let z = [0.0, 0.0];
        let cone = CompositeCone::from_specs(&[ConeSpec::SecondOrder(2)]);
        assert_eq!(
            detect_infeasibility_cone(&prob, &x, &y, &z, &opts, &cone),
            Some(QpStatus::DualInfeasible),
            "−Gd=(1,0) IS in the SOC ⇒ genuine recession ray"
        );
    }

    /// Orthant LP unboundedness still detected by the cone-aware path
    /// (Nonneg cone), confirming the closure is consistent with the old
    /// componentwise behavior for the orthant.
    #[test]
    fn orthant_unbounded_lp_detected_both_paths() {
        // min −x₀ s.t. −x₀ ≤ 0 (x₀ ≥ 0). d=(1): Gd=(−1) ≤ 0, −Gd=(1) ≥ 0.
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![-1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0)],
            h: vec![0.0],
            lb: vec![],
            ub: vec![],
        };
        let opts = QpOptions::default();
        let x = [1.0];
        let y: [f64; 0] = [];
        let z = [0.0];
        assert_eq!(
            detect_infeasibility(&prob, &x, &y, &z, &opts),
            Some(QpStatus::DualInfeasible)
        );
        let cone = CompositeCone::from_specs(&[ConeSpec::Nonneg(1)]);
        assert_eq!(
            detect_infeasibility_cone(&prob, &x, &y, &z, &opts, &cone),
            Some(QpStatus::DualInfeasible)
        );
    }

    /// POWELL20 regression: a Farkas pair `(y,z)` whose certificate *value*
    /// is strongly negative (`hᵀz = −1`) and whose `z` is in the dual cone,
    /// but whose residual `‖Gᵀz‖ = 7.5e-8` sits in the danger zone *between*
    /// `FARKAS_RESID_TOL` (1e-10) and `infeas_tol` (1e-7) — exactly the
    /// spurious near-certificate a feasible large-`‖x*‖` QP (POWELL20)
    /// produces. The OLD code (residual bound = `infeas_tol·dual_norm`)
    /// accepted it and declared the feasible problem primal-infeasible; the
    /// tightened residual bound must reject it (`None`).
    #[test]
    fn spurious_farkas_with_residual_floor_is_not_infeasible() {
        // n=1 inequality-only LP. z=[1] ⇒ dual_norm=1, cert=hᵀz=−1,
        // resid=Gᵀz=[7.5e-8] (the POWELL20 floor).
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![0.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 7.5e-8)],
            h: vec![-1.0],
            lb: vec![],
            ub: vec![],
        };
        let opts = QpOptions::default();
        let x = [0.0]; // no recession direction ⇒ dual-infeasibility branch inert
        let y: [f64; 0] = [];
        let z = [1.0];
        assert_eq!(
            detect_infeasibility(&prob, &x, &y, &z, &opts),
            None,
            "residual 7.5e-8 (between FARKAS_RESID_TOL and infeas_tol) is a \
             feasibility floor, not a certificate — must not report infeasible"
        );

        // A genuine, machine-tight certificate (residual 1e-12 ≪ 1e-10) on the
        // same structure must still be detected — the tightening only rejects
        // the floor, not real certificates.
        let tight = QpProblem {
            g: vec![Triplet::new(0, 0, 1e-12)],
            ..prob
        };
        assert_eq!(
            detect_infeasibility(&tight, &x, &y, &z, &opts),
            Some(QpStatus::PrimalInfeasible),
            "residual 1e-12 ≪ FARKAS_RESID_TOL is a genuine Farkas certificate"
        );
    }
}

#[cfg(test)]
mod non_finite_guard_tests {
    //! gh #222: a success verdict must never accompany an unusable solution.
    use super::{all_finite, demote_unusable, inf_norm};
    use crate::qp::QpStatus;

    #[test]
    fn inf_norm_propagates_nan_instead_of_swallowing_it() {
        // The bug. `f64::max` is specified to IGNORE NaN, so the natural
        // `fold(0.0, |m, x| m.max(x.abs()))` reports the ∞-norm of an all-NaN
        // vector as a perfect 0.0. Every convergence test compares such a norm
        // against `tol`, so that made a fully diverged iterate read as
        // converged — the direct driver returned `Optimal` with `x = [NaN,NaN]`.
        assert!(
            0.0_f64.max(f64::NAN) == 0.0,
            "premise: f64::max ignores NaN"
        );

        assert!(inf_norm(&[f64::NAN, f64::NAN]).is_nan());
        assert!(inf_norm(&[1.0, f64::NAN, 2.0]).is_nan());
        // NaN anywhere wins, including after a larger finite entry (a fold that
        // let `max` swallow it would return 5.0 here).
        assert!(inf_norm(&[5.0, f64::NAN]).is_nan());
        // And the convergence test then rejects it, which is the point: the
        // drivers all decide by comparing such a norm against `tol`.
        let converged = |residual: f64| residual < 1e-8;
        assert!(!converged(inf_norm(&[f64::NAN])));

        // Ordinary inputs are unchanged, infinities included.
        assert_eq!(inf_norm(&[]), 0.0);
        assert_eq!(inf_norm(&[-3.0, 2.0]), 3.0);
        assert_eq!(inf_norm(&[f64::INFINITY]), f64::INFINITY);
        assert!(!converged(inf_norm(&[f64::INFINITY])));
    }

    #[test]
    fn all_finite_spots_a_single_bad_entry_in_any_block() {
        let good = [1.0, 2.0];
        let nan = [1.0, f64::NAN];
        let inf = [f64::INFINITY];
        assert!(all_finite(&[&good, &good]));
        assert!(!all_finite(&[&good, &nan]));
        assert!(!all_finite(&[&inf, &good]));
        assert!(all_finite(&[]));
    }

    #[test]
    fn success_verdicts_are_demoted_when_the_solution_is_unusable() {
        let bad = [f64::NAN, 1.0];
        let good = [1.0, 2.0];
        for claim in [QpStatus::Optimal, QpStatus::OptimalInaccurate] {
            assert_eq!(
                demote_unusable(claim, &bad, 1.0),
                QpStatus::NumericalFailure,
                "{claim:?} with a NaN x must not survive"
            );
            assert_eq!(
                demote_unusable(claim, &good, f64::NAN),
                QpStatus::NumericalFailure,
                "{claim:?} with a NaN objective must not survive"
            );
            // A usable solution is left alone.
            assert_eq!(demote_unusable(claim, &good, 1.0), claim);
        }
        // Failure verdicts are reported as-is; the guard only demotes, never
        // promotes, so it cannot manufacture a success.
        for keep in [
            QpStatus::NumericalFailure,
            QpStatus::IterationLimit,
            QpStatus::PrimalInfeasible,
            QpStatus::DualInfeasible,
        ] {
            assert_eq!(demote_unusable(keep, &bad, f64::NAN), keep);
            assert_eq!(demote_unusable(keep, &good, 1.0), keep);
        }
    }
}

#[cfg(test)]
mod forward_error_operator_tests {
    //! gh #880: the arithmetic corners of [`super::barrier_ratio`], which no
    //! integration test in `tests/issue880_coupled_sigma_forward_error.rs`
    //! reaches. A converged interior point stops at a slack of order `μ`, so
    //! the non-positive-slack branch is defensive by construction — and a
    //! defensive branch with no test is a branch that has never run.
    //!
    //! What it defends is stated in full on `barrier_ratio`: this arm's
    //! right-hand side is `−(Px + c)`, so a `Σ` that reads too *small*
    //! inflates `‖Δ‖` and rejects a correct answer. Every case below is
    //! therefore about which unusable input maps to `None` (decline, accept)
    //! rather than to `0` (declare an active bound free, reject).

    use super::barrier_ratio;

    /// The ordinary case: an interior slack, `Σ = z/s` verbatim.
    #[test]
    fn an_interior_slack_is_the_plain_ratio() {
        assert_eq!(barrier_ratio(2.0, 8.0), Some(0.25));
    }

    /// A tight slack is *stiff*, and nothing clamps it. This is the case the
    /// guard depends on at an active bound.
    #[test]
    fn a_tight_slack_is_stiff_and_unclamped() {
        assert_eq!(barrier_ratio(1.0, 1e-12), Some(1e12));
    }

    /// A zero slack — an exactly active bound — is `None`, not `0`. `0` there
    /// says "this bound is not holding anything", which is the reverse of the
    /// truth and rejects correct answers.
    #[test]
    fn an_exactly_active_bound_declines_rather_than_reading_as_free() {
        assert_eq!(barrier_ratio(1.0, 0.0), None);
    }

    /// A slack driven *negative* by rounding is the same case, not a different
    /// one: it is an active bound the arithmetic overshot.
    #[test]
    fn a_negative_slack_declines_too() {
        assert_eq!(barrier_ratio(1.0, -1e-30), None);
        assert_eq!(barrier_ratio(1.0, f64::NAN), None);
    }

    /// A slack so small the ratio overflows carries no stiffness *number*,
    /// only the knowledge that it is enormous — which is `None`, because an
    /// infinity in the operator turns the CG residual into a `NaN` and the
    /// verdict into a coin flip.
    #[test]
    fn an_overflowing_ratio_declines() {
        assert_eq!(barrier_ratio(1e300, f64::MIN_POSITIVE), None);
    }

    /// A junk or wrong-signed multiplier is no evidence about stiffness, and
    /// no evidence declines. `0` here would again read as "free".
    #[test]
    fn a_junk_multiplier_declines() {
        assert_eq!(barrier_ratio(f64::NAN, 1.0), None);
        assert_eq!(barrier_ratio(-1.0, 1.0), None);
        assert_eq!(barrier_ratio(f64::INFINITY, 1.0), None);
    }

    /// A genuinely zero multiplier on an interior bound is *not* junk: the
    /// bound holds nothing and contributes nothing, and `Some(0.0)` is the
    /// correct answer rather than a decline.
    #[test]
    fn an_inactive_bound_contributes_zero() {
        assert_eq!(barrier_ratio(0.0, 10.0), Some(0.0));
    }

    // -----------------------------------------------------------------------
    // `sigma_forward_error_is_small` called directly.
    //
    // Three parts of it are invisible from outside the crate, because the `σ`
    // cascade's later candidates are also correct on the fixtures that reach
    // it: deleting `Σ_bnd`, deleting `Gᵀ Σ_row G`, or deleting the `‖x‖` scale
    // changes the *verdict* without changing the *answer* the caller finally
    // gets, and removing the `m_eq` early return does the same. Measured, not
    // assumed: each of those four mutations leaves every integration test in
    // `tests/issue880_coupled_sigma_forward_error.rs` green. So the arm is
    // called here directly, on hand-built points whose true optimum is known
    // in closed form.
    // -----------------------------------------------------------------------

    use super::{QpProblem, QpSolution, QpStatus, sigma_forward_error_is_small};
    use crate::Triplet;

    /// A point, its multipliers, and nothing the solver had to produce.
    fn at(
        prob: &QpProblem,
        x: Vec<f64>,
        z_lb: Vec<f64>,
        z_ub: Vec<f64>,
        z: Vec<f64>,
    ) -> QpSolution {
        QpSolution {
            status: QpStatus::Optimal,
            y: vec![0.0; prob.b.len()],
            x,
            z,
            z_lb,
            z_ub,
            obj: 0.0,
            iters: 0,
            iterates: vec![],
        }
    }

    /// `min ½x² − 3x` subject to `x ≤ 1`, expressed as a bound. The minimiser
    /// is the bound, `x = 1`, held there by `z_ub = 2`.
    fn active_bound_qp() -> QpProblem {
        QpProblem {
            n: 1,
            p_lower: vec![Triplet::new(0, 0, 1.0)],
            c: vec![-3.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![f64::NEG_INFINITY],
            ub: vec![1.0],
        }
    }

    /// **`Σ_bnd` is load-bearing.** At `x = 1` the right-hand side `−(Px + c)`
    /// is `2` — the multiplier, in full — and the only thing that makes `Δ`
    /// small is the bound's own stiffness `z/s`. Delete `Σ_bnd` from the
    /// operator and `Δ = 2/P = 2`, so a correct answer is rejected.
    #[test]
    fn an_active_bound_is_held_down_by_its_own_sigma() {
        let prob = active_bound_qp();
        // A converged interior point sits `μ/z` short of the bound.
        let (mu, z) = (1e-9, 2.0);
        let sol = at(&prob, vec![1.0 - mu / z], vec![0.0], vec![z], vec![]);
        assert!(
            sigma_forward_error_is_small(&prob, &sol, 1e-6),
            "the exact constrained minimiser was rejected; without Σ_bnd the \
             estimate is the whole multiplier, ‖Δ‖ = 2"
        );
    }

    /// The same model with the bound written as an inequality **row**, which
    /// is the shape `Σ_row` covers and the shape the cascade actually hands
    /// this arm once bounds are expanded.
    #[test]
    fn an_active_row_is_held_down_by_its_own_sigma() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![Triplet::new(0, 0, 1.0)],
            c: vec![-3.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0)],
            h: vec![1.0],
            lb: vec![f64::NEG_INFINITY],
            ub: vec![f64::INFINITY],
        };
        let (mu, z) = (1e-9, 2.0);
        let sol = at(&prob, vec![1.0 - mu / z], vec![0.0], vec![0.0], vec![z]);
        assert!(
            sigma_forward_error_is_small(&prob, &sol, 1e-6),
            "the exact constrained minimiser was rejected; without Gᵀ Σ_row G \
             the estimate is the whole multiplier, ‖Δ‖ = 2"
        );
    }

    /// The same point with the bound's multiplier zeroed — the shape a
    /// dropped `Σ` produces — must be **rejected**, so the two tests above are
    /// pinning the operator and not merely the tolerance.
    #[test]
    fn the_same_point_without_the_stiffness_is_rejected() {
        let prob = active_bound_qp();
        let sol = at(&prob, vec![1.0 - 5e-10], vec![0.0], vec![0.0], vec![]);
        assert!(
            !sigma_forward_error_is_small(&prob, &sol, 1e-6),
            "with no stiffness anywhere the estimate is ‖Δ‖ = 2 and this must \
             reject; if it accepts, the arm is inert rather than passing"
        );
    }

    /// **The `‖x‖` scale is load-bearing.** `min ½(x − 10⁶)²` solved to a
    /// *relative* `1e-9` leaves `‖Δ‖ = 1e-3`, three orders above a bare `cut`
    /// of `1e-6` and three below `cut·‖x‖`. The verdict has to be relative or
    /// every large-`x` model is rejected for being large.
    #[test]
    fn the_verdict_is_relative_to_x() {
        let t = 1e6;
        let prob = QpProblem {
            n: 1,
            p_lower: vec![Triplet::new(0, 0, 1.0)],
            c: vec![-t],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![f64::NEG_INFINITY],
            ub: vec![f64::INFINITY],
        };
        let sol = at(&prob, vec![t - 1e-3], vec![0.0], vec![0.0], vec![]);
        assert!(
            sigma_forward_error_is_small(&prob, &sol, 1e-6),
            "x = 1e6 − 1e-3 is right to 1e-9 relative and was rejected"
        );
        // …and the same absolute error at x ≈ 1 is genuinely a failure.
        let small = QpProblem {
            c: vec![-1.0],
            ..prob.clone()
        };
        let sol = at(&small, vec![1.0 - 1e-3], vec![0.0], vec![0.0], vec![]);
        assert!(
            !sigma_forward_error_is_small(&small, &sol, 1e-6),
            "the scale must not swallow a genuine 1e-3 error at x ≈ 1"
        );
    }

    /// **The operator is applied afresh each CG iteration.** `p_mul` is
    /// `y += P v`, not `y = P v`, and `mp` is reused across iterations, so
    /// omitting the zeroing makes the operator accumulate — which corrupts the
    /// residual update from the second iteration on and *under*-estimates
    /// `‖Δ‖`, the accepting direction. It takes four coupled variables to see:
    /// at `n ≤ 2` conjugate gradients finishes before the pollution can reach
    /// the `x` update, which is why no fixture in
    /// `tests/issue880_coupled_sigma_forward_error.rs` catches this.
    ///
    /// The point below sits `6.2e-7` from the minimiser of a well-conditioned
    /// (`cond ≈ 16`) integer Hessian; correct CG reproduces the direct solve to
    /// the last digit, the accumulating one reads `2.8e-8`, a 22× under-count,
    /// and a `cut` of `1e-7` lies between them.
    #[test]
    fn the_operator_does_not_accumulate_across_cg_iterations() {
        // Lower triangle of `MᵀM + I`, so symmetric positive definite by
        // construction rather than by a spectral assertion.
        let p_lower = vec![
            Triplet::new(0, 0, 27.0),
            Triplet::new(1, 0, 16.0),
            Triplet::new(1, 1, 18.0),
            Triplet::new(2, 0, 7.0),
            Triplet::new(2, 1, -3.0),
            Triplet::new(2, 2, 24.0),
            Triplet::new(3, 0, -19.0),
            Triplet::new(3, 1, -11.0),
            Triplet::new(3, 2, -12.0),
            Triplet::new(3, 3, 24.0),
        ];
        // At `x = 0` the right-hand side is exactly `−c`, so the step is
        // `P⁻¹ (2, −1, 3, 3)ᵀ · 1e-6` and nothing about the point is implicit.
        let prob = QpProblem {
            n: 4,
            p_lower,
            c: vec![2e-6, -1e-6, 3e-6, 3e-6],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![f64::NEG_INFINITY; 4],
            ub: vec![f64::INFINITY; 4],
        };
        let sol = at(&prob, vec![0.0; 4], vec![0.0; 4], vec![0.0; 4], vec![]);
        assert!(
            !sigma_forward_error_is_small(&prob, &sol, 1e-7),
            "‖Δ‖ is 6.2e-7 here and must be rejected against a cut of 1e-7; an \
             accumulating operator reads 2.8e-8 and accepts"
        );
        // …and the same point is accepted one order the other side of it, so
        // the assertion above is about the magnitude and not about the arm
        // rejecting everything.
        assert!(sigma_forward_error_is_small(&prob, &sol, 1e-5));
    }

    /// **The `m_eq > 0` decline.** `A` restricts motion exactly rather than
    /// through a diagonal, so this arm's operator is the wrong one under an
    /// equality and would reject points that are fine: at the minimiser of
    /// `½x₀² + ½x₁²` on `x₀ + x₁ = 2` the right-hand side `−(Px + c)` is
    /// `(−1, −1)`, entirely carried by the equality multiplier that the
    /// operator has no term for, so `‖Δ‖ = 1`. The arm declines instead.
    #[test]
    fn an_equality_row_makes_the_arm_decline() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![2.0],
            g: vec![],
            h: vec![],
            lb: vec![f64::NEG_INFINITY; 2],
            ub: vec![f64::INFINITY; 2],
        };
        let sol = at(&prob, vec![1.0, 1.0], vec![0.0; 2], vec![0.0; 2], vec![]);
        assert!(
            sigma_forward_error_is_small(&prob, &sol, 1e-6),
            "the exact minimiser under an equality row was rejected; the \
             m_eq early return is what keeps this arm out of a system it does \
             not model"
        );
    }
}

#[cfg(test)]
mod false_optimum_metric_tests {
    //! gh #414: the measurement the false-optimum repair rests on, and the
    //! floor it guarantees when the repair cannot succeed.

    use super::{
        FALSE_OPTIMUM_REL_TOL, QpOptions, SparseSymLinearSolverInterface,
        cost_normalized_hsde_solve, equilibrated_kkt_rel, expand_bounds, hsde_cost_scale,
        normalized_optimum_is_genuine, normalized_optimum_is_genuine_relative, optimum_is_genuine,
        solve_qp_ipm_unscaled, split_bound_duals, verify_or_repair_optimum,
    };
    use crate::cones::CompositeCone;
    use crate::qp::{QpProblem, QpStatus, Triplet};
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    /// The reported instance: a strictly convex box- and inequality-constrained
    /// QP stated in variables scaled `10^-6‥10^6` (`cond(P) ~ 1e24`, `cond = 10`
    /// after `z = x/s`). True optimum `-3.9585018079`.
    fn illscaled_qp() -> QpProblem {
        QpProblem {
            n: 3,
            p_lower: vec![
                Triplet::new(0, 0, 8395050448209.196),
                Triplet::new(1, 0, -1902145.0448367258),
                Triplet::new(1, 1, 3.251598903330246),
                Triplet::new(2, 0, 2.8600351667480353),
                Triplet::new(2, 1, 1.1387032854093064e-07),
                Triplet::new(2, 2, 2.5156283086289338e-12),
            ],
            c: vec![
                -2410857.501637979,
                -0.47196110542608866,
                1.9297552321865365e-06,
            ],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, -1363466.266639565),
                Triplet::new(0, 1, -0.34926083632221316),
                Triplet::new(0, 2, -3.621387263107342e-07),
            ],
            h: vec![2.455293068675696],
            lb: vec![
                -1.1096628522428714e-05,
                -10.254262623099589,
                -9953548.897040607,
            ],
            ub: vec![8.903371477571285e-06, 9.745737376900411, 10046451.102959393],
        }
    }

    /// The point the cost-normalized embedding certifies on [`illscaled_qp`] —
    /// the one the issue reports: `obj = 67.1341`, its own `kkt_error = 8.28e3`,
    /// under `status = Optimal`.
    ///
    /// Reached through [`cost_normalized_hsde_solve`] rather than
    /// [`super::solve_qp_ipm`] because the `σ` guard now rejects it before any
    /// public entry point can return it (gh #414 reopened) — which is the fix,
    /// and which would otherwise leave the two properties below with no subject
    /// to measure. This is the same arithmetic `solve_qp_core` runs, minus the
    /// verdict check the tests are about.
    fn escaped_sigma_optimum(prob: &QpProblem, opts: &QpOptions) -> super::QpSolution {
        let sigma = hsde_cost_scale(prob, opts.tol);
        assert!(
            sigma > 1.0,
            "premise: this instance triggers cost normalization"
        );
        let (expanded, bound_rows) = expand_bounds(prob);
        let cone = CompositeCone::single_nonneg(expanded.m_ineq());
        let scaled = expanded.scaled_objective(1.0 / sigma);
        let inner = QpOptions {
            obj_constant: opts.obj_constant / sigma,
            ..*opts
        };
        let sol = cost_normalized_hsde_solve(&scaled, &cone, &inner, sigma, &mut backend, None);
        split_bound_duals(prob, &bound_rows, sol)
    }

    /// Why the metric had to change, stated as a measurement rather than an
    /// argument: on the *same point*, the unscaled **relative** test (gh #324)
    /// sees nothing wrong and the equilibrated one sees an `O(100)` violation.
    ///
    /// This is the load-bearing claim of the fix. If a future change to the
    /// normalizers makes the unscaled relative test able to catch this on its
    /// own, this test fails loudly rather than leaving the extra machinery
    /// unexplained.
    ///
    /// The gh #414-reopened gate does not make it redundant, and the two
    /// assertions below say so as a measurement: the gate rejects this point
    /// through the **absolute** arm (`8.3e3 > tol` at a gradient scale that
    /// forbids a relative test), while the relative formula it wraps still
    /// cannot see the violation. The equilibrated metric remains the only thing
    /// that can, and it is what guards the non-`σ` HSDE door, which no gate
    /// covers.
    #[test]
    fn the_false_optimum_is_invisible_unscaled_and_obvious_equilibrated() {
        let prob = illscaled_qp();
        let opts = QpOptions::default();
        // The point the cost-normalized embedding certifies — what `solve_qp_ipm`
        // used to return before the repair, and before the gate.
        let bad = escaped_sigma_optimum(&prob, &opts);
        assert_eq!(
            bad.status,
            QpStatus::Optimal,
            "premise: the embedding certifies this point"
        );
        let abs = bad.kkt_residuals(&prob).kkt_error();
        assert!(
            abs > 1e3,
            "premise: the certified point's own KKT error is huge (got {abs:.3e}, \
             the issue reports 8.3e3)"
        );

        // The gh #324 test normalizes by *global* ∞-norms in the original
        // coordinates, where the badly-scaled column inflates ‖Px‖ enough to
        // hide the violation — it passes this point.
        assert!(
            normalized_optimum_is_genuine_relative(&prob, &bad),
            "premise: the unscaled relative test cannot see this failure"
        );
        // The gh #414-reopened gate rejects it anyway, on the absolute arm.
        assert!(
            !normalized_optimum_is_genuine(
                &prob,
                &CompositeCone::single_nonneg(prob.m_ineq()),
                &bad,
                opts.tol
            ),
            "the gated test must reject a point whose own KKT error is {abs:.3e}"
        );

        // Measured in the equilibrated metric the same point is plainly not a
        // KKT point, by orders of magnitude either side of the cut.
        let rel = equilibrated_kkt_rel(&prob, &bad, 0.0);
        assert!(
            rel > 1.0,
            "equilibrated relative KKT should be O(1) or worse, got {rel:.3e}"
        );
        assert!(!optimum_is_genuine(&prob, &bad, opts.tol, 0.0));

        // And the repaired solve — the same problem, the real optimum — sits far
        // below the cut, so the two are separated with room to spare.
        let good = super::solve_qp_ipm(&prob, &opts, backend);
        assert_eq!(good.status, QpStatus::Optimal);
        let good_rel = equilibrated_kkt_rel(&prob, &good, 0.0);
        assert!(
            good_rel < 1e-6,
            "the true optimum must be far inside the cut, got {good_rel:.3e}"
        );
        assert!(
            good_rel < FALSE_OPTIMUM_REL_TOL / 100.0 && rel > FALSE_OPTIMUM_REL_TOL * 100.0,
            "cut {FALSE_OPTIMUM_REL_TOL:.0e} must sit with margin between \
             {good_rel:.3e} and {rel:.3e}"
        );
    }

    /// The floor: when the equilibrated re-solve cannot certify an optimum
    /// either, the verdict is demoted — never handed back as a success.
    ///
    /// Forced deterministically by capping the retry at a single iteration, so
    /// the repair provably cannot converge. What must survive is the guarantee,
    /// not the repair: no `Optimal`, and no `OptimalInaccurate` either, since
    /// that still reports `ok` / exit 0 through the CLI.
    #[test]
    fn an_unrepairable_false_optimum_is_demoted_never_reported_optimal() {
        let prob = illscaled_qp();
        let bad = escaped_sigma_optimum(&prob, &QpOptions::default());
        assert_eq!(bad.status, QpStatus::Optimal, "premise");

        let starved = QpOptions {
            max_iter: 1,
            ..QpOptions::default()
        };
        let mut mb = backend;
        let out = verify_or_repair_optimum(&prob, &starved, bad, &mut mb);
        assert_eq!(
            out.status,
            QpStatus::NumericalFailure,
            "an uncertifiable point must not keep a success status (got {:?})",
            out.status
        );
    }

    /// The check must not tax a solve that is simply correct: a well-scaled QP
    /// reaches absolute `tol` accuracy, short-circuits before any equilibration,
    /// and is returned untouched.
    #[test]
    fn a_well_scaled_optimum_short_circuits_without_equilibrating() {
        // min ½‖x‖² − xᵀ[1,2]  s.t.  −5 ≤ x ≤ 5 — optimum x* = [1, 2].
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![-1.0, -2.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![-5.0, -5.0],
            ub: vec![5.0, 5.0],
        };
        let opts = QpOptions::default();
        let sol = solve_qp_ipm_unscaled(&prob, &opts, backend, None);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!(
            sol.kkt_residuals(&prob).kkt_error() <= opts.tol,
            "premise: this solve is absolutely tol-accurate"
        );
        assert!(optimum_is_genuine(&prob, &sol, opts.tol, 0.0));

        let x = sol.x.clone();
        let mut mb = backend;
        let out = verify_or_repair_optimum(&prob, &opts, sol, &mut mb);
        assert_eq!(out.status, QpStatus::Optimal);
        assert_eq!(out.x, x, "a genuine optimum must be returned unchanged");
    }
}

#[cfg(test)]
mod objective_constant_metric_tests {
    //! gh #712: the second place the objective magnitude normalizes a duality
    //! gap, and the noise floor that keeps the correction from over-rejecting.

    use super::{
        FALSE_OPTIMUM_REL_TOL, QpOptions, optimum_is_genuine, resolvable_complementarity,
        slack_is_resolvable,
    };
    use crate::qp::{QpProblem, QpSolution, QpStatus, Triplet};

    /// `min (x − a)²` — a one-variable least squares, `a = 5e5`, over the box
    /// `0 ≤ x ≤ 1e6`. As a [`QpProblem`] that is `½·2x² − 2a·x`, and the `a² =
    /// 2.5e11` the caller's objective also carries lives nowhere in the data:
    /// it is exactly what [`QpOptions::obj_constant`] is for. The shape of
    /// `scaled_feasible_a`, in one variable.
    fn least_squares_with_constant(a: f64) -> QpProblem {
        QpProblem {
            n: 1,
            p_lower: vec![Triplet::new(0, 0, 2.0)],
            c: vec![-2.0 * a],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![0.0],
            ub: vec![2.0 * a],
        }
    }

    fn point(x: f64, z_lb: f64) -> QpSolution {
        QpSolution {
            status: QpStatus::Optimal,
            x: vec![x],
            y: vec![],
            z: vec![],
            z_lb: vec![z_lb],
            z_ub: vec![0.0],
            obj: 0.0,
            iters: 0,
            iterates: Vec::new(),
        }
    }

    /// The defect, in the small: a point carrying a multiplier on a bound it is
    /// `5e5` away from is not a KKT point, and the only reason the equilibrated
    /// test certified it is that it divided that `2.3e3` of complementarity by
    /// an objective magnitude which is the *constant* `QpProblem` never models.
    ///
    /// Told the constant — the same correction gh #696 made to HSDE's `scale_g`
    /// — the normalizer measures the objective the caller actually reads and the
    /// point is refused. `0.0`, the default, reproduces the old reading exactly,
    /// which is what keeps every caller that has no constant bit-for-bit
    /// unchanged.
    #[test]
    fn the_objective_constant_reaches_the_equilibrated_gap_normalizer() {
        let a = 5.0e5;
        let prob = least_squares_with_constant(a);
        // `z_lb` is the whole defect: at `x = a` stationarity would want it at
        // `0`, and the bound is `5e5` away, so the product is a violation.
        let bad = point(a, 4.566e-3);
        let tol = QpOptions::default().tol;
        assert!(
            bad.kkt_residuals(&prob).kkt_error() > 1e3,
            "premise: the point's own absolute KKT error is huge ({:.3e})",
            bad.kkt_residuals(&prob).kkt_error()
        );

        assert!(
            optimum_is_genuine(&prob, &bad, tol, 0.0),
            "premise (the gh #712 defect): normalized by the displaced objective \
             magnitude `a² = {:.1e}`, this point reads genuine",
            a * a
        );
        assert!(
            !optimum_is_genuine(&prob, &bad, tol, a * a),
            "told the objective constant, the same point must be refused"
        );
    }

    /// And the correction must not reject a point that *is* one: the same
    /// problem, the same constant, with the spurious multiplier gone.
    #[test]
    fn the_correction_still_certifies_a_genuine_optimum() {
        let a = 5.0e5;
        let prob = least_squares_with_constant(a);
        let good = point(a, 0.0);
        let tol = QpOptions::default().tol;
        assert!(optimum_is_genuine(&prob, &good, tol, a * a));
    }

    /// The floor that makes the correction survivable, measured on the two
    /// geometries that forced it (both are a variable pinned inside a box far
    /// tighter than the variable's own magnitude, with large multipliers on
    /// both sides — the difference is entirely whether the slack is a number
    /// double precision can hold).
    ///
    /// `feasible_x0_wide_scale`: presolve derives a box `1.4e-8` wide around
    /// `|x| ≈ 7.1e5`, the converged iterate sits `7e-9` inside it — `46` ulps —
    /// and against a multiplier of `1.8e7` that is a `0.13` product on a point
    /// matching the NLP oracle to 13 digits. Nothing about it is a violation:
    /// the slack is not distinguishable from zero.
    #[test]
    fn a_slack_under_its_own_rounding_quantum_is_not_a_violation() {
        let x = 7.071044e5;
        let (lo, hi) = (7.0e-9, 7.2e-9);
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![0.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![x - lo],
            ub: vec![x + hi],
        };
        let sol = point(x, 1.847e7);
        let sol = QpSolution {
            z_ub: vec![1.847e7],
            ..sol
        };
        assert!(
            sol.kkt_residuals(&prob).complementarity > 0.1,
            "premise: measured absolutely these products are O(0.1)"
        );
        assert_eq!(
            resolvable_complementarity(&prob, &sol),
            0.0,
            "a slack of {lo:.1e}/{hi:.1e} at |x| = {x:.3e} is under the quantum \
             of the subtraction that produced it"
        );
    }

    /// The other side of the same measurement, and the reason the floor cannot
    /// simply be "a tight box abstains": on `scaled_feasible_a` — the model
    /// gh #712 exists to reject — the box is `1e-9` wide at `|x| ≈ 3.8`, the
    /// iterate sits `5e-10` inside it, and *that* slack is six orders above its
    /// own quantum. It is counted, and the point is refused.
    #[test]
    fn a_resolvable_slack_in_an_equally_tight_box_is_counted() {
        let x = -3.8075263197246607;
        let half = 5.0e-10;
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![0.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![x - half],
            ub: vec![x + half],
        };
        let sol = point(x, 4.566e12);
        let comp = resolvable_complementarity(&prob, &sol);
        assert!(
            comp > 1e3,
            "a slack of {half:.1e} at |x| = {:.3e} is {:.0e} quanta wide and must \
             be counted, got {comp:.3e}",
            x.abs(),
            half / (f64::EPSILON * x.abs())
        );
        assert!(
            comp / 1.0 > FALSE_OPTIMUM_REL_TOL,
            "and it must clear the cut once the objective normalizer is honest"
        );
    }

    /// The rule itself, stated once: the quantum scales with the *magnitude of
    /// the quantities subtracted*, not with the slack.
    #[test]
    fn the_quantum_scales_with_the_operands_not_the_slack() {
        // The same absolute slack is noise next to `1e12` and plain data next
        // to `1.0`.
        assert!(!slack_is_resolvable(1e-3, 1e12));
        assert!(slack_is_resolvable(1e-3, 1.0));
        // An exact zero never counts, at any magnitude.
        assert!(!slack_is_resolvable(0.0, 0.0));
    }
}

#[cfg(test)]
/// gh #293 retry budget. The halving is gated on two conditions and both
/// are load-bearing, so both are pinned here rather than left to the corpus.
mod equilibrated_retry_budget_tests {
    use super::equilibrated_retry_budget;
    use crate::qp::{QpProblem, QpStatus, Triplet};

    fn lp() -> QpProblem {
        QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![1.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![0.0],
            ub: vec![1.0],
        }
    }

    fn qp() -> QpProblem {
        QpProblem {
            p_lower: vec![Triplet::new(0, 0, 2.0)],
            ..lp()
        }
    }

    /// An LP retry that has not converged is only ever accepted as a
    /// certified `Optimal`, so half a budget is all it can usefully spend.
    #[test]
    fn a_stalled_lp_retry_gets_half_the_budget() {
        for first in [QpStatus::IterationLimit, QpStatus::OptimalInaccurate] {
            assert_eq!(equilibrated_retry_budget(&lp(), first, 200), 100);
        }
    }

    /// The retry exists for Hessian curvature that NT scaling cannot see,
    /// which a QP can legitimately spend most of a budget recovering from —
    /// `QSCFXM1/2/3` and `Q25FV47` are accepted at 131–168 iterations, and a
    /// blanket cap would demote all four to `OptimalInaccurate`.
    #[test]
    fn a_qp_retry_keeps_the_full_budget() {
        for first in [QpStatus::IterationLimit, QpStatus::OptimalInaccurate] {
            assert_eq!(equilibrated_retry_budget(&qp(), first, 200), 200);
        }
    }

    /// A `NumericalFailure` first solve accepts *any* non-failing retry
    /// status as an improvement on a breakdown. Capping the budget there
    /// would let the cap manufacture an `IterationLimit` and have it
    /// accepted — returning the capped iterate and reporting "Maximum
    /// iterations exceeded" where the honest answer is "Numerical failure".
    /// `issue_535_lp_falls_back_to_nlp` catches the end-to-end symptom on
    /// `lp_afiro` at `qp_tau=0.99`; this pins the cause.
    #[test]
    fn a_broken_down_lp_retry_keeps_the_full_budget() {
        assert_eq!(
            equilibrated_retry_budget(&lp(), QpStatus::NumericalFailure, 200),
            200
        );
    }

    /// The budget is a fraction of the caller's, not a constant, so a
    /// user-set `max_iter` carries through instead of being overridden.
    #[test]
    fn the_budget_scales_with_a_user_set_max_iter() {
        assert_eq!(
            equilibrated_retry_budget(&lp(), QpStatus::IterationLimit, 40),
            20
        );
    }
}

#[cfg(test)]
mod finite_guard_tests {
    //! gh #491: a non-finite entry never leaves the crate.

    use super::{finite_or_failed, mark_timed_out};
    use crate::qp::{QpProblem, QpSolution, QpStatus, Triplet};

    fn prob() -> QpProblem {
        QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-1.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![1.0],
            lb: vec![],
            ub: vec![],
        }
    }

    fn sol(status: QpStatus, x: Vec<f64>, obj: f64) -> QpSolution {
        QpSolution {
            status,
            x,
            y: vec![],
            z: vec![0.0],
            z_lb: vec![0.0; 2],
            z_ub: vec![0.0; 2],
            obj,
            iters: 7,
            iterates: Vec::new(),
        }
    }

    /// A finite solution passes through untouched — *including* a failed one,
    /// whose iterate a caller may still want to look at.
    #[test]
    fn finite_solutions_are_returned_unchanged() {
        for status in [QpStatus::Optimal, QpStatus::NumericalFailure] {
            let s = sol(status, vec![0.25, 0.75], -0.5);
            let out = finite_or_failed(&prob(), s.clone());
            assert_eq!(out.status, status);
            assert_eq!(out.x, s.x);
            assert_eq!(out.obj, s.obj);
            assert_eq!(out.iters, 7, "the iteration count is not a casualty");
        }
    }

    /// A `NaN` anywhere — the iterate, a multiplier, or the objective — is
    /// replaced by a zero-filled `NumericalFailure`. `NaN` is never
    /// information: it cannot be checked against a bound, printed into a
    /// `.sol`, or warm-started from, and it turns every arithmetic downstream
    /// into another `NaN`.
    #[test]
    fn non_finite_solutions_become_an_honest_failure() {
        let cases = [
            sol(QpStatus::NumericalFailure, vec![f64::NAN, 0.0], f64::NAN),
            sol(QpStatus::Optimal, vec![f64::INFINITY, 0.0], 1.0),
            sol(QpStatus::Optimal, vec![0.0, 0.0], f64::NAN),
            QpSolution {
                z: vec![f64::NAN],
                ..sol(QpStatus::Optimal, vec![0.0, 0.0], 0.0)
            },
            QpSolution {
                z_ub: vec![0.0, f64::NAN],
                ..sol(QpStatus::Optimal, vec![0.0, 0.0], 0.0)
            },
        ];
        for (i, s) in cases.into_iter().enumerate() {
            let out = finite_or_failed(&prob(), s);
            assert_eq!(out.status, QpStatus::NumericalFailure, "case {i}");
            assert!(
                out.x.iter().chain(&out.z).all(|v| v.is_finite()) && out.obj.is_finite(),
                "case {i}: still non-finite"
            );
            assert_eq!(out.x, vec![0.0, 0.0], "case {i}");
            assert_eq!(out.iters, 7, "case {i}: the iteration count is kept");
        }
    }

    /// A deadline crossing can only ever *land on* a finished solve — every
    /// caller of [`mark_timed_out`] runs after an inner solve returned. So a
    /// conclusion the solve actually reached must survive it: the user whose
    /// problem converges a millisecond past `time_limit` gets the optimum they
    /// computed, not a report that nothing was solved.
    #[test]
    fn a_verdict_outranks_the_clock() {
        for status in [
            QpStatus::Optimal,
            QpStatus::OptimalInaccurate,
            QpStatus::PrimalInfeasible,
            QpStatus::DualInfeasible,
        ] {
            let s = sol(status, vec![0.25, 0.75], -0.5);
            let out = mark_timed_out(s.clone());
            assert_eq!(out.status, status, "a timeout erased a {status:?} verdict");
            assert_eq!(out.x, s.x, "and it must not disturb the iterate");
        }
    }

    /// The other half of the rule: a solve that stopped *without* concluding
    /// anything is relabelled, because "the clock ran out" is the more useful
    /// account of why it stopped.
    #[test]
    fn a_non_verdict_is_relabelled_as_cancelled() {
        for status in [QpStatus::IterationLimit, QpStatus::NumericalFailure] {
            let s = sol(status, vec![0.25, 0.75], -0.5);
            let out = mark_timed_out(s.clone());
            assert_eq!(out.status, QpStatus::TimeLimit, "from {status:?}");
            assert_eq!(out.x, s.x, "the best iterate is still worth returning");
        }
    }
}

#[cfg(test)]
mod compensated_residual_tests {
    //! gh #880: the `σ` forward-error arm's right-hand side is `−(Px + c)`,
    //! and it now drives a user-visible status, so the arithmetic that builds
    //! it has to be better than the quantity it is measuring.

    use super::residual_compensated;
    use crate::qp::{QpProblem, Triplet};

    fn row_problem(vals: &[f64], c0: f64) -> QpProblem {
        QpProblem {
            n: vals.len(),
            // One dense row 0 of `P`, off-diagonals held on the lower triangle.
            // Row 0 of a symmetric `P`, held on the lower triangle: `P[0][j]`
            // is stored at `(j, 0)`.
            p_lower: vals
                .iter()
                .enumerate()
                .map(|(j, v)| Triplet::new(j, 0, *v))
                .collect(),
            c: std::iter::once(c0)
                .chain(std::iter::repeat_n(0.0, vals.len() - 1))
                .collect(),
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![f64::NEG_INFINITY; vals.len()],
            ub: vec![f64::INFINITY; vals.len()],
        }
    }

    /// **The plain sum does not merely lose digits here — it loses the
    /// answer.** `1e16 + 1 − 1e16` left to right is `0.0` in `f64`: the `1`
    /// falls off the end of the accumulator and never comes back. The exact
    /// row value is `1.0`, so the estimator would read a converged residual
    /// where the truth is `1.0`, and `sigma_forward_error_is_small` would
    /// certify a point it should reject.
    ///
    /// This is the mechanism behind the `0.29×`–`5.4×` spread recorded at the
    /// call site, isolated to a case whose exact answer is known by
    /// construction rather than by a reference implementation — every value
    /// here is exactly representable, so `1.0` is arithmetic, not a
    /// tolerance.
    ///
    /// | change | effect |
    /// |---|---|
    /// | restore `prob.p_mul` + a plain `+= c` | reads `0.0`; this test fails |
    /// | drop the `lo` correction term when summing | reads `0.0`; this test fails |
    /// | drop the `mul_add` product-error term | the scaled arm below fails |
    #[test]
    fn the_compensated_residual_survives_a_cancelling_row() {
        // Row 0 of `P` is (1e16, 1, -1e16) and `x = (1, 1, 1)`, so
        // `(Px)₀ = 1.0` exactly and `c₀ = 0`.
        let prob = row_problem(&[1e16, 1.0, -1e16], 0.0);
        let x = vec![1.0, 1.0, 1.0];

        let mut plain = vec![0.0; prob.n];
        prob.p_mul(&x, &mut plain);
        assert_eq!(
            plain[0], 0.0,
            "the premise: the plain traversal loses the row entirely"
        );

        // `residual_compensated` returns `−(Px + c)`, the affine-scaling
        // right-hand side, so the exact row value `1.0` comes back negated.
        let got = residual_compensated(&prob, &x);
        assert_eq!(
            got[0], -1.0,
            "compensated residual must recover the exact row value"
        );
    }

    /// The same row with a non-unit `x`, so the *products* cancel rather than
    /// the stored coefficients. This is the half `two_sum` alone cannot fix:
    /// `1e16 · 3` and `−1e16 · 3` are exact, but `1 · 0.1` is not, and the
    /// `mul_add` term is what carries that product's rounding error.
    #[test]
    fn the_product_error_term_is_carried_too() {
        let prob = row_problem(&[1e16, 1.0, -1e16], 0.0);
        let x = vec![3.0, 0.1, 3.0];
        // Exact: 1e16·3 + 1·0.1 + (−1e16)·3 = 0.1 (the `f64` nearest 0.1),
        // negated by the `−(Px + c)` convention.
        let got = residual_compensated(&prob, &x);
        assert!(
            (got[0] + 0.1_f64).abs() <= f64::EPSILON,
            "expected the f64 nearest −0.1, got {}",
            got[0]
        );
    }
}

#[cfg(test)]
mod complementarity_scale_tests {
    //! gh #880: the slack scale in [`super::sigma_complementarity_is_genuine`]
    //! must not collapse to zero on a bound of `0`.

    use crate::cones::CompositeCone;
    use crate::qp::{QpProblem, QpSolution, QpStatus};

    /// A variable resting on `lb = 0` with a converged slack and a large
    /// multiplier. `z·slack` exceeds an absolute `tol` on the multiplier's
    /// magnitude alone, so the guard falls through to its two ratio tests; the
    /// slack one is the escape that must fire.
    ///
    /// Without the `max(1, ·)` floor the scale is `max(|0|, |0|) = 0`, so
    /// `|1e-16| <= cut·0` is false and a converged point is declared
    /// non-complementary. The multiplier test cannot save it — the multiplier
    /// is genuinely the dominant term in its row, which is the whole point of
    /// an active bound.
    ///
    /// | change | effect |
    /// |---|---|
    /// | drop `.max(1.0)` from `negligible` | this test fails |
    #[test]
    fn a_converged_slack_on_a_zero_bound_is_negligible() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![1e10],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![0.0],
            ub: vec![f64::INFINITY],
        };
        let sol = QpSolution {
            status: QpStatus::Optimal,
            // Converged: on the bound to within one ulp of the scale.
            x: vec![1e-16],
            y: vec![],
            z: vec![],
            z_lb: vec![1e10],
            z_ub: vec![0.0],
            obj: 1e-6,
            iters: 1,
            iterates: vec![],
        };
        // `z·slack = 1e10 · 1e-16 = 1e-6`, above `tol`, so the absolute arm
        // does not apply and the ratio tests decide.
        let dscale = vec![1e10];
        assert!(
            super::sigma_complementarity_is_genuine(
                &prob,
                &CompositeCone::single_nonneg(0),
                &sol,
                1e-8,
                1e-6,
                &dscale,
            ),
            "a slack of 1e-16 on a bound of 0 is complementary; the scale must \
             not collapse to zero underneath it"
        );
    }
}

#[cfg(test)]
mod sigma_crossover_rerecord_tests {
    //! gh#880 follow-up: the `σ` demotion must describe the point being
    //! returned, not the one `maybe_crossover` replaced.
    //!
    //! Unit tests rather than a fixture because the path is very nearly
    //! unreachable by construction — `σ` engages only when
    //! `max(‖P‖∞, ‖c‖∞)·ε > tol`, which 1 of 79 CLI fixtures, 0 of 138
    //! Maros-Mészáros problems and 0 of 150 purpose-built degenerate LPs
    //! with `‖c‖∞` from `1e8` to `1e13` satisfy — while the consequence
    //! is not small: gh#888 reroutes `ProblemClass::Lp` on
    //! `OptimalInaccurate`, so a stale demotion discards a
    //! crossover-purified **exact vertex** and re-solves on the NLP arm.
    //!
    //! All three branches, because the rule branches.
    //!
    //! | mutation | what goes red |
    //! |---|---|
    //! | drop the `before != sol.x` guard | `an_unmoved_answer_keeps_its_recorded_verdict` |
    //! | return `recorded` unconditionally | `a_crossover_that_moved_the_answer_re_asks` |
    //! | drop the `None` arm (always re-ask) | `no_recorded_verdict_means_nothing_to_re_ask` |

    use super::{QpSolution, QpStatus, sigma_verdict_after_crossover};
    use crate::qp::{QpProblem, Triplet};

    /// `min ½‖x‖² − xᵀe`, whose unconstrained minimiser is `x = e`.
    /// `P = I` so the forward-error estimator is exactly `‖x − e‖`.
    fn unit_problem(n: usize) -> QpProblem {
        QpProblem {
            n,
            p_lower: (0..n).map(|i| Triplet::new(i, i, 1.0)).collect(),
            c: vec![-1.0; n],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![f64::NEG_INFINITY; n],
            ub: vec![f64::INFINITY; n],
        }
    }

    fn solution_at(x: Vec<f64>) -> QpSolution {
        QpSolution {
            status: QpStatus::Optimal,
            x,
            y: vec![],
            z: vec![],
            z_lb: vec![],
            z_ub: vec![],
            obj: 0.0,
            iters: 0,
            iterates: vec![],
        }
    }

    /// No verdict was recorded, so there is nothing to re-ask and the
    /// estimator must not run — this is the ordinary path, and it pays
    /// neither the clone nor the back-solve.
    #[test]
    fn no_recorded_verdict_means_nothing_to_re_ask() {
        let prob = unit_problem(3);
        // A point far from the optimum, which the estimator *would*
        // reject if it were consulted. `None` says it is not.
        let sol = solution_at(vec![100.0, 100.0, 100.0]);
        assert!(!sigma_verdict_after_crossover(
            &prob, &sol, None, 1e-8, false
        ));
        assert!(sigma_verdict_after_crossover(&prob, &sol, None, 1e-8, true));
    }

    /// Crossover declined, so the recorded verdict still describes the
    /// answer and stands unchanged — including when it demoted.
    #[test]
    fn an_unmoved_answer_keeps_its_recorded_verdict() {
        let prob = unit_problem(3);
        let x = vec![1.0, 1.0, 1.0];
        let sol = solution_at(x.clone());
        // Recorded `true` on a point that is in fact exact: the verdict
        // is carried through regardless, because nothing moved.
        assert!(sigma_verdict_after_crossover(
            &prob,
            &sol,
            Some(&x),
            1e-8,
            true
        ));
        assert!(!sigma_verdict_after_crossover(
            &prob,
            &sol,
            Some(&x),
            1e-8,
            false
        ));
    }

    /// Crossover moved the answer, so the verdict is re-asked of the new
    /// point — and a demotion recorded about the interior iterate is
    /// dropped when the vertex it was replaced by is exact.
    ///
    /// This is the case gh#888's reroute turns from a label into a
    /// discarded answer.
    #[test]
    fn a_crossover_that_moved_the_answer_re_asks() {
        let prob = unit_problem(3);
        let interior = vec![0.5, 0.5, 0.5];
        // The purified vertex: the exact minimiser of `unit_problem`.
        let exact = solution_at(vec![1.0, 1.0, 1.0]);
        assert!(
            !sigma_verdict_after_crossover(&prob, &exact, Some(&interior), 1e-8, true),
            "a demotion about the discarded interior iterate survived onto \
             the exact vertex that replaced it"
        );
        // ...and the re-ask is a measurement, not a clear: a moved answer
        // that is still far from optimal stays demoted.
        let far = solution_at(vec![1.0e3, 1.0e3, 1.0e3]);
        assert!(sigma_verdict_after_crossover(
            &prob,
            &far,
            Some(&interior),
            1e-8,
            false
        ));
    }
}
