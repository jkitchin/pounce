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
        solve_qp_ipm_scoped(prob, opts, make_backend)
    })
}

fn solve_qp_ipm_scoped<F>(prob: &QpProblem, opts: &QpOptions, make_backend: F) -> QpSolution
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
    let sol = solve_qp_ipm_core(prob, opts, &mut make_backend);
    if crate::deadline::expired() {
        return mark_timed_out(sol);
    }
    // LP-crossover refinement: for a pure LP, purify the interior iterate to an
    // exact optimal vertex via the active-set engine. Gated to pure LPs and
    // never-regressing — a no-op for QPs and whenever the vertex is not a
    // strict improvement. Runs against the same un-equilibrated `prob` so the
    // `z`/`s` conventions line up. See [`crate::crossover`].
    let sol = crate::crossover::maybe_crossover(prob, sol, opts, &mut make_backend);
    // Crossover declines rather than restamps when the budget runs out mid-
    // refinement, so account for a deadline crossed in there here — where
    // `mark_timed_out`'s verdict rule applies and an `Optimal` cannot be lost.
    let sol = if crate::deadline::expired() {
        mark_timed_out(sol)
    } else {
        sol
    };
    finite_or_failed(prob, sol)
}

/// The interior-point solve (the historical [`solve_qp_ipm`] body): bounds-aware
/// orthant solve with optional Ruiz equilibration, returning a solution in the
/// original problem's coordinates. Factored out so [`solve_qp_ipm`] can layer
/// the LP-crossover refinement on top.
fn solve_qp_ipm_core<F>(prob: &QpProblem, opts: &QpOptions, make_backend: F) -> QpSolution
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
        return equilibrated_solve(prob, opts, /* use_hsde */ false, &mut make_backend);
    }
    let sol = solve_qp_ipm_unscaled(prob, opts, &mut make_backend);
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
    if opts.use_hsde && opts.equilibrate && retry_on {
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
    let sol = verify_or_repair_optimum(prob, opts, sol, &mut make_backend);
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
    {
        if crate::deadline::expired() {
            return mark_timed_out(sol);
        }
        let verify = equilibrated_solve(prob, opts, /* use_hsde */ false, &mut make_backend);
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
    if sol.status == QpStatus::DualInfeasible {
        if crate::deadline::expired() {
            return mark_timed_out(sol);
        }
        let twin = QpProblem {
            p_lower: Vec::new(),
            c: vec![0.0; prob.n],
            ..prob.clone()
        };
        if solve_qp_ipm_unscaled(&twin, opts, &mut make_backend).status
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
    let mut sol = solve_qp_ipm_unscaled(&scaled, &inner, make_backend);
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
    let retry = equilibrated_solve(prob, opts, /* use_hsde */ true, make_backend);
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
    let direct = equilibrated_solve(prob, opts, /* use_hsde */ false, make_backend);
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
fn solve_qp_ipm_unscaled<F>(prob: &QpProblem, opts: &QpOptions, make_backend: F) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if !prob.has_bounds() {
        let cone = CompositeCone::single_nonneg(prob.m_ineq());
        return solve_qp_core(prob, &cone, opts, None, make_backend);
    }
    let (expanded, bound_rows) = expand_bounds(prob);
    let cone = CompositeCone::single_nonneg(expanded.m_ineq());
    let sol = solve_qp_core(&expanded, &cone, opts, None, make_backend);
    split_bound_duals(prob, &bound_rows, sol)
}

/// Solve a convex LP / QP with an interactive [`DebugHook`] attached: the
/// hook is fired at each interior-point checkpoint (iteration start, after
/// the Newton step, after the step is applied, and at termination) so a
/// debugger can step, inspect, and break on the solve.
///
/// Targets the direct (non-HSDE) convex IPM, so the debugged `x` block is
/// the user's variables (finite bounds are expanded into a trailing
/// nonnegative block, as in [`solve_qp_ipm`], and surface in the `s`/`z`
/// blocks). Apart from the hook the result is identical to
/// [`solve_qp_ipm`].
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
        solve_qp_ipm_debug_scoped(prob, opts, hook, make_backend)
    })
}

fn solve_qp_ipm_debug_scoped<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    hook: &mut dyn DebugHook,
    mut make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if crate::deadline::expired() {
        return timed_out_solution(prob);
    }
    // Screen the variable box before bound expansion, as the non-debug path
    // does (gh #295, gh #491).
    let snapped;
    let prob = match screen_variable_box(prob) {
        BoxScreen::Feasible => prob,
        BoxScreen::Empty => return trivial_primal_infeasible_solution(prob),
        BoxScreen::Snapped(p) => {
            snapped = p;
            &snapped
        }
    };
    // Build the factorization and run the core loop directly with the hook
    // (mirrors `solve_qp_core`'s non-HSDE path; `solve_qp_core` itself can't
    // carry the borrowed hook through its generic plumbing). When the HSDE
    // driver is selected, debug it instead — it self-starts and builds its
    // own factorization.
    let run = |p: &QpProblem, cone: &CompositeCone, mk: &mut F, hook: &mut dyn DebugHook| {
        if opts.use_hsde {
            return crate::hsde::solve_conic_hsde(p, cone, opts, mk, Some(hook));
        }
        match build_factorization(p, cone, opts, mk) {
            Ok((kkt, mut fact)) => run_ipm(p, cone, opts, &kkt, &mut fact, None, Some(hook)),
            Err(()) => failed_solution(
                p,
                vec![0.0; p.n],
                vec![0.0; p.m_eq()],
                vec![0.0; p.m_ineq()],
                0,
            ),
        }
    };
    if !prob.has_bounds() {
        let cone = CompositeCone::single_nonneg(prob.m_ineq());
        return run(prob, &cone, &mut make_backend, hook);
    }
    let (expanded, bound_rows) = expand_bounds(prob);
    let cone = CompositeCone::single_nonneg(expanded.m_ineq());
    let sol = run(&expanded, &cone, &mut make_backend, hook);
    split_bound_duals(prob, &bound_rows, sol)
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
            // One gate over every exit of the body below — see [`finite_or_failed`].
            let sol = finite_or_failed(
                prob,
                solve_qp_ipm_warm_inner(prob, opts, warm, make_backend),
            );
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
        return solve_qp_core(prob, &cone, &direct, Some(&w), make_backend);
    }
    let (expanded, bound_rows) = expand_bounds(prob);
    let w = WarmStart {
        x: warm.x.clone(),
        y: warm.y.clone(),
        z: merge_bound_duals(prob, &bound_rows, warm),
    };
    let cone = CompositeCone::single_nonneg(expanded.m_ineq());
    let sol = solve_qp_core(&expanded, &cone, &direct, Some(&w), make_backend);
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
        if crate::deadline::expired() {
            timed_out_solution(prob)
        } else {
            // One gate over every exit of the body below — see [`finite_or_failed`].
            let sol = finite_or_failed(prob, solve_socp_ipm_inner(prob, cones, opts, make_backend));
            if crate::deadline::expired() {
                mark_timed_out(sol)
            } else {
                sol
            }
        }
    })
}

fn solve_socp_ipm_inner<F>(
    prob: &QpProblem,
    cones: &[ConeSpec],
    opts: &QpOptions,
    make_backend: F,
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
        return solve_nonsym(prob, cones, opts, make_backend, None);
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
        let run = |o: &QpOptions, mk: &mut F| {
            let sol2 = solve_socp_symmetric(&prob2, &cones2, o, mk);
            let sol1 = chordal_reconstruct(sol2, &recon, &prob1);
            remap_decomposed_z(sol1, &row_map, prob.m_ineq())
        };
        let sol = run(opts, &mut make_backend);
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
        {
            let hsde_opts = QpOptions {
                use_hsde: true,
                ..*opts
            };
            let retry = run(&hsde_opts, &mut make_backend);
            if hsde_retry_is_upgrade(sol.status, retry.status) {
                return retry;
            }
        }
        return sol;
    }
    let mut make_backend = make_backend;
    let sol = solve_socp_symmetric(prob, cones, opts, &mut make_backend);
    // gh #414: a cone program whose cones are *all* nonnegative is an LP/QP
    // wearing the conic entry point's clothes (`solver_selection=socp` on a
    // box-constrained QP lands here), and it inherits the same false `Optimal`
    // under a variable-scale spread. Ruiz equilibration — which the repair
    // rests on — is a per-row scaling and stays sound exactly on the orthant,
    // so the check is gated on that and every genuine cone program is
    // untouched. See [`verify_or_repair_optimum`].
    if cones.iter().all(|c| matches!(c, ConeSpec::Nonneg(_))) {
        return verify_or_repair_optimum(prob, opts, sol, &mut make_backend);
    }
    sol
}

/// Debug-enabled [`solve_socp_ipm`]: fires the interactive [`DebugHook`] at
/// each interior-point checkpoint. Exponential / power cones run on the
/// non-symmetric HSDE driver; all other cones (orthant / SOC / PSD) run on
/// the direct symmetric IPM. Under the debugger a PSD cone is solved
/// *directly* (no chordal decomposition) so the debugged `x`/`s`/`y`/`z`
/// blocks correspond to the user's problem; the solution is unchanged.
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
        solve_socp_ipm_debug_scoped(prob, cones, opts, hook, make_backend)
    })
}

fn solve_socp_ipm_debug_scoped<F>(
    prob: &QpProblem,
    cones: &[ConeSpec],
    opts: &QpOptions,
    hook: &mut dyn DebugHook,
    mut make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if crate::deadline::expired() {
        return timed_out_solution(prob);
    }
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
        return solve_nonsym(prob, cones, opts, make_backend, Some(hook));
    }
    // Symmetric cones: debug the direct IPM (build the factorization and run
    // the core loop with the hook), bound-expanded as in
    // `solve_socp_symmetric`. PSD is solved directly here (no decomposition).
    let run = |p: &QpProblem, cone: &CompositeCone, mk: &mut F, hook: &mut dyn DebugHook| {
        match build_factorization(p, cone, opts, mk) {
            Ok((kkt, mut fact)) => run_ipm(p, cone, opts, &kkt, &mut fact, None, Some(hook)),
            Err(()) => failed_solution(
                p,
                vec![0.0; p.n],
                vec![0.0; p.m_eq()],
                vec![0.0; p.m_ineq()],
                0,
            ),
        }
    };
    if !prob.has_bounds() {
        let cone = CompositeCone::from_specs(cones);
        return run(prob, &cone, &mut make_backend, hook);
    }
    let (expanded, bound_rows) = expand_bounds(prob);
    let mut specs = cones.to_vec();
    specs.push(ConeSpec::Nonneg(bound_rows.len()));
    let cone = CompositeCone::from_specs(&specs);
    let sol = run(&expanded, &cone, &mut make_backend, hook);
    split_bound_duals(prob, &bound_rows, sol)
}

/// The symmetric-cone solve (orthant / SOC / PSD): expand finite bounds into
/// a trailing orthant block, run the Mehrotra core, and split the bound
/// duals back out. Shared by [`solve_socp_ipm`] and the PSD-decomposed path.
fn solve_socp_symmetric<F>(
    prob: &QpProblem,
    cones: &[ConeSpec],
    opts: &QpOptions,
    make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if !prob.has_bounds() {
        let cone = CompositeCone::from_specs(cones);
        return solve_qp_core(prob, &cone, opts, None, make_backend);
    }
    // Bounds expand into a trailing nonnegative block after the user cones.
    let (expanded, bound_rows) = expand_bounds(prob);
    let mut specs = cones.to_vec();
    specs.push(ConeSpec::Nonneg(bound_rows.len()));
    let cone = CompositeCone::from_specs(&specs);
    let sol = solve_qp_core(&expanded, &cone, opts, None, make_backend);
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
            let sol = finite_or_failed(
                prob,
                solve_socp_ipm_warm_scoped(prob, cones, warm, opts, make_backend),
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
        let sol = solve_qp_core(&expanded, &cone, &direct_opts, Some(&w), &mut make_backend);
        split_bound_duals(prob, &bound_rows, sol)
    };

    // `use_hsde` is the fallback permission here, not the initial-driver
    // selector.
    if opts.use_hsde && warm_hsde_retry_needed(direct.status) && !crate::deadline::expired() {
        let hsde_opts = QpOptions {
            use_hsde: true,
            ..*opts
        };
        let retry = solve_socp_ipm_inner(prob, cones, &hsde_opts, &mut make_backend);
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
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let mut sol = crate::hsde::solve_conic_hsde(scaled, cone, inner, make_backend, None);
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
    crate::hsde::relative_stop_permitted(gscale.max(pscale), tol)
        && unscaled_relative_kkt(prob, sol) <= cut
        // ... and the same question asked one row at a time, because
        // `gscale` and `pscale` are aggregates and an aggregate cannot see a
        // flat direction (gh #846). Strictly a further conjunct: it can only
        // turn an accept into a reject, and a reject costs one un-normalized
        // re-solve.
        && sigma_complementarity_is_genuine(prob, cone, sol, tol, cut)
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
/// # Complementarity is the binding half, and that is measured
///
/// A companion arm asking the same question of the **stationarity** rows
/// (`|rᵢ|` against the largest term that built row `i`) was written, kept
/// through the whole investigation, and then removed, because on this family
/// it rejects nothing the test below does not already reject. Removing it
/// turned no test red; removing the test below turns four red. Nor is that an
/// accident of the fixtures: the same spectrum *unconstrained*, in a wide box
/// it never reaches, and under an equality row all come back exact to
/// `3e-16`. The failure needs an **active bound**, because what buys the slack
/// is the embedding's objective-relative gap test (`gap / (1 + |obj|)`) and a
/// gap is spent on the bound multipliers. So this is where the guard belongs.
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
) -> bool {
    let n = prob.n;
    let mut px = vec![0.0; n];
    prob.p_mul(&sol.x, &mut px);
    let mut aty = vec![0.0; n];
    prob.at_mul(&sol.y, &mut aty);
    let mut gtz = vec![0.0; n];
    prob.gt_mul(&sol.z, &mut gtz);
    // Each variable row's stationarity denominator, the scale a multiplier
    // landing in it has to be significant against.
    let dscale: Vec<f64> = (0..n)
        .map(|i| {
            [px[i], prob.c[i], -sol.z_lb[i], sol.z_ub[i], aty[i], gtz[i]]
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs()))
        })
        .collect();

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

    let negligible = |v: f64, scale: f64| v.abs() <= cut * scale;
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
    solve_qp_ipm_core(prob, &direct, make_backend)
}

/// Bounds-agnostic Mehrotra predictor-corrector core. `prob.lb`/`ub` are
/// ignored here; the public [`solve_qp_ipm`] handles bound expansion.
fn solve_qp_core<F>(
    prob: &QpProblem,
    cone: &CompositeCone,
    opts: &QpOptions,
    warm: Option<&WarmStart>,
    mut make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if crate::deadline::expired() {
        return timed_out_solution(prob);
    }
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
            let sol = cost_normalized_hsde_solve(&scaled, cone, &inner, sigma, &mut make_backend);
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
            if sol.status != QpStatus::Optimal
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
            return candidates.swap_remove(pick);
        }
        return crate::hsde::solve_conic_hsde(prob, cone, opts, make_backend, None);
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
    run_ipm(prob, cone, opts, &kkt, &mut fact, warm, None)
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
        let res = dinf.max(pinf).max(mu);
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
        kkt.update_blocks(cone, &s, &z, opts.reg, &mut kkt_vals);
        // Adaptive μ-scaled regularization on the equality block: bounds the
        // duals of a rank-deficient equality Jacobian so the primal residual
        // converges below `tol` (see `adaptive_eq_reg`). Reduces to the static
        // `opts.reg` at the tolerance, leaving already-converging LPs/QPs
        // unchanged at the optimum.
        kkt.update_eq_reg(adaptive_eq_reg(mu, opts.reg), &mut kkt_vals);
        if fact.refactor(&kkt_vals).is_err() {
            status = QpStatus::NumericalFailure;
            break;
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
        let sol = cost_normalized_hsde_solve(&scaled, &cone, &inner, sigma, &mut backend);
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
        let sol = solve_qp_ipm_unscaled(&prob, &opts, backend);
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
