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
use crate::debug::{ConvexDebugState, fire};
use crate::qp::{QpIterate, QpProblem, QpSolution, QpStatus};
use pounce_common::debug::{Checkpoint, DebugAction, DebugHook};
use pounce_common::types::{Index, Number};
use pounce_linsol::{Factorization, SparseSymLinearSolverInterface};
use std::collections::BTreeMap;

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
const FARKAS_RESID_TOL: f64 = 1e-10;

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

/// Options for the QP interior-point solve.
#[derive(Debug, Clone, Copy)]
pub struct QpOptions {
    /// Convergence tolerance on the max KKT residual and duality measure.
    pub tol: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Fraction-to-boundary parameter τ ∈ (0, 1). (The centering
    /// parameter σ is computed adaptively by the Mehrotra predictor;
    /// it is not an option.)
    pub tau: f64,
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
}

impl Default for QpOptions {
    fn default() -> Self {
        QpOptions {
            tol: 1e-8,
            max_iter: 200,
            tau: 0.95,
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
            // Opt-in: off by default. See the field doc — correct but slow on
            // the LPs it targets, and does not yet reach Optimal on GEN (#133).
            crossover: false,
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
    let mut make_backend = make_backend;
    // gh #295: reject a box that admits no finite point (a *present* `+∞`
    // lower / `−∞` upper bound) before bound expansion — `expand_bounds` is
    // sign-agnostic and would otherwise mishandle it and report a violating
    // point `Optimal`.
    if prob.bounds_admit_no_point() {
        return trivial_primal_infeasible_solution(prob);
    }
    // Interior-point solve in the original problem's coordinates (the core
    // already unscales any internal Ruiz equilibration before returning).
    let sol = solve_qp_ipm_core(prob, opts, &mut make_backend);
    // LP-crossover refinement: for a pure LP, purify the interior iterate to an
    // exact optimal vertex via the active-set engine. Gated to pure LPs and
    // never-regressing — a no-op for QPs and whenever the vertex is not a
    // strict improvement. Runs against the same un-equilibrated `prob` so the
    // `z`/`s` conventions line up. See [`crate::crossover`].
    crate::crossover::maybe_crossover(prob, sol, opts, &mut make_backend)
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
    if opts.equilibrate && !opts.use_hsde {
        let (scaled, scaling) = crate::equilibrate::equilibrate(prob);
        let inner = QpOptions {
            equilibrate: false,
            ..*opts
        };
        let mut sol = solve_qp_ipm_unscaled(&scaled, &inner, make_backend);
        scaling.unscale_solution(prob, &mut sol);
        return sol;
    }
    let mut make_backend = make_backend;
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
    if opts.use_hsde && opts.equilibrate && sol.status == QpStatus::NumericalFailure {
        let (scaled, scaling) = crate::equilibrate::equilibrate(prob);
        let inner = QpOptions {
            equilibrate: false,
            ..*opts
        };
        let mut retry = solve_qp_ipm_unscaled(&scaled, &inner, &mut make_backend);
        scaling.unscale_solution(prob, &mut retry);
        if retry.status != QpStatus::NumericalFailure {
            return retry;
        }
    }
    sol
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
    mut make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    // gh #295: reject an impossible box (present `+∞` lower / `−∞` upper
    // bound) before bound expansion, as the non-debug path does.
    if prob.bounds_admit_no_point() {
        return trivial_primal_infeasible_solution(prob);
    }
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
    // Warm-starting requires the direct infeasible-start solver: HSDE
    // self-starts and ignores a warm point (see `QpOptions::use_hsde`). So this
    // path always runs the direct method, independent of the (HSDE) default —
    // otherwise the warm start would silently do nothing. A caller that
    // warm-starts is doing nearby reoptimization (a known-solvable
    // neighborhood), where the direct path's fragility is not a concern.
    // gh #295: reject an impossible box (present `+∞` lower / `−∞` upper
    // bound) before equilibration and bound expansion.
    if prob.bounds_admit_no_point() {
        return trivial_primal_infeasible_solution(prob);
    }
    let direct = QpOptions {
        use_hsde: false,
        equilibrate: false,
        ..*opts
    };
    // Equilibrate (default on) just as the cold path does, mapping the
    // warm-start point into the scaled coordinates so the warm benefit is
    // preserved and the two paths run on identically-conditioned data.
    if opts.equilibrate {
        let (scaled, scaling) = crate::equilibrate::equilibrate(prob);
        let scaled_warm = scaling.scale_warm_start(warm);
        let mut sol = solve_qp_ipm_warm(&scaled, &direct, &scaled_warm, make_backend);
        scaling.unscale_solution(prob, &mut sol);
        return sol;
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
    // gh #295: reject an impossible box (present `+∞` lower / `−∞` upper
    // bound) before bound expansion; `expand_bounds` is sign-agnostic.
    if prob.bounds_admit_no_point() {
        return trivial_primal_infeasible_solution(prob);
    }
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
            let upgraded = match (sol.status, retry.status) {
                // A clean optimum always wins.
                (_, QpStatus::Optimal) => true,
                // A failed solve carries no usable answer, so any verdict
                // with information beats it. An `OptimalInaccurate` original
                // is a usable answer, so only the clean arm above replaces it
                // (in particular a contradictory infeasibility claim does not).
                (
                    QpStatus::NumericalFailure | QpStatus::IterationLimit,
                    QpStatus::OptimalInaccurate
                    | QpStatus::PrimalInfeasible
                    | QpStatus::DualInfeasible,
                ) => true,
                _ => false,
            };
            if upgraded {
                return retry;
            }
        }
        return sol;
    }
    solve_socp_symmetric(prob, cones, opts, make_backend)
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
    mut make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
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
/// start-independent, so warm starting only reduces the iteration count.
/// `prob` must be bound-free (use `G`/`h` rows for all constraints).
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
    assert!(
        !prob.has_bounds(),
        "solve_socp_ipm_warm: encode bounds as G/h rows (bound expansion + warm not combined)"
    );
    if !cone_dims_cover(cones, prob.m_ineq()) {
        return failed_solution(
            prob,
            vec![0.0; prob.n],
            vec![0.0; prob.m_eq()],
            vec![0.0; prob.m_ineq()],
            0,
        );
    }
    let cone = CompositeCone::from_specs(cones);
    let w = WarmStart {
        x: warm.x.clone(),
        y: warm.y.clone(),
        z: warm.z.clone(),
    };
    solve_qp_core(prob, &cone, opts, Some(&w), make_backend)
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
            let mut sol = crate::hsde::solve_conic_hsde(&scaled, cone, opts, make_backend, None);
            for v in sol.y.iter_mut().chain(sol.z.iter_mut()) {
                *v *= sigma;
            }
            sol.obj *= sigma;
            return sol;
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

/// Build the starting iterate `(x, y, z, s)` for [`run_ipm`].
///
/// With no warm start (`warm = None`) this is the cold default
/// `x = 0, y = 0, z = 1, s = 1` — a perfectly centered interior point
/// (`s∘z = 1`) — preserving the established cold-start behavior exactly.
///
/// With a warm start it applies a **Mehrotra-style recentering** seeded
/// from the warm point (Mehrotra 1992, §7, adapted for warm starting):
///
/// 1. Keep the warm primal `x` and equality multipliers `y`.
/// 2. Take the implied slacks `s̃ = h − Gx` (their signs encode which
///    inequalities the warm `x` makes active/violated) and the warm `z`.
/// 3. Shift both into the strict interior by `δ = max(−1.5·min(·), floor)`.
///    The `floor` is **adaptive**: it is the warm point's KKT residual `ρ`
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
///    worse than a centered start.
/// 4. A final centering shift `½(s·z)/Σz`, `½(s·z)/Σs` balances `s` and
///    `z` (Mehrotra's second step).
///
/// The returned iterate always satisfies `s > 0, z > 0`. If `warm`'s
/// dimensions don't match the (expanded) problem it is ignored and the
/// cold start is used, so a stale warm start can never corrupt a solve.
fn init_iterate(
    prob: &QpProblem,
    cone: &CompositeCone,
    n: usize,
    m_eq: usize,
    m_ineq: usize,
    warm: Option<&WarmStart>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Cold start at the cone identity e (orthant: all ones; SOC: (1,0,…)),
    // a perfectly centered interior point (s∘z = e).
    let cold = || {
        let mut e = vec![0.0; m_ineq];
        cone.identity(&mut e);
        (vec![0.0; n], vec![0.0; m_eq], e.clone(), e)
    };
    // A matching primal `x` is enough to warm start; `y`/`z` fall back to
    // the cold values when they don't match (so a primal-only warm start —
    // e.g. feeding back just the previous primal — is supported).
    let w = match warm {
        Some(w) if w.x.len() == n => w,
        _ => return cold(),
    };

    let x = w.x.clone();
    let y = if w.y.len() == m_eq {
        w.y.clone()
    } else {
        vec![0.0; m_eq]
    };
    let mut z = if w.z.len() == m_ineq {
        w.z.clone()
    } else {
        let mut e = vec![0.0; m_ineq];
        cone.identity(&mut e);
        e
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

    // Adaptive interior floor sized to the warm point's KKT residual ρ on
    // *this* problem. ρ measures how far the warm point is from satisfying
    // the new KKT system: a small ρ (nearby problem, stable active set)
    // lets the slacks/multipliers stay near their warm — correctly
    // structured — values, so the IPM exploits the warm duals and needs
    // few steps; a large ρ (the active set moved, so the warm point is
    // badly infeasible) softens the floor toward the conservative cold
    // level `0.1·scale`. This self-corrects: warm starting never does
    // worse than a centered start, and gains the most when it can.
    let floor = {
        let mut rd = prob.c.clone();
        prob.p_mul_add(&x, &mut rd);
        prob.at_mul_add(&y, &mut rd);
        prob.gt_mul_add(&z, &mut rd);
        let mut rp: Vec<f64> = prob.b.iter().map(|b| -b).collect();
        prob.a_mul_add(&x, &mut rp);
        // Inequality infeasibility of the warm point: max(0, Gx − h) = −s̃.
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

    let mut iters = 0;
    let mut status = QpStatus::IterationLimit;
    let mut iterates: Vec<QpIterate> = Vec::new();

    for it in 0..opts.max_iter {
        iters = it;

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

        if res < opts.tol {
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

        // Affine step lengths and the predicted duality measure μ_aff.
        let (alpha_p_aff, alpha_d_aff) =
            step_lengths(cone, &s, &ds_aff, &z, &dz_aff, opts.tau, m_ineq);
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

            let (alpha_p, alpha_d) = step_lengths(cone, &s, &ds, &z, &dz, opts.tau, m_ineq);
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

    // Final verdict from the true KKT error of the point being returned — the
    // same rule the HSDE driver applies (see `VERDICT` in `hsde.rs`), so the two
    // drivers cannot drift apart on whether a solve that ended without its own
    // verdict actually produced an answer. Strictly an upgrade.
    if matches!(
        status,
        QpStatus::NumericalFailure | QpStatus::IterationLimit
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
        self.solve_inner(prob, None)
    }

    /// Solve `prob` reusing the captured symbolic factor **and** warm
    /// starting from `warm` (a nearby problem's solution). Combines the
    /// two reuse axes: the symbolic factorization is paid once at `build`,
    /// and the interior-point iteration is seeded from the warm point (see
    /// [`QpWarmStart`]). Same structure requirement as [`Self::solve`].
    pub fn solve_warm(&mut self, prob: &QpProblem, warm: &QpWarmStart) -> QpSolution {
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
        self.solve_inner(prob, Some(&w))
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
fn step_lengths(
    cone: &CompositeCone,
    s: &[f64],
    ds: &[f64],
    z: &[f64],
    dz: &[f64],
    tau: f64,
    m_ineq: usize,
) -> (f64, f64) {
    if m_ineq == 0 {
        return (1.0, 1.0);
    }
    (cone.max_step(s, ds, tau), cone.max_step(z, dz, tau))
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

        // (x,x): P + δI.
        for t in &prob.p_lower {
            add(t.row, t.col, t.val);
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

        KktStructure {
            airn,
            ajcn,
            values,
            dim,
            z_blocks,
            y_diag_pos,
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
