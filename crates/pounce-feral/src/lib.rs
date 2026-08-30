//! FERAL backend — pure-Rust sparse symmetric LDL^T factor.
//!
//! Implements [`SparseSymLinearSolverInterface`] over [`feral::Solver`].
//! The lifecycle mirrors `pounce_hsl::Ma57SolverInterface`:
//!
//! * `matrix_format()` returns [`EMatrixFormat::TripletFormat`] (1-based,
//!   lower-triangle COO) so the IPM `TSymLinearSolver` wrapper requires
//!   no changes versus the MA57 path.
//! * `initialize_structure` caches the 0-based row/col arrays needed by
//!   FERAL's [`CscMatrix::from_triplets`] and allocates the values
//!   buffer.
//! * `multi_solve` refreshes the `CscMatrix` from the cached pattern +
//!   caller-filled values and dispatches to [`feral::Solver::factor`] /
//!   [`feral::Solver::solve_many`]. The CSC is built once per pattern; later
//!   factorizations scatter the new values through a cached triplet → slot
//!   permutation instead of re-running `from_triplets` (gh#562). FERAL's
//!   pattern-fingerprint cache likewise reuses the symbolic factorization
//!   across iterates with identical structure (the IPM common case).
//! * `increase_quality` delegates to [`feral::Solver::increase_quality`]
//!   and uses MA57's `pivtol_changed` / `CallAgain` protocol so the
//!   upper-layer reload-and-retry semantics line up.

use std::sync::{Arc, Mutex};

pub mod schur;
pub use schur::FeralSchurSolver;

use feral::symbolic::SupernodeParams;
use feral::{CscMatrix, FactorStats, FactorStatus, NumericParams, RefineOptions, Solver};

/// Re-export so option-aware callers can construct a
/// [`FeralConfig`] without taking a direct dependency on `feral`.
pub use feral::scaling::ScalingStrategy;
pub use feral::symbolic::OrderingMethod;
use pounce_common::types::{Index, Number};
use pounce_linsol::summary::LinearSolverSummary;
use pounce_linsol::{
    EMatrixFormat, ESymSolverStatus, FactorPattern, SparseSymLinearSolverInterface,
};

/// Largest `nrhs` at which feral's blocked back-substitution is still
/// **bit-identical** to looping one column at a time — the ceiling
/// [`FeralSolverInterface::multi_solve_matches_single_solve`] reports.
///
/// feral routes a multi-RHS solve two ways (feral#57): below its
/// private `BLAS3_NRHS_THRESHOLD` each column runs the same rank-1
/// cascade a single-RHS solve would, in the same order, so no sum is
/// reassociated; at or above it a register-blocked TRSM/GEMM panel
/// kernel runs, which reassociates and is therefore only
/// tolerance-equal. That threshold is `32` in feral 0.17.0 and is not
/// exported, so this is our own conservative floor under it rather than
/// a mirror of it.
///
/// `multi_solve_bitwise_matches_single_solve_at_the_documented_ceiling`
/// is the guard: it exercises a real factor at exactly this width and
/// fails if feral ever lowers its threshold to or below it. Without
/// that test this constant is an assumption about another crate's
/// private internals, which is the shape of defect
/// `dev-notes/trajectory-regressions-and-the-fixture-sweep.md`
/// is about.
const FERAL_BITWISE_MULTI_SOLVE_MAX_NRHS: usize = 16;

/// FERAL solver implementing the IPM-side sparse symmetric backend
/// contract.
pub struct FeralSolverInterface {
    solver: Solver,

    initialized: bool,
    pivtol_changed: bool,
    refactorize: bool,
    refine: bool,
    /// See [`FeralConfig::increase_quality`].
    increase_quality: bool,
    /// See [`FeralConfig::refine_max_steps`].
    refine_max_steps: usize,
    /// See [`FeralConfig::refine_target`]. `0.0` disables the pre-check.
    refine_target: f64,

    /// Destination buffer for the in-place solve entry points, grown once
    /// and reused. feral's allocating entry points return an owned `Vec`
    /// per call — `dim * nrhs` doubles, ~946 KB per back-solve on the
    /// 118 276-dimension KKT of gh#698 — and the `_into` forms
    /// (feral#178) exist to take that back. Held here rather than
    /// allocated per call because `solve` is the hot path: an IPM does two
    /// of these per iteration for hundreds of iterations.
    ///
    /// It is a separate buffer rather than `rhs_vals` itself because
    /// feral makes aliasing `rhs` with `x_out` unrepresentable in safe
    /// Rust; the copy back into the caller's slice stays, the allocation
    /// does not.
    x_scratch: Vec<Number>,

    /// Residual buffer for the [`FeralConfig::refine_target`] pre-check,
    /// grown once and reused on the same terms as `x_scratch`. Unused
    /// (and never grown) when the target is disabled.
    resid_scratch: Vec<Number>,

    dim: Index,
    nonzeros: Index,

    /// 0-based row indices, fixed by `initialize_structure`.
    rows_0: Vec<usize>,
    /// 0-based column indices, fixed by `initialize_structure`.
    cols_0: Vec<usize>,
    /// Caller-filled numerical values, in the same order as
    /// `(rows_0, cols_0)`.
    values: Vec<Number>,

    /// Last factored matrix, retained so `backsolve` can run iterative
    /// refinement against it (feral's `solve_*_refined` requires `A`), and
    /// re-used as the destination of the values refill (see [`Self::slot`]).
    matrix: Option<CscMatrix>,

    /// Triplet → CSC slot permutation: `slot[k]` is the index into
    /// `matrix.values` that triplet `k` of `(rows_0, cols_0)` lands in.
    ///
    /// The sparsity pattern is fixed by `initialize_structure` and only the
    /// values change between IPM iterations, so the bucket-count / place /
    /// sort / sum-duplicates pass inside `CscMatrix::from_triplets`
    /// reproduces the same `col_ptr` and `row_idx` on every factorization —
    /// at the cost of one `Vec<(usize, f64)>` allocation and one sort *per
    /// column* (gh#562: 2.7 ms per factor on clnlbeam, 16.5 ms on dtoc2).
    /// Recorded on the first `factor()` after an `initialize_structure` and
    /// replayed as an allocation-free O(nnz) scatter thereafter; `None`
    /// until then, and reset to `None` by `initialize_structure` — the only
    /// place the pattern can change.
    slot: Option<Vec<usize>>,

    negevals: Index,

    /// Fill-reducing ordering configured at construction; surfaced on
    /// the `linear_solve` tracing span after each factorization
    /// (pounce#71).
    ordering: OrderingMethod,

    /// Absolute near-singularity floor; see
    /// [`FeralConfig::singular_pivot_floor`].
    singular_pivot_floor: f64,

    /// Floor under which a mismatching inertia count is treated as
    /// noise; `None` selects the dimension-aware default. See
    /// [`FeralConfig::inertia_pivot_floor`] and
    /// [`inertia_trust_floor`].
    inertia_pivot_floor: Option<f64>,

    /// Running aggregate updated after every successful `factor()`.
    /// Exposed via [`Self::summary`] and (mirrored to) the optional
    /// `sink` so an out-of-band consumer can read it post-solve
    /// without plumbing through the algorithm's wrapper layers.
    summary: LinearSolverSummary,

    /// Optional shared sink updated alongside `summary`. Decouples
    /// the algorithm-internal solver lifecycle from CLI / report
    /// consumers — pounce-cli installs an `Arc<Mutex<...>>` via
    /// [`Self::with_summary_sink`] and reads it after
    /// `optimize_tnlp` returns.
    sink: Option<Arc<Mutex<LinearSolverSummary>>>,
}

/// Construction-time configuration for [`FeralSolverInterface`].
///
/// Mirrors the pounce-extension options registered in
/// `pounce-algorithm`'s `upstream_options::register_all_options`
/// (`feral_cascade_break`, `feral_fma`, `feral_refine`,
/// `feral_singular_pivot_floor`). The IPM
/// caller reads those off its `OptionsList`, builds a `FeralConfig`,
/// and passes it to [`FeralSolverInterface::with_config`]. For
/// non-option callers (tests, standalone use, the env-only legacy
/// path), [`FeralSolverInterface::new`] keeps reading the
/// `POUNCE_FERAL_*` env vars to preserve the historic defaults.
/// Effective inertia-trust floor for a factorization of order `dim`.
///
/// `configured` is [`FeralConfig::inertia_pivot_floor`]: `Some(v)` pins
/// an absolute floor (`Some(0.0)` disables the trigger), `None` selects
/// the dimension-aware default.
///
/// That default is `n · eps`, the backward-error bound on the smallest
/// pivot of an equilibrated matrix of order `n`: below it a pivot's
/// sign is a rounding artefact, above it the sign is a measurement.
/// Before pounce gh#592 this was the fixed `1e-12` that sits mid-range
/// over `n = 10 … 10^6`, but a constant cannot be right at both ends of
/// six orders of magnitude in `n`, and at the small-to-middling
/// dimensions an IPM actually factors (`n` in the hundreds, `n · eps`
/// ≈ 5e-14) `1e-12` is more than an order of magnitude too generous.
/// It then convicts pivots whose sign is perfectly good, spends `δ_c`
/// on a constraint Jacobian that has full rank, and — because `δ_c`
/// persists once switched on — makes the inertia *harder* to hit for
/// the rest of the solve. See `dev-notes/issue-592-restart-non-idempotence.md`.
pub fn inertia_trust_floor(configured: Option<f64>, dim: usize) -> f64 {
    match configured {
        Some(v) => v,
        None => dim as f64 * f64::EPSILON,
    }
}

#[derive(Debug, Clone)]
pub struct FeralConfig {
    /// Tri-state. `None` (the pounce default) inherits whatever FERAL's
    /// `NumericParams::default()` ships with — as of FERAL Phase B
    /// (issue #55, commit 7554a78) that is CB armed with
    /// `ratio = 0.5, eps = 1e-10` and a symbolic-time delay budget.
    /// `Some(true)` explicitly arms with the same constants (matches
    /// the FERAL default; useful when a caller wants the intent
    /// recorded). `Some(false)` explicitly disarms by setting both
    /// `cascade_break_ratio` and `cascade_break_eps` to `None`; this
    /// is what enables FERAL's `DelayBudgetExceeded` path for non-root
    /// cascade victims and should only be used to reproduce the
    /// pre-Phase-B behaviour.
    pub cascade_break: Option<bool>,
    pub fma: bool,
    /// Whether every back-solve runs feral's own iterative refinement
    /// (`solve_refined` / `solve_many_refined` rather than `solve` /
    /// `solve_many`). Defaults to `true`, which is right for callers
    /// driving feral directly: nothing else is correcting the residual.
    ///
    /// Under pounce's IPM it is a **nested** loop, and that is worth
    /// knowing before tuning it. `PdFullSpaceSolver` runs Ipopt's own
    /// refinement (`min/max_refinement_steps`, `residual_ratio_max`) over
    /// the augmented system, and feral's inner loop runs up to
    /// [`Self::refine_max_steps`] correction steps plus the initial solve
    /// inside *each* of those — at feral's own default of 10 that is up
    /// to 10x11 substitution passes for one aug-system solve, each
    /// preceded by a matvec against the original matrix. The outer loop
    /// is the one that measures the residual ratio and decides; the
    /// inner one drives a residual nobody consults to a tolerance nobody
    /// set. `Ma57SolverInterface` and `TSymLinearSolver` have no
    /// in-backend refinement, so this is feral-only.
    ///
    /// Measured on a 118 276-dimension KKT (gh#698): `feral_refine = no`
    /// cut back-solve time 60% and wall time 20% and still converged.
    ///
    /// **This default is `true`, and the NLP solver's is `false`
    /// (gh#710, reported as gh#698 observation 5). The split is
    /// deliberate.** Turning the loop off is
    /// only safe for a caller that does two things Ipopt's architecture
    /// assumes of one: refine the *unreduced* system itself, and ask the
    /// backend for a better factorization when that stalls. Refinement
    /// is needed at all because feral's
    /// `ZeroPivotAction::ForceAccept` can leave real residual against
    /// the system it factorized — something Ipopt assumes a backend does
    /// not do — and with nothing to catch it, gh#590's data-scale-1e11
    /// LP exits `RestorationFailed`. Ipopt's own answer to a
    /// factorization that cannot deliver is not backend refinement; it
    /// is `IncreaseQuality` (`IpPDFullSpaceSolver.cpp:296`), which
    /// [`FeralSolverInterface::increase_quality`] now implements. So
    /// `pounce_algorithm::application::feral_config_from_options` turns
    /// this off for the IPM — `PdFullSpaceSolver` does both halves —
    /// and on the 126 028-dimension `laptime` KKT under limited-memory
    /// the pair is 68.9 s -> 18.8 s, against MA57's 10.7 s.
    ///
    /// It stays `true` here because every other caller of
    /// [`FeralSolverInterface::new`] — `pounce-convex`'s HSDE / SOS /
    /// active-set solvers, `pounce-rs`, `pounce-py`'s QP and SOS
    /// entry points — has the first half (`hsde::IR_MAX_PASSES`) but
    /// not the second: none of them calls `increase_quality`. Flipping
    /// the library default instead of the IPM's took backend
    /// refinement away from them with nothing in its place, and the
    /// Motzkin SOS relaxation at its minimal order went from `Optimal`
    /// to `IterationLimit` — a rank-deficient, non-strictly-feasible
    /// SDP is exactly the case that needs it. Wire `increase_quality`
    /// into a caller before turning this off for it.
    ///
    /// The grounds it used to stand on were already gone before that.
    /// That was pinene_3200, said to stall in the IPM tail when
    /// the cascade-break residual floor is left uncorrected, and never
    /// re-measured against the outer loop alone. It has now been measured
    /// (gh#710) and it does not stall: at 64 000 variables it converges to
    /// the same objective in 12 iterations with `feral_refine = no`, and
    /// in 13 at every budget from 0 to 10, in about two seconds either
    /// way. For what the fixture corpus says about the budget, and why
    /// it now defaults to 0, see [`Self::refine_max_steps`].
    ///
    /// This knob is no longer all-or-nothing. feral 0.17.0 shipped
    /// `RefineOptions { max_steps }` (feral#178), so the inner budget is
    /// now [`Self::refine_max_steps`], which defaults to 0 — so the
    /// inner loop already does the initial solve and stops, and `false`
    /// buys only the entry point. See that field.
    ///
    /// Worth knowing before retuning this: feral#179 measured that
    /// nothing merely ill-conditioned reaches the 10-step budget at all
    /// (Hilbert n = 8..40 stops at 3-7, ill-conditioned bordered KKTs at
    /// 1). It is only reachable when the factor is a genuinely perturbed
    /// approximate inverse — which is pounce's case, because pounce
    /// perturbs the L-factor. The full budget was therefore the normal
    /// case here, not the tail — which is what made it expensive.
    ///
    /// Tracked as gh#710, which carries the acceptance criteria.
    pub refine: bool,

    /// Whether [`FeralSolverInterface::increase_quality`] may escalate the
    /// factorization, or must decline the rung (gh #850).
    ///
    /// `true` here — the library default — because a caller that has no other
    /// recourse when a factorization cannot deliver wants the ladder. The NLP
    /// binding sets it `false`; `feral_config_from_options` explains why, and
    /// `feral_increase_quality` is the option that turns it back on.
    pub increase_quality: bool,
    /// How many correction steps feral's inner refinement may take per
    /// back-solve, when [`Self::refine`] is on. Passed straight through
    /// as `feral::RefineOptions::with_max_steps`.
    ///
    /// This is the knob [`Self::refine`]'s documentation asks for, and it
    /// is the reason feral 0.17.0 exists (feral#178, from pounce gh#698
    /// observation 5, tracked as gh#710). Before it, the budget was
    /// hard-coded at 10 and pounce's only choice was 10 or nothing —
    /// and it can use neither. Ten is the wrong number for a caller that
    /// runs its own refinement over the same system, which
    /// `PdFullSpaceSolver` does.
    ///
    /// The cap is an upper bound only. feral's `eps*sqrt(n)` residual
    /// target, its 100x divergence guard and its 2-strike plateau exit
    /// all keep priority, and the best-iterate contract holds at every
    /// value, so no cap returns an answer worse than the unrefined
    /// solve.
    ///
    /// **Defaults to feral's own 10, and no smaller constant is safe.**
    /// This is not a number anyone chose for pounce; it is the cap that
    /// happens to survive the corpus. The three obvious alternatives each
    /// lose something different, and they do not lose the same thing:
    ///
    /// | budget | gh#590 LP (scale 1e11) | `deb7` exact | `cresc4` lbfgs | `laptime` wall |
    /// |--------|------------------------|--------------|----------------|----------------|
    /// | 0      | **RestorationFailed**  | ok, 146 it   | ok, 105 it     | 16.7 s |
    /// | 1      | ok                     | **Error, 183** | **Infeasible, 32** | 26.1 s |
    /// | 2      | ok                     | **Error, 258** | ok, but 997 it | 33.9 s |
    /// | 10     | ok                     | ok, 171 it   | ok, 143 it     | 61.7 s |
    ///
    /// **Why it is chaotic rather than monotone.** feral's convergence
    /// test is `‖r‖₂/‖b‖₂ < ε·√n` — machine precision — and on a
    /// 118276-dimension near-singular KKT that target is simply not
    /// reachable. The loop therefore runs to the cap on every back-solve,
    /// and the best-iterate contract returns whichever of the `k+1`
    /// iterates had the smallest `‖r‖₂`. Those iterates are bouncing
    /// around the precision floor without converging, and smallest `‖r‖₂`
    /// on the condensed system has no relationship to which gives the
    /// better Newton step. So the verdict at each cap is close to a
    /// lottery draw, which is what the table above is showing. Ten is a
    /// lucky ticket, not a quality property: `eigena2` under
    /// limited-memory is `Optimal` at 5, `SolvedToAcceptableLevel` at 10,
    /// and `ErrorInStepComputation` at 0, 1, 2, 3 and 4.
    ///
    /// **It is not cascade-break.** The claim that the full budget is only
    /// reachable because pounce perturbs the L-factor does not hold here:
    /// `laptime` with `feral_cascade_break=no` runs 62.113 s against
    /// 61.741 s armed, to a bit-identical objective. The budget is
    /// exhausted because the target is unreachable, not because of CB.
    ///
    /// **Why Ipopt does not have this problem.** It disables
    /// backend-internal refinement on every direct solver it ships: MA27
    /// has no such routine, MA57's `MA57D` is never declared or called
    /// (`IpMa57TSolverInterface.cpp:785` calls only `ma57c`), MUMPS sets
    /// `icntl[9] = 0` under the comment "no iterative refinement
    /// iterations", and Pardiso Project registers a default of 0; only MKL
    /// Pardiso (1) and WSMP (`IPARM(3)=5`) opt in. Wächter-Biegler §3.10
    /// gives the reason — refinement is applied to the **unreduced**
    /// non-symmetric Newton system, because the condensation `Σ = S⁻¹Z`
    /// destroys information as `μ → 0` and a residual measured on the
    /// condensed system is blind to that loss. A backend can only refine
    /// the condensed system it factorized, so its inner loop drives the
    /// wrong residual. `PdFullSpaceSolver::compute_residuals` is the
    /// unreduced one, so pounce already refines the right system in the
    /// right place; the inner loop is redundant *and* aimed at the wrong
    /// target.
    ///
    /// **Why pounce cannot just follow Ipopt and pass 0.** feral's
    /// factorization defaults to `ZeroPivotAction::ForceAccept`, so unlike
    /// MA27/MA57/MUMPS its raw solve can return a non-trivial residual
    /// against the very system it factorized. Ipopt's architecture assumes
    /// a backend hands back an honest solve; feral needs *some* inner
    /// refinement to meet that assumption, which is why 0 loses the
    /// gh#590 badly-scaled LP outright. MA57 under pounce runs with no
    /// inner refinement at all and keeps that LP, which is the control.
    ///
    /// **What the actual fix is.** The inner loop should stop when it
    /// reaches what the host needs — `PdFullSpaceSolver`'s
    /// `residual_ratio_max` of 1e-10 on the unreduced system — rather than
    /// chasing machine precision on the condensed one. feral cannot
    /// express that: `RefineOptions` carries `max_steps` and nothing else,
    /// and the tolerance is hard-wired. That is filed as feral#190, in the
    /// same shape as feral#178 which produced this knob. Until it lands,
    /// [`Self::refine_target`] is pounce's own approximation of it — a
    /// pre-check that decides *whether* to refine rather than for how
    /// long, which is the discrimination this cap cannot make at any
    /// value. Of the caps, 10 remains the only one the corpus does not
    /// reject.
    ///
    /// Cost of leaving it at 10, on the 58014-variable `laptime`
    /// benchmark under limited-memory: `LinearSystemBackSolve` 48.962 s of
    /// a 61.741 s solve, against 7.443 s of 16.728 s at a cap of 0 —
    /// 45 seconds, 73% of wall time, for a residual nobody consults. Lower
    /// it per problem when back-solve dominates the timing report, and
    /// re-check the answer; it is not safe to lower globally.
    pub refine_max_steps: usize,
    /// Residual level at which the inner refinement is skipped entirely,
    /// as a relative 2-norm `‖b − A·x‖₂ / ‖b‖₂` measured on the
    /// *unrefined* solve. `0.0` (the default) disables the check and
    /// every back-solve takes the refined entry point, which is the
    /// behaviour every release through 0.10.0 shipped.
    ///
    /// This exists because feral's refinement has a step cap but no
    /// target: `RefineOptions` carries only `max_steps`, and the
    /// convergence test is hard-wired to `‖r‖₂/‖b‖₂ < ε·√n`
    /// (`feral-0.17.0/src/numeric/solve.rs:2574`). That is the tightest
    /// residual the arithmetic admits — and roughly 300× tighter than
    /// anything the caller reads. `PdFullSpaceSolver` accepts a solve at
    /// `residual_ratio_max = 1e-10`, on the *unreduced* system at that,
    /// so the digits feral works for are discarded on arrival.
    ///
    /// Traced on the 126028-dimension KKT of the `laptime` benchmark
    /// (target `ε·√n` = 7.88e-14), the unrefined solve already lands at
    /// 1.5e-11 and 2.3e-11 — two orders inside what the outer loop asks
    /// for — and the loop then spends four to five steps chasing the last
    /// two digits, at 48.962 s of a 61.741 s solve. One of the two traced
    /// calls never reaches the hard-wired target at all: it plateaus at
    /// 8.19e-14 against 7.88e-14 and exits on the stagnation rule, having
    /// paid two full steps to notice a bit-identical residual.
    ///
    /// So the pre-check: solve once without refinement, measure, and take
    /// the refined path only if the answer is not already good enough.
    /// The cheap path costs one sparse matvec against the four-to-five
    /// matvec-plus-substitution passes it replaces; the expensive path
    /// pays one extra substitution, since the refined entry point redoes
    /// the initial solve internally.
    ///
    /// The upstream fix is feral#190 — a residual target on
    /// `RefineOptions`, so the loop stops when the caller's tolerance is
    /// met rather than when the arithmetic runs out. This field is the
    /// pounce-side prototype that establishes what that number should be;
    /// it goes away when feral#190 lands.
    ///
    /// Not a substitute for [`Self::refine_max_steps`] `= 0`: the check
    /// still runs refinement on the back-solves that need it, which is
    /// what keeps the gh#590 noise-floor LP (data scale 1e11) solving.
    /// Consulted only when [`Self::refine`] is on.
    pub refine_target: f64,
    /// Near-singularity trigger: if the smallest accepted D-block pivot
    /// magnitude `min|λ(D)|` (scaled space) falls below this absolute
    /// floor, `factor()` returns [`ESymSolverStatus::Singular`] even
    /// though feral force-accepted the pivot and reported `Success`.
    /// This is pounce's analog of MA57's `CNTL(2)` small-pivot
    /// threshold — an absolute magnitude on the *scaled* pivot, not a
    /// ratio: a genuinely rank-deficient pivot sits at the working-
    /// precision floor regardless of the rest of the spectrum, whereas
    /// `min/max` ≈ 1/κ(D) collapses on any healthy interior-point KKT
    /// as `μ→0`. Routes into the IPM's `PerturbForSingularity` branch
    /// so `δ_w` is bumped. `0` disables the trigger. See
    /// `dev/research/near-singularity-signal.md` (feral) §4.
    pub singular_pivot_floor: f64,
    /// Floor below which a *negative-eigenvalue count* is treated as
    /// noise rather than as evidence (pounce gh#540). Applies only
    /// when the count already disagrees with what the caller asked
    /// for: if the smallest accepted pivot magnitude (scaled space) is
    /// under this floor, `factor()` reports
    /// [`ESymSolverStatus::Singular`] instead of
    /// [`ESymSolverStatus::WrongInertia`], so the IPM adds `δ_c` to
    /// de-singularize the system before it starts escalating `δ_w`.
    ///
    /// A factorization whose smallest pivot sits at the working-
    /// precision floor of an equilibrated matrix has no reliable sign
    /// on that pivot, so the inertia it reports is not a measurement.
    /// Escalating `δ_w` against such a reading multiplies the Hessian
    /// perturbation by 8 per retry and damps the Newton step to
    /// nothing, while the perturbation that actually repairs a
    /// rank-deficient constraint Jacobian is `δ_c`. This trigger
    /// can only fire on a factorization the caller was already going
    /// to reject, so it never turns a usable factor into a failure —
    /// it only changes *which* perturbation is reached for first.
    /// Necessarily larger than [`Self::singular_pivot_floor`], which
    /// governs factors that are unusable outright.
    ///
    /// `None` — the default — selects the dimension-aware floor
    /// `n * f64::EPSILON` computed by [`inertia_trust_floor`], where
    /// `n` is the order of the factored matrix. `Some(v)` pins an
    /// absolute floor for every dimension; `Some(0.0)` disables the
    /// trigger.
    pub inertia_pivot_floor: Option<f64>,
    /// Relative Bunch-Kaufman partial-pivoting threshold `u`: a
    /// candidate diagonal pivot is rejected when `|d| < u * col_max`.
    /// Direct analog of Ipopt's `ma27_pivtol` / `ma57_pivtol`. Smaller
    /// `u` preserves the AMD ordering and keeps `L` sparse; larger
    /// `u` rejects more candidates, delaying pivots / forcing 2x2
    /// blocks for accuracy. LAPACK's textbook maximum-stability value
    /// is `0.5`. Default `1e-8` matches feral's `NumericParams`.
    pub pivtol: f64,
    /// Fill-reducing ordering method passed to
    /// [`feral::Solver::with_ordering`]. Default
    /// [`OrderingMethod::Auto`]: the adaptive dispatcher picks a
    /// concrete method per matrix from cheap pattern features (very-
    /// large-and-sparse → AMD; `n ≤ 10 000` → AMF; otherwise →
    /// MetisND). Override via the `feral_ordering` OptionsList option
    /// or the `POUNCE_FERAL_ORDERING` env var when a specific
    /// concrete method (`amd`, `amf`, `metis`, `scotch`, `kahip`) or
    /// the symbolic-time race (`auto_race`) is wanted. See
    /// `feral/src/symbolic/mod.rs::OrderingMethod` for the
    /// per-variant rationale.
    pub ordering: OrderingMethod,
    /// Diagonal scaling strategy passed to
    /// [`feral::Solver::with_scaling`]. Default
    /// [`ScalingStrategy::Auto`]: FERAL's adaptive shape-based router
    /// picks `Mc64Symmetric` on arrow-KKT signatures and `InfNorm`
    /// otherwise. Override via the `feral_scaling` OptionsList option
    /// or the `POUNCE_FERAL_SCALING` env var. The opt-in choices are
    /// `infnorm` (Knight-Ruiz ∞-norm equilibration), `mc64`
    /// (MC64-style symmetric matching — MUMPS SYM=2 / SSIDS default,
    /// recovers exact inertia on some ill-conditioned KKTs where
    /// `Auto` mis-pivots; see feral#65), and `identity` (no-op, for
    /// regression testing). See `feral/src/scaling/mod.rs::
    /// ScalingStrategy` for the per-variant rationale.
    pub scaling: ScalingStrategy,
    /// Per-backend internal-parallelism toggle (tri-state). `None` (the
    /// default) leaves feral's `Solver` at its own default and lets the
    /// legacy `FERAL_PARALLEL` env var force it either way (`1|on|true|yes`
    /// / `0|off|false|no`); `Some(false)`
    /// builds an explicitly **serial** factor; `Some(true)` forces feral's
    /// internal rayon parallelism on. This is the first-class lever for
    /// outer-parallel / inner-serial batched solving — each rayon worker
    /// builds its own `Some(false)` backend, with no global state (pounce
    /// issue #79). feral reads `Solver::use_parallel` fresh on every
    /// `factor()`, so two backends with different settings never interfere.
    pub parallel: Option<bool>,
    /// FERAL's parallel-dispatch flop gate (`min_parallel_flops`,
    /// feral#19). A supernode tree is only handed to rayon once its
    /// estimated flop count clears this threshold. `None` (the pounce
    /// default) inherits feral's `NumericParams::default()` value
    /// (10^8). `Some(0)` fires the gate on every multi-child tree ≥
    /// `N_PAR_MIN` supernodes; `Some(u64::MAX)` rejects all
    /// tree-level parallel dispatch. Overridable via the
    /// `feral_min_par_flops` OptionsList option or the
    /// `POUNCE_FERAL_MIN_PAR_FLOPS` env var.
    pub min_par_flops: Option<u64>,
    /// FERAL static-pivoting toggle (tri-state), i.e.
    /// `NumericParams::allow_delayed_pivots = false` via
    /// [`feral::Solver::with_static_pivoting`]. `None` (default) inherits
    /// feral's SSIDS-style delayed pivoting. `Some(true)` runs every
    /// supernode as the root does — a pivot that fails the threshold is
    /// force-accepted in place (`ZeroPivotAction::ForceAccept`) with
    /// iterative refinement recovering the residual, instead of being
    /// delayed up the elimination tree. This is feral's analogue of
    /// MA57's `cntl[4]` static-pivoting fallback and the fast path out of
    /// the delayed-pivot cascade of feral#8 (pinene_3200: an 87 s factor
    /// on an otherwise sub-second problem). `Some(false)` explicitly keeps
    /// delayed pivoting on.
    ///
    /// Deliberately an **explicit, opt-in** knob (`feral_static_pivoting`
    /// OptionsList option / `POUNCE_FERAL_STATIC_PIVOTING` env var), never
    /// triggered implicitly by a tight `max_wall_time`: a budget-coupled
    /// numeric switch would make a solve's result — and, for a
    /// branch-and-bound host, a node's dual bound — depend on the clock,
    /// so the accuracy/speed trade is the caller's to make deliberately.
    /// The empirical motivation is pounce#254 (emfl050's ~44 s single
    /// factorization); see `dev-notes/feral-factor-interrupt.md`.
    pub static_pivoting: Option<bool>,
}

impl Default for FeralConfig {
    fn default() -> Self {
        Self {
            cascade_break: None,
            fma: false,
            // On -- see the field doc. The NLP solver turns it off in
            // `feral_config_from_options` because it refines the
            // unreduced system itself *and* escalates through
            // `increase_quality`; a caller doing only the first still
            // needs this.
            refine: true,
            increase_quality: true,
            refine_max_steps: feral::DEFAULT_REFINE_MAX_STEPS,
            // Disabled: every back-solve refines, as it always has.
            refine_target: 0.0,
            // MA57 `CNTL(2)` default — an absolute small-pivot
            // magnitude on the scaled matrix. Only pivots essentially
            // at the working-precision floor are flagged singular.
            singular_pivot_floor: 1e-20,
            // Dimension-aware: `n · eps`, the level at which a pivot
            // of an equilibrated matrix of order `n` stops carrying a
            // trustworthy sign. Only consulted when the inertia
            // already mismatched (pounce gh#540, gh#592).
            inertia_pivot_floor: None,
            pivtol: 1e-8,
            ordering: OrderingMethod::Auto,
            scaling: ScalingStrategy::Auto,
            parallel: None,
            min_par_flops: None,
            static_pivoting: None,
        }
    }
}

/// Parse a boolean env value in the grammar every knob in this file shares:
/// `1|on|true|yes` → `Some(true)`, `0|off|false|no` → `Some(false)`, and
/// anything else (including unset) → `None`, meaning "leave the default
/// alone". Kept as a pure helper so the vocabulary is unit-testable without
/// mutating the process environment (a data race under rayon-parallel
/// solves — the hazard behind L9/L12).
fn parse_bool_env(v: Option<&str>) -> Option<bool> {
    match v {
        Some("1") | Some("on") | Some("true") | Some("yes") => Some(true),
        Some("0") | Some("off") | Some("false") | Some("no") => Some(false),
        _ => None,
    }
}

/// Resolve the feral pivot-threshold env override from its two accepted
/// variable names. The documented `POUNCE_FERAL_PIVTOL` — the
/// `POUNCE_FERAL_*` convention shared by every other knob in
/// [`FeralConfig::from_env`] — takes precedence; the bare `FERAL_PIVTOL` is
/// retained as a deprecated legacy alias for back-compatibility. An unset or
/// unparseable value falls through to the next source, and with neither set
/// to the `1e-8` default (matching [`FeralConfig::default`]).
fn resolve_pivtol_env(pounce: Option<&str>, legacy: Option<&str>) -> f64 {
    pounce
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| legacy.and_then(|s| s.parse::<f64>().ok()))
        .unwrap_or(1e-8)
}

impl FeralConfig {
    /// Read the knobs from `POUNCE_FERAL_CASCADE_BREAK`,
    /// `POUNCE_FERAL_FMA`, `POUNCE_FERAL_REFINE`,
    /// `POUNCE_FERAL_SINGULAR_PIVOT_FLOOR`, `POUNCE_FERAL_PIVTOL`,
    /// `POUNCE_FERAL_ORDERING`, `POUNCE_FERAL_SCALING`,
    /// `POUNCE_FERAL_MIN_PAR_FLOPS` environment
    /// variables. Used as a fallback when the IPM has no `OptionsList` to
    /// consult (tests, legacy callers). The pivot threshold also accepts the
    /// deprecated bare `FERAL_PIVTOL` as a legacy alias (see
    /// [`resolve_pivtol_env`]).
    pub fn from_env() -> Self {
        Self {
            cascade_break: parse_bool_env(
                std::env::var("POUNCE_FERAL_CASCADE_BREAK").ok().as_deref(),
            ),
            fma: parse_bool_env(std::env::var("POUNCE_FERAL_FMA").ok().as_deref()).unwrap_or(false),
            refine: parse_bool_env(std::env::var("POUNCE_FERAL_REFINE").ok().as_deref())
                .unwrap_or(true),
            increase_quality: parse_bool_env(
                std::env::var("POUNCE_FERAL_INCREASE_QUALITY")
                    .ok()
                    .as_deref(),
            )
            .unwrap_or(true),
            // `feral::env` rather than a local parse: see the
            // `min_par_flops` note below for why every numeric knob here
            // goes through it now.
            refine_max_steps: feral::env::usize_var("POUNCE_FERAL_REFINE_STEPS")
                .unwrap_or(feral::DEFAULT_REFINE_MAX_STEPS),
            refine_target: feral::env::f64_var_where(
                "POUNCE_FERAL_REFINE_TARGET",
                ">= 0 and finite",
                |v| v >= 0.0,
            )
            .unwrap_or(0.0),
            singular_pivot_floor: feral::env::f64_var_where(
                "POUNCE_FERAL_SINGULAR_PIVOT_FLOOR",
                ">= 0 and finite",
                |v| v >= 0.0,
            )
            .unwrap_or(1e-20),
            inertia_pivot_floor: feral::env::f64_var_where(
                "POUNCE_FERAL_INERTIA_PIVOT_FLOOR",
                ">= 0 and finite",
                |v| v >= 0.0,
            ),
            pivtol: resolve_pivtol_env(
                std::env::var("POUNCE_FERAL_PIVTOL").ok().as_deref(),
                std::env::var("FERAL_PIVTOL").ok().as_deref(),
            ),
            ordering: std::env::var("POUNCE_FERAL_ORDERING")
                .ok()
                .as_deref()
                .and_then(parse_ordering_method)
                .unwrap_or(OrderingMethod::Auto),
            scaling: std::env::var("POUNCE_FERAL_SCALING")
                .ok()
                .as_deref()
                .and_then(parse_scaling_strategy)
                .unwrap_or(ScalingStrategy::Auto),
            // Left `None` so the legacy `FERAL_PARALLEL` env var still acts
            // as the fallback on/off switch in `with_config`; callers that
            // want an explicit per-backend setting use `FeralConfig.parallel`
            // directly (e.g. `FeralSolverInterface::serial`).
            parallel: None,
            // `None` inherits feral's built-in `min_parallel_flops`
            // default; an unset or unparseable env value falls through
            // to it. A `u64` so the `u64::MAX` reject-all sentinel and
            // large flop counts stay representable.
            //
            // Read through `feral::env::u64_var`, not `parse::<u64>()`.
            // `"1e8".parse::<u64>()` fails — scientific notation is not an
            // integer literal — so the documented default for this very
            // knob, which `docs/src/options.md` and the option help both
            // print as `1e8`, was silently discarded here and replaced by
            // feral's built-in. The option spelling
            // (`feral_min_par_flops`) is registered as a *number* option
            // and accepted `1e8` all along, so the same knob accepted a
            // value one way and dropped it the other, without a word.
            // This is feral#176's defect, in pounce; `feral::env` is the
            // policy that release made public precisely so a caller
            // reading its own knobs gets the same rules — scientific
            // notation accepted, an over-range magnitude clamped rather
            // than dropped (it means "switch this off"), and anything
            // refused warned about once instead of vanishing.
            min_par_flops: feral::env::u64_var("POUNCE_FERAL_MIN_PAR_FLOPS"),
            // Tri-state, mirroring `cascade_break`: an unset or
            // unrecognized value leaves `None` (inherit feral's
            // delayed-pivot default); only an explicit on/off token forces
            // the override.
            static_pivoting: parse_bool_env(
                std::env::var("POUNCE_FERAL_STATIC_PIVOTING")
                    .ok()
                    .as_deref(),
            ),
        }
    }
}

/// Parse a case-insensitive ordering tag (the values accepted by the
/// `feral_ordering` OptionsList option and the `POUNCE_FERAL_ORDERING`
/// env var) into the corresponding [`OrderingMethod`]. Returns `None`
/// for unrecognized tags so the caller can fall back to the default.
pub fn parse_ordering_method(s: &str) -> Option<OrderingMethod> {
    match s.trim().to_ascii_lowercase().as_str() {
        "auto" => Some(OrderingMethod::Auto),
        "auto_race" | "autorace" | "race" => Some(OrderingMethod::AutoRace),
        "amd" => Some(OrderingMethod::Amd),
        "amf" => Some(OrderingMethod::Amf),
        "metis" | "metis_nd" | "metisnd" => Some(OrderingMethod::MetisND),
        "scotch" | "scotch_nd" | "scotchnd" => Some(OrderingMethod::ScotchND),
        "kahip" | "kahip_nd" | "kahipnd" => Some(OrderingMethod::KahipND),
        _ => None,
    }
}

/// Build a configured `feral::Solver` from a [`FeralConfig`]. Extracted from
/// [`FeralSolverInterface::with_config`] so the Schur backend
/// ([`crate::schur::FeralSchurSolver`]) configures its per-block solvers
/// identically (same pivot threshold, cascade-break, parallelism, ordering,
/// scaling) as the monolithic path.
pub(crate) fn configure_solver(cfg: &FeralConfig) -> Solver {
    // FERAL's parallel-dispatch flop gate (`feral_min_par_flops` option /
    // `POUNCE_FERAL_MIN_PAR_FLOPS` env, resolved into `cfg.min_par_flops`).
    // feral#19, default 10^8. `Some(0)` fires the gate on every multi-child
    // tree ≥ N_PAR_MIN supernodes; `Some(u64::MAX)` rejects all parallel
    // dispatch at the tree level. `None` inherits feral's own default.
    let mut np = NumericParams::default();
    if let Some(v) = cfg.min_par_flops {
        np.min_parallel_flops = Some(v);
    }
    // Relative Bunch-Kaufman partial-pivoting threshold — the analog of
    // Ipopt's `ma27_pivtol` / `ma57_pivtol` (`feral_pivtol` option /
    // `POUNCE_FERAL_PIVTOL` env, read in `FeralConfig::from_env`).
    np.bk.pivot_threshold = cfg.pivtol;
    // Cascade-break (FERAL issue #55 Phase B). `NumericParams::default()`
    // arms CB; the tri-state `cfg.cascade_break` only intervenes on an
    // explicit set: None inherits the FERAL default (on), Some(true)
    // re-arms explicitly, Some(false) disarms (reproduces pre-Phase-B).
    match cfg.cascade_break {
        None => {}
        Some(true) => {
            np.cascade_break_ratio = Some(0.5);
            np.cascade_break_eps = Some(1e-10);
        }
        Some(false) => {
            np.cascade_break_ratio = None;
            np.cascade_break_eps = None;
        }
    }
    let mut solver = Solver::with_params(np, SupernodeParams::default());
    // Internal-parallelism toggle. Explicit `cfg.parallel` is the primary
    // per-backend lever; when unset, fall back to the legacy process-wide
    // `FERAL_PARALLEL` env var. The env var is bidirectional and uses the
    // same `1|on|true|yes` / `0|off|false|no` grammar as every other knob
    // here and as feral's own C-ABI shim (`feral/src/capi.rs`) — an
    // unset or unrecognized value leaves feral's default in place. The
    // force-*on* direction matters because feral derives its default from
    // the platform and falls back to sequential when the rayon pool fails
    // to build (feral#156): on hosts where that autodetection is wrong
    // (threaded wasm), `FERAL_PARALLEL=1` is the only escape hatch a
    // CLI/Python/NL caller has — `FeralConfig.parallel` is Rust-API-only.
    match cfg.parallel {
        Some(p) => solver = solver.with_parallel(p),
        None => {
            if let Some(p) = parse_bool_env(std::env::var("FERAL_PARALLEL").ok().as_deref()) {
                solver = solver.with_parallel(p);
            }
        }
    }
    if cfg.fma {
        solver = solver.with_fma(true);
    }
    // Static pivoting (feral#8 delayed-pivot-cascade breaker;
    // `feral_static_pivoting` option / `POUNCE_FERAL_STATIC_PIVOTING` env).
    // Only intervene on an explicit set — `None` inherits feral's
    // delayed-pivot default. `Some(true)` disables delayed pivoting
    // (`allow_delayed_pivots = false`) so a failing pivot is force-accepted
    // in place with refinement recovering the residual, keeping a
    // pathological factorization cheap. See pounce#254.
    if let Some(on) = cfg.static_pivoting {
        solver = solver.with_static_pivoting(on);
    }
    // Fill-reducing ordering (`feral_ordering` / `POUNCE_FERAL_ORDERING`),
    // and diagonal scaling (`feral_scaling` / `POUNCE_FERAL_SCALING`).
    // `.clone()` since FERAL 0.13's `OrderingMethod` / `ScalingStrategy`
    // carry heap `External` variants and are no longer `Copy`.
    solver = solver.with_ordering(cfg.ordering.clone());
    solver = solver.with_scaling(cfg.scaling.clone());
    solver
}

/// Parse a case-insensitive scaling tag (the values accepted by the
/// `feral_scaling` OptionsList option and the `POUNCE_FERAL_SCALING`
/// env var) into the corresponding [`ScalingStrategy`]. Returns `None`
/// for unrecognized tags (and for `external`, which carries a vector
/// that cannot be supplied via a string option) so the caller can fall
/// back to the default.
pub fn parse_scaling_strategy(s: &str) -> Option<ScalingStrategy> {
    match s.trim().to_ascii_lowercase().as_str() {
        "auto" => Some(ScalingStrategy::Auto),
        "infnorm" | "inf_norm" | "inf" => Some(ScalingStrategy::InfNorm),
        "mc64" | "mc64symmetric" | "mc64_symmetric" => Some(ScalingStrategy::Mc64Symmetric),
        "identity" | "none" => Some(ScalingStrategy::Identity),
        _ => None,
    }
}

/// Record where each triplet landed in a matrix just built from it by
/// [`CscMatrix::from_triplets`] — `slot[k]` is the index into
/// `matrix.values` holding triplet `k`'s contribution (gh#562).
///
/// `from_triplets` leaves each column's `row_idx` sorted and duplicate-free,
/// so the slot is a binary search inside the column's range. Returns `None`
/// if any triplet cannot be located, which is a contradiction of
/// `from_triplets`'s own postcondition; the caller then keeps rebuilding
/// from scratch, so an unexpected feral-side change degrades to the old
/// behavior rather than to a wrong matrix.
fn slot_map(matrix: &CscMatrix, rows: &[usize], cols: &[usize]) -> Option<Vec<usize>> {
    let mut slot = Vec::with_capacity(rows.len());
    for (&r, &c) in rows.iter().zip(cols.iter()) {
        let start = *matrix.col_ptr.get(c)?;
        let end = *matrix.col_ptr.get(c + 1)?;
        let within = matrix.row_idx.get(start..end)?.binary_search(&r).ok()?;
        slot.push(start + within);
    }
    Some(slot)
}

impl FeralSolverInterface {
    /// Construct with config read from environment variables. Retained
    /// for legacy callers (tests, anything without an IPM options
    /// list). Prefer [`Self::with_config`] from option-aware sites so
    /// the `.opt`-file knobs take effect.
    pub fn new() -> Self {
        Self::with_config(FeralConfig::from_env())
    }

    /// Construct a backend with feral's internal parallelism **disabled**
    /// (inheriting all other env-driven config). Each rayon worker in an
    /// outer-parallel / inner-serial batch builds one of these directly, so
    /// the only parallelism is across instances — no global `FERAL_PARALLEL`
    /// mutation (pounce issue #79).
    pub fn serial() -> Self {
        Self::with_config(FeralConfig {
            parallel: Some(false),
            ..FeralConfig::from_env()
        })
    }

    /// Construct with explicit configuration. Cascade-break
    /// (`ratio=0.5, eps=1e-10`) was off by default in pounce for a
    /// period after the issue-17/issue-18 inertia investigations,
    /// when the FERAL Bunch-Kaufman heuristic could not bound the
    /// per-supernode delayed-pivot catchment and spurious
    /// `WrongInertia` returns on borderline iterates (robot_1600
    /// iter-3, NARX_CFy iters 1+, ~250 spurious records — feral
    /// journal 2026-05-16 21:30) cost more than CB's per-factor
    /// speedup (pinene_3200_0009: 33 ms cb-on vs 94 s cb-off).
    /// FERAL issue #55 Phase B (commit 7554a78) bounds the catchment
    /// at symbolic-analysis time and arms CB out of the box, so
    /// pounce now inherits the FERAL default (CB on) when the
    /// `feral_cascade_break` option is left unset. See
    /// [`FeralConfig::cascade_break`] for the tri-state semantics.
    pub fn with_config(cfg: FeralConfig) -> Self {
        let solver = configure_solver(&cfg);
        Self {
            solver,
            initialized: false,
            pivtol_changed: false,
            refactorize: false,
            refine: cfg.refine,
            increase_quality: cfg.increase_quality,
            refine_max_steps: cfg.refine_max_steps,
            refine_target: cfg.refine_target,
            x_scratch: Vec::new(),
            resid_scratch: Vec::new(),
            dim: 0,
            nonzeros: 0,
            rows_0: Vec::new(),
            cols_0: Vec::new(),
            values: Vec::new(),
            matrix: None,
            slot: None,
            negevals: 0,
            ordering: cfg.ordering,
            singular_pivot_floor: cfg.singular_pivot_floor,
            inertia_pivot_floor: cfg.inertia_pivot_floor,
            summary: LinearSolverSummary {
                solver_name: "feral".to_string(),
                ..Default::default()
            },
            sink: None,
        }
    }

    /// Install a shared summary sink. The interface updates the sink
    /// (and the internal `summary`) after every successful
    /// `factor()`. Default is no sink — calls then go only to the
    /// internal `summary`.
    pub fn with_summary_sink(mut self, sink: Arc<Mutex<LinearSolverSummary>>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Snapshot of the post-solve aggregate. Always populated (no
    /// opt-in needed for the always-on Phase A stats from feral 0.7).
    pub fn summary(&self) -> LinearSolverSummary {
        self.summary.clone()
    }

    /// Fold a single feral `FactorStats` into the running summary,
    /// then mirror the snapshot into the sink if one is installed.
    fn record_factor_stats(&mut self, stats: FactorStats) {
        let s = &mut self.summary;
        s.n_factors += 1;
        if stats.pattern_reused {
            s.n_pattern_reuse += 1;
        } else {
            s.n_pattern_changes += 1;
        }
        s.max_fill_ratio = Some(match s.max_fill_ratio {
            Some(prev) => prev.max(stats.fill_ratio),
            None => stats.fill_ratio,
        });
        s.min_abs_pivot = Some(match s.min_abs_pivot {
            Some(prev) => prev.min(stats.min_abs_pivot),
            None => stats.min_abs_pivot,
        });
        s.max_abs_pivot = Some(match s.max_abs_pivot {
            Some(prev) => prev.max(stats.max_abs_pivot),
            None => stats.max_abs_pivot,
        });
        s.last_inertia = Some((
            stats.inertia.positive,
            stats.inertia.negative,
            stats.inertia.zero,
        ));
        s.last_nnz_a = Some(stats.nnz_a);
        s.last_nnz_l = Some(stats.nnz_l);

        if let Some(sink) = self.sink.as_ref() {
            if let Ok(mut guard) = sink.lock() {
                *guard = s.clone();
            }
        }

        // Surface the linear-solve characteristics on the enclosing
        // `linear_solve` tracing span (pounce#71). A no-op unless that
        // span is active and declared these fields, so non-IPM callers
        // and the no-subscriber case pay nothing. Re-factorizations
        // (regularization retries) overwrite with last-wins, so the
        // span reflects the accepted factorization.
        let span = tracing::Span::current();
        span.record("n", self.dim);
        span.record("matrix_nnz", stats.nnz_a);
        span.record("factor_nnz", stats.nnz_l);
        span.record("inertia_neg", stats.inertia.negative);
        span.record("fill_ratio", stats.fill_ratio);
        // Borrow rather than move: `OrderingMethod` is no longer `Copy`
        // (FERAL 0.13). FERAL's hand-written `Debug` prints the `External`
        // arm compactly as `External { len: N }`, so this stays a
        // one-liner even for a caller-supplied permutation.
        span.record("ordering", tracing::field::debug(&self.ordering));
    }

    /// Build the CSC view of the caller-filled triplets, ready to hand to
    /// `feral`. The first call after an `initialize_structure` goes through
    /// [`CscMatrix::from_triplets`] and records the triplet → slot
    /// permutation; every later call replays that permutation into the
    /// retained matrix (gh#562).
    ///
    /// Returns `false` on a structurally invalid triplet set (the
    /// `from_triplets` error paths: out-of-range index, upper-triangle
    /// entry, length mismatch), leaving `self.matrix` untouched.
    fn refresh_matrix(&mut self) -> bool {
        let n = self.dim as usize;

        // Fast path: the pattern is unchanged, so `col_ptr` / `row_idx`
        // already hold the answer `from_triplets` would recompute. Zero the
        // values and scatter the caller's triplets through the cached
        // permutation. The `+=` (rather than a plain store) reproduces
        // `from_triplets`'s duplicate-summing semantics for the case where
        // several triplets share one (row, col) slot.
        if let (Some(slot), Some(matrix)) = (self.slot.as_ref(), self.matrix.as_mut()) {
            debug_assert_eq!(slot.len(), self.values.len());
            debug_assert_eq!(matrix.n, n);
            matrix.values.fill(0.0);
            for (&s, &v) in slot.iter().zip(self.values.iter()) {
                matrix.values[s] += v;
            }
            // Correctness net: in debug builds redo the full rebuild and
            // check that the scatter landed on the same structure and the
            // same numbers. This is also the assertion that the pattern
            // really is immutable — a mutation of `rows_0` / `cols_0` behind
            // `initialize_structure`'s back shows up here as a `col_ptr` /
            // `row_idx` mismatch rather than as silent wrong arithmetic.
            #[cfg(debug_assertions)]
            self.assert_refill_matches_rebuild();
            return true;
        }

        // Hand the KKT to feral with its structure intact. Where a
        // constraint multiplier is zero (e.g. the initial point) the
        // (2,2) diagonal lands as structurally-present exact `0.0`
        // values; feral handles those explicit zeros correctly and
        // without a delayed-pivot penalty, so pounce must NOT strip
        // them — dropping them leaves the constraint columns with no
        // diagonal, which is the structurally-absent-(2,2) cascade
        // (feral#46) the strip was meant to avoid. The permutation
        // replay above inherits that property by construction: it keeps
        // every slot this build created, whatever the values do later.
        let matrix = match CscMatrix::from_triplets(n, &self.rows_0, &self.cols_0, &self.values) {
            Ok(m) => m,
            Err(_) => return false,
        };
        self.slot = slot_map(&matrix, &self.rows_0, &self.cols_0);
        self.matrix = Some(matrix);
        true
    }

    /// Debug-only cross-check of the [`Self::refresh_matrix`] fast path
    /// against a fresh [`CscMatrix::from_triplets`].
    ///
    /// Values are compared with a relative tolerance rather than bit-wise:
    /// where duplicate triplets are summed, the rebuild adds them in
    /// row-sorted order and the scatter adds them in triplet order, and
    /// floating-point addition is not associative. The structure
    /// (`col_ptr`, `row_idx`) is compared exactly — that is the part the
    /// permutation is asserting.
    #[cfg(debug_assertions)]
    fn assert_refill_matches_rebuild(&self) {
        let matrix = self.matrix.as_ref().expect("fast path holds a matrix");
        let rebuilt =
            CscMatrix::from_triplets(self.dim as usize, &self.rows_0, &self.cols_0, &self.values)
                .expect("pattern factored once already, so the triplets are structurally valid");
        assert_eq!(
            matrix.col_ptr, rebuilt.col_ptr,
            "cached CSC slot permutation is stale: col_ptr changed since \
             initialize_structure (gh#562)"
        );
        assert_eq!(
            matrix.row_idx, rebuilt.row_idx,
            "cached CSC slot permutation is stale: row_idx changed since \
             initialize_structure (gh#562)"
        );
        for (i, (&got, &want)) in matrix.values.iter().zip(rebuilt.values.iter()).enumerate() {
            if got == want || (got.is_nan() && want.is_nan()) {
                continue;
            }
            let scale = 1.0f64.max(got.abs()).max(want.abs());
            assert!(
                (got - want).abs() <= 1e-12 * scale,
                "refilled value at CSC slot {i} is {got}, rebuild says {want} (gh#562)"
            );
        }
    }

    /// Build the CSC view, factor it, and stash the
    /// strict-negative-eigenvalue count (IPOPT / MA57 `INFO(24)`
    /// convention). Rank deficiency (zero pivots) is reported as
    /// `Singular` so the outer loop routes to `perturb_for_singular`.
    fn factor(&mut self, check_neg_evals: bool, number_of_neg_evals: Index) -> ESymSolverStatus {
        // The matrix is retained across calls for refinement in backsolve
        // and as the refill destination, so it is in place before the
        // factorization rather than after it — the caller may still issue
        // solves against a stale factor in some restart paths, and this
        // keeps that matrix consistent with the values just supplied
        // regardless of the factor outcome.
        if !self.refresh_matrix() {
            return ESymSolverStatus::FatalError;
        }
        let matrix = self.matrix.as_ref().expect("refresh_matrix stored one");

        let status = self.solver.factor(matrix, None);
        match status {
            FactorStatus::Success => {
                if let Some(stats) = self.solver.last_factor_stats() {
                    self.record_factor_stats(stats);
                }
                // IPOPT / MA57 convention: `number_of_neg_evals` is the
                // count of strict negative pivots (MA57's INFO(24)). Zero
                // pivots are reported separately by signalling `Singular`,
                // which routes the outer loop to `perturb_for_singular`
                // (bumping δ_c on rank-deficient constraint rows) instead
                // of `perturb_for_wrong_inertia` (bumping δ_x). Folding
                // zero into negevals — the SSIDS bookkeeping convention —
                // is correct for spectral accounting but breaks IPOPT's
                // singularity branch on LP-shaped KKTs whose (3,3) block
                // is structurally zero. See pounce gh#52 / feral gh#54.
                let (neg, zero) = match self.solver.inertia() {
                    Some(i) => (i.negative, i.zero),
                    None => (self.solver.num_negative_eigenvalues(), 0),
                };
                self.negevals = neg as Index;
                if zero > 0 {
                    tracing::debug!(
                        target: "pounce::linsol",
                        neg, zero, expected = number_of_neg_evals, dim = self.dim,
                        "inertia singular"
                    );
                    return ESymSolverStatus::Singular;
                }
                if check_neg_evals && self.negevals != number_of_neg_evals {
                    // pounce gh#540: a count read off a factorization whose
                    // smallest pivot sits at the working-precision floor is
                    // noise, not a measurement — on eigena2 the same
                    // factorization returns 64/58/62 negatives for the same
                    // 55 the caller wants, and LAPACK's exact count on the
                    // dumped matrices agrees with none of them. Reporting
                    // `WrongInertia` sends the IPM up the `δ_w` ladder (×8 per
                    // retry), which damps the Newton step to nothing; the
                    // perturbation that actually repairs the rank-deficient
                    // constraint Jacobian underneath is `δ_c`, which is what
                    // `Singular` reaches for. Upstream's MA27/MA57/MUMPS
                    // interfaces likewise test singularity *before* comparing
                    // the count.
                    let floor = inertia_trust_floor(self.inertia_pivot_floor, self.dim as usize);
                    if floor > 0.0 {
                        if let Some(min_piv) = self.solver.min_pivot_magnitude() {
                            if min_piv < floor {
                                tracing::debug!(
                                    target: "pounce::linsol",
                                    got_neg = self.negevals, expected = number_of_neg_evals,
                                    min_piv, floor, dim = self.dim,
                                    "inertia untrustworthy; reporting singular"
                                );
                                return ESymSolverStatus::Singular;
                            }
                        }
                    }
                    tracing::debug!(
                        target: "pounce::linsol",
                        got_neg = self.negevals, expected = number_of_neg_evals, dim = self.dim,
                        min_piv = self.solver.min_pivot_magnitude(), floor,
                        "inertia mismatch"
                    );
                    return ESymSolverStatus::WrongInertia;
                }
                // Near-singularity (MA57 CNTL(2) analog). feral's default
                // `ZeroPivotAction::ForceAccept` completes the factorization
                // and reports `Success` even on a pivot at the working-
                // precision floor. We flag `Singular` only when the smallest
                // accepted D-block pivot magnitude drops below an absolute
                // floor — the literal `CNTL(2)` quantity. A ratio test
                // `min/max` ≈ 1/κ(D) is wrong here: an interior-point KKT
                // is *designed* to become ill-conditioned as `μ→0`, so the
                // ratio collapses on healthy full-rank systems near the
                // solution. The absolute floor moves with neither `μ` nor
                // the spectral spread. See
                // `dev/research/near-singularity-signal.md` (feral) §4.
                if self.singular_pivot_floor > 0.0 {
                    if let Some(min_piv) = self.solver.min_pivot_magnitude() {
                        if min_piv < self.singular_pivot_floor {
                            return ESymSolverStatus::Singular;
                        }
                    }
                }
                ESymSolverStatus::Success
            }
            FactorStatus::Singular => ESymSolverStatus::Singular,
            FactorStatus::WrongInertia { .. } => {
                // Should not occur — we passed `None` for check_inertia.
                ESymSolverStatus::FatalError
            }
            FactorStatus::FatalError(_) => ESymSolverStatus::FatalError,
        }
    }

    fn backsolve(&mut self, nrhs: Index, rhs_vals: &mut [Number]) -> ESymSolverStatus {
        let n = self.dim as usize;
        let nrhs = nrhs as usize;
        debug_assert_eq!(rhs_vals.len(), n * nrhs);

        // feral#178 shipped the `_into` forms in 0.17.0, so the owned
        // `Vec` every entry point used to return — and the allocation
        // behind it — is gone. `x_scratch` is grown once and reused;
        // `resize` is a no-op after the first call at a given shape.
        self.x_scratch.resize(n * nrhs, 0.0);
        let x_out = &mut self.x_scratch[..n * nrhs];

        // See `FeralConfig::refine` and `refine_max_steps`: under the IPM
        // this inner refinement is nested inside `PdFullSpaceSolver`'s own
        // loop (gh#698, gh#710), so the budget is ours to set rather than
        // feral's to assume.
        let opts = RefineOptions::with_max_steps(self.refine_max_steps);
        // `FeralConfig::refine_target`: solve once without refinement and
        // measure, so the refined entry point is only reached on the
        // back-solves that are not already inside the caller's tolerance.
        // feral's own convergence test cannot do this — `RefineOptions`
        // has no target, only a step cap (feral#190).
        if self.refine
            && self.refine_target > 0.0
            && let Some(m) = self.matrix.as_ref()
        {
            let first = if nrhs == 1 {
                self.solver.solve_into(rhs_vals, x_out)
            } else {
                self.solver.solve_many_into(rhs_vals, nrhs, x_out)
            };
            match first {
                Err(_) => return ESymSolverStatus::FatalError,
                Ok(()) => {
                    self.resid_scratch.resize(n, 0.0);
                    // Every column has to clear the target: a back-solve
                    // is one solve as far as the caller is concerned, and
                    // refinement is all-or-nothing per call.
                    let good = (0..nrhs).all(|col| {
                        let lo = col * n;
                        relative_residual(
                            m,
                            &rhs_vals[lo..lo + n],
                            &x_out[lo..lo + n],
                            &mut self.resid_scratch,
                        )
                        .is_some_and(|r| r <= self.refine_target)
                    });
                    if good {
                        rhs_vals.copy_from_slice(x_out);
                        return ESymSolverStatus::Success;
                    }
                }
            }
            // Fall through: the unrefined answer is not good enough, so
            // pay for the refined path. It redoes the initial solve
            // internally — one wasted substitution on this branch, against
            // the four-to-five passes the branch above skips entirely.
        }

        let solved = match (self.refine, self.matrix.as_ref(), nrhs == 1) {
            (true, Some(m), true) => self.solver.solve_refined_into(m, rhs_vals, x_out, opts),
            (true, Some(m), false) => self
                .solver
                .solve_many_refined_into(m, rhs_vals, nrhs, x_out, opts),
            (_, _, true) => self.solver.solve_into(rhs_vals, x_out),
            (_, _, false) => self.solver.solve_many_into(rhs_vals, nrhs, x_out),
        };
        match solved {
            Ok(()) => {
                // The copy back stays: feral makes aliasing `rhs` with
                // `x_out` unrepresentable in safe Rust, so the solve
                // cannot write through the caller's slice directly.
                rhs_vals.copy_from_slice(x_out);
                ESymSolverStatus::Success
            }
            Err(_) => ESymSolverStatus::FatalError,
        }
    }
}

/// Relative residual `‖b − A·x‖₂ / ‖b‖₂` of a single solve, for the
/// [`FeralConfig::refine_target`] pre-check.
///
/// `a` holds the **lower** triangle of a symmetric matrix — that is what
/// `factor()` builds and what feral's refinement is handed — so each
/// stored entry off the diagonal contributes to two rows of the product.
///
/// `scratch` is the residual accumulator, sized `a.n` by the caller;
/// taken as a parameter so the hot path allocates nothing.
///
/// Returns `None` when `‖b‖₂` is zero (the ratio is undefined and there
/// is nothing to refine) or when any norm is not finite, which reads as
/// "cannot certify this solve" and sends the caller down the refined
/// path.
fn relative_residual(
    a: &CscMatrix,
    b: &[Number],
    x: &[Number],
    scratch: &mut [Number],
) -> Option<f64> {
    let n = a.n;
    debug_assert_eq!(b.len(), n);
    debug_assert_eq!(x.len(), n);
    scratch[..n].copy_from_slice(&b[..n]);
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let v = a.values[k];
            scratch[i] -= v * x[j];
            if i != j {
                scratch[j] -= v * x[i];
            }
        }
    }
    let r_norm = scratch[..n].iter().map(|v| v * v).sum::<f64>().sqrt();
    let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !r_norm.is_finite() || !b_norm.is_finite() || b_norm == 0.0 {
        return None;
    }
    Some(r_norm / b_norm)
}

impl Default for FeralSolverInterface {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for FeralSolverInterface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeralSolverInterface")
            .field("dim", &self.dim)
            .field("nonzeros", &self.nonzeros)
            .field("initialized", &self.initialized)
            .field("negevals", &self.negevals)
            .finish_non_exhaustive()
    }
}

impl SparseSymLinearSolverInterface for FeralSolverInterface {
    fn multi_solve_matches_single_solve(&self, nrhs: usize) -> bool {
        nrhs <= FERAL_BITWISE_MULTI_SOLVE_MAX_NRHS
    }

    fn initialize_structure(
        &mut self,
        dim: Index,
        nonzeros: Index,
        ia: &[Index],
        ja: &[Index],
    ) -> ESymSolverStatus {
        assert_eq!(ia.len(), nonzeros as usize);
        assert_eq!(ja.len(), nonzeros as usize);

        self.dim = dim;
        self.nonzeros = nonzeros;
        self.values = vec![0.0; nonzeros as usize];

        // This is the only place the sparsity pattern can change, so it is
        // the only place the cached triplet → CSC permutation (and the
        // matrix it indexes into) has to be invalidated. The next `factor`
        // rebuilds both from scratch. gh#562.
        self.slot = None;
        self.matrix = None;

        // Convert 1-based MA57-style indices to 0-based for FERAL, and
        // canonicalize each entry to the lower triangle. MA57 accepts
        // either triangle of a symmetric COO; pounce's KKT assembly
        // takes advantage of that and emits a mix of lower- and
        // upper-triangle entries. FERAL's `CscMatrix::from_triplets`
        // documents "Entries must be lower-triangle (row >= col)" but
        // does NOT check it — upper-triangle entries get stored in the
        // CSC structure where the LDL^T factorization ignores them,
        // silently dropping them from the factored matrix.
        self.rows_0 = Vec::with_capacity(nonzeros as usize);
        self.cols_0 = Vec::with_capacity(nonzeros as usize);
        for k in 0..nonzeros as usize {
            let i = (ia[k] - 1) as usize;
            let j = (ja[k] - 1) as usize;
            if i >= j {
                self.rows_0.push(i);
                self.cols_0.push(j);
            } else {
                self.rows_0.push(j);
                self.cols_0.push(i);
            }
        }

        self.initialized = true;
        ESymSolverStatus::Success
    }

    fn values_array_mut(&mut self) -> &mut [Number] {
        debug_assert!(self.initialized);
        &mut self.values
    }

    fn multi_solve(
        &mut self,
        new_matrix: bool,
        _ia: &[Index],
        _ja: &[Index],
        nrhs: Index,
        rhs_vals: &mut [Number],
        check_neg_evals: bool,
        number_of_neg_evals: Index,
    ) -> ESymSolverStatus {
        // Quality was bumped since the last factor → caller must refill
        // values and we'll re-factor. Mirrors MA57's protocol.
        if self.pivtol_changed {
            self.pivtol_changed = false;
            if !new_matrix {
                self.refactorize = true;
                return ESymSolverStatus::CallAgain;
            }
        }

        if new_matrix || self.refactorize {
            let status = self.factor(check_neg_evals, number_of_neg_evals);
            if status != ESymSolverStatus::Success {
                return status;
            }
            self.refactorize = false;
        }

        self.backsolve(nrhs, rhs_vals)
    }

    fn number_of_neg_evals(&self) -> Index {
        debug_assert!(self.initialized);
        self.negevals
    }

    fn increase_quality(&mut self) -> bool {
        // Ipopt's `IncreaseQuality` contract: when `PdFullSpaceSolver`'s
        // refinement loop stagnates it asks the backend for a better
        // factorization before falling back on `pretend_singular` and the
        // perturbation handler (`IpPDFullSpaceSolver.cpp:296`). Every
        // upstream backend that can escalate does — MA57 raises `pivtol`
        // to `min(pivtolmax, pivtol^0.75)` (`IpMa57TSolverInterface.cpp:832`),
        // and `Ma57SolverInterface::increase_quality` mirrors it.
        //
        // feral has the same ladder (scaling Identity → InfNorm, then
        // `pivot_threshold^0.75`); this hands it the rung. The earlier
        // `false` here cited the ipopt-feral shim, but that shim's own
        // comment reads "POC: no escalation" — a proof-of-concept
        // shortcut, not a design.
        //
        // Wiring it is what lets `refine` default off (as it is for every
        // other backend, upstream and here): with the escalation missing,
        // feral's unconditional inner refinement was the only thing
        // standing between a hard KKT and `RestorationFailed`, at ~4x the
        // back-solve cost. Measured on gh#590's data-scale-1e11 LP,
        // `refine = no` fails `RestorationFailed` with this returning
        // `false` and succeeds with it wired; on the 126028-dimension
        // `laptime` KKT (L-BFGS leg) the pair is 68.9s -> 18.8s against
        // MA57's 10.7s.
        // gh #850: declined unless the caller asked for it. See
        // `FeralConfig::increase_quality` and the option's own help text --
        // the rung is a *lateral* move in trajectory terms, not a monotone
        // one, and on the models where it costs a verdict it costs the whole
        // solve.
        if !self.increase_quality {
            return false;
        }
        if self.solver.increase_quality() {
            self.pivtol_changed = true;
            true
        } else {
            false
        }
    }

    fn provides_inertia(&self) -> bool {
        true
    }

    fn matrix_format(&self) -> EMatrixFormat {
        EMatrixFormat::TripletFormat
    }

    fn provides_degeneracy_detection(&self) -> bool {
        true
    }

    /// Find the dependent rows of a constraint Jacobian `J` (an
    /// `n_rows × n_cols` matrix given as a 1-based triplet) by
    /// rank-revealing the scaled augmented system
    ///
    /// ```text
    ///     M = [ s·I_n   Jᵀ ]    s = max(1, max|J|),  d = n_cols + n_rows
    ///         [   J     0  ]
    /// ```
    ///
    /// `M` is symmetric with `rank(M) = n_cols + rank(J)`, so it has
    /// exactly `n_rows − rank(J)` singular pivots — one per dependent
    /// row. Scaling the identity block by `s ≥ max|J|` makes every
    /// `x`-column's diagonal the column maximum, so feral's threshold
    /// partial pivoting pins each of the first `n_cols` rows on its own
    /// column; the singular pivots therefore fall exclusively in the
    /// `J`-row block `[n_cols, d)`, and `perm[k] − n_cols` is the
    /// dependent row for each singular pivot position `k`. Working off
    /// the well-conditioned augmented system (rather than `JJᵀ`, whose
    /// squared conditioning blurs near-dependent rows) is the standard
    /// Ipopt `DetermineDependentRows` recipe.
    fn determine_dependent_rows(
        &mut self,
        n_rows: Index,
        n_cols: Index,
        irn: &[Index],
        jcn: &[Index],
        vals: &[Number],
        c_deps: &mut Vec<Index>,
    ) -> ESymSolverStatus {
        use feral::{
            LuParams, LuPivoting, LuScaling, LuSingularAction, SparseColMatrix, SparseLu,
            SparseLuSymbolic,
        };

        c_deps.clear();
        let n_rows = n_rows as usize;
        let n_cols = n_cols as usize;
        if n_rows == 0 {
            return ESymSolverStatus::Success;
        }
        let nnz = vals.len();
        if irn.len() != nnz || jcn.len() != nnz {
            return ESymSolverStatus::FatalError;
        }

        // Identity-block scale: dominate every J entry so the x-columns
        // pivot on their own diagonal (see method doc).
        let a_max = vals.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        let s = a_max.max(1.0);

        // Build M column-by-column: cols[col] = list of (row, value).
        let d = n_cols + n_rows;
        let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); d];
        for c in 0..n_cols {
            cols[c].push((c, s)); // s·I_n on the leading x-columns
        }
        for t in 0..nnz {
            let i = (irn[t] - 1) as usize; // 0-based J row
            let c = (jcn[t] - 1) as usize; // 0-based J col
            if i >= n_rows || c >= n_cols {
                return ESymSolverStatus::FatalError; // malformed triplet
            }
            let v = vals[t];
            cols[c].push((n_cols + i, v)); // J in the bottom-left block
            cols[n_cols + i].push((c, v)); // Jᵀ in the top-right block
        }

        let m = match SparseColMatrix::from_sparse_columns(d, &cols) {
            Ok(m) => m,
            Err(_) => return ESymSolverStatus::FatalError,
        };
        // Fill-reducing (AMD) order — `natural` blows up on large augmented
        // systems (gen: d≈6000). The dependent-row mapping below is unaffected:
        // it reads the *row pivot* permutation `perm()`, and the s-scaling pins
        // each x-column to its own x-row diagonal regardless of column order.
        //
        // This symbolic is only honoured because `params` below pins
        // `LuPivoting::GilbertPeierls`. feral 0.16 (feral#171) made
        // `Markowitz` the default, and Markowitz chooses its column order
        // during the factorization and ignores the symbolic argument outright
        // — the AMD claim in the paragraph above would quietly stop being
        // true about the factor this function actually reads.
        let symbolic = match SparseLuSymbolic::analyze(&m) {
            Ok(sym) => sym,
            Err(_) => return ESymSolverStatus::FatalError,
        };

        // feral computes its singularity floor as ztol = zero_pivot_tol·max|M|
        // (= zero_pivot_tol·s, since the identity scale dominates J) and
        // perturbs singular pivots up to `abs_floor`; setting abs_floor = ztol
        // keeps a perturbed pivot at the floor so it is detectable below.
        let zero_pivot_tol = 1e-13;
        let ztol = zero_pivot_tol * s;
        let params = LuParams {
            on_singular: LuSingularAction::PerturbToEps { abs_floor: ztol },
            scaling: LuScaling::None,
            zero_pivot_tol,
            // See the `SparseLuSymbolic::analyze` comment above: the
            // dependent-row scan is written against the AMD order, so the
            // symbolic has to be the thing that decides it.
            pivoting: LuPivoting::GilbertPeierls,
            ..LuParams::default()
        };
        let lu = match SparseLu::factor(&m, &symbolic, params) {
            Ok(lu) => lu,
            Err(_) => return ESymSolverStatus::FatalError,
        };

        // A pivot position is singular when its U diagonal sits at/below the
        // detection floor; independent pivots are O(s) ≫ this floor. The
        // s-scaling normally maps singular pivots to J-rows (perm[k] ≥
        // n_cols), but under heavy degeneracy (e.g. the elastic phase-1
        // vertices of NETLIB afiro/gen) AMD fill-in plus the Jᵀ coupling can
        // push a near-zero pivot onto an x-row. That is not an invariant
        // violation — we only collect J-row dependencies, so an x-row
        // singular pivot is simply skipped. Under-reporting here is safe:
        // the caller re-prunes / inertia-shifts on the next iteration.
        let dep_tol = 1e-9 * s;
        let perm = lu.perm();
        for k in 0..d {
            if lu.u_dense(k, k).abs() <= dep_tol {
                let r = perm[k];
                if r >= n_cols {
                    c_deps.push((r - n_cols) as Index);
                }
            }
        }
        c_deps.sort_unstable();
        ESymSolverStatus::Success
    }

    /// Walk feral's per-supernode `NodeFactors` to assemble the LDLᵀ
    /// factor's strict-lower nonzero pattern in *permuted*
    /// coordinates. `perm` is feral's global fill-reducing
    /// permutation (new-to-old). When `want_values` is true the
    /// per-supernode `l` block is also gathered into `l_vals` —
    /// indexed by the post-BK pivot perm so the order matches the
    /// (irn, jcn) arrays.
    ///
    /// Returns `None` before the first successful factor (`factors()`
    /// returns `None`).
    fn factor_pattern(&self, want_values: bool) -> Option<FactorPattern> {
        let factors = self.solver.factors()?;

        // Conservative upper bound on nnz_L (strict-lower): per-supernode
        //   nelim*(nelim-1)/2 + (nrow - nelim) * nelim
        // (the diagonal is excluded). Doubles as a single allocation
        // for both irn and jcn, plus l_vals when requested.
        let mut nnz_upper: usize = 0;
        for nf in &factors.node_factors {
            let ff = &nf.frontal_factors;
            let nelim = ff.nelim;
            let nrow = ff.nrow;
            let trailing = nrow.saturating_sub(nelim) * nelim;
            nnz_upper += nelim * nelim.saturating_sub(1) / 2 + trailing;
        }

        let mut l_irn: Vec<Index> = Vec::with_capacity(nnz_upper);
        let mut l_jcn: Vec<Index> = Vec::with_capacity(nnz_upper);
        let mut l_vals: Option<Vec<Number>> = if want_values {
            Some(Vec::with_capacity(nnz_upper))
        } else {
            None
        };

        for nf in &factors.node_factors {
            let ff = &nf.frontal_factors;
            let nelim = ff.nelim;
            let nrow = ff.nrow;
            // perm[i] = pre-BK supernode row that landed at post-BK
            // pivot position i. Indices [nelim, nrow) are identity.
            let perm = &ff.perm;
            let l = &ff.l;
            for j in 0..nelim {
                // Column j of L: global col index in permuted coords.
                let col_local = perm[j];
                let col_global = nf.row_indices[col_local];
                let col1 = (col_global as Index) + 1;
                // Strict-lower entries: rows i in (j, nrow).
                for i in (j + 1)..nrow {
                    let row_local = if i < nelim { perm[i] } else { i };
                    let row_global = nf.row_indices[row_local];
                    l_irn.push((row_global as Index) + 1);
                    l_jcn.push(col1);
                    if let Some(vals) = l_vals.as_mut() {
                        vals.push(l[j * nrow + i]);
                    }
                }
            }
        }

        Some(FactorPattern {
            n: factors.n,
            perm: factors.perm.clone(),
            l_irn,
            l_jcn,
            l_vals,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every boolean env knob in this crate speaks one grammar, in both
    /// directions. `FERAL_PARALLEL` used to be the outlier: it parsed only
    /// `0|false|off`, so it was missing `no` (which feral's own C-ABI shim
    /// accepts) and had no force-*on* spelling at all — `FERAL_PARALLEL=1`
    /// silently did nothing, leaving CLI/Python/NL callers with no way to
    /// override feral's platform-derived default (feral#156), since
    /// `FeralConfig.parallel` is reachable only from the Rust API.
    #[test]
    fn parse_bool_env_accepts_the_full_grammar_both_ways() {
        for on in ["1", "on", "true", "yes"] {
            assert_eq!(parse_bool_env(Some(on)), Some(true), "on-token {on:?}");
        }
        for off in ["0", "off", "false", "no"] {
            assert_eq!(parse_bool_env(Some(off)), Some(false), "off-token {off:?}");
        }
        // Unset and unrecognized both mean "leave the default alone" — the
        // tri-state `None`, not a silent `false`.
        assert_eq!(parse_bool_env(None), None);
        for other in ["", "2", "TRUE", "yep", "off ", "disabled"] {
            assert_eq!(parse_bool_env(Some(other)), None, "unrecognized {other:?}");
        }
    }

    /// L12: the pivot-threshold env override honors the documented
    /// `POUNCE_FERAL_*` convention (`POUNCE_FERAL_PIVTOL`) and keeps the bare
    /// `FERAL_PIVTOL` only as a deprecated legacy alias. Tested on the pure
    /// `resolve_pivtol_env` helper so it never mutates the process
    /// environment (which would be a data race under the rayon-parallel
    /// solves — the same hazard fixed in L9).
    #[test]
    fn resolve_pivtol_env_honors_pounce_convention() {
        // The documented POUNCE_FERAL_* name is read (this is the bug: the
        // old code only looked at the bare FERAL_PIVTOL, so this was ignored).
        assert_eq!(resolve_pivtol_env(Some("0.3"), None), 0.3);
        // Legacy FERAL_PIVTOL is still honored when the convention var is unset.
        assert_eq!(resolve_pivtol_env(None, Some("0.4")), 0.4);
        // Both set: the convention name takes precedence over the legacy alias.
        assert_eq!(resolve_pivtol_env(Some("0.3"), Some("0.4")), 0.3);
        // Neither set: the 1e-8 default (matches FeralConfig::default).
        assert_eq!(resolve_pivtol_env(None, None), 1e-8);
        // Unparseable convention value falls through to the legacy alias...
        assert_eq!(resolve_pivtol_env(Some("garbage"), Some("0.4")), 0.4);
        // ...and, with no legacy value, to the default.
        assert_eq!(resolve_pivtol_env(Some("garbage"), None), 1e-8);
    }

    /// 2x2 SPD matrix `[[2,1],[1,3]]`. Lower-triangle 1-based triplets.
    /// Solving against (3, 4) gives x = (1, 1).
    #[test]
    fn factor_and_solve_spd_2x2() {
        let mut s = FeralSolverInterface::new();
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];

        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);

        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        assert!((rhs[0] - 1.0).abs() < 1e-12, "x0 = {}", rhs[0]);
        assert!((rhs[1] - 1.0).abs() < 1e-12, "x1 = {}", rhs[1]);
        assert_eq!(s.number_of_neg_evals(), 0);
        assert!(s.provides_inertia());
        assert_eq!(s.matrix_format(), EMatrixFormat::TripletFormat);
    }

    /// 2x2 indefinite `[[1,2],[2,1]]` — eigenvalues 3, -1.
    #[test]
    fn detects_one_negative_eigenvalue() {
        let mut s = FeralSolverInterface::new();
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];

        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[1.0, 2.0, 1.0]);

        let mut rhs = [3.0, 3.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 1),
            ESymSolverStatus::Success
        );
        assert_eq!(s.number_of_neg_evals(), 1);
        assert!((rhs[0] - 1.0).abs() < 1e-12);
        assert!((rhs[1] - 1.0).abs() < 1e-12);
    }

    /// Wrong expected inertia → `WrongInertia` (and no solve).
    #[test]
    fn check_neg_evals_mismatch_returns_wrong_inertia() {
        let mut s = FeralSolverInterface::new();
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]); // SPD
        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 1),
            ESymSolverStatus::WrongInertia
        );
    }

    /// gh#540: when the inertia count disagrees *and* the factorization's
    /// smallest pivot is at the working-precision floor, the count is noise
    /// and the factor is reported `Singular` — so the IPM raises `δ_c`
    /// (which repairs a rank-deficient constraint block) instead of walking
    /// the `δ_w` ladder against a reading that does not respond to it.
    ///
    /// `[[1, 1], [1, 1 + 2^-50]]` is positive definite by a hair: its
    /// eigenvalues are ≈ 2 and ≈ 4.4e-16, so no diagonal equilibration can
    /// lift the small one — it is near-singular, not merely badly scaled.
    ///
    /// The floor is pinned explicitly here. Since gh#592 the default is
    /// `n · eps`, and at `n = 2` that is 4.4e-16 — under this matrix's
    /// 8.9e-16 pivot, so the *default* declines to fire on it (which is
    /// the point of the dimension-aware floor and is asserted below).
    /// A 2x2 cannot exhibit a floor that scales with `n`; the routing it
    /// does pin is exercised at solver scale by the `eigena2` / `eigenb2`
    /// CLI regressions.
    #[test]
    fn untrustworthy_inertia_is_reported_singular_not_wrong() {
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        let vals = [1.0, 1.0, 1.0 + f64::powi(2.0, -50)];

        let mut s = FeralSolverInterface::with_config(FeralConfig {
            inertia_pivot_floor: Some(1e-12),
            ..FeralConfig::default()
        });
        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&vals);
        let mut rhs = [1.0, 1.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 1),
            ESymSolverStatus::Singular,
            "a count read off a pivot at the working-precision floor was \
             passed on as evidence",
        );

        // gh#592: the dimension-aware default is 2 * eps = 4.4e-16 here,
        // and this pivot (8.9e-16) clears it — so the same factorization
        // is reported `WrongInertia` under defaults. A 2x2's pivots are
        // measurable at a level a 300x300's are not, and the floor now
        // says so instead of applying a 1e-12 constant to both.
        let mut s = FeralSolverInterface::new();
        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&vals);
        let mut rhs = [1.0, 1.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 1),
            ESymSolverStatus::WrongInertia,
            "the n * eps floor convicted a pivot three orders of magnitude \
             above it",
        );

        // Disabling the trigger restores the pre-#540 verdict, which is what
        // makes this a routing change and not a new failure mode: the same
        // factorization is still rejected, just with the other status.
        let mut s = FeralSolverInterface::with_config(FeralConfig {
            inertia_pivot_floor: Some(0.0),
            ..FeralConfig::default()
        });
        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&vals);
        let mut rhs = [1.0, 1.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 1),
            ESymSolverStatus::WrongInertia
        );
    }

    /// gh#592: the floor the trigger consults is `n · eps`, not a
    /// constant. The option's own rationale always named `n · eps` as the
    /// level at which an equilibrated pivot loses its sign; the constant
    /// `1e-12` it shipped with corresponds to `n ≈ 4500`, so on the
    /// few-hundred-order KKTs an IPM actually factors it convicted pivots
    /// more than an order of magnitude above the noise. Explicit settings
    /// are still absolute, and `0` still disables.
    #[test]
    fn inertia_trust_floor_is_dimension_aware_unless_pinned() {
        // Auto: `n · eps`, so it tracks the matrix it is judging.
        assert_eq!(inertia_trust_floor(None, 1), f64::EPSILON);
        assert_eq!(inertia_trust_floor(None, 311), 311.0 * f64::EPSILON);
        assert!(inertia_trust_floor(None, 311) < 1e-13);
        assert!(inertia_trust_floor(None, 311) > 1e-14);
        // ...and the old constant is only right around n ≈ 4500.
        assert!((inertia_trust_floor(None, 4503) - 1e-12).abs() < 1e-14);

        // Explicit: absolute at every dimension.
        assert_eq!(inertia_trust_floor(Some(1e-12), 2), 1e-12);
        assert_eq!(inertia_trust_floor(Some(1e-12), 100_000), 1e-12);

        // Explicit zero disables the trigger (the pre-#540 routing).
        assert_eq!(inertia_trust_floor(Some(0.0), 100_000), 0.0);
    }

    /// gh#592, the mechanism, at the dimension the constant got wrong.
    ///
    /// Order 400, identity but for one diagonal entry at `2e-13`. That
    /// pivot is measured, not noise: it is more than twice `n · eps`
    /// (8.9e-14 here), and its sign is as trustworthy as the other 399.
    /// The requested count is deliberately wrong, so the trigger is
    /// consulted — and must decline, leaving `WrongInertia` and the
    /// `δ_w` ladder that answers a genuine curvature mismatch.
    ///
    /// Under the pre-#592 constant the same pivot is under `1e-12` and
    /// the factor was reported `Singular`, which spends `δ_c` on a
    /// constraint Jacobian that has full rank and then keeps it switched
    /// on. Both verdicts are asserted, so this test says which floor is
    /// in force rather than merely that some floor is.
    #[test]
    fn a_measurable_pivot_at_solver_scale_is_not_convicted_by_the_floor() {
        const N: usize = 400;
        const SMALL: f64 = 2e-13;
        assert!(
            SMALL > N as f64 * f64::EPSILON && SMALL < 1e-12,
            "the pivot must sit between the two floors for this test to \
             distinguish them",
        );

        // Identity on the first N-2 rows, then a 2x2 block
        // `[[1, 1], [1, 1 + SMALL]]` whose Schur pivot is `SMALL`. A bare
        // small *diagonal* entry would not do: equilibration lifts it back
        // to 1, which is the whole point of `feral_scaling`. This block is
        // near-singular after any diagonal scaling.
        let mut irn: Vec<Index> = (1..=(N - 2) as Index).collect();
        let mut jcn = irn.clone();
        let mut vals = vec![1.0; N - 2];
        let (a, b) = ((N - 1) as Index, N as Index);
        irn.extend_from_slice(&[a, b, b]);
        jcn.extend_from_slice(&[a, a, b]);
        vals.extend_from_slice(&[1.0, 1.0, 1.0 + SMALL]);
        let nnz = irn.len() as Index;

        // The IPM asks for one negative eigenvalue; the matrix has none,
        // so the count mismatches and the floor is consulted.
        let mut solve_with = |cfg: FeralConfig| {
            let mut s = FeralSolverInterface::with_config(cfg);
            assert_eq!(
                s.initialize_structure(N as Index, nnz, &irn, &jcn),
                ESymSolverStatus::Success
            );
            s.values_array_mut().copy_from_slice(&vals);
            let mut rhs = vec![1.0; N];
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 1)
        };

        assert_eq!(
            solve_with(FeralConfig::default()),
            ESymSolverStatus::WrongInertia,
            "the n * eps floor convicted a pivot above it, so delta_c is \
             being spent on a full-rank constraint Jacobian (gh#592)",
        );
        assert_eq!(
            solve_with(FeralConfig {
                inertia_pivot_floor: Some(1e-12),
                ..FeralConfig::default()
            }),
            ESymSolverStatus::Singular,
            "the pre-#592 constant no longer reproduces the misrouting, so \
             the assertion above is no longer pinning the fix it describes",
        );
    }

    /// ...and a healthy factor with a mismatching count keeps reporting
    /// `WrongInertia`: the floor must not swallow the ordinary
    /// negative-curvature signal the `δ_w` ladder exists to answer.
    /// `[[2, 1], [1, 3]]` is SPD with pivots ≈ 2 and 2.5.
    #[test]
    fn well_conditioned_inertia_mismatch_still_reports_wrong_inertia() {
        let mut s = FeralSolverInterface::new();
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 1),
            ESymSolverStatus::WrongInertia
        );
    }

    /// The W=0 saddle-point shape, unregularized: feral reports the
    /// inertia correctly, flags a wrong count, and calls a structurally
    /// singular system singular rather than inventing a count.
    ///
    /// This is the structure review item M3 was concerned about — the
    /// least-squares multiplier system `[A Bᵀ; B 0]` with a
    /// structurally-zero `(2,2)` block and `δ = 0` — and the reason it
    /// is pinned here is #688, which removed M3's blanket `1e-8`
    /// perturbation from the `recalc_y` caller. M3's justification was
    /// that feral mis-reports the inertia of exactly this shape (0
    /// negatives on `nuffield2_trap` against a true `n_c + n_d`). That
    /// matrix is not in this repo, so the claim cannot be reproduced
    /// directly; what can be established is the well-conditioned case,
    /// which is where a defect would be unambiguous.
    ///
    /// `[[1,0,1],[0,1,1],[1,1,0]]`: `A = I` (SPD, n=2), `B = [1 1]`
    /// (full row rank, m=1). Saddle-point inertia theory gives exactly
    /// `n` positive and `m` negative, so the true negative count is 1.
    /// Eigenvalues are 1, 2 and −1 — no pivot anywhere near the
    /// working-precision floor, so the count is a measurement rather
    /// than noise, and `inertia_trust_floor` has nothing to say about
    /// it.
    ///
    /// A failure here would mean feral cannot count the inertia of a
    /// healthy saddle-point system, which would be a defect worth its
    /// own issue. It passes, which is evidence the other way: whatever
    /// M3 saw was on a factorization whose pivots had lost their sign,
    /// the case gh#540 and gh#592 addressed at source by routing to
    /// `Singular` and `δ_c`.
    #[test]
    fn saddle_point_inertia_is_correct_without_regularization() {
        let mut s = FeralSolverInterface::new();
        let irn: [Index; 5] = [1, 2, 3, 3, 3];
        let jcn: [Index; 5] = [1, 2, 1, 2, 3];
        assert_eq!(
            s.initialize_structure(3, 5, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut()
            .copy_from_slice(&[1.0, 1.0, 1.0, 1.0, 0.0]);
        let mut rhs = [1.0, 1.0, 1.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 1),
            ESymSolverStatus::Success,
            "feral disagreed with the true saddle-point inertia (n=2, m=1) \
             on a well-conditioned system — that would be a real defect",
        );

        // The check is live, not vacuously passing.
        let mut s2 = FeralSolverInterface::new();
        assert_eq!(
            s2.initialize_structure(3, 5, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s2.values_array_mut()
            .copy_from_slice(&[1.0, 1.0, 1.0, 1.0, 0.0]);
        let mut rhs2 = [1.0, 1.0, 1.0];
        assert_eq!(
            s2.multi_solve(true, &irn, &jcn, 1, &mut rhs2, true, 2),
            ESymSolverStatus::WrongInertia,
        );

        // And a structurally singular system is called singular rather
        // than answered with a fabricated count.
        let mut s3 = FeralSolverInterface::new();
        let irn3: [Index; 3] = [1, 2, 3];
        let jcn3: [Index; 3] = [1, 2, 3];
        assert_eq!(
            s3.initialize_structure(3, 3, &irn3, &jcn3),
            ESymSolverStatus::Success
        );
        s3.values_array_mut().copy_from_slice(&[1.0, 1.0, 0.0]);
        let mut rhs3 = [1.0, 1.0, 1.0];
        assert_eq!(
            s3.multi_solve(true, &irn3, &jcn3, 1, &mut rhs3, true, 1),
            ESymSolverStatus::Singular,
        );
    }

    /// `increase_quality` then resolve with `new_matrix=false`
    /// returns `CallAgain`; refilling values and retrying succeeds.
    #[test]
    fn increase_quality_then_resolve_triggers_call_again() {
        let mut s = FeralSolverInterface::new();
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );

        if s.increase_quality() {
            // Quality bumped; new_matrix=false → CallAgain.
            let mut rhs = [3.0, 4.0];
            assert_eq!(
                s.multi_solve(false, &irn, &jcn, 1, &mut rhs, false, 0),
                ESymSolverStatus::CallAgain
            );
            // Refill values and retry.
            s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
            let mut rhs = [3.0, 4.0];
            assert_eq!(
                s.multi_solve(false, &irn, &jcn, 1, &mut rhs, false, 0),
                ESymSolverStatus::Success
            );
            assert!((rhs[0] - 1.0).abs() < 1e-12);
            assert!((rhs[1] - 1.0).abs() < 1e-12);
        }
    }

    /// Issue #79: the first-class per-backend `parallel` toggle builds a
    /// serial factor without touching any global state, and its result is
    /// bit-identical to the parallel driver (feral guarantees parity).
    #[test]
    fn per_backend_parallel_toggle_serial_matches_parallel() {
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        let solve = |mut s: FeralSolverInterface| -> [f64; 2] {
            assert_eq!(
                s.initialize_structure(2, 3, &irn, &jcn),
                ESymSolverStatus::Success
            );
            s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
            let mut rhs = [3.0, 4.0];
            assert_eq!(
                s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
                ESymSolverStatus::Success
            );
            rhs
        };
        let par = solve(FeralSolverInterface::with_config(FeralConfig {
            parallel: Some(true),
            ..FeralConfig::default()
        }));
        let ser = solve(FeralSolverInterface::serial());
        // [[2,1],[1,3]] x = [3,4] ⇒ x = [1, 1], same both ways.
        assert!((par[0] - 1.0).abs() < 1e-12 && (par[1] - 1.0).abs() < 1e-12);
        assert_eq!(
            par, ser,
            "serial and parallel factors must agree bit-for-bit"
        );
    }

    /// Pounce emits some symmetric entries as upper-triangle
    /// `(i, j)` with `i < j` because MA57 accepts either half. The
    /// FERAL wrapper must canonicalize to lower triangle (row >= col)
    /// before handing entries to `CscMatrix::from_triplets`, which
    /// silently drops upper-triangle entries during LDL^T. A regression
    /// in this canonicalization would corrupt residuals and inertia
    /// (see jkitchin/feral#6).
    #[test]
    fn upper_triangle_entries_are_canonicalized() {
        let mut s = FeralSolverInterface::new();
        // Same matrix as `factor_and_solve_spd_2x2`, but the (2,1)
        // off-diagonal is given as upper-triangle (1,2).
        let irn: [Index; 3] = [1, 1, 2];
        let jcn: [Index; 3] = [1, 2, 2];
        s.initialize_structure(2, 3, &irn, &jcn);
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);

        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        assert!((rhs[0] - 1.0).abs() < 1e-12, "x0 = {}", rhs[0]);
        assert!((rhs[1] - 1.0).abs() < 1e-12, "x1 = {}", rhs[1]);
    }

    /// `factor_pattern` returns the L sparsity (strict-lower) after a
    /// successful factor. For the SPD 2x2 above, L has exactly one
    /// strict-lower entry (the single off-diagonal), and `perm` is a
    /// permutation of `0..n`.
    #[test]
    fn factor_pattern_returns_l_after_factor() {
        let mut s = FeralSolverInterface::new();
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        s.initialize_structure(2, 3, &irn, &jcn);
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
        let mut rhs = [3.0, 4.0];
        s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0);

        // Pattern-only.
        let pat = s.factor_pattern(false).expect("factors present");
        assert_eq!(pat.n, 2);
        assert_eq!(pat.perm.len(), 2);
        assert!(pat.perm.contains(&0) && pat.perm.contains(&1));
        assert_eq!(pat.l_irn.len(), 1, "L strict-lower nnz = 1 for SPD 2x2");
        assert_eq!(pat.l_jcn.len(), 1);
        assert!(pat.l_vals.is_none(), "values not requested");

        // With values.
        let pat = s.factor_pattern(true).expect("factors present");
        let vals = pat.l_vals.as_ref().expect("values requested");
        assert_eq!(vals.len(), pat.l_irn.len());
        // The single strict-lower L entry should be finite.
        assert!(vals[0].is_finite());
    }

    /// Before any factor, `factor_pattern` returns `None`.
    #[test]
    fn factor_pattern_none_before_factor() {
        let s = FeralSolverInterface::new();
        assert!(s.factor_pattern(false).is_none());
        assert!(s.factor_pattern(true).is_none());
    }

    /// Two-RHS solve via `solve_many`.
    #[test]
    fn multi_rhs_solve() {
        let mut s = FeralSolverInterface::new();
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);

        // Column 1: A x = (3, 4) → x = (1, 1)
        // Column 2: A x = (4, 5) → x = (7/5, 6/5)
        let mut rhs = [3.0, 4.0, 4.0, 5.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 2, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        assert!((rhs[0] - 1.0).abs() < 1e-10);
        assert!((rhs[1] - 1.0).abs() < 1e-10);
        assert!((rhs[2] - 7.0 / 5.0).abs() < 1e-10);
        assert!((rhs[3] - 6.0 / 5.0).abs() < 1e-10);
    }

    /// `parse_scaling_strategy` accepts every documented tag (in either
    /// case / with aliases) and rejects unknown ones. `external` is not
    /// reachable from a string tag (it carries a vector).
    #[test]
    fn parse_scaling_strategy_accepts_documented_tags() {
        use ScalingStrategy::*;
        let cases: &[(&str, ScalingStrategy)] = &[
            ("auto", Auto),
            ("AUTO", Auto),
            ("infnorm", InfNorm),
            ("inf_norm", InfNorm),
            ("inf", InfNorm),
            ("mc64", Mc64Symmetric),
            ("MC64", Mc64Symmetric),
            ("mc64symmetric", Mc64Symmetric),
            ("mc64_symmetric", Mc64Symmetric),
            ("identity", Identity),
            ("none", Identity),
        ];
        for (tag, expected) in cases {
            assert_eq!(
                parse_scaling_strategy(tag),
                Some(expected.clone()),
                "tag {tag:?} should parse"
            );
        }
        assert_eq!(parse_scaling_strategy("external"), None);
        assert_eq!(parse_scaling_strategy("not_a_strategy"), None);
        assert_eq!(parse_scaling_strategy(""), None);
    }

    /// The pounce default is FERAL's default — `ScalingStrategy::Auto` —
    /// so behaviour is unchanged when the option is left unset.
    #[test]
    fn default_scaling_is_auto() {
        assert_eq!(FeralConfig::default().scaling, ScalingStrategy::Auto);
    }

    /// `with_config` actually propagates the configured scaling strategy
    /// into the underlying FERAL solver (and each variant still
    /// constructs + factors a tiny SPD system).
    #[test]
    fn every_scaling_propagates_and_factors() {
        use ScalingStrategy::*;
        for strategy in [Auto, InfNorm, Mc64Symmetric, Identity] {
            let cfg = FeralConfig {
                scaling: strategy.clone(),
                ..FeralConfig::default()
            };
            let mut s = FeralSolverInterface::with_config(cfg);
            assert_eq!(
                s.solver.scaling_strategy(),
                &strategy,
                "configured strategy should reach the solver for {strategy:?}"
            );
            let irn: [Index; 3] = [1, 2, 2];
            let jcn: [Index; 3] = [1, 1, 2];
            assert_eq!(
                s.initialize_structure(2, 3, &irn, &jcn),
                ESymSolverStatus::Success,
                "structure init for {strategy:?}"
            );
            s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
            let mut rhs = [3.0, 4.0];
            assert_eq!(
                s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
                ESymSolverStatus::Success,
                "solve for {strategy:?}"
            );
            assert!((rhs[0] - 1.0).abs() < 1e-10, "x0 for {strategy:?}");
            assert!((rhs[1] - 1.0).abs() < 1e-10, "x1 for {strategy:?}");
        }
    }

    /// The pounce default leaves static pivoting unset (`None`), so the
    /// backend inherits feral's SSIDS-style delayed-pivot default and the
    /// numeric path is unchanged unless a caller opts in.
    #[test]
    fn default_static_pivoting_is_none() {
        assert_eq!(FeralConfig::default().static_pivoting, None);
    }

    /// Both explicit static-pivoting settings still construct, factor, and
    /// solve a tiny SPD system exactly — the `with_static_pivoting` builder
    /// call is wired through `configure_solver` and does not break solving.
    /// (feral exposes no public getter for `allow_delayed_pivots`, so the
    /// propagation is exercised through an end-to-end solve rather than a
    /// state assertion.)
    #[test]
    fn static_pivoting_settings_propagate_and_factor() {
        for on in [true, false] {
            let cfg = FeralConfig {
                static_pivoting: Some(on),
                ..FeralConfig::default()
            };
            let mut s = FeralSolverInterface::with_config(cfg);
            let irn: [Index; 3] = [1, 2, 2];
            let jcn: [Index; 3] = [1, 1, 2];
            assert_eq!(
                s.initialize_structure(2, 3, &irn, &jcn),
                ESymSolverStatus::Success,
                "structure init for static_pivoting={on}"
            );
            s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
            let mut rhs = [3.0, 4.0];
            assert_eq!(
                s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
                ESymSolverStatus::Success,
                "solve for static_pivoting={on}"
            );
            assert!((rhs[0] - 1.0).abs() < 1e-10, "x0 for static_pivoting={on}");
            assert!((rhs[1] - 1.0).abs() < 1e-10, "x1 for static_pivoting={on}");
        }
    }

    /// `parse_ordering_method` accepts every documented tag (in either
    /// case) and rejects unknown ones.
    #[test]
    fn parse_ordering_method_accepts_documented_tags() {
        use OrderingMethod::*;
        let cases: &[(&str, OrderingMethod)] = &[
            ("auto", Auto),
            ("AUTO", Auto),
            ("auto_race", AutoRace),
            ("autorace", AutoRace),
            ("race", AutoRace),
            ("amd", Amd),
            ("AMD", Amd),
            ("amf", Amf),
            ("metis", MetisND),
            ("metis_nd", MetisND),
            ("MetisND", MetisND),
            ("scotch", ScotchND),
            ("kahip", KahipND),
        ];
        for (tag, expected) in cases {
            assert_eq!(
                parse_ordering_method(tag),
                Some(expected.clone()),
                "tag {tag:?} should parse"
            );
        }
        assert_eq!(parse_ordering_method("not_a_method"), None);
        assert_eq!(parse_ordering_method(""), None);
    }

    /// Each `OrderingMethod` variant constructs a usable solver and
    /// can factor a tiny SPD system.
    #[test]
    fn every_ordering_constructs_and_factors() {
        use OrderingMethod::*;
        for method in [Auto, AutoRace, Amd, Amf, MetisND, ScotchND, KahipND] {
            let cfg = FeralConfig {
                // `.clone()` — `OrderingMethod` is no longer `Copy` (FERAL
                // 0.13); `method` is reused in the assert messages below.
                ordering: method.clone(),
                ..FeralConfig::default()
            };
            let mut s = FeralSolverInterface::with_config(cfg);
            let irn: [Index; 3] = [1, 2, 2];
            let jcn: [Index; 3] = [1, 1, 2];
            assert_eq!(
                s.initialize_structure(2, 3, &irn, &jcn),
                ESymSolverStatus::Success,
                "structure init for {method:?}"
            );
            s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
            let mut rhs = [3.0, 4.0];
            assert_eq!(
                s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
                ESymSolverStatus::Success,
                "solve for {method:?}"
            );
            assert!((rhs[0] - 1.0).abs() < 1e-10, "x0 for {method:?}");
            assert!((rhs[1] - 1.0).abs() < 1e-10, "x1 for {method:?}");
        }
    }

    /// A caller-supplied `OrderingMethod::External` permutation (pounce#180
    /// item 1 / FERAL#107) factors and solves correctly. Both the identity
    /// and a swapped permutation are valid bijections, so each must recover
    /// the same solution as every built-in ordering does above.
    #[test]
    fn external_ordering_solves_correctly() {
        for perm in [vec![0usize, 1], vec![1usize, 0]] {
            let cfg = FeralConfig {
                ordering: OrderingMethod::External(perm.clone()),
                ..FeralConfig::default()
            };
            let mut s = FeralSolverInterface::with_config(cfg);
            // [[2,1],[1,3]] x = [3,4]  →  x = [1,1].
            let irn: [Index; 3] = [1, 2, 2];
            let jcn: [Index; 3] = [1, 1, 2];
            assert_eq!(
                s.initialize_structure(2, 3, &irn, &jcn),
                ESymSolverStatus::Success,
                "structure init for perm {perm:?}"
            );
            s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
            let mut rhs = [3.0, 4.0];
            assert_eq!(
                s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
                ESymSolverStatus::Success,
                "solve for perm {perm:?}"
            );
            assert!((rhs[0] - 1.0).abs() < 1e-10, "x0 for perm {perm:?}");
            assert!((rhs[1] - 1.0).abs() < 1e-10, "x1 for perm {perm:?}");
        }
    }

    /// A wrong-length / non-bijection `External` permutation is rejected
    /// by FERAL as `InvalidInput` and surfaces as a non-`Success` status —
    /// never a panic and never a silently-wrong solve. Here a length-3
    /// permutation is supplied for a 2×2 matrix.
    #[test]
    fn external_ordering_wrong_length_fails_without_panic() {
        let cfg = FeralConfig {
            ordering: OrderingMethod::External(vec![0, 1, 2]),
            ..FeralConfig::default()
        };
        let mut s = FeralSolverInterface::with_config(cfg);
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        // The validation can fire at symbolic (structure init) or at the
        // first factor (solve); either way the status must not be Success.
        let init_st = s.initialize_structure(2, 3, &irn, &jcn);
        if init_st == ESymSolverStatus::Success {
            s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
            let mut rhs = [3.0, 4.0];
            let solve_st = s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0);
            assert_ne!(
                solve_st,
                ESymSolverStatus::Success,
                "wrong-length external ordering must fail the solve"
            );
        }
    }

    /// Rank-deficient J: rows 1,2 over 3 cols with row 2 = 2·row 1.
    /// `determine_dependent_rows` must flag exactly the *one* redundant
    /// row, and (per the s-scaling argument) it must be a real J-row
    /// index in `[0, n_rows)`. Pins R2: the singular-pivot → row map.
    #[test]
    fn determine_dependent_rows_flags_the_redundant_row() {
        let mut s = FeralSolverInterface::new();
        assert!(s.provides_degeneracy_detection());
        // J = [[1,1,1],[2,2,2]] as a 1-based triplet.
        let irn: [Index; 6] = [1, 1, 1, 2, 2, 2];
        let jcn: [Index; 6] = [1, 2, 3, 1, 2, 3];
        let vals = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let mut c_deps = Vec::new();
        let st = s.determine_dependent_rows(2, 3, &irn, &jcn, &vals, &mut c_deps);
        assert_eq!(st, ESymSolverStatus::Success);
        assert_eq!(c_deps.len(), 1, "exactly one dependent row, got {c_deps:?}");
        assert!(
            c_deps[0] == 0 || c_deps[0] == 1,
            "dep row in range: {c_deps:?}"
        );
    }

    /// Full-rank J (identity rows): no dependent rows.
    #[test]
    fn determine_dependent_rows_full_rank_reports_none() {
        let mut s = FeralSolverInterface::new();
        // J = I_3.
        let irn: [Index; 3] = [1, 2, 3];
        let jcn: [Index; 3] = [1, 2, 3];
        let vals = [1.0, 1.0, 1.0];
        let mut c_deps = Vec::new();
        let st = s.determine_dependent_rows(3, 3, &irn, &jcn, &vals, &mut c_deps);
        assert_eq!(st, ESymSolverStatus::Success);
        assert!(c_deps.is_empty(), "full-rank J has no deps, got {c_deps:?}");
    }

    /// Three rows, rank 2: row 3 = row 1 + row 2. Exactly one dependency,
    /// and dropping it must leave the other two independent.
    #[test]
    fn determine_dependent_rows_rank_two_of_three() {
        let mut s = FeralSolverInterface::new();
        // r1=[1,0,1], r2=[0,1,1], r3=r1+r2=[1,1,2].
        let irn: [Index; 7] = [1, 1, 2, 2, 3, 3, 3];
        let jcn: [Index; 7] = [1, 3, 2, 3, 1, 2, 3];
        let vals = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0];
        let mut c_deps = Vec::new();
        let st = s.determine_dependent_rows(3, 3, &irn, &jcn, &vals, &mut c_deps);
        assert_eq!(st, ESymSolverStatus::Success);
        assert_eq!(
            c_deps.len(),
            1,
            "one dependency among 3 rows, got {c_deps:?}"
        );
        assert!(c_deps[0] <= 2, "row index in range: {c_deps:?}");
    }

    // ---------------------------------------------------------------
    // gh#562: cached triplet → CSC slot permutation.
    //
    // A 3×3 pattern chosen to exercise every branch `from_triplets`
    // takes: an upper-triangle entry that `initialize_structure` must
    // canonicalize, rows arriving out of order within a column, and a
    // duplicate pair that has to be summed — plus a (3,3) diagonal
    // standing in for the KKT's (2,2) multiplier block, which the
    // refill tests drive to an explicit structural zero.
    //
    //   k:      0      1      2      3      4      5      6
    //   1-based (1,1)  (3,1)  (2,2)  (1,2)  (3,3)  (3,1)  (3,2)
    //   0-based (0,0)  (2,0)  (1,1)  (1,0)  (2,2)  (2,0)  (2,1)
    //                                 ^ canonicalized from (0,1)
    //
    // Lower triangle: col 0 = rows {0,1,2}, col 1 = rows {1,2},
    // col 2 = row {2} — six stored nonzeros from seven triplets.
    // ---------------------------------------------------------------

    const SLOT_IRN: [Index; 7] = [1, 3, 2, 1, 3, 3, 3];
    const SLOT_JCN: [Index; 7] = [1, 1, 2, 2, 3, 1, 2];

    /// Dense symmetric expansion of the triplet set above, for residual
    /// checks. Values are the caller-supplied `values_array` order.
    fn dense_from_triplets(vals: &[Number; 7]) -> [[f64; 3]; 3] {
        let mut a = [[0.0f64; 3]; 3];
        for k in 0..7 {
            let i = (SLOT_IRN[k] - 1) as usize;
            let j = (SLOT_JCN[k] - 1) as usize;
            a[i][j] += vals[k];
            if i != j {
                a[j][i] += vals[k];
            }
        }
        a
    }

    /// `slot_map` locates every triplet — including the duplicate pair
    /// and the canonicalized upper-triangle entry — at the CSC position
    /// `from_triplets` actually used.
    #[test]
    fn slot_map_matches_from_triplets_layout() {
        let rows = vec![0usize, 2, 1, 1, 2, 2, 2];
        let cols = vec![0usize, 0, 1, 0, 2, 0, 1];
        let vals = vec![4.0, 1.0, 5.0, 1.0, 3.0, 1.0, 1.0];
        let m = CscMatrix::from_triplets(3, &rows, &cols, &vals).unwrap();

        assert_eq!(m.col_ptr, vec![0, 3, 5, 6]);
        assert_eq!(m.row_idx, vec![0, 1, 2, 1, 2, 2]);

        let slot = slot_map(&m, &rows, &cols).expect("every triplet is locatable");
        assert_eq!(slot, vec![0, 2, 3, 1, 5, 2, 4]);

        // Replaying the permutation reproduces the built values exactly,
        // duplicate summing included (slot 2 takes triplets 1 and 5).
        let mut refilled = vec![0.0f64; m.nnz()];
        for (&s, &v) in slot.iter().zip(vals.iter()) {
            refilled[s] += v;
        }
        assert_eq!(refilled, m.values);
    }

    /// The load-bearing property from gh#562: a values refill that drives
    /// the (2,2)-block diagonal to an explicit `0.0` must not drop that
    /// slot. The refilled matrix has to be structurally bit-identical to a
    /// fresh `from_triplets` over the same values — same `nnz`, same
    /// `col_ptr`, same `row_idx` — or the feral#46 structurally-absent
    /// diagonal cascade comes back.
    #[test]
    fn refill_preserves_structure_when_a_diagonal_goes_to_zero() {
        let mut s = FeralSolverInterface::new();
        assert_eq!(
            s.initialize_structure(3, 7, &SLOT_IRN, &SLOT_JCN),
            ESymSolverStatus::Success
        );

        // First factorization: builds the CSC and records the permutation.
        let first = [4.0, 1.0, 5.0, 1.0, 3.0, 1.0, 1.0];
        s.values_array_mut().copy_from_slice(&first);
        let mut rhs = [1.0, 1.0, 1.0];
        assert_eq!(
            s.multi_solve(true, &SLOT_IRN, &SLOT_JCN, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        assert!(s.slot.is_some(), "first factor must record the permutation");
        let built = s.matrix.as_ref().unwrap();
        let (col_ptr, row_idx, nnz) = (built.col_ptr.clone(), built.row_idx.clone(), built.nnz());

        // Second factorization, same pattern, (3,3) now an exact zero.
        let second = [4.0, 1.0, 5.0, 1.0, 0.0, 1.0, 1.0];
        s.values_array_mut().copy_from_slice(&second);
        let mut rhs = [1.0, 1.0, 1.0];
        assert_eq!(
            s.multi_solve(true, &SLOT_IRN, &SLOT_JCN, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );

        let refilled = s.matrix.as_ref().unwrap();
        let rebuilt =
            CscMatrix::from_triplets(3, &s.rows_0, &s.cols_0, &s.values).expect("valid triplets");
        assert_eq!(
            refilled.nnz(),
            nnz,
            "the zeroed diagonal's slot was dropped by the refill"
        );
        assert_eq!(refilled.col_ptr, col_ptr, "col_ptr moved under the refill");
        assert_eq!(refilled.row_idx, row_idx, "row_idx moved under the refill");
        assert_eq!(refilled.col_ptr, rebuilt.col_ptr);
        assert_eq!(refilled.row_idx, rebuilt.row_idx);
        // Integer-valued and exactly representable, so the duplicate sum is
        // order-independent and this can be a bit-wise comparison.
        assert_eq!(refilled.values, rebuilt.values);
        assert_eq!(
            refilled.values[5], 0.0,
            "the (3,3) entry must still be stored, as an explicit zero"
        );
    }

    /// The refilled matrix is not just structurally right — the second
    /// solve answers the second matrix. Guards against a permutation that
    /// scatters values into the wrong slots (a transposed or stale map
    /// would still pass the structural check above).
    #[test]
    fn refill_solves_the_new_matrix_not_the_old_one() {
        let mut s = FeralSolverInterface::new();
        assert_eq!(
            s.initialize_structure(3, 7, &SLOT_IRN, &SLOT_JCN),
            ESymSolverStatus::Success
        );

        let first = [4.0, 1.0, 5.0, 1.0, 3.0, 1.0, 1.0];
        s.values_array_mut().copy_from_slice(&first);
        let mut rhs = [7.0, 7.0, 6.0];
        assert_eq!(
            s.multi_solve(true, &SLOT_IRN, &SLOT_JCN, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        check_residual(&dense_from_triplets(&first), &rhs, &[7.0, 7.0, 6.0]);

        // Zeroing the (3,3) diagonal flips this to an indefinite matrix,
        // so a stale factor cannot pass by accident.
        let second = [4.0, 1.0, 5.0, 1.0, 0.0, 1.0, 1.0];
        s.values_array_mut().copy_from_slice(&second);
        let b = [1.0, -2.0, 3.0];
        let mut rhs = b;
        assert_eq!(
            s.multi_solve(true, &SLOT_IRN, &SLOT_JCN, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        check_residual(&dense_from_triplets(&second), &rhs, &b);
    }

    /// `A x` must reproduce `b`.
    fn check_residual(a: &[[f64; 3]; 3], x: &[Number], b: &[f64; 3]) {
        for i in 0..3 {
            let ax: f64 = (0..3).map(|j| a[i][j] * x[j]).sum();
            assert!(
                (ax - b[i]).abs() < 1e-9,
                "row {i}: A·x = {ax}, b = {} (x = {x:?})",
                b[i]
            );
        }
    }

    /// `initialize_structure` is the only place the pattern can change, so
    /// it must drop the cached permutation. Re-initializing with a
    /// *different* pattern and solving proves the stale map is gone — a
    /// retained one would index into the previous matrix's slots.
    #[test]
    fn re_initializing_the_structure_invalidates_the_cached_permutation() {
        let mut s = FeralSolverInterface::new();
        assert_eq!(
            s.initialize_structure(3, 7, &SLOT_IRN, &SLOT_JCN),
            ESymSolverStatus::Success
        );
        s.values_array_mut()
            .copy_from_slice(&[4.0, 1.0, 5.0, 1.0, 3.0, 1.0, 1.0]);
        let mut rhs = [1.0, 1.0, 1.0];
        assert_eq!(
            s.multi_solve(true, &SLOT_IRN, &SLOT_JCN, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        assert!(s.slot.is_some());

        // A different pattern (and a different dimension) on the same
        // interface: the 2×2 SPD `[[2,1],[1,3]]`, solved against (3, 4).
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        assert!(s.slot.is_none(), "stale permutation survived re-init");
        assert!(s.matrix.is_none(), "stale matrix survived re-init");

        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        assert!((rhs[0] - 1.0).abs() < 1e-12, "x0 = {}", rhs[0]);
        assert!((rhs[1] - 1.0).abs() < 1e-12, "x1 = {}", rhs[1]);
    }

    /// Many refills in a row stay correct — the fast path is idempotent
    /// and does not accumulate across calls (the `fill(0.0)` before the
    /// scatter is what makes the `+=` duplicate-summing safe to repeat).
    #[test]
    fn repeated_refills_do_not_accumulate() {
        let mut s = FeralSolverInterface::new();
        assert_eq!(
            s.initialize_structure(3, 7, &SLOT_IRN, &SLOT_JCN),
            ESymSolverStatus::Success
        );
        let vals = [4.0, 1.0, 5.0, 1.0, 3.0, 1.0, 1.0];
        for i in 0..4 {
            s.values_array_mut().copy_from_slice(&vals);
            let b = [1.0, 2.0, 3.0];
            let mut rhs = b;
            assert_eq!(
                s.multi_solve(true, &SLOT_IRN, &SLOT_JCN, 1, &mut rhs, false, 0),
                ESymSolverStatus::Success,
                "factorization {i}"
            );
            let rebuilt = CscMatrix::from_triplets(3, &s.rows_0, &s.cols_0, &s.values).unwrap();
            assert_eq!(
                s.matrix.as_ref().unwrap().values,
                rebuilt.values,
                "values drifted on factorization {i}"
            );
            check_residual(&dense_from_triplets(&vals), &rhs, &b);
        }
    }

    /// gh#710 acceptance (b): the `feral_refine_steps` cap has to reach the
    /// **multi-RHS** back-solve, not only the single-RHS one.
    ///
    /// This matters because of where the IPM actually spends its solves.
    /// `PdFullSpaceSolver`'s predictor-corrector step hands the augmented
    /// system two right-hand sides at once, and feral routes wide and narrow
    /// `nrhs` through different entry points — `solve_many_refined_into` vs
    /// `solve_refined_into` — with a further BLAS3 dispatch inside the wide
    /// one (feral#179 found the narrow arm had duplicated that dispatch). A
    /// cap plumbed into only one arm would leave the path that carries the
    /// IPM's step uncapped, and the back-solve time gh#698 measured would not
    /// move where it was measured.
    ///
    /// The check is behavioural rather than a spy on the call count: on a
    /// system whose corrections are large enough to see, the residual must
    /// fall as the budget rises, and must fall on both columns of an
    /// `nrhs = 2` solve, not just the first.
    #[test]
    fn refine_steps_cap_reaches_the_two_rhs_path() {
        // Hilbert n=8: symmetric, dense, condition ~1e10 — one back-solve
        // lands around 1e-7 and refinement still has several digits to
        // recover. Deliberately not larger: past n≈16 Hilbert is numerically
        // singular in double precision, refinement stops improving on the
        // first step, and the test would pass vacuously.
        const N: usize = 8;
        let a = |i: usize, j: usize| 1.0 / ((i + j + 1) as f64);
        let (mut irn, mut jcn, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..N {
            for j in 0..=i {
                irn.push((i + 1) as Index);
                jcn.push((j + 1) as Index);
                vals.push(a(i, j));
            }
        }
        // Two distinct right-hand sides, column-major, exactly as
        // `multi_solve` takes them on the predictor-corrector path.
        let rhs0: Vec<Number> = (0..2 * N).map(|k| ((k % 7) as f64) - 3.0).collect();

        // Residual against the *original* matrix, per column, infinity norm.
        let resid = |x: &[Number], nrhs: usize| -> Vec<f64> {
            (0..nrhs)
                .map(|c| {
                    let (xc, bc) = (&x[c * N..(c + 1) * N], &rhs0[c * N..(c + 1) * N]);
                    (0..N)
                        .map(|i| ((0..N).map(|j| a(i, j) * xc[j]).sum::<f64>() - bc[i]).abs())
                        .fold(0.0, f64::max)
                })
                .collect()
        };

        let solve_with_cap = |k: usize, nrhs: Index| -> Vec<Number> {
            let mut s = FeralSolverInterface::with_config(FeralConfig {
                refine: true,
                refine_max_steps: k,
                ..FeralConfig::default()
            });
            assert_eq!(
                s.initialize_structure(N as Index, vals.len() as Index, &irn, &jcn),
                ESymSolverStatus::Success
            );
            s.values_array_mut().copy_from_slice(&vals);
            let mut rhs = rhs0[..N * nrhs as usize].to_vec();
            assert_eq!(
                s.multi_solve(true, &irn, &jcn, nrhs, &mut rhs, false, 0),
                ESymSolverStatus::Success
            );
            rhs
        };

        // Both arms: nrhs=1 is the single-RHS entry point, nrhs=2 is the one
        // the predictor-corrector step actually uses.
        for nrhs in [1usize, 2] {
            let budgets = [0usize, 1, 2, 3, feral::DEFAULT_REFINE_MAX_STEPS];
            let curves: Vec<Vec<f64>> = budgets
                .iter()
                .map(|&k| resid(&solve_with_cap(k, nrhs as Index), nrhs))
                .collect();
            for col in 0..nrhs {
                // Monotone in the budget — feral's best-iterate contract says
                // a larger cap can never return a worse answer.
                for w in curves.windows(2) {
                    assert!(
                        w[1][col] <= w[0][col],
                        "nrhs={nrhs} column {col}: raising the refinement \
                         budget made the residual worse ({:e} -> {:e}); the \
                         best-iterate contract is not holding",
                        w[0][col],
                        w[1][col],
                    );
                }
                // ...and the cap is not being ignored: the full budget must
                // actually beat the zero budget on *every* column, or the
                // knob is not reaching this arm at all.
                let (first, last) = (curves[0][col], curves[curves.len() - 1][col]);
                assert!(
                    last < first,
                    "nrhs={nrhs} column {col}: a {}-step budget left the same \
                     residual as a 0-step one ({first:e} vs {last:e}) — \
                     `refine_max_steps` is not reaching this back-solve arm \
                     (gh#710)",
                    feral::DEFAULT_REFINE_MAX_STEPS,
                );
            }
        }
    }

    /// `relative_residual` reads the **lower** triangle of a symmetric
    /// matrix, so every stored off-diagonal has to land in two rows of the
    /// product. Checked against a dense reference rather than a
    /// hand-computed number, so the mirroring is what is being tested and
    /// not an arithmetic transcription.
    #[test]
    fn the_residual_probe_mirrors_the_stored_lower_triangle() {
        // [[4, 1, 1], [1, 5, 1], [1, 1, 3]], lower triangle only.
        let dense = [[4.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 3.0]];
        let rows = [0usize, 1, 1, 2, 2, 2];
        let cols = [0usize, 0, 1, 0, 1, 2];
        let vals = [4.0, 1.0, 5.0, 1.0, 1.0, 3.0];
        let a = CscMatrix::from_triplets(3, &rows, &cols, &vals).unwrap();

        let b = [1.0, -2.0, 3.0];
        // A deliberately wrong `x`, so the residual is large and a probe
        // that dropped the mirrored half would land somewhere else.
        let x = [0.5, 0.25, -1.0];
        let mut scratch = vec![0.0; 3];
        let got = relative_residual(&a, &b, &x, &mut scratch).expect("finite, b nonzero");

        let r: Vec<f64> = (0..3)
            .map(|i| b[i] - (0..3).map(|j| dense[i][j] * x[j]).sum::<f64>())
            .collect();
        let want = r.iter().map(|v| v * v).sum::<f64>().sqrt()
            / b.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (got - want).abs() <= 1e-15 * want,
            "probe {got:e} vs dense reference {want:e}",
        );

        // An exact solve reads as ~0, which is what makes the comparison
        // against a target meaningful in the first place.
        let x_exact = [0.12000000000000005, -0.6599999999999999, 1.1799999999999997];
        let got = relative_residual(&a, &b, &x_exact, &mut scratch).expect("finite");
        assert!(got < 1e-15, "exact solve read as {got:e}");

        // `b = 0` makes the ratio undefined; the caller reads `None` as
        // "cannot certify" and refines.
        assert!(relative_residual(&a, &[0.0; 3], &x, &mut scratch).is_none());
    }

    /// gh#710 follow-on: `FeralConfig::refine_target` decides *whether* the
    /// refinement runs, where `refine_max_steps` only decides how long it
    /// runs once started. Both back-solve arms have to honour it — the
    /// predictor-corrector step is the `nrhs = 2` one, and a target plumbed
    /// into only the narrow arm would leave the IPM's hot path refining
    /// unconditionally.
    ///
    /// Behavioural, and bit-exact on purpose: the check is not "the answer
    /// got worse" (which is a tolerance argument) but "this is *the same
    /// answer* the unrefined path returns", which can only be true if the
    /// refinement was skipped outright.
    #[test]
    fn the_refine_target_decides_whether_refinement_runs_on_both_arms() {
        // Hilbert n=8 again: conditioned so one back-solve lands around
        // 1e-7 and refinement has digits left to recover, so "refined" and
        // "unrefined" are distinguishable answers rather than the same one.
        const N: usize = 8;
        let a = |i: usize, j: usize| 1.0 / ((i + j + 1) as f64);
        let (mut irn, mut jcn, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..N {
            for j in 0..=i {
                irn.push((i + 1) as Index);
                jcn.push((j + 1) as Index);
                vals.push(a(i, j));
            }
        }
        let rhs0: Vec<Number> = (0..2 * N).map(|k| ((k % 7) as f64) - 3.0).collect();

        let solve_with = |cfg: FeralConfig, nrhs: Index| -> Vec<Number> {
            let mut s = FeralSolverInterface::with_config(cfg);
            assert_eq!(
                s.initialize_structure(N as Index, vals.len() as Index, &irn, &jcn),
                ESymSolverStatus::Success
            );
            s.values_array_mut().copy_from_slice(&vals);
            let mut rhs = rhs0[..N * nrhs as usize].to_vec();
            assert_eq!(
                s.multi_solve(true, &irn, &jcn, nrhs, &mut rhs, false, 0),
                ESymSolverStatus::Success
            );
            rhs
        };

        for nrhs in [1 as Index, 2] {
            let refined = solve_with(FeralConfig::default(), nrhs);
            let unrefined = solve_with(
                FeralConfig {
                    refine: false,
                    ..FeralConfig::default()
                },
                nrhs,
            );
            assert_ne!(
                refined, unrefined,
                "nrhs={nrhs}: refinement changed nothing on this matrix, so \
                 the test cannot tell the two paths apart",
            );

            // A target of 0 is the shipped default and must be inert: the
            // pre-check is skipped entirely and every back-solve refines.
            assert_eq!(
                solve_with(
                    FeralConfig {
                        refine_target: 0.0,
                        ..FeralConfig::default()
                    },
                    nrhs
                ),
                refined,
                "nrhs={nrhs}: a target of 0 changed the answer — the default \
                 is not inert",
            );

            // A target no solve can miss accepts the first answer, so the
            // result is the unrefined one, bit for bit.
            assert_eq!(
                solve_with(
                    FeralConfig {
                        refine_target: 1.0,
                        ..FeralConfig::default()
                    },
                    nrhs
                ),
                unrefined,
                "nrhs={nrhs}: an unmissable target still refined — \
                 `refine_target` is not reaching this back-solve arm",
            );

            // ...and a target no solve can meet must still refine, which is
            // what keeps the check from being a disguised `refine = no`
            // (pounce gh#590: the noise-floor LP needs the refinement it
            // asks for, and gets it, because its residuals sit above any
            // target worth setting).
            assert_eq!(
                solve_with(
                    FeralConfig {
                        refine_target: f64::MIN_POSITIVE,
                        ..FeralConfig::default()
                    },
                    nrhs
                ),
                refined,
                "nrhs={nrhs}: an unreachable target skipped the refinement",
            );
        }
    }

    /// Guard for [`FERAL_BITWISE_MULTI_SOLVE_MAX_NRHS`].
    ///
    /// `LowRankAugSystemSolver` batches its SMW correction columns into one
    /// `multi_solve` purely to save time (gh#729). That is only sound while
    /// the batched answer is *bit-identical* to the per-column one: the
    /// batch sits inside an iteration whose trajectory must not move, and a
    /// tolerance-legal perturbation there can select a different local
    /// optimum on a nonconvex problem. MA57 reassociates and does move
    /// `pooling_rt2stp` to an objective 25% worse while still reporting
    /// `Optimal Solution Found`, which is why the batching is gated on this
    /// predicate rather than applied unconditionally.
    ///
    /// feral is bit-identical only below its private
    /// `BLAS3_NRHS_THRESHOLD` (32 in 0.17.0); above it a blocked TRSM/GEMM
    /// panel kernel runs. Our ceiling is a conservative 16, and this test is
    /// what keeps that an argument rather than a hope — it fails if feral
    /// ever lowers the threshold to or below our ceiling, or changes the
    /// narrow arm to reassociate. Run at the **default** config, refinement
    /// included, because that is the configuration the gate actually admits.
    #[test]
    fn multi_solve_bitwise_matches_single_solve_at_the_documented_ceiling() {
        // A 2-D 5-point Laplacian, NOT a tridiagonal band. Bandwidth is
        // what decides whether this test can see anything: a tridiagonal
        // matrix eliminates one row per supernode, so feral's blocked
        // TRSM/GEMM panel degenerates to the same scalar operations as the
        // rank-1 cascade and the two paths agree bit-for-bit at *every*
        // `nrhs` — the guard would pass with the ceiling set anywhere.
        // Nested dissection on a 2-D grid produces genuinely wide
        // separators, so the panel kernel runs as itself.
        const K: usize = 24;
        const N: usize = K * K;
        let (mut irn, mut jcn, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        let mut push = |i: usize, j: usize, v: Number| {
            irn.push((i + 1) as Index);
            jcn.push((j + 1) as Index);
            vals.push(v);
        };
        for r in 0..K {
            for c in 0..K {
                let i = r * K + c;
                push(i, i, 4.0 + ((r + c) % 5) as Number * 0.25);
                if c + 1 < K {
                    push(i + 1, i, -1.0 - (r % 3) as Number * 0.125);
                }
                if r + 1 < K {
                    push(i + K, i, -1.0 - (c % 3) as Number * 0.125);
                }
            }
        }
        let nrhs_max = FERAL_BITWISE_MULTI_SOLVE_MAX_NRHS;
        // Irrational-ish, non-repeating entries: a right-hand side of small
        // integers can be reassociation-insensitive by luck and pass a
        // bit-identity check that a real RHS would fail.
        let rhs_all: Vec<Number> = (0..N * nrhs_max)
            .map(|k| ((k as Number) * 0.7390851332151607).sin() * 3.0 + 0.5)
            .collect();

        let solve = |nrhs: usize, rhs: &[Number], refine: bool| -> Vec<Number> {
            let mut s = FeralSolverInterface::with_config(FeralConfig {
                refine,
                ..FeralConfig::default()
            });
            assert_eq!(
                s.initialize_structure(N as Index, vals.len() as Index, &irn, &jcn),
                ESymSolverStatus::Success
            );
            s.values_array_mut().copy_from_slice(&vals);
            let mut buf = rhs.to_vec();
            assert_eq!(
                s.multi_solve(true, &irn, &jcn, nrhs as Index, &mut buf, false, 0),
                ESymSolverStatus::Success
            );
            buf
        };

        // `refine = false` isolates the substitution kernel, which is what
        // the ceiling is a statement about; the default arm is the
        // configuration the gate actually admits in production. Refinement
        // could in principle drive two differing solves back onto the same
        // answer, so checking only the default arm would be a weaker claim
        // than the constant makes.
        for refine in [false, true] {
            for nrhs in 2..=nrhs_max {
                let rhs = &rhs_all[..N * nrhs];
                let batched = solve(nrhs, rhs, refine);
                let looped: Vec<Number> = (0..nrhs)
                    .flat_map(|c| solve(1, &rhs[c * N..(c + 1) * N], refine))
                    .collect();
                assert_eq!(
                    batched, looped,
                    "feral's nrhs={nrhs} solve (refine={refine}) is no longer \
                     bit-identical to looping single-RHS; \
                     FERAL_BITWISE_MULTI_SOLVE_MAX_NRHS \
                     ({FERAL_BITWISE_MULTI_SOLVE_MAX_NRHS}) is too high and the \
                     gh#729 SMW batching is silently perturbing trajectories"
                );
            }
        }

        // The predicate must actually say `false` past the ceiling, or the
        // constant is documentation and the gate admits everything.
        let s = FeralSolverInterface::default();
        assert!(s.multi_solve_matches_single_solve(nrhs_max));
        assert!(!s.multi_solve_matches_single_solve(nrhs_max + 1));
    }
}
