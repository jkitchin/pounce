//! Main optimization loop — port of
//! `Algorithm/IpIpoptAlg.{hpp,cpp}`.
//!
//! Phase 7 ships the loop scaffold matching `Optimize()` lines
//! 292-563 in upstream. The body invokes:
//!
//!   1. `IterateInitializer::set_initial_iterates`
//!   2. (loop) `OutputIteration` → `CheckConvergence` →
//!      `UpdateBarrierParameter` → `UpdateHessian` →
//!      `ComputeSearchDirection` → `ComputeAcceptableTrialPoint` →
//!      `AcceptTrialPoint`
//!   3. `correct_bound_multiplier` (kappa_sigma) per `MAIN_LOOP.md`
//!      §"Bound multiplier reset" lines 1055-1134
//!   4. exception → `SolverReturn` mapping per the table in
//!      `MAIN_LOOP.md`.
//!
//! The NLP handle and search-direction calculator are optional:
//! when both are present, `iterate()` computes a real Newton step and
//! drives the line search. Without them, `iterate()` runs the bookkeeping
//! pieces (mu update, hessian update, conv check, kappa_sigma reset)
//! and is exercised by structural unit tests. The full path lights up
//! once `pounce-nlp::OrigIpoptNLP` lands.

use crate::alg_builder::AlgorithmBundle;
use crate::conv_check::r#trait::ConvergenceStatus;
use crate::intermediate::{CtxGuard, IntermediateContext};
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iter_dump::IterDumper;
use crate::iterate_dump::emit_record as emit_iterate_record;
use crate::kkt::pd_search_dir_calc::PdSearchDirCalc;
use crate::line_search::backtracking::Outcome;
use crate::restoration::{RestorationOutcome, RestorationPhase};
use pounce_common::diagnostics::DiagnosticsState;
use pounce_common::types::{Index, Number};
use pounce_linalg::Vector;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::return_codes::AlgorithmMode;
use pounce_nlp::tnlp::{IpoptCq as TnlpIpoptCq, IpoptData as TnlpIpoptData, IterStats, TNLP};
use std::cell::RefCell;
use std::rc::Rc;

/// Dual-divergence guard (pounce#246): only dual-infeasibility growth in the
/// *elevated* regime (`inf_du` above this) counts toward the streak, so the
/// noisy early iterations of a normal solve never build one.
const DUAL_DIV_COUNT_FLOOR: Number = 1e2;
/// Dual-divergence guard: the guard fires only once `inf_du` is this large in
/// absolute terms — well above the transient peaks a converging solve reaches
/// (e.g. the least-square-init path recovers from ~6e9 on emfl050), and below
/// the ~1e10 where the KKT factorizations begin to choke, so the diversion to
/// restoration happens *before* the seconds-long factorizations start.
const DUAL_DIV_FIRE_TOL: Number = 1e8;

/// gh#884 — the scale-relative search direction below which the iterate
/// counts as **settled** for the dual-divergence-retry signature.
///
/// Measured on `d89771bc`, minimum over the iterates where the primal is
/// already converged: `qpec_small`/`ncp_eq`/origin reaches `8.6e-14`
/// through the `.nl` path and `4.3e-8` through a Rust TNLP, while
/// `ralph1`/`direct`/origin — which *must not* fire, because no
/// sign-feasible multiplier exists at its origin and failing there is
/// correct — bottoms out at `7.2e-3`. Five orders of separation; `1e-5`
/// sits near the middle of it in log terms. The census behind this, and
/// the corpus fixtures that come closest on either side, are in
/// `dev-notes/mpcc-biactive-dual-divergence.md`.
const DUAL_DIV_RETRY_STEP_TOL: Number = 1e-5;

/// gh#884 — the unscaled `‖∇L‖∞` floor for that signature.
///
/// Deliberately the same `1e2` as [`DUAL_DIV_COUNT_FLOOR`], and for the
/// same reason: below it a solve is merely mid-flight. It is what
/// excludes `eigena2` on the L-BFGS leg, which reaches a settled step of
/// `7.9e-9` but at an unscaled dual of only `37`.
pub(crate) const DUAL_DIV_RETRY_DU_FLOOR: Number = 1e2;

/// gh#884 — primal infeasibility below which the primal counts as
/// converged for that signature. The failure mode is defined by the
/// primal being *done* while the duals run away, so this conjunct is
/// what separates it from an ordinary struggling solve.
const DUAL_DIV_RETRY_PRIMAL_TOL: Number = 1e-8;

/// gh #534 — how many consecutive outer NLP errors the progress test reads.
/// Four samples give three ratios: enough that a single lucky step cannot pass
/// the test, short enough to still be inside the endgame it is meant to
/// recognise. `eigena2`'s quoted tail is exactly four iterations long
/// (`1.19e-5 → 2.96e-6 → 7.38e-7 → 1.84e-7`).
const DECLINE_PROGRESS_SAMPLES: usize = 4;
/// gh #534 — default `resto_decline_progress_ratio`: every one of those ratios
/// must be at least this contraction for the decline to be deferred. `eigena2`
/// quarters (ratio `0.249`) and passes; `eigenb2`'s tail *rises*
/// (`1.88e-7, 2.69e-7, 2.89e-7, 2.93e-7`) and fails, which is the intended
/// split — the issue calls `eigenb2` a plausible genuine stall and the guard
/// plausibly right there.
const DEFAULT_DECLINE_PROGRESS_RATIO: Number = 0.5;
/// gh #534 — outer iterations a deferred continuation gets to produce a strict
/// certificate before it is cut and the floor reported. `eigena2`'s
/// extrapolation needs three; ten leaves room for a slower but still genuine
/// endgame while keeping the cost of a lost bet bounded and small.
const DECLINE_CONTINUATION_BUDGET: Index = 10;
/// gh #534 — default for `resto_decline_deferrals`. One deferral is enough for
/// the reported case (the continuation either converges within the budget or it
/// does not); more entries would mostly re-bet on a point the first bet already
/// failed to improve.
const DEFAULT_RESTO_DECLINE_DEFERRALS: usize = 1;
/// gh #797 — default for `neg_curv_escapes`. One is enough for the reported
/// shape: the escape lands on a point whose reduced Hessian *is* positive
/// definite, so the probe declines there and a second escape would have nothing
/// to spend itself on. It is also the conservative default — each escape is a
/// separate bet, and while none of them can return a worse point than the
/// certificate it left, each costs its own continuation budget.
/// Default `limited_memory_ls_failure_restarts` (gh #818): **off**. The
/// rung is available, and it is not what fixes gh #818.
///
/// It shipped in the first draft of this work defaulted to one, on a
/// measurement taken before `ALPHA_INTERP_MIN_TRIALS` existed: with the
/// interpolation firing on every trial, a line search failed often
/// enough that standing in front of the restoration hand-off was worth
/// something. Gating the interpolation removed most of those failures,
/// and re-measuring the rung on top of the gate turned the trade
/// negative. `scripts/sweep-fixtures.sh` against `a5e0a837`, both with
/// the gate, rung off against rung on:
///
/// | fixture | rung off | rung on |
/// |---|---|---|
/// | `pooling_rt2stp` | `ErrorInStepComputation`/716 — *unmoved from `main`* | `ErrorInStepComputation`/**744** |
/// | `infeasible_square_scaled_1em4` | `InfeasibleProblemDetected`/24 — *unmoved from `main`* | **26** |
/// | `deb7` | `ErrorInStepComputation`/1010 | **`RestorationFailed`**/460 |
/// | `eigena2` | `ErrorInStepComputation`/201 | **`SolvedToAcceptableLevel`**/174 |
/// | `issue_508_infeasible_gap_1em4` | `InfeasibleProblemDetected`/79 | 76 |
///
/// **This ledger is not the one that set the default.** At the gate of
/// 5 an earlier revision shipped, the rung cost iterations on both
/// `eigena2` and `infeasible_square_scaled_1em4` to the same verdict,
/// and that pair is what kept it off. At 6, `eigena2` *gains* a
/// reportable point. What still argues for off is narrower: the rung
/// moves two fixtures off the numbers they have on `main` at no benefit
/// (`pooling_rt2stp`, `infeasible_square_scaled_1em4`), and it changes
/// `deb7`'s verdict rather than shortening it — a different answer, not
/// a faster one. Turning it on is a trajectory change over the whole
/// corpus and needs its own `scripts/sweep-fixtures.sh` run to justify;
/// **that case has improved and is worth re-opening.** Every
/// `issue_818_*` test in `pounce-rs` passes with the rung compiled out
/// — the interpolation is the fix, not this.
///
/// Left in the tree rather than deleted because the reasoning behind it
/// is sound and unaddressed elsewhere: a restoration phase entered at a
/// feasible point has no constraint violation to minimize and cannot
/// help. Some model will want it. Setting the option to a positive value
/// enables it — and note that setting it *at all*, including to 0, opts
/// out of the `Solved_To_Acceptable_Level` re-solve, because it is a
/// [`TERMINATION_POLICY_OPTIONS`](crate::application) key.
const DEFAULT_LBFGS_LS_FAILURE_RESTARTS: usize = 0;

const DEFAULT_NEG_CURV_ESCAPES: usize = 1;
/// gh #797 — outer iterations a negative-curvature escape gets to produce a
/// certificate of its own before it is cut and the stationary point reported.
/// Generous relative to gh #534's ten, because the escape deliberately lands
/// far from the point it left (a full fraction-to-the-boundary step) and the
/// continuation is a fresh endgame rather than the tail of one already in
/// progress.
const NEG_CURV_CONTINUATION_BUDGET: Index = 30;
/// gh #797 — cap on the escape step as a multiple of `1 + ‖(x, s)‖∞`. The
/// probe's direction has unit infinity-norm, so this bounds the escape by the
/// iterate's own scale; an absolute cap would mean different things on
/// differently scaled models. Only binds when the fraction-to-the-boundary rule
/// does not, i.e. when nothing in the direction runs into a bound.
const NEG_CURV_MAX_STEP_FACTOR: Number = 10.0;
/// gh #797 — backtracking steps available to the escape, and the ratio between
/// them. `0.5^12 ≈ 2.4e-4` of the boundary step, past which a direction that
/// still fails the decrease test is not one worth taking.
const NEG_CURV_BACKTRACKS: usize = 12;
const NEG_CURV_BACKTRACK_FACTOR: Number = 0.5;
/// gh #797 — Armijo factor on the escape's *second-order* decrease model. The
/// gradient is (near) zero at a stationary point, so the model is
/// `½α²dᵀ(W + Σ)d` and this is the fraction of it the trial must actually
/// realise. Mirrors the line search's `eta_phi` in role, not in value: the
/// quantity being tested is curvature, and a nonlinear objective gives back
/// less of a quadratic model's prediction than a linear one gives of a
/// first-order model's.
const NEG_CURV_ARMIJO: Number = 0.1;

/// Number of `+1` entries across an activity-sign fingerprint — the
/// bounds the barrier currently treats as active.
///
/// `element_wise_max` against zero maps `{-1, 0, +1}` to `{0, 0, 1}`, so
/// the sum is an exact integer count and an exact tie (`z_i == s_i`,
/// sign `0`) reads as inactive rather than as half a bound.
///
/// A non-finite slack or multiplier propagates to a non-finite sum,
/// which the saturating cast reports as `0`. That is not guarded: the
/// event is emitted just ahead of the convergence check that rejects a
/// non-finite iterate outright, so the only run it can mislabel is one
/// about to exit `InvalidNumberDetected`.
fn count_active(signs: &[Box<dyn Vector>]) -> Index {
    let mut total = 0.0;
    for block in signs {
        let mut hit = block.make_new();
        hit.set(0.0);
        hit.element_wise_max(&**block);
        total += hit.sum();
    }
    total as Index
}

/// Number of indices whose activity sign differs between two
/// fingerprints, or `None` when the two do not describe the same
/// index space.
///
/// `|now − prev|` is `0`, `1` or `2` per index, so clamping it at one
/// before summing counts each moved index exactly once — including the
/// tie-crossing `0 → ±1`, which a plain `sum / 2` would have counted as
/// a half. The shape check is not defensive dressing: a fingerprint
/// carried across a problem whose bound blocks changed length would
/// otherwise compare two different index spaces and report a number
/// that means nothing, which is the failure mode this crate's index
/// work exists to make loud rather than silent.
fn count_activity_changes(now: &[Box<dyn Vector>], prev: &[Box<dyn Vector>]) -> Option<Index> {
    if now.len() != prev.len() {
        return None;
    }
    let mut total = 0.0;
    for (a, b) in now.iter().zip(prev.iter()) {
        if a.dim() != b.dim() {
            return None;
        }
        let mut diff = a.make_new_copy();
        diff.axpy(-1.0, &**b);
        diff.element_wise_abs();
        let mut ones = a.make_new();
        ones.set(1.0);
        diff.element_wise_min(&*ones);
        total += diff.sum();
    }
    Some(total as Index)
}

pub struct IpoptAlgorithm {
    pub data: IpoptDataHandle,
    pub cq: IpoptCqHandle,
    pub bundle: AlgorithmBundle,
    /// Optional NLP handle. Required for any step that evaluates
    /// problem functions or pulls bound expansion matrices (init,
    /// search direction, line-search trial-point evaluation). Absent
    /// in the structural unit tests of Phases 5-6.
    pub nlp: Option<Rc<RefCell<dyn IpoptNlp>>>,
    /// Optional TNLP handle — the user-facing problem. When present,
    /// `iterate()` fires `TNLP::intermediate_callback` once per outer
    /// iteration so callers can monitor progress or request early
    /// termination (returning `false` from the callback surfaces as
    /// `SolverReturn::UserRequestedStop`). Kept separate from `nlp`
    /// because the algorithm-side NLP is the *compressed* `OrigIpoptNlp`
    /// view (fixed-variable elimination, c/d split) while the callback
    /// payload needs to expose the original-coordinate iterate.
    pub tnlp: Option<Rc<RefCell<dyn TNLP>>>,
    /// Set on the *restoration* inner IPM so its callback fires report
    /// `AlgorithmMode::RestorationPhaseMode` (gh#645). Two things hang
    /// off it, both in [`Self::fire_intermediate`]:
    ///
    /// 1. the mode field of the [`IterStats`] payload, which is what
    ///    tells a caller the numbers beside it (`obj`, `inf_pr`,
    ///    `inf_du`, `alpha_*`) describe the min-C1-norm feasibility
    ///    subproblem rather than the user's NLP;
    /// 2. whether the live-inspector `IntermediateContext` is installed
    ///    — and for a restoration fire it deliberately is **not**. The
    ///    inner iterate is a `CompoundVector` over `(x_orig, n, p)`, so
    ///    it does not even have the user's `n`, and the C API's
    ///    `GetIpoptCurrent*` family checks the caller's `n`/`m` against
    ///    the *problem's* registered dimensions rather than the
    ///    context's. Installing this context would sail past that check
    ///    and read a differently-shaped `cq`. The live accessors
    ///    therefore report "no data" during restoration, which is the
    ///    truth: there is no current iterate of the user's problem
    ///    while the subproblem is being solved.
    pub fires_as_restoration: bool,
    /// Previous iteration's bound-activity fingerprint, for the phase
    /// profile emitted on the per-iteration event.
    ///
    /// Held on the algorithm rather than in `IpoptData` because it is
    /// pure telemetry: nothing in the step computation reads it, and a
    /// restoration sub-solve runs its own `IpoptAlgorithm` with its own
    /// fingerprint, which is the scoping the collector already applies
    /// when it drops iterations nested in a `restoration` span.
    ///
    /// `None` until the first iteration that computes one — and note
    /// that the first *captured* iteration therefore reports no change
    /// count at all rather than a zero, since "nothing to compare
    /// against" and "nothing changed" are different readings.
    prev_activity: Option<Vec<Box<dyn Vector>>>,
    /// Search-direction calculator (`PdSearchDirCalc`). Lands once a
    /// concrete `SymLinearSolver` backend (MUMPS / FERAL) is wired
    /// through `AlgBuilder` in Phase 7's tail.
    pub search_dir: Option<PdSearchDirCalc>,
    /// Restoration-phase strategy. Invoked when the line search
    /// returns [`Outcome::Failed`] (port of upstream
    /// `IpBacktrackingLineSearch::ActivateLineSearch`'s resto
    /// fallback). Optional: in its absence, line-search failure maps
    /// directly to [`SolverReturn::RestorationFailure`] so the main
    /// loop's exit-code semantics match upstream's "no resto built"
    /// case.
    pub restoration: Option<Box<dyn RestorationPhase>>,

    /// `kappa_sigma` for the post-AcceptTrialPoint multiplier reset
    /// (`IpIpoptAlg.cpp:correct_bound_multiplier`, line 1055-1134).
    pub kappa_sigma: Number,
    /// `recalc_y` — recompute `y_c`/`y_d` as least-square estimates
    /// once the iterate is feasible enough, instead of carrying the
    /// multipliers the Newton step produced. Upstream registers this
    /// `no`, but its own option text says "If a limited memory
    /// quasi-Newton option is chosen, this is used by default", so the
    /// L-BFGS path auto-enables it (see
    /// `application.rs`). Costs one extra augmented-system solve on
    /// every iteration where it fires.
    ///
    /// It exists because a quasi-Newton model's multipliers are only as
    /// good as the Hessian approximation behind them: L-BFGS can reach
    /// a feasible primal and still fail to drive `inf_du` down, because
    /// the dual step is computed from an approximate `W`. Re-estimating
    /// `y` by least squares side-steps the approximation entirely.
    /// `linear_system_scaling=slack-based` is active, so the
    /// iterate-dependent `s`-block scaling must be refreshed each
    /// iteration. See [`Self::push_slack_scaling`].
    pub slack_based_scaling: bool,
    pub recalc_y: bool,
    /// `recalc_y_feas_tol` — the constraint-violation threshold below
    /// which [`Self::recalc_y`] fires. Upstream default `1e-6`.
    pub recalc_y_feas_tol: Number,
    pub max_iter: Index,
    /// `start_with_resto` — force the feasibility restoration phase in
    /// the first iteration.
    ///
    /// This is an **outer**-loop behaviour, which is where it went wrong
    /// before: the option was threaded from the `OptionsList` through
    /// `AlgorithmBuilder::resto` into `RestoAlgorithmBuilder` and on into
    /// `MinC1NrmDriver`, a field on the *inner* restoration solver, where
    /// there is no first iteration of the outer algorithm to act on. It
    /// was set by everything and read by nothing, so `start_with_resto
    /// yes` was a silent no-op. `unimplemented_options.rs`'s
    /// `the_restoration_switches_reach_the_builder` asserted only that the
    /// value reached the builder — the very "read site populating a field
    /// nobody consumes" its own comment names as the defect to avoid.
    pub start_with_resto: bool,
    /// Initial primal step length offered to the line search at the
    /// top of each iteration. Mirrors `IpBacktrackingLineSearch`'s
    /// fraction-to-the-boundary primal step (with τ = `data.curr_tau`).
    /// In v1.0 the structural value here is 1.0 and the FTB cap is
    /// applied per-component when the line-search driver computes
    /// trial slacks; the simplification holds for non-degenerate runs.
    pub alpha_init: Number,
    /// Tiny-step relative tolerance — port of upstream
    /// `IpBacktrackingLineSearch::tiny_step_tol_` (default `10·EPSILON`).
    /// Step is "tiny" when `max_i |δx_i|/(1+|x_i|) ≤ tiny_step_tol`
    /// (and same for s, and `c_viol ≤ 1e-4`).
    pub tiny_step_tol: Number,
    /// Port of upstream `IpIpoptAlg.cpp` divergence guard: when
    /// `max_i |x_i|` exceeds this threshold the optimization aborts with
    /// `SolverReturn::DivergingIterates`. Default `1e20` matches the
    /// registered `diverging_iterates_tol` option. Catches MESH and
    /// similar cases where the normal-mode IPM heads off to infinity
    /// (orig `f` to ±1e33 by iter 90) before line-search failure forces
    /// a degenerate restoration entry.
    pub diverging_iterates_tol: Number,
    /// #248 divergence persistence — consecutive iterations the primal
    /// iterate has kept *growing* while past `diverging_iterates_tol` on a
    /// structurally unbounded side. A genuine recession ray sustains this;
    /// a transient ill-scaling excursion on a bounded-below problem peaks
    /// and recedes (MINLPLib `jit1`: `|x|` climbs to ~16 then falls back to
    /// ~2.9 at the finite optimum). Reset to zero whenever the iterate is
    /// within the threshold or is not growing.
    divergence_streak: u32,
    /// Consecutive iterations for which the primal divergence guard has
    /// been suppressed because the line search reported
    /// [`crate::line_search::backtracking::BacktrackingLineSearch::in_watchdog`].
    ///
    /// The suppression is a *deferral*, and this is what bounds it. A
    /// watchdog sequence is supposed to end within
    /// `watchdog_trial_iter_max` (default 3) iterations, but the flag can
    /// outlive one: `run_filter_line_search`'s `TinyStep` arm returns
    /// without consulting `in_watchdog`, so a tiny step taken mid-watchdog
    /// hands off to restoration with the flag still set, and
    /// `reset_after_restoration` clears `watchdog_shortened_iter` but not
    /// `in_watchdog`. Rather than change the line search's state machine
    /// (a trajectory change, for a hole this guard need not depend on),
    /// the guard simply stops deferring past
    /// [`Self::WATCHDOG_DEFER_MAX`] and checks the iterate anyway. Reset
    /// to zero on any iteration the guard actually runs.
    watchdog_defer_streak: u32,
    /// Largest `|x|` seen in the current growth run (companion to
    /// [`Self::divergence_streak`]). Zero when no run is active.
    divergence_prev_amax: Number,
    /// #252 objective at the previous over-threshold iterate of the current
    /// growth run (companion to [`Self::divergence_streak`]). A genuine
    /// recession ray drives the (minimized) objective toward `−∞`, so a
    /// diverging iterate only counts toward the streak when the objective is
    /// *still descending* against this reference. A transient ill-scaling
    /// excursion past a finite optimum grows `|x|` while the objective
    /// *worsens* (the linear tail dominates), so it never accumulates the
    /// streak — this is the fix for the unbounded-box (`ub = +∞`) B&B node
    /// subproblems of jit1 that #248's growth-only check still mislabelled
    /// `UNBOUNDED`. `+∞` when no run is active.
    divergence_prev_f: Number,
    /// #252 objective *decrease* at the previous step of the current growth
    /// run (`prev_prev_f − prev_f`), the companion that lets the streak
    /// require the descent to be *non-decelerating*. A recession ray's
    /// per-step objective drop keeps up or accelerates as `|x|` grows
    /// geometrically (`f` is at least linear along the ray); an excursion
    /// converging to a finite optimum has a per-step drop that shrinks
    /// toward zero. `NaN` (non-finite) until the run has a first finite
    /// decrease to compare against, which bootstraps the check.
    divergence_prev_decrease: Number,
    /// #285 recession-ray persistence — consecutive iterations for which the
    /// *checked recession-ray proof* ([`Self::curr_is_recession_ray`]) held
    /// while the primal iterate kept growing. This is a second, independent
    /// unboundedness path that catches a genuine recession ray in
    /// `null(A_eq)` over free variables whose `|x|` grows only *linearly*
    /// (the regularized zero-Hessian step in an equality null space marches
    /// out at a bounded rate), so it never crosses `diverging_iterates_tol`
    /// (`1e20`) within `max_iter` and the geometric-growth
    /// [`Self::divergence_streak`] never accumulates. Reset to zero whenever
    /// the proof fails or the iterate stops growing.
    recession_streak: u32,
    /// Largest `|x|` seen in the current recession-ray run (companion to
    /// [`Self::recession_streak`]). Zero when no run is active.
    recession_prev_amax: Number,
    /// Companion threshold on the dual step — when both primal and dual
    /// steps are tiny in two consecutive iterations the algorithm
    /// declares convergence at the best attainable accuracy. Default
    /// `1e-2` matches upstream.
    pub tiny_step_y_tol: Number,
    /// `dual_diverging_streak` (pounce#246) — number of consecutive
    /// iterations of *growing* dual infeasibility (in the elevated regime,
    /// `inf_du > `[`DUAL_DIV_COUNT_FLOOR`]) that must accumulate before the
    /// dual-divergence guard fires. When the streak reaches the limit and
    /// `inf_du > `[`DUAL_DIV_FIRE_TOL`], the outer routes to restoration.
    ///
    /// **`0` (off) is the default**, set from the option of the same name
    /// (`application.rs`). It defaulted to `15` when introduced; see the option
    /// help in `upstream_options.rs` for why that changed, and
    /// [`Self::honour_best_acceptable_after_dual_guard`] for what protects a
    /// solve when it is enabled. See the guard itself in [`Self::iterate`].
    pub dual_diverging_streak: usize,
    dual_inf_prev: Number,
    dual_growth_streak: usize,
    /// gh#884 — thresholds for the dual-divergence *retry* signature.
    /// Distinct from the pounce#246 guard above in both mechanism and
    /// consequence: that one diverts a running solve to restoration, this
    /// one only *records* that a cold retry is worth attempting after the
    /// solve has already given up. Set from options of the same name; see
    /// [`DUAL_DIV_RETRY_STEP_TOL`] for the measured populations.
    pub dual_divergence_retry_step_tol: Number,
    /// Companion floor on the unscaled dual — see
    /// [`DUAL_DIV_RETRY_DU_FLOOR`].
    pub dual_divergence_retry_du_floor: Number,
    /// Scale-relative magnitude of the most recent search direction,
    /// `max(max_i |δx_i|/(1+|x_i|), max_i |δs_i|/(1+|s_i|))` — the
    /// `detect_tiny_step` measure, kept as a number rather than a
    /// predicate. `INFINITY` before the first direction is computed, so
    /// the signature cannot fire on iteration 0.
    last_step_rel: Number,
    /// gh#884 — sticky: set once the four-conjunct signature is seen at a
    /// single iterate, never cleared. Read by the application layer after
    /// the solve to decide whether a cold retry is authorized. It never
    /// changes a verdict by itself.
    dual_divergence_signature: bool,
    /// Set true when the previous iterate was tagged tiny; on the
    /// second consecutive tiny step the loop sets `data.tiny_step_flag`
    /// so the mu update can attempt to terminate. Mirrors
    /// `IpBacktrackingLineSearch::tiny_step_last_iteration_`.
    pub tiny_step_last_iteration: bool,
    /// Cycle-detection state for [`Self::invoke_restoration`]: the
    /// outer `(x, s)` snapshot from the previous restoration entry,
    /// cleared on any iteration that exits via a normal line-search
    /// accept. When restoration is invoked twice in a row and the
    /// outer iterate has not moved between entries (relative
    /// 2-norm < 1e-10 on both `x` and `s`), the inner resto-IPM is
    /// returning Recovered points indistinguishable from `curr` — a
    /// cycle. Surfaces as `ErrorInStepComputation`. Mirrors the
    /// *intent* of upstream `IpBacktrackingLineSearch.cpp:580-600`'s
    /// almost-feasible-resto guard while staying robust against the
    /// `inf_pr` micro-drift seen on ACOPR14 (delta ~3e-12 per entry,
    /// inf_du essentially constant) where a scalar-`inf_pr` heuristic
    /// fails. Productive single-restoration sequences (BT8, HIMMELBJ,
    /// LINSPANH, LSNNODOC, ODFITS, OET3) clear the snapshot via
    /// `Outcome::Accepted` between entries and are unaffected.
    last_resto_entry_x: Option<Box<dyn Vector>>,
    last_resto_entry_s: Option<Box<dyn Vector>>,
    /// Snapshot of the *recovery* iterate from the previous
    /// restoration. Compared against the next entry's `(x, s)` to
    /// detect "outer made no progress between consecutive resto
    /// invocations". When this distance is below threshold for
    /// several consecutive entries, terminate — catching
    /// slow-non-convergence cycles (ACOPR14, TRO3X3, ACOPR30) where
    /// resto's *inner* moves substantively each call but the *outer*
    /// makes no progress between calls. Cleared on any LS-accepted
    /// step.
    last_resto_recovery_x: Option<Box<dyn Vector>>,
    last_resto_recovery_s: Option<Box<dyn Vector>>,
    /// Count of consecutive restoration entries on which the outer
    /// step (recovery → next-entry) was below the iterate-distance
    /// threshold. Cleared on any LS-accepted step. Limit chosen to
    /// let MAKELA3, HAIFAM, HALDMADS, ROBOT, TENBARS2 — which need
    /// 2-3 consecutive resto entries to recover — pass through.
    resto_no_outer_progress_count: usize,
    /// `resto_decline_deferrals` (gh #534) — how many times the
    /// acceptable-point restoration decline in [`Self::invoke_restoration`] may
    /// be *deferred* on a solve whose NLP error is still contracting. `0`
    /// restores the pre-#534 behaviour (decline immediately, always).
    ///
    /// See [`Self::may_defer_acceptable_decline`] for the progress test and
    /// [`Self::honour_decline_floor`] for what makes a spent deferral harmless.
    pub resto_decline_deferrals: usize,
    /// `resto_decline_progress_ratio` (gh #534) — the contraction each of the
    /// last [`DECLINE_PROGRESS_SAMPLES`]` - 1` iterations must have achieved for
    /// the decline to be deferred. Default
    /// [`DEFAULT_DECLINE_PROGRESS_RATIO`]. A value of `1` admits any
    /// non-increasing window and a large one drops the progress requirement
    /// altogether, which is the "patch the guard and see" experiment the issue
    /// asks for, available without patching.
    pub resto_decline_progress_ratio: Number,
    /// The most recent outer-iteration NLP errors, oldest first, with
    /// [`Self::nlp_err_recent_len`] entries live. Feeds the gh #534 progress
    /// test and nothing else.
    nlp_err_recent: [Number; DECLINE_PROGRESS_SAMPLES],
    nlp_err_recent_len: usize,
    /// gh #534 — deferrals of the acceptable-point decline spent so far.
    decline_deferrals_used: usize,
    /// gh #534 — the iterate the guard would have returned had it not been
    /// deferred. Captured at the *first* deferral only, because that point is
    /// precisely the answer the pre-#534 build reports; it is the floor the
    /// continuation is never allowed to fall below.
    decline_floor: Option<VetoSnapshot>,
    /// gh #534 — outer iteration by which the deferred continuation must have
    /// produced a strict certificate. Past it the continuation is cut and the
    /// floor is reported, so the bet costs a bounded number of iterations.
    decline_deadline_iter: Option<Index>,
    /// `neg_curv_escapes` (gh #797) — how many times a certified stationary
    /// point whose reduced Hessian is not positive semidefinite may be *left*
    /// along a direction of negative curvature instead of reported. `0` restores
    /// the pre-#797 behaviour (report the first-order certificate, whatever its
    /// curvature).
    ///
    /// See [`Self::try_neg_curv_escape`] for the test and the step, and
    /// [`Self::honour_neg_curv_floor`] for what makes a lost bet harmless.
    pub neg_curv_escapes: usize,
    /// `limited_memory_ls_failure_restarts` (gh #818) — how many times a
    /// line-search failure at an *already feasible* point may re-anchor
    /// the quasi-Newton model and retry, instead of handing off to a
    /// restoration phase that has no constraint violation to reduce.
    /// `0` restores the pre-#818 behaviour (always hand off).
    ///
    /// See [`Self::try_reanchor_before_restoration`] for the rung and
    /// what bounds it.
    pub lbfgs_ls_failure_restarts: usize,
    /// gh #818 — re-anchors spent so far this solve.
    lbfgs_ls_restarts_used: usize,
    /// gh #797 — escapes spent so far this solve.
    neg_curv_escapes_used: usize,
    /// Sink for the last `IterStats` handed to the user's
    /// `intermediate_callback` (pounce#870). A second-opinion retry that loses
    /// needs it, so that the trace a consumer accumulated can be made to end
    /// on the iterate actually reported. Set by `IpoptApplication`; `None`
    /// everywhere else.
    pub last_iter_stats_sink: Option<Rc<RefCell<Option<IterStats>>>>,
    /// gh #797 — the certified stationary point the escape left. It is a strict
    /// certificate, so it is the floor the continuation must beat *with a
    /// certificate of its own* to be preferred.
    ///
    /// With more than one escape available this holds the **best** of the
    /// certificates left so far, not the most recent (gh #805). The first entry
    /// is the answer a `neg_curv_escapes = 0` build returns, and it is only
    /// ever displaced by a point that outranks it, so the guarantee holds at
    /// any number of escapes and the floor never moves backwards.
    neg_curv_floor: Option<VetoSnapshot>,
    /// gh #797 — outer iteration by which the escape's continuation must have
    /// produced a certificate. Past it the continuation is cut and the floor
    /// reported, so the bet costs a bounded number of iterations.
    neg_curv_deadline_iter: Option<Index>,
    /// Count of consecutive restoration entries on which the outer
    /// constraint violation at entry was already below `tol` (the
    /// outer optimality tolerance). Matches the *intent* of upstream
    /// `IpBacktrackingLineSearch.cpp:580-600`'s almost-feasible-resto
    /// guard while using a looser cv threshold (`tol` vs `1e-2·tol`)
    /// — catches DECONVBNE's resto-thrash where each cycle re-enters
    /// at cv ≈ 3e-10 < tol with bound multipliers reset to 1, the
    /// outer's σ-blowup explodes inf_du to 1.9e7, alpha-min triggers
    /// resto re-entry, and the (inf_pr, inf_du) post-recovery state
    /// is essentially identical across cycles but `x` drifts enough
    /// that [`Self::last_resto_recovery_x`]-based detection misses.
    /// Cumulative (never cleared on LS-accept), since DECONVBNE's
    /// cycle interleaves R-recoveries with sub-tol accepts that
    /// accomplish no real outer progress. Fires after 3 near-feasible
    /// entries — surfaces as `StopAtAcceptablePoint` since the
    /// recovered point already satisfies constraint feasibility
    /// within `tol`.
    resto_near_feasible_count: usize,
    /// Snapshot of the most recent iterate that the convergence check
    /// flagged "acceptable" (NLP error ≤ `acceptable_tol`). Mirrors
    /// upstream `IpBacktrackingLineSearch::acceptable_iterate_`
    /// (`IpBacktrackingLineSearch.cpp:1286-1310`). Used by
    /// [`Self::restore_acceptable_point`] to roll back when restoration
    /// fails — if such an iterate exists, the algorithm exits with
    /// `SolverReturn::StopAtAcceptablePoint` rather than
    /// `RestorationFailure`. Cleared/refreshed on every iteration that
    /// satisfies the acceptable predicate.
    acceptable_iterate: Option<crate::iterates_vector::IteratesVector>,
    /// The first iterate whose *strict* certificate the masked-scale veto
    /// refused (gh #200), kept so the refusal can be undone verbatim if the
    /// continued run does not do better. Deliberately not the acceptable
    /// snapshot: that one is overwritten unconditionally and drifts.
    vetoed: Option<VetoSnapshot>,
    /// The iterate at which a refused *acceptable-level* termination would have
    /// fired. Held separately from `vetoed` because it restores under a weaker
    /// status, and claiming `Success` for it would over-report.
    vetoed_acceptable: Option<VetoSnapshot>,
    /// Whether a strict refusal has already been *seen*, independent of whether
    /// a snapshot was successfully captured for it.
    ///
    /// This is the first-only latch, held apart from `vetoed` on purpose.
    /// Testing `vetoed.is_none()` instead would let a refusal whose capture
    /// failed be "completed" at a later iterate — the veto flag on the
    /// convergence check is sticky, so it still reads true next pass, and the
    /// fallback would then restore a point that never passed the strict test.
    /// With the latch, a failed capture stays failed and the fallback declines.
    ///
    /// Declining is *not* the baseline outcome — the baseline stopped and
    /// reported a certificate at the uncaptured iterate, and declining fails to
    /// reproduce it. It is the least-bad handling of an unidentifiable baseline,
    /// not a faithful one.
    vetoed_seen: bool,
    /// Same latch for the acceptable-level refusal.
    vetoed_acceptable_seen: bool,
    /// Whether the dual-divergence guard (pounce#246) actually fired this
    /// solve. Gates the *use* of [`Self::best_acceptable`], so solves the guard
    /// never touches behave identically — see
    /// [`Self::honour_best_acceptable_after_dual_guard`].
    dual_guard_fired: bool,
    /// Best (lowest scaled objective) acceptable-quality iterate seen anywhere
    /// in this solve. Recorded unconditionally — including *before* any
    /// diversion, which is the point: the guard returns to the driver before
    /// the recording site on the iteration it fires, so gating the recording on
    /// `dual_guard_fired` would miss everything up to and including the
    /// diversion. Only read when `dual_guard_fired`.
    best_acceptable: Option<VetoSnapshot>,
    /// `kkt_fidelity_tol` (pounce#173), needed here — not just at termination —
    /// because the fallback's tiebreak has to predict the post-solve status
    /// gate. See [`Self::honour_refused_certificate`]. Zero (the default)
    /// disables the gate, and with it every tiebreak effect it has.
    pub kkt_fidelity_tol: Number,
    acceptable_iter_number: Index,
    /// Shared per-solve diagnostics state. `None` unless the CLI
    /// requested `--dump <cat>:<spec>`. When set, the outer loop
    /// advances the state's iter counter and the augmented-system
    /// solver consults it to gate KKT dumps.
    diagnostics: Option<Rc<DiagnosticsState>>,
    /// Optional interactive debugger. Shared (`Rc<RefCell<…>>`) so the
    /// same debugger instance also drives the restoration inner IPM —
    /// one debugger sees both levels. Fired at every
    /// [`crate::debug::Checkpoint`]. See `crate::debug`.
    debug: Option<Rc<RefCell<dyn crate::debug::DebugHook>>>,

    // ---- Restoration-phase audit counters (pounce#12). ----
    //
    // Drained into `SolveStatistics` by `IpoptApplication::optimize_constrained`
    // after the solve completes. Counts are cumulative across the run.
    /// Number of `invoke_restoration` entries.
    pub resto_calls: Index,
    /// Sum of inner-IPM iter counts across every restoration call.
    pub resto_inner_iters: Index,
    /// Number of outer iters that ran in restoration mode (R-line
    /// equivalents in `print_level=5` output).
    pub resto_outer_iters: Index,
    /// Cumulative wall-clock seconds spent inside `perform_restoration`.
    pub resto_wall_secs: Number,

    // ---- Per-iteration history capture (pounce#8, pounce#71). ----
    //
    // The per-iteration trajectory is no longer accumulated on the
    // algorithm: `iterate()` emits a structured `pounce::iteration`
    // event each step, and `pounce_observability::IterCollectorLayer`
    // rebuilds the `IterRecord`s into the active `IterCaptureGuard`
    // that `IpoptApplication` installs around the solve.
    /// When `false`, the per-iteration table that `iterate()` writes
    /// straight to stdout is suppressed. Wired from
    /// `IpoptApplication`'s `print_level` option: level 0 turns this
    /// off (matches upstream's "no console output" contract). Default
    /// `true` so CLI / direct-driver users keep the familiar trace.
    pub print_iter_output: bool,
}

impl IpoptAlgorithm {
    /// Diagnostics from the safeguarded `least_square_init_primal`
    /// initializer step (gh#605). `None` when the step was not run.
    pub fn least_square_init_report(&self) -> Option<crate::init::default::LeastSquareInitReport> {
        self.bundle.init.least_square_report()
    }

    pub fn new(data: IpoptDataHandle, cq: IpoptCqHandle, mut bundle: AlgorithmBundle) -> Self {
        // The builder may pre-populate `bundle.search_dir` when given a
        // `LinearBackendFactory`; lift it onto the algorithm so the
        // iterate body can call into it directly.
        let search_dir = bundle.search_dir.take();
        Self {
            data,
            cq,
            bundle,
            nlp: None,
            tnlp: None,
            fires_as_restoration: false,
            prev_activity: None,
            search_dir,
            restoration: None,
            kappa_sigma: 1e10,
            slack_based_scaling: false,
            recalc_y: false,
            recalc_y_feas_tol: 1e-6,
            max_iter: 3000,
            start_with_resto: false,
            alpha_init: 1.0,
            tiny_step_tol: 10.0 * Number::EPSILON,
            diverging_iterates_tol: 1e20,
            divergence_streak: 0,
            watchdog_defer_streak: 0,
            divergence_prev_amax: 0.0,
            divergence_prev_f: Number::INFINITY,
            divergence_prev_decrease: Number::NAN,
            recession_streak: 0,
            recession_prev_amax: 0.0,
            tiny_step_y_tol: 1e-2,
            dual_diverging_streak: 15,
            dual_inf_prev: 0.0,
            dual_growth_streak: 0,
            dual_divergence_retry_step_tol: DUAL_DIV_RETRY_STEP_TOL,
            dual_divergence_retry_du_floor: DUAL_DIV_RETRY_DU_FLOOR,
            last_step_rel: Number::INFINITY,
            dual_divergence_signature: false,
            tiny_step_last_iteration: false,
            last_resto_entry_x: None,
            last_resto_entry_s: None,
            last_resto_recovery_x: None,
            last_resto_recovery_s: None,
            resto_no_outer_progress_count: 0,
            resto_decline_deferrals: DEFAULT_RESTO_DECLINE_DEFERRALS,
            neg_curv_escapes: DEFAULT_NEG_CURV_ESCAPES,
            neg_curv_escapes_used: 0,
            last_iter_stats_sink: None,
            lbfgs_ls_failure_restarts: DEFAULT_LBFGS_LS_FAILURE_RESTARTS,
            lbfgs_ls_restarts_used: 0,
            neg_curv_floor: None,
            neg_curv_deadline_iter: None,
            resto_decline_progress_ratio: DEFAULT_DECLINE_PROGRESS_RATIO,
            nlp_err_recent: [Number::NAN; DECLINE_PROGRESS_SAMPLES],
            nlp_err_recent_len: 0,
            decline_deferrals_used: 0,
            decline_floor: None,
            decline_deadline_iter: None,
            resto_near_feasible_count: 0,
            acceptable_iterate: None,
            vetoed: None,
            vetoed_acceptable: None,
            dual_guard_fired: false,
            best_acceptable: None,
            vetoed_seen: false,
            vetoed_acceptable_seen: false,
            kkt_fidelity_tol: 0.0,
            acceptable_iter_number: 0,
            diagnostics: None,
            debug: None,
            resto_calls: 0,
            resto_inner_iters: 0,
            resto_outer_iters: 0,
            resto_wall_secs: 0.0,
            print_iter_output: true,
        }
    }

    /// Stash the current iterate as the "last acceptable" backup —
    /// port of `IpBacktrackingLineSearch::StoreAcceptablePoint`
    /// (`IpBacktrackingLineSearch.cpp:1286-1293`).
    fn store_acceptable_point(&mut self) {
        let d = self.data.borrow();
        if let Some(curr) = d.curr.as_ref() {
            self.acceptable_iterate = Some(curr.clone());
            self.acceptable_iter_number = d.iter_count;
        }
    }

    /// Record this outer iteration's NLP error for the gh #534 progress test.
    ///
    /// One push per `iterate()` call, so the samples are consecutive outer
    /// iterations by construction. Deliberately *not* cleared when restoration
    /// recovers: a recovery that helped shows up as continued contraction and a
    /// recovery that hurt shows up as a jump, and the ratio test reads both
    /// correctly without needing to know which happened.
    fn note_nlp_err(&mut self, nlp_err: Number) {
        push_sample(
            &mut self.nlp_err_recent,
            &mut self.nlp_err_recent_len,
            nlp_err,
        );
    }

    /// Whether the last [`DECLINE_PROGRESS_SAMPLES`] outer iterations each cut
    /// the NLP error by at least `resto_decline_progress_ratio` (gh #534).
    ///
    /// The question the restoration-decline guard never asked: *is this solve
    /// still converging?* A full window is required, so the test cannot pass on
    /// a short history — the early iterations of every solve included.
    ///
    /// The test itself lives in the pure [`window_is_contracting`], for the
    /// reason [`ranks_better_within_band`] does: what it must and must not fire
    /// on is stated in the issue as two recorded traces, and those are provable
    /// by deterministic unit test rather than inferable from a solve.
    fn nlp_err_contracting(&self) -> bool {
        if self.nlp_err_recent_len < DECLINE_PROGRESS_SAMPLES {
            return false;
        }
        window_is_contracting(&self.nlp_err_recent, self.resto_decline_progress_ratio)
    }

    /// The live progress window, oldest first, for the gh #534 trace lines.
    fn nlp_err_window_str(&self) -> String {
        let live = &self.nlp_err_recent[..self.nlp_err_recent_len];
        let parts: Vec<String> = live.iter().map(|e| format!("{e:.3e}")).collect();
        format!("[{}]", parts.join(" -> "))
    }

    /// Roll the iterate back to the last acceptable snapshot — port of
    /// `IpBacktrackingLineSearch::RestoreAcceptablePoint`
    /// (`IpBacktrackingLineSearch.cpp:1295-1310`). Returns `true` if a
    /// snapshot was available and applied; `false` otherwise (caller
    /// then surfaces the original failure status).
    fn restore_acceptable_point(&mut self) -> bool {
        let Some(prev) = self.acceptable_iterate.clone() else {
            return false;
        };
        let mut d = self.data.borrow_mut();
        d.set_trial(prev);
        // `accept_trial_point` promotes `trial → curr`, mirroring the
        // upstream sequence `set_trial(...); AcceptTrialPoint();`.
        d.accept_trial_point();
        true
    }

    /// Whether a diverging primal iterate is consistent with the feasible
    /// region actually being *unbounded* (issue #248).
    ///
    /// `DivergingIterates` is Ipopt's unboundedness verdict, but a large
    /// `|x_i|` only proves unboundedness if variable `i` is free to escape
    /// to infinity in the direction it is heading — i.e. it has no finite
    /// bound on that side. This lifts a vector of ones from the compressed
    /// lower/upper bound spaces through the `Px_L` / `Px_U` expansion
    /// matrices to obtain full-length indicators of which variables carry a
    /// finite bound, then returns `true` only when some component whose
    /// magnitude exceeds `diverging_iterates_tol` is heading toward a side
    /// with no finite bound.
    ///
    /// When every large component is pinned by a finite bound — in
    /// particular when all variables are boxed, so the feasible region is a
    /// bounded box and unboundedness is structurally impossible — this
    /// returns `false`, and the caller reports the best iterate via the
    /// normal convergence / restoration path instead of a spurious
    /// `Unbounded`.
    /// #248: consecutive growing, over-threshold iterations required before
    /// a structurally-free divergence is reported as `DivergingIterates`.
    /// `jit1`'s transient excursion lasts ~2 growing steps and then
    /// recedes, so a small persistence requirement clears it without
    /// materially delaying a genuine ray.
    const DIVERGENCE_PERSIST_ITERS: u32 = 4;
    /// #248: an iterate counts as "still growing" toward divergence when it
    /// grows at least this factor over the previous over-threshold iterate.
    /// A recession ray in an interior-point method grows geometrically; an
    /// iterate settling onto a finite optimum above the threshold does not.
    const DIVERGENCE_GROWTH_FACTOR: Number = 2.0;
    /// #252: the objective descent must *keep up* — each step's drop must be
    /// at least this fraction of the previous step's drop for the iterate to
    /// count toward the divergence streak. A recession ray descends `f` to
    /// `−∞` with per-step drops that grow (ratio ≥ 1) as `|x|` grows
    /// geometrically; an excursion converging to a finite optimum decelerates
    /// (ratio → 0). The slack below 1 tolerates ordinary interior-point noise
    /// on a genuine ray without admitting a decelerating excursion — jit1's
    /// node subproblems shrink the drop by 3–15× per step, far past this bar.
    const DIVERGENCE_DESCENT_KEEPUP: Number = 0.9;
    /// #248: absolute runaway backstop. An iterate this large is reported
    /// unbounded regardless of persistence. It sits at or below the default
    /// `diverging_iterates_tol = 1e20`, so the default behaviour (fire the
    /// instant `|x|` crosses the threshold) is preserved, while a low
    /// user threshold no longer fires on the way to a finite optimum.
    const DIVERGENCE_ABS_RUNAWAY: Number = 1e18;
    /// Most consecutive iterations the primal divergence guard will defer
    /// to a watchdog sequence before checking the iterate anyway. Upstream's
    /// `watchdog_trial_iter_max` default is 3; one spare covers the
    /// iteration on which the watchdog is armed. See
    /// [`Self::watchdog_defer_streak`] for why the bound is not simply
    /// "until `in_watchdog` clears".
    const WATCHDOG_DEFER_MAX: u32 = 4;

    /// #285: magnitude floor for the checked recession-ray unboundedness path.
    /// Below this the (slightly more expensive) recession proof is not even
    /// attempted, so it is inert on every normal, well-scaled solve. Above it,
    /// unboundedness is only ever concluded through the full checked proof in
    /// [`Self::curr_is_recession_ray`] — a genuinely *feasible* iterate of this
    /// magnitude already witnesses an unbounded feasible region, and the proof
    /// additionally certifies the escape direction. Sits far below the
    /// `diverging_iterates_tol` (`1e20`) magnitude guard so a linearly-growing
    /// ray (which never reaches `1e20` within `max_iter`) is still caught.
    const RECESSION_MIN_NORM: Number = 1e10;
    /// #285: consecutive growing, proof-passing iterations required before the
    /// recession-ray path reports `DivergingIterates`. A bounded feasible
    /// region cannot supply a *growing* sequence of feasible over-floor
    /// iterates, so persistence is defense-in-depth against a lone numerical
    /// fluke rather than a soundness requirement.
    const RECESSION_PERSIST_ITERS: u32 = 4;
    /// #285: relative feasibility bar for the recession proof. The current
    /// iterate counts as feasible (hence a witness that the feasible region
    /// reaches its magnitude) when its unscaled max-norm primal infeasibility
    /// is at most this fraction of `|x|_∞`. The check is *relative* on purpose:
    /// evaluating `A_eq x − b` at `|x| ~ 1e17` carries floating-point roundoff
    /// that scales with `|x|`, while a genuinely infeasible excursion (e.g.
    /// mid-restoration) has a residual comparable to `|x|` itself.
    const RECESSION_FEAS_REL: Number = 1e-6;
    /// #285: relative bar for "the escape direction lies in `null(A_eq)`".
    /// `‖J_c x‖_∞ ≤ this · |x|_∞` certifies that moving along `d ≈ x` preserves
    /// the (linearized) equality constraints — `A_eq d ≈ 0`.
    const RECESSION_DIR_TOL: Number = 1e-6;
    /// #285: relative descent bar. The objective must strictly decrease along
    /// the escape direction with a real margin — `∇f·x ≤ −this · ‖∇f‖ ‖x‖` —
    /// so a variable drifting orthogonally to the objective (`∇f·x ≈ 0`) can
    /// never be mistaken for a recession ray driving `f → −∞`.
    const RECESSION_DESC_REL: Number = 1e-6;

    /// Update the divergence-persistence state for the current iterate and
    /// return whether `DivergingIterates` should be reported now (issues
    /// #248 / #252). `amax` is `max_i |x_i|`; `structural_free` is the result
    /// of [`Self::divergence_is_true_unboundedness`] (already gated on
    /// `amax > diverging_iterates_tol`); `f` is the (minimized, internally
    /// scaled) objective at the current iterate, supplied only while
    /// `structural_free` holds.
    ///
    /// A large `|x|` is reported as unbounded only when it is heading to an
    /// unbounded side (`structural_free`) *and* the divergence looks like a
    /// genuine recession ray: the iterate keeps *growing* while the objective
    /// keeps *descending toward `−∞` without decelerating* — the per-step drop
    /// holds up as `|x|` grows geometrically — for
    /// [`Self::DIVERGENCE_PERSIST_ITERS`] consecutive iterations (or it has
    /// blown past the absolute runaway backstop). Two failure modes are thereby
    /// left to the normal convergence machinery instead of being mislabelled
    /// `UNBOUNDED`:
    ///
    /// * #248 — a transient ill-scaling excursion that peaks in `|x|` and
    ///   recedes never sustains the growth streak.
    /// * #252 — an excursion that *keeps* growing in `|x|` toward an unbounded
    ///   box side (a jit1 B&B node subproblem with `ub = +∞`), lowering `f` as
    ///   it goes, but with a per-step objective drop that *decelerates* toward
    ///   zero: it is settling onto a finite optimum, not riding a recession
    ///   ray. The descent must keep up (not merely exist), so this no longer
    ///   accumulates the streak.
    fn update_divergence_verdict(
        &mut self,
        amax: Option<Number>,
        structural_free: bool,
        f: Option<Number>,
    ) -> bool {
        let over = matches!(amax, Some(a) if a > self.diverging_iterates_tol) && structural_free;
        if !over {
            self.divergence_streak = 0;
            self.divergence_prev_amax = 0.0;
            self.divergence_prev_f = Number::INFINITY;
            self.divergence_prev_decrease = Number::NAN;
            return false;
        }
        let a = amax.expect("over implies amax is Some");
        // A recession ray in an interior-point method grows the iterate
        // geometrically *and* drives the objective down without bound, with a
        // per-step drop that keeps up as `|x|` grows. A finite-optimum
        // excursion may grow `|x|` and even lower `f` for a few steps, but its
        // per-step objective drop decelerates toward zero as it settles onto
        // the finite floor. Require all three — growth, descent, and
        // non-decelerating descent — before a step counts toward the streak.
        let growing = a >= self.divergence_prev_amax * Self::DIVERGENCE_GROWTH_FACTOR;
        // `f` is `None` only when `structural_free` is false, already handled
        // by the `!over` branch; treat a missing value as non-descending so a
        // run can never accumulate without objective evidence.
        let fv = f.unwrap_or(Number::INFINITY);
        let decrease = self.divergence_prev_f - fv;
        let descending = decrease > 0.0;
        // Non-decelerating: the drop must be at least a fixed fraction of the
        // previous step's drop. Bootstrapped `true` until a first finite
        // decrease has been recorded (`divergence_prev_decrease` non-finite),
        // so the run's opening steps are admitted on growth + descent alone.
        let keeping_up = !self.divergence_prev_decrease.is_finite()
            || decrease >= self.divergence_prev_decrease * Self::DIVERGENCE_DESCENT_KEEPUP;
        if growing && descending && keeping_up {
            self.divergence_streak += 1;
        } else {
            // Over the threshold on an unbounded side, but the divergence is
            // not sustaining a recession ray's growth-and-accelerating-descent
            // profile — the hallmark of a scaling excursion toward a finite
            // optimum. Drop the streak; a genuine ray re-accumulates it on its
            // next qualifying step (or trips the absolute runaway backstop).
            self.divergence_streak = 0;
        }
        self.divergence_prev_amax = a;
        self.divergence_prev_f = fv;
        // Record the baseline for the next step's keep-up comparison only from
        // a finite, real decrease; skip the `+∞` opening step and reset the
        // baseline whenever the objective stops descending.
        self.divergence_prev_decrease = if decrease.is_finite() && descending {
            decrease
        } else {
            Number::NAN
        };
        a >= Self::DIVERGENCE_ABS_RUNAWAY
            || self.divergence_streak >= Self::DIVERGENCE_PERSIST_ITERS
    }

    fn divergence_is_true_unboundedness(&self, x: &dyn Vector) -> bool {
        self.free_to_escape_over(x, self.diverging_iterates_tol)
    }

    /// Shared core of the free-variable structural check: returns `true` when
    /// some component of `x` with magnitude exceeding `thresh` is heading
    /// toward a side (positive → upper, negative → lower) that carries *no*
    /// finite bound, so it is free to escape to infinity. Parameterized on the
    /// magnitude threshold so both the `diverging_iterates_tol` (`1e20`)
    /// divergence guard and the lower `RECESSION_MIN_NORM` recession-ray path
    /// (#285) share one implementation.
    fn free_to_escape_over(&self, x: &dyn Vector, thresh: Number) -> bool {
        use pounce_linalg::DenseVector;

        let cq = self.cq.borrow();
        let nlp = cq.nlp().borrow();

        // Full-length 0/1 indicators of finite lower / upper bounds,
        // built by scattering ones through the bound expansion matrices.
        let mut ones_l = nlp.x_l().make_new();
        ones_l.set(1.0);
        let mut has_lb = x.make_new();
        nlp.px_l().mult_vector(1.0, &*ones_l, 0.0, &mut *has_lb);

        let mut ones_u = nlp.x_u().make_new();
        ones_u.set(1.0);
        let mut has_ub = x.make_new();
        nlp.px_u().mult_vector(1.0, &*ones_u, 0.0, &mut *has_ub);

        let downcast = |v: &dyn Vector| -> Option<Vec<Number>> {
            v.as_any()
                .downcast_ref::<DenseVector>()
                .map(|d| d.expanded_values())
        };

        // POUNCE is dense-only; if a backing is unexpectedly non-dense we
        // cannot prove the divergence is spurious, so fall back to the
        // original (magnitude-only) verdict to avoid changing behaviour.
        let (Some(xv), Some(lb), Some(ub)) = (downcast(x), downcast(&*has_lb), downcast(&*has_ub))
        else {
            return true;
        };

        for i in 0..xv.len() {
            if xv[i].abs() > thresh {
                let free_to_diverge = if xv[i] > 0.0 {
                    ub[i] == 0.0
                } else {
                    lb[i] == 0.0
                };
                if free_to_diverge {
                    return true;
                }
            }
        }
        false
    }

    /// #285: checked recession-ray unboundedness proof at the current iterate.
    ///
    /// Returns `true` only when the current iterate `x` (with `|x|_∞ = amax`,
    /// already known `> RECESSION_MIN_NORM` by the caller) *proves* the
    /// problem is unbounded below via a genuine recession ray — the same
    /// standard the LP/symmetric path holds itself to, not a magnitude
    /// heuristic. All of the following must hold:
    ///
    /// 1. **Feasible witness.** The iterate's unscaled primal infeasibility is
    ///    at most `RECESSION_FEAS_REL · amax`. A genuinely feasible iterate of
    ///    norm `≥ 1e10` witnesses that the feasible region reaches that far —
    ///    a *bounded* region cannot contain it. (Relative bar: the residual of
    ///    `A_eq x − b` carries roundoff that scales with `|x|`.)
    /// 2. **Free to escape.** Some over-floor component heads toward a side
    ///    with no finite variable bound ([`Self::free_to_escape_over`] at
    ///    `RECESSION_MIN_NORM`).
    /// 3. **Direction in `null(A_eq)`.** `‖J_c x‖_∞ ≤ RECESSION_DIR_TOL · amax`
    ///    — moving along `d ≈ x` preserves the equality constraints.
    /// 4. **Inequalities not blocking.** No finitely-bounded inequality row is
    ///    driven toward its bound along `d ≈ x`
    ///    ([`Self::recession_blocked_by_inequality`]).
    /// 5. **Objective descending.** `∇f·x ≤ −RECESSION_DESC_REL · ‖∇f‖ ‖x‖` —
    ///    the objective strictly decreases along the escape direction with a
    ///    real (non-orthogonal) margin, so `f → −∞` along the ray.
    ///
    /// On a *bounded* problem at least one of (1)/(2)/(3)/(4)/(5) fails, so
    /// this can never manufacture a spurious `DivergingIterates`.
    fn curr_is_recession_ray(&self, x: &dyn Vector, amax: Number) -> bool {
        // (1) Feasible witness (relative bar).
        let primal_inf = self.cq.borrow().curr_unscaled_primal_infeasibility_max();
        if !(primal_inf.is_finite() && primal_inf <= Self::RECESSION_FEAS_REL * amax) {
            return false;
        }
        // (2) Some over-floor component free to escape to infinity.
        if !self.free_to_escape_over(x, Self::RECESSION_MIN_NORM) {
            return false;
        }
        // (3) Escape direction lies in the equality null space. A non-finite
        // (NaN) residual is treated as failing, so the direction is only
        // accepted on a genuinely small, finite `‖J_c x‖∞`.
        let jc_x_amax = self.cq.borrow().curr_jac_c_times_vec(x).amax();
        if !jc_x_amax.is_finite() || jc_x_amax > Self::RECESSION_DIR_TOL * amax {
            return false;
        }
        // (4) No finitely-bounded inequality blocks the direction.
        if self.recession_blocked_by_inequality(x, amax) {
            return false;
        }
        // (5) Objective strictly descending along the escape direction.
        let (dot, gnorm, xnorm) = {
            let cq = self.cq.borrow();
            let g = cq.curr_grad_f();
            (g.dot(x), g.nrm2(), x.nrm2())
        };
        if !(dot < 0.0 && dot <= -Self::RECESSION_DESC_REL * gnorm * xnorm) {
            return false;
        }
        true
    }

    /// #285: does any *finitely-bounded* inequality constraint block motion
    /// along the escape direction `d ≈ x`? For each inequality row the
    /// constraint value `d(x)` changes at rate `(J_d x)_j` per unit of the
    /// direction; if that row has a finite upper bound and the rate is
    /// positive (or a finite lower bound and the rate is negative) beyond a
    /// relative tolerance, moving out along `d` would eventually violate it,
    /// so it is not a feasible recession direction. Bounds are detected via
    /// the `Pd_L / Pd_U` expansion matrices exactly as the variable-bound
    /// check uses `Px_L / Px_U`.
    fn recession_blocked_by_inequality(&self, x: &dyn Vector, amax: Number) -> bool {
        use pounce_linalg::DenseVector;

        let cq = self.cq.borrow();
        // Rate of change of each inequality value along d ≈ x (length m_ineq).
        // Compute first so the internal `nlp.borrow_mut()` is released before
        // the immutable borrow below.
        let jd_x = cq.curr_jac_d_times_vec(x);
        let (has_dlb, has_dub) = {
            let nlp = cq.nlp().borrow();
            let mut ones_dl = nlp.d_l().make_new();
            ones_dl.set(1.0);
            let mut has_dlb = jd_x.make_new();
            nlp.pd_l().mult_vector(1.0, &*ones_dl, 0.0, &mut *has_dlb);

            let mut ones_du = nlp.d_u().make_new();
            ones_du.set(1.0);
            let mut has_dub = jd_x.make_new();
            nlp.pd_u().mult_vector(1.0, &*ones_du, 0.0, &mut *has_dub);
            // Order matters: bind `(has_dlb, has_dub)` in that exact order so
            // the finite-lower / finite-upper indicators are not transposed.
            // #314: this pair was returned swapped, inverting the bound
            // semantics below — a ray *increasing* a lower-bounded row (moving
            // deeper into the feasible set, slack growing) was wrongly treated
            // as blocked, so a genuine inequality-slack recession ray was never
            // proven unbounded.
            (has_dlb, has_dub)
        };

        let downcast = |v: &dyn Vector| -> Option<Vec<Number>> {
            v.as_any()
                .downcast_ref::<DenseVector>()
                .map(|d| d.expanded_values())
        };
        // Dense-only fallback: if we cannot inspect the rows, conservatively
        // treat the direction as blocked (no spurious unbounded verdict).
        let (Some(jd), Some(dlb), Some(dub)) =
            (downcast(&*jd_x), downcast(&*has_dlb), downcast(&*has_dub))
        else {
            return true;
        };
        let tol = Self::RECESSION_DIR_TOL * amax;
        for j in 0..jd.len() {
            // Increasing a row that has a finite upper bound, or decreasing a
            // row that has a finite lower bound, would leave the feasible set.
            if (jd[j] > tol && dub[j] != 0.0) || (jd[j] < -tol && dlb[j] != 0.0) {
                return true;
            }
        }
        false
    }

    /// #285: update the recession-ray persistence state and return whether
    /// `DivergingIterates` should be reported now. `amax` is `|x|_∞`;
    /// `is_ray` is the result of [`Self::curr_is_recession_ray`]. The verdict
    /// fires once the checked proof has held for
    /// [`Self::RECESSION_PERSIST_ITERS`] consecutive *growing* iterations — a
    /// bounded region cannot supply a growing sequence of feasible over-floor
    /// iterates, so this is impossible to satisfy on a bounded problem.
    fn update_recession_verdict(&mut self, amax: Number, is_ray: bool) -> bool {
        if !is_ray {
            self.recession_streak = 0;
            self.recession_prev_amax = 0.0;
            return false;
        }
        if amax > self.recession_prev_amax {
            self.recession_streak += 1;
        } else {
            // Proof holds but the iterate is not growing (a stalled or rejected
            // step). Restart the run at the current witness rather than firing
            // on a plateau; a genuine ray resumes growing next step.
            self.recession_streak = 1;
        }
        self.recession_prev_amax = amax;
        self.recession_streak >= Self::RECESSION_PERSIST_ITERS
    }

    /// Honour a certificate the masked-scale veto refused, when the run that
    /// was allowed to continue did not end in one of its own (gh #200).
    ///
    /// The veto's bargain is "never worse off": it refuses a point that had
    /// *already passed the strict test*, betting that continuing reaches a
    /// better one. This is the losing side of that bet — so hand back exactly
    /// what would have been returned without the veto, point and status both.
    ///
    /// Two details make that guarantee real rather than approximate:
    ///
    /// - It runs on **every** non-success exit, applied once where the driver
    ///   loop's result is finalized. Wiring individual termination sites was
    ///   tried and is not safe: there are sixteen, and the ones easiest to
    ///   overlook are the ones most likely to fire here — the veto's extra
    ///   iterations are exactly what pushes a run past `max_cpu_time`.
    /// - It restores the **refused iterate itself** (`vetoed`), not the last
    ///   acceptable snapshot. `store_acceptable_point` overwrites
    ///   unconditionally, so after the veto the stored point drifts to whatever
    ///   the continued run last touched — which may be worse than the point
    ///   that was refused.
    ///
    /// "Better" is **status-dominant lexicographic**: the reported status first,
    /// and the objective only to break a tie *within equal status*. Both halves
    /// matter and the order between them is not cosmetic — see the `Success`
    /// branch, where reading it as a plain objective comparison costs a status.
    fn honour_refused_certificate(&mut self, result: SolverReturn) -> SolverReturn {
        if matches!(result, SolverReturn::Success) {
            // The continued run produced a certificate of its own — but not
            // necessarily a better *outcome*.
            //
            // This is what makes "never worse" hold even when the bet loses in a
            // way that still converges: on a non-convex problem the extra travel
            // can reach a different, worse stationary point, and the budget cap
            // (`VETO_MAX_EXTRA_ITERS`) can also hand back a late-but-converged
            // one. Neither may silently replace a better answer the solver
            // already had in hand.
            //
            // The comparison is NOT objective-only. That was the original bug
            // here: both points passed `passes_component_tols`, which looked
            // like a licence to treat them as equally valid certificates and
            // just take the lower objective. They are not equally valid when
            // `kkt_fidelity_tol` is set — `apply_kkt_fidelity_gate` re-grades a
            // `Success` on the unscaled KKT error afterwards, on a strictly
            // finer criterion than the convergence test. Taking a 3-ulp
            // objective win at a point whose unscaled error is 5x worse traded
            // `Solve_Succeeded` for `Solved_To_Acceptable_Level`: a status
            // regression against baseline, which is the strongest form of the
            // guarantee breaking. So rank by the status each point will actually
            // be *reported* under, and only then by objective.
            let Some((refused, refused_status)) = self.baseline_outcome() else {
                return result;
            };
            self.assert_comparable_scale(&refused);
            let (curr_f, curr_kkt) = self.curr_obj_and_unscaled_kkt();
            // Rank each candidate by the status it will actually be *reported*
            // under, which for a `Success` means after the fidelity gate has had
            // its say.
            let continued_success = self.survives_fidelity_gate(curr_kkt);
            let refused_success = matches!(refused_status, SolverReturn::Success)
                && self.survives_fidelity_gate(refused.unscaled_kkt);
            let keep_refused = match (continued_success, refused_success) {
                // Equal reported status: the objective breaks the tie, which is
                // legitimate because both points are feasible to tolerance.
                //
                // Negated `<=`, not `>`: they differ at NaN, and the difference
                // matters. A `Converged` exit at an iterate whose objective is
                // NaN but whose residuals are finite and tiny is reachable (the
                // convergence test never inspects `f`), and `NaN > x` is false,
                // which would keep the NaN point over a finite refused one.
                // Phrased as a negated `<=`, an incomparable objective fails to
                // justify keeping the continued point and the refused one wins.
                (true, true) | (false, false) => !(curr_f <= refused.obj),
                // The refused point keeps a status the continued one loses.
                (false, true) => true,
                (true, false) => false,
            };
            if !keep_refused {
                return result;
            }
            self.restore_snapshot(&refused);
            // The restored point's own status, which is what the baseline
            // reported for it. For a strict refusal that is `Success` even when
            // it fails the fidelity gate — the gate re-grades the restored point
            // downstream, exactly as it would have re-graded the baseline's. For
            // an acceptable-level refusal it is `StopAtAcceptablePoint`, since
            // claiming `Success` for a point that only ever qualified at the
            // acceptable level would over-report.
            return refused_status;
        }
        // The continued run did not certify — but its final point can still be a
        // *better* would-be certificate than the one the baseline stopped at, and
        // restoring the chronologically-first refusal unconditionally throws it
        // away (gh #327). The masking veto keeps refusing at the true optimum too
        // (its unscaled error stays above `acceptable_tol` under an extreme
        // objective scale), so a run that actually reaches the optimum never gets
        // to certify there and instead exits non-`Success` — typically on a tiny
        // step once it settles. Rolling straight back to the first refusal then
        // hands back the point the baseline stopped at, which can be far worse:
        // on `min 1/x` over `[1e-12, 10]` the solve reaches x≈10 (f≈0.1) but was
        // rolled back to the first refusal at x≈2.84 (f≈0.35) and reported
        // success there.
        //
        // The extra candidate is admitted *narrowly*, and the gate is
        // load-bearing: the continued point may displace the refused snapshot
        // only if it itself passes the strict per-component tolerances — i.e. it
        // is a would-be strict certificate the veto refused solely because of
        // masking. That is precisely what tells the settled true optimum apart
        // from a merely lower objective reached on an unbounded ray (e.g.
        // `A(x−a)⁴ − K·√(1+y²)`, unbounded below in y): the diverging iterate
        // never passes the strict test, so it can never win here, and those runs
        // stay bit-for-bit as before. When the gate does open, keep whichever
        // point ranks better under the same feasibility-aware key the dual-guard
        // fallback uses, and report the baseline's restored status either way —
        // never worse than baseline on status, never worse (often better) on the
        // point.
        let Some((refused, restored_status)) = self.baseline_outcome() else {
            return result;
        };
        self.assert_comparable_scale(&refused);
        let curr_nlp_err = self.cq.borrow().curr_nlp_error();
        let curr_passes_strict =
            self.bundle
                .conv_check
                .current_passes_strict(curr_nlp_err, &self.data, &self.cq);
        let curr_f = self.cq.borrow().curr_f();
        let curr_viol = self.cq.borrow().curr_unscaled_primal_infeasibility_max();
        // The second admissible candidate: a continued run that ends *at the
        // acceptable level itself* (gh #533). The `curr_passes_strict` gate
        // exists to tell a settled optimum from a diverging ray, and on this
        // exit the exit itself already answers that — `StopAtAcceptablePoint` is
        // only reachable at a point that passed the acceptable per-component
        // tolerances, either by qualifying here or by being the stashed
        // acceptable iterate a rollback restored. A diverging iterate cannot
        // produce it.
        //
        // This matters because the gh #533 progress refusal is frequently paid
        // off by a *better acceptable point* rather than by a strict
        // certificate: the streak refuses while the solve is still descending,
        // the solve descends, and then settles somewhere better but still short
        // of `tol`. Without this the refused point is restored and the entire
        // continuation is discarded — never worse than baseline, but never
        // better either, which for that whole class is pure cost.
        //
        // Gated on the *restored* status also being `StopAtAcceptablePoint`, so
        // a strict refusal's `Success` is never reported at a point that only
        // ever qualified at the acceptable level.
        let continued_is_acceptable_exit = matches!(result, SolverReturn::StopAtAcceptablePoint)
            && matches!(restored_status, SolverReturn::StopAtAcceptablePoint);
        // Keep the continued point in place only when it is an admissible
        // candidate that also ranks strictly better; otherwise restore the
        // refused snapshot exactly as before. `ranks_better` treats a non-finite
        // continued objective as worst, so a NaN-objective continuation never
        // displaces a finite refused point (the NaN-loses convention the
        // `Success` branch relies on).
        let keep_continued = (curr_passes_strict || continued_is_acceptable_exit)
            && self.ranks_better(curr_f, curr_viol, refused.obj, refused.constr_viol);
        if !keep_continued {
            self.restore_snapshot(&refused);
        }
        if self.cq.borrow().curr_f().is_finite() {
            restored_status
        } else {
            result
        }
    }

    /// What the baseline — the same solve with the veto disabled — would have
    /// returned, as (point, status), or `None` if nothing was ever refused.
    ///
    /// The **chronologically first** refusal, not the strictest one. Both arms
    /// follow the same trajectory until the first refusal, so that iterate is
    /// where the baseline stopped and what it reported. A refusal recorded later
    /// sits on the continued trajectory, which the baseline never walked — its
    /// point was never on offer, and restoring it would neither reproduce the
    /// baseline nor be comparable to it.
    ///
    /// Both kinds do occur, and in either order: an acceptable-level refusal
    /// needs `acceptable_iter` consecutive qualifying iterates, so a strict
    /// refusal can precede it, while a run that first drifts through the
    /// acceptable band can refuse there and only later pass the strict test.
    /// Preferring `Success` unconditionally was wrong for exactly the second
    /// case — it compared against a strict point from iteration 50-odd when the
    /// baseline had already stopped and reported acceptable at iteration 43.
    fn baseline_outcome(&self) -> Option<(VetoSnapshot, SolverReturn)> {
        // A refusal that was seen but not captured makes the baseline
        // unidentifiable, so decline rather than guess. Without this, a failed
        // strict capture alongside a successful acceptable one would present the
        // acceptable snapshot as the baseline outcome — but that snapshot sits
        // on the continued trajectory, so this would silently reintroduce the
        // very misidentification the chronological rule exists to prevent.
        // Declining loses the restore; misidentifying reports a wrong point
        // under a confident status.
        //
        // Unreachable today (`data.curr` is always `Some` inside `iterate()`, so
        // `snapshot_current` cannot fail), but the latches make the state
        // representable, and it must not be handled by accident.
        if (self.vetoed_seen && self.vetoed.is_none())
            || (self.vetoed_acceptable_seen && self.vetoed_acceptable.is_none())
        {
            return None;
        }
        match (&self.vetoed, &self.vetoed_acceptable) {
            // Ties go to the strict refusal, and the tie is reachable: both can
            // arm in the same call when the acceptable streak crosses on the
            // same iterate a strict certificate is refused. Strict is correct
            // there because of the baseline's own branch order — the `Converged`
            // gate (`opt_error.rs`, in `check_convergence_with_state`) precedes
            // `note_acceptable`, so the baseline returned `Converged` at that
            // iterate. Reordering those two branches would invert this.
            (Some(strict), Some(acc)) => Some(if strict.iter <= acc.iter {
                (strict.clone(), SolverReturn::Success)
            } else {
                (acc.clone(), SolverReturn::StopAtAcceptablePoint)
            }),
            (Some(strict), None) => Some((strict.clone(), SolverReturn::Success)),
            (None, Some(acc)) => Some((acc.clone(), SolverReturn::StopAtAcceptablePoint)),
            (None, None) => None,
        }
    }

    /// Capture the current iterate as a veto snapshot, or `None` if there is no
    /// current iterate to capture.
    ///
    /// All-or-nothing by construction — see [`VetoSnapshot`].
    fn snapshot_current(&self, iter: Index) -> Option<VetoSnapshot> {
        let iterate = self.data.borrow().curr.as_ref().cloned()?;
        let cq = self.cq.borrow();
        Some(VetoSnapshot {
            iterate,
            iter,
            obj: cq.curr_f(),
            mu: self.data.borrow().curr_mu,
            unscaled_kkt: cq.curr_unscaled_nlp_error(),
            constr_viol: cq.curr_unscaled_primal_infeasibility_max(),
            obj_scale: cq.obj_scaling_factor(),
        })
    }

    /// Current objective and max-norm unscaled KKT error, read together so the
    /// pair cannot describe different iterates.
    fn curr_obj_and_unscaled_kkt(&self) -> (Number, Number) {
        let cq = self.cq.borrow();
        (cq.curr_f(), cq.curr_unscaled_nlp_error())
    }

    /// Guard the precondition of every scaled-objective comparison in
    /// [`Self::honour_refused_certificate`]: the factor must not have moved
    /// between the refusal and now, or the two numbers are not comparable.
    fn assert_comparable_scale(&self, snap: &VetoSnapshot) {
        debug_assert_eq!(
            snap.obj_scale,
            self.cq.borrow().obj_scaling_factor(),
            "objective scaling factor moved during the solve; the refused and \
             continued objectives are scaled differently and cannot be compared \
             (gh #200)"
        );
    }

    /// Whether a point with this unscaled KKT error would keep `Solve_Succeeded`
    /// through [`IpoptApplication::apply_kkt_fidelity_gate`].
    ///
    /// Mirrors that gate rather than approximating it: same quantity
    /// (`final_unscaled_kkt_error`), same strict comparison, same "non-positive
    /// tolerance disables". With the default `kkt_fidelity_tol = 0` this is
    /// always `true`, so every caller collapses to the plain objective
    /// comparison and the mechanism's behaviour is unchanged.
    fn survives_fidelity_gate(&self, unscaled_kkt: Number) -> bool {
        // Phrased as the negation of the gate's own `> tol` test rather than as
        // `<= tol`, because the two disagree at NaN and the gate is the
        // authority: it demotes only on `> tol`, so a NaN error keeps `Success`
        // there and must keep it here. Written as `<= tol` this mirror said the
        // opposite, which would rank a NaN-error continued point below a refused
        // one. Benign in that direction — it restores the baseline point — but a
        // mirror that disagrees with the thing it mirrors is a latent trap.
        !(self.kkt_fidelity_tol > 0.0) || !(unscaled_kkt > self.kkt_fidelity_tol)
    }

    /// Record the current iterate as the best acceptable-quality point seen so
    /// far in this solve (pounce#250 follow-up).
    ///
    /// Recording runs on **every** acceptable iterate, not only after the
    /// dual-divergence guard has fired. Gating it on the guard was the first
    /// attempt and left a hole: the guard fires and returns to the driver
    /// *before* this site is reached on that iteration (see the guard block in
    /// [`Self::iterate`]), so nothing at or before the diversion was ever
    /// captured. A diversion that wrecks the solve immediately — reaching no
    /// acceptable point afterwards — therefore had nothing to hand back, which
    /// is precisely the case the fallback exists for. `autocorr_bern55-06` hid
    /// this, because its better point happens to arrive at iteration 86, well
    /// after the guard fires at 23.
    ///
    /// Recording always is still behaviour-neutral, because the record is only
    /// ever *read* under `dual_guard_fired` — see
    /// [`Self::honour_best_acceptable_after_dual_guard`]. A solve the guard
    /// never touches computes a comparison per acceptable iterate and nothing
    /// else.
    ///
    /// The cost is one `f64` comparison per acceptable iterate; the iterate is
    /// cloned only on an actual improvement, so this does not double the
    /// per-iteration clone `store_acceptable_point` already pays.
    ///
    /// "Best" is a feasibility-aware ranking, **not** the lowest objective:
    /// candidates are ordered by [`Self::ranks_better`]'s `(feasible_enough,
    /// objective)` key, so objective only decides among points already inside a
    /// capped feasibility band. Being *bounded* by `acceptable_constr_viol_tol`
    /// is not the same as *not trading* feasibility within it — that band is a
    /// user option and can be widened to `1e1` or beyond. A pure-objective argmax
    /// over it has no lower bound on the feasibility it will spend, and one
    /// option-value away it returns a point `pounce verify` rejects under a
    /// `Solved_To_Acceptable_Level` status (gh #267). Whether an early
    /// low-objective iterate is even a candidate is the user's
    /// `acceptable_constr_viol_tol`; the capped feasibility key is what keeps a
    /// grossly-infeasible one from winning even when the band admits it.
    fn record_best_acceptable(&mut self, curr_f: Number) {
        if !curr_f.is_finite() {
            return;
        }
        // Same quantity the acceptable-point gate keys on, so the recorded
        // feasibility matches the band the candidate just passed.
        let curr_viol = self.cq.borrow().curr_unscaled_primal_infeasibility_max();
        // Reject before cloning: only a strictly better candidate — by the
        // feasibility-aware key, not objective alone — is worth a snapshot.
        if let Some(best) = self.best_acceptable.as_ref() {
            let (b_obj, b_viol) = (best.obj, best.constr_viol);
            if !self.ranks_better(curr_f, curr_viol, b_obj, b_viol) {
                return;
            }
        }
        let iter = self.data.borrow().iter_count;
        let Some(snap) = self.snapshot_current(iter) else {
            return;
        };
        // Scaled objectives are only comparable under an unchanged factor; if it
        // ever moved, keep the earlier point rather than compare noise.
        if let Some(best) = self.best_acceptable.as_ref() {
            if snap.obj_scale != best.obj_scale {
                return;
            }
        }
        self.best_acceptable = Some(snap);
    }

    /// Cap on the feasibility band [`Self::ranks_better`] admits, matching the
    /// upstream default `acceptable_constr_viol_tol`. The fallback treats a point
    /// as "feasible enough to win on objective" only within this band, *however
    /// loose the user made `acceptable_constr_viol_tol`*, so widening that option
    /// cannot let the fallback trade feasibility for objective (gh #267).
    const FEASIBLE_ENOUGH_CAP: Number = 1e-2;

    /// Whether candidate `(a_obj, a_viol)` ranks strictly better than
    /// `(b_obj, b_viol)` for the best-acceptable fallback (gh #267, gh #280).
    ///
    /// The key is `(band_clamped_viol, objective)` compared lexicographically,
    /// where each violation is clamped *up* to
    /// `band = min(acceptable_constr_viol_tol, FEASIBLE_ENOUGH_CAP)` before it is
    /// compared. Inside the band every point clamps to `band`, so they tie on
    /// feasibility and objective decides — objective still rules *only among
    /// points already feasible-enough*. Outside the band the actual violation
    /// decides, so the less-infeasible point always wins and a
    /// strictly-more-infeasible point can never rank better (gh #280 — the
    /// earlier `feasible_enough` partition fell through to objective-only once
    /// both points were outside the band). The cap keeps the band no looser than
    /// the upstream default: `acceptable_constr_viol_tol` is user-widenable, and
    /// admitting a wide band into the *objective-decides* region would let a
    /// grossly-infeasible low-objective iterate win. Capping the band bounds that.
    ///
    /// At default (or tighter) tolerances this is behaviour-neutral: every
    /// recorded point already passed the `acceptable_constr_viol_tol` gate, so
    /// with that band at or below the cap every candidate clamps to `band` and
    /// objective alone decides, exactly as before. The feasibility ordering only
    /// bites once the user loosens `acceptable_constr_viol_tol` past its default
    /// and two candidates both sit outside the cap.
    ///
    /// A non-finite objective ranks worst and can never win — feasibility never
    /// rescues a `NaN`/`Inf` `f`. This mirrors the `NaN`-loses convention the
    /// gh #200 comparisons already rely on, and it keeps a `NaN`-objective
    /// returned point losing to a finite recorded one in
    /// [`Self::honour_best_acceptable_after_dual_guard`].
    ///
    /// The ranking itself lives in the pure [`ranks_better_within_band`] so its
    /// never-worse-off guarantee can be proven by deterministic unit tests rather
    /// than inferred from a host-dependent end-to-end objective comparison (see
    /// gh #267, which flagged an earlier CLI test for measuring the wrong,
    /// host-varying property). This method only resolves the admitted band.
    fn ranks_better(&self, a_obj: Number, a_viol: Number, b_obj: Number, b_viol: Number) -> bool {
        let band = self
            .bundle
            .conv_check
            .acceptable_constr_viol_tol_or_default()
            .min(Self::FEASIBLE_ENOUGH_CAP);
        ranks_better_within_band(a_obj, a_viol, b_obj, b_viol, band)
    }

    /// Make the dual-divergence guard's diversion non-destructive (pounce#250
    /// follow-up).
    ///
    /// The guard bets that routing to restoration beats grinding on, and nothing
    /// made losing that bet safe: it could return a materially worse point than
    /// the solve already had, under a status that does not admit it.
    ///
    /// WHAT THIS DOES AND DOES NOT GUARANTEE. It guarantees the diverted run
    /// never returns worse than the best acceptable-quality point **that same
    /// run visited**. It does *not* guarantee the diverted run is no worse than
    /// not diverting at all — that counterfactual solve never happened, and its
    /// points were never on offer to compare against. The distinction is not
    /// academic: on the Linux CI host `deb7` returns 97.56 with the guard off and
    /// 127.87 with it on at streak 15, and this fallback cannot close that gap,
    /// because 127.87 is the best acceptable point the diverted run ever reached.
    /// Bounding the diversion's damage is a weaker property than making the
    /// diversion harmless, and only the weaker one is available from inside a
    /// single solve. It is a large part of why the guard is off by default.
    ///
    /// The observed case is `autocorr_bern55-06`. The guard fires at iteration
    /// 23, the diverted run reaches the true optimum (-2304.0000278, matching
    /// Ipopt to 12 significant figures) and holds it from iteration 57 to 86 —
    /// but the dual residual sawtooths between 1e-8 and 2e-1 there, so it never
    /// strings together the `acceptable_iter` consecutive qualifying iterates
    /// that would stop the solve. It then enters restoration a second time,
    /// wanders into a worse basin, and terminates `StopAtAcceptablePoint` at
    /// -2263.46 — 1.8 % worse, with an overall NLP error of 1.0. The better
    /// point was *visited and passed the acceptable test*; it was simply
    /// overwritten, because `store_acceptable_point` keeps the latest rather
    /// than the best.
    ///
    /// So: on a non-`Success` exit, if the best acceptable-quality iterate seen
    /// anywhere in the solve beats the point being returned, hand that back
    /// instead. This is the same "never worse off" bargain the gh #200 veto
    /// makes, applied to the other bet in the algorithm.
    ///
    /// "Beats" is the feasibility-aware ranking in [`Self::ranks_better`], not a
    /// bare objective comparison: the recorded point wins only if it is
    /// feasible-enough while the returned point is not, or both are in the same
    /// feasibility class and it has a lower objective. Ranking by objective alone
    /// let a widened `acceptable_constr_viol_tol` band trade feasibility for
    /// objective here — restoring a lower-objective point that `pounce verify`
    /// rejects, under a success-mapped status (gh #267). The key prevents that:
    /// objective can only win among points already inside the capped acceptable
    /// feasibility band.
    ///
    /// Note "anywhere in the solve", not "since the guard fired":
    /// [`Self::record_best_acceptable`] runs unconditionally and explains why —
    /// points at or before the diversion have to be on offer, or a diversion that
    /// wrecks the solve immediately has nothing to hand back. Only this *read* is
    /// gated on `dual_guard_fired`.
    ///
    /// A strict `Success` is never overridden — that point carries a real
    /// certificate, and a lower objective at a merely-acceptable point must not
    /// displace it.
    ///
    /// Tuning the guard's firing threshold was tried first and rejected: no
    /// setting separates the models it helps from the ones it harms, and the
    /// effect turned out to differ by host anyway (see the option help in
    /// `upstream_options.rs`). Fixing the consequence is what remained available.
    fn honour_best_acceptable_after_dual_guard(&mut self, result: SolverReturn) -> SolverReturn {
        if !self.dual_guard_fired || matches!(result, SolverReturn::Success) {
            return result;
        }
        let Some(best) = self.best_acceptable.clone() else {
            return result;
        };
        let (curr_f, _) = self.curr_obj_and_unscaled_kkt();
        let curr_viol = self.cq.borrow().curr_unscaled_primal_infeasibility_max();
        let curr_scale = self.cq.borrow().obj_scaling_factor();
        // Only comparable under the same factor, sign included.
        if curr_scale != best.obj_scale {
            return result;
        }
        // Restore only when the recorded point ranks strictly better under the
        // feasibility-aware key — more feasible, or equally feasible at a lower
        // objective. `ranks_better` also handles the `NaN` case the previous
        // bare `!(curr_f <= best.obj)` did: a non-finite returned objective
        // ranks worst, so a finite recorded point wins and is restored.
        if self.ranks_better(best.obj, best.constr_viol, curr_f, curr_viol) {
            tracing::debug!(target: "pounce::algorithm",
                "[POUNCE] dual-divergence diversion ended worse than a point already \
                 in hand (obj {:.10e} viol {:.3e} -> obj {:.10e} viol {:.3e}, iter {}); \
                 restoring it (pounce#250, gh#267).",
                curr_f, curr_viol, best.obj, best.constr_viol, best.iter,
            );
            self.restore_snapshot(&best);
            // Swap the *point*, but never let the swap erase why the solve
            // stopped. A budget that was exhausted stays reported as exhausted:
            // a caller polling for "did I run out of time" must not be told
            // "solved to acceptable level" merely because a better point was
            // recoverable. Only the outcomes that carry no such fact of their
            // own are relabelled to describe what is now being returned.
            return match result {
                SolverReturn::MaxiterExceeded
                | SolverReturn::CpuTimeExceeded
                | SolverReturn::WallTimeExceeded
                | SolverReturn::UserRequestedStop => result,
                _ => SolverReturn::StopAtAcceptablePoint,
            };
        }
        result
    }

    /// Make a refused snapshot the current iterate again.
    fn restore_snapshot(&mut self, snap: &VetoSnapshot) {
        let mut d = self.data.borrow_mut();
        d.set_trial(snap.iterate.clone());
        d.accept_trial_point();
        // The restored point's own barrier parameter, not the continued run's —
        // see `VetoSnapshot::mu`.
        d.curr_mu = snap.mu;
    }

    /// Decide whether the acceptable-point restoration decline may be deferred
    /// this once (gh #534), and arm the bookkeeping that bounds the bet.
    ///
    /// Four conditions, all required:
    ///
    /// * the option leaves deferrals available at all
    ///   (`resto_decline_deferrals`, `0` = pre-#534 behaviour);
    /// * the budget is not already spent;
    /// * the NLP error has contracted on every one of the last
    ///   [`DECLINE_PROGRESS_SAMPLES`]` - 1` iterations
    ///   ([`Self::nlp_err_contracting`]) — the progress test the guard lacked;
    /// * the iteration budget has room for a continuation, and the entry point
    ///   can actually be captured. Without a floor there is nothing to fall
    ///   back to, and a bet with no floor is exactly what must not be placed.
    ///
    /// The deadline is clamped below `max_iter` so a lost bet can never turn a
    /// reportable `StopAtAcceptablePoint` into `Maximum_Iterations_Exceeded`:
    /// the continuation is always cut before the iteration budget runs out. A
    /// *time* budget is not clamped the same way — elapsed time is an external
    /// fact and the deadline cannot predict it — so a solve that expires inside
    /// the continuation window still reports the time limit, at the floor
    /// iterate rather than at whatever the continuation last touched.
    fn may_defer_acceptable_decline(&mut self) -> bool {
        if self.decline_deferrals_used >= self.resto_decline_deferrals {
            return false;
        }
        if !self.nlp_err_contracting() {
            return false;
        }
        let iter = self.data.borrow().iter_count;
        // No room to continue: the deadline below would fire on the very next
        // iteration, so the deferral would buy nothing and cost a restoration.
        if iter.saturating_add(1) >= self.max_iter {
            return false;
        }
        if self.decline_floor.is_none() {
            let Some(snap) = self.snapshot_current(iter) else {
                return false;
            };
            self.decline_floor = Some(snap);
        }
        self.decline_deferrals_used += 1;
        self.decline_deadline_iter = Some(
            iter.saturating_add(DECLINE_CONTINUATION_BUDGET)
                .min(self.max_iter.saturating_sub(1)),
        );
        true
    }

    /// The deferred continuation ran out of budget without a strict certificate
    /// (gh #534). Report the floor — the point the pre-#534 guard would have
    /// returned — unless the continuation is standing somewhere at least as
    /// good.
    fn terminate_at_decline_floor(&mut self) -> IterateOutcome {
        let Some(floor) = self.decline_floor.clone() else {
            // Unreachable in practice: the deadline is only ever set after a
            // floor is captured. Stopping at the current point is still the
            // right thing if it somehow is not — the point passed the
            // acceptable-level triplet when the deferral was taken.
            return IterateOutcome::Terminate(SolverReturn::StopAtAcceptablePoint);
        };
        tracing::debug!(target: "pounce::algorithm",
            "[POUNCE] deferred restoration decline expired at iter {} without a strict \
             certificate; falling back to the floor from iter {} (gh #534).",
            self.data.borrow().iter_count, floor.iter,
        );
        if !self.continuation_outranks(&floor) {
            self.restore_snapshot(&floor);
        }
        IterateOutcome::Terminate(SolverReturn::StopAtAcceptablePoint)
    }

    /// Whether the current iterate is a *better* answer than the gh #534 floor.
    ///
    /// Two gates, in order. The current point must itself pass the
    /// acceptable-level triplet — the floor is going to be reported under
    /// `Solved_To_Acceptable_Level`, and a continuation that wandered off is not
    /// entitled to that status however attractive its objective looks. Only then
    /// does [`Self::ranks_better`]'s feasibility-first key decide, and only under
    /// an unmoved objective scaling factor, since the two objectives are
    /// otherwise not comparable.
    fn continuation_outranks(&self, floor: &VetoSnapshot) -> bool {
        let (curr_f, _) = self.curr_obj_and_unscaled_kkt();
        if !curr_f.is_finite() {
            return false;
        }
        let nlp_err = self.cq.borrow().curr_nlp_error();
        if !self
            .bundle
            .conv_check
            .current_is_acceptable_with_state(nlp_err, &self.data, &self.cq)
        {
            return false;
        }
        let (curr_viol, curr_scale) = {
            let cq = self.cq.borrow();
            (
                cq.curr_unscaled_primal_infeasibility_max(),
                cq.obj_scaling_factor(),
            )
        };
        if curr_scale != floor.obj_scale {
            return false;
        }
        !self.ranks_better(floor.obj, floor.constr_viol, curr_f, curr_viol)
    }
    /// Try to leave a first-order-stationary point that is not a local minimum
    /// (gh #797). Returns `true` when the current iterate has been replaced and
    /// the solve should continue instead of terminating.
    ///
    /// # What this is for
    ///
    /// The convergence check is a *first-order* test, and on a nonconvex model
    /// that is strictly weaker than "local minimum". `nonconvex_qp.nl` is the
    /// reported case: `min x₀x₁ s.t. x₀ + x₁ = 2, 0 ≤ x ≤ 4` restricted to its
    /// feasible segment is the concave `f(x₀) = x₀(2 - x₀)`, *maximized* at
    /// `(1,1)` and minimized at the two endpoints. From the bound-pushed start
    /// `(0.01, 0.01)` the first Newton step lands exactly on `(1,1)`, every KKT
    /// residual there is zero, and the solve reports `Solve_Succeeded` at
    /// `obj = 1` — the constrained maximum.
    ///
    /// Inertia correction does not save this. It engages (the iteration log
    /// shows `lg(rg)` from the second iteration on) and cannot help: `δ_x I` is
    /// symmetric, the model and the iterate are symmetric under `x₀ ↔ x₁`, and
    /// a symmetric correction applied to a zero gradient gives a zero step
    /// however indefinite the reduced Hessian is. The regularization makes the
    /// *step* well-posed; nothing in the algorithm asks whether the point it
    /// has converged to is a minimum.
    ///
    /// # What it does
    ///
    /// [`PdFullSpaceSolver::negative_curvature_direction`] answers that
    /// question with the KKT factorization already in hand, and hands back a
    /// measured direction `d` with `J_c d_x = 0` and `dᵀ(W + Σ)d < 0`. Since
    /// the gradient is (near) zero the barrier objective along `±d` is
    /// `φ(α) ≈ φ(0) + ½α²dᵀ(W + Σ)d`, decreasing on *both* sides, so both signs
    /// are tried and the better trial wins. The step is capped by the ordinary
    /// fraction-to-the-boundary rule and by [`NEG_CURV_MAX_STEP_FACTOR`] times
    /// the iterate's own scale, then backtracked until it satisfies that
    /// second-order decrease model with an Armijo factor — the same shape as
    /// the line search, on the curvature term rather than the gradient term.
    /// A trial whose constraint violation exceeds what the convergence check
    /// itself calls feasible is refused outright.
    ///
    /// # Why it cannot make an answer worse
    ///
    /// The point being left is a *strict certificate* — the solve was about to
    /// report `Solve_Succeeded` at it — so it is snapshotted as a floor before
    /// the step, exactly as gh #534's deferred restoration decline does with
    /// the point its guard would have returned. The continuation gets
    /// [`NEG_CURV_CONTINUATION_BUDGET`] outer iterations; past that, and at
    /// every other exit of the driver loop, [`Self::honour_neg_curv_floor`]
    /// hands the floor back unless the continuation is standing somewhere that
    /// both outranks it and carries a certificate of its own. So the escape
    /// costs a bounded number of iterations and can only trade the stationary
    /// point for a strictly better one.
    ///
    /// That holds however many escapes are spent, not only at the default of
    /// one: a later escape displaces the floor only with a certificate that
    /// outranks the one already held (gh #805), so the floor is always at
    /// least as good as the point a `neg_curv_escapes = 0` build reports.
    fn try_neg_curv_escape(&mut self, iter_count: Index) -> bool {
        if self.neg_curv_escapes_used >= self.neg_curv_escapes {
            return false;
        }
        // No room to continue: the deadline below would fire on the very next
        // iteration, so the escape would buy nothing and cost a factorization.
        if iter_count.saturating_add(1) >= self.max_iter {
            return false;
        }
        if self.nlp.is_none() || self.search_dir.is_none() {
            return false;
        }
        // `data.w` still holds `W(curr_{N-1})` — step 3 of `iterate()` runs
        // *after* the convergence check — and curvature at the previous iterate
        // is not the question being asked. Re-evaluate it here rather than
        // running the Hessian updater a second time at this iterate: that would
        // hand the limited-memory updater a zero-length curvature pair to skip
        // and count against `limited_memory_max_skipping`, and it would leave
        // `data.w` describing a different iterate than it did before, which the
        // post-optimal sensitivity hook reads.
        //
        // With a quasi-Newton `B` there is nothing to re-evaluate and the stale
        // one is used. That is not a gap being papered over: BFGS maintains `B`
        // positive definite by construction, so under
        // `hessian_approximation=limited-memory` the probe's inertia test
        // passes at δ_x = 0 and the escape declines — correctly, since the only
        // curvature information the solve has says the point is a minimum.
        //
        // That argument is about BFGS's definiteness, NOT about "not exact",
        // and gating on `provides_exact_hessian` conflated the two. A
        // finite-difference `W` is not exact but does carry genuine negative
        // curvature, so judging the current iterate by the previous one's
        // matrix let a stationary maximum be reported as optimal where the
        // exact path escapes it (gh#823 review, finding 1, @srikanth-gm).
        // `hessian_at_current` asks the question the probe actually has:
        // can you give me `W` here? Quasi-Newton updaters still answer `None`
        // and still take the stale path, for the reason above.
        let w_at_curr = self.bundle.hess.hessian_at_current(&self.data, &self.cq);

        let probe = {
            let (Some(nlp), Some(sd)) = (self.nlp.as_ref(), self.search_dir.as_mut()) else {
                return false;
            };
            let mut pd = sd.pd_solver_mut();
            pd.negative_curvature_direction(&self.data, &self.cq, nlp, w_at_curr)
        };
        let Some(probe) = probe else {
            return false;
        };

        let Some(floor) = self.snapshot_current(iter_count) else {
            // Nothing to fall back to, and a bet with no floor is exactly what
            // must not be placed (the gh #534 rule, for the same reason).
            return false;
        };
        let curr = floor.iterate.clone();
        let (tau, curr_barr, curr_theta) = {
            let d = self.data.borrow();
            let cq = self.cq.borrow();
            (
                d.curr_tau,
                cq.curr_barrier_obj(),
                cq.curr_constraint_violation(),
            )
        };
        if !curr_barr.is_finite() {
            return false;
        }
        // A trial may raise the violation up to what this solve's own
        // convergence check calls feasible, and no further.
        let theta_cap = curr_theta.max(self.bundle.conv_check.constr_viol_tol_or_default());
        // The direction has unit infinity-norm, so this caps the escape at a
        // multiple of the iterate's own scale rather than at an absolute
        // distance, which would mean different things on differently scaled
        // models.
        let step_cap = NEG_CURV_MAX_STEP_FACTOR * (1.0 + curr.x.amax().max(curr.s.amax()));

        let mut accepted: Option<(crate::iterates_vector::IteratesVector, Number, Number)> = None;
        for sign in [1.0, -1.0] {
            let mut dir = probe.delta.deep_copy();
            dir.scal(sign);
            let dir = dir.freeze();
            let alpha_max = self.cq.borrow().aff_step_alpha_primal_max(&dir, tau);
            if !(alpha_max > 0.0) {
                continue;
            }
            let mut alpha = alpha_max.min(step_cap);
            for _ in 0..NEG_CURV_BACKTRACKS {
                if !(alpha > 0.0) || !alpha.is_finite() {
                    break;
                }
                let mut trial = curr.deep_copy();
                trial.x.axpy(alpha, &*dir.x);
                trial.s.axpy(alpha, &*dir.s);
                let trial = trial.freeze();
                self.data.borrow_mut().set_trial(trial.clone());
                let (barr, theta) = {
                    let cq = self.cq.borrow();
                    (cq.trial_barrier_obj(), cq.trial_constraint_violation())
                };
                self.data.borrow_mut().trial = None;
                // Second-order sufficient decrease. `probe.curvature` is
                // negative, so this asks the trial to realise at least
                // `NEG_CURV_ARMIJO` of the decrease the curvature model
                // predicts — the guard against a direction that is only
                // downhill in the quadratic model and uphill in the function.
                let predicted = 0.5 * alpha * alpha * probe.curvature;
                if barr.is_finite()
                    && theta.is_finite()
                    && theta <= theta_cap
                    && barr <= curr_barr + NEG_CURV_ARMIJO * predicted
                {
                    let better = match accepted.as_ref() {
                        Some((_, best, _)) => barr < *best,
                        None => true,
                    };
                    if better {
                        accepted = Some((trial, barr, alpha));
                    }
                    break;
                }
                alpha *= NEG_CURV_BACKTRACK_FACTOR;
            }
        }

        let Some((trial, barr, alpha)) = accepted else {
            return false;
        };

        tracing::debug!(target: "pounce::algorithm",
            "[POUNCE] iter {}: certified point has negative reduced curvature \
             (dᵀ(W+Σ)d = {:.3e}); escaping along it with α = {:.3e} \
             (barrier obj {:.10e} -> {:.10e}) and continuing (gh#797).",
            iter_count, probe.curvature, alpha, curr_barr, barr,
        );

        self.neg_curv_escapes_used += 1;
        // gh #805 — the floor is the *best* certificate the escapes have left,
        // never simply the most recent one. Replacing it unconditionally is
        // what broke the guarantee above at `neg_curv_escapes >= 2`: escape 1
        // floors at A, the continuation certifies a different indefinite point
        // B, escape 2 overwrites the floor with B, and if that bet is lost the
        // run reports B — which nothing has ever compared against A. With
        // `f(B) > f(A)` that is worse than `neg_curv_escapes = 0` returns, the
        // one outcome the mechanism promises cannot happen. The filter method's
        // barrier objective is not monotone across μ updates, so the escape's
        // own accounting does not exclude it.
        //
        // Ranked rather than merely kept (gh #805's suggested fix), because
        // keeping A unconditionally has the mirror-image flaw: where B *is* the
        // better certificate, a lost second bet would hand back A and make
        // `neg_curv_escapes = 2` worse than `= 1`, which returns B. Both points
        // are strict certificates — the escape only fires on
        // `ConvergenceStatus::Converged` — so the same status-dominant ranking
        // [`Self::honour_neg_curv_floor`] uses on the way out decides between
        // them, and the floor is monotone in the number of escapes spent.
        //
        // A provable no-op at the default: the branch is only reachable with a
        // floor already held, which takes a second escape, which takes
        // `neg_curv_escapes >= 2`.
        let replace_floor = match self.neg_curv_floor.as_ref() {
            None => true,
            Some(held) => self.current_outranks_neg_curv_floor(held, true),
        };
        if replace_floor {
            self.neg_curv_floor = Some(floor);
        }
        self.neg_curv_deadline_iter = Some(
            iter_count
                .saturating_add(NEG_CURV_CONTINUATION_BUDGET)
                .min(self.max_iter.saturating_sub(1)),
        );
        {
            let mut d = self.data.borrow_mut();
            d.set_trial(trial);
            d.accept_trial_point();
            // Iteration-log marker. `n` is not one of upstream's codes and is
            // unused elsewhere in POUNCE; it is what tells a reader of the
            // table that the jump between two iterates was an escape and not a
            // line-search step.
            d.append_info_string("n");
        }
        // The filter's entries were computed at, and around, a point the
        // algorithm has just left discontinuously — the same situation a
        // successful restoration leaves behind, and handled the same way.
        self.bundle.line_search.reset();
        self.bundle.line_search.reset_after_restoration();
        true
    }

    /// Rank the current iterate against a held negative-curvature floor
    /// (gh #797, gh #805).
    ///
    /// Rank by the status each point will actually be *reported* under, and
    /// only then by objective — the status-dominant order gh #200 arrived at
    /// the hard way. `apply_kkt_fidelity_gate` re-grades a `Success` on the
    /// unscaled KKT error after the driver loop returns, so with
    /// `kkt_fidelity_tol` set a lower objective at a coarser point is a status
    /// regression dressed up as a win.
    ///
    /// `current_certifies` is whether the current point comes with a
    /// certificate of its own: at the exit hook that is the driver's status,
    /// and at an escape site it is true by construction, since the escape only
    /// fires on `ConvergenceStatus::Converged`.
    ///
    /// Shared by the two sites that have to answer this question — the exit
    /// hook [`Self::honour_neg_curv_floor`], and the point at which a second
    /// escape decides which of two certificates to hold (gh #805) — so the two
    /// cannot drift into disagreeing about which point is the better answer.
    fn current_outranks_neg_curv_floor(
        &self,
        floor: &VetoSnapshot,
        current_certifies: bool,
    ) -> bool {
        let (_, curr_kkt) = self.curr_obj_and_unscaled_kkt();
        let current_success = current_certifies && self.survives_fidelity_gate(curr_kkt);
        let floor_success = self.survives_fidelity_gate(floor.unscaled_kkt);
        match (current_success, floor_success) {
            (true, true) => self.continuation_outranks(floor),
            // The current point reports `Solve_Succeeded` where the floor would
            // be re-graded down. A better status wins outright.
            (true, false) => true,
            // No certificate of its own — the escape was a bet placed *from*
            // one, so anything short of that loses it.
            (false, _) => false,
        }
    }

    /// Make a negative-curvature escape non-destructive (gh #797).
    ///
    /// The escape is a bet placed *from a strict certificate*, which is what
    /// makes its accounting stricter than gh #534's: the floor is a point the
    /// solve was about to report `Solve_Succeeded` at, so the continuation has
    /// to come back with a certificate of its own at a better point to be
    /// preferred. Anything else — a worse point, an acceptable-level stall, a
    /// spent iteration budget, a restoration failure — restores the floor and
    /// reports it under the status it always had.
    ///
    /// Restoring reports `Success` even over a budget or user-stop exit, which
    /// is the opposite of what [`Self::honour_decline_floor`] and
    /// [`Self::honour_best_acceptable_after_dual_guard`] do — and deliberately
    /// so. Those two hold an *acceptable-level* point and must not let it erase
    /// why the solve stopped. Here a pre-#797 build returns `Solve_Succeeded`
    /// at this exact point, and returns it *before* the continuation that spent
    /// the budget ever runs: the budget was spent by the bet, not by the
    /// caller's problem. Reproducing the baseline outcome — point and status
    /// both — is the guarantee, and it is the reading gh #200's hook already
    /// takes of a refused certificate.
    fn honour_neg_curv_floor(&mut self, result: SolverReturn) -> SolverReturn {
        let Some(floor) = self.neg_curv_floor.clone() else {
            return result;
        };
        self.assert_comparable_scale(&floor);
        let keep_continuation =
            self.current_outranks_neg_curv_floor(&floor, matches!(result, SolverReturn::Success));
        if keep_continuation {
            tracing::debug!(target: "pounce::algorithm",
                "[POUNCE] the negative-curvature escape paid off: the continuation \
                 certified a better point than the stationary one it left \
                 (obj {:.10e} -> {:.10e}, gh#797).",
                floor.obj, self.curr_obj_and_unscaled_kkt().0,
            );
            return result;
        }
        tracing::debug!(target: "pounce::algorithm",
            "[POUNCE] the negative-curvature escape did not pay off; restoring the \
             certified stationary point from iter {} (obj {:.10e} viol {:.3e}) \
             and reporting it (gh#797).",
            floor.iter, floor.obj, floor.constr_viol,
        );
        self.restore_snapshot(&floor);
        SolverReturn::Success
    }

    /// The escape's continuation ran out of budget without a certificate of its
    /// own (gh #797). Restore the stationary point the escape left and report
    /// it — that point is what a pre-#797 build returns, and it is a genuine
    /// strict certificate, so `Success` is the honest status for it.
    ///
    /// Takes the floor rather than cloning it, so
    /// [`Self::honour_neg_curv_floor`] is a no-op on the way out.
    fn terminate_at_neg_curv_floor(&mut self) -> IterateOutcome {
        if let Some(floor) = self.neg_curv_floor.take() {
            tracing::debug!(target: "pounce::algorithm",
                "[POUNCE] the negative-curvature escape spent its {} iterations \
                 without a new certificate; restoring the stationary point from \
                 iter {} (obj {:.10e}) and reporting it (gh#797).",
                NEG_CURV_CONTINUATION_BUDGET, floor.iter, floor.obj,
            );
            self.restore_snapshot(&floor);
        }
        IterateOutcome::Terminate(SolverReturn::Success)
    }

    /// Make a deferred restoration decline non-destructive (gh #534).
    ///
    /// The deferral is a bet that a contracting endgame is three iterations from
    /// a certificate. This is what makes losing it cost only those iterations:
    /// whatever the continued run ends up returning, if it is not at least as
    /// good an answer as the floor — the point the pre-#534 guard would have
    /// reported — the floor is restored and reported instead.
    ///
    /// Applied once, in [`Self::optimize`], for the same reason the gh #200 and
    /// pounce#250 hooks are: the driver loop has many `return`s and this must
    /// see all of them.
    ///
    /// A strict `Success` is never overridden — that is the bet paying off, and
    /// a real certificate outranks any acceptable-level point by construction.
    /// The budget statuses keep their own status, as they do in
    /// [`Self::honour_best_acceptable_after_dual_guard`]: a caller polling for
    /// "did I run out of time" must be told so, even while the point it gets
    /// back is swapped for the better one.
    fn honour_decline_floor(&mut self, result: SolverReturn) -> SolverReturn {
        if matches!(result, SolverReturn::Success) {
            if self.decline_floor.is_some() {
                tracing::debug!(target: "pounce::algorithm",
                    "[POUNCE] the deferred restoration decline paid off: the continuation \
                     reached a strict certificate (gh #534).",
                );
            }
            return result;
        }
        let Some(floor) = self.decline_floor.clone() else {
            return result;
        };
        if self.continuation_outranks(&floor) {
            return result;
        }
        tracing::debug!(target: "pounce::algorithm",
            "[POUNCE] the deferred restoration decline did not pay off; restoring the \
             floor from iter {} (obj {:.10e} viol {:.3e}) and reporting it (gh #534).",
            floor.iter, floor.obj, floor.constr_viol,
        );
        self.restore_snapshot(&floor);
        match result {
            SolverReturn::MaxiterExceeded
            | SolverReturn::CpuTimeExceeded
            | SolverReturn::WallTimeExceeded
            | SolverReturn::UserRequestedStop => result,
            _ => SolverReturn::StopAtAcceptablePoint,
        }
    }

    /// Terminal fallback for a near-feasible numerical breakdown (a
    /// restoration cycle or a failed step computation). If a finite
    /// acceptable iterate was recorded earlier in the solve, roll back
    /// to it and stop at [`SolverReturn::StopAtAcceptablePoint`] (mapped
    /// by the application layer to `Solved_To_Acceptable_Level`) rather
    /// than surfacing the hard `fallback` error. This mirrors upstream
    /// `IpBacktrackingLineSearch`'s `ACCEPTABLE_POINT_REACHED`
    /// precedence: when the line search exhausts but an acceptable point
    /// was stored, that point is returned instead of the failure. With
    /// no snapshot — or if the restored objective is non-finite — the
    /// original `fallback` status is surfaced unchanged, so genuinely
    /// failed/infeasible solves keep their honest status. Catches
    /// degenerate LPs (kleemin8, nsir2) whose μ-endgame reaches the
    /// optimum, then destabilizes on the ill-conditioned vertex and
    /// cycles in restoration instead of stopping at the acceptable
    /// iterate it already passed through.
    fn terminate_acceptable_or(&mut self, fallback: SolverReturn) -> IterateOutcome {
        if self.restore_acceptable_point() && self.cq.borrow().curr_f().is_finite() {
            IterateOutcome::Terminate(SolverReturn::StopAtAcceptablePoint)
        } else {
            IterateOutcome::Terminate(fallback)
        }
    }

    /// The single place this module turns a local-infeasibility conclusion
    /// into a returned status (gh #505).
    ///
    /// Three separate routes reach that verdict — the conv-check's rapid
    /// detection, restoration layer 2, and the slow-cycle exits — and the same
    /// defect was found in two of them independently: returning the hard
    /// verdict without consulting the acceptable-point stash, so a solve that
    /// had already passed through an acceptable iterate discarded it. Only the
    /// cycle exits got it right, and nothing structural said the other two were
    /// wrong.
    ///
    /// That is the shape of a defect that comes back. The route a solve takes
    /// to the verdict is an internal detail — the user sees one status either
    /// way — so the *decision* about what that status means must not live at
    /// each route. It lives here, and
    /// [`no_route_concludes_local_infeasibility_alone`] is a tripwire against
    /// a new site rebuilding the outcome inline.
    ///
    /// The cycle exits are not routed through here because their fallback is
    /// chosen between `LocalInfeasibility` and `ErrorInStepComputation` at the
    /// call site; they already reach `terminate_acceptable_or`, which is the
    /// behaviour this guarantees.
    ///
    /// Scope: this governs how *this module* returns the verdict. Other layers
    /// name `SolverReturn::LocalInfeasibility` for their own reasons — the SQP
    /// status map and the ℓ₁ elastic path in `application.rs`, for instance —
    /// and are outside both this funnel and its tripwire.
    fn terminate_local_infeasibility(&mut self) -> IterateOutcome {
        self.terminate_acceptable_or(SolverReturn::LocalInfeasibility)
    }

    pub fn with_nlp(mut self, nlp: Rc<RefCell<dyn IpoptNlp>>) -> Self {
        self.nlp = Some(nlp);
        self
    }

    /// Install a user-facing TNLP handle. Enables per-iteration
    /// `TNLP::intermediate_callback` invocation from `optimize()`.
    pub fn with_tnlp(mut self, tnlp: Rc<RefCell<dyn TNLP>>) -> Self {
        self.tnlp = Some(tnlp);
        self
    }

    /// Build an [`IterStats`] payload from the current `IpoptData` /
    /// `IpoptCq` state. Mirrors the field set the upstream Ipopt main
    /// loop hands to `IntermediateCallback` after each `AcceptTrialPoint`.
    fn build_iter_stats(&self) -> IterStats {
        let d = self.data.borrow();
        let c = self.cq.borrow();
        let dnrm = match d.delta.as_ref() {
            Some(delta) => delta.x.amax().max(delta.s.amax()),
            None => 0.0,
        };
        IterStats {
            // Regular from the outer loop; restoration from the inner
            // sub-IPM, which the outer driver flags at construction
            // (gh#645). The outer loop never sets the flag, so every
            // fire from here is still `RegularMode` — what changed is
            // that restoration now fires at all.
            mode: if self.fires_as_restoration {
                AlgorithmMode::RestorationPhaseMode
            } else {
                AlgorithmMode::RegularMode
            },
            iter: d.iter_count,
            obj_value: c.curr_f(),
            inf_pr: c.curr_primal_infeasibility_max(),
            inf_du: c.curr_dual_infeasibility_max(),
            mu: d.curr_mu,
            d_norm: dnrm,
            regularization_size: d.info_regu_x,
            alpha_du: d.info_alpha_dual,
            alpha_pr: d.info_alpha_primal,
            ls_trials: d.info_ls_count,
        }
    }

    /// Fire `TNLP::intermediate_callback` if a TNLP handle and NLP
    /// handle are installed. Wraps the call in an [`IntermediateContext`]
    /// guard so downstream inspector entry points (the C API's
    /// `GetIpoptCurrent*`) can read live state for the duration. Returns
    /// `true` to continue, `false` if the user requested termination.
    fn fire_intermediate(&self) -> bool {
        let timing = self.data.borrow().timing.clone();
        let _guard = timing.fire_intermediate.guard();
        let Some(tnlp) = self.tnlp.as_ref() else {
            return true;
        };
        let Some(nlp) = self.nlp.as_ref() else {
            return true;
        };
        let stats = self.build_iter_stats();
        // Record exactly what the callback is about to receive, so a losing
        // retry can re-emit the winning attempt's final row (pounce#870).
        if let Some(sink) = self.last_iter_stats_sink.as_ref() {
            *sink.borrow_mut() = Some(stats);
        }
        // The live-inspector context is for iterates of the *user's*
        // problem only. During restoration the iterate belongs to the
        // feasibility subproblem and is not even the same length, so no
        // context is installed and `GetIpoptCurrent*` reports no data
        // for the duration. See `fires_as_restoration`.
        let _guard = (!self.fires_as_restoration).then(|| {
            CtxGuard::install(IntermediateContext {
                data: Rc::clone(&self.data),
                cq: Rc::clone(&self.cq),
                nlp: Rc::clone(nlp),
            })
        });
        tnlp.borrow_mut().intermediate_callback(
            stats,
            &TnlpIpoptData::default(),
            &TnlpIpoptCq::default(),
        )
    }

    pub fn with_search_dir(mut self, sd: PdSearchDirCalc) -> Self {
        self.search_dir = Some(sd);
        self
    }

    pub fn with_restoration(mut self, resto: Box<dyn RestorationPhase>) -> Self {
        self.restoration = Some(resto);
        self
    }

    /// Install the shared diagnostics state. The state is propagated
    /// to the augmented-system solver at the top of [`Self::optimize`]
    /// so dump sites can consult per-iter gating.
    pub fn with_diagnostics(mut self, diag: Rc<DiagnosticsState>) -> Self {
        self.diagnostics = Some(diag);
        self
    }

    /// Install an interactive debugger hook. Fired at each checkpoint
    /// in [`Self::optimize`]; returning [`crate::debug::DebugAction::Stop`]
    /// ends the solve with `SolverReturn::UserRequestedStop`.
    pub fn with_debug_hook(mut self, hook: Rc<RefCell<dyn crate::debug::DebugHook>>) -> Self {
        self.debug = Some(hook);
        self
    }

    /// Shared handle to the installed debugger, if any — used to forward
    /// it into the restoration inner IPM.
    pub fn debug_hook(&self) -> Option<Rc<RefCell<dyn crate::debug::DebugHook>>> {
        self.debug.as_ref().map(Rc::clone)
    }

    /// Fire the debugger hook (if installed) at `cp`, building a live
    /// [`crate::debug::DebugCtx`] over cheap handle clones. Returns the
    /// requested action, defaulting to `Resume` when no hook is set.
    fn fire_debug(&mut self, cp: crate::debug::Checkpoint) -> crate::debug::DebugAction {
        use crate::debug::{DebugAction, DebugCtx};
        // Clone the Rc so the hook borrow is released before we touch
        // `self.bundle` to apply any live option changes below.
        let Some(hook) = self.debug.as_ref().map(Rc::clone) else {
            return DebugAction::Resume;
        };
        let mut ctx = DebugCtx::new(Rc::clone(&self.data), Rc::clone(&self.cq), cp);
        let action = hook.borrow_mut().at_checkpoint(&mut ctx);
        // Drain any tolerances the hook asked to hot-swap and write them
        // into the live convergence-check policy, so the next iteration's
        // termination test uses the new value (no `resolve` needed).
        for (name, value) in ctx.take_live_tolerances() {
            self.bundle.conv_check.set_tolerance(&name, value);
        }
        action
    }

    /// Run the restoration phase, bracketed by the `PreRestoration` /
    /// `PostRestoration` debug checkpoints so a debugger can inspect the
    /// iterate just before entry and just after exit. With no debugger
    /// installed this is exactly `invoke_restoration()`.
    fn invoke_restoration_debugged(&mut self) -> IterateOutcome {
        if let Some(o) = self.debug_stop(crate::debug::Checkpoint::PreRestoration) {
            return o;
        }
        let outcome = self.invoke_restoration();
        if let Some(o) = self.debug_stop(crate::debug::Checkpoint::PostRestoration) {
            return o;
        }
        outcome
    }

    /// Fire a sub-iteration checkpoint from inside [`Self::iterate`].
    /// Returns `Some(Terminate(UserRequestedStop))` if the debugger asked
    /// to stop, so the caller can `return` it; `None` to continue.
    fn debug_stop(&mut self, cp: crate::debug::Checkpoint) -> Option<IterateOutcome> {
        if self.debug.is_none() {
            return None;
        }
        if self.fire_debug(cp) == crate::debug::DebugAction::Stop {
            Some(IterateOutcome::Terminate(SolverReturn::UserRequestedStop))
        } else {
            None
        }
    }

    /// Fire the terminal post-mortem checkpoint (if a debugger is set),
    /// carrying the solve outcome so the hook can decide whether to pause
    /// at the final iterate. The action is advisory — the loop returns
    /// `result` regardless — so the hook just gets a last look.
    fn fire_debug_terminal(&mut self, result: SolverReturn) {
        use crate::debug::{Checkpoint, DebugCtx};
        let Some(hook) = self.debug.as_ref() else {
            return;
        };
        let mut ctx = DebugCtx::new(
            Rc::clone(&self.data),
            Rc::clone(&self.cq),
            Checkpoint::Terminated,
        )
        .with_status(format!("{result:?}"));
        let _ = hook.borrow_mut().at_checkpoint(&mut ctx);
    }

    /// Cheap mid-iteration time-budget check (pounce#242). Returns the
    /// terminal [`SolverReturn`] when the shared [`Deadline`] has been
    /// crossed, so the caller can bail *within* an iteration — after the
    /// KKT factorization, before the line search — rather than only at the
    /// next outer-iteration convergence check. Returns `None` (never
    /// terminating) when no deadline is installed, keeping the
    /// direct-driver / unit-test paths on their `overall_alg`-based gate.
    /// A `None` here is not "no budget" but "check it at the coarse site".
    fn deadline_status(&self) -> Option<SolverReturn> {
        let d = self.data.borrow();
        let kind = d.deadline.as_ref()?.exceeded()?;
        Some(match kind {
            pounce_common::timing::DeadlineKind::Cpu => SolverReturn::CpuTimeExceeded,
            pounce_common::timing::DeadlineKind::Wall => SolverReturn::WallTimeExceeded,
        })
    }

    /// One iteration body — port of `Optimize()`'s inner loop.
    /// Returns either `Continue` to keep iterating or a terminal
    /// [`SolverReturn`] mirroring upstream's exception → return-code
    /// translation table (see `MAIN_LOOP.md` §"Exception mapping").
    fn iterate(&mut self) -> IterateOutcome {
        // Shared timing accumulator — cheap Rc clone so each phase can
        // bump its own counter without re-borrowing `data`.
        let timing = self.data.borrow().timing.clone();

        // Per-iteration span so every event emitted in this body (the
        // structured iteration record, restoration/linear-solve spans)
        // is tagged with the iteration index.
        let _iter_span =
            tracing::info_span!("iteration", iter = self.data.borrow().iter_count).entered();

        // 1. Output iteration row. Header every 10 iters; the row itself
        //    is built plain by the strategy (so the column widths stay
        //    exact and unit-testable) and wrapped in a tiger/rust style
        //    at the print site (pounce#71). `anstream::stdout()` strips
        //    the escapes automatically when stdout is redirected or
        //    `NO_COLOR` is set, so non-TTY output is plain text.
        //
        //    Print BEFORE `reset_info` so the row reflects the accepted
        //    step from the previous iteration (alphas, ls count,
        //    alpha_char), matching upstream's `IpIpoptAlgorithm::Optimize`
        //    ordering.
        timing.output_iteration.start();
        self.bundle.iter_output.write_output();
        if self.print_iter_output {
            use std::io::Write as _;
            let (iter_count, alpha_pr, alpha_char) = {
                let d = self.data.borrow();
                (d.iter_count, d.info_alpha_primal, d.info_alpha_primal_char)
            };
            let row = self.bundle.iter_output.format_row(&self.data, &self.cq);
            // Iteration 0 is the initial point — no step has been taken
            // yet, so `alpha_primal` is 0; treat it as a full step
            // (neutral black) rather than a stalling alarm (red).
            let style_alpha = if iter_count == 0 { 1.0 } else { alpha_pr };
            let style = pounce_common::style::iteration_row_style(style_alpha, alpha_char);
            let mut out = anstream::stdout();
            // Write errors (e.g. a closed pipe / `head` on the output)
            // are deliberately ignored: a vanished terminal must not
            // panic the solver, unlike the old `println!`.
            if iter_count % 10 == 0 {
                let _ = write!(out, "{}", crate::output::orig::OrigIterationOutput::HEADER);
            }
            let _ = writeln!(out, "{}{}{}", style.render(), row, style.render_reset());
        }
        timing.output_iteration.end();

        // Structured per-iteration event (pounce#71) — the single source
        // of truth for the per-iteration trajectory. The JSON log sink
        // and the solve-report collector
        // (`pounce_observability::IterCollectorLayer`) both derive from
        // it. The text console layer filters this target out (its human
        // form is the colored table above).
        //
        // Skipped entirely when nothing consumes it (no iter-history
        // capture active and JSON logging off) so the default run pays
        // no per-iteration field-evaluation / allocation cost.
        if pounce_observability::iteration_event_wanted() {
            // Clone the handles before borrowing so the fingerprint can
            // be moved onto `self.prev_activity` below without the data
            // and cq borrows keeping `self` frozen.
            let data = std::rc::Rc::clone(&self.data);
            let cq = std::rc::Rc::clone(&self.cq);
            let d = data.borrow();
            let c = cq.borrow();

            // Phase profile: how many bounds the barrier
            // currently treats as active, and how many changed class
            // since the previous captured iteration. Computed here
            // rather than in the step so it is paid for only when a
            // consumer is attached, and read as a within-run shape --
            // churn large while the solve is still deciding which
            // constraints bind, falling to zero once it has decided.
            // See `IterRecord::active_bounds` for what the numbers are
            // not evidence about.
            let signs = c.bound_activity_signs();
            let active_bounds = count_active(&signs);
            let active_set_changes = self
                .prev_activity
                .as_ref()
                .and_then(|prev| count_activity_changes(&signs, prev));

            let alpha_char = d.info_alpha_primal_char;
            let alpha_char_s = alpha_char.to_string();
            let d_norm = match &d.delta {
                Some(delta) => delta.x.amax().max(delta.s.amax()),
                None => 0.0,
            };
            tracing::info!(
                target: pounce_observability::ITER_TARGET,
                iter = d.iter_count,
                objective = c.unscaled_curr_f(),
                inf_pr = c.curr_primal_infeasibility_max(),
                inf_du = c.curr_dual_infeasibility_max(),
                mu = d.curr_mu,
                d_norm = d_norm,
                regularization = d.info_regu_x,
                alpha_dual = d.info_alpha_dual,
                alpha_primal = d.info_alpha_primal,
                ls_trials = d.info_ls_count,
                alpha_char = alpha_char_s.as_str(),
                resto_kind = pounce_common::style::resto_kind_str(alpha_char),
                active_bounds = active_bounds,
                // Absent on the first captured iteration: there is no
                // predecessor, which the record distinguishes from a
                // measured zero.
                active_set_changes = active_set_changes,
            );
            drop(d);
            drop(c);
            self.prev_activity = Some(signs);
        }

        // Reset per-iteration info on data (after printing previous
        // iter's accepted-step info; before the next line search).
        self.data.borrow_mut().reset_info();

        // 2. Convergence check.
        timing.check_convergence.start();
        let nlp_err = self.cq.borrow().curr_nlp_error();
        let iter_count = self.data.borrow().iter_count;
        if !nlp_err.is_finite() {
            timing.check_convergence.end();
            return IterateOutcome::Terminate(SolverReturn::InvalidNumberDetected);
        }
        // gh #534 progress history. One sample per outer iteration, recorded
        // before any of the guards below can divert, so the samples the
        // restoration-decline test reads are consecutive by construction.
        self.note_nlp_err(nlp_err);
        // Divergence guard — port of upstream `IpIpoptAlg.cpp` post-
        // AcceptTrialPoint check. When `max_i |x_i|` exceeds the
        // registered `diverging_iterates_tol` (default `1e20`), exit
        // cleanly with `DivergingIterates` rather than spiralling into
        // a degenerate restoration whose inner sub-NLP can't recover
        // (MESH: orig `f` already at -3.6e33 by iter 90, restoration
        // entered too late to bound `x`).
        //
        // A large `|x|` alone does not prove unboundedness, though:
        // `DivergingIterates` is Ipopt's *unboundedness* signal (it maps
        // to the AMPL 300 "unbounded" range), and under severe objective
        // ill-scaling the normal-mode IPM can take a large but transient
        // excursion on a problem that is bounded below with a finite
        // optimum (issue #248: MINLPLib `jit1`). Only conclude divergence
        // when the growth is *structurally* consistent with an unbounded
        // feasible region — some over-threshold component heading toward a
        // side with no finite bound. If every large component is pinned by
        // a finite bound (in particular, all variables boxed), the growth
        // is a scaling artifact, so let the normal convergence / iteration
        // machinery return the best iterate instead of a spurious
        // `Unbounded`.
        // Evaluate the structural check under an immutable borrow, then
        // update the persistence state and take the verdict separately so
        // the mutable field updates don't clash with the `data` borrow.
        // Two independent unboundedness paths share this block:
        //   * the `diverging_iterates_tol` (`1e20`) magnitude guard, gated on
        //     the free-variable structural check + geometric-growth streak
        //     (issues #248 / #252); and
        //   * the #285 recession-ray path — a *checked proof*, active from a
        //     far lower magnitude floor, that catches a genuine recession ray
        //     in `null(A_eq)` over free variables whose `|x|` grows only
        //     linearly and so never reaches `1e20` within `max_iter`.
        //
        // The whole block is skipped while the line search is inside a
        // watchdog trial sequence (gh #818 review). A `'w'` iterate is
        // provisional by construction: the acceptor *rejected* it, the
        // filter was not augmented, and the line search is holding a
        // snapshot it will revert to within `watchdog_trial_iter_max`
        // (default 3) iterations. Reporting `DivergingIterates` there
        // throws that snapshot away and calls a problem unbounded on a
        // point the algorithm had already decided not to keep. This is
        // the same false positive `DIVERGENCE_PERSIST_ITERS` was
        // introduced for — a transient excursion that peaks and recedes —
        // except that a watchdog excursion recedes *by construction*, and
        // `DIVERGENCE_ABS_RUNAWAY` bypasses the streak, so the streak
        // alone does not cover it. Skipping rather than resetting leaves
        // the streak state untouched, so a watchdog gamble in the middle
        // of a genuine ray neither accumulates nor erases evidence; a real
        // divergence is reported at most three iterations later, from a
        // committed iterate. The deferral is capped at
        // `WATCHDOG_DEFER_MAX` consecutive iterations so it can never be
        // held open by a stale `in_watchdog` — see
        // `Self::watchdog_defer_streak` for the path that leaks one.
        //
        // Measured on the gh #818 quadratic at `n = 8`, cond `1e12`,
        // `limited_memory_max_history 6`: the solve reaches iteration 352
        // on the *third* watchdog trial of a sequence, at `|x|_inf ~ 5e22`
        // with the objective climbing to `+2.0e45` — the opposite of the
        // `f -> -inf` that `DivergingIterates` is supposed to mean — one
        // iteration before `StopWatchDog` would have restored an iterate
        // at `f = 2.26e4`.
        let in_watchdog = self.bundle.line_search.in_watchdog()
            && self.watchdog_defer_streak < Self::WATCHDOG_DEFER_MAX;
        if in_watchdog {
            self.watchdog_defer_streak += 1;
        } else {
            self.watchdog_defer_streak = 0;
        }
        let (amax, structural_free, is_ray) = {
            let data = self.data.borrow();
            match data.curr.as_ref() {
                Some(curr) if !in_watchdog => {
                    let amax = curr.x.amax();
                    let structural = amax > self.diverging_iterates_tol
                        && self.divergence_is_true_unboundedness(&*curr.x);
                    let is_ray = amax > Self::RECESSION_MIN_NORM
                        && self.curr_is_recession_ray(&*curr.x, amax);
                    (Some(amax), structural, is_ray)
                }
                _ => (None, false, false),
            }
        };
        // Evaluate the (scaled) objective only while a structural divergence
        // is live — the streak's descent gate needs it, and it costs an
        // objective evaluation, so skip it on the common non-diverging path.
        let curr_f = structural_free.then(|| self.cq.borrow().curr_f());
        // Evaluate both streak updates (no short-circuit) so each keeps its
        // state current, then fire if either concludes divergence.
        // ... but only when the streaks are actually being fed. Inside a
        // watchdog sequence `amax` is `None`, and running the updates
        // would reset both streaks on a point they never saw.
        let (fire_magnitude, fire_recession) = if in_watchdog {
            (false, false)
        } else {
            (
                self.update_divergence_verdict(amax, structural_free, curr_f),
                self.update_recession_verdict(amax.unwrap_or(0.0), is_ray),
            )
        };
        if fire_magnitude || fire_recession {
            if fire_recession && !fire_magnitude {
                tracing::debug!(target: "pounce::algorithm",
                    "[POUNCE] recession-ray guard fired at iter {} (|x|_inf={:.2e}); \
                     reporting DivergingIterates (pounce#285).",
                    self.data.borrow().iter_count,
                    amax.unwrap_or(f64::NAN),
                );
            }
            timing.check_convergence.end();
            return IterateOutcome::Terminate(SolverReturn::DivergingIterates);
        }
        // Dual-divergence guard (pounce#246). The primal guard above only
        // catches `|x|` blowing up; a bad warm start can instead send the
        // *dual* infeasibility diverging — `inf_du` 1 -> 1e14, the inertia
        // regularization -> 1e14, the barrier parameter frozen, full steps
        // still accepted by the filter because primal feasibility inches
        // down — while `|x|` stays bounded. `diverging_iterates_tol` never
        // trips, restoration is never entered, and the solve grinds in
        // ever-more-ill-conditioned KKT factorizations that each take
        // seconds (the emfl050 warm-start overshoot: one 3.8 s factorization
        // per iteration, forever). Detect a sustained streak of growing dual
        // infeasibility in the elevated regime and route to restoration —
        // the same recovery the least-square-multiplier init path reaches on
        // its own — before the factorizations start choking. Gated on a
        // large absolute `inf_du` so a well-behaved solve whose dual
        // residual transiently rises (then falls) is never diverted:
        // restoration is a heavier hammer than the guard should swing at a
        // merely-bumpy-but-converging iterate.
        //
        // OFF BY DEFAULT (pounce#250 follow-up). The emfl050 overshoot above is
        // how this was justified, and it did not reproduce: that measurement was
        // caller-side JAX compilation, and the build predating the guard solves
        // both emfl050 instances to the same optimum in the same time. What is
        // left is an effect on four of 1284 MINLPLib models that is knife-edge
        // and non-monotone in `dual_diverging_streak` — a better local optimum on
        // deb7/deb9 at exactly 15, and pooling_rt2stp turning Solve_Succeeded
        // into Maximum_Iterations_Exceeded at 10 and 15 only. Kept because it
        // does help when it helps, but not imposed. Full account in the option
        // help (`upstream_options.rs`).
        //
        // Two things to know before changing this:
        //
        // * `curr_dual_infeasibility_max` is the RAW ‖∇L‖∞, not divided by the
        //   `s_d` optimality scaling the convergence check applies, and this runs
        //   *before* `conv_check`. So the thresholds below are not on the same
        //   quantity the solver's own tolerances are on, and the claim that they
        //   are scale-robust holds only while `nlp_scaling_method != none`. No
        //   exploit is known; the margin is thinner than it looks.
        // * The `DivergingIterates` fallback at the end is unreachable from every
        //   shipped front end — CLI, pounce-py and cinterface all wire a
        //   restoration provider, so the guard can only ever route to
        //   restoration. Do not assume it is dead code and delete the provider
        //   check; do not assume it is live and rely on the status either.
        // gh#884 — the dual-divergence-at-a-settled-primal signature.
        //
        // Four conjuncts, all required **at the same iterate**. That is not
        // stylistic: measured on the corpus, `deb7` on the L-BFGS leg reaches
        // a settled step of `5.9e-13` at an unscaled dual of `8.6e-7`, and its
        // *maximum* unscaled dual over settled iterates is `7.2e6`. A
        // formulation that took the minimum step and the maximum dual over a
        // window would fire on it; this one does not.
        //
        // What it is for: at a biactive complementarity pair the product row's
        // multiplier is arbitrary rather than determined, so it runs away, `s_d`
        // grows with it, and the `s_d`-normalised aggregate the convergence
        // check reads stays clean while the honest residual is `7.9e+04`. The
        // one quantity that feedback loop cannot fake is the step: with the
        // primal settled and the duals diverging the direction collapses. So
        // this reads the *raw unscaled* residual, and only at iterates where
        // the algorithm has demonstrably stopped moving with the primal solved.
        // A multiplier of `1e9` on a `1e-9` gradient cannot satisfy it, because
        // nothing here is normalised by a multiplier — which is gh#884's
        // criterion 2.
        //
        // It never changes a verdict. All it does is authorize the application
        // layer to spend a second solve; see `run_with_dual_divergence_retry`.
        if !self.dual_divergence_signature && self.dual_divergence_retry_step_tol > 0.0 {
            let cq = self.cq.borrow();
            let has_rows = cq.curr_c().dim() > 0 || cq.curr_d().dim() > 0;
            if has_rows && self.last_step_rel <= self.dual_divergence_retry_step_tol {
                let inf_pr = cq.curr_primal_infeasibility_max();
                let unscaled_du = cq.curr_unscaled_dual_infeasibility_max();
                if inf_pr <= DUAL_DIV_RETRY_PRIMAL_TOL
                    && unscaled_du >= self.dual_divergence_retry_du_floor
                    && unscaled_du.is_finite()
                {
                    self.dual_divergence_signature = true;
                    tracing::debug!(target: "pounce::algorithm",
                        "[POUNCE] gh#884 dual-divergence signature at iter {}: \
                         step_rel={:.2e} inf_pr={:.2e} unscaled_inf_du={:.2e}; \
                         a cold retry is authorized if this solve does not succeed.",
                        self.data.borrow().iter_count,
                        self.last_step_rel, inf_pr, unscaled_du,
                    );
                }
            }
        }

        if self.dual_diverging_streak > 0 {
            let inf_du = self.cq.borrow().curr_dual_infeasibility_max();
            if inf_du.is_finite() && inf_du > self.dual_inf_prev && inf_du > DUAL_DIV_COUNT_FLOOR {
                self.dual_growth_streak += 1;
            } else {
                self.dual_growth_streak = 0;
            }
            self.dual_inf_prev = inf_du;
            if self.dual_growth_streak >= self.dual_diverging_streak && inf_du > DUAL_DIV_FIRE_TOL {
                self.dual_growth_streak = 0;
                self.dual_inf_prev = 0.0;
                // Arm the "never worse off" bookkeeping for the bet about to be
                // placed (pounce#250 follow-up).
                self.dual_guard_fired = true;
                timing.check_convergence.end();
                tracing::debug!(target: "pounce::algorithm",
                    "[POUNCE] dual-divergence guard fired at iter {} (inf_du={:.2e}); \
                     routing to restoration (pounce#246).",
                    self.data.borrow().iter_count, inf_du,
                );
                if self.restoration.is_some() {
                    return self.invoke_restoration_debugged();
                }
                return IterateOutcome::Terminate(SolverReturn::DivergingIterates);
            }
        }
        let conv_status = self
            .bundle
            .conv_check
            .check_convergence_with_state(nlp_err, iter_count, &self.data, &self.cq);
        // Snapshot the *first* refused certificate. Baseline would have stopped
        // and returned exactly this point, so keeping it — and only it — is what
        // makes the "never worse" guarantee exact rather than approximate. A
        // later refusal is also a valid certificate but not necessarily a
        // better one, so it must not overwrite this.
        if !self.vetoed_seen && self.bundle.conv_check.certificate_vetoed() {
            // Latch on *seeing* the refusal, not on the snapshot being present:
            // the veto flag is sticky, so keying off `vetoed.is_none()` would
            // let a failed capture be completed at a later, arbitrary iterate.
            // See `IpoptAlgorithm::vetoed_seen`.
            self.vetoed_seen = true;
            self.vetoed = self.snapshot_current(iter_count);
        }
        if !self.vetoed_acceptable_seen && self.bundle.conv_check.acceptable_certificate_vetoed() {
            self.vetoed_acceptable_seen = true;
            self.vetoed_acceptable = self.snapshot_current(iter_count);
        }
        // gh #695: a successful verdict asserts the convergence test passed;
        // reporting one alongside a non-finite objective is self-contradictory,
        // and a caller that gates on `status` and then reads `obj_val` silently
        // receives `NaN`. The convergence test cannot notice on its own — it
        // reads gradients, residuals and complementarity, never the objective
        // *value* — so with finite derivatives and a satisfied equality the KKT
        // residuals are genuinely small and the solve converges on a point
        // whose objective is not a number.
        //
        // Only the *equality*-constrained shape reached here: the unconstrained
        // and bounds-only shapes fail in the step computation and the
        // inequality-constrained one already trips an invalid-number guard, so
        // this closes the one column of that matrix that was reporting success.
        // gh #292 closed the NaN-*gradient* hole and recorded `f`-returns-NaN as
        // the safe contrast case, which held for the shapes it exercised and not
        // for this one.
        //
        // `Invalid_Number_Detected` is the status Ipopt's `Eval_f` gives a
        // non-finite objective, which POUNCE's own inequality-constrained shape
        // already agreed with. The same check already guards the restoration
        // near-feasible exit below, for the same reason on a different path
        // (CUTE `himmelbj`); this extends it to the ordinary convergence exit
        // rather than adding a second rule.
        let converged_success = matches!(
            conv_status,
            ConvergenceStatus::Converged | ConvergenceStatus::ConvergedToAcceptable
        );
        if converged_success && !self.cq.borrow().curr_f().is_finite() {
            timing.check_convergence.end();
            return IterateOutcome::Terminate(SolverReturn::InvalidNumberDetected);
        }
        match conv_status {
            ConvergenceStatus::Continue => {}
            ConvergenceStatus::Converged => {
                timing.check_convergence.end();
                // gh #797: first-order stationarity is not a local minimum on a
                // nonconvex model. If the reduced Hessian here is indefinite,
                // leave along a direction of negative curvature instead of
                // certifying a constrained maximum — bounded in cost, and
                // floored at this very point.
                if self.try_neg_curv_escape(iter_count) {
                    return IterateOutcome::Continue;
                }
                return IterateOutcome::Terminate(SolverReturn::Success);
            }
            ConvergenceStatus::ConvergedToAcceptable => {
                timing.check_convergence.end();
                return IterateOutcome::Terminate(SolverReturn::StopAtAcceptablePoint);
            }
            ConvergenceStatus::MaxIterExceeded => {
                timing.check_convergence.end();
                return IterateOutcome::Terminate(SolverReturn::MaxiterExceeded);
            }
            ConvergenceStatus::CpuTimeExceeded => {
                timing.check_convergence.end();
                return IterateOutcome::Terminate(SolverReturn::CpuTimeExceeded);
            }
            ConvergenceStatus::WallTimeExceeded => {
                timing.check_convergence.end();
                return IterateOutcome::Terminate(SolverReturn::WallTimeExceeded);
            }
            ConvergenceStatus::LocallyInfeasible => {
                timing.check_convergence.end();
                // gh #505: consult the acceptable-point stash, as the
                // restoration-cycle exits below already do (`:2686`, `:2716`,
                // both via `terminate_acceptable_or`). This arm used to return
                // without it, so a solve that had passed through an acceptable
                // iterate — stashed, un-vetoed, sitting there as a rollback
                // target — discarded it and surfaced the hard verdict instead.
                // The stashing code sits *after* this match, so the firing
                // iteration returns before it would even consider stashing;
                // only iterates from earlier in the solve are on offer, which
                // is exactly what a rollback target is.
                //
                // This is about what to *return* once the verdict has fired,
                // not about when it fires. Whether the rapid detector should
                // have convicted this point at all is a separate question,
                // answered by the violation floor in `OptErrorConvCheck`
                // (gh #519).
                //
                // Inert on genuinely infeasible models: `store_acceptable_point`
                // is gated on `current_is_acceptable_with_state`, which requires
                // `acceptable_tol` *and* the unscaled violation against
                // `acceptable_constr_viol_tol`, and the scale-relative veto
                // blocks the stash outright for a row violated relative to its
                // own magnitude. Nothing is stashed on such a model, so
                // `terminate_acceptable_or` falls through to the verdict
                // unchanged. `infeasible_models_are_never_reported_solved`
                // (`infeasible_status_tol_invariance.rs`) is the standing guard.
                //
                // That inertness rests entirely on the stash gate having no
                // iteration budget (gh #693). While it carried the certificate
                // veto's `VETO_MAX_EXTRA_ITERS`, a solve that took more than 60
                // blocked iterations to convict stashed the very point the veto
                // exists to reject and rolled back to it here — an infeasible
                // model reported `Solved_To_Acceptable_Level`. See
                // `issue_693_relative_infeasibility_stash.rs`.
                return self.terminate_local_infeasibility();
            }
            ConvergenceStatus::Failed => {
                timing.check_convergence.end();
                return IterateOutcome::Terminate(SolverReturn::InternalError);
            }
        }

        // Stash the iterate if it satisfies the per-component
        // `acceptable_*_tol` triplet. Mirrors upstream
        // `IpBacktrackingLineSearch.cpp:282-289` — checked at the top
        // of every line-search call so the most recent acceptable
        // iterate is always available as a rollback target if
        // restoration later fails. The recorder feeds
        // `acceptable_obj_change_tol`'s stability cross-check on
        // subsequent iterates.
        if self
            .bundle
            .conv_check
            .current_is_acceptable_with_state(nlp_err, &self.data, &self.cq)
        {
            self.store_acceptable_point();
            let curr_f = self.cq.borrow().curr_f();
            self.bundle.conv_check.set_curr_acceptable_obj(curr_f);
            // pounce#250 follow-up: keep the *best* acceptable iterate, not just
            // the latest. `store_acceptable_point` overwrites unconditionally,
            // so once the dual-divergence guard diverts a solve the rollback
            // target drifts to whatever the diverted run last touched — which
            // may be far worse than a point already in hand. Recorded on every
            // acceptable iterate (including before any diversion) and read only
            // when the guard fired; see `honour_best_acceptable_after_dual_guard`.
            self.record_best_acceptable(curr_f);
        }
        timing.check_convergence.end();

        // gh #534: a deferred restoration decline is a bet with a deadline.
        // Checked *after* the convergence check, so a strict certificate the
        // continuation reached in the meantime wins the bet rather than being
        // pre-empted by its own expiry; and after the acceptable stash, so a
        // continuation that ended somewhere better has been recorded before the
        // floor comparison reads it.
        if self.decline_deadline_iter.is_some_and(|d| iter_count > d) {
            return self.terminate_at_decline_floor();
        }

        // gh #797: the negative-curvature escape is the same kind of bet, and
        // its deadline is checked in the same place and for the same reason —
        // after the convergence check, so a certificate the continuation
        // reached wins rather than being pre-empted by its own expiry.
        if self.neg_curv_deadline_iter.is_some_and(|d| iter_count > d) {
            return self.terminate_at_neg_curv_floor();
        }

        // 3. Hessian update. Must run BEFORE `update_barrier_parameter`
        // so the adaptive-μ oracles (probing, quality-function) drive
        // their affine/centering solves against `W(curr_N)`, not the
        // stale `W(curr_{N-1})` left in `data.w` by the previous iter's
        // tail-end Hessian update. Upstream calls `UpdateHessian()`
        // first in every main-loop body (`IpIpoptAlg.cpp:386`); pounce
        // previously reordered this to the tail, which made iters 1+
        // pick μ from the prior iterate's Hessian on adaptive-mu +
        // quality-function — visible on CRESC50 as a catastrophic
        // early-iter divergence (theta=5.8e5 by iter 61 vs upstream
        // never entering restoration).
        timing.update_hessian.start();
        let _ = self.bundle.hess.update_hessian(&self.data, &self.cq);
        timing.update_hessian.end();

        // 4. Barrier parameter. Pass nlp + search_dir through so the
        // adaptive μ oracles (probing, quality-function) can drive
        // their own affine-step solves; monotone ignores them.
        // Snapshot the tiny-step flag (set by the previous iteration's
        // tiny-step branch) and the entry mu — if μ can't reduce while
        // the flag is on, upstream `IpMonotoneMuUpdate.cpp:158-161`
        // throws TINY_STEP_DETECTED → STOP_AT_TINY_STEP, which we
        // realise as a clean termination here.
        //
        // Both updates terminate, by different routes (pounce#512).
        // Monotone has one throw site covering its whole update, so the
        // μ-unchanged comparison below reconstructs it exactly, gated on
        // `terminates_on_tiny_step()`. `IpAdaptiveMuUpdate.cpp` throws at
        // two specific sites (`:330-333`, `:377-380`) and merely fixes μ
        // and keeps iterating elsewhere, so the comparison would over-fire
        // there — on the no-bounds short-circuit, which returns before
        // upstream even reads the flag, and on a free-mode oracle that
        // re-picks the current μ. The adaptive update therefore raises
        // `request_tiny_step_stop` at its own two sites and opts out of
        // the comparison. (An earlier comment here claimed the adaptive
        // update never self-terminates; it does — `force_no_progress` is
        // what happens on the iterations that do *not* throw.)
        timing.update_barrier_parameter.start();
        let tiny_at_entry = self.data.borrow().tiny_step_flag;
        let mu_before = self.data.borrow().curr_mu;
        let mu_terminates_on_tiny = self.bundle.mu_update.terminates_on_tiny_step();
        let next_mu = self.bundle.mu_update.update_barrier_parameter(
            &self.data,
            &self.cq,
            self.nlp.as_ref(),
            self.search_dir.as_mut(),
        );
        self.data.borrow_mut().curr_mu = next_mu;
        timing.update_barrier_parameter.end();

        // pounce#510 — line-search reset. Upstream's μ updates own a
        // `linesearch_` handle and call `linesearch_->Reset()` (which
        // clears the filter via `FilterLSAcceptor::Reset`,
        // `IpFilterLSAcceptor.cpp:524-532`) at four fixed points:
        // `IpAdaptiveMuUpdate.cpp:339` (fixed-mode decrease), `:386`
        // (free→fixed switch), `:431` (**unconditionally** on every
        // free-mode iteration, μ moved or not), and
        // `IpMonotoneMuUpdate.cpp:165` (after a monotone reduction).
        // Pounce's `MuUpdate` trait has no line-search handle, so each
        // update raises `request_ls_reset` at exactly those points and
        // we honour it here — the same plumbing `request_resto` uses
        // below.
        //
        // This used to be inferred from `next_mu != mu_before`. That
        // proxy is right for the monotone update but wrong for the
        // adaptive one, which resets every free-mode iteration
        // regardless of μ: whenever μ stayed numerically put (the
        // free-mode endgame, and any iteration after a restoration that
        // returns at the same μ) the filter kept entries computed
        // against a barrier parameter and an iterate the algorithm had
        // already left. On #505's reproducer that rejected every trial
        // step from α=2.4e-6 down to 1e-12 on the filter alone and
        // forced a spurious restoration.
        //
        // Both flags are consumed here, but the tiny-step stop is
        // answered first: at each of the two adaptive sites that raise
        // it, upstream's `TINY_STEP_DETECTED` throw sits *above* the
        // reset it would otherwise reach (`cpp:330-333` before `:339`,
        // `:377-380` before `:386`), so a terminating iteration never
        // resets the line search.
        let (tiny_step_stop_requested, ls_reset) = {
            let mut d = self.data.borrow_mut();
            let flags = (d.request_tiny_step_stop, d.request_ls_reset);
            d.request_tiny_step_stop = false;
            d.request_ls_reset = false;
            flags
        };
        if tiny_step_stop_requested
            || (tiny_at_entry
                && mu_terminates_on_tiny
                && (next_mu - mu_before).abs() < Number::EPSILON)
        {
            return IterateOutcome::Terminate(SolverReturn::StopAtTinyStep);
        }
        if ls_reset {
            self.bundle.line_search.reset();
        }

        // pounce#58 — iterate-quality guard for the probing oracle.
        // The μ-update layer sets `request_resto` when the input
        // iterate is too corrupted for the probing rule to produce a
        // sane μ (see `mu/adaptive.rs` Probing dispatch). Restoration
        // re-initialises the multipliers and gives the outer loop a
        // clean iterate to continue from. When no restoration phase
        // is configured (embedded callers, tests), emit a one-line
        // notice and continue with the current μ — the guard has
        // already prevented the destabilising 4-order μ jump.
        let request_resto = {
            let mut d = self.data.borrow_mut();
            let f = d.request_resto;
            d.request_resto = false;
            // `start_with_resto` — upstream's "switch to the feasibility
            // restoration phase in the first iteration". It rides the
            // same request flag rather than adding a second path into
            // restoration, and it is consumed here so it fires exactly
            // once: `iter_count` is 0 only on the first pass, and
            // restoration advances it.
            f || (self.start_with_resto && d.iter_count == 0)
        };
        if request_resto {
            if self.restoration.is_some() {
                return self.invoke_restoration_debugged();
            } else {
                tracing::warn!(target: "pounce::algorithm",
                    "[POUNCE] probing-oracle iterate-quality guard fired \
                     at iter {}, but no restoration phase is configured; \
                     continuing with μ={:.3e}.",
                    self.data.borrow().iter_count,
                    next_mu,
                );
            }
        }

        // Sub-iteration checkpoint: μ has been updated for this iteration.
        if let Some(o) = self.debug_stop(crate::debug::Checkpoint::AfterBarrierUpdate) {
            return o;
        }

        // 4b. `linear_system_scaling=slack-based` — refresh the
        //     iterate-dependent part of the augmented-system scaling
        //     before anything factorizes it.
        //
        //     The other scaling methods (Ruiz, MC19) derive their
        //     factors from the matrix they are handed and need nothing
        //     from here. Slack-based is a function of the iterate, and
        //     upstream's method reads `IpCq()` directly; pounce's
        //     scaling methods live in `pounce-linsol`, below the
        //     algorithm, so the value is computed here and pushed down.
        //     Inert unless the option selected it.
        self.push_slack_scaling();

        // 5. Search direction. Skipped without an NLP + search_dir.
        // (Hessian was updated in step 3 above before the barrier-μ
        // oracle so that adaptive-μ uses W(curr_N), not stale W.)
        if let (Some(nlp), Some(sd)) = (self.nlp.as_ref(), self.search_dir.as_mut()) {
            timing.compute_search_direction.start();
            // Fields are declared `Empty` and filled by the linear
            // solver (matrix size, factor nnz, inertia, ordering — see
            // `pounce_feral::record_factor_stats`) and below
            // (regularization), so the `linear_solve` span carries the
            // KKT-solve characteristics for the JSON sink (pounce#71).
            let ls_span = tracing::info_span!(
                target: "pounce::linsol",
                "linear_solve",
                n = tracing::field::Empty,
                matrix_nnz = tracing::field::Empty,
                factor_nnz = tracing::field::Empty,
                inertia_neg = tracing::field::Empty,
                fill_ratio = tracing::field::Empty,
                ordering = tracing::field::Empty,
                regularization = tracing::field::Empty,
            );
            let ls_enter = ls_span.enter();
            let ok = sd.compute_search_direction(&self.data, &self.cq, nlp);
            ls_span.record("regularization", self.data.borrow().info_regu_x);
            // Within-span marker so the enriched `linear_solve` fields
            // (filled by the solver above) surface to the JSON sink at
            // debug level; off at the default `info` level.
            tracing::debug!(target: "pounce::linsol", "kkt solve complete");
            drop(ls_enter);
            timing.compute_search_direction.end();
            // Fine-grained time-budget gate (pounce#244). The KKT solve now
            // checks the shared deadline *between* its major factorization
            // steps (inertia correction / iterative refinement) and aborts
            // cooperatively when the budget is crossed — bounding the
            // overshoot to roughly one factorization instead of the whole
            // multi-factorization sweep that #242's post-solve check let run
            // to completion. Whether the solve returned a completed step or
            // bailed mid-escalation, if the deadline tripped, stop here with
            // the time-limit status *before* the `!ok` branch below would
            // otherwise route a deadline-aborted solve into restoration.
            // `data.curr` is untouched by the step computation, so it still
            // holds the last accepted iterate.
            if let Some(ret) = self.deadline_status() {
                return IterateOutcome::Terminate(ret);
            }
            if !ok {
                // Mirror upstream `IpIpoptAlg.cpp:417-430`: a failed
                // step computation puts the algorithm in emergency
                // mode, which calls `BacktrackingLineSearch::
                // ActivateFallbackMechanism` (cpp:1312-1328). When a
                // restoration phase is configured, the next pass of
                // `ComputeAcceptableTrialPoint` sees `goto_resto` at
                // cpp:299-306 and hands control to restoration. Only
                // when neither restoration nor an acceptor-level
                // fallback is available does upstream throw
                // `STEP_COMPUTATION_FAILED`.
                if self.restoration.is_some() {
                    return self.invoke_restoration_debugged();
                }
                return IterateOutcome::Terminate(SolverReturn::ErrorInStepComputation);
            }
            if std::env::var_os("POUNCE_DBG_DELTA").is_some() {
                let d = self.data.borrow();
                let it = d.iter_count;
                if let Some(delta) = d.delta.as_ref() {
                    use crate::iterates_vector::IteratesVector;
                    use pounce_linalg::{Vector, compound_vector::CompoundVector};
                    let dv: &IteratesVector = delta;
                    tracing::debug!(target: "pounce::algorithm",
                        "[PN_DELTA] iter={} mu={:.6e} dx_amax={:.6e} ds_amax={:.6e} dyc_amax={:.6e} dyd_amax={:.6e} dzL_amax={:.6e} dzU_amax={:.6e} dvL_amax={:.6e} dvU_amax={:.6e}",
                        it, d.curr_mu,
                        dv.x.amax(), dv.s.amax(), dv.y_c.amax(), dv.y_d.amax(),
                        dv.z_l.amax(), dv.z_u.amax(), dv.v_l.amax(), dv.v_u.amax()
                    );
                    if let Some(cdx) = dv.x.as_any().downcast_ref::<CompoundVector>() {
                        tracing::debug!(target: "pounce::algorithm",
                            "[PN_DELTA] iter={} dx_blocks_amax: orig={:.6e} nc={:.6e} pc={:.6e} nd={:.6e} pd={:.6e}",
                            it,
                            cdx.comp(0).amax(),
                            cdx.comp(1).amax(),
                            cdx.comp(2).amax(),
                            cdx.comp(3).amax(),
                            cdx.comp(4).amax(),
                        );
                        tracing::debug!(target: "pounce::algorithm",
                            "[PN_DELTA] iter={} dx_blocks_nrm2: orig={:.6e} nc={:.6e} pc={:.6e} nd={:.6e} pd={:.6e}",
                            it,
                            cdx.comp(0).nrm2(),
                            cdx.comp(1).nrm2(),
                            cdx.comp(2).nrm2(),
                            cdx.comp(3).nrm2(),
                            cdx.comp(4).nrm2(),
                        );
                        tracing::debug!(target: "pounce::algorithm",
                            "[PN_DELTA] iter={} dx_blocks_asum: orig={:.6e} nc={:.6e} pc={:.6e} nd={:.6e} pd={:.6e}",
                            it,
                            cdx.comp(0).asum(),
                            cdx.comp(1).asum(),
                            cdx.comp(2).asum(),
                            cdx.comp(3).asum(),
                            cdx.comp(4).asum(),
                        );
                        // Argmax of orig block via dot with sign — print first few values.
                        if let Some(dv_orig) =
                            cdx.comp(0)
                                .as_any()
                                .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
                        {
                            let v = dv_orig.values();
                            let mut imax = 0usize;
                            let mut amax = 0.0f64;
                            for (i, &x) in v.iter().enumerate() {
                                if x.abs() > amax {
                                    amax = x.abs();
                                    imax = i;
                                }
                            }
                            tracing::debug!(target: "pounce::algorithm",
                                "[PN_DELTA] iter={} dx_orig argmax: i={} v={:.17e} (n={})",
                                it,
                                imax,
                                v[imax],
                                v.len()
                            );
                        }
                    }
                    let p = &d.perturbations;
                    tracing::debug!(target: "pounce::algorithm",
                        "[PN_DELTA] iter={} pert: dx={:.6e} ds={:.6e} dc={:.6e} dd={:.6e}",
                        it, p.delta_x, p.delta_s, p.delta_c, p.delta_d
                    );
                    drop(d);
                    let cq = self.cq.borrow();
                    let gf = cq.curr_grad_f();
                    let gl = cq.curr_grad_lag_x();
                    let cc = cq.curr_c();
                    let cd = cq.curr_d_minus_s();
                    let sx = cq.curr_sigma_x();
                    let ss = cq.curr_sigma_s();
                    tracing::debug!(target: "pounce::algorithm",
                        "[PN_DELTA] iter={} cq: gradf_amax={:.6e} gradf_nrm2={:.6e} gradlag_amax={:.6e} gradlag_nrm2={:.6e} c_amax={:.6e} c_nrm2={:.6e} d_amax={:.6e} d_nrm2={:.6e} sigx_amax={:.6e} sigx_nrm2={:.6e} sigs_amax={:.6e} sigs_nrm2={:.6e}",
                        it,
                        gf.amax(), gf.nrm2(),
                        gl.amax(), gl.nrm2(),
                        cc.amax(), cc.nrm2(),
                        cd.amax(), cd.nrm2(),
                        sx.amax(), sx.nrm2(),
                        ss.amax(), ss.nrm2(),
                    );
                    if let Some(cgf) = gf.as_any().downcast_ref::<CompoundVector>() {
                        tracing::debug!(target: "pounce::algorithm",
                            "[PN_DELTA] iter={} gradf_blocks_amax: orig={:.6e} nc={:.6e} pc={:.6e} nd={:.6e} pd={:.6e}",
                            it,
                            cgf.comp(0).amax(),
                            cgf.comp(1).amax(),
                            cgf.comp(2).amax(),
                            cgf.comp(3).amax(),
                            cgf.comp(4).amax(),
                        );
                    }
                    if let Some(curr) = self.data.borrow().curr.clone() {
                        tracing::debug!(target: "pounce::algorithm",
                            "[PN_DELTA] iter={} bound_mults: zL_amax={:.6e} zU_amax={:.6e} vL_amax={:.6e} vU_amax={:.6e} s_amax={:.6e} s_nrm2={:.6e} x_amax={:.6e} x_nrm2={:.6e}",
                            it,
                            curr.z_l.amax(), curr.z_u.amax(),
                            curr.v_l.amax(), curr.v_u.amax(),
                            curr.s.amax(), curr.s.nrm2(),
                            curr.x.amax(), curr.x.nrm2(),
                        );
                        if let Some(czl) = curr.z_l.as_any().downcast_ref::<CompoundVector>() {
                            tracing::debug!(target: "pounce::algorithm",
                                "[PN_DELTA] iter={} zL_blocks_amax: orig={:.6e} nc={:.6e} pc={:.6e} nd={:.6e} pd={:.6e}",
                                it,
                                czl.comp(0).amax(),
                                czl.comp(1).amax(),
                                czl.comp(2).amax(),
                                czl.comp(3).amax(),
                                czl.comp(4).amax(),
                            );
                        }
                        if let Some(czu) = curr.z_u.as_any().downcast_ref::<CompoundVector>() {
                            tracing::debug!(target: "pounce::algorithm", "[PN_DELTA] iter={} zU_ncomps={}", it, czu.n_comps());
                            for ic in 0..czu.n_comps() {
                                tracing::debug!(target: "pounce::algorithm",
                                    "[PN_DELTA] iter={} zU_block[{}]_amax={:.6e} dim={}",
                                    it,
                                    ic,
                                    czu.comp(ic).amax(),
                                    czu.comp(ic).dim()
                                );
                            }
                        }
                    }
                    if let Some(csx) = sx.as_any().downcast_ref::<CompoundVector>() {
                        tracing::debug!(target: "pounce::algorithm",
                            "[PN_DELTA] iter={} sigx_blocks_amax: orig={:.6e} nc={:.6e} pc={:.6e} nd={:.6e} pd={:.6e}",
                            it,
                            csx.comp(0).amax(),
                            csx.comp(1).amax(),
                            csx.comp(2).amax(),
                            csx.comp(3).amax(),
                            csx.comp(4).amax(),
                        );
                    }
                    drop(cq);
                    let d = self.data.borrow();
                    // Also dump curr.x_orig argmax
                    if let Some(curr) = d.curr.as_ref() {
                        if let Some(cx) = curr.x.as_any().downcast_ref::<CompoundVector>() {
                            if let Some(xo) =
                                cx.comp(0)
                                    .as_any()
                                    .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
                            {
                                let v = xo.values();
                                let mut imax = 0usize;
                                let mut amax = 0.0f64;
                                for (i, &x) in v.iter().enumerate() {
                                    if x.abs() > amax {
                                        amax = x.abs();
                                        imax = i;
                                    }
                                }
                                tracing::debug!(target: "pounce::algorithm", "[PN_DELTA] iter={} curr_x_orig argmax: i={} v={:.17e} amax={:.17e} nrm2={:.17e}",
                                it, imax, v[imax], xo.amax(), xo.nrm2());
                            }
                        }
                    }
                }
            }
        }

        // Capture KKT-factorization diagnostics for the debugger before
        // the line search runs. Only when a debugger is installed. The
        // inertia/status fields are cheap and always captured; the matrix
        // triplets and `LDLᵀ` factor are O(nnz) assemblies, so they're
        // captured only while the debugger is stepping (`wants_kkt_capture`)
        // — a detached/free-running debugger drops them to keep the run
        // cheap. `kkt_debug` is overwritten every iteration and never
        // cleared at `iter_start`, so a stepping session always has the
        // previous iteration's system to look back at via `viz kkt`/`viz L`.
        if let Some(hook) = self.debug.as_ref() {
            let capture_heavy = hook.borrow().wants_kkt_capture();
            let captured_iter = self.data.borrow().iter_count;
            let info = self.search_dir.as_ref().map(|sd| {
                let pd = sd.pd_solver_mut();
                let aug = pd.aug_solver();
                let provides = aug.provides_inertia();
                crate::ipopt_data::KktDebug {
                    iter: captured_iter,
                    dim: aug.system_dim(),
                    n_neg: if provides {
                        aug.number_of_neg_evals()
                    } else {
                        -1
                    },
                    provides_inertia: provides,
                    status: format!("{:?}", aug.last_solve_status()),
                    matrix: if capture_heavy {
                        aug.kkt_triplets()
                    } else {
                        None
                    },
                    l_factor: if capture_heavy {
                        aug.l_factor(true)
                    } else {
                        None
                    },
                }
            });
            self.data.borrow_mut().kkt_debug = info;
        }

        // Sub-iteration checkpoint: the Newton step `δ` (data.delta) and
        // the applied regularization are now available, before the line
        // search consumes them.
        if let Some(o) = self.debug_stop(crate::debug::Checkpoint::AfterSearchDirection) {
            return o;
        }

        // Fine-grained time-budget gate (pounce#242). The KKT
        // factorization (and any inertia-correction / quality-escalation
        // refactorizations) is the single most expensive step of a large
        // solve, and it has just finished. Check the deadline here so an
        // over-budget solve returns its current best iterate *before*
        // spending a line search and another whole iteration — bounding
        // the overshoot to roughly one search-direction computation
        // instead of a full outer iteration. `data.curr` is untouched by
        // the step computation (the trial lives in `data.trial`), so it
        // still holds the last accepted iterate.
        if let Some(ret) = self.deadline_status() {
            return IterateOutcome::Terminate(ret);
        }

        // 6. Acceptable trial point — run the line search if we have a
        //    primal/dual step on `data.delta`. Wrap in a guard so all
        //    early-return paths (ErrorInStepComputation, InternalError,
        //    restoration entry) still stop the timer.
        let _ls_guard = timing.compute_acceptable_trial_point.guard();
        let have_delta = self.data.borrow().delta.is_some();
        if have_delta {
            let delta = match self.data.borrow().delta.as_ref().cloned() {
                Some(d) => d,
                None => {
                    return IterateOutcome::Terminate(SolverReturn::ErrorInStepComputation);
                }
            };
            // Cap alpha by the primal fraction-to-the-boundary so the
            // first trial cannot push slacks past their bounds, and by
            // the dual FTB so bound multipliers stay positive. Mirrors
            // upstream `IpBacktrackingLineSearch::FindAcceptableTrialPoint`'s
            // calls to `IpCq.primal_frac_to_the_bound` /
            // `IpCq.dual_frac_to_the_bound` with τ = `curr_tau`.
            let tau = self.data.borrow().curr_tau;
            let alpha_p_max = self.cq.borrow().aff_step_alpha_primal_max(&delta, tau);
            let alpha_d_max = self.cq.borrow().aff_step_alpha_dual_max(&delta, tau);

            // Tiny-step gate — port of `IpBacktrackingLineSearch.cpp:363`
            // and the handling block at lines 382-435. When the search
            // direction is so small that any nonzero α would just
            // bounce inside floating-point noise, take the FTB step
            // unchecked and skip the line search; that's the only way
            // to hit `STOP_AT_TINY_STEP` cleanly when the iterate is
            // already at a converged point but `nlp_error > tol` due to
            // scaling or unbounded duals.
            // gh#884 — record the direction's scale-relative magnitude for
            // the dual-divergence-retry signature. Done here, where `delta`
            // is already in hand, rather than recomputed at the gate.
            self.last_step_rel = self.scale_relative_step_max(&delta);

            if self.detect_tiny_step(&delta) {
                let alpha_p = alpha_p_max;
                let alpha_d = alpha_d_max;
                let curr = match self.data.borrow().curr.clone() {
                    Some(c) => c,
                    None => return IterateOutcome::Terminate(SolverReturn::InternalError),
                };
                let trial_iv = scaled_step_unchecked(&curr, &delta, alpha_p, alpha_d);
                {
                    let mut d = self.data.borrow_mut();
                    d.set_trial(trial_iv);
                    d.info_alpha_primal = alpha_p;
                    d.info_alpha_dual = alpha_d;
                    d.info_ls_count = 0;
                    if self.tiny_step_last_iteration {
                        d.info_alpha_primal_char = 'T';
                        d.tiny_step_flag = true;
                    } else {
                        d.info_alpha_primal_char = 't';
                    }
                }
                let dy_amax = delta.y_c.amax().max(delta.y_d.amax());
                self.tiny_step_last_iteration = dy_amax < self.tiny_step_y_tol;
            } else {
                self.tiny_step_last_iteration = false;
                let alpha_init = self.alpha_init.min(alpha_p_max);
                let alpha_dual = self.alpha_init.min(alpha_d_max);
                let outcome = self.bundle.line_search.find_acceptable_trial_point(
                    &self.data,
                    &self.cq,
                    &delta,
                    alpha_init,
                    alpha_dual,
                    self.nlp.as_ref(),
                    self.search_dir.as_mut(),
                );
                match outcome {
                    Outcome::Accepted => {
                        // A normal LS-accepted step breaks any in-flight
                        // restoration cycle — clear the cycle detector
                        // so the next resto entry starts fresh.
                        self.last_resto_entry_x = None;
                        self.last_resto_entry_s = None;
                        self.last_resto_recovery_x = None;
                        self.last_resto_recovery_s = None;
                        self.resto_no_outer_progress_count = 0;
                        // Intentionally *not* clearing
                        // `resto_near_feasible_count` here: DECONVBNE's
                        // cycle interleaves R-recoveries with 2-3
                        // LS-accepted 'f'/'h' steps (which return
                        // `Outcome::Accepted` but accomplish no real
                        // outer progress — alpha drops to 1e-6 and
                        // inf_du remains pinned at 1.9e7), so resetting
                        // on every accept would zero the counter every
                        // cycle and never fire. The counter persists
                        // for the duration of the run and trips after
                        // 3 cumulative near-feasible entries; legitimate
                        // solves enter resto at most once at near-
                        // feasibility (POLAK6, HAIFAM) and stay under
                        // the limit.
                    }
                    Outcome::TinyStep | Outcome::Failed => {
                        // Debugger stop: the line search rejected the step
                        // (tiny-step floor or all backtracks failed), before
                        // we fall into restoration. Lets a "why did the line
                        // search give up?" inspection happen at the failing
                        // point distinctly from the restoration entry.
                        if let Some(o) = self.debug_stop(crate::debug::Checkpoint::StepRejected) {
                            return o;
                        }
                        // Upstream `IpBacktrackingLineSearch.cpp` raises
                        // `LINE_SEARCH_FAILED` when α drops below
                        // `alpha_min` or all retries reject, which in
                        // turn triggers `ActivateLineSearch` →
                        // restoration.
                        return self.invoke_restoration_debugged();
                    }
                    Outcome::Deadline => {
                        // The time budget was crossed inside the line
                        // search (pounce#242). No trial was promoted, so
                        // `data.curr` still holds the best iterate; stop
                        // with the matching time-limit status. Re-derive
                        // wall vs CPU from the deadline (it can only still
                        // be exceeded — time is monotonic).
                        return IterateOutcome::Terminate(
                            self.deadline_status()
                                .unwrap_or(SolverReturn::WallTimeExceeded),
                        );
                    }
                }
            }
        }

        // End the line-search/trial timer here so the bookkeeping in
        // steps 7-8 below is attributed to `accept_trial_point` (which
        // mirrors upstream's split: filter update and FTB reset are
        // accept-side, not line-search-side).
        _ls_guard.stop();

        // 7. Accept trial point (promotes `trial` to `curr` if set).
        //    The acceptor's filter has already been augmented (when
        //    appropriate) inside `find_acceptable_trial_point` via
        //    `update_for_next_iteration`, mirroring upstream's call
        //    chain in `IpBacktrackingLineSearch.cpp:839`.
        let _accept_guard = timing.accept_trial_point.guard();

        // 7a. Safe-slack bound adjustment. Before promoting `trial`, move
        //     any `x_L/x_U/d_L/d_U` whose trial slack fell below
        //     `eps*min(1,mu)` so the slack becomes representable (port of
        //     the bound-adjustment block in
        //     `IpoptAlgorithm::AcceptTrialPoint`, `IpIpoptAlg.cpp:664-706`).
        self.adjust_variable_bounds_for_small_slacks();

        self.data.borrow_mut().accept_trial_point();

        // 8. Bound multiplier kappa_sigma reset.
        self.correct_bound_multiplier();

        // 8b. `recalc_y` — re-estimate the equality/inequality
        //     multipliers by least squares once the iterate is feasible
        //     enough (`IpIpoptAlg.cpp:AcceptTrialPoint`). Off unless the
        //     user asks; see `application.rs` for why we do not turn it
        //     on for L-BFGS the way upstream's option text says it does.
        //
        //     Ordering: this runs after the kappa_sigma reset. The two
        //     are not obviously independent — the least-square RHS is
        //     `−∇f + Pₗz_L − Pᵤz_U` (`IpLeastSquareMults.cpp:54`), so it
        //     reads the bound multipliers step 8 just corrected — but
        //     running the sweep with the two swapped produces a
        //     byte-identical corpus, so the coupling does not bite in
        //     practice. Kept here on the argument that `y` should be
        //     estimated against the multipliers the iteration actually
        //     ends with.
        self.maybe_recalc_y();

        // 8c. Square-problem multipliers. `IpIpoptAlg.cpp:409` runs this
        //     between `AcceptTrialPoint` and the next `CheckConvergence`,
        //     on every iteration, for square problems only. pounce's
        //     `iterate()` boundary falls between those two — the outer
        //     loop bumps `iter_count` and the next `iterate()` opens with
        //     the convergence check — so this is the same slot, and the
        //     `+ 1` inside is that pending bump (upstream increments
        //     before the call, at `IpIpoptAlg.cpp:407`).
        if self.is_square_problem() {
            self.compute_feasibility_multipliers();
        }

        // Sub-iteration checkpoint: the trial point was accepted; α and
        // the new iterate are in place (before the loop's iter bookkeeping
        // and the next `IterStart`).
        drop(_accept_guard);
        if let Some(o) = self.debug_stop(crate::debug::Checkpoint::AfterStep) {
            return o;
        }

        IterateOutcome::Continue
    }

    /// `max(max_i |δx_i|/(1+|x_i|), max_i |δs_i|/(1+|s_i|))` — the same
    /// scale-relative measure [`Self::detect_tiny_step`] thresholds, kept
    /// as a magnitude.
    ///
    /// Scale-relative rather than a bare `‖d‖` on purpose: a bare norm is
    /// a length in the model's units, so on a badly scaled model it says
    /// more about the units than about whether the iterate has stopped
    /// moving. gh#884's signature needs the latter.
    fn scale_relative_step_max(&self, delta: &crate::iterates_vector::IteratesVector) -> Number {
        let curr = match self.data.borrow().curr.clone() {
            Some(c) => c,
            None => return Number::INFINITY,
        };

        let mut tmp = curr.x.make_new_copy();
        tmp.element_wise_abs();
        tmp.add_scalar(1.0);
        let mut tmp2 = delta.x.make_new_copy();
        tmp2.element_wise_divide(&*tmp);
        let mut worst = tmp2.amax();

        if curr.s.dim() > 0 {
            let mut tmp = curr.s.make_new_copy();
            tmp.element_wise_abs();
            tmp.add_scalar(1.0);
            let mut tmp2 = delta.s.make_new_copy();
            tmp2.element_wise_divide(&*tmp);
            worst = worst.max(tmp2.amax());
        }
        worst
    }

    /// Whether this solve ever saw gh#884's dual-divergence-at-a-settled-primal
    /// signature. Sticky; see [`Self::dual_divergence_signature`].
    pub fn dual_divergence_signature(&self) -> bool {
        self.dual_divergence_signature
    }

    /// Port of `IpBacktrackingLineSearch::DetectTinyStep`
    /// (`IpBacktrackingLineSearch.cpp:1219-1278`). Returns true iff
    /// `max_i |δx_i|/(1+|x_i|) ≤ tiny_step_tol`,
    /// `max_i |δs_i|/(1+|s_i|) ≤ tiny_step_tol`, AND
    /// `curr_constraint_violation ≤ 1e-4`. Disabled when
    /// `tiny_step_tol == 0`.
    fn detect_tiny_step(&self, delta: &crate::iterates_vector::IteratesVector) -> bool {
        if self.tiny_step_tol == 0.0 {
            return false;
        }
        let curr = match self.data.borrow().curr.clone() {
            Some(c) => c,
            None => return false,
        };

        // |x_i|+1
        let mut tmp = curr.x.make_new_copy();
        tmp.element_wise_abs();
        tmp.add_scalar(1.0);
        // |δx_i|/(|x_i|+1) ; checked via Amax of (δx ./ (|x|+1)).
        let mut tmp2 = delta.x.make_new_copy();
        tmp2.element_wise_divide(&*tmp);
        if tmp2.amax() > self.tiny_step_tol {
            return false;
        }

        if curr.s.dim() > 0 {
            let mut tmp = curr.s.make_new_copy();
            tmp.element_wise_abs();
            tmp.add_scalar(1.0);
            let mut tmp2 = delta.s.make_new_copy();
            tmp2.element_wise_divide(&*tmp);
            if tmp2.amax() > self.tiny_step_tol {
                return false;
            }
        }

        let cviol = self.cq.borrow().curr_constraint_violation();
        if cviol > 1e-4 {
            return false;
        }
        true
    }

    /// Re-anchor the quasi-Newton model instead of handing off to
    /// restoration, when the line search has failed at a point
    /// restoration cannot improve (gh#818). Returns `true` if the model
    /// was re-anchored, in which case the caller retries this iterate.
    ///
    /// **The two failures the line search cannot tell apart.** When no
    /// trial step is acceptable, either the *point* is bad — infeasible,
    /// and restoration is exactly the right tool — or the *direction* is,
    /// because `W` is a quasi-Newton model carrying curvature the iterate
    /// has left behind. Upstream has one answer for both, because
    /// restoration is the only fallback it has. At an already-feasible
    /// point that answer is a no-op: the restoration NLP minimizes the
    /// constraint violation, and there is none to minimize, so it wanders
    /// at `theta ~ 1e-13` and reports `Restoration_Failed`.
    ///
    /// Measured on the `deb7` fixture under `limited-memory`: the solve
    /// stalls with `inf_pr ~ 1e-12` and `inf_du ~ 1e5`, enters
    /// restoration at a point feasible to 8e-13, and spends 340 of its
    /// 1242 iterations there before failing. On the unconstrained
    /// gh#818 quadratic under `alpha_red_factor 0.8` it is starker
    /// still — `theta` is identically zero, so restoration cannot move
    /// at all, and the solve dies at **iteration 1** with
    /// `Error_In_Step_Computation` and the objective still at its
    /// starting value.
    ///
    /// So this is a rung, not a refusal: it fires only where restoration
    /// has nothing to reduce, it is bounded, and every path that reached
    /// restoration before still reaches it once the rung is spent.
    ///
    /// **Deliberately *after* the acceptable-point decline.** The call
    /// site is inside [`Self::invoke_restoration`], immediately behind
    /// that decline, and not at the `Outcome::Failed` arm in
    /// [`Self::iterate`] where the hand-off is decided — `eigena2` and
    /// `csfi2` reach the hand-off at feasible points that already pass
    /// the acceptable tolerances, and those must go on being reported
    /// rather than re-anchored and continued. Being inside
    /// `invoke_restoration` means the `PreRestoration` debug checkpoint
    /// fires ahead of a rung that then does not enter restoration; that
    /// is the price of the ordering and is the checkpoint's documented
    /// meaning ("just before entry"), not a promise that entry follows.
    ///
    /// **And deliberately not a feasibility gate on restoration itself.**
    /// That was tried and rejected before (see the `constr_viol_tol`
    /// paragraph in [`Self::invoke_restoration`]): feasible entries are
    /// ordinary, and nothing observable at the doorway separates a
    /// restoration that recovers from one that does not. This rung does
    /// not decide that question — it spends one cheap retry on the
    /// hypothesis that the model, not the point, is at fault, and hands
    /// over unchanged if the retry fails too.
    ///
    /// The bound is structural as well as counted. `reanchor` returns
    /// `false` once the history is down to its newest pair, so a second
    /// failure at the same iterate finds nothing to give up and falls
    /// through; refilling the history takes accepted steps, so the
    /// counter only advances once per genuine stall.
    /// `limited_memory_ls_failure_restarts` caps the total.
    fn try_reanchor_before_restoration(&mut self) -> bool {
        if self.lbfgs_ls_failure_restarts == 0
            || self.lbfgs_ls_restarts_used >= self.lbfgs_ls_failure_restarts
        {
            return false;
        }
        // Restoration's objective is the constraint violation. Only
        // stand in front of it where that objective is already at its
        // floor, so the rung can never pre-empt a restoration that had
        // real work to do. `constr_viol_tol` is the same tolerance the
        // convergence check calls feasible.
        let theta = self.cq.borrow().curr_constraint_violation();
        if !(theta <= self.bundle.conv_check.constr_viol_tol_or_default()) {
            return false;
        }
        if !self.bundle.hess.reanchor() {
            return false;
        }
        self.lbfgs_ls_restarts_used += 1;
        // 'Wa' alongside the updater's own 'Wr' (the
        // `limited_memory_max_skipping` reset), so the two re-anchorings
        // are distinguishable in the iteration table rather than both
        // reading as "the Hessian did something".
        self.data.borrow_mut().append_info_string("Wa");
        tracing::debug!(target: "pounce::algorithm",
            "[POUNCE] line search failed at a feasible point (theta {:.3e}); re-anchoring \
             the limited-memory Hessian on its newest curvature pair and retrying instead \
             of entering restoration, which has nothing to reduce here (gh#818). \
             Restart {} of {}.",
            theta, self.lbfgs_ls_restarts_used, self.lbfgs_ls_failure_restarts,
        );
        true
    }

    /// Drive the restoration phase after a line-search failure.
    /// Returns `IterateOutcome::Continue` if the restoration driver
    /// recovered (the algorithm carries on from the recovered iterate);
    /// otherwise terminates with [`SolverReturn::RestorationFailure`].
    /// Mirrors upstream's
    /// `IpBacktrackingLineSearch::ActivateLineSearch` → `PerformRestoration`
    /// chain.
    fn invoke_restoration(&mut self) -> IterateOutcome {
        // Snapshot the outer reference iterate's `(theta, barr)` and
        // build the orig-progress callback the inner IPM will consult
        // at every iteration (mirrors upstream
        // `IpRestoFilterConvCheck::SetOrigLSAcceptor` plus
        // `IpFilterLSAcceptor::Reset`'s `reference_*_` snapshot).
        let reference_theta = self.cq.borrow().curr_constraint_violation();
        let reference_barr = self.cq.borrow().curr_barrier_obj();

        if std::env::var("POUNCE_DBG_RESTO").is_ok() {
            let iter = self.data.borrow().iter_count;
            tracing::debug!(target: "pounce::algorithm",
                "RESTO_ENTRY iter={} theta={:.6e} barr={:.6e} near_feas_ct={}",
                iter, reference_theta, reference_barr, self.resto_near_feasible_count,
            );
        }

        // Port gap: upstream refuses to enter restoration from an acceptable
        // point, and this was missing. `IpBacktrackingLineSearch.cpp:557-570`,
        // in the `if (!accept)` arm that hands off to restoration:
        //
        //     if( CurrentIsAcceptable() )
        //     {
        //        THROW_EXCEPTION(ACCEPTABLE_POINT_REACHED,
        //                        "Restoration phase called at acceptable point.");
        //     }
        //
        // The rationale is the obvious one: restoration reduces the constraint
        // violation, so from a point that already passes the acceptable-level
        // tolerances it has nothing to reduce, and entering can only risk a
        // reportable solution.
        //
        // What the gap cost, measured on mittelmann `qcqp1000-1nc` (n=1000):
        // the line search fails at iteration 187 on a point carrying the
        // published optimum (`-2.6628866e+07`, matching ipopt-ma57 to 9
        // significant figures) with overall NLP error `6.0e-8` — two orders
        // inside `acceptable_tol`. Restoration walked it to `theta 5e-3` and
        // ground out 2780 further iterations without recovering, so a solved
        // problem reported a failure.
        //
        // The predicate is upstream's, unmodified: acceptability alone. A
        // strict `constr_viol_tol` gate was tried on top and is both a
        // deviation and useless — at their restoration entries `qcqp1000-1nc`
        // sits at `theta = 6.0e-8`, `csfi2` at `1.5e-7`, `eigena2` at
        // `2.1e-10`, all strictly feasible, one by six orders. Nothing
        // observable at the doorway separates a restoration that recovers from
        // one that does not, which is why upstream does not try to.
        //
        // Placed ahead of the cycle detectors below rather than beside
        // upstream's `PrepareRestoPhaseStart()`: those detectors are a
        // pounce-side addition, and an acceptable point should be reported
        // regardless of cycle state. Filter augmentation is skipped on this
        // path, which is immaterial — the run stops here.
        //
        // `current_is_acceptable_with_state` is the full triplet, never
        // `theta` alone: gh #274, a perfectly feasible point can be
        // arbitrarily far from stationary (`min -exp(x) s.t. x >= 0` reaches
        // here with `inf_pr = 1.7e-10` and `inf_du = 8.8e+47`), and the
        // triplet carries `acceptable_dual_inf_tol` to reject it. The
        // finiteness check mirrors the one below (CUTE `himmelbj` reaches a
        // near-feasible point where `f` evaluates to NaN) and matches
        // upstream's own `curr_f` precondition for acceptability.
        let (entry_f_finite, entry_nlp_err) = {
            let cq = self.cq.borrow();
            (cq.curr_f().is_finite(), cq.curr_nlp_error())
        };
        //
        // What the guard still did not ask is whether the solve was *converging*
        // (gh #534). It reads the entry point and nothing about the trajectory
        // that reached it, so it stops a contracting endgame and a dead stall
        // with equal confidence. On `eigena2` it fires while the dual
        // infeasibility is quartering every iteration on unit steps
        // (`1.19e-5 → 2.96e-6 → 7.38e-7 → 1.84e-7`), three iterations short of a
        // strict certificate that costs nothing but those three iterations.
        // [`Self::may_defer_acceptable_decline`] adds that missing question,
        // and only that: when the answer is no — `eigenb2`'s tail rises, and
        // `csfi2`'s last two iterations are flat to three digits — the guard
        // fires exactly as before.
        if entry_f_finite
            && self.bundle.conv_check.current_is_acceptable_with_state(
                entry_nlp_err,
                &self.data,
                &self.cq,
            )
        {
            if self.may_defer_acceptable_decline() {
                tracing::debug!(target: "pounce::algorithm",
                    "[POUNCE] deferring the restoration decline at theta {:.3e}: the entry \
                     point passes the acceptable-level tolerances (nlp_err {:.3e}) but the \
                     NLP error has contracted every iteration over the last {} \
                     ({:.3e} -> {:.3e}); continuing for up to {} iterations, with that point \
                     held as the floor (gh #534).",
                    reference_theta, entry_nlp_err, DECLINE_PROGRESS_SAMPLES - 1,
                    self.nlp_err_recent[0], entry_nlp_err, DECLINE_CONTINUATION_BUDGET,
                );
            } else {
                // The window is on the line because "why did the guard not
                // defer?" is the first question anyone reading this trace has
                // (gh #534), and reconstructing it from the iteration table
                // means recomputing the scaled aggregate by hand.
                tracing::debug!(target: "pounce::algorithm",
                    "[POUNCE] declining restoration at theta {:.3e}: the entry point already \
                     passes the acceptable-level tolerances (nlp_err {:.3e}); reporting it \
                     rather than risking it in restoration. Recent NLP errors {} \
                     (contracting: {}).",
                    reference_theta, entry_nlp_err,
                    self.nlp_err_window_str(), self.nlp_err_contracting(),
                );
                return IterateOutcome::Terminate(SolverReturn::StopAtAcceptablePoint);
            }
        }

        // gh#818 — one rung before the hand-off proper: re-anchor the
        // quasi-Newton model and retry this iterate. Placed *here*, and
        // not at the `Outcome::Failed` arm in `iterate`, because it has
        // to run behind the acceptable-point decline above: `eigena2`
        // and `csfi2` arrive at feasible points that already pass the
        // acceptable tolerances, and those must go on being reported
        // rather than re-anchored and continued. The gh#534 deferral
        // path falls through to here, which is the right order too —
        // the deferral has already captured its floor, so a rung taken
        // under a live deferral is protected by it.
        if self.try_reanchor_before_restoration() {
            return IterateOutcome::Continue;
        }

        // No-progress restoration cycle detector. Two layered checks
        // surface as `ErrorInStepComputation` instead of cycling to
        // `max_iter` exhaustion (mirrors the *intent* of upstream
        // `IpBacktrackingLineSearch.cpp:580-600`'s almost-feasible
        // resto guard):
        //
        // 1. *Static cycle*: entry-to-entry — when the curr `(x, s)`
        //    at this entry is essentially identical to the snapshot
        //    from the previous entry, the inner resto-IPM is
        //    returning recovered iterates indistinguishable from
        //    entry, AND the outer didn't move either. Fires
        //    immediately. Catches QCNEW, EQC, MESH, POLAK6, S365,
        //    S365MOD, SIPOW2M, PFIT4.
        //
        // 2. *Slow-progress cycle*: recovery-to-entry — when curr at
        //    this entry is essentially identical to the *recovery*
        //    iterate from the previous resto, the outer made no
        //    progress between resto invocations even though resto's
        //    inner moved substantively. Counted, fires after 5
        //    consecutive entries. Catches ACOPR14, ACOPR30, TRO3X3
        //    while letting MAKELA3, HAIFAM, HALDMADS, ROBOT,
        //    TENBARS2 — which need 2-3 productive resto entries
        //    before LS accepts — pass through.
        //
        // A productive single-restoration sequence (BT8, HIMMELBJ,
        // LINSPANH, LSNNODOC, ODFITS, OET3) clears both snapshots via
        // `Outcome::Accepted` between entries and is unaffected.
        let curr = self
            .data
            .borrow()
            .curr
            .as_ref()
            .expect("curr set before invoke_restoration")
            .clone();
        // Helper: when the cycle detector fires and the orig cv is a
        // violation the *user* calls a violation (e.g. PFIT1's 2.73e-2),
        // the outer is stuck at a feasibility-stationary point and the
        // honest exit is `LocalInfeasibility`. Below that threshold the
        // iterate is primal-feasible by the user's own declaration, so there
        // is no infeasibility to certify — the failure is numerical, not
        // algorithmic, and `ErrorInStepComputation` is retained.
        //
        // The threshold is `constr_viol_tol`, and *only* `constr_viol_tol`
        // (gh #508). The question this ternary asks — "is this violation
        // real?" — is a question about the constraint violation, so it has to
        // be asked with the option that declares what a violated constraint
        // is. The previous form, `max(100·tol, 1e-4)`, was built from `tol`, a
        // tolerance on the **KKT error**: different quantity, different units,
        // and it never consulted `constr_viol_tol` at all. Two consequences,
        // both measured on `min (x-5)² s.t. x²+δ = 0` (infeasible for every
        // δ>0, reported violation exactly δ):
        //
        //   * sweeping `constr_viol_tol` over four orders moved the boundary
        //     not at all — at `constr_viol_tol = 1e-3` a violation of `1e-4`,
        //     comfortably inside the user's declared feasibility tolerance,
        //     still exited 500;
        //   * sweeping `tol` moved it a great deal, and in the wrong
        //     direction: at `tol = 1e-4` the `1e-2` threshold swallowed every
        //     δ from `3e-4` to `1e-2` — a model infeasible by a full percent
        //     answered "your solver broke". Loosening `tol` is the standard
        //     user reaction to a struggling solve, so the failure widened
        //     exactly when the user tried to help.
        //
        // No `infeas_viol_kappa` margin on top, unlike the rapid-infeasibility
        // pre-filter in `conv_check`. That detector fires *during* the solve
        // off a streak heuristic and needs the margin to avoid convicting an
        // iterate that is still converging; here restoration has already
        // demonstrably cycled, so the certainty comes from the cycle evidence
        // rather than from extra violation headroom. Widening to
        // `kappa·constr_viol_tol` would move the default threshold from `1e-4`
        // to `1e-2` and hand back 500 on the whole band in between.
        //
        // The comparison is `>=`, not `>`. A violation landing exactly on the
        // threshold is a violation at the user's declared tolerance, and the
        // reproducer above hits the boundary to the digit (`δ = 1e-4` at the
        // default `constr_viol_tol`), where `>` returned 500 for a model
        // infeasible by precisely the amount the user said was too much.
        //
        // The violation is measured **unscaled**. `reference_theta` is the
        // row-scaled residual, but the floor below is an absolute, user-facing
        // magnitude, so comparing the two mixes unit systems — and on a problem
        // whose rows are scaled down the scaled residual can never clear it.
        // `infeasible_equalities.nl` is the worked example: a square 2x2 system
        // with a true violation of 2.0 that NLP scaling reports as 6.67e-7, so
        // this test read `6.67e-7 > 1e-4` = false and a blatantly infeasible
        // model exited `Error_In_Step_Computation` (AMPL 500, Pyomo
        // `internalSolverError`). Square problems have no restoration-side
        // locally-infeasible gate — `strict` carves them out so the outer gets
        // another shot — so this cycle exit *is* their safety net, and it was
        // disabled by the unit mismatch. Same user-visible family as gh #372.
        //
        // Note this also moves from a 1-norm (`curr_constraint_violation`) to a
        // max-norm. Max-norm <= 1-norm, so the test is marginally stricter
        // about declaring infeasibility on an unscaled problem — the safe
        // direction for a verdict this consequential.
        //
        // `theta > 0` in front of the `>=` is not redundant: the options layer
        // registers `constr_viol_tol` with a *strict* lower bound of zero, but
        // a library embedder setting `ConvCheckOptions` directly is not bound
        // by that, and `0 >= 0` would turn an exactly-feasible iterate into an
        // infeasibility certificate. A zero violation never proves anything.
        let cycle_viol_tol = self.bundle.conv_check.constr_viol_tol_or_default();
        let reference_theta_unscaled = self.cq.borrow().curr_unscaled_primal_infeasibility_max();
        let cycle_exit =
            if reference_theta_unscaled > 0.0 && reference_theta_unscaled >= cycle_viol_tol {
                SolverReturn::LocalInfeasibility
            } else {
                SolverReturn::ErrorInStepComputation
            };
        let static_cycle = if let (Some(prev_x), Some(prev_s)) = (
            self.last_resto_entry_x.as_ref(),
            self.last_resto_entry_s.as_ref(),
        ) {
            let dx_rel = relative_distance(&*curr.x, &**prev_x);
            let ds_rel = relative_distance(&*curr.s, &**prev_s);
            if std::env::var_os("POUNCE_DBG_RESTO_CYCLE").is_some() {
                tracing::debug!(target: "pounce::algorithm",
                    "[PN_RESTO_CYCLE] entry-vs-entry dx_rel={:.6e} ds_rel={:.6e}",
                    dx_rel, ds_rel
                );
            }
            dx_rel <= 1e-10 && ds_rel <= 1e-10
        } else {
            false
        };
        if static_cycle {
            // Prefer the last acceptable point over the cycle error —
            // the borrows above are released, so the `&mut self` helper
            // is free to roll back.
            return self.terminate_acceptable_or(cycle_exit);
        }
        let recovery_cycle = if let (Some(prev_x), Some(prev_s)) = (
            self.last_resto_recovery_x.as_ref(),
            self.last_resto_recovery_s.as_ref(),
        ) {
            let dx_rel = relative_distance(&*curr.x, &**prev_x);
            let ds_rel = relative_distance(&*curr.s, &**prev_s);
            if std::env::var_os("POUNCE_DBG_RESTO_CYCLE").is_some() {
                tracing::debug!(target: "pounce::algorithm",
                    "[PN_RESTO_CYCLE] entry-vs-recovery dx_rel={:.6e} ds_rel={:.6e} count={}",
                    dx_rel, ds_rel, self.resto_no_outer_progress_count
                );
            }
            dx_rel <= 1e-10 && ds_rel <= 1e-10
        } else {
            false
        };
        if recovery_cycle {
            self.resto_no_outer_progress_count =
                self.resto_no_outer_progress_count.saturating_add(1);
            // 10-strike limit: tuned to give OET7-style traces room
            // to break through (inner inf_pr still decreasing across
            // strikes) while still bounding DECONVBNE-style cycles
            // (which need a guard but tolerate a wider window —
            // ~3 outer steps per cycle, so 10 strikes ≈ 30 outer
            // iters, well below the 2987-iter pathological run).
            if self.resto_no_outer_progress_count >= 10 {
                // Prefer the last acceptable point over the cycle error;
                // borrows are released, so the `&mut self` helper is free.
                return self.terminate_acceptable_or(cycle_exit);
            }
        } else {
            self.resto_no_outer_progress_count = 0;
        }
        // Near-feasible resto re-entry detector — matches the *intent*
        // of upstream `IpBacktrackingLineSearch.cpp:580-600`'s almost-
        // feasible-resto guard with a looser cv threshold. When the
        // outer enters restoration with the constraint violation
        // already at or below `tol`, the resto sub-IPM will produce a
        // recovered iterate that's at most marginally more feasible,
        // and any post-recovery σ-blowup from the next outer KKT solve
        // will re-trigger resto on the next iteration. Counting these
        // entries surfaces the cycle as `StopAtAcceptablePoint` —
        // primal feasibility is already met, only the dual residual
        // remains. Catches DECONVBNE: pounce ran 2987 iters before
        // this guard (cycle of ~30-inner-resto + 3 outer per cycle);
        // upstream solves in 505 iters via a different x trajectory.
        // Single-entry productive restos (BT8, HIMMELBJ, ODFITS) and
        // sub-tol-but-recoverable starts pass through under the 3-
        // strike limit.
        let outer_tol = self.bundle.conv_check.tol_or_default();
        if reference_theta <= outer_tol {
            self.resto_near_feasible_count = self.resto_near_feasible_count.saturating_add(1);
            if self.resto_near_feasible_count >= 3 {
                // Constraint feasibility is met, but a near-feasible iterate is
                // only "acceptable" if its objective is finite. CUTE `himmelbj`
                // reaches a point with cv ≈ 2e-9 where f evaluates to NaN; that
                // must surface as Invalid_Number_Detected rather than be
                // reported as Solved_To_Acceptable_Level with a `nan` objective.
                if !self.cq.borrow().curr_f().is_finite() {
                    return IterateOutcome::Terminate(SolverReturn::InvalidNumberDetected);
                }
                // Constraint feasibility alone does not make a point
                // acceptable. `reference_theta` measures only the *primal*
                // residual, so a perfectly feasible iterate can still be
                // arbitrarily far from stationary — which is exactly what an
                // unbounded objective looks like from here: the constraints
                // stay satisfied while the iterates run off toward -inf.
                //
                // `min -exp(x) s.t. x >= 0` re-enters restoration with
                // `inf_pr = 1.7e-10` and `inf_du = 8.8e+47`; before gh #274
                // the finiteness check was the only gate, `-8.8e47` is
                // finite, and the solve was reported as
                // `Solved_To_Acceptable_Level` — which Pyomo maps into the
                // *solved* family, loading the diverging iterate as an
                // optimal solution.
                //
                // So require the point to pass the full acceptable-level
                // triplet (which includes `acceptable_dual_inf_tol`) before
                // claiming acceptability. When it does not, surface
                // `cycle_exit` — the same honest status the other two
                // restoration-cycle exits in this function use.
                let nlp_err = self.cq.borrow().curr_nlp_error();
                if !self
                    .bundle
                    .conv_check
                    .current_is_acceptable_with_state(nlp_err, &self.data, &self.cq)
                {
                    tracing::debug!(target: "pounce::algorithm",
                        "[POUNCE] near-feasible restoration re-entry at theta {:.3e} \
                         but the point fails the acceptable-level tolerances \
                         (nlp_err {:.3e}); reporting {:?} rather than \
                         Solved_To_Acceptable_Level (gh#274).",
                        reference_theta, nlp_err, cycle_exit,
                    );
                    return IterateOutcome::Terminate(cycle_exit);
                }
                return IterateOutcome::Terminate(SolverReturn::StopAtAcceptablePoint);
            }
        } else {
            self.resto_near_feasible_count = 0;
        }
        self.last_resto_entry_x = Some(curr.x.make_new_copy());
        self.last_resto_entry_s = Some(curr.s.make_new_copy());

        // Augment the outer's filter with the resto-entry envelope —
        // mirrors upstream `IpBacktrackingLineSearch.cpp:566`:
        // `acceptor_->PrepareRestoPhaseStart()`. Adds
        // `((1-γ_θ)·θ_entry, φ_entry - γ_φ·θ_entry)` to the filter so
        // that after restoration recovers, the outer's Newton step is
        // forced by the filter to make real progress vs the entry
        // point. Without this, the outer accepts null-progress 'h'
        // steps and re-enters restoration on the next iteration (root
        // cause of DECONVBNE's 323 R-accepts vs ipopt's 21).
        self.bundle
            .line_search
            .acceptor_mut()
            .prepare_resto_phase_start(reference_theta, reference_barr);

        let orig_progress_cb = self.bundle.line_search.acceptor().make_orig_progress_check(
            reference_theta,
            reference_barr,
            5.0,
        );

        let (Some(nlp), Some(sd), Some(resto)) = (
            self.nlp.as_ref(),
            self.search_dir.as_mut(),
            self.restoration.as_mut(),
        ) else {
            return IterateOutcome::Terminate(SolverReturn::RestorationFailure);
        };
        resto.set_orig_progress_check(orig_progress_cb);
        // Forward the shared debugger so it can step the inner solve.
        resto.set_debug_hook(self.debug.as_ref().map(Rc::clone));
        // Forward the user's TNLP so the callback fires from the inner
        // solve too (gh#645). `None` when the caller installed no
        // callback, which keeps the whole path inert for them.
        resto.set_intermediate_tnlp(self.tnlp.as_ref().map(Rc::clone));
        let mut pd_guard = sd.pd_solver_mut();
        let aug = pd_guard.aug_solver_mut();
        // Audit counters (pounce#12). Increment call count + outer-iter
        // count (one outer iter is consumed per restoration call) and
        // wall-time around the inner call. Inner iter count is read
        // after via the trait accessor.
        //
        // `outer_iter_at_entry` is captured *before* the call because the
        // inner IPM's counter is seeded from it (`inner.iter_count =
        // outer_iter + 1`, upstream `IpRestoMinC_1Nrm.cpp:181`). The
        // accessor hands back that seeded, absolute number; the sub-solve's
        // own length is the difference. Adding the raw accessor value was
        // gh #819's second defect — `restoration_inner_iters` was a sum of
        // absolute positions, a quantity with no meaning — and it is the
        // same misreading gh#664 records for the stall gate.
        let outer_iter_at_entry = self.data.borrow().iter_count;
        self.resto_calls = self.resto_calls.saturating_add(1);
        self.resto_outer_iters = self.resto_outer_iters.saturating_add(1);
        let resto_t0 = std::time::Instant::now();
        let outcome = resto.perform_restoration(&self.data, &self.cq, nlp, aug);
        drop(pd_guard);
        self.resto_wall_secs += resto_t0.elapsed().as_secs_f64();
        let inner_final_iter = resto.last_inner_iter_count();
        self.resto_inner_iters = self
            .resto_inner_iters
            .saturating_add((inner_final_iter - outer_iter_at_entry).max(0));
        // gh #819. Roll the reported iteration count forward over the
        // restoration rows on the paths that *terminate* the solve.
        //
        // `RestorationOutcome::Recovered` already does this for itself, in
        // `min_c_1nrm.rs`'s step 2g (`Set_iter_count(resto_iter_count - 1)`,
        // one short because the outer loop is about to increment). Every
        // other outcome returns `Terminate` from the match below without
        // passing through that block, so the whole sub-solve used to vanish
        // from the summary: on gh #815's flowsheet the log ends at row
        // `3000r`, above a summary that said `Number of Iterations....: 3`.
        //
        // Ipopt reports the index of the last row it printed, `r` rows
        // included, on every exit path. Measured on the gh #815 model and
        // three variants of it: last row `2418r` / reported 2418, `412r` /
        // 412, `1547r` / 1547, `1348r` / 1348 — exits `Restoration Failed`,
        // local infeasibility and `Maximum Number of Iterations Exceeded`
        // respectively. Assigning the absolute inner count reproduces that
        // rule exactly.
        //
        // Safe against trajectory: every non-`Recovered` arm below returns
        // `IterateOutcome::Terminate`, and `optimize_inner`'s loop breaks on
        // `Terminate` without consulting the counter again. Nothing reads
        // `iter_count` between here and the summary.
        if !matches!(outcome, RestorationOutcome::Recovered)
            && inner_final_iter > outer_iter_at_entry
        {
            self.data.borrow_mut().iter_count = inner_final_iter;
        }
        // pounce#244: the restoration inner IPM shares the outer solve's
        // `Deadline` (both its convergence check and — post-#244 — its KKT
        // solves consult it), so a budget crossing inside restoration
        // terminates the inner solve with a time-limit status. Surface that
        // as the time limit directly instead of letting the `Failed` arm map
        // it onto `RestorationFailure` / `StopAtAcceptablePoint`. `data.curr`
        // is the last accepted outer iterate — restoration stages its
        // recovered point onto `trial`, not `curr`, and we return before
        // promoting it — so this hands back a valid iterate.
        if let Some(ret) = self.deadline_status() {
            return IterateOutcome::Terminate(ret);
        }
        match outcome {
            RestorationOutcome::Recovered => {
                // Mirror upstream `IpBacktrackingLineSearch.cpp:624-631`:
                // a successful restoration clears the line search's
                // cross-iteration globalization counters. Upstream runs
                // restoration inside `FindAcceptableTrialPoint` so those
                // assignments are inline; pounce runs it here, so the
                // reset has to be driven from here. Without it
                // `watchdog_shortened_iter` survives a restoration
                // episode and runs of shortened steps on either side of
                // one accumulate as if consecutive, arming the watchdog
                // where upstream would not. See
                // `BacktrackingLineSearch::reset_after_restoration`.
                self.bundle.line_search.reset_after_restoration();
                // The driver has staged the recovered point on
                // `data.trial`; apply the safe-slack bound adjustment
                // (as the main accept path does), then promote it and
                // continue iterating.
                self.adjust_variable_bounds_for_small_slacks();
                self.data.borrow_mut().accept_trial_point();
                // Snapshot the recovery iterate for the slow-cycle
                // detector at the top of the next `invoke_restoration`.
                // Compared against next-entry curr, dx_rel ≈ ‖α·d‖ —
                // measures purely the outer step. See header comment
                // on the cycle detector above.
                let recovered = self
                    .data
                    .borrow()
                    .curr
                    .as_ref()
                    .expect("accept_trial_point sets curr")
                    .clone();
                self.last_resto_recovery_x = Some(recovered.x.make_new_copy());
                self.last_resto_recovery_s = Some(recovered.s.make_new_copy());
                // Mirror upstream `IpoptAlgorithm::AcceptTrialPoint`
                // (`IpIpoptAlg.cpp:917-963`): kappa_sigma clamp on the
                // four bound-multiplier vectors. Upstream applies this
                // unconditionally inside AcceptTrialPoint, so the
                // post-restoration path inherits it; pounce factored
                // the clamp out of the data swap so we must call it
                // explicitly here. Without it the all-1 multiplier
                // reset (`bound_mult_reset_threshold`) leaves z*s far
                // from mu at the recovered iterate, blowing up the
                // next KKT solve's σ = z/s diagonal.
                self.correct_bound_multiplier();
                IterateOutcome::Continue
            }
            RestorationOutcome::Failed => {
                // Mirrors upstream `IpBacktrackingLineSearch.cpp:611-623`:
                // when `PerformRestoration` returns false, attempt to
                // roll back to the most recent acceptable iterate before
                // surfacing failure. If a snapshot is available we exit
                // cleanly with `StopAtAcceptablePoint` (mapped by the
                // application layer to `Solved_To_Acceptable_Level`),
                // matching the upstream `ACCEPTABLE_POINT_REACHED`
                // throw. Without a snapshot we surface
                // `RestorationFailure` — unless the restoration left the
                // iterate diverging (`|x|_∞ > diverging_iterates_tol`), in
                // which case we surface `DivergingIterates` to mirror the
                // outcome upstream produces on pathological problems like
                // MESH (where ipopt reports `Diverging_Iterates` and
                // pounce previously reported `Restoration_Failed` with an
                // obj of −3.6e+33). As in the running guard above, a large
                // `|x|` is only reported as unbounded when it is
                // structurally consistent with an unbounded feasible region
                // and the divergence is genuine — either it has persisted
                // (the running guard's growth-and-descent streak, which only
                // accumulates on a real recession ray; issues #248 / #252) or
                // blown past the absolute runaway backstop; otherwise the
                // failure is a plain `RestorationFailure`, never a spurious
                // `Unbounded`.
                if self.restore_acceptable_point() {
                    IterateOutcome::Terminate(SolverReturn::StopAtAcceptablePoint)
                } else {
                    let diverging = {
                        let data = self.data.borrow();
                        match data.curr.as_ref() {
                            Some(curr) => {
                                let amax = curr.x.amax();
                                amax > self.diverging_iterates_tol
                                    && self.divergence_is_true_unboundedness(&*curr.x)
                                    && (amax >= Self::DIVERGENCE_ABS_RUNAWAY
                                        || self.divergence_streak >= Self::DIVERGENCE_PERSIST_ITERS)
                            }
                            None => false,
                        }
                    };
                    if diverging {
                        IterateOutcome::Terminate(SolverReturn::DivergingIterates)
                    } else {
                        IterateOutcome::Terminate(SolverReturn::RestorationFailure)
                    }
                }
            }
            RestorationOutcome::UserRequestedStop => {
                // gh#645. Same discipline as the pounce#244 deadline
                // exit a few lines up, and for the same reason: the
                // recovered point is staged on `data.trial` and we
                // return without promoting it, so `data.curr` is still
                // the last iterate accepted for the *original* NLP.
                // That matters more than the status code to the caller
                // this exists for — a controller that aborts on a
                // deadline still has to apply something, and the
                // subproblem's iterate is not a point it should apply.
                IterateOutcome::Terminate(SolverReturn::UserRequestedStop)
            }
            RestorationOutcome::FeasiblePointFound => {
                // Port of `IpIpoptAlg.cpp:542` — the catch of
                // `FEASIBILITY_PROBLEM_SOLVED`, thrown by
                // `IpRestoMinC_1Nrm.cpp:269` when restoration reaches a
                // point feasible for a *square* original NLP. Upstream
                // recomputes the multipliers before returning
                // `FEASIBLE_POINT_FOUND`; without that step the reported
                // dual infeasibility is `∇f` at a point whose status says
                // the constraints are satisfied. On the gh#508 probe that
                // is the difference between Ipopt's `1.78e-15` and a bare
                // `10.0`.
                //
                // The driver has already promoted the recovered point to
                // `data.curr`, so the multipliers are computed at the
                // point that will be reported.
                if self.is_square_problem() {
                    self.compute_feasibility_multipliers_postprocess();
                }
                IterateOutcome::Terminate(SolverReturn::FeasiblePointFound)
            }
            RestorationOutcome::LocallyInfeasible => {
                // Mirrors upstream's catch of `LOCALLY_INFEASIBLE` thrown
                // from `IpRestoConvCheck.cpp:240` — the resto sub-IPM
                // settled at a stationary point of `||c(x)||_1` whose
                // residual is still well above `tol`. Without this
                // detection the outer would re-enter restoration on the
                // unchanged iterate forever.
                //
                // gh #505: consult the acceptable-point stash, for the same
                // reason the conv-check arm above does and the cycle exits
                // already did. This is the *third* site that produced
                // `LocalInfeasibility`, and the only one not gated on
                // `infeas_max_streak` — which matters, because on the reported
                // instance raising that knob to 15 did not move the run by a
                // single iteration, so the verdict there is not the outer
                // detector's. Whichever route reaches it, a solve that passed
                // through an acceptable iterate must not discard it.
                //
                // Inert on genuinely infeasible models by the same argument as
                // the other two: nothing is stashed unless the whole acceptable
                // triplet passed, so `terminate_acceptable_or` falls through to
                // the verdict unchanged.
                self.terminate_local_infeasibility()
            }
        }
    }

    /// Safe-slack bound adjustment, applied to the staged `trial`
    /// iterate before it is promoted to `curr`. When one or more trial
    /// slacks fell below `eps*min(1,mu)`, [`IpoptCalculatedQuantities::
    /// adjusted_trial_bounds`] returns the moved `x_L/x_U/d_L/d_U`; we
    /// install them on the NLP so the slack becomes representable. Port
    /// of the bound-adjustment block in `IpoptAlgorithm::AcceptTrialPoint`
    /// (`IpIpoptAlg.cpp:664-706`).
    fn adjust_variable_bounds_for_small_slacks(&mut self) {
        // Compute the moved bounds (releases the CQ/NLP borrows on return).
        let adjusted = {
            let trial_set = self.data.borrow().trial.is_some();
            if !trial_set {
                return;
            }
            self.cq.borrow().adjusted_trial_bounds()
        };
        let Some(bounds) = adjusted else {
            return;
        };
        tracing::debug!(
            target: "pounce::algorithm",
            "slack_move: {} slack(s) too small, adjusting variable bound(s) at iter {}",
            bounds.adjusted,
            self.data.borrow().iter_count,
        );
        let nlp = Rc::clone(self.cq.borrow().nlp());
        nlp.borrow_mut().adjust_variable_bounds(
            &*bounds.x_l,
            &*bounds.x_u,
            &*bounds.d_l,
            &*bounds.d_u,
        );
    }

    /// Refresh the `s`-block factors for
    /// `linear_system_scaling=slack-based`.
    ///
    /// A no-op for every other scaling choice: the flag is set only when
    /// the builder installed a `SlackBasedTSymScalingMethod`, and the
    /// method itself ignores the push if it never receives one.
    ///
    /// Silently skips when the quantity cannot be formed — no NLP, no
    /// current iterate, no inequality rows, or a primal vector shape the
    /// CQ does not recognise. The scaling method then keeps behaving as
    /// identity, which is what `linear_system_scaling=none` would have
    /// done, so a missing push costs conditioning and never correctness.
    fn push_slack_scaling(&mut self) {
        if !self.slack_based_scaling {
            return;
        }
        if self.nlp.is_none() {
            return;
        }
        let nx = match self.data.borrow().curr.as_ref() {
            Some(c) => c.x.dim(),
            None => return,
        };
        let Some(s_scale) = self.cq.borrow().curr_slack_based_s_scaling() else {
            return;
        };
        if s_scale.is_empty() {
            return;
        }
        if let Some(sd) = self.search_dir.as_mut() {
            sd.pd_solver_mut()
                .aug_solver_mut()
                .set_slack_scaling(nx, &s_scale);
        }
    }

    /// `recalc_y` — replace `y_c`/`y_d` with least-square estimates once
    /// the iterate is feasible enough. Port of the `recalc_y_` block in
    /// `IpIpoptAlg.cpp:AcceptTrialPoint`.
    ///
    /// Silently does nothing — leaving the Newton-step multipliers in
    /// place — when disabled, when the violation is still above
    /// `recalc_y_feas_tol`, when there is nothing to estimate, or when
    /// the augmented-system solve fails. A failed estimate is not an
    /// error: the multipliers we already have are valid, just less
    /// accurate, so falling back to them costs accuracy and never
    /// correctness. Same reasoning as the initializer's `y0` fallback in
    /// `init/default.rs`.
    fn maybe_recalc_y(&mut self) {
        if !self.recalc_y {
            return;
        }
        let Some(nlp) = self.nlp.as_ref().map(Rc::clone) else {
            return;
        };
        // Feasibility gate. Upstream compares against the same
        // `curr_constraint_violation` the convergence check uses.
        if self.cq.borrow().curr_constraint_violation() >= self.recalc_y_feas_tol {
            return;
        }
        let (n_yc, n_yd) = {
            let d = self.data.borrow();
            match d.curr.as_ref() {
                Some(c) => (c.y_c.dim(), c.y_d.dim()),
                None => return,
            }
        };
        if n_yc + n_yd == 0 {
            return;
        }
        // The augmented-system solver is owned by the search-direction
        // calculator, as it is for the initializer's least-square call.
        let Some(sd) = self.search_dir.as_mut() else {
            return;
        };
        let mut new_y_c = pounce_linalg::dense_vector::DenseVectorSpace::new(n_yc).make_new_dense();
        let mut new_y_d = pounce_linalg::dense_vector::DenseVectorSpace::new(n_yd).make_new_dense();
        let mut pd_guard = sd.pd_solver_mut();
        // This was the first call site to drop review item M3's 1e-8
        // perturbation (#688), on the argument that `recalc_y` overwrites
        // `y` every iteration — so a bias in the estimator is a *fixed
        // point* rather than a transient, nothing downstream corrects it,
        // and it lands directly in `inf_du`, the quantity the run is
        // judged on. gh#693 extended the same treatment to the other
        // three sites, so `calculate_y_eq` no longer takes a flag: every
        // caller now gets δ=0 with the perturbed solve as a retry.
        let ok = self.bundle.eq_mult.calculate_y_eq(
            &self.data,
            &self.cq,
            &nlp,
            pd_guard.aug_solver_mut(),
            &mut new_y_c,
            &mut new_y_d,
        );
        drop(pd_guard);
        if !ok {
            tracing::debug!(
                target: "pounce::algorithm",
                "recalc_y: least-square solve failed at iter {}, keeping Newton multipliers",
                self.data.borrow().iter_count,
            );
            return;
        }
        // Mark the iteration, exactly as upstream does
        // (`IpData().Append_info_string("y ")` in
        // `IpIpoptAlg.cpp:AcceptTrialPoint`). Without it the iteration
        // log gives no way to tell which iterations re-estimated `y`
        // and which carried the Newton multipliers — and when a solve
        // stalls with an oscillating `inf_du`, whether the oscillation
        // tracks the re-estimation is the first thing worth knowing.
        self.data.borrow_mut().append_info_string("y ");

        // Share x/s/z/v; swap only the equality/inequality multipliers.
        let curr = match self.data.borrow().curr.clone() {
            Some(c) => c,
            None => return,
        };
        let new_iv = crate::iterates_vector::IteratesVector::new(
            curr.x.clone(),
            curr.s.clone(),
            Rc::new(new_y_c),
            Rc::new(new_y_d),
            curr.z_l.clone(),
            curr.z_u.clone(),
            curr.v_l.clone(),
            curr.v_u.clone(),
        );
        self.data.borrow_mut().set_curr(new_iv);
    }

    /// Port of `IpoptCalculatedQuantities::IsSquareProblem`
    /// (`IpIpoptCalculatedQuantities.cpp:3732`): as many equality
    /// constraints as variables, so the NLP has zero degrees of freedom.
    /// There is nothing to optimise — only a system to solve — and the
    /// objective is decorative.
    ///
    /// The consequence that matters is algebraic. `J_c` is square, so the
    /// least-square multiplier system `J_cᵀ y = −∇f (+ bound terms)` is
    /// exactly solvable and the dual residual can always be driven to
    /// zero, however large `y` has to be. On a non-square problem it
    /// generally cannot, which is why this is the right gate and not a
    /// heuristic.
    fn is_square_problem(&self) -> bool {
        match self.data.borrow().curr.as_ref() {
            Some(c) => c.x.dim() == c.y_c.dim(),
            None => false,
        }
    }

    /// Zero the four bound multipliers and replace `y_c`/`y_d` with the
    /// least-square multipliers of the resulting feasibility problem —
    /// the shared body of `ComputeFeasibilityMultipliers`
    /// (`IpIpoptAlg.cpp:893-922`) and
    /// `ComputeFeasibilityMultipliersPostprocess` (`cpp:964-984`), which
    /// upstream writes out twice.
    ///
    /// Returns the iterate that was in place beforehand, so a caller that
    /// must be able to undo the swap can. `None` means nothing was
    /// installed — no iterate, no multipliers, or the least-square solve
    /// failed — in which case the original iterate is left untouched.
    fn install_feasibility_multipliers(
        &mut self,
    ) -> Option<crate::iterates_vector::IteratesVector> {
        let nlp = self.nlp.as_ref().map(Rc::clone)?;
        let curr_backup = self.data.borrow().curr.clone()?;
        let (n_yc, n_yd) = (curr_backup.y_c.dim(), curr_backup.y_d.dim());
        if n_yc + n_yd == 0 {
            return None;
        }

        // Zero the bound multipliers and install that iterate *before* the
        // solve, so the least-square RHS is the feasibility problem's
        // (`cpp:893-910`): the calculator reads `curr`, so the zeroing has
        // to be visible to it, not applied to the result afterwards.
        let zeroed = |v: &Rc<dyn Vector>| -> Rc<dyn Vector> {
            let mut t = v.make_new();
            t.set(0.0);
            Rc::from(t)
        };
        let z_l = zeroed(&curr_backup.z_l);
        let z_u = zeroed(&curr_backup.z_u);
        let v_l = zeroed(&curr_backup.v_l);
        let v_u = zeroed(&curr_backup.v_u);
        self.data
            .borrow_mut()
            .set_curr(crate::iterates_vector::IteratesVector::new(
                curr_backup.x.clone(),
                curr_backup.s.clone(),
                curr_backup.y_c.clone(),
                curr_backup.y_d.clone(),
                z_l.clone(),
                z_u.clone(),
                v_l.clone(),
                v_u.clone(),
            ));

        let mut new_y_c = pounce_linalg::dense_vector::DenseVectorSpace::new(n_yc).make_new_dense();
        let mut new_y_d = pounce_linalg::dense_vector::DenseVectorSpace::new(n_yd).make_new_dense();
        let ok = match self.search_dir.as_mut() {
            None => false,
            Some(sd) => {
                let mut pd_guard = sd.pd_solver_mut();
                let ok = self.bundle.eq_mult.calculate_y_eq(
                    &self.data,
                    &self.cq,
                    &nlp,
                    pd_guard.aug_solver_mut(),
                    &mut new_y_c,
                    &mut new_y_d,
                );
                drop(pd_guard);
                ok
            }
        };
        if !ok {
            // `cpp:986` logs a warning and keeps whatever `y` was there.
            tracing::debug!(
                target: "pounce::algorithm",
                "square problem: least-square multiplier solve failed, keeping Newton multipliers",
            );
            self.data.borrow_mut().set_curr(curr_backup);
            return None;
        }

        self.data
            .borrow_mut()
            .set_curr(crate::iterates_vector::IteratesVector::new(
                curr_backup.x.clone(),
                curr_backup.s.clone(),
                Rc::new(new_y_c),
                Rc::new(new_y_d),
                z_l,
                z_u,
                v_l,
                v_u,
            ));
        Some(curr_backup)
    }

    /// Port of `IpoptAlgorithm::ComputeFeasibilityMultipliersPostprocess`
    /// (`IpIpoptAlg.cpp:949`). Same swap as
    /// [`Self::compute_feasibility_multipliers`], but unconditional: the
    /// run is over and the point has already been judged, so there is no
    /// convergence check to gate on and nothing to restore. Called on the
    /// two square-problem exits that report a feasible point
    /// (`cpp:484`, `cpp:542`), whose whole claim is that the constraints
    /// are satisfied — reporting `∇f` as the dual residual of such a point
    /// would contradict the status printed next to it.
    fn compute_feasibility_multipliers_postprocess(&mut self) {
        debug_assert!(self.is_square_problem());
        let _ = self.install_feasibility_multipliers();
    }

    /// Port of `IpoptAlgorithm::ComputeFeasibilityMultipliers`
    /// (`IpIpoptAlg.cpp:857`). On a square problem, once the iterate is
    /// primal-feasible to `constr_viol_tol`, re-estimate `y_c`/`y_d` as
    /// the multipliers of the *feasibility* problem: zero the four bound
    /// multipliers and take the least-square `y` against that iterate. If
    /// the convergence check then accepts, keep them; otherwise restore
    /// the iterate untouched.
    ///
    /// Why it exists (gh#508). A square problem is a system of equations.
    /// If the solver has found a point satisfying them to the tolerance
    /// the user declared, that point *is* the answer, and the leftover
    /// objective gradient is not evidence of anything. Without this,
    /// `inf_du` carries `∇f` — on the gh#508 probe `|2(x−5)| = 10` at a
    /// point whose violation is `1e-4` inside a `constr_viol_tol` of
    /// `1e-3` — the convergence check refuses, and the rapid-infeasibility
    /// detector convicts a point Ipopt calls feasible.
    ///
    /// Note the double convergence check, which is upstream's too: one to
    /// decide whether to bother (`cpp:880`), one to decide whether to keep
    /// the result (`cpp:924`). Both go through
    /// [`ConvergenceCheck::probe_convergence`], not the real check —
    /// upstream can afford `CheckConvergence` here because the only state
    /// it carries is `acceptable_counter_`, whereas pounce's also carries
    /// the gh#505 rapid-infeasibility streak, the gh#200 veto budget and
    /// the gh#533 progress window. Advancing those three times per
    /// iteration instead of once is not a faithful port of anything: on
    /// the gh#508 probe it moved the infeasibility conviction from
    /// iteration 86 to 33.
    fn compute_feasibility_multipliers(&mut self) {
        debug_assert!(self.is_square_problem());

        // Not primal feasible yet → no multipliers to compute (cpp:864).
        // Upstream measures this on the *unscaled* violation in the max
        // norm, against `constr_viol_tol`.
        let constr_viol_tol = self.bundle.conv_check.constr_viol_tol_or_default();
        if self.cq.borrow().curr_unscaled_primal_infeasibility_max() > constr_viol_tol {
            return;
        }

        // No calculator → upstream logs and leaves `y` alone (cpp:872).
        if self.nlp.is_none() {
            return;
        }

        // `iter_count + 1`: see the call site. Upstream has already
        // incremented when it reaches here.
        let iter_count = self.data.borrow().iter_count + 1;
        let nlp_err = self.cq.borrow().curr_nlp_error();
        if !nlp_err.is_finite() {
            return;
        }

        // Already converged, or out of iterations/time → do not touch the
        // multipliers (cpp:884). `Continue` is the case worth acting on:
        // it usually means dual feasibility is what is still missing.
        if self
            .bundle
            .conv_check
            .probe_convergence(nlp_err, iter_count, &self.data, &self.cq)
            != ConvergenceStatus::Continue
        {
            return;
        }

        let Some(curr_backup) = self.install_feasibility_multipliers() else {
            return;
        };

        // Keep them only if they actually buy a verdict (cpp:924).
        let nlp_err = self.cq.borrow().curr_nlp_error();
        if nlp_err.is_finite()
            && matches!(
                self.bundle
                    .conv_check
                    .probe_convergence(nlp_err, iter_count, &self.data, &self.cq,),
                ConvergenceStatus::Converged | ConvergenceStatus::ConvergedToAcceptable
            )
        {
            // Upstream marks nothing here; `"y "` is `recalc_y`'s. Use a
            // distinct tag so the iteration log says which mechanism
            // moved the multipliers.
            self.data.borrow_mut().append_info_string("f ");
            return;
        }

        tracing::debug!(
            target: "pounce::algorithm",
            "square problem: feasibility multipliers at iter {} did not converge the check, restoring",
            iter_count,
        );
        self.data.borrow_mut().set_curr(curr_backup);
    }

    /// Port of `IpIpoptAlg::correct_bound_multiplier`
    /// (`IpIpoptAlg.cpp:1055-1134`). Clamp each bound multiplier
    /// component into `[mu/(kappa_sigma * s_i), kappa_sigma * mu / s_i]`
    /// for all four bound-multiplier vectors.
    fn correct_bound_multiplier(&mut self) {
        if self.kappa_sigma < 1.0 {
            return;
        }
        let mu = self.data.borrow().curr_mu;
        let curr = match self.data.borrow().curr.clone() {
            Some(c) => c,
            None => return,
        };

        let cq = self.cq.borrow();

        let z_l_new = clamp_against_slack(&*curr.z_l, &*cq.curr_slack_x_l(), mu, self.kappa_sigma);
        let z_u_new = clamp_against_slack(&*curr.z_u, &*cq.curr_slack_x_u(), mu, self.kappa_sigma);
        let v_l_new = clamp_against_slack(&*curr.v_l, &*cq.curr_slack_s_l(), mu, self.kappa_sigma);
        let v_u_new = clamp_against_slack(&*curr.v_u, &*cq.curr_slack_s_u(), mu, self.kappa_sigma);
        drop(cq);

        let new_iv = crate::iterates_vector::IteratesVector::new(
            curr.x.clone(),
            curr.s.clone(),
            curr.y_c.clone(),
            curr.y_d.clone(),
            z_l_new,
            z_u_new,
            v_l_new,
            v_u_new,
        );
        self.data.borrow_mut().set_curr(new_iv);
    }

    /// Outer entry point — port of `IpoptAlgorithm::Optimize()`. Calls
    /// the iterate-initializer once, then loops `iterate()` until a
    /// terminal status. The exception → SolverReturn mapping
    /// (TINY_STEP_DETECTED → STEP_BECOMES_TINY,
    /// RESTORATION_FAILED → RESTORATION_FAILURE, etc.) lands in
    /// Phase 9 alongside the restoration phase.
    /// Run the solve and finalize its result.
    ///
    /// A thin wrapper on purpose. The gh #200 fallback must see **every** exit
    /// of the driver loop, and wiring it into individual termination sites was
    /// tried and failed — there are sixteen, and the ones easiest to overlook
    /// are the ones most likely to matter. Keeping the loop in a separate
    /// function means every `return` inside it, present or future, flows through
    /// [`Self::honour_refused_certificate`] by construction rather than by the
    /// author remembering to.
    ///
    /// This got more important once the fallback started changing the status in
    /// *both* directions: it can now hand back `StopAtAcceptablePoint` for a
    /// `Success` it was given. Anything reading `result` before the hook is
    /// reading a status that is not the one reported.
    pub fn optimize(&mut self) -> SolverReturn {
        let result = self.optimize_inner();

        // gh #200: a refused certificate outranks any non-success verdict the
        // continued run reached, and an earlier refusal can outrank the
        // continued run's own certificate. Applied here, once.
        let result = self.honour_refused_certificate(result);

        // pounce#250 follow-up: the dual-divergence guard's diversion to
        // restoration is a bet, and a lost bet must not return a worse point
        // than the solve already had in hand. Applied here, once, for the same
        // reason the #200 hook is — every `return` in the loop flows through
        // this point by construction.
        let result = self.honour_best_acceptable_after_dual_guard(result);

        // gh #534: deferring the acceptable-point restoration decline is also a
        // bet, and this is the net under it — a continuation that did not beat
        // the point the guard would have returned hands that point back. Last of
        // the three, so it compares against whatever the hooks above settled on.
        let result = self.honour_decline_floor(result);

        // gh #797: leaving a certified stationary point along a direction of
        // negative curvature is a bet placed *from* a certificate, so it is
        // settled last — whatever the hooks above arrived at, the escape either
        // beat the point it left with a certificate of its own or that point is
        // handed back.
        let result = self.honour_neg_curv_floor(result);

        // Terminal post-mortem checkpoint. Skipped when the user already
        // asked to stop (they were just at a prompt); otherwise the
        // debugger gets a last look at the final/failing iterate.
        if !matches!(result, SolverReturn::UserRequestedStop) {
            self.fire_debug_terminal(result);
        }
        result
    }

    fn optimize_inner(&mut self) -> SolverReturn {
        // Top-level span for the whole solve; every iteration / linear
        // solve / restoration event nests under it (pounce#71).
        let _solve_span = tracing::info_span!("solve").entered();

        // Shared timing accumulator — every phase below records into it.
        let timing = self.data.borrow().timing.clone();

        // Install the shared accumulator on the augmented-system solver
        // so its factor / back-solve calls are attributed to
        // `linear_system_factorization` / `linear_system_back_solve`.
        // Same pattern for the diagnostics state when present, so KKT
        // dump sites can consult per-iter gating.
        if let Some(sd) = self.search_dir.as_mut() {
            sd.pd_solver_mut()
                .aug_solver_mut()
                .set_timing_stats(std::rc::Rc::clone(&timing));
            if let Some(diag) = self.diagnostics.as_ref() {
                sd.pd_solver_mut()
                    .aug_solver_mut()
                    .set_diagnostics(Rc::clone(diag));
            }
        }

        // 0a. Strategy initialization — port of upstream's
        //     `IpoptAlgorithm::InitializeImpl` calls. The mu update needs
        //     `data.curr_mu`/`curr_tau` seeded before the iterate
        //     initializer runs (`CalculateSafeSlack` reads them).
        self.bundle.mu_update.initialize(&self.data);

        // 0b. Iterate initializer. Requires NLP; without one the caller
        //    must have populated `data.curr` themselves.
        if let Some(nlp) = self.nlp.as_ref() {
            // The initializer needs an aug-system solver for the
            // least-square multiplier branch; until that's wired we
            // route through whatever the search-direction calculator
            // owns when present. For the stub flow we skip the LSM
            // path by giving the initializer a dummy solver only if
            // the search_dir is present (otherwise the init function
            // is responsible for not consulting it).
            if let Some(sd) = self.search_dir.as_mut() {
                timing.initialize_iterates.start();
                let mut pd_guard = sd.pd_solver_mut();
                let aug_solver = pd_guard.aug_solver_mut();
                let ok = self
                    .bundle
                    .init
                    .set_initial_iterates(&self.data, &self.cq, nlp, aug_solver);
                drop(pd_guard);
                timing.initialize_iterates.end();
                if !ok {
                    return SolverReturn::InvalidProblemDefinition;
                }
            }
        }

        // 0c. Seed `IpoptData::w` with the initial-iterate Hessian.
        //     Redundant with the iter-body `update_hessian` call (which
        //     now runs BEFORE `update_barrier_parameter`) but kept to
        //     cover any code path that consults `data.w` between
        //     `set_initial_iterates` and the first `iterate()` call
        //     (e.g. the iter-0 trace dump below).
        if self.data.borrow().curr.is_some() {
            timing.update_hessian.start();
            let _ = self.bundle.hess.update_hessian(&self.data, &self.cq);
            timing.update_hessian.end();
        }

        // Track-A iterate-trace dumper. Activated by
        // `IPOPT_ITER_DUMP_PATH`; otherwise no-op. See `iter_dump.rs`.
        let mut dumper = IterDumper::from_env();
        // Iter 0 record — captures the initialised iterate before any
        // step. Mirrors upstream's "after InitializeIterates(), before
        // the loop" emission point.
        if let Some(d) = dumper.as_mut() {
            d.write_record(&self.data, &self.cq);
        }

        // Advance the diagnostics iter counter so the first `iterate()`
        // body reports as iter 0 (matches `data.iter_count`). Subsequent
        // bumps live at the bottom of the loop alongside the iter_count
        // bookkeeping.
        if let Some(diag) = self.diagnostics.as_ref() {
            diag.bump_iter();
            // Iter-0 iterate row (issue #68). Same hook point as
            // the binary IterDumper above; emits only when
            // `--dump iterates:*` is configured.
            emit_iterate_record(diag.as_ref(), &self.data, &self.cq);
        }

        // Iter 0 intermediate callback — upstream fires once after
        // `InitializeIterates` before the loop body starts so users
        // observe the initial point.
        if !self.fire_intermediate() {
            return SolverReturn::UserRequestedStop;
        }
        if self.fire_debug(crate::debug::Checkpoint::IterStart) == crate::debug::DebugAction::Stop {
            return SolverReturn::UserRequestedStop;
        }

        // pounce#246: bound the initialization / restoration-entry window.
        // Everything above — `mu_update.initialize`, `set_initial_iterates`
        // (which for a bad warm start can grind in its least-square /
        // feasibility setup), and the initial `update_hessian` — runs
        // *before* the first `iterate()`, whose convergence check and
        // post-`compute_search_direction` gate are the earliest deadline
        // checks (#242/#244/#245). A solve handed a poor warm start could
        // therefore spend the whole budget here and only consult the
        // deadline once it reached the first outer-iteration /
        // KKT-factorization boundary. Consult it now, before the loop, so a
        // bad-start init stall returns promptly with the time-limit status
        // (best-so-far being the initialised iterate) instead of running to
        // a multiple of the budget. This also bounds the *restoration
        // entry*: the nested restoration IPM shares this `Deadline` and runs
        // the same `optimize_inner`, so a budget already crossed by the time
        // the inner solve starts up terminates it here rather than after its
        // own first iterate.
        if let Some(ret) = self.deadline_status() {
            return ret;
        }

        let result = loop {
            match self.iterate() {
                IterateOutcome::Terminate(ret) => break ret,
                IterateOutcome::Continue => {
                    // Source the local counter from `data.iter_count`
                    // each pass so a pre-seeded counter (e.g. the inner
                    // restoration IPM at `outer.iter + 1`, matching
                    // upstream `IpRestoMinC_1Nrm.cpp:181`) and any
                    // restoration step that set
                    // `data.iter_count = inner.iter_count - 1`
                    // (mirroring `IpRestoMinC_1Nrm.cpp:Set_iter_count`)
                    // are honored — without this the local counter
                    // would advance from its pre-restoration value,
                    // ignoring the inner-IPM iterations.
                    let mut iter_count: Index = self.data.borrow().iter_count;
                    iter_count += 1;
                    // Do NOT short-circuit to `MaxiterExceeded` here: bump the
                    // counter and loop, letting the next `iterate()` run its
                    // convergence check (`OptimalityErrorConvergenceCheck`,
                    // which tests the component tolerances *before* its own
                    // `iter_count >= max_iter` gate at
                    // `conv_check/opt_error.rs`). Breaking before that call
                    // skipped the convergence test on the iterate produced by
                    // the final permitted step, so a solve converging on
                    // exactly the `max_iter`-th iterate reported
                    // `Maximum_Iterations_Exceeded` where upstream Ipopt —
                    // which runs `CheckConvergence` at the top of its loop,
                    // convergence-first — reports success. The check is
                    // guaranteed to terminate the loop: once `iter_count`
                    // reaches `max_iter`, `check_convergence_with_state`
                    // returns either `Converged`/`ConvergedToAcceptable` or
                    // `MaxIterExceeded`, never `Continue` (L1).
                    self.data.borrow_mut().iter_count = iter_count;
                    // Floor evidence for restoration's gh#661 divergence
                    // guard: how long this solve has sat at a violation
                    // it could not get below. Sampled here, once per
                    // accepted iterate, from the same quantity the
                    // `inf_pr` column reports — so it is free, and it
                    // sees the whole outer trajectory rather than the
                    // handful of iterations a restoration sub-solve runs.
                    // The nested restoration IPM reaches this line too,
                    // but writes its own `IpoptData`, so the two
                    // trajectories never mix.
                    let inf_pr_now = self.cq.borrow().curr_primal_infeasibility_max();
                    self.data.borrow_mut().inf_pr_floor.observe(inf_pr_now);
                    // Keep the diagnostics counter in lock-step with
                    // `data.iter_count` so KKT-dump gating reflects the
                    // about-to-execute iteration.
                    if let Some(diag) = self.diagnostics.as_ref() {
                        diag.bump_iter();
                        // Per-iter iterate row (issue #68). Mirrors
                        // the binary IterDumper hook below.
                        emit_iterate_record(diag.as_ref(), &self.data, &self.cq);
                    }
                    // Per-iteration record — emitted after the
                    // iter_count bump so the recorded `iter` field
                    // matches `IpData().iter_count()` at the moment of
                    // emission, identical to upstream's writer.
                    if let Some(d) = dumper.as_mut() {
                        d.write_record(&self.data, &self.cq);
                    }
                    // Per-iteration intermediate callback — fired with
                    // an `IntermediateContext` guard so downstream
                    // inspector entry points (the C API
                    // `GetIpoptCurrent*` family) see live state for the
                    // duration of the user callback.
                    if !self.fire_intermediate() {
                        break SolverReturn::UserRequestedStop;
                    }
                    if self.fire_debug(crate::debug::Checkpoint::IterStart)
                        == crate::debug::DebugAction::Stop
                    {
                        break SolverReturn::UserRequestedStop;
                    }
                }
            }
        };

        result
    }
}

/// A termination certificate the masked-scale veto refused (gh #200), with
/// everything the fallback needs to undo the refusal verbatim.
///
/// One struct rather than a field per component. The fallback is only correct if
/// these all describe *the same iterate*, and parallel `Option`s make
/// "objective recorded, iterate missing" representable — which was reachable:
/// the iterate is cloned out of `data.curr` and can come back `None`, while the
/// objective and barrier parameter were written unconditionally. Capture is now
/// all-or-nothing, so the disagreement cannot be constructed.
#[derive(Clone)]
struct VetoSnapshot {
    /// The refused iterate itself.
    iterate: crate::iterates_vector::IteratesVector,
    /// Iteration at which the refusal happened.
    ///
    /// Needed to identify which refusal is the *baseline-equivalent* one. The
    /// baseline stops at the first iterate where it would terminate, so when
    /// both a strict and an acceptable-level refusal are on record it is the
    /// chronologically earlier one that says what the baseline returned — not
    /// the stricter one. The later refusal sits on the continued trajectory,
    /// which the baseline never walked, so comparing against it compares
    /// against a point that was never on offer.
    iter: Index,
    /// Scaled objective there, so the refused point can be compared against
    /// whatever the continued run reached without re-evaluating it.
    obj: Number,
    /// Barrier parameter there.
    ///
    /// `curr_mu` lives on `IpoptData` rather than in the `IteratesVector`, so
    /// restoring the iterate does not rewind it, and `stats.final_mu` is read
    /// after the restore — leaving the continued run's barrier parameter
    /// reported next to the refused run's `x`. That pair feeds a warm-started
    /// corrector's `mu_init` and reaches callers as `info["mu"]`, so it must
    /// describe the point actually returned. (Not currently observable: `mu` has
    /// bottomed out at its floor in every fallback case reachable so far, making
    /// the two values coincide. Kept correct rather than left to depend on that.)
    mu: Number,
    /// Max-norm unscaled KKT error there, so the tiebreak can see what
    /// `apply_kkt_fidelity_gate` will see. Recorded at refusal time because the
    /// gate runs post-solve, long after this iterate is gone.
    unscaled_kkt: Number,
    /// Unscaled max-norm constraint violation there — the same quantity the
    /// `acceptable_constr_viol_tol` gate is defined against
    /// (`curr_unscaled_primal_infeasibility_max`, cf. gh #261). The
    /// best-acceptable fallback ranks candidates by `(feasible_enough,
    /// objective)` rather than objective alone, so a point outside a capped
    /// feasibility band can never displace one inside it on objective grounds;
    /// without this field the ranking has no feasibility term and, under a
    /// user-widened `acceptable_constr_viol_tol`, will trade feasibility for
    /// objective and hand back a verifiably infeasible point under a success
    /// status (gh #267). Unused by the gh #200 masked-scale paths, which key on
    /// objective only.
    constr_viol: Number,
    /// Objective scaling factor in force when `obj` was recorded.
    ///
    /// `obj` is a *scaled* objective, so comparing it against the continued
    /// run's is only meaningful under the same factor, sign included. Held so
    /// that assumption is asserted rather than trusted: periodic or adaptive
    /// rescaling is a natural thing to add for exactly the ill-scaled problems
    /// this mechanism targets, and it would silently turn the comparison into
    /// noise.
    obj_scale: Number,
}

/// Internal result of one [`IpoptAlgorithm::iterate`] call. Mirrors the
/// upstream try/catch around `IpoptAlg::Optimize` — anything that's not
/// `Continue` carries the [`SolverReturn`] that the outer loop will
/// surface to `IpoptApplication`.
enum IterateOutcome {
    Continue,
    Terminate(SolverReturn),
}

/// Feasibility-aware ranking core for the best-acceptable fallback (gh #267).
///
/// `true` iff candidate `(a_obj, a_viol)` ranks **strictly** better than
/// `(b_obj, b_viol)` under the `(band_clamped_viol, objective)` key, where each
/// violation is clamped up to `band` before it is compared. Lexicographic: the
/// point with the smaller clamped violation wins outright; only when the clamped
/// violations tie does the lower objective decide. A non-finite objective ranks
/// worst (never wins, always loses to a finite one); a non-finite violation is
/// treated as infinitely infeasible.
///
/// The clamp is what makes this a **total order** rather than a two-class
/// partition, and it is the whole gh #280 fix. Clamping the violation *up* to
/// `band` collapses every point inside the feasibility band to the single value
/// `band`, so within the band those points tie on feasibility and objective
/// decides — the intended, gh #267-preserving behaviour. Outside the band the
/// clamp is the identity, so the *actual* violation decides and the
/// less-infeasible point always wins. The earlier `(feasible_enough, objective)`
/// key was a two-class partition: once **both** points sat outside the band it
/// fell through to a bare `a_obj < b_obj`, reading neither violation — the exact
/// pre-#267 objective-only rule, which lets a strictly-more-infeasible point win
/// on objective (gh #280). Under the clamped key a strictly-more-infeasible
/// point can never rank better, at any band.
///
/// Pure and total, so the fallback's "never worse off" guarantee is a theorem
/// this function's unit tests prove by cases — host-independent by construction,
/// unlike an end-to-end objective comparison across two live nonconvex solves
/// (the trap gh #267 caught). [`IpoptAlgorithm::ranks_better`] supplies `band`
/// as `min(acceptable_constr_viol_tol, FEASIBLE_ENOUGH_CAP)`; both the record
/// and the read side route through here, so they cannot disagree.
/// Append `value` to a fixed-capacity oldest-first window, dropping the oldest
/// sample once the window is full (gh #534).
fn push_sample(buf: &mut [Number; DECLINE_PROGRESS_SAMPLES], len: &mut usize, value: Number) {
    if *len < DECLINE_PROGRESS_SAMPLES {
        buf[*len] = value;
        *len += 1;
    } else {
        buf.rotate_left(1);
        buf[DECLINE_PROGRESS_SAMPLES - 1] = value;
    }
}

/// Whether every consecutive pair in `samples` (oldest first) contracted by at
/// least `ratio` — the gh #534 progress test, as a pure function.
///
/// A sample that is not finite, or a predecessor that is not strictly positive,
/// fails the window: neither is evidence of progress, and a zero predecessor
/// makes the ratio meaningless. A `ratio` of `1` admits any non-increasing
/// window, and a large one admits every finite window — which is how
/// `resto_decline_progress_ratio` doubles as the "drop the progress
/// requirement" switch.
///
/// Pure and total for the same reason [`ranks_better_within_band`] is: the two
/// traces the issue records — `eigena2` quartering and `eigenb2` rising — decide
/// what this must do, and a unit test can hold it to them exactly.
fn window_is_contracting(samples: &[Number], ratio: Number) -> bool {
    samples.windows(2).all(|w| {
        let (prev, next) = (w[0], w[1]);
        prev.is_finite() && prev > 0.0 && next.is_finite() && next <= ratio * prev
    })
}

fn ranks_better_within_band(
    a_obj: Number,
    a_viol: Number,
    b_obj: Number,
    b_viol: Number,
    band: Number,
) -> bool {
    if !a_obj.is_finite() {
        return false;
    }
    if !b_obj.is_finite() {
        return true;
    }
    // Clamp each violation up to `band`: everything inside the feasibility band
    // maps to the single value `band` (so objective decides there), while outside
    // it the actual violation is kept (so the less-infeasible point strictly
    // wins). A non-finite violation is infinitely infeasible. This is a total
    // order — a strictly-more-infeasible point can never rank better (gh #280).
    let clamped = |v: Number| {
        if v.is_finite() {
            v.max(band)
        } else {
            Number::INFINITY
        }
    };
    let (a_key, b_key) = (clamped(a_viol), clamped(b_viol));
    if a_key != b_key {
        // The less-infeasible point wins outright, whatever the objectives.
        return a_key < b_key;
    }
    // Same clamped feasibility (both inside the band, or an exact tie outside it):
    // lower objective wins (the original within-band behaviour).
    a_obj < b_obj
}

/// `||a - b||_2 / (1 + ||b||_2)`. Used by the restoration cycle
/// detector in [`IpoptAlgorithm::invoke_restoration`] to test whether
/// the outer iterate has moved between two consecutive restoration
/// entries.
fn relative_distance(a: &dyn Vector, b: &dyn Vector) -> Number {
    if a.dim() == 0 {
        return 0.0;
    }
    let mut diff = a.make_new_copy();
    diff.axpy(-1.0, b);
    diff.nrm2() / (1.0 + b.nrm2())
}

/// `out = curr + α_p · δ` for the primal/equality blocks and
/// `out = curr + α_d · δ` for the bound multipliers, returned as a
/// fresh frozen `IteratesVector`. Mirrors `scaled_step` in the line
/// search; duplicated here for the tiny-step branch which bypasses
/// the line-search driver.
fn scaled_step_unchecked(
    curr: &crate::iterates_vector::IteratesVector,
    delta: &crate::iterates_vector::IteratesVector,
    alpha_primal: Number,
    alpha_dual: Number,
) -> crate::iterates_vector::IteratesVector {
    let mut out = curr.make_new_zeroed();
    out.add_one_vector(1.0, curr, 0.0);
    out.x.axpy(alpha_primal, &*delta.x);
    out.s.axpy(alpha_primal, &*delta.s);
    out.y_c.axpy(alpha_primal, &*delta.y_c);
    out.y_d.axpy(alpha_primal, &*delta.y_d);
    out.z_l.axpy(alpha_dual, &*delta.z_l);
    out.z_u.axpy(alpha_dual, &*delta.z_u);
    out.v_l.axpy(alpha_dual, &*delta.v_l);
    out.v_u.axpy(alpha_dual, &*delta.v_u);
    out.freeze()
}

/// Allocate a fresh `Rc<dyn Vector>` with `kappa_sigma_clamp`
/// applied component-wise against the supplied `slack`. Inputs are
/// borrowed; the original `z` is never mutated. Ports the per-vector
/// piece of `IpIpoptAlg.cpp:1080-1133`.
fn clamp_against_slack(
    z: &dyn Vector,
    slack: &dyn Vector,
    mu: Number,
    kappa_sigma: Number,
) -> Rc<dyn Vector> {
    debug_assert_eq!(z.dim(), slack.dim());
    let n = z.dim() as usize;
    // Flatten both z and slack into contiguous slices so the
    // elementwise clamp doesn't care whether the inputs are
    // [`DenseVector`] (regular IPM path) or [`CompoundVector`]
    // (resto IPM path). The result is reconstructed into a
    // same-shape Vector via `Vector::make_new` + a flat-write
    // helper so the caller sees a vector with the same blocking as
    // its input.
    let mut buf = vec![0.0_f64; n];
    flat_read_into(z, &mut buf);
    let s_vals = flat_read_owned(slack);
    let _ = kappa_sigma_clamp(&mut buf, &s_vals, mu, kappa_sigma);
    let mut out: Box<dyn Vector> = z.make_new();
    flat_write_into(&mut *out, &buf);
    Rc::from(out)
}

pub(crate) fn flat_read_into(v: &dyn Vector, dst: &mut [Number]) {
    if let Some(dv) = v
        .as_any()
        .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
    {
        let vs = dv.expanded_values();
        dst.copy_from_slice(&vs);
        return;
    }
    if let Some(cv) = v.as_any().downcast_ref::<pounce_linalg::CompoundVector>() {
        let mut off = 0usize;
        for k in 0..cv.n_comps() {
            let blk = cv.comp(k);
            let dim = blk.dim() as usize;
            let dblk = blk
                .as_any()
                .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
                .expect("clamp_against_slack: CompoundVector blocks must be DenseVectors");
            let vs = dblk.expanded_values();
            dst[off..off + dim].copy_from_slice(&vs);
            off += dim;
        }
        return;
    }
    panic!("clamp_against_slack: unsupported Vector kind");
}

pub(crate) fn flat_read_owned(v: &dyn Vector) -> Vec<Number> {
    let mut out = vec![0.0; v.dim() as usize];
    flat_read_into(v, &mut out);
    out
}

pub(crate) fn flat_write_into(v: &mut dyn Vector, src: &[Number]) {
    if let Some(dv) = v
        .as_any_mut()
        .downcast_mut::<pounce_linalg::dense_vector::DenseVector>()
    {
        dv.set_values(src);
        return;
    }
    if let Some(cv) = v
        .as_any_mut()
        .downcast_mut::<pounce_linalg::CompoundVector>()
    {
        let mut off = 0usize;
        for k in 0..cv.n_comps() {
            let blk = cv.comp_mut(k);
            let dim = blk.dim() as usize;
            let dblk = blk
                .as_any_mut()
                .downcast_mut::<pounce_linalg::dense_vector::DenseVector>()
                .expect("clamp_against_slack: CompoundVector blocks must be DenseVectors");
            dblk.set_values(&src[off..off + dim]);
            off += dim;
        }
        return;
    }
    panic!("clamp_against_slack: unsupported Vector kind");
}

/// Per-element kappa-sigma clamp — the elementwise arithmetic at the
/// heart of `IpIpoptAlg.cpp:correct_bound_multiplier` (lines
/// 1090-1133). For each index `i`:
///
/// ```text
///   slack_i  = max(slack_i, tiny_double)   // avoid /0
///   z_lo_i   = mu / (kappa_sigma * slack_i)
///   z_hi_i   = kappa_sigma * mu / slack_i
///   z_i      ← clamp(z_i, z_lo_i, z_hi_i)
/// ```
///
/// Returns the maximum elementwise correction magnitude (matching
/// upstream's `Max(max_correction_up, max_correction_low)`).
///
/// `kappa_sigma < 1` short-circuits to the identity per upstream's
/// guard at line 1065.
pub fn kappa_sigma_clamp(
    z: &mut [Number],
    slack: &[Number],
    mu: Number,
    kappa_sigma: Number,
) -> Number {
    debug_assert_eq!(z.len(), slack.len());
    if kappa_sigma < 1.0 {
        return 0.0;
    }
    let mut max_correction = 0.0_f64;
    for (zi, &si) in z.iter_mut().zip(slack.iter()) {
        let s_safe = si.max(Number::MIN_POSITIVE);
        let lo = mu / (kappa_sigma * s_safe);
        let hi = kappa_sigma * mu / s_safe;
        let clamped = zi.clamp(lo, hi);
        let delta = (clamped - *zi).abs();
        if delta > max_correction {
            max_correction = delta;
        }
        *zi = clamped;
    }
    max_correction
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- phase-profile counting ----

    fn signs(vals: &[&[Number]]) -> Vec<Box<dyn Vector>> {
        use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
        vals.iter()
            .map(|block| {
                let mut v = DenseVector::new(DenseVectorSpace::new(block.len() as Index));
                v.set_values(block);
                Box::new(v) as Box<dyn Vector>
            })
            .collect()
    }

    #[test]
    fn active_count_takes_the_positive_signs_across_every_block() {
        // Two positives in the first block, none in the second, an
        // empty third block, one in the fourth.
        let f = signs(&[&[1.0, -1.0, 1.0], &[-1.0], &[], &[1.0]]);
        assert_eq!(count_active(&f), 3);
    }

    #[test]
    fn an_exact_tie_counts_as_inactive_not_as_half_a_bound() {
        // `z_i == s_i` gives sign 0. It has to land on one side or the
        // other of an integer count; inactive is the conservative read,
        // and the alternative -- letting it contribute a fraction --
        // would make the count non-integral for a measure-zero event.
        let f = signs(&[&[0.0, 0.0, 1.0]]);
        assert_eq!(count_active(&f), 1);
    }

    #[test]
    fn changes_count_each_moved_index_exactly_once() {
        let now = signs(&[&[1.0, -1.0, 1.0, -1.0]]);
        let prev = signs(&[&[-1.0, -1.0, 1.0, 1.0]]);
        assert_eq!(count_activity_changes(&now, &prev), Some(2));
    }

    #[test]
    fn a_tie_crossing_counts_as_one_move_not_a_half() {
        // |now - prev| is 1 here and 2 for a full sign flip; clamping at
        // one before summing is what keeps both worth exactly one index.
        // A plain `sum / 2` would report 0.5 for this row.
        let now = signs(&[&[1.0, -1.0]]);
        let prev = signs(&[&[0.0, 0.0]]);
        assert_eq!(count_activity_changes(&now, &prev), Some(2));
    }

    #[test]
    fn an_unmoved_fingerprint_reports_zero_changes() {
        let now = signs(&[&[1.0, -1.0], &[1.0]]);
        let prev = signs(&[&[1.0, -1.0], &[1.0]]);
        assert_eq!(count_activity_changes(&now, &prev), Some(0));
    }

    #[test]
    fn a_fingerprint_from_a_different_index_space_is_refused() {
        // Comparing two different index spaces would produce a number
        // that reads like a measurement and is not one. `None` -- the
        // same value the first iteration reports -- says "not measured"
        // instead.
        let now = signs(&[&[1.0, -1.0, 1.0]]);
        assert_eq!(count_activity_changes(&now, &signs(&[&[1.0, -1.0]])), None);
        assert_eq!(
            count_activity_changes(&now, &signs(&[&[1.0, -1.0, 1.0], &[1.0]])),
            None
        );
    }

    #[test]
    fn kappa_sigma_below_one_is_identity() {
        let mut z = vec![1.0, 2.0, 3.0];
        let slack = [1.0, 1.0, 1.0];
        let m = kappa_sigma_clamp(&mut z, &slack, 1.0, 0.5);
        assert_eq!(m, 0.0);
        assert_eq!(z, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn within_band_is_unchanged() {
        // mu=1, kappa=10, slack=1 → band [0.1, 10]. z=1 → unchanged.
        let mut z = vec![1.0];
        let slack = [1.0];
        let m = kappa_sigma_clamp(&mut z, &slack, 1.0, 10.0);
        assert_eq!(m, 0.0);
        assert_eq!(z, [1.0]);
    }

    #[test]
    fn above_upper_clamped_down() {
        // mu=1, kappa=10, slack=1 → upper = 10. z=100 → 10.
        let mut z = vec![100.0];
        let slack = [1.0];
        let m = kappa_sigma_clamp(&mut z, &slack, 1.0, 10.0);
        assert!((m - 90.0).abs() < 1e-13);
        assert_eq!(z, [10.0]);
    }

    #[test]
    fn below_lower_clamped_up() {
        // mu=1, kappa=10, slack=1 → lower = 0.1. z=0.001 → 0.1.
        let mut z = vec![0.001];
        let slack = [1.0];
        let m = kappa_sigma_clamp(&mut z, &slack, 1.0, 10.0);
        assert!((m - 0.099).abs() < 1e-13);
        assert!((z[0] - 0.1).abs() < 1e-15);
    }

    #[test]
    fn returns_max_over_components() {
        let mut z = vec![100.0, 0.001];
        let slack = [1.0, 1.0];
        let m = kappa_sigma_clamp(&mut z, &slack, 1.0, 10.0);
        assert!((m - 90.0).abs() < 1e-13);
        assert_eq!(z[0], 10.0);
        assert!((z[1] - 0.1).abs() < 1e-15);
    }

    #[test]
    fn slack_clamped_to_min_positive_avoids_division_by_zero() {
        let mut z = vec![1e100];
        let slack = [0.0];
        let _ = kappa_sigma_clamp(&mut z, &slack, 1.0, 10.0);
        assert!(z[0].is_finite() || z[0] == 1e100);
    }

    /// The restoration slot is exercised structurally:
    /// `IpoptAlgorithm::with_restoration` accepts a
    /// `Box<dyn RestorationPhase>` and the trait's default
    /// `perform_restoration` returns `Failed`. End-to-end coverage
    /// (iterate() → line-search-Failed → restoration → recovered)
    /// lands in the Phase 9 integration suite alongside the nested
    /// IPM driver.
    struct _DummyResto;
    impl RestorationPhase for _DummyResto {}

    // --------------------------------------------------------------
    // Best-acceptable fallback ranking (gh #267).
    //
    // These prove the "never worse off" guarantee host-independently, by
    // cases, on the pure ranking core — the property the earlier end-to-end
    // `hair_trigger_*` objective comparison could only *approximate* on one
    // host's basin luck (gh #267's secondary finding). `band` here stands in
    // for the resolved `min(acceptable_constr_viol_tol, FEASIBLE_ENOUGH_CAP)`.
    // --------------------------------------------------------------

    #[test]
    fn ranks_better_is_a_strict_order_within_a_feasibility_class() {
        let band = 1e-2;
        // Both feasible: lower objective wins, strictly.
        assert!(ranks_better_within_band(-2.0, 1e-4, -1.0, 1e-4, band));
        assert!(!ranks_better_within_band(-1.0, 1e-4, -2.0, 1e-4, band));
        // Ties are not "strictly better" in either direction — so an equal
        // returned point is never displaced, matching the read side's
        // keep-current-on-tie contract.
        assert!(!ranks_better_within_band(-1.0, 1e-4, -1.0, 5e-3, band));
        assert!(!ranks_better_within_band(-1.0, 5e-3, -1.0, 1e-4, band));
        // Both infeasible: the less-infeasible point wins. Here it also has the
        // lower objective, so this held under the old objective-only fall-through
        // too — `ranks_better_puts_feasibility_first_among_two_infeasibles` is the
        // case that separates the two rules (gh #280).
        assert!(ranks_better_within_band(-2.0, 5.0, -1.0, 9.0, band));
    }

    #[test]
    fn ranks_better_puts_feasibility_first_among_two_infeasibles() {
        // The gh #280 hole: once BOTH points sit outside the (capped) band the
        // old `(feasible_enough, objective)` partition read a_ok == b_ok == false
        // and fell through to `a_obj < b_obj` — objective alone, the exact
        // pre-#267 rule — so a strictly-MORE-infeasible point could win by having
        // a better objective. This is the deb7 swap the fallback made: incumbent
        // at viol 5.292e-1, recorded point at viol 9.951e-1 with a 36%-better
        // objective. The less-infeasible point must win regardless of objective.
        let band = 1e-2;
        // Less-infeasible incumbent, WORSE objective — must still win.
        assert!(ranks_better_within_band(
            89.0, 5.292e-1, 56.9, 9.951e-1, band
        ));
        // The more-infeasible, better-objective point must NOT win — the swap
        // gh #280 forbids.
        assert!(!ranks_better_within_band(
            56.9, 9.951e-1, 89.0, 5.292e-1, band
        ));
        // A strictly-more-infeasible point never replaces the incumbent even with
        // an arbitrarily better objective.
        assert!(!ranks_better_within_band(-1e12, 9.0, 0.0, 5.0, band));
        assert!(ranks_better_within_band(0.0, 5.0, -1e12, 9.0, band));
    }

    #[test]
    fn ranks_better_puts_feasibility_first_regardless_of_objective() {
        let band = 1e-2;
        // The gh #267 case in miniature: a feasible point outranks an
        // arbitrarily-lower-objective infeasible one, and vice-versa.
        assert!(ranks_better_within_band(0.0, 1e-4, -1e9, 9.94, band));
        assert!(!ranks_better_within_band(-1e9, 9.94, 0.0, 1e-4, band));
        // Exactly at the band is still feasible_enough; just past it is not.
        assert!(ranks_better_within_band(
            1.0,
            band,
            -1.0,
            band * 1.000_001,
            band
        ));
    }

    #[test]
    fn ranks_better_never_lets_a_widened_band_matter_past_the_cap() {
        // The band the method feeds is capped at FEASIBLE_ENOUGH_CAP, so a
        // point beyond the cap is never feasible_enough however loose the
        // user's `acceptable_constr_viol_tol`. Model that by passing the capped
        // band: the near-optimal-but-mildly-infeasible endpoint (viol 1.13e-4,
        // within the cap) must beat the grossly-infeasible lower-objective
        // point (viol 9.94, past it) — the exact swap the fix forbids.
        let band = IpoptAlgorithm::FEASIBLE_ENOUGH_CAP;
        assert!(ranks_better_within_band(
            -2303.99, 1.13e-4, -2307.32, 9.94, band
        ));
        assert!(!ranks_better_within_band(
            -2307.32, 9.94, -2303.99, 1.13e-4, band
        ));
    }

    #[test]
    fn ranks_better_ranks_a_nonfinite_objective_worst() {
        let band = 1e-2;
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            // A non-finite objective never ranks better than a finite one...
            assert!(!ranks_better_within_band(bad, 0.0, 0.0, 9.9, band));
            // ...and always loses to one, even on worse feasibility — so a
            // finite recorded point is restored over a NaN-objective return,
            // preserving the pre-fix `!(curr <= best)` NaN behaviour.
            assert!(ranks_better_within_band(0.0, 9.9, bad, 0.0, band));
        }
    }

    #[test]
    fn ranks_better_treats_a_nonfinite_violation_as_infeasible() {
        let band = 1e-2;
        // A NaN violation is never feasible_enough, so a genuinely feasible
        // point outranks it regardless of objective.
        assert!(ranks_better_within_band(0.0, 0.0, -100.0, f64::NAN, band));
        assert!(!ranks_better_within_band(-100.0, f64::NAN, 0.0, 0.0, band));
    }

    /// gh #534, the case the guard was stopping: `eigena2`'s dual infeasibility
    /// quarters on unit steps for four straight iterations, three short of a
    /// strict certificate. Quoted from the issue's own iteration table.
    #[test]
    fn eigena2_endgame_reads_as_contracting() {
        let eigena2 = [1.19e-05, 2.96e-06, 7.38e-07, 1.84e-07];
        assert!(window_is_contracting(
            &eigena2,
            DEFAULT_DECLINE_PROGRESS_RATIO
        ));
    }

    /// gh #534, the case the guard was right about: `eigenb2`'s tail *rises*
    /// on heavily backtracked steps. The issue calls it a plausible genuine
    /// stall, so the progress test must refuse it and leave the guard alone.
    #[test]
    fn eigenb2_stall_does_not_read_as_contracting() {
        let eigenb2 = [1.88e-07, 2.69e-07, 2.89e-07, 2.93e-07];
        assert!(!window_is_contracting(
            &eigenb2,
            DEFAULT_DECLINE_PROGRESS_RATIO
        ));
    }

    /// gh #534: `csfi2`'s window, measured on this build at the guard. Three
    /// healthy contractions and then a flat step — the solve has stopped
    /// moving, so the deferral must not fire however good the earlier ratios
    /// look. This is the shape every live guard firing reachable from the
    /// in-repo corpus has, which is why the whole window is tested and not
    /// just its first ratios.
    #[test]
    fn csfi2_flat_final_step_does_not_read_as_contracting() {
        let csfi2 = [3.267e0, 1.845e-6, 8.468e-8, 8.524e-8];
        assert!(!window_is_contracting(
            &csfi2,
            DEFAULT_DECLINE_PROGRESS_RATIO
        ));
        // ... and it is the *last* step that decides: drop it and the same
        // trace passes, which is exactly the distinction the test exists for.
        assert!(window_is_contracting(
            &csfi2[..3],
            DEFAULT_DECLINE_PROGRESS_RATIO
        ));
    }

    /// gh #534: a large ratio drops the progress requirement, so the decline is
    /// deferred on any window. That is the "bypass the guard and see how far the
    /// solve gets" switch the issue asks for. A ratio of exactly `1` is the
    /// weaker "no backsliding" reading and still refuses `csfi2`, whose last
    /// step rises.
    #[test]
    fn a_large_ratio_accepts_a_stalled_window() {
        let csfi2 = [3.267e0, 1.845e-6, 8.468e-8, 8.524e-8];
        assert!(!window_is_contracting(&csfi2, 1.0));
        assert!(window_is_contracting(&[1e-8, 1e-8, 1e-8, 1e-8], 1.0));
        assert!(window_is_contracting(&csfi2, 1e20));
        // Still not a licence to read garbage as progress.
        assert!(!window_is_contracting(
            &[1.0, Number::NAN, 1e-9, 1e-12],
            1e20
        ));
    }

    /// gh #534 edge cases: the ratio must never be evaluated against a
    /// non-positive or non-finite predecessor.
    #[test]
    fn degenerate_windows_never_read_as_contracting() {
        let r = DEFAULT_DECLINE_PROGRESS_RATIO;
        // A zero predecessor makes the ratio meaningless (0 <= 0.5*0 would
        // otherwise read as "contracting" forever).
        assert!(!window_is_contracting(&[0.0, 0.0, 0.0, 0.0], r));
        assert!(!window_is_contracting(&[1e-9, 0.0, 0.0, 0.0], r));
        assert!(!window_is_contracting(
            &[Number::INFINITY, 1e-3, 1e-6, 1e-9],
            r
        ));
        assert!(!window_is_contracting(&[1e-3, 1e-6, 1e-9, Number::NAN], r));
        // A genuine run down to exactly zero is progress, not a degenerate
        // window — the predecessor is positive at every step.
        assert!(window_is_contracting(&[1e-3, 1e-6, 1e-9, 0.0], r));
    }

    /// gh #534: the window slides one sample per outer iteration and holds the
    /// most recent [`DECLINE_PROGRESS_SAMPLES`]. A short history is never a full
    /// window, which is what stops the first restoration entry of a solve from
    /// being deferred on no evidence at all — `nlp_err_contracting` requires
    /// `len == DECLINE_PROGRESS_SAMPLES` before it consults the samples.
    #[test]
    fn progress_window_slides_oldest_out() {
        let mut buf = [Number::NAN; DECLINE_PROGRESS_SAMPLES];
        let mut len = 0usize;
        for e in [1e-1, 1e-2, 1e-3] {
            push_sample(&mut buf, &mut len, e);
        }
        assert_eq!(len, 3);
        push_sample(&mut buf, &mut len, 1e-4);
        assert_eq!(len, DECLINE_PROGRESS_SAMPLES);
        assert_eq!(buf, [1e-1, 1e-2, 1e-3, 1e-4]);
        assert!(window_is_contracting(&buf, DEFAULT_DECLINE_PROGRESS_RATIO));
        // One flat iteration slides the oldest sample out and withdraws the
        // verdict.
        push_sample(&mut buf, &mut len, 1e-4);
        assert_eq!(len, DECLINE_PROGRESS_SAMPLES);
        assert_eq!(buf, [1e-2, 1e-3, 1e-4, 1e-4]);
        assert!(!window_is_contracting(&buf, DEFAULT_DECLINE_PROGRESS_RATIO));
    }

    /// gh #505: no route may conclude `LocalInfeasibility` on its own.
    ///
    /// Three routes reach that verdict, and two of them independently shipped
    /// the same defect — building the terminate outcome directly, so the
    /// acceptable-point stash was never consulted and a good point the solve
    /// already had in hand was discarded. They were found one at a time,
    /// because nothing tied them together.
    ///
    /// The route a solve takes is an internal detail; the user sees one status
    /// either way. So what that status means is decided in one place — every
    /// route goes through [`IpoptAlgorithm::terminate_local_infeasibility`],
    /// or, for the cycle exits whose fallback is chosen between two statuses
    /// at the call site, through `terminate_acceptable_or`. Both consult the
    /// stash.
    ///
    /// **This is a tripwire, not a proof.** It is a substring scan of this
    /// file's source for the bare `IterateOutcome::Terminate(SolverReturn::
    /// LocalInfeasibility)` construction. A rustfmt line break through that
    /// expression, a `let` binding for the status, or a construction in
    /// another module all evade it — `application.rs` names the same variant
    /// on the SQP and ℓ₁ elastic paths and is deliberately out of scope. What
    /// it does buy is that the *obvious* way to add a fourth bare exit here
    /// fails loudly and points at the helper, which is the mistake that was
    /// actually made twice.
    ///
    /// The needle is assembled at runtime so this test's own source cannot
    /// satisfy the pattern it is checking for; an earlier version counted its
    /// own lines and failed against clean code.
    #[test]
    fn no_route_concludes_local_infeasibility_alone() {
        let needle = format!(
            "IterateOutcome::Terminate(SolverReturn::{})",
            "LocalInfeasibility"
        );
        let offenders: Vec<usize> = include_str!("ipopt_alg.rs")
            .lines()
            .enumerate()
            .filter(|(_, l)| {
                let t = l.trim_start();
                !t.starts_with("//") && !t.starts_with("///")
            })
            .filter(|(_, l)| l.contains(&needle))
            .map(|(i, _)| i + 1)
            .collect();
        assert!(
            offenders.is_empty(),
            "line(s) {offenders:?} build the local-infeasibility verdict directly. \
             Call `terminate_local_infeasibility()` instead — it consults the \
             acceptable-point stash first, so a solve that already passed through an \
             acceptable iterate returns that point rather than a hard failure. Two \
             routes shipped this bug before the helper existed (gh #505)."
        );
    }
}
