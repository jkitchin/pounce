//! User-facing application object — port of `Interfaces/IpIpoptApplication.{hpp,cpp}`.
//!
//! # Crate placement
//!
//! `IpoptApplication` lives in `pounce-algorithm` (rather than
//! alongside the other Interfaces-side ports in `pounce-nlp`) because
//! `optimize_tnlp` needs to drive the full IPM: it constructs a
//! `TNLPAdapter` + `OrigIpoptNlp` (from `pounce-nlp`) and hands the
//! NLP off to an [`IpoptAlgorithm`] (this crate). `pounce-nlp` cannot
//! depend on `pounce-algorithm` (the reverse already exists), so
//! orchestration must live on the algorithm side. Public callers
//! continue to import via `pounce_algorithm::IpoptApplication`.
//!
//! `optimize_tnlp` routes every problem — constrained or not —
//! through the same primal-dual IPM, exactly as upstream Ipopt does:
//! it builds the algorithm via [`crate::alg_builder::AlgorithmBuilder`]
//! (default backend MA57 from `pounce-hsl`) and runs
//! [`IpoptAlgorithm::optimize`].

use crate::alg_builder::{
    AlgorithmBuilder, HessianApproxChoice, LineSearchChoice, LinearBackendFactory,
    LinearSolverChoice, MuStrategyChoice,
};
use crate::hess::lim_mem_quasi_newton::UpdateType;
use crate::ipopt_alg::{DUAL_DIV_RETRY_DU_FLOOR, IpoptAlgorithm};
use crate::ipopt_cq::IpoptCalculatedQuantities;
use crate::ipopt_data::IpoptData as AlgIpoptData;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::IteratesVector;
use crate::restoration::RestorationPhase;
use crate::upstream_options::register_all_upstream_options;

/// Options-file names probed in the working directory when the caller
/// names none, in probe order: pounce's own name first, then upstream's
/// so an `ipopt.opt` written for Ipopt is honored unchanged.
///
/// Upstream probes only `ipopt.opt` (the registered default of
/// `option_file_name`). Both are read here because a port that answers
/// to `ipopt.opt` but not to its own name is the more surprising of the
/// two behaviours — and gh#518 reported trying both.
pub const DEFAULT_OPTION_FILE_NAMES: &[&str] = &["pounce.opt", "ipopt.opt"];

/// gh#887 — how far the *runaway* must dominate everything else in the
/// answer a solve finally reports, before
/// [`IpoptApplication::run_with_dual_divergence_retry`] will spend a cold
/// re-solve on it.
///
/// gh#884's defect is a point that is converged **except** that one
/// multiplier ran away: the primal is exact, complementarity is met, and
/// the entire residual is dual infeasibility. That is the shape
/// `perturb_always_cd` repairs. A point whose other residuals are within
/// a few orders of its dual one is not that — it is an ordinary
/// unconverged answer, and re-solving it cold is what
/// `mu_strategy_fallback` and the second-opinion ladder already are.
///
/// So the retry requires `max(viol, compl) <= 1e-6 * dual_inf`, all
/// unscaled. Measured on every run in the corpus that reaches the test:
///
/// | run | dual inf | viol | compl | ratio |
/// |---|---|---|---|---|
/// | reproducer, `.nl` route | `7.90e4` | `1.1e-16` | `1.1e-9` | `1.5e-14` |
/// | reproducer, TNLP route | `3.25e11` | `2.5e-16` | `2.8e-3` | `8.7e-15` |
/// | `deb7` + L-BFGS + rung | `9.90e1` | `8.0e-13` | `4.65e0` | `4.7e-2` |
///
/// Twelve orders between the keeps and the reject, and `1e-6` leaves
/// eight orders of margin on the tightest keep and four on the reject.
///
/// It is a **ratio of two residuals of the same answer**, so it carries
/// no units and does not move with the model's scaling — and, unlike any
/// test on the trajectory, it cannot depend on which attempt fired or on
/// how a platform rounded its way there. That is not hypothetical: the
/// first version of this gate compared the reported answer against the
/// runaway the detector had seen, and `deb7`'s detector value is `9.2e5`
/// on one attempt and `8.7e2` on another, differing between build
/// profiles and again on CI's Linux runner, where the retry ran anyway.
///
/// Deliberately a constant and not an option. It does not express a
/// tolerance a caller trades against — it says "the runaway is the whole
/// residual" — and the escape hatch for the remedy is
/// `dual_divergence_retry=no`.
const DUAL_DIV_RETRY_DOMINANCE: Number = 1e-6;

/// Does this answer have gh#884's *shape*?
///
/// gh#884's defect is a point converged **except** that one multiplier
/// ran away: the primal is exact, complementarity is met, and the entire
/// residual is dual infeasibility. `perturb_always_cd` has something to
/// repair only there, so this is what opens the retry (gh#887).
///
/// All three arguments are in the **model's own units**, never the
/// `s_d`-normalised frame the convergence gate reads — that frame is what
/// hid gh#884 in the first place. The test is a ratio *within one
/// answer*, so it carries no units and cannot depend on which attempt
/// produced it or on how a platform rounded. That is the property that
/// matters, not the margin; see the module tests for what happened to
/// the two gates that did not have it.
///
/// Non-finite input disables the retry rather than enabling it: a NaN
/// compares false everywhere, and writing this so a NaN *passed* would
/// turn "we cannot tell" into "retry anyway". A non-positive dual
/// residual cannot be a runaway either, and makes the ratio meaningless.
///
/// **The dominance ratio alone is not the whole test, and reading it as
/// one is how a `dual_inf` of `0.44` opened a retry.** The ratio says the
/// dual residual *dominates* the other two; it cannot say the residual is
/// large, because a point converged to `1e-30` primal and `4.4e-1` dual
/// passes it as comfortably as gh#884's `7.9e+04` does. So `dual_inf`
/// must also clear the same absolute floor the detector's third conjunct
/// applies to the iterate (`dual_divergence_retry_du_floor`, default
/// `1e2`) — the doc above says "the entire residual is dual
/// infeasibility", and without the floor the code only said "the largest
/// third of it is".
///
/// The floor is the *detector's own*, deliberately, rather than a new
/// constant: the detector fires on an iterate and this asks whether the
/// **answer** still exhibits what the detector saw, so the two have to be
/// asking about the same magnitude or the answer-level gate is a strictly
/// looser copy of the iterate-level one. Measured on the 400-model QPEC
/// family in `dev-notes/mpcc-biactive-dual-divergence.md`, this alone
/// removes 7 of 68 promotions, every one of them on an answer whose
/// reported dual residual was below `1e2` and therefore not a runaway by
/// the issue's own description.
fn runaway_is_the_whole_residual(
    dual_inf: Number,
    viol: Number,
    compl: Number,
    du_floor: Number,
) -> bool {
    dual_inf.is_finite()
        && dual_inf > 0.0
        && dual_inf >= du_floor
        && viol.is_finite()
        && compl.is_finite()
        && viol.max(compl) <= DUAL_DIV_RETRY_DOMINANCE * dual_inf
}

/// Is the retry's answer admissible *as an answer*, next to the base
/// attempt's — independent of which has the better multiplier?
///
/// gh#884's promotion gate ranked the two attempts on unscaled KKT error
/// alone. That is a statement about the **certificate**, and it was
/// allowed to decide which **point** shipped, on the argument that
/// "conjunct 4 requires the promoted answer to satisfy the KKT conditions
/// in the model's own units" — so, unlike the μ flip, this retry could not
/// return a different local solution. The inference does not hold: *any*
/// other KKT point satisfies the KKT conditions in the model's own units
/// too. Measured on 400 random QPECs (`prod_eq` lowering), 42 of 68
/// promotions moved the objective materially, i.e. returned a different
/// local solution, and three returned a **worse feasible point** — worst
/// case `-13.0057 → -1.2072`, both independently verified feasible.
///
/// Two rules, and both are about the *answer* rather than its certificate:
///
/// 1. **Never hand back a feasible point whose objective is worse than one
///    this run already computed.** The base attempt's point is feasible
///    and in hand; returning a worse one is a regression no certificate
///    buys back.
/// 2. **An objective *improvement* may not be bought with primal slack.**
///    Two costs this carries, named rather than hidden (R3). It is an exact
///    non-increase with **no noise floor**, so it fires the same on
///    `2.07e-25 -> 1.09e-09` (where the move is the whole defect) as on
///    `1e-17 -> 1.1e-17` (where which side a retry lands on is arithmetic):
///    of the 45 promotions this PR removes, ~35 are improvements refused
///    because the violation ticked up somewhere far below any tolerance. The
///    direction is conservative — the base answer is feasible and in hand —
///    but "refuses purchases, not improvements" is only true above the noise.
///    And the rule detects *the primal moved*, not *below the optimum*: had
///    `scholtes4`'s retry **held** its violation, `f = -6.6088e-05` would have
///    been admitted, which the second assertion of
///    `an_improvement_bought_with_primal_slack_is_refused` states outright.
///    On that model the two coincided.
///    The remedy is for a *dual* defect — the premise is that the primal
///    has settled — so a retry that lands further outside the constraints
///    *and* reports a better objective has not repaired the runaway, it
///    has moved somewhere the model does not reach. This is what
///    `scholtes4` does: from a base at `f = +1.82e-09` with a constraint
///    violation of `2.07e-25` it promotes `f = -6.61e-05` at `1.09e-09`,
///    below the model's exactly-known `f* = 0`, and reports
///    `Optimal Solution Found`.
///
/// Both comparisons are skipped when the base attempt is not itself
/// feasible within [`IpoptApplication::dual_divergence_retry_accept_tol`]:
/// there is then no admissible point to protect, and the retry's is
/// strictly better information.
///
/// `tol` is that same acceptable tolerance, scaled by
/// `max(1, |base_obj|)` so the comparison is relative on a large
/// objective and absolute on a small one — the convention
/// `sigma_forward_error_is_small` uses for `‖x‖` and for the same reason.
/// It is not a fitted constant: the objective moves this has to admit
/// (`qpec_small`, `5.8e-11`) and the ones it has to refuse (`r201`,
/// `0.198`; `scholtes4`, `6.6e-05`) sit four and five orders away from it
/// on either side.
///
/// `sense` is `+1` for a minimization and `-1` for a maximization, and both
/// objectives are multiplied by it before either rule looks at them.
/// [`SolveStatistics::final_objective`] is the objective evaluated on the
/// **user** TNLP — signed, and *not* premultiplied by `obj_scaling_factor` —
/// and a negative `obj_scaling_factor` is the documented way to pose a
/// maximization. Without the normalization both rules invert under it: rule 1
/// would refuse genuine *improvements* (a regression against the behaviour
/// before this conjunct existed) and rule 2 would admit strictly worse
/// answers, re-arming the exact class the conjunct was added to block. The
/// repo has shipped one defect from this sign already —
/// `masked_certificate_fuzz.rs::the_veto_is_not_disabled_by_a_negative_objective_scaling_factor`,
/// whose fix took `.abs()` in the residual accessors, which is also what keeps
/// the detector firing under maximization and so keeps this path reachable.
fn retry_answer_is_admissible(
    base_obj: Number,
    base_viol: Number,
    retry_obj: Number,
    retry_viol: Number,
    accept_tol: Number,
    sense: Number,
) -> bool {
    // Nothing to compare against: a non-finite or infeasible base answer
    // is not a point worth protecting.
    if !base_obj.is_finite() || !base_viol.is_finite() || base_viol > accept_tol {
        return true;
    }
    // A non-finite retry objective is refused rather than admitted, for
    // the same reason `runaway_is_the_whole_residual` refuses a NaN.
    if !retry_obj.is_finite() {
        return false;
    }
    // Both into a minimization frame, so "lower is better" below is true
    // whichever sense the caller posed.
    let base_obj = sense * base_obj;
    let retry_obj = sense * retry_obj;
    let tol = accept_tol * base_obj.abs().max(1.0);
    if retry_obj > base_obj + tol {
        return false; // rule 1: strictly worse feasible point
    }
    if retry_obj < base_obj - tol {
        // Rule 2: an improvement is admissible only if the retry did not
        // give up primal accuracy to get it.
        return retry_viol.is_finite() && retry_viol <= base_viol;
    }
    true
}

/// What [`IpoptApplication::initialize_with_option_file`] did — enough
/// for a caller to tell the user which file (if any) configured the run.
#[derive(Debug, Default, Clone)]
pub struct OptionFileLoad {
    /// The file actually read. `None` means no options file was read:
    /// nobody named one and neither default was present.
    pub path: Option<PathBuf>,
    /// Whether [`Self::path`] was named by the caller rather than found
    /// by probing the working directory.
    pub explicit: bool,
    /// Non-fatal notes about option-file settings that did *not* take
    /// effect. Nothing here stops a solve; the point is that it not
    /// happen silently.
    pub warnings: Vec<String>,
}

/// Factory that constructs a fresh restoration-phase strategy on
/// demand. The outer algorithm owns at most one restoration object,
/// so the factory is invoked once per `optimize_tnlp` call. The
/// factory is `FnMut` to allow callers to capture a builder that
/// internally reuses caches across builds.
pub type RestorationFactory = Box<dyn FnMut() -> Box<dyn RestorationPhase>>;

/// Provider that mints fresh [`RestorationFactory`] instances on
/// demand. Used by drivers that need to run the inner IPM more than
/// once per `optimize_tnlp` call — notably the Phase-3 ℓ₁-exact
/// penalty-barrier outer loop (pounce#10), which the existing
/// `RestorationFactory` cannot support because pounce's default
/// `make_default_restoration_factory` is a one-shot. Callers wire
/// this via [`IpoptApplication::set_restoration_factory_provider`].
pub type RestorationFactoryProvider = Box<dyn FnMut() -> RestorationFactory>;

/// Callback fired by [`IpoptApplication::optimize_constrained`] once
/// the IPM has converged (status `SolveSucceeded` or
/// `SolvedToAcceptableLevel`) and before the user TNLP's
/// `finalize_solution` runs. Receives borrowed handles into the
/// algorithm's converged state.
///
/// **Use case**: post-optimal sensitivity analysis (pounce#7 /
/// `pounce-sensitivity`). The callback receives a shared handle to
/// the PD solver so a `SensBacksolver` adapter can run backsolves
/// against the converged KKT factor — and so that handle may outlive
/// the call frame (e.g. the public `Solver` session API retains the
/// factor for repeated `parametric_step` / `kkt_solve` calls);
/// receives the data / cq / nlp handles so the adapter can reproduce
/// the augmented-system coefficient layout the IPM converged at.
///
/// **Not** the same as `set_intermediate_callback` (per-iteration
/// progress notification) — this fires exactly once per `optimize_*`
/// call, only on success.
pub type ConvergedCallback = Box<
    dyn FnMut(
        &crate::ipopt_data::IpoptDataHandle,
        &crate::ipopt_cq::IpoptCqHandle,
        &Rc<RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>,
        Rc<RefCell<crate::kkt::pd_full_space_solver::PdFullSpaceSolver>>,
    ),
>;
use pounce_common::diagnostics::DiagnosticsState;
use pounce_common::exception::{ExceptionKind, SolverException};
use pounce_common::journalist::{JournalLevel, Journalist};
use pounce_common::options_list::OptionsList;
use pounce_common::reg_options::{PrintOptionsMode, RegisteredOptions};
use pounce_common::timing::TimingStatistics;
use pounce_common::types::{Index, Number};
use pounce_linalg::dense_vector::DenseVectorSpace;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_linsol::summary::LinearSolverSummary;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::derivative_test::{DerivativeTest, DerivativeTestOptions};
use pounce_nlp::orig_ipopt_nlp::{ConstObjScaling, OrigIpoptNlp, ScalingMethod};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{
    BoundsInfo, IpoptCq as TnlpIpoptCq, IpoptData as TnlpIpoptData, NlpInfo, Solution, TNLP,
};
use pounce_nlp::tnlp_adapter::{
    DEFAULT_NLP_LOWER_BOUND_INF, DEFAULT_NLP_UPPER_BOUND_INF, FixedVarTreatment, TNLPAdapter,
};
use std::cell::RefCell;
use std::fmt;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct IpoptApplication {
    options: OptionsList,
    /// Diagnostics from the safeguarded `least_square_init_primal`
    /// initializer step of the most recent solve (gh#605). `None` when
    /// the step was not run. Read with
    /// [`Self::least_square_init_report`].
    least_square_init_report: Option<crate::init::default::LeastSquareInitReport>,
    /// Per-variable scaling factors applied by the wrapper installed in
    /// [`Self::optimize_tnlp`] (gh#486). Recorded so consumers that read
    /// the algorithm's own iterate rather than the `finalize_solution`
    /// payload — the CLI's `on_converged` hook feeding the `.sol` and
    /// the JSON report — can undo the substitution. `None` when no
    /// variable scaling was applied.
    variable_scaling: RefCell<Option<Vec<Number>>>,
    /// Whether the most recent `optimize_constrained` ran with per-row
    /// constraint scaling active (`c_scale_vec`/`d_scale_vec` present).
    ///
    /// Read by [`Self::run_l1_penalty_outer_loop`], which measures the
    /// user's rows in the model's own units and so may only write that
    /// number into the `final_unscaled_*` family. `SolveStatistics`
    /// documents `final_*` as the internally-scaled residuals and
    /// `final_unscaled_*` as the same quantities in original units,
    /// *equal when no scaling is active* — so the measurement may be
    /// mirrored into the scaled family exactly when this is `false`
    /// (gh#794 review). `None` when no solve has run.
    row_scaling_active: std::cell::Cell<Option<bool>>,
    /// Whether the submitted TNLP has already been explicitly wrapped by the
    /// caller's presolve layer.
    presolve_already_applied: bool,
    reg_options: Rc<RegisteredOptions>,
    journalist: Rc<Journalist>,
    statistics: RefCell<SolveStatistics>,
    /// Shared per-subsystem timing accumulator. Re-created at the top of
    /// every solve (so back-to-back `optimize_tnlp` calls don't bleed
    /// timings across invocations) and handed to the data, the NLP, and
    /// any other consumer via `Rc`. Reported by [`Self::timing_stats`]
    /// after the solve completes.
    timing: RefCell<Rc<TimingStatistics>>,
    /// Optional override factory for the symmetric linear-solver
    /// backend. When `None`, we ship the workspace default (MA57 via
    /// `pounce-hsl`). Tests can plug a stub via [`Self::set_linear_backend_factory`].
    linear_backend_factory: Option<LinearBackendFactory>,
    /// Optional factory for the restoration phase. Lives outside this
    /// crate because `pounce-algorithm` cannot depend on
    /// `pounce-restoration` (the dep edge is the other way). Callers
    /// that need restoration plug a factory via
    /// [`Self::set_restoration_factory`]; when unset, the outer
    /// algorithm runs without a restoration fallback and surfaces
    /// `RestorationFailure` as soon as the line-search would otherwise
    /// jump into restoration.
    restoration_factory: Option<RestorationFactory>,
    /// Shared diagnostic-dump state, installed by the CLI when the
    /// user passes `--dump <cat>:<spec>`. When set, the application
    /// propagates an `Rc<DiagnosticsState>` into [`IpoptAlgorithm`]
    /// via [`IpoptAlgorithm::with_diagnostics`] so the KKT solver and
    /// other dump sites can consult per-iter gating.
    diagnostics: Option<Rc<DiagnosticsState>>,
    /// Optional interactive debugger hook. When set, it is moved into
    /// the main [`IpoptAlgorithm`] for the next `optimize_*` call via
    /// [`IpoptAlgorithm::with_debug_hook`], so a REPL or agent can pause
    /// at each iteration to inspect / mutate live state. Consumed on use
    /// (one solve per installed hook).
    debug_hook: Option<std::rc::Rc<std::cell::RefCell<dyn crate::debug::DebugHook>>>,
    /// Provider for the BNW outer loop (pounce#10 Phase 3). When set,
    /// `optimize_constrained` consults the provider before each inner
    /// solve, replacing `restoration_factory` with a fresh one so
    /// multi-pass drivers can run the inner IPM repeatedly without
    /// tripping the default factory's one-shot guard.
    restoration_factory_provider: Option<RestorationFactoryProvider>,
    /// Optional hook fired once per `optimize_*` call on convergence,
    /// before the user TNLP's `finalize_solution`. See
    /// [`ConvergedCallback`].
    on_converged: Option<ConvergedCallback>,
    /// When `true`, the per-iteration `IterRecord` trajectory is
    /// captured into [`SolveStatistics::iterations`] for downstream
    /// consumers (the JSON solve report in pounce-cli, pounce#8). Off
    /// by default so library callers that never read the iterations
    /// vector don't pay the per-iter alloc.
    record_iter_history: bool,
    /// Whether [`Self::initialize_with_option_file`] ran — i.e. whether
    /// anything on this application actually consulted
    /// `option_file_name` and resolved it to a file. Only the `pounce`
    /// CLI does; a library caller sets its options directly. The guard
    /// in [`Self::unhonored_option_file_name`] reads this so that
    /// setting the option on a surface that cannot honor it is refused
    /// rather than dropped (gh#518).
    option_file_resolved: bool,
    /// Whether this caller can route a model to the convex LP/QP/SOCP
    /// engines — i.e. whether the `qp_*` knobs those engines read
    /// configure anything here. Only the `pounce` CLI can (it owns the
    /// `.nl` structure extraction that classifies a model), and it says
    /// so via [`Self::set_convex_routing_available`]. The guard in
    /// [`Self::unhonored_convex_option`] reads this, on the same
    /// contract as `option_file_resolved` above (gh#604).
    convex_routing_available: bool,
    /// Whether the backend-knob warnings (gh#551) have already been
    /// printed for this application. The CLI emits them before routing —
    /// a convex model never reaches `optimize_tnlp` — and `optimize_tnlp`
    /// emits them for every other frontend; without this flag a CLI run
    /// would print each line twice, which is how a warning teaches its
    /// reader to skip it.
    backend_warnings_emitted: bool,
    /// Shared sink that the linear-solver backend writes a rolling
    /// [`LinearSolverSummary`] into after every factor. Reset at the
    /// top of every solve (so back-to-back `optimize_tnlp` calls don't
    /// bleed stats across invocations) and read out via
    /// [`Self::linear_solver_summary`] once the solve returns. Only
    /// the workspace-default FERAL backend (via
    /// [`default_backend_factory_with_sink`]) wires the sink today;
    /// custom factories plugged through [`Self::set_linear_backend_factory`]
    /// and the HSL MA57 backend leave the sink empty.
    linsol_summary_sink: Arc<Mutex<LinearSolverSummary>>,
    /// Shared tally of successful linear-solver quality escalations for
    /// the current solve (gh#857). Handed to every `AlgorithmBuilder`
    /// this application mints — [`Self::algorithm_builder_from_options`]
    /// and [`Self::algorithm_builder_snapshot`] — which is how the
    /// restoration sub-solve counts into the same total: each frontend
    /// builds the restoration provider's inner builder from
    /// `algorithm_builder_from_options`, so it receives this same `Rc`
    /// without any frontend having to know the counter exists.
    ///
    /// Reset at the top of every solve, beside the linear-solver summary
    /// sink and for the same reason: a second-opinion retry, or two
    /// back-to-back `optimize_tnlp` calls, must not inherit the previous
    /// solve's count. Read out into
    /// [`SolveStatistics::quality_escalations`] once the solve returns.
    quality_escalations: Rc<std::cell::Cell<u64>>,
    /// gh#884. Set when the running [`IpoptAlgorithm`] observed the
    /// biactive dual-divergence signature — a converged primal, a step
    /// that has gone to zero on a scale-relative measure, and an
    /// *unscaled* dual infeasibility still far above `dual_inf_tol`, all
    /// at the same iterate.
    ///
    /// Read out of the algorithm once `optimize_constrained` returns, so
    /// [`Self::run_with_dual_divergence_retry`] — which sits above that
    /// call — can see it. Reset at the top of every solve for the same
    /// reason as `quality_escalations`: a retry must not inherit the
    /// previous attempt's verdict. Also copied into
    /// [`SolveStatistics::dual_divergence_signature`].
    dual_divergence_signature: std::cell::Cell<bool>,
    /// gh#884. Set when a dual-divergence retry actually replaced the base
    /// attempt's answer. Copied into
    /// [`SolveStatistics::dual_divergence_retry_promoted`].
    dual_divergence_retry_promoted: std::cell::Cell<bool>,
    /// Set when a losing retry's answer was thrown away and an earlier
    /// attempt's replayed through `FinalizeSnapshot::replay` — by the μ
    /// fallback (pounce#870) or the gh#884 dual-divergence retry.
    ///
    /// It exists because [`Self::set_on_converged`] fires **per attempt**
    /// and the floor does not reach it. A losing retry that reached
    /// `Solve_Succeeded` has already run the callback, so a consumer
    /// capturing the converged iterate there — the CLI's `nominal_capture`,
    /// which is where `.sol` `x`, the JSON's `solution.x` and the dual
    /// block all come from — holds the *discarded* attempt's point while
    /// the status, objective and every statistic beside it have been
    /// floored back to the attempt that won. Measured before this flag
    /// existed: on a declined retry the `.sol` carried `f = -6.3274` while
    /// the JSON report next to it said `-6.1768`, and `pounce verify` on
    /// the `.sol` confirmed the file held the losing point.
    ///
    /// The `finalize_solution` payload does not have this problem — the
    /// replay *is* a `finalize_solution` call, so the last one always
    /// carries the answer being reported — which is why the fix is to tell
    /// the caller "prefer that payload", not to re-run the callback (the
    /// converged KKT state the callback borrows belongs to the retry by
    /// then, and cannot be rewound).
    answer_restored_from_floor: std::cell::Cell<bool>,
    /// The payload of the most recent `finalize_solution` POUNCE sent to the
    /// user's TNLP, so a second-opinion retry that loses can put the winning
    /// attempt's answer back (pounce#870).
    ///
    /// Written by [`finalize_via_orig_nlp`] and [`finalize_via_sqp`], which are
    /// free functions and so take this as a sink rather than reaching for
    /// `self`. Read only by [`Self::run_with_mu_strategy_fallback`].
    last_finalize: RefCell<Option<FinalizeSnapshot>>,
    /// The last `IterStats` sent to the user's `intermediate_callback`
    /// (pounce#870), shared with the running `IpoptAlgorithm`.
    last_iter_stats: Rc<RefCell<Option<pounce_nlp::tnlp::IterStats>>>,
    /// Phase 5c (§6) SQP warm-start input. When `Some`, the next
    /// `optimize_tnlp` call on the SQP path consumes the iterate
    /// instead of cold-starting; consumed once per solve, then
    /// auto-cleared. The IPM path ignores this field. Wire-set
    /// via [`Self::set_sqp_warm_start`].
    sqp_warm_start: Option<crate::sqp::SqpIterates>,
    /// Phase 5c (§6) SQP warm-start output. Populated by every
    /// `optimize_sqp_tnlp` call with the final QP working set.
    /// Stays valid until the next solve (which overwrites it).
    /// Accessed via [`Self::last_sqp_working_set`].
    sqp_last_working_set: Option<pounce_qp::WorkingSet>,
    /// What the post-convergence crossover phase did on the most recent
    /// IPM solve (gh#612). `None` when `crossover=no` (the default) or
    /// when the solve did not converge — crossover only runs on a
    /// converged interior iterate. Read via [`Self::crossover_report`].
    ///
    /// Kept separate from `sqp_last_working_set`, which crossover *also*
    /// populates on success: that field answers "what can the next solve
    /// warm-start from", this one answers "was the active set I am about
    /// to trust actually identified, or merely inferred".
    crossover_report: Option<crate::crossover::CrossoverReport>,
    /// Full primal-dual warm-start iterate for the IPM path, captured by
    /// the interactive debugger's `resolve` command. When `Some`, the
    /// next `optimize_tnlp` installs this 8-vector (algorithm space)
    /// directly onto `data.curr` before the iterate initializer runs, so
    /// a warm `resolve` continues from the paused interior point rather
    /// than cold-restarting the duals. Consumed once per solve, then
    /// auto-cleared. Requires `warm_start_init_point=yes` so the
    /// re-optimize branch of `WarmStartIterateInitializer` keeps the
    /// installed iterate. Wire-set via [`Self::set_warm_start_iterate`].
    warm_start_iterate: Option<crate::debug::IterateSnapshot>,
    /// The warm-start initializer's verdict on the most recent solve's
    /// supplied iterate (gh#606). Lifted off the solve-local
    /// `IpoptData` so a caller can read it after `optimize_tnlp`
    /// returns; `None` when the last solve was a cold start.
    warm_start_diag: RefCell<Option<crate::init::warm_start::WarmStartDiagnostics>>,
    /// Caller-supplied fill-reducing permutation for the KKT linear
    /// solver (pounce#180 item 1 / FERAL#107). When `Some`, it overrides
    /// whatever `feral_ordering` / `POUNCE_FERAL_ORDERING` resolves to,
    /// installing [`pounce_feral::OrderingMethod::External`] on the FERAL
    /// backend for the next solve. The vector is a **0-based, new-to-old
    /// permutation** whose length must equal the augmented KKT system
    /// dimension; FERAL validates it as a bijection and returns
    /// `InvalidInput` (never panics) on a wrong length / index. Unlike the
    /// warm-start hooks this is treated as persistent config — it is *not*
    /// auto-cleared after a solve, so a caller sets it once for a run.
    /// Wire-set via [`Self::set_external_ordering`]. Ignored by non-FERAL
    /// backends and by any custom factory plugged via
    /// [`Self::set_linear_backend_factory`].
    external_ordering: Option<Vec<usize>>,
    /// Caller-supplied block-triangular / Schur KKT partition (pounce#180
    /// item 2). When `Some`, the next IPM solve on the feral + exact-Hessian
    /// path routes the KKT linear solve through a
    /// [`crate::kkt::SchurAugSystemSolver`] over these **KKT-space indices**
    /// (`0..dim` in the `x, s, c, d` block order the aug-system solver
    /// assembles): the `S` block is Schur-complemented out and only the two
    /// diagonal blocks are factorized, with inertia recovered via Sylvester's
    /// law. Beneficial only when `|S| ≪` the eliminated block; the Schur solver
    /// falls back to the standard full-space solver transparently when the
    /// partition is unsuitable (too large, malformed, or the backend errors),
    /// so a stray hook never breaks a solve. Persistent config (not
    /// auto-cleared). Wire-set via [`Self::set_kkt_schur_block`].
    kkt_schur_block: Option<Vec<usize>>,
}

impl fmt::Debug for IpoptApplication {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IpoptApplication")
            .field("options", &self.options)
            .field("statistics", &self.statistics)
            .finish_non_exhaustive()
    }
}

impl Default for IpoptApplication {
    fn default() -> Self {
        Self::new()
    }
}

impl IpoptApplication {
    /// New application with empty options and a default journalist.
    /// Equivalent to `IpoptApplication::IpoptApplication(true,true)`.
    pub fn new() -> Self {
        let reg = RegisteredOptions::default();
        // Registration of a fresh registry can only fail on a duplicate
        // name, which would be a programming error in `reg_op`.
        register_all_upstream_options(&reg)
            .unwrap_or_else(|e| panic!("Upstream options registration failed: {e}"));
        pounce_presolve::register_options(&reg)
            .unwrap_or_else(|e| panic!("Presolve options registration failed: {e}"));
        let reg = Rc::new(reg);
        Self {
            options: OptionsList::with_registered(Rc::clone(&reg)),
            least_square_init_report: None,
            variable_scaling: RefCell::new(None),
            row_scaling_active: std::cell::Cell::new(None),
            presolve_already_applied: false,
            reg_options: reg,
            journalist: Rc::new(Journalist::new()),
            statistics: RefCell::new(SolveStatistics::new()),
            timing: RefCell::new(Rc::new(TimingStatistics::new())),
            linear_backend_factory: None,
            restoration_factory: None,
            diagnostics: None,
            debug_hook: None,
            restoration_factory_provider: None,
            on_converged: None,
            record_iter_history: false,
            option_file_resolved: false,
            convex_routing_available: false,
            backend_warnings_emitted: false,
            linsol_summary_sink: Arc::new(Mutex::new(LinearSolverSummary::default())),
            quality_escalations: Rc::new(std::cell::Cell::new(0)),
            dual_divergence_signature: std::cell::Cell::new(false),
            dual_divergence_retry_promoted: std::cell::Cell::new(false),
            answer_restored_from_floor: std::cell::Cell::new(false),
            last_finalize: RefCell::new(None),
            last_iter_stats: Rc::new(RefCell::new(None)),
            sqp_warm_start: None,
            sqp_last_working_set: None,
            crossover_report: None,
            warm_start_iterate: None,
            warm_start_diag: RefCell::new(None),
            external_ordering: None,
            kkt_schur_block: None,
        }
    }

    pub fn options(&self) -> &OptionsList {
        &self.options
    }

    pub fn options_mut(&mut self) -> &mut OptionsList {
        &mut self.options
    }

    /// Declare whether callers have already applied an explicit presolve
    /// wrapper to the TNLPs submitted to [`Self::optimize_tnlp`].
    ///
    /// When set, `optimize_tnlp` leaves its input TNLP unchanged even if the
    /// `presolve` option is enabled. This preserves the option table for
    /// reporting and debugger use while allowing specialized frontends to
    /// supply a wrapper with capabilities unavailable to generic callback
    /// TNLPs, such as an expression provider for FBBT.
    pub fn set_presolve_already_applied(&mut self, applied: bool) {
        self.presolve_already_applied = applied;
    }

    /// Solve without materializing the generic presolve wrapper.
    ///
    /// This is for consumers that require the original TNLP coordinate system
    /// for the solve's KKT matrix, such as sensitivity and reduced-Hessian
    /// drivers. It is scoped to this invocation and does not change the
    /// application's `presolve` option or persistent explicit-wrapper setting.
    pub fn optimize_tnlp_without_presolve(
        &mut self,
        tnlp: Rc<RefCell<dyn TNLP>>,
    ) -> ApplicationReturnStatus {
        let explicit_wrapper = self.presolve_already_applied;
        self.presolve_already_applied = true;
        let status = self.optimize_tnlp(tnlp);
        self.presolve_already_applied = explicit_wrapper;
        status
    }

    pub fn registered_options(&self) -> &Rc<RegisteredOptions> {
        &self.reg_options
    }

    pub fn journalist(&self) -> &Rc<Journalist> {
        &self.journalist
    }

    /// Plug a custom symmetric-linear-solver factory. Useful for tests
    /// that want to swap MA57 for a stub. Production callers should
    /// leave this unset — the default ([`default_backend_factory`])
    /// returns the workspace's MA57 binding.
    pub fn set_linear_backend_factory(&mut self, factory: LinearBackendFactory) {
        self.linear_backend_factory = Some(factory);
    }

    /// Plug a restoration-phase factory. Called once per
    /// `optimize_tnlp` invocation to mint a fresh
    /// `Box<dyn RestorationPhase>` that the outer algorithm uses as
    /// its line-search restoration fallback. Lives behind a setter
    /// (rather than at construction) because the concrete restoration
    /// strategies live in `pounce-restoration`, which depends on this
    /// crate; consumers in `pounce-cli` / integration tests wire the
    /// factory at the application boundary.
    pub fn set_restoration_factory(&mut self, factory: RestorationFactory) {
        self.restoration_factory = Some(factory);
    }

    /// Install the shared diagnostics state. Once set, every
    /// subsequent `optimize_tnlp` call forwards the state into the
    /// algorithm via [`IpoptAlgorithm::with_diagnostics`] so the KKT
    /// solver can emit `--dump kkt:...` artifacts.
    pub fn set_diagnostics(&mut self, diag: Rc<DiagnosticsState>) {
        self.diagnostics = Some(diag);
    }

    /// Install an interactive debugger hook for the next `optimize_*`
    /// call. The hook is moved into the main [`IpoptAlgorithm`] and
    /// consumed by that solve; reinstall it to debug a subsequent solve.
    pub fn set_debug_hook(
        &mut self,
        hook: std::rc::Rc<std::cell::RefCell<dyn crate::debug::DebugHook>>,
    ) {
        self.debug_hook = Some(hook);
    }

    /// Read-side accessor for the installed diagnostics state, if any.
    /// Lets the CLI write the top-level manifest/timing files after
    /// the solve completes.
    pub fn diagnostics(&self) -> Option<Rc<DiagnosticsState>> {
        self.diagnostics.as_ref().map(Rc::clone)
    }

    /// Plug a restoration-phase **factory provider** for drivers that
    /// need to run the inner IPM more than once per `optimize_tnlp`
    /// call (notably the Phase-3 ℓ₁-exact penalty-barrier outer loop,
    /// pounce#10). On each inner solve, the application consults the
    /// provider to mint a fresh [`RestorationFactory`], replacing any
    /// stale one, so the default one-shot restoration factory does
    /// not panic on its second invocation. If both `set_restoration_factory`
    /// and this are configured, the provider wins.
    pub fn set_restoration_factory_provider(&mut self, provider: RestorationFactoryProvider) {
        self.restoration_factory_provider = Some(provider);
    }

    /// Register a callback to run once the IPM has converged (status
    /// [`ApplicationReturnStatus::SolveSucceeded`] or
    /// [`ApplicationReturnStatus::SolvedToAcceptableLevel`]) but before
    /// `finalize_solution` flows back to the TNLP. See
    /// [`ConvergedCallback`] for the use case (post-optimal sensitivity).
    pub fn set_on_converged(&mut self, cb: ConvergedCallback) {
        self.on_converged = Some(cb);
    }

    /// Was the reported answer replayed from an earlier attempt's floor,
    /// after a later attempt lost?
    ///
    /// **A caller that captures the solution in [`Self::set_on_converged`]
    /// must consult this.** That callback fires once per *attempt*, and a
    /// losing retry that converged has already run it, so the capture
    /// belongs to the discarded point while everything else — status,
    /// objective, statistics — has been floored back. When this is `true`,
    /// take `x` and the multipliers from the last `finalize_solution`
    /// payload instead; that one is always the answer being reported,
    /// because the floor restores it *by* calling `finalize_solution`.
    ///
    /// The payload is in the **model's own units** and in the **reduced**
    /// presolve space: `CountingTnlp`-style consumers sit inside the gh#486
    /// scaling wrapper (so `x /= d`, `z *= d` have already been applied) and
    /// outside the presolve one (so the row/column lift has not). A caller
    /// swapping it in must therefore skip its own scaling correction and keep
    /// its own lift; getting that backwards squares the factor, which is a
    /// silent wrong answer of exactly the shape this flag exists to remove.
    ///
    /// **Not consulted, and known not to be**: the three other
    /// [`Self::set_on_converged`] consumers —
    /// `pounce-sensitivity/src/{solver,convenience}.rs` and
    /// `pounce-cli/src/minima/mod.rs` — read the converged KKT state and the
    /// factorization, which the `finalize_solution` payload does not carry
    /// and which cannot be rewound. After any floor replay their result
    /// describes the attempt that lost. Pre-existing, unfixed, and annotated
    /// at each site.
    ///
    /// `false` on every solve that never spent a second attempt, which is
    /// almost all of them, so the ordinary path is unaffected.
    pub fn answer_restored_from_floor(&self) -> bool {
        self.answer_restored_from_floor.get()
    }

    /// Enable per-iteration trajectory capture. After the solve
    /// returns, [`Self::statistics()`] exposes
    /// [`pounce_nlp::solve_statistics::SolveStatistics::iterations`]
    /// populated with one [`pounce_nlp::solve_statistics::IterRecord`]
    /// per accepted iterate. Off by default — the `pounce_sens` and
    /// `pounce` binaries opt in when `--json-output` is passed.
    pub fn enable_iter_history(&mut self) {
        self.record_iter_history = true;
    }

    /// Read the run's options file, resolving *which* file the way
    /// upstream's `IpoptApplication::Initialize` does — with one
    /// deliberate difference, below.
    ///
    /// `explicit` is the file the caller named (upstream: the
    /// `option_file_name` option, read out of the option store before
    /// this point). With `None`, the working directory is probed for
    /// [`DEFAULT_OPTION_FILE_NAMES`] and the first hit is read; an
    /// absent default file is not an error, it just means "no file".
    ///
    /// The difference: upstream opens a named file with a bare
    /// `std::ifstream` and reads nothing if the open fails, so a typo'd
    /// `option_file_name` runs at stock defaults without a word. That
    /// silence is what gh#518 was reported for — a benchmark that
    /// measured defaults while claiming to measure a configuration — so
    /// a named file that cannot be read is an error here.
    pub fn initialize_with_option_file(
        &mut self,
        explicit: Option<&Path>,
    ) -> Result<OptionFileLoad, SolverException> {
        let mut load = OptionFileLoad::default();
        // Set before the early returns below: what this flag records is
        // that `option_file_name` was *consulted*, not that a file turned
        // up. A caller on this path who names nothing and has no
        // `pounce.opt` to find still gets the option honored — there was
        // simply nothing to read.
        self.option_file_resolved = true;
        let path = match explicit {
            Some(p) => {
                if !p.is_file() {
                    return Err(SolverException::new(
                        ExceptionKind::IPOPT_APPLICATION_ERROR,
                        format!(
                            "options file \"{}\" does not exist. It was named by \
                             --options-file / option_file_name, so the run would \
                             otherwise proceed at stock defaults with none of its \
                             settings applied.",
                            p.display()
                        ),
                        file!(),
                        line!() as Index,
                    ));
                }
                load.explicit = true;
                p.to_path_buf()
            }
            None => {
                let present: Vec<&&str> = DEFAULT_OPTION_FILE_NAMES
                    .iter()
                    .filter(|n| Path::new(n).is_file())
                    .collect();
                let Some(first) = present.first() else {
                    return Ok(load);
                };
                // Both default names in one directory: say which one lost,
                // rather than let the unread one look applied.
                for other in &present[1..] {
                    load.warnings.push(format!(
                        "`{first}` and `{other}` are both present; reading `{first}` \
                         only (pounce's own name wins). Pass \
                         `option_file_name={other}` to read that one instead."
                    ));
                }
                PathBuf::from(**first)
            }
        };
        self.initialize_with_options_file(&path)?;
        // `option_file_name` set *inside* an options file chains nowhere —
        // by the time it is read, the file naming it has already been
        // chosen. Upstream documents that ("it does not make any sense to
        // specify this option within the options file") and then ignores
        // it; name it instead, since an ignored setting that looks live is
        // the whole complaint behind gh#518.
        if let Ok((named, true)) = self.options.get_string_value("option_file_name", "")
            && !named.is_empty()
            && Path::new(&named) != path
        {
            load.warnings.push(format!(
                "`{}` sets option_file_name to `{named}`, which has no effect: \
                 the options file is chosen before it is read. Pass \
                 `option_file_name={named}` on the command line to read that file.",
                path.display()
            ));
        }
        load.path = Some(path);
        Ok(load)
    }

    /// Read an `ipopt.opt`-format options file. Equivalent to
    /// `IpoptApplication::Initialize(const std::string& options_file)`.
    pub fn initialize_with_options_file(&mut self, path: &Path) -> Result<(), SolverException> {
        let txt = std::fs::read_to_string(path).map_err(|e| {
            SolverException::new(
                ExceptionKind::IPOPT_APPLICATION_ERROR,
                format!("could not read options file {}: {}", path.display(), e),
                file!(),
                line!() as Index,
            )
        })?;
        self.options.read_from_str(&txt, true)?;
        self.open_output_file_journal();
        Ok(())
    }

    /// Read options from a string in `ipopt.opt` format. Useful for
    /// tests and embedded callers.
    pub fn initialize_with_options_str(&mut self, s: &str) -> Result<(), SolverException> {
        self.options.read_from_str(s, true)?;
        self.open_output_file_journal();
        Ok(())
    }

    /// Honor `output_file` / `file_print_level` / `file_append`: when
    /// `output_file` is non-empty, attach a `FileJournal` named
    /// `"OutputFile:<fname>"` at the requested level. Mirrors
    /// `IpoptApplication::OpenOutputFile` (called from `Initialize`).
    /// No-op if `output_file` is unset, empty, or could not be opened.
    ///
    /// NOTE: pounce's iteration output currently bypasses the
    /// journalist and writes directly to stdout. The file journal is
    /// attached and the timing report (gated by `print_timing_statistics`)
    /// is mirrored to it; per-iter rows will start landing in the file
    /// once the iter-output path is routed through the journalist.
    fn open_output_file_journal(&self) {
        let fname = match self.options.get_string_value("output_file", "") {
            Ok((v, true)) if !v.is_empty() => v,
            _ => return,
        };
        let level_int = self
            .options
            .get_integer_value("file_print_level", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(5);
        let level = journal_level_from_int(level_int);
        let append = self
            .options
            .get_bool_value("file_append", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(false);
        let jname = format!("OutputFile:{}", fname);
        let _ = self
            .journalist
            .add_file_journal(&jname, &fname, level, append);
    }

    /// No-op initialize (just succeeds). Mirrors
    /// `IpoptApplication::Initialize(bool allow_clobber)` with no
    /// options file.
    pub fn initialize(&mut self) -> Result<(), SolverException> {
        Ok(())
    }

    /// Mirror `IpoptApplication::OpenOutputFile`. Sets the `output_file`
    /// / `file_print_level` options and attaches a matching
    /// `FileJournal` named `OutputFile:<fname>` to the journalist.
    /// Returns `false` if the file could not be opened or the option
    /// store rejected the request (e.g. clamped print level).
    pub fn open_output_file(&mut self, fname: &str, print_level: i32) -> bool {
        if self
            .options
            .set_string_value("output_file", fname, true, false)
            .is_err()
        {
            return false;
        }
        if self
            .options
            .set_integer_value("file_print_level", print_level as Index, true, false)
            .is_err()
        {
            return false;
        }
        let level = journal_level_from_int(print_level);
        let jname = format!("OutputFile:{}", fname);
        // Drop any previous file journal so a second call switches files
        // cleanly. `add_file_journal` would otherwise refuse to attach
        // a duplicate by name; remove-by-name isn't in the journalist
        // API, so we settle for the name-collision case here.
        self.journalist
            .add_file_journal(&jname, fname, level, false)
            .is_some()
    }

    /// Wrap a TNLP and report problem dimensions. Used in tests until
    /// the full IPM path covers every entry shape.
    pub fn problem_dimensions(&self, tnlp: &mut dyn TNLP) -> Option<NlpInfo> {
        tnlp.get_nlp_info()
    }

    /// Diagnostics from the safeguarded `least_square_init_primal`
    /// initializer step of the last solve (gh#605): the nonlinear
    /// violation before and after, the accepted step norm, how many
    /// backtracking trials were rejected, and why it stopped. `None`
    /// when `least_square_init_primal` was off or the model had no
    /// constraints.
    pub fn least_square_init_report(&self) -> Option<crate::init::default::LeastSquareInitReport> {
        self.least_square_init_report.clone()
    }

    pub fn statistics(&self) -> SolveStatistics {
        self.statistics.borrow().clone()
    }

    /// What the warm-start initializer made of the iterate the caller
    /// supplied to the most recent solve (gh#606): the residuals it
    /// measured, whether each multiplier block was accepted,
    /// reconstructed or discarded, and the barrier parameter it
    /// settled on.
    ///
    /// `None` when the last solve was a cold start
    /// (`warm_start_init_point=no`), or when no solve has run. Reset at
    /// the top of every solve, like [`Self::timing_stats`].
    pub fn warm_start_diagnostics(&self) -> Option<crate::init::warm_start::WarmStartDiagnostics> {
        self.warm_start_diag.borrow().clone()
    }

    /// Shared timing accumulator from the most recent `optimize_tnlp`
    /// call. Each subsystem (algorithm, NLP, KKT solver) bumped its own
    /// fields during the solve; consumers read totals out of the
    /// returned `Rc`. The instance is replaced at the top of every
    /// subsequent solve, so cloning the `Rc` and holding it past a
    /// re-solve will give you the previous solve's timings — by design.
    pub fn timing_stats(&self) -> Rc<TimingStatistics> {
        Rc::clone(&self.timing.borrow())
    }

    /// Aggregate linear-solver post-mortem from the most recent
    /// `optimize_tnlp` call. `Some` when the workspace-default FERAL
    /// backend ran at least one factor; `None` when no factors were
    /// recorded (custom factory plugged via
    /// [`Self::set_linear_backend_factory`], or solve aborted before
    /// the first KKT factor). Reset at the top of every solve.
    pub fn linear_solver_summary(&self) -> Option<LinearSolverSummary> {
        let guard = self.linsol_summary_sink.lock().ok()?;
        if guard.is_empty() {
            None
        } else {
            Some(guard.clone())
        }
    }

    /// Drive a solve.
    ///
    /// * Constrained problems (`m > 0`) take the primal-dual IPM path:
    ///   build a `TNLPAdapter` → `OrigIpoptNlp`, run the
    ///   [`AlgorithmBuilder`] with the workspace MA57 backend, and
    ///   call [`IpoptAlgorithm::optimize`]. The `SolverReturn` →
    ///   `ApplicationReturnStatus` mapping mirrors the table in
    ///   `ref/Ipopt/AGENT_REFERENCE/MAIN_LOOP.md` ("exception →
    ///   SolverReturn map").
    /// * Unconstrained problems (`m == 0`) keep going through the
    ///   in-`pounce-nlp` Newton driver so the trivial path is
    ///   independent of the linear-solver backend.
    /// Wrap `tnlp` so per-variable scaling factors are applied as a
    /// change of variables, when `nlp_scaling_method=user-scaling` is
    /// in effect and the problem supplies non-unit factors (gh#486).
    ///
    /// Returns the TNLP unchanged under any other scaling method, or
    /// when the problem asks for no variable scaling, so an unscaled
    /// solve pays nothing. The `Err` carries a message ready to print.
    fn install_variable_scaling(
        &self,
        tnlp: Rc<RefCell<dyn TNLP>>,
    ) -> Result<Rc<RefCell<dyn TNLP>>, String> {
        // Cleared first so the accessor describes *this* solve. An
        // application is reusable across solves (`pounce-cinterface`
        // holds one across `IpoptSolve` calls), and a stale vector
        // would have a later unscaled solve reporting the previous
        // solve's factors.
        *self.variable_scaling.borrow_mut() = None;
        let method = self
            .options
            .get_string_value("nlp_scaling_method", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or_else(|| "gradient-based".to_string());
        // `curvature-based` (gh #703) delivers its factors through the
        // same `get_scaling_parameters` callback, so the per-variable half
        // needs the same substitution wrapper user factors get.
        if method != "user-scaling" && method != "curvature-based" {
            return Ok(tnlp);
        }
        match pounce_nlp::scaling_tnlp::wrap_with_scaling(
            Rc::clone(&tnlp),
            self.nlp_lower_bound_inf(),
            self.nlp_upper_bound_inf(),
        ) {
            Ok(Some(wrapped)) => {
                *self.variable_scaling.borrow_mut() =
                    pounce_nlp::scaling_tnlp::factors_of(&wrapped);
                Ok(wrapped)
            }
            Ok(None) => Ok(tnlp),
            Err(why) => Err(format!(
                // The trailing newline belongs to the message: the
                // caller emits it with `eprint!` and hands the same
                // string to the journalist, as the refusals below do.
                "pounce: nlp_scaling_method={method} supplied per-variable \
                 scaling factors that cannot be applied. {why}. Correct the \
                 factors, or drop nlp_scaling_method={method}.\n"
            )),
        }
    }

    /// The per-variable scaling factors applied to the last solve, if
    /// any (gh#486). A consumer reading the algorithm's iterate rather
    /// than the `finalize_solution` payload sees scaled coordinates and
    /// must divide `x` by these, and multiply bound multipliers.
    pub fn variable_scaling(&self) -> Option<Vec<Number>> {
        self.variable_scaling.borrow().clone()
    }

    /// Install a starting-point conditioner, if one is asked for.
    ///
    /// Returns the TNLP unchanged when neither `start_point_perturbation` nor
    /// `start_point_conditioner` is set, which is the default — so an ordinary
    /// solve pays nothing and its trajectory is untouched.
    ///
    /// Sits *above* the presolve wrapper and *below* the variable-scaling one,
    /// so the point it conditions is the point the algorithm will actually
    /// start from, in the coordinates the algorithm will see. Both
    /// conditioners override only `get_starting_point`; every other callback
    /// forwards, so the solve that follows is the solve pounce would have run
    /// had the conditioned point been submitted directly.
    ///
    /// The two are mutually exclusive by construction rather than by refusal:
    /// the displacement is the failure-recovery rung and the Adam warm-up is a
    /// user opt-in, and stacking them would put random noise on top of a point
    /// the warm-up just spent 200 evaluations choosing. The displacement wins
    /// when both are set, because the only thing that sets it is a solve that
    /// has already failed.
    fn install_start_conditioner(&self, tnlp: Rc<RefCell<dyn TNLP>>) -> Rc<RefCell<dyn TNLP>> {
        use pounce_nlp::start_conditioner::{AdamConfig, ConditionedStartTnlp, StartConditioner};
        // Each option is read with its literal tag rather than through a
        // `|name, fallback|` helper. A helper reads better and costs the
        // wiring guard in `tests/init_options_wiring.rs` its evidence: that
        // test scans the source for `get_*_value("<tag>"` to prove every
        // registered Initialization option is actually consumed, and a tag
        // passed as a variable is invisible to it. A registered knob nothing
        // reads validates, accepts a value, and lies.
        let perturbation = self
            .options
            .get_numeric_value("start_point_perturbation", "")
            .map(|(v, _found)| v)
            .unwrap_or(0.0);
        let conditioner = if perturbation > 0.0 {
            let seed = self
                .options
                .get_integer_value("start_point_perturbation_seed", "")
                .map(|(v, _found)| v)
                .unwrap_or(0);
            StartConditioner::Jitter {
                // `Index` is signed and the option is lower-bounded at 0, so
                // this cast cannot lose a set bit for any accepted value.
                seed: seed.max(0) as u64,
                scale: perturbation,
            }
        } else {
            let which = self
                .options
                .get_string_value("start_point_conditioner", "")
                .map(|(v, _found)| v)
                .unwrap_or_else(|_| "none".to_string());
            if which != "adam" {
                return tnlp;
            }
            let d = AdamConfig::default();
            StartConditioner::Adam(AdamConfig {
                iters: self
                    .options
                    .get_integer_value("adam_warmup_iters", "")
                    .map(|(v, _found)| v.max(0) as usize)
                    .unwrap_or(d.iters),
                lr: self
                    .options
                    .get_numeric_value("adam_warmup_learning_rate", "")
                    .map(|(v, _found)| v)
                    .unwrap_or(d.lr),
                rho: self
                    .options
                    .get_numeric_value("adam_warmup_penalty", "")
                    .map(|(v, _found)| v)
                    .unwrap_or(d.rho),
                ..d
            })
        };
        // The sentinels have to come from the options, not from the
        // conditioner's own default: a caller who moved `nlp_lower_bound_inf`
        // would otherwise have a bound the algorithm treats as absent clipped
        // against as if it were real.
        //
        // Passed through unclamped. These were once `lower.min(-DEFAULT)` /
        // `upper.max(DEFAULT)`, which honours only a *loosened* sentinel and
        // leaves the failure above intact for a tightened one: at
        // `nlp_lower_bound_inf=-1e10` the conditioner still used `-1e19`, so a
        // `-1e15` bound was absent to the algorithm and present to the
        // clipper — the exact case the comment exists to rule out. The
        // sentinel means "absent"; there is only one right answer for what it
        // is, and it is the caller's.
        let lower = self.nlp_lower_bound_inf();
        let upper = self.nlp_upper_bound_inf();
        let wrapped = ConditionedStartTnlp::new(tnlp, conditioner).with_bound_inf(lower, upper);
        Rc::new(RefCell::new(wrapped))
    }

    pub fn optimize_tnlp(&mut self, tnlp: Rc<RefCell<dyn TNLP>>) -> ApplicationReturnStatus {
        self.optimize_tnlp_with_derivative_test_tnlp(tnlp, None)
    }

    /// Solve through `tnlp`, optionally overriding the derivative-test target.
    /// `None` tests the scaled and conditioned TNLP.
    pub fn optimize_tnlp_with_derivative_test_tnlp(
        &mut self,
        tnlp: Rc<RefCell<dyn TNLP>>,
        derivative_test_tnlp: Option<Rc<RefCell<dyn TNLP>>>,
    ) -> ApplicationReturnStatus {
        // gh#884. Both belong to this solve, not to whatever ran before
        // it. Reset *here* rather than in `optimize_constrained`, which
        // runs once per attempt: the signature accumulates across the
        // attempts a lower wrapper may spend, so
        // `run_with_dual_divergence_retry` reads "some attempt of the base
        // solve saw it" rather than "the last one did".
        self.dual_divergence_signature.set(false);
        self.dual_divergence_retry_promoted.set(false);
        self.answer_restored_from_floor.set(false);
        // gh#486 stage 2: per-variable `scaling_factor` is applied by
        // substituting variables one level below the algorithm, since
        // the core's scaling models the objective and the constraint
        // rows only. The wrapper consumes the variable factors and
        // forwards the rest, so `OrigIpoptNlp` sees exactly what it
        // has always handled. Installed here because every entry point
        // funnels through this method, and only under `user-scaling`,
        // the one method that consults the TNLP for factors at all.
        let tnlp = match self.install_variable_scaling(tnlp) {
            Ok(t) => t,
            Err(msg) => {
                use pounce_common::journalist::JournalCategory;
                eprint!("{msg}");
                self.journalist
                    .print(JournalLevel::J_ERROR, JournalCategory::J_MAIN, &msg);
                return ApplicationReturnStatus::InvalidOption;
            }
        };

        // Starting-point conditioning (`start_point_perturbation`,
        // `start_point_conditioner`). A no-op unless one is set, and set by
        // nothing automatic except the local-infeasibility ladder's third
        // rung, which only runs after a solve has already failed.
        let tnlp = self.install_start_conditioner(tnlp);
        let derivative_test_tnlp = derivative_test_tnlp.as_ref().unwrap_or(&tnlp);

        if let Some(value) = self.unsupported_library_solver_selection() {
            use pounce_common::journalist::JournalCategory;
            self.journalist.print(
                JournalLevel::J_ERROR,
                JournalCategory::J_MAIN,
                &format!(
                    "pounce: solver_selection={value} routing is only available \
                     through the pounce CLI (.nl input); library consumers can use \
                     qp-active-set, nlp, or auto.\n"
                ),
            );
            return ApplicationReturnStatus::InvalidOption;
        }

        // A `linear_solver` pounce does not implement is refused rather
        // than quietly served by FERAL (gh#483 follow-up). Checked here,
        // before any work, so a library consumer gets the same verdict the
        // CLI gives before its banner.
        if let Some(value) = self.unimplemented_linear_solver() {
            use pounce_common::journalist::JournalCategory;
            let msg = format!("{}\n", Self::unimplemented_linear_solver_message(&value));
            eprint!("{msg}");
            self.journalist
                .print(JournalLevel::J_ERROR, JournalCategory::J_MAIN, &msg);
            return ApplicationReturnStatus::InvalidOption;
        }

        // gh#483 follow-up: an option naming a feature pounce does not
        // implement is refused, not shrugged off. See
        // `unimplemented_options` for how membership was established and
        // why an explicitly-set *default* is deliberately still allowed.
        // gh#518: same treatment for `option_file_name` on an entry point
        // that cannot resolve it. Separate from the table above because
        // the *feature* now exists — just not here.
        // gh#604: same treatment one level down, for a registered *value*
        // of an option pounce otherwise reads (`bound_mult_init_method=
        // mu-based`).
        // gh#604: and for a convex-engine knob on an entry point that
        // cannot route to that engine — `option_file_name`'s case, one
        // feature over.
        if let Some(msg) = self
            .unimplemented_option_refusal()
            .or_else(|| self.unimplemented_option_value_refusal())
            .or_else(|| self.unhonored_option_file_name())
            .or_else(|| self.unhonored_convex_option())
        {
            use pounce_common::journalist::JournalCategory;
            eprintln!("{msg}");
            self.journalist.print(
                JournalLevel::J_ERROR,
                JournalCategory::J_MAIN,
                &format!("{msg}\n"),
            );
            return ApplicationReturnStatus::InvalidOption;
        }
        // A `ma57_pivtolmax` the user set *below* `ma57_pivtol` is a
        // contradiction — the escalation ceiling under its floor — and
        // upstream refuses it outright
        // (`IpMa57TSolverInterface.cpp:313`, `OPTION_INVALID`). pounce
        // used to silently rewrite it to `ma57_pivtol`. That was
        // unreachable while gh#825 was live, since no `ma57_*` value
        // reached the backend at all, and became reachable the moment
        // that was fixed — so it is refused here rather than shipped as
        // a new way to be quietly ignored.
        if let Some(msg) = self.ma57_pivtol_bracket_refusal() {
            use pounce_common::journalist::JournalCategory;
            eprintln!("{msg}");
            self.journalist.print(
                JournalLevel::J_ERROR,
                JournalCategory::J_MAIN,
                &format!("{msg}\n"),
            );
            return ApplicationReturnStatus::InvalidOption;
        }

        let backend_warnings = self.take_unimplemented_backend_warnings();
        for warning in self
            .unexploited_hint_warnings()
            .into_iter()
            .chain(backend_warnings)
        {
            eprintln!("{warning}");
        }

        // Test before presolve, using the requested coordinate space.
        self.run_derivative_test(derivative_test_tnlp);

        // Top-level algorithm dispatch (Phase 5b §7.1). When the
        // `algorithm` option resolves to "active-set-sqp", route
        // to the Phase 5b SQP path; otherwise fall through to the
        // existing IPM flow unchanged.
        // Materialize generic TNLP presolve once at the public entry point.
        // The wrapper owns the submitted callback TNLP, so every algorithm
        // path below (including retry paths) continues to postsolve into
        // the original user-facing space. With `presolve=no`, this returns
        // the exact same Rc unchanged.
        let tnlp = if self.presolve_already_applied {
            tnlp
        } else {
            match pounce_presolve::wrap_from_options(tnlp, &self.options) {
                Ok(tnlp) => tnlp,
                Err(err) => {
                    use pounce_common::journalist::JournalCategory;
                    self.journalist.print(
                        JournalLevel::J_ERROR,
                        JournalCategory::J_MAIN,
                        &format!("pounce: could not materialize presolve options: {err}\n"),
                    );
                    return ApplicationReturnStatus::InvalidOption;
                }
            }
        };

        if self.is_sqp_algorithm_selected() {
            return self.optimize_sqp_tnlp(tnlp);
        }
        let info = match tnlp.borrow_mut().get_nlp_info() {
            Some(info) => info,
            None => return ApplicationReturnStatus::InvalidProblemDefinition,
        };

        // Presolve-certified infeasibility. `get_nlp_info` above is what forces
        // the (lazy) presolve init, so this is the first point at which the
        // proof exists. If bound propagation or FBBT established that the
        // feasible region is empty, there is nothing left to compute: return
        // the verdict directly.
        //
        // Short-circuiting *here*, before dispatch, is deliberate. Running the
        // solve anyway would only re-derive a strictly weaker result — a
        // stationary point of the constraint violation, which for a nonconvex
        // problem proves nothing globally — and would also hand an
        // `InfeasibleProblemDetected` to the ℓ₁ auto-fallback below
        // (`is_l1_fallback_trigger`), which would then burn a whole second
        // solve retrying a problem already proved to have no solution.
        //
        // Soundness rests on `presolve_infeasibility_proof` returning `Some`
        // only for a contradiction derived on an *un-clamped* box — a
        // detection made while a Phase-0 auxiliary elimination is in force can
        // be an artifact of that elimination and is re-checked after rollback
        // before it is certified. See `PresolveState::certified_infeasible`.
        if let Some(proof) = tnlp.borrow().presolve_infeasibility_proof() {
            use pounce_common::journalist::JournalCategory;
            let detail = match proof {
                pounce_nlp::tnlp::InfeasibilityProof::BoundPropagation => {
                    "bound propagation crossed a variable's bounds".to_string()
                }
                pounce_nlp::tnlp::InfeasibilityProof::IntervalArithmetic { witness } => {
                    format!("interval arithmetic emptied constraint {witness}'s range")
                }
            };
            self.journalist.print(
                JournalLevel::J_SUMMARY,
                JournalCategory::J_MAIN,
                &format!(
                    "\nEXIT: Presolve detected the feasible region is empty ({detail}).\n\
                     No feasible point exists; the solve was not run.\n"
                ),
            );
            return ApplicationReturnStatus::InfeasibleProblemDetected;
        }
        // ℓ₁-exact penalty-barrier opt-in (pounce#10).
        // Phase 3 wraps the user TNLP and runs an outer Byrd-Nocedal-
        // Waltz ρ-escalation loop around the constrained IPM, with a
        // honest-infeasibility status upgrade when the slacks fail to
        // collapse at saturated ρ. Phase-1/2 one-shot use is preserved
        // when `l1_penalty_max_outer_iter == 1`. The wrapper is a
        // no-op for problems with no equality rows, so the
        // unconstrained dispatch below is unaffected when there is
        // nothing to wrap.
        if info.m > 0 && self.is_l1_penalty_enabled() {
            if let Some(status) = self.run_l1_penalty_outer_loop(Rc::clone(&tnlp)) {
                return status;
            }
            // Falls through: wrapper construction failed (inner refused
            // get_nlp_info / get_bounds_info) or no equality rows to
            // slack. Standard dispatch runs unmodified.
        }
        // Phase 3.5 auto-fallback (pounce#10): if the standard solve
        // ends in a trigger-class status, retry transparently with
        // the wrapper. Promote the retry's status only if it returns
        // SolveSucceeded — otherwise return the original. Skipped if
        // the user already opted into the wrapper above (this avoids
        // a double pass and keeps semantics predictable).
        if info.m > 0 && self.is_l1_fallback_enabled() && !self.is_l1_penalty_enabled() {
            return self.run_with_l1_fallback(tnlp);
        }
        // Biactive dual-divergence retry (gh#884): if the solve settles
        // its primal while its multipliers run away, throw the iterate
        // away and solve again from scratch with the constraint-Jacobian
        // perturbation on. Outermost of the two retry wrappers, so its
        // "base solve" is the whole standard dispatch below including the
        // μ flip — the signature is about a *solve*, and spending the μ
        // flip first is strictly cheaper than spending this one first.
        if self.is_dual_divergence_retry_enabled() {
            return self.run_with_dual_divergence_retry(tnlp);
        }
        self.dispatch_standard_solve(tnlp)
    }

    /// The standard solve dispatch: the μ-strategy fallback if it is
    /// enabled, otherwise one `optimize_constrained` call.
    ///
    /// Factored out of `optimize_tnlp_with_derivative_test_tnlp` so
    /// [`Self::run_with_dual_divergence_retry`] can wrap the whole of it
    /// rather than only the bare IPM call (gh#884).
    fn dispatch_standard_solve(&mut self, tnlp: Rc<RefCell<dyn TNLP>>) -> ApplicationReturnStatus {
        // μ-strategy auto-fallback (pounce#138): if the standard solve
        // stalls, retry once with the opposite mu_strategy and promote
        // only on Solve_Succeeded. Which stalls qualify depends on
        // whether the caller asked for the retry — see
        // `run_with_mu_strategy_fallback` (pounce#748).
        // Applies to constrained and unconstrained alike (both run the
        // same IPM). Independent of, and lower priority than, the ℓ₁
        // fallback above.
        if self.is_mu_strategy_fallback_enabled() {
            return self.run_with_mu_strategy_fallback(tnlp);
        }
        // Every problem — constrained or not — goes through the same
        // primal-dual IPM, exactly as upstream Ipopt does. There is no
        // separate "unconstrained Newton" path: the linear-solver
        // backend (FERAL/MA57) handles the augmented system, so the
        // sparse IPM covers `m == 0` at any `n` without a dense-Hessian
        // blowup.
        self.optimize_constrained(tnlp)
    }

    /// Read the ℓ₁ wrapper master switch from the OptionsList.
    /// Default `false` when the option is not set.
    fn is_l1_penalty_enabled(&self) -> bool {
        self.options
            .get_bool_value("l1_exact_penalty_barrier", "")
            .ok()
            .and_then(|(v, found)| found.then_some(v))
            .unwrap_or(false)
    }

    fn l1_penalty_init(&self) -> Number {
        self.options
            .get_numeric_value("l1_penalty_init", "")
            .ok()
            .and_then(|(v, found)| found.then_some(v))
            .unwrap_or(1.0)
    }
    fn l1_penalty_max(&self) -> Number {
        self.options
            .get_numeric_value("l1_penalty_max", "")
            .ok()
            .and_then(|(v, found)| found.then_some(v))
            .unwrap_or(1.0e6)
    }
    fn l1_penalty_increase_factor(&self) -> Number {
        self.options
            .get_numeric_value("l1_penalty_increase_factor", "")
            .ok()
            .and_then(|(v, found)| found.then_some(v))
            .unwrap_or(8.0)
    }
    fn l1_penalty_max_outer_iter(&self) -> usize {
        self.options
            .get_integer_value("l1_penalty_max_outer_iter", "")
            .ok()
            .and_then(|(v, found)| found.then_some(v))
            .unwrap_or(8) as usize
    }
    fn l1_slack_tol(&self) -> Number {
        self.options
            .get_numeric_value("l1_slack_tol", "")
            .ok()
            .and_then(|(v, found)| found.then_some(v))
            .unwrap_or(1.0e-6)
    }
    fn l1_steering_factor(&self) -> Number {
        self.options
            .get_numeric_value("l1_steering_factor", "")
            .ok()
            .and_then(|(v, found)| found.then_some(v))
            .unwrap_or(10.0)
    }
    fn is_l1_fallback_enabled(&self) -> bool {
        self.options
            .get_bool_value("l1_fallback_on_restoration_failure", "")
            .ok()
            .and_then(|(v, found)| found.then_some(v))
            .unwrap_or(false)
    }

    /// Did the caller set `mu_strategy` explicitly?
    ///
    /// The answer decides whether the limited-memory default applies
    /// (see `algorithm_builder_from_options`): upstream only substitutes
    /// `adaptive` for the registered `monotone` when the option is
    /// absent from the list.
    fn mu_strategy_was_set(&self) -> bool {
        matches!(
            self.options.get_string_value("mu_strategy", ""),
            Ok((_, true))
        )
    }

    /// Did the caller set `mu_strategy_fallback` themselves? Separates
    /// the opted-in retry from the default-on one, which triggers on a
    /// narrower set of statuses (pounce#748).
    ///
    /// True for an explicit `no` as well as an explicit `yes`, which is
    /// harmless: this is only ever read downstream of
    /// [`Self::is_mu_strategy_fallback_enabled`], and an explicit `no`
    /// stops there.
    fn mu_strategy_fallback_was_set(&self) -> bool {
        matches!(
            self.options.get_bool_value("mu_strategy_fallback", ""),
            Ok((_, true))
        )
    }

    /// Options whose presence means a `Solved_To_Acceptable_Level`
    /// exit may be something the *caller* asked for rather than a stall
    /// POUNCE fell into (gh #757).
    ///
    /// Every one of them either moves the bar a certificate has to clear
    /// (`tol` and the component tolerances, the `acceptable_*` family),
    /// arms a guard that refuses a certificate the iterate would
    /// otherwise have earned (`kkt_fidelity_tol`, the certificate-mask
    /// and noise-floor kappas, the divergence / infeasibility streaks,
    /// the restoration-decline pair).
    ///
    /// Options that only move the *starting point* are deliberately not
    /// here. `least_square_init_primal` was tried and removed: a caller
    /// who picks an initialization heuristic has said nothing about what
    /// convergence means, and listing it made an explicit `=no` behave
    /// differently from omitting the option, which is a distinction the
    /// rest of the solver does not draw.
    const TERMINATION_POLICY_OPTIONS: &'static [&'static str] = &[
        "tol",
        "dual_inf_tol",
        "constr_viol_tol",
        "compl_inf_tol",
        "acceptable_tol",
        "acceptable_iter",
        "acceptable_dual_inf_tol",
        "acceptable_constr_viol_tol",
        "acceptable_compl_inf_tol",
        "acceptable_obj_change_tol",
        "kkt_fidelity_tol",
        "obj_scale_certificate_threshold",
        "dual_inf_scale_kappa",
        "primal_noise_floor_kappa",
        "dual_diverging_streak",
        "infeas_max_streak",
        "resto_decline_deferrals",
        "resto_decline_progress_ratio",
        "neg_curv_escapes",
        "limited_memory_ls_failure_restarts",
    ];

    /// Did the caller set any option from
    /// [`Self::TERMINATION_POLICY_OPTIONS`]?
    ///
    /// This is what separates the two readings of a
    /// `Solved_To_Acceptable_Level` exit. Under stock convergence
    /// settings it means POUNCE's own schedule parked the dual term
    /// above `tol` and a flipped schedule is worth one try. Under a
    /// caller-modified one it may be the signal the caller armed the
    /// option to receive, and erasing it with a retry is exactly the
    /// laundering pounce#748 refused to do by default.
    fn caller_set_termination_policy(&self) -> bool {
        Self::TERMINATION_POLICY_OPTIONS.iter().any(|name| {
            matches!(self.options.get_numeric_value(name, ""), Ok((_, true)))
                || matches!(self.options.get_integer_value(name, ""), Ok((_, true)))
                || matches!(self.options.get_bool_value(name, ""), Ok((_, true)))
                || matches!(self.options.get_string_value(name, ""), Ok((_, true)))
        })
    }

    /// The μ strategy this option table actually resolves to, as
    /// `algorithm_builder_from_options` will build it: the explicit
    /// value when there is one, otherwise `adaptive` for a
    /// limited-memory Hessian and `monotone` for anything else.
    ///
    /// The fallback below flips *this*, not the registered default —
    /// flipping the registered default under limited-memory would
    /// "retry" with the strategy that just failed.
    fn effective_mu_strategy_is_adaptive(&self) -> bool {
        if let Ok((v, true)) = self.options.get_string_value("mu_strategy", "") {
            return v == "adaptive";
        }
        matches!(
            self.options.get_string_value("hessian_approximation", ""),
            Ok((ref v, true)) if v == "limited-memory"
        )
    }

    /// Read the μ-strategy auto-fallback switch (pounce#138).
    ///
    /// An explicit setting always wins. Absent one the default is **on**
    /// (pounce#748) — but only while the user has not chosen a
    /// `mu_strategy` themselves. Retrying under the other schedule is a
    /// recovery for a solve that stalled on a strategy POUNCE picked; it
    /// is not licence to override a strategy the caller named. Without
    /// that condition, flipping the default would silently contaminate
    /// every controlled comparison that pins `mu_strategy` on purpose,
    /// this repository's own benchmark arms included. The motivating
    /// case is unaffected: `dirichlet120` stalls under the
    /// limited-memory substitution (pounce#746), which by definition
    /// only happens when `mu_strategy` is unset.
    fn is_mu_strategy_fallback_enabled(&self) -> bool {
        match self.options.get_bool_value("mu_strategy_fallback", "") {
            Ok((v, true)) => v,
            _ => !self.mu_strategy_was_set(),
        }
    }

    /// Has the user set `algorithm = active-set-sqp`? Reads the
    /// string option and matches case-insensitively against the
    /// design-note §7.1 spelling. Any value other than
    /// "active-set-sqp" (including absence) routes to the
    /// default IPM path.
    /// Stash a warm-start iterate for the SQP path. Consumed by
    /// the next `optimize_tnlp` call when the `algorithm` option
    /// resolves to `active-set-sqp`; the IPM path ignores it.
    /// Phase 5c (§6) — the parametric / MPC warm-start hand-off.
    ///
    /// The iterate is auto-cleared after use, so a follow-up
    /// solve without an intervening `set_sqp_warm_start` call
    /// cold-starts.
    pub fn set_sqp_warm_start(&mut self, warm: crate::sqp::SqpIterates) {
        self.sqp_warm_start = Some(warm);
    }

    /// Drop any pending warm-start iterate without solving.
    pub fn clear_sqp_warm_start(&mut self) {
        self.sqp_warm_start = None;
    }

    /// What the crossover phase did on the most recent solve (gh#612).
    ///
    /// `None` means crossover never ran — either `crossover=no` (the
    /// default) or the solve did not converge. A `Some` whose
    /// [`CrossoverReport::accepted`] is false means it ran and *declined*;
    /// the two are different facts about a solve and consumers that reason
    /// about active-set certainty (sensitivity's AMBIGUOUS class, a
    /// downstream `var_status`) need to tell them apart.
    ///
    /// [`CrossoverReport::accepted`]: crate::crossover::CrossoverReport::accepted
    pub fn crossover_report(&self) -> Option<&crate::crossover::CrossoverReport> {
        self.crossover_report.as_ref()
    }

    /// Install a full primal-dual warm-start iterate for the next IPM
    /// `optimize_tnlp`. Captured by the debugger's `resolve` so the
    /// re-solve continues from the paused interior point. The caller is
    /// responsible for also enabling `warm_start_init_point=yes` (and
    /// usually `warm_start_target_mu=<μ>`) so the re-optimize branch of
    /// `WarmStartIterateInitializer` preserves the installed iterate.
    /// Consumed once per solve, then auto-cleared.
    pub fn set_warm_start_iterate(&mut self, snap: crate::debug::IterateSnapshot) {
        self.warm_start_iterate = Some(snap);
    }

    /// Install a caller-supplied fill-reducing permutation for the KKT
    /// linear solver (pounce#180 item 1). The next `optimize_*` builds
    /// the FERAL backend with [`pounce_feral::OrderingMethod::External`],
    /// overriding the `feral_ordering` string option / env var. Use this
    /// to inject a block-triangular / Schur ordering a generic algorithm
    /// cannot see (Parker, Garcia & Bent, arXiv:2602.17968) or a tearing
    /// ordering from equation-oriented decomposition.
    ///
    /// `perm` is a **0-based, new-to-old permutation** (`perm[k]` is the
    /// original index that becomes index `k`), and its length must equal
    /// the augmented KKT system dimension (variables + slacks +
    /// constraint duals), *not* the problem's `n`. A wrong length or a
    /// non-bijection is rejected by FERAL at the first factorization with
    /// an `InvalidInput` error (never a panic), surfacing as a solver
    /// failure rather than a silently-wrong solve — the ordering only
    /// affects fill/time, never the computed solution.
    ///
    /// Persistent config: unlike the warm-start hooks it is *not*
    /// auto-cleared after a solve. Call [`Self::clear_external_ordering`]
    /// to drop it. Ignored by non-FERAL backends and by any custom
    /// factory plugged via [`Self::set_linear_backend_factory`].
    pub fn set_external_ordering(&mut self, perm: Vec<usize>) {
        self.external_ordering = Some(perm);
    }

    /// Drop any installed external KKT ordering, restoring the
    /// `feral_ordering`-driven default for subsequent solves.
    pub fn clear_external_ordering(&mut self) {
        self.external_ordering = None;
    }

    /// The currently-installed external KKT ordering, if any.
    pub fn external_ordering(&self) -> Option<&[usize]> {
        self.external_ordering.as_deref()
    }

    /// Install a block-triangular / Schur KKT partition (pounce#180 item 2).
    /// `indices` are KKT-space indices (`0..dim` in the `x, s, c, d` block
    /// order the aug-system solver assembles) naming the Schur block `S`; that
    /// block is Schur-complemented out and only the two diagonal blocks are
    /// factorized (inertia via Sylvester's law). Honored on the IPM + feral +
    /// exact-Hessian path; the Schur solver falls back to the standard
    /// full-space solver transparently when the partition is unsuitable (too
    /// large a fraction of the system, malformed, or a backend error), so a
    /// stray hook never breaks a solve. Persistent config (not auto-cleared);
    /// drop it via [`Self::clear_kkt_schur_block`].
    pub fn set_kkt_schur_block(&mut self, indices: Vec<usize>) {
        self.kkt_schur_block = Some(indices);
    }

    /// Drop any installed Schur KKT partition, restoring the standard
    /// full-space solver for subsequent solves.
    pub fn clear_kkt_schur_block(&mut self) {
        self.kkt_schur_block = None;
    }

    /// The currently-installed Schur KKT partition, if any.
    pub fn kkt_schur_block(&self) -> Option<&[usize]> {
        self.kkt_schur_block.as_deref()
    }

    /// Return the final QP working set from the most recent SQP
    /// solve, or `None` if the last solve wasn't SQP, didn't
    /// produce a working set (cold-start declared the iterate
    /// optimal before solving any QP), or no SQP solve has run.
    pub fn last_sqp_working_set(&self) -> Option<&pounce_qp::WorkingSet> {
        self.sqp_last_working_set.as_ref()
    }

    /// If `solver_selection` is explicitly set to a value whose routing lives
    /// only in the CLI's `.nl` dispatch, return  it; otherwise `None`.
    /// `optimize_tnlp` uses this to reject a forced convex selection a library
    /// consumer cannot honor.
    fn unsupported_library_solver_selection(&self) -> Option<&'static str> {
        let (v, found) = self.options.get_string_value("solver_selection", "").ok()?;
        if !found {
            return None;
        }
        ["lp-ipm", "qp-ipm", "socp"]
            .into_iter()
            .find(|c| v.eq_ignore_ascii_case(c))
    }

    /// The `linear_solver` value when the caller explicitly asked for a
    /// backend pounce does not implement; `None` when the request can be
    /// served (or was never made).
    ///
    /// pounce ships two: **FERAL** (pure Rust, the effective default) and
    /// **MA57** (HSL, behind the `ma57` feature). The option's valid-value
    /// list is a faithful port of upstream Ipopt's — `ma27`, `ma77`,
    /// `ma86`, `ma97`, `mumps`, `pardiso`, `pardisomkl`, `spral`, `wsmp`,
    /// `custom` — so an `ipopt.opt` written for Ipopt parses here, and
    /// every one of those names used to fall through a `_ =>` arm to
    /// FERAL. A run "using MUMPS" was a FERAL run; a benchmark comparing
    /// backends compared FERAL with itself (gh#483 follow-up).
    ///
    /// The registered default is `feral`, which pounce implements, so no
    /// explicit-vs-default distinction is needed: whatever the option
    /// resolves to must be a backend that exists. (It is checked
    /// unconditionally on purpose — a future default naming something
    /// unimplemented should trip this, not slip past it.)
    ///
    /// Explicit `ma57` on a build that lacks the feature is *not* refused;
    /// that fallback is reported in the banner ("ma57 requested but not
    /// compiled"), so it is visible rather than silent, and failing a
    /// portable `ipopt.opt` over a build flag would cost more than it buys.
    pub fn unimplemented_linear_solver(&self) -> Option<String> {
        let (v, _) = self.options.get_string_value("linear_solver", "").ok()?;
        ["feral", "ma57"]
            .iter()
            .all(|ok| !v.eq_ignore_ascii_case(ok))
            .then_some(v)
    }

    /// The message for the first option the caller set that names a
    /// feature pounce does not implement, or `None`. Public so the CLI
    /// can refuse before routing — the convex dispatch never reaches
    /// `optimize_tnlp`. See [`crate::unimplemented_options`].
    ///
    /// A run configuring nothing but backends pounce does not ship is
    /// refused here too, after the per-option table has had its say —
    /// see [`crate::unimplemented_options::backend_only_refusal`]. It
    /// is folded in rather than given its own accessor so that every
    /// surface already refusing on this method refuses on it as well;
    /// the CLI is not the only frontend, and a condition worth failing
    /// on is not worth failing on only from the CLI.
    pub fn unimplemented_option_refusal(&self) -> Option<String> {
        crate::unimplemented_options::refusal(&self.options, &self.reg_options).or_else(|| {
            crate::unimplemented_options::backend_only_refusal(&self.options, &self.reg_options)
        })
    }

    /// The message for the first string option the caller set to a
    /// registered *value* pounce does not implement, or `None`.
    ///
    /// Separate from [`Self::unimplemented_option_refusal`] because the
    /// option itself is read and its other values work — it is one mode
    /// that is missing, not the feature. See
    /// [`crate::unimplemented_options::UNIMPLEMENTED_VALUES`].
    pub fn unimplemented_option_value_refusal(&self) -> Option<String> {
        crate::unimplemented_options::value_refusal(&self.options)
    }

    /// `option_file_name` set on a surface that never resolves it.
    ///
    /// The option reaches a file through exactly one path —
    /// [`Self::initialize_with_option_file`], which the `pounce` CLI
    /// drives. A library caller (Python, the C interface, WASM) sets its
    /// options directly and calls no such thing, so on those surfaces the
    /// option names a whole configuration and applies none of it: gh#518's
    /// failure mode, one surface over. It used to be caught by the blanket
    /// [`crate::unimplemented_options`] refusal, which no longer covers it
    /// now that the feature exists; this keeps the guard exactly where the
    /// feature still doesn't.
    ///
    /// Deliberately *not* fixed by having the library read an options file
    /// too: an implicit `./ipopt.opt` lookup under Python or the GAMS C
    /// link would be a surprising action at a distance, and `pounce.opt`
    /// already means something else to GAMS.
    pub fn unhonored_option_file_name(&self) -> Option<String> {
        if self.option_file_resolved {
            return None;
        }
        // Same default gate as the table: an explicitly-set *default*
        // asks for nothing, so it must not fail. `option_file_name`
        // defaults to `ipopt.opt`, and a caller round-tripping a full
        // option dump — or a generated config that spells out every
        // registered name — hits that value without asking for anything.
        if !crate::unimplemented_options::set_to_a_non_default(
            &self.options,
            &self.reg_options,
            "option_file_name",
        ) {
            return None;
        }
        match self.options.get_string_value("option_file_name", "") {
            Ok((name, true)) if !name.is_empty() => Some(format!(
                "pounce: `option_file_name` was set to `{name}`, but this entry \
                 point does not read options files — it would configure nothing. \
                 The `pounce` CLI honors it (and `./pounce.opt` / `./ipopt.opt`); \
                 from a library, read the file yourself and pass its contents to \
                 `initialize_with_options_str`, or set the options directly. \
                 Tracking issue: https://github.com/jkitchin/pounce/issues/518"
            )),
            _ => None,
        }
    }

    /// Declare that this caller can route a model to the convex LP/QP /
    /// SOCP engines, so the `qp_*` knobs they read configure something.
    ///
    /// The `pounce` CLI calls this: it owns the `.nl` structure
    /// extraction that classifies a model, which is the whole of what
    /// `solver_selection`'s convex values need. No library frontend can
    /// (see [`Self::unsupported_library_solver_selection`]), so the
    /// default is `false` and [`Self::unhonored_convex_option`] refuses
    /// the knobs there.
    ///
    /// Declaring it also covers the CLI's *fallback*: a convex attempt
    /// that returns no verified point hands the model to
    /// [`Self::optimize_tnlp`], and the `qp_*` values it was given
    /// configured that attempt for real. Refusing them at the handoff
    /// would fail a run that used them.
    pub fn set_convex_routing_available(&mut self, available: bool) {
        self.convex_routing_available = available;
    }

    /// The convex LP/QP knobs are registered core-side so every frontend
    /// parses them — but only the CLI can reach the engines that read
    /// them. On any other entry point the option names a whole
    /// configuration and applies none of it; this is the message that
    /// says so, in place of the silence.
    ///
    /// gh#604. Same shape and same default gate as
    /// [`Self::unhonored_option_file_name`]: an explicitly-set *default*
    /// asks for nothing and must keep working, so only a value that
    /// differs is refused.
    pub fn unhonored_convex_option(&self) -> Option<String> {
        if self.convex_routing_available {
            return None;
        }
        const CONVEX_ONLY: &[&str] = &[
            "qp_presolve",
            "qp_tau",
            "qp_tau_max",
            "qp_reg",
            "qp_gondzio_corr",
            "qp_infeas_tol",
            "qp_hsde",
            "qp_equilibrate",
            "qp_crossover",
        ];
        let name = CONVEX_ONLY.iter().find(|name| {
            crate::unimplemented_options::set_to_a_non_default(
                &self.options,
                &self.reg_options,
                name,
            )
        })?;
        Some(format!(
            "pounce: `{name}` tunes the convex LP/QP interior-point engine, but \
             this entry point cannot route a model to it — the option would \
             configure nothing. The `pounce` CLI reaches that engine on `.nl` \
             input (`solver_selection=lp-ipm` / `qp-ipm` / `socp`, or `auto` on \
             a model that classifies as one); from Python, `pounce.solve_qp` / \
             `pounce.solve_cone` drive it directly and take the same knobs as \
             typed arguments. On this path, `solver_selection=qp-active-set` \
             (or `algorithm=active-set-sqp`) is the nearest thing, tuned by the \
             `sqp_qp_*` options. Tracking issue: \
             https://github.com/jkitchin/pounce/issues/604"
        ))
    }

    /// Warnings for caching hints pounce does not exploit. These never
    /// block a solve: the answer is identical either way, so refusing
    /// would cost the caller more than the silence did.
    pub fn unexploited_hint_warnings(&self) -> Vec<String> {
        crate::unimplemented_options::hint_warnings(&self.options, &self.reg_options)
    }

    /// Warnings for the constant-derivative hints when the solve routes to
    /// `pounce-convex` instead of here. Call site is the convex dispatch
    /// in the CLI, next to the other guards that live there for the same
    /// reason: that dispatch never reaches [`Self::optimize_tnlp`], so
    /// `install_constant_derivative_hints` never runs and the hints are
    /// unread. On the NLP route they are honoured, so this must not be
    /// called there.
    pub fn convex_unexploited_hint_warnings(&self) -> Vec<String> {
        crate::unimplemented_options::convex_hint_warnings(&self.options, &self.reg_options)
    }

    /// Which of the four constant-derivative hints the caller actually
    /// asserted, in [`pounce_nlp::constant_derivatives::HINT_OPTIONS`]
    /// order — the order [`reconcile`] pairs against the model's own
    /// proofs.
    ///
    /// Each name is read as a literal rather than through the loop
    /// variable the caller used to use. The registered-but-unread scan
    /// (`tests/no_silent_options.rs`) keys on the option name as it
    /// appears at the accessor, so `get_bool_value(name, "")` over an
    /// array read as "no key here" and left all four sitting in the
    /// silent list while they were fully wired and consumed (#551 /
    /// #677).
    ///
    /// Matching over `HINT_OPTIONS` rather than writing a bare array
    /// keeps the slots right by construction, and a fifth hint added to
    /// `HINT_OPTIONS` trips the fallback arm instead of silently reading
    /// as "not asserted" — which is the failure mode this whole line of
    /// work exists to kill.
    fn asserted_constant_derivative_hints(&self) -> [bool; 4] {
        use pounce_nlp::constant_derivatives::HINT_OPTIONS;
        let read_yes = |key: &str| matches!(self.options.get_bool_value(key, ""), Ok((true, true)));
        HINT_OPTIONS.map(|name| match name {
            "grad_f_constant" => read_yes("grad_f_constant"),
            "hessian_constant" => read_yes("hessian_constant"),
            "jac_c_constant" => read_yes("jac_c_constant"),
            "jac_d_constant" => read_yes("jac_d_constant"),
            other => unreachable!("`{other}` is in HINT_OPTIONS with no read site"),
        })
    }

    /// Resolve the four constant-derivative hints for this solve and
    /// install the result on the NLP (gh #588, phase Q6).
    ///
    /// `grad_f_constant` / `hessian_constant` / `jac_c_constant` /
    /// `jac_d_constant` are, upstream, unchecked user assertions: Ipopt
    /// reuses the derivative and returns a wrong answer if the assertion
    /// was false. pounce asks the model first. Where the model *proves*
    /// the derivative constant, the reuse happens whether or not the
    /// option was set — the hint is redundant. Where the model proves it
    /// **varies** and the option was set anyway, the option is refused
    /// with a warning, which is the deliberate divergence. Where the
    /// model can prove nothing — every callback front end, both GAMS
    /// links — the user's assertion is honoured on trust, exactly as
    /// upstream, because "unproved" is not "disproved" and silently
    /// overriding the caller there would be its own wrong answer.
    fn install_constant_derivative_hints(&self, orig_nlp: &mut OrigIpoptNlp) {
        use pounce_common::journalist::JournalCategory;
        use pounce_nlp::constant_derivatives::reconcile;

        // Each name is read as a literal rather than through the loop
        // variable: the registered-but-unread scan
        // (`tests/no_silent_options.rs`) keys on the option name as it
        // appears at the accessor, so `get_bool_value(name, "")` read as
        // "no key here" and left all four of these sitting in the silent
        // list while they were fully wired and consumed (#551 / #677).
        //
        // Matching over `HINT_OPTIONS` rather than writing a bare array
        // keeps the order right by construction — `reconcile` pairs
        // `asserted[k]` with `proofs[k]` — and a fifth hint added to
        // `HINT_OPTIONS` trips the fallback arm instead of silently
        // reading as "not asserted", which is the failure mode this
        // whole line of work exists to kill.
        let asserted = self.asserted_constant_derivative_hints();
        let proofs = orig_nlp.derivative_proofs();
        let (outcomes, enabled) = reconcile(proofs, asserted);

        for outcome in &outcomes {
            if let Some(warning) = outcome.warning() {
                eprintln!("{warning}");
                self.journalist.print(
                    JournalLevel::J_STRONGWARNING,
                    JournalCategory::J_MAIN,
                    &format!("{warning}\n"),
                );
            }
        }
        if std::env::var("POUNCE_DBG_CONSTDERIV").is_ok() {
            for outcome in &outcomes {
                eprintln!(
                    "[const deriv] {:<15} proof={:?} asserted={} reused={}",
                    outcome.name, outcome.proof, outcome.asserted, outcome.honoured,
                );
            }
        }
        orig_nlp.set_constant_derivatives(enabled);
    }

    /// Warnings for knobs of a linear-solver backend pounce does not
    /// ship (`ma97_*`, `pardiso_*`, …), one line per backend family.
    ///
    /// Warnings and not refusals: an `ipopt.opt` carrying settings for
    /// several backends so that one file runs everywhere is exactly what
    /// the registry exists to accept, and refusing it would fail a run
    /// over knobs it never touches. See the "Backend knobs warn, they do
    /// not refuse" section of [`crate::unimplemented_options`]. gh#551.
    pub fn unimplemented_backend_warnings(&self) -> Vec<String> {
        crate::unimplemented_options::backend_warnings(&self.options, &self.reg_options)
    }

    /// The same warnings, but at most once per application: the second
    /// caller gets nothing.
    ///
    /// Two sites emit them — the CLI, before routing, because a convex
    /// model never reaches [`Self::optimize_tnlp`], and `optimize_tnlp`
    /// itself, for every frontend that is not the CLI. A CLI run passes
    /// through both, and printing the identical paragraph twice is how a
    /// warning teaches its reader to skip warnings.
    pub fn take_unimplemented_backend_warnings(&mut self) -> Vec<String> {
        if self.backend_warnings_emitted {
            return Vec::new();
        }
        self.backend_warnings_emitted = true;
        self.unimplemented_backend_warnings()
    }

    /// Resolve the five registered `derivative_test*` knobs. Every one
    /// of them was registered and never read, so `derivative_test=
    /// first-order` ran no test and printed nothing — a checker that
    /// silently checks nothing reports success by omission (gh#483
    /// follow-up).
    ///
    /// The numeric helper is named `read_num` to match the accessor
    /// idiom the rest of this file uses: the registered-but-unread scan
    /// (`tests/no_silent_options.rs`) discovers `read_*` helpers and the
    /// literal key passed to them, so a differently-named local closure
    /// made `derivative_test_perturbation` and `derivative_test_tol`
    /// read as silent when they are wired and consumed (#677, #551).
    fn derivative_test_options(&self) -> DerivativeTestOptions {
        let read_num = |key: &str, default: Number| -> Number {
            self.options
                .get_numeric_value(key, "")
                .ok()
                .and_then(|(v, f)| f.then_some(v))
                .unwrap_or(default)
        };
        DerivativeTestOptions {
            mode: self
                .options
                .get_string_value("derivative_test", "")
                .ok()
                .and_then(|(v, f)| f.then_some(v))
                .map(|v| DerivativeTest::from_option(&v))
                .unwrap_or_default(),
            perturbation: read_num("derivative_test_perturbation", 1e-8),
            tol: read_num("derivative_test_tol", 1e-4),
            first_index: self
                .options
                .get_integer_value("derivative_test_first_index", "")
                .ok()
                .and_then(|(v, f)| f.then_some(v))
                .unwrap_or(-2),
            print_all: self
                .options
                .get_bool_value("derivative_test_print_all", "")
                .ok()
                .and_then(|(v, f)| f.then_some(v))
                .unwrap_or(false),
        }
    }

    /// Run the derivative checker, if requested, against `tnlp`.
    ///
    /// Advisory, like upstream: a suspicious entry is reported and the
    /// solve continues. The report goes to stderr so it survives
    /// `print_level=0` and leaves `--json-output`'s stdout clean.
    pub fn run_derivative_test(&self, tnlp: &Rc<RefCell<dyn TNLP>>) {
        let opts = self.derivative_test_options();
        if matches!(opts.mode, DerivativeTest::None) {
            return;
        }
        let report = {
            let mut borrowed = tnlp.borrow_mut();
            pounce_nlp::derivative_test::run(&mut *borrowed, &opts)
        };
        let Some(report) = report else {
            eprintln!(
                "pounce: derivative_test was requested but the TNLP declined to \
                 supply the information the check needs (dimensions, bounds, or \
                 a starting point); no test was run."
            );
            return;
        };
        use pounce_common::journalist::JournalCategory;
        for line in &report.lines {
            eprintln!("{line}");
            self.journalist.print(
                JournalLevel::J_SUMMARY,
                JournalCategory::J_MAIN,
                &format!("{line}\n"),
            );
        }
    }

    /// The message [`Self::unimplemented_linear_solver`] earns, shared by
    /// every frontend so they cannot drift apart.
    pub fn unimplemented_linear_solver_message(value: &str) -> String {
        format!(
            "pounce: linear_solver={value} is not implemented. pounce provides \
             `feral` (pure-Rust sparse symmetric, the default) and `ma57` (HSL, \
             in a `--features ma57` build); the other names in the option's \
             list come from the upstream Ipopt registry so an ipopt.opt written \
             for Ipopt still parses. Selecting one used to run FERAL silently, \
             which makes a backend comparison measure nothing — so it is \
             refused instead. Use linear_solver=feral or linear_solver=ma57."
        )
    }

    fn is_sqp_algorithm_selected(&self) -> bool {
        // `algorithm` is the primary selector.
        // `solver_selection = qp-active-set` selects the
        // same active-set SQP engine.
        let algo_sqp = matches!(
            self.options.get_string_value("algorithm", ""),
            Ok((v, true)) if v.eq_ignore_ascii_case("active-set-sqp")
        );
        let selection_sqp = matches!(
            self.options.get_string_value("solver_selection", ""),
            Ok((v, true)) if v.eq_ignore_ascii_case("qp-active-set")
        );
        algo_sqp || selection_sqp
    }

    /// Phase 5b SQP entry point. Builds the same NLP chain
    /// (`TNLPAdapter` → `OrigIpoptNlp` → `IpoptNlpAdapter`) the
    /// IPM uses, then runs `SqpAlgorithm::optimize`. Maps the
    /// `SqpResult.status` back to `ApplicationReturnStatus` and
    /// hands the final iterate to the user TNLP's
    /// `finalize_solution` callback via `finalize_via_sqp`.
    fn optimize_sqp_tnlp(&mut self, tnlp: Rc<RefCell<dyn TNLP>>) -> ApplicationReturnStatus {
        use pounce_nlp::ConstObjScaling;
        use pounce_nlp::orig_ipopt_nlp::OrigIpoptNlp;
        use pounce_nlp::tnlp_adapter::TNLPAdapter;

        // Wall-clock for the whole SQP solve, mirroring the IPM path's
        // `t_start` (see the `total_wallclock_time_secs` assignment in
        // `optimize_tnlp`). Without this the field stayed at its struct
        // default of 0.0 on every active-set solve, so `--json-output`
        // reported an instantaneous solve regardless of actual runtime and
        // the engine could not be speed-compared against qp-ipm at all
        // (benchmarks/scripts/compare_qp_four_way.py had to skip the column).
        let t_start = std::time::Instant::now();

        let adapter = match TNLPAdapter::new(Rc::clone(&tnlp)) {
            Ok(a) => Rc::new(RefCell::new(a)),
            Err(_) => return ApplicationReturnStatus::InvalidProblemDefinition,
        };
        // The SQP path never runs gradient-based scaling, but the
        // constant `obj_scaling_factor` (negative ⇒ maximize) still
        // applies via the OrigIpoptNlp constructor.
        let obj_scaling_factor = self
            .options
            .get_numeric_value("obj_scaling_factor", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(1.0);
        let mut orig_nlp = match OrigIpoptNlp::new(
            Rc::clone(&adapter),
            Rc::new(ConstObjScaling(obj_scaling_factor)),
        ) {
            Ok(n) => n,
            Err(_) => return ApplicationReturnStatus::InternalError,
        };
        // Same Q6 reconciliation as the IPM route: the SQP driver
        // evaluates the same derivatives through the same NLP object.
        self.install_constant_derivative_hints(&mut orig_nlp);
        let nlp_rc: Rc<RefCell<dyn IpoptNlp>> = Rc::new(RefCell::new(orig_nlp));

        let mut sqp_adapter = crate::sqp::IpoptNlpAdapter::new(Rc::clone(&nlp_rc));

        let mut builder = self.algorithm_builder_snapshot();
        builder.algorithm = crate::alg_builder::AlgorithmChoice::ActiveSetSqp;
        let factory = self.make_backend_factory();
        let mut alg = match builder.build_sqp_with_backend(factory) {
            Some(a) => a,
            None => return ApplicationReturnStatus::InternalError,
        };

        // Problem statistics + end-of-run summary are emitted by the engine
        // itself here (#206), gated on the main `print_level`, so the SQP
        // route matches the IPM route across every frontend (CLI, Python, C).
        // The SQP's own per-iteration rows stay gated on the separate
        // `sqp_print_level`.
        let console_output = match self.options.get_integer_value("print_level", "") {
            Ok((v, true)) => v >= 1,
            _ => true,
        };
        self.emit_problem_stats(&tnlp, console_output);

        // Phase 5c (§6): consume any stashed warm-start iterate.
        // `optimize_with_warm_start(warm=None)` is equivalent to
        // `optimize`, so cold callers see no change.
        let warm = self.sqp_warm_start.take();
        let res = match alg.optimize_with_warm_start(&mut sqp_adapter, warm) {
            Ok(r) => r,
            Err(e) => {
                // Always surface this. It used to be gated on the
                // undocumented `POUNCE_DBG_SQP`, so the only thing a user saw
                // was a bare `Internal_Error` with no indication of what went
                // wrong -- the underlying message here was
                // `QpFailure(LinearSolverFailure("QP subproblem returned
                // status unbounded"))`, which points straight at the cause.
                // A solve that is about to fail is exactly when the reason
                // should be cheapest to obtain.
                tracing::warn!(
                    target: "pounce::sqp",
                    "SQP solve failed: {e:?}"
                );
                return ApplicationReturnStatus::InternalError;
            }
        };
        // Stash the result's working set so the next solve in a
        // sequence can fetch it via `last_sqp_working_set`.
        self.sqp_last_working_set = res.working_set.clone();
        // Populate the shared `SolveStatistics` so the Python /
        // C-API post-solve accessors (`GetIpoptIterCount`,
        // `info["iter_count"]`, etc.) report the SQP outer-iter
        // count rather than zero. Constraint-violation /
        // dual-infeasibility residuals get the SQP-side values
        // too. The IPM path overwrites this dict on its own
        // solves, so SQP-vs-IPM mixing across solves stays
        // honest.
        {
            let mut stats = self.statistics.borrow_mut();
            stats.iteration_count = res.n_iter as Index;
            // Subproblem counters. The outer iteration count alone
            // cannot show what a working-set warm start bought — the
            // saved work is inside the QPs — so both are reported.
            stats.sqp_qp_solves = res.n_qp_solves as Index;
            stats.sqp_qp_working_set_changes = res.n_qp_working_set_changes as Index;
            stats.final_objective = res.obj;
            // `final_scaled_objective` defaults to NaN; the SQP path does not
            // thread nlp_scaling through the objective (same as the residuals
            // mirrored below), so the scaled objective equals the unscaled
            // one. Without this it stayed NaN and the console printed
            // "Objective ...: nan  <unscaled>" on every active-set solve
            // (gh #313), even on a clean optimal solve.
            stats.final_scaled_objective = res.obj;
            stats.final_dual_inf = res.final_stationarity;
            stats.final_constr_viol = res.final_constr_viol;
            stats.final_compl = 0.0; // SQP has no barrier — no compl term.
            // Overall KKT error. This was previously left at the struct
            // default, which made every successful SQP solve report an
            // overall error of exactly 0.0 — indistinguishable from a
            // genuinely perfect solve, and enough on its own to make
            // `pounce.minimize`'s acceptable-KKT fallback upgrade any status
            // on this path to `success=True`. Same expression as the unscaled
            // twin below; the two agree because the SQP path does not thread
            // nlp_scaling through its residuals.
            stats.final_kkt_error = res.final_stationarity.max(res.final_constr_viol);
            // Unscaled residuals (pounce#173). The SQP path does not thread
            // the nlp_scaling factors through to its residuals yet, so these
            // mirror the SQP-side values: correct when no scaling is active
            // (the common case) and a conservative proxy otherwise. Populated
            // here so the info dict's `final_unscaled_*` keys are honest
            // rather than left at the 0.0 default.
            stats.final_unscaled_dual_inf = res.final_stationarity;
            stats.final_unscaled_constr_viol = res.final_constr_viol;
            stats.final_unscaled_compl = 0.0;
            stats.final_unscaled_kkt_error = res.final_stationarity.max(res.final_constr_viol);
            stats.total_wallclock_time_secs = t_start.elapsed().as_secs_f64();
        }
        let (app_status, solver_status) = match res.status {
            crate::sqp::SqpStatus::Optimal => (
                ApplicationReturnStatus::SolveSucceeded,
                pounce_nlp::SolverReturn::Success,
            ),
            crate::sqp::SqpStatus::MaxIter => (
                ApplicationReturnStatus::MaximumIterationsExceeded,
                pounce_nlp::SolverReturn::MaxiterExceeded,
            ),
            crate::sqp::SqpStatus::InfeasibleSubproblem => (
                ApplicationReturnStatus::InfeasibleProblemDetected,
                pounce_nlp::SolverReturn::LocalInfeasibility,
            ),
            crate::sqp::SqpStatus::LineSearchFailed => (
                ApplicationReturnStatus::SearchDirectionBecomesTooSmall,
                pounce_nlp::SolverReturn::ErrorInStepComputation,
            ),
            // Honest non-committal QP-subproblem failure (#282): the QP
            // solver could not compute a step and did NOT certify
            // infeasibility. Never report Infeasible_Problem_Detected here
            // — a feasible problem has no infeasibility certificate.
            crate::sqp::SqpStatus::QpStepFailed => (
                ApplicationReturnStatus::SearchDirectionBecomesTooSmall,
                pounce_nlp::SolverReturn::ErrorInStepComputation,
            ),
            // The QP subproblem ran out of its own iteration budget. Same
            // #282 guarantee — no infeasibility is asserted — but reported as
            // the budget exhaustion it is, so the user sees a limit they can
            // raise (`sqp_qp_max_iter`) instead of a step-size stall with no
            // remedy. See `SqpStatus::QpIterationLimit` for why these were
            // split.
            crate::sqp::SqpStatus::QpIterationLimit => (
                ApplicationReturnStatus::MaximumIterationsExceeded,
                pounce_nlp::SolverReturn::MaxiterExceeded,
            ),
            // Unbounded below, with a recession ray verified against the
            // true NLP (gh #388). `Diverging_Iterates` is POUNCE's (Ipopt's)
            // unboundedness verdict and maps to AMPL `solve_result_num=300`
            // — the same answer the IPM selectors give on the same model,
            // instead of the `Internal_Error` / 500 ("the solver broke")
            // this path used to report.
            crate::sqp::SqpStatus::Unbounded => (
                ApplicationReturnStatus::DivergingIterates,
                pounce_nlp::SolverReturn::DivergingIterates,
            ),
            // A non-finite iterate or constraint value (gh #876). Same
            // verdict, from the same condition, as the interior-point arm's
            // `if !nlp_err.is_finite()` screen — the two arms must not
            // disagree about what a `NaN` iterate means.
            crate::sqp::SqpStatus::InvalidNumber => (
                ApplicationReturnStatus::InvalidNumberDetected,
                pounce_nlp::SolverReturn::InvalidNumberDetected,
            ),
        };

        // Same gate as the IPM path: an infeasible-subproblem exit is a
        // numerical inference, and a feasible starting point disproves it
        // (gh #379). Only the infeasibility verdict is rewritten — every other
        // status passes through untouched, so the pair stays in lockstep.
        let refuted = withdraw_infeasibility_if_refuted(
            &tnlp,
            solver_status,
            self.nlp_lower_bound_inf(),
            self.nlp_upper_bound_inf(),
            self.user_tol(),
        );
        let (app_status, solver_status) = if refuted == solver_status {
            (app_status, solver_status)
        } else {
            (solver_return_to_app_status(refuted), refuted)
        };

        // Forward to the user TNLP's finalize_solution. We pass
        // the SQP iterate and recovered multipliers via the
        // OrigIpoptNlp's lifting hooks. Failure here is silent
        // (we still return the algorithm's status) — the user
        // sees the right ApplicationReturnStatus regardless.
        let _ = finalize_via_sqp(&nlp_rc, &res, solver_status, &tnlp, &self.last_finalize);

        // Honor the opt-in status-fidelity gate on the SQP path too
        // (pounce#173), then emit the end-of-run summary with the final
        // (possibly downgraded) status so the console matches the returned
        // ApplicationReturnStatus.
        let final_status = self.apply_kkt_fidelity_gate(app_status);
        self.emit_end_summary(final_status, &nlp_rc, console_output);
        final_status
    }

    /// Opt-in status-fidelity gate (pounce#173), shared by the IPM and
    /// SQP solve paths. When the user sets a positive `kkt_fidelity_tol`,
    /// a reported `Solve_Succeeded` whose max-norm UNSCALED KKT error
    /// (`SolveStatistics::final_unscaled_kkt_error`) exceeds it is
    /// downgraded to `Solved_To_Acceptable_Level` — the honest "this is a
    /// point, but not converged to the requested fidelity" status. This
    /// catches the ill-conditioned / nlp_scaling-deflated case where the
    /// scaled convergence test passes but the user-space duals have
    /// drifted. It is a pure relabel at termination (no extra iterations);
    /// unset or non-positive (the default) is a strict no-op, so every
    /// existing caller keeps the Ipopt-faithful status.
    fn apply_kkt_fidelity_gate(
        &self,
        app_status: ApplicationReturnStatus,
    ) -> ApplicationReturnStatus {
        if !matches!(app_status, ApplicationReturnStatus::SolveSucceeded) {
            return app_status;
        }
        if let Ok((ftol, true)) = self.options.get_numeric_value("kkt_fidelity_tol", "") {
            if ftol > 0.0 {
                let unscaled_kkt = self.statistics.borrow().final_unscaled_kkt_error;
                if unscaled_kkt > ftol {
                    tracing::info!(target: "pounce::diagnostics",
                        "kkt_fidelity_tol={ftol:.3e}: unscaled KKT error {unscaled_kkt:.3e} \
                         exceeds it — downgrading Solve_Succeeded → \
                         Solved_To_Acceptable_Level (pounce#173)");
                    return ApplicationReturnStatus::SolvedToAcceptableLevel;
                }
            }
        }
        app_status
    }

    /// `nlp_lower_bound_inf` — the magnitude at or below which a bound is
    /// treated as absent.
    fn nlp_lower_bound_inf(&self) -> Number {
        self.options
            .get_numeric_value("nlp_lower_bound_inf", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(DEFAULT_NLP_LOWER_BOUND_INF)
    }

    /// `nlp_upper_bound_inf` — the magnitude at or above which a bound is
    /// treated as absent.
    fn nlp_upper_bound_inf(&self) -> Number {
        self.options
            .get_numeric_value("nlp_upper_bound_inf", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(DEFAULT_NLP_UPPER_BOUND_INF)
    }

    /// The user's convergence tolerance `tol`.
    fn user_tol(&self) -> Number {
        self.options
            .get_numeric_value("tol", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(1e-8)
    }

    /// The user's `acceptable_tol` — the standard behind
    /// `Solved_To_Acceptable_Level`.
    fn user_acceptable_tol(&self) -> Number {
        self.options
            .get_numeric_value("acceptable_tol", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(1e-6)
    }

    /// The user's `constr_viol_tol` — the **absolute** feasibility standard
    /// the strict gate's primal component judges by
    /// (`OptErrorConvCheck::primal_component_passes`).
    fn user_constr_viol_tol(&self) -> Number {
        self.options
            .get_numeric_value("constr_viol_tol", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(1e-4)
    }

    /// The user's `acceptable_constr_viol_tol` — the same standard at the
    /// acceptable tier.
    fn user_acceptable_constr_viol_tol(&self) -> Number {
        self.options
            .get_numeric_value("acceptable_constr_viol_tol", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(1e-2)
    }

    /// `primal_noise_floor_kappa` — the safety factor on the per-row
    /// floating-point noise floor (gh#528/gh#590). `0` opts out.
    fn user_primal_noise_floor_kappa(&self) -> Number {
        self.options
            .get_numeric_value("primal_noise_floor_kappa", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(64.0)
    }

    /// Emit the Ipopt-style problem-statistics block (#206) from the
    /// engine's own reduced problem, gated on `console_output`
    /// (print_level >= 1). Shared by the IPM (`optimize_tnlp`) and SQP
    /// (`optimize_sqp_tnlp`) entry points so every algorithm and every
    /// frontend (CLI, Python, C) gets the identical block. Built from the
    /// same `collect_stats` inputs the CLI used, so the output is
    /// byte-identical to the historical CLI block.
    fn emit_problem_stats(&self, tnlp: &Rc<RefCell<dyn TNLP>>, console_output: bool) {
        if !console_output {
            return;
        }
        let lo_inf = self
            .options
            .get_numeric_value("nlp_lower_bound_inf", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(DEFAULT_NLP_LOWER_BOUND_INF);
        let up_inf = self
            .options
            .get_numeric_value("nlp_upper_bound_inf", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(DEFAULT_NLP_UPPER_BOUND_INF);
        let fixed_treatment = match self
            .options
            .get_string_value("fixed_variable_treatment", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .as_deref()
        {
            Some("relax_bounds") => FixedVarTreatment::RelaxBounds,
            _ => FixedVarTreatment::MakeParameter,
        };
        if let Some(stats) =
            pounce_solve_report::console::collect_stats(tnlp, lo_inf, up_inf, fixed_treatment)
        {
            pounce_solve_report::console::print_problem_stats(&stats);
        }
    }

    /// Drain the NLP's per-eval counters into the shared `SolveStatistics`
    /// and emit the Ipopt-style end-of-run summary (#206). Shared by both
    /// solve paths. The counts are read from the NLP AFTER the solve (so the
    /// final solution evaluation is included, matching the historical count)
    /// and written into `SolveStatistics` so the post-solve API accessors
    /// (`info["n_obj_evals"]`, …) report them even when the console is
    /// silent. c/d (and jac_c/jac_d) are per-subsystem, so the max recovers
    /// the eval_g / eval_jac_g call count. The console summary itself is
    /// gated on `console_output` (print_level >= 1).
    fn emit_end_summary(
        &self,
        app_status: ApplicationReturnStatus,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        console_output: bool,
    ) {
        {
            let ec = nlp.borrow().eval_counts();
            let mut stats = self.statistics.borrow_mut();
            stats.num_obj_evals = ec[0];
            stats.num_obj_grad_evals = ec[1];
            stats.num_constr_evals = ec[2].max(ec[3]);
            stats.num_constr_jac_evals = ec[4].max(ec[5]);
            stats.num_hess_evals = ec[6];
        }
        if !console_output {
            return;
        }
        let stats = self.statistics.borrow();
        let counts = pounce_solve_report::console::EvalCounts {
            n_obj: stats.num_obj_evals as u64,
            n_grad_f: stats.num_obj_grad_evals as u64,
            n_g: stats.num_constr_evals as u64,
            n_jac_g: stats.num_constr_jac_evals as u64,
            n_h: stats.num_hess_evals as u64,
        };
        pounce_solve_report::console::print_summary(app_status, &stats, &counts);
    }

    /// Build a *copy* of the algorithm builder configured per the
    /// current options. The SQP path uses this so it gets a
    /// fresh builder without mutating the application's state.
    /// Read the `crossover*` option family into a [`CrossoverOptions`].
    fn crossover_options(&self) -> crate::crossover::CrossoverOptions {
        let mut o = crate::crossover::CrossoverOptions::default();
        if let Ok((v, true)) = self.options.get_bool_value("crossover", "") {
            o.enabled = v;
        }
        if let Ok((v, true)) = self.options.get_integer_value("crossover_max_iter", "") {
            o.max_iter = v.max(0) as u32;
        }
        if let Ok((v, true)) = self.options.get_numeric_value("crossover_mult_tol", "") {
            o.mult_tol = v;
        }
        if let Ok((v, true)) = self.options.get_numeric_value("crossover_primal_tol", "") {
            o.primal_tol = v;
        }
        o
    }

    /// Post-convergence crossover (gh#612): hand the converged interior
    /// iterate to the active-set path so the solve ends on an exact active
    /// set. See [`crate::crossover`] for the algorithm.
    ///
    /// On acceptance this **replaces `data.curr`** rather than reporting
    /// alongside it. Everything downstream — the residual drain, the KKT
    /// fidelity gate, `on_converged`, `finalize_via_orig_nlp`, the end
    /// summary — already reads that one iterate, so replacing it is what
    /// makes the crossed-over point the solution instead of an annotation on
    /// it, and it does so without a second copy of the unscaling path.
    ///
    /// A no-op unless `crossover=yes` and the solve converged: an
    /// unconverged interior point is not a KKT point, so there is no active
    /// set at it worth identifying.
    fn maybe_crossover(
        &mut self,
        alg: &mut IpoptAlgorithm,
        nlp_handle: &Rc<RefCell<dyn IpoptNlp>>,
        solver_status: SolverReturn,
    ) {
        self.crossover_report = None;
        // Cleared on every entry, alongside the report, so the flag
        // describes this solve and not a previous one — the same reason
        // `crossover_report` is reset here rather than only written on
        // the accept path.
        alg.data.borrow_mut().curr_from_crossover = false;
        let xopts = self.crossover_options();
        if !xopts.enabled {
            return;
        }
        if !matches!(
            solver_status,
            SolverReturn::Success | SolverReturn::StopAtAcceptablePoint
        ) {
            return;
        }
        let Some(curr) = alg.data.borrow().curr.clone() else {
            return;
        };

        // Seed, in the algorithm's compressed / scaled space. The SQP
        // adapter presents that same space, so nothing is translated here
        // beyond repacking the bound duals: `λ_x = z_l − z_u`.
        let seed = crate::crossover::CrossoverSeed {
            x: dense_values(&*curr.x),
            lambda_g: {
                let mut v = dense_values(&*curr.y_c);
                v.extend(dense_values(&*curr.y_d));
                v
            },
            lambda_x: crate::sqp::ipopt_adapter::pack_bound_multipliers(
                nlp_handle,
                &dense_values(&*curr.z_l),
                &dense_values(&*curr.z_u),
            ),
        };

        let snapshot = self.algorithm_builder_snapshot();
        let sqp_opts = snapshot.sqp.clone();
        let qp_opts = snapshot.sqp_qp.clone();
        // Declared bounds, not the live relaxed ones: crossover's whole claim
        // is that the returned point sits *on* the constraints the user
        // wrote. Against the `bound_relax_factor`-widened box it would pivot
        // to a point `1e-8` shy of every one of them and then correctly
        // report an empty active set.
        let mut adapter =
            crate::sqp::IpoptNlpAdapter::new_with_declared_bounds(Rc::clone(nlp_handle));
        let (report, accepted) = crate::crossover::run(
            &mut adapter,
            &seed,
            &xopts,
            &sqp_opts,
            &qp_opts,
            || {
                let mut f = self.make_backend_factory();
                // `make_backend_factory` ships the workspace-default
                // backend regardless of the choice passed, exactly as the
                // SQP path's `build_sqp_with_backend` call does; naming
                // `Feral` here keeps that visible rather than implicit.
                f(crate::alg_builder::LinearSolverChoice::Feral)
            },
            |step4_opts| {
                let mut b = self.algorithm_builder_snapshot();
                b.algorithm = crate::alg_builder::AlgorithmChoice::ActiveSetSqp;
                b.sqp = step4_opts;
                b.build_sqp_with_backend(self.make_backend_factory())
            },
        );

        if let Some(res) = accepted {
            self.install_crossover_iterate(alg, nlp_handle, &curr, &res);
            // Publish the identified set as the SQP warm-start output. This
            // is the IPM → SQP handoff the active-set path never had: a
            // sequence whose first solve wants the interior method can now
            // feed the next `algorithm=active-set-sqp` solve a working set.
            self.sqp_last_working_set = res.working_set.clone();
        }
        tracing::debug!(target: "pounce::crossover", "crossover: {report:?}");
        self.crossover_report = Some(report);
    }

    /// Write an accepted crossover result back onto the IPM iterate.
    ///
    /// All eight components have to move together: leaving `s`, `v_l` or
    /// `v_u` describing the interior point while `x` and the duals describe
    /// the crossed-over one would make the calculated quantities read off an
    /// iterate that never existed, and the residuals reported to the user
    /// would be neither point's. The slack relations are the barrier
    /// problem's own — `s = d(x)` and `v_l − v_u = −y_d`.
    fn install_crossover_iterate(
        &self,
        alg: &mut IpoptAlgorithm,
        nlp_handle: &Rc<RefCell<dyn IpoptNlp>>,
        curr: &crate::iterates_vector::IteratesVector,
        res: &crate::sqp::SqpResult,
    ) {
        let (m_c, m_d) = {
            let b = nlp_handle.borrow();
            (b.m_eq() as usize, b.m_ineq() as usize)
        };
        let y_c = &res.lambda_g[..m_c];
        let y_d = &res.lambda_g[m_c..];
        // `s = d(x)`: the adapter's combined constraint vector is `[c ; d]`,
        // so the inequality block is its tail.
        let mut adapter = crate::sqp::IpoptNlpAdapter::new(Rc::clone(nlp_handle));
        let c_all = crate::sqp::SqpProblemSpec::eval_c(&mut adapter, &res.x);
        let s_new = &c_all[m_c..];
        debug_assert_eq!(s_new.len(), m_d);
        let (z_l, z_u) =
            crate::sqp::ipopt_adapter::split_bound_multipliers(nlp_handle, &res.lambda_x);
        let (v_l, v_u) = crate::sqp::ipopt_adapter::split_slack_multipliers(nlp_handle, y_d);

        let mut out = curr.deep_copy();
        let ok = set_dense(&mut *out.x, &res.x)
            && set_dense(&mut *out.s, s_new)
            && set_dense(&mut *out.y_c, y_c)
            && set_dense(&mut *out.y_d, y_d)
            && set_dense(&mut *out.z_l, &z_l)
            && set_dense(&mut *out.z_u, &z_u)
            && set_dense(&mut *out.v_l, &v_l)
            && set_dense(&mut *out.v_u, &v_u);
        if !ok {
            // A non-dense backing or a length mismatch. POUNCE is dense-only,
            // so this is defensive — but a partially-written iterate is worse
            // than no crossover at all, so bail without touching `curr`.
            tracing::warn!(
                target: "pounce::crossover",
                "crossover result did not fit the iterate; keeping the interior point"
            );
            return;
        }
        let mut d = alg.data.borrow_mut();
        d.set_curr(out.freeze());
        // Mark which frame the installed iterate belongs to. It sits on the
        // *declared* bounds, and every barrier quantity built off `curr` —
        // slacks, and through them `Σ = z/s` — is measured against the
        // `bound_relax_factor`-widened ones, so a consumer that wants the
        // point's own geometry rather than the barrier's has to know this
        // happened (gh#654). Set only on the path that actually replaced
        // `curr`: a declined or abandoned crossover leaves the interior
        // iterate, which is an interior-frame point.
        d.curr_from_crossover = true;
    }

    fn algorithm_builder_snapshot(&self) -> AlgorithmBuilder {
        let mut builder = AlgorithmBuilder {
            quality_escalation_counter: Some(Rc::clone(&self.quality_escalations)),
            ..AlgorithmBuilder::default()
        };
        apply_sqp_options(&self.options, &mut builder.sqp);
        apply_qp_subproblem_options(&self.options, &mut builder.sqp_qp);
        builder
    }

    /// Refuse an explicitly set `ma57_pivtolmax` that sits below
    /// `ma57_pivtol`, at either option prefix.
    ///
    /// Upstream's `Ma57TSolverInterface::InitializeImpl` asserts
    /// `pivtolmax >= pivtol` and raises `OPTION_INVALID`, but only when
    /// the user set `ma57_pivtolmax` explicitly; left unset, the
    /// registered default is lifted to `ma57_pivtol` instead. Both
    /// halves are mirrored — the lifting in
    /// `pounce_hsl::ma57::Options::from_options_list`, the refusal here.
    ///
    /// Checked at **both** prefixes because the restoration sub-IPM
    /// configures its own MA57 backend from `"resto."`-scoped options
    /// (gh#825), so `resto.ma57_pivtolmax` can contradict
    /// `resto.ma57_pivtol` without the un-prefixed pair being wrong.
    ///
    /// Deliberately **not** gated on the `ma57` cargo feature or on
    /// `linear_solver` resolving to MA57. It is a consistency check on
    /// two numbers the user wrote, needs no HSL to perform, and a
    /// verdict that changed with a build flag would be worse than a
    /// consistent one — it would also be untestable in CI, which cannot
    /// link CoinHSL. Only an *explicitly set* `ma57_pivtolmax` can
    /// trigger it, so an options file that never mentions the option is
    /// unaffected.
    fn ma57_pivtol_bracket_refusal(&self) -> Option<String> {
        for prefix in ["", "resto."] {
            // `(_, true)` is the explicitly-set arm; an unset option
            // reports the registry default with `false` and is the
            // branch upstream lifts rather than refuses.
            let Ok((pivtolmax, true)) = self.options.get_numeric_value("ma57_pivtolmax", prefix)
            else {
                continue;
            };
            let pivtol = self
                .options
                .get_numeric_value("ma57_pivtol", prefix)
                .map(|(v, _)| v)
                .unwrap_or(1e-8);
            if pivtolmax < pivtol {
                return Some(format!(
                    "pounce: {prefix}ma57_pivtolmax ({pivtolmax:e}) is below \
                     {prefix}ma57_pivtol ({pivtol:e}). ma57_pivtolmax is the ceiling MA57 \
                     may raise the pivot tolerance to when it escalates for accuracy, so it \
                     cannot sit below the tolerance it starts from. Raise \
                     {prefix}ma57_pivtolmax to at least {pivtol:e}, or lower \
                     {prefix}ma57_pivtol."
                ));
            }
        }
        None
    }

    /// Construct a LinearBackendFactory honoring the
    /// `linear_solver` option. Default FERAL; HSL MA57 when
    /// built with the `ma57` feature.
    fn make_backend_factory(&self) -> LinearBackendFactory {
        Box::new(
            |_choice| -> Box<dyn pounce_linsol::SparseSymLinearSolverInterface> {
                Box::new(pounce_feral::FeralSolverInterface::new())
            },
        )
    }

    /// Phase 3.5 auto-fallback driver.
    ///
    /// Runs the standard solve (no wrapper) first. If it ends in a
    /// trigger-class status (`Restoration_Failed`, `Infeasible_Problem_Detected`,
    /// `Solved_To_Acceptable_Level`, `Maximum_Iterations_Exceeded`, or
    /// `Not_Enough_Degrees_Of_Freedom`), retries transparently with
    /// the ℓ₁ wrapper enabled. Promotes the retry's status only if
    /// it returns `Solve_Succeeded`; otherwise returns the original
    /// status.
    ///
    /// Caveat: the user TNLP's `finalize_solution` runs once per
    /// attempt. When the retry doesn't promote, the user's captured
    /// fields hold the retry's iterate (the ℓ₁-best least-infeasible
    /// point) even though the returned status is the original's.
    /// Documented on the option's help text; tightening this is a
    /// Phase-4 follow-up.
    fn run_with_l1_fallback(&mut self, tnlp: Rc<RefCell<dyn TNLP>>) -> ApplicationReturnStatus {
        // First attempt: the standard IPM solve, no ℓ₁ wrapper. Only
        // reached for `m > 0`, so `optimize_constrained` is exact.
        let first_status = self.optimize_constrained(Rc::clone(&tnlp));
        if !is_l1_fallback_trigger(first_status) {
            return first_status;
        }
        // Trigger fired. Flip the wrapper option for the retry and
        // restore it after — keeps the user's option-table view of the
        // session exactly as they left it.
        let prev = self
            .options
            .get_string_value("l1_exact_penalty_barrier", "")
            .ok();
        let _ = self
            .options
            .set_string_value("l1_exact_penalty_barrier", "yes", true, false);
        let retry_status = self
            .run_l1_penalty_outer_loop(Rc::clone(&tnlp))
            .unwrap_or(ApplicationReturnStatus::InternalError);
        let _ = self.options.set_string_value(
            "l1_exact_penalty_barrier",
            prev.as_ref().map(|(v, _)| v.as_str()).unwrap_or("no"),
            true,
            false,
        );
        if matches!(retry_status, ApplicationReturnStatus::SolveSucceeded) {
            retry_status
        } else {
            first_status
        }
    }

    /// μ-strategy auto-fallback driver (pounce#138).
    ///
    /// Runs the standard solve first. If it stalls short of optimal in a
    /// way a μ-strategy flip can plausibly fix — `Solved_To_Acceptable_Level`
    /// or `Maximum_Iterations_Exceeded`, the two signatures seen on the
    /// princetonlib instances where the dual infeasibility parks above
    /// `tol` while constraint violation and complementarity are already
    /// deeply converged — it flips `mu_strategy` (adaptive↔monotone) and
    /// solves once more. The retry's status is promoted only if it returns
    /// `Solve_Succeeded`; otherwise the original status is returned.
    ///
    /// (maxcut/price stall at acceptable-level under adaptive; fermat2_vareps
    /// stalls at `max_iter` — hence both triggers. flosp2tm is μ-independent
    /// and correctly does not promote.)
    ///
    /// The flip direction is taken from the strategy the option table
    /// actually resolves to (`effective_mu_strategy_is_adaptive`):
    /// `adaptive` → `monotone`, otherwise → `adaptive`. Absence is not
    /// the same as `monotone` — under a limited-memory Hessian an unset
    /// `mu_strategy` resolves to `adaptive` (gh#746), and flipping the
    /// *registered* default there would re-run the strategy that just
    /// stalled. The option table is restored to the resolved view
    /// afterward.
    ///
    /// Caveat (shared with the ℓ₁ fallback): the user TNLP's
    /// `finalize_solution` runs once per attempt, so when the retry
    /// doesn't promote the captured fields hold the retry's iterate.
    fn run_with_mu_strategy_fallback(
        &mut self,
        tnlp: Rc<RefCell<dyn TNLP>>,
    ) -> ApplicationReturnStatus {
        let first_status = self.optimize_constrained(Rc::clone(&tnlp));
        // Which statuses are worth a second solve depends on who asked
        // for the retry (pounce#748).
        //
        // An explicit `mu_strategy_fallback=yes` keeps the historical
        // pair: the caller opted in and can afford the second solve.
        //
        // The *default*-on retry takes `Maximum_Iterations_Exceeded`
        // unconditionally, and `Solved_To_Acceptable_Level` only when
        // the caller left the convergence configuration alone
        // (gh #757). pounce#748 refused the latter status outright, for
        // three reasons; two of them are properties of a *caller-
        // modified* configuration, not of the status. It launders
        // downgrades the caller induced deliberately -- a tight
        // `kkt_fidelity_tol`, a certificate veto, `least_square_init_
        // primal` -- so the signal the option exists to produce never
        // reaches them. And because the retry returns the other run's
        // *point*, not just its status, it can hand back a different
        // local solution: on `autocorr_bern55-06` with the
        // dual-divergence guard on it swaps -2304.0000278 for
        // -2320.0000298 (crates/pounce-cli/tests/
        // issue_250_dual_guard_never_worse.rs). Both cases -- and all
        // five test targets the wide trigger broke -- arm a
        // non-default option from `TERMINATION_POLICY_OPTIONS`, so
        // deferring to that set preserves every one of them while
        // leaving a stock-options stall retryable. The third reason,
        // cost, stands and is the price: one extra solve on a run that
        // reached only the acceptable tolerance, paid to try for the
        // certificate. `cho_parmest` is the motivating case -- monotone
        // parks `inf_du` on a ~1e-6 evaluation-noise floor and misses
        // `tol` by 5%, taking six null steps of 1e-12 at `mu_min`,
        // while adaptive certifies it in 20 iterations.
        //
        // `dirichlet120`, the case that motivated turning the retry on,
        // stalls at `Maximum_Iterations_Exceeded`, so it is recovered
        // either way.
        let retry_worthy = match first_status {
            ApplicationReturnStatus::MaximumIterationsExceeded => true,
            ApplicationReturnStatus::SolvedToAcceptableLevel => {
                self.mu_strategy_fallback_was_set() || !self.caller_set_termination_policy()
            }
            _ => false,
        };
        if !retry_worthy {
            return first_status;
        }
        // gh#857: decline the flip when the solve that just failed escalated
        // the factorization and an escalation-off re-solve is enabled.
        //
        // The mu flip is a *blind* second opinion: it changes the barrier
        // schedule and hopes. A `feral_increase_quality` escalation is a
        // *measured* fact about the run that just failed -- FERAL reroutes
        // which pivots are taken and never steps back down, so every
        // iteration after the first escalation, restoration sub-solves
        // included, ran on a trajectory the defaults do not describe. Flipping
        // `mu_strategy` while leaving that in place is not a controlled
        // experiment: it varies the knob that is not implicated and holds the
        // one that is.
        //
        // It is not free either. On `square_flowsheet_resto`'s lbfgs leg the
        // flip escalates 25 times all over again, burns a second full
        // 3000-iteration budget, and ends no better than the first -- after
        // which rung 4 of the second-opinion ladder converges the model in 178
        // with the escalation off. That is 6178 real iterations to reach an
        // answer that 3178 reach without the flip, and the flip contributes
        // nothing to it. Measured both ways: `mu_strategy=adaptive` alone
        // still gives 3000 with 25 escalations, and `feral_increase_quality=no`
        // gives 178 under *either* mu strategy.
        //
        // Why decline rather than fold the escalation off into this retry: the
        // backend factory is minted from an options snapshot taken by the
        // caller *before* `solve()`, so writing `feral_increase_quality` here
        // is too late to reach the retry's linear solver. `mu_strategy` is read
        // per-solve from the option table and is not; that asymmetry is why
        // this layer can only choose whether to spend the solve, not what to
        // spend it on.
        //
        // Restricted to `Maximum_Iterations_Exceeded`, which is exactly the
        // status rung 4 opens on. A `Solved_To_Acceptable_Level` exit opens no
        // escalation rung, so declining there would drop a retry with nothing
        // in its place. Gated on `feral_increase_quality_retry`, so setting
        // that option to `no` restores the historical behaviour on both sides
        // at once: no rung 4, and no decline here.
        //
        // The one place the stand-down is not paired with the rung is the
        // multi-start paths (`solve_nlp_batch`, the CLI's `minima` search),
        // which deliberately do not drive the ladder -- a failed start there is
        // routine and extra solves per start multiply. Those paths lose the
        // flip on an escalating budget exit and gain nothing back, which is the
        // one behaviour change here that is not a strict improvement. It is the
        // same trade they already take on every other rung, and for the same
        // reason: an escalating capped start is one of many, and doubling its
        // cost to re-run the trajectory the escalation governs is the worse end
        // of it.
        if matches!(
            first_status,
            ApplicationReturnStatus::MaximumIterationsExceeded
        ) && self.quality_escalations.get() >= 1
            && self
                .options
                .get_bool_value("feral_increase_quality_retry", "")
                .map(|(v, _found)| v)
                .unwrap_or(true)
            && self
                .options
                .get_bool_value("feral_increase_quality", "")
                .map(|(v, _found)| v)
                .unwrap_or(true)
        {
            return first_status;
        }
        // Flip the strategy for one retry. The parser maps "adaptive" →
        // Adaptive and every other value (incl. unset) → Monotone, so the
        // opposite of an explicit "adaptive" is "monotone" and the
        // opposite of anything else is "adaptive".
        let prev = self.options.get_string_value("mu_strategy", "").ok();
        let was_adaptive = self.effective_mu_strategy_is_adaptive();
        let flipped = if was_adaptive { "monotone" } else { "adaptive" };
        let _ = self
            .options
            .set_string_value("mu_strategy", flipped, true, false);
        // Floor the *answer*, not just the status (pounce#870).
        //
        // The promote-only-on-`Solve_Succeeded` rule below has always floored
        // the status. It did not floor the point, and the point is what the
        // caller consumes: `optimize_constrained` calls the user TNLP's
        // `finalize_solution` once per attempt, so a retry that fails to
        // promote still overwrites the answer with its own iterate, and the
        // statistics with its own residuals. The result is a status describing
        // one attempt attached to a point from another.
        //
        // Measured on a random corpus of 1200 nonconvex models, 20 of them
        // (1.7%) returned a materially worse point under an unchanged status,
        // the worst flipping sign: a `Maximum_Iterations_Exceeded` exit went
        // from -2.38e7 to +7.89e7, and a `Solved_To_Acceptable_Level` one from
        // -3.83e7 to +3.41e5 while its reported `final_kkt_error` rose to
        // 2.85e-4 — 285x the `acceptable_tol` its own status names, so the
        // report contradicted itself. The known example on record understated
        // it by three orders: `autocorr_bern55-06` swapping -2304.0000278 for
        // -2320.0000298 is the same defect at 0.07%.
        //
        // This is the floor idiom the rest of the codebase already uses for a
        // bet it might lose — `honour_neg_curv_floor` (gh#797),
        // `honour_decline_floor` (gh#534), `honour_best_acceptable_after_dual_
        // guard` — applied to the one bet that was only half-floored.
        //
        // Not extended to `run_with_l1_fallback`, which carries the same
        // caveat in its doc comment but is NOT the same call: its retry
        // deliberately reports the l1-best least-infeasible point, which the
        // option help calls informative in its own right. Changing that needs
        // its own measurement.
        let solution_floor = self.last_finalize.borrow().clone();
        let certificate_floor = SolutionCertificate::of(&self.statistics.borrow());
        let trace_floor = *self.last_iter_stats.borrow();
        let retry_status = self.optimize_constrained(Rc::clone(&tnlp));
        // Restore the user's original option-table view.
        let _ = self.options.set_string_value(
            "mu_strategy",
            prev.as_ref()
                .filter(|(_, found)| *found)
                .map(|(v, _)| v.as_str())
                .unwrap_or(if was_adaptive { "adaptive" } else { "monotone" }),
            true,
            false,
        );
        if matches!(retry_status, ApplicationReturnStatus::SolveSucceeded) {
            return retry_status;
        }
        // The bet lost. Put the first attempt's answer back, so the point and
        // the statistics describe the same solve the returned status does.
        //
        // `finalize_solution` therefore runs once more than the number of
        // attempts on this path. That is deliberate, and is the cheaper of the
        // two corrections: withholding the retry's `finalize_solution` until it
        // is known to promote would deprive a caller that watches the callback
        // of the retry's progress, and buys nothing, since the retry's payload
        // is discarded either way.
        if let Some(floor) = solution_floor {
            tracing::debug!(target: "pounce::algorithm",
                "[POUNCE] the mu_strategy_fallback retry did not promote \
                 ({:?} is not Solve_Succeeded); restoring the first attempt's \
                 solution and statistics alongside its status (pounce#870).",
                retry_status);
            floor.replay(&tnlp);
            self.answer_restored_from_floor.set(true);
            certificate_floor.restore_into(&mut self.statistics.borrow_mut());
            // Third sink: the per-iteration trace. Consumers accumulate that
            // themselves from `intermediate_callback` — the CasADi plugin
            // pushes into its own vectors and clears once per `nlpsol` call —
            // so POUNCE cannot rewind it, and both attempts concatenate into
            // one trace. Restoring the certificate without touching the trace
            // leaves the reported numbers describing attempt 1 while the trace
            // ends on the retry, which breaks the invariant
            // `casadi/test_parity.py` states outright: "The final numbers and
            // the end of the trace are the same quantities, and must not come
            // from two different places."
            //
            // Re-emitting the winning attempt's final row restores it. It is
            // the trace analogue of `FinalizeSnapshot::replay` above, and it
            // makes the property hold by construction rather than by hoping a
            // consumer resets on an attempt boundary — nothing in the callback
            // contract marks one, and gh#634 is what happens when a consumer
            // has to guess the scope of a trace.
            //
            // The row is a real iterate that was already sent once, not a
            // synthesized one, so a trace still contains only points the solver
            // actually visited.
            if let Some(stats) = trace_floor {
                let _ = tnlp.borrow_mut().intermediate_callback(
                    stats,
                    &TnlpIpoptData::default(),
                    &TnlpIpoptCq::default(),
                );
            }
        }
        first_status
    }

    /// Is the gh#884 biactive dual-divergence retry enabled? Default
    /// `yes`; `dual_divergence_retry=no` is the kill switch, and
    /// restores the pre-gh#884 behaviour outright (no detector cost, no
    /// second solve, and the base attempt's verdict returned unchanged).
    fn is_dual_divergence_retry_enabled(&self) -> bool {
        self.options
            .get_bool_value("dual_divergence_retry", "")
            .map(|(v, _found)| v)
            .unwrap_or(true)
    }

    /// gh#884: throw the iterate away and solve again from scratch with
    /// `perturb_always_cd` on, when the base solve settled its primal
    /// while its multipliers ran away.
    ///
    /// # The defect
    ///
    /// On an MPCC lowered through an exact complementarity product
    /// `G·H = 0`, a pair that is **biactive** at the solution — both
    /// `G` and `H` zero — leaves that row's gradient
    /// `H∇G + G∇H` identically zero. The row is still there, so its
    /// multiplier is *arbitrary* rather than nonexistent, and the IPM
    /// drives it to infinity while the primal iterate sits on the
    /// answer. Because the convergence verdict is reached on an
    /// `s_d`-normalised aggregate and `s_d` grows with the mean
    /// multiplier magnitude, the aggregate reads clean: MacMPEC's
    /// `qpec_small` under `ncp_eq`/`prod_eq` reported
    /// `Solved_To_Acceptable_Level` at an *unscaled* dual infeasibility
    /// of `7.9e+04`.
    ///
    /// # Why a retry rather than a gate
    ///
    /// Four other shapes of fix were measured and rejected — a Hessian
    /// sparsity hypothesis, engaging `delta_c` *in flight*, flipping
    /// `perturb_always_cd` on globally, and putting a dual ceiling on
    /// the acceptable-level gate. `dev-notes/mpcc-biactive-dual-
    /// divergence.md` records all four with numbers. The short version:
    /// by the time the runaway is visible this iterate is unrecoverable,
    /// so the only action left is to *stop using it*; and the remedy
    /// that works — `perturb_always_cd=yes` — is measured to return a
    /// wrong answer reported as success on `ralph1`
    /// (`Solve_Succeeded` at `f = -2.71e-5`, below `f* = 0`), so it
    /// cannot be turned on for everyone.
    ///
    /// **The detector is therefore the entire safety barrier**, and the
    /// promotion gate below is the second one. `ralph1` is exactly the
    /// model the detector must not fire on, and
    /// `IpoptAlgorithm`'s scale-relative step floor is what keeps it
    /// from doing so: `qpec_small` settles to `4.3e-8` while `ralph1`
    /// bottoms out at `7.2e-3`, five orders apart.
    ///
    /// # The gate
    ///
    /// The retry's verdict replaces the base one only when **all** of:
    ///
    /// 1. the base attempt saw the signature (a converged primal, a step
    ///    at zero, and an unscaled `‖∇L‖∞` far above `dual_inf_tol`, all
    ///    at one iterate — see `IpoptAlgorithm::dual_divergence_signature`);
    /// 2. the base status is `Solved_To_Acceptable_Level` or
    ///    `Restoration_Failed` — the two verdicts the vanishing-gradient
    ///    row produces directly. Generic exhaustion exits are excluded;
    ///    `deb7` under L-BFGS is why, and the reason is in the code
    ///    below;
    /// 3. the retry returns `Solve_Succeeded`;
    /// 4. the retry's claimed success is **real in the model's own
    ///    units** — unscaled KKT error at or below `tol`-scale, which is
    ///    the property the base attempt failed and the whole point of
    ///    the issue;
    /// 5. the retry's unscaled KKT error is *strictly better* than the
    ///    base attempt's.
    ///
    /// Otherwise the base attempt's status, point and statistics are all
    /// put back, by the same three-sink floor
    /// [`Self::run_with_mu_strategy_fallback`] uses and for the same
    /// reason (pounce#870): a status describing one attempt attached to
    /// a point from another is worse than either.
    fn run_with_dual_divergence_retry(
        &mut self,
        tnlp: Rc<RefCell<dyn TNLP>>,
    ) -> ApplicationReturnStatus {
        let first_status = self.dispatch_standard_solve(Rc::clone(&tnlp));
        if !self.dual_divergence_signature.get() {
            return first_status;
        }
        // Which base verdicts this remedy is *for*.
        //
        // Two, and the narrowing was bought with a measurement. The
        // detector is a statement about the iterate; it is not a
        // statement that `perturb_always_cd` will help. `deb7` under
        // L-BFGS is the corpus case that separates the two: the
        // signature is real there — iteration 346, scale-relative step
        // `6.5e-6`, `inf_pr` `3.0e-12`, unscaled `inf_du` `9.2e5`,
        // which is an order *above* the gh#884 reproducer's `7.9e4`, so
        // no dual floor excludes it, and the step conjunct separates the
        // two only by fitting the default onto one fixture and spending
        // the margin that holds `ralph1` out — and the retry still does
        // not work: 3000 iterations to
        // `Maximum_Iterations_Exceeded` at an unscaled KKT error of
        // `6.7e1` against the base attempt's `9.9e1`. Pure cost, 4x the
        // base trajectory, on a fixture whose verdict does not move.
        //
        // The exclusion is by *status*, so it is only as complete as the
        // status is stable, and on this very fixture it is not. `deb7`
        // reaches `Error_In_Step_Computation` at default options and is
        // out; under `limited_memory_ls_failure_restarts=1` (gh#818's
        // rung, off by default) it reaches `Restoration_Failed` instead
        // and is therefore *in*. It used to pay exactly the cost above
        // there — 6.1 s to 25.2 s wall clock for the same
        // `Restoration_Failed` verdict and a declined retry, which is
        // gh#887. That is the price of scoping by status rather than by
        // model, and it is why the second gate below scopes by the
        // *answer* instead: `deb7` now declines before spending
        // anything. The status scope is kept because it is cheap and
        // reads on the verdict a caller sees, but it is not what is
        // being relied on.
        //
        // What separates them is the *status*, and it separates them
        // for a reason rather than by luck:
        //
        //  * `Solved_To_Acceptable_Level` is gh#884 verbatim — the
        //    runaway laundered itself through `s_d` into a success.
        //  * `Restoration_Failed` is the same defect one step earlier:
        //    the rows whose gradients vanished drive the solve into
        //    restoration, and restoration cannot repair a row that has
        //    no gradient. It is where the `qpec_small` TNLP fixture
        //    lands (unscaled KKT `3.3e11`).
        //  * `Error_In_Step_Computation` and
        //    `Maximum_Iterations_Exceeded` are generic exhaustion
        //    exits. Every hard model reaches them, for every reason.
        //    Retrying *those* is not "repair the runaway", it is "try
        //    again harder" — which is what `mu_strategy_fallback` and
        //    the second-opinion ladder already are, and `deb7` is what
        //    that costs.
        //
        // `Solve_Succeeded` is excluded because it is already the best
        // verdict available and its certificate has already been
        // checked in the model's own units; there is nothing to buy.
        //
        // Worst-case cost is therefore one extra solve under the
        // caller's own `max_iter` — the same contract
        // `run_with_mu_strategy_fallback` already has.
        //
        // Deliberately **not** deferred to `TERMINATION_POLICY_OPTIONS`
        // the way the μ flip's acceptable-level trigger is (gh#757).
        // That deferral exists because the μ flip returns a different
        // *local solution*, so laundering a caller's deliberate
        // downgrade loses a signal they asked for. This retry cannot:
        // conjunct 4 requires the promoted answer to satisfy the KKT
        // conditions in the model's own units, which a downgrade the
        // caller induced on purpose does not. And the gh#884 reproducer
        // sets `tol=1e-8` explicitly, so deferring would decline the
        // retry on the one case the issue is about.
        let retry_worthy = matches!(
            first_status,
            ApplicationReturnStatus::SolvedToAcceptableLevel
                | ApplicationReturnStatus::RestorationFailed
        );
        if !retry_worthy {
            return first_status;
        }
        let base_unscaled_kkt = self.statistics.borrow().final_unscaled_kkt_error;
        // The runaway has to be the *whole* residual of the answer being
        // reported.
        //
        // The detector is a statement about an *iterate*, and the iterate
        // it fires on need not be the one the solve ends at. A run can
        // pass through a settled point with a diverged multiplier, work
        // its way back down, and report something ordinary — and then
        // there is nothing left here for `perturb_always_cd` to repair,
        // whatever the trajectory did in the middle. This is what makes
        // "one extra solve" a cost the caller only pays on a run that
        // still *exhibits* the defect (gh#887).
        //
        // The test is `runaway_is_the_whole_residual`, which reads only
        // the reported answer and only as a ratio within it; its doc
        // comment carries the rule and the measured populations.
        let (base_viol, base_compl) = {
            let st = self.statistics.borrow();
            (st.final_unscaled_constr_viol, st.final_unscaled_compl)
        };
        let base_dual_inf = self.statistics.borrow().final_unscaled_dual_inf;
        if !runaway_is_the_whole_residual(
            base_dual_inf,
            base_viol,
            base_compl,
            self.options
                .get_numeric_value("dual_divergence_retry_du_floor", "")
                .map(|(v, _)| v)
                .unwrap_or(DUAL_DIV_RETRY_DU_FLOOR),
        ) {
            tracing::debug!(target: "pounce::algorithm",
                "[POUNCE] gh#884: the signature fired mid-trajectory, but the \
                 answer being reported is not a converged point with a runaway \
                 multiplier — unscaled dual {:.3e} against viol {:.3e} and \
                 complementarity {:.3e}. Nothing here for perturb_always_cd to \
                 repair, so no retry (gh#887).",
                base_dual_inf, base_viol, base_compl);
            return first_status;
        }
        // Floor all three sinks — solution payload, certificate, and the
        // last trace row — exactly as the μ fallback does (pounce#870).
        let solution_floor = self.last_finalize.borrow().clone();
        let certificate_floor = SolutionCertificate::of(&self.statistics.borrow());
        let trace_floor = *self.last_iter_stats.borrow();
        tracing::debug!(target: "pounce::algorithm",
            "[POUNCE] gh#884: the primal settled while the multipliers ran away \
             (base {:?}, unscaled KKT {:.3e}); re-solving from scratch with \
             perturb_always_cd=yes.",
            first_status, base_unscaled_kkt);
        let prev = self.options.get_string_value("perturb_always_cd", "").ok();
        let _ = self
            .options
            .set_string_value("perturb_always_cd", "yes", true, false);
        let retry_status = self.dispatch_standard_solve(Rc::clone(&tnlp));
        // Restore the caller's option-table view. Absence is restored as
        // absence would be seen: the registered default is `no`.
        let _ = self.options.set_string_value(
            "perturb_always_cd",
            prev.as_ref()
                .filter(|(_, found)| *found)
                .map(|(v, _)| v.as_str())
                .unwrap_or("no"),
            true,
            false,
        );
        let retry_unscaled_kkt = self.statistics.borrow().final_unscaled_kkt_error;
        let retry_viol = self.statistics.borrow().final_unscaled_constr_viol;
        // Conjuncts 3, 4 and 5. Conjunct 4 is the one that distinguishes
        // this gate from every other promote-on-`Solve_Succeeded` retry
        // in this file: the base attempt's defect *was* a status that its
        // own unscaled residual contradicts, so promoting on the status
        // alone would reproduce the bug one attempt later.
        let claimed_success_is_real = retry_unscaled_kkt <= self.dual_divergence_retry_accept_tol()
            && retry_viol <= self.dual_divergence_retry_accept_tol();
        // Conjuncts 6 and 7 — see `retry_answer_is_admissible`. Everything
        // above this line ranks the two attempts on their *certificates*;
        // this ranks them as *answers*, which is what a caller receives.
        // Without it a better multiplier is allowed to buy a worse point:
        // measured, `-13.0057 -> -1.2072` on a random QPEC, and
        // `+1.82e-09 -> -6.61e-05` on `scholtes4`, whose `f*` is exactly 0.
        let retry_obj = self.statistics.borrow().final_objective;
        // `obj_scaling_factor < 0` is how a maximization is posed, and
        // `final_objective` is the user's signed objective, so the
        // comparison direction has to follow it (R2).
        let sense = if self
            .options
            .get_numeric_value("obj_scaling_factor", "")
            .map(|(v, _)| v)
            .unwrap_or(1.0)
            < 0.0
        {
            -1.0
        } else {
            1.0
        };
        let answer_is_admissible = retry_answer_is_admissible(
            certificate_floor.objective,
            certificate_floor.unscaled_constr_viol,
            retry_obj,
            retry_viol,
            self.dual_divergence_retry_accept_tol(),
            sense,
        );
        let promote = matches!(retry_status, ApplicationReturnStatus::SolveSucceeded)
            && claimed_success_is_real
            && retry_unscaled_kkt < base_unscaled_kkt
            && answer_is_admissible;
        // Say on the console which of the two answers shipped.
        //
        // The per-attempt end summary cannot: `emit_end_summary` runs
        // inside `optimize_constrained`, once per attempt, and the
        // promotion is decided after the last of them. So a summary that
        // reported the promotion would report `false` on the very run that
        // promotes, contradicting the JSON report written from the same
        // statistics. The summary reports the *signature*, which is true
        // when it prints; this line reports the *outcome*, printed once,
        // where it is known.
        //
        // Gated on `print_level >= 1`, matching `emit_end_summary` — the
        // block this line follows.
        let console_output = match self.options.get_integer_value("print_level", "") {
            Ok((v, true)) => v >= 1,
            _ => true,
        };
        if console_output {
            println!();
            if promote {
                println!(
                    "gh#884 dual-divergence retry: promoted — unscaled KKT error \
                     {base_unscaled_kkt:.4e} -> {retry_unscaled_kkt:.4e}."
                );
            } else if !answer_is_admissible {
                // A distinct line, because this decline looks like a
                // contradiction otherwise: the retry converged, its
                // certificate is clean, and it was still refused. Say
                // which of the two rules refused it and on what numbers,
                // so the reader is not left comparing KKT errors that
                // had nothing to do with it.
                println!(
                    "gh#884 dual-divergence retry: declined on the ANSWER, not the \
                     certificate — the retry converged (unscaled KKT error \
                     {retry_unscaled_kkt:.4e} against the base attempt's \
                     {base_unscaled_kkt:.4e}) but its objective {retry_obj:.8e} at \
                     constraint violation {retry_viol:.4e} is not admissible next to \
                     the base attempt's {:.8e} at {:.4e}; the base attempt's answer \
                     is the one reported.",
                    certificate_floor.objective, certificate_floor.unscaled_constr_viol
                );
            } else {
                println!(
                    "gh#884 dual-divergence retry: declined ({retry_status:?}, \
                     unscaled KKT error {retry_unscaled_kkt:.4e} against the base \
                     attempt's {base_unscaled_kkt:.4e}); the base attempt's answer \
                     is the one reported."
                );
            }
        }
        if promote {
            self.dual_divergence_retry_promoted.set(true);
            self.statistics.borrow_mut().dual_divergence_retry_promoted = true;
            tracing::debug!(target: "pounce::algorithm",
                "[POUNCE] gh#884: the retry promoted — unscaled KKT {:.3e} \
                 (base {:.3e}).",
                retry_unscaled_kkt, base_unscaled_kkt);
            return retry_status;
        }
        if let Some(floor) = solution_floor {
            tracing::debug!(target: "pounce::algorithm",
                "[POUNCE] gh#884: the retry did not promote ({:?}, unscaled KKT \
                 {:.3e} vs base {:.3e}); restoring the first attempt's solution \
                 and statistics alongside its status.",
                retry_status, retry_unscaled_kkt, base_unscaled_kkt);
            floor.replay(&tnlp);
            self.answer_restored_from_floor.set(true);
            certificate_floor.restore_into(&mut self.statistics.borrow_mut());
            // The signature belongs to the attempt whose numbers are now
            // reported, and `certificate_floor` does not carry it.
            self.statistics.borrow_mut().dual_divergence_signature = true;
            if let Some(stats) = trace_floor {
                let _ = tnlp.borrow_mut().intermediate_callback(
                    stats,
                    &TnlpIpoptData::default(),
                    &TnlpIpoptCq::default(),
                );
            }
        }
        first_status
    }

    /// The tolerance conjunct 4 of the dual-divergence promotion gate
    /// tests the retry's *unscaled* residuals against.
    ///
    /// `acceptable_tol`-scale rather than `tol`-scale on purpose: the
    /// unscaled residual is the one quantity nothing in the solve is
    /// driven against, so holding it to `tol` would decline honest
    /// retries on badly scaled models for a reason that has nothing to
    /// do with gh#884. `qpec_small`'s honest retry reaches `9.96e-8`,
    /// four orders inside it.
    fn dual_divergence_retry_accept_tol(&self) -> Number {
        self.options
            .get_numeric_value("acceptable_tol", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(1e-6)
    }

    /// Phase-3 ℓ₁-exact penalty-barrier outer loop.
    ///
    /// Builds an [`L1PenaltyBarrierTnlp`] wrapper around the user
    /// TNLP, runs the constrained IPM at the current ρ, escalates ρ
    /// per Byrd-Nocedal-Waltz steering, and terminates on any of:
    ///   - slack sum collapses (`Σ(p+n) ≤ l1_slack_tol`)
    ///   - inner solve returns non-Optimal (escalation won't fix
    ///     numerical / restoration failure at this ρ)
    ///   - ρ already at `l1_penalty_max`
    ///   - `l1_penalty_max_outer_iter` reached
    ///
    /// After the loop, if the inner status is `SolveSucceeded` or
    /// `SolvedToAcceptableLevel` but slacks didn't collapse, override
    /// to `Infeasible_Problem_Detected` — the returned point is the
    /// ℓ₁-best least-infeasible iterate, which is informative even
    /// though the original constraints are not satisfied.
    ///
    /// Returns `Some(status)` if the wrapper ran the solve, `None` if
    /// wrapper construction failed (caller should fall through to the
    /// standard dispatch path).
    fn run_l1_penalty_outer_loop(
        &mut self,
        tnlp: Rc<RefCell<dyn TNLP>>,
    ) -> Option<ApplicationReturnStatus> {
        let rho_init = self.l1_penalty_init();
        let rho_max = self.l1_penalty_max().max(rho_init);
        let factor = self.l1_penalty_increase_factor().max(1.0);
        let tau = self.l1_steering_factor();
        let slack_tol = self.l1_slack_tol();
        let max_outer = self.l1_penalty_max_outer_iter().max(1);

        let mut wrapper = pounce_l1penalty::L1PenaltyBarrierTnlp::new(Rc::clone(&tnlp), rho_init)?;
        if wrapper.m_eq() == 0 {
            // Nothing to slack — let the standard dispatch path handle
            // this TNLP unmodified.
            return None;
        }
        wrapper.set_defer_inner_finalize(true);
        let wrapper_rc = Rc::new(RefCell::new(wrapper));

        let mut rho = rho_init;
        let mut last_status = ApplicationReturnStatus::InternalError;
        for _outer in 0..max_outer {
            wrapper_rc.borrow_mut().set_rho(rho);
            let dyn_tnlp: Rc<RefCell<dyn TNLP>> = wrapper_rc.clone();
            last_status = self.optimize_constrained(dyn_tnlp);

            let w = wrapper_rc.borrow();
            if !w.has_solution() {
                // Inner solve aborted before producing an iterate.
                drop(w);
                break;
            }
            let slack_sum = w.last_slack_sum();
            let y_eq_inf = w.last_y_eq_inf_norm();
            let x_here: Vec<Number> = w.last_x_trunc().to_vec();
            drop(w);

            // Termination decisions.
            let inner_ok = matches!(
                last_status,
                ApplicationReturnStatus::SolveSucceeded
                    | ApplicationReturnStatus::SolvedToAcceptableLevel
            );
            if !inner_ok {
                break;
            }
            // Stop escalating ρ once the **user's** constraints are
            // satisfied to the tolerance the caller asked for — not once
            // `Σ(p + n)` falls under `l1_slack_tol`, which is a different
            // quantity judged by a different number (gh#794 P1). The
            // slack sum stays the BNW steering signal below, which is
            // the job it is right for. One extra `eval_g` per outer
            // iteration buys the difference between stopping at the
            // penalty solution and stopping at the model's own.
            let feasible_here = {
                let m_inner = tnlp
                    .borrow_mut()
                    .get_nlp_info()
                    .map(|i| i.m.max(0) as usize)
                    .unwrap_or(0);
                let mut g_here = vec![0.0; m_inner];
                let evaluated =
                    m_inner == 0 || tnlp.borrow_mut().eval_g(&x_here, true, &mut g_here);
                evaluated
                    .then(|| {
                        original_space_feasibility(
                            &tnlp,
                            &x_here,
                            &g_here,
                            self.nlp_lower_bound_inf(),
                            self.nlp_upper_bound_inf(),
                            self.user_tol(),
                            self.user_acceptable_tol(),
                            self.user_constr_viol_tol(),
                            self.user_acceptable_constr_viol_tol(),
                            self.user_primal_noise_floor_kappa(),
                        )
                    })
                    .flatten()
                    .map(|f| f.negligible_at_tol)
            };
            match feasible_here {
                Some(true) => break,
                Some(false) => {}
                // Unmeasurable model: fall back to the historical
                // slack-sum test rather than looping to the cap.
                None if slack_sum.is_finite() && slack_sum <= slack_tol => break,
                None => {}
            }
            if rho >= rho_max {
                break;
            }
            // BNW steering: ρ_new = max(ρ·factor, τ·‖y_eq‖∞ + ε)
            let geom = rho * factor;
            let steer = tau * y_eq_inf + 1.0e-12;
            rho = geom.max(steer).min(rho_max);
        }

        // Forward to the user's inner.finalize_solution exactly once.
        let w = wrapper_rc.borrow();
        if w.has_solution() {
            let x_trunc: Vec<Number> = w.last_x_trunc().to_vec();
            let lambda: Vec<Number> = w.last_lambda().to_vec();
            let z_l: Vec<Number> = w.last_z_l_trunc().to_vec();
            let z_u: Vec<Number> = w.last_z_u_trunc().to_vec();
            let solver_status = w.last_status().unwrap_or(SolverReturn::InternalError);
            let slack_sum = w.last_slack_sum();
            drop(w);

            // Recompute f(x*) and c(x*) on the inner. Both are needed
            // before the status is decided, because the status now turns
            // on the *original-space* feasibility at this point.
            let f_inner = tnlp
                .borrow_mut()
                .eval_f(&x_trunc, true)
                .unwrap_or(Number::NAN);
            let m = tnlp
                .borrow_mut()
                .get_nlp_info()
                .map(|i| i.m as usize)
                .unwrap_or(0);
            // The success flag decides whether `g_inner` is a measurement
            // or a zero-filled buffer. Dropping it would let a TNLP whose
            // final `eval_g` fails fabricate feasibility: every row would
            // read `0`, `original_space_feasibility` would return a
            // violation of zero, and that would flow into both the exit
            // status and the reported residuals. Gate it exactly as the
            // ρ-escalation measurement above does, so an evaluation
            // failure produces `None` and follows the documented
            // `l1_slack_tol` fallback instead (gh#794 review).
            let mut g_inner = vec![0.0; m];
            let g_evaluated = m == 0 || tnlp.borrow_mut().eval_g(&x_trunc, false, &mut g_inner);

            // gh#794 P1. Everything below used to argue from `Σ(p + n)`,
            // the sum of the augmented slacks, judged against
            // `l1_slack_tol`. That is not the user's constraint
            // violation and it is not judged by the user's tolerance:
            //
            //   * the violation of equality row `i` is `|p_i − n_i|`,
            //     not `p_i + n_i`, and at the barrier's interior both
            //     slacks stay positive where their difference is zero,
            //     so the sum is an upper bound that is loose in one
            //     direction; and
            //   * `l1_slack_tol` defaults to `1e-6`, four orders looser
            //     than a `tol = 1e-8` solve asked for, so a violation
            //     that the solver's own strict gate would refuse on the
            //     unwrapped problem read as "the constraints are
            //     satisfied".
            //
            // Measured, not argued: the MPCC benchmark's `ralph1`
            // (`benchmarks/mpcc/`) returned `Solve_Succeeded` at a point
            // violating its one equality row by `2.5e-07`, with the
            // reported `final_constr_viol` — the *augmented* residual —
            // at `9.6e-15`, so no field in the result disclosed it. The
            // objective came back `5.0e-04` below the true optimum,
            // which is reachable only off the feasible set.
            //
            // So: measure the user's own rows at the returned point, and
            // judge them by the tolerances the caller set. The slack sum
            // keeps its other job unchanged — it is the BNW steering
            // signal for ρ escalation inside the loop above, which is
            // what it is the right quantity for.
            let feas = g_evaluated
                .then(|| {
                    original_space_feasibility(
                        &tnlp,
                        &x_trunc,
                        &g_inner,
                        self.nlp_lower_bound_inf(),
                        self.nlp_upper_bound_inf(),
                        self.user_tol(),
                        self.user_acceptable_tol(),
                        self.user_constr_viol_tol(),
                        self.user_acceptable_constr_viol_tol(),
                        self.user_primal_noise_floor_kappa(),
                    )
                })
                .flatten();

            // The reported constraint violation must be the user's, not
            // the augmented problem's. Without this the KKT block of a
            // successful ℓ₁ solve describes a problem the caller never
            // posed. The aggregate errors take a `max` rather than a
            // rewrite: the NLP error's primal term enters undivided, so
            // the aggregate is never below the constraint violation, and
            // the other two components are unaffected by the wrapper.
            //
            // `f.max_violation` is measured on the inner TNLP's own rows
            // and bounds, so it is in the model's **original units**. That
            // decides which field family may carry it. `SolveStatistics`
            // documents `final_*` as the max-norms in the internally
            // scaled NLP space and `final_unscaled_*` as the same
            // residuals with the scaling divided back out — equal only
            // when no scaling is active — and `docs/src/python.md` states
            // the same contract to Python callers. Writing an
            // original-units number into `final_constr_viol` would break
            // it on any run with `nlp_scaling_method` engaged (gh#794
            // review).
            //
            // So the unscaled family always takes the measurement, and
            // the scaled family mirrors it exactly when per-row scaling
            // did not engage — the case in which the contract requires
            // the two to agree anyway. Under active row scaling the
            // scaled fields keep what the inner solve reported; the
            // converted number is not available here, because the
            // augmented problem's row-scale factors belong to an NLP that
            // `optimize_constrained` has already dropped. The status
            // decision below does not read these fields — it reads
            // `feas` directly — so an active-scaling run is judged on the
            // user's rows either way.
            if let Some(f) = feas.as_ref() {
                let scaled_may_mirror = self.row_scaling_active.get() == Some(false);
                let mut stats = self.statistics.borrow_mut();
                stats.final_unscaled_constr_viol = f.max_violation;
                stats.final_unscaled_kkt_error =
                    stats.final_unscaled_kkt_error.max(f.max_violation);
                if scaled_may_mirror {
                    stats.final_constr_viol = f.max_violation;
                    stats.final_kkt_error = stats.final_kkt_error.max(f.max_violation);
                    stats.final_kkt_error_above_noise =
                        stats.final_kkt_error_above_noise.max(f.max_violation);
                }
            }

            let inner_claimed_success = matches!(
                last_status,
                ApplicationReturnStatus::SolveSucceeded
                    | ApplicationReturnStatus::SolvedToAcceptableLevel
            );

            // Downgrade, not upgrade: a solve that reached the strict
            // standard on the user's own rows keeps whatever status the
            // inner gave it, and nothing here can turn a failure into a
            // success.
            let downgrade_to_acceptable = inner_claimed_success
                && feas
                    .as_ref()
                    .is_some_and(|f| !f.negligible_at_tol && f.negligible_at_acceptable);

            let infeasible_certificate = inner_claimed_success
                && match feas.as_ref() {
                    // Measured: the point does not satisfy the user's
                    // constraints even to `acceptable_tol`.
                    Some(f) => !f.negligible_at_acceptable,
                    // Unmeasurable model — fall back to the historical
                    // slack-sum argument rather than to silence.
                    None => slack_sum.is_finite() && slack_sum > slack_tol,
                };

            if let Some(f) = feas.as_ref()
                && inner_claimed_success
                && !f.negligible_at_tol
            {
                tracing::info!(
                    target: "pounce::application",
                    "l1 penalty-barrier: the inner solve converged the augmented NLP, \
                     but the returned point violates the model's own constraints by \
                     {:.3e}, which does not meet tol; reporting {} rather than success \
                     (gh#794)",
                    f.max_violation,
                    if f.negligible_at_acceptable {
                        "Solved_To_Acceptable_Level"
                    } else {
                        "an infeasibility verdict"
                    },
                );
            }
            // …unless the model's own starting point satisfies every
            // constraint, which disproves the certificate outright (gh #379).
            // Same gate as the IPM and SQP paths; see
            // `withdraw_infeasibility_if_refuted`.
            let refuted = infeasible_certificate
                && withdraw_infeasibility_if_refuted(
                    &tnlp,
                    SolverReturn::LocalInfeasibility,
                    self.nlp_lower_bound_inf(),
                    self.nlp_upper_bound_inf(),
                    self.user_tol(),
                ) != SolverReturn::LocalInfeasibility;
            let final_solver_status = match (infeasible_certificate, refuted) {
                (true, false) => SolverReturn::LocalInfeasibility,
                // The point is not feasible, so `Solve_Succeeded` would be
                // just as wrong as `Infeasible_Problem_Detected`. Report the
                // breakdown.
                (true, true) => SolverReturn::ErrorInStepComputation,
                (false, _) if downgrade_to_acceptable => SolverReturn::StopAtAcceptablePoint,
                (false, _) => solver_status,
            };
            let final_app_status = match (infeasible_certificate, refuted) {
                (true, false) => ApplicationReturnStatus::InfeasibleProblemDetected,
                (true, true) => ApplicationReturnStatus::ErrorInStepComputation,
                (false, _) if downgrade_to_acceptable => {
                    ApplicationReturnStatus::SolvedToAcceptableLevel
                }
                (false, _) => last_status,
            };

            tnlp.borrow_mut().finalize_solution(
                Solution {
                    status: final_solver_status,
                    x: &x_trunc,
                    z_l: &z_l,
                    z_u: &z_u,
                    g: &g_inner,
                    lambda: &lambda,
                    obj_value: f_inner,
                },
                &TnlpIpoptData::default(),
                &TnlpIpoptCq::default(),
            );
            return Some(final_app_status);
        }
        // No solution captured at all — pass the inner status through.
        Some(last_status)
    }

    /// Constrained-NLP path: build adapter → OrigIpoptNlp → algorithm
    /// bundle, run `optimize`, populate statistics, and call
    /// `finalize_solution` on the user's TNLP.
    /// Whether an over-determined model (more equality rows than free
    /// variables) is *provably* infeasible by linear bound propagation.
    ///
    /// Consulted only on the `NotEnoughDegreesOfFreedom` failure path, where
    /// the solve never runs and therefore can never itself discover the
    /// infeasibility (gh#387). Builds a throwaway presolve wrapper with only
    /// Phase 1 (bound tightening) enabled and asks it for a certified proof —
    /// this inherits the certification safety net wholesale: the crossing must
    /// exceed the solver's own acceptance margin at the crossed pair's scale,
    /// and a concrete witness point satisfying every constraint withdraws the
    /// verdict. A `false` here costs nothing but keeping the DOF error.
    ///
    /// The witness gate runs under [`pounce_presolve::WitnessRule`]'s
    /// `DeclaredRowRelative` form, which is admissible only because the solve
    /// cannot run on this path (gh#391) — see the comment on the probe below.
    ///
    /// Deliberately independent of the `presolve` master switch: this is not a
    /// model transformation (the wrapper is dropped without solving through
    /// it), it is a last check before reporting a structural error for a
    /// problem whose verdict is already decided.
    fn overdetermined_model_certified_infeasible(&self, tnlp: &Rc<RefCell<dyn TNLP>>) -> bool {
        let mut opts = pounce_presolve::PresolveOptions::from_options_list(&self.options)
            .unwrap_or_else(|_| pounce_presolve::PresolveOptions::defaults());
        opts.enabled = true;
        opts.bound_tightening = true;
        // Certification needs Phase 1 only; every transformative or
        // diagnostic phase is dead weight on a wrapper that is never
        // solved through.
        opts.auxiliary = false;
        opts.fbbt = false;
        opts.redundant_constraint_removal = false;
        opts.licq_check = false;
        opts.warm_z_bounds = false;
        // The one place the witness rule is raised off the solver's own
        // acceptance test (gh#391). It is sound *here specifically* because the
        // gate has already established the solve cannot run: the alternative to
        // the proof is the structural 5xx error, never `Solve_Succeeded`, so
        // the #380 "two routes, two answers" contradiction the clamp exists to
        // prevent has no second route to contradict. See `WitnessRule` for the
        // full argument and the homogeneous-row fallback.
        let mut probe =
            pounce_presolve::PresolveTnlp::new(Rc::clone(tnlp), opts).probing_without_a_solve();
        if probe.get_nlp_info().is_none() {
            return false;
        }
        probe.certified_infeasible().is_some()
    }

    fn optimize_constrained(&mut self, tnlp: Rc<RefCell<dyn TNLP>>) -> ApplicationReturnStatus {
        let t_start = Instant::now();

        // Invalidate the row-scaling record before anything can read it.
        //
        // It is written near the end of this function, from the NLP the
        // solve actually built, so a solve that bails before that point
        // leaves whatever the *previous* one recorded. The ℓ₁ outer loop
        // calls this repeatedly and reads the flag after each call, so a
        // stale `Some(false)` would let it mirror an original-units
        // violation into the scaled family — the exact contract this flag
        // exists to protect (gh#794 review round 2). Clearing it here
        // makes the failure mode fail-closed: "not recorded" reads as
        // "cannot mirror", never as "scaling was off".
        self.row_scaling_active.set(None);

        // `print_user_options yes` — dump the OptionsList before the
        // solve. Mirrors `IpoptApplication::call_optimize` (upstream
        // calls `Jnlst().Printf(.., "%s", options_->PrintUserOptions())`).
        let print_opts = self
            .options
            .get_bool_value("print_user_options", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(false);
        if print_opts {
            print!(
                "\nList of user-set options:\n\n{}",
                self.options.print_user_options()
            );
        }

        // `print_options_documentation yes` — dump the full registry
        // (every option with type, default, valid range/strings, and
        // long description) before the solve. Honors
        // `print_options_mode` (`text` / `latex` / `doxygen`; only
        // `text` is implemented today, the others fall through with a
        // one-line note) and `print_advanced_options`. Mirrors
        // upstream `IpoptApplication::call_optimize`'s
        // `print_options_documentation` branch and `Common/IpRegOptions.cpp`
        // `OutputOptionDocumentation`.
        let print_doc = self
            .options
            .get_bool_value("print_options_documentation", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(false);
        if print_doc {
            let mode = self
                .options
                .get_string_value("print_options_mode", "")
                .ok()
                .map(|(v, _)| PrintOptionsMode::from_tag(&v))
                .unwrap_or(PrintOptionsMode::Text);
            let advanced = self
                .options
                .get_bool_value("print_advanced_options", "")
                .ok()
                .map(|(v, _)| v)
                .unwrap_or(false);
            print!(
                "\n# Pounce options registry\n\n{}",
                self.reg_options.print_options_documentation(mode, advanced)
            );
        }

        // Mint a fresh `TimingStatistics` for this solve — shared (via
        // `Rc`) with the data and the NLP below so every `eval_*` and
        // every iterate-phase records into the same accumulator. The
        // application keeps its own `Rc` so callers can read totals out
        // via [`Self::timing_stats`].
        let timing = Rc::new(TimingStatistics::new());
        *self.timing.borrow_mut() = Rc::clone(&timing);
        // gh#606: same lifetime as the timings — a solve that bails out
        // before the initializer runs must not report the previous
        // solve's warm-start verdict.
        *self.warm_start_diag.borrow_mut() = None;
        // Gate the *detailed* per-subsystem timers on `timing_statistics`
        // (default "no"), matching upstream Ipopt. Without this, every
        // timed `eval_*` / phase section pays two `getrusage` syscalls per
        // start/end even when statistics are off — 16-20% of busy CPU on
        // fast-objective NLPs (issue #190). `print_timing_statistics=yes`
        // implies `timing_statistics=yes` (per its option help), so either
        // one enables the detailed timers. `overall_alg` is started
        // unconditionally below: it feeds the `max_cpu_time` check and is
        // reported regardless of the option.
        //
        // Each name is read as a literal rather than looped over an
        // array: the registered-but-unread scan
        // (`tests/no_silent_options.rs`) keys on the option name as it
        // appears at the accessor, so a loop variable reads as "no key
        // here" and hid `timing_statistics` among the silent options
        // when it has been wired since #190 (#677, #551).
        let read_yes = |key: &str| -> bool {
            self.options
                .get_bool_value(key, "")
                .ok()
                .and_then(|(v, f)| f.then_some(v))
                .unwrap_or(false)
        };
        let timing_enabled = read_yes("timing_statistics") || read_yes("print_timing_statistics");
        timing.set_detailed_enabled(timing_enabled);
        timing.overall_alg.start();

        // Reset the linear-solver summary sink so back-to-back solves
        // don't bleed factor counters / extremal pivots into each
        // other. Surviving the lock failure with a debug-assert keeps
        // a poisoned mutex from sinking a release build that doesn't
        // even consume the summary.
        match self.linsol_summary_sink.lock() {
            Ok(mut guard) => {
                *guard = LinearSolverSummary::default();
            }
            _ => {
                debug_assert!(false, "linsol summary sink mutex poisoned");
            }
        }
        // Same reasoning for the quality-escalation tally (gh#857): the
        // number belongs to this solve, not to whatever ran before it.
        self.quality_escalations.set(0);

        // Build adapter + Nlp. Honor `fixed_variable_treatment` (default
        // `make_parameter`; pounce additionally implements `relax_bounds`,
        // which the adapter also auto-selects as a fallback when
        // `make_parameter` would leave `n_x_var < n_c` — mirrors upstream
        // `IpTNLPAdapter.cpp:623-633`).
        let lo_inf = self
            .options
            .get_numeric_value("nlp_lower_bound_inf", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(DEFAULT_NLP_LOWER_BOUND_INF);
        let up_inf = self
            .options
            .get_numeric_value("nlp_upper_bound_inf", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(DEFAULT_NLP_UPPER_BOUND_INF);
        let fixed_treatment = match self
            .options
            .get_string_value("fixed_variable_treatment", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .as_deref()
        {
            Some("relax_bounds") => FixedVarTreatment::RelaxBounds,
            // `make_constraint` / `make_parameter_nodual` not yet
            // implemented; fall back to `make_parameter` (auto-retry to
            // `relax_bounds` will still kick in if DOF runs short).
            _ => FixedVarTreatment::MakeParameter,
        };
        let adapter = match TNLPAdapter::new_with_options(
            Rc::clone(&tnlp),
            lo_inf,
            up_inf,
            fixed_treatment,
        ) {
            Ok(a) => Rc::new(RefCell::new(a)),
            Err(_) => {
                timing.overall_alg.end();
                return ApplicationReturnStatus::InvalidProblemDefinition;
            }
        };
        // Carry the user's constant `obj_scaling_factor` (default 1.0;
        // negative ⇒ maximize) into the NLP. Until pounce#128's
        // follow-up this option was registered but never read, so it
        // was silently a no-op — maximization diverged because the
        // algorithm minimized the unscaled objective.
        let obj_scaling_factor = self
            .options
            .get_numeric_value("obj_scaling_factor", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(1.0);
        let mut orig_nlp = match OrigIpoptNlp::new(
            Rc::clone(&adapter),
            Rc::new(ConstObjScaling(obj_scaling_factor)),
        ) {
            Ok(n) => n,
            Err(_) => {
                timing.overall_alg.end();
                return ApplicationReturnStatus::InternalError;
            }
        };
        orig_nlp.set_timing_stats(Rc::clone(&timing));
        // Q6: decide which derivatives may be reused across iterates,
        // before anything is evaluated (gh #588).
        self.install_constant_derivative_hints(&mut orig_nlp);

        // Mirror upstream `OrigIpoptNLP::InitializeStructures` (IpOrigIpoptNLP.cpp:299):
        // bail out with NotEnoughDegreesOfFreedom when there are fewer free
        // variables than equality constraints. Without this gate, square /
        // over-determined systems push the algorithm into restoration on
        // iter 0 and exit Restoration_Failed instead of the cleaner DOF code.
        let n_x_var = orig_nlp.x_space().dim();
        let n_c = orig_nlp.c_space().dim();
        if n_x_var > 0 && n_x_var < n_c {
            timing.overall_alg.end();
            // An over-determined system can still be *provably* infeasible —
            // `x == 0.2` with `x == 0.8` is about as provable as infeasibility
            // gets — and for such a model the structural DOF error is the
            // strictly weaker answer: it reports "cannot attempt this" for a
            // problem whose verdict is already decided (gh#387). The DOF gate
            // fires before any iteration runs, so nothing downstream will ever
            // get the chance to detect the infeasibility; check for a
            // bound-propagation proof here, on the rare failure path only.
            // The probe reuses presolve's full certification pipeline
            // (crossing margin + witness refutation), so a model the solver
            // would accept as feasible at its own tolerance is never upgraded
            // to "proved infeasible" — those still report the DOF error.
            if self.overdetermined_model_certified_infeasible(&tnlp) {
                use pounce_common::journalist::JournalCategory;
                self.journalist.print(
                    JournalLevel::J_SUMMARY,
                    JournalCategory::J_MAIN,
                    "\nEXIT: Problem has too few degrees of freedom, and bound \
                     propagation proves its constraints inconsistent.\n\
                     No feasible point exists; the solve was not run.\n",
                );
                return ApplicationReturnStatus::InfeasibleProblemDetected;
            }
            return ApplicationReturnStatus::NotEnoughDegreesOfFreedom;
        }

        // Relax `x_L / x_U / d_L / d_U` by `bound_relax_factor` (default
        // 1e-8), capped by `constr_viol_tol` (default 1e-4). Matches
        // `OrigIpoptNLP::InitializeStructures` lines 343-358.
        let bound_relax_factor = self
            .options
            .get_numeric_value("bound_relax_factor", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(1e-8);
        let constr_viol_tol = self
            .options
            .get_numeric_value("constr_viol_tol", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(1e-4);
        orig_nlp.relax_bounds(bound_relax_factor, constr_viol_tol);

        // `honor_original_bounds` (default `no`, matching upstream):
        // project the reported point back into the un-relaxed box. Must
        // follow `relax_bounds`, which snapshots the bounds to project
        // onto. Registered but never read before, so a user asking for
        // it still got a bound-pinned solution sitting up to
        // `min(bound_relax_factor·max(1,|b|), constr_viol_tol)` outside
        // its own bounds (gh#483 follow-up).
        let honor_original_bounds = self
            .options
            .get_bool_value("honor_original_bounds", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(false);
        orig_nlp.set_honor_original_bounds(honor_original_bounds);

        // Apply automatic NLP scaling per `nlp_scaling_method` option
        // (port of `OrigIpoptNLP::InitializeStructures` →
        // `NLPScalingObject::DetermineScaling`). Default is
        // `gradient-based` to match upstream Ipopt 3.14.
        let scaling_method = self
            .options
            .get_string_value("nlp_scaling_method", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or_else(|| "gradient-based".to_string());
        let scaling_method = match scaling_method.as_str() {
            "none" => ScalingMethod::None,
            "gradient-based" => ScalingMethod::GradientBased,
            // `curvature-based` computes the factors from the model's
            // quadratic coefficients and hands them back through
            // `TNLP::get_scaling_parameters` (gh #703), so from the
            // engine's side it *is* user scaling — the only difference is
            // who filled the vectors in.
            "user-scaling" | "curvature-based" => ScalingMethod::UserScaling,
            // `equilibration-based` is registered upstream but not yet
            // implemented in pounce; fall back to gradient-based (the
            // upstream default) to keep behavior predictable.
            _ => ScalingMethod::GradientBased,
        };
        let max_gradient = self
            .options
            .get_numeric_value("nlp_scaling_max_gradient", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(100.0);
        let min_value = self
            .options
            .get_numeric_value("nlp_scaling_min_value", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(1e-8);
        let obj_target_gradient = self
            .options
            .get_numeric_value("nlp_scaling_obj_target_gradient", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(0.0);
        let constr_target_gradient = self
            .options
            .get_numeric_value("nlp_scaling_constr_target_gradient", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(0.0);
        orig_nlp.determine_scaling_from_starting_point(
            scaling_method,
            max_gradient,
            min_value,
            obj_target_gradient,
            constr_target_gradient,
        );

        let nlp_handle: Rc<RefCell<dyn IpoptNlp>> = Rc::new(RefCell::new(orig_nlp));

        // Build the algorithm strategy bundle. Read coarse knobs from
        // the OptionsList where we have them; fall through to defaults
        // otherwise. The full upstream parsing surface (mu_strategy,
        // hessian_approximation, line_search_method, ...) is wired by
        // `AlgBuilder::RegisterOptions` in upstream — that registry
        // hookup lands as a follow-up; default builder is correct for
        // HS71-class problems.
        let mut builder = self.algorithm_builder_from_options();

        // The objective element's support for the partitioned Hessian.
        // Every constraint element reads its support off a Jacobian row;
        // only the objective has none declared, so without this the
        // updater falls back to the first `∇f`'s nonzeros — a
        // value-derived pattern. See
        // `TNLPAdapter::objective_nonlinear_vars`.
        if matches!(
            builder.hessian_approximation,
            HessianApproxChoice::Partitioned | HessianApproxChoice::FiniteDifference
        ) {
            builder.objective_nonlinear_vars = adapter.borrow().objective_nonlinear_vars();
        }

        // Which variables the limited-memory Hessian should span (gh#624).
        // Upstream's precedence: a TNLP that implements
        // `get_number_of_nonlinear_variables` wins, and
        // `num_linear_variables` is only the contiguous-prefix fallback.
        // Exact-Hessian solves never consult either.
        if matches!(
            builder.hessian_approximation,
            HessianApproxChoice::LimitedMemory | HessianApproxChoice::FiniteDifference
        ) {
            let num_linear_variables = self
                .options
                .get_integer_value("num_linear_variables", "")
                .ok()
                .and_then(|(v, f)| f.then_some(v))
                .unwrap_or(0);
            match adapter
                .borrow()
                .quasi_newton_nonlinear_vars(num_linear_variables)
            {
                Ok(mask) => builder.limited_memory_nonlinear_vars = mask,
                Err(e) => {
                    use pounce_common::journalist::JournalCategory;
                    self.journalist.print(
                        JournalLevel::J_ERROR,
                        JournalCategory::J_MAIN,
                        &format!("\nEXIT: Invalid nonlinear-variable list: {}\n", e.message),
                    );
                    timing.overall_alg.end();
                    return ApplicationReturnStatus::InvalidProblemDefinition;
                }
            }
        }

        // Linear-solver backend. The default factory is option-aware
        // — it reads the `feral_*` extension options off the same
        // `OptionsList` that drove the IPM-level builder above so
        // per-problem `.opt` files can flip backend knobs without
        // rebuilding pounce.
        let mut feral_cfg = feral_config_from_options(&self.options);
        // Block-triangular / Schur KKT partition (pounce#180 item 2). Configure
        // the Schur block solvers from the *base* feral cfg: a full-KKT external
        // ordering (item 1) is sized for the whole system and cannot apply to
        // the A_FF sub-block, so the Schur path keeps the default sub-block
        // ordering. `build_with_backend` honors this only on the IPM + feral +
        // exact-Hessian path and falls back to the standard solver otherwise.
        if let Some(indices) = &self.kkt_schur_block {
            builder.set_kkt_schur(indices.clone(), feral_cfg.clone());
        }
        // A caller-supplied KKT permutation (pounce#180 item 1) overrides
        // the string-option / env ordering: `OrderingMethod::External`
        // can't be expressed through the OptionsList (it carries a
        // vector), so it is injected here from the side-channel field.
        // Only applies to the workspace-default FERAL backend below; a
        // custom `linear_backend_factory` owns its own config.
        if let Some(perm) = &self.external_ordering {
            feral_cfg.ordering = pounce_feral::OrderingMethod::External(perm.clone());
        }
        // MA57's knobs come off the same `OptionsList`, at the main-IPM
        // prefix. The restoration sub-IPM reads them again under
        // `"resto."` when its caller mints the inner backend factory —
        // see `ma57_config_from_options`.
        let ma57_cfg = ma57_config_from_options(&self.options, "");
        let factory = self.linear_backend_factory.take().unwrap_or_else(|| {
            default_backend_factory_with_sink(
                feral_cfg,
                ma57_cfg,
                Arc::clone(&self.linsol_summary_sink),
            )
        });
        let bundle = builder.build_with_backend(factory);

        // Wire the data / cq pair around the NLP. Install the shared
        // `TimingStatistics` so the algorithm's iterate phases
        // (output, convergence, hessian, μ, search-direction,
        // line-search, accept) all record into the same accumulator
        // the application exposes via `timing_stats()`.
        let data: crate::ipopt_data::IpoptDataHandle = Rc::new(RefCell::new(AlgIpoptData::new()));
        data.borrow_mut().timing = Rc::clone(&timing);
        // Install a shared wall/CPU-time deadline (pounce#242) so the time
        // budget is honored at the granularity of the expensive inner
        // steps — the main loop's KKT factorization / line search and the
        // restoration inner IPM — instead of only between outer iterations.
        // The `Deadline` starts its clock now (right after `overall_alg`),
        // and the restoration sub-solve reuses this same instance, so the
        // caller's budget bounds the whole solve rather than each nested
        // level independently. The convergence check treats it as
        // authoritative when present (see `conv_check::opt_error`).
        data.borrow_mut().deadline = Some(pounce_common::timing::Deadline::new(
            builder.conv_check.max_wall_time,
            builder.conv_check.max_cpu_time,
        ));
        let cq: crate::ipopt_cq::IpoptCqHandle = Rc::new(RefCell::new(
            IpoptCalculatedQuantities::new(Rc::clone(&data), Rc::clone(&nlp_handle)),
        ));
        // Correction size for very small slacks (default mach_eps^{3/4});
        // drives the safe-slack bound-adjustment mechanism.
        if let Ok((v, true)) = self.options.get_numeric_value("slack_move", "") {
            cq.borrow_mut().slack_move = v;
        }
        // `kappa_d` — weight of the linear damping term added to the
        // barrier objective/gradient to handle one-sided bounds
        // (`IpIpoptCalculatedQuantities.cpp`). Registered (default 1e-5)
        // but previously never read, so a user override was silently
        // ignored (#191). Routed through the builder for parity with the
        // other numeric knobs; the default matches the registered
        // default, so only explicit overrides change behavior.
        cq.borrow_mut().kappa_d = builder.kappa_d;
        // `s_max` — cap on the average multiplier magnitude in the
        // `(s_d, s_c)` scaling of the KKT error test. Same shape as
        // `kappa_d`: registered (default 100) and previously never read,
        // so an override was silently ignored (#551 / #677).
        cq.borrow_mut().s_max = builder.s_max;

        // Seed `data.curr` with a zero-valued iterate of the correct
        // dimensions. The `IterateInitializer` consumes these as its
        // template (it overwrites `x`, `s`, multipliers in place); we
        // just need the dim metadata.
        {
            let nlp_borrow = nlp_handle.borrow();
            let n_x = nlp_borrow.n();
            let n_s = nlp_borrow.m_ineq();
            let n_yc = nlp_borrow.m_eq();
            let n_yd = nlp_borrow.m_ineq();
            let n_zl = nlp_borrow.x_l().dim();
            let n_zu = nlp_borrow.x_u().dim();
            let n_vl = nlp_borrow.d_l().dim();
            let n_vu = nlp_borrow.d_u().dim();
            drop(nlp_borrow);
            let iv = IteratesVector::new(
                Rc::new(DenseVectorSpace::new(n_x).make_new_dense()),
                Rc::new(DenseVectorSpace::new(n_s).make_new_dense()),
                Rc::new(DenseVectorSpace::new(n_yc).make_new_dense()),
                Rc::new(DenseVectorSpace::new(n_yd).make_new_dense()),
                Rc::new(DenseVectorSpace::new(n_zl).make_new_dense()),
                Rc::new(DenseVectorSpace::new(n_zu).make_new_dense()),
                Rc::new(DenseVectorSpace::new(n_vl).make_new_dense()),
                Rc::new(DenseVectorSpace::new(n_vu).make_new_dense()),
            );
            data.borrow_mut().set_curr(iv);
        }

        // Full primal-dual warm restart (debugger `resolve`): if a
        // captured iterate is queued, install it onto `data.curr` over
        // the placeholder so the `WarmStartIterateInitializer`'s
        // re-optimize branch (x already initialized) keeps it and only
        // clamps multipliers / sets target_mu — no cold re-seed from the
        // NLP. Skipped (with a warning) if the dimensions don't line up,
        // e.g. an option changed the problem structure between solves.
        if let Some(snap) = self.warm_start_iterate.take() {
            let dims_match = {
                let borrow = data.borrow();
                borrow
                    .curr
                    .as_ref()
                    .map(|c| iterates_dims(c) == iterates_dims(snap.iterates()))
                    .unwrap_or(false)
            };
            if dims_match {
                data.borrow_mut().set_curr(snap.iterates().clone());
                data.borrow_mut().curr_mu = snap.mu();
            } else {
                tracing::warn!(
                    target: "pounce::warm_start",
                    "debugger warm-restart iterate dimensions differ from the fresh \
                     solve; ignoring the captured iterate and seeding normally"
                );
            }
        }

        let max_iter = self
            .options
            .get_integer_value("max_iter", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(3000);
        let tol = self
            .options
            .get_numeric_value("tol", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(1e-8);
        data.borrow_mut().tol = tol;

        let mut alg = IpoptAlgorithm::new(data, cq, bundle)
            .with_nlp(Rc::clone(&nlp_handle))
            .with_tnlp(Rc::clone(&tnlp));
        alg.last_iter_stats_sink = Some(Rc::clone(&self.last_iter_stats));
        // Mint a fresh restoration factory per inner solve if a
        // provider is configured (pounce#10 Phase 3). Falls back to
        // the legacy one-shot `restoration_factory` slot when no
        // provider is set, preserving single-shot caller behavior.
        if let Some(provider) = self.restoration_factory_provider.as_mut() {
            self.restoration_factory = Some(provider());
        }
        if let Some(factory) = self.restoration_factory.as_mut() {
            alg = alg.with_restoration(factory());
        }
        if let Some(diag) = self.diagnostics.as_ref() {
            alg = alg.with_diagnostics(Rc::clone(diag));
        }
        // Move the interactive debugger hook (if any) into the main
        // algorithm. Taken — not cloned — so it drives exactly this
        // solve; a subsequent solve must reinstall it.
        if let Some(hook) = self.debug_hook.take() {
            alg = alg.with_debug_hook(hook);
        }
        alg.max_iter = max_iter;
        // `kappa_sigma` — factor bounding how far the bound multipliers
        // may deviate from their primal estimates; the clamp runs after
        // every accepted step (`IpIpoptAlg.cpp`, Eqn. (16)). Registered
        // (default 1e10) but previously never read, so a user override —
        // including the documented `< 1` "disable the correction" — was
        // silently ignored (#191). Routed through the builder; the struct
        // default matches the registered default, so default runs are
        // unchanged.
        alg.kappa_sigma = builder.kappa_sigma;
        // `recalc_y` (#677) — see the read site above for why the
        // limited-memory path defaults it on.
        // `linear_system_scaling=slack-based` (#677) — the only scaling
        // choice whose factors depend on the iterate, so the main loop
        // has to refresh them. See `IpoptAlgorithm::push_slack_scaling`.
        alg.slack_based_scaling = matches!(
            builder.linear_system_scaling,
            crate::alg_builder::LinearSystemScalingChoice::SlackBased
        );
        alg.recalc_y = builder.recalc_y;
        alg.recalc_y_feas_tol = builder.recalc_y_feas_tol;
        // `start_with_resto` — the outer loop is what acts on it. It was
        // previously copied only into the restoration sub-solver's own
        // builder, which has no first outer iteration to force, so
        // setting the option did nothing at all.
        alg.start_with_resto = builder.resto.start_with_resto;
        // Tiny-step and divergence guards (#191): registered but
        // previously never read. Struct defaults match the registered
        // defaults, so default runs are unchanged.
        alg.tiny_step_tol = builder.tiny_step_tol;
        alg.tiny_step_y_tol = builder.tiny_step_y_tol;
        alg.diverging_iterates_tol = builder.diverging_iterates_tol;
        alg.dual_diverging_streak = builder.dual_diverging_streak.max(0) as usize;
        alg.dual_divergence_retry_step_tol = builder.dual_divergence_retry_step_tol;
        alg.dual_divergence_retry_du_floor = builder.dual_divergence_retry_du_floor;
        alg.resto_decline_deferrals = builder.resto_decline_deferrals.max(0) as usize;
        alg.resto_decline_progress_ratio = builder.resto_decline_progress_ratio;
        alg.neg_curv_escapes = builder.neg_curv_escapes.max(0) as usize;
        alg.lbfgs_ls_failure_restarts = builder.limited_memory_ls_failure_restarts.max(0) as usize;
        alg.kkt_fidelity_tol = builder.kkt_fidelity_tol;
        // Honor `print_level == 0`: silence the algorithm's direct-to-stdout
        // output — the per-iteration table and, new in #206, the
        // problem-statistics and end-of-run summary blocks the engine now
        // emits itself. Default (unset) or any positive level shows them; the
        // CLI's JSON mode forces print_level 0, so structured output stays
        // clean. (The Phase-7 journalist surface respects print_level already;
        // this is the legacy direct-print site that needs the same gate.)
        let console_output = match self.options.get_integer_value("print_level", "") {
            Ok((v, true)) => v >= 1,
            _ => true,
        };
        if !console_output {
            alg.print_iter_output = false;
            // The nested restoration IPM is built inside the restoration
            // driver, not by `IpoptAlgorithm::new`, so it never sees this
            // gate unless we forward it.
            if let Some(resto) = alg.restoration.as_mut() {
                resto.set_print_iter_output(false);
            }
        }

        // Problem statistics, Ipopt-style, emitted before the iteration table
        // from the engine's own reduced problem (#206). Built from the same
        // collect_stats inputs the CLI used, so the block is byte-identical;
        // emitting it here means every frontend (CLI, Python, C) and every
        // algorithm (IPM, SQP) gets it.
        self.emit_problem_stats(&tnlp, console_output);

        // Per-iteration history (pounce#71): when requested, capture the
        // `pounce::iteration` events emitted during the solve into an
        // `IterRecord` trajectory via the observability collector layer.
        // This replaces the old in-loop `iter_history` accumulation; it
        // requires the collector to be installed in the active
        // subscriber (the CLI / Python / C frontends install it via
        // `pounce_observability::init_subscriber`; tests call
        // `init_for_tests`). The collector scopes out restoration
        // sub-solve iterations via the `restoration` span, so the
        // trajectory matches the previous behavior (outer iters only).
        let iter_capture = self
            .record_iter_history
            .then(pounce_observability::IterCaptureGuard::start);

        let solver_status = alg.optimize();
        // Keep the initializer's feasibility diagnostics reachable
        // after the algorithm goes out of scope (gh#605).
        self.least_square_init_report = alg.least_square_init_report();

        let captured_iters = iter_capture.map(|g| g.finish()).unwrap_or_default();
        // Propagate to any enclosing capture (e.g. `with_iter_capture`
        // wrapped around a solve with iteration history enabled), whose
        // buffer this inner guard would otherwise leave empty.
        pounce_observability::extend_active_capture(&captured_iters);
        // Close the overall-algorithm timer on the success path. The
        // early-return arms above end it themselves before bailing out;
        // this one matches upstream `IpoptApplication::call_optimize`
        // (which calls `EndCpuTime()` on overall_alg right after
        // `Optimize` returns, regardless of solver_status).
        timing.overall_alg.end();

        // gh#612: opt-in crossover. Runs here — after the algorithm is done
        // but BEFORE the statistics drain, the status gates, the
        // `on_converged` hook and `finalize_via_orig_nlp` — so that when it
        // accepts, every one of those describes the point actually returned.
        // Placing it later would mean reporting residuals for an iterate the
        // user is not given, and would hide the exact active set from the
        // post-optimal sensitivity hook, which is the first of the three
        // consumers this exists for.
        self.maybe_crossover(&mut alg, &nlp_handle, solver_status);

        // Drain counters / iter count off the algorithm.
        {
            let mut stats = self.statistics.borrow_mut();
            {
                let d = alg.data.borrow();
                stats.iteration_count = d.iter_count;
                // Converged barrier parameter μ — threaded forward into a
                // warm-started corrector's `mu_init` / `warm_start_target_mu`
                // for predictor–corrector path following (pounce#86).
                stats.final_mu = d.curr_mu;
            }
            // gh#606: the warm-start initializer's verdict on what the
            // caller supplied. Lifted off the (solve-local) data handle
            // so it outlives the solve; `None` on a cold start.
            *self.warm_start_diag.borrow_mut() = alg.data.borrow().warm_start_diagnostics.clone();
            stats.total_wallclock_time_secs = t_start.elapsed().as_secs_f64();
            // Restoration-phase audit counters (pounce#12). Zero on
            // problems where restoration never fires; populated by
            // `IpoptAlgorithm::invoke_restoration`.
            // Finite-difference Hessian census (gh#823 review). `None` on
            // every other updater, which leaves the `-1` sentinel in
            // place and says "this mode did not run" rather than "it ran
            // and found nothing".
            if let Some(fd) = alg.bundle.hess.fd_hessian_stats() {
                use crate::hess::fd_hessian::FdPatternSource;
                stats.fd_hessian_pattern_used = match fd.pattern_used {
                    Some(FdPatternSource::Declared) => 0,
                    Some(FdPatternSource::Jacobian) => 1,
                    None => -1,
                };
                stats.fd_hessian_nnz = fd.nnz as Index;
                stats.fd_hessian_n = fd.n as Index;
                stats.fd_hessian_groups = fd.groups as Index;
                stats.fd_hessian_rho_max = fd.rho_max as Index;
                stats.fd_hessian_coloring_fell_back = fd.coloring_fell_back;
                stats.fd_hessian_objective_clique_widened = fd.objective_clique_widened;
            }
            stats.restoration_calls = alg.resto_calls;
            stats.restoration_inner_iters = alg.resto_inner_iters;
            stats.restoration_outer_iters = alg.resto_outer_iters;
            stats.restoration_wall_secs = alg.resto_wall_secs;
            // gh#857. Read off the shared cell rather than the algorithm's
            // own `PdFullSpaceSolver`, so restoration sub-solves — which
            // run their own solver instance — are included.
            stats.quality_escalations = self.quality_escalations.get() as Index;
            // gh#884. Read off the algorithm that just ran, so
            // `run_with_dual_divergence_retry` — which sits above this
            // call — can see what it observed.
            self.dual_divergence_signature
                .set(self.dual_divergence_signature.get() || alg.dual_divergence_signature());
            stats.dual_divergence_signature = self.dual_divergence_signature.get();
            stats.dual_divergence_retry_promoted = self.dual_divergence_retry_promoted.get();
            stats.iterations = captured_iters;
            // A refused starting point does not produce a valid iterate.
            // Leave final objective/residual fields at their NaN defaults.
            // Capture the final *scaled* objective at the algorithm's
            // (compressed `x_var`-space) iterate via the NLP: the
            // algorithm-side `eval_f` returns `f * obj_scale_factor`.
            // `final_objective` is seeded with it only as a best-effort
            // fallback; the success path below overwrites it with the
            // true unscaled objective from `finalize_via_orig_nlp`
            // (which evaluates the user TNLP directly).
            if solver_status != SolverReturn::InvalidProblemDefinition {
                let curr_x = alg.data.borrow().curr.as_ref().map(|c| c.x.clone());
                if let Some(x) = curr_x {
                    if let Ok(f) = try_eval_curr_f(&nlp_handle, &x) {
                        stats.final_objective = f;
                        stats.final_scaled_objective = f;
                    }
                }
                // Final residuals straight off the cq cache. These mirror
                // the values upstream prints in its end-of-run summary
                // ("Dual infeasibility / Constraint violation /
                // Complementarity / Overall NLP error").
                let cq = alg.cq.borrow();
                stats.final_dual_inf = cq.curr_dual_infeasibility_max();
                // Stays on the *internal* measure deliberately: the summary's
                // "Overall NLP error" is `curr_nlp_error`, and it is built from
                // this same `max(||c||, ||d - s||)`. Switching the violation line
                // alone to the original-NLP measure
                // (`curr_unscaled_nlp_constraint_violation_max`, now used by the
                // `inf_pr` column) would leave the block self-inconsistent —
                // an error larger than the max of its own components. Making
                // them agree means deciding whether *convergence* should be
                // judged on the original NLP, which is a behaviour change for
                // every model, not a reporting fix. See pounce#476.
                //
                // NOTE (gh #528): "Overall NLP error" is no longer the number
                // the strict gate tests. That gate judges
                // `curr_nlp_error_above_primal_noise` — the same aggregate with
                // each row's residual counted only above what it can represent
                // in floating point — so on a model whose constraint values run
                // to `~1e8` the summary can report an error above `tol` beside
                // `EXIT: Optimal Solution Found`. The gap is exactly the part
                // of the residual that is quantisation noise, and it is bounded
                // by `constr_viol_tol`, which is still tested here on the full
                // unfloored residual. Reporting is deliberately left on the raw
                // value: it is the honest measurement, and at these magnitudes
                // the default `bound_relax_factor = 1e-8` has already moved
                // every bound by orders of magnitude more than the floor
                // forgives, so the raw number was never an exact statement
                // about the original NLP either.
                stats.final_constr_viol = cq.curr_primal_infeasibility_max();
                // How far outside the model AS DECLARED the returned point
                // sits. The line above is the internal slack measure the
                // convergence test reads, on the `bound_relax_factor`-widened
                // model this arm genuinely solves; the widening stays here
                // because a feasible-iterate log-barrier needs `x` strictly
                // inside its bounds (the convex arm's does not -- see
                // `qp_extract::BoundRelax`). The two can differ by orders and
                // nothing used to say so: on netlib `wood1p` this reports
                // `1.71e-14` at a point `7.96e-09` outside the declared rows
                // and `9.84e-09` outside the declared box. Only reported when
                // a widening was applied; without one the two coincide and
                // `NaN` says "nothing to add".
                stats.final_declared_constr_viol = if bound_relax_factor > 0.0 {
                    cq.curr_declared_primal_violation_max()
                } else {
                    Number::NAN
                };
                // The box half of the same measurement, unconditionally: this
                // one is a *summary line* (Ipopt's `Variable bound
                // violation`), not an extra warning, so it has to carry a real
                // number on every solve rather than only on a widened one. At
                // `bound_relax_factor = 0` the honest number is `0` and the
                // accessor returns it by measurement, not by assumption.
                stats.final_declared_box_viol = cq.curr_declared_box_violation_max();
                // Infinity-norm complementarity, max over all four bound
                // blocks (s_xl·z_l, s_xu·z_u, s_sl·v_l, s_su·v_u). The
                // empty-bound blocks return `0` from amax(), so the max is
                // safe even when only one side has bounds.
                let compl = cq
                    .curr_compl_x_l()
                    .amax()
                    .max(cq.curr_compl_x_u().amax())
                    .max(cq.curr_compl_s_l().amax())
                    .max(cq.curr_compl_s_u().amax());
                stats.final_compl = compl;
                stats.final_kkt_error = cq.curr_nlp_error();
                // The aggregate the strict gate tested (gh #528). Reported
                // alongside the raw one so a summary can account for the gap
                // between them; equal to it on every `O(1)` model, and on any
                // run with `primal_noise_floor_kappa = 0`.
                stats.final_kkt_error_above_noise = cq
                    .curr_nlp_error_above_primal_noise(builder.conv_check.primal_noise_floor_kappa);
                // Unscaled (user-space) counterparts — divide the nlp_scaling
                // back out so a consumer can verify the certificate in its own
                // units (pounce#173). Identical to the scaled fields when no
                // scaling is active.
                stats.final_unscaled_dual_inf = cq.curr_unscaled_dual_infeasibility_max();
                stats.final_unscaled_constr_viol = cq.curr_unscaled_primal_infeasibility_max();
                // Record whether per-row scaling actually engaged, so a
                // wrapper that measures the user's rows in the model's own
                // units knows which field family may carry that number.
                // `curr_unscaled_primal_infeasibility_max` treats both
                // vectors absent as "scaled == unscaled"; the ℓ₁ outer loop
                // relies on the same equivalence (gh#794 review).
                {
                    let nlp_ref = nlp_handle.borrow();
                    self.row_scaling_active.set(Some(
                        nlp_ref.c_scale_vec().is_some() || nlp_ref.d_scale_vec().is_some(),
                    ));
                }
                stats.final_unscaled_compl = cq.curr_unscaled_complementarity_max();
                stats.final_unscaled_kkt_error = cq.curr_unscaled_nlp_error();

                // Report an accepted crossover in the frame it solved in
                // (#646). Everything above measures against the bounds the
                // interior iteration ran against, which `bound_relax_factor`
                // widened by `δ` before the solve. That is the right frame
                // for an interior iterate — it never touches a bound — but a
                // crossed-over point sits *exactly* on the constraints of the
                // problem as declared, i.e. `δ` inside the relaxed ones, so
                // the four `s·z` blocks above read `|multiplier| · δ`. For a
                // unit multiplier and the default `δ = 1e-8` that is `1e-8`,
                // which is `tol`: a strictly better point printed an `Overall
                // NLP error` above tolerance, and the opt-in
                // `kkt_fidelity_tol` gate below downgraded it.
                //
                // Only complementarity moves. Stationarity involves no
                // bounds, and the crossed-over point is *interior* to the
                // relaxed box, so its violation is zero under either reading.
                //
                // The substitution is confined to reporting. Crossover runs
                // after the status is decided, and it only ever installs a
                // point the never-regress gate accepted on the declared-bound
                // residuals, so this cannot dress up a worse iterate — the
                // measurement it replaces is the artifact.
                if let Some(report) = self.crossover_report.as_ref()
                    && report.accepted()
                    && report.compl_after.is_finite()
                {
                    let compl_declared = report.compl_after;
                    stats.final_compl = compl_declared;
                    stats.final_kkt_error =
                        cq.curr_nlp_error_with_complementarity(compl_declared, 0.0);
                    stats.final_kkt_error_above_noise = cq.curr_nlp_error_with_complementarity(
                        compl_declared,
                        builder.conv_check.primal_noise_floor_kappa,
                    );
                    // Same unscaling as `curr_unscaled_complementarity_max`:
                    // the slack's row factor and the multiplier's cancel in
                    // the product, leaving the objective factor. Magnitude —
                    // `obj_scaling_factor` is signed, `-1` being the
                    // documented way to pose a maximization.
                    let df = cq.obj_scaling_factor().abs();
                    stats.final_unscaled_compl = if df == 0.0 || df == 1.0 {
                        compl_declared
                    } else {
                        compl_declared / df
                    };
                    stats.final_unscaled_kkt_error = stats
                        .final_unscaled_dual_inf
                        .max(stats.final_unscaled_constr_viol)
                        .max(stats.final_unscaled_compl);
                }
            }
        }

        // Never report `Infeasible_Problem_Detected` while holding a point that
        // satisfies every constraint. The gates that produce this verdict argue
        // from a stalled feasibility sub-problem, and gh #379 is what that looks
        // like when the argument is wrong — a model whose own starting point is
        // exactly feasible, reported infeasible. See
        // `withdraw_infeasibility_if_refuted`.
        let solver_status =
            withdraw_infeasibility_if_refuted(&tnlp, solver_status, lo_inf, up_inf, tol);

        // Map SolverReturn → ApplicationReturnStatus per
        // MAIN_LOOP.md's exception table, then apply the opt-in
        // status-fidelity gate (pounce#173).
        let app_status = self.apply_kkt_fidelity_gate(solver_return_to_app_status(solver_status));

        // On convergence, fire the user-supplied callback (post-optimal
        // sensitivity hook, pounce#16) before flowing back through
        // `finalize_via_orig_nlp`. Borrowed handles into the converged
        // KKT state stay alive for the duration of the closure.
        if matches!(
            app_status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ) {
            if let Some(cb) = self.on_converged.as_mut() {
                if let Some(sd) = alg.search_dir.as_mut() {
                    let pd = sd.pd_solver_rc();
                    cb(&alg.data, &alg.cq, &nlp_handle, pd);
                }
            }
        }

        // Finalize: forward the final iterate to the user's TNLP. The
        // returned objective is evaluated on the *user* TNLP at the
        // unscaled iterate, so it overrides the scaled best-effort
        // value stashed in `final_objective` above (the algorithm-side
        // `eval_f` returns `f * obj_scale_factor`).
        if solver_status != SolverReturn::InvalidProblemDefinition {
            match finalize_via_orig_nlp(
                &nlp_handle,
                &alg,
                solver_status,
                app_status,
                &tnlp,
                &self.last_finalize,
            ) {
                Ok(f_unscaled) => {
                    self.statistics.borrow_mut().final_objective = f_unscaled;
                }
                Err(()) => {}
            }
        }

        // End-of-solve timing report. Gated on `print_timing_statistics`
        // (default "no"); mirrors upstream's
        // `IpoptApplication::call_optimize` →
        // `IpTimingStatistics::PrintAllValues` call site. The report
        // goes to stdout (for parity with the banner / iter-row output
        // path) and is also fanned out to the journalist so an
        // `output_file` attached via `Initialize` picks it up.
        let print_timing = self
            .options
            .get_bool_value("print_timing_statistics", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(false);
        if print_timing {
            let report = timing.report();
            print!("{}", report);
            use pounce_common::journalist::{JournalCategory, JournalLevel};
            self.journalist.print(
                JournalLevel::J_SUMMARY,
                JournalCategory::J_TIMING_STATISTICS,
                &report,
            );
        }

        // End-of-run summary, Ipopt-style, emitted last (after any timing
        // report) from the engine's own statistics (#206). Drains the eval
        // tallies into SolveStatistics (read AFTER finalize so the final
        // solution evaluation is included) and prints the summary, gated on
        // the same print_level as the rest of the console.
        self.emit_end_summary(app_status, &nlp_handle, console_output);

        app_status
    }

    /// Build an [`AlgorithmBuilder`] populated from the app's
    /// [`OptionsList`]. Public so callers wiring the restoration
    /// factory can hand the *inner* IPM a builder that mirrors the
    /// outer's `mu_strategy`/`mu_oracle`/line-search choices —
    /// matching upstream `IpAlgBuilder::BuildRestoIpoptAlgorithm`,
    /// which reads the same `mu_strategy` option with prefix `"resto."
    /// + prefix` and falls back to the outer setting.
    pub fn algorithm_builder_from_options(&self) -> AlgorithmBuilder {
        let mut builder = AlgorithmBuilder::new();
        // gh#857: share this application's escalation tally. Every
        // frontend builds the restoration provider's inner builder from
        // this method, so restoration escalations aggregate here too.
        builder.quality_escalation_counter = Some(Rc::clone(&self.quality_escalations));

        // `mehrotra_algorithm` is parsed first so its cascading
        // defaults (mu_strategy=adaptive, mu_oracle=probing) can be
        // overridden by an explicit user setting of those keys
        // below. Mirrors `IpAlgBuilder.cpp:Mehrotra`.
        // `fast_step_computation` — skip the search-direction residual
        // check and allow an inexact linear solve. `PdSearchDirCalc` has
        // consumed this flag since it landed, hard-coded to `false`; the
        // option's read site was simply missing, so setting it did
        // nothing at all (gh#483 follow-up, #191 round 2).
        if let Ok((v, true)) = self.options.get_string_value("fast_step_computation", "") {
            builder.fast_step_computation = v.eq_ignore_ascii_case("yes");
        }

        let mut mehrotra_on = false;
        if let Ok((v, found)) = self.options.get_string_value("mehrotra_algorithm", "") {
            if found && v == "yes" {
                mehrotra_on = true;
                builder.mehrotra_algorithm = true;
                builder.mu_strategy = MuStrategyChoice::Adaptive;
                builder.mu_oracle = crate::mu::adaptive::MuOracleKind::Probing;
                // `accept_every_trial_step` short-circuits the alpha
                // loop / filter — Mehrotra steps would otherwise be
                // rejected by the filter on LP-shaped problems because
                // the barrier objective is non-monotone along the
                // corrector. Mirrors upstream `IpAlgBuilder.cpp:Mehrotra`.
                builder.line_search.accept_every_trial_step = true;
                // Aggressive iterate-push defaults (`SetNumericValueIfUnset`
                // in upstream). The explicit user parses below will
                // overwrite these if the user set them explicitly.
                builder.init.bound_push = 10.0;
                builder.init.bound_frac = 0.2;
                builder.init.slack_bound_push = 10.0;
                builder.init.slack_bound_frac = 0.2;
                builder.init.bound_mult_init_val = 10.0;
                builder.init.constr_mult_init_max = 0.0;
                // `alpha_for_y=bound-mult` — Mehrotra wants the
                // equality multipliers to advance with the dual
                // alpha so they stay in step with z/v. Mirrors
                // upstream `IpIpoptAlg.cpp:InitializeImpl`.
                builder.line_search.alpha_for_y =
                    crate::line_search::backtracking::AlphaForY::BoundMult;
                // `adaptive_mu_globalization=never-monotone-mode` —
                // upstream `IpIpoptAlg.cpp:148-154` enforces this:
                // Mehrotra disables the globalization switch entirely
                // (no fallback to monotone mode when convergence
                // stalls). Required for the unsafeguarded Mehrotra
                // path to function.
                builder.mu.adaptive_mu_globalization =
                    crate::mu::adaptive::AdaptiveMuGlobalization::NeverMonotoneMode;
                // `least_square_init_primal=yes` — upstream
                // `IpIpoptAlg.cpp:182` enables this for the Mehrotra
                // cascade. Replaces the user's starting `x` with the
                // min-norm primal that satisfies the linearized
                // equality+inequality constraints. Critical on
                // LP-shaped problems where the user's starting point
                // can be wildly infeasible (e.g. nuffield2_trap).
                builder.init.least_square_init_primal = true;
            }
        }

        if let Ok((v, found)) = self.options.get_string_value("mu_strategy", "") {
            if found {
                let parsed = match v.as_str() {
                    "adaptive" => MuStrategyChoice::Adaptive,
                    _ => MuStrategyChoice::Monotone,
                };
                if mehrotra_on && matches!(parsed, MuStrategyChoice::Monotone) {
                    // Upstream Ipopt refuses this combination: Mehrotra
                    // needs an affine step every iter, which only the
                    // adaptive path computes. Keep adaptive and warn.
                    tracing::warn!(target: "pounce::algorithm",
                        "pounce: mehrotra_algorithm=yes requires \
                         mu_strategy=adaptive; ignoring \
                         mu_strategy=monotone."
                    );
                } else {
                    builder.mu_strategy = parsed;
                }
            }
        }
        if let Ok((v, found)) = self.options.get_string_value("mu_oracle", "") {
            if found {
                builder.mu_oracle = match v.as_str() {
                    "loqo" => crate::mu::adaptive::MuOracleKind::Loqo,
                    "probing" => crate::mu::adaptive::MuOracleKind::Probing,
                    _ => crate::mu::adaptive::MuOracleKind::QualityFunction,
                };
            }
        }
        if let Ok((v, found)) = self
            .options
            .get_string_value("adaptive_mu_globalization", "")
        {
            if found {
                use crate::mu::adaptive::AdaptiveMuGlobalization;
                builder.mu.adaptive_mu_globalization = match v.as_str() {
                    "kkt-error" => AdaptiveMuGlobalization::KktError,
                    "never-monotone-mode" => AdaptiveMuGlobalization::NeverMonotoneMode,
                    _ => AdaptiveMuGlobalization::ObjConstrFilter,
                };
            }
        }
        if let Ok((v, found)) = self.options.get_string_value("hessian_approximation", "") {
            if found {
                builder.hessian_approximation = match v.as_str() {
                    "limited-memory" => HessianApproxChoice::LimitedMemory,
                    "partitioned" => HessianApproxChoice::Partitioned,
                    "finite-difference" => HessianApproxChoice::FiniteDifference,
                    _ => HessianApproxChoice::Exact,
                };
            }
        }
        // **Upstream changes the `mu_strategy` default for a
        // limited-memory Hessian.** `IpAlgBuilder.cpp:1059`:
        //
        //     if( !options.GetStringValue("mu_strategy", smuupdate, prefix) )
        //     {
        //        // Change default for quasi-Newton option (then we use adaptive)
        //        ... if( hessian_approximation == LIMITED_MEMORY )
        //               smuupdate = "adaptive";
        //     }
        //
        // and again at `:920` for the restoration-phase algorithm.
        // Registered default is `monotone`; the quasi-Newton path takes
        // `adaptive` unless the caller says otherwise. pounce read the
        // registered default unconditionally, so every L-BFGS solve ran
        // a barrier schedule Ipopt does not use on that path — a
        // trajectory divergence on the arm the Python frontend and the
        // CasADi plugin select automatically (gh#746).
        //
        // The restoration sub-IPM inherits this: `run_inner_resto`
        // clones the configured `inner_alg_builder`, so the flag set
        // here is what the resto algorithm gets, matching `:920`.
        //
        // Only when unset — an explicit `mu_strategy` still wins, and
        // `mehrotra_algorithm` (parsed above) has already forced
        // adaptive on its own terms.
        if builder.hessian_approximation == HessianApproxChoice::LimitedMemory
            && !self.mu_strategy_was_set()
        {
            builder.mu_strategy = MuStrategyChoice::Adaptive;
        }
        // Limited-memory quasi-Newton update formula. Registered upstream
        // (`limited_memory_update_type`, IpLimMemQuasiNewtonUpdater.cpp) but
        // until now read nowhere on the IPM path — the updater was hard-wired
        // to Powell-damped BFGS. SR1 is honored too (the updater and the
        // low-rank/inertia path already handle its indefinite models).
        // Partitioned quasi-Newton knobs. `partitioned_update_type`
        // defaults to SR1 rather than BFGS — see
        // `crates::hess::partitioned_quasi_newton` for why damping is
        // the wrong choice on a per-constraint element.
        if let Ok((v, found)) = self.options.get_string_value("partitioned_update_type", "") {
            if found {
                builder.partitioned_update_type = match v.as_str() {
                    "bfgs" => UpdateType::Bfgs,
                    _ => UpdateType::Sr1,
                };
                builder.partitioned_update_type_was_set = true;
            }
        }
        if let Ok((v, found)) = self
            .options
            .get_integer_value("partitioned_max_element", "")
        {
            if found && v > 0 {
                builder.partitioned_max_element = v as usize;
            }
        }
        if let Ok((v, found)) = self.options.get_string_value("fd_hessian_pattern", "") {
            if found {
                builder.fd_hessian_pattern = match v.as_str() {
                    "jacobian" => crate::hess::fd_hessian::FdPatternSource::Jacobian,
                    _ => crate::hess::fd_hessian::FdPatternSource::Declared,
                };
            }
        }
        if let Ok((v, found)) = self.options.get_string_value("fd_hessian_coloring", "") {
            if found {
                builder.fd_hessian_coloring = match v.as_str() {
                    "cpr" => crate::hess::fd_hessian::FdColoring::Cpr,
                    _ => crate::hess::fd_hessian::FdColoring::Star,
                };
            }
        }
        if let Ok((v, found)) = self.options.get_numeric_value("fd_hessian_reuse_tol", "") {
            if found && v >= 0.0 {
                builder.fd_hessian_reuse_tol = v;
            }
        }
        if let Ok((v, found)) = self.options.get_string_value("partitioned_elements", "") {
            if found {
                builder.partitioned_elements = match v.as_str() {
                    "blocks" => crate::hess::partitioned_quasi_newton::ElementMode::PrimalBlock,
                    _ => crate::hess::partitioned_quasi_newton::ElementMode::PerConstraint,
                };
            }
        }
        if let Ok((v, found)) = self.options.get_integer_value("partitioned_block_size", "") {
            if found && v > 0 {
                builder.partitioned_block_size = v as usize;
            }
        }
        if let Ok((v, found)) = self
            .options
            .get_numeric_value("partitioned_curvature_cap", "")
        {
            if found && v > 0.0 {
                builder.partitioned_curvature_cap = v;
            }
        }
        if let Ok((v, found)) = self
            .options
            .get_string_value("limited_memory_update_type", "")
        {
            if found {
                builder.limited_memory_update_type = match v.as_str() {
                    "sr1" => UpdateType::Sr1,
                    _ => UpdateType::Bfgs,
                };
            }
        }
        // Limited-memory history length (`limited_memory_max_history`).
        if let Ok((v, found)) = self
            .options
            .get_integer_value("limited_memory_max_history", "")
        {
            if found && v >= 0 {
                builder.limited_memory_max_history = v as Index;
            }
        }
        // `limited_memory_initialization` — which formula picks the
        // initial Hessian scalar σ. Registered since the option port with
        // upstream's `scalar1` default and read nowhere until #677, so
        // the updater's own `Scalar2` default was the only value any
        // solve ever used: setting the option did nothing, and it warned
        // nothing. Same miss as gh#483 / #191 round 2 (which wired
        // `limited_memory_init_val_max`/`_min`) — this is the third
        // argument to that same `initial_hessian_scalar` call.
        //
        // The effective default now follows the registry (`scalar1`),
        // matching Ipopt. σ_scalar2/σ_scalar1 = (yᵀy·sᵀs)/(sᵀy)² ≥ 1 and
        // is unbounded as the curvature pair degrades, so on an
        // ill-conditioned problem `scalar2` inflates `B0 = σI` until it
        // swamps the rank-2 corrections and the step collapses.
        if let Ok((v, found)) = self
            .options
            .get_string_value("limited_memory_initialization", "")
        {
            if found {
                use crate::hess::lim_mem_quasi_newton::InitialApprox;
                // Every registered value is named explicitly. #551 §3
                // held this option back precisely because wiring only
                // the values that mapped would leave the rest falling
                // back silently — a new no-op created by the fix — so a
                // catch-all standing in for a real value is the one
                // shape to avoid here. `OptionsList` rejects any
                // unregistered value before this runs (`options_list.rs`
                // `OPTION_INVALID`), so the final arm is unreachable
                // rather than a fallback.
                builder.limited_memory_initialization = match v.as_str() {
                    "scalar1" => InitialApprox::Scalar1,
                    "scalar2" => InitialApprox::Scalar2,
                    "scalar3" => InitialApprox::Scalar3,
                    "scalar4" => InitialApprox::Scalar4,
                    "constant" => InitialApprox::Constant,
                    "history-max" => InitialApprox::HistoryMax,
                    _ => InitialApprox::Scalar2,
                };
            }
        }
        // `recalc_y` / `recalc_y_feas_tol` — least-square re-estimation
        // of the equality multipliers once feasible (#677).
        //
        // Upstream registers `recalc_y` as `no`, but its own option text
        // ends "If a limited memory quasi-Newton option is chosen, this
        // is used by default", so upstream's effective default is
        // conditional on the Hessian approximation.
        //
        // **pounce does not follow that, and the discrepancy is
        // deliberate.** Auto-enabling it for the limited-memory path was
        // implemented and measured against the fixture corpus first: it
        // moved 16 of 57 fixtures on the L-BFGS leg and took **7 from
        // solved to not solved, with nothing moving the other way** —
        // `airport` 56 it → `SearchDirectionBecomesTooSmall` at 541,
        // `pooling_rt2stp` 413 it → the same at 1775, all three `jit1`
        // variants, `linear_eq_collapsed_box`, and `hs13_bigstart` to
        // the iteration cap. The signature is consistent: re-estimating
        // `y` on every feasible iteration overwrites Newton multipliers
        // that were converging, the dual never settles, and the step
        // vanishes short of the certificate.
        //
        // So the feature is available and off by default. That still
        // closes the gap that mattered — until #677 the option was
        // refused outright as unimplemented, so an L-BFGS user could not
        // reach Ipopt's behaviour at all. They can now, by asking.
        //
        // Why it is worth having: a quasi-Newton model's dual step is
        // computed from an approximate `W`, so L-BFGS can settle a
        // feasible primal and still not drive `inf_du` to tolerance —
        // the failure a 59,939-variable CasADi model hit, oscillating
        // `inf_du` between 3.6e-3 and 1.8e+01 for 300 iterations with
        // the objective already settled. On that shape it is the fix;
        // on a corpus of small well-conditioned models it is a
        // pessimisation. Matching upstream's conditional default needs
        // to explain the 7 regressions first.
        if let Ok((v, true)) = self.options.get_string_value("recalc_y", "") {
            builder.recalc_y = v == "yes";
        }
        if let Ok((v, true)) = self.options.get_numeric_value("recalc_y_feas_tol", "") {
            builder.recalc_y_feas_tol = v;
        }
        // `limited_memory_init_val` — σ before any curvature pair exists,
        // and every iteration under `constant`. Also unread until #677;
        // the empty-history branch hard-coded the same `1.0`.
        if let Ok((v, true)) = self
            .options
            .get_numeric_value("limited_memory_init_val", "")
        {
            builder.limited_memory_init_val = v;
        }
        // `limited_memory_max_skipping` (#686) — registered and unread,
        // and the feature behind it did not exist either: the updater
        // counted nothing and never discarded its history.
        if let Ok((v, true)) = self
            .options
            .get_integer_value("limited_memory_max_skipping", "")
        {
            if v >= 0 {
                builder.limited_memory_max_skipping = v as Index;
            }
        }
        if let Ok((v, found)) = self.options.get_string_value("line_search_method", "") {
            if found {
                builder.line_search_method = match v.as_str() {
                    "cg-penalty" => LineSearchChoice::CgPenalty,
                    "penalty" => LineSearchChoice::Penalty,
                    _ => LineSearchChoice::Filter,
                };
            }
        }
        // `accept_every_trial_step` — direct user override. Parsed
        // after the Mehrotra cascade so an explicit `no` still wins.
        if let Ok((v, found)) = self.options.get_string_value("accept_every_trial_step", "") {
            if found {
                builder.line_search.accept_every_trial_step = v == "yes";
            }
        }
        // `alpha_for_y` — direct user override. Parsed after the
        // Mehrotra cascade so an explicit value still wins.
        if let Ok((v, found)) = self.options.get_string_value("alpha_for_y", "") {
            if found {
                use crate::line_search::backtracking::AlphaForY;
                builder.line_search.alpha_for_y = match v.as_str() {
                    "primal" => AlphaForY::Primal,
                    "bound-mult" | "bound_mult" => AlphaForY::BoundMult,
                    "full" => AlphaForY::Full,
                    "min" => AlphaForY::Min,
                    "max" => AlphaForY::Max,
                    "primal-and-full" | "dual-and-full" => AlphaForY::Primal,
                    _ => AlphaForY::Primal,
                };
            }
        }
        // `nlp_scaling_method` is consumed NLP-side in
        // `OrigIpoptNlp::determine_scaling_from_starting_point` (see the
        // `determine_scaling_from_starting_point` call earlier in this
        // method); there is no algorithm-side scaling strategy to wire.
        // `limited_memory_init_val_max` / `_min` — the clamp on the
        // initial Hessian scalar. `LimMemQuasiNewtonUpdater` consumes
        // both in `initial_hessian_scalar`; only the read sites were
        // missing, so setting either did nothing (gh#483, #191 round 2).
        if let Ok((v, true)) = self
            .options
            .get_numeric_value("limited_memory_init_val_max", "")
        {
            builder.limited_memory_init_val_max = v;
        }
        if let Ok((v, true)) = self
            .options
            .get_numeric_value("limited_memory_init_val_min", "")
        {
            builder.limited_memory_init_val_min = v;
        }

        // Unlike the other options here, we always honor the registry
        // value (not just when the user set it explicitly): the option
        // registry default is "ma57" but `AlgorithmBuilder::default`
        // has `linear_solver: Feral`, so gating on `found` would
        // silently route default runs through Feral while the banner
        // (and ipopt-compatible behavior) advertises MA57.
        //
        // Record the **effective** backend, not the requested one. MA57 lives
        // behind the optional `ma57` cargo feature (HSL is licensed and needs a
        // Fortran toolchain); without it `default_backend_factory` silently
        // substitutes FERAL. Storing `Ma57` here therefore made
        // `builder.linear_solver` disagree with the backend actually built, and
        // consumers acted on the lie: the Schur KKT gate in
        // `alg_builder::build_with_backend` tests `== Feral`, so on the
        // pure-Rust default build — where the registry default (then upstream's
        // "ma57") resolved to FERAL anyway — `set_kkt_schur_block()` silently
        // never engaged for ANY user. Resolving here keeps the field truthful
        // for every consumer.
        //
        // The `_ =>` arm is now only reachable for `feral`: every other name
        // is refused up front by `unimplemented_linear_solver`. It used to
        // swallow `mumps`, `pardiso`, `ma97`, … and run FERAL instead.
        if let Ok((v, _found)) = self.options.get_string_value("linear_solver", "") {
            let requested = if v.eq_ignore_ascii_case("ma57") {
                LinearSolverChoice::Ma57
            } else {
                LinearSolverChoice::Feral
            };
            builder.linear_solver =
                if matches!(requested, LinearSolverChoice::Ma57) && !cfg!(feature = "ma57") {
                    LinearSolverChoice::Feral
                } else {
                    requested
                };
        }

        // `linear_system_scaling` — symmetric scaling of the augmented
        // KKT matrix before factorization. Port of
        // `IpTSymLinearSolver.cpp:RegisterOptions` plumbing. Default
        // "none"; "ruiz" invokes the Ruiz-2001 symmetric ∞-norm
        // equilibration in `RuizTSymScalingMethod`. "mc19" and
        // "slack-based" are accepted by the registry but not yet
        // implemented at this layer; they fall back to no scaling
        // with a one-line notice.
        //
        // `slack-based` is implemented as of #677. It used to reach the
        // no-scaling fallback through the catch-all arm, which meant it
        // fell back **silently** — the comment above promised a notice
        // that only `mc19` actually emitted. It is not a hypothetical
        // value: it is what Ipopt's own recommended configuration for
        // large collocation NLPs uses, so the users most likely to set
        // it were the least likely to be told it did nothing. The
        // catch-all is left for genuinely unreachable input —
        // `OptionsList` rejects anything the registry does not list.
        if let Ok((v, found)) = self.options.get_string_value("linear_system_scaling", "") {
            if found {
                builder.linear_system_scaling = match v.as_str() {
                    "ruiz" => crate::alg_builder::LinearSystemScalingChoice::Ruiz,
                    "mc19" => crate::alg_builder::LinearSystemScalingChoice::Mc19,
                    "slack-based" => crate::alg_builder::LinearSystemScalingChoice::SlackBased,
                    _ => crate::alg_builder::LinearSystemScalingChoice::None,
                };
            }
        }
        if let Ok((v, found)) = self.options.get_bool_value("linear_scaling_on_demand", "") {
            if found {
                builder.linear_scaling_on_demand = v;
            }
        }

        // Convergence tolerances (port of `IpOptErrorConvCheck.cpp`'s
        // `RegisterOptions` consumers). Defaults already match upstream
        // — only override when the user set the key explicitly.
        let read_num = |key: &str| -> Option<f64> {
            self.options
                .get_numeric_value(key, "")
                .ok()
                .and_then(|(v, f)| f.then_some(v))
        };
        let read_int = |key: &str| -> Option<i32> {
            self.options
                .get_integer_value(key, "")
                .ok()
                .and_then(|(v, f)| f.then_some(v))
        };
        if let Some(v) = read_num("tol") {
            builder.conv_check.tol = v;
        }
        if let Some(v) = read_num("obj_scale_certificate_threshold") {
            builder.conv_check.obj_scale_certificate_threshold = v;
        }
        if let Some(v) = read_num("primal_noise_floor_kappa") {
            builder.conv_check.primal_noise_floor_kappa = v;
        }
        if let Some(v) = read_num("acceptable_progress_kappa") {
            builder.conv_check.acceptable_progress_kappa = v;
        }
        if let Some(v) = read_num("dual_inf_scale_kappa") {
            builder.conv_check.dual_inf_scale_kappa = v;
        }
        if let Some(v) = read_num("kkt_fidelity_tol") {
            builder.kkt_fidelity_tol = v;
        }
        if let Some(v) = read_num("dual_inf_tol") {
            builder.conv_check.dual_inf_tol = v;
        }
        if let Some(v) = read_num("constr_viol_tol") {
            builder.conv_check.constr_viol_tol = v;
        }
        if let Some(v) = read_num("compl_inf_tol") {
            builder.conv_check.compl_inf_tol = v;
        }
        if let Some(v) = read_int("max_iter") {
            builder.conv_check.max_iter = v;
        }
        if let Some(v) = read_num("max_cpu_time") {
            builder.conv_check.max_cpu_time = v;
        }
        if let Some(v) = read_num("max_wall_time") {
            builder.conv_check.max_wall_time = v;
        }
        if let Some(v) = read_num("acceptable_tol") {
            builder.conv_check.acceptable_tol = v;
        }
        if let Some(v) = read_num("acceptable_dual_inf_tol") {
            builder.conv_check.acceptable_dual_inf_tol = v;
        }
        if let Some(v) = read_num("acceptable_constr_viol_tol") {
            builder.conv_check.acceptable_constr_viol_tol = v;
        }
        if let Some(v) = read_num("acceptable_compl_inf_tol") {
            builder.conv_check.acceptable_compl_inf_tol = v;
        }
        if let Some(v) = read_num("acceptable_obj_change_tol") {
            builder.conv_check.acceptable_obj_change_tol = v;
        }
        if let Some(v) = read_int("acceptable_iter") {
            builder.conv_check.acceptable_iter = v;
        }
        if let Some(v) = read_num("infeas_stationarity_tol") {
            builder.conv_check.infeas_stationarity_tol = v;
        }
        if let Some(v) = read_num("infeas_viol_kappa") {
            builder.conv_check.infeas_viol_kappa = v;
        }
        if let Some(v) = read_int("infeas_max_streak") {
            builder.conv_check.infeas_max_streak = v;
        }

        // Bound-multiplier / barrier damping constants (#191). Both were
        // registered but never read, so user overrides were silently
        // dropped; the algorithm ran with the hard-coded struct defaults.
        // Defaults equal the registered defaults, so this changes nothing
        // for a run that doesn't set them.
        if let Some(v) = read_num("kappa_sigma") {
            builder.kappa_sigma = v;
        }
        if let Some(v) = read_num("kappa_d") {
            builder.kappa_d = v;
        }
        // `s_max` — the cap in the `(s_d, s_c)` scaling of the KKT error
        // test (#551 / #677). `IpoptCalculatedQuantities` carried it as a
        // hard-coded 100 (the registered default) and nothing read the
        // option; a run that does not set it is unaffected.
        if let Some(v) = read_num("s_max") {
            builder.s_max = v;
        }
        if let Some(v) = read_num("tiny_step_tol") {
            builder.tiny_step_tol = v;
        }
        if let Some(v) = read_num("tiny_step_y_tol") {
            builder.tiny_step_y_tol = v;
        }
        if let Some(v) = read_num("diverging_iterates_tol") {
            builder.diverging_iterates_tol = v;
        }
        if let Some(v) = read_int("dual_diverging_streak") {
            builder.dual_diverging_streak = v;
        }
        if let Some(v) = read_num("dual_divergence_retry_step_tol") {
            builder.dual_divergence_retry_step_tol = v;
        }
        if let Some(v) = read_num("dual_divergence_retry_du_floor") {
            builder.dual_divergence_retry_du_floor = v;
        }
        if let Some(v) = read_int("resto_decline_deferrals") {
            builder.resto_decline_deferrals = v;
        }
        if let Some(v) = read_num("resto_decline_progress_ratio") {
            builder.resto_decline_progress_ratio = v;
        }
        if let Some(v) = read_int("neg_curv_escapes") {
            builder.neg_curv_escapes = v;
        }
        if let Some(v) = read_int("limited_memory_ls_failure_restarts") {
            builder.limited_memory_ls_failure_restarts = v;
        }

        // Barrier-parameter (μ) options — consumers in
        // `IpMonotoneMuUpdate.cpp` / `IpAdaptiveMuUpdate.cpp`. Both
        // updaters share the same option names; the builder forwards
        // each into whichever strategy is assembled.
        if let Some(v) = read_num("mu_init") {
            builder.mu.mu_init = v;
        }
        if let Some(v) = read_num("mu_max") {
            builder.mu.mu_max = v;
        }
        if let Some(v) = read_num("mu_max_fact") {
            builder.mu.mu_max_fact = v;
        }
        if let Some(v) = read_num("mu_min") {
            builder.mu.mu_min = v;
        }
        if let Some(v) = read_num("mu_target") {
            builder.mu.mu_target = v;
        }
        if let Some(v) = read_num("mu_linear_decrease_factor") {
            builder.mu.mu_linear_decrease_factor = v;
        }
        if let Some(v) = read_num("mu_superlinear_decrease_power") {
            builder.mu.mu_superlinear_decrease_power = v;
        }
        if let Ok((v, found)) = self
            .options
            .get_string_value("mu_allow_fast_monotone_decrease", "")
        {
            if found {
                builder.mu.mu_allow_fast_monotone_decrease = v == "yes";
            }
        }
        if let Some(v) = read_num("barrier_tol_factor") {
            builder.mu.barrier_tol_factor = v;
        }
        // `tau_min` — floor on the fraction-to-the-boundary parameter
        // (#551 / #677). Both `MonotoneMuUpdate` and `AdaptiveMuUpdate`
        // carried the field with upstream's 0.99 default and nothing
        // read the option, so an override was silently dropped. The
        // default equals the registered default, so this changes
        // nothing for a run that does not set it.
        if let Some(v) = read_num("tau_min") {
            builder.mu.tau_min = v;
        }
        if let Some(v) = read_num("sigma_max") {
            builder.mu.sigma_max = v;
        }
        if let Some(v) = read_num("sigma_min") {
            builder.mu.sigma_min = v;
        }

        // Quality-function oracle knobs — consumers in
        // `IpQualityFunctionMuOracle.cpp:RegisterOptions`. Forwarded
        // to the oracle on every free-mode call.
        if let Ok((v, found)) = self
            .options
            .get_string_value("quality_function_norm_type", "")
        {
            if found {
                use crate::mu::oracle::quality_function::NormType;
                builder.mu.quality_function_norm_type = match v.as_str() {
                    "1-norm" => NormType::OneNorm,
                    "2-norm" => NormType::TwoNorm,
                    "max-norm" => NormType::MaxNorm,
                    _ => NormType::TwoNormSquared,
                };
            }
        }
        if let Ok((v, found)) = self
            .options
            .get_string_value("quality_function_centrality", "")
        {
            if found {
                use crate::mu::oracle::quality_function::CentralityType;
                builder.mu.quality_function_centrality = match v.as_str() {
                    "log" => CentralityType::LogCenter,
                    "reciprocal" => CentralityType::ReciprocalCenter,
                    "cubed-reciprocal" => CentralityType::CubedReciprocalCenter,
                    _ => CentralityType::None,
                };
            }
        }
        if let Ok((v, found)) = self
            .options
            .get_string_value("quality_function_balancing_term", "")
        {
            if found {
                use crate::mu::oracle::quality_function::BalancingTermType;
                builder.mu.quality_function_balancing_term = match v.as_str() {
                    "cubic" => BalancingTermType::CubicTerm,
                    _ => BalancingTermType::None,
                };
            }
        }
        if let Some(v) = read_int("quality_function_max_section_steps") {
            builder.mu.quality_function_max_section_steps = v;
        }
        if let Some(v) = read_num("quality_function_section_sigma_tol") {
            builder.mu.quality_function_section_sigma_tol = v;
        }
        if let Some(v) = read_num("quality_function_section_qf_tol") {
            builder.mu.quality_function_section_qf_tol = v;
        }

        // `probing_iterate_quality_factor` — pounce-specific guard
        // (pounce#58) on the probing μ-oracle's input iterate. When
        // `curr_avrg_compl / curr_mu` exceeds this factor, the
        // μ-update layer signals restoration via
        // `IpoptData::request_resto` instead of letting probing
        // return `σ · mu_curr` ≫ previous μ. Default 1e4; set to ≤ 0
        // to disable. No upstream Ipopt counterpart.
        if let Some(v) = read_num("probing_iterate_quality_factor") {
            builder.mu.probing_iterate_quality_factor = v;
        }

        // Adaptive-μ extras — consumers in
        // `IpAdaptiveMuUpdate.cpp:RegisterOptions`. Only active when
        // `mu_strategy=adaptive`.
        if let Some(v) = read_num("adaptive_mu_safeguard_factor") {
            builder.mu.adaptive_mu_safeguard_factor = v;
        }
        if let Some(v) = read_num("adaptive_mu_monotone_init_factor") {
            builder.mu.adaptive_mu_monotone_init_factor = v;
        }
        if let Ok((v, found)) = self
            .options
            .get_bool_value("adaptive_mu_restore_previous_iterate", "")
        {
            if found {
                builder.mu.adaptive_mu_restore_previous_iterate = v;
            }
        }
        if let Some(v) = read_int("adaptive_mu_max_free_returns") {
            builder.mu.adaptive_mu_max_free_returns = v;
        }
        if let Some(v) = read_num("adaptive_mu_budget_pin_fraction") {
            builder.mu.adaptive_mu_budget_pin_fraction = v;
        }
        if let Some(v) = read_int("adaptive_mu_kkterror_red_iters") {
            if v >= 0 {
                builder.mu.adaptive_mu_kkterror_red_iters = v as usize;
            }
        }
        if let Some(v) = read_num("adaptive_mu_kkterror_red_fact") {
            builder.mu.adaptive_mu_kkterror_red_fact = v;
        }
        // `filter_margin_fact` / `filter_max_margin` (#551) — the margin
        // an entry must clear in the `obj-constr-filter` globalization
        // test. `AdaptiveMuUpdate` computes
        // `filter_margin_fact * min(filter_max_margin, err)` and has
        // always done so; only these two read sites were missing, so
        // setting either did nothing. Defaults equal the registered
        // defaults (1e-5 / 1.0), so an unset run is unchanged.
        if let Some(v) = read_num("filter_margin_fact") {
            builder.mu.filter_margin_fact = v;
        }
        if let Some(v) = read_num("filter_max_margin") {
            builder.mu.filter_max_margin = v;
        }
        if let Ok((v, found)) = self
            .options
            .get_string_value("adaptive_mu_kkt_norm_type", "")
        {
            if found {
                use crate::mu::adaptive::AdaptiveMuKktNorm;
                builder.mu.adaptive_mu_kkt_norm_type = match v.as_str() {
                    "1-norm" => AdaptiveMuKktNorm::OneNorm,
                    "2-norm" => AdaptiveMuKktNorm::TwoNorm,
                    "max-norm" => AdaptiveMuKktNorm::MaxNorm,
                    _ => AdaptiveMuKktNorm::TwoNormSquared,
                };
            }
        }

        // Watchdog options — consumers in
        // `IpBacktrackingLineSearch.cpp:RegisterOptions`. Baked into
        // the `BacktrackingLineSearch` at build time.
        if let Some(v) = read_int("watchdog_shortened_iter_trigger") {
            builder.line_search.watchdog_shortened_iter_trigger = v;
        }
        if let Some(v) = read_int("watchdog_trial_iter_max") {
            builder.line_search.watchdog_trial_iter_max = v;
        }
        if let Some(v) = read_num("soft_resto_pderror_reduction_factor") {
            builder.line_search.soft_resto_pderror_reduction_factor = v;
        }
        if let Some(v) = read_int("max_soft_resto_iters") {
            builder.line_search.max_soft_resto_iters = v;
        }
        // `alpha_red_factor` (#678) and `accept_after_max_steps` (#551)
        // — both consumed by the α-loop in `BacktrackingLineSearch`,
        // both registered without a read site until those issues.
        // `alpha_red_factor`'s default (0.5) equals the registered one,
        // and `accept_after_max_steps` defaults to `-1`, which disables
        // the escape hatch, so neither moves a solve that leaves them
        // alone.
        if let Some(v) = read_num("alpha_red_factor") {
            builder.line_search.alpha_red_factor = v;
        }
        if let Some(v) = read_num("alpha_red_factor_min") {
            builder.line_search.alpha_red_factor_min = Some(v);
        }
        if let Some(v) = read_int("accept_after_max_steps") {
            builder.line_search.accept_after_max_steps = v;
        }

        // Filter switching / Armijo / margin constants (#191). Consumed
        // by `FilterLsAcceptor` (only on the `Filter` line-search path);
        // registered but never read, so overrides were silently dropped.
        // Defaults equal the registered defaults.
        if let Some(v) = read_num("eta_phi") {
            builder.line_search.eta_phi = v;
        }
        // `delta` (#551) — the switching rule's multiplier on the
        // constraint violation (Eqn. (19)); `FilterLsAcceptor` has
        // always used it as `delta_armijo`, only the read site was
        // missing. Default 1.0 equals the registered default.
        if let Some(v) = read_num("delta") {
            builder.line_search.delta = v;
        }
        if let Some(v) = read_num("theta_min_fact") {
            builder.line_search.theta_min_fact = v;
        }
        if let Some(v) = read_num("theta_max_row_scale_kappa") {
            builder.line_search.theta_max_row_scale_kappa = v;
        }
        if let Some(v) = read_int("theta_max_adaptive_trigger") {
            builder.line_search.theta_max_adaptive_trigger = v.max(0) as u32;
        }
        if let Some(v) = read_num("theta_max_adaptive_factor") {
            builder.line_search.theta_max_adaptive_factor = v;
        }
        if let Some(v) = read_int("theta_max_adaptive_max_raises") {
            builder.line_search.theta_max_adaptive_max_raises = v.max(0) as u32;
        }
        if let Some(v) = read_num("theta_max_fact") {
            builder.line_search.theta_max_fact = v;
        }
        if let Some(v) = read_num("gamma_phi") {
            builder.line_search.gamma_phi = v;
        }
        if let Some(v) = read_num("gamma_theta") {
            builder.line_search.gamma_theta = v;
        }
        if let Some(v) = read_num("s_phi") {
            builder.line_search.s_phi = v;
        }
        if let Some(v) = read_num("s_theta") {
            builder.line_search.s_theta = v;
        }
        if let Some(v) = read_num("alpha_min_frac") {
            builder.line_search.alpha_min_frac = v;
        }
        if let Some(v) = read_num("obj_max_inc") {
            builder.line_search.obj_max_inc = v;
        }
        if let Some(v) = read_int("max_filter_resets") {
            builder.line_search.max_filter_resets = v;
        }
        if let Some(v) = read_int("filter_reset_trigger") {
            builder.line_search.filter_reset_trigger = v;
        }
        // Penalty line-search constants (#551), consumed by
        // `PenaltyLsAcceptor` (only on the `line_search_method=penalty`
        // / `cg-penalty` paths). The acceptor implements ν and the
        // Armijo test on the penalty merit function already; these four
        // were registered with no read site, so tuning the penalty
        // update did nothing. Defaults equal the registered defaults.
        if let Some(v) = read_num("nu_init") {
            builder.line_search.nu_init = v;
        }
        if let Some(v) = read_num("nu_inc") {
            builder.line_search.nu_inc = v;
        }
        if let Some(v) = read_num("rho") {
            builder.line_search.rho = v;
        }
        if let Some(v) = read_num("eta_penalty") {
            builder.line_search.eta_penalty = v;
        }

        // Second-order-correction constants (#191), consumed by
        // `BacktrackingLineSearch`. `max_soc = 0` disables SOC.
        if let Some(v) = read_int("max_soc") {
            builder.line_search.max_soc = v;
        }
        if let Some(v) = read_num("kappa_soc") {
            builder.line_search.kappa_soc = v;
        }
        if let Some(v) = read_int("soc_method") {
            builder.line_search.soc_method = v;
        }

        // Inertia-correction / Jacobian-regularization constants (#191),
        // consumed by `PdPerturbationHandler`. Registered but never read.
        if let Some(v) = read_num("max_hessian_perturbation") {
            builder.perturbation.max_hessian_perturbation = v;
        }
        if let Some(v) = read_num("min_hessian_perturbation") {
            builder.perturbation.min_hessian_perturbation = v;
        }
        if let Some(v) = read_num("perturb_inc_fact_first") {
            builder.perturbation.perturb_inc_fact_first = v;
        }
        if let Some(v) = read_num("perturb_inc_fact") {
            builder.perturbation.perturb_inc_fact = v;
        }
        if let Some(v) = read_num("perturb_dec_fact") {
            builder.perturbation.perturb_dec_fact = v;
        }
        if let Some(v) = read_num("first_hessian_perturbation") {
            builder.perturbation.first_hessian_perturbation = v;
        }
        if let Some(v) = read_num("jacobian_regularization_value") {
            builder.perturbation.jacobian_regularization_value = v;
        }
        if let Some(v) = read_num("jacobian_regularization_exponent") {
            builder.perturbation.jacobian_regularization_exponent = v;
        }
        if let Ok((v, true)) = self.options.get_bool_value("perturb_always_cd", "") {
            builder.perturbation.perturb_always_cd = v;
        }
        if let Some(v) = read_int("perturb_delta_c_max_rungs") {
            builder.perturbation.perturb_delta_c_max_rungs = v;
        }

        // Iterative-refinement constants (#191), consumed by
        // `PdFullSpaceSolver`. Registered but never read.
        if let Some(v) = read_int("min_refinement_steps") {
            builder.refinement.min_refinement_steps = v;
        }
        if let Some(v) = read_int("max_refinement_steps") {
            builder.refinement.max_refinement_steps = v;
        }
        if let Some(v) = read_num("residual_ratio_max") {
            builder.refinement.residual_ratio_max = v;
        }
        if let Some(v) = read_num("residual_ratio_singular") {
            builder.refinement.residual_ratio_singular = v;
        }
        if let Some(v) = read_num("residual_improvement_factor") {
            builder.refinement.residual_improvement_factor = v;
        }

        // Inertia-free curvature test (#551 / #677), also consumed by
        // `PdFullSpaceSolver`. `neg_curv_test_tol` had a field that only
        // ever held its 0.0 default, and `neg_curv_test_reg` had none at
        // all; both are now read, and the curvature test they configure
        // is implemented in `PdFullSpaceSolver::solve_once`. At the
        // registered default (`0.0`) the heuristic is off and the
        // inertia check runs as before.
        if let Some(v) = read_num("neg_curv_test_tol") {
            builder.refinement.neg_curv_test_tol = v;
        }
        if let Ok((v, true)) = self.options.get_bool_value("neg_curv_test_reg", "") {
            builder.refinement.neg_curv_test_reg = v;
        }

        // Restoration-phase constants (#191). Carried on the outer builder
        // and copied into the `RestoAlgorithmBuilder` when the restoration
        // factory is minted (the frontends pass this builder in). The
        // restoration builder was never options-configured, so these were
        // registered but never read. Defaults equal the registered
        // defaults.
        if let Some(v) = read_num("bound_mult_reset_threshold") {
            builder.resto.bound_mult_reset_threshold = v;
        }
        if let Some(v) = read_num("constr_mult_reset_threshold") {
            builder.resto.constr_mult_reset_threshold = v;
        }
        if let Some(v) = read_num("resto_penalty_parameter") {
            builder.resto.resto_penalty_parameter = v;
        }
        if let Some(v) = read_num("resto_proximity_weight") {
            builder.resto.resto_proximity_weight = v;
        }
        // `required_infeasibility_reduction` (#439) — the κ_resto guard the
        // restoration sub-solve exits on. Registered since #191 but the
        // value was hardcoded at the callsite, so setting it was a silent
        // no-op.
        if let Some(v) = read_num("required_infeasibility_reduction") {
            builder.resto.required_infeasibility_reduction = v;
        }
        // gh#483 / #191 round 2: three restoration switches whose fields
        // `RestoAlgorithmBuilder` has consumed all along — the read site
        // was the only missing piece, so setting them did nothing.
        let read_yes = |key: &str| -> Option<bool> {
            match self.options.get_string_value(key, "") {
                Ok((v, true)) => Some(v.eq_ignore_ascii_case("yes")),
                _ => None,
            }
        };
        if let Some(v) = read_yes("evaluate_orig_obj_at_resto_trial") {
            builder.resto.evaluate_orig_obj_at_resto_trial = v;
        }
        if let Some(v) = read_yes("expect_infeasible_problem") {
            builder.resto.expect_infeasible_problem = v;
        }
        if let Some(v) = read_yes("start_with_resto") {
            builder.resto.start_with_resto = v;
        }
        // `max_resto_iter` (#551 / #677) — the cap on *successive*
        // restoration iterations. `RestoConvCheckAdapter` has enforced a cap
        // all along (returning `MaxIterExceeded` at the limit); the number
        // it enforced was a hard-coded constant in `resto_inner_solver.rs`,
        // so setting the option did nothing. The consumer's field is
        // `maximum_resto_iters`, not the option name — grepping for
        // `max_resto_iter` found only the registry (#551 caution 2).
        //
        // DEFAULT MISMATCH, LEFT AS IT IS ON PURPOSE: the registry declares
        // upstream's 3000000, pounce's effective cap is 3000
        // (`RestoOptions::default`). `read_int` fires only when the user set
        // the key, so an unset `max_resto_iter` still means 3000 and this
        // wiring is trajectory-neutral. Adopting upstream's number would let
        // restorations pounce currently truncates run on, which is a
        // trajectory change and needs its own measurement.
        if let Some(v) = read_int("max_resto_iter") {
            builder.resto.max_resto_iter = v;
        }

        // Iteration-output options — consumed by `OrigIterationOutput`.
        if let Some(v) = read_int("print_frequency_iter") {
            builder.output.print_frequency_iter = v;
        }
        if let Some(v) = read_num("print_frequency_time") {
            builder.output.print_frequency_time = v;
        }
        if let Ok((v, found)) = self.options.get_bool_value("print_info_string", "") {
            if found {
                builder.output.print_info_string = v;
            }
        }
        if let Ok((v, found)) = self.options.get_string_value("inf_pr_output", "") {
            if found {
                builder.output.inf_pr_output_internal = v == "internal";
            }
        }

        // Warm-start options — consumed by `WarmStartIterateInitializer`
        // (port of `IpWarmStartIterateInitializer.cpp:RegisterOptions`).
        // `warm_start_init_point` is the toggle that picks between the
        // default (cold) and warm-start initializers; the remaining
        // knobs are baked onto the chosen initializer at build time.
        if let Ok((v, found)) = self.options.get_bool_value("warm_start_init_point", "") {
            if found {
                builder.warm_start_init_point = v;
            }
        }
        if let Some(v) = read_num("warm_start_bound_push") {
            builder.warm.bound_push = v;
        }
        if let Some(v) = read_num("warm_start_bound_frac") {
            builder.warm.bound_frac = v;
        }
        if let Some(v) = read_num("warm_start_slack_bound_push") {
            builder.warm.slack_bound_push = v;
        }
        if let Some(v) = read_num("warm_start_slack_bound_frac") {
            builder.warm.slack_bound_frac = v;
        }
        if let Some(v) = read_num("warm_start_mult_bound_push") {
            builder.warm.mult_bound_push = v;
        }
        if let Some(v) = read_num("warm_start_mult_init_max") {
            builder.warm.mult_init_max = v;
        }
        if let Some(v) = read_num("warm_start_target_mu") {
            builder.warm.target_mu = v;
        }
        // gh#606: residual-adaptive recentering. `warm_start_entire_iterate`
        // and `warm_start_same_structure` used to be parsed here into
        // fields nothing read; they are refused by
        // `unimplemented_options` instead (they name the
        // `GetWarmStartIterate` TNLP surface pounce does not expose).
        if let Ok((v, found)) = self.options.get_string_value("warm_start_recentering", "") {
            if found {
                builder.warm.recentering = if v.eq_ignore_ascii_case("none") {
                    crate::alg_builder::WarmStartRecentering::None
                } else {
                    crate::alg_builder::WarmStartRecentering::Residual
                };
            }
        }

        // `DefaultIterateInitializer` knobs — parsed after the Mehrotra
        // cascade so explicit user values win
        // (mirrors upstream's `SetNumericValueIfUnset` semantics).
        if let Some(v) = read_num("bound_push") {
            builder.init.bound_push = v;
        }
        if let Some(v) = read_num("bound_frac") {
            builder.init.bound_frac = v;
        }
        if let Some(v) = read_num("slack_bound_push") {
            builder.init.slack_bound_push = v;
        }
        if let Some(v) = read_num("slack_bound_frac") {
            builder.init.slack_bound_frac = v;
        }
        if let Some(v) = read_num("constr_mult_init_max") {
            builder.init.constr_mult_init_max = v;
        }
        if let Some(v) = read_num("bound_mult_init_val") {
            builder.init.bound_mult_init_val = v;
        }
        if let Ok((v, found)) = self.options.get_string_value("bound_mult_init_method", "") {
            if found {
                builder.init.bound_mult_init_method = v;
            }
        }
        if let Ok((v, found)) = self
            .options
            .get_string_value("least_square_init_primal", "")
        {
            if found {
                builder.init.least_square_init_primal = v == "yes";
            }
        }
        builder
    }
}

/// Map the integer `print_level` / `file_print_level` option to the
/// matching [`JournalLevel`] variant. Mirrors upstream's
/// `static_cast<EJournalLevel>(int_value)` with clamping.
/// The eight block dimensions of an iterate, in canonical order
/// (x, s, y_c, y_d, z_l, z_u, v_l, v_u). Used to guard the debugger's
/// warm-restart install against a structural mismatch between solves.
fn iterates_dims(c: &IteratesVector) -> [i32; 8] {
    [
        c.x.dim(),
        c.s.dim(),
        c.y_c.dim(),
        c.y_d.dim(),
        c.z_l.dim(),
        c.z_u.dim(),
        c.v_l.dim(),
        c.v_u.dim(),
    ]
}

fn journal_level_from_int(v: i32) -> JournalLevel {
    match v.clamp(0, 12) {
        0 => JournalLevel::J_NONE,
        1 => JournalLevel::J_ERROR,
        2 => JournalLevel::J_STRONGWARNING,
        3 => JournalLevel::J_SUMMARY,
        4 => JournalLevel::J_WARNING,
        5 => JournalLevel::J_ITERSUMMARY,
        6 => JournalLevel::J_DETAILED,
        7 => JournalLevel::J_MOREDETAILED,
        8 => JournalLevel::J_VECTOR,
        9 => JournalLevel::J_MOREVECTOR,
        10 => JournalLevel::J_MATRIX,
        11 => JournalLevel::J_MOREMATRIX,
        _ => JournalLevel::J_ALL,
    }
}

/// MA57 backend knobs snapshotted off an `OptionsList`, ready to be
/// handed to a backend factory.
///
/// The type exists — rather than passing `pounce_hsl::Ma57Options`
/// directly — so that the factory signature is the same shape whether or
/// not the `ma57` cargo feature is on. `pounce-py`, `pounce-cinterface`
/// and `pounce-restoration` all call [`default_backend_factory`] and none
/// of them can name a `pounce-hsl` type; without the feature this struct
/// carries nothing and costs nothing.
///
/// Build it with [`ma57_config_from_options`]. It is the MA57 half of
/// what [`feral_config_from_options`] does for FERAL, and the asymmetry
/// between the two was the root cause of gh#825: FERAL's knobs were
/// threaded into the factory and MA57's were not, so the factory had
/// nothing to give MA57 and called `Ma57SolverInterface::new()` — which
/// hard-codes the defaults. Every `ma57_*` option was registered,
/// documented, accepted, and then silently discarded.
#[derive(Debug, Clone, Default)]
pub struct Ma57Config {
    #[cfg(feature = "ma57")]
    opts: pounce_hsl::Ma57Options,
}

impl Ma57Config {
    /// The wrapped MA57 settings.
    #[cfg(feature = "ma57")]
    pub fn options(&self) -> &pounce_hsl::Ma57Options {
        &self.opts
    }
}

/// Read the `ma57_*` options off `options` under `prefix`, for handing
/// to [`default_backend_factory`] / [`default_backend_factory_with_sink`].
///
/// `prefix` is `""` for the main IPM and `"resto."` for the restoration
/// sub-IPM, mirroring upstream's
/// `Ma57TSolverInterface::InitializeImpl(options, prefix)`. The
/// restoration sub-IPM builds its own backend through its own
/// `InnerBackendFactoryFactory` (see
/// `pounce_restoration::resto_inner_solver`), so the two really are
/// separately configurable — before gh#825 neither was.
///
/// Without the `ma57` cargo feature this returns an empty config and
/// does not touch `options`; nothing downstream can consume MA57
/// settings in that build.
pub fn ma57_config_from_options(
    options: &pounce_common::options_list::OptionsList,
    prefix: &str,
) -> Ma57Config {
    #[cfg(feature = "ma57")]
    {
        Ma57Config {
            opts: pounce_hsl::Ma57Options::from_options_list(options, prefix),
        }
    }
    #[cfg(not(feature = "ma57"))]
    {
        let _ = (options, prefix);
        Ma57Config::default()
    }
}

/// Construct the MA57 backend for a factory, or fall back to FERAL when
/// the `ma57` cargo feature is off.
///
/// Factored out of the two factories below so there is exactly one place
/// that decides how an MA57 backend is built from a [`Ma57Config`]. Both
/// factories used to inline `Ma57SolverInterface::new()`, and the
/// duplication is half of why gh#825 was easy to miss.
fn make_ma57_backend(
    ma57_cfg: &Ma57Config,
    feral_fallback: impl FnOnce() -> Box<dyn SparseSymLinearSolverInterface>,
) -> Box<dyn SparseSymLinearSolverInterface> {
    #[cfg(feature = "ma57")]
    {
        let _ = feral_fallback;
        Box::new(pounce_hsl::Ma57SolverInterface::with_options(
            *ma57_cfg.options(),
        ))
    }
    #[cfg(not(feature = "ma57"))]
    {
        // ma57 feature not compiled in — fall back to FERAL.
        let _ = ma57_cfg;
        feral_fallback()
    }
}

/// Default symmetric linear-solver factory, parameterized by the
/// pounce-extension FERAL knobs and the `ma57_*` knobs read off the
/// application's `OptionsList`.
///
/// FERAL (pure-Rust) is the shipping default. The HSL MA57 backend is
/// available when the `ma57` cargo feature is enabled; without it,
/// requesting `linear_solver = ma57` falls back to FERAL with a
/// warning printed by the journalist (see [`AlgorithmBuilder`]).
///
/// Both configs are snapshots, not live views: take them with
/// [`feral_config_from_options`] and [`ma57_config_from_options`] at the
/// point the application's options are fully populated. The factory is
/// called more than once per solve (the main KKT solver and, under
/// limited memory, the Hessian-free bypass solver), so each call gets a
/// fresh backend built from the same snapshot.
pub fn default_backend_factory(
    feral_cfg: pounce_feral::FeralConfig,
    ma57_cfg: Ma57Config,
) -> LinearBackendFactory {
    Box::new(
        move |choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
            match choice {
                LinearSolverChoice::Feral => Box::new(
                    pounce_feral::FeralSolverInterface::with_config(feral_cfg.clone()),
                ),
                LinearSolverChoice::Ma57 => make_ma57_backend(&ma57_cfg, || {
                    Box::new(pounce_feral::FeralSolverInterface::with_config(
                        feral_cfg.clone(),
                    ))
                }),
            }
        },
    )
}

/// Sink-aware variant of [`default_backend_factory`]. Identical
/// dispatch, but the FERAL backend is constructed with a
/// `LinearSolverSummary` sink so [`IpoptApplication`] can read out
/// aggregate post-mortem stats (factor counts, fill ratio, extremal
/// pivots, final inertia) after the solve returns. MA57 ignores the
/// sink — the HSL backend doesn't carry the same instrumentation yet.
pub fn default_backend_factory_with_sink(
    feral_cfg: pounce_feral::FeralConfig,
    ma57_cfg: Ma57Config,
    sink: Arc<Mutex<LinearSolverSummary>>,
) -> LinearBackendFactory {
    Box::new(
        move |choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
            match choice {
                LinearSolverChoice::Feral => Box::new(
                    pounce_feral::FeralSolverInterface::with_config(feral_cfg.clone())
                        .with_summary_sink(Arc::clone(&sink)),
                ),
                LinearSolverChoice::Ma57 => make_ma57_backend(&ma57_cfg, || {
                    Box::new(
                        pounce_feral::FeralSolverInterface::with_config(feral_cfg.clone())
                            .with_summary_sink(Arc::clone(&sink)),
                    )
                }),
            }
        },
    )
}

/// Read the `feral_*` extension options off `options`, falling
/// back to the env-var defaults baked into [`pounce_feral::FeralConfig::from_env`]
/// for any knob the caller did not set explicitly. The returned
/// config is what every default-factory invocation (main IPM and
/// restoration sub-IPM) consumes.
pub fn feral_config_from_options(
    options: &pounce_common::options_list::OptionsList,
) -> pounce_feral::FeralConfig {
    let mut cfg = pounce_feral::FeralConfig::from_env();
    // Tri-state: the `(_, true)` arm only fires when the user set the
    // option explicitly. Leaving it unset keeps `cfg.cascade_break` at
    // `None`, which inherits FERAL's `NumericParams::default()` (CB on
    // as of FERAL Phase B / pounce#55). `Some(false)` explicitly
    // disarms (reproduces pre-Phase-B behaviour, surfaces FERAL's
    // `DelayBudgetExceeded` on non-root cascade victims).
    if let Ok((v, true)) = options.get_bool_value("feral_cascade_break", "") {
        cfg.cascade_break = Some(v);
    }
    if let Ok((v, true)) = options.get_bool_value("feral_fma", "") {
        cfg.fma = v;
    }
    // Not tri-state, and deliberately not: on the limited-memory path
    // the IPM's default is the opposite of the library's (gh#710,
    // gh#698 obs 5). `FeralConfig` ships `refine = true` because a
    // caller that only refines its own system needs the backend loop.
    //
    // Scoped to limited-memory, because that is where the win was
    // measured and where it comes from. Under L-BFGS the Hessian-free
    // bypass batches the low-rank SMW correction into one multi-RHS
    // back-solve, and the backend loop refines per right-hand side, so
    // its cost scales with the memory depth — while what it polishes is
    // the *condensed* system, which is not where Waechter-Biegler 3.10
    // puts refinement. Turning it off takes `laptime` under
    // limited-memory from 1397 s to 394 s.
    //
    // The same switch is a net loss on the exact path, so it is not
    // applied there. It costs `NARX_CFy` 230 iterations (400 -> 630,
    // 173 s -> 250 s) to buy back 34 on `laptime` (380 -> 346). And the
    // exact path has no rung left to stand in for it: the
    // `increase_quality` escalation that once argued for dropping the
    // backend loop here proved inert and was itself removed, so with
    // `refine` off the host loop in `PdFullSpaceSolver` is the only
    // refinement in the stack — and it stops the moment it crosses
    // `residual_ratio_max` (1e-10), three orders looser than the ~1e-16
    // the backend-refined solves reach.
    //
    // Tightening that threshold instead is not the fix, and was measured
    // rather than assumed: it is chaotic on this corpus, flipping `deb7`
    // to `Error_In_Step_Computation` at 1e-12 and swinging
    // `pooling_rt2stp` between 107 and 199 iterations across 1e-11 to
    // 1e-13. Forcing extra passes via `min_refinement_steps` behaves the
    // same way (both fixtures break at 2). Restoring the backend loop is
    // trajectory-neutral next to either (`deb7` 146 -> 147,
    // `pooling_rt2stp` 107 -> 109).
    //
    // Ordered env-then-registry-then-option so `POUNCE_FERAL_REFINE`
    // still reaches this path and an explicit `feral_refine` still beats
    // both.
    //
    // gh#909: the registry read below takes the value and IGNORES the
    // found flag, which is the whole point. Reading it as `Ok((v, true))`
    // — the tri-state shape every other option in this function uses —
    // consults the user's setting and never the *registered default*, so
    // an unset option left `cfg.refine` at `FeralConfig::default()`'s
    // `true` while `pounce --print-options` reported the registry's.
    // Registered `no`, ran `yes`: the gh#677 shape, one option family
    // over. It is fixed by making the registry authoritative here rather
    // than by changing what runs — see below for why — so the registered
    // default moved to `yes` in the same commit and this read is what
    // keeps the two from drifting apart again.
    //
    // The scoping stays. There is no single default that is right on both
    // paths, and this is a two-sided trade of the same shape as
    // `feral_increase_quality` above: the option is the lever, and the
    // default is a choice rather than a fact. Measured on `e5f5830b`,
    // one binary, `feral_refine=no` against the default:
    //
    //   fixture corpus, exact leg: 10 legs move, NO status flips, and
    //     the net is favourable -- `issue_508_infeasible_gap_1em4`
    //     441 -> 245, `square_flowsheet_resto` 54 -> 47 (tot 185 -> 176),
    //     `issue_508_infeasible_gap_1em2` 114 -> 112, `deb7` 147 -> 146,
    //     `pooling_rt2stp` 109 -> 107, against one loss
    //     (`mu_fallback_point_floor` 31 -> 32).
    //   fixture corpus, lbfgs leg: identical, refine is already off there.
    //   `NARX_CFy` (Mittelmann): 400 -> 630 iterations, 208.7 s -> 254.2 s.
    //   `eigena2` final dual infeasibility: 3.43e-10 -> 5.20e-09, both
    //     inside `tol`, and see `issue_540_eigena2_superlinear_tail.rs`.
    //   AC-OPF (PGLib), gh#909: 10-18% faster, iteration counts
    //     unchanged, solution identical to ~13 significant figures.
    //
    // So the corpus and the AC-OPF benchmarks both prefer `no` and
    // `NARX_CFy` pays 57% more iterations for it. Nothing here separates
    // those, which is why the default is scoped rather than flipped.
    let limited_memory = matches!(
        options.get_string_value("hessian_approximation", ""),
        Ok((ref s, true)) if s == "limited-memory"
    );
    if std::env::var_os("POUNCE_FERAL_REFINE").is_none() {
        // Deliberately `_`, not `true`: see above.
        if let Ok((v, _)) = options.get_bool_value("feral_refine", "") {
            cfg.refine = v;
        }
        if limited_memory {
            cfg.refine = false;
        }
    }
    // An explicit setting beats both the carve-out and the environment.
    if let Ok((v, true)) = options.get_bool_value("feral_refine", "") {
        cfg.refine = v;
    }
    // gh #850: `feral_increase_quality` exists because this rung is a genuine
    // two-sided trade, and the option is the lever — the default is left ON,
    // which is the 0.11 behaviour.
    //
    // Ipopt's `IncreaseQuality` contract assumes a *monotone* escalation: MA57
    // raises `pivtol` toward `pivtolmax`, strictly more conservative each time,
    // so keeping it raised for the rest of the solve can only make the
    // factorization safer. FERAL's ladder changes which pivots are taken, which
    // is lateral in trajectory terms, and it persists the same way. So it
    // reroutes solves, and the reroute goes both ways:
    //
    //   it COSTS two whole solves, both `square_flowsheet_resto`:
    //     exact  Optimal/99  -> RestorationFailed/131 (a second-opinion rung
    //                          rescues it, at 185 iterations total)
    //     lbfgs  Optimal/178 -> 3000 iterations at the cap, rescued by nothing
    //   it BUYS accuracy where nothing else does:
    //     `watchdog_trial_is_not_a_divergence_verdict`'s 12-variable model ends
    //     `SolvedToAcceptableLevel` at obj 3.7e-6 with the rung and at obj 3.42
    //     against `f* = 0` without it — a wrong-ish answer under a
    //     success-shaped status, which is worse than an honest failure.
    //   and it buys iterations on five more fixture-legs (15-25%).
    //
    // There is no scoping that separates those. Measured with a process-global
    // firing cap on `square_flowsheet_resto`: the rung fires twice, once in the
    // main solve at iteration 25 and once inside restoration at `76r`, and
    // allowing only the first still loses the leg — so declining it just for
    // the restoration sub-solve would not help. Nor does a count separate them:
    // `deb7` and `square_flowsheet_resto` each fire it exactly twice on their
    // exact legs, one gaining 16% of its iterations and the other losing its
    // verdict.
    //
    // So the default stands, and the losing direction now recovers itself:
    // `feral_increase_quality_retry` (gh#857) re-solves once with this off when
    // a solve that actually escalated ends `Restoration_Failed` or
    // `Maximum_Iterations_Exceeded`.
    //
    // It is a re-solve rather than a *revertible* escalation because the
    // revertible one was tried. jkitchin/feral#192 landed as `reset_quality`,
    // was plumbed here and instrumented (376 escalations, 376 matching resets
    // on one solve), and recovers neither leg at either re-baselining boundary
    // -- the harm is the destination, not the duration. See
    // `dev-notes/second-opinion-promotions-in-the-sweep.md`.
    if let Ok((v, true)) = options.get_bool_value("feral_increase_quality", "") {
        cfg.increase_quality = v;
    }
    // Only consulted when `refine` is on; see `FeralConfig::refine_max_steps`
    // (gh#710). Registered as an integer option with lower bound 0, so the
    // cast cannot go negative.
    if let Ok((v, true)) = options.get_integer_value("feral_refine_steps", "") {
        cfg.refine_max_steps = v.max(0) as usize;
    }
    // Also only consulted when `refine` is on. Registered with lower bound
    // 0, and 0 disables the pre-check; see `FeralConfig::refine_target`.
    if let Ok((v, true)) = options.get_numeric_value("feral_refine_target", "") {
        cfg.refine_target = v.max(0.0);
    }
    // Explicit static-pivoting opt-in (feral#8 cascade breaker, pounce#254).
    // Same tri-state discipline: unset leaves `cfg.static_pivoting` at
    // whatever `from_env` resolved (`None` → inherit feral's delayed-pivot
    // default), so the default numeric path is unchanged.
    if let Ok((v, true)) = options.get_bool_value("feral_static_pivoting", "") {
        cfg.static_pivoting = Some(v);
    }
    if let Ok((v, true)) = options.get_numeric_value("feral_singular_pivot_floor", "") {
        cfg.singular_pivot_floor = v;
    }
    // Explicitly set pins an absolute floor for every dimension (`0`
    // disables the trigger); left unset, `None` keeps the dimension-aware
    // `n * eps` default (pounce gh#592).
    if let Ok((v, true)) = options.get_numeric_value("feral_inertia_pivot_floor", "") {
        cfg.inertia_pivot_floor = Some(v);
    }
    // Number option (not integer): the gate is a u64 and Index is i32, too
    // narrow for large flop counts or the u64::MAX reject-all sentinel. The
    // lower bound (0.0) rules out negatives; `as u64` then saturates a very
    // large finite value to u64::MAX (reject all tree-level parallelism).
    if let Ok((v, true)) = options.get_numeric_value("feral_min_par_flops", "") {
        cfg.min_par_flops = Some(v as u64);
    }
    if let Ok((v, true)) = options.get_numeric_value("feral_pivtol", "") {
        cfg.pivtol = v;
    }
    // Only override on explicit set so `from_env` (which itself
    // defaults to OrderingMethod::Auto) keeps governing unset cases.
    // Unrecognized tags are silently ignored — the registered enum
    // restricts inputs at the OptionsList layer.
    if let Ok((v, true)) = options.get_string_value("feral_ordering", "") {
        if let Some(m) = pounce_feral::parse_ordering_method(&v) {
            cfg.ordering = m;
        }
    }
    // Same explicit-set discipline as `feral_ordering`: `from_env`
    // defaults to ScalingStrategy::Auto (FERAL's current default), so
    // leaving the option unset preserves existing behaviour exactly.
    if let Ok((v, true)) = options.get_string_value("feral_scaling", "") {
        if let Some(s) = pounce_feral::parse_scaling_strategy(&v) {
            cfg.scaling = s;
        }
    }
    cfg
}

/// Withdraw a numerical infeasibility verdict the model's own starting point
/// disproves.
///
/// Applied at every site in this file that can return
/// `Infeasible_Problem_Detected` from a *numerical* argument — the IPM path's
/// restoration / cycle gates, the SQP path's infeasible-subproblem exit, and the
/// ℓ₁ wrapper's uncollapsed-slack certificate. Deliberately one gate rather than
/// three: the two preceding safeguards in this area (gh #376, gh #380) were each
/// added to one path and not its twin, and a hole survived both times.
///
/// Not applied to a presolve *certificate*
/// (`TNLP::presolve_infeasibility_proof`), which carries its own, tighter
/// refutation
/// (`pounce_presolve::witness_refutes_infeasibility`) and is a proof rather than
/// a numerical inference.
///
/// The replacement is `Error_In_Step_Computation`, the status this codebase
/// already uses for "the solve broke down and we are **not** claiming
/// infeasibility" — see the `cycle_exit` fallback in
/// [`crate::ipopt_alg::IpoptAlgorithm::invoke_restoration`], which picks between
/// exactly these two on exactly this question. It maps to AMPL 500, an honest
/// failure the caller can see, instead of AMPL 200, a wrong answer they cannot.
///
/// gh #379.
fn withdraw_infeasibility_if_refuted(
    tnlp: &Rc<RefCell<dyn TNLP>>,
    solver_status: SolverReturn,
    lo_inf: Number,
    up_inf: Number,
    tol: Number,
) -> SolverReturn {
    if solver_status != SolverReturn::LocalInfeasibility {
        return solver_status;
    }
    // A presolve proof is not a numerical inference; it does its own refutation.
    if tnlp.borrow().presolve_infeasibility_proof().is_some() {
        return solver_status;
    }
    match crate::infeasibility_refutation::starting_point_refutes_infeasibility(
        tnlp, lo_inf, up_inf, tol,
    ) {
        Some(w) => {
            tracing::debug!(
                target: "pounce::application",
                "[PN_INFEAS_REFUTED] the model's starting point satisfies every constraint \
                 (max violation {:.3e}) — withdrawing Infeasible_Problem_Detected",
                w.max_violation
            );
            SolverReturn::ErrorInStepComputation
        }
        None => solver_status,
    }
}

/// How well a point satisfies the **user's own** rows and bounds.
///
/// Computed from `g = c(x)` and the inner TNLP's declared bounds, in the
/// user's units, with no reference to whatever problem the algorithm
/// actually iterated on. That distinction is the whole reason this
/// exists: on the ℓ₁ path the IPM converges the *augmented* NLP
/// `c(x) − p + n = target`, whose equality rows the slacks satisfy to
/// machine precision by construction, and reporting that residual as
/// the solve's constraint violation says nothing about `c(x) − target`
/// (gh#794 finding P1).
struct OriginalSpaceFeasibility {
    /// Largest absolute violation of any row or bound, in the user's units.
    max_violation: Number,
    /// Every row and bound negligible at `tol`, judged scale-relative.
    negligible_at_tol: bool,
    /// The same at `acceptable_tol`.
    negligible_at_acceptable: bool,
}

/// Measure [`OriginalSpaceFeasibility`] at `x`, given `g = c(x)`.
///
/// `is_negligible` rather than `!is_significant`, deliberately, and for
/// the reason that function's own documentation gives: the question here
/// is "did the solve converge well enough to call this point feasible",
/// which must never demand more precision than the solver promised, so
/// the threshold is clamped at `tol` from below (`tol · max(|scale|, 1)`).
/// The refutation path next door asks the opposite question — "is this
/// residual real at this row's scale" — and correctly uses the pure
/// relative form.
///
/// Returns `None` when the model cannot be measured (bounds unreadable, a
/// non-finite value). `None` means "not measured", never "feasible": the
/// caller keeps whatever verdict it already had.
fn original_space_feasibility(
    tnlp: &Rc<RefCell<dyn TNLP>>,
    x: &[Number],
    g: &[Number],
    lower_bound_inf: Number,
    upper_bound_inf: Number,
    tol: Number,
    acceptable_tol: Number,
    constr_viol_tol: Number,
    acceptable_constr_viol_tol: Number,
    noise_floor_kappa: Number,
) -> Option<OriginalSpaceFeasibility> {
    use pounce_common::tolerance::is_negligible;

    let info = tnlp.borrow_mut().get_nlp_info()?;
    let n = info.n.max(0) as usize;
    let m = info.m.max(0) as usize;
    if x.len() < n || g.len() < m {
        return None;
    }

    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    if !tnlp.borrow_mut().get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) {
        return None;
    }

    let mut max_violation: Number = 0.0;
    let mut ok_tol = true;
    let mut ok_acceptable = true;

    // Only *finite, present* bounds inform a row's magnitude: letting the
    // `±1e19` sentinel set the scale would make every row look satisfied,
    // the same trap `infeasibility_refutation` documents.
    let present = |b: Number, is_lower: bool| -> Option<Number> {
        let absent = if is_lower {
            b <= lower_bound_inf
        } else {
            b >= upper_bound_inf
        };
        (b.is_finite() && !absent).then_some(b)
    };

    // Accumulates into `max_violation` / `ok_tol` / `ok_acceptable`; a
    // non-positive `viol` means the side is satisfied and contributes
    // nothing. Returns nothing — every caller is a statement.
    let mut judge = |viol: Number, scale: Number| {
        if viol <= 0.0 {
            return;
        }
        max_violation = max_violation.max(viol);
        // The scale-relative test alone is not a feasibility standard, and
        // on a large-magnitude row it is not even close to one: it accepts
        // anything up to `tol · |row|`, which on a row near `1e10` is `1e2`
        // at the default `tol`. An adversary probe on this branch built a
        // model infeasible by exactly `50` with its row at `1e10` and got
        // `Solve_Succeeded` — a *worse* verdict than this branch's own
        // parent, which refused the same point (the old `Σ(p+n)` argument
        // was crude but absolute). So the wrapper has to judge feasibility
        // the way the rest of the solver does.
        //
        // `OptErrorConvCheck::primal_component_passes` is that standard:
        // an absolute `constr_viol <= constr_viol_tol`, with scale-awareness
        // supplied by an abstention when every row sits at its own
        // floating-point noise floor (gh#528/gh#590) rather than by
        // multiplying the tolerance by the row's magnitude. That
        // abstention "cannot fabricate a success on a genuinely infeasible
        // model: such a model's violation is pinned at its infeasibility
        // gap, orders above `eps ·` the row's own magnitude" — which is
        // exactly the property `is_negligible` lacks and the probe
        // exploited (`50` is `~2e7 ×` this row's floor).
        //
        // The strict gate's `primal_resolvable` cannot be reused verbatim:
        // it is computed by the CQ on the *augmented* NLP, whose rows the
        // slacks satisfy to machine precision, so it would abstain always
        // and accept everything. The floor is therefore recomputed here on
        // the user's own row, from the same `kappa · eps · magnitude` the
        // option documents. `primal_noise_floor_kappa = 0` opts out, as it
        // does for the strict gate.
        //
        // Both arms are conjoined rather than substituted: the relative
        // test still catches a violation that is small in absolute terms
        // but large for its row, which is the gh#794 P1 case itself
        // (`ralph1` at `2.5e-7` under a `2.5e-11` tol).
        let noise = noise_floor_kappa * Number::EPSILON * scale.abs();
        let absolute_ok = |bound: Number| viol <= bound || viol <= noise;
        ok_tol &= is_negligible(viol, scale, tol) && absolute_ok(constr_viol_tol);
        ok_acceptable &=
            is_negligible(viol, scale, acceptable_tol) && absolute_ok(acceptable_constr_viol_tol);
    };

    for i in 0..m {
        let v = g[i];
        if !v.is_finite() {
            return None;
        }
        let lo = present(g_l[i], true);
        let hi = present(g_u[i], false);
        let scale = v
            .abs()
            .max(lo.map_or(0.0, Number::abs))
            .max(hi.map_or(0.0, Number::abs));
        judge(lo.map_or(0.0, |b| b - v), scale);
        judge(hi.map_or(0.0, |b| v - b), scale);
    }
    for j in 0..n {
        let v = x[j];
        if !v.is_finite() {
            return None;
        }
        let lo = present(x_l[j], true);
        let hi = present(x_u[j], false);
        let scale = v
            .abs()
            .max(lo.map_or(0.0, Number::abs))
            .max(hi.map_or(0.0, Number::abs));
        judge(lo.map_or(0.0, |b| b - v), scale);
        judge(hi.map_or(0.0, |b| v - b), scale);
    }

    Some(OriginalSpaceFeasibility {
        max_violation,
        negligible_at_tol: ok_tol,
        negligible_at_acceptable: ok_acceptable,
    })
}

/// Map upstream `SolverReturn` codes to `ApplicationReturnStatus`.
/// Mirrors the table in
/// `ref/Ipopt/AGENT_REFERENCE/MAIN_LOOP.md` ("exception → SolverReturn
/// map") and the corresponding switch in
/// `IpIpoptApplication.cpp:call_optimize`.
fn solver_return_to_app_status(s: SolverReturn) -> ApplicationReturnStatus {
    match s {
        SolverReturn::Success => ApplicationReturnStatus::SolveSucceeded,
        SolverReturn::StopAtAcceptablePoint => ApplicationReturnStatus::SolvedToAcceptableLevel,
        SolverReturn::FeasiblePointFound => ApplicationReturnStatus::FeasiblePointFound,
        SolverReturn::MaxiterExceeded => ApplicationReturnStatus::MaximumIterationsExceeded,
        SolverReturn::CpuTimeExceeded => ApplicationReturnStatus::MaximumCpuTimeExceeded,
        SolverReturn::WallTimeExceeded => ApplicationReturnStatus::MaximumWallTimeExceeded,
        SolverReturn::StopAtTinyStep => ApplicationReturnStatus::SearchDirectionBecomesTooSmall,
        SolverReturn::LocalInfeasibility => ApplicationReturnStatus::InfeasibleProblemDetected,
        SolverReturn::UserRequestedStop => ApplicationReturnStatus::UserRequestedStop,
        SolverReturn::DivergingIterates => ApplicationReturnStatus::DivergingIterates,
        SolverReturn::RestorationFailure => ApplicationReturnStatus::RestorationFailed,
        SolverReturn::ErrorInStepComputation => ApplicationReturnStatus::ErrorInStepComputation,
        SolverReturn::InvalidNumberDetected => ApplicationReturnStatus::InvalidNumberDetected,
        SolverReturn::TooFewDegreesOfFreedom => ApplicationReturnStatus::NotEnoughDegreesOfFreedom,
        SolverReturn::InvalidProblemDefinition => ApplicationReturnStatus::InvalidProblemDefinition,
        SolverReturn::InvalidOption => ApplicationReturnStatus::InvalidOption,
        SolverReturn::OutOfMemory => ApplicationReturnStatus::InsufficientMemory,
        SolverReturn::InternalError | SolverReturn::Unassigned => {
            ApplicationReturnStatus::InternalError
        }
    }
}

/// Best-effort evaluation of the objective at the algorithm's final
/// `x`. Returns the *scaled* objective (`f * obj_scale_factor`); used
/// to populate `SolveStatistics::final_scaled_objective`.
fn try_eval_curr_f(
    nlp: &Rc<RefCell<dyn IpoptNlp>>,
    x: &Rc<dyn pounce_linalg::Vector>,
) -> Result<Number, ()> {
    let mut nlp_mut = nlp.borrow_mut();
    Ok(nlp_mut.eval_f(&**x))
}

/// Trigger predicate for the Phase-3.5 ℓ₁ auto-fallback path. Returns
/// `true` when a status warrants a retry through the wrapper. Mirrors
/// ripopt#23's trigger set, extended per the audit's Refinement B
/// (pounce-side `Not_Enough_Degrees_Of_Freedom` is added because
/// pounce's DOF early-exit blocks NE-suffix problems that ripopt's
/// equivalent would let pass to the wrapper).
fn is_l1_fallback_trigger(status: ApplicationReturnStatus) -> bool {
    matches!(
        status,
        ApplicationReturnStatus::RestorationFailed
            | ApplicationReturnStatus::InfeasibleProblemDetected
            | ApplicationReturnStatus::SolvedToAcceptableLevel
            | ApplicationReturnStatus::MaximumIterationsExceeded
            | ApplicationReturnStatus::NotEnoughDegreesOfFreedom
    )
}

/// Forward the final iterate back to the user's `TNLP::finalize_solution`.
/// We pull `x` (compressed in `x_var`-space) off the algorithm's
/// `data.curr`, lift it back to full TNLP indexing, and pass empty
/// multipliers for now (the algorithm's `y_c`, `y_d`, `z_l`, `z_u` are
/// in compressed split form — re-assembling them into the user's
/// `lambda` / `z_l` / `z_u` is mechanical but lives behind a
/// `OrigIpoptNlp::finalize_solution_*` accessor that's still being
/// fleshed out). On success returns the unscaled objective evaluated
/// on the user TNLP at the final iterate; returns `Err` if the final
/// iterate is missing.
/// Read a `dyn Vector`'s entries. Empty for a non-dense backing; POUNCE is
/// dense-only, so that is defensive rather than a supported case.
fn dense_values(v: &dyn pounce_linalg::Vector) -> Vec<Number> {
    v.as_any()
        .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
        .map(|d| d.expanded_values())
        .unwrap_or_default()
}

/// Overwrite a `dyn Vector`'s entries. Returns false — writing nothing —
/// when the backing is not dense or the lengths disagree, so a caller
/// updating several components together can abandon the whole update rather
/// than leave a half-written iterate.
fn set_dense(v: &mut dyn pounce_linalg::Vector, vals: &[Number]) -> bool {
    match v
        .as_any_mut()
        .downcast_mut::<pounce_linalg::dense_vector::DenseVector>()
    {
        Some(d) if pounce_linalg::Vector::dim(d) as usize == vals.len() => {
            d.set_values(vals);
            true
        }
        _ => false,
    }
}

/// An owned copy of a [`Solution`] payload already delivered to the user's
/// TNLP, enough to deliver it again.
///
/// Exists because a losing second-opinion retry has to be undoable. The status
/// a retry earns is already floored — `run_with_mu_strategy_fallback` returns
/// `first_status` unless the retry promotes — but the *point* was not, and the
/// user consumes the point. See that function for what went wrong without this.
#[derive(Debug, Clone)]
struct FinalizeSnapshot {
    status: SolverReturn,
    x: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    g: Vec<Number>,
    lambda: Vec<Number>,
    obj_value: Number,
}

/// The `final_*` half of [`SolveStatistics`] — the numbers that describe the
/// answer and the certificate attached to it.
///
/// Deliberately **not** the whole struct. `SolveStatistics` mixes two kinds of
/// number and they float back differently when a second-opinion retry loses:
///
/// * the `final_*` fields describe *the point being reported*, so they have to
///   agree with the status reported beside them. A `Solved_To_Acceptable_Level`
///   carrying a `final_kkt_error` two orders above `acceptable_tol` is
///   self-contradictory, and that is pounce#870.
/// * `iteration_count`, the evaluation counts, the timers, the restoration
///   tallies and `quality_escalations` describe *what the invocation did*.
///   Both attempts really ran, so rewinding those would under-report the work
///   actually spent — a different falsehood, not a fix. `deb7` at
///   `max_iter=100` is the case that caught this: rewinding the counter made
///   the run claim an iteration count belonging to only one of its two solves
///   (`issue857_escalation_gated_quality_rung.rs`).
///
/// So the certificate is floored and the cost is not.
#[derive(Debug, Clone, Copy)]
struct SolutionCertificate {
    objective: Number,
    scaled_objective: Number,
    dual_inf: Number,
    constr_viol: Number,
    compl: Number,
    kkt_error: Number,
    unscaled_dual_inf: Number,
    unscaled_constr_viol: Number,
    unscaled_compl: Number,
    unscaled_kkt_error: Number,
    kkt_error_above_noise: Number,
    mu: Number,
}

impl SolutionCertificate {
    fn of(s: &pounce_nlp::solve_statistics::SolveStatistics) -> Self {
        Self {
            objective: s.final_objective,
            scaled_objective: s.final_scaled_objective,
            dual_inf: s.final_dual_inf,
            constr_viol: s.final_constr_viol,
            compl: s.final_compl,
            kkt_error: s.final_kkt_error,
            unscaled_dual_inf: s.final_unscaled_dual_inf,
            unscaled_constr_viol: s.final_unscaled_constr_viol,
            unscaled_compl: s.final_unscaled_compl,
            unscaled_kkt_error: s.final_unscaled_kkt_error,
            kkt_error_above_noise: s.final_kkt_error_above_noise,
            mu: s.final_mu,
        }
    }

    fn restore_into(&self, s: &mut pounce_nlp::solve_statistics::SolveStatistics) {
        s.final_objective = self.objective;
        s.final_scaled_objective = self.scaled_objective;
        s.final_dual_inf = self.dual_inf;
        s.final_constr_viol = self.constr_viol;
        s.final_compl = self.compl;
        s.final_kkt_error = self.kkt_error;
        s.final_unscaled_dual_inf = self.unscaled_dual_inf;
        s.final_unscaled_constr_viol = self.unscaled_constr_viol;
        s.final_unscaled_compl = self.unscaled_compl;
        s.final_unscaled_kkt_error = self.unscaled_kkt_error;
        s.final_kkt_error_above_noise = self.kkt_error_above_noise;
        s.final_mu = self.mu;
    }
}

impl FinalizeSnapshot {
    /// Re-deliver this payload to `tnlp`, overwriting whatever a later attempt
    /// captured there.
    fn replay(&self, tnlp: &Rc<RefCell<dyn TNLP>>) {
        tnlp.borrow_mut().finalize_solution(
            Solution {
                status: self.status,
                x: &self.x,
                z_l: &self.z_l,
                z_u: &self.z_u,
                g: &self.g,
                lambda: &self.lambda,
                obj_value: self.obj_value,
            },
            &TnlpIpoptData::default(),
            &TnlpIpoptCq::default(),
        );
    }
}

fn finalize_via_orig_nlp(
    nlp: &Rc<RefCell<dyn IpoptNlp>>,
    alg: &IpoptAlgorithm,
    solver_status: SolverReturn,
    _app_status: ApplicationReturnStatus,
    tnlp: &Rc<RefCell<dyn TNLP>>,
    sink: &RefCell<Option<FinalizeSnapshot>>,
) -> Result<Number, ()> {
    let curr = alg.data.borrow().curr.clone().ok_or(())?;
    // Lift compressed x_var → full-x (length `info.n`) so the user
    // TNLP receives the same shape it provided. With `make_parameter`
    // the fixed components are spliced back in by the IpoptNlp.
    let nlp_borrow = nlp.borrow();
    // `finalize_solution_x`, not `lift_x_to_full`: the reported point also
    // owes the user the `honor_original_bounds` projection. `f` and `g`
    // below are then evaluated at the point actually reported, so x/f/g
    // agree with each other.
    let x_vec: Vec<Number> = nlp_borrow.finalize_solution_x(&*curr.x);
    let info = tnlp.borrow_mut().get_nlp_info().ok_or(())?;
    let n = info.n as usize;
    let m = info.m as usize;
    debug_assert_eq!(x_vec.len(), n);
    // Lift algorithm-side multipliers back into user-space (pounce#11).
    // Use the `finalize_solution_*` family (not the `pack_*` family): the
    // final solution duals must be reported in the user's *unscaled-
    // Lagrangian* convention `∇f + λ·∇g + z = 0`, which divides out the
    // `obj_scale_factor` the algorithm threads through `eval_h`. The `pack_*`
    // family deliberately omits that division because it feeds the scaled
    // `eval_h`; calling it here left every dual scaled by `obj_scale_factor`
    // whenever gradient-based scaling triggered (pounce#11 F1).
    // Backends without overrides return empty; fall back to zero stubs so the
    // user sees a length-consistent vector.
    let mut z_l = nlp_borrow.finalize_solution_z_l(&*curr.z_l);
    if z_l.is_empty() {
        z_l = vec![0.0; n];
    }
    let mut z_u = nlp_borrow.finalize_solution_z_u(&*curr.z_u);
    if z_u.is_empty() {
        z_u = vec![0.0; n];
    }
    let mut lambda = nlp_borrow.finalize_solution_lambda(&*curr.y_c, &*curr.y_d);
    if lambda.is_empty() {
        lambda = vec![0.0; m];
    }
    drop(nlp_borrow);
    // Compute g(x) via the user TNLP so the final residual is
    // populated for the user.
    let mut g_final = vec![0.0; m];
    let _ = tnlp.borrow_mut().eval_g(&x_vec, true, &mut g_final);
    let f_final = tnlp
        .borrow_mut()
        .eval_f(&x_vec, true)
        .unwrap_or(Number::NAN);
    let snap = FinalizeSnapshot {
        status: solver_status,
        x: x_vec,
        z_l,
        z_u,
        g: g_final,
        lambda,
        obj_value: f_final,
    };
    snap.replay(tnlp);
    *sink.borrow_mut() = Some(snap);
    Ok(f_final)
}

/// Bind SQP suboptions registered in `upstream_options.rs`
/// (`sqp_globalization`, `sqp_hessian`, `sqp_max_iter`, `sqp_tol`,
/// `sqp_constr_viol_tol`, `sqp_dual_inf_tol`, `sqp_l1_penalty`,
/// `sqp_bt_reduction`, `sqp_bt_min_alpha`, `sqp_print_level`,
/// `sqp_lbfgs_max_history`) onto
/// `opts`. Used by [`IpoptApplication::algorithm_builder_snapshot`]
/// before constructing an SQP algorithm.
fn apply_sqp_options(options: &OptionsList, opts: &mut crate::sqp::SqpOptions) {
    use crate::sqp::{SqpGlobalization, SqpHessianSource};

    if let Ok((s, true)) = options.get_string_value("sqp_globalization", "") {
        opts.globalization = match s.as_str() {
            "filter" => SqpGlobalization::Filter,
            "l1-elastic" => SqpGlobalization::L1Elastic,
            _ => opts.globalization,
        };
    }
    // `hessian_approximation` is the upstream Ipopt option a frontend sets
    // when the caller supplies no second derivatives -- `pounce.minimize` does
    // it automatically, and warns that it is doing so. It was only ever read
    // on the IPM path, so an SQP solve ignored it and fell back to the
    // `Exact` default, asking the NLP for a Lagrangian Hessian that was never
    // provided. A zero Hessian turns the QP subproblem into an LP, which is
    // unbounded whenever the objective gradient has a component in the null
    // space of the active constraints -- so the solve died with
    // `Internal_Error` on problems the IPM handles without complaint:
    //
    //     min (x0-3)^2 + (x1-2)^2  s.t.  4 - x0 - x1 >= 0
    //
    // (IPM: x = [2.5, 1.5]. Active-set SQP before this: Internal_Error, or
    // with variable bounds, a run to the box corner along the null-space
    // direction.)
    //
    // The quasi-Newton source picked here is the *dense Powell-damped BFGS*,
    // not the limited-memory one, even though the requesting option is spelled
    // `limited-memory`. On this active-set-SQP path L-BFGS buys nothing: its
    // `as_triplet` materializes a full dense `n×n` Hessian for the QP
    // subproblem exactly as `DampedBfgs` does (the matrix-free product
    // interface that would make L-BFGS cheaper is not implemented yet), and it
    // is markedly less robust -- it stalls with
    // `Search_Direction_Becomes_Too_Small` (or reports the QP subproblem
    // `unbounded`) on easy, well-conditioned convex QPs whenever a general
    // inequality is active at the optimum, returning `success=False` with a
    // wrong `x` (issue #358). `DampedBfgs` solves those. So the automatic
    // approximation the facade injects when no analytic Hessian is available
    // maps to the robust dense update; a caller who genuinely wants
    // limited-memory storage can still request it explicitly with
    // `sqp_hessian = "lbfgs"` below (read after this, so it wins).
    //
    // Read this before `sqp_hessian` so an explicit setting still wins.
    if let Ok((s, true)) = options.get_string_value("hessian_approximation", "") {
        if s == "limited-memory" {
            opts.hessian = SqpHessianSource::DampedBfgs;
        }
    }
    if let Ok((s, true)) = options.get_string_value("sqp_hessian", "") {
        opts.hessian = match s.as_str() {
            "exact" => SqpHessianSource::Exact,
            "damped-bfgs" => SqpHessianSource::DampedBfgs,
            "lbfgs" => SqpHessianSource::Lbfgs,
            _ => opts.hessian,
        };
    }
    if let Ok((v, true)) = options.get_integer_value("sqp_max_iter", "") {
        if v >= 0 {
            opts.max_iter = v as u32;
        }
    }
    if let Ok((v, true)) = options.get_numeric_value("sqp_tol", "") {
        opts.tol = v;
    }
    if let Ok((v, true)) = options.get_numeric_value("sqp_constr_viol_tol", "") {
        opts.constr_viol_tol = v;
    }
    if let Ok((v, true)) = options.get_numeric_value("sqp_dual_inf_tol", "") {
        opts.dual_inf_tol = v;
    }
    if let Ok((v, true)) = options.get_numeric_value("sqp_l1_penalty", "") {
        opts.l1_penalty = v;
    }
    if let Ok((v, true)) = options.get_numeric_value("sqp_l1_penalty_safety", "") {
        opts.l1_penalty_safety = v;
    }
    if let Ok((v, true)) = options.get_numeric_value("sqp_l1_penalty_max", "") {
        opts.l1_penalty_max = v;
    }
    if let Ok((v, true)) = options.get_numeric_value("sqp_bt_reduction", "") {
        opts.bt_reduction = v;
    }
    if let Ok((v, true)) = options.get_numeric_value("sqp_bt_min_alpha", "") {
        opts.bt_min_alpha = v;
    }
    if let Ok((v, true)) = options.get_integer_value("sqp_print_level", "") {
        opts.print_level = v.clamp(0, u8::MAX as i32) as u8;
    }
    if let Ok((v, true)) = options.get_integer_value("sqp_lbfgs_max_history", "") {
        if v >= 1 {
            opts.lbfgs_max_history = v as u32;
        }
    }
}

/// Populate the active-set SQP **QP-subproblem** options
/// ([`pounce_qp::QpOptions`]) from the `sqp_qp_*` option family.
///
/// Sister to [`apply_sqp_options`], which handles the SQP *outer-loop*
/// options ([`crate::sqp::SqpOptions`]); this one feeds the inner QP
/// solver that `SqpAlgorithm` delegates each subproblem to. Consulted
/// only on the `ActiveSetSqp` path. Each knob is forwarded only when
/// the user explicitly set it, so the `pounce_qp` defaults stand
/// otherwise.
///
/// The reading itself is [`pounce_qp::ActiveSetOverrides`], shared with
/// `pounce_convex`'s direct active-set driver, which overlays the same
/// eight names onto the same `QpOptions` type. This function had its own
/// copy until then, and the two had drifted: this one silently ignored a
/// `sqp_qp_max_iter` of 0 and an unknown `sqp_qp_anti_cycling` value where
/// the other rejected them. Neither divergence was reachable — the
/// registry bounds `sqp_qp_max_iter` at 1 and restricts `anti_cycling` to
/// three values — but two readers of one option family is how a
/// reachable one starts.
fn apply_qp_subproblem_options(options: &OptionsList, opts: &mut pounce_qp::QpOptions) {
    match pounce_qp::ActiveSetOverrides::try_from_options_list(options) {
        Ok(overrides) => overrides.apply(opts),
        // Unreachable from here: this runs after `initialize()`, so every
        // value present has already been validated against the registered
        // bound that the reader re-checks. Say so out loud anyway rather
        // than solving with a configuration the user did not ask for —
        // silently dropping the whole family is exactly the failure mode
        // `tests/no_silent_options.rs` exists to prevent.
        Err(error) => tracing::error!(
            target: "pounce::options",
            %error,
            "sqp_qp_* options were rejected after the registry accepted them; \
             the QP subproblem is running on pounce-qp defaults"
        ),
    }
}

/// SQP-side analog of [`finalize_via_orig_nlp`]. Hands the SQP
/// solution iterate to the user TNLP via the standard
/// `finalize_solution` callback. Multiplier lifting goes through
/// the same OrigIpoptNlp hooks so the user sees the same shape
/// regardless of which algorithm produced the iterate.
///
/// Returns the user-space objective value on success.
fn finalize_via_sqp(
    nlp: &Rc<RefCell<dyn IpoptNlp>>,
    res: &crate::sqp::SqpResult,
    solver_status: pounce_nlp::SolverReturn,
    tnlp: &Rc<RefCell<dyn TNLP>>,
    sink: &RefCell<Option<FinalizeSnapshot>>,
) -> Result<Number, ()> {
    use pounce_linalg::dense_vector::DenseVectorSpace;

    let info = tnlp.borrow_mut().get_nlp_info().ok_or(())?;
    let n = info.n as usize;
    let m = info.m as usize;

    // Wrap SQP slices in DenseVectors so we can pass them through
    // the OrigIpoptNlp lift_x_to_full / pack_*_for_user hooks.
    let nlp_borrow = nlp.borrow();
    let n_alg = nlp_borrow.n() as usize;
    let m_eq = nlp_borrow.m_eq() as usize;
    let m_ineq = nlp_borrow.m_ineq() as usize;
    debug_assert_eq!(res.x.len(), n_alg);
    debug_assert_eq!(res.lambda_g.len(), m_eq + m_ineq);
    debug_assert_eq!(res.lambda_x.len(), n_alg);

    let x_space = DenseVectorSpace::new(n_alg as Index);
    let c_space = DenseVectorSpace::new(m_eq as Index);
    let d_space = DenseVectorSpace::new(m_ineq as Index);

    let mut x_dv = x_space.make_new_dense();
    x_dv.set_values(&res.x);
    let x_vec: Vec<Number> = nlp_borrow.finalize_solution_x(&x_dv);
    debug_assert_eq!(x_vec.len(), n);

    // λ_x is packed signed (z_l − z_u). Split for lift.
    let mut z_l_compressed = x_space.make_new_dense();
    let mut z_u_compressed = x_space.make_new_dense();
    let zl_vals: Vec<Number> = res.lambda_x.iter().map(|v| v.max(0.0)).collect();
    let zu_vals: Vec<Number> = res.lambda_x.iter().map(|v| (-v).max(0.0)).collect();
    z_l_compressed.set_values(&zl_vals);
    z_u_compressed.set_values(&zu_vals);
    // `finalize_solution_*` (not `pack_*`): report unscaled-Lagrangian duals,
    // dividing out `obj_scale_factor` — see `finalize_via_orig_nlp` (F1).
    let mut z_l = nlp_borrow.finalize_solution_z_l(&z_l_compressed);
    if z_l.is_empty() {
        z_l = vec![0.0; n];
    }
    let mut z_u = nlp_borrow.finalize_solution_z_u(&z_u_compressed);
    if z_u.is_empty() {
        z_u = vec![0.0; n];
    }

    // λ_g is [y_c; y_d]; split into the c/d blocks for lift.
    let mut y_c_dv = c_space.make_new_dense();
    let mut y_d_dv = d_space.make_new_dense();
    if m_eq > 0 {
        y_c_dv.set_values(&res.lambda_g[..m_eq]);
    }
    if m_ineq > 0 {
        y_d_dv.set_values(&res.lambda_g[m_eq..]);
    }
    let mut lambda = nlp_borrow.finalize_solution_lambda(&y_c_dv, &y_d_dv);
    if lambda.is_empty() {
        lambda = vec![0.0; m];
    }
    drop(nlp_borrow);

    let mut g_final = vec![0.0; m];
    let _ = tnlp.borrow_mut().eval_g(&x_vec, true, &mut g_final);
    let f_final = tnlp
        .borrow_mut()
        .eval_f(&x_vec, true)
        .unwrap_or(Number::NAN);
    let snap = FinalizeSnapshot {
        status: solver_status,
        x: x_vec,
        z_l,
        z_u,
        g: g_final,
        lambda,
        obj_value: f_final,
    };
    snap.replay(tnlp);
    *sink.borrow_mut() = Some(snap);
    Ok(f_final)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// pounce#748 — the flipped default is conditional on the caller not
    /// having named a `mu_strategy`. All four combinations, because the
    /// point of the condition is that an explicit strategy suppresses the
    /// automatic retry while an explicit `mu_strategy_fallback` does not.
    #[test]
    fn mu_strategy_fallback_default_defers_to_an_explicit_strategy() {
        // Nothing set: the retry is on.
        let app = IpoptApplication::new();
        assert!(app.is_mu_strategy_fallback_enabled());

        // Caller named a strategy: the automatic retry stands down.
        let mut app = IpoptApplication::new();
        app.options_mut()
            .set_string_value("mu_strategy", "monotone", true, false)
            .unwrap();
        assert!(!app.is_mu_strategy_fallback_enabled());

        // ... unless they also asked for the retry explicitly.
        app.options_mut()
            .set_string_value("mu_strategy_fallback", "yes", true, false)
            .unwrap();
        assert!(app.is_mu_strategy_fallback_enabled());

        // An explicit "no" is honoured with no strategy set.
        let mut app = IpoptApplication::new();
        app.options_mut()
            .set_string_value("mu_strategy_fallback", "no", true, false)
            .unwrap();
        assert!(!app.is_mu_strategy_fallback_enabled());
    }

    use pounce_nlp::tnlp::{
        BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, ScalingRequest, Solution,
        SparsityRequest, StartingPoint,
    };

    struct Hs071Stub;
    impl TNLP for Hs071Stub {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            // HS071 dimensions: n=4, m=2, dense Jacobian (8 nz),
            // dense lower-triangular Hessian (10 nz).
            Some(NlpInfo {
                n: 4,
                m: 2,
                nnz_jac_g: 8,
                nnz_h_lag: 10,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[1.0; 4]);
            b.x_u.copy_from_slice(&[5.0; 4]);
            b.g_l.copy_from_slice(&[25.0, 40.0]);
            b.g_u.copy_from_slice(&[2.0e19, 40.0]);
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x.copy_from_slice(&[1.0, 5.0, 5.0, 1.0]);
            true
        }
        fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
            Some(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
            grad.fill(0.0);
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            if let SparsityRequest::Structure { irow, jcol } = mode {
                irow.copy_from_slice(&[0, 0, 0, 0, 1, 1, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 2, 3, 0, 1, 2, 3]);
            }
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    #[test]
    fn application_default_does_not_select_sqp() {
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        assert!(!app.is_sqp_algorithm_selected());
    }

    #[test]
    fn application_routes_to_sqp_when_algorithm_option_set() {
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str("algorithm active-set-sqp\n")
            .unwrap();
        assert!(app.is_sqp_algorithm_selected());
    }

    #[test]
    fn feral_min_par_flops_option_reaches_config() {
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        // Unset on the OptionsList: falls through to FeralConfig::from_env,
        // which leaves it None (inherit feral's built-in default) when the
        // POUNCE_FERAL_MIN_PAR_FLOPS env var is also absent.
        assert_eq!(
            feral_config_from_options(app.options()).min_par_flops,
            None,
            "unset feral_min_par_flops should not force an override"
        );
        // Explicit set is mapped through, cast to u64. `0` is the "dispatch
        // on every eligible tree" setting and must survive the cast.
        app.initialize_with_options_str("feral_min_par_flops 0\n")
            .unwrap();
        assert_eq!(
            feral_config_from_options(app.options()).min_par_flops,
            Some(0)
        );
        // A large finite value passes through intact (5e8 > i32::MAX, which
        // is why this is a number option, not an integer one).
        app.initialize_with_options_str("feral_min_par_flops 5e8\n")
            .unwrap();
        assert_eq!(
            feral_config_from_options(app.options()).min_par_flops,
            Some(500_000_000)
        );
    }

    #[test]
    fn feral_refine_steps_option_reaches_config() {
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        // Unset: falls through to FeralConfig::from_env, which resolves to
        // feral's own DEFAULT_REFINE_MAX_STEPS. The bump to feral 0.17.0 is
        // deliberately behaviour-preserving here — the cap that gh#710 wants
        // to measure (1) is not yet the default.
        assert_eq!(
            feral_config_from_options(app.options()).refine_max_steps,
            pounce_feral::FeralConfig::default().refine_max_steps,
            "unset feral_refine_steps must keep feral's own default"
        );
        // The setting gh#710 exists to evaluate.
        app.initialize_with_options_str("feral_refine_steps 1\n")
            .unwrap();
        assert_eq!(feral_config_from_options(app.options()).refine_max_steps, 1);
        // `0` is a legal cap (zero corrections, refined entry point still
        // taken) and must not be read as "unset".
        app.initialize_with_options_str("feral_refine_steps 0\n")
            .unwrap();
        assert_eq!(feral_config_from_options(app.options()).refine_max_steps, 0);
    }

    #[test]
    fn feral_static_pivoting_option_reaches_config() {
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        // Unset on the OptionsList: falls through to FeralConfig::from_env,
        // which leaves it None (inherit feral's delayed-pivot default) when
        // the POUNCE_FERAL_STATIC_PIVOTING env var is also absent — so the
        // default numeric path is unchanged.
        assert_eq!(
            feral_config_from_options(app.options()).static_pivoting,
            None,
            "unset feral_static_pivoting must not force a numeric override"
        );
        // Explicit `yes` maps to Some(true): every supernode factors with
        // delayed pivoting disabled (feral#8 cascade breaker).
        app.initialize_with_options_str("feral_static_pivoting yes\n")
            .unwrap();
        assert_eq!(
            feral_config_from_options(app.options()).static_pivoting,
            Some(true)
        );
        // Explicit `no` maps to Some(false): keep delayed pivoting on
        // (distinct from unset, which merely inherits the default).
        app.initialize_with_options_str("feral_static_pivoting no\n")
            .unwrap();
        assert_eq!(
            feral_config_from_options(app.options()).static_pivoting,
            Some(false)
        );
    }

    /// Convex equality NLP fixture for end-to-end SQP testing
    /// through `IpoptApplication`:
    ///
    ///     min ½(x₁² + x₂²) − x₁ − 2x₂  s.t.  x₁ + x₂ = 1
    ///
    /// Closed form: x* = (0, 1), obj = -1.5, λ_g = 1.
    struct ConvexEqTnlp {
        finalize_called: std::rc::Rc<std::cell::RefCell<Option<(Vec<Number>, Number)>>>,
    }
    impl TNLP for ConvexEqTnlp {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 2,
                m: 1,
                nnz_jac_g: 2,
                nnz_h_lag: 2,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[-2.0e19; 2]);
            b.x_u.copy_from_slice(&[2.0e19; 2]);
            b.g_l.copy_from_slice(&[1.0]);
            b.g_u.copy_from_slice(&[1.0]);
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x.copy_from_slice(&[0.0, 0.0]);
            true
        }
        fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
            Some(0.5 * (x[0] * x[0] + x[1] * x[1]) - x[0] - 2.0 * x[1])
        }
        fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
            grad[0] = x[0] - 1.0;
            grad[1] = x[1] - 2.0;
            true
        }
        fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g[0] = x[0] + x[1];
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow.copy_from_slice(&[0, 0]);
                    jcol.copy_from_slice(&[0, 1]);
                }
                SparsityRequest::Values { values, .. } => {
                    values.copy_from_slice(&[1.0, 1.0]);
                }
            }
            true
        }
        fn eval_h(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _obj_factor: Number,
            _lambda: Option<&[Number]>,
            _new_lambda: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow.copy_from_slice(&[0, 1]);
                    jcol.copy_from_slice(&[0, 1]);
                }
                SparsityRequest::Values { values, .. } => {
                    values.copy_from_slice(&[1.0, 1.0]);
                }
            }
            true
        }
        fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
            *self.finalize_called.borrow_mut() = Some((sol.x.to_vec(), sol.obj_value));
        }
    }

    /// A TNLP that solves normally once, then declines to supply bounds.
    ///
    /// The second `optimize_constrained` therefore bails long before the
    /// statistics block that records `row_scaling_active`, which is the
    /// path the fail-closed reset exists for.
    struct DescribesItselfOnce {
        /// Flipped by the test between the two solves. A call counter
        /// would not work: these hooks are called more than once per
        /// solve, so it would trip inside the first one.
        refuse: std::rc::Rc<std::cell::Cell<bool>>,
        inner: ExactQuadratic,
    }
    impl TNLP for DescribesItselfOnce {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            self.inner.get_nlp_info()
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            // Declining here is a documented TNLP outcome and unwinds
            // cleanly; declining `get_nlp_info` mid-flight panics instead,
            // which would test the panic path rather than the reset.
            if self.refuse.get() {
                return false;
            }
            self.inner.get_bounds_info(b)
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            self.inner.get_starting_point(sp)
        }
        fn eval_f(&mut self, x: &[Number], n: bool) -> Option<Number> {
            self.inner.eval_f(x, n)
        }
        fn eval_grad_f(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            self.inner.eval_grad_f(x, n, g)
        }
        fn eval_g(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            self.inner.eval_g(x, n, g)
        }
        fn eval_jac_g(&mut self, x: Option<&[Number]>, n: bool, mode: SparsityRequest<'_>) -> bool {
            self.inner.eval_jac_g(x, n, mode)
        }
        fn eval_h(
            &mut self,
            x: Option<&[Number]>,
            n: bool,
            o: Number,
            l: Option<&[Number]>,
            nl: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            self.inner.eval_h(x, n, o, l, nl, mode)
        }
        fn finalize_solution(&mut self, s: Solution<'_>, d: &IpoptData, c: &IpoptCq) {
            self.inner.finalize_solution(s, d, c)
        }
    }

    /// gh#794 review round 2: `row_scaling_active` must be fail-closed.
    ///
    /// It is written near the end of `optimize_constrained`, from the NLP
    /// that solve built. A solve that bails before that point used to
    /// leave the *previous* solve's value in place — and the ℓ₁ outer
    /// loop calls `optimize_constrained` repeatedly and reads the flag
    /// after each call, so a stale `Some(false)` would let it mirror an
    /// original-units violation into the scaled family. That is exactly
    /// the units contract the flag was added to protect.
    ///
    /// The two solves here are the shape that matters: the first records
    /// a verdict, the second never gets far enough to record one.
    /// Removing the reset at the top of `optimize_constrained` makes this
    /// fail with `Some(false)` where `None` is required — checked, not
    /// assumed.
    #[test]
    fn row_scaling_active_is_cleared_when_a_later_solve_bails_early() {
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();

        // Solve 1: reaches the statistics block and records a verdict.
        // `ExactQuadratic` supplies `eval_h`, so the solve stays on the
        // exact-Hessian NLP path that writes this flag; a fixture without
        // one lands on L-BFGS instead.
        let refuse = std::rc::Rc::new(std::cell::Cell::new(false));
        let first = std::rc::Rc::new(std::cell::RefCell::new(DescribesItselfOnce {
            refuse: std::rc::Rc::clone(&refuse),
            inner: ExactQuadratic,
        }));
        let _ = app
            .optimize_tnlp(std::rc::Rc::clone(&first) as std::rc::Rc<std::cell::RefCell<dyn TNLP>>);
        assert!(
            app.row_scaling_active.get().is_some(),
            "the first solve did not record a row-scaling verdict, so this \
             test cannot show the second one clearing it",
        );

        // Solve 2, same application: bails before recording anything.
        refuse.set(true);
        let _ = app.optimize_tnlp(first as std::rc::Rc<std::cell::RefCell<dyn TNLP>>);

        assert_eq!(
            app.row_scaling_active.get(),
            None,
            "a solve that bailed before recording row scaling left the \
             previous solve's verdict in place; the ℓ₁ outer loop would \
             read it as fact and mirror an original-units violation into \
             the scaled family (gh#794 review round 2)",
        );
    }

    #[test]
    fn application_sqp_path_solves_convex_eq_nlp_and_finalizes() {
        let finalize_slot = std::rc::Rc::new(std::cell::RefCell::new(None));
        let tnlp = std::rc::Rc::new(std::cell::RefCell::new(ConvexEqTnlp {
            finalize_called: std::rc::Rc::clone(&finalize_slot),
        }));

        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str("algorithm active-set-sqp\n")
            .unwrap();
        let status = app.optimize_tnlp(tnlp);
        assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);

        // The TNLP's finalize_solution must have been invoked.
        let recv = finalize_slot.borrow().clone();
        let (x_recv, obj_recv) = recv.expect("finalize_solution was not called");
        assert_eq!(x_recv.len(), 2);
        assert!((x_recv[0] - 0.0).abs() < 1e-6, "x[0] = {}", x_recv[0]);
        assert!((x_recv[1] - 1.0).abs() < 1e-6, "x[1] = {}", x_recv[1]);
        assert!(
            (obj_recv - (-1.5)).abs() < 1e-6,
            "obj = {} but expected -1.5",
            obj_recv
        );
    }

    #[test]
    fn application_routes_to_sqp_case_insensitively() {
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str("algorithm Active-Set-SQP\n")
            .unwrap();
        // get_string_value may return the value as-stored (no
        // normalization); the dispatch must handle case
        // insensitively per the c11 design choice.
        assert!(app.is_sqp_algorithm_selected());
    }

    #[test]
    fn application_constructs_and_loads_options() {
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        // ipopt.opt-style file: an integer-typed option registered by
        // the Interfaces layer.
        app.initialize_with_options_str("print_level 5\nfile_print_level 7\n")
            .unwrap();
        let (level, found) = app.options().get_integer_value("print_level", "").unwrap();
        assert!(found);
        assert_eq!(level, 5);
    }

    #[test]
    fn application_sqp_suboptions_propagate_to_builder() {
        // All SQP suboptions are read by algorithm_builder_snapshot
        // and baked into the builder's `sqp` field.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str(
            "algorithm active-set-sqp\n\
             sqp_globalization l1-elastic\n\
             sqp_hessian lbfgs\n\
             sqp_max_iter 17\n\
             sqp_tol 1e-7\n\
             sqp_constr_viol_tol 1e-5\n\
             sqp_dual_inf_tol 1e-3\n\
             sqp_l1_penalty 2.5\n\
             sqp_bt_reduction 0.25\n\
             sqp_bt_min_alpha 1e-10\n\
             sqp_print_level 2\n\
             sqp_lbfgs_max_history 12\n",
        )
        .unwrap();
        let snap = app.algorithm_builder_snapshot();
        assert_eq!(
            snap.sqp.globalization,
            crate::sqp::SqpGlobalization::L1Elastic
        );
        assert_eq!(snap.sqp.hessian, crate::sqp::SqpHessianSource::Lbfgs);
        assert_eq!(snap.sqp.max_iter, 17);
        assert!((snap.sqp.tol - 1e-7).abs() < 1e-18);
        assert!((snap.sqp.constr_viol_tol - 1e-5).abs() < 1e-18);
        assert!((snap.sqp.dual_inf_tol - 1e-3).abs() < 1e-18);
        assert!((snap.sqp.l1_penalty - 2.5).abs() < 1e-18);
        assert!((snap.sqp.bt_reduction - 0.25).abs() < 1e-18);
        assert!((snap.sqp.bt_min_alpha - 1e-10).abs() < 1e-18);
        assert_eq!(snap.sqp.print_level, 2);
        assert_eq!(snap.sqp.lbfgs_max_history, 12);
    }

    /// Every `sqp_qp_*` key that [`apply_qp_subproblem_options`] reads must
    /// actually be *registered*, and must reach `pounce_qp::QpOptions`.
    ///
    /// The whole family was readable-but-unregistered (gh #360): the options
    /// registry rejected each one with OPTION_INVALID, so the reader was
    /// unreachable and the documented knobs were unusable. This is the guard
    /// that class of omission needs — it fails both if a key stops being
    /// registered and if a newly-read key is never registered at all.
    #[test]
    fn application_sqp_qp_subproblem_options_are_registered_and_propagate() {
        use pounce_qp::AntiCyclingChoice;

        // Source of truth: the keys `apply_qp_subproblem_options` reads.
        // Kept in step with that function by the round-trip assertions below.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str(
            "algorithm active-set-sqp\n\
             sqp_qp_max_iter 37\n\
             sqp_qp_feas_tol 1e-7\n\
             sqp_qp_opt_tol 2e-7\n\
             sqp_qp_elastic_gamma 1e4\n\
             sqp_qp_anti_cycling bland\n\
             sqp_qp_use_schur_updates yes\n\
             sqp_qp_max_schur_updates_before_refactor 12\n\
             sqp_qp_use_homotopy yes\n\
             sqp_qp_certify_second_order yes\n",
        )
        .expect("every sqp_qp_* option must be registered (gh #360)");

        let qp = &app.algorithm_builder_snapshot().sqp_qp;
        assert_eq!(qp.max_iter, 37);
        assert!((qp.feas_tol - 1e-7).abs() < 1e-20);
        assert!((qp.opt_tol - 2e-7).abs() < 1e-20);
        assert!((qp.elastic_gamma - 1e4).abs() < 1e-9);
        assert_eq!(qp.anti_cycling, AntiCyclingChoice::Bland);
        // The Schur update path was implemented but reachable only through
        // `SqpAlgorithm::with_qp_options`, so no CLI/library user could turn
        // it on — the same unreachable-knob defect gh #360 fixed for the rest
        // of this family.
        assert!(qp.use_schur_updates);
        assert_eq!(qp.max_schur_updates_before_refactor, 12);
        assert!(qp.use_homotopy);
        // gh #848. Off by default on this path (see
        // `QpOptions::sqp_subproblem`), so the value that proves the wire is
        // live is `yes` — asserting the default would pass on a reader that
        // never ran.
        assert!(qp.certify_second_order);

        // Untouched options must keep the SQP subproblem base, not be
        // overwritten with zeros by the "explicitly set" gate. That base is
        // `QpOptions::default()` in every field but one; the exception is
        // asserted below.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str("algorithm active-set-sqp\n")
            .unwrap();
        let defaults = pounce_qp::QpOptions::default();
        let qp = &app.algorithm_builder_snapshot().sqp_qp;
        assert_eq!(qp.max_iter, defaults.max_iter);
        assert!((qp.feas_tol - defaults.feas_tol).abs() < 1e-20);
        assert!((qp.opt_tol - defaults.opt_tol).abs() < 1e-20);
        assert_eq!(qp.anti_cycling, defaults.anti_cycling);
        // Default stays OFF, and that is a measured choice: enabling it breaks
        // 9 of the 46 Maros-Meszaros instances the default path solves
        // correctly. Do not flip this without re-running that comparison.
        assert!(!qp.use_schur_updates);
        assert_eq!(
            qp.max_schur_updates_before_refactor,
            defaults.max_schur_updates_before_refactor
        );
        // Off by default *on the SQP path*, and deliberately different from
        // `QpOptions::default()` — which is what a standalone `solve_qp`
        // gets, and where it is on. gh #848 / gh #856.
        assert!(!qp.certify_second_order);
        assert!(pounce_qp::QpOptions::default().certify_second_order);
    }

    /// The other direction of the gh #360 guard: every **registered**
    /// `sqp_qp_*` option must be one `apply_qp_subproblem_options` actually
    /// reads.
    ///
    /// The sister test above checks read-keys-are-registered. It cannot catch
    /// the inverse, and the inverse happened: `sqp_qp_use_homotopy` was
    /// registered with the homotopy work and never wired into the reader, so
    /// setting it on the SQP path silently did nothing while the option's own
    /// documentation described what it would do. A registered knob that no
    /// code reads is worse than a missing one — it validates, it accepts a
    /// value, and it lies.
    ///
    /// Adding a new `sqp_qp_*` option therefore fails here until it is both
    /// read by `apply_qp_subproblem_options` and asserted in the round-trip
    /// test above.
    #[test]
    fn application_every_registered_sqp_qp_option_is_read_by_the_subproblem_reader() {
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();

        let mut registered: Vec<String> = app
            .registered_options()
            .registered_options_in_order()
            .iter()
            .map(|o| o.name.clone())
            .filter(|n| n.starts_with("sqp_qp_"))
            .collect();
        registered.sort();

        // Kept in step by hand with the `options.get_*_value("sqp_qp_…")`
        // calls in `apply_qp_subproblem_options`, and cross-checked by the
        // round-trip assertions in the sister test.
        let mut read_by_the_reader = vec![
            "sqp_qp_anti_cycling".to_string(),
            "sqp_qp_certify_second_order".to_string(),
            "sqp_qp_elastic_gamma".to_string(),
            "sqp_qp_feas_tol".to_string(),
            "sqp_qp_max_iter".to_string(),
            "sqp_qp_max_schur_updates_before_refactor".to_string(),
            "sqp_qp_opt_tol".to_string(),
            "sqp_qp_use_homotopy".to_string(),
            "sqp_qp_use_schur_updates".to_string(),
        ];
        read_by_the_reader.sort();

        assert_eq!(
            registered, read_by_the_reader,
            "registered sqp_qp_* options and the ones \
             `apply_qp_subproblem_options` reads have diverged. A key that is \
             registered but unread is a no-op knob with working documentation \
             (that is how `sqp_qp_use_homotopy` shipped); a key read but not \
             registered is gh #360. Wire it up in both places, assert it in \
             `application_sqp_qp_subproblem_options_are_registered_and_propagate`, \
             then add it here."
        );
    }

    #[test]
    fn application_sqp_hessian_approximation_maps_to_damped_bfgs() {
        // The frontend sets `hessian_approximation = limited-memory` when no
        // exact Lagrangian Hessian is available (e.g. `pounce.minimize` with
        // no `hess`). On the active-set-SQP path that must resolve to the
        // dense Powell-damped BFGS, NOT the limited-memory update: L-BFGS
        // materializes the same dense Hessian for the QP subproblem yet stalls
        // (`Search_Direction_Becomes_Too_Small` / wrong `x`) on convex QPs with
        // an active inequality (issue #358); damped-BFGS solves them.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str(
            "algorithm active-set-sqp\n\
             hessian_approximation limited-memory\n",
        )
        .unwrap();
        assert_eq!(
            app.algorithm_builder_snapshot().sqp.hessian,
            crate::sqp::SqpHessianSource::DampedBfgs
        );

        // An explicit `sqp_hessian = lbfgs` is still honored (it is read after
        // `hessian_approximation`, so it wins): callers who genuinely want the
        // limited-memory update can still ask for it.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str(
            "algorithm active-set-sqp\n\
             hessian_approximation limited-memory\n\
             sqp_hessian lbfgs\n",
        )
        .unwrap();
        assert_eq!(
            app.algorithm_builder_snapshot().sqp.hessian,
            crate::sqp::SqpHessianSource::Lbfgs
        );
    }

    /// `builder.linear_solver` must name the backend that will actually be
    /// built, not the one the option string asked for.
    ///
    /// MA57 is behind the optional `ma57` cargo feature; without it
    /// `default_backend_factory` silently substitutes FERAL. Recording `Ma57`
    /// anyway made the field disagree with reality, and the Schur KKT gate in
    /// `alg_builder::build_with_backend` (which tests `== Feral`) consumed that
    /// disagreement — so `set_kkt_schur_block()` never engaged on the default
    /// pure-Rust build for any user, while the transparent fallback kept every
    /// answer correct and every test green.
    #[test]
    fn application_linear_solver_records_the_effective_backend() {
        // Default options resolve to FERAL in *every* build. The registry
        // used to default to upstream's "ma57", which meant an HSL build
        // silently ran MA57 without being asked and a pure-Rust build
        // advertised a backend it did not contain; the default now names
        // pounce's own solver and HSL is opt-in (gh#483 follow-up).
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        assert_eq!(
            app.algorithm_builder_from_options().linear_solver,
            LinearSolverChoice::Feral,
            "the registered default is `feral`, in an ma57 build too"
        );

        // An explicit ma57 request resolves the same way.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str("linear_solver ma57\n")
            .unwrap();
        let got = app.algorithm_builder_from_options().linear_solver;
        if cfg!(feature = "ma57") {
            assert_eq!(got, LinearSolverChoice::Ma57);
        } else {
            assert_eq!(got, LinearSolverChoice::Feral);
        }

        // An explicit feral request is honored in every build.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str("linear_solver feral\n")
            .unwrap();
        assert_eq!(
            app.algorithm_builder_from_options().linear_solver,
            LinearSolverChoice::Feral
        );
    }

    /// gh#746. `IpAlgBuilder.cpp:1059` substitutes `adaptive` for the
    /// registered `monotone` when `hessian_approximation` is
    /// limited-memory and the caller left `mu_strategy` alone. pounce
    /// read the registered default unconditionally, which is a
    /// different barrier schedule on the whole quasi-Newton arm.
    #[test]
    fn limited_memory_defaults_mu_strategy_to_adaptive() {
        // Exact Hessian, nothing set: monotone, as registered.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        assert_eq!(
            app.algorithm_builder_from_options().mu_strategy,
            MuStrategyChoice::Monotone,
            "the exact arm must keep the registered default"
        );

        // Limited memory, nothing set: adaptive.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str("hessian_approximation limited-memory\n")
            .unwrap();
        assert_eq!(
            app.algorithm_builder_from_options().mu_strategy,
            MuStrategyChoice::Adaptive,
            "limited-memory must take upstream's quasi-Newton default"
        );

        // An explicit `monotone` still wins — the substitution is only
        // for an absent option.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str(
            "hessian_approximation limited-memory\n\
             mu_strategy monotone\n",
        )
        .unwrap();
        assert_eq!(
            app.algorithm_builder_from_options().mu_strategy,
            MuStrategyChoice::Monotone,
            "an explicit mu_strategy must not be overridden"
        );
    }

    /// The μ-strategy auto-fallback retries with the *other* strategy.
    /// Under limited-memory the first attempt is adaptive, so the flip
    /// has to be monotone — reading the registered default there would
    /// re-run the strategy that just stalled (gh#746).
    #[test]
    fn fallback_flip_follows_the_resolved_mu_strategy() {
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        assert!(
            !app.effective_mu_strategy_is_adaptive(),
            "unset + exact resolves to monotone"
        );

        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str("hessian_approximation limited-memory\n")
            .unwrap();
        assert!(
            app.effective_mu_strategy_is_adaptive(),
            "unset + limited-memory resolves to adaptive"
        );

        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str(
            "hessian_approximation limited-memory\n\
             mu_strategy monotone\n",
        )
        .unwrap();
        assert!(
            !app.effective_mu_strategy_is_adaptive(),
            "an explicit monotone under limited-memory resolves to monotone"
        );
    }

    #[test]
    fn application_limited_memory_options_propagate_to_builder() {
        use crate::hess::lim_mem_quasi_newton::UpdateType;

        // Default: no options set -> bit-exact with Ipopt's default
        // (bfgs, history 6). This is what the IPM path runs unless the
        // user opts in, so it must not drift.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        let def = app.algorithm_builder_from_options();
        assert_eq!(def.limited_memory_update_type, UpdateType::Bfgs);
        assert_eq!(def.limited_memory_max_history, 6);

        // `limited_memory_update_type=sr1` and a custom history length
        // must reach the builder (these were registered upstream but
        // read nowhere on the IPM path before — see #131). Honoring
        // them is what lets SR1 break the monotone L-BFGS stall.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str(
            "hessian_approximation limited-memory\n\
             limited_memory_update_type sr1\n\
             limited_memory_max_history 9\n",
        )
        .unwrap();
        let snap = app.algorithm_builder_from_options();
        assert_eq!(snap.limited_memory_update_type, UpdateType::Sr1);
        assert_eq!(snap.limited_memory_max_history, 9);
    }

    #[test]
    fn application_recalc_y_is_wired_and_defaults_off() {
        // #677. Upstream registers `recalc_y` as `no`, but its option
        // text ends "If a limited memory quasi-Newton option is chosen,
        // this is used by default" — the effective default is
        // conditional on the Hessian approximation. pounce refused the
        // option outright before this, so an L-BFGS user had no way to
        // reach Ipopt's behaviour.

        // Exact Hessian: off, matching the registered default.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        let b = app.algorithm_builder_from_options();
        assert!(!b.recalc_y, "exact-Hessian default must stay off");
        assert_eq!(b.recalc_y_feas_tol, 1e-6, "default changed");

        // Limited memory: also off. Upstream's option text says it is
        // used by default there; pounce deliberately does not, because
        // auto-enabling took 7 of 57 fixtures from solved to not solved
        // on the L-BFGS leg with nothing moving the other way. See the
        // read site in `algorithm_builder_from_options`. If this
        // assertion is what fails, the auto-enable is being restored —
        // re-run `scripts/sweep-fixtures.sh` and explain those 7 first.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str("hessian_approximation limited-memory\n")
            .unwrap();
        assert!(
            !app.algorithm_builder_from_options().recalc_y,
            "limited-memory must not silently enable recalc_y"
        );

        // An explicit `yes` reaches the exact-Hessian path, which
        // used to be refused as unimplemented.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str("recalc_y yes\nrecalc_y_feas_tol 1e-3\n")
            .unwrap();
        let b = app.algorithm_builder_from_options();
        assert!(b.recalc_y);
        assert_eq!(b.recalc_y_feas_tol, 1e-3);
    }

    #[test]
    fn application_limited_memory_initialization_propagates_to_builder() {
        use crate::hess::lim_mem_quasi_newton::InitialApprox;

        // #677: registered with upstream's `scalar1` default and read
        // nowhere, so every limited-memory solve ran `scalar2` and
        // setting the option was a silent no-op. Each keyword must now
        // reach the builder.
        for (kw, want) in [
            ("scalar1", InitialApprox::Scalar1),
            ("scalar2", InitialApprox::Scalar2),
            ("scalar3", InitialApprox::Scalar3),
            ("scalar4", InitialApprox::Scalar4),
            ("constant", InitialApprox::Constant),
            ("history-max", InitialApprox::HistoryMax),
        ] {
            let mut app = IpoptApplication::new();
            app.initialize().unwrap();
            app.initialize_with_options_str(&format!(
                "hessian_approximation limited-memory\n\
                 limited_memory_initialization {kw}\n"
            ))
            .unwrap();
            assert_eq!(
                app.algorithm_builder_from_options()
                    .limited_memory_initialization,
                want,
                "limited_memory_initialization={kw} did not reach the builder"
            );
        }

        // `limited_memory_init_val` was unread too — the empty-history
        // branch hard-coded the same 1.0, so the miss was invisible.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str(
            "hessian_approximation limited-memory\n\
             limited_memory_init_val 4.5\n",
        )
        .unwrap();
        assert_eq!(
            app.algorithm_builder_from_options().limited_memory_init_val,
            4.5
        );

        // The effective default matches the registry and Ipopt
        // (`scalar1`). This is the assertion that would have caught #677
        // when the option was first registered: it pins the *selection*,
        // which the per-formula tests in `lim_mem_quasi_newton` cannot
        // see. Do not relax it to make an unrelated change pass.
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        let def = app.algorithm_builder_from_options();
        assert_eq!(def.limited_memory_initialization, InitialApprox::Scalar1);
        assert_eq!(def.limited_memory_init_val, 1.0);
    }

    #[test]
    fn application_sqp_warm_start_round_trip() {
        // Drive the convex-equality TNLP through the SQP path
        // twice. The first solve produces a working set; the
        // second is warm-started from it. The second must converge
        // with zero QP solves (the first KKT check declares
        // optimality immediately).
        let finalize_slot = std::rc::Rc::new(std::cell::RefCell::new(None));
        let tnlp_rc: std::rc::Rc<std::cell::RefCell<dyn TNLP>> =
            std::rc::Rc::new(std::cell::RefCell::new(ConvexEqTnlp {
                finalize_called: std::rc::Rc::clone(&finalize_slot),
            }));

        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str("algorithm active-set-sqp\n")
            .unwrap();

        // Cold solve.
        let status_a = app.optimize_tnlp(std::rc::Rc::clone(&tnlp_rc));
        assert_eq!(status_a, ApplicationReturnStatus::SolveSucceeded);
        let ws = app.last_sqp_working_set().cloned();
        assert!(ws.is_some(), "cold solve must yield a working set");

        // Build the warm-start iterate from the converged finalize
        // payload (just x; pad multipliers to 0 since the test
        // problem is convex).
        let (x_recv, _) = finalize_slot.borrow().clone().unwrap();
        let warm = crate::sqp::SqpIterates {
            x: x_recv,
            lambda_g: vec![1.0],
            lambda_x: vec![0.0, 0.0],
            working: ws,
        };
        app.set_sqp_warm_start(warm);

        // Warm solve.
        let status_b = app.optimize_tnlp(std::rc::Rc::clone(&tnlp_rc));
        assert_eq!(status_b, ApplicationReturnStatus::SolveSucceeded);
        assert!(app.last_sqp_working_set().is_some());
    }

    #[test]
    fn application_sqp_warm_start_auto_clears_after_use() {
        let finalize_slot = std::rc::Rc::new(std::cell::RefCell::new(None));
        let tnlp_rc: std::rc::Rc<std::cell::RefCell<dyn TNLP>> =
            std::rc::Rc::new(std::cell::RefCell::new(ConvexEqTnlp {
                finalize_called: std::rc::Rc::clone(&finalize_slot),
            }));
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str("algorithm active-set-sqp\n")
            .unwrap();
        app.set_sqp_warm_start(crate::sqp::SqpIterates {
            x: vec![0.0, 1.0],
            lambda_g: vec![1.0],
            lambda_x: vec![0.0, 0.0],
            working: None,
        });
        assert!(app.sqp_warm_start.is_some());
        let _ = app.optimize_tnlp(std::rc::Rc::clone(&tnlp_rc));
        assert!(
            app.sqp_warm_start.is_none(),
            "warm-start input must be auto-cleared after use"
        );
    }

    #[test]
    fn application_sqp_suboptions_default_when_unset() {
        // Without any sqp_* settings, the snapshot should equal
        // SqpOptions::default().
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        let snap = app.algorithm_builder_snapshot();
        let d = crate::sqp::SqpOptions::default();
        assert_eq!(snap.sqp.globalization, d.globalization);
        assert_eq!(snap.sqp.hessian, d.hessian);
        assert_eq!(snap.sqp.max_iter, d.max_iter);
        assert!((snap.sqp.tol - d.tol).abs() < 1e-18);
        assert!((snap.sqp.constr_viol_tol - d.constr_viol_tol).abs() < 1e-18);
        assert!((snap.sqp.dual_inf_tol - d.dual_inf_tol).abs() < 1e-18);
        assert!((snap.sqp.l1_penalty - d.l1_penalty).abs() < 1e-18);
        assert!((snap.sqp.bt_reduction - d.bt_reduction).abs() < 1e-18);
        assert!((snap.sqp.bt_min_alpha - d.bt_min_alpha).abs() < 1e-18);
        assert_eq!(snap.sqp.print_level, d.print_level);
        assert_eq!(snap.sqp.lbfgs_max_history, d.lbfgs_max_history);
    }

    #[test]
    fn application_reports_problem_dimensions() {
        let app = IpoptApplication::new();
        let mut tnlp = Hs071Stub;
        let info = app.problem_dimensions(&mut tnlp).unwrap();
        assert_eq!(info.n, 4);
        assert_eq!(info.m, 2);
        assert_eq!(info.nnz_jac_g, 8);
        assert_eq!(info.nnz_h_lag, 10);
    }

    /// Each of the four constant-derivative hints reaches the algorithm,
    /// and reaches its *own* slot (#551 / #677).
    ///
    /// All four were wired and consumed — gh#588 Q6 made pounce exploit
    /// them — but the read site looped over
    /// `constant_derivatives::HINT_OPTIONS`, so the registered-but-unread
    /// scan saw a loop variable where it needs a literal key and reported
    /// all four as silent no-ops. They are literals now, and this pins
    /// what a literal-per-name rewrite can get wrong that a loop could
    /// not: setting one hint must light up that hint's slot and no other,
    /// because `reconcile` pairs `asserted[k]` with the model's proof for
    /// `HINT_OPTIONS[k]` and a transposed pair would reuse the wrong
    /// derivative.
    #[test]
    fn each_constant_derivative_hint_lights_up_its_own_slot() {
        use pounce_nlp::constant_derivatives::HINT_OPTIONS;

        let app = IpoptApplication::new();
        assert_eq!(
            app.asserted_constant_derivative_hints(),
            [false; 4],
            "no hint is asserted on a fresh options list",
        );

        for (k, name) in HINT_OPTIONS.iter().enumerate() {
            let mut app = IpoptApplication::new();
            app.initialize().unwrap();
            app.initialize_with_options_str(&format!("{name} yes\n"))
                .unwrap();
            let mut expected = [false; 4];
            expected[k] = true;
            assert_eq!(
                app.asserted_constant_derivative_hints(),
                expected,
                "`{name}=yes` must set slot {k} and nothing else",
            );

            // …and the registered default asks for nothing, so an
            // `ipopt.opt` that spells it out changes no derivative reuse.
            let mut app = IpoptApplication::new();
            app.initialize().unwrap();
            app.initialize_with_options_str(&format!("{name} no\n"))
                .unwrap();
            assert_eq!(
                app.asserted_constant_derivative_hints(),
                [false; 4],
                "`{name}=no` is the registered default and asserts nothing",
            );
        }
    }

    /// `min x²` on `[-10, 10]` from `x = 1`, with an *exactly correct*
    /// gradient `2x`. Unconstrained and one-dimensional so the only
    /// thing the derivative checker can react to is its own step size
    /// and threshold.
    ///
    /// The forward difference at step `h` is `((1+h)² − 1)/h = 2 + h`,
    /// so the deviation from the analytic `2` is exactly `h`, and the
    /// relative test flags it when `h > tol·(2 + h)`. That makes the
    /// verdict a closed-form function of the two knobs under test.
    struct ExactQuadratic;
    impl TNLP for ExactQuadratic {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 1,
                m: 0,
                nnz_jac_g: 0,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[-10.0]);
            b.x_u.copy_from_slice(&[10.0]);
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x.copy_from_slice(&[1.0]);
            true
        }
        fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
            Some(x[0] * x[0])
        }
        fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
            grad[0] = 2.0 * x[0];
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _mode: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn eval_h(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _obj_factor: Number,
            _lambda: Option<&[Number]>,
            _new_lambda: bool,
            _mode: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    struct RecordingQuadratic {
        gradient_points: Rc<RefCell<Vec<Number>>>,
        objective_points: Rc<RefCell<Vec<Number>>>,
        x_scaling: Number,
    }

    impl TNLP for RecordingQuadratic {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            ExactQuadratic.get_nlp_info()
        }

        fn get_bounds_info(&mut self, bounds: BoundsInfo<'_>) -> bool {
            ExactQuadratic.get_bounds_info(bounds)
        }

        fn get_starting_point(&mut self, start: StartingPoint<'_>) -> bool {
            ExactQuadratic.get_starting_point(start)
        }

        fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
            self.objective_points.borrow_mut().push(x[0]);
            ExactQuadratic.eval_f(x, new_x)
        }

        fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
            self.gradient_points.borrow_mut().push(x[0]);
            grad[0] = 2.0 * x[0];
            true
        }

        fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
            ExactQuadratic.eval_g(x, new_x, g)
        }

        fn eval_jac_g(
            &mut self,
            x: Option<&[Number]>,
            new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            ExactQuadratic.eval_jac_g(x, new_x, mode)
        }

        fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
            *req.obj_scaling = 1.0;
            *req.use_x_scaling = true;
            req.x_scaling[0] = self.x_scaling;
            *req.use_g_scaling = false;
            true
        }

        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    fn derivative_test_verdict(extra: &str) -> pounce_nlp::derivative_test::DerivativeTestReport {
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str(&format!("derivative_test first-order\n{extra}"))
            .unwrap();
        let opts = app.derivative_test_options();
        pounce_nlp::derivative_test::run(&mut ExactQuadratic, &opts).expect("a report")
    }

    /// `derivative_test_perturbation` and `derivative_test_tol` were
    /// registered, read into [`DerivativeTestOptions`], and consumed by
    /// the checker — but nothing proved a *set value* reached it, which
    /// is the only assertion that distinguishes a read site from a
    /// parse-and-discard (#677, #551).
    ///
    /// Each of the three verdicts below differs from the one above it by
    /// exactly one option, so a knob that stopped reaching the checker
    /// would collapse two of them together and fail here.
    #[test]
    fn the_derivative_checker_knobs_change_the_verdict() {
        // Registered defaults (1e-8 / 1e-4), which are also the read
        // site's fallbacks: a correct gradient looks correct.
        let clean = derivative_test_verdict("");
        assert_eq!(clean.checked, 1);
        assert_eq!(clean.suspicious, 0, "{:#?}", clean.lines);

        // A coarse step makes the *same correct gradient* look wrong:
        // deviation 0.5 > 1e-4·2.5. Only `derivative_test_perturbation`
        // changed, so this is that option and nothing else.
        let coarse = derivative_test_verdict("derivative_test_perturbation 0.5\n");
        assert_eq!(coarse.checked, 1);
        assert_eq!(
            coarse.suspicious, 1,
            "derivative_test_perturbation never reached the checker: {:#?}",
            coarse.lines,
        );

        // …and loosening the threshold at that same coarse step clears
        // it again: 0.5 < 0.5·2.5. Only `derivative_test_tol` changed.
        let tolerant =
            derivative_test_verdict("derivative_test_perturbation 0.5\nderivative_test_tol 0.5\n");
        assert_eq!(tolerant.checked, 1);
        assert_eq!(
            tolerant.suspicious, 0,
            "derivative_test_tol never reached the checker: {:#?}",
            tolerant.lines,
        );

        // The report also prints what it used, so a user reading the
        // output can tell which step and threshold produced the verdict.
        assert!(
            tolerant.lines[0].contains("5.0e-1"),
            "{:#?}",
            tolerant.lines,
        );
    }

    #[test]
    fn ordinary_derivative_test_keeps_the_conditioned_start() {
        let gradient_points = Rc::new(RefCell::new(Vec::new()));
        let tnlp = Rc::new(RefCell::new(RecordingQuadratic {
            gradient_points: Rc::clone(&gradient_points),
            objective_points: Rc::new(RefCell::new(Vec::new())),
            x_scaling: 1.0,
        })) as Rc<RefCell<dyn TNLP>>;
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str(
            "derivative_test first-order\n\
             start_point_perturbation 0.5\n\
             hessian_approximation limited-memory\n\
             max_iter 0\n\
             print_level 0\n",
        )
        .unwrap();

        let _ = app.optimize_tnlp(tnlp);

        let first = gradient_points.borrow()[0];
        assert!((first - 1.7666216164272852).abs() < 1e-12, "{first}");
    }

    #[test]
    fn ordinary_derivative_test_keeps_variable_scaling() {
        let objective_points = Rc::new(RefCell::new(Vec::new()));
        let tnlp = Rc::new(RefCell::new(RecordingQuadratic {
            gradient_points: Rc::new(RefCell::new(Vec::new())),
            objective_points: Rc::clone(&objective_points),
            x_scaling: 0.5,
        })) as Rc<RefCell<dyn TNLP>>;
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str(
            "derivative_test first-order\n\
             derivative_test_perturbation 0.5\n\
             nlp_scaling_method user-scaling\n\
             hessian_approximation limited-memory\n\
             max_iter 0\n\
             print_level 0\n",
        )
        .unwrap();

        let _ = app.optimize_tnlp(tnlp);

        assert_eq!(&objective_points.borrow()[..2], &[1.0, 2.0]);
    }

    // ---- gh#887: the dominance gate ------------------------------------
    //
    // These pin the rule directly, on numbers measured off real runs,
    // because the fixture that motivated the gate turned out not to be a
    // portable witness for it. `deb7` on the L-BFGS leg with
    // `limited_memory_ls_failure_restarts=1` reaches a *materially
    // different answer* on macOS and on Linux -- different objective
    // (99.677 vs 99.651) and, decisively, a different shape:
    //
    //   | run                 | unscaled dual | viol    | compl   | ratio   |
    //   | reproducer, .nl     | 7.90e4        | 1.1e-16 | 1.1e-9  | 1.5e-14 |
    //   | reproducer, TNLP    | 3.25e11       | 2.5e-16 | 2.8e-3  | 8.7e-15 |
    //   | deb7 + rung, macOS  | 9.90e1        | 8.0e-13 | 4.65e0  | 4.7e-2  |
    //   | deb7 + rung, Linux  | 5.5743e3      | 5.6e-14 | 2.08e-5 | 3.7e-9  |
    //
    // The Linux row is the one worth reading twice: that answer really is
    // gh#884's shape (scaled overall error 5.28e-1 against unscaled
    // 5.57e3, which is the `s_d` normalisation hiding a runaway exactly
    // as it did on `qpec_small`), so the retry there is the designed cost
    // and not the waste gh#887 filed. A CLI assertion of "deb7 declines"
    // is therefore false on Linux no matter what the threshold is, which
    // is why the pin lives here instead.

    /// The dominance rule on its own, with the absolute floor switched
    /// off (`du_floor = 0`), so each of these tests is about one rule.
    fn ratio_only(dual_inf: Number, viol: Number, compl: Number) -> bool {
        runaway_is_the_whole_residual(dual_inf, viol, compl, 0.0)
    }

    /// The gate as it actually runs: dominance *and* the detector's own
    /// absolute floor.
    fn runaway(dual_inf: Number, viol: Number, compl: Number) -> bool {
        runaway_is_the_whole_residual(dual_inf, viol, compl, DUAL_DIV_RETRY_DU_FLOOR)
    }

    #[test]
    fn a_converged_point_with_a_runaway_multiplier_opens_the_retry() {
        // The gh#884 reproducer, both routes it reaches the gate by.
        assert!(runaway(7.90e4, 1.1e-16, 1.1e-9));
        assert!(runaway(3.25e11, 2.5e-16, 2.8e-3));
        // deb7 under the rung on Linux: primal exact, complementarity
        // eight orders under its own dual residual. Same shape.
        assert!(runaway(5.5743e3, 5.6e-14, 2.08e-5));
    }

    #[test]
    fn an_unconverged_point_does_not_open_the_retry() {
        // deb7 under the rung on macOS: complementarity 4.65, five
        // percent of its own KKT error. Not a runaway multiplier on an
        // otherwise-converged point -- just an unconverged point.
        assert!(!ratio_only(9.90e1, 8.0e-13, 4.65e0));
        // Either residual alone is enough to close it: the gate takes the
        // max, so a clean complementarity does not excuse a violated
        // constraint. (`1.0e1` against `1.0e6` is a ratio of `1e-5`; note
        // that `1.0e0` there would be `1e-6` exactly, i.e. inside.)
        assert!(!ratio_only(1.0e6, 1.0e1, 1.0e-16));
        assert!(!ratio_only(1.0e6, 1.0e-16, 1.0e1));
    }

    /// The dominance ratio is scale-free, and that is exactly why it
    /// cannot be the whole gate: a point converged to `1e-30` primal with
    /// a dual residual of `4.4e-1` satisfies it as comfortably as
    /// gh#884's `7.9e+04` does, and `4.4e-1` is not a runaway by any
    /// reading of the issue. Measured on the 400-model QPEC family, the
    /// floor alone removes 7 of 68 promotions, every one of them on an
    /// answer whose reported dual residual was below `1e2`.
    ///
    /// The floor is the *detector's*, so this is the answer-level gate
    /// asking about the same magnitude the iterate-level one did rather
    /// than being a strictly looser copy of it.
    #[test]
    fn a_small_dual_residual_is_not_a_runaway_however_dominant() {
        // Passes the ratio comfortably (1e-30 / 4.4e-1 = 2.3e-30) ...
        assert!(ratio_only(4.4e-1, 1.0e-30, 1.0e-30));
        // ... and is still refused, because it is not a runaway.
        assert!(!runaway(4.4e-1, 1.0e-30, 1.0e-30));
        // The two real `r`-family answers that reached the gate this way.
        assert!(!runaway(4.397e-1, 1.0e-16, 1.0e-16));
        assert!(!runaway(2.026e1, 1.0e-16, 1.0e-16));
        // Exactly at the floor is inside; a hair under is outside.
        assert!(runaway(DUAL_DIV_RETRY_DU_FLOOR, 0.0, 0.0));
        assert!(!runaway(DUAL_DIV_RETRY_DU_FLOOR * 0.999, 0.0, 0.0));
    }

    #[test]
    fn the_threshold_is_where_the_constant_says_it_is() {
        // Exactly at the ratio is inside; a hair past it is outside.
        assert!(ratio_only(1.0, DUAL_DIV_RETRY_DOMINANCE, 0.0));
        assert!(!ratio_only(1.0, DUAL_DIV_RETRY_DOMINANCE * 1.001, 0.0));
    }

    #[test]
    fn what_we_cannot_measure_does_not_buy_a_retry() {
        // A NaN compares false everywhere, so the condition is written so
        // that "we cannot tell" declines rather than retries. Each
        // argument in turn.
        assert!(!ratio_only(Number::NAN, 0.0, 0.0));
        assert!(!ratio_only(1.0e6, Number::NAN, 0.0));
        assert!(!ratio_only(1.0e6, 0.0, Number::NAN));
        assert!(!ratio_only(Number::INFINITY, 0.0, 0.0));
        // And a dual residual that is not a runaway at all: the ratio
        // would be meaningless, and a zero-dual point is not gh#884.
        assert!(!ratio_only(0.0, 0.0, 0.0));
        assert!(!ratio_only(-1.0, 0.0, 0.0));
    }

    // ---- the promotion gate reads the ANSWER, not only the certificate --
    //
    // gh#884 ranked the two attempts on unscaled KKT error alone and
    // argued that this could not return a different local solution,
    // because "conjunct 4 requires the promoted answer to satisfy the KKT
    // conditions in the model's own units". Any other KKT point does too.
    // Measured on 400 random QPECs under the `prod_eq` lowering
    // (`bound_relax_factor=0 mu_strategy_fallback=no tol=1e-8`): 68
    // promotions, 42 of which moved the objective materially, and three
    // of which returned a strictly worse *feasible* point.
    //
    // Every number below is off a real run, and the two branches are
    // separated on purpose -- a rule that branches needs a case on each
    // side or the untaken one stays broken while the test is green.
    //
    //   | model      | base f      | base viol | retry f     | retry viol | branch |
    //   |------------|-------------|-----------|-------------|------------|--------|
    //   | qpec_small | +3.586e-28  | 1.11e-16  | +5.835e-11  | 5.47e-12   | admit  |
    //   | r116       | -1.3006e+01 | 2.22e-16  | -1.2072e+00 | 4.55e-13   | rule 1 |
    //   | r261       | -4.7919e+00 | 1.07e-14  | -9.8563e-01 | 7.99e-14   | rule 1 |
    //   | r201       | -2.9559e-01 | 6.25e-17  | -9.7321e-02 | 6.21e-13   | rule 1 |
    //   | scholtes4  | +1.8176e-09 | 2.07e-25  | -6.6088e-05 | 1.09e-09   | rule 2 |

    const ACCEPT: Number = 1e-6;
    /// Minimization, which is every row of the table above.
    const MIN: Number = 1.0;
    /// Maximization, i.e. `obj_scaling_factor < 0`.
    const MAX: Number = -1.0;

    fn admissible(bo: Number, bv: Number, ro: Number, rv: Number) -> bool {
        retry_answer_is_admissible(bo, bv, ro, rv, ACCEPT, MIN)
    }

    #[test]
    fn the_reproducers_promotion_is_still_admissible() {
        // `qpec_small`: the retry's objective is *worse*, by 5.8e-11 --
        // deliberately, since it buys nine orders of unscaled dual
        // residual. Five orders inside the tolerance, so rule 1 admits
        // it, which is the whole point of having a tolerance at all.
        assert!(admissible(3.586e-28, 1.11e-16, 5.835e-11, 5.47e-12));
    }

    /// Rule 1: a strictly worse feasible point is never an upgrade.
    #[test]
    fn a_worse_feasible_objective_is_refused_however_clean_the_certificate() {
        // r116: -13.0057 -> -1.2072, both independently verified feasible
        // by `pounce verify`. The retry's unscaled KKT error is 2.9e-11
        // against the base attempt's 3.0e+03, so every certificate
        // conjunct passes and only this one refuses it.
        assert!(!admissible(
            -1.3005680756e1,
            2.22e-16,
            -1.2072337962e0,
            4.55e-13
        ));
        assert!(!admissible(
            -4.7919265770e0,
            1.07e-14,
            -9.8562977711e-1,
            7.99e-14
        ));
        assert!(!admissible(
            -2.9558632401e-1,
            6.25e-17,
            -9.7321185691e-2,
            6.21e-13
        ));
    }

    /// Rule 2: an objective *improvement* bought with primal slack.
    ///
    /// `scholtes4` (`benchmarks/mpcc/cases.py`, and now a CLI fixture) has
    /// `f* = 0` exactly -- for the MPCC and for the smooth lowering alike,
    /// since `x1*x2 = 0` forces one of them to zero, hence `x3 <= 0`, hence
    /// `f = x1 - x3 >= 0`. The retry reports `-6.61e-05`, which no feasible
    /// point reaches, by moving the complementarity row 16 orders further
    /// out. Rule 1 cannot see it: the objective got *better*.
    #[test]
    fn an_improvement_bought_with_primal_slack_is_refused() {
        assert!(!admissible(
            1.8175997416e-9,
            2.07e-25,
            -6.6088333055e-5,
            1.09e-9
        ));
        // The same numbers with the primal *held* would be admissible --
        // this is the conjunct that refuses it, not the objective move.
        assert!(admissible(
            1.8175997416e-9,
            2.07e-25,
            -6.6088333055e-5,
            2.07e-25
        ));
    }

    /// The window the tolerance sits in, from both sides, so it is a
    /// checkable claim rather than a fitted constant: the smallest move
    /// that must be admitted is `qpec_small`'s `5.8e-11` and the smallest
    /// that must be refused is `r201`'s `0.198`, four and five orders
    /// away from `acceptable_tol` on either side.
    #[test]
    fn the_objective_tolerance_is_not_fitted_to_one_model() {
        let admit = 5.835e-11;
        let refuse = 0.198;
        assert!(admit < ACCEPT / 1.0e4, "{admit} is not well inside");
        assert!(refuse > ACCEPT * 1.0e4, "{refuse} is not well outside");
        // Scale-relative above 1, absolute below it -- the same
        // convention `sigma_forward_error_is_small` uses for `norm(x)`.
        assert!(admissible(1.0e6, 0.0, 1.0e6 + 0.5, 0.0));
        assert!(!admissible(1.0e6, 0.0, 1.0e6 + 5.0, 0.0));
    }

    /// An infeasible base attempt is not a point worth protecting, so
    /// both rules stand down and the certificate conjuncts decide alone.
    /// R2: `obj_scaling_factor < 0` poses a maximization, and
    /// `final_objective` is the user's **signed** objective, so both rules
    /// have to follow the sense. This is the table above with every
    /// objective negated: the same three answers must be refused and the
    /// same one admitted, with `MAX` instead of `MIN`.
    ///
    /// Without the normalization each row flips — rule 1 starts refusing
    /// genuine improvements (a regression against the behaviour before the
    /// conjunct existed) and rule 2 starts admitting strictly worse
    /// answers, which is the class it was added to block.
    #[test]
    fn the_rules_follow_the_objective_sense() {
        // qpec_small, mirrored: the retry is worse by 5.8e-11, well inside
        // the tolerance, so it is still admitted.
        assert!(retry_answer_is_admissible(
            -3.586e-28, 1.11e-16, -5.835e-11, 5.47e-12, ACCEPT, MAX
        ));
        // r116, mirrored: +13.0057 given up for +1.2072 is now a *worse*
        // maximum, and rule 1 must still refuse it.
        assert!(!retry_answer_is_admissible(
            1.3005680756e1,
            2.22e-16,
            1.2072337962e0,
            4.55e-13,
            ACCEPT,
            MAX
        ));
        // scholtes4, mirrored: +6.6088e-05 is now an improvement bought
        // with primal slack, and rule 2 must still refuse it.
        assert!(!retry_answer_is_admissible(
            -1.8175997416e-9,
            2.07e-25,
            6.6088333055e-5,
            1.09e-9,
            ACCEPT,
            MAX
        ));
        // ... and the same improvement with the primal held is admitted.
        assert!(retry_answer_is_admissible(
            -1.8175997416e-9,
            2.07e-25,
            6.6088333055e-5,
            2.07e-25,
            ACCEPT,
            MAX
        ));
    }

    /// The mirror of the above, stated as the property rather than the
    /// rows: negating both objectives and flipping the sense must leave
    /// every verdict unchanged.
    #[test]
    fn negating_the_objective_and_the_sense_is_inert() {
        for &(bo, bv, ro, rv) in &[
            (3.586e-28, 1.11e-16, 5.835e-11, 5.47e-12),
            (-1.3005680756e1, 2.22e-16, -1.2072337962e0, 4.55e-13),
            (1.8175997416e-9, 2.07e-25, -6.6088333055e-5, 1.09e-9),
            (1.8175997416e-9, 2.07e-25, -6.6088333055e-5, 2.07e-25),
            (-1.0e3, 1.0e-2, 0.0, 1.0e-12),
        ] {
            assert_eq!(
                retry_answer_is_admissible(bo, bv, ro, rv, ACCEPT, MIN),
                retry_answer_is_admissible(-bo, bv, -ro, rv, ACCEPT, MAX),
                "verdict moved under (obj, sense) -> (-obj, -sense) at {bo:e}/{ro:e}"
            );
        }
    }

    #[test]
    fn an_infeasible_base_attempt_protects_nothing() {
        assert!(admissible(-1.0e3, 1.0e-2, 0.0, 1.0e-12));
        // ... and what cannot be measured is refused, not admitted, the
        // same way `runaway_is_the_whole_residual` treats a NaN.
        assert!(admissible(Number::NAN, 0.0, 0.0, 0.0));
        assert!(!admissible(0.0, 0.0, Number::NAN, 0.0));
        assert!(!admissible(0.0, 0.0, -1.0, Number::NAN));
    }
}
