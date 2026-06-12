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
use crate::ipopt_alg::IpoptAlgorithm;
use crate::ipopt_cq::IpoptCalculatedQuantities;
use crate::ipopt_data::IpoptData as AlgIpoptData;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::IteratesVector;
use crate::restoration::RestorationPhase;
use crate::upstream_options::register_all_upstream_options;

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
use pounce_linsol::summary::LinearSolverSummary;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::orig_ipopt_nlp::{ConstObjScaling, OrigIpoptNlp, ScalingMethod};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{
    IpoptCq as TnlpIpoptCq, IpoptData as TnlpIpoptData, NlpInfo, Solution, TNLP,
};
use pounce_nlp::tnlp_adapter::{
    FixedVarTreatment, TNLPAdapter, DEFAULT_NLP_LOWER_BOUND_INF, DEFAULT_NLP_UPPER_BOUND_INF,
};
use std::cell::RefCell;
use std::fmt;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct IpoptApplication {
    options: OptionsList,
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
            linsol_summary_sink: Arc::new(Mutex::new(LinearSolverSummary::default())),
            sqp_warm_start: None,
            sqp_last_working_set: None,
            warm_start_iterate: None,
        }
    }

    pub fn options(&self) -> &OptionsList {
        &self.options
    }

    pub fn options_mut(&mut self) -> &mut OptionsList {
        &mut self.options
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

    /// Enable per-iteration trajectory capture. After the solve
    /// returns, [`Self::statistics()`] exposes
    /// [`pounce_nlp::solve_statistics::SolveStatistics::iterations`]
    /// populated with one [`pounce_nlp::solve_statistics::IterRecord`]
    /// per accepted iterate. Off by default — the `pounce_sens` and
    /// `pounce` binaries opt in when `--json-output` is passed.
    pub fn enable_iter_history(&mut self) {
        self.record_iter_history = true;
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

    pub fn statistics(&self) -> SolveStatistics {
        self.statistics.borrow().clone()
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
    pub fn optimize_tnlp(&mut self, tnlp: Rc<RefCell<dyn TNLP>>) -> ApplicationReturnStatus {
        // Top-level algorithm dispatch (Phase 5b §7.1). When the
        // `algorithm` option resolves to "active-set-sqp", route
        // to the Phase 5b SQP path; otherwise fall through to the
        // existing IPM flow unchanged.
        if self.is_sqp_algorithm_selected() {
            return self.optimize_sqp_tnlp(tnlp);
        }
        let info = match tnlp.borrow_mut().get_nlp_info() {
            Some(info) => info,
            None => return ApplicationReturnStatus::InvalidProblemDefinition,
        };
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

    /// Return the final QP working set from the most recent SQP
    /// solve, or `None` if the last solve wasn't SQP, didn't
    /// produce a working set (cold-start declared the iterate
    /// optimal before solving any QP), or no SQP solve has run.
    pub fn last_sqp_working_set(&self) -> Option<&pounce_qp::WorkingSet> {
        self.sqp_last_working_set.as_ref()
    }

    fn is_sqp_algorithm_selected(&self) -> bool {
        match self.options.get_string_value("algorithm", "") {
            Ok((v, true)) => v.eq_ignore_ascii_case("active-set-sqp"),
            _ => false,
        }
    }

    /// Phase 5b SQP entry point. Builds the same NLP chain
    /// (`TNLPAdapter` → `OrigIpoptNlp` → `IpoptNlpAdapter`) the
    /// IPM uses, then runs `SqpAlgorithm::optimize`. Maps the
    /// `SqpResult.status` back to `ApplicationReturnStatus` and
    /// hands the final iterate to the user TNLP's
    /// `finalize_solution` callback via `finalize_via_sqp`.
    fn optimize_sqp_tnlp(&mut self, tnlp: Rc<RefCell<dyn TNLP>>) -> ApplicationReturnStatus {
        use pounce_nlp::orig_ipopt_nlp::OrigIpoptNlp;
        use pounce_nlp::tnlp_adapter::TNLPAdapter;
        use pounce_nlp::ConstObjScaling;

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
        let orig_nlp = match OrigIpoptNlp::new(
            Rc::clone(&adapter),
            Rc::new(ConstObjScaling(obj_scaling_factor)),
        ) {
            Ok(n) => n,
            Err(_) => return ApplicationReturnStatus::InternalError,
        };
        let nlp_rc: Rc<RefCell<dyn IpoptNlp>> = Rc::new(RefCell::new(orig_nlp));

        let mut sqp_adapter = crate::sqp::IpoptNlpAdapter::new(Rc::clone(&nlp_rc));

        let mut builder = self.algorithm_builder_snapshot();
        builder.algorithm = crate::alg_builder::AlgorithmChoice::ActiveSetSqp;
        let factory = self.make_backend_factory();
        let mut alg = match builder.build_sqp_with_backend(factory) {
            Some(a) => a,
            None => return ApplicationReturnStatus::InternalError,
        };

        // Phase 5c (§6): consume any stashed warm-start iterate.
        // `optimize_with_warm_start(warm=None)` is equivalent to
        // `optimize`, so cold callers see no change.
        let warm = self.sqp_warm_start.take();
        let res = match alg.optimize_with_warm_start(&mut sqp_adapter, warm) {
            Ok(r) => r,
            Err(e) => {
                if std::env::var_os("POUNCE_DBG_SQP").is_some() {
                    tracing::warn!(target: "pounce::sqp", "[SQP] optimize_with_warm_start error: {e:?}");
                }
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
            stats.final_objective = res.obj;
            stats.final_dual_inf = res.final_stationarity;
            stats.final_constr_viol = res.final_constr_viol;
            stats.final_compl = 0.0; // SQP has no barrier — no compl term.
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
        };

        // Forward to the user TNLP's finalize_solution. We pass
        // the SQP iterate and recovered multipliers via the
        // OrigIpoptNlp's lifting hooks. Failure here is silent
        // (we still return the algorithm's status) — the user
        // sees the right ApplicationReturnStatus regardless.
        let _ = finalize_via_sqp(&nlp_rc, &res, solver_status, &tnlp);

        app_status
    }

    /// Build a *copy* of the algorithm builder configured per the
    /// current options. The SQP path uses this so it gets a
    /// fresh builder without mutating the application's state.
    fn algorithm_builder_snapshot(&self) -> AlgorithmBuilder {
        let mut builder = AlgorithmBuilder::default();
        apply_sqp_options(&self.options, &mut builder.sqp);
        builder
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
            if slack_sum.is_finite() && slack_sum <= slack_tol {
                break;
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

            // Honest-infeasibility upgrade (Phase 3): if the inner
            // solve says SolveSucceeded / SolvedToAcceptableLevel but
            // the slacks did not collapse, the original problem is
            // locally infeasible at the returned point. Override the
            // application status; the user-visible Solution.status is
            // updated below to the matching SolverReturn so the inner
            // TNLP sees a consistent picture.
            let infeasible_certificate = matches!(
                last_status,
                ApplicationReturnStatus::SolveSucceeded
                    | ApplicationReturnStatus::SolvedToAcceptableLevel
            ) && slack_sum.is_finite()
                && slack_sum > slack_tol;
            let final_app_status = if infeasible_certificate {
                ApplicationReturnStatus::InfeasibleProblemDetected
            } else {
                last_status
            };
            let final_solver_status = if infeasible_certificate {
                SolverReturn::LocalInfeasibility
            } else {
                solver_status
            };

            // Recompute f(x*) and c(x*) on the inner.
            let f_inner = tnlp
                .borrow_mut()
                .eval_f(&x_trunc, true)
                .unwrap_or(Number::NAN);
            let m = tnlp
                .borrow_mut()
                .get_nlp_info()
                .map(|i| i.m as usize)
                .unwrap_or(0);
            let mut g_inner = vec![0.0; m];
            if m > 0 {
                let _ = tnlp.borrow_mut().eval_g(&x_trunc, false, &mut g_inner);
            }
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
    fn optimize_constrained(&mut self, tnlp: Rc<RefCell<dyn TNLP>>) -> ApplicationReturnStatus {
        let t_start = Instant::now();

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
        timing.overall_alg.start();

        // Reset the linear-solver summary sink so back-to-back solves
        // don't bleed factor counters / extremal pivots into each
        // other. Surviving the lock failure with a debug-assert keeps
        // a poisoned mutex from sinking a release build that doesn't
        // even consume the summary.
        if let Ok(mut guard) = self.linsol_summary_sink.lock() {
            *guard = LinearSolverSummary::default();
        } else {
            debug_assert!(false, "linsol summary sink mutex poisoned");
        }

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

        // Mirror upstream `OrigIpoptNLP::InitializeStructures` (IpOrigIpoptNLP.cpp:299):
        // bail out with NotEnoughDegreesOfFreedom when there are fewer free
        // variables than equality constraints. Without this gate, square /
        // over-determined systems push the algorithm into restoration on
        // iter 0 and exit Restoration_Failed instead of the cleaner DOF code.
        let n_x_var = orig_nlp.x_space().dim();
        let n_c = orig_nlp.c_space().dim();
        if n_x_var > 0 && n_x_var < n_c {
            timing.overall_alg.end();
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
            "user-scaling" => ScalingMethod::UserScaling,
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
        let builder = self.algorithm_builder_from_options();

        // Linear-solver backend. The default factory is option-aware
        // — it reads the `feral_*` extension options off the same
        // `OptionsList` that drove the IPM-level builder above so
        // per-problem `.opt` files can flip backend knobs without
        // rebuilding pounce.
        let feral_cfg = feral_config_from_options(&self.options);
        let factory = self.linear_backend_factory.take().unwrap_or_else(|| {
            default_backend_factory_with_sink(feral_cfg, Arc::clone(&self.linsol_summary_sink))
        });
        let bundle = builder.build_with_backend(factory);

        // Wire the data / cq pair around the NLP. Install the shared
        // `TimingStatistics` so the algorithm's iterate phases
        // (output, convergence, hessian, μ, search-direction,
        // line-search, accept) all record into the same accumulator
        // the application exposes via `timing_stats()`.
        let data: crate::ipopt_data::IpoptDataHandle = Rc::new(RefCell::new(AlgIpoptData::new()));
        data.borrow_mut().timing = Rc::clone(&timing);
        let cq: crate::ipopt_cq::IpoptCqHandle = Rc::new(RefCell::new(
            IpoptCalculatedQuantities::new(Rc::clone(&data), Rc::clone(&nlp_handle)),
        ));
        // Correction size for very small slacks (default mach_eps^{3/4});
        // drives the safe-slack bound-adjustment mechanism.
        if let Ok((v, true)) = self.options.get_numeric_value("slack_move", "") {
            cq.borrow_mut().slack_move = v;
        }

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
        // Honor `print_level == 0`: suppress the per-iteration table
        // that the algorithm writes straight to stdout. (The Phase-7
        // journalist surface respects `print_level` already; this is
        // the legacy direct-print site that needs the same gate.)
        if let Ok((v, found)) = self.options.get_integer_value("print_level", "") {
            if found && v <= 0 {
                alg.print_iter_output = false;
                // The nested restoration IPM is built inside the
                // restoration driver, not by `IpoptAlgorithm::new`, so
                // it never sees this gate unless we forward it.
                if let Some(resto) = alg.restoration.as_mut() {
                    resto.set_print_iter_output(false);
                }
            }
        }

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

        let captured_iters = iter_capture.map(|g| g.finish()).unwrap_or_default();
        // Close the overall-algorithm timer on the success path. The
        // early-return arms above end it themselves before bailing out;
        // this one matches upstream `IpoptApplication::call_optimize`
        // (which calls `EndCpuTime()` on overall_alg right after
        // `Optimize` returns, regardless of solver_status).
        timing.overall_alg.end();

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
            stats.total_wallclock_time_secs = t_start.elapsed().as_secs_f64();
            // Restoration-phase audit counters (pounce#12). Zero on
            // problems where restoration never fires; populated by
            // `IpoptAlgorithm::invoke_restoration`.
            stats.restoration_calls = alg.resto_calls;
            stats.restoration_inner_iters = alg.resto_inner_iters;
            stats.restoration_outer_iters = alg.resto_outer_iters;
            stats.restoration_wall_secs = alg.resto_wall_secs;
            stats.iterations = captured_iters;
            // Capture the final *scaled* objective at the algorithm's
            // (compressed `x_var`-space) iterate via the NLP: the
            // algorithm-side `eval_f` returns `f * obj_scale_factor`.
            // `final_objective` is seeded with it only as a best-effort
            // fallback; the success path below overwrites it with the
            // true unscaled objective from `finalize_via_orig_nlp`
            // (which evaluates the user TNLP directly).
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
            stats.final_constr_viol = cq.curr_primal_infeasibility_max();
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
        }

        // Map SolverReturn → ApplicationReturnStatus per
        // MAIN_LOOP.md's exception table.
        let app_status = solver_return_to_app_status(solver_status);

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
        match finalize_via_orig_nlp(&nlp_handle, &alg, solver_status, app_status, &tnlp) {
            Ok(f_unscaled) => {
                self.statistics.borrow_mut().final_objective = f_unscaled;
            }
            Err(()) => {
                // Couldn't finalize; keep the scaled fallback and
                // surface the original status.
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

        // `mehrotra_algorithm` is parsed first so its cascading
        // defaults (mu_strategy=adaptive, mu_oracle=probing) can be
        // overridden by an explicit user setting of those keys
        // below. Mirrors `IpAlgBuilder.cpp:Mehrotra`.
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
                // `alpha_for_y=bound_mult` — Mehrotra wants the
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
                    _ => HessianApproxChoice::Exact,
                };
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

        // Unlike the other options here, we always honor the registry
        // value (not just when the user set it explicitly): the option
        // registry default is "ma57" but `AlgorithmBuilder::default`
        // has `linear_solver: Feral`, so gating on `found` would
        // silently route default runs through Feral while the banner
        // (and ipopt-compatible behavior) advertises MA57.
        if let Ok((v, _found)) = self.options.get_string_value("linear_solver", "") {
            builder.linear_solver = match v.as_str() {
                "ma57" => LinearSolverChoice::Ma57,
                _ => LinearSolverChoice::Feral,
            };
        }

        // `linear_system_scaling` — symmetric scaling of the augmented
        // KKT matrix before factorization. Port of
        // `IpTSymLinearSolver.cpp:RegisterOptions` plumbing. Default
        // "none"; "ruiz" invokes the Ruiz-2001 symmetric ∞-norm
        // equilibration in `RuizTSymScalingMethod`. "mc19" and
        // "slack-based" are accepted by the registry but not yet
        // implemented at this layer; they fall back to no scaling
        // with a one-line stderr notice.
        if let Ok((v, found)) = self.options.get_string_value("linear_system_scaling", "") {
            if found {
                builder.linear_system_scaling = match v.as_str() {
                    "ruiz" => crate::alg_builder::LinearSystemScalingChoice::Ruiz,
                    "mc19" => crate::alg_builder::LinearSystemScalingChoice::Mc19,
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
        if let Some(v) = read_int("adaptive_mu_kkterror_red_iters") {
            if v >= 0 {
                builder.mu.adaptive_mu_kkterror_red_iters = v as usize;
            }
        }
        if let Some(v) = read_num("adaptive_mu_kkterror_red_fact") {
            builder.mu.adaptive_mu_kkterror_red_fact = v;
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
        if let Ok((v, found)) = self.options.get_bool_value("warm_start_same_structure", "") {
            if found {
                builder.warm.same_structure = v;
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
        if let Ok((v, found)) = self
            .options
            .get_string_value("warm_start_entire_iterate", "")
        {
            if found {
                builder.warm.entire_iterate = v == "yes";
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

/// Default symmetric linear-solver factory, parameterized by the
/// pounce-extension FERAL knobs read off the application's
/// `OptionsList`.
///
/// FERAL (pure-Rust) is the shipping default. The HSL MA57 backend is
/// available when the `ma57` cargo feature is enabled; without it,
/// requesting `linear_solver = ma57` falls back to FERAL with a
/// warning printed by the journalist (see [`AlgorithmBuilder`]).
pub fn default_backend_factory(feral_cfg: pounce_feral::FeralConfig) -> LinearBackendFactory {
    Box::new(
        move |choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
            match choice {
                LinearSolverChoice::Feral => Box::new(
                    pounce_feral::FeralSolverInterface::with_config(feral_cfg.clone()),
                ),
                LinearSolverChoice::Ma57 => {
                    #[cfg(feature = "ma57")]
                    {
                        Box::new(pounce_hsl::Ma57SolverInterface::new())
                    }
                    #[cfg(not(feature = "ma57"))]
                    {
                        // ma57 feature not compiled in — fall back to FERAL.
                        Box::new(pounce_feral::FeralSolverInterface::with_config(
                            feral_cfg.clone(),
                        ))
                    }
                }
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
    sink: Arc<Mutex<LinearSolverSummary>>,
) -> LinearBackendFactory {
    Box::new(
        move |choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
            match choice {
                LinearSolverChoice::Feral => Box::new(
                    pounce_feral::FeralSolverInterface::with_config(feral_cfg.clone())
                        .with_summary_sink(Arc::clone(&sink)),
                ),
                LinearSolverChoice::Ma57 => {
                    #[cfg(feature = "ma57")]
                    {
                        Box::new(pounce_hsl::Ma57SolverInterface::new())
                    }
                    #[cfg(not(feature = "ma57"))]
                    {
                        Box::new(
                            pounce_feral::FeralSolverInterface::with_config(feral_cfg.clone())
                                .with_summary_sink(Arc::clone(&sink)),
                        )
                    }
                }
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
    if let Ok((v, true)) = options.get_bool_value("feral_refine", "") {
        cfg.refine = v;
    }
    if let Ok((v, true)) = options.get_numeric_value("feral_singular_pivot_floor", "") {
        cfg.singular_pivot_floor = v;
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
fn finalize_via_orig_nlp(
    nlp: &Rc<RefCell<dyn IpoptNlp>>,
    alg: &IpoptAlgorithm,
    solver_status: SolverReturn,
    _app_status: ApplicationReturnStatus,
    tnlp: &Rc<RefCell<dyn TNLP>>,
) -> Result<Number, ()> {
    let curr = alg.data.borrow().curr.clone().ok_or(())?;
    // Lift compressed x_var → full-x (length `info.n`) so the user
    // TNLP receives the same shape it provided. With `make_parameter`
    // the fixed components are spliced back in by the IpoptNlp.
    let nlp_borrow = nlp.borrow();
    let x_vec: Vec<Number> = nlp_borrow.lift_x_to_full(&*curr.x);
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
    tnlp.borrow_mut().finalize_solution(
        Solution {
            status: solver_status,
            x: &x_vec,
            z_l: &z_l,
            z_u: &z_u,
            g: &g_final,
            lambda: &lambda,
            obj_value: f_final,
        },
        &TnlpIpoptData::default(),
        &TnlpIpoptCq::default(),
    );
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
    let x_vec: Vec<Number> = nlp_borrow.lift_x_to_full(&x_dv);
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
    tnlp.borrow_mut().finalize_solution(
        pounce_nlp::tnlp::Solution {
            status: solver_status,
            x: &x_vec,
            z_l: &z_l,
            z_u: &z_u,
            g: &g_final,
            lambda: &lambda,
            obj_value: f_final,
        },
        &TnlpIpoptData::default(),
        &TnlpIpoptCq::default(),
    );
    Ok(f_final)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_nlp::tnlp::{
        BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest,
        StartingPoint,
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
}
