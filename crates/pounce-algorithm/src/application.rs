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
/// `pounce-sensitivity`). The callback receives a mutable reference
/// to the PD solver so a `SensBacksolver` adapter can run backsolves
/// against the converged KKT factor; receives the data / cq / nlp
/// handles so the adapter can reproduce the augmented-system
/// coefficient layout the IPM converged at.
///
/// **Not** the same as `set_intermediate_callback` (per-iteration
/// progress notification) — this fires exactly once per `optimize_*`
/// call, only on success.
pub type ConvergedCallback = Box<
    dyn FnMut(
        &crate::ipopt_data::IpoptDataHandle,
        &crate::ipopt_cq::IpoptCqHandle,
        &Rc<RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>,
        &mut crate::kkt::pd_full_space_solver::PdFullSpaceSolver,
    ),
>;
use pounce_common::diagnostics::DiagnosticsState;
use pounce_common::exception::{ExceptionKind, SolverException};
use pounce_common::journalist::{JournalLevel, Journalist};
use pounce_common::options_list::OptionsList;
use pounce_common::reg_options::RegisteredOptions;
use pounce_common::timing::TimingStatistics;
use pounce_common::types::{Index, Number};
use pounce_linalg::dense_vector::DenseVectorSpace;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::orig_ipopt_nlp::{NoScaling, OrigIpoptNlp, ScalingMethod};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{
    IpoptCq as TnlpIpoptCq, IpoptData as TnlpIpoptData, NlpInfo, Solution, TNLP,
};
use pounce_nlp::tnlp_adapter::TNLPAdapter;
use std::cell::RefCell;
use std::fmt;
use std::path::Path;
use std::rc::Rc;
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
            restoration_factory_provider: None,
            on_converged: None,
            record_iter_history: false,
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

    /// **Stub.** Re-solve with a warm start. Phase 7+.
    pub fn reoptimize_tnlp(&mut self, tnlp: Rc<RefCell<dyn TNLP>>) -> ApplicationReturnStatus {
        // Same dispatch as `optimize_tnlp` for now; warm-start handling
        // lands once the IPM path's warm-start hooks are exposed.
        self.optimize_tnlp(tnlp)
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

        // Mint a fresh `TimingStatistics` for this solve — shared (via
        // `Rc`) with the data and the NLP below so every `eval_*` and
        // every iterate-phase records into the same accumulator. The
        // application keeps its own `Rc` so callers can read totals out
        // via [`Self::timing_stats`].
        let timing = Rc::new(TimingStatistics::new());
        *self.timing.borrow_mut() = Rc::clone(&timing);
        timing.overall_alg.start();

        // Build adapter + Nlp.
        let adapter = match TNLPAdapter::new(Rc::clone(&tnlp)) {
            Ok(a) => Rc::new(RefCell::new(a)),
            Err(_) => {
                timing.overall_alg.end();
                return ApplicationReturnStatus::InvalidProblemDefinition;
            }
        };
        let mut orig_nlp = match OrigIpoptNlp::new(Rc::clone(&adapter), Rc::new(NoScaling)) {
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
            // user-scaling / equilibration not yet implemented; fall back
            // to gradient-based which matches the upstream default.
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
        orig_nlp.determine_scaling_from_starting_point(scaling_method, max_gradient, min_value);

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
        let factory = self
            .linear_backend_factory
            .take()
            .unwrap_or_else(|| default_backend_factory(feral_cfg));
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
        alg.record_iter_history = self.record_iter_history;
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

        let solver_status = alg.optimize();
        // Close the overall-algorithm timer on the success path. The
        // early-return arms above end it themselves before bailing out;
        // this one matches upstream `IpoptApplication::call_optimize`
        // (which calls `EndCpuTime()` on overall_alg right after
        // `Optimize` returns, regardless of solver_status).
        timing.overall_alg.end();

        // Drain counters / iter count off the algorithm.
        {
            let mut stats = self.statistics.borrow_mut();
            stats.iteration_count = alg.data.borrow().iter_count;
            stats.total_wallclock_time_secs = t_start.elapsed().as_secs_f64();
            // Restoration-phase audit counters (pounce#12). Zero on
            // problems where restoration never fires; populated by
            // `IpoptAlgorithm::invoke_restoration`.
            stats.restoration_calls = alg.resto_calls;
            stats.restoration_inner_iters = alg.resto_inner_iters;
            stats.restoration_outer_iters = alg.resto_outer_iters;
            stats.restoration_wall_secs = alg.resto_wall_secs;
            stats.iterations = std::mem::take(&mut alg.iter_history);
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
                    let pd = sd.pd_solver_mut();
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

    fn algorithm_builder_from_options(&self) -> AlgorithmBuilder {
        let mut builder = AlgorithmBuilder::new();
        if let Ok((v, found)) = self.options.get_string_value("mu_strategy", "") {
            if found {
                builder.mu_strategy = match v.as_str() {
                    "adaptive" => MuStrategyChoice::Adaptive,
                    _ => MuStrategyChoice::Monotone,
                };
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
        if let Some(v) = read_num("sigma_max") {
            builder.mu.sigma_max = v;
        }
        if let Some(v) = read_num("sigma_min") {
            builder.mu.sigma_min = v;
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
        builder
    }
}

/// Map the integer `print_level` / `file_print_level` option to the
/// matching [`JournalLevel`] variant. Mirrors upstream's
/// `static_cast<EJournalLevel>(int_value)` with clamping.
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
                LinearSolverChoice::Feral => {
                    Box::new(pounce_feral::FeralSolverInterface::with_config(feral_cfg))
                }
                LinearSolverChoice::Ma57 => {
                    #[cfg(feature = "ma57")]
                    {
                        Box::new(pounce_hsl::Ma57SolverInterface::new())
                    }
                    #[cfg(not(feature = "ma57"))]
                    {
                        // ma57 feature not compiled in — fall back to FERAL.
                        Box::new(pounce_feral::FeralSolverInterface::with_config(feral_cfg))
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
    if let Ok((v, true)) = options.get_bool_value("feral_cascade_break", "") {
        cfg.cascade_break = v;
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
    // Backends without overrides return empty; fall back to zero stubs
    // so the user sees a length-consistent vector.
    let mut z_l = nlp_borrow.pack_z_l_for_user(&*curr.z_l);
    if z_l.is_empty() {
        z_l = vec![0.0; n];
    }
    let mut z_u = nlp_borrow.pack_z_u_for_user(&*curr.z_u);
    if z_u.is_empty() {
        z_u = vec![0.0; n];
    }
    let mut lambda = nlp_borrow.pack_lambda_for_user(&*curr.y_c, &*curr.y_d);
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
