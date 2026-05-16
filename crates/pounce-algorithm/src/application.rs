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
//! `optimize_tnlp` dispatches:
//!
//! * `m == 0` (no constraints reported by the TNLP) → falls through
//!   to `pounce_nlp::newton_driver::solve` so unconstrained problems
//!   keep working without the full primal-dual stack.
//! * `m > 0` → builds the primal-dual algorithm via
//!   [`crate::alg_builder::AlgorithmBuilder`] (default backend MA57
//!   from `pounce-hsl`) and runs [`IpoptAlgorithm::optimize`].

use crate::alg_builder::{
    AlgorithmBuilder, HessianApproxChoice, LineSearchChoice, LinearBackendFactory,
    LinearSolverChoice, MuStrategyChoice, NlpScalingChoice,
};
use crate::upstream_options::register_all_upstream_options;
use crate::ipopt_alg::IpoptAlgorithm;
use crate::ipopt_cq::IpoptCalculatedQuantities;
use crate::ipopt_data::IpoptData as AlgIpoptData;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::IteratesVector;
use crate::restoration::RestorationPhase;

/// Factory that constructs a fresh restoration-phase strategy on
/// demand. The outer algorithm owns at most one restoration object,
/// so the factory is invoked once per `optimize_tnlp` call. The
/// factory is `FnMut` to allow callers to capture a builder that
/// internally reuses caches across builds.
pub type RestorationFactory = Box<dyn FnMut() -> Box<dyn RestorationPhase>>;
use pounce_linalg::dense_vector::DenseVectorSpace;
use pounce_common::exception::{ExceptionKind, SolverException};
use pounce_common::journalist::{JournalLevel, Journalist};
use pounce_common::options_list::OptionsList;
use pounce_common::reg_options::RegisteredOptions;
use pounce_common::timing::TimingStatistics;
use pounce_common::types::{Index, Number};
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::orig_ipopt_nlp::{NoScaling, OrigIpoptNlp, ScalingMethod};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{IpoptCq as TnlpIpoptCq, IpoptData as TnlpIpoptData, NlpInfo, Solution, TNLP};
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
        register_all_upstream_options(&reg).unwrap_or_else(|e| {
            panic!("Upstream options registration failed: {e}")
        });
        pounce_presolve::register_options(&reg).unwrap_or_else(|e| {
            panic!("Presolve options registration failed: {e}")
        });
        let reg = Rc::new(reg);
        Self {
            options: OptionsList::with_registered(Rc::clone(&reg)),
            reg_options: reg,
            journalist: Rc::new(Journalist::new()),
            statistics: RefCell::new(SolveStatistics::new()),
            timing: RefCell::new(Rc::new(TimingStatistics::new())),
            linear_backend_factory: None,
            restoration_factory: None,
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
        // Upstream Ipopt routes every problem through the same primal-dual
        // IPM regardless of `m` — there is no separate "unconstrained
        // Newton" path. Pounce historically dispatched `m == 0` to a dense
        // Newton driver, which built a full n×n Hessian and blew up on
        // anything large (e.g. bearing_400 with n = 160 000 → 25 GB
        // matrix). Route small unconstrained problems to that driver only
        // as a fast path; everything else goes through the sparse IPM so
        // the linear-solver backend (MA57/FERAL) handles the augmented
        // system instead of dense BLAS.
        if info.m == 0 && info.n <= 1000 {
            let mut borrow = tnlp.borrow_mut();
            let opts = self.newton_options_from_options_list();
            let (status, stats) = pounce_nlp::newton_driver::solve(&mut *borrow, opts);
            *self.statistics.borrow_mut() = stats;
            return status;
        }
        self.optimize_constrained(tnlp)
    }

    /// **Stub.** Re-solve with a warm start. Phase 7+.
    pub fn reoptimize_tnlp(&mut self, tnlp: Rc<RefCell<dyn TNLP>>) -> ApplicationReturnStatus {
        // Same dispatch as `optimize_tnlp` for now; warm-start handling
        // lands once the IPM path's warm-start hooks are exposed.
        self.optimize_tnlp(tnlp)
    }

    fn newton_options_from_options_list(&self) -> pounce_nlp::newton_driver::NewtonOptions {
        let mut opts = pounce_nlp::newton_driver::NewtonOptions::default();
        if let Ok((v, found)) = self.options.get_numeric_value("tol", "") {
            if found {
                opts.tol = v;
            }
        }
        if let Ok((v, found)) = self.options.get_integer_value("max_iter", "") {
            if found {
                opts.max_iter = v;
            }
        }
        opts
    }

    /// Constrained-NLP path: build adapter → OrigIpoptNlp → algorithm
    /// bundle, run `optimize`, populate statistics, and call
    /// `finalize_solution` on the user's TNLP.
    fn optimize_constrained(&mut self, tnlp: Rc<RefCell<dyn TNLP>>) -> ApplicationReturnStatus {
        let t_start = Instant::now();

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
        let data: crate::ipopt_data::IpoptDataHandle =
            Rc::new(RefCell::new(AlgIpoptData::new()));
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

        let mut alg = IpoptAlgorithm::new(data, cq, bundle).with_nlp(Rc::clone(&nlp_handle));
        if let Some(factory) = self.restoration_factory.as_mut() {
            alg = alg.with_restoration(factory());
        }
        alg.max_iter = max_iter;

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
            // Best-effort: capture final objective at the algorithm's
            // (compressed `x_var`-space) iterate via the NLP. The
            // fine-grained eval counters live on the concrete
            // `OrigIpoptNlp`; threading them up through a generic
            // `IpoptNlp` accessor is left for follow-up work.
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

        // Finalize: forward the final iterate to the user's TNLP.
        if let Err(()) = finalize_via_orig_nlp(&nlp_handle, &alg, solver_status, app_status, &tnlp)
        {
            // Couldn't finalize; still surface the original status.
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
        if let Ok((v, found)) = self.options.get_string_value("nlp_scaling_method", "") {
            if found {
                builder.nlp_scaling_method = match v.as_str() {
                    "none" => NlpScalingChoice::None,
                    "user-scaling" => NlpScalingChoice::User,
                    "equilibration-based" => NlpScalingChoice::EquilibrationBased,
                    _ => NlpScalingChoice::GradientBased,
                };
            }
        }
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

        // Watchdog options — consumers in
        // `IpBacktrackingLineSearch.cpp:RegisterOptions`. Baked into
        // the `BacktrackingLineSearch` at build time.
        if let Some(v) = read_int("watchdog_shortened_iter_trigger") {
            builder.line_search.watchdog_shortened_iter_trigger = v;
        }
        if let Some(v) = read_int("watchdog_trial_iter_max") {
            builder.line_search.watchdog_trial_iter_max = v;
        }

        // Iteration-output options — consumed by `OrigIterationOutput`.
        if let Some(v) = read_int("print_frequency_iter") {
            builder.output.print_frequency_iter = v;
        }
        if let Some(v) = read_num("print_frequency_time") {
            builder.output.print_frequency_time = v;
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
    Box::new(move |choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
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
    })
}

/// Read the three `feral_*` extension options off `options`, falling
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
        SolverReturn::TooFewDegreesOfFreedom => {
            ApplicationReturnStatus::NotEnoughDegreesOfFreedom
        }
        SolverReturn::InvalidOption => ApplicationReturnStatus::InvalidOption,
        SolverReturn::OutOfMemory => ApplicationReturnStatus::InsufficientMemory,
        SolverReturn::InternalError | SolverReturn::Unassigned => {
            ApplicationReturnStatus::InternalError
        }
    }
}

/// Best-effort evaluation of the objective at the algorithm's final
/// `x`. Used only to populate `SolveStatistics::final_objective`.
fn try_eval_curr_f(
    nlp: &Rc<RefCell<dyn IpoptNlp>>,
    x: &Rc<dyn pounce_linalg::Vector>,
) -> Result<Number, ()> {
    let mut nlp_mut = nlp.borrow_mut();
    Ok(nlp_mut.eval_f(&**x))
}

/// Forward the final iterate back to the user's `TNLP::finalize_solution`.
/// We pull `x` (compressed in `x_var`-space) off the algorithm's
/// `data.curr`, lift it back to full TNLP indexing, and pass empty
/// multipliers for now (the algorithm's `y_c`, `y_d`, `z_l`, `z_u` are
/// in compressed split form — re-assembling them into the user's
/// `lambda` / `z_l` / `z_u` is mechanical but lives behind a
/// `OrigIpoptNlp::finalize_solution_*` accessor that's still being
/// fleshed out). Returns `Err` if the final iterate is missing.
fn finalize_via_orig_nlp(
    nlp: &Rc<RefCell<dyn IpoptNlp>>,
    alg: &IpoptAlgorithm,
    solver_status: SolverReturn,
    _app_status: ApplicationReturnStatus,
    tnlp: &Rc<RefCell<dyn TNLP>>,
) -> Result<(), ()> {
    let curr = alg.data.borrow().curr.clone().ok_or(())?;
    // Lift compressed x_var → full-x (length `info.n`) so the user
    // TNLP receives the same shape it provided. With `make_parameter`
    // the fixed components are spliced back in by the IpoptNlp.
    let x_vec: Vec<Number> = nlp.borrow().lift_x_to_full(&*curr.x);
    let info = tnlp.borrow_mut().get_nlp_info().ok_or(())?;
    let n = info.n as usize;
    let m = info.m as usize;
    debug_assert_eq!(x_vec.len(), n);
    // For now we forward `x` only; the multiplier vectors come through
    // as zeros until `OrigIpoptNlp` ships its
    // `finalize_solution_lambda/z_l/z_u` accessors. Compute g(x) via
    // the user TNLP so the final residual is at least populated.
    let mut g_final = vec![0.0; m];
    let _ = tnlp.borrow_mut().eval_g(&x_vec, true, &mut g_final);
    let f_final = tnlp
        .borrow_mut()
        .eval_f(&x_vec, true)
        .unwrap_or(Number::NAN);
    let z_l = vec![0.0; n];
    let z_u = vec![0.0; n];
    let lambda = vec![0.0; m];
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
    Ok(())
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
        let (level, found) = app
            .options()
            .get_integer_value("print_level", "")
            .unwrap();
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
