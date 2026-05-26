//! Algorithm builder — port of `Algorithm/IpAlgBuilder.{hpp,cpp}`.
//!
//! Reads `OptionsList`, walks the dependency order documented in
//! `ref/Ipopt/AGENT_REFERENCE/ARCHITECTURE.md` §"BuildBasicAlgorithm",
//! and assembles the strategy objects needed by `IpoptAlgorithm`:
//!
//! * `SymLinearSolver` (MA57 / MUMPS / FERAL) → `AugSystemSolver`
//!   (`StdAugSystemSolver`) → `PdSystemSolver` (`PdFullSpaceSolver`)
//!   → `SearchDirCalculator` (`PdSearchDirCalc`).
//! * `BacktrackingLsAcceptor` (filter / penalty / cg-penalty) →
//!   `BacktrackingLineSearch`.
//! * `MuUpdate` (monotone / adaptive[+oracle]).
//! * `ConvCheck` (`OptErrorConvCheck`).
//! * `IterateInitializer` (default / warm-start) and
//!   `EqMultCalculator` (`LeastSquareMults`).
//! * `HessianUpdater` (exact / limited-memory).
//! * `IterationOutput` (`OrigIterationOutput`).
//! * `NLPScalingObject` (none / user / gradient-based / equilibration-based).
//!
//! Phase 7 ships the option-driven dispatch surface; the assembled
//! `IpoptAlgorithm` lands once each strategy's arithmetic does.

use crate::conv_check::opt_error::OptErrorConvCheck;
use crate::eq_mult::least_square::LeastSquareMults;
use crate::hess::exact::ExactHessianUpdater;
use crate::hess::lim_mem_quasi_newton::{LimMemQuasiNewtonUpdater, UpdateType};
use crate::init::default::DefaultIterateInitializer;
use crate::init::warm_start::WarmStartIterateInitializer;
use crate::kkt::pd_full_space_solver::PdFullSpaceSolver;
use crate::kkt::pd_search_dir_calc::PdSearchDirCalc;
use crate::kkt::perturbation_handler::PdPerturbationHandler;
use crate::kkt::std_aug_system_solver::StdAugSystemSolver;
use crate::line_search::backtracking::BacktrackingLineSearch;
use crate::line_search::filter_acceptor::FilterLsAcceptor;
use crate::line_search::ls_acceptor::BacktrackingLsAcceptor;
use crate::line_search::penalty_acceptor::PenaltyLsAcceptor;
use crate::mu::adaptive::{AdaptiveMuUpdate, MuOracleKind};
use crate::mu::monotone::MonotoneMuUpdate;
use crate::output::orig::OrigIterationOutput;
use pounce_common::types::{Index, Number};
use pounce_linsol::{SparseSymLinearSolverInterface, TSymLinearSolver};
use std::cell::RefCell;
use std::rc::Rc;

/// Backend factory — the application supplies one before calling
/// [`AlgorithmBuilder::build`]. Mirrors upstream's
/// `SymLinearSolverFactory` knob in `IpAlgBuilder.cpp`. The default
/// factory wires in FERAL; MA57 is selectable when the `ma57` cargo
/// feature is enabled.
pub type LinearBackendFactory =
    Box<dyn FnMut(LinearSolverChoice) -> Box<dyn SparseSymLinearSolverInterface>>;

/// Top-level algorithm choice. `InteriorPoint` is pounce's default
/// (the existing `IpoptAlgorithm`); `ActiveSetSqp` is the
/// Phase 5b SQP driver in `crate::sqp::SqpAlgorithm`, which uses
/// `pounce-qp` for QP subproblem solves and reuses
/// `FilterLsAcceptor` for globalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlgorithmChoice {
    #[default]
    InteriorPoint,
    ActiveSetSqp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverChoice {
    Ma57,
    Feral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuStrategyChoice {
    Monotone,
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HessianApproxChoice {
    Exact,
    LimitedMemory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineSearchChoice {
    Filter,
    CgPenalty,
    Penalty,
}

/// Assembled strategy bundle. Phase 7 ships the structural bundle;
/// `IpoptAlgorithm::new` reads from this when it lands.
pub struct AlgorithmBundle {
    pub mu_update: Box<dyn crate::mu::r#trait::MuUpdate>,
    pub conv_check: Box<dyn crate::conv_check::r#trait::ConvCheck>,
    pub init: Box<dyn crate::init::r#trait::IterateInitializer>,
    pub eq_mult: Box<dyn crate::eq_mult::r#trait::EqMultCalculator>,
    pub hess: Box<dyn crate::hess::r#trait::HessianUpdater>,
    pub line_search: BacktrackingLineSearch,
    pub iter_output: Box<dyn crate::output::r#trait::IterationOutput>,
    /// `Some` when the builder was given a [`LinearBackendFactory`];
    /// `None` for the bare structural bundle that pre-Phase-6 unit
    /// tests still rely on.
    pub search_dir: Option<PdSearchDirCalc>,
}

/// Knobs read off `OptionsList` and baked into the assembled
/// `OptErrorConvCheck`. Defaults mirror
/// `IpOptErrorConvCheck.cpp:RegisterOptions`.
#[derive(Debug, Clone)]
pub struct ConvCheckOptions {
    pub tol: Number,
    pub dual_inf_tol: Number,
    pub constr_viol_tol: Number,
    pub compl_inf_tol: Number,
    pub acceptable_tol: Number,
    pub acceptable_dual_inf_tol: Number,
    pub acceptable_constr_viol_tol: Number,
    pub acceptable_compl_inf_tol: Number,
    pub acceptable_obj_change_tol: Number,
    pub acceptable_iter: Index,
    pub max_iter: Index,
    pub max_cpu_time: Number,
    pub max_wall_time: Number,
    pub infeas_stationarity_tol: Number,
    pub infeas_viol_kappa: Number,
    pub infeas_max_streak: Index,
}

impl Default for ConvCheckOptions {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            dual_inf_tol: 1.0,
            constr_viol_tol: 1e-4,
            compl_inf_tol: 1e-4,
            acceptable_tol: 1e-6,
            acceptable_dual_inf_tol: 1e10,
            acceptable_constr_viol_tol: 1e-2,
            acceptable_compl_inf_tol: 1e-2,
            acceptable_obj_change_tol: 1e20,
            acceptable_iter: 15,
            max_iter: 3000,
            max_cpu_time: 1e6,
            max_wall_time: 1e6,
            infeas_stationarity_tol: 1e-8,
            infeas_viol_kappa: 1e2,
            infeas_max_streak: 5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlgorithmBuilder {
    /// Top-level algorithm dispatch. Default `InteriorPoint` ⇒
    /// `build_with_backend` returns the existing `AlgorithmBundle`
    /// (consumed by `IpoptAlgorithm`). `ActiveSetSqp` ⇒ caller
    /// must use `build_sqp_with_backend` to assemble the Phase 5b
    /// `SqpAlgorithm`. The two builder methods sit side by side
    /// because the assembled algorithm shape differs (IPM bundle
    /// vs SQP struct).
    pub algorithm: AlgorithmChoice,
    pub linear_solver: LinearSolverChoice,
    pub mu_strategy: MuStrategyChoice,
    /// Selector forwarded to [`AdaptiveMuUpdate`] when
    /// `mu_strategy = Adaptive`. Ignored for `Monotone`. Defaults to
    /// `QualityFunction` per upstream's `RegisterOptions` default.
    pub mu_oracle: MuOracleKind,
    pub hessian_approximation: HessianApproxChoice,
    pub limited_memory_update_type: UpdateType,
    pub line_search_method: LineSearchChoice,
    pub warm_start_init_point: bool,
    /// `mehrotra_algorithm` — when true, [`PdSearchDirCalc`] folds
    /// the Mehrotra second-order complementarity term into the
    /// search-direction RHS. Mirrors upstream's
    /// `IpAlgBuilder.cpp:Mehrotra` flag. Requires `mu_strategy =
    /// Adaptive` so that an affine step is computed each iteration;
    /// [`Self::build_with_backend`] does not enforce this — the
    /// option-parser in `application.rs` is responsible for the
    /// cascading defaults (`mu_oracle = probing` etc.).
    pub mehrotra_algorithm: bool,
    pub conv_check: ConvCheckOptions,
    pub mu: MuOptions,
    pub line_search: LineSearchOptions,
    pub output: OutputOptions,
    pub warm: WarmStartOptions,
    /// SQP-specific options (consulted only when
    /// `algorithm = ActiveSetSqp`).
    pub sqp: crate::sqp::SqpOptions,
    pub init: InitOptions,
}

/// Knobs read off `OptionsList` and baked into
/// [`DefaultIterateInitializer`]. Defaults mirror
/// `IpDefaultIterateInitializer.cpp:RegisterOptions`. The Mehrotra
/// cascade in `application.rs` overrides `bound_push`, `bound_frac`,
/// and `bound_mult_init_val` to upstream's more-aggressive values
/// (`10`, `0.2`, `1.0`).
#[derive(Debug, Clone)]
pub struct InitOptions {
    pub bound_push: Number,
    pub bound_frac: Number,
    pub slack_bound_push: Number,
    pub slack_bound_frac: Number,
    pub constr_mult_init_max: Number,
    pub bound_mult_init_val: Number,
    /// `bound_mult_init_method`: `"constant"` (default) or `"mu-based"`
    /// (matches upstream's `IpDefaultIterateInitializer.cpp`).
    pub bound_mult_init_method: String,
}

impl Default for InitOptions {
    fn default() -> Self {
        Self {
            bound_push: 1e-2,
            bound_frac: 1e-2,
            slack_bound_push: 1e-2,
            slack_bound_frac: 1e-2,
            constr_mult_init_max: 1e3,
            bound_mult_init_val: 1.0,
            bound_mult_init_method: "constant".into(),
        }
    }
}

/// Knobs read off `OptionsList` and baked into
/// [`WarmStartIterateInitializer`]. Defaults mirror
/// `IpWarmStartIterateInitializer.cpp:RegisterOptions`.
///
/// Wired today: `mult_init_max` (clamps |y_c|, |y_d| and caps z/v
/// blocks) and `target_mu` (overrides `data.curr_mu` at iter 0).
/// The remaining knobs (`bound_push`, `bound_frac`, `slack_bound_push`,
/// `slack_bound_frac`, `mult_bound_push`, `entire_iterate`,
/// `same_structure`) are stored on the initializer but not yet
/// consumed — `WarmStartIterateInitializer::set_initial_iterates`
/// currently trusts the caller-populated `data.curr` rather than
/// re-running the upstream `push_variables` machinery.
#[derive(Debug, Clone)]
pub struct WarmStartOptions {
    pub bound_push: Number,
    pub bound_frac: Number,
    pub slack_bound_push: Number,
    pub slack_bound_frac: Number,
    pub mult_bound_push: Number,
    pub mult_init_max: Number,
    pub target_mu: Number,
    pub entire_iterate: bool,
    pub same_structure: bool,
}

impl Default for WarmStartOptions {
    fn default() -> Self {
        Self {
            bound_push: 1e-3,
            bound_frac: 1e-3,
            slack_bound_push: 1e-3,
            slack_bound_frac: 1e-3,
            mult_bound_push: 1e-3,
            mult_init_max: 1e6,
            target_mu: 0.0,
            entire_iterate: false,
            same_structure: false,
        }
    }
}

/// Knobs read off `OptionsList` and baked into the assembled
/// `MonotoneMuUpdate` or `AdaptiveMuUpdate`. Defaults mirror
/// `IpMonotoneMuUpdate.cpp` / `IpAdaptiveMuUpdate.cpp:RegisterOptions`.
/// `mu_max` defaults to the sentinel `-1`; positive values are baked
/// into both updaters at build time (adaptive interprets `-1` as
/// "lazy-init from `mu_max_fact * avrg_compl`").
#[derive(Debug, Clone)]
pub struct MuOptions {
    pub mu_init: Number,
    pub mu_max: Number,
    pub mu_max_fact: Number,
    pub mu_min: Number,
    pub mu_target: Number,
    pub mu_linear_decrease_factor: Number,
    pub mu_superlinear_decrease_power: Number,
    pub mu_allow_fast_monotone_decrease: bool,
    pub barrier_tol_factor: Number,
    /// `sigma_max` / `sigma_min` — clamp on the centering parameter σ
    /// chosen by `QualityFunctionMuOracle`. Only consumed when
    /// `mu_strategy=adaptive` and `mu_oracle=quality-function`.
    /// Defaults from `IpQualityFunctionMuOracle.cpp:RegisterOptions`.
    pub sigma_max: Number,
    pub sigma_min: Number,
}

impl Default for MuOptions {
    fn default() -> Self {
        Self {
            mu_init: 0.1,
            mu_max: -1.0,
            mu_max_fact: 1e3,
            mu_min: 1e-11,
            mu_target: 0.0,
            mu_linear_decrease_factor: 0.2,
            mu_superlinear_decrease_power: 1.5,
            mu_allow_fast_monotone_decrease: true,
            barrier_tol_factor: 10.0,
            sigma_max: 1e2,
            sigma_min: 1e-6,
        }
    }
}

/// Knobs baked into the assembled [`BacktrackingLineSearch`]. Defaults
/// mirror `IpBacktrackingLineSearch.cpp:RegisterOptions`.
#[derive(Debug, Clone)]
pub struct LineSearchOptions {
    pub watchdog_shortened_iter_trigger: Index,
    pub watchdog_trial_iter_max: Index,
    /// `soft_resto_pderror_reduction_factor` — required relative
    /// reduction in the primal-dual error for a soft-resto step.
    /// `0` disables the soft restoration phase.
    pub soft_resto_pderror_reduction_factor: Number,
    /// `max_soft_resto_iters` — cap on consecutive soft-resto
    /// iterations before full restoration is forced.
    pub max_soft_resto_iters: Index,
    /// `accept_every_trial_step` — short-circuits the filter / alpha
    /// loop and accepts the full fraction-to-the-boundary step every
    /// outer iteration. Mirrors upstream's
    /// `IpBacktrackingLineSearch::accept_every_trial_step_`. Drops
    /// global convergence guarantees; only safe for problems where the
    /// Newton step is already a descent step (LPs, convex QPs). The
    /// Mehrotra cascade in `application.rs` flips this on.
    pub accept_every_trial_step: bool,
    /// `alpha_for_y` — policy for the equality-multiplier (y_c / y_d)
    /// step length. Upstream default is `Primal`; the Mehrotra cascade
    /// switches to `BoundMult`.
    pub alpha_for_y: crate::line_search::backtracking::AlphaForY,
}

impl Default for LineSearchOptions {
    fn default() -> Self {
        Self {
            watchdog_shortened_iter_trigger: 10,
            watchdog_trial_iter_max: 3,
            soft_resto_pderror_reduction_factor: 1.0 - 1e-4,
            max_soft_resto_iters: 10,
            accept_every_trial_step: false,
            alpha_for_y: crate::line_search::backtracking::AlphaForY::Primal,
        }
    }
}

/// Knobs baked into the assembled [`OrigIterationOutput`]. Defaults
/// mirror `IpOrigIterationOutput.cpp:RegisterOptions` /
/// `IpAlgorithmRegOp.cpp`.
#[derive(Debug, Clone)]
pub struct OutputOptions {
    pub print_frequency_iter: Index,
    pub print_frequency_time: Number,
    /// `print_info_string` (default `false`). When on, the iter row
    /// ends with the contents of `IpoptData::info_string` so users
    /// can read the per-iteration diagnostic tags.
    pub print_info_string: bool,
    /// `inf_pr_output` — `"original"` (default) prints the unscaled
    /// NLP primal infeasibility; `"internal"` prints the internal
    /// reformulated violation. Only meaningful once NLP-side scaling
    /// is in play; until then both modes produce the same number.
    pub inf_pr_output_internal: bool,
}

impl Default for OutputOptions {
    fn default() -> Self {
        Self {
            print_frequency_iter: 1,
            print_frequency_time: 0.0,
            print_info_string: false,
            inf_pr_output_internal: false,
        }
    }
}

impl Default for AlgorithmBuilder {
    fn default() -> Self {
        Self {
            algorithm: AlgorithmChoice::default(),
            linear_solver: LinearSolverChoice::Feral,
            mu_strategy: MuStrategyChoice::Monotone,
            mu_oracle: MuOracleKind::QualityFunction,
            hessian_approximation: HessianApproxChoice::Exact,
            limited_memory_update_type: UpdateType::Bfgs,
            line_search_method: LineSearchChoice::Filter,
            warm_start_init_point: false,
            mehrotra_algorithm: false,
            conv_check: ConvCheckOptions::default(),
            mu: MuOptions::default(),
            line_search: LineSearchOptions::default(),
            output: OutputOptions::default(),
            warm: WarmStartOptions::default(),
            sqp: crate::sqp::SqpOptions::default(),
            init: InitOptions::default(),
        }
    }
}

impl AlgorithmBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Assemble the strategy bundle without a search-direction
    /// calculator. Used by structural unit tests that don't want to
    /// pull in a linear-solver backend.
    pub fn build(&self) -> AlgorithmBundle {
        self.build_inner(None)
    }

    /// Same as [`Self::build`] but also constructs the
    /// `SymLinearSolver → AugSystemSolver → PdFullSpaceSolver →
    /// PdSearchDirCalc` chain via the supplied `factory`.
    pub fn build_with_backend(&self, mut factory: LinearBackendFactory) -> AlgorithmBundle {
        let backend = factory(self.linear_solver);
        let linsol = TSymLinearSolver::new(backend, None, false);
        let aug_solver = StdAugSystemSolver::new(linsol);
        let perturb = Rc::new(RefCell::new(PdPerturbationHandler::new()));
        let pd_solver = PdFullSpaceSolver::new(Box::new(aug_solver), perturb);
        let mut search_dir = PdSearchDirCalc::new(pd_solver);
        search_dir.mehrotra_algorithm = self.mehrotra_algorithm;
        self.build_inner(Some(search_dir))
    }

    /// Phase 5b assembly path for the SQP algorithm. Consults
    /// `self.algorithm`: when `ActiveSetSqp`, constructs an
    /// `SqpAlgorithm` using the supplied backend factory for the
    /// QP subproblem solver; otherwise returns `None` so the
    /// caller can fall back to the IPM `build_with_backend`.
    ///
    /// Sister to `build_with_backend`: the SQP algorithm doesn't
    /// share `AlgorithmBundle`'s shape (no mu_update / no IPM
    /// line search), so the two paths return different types.
    pub fn build_sqp_with_backend(
        &self,
        mut factory: LinearBackendFactory,
    ) -> Option<crate::sqp::SqpAlgorithm> {
        if !matches!(self.algorithm, AlgorithmChoice::ActiveSetSqp) {
            return None;
        }
        let backend = factory(self.linear_solver);
        let qp_solver = pounce_qp::ParametricActiveSetSolver::new(backend);
        Some(crate::sqp::SqpAlgorithm::new(qp_solver, self.sqp.clone()))
    }

    fn build_inner(&self, search_dir: Option<PdSearchDirCalc>) -> AlgorithmBundle {
        let mu_update: Box<dyn crate::mu::r#trait::MuUpdate> = match self.mu_strategy {
            MuStrategyChoice::Monotone => {
                let mut m = MonotoneMuUpdate::new();
                m.mu_init = self.mu.mu_init;
                // `mu_max` sentinel `-1` keeps the monotone default
                // (1e5); only override on a user-supplied positive.
                if self.mu.mu_max > 0.0 {
                    m.mu_max = self.mu.mu_max;
                }
                m.mu_min = self.mu.mu_min;
                m.mu_target = self.mu.mu_target;
                m.mu_linear_decrease_factor = self.mu.mu_linear_decrease_factor;
                m.mu_superlinear_decrease_power = self.mu.mu_superlinear_decrease_power;
                m.mu_allow_fast_monotone_decrease = self.mu.mu_allow_fast_monotone_decrease;
                m.barrier_tol_factor = self.mu.barrier_tol_factor;
                m.compl_inf_tol = self.conv_check.compl_inf_tol;
                Box::new(m)
            }
            MuStrategyChoice::Adaptive => {
                let mut adaptive = AdaptiveMuUpdate::new();
                adaptive.mu_oracle = self.mu_oracle;
                adaptive.mu_init = self.mu.mu_init;
                // Adaptive treats `mu_max == -1` as "lazy init from
                // `mu_max_fact * curr_avrg_compl`" — forward the
                // sentinel as-is.
                adaptive.mu_max = self.mu.mu_max;
                adaptive.mu_max_fact = self.mu.mu_max_fact;
                adaptive.mu_min = self.mu.mu_min;
                adaptive.mu_linear_decrease_factor = self.mu.mu_linear_decrease_factor;
                adaptive.mu_superlinear_decrease_power = self.mu.mu_superlinear_decrease_power;
                adaptive.barrier_tol_factor = self.mu.barrier_tol_factor;
                adaptive.sigma_min = self.mu.sigma_min;
                adaptive.sigma_max = self.mu.sigma_max;
                Box::new(adaptive)
            }
        };

        let acceptor: Box<dyn BacktrackingLsAcceptor> = match self.line_search_method {
            LineSearchChoice::Filter => Box::new(FilterLsAcceptor::default()),
            LineSearchChoice::Penalty => Box::new(PenaltyLsAcceptor::default()),
            // CG-penalty acceptor lands with the rest of the
            // CG-penalty path; fall back to the penalty acceptor's
            // surface for now.
            LineSearchChoice::CgPenalty => Box::new(PenaltyLsAcceptor::default()),
        };
        let mut line_search = BacktrackingLineSearch::new(acceptor);
        line_search.watchdog_shortened_iter_trigger =
            self.line_search.watchdog_shortened_iter_trigger;
        line_search.watchdog_trial_iter_max = self.line_search.watchdog_trial_iter_max;
        line_search.soft_resto_pderror_reduction_factor =
            self.line_search.soft_resto_pderror_reduction_factor;
        line_search.max_soft_resto_iters = self.line_search.max_soft_resto_iters;
        line_search.accept_every_trial_step = self.line_search.accept_every_trial_step;
        line_search.alpha_for_y = self.line_search.alpha_for_y;

        let conv_check: Box<dyn crate::conv_check::r#trait::ConvCheck> =
            Box::new(OptErrorConvCheck {
                tol: self.conv_check.tol,
                dual_inf_tol: self.conv_check.dual_inf_tol,
                constr_viol_tol: self.conv_check.constr_viol_tol,
                compl_inf_tol: self.conv_check.compl_inf_tol,
                acceptable_tol: self.conv_check.acceptable_tol,
                acceptable_dual_inf_tol: self.conv_check.acceptable_dual_inf_tol,
                acceptable_constr_viol_tol: self.conv_check.acceptable_constr_viol_tol,
                acceptable_compl_inf_tol: self.conv_check.acceptable_compl_inf_tol,
                acceptable_obj_change_tol: self.conv_check.acceptable_obj_change_tol,
                acceptable_iter: self.conv_check.acceptable_iter,
                max_iter: self.conv_check.max_iter,
                max_cpu_time: self.conv_check.max_cpu_time,
                max_wall_time: self.conv_check.max_wall_time,
                acceptable_count: 0,
                last_acceptable_obj: None,
                infeas_stationarity_tol: self.conv_check.infeas_stationarity_tol,
                infeas_viol_kappa: self.conv_check.infeas_viol_kappa,
                infeas_max_streak: self.conv_check.infeas_max_streak,
                infeas_streak: 0,
            });

        let init: Box<dyn crate::init::r#trait::IterateInitializer> = if self.warm_start_init_point
        {
            Box::new(WarmStartIterateInitializer::with_options(self.warm.clone()))
        } else {
            let mut d = DefaultIterateInitializer::with_eq_mult_calculator(Box::new(
                LeastSquareMults::new(),
            ));
            d.bound_push = self.init.bound_push;
            d.bound_frac = self.init.bound_frac;
            d.slack_bound_push = self.init.slack_bound_push;
            d.slack_bound_frac = self.init.slack_bound_frac;
            d.constr_mult_init_max = self.init.constr_mult_init_max;
            d.bound_mult_init_val = self.init.bound_mult_init_val;
            d.bound_mult_init_method = self.init.bound_mult_init_method.clone();
            Box::new(d)
        };

        let eq_mult: Box<dyn crate::eq_mult::r#trait::EqMultCalculator> =
            Box::new(LeastSquareMults::new());

        let hess: Box<dyn crate::hess::r#trait::HessianUpdater> = match self.hessian_approximation {
            HessianApproxChoice::Exact => Box::new(ExactHessianUpdater::new()),
            HessianApproxChoice::LimitedMemory => Box::new(LimMemQuasiNewtonUpdater {
                update_type: self.limited_memory_update_type,
                ..LimMemQuasiNewtonUpdater::default()
            }),
        };

        let iter_output: Box<dyn crate::output::r#trait::IterationOutput> = {
            use crate::output::orig::{InfPrTag, PrintInfoString};
            let mut o = OrigIterationOutput::new();
            o.print_frequency_iter = self.output.print_frequency_iter;
            o.print_frequency_time = self.output.print_frequency_time;
            o.print_info_string = if self.output.print_info_string {
                PrintInfoString::Yes
            } else {
                PrintInfoString::No
            };
            o.inf_pr_output = if self.output.inf_pr_output_internal {
                InfPrTag::Internal
            } else {
                InfPrTag::Original
            };
            Box::new(o)
        };

        AlgorithmBundle {
            mu_update,
            conv_check,
            init,
            eq_mult,
            hess,
            line_search,
            iter_output,
            search_dir,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_builder_assembles() {
        let bundle = AlgorithmBuilder::new().build();
        // Sanity: the placeholder traits compile and the boxed
        // strategies don't panic on construction.
        let _ = bundle.line_search.acceptor();
        assert!(bundle.search_dir.is_none());
    }

    #[test]
    fn build_with_backend_assembles_search_dir_chain() {
        // Drive the builder with the FERAL backend factory; the
        // resulting bundle should expose a populated `PdSearchDirCalc`.
        let factory: LinearBackendFactory = Box::new(|_| {
            Box::new(pounce_feral::FeralSolverInterface::new())
                as Box<dyn SparseSymLinearSolverInterface>
        });
        let bundle = AlgorithmBuilder::new().build_with_backend(factory);
        assert!(bundle.search_dir.is_some());
    }

    #[test]
    fn limited_memory_sr1_propagates() {
        let b = AlgorithmBuilder {
            hessian_approximation: HessianApproxChoice::LimitedMemory,
            limited_memory_update_type: UpdateType::Sr1,
            ..AlgorithmBuilder::default()
        };
        let _bundle = b.build();
    }

    #[test]
    fn every_strategy_combination_assembles_without_panic() {
        let solvers = [LinearSolverChoice::Ma57, LinearSolverChoice::Feral];
        let mu = [MuStrategyChoice::Monotone, MuStrategyChoice::Adaptive];
        let hess = [
            HessianApproxChoice::Exact,
            HessianApproxChoice::LimitedMemory,
        ];
        let ls = [
            LineSearchChoice::Filter,
            LineSearchChoice::CgPenalty,
            LineSearchChoice::Penalty,
        ];
        for &linear_solver in &solvers {
            for &mu_strategy in &mu {
                for &hessian_approximation in &hess {
                    for &line_search_method in &ls {
                        let _ = AlgorithmBuilder {
                            algorithm: AlgorithmChoice::default(),
                            linear_solver,
                            mu_strategy,
                            mu_oracle: MuOracleKind::QualityFunction,
                            hessian_approximation,
                            limited_memory_update_type: UpdateType::Bfgs,
                            line_search_method,
                            warm_start_init_point: false,
                            mehrotra_algorithm: false,
                            conv_check: ConvCheckOptions::default(),
                            mu: MuOptions::default(),
                            line_search: LineSearchOptions::default(),
                            output: OutputOptions::default(),
                            warm: WarmStartOptions::default(),
                            sqp: crate::sqp::SqpOptions::default(),
                            init: InitOptions::default(),
                        }
                        .build();
                    }
                }
            }
        }
    }
}
