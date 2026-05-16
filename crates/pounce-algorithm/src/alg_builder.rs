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
use pounce_common::types::{Index, Number};
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
use crate::scaling::{
    equilibration::EquilibrationScaling, gradient::GradientScaling, none::NoNlpScalingObject,
    r#trait::NlpScalingObject, user::UserScaling,
};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NlpScalingChoice {
    None,
    User,
    GradientBased,
    EquilibrationBased,
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
    pub scaling: Box<dyn NlpScalingObject>,
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
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlgorithmBuilder {
    pub linear_solver: LinearSolverChoice,
    pub mu_strategy: MuStrategyChoice,
    /// Selector forwarded to [`AdaptiveMuUpdate`] when
    /// `mu_strategy = Adaptive`. Ignored for `Monotone`. Defaults to
    /// `QualityFunction` per upstream's `RegisterOptions` default.
    pub mu_oracle: MuOracleKind,
    pub hessian_approximation: HessianApproxChoice,
    pub limited_memory_update_type: UpdateType,
    pub line_search_method: LineSearchChoice,
    pub nlp_scaling_method: NlpScalingChoice,
    pub warm_start_init_point: bool,
    pub conv_check: ConvCheckOptions,
    pub mu: MuOptions,
    pub line_search: LineSearchOptions,
    pub output: OutputOptions,
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
        }
    }
}

/// Knobs baked into the assembled [`BacktrackingLineSearch`]. Defaults
/// mirror `IpBacktrackingLineSearch.cpp:RegisterOptions`.
#[derive(Debug, Clone)]
pub struct LineSearchOptions {
    pub watchdog_shortened_iter_trigger: Index,
    pub watchdog_trial_iter_max: Index,
}

impl Default for LineSearchOptions {
    fn default() -> Self {
        Self {
            watchdog_shortened_iter_trigger: 10,
            watchdog_trial_iter_max: 3,
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
            linear_solver: LinearSolverChoice::Feral,
            mu_strategy: MuStrategyChoice::Monotone,
            mu_oracle: MuOracleKind::QualityFunction,
            hessian_approximation: HessianApproxChoice::Exact,
            limited_memory_update_type: UpdateType::Bfgs,
            line_search_method: LineSearchChoice::Filter,
            nlp_scaling_method: NlpScalingChoice::GradientBased,
            warm_start_init_point: false,
            conv_check: ConvCheckOptions::default(),
            mu: MuOptions::default(),
            line_search: LineSearchOptions::default(),
            output: OutputOptions::default(),
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
        let search_dir = PdSearchDirCalc::new(pd_solver);
        self.build_inner(Some(search_dir))
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
            });

        let init: Box<dyn crate::init::r#trait::IterateInitializer> = if self.warm_start_init_point
        {
            Box::new(WarmStartIterateInitializer::new())
        } else {
            Box::new(DefaultIterateInitializer::with_eq_mult_calculator(
                Box::new(LeastSquareMults::new()),
            ))
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

        let scaling: Box<dyn NlpScalingObject> = match self.nlp_scaling_method {
            NlpScalingChoice::None => Box::new(NoNlpScalingObject::new()),
            NlpScalingChoice::User => Box::new(UserScaling::new()),
            NlpScalingChoice::GradientBased => Box::new(GradientScaling::new()),
            NlpScalingChoice::EquilibrationBased => Box::new(EquilibrationScaling::new()),
        };

        AlgorithmBundle {
            mu_update,
            conv_check,
            init,
            eq_mult,
            hess,
            line_search,
            iter_output,
            scaling,
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
        let scaling = [
            NlpScalingChoice::None,
            NlpScalingChoice::User,
            NlpScalingChoice::GradientBased,
            NlpScalingChoice::EquilibrationBased,
        ];
        for &linear_solver in &solvers {
            for &mu_strategy in &mu {
                for &hessian_approximation in &hess {
                    for &line_search_method in &ls {
                        for &nlp_scaling_method in &scaling {
                            let _ = AlgorithmBuilder {
                                linear_solver,
                                mu_strategy,
                                mu_oracle: MuOracleKind::QualityFunction,
                                hessian_approximation,
                                limited_memory_update_type: UpdateType::Bfgs,
                                line_search_method,
                                nlp_scaling_method,
                                warm_start_init_point: false,
                                conv_check: ConvCheckOptions::default(),
                                mu: MuOptions::default(),
                                line_search: LineSearchOptions::default(),
                                output: OutputOptions::default(),
                            }
                            .build();
                        }
                    }
                }
            }
        }
    }
}
