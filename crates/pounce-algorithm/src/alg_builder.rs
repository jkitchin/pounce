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
use crate::kkt::aug_system_solver::AugSystemSolver;
use crate::kkt::low_rank_aug_system_solver::LowRankAugSystemSolver;
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

/// Symmetric scaling method applied to the augmented KKT system by
/// [`TSymLinearSolver`]. Mirrors the `linear_system_scaling` option
/// in `IpAlgBuilder.cpp:302-318` and the `RuizTSymScalingMethod` /
/// `Mc19TSymScalingMethod` strategies in upstream Ipopt.
///
/// * `None` (default) — no scaling; `TSymLinearSolver` runs with a
///   null scaling method. Matches upstream's default.
/// * `Ruiz` — iterative symmetric ∞-norm equilibration (Ruiz, 2001).
///   Implemented in `pounce_linsol::RuizTSymScalingMethod`.
/// * `Mc19` — Curtis-Reid (HSL MC19) scaling. Not yet implemented;
///   falls back to `None` with a warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinearSystemScalingChoice {
    #[default]
    None,
    Ruiz,
    Mc19,
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
    /// Objective-scale floor below which a strict termination certificate is
    /// refused while the unscaled KKT error is still above `acceptable_tol`
    /// (gh #200). `0` disables the mechanism.
    pub obj_scale_certificate_threshold: Number,
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
            obj_scale_certificate_threshold: 1e-4,
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
    /// Symmetric scaling method for the augmented KKT system. Wired
    /// into [`TSymLinearSolver`] by [`Self::build_with_backend`].
    /// Mirrors upstream `linear_system_scaling` (`IpAlgBuilder.cpp:538-560`).
    pub linear_system_scaling: LinearSystemScalingChoice,
    /// Lazy-vs-eager scaling toggle (`linear_scaling_on_demand`,
    /// `IpTSymLinearSolver.cpp:50-58`). Only consulted when
    /// `linear_system_scaling != None`. Upstream default is `true`
    /// (compute scaling only on the first solve that fails / shows
    /// poor conditioning); pounce mirrors that. Set to `false` to
    /// scale every factorization.
    pub linear_scaling_on_demand: bool,
    pub mu_strategy: MuStrategyChoice,
    /// Selector forwarded to [`AdaptiveMuUpdate`] when
    /// `mu_strategy = Adaptive`. Ignored for `Monotone`. Defaults to
    /// `QualityFunction` per upstream's `RegisterOptions` default.
    pub mu_oracle: MuOracleKind,
    pub hessian_approximation: HessianApproxChoice,
    pub limited_memory_update_type: UpdateType,
    /// History length for the limited-memory quasi-Newton approximation
    /// (`limited_memory_max_history`). Defaults to upstream's 6.
    pub limited_memory_max_history: i32,
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
    /// `kappa_sigma` — factor bounding how far the bound multipliers may
    /// deviate from their primal estimates. The clamp
    /// (`kappa_sigma_clamp`) runs after every accepted step; `< 1`
    /// disables the correction. Mirrors `IpIpoptAlg.cpp` (Eqn. (16)),
    /// default `1e10`. Baked onto [`crate::ipopt_alg::IpoptAlgorithm`] by
    /// the solve path.
    pub kappa_sigma: Number,
    /// `kappa_d` — weight of the linear damping term added to the barrier
    /// objective/gradient (and dual-infeasibility) to handle one-sided
    /// bounds. Mirrors `IpIpoptCalculatedQuantities.cpp`, default `1e-5`.
    /// Baked onto [`crate::ipopt_cq::IpoptCalculatedQuantities`] by the
    /// solve path.
    pub kappa_d: Number,
    /// `tiny_step_tol` — relative primal step size below which the full
    /// step is accepted without line search; repeated tiny steps
    /// terminate the solve. Mirrors `IpBacktrackingLineSearch.cpp`,
    /// default `10·EPSILON`. Baked onto
    /// [`crate::ipopt_alg::IpoptAlgorithm`] by the solve path.
    pub tiny_step_tol: Number,
    /// `tiny_step_y_tol` — dual-step threshold; when both primal and dual
    /// steps are tiny in consecutive iterations the algorithm stops at the
    /// best attainable accuracy. Default `1e-2`.
    pub tiny_step_y_tol: Number,
    /// `diverging_iterates_tol` — if `max_i |x_i|` exceeds this the solve
    /// aborts as diverging. Default `1e20`.
    pub diverging_iterates_tol: Number,
    /// `dual_diverging_streak` (pounce#246) — consecutive growing-dual-
    /// infeasibility iterations before the dual-divergence guard routes to
    /// restoration. **Default `0` (off).**
    ///
    /// It defaulted to `15` when introduced, on the strength of a reported
    /// emfl050 bad-warm-start grind. That justification did not survive being
    /// reproduced: the measurement was caller-side JAX compilation, and the
    /// build predating the guard solves both emfl050 instances to the same
    /// optimum in the same time (pounce#246 / pounce#250). What remained was a
    /// knife-edge, non-monotone effect on four of 1284 MINLPLib models — so it
    /// is opt-in rather than imposed. See `upstream_options.rs` for the full
    /// account.
    pub dual_diverging_streak: Index,
    /// `kkt_fidelity_tol` (pounce#173). Read by the algorithm as well as by the
    /// post-solve gate, because the #200 fallback's tiebreak has to rank the two
    /// candidate points by the status each will be *reported* under. Default
    /// `0.0` (gate disabled).
    pub kkt_fidelity_tol: Number,
    pub conv_check: ConvCheckOptions,
    pub mu: MuOptions,
    pub line_search: LineSearchOptions,
    pub refinement: RefinementOptions,
    pub perturbation: PerturbationOptions,
    pub resto: RestoOptions,
    pub output: OutputOptions,
    pub warm: WarmStartOptions,
    /// SQP-specific options (consulted only when
    /// `algorithm = ActiveSetSqp`).
    pub sqp: crate::sqp::SqpOptions,
    /// QP-subproblem-solver options for the active-set SQP path
    /// (`pounce_qp::QpOptions`), threaded into the `SqpAlgorithm` via
    /// `with_qp_options`. Consulted only when `algorithm = ActiveSetSqp`.
    /// Populated from the `sqp_qp_*` CLI options by
    /// `application::apply_qp_subproblem_options`.
    pub sqp_qp: pounce_qp::QpOptions,
    pub init: InitOptions,
    /// Optional block-triangular / Schur KKT partition (pounce#180 item 2):
    /// `(schur_indices, feral_cfg)`. When `Some` and the IPM path is selected
    /// with the feral linear solver and an exact Hessian, `build_with_backend`
    /// wraps the standard aug-system solver in a
    /// [`crate::kkt::SchurAugSystemSolver`] over the given KKT-space indices.
    /// The Schur solver falls back to the standard solver transparently when
    /// the partition is unsuitable. Set via [`Self::set_kkt_schur`].
    pub kkt_schur: Option<(Vec<usize>, pounce_feral::FeralConfig)>,
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
    /// `least_square_init_primal` — replace the user's starting `x`
    /// with the min-norm primal that satisfies the linearized
    /// constraints. Used by the Mehrotra cascade in `application.rs`
    /// to drop iter-0 primal infeasibility on LP-shaped problems.
    /// Mirrors upstream `IpDefaultIterateInitializer.cpp:200-222`.
    pub least_square_init_primal: bool,
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
            least_square_init_primal: false,
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
    /// `adaptive_mu_globalization` — globalization strategy for the
    /// adaptive μ-selection mode. Mirrors
    /// `IpAdaptiveMuUpdate.cpp:RegisterOptions`. Default is
    /// `ObjConstrFilter`; the Mehrotra cascade switches to
    /// `NeverMonotoneMode` to disable globalization entirely.
    pub adaptive_mu_globalization: crate::mu::adaptive::AdaptiveMuGlobalization,
    /// `quality_function_norm_type` — norm used inside the quality
    /// function to aggregate the three KKT components. Forwarded to
    /// `QualityFunctionMuOracle` when `mu_oracle=quality-function`.
    pub quality_function_norm_type: crate::mu::oracle::quality_function::NormType,
    /// `quality_function_centrality` — centrality penalty term added
    /// to the quality function.
    pub quality_function_centrality: crate::mu::oracle::quality_function::CentralityType,
    /// `quality_function_balancing_term` — balancing penalty term in
    /// the quality function (kicks in when complementarity is far
    /// below infeasibilities).
    pub quality_function_balancing_term: crate::mu::oracle::quality_function::BalancingTermType,
    /// `quality_function_max_section_steps` — cap on golden-section
    /// iterations when picking σ. Default 8.
    pub quality_function_max_section_steps: i32,
    /// `quality_function_section_sigma_tol` — width tolerance in
    /// σ-space for golden section. Default 1e-2.
    pub quality_function_section_sigma_tol: Number,
    /// `quality_function_section_qf_tol` — relative flatness
    /// tolerance for golden section. Default 0.0.
    pub quality_function_section_qf_tol: Number,
    /// `adaptive_mu_safeguard_factor` — guard for the LOQO fallback
    /// in adaptive mode. Default 0.0.
    pub adaptive_mu_safeguard_factor: Number,
    /// `adaptive_mu_monotone_init_factor` — multiplier on the
    /// average complementarity when seeding monotone mode after a
    /// free-mode bailout. Default 0.8.
    pub adaptive_mu_monotone_init_factor: Number,
    /// `adaptive_mu_restore_previous_iterate` — restore the most
    /// recent free-mode iterate when switching to fixed mode.
    /// Default `false`.
    pub adaptive_mu_restore_previous_iterate: bool,
    /// `adaptive_mu_kkterror_red_iters` — window length for the
    /// `KKT_ERROR` globalization history. Default 4.
    pub adaptive_mu_kkterror_red_iters: usize,
    /// `adaptive_mu_kkterror_red_fact` — required relative reduction
    /// of the KKT error over the window. Default 0.9999.
    pub adaptive_mu_kkterror_red_fact: Number,
    /// `adaptive_mu_kkt_norm_type` — norm used to score the iterate
    /// in adaptive globalization decisions.
    pub adaptive_mu_kkt_norm_type: crate::mu::adaptive::AdaptiveMuKktNorm,
    /// `probing_iterate_quality_factor` (default 1e4, pounce-specific
    /// — see pounce#58). When the probing (Mehrotra) μ-oracle is
    /// about to read `curr_avrg_compl()` for its `mu_curr` input, a
    /// single imbalanced `(s_i, z_i)` pair can inflate the average
    /// 5+ orders above the stored `data.curr_mu`. The oracle then
    /// returns `σ · mu_curr` ≫ previous μ, throwing the iterate out
    /// of the convergence neighborhood. This guard short-circuits
    /// that case by signalling restoration when the ratio
    /// `curr_avrg_compl / curr_mu` exceeds the factor. Set to 0 or
    /// any non-positive value to disable.
    pub probing_iterate_quality_factor: Number,
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
            adaptive_mu_globalization:
                crate::mu::adaptive::AdaptiveMuGlobalization::ObjConstrFilter,
            quality_function_norm_type:
                crate::mu::oracle::quality_function::NormType::TwoNormSquared,
            quality_function_centrality: crate::mu::oracle::quality_function::CentralityType::None,
            quality_function_balancing_term:
                crate::mu::oracle::quality_function::BalancingTermType::None,
            quality_function_max_section_steps: 8,
            quality_function_section_sigma_tol: 1e-2,
            quality_function_section_qf_tol: 0.0,
            adaptive_mu_safeguard_factor: 0.0,
            adaptive_mu_monotone_init_factor: 0.8,
            adaptive_mu_restore_previous_iterate: false,
            adaptive_mu_kkterror_red_iters: 4,
            adaptive_mu_kkterror_red_fact: 0.9999,
            adaptive_mu_kkt_norm_type: crate::mu::adaptive::AdaptiveMuKktNorm::TwoNormSquared,
            probing_iterate_quality_factor: 1e4,
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

    // Filter switching / Armijo / margin constants baked onto the
    // assembled [`crate::line_search::filter_acceptor::FilterLsAcceptor`]
    // (only when `line_search_method = Filter`). All were registered but
    // never read (#191); defaults mirror `IpFilterLSAcceptor.cpp`.
    /// `eta_phi` — relaxation factor in the Armijo condition (Eqn. (20)).
    pub eta_phi: Number,
    /// `theta_min_fact` — constraint-violation threshold factor in the
    /// switching rule.
    pub theta_min_fact: Number,
    /// `theta_max_fact` — upper-bound factor for constraint violation in
    /// the filter (Eqn. (21)).
    pub theta_max_fact: Number,
    /// `gamma_phi` — filter margin factor for the barrier function
    /// (Eqn. (18a)).
    pub gamma_phi: Number,
    /// `gamma_theta` — filter margin factor for the constraint violation
    /// (Eqn. (18b)).
    pub gamma_theta: Number,
    /// `s_phi` — exponent for the linear barrier model in the switching
    /// rule (Eqn. (19)).
    pub s_phi: Number,
    /// `s_theta` — exponent for the current constraint violation in the
    /// switching rule (Eqn. (19)).
    pub s_theta: Number,
    /// `alpha_min_frac` — safety factor for the minimal step size before
    /// switching to restoration (gamma_alpha, Eqn. (23)).
    pub alpha_min_frac: Number,
    /// `obj_max_inc` — max acceptable increase (orders of magnitude) of
    /// the barrier objective for a trial point.
    pub obj_max_inc: Number,
    /// `max_filter_resets` — maximum number of filter resets allowed
    /// (`0` disables the reset heuristic).
    pub max_filter_resets: Index,
    /// `filter_reset_trigger` — successive filter-rejected iterations that
    /// trigger a filter reset.
    pub filter_reset_trigger: Index,

    // Second-order-correction constants baked onto the assembled
    // [`BacktrackingLineSearch`]. Registered but never read (#191);
    // defaults mirror `IpBacktrackingLineSearch.cpp`.
    /// `max_soc` — max second-order-correction trial steps per iteration;
    /// `0` disables SOC.
    pub max_soc: Index,
    /// `kappa_soc` — sufficient-reduction factor for a SOC step to be
    /// continued.
    pub kappa_soc: Number,
    /// `soc_method` — `0` (paper method) or `1` (alpha-on-rhs variant).
    pub soc_method: Index,
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
            eta_phi: 1e-8,
            theta_min_fact: 1e-4,
            theta_max_fact: 1e4,
            gamma_phi: 1e-8,
            gamma_theta: 1e-5,
            s_phi: 2.3,
            s_theta: 1.1,
            alpha_min_frac: 0.05,
            obj_max_inc: 5.0,
            max_filter_resets: 5,
            filter_reset_trigger: 5,
            max_soc: 4,
            kappa_soc: 0.99,
            soc_method: 0,
        }
    }
}

/// Inertia-correction / regularization knobs baked onto the assembled
/// [`crate::kkt::perturbation_handler::PdPerturbationHandler`]. Field
/// names use the option names; they map to the handler's `delta_xs_*` /
/// `delta_cd_*` fields. Defaults mirror
/// `IpPDPerturbationHandler.cpp:RegisterOptions`. All were registered but
/// never read (#191).
#[derive(Debug, Clone)]
pub struct PerturbationOptions {
    /// `max_hessian_perturbation` → `delta_xs_max`.
    pub max_hessian_perturbation: Number,
    /// `min_hessian_perturbation` → `delta_xs_min`.
    pub min_hessian_perturbation: Number,
    /// `perturb_inc_fact_first` → `delta_xs_first_inc_fact`.
    pub perturb_inc_fact_first: Number,
    /// `perturb_inc_fact` → `delta_xs_inc_fact`.
    pub perturb_inc_fact: Number,
    /// `perturb_dec_fact` → `delta_xs_dec_fact`.
    pub perturb_dec_fact: Number,
    /// `first_hessian_perturbation` → `delta_xs_init`.
    pub first_hessian_perturbation: Number,
    /// `jacobian_regularization_value` → `delta_cd_val`.
    pub jacobian_regularization_value: Number,
    /// `jacobian_regularization_exponent` → `delta_cd_exp`.
    pub jacobian_regularization_exponent: Number,
    /// `perturb_always_cd` — always regularize the c/d (Jacobian) block.
    pub perturb_always_cd: bool,
}

impl Default for PerturbationOptions {
    fn default() -> Self {
        Self {
            max_hessian_perturbation: 1e20,
            min_hessian_perturbation: 1e-20,
            perturb_inc_fact_first: 100.0,
            perturb_inc_fact: 8.0,
            perturb_dec_fact: 1.0 / 3.0,
            first_hessian_perturbation: 1e-4,
            jacobian_regularization_value: 1e-8,
            jacobian_regularization_exponent: 0.25,
            perturb_always_cd: false,
        }
    }
}

/// Restoration-phase knobs carried on the outer builder and copied into
/// the `RestoAlgorithmBuilder` when the restoration factory is minted
/// (`pounce-restoration`). The restoration builder is constructed with
/// defaults by each frontend and never options-configured, so these were
/// registered but never read (#191). Defaults mirror upstream's
/// restoration `RegisterOptions`.
#[derive(Debug, Clone)]
pub struct RestoOptions {
    /// `bound_mult_reset_threshold` — reset bound multipliers to 1 after
    /// restoration if the largest exceeds this.
    pub bound_mult_reset_threshold: Number,
    /// `constr_mult_reset_threshold` — ignore the least-square constraint
    /// multiplier estimate after restoration if its norm exceeds this
    /// (`0` keeps the estimate).
    pub constr_mult_reset_threshold: Number,
    /// `resto_penalty_parameter` — penalty on the slack 1-norm in the
    /// restoration objective (`rho`).
    pub resto_penalty_parameter: Number,
    /// `resto_proximity_weight` — proximity-term weight (`eta_factor`;
    /// `η = eta_factor · sqrt(μ)`).
    pub resto_proximity_weight: Number,
}

impl Default for RestoOptions {
    fn default() -> Self {
        Self {
            bound_mult_reset_threshold: 1e3,
            constr_mult_reset_threshold: 0.0,
            resto_penalty_parameter: 1e3,
            resto_proximity_weight: 1.0,
        }
    }
}

/// Iterative-refinement knobs baked onto the assembled
/// [`crate::kkt::pd_full_space_solver::PdFullSpaceSolver`]. Defaults
/// mirror `IpPDFullSpaceSolver.cpp:RegisterOptions`. All were registered
/// but never read (#191).
#[derive(Debug, Clone)]
pub struct RefinementOptions {
    /// `min_refinement_steps` — minimum iterative-refinement steps per
    /// linear solve.
    pub min_refinement_steps: Index,
    /// `max_refinement_steps` — maximum iterative-refinement steps.
    pub max_refinement_steps: Index,
    /// `residual_ratio_max` — refine until the residual test ratio drops
    /// below this (or `max_refinement_steps` is reached).
    pub residual_ratio_max: Number,
    /// `residual_ratio_singular` — above this ratio after failed
    /// refinement, the system is declared singular.
    pub residual_ratio_singular: Number,
    /// `residual_improvement_factor` — minimum per-step reduction of the
    /// residual test ratio before refinement is aborted.
    pub residual_improvement_factor: Number,
}

impl Default for RefinementOptions {
    fn default() -> Self {
        Self {
            min_refinement_steps: 1,
            max_refinement_steps: 10,
            residual_ratio_max: 1e-10,
            residual_ratio_singular: 1e-5,
            residual_improvement_factor: 0.999_999_999,
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
            linear_system_scaling: LinearSystemScalingChoice::None,
            linear_scaling_on_demand: true,
            mu_strategy: MuStrategyChoice::Monotone,
            mu_oracle: MuOracleKind::QualityFunction,
            hessian_approximation: HessianApproxChoice::Exact,
            limited_memory_update_type: UpdateType::Bfgs,
            limited_memory_max_history: 6,
            line_search_method: LineSearchChoice::Filter,
            warm_start_init_point: false,
            mehrotra_algorithm: false,
            kappa_sigma: 1e10,
            kappa_d: 1e-5,
            tiny_step_tol: 10.0 * Number::EPSILON,
            tiny_step_y_tol: 1e-2,
            diverging_iterates_tol: 1e20,
            dual_diverging_streak: 0,
            kkt_fidelity_tol: 0.0,
            conv_check: ConvCheckOptions::default(),
            mu: MuOptions::default(),
            line_search: LineSearchOptions::default(),
            refinement: RefinementOptions::default(),
            perturbation: PerturbationOptions::default(),
            resto: RestoOptions::default(),
            output: OutputOptions::default(),
            warm: WarmStartOptions::default(),
            sqp: crate::sqp::SqpOptions::default(),
            sqp_qp: pounce_qp::QpOptions::default(),
            init: InitOptions::default(),
            kkt_schur: None,
        }
    }
}

impl AlgorithmBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Install a Schur KKT partition (pounce#180 item 2). `schur_indices` are
    /// KKT-space indices (`0..dim`, the `x,s,c,d` block order the aug-system
    /// solver assembles); `cfg` configures the per-block feral solvers. Only
    /// honored on the IPM + feral + exact-Hessian path by
    /// [`Self::build_with_backend`]; ignored otherwise.
    pub fn set_kkt_schur(&mut self, schur_indices: Vec<usize>, cfg: pounce_feral::FeralConfig) {
        self.kkt_schur = Some((schur_indices, cfg));
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
        let scaling: Option<Box<dyn pounce_linsol::TSymScalingMethod>> =
            match self.linear_system_scaling {
                LinearSystemScalingChoice::None => None,
                LinearSystemScalingChoice::Ruiz => {
                    Some(Box::new(pounce_linsol::RuizTSymScalingMethod::new()))
                }
                LinearSystemScalingChoice::Mc19 => {
                    tracing::warn!(target: "pounce::algorithm",
                        "pounce: linear_system_scaling=mc19 not yet implemented; using no scaling"
                    );
                    None
                }
            };
        let linsol = TSymLinearSolver::new(backend, scaling, self.linear_scaling_on_demand);
        let inner_aug = StdAugSystemSolver::new(linsol);
        // Limited-memory mode publishes the Hessian as a
        // `LowRankUpdateSymMatrix`; wrap the standard solver in the
        // Sherman-Morrison-Woodbury low-rank solver so the augmented
        // system factorizes only the diagonal `B0` and the quasi-Newton
        // update is applied as a rank-`m` correction (`O(n·m)` memory).
        let is_lbfgs = matches!(
            self.hessian_approximation,
            HessianApproxChoice::LimitedMemory
        );
        let aug_solver: Box<dyn AugSystemSolver> = if is_lbfgs {
            Box::new(LowRankAugSystemSolver::new(Box::new(inner_aug)))
        } else if let Some((indices, cfg)) = self.kkt_schur.clone() {
            // Block-triangular / Schur KKT path (pounce#180 item 2). Only on the
            // exact-Hessian feral path — the Schur backend is feral-specific,
            // and the L-BFGS low-rank Woodbury wrapper owns the (2,2) block.
            // The Schur solver falls back to `StdAugSystemSolver` transparently
            // when the partition is unsuitable, so a stray hook never breaks a
            // solve; we gate on `linear_solver == Feral` here to avoid silently
            // ignoring a user's explicit MA57 selection.
            if matches!(self.linear_solver, LinearSolverChoice::Feral) {
                Box::new(crate::kkt::SchurAugSystemSolver::new(
                    inner_aug, indices, cfg,
                ))
            } else {
                Box::new(inner_aug)
            }
        } else {
            Box::new(inner_aug)
        };
        // Inertia-correction / Jacobian-regularization constants (#191):
        // registered but previously never read. Defaults equal the
        // registered defaults. `perturb_always_cd` goes through the setter
        // because it also rebuilds the initial jac-degeneracy state.
        let mut ph = PdPerturbationHandler::new();
        ph.delta_xs_max = self.perturbation.max_hessian_perturbation;
        ph.delta_xs_min = self.perturbation.min_hessian_perturbation;
        ph.delta_xs_first_inc_fact = self.perturbation.perturb_inc_fact_first;
        ph.delta_xs_inc_fact = self.perturbation.perturb_inc_fact;
        ph.delta_xs_dec_fact = self.perturbation.perturb_dec_fact;
        ph.delta_xs_init = self.perturbation.first_hessian_perturbation;
        ph.delta_cd_val = self.perturbation.jacobian_regularization_value;
        ph.delta_cd_exp = self.perturbation.jacobian_regularization_exponent;
        ph.set_perturb_always_cd(self.perturbation.perturb_always_cd);
        let perturb = Rc::new(RefCell::new(ph));
        let mut pd_solver = PdFullSpaceSolver::new(aug_solver, perturb);
        // Iterative-refinement constants (#191): registered but previously
        // never read, so overrides were silently dropped. Defaults equal
        // the registered defaults.
        pd_solver.min_refinement_steps = self.refinement.min_refinement_steps;
        pd_solver.max_refinement_steps = self.refinement.max_refinement_steps;
        pd_solver.residual_ratio_max = self.refinement.residual_ratio_max;
        pd_solver.residual_ratio_singular = self.refinement.residual_ratio_singular;
        pd_solver.residual_improvement_factor = self.refinement.residual_improvement_factor;
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
        Some(
            crate::sqp::SqpAlgorithm::new(qp_solver, self.sqp.clone())
                .with_qp_options(self.sqp_qp.clone()),
        )
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
                adaptive.adaptive_mu_globalization = self.mu.adaptive_mu_globalization;
                adaptive.qf_norm_type = self.mu.quality_function_norm_type;
                adaptive.qf_centrality_type = self.mu.quality_function_centrality;
                adaptive.qf_balancing_term = self.mu.quality_function_balancing_term;
                adaptive.qf_max_section_steps = self.mu.quality_function_max_section_steps;
                adaptive.qf_section_sigma_tol = self.mu.quality_function_section_sigma_tol;
                adaptive.qf_section_qf_tol = self.mu.quality_function_section_qf_tol;
                adaptive.probing_iterate_quality_factor = self.mu.probing_iterate_quality_factor;
                adaptive.adaptive_mu_safeguard_factor = self.mu.adaptive_mu_safeguard_factor;
                adaptive.adaptive_mu_monotone_init_factor =
                    self.mu.adaptive_mu_monotone_init_factor;
                adaptive.restore_accepted_iterate = self.mu.adaptive_mu_restore_previous_iterate;
                adaptive.adaptive_mu_kkterror_red_iters = self.mu.adaptive_mu_kkterror_red_iters;
                adaptive.adaptive_mu_kkterror_red_fact = self.mu.adaptive_mu_kkterror_red_fact;
                adaptive.adaptive_mu_kkt_norm = self.mu.adaptive_mu_kkt_norm_type;
                Box::new(adaptive)
            }
        };

        let acceptor: Box<dyn BacktrackingLsAcceptor> = match self.line_search_method {
            LineSearchChoice::Filter => {
                // Filter switching / Armijo / margin constants (#191):
                // registered but previously never read. Set them on the
                // concrete acceptor before boxing; defaults equal the
                // registered defaults, so a run that doesn't set them is
                // unchanged.
                let mut f = FilterLsAcceptor::default();
                f.eta_phi = self.line_search.eta_phi;
                f.theta_min_fact = self.line_search.theta_min_fact;
                f.theta_max_fact = self.line_search.theta_max_fact;
                f.gamma_phi = self.line_search.gamma_phi;
                f.gamma_theta = self.line_search.gamma_theta;
                f.s_phi = self.line_search.s_phi;
                f.s_theta = self.line_search.s_theta;
                f.alpha_min_frac = self.line_search.alpha_min_frac;
                f.obj_max_inc = self.line_search.obj_max_inc;
                f.max_filter_resets = self.line_search.max_filter_resets;
                f.filter_reset_trigger = self.line_search.filter_reset_trigger;
                Box::new(f)
            }
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
        // Second-order-correction constants (#191): registered but
        // previously never read. Same direct-field pattern as the
        // watchdog knobs above.
        line_search.max_soc = self.line_search.max_soc;
        line_search.kappa_soc = self.line_search.kappa_soc;
        line_search.soc_method = self.line_search.soc_method;

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
                obj_scale_certificate_threshold: self.conv_check.obj_scale_certificate_threshold,
                veto_fired: false,
                acceptable_veto_fired: false,
                veto_extra_iters: 0,
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
            d.least_square_init_primal = self.init.least_square_init_primal;
            Box::new(d)
        };

        let eq_mult: Box<dyn crate::eq_mult::r#trait::EqMultCalculator> =
            Box::new(LeastSquareMults::new());

        let hess: Box<dyn crate::hess::r#trait::HessianUpdater> = match self.hessian_approximation {
            HessianApproxChoice::Exact => Box::new(ExactHessianUpdater::new()),
            HessianApproxChoice::LimitedMemory => Box::new(LimMemQuasiNewtonUpdater {
                update_type: self.limited_memory_update_type,
                max_history: self.limited_memory_max_history,
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
                            linear_system_scaling: LinearSystemScalingChoice::None,
                            linear_scaling_on_demand: true,
                            mu_strategy,
                            mu_oracle: MuOracleKind::QualityFunction,
                            hessian_approximation,
                            limited_memory_update_type: UpdateType::Bfgs,
                            limited_memory_max_history: 6,
                            line_search_method,
                            warm_start_init_point: false,
                            mehrotra_algorithm: false,
                            kappa_sigma: 1e10,
                            kappa_d: 1e-5,
                            tiny_step_tol: 10.0 * Number::EPSILON,
                            tiny_step_y_tol: 1e-2,
                            diverging_iterates_tol: 1e20,
                            dual_diverging_streak: 0,
                            kkt_fidelity_tol: 0.0,
                            conv_check: ConvCheckOptions::default(),
                            mu: MuOptions::default(),
                            line_search: LineSearchOptions::default(),
                            refinement: RefinementOptions::default(),
                            perturbation: PerturbationOptions::default(),
                            resto: RestoOptions::default(),
                            output: OutputOptions::default(),
                            warm: WarmStartOptions::default(),
                            sqp: crate::sqp::SqpOptions::default(),
                            sqp_qp: pounce_qp::QpOptions::default(),
                            init: InitOptions::default(),
                            kkt_schur: None,
                        }
                        .build();
                    }
                }
            }
        }
    }
}
