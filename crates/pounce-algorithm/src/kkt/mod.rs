//! KKT subsystem — port of `Algorithm/IpAugSystemSolver*`,
//! `IpStdAugSystemSolver*`, `IpPDPerturbationHandler*`,
//! `IpPDSystemSolver*`, `IpPDFullSpaceSolver*`,
//! `IpSearchDirCalculator*`, `IpPDSearchDirCalc*`.
//!
//! Phase 6 traits and skeleton state machines live here; concrete
//! arithmetic is filled in once the linear-solver wrapper
//! (TSymLinearSolver) lands together with the `SymMatrix`/`Vector`
//! plumbing of Phase 5.

pub mod aug_system_solver;
pub mod low_rank_aug_system_solver;
pub mod pd_full_space_solver;
pub mod pd_search_dir_calc;
pub mod pd_system_solver;
pub mod perturbation_handler;
pub mod schur_aug_system_solver;
pub mod search_dir_calc;
pub mod slack_scaling;
pub mod std_aug_system_solver;

pub use aug_system_solver::AugSystemSolver;
pub use low_rank_aug_system_solver::LowRankAugSystemSolver;
pub use schur_aug_system_solver::SchurAugSystemSolver;
pub use slack_scaling::SlackBasedTSymScalingMethod;
pub use std_aug_system_solver::StdAugSystemSolver;
