//! POUNCE algorithm-side core.
//!
//! Port of Ipopt's `src/Algorithm/`: the `IteratesVector` data
//! object, the mutable `IpoptData` state, the `IpoptCalculatedQuantities`
//! lazy-cache layer, the KKT subsystem (augmented system, perturbation
//! handler, full-space PD solver, search-direction calculator), the
//! line search (filter + backtracking), barrier-update strategies
//! (monotone now, adaptive in Phase 10), convergence check, iterate
//! initialization, equality-multiplier estimation, Hessian update
//! strategies (exact + L-BFGS/SR1 in Phase 8), iteration output,
//! timing statistics, the algorithm builder, and the main
//! `IpoptAlgorithm::optimize()` loop.
//!
//! NLP scaling (gradient-based objective/constraint scaling) lives
//! NLP-side in [`pounce_nlp::orig_ipopt_nlp`].
//!
//! Strategies are wired together by [`alg_builder::AlgorithmBuilder`]
//! per the dependency order documented in
//! `ref/Ipopt/AGENT_REFERENCE/ARCHITECTURE.md` §"BuildBasicAlgorithm".

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod alg_builder;
pub mod application;
pub mod batch;
pub mod conv_check;
pub mod debug;
pub mod debug_rank;
pub mod eq_mult;
pub mod hess;
pub mod init;
pub mod intermediate;
pub mod ipopt_alg;
pub mod ipopt_cq;
pub mod ipopt_data;
pub mod ipopt_nlp;
pub mod iter_dump;
pub mod iterate_dump;
pub mod iterates_vector;
pub mod kkt;
pub mod line_search;
pub mod mu;
pub mod output;
pub mod restoration;
pub mod sqp;
pub mod strategy;
pub mod timing_stats;
pub mod upstream_options;

pub use application::IpoptApplication;
pub use batch::{
    FeralBackendPool, NlpBatchResult, NlpBatchSolution, NlpWarmStart,
    install_pooled_serial_feral_backend, install_serial_feral_backend, solve_nlp_batch,
    solve_nlp_batch_parallel, solve_nlp_batch_parallel_warm, solve_nlp_batch_warm,
};
pub use ipopt_cq::{IpoptCalculatedQuantities, IpoptCqHandle};
pub use ipopt_data::{IpoptData, IpoptDataHandle, PdPerturbations};
pub use ipopt_nlp::{IpoptNlp, Nlp};
pub use iterates_vector::IteratesVector;
pub use strategy::AlgorithmStrategy;
