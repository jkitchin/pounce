//! Phase 5b — active-set SQP algorithm driver.
//!
//! Sits parallel to [`crate::ipopt_alg::IpoptAlgorithm`]. Consumes
//! the same [`crate::ipopt_nlp::IpoptNlp`] for function /
//! derivative evaluations and reuses [`crate::line_search`] /
//! [`crate::conv_check`] machinery where it makes sense; the QP
//! subproblem solve is delegated to the `pounce-qp` crate
//! (Phases 5a, 5a.1, 5a.2 of the design note).
//!
//! Design reference:
//! [`docs/research/active-set-sqp-warm-start.md`].
//!
//! Selected via [`crate::alg_builder::AlgorithmChoice::ActiveSetSqp`].
//! The `InteriorPoint` choice (default) remains the
//! `IpoptAlgorithm` path with no changes.

pub mod bfgs;
pub mod filter;
pub mod ipopt_adapter;
pub mod iterates;
pub mod lbfgs;
pub mod line_search;
pub mod options;
pub mod problem;
pub mod qp_assembly;
pub mod result;
pub mod sqp_alg;
pub mod warm_start;

#[cfg(test)]
mod tests;

pub use bfgs::DampedBfgs;
pub use filter::{SqpFilter, filter_line_search};
pub use ipopt_adapter::IpoptNlpAdapter;
pub use iterates::SqpIterates;
pub use lbfgs::LBfgs;
pub use options::{SqpGlobalization, SqpHessianSource, SqpOptions};
pub use problem::SqpProblemSpec;
pub use qp_assembly::{SqpQpData, Triplet};
pub use result::{SqpError, SqpResult, SqpStatus};
pub use sqp_alg::SqpAlgorithm;
pub use warm_start::classify_working_set;
