//! Sparse parametric active-set quadratic programming solver for
//! POUNCE.
//!
//! # Algorithm
//!
//! The solver family is **sparse Schur-complement parametric
//! active-set** — the qpOASES lineage extended to sparse Hessian and
//! Jacobian, after Kirches 2011 (*Fast Numerical Methods for
//! Mixed-Integer Nonlinear Model-Predictive Control*) and Janka,
//! Kirches, Sager, Schlöder 2016 (*Math. Prog. Comp.* **8**). It is
//! the only QP family in the literature combining sparse storage,
//! indefinite-Hessian handling, and true parametric warm starting
//! across solves — the trio required for the SQP / MPC / parametric-
//! continuation workloads pounce targets.
//!
//! See [`docs/research/active-set-sqp-warm-start.md`] (§4.2) for the
//! literature pinning and [§5](`docs/research/active-set-sqp-warm-start.md`)
//! for the type-level design this module realizes.
//!
//! # Status
//!
//! Phase 5a scaffold. Types and trait surface are stable; the
//! solver internals (Schur-complement factor maintenance, EXPAND
//! anti-cycling, l1-elastic phase-1, parametric homotopy, inertia
//! control) are stubbed and land in subsequent commits.
//!
//! [`docs/research/active-set-sqp-warm-start.md`]: ../../../../docs/research/active-set-sqp-warm-start.md

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod error;
pub mod options;
pub mod problem;
pub mod solver;
pub mod working_set;

#[cfg(test)]
mod tests;

pub use error::{QpError, QpStatus};
pub use options::{AntiCyclingChoice, QpAlgorithm, QpOptions};
pub use problem::{HessianInertia, QpProblem, QpSolution, QpStats, QpWarmStart};
pub use solver::QpSolver;
pub use working_set::{BoundStatus, ConsStatus, WorkingSet};
