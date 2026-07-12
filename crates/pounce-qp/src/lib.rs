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
//! Implemented and tested. The solver internals — Schur-complement
//! factor maintenance ([`schur`]), GMSW EXPAND anti-cycling
//! ([`working_set`]), l1-elastic phase-1 ([`elastic`]), parametric
//! homotopy ([`ParametricActiveSetSolver::solve_parametric`]), and
//! inertia control ([`HessianInertia`]) — are live, exercised by the
//! crate's unit tests and the published-optimum fixtures under
//! `tests/` (Maros-Mészáros-style closed-form KKT optima). The active
//! set engine is reached from Python through the SQP path
//! (`Problem(algorithm = "active-set-sqp")`, with `working_set`
//! warm-starting) and through `QpSensitivity` (parametric `dx/dp`), and
//! from the CLI via `solver_selection = "qp-active-set"` on an LP /
//! convex-QP `.nl` (routed through the SQP driver, which solves its step
//! QPs here). Still outstanding: cross-checking the full 138-problem
//! Maros-Mészáros `.qps` set against external oracles (qpOASES / OSQP),
//! which is gated on the `.qps` distribution and FFI.
//!
//! [`docs/research/active-set-sqp-warm-start.md`]: ../../../../docs/research/active-set-sqp-warm-start.md

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod elastic;
pub mod error;
pub mod factor;
pub mod kkt;
pub mod options;
pub mod problem;
pub mod qps;
pub mod schur;
pub mod solver;
pub mod working_set;

#[cfg(test)]
mod tests;

pub use elastic::ElasticReformulation;
pub use error::{QpError, QpStatus};
pub use factor::LinearSolver;
pub use kkt::{
    KktTriplet, a_times_x, assemble_active_set_kkt, assemble_box_with_active,
    assemble_equality_plus_bounds, h_times_x, is_all_equality_constraints, is_pure_box,
    is_pure_equality_no_bounds, rhs_equality_only,
};
pub use options::{AntiCyclingChoice, QpAlgorithm, QpOptions};
pub use problem::{HessianInertia, QpProblem, QpSolution, QpStats, QpWarmStart};
pub use qps::{QpsModel, parse_qps};
pub use solver::{ParametricActiveSetSolver, QpSolver};
pub use working_set::{BoundStatus, ConsStatus, WorkingSet};
