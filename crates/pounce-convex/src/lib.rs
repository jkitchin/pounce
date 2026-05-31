//! `pounce-convex` — interior-point solvers for POUNCE's convex problem
//! classes.
//!
//! Phase 2 of the LP/QP routing plan (see `dev-notes/lp-qp-routing.md`):
//! a bare primal-dual interior-point solver for convex QP (and LP, which
//! is the `P = 0` case), built over a [`cones::Cone`] abstraction with
//! only the nonnegative orthant implemented so that later phases
//! (Mehrotra + HSDE, SOCP, exponential/power cones, SDP) extend rather
//! than rewrite the driver.
//!
//! The augmented-system factorization is shared with the NLP path via
//! [`pounce_linsol::Factorization`]; this crate adds no new linear-solver
//! dependency.
//!
//! Entry points:
//! - [`solve_qp_ipm`] — solve a [`qp::QpProblem`] (covers LP via an empty
//!   `P`).

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod batch;
pub mod cones;
pub mod ipm;
pub mod presolve;
pub mod qp;

pub use batch::{
    solve_qp_batch, solve_qp_batch_parallel, solve_qp_multi_rhs, solve_qp_multi_rhs_parallel,
};
pub use ipm::{solve_qp_ipm, QpOptions};
pub use qp::{QpProblem, QpSolution, QpStatus, Triplet, NEG_INF, POS_INF};
