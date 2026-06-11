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
pub(crate) mod debug;
pub(crate) mod equilibrate;
pub mod hsde;
pub mod hsde_nonsym;
pub mod ipm;
pub mod presolve;
pub mod qp;
pub mod sensitivity;
pub mod sos;

pub use batch::{
    solve_qp_batch, solve_qp_batch_parallel, solve_qp_batch_parallel_warm, solve_qp_multi_rhs,
    solve_qp_multi_rhs_parallel,
};
pub use cones::ConeSpec;
pub use ipm::{
    solve_qp_ipm, solve_qp_ipm_debug, solve_qp_ipm_warm, solve_socp_ipm, solve_socp_ipm_debug,
    solve_socp_ipm_warm, QpFactorization, QpOptions, QpWarmStart,
};
pub use qp::{QpIterate, QpProblem, QpResiduals, QpSolution, QpStatus, Triplet, NEG_INF, POS_INF};
pub use sensitivity::{QpSensitivity, ReducedHessian, SensError};
pub use sos::{
    sos_constrained_lower_bound, sos_lower_bound, sos_minimize, PolyProblem, Polynomial, SosBound,
    SosSolution,
};
