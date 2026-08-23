//! `pounce-convex` — interior-point solvers for POUNCE's convex problem
//! classes.
//!
//! Originally Phase 2 of the LP/QP routing plan (see
//! `dev-notes/lp-qp-routing.md`): a primal-dual interior-point solver for
//! convex QP (and LP, the `P = 0` case), built over a [`cones::Cone`]
//! abstraction so that later cone families extend rather than rewrite the
//! driver. Those phases have since landed: beyond the nonnegative orthant
//! (`cones::nonneg`), the crate implements and production-wires the
//! **second-order (SOC/SOCP)** cone (`cones::soc`), **exponential** and
//! **power** cones (`cones::exp`, `cones::power`), and **PSD** blocks
//! (`cones::psd`), together with **Mehrotra** predictor–corrector, the
//! **homogeneous self-dual embedding** (`hsde`, `hsde_nonsym`), and **Ruiz
//! equilibration** / presolve (`equilibrate`, `presolve`).
//!
//! The augmented-system factorization is shared with the NLP path via
//! [`pounce_linsol::Factorization`]; this crate adds no new linear-solver
//! dependency.
//!
//! Entry points:
//! - [`solve_qp_ipm`] — solve a [`qp::QpProblem`] (covers LP via an empty
//!   `P`).
//! - [`certify_psd_lower_triangle`] — conservatively certify a sparse
//!   symmetric matrix from a caller-supplied inertia-reporting backend;
//!   malformed input is returned as [`PsdCertificateError`].

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod active_set;
pub(crate) mod aggregate;
pub mod batch;
pub mod cones;
pub(crate) mod correctors;
pub mod crossover;
mod deadline;
pub(crate) mod debug;
pub(crate) mod equilibrate;
pub mod hsde;
pub mod hsde_nonsym;
pub mod ipm;
mod options;
pub mod presolve;
mod psd_certificate;
pub mod qp;
pub mod sensitivity;
pub(crate) mod simplex;
pub mod sos;

pub use active_set::{ActiveSetOverrides, solve_qp_active_set};
pub use batch::{
    solve_qp_batch, solve_qp_batch_parallel, solve_qp_batch_parallel_warm, solve_qp_multi_rhs,
    solve_qp_multi_rhs_parallel,
};
pub use cones::ConeSpec;
pub use ipm::{
    QpFactorization, QpOptions, QpWarmStart, solve_qp_ipm, solve_qp_ipm_debug, solve_qp_ipm_warm,
    solve_socp_ipm, solve_socp_ipm_debug, solve_socp_ipm_warm,
};
pub use options::ConvexPresolveOptions;
pub use psd_certificate::{PsdCertificateError, certify_psd_lower_triangle};
pub use qp::{NEG_INF, POS_INF, QpIterate, QpProblem, QpResiduals, QpSolution, QpStatus, Triplet};
pub use sensitivity::{QpSensitivity, ReducedHessian, SensError};
pub use sos::{
    PolyProblem, Polynomial, SosBound, SosSolution, sos_constrained_lower_bound,
    sos_constrained_lower_bound_opts, sos_lower_bound, sos_lower_bound_opts, sos_minimize,
    sos_minimize_opts, sos_opts,
};
