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
//! - [`solve_qp_active_set`] — the same problem class through the
//!   [`pounce_qp`] parametric active-set engine.
//! - [`ActiveSetSession`] — a *persistent* handle over that driver, for a
//!   family of QPs rather than one: it owns the convex → `pounce-qp`
//!   translation ([`ActiveSetQp`]) and the presolve/postsolve wrapper, and
//!   reuses the previous solve parametrically when that is valid (gh #769).
//! - [`certify_psd_lower_triangle`] — conservatively certify a sparse
//!   symmetric matrix from a caller-supplied inertia-reporting backend;
//!   malformed input is returned as [`PsdCertificateError`].

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod active_set;
pub mod active_set_session;
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

pub use active_set::{
    ActiveSetQp, back_translate, back_translate_verified, engine_options, solve_qp_active_set,
    solve_qp_active_set_inertia, verify_status,
};
pub use active_set_session::{ActiveSetSession, PresolveNote, Reuse, SessionStats};
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
// Defined in `pounce-qp` alongside the `QpOptions` it overlays, and shared
// with the SQP subproblem reader there; re-exported so the public path is
// unchanged for callers who reach it through this crate.
pub use pounce_qp::ActiveSetOverrides;
// The caller's claim about the inertia of `P`, for
// [`solve_qp_active_set_inertia`]. Defined in `pounce-qp` next to the
// `QpProblem` field it fills; re-exported so a caller reaching the active-set
// driver through this crate does not have to depend on `pounce-qp` directly.
pub use pounce_qp::HessianInertia;
// The engine's second-order finding, the third argument [`verify_status`]
// needs and the one thing about the returned point that cannot be re-derived
// from it (gh #848). Defined in `pounce-qp` next to the `QpStats` field it
// fills; re-exported for the same reason `HessianInertia` is.
pub use pounce_qp::SecondOrderVerdict;
pub use psd_certificate::{PsdCertificateError, certify_psd_lower_triangle};
pub use qp::{
    BoxScreen, NEG_INF, POS_INF, QpIterate, QpProblem, QpResiduals, QpSolution, QpStatus, Triplet,
    screen_variable_box,
};
pub use sensitivity::{QpSensitivity, ReducedHessian, SensError};
pub use sos::{
    PolyProblem, Polynomial, SosBound, SosSolution, sos_constrained_lower_bound,
    sos_constrained_lower_bound_opts, sos_lower_bound, sos_lower_bound_opts, sos_minimize,
    sos_minimize_opts, sos_opts,
};
