//! POUNCE symmetric linear-solver trait layer.
//!
//! Port of Ipopt's `src/Algorithm/LinearSolvers/`:
//!
//! * [`sym_solver`] — high-level [`SymLinearSolver`] trait
//!   (port of `IpSymLinearSolver.hpp`).
//! * [`sparse_sym_iface`] — low-level [`SparseSymLinearSolverInterface`]
//!   trait that backends like MA57 / MUMPS / FERAL implement
//!   (port of `IpSparseSymLinearSolverInterface.hpp`).
//! * [`status`] — [`ESymSolverStatus`] return enum
//!   (port of the enum in `IpSymLinearSolver.hpp`).
//!
//! Concrete backends live outside this crate: `pounce-hsl` ships the
//! MA57 backend in v1.0; MUMPS and FERAL are slotted behind the same
//! traits in v1.1.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod error;
pub mod factorization;
pub mod scaling;
pub mod sparse_sym_iface;
pub mod status;
pub mod summary;
pub mod sym_solver;
pub mod t_sym_solver;

pub use error::FactorizationError;
pub use factorization::Factorization;
pub use scaling::{IdentityScalingMethod, TSymScalingMethod};
pub use sparse_sym_iface::{EMatrixFormat, SparseSymLinearSolverInterface};
pub use status::ESymSolverStatus;
pub use summary::LinearSolverSummary;
pub use sym_solver::SymLinearSolver;
pub use t_sym_solver::TSymLinearSolver;
