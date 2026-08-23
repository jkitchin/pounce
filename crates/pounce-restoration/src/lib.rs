//! POUNCE restoration phase.
//!
//! Port of Ipopt's `Algorithm/IpResto*` files. When the line search
//! cannot accept any step, control switches into the restoration
//! phase, which solves a relaxed feasibility problem (minimize the
//! ℓ1 norm of the constraint violation, plus a regularization
//! against `x_R`). On success it returns to the regular IPM with a
//! new iterate. See `ref/Ipopt/AGENT_REFERENCE/RESTORATION.md`.
//!
//! Phase 9 ships the trait surface and the structural skeletons of
//! each `Resto*` strategy; concrete arithmetic ports incrementally
//! once the regular-phase strategies (Phase 7) are exercising the
//! main loop on real problems.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod aug_resto_system_solver;
pub mod conv_check;
pub mod init;
pub mod install;
pub mod min_c_1nrm;
pub mod output;
pub mod resto_alg_builder;
pub mod resto_inner_solver;
pub mod resto_nlp;
pub mod resto_resto;
pub mod r#trait;

pub use r#trait::RestorationPhase;
