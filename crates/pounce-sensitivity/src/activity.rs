//! Post-solve activity classification (the covariance/information
//! roadmap's item 0, gh #362).
//!
//! The classification lives in `pounce-sens-core`, written against
//! [`pounce_sens_core::barrier::BarrierIterate`], and is re-exported here so
//! `pounce_sensitivity::activity::WEAKLY_ACTIVE` and its siblings keep their
//! paths. `crate::barrier_iterate` fills that trait from pounce's own
//! converged iterate, and [`crate::Solver::row_normal`] and
//! [`crate::Solver::hessian_vec`] read the model in user space through
//! `pounce_sens_core::converged::ConvergedSolve`.

pub use pounce_sens_core::activity::*;
