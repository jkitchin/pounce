//! Sensitivity analysis for POUNCE — port of upstream Ipopt's `contrib/sIPOPT/`.
//!
//! # Status
//!
//! Phase A (current): trait surface for [`schur_data::SchurData`] and
//! [`p_calculator::PCalculator`] plus the `Index` flavors. Pure
//! data-shuttling; the IPM-side wiring lands in Phases B–E per
//! [pounce#7](https://github.com/jkitchin/pounce/issues/7).
//!
//! # Algorithmic reference
//!
//! Pirnay, H., López-Negrete, R., and Biegler, L.T. (2012).
//! *Optimal sensitivity based on IPOPT.*
//! Mathematical Programming Computation, **4**(4), 307–331.
//! [DOI: 10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2).
//!
//! Verified 2026-05-14 via Crossref: title, authors (Hans Pirnay; Rodrigo
//! López-Negrete; Lorenz T. Biegler), MPC volume 4 issue 4 pp 307–331.
//!
//! # Upstream source mirror
//!
//! Port targets `ref/Ipopt/contrib/sIPOPT/src/` in this repo
//! (EPL-2.0, © Hans Pirnay 2009–2011 per the file headers). Each
//! public item in this crate documents the upstream symbol it mirrors
//! with file path and (where stable) line numbers.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod algorithm_backsolver;
pub mod backsolver;
pub mod p_calculator;
pub mod reduced_hessian;
pub mod schur_data;
pub mod schur_driver;
pub mod sens_app;
pub mod step_calc;

pub use algorithm_backsolver::PdSensBacksolver;
pub use backsolver::{DenseLuBacksolver, SensBacksolver};
pub use p_calculator::{IndexPCalculator, PCalculator};
pub use reduced_hessian::compute_reduced_hessian;
pub use schur_data::{IndexSchurData, SchurData};
pub use schur_driver::{DenseGenSchurDriver, SchurDriver};
pub use sens_app::{register_options, SensApplication, SensOptions};
pub use step_calc::{SensStepCalc, StdStepCalc, WithBacksolver};
