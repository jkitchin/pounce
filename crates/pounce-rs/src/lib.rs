//! # pounce-rs — solve nonlinear programs with POUNCE from Rust
//!
//! POUNCE's solver lives across several crates (`pounce-nlp` for the
//! [`TNLP`] problem trait, `pounce-algorithm` for the [`IpoptApplication`]
//! driver, `pounce-common` for the scalar types). This crate is a thin
//! **facade**: it re-exports everything needed to define and solve a problem,
//! so a Rust user depends on one crate and writes `use pounce_rs::prelude::*;`
//! — the Rust counterpart to the one-import `import pounce` Python API.
//!
//! It is re-exports only (no logic of its own), and it pins a single curated
//! public surface, so downstream code is insulated from churn in the internal
//! crate layout.
//!
//! ## Example: HS071 (Hock–Schittkowski problem 71)
//!
//! ```text
//! min  x1*x4*(x1 + x2 + x3) + x3
//! s.t. x1*x2*x3*x4 >= 25
//!      x1^2 + x2^2 + x3^2 + x4^2 == 40
//!      1 <= xi <= 5
//! ```
//!
//! ```
//! use pounce_rs::prelude::*;
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! #[derive(Default)]
//! struct Hs071 {
//!     obj: Option<f64>,
//!     x: Option<[f64; 4]>,
//! }
//!
//! impl TNLP for Hs071 {
//!     fn get_nlp_info(&mut self) -> Option<NlpInfo> {
//!         Some(NlpInfo { n: 4, m: 2, nnz_jac_g: 8, nnz_h_lag: 10, index_style: IndexStyle::C })
//!     }
//!
//!     fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
//!         b.x_l.copy_from_slice(&[1.0; 4]);
//!         b.x_u.copy_from_slice(&[5.0; 4]);
//!         b.g_l.copy_from_slice(&[25.0, 40.0]);          // g0 >= 25, g1 == 40
//!         b.g_u.copy_from_slice(&[2.0e19, 40.0]);
//!         true
//!     }
//!
//!     fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
//!         sp.x.copy_from_slice(&[1.0, 5.0, 5.0, 1.0]);
//!         true
//!     }
//!
//!     fn eval_f(&mut self, x: &[f64], _new_x: bool) -> Option<f64> {
//!         Some(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])
//!     }
//!
//!     fn eval_grad_f(&mut self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
//!         g[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
//!         g[1] = x[0] * x[3];
//!         g[2] = x[0] * x[3] + 1.0;
//!         g[3] = x[0] * (x[0] + x[1] + x[2]);
//!         true
//!     }
//!
//!     fn eval_g(&mut self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
//!         g[0] = x[0] * x[1] * x[2] * x[3];
//!         g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
//!         true
//!     }
//!
//!     fn eval_jac_g(&mut self, x: Option<&[f64]>, _new_x: bool, mode: SparsityRequest<'_>) -> bool {
//!         match mode {
//!             SparsityRequest::Structure { irow, jcol } => {
//!                 irow.copy_from_slice(&[0, 0, 0, 0, 1, 1, 1, 1]);
//!                 jcol.copy_from_slice(&[0, 1, 2, 3, 0, 1, 2, 3]);
//!             }
//!             SparsityRequest::Values { values } => {
//!                 let x = x.unwrap();
//!                 values.copy_from_slice(&[
//!                     x[1] * x[2] * x[3], x[0] * x[2] * x[3], x[0] * x[1] * x[3], x[0] * x[1] * x[2],
//!                     2.0 * x[0], 2.0 * x[1], 2.0 * x[2], 2.0 * x[3],
//!                 ]);
//!             }
//!         }
//!         true
//!     }
//!
//!     fn eval_h(&mut self, x: Option<&[f64]>, _new_x: bool, of: f64,
//!               lambda: Option<&[f64]>, _new_lambda: bool, mode: SparsityRequest<'_>) -> bool {
//!         match mode {
//!             SparsityRequest::Structure { irow, jcol } => {
//!                 irow.copy_from_slice(&[0, 1, 1, 2, 2, 2, 3, 3, 3, 3]);
//!                 jcol.copy_from_slice(&[0, 0, 1, 0, 1, 2, 0, 1, 2, 3]);
//!             }
//!             SparsityRequest::Values { values } => {
//!                 let x = x.unwrap();
//!                 let l = lambda.unwrap();
//!                 values.copy_from_slice(&[
//!                     of * (2.0 * x[3]) + l[1] * 2.0,
//!                     of * x[3] + l[0] * (x[2] * x[3]),
//!                     l[1] * 2.0,
//!                     of * x[3] + l[0] * (x[1] * x[3]),
//!                     l[0] * (x[0] * x[3]),
//!                     l[1] * 2.0,
//!                     of * (2.0 * x[0] + x[1] + x[2]) + l[0] * (x[1] * x[2]),
//!                     of * x[0] + l[0] * (x[0] * x[2]),
//!                     of * x[0] + l[0] * (x[0] * x[1]),
//!                     l[1] * 2.0,
//!                 ]);
//!             }
//!         }
//!         true
//!     }
//!
//!     fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
//!         self.obj = Some(sol.obj_value);
//!         self.x = Some([sol.x[0], sol.x[1], sol.x[2], sol.x[3]]);
//!     }
//! }
//!
//! let mut app = IpoptApplication::new();
//! app.initialize().unwrap();
//! let prob = Rc::new(RefCell::new(Hs071::default()));
//! let status = app.optimize_tnlp(Rc::clone(&prob) as Rc<RefCell<dyn TNLP>>);
//!
//! assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
//! let obj = prob.borrow().obj.unwrap();
//! assert!((obj - 17.014_017).abs() < 1e-4);            // known optimum
//! ```

// --- scalar types -----------------------------------------------------------
pub use pounce_common::types::{Index, Number};

// --- the problem trait and its supporting types -----------------------------
pub use pounce_nlp::return_codes::{AlgorithmMode, ApplicationReturnStatus};
pub use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, Linearity, MetaData, NlpInfo,
    ScalingRequest, Solution, SparsityRequest, StartingPoint, TNLP,
};

// --- the solver driver ------------------------------------------------------
pub use pounce_algorithm::application::IpoptApplication;

// --- the underlying crates, for anything not surfaced above -----------------
pub use pounce_algorithm;
pub use pounce_common;
pub use pounce_nlp;

// --- ergonomic builder API (argmin-style small trait + builder; #168) -------
pub mod builder;
pub use builder::{Nlp, Problem, Solution as NlpSolution};

/// The common case in one glob import. Brings in the ergonomic [`Problem`]
/// trait + [`Nlp`] builder, plus the low-level [`TNLP`] surface and the
/// [`IpoptApplication`] driver for full control.
///
/// ```
/// use pounce_rs::prelude::*;
/// ```
pub mod prelude {
    pub use crate::builder::{Nlp, Problem};
    pub use pounce_algorithm::application::IpoptApplication;
    pub use pounce_common::types::{Index, Number};
    pub use pounce_nlp::return_codes::ApplicationReturnStatus;
    pub use pounce_nlp::tnlp::{
        BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, ScalingRequest, Solution,
        SparsityRequest, StartingPoint, TNLP,
    };
}
