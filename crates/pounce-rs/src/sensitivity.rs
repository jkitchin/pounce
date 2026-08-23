//! NLP sensitivity, parametric warm starts, and reduced Hessians — the
//! `pounce-sensitivity` port of Ipopt's sIPOPT, re-exported (feature
//! `sensitivity`).
//!
//! ```toml
//! [dependencies]
//! pounce-rs = { version = "0.9", features = ["sensitivity"] }
//! ```
//!
//! The problem shape sIPOPT assumes: the TNLP declares equality constraints
//! of the form `g_i(x) − p_i = 0`, and the 0-based row indices `i` of those
//! constraints are the *pins*. [`SensSolve`] solves the NLP once and then,
//! from the converged KKT factor, returns a first-order predictor
//! `Δx ≈ (∂x*/∂p)·Δp` for a perturbation of the pinned parameters — no
//! re-solve, one Schur complement.
//!
//! ```no_run
//! use pounce_rs::prelude::*;
//! use pounce_rs::sensitivity::SensSolve;
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! # fn demo(tnlp: Rc<RefCell<dyn TNLP>>) {
//! let mut app = pounce_rs::application();
//! app.initialize().unwrap();
//!
//! let result = SensSolve::new(vec![2, 3])          // pinned constraint rows
//!     .with_deltas(vec![-0.5, 0.0])                // Δp
//!     .with_reduced_hessian()
//!     .run(&mut app, tnlp);
//!
//! assert_eq!(result.status, ApplicationReturnStatus::SolveSucceeded);
//! let dx = result.dx.expect("Δx populated when with_deltas was set");
//! # }
//! ```
//!
//! A sensitivity-stage failure is reported through [`SensResult::error`], not
//! through `status` — the underlying solve can converge while the post-solve
//! step fails (a pin that isn't an equality, a Schur setup error). Both
//! "sensitivity failed" and "sensitivity not requested" leave the outputs
//! `None`, so `error` is the only signal that separates them.
//!
//! [`compute_reduced_hessian`] gives the curvature on the null space of the
//! active constraints directly, and
//! [`SensSolve::with_reduced_hessian_eigen`] adds its eigendecomposition via
//! the shared [`symmetric_eigen`].
//!
//! For the long form — [`SensApplication`] driven by an explicit
//! [`IndexSchurData`] / [`PdSensBacksolver`] / [`DenseGenSchurDriver`] stack —
//! every piece is re-exported below, and [`pounce_sensitivity`] itself is
//! re-exported for anything not listed.

pub use pounce_sensitivity::{
    ConvergedState, DEFAULT_ACTIVE_TOL, DenseGenSchurDriver, DenseLuBacksolver, DiffHandoff,
    IndexPCalculator, IndexSchurData, PCalculator, PdSensBacksolver, SchurData, SchurDriver,
    SensApplication, SensBacksolver, SensOptions, SensResult, SensSolve, SensStepCalc, Solver,
    SolverError, StdStepCalc, WithBacksolver, compute_reduced_hessian, register_options,
    symmetric_eigen,
};

/// The underlying crate, for anything not surfaced above.
pub use pounce_sensitivity;
