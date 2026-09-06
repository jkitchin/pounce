//! Algorithmic preprocessing — the `pounce-presolve` TNLP wrapper, re-exported.
//!
//! Presolve is normally invisible: [`crate::IpoptApplication::optimize_tnlp`]
//! calls [`wrap_from_options`] itself when `presolve=yes`, so a caller who
//! only wants the reduction sets the option and is done. This module is for
//! the caller who wants the *reports* — which bounds were tightened, whether
//! the equality rows have structural LICQ, whether infeasibility was proved
//! before a single iteration ran — or who drives the wrapper around a
//! [`TNLP`](crate::TNLP) of their own.
//!
//! ## Wrapping, and the one path that drops a phase
//!
//! [`wrap_with_presolve`] is the composition entry point: it builds a
//! [`PresolveTnlp`] **and** stacks the Phase-6 linear-equality elimination
//! ([`pounce_presolve::LinearEqElimTnlp`]) outside it. [`PresolveTnlp::new`]
//! builds the inner wrapper alone, so a caller who reaches for it directly
//! turns `presolve_linear_eq_reduction=yes` into a silent no-op — the option
//! is read, and nothing removes a column. Prefer `wrap_with_presolve`, or
//! [`wrap_from_options`] when the options already live in an
//! [`OptionsList`](pounce_common::options_list::OptionsList):
//!
//! ```
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! use pounce_rs::prelude::*;
//! use pounce_rs::presolve::{PresolveOptions, wrap_with_presolve};
//! # struct P;
//! # impl TNLP for P {
//! #     fn get_nlp_info(&mut self) -> Option<NlpInfo> {
//! #         Some(NlpInfo { n: 1, m: 0, nnz_jac_g: 0, nnz_h_lag: 1, index_style: IndexStyle::C })
//! #     }
//! #     fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
//! #         b.x_l[0] = -10.0; b.x_u[0] = 10.0; true
//! #     }
//! #     fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool { sp.x[0] = 0.0; true }
//! #     fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> { Some((x[0] - 1.0).powi(2)) }
//! #     fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
//! #         g[0] = 2.0 * (x[0] - 1.0); true
//! #     }
//! #     fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool { true }
//! #     fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, _m: SparsityRequest<'_>) -> bool { true }
//! #     fn eval_h(
//! #         &mut self, _x: Option<&[Number]>, _n: bool, obj: Number,
//! #         _l: Option<&[Number]>, _nl: bool, mode: SparsityRequest<'_>,
//! #     ) -> bool {
//! #         match mode {
//! #             SparsityRequest::Structure { irow, jcol } => { irow[0] = 0; jcol[0] = 0; }
//! #             SparsityRequest::Values { values } => values[0] = 2.0 * obj,
//! #         }
//! #         true
//! #     }
//! #     fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
//! # }
//!
//! let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(P));
//! let wrapped = wrap_with_presolve(inner, PresolveOptions::defaults())?;
//!
//! let mut app = IpoptApplication::new();
//! app.initialize()?;
//! // The wrapper is already on; without this the driver adds a second one.
//! app.set_presolve_already_applied(true);
//! assert_eq!(app.optimize_tnlp(wrapped), ApplicationReturnStatus::SolveSucceeded);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## The `presolve_already_applied` protocol
//!
//! Because `optimize_tnlp` wraps on its own when `presolve=yes`, a caller who
//! has already wrapped must say so with
//! [`crate::IpoptApplication::set_presolve_already_applied`] — otherwise the
//! problem is presolved twice. (The inner wrapper reports itself through
//! `TNLP::is_presolve_wrapper`, so the second wrap is a no-op whenever the
//! wrapper is the *outermost* layer; the flag is what covers the case where
//! it is not.)
//!
//! ## Reading the reports
//!
//! The accessors that make preprocessing legible hang off the concrete
//! [`PresolveTnlp`], not off `dyn TNLP`, so reading them means holding the
//! concrete wrapper: [`PresolveTnlp::licq_verdict`] → [`LicqVerdict`],
//! [`PresolveTnlp::tighten_report`] → [`TightenReport`],
//! [`PresolveTnlp::auxiliary_diagnostics`] →
//! [`AuxiliaryPreprocessingDiagnostics`], [`PresolveTnlp::cached_bounds`] →
//! [`CachedBounds`], [`PresolveTnlp::fbbt_report`] → [`FbbtReport`], and
//! [`PresolveTnlp::certified_infeasible`] → `InfeasibilityProof`. Every one
//! of those types is named here, so a report can be bound, matched, and
//! stored rather than only `{:?}`-printed.
//!
//! Holding the concrete wrapper and still getting Phase 6 means stacking the
//! elimination yourself with [`pounce_presolve::LinearEqElimTnlp`] — which is
//! exactly what `wrap_with_presolve` does.
//!
//! [`pounce_presolve`] itself is re-exported for anything not surfaced here.

pub use pounce_nlp::expression_provider::{ExpressionProvider, FbbtOp, FbbtTape};
pub use pounce_presolve::fbbt::FbbtReport;
pub use pounce_presolve::{
    AuxiliaryPreprocessingDiagnostics, CachedBounds, LicqVerdict, LinearEqElimTnlp, PresolveError,
    PresolveOptions, PresolveTnlp, TightenReport, wrap_from_options, wrap_with_presolve,
    wrap_with_presolve_provider,
};

/// The underlying crate, for anything not surfaced above.
pub use pounce_presolve;
