//! Second-opinion recovery — the `pounce-restoration` ladder, re-exported.
//!
//! A failing verdict is not always the model's fault. When a solve returns
//! `Infeasible_Problem_Detected`, `Invalid_Number_Detected`, or another
//! status the ladder opens on, [`run_second_opinion_ladder`] re-solves along
//! up to four deliberately different trajectories — different scaling, a
//! different barrier strategy, a displaced starting point — and promotes one
//! **only if it converges**. A converged solve pays nothing: the ladder reads
//! the status and returns without touching the application.
//!
//! ## You probably already have this
//!
//! [`crate::builder::Nlp::solve`] runs the ladder itself and reports what it
//! did on [`crate::builder::Solution::second_opinion`], as the CLI and the
//! Python and C frontends do. This module is for the caller who drives
//! [`IpoptApplication`](crate::IpoptApplication) directly — that path has no
//! ladder of its own, so a low-level embedder gets the bare verdict unless it
//! calls this.
//!
//! ```no_run
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! use pounce_rs::prelude::*;
//! use pounce_rs::restoration::{SecondOpinionOutcome, run_second_opinion_ladder};
//! # fn demo(mut app: IpoptApplication, tnlp: Rc<RefCell<dyn TNLP>>) {
//!
//! let status = app.optimize_tnlp(Rc::clone(&tnlp));
//! let stats = app.statistics();
//!
//! // Pass the *same* TNLP the original solve ran on. The closure receives one
//! // progress line per event; a no-op closure runs the ladder silently.
//! let outcome: SecondOpinionOutcome =
//!     run_second_opinion_ladder(&mut app, tnlp, status, stats, &mut |_line| {});
//!
//! if outcome.ran() {
//!     // `outcome.statistics` is the *shipped* solve's alone, so the true cost
//!     // is `total_iteration_count()`, and `base_status` is the only trace
//!     // left that the base solver did not converge (gh#850).
//!     println!(
//!         "{:?} -> {:?} via {:?}, {} iterations in total",
//!         outcome.base_status,
//!         outcome.status,
//!         outcome.promoted_by,
//!         outcome.total_iteration_count(),
//!     );
//! }
//! # }
//! ```
//!
//! ## What the driver guarantees
//!
//! Each rung writes `feral_scaling`, `mu_strategy` and
//! `start_point_perturbation` into the live options list, and the driver
//! **restores them** afterwards — to their set-ness, not to a resolved value,
//! so an env-configured run is not overridden. An application that solves
//! twice therefore does not inherit a rung's settings on the second solve.
//! A rejected rung never reaches the caller's TNLP.
//!
//! **Not for multi-start drivers.** A failed start is routine in a
//! multi-start search, and paying up to four extra solves per failed start
//! multiplies cost for no benefit.
//!
//! [`pounce_restoration`] itself is re-exported for anything not surfaced
//! here — the restoration sub-IPM, its algorithm builder, and the inner
//! solver factories.

pub use pounce_restoration::second_opinion_driver::{
    SecondOpinionOutcome, run_second_opinion_ladder,
};

/// The underlying crate, for anything not surfaced above.
pub use pounce_restoration;
