//! Multistart / find-minima global search — `pounce-minima`, re-exported
//! (feature `minima`).
//!
//! ```toml
//! [dependencies]
//! pounce-rs = { version = "0.12", features = ["minima"] }
//! ```
//!
//! [`find_minima`] drives the NLP solver from many starts and returns the
//! distinct local minima it found, deduplicated and ranked best first. It is
//! the search `pounce model.nl --minima <method>` runs and the algorithm of
//! Python's `pounce.find_minima`; [`Method`] picks the strategy (multistart,
//! MLSL, basin-hopping, flooding, deflation, tunneling) and [`MinimaOptions`]
//! carries its knobs, with the same defaults as the CLI.
//!
//! ```no_run
//! use pounce_rs::minima::{IpoptLocalSolve, Method, MinimaOptions, find_minima};
//! use pounce_rs::prelude::*;
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! # fn demo(problem: Rc<RefCell<dyn TNLP>>) {
//! let mut app = IpoptApplication::new();
//! app.initialize().unwrap();
//!
//! let opts = MinimaOptions { method: Method::Multistart, n_minima: 6, ..Default::default() };
//! let mut solver = IpoptLocalSolve::new(&mut app);
//! let result = find_minima(&problem, &mut solver, &opts).unwrap();
//! for m in &result.minima {
//!     println!("f = {:.6}  at {:?}", m.objective, m.x);
//! }
//! println!("stopped: {}", result.stop.as_str());
//! # }
//! ```
//!
//! The search reads the box and starting point from the TNLP and draws its
//! further starts from that box, so bound the variables whose range you want
//! explored. A start is accepted as a minimum when its solve converged, the
//! solver's unscaled stationarity residual is within `kkt_tol`, the point is
//! in bounds, the objective Hessian is positive semidefinite (when one is
//! available and `n <= 256`), and it is not a duplicate. Each minimum carries
//! the base problem's constraint duals, recovered by a clean re-solve.
//!
//! [`IpoptLocalSolve`] runs every local solve on the one application you hand
//! it, toggling `hessian_approximation` per solve, so the application's other
//! options (tolerances, `print_level`, …) apply to all of them. To solve
//! starts some other way, implement [`LocalSolve`]; overriding
//! [`LocalSolve::should_stop`] lets it cancel the search cooperatively (for
//! example at a deadline), keeping the minima accepted so far.
//!
//! [`find_minima_with`] takes [`Hooks`]: a callback on each accepted minimum
//! (which can stop the search), a custom dedup metric such as a periodic
//! distance, and a [`TraceEntry`] per candidate considered.

pub use pounce_minima::{
    HessianMode, Hooks, IpoptLocalSolve, LocalOutcome, LocalSolve, Method, MinimaError,
    MinimaOptions, MinimaResult, Minimum, SolveQuality, Stop, TraceEntry, find_minima,
    find_minima_with,
};
