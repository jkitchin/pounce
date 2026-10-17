//! POUNCE's multistart / find-minima global search.
//!
//! [`find_minima`] drives a local solver in a loop — six strategies:
//! multistart, MLSL, basin-hopping, flooding, deflation, tunneling — and
//! returns the distinct local minima it found, deduplicated and ranked. The
//! local solves go through the [`LocalSolve`] trait; [`IpoptLocalSolve`] runs
//! them on a reused [`IpoptApplication`](pounce_algorithm::application::IpoptApplication).
//!
//! This is the search the `pounce --minima` CLI runs, extracted from
//! `pounce-cli` (gh#966) so every frontend shares one implementation instead
//! of each carrying its own. `pounce-rs` re-exports it behind its `minima`
//! feature.

pub mod archive;
mod capture_tnlp;
pub mod local_solve;
pub mod options;
pub mod penalty_tnlp;
pub mod sampling;
pub mod search;

pub use local_solve::{HessianMode, IpoptLocalSolve, LocalOutcome, LocalSolve, SolveQuality};
pub use options::{Method, MinimaOptions};
pub use search::{
    Hooks, MinimaError, MinimaResult, Minimum, Stop, TraceEntry, find_minima, find_minima_with,
};
