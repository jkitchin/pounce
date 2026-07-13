//! Re-export of the shared Ipopt-style console printers.
//!
//! The printer implementations live in [`pounce_solve_report::console`] so
//! the algorithm's output layer can emit the problem-statistics and
//! end-of-run summary blocks itself, gated on `print_level` (#206). The CLI
//! still owns the up-front banner and the convex-QP summary, and re-exports
//! the shared functions here so existing `print::…` call sites are
//! unchanged. This module carries no implementation of its own — the single
//! source of truth is `console`.
pub use pounce_solve_report::console::*;
