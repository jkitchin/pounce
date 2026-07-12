#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
//! Pure-Rust analysis core for pounce-studio.
//!
//! Loads `pounce.solve-report/v1` JSON (see
//! `crates/pounce-cli/src/solve_report.rs` for the writer) and parses
//! `POUNCEIT v1` binary iter-dumps (see `tools/iter-dump/FORMAT.md`),
//! then exposes derived series: convergence-stall detection, restoration
//! window extraction, common-failure-mode diagnostics, side-by-side
//! comparisons, and a Markdown summary renderer.
//!
//! The library is intentionally WASM-clean: it takes byte slices and
//! returns owned data, never touching `std::fs`. The bundled
//! `pounce-studio` binary in `src/bin/` is the file-I/O front-end.
//!
//! # Versioning
//!
//! The JSON schema is pinned to [`SOLVE_REPORT_SCHEMA`]; loading any
//! other tag is rejected with [`Error::SchemaMismatch`]. The binary
//! format is pinned to [`iter_dump::FORMAT_VERSION`]. Both can be
//! widened additively (new optional fields) without bumping; breaking
//! changes bump the major version and add a new branch here.

pub mod analysis;
pub mod glossary;
pub mod iter_dump;
pub mod markdown;
pub mod preflight;
pub mod report;

pub use analysis::{
    Finding, Severity, Stall, Summary, compare_runs, convergence_trace, diagnose, find_stalls,
    get_iterate, restoration_windows, summarize,
};
pub use iter_dump::{IterDumpHeader, IterDumpRecord, IterDumpTrace};
pub use report::{Error, IterRecord, SOLVE_REPORT_SCHEMA, SolveReport};
