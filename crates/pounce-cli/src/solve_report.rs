//! Compatibility shim — the report writer now lives in `pounce-solve-report`
//! so the GAMS solver link (via `pounce-cinterface`) can emit reports too.
//! This re-export preserves all `pounce_cli::solve_report::*` call sites.

pub use pounce_solve_report::*;
