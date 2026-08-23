//! Library face of `pounce-cli`. Exists so the CLI's argv parser and
//! built-in problems can be unit-tested without invoking `main`.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

// The built-in TNLP test problems moved to `pounce-nlp`: they are
// self-contained `impl TNLP`s with no CLI or `.nl` coupling, and an embedder
// wiring up their own TNLP wants a known-good problem to check against just
// as much as this CLI does. The CBF reader moved to `pounce-convex`, next to
// the `QpProblem` / `ConeSpec` types it builds and the conic solver that
// consumes them. Re-exported under their historical names so existing
// `crate::builtin::…` / `pounce_cli::cbf::…` paths keep resolving unchanged.
pub use pounce_convex::cbf;
pub use pounce_nlp::builtin;
pub mod check_x0;
pub mod citations;
pub mod cli;
pub mod debug_repl;
// The `.nl` pipeline (reader, AD tape, external functions, FBBT lowering)
// now lives in the leaf `pounce-nl` crate so the Python bindings can reuse
// it. Re-export the modules so existing `crate::nl_reader::…` /
// `pounce_cli::nl_reader::…` paths keep resolving unchanged.
pub use pounce_nl::{nl_external, nl_fbbt_translate, nl_quadratic, nl_reader, nl_tape};
// The AMPL `.sol` writer moved to `pounce-nl` alongside the `.nl` reader it
// inverts, so the wasm frontend can emit the same file. Re-exported under its
// historical name to keep `nl_writer::…` call sites resolving.
pub use pounce_nl::sol_writer as nl_writer;
pub mod dispatch;
pub mod minima;
pub mod nl_hessian_program;
pub mod print;
pub mod qp_extract;
pub mod sens;
pub mod solve_report;
pub mod verify;
