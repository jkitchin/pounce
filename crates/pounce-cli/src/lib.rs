//! Library face of `pounce-cli`. Exists so the CLI's argv parser and
//! built-in problems can be unit-tested without invoking `main`.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod builtin;
pub mod cbf;
pub mod citations;
pub mod cli;
pub mod counting_tnlp;
pub mod debug_repl;
// The `.nl` pipeline (reader, AD tape, external functions, FBBT lowering)
// now lives in the leaf `pounce-nl` crate so the Python bindings can reuse
// it. Re-export the modules so existing `crate::nl_reader::…` /
// `pounce_cli::nl_reader::…` paths keep resolving unchanged.
pub use pounce_nl::{nl_external, nl_fbbt_translate, nl_reader, nl_tape};
pub mod dispatch;
pub mod minima;
pub mod nl_hessian_program;
pub mod nl_writer;
pub mod print;
pub mod qp_extract;
pub mod seeded_tnlp;
pub mod sens;
pub mod solve_report;
pub mod verify;
