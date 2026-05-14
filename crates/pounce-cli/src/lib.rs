//! Library face of `pounce-cli`. Exists so the CLI's argv parser and
//! built-in problems can be unit-tested without invoking `main`.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod builtin;
pub mod cli;
pub mod counting_tnlp;
pub mod nl_hessian_program;
pub mod nl_reader;
pub mod nl_tape;
pub mod nl_writer;
pub mod print;
pub mod solve_report;
