//! Library face of `pounce-cli`. Exists so the CLI's argv parser and
//! built-in problems can be unit-tested without invoking `main`.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod builtin;
pub mod cli;
pub mod counting_tnlp;
pub mod debug_repl;
pub mod nl_external;
pub mod nl_fbbt_translate;
pub mod nl_hessian_program;
pub mod nl_reader;
pub mod nl_tape;
pub mod nl_writer;
pub mod print;
pub mod seeded_tnlp;
pub mod sens;
pub mod solve_report;
