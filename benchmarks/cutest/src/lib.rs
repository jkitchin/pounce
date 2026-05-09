//! CUTEst harness for POUNCE — Rust wrappers around the CUTEst Fortran
//! interface plus a [`TNLP`](pounce_nlp::tnlp::TNLP) adapter that lets
//! the standard MASTSIF problem set drive `IpoptApplication`. The
//! comparison binary lives at `src/bin/cutest_suite.rs`.

pub mod cutest_ffi;
pub mod cutest_problem;

pub use cutest_problem::CutestProblem;
