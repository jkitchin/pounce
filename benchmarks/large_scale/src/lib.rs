//! Synthetic large-scale NLP test problems, each implementing the
//! [`pounce_nlp::tnlp::TNLP`] trait.
//!
//! This is the modern port of the original `benchmarks/large_scale/problems.rs`
//! file (which targeted the long-since-removed `ripopt::NlpProblem` trait).
//! The math is preserved verbatim; only the trait surface changed.

pub mod problems;

pub use problems::{BratuProblem, ChainedRosenbrock, OptimalControl, PoissonControl, SparseQP};
