//! AMPL `.nl` reader, reverse-mode AD tape, and TNLP evaluator.
//!
//! This crate holds the `.nl` pipeline that used to live in `pounce-cli`:
//!
//! - [`nl_reader`] parses a `.nl` file into an [`nl_reader::NlProblem`] (the
//!   `Expr` DAG, linear parts, bounds, names, starting point) and provides
//!   [`nl_reader::NlTnlp`], a [`pounce_nlp::tnlp::TNLP`] implementation that
//!   evaluates objective/gradient/Hessian and constraints/Jacobian.
//! - [`nl_tape`] flattens an `Expr` DAG into a reverse-mode AD tape with
//!   colored forward-over-reverse Hessian products.
//! - [`nl_external`] supports AMPL imported (external) functions via the
//!   `funcadd_ASL` ABI.
//! - [`nl_fbbt_translate`] lowers an `Expr` to an `FbbtTape` for
//!   feasibility-based bound tightening.
//!
//! It is a leaf crate (depends only on `pounce-common` and `pounce-nlp`) so
//! both the CLI and the Python bindings can read and evaluate `.nl` models
//! without depending on each other.

pub mod nl_external;
pub mod nl_fbbt_translate;
pub mod nl_reader;
pub mod nl_tape;
