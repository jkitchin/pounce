//! PyO3 bindings for POUNCE.
//!
//! Exposes:
//!
//! * [`PyProblem`] — cyipopt-compatible `Problem` class wrapping
//!   [`pounce_algorithm::IpoptApplication`].
//! * `solve(...)` standalone function and `_options_keys()` for
//!   introspection from the Python facade.
//!
//! The Rust → Python callback bridge lives in [`tnlp_bridge`], which
//! implements [`pounce_nlp::TNLP`] in terms of held `Py<PyAny>` callables.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
// PyO3 0.22's `#[pyclass]` / `#[pymethods]` macros expand to code that
// trips `unsafe_op_in_unsafe_fn` (and a handful of `clippy::needless_*`
// lints) — the workspace lint level otherwise treats these as warnings.
// Suppress them here so a clean `cargo check -p pounce-py` is achievable.
#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

mod nl_problem;
mod nlp_batch;
mod problem;
mod qp;
mod solver;
mod sos;
mod tnlp_bridge;
mod warm_start;

pub use nl_problem::{read_nl, PyNlProblem};
pub use problem::PyProblem;
pub use qp::{PyQpFactorization, PyQpProblem, PyQpSensitivity};
pub use solver::PySolver;

/// Python module entry point. The crate name (`_pounce`) and the
/// `#[pymodule]` function name must agree; maturin uses the lib name
/// from Cargo.toml to find this symbol.
#[pymodule]
fn _pounce(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Install the tracing subscriber on import so Python consumers get
    // logging and the iteration collector (pounce#71). Idempotent, so
    // re-imports / sub-interpreters are safe.
    pounce_observability::init_subscriber();
    m.add_class::<PyProblem>()?;
    m.add_class::<PySolver>()?;
    m.add_class::<PyNlProblem>()?;
    m.add_function(wrap_pyfunction!(read_nl, m)?)?;
    // Batched NLP solving (pounce#126): native `.nl` path (phase 1)
    // and callback-Problem path (phase 2).
    m.add_function(wrap_pyfunction!(nlp_batch::solve_nlp_batch, m)?)?;
    m.add_function(wrap_pyfunction!(nlp_batch::solve_problem_batch, m)?)?;
    m.add_function(wrap_pyfunction!(warm_start::classify_working_set, m)?)?;
    // Convex LP/QP solver (pounce-convex) bindings.
    m.add_class::<PyQpProblem>()?;
    m.add_class::<PyQpFactorization>()?;
    m.add_class::<PyQpSensitivity>()?;
    m.add_function(wrap_pyfunction!(qp::solve_qp, m)?)?;
    m.add_function(wrap_pyfunction!(qp::solve_socp, m)?)?;
    m.add_function(wrap_pyfunction!(qp::solve_qp_batch, m)?)?;
    m.add_function(wrap_pyfunction!(qp::solve_qp_multi_rhs, m)?)?;
    // SOS polynomial global optimizer (pounce-convex::sos).
    m.add_function(wrap_pyfunction!(sos::sos_minimize, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Single source of truth for the differentiable-solve active-set
    // tolerance (the `DiffHandoff` contract). The JAX / torch frontends
    // import this instead of each hardcoding `1e-6`, so the producer's
    // `info["active_tol"]` and every consumer's threshold can never drift.
    m.add("DEFAULT_ACTIVE_TOL", pounce_sensitivity::DEFAULT_ACTIVE_TOL)?;
    Ok(())
}
