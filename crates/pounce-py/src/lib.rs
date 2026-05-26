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

mod problem;
mod solver;
mod tnlp_bridge;
mod warm_start;

pub use problem::PyProblem;
pub use solver::PySolver;

/// Python module entry point. The crate name (`_pounce`) and the
/// `#[pymodule]` function name must agree; maturin uses the lib name
/// from Cargo.toml to find this symbol.
#[pymodule]
fn _pounce(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProblem>()?;
    m.add_class::<PySolver>()?;
    m.add_function(wrap_pyfunction!(warm_start::classify_working_set, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
