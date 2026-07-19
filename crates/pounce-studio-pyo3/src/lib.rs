//! PyO3 bindings for `pounce-studio-core`.
//!
//! Built as the `_native` extension module of the `pounce-studio-mcp`
//! Python package (see `studio/mcp/pyproject.toml`). The Python MCP
//! server in `studio/mcp/pounce_studio_mcp/` delegates all analysis
//! through these bindings so the Rust core is the single source of
//! truth for derived series and diagnostics.
//!
//! FFI strategy: the bindings return JSON strings rather than Python
//! objects, and the Python wrapper does the `json.loads`. Trivial code
//! on both sides, no `pythonize` dep, plenty fast at our data sizes
//! (a Full-detail solve report is a few hundred KB).
//!
//! Method names intentionally do **not** carry a `_json` suffix: the
//! Python wrapper hides marshalling, so from a caller's perspective
//! `Report.summarize()` and `R.summarize(report)` are the same
//! operation. The Rust-side return is a string; the Python wrapper
//! returns the parsed dict.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#![allow(unsafe_op_in_unsafe_fn)]

use std::cell::OnceCell;

use pounce_studio_core as core;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Convert a `core::Error` to a Python exception, choosing the closest
/// builtin so callers can `except` against familiar types.
fn err_to_py(e: core::Error) -> PyErr {
    match e {
        core::Error::Json(_) | core::Error::SchemaMismatch { .. } | core::Error::IterDump(_) => {
            PyValueError::new_err(e.to_string())
        }
        core::Error::IterOutOfRange { .. } | core::Error::NoIterations => {
            PyValueError::new_err(e.to_string())
        }
    }
}

fn json_or_err<T: serde::Serialize>(value: T) -> PyResult<String> {
    serde_json::to_string(&value).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Cache that holds a computed JSON string for the lifetime of a
/// [`PyReport`]. Uses `OnceCell` so successful caching is one-shot;
/// errors are not cached (a future call re-tries the computation).
type JsonCache = OnceCell<String>;

fn cached<F>(cell: &JsonCache, compute: F) -> PyResult<String>
where
    F: FnOnce() -> PyResult<String>,
{
    if let Some(s) = cell.get() {
        return Ok(s.clone());
    }
    let s = compute()?;
    let _ = cell.set(s.clone()); // ignore "already set" race (GIL prevents it anyway)
    Ok(s)
}

/// A parsed `pounce.solve-report/v1` JSON document.
///
/// Construct via [`Report::from_bytes`] or [`Report::from_path`]. The
/// instance owns the parsed [`core::SolveReport`]; analysis methods
/// borrow it without re-parsing, and parameter-less results
/// (`summarize`, `convergence_trace`, `restoration_windows`,
/// `diagnose`, `render_markdown`) are memoised per-instance for cheap
/// repeat MCP-tool calls.
#[pyclass(name = "Report", unsendable)]
struct PyReport {
    inner: core::SolveReport,
    summary_cache: JsonCache,
    trace_cache: JsonCache,
    restoration_cache: JsonCache,
    diagnose_cache: JsonCache,
    markdown_cache: OnceCell<String>,
}

impl PyReport {
    fn from_inner(inner: core::SolveReport) -> Self {
        Self {
            inner,
            summary_cache: JsonCache::new(),
            trace_cache: JsonCache::new(),
            restoration_cache: JsonCache::new(),
            diagnose_cache: JsonCache::new(),
            markdown_cache: OnceCell::new(),
        }
    }
}

#[pymethods]
impl PyReport {
    #[staticmethod]
    fn from_bytes(bytes: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let inner = core::SolveReport::from_json_slice(bytes.as_bytes()).map_err(err_to_py)?;
        Ok(Self::from_inner(inner))
    }

    #[staticmethod]
    fn from_path(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let inner = core::SolveReport::from_json_slice(&bytes).map_err(err_to_py)?;
        Ok(Self::from_inner(inner))
    }

    fn summarize(&self) -> PyResult<String> {
        cached(&self.summary_cache, || {
            json_or_err(core::summarize(&self.inner))
        })
    }

    fn convergence_trace(&self) -> PyResult<String> {
        cached(&self.trace_cache, || {
            json_or_err(core::analysis::convergence_trace(&self.inner))
        })
    }

    #[pyo3(signature = (min_window = None, max_log10_progress = None))]
    fn find_stalls(
        &self,
        min_window: Option<usize>,
        max_log10_progress: Option<f64>,
    ) -> PyResult<String> {
        // Partial overrides are honoured: passing only one of the two
        // tuneables substitutes the default for the other. Defaults
        // mirror `core::find_stalls`. Not memoised because results are
        // parameter-dependent.
        let min_window = min_window.unwrap_or(5);
        // Reject rather than clamp: a stall spans consecutive iterations, so
        // `min_window < 2` is not a smaller request but an ill-posed one, and
        // silently honouring it reports every iteration of a healthy solve as
        // a stall.
        if min_window < core::analysis::MIN_STALL_WINDOW {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "min_window must be >= {} (a stall spans at least two \
                 consecutive iterations); got {min_window}",
                core::analysis::MIN_STALL_WINDOW
            )));
        }
        let max_log10_progress = max_log10_progress.unwrap_or(0.3);
        json_or_err(core::analysis::find_stalls_with(
            &self.inner,
            min_window,
            max_log10_progress,
        ))
    }

    fn restoration_windows(&self) -> PyResult<String> {
        cached(&self.restoration_cache, || {
            json_or_err(core::analysis::restoration_windows(&self.inner))
        })
    }

    fn diagnose(&self) -> PyResult<String> {
        cached(&self.diagnose_cache, || {
            json_or_err(core::diagnose(&self.inner))
        })
    }

    fn get_iterate(&self, k: usize) -> PyResult<String> {
        // Not memoised — parameter-dependent.
        let aug = core::get_iterate(&self.inner, k).map_err(err_to_py)?;
        json_or_err(aug)
    }

    fn render_markdown(&self) -> String {
        if let Some(s) = self.markdown_cache.get() {
            return s.clone();
        }
        let s = core::markdown::render_inspect(&self.inner);
        let _ = self.markdown_cache.set(s.clone());
        s
    }

    /// Aggregate linear-solver post-mortem from the report's
    /// `linear_solver` field, as a JSON object string. `None` when the
    /// report carried no `linear_solver` block (older reports, or the
    /// solve used a backend that doesn't self-instrument — HSL MA57 or
    /// a custom factory).
    fn linear_solver_summary(&self) -> PyResult<Option<String>> {
        match &self.inner.linear_solver {
            Some(s) => json_or_err(s).map(Some),
            None => Ok(None),
        }
    }
}

/// A parsed POUNCEIT v1 binary trace.
#[pyclass(name = "IterDump", unsendable)]
struct PyIterDump {
    inner: core::IterDumpTrace,
    header_cache: JsonCache,
    records_cache: JsonCache,
}

impl PyIterDump {
    fn from_inner(inner: core::IterDumpTrace) -> Self {
        Self {
            inner,
            header_cache: JsonCache::new(),
            records_cache: JsonCache::new(),
        }
    }
}

#[pymethods]
impl PyIterDump {
    #[staticmethod]
    fn from_bytes(bytes: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let inner = core::IterDumpTrace::from_bytes(bytes.as_bytes()).map_err(err_to_py)?;
        Ok(Self::from_inner(inner))
    }

    #[staticmethod]
    fn from_path(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let inner = core::IterDumpTrace::from_bytes(&bytes).map_err(err_to_py)?;
        Ok(Self::from_inner(inner))
    }

    fn header(&self) -> PyResult<String> {
        cached(&self.header_cache, || json_or_err(&self.inner.header))
    }

    fn records(&self) -> PyResult<String> {
        cached(&self.records_cache, || json_or_err(&self.inner.records))
    }

    fn record_count(&self) -> usize {
        self.inner.records.len()
    }
}

/// Compare a sequence of `(label, Report)` pairs side-by-side. Returns
/// a JSON-encoded list of comparison rows.
///
/// Takes `Py<PyReport>` (owned reference) rather than `PyRef` so the
/// same `Report` can appear in multiple entries — e.g.
/// `compare([("a", r), ("b", r)])` — without panicking on overlapping
/// `RefCell` borrows. The actual borrow happens per-iteration and is
/// dropped immediately after the inner `SolveReport` is cloned out.
#[pyfunction]
fn compare_reports(py: Python<'_>, pairs: Vec<(String, Py<PyReport>)>) -> PyResult<String> {
    let owned: Vec<(String, core::SolveReport)> = pairs
        .iter()
        .map(|(label, handle)| {
            let bound = handle.bind(py);
            let r = bound.borrow();
            (label.clone(), r.inner.clone())
        })
        .collect();
    let rows = core::compare_runs(owned.iter().map(|(l, r)| (l.as_str(), r)));
    json_or_err(rows)
}

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReport>()?;
    m.add_class::<PyIterDump>()?;
    m.add_function(wrap_pyfunction!(compare_reports, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "SOLVE_REPORT_SCHEMA",
        pounce_studio_core::report::SOLVE_REPORT_SCHEMA,
    )?;
    Ok(())
}
