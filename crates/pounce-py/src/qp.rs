//! PyO3 bindings for the convex LP/QP solver (`pounce-convex`).
//!
//! Exposes the standard-form convex QP
//!
//! ```text
//! minimize    ½ xᵀP x + cᵀx
//! subject to  A x = b,  G x ≤ h,  lb ≤ x ≤ ub
//! ```
//!
//! as a Python `QpProblem`, with one-shot `solve_qp`, the batched /
//! multiple-RHS entry points (`solve_qp_batch`, `solve_qp_multi_rhs`),
//! and the build-once / solve-many `QpFactorization` handle — the same
//! capabilities the Rust crate offers, including the parallel batch.
//!
//! Sparse matrices are passed as COO triplets `(rows, cols, vals)` (three
//! equal-length sequences), matching how scipy `coo_matrix` exposes its
//! data; `P` is the **lower triangle** of the symmetric Hessian.

use numpy::IntoPyArray;
use pounce_convex::{
    solve_qp_batch_parallel, solve_qp_batch_parallel_warm, solve_qp_ipm, solve_qp_ipm_warm,
    QpFactorization, QpOptions, QpProblem, QpSolution, QpStatus, QpWarmStart, Triplet,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Build a triplet list from `(rows, cols, vals)`, validating equal
/// lengths and (for `lower_only`) that no strict-upper entry is given.
fn triplets(
    rows: &[i64],
    cols: &[i64],
    vals: &[f64],
    what: &str,
    lower_only: bool,
) -> PyResult<Vec<Triplet>> {
    if rows.len() != cols.len() || rows.len() != vals.len() {
        return Err(PyValueError::new_err(format!(
            "{what}: rows/cols/vals must have equal length ({}, {}, {})",
            rows.len(),
            cols.len(),
            vals.len()
        )));
    }
    let mut out = Vec::with_capacity(rows.len());
    for k in 0..rows.len() {
        let (r, c) = (rows[k], cols[k]);
        if r < 0 || c < 0 {
            return Err(PyValueError::new_err(format!(
                "{what}: negative index at entry {k}"
            )));
        }
        let (r, c) = (r as usize, c as usize);
        if lower_only && c > r {
            return Err(PyValueError::new_err(format!(
                "{what}: entry ({r},{c}) is in the strict upper triangle; \
                 pass only the lower triangle of the symmetric Hessian P"
            )));
        }
        out.push(Triplet::new(r, c, vals[k]));
    }
    Ok(out)
}

/// Convex QP in standard form. Construct from dense `c` and COO triplets
/// for `P` (lower triangle), `A`, and `G`; `b`, `h`, `lb`, `ub` are dense
/// (omit `lb`/`ub` or pass empty for unbounded).
#[pyclass(name = "QpProblem", module = "pounce._pounce")]
#[derive(Clone)]
pub struct PyQpProblem {
    inner: QpProblem,
}

#[pymethods]
impl PyQpProblem {
    #[new]
    #[pyo3(signature = (
        n, c,
        p_rows=vec![], p_cols=vec![], p_vals=vec![],
        a_rows=vec![], a_cols=vec![], a_vals=vec![], b=vec![],
        g_rows=vec![], g_cols=vec![], g_vals=vec![], h=vec![],
        lb=vec![], ub=vec![],
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: usize,
        c: Vec<f64>,
        p_rows: Vec<i64>,
        p_cols: Vec<i64>,
        p_vals: Vec<f64>,
        a_rows: Vec<i64>,
        a_cols: Vec<i64>,
        a_vals: Vec<f64>,
        b: Vec<f64>,
        g_rows: Vec<i64>,
        g_cols: Vec<i64>,
        g_vals: Vec<f64>,
        h: Vec<f64>,
        lb: Vec<f64>,
        ub: Vec<f64>,
    ) -> PyResult<Self> {
        if c.len() != n {
            return Err(PyValueError::new_err(format!(
                "c has length {}, expected n = {n}",
                c.len()
            )));
        }
        if !lb.is_empty() && lb.len() != n {
            return Err(PyValueError::new_err(format!(
                "lb has length {}, expected 0 or n = {n}",
                lb.len()
            )));
        }
        if !ub.is_empty() && ub.len() != n {
            return Err(PyValueError::new_err(format!(
                "ub has length {}, expected 0 or n = {n}",
                ub.len()
            )));
        }
        let inner = QpProblem {
            n,
            p_lower: triplets(&p_rows, &p_cols, &p_vals, "P", true)?,
            c,
            a: triplets(&a_rows, &a_cols, &a_vals, "A", false)?,
            b,
            g: triplets(&g_rows, &g_cols, &g_vals, "G", false)?,
            h,
            lb,
            ub,
        };
        Ok(Self { inner })
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    #[getter]
    fn m_eq(&self) -> usize {
        self.inner.m_eq()
    }

    #[getter]
    fn m_ineq(&self) -> usize {
        self.inner.m_ineq()
    }
}

/// Turn a `QpStatus` into the lowercase string used in the result dict.
fn status_str(s: QpStatus) -> &'static str {
    match s {
        QpStatus::Optimal => "optimal",
        QpStatus::PrimalInfeasible => "primal_infeasible",
        QpStatus::DualInfeasible => "dual_infeasible",
        QpStatus::IterationLimit => "iteration_limit",
        QpStatus::NumericalFailure => "numerical_failure",
    }
}

/// Build the Python result dict `{x, y, z, z_lb, z_ub, obj, iters,
/// status}` from a `QpSolution`.
fn solution_dict<'py>(py: Python<'py>, sol: QpSolution) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("status", status_str(sol.status))?;
    d.set_item("obj", sol.obj)?;
    d.set_item("iters", sol.iters)?;
    d.set_item("x", sol.x.into_pyarray_bound(py))?;
    d.set_item("y", sol.y.into_pyarray_bound(py))?;
    d.set_item("z", sol.z.into_pyarray_bound(py))?;
    d.set_item("z_lb", sol.z_lb.into_pyarray_bound(py))?;
    d.set_item("z_ub", sol.z_ub.into_pyarray_bound(py))?;
    Ok(d)
}

/// Extract a `QpWarmStart` from a Python mapping (typically a previous
/// result dict). Missing vector keys default to empty, so a partial warm
/// start (e.g. only `x`) is accepted; the solver validates dimensions and
/// falls back to a cold start if they don't match.
fn warm_from_dict(warm: &Bound<'_, PyDict>) -> PyResult<QpWarmStart> {
    let get = |key: &str| -> PyResult<Vec<f64>> {
        match warm.get_item(key)? {
            Some(v) => v.extract::<Vec<f64>>(),
            None => Ok(Vec::new()),
        }
    };
    Ok(QpWarmStart {
        x: get("x")?,
        y: get("y")?,
        z: get("z")?,
        z_lb: get("z_lb")?,
        z_ub: get("z_ub")?,
    })
}

fn opts(tol: Option<f64>, max_iter: Option<usize>) -> QpOptions {
    let mut o = QpOptions::default();
    if let Some(t) = tol {
        o.tol = t;
    }
    if let Some(m) = max_iter {
        o.max_iter = m;
    }
    o
}

/// Solve one convex QP. Returns a dict with the primal `x`, duals `y`
/// (equalities), `z` (inequalities), bound duals `z_lb`/`z_ub`, the
/// objective, iteration count, and a status string.
///
/// `warm_start` (optional) is a mapping with `x`/`y`/`z`/`z_lb`/`z_ub`
/// keys — e.g. a previous result dict for a nearby problem. It only
/// affects the iteration count, not the solution; a dimension mismatch is
/// ignored (cold start).
#[pyfunction]
#[pyo3(signature = (prob, tol=None, max_iter=None, warm_start=None))]
pub fn solve_qp<'py>(
    py: Python<'py>,
    prob: &PyQpProblem,
    tol: Option<f64>,
    max_iter: Option<usize>,
    warm_start: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let o = opts(tol, max_iter);
    let warm = warm_start.map(warm_from_dict).transpose()?;
    let sol = py.allow_threads(|| match &warm {
        Some(w) => solve_qp_ipm_warm(&prob.inner, &o, w, backend),
        None => solve_qp_ipm(&prob.inner, &o, backend),
    });
    solution_dict(py, sol)
}

/// Solve a batch of convex QPs in parallel (across instances). Returns a
/// list of result dicts in input order. Releases the GIL for the solve.
///
/// `warm_starts` (optional) is a list of warm-start mappings (one per
/// problem, same length as `probs`) — e.g. the previous batch's result
/// dicts for a sequence of nearby batches. Each only affects its
/// instance's iteration count; a per-instance mismatch is ignored.
#[pyfunction]
#[pyo3(signature = (probs, tol=None, max_iter=None, warm_starts=None))]
pub fn solve_qp_batch<'py>(
    py: Python<'py>,
    probs: Vec<PyQpProblem>,
    tol: Option<f64>,
    max_iter: Option<usize>,
    warm_starts: Option<Vec<Bound<'py, PyDict>>>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let o = opts(tol, max_iter);
    let inners: Vec<QpProblem> = probs.into_iter().map(|p| p.inner).collect();
    let warms: Option<Vec<QpWarmStart>> = match warm_starts {
        Some(ws) => {
            if ws.len() != inners.len() {
                return Err(PyValueError::new_err(format!(
                    "warm_starts has length {}, expected {} (one per problem)",
                    ws.len(),
                    inners.len()
                )));
            }
            Some(ws.iter().map(warm_from_dict).collect::<PyResult<_>>()?)
        }
        None => None,
    };
    let sols = py.allow_threads(|| match &warms {
        Some(w) => solve_qp_batch_parallel_warm(&inners, w, &o, backend),
        None => solve_qp_batch_parallel(&inners, &o, backend),
    });
    sols.into_iter().map(|s| solution_dict(py, s)).collect()
}

/// Solve one QP structure (`base`) against many linear objectives `cs`
/// (a sequence of length-`n` vectors), in parallel. Returns a list of
/// result dicts in order.
#[pyfunction]
#[pyo3(signature = (base, cs, tol=None, max_iter=None))]
pub fn solve_qp_multi_rhs<'py>(
    py: Python<'py>,
    base: &PyQpProblem,
    cs: Vec<Vec<f64>>,
    tol: Option<f64>,
    max_iter: Option<usize>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    for (k, c) in cs.iter().enumerate() {
        if c.len() != base.inner.n {
            return Err(PyValueError::new_err(format!(
                "cs[{k}] has length {}, expected n = {}",
                c.len(),
                base.inner.n
            )));
        }
    }
    let o = opts(tol, max_iter);
    let base_inner = base.inner.clone();
    let sols = py.allow_threads(|| {
        pounce_convex::solve_qp_multi_rhs_parallel(&base_inner, &cs, &o, backend)
    });
    sols.into_iter().map(|s| solution_dict(py, s)).collect()
}

/// Build-once / solve-many handle: builds the KKT symbolic factor once
/// for a fixed problem *structure* (same sparsity and set of finite
/// bounds), then reuses it across `solve()` calls that vary only the
/// numeric data. Mirrors `pounce.jax.JaxProblem`'s build-once ergonomics
/// for the convex QP solver.
#[pyclass(name = "QpFactorization", module = "pounce._pounce", unsendable)]
pub struct PyQpFactorization {
    inner: QpFactorization,
}

#[pymethods]
impl PyQpFactorization {
    #[new]
    #[pyo3(signature = (base, tol=None, max_iter=None))]
    fn new(base: &PyQpProblem, tol: Option<f64>, max_iter: Option<usize>) -> PyResult<Self> {
        let o = opts(tol, max_iter);
        let inner = QpFactorization::build(&base.inner, &o, backend).ok_or_else(|| {
            PyValueError::new_err(
                "QpFactorization: initial factorization failed (structurally singular KKT system)",
            )
        })?;
        Ok(Self { inner })
    }

    /// Solve `prob`, reusing the captured symbolic factor. `prob` must
    /// share the captured structure; otherwise the result dict has
    /// status `"numerical_failure"`.
    ///
    /// `warm_start` (optional) seeds the iteration from a nearby problem's
    /// solution, combining symbolic-factor reuse with warm starting.
    #[pyo3(signature = (prob, warm_start=None))]
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        prob: &PyQpProblem,
        warm_start: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let sol = match warm_start {
            Some(w) => self.inner.solve_warm(&prob.inner, &warm_from_dict(w)?),
            None => self.inner.solve(&prob.inner),
        };
        solution_dict(py, sol)
    }
}
