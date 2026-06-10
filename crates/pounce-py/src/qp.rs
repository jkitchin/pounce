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
    solve_socp_ipm, ConeSpec, QpFactorization, QpOptions, QpProblem, QpSensitivity, QpSolution,
    QpStatus, QpWarmStart, SensError, Triplet,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Transparent `Send` shim used to move a non-`Send` value (a factorization /
/// sensitivity holding `dyn SparseSymLinearSolverInterface` trait objects)
/// across a `py.allow_threads` boundary. SAFETY: the closure runs on the
/// calling thread after `PyEval_SaveThread` (it never actually crosses OS
/// threads), so the wrapped value is only ever touched by this one thread.
/// Method-call accessors (vs. field `.0`) defeat the 2021-edition
/// disjoint-capture rule so the closure captures the whole guard.
struct SendGuard<T>(T);
unsafe impl<T> Send for SendGuard<T> {}
impl<T> SendGuard<T> {
    fn new(v: T) -> Self {
        Self(v)
    }
    fn into_inner(self) -> T {
        self.0
    }
}

/// Inner-serial backend for the rayon-parallel batch / multi-RHS paths:
/// each worker builds its own serial factor so the only parallelism is
/// across instances (outer-parallel / inner-serial). No global state.
fn serial_backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::serial())
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
        QpStatus::OptimalInaccurate => "optimal_inaccurate",
        QpStatus::PrimalInfeasible => "primal_infeasible",
        QpStatus::DualInfeasible => "dual_infeasible",
        QpStatus::IterationLimit => "iteration_limit",
        QpStatus::NumericalFailure => "numerical_failure",
    }
}

/// Build the Python result dict `{x, y, z, z_lb, z_ub, obj, iters, status,
/// iterates, residuals}` from a `QpSolution`.
///
/// When `prob` is `Some`, the final KKT `residuals` block is attached — but
/// only for the plain-QP path, where `Gx ≤ h` is an orthant constraint and
/// [`QpSolution::kkt_residuals`] applies. Conic (SOCP/exp/power) solves pass
/// `None`: there the slack lives in a non-orthant cone, so those orthant
/// residuals would be meaningless. The `iterates` trace is always attached
/// (empty unless `collect_iterates` was set, so there is no overhead off the
/// opt-in path).
fn solution_dict<'py>(
    py: Python<'py>,
    sol: QpSolution,
    prob: Option<&QpProblem>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("status", status_str(sol.status))?;
    d.set_item("obj", sol.obj)?;
    d.set_item("iters", sol.iters)?;

    // Final KKT residuals (plain QP only — see the doc comment).
    if let Some(p) = prob {
        let r = sol.kkt_residuals(p);
        let rd = PyDict::new_bound(py);
        rd.set_item("primal_infeasibility", r.primal_infeasibility)?;
        rd.set_item("dual_infeasibility", r.dual_infeasibility)?;
        rd.set_item("complementarity", r.complementarity)?;
        rd.set_item("kkt_error", r.kkt_error())?;
        d.set_item("residuals", rd)?;
    }

    // Per-iteration convergence trace (empty unless `collect_iterates` set).
    let trace = PyList::empty_bound(py);
    for it in &sol.iterates {
        let row = PyDict::new_bound(py);
        row.set_item("iter", it.iter)?;
        row.set_item("objective", it.objective)?;
        row.set_item("primal_infeasibility", it.primal_infeasibility)?;
        row.set_item("dual_infeasibility", it.dual_infeasibility)?;
        row.set_item("mu", it.mu)?;
        row.set_item("alpha_primal", it.alpha_primal)?;
        row.set_item("alpha_dual", it.alpha_dual)?;
        trace.append(row)?;
    }
    d.set_item("iterates", trace)?;

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

/// Parse `(kind, value)` tuples into [`ConeSpec`]s. `kind` is
/// case-insensitive. The float `value` means the **dimension** for
/// `"nonneg"`/`"nn"`/`"+"` and `"soc"`/`"q"` (rounded to an integer), the
/// **exponent α** for `"pow"`/`"power"` (the 3-D power cone, `α ∈ (0,1)`),
/// and the **matrix size n** for `"psd"`/`"sdp"` (which spans `n(n+1)/2`
/// svec rows). `"exp"`/`"exponential"` is the fixed-dimension-3 exponential
/// cone (its `value` is ignored).
fn parse_cones(specs: Vec<(String, f64)>) -> PyResult<Vec<ConeSpec>> {
    specs
        .into_iter()
        .map(|(kind, v)| match kind.to_ascii_lowercase().as_str() {
            "nonneg" | "nn" | "+" => Ok(ConeSpec::Nonneg(v.round() as usize)),
            "soc" | "q" | "secondorder" => Ok(ConeSpec::SecondOrder(v.round() as usize)),
            "exp" | "exponential" | "e" => Ok(ConeSpec::Exponential),
            "pow" | "power" | "p" if v > 0.0 && v < 1.0 => Ok(ConeSpec::Power(v)),
            "pow" | "power" | "p" => Err(PyValueError::new_err(format!(
                "power-cone exponent α must be in (0, 1), got {v}"
            ))),
            "psd" | "sdp" | "s" => Ok(ConeSpec::Psd(v.round() as usize)),
            other => Err(PyValueError::new_err(format!(
                "unknown cone kind '{other}' (use 'nonneg', 'soc', 'exp', 'pow', or 'psd')"
            ))),
        })
        .collect()
}

fn opts(tol: Option<f64>, max_iter: Option<usize>, collect_iterates: bool) -> QpOptions {
    let mut o = QpOptions::default();
    if let Some(t) = tol {
        o.tol = t;
    }
    if let Some(m) = max_iter {
        o.max_iter = m;
    }
    o.collect_iterates = collect_iterates;
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
///
/// `collect_iterates` (default `false`) opts into the per-iteration
/// convergence trace, returned under the `iterates` key.
#[pyfunction]
#[pyo3(signature = (prob, tol=None, max_iter=None, warm_start=None, collect_iterates=false))]
pub fn solve_qp<'py>(
    py: Python<'py>,
    prob: &PyQpProblem,
    tol: Option<f64>,
    max_iter: Option<usize>,
    warm_start: Option<&Bound<'py, PyDict>>,
    collect_iterates: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let o = opts(tol, max_iter, collect_iterates);
    let warm = warm_start.map(warm_from_dict).transpose()?;
    let sol = py.allow_threads(|| match &warm {
        Some(w) => solve_qp_ipm_warm(&prob.inner, &o, w, backend),
        None => solve_qp_ipm(&prob.inner, &o, backend),
    });
    solution_dict(py, sol, Some(&prob.inner))
}

/// Solve a standard-form conic program (LP/QP plus second-order, exponential,
/// and/or **power** cones). The inequality block `Gx ≤ h` is partitioned by
/// `cones`, a list of `(kind, value)` tuples covering the `m_ineq` rows in
/// order; each `s = h − Gx` block must lie in its cone. `value` is the
/// dimension for `"nonneg"`/`"soc"` and the exponent α for `"pow"`; `"exp"`
/// is the fixed 3-D exponential cone. Variable bounds are appended as a
/// trailing nonnegative block. Returns the usual result dict.
///
/// Problems containing an exponential or power cone route to the
/// non-symmetric HSDE driver, which also handles second-order cones — so a
/// SOC may be freely mixed with an exp/power cone.
#[pyfunction]
#[pyo3(signature = (prob, cones, tol=None, max_iter=None, collect_iterates=false))]
pub fn solve_socp<'py>(
    py: Python<'py>,
    prob: &PyQpProblem,
    cones: Vec<(String, f64)>,
    tol: Option<f64>,
    max_iter: Option<usize>,
    collect_iterates: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let o = opts(tol, max_iter, collect_iterates);
    let specs = parse_cones(cones)?;
    // PSD (self-scaled, symmetric driver) cannot be mixed with the
    // exponential/power cones (non-symmetric driver) in one problem.
    let has_nonsym = specs
        .iter()
        .any(|c| matches!(c, ConeSpec::Exponential | ConeSpec::Power(_)));
    let has_psd = specs.iter().any(|c| matches!(c, ConeSpec::Psd(_)));
    if has_nonsym && has_psd {
        return Err(PyValueError::new_err(
            "the PSD cone cannot be combined with exponential/power cones in \
             one problem (they use different drivers)",
        ));
    }
    // The cones must partition the rows of G exactly (an exp/power cone is
    // always 3 rows; a PSD(n) cone is n(n+1)/2 svec rows). Catch the mismatch
    // here with a clear, catchable error rather than letting the conic driver
    // index past the slack vector.
    let cone_rows: usize = specs.iter().map(|c| c.dim()).sum();
    if cone_rows != prob.inner.m_ineq() {
        return Err(PyValueError::new_err(format!(
            "cone dimensions sum to {cone_rows}, but G has {} inequality row(s); \
             the cones must partition the rows of G exactly \
             (an exponential or power cone is always 3 rows)",
            prob.inner.m_ineq()
        )));
    }
    let sol = py.allow_threads(|| solve_socp_ipm(&prob.inner, &specs, &o, backend));
    // Conic slack lives in a non-orthant cone: skip the orthant residuals.
    solution_dict(py, sol, None)
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
    let o = opts(tol, max_iter, false);
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
        Some(w) => solve_qp_batch_parallel_warm(&inners, w, &o, serial_backend),
        None => solve_qp_batch_parallel(&inners, &o, serial_backend),
    });
    sols.into_iter()
        .zip(inners.iter())
        .map(|(s, p)| solution_dict(py, s, Some(p)))
        .collect()
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
    let o = opts(tol, max_iter, false);
    let base_inner = base.inner.clone();
    let sols = py.allow_threads(|| {
        pounce_convex::solve_qp_multi_rhs_parallel(&base_inner, &cs, &o, serial_backend)
    });
    // Each solve shares the base structure but uses its own objective `cs[k]`;
    // attach residuals against that instance (a clone with `c` swapped in).
    sols.into_iter()
        .zip(cs.iter())
        .map(|(s, c)| {
            let mut prob = base_inner.clone();
            prob.c = c.clone();
            solution_dict(py, s, Some(&prob))
        })
        .collect()
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
        let o = opts(tol, max_iter, false);
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
        // Parse the (Python) warm-start dict *before* dropping the GIL, then
        // run the pure-Rust solve with the GIL released so other threads make
        // progress — the QP path never calls back into Python (mirrors the
        // one-shot `solve_qp` above).
        let warm = match warm_start {
            Some(w) => Some(warm_from_dict(w)?),
            None => None,
        };
        let qp = &prob.inner;
        // `self.inner` holds non-`Send` linear-solver trait objects, so wrap the
        // exclusive borrow in `SendGuard` to cross the GIL-release boundary.
        let guard = SendGuard::new(&mut self.inner);
        let sol = py.allow_threads(move || {
            let inner = guard.into_inner();
            match &warm {
                Some(w) => inner.solve_warm(qp, w),
                None => inner.solve(qp),
            }
        });
        solution_dict(py, sol, Some(&prob.inner))
    }
}

/// Post-optimal sensitivity for a convex QP — the sIPOPT analog. Solves the
/// problem on construction, then holds the active-set KKT factorization so
/// each `parametric_step` is a single back-substitution. Mirrors the NLP
/// `Solver` session (which caches the converged factor for
/// `parametric_step` / `reduced_hessian`), specialized to a QP.
#[pyclass(name = "QpSensitivity", module = "pounce._pounce", unsendable)]
pub struct PyQpSensitivity {
    inner: QpSensitivity,
    x: Vec<f64>,
    obj: f64,
    m_eq: usize,
}

#[pymethods]
impl PyQpSensitivity {
    /// Solve `prob` and build its sensitivity. `active_tol` (default `1e-7`)
    /// is the multiplier threshold used to read the active set. Raises
    /// `ValueError` if the QP does not solve to optimality, or if the
    /// active-set KKT is singular (the parametric step is not unique).
    #[new]
    #[pyo3(signature = (prob, tol=None, max_iter=None, active_tol=1e-7))]
    fn new(
        py: Python<'_>,
        prob: &PyQpProblem,
        tol: Option<f64>,
        max_iter: Option<usize>,
        active_tol: f64,
    ) -> PyResult<Self> {
        let o = opts(tol, max_iter, false);
        // Both the IPM solve and the sensitivity factorization are pure Rust
        // (no Python callbacks), so run them with the GIL released — other
        // threads make progress during the solve (mirrors `solve_qp`).
        let qp = &prob.inner;
        // The built `QpSensitivity` holds non-`Send` linear-solver trait
        // objects, so wrap the closure's result in `SendGuard` to return it
        // across the GIL-release boundary. The factorization runs only when the
        // solve is optimal; otherwise the closure returns just the status and
        // the caller raises below (no panic-on-`None` unwrap).
        type Payload = (Vec<f64>, f64, Result<QpSensitivity, SensError>);
        let (status, payload): (QpStatus, Option<Payload>) = py
            .allow_threads(|| {
                let sol = solve_qp_ipm(qp, &o, backend);
                let payload = (sol.status == QpStatus::Optimal).then(|| {
                    (
                        sol.x.clone(),
                        sol.obj,
                        QpSensitivity::build(qp, &sol, &o, active_tol, backend),
                    )
                });
                SendGuard::new((sol.status, payload))
            })
            .into_inner();
        let (x, obj, build_res) = match payload {
            Some(p) => p,
            None => {
                return Err(PyValueError::new_err(format!(
                    "QpSensitivity: the QP did not solve to optimality (status {}); \
                     sensitivity is only defined at an optimum",
                    status_str(status)
                )));
            }
        };
        let inner = build_res.map_err(|e| match e {
            SensError::NotOptimal => {
                PyValueError::new_err("QpSensitivity: solution is not optimal")
            }
            SensError::FactorizationFailed => PyValueError::new_err(
                "QpSensitivity: the active-set KKT is singular (the active constraint \
                 gradients are rank-deficient), so the parametric step is not unique",
            ),
            SensError::EigenFailed => {
                PyValueError::new_err("QpSensitivity: a symmetric eigensolve did not converge")
            }
        })?;
        Ok(Self {
            inner,
            x,
            obj,
            m_eq: prob.inner.m_eq(),
        })
    }

    /// First-order primal step `dx ≈ x*(b + Δb) − x*(b)` for a perturbation
    /// of the equality right-hand side `b`: constraint
    /// `pin_constraint_indices[k]` is perturbed by `deltas[k]`. Returns the
    /// length-`n` sensitivity, so `sensitivity.x + dx` predicts the
    /// perturbed solution (exact to first order while the active set holds).
    fn parametric_step<'py>(
        &mut self,
        py: Python<'py>,
        pin_constraint_indices: Vec<usize>,
        deltas: Vec<f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        if pin_constraint_indices.len() != deltas.len() {
            return Err(PyValueError::new_err(format!(
                "pin_constraint_indices has length {} but deltas has length {}",
                pin_constraint_indices.len(),
                deltas.len()
            )));
        }
        for &i in &pin_constraint_indices {
            if i >= self.m_eq {
                return Err(PyValueError::new_err(format!(
                    "pin constraint index {i} out of range (the QP has {} equality \
                     constraint(s); only equality-constraint RHS values are parameters)",
                    self.m_eq
                )));
            }
        }
        let dx = self.inner.parametric_step(&pin_constraint_indices, &deltas);
        Ok(dx.into_pyarray_bound(py))
    }

    /// Reduced Hessian of the QP on its active manifold (`Zᵀ P Z`) with its
    /// eigendecomposition. Returns a dict with `n_dof` (degrees of freedom),
    /// `matrix` and `eigenvectors` (flat, column-major `n_dof × n_dof`), and
    /// `eigenvalues` (ascending). `rank_tol` (default `1e-9`) is the relative
    /// threshold for the rank of the active Jacobian.
    #[pyo3(signature = (rank_tol = 1e-9))]
    fn reduced_hessian<'py>(&self, py: Python<'py>, rank_tol: f64) -> PyResult<Bound<'py, PyDict>> {
        let rh = self.inner.reduced_hessian(rank_tol).map_err(|e| match e {
            SensError::EigenFailed => PyValueError::new_err(
                "QpSensitivity.reduced_hessian: a symmetric eigensolve did not converge, \
                 so the reduced Hessian's rank / null-space cannot be trusted",
            ),
            SensError::NotOptimal => {
                PyValueError::new_err("QpSensitivity: solution is not optimal")
            }
            SensError::FactorizationFailed => {
                PyValueError::new_err("QpSensitivity: the active-set KKT is singular")
            }
        })?;
        let d = PyDict::new_bound(py);
        d.set_item("n_dof", rh.n_dof)?;
        d.set_item("matrix", rh.matrix.into_pyarray_bound(py))?;
        d.set_item("eigenvalues", rh.eigenvalues.into_pyarray_bound(py))?;
        d.set_item("eigenvectors", rh.eigenvectors.into_pyarray_bound(py))?;
        Ok(d)
    }

    /// The optimal primal solution `x*`.
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        self.x.clone().into_pyarray_bound(py)
    }

    /// The optimal objective value.
    #[getter]
    fn obj(&self) -> f64 {
        self.obj
    }

    /// The active-set KKT dimension `n + m_eq + n_active`.
    #[getter]
    fn kkt_dim(&self) -> usize {
        self.inner.kkt_dim()
    }
}
