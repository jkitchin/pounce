//! Python → Rust TNLP bridge.
//!
//! [`PyTnlp`] implements [`pounce_nlp::TNLP`] by calling Python
//! callables stored as [`Py<PyAny>`]. The Python user provides:
//!
//! ```python
//! class MyProblem:
//!     def objective(self, x):           -> float
//!     def gradient(self, x):            -> array(n)
//!     def constraints(self, x):         -> array(m)             # if m > 0
//!     def jacobian(self, x):            -> array(nnz_jac)       # if m > 0
//!     def jacobianstructure(self):      -> (rows, cols)         # called once
//!     def hessian(self, x, lam, obj_f): -> array(nnz_hess)      # if exact-hess
//!     def hessianstructure(self):       -> (rows, cols)         # called once
//!     def intermediate(self, *stats):   -> bool | None          # optional
//! ```
//!
//! Bounds (`lb`, `ub`, `cl`, `cu`) and starting point come through
//! [`PyTnlpInit`], populated by the `Problem.solve` wrapper.
//!
//! All NumPy ↔ Rust transfers use zero-copy views where the alignment
//! and dtype permit, with a copy fallback. Buffers handed back to the
//! solver (gradient, jacobian values, hessian values) are written into
//! the `&mut [f64]` slice the trait provides, never reallocated.

use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, NlpInfo, ScalingRequest, Solution,
    SparsityRequest, StartingPoint, TNLP,
};

use crate::problem::UserScaling;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// One-shot initialization payload assembled by `Problem.solve`.
pub(crate) struct PyTnlpInit {
    pub n: Index,
    pub m: Index,
    pub nele_jac: Index,
    pub nele_hess: Index,
    pub x_l: Vec<Number>,
    pub x_u: Vec<Number>,
    pub g_l: Vec<Number>,
    pub g_u: Vec<Number>,
    pub x0: Vec<Number>,
    /// Optional warm-start duals. When present and the solver option
    /// `warm_start_init_point` is `"yes"`, they are reported back to
    /// the IPM via [`TNLP::get_starting_point`].
    pub lam0: Option<Vec<Number>>,
    pub z_l0: Option<Vec<Number>>,
    pub z_u0: Option<Vec<Number>>,
    /// The Python `problem_obj` (with `objective`, `gradient`, ...).
    pub py_obj: Py<PyAny>,
    /// Pre-resolved sparsity patterns. We fetch these once at solve
    /// time so the `Structure` call below is a copy, not a re-entrant
    /// Python dispatch (which would need the GIL in a context the
    /// solver doesn't grant).
    pub jac_rows: Vec<Index>,
    pub jac_cols: Vec<Index>,
    pub hess_rows: Vec<Index>,
    pub hess_cols: Vec<Index>,
    /// `true` when the user provided an `hessian` method; otherwise the
    /// solver falls back to L-BFGS.
    pub has_hessian: bool,
    /// User-supplied NLP scaling installed via
    /// `Problem.set_problem_scaling`. Consulted by
    /// [`TNLP::get_scaling_parameters`] only; the IPM picks it up when
    /// `nlp_scaling_method=user-scaling` is set.
    pub user_scaling: Option<UserScaling>,
    /// Final-iterate capture targets. Populated by
    /// [`TNLP::finalize_solution`].
    pub final_x: Vec<Number>,
    pub final_z_l: Vec<Number>,
    pub final_z_u: Vec<Number>,
    pub final_g: Vec<Number>,
    pub final_lambda: Vec<Number>,
    pub final_obj: Number,
    pub final_status_code: i32,
}

/// Trait-impl side of the bridge.
pub(crate) struct PyTnlp {
    pub(crate) state: PyTnlpInit,
}

impl PyTnlp {
    pub(crate) fn new(state: PyTnlpInit) -> Self {
        Self { state }
    }
}

/// Acquire the GIL and call `method` on `obj` with positional args
/// `(x,)`. Returns the result as a `Py<PyAny>` (caller decodes).
fn call1(obj: &Py<PyAny>, method: &str, x: &[Number]) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| {
        let arr = PyArray1::<Number>::from_slice_bound(py, x);
        let bound = obj.bind(py);
        let res = bound.call_method1(method, (arr,))?;
        Ok(res.unbind())
    })
}

/// Copy a 1-D float NumPy / sequence return value into `out`. Errors
/// when the length does not match.
fn copy_pyarray_into(val: &Py<PyAny>, out: &mut [Number], what: &str) -> PyResult<()> {
    Python::with_gil(|py| {
        let bound = val.bind(py);
        // Fast path: f64 NumPy array.
        if let Ok(arr) = bound.downcast::<PyArray1<Number>>() {
            let len = arr.len();
            if len != out.len() {
                return Err(PyValueError::new_err(format!(
                    "{what}: expected length {}, got {}",
                    out.len(),
                    len
                )));
            }
            // `as_slice()` requires C-contiguity; a valid non-contiguous
            // float64 array (e.g. a strided view returned by a user callback)
            // still downcasts here, so copy it via a strided ndarray view
            // rather than erroring (L49).
            match unsafe { arr.as_slice() } {
                Ok(view) => out.copy_from_slice(view),
                Err(_) => {
                    for (dst, src) in out.iter_mut().zip(arr.readonly().as_array().iter()) {
                        *dst = *src;
                    }
                }
            }
            return Ok(());
        }
        // Generic fallback: iterate the sequence, expect floats.
        let iter = bound.iter()?;
        let mut i = 0usize;
        for item in iter {
            let v: Number = item?.extract()?;
            if i >= out.len() {
                return Err(PyValueError::new_err(format!(
                    "{what}: too many entries (expected {})",
                    out.len()
                )));
            }
            out[i] = v;
            i += 1;
        }
        if i != out.len() {
            return Err(PyValueError::new_err(format!(
                "{what}: got {} entries, expected {}",
                i,
                out.len()
            )));
        }
        Ok(())
    })
}

impl TNLP for PyTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.state.n,
            m: self.state.m,
            nnz_jac_g: self.state.nele_jac,
            nnz_h_lag: self.state.nele_hess,
            // We expose 0-based indices to the Python user; the
            // adapter translates if a backend wants 1-based.
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.state.x_l);
        b.x_u.copy_from_slice(&self.state.x_u);
        b.g_l.copy_from_slice(&self.state.g_l);
        b.g_u.copy_from_slice(&self.state.g_u);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if sp.init_x {
            sp.x.copy_from_slice(&self.state.x0);
        }
        if sp.init_lambda {
            if let Some(l) = &self.state.lam0 {
                sp.lambda.copy_from_slice(l);
            }
        }
        if sp.init_z {
            if let Some(z) = &self.state.z_l0 {
                sp.z_l.copy_from_slice(z);
            }
            if let Some(z) = &self.state.z_u0 {
                sp.z_u.copy_from_slice(z);
            }
        }
        true
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        let Some(s) = self.state.user_scaling.as_ref() else {
            return false;
        };
        *req.obj_scaling = s.obj;
        if let Some(x) = s.x_scaling.as_ref() {
            if x.len() == req.x_scaling.len() {
                req.x_scaling.copy_from_slice(x);
                *req.use_x_scaling = true;
            } else {
                *req.use_x_scaling = false;
            }
        } else {
            *req.use_x_scaling = false;
        }
        if let Some(g) = s.g_scaling.as_ref() {
            if g.len() == req.g_scaling.len() {
                req.g_scaling.copy_from_slice(g);
                *req.use_g_scaling = true;
            } else {
                *req.use_g_scaling = false;
            }
        } else {
            *req.use_g_scaling = false;
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        match call1(&self.state.py_obj, "objective", x) {
            Ok(v) => Python::with_gil(|py| v.bind(py).extract::<Number>().ok()),
            Err(e) => {
                tracing::error!(target: "pounce::py", "pounce-py: objective(): {e}");
                None
            }
        }
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
        let res = match call1(&self.state.py_obj, "gradient", x) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(target: "pounce::py", "pounce-py: gradient(): {e}");
                return false;
            }
        };
        copy_pyarray_into(&res, grad_f, "gradient").is_ok()
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        if g.is_empty() {
            return true;
        }
        let res = match call1(&self.state.py_obj, "constraints", x) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(target: "pounce::py", "pounce-py: constraints(): {e}");
                return false;
            }
        };
        copy_pyarray_into(&res, g, "constraints").is_ok()
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                if irow.len() != self.state.jac_rows.len() {
                    return false;
                }
                irow.copy_from_slice(&self.state.jac_rows);
                jcol.copy_from_slice(&self.state.jac_cols);
                true
            }
            // No constraints (m == 0) → no Jacobian entries to fill. Skip the
            // Python call entirely: the user object may legitimately omit
            // `jacobian` in the unconstrained case, so calling it would raise
            // a spurious AttributeError on every iteration.
            SparsityRequest::Values { values } if values.is_empty() => true,
            SparsityRequest::Values { values } => {
                let xx = x.unwrap_or(&[]);
                let res = match call1(&self.state.py_obj, "jacobian", xx) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::error!(target: "pounce::py", "pounce-py: jacobian(): {e}");
                        return false;
                    }
                };
                copy_pyarray_into(&res, values, "jacobian").is_ok()
            }
        }
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                if irow.len() != self.state.hess_rows.len() {
                    return false;
                }
                irow.copy_from_slice(&self.state.hess_rows);
                jcol.copy_from_slice(&self.state.hess_cols);
                true
            }
            // No user Hessian → we declared a dense lower-triangle
            // sparsity in `Problem.solve` so the L-BFGS updater has a
            // non-empty work space to project onto. Hand back zeros; the
            // quasi-Newton updater overwrites them each iteration.
            SparsityRequest::Values { values } if !self.state.has_hessian => {
                for v in values.iter_mut() {
                    *v = 0.0;
                }
                true
            }
            SparsityRequest::Values { values } => {
                let xx = x.unwrap_or(&[]);
                let lam = lambda.unwrap_or(&[]);
                let res = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
                    let x_arr = PyArray1::<Number>::from_slice_bound(py, xx);
                    let lam_arr = PyArray1::<Number>::from_slice_bound(py, lam);
                    let bound = self.state.py_obj.bind(py);
                    let res = bound.call_method1("hessian", (x_arr, lam_arr, obj_factor))?;
                    Ok(res.unbind())
                });
                let res = match res {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::error!(target: "pounce::py", "pounce-py: hessian(): {e}");
                        return false;
                    }
                };
                copy_pyarray_into(&res, values, "hessian").is_ok()
            }
        }
    }

    fn intermediate_callback(
        &mut self,
        stats: IterStats,
        _ip_data: &IpoptData,
        _ip_cq: &IpoptCq,
    ) -> bool {
        // Optional. If the user object has no `intermediate` method we
        // just keep going; any exception aborts the iteration with a
        // user-stop status (consistent with cyipopt).
        let r: PyResult<Option<bool>> = Python::with_gil(|py| {
            let bound = self.state.py_obj.bind(py);
            if !bound.hasattr("intermediate")? {
                return Ok(None);
            }
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("alg_mod", stats.mode as i32)?;
            kwargs.set_item("iter_count", stats.iter)?;
            kwargs.set_item("obj_value", stats.obj_value)?;
            kwargs.set_item("inf_pr", stats.inf_pr)?;
            kwargs.set_item("inf_du", stats.inf_du)?;
            kwargs.set_item("mu", stats.mu)?;
            kwargs.set_item("d_norm", stats.d_norm)?;
            kwargs.set_item("regularization_size", stats.regularization_size)?;
            kwargs.set_item("alpha_du", stats.alpha_du)?;
            kwargs.set_item("alpha_pr", stats.alpha_pr)?;
            kwargs.set_item("ls_trials", stats.ls_trials)?;
            let res = bound.call_method("intermediate", PyTuple::empty_bound(py), Some(&kwargs))?;
            if res.is_none() {
                return Ok(Some(true));
            }
            // cyipopt truthiness: any falsy return (`False`, `0`, `0.0`, an
            // empty container) requests a stop; truthy continues. A strict
            // `extract::<bool>()` rejects a valid falsy int `0` and, via
            // `unwrap_or(true)`, silently *continued* — ignoring the user's
            // stop. Use Python truthiness so `0` stops like cyipopt.
            Ok(Some(res.is_truthy()?))
        });
        match r {
            Ok(Some(v)) => v,
            Ok(None) => true,
            // A raising `intermediate` aborts the solve with a user-stop
            // status (consistent with cyipopt). Log it like the eval
            // callbacks (`objective`/`gradient`/…) so a crashing callback
            // leaves a trace instead of masquerading as a silent
            // `User_Requested_Stop`.
            Err(e) => {
                tracing::error!(target: "pounce::py", "pounce-py: intermediate(): {e}");
                false
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.state.final_x.clear();
        self.state.final_x.extend_from_slice(sol.x);
        self.state.final_z_l.clear();
        self.state.final_z_l.extend_from_slice(sol.z_l);
        self.state.final_z_u.clear();
        self.state.final_z_u.extend_from_slice(sol.z_u);
        self.state.final_g.clear();
        self.state.final_g.extend_from_slice(sol.g);
        self.state.final_lambda.clear();
        self.state.final_lambda.extend_from_slice(sol.lambda);
        self.state.final_obj = sol.obj_value;
        self.state.final_status_code = sol.status as i32;
    }
}

/// Bind a Python object's `<attr>` and call it with no args. Returns
/// the raw `Py<PyAny>` so the caller can convert.
pub(crate) fn call0(obj: &Py<PyAny>, method: &str) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| {
        let bound = obj.bind(py);
        let res = bound.call_method0(method)?;
        Ok(res.unbind())
    })
}

/// Decode a `(rows, cols)` tuple-of-int-sequences into two `Vec<Index>`.
/// Used by `Problem.solve` to materialize `jacobianstructure()` and
/// `hessianstructure()` once before handing the bridge to the IPM.
pub(crate) fn decode_structure(val: &Py<PyAny>, nnz: usize) -> PyResult<(Vec<Index>, Vec<Index>)> {
    Python::with_gil(|py| {
        let bound = val.bind(py);
        let tup = bound
            .downcast::<PyTuple>()
            .map_err(|_| PyValueError::new_err("structure: expected a (rows, cols) tuple"))?;
        if tup.len() != 2 {
            return Err(PyValueError::new_err(
                "structure: expected a (rows, cols) tuple of length 2",
            ));
        }
        let rows_obj = tup.get_item(0)?;
        let cols_obj = tup.get_item(1)?;
        let rows = extract_index_vec(&rows_obj.unbind(), nnz, "structure rows")?;
        let cols = extract_index_vec(&cols_obj.unbind(), nnz, "structure cols")?;
        Ok((rows, cols))
    })
}

fn extract_index_vec(val: &Py<PyAny>, nnz: usize, what: &str) -> PyResult<Vec<Index>> {
    Python::with_gil(|py| {
        let bound = val.bind(py);
        // Try int NumPy array first.
        if let Ok(arr) = bound.downcast::<PyArray1<i64>>() {
            let v = unsafe { arr.as_slice()? }
                .iter()
                .map(|&x| x as Index)
                .collect::<Vec<_>>();
            if v.len() != nnz {
                return Err(PyValueError::new_err(format!(
                    "{what}: expected {} entries, got {}",
                    nnz,
                    v.len()
                )));
            }
            return Ok(v);
        }
        if let Ok(arr) = bound.downcast::<PyArray1<i32>>() {
            let v = unsafe { arr.as_slice()? }
                .iter()
                .map(|&x| x as Index)
                .collect::<Vec<_>>();
            if v.len() != nnz {
                return Err(PyValueError::new_err(format!(
                    "{what}: expected {} entries, got {}",
                    nnz,
                    v.len()
                )));
            }
            return Ok(v);
        }
        // Sequence fallback (list / tuple / generic iterable).
        let mut out = Vec::with_capacity(nnz);
        for item in bound.iter()? {
            let v: i64 = item?.extract()?;
            out.push(v as Index);
        }
        if out.len() != nnz {
            return Err(PyValueError::new_err(format!(
                "{what}: expected {} entries, got {}",
                nnz,
                out.len()
            )));
        }
        Ok(out)
    })
}
