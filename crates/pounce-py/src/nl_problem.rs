//! `.nl` loader for Python.
//!
//! [`read_nl`] parses an AMPL `.nl` file through pounce's own reader
//! ([`pounce_nl::nl_reader::read_nl_file`]) and returns a [`PyNlProblem`]
//! that hands back the model's evaluators: objective, gradient, and
//! Lagrangian Hessian plus the constraint values and Jacobian. The heavy
//! lifting (the reverse-mode AD tape, sparsity, external functions) is the
//! same [`pounce_nl::nl_reader::NlTnlp`] the CLI solves with, so a Python
//! caller sees exactly the derivatives pounce itself uses.
//!
//! ```python
//! import pounce, numpy as np
//! p = pounce.read_nl("model.nl")
//! x = np.asarray(p.x0)
//! f  = p.objective(x)              # float
//! g  = p.gradient(x)               # ndarray[n]
//! c  = p.constraints(x)            # ndarray[m]
//! Jr, Jc = p.jacobian_structure()  # COO rows / cols (0-based)
//! Jv = p.jacobian(x)               # ndarray[nnz_jac], aligned to (Jr, Jc)
//! Hr, Hc = p.hessian_structure()   # lower-triangle rows / cols
//! Hv = p.hessian(x)                # ndarray[nnz_h] of the Lagrangian Hessian
//! ```
//!
//! Values follow the solver's (minimization) convention: for a `.nl` whose
//! original sense is `maximize`, the objective/gradient/Hessian are negated
//! so that minimizing them solves the model. The original sense is exposed
//! as [`PyNlProblem::minimize`].

use std::cell::RefCell;

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use pounce_common::types::{Index, Number};
use pounce_nl::nl_reader::{read_nl_file, NlTnlp};
use pounce_nlp::tnlp::{SparsityRequest, TNLP};

/// A `.nl` model loaded through pounce's reader, exposing its evaluators.
// `NlTnlp` shares `Expr` nodes via `Rc` (CSE), so it is not `Send`; the
// pyclass is `unsendable` and stays on the thread that created it (fine
// under the GIL).
#[pyclass(unsendable, module = "pounce", name = "NlProblem")]
pub struct PyNlProblem {
    tnlp: RefCell<NlTnlp>,
    n: usize,
    m: usize,
    nnz_jac: usize,
    nnz_h: usize,
    // Metadata captured before `prob` was moved into `NlTnlp`.
    minimize: bool,
    obj_constant: Number,
    x0: Vec<Number>,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    g_l: Vec<Number>,
    g_u: Vec<Number>,
    var_names: Vec<String>,
    con_names: Vec<String>,
}

/// Decode a 1-D float input (NumPy `float64` array or any float sequence)
/// into a `Vec<f64>` of the expected length.
fn decode_vec(val: &Bound<'_, PyAny>, expected: usize, what: &str) -> PyResult<Vec<Number>> {
    if let Ok(arr) = val.downcast::<PyArray1<Number>>() {
        let len = arr.len();
        if len != expected {
            return Err(PyValueError::new_err(format!(
                "{what}: expected length {expected}, got {len}"
            )));
        }
        return Ok(unsafe { arr.as_slice()? }.to_vec());
    }
    let mut out = Vec::with_capacity(expected);
    for item in val.iter()? {
        out.push(item?.extract::<Number>()?);
    }
    if out.len() != expected {
        return Err(PyValueError::new_err(format!(
            "{what}: expected length {expected}, got {}",
            out.len()
        )));
    }
    Ok(out)
}

#[pymethods]
impl PyNlProblem {
    /// Number of variables.
    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    /// Number of constraints.
    #[getter]
    fn m(&self) -> usize {
        self.m
    }

    /// Number of structurally non-zero Jacobian entries.
    #[getter]
    fn nnz_jac(&self) -> usize {
        self.nnz_jac
    }

    /// Number of stored (lower-triangle) Lagrangian-Hessian entries.
    #[getter]
    fn nnz_hess(&self) -> usize {
        self.nnz_h
    }

    /// `True` if the model's original sense is minimize, `False` if it was
    /// `maximize` (in which case the returned objective is negated).
    #[getter]
    fn minimize(&self) -> bool {
        self.minimize
    }

    /// Constant offset of the objective.
    #[getter]
    fn obj_constant(&self) -> Number {
        self.obj_constant
    }

    /// Starting point from the `.nl` file (length `n`).
    #[getter]
    fn x0<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Number>> {
        self.x0.clone().into_pyarray_bound(py)
    }

    /// Variable lower bounds (length `n`).
    #[getter]
    fn x_l<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Number>> {
        self.x_l.clone().into_pyarray_bound(py)
    }

    /// Variable upper bounds (length `n`).
    #[getter]
    fn x_u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Number>> {
        self.x_u.clone().into_pyarray_bound(py)
    }

    /// Constraint lower bounds (length `m`).
    #[getter]
    fn g_l<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Number>> {
        self.g_l.clone().into_pyarray_bound(py)
    }

    /// Constraint upper bounds (length `m`).
    #[getter]
    fn g_u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Number>> {
        self.g_u.clone().into_pyarray_bound(py)
    }

    /// Variable names from the sibling `.col` file (empty if absent).
    #[getter]
    fn var_names(&self) -> Vec<String> {
        self.var_names.clone()
    }

    /// Constraint names from the sibling `.row` file (empty if absent).
    #[getter]
    fn con_names(&self) -> Vec<String> {
        self.con_names.clone()
    }

    /// Objective value `f(x)`.
    fn objective(&self, x: &Bound<'_, PyAny>) -> PyResult<Number> {
        let xv = decode_vec(x, self.n, "objective: x")?;
        self.tnlp
            .borrow_mut()
            .eval_f(&xv, true)
            .ok_or_else(|| PyValueError::new_err("objective evaluation failed"))
    }

    /// Objective gradient `∇f(x)` (length `n`).
    fn gradient<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let xv = decode_vec(x, self.n, "gradient: x")?;
        let mut grad = vec![0.0; self.n];
        if !self.tnlp.borrow_mut().eval_grad_f(&xv, true, &mut grad) {
            return Err(PyValueError::new_err("gradient evaluation failed"));
        }
        Ok(grad.into_pyarray_bound(py))
    }

    /// Constraint values `g(x)` (length `m`).
    fn constraints<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let xv = decode_vec(x, self.n, "constraints: x")?;
        let mut g = vec![0.0; self.m];
        if !self.tnlp.borrow_mut().eval_g(&xv, true, &mut g) {
            return Err(PyValueError::new_err("constraint evaluation failed"));
        }
        Ok(g.into_pyarray_bound(py))
    }

    /// Jacobian sparsity as 0-based COO `(rows, cols)`, each length
    /// `nnz_jac`. Aligns entry-for-entry with [`Self::jacobian`].
    fn jacobian_structure<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<Index>>, Bound<'py, PyArray1<Index>>)> {
        let mut rows = vec![0 as Index; self.nnz_jac];
        let mut cols = vec![0 as Index; self.nnz_jac];
        let ok = self.tnlp.borrow_mut().eval_jac_g(
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut rows,
                jcol: &mut cols,
            },
        );
        if !ok {
            return Err(PyValueError::new_err("jacobian structure failed"));
        }
        Ok((rows.into_pyarray_bound(py), cols.into_pyarray_bound(py)))
    }

    /// Jacobian values at `x` (length `nnz_jac`), aligned to
    /// [`Self::jacobian_structure`].
    fn jacobian<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let xv = decode_vec(x, self.n, "jacobian: x")?;
        let mut values = vec![0.0; self.nnz_jac];
        let ok = self.tnlp.borrow_mut().eval_jac_g(
            Some(&xv),
            true,
            SparsityRequest::Values {
                values: &mut values,
            },
        );
        if !ok {
            return Err(PyValueError::new_err("jacobian evaluation failed"));
        }
        Ok(values.into_pyarray_bound(py))
    }

    /// Lower-triangle Lagrangian-Hessian sparsity as 0-based COO
    /// `(rows, cols)`, each length `nnz_hess`. Aligns with
    /// [`Self::hessian`].
    fn hessian_structure<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<Index>>, Bound<'py, PyArray1<Index>>)> {
        let mut rows = vec![0 as Index; self.nnz_h];
        let mut cols = vec![0 as Index; self.nnz_h];
        let ok = self.tnlp.borrow_mut().eval_h(
            None,
            false,
            1.0,
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut rows,
                jcol: &mut cols,
            },
        );
        if !ok {
            return Err(PyValueError::new_err("hessian structure failed"));
        }
        Ok((rows.into_pyarray_bound(py), cols.into_pyarray_bound(py)))
    }

    /// Lower-triangle of the Lagrangian Hessian
    /// `obj_factor·∇²f + Σ_i lam_i·∇²g_i` at `x` (length `nnz_hess`),
    /// aligned to [`Self::hessian_structure`].
    ///
    /// `lam` defaults to zeros (the objective Hessian alone); `obj_factor`
    /// defaults to 1.0.
    #[pyo3(signature = (x, lam=None, obj_factor=1.0))]
    fn hessian<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        lam: Option<&Bound<'_, PyAny>>,
        obj_factor: Number,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let xv = decode_vec(x, self.n, "hessian: x")?;
        let lamv = match lam {
            Some(l) => decode_vec(l, self.m, "hessian: lam")?,
            None => vec![0.0; self.m],
        };
        let mut values = vec![0.0; self.nnz_h];
        let ok = self.tnlp.borrow_mut().eval_h(
            Some(&xv),
            true,
            obj_factor,
            Some(&lamv),
            true,
            SparsityRequest::Values {
                values: &mut values,
            },
        );
        if !ok {
            return Err(PyValueError::new_err("hessian evaluation failed"));
        }
        Ok(values.into_pyarray_bound(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "NlProblem(n={}, m={}, nnz_jac={}, nnz_hess={}, minimize={})",
            self.n, self.m, self.nnz_jac, self.nnz_h, self.minimize
        )
    }
}

/// Parse an AMPL `.nl` file and return its evaluable [`PyNlProblem`].
///
/// Sibling `.col` / `.row` files (if present) supply variable / constraint
/// names. External (imported) functions are resolved via `AMPLFUNC` exactly
/// as the CLI does.
#[pyfunction]
pub fn read_nl(path: &str) -> PyResult<PyNlProblem> {
    let prob = read_nl_file(std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(format!("read_nl: {e}")))?;

    // Capture metadata before `prob` is consumed by `NlTnlp::new`.
    let minimize = prob.minimize;
    let obj_constant = prob.obj_constant;
    let x0 = prob.x0.clone();
    let x_l = prob.x_l.clone();
    let x_u = prob.x_u.clone();
    let g_l = prob.g_l.clone();
    let g_u = prob.g_u.clone();
    let var_names = prob.var_names.clone();
    let con_names = prob.con_names.clone();
    let n = prob.n;
    let m = prob.m;

    let mut tnlp = NlTnlp::new(prob);
    let info = tnlp
        .get_nlp_info()
        .ok_or_else(|| PyValueError::new_err("read_nl: get_nlp_info returned None"))?;

    Ok(PyNlProblem {
        tnlp: RefCell::new(tnlp),
        n,
        m,
        nnz_jac: info.nnz_jac_g as usize,
        nnz_h: info.nnz_h_lag as usize,
        minimize,
        obj_constant,
        x0,
        x_l,
        x_u,
        g_l,
        g_u,
        var_names,
        con_names,
    })
}
