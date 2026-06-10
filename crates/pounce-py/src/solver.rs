//! `pounce.Solver` PyO3 class — session-style wrapper around
//! [`pounce_sensitivity::Solver`].
//!
//! Holds a converged factor between calls so multiple cheap operations
//! (`kkt_solve`, `parametric_step`, `reduced_hessian`) can run against
//! the same factorization without re-running the IPM:
//!
//! ```python
//! import pounce
//! solver = pounce.Solver(problem)
//! x, info = solver.solve(x0=x0)
//! dx = solver.parametric_step([2, 3], [-0.5, 0.0])
//! H_R = solver.reduced_hessian([2, 3])
//! ```
//!
//! Each `solve()` call rebuilds the underlying [`pounce_algorithm::IpoptApplication`]
//! from the [`crate::PyProblem`]'s current option set (so option
//! changes between calls take effect); future Phase 3b work will add a
//! `resolve()` that reuses the cached symbolic factor across solves.

use numpy::{IntoPyArray, PyArray1};
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::TNLP;
use pounce_sensitivity::{Solver as RustSolver, SolverError};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cell::RefCell;
use std::rc::Rc;

use crate::problem::{build_info_dict, PyProblem};

/// Session-style wrapper around [`pounce_sensitivity::Solver`].
#[pyclass(name = "Solver", module = "pounce._pounce", unsendable)]
pub struct PySolver {
    /// Reference to the owning Python `Problem`. Used to re-prepare an
    /// application on each `solve()` call.
    problem: Py<PyProblem>,
    /// Held after a successful `solve()`. None before the first solve
    /// or if the last solve failed before convergence.
    state: Option<SessionState>,
}

struct SessionState {
    inner: RustSolver,
    /// Number of constraints `m` — cached so post-convergence range
    /// checks on `pin_constraint_indices` don't need a GIL acquire.
    m: usize,
}

#[pymethods]
impl PySolver {
    #[new]
    fn new(problem: Py<PyProblem>) -> Self {
        Self {
            problem,
            state: None,
        }
    }

    /// Run a (possibly cold-start) solve. Returns `(x, info_dict)` in
    /// the same shape as [`crate::PyProblem::solve`].
    #[pyo3(signature = (x0, lagrange=None, zl=None, zu=None))]
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        x0: Py<PyAny>,
        lagrange: Option<Py<PyAny>>,
        zl: Option<Py<PyAny>>,
        zu: Option<Py<PyAny>>,
    ) -> PyResult<(Bound<'py, PyArray1<Number>>, Bound<'py, PyDict>)> {
        let problem = self.problem.bind(py).borrow();
        let (app, bridge) = problem.prepare(py, x0, lagrange, zl, zu)?;
        let m = problem.m_index() as usize;
        drop(problem);

        let bridge_for_solver: Rc<RefCell<dyn TNLP>> = bridge.clone();
        let mut inner = RustSolver::new(app, bridge_for_solver);
        let status: ApplicationReturnStatus = inner.solve();
        let stats = inner.app().statistics();
        let info = build_info_dict(
            py,
            &bridge.borrow(),
            status,
            stats.iteration_count,
            stats.final_mu,
            stats.final_kkt_error,
            stats.final_dual_inf,
            stats.final_constr_viol,
            stats.final_compl,
        )?;
        let x_out = bridge.borrow().state.final_x.clone().into_pyarray_bound(py);
        let _ = bridge; // alive via inner's Rc<RefCell<dyn TNLP>> clone
        self.state = Some(SessionState { inner, m });
        Ok((x_out, info))
    }

    /// Solve `K · lhs = rhs` against the converged KKT factor. Returns
    /// the solution vector. `rhs` must have length [`Self::kkt_dim`].
    ///
    /// `K` is the **natural-units** (unscaled) KKT matrix: when the
    /// IPM solved with active NLP scaling (`nlp_scaling_method`,
    /// `obj_scaling_factor`, user scaling) the back-solve is
    /// conjugated so RHS and solution are in the user's own units
    /// (pounce#128). Pass `scaled=True` for the raw back-solve against
    /// the factor exactly as the IPM holds it.
    #[pyo3(signature = (rhs, scaled = false))]
    fn kkt_solve<'py>(
        &self,
        py: Python<'py>,
        rhs: Vec<Number>,
        scaled: bool,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("kkt_solve: no converged factor (call solve() first)")
        })?;
        let mut lhs = vec![0.0; rhs.len()];
        let res = if scaled {
            s.inner.kkt_solve_scaled(&rhs, &mut lhs)
        } else {
            s.inner.kkt_solve(&rhs, &mut lhs)
        };
        res.map_err(solver_error_to_py)?;
        Ok(lhs.into_pyarray_bound(py))
    }

    /// Batched-RHS back-solve. `rhs_flat` is a row-major
    /// `(n_rhs, kkt_dim)` buffer (one RHS per row); the returned flat
    /// array has the same length and layout. Equivalent to looping
    /// [`Self::kkt_solve`] over each row, but with a single FFI hop —
    /// which matters for `jax.jacrev` over a JaxProblem batched solve,
    /// where the JAX backward is vmap'd once per cotangent and each
    /// cross-thread `pure_callback` dispatch otherwise dominates the
    /// real back-solve cost (pounce#77 follow-up). Same converged
    /// factor and same per-RHS work — only the per-call FFI / executor
    /// pin overhead is amortised.
    ///
    /// Like [`Self::kkt_solve`], results are in natural (unscaled)
    /// units; pass `scaled=True` for the raw solver-space back-solve
    /// (pounce#128).
    #[pyo3(signature = (rhs_flat, n_rhs, scaled = false))]
    fn kkt_solve_many<'py>(
        &self,
        py: Python<'py>,
        rhs_flat: Vec<Number>,
        n_rhs: usize,
        scaled: bool,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("kkt_solve_many: no converged factor (call solve() first)")
        })?;
        let dim = s.inner.kkt_dim().ok_or_else(|| {
            PyRuntimeError::new_err("kkt_solve_many: no converged factor (call solve() first)")
        })?;
        if n_rhs == 0 {
            return Ok(Vec::<Number>::new().into_pyarray_bound(py));
        }
        if rhs_flat.len() != n_rhs * dim {
            return Err(PyValueError::new_err(format!(
                "kkt_solve_many: rhs_flat length {} != n_rhs ({}) * kkt_dim ({}) = {}",
                rhs_flat.len(),
                n_rhs,
                dim,
                n_rhs * dim,
            )));
        }
        let mut lhs_flat = vec![0.0; n_rhs * dim];
        let res = if scaled {
            s.inner
                .kkt_solve_many_scaled(&rhs_flat, &mut lhs_flat, n_rhs)
        } else {
            s.inner.kkt_solve_many(&rhs_flat, &mut lhs_flat, n_rhs)
        };
        res.map_err(solver_error_to_py)?;
        Ok(lhs_flat.into_pyarray_bound(py))
    }

    /// First-order parametric step `Δx ≈ ∂x*/∂p · Δp` against the held
    /// factor. `pin_constraint_indices` are 0-based indices into
    /// `g(x)` (must equal the parameter-pin equality constraints).
    fn parametric_step<'py>(
        &self,
        py: Python<'py>,
        pin_constraint_indices: Vec<i64>,
        deltas: Vec<Number>,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("parametric_step: no converged factor (call solve() first)")
        })?;
        let pins = validate_pins(&pin_constraint_indices, s.m)?;
        if deltas.len() != pins.len() {
            return Err(PyValueError::new_err(format!(
                "deltas length {} must equal pin_constraint_indices length {}",
                deltas.len(),
                pins.len(),
            )));
        }
        let dx = s
            .inner
            .parametric_step(&pins, &deltas)
            .map_err(solver_error_to_py)?;
        Ok(dx.into_pyarray_bound(py))
    }

    /// Reduced Hessian `H_R = obj_scal · B K⁻¹ Bᵀ` over the pinned
    /// rows, in **natural (unscaled) units**: any NLP scaling baked
    /// into the converged factor is undone, so `-inv(H_R)` is directly
    /// the parameter covariance of an estimation problem regardless of
    /// `nlp_scaling_method` (pounce#128). Pass `scaled=True` for the
    /// solver-space value pounce returned before #128. `obj_scal`
    /// survives as a plain extra multiplier (default 1.0); it is no
    /// longer needed to undo pounce's own scaling. Returned as a
    /// `n²`-long column-major flat array
    /// (`n = pin_constraint_indices.len()`).
    #[pyo3(signature = (pin_constraint_indices, obj_scal = 1.0, scaled = false))]
    fn reduced_hessian<'py>(
        &self,
        py: Python<'py>,
        pin_constraint_indices: Vec<i64>,
        obj_scal: Number,
        scaled: bool,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("reduced_hessian: no converged factor (call solve() first)")
        })?;
        let pins = validate_pins(&pin_constraint_indices, s.m)?;
        let hr = if scaled {
            s.inner.compute_reduced_hessian_scaled(&pins, obj_scal)
        } else {
            s.inner.compute_reduced_hessian(&pins, obj_scal)
        }
        .map_err(solver_error_to_py)?;
        Ok(hr.into_pyarray_bound(py))
    }

    /// Effective NLP scaling the IPM applied on the held solve, as a
    /// dict with keys `obj` (float, the objective factor `df`),
    /// `c_scale` and `d_scale` (each an ndarray of per-row factors
    /// over the algorithm's equality / inequality blocks, or `None`
    /// when that block carries no row scaling).
    /// `{"obj": 1.0, "c_scale": None, "d_scale": None}` ⇔ no scaling
    /// was active.
    #[getter]
    fn nlp_scaling<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("nlp_scaling: no converged factor (call solve() first)")
        })?;
        let (df, dc, dd) = s.inner.nlp_scaling().map_err(solver_error_to_py)?;
        let out = PyDict::new_bound(py);
        out.set_item("obj", df)?;
        out.set_item("c_scale", crate::problem::opt_vec_to_py(py, dc))?;
        out.set_item("d_scale", crate::problem::opt_vec_to_py(py, dd))?;
        Ok(out)
    }

    /// Dimension of the full compound KKT vector. `None` if no
    /// converged factor is held yet.
    #[getter]
    fn kkt_dim(&self) -> Option<usize> {
        self.state.as_ref().and_then(|s| s.inner.kkt_dim())
    }

    /// Per-block dimensions of the compound KKT vector in the flat
    /// `x || s || y_c || y_d || z_l || z_u || v_l || v_u` packing
    /// order. Returns an 8-tuple; `None` if no converged factor is
    /// held. Useful for callers that need to pack a partial RHS for
    /// `kkt_solve` (e.g. the JAX `custom_vjp` backward in
    /// `pounce.jax`, which puts the upstream cotangent in the x-block
    /// and reads back the y_c block).
    #[getter]
    fn block_dims(&self) -> Option<[usize; 8]> {
        self.state.as_ref().and_then(|s| s.inner.block_dims())
    }

    /// `True` iff a converged factor is currently held.
    #[getter]
    fn converged(&self) -> bool {
        self.state
            .as_ref()
            .map(|s| s.inner.converged().is_some())
            .unwrap_or(false)
    }
}

fn validate_pins(pin_constraint_indices: &[i64], m: usize) -> PyResult<Vec<Index>> {
    pin_constraint_indices
        .iter()
        .map(|&i| {
            if i < 0 || (i as usize) >= m {
                Err(PyValueError::new_err(format!(
                    "pin_constraint_indices[..] = {i} out of range [0, m={m})",
                )))
            } else {
                Ok(i as Index)
            }
        })
        .collect()
}

fn solver_error_to_py(e: SolverError) -> PyErr {
    match e {
        SolverError::NotConverged => PyRuntimeError::new_err("Solver: not converged"),
        SolverError::BadShape {
            what,
            got,
            expected,
        } => PyValueError::new_err(format!(
            "Solver: {what} length {got} != expected {expected}"
        )),
        SolverError::BacksolveFailed => PyRuntimeError::new_err("Solver: back-solve failed"),
        SolverError::SensComputationFailed(msg) => {
            PyRuntimeError::new_err(format!("Solver: sensitivity computation failed: {msg}"))
        }
    }
}
