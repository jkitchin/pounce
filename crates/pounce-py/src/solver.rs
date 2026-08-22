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
use pounce_sensitivity::activity;
use pounce_sensitivity::{Solver as RustSolver, SolverError};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cell::RefCell;
use std::rc::Rc;

use crate::problem::{PyProblem, build_info_dict};

/// Session-style wrapper around [`pounce_sensitivity::Solver`].
//
// Still `unsendable`, unlike `NlProblem` / `DenseLU` / `SparseLU`
// (pounce#477). This one is not a policy default: `RustSolver` is
// genuinely `!Send` — the ported Ipopt object graph is `Rc`-based
// (`Rc<RefCell<dyn TNLP>>`, `Rc<RegisteredOptions>`, `Rc<Journalist>`,
// …) and holds non-`Send` linear-solver and restoration trait objects,
// so there is nothing to drop the marker in favor of.
//
// The cost is the one #477 describes: a cross-thread call raises
// `PanicException` (a `BaseException`, so `except Exception` misses it)
// and a foreign-thread collection writes an unraisable and leaks. The
// Python layer keeps `Solver` instances on their creating thread via
// `threading.local` for exactly this reason — see
// `pounce/jax/_problem.py`. Trading the panic for a catchable error
// would mean an `unsafe impl Send` wrapper doing the affinity check by
// hand; not worth it while the callers are already thread-pinned.
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
    /// the same shape as [`crate::PyProblem::solve`], including its
    /// multiplier-seed semantics: under `warm_start_init_point=yes` a
    /// NaN entry in `lagrange`/`zl`/`zu` means "unseeded" and takes
    /// the initializer's own resolved default.
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
        let inner = RustSolver::new(app, bridge_for_solver);

        // Release the GIL during the IPM run so `Solver` instances on other OS
        // threads (e.g. concurrent `curve_fit` / jax-host solves) overlap their
        // iterations. Every TNLP callback in `tnlp_bridge.rs` re-acquires the
        // GIL via `Python::with_gil` before touching Python, so re-entrancy is
        // safe and serialized the usual way. Mirrors `PyProblem::solve`.
        //
        // SAFETY: `inner` carries `Rc<RefCell<…>>` (pounce_nlp is single-thread
        // refcounted). `allow_threads` requires `Send`, so we wrap the move in a
        // transparent `SendGuard`; the closure does not actually cross OS
        // threads — `allow_threads` runs its body on the calling thread after
        // `PyEval_SaveThread`, so the `Rc`/`RefCell` are only ever touched by
        // this one thread. Method-call captures (vs. field-access `.0`) defeat
        // the 2021-edition disjoint-capture rule, so the closure captures the
        // whole `SendGuard` rather than peeking at the inner `Rc`.
        struct SendGuard<T>(T);
        unsafe impl<T> Send for SendGuard<T> {}
        impl<T> SendGuard<T> {
            fn into_inner(self) -> T {
                self.0
            }
            fn new(v: T) -> Self {
                Self(v)
            }
        }
        let inner_guard = SendGuard::new(inner);
        let (status, inner_back): (ApplicationReturnStatus, SendGuard<RustSolver>) = py
            .allow_threads(move || {
                let mut inner = inner_guard.into_inner();
                let status = inner.solve();
                (status, SendGuard::new(inner))
            });
        let inner = inner_back.into_inner();
        let stats = inner.app().statistics();
        let info = build_info_dict(
            py,
            &bridge.borrow(),
            status,
            &stats,
            inner.app().warm_start_diagnostics(),
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
    /// `obj_scaling_factor`, user scaling — including a per-variable
    /// change of variables, gh#486) the back-solve is conjugated so
    /// RHS and solution are in the user's own units (pounce#128). Pass
    /// `scaled=True` for the raw back-solve against the factor exactly
    /// as the IPM holds it.
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

    /// Parametric step with the bounds respected by pinning rather than
    /// clamping, as `(dx, pinned, stop)`.
    ///
    /// `parametric_step` points where the linear predictor points,
    /// which can be outside the box. Clamping a coordinate back to its
    /// bound leaves every other coordinate at its predictor value, so
    /// the answer is feasible but no longer consistent with the KKT
    /// relations. This adds a row pinning the offending coordinate at
    /// the bound and re-solves, so the others move with it, which is
    /// the refinement upstream runs under `sens_boundcheck`.
    ///
    /// `pinned` lists the rows the refinement constrained, worst
    /// violator first, as **compound KKT rows** rather than variable
    /// indices. Do not index a variable vector with them. Read them
    /// against `block_dims()`, which gives the eight block lengths in
    /// the `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)` order:
    ///
    /// * a row below `dims[0]` is a variable pinned at its bound, and
    ///   the row is its var-x index;
    /// * a row at or above `dims[0] + dims[1] + dims[2] + dims[3]` is a
    ///   bound multiplier driven to zero, which releases that bound and
    ///   is the opposite action. Subtracting that offset gives its
    ///   position in the `z_l` block, or in `z_u` past `dims[4]`.
    ///
    /// No other block appears. The list is empty when the plain step
    /// already respects every bound, and the step is then the plain
    /// one.
    ///
    /// A pass takes every crossing it can see, pins them together and
    /// re-solves, so the number of passes tracks how far the active set
    /// has to move rather than how many bounds it moves. The
    /// factorization is never rebuilt for a pin, but the Schur
    /// complement is, so a pass carrying `k` pins costs one dense
    /// `k × k` solve and `k + 1` back-solves. What counts as outside a
    /// bound is `bound_eps` when one is given, and the solve's own
    /// `bound_relax_factor`, floored, when it is `None`.
    ///
    /// `max_iter` is a safety limit rather than a budget. It was a
    /// budget while a pass took one pin, which needed as many passes as
    /// there were crossings, and on a model with more crossings than
    /// passes the limit picked the answer (gh#732).
    ///
    /// `stop` says why the refinement stopped, and is one of:
    ///
    /// * `"settled"` — nothing is outside a bound and no multiplier is
    ///   negative. Only this one says it finished.
    /// * `"iteration_limit"` — `max_iter` passes were spent with
    ///   violations left. Raising it may finish the job.
    /// * `"degrees_of_freedom"` — the conditions exhausted the
    ///   problem's degrees of freedom, so no step holds every bound at
    ///   once. No budget helps.
    /// * `"worse_than_plain"` — the refinement ended further outside
    ///   the bounds than the unrefined step, which was returned instead
    ///   with an empty `pinned`.
    ///
    /// None is an error, and `pinned` says how far the refinement got.
    #[pyo3(signature = (pin_constraint_indices, deltas, max_iter=16, bound_eps=None))]
    fn parametric_step_bounded<'py>(
        &self,
        py: Python<'py>,
        pin_constraint_indices: Vec<i64>,
        deltas: Vec<Number>,
        max_iter: usize,
        bound_eps: Option<Number>,
    ) -> PyResult<(Bound<'py, PyArray1<Number>>, Vec<i64>, &'static str)> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "parametric_step_bounded: no converged factor (call solve() first)",
            )
        })?;
        let pins = validate_pins(&pin_constraint_indices, s.m)?;
        if deltas.len() != pins.len() {
            return Err(PyValueError::new_err(format!(
                "deltas length {} must equal pin_constraint_indices length {}",
                deltas.len(),
                pins.len(),
            )));
        }
        let (dx, pinned, stop) = s
            .inner
            .parametric_step_bounded(&pins, &deltas, max_iter, bound_eps)
            .map_err(solver_error_to_py)?;
        Ok((
            dx.into_pyarray_bound(py),
            pinned.into_iter().map(|p| p as i64).collect(),
            stop.as_str(),
        ))
    }

    /// Parametric step applied a little at a time, stopping wherever
    /// the active set changes, as `(dx, segments)`.
    ///
    /// `parametric_step_bounded` decides every condition at the base
    /// point. This advances to the fraction of the perturbation that
    /// reaches the first change, applies it, and continues from there
    /// under the new set, so the result is piecewise linear in the
    /// parameter.
    ///
    /// `segments` is one tuple per breakpoint crossed, in the order
    /// crossed, as `(fraction, var_row, lower, pinned)`. `fraction` is
    /// how far along the perturbation the change happened, `var_row`
    /// is the variable's row in the factor's x block for every kind of
    /// change, `lower` is `True` when the bound involved is the lower
    /// one, and `pinned` is `True` for a variable reaching a bound and
    /// `False` for a variable leaving one.
    ///
    /// `max_iter` caps the breakpoints crossed, and is in practice a
    /// budget on factorizations, since a pin is a back-solve against
    /// the held factor while a release re-factors. Reaching the cap is
    /// not an error: the rest of the perturbation is taken in one step
    /// under the active set reached, and a returned `segments` of
    /// length `max_iter` is what says so.
    #[pyo3(signature = (pin_constraint_indices, deltas, max_iter=16))]
    fn parametric_step_path<'py>(
        &self,
        py: Python<'py>,
        pin_constraint_indices: Vec<i64>,
        deltas: Vec<Number>,
        max_iter: usize,
    ) -> PyResult<(Bound<'py, PyArray1<Number>>, Vec<(Number, i64, bool, bool)>)> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "parametric_step_path: no converged factor (call solve() first)",
            )
        })?;
        let pins = validate_pins(&pin_constraint_indices, s.m)?;
        if deltas.len() != pins.len() {
            return Err(PyValueError::new_err(format!(
                "deltas length {} must equal pin_constraint_indices length {}",
                deltas.len(),
                pins.len(),
            )));
        }
        let (dx, segments) = s
            .inner
            .parametric_step_path(&pins, &deltas, max_iter)
            .map_err(solver_error_to_py)?;
        Ok((
            dx.into_pyarray_bound(py),
            segments
                .into_iter()
                .map(|g| (g.at, g.var_row as i64, g.lower, g.pinned))
                .collect(),
        ))
    }

    /// The weakly active bounds at the converged base point, as
    /// `(var_row, lower)` pairs: on the bound with a multiplier of the
    /// same order as the slack, the classifier's WEAKLY_ACTIVE and
    /// AMBIGUOUS labels. Empty when the base point is clean. Requires
    /// `bound_relax_factor=0`, like `classify_activity`.
    fn weakly_active_bounds(&self) -> PyResult<Vec<(i64, bool)>> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "weakly_active_bounds: no converged factor (call solve() first)",
            )
        })?;
        let weak = s.inner.weakly_active_bounds().map_err(solver_error_to_py)?;
        Ok(weak.iter().map(|w| (w.var_row as i64, w.lower)).collect())
    }

    /// `parametric_step_bounded` with the weak-row decision
    /// supplied by the caller (var-x rows the direction holds) instead
    /// of searched for, as `(dx, pinned, stop)`. Study surface for an
    /// externally solved eq. 14 QP.
    #[pyo3(signature = (pin_constraint_indices, deltas, held_var_rows, max_iter=16, bound_eps=None))]
    fn parametric_step_bounded_decided<'py>(
        &self,
        py: Python<'py>,
        pin_constraint_indices: Vec<i64>,
        deltas: Vec<Number>,
        held_var_rows: Vec<i64>,
        max_iter: usize,
        bound_eps: Option<Number>,
    ) -> PyResult<(Bound<'py, PyArray1<Number>>, Vec<i64>, &'static str)> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "parametric_step_bounded_decided: no converged factor (call solve() first)",
            )
        })?;
        let pins = validate_pins(&pin_constraint_indices, s.m)?;
        if deltas.len() != pins.len() {
            return Err(PyValueError::new_err(format!(
                "deltas length {} must equal pin_constraint_indices length {}",
                deltas.len(),
                pins.len(),
            )));
        }
        // The rows index the primal block directly, and an
        // out-of-range one reaches a raw `d[i]` inside the direction,
        // so it is a panic out of the extension rather than a raised
        // error. `Index` is signed, so a negative row becomes a very
        // large `usize` there. Checked here, the way the sibling
        // argument on this same call already is.
        let n_x = s
            .inner
            .block_dims()
            .ok_or_else(|| PyRuntimeError::new_err("no converged factor (call solve() first)"))?[0];
        let held: Vec<Index> = validate_var_rows(&held_var_rows, n_x)?;
        let (dx, pinned, stop) = s
            .inner
            .parametric_step_bounded_decided(&pins, &deltas, max_iter, &held, bound_eps)
            .map_err(solver_error_to_py)?;
        Ok((
            dx.into_pyarray_bound(py),
            pinned.into_iter().map(|p| p as i64).collect(),
            stop.as_str(),
        ))
    }

    /// `parametric_step_path` with the weak-row decision supplied by
    /// the caller (var-x rows the direction holds) instead of searched
    /// for, as `(dx, segments)`. Study surface for an externally
    /// solved eq. 14 QP.
    #[pyo3(signature = (pin_constraint_indices, deltas, held_var_rows, max_iter=16))]
    fn parametric_step_path_decided<'py>(
        &self,
        py: Python<'py>,
        pin_constraint_indices: Vec<i64>,
        deltas: Vec<Number>,
        held_var_rows: Vec<i64>,
        max_iter: usize,
    ) -> PyResult<(Bound<'py, PyArray1<Number>>, Vec<(Number, i64, bool, bool)>)> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "parametric_step_path_decided: no converged factor (call solve() first)",
            )
        })?;
        let pins = validate_pins(&pin_constraint_indices, s.m)?;
        if deltas.len() != pins.len() {
            return Err(PyValueError::new_err(format!(
                "deltas length {} must equal pin_constraint_indices length {}",
                deltas.len(),
                pins.len(),
            )));
        }
        // The rows index the primal block directly, and an
        // out-of-range one reaches a raw `d[i]` inside the direction,
        // so it is a panic out of the extension rather than a raised
        // error. `Index` is signed, so a negative row becomes a very
        // large `usize` there. Checked here, the way the sibling
        // argument on this same call already is.
        let n_x = s
            .inner
            .block_dims()
            .ok_or_else(|| PyRuntimeError::new_err("no converged factor (call solve() first)"))?[0];
        let held: Vec<Index> = validate_var_rows(&held_var_rows, n_x)?;
        let (dx, segments) = s
            .inner
            .parametric_step_path_decided(&pins, &deltas, max_iter, &held)
            .map_err(solver_error_to_py)?;
        Ok((
            dx.into_pyarray_bound(py),
            segments
                .into_iter()
                .map(|g| (g.at, g.var_row as i64, g.lower, g.pinned))
                .collect(),
        ))
    }

    /// The eq. 14 directional derivative, decided by the pounce-qp
    /// active-set engine on the reduced weak-row problem. `max_iter`
    /// is the total
    /// back-solve budget: the all-released solve, one basis column per
    /// engaged weak row, and the combined solve that recovers the
    /// direction all count against it. Only a budget of zero fails
    /// before any work, since which rows engage is not known until the
    /// all-released solve has run, so a budget too small to finish
    /// still pays that factorization. The error names the engaged
    /// count and the number to raise `degeneracy_iter` to. Returns
    /// `(dx, held_var_rows, backsolves_spent)`.
    #[pyo3(signature = (pin_constraint_indices, deltas, max_iter=16))]
    fn parametric_step_directional<'py>(
        &self,
        py: Python<'py>,
        pin_constraint_indices: Vec<i64>,
        deltas: Vec<Number>,
        max_iter: usize,
    ) -> PyResult<(Bound<'py, PyArray1<Number>>, Vec<i64>, usize)> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "parametric_step_directional: no converged factor (call solve() first)",
            )
        })?;
        let pins = validate_pins(&pin_constraint_indices, s.m)?;
        if deltas.len() != pins.len() {
            return Err(PyValueError::new_err(format!(
                "deltas length {} must equal pin_constraint_indices length {}",
                deltas.len(),
                pins.len(),
            )));
        }
        let (dx, held, work) = s
            .inner
            .parametric_step_directional(&pins, &deltas, max_iter)
            .map_err(solver_error_to_py)?;
        Ok((
            dx.into_pyarray_bound(py),
            held.into_iter().map(|p| p as i64).collect(),
            work,
        ))
    }

    /// Full KKT-space parametric step: like `parametric_step` but
    /// returns the whole compound vector `(x, s, y_c, y_d, z_l, z_u,
    /// v_l, v_u)` so multiplier sensitivities are available. Use
    /// `block_dims()` for the layout and `multiplier_rows()` to find a
    /// constraint's row.
    fn parametric_step_full<'py>(
        &self,
        py: Python<'py>,
        pin_constraint_indices: Vec<i64>,
        deltas: Vec<Number>,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "parametric_step_full: no converged factor (call solve() first)",
            )
        })?;
        let pins = validate_pins(&pin_constraint_indices, s.m)?;
        if deltas.len() != pins.len() {
            return Err(PyValueError::new_err(format!(
                "deltas length {} must equal pin_constraint_indices length {}",
                deltas.len(),
                pins.len(),
            )));
        }
        let dx_full = s
            .inner
            .parametric_step_full(&pins, &deltas)
            .map_err(solver_error_to_py)?;
        Ok(dx_full.into_pyarray_bound(py))
    }

    /// Newton iterations on the barrier system, refining a compound
    /// step against the held factor. One back-solve per iteration and
    /// no factorization.
    ///
    /// `step` is a full compound step, the shape
    /// [`Self::parametric_step_full`] returns, so any mode's result
    /// can be corrected. Returns the refined step and a dict holding
    /// the iterations spent, the residual before and after, whether
    /// the loop stopped early, how many bounds changed status, and the
    /// residual split into stationarity, feasibility and
    /// complementarity, which carry different units and different
    /// consequences for whether an estimate can be acted on.
    ///
    /// The corrector aims at the barrier solution at the μ the solve
    /// finished on, so the accuracy it reaches is bounded by that
    /// offset, and where the perturbation needs a bound the base point
    /// held tightly to leave, the held barrier diagonal cannot
    /// represent the change and the iterations make no progress. A
    /// returned `residual` equal to `initial_residual` is that case.
    /// The returned point always satisfies the variable bounds, so
    /// `max_iter = 0` costs one evaluation and reports the step's
    /// residual without iterating.
    #[pyo3(signature = (pin_constraint_indices, deltas, step, max_iter=5))]
    fn correct_step<'py>(
        &self,
        py: Python<'py>,
        pin_constraint_indices: Vec<i64>,
        deltas: Vec<Number>,
        step: Vec<Number>,
        max_iter: usize,
    ) -> PyResult<(Bound<'py, PyArray1<Number>>, Bound<'py, PyDict>)> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("correct_step: no converged factor (call solve() first)")
        })?;
        let pins = validate_pins(&pin_constraint_indices, s.m)?;
        if deltas.len() != pins.len() {
            return Err(PyValueError::new_err(format!(
                "deltas length {} must equal pin_constraint_indices length {}",
                deltas.len(),
                pins.len(),
            )));
        }
        let (out, rep) = s
            .inner
            .correct_step(&pins, &deltas, &step, max_iter)
            .map_err(solver_error_to_py)?;
        let info = PyDict::new_bound(py);
        info.set_item("iterations", rep.iterations)?;
        info.set_item("residual", rep.residual)?;
        info.set_item("initial_residual", rep.initial_residual)?;
        info.set_item("converged", rep.converged)?;
        info.set_item("active_set_changes", rep.released + rep.pinned)?;
        info.set_item("released", rep.released)?;
        info.set_item("pinned", rep.pinned)?;
        info.set_item("stationarity", rep.stationarity)?;
        info.set_item("feasibility", rep.feasibility)?;
        info.set_item("complementarity", rep.complementarity)?;
        Ok((out.into_pyarray_bound(py), info))
    }

    /// Rows of the compound KKT vector holding the equality
    /// multipliers for the given 0-based constraint (`g`) indices;
    /// `None` for inequality constraints.
    fn multiplier_rows(&self, g_indices: Vec<i64>) -> PyResult<Vec<Option<i64>>> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("multiplier_rows: no converged factor (call solve() first)")
        })?;
        let gs = validate_pins(&g_indices, s.m)?;
        let rows = s.inner.g_multiplier_rows(&gs).map_err(solver_error_to_py)?;
        Ok(rows.into_iter().map(|r| r.map(|v| v as i64)).collect())
    }

    /// Rows of the compound KKT vector holding the primal values for
    /// the given 0-based **user-space** variable (`x`) indices; `None`
    /// where the solve removed the column as a fixed variable
    /// (`lb == ub` under `fixed_variable_treatment=make_parameter`).
    ///
    /// The `x` counterpart of `multiplier_rows`. User-space indices —
    /// from the `.col` file, from `classify_activity()`, from
    /// `row_normal()` — are NOT factor rows: one fixed variable
    /// anywhere shifts every later column, and on a model without any
    /// the two spaces coincide, so the mistake is invisible in testing.
    /// Translate here before indexing `kkt_solve` / `parametric_step`
    /// results.
    fn primal_rows(&self, x_indices: Vec<i64>) -> PyResult<Vec<Option<i64>>> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("primal_rows: no converged factor (call solve() first)")
        })?;
        let n_full = s.inner.n_full_x().map_err(solver_error_to_py)?;
        let mut xs = Vec::with_capacity(x_indices.len());
        for &i in &x_indices {
            if i < 0 || (i as usize) >= n_full {
                return Err(PyValueError::new_err(format!(
                    "primal_rows: variable index {i} out of range [0, n={n_full})"
                )));
            }
            xs.push(i as Index);
        }
        let rows = s.inner.x_primal_rows(&xs).map_err(solver_error_to_py)?;
        Ok(rows.into_iter().map(|r| r.map(|v| v as i64)).collect())
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
    /// when that block carries no row scaling), and `x_scaling` (an
    /// ndarray of length `n` when the solve applied a per-variable
    /// change of variables `x̃ = d ⊙ x`, else `None`).
    /// `{"obj": 1.0, "c_scale": None, "d_scale": None,
    /// "x_scaling": None}` ⇔ no scaling was active.
    ///
    /// Every accessor on this class already reports natural units,
    /// `x_scaling` included (gh#486), so this dict describes the
    /// solve rather than handing back a correction to apply.
    #[getter]
    fn nlp_scaling<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("nlp_scaling: no converged factor (call solve() first)")
        })?;
        let (df, dc, dd) = s.inner.nlp_scaling().map_err(solver_error_to_py)?;
        let dx = s.inner.variable_scaling().map_err(solver_error_to_py)?;
        let out = PyDict::new_bound(py);
        out.set_item("obj", df)?;
        out.set_item("c_scale", crate::problem::opt_vec_to_py(py, dc))?;
        out.set_item("d_scale", crate::problem::opt_vec_to_py(py, dd))?;
        out.set_item("x_scaling", crate::problem::opt_vec_to_py(py, dx))?;
        Ok(out)
    }

    /// Inertia-correction perturbations `(delta_x, delta_s, delta_c,
    /// delta_d)` baked into the held KKT factor, as a length-4
    /// ndarray. All zero means the final factorization was
    /// unregularized and the natural-units back-solves (covariance in
    /// particular) invert the exact KKT matrix; nonzero means the
    /// factor carries a regularization and `-inv(reduced_hessian)` is
    /// perturbed (pounce#128 follow-up).
    #[getter]
    fn kkt_perturbations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("kkt_perturbations: no converged factor (call solve() first)")
        })?;
        let p = s.inner.kkt_perturbations().map_err(solver_error_to_py)?;
        Ok(p.to_vec().into_pyarray_bound(py))
    }

    /// Whether a `max_pdpert` of `limit` turns this factor down, and
    /// the worst correction it read, as `(refuse, worst)`.
    ///
    /// The comparison is `pounce_sensitivity::pdpert_verdict`, which
    /// the CLI's `sens_max_pdpert` reads too, so the two surfaces
    /// cannot drift apart on what counts as too perturbed. The caller
    /// words its own message from `worst`, since the option's message
    /// names an option and says the sensitivity was skipped, and
    /// neither is true of a call that raises.
    ///
    /// `kkt_perturbations` reports the same four numbers for a caller
    /// who would rather read them than stop on them.
    fn pdpert_verdict(&self, limit: Number) -> PyResult<(bool, Number)> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("pdpert_verdict: no converged factor (call solve() first)")
        })?;
        let p = s.inner.kkt_perturbations().map_err(solver_error_to_py)?;
        Ok(pounce_sensitivity::pdpert_verdict(&p, limit))
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

    /// The `bound_relax_factor` the held solve actually ran under, or
    /// `None` with no converged factor. Non-zero means a variable may
    /// have settled outside the bound the model declares.
    ///
    /// This is the value `classify_activity` guards on, readable
    /// without provoking the guard. A caller that wants to report a
    /// relaxed solve rather than fail on one would otherwise have to
    /// call `classify_activity`, catch its error, and match the option
    /// name in the message, which silently becomes a re-raise the day
    /// that message is reworded.
    ///
    /// Reads the solve's own value, not the live option: relaxing (or
    /// not) happened once, and setting the option afterwards
    /// re-measures nothing.
    #[getter]
    fn bound_relax_factor(&self) -> Option<Number> {
        self.state
            .as_ref()
            .and_then(|s| s.inner.converged().map(|c| c.bound_relax_factor))
    }

    /// Classify every bounded variable and every finite-bounded
    /// inequality row of the held solve by activity, from the ratio of
    /// barrier curvature to the model's own curvature (the exact
    /// Lagrangian Hessian, so constraint curvature contributes) at the
    /// converged iterate (covariance-information-roadmap item 0,
    /// pounce#362).
    ///
    /// Indexing is **user space**: `var_*` entries follow your `n`
    /// variables and `row_*` entries your `m` constraints, in your
    /// order. A variable fixed by its bounds (`lb == ub`) reports
    /// `"fixed"` (the solve removes it and classifies no barrier
    /// geometry for it), and an equality constraint reports
    /// `"equality"`, so indices never shift.
    ///
    /// Returns a dict:
    ///
    /// - `"mu"`: the converged barrier parameter.
    /// - `"var_status"`, `"row_status"`: list of str per variable /
    ///   constraint: `"inactive"`, `"weakly_active"`,
    ///   `"strongly_active"`, `"ambiguous"`, `"unidentified"`,
    ///   `"unbounded"` where there is no finite bound, plus the
    ///   `"fixed"` / `"equality"` placeholders above.
    /// - `"var_ratio"`, `"row_ratio"`: ndarray of the ratio `Σ/q`
    ///   (NaN where nothing was classified). An `"unidentified"` entry
    ///   holds `Σ/floor`, a lower bound on any honest ratio rather
    ///   than the ratio itself. Rows classify on the scale-invariant
    ///   form `Σ‖∇d‖⁴/|∇dᵀH∇d|`, so rescaling a constraint row
    ///   does not change its status.
    /// - `"var_q_sign"`, `"row_q_sign"`: ndarray of the sign of the
    ///   signed curvature, so an indefinite direction is visible.
    /// - `"var_off_central_path"`, `"row_off_central_path"`: list of
    ///   bool, true where `s·z` differs from `μ` by more than 10× on
    ///   some side.
    /// - `"var_contaminated"`, `"row_contaminated"`: list of bool,
    ///   true where classified inactive yet carrying non-negligible
    ///   barrier curvature.
    /// - `"var_sigma"`, `"row_sigma"`: ndarray of the barrier
    ///   diagonal `Σ` itself (both sides summed; 0 where nothing was
    ///   classified), the quantity item 1 of the covariance roadmap
    ///   subtracts from the factor's reduced Hessian. In **natural
    ///   (unscaled) units** like every other sensitivity output:
    ///   classification runs on the solver's scaled quantities (the
    ///   ratios above are scale-invariant), and the exported `Σ` is
    ///   unscaled at the boundary, so it composes directly with a
    ///   natural-units reduced Hessian. `row_sigma` is additionally
    ///   the RAW diagonal, not the geometric weight `Σ‖∇d‖²` the
    ///   classification uses, so a consumer applies whichever `‖a‖²`
    ///   its own restriction of the normal calls for.
    ///
    /// Requires the **held solve** to have run with
    /// `bound_relax_factor=0` (raises `ValueError` otherwise; the
    /// Ipopt default is `1e-8`): relaxed bounds shift the slacks the
    /// classifier reads. The guard reads the value that solve ran
    /// under, so setting the option afterwards neither unlocks a
    /// relaxed solve nor invalidates an unrelaxed one — set it on the
    /// `Problem` and solve again.
    fn classify_activity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("classify_activity: no converged factor (call solve() first)")
        })?;
        let rep = s.inner.classify_activity().map_err(solver_error_to_py)?;
        let status_str = |codes: &[i8]| -> Vec<&'static str> {
            codes
                .iter()
                .map(|&c| match c {
                    activity::UNBOUNDED => "unbounded",
                    activity::INACTIVE => "inactive",
                    activity::WEAKLY_ACTIVE => "weakly_active",
                    activity::STRONGLY_ACTIVE => "strongly_active",
                    activity::AMBIGUOUS => "ambiguous",
                    activity::UNIDENTIFIED => "unidentified",
                    activity::FIXED => "fixed",
                    activity::EQUALITY => "equality",
                    _ => unreachable!("unknown activity status code {c}"),
                })
                .collect()
        };
        let out = PyDict::new_bound(py);
        out.set_item("mu", rep.mu)?;
        out.set_item("var_status", status_str(&rep.var_status))?;
        out.set_item("var_ratio", rep.var_ratio.into_pyarray_bound(py))?;
        out.set_item(
            "var_q_sign",
            rep.var_q_sign
                .iter()
                .map(|&s| s as i64)
                .collect::<Vec<_>>()
                .into_pyarray_bound(py),
        )?;
        out.set_item("var_off_central_path", rep.var_off_central_path)?;
        out.set_item("var_contaminated", rep.var_contaminated)?;
        out.set_item("var_sigma", rep.var_sigma.into_pyarray_bound(py))?;
        out.set_item("row_status", status_str(&rep.row_status))?;
        out.set_item("row_ratio", rep.row_ratio.into_pyarray_bound(py))?;
        out.set_item(
            "row_q_sign",
            rep.row_q_sign
                .iter()
                .map(|&s| s as i64)
                .collect::<Vec<_>>()
                .into_pyarray_bound(py),
        )?;
        out.set_item("row_off_central_path", rep.row_off_central_path)?;
        out.set_item("row_contaminated", rep.row_contaminated)?;
        out.set_item("row_sigma", rep.row_sigma.into_pyarray_bound(py))?;
        Ok(out)
    }

    /// The exact Lagrangian Hessian times a user-space vector, as an
    /// ndarray in user variable order and natural units (the internal
    /// Hessian's objective scale is divided out, matching the
    /// sensitivity-output contract). Entries for fixed (`lb == ub`)
    /// variables are 0 in and out. One product per call; the
    /// covariance roadmap's item 2 builds the tangent-recovered
    /// reduced Hessian from these.
    fn hessian_vec<'py>(
        &self,
        py: Python<'py>,
        v: Vec<Number>,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("hessian_vec: no converged factor (call solve() first)")
        })?;
        let out = s.inner.hessian_vec(&v).map_err(solver_error_to_py)?;
        Ok(out.into_pyarray_bound(py))
    }

    /// The gradient of user constraint row `j` at the converged
    /// iterate, as an ndarray in user variable order (length n), in
    /// **natural (unscaled) units**: the solver's internal Jacobian
    /// row carries its per-row `d_scale`/`c_scale`, which is divided
    /// out here, so this is the gradient of the row the user wrote.
    /// Equality and inequality rows alike; entries for fixed
    /// (`lb == ub`) variables are 0 because the solve removed their
    /// columns. A binding row's normal restricted to the fitted
    /// columns is the projection direction of the covariance
    /// roadmap's item 1.
    fn row_normal<'py>(&self, py: Python<'py>, j: i64) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let s = self.state.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("row_normal: no converged factor (call solve() first)")
        })?;
        if j < 0 || (j as usize) >= s.m {
            return Err(PyValueError::new_err(format!(
                "row_normal: constraint index {j} out of range [0, m={})",
                s.m
            )));
        }
        let g = s.inner.row_normal(j as usize).map_err(solver_error_to_py)?;
        Ok(g.into_pyarray_bound(py))
    }
}

fn validate_var_rows(held_var_rows: &[i64], n_x: usize) -> PyResult<Vec<Index>> {
    held_var_rows
        .iter()
        .map(|&r| {
            if r < 0 || (r as usize) >= n_x {
                Err(PyValueError::new_err(format!(
                    "held_var_rows[..] = {r} out of range [0, n_x={n_x})",
                )))
            } else {
                Ok(r as Index)
            }
        })
        .collect()
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
        SolverError::BadOptions(msg) => PyValueError::new_err(format!("Solver: {msg}")),
        // SolverError is #[non_exhaustive]
        other => PyRuntimeError::new_err(format!("Solver: {other:?}")),
    }
}
