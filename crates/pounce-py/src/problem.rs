//! `Problem` PyO3 class — the cyipopt-compatible user-facing handle.
//!
//! Construction mirrors cyipopt: pass dims, bounds, and a `problem_obj`
//! with `objective`/`gradient`/`constraints`/`jacobian`/... methods.
//! Options are set with `add_option(name, value)`. `solve(x0)` returns
//! `(x_opt, info)` where `info` is a dict (status, status_msg, obj_val,
//! mult_g, mult_x_L, mult_x_U, iter_count, ...).

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pounce_algorithm::alg_builder::{LinearBackendFactory, LinearSolverChoice};
use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_linsol::sparse_sym_iface::SparseSymLinearSolverInterface;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::TNLP;
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    make_default_restoration_factory_provider, InnerBackendFactoryFactory,
};
use pounce_sensitivity::SensSolve;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::cell::RefCell;
use std::rc::Rc;

use crate::tnlp_bridge::{call0, decode_structure, PyTnlp, PyTnlpInit};

/// One pounce problem instance. Holds the user object and bound
/// vectors; the underlying `IpoptApplication` is rebuilt per `solve()`
/// so options changes always take effect and warm-start state is owned
/// by the call site.
#[pyclass(name = "Problem", module = "pounce._pounce")]
pub struct PyProblem {
    n: Index,
    m: Index,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    g_l: Vec<Number>,
    g_u: Vec<Number>,
    /// Held until `solve` mints the bridge state.
    problem_obj: Py<PyAny>,
    /// Pending options, applied to the freshly-built application at
    /// every `solve()`. We store them as a sequence of typed setters
    /// because `OptionsList` distinguishes string / number / integer.
    str_opts: Vec<(String, String)>,
    num_opts: Vec<(String, Number)>,
    int_opts: Vec<(String, Index)>,
    /// Detected once on construction by probing whether the user object
    /// has `hessian` + `hessianstructure`. If absent, the solver runs
    /// with `hessian_approximation = limited-memory`.
    has_hessian: bool,
    /// Phase 5c §7.3 — SQP warm-start working set queued for the
    /// next solve. Set via `solve(..., working_set=…)`; consumed
    /// once and cleared after the solve completes. The IPM path
    /// ignores this.
    pending_working_set: Option<pounce_qp::WorkingSet>,
    /// Phase 5c §7.3 — most recent SQP working set, written by
    /// `solve` when the SQP path produces one. Retrieved via
    /// `get_working_set()`.
    last_working_set: Option<pounce_qp::WorkingSet>,
    /// User-supplied scaling installed via `set_problem_scaling`.
    /// Forwarded to `PyTnlpInit` on `prepare`, and from there into
    /// `TNLP::get_scaling_parameters`. Only consulted by the IPM when
    /// `nlp_scaling_method=user-scaling`.
    user_scaling: Option<UserScaling>,
}

/// Per-problem user scaling vector, mirroring `SetIpoptProblemScaling`
/// in the C interface. `x_scaling` / `g_scaling` are `None` when the
/// user wants that axis left unscaled.
#[derive(Clone)]
pub(crate) struct UserScaling {
    pub(crate) obj: Number,
    pub(crate) x_scaling: Option<Vec<Number>>,
    pub(crate) g_scaling: Option<Vec<Number>>,
}

#[pymethods]
impl PyProblem {
    #[new]
    #[pyo3(signature = (n, m, problem_obj, lb=None, ub=None, cl=None, cu=None))]
    fn new(
        n: i64,
        m: i64,
        problem_obj: Py<PyAny>,
        lb: Option<Py<PyAny>>,
        ub: Option<Py<PyAny>>,
        cl: Option<Py<PyAny>>,
        cu: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if n < 0 {
            return Err(PyValueError::new_err("n must be non-negative"));
        }
        if m < 0 {
            return Err(PyValueError::new_err("m must be non-negative"));
        }
        // PR #50 review S3: guard against silent i64→i32 truncation.
        if n > i32::MAX as i64 {
            return Err(PyValueError::new_err(format!(
                "n = {n} exceeds the solver's signed-32-bit Index range"
            )));
        }
        if m > i32::MAX as i64 {
            return Err(PyValueError::new_err(format!(
                "m = {m} exceeds the solver's signed-32-bit Index range"
            )));
        }
        let n_i = n as Index;
        let m_i = m as Index;
        let x_l = decode_bounds(lb, n_i as usize, f64::NEG_INFINITY)?;
        let x_u = decode_bounds(ub, n_i as usize, f64::INFINITY)?;
        if m_i > 0 && (cl.is_none() || cu.is_none()) {
            return Err(PyValueError::new_err(
                "cl and cu must be provided when m > 0",
            ));
        }
        let g_l = decode_bounds(cl, m_i as usize, f64::NEG_INFINITY)?;
        let g_u = decode_bounds(cu, m_i as usize, f64::INFINITY)?;
        let has_hessian = Python::with_gil(|py| {
            let bound = problem_obj.bind(py);
            bound.hasattr("hessian").unwrap_or(false)
                && bound.hasattr("hessianstructure").unwrap_or(false)
        });
        Ok(Self {
            n: n_i,
            m: m_i,
            x_l,
            x_u,
            g_l,
            g_u,
            problem_obj,
            str_opts: Vec::new(),
            num_opts: Vec::new(),
            int_opts: Vec::new(),
            has_hessian,
            pending_working_set: None,
            last_working_set: None,
            user_scaling: None,
        })
    }

    /// Set a solver option. Accepts `str`, `int`, or `float` for
    /// `value`; routed to the matching `OptionsList` setter.
    fn add_option(&mut self, name: &str, value: Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(s) = value.extract::<String>() {
            self.str_opts.push((name.to_string(), s));
            return Ok(());
        }
        // Order matters: in Python `bool` is a subclass of `int`, so
        // PyO3 will happily extract `True`/`False` as `1`/`0`. We want
        // cyipopt-style `True → "yes"`, so isinstance-check for `bool`
        // *before* falling through to int extraction.
        let is_bool = value.is_instance_of::<pyo3::types::PyBool>();
        if is_bool {
            if let Ok(b) = value.extract::<bool>() {
                self.str_opts
                    .push((name.to_string(), if b { "yes".into() } else { "no".into() }));
                return Ok(());
            }
        }
        if let Ok(i) = value.extract::<i64>() {
            self.int_opts.push((name.to_string(), i as Index));
            return Ok(());
        }
        if let Ok(f) = value.extract::<f64>() {
            self.num_opts.push((name.to_string(), f));
            return Ok(());
        }
        Err(PyValueError::new_err(format!(
            "add_option({name}): expected str / int / float / bool, got {}",
            value.get_type().name()?
        )))
    }

    /// cyipopt-compat camelCase alias.
    #[pyo3(name = "addOption")]
    #[allow(non_snake_case)]
    fn add_option_camel(&mut self, name: &str, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.add_option(name, value)
    }

    /// Solve the problem. Returns `(x, info_dict)`.
    ///
    /// The optional `working_set` kwarg (Phase 5c §7.3) accepts a
    /// 2-tuple `(bounds, constraints)` of numpy int arrays
    /// (length `n` and `m` respectively, status codes 0..=3).
    /// Consumed only by the SQP path (`algorithm = active-set-sqp`);
    /// the IPM ignores it. When provided, it overrides any value
    /// previously stashed via `set_working_set`. After every SQP
    /// solve `info_dict["working_set"]` holds the final working
    /// set, and `get_working_set()` returns the same tuple.
    #[pyo3(signature = (x0, lagrange=None, zl=None, zu=None, working_set=None))]
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        x0: Py<PyAny>,
        lagrange: Option<Py<PyAny>>,
        zl: Option<Py<PyAny>>,
        zu: Option<Py<PyAny>>,
        working_set: Option<Py<PyAny>>,
    ) -> PyResult<(Bound<'py, PyArray1<Number>>, Bound<'py, PyDict>)> {
        let (mut app, bridge) = self.prepare(py, x0, lagrange, zl, zu)?;
        // Per-call working_set overrides any pending one set via
        // `set_working_set`. Either path lands as
        // `IpoptApplication::set_sqp_warm_start`.
        let ws_for_solve = match working_set {
            Some(obj) => Some(decode_working_set(
                py,
                &obj,
                self.n as usize,
                self.m as usize,
            )?),
            None => self.pending_working_set.take(),
        };
        if let Some(ws) = ws_for_solve {
            // PR #50 review A3: warn if the caller supplied a
            // working set but the algorithm wasn't switched to the
            // SQP path. The IPM silently ignores `set_sqp_warm_start`,
            // so users could otherwise lose their warm-start data
            // without any hint that something was misconfigured.
            let sqp_selected = self
                .str_opts
                .iter()
                .any(|(k, v)| k == "algorithm" && v.eq_ignore_ascii_case("active-set-sqp"));
            if !sqp_selected {
                let warnings = py.import_bound("warnings")?;
                let _ = warnings.call_method1(
                    "warn",
                    ("working_set was supplied but `algorithm` is not \
                         \"active-set-sqp\"; the IPM path ignores working sets. \
                         Either call add_option(\"algorithm\", \"active-set-sqp\") \
                         before solve(), or drop the working_set argument.",),
                );
            }
            // Seed the warm-start payload from the same x0 / dual
            // inputs the bridge already received. Falling back to
            // all-zeros here (the previous behavior) silently ignored
            // the caller's `x0=` argument and started the SQP from
            // x=0, which on bound-constrained or JAX-built NLPs is
            // typically far outside the feasible region and produces
            // a degenerate KKT at iter 0. See gh#57.
            let bridge_ref = bridge.borrow();
            let x_warm = bridge_ref.state.x0.clone();
            let lambda_g_warm = bridge_ref
                .state
                .lam0
                .clone()
                .unwrap_or_else(|| vec![0.0; self.m as usize]);
            let zl_warm = bridge_ref.state.z_l0.as_deref();
            let zu_warm = bridge_ref.state.z_u0.as_deref();
            // SQP's `lambda_x` follows IPOPT's user-facing
            // convention `lambda_x = z_l − z_u`. When the caller
            // supplies neither, leave it at zero.
            let lambda_x_warm = match (zl_warm, zu_warm) {
                (Some(zl), Some(zu)) => zl.iter().zip(zu).map(|(l, u)| l - u).collect(),
                (Some(zl), None) => zl.to_vec(),
                (None, Some(zu)) => zu.iter().map(|u| -u).collect(),
                (None, None) => vec![0.0; self.n as usize],
            };
            drop(bridge_ref);
            app.set_sqp_warm_start(pounce_algorithm::sqp::SqpIterates {
                x: x_warm,
                lambda_g: lambda_g_warm,
                lambda_x: lambda_x_warm,
                working: Some(ws),
            });
        }
        // Release the GIL across `optimize_tnlp` so independent
        // `Problem` instances on different OS threads can run their
        // IPM iterations in parallel. Every TNLP callback (`eval_f`,
        // `eval_grad_f`, `eval_g`, `eval_jac_g`, `eval_h`,
        // `intermediate_callback`) in `tnlp_bridge.rs` already takes
        // its own `Python::with_gil(...)` before touching Python
        // state, so re-acquiring the GIL inside the call is safe
        // and serialized by Python the usual way.
        //
        // SAFETY: `app` and `bridge` carry `Rc<RefCell<…>>` (because
        // `pounce_nlp` uses single-threaded refcounting throughout).
        // PyO3's `allow_threads` requires `Send`, so we wrap both
        // moves in a transparent `SendGuard`. The closure does *not*
        // actually cross OS threads — `Python::allow_threads` runs
        // its body on the calling thread after `PyEval_SaveThread`,
        // so the `Rc` refcount and `RefCell` borrow flag are only
        // ever touched by this one thread (no concurrent access, no
        // happens-before issue with the eventual `Drop`).
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
        // Method-call captures (vs. field-access `.0`) defeat the
        // 2021-edition disjoint-capture rule, so the closure captures
        // the whole `SendGuard<T>` (which is `Send` by our `unsafe
        // impl`) rather than peeking at the inner `Rc` directly.
        let app_guard = SendGuard::new(app);
        let bridge_guard = SendGuard::new(bridge);
        let (status, app_back, bridge_back): (
            ApplicationReturnStatus,
            SendGuard<IpoptApplication>,
            SendGuard<Rc<RefCell<PyTnlp>>>,
        ) = py.allow_threads(move || {
            let mut app = app_guard.into_inner();
            let bridge = bridge_guard.into_inner();
            let bridge_for_solve: Rc<RefCell<dyn TNLP>> = bridge.clone();
            let status = app.optimize_tnlp(bridge_for_solve);
            (status, SendGuard::new(app), SendGuard::new(bridge))
        });
        let app = app_back.into_inner();
        let bridge = bridge_back.into_inner();
        let stats = app.statistics();
        // Pick up any working set the SQP path produced; surface
        // it in the info dict and stash on the Problem instance.
        self.last_working_set = app.last_sqp_working_set().cloned();
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
        let ws_obj: PyObject = match &self.last_working_set {
            Some(ws) => encode_working_set(py, ws).into_any().unbind(),
            None => py.None(),
        };
        info.set_item("working_set", ws_obj)?;
        let x_out = bridge.borrow().state.final_x.clone().into_pyarray_bound(py);
        Ok((x_out, info))
    }

    /// Stash a warm-start working set consumed by the next
    /// `solve()` call. Equivalent to passing `working_set=` to
    /// `solve()`, but useful when configuring the problem ahead
    /// of time. Only consulted by the SQP path.
    fn set_working_set(&mut self, py: Python<'_>, working_set: Py<PyAny>) -> PyResult<()> {
        let ws = decode_working_set(py, &working_set, self.n as usize, self.m as usize)?;
        self.pending_working_set = Some(ws);
        Ok(())
    }

    /// Drop any pending warm-start working set without solving.
    fn clear_working_set(&mut self) {
        self.pending_working_set = None;
    }

    /// Return the most recent SQP working set as a
    /// `(bounds, constraints)` tuple of numpy int8 arrays, or
    /// `None` when no SQP solve has run.
    fn get_working_set<'py>(&self, py: Python<'py>) -> Option<Bound<'py, pyo3::types::PyTuple>> {
        self.last_working_set
            .as_ref()
            .map(|ws| encode_working_set(py, ws))
    }

    /// Install user-supplied NLP scaling. Mirrors
    /// `SetIpoptProblemScaling` in the C interface.
    ///
    /// * `obj_scaling` — multiplier applied to the objective (and the
    ///   final reported value is divided back out).
    /// * `x_scaling` — length-`n` per-variable factors, or `None` to
    ///   leave variable scaling off. (Note: the algorithm currently
    ///   accepts this channel but does not yet act on it; only
    ///   `obj_scaling` and `g_scaling` affect the IPM. See
    ///   `docs/src/scaling.md`.)
    /// * `g_scaling` — length-`m` per-constraint factors, or `None`
    ///   to leave constraint scaling off.
    ///
    /// The scaling only takes effect when `nlp_scaling_method` is set
    /// to `"user-scaling"`. Call once before `solve()`; cleared by
    /// `clear_problem_scaling()`.
    #[pyo3(signature = (obj_scaling, x_scaling=None, g_scaling=None))]
    fn set_problem_scaling(
        &mut self,
        obj_scaling: Number,
        x_scaling: Option<Py<PyAny>>,
        g_scaling: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let x = x_scaling
            .map(|v| extract_f64_vec(&v, self.n as usize, "x_scaling"))
            .transpose()?;
        let g = g_scaling
            .map(|v| extract_f64_vec(&v, self.m as usize, "g_scaling"))
            .transpose()?;
        self.user_scaling = Some(UserScaling {
            obj: obj_scaling,
            x_scaling: x,
            g_scaling: g,
        });
        Ok(())
    }

    /// Drop any installed user scaling. The next `solve()` will rely
    /// on the active `nlp_scaling_method` (the default `"gradient-based"`
    /// computes scales from the starting-point gradients).
    fn clear_problem_scaling(&mut self) {
        self.user_scaling = None;
    }

    /// Solve, then run a parametric sensitivity step at the converged
    /// iterate. Returns `(x, info_dict)`; `info_dict` includes the
    /// extra keys `dx`, `dx_full`, `reduced_hessian`,
    /// `reduced_hessian_scaled`, `obj_scaling_factor`, `pin_g_scaling`,
    /// `kkt_perturbations` (the inertia-correction `(δ_x, δ_s, δ_c,
    /// δ_d)` baked into the converged factor — all zero means the
    /// factor is unregularized and the covariance reading is exact),
    /// `reduced_hessian_eigenvalues`, and `reduced_hessian_eigenvectors`
    /// (each may be `None` when the corresponding output was not
    /// requested or the solve did not converge).
    ///
    /// `reduced_hessian` is in **natural (unscaled) units**: any NLP
    /// scaling the IPM applied (`nlp_scaling_method`, default
    /// `"gradient-based"`) is undone, so `-inv(reduced_hessian)` is
    /// directly the parameter covariance of an estimation problem,
    /// independent of problem scaling and discretization (pounce#128).
    /// `reduced_hessian_scaled` is the value as the solver's internal
    /// scaled space sees it (what pounce returned before #128), and
    /// `obj_scaling_factor` / `pin_g_scaling` are the factors relating
    /// the two: `H_scaled[i,j] = obj_scaling_factor /
    /// (pin_g_scaling[i]*pin_g_scaling[j]) * H[i,j]`. `obj_scal`
    /// survives as a plain extra multiplier on both (default 1.0).
    ///
    /// `pin_constraint_indices` are 0-based indices into `g(x)`
    /// identifying the parameter-pin equalities `g_i(x) = p_i`. The
    /// caller must have declared these as exact equalities in the
    /// `Problem` constructor (`cl[i] == cu[i] == p_i`).
    ///
    /// Passing `rh_eigendecomp=True` implies `compute_reduced_hessian=True`
    /// and additionally returns the ascending eigenvalues plus the
    /// column-major eigenvector matrix of `H_R` (mirrors upstream
    /// sIPOPT's `rh_eigendecomp` option).
    ///
    /// Passing `sens_boundcheck=True` clamps the perturbed primal step
    /// against the variable bounds (single-pass projection — simpler
    /// than upstream's iterative Schur refinement; see
    /// `pounce_sensitivity::boundcheck`). `sens_bound_eps` is the
    /// tolerance (default `1e-9`).
    #[pyo3(signature = (
        x0,
        pin_constraint_indices,
        deltas = None,
        compute_reduced_hessian = false,
        rh_eigendecomp = false,
        sens_boundcheck = false,
        sens_bound_eps = 1e-9,
        obj_scal = 1.0,
        lagrange = None,
        zl = None,
        zu = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn solve_with_sens<'py>(
        &mut self,
        py: Python<'py>,
        x0: Py<PyAny>,
        pin_constraint_indices: Vec<i64>,
        deltas: Option<Vec<Number>>,
        compute_reduced_hessian: bool,
        rh_eigendecomp: bool,
        sens_boundcheck: bool,
        sens_bound_eps: Number,
        obj_scal: Number,
        lagrange: Option<Py<PyAny>>,
        zl: Option<Py<PyAny>>,
        zu: Option<Py<PyAny>>,
    ) -> PyResult<(Bound<'py, PyArray1<Number>>, Bound<'py, PyDict>)> {
        let compute_reduced_hessian = compute_reduced_hessian || rh_eigendecomp;
        let m = self.m as usize;
        let pins: Vec<Index> = pin_constraint_indices
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
            .collect::<PyResult<_>>()?;
        if let Some(d) = &deltas {
            if d.len() != pins.len() {
                return Err(PyValueError::new_err(format!(
                    "deltas length {} must equal pin_constraint_indices length {}",
                    d.len(),
                    pins.len(),
                )));
            }
        }
        if !compute_reduced_hessian && deltas.is_none() {
            return Err(PyValueError::new_err(
                "solve_with_sens: pass deltas=..., compute_reduced_hessian=True, or both",
            ));
        }

        let (mut app, bridge) = self.prepare(py, x0, lagrange, zl, zu)?;
        let bridge_for_solve: Rc<RefCell<dyn TNLP>> = bridge.clone();

        let mut builder = SensSolve::new(pins).with_obj_scal(obj_scal);
        if let Some(d) = deltas {
            builder = builder.with_deltas(d);
        }
        if rh_eigendecomp {
            builder = builder.with_reduced_hessian_eigen();
        } else if compute_reduced_hessian {
            builder = builder.with_reduced_hessian();
        }
        if sens_boundcheck {
            builder = builder.with_boundcheck(sens_bound_eps);
        }
        let result = builder.run(&mut app, bridge_for_solve);
        let stats = app.statistics();

        let info = build_info_dict(
            py,
            &bridge.borrow(),
            result.status,
            stats.iteration_count,
            stats.final_mu,
            stats.final_kkt_error,
            stats.final_dual_inf,
            stats.final_constr_viol,
            stats.final_compl,
        )?;
        info.set_item("dx", opt_vec_to_py(py, result.dx))?;
        info.set_item("dx_full", opt_vec_to_py(py, result.dx_full))?;
        info.set_item("reduced_hessian", opt_vec_to_py(py, result.reduced_hessian))?;
        info.set_item(
            "reduced_hessian_scaled",
            opt_vec_to_py(py, result.reduced_hessian_scaled),
        )?;
        let obj_scaling_obj: PyObject = match result.obj_scaling_factor {
            Some(v) => v.into_py(py),
            None => py.None(),
        };
        info.set_item("obj_scaling_factor", obj_scaling_obj)?;
        info.set_item("pin_g_scaling", opt_vec_to_py(py, result.pin_g_scaling))?;
        info.set_item(
            "kkt_perturbations",
            opt_vec_to_py(py, result.kkt_perturbations.map(|p| p.to_vec())),
        )?;
        info.set_item(
            "reduced_hessian_eigenvalues",
            opt_vec_to_py(py, result.reduced_hessian_eigenvalues),
        )?;
        info.set_item(
            "reduced_hessian_eigenvectors",
            opt_vec_to_py(py, result.reduced_hessian_eigenvectors),
        )?;
        // Surface a post-solve sensitivity-stage failure (M6). The
        // underlying solve can converge (so `status` is success) while
        // the sensitivity step still fails, leaving every `dx`/Hessian
        // output `None`. Without this key, that is indistinguishable
        // from "sensitivity not requested". `Option<String>` maps
        // `None` ⇒ Python `None`, `Some(msg)` ⇒ the error string.
        info.set_item("sens_error", result.error.clone())?;

        let x_out = bridge.borrow().state.final_x.clone().into_pyarray_bound(py);
        Ok((x_out, info))
    }

    #[getter]
    fn n(&self) -> i64 {
        self.n as i64
    }
    #[getter]
    fn m(&self) -> i64 {
        self.m as i64
    }
    #[getter]
    fn has_hessian(&self) -> bool {
        self.has_hessian
    }
}

/// `Some(vec)` → 1-D float ndarray, `None` → Python `None`. Shared by
/// the optional-output info-dict keys here and in the sibling
/// `Solver` pyclass.
pub(crate) fn opt_vec_to_py(py: Python<'_>, v: Option<Vec<Number>>) -> PyObject {
    match v {
        Some(v) => v.into_pyarray_bound(py).into_any().unbind(),
        None => py.None(),
    }
}

impl PyProblem {
    /// Number of constraints (m). Internal accessor for sibling Solver pyclass.
    pub(crate) fn m_index(&self) -> Index {
        self.m
    }

    /// Shared setup for [`Self::solve`] / [`Self::solve_with_sens`]:
    /// decode warm-start vectors, materialize Jac/Hess sparsity, build
    /// and configure the application (options + restoration), and mint
    /// the Py↔Rust TNLP bridge. Returns the application ready for
    /// `optimize_tnlp` and the bridge whose `final_*` fields the
    /// callback writes into.
    pub(crate) fn prepare(
        &self,
        py: Python<'_>,
        x0: Py<PyAny>,
        lagrange: Option<Py<PyAny>>,
        zl: Option<Py<PyAny>>,
        zu: Option<Py<PyAny>>,
    ) -> PyResult<(IpoptApplication, Rc<RefCell<PyTnlp>>)> {
        let n = self.n as usize;
        let m = self.m as usize;
        let x0_vec = extract_f64_vec(&x0, n, "x0")?;
        let lam0 = lagrange
            .map(|v| extract_f64_vec(&v, m, "lagrange"))
            .transpose()?;
        let z_l0 = zl.map(|v| extract_f64_vec(&v, n, "zl")).transpose()?;
        let z_u0 = zu.map(|v| extract_f64_vec(&v, n, "zu")).transpose()?;
        let init = self.build_tnlp_init(py, x0_vec, lam0, z_l0, z_u0)?;

        let mut app = IpoptApplication::new();
        if !self.has_hessian {
            let _ = app.options_mut().set_string_value(
                "hessian_approximation",
                "limited-memory",
                true,
                false,
            );
        }
        for (k, v) in &self.str_opts {
            app.options_mut()
                .set_string_value(k, v, true, false)
                .map_err(|e| PyRuntimeError::new_err(format!("option {k}={v}: {e}")))?;
        }
        for (k, v) in &self.num_opts {
            app.options_mut()
                .set_numeric_value(k, *v, true, false)
                .map_err(|e| PyRuntimeError::new_err(format!("option {k}={v}: {e}")))?;
        }
        for (k, v) in &self.int_opts {
            app.options_mut()
                .set_integer_value(k, *v, true, false)
                .map_err(|e| PyRuntimeError::new_err(format!("option {k}={v}: {e}")))?;
        }
        app.initialize()
            .map_err(|e| PyRuntimeError::new_err(format!("initialize: {e}")))?;

        let feral_cfg = pounce_algorithm::application::feral_config_from_options(app.options());
        let bff_mint = move || -> InnerBackendFactoryFactory {
            let feral_cfg = feral_cfg.clone();
            Box::new(move || default_backend_factory(feral_cfg.clone()))
        };
        let resto_provider = make_default_restoration_factory_provider(
            RestoAlgorithmBuilder::new(),
            app.algorithm_builder_from_options(),
            bff_mint,
        );
        app.set_restoration_factory_provider(resto_provider);

        let bridge = Rc::new(RefCell::new(PyTnlp::new(init)));
        Ok((app, bridge))
    }

    /// Assemble the [`PyTnlpInit`] payload for one solve: resolve the
    /// Jacobian / Hessian sparsity through the Python object (once, so
    /// the solver's `Structure` calls are GIL-free copies) and capture
    /// bounds, starting point, optional warm duals, and the callback
    /// handle. Factored out of [`Self::prepare`] so the batch path
    /// (pounce#126 phase 2) can mint per-instance bridges without
    /// building the application on this thread — the resulting
    /// `PyTnlpInit` is plain data + `Py<PyAny>` handles, hence `Send`,
    /// and moves to the rayon worker that owns the solve.
    pub(crate) fn build_tnlp_init(
        &self,
        py: Python<'_>,
        x0_vec: Vec<Number>,
        lam0: Option<Vec<Number>>,
        z_l0: Option<Vec<Number>>,
        z_u0: Option<Vec<Number>>,
    ) -> PyResult<PyTnlpInit> {
        let n = self.n as usize;
        let m = self.m as usize;
        let (jac_rows, jac_cols, nele_jac) = if m > 0 {
            let s = call0(&self.problem_obj, "jacobianstructure")?;
            let (rows, cols) = decode_structure_inferred(&s)?;
            (rows.clone(), cols.clone(), rows.len() as Index)
        } else {
            (Vec::new(), Vec::new(), 0)
        };

        // Hessian sparsity. When the user provides one, use it
        // verbatim. Without one we still need a non-empty pattern: the
        // L-BFGS updater pins its work-space sparsity from
        // `curr_exact_hessian()`, so an empty space means nowhere for
        // the quasi-Newton approximation to land. Declare the dense
        // lower triangle — `eval_h(Values)` returns zeros and the
        // updater overwrites them with its rank-update approximation.
        let (hess_rows, hess_cols, nele_hess) = if self.has_hessian {
            let s = call0(&self.problem_obj, "hessianstructure")?;
            let (rows, cols) = decode_structure_inferred(&s)?;
            (rows.clone(), cols.clone(), rows.len() as Index)
        } else {
            let mut rows = Vec::with_capacity(n * (n + 1) / 2);
            let mut cols = Vec::with_capacity(n * (n + 1) / 2);
            for i in 0..n {
                for j in 0..=i {
                    rows.push(i as Index);
                    cols.push(j as Index);
                }
            }
            let nele = rows.len() as Index;
            (rows, cols, nele)
        };

        Ok(PyTnlpInit {
            n: self.n,
            m: self.m,
            nele_jac,
            nele_hess,
            x_l: self.x_l.clone(),
            x_u: self.x_u.clone(),
            g_l: self.g_l.clone(),
            g_u: self.g_u.clone(),
            x0: x0_vec,
            lam0,
            z_l0,
            z_u0,
            py_obj: self.problem_obj.clone_ref(py),
            jac_rows,
            jac_cols,
            hess_rows,
            hess_cols,
            has_hessian: self.has_hessian,
            user_scaling: self.user_scaling.clone(),
            final_x: vec![0.0; n],
            final_z_l: vec![0.0; n],
            final_z_u: vec![0.0; n],
            final_g: vec![0.0; m],
            final_lambda: vec![0.0; m],
            final_obj: 0.0,
            final_status_code: 0,
        })
    }

    /// The pending option sets, in `OptionsList`'s three value classes
    /// — for the batch path, which applies them to a fresh per-worker
    /// application instead of going through [`Self::prepare`].
    pub(crate) fn option_sets(
        &self,
    ) -> (&[(String, String)], &[(String, Number)], &[(String, Index)]) {
        (&self.str_opts, &self.num_opts, &self.int_opts)
    }

    pub(crate) fn uses_exact_hessian(&self) -> bool {
        self.has_hessian
    }

    pub(crate) fn dims(&self) -> (usize, usize) {
        (self.n as usize, self.m as usize)
    }

    /// Per-constraint equality mask (`g_l[i] == g_u[i]`), used by the batch
    /// info-dict builder to reproduce the single-solve `active_constraints`
    /// classification (equalities are always active).
    pub(crate) fn equality_mask(&self) -> Vec<bool> {
        self.g_l
            .iter()
            .zip(&self.g_u)
            .map(|(l, u)| l == u)
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_info_dict<'py>(
    py: Python<'py>,
    bridge: &PyTnlp,
    status: ApplicationReturnStatus,
    iter_count: i32,
    final_mu: Number,
    final_kkt_error: Number,
    final_dual_inf: Number,
    final_constr_viol: Number,
    final_compl: Number,
) -> PyResult<Bound<'py, PyDict>> {
    let info = PyDict::new_bound(py);
    info.set_item("status", status as i32)?;
    info.set_item("status_msg", status_message(status))?;
    info.set_item("obj_val", bridge.state.final_obj)?;
    info.set_item("g", bridge.state.final_g.clone().into_pyarray_bound(py))?;
    info.set_item(
        "mult_g",
        bridge.state.final_lambda.clone().into_pyarray_bound(py),
    )?;
    info.set_item(
        "mult_x_L",
        bridge.state.final_z_l.clone().into_pyarray_bound(py),
    )?;
    info.set_item(
        "mult_x_U",
        bridge.state.final_z_u.clone().into_pyarray_bound(py),
    )?;
    info.set_item("iter_count", iter_count)?;
    // Converged barrier parameter μ. Thread this into the next
    // warm-started solve's `mu_init` / `warm_start_target_mu` to seed
    // the corrector in predictor–corrector path following (pounce#86).
    // `0.0` on the barrier-free SQP path.
    info.set_item("mu", final_mu)?;

    // Final convergence metrics (the values the IPM/SQP convergence check
    // saw at the last iterate). `final_kkt_error` is the overall NLP error
    // that `OptErrorConvCheck` tests against `tol` / `acceptable_tol`;
    // surfacing it lets the scipy-style facade judge `success` on the actual
    // optimality residual rather than solely on the exit-status enum, so a
    // verified optimum reached via a tiny-step exit (gh #119/#123) is still
    // reported as a success. NaN on a path that never computed them.
    info.set_item("final_kkt_error", final_kkt_error)?;
    info.set_item("final_dual_inf", final_dual_inf)?;
    info.set_item("final_constr_viol", final_constr_viol)?;
    info.set_item("final_compl", final_compl)?;

    // DiffHandoff active-set masks (dev-notes/diff-handoff-contract.md):
    // compute the active set ONCE here, in the producer, so the JAX /
    // torch backward passes read it instead of each re-deriving
    // `|mult| > tol` under its own tolerance. `pinned_vars[i]` = an
    // active variable bound (dx/dp = 0); `active_constraints[i]` = an
    // equality row or a binding inequality. We want only the two masks,
    // not a full `DiffHandoff`, so go through the shared `masks` helper on
    // the borrowed duals rather than cloning x / λ / z_L / z_U just to
    // discard them.
    let equality_mask: Vec<bool> = bridge
        .state
        .g_l
        .iter()
        .zip(bridge.state.g_u.iter())
        .map(|(l, u)| l == u)
        .collect();
    let (pinned_vars, active_constraints) = pounce_sensitivity::DiffHandoff::masks(
        &bridge.state.final_z_l,
        &bridge.state.final_z_u,
        &bridge.state.final_lambda,
        &equality_mask,
        pounce_sensitivity::DEFAULT_ACTIVE_TOL,
    );
    info.set_item("pinned_vars", pinned_vars.into_pyarray_bound(py))?;
    info.set_item(
        "active_constraints",
        active_constraints.into_pyarray_bound(py),
    )?;
    info.set_item("active_tol", pounce_sensitivity::DEFAULT_ACTIVE_TOL)?;
    Ok(info)
}

/// Variant of `decode_structure` that infers `nnz` from the input
/// instead of validating against a pre-computed count.
fn decode_structure_inferred(val: &Py<PyAny>) -> PyResult<(Vec<Index>, Vec<Index>)> {
    Python::with_gil(|py| {
        let bound = val.bind(py);
        let rows_obj: Py<PyAny>;
        let cols_obj: Py<PyAny>;
        if let Ok(list) = bound.downcast::<PyList>() {
            if list.len() != 2 {
                return Err(PyValueError::new_err(
                    "structure must be a (rows, cols) pair",
                ));
            }
            rows_obj = list.get_item(0)?.unbind();
            cols_obj = list.get_item(1)?.unbind();
        } else if let Ok(tup) = bound.downcast::<pyo3::types::PyTuple>() {
            if tup.len() != 2 {
                return Err(PyValueError::new_err(
                    "structure must be a (rows, cols) pair",
                ));
            }
            rows_obj = tup.get_item(0)?.unbind();
            cols_obj = tup.get_item(1)?.unbind();
        } else {
            return Err(PyValueError::new_err(
                "structure must be a tuple or list (rows, cols)",
            ));
        }
        let rows = extract_index_vec_inferred(&rows_obj, "structure rows")?;
        let cols = extract_index_vec_inferred(&cols_obj, "structure cols")?;
        if rows.len() != cols.len() {
            return Err(PyValueError::new_err(format!(
                "structure rows and cols length mismatch: {} vs {}",
                rows.len(),
                cols.len()
            )));
        }
        let _ = decode_structure;
        Ok((rows, cols))
    })
}

fn extract_index_vec_inferred(val: &Py<PyAny>, what: &str) -> PyResult<Vec<Index>> {
    Python::with_gil(|py| {
        let bound = val.bind(py);
        // Non-contiguous integer views (e.g. `idx[::2]`) still downcast to
        // these array types, so fall back to a strided ndarray view instead
        // of letting `as_slice()` error (L49).
        if let Ok(arr) = bound.downcast::<PyArray1<i64>>() {
            return Ok(match unsafe { arr.as_slice() } {
                Ok(s) => s.iter().map(|&x| x as Index).collect(),
                Err(_) => arr
                    .readonly()
                    .as_array()
                    .iter()
                    .map(|&x| x as Index)
                    .collect(),
            });
        }
        if let Ok(arr) = bound.downcast::<PyArray1<i32>>() {
            return Ok(match unsafe { arr.as_slice() } {
                Ok(s) => s.iter().map(|&x| x as Index).collect(),
                Err(_) => arr
                    .readonly()
                    .as_array()
                    .iter()
                    .map(|&x| x as Index)
                    .collect(),
            });
        }
        let mut out = Vec::new();
        for item in bound.iter()? {
            let v: i64 = item?.extract().map_err(|e| {
                PyValueError::new_err(format!("{what}: expected int sequence ({e})"))
            })?;
            out.push(v as Index);
        }
        Ok(out)
    })
}

pub(crate) fn extract_f64_vec(
    val: &Py<PyAny>,
    expected: usize,
    what: &str,
) -> PyResult<Vec<Number>> {
    Python::with_gil(|py| {
        let bound = val.bind(py);
        if let Ok(arr) = bound.downcast::<PyArray1<Number>>() {
            let got = arr.len();
            if got != expected {
                return Err(PyValueError::new_err(format!(
                    "{what}: expected length {expected}, got {got}",
                )));
            }
            // `as_slice()` requires C-contiguity. A valid but non-contiguous
            // float64 array (a strided view such as `x[::2]`, or a column of
            // a 2-D array) is still a `PyArray1<f64>`, so it reaches this fast
            // path; copy it via a strided ndarray view rather than erroring
            // (L49). The generic fallback below is never hit for such arrays.
            return Ok(match unsafe { arr.as_slice() } {
                Ok(s) => s.to_vec(),
                Err(_) => arr.readonly().as_array().iter().copied().collect(),
            });
        }
        let mut out = Vec::with_capacity(expected);
        for item in bound.iter()? {
            let v: Number = item?.extract()?;
            out.push(v);
        }
        if out.len() != expected {
            return Err(PyValueError::new_err(format!(
                "{what}: expected length {expected}, got {}",
                out.len()
            )));
        }
        Ok(out)
    })
}

fn decode_bounds(
    val: Option<Py<PyAny>>,
    expected: usize,
    default: Number,
) -> PyResult<Vec<Number>> {
    if expected == 0 {
        return Ok(Vec::new());
    }
    match val {
        None => Ok(vec![default; expected]),
        Some(v) => extract_f64_vec(&v, expected, "bounds"),
    }
}

/// Mirror of `pounce-cli`'s `default_backend_factory`: FERAL with the
/// caller's `feral_*` overrides. The python wrapper always uses FERAL —
/// the optional MA57 backend would require linking against
/// `pounce-hsl`, which we deliberately keep out of the wheel.
fn default_backend_factory(feral_cfg: pounce_feral::FeralConfig) -> LinearBackendFactory {
    Box::new(
        move |_choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
            // Only FERAL is wired into the wheel build; the `_choice`
            // argument is honored by the CLI build (which can route to
            // MA57) but ignored here.
            Box::new(pounce_feral::FeralSolverInterface::with_config(
                feral_cfg.clone(),
            ))
        },
    )
}

// ─────────────────────────────────────────────────────────────
// §7.3 SQP working-set encoding helpers. Python representation:
// a 2-tuple `(bounds, constraints)` of numpy int8 arrays whose
// element values are the same `POUNCE_WS_*` integer codes used
// by the C ABI (0..=3). int8 (vs int64) keeps wire size minimal
// for large `n`/`m`.
// ─────────────────────────────────────────────────────────────

fn encode_working_set<'py>(
    py: Python<'py>,
    ws: &pounce_qp::WorkingSet,
) -> Bound<'py, pyo3::types::PyTuple> {
    use pounce_qp::{BoundStatus, ConsStatus};
    let bounds_vec: Vec<i8> = ws
        .bounds
        .iter()
        .map(|s| match s {
            BoundStatus::Inactive => 0,
            BoundStatus::AtLower => 1,
            BoundStatus::AtUpper => 2,
            BoundStatus::Fixed => 3,
        })
        .collect();
    let cons_vec: Vec<i8> = ws
        .constraints
        .iter()
        .map(|s| match s {
            ConsStatus::Inactive => 0,
            ConsStatus::AtLower => 1,
            ConsStatus::AtUpper => 2,
            ConsStatus::Equality => 3,
        })
        .collect();
    let b_arr = bounds_vec.into_pyarray_bound(py).into_any();
    let c_arr = cons_vec.into_pyarray_bound(py).into_any();
    pyo3::types::PyTuple::new_bound(py, &[b_arr, c_arr])
}

fn decode_working_set(
    py: Python<'_>,
    obj: &Py<PyAny>,
    n: usize,
    m: usize,
) -> PyResult<pounce_qp::WorkingSet> {
    let bound = obj.bind(py);
    let tup: &Bound<'_, pyo3::types::PyTuple> = bound
        .downcast::<pyo3::types::PyTuple>()
        .map_err(|_| PyValueError::new_err("working_set must be a (bounds, constraints) tuple"))?;
    if tup.len() != 2 {
        return Err(PyValueError::new_err(
            "working_set tuple must have exactly two elements",
        ));
    }
    let bounds_obj = tup.get_item(0)?;
    let cons_obj = tup.get_item(1)?;
    let bounds_codes = extract_i8_vec(&bounds_obj.unbind(), n, "working_set[0] (bounds)")?;
    let cons_codes = extract_i8_vec(&cons_obj.unbind(), m, "working_set[1] (constraints)")?;
    let mut bounds = Vec::with_capacity(n);
    for (i, &c) in bounds_codes.iter().enumerate() {
        bounds.push(match c {
            0 => pounce_qp::BoundStatus::Inactive,
            1 => pounce_qp::BoundStatus::AtLower,
            2 => pounce_qp::BoundStatus::AtUpper,
            3 => pounce_qp::BoundStatus::Fixed,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "working_set bounds[{i}] = {c} not in 0..=3"
                )))
            }
        });
    }
    let mut constraints = Vec::with_capacity(m);
    for (i, &c) in cons_codes.iter().enumerate() {
        constraints.push(match c {
            0 => pounce_qp::ConsStatus::Inactive,
            1 => pounce_qp::ConsStatus::AtLower,
            2 => pounce_qp::ConsStatus::AtUpper,
            3 => pounce_qp::ConsStatus::Equality,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "working_set constraints[{i}] = {c} not in 0..=3"
                )))
            }
        });
    }
    Ok(pounce_qp::WorkingSet {
        bounds,
        constraints,
    })
}

fn extract_i8_vec(val: &Py<PyAny>, expected: usize, what: &str) -> PyResult<Vec<i8>> {
    Python::with_gil(|py| {
        // Best-effort decode: pull an i64 list out of the object,
        // then narrow. Supports python lists, tuples, and numpy
        // arrays of any integer dtype.
        let bound = val.bind(py);
        let vals: Vec<i64> = bound
            .extract()
            .map_err(|e| PyValueError::new_err(format!("{what}: cannot extract integers: {e}")))?;
        if vals.len() != expected {
            return Err(PyValueError::new_err(format!(
                "{what}: length {} != expected {expected}",
                vals.len()
            )));
        }
        let mut out = Vec::with_capacity(expected);
        for (i, &v) in vals.iter().enumerate() {
            if !(-128..=127).contains(&v) {
                return Err(PyValueError::new_err(format!(
                    "{what}[{i}] = {v} outside int8 range"
                )));
            }
            out.push(v as i8);
        }
        Ok(out)
    })
}

pub(crate) fn status_message(status: ApplicationReturnStatus) -> &'static str {
    use ApplicationReturnStatus::*;
    match status {
        SolveSucceeded => "Solve_Succeeded",
        SolvedToAcceptableLevel => "Solved_To_Acceptable_Level",
        InfeasibleProblemDetected => "Infeasible_Problem_Detected",
        SearchDirectionBecomesTooSmall => "Search_Direction_Becomes_Too_Small",
        DivergingIterates => "Diverging_Iterates",
        UserRequestedStop => "User_Requested_Stop",
        FeasiblePointFound => "Feasible_Point_Found",
        MaximumIterationsExceeded => "Maximum_Iterations_Exceeded",
        RestorationFailed => "Restoration_Failed",
        ErrorInStepComputation => "Error_In_Step_Computation",
        MaximumCpuTimeExceeded => "Maximum_CpuTime_Exceeded",
        MaximumWallTimeExceeded => "Maximum_WallTime_Exceeded",
        NotEnoughDegreesOfFreedom => "Not_Enough_Degrees_Of_Freedom",
        InvalidProblemDefinition => "Invalid_Problem_Definition",
        InvalidOption => "Invalid_Option",
        InvalidNumberDetected => "Invalid_Number_Detected",
        UnrecoverableException => "Unrecoverable_Exception",
        NonIpoptExceptionThrown => "NonIpopt_Exception_Thrown",
        InsufficientMemory => "Insufficient_Memory",
        InternalError => "Internal_Error",
    }
}
