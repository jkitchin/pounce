//! `Problem` PyO3 class — the cyipopt-compatible user-facing handle.
//!
//! Construction mirrors cyipopt: pass dims, bounds, and a `problem_obj`
//! with `objective`/`gradient`/`constraints`/`jacobian`/... methods.
//! Options are set with `add_option(name, value)`. `solve(x0)` returns
//! `(x_opt, info)` where `info` is a dict (status, status_msg, obj_val,
//! mult_g, mult_x_L, mult_x_U, iter_count, ...).

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pounce_algorithm::alg_builder::{AlgorithmBuilder, LinearBackendFactory, LinearSolverChoice};
use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_linsol::sparse_sym_iface::SparseSymLinearSolverInterface;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::TNLP;
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{make_default_restoration_factory, InnerBackendFactoryFactory};
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
        let is_bool = value
            .is_instance_of::<pyo3::types::PyBool>();
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
    #[pyo3(signature = (x0, lagrange=None, zl=None, zu=None))]
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        x0: Py<PyAny>,
        lagrange: Option<Py<PyAny>>,
        zl: Option<Py<PyAny>>,
        zu: Option<Py<PyAny>>,
    ) -> PyResult<(Bound<'py, PyArray1<Number>>, Bound<'py, PyDict>)> {
        let n = self.n as usize;
        let m = self.m as usize;
        let x0_vec = extract_f64_vec(&x0, n, "x0")?;
        let lam0 = lagrange.map(|v| extract_f64_vec(&v, m, "lagrange")).transpose()?;
        let z_l0 = zl.map(|v| extract_f64_vec(&v, n, "zl")).transpose()?;
        let z_u0 = zu.map(|v| extract_f64_vec(&v, n, "zu")).transpose()?;

        // Materialize Jacobian sparsity once.
        let (jac_rows, jac_cols, nele_jac) = if m > 0 {
            let s = call0(&self.problem_obj, "jacobianstructure")?;
            // We don't know nnz_jac up front; infer it from the
            // returned tuple's row length.
            let (rows, cols) = decode_structure_inferred(&s)?;
            (rows.clone(), cols.clone(), rows.len() as Index)
        } else {
            (Vec::new(), Vec::new(), 0)
        };

        // And Hessian sparsity. When the user provides one, use it
        // verbatim. Without one we still need a non-empty pattern: the
        // L-BFGS updater pins its work-space sparsity from
        // `curr_exact_hessian()`, so an empty space means nowhere for
        // the quasi-Newton approximation to land (and the solver
        // wanders). Declare the dense lower triangle — `eval_h(Values)`
        // returns zeros, and `LimMemQuasiNewtonUpdater` overwrites them
        // with its rank-update approximation each iteration.
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

        // Build app and apply options.
        let mut app = IpoptApplication::new();
        // Force L-BFGS unless the user gave us a Hessian.
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

        // Wire restoration (mirrors what `pounce-cli` does — without it,
        // any line-search failure surfaces as RestorationFailure
        // instead of falling back to the ℓ₁-feasibility sub-IPM).
        let feral_cfg = pounce_algorithm::application::feral_config_from_options(app.options());
        let bff: InnerBackendFactoryFactory =
            Box::new(move || default_backend_factory(feral_cfg));
        let resto_factory = make_default_restoration_factory(
            RestoAlgorithmBuilder::new(),
            AlgorithmBuilder::new(),
            bff,
        );
        app.set_restoration_factory(resto_factory);

        // Build the bridge.
        let init = PyTnlpInit {
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
            final_x: vec![0.0; n],
            final_z_l: vec![0.0; n],
            final_z_u: vec![0.0; n],
            final_g: vec![0.0; m],
            final_lambda: vec![0.0; m],
            final_obj: 0.0,
            final_status_code: 0,
        };
        let bridge = Rc::new(RefCell::new(PyTnlp::new(init)));
        let bridge_for_solve: Rc<RefCell<dyn TNLP>> = bridge.clone();

        // We do NOT `py.allow_threads` here: `IpoptApplication` and the
        // TNLP bridge hold `Rc<…>` internally, so they aren't `Send`.
        // The solver is single-threaded anyway, and intermediate
        // callbacks re-enter Python through `Python::with_gil` (which
        // is a no-op when the GIL is already held).
        let status: ApplicationReturnStatus = app.optimize_tnlp(bridge_for_solve);
        let _ = py;
        let stats = app.statistics();

        let b = bridge.borrow();
        let x_out = b.state.final_x.clone().into_pyarray_bound(py);
        let info = PyDict::new_bound(py);
        info.set_item("status", status as i32)?;
        info.set_item("status_msg", status_message(status))?;
        info.set_item("obj_val", b.state.final_obj)?;
        info.set_item(
            "g",
            b.state.final_g.clone().into_pyarray_bound(py),
        )?;
        info.set_item(
            "mult_g",
            b.state.final_lambda.clone().into_pyarray_bound(py),
        )?;
        info.set_item(
            "mult_x_L",
            b.state.final_z_l.clone().into_pyarray_bound(py),
        )?;
        info.set_item(
            "mult_x_U",
            b.state.final_z_u.clone().into_pyarray_bound(py),
        )?;
        info.set_item("iter_count", stats.iteration_count)?;
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
        if let Ok(arr) = bound.downcast::<PyArray1<i64>>() {
            return Ok(unsafe { arr.as_slice()? }
                .iter()
                .map(|&x| x as Index)
                .collect());
        }
        if let Ok(arr) = bound.downcast::<PyArray1<i32>>() {
            return Ok(unsafe { arr.as_slice()? }
                .iter()
                .map(|&x| x as Index)
                .collect());
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

fn extract_f64_vec(val: &Py<PyAny>, expected: usize, what: &str) -> PyResult<Vec<Number>> {
    Python::with_gil(|py| {
        let bound = val.bind(py);
        if let Ok(arr) = bound.downcast::<PyArray1<Number>>() {
            let got = arr.len();
            if got != expected {
                return Err(PyValueError::new_err(format!(
                    "{what}: expected length {expected}, got {got}",
                )));
            }
            return Ok(unsafe { arr.as_slice()? }.to_vec());
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

fn decode_bounds(val: Option<Py<PyAny>>, expected: usize, default: Number) -> PyResult<Vec<Number>> {
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
            Box::new(pounce_feral::FeralSolverInterface::with_config(feral_cfg))
        },
    )
}

fn status_message(status: ApplicationReturnStatus) -> &'static str {
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
