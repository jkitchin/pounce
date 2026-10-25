//! `_pounce.find_minima`: the multistart / find-minima search
//! (`pounce_minima::find_minima_with`) run on a Python `Problem` (gh#966).
//!
//! `pounce.find_minima` (`python/pounce/_minima.py`) builds the `Problem`
//! exactly as `pounce.minimize` does and hands it here; the search loop, the
//! repulsion kernels, the archive and the acceptance tests all run in Rust,
//! the same code the `pounce --minima` CLI runs. Only the model's own
//! callbacks (and the caller's `callback` / `distance`) call back into
//! Python.

use std::cell::RefCell;
use std::ops::ControlFlow;
use std::rc::Rc;

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use pounce_common::types::Number;
use pounce_minima::{
    HessianMode, Hooks, IpoptLocalSolve, LocalOutcome, LocalSolve, Method, MinimaOptions,
    find_minima_with,
};
use pounce_nlp::tnlp::TNLP;

use crate::problem::{PyProblem, extract_f64_vec};

/// [`IpoptLocalSolve`] for a Python `Problem`. When the problem exposes no
/// `hessian`, `PyProblem::prepare` already set `limited-memory` (and pinned
/// `mu_strategy`), and the bridge fills its declared Hessian structure with
/// zeros; an `Exact` request is then demoted, because asking for the exact
/// Hessian would hand the IPM that zero matrix (issue #348).
struct PyLocalSolve<'a> {
    inner: IpoptLocalSolve<'a>,
    has_hessian: bool,
}

impl LocalSolve for PyLocalSolve<'_> {
    fn begin_search(&mut self) {
        self.inner.begin_search();
    }

    fn solve(&mut self, problem: Rc<RefCell<dyn TNLP>>, mode: HessianMode) -> LocalOutcome {
        let mode = if self.has_hessian {
            mode
        } else {
            HessianMode::QuasiNewton
        };
        self.inner.solve(problem, mode)
    }

    fn has_exact_hessian(&self) -> bool {
        self.has_hessian
    }

    fn should_stop(&self) -> bool {
        self.inner.should_stop()
    }
}

fn get<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => {
            Ok(Some(v.extract::<T>().map_err(|e| {
                PyValueError::new_err(format!("find_minima: option {key:?}: {e}"))
            })?))
        }
        _ => Ok(None),
    }
}

/// A width given as a scalar or a per-dimension vector: `(scalar, vector)`.
fn width(d: &Bound<'_, PyDict>, key: &str, n: usize) -> PyResult<(Option<f64>, Option<Vec<f64>>)> {
    let Some(v) = d.get_item(key)? else {
        return Ok((None, None));
    };
    if v.is_none() {
        return Ok((None, None));
    }
    if let Ok(s) = v.extract::<f64>() {
        return Ok((Some(s), None));
    }
    let vec = extract_f64_vec(&v.unbind(), n, key)?;
    Ok((None, Some(vec)))
}

fn options_from_dict(d: &Bound<'_, PyDict>, n: usize) -> PyResult<MinimaOptions> {
    let mut o = MinimaOptions::default();
    if let Some(m) = get::<String>(d, "method")? {
        o.method = Method::parse(&m).map_err(PyValueError::new_err)?;
    }
    if let Some(v) = get(d, "n_minima")? {
        o.n_minima = v;
    }
    o.max_solves = get(d, "max_solves")?;
    if let Some(v) = get(d, "patience")? {
        o.patience = v;
    }
    if let Some(v) = get(d, "dedup")? {
        o.dedup = v;
    }
    if let Some(v) = get(d, "psd_tol")? {
        o.psd_tol = v;
    }
    // `kkt_tol` is the one knob whose `None` means "off", not "default".
    o.kkt_tol = get(d, "kkt_tol")?;
    if let Some(v) = get(d, "seed")? {
        o.seed = v;
    }
    if let Some(v) = get(d, "sobol")? {
        o.sobol = v;
    }
    (o.sigma, o.sigma_per_dim) = width(d, "sigma", n)?;
    (o.length, o.length_per_dim) = width(d, "length", n)?;
    o.sigma_frac = get(d, "sigma_frac")?;
    o.amplitude = get(d, "amplitude")?;
    o.amp_margin = get(d, "amp_margin")?;
    o.amp_bump = get(d, "amp_bump")?;
    o.amp_bump_cap = get(d, "amp_bump_cap")?;
    o.amp_fallback = get(d, "amp_fallback")?;
    o.eta = get(d, "eta")?;
    o.power = get(d, "power")?;
    o.soft = get(d, "soft")?;
    o.length_frac = get(d, "length_frac")?;
    o.gamma = get(d, "gamma")?;
    o.samples_per_round = get(d, "samples_per_round")?;
    o.max_samples = get(d, "max_samples")?;
    o.step = get(d, "step")?;
    o.temperature = get(d, "temperature")?;
    o.restart_jitter = get(d, "restart_jitter")?;
    Ok(o)
}

fn array<'py>(py: Python<'py>, v: &[Number]) -> Bound<'py, PyArray1<Number>> {
    v.to_vec().into_pyarray_bound(py)
}

/// Run the search on `problem` from `x0`. `options` holds the
/// [`MinimaOptions`] fields by name (absent or `None` = default; widths may
/// be a scalar or a length-`n` sequence). `callback(x, f)` is called on each
/// accepted minimum and `distance(a, b)` replaces the dedup metric; an
/// exception raised by either ends the search and is re-raised here.
///
/// Returns a dict: `minima` (arrays, best first), `values`, `lambdas`
/// (base-problem constraint duals per minimum), `status`, `n_solves`,
/// `n_kkt_rejected`, `trace` (list of `{x, f, success, accepted, n_found}`),
/// `warnings`, `scale` (the dedup metric's per-dimension length).
#[pyfunction]
#[pyo3(signature = (problem, x0, options, callback=None, distance=None))]
pub fn find_minima<'py>(
    py: Python<'py>,
    problem: Bound<'py, PyProblem>,
    x0: Py<PyAny>,
    options: &Bound<'py, PyDict>,
    callback: Option<Py<PyAny>>,
    distance: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let pb = problem.borrow();
    let (n, _m) = pb.dims();
    let cfg = options_from_dict(options, n)?;
    let has_hessian = pb.uses_exact_hessian();
    let (mut app, bridge) = pb.prepare(py, x0, None, None, None)?;
    drop(pb);
    let base: Rc<RefCell<dyn TNLP>> = bridge;

    // The first Python exception a hook raises; it ends the search and is
    // re-raised once the search has returned.
    let raised: RefCell<Option<PyErr>> = RefCell::new(None);
    let mut on_accept = |x: &[Number], f: Number| -> ControlFlow<()> {
        let Some(cb) = callback.as_ref() else {
            return ControlFlow::Continue(());
        };
        match cb.call1(py, (array(py, x), f)) {
            Ok(_) => ControlFlow::Continue(()),
            Err(e) => {
                raised.borrow_mut().get_or_insert(e);
                ControlFlow::Break(())
            }
        }
    };
    let mut metric = |a: &[Number], b: &[Number]| -> Number {
        if raised.borrow().is_some() {
            // Already failing: call "known" so every candidate is rejected
            // and patience ends the search quickly.
            return 0.0;
        }
        let Some(d) = distance.as_ref() else {
            return 0.0; // not installed without `distance`
        };
        match d
            .call1(py, (array(py, a), array(py, b)))
            .and_then(|v| v.extract::<f64>(py))
        {
            Ok(v) => v,
            Err(e) => {
                raised.borrow_mut().get_or_insert(e);
                0.0
            }
        }
    };
    let hooks = Hooks {
        on_accept: Some(&mut on_accept),
        distance: if distance.is_some() {
            Some(&mut metric)
        } else {
            None
        },
        trace: true,
    };

    let result = {
        let mut solver = PyLocalSolve {
            inner: IpoptLocalSolve::new(&mut app),
            has_hessian,
        };
        find_minima_with(&base, &mut solver, &cfg, hooks)
            .map_err(|e| PyValueError::new_err(format!("find_minima: {e}")))?
    };
    if let Some(e) = raised.into_inner() {
        return Err(e);
    }

    let out = PyDict::new_bound(py);
    let minima = PyList::empty_bound(py);
    let lambdas = PyList::empty_bound(py);
    let values = PyList::empty_bound(py);
    for mn in &result.minima {
        minima.append(array(py, &mn.x))?;
        lambdas.append(array(py, &mn.lambda))?;
        values.append(mn.objective)?;
    }
    let trace = PyList::empty_bound(py);
    for t in &result.trace {
        let e = PyDict::new_bound(py);
        e.set_item("x", array(py, &t.x))?;
        e.set_item("f", t.f)?;
        e.set_item("success", t.success)?;
        e.set_item("accepted", t.accepted)?;
        e.set_item("n_found", t.n_found)?;
        trace.append(e)?;
    }
    out.set_item("minima", minima)?;
    out.set_item("values", values)?;
    out.set_item("lambdas", lambdas)?;
    out.set_item("status", result.stop.as_str())?;
    out.set_item("n_solves", result.n_solves)?;
    out.set_item("n_kkt_rejected", result.n_kkt_rejected)?;
    out.set_item("trace", trace)?;
    out.set_item("warnings", result.warnings)?;
    out.set_item("scale", array(py, &result.scale))?;
    Ok(out)
}
