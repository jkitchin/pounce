//! Batched NLP solving from Python (pounce#126) — native `.nl` path.
//!
//! `solve_nlp_batch(problems, ...)` takes a list of [`PyNlProblem`]s
//! (the native-Rust evaluators `read_nl` returns — reverse-mode AD
//! tapes, no Python callbacks, no GIL during evaluation), clones each
//! problem's owned `NlTnlp` out of its pyclass, releases the GIL, and
//! runs `pounce_algorithm::solve_nlp_batch_parallel` on rayon's global
//! pool (outer-parallel across instances, inner-serial FERAL factor
//! per worker). One `(x, info)` pair per input, in input order.
//!
//! Callback-based `pounce.Problem` objects cannot take this path —
//! every `eval_*` would re-acquire the GIL, serializing the batch (see
//! the phase-2 discussion on pounce#126). The pure-Python
//! `pounce.solve_nlp_batch` wrapper routes those to a documented
//! sequential fallback instead.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use pounce_algorithm::application::{default_backend_factory, feral_config_from_options};
use pounce_algorithm::batch::{
    install_serial_feral_backend, solve_nlp_batch as solve_batch_seq,
    solve_nlp_batch_parallel as solve_batch_par, NlpBatchResult,
};
use pounce_algorithm::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    make_default_restoration_factory_provider, InnerBackendFactoryFactory,
};

use crate::nl_problem::PyNlProblem;
use crate::problem::status_message;

/// Options decoded from the user dict, in the three value classes the
/// `OptionsList` distinguishes. Plain data — `Sync`, shared by the
/// per-worker configure closure.
#[derive(Default)]
struct BatchOptions {
    str_opts: Vec<(String, String)>,
    int_opts: Vec<(String, Index)>,
    num_opts: Vec<(String, Number)>,
}

/// Decode an options dict with the same value coercion as
/// `Problem.add_option`: `bool` → `"yes"`/`"no"` (checked before
/// `int`, since Python's `bool` subclasses `int`), then `int`, then
/// `float`, then `str`.
fn decode_options(options: Option<&Bound<'_, PyDict>>) -> PyResult<BatchOptions> {
    let mut out = BatchOptions::default();
    let Some(dict) = options else {
        return Ok(out);
    };
    for (key, value) in dict.iter() {
        let name: String = key.extract().map_err(|_| {
            PyValueError::new_err("solve_nlp_batch: option names must be strings")
        })?;
        if value.is_instance_of::<pyo3::types::PyBool>() {
            if let Ok(b) = value.extract::<bool>() {
                out.str_opts
                    .push((name, if b { "yes".into() } else { "no".into() }));
                continue;
            }
        }
        if let Ok(i) = value.extract::<i64>() {
            out.int_opts.push((name, i as Index));
            continue;
        }
        if let Ok(f) = value.extract::<f64>() {
            out.num_opts.push((name, f));
            continue;
        }
        if let Ok(s) = value.extract::<String>() {
            out.str_opts.push((name, s));
            continue;
        }
        return Err(PyValueError::new_err(format!(
            "solve_nlp_batch: option {name}: expected str / int / float / bool, got {}",
            value.get_type().name()?
        )));
    }
    Ok(out)
}

/// Apply decoded options to a fresh application, surfacing the first
/// rejected option as an error.
fn apply_options(app: &mut IpoptApplication, opts: &BatchOptions) -> Result<(), String> {
    for (k, v) in &opts.str_opts {
        app.options_mut()
            .set_string_value(k, v, true, false)
            .map_err(|e| format!("option {k}={v}: {e}"))?;
    }
    for (k, v) in &opts.num_opts {
        app.options_mut()
            .set_numeric_value(k, *v, true, false)
            .map_err(|e| format!("option {k}={v}: {e}"))?;
    }
    for (k, v) in &opts.int_opts {
        app.options_mut()
            .set_integer_value(k, *v, true, false)
            .map_err(|e| format!("option {k}={v}: {e}"))?;
    }
    Ok(())
}

/// The per-worker application recipe — the batch analog of
/// `Problem::prepare`: user options, a FERAL linear-solver backend
/// (inner-serial when the batch is parallel), and the default
/// restoration-phase provider. Options were validated on a probe app
/// before the batch started, so rejections here are unreachable and
/// ignored.
fn configure_worker(app: &mut IpoptApplication, opts: &BatchOptions, parallel: bool) {
    let _ = apply_options(app, opts);
    let mut feral_cfg = feral_config_from_options(app.options());
    if parallel {
        install_serial_feral_backend(app);
        feral_cfg.parallel = Some(false);
    } else {
        let cfg = feral_cfg.clone();
        app.set_linear_backend_factory(default_backend_factory(cfg));
    }
    // Restoration phase, including for the inner solves it runs.
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
}

/// Per-instance `(x, info)` pairs, one per input, in input order.
type BatchResults<'py> = Vec<(Bound<'py, PyArray1<Number>>, Bound<'py, PyDict>)>;

/// Build the per-instance `(x, info)` pair. Mirrors the key layout of
/// `Problem.solve`'s info dict (status / status_msg / obj_val / g /
/// mult_g / mult_x_L / mult_x_U / iter_count / mu / final_* metrics)
/// so downstream code can treat both uniformly.
fn build_result<'py>(
    py: Python<'py>,
    r: &NlpBatchResult,
    n: usize,
    m: usize,
) -> PyResult<(Bound<'py, PyArray1<Number>>, Bound<'py, PyDict>)> {
    let info = PyDict::new_bound(py);
    info.set_item("status", r.status as i32)?;
    info.set_item("status_msg", status_message(r.status))?;
    info.set_item("iter_count", r.stats.iteration_count)?;
    info.set_item("mu", r.stats.final_mu)?;
    info.set_item("final_kkt_error", r.stats.final_kkt_error)?;
    info.set_item("final_dual_inf", r.stats.final_dual_inf)?;
    info.set_item("final_constr_viol", r.stats.final_constr_viol)?;
    info.set_item("final_compl", r.stats.final_compl)?;
    let x = match &r.solution {
        Some(sol) => {
            info.set_item("obj_val", sol.obj)?;
            info.set_item("g", sol.g.clone().into_pyarray_bound(py))?;
            info.set_item("mult_g", sol.lambda.clone().into_pyarray_bound(py))?;
            info.set_item("mult_x_L", sol.z_l.clone().into_pyarray_bound(py))?;
            info.set_item("mult_x_U", sol.z_u.clone().into_pyarray_bound(py))?;
            sol.x.clone()
        }
        // The solve aborted before `finalize_solution` ran (e.g. an
        // invalid problem definition): no iterate to report.
        None => {
            info.set_item("obj_val", f64::NAN)?;
            info.set_item("g", vec![f64::NAN; m].into_pyarray_bound(py))?;
            info.set_item("mult_g", vec![f64::NAN; m].into_pyarray_bound(py))?;
            info.set_item("mult_x_L", vec![f64::NAN; n].into_pyarray_bound(py))?;
            info.set_item("mult_x_U", vec![f64::NAN; n].into_pyarray_bound(py))?;
            vec![f64::NAN; n]
        }
    };
    Ok((x.into_pyarray_bound(py), info))
}

/// Solve a batch of independent native (`.nl`-loaded) NLPs, one result
/// per input in input order.
///
/// `problems` is a sequence of `pounce.NlProblem` (from `read_nl` /
/// `NlProblem.variant`). Each instance is solved end-to-end with its
/// own application; with `parallel=True` (default) instances run
/// concurrently on rayon with the GIL released and each worker using
/// an inner-serial FERAL factor; `parallel=False` solves sequentially
/// (each factor may then parallelize internally).
///
/// `options` accepts the same IPOPT-style names and value coercion as
/// `Problem.add_option`, applied identically to every instance.
/// `print_level` defaults to 0 for the batch (explicit values win).
#[pyfunction]
#[pyo3(signature = (problems, options=None, parallel=true))]
pub fn solve_nlp_batch<'py>(
    py: Python<'py>,
    problems: Vec<Bound<'py, PyNlProblem>>,
    options: Option<&Bound<'py, PyDict>>,
    parallel: bool,
) -> PyResult<BatchResults<'py>> {
    let mut opts = decode_options(options)?;
    // Quiet by default: N workers interleaving per-iteration tables on
    // one stdout is noise, not output. An explicit user `print_level`
    // (or `sb`) still wins.
    if !opts.int_opts.iter().any(|(k, _)| k == "print_level")
        && !opts.str_opts.iter().any(|(k, _)| k == "print_level")
    {
        opts.int_opts.push(("print_level".into(), 0));
    }
    // Validate the option set once, up front, so a typo is a Python
    // error here rather than something each worker silently drops.
    {
        let mut probe = IpoptApplication::new();
        apply_options(&mut probe, &opts).map_err(PyRuntimeError::new_err)?;
    }

    let dims: Vec<(usize, usize)> = problems.iter().map(|p| p.borrow().dims()).collect();
    let tnlps: Vec<_> = problems.iter().map(|p| p.borrow().clone_tnlp()).collect();

    // The evaluators are native Rust (`NlTnlp: Send`) — no Python
    // callbacks — so the whole batch runs with the GIL released and
    // only plain-data results come back.
    let results: Vec<NlpBatchResult> = py.allow_threads(move || {
        if parallel {
            solve_batch_par(tnlps, |app| configure_worker(app, &opts, true))
        } else {
            solve_batch_seq(tnlps, |app| configure_worker(app, &opts, false))
        }
    });

    results
        .iter()
        .zip(dims)
        .map(|(r, (n, m))| build_result(py, r, n, m))
        .collect()
}
