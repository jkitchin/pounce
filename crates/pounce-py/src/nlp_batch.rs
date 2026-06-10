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
    install_pooled_serial_feral_backend, install_serial_feral_backend, FeralBackendPool,
    solve_nlp_batch as solve_batch_seq, solve_nlp_batch_parallel as solve_batch_par,
    solve_nlp_batch_parallel_warm as solve_batch_par_warm,
    solve_nlp_batch_warm as solve_batch_seq_warm, NlpBatchResult, NlpWarmStart,
};
use std::sync::Arc;
use pounce_algorithm::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    make_default_restoration_factory_provider, InnerBackendFactoryFactory,
};

use crate::nl_problem::PyNlProblem;
use crate::problem::{extract_f64_vec, status_message, PyProblem};
use crate::tnlp_bridge::PyTnlp;

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
/// (inner-serial when the batch is parallel; drawn from `pool` when
/// the caller opted into cross-instance structure reuse), and the
/// default restoration-phase provider. Options were validated on a
/// probe app before the batch started, so rejections here are
/// unreachable and ignored.
fn configure_worker(
    app: &mut IpoptApplication,
    opts: &BatchOptions,
    parallel: bool,
    pool: Option<&Arc<FeralBackendPool>>,
) {
    let _ = apply_options(app, opts);
    let mut feral_cfg = feral_config_from_options(app.options());
    if let Some(pool) = pool {
        install_pooled_serial_feral_backend(app, pool);
        feral_cfg.parallel = Some(false);
    } else if parallel {
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

/// Decode per-instance warm starts from previous batch results: a
/// sequence of `(x, info)` pairs as returned by `solve_nlp_batch` /
/// `solve_problem_batch` (or any dict carrying `mult_g` / `mult_x_L` /
/// `mult_x_U`). Validated against each instance's dims so a mismatch
/// is a Python error here, not a silent cold start.
fn decode_warms(
    py: Python<'_>,
    warms: &Bound<'_, PyAny>,
    dims: &[(usize, usize)],
) -> PyResult<Vec<NlpWarmStart>> {
    let mut out = Vec::with_capacity(dims.len());
    for (i, item) in warms.iter()?.enumerate() {
        let item = item?;
        let (n, m) = *dims.get(i).ok_or_else(|| {
            PyValueError::new_err(format!(
                "warms: got more than the {} problems in the batch",
                dims.len()
            ))
        })?;
        let pair: (Py<PyAny>, Py<PyAny>) = item.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "warms[{i}]: expected an (x, info) pair as returned by a previous batch solve"
            ))
        })?;
        let x = extract_f64_vec(&pair.0, n, &format!("warms[{i}].x"))?;
        let info = pair.1.bind(py);
        let pull = |key: &str, len: usize| -> PyResult<Vec<Number>> {
            let v = info.get_item(key).map_err(|_| {
                PyValueError::new_err(format!("warms[{i}]: info dict is missing \"{key}\""))
            })?;
            extract_f64_vec(&v.unbind(), len, &format!("warms[{i}].{key}"))
        };
        // Resume from the previous solve's barrier parameter when the
        // info dict carries one (`Problem.solve` and both batch paths
        // emit "mu" for exactly this hand-off).
        let mu = info
            .get_item("mu")
            .ok()
            .and_then(|v| v.extract::<Number>().ok());
        out.push(NlpWarmStart {
            x,
            lambda: pull("mult_g", m)?,
            z_l: pull("mult_x_L", n)?,
            z_u: pull("mult_x_U", n)?,
            mu,
        });
    }
    if out.len() != dims.len() {
        return Err(PyValueError::new_err(format!(
            "warms: got {} entries for {} problems",
            out.len(),
            dims.len()
        )));
    }
    Ok(out)
}

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
///
/// `warms` (optional) seeds each instance from a previous batch's
/// `(x, info)` results (MPC / sequential-chaining);
/// `warm_start_init_point=yes` is forced for the warm batch.
///
/// `share_structure=True` opts identical-sparsity batches into the
/// per-worker backend pool: each worker keeps its FERAL solver across
/// instances, so the symbolic analysis (fill-reducing ordering +
/// supernode structure) is computed once per worker instead of once
/// per instance. Always correct (a pattern change triggers a fresh
/// analysis); only profitable when instances share their KKT
/// sparsity, and results are then within solver tolerance of — but
/// not guaranteed bit-identical to — fresh-backend solves.
#[pyfunction]
#[pyo3(signature = (problems, options=None, parallel=true, warms=None, share_structure=false))]
pub fn solve_nlp_batch<'py>(
    py: Python<'py>,
    problems: Vec<Bound<'py, PyNlProblem>>,
    options: Option<&Bound<'py, PyDict>>,
    parallel: bool,
    warms: Option<&Bound<'py, PyAny>>,
    share_structure: bool,
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
    // The probe also resolves the `feral_*` options the optional
    // backend pool is configured with.
    let pool = {
        let mut probe = IpoptApplication::new();
        apply_options(&mut probe, &opts).map_err(PyRuntimeError::new_err)?;
        share_structure.then(|| FeralBackendPool::serial(feral_config_from_options(probe.options())))
    };

    let dims: Vec<(usize, usize)> = problems.iter().map(|p| p.borrow().dims()).collect();
    let tnlps: Vec<_> = problems.iter().map(|p| p.borrow().clone_tnlp()).collect();
    let warm_starts = warms.map(|w| decode_warms(py, w, &dims)).transpose()?;

    // The evaluators are native Rust (`NlTnlp: Send`) — no Python
    // callbacks — so the whole batch runs with the GIL released and
    // only plain-data results come back.
    let results: Vec<NlpBatchResult> = py.allow_threads(move || {
        let configure = |_i: usize, app: &mut IpoptApplication| {
            configure_worker(app, &opts, parallel, pool.as_ref())
        };
        match (parallel, warm_starts) {
            (true, None) => solve_batch_par(tnlps, configure),
            (false, None) => solve_batch_seq(tnlps, configure),
            (true, Some(ws)) => solve_batch_par_warm(tnlps, ws, configure),
            (false, Some(ws)) => solve_batch_seq_warm(tnlps, ws, configure),
        }
    });

    results
        .iter()
        .zip(dims)
        .map(|(r, (n, m))| build_result(py, r, n, m))
        .collect()
}

/// Solve a batch of independent callback-based `pounce.Problem`s —
/// pounce#126 **phase 2**.
///
/// Each instance's bridge (the Python callables plus pre-resolved
/// sparsity) is assembled here under the GIL, then *moved* to a rayon
/// worker that owns the whole solve; the GIL is released for the
/// duration of the batch and re-acquired transiently inside every
/// `objective` / `gradient` / `constraints` / `jacobian` / `hessian`
/// callback. Correctness does not depend on timing — the GIL
/// serializes the Python work — but the *speedup* does: only the
/// solver's Rust share (KKT factorization, backsolves, line search)
/// runs concurrently, so the win scales with the Rust/Python work
/// ratio and tops out near zero for tiny problems whose callbacks
/// dominate. Native `.nl` batches (`solve_nlp_batch`) don't have this
/// ceiling.
///
/// Per-instance options are honored: each worker applies its
/// `Problem`'s own `add_option` settings (plus the L-BFGS default when
/// that instance lacks a Hessian callback), then the batch-level
/// `options` overlay, then the batch's quiet `print_level` default.
///
/// `x0s` supplies one starting point per instance. With `warms`, the
/// warm iterate (its `x` and duals) takes precedence and
/// `warm_start_init_point=yes` is forced. `share_structure` — see
/// [`solve_nlp_batch`]; the pool's FERAL configuration is resolved
/// from the batch-level `options` overlay.
#[pyfunction]
#[pyo3(signature = (problems, x0s, options=None, parallel=true, warms=None, share_structure=false))]
pub fn solve_problem_batch<'py>(
    py: Python<'py>,
    problems: Vec<Bound<'py, PyProblem>>,
    x0s: Vec<Py<PyAny>>,
    options: Option<&Bound<'py, PyDict>>,
    parallel: bool,
    warms: Option<&Bound<'py, PyAny>>,
    share_structure: bool,
) -> PyResult<BatchResults<'py>> {
    if x0s.len() != problems.len() {
        return Err(PyValueError::new_err(format!(
            "solve_problem_batch: got {} problems but {} starting points",
            problems.len(),
            x0s.len()
        )));
    }
    let overlay = decode_options(options)?;
    let dims: Vec<(usize, usize)> = problems.iter().map(|p| p.borrow().dims()).collect();
    let warm_starts = warms.map(|w| decode_warms(py, w, &dims)).transpose()?;
    let pool = share_structure.then(|| {
        let mut probe = IpoptApplication::new();
        let _ = apply_options(&mut probe, &overlay);
        FeralBackendPool::serial(feral_config_from_options(probe.options()))
    });

    // Per-instance option sets: the instance's own `add_option`
    // settings first (after its L-BFGS default), then the batch
    // overlay so the caller can globally override, then the quiet
    // default. Validated on a probe app so a typo surfaces here.
    let mut per_instance: Vec<BatchOptions> = Vec::with_capacity(problems.len());
    for (i, p) in problems.iter().enumerate() {
        let pb = p.borrow();
        let mut o = BatchOptions::default();
        if !pb.uses_exact_hessian() {
            o.str_opts
                .push(("hessian_approximation".into(), "limited-memory".into()));
        }
        let (s, num, int) = pb.option_sets();
        o.str_opts.extend(s.iter().cloned());
        o.num_opts.extend(num.iter().cloned());
        o.int_opts.extend(int.iter().cloned());
        o.str_opts.extend(overlay.str_opts.iter().cloned());
        o.num_opts.extend(overlay.num_opts.iter().cloned());
        o.int_opts.extend(overlay.int_opts.iter().cloned());
        if !o.int_opts.iter().any(|(k, _)| k == "print_level")
            && !o.str_opts.iter().any(|(k, _)| k == "print_level")
        {
            o.int_opts.push(("print_level".into(), 0));
        }
        let mut probe = IpoptApplication::new();
        apply_options(&mut probe, &o)
            .map_err(|e| PyRuntimeError::new_err(format!("problem {i}: {e}")))?;
        per_instance.push(o);
    }

    // Assemble the bridges under the GIL (sparsity-structure callbacks
    // run here, once per instance); each `PyTnlp` is plain data plus
    // `Py<PyAny>` handles, hence `Send`, and moves to its worker.
    let mut bridges: Vec<PyTnlp> = Vec::with_capacity(problems.len());
    for (i, (p, x0)) in problems.iter().zip(&x0s).enumerate() {
        let pb = p.borrow();
        let (n, _m) = pb.dims();
        let x0_vec = extract_f64_vec(x0, n, &format!("x0s[{i}]"))?;
        let init = pb.build_tnlp_init(py, x0_vec, None, None, None)?;
        bridges.push(PyTnlp::new(init));
    }

    let results: Vec<NlpBatchResult> = py.allow_threads(move || {
        let configure = |i: usize, app: &mut IpoptApplication| {
            configure_worker(app, &per_instance[i], parallel, pool.as_ref())
        };
        match (parallel, warm_starts) {
            (true, None) => solve_batch_par(bridges, configure),
            (false, None) => solve_batch_seq(bridges, configure),
            (true, Some(ws)) => solve_batch_par_warm(bridges, ws, configure),
            (false, Some(ws)) => solve_batch_seq_warm(bridges, ws, configure),
        }
    });

    results
        .iter()
        .zip(dims)
        .map(|(r, (n, m))| build_result(py, r, n, m))
        .collect()
}

#[cfg(test)]
mod tests {
    /// The phase-2 contract: a callback bridge must be movable to a
    /// rayon worker (its `Py<PyAny>` handles are `Send`; everything
    /// else is plain data). Regresses if `!Send` state lands in
    /// `PyTnlp` / `PyTnlpInit`.
    #[test]
    fn py_tnlp_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<super::PyTnlp>();
    }
}
