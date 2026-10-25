//! The engine-agnostic statistics and steps of `pounce_sens_core`, over a
//! converged [`crate::solver::PySolver`], for `pounce.sensitivity`.
//!
//! Each function returns a dict with the answer, the findings the
//! calculation met in order, and the error that stopped it, if one did.
//! Findings and errors cross as tuples whose first entry names their kind,
//! with indices into the model rather than names, because the names and the
//! prose are the Python front end's. An error raised by the factor itself
//! crosses as the exception the `Solver` methods raise for it, so a caller
//! sees the same exception it always did.

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pounce_common::types::{Index, Number};
use pounce_sens_core::boundcheck::RefineStop;
use pounce_sens_core::converged::SolverError;
use pounce_sens_core::corrector::CorrectorReport;
use pounce_sens_core::dense::Dense;
use pounce_sens_core::model::{Declarations, ModelFunctions, SolveRecord};
use pounce_sens_core::stats::{self, Deficiency, HessianForm, RemovedAt, StatsError, StatsFinding};
use pounce_sens_core::step::{
    self, BoundTarget, Degeneracy, Mode, MultiplierRefusal, StepError, StepFinding, StepOptions,
};
use pounce_sensitivity::pounce_sens_core;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::cell::RefCell;

use crate::solver::{PySolver, solver_error_to_py, status_name};

fn dense_to_py<'py>(py: Python<'py>, m: &Dense) -> PyResult<Bound<'py, PyArray2<Number>>> {
    m.data
        .clone()
        .into_pyarray_bound(py)
        .reshape([m.rows, m.cols])
}

fn dense_from_py(m: &Bound<'_, PyAny>) -> PyResult<Dense> {
    let py = m.py();
    let np = py.import_bound("numpy")?;
    let a = np.call_method1("asarray", (m, "float64"))?;
    let shape: (usize, usize) = a.getattr("shape")?.extract()?;
    let flat = a.call_method0("ravel")?;
    let v: PyReadonlyArray1<Number> = flat.extract()?;
    Ok(Dense {
        rows: shape.0,
        cols: shape.1,
        data: v.as_slice()?.to_vec(),
    })
}

fn floats(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Number>> {
    let py = obj.py();
    let np = py.import_bound("numpy")?;
    let a = np
        .call_method1("asarray", (obj, "float64"))?
        .call_method0("ravel")?;
    let v: PyReadonlyArray1<Number> = a.extract()?;
    Ok(v.as_slice()?.to_vec())
}

fn exception(py: Python<'_>, e: SolverError) -> PyObject {
    solver_error_to_py(e).into_value(py).into_any()
}

fn stop_name(s: RefineStop) -> &'static str {
    match s {
        RefineStop::Settled => "settled",
        RefineStop::IterationLimit => "iteration_limit",
        RefineStop::DegreesOfFreedom => "degrees_of_freedom",
        RefineStop::WorseThanPlain => "worse_than_plain",
    }
}

fn deficiency(d: Deficiency) -> &'static str {
    match d {
        Deficiency::Count => "count",
        Deficiency::Dependent => "dependent",
    }
}

fn corrector_dict<'py>(py: Python<'py>, rep: &CorrectorReport) -> PyResult<Bound<'py, PyDict>> {
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
    Ok(info)
}

fn combo(py: Python<'_>, c: &[(usize, Number)]) -> PyObject {
    c.iter()
        .map(|&(k, v)| (k, v))
        .collect::<Vec<_>>()
        .into_py(py)
}

fn stats_finding(py: Python<'_>, f: &StatsFinding) -> PyObject {
    match f {
        StatsFinding::Perturbed { perturbations } => {
            ("perturbed", perturbations.to_vec()).into_py(py)
        }
        StatsFinding::MemberHeld { member } => ("member_held", *member).into_py(py),
        StatsFinding::MemberWeak { member } => ("member_weak", *member).into_py(py),
        StatsFinding::MemberAmbiguous { member } => ("member_ambiguous", *member).into_py(py),
        StatsFinding::MemberUnidentified { member } => ("member_unidentified", *member).into_py(py),
        StatsFinding::RowMixedHeld { row } => ("row_mixed_held", *row).into_py(py),
        StatsFinding::RowMixedKept { row, status } => {
            ("row_mixed_kept", *row, status_name(*status)).into_py(py)
        }
        StatsFinding::RowPins {
            row,
            combo: c,
            information,
        } => ("row_pins", *row, combo(py, c), *information).into_py(py),
        StatsFinding::RowWeak { row, combo: c } => ("row_weak", *row, combo(py, c)).into_py(py),
        StatsFinding::RowAmbiguous { row, combo: c } => {
            ("row_ambiguous", *row, combo(py, c)).into_py(py)
        }
        StatsFinding::RowUnidentified { row, combo: c } => {
            ("row_unidentified", *row, combo(py, c)).into_py(py)
        }
        StatsFinding::NDataIgnored => ("n_data_ignored",).into_py(py),
        StatsFinding::DeficientMembers { members, reason } => {
            ("deficient_members", members.clone(), deficiency(*reason)).into_py(py)
        }
        StatsFinding::NegativeVariance => ("negative_variance",).into_py(py),
        StatsFinding::Indefinite => ("indefinite",).into_py(py),
        StatsFinding::SubblockFallback => ("subblock_fallback",).into_py(py),
    }
}

fn removed_at(py: Python<'_>, at: RemovedAt) -> PyObject {
    match at {
        RemovedAt::Member(k) => ("member", k).into_py(py),
        RemovedAt::Residual => ("residual",).into_py(py),
        RemovedAt::Fitted => ("fitted",).into_py(py),
        RemovedAt::ConditionedOn => ("conditioned_on",).into_py(py),
    }
}

fn step_error(py: Python<'_>, e: StepError, model_err: Option<PyErr>) -> PyObject {
    match e {
        StepError::Solver(s) => ("raise", exception(py, s)).into_py(py),
        StepError::NotPositive { name, value } => ("not_positive", name, value).into_py(py),
        StepError::Perturbed {
            worst,
            perturbations,
            limit,
        } => ("pdpert", worst, perturbations.to_vec(), limit).into_py(py),
        StepError::Model(m) => match model_err {
            Some(err) => ("raise", err.into_value(py).into_any()).into_py(py),
            None => (
                "raise",
                PyRuntimeError::new_err(m).into_value(py).into_any(),
            )
                .into_py(py),
        },
    }
}

fn stats_error(py: Python<'_>, e: StatsError) -> PyObject {
    match e {
        StatsError::Solver(s) => ("raise", exception(py, s)).into_py(py),
        StatsError::Step(s) => step_error(py, s, None),
        StatsError::Numeric(m) => (
            "raise",
            PyRuntimeError::new_err(m).into_value(py).into_any(),
        )
            .into_py(py),
        StatsError::SingularBlock => ("singular_block",).into_py(py),
        StatsError::Removed { full_x, at } => ("removed", full_x, removed_at(py, at)).into_py(py),
        StatsError::NoBlock => ("no_block",).into_py(py),
        StatsError::GaussNewtonNeedsResiduals => ("gn_needs_residuals",).into_py(py),
        StatsError::NoFitForResiduals => ("no_fit_for_residuals",).into_py(py),
        StatsError::GroupTooSmall {
            group,
            members,
            n_fit,
        } => ("group_too_small", group, members, n_fit).into_py(py),
        StatsError::NDataNoFit => ("n_data_no_fit",).into_py(py),
        StatsError::NDataTooSmall { n_data, n_fit } => {
            ("n_data_too_small", n_data, n_fit).into_py(py)
        }
        StatsError::NoObjectiveValue { obj } => ("no_objective_value", obj).into_py(py),
        StatsError::NoiseUnknown => ("noise_unknown",).into_py(py),
        StatsError::NoSolvePoint => ("no_solve_point",).into_py(py),
        StatsError::DeficientNeedsLagrangian { reason } => {
            ("deficient_needs_lagrangian", deficiency(reason)).into_py(py)
        }
        StatsError::GaussNewtonSingular => ("gn_singular",).into_py(py),
        StatsError::FreeSingular => ("free_singular",).into_py(py),
        StatsError::InformationDeficient { reason } => {
            ("information_deficient", deficiency(reason)).into_py(py)
        }
        StatsError::NotSquare {
            n_var,
            n_eq,
            n_params,
        } => ("not_square", n_var, n_eq, n_params).into_py(py),
        StatsError::PinnedUndefined => ("pinned_undefined",).into_py(py),
    }
}

fn hessian_form(h: &str) -> PyResult<HessianForm> {
    match h {
        "lagrangian" => Ok(HessianForm::Lagrangian),
        "gauss-newton" => Ok(HessianForm::GaussNewton),
        other => Err(PyValueError::new_err(format!(
            "hessian must be 'lagrangian' or 'gauss-newton', got {other:?}"
        ))),
    }
}

/// `pounce_sens_core::stats::covariance` over the solver's held factor.
#[pyfunction]
#[pyo3(signature = (solver, pins, fit_rows, res_groups, base_x, base_obj, obj_sign, block, known_sigma, n_data, hessian, max_pdpert))]
#[allow(clippy::too_many_arguments)]
pub fn _sens_covariance<'py>(
    py: Python<'py>,
    solver: &PySolver,
    pins: Vec<usize>,
    fit_rows: Vec<usize>,
    res_groups: Vec<Vec<usize>>,
    base_x: &Bound<'py, PyAny>,
    base_obj: Number,
    obj_sign: Number,
    block: Option<Vec<usize>>,
    known_sigma: Option<Vec<Number>>,
    n_data: Option<Number>,
    hessian: &str,
    max_pdpert: Option<Number>,
) -> PyResult<Bound<'py, PyDict>> {
    let form = hessian_form(hessian)?;
    let x = if base_x.is_none() {
        Vec::new()
    } else {
        floats(base_x)?
    };
    let out = solver.with_converged(|c| {
        Ok(stats::covariance(
            c,
            Declarations {
                pins: &pins,
                fit_rows: &fit_rows,
                res_groups: &res_groups,
            },
            SolveRecord {
                x: &x,
                obj: base_obj,
                obj_sign,
            },
            block.as_deref(),
            known_sigma.as_deref(),
            n_data,
            form,
            max_pdpert,
        ))
    })?;
    let d = PyDict::new_bound(py);
    let findings: Vec<PyObject> = out.findings.iter().map(|f| stats_finding(py, f)).collect();
    d.set_item("findings", findings)?;
    match out.result {
        Ok(cov) => {
            d.set_item("matrix", dense_to_py(py, &cov.matrix)?)?;
            d.set_item("sigma_sq", cov.sigma_sq)?;
            d.set_item("conditioned_on", cov.conditioned_on)?;
            d.set_item("error", py.None())?;
        }
        Err(e) => d.set_item("error", stats_error(py, e))?,
    }
    Ok(d)
}

/// `pounce_sens_core::stats::information` over the solver's held factor.
#[pyfunction]
#[pyo3(signature = (solver, pins, fit_rows, res_groups, block, hessian, max_pdpert))]
#[allow(clippy::too_many_arguments)]
pub fn _sens_information<'py>(
    py: Python<'py>,
    solver: &PySolver,
    pins: Vec<usize>,
    fit_rows: Vec<usize>,
    res_groups: Vec<Vec<usize>>,
    block: Option<Vec<usize>>,
    hessian: &str,
    max_pdpert: Option<Number>,
) -> PyResult<Bound<'py, PyDict>> {
    let form = hessian_form(hessian)?;
    let out = solver.with_converged(|c| {
        Ok(stats::information(
            c,
            Declarations {
                pins: &pins,
                fit_rows: &fit_rows,
                res_groups: &res_groups,
            },
            block.as_deref(),
            form,
            max_pdpert,
        ))
    })?;
    let d = PyDict::new_bound(py);
    let findings: Vec<PyObject> = out.findings.iter().map(|f| stats_finding(py, f)).collect();
    d.set_item("findings", findings)?;
    match out.result {
        Ok(info) => {
            d.set_item("matrix", dense_to_py(py, &info.matrix)?)?;
            d.set_item("conditioned_on", info.conditioned_on)?;
            d.set_item("error", py.None())?;
        }
        Err(e) => d.set_item("error", stats_error(py, e))?,
    }
    Ok(d)
}

/// The eigenvalues in ascending order and the sign-pinned eigenvectors of
/// a symmetric matrix. See `pounce_sens_core::dense::sym_eigen`.
#[pyfunction]
pub fn _sym_eigen<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<(Vec<Number>, Bound<'py, PyArray2<Number>>)> {
    let m = dense_from_py(matrix)?;
    let (w, v) = pounce_sens_core::dense::sym_eigen(&m).map_err(PyRuntimeError::new_err)?;
    Ok((w, dense_to_py(py, &v)?))
}

/// The standard errors and correlation matrix of a covariance. See
/// `pounce_sens_core::stats::correlation`.
#[pyfunction]
pub fn _correlation<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<(Vec<Number>, Bound<'py, PyArray2<Number>>)> {
    let m = dense_from_py(matrix)?;
    let (se, corr) = stats::correlation(&m);
    Ok((se, dense_to_py(py, &corr)?))
}

/// The model functions of a `read_nl` problem, or any object with its
/// attributes: `x_l`, `x_u`, `g_l`, `g_u`, `constraints`,
/// `jacobian_structure`, `jacobian` and `gradient`. A callback that raises
/// keeps its exception, which the binding raises again unchanged.
struct PyModel<'py> {
    nl: Bound<'py, PyAny>,
    var_bounds: (Vec<Number>, Vec<Number>),
    row_bounds: (Vec<Number>, Vec<Number>),
    err: RefCell<Option<PyErr>>,
}

impl<'py> PyModel<'py> {
    fn new(nl: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Self {
            nl: nl.clone(),
            var_bounds: (floats(&nl.getattr("x_l")?)?, floats(&nl.getattr("x_u")?)?),
            row_bounds: (floats(&nl.getattr("g_l")?)?, floats(&nl.getattr("g_u")?)?),
            err: RefCell::new(None),
        })
    }

    fn keep<T>(&self, r: PyResult<T>) -> Result<T, String> {
        r.map_err(|e| {
            let msg = e.to_string();
            *self.err.borrow_mut() = Some(e);
            msg
        })
    }

    fn array(&self, x: &[Number]) -> Bound<'py, numpy::PyArray1<Number>> {
        x.to_vec().into_pyarray_bound(self.nl.py())
    }
}

impl ModelFunctions for PyModel<'_> {
    fn variable_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        self.var_bounds.clone()
    }

    fn row_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        self.row_bounds.clone()
    }

    fn constraints(&mut self, x: &[Number]) -> Result<Vec<Number>, String> {
        let r = self
            .nl
            .call_method1("constraints", (self.array(x),))
            .and_then(|v| floats(&v));
        self.keep(r)
    }

    fn jacobian(&mut self, x: &[Number]) -> Result<Vec<(usize, usize, Number)>, String> {
        let r = (|| -> PyResult<Vec<(usize, usize, Number)>> {
            let st = self.nl.call_method0("jacobian_structure")?;
            let (rows, cols): (Bound<'_, PyAny>, Bound<'_, PyAny>) = st.extract()?;
            let rows = indices("Jacobian rows", &floats(&rows)?)?;
            let cols = indices("Jacobian columns", &floats(&cols)?)?;
            let vals = floats(&self.nl.call_method1("jacobian", (self.array(x),))?)?;
            if rows.len() != cols.len() || vals.len() != rows.len() {
                return Err(PyValueError::new_err(format!(
                    "the Jacobian has {} rows, {} columns and {} values; they must \
                     be one per entry",
                    rows.len(),
                    cols.len(),
                    vals.len()
                )));
            }
            Ok(rows
                .into_iter()
                .zip(cols)
                .zip(vals)
                .map(|((r, c), v)| (r, c, v))
                .collect())
        })();
        self.keep(r)
    }

    fn objective_gradient(&mut self, x: &[Number]) -> Result<Vec<Number>, String> {
        let r = self
            .nl
            .call_method1("gradient", (self.array(x),))
            .and_then(|v| floats(&v));
        self.keep(r)
    }
}

#[allow(clippy::too_many_arguments)]
fn step_options(
    mode: &str,
    predictor_iter: usize,
    degeneracy: &str,
    degeneracy_iter: Option<usize>,
    corrector_iter: usize,
    bound_eps: Option<Number>,
    max_pdpert: Option<Number>,
) -> PyResult<StepOptions> {
    let mode = match mode {
        "linear" => Mode::Linear,
        "fix_relax" => Mode::FixRelax,
        "path" => Mode::Path,
        other => {
            return Err(PyValueError::new_err(format!(
                "mode must be 'linear', 'fix_relax' or 'path', got {other:?}"
            )));
        }
    };
    let degeneracy = match degeneracy {
        "directional" => Degeneracy::Directional,
        "one_sided" => Degeneracy::OneSided,
        "release_all" => Degeneracy::ReleaseAll,
        other => {
            return Err(PyValueError::new_err(format!(
                "degeneracy must be 'directional', 'one_sided' or 'release_all', got {other:?}"
            )));
        }
    };
    Ok(StepOptions {
        mode,
        predictor_iter,
        degeneracy,
        degeneracy_iter,
        corrector_iter,
        bound_eps,
        max_pdpert,
    })
}

fn step_finding(py: Python<'_>, f: &StepFinding) -> PyResult<PyObject> {
    Ok(match f {
        StepFinding::DegeneracyIterInert => ("degeneracy_iter_inert",).into_py(py),
        StepFinding::BoundEpsInert => ("bound_eps_inert",).into_py(py),
        StepFinding::DirectionalFallback { error } => (
            "directional_fallback",
            solver_error_to_py(error.clone())
                .value_bound(py)
                .str()?
                .to_string(),
        )
            .into_py(py),
        StepFinding::CorrectorWeak { report } => {
            ("corrector_weak", corrector_dict(py, report)?).into_py(py)
        }
        StepFinding::StillOutside {
            vars,
            pinned,
            changes,
            clamped,
        } => (
            "still_outside",
            vars.clone(),
            pinned.map(|(n, s)| (n, stop_name(s))),
            *changes,
            *clamped,
        )
            .into_py(py),
        StepFinding::LinearClamped { vars } => ("linear_clamped", vars.clone()).into_py(py),
    })
}

fn findings_list(py: Python<'_>, fs: &[StepFinding]) -> PyResult<Vec<PyObject>> {
    fs.iter().map(|f| step_finding(py, f)).collect()
}

/// Entries of a sparsity structure as indices, refusing anything that is
/// not a non-negative whole number rather than saturating it to one.
fn indices(what: &str, v: &[Number]) -> PyResult<Vec<usize>> {
    v.iter()
        .map(|&e| {
            if e >= 0.0 && e.fract() == 0.0 && e <= usize::MAX as Number {
                Ok(e as usize)
            } else {
                Err(PyValueError::new_err(format!(
                    "{what}: {e} is not an index"
                )))
            }
        })
        .collect()
}

/// The pins as the core's index type, refusing one that does not fit
/// rather than wrapping it onto another row.
fn pins_of(pins: &[i64]) -> PyResult<Vec<Index>> {
    pins.iter()
        .map(|&p| {
            Index::try_from(p)
                .map_err(|_| PyValueError::new_err(format!("pin {p} is not a row index")))
        })
        .collect()
}

/// `pounce_sens_core::step::solution` over the solver's held factor.
#[pyfunction]
#[pyo3(signature = (solver, nl, base_x, pins, deltas, mode, predictor_iter, degeneracy, degeneracy_iter, corrector_iter, bound_eps, max_pdpert, clamp))]
#[allow(clippy::too_many_arguments)]
pub fn _sens_solution<'py>(
    py: Python<'py>,
    solver: &PySolver,
    nl: &Bound<'py, PyAny>,
    base_x: &Bound<'py, PyAny>,
    pins: Vec<i64>,
    deltas: Vec<Number>,
    mode: &str,
    predictor_iter: usize,
    degeneracy: &str,
    degeneracy_iter: Option<usize>,
    corrector_iter: usize,
    bound_eps: Option<Number>,
    max_pdpert: Option<Number>,
    clamp: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let o = step_options(
        mode,
        predictor_iter,
        degeneracy,
        degeneracy_iter,
        corrector_iter,
        bound_eps,
        max_pdpert,
    )?;
    let model = PyModel::new(nl)?;
    let x = floats(base_x)?;
    let pins = pins_of(&pins)?;
    let out =
        solver.with_converged(|c| Ok(step::solution(c, &model, &x, &pins, &deltas, &o, clamp)))?;
    let d = PyDict::new_bound(py);
    d.set_item("findings", findings_list(py, &out.findings)?)?;
    match out.result {
        Ok(v) => {
            d.set_item("x", v.into_pyarray_bound(py))?;
            d.set_item("error", py.None())?;
        }
        Err(e) => d.set_item("error", step_error(py, e, model.err.take()))?,
    }
    Ok(d)
}

fn target(py: Python<'_>, t: &BoundTarget) -> PyObject {
    match t {
        BoundTarget::Variable(i) => ("variable", *i).into_py(py),
        BoundTarget::Row(j) => ("constraint", *j).into_py(py),
    }
}

/// `pounce_sens_core::step::solution_report` over the solver's held factor.
#[pyfunction]
#[pyo3(signature = (solver, nl, base_x, pins, deltas, mode, predictor_iter, degeneracy, degeneracy_iter, corrector_iter, bound_eps, max_pdpert, refine_activity))]
#[allow(clippy::too_many_arguments)]
pub fn _sens_solution_report<'py>(
    py: Python<'py>,
    solver: &PySolver,
    nl: &Bound<'py, PyAny>,
    base_x: &Bound<'py, PyAny>,
    pins: Vec<i64>,
    deltas: Vec<Number>,
    mode: &str,
    predictor_iter: usize,
    degeneracy: &str,
    degeneracy_iter: Option<usize>,
    corrector_iter: usize,
    bound_eps: Option<Number>,
    max_pdpert: Option<Number>,
    refine_activity: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let o = step_options(
        mode,
        predictor_iter,
        degeneracy,
        degeneracy_iter,
        corrector_iter,
        bound_eps,
        max_pdpert,
    )?;
    let mut model = PyModel::new(nl)?;
    let x = floats(base_x)?;
    let pins = pins_of(&pins)?;
    let out = solver.with_converged(|c| {
        Ok(step::solution_report(
            c,
            &mut model,
            &x,
            &pins,
            &deltas,
            &o,
            refine_activity,
        ))
    })?;
    let d = PyDict::new_bound(py);
    d.set_item("findings", findings_list(py, &out.findings)?)?;
    match out.result {
        Ok(r) => {
            d.set_item("alpha", r.alpha)?;
            d.set_item("first", r.first.map(|t| target(py, &t)))?;
            d.set_item("crossed", r.crossed)?;
            d.set_item("crossed_rows", r.crossed_rows)?;
            d.set_item("violation", r.violation)?;
            d.set_item("mu", r.mu)?;
            let vs: Vec<&str> = r.var_status.iter().map(|&s| status_name(s)).collect();
            let rs: Vec<&str> = r.row_status.iter().map(|&s| status_name(s)).collect();
            d.set_item("var_status", vs)?;
            d.set_item("row_status", rs)?;
            let refined: Vec<PyObject> = r
                .refined
                .iter()
                .map(|(t, b, a)| (target(py, t), status_name(*b), status_name(*a)).into_py(py))
                .collect();
            d.set_item("refined", refined)?;
            let entries: Vec<PyObject> = r.refined_entries.iter().map(|t| target(py, t)).collect();
            d.set_item("refined_entries", entries)?;
            d.set_item("perturbations", r.perturbations.map(|p| p.to_vec()))?;
            d.set_item("bounds_relaxed", r.bounds_relaxed)?;
            match &r.corrector {
                Some(c) => d.set_item("corrector", corrector_dict(py, c)?)?,
                None => d.set_item("corrector", py.None())?,
            }
            d.set_item("refine_stop", r.refine_stop.map(stop_name))?;
            d.set_item("error", py.None())?;
        }
        Err(e) => d.set_item("error", step_error(py, e, model.err.take()))?,
    }
    Ok(d)
}

/// `pounce_sens_core::step::active_set_changes` over the solver's held
/// factor.
#[pyfunction]
#[pyo3(signature = (solver, pins, deltas, predictor_iter, degeneracy, degeneracy_iter, max_pdpert))]
#[allow(clippy::too_many_arguments)]
pub fn _sens_active_set_changes<'py>(
    py: Python<'py>,
    solver: &PySolver,
    pins: Vec<i64>,
    deltas: Vec<Number>,
    predictor_iter: usize,
    degeneracy: &str,
    degeneracy_iter: Option<usize>,
    max_pdpert: Option<Number>,
) -> PyResult<Bound<'py, PyDict>> {
    let o = step_options(
        "path",
        predictor_iter,
        degeneracy,
        degeneracy_iter,
        0,
        None,
        max_pdpert,
    )?;
    let pins = pins_of(&pins)?;
    let out = solver.with_converged(|c| Ok(step::active_set_changes(c, &pins, &deltas, &o)))?;
    let d = PyDict::new_bound(py);
    d.set_item("findings", findings_list(py, &out.findings)?)?;
    match out.result {
        Ok(changes) => {
            let list = PyList::empty_bound(py);
            for ch in &changes {
                list.append(PyTuple::new_bound(
                    py,
                    [
                        ch.fraction.into_py(py),
                        target(py, &ch.target),
                        ch.lower.into_py(py),
                        ch.reaches.into_py(py),
                    ],
                ))?;
            }
            d.set_item("changes", list)?;
            d.set_item("error", py.None())?;
        }
        Err(e) => d.set_item("error", step_error(py, e, None))?,
    }
    Ok(d)
}

/// `pounce_sens_core::step::multiplier_row`: `("row", r)`, or a refusal,
/// `("no_block",)`, `("unclassified", exception)` or
/// `("not_strictly_active", status)`.
#[pyfunction]
pub fn _sens_multiplier_row(py: Python<'_>, solver: &PySolver, g: usize) -> PyResult<PyObject> {
    let r = solver.with_converged(|c| step::multiplier_row(c, g))?;
    Ok(match r {
        Ok(row) => ("row", row).into_py(py),
        Err(MultiplierRefusal::NoBlock) => ("no_block",).into_py(py),
        Err(MultiplierRefusal::Unclassified(e)) => ("unclassified", exception(py, e)).into_py(py),
        Err(MultiplierRefusal::NotStrictlyActive(s)) => {
            ("not_strictly_active", status_name(s)).into_py(py)
        }
    })
}

/// `pounce_sens_core::step::total_objective_derivative`.
#[pyfunction]
pub fn _sens_total_derivative(
    solver: &PySolver,
    grad: &Bound<'_, PyAny>,
    col: &Bound<'_, PyAny>,
) -> PyResult<Number> {
    let g = floats(grad)?;
    let c = floats(col)?;
    solver.with_converged(|conv| step::total_objective_derivative(conv, &g, &c))
}

/// `pounce_sens_core::stats::classify_ratio`, by name.
#[pyfunction]
pub fn _classify_ratio(r: Number, mu: Number) -> &'static str {
    status_name(stats::classify_ratio(r, mu))
}

/// `pounce_sens_core::dense::indefinite`.
#[pyfunction]
pub fn _indefinite(matrix: &Bound<'_, PyAny>) -> PyResult<bool> {
    pounce_sens_core::dense::indefinite(&dense_from_py(matrix)?).map_err(PyRuntimeError::new_err)
}

/// `pounce_sens_core::dense::rank_deficient`.
#[pyfunction]
pub fn _rank_deficient(matrix: &Bound<'_, PyAny>) -> PyResult<bool> {
    pounce_sens_core::dense::rank_deficient(&dense_from_py(matrix)?)
        .map_err(PyRuntimeError::new_err)
}

/// `pounce_sens_core::dense::pin_eigenvector_signs`, on a copy.
#[pyfunction]
pub fn _pin_eigenvector_signs<'py>(
    py: Python<'py>,
    vecs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<Number>>> {
    let mut m = dense_from_py(vecs)?;
    pounce_sens_core::dense::pin_eigenvector_signs(&mut m);
    dense_to_py(py, &m)
}

/// `pounce_sens_core::step::ratio_test`: the fraction and the index it is
/// reached at, `None` when no bound lies along the step.
#[pyfunction]
#[pyo3(signature = (base, step, lo, hi, live, on_bound, mu, tol))]
#[allow(clippy::too_many_arguments)]
pub fn _ratio_test(
    base: &Bound<'_, PyAny>,
    step: &Bound<'_, PyAny>,
    lo: &Bound<'_, PyAny>,
    hi: &Bound<'_, PyAny>,
    live: Option<Vec<bool>>,
    on_bound: Option<Vec<bool>>,
    mu: Number,
    tol: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Number, Option<usize>)> {
    let tol = match tol {
        Some(t) => Some(floats(t)?),
        None => None,
    };
    let (base, step, lo, hi) = (floats(base)?, floats(step)?, floats(lo)?, floats(hi)?);
    let n = base.len();
    for (what, len) in [
        ("step", Some(step.len())),
        ("lo", Some(lo.len())),
        ("hi", Some(hi.len())),
        ("live", live.as_ref().map(Vec::len)),
        ("on_bound", on_bound.as_ref().map(Vec::len)),
        ("tol", tol.as_ref().map(Vec::len)),
    ] {
        if let Some(len) = len
            && len != n
        {
            return Err(PyValueError::new_err(format!(
                "ratio test: {what} has {len} entries for a base of {n}"
            )));
        }
    }
    Ok(step::ratio_test(
        &base,
        &step,
        &lo,
        &hi,
        live.as_deref(),
        on_bound.as_deref(),
        mu,
        tol.as_deref(),
    ))
}
