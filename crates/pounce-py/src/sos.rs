//! PyO3 bindings for the sum-of-squares polynomial global optimizer
//! (`pounce-convex`'s `sos` module): `min p(x) s.t. gᵢ(x) ≥ 0, hⱼ(x) = 0`
//! solved by the SOS / Lasserre relaxation on the SDP cone, with a certified
//! lower bound and (when the moment matrix is flat) the global minimizers.
//!
//! Polynomials cross the FFI boundary as a list of `(exponent vector,
//! coefficient)` terms; the friendly `{exponent-tuple: coeff}` dict form is
//! handled in `python/pounce/sos.py`.

use numpy::IntoPyArray;
use pounce_convex::{PolyProblem, Polynomial, QpStatus, sos_minimize as core_sos_minimize};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn status_str(s: QpStatus) -> &'static str {
    match s {
        QpStatus::Optimal => "optimal",
        QpStatus::OptimalInaccurate => "optimal_inaccurate",
        QpStatus::PrimalInfeasible => "primal_infeasible",
        QpStatus::DualInfeasible => "dual_infeasible",
        QpStatus::IterationLimit => "iteration_limit",
        QpStatus::NumericalFailure => "numerical_failure",
    }
}

/// Validate that every term's exponent vector has length `n_vars` and build a
/// [`Polynomial`].
fn poly(n_vars: usize, terms: Vec<(Vec<usize>, f64)>, what: &str) -> PyResult<Polynomial> {
    for (e, _) in &terms {
        if e.len() != n_vars {
            return Err(PyValueError::new_err(format!(
                "{what}: exponent vector has length {}, expected n_vars = {n_vars}",
                e.len()
            )));
        }
    }
    Ok(Polynomial::new(n_vars, terms))
}

/// Globally minimize a polynomial via the SOS/Lasserre relaxation. Returns a
/// dict with `lower_bound`, `status`, `is_exact`, `num_minimizers`, and
/// `minimizers` (a list of length-`n_vars` arrays — the global optimizers,
/// populated when the moment matrix is flat).
#[pyfunction]
#[pyo3(signature = (n_vars, objective, inequalities=vec![], equalities=vec![], order=None))]
pub fn sos_minimize<'py>(
    py: Python<'py>,
    n_vars: usize,
    objective: Vec<(Vec<usize>, f64)>,
    inequalities: Vec<Vec<(Vec<usize>, f64)>>,
    equalities: Vec<Vec<(Vec<usize>, f64)>>,
    order: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut prob = PolyProblem::new(poly(n_vars, objective, "objective")?);
    prob.inequalities = inequalities
        .into_iter()
        .map(|t| poly(n_vars, t, "inequality"))
        .collect::<PyResult<_>>()?;
    prob.equalities = equalities
        .into_iter()
        .map(|t| poly(n_vars, t, "equality"))
        .collect::<PyResult<_>>()?;

    let sol = py.allow_threads(|| core_sos_minimize(&prob, order, backend));

    let d = PyDict::new_bound(py);
    d.set_item("lower_bound", sol.lower_bound)?;
    d.set_item("status", status_str(sol.status))?;
    d.set_item("is_exact", sol.is_exact)?;
    d.set_item("num_minimizers", sol.num_minimizers)?;
    let mins = PyList::empty_bound(py);
    for m in sol.minimizers {
        mins.append(m.into_pyarray_bound(py))?;
    }
    d.set_item("minimizers", mins)?;
    Ok(d)
}
