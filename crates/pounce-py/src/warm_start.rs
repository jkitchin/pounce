//! Python binding for `pounce_algorithm::sqp::classify_working_set` —
//! the bridge from IPM-converged multipliers to an SQP warm-start
//! working set (Phase 5c §7.5).
//!
//! Exposed at module level as `pounce.classify_working_set(...)`, so
//! a parametric-continuation Python user can pass the
//! `info["mult_g"] / info["mult_x_L"] / info["mult_x_U"]`
//! arrays from a prior IPM solve directly into the next SQP
//! `Problem.solve(..., working_set=ws)` call.

use pounce_common::types::Number;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Classify the active set at `(x, λ_g, z_l, z_u)`. Returns a
/// `(bounds, constraints)` tuple of `numpy.ndarray` int8 arrays
/// with the same encoding `Problem.solve(..., working_set=…)`
/// expects:
///
///   0 = Inactive,
///   1 = AtLower (active at lower bound),
///   2 = AtUpper,
///   3 = Fixed (variables) or Equality (constraints).
///
/// Inputs:
/// - `x`, `x_l`, `x_u`: length `n` primal + bounds.
/// - `g`, `g_l`, `g_u`: length `m` constraint values + bounds.
/// - `lambda_g`: length `m` constraint multipliers.
/// - `z_l`, `z_u`: length `n` bound multipliers (non-negative each).
/// - `m_eq`: number of equality rows at the start of `g_l`/`g_u`
///   (unconditionally classified as `Equality`).
/// - `mult_tol`, `primal_tol`: thresholds; sensible defaults are
///   `1e-8` and `1e-6` respectively.
#[pyfunction]
#[pyo3(signature = (x, x_l, x_u, g, g_l, g_u, lambda_g, z_l, z_u, m_eq, mult_tol=1e-8, primal_tol=1e-6))]
#[allow(clippy::too_many_arguments)]
pub fn classify_working_set<'py>(
    py: Python<'py>,
    x: Py<PyAny>,
    x_l: Py<PyAny>,
    x_u: Py<PyAny>,
    g: Py<PyAny>,
    g_l: Py<PyAny>,
    g_u: Py<PyAny>,
    lambda_g: Py<PyAny>,
    z_l: Py<PyAny>,
    z_u: Py<PyAny>,
    m_eq: usize,
    mult_tol: Number,
    primal_tol: Number,
) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
    let x_v = extract_f64_vec(py, &x, "x")?;
    let x_l_v = extract_f64_vec(py, &x_l, "x_l")?;
    let x_u_v = extract_f64_vec(py, &x_u, "x_u")?;
    let g_v = extract_f64_vec(py, &g, "g")?;
    let g_l_v = extract_f64_vec(py, &g_l, "g_l")?;
    let g_u_v = extract_f64_vec(py, &g_u, "g_u")?;
    let lambda_g_v = extract_f64_vec(py, &lambda_g, "lambda_g")?;
    let z_l_v = extract_f64_vec(py, &z_l, "z_l")?;
    let z_u_v = extract_f64_vec(py, &z_u, "z_u")?;

    let n = x_v.len();
    let m = g_v.len();
    let lengths = [
        ("x_l", x_l_v.len(), n),
        ("x_u", x_u_v.len(), n),
        ("z_l", z_l_v.len(), n),
        ("z_u", z_u_v.len(), n),
        ("g_l", g_l_v.len(), m),
        ("g_u", g_u_v.len(), m),
        ("lambda_g", lambda_g_v.len(), m),
    ];
    for (name, got, want) in lengths {
        if got != want {
            return Err(PyValueError::new_err(format!(
                "{name}.len() = {got} but expected {want}"
            )));
        }
    }
    if m_eq > m {
        return Err(PyValueError::new_err(format!(
            "m_eq = {m_eq} exceeds m = {m}"
        )));
    }

    // Pack λ_x = z_l − z_u (the SQP-side packed signed bound mult).
    let lambda_x: Vec<Number> = z_l_v.iter().zip(z_u_v.iter()).map(|(l, u)| l - u).collect();

    let ws = pounce_algorithm::sqp::classify_working_set(
        &lambda_x,
        &lambda_g_v,
        m_eq,
        &x_v,
        &x_l_v,
        &x_u_v,
        &g_v,
        &g_l_v,
        &g_u_v,
        mult_tol,
        primal_tol,
    );
    let bounds_vec: Vec<i8> = ws
        .bounds
        .iter()
        .map(|s| match s {
            pounce_qp::BoundStatus::Inactive => 0,
            pounce_qp::BoundStatus::AtLower => 1,
            pounce_qp::BoundStatus::AtUpper => 2,
            pounce_qp::BoundStatus::Fixed => 3,
        })
        .collect();
    let cons_vec: Vec<i8> = ws
        .constraints
        .iter()
        .map(|s| match s {
            pounce_qp::ConsStatus::Inactive => 0,
            pounce_qp::ConsStatus::AtLower => 1,
            pounce_qp::ConsStatus::AtUpper => 2,
            pounce_qp::ConsStatus::Equality => 3,
        })
        .collect();
    use numpy::IntoPyArray;
    let b_arr = bounds_vec.into_pyarray_bound(py).into_any();
    let c_arr = cons_vec.into_pyarray_bound(py).into_any();
    Ok(pyo3::types::PyTuple::new_bound(py, &[b_arr, c_arr]))
}

fn extract_f64_vec(py: Python<'_>, val: &Py<PyAny>, what: &str) -> PyResult<Vec<Number>> {
    let bound = val.bind(py);
    bound
        .extract()
        .map_err(|e| PyValueError::new_err(format!("{what}: cannot extract f64 sequence: {e}")))
}
