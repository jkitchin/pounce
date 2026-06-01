//! PyO3 bindings for the spatial branch-and-bound global optimizer
//! (`pounce-global`): `min f(x) s.t. cl_j ≤ g_j(x) ≤ cu_j, lo ≤ x ≤ hi` for
//! factorable nonconvex `f`/`g`, solved to a certified global optimum.
//!
//! Expressions cross the FFI as a flat tape: a list of `(tag, a, b, c)` tuples,
//! where `a`/`b` are slot indices into earlier entries and `c` is a constant /
//! exponent. The friendly builder lives in `python/pounce/global_opt.py`.

use numpy::IntoPyArray;
use pounce_feral::FeralSolverInterface;
use pounce_global::{
    solve_global as solve_global_core, Constraint, GlobalOptions, GlobalProblem, GlobalStatus,
};
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_nlp::{FbbtOp, FbbtTape};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A flat expression tape crossing the FFI: `(tag, slot_a, slot_b, const)`.
type OpTape = Vec<(u8, i64, i64, f64)>;
/// A constraint as `(body tape, lower bound, upper bound)`.
type ConTape = (OpTape, f64, f64);

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn status_str(s: GlobalStatus) -> &'static str {
    match s {
        GlobalStatus::Optimal => "optimal",
        GlobalStatus::Infeasible => "infeasible",
        GlobalStatus::NodeLimit => "node_limit",
    }
}

/// Decode a flat op tape `(tag, a, b, c)` into an [`FbbtTape`], validating that
/// operand slots reference strictly-earlier entries and variable indices are in
/// range. Tags mirror `python/pounce/global_opt.py`.
fn tape_from_ops(ops: &[(u8, i64, i64, f64)], n_vars: usize) -> PyResult<FbbtTape> {
    let mut out: Vec<FbbtOp> = Vec::with_capacity(ops.len());
    for (k, &(tag, a, b, c)) in ops.iter().enumerate() {
        // Operand slots must be earlier entries.
        let slot = |idx: i64| -> PyResult<usize> {
            if idx < 0 || idx as usize >= k {
                Err(PyValueError::new_err(format!(
                    "op {k}: operand slot {idx} out of range (must be 0..{k})"
                )))
            } else {
                Ok(idx as usize)
            }
        };
        let op = match tag {
            0 => FbbtOp::Const(c),
            1 => {
                if a < 0 || a as usize >= n_vars {
                    return Err(PyValueError::new_err(format!(
                        "op {k}: variable index {a} out of range (n_vars = {n_vars})"
                    )));
                }
                FbbtOp::Var(a as usize)
            }
            2 => FbbtOp::Add(slot(a)?, slot(b)?),
            3 => FbbtOp::Sub(slot(a)?, slot(b)?),
            4 => FbbtOp::Mul(slot(a)?, slot(b)?),
            5 => FbbtOp::Div(slot(a)?, slot(b)?),
            6 => {
                if c < 0.0 || c.fract() != 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "op {k}: power exponent {c} must be a non-negative integer"
                    )));
                }
                FbbtOp::PowInt(slot(a)?, c as u32)
            }
            7 => FbbtOp::Neg(slot(a)?),
            8 => FbbtOp::Sqrt(slot(a)?),
            9 => FbbtOp::Exp(slot(a)?),
            10 => FbbtOp::Ln(slot(a)?),
            11 => FbbtOp::Abs(slot(a)?),
            12 => FbbtOp::Sin(slot(a)?),
            13 => FbbtOp::Cos(slot(a)?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "op {k}: unknown tag {other}"
                )))
            }
        };
        out.push(op);
    }
    Ok(FbbtTape { ops: out })
}

/// Globally minimize a factorable nonconvex problem by spatial branch-and-bound.
/// Returns a dict with `status`, `x`, `objective` (upper bound), `lower_bound`,
/// `gap`, and `nodes`.
#[pyfunction]
#[pyo3(signature = (
    n_vars, x_lo, x_hi, objective, constraints=vec![], *,
    abs_gap=1e-6, rel_gap=1e-6, feas_tol=1e-6, box_tol=1e-7, max_nodes=5000,
    local_solve_iters=50, sandwich_rounds=4, obbt_passes=2, alphabb_cuts=1,
    rlt=true, multilinear=true, threads=1
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_global<'py>(
    py: Python<'py>,
    n_vars: usize,
    x_lo: Vec<f64>,
    x_hi: Vec<f64>,
    objective: OpTape,
    constraints: Vec<ConTape>,
    abs_gap: f64,
    rel_gap: f64,
    feas_tol: f64,
    box_tol: f64,
    max_nodes: usize,
    local_solve_iters: usize,
    sandwich_rounds: usize,
    obbt_passes: usize,
    alphabb_cuts: usize,
    rlt: bool,
    multilinear: bool,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if x_lo.len() != n_vars || x_hi.len() != n_vars {
        return Err(PyValueError::new_err(format!(
            "x_lo / x_hi must have length n_vars = {n_vars}"
        )));
    }
    let prob = GlobalProblem {
        n_vars,
        x_lo,
        x_hi,
        objective: tape_from_ops(&objective, n_vars)?,
        constraints: constraints
            .into_iter()
            .map(|(ops, lo, hi)| {
                Ok(Constraint {
                    tape: tape_from_ops(&ops, n_vars)?,
                    lo,
                    hi,
                })
            })
            .collect::<PyResult<_>>()?,
    };
    let opts = GlobalOptions {
        abs_gap,
        rel_gap,
        feas_tol,
        box_tol,
        max_nodes,
        local_solve_iters,
        sandwich_rounds,
        obbt_passes,
        alphabb_cuts,
        rlt,
        multilinear,
        threads,
        ..GlobalOptions::default()
    };

    let sol = py.allow_threads(|| solve_global_core(&prob, &opts, backend));

    let d = PyDict::new_bound(py);
    d.set_item("status", status_str(sol.status))?;
    d.set_item("objective", sol.objective)?;
    d.set_item("lower_bound", sol.lower_bound)?;
    d.set_item("gap", sol.gap())?;
    d.set_item("nodes", sol.nodes)?;
    d.set_item("x", sol.x.into_pyarray_bound(py))?;
    Ok(d)
}
