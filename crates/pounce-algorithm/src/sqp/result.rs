//! `SqpResult` / `SqpStatus` / `SqpError` — return types for
//! `SqpAlgorithm::optimize`.

use pounce_common::Number;
use pounce_qp::{QpError, WorkingSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqpStatus {
    /// KKT residuals all below their tolerances.
    Optimal,
    /// `max_iter` reached without convergence.
    MaxIter,
    /// QP subproblem returned an `Infeasible` status (elastic
    /// mode certified the QP infeasible).
    InfeasibleSubproblem,
    /// Line search failed to find an acceptable step (Phase 5b
    /// commit 5+; not produced by the c3 always-full-step loop).
    LineSearchFailed,
}

impl fmt::Display for SqpStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SqpStatus::Optimal => write!(f, "optimal"),
            SqpStatus::MaxIter => write!(f, "max-iter"),
            SqpStatus::InfeasibleSubproblem => write!(f, "infeasible-subproblem"),
            SqpStatus::LineSearchFailed => write!(f, "line-search-failed"),
        }
    }
}

#[derive(Debug)]
pub enum SqpError {
    /// Hard QP-solver failure (singular, dimension mismatch, etc.).
    QpFailure(QpError),
    /// Caller-supplied dimensions disagree.
    DimensionMismatch(String),
}

impl fmt::Display for SqpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SqpError::QpFailure(e) => write!(f, "QP subproblem failure: {e}"),
            SqpError::DimensionMismatch(s) => write!(f, "dimension mismatch: {s}"),
        }
    }
}

impl From<QpError> for SqpError {
    fn from(e: QpError) -> Self {
        SqpError::QpFailure(e)
    }
}

#[derive(Debug, Clone)]
pub struct SqpResult {
    pub x: Vec<Number>,
    pub lambda_g: Vec<Number>,
    pub lambda_x: Vec<Number>,
    pub obj: Number,
    pub status: SqpStatus,
    pub n_iter: u32,
    pub n_qp_solves: u32,
    /// Final stationarity residual (max-norm of `∇f + Jᵀ λ_g + λ_x`).
    pub final_stationarity: Number,
    /// Final constraint violation (max-norm of `c(x*)` for
    /// equalities plus bound-violation slack).
    pub final_constr_viol: Number,
    /// Final QP working set, suitable for warm-starting the next
    /// `optimize_with_warm_start` call (§6 design-note contract).
    /// `None` only when no QP was solved (e.g. cold-start declared
    /// the iterate optimal at the very first KKT check).
    pub working_set: Option<WorkingSet>,
}
