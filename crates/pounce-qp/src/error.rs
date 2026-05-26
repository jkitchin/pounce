//! Error and status types for the QP solver.

use std::fmt;

/// Terminal status of a QP solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpStatus {
    /// KKT residual and feasibility within tolerance.
    Optimal,
    /// Phase-1 elastic mode certified the QP as infeasible
    /// (residual elastic slacks are nonzero at the elastic
    /// solution).
    Infeasible,
    /// Descent direction of unbounded length found (only possible
    /// when the reduced Hessian is indefinite or negative semi-
    /// definite along a feasible ray).
    Unbounded,
    /// Iteration limit reached before convergence.
    MaxIter,
    /// Solver detected numerical breakdown (e.g., factor failure
    /// not recoverable by inertia correction).
    NumericalError,
}

impl fmt::Display for QpStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QpStatus::Optimal => write!(f, "optimal"),
            QpStatus::Infeasible => write!(f, "infeasible"),
            QpStatus::Unbounded => write!(f, "unbounded"),
            QpStatus::MaxIter => write!(f, "max-iter"),
            QpStatus::NumericalError => write!(f, "numerical-error"),
        }
    }
}

/// Hard errors — problems the solver cannot return any meaningful
/// solution for. Soft outcomes (max-iter, infeasible, unbounded) are
/// reported via [`QpStatus`] inside a successful
/// [`crate::QpSolution`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QpError {
    /// Problem-data dimensions disagree (e.g., `g.len() != n`).
    DimensionMismatch(String),
    /// A bound vector contains `bl > bu` for some index.
    InvertedBounds(String),
    /// Warm-start working set has the wrong length for the problem
    /// dimensions.
    WarmStartDimensionMismatch(String),
    /// Linear-solver backend reported a hard failure that cannot be
    /// recovered by the inertia / refactor logic.
    LinearSolverFailure(String),
    /// Feature required by this QP is not yet implemented in the
    /// current crate phase (e.g., one-sided inequality constraints
    /// before the working-set machinery lands).
    UnsupportedFeature(String),
}

impl fmt::Display for QpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QpError::DimensionMismatch(s) => write!(f, "dimension mismatch: {s}"),
            QpError::InvertedBounds(s) => write!(f, "inverted bounds: {s}"),
            QpError::WarmStartDimensionMismatch(s) => {
                write!(f, "warm-start dimension mismatch: {s}")
            }
            QpError::LinearSolverFailure(s) => write!(f, "linear solver failure: {s}"),
            QpError::UnsupportedFeature(s) => write!(f, "unsupported feature: {s}"),
        }
    }
}

impl std::error::Error for QpError {}
