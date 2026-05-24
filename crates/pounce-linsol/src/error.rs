//! Error type for the public [`crate::Factorization`] API.

use crate::status::ESymSolverStatus;

/// Outcome of a [`crate::Factorization`] operation.
///
/// Wraps the lower-level [`ESymSolverStatus`] with `Success` removed
/// (success is expressed as `Result::Ok`). Variants mirror upstream
/// Ipopt's status enum so a caller can map back if needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FactorizationError {
    /// Matrix is singular; factor was aborted.
    Singular,
    /// Backend's reported negative-eigenvalue count did not match the
    /// caller's expectation.
    WrongInertia,
    /// Unrecoverable backend error.
    FatalError,
}

impl FactorizationError {
    /// Convert a backend status into a `Result`. `Success` becomes
    /// `Ok(())`; `CallAgain` is treated as a fatal error here because
    /// the public API drives the retry loop internally — a leaked
    /// `CallAgain` indicates a backend bug.
    pub(crate) fn from_status(status: ESymSolverStatus) -> Result<(), Self> {
        match status {
            ESymSolverStatus::Success => Ok(()),
            ESymSolverStatus::Singular => Err(Self::Singular),
            ESymSolverStatus::WrongInertia => Err(Self::WrongInertia),
            ESymSolverStatus::CallAgain | ESymSolverStatus::FatalError => {
                Err(Self::FatalError)
            }
        }
    }
}

impl std::fmt::Display for FactorizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Singular => write!(f, "matrix is singular"),
            Self::WrongInertia => write!(f, "factorization inertia did not match expectation"),
            Self::FatalError => write!(f, "fatal linear-solver error"),
        }
    }
}

impl std::error::Error for FactorizationError {}
