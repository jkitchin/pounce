//! POUNCE exceptions.
//!
//! Mirrors `Common/IpException.hpp`. Ipopt uses inheritance to
//! distinguish exception types and a `DECLARE_STD_EXCEPTION` macro
//! to name them; we use a single `SolverException` struct carrying a
//! `kind: ExceptionKind` enum.
//!
//! The variant names are byte-identical to the upstream class names
//! (`TINY_STEP_DETECTED`, `RESTORATION_FAILED`, ...) so that
//! `ReportException` formatting matches upstream when we eventually
//! Display them.

use crate::types::Index;
use std::fmt;

/// All exception kinds Ipopt declares via `DECLARE_STD_EXCEPTION` in
/// `src/{Common,Algorithm,LinAlg,Interfaces,Apps}/`. Names match upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum ExceptionKind {
    // Common
    OPTION_INVALID,
    OPTION_ALREADY_REGISTERED,
    DYNAMIC_LIBRARY_FAILURE,
    // LinAlg
    UNIMPLEMENTED_LINALG_METHOD_CALLED,
    UNKNOWN_MATRIX_TYPE,
    UNKNOWN_VECTOR_TYPE,
    LAPACK_NOT_INCLUDED,
    METADATA_ERROR,
    // Algorithm — line search / step
    TINY_STEP_DETECTED,
    ACCEPTABLE_POINT_REACHED,
    STEP_COMPUTATION_FAILED,
    LOCALLY_INFEASIBLE,
    FEASIBILITY_PROBLEM_SOLVED,
    // Algorithm — restoration
    RESTORATION_FAILED,
    RESTORATION_CONVERGED_TO_FEASIBLE_POINT,
    RESTORATION_MAXITER_EXCEEDED,
    RESTORATION_CPUTIME_EXCEEDED,
    RESTORATION_WALLTIME_EXCEEDED,
    RESTORATION_USER_STOP,
    // Linear solvers / scaling
    FATAL_ERROR_IN_LINEAR_SOLVER,
    ERROR_IN_LINEAR_SCALING_METHOD,
    NONPOSITIVE_SCALING_FACTOR,
    USER_SCALING_NOT_IMPLEMENTED,
    // NLP / TNLP
    INVALID_NLP,
    INVALID_TNLP,
    INVALID_STDINTERFACE_NLP,
    INVALID_WARMSTART,
    INCONSISTENT_BOUNDS,
    TOO_FEW_DOF,
    ERROR_IN_TNLP_DERIVATIVE_TEST,
    // Application
    IPOPT_APPLICATION_ERROR,
    FAILED_INITIALIZATION,
    INTERNAL_ABORT,
    // Misc
    ERROR_CONVERTING_STRING_TO_ENUM,
}

impl ExceptionKind {
    /// Class name as Ipopt prints it in `ReportException` (`type_` field).
    pub fn name(self) -> &'static str {
        use ExceptionKind::*;
        match self {
            OPTION_INVALID => "OPTION_INVALID",
            OPTION_ALREADY_REGISTERED => "OPTION_ALREADY_REGISTERED",
            DYNAMIC_LIBRARY_FAILURE => "DYNAMIC_LIBRARY_FAILURE",
            UNIMPLEMENTED_LINALG_METHOD_CALLED => "UNIMPLEMENTED_LINALG_METHOD_CALLED",
            UNKNOWN_MATRIX_TYPE => "UNKNOWN_MATRIX_TYPE",
            UNKNOWN_VECTOR_TYPE => "UNKNOWN_VECTOR_TYPE",
            LAPACK_NOT_INCLUDED => "LAPACK_NOT_INCLUDED",
            METADATA_ERROR => "METADATA_ERROR",
            TINY_STEP_DETECTED => "TINY_STEP_DETECTED",
            ACCEPTABLE_POINT_REACHED => "ACCEPTABLE_POINT_REACHED",
            STEP_COMPUTATION_FAILED => "STEP_COMPUTATION_FAILED",
            LOCALLY_INFEASIBLE => "LOCALLY_INFEASIBLE",
            FEASIBILITY_PROBLEM_SOLVED => "FEASIBILITY_PROBLEM_SOLVED",
            RESTORATION_FAILED => "RESTORATION_FAILED",
            RESTORATION_CONVERGED_TO_FEASIBLE_POINT => "RESTORATION_CONVERGED_TO_FEASIBLE_POINT",
            RESTORATION_MAXITER_EXCEEDED => "RESTORATION_MAXITER_EXCEEDED",
            RESTORATION_CPUTIME_EXCEEDED => "RESTORATION_CPUTIME_EXCEEDED",
            RESTORATION_WALLTIME_EXCEEDED => "RESTORATION_WALLTIME_EXCEEDED",
            RESTORATION_USER_STOP => "RESTORATION_USER_STOP",
            FATAL_ERROR_IN_LINEAR_SOLVER => "FATAL_ERROR_IN_LINEAR_SOLVER",
            ERROR_IN_LINEAR_SCALING_METHOD => "ERROR_IN_LINEAR_SCALING_METHOD",
            NONPOSITIVE_SCALING_FACTOR => "NONPOSITIVE_SCALING_FACTOR",
            USER_SCALING_NOT_IMPLEMENTED => "USER_SCALING_NOT_IMPLEMENTED",
            INVALID_NLP => "INVALID_NLP",
            INVALID_TNLP => "INVALID_TNLP",
            INVALID_STDINTERFACE_NLP => "INVALID_STDINTERFACE_NLP",
            INVALID_WARMSTART => "INVALID_WARMSTART",
            INCONSISTENT_BOUNDS => "INCONSISTENT_BOUNDS",
            TOO_FEW_DOF => "TOO_FEW_DOF",
            ERROR_IN_TNLP_DERIVATIVE_TEST => "ERROR_IN_TNLP_DERIVATIVE_TEST",
            IPOPT_APPLICATION_ERROR => "IPOPT_APPLICATION_ERROR",
            FAILED_INITIALIZATION => "FAILED_INITIALIZATION",
            INTERNAL_ABORT => "INTERNAL_ABORT",
            ERROR_CONVERTING_STRING_TO_ENUM => "ERROR_CONVERTING_STRING_TO_ENUM",
        }
    }
}

impl fmt::Display for ExceptionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A single exception value carrying kind + message + source location.
///
/// Equivalent to `IpoptException`; raise via [`throw`] or
/// [`assert_or_throw`] to capture file/line.
#[derive(Debug, Clone)]
pub struct SolverException {
    pub kind: ExceptionKind,
    pub message: String,
    pub file: &'static str,
    pub line: Index,
}

impl SolverException {
    pub fn new(
        kind: ExceptionKind,
        message: impl Into<String>,
        file: &'static str,
        line: Index,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            file,
            line,
        }
    }
}

impl fmt::Display for SolverException {
    /// Matches the format used in `IpoptException::ReportException`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Exception of type: {} in file \"{}\" at line {}:\n Exception message: {}",
            self.kind, self.file, self.line, self.message
        )
    }
}

impl std::error::Error for SolverException {}

/// Macro replacement for Ipopt's `THROW_EXCEPTION(kind, msg)`. Produces
/// a `Result::Err(SolverException)` using `file!()`/`line!()`.
#[macro_export]
macro_rules! throw {
    ($kind:expr_2021, $msg:expr_2021) => {
        return ::core::result::Result::Err($crate::exception::SolverException::new(
            $kind,
            $msg,
            file!(),
            line!() as $crate::types::Index,
        ))
    };
}

/// Macro replacement for `ASSERT_EXCEPTION(cond, kind, msg)`.
#[macro_export]
macro_rules! assert_exc {
    ($cond:expr_2021, $kind:expr_2021, $msg:expr_2021) => {
        if !($cond) {
            let mut newmsg = ::std::string::String::from(stringify!($cond));
            newmsg.push_str(" evaluated false: ");
            newmsg.push_str($msg);
            return ::core::result::Result::Err($crate::exception::SolverException::new(
                $kind,
                newmsg,
                file!(),
                line!() as $crate::types::Index,
            ));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_matches_upstream_format() {
        let e = SolverException::new(
            ExceptionKind::OPTION_INVALID,
            "bad option",
            "src/foo.rs",
            42,
        );
        let s = format!("{}", e);
        assert!(s.contains("Exception of type: OPTION_INVALID"));
        assert!(s.contains("src/foo.rs"));
        assert!(s.contains("line 42"));
        assert!(s.contains("Exception message: bad option"));
    }

    #[test]
    fn names_all_round_trip() {
        for k in [
            ExceptionKind::OPTION_INVALID,
            ExceptionKind::TINY_STEP_DETECTED,
            ExceptionKind::RESTORATION_FAILED,
            ExceptionKind::LOCALLY_INFEASIBLE,
        ] {
            assert!(!k.name().is_empty());
            assert_eq!(format!("{k}"), k.name());
        }
    }
}
