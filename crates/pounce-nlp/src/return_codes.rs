//! Application-level return codes.
//!
//! Mirrors `Interfaces/IpReturnCodes.{h,hpp}` and `IpReturnCodes_inc.h`.
//! The integer values **must** match upstream — `pounce-cinterface`
//! uses `#[repr(i32)]` so the C ABI emits identical numeric codes for
//! drop-in compatibility with PyIpopt / cyipopt / JuMP.

use pounce_common::types::Index;

/// Mirrors `enum ApplicationReturnStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(i32)]
pub enum ApplicationReturnStatus {
    SolveSucceeded = 0,
    SolvedToAcceptableLevel = 1,
    InfeasibleProblemDetected = 2,
    SearchDirectionBecomesTooSmall = 3,
    DivergingIterates = 4,
    UserRequestedStop = 5,
    FeasiblePointFound = 6,

    MaximumIterationsExceeded = -1,
    RestorationFailed = -2,
    ErrorInStepComputation = -3,
    MaximumCpuTimeExceeded = -4,
    MaximumWallTimeExceeded = -5,

    NotEnoughDegreesOfFreedom = -10,
    InvalidProblemDefinition = -11,
    InvalidOption = -12,
    InvalidNumberDetected = -13,

    UnrecoverableException = -100,
    NonIpoptExceptionThrown = -101,
    InsufficientMemory = -102,
    InternalError = -199,
}

impl ApplicationReturnStatus {
    pub fn as_int(self) -> Index {
        self as Index
    }
}

/// Mirrors `enum AlgorithmMode`. Exposed in `intermediate_callback`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum AlgorithmMode {
    RegularMode = 0,
    RestorationPhaseMode = 1,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// From `IpReturnCodes_inc.h` — these values are the C ABI
    /// contract for `pounce-cinterface`. Never change them.
    #[test]
    fn integer_values_match_upstream() {
        assert_eq!(ApplicationReturnStatus::SolveSucceeded.as_int(), 0);
        assert_eq!(ApplicationReturnStatus::SolvedToAcceptableLevel.as_int(), 1);
        assert_eq!(
            ApplicationReturnStatus::InfeasibleProblemDetected.as_int(),
            2
        );
        assert_eq!(
            ApplicationReturnStatus::SearchDirectionBecomesTooSmall.as_int(),
            3
        );
        assert_eq!(ApplicationReturnStatus::DivergingIterates.as_int(), 4);
        assert_eq!(ApplicationReturnStatus::UserRequestedStop.as_int(), 5);
        assert_eq!(ApplicationReturnStatus::FeasiblePointFound.as_int(), 6);

        assert_eq!(
            ApplicationReturnStatus::MaximumIterationsExceeded.as_int(),
            -1
        );
        assert_eq!(ApplicationReturnStatus::RestorationFailed.as_int(), -2);
        assert_eq!(
            ApplicationReturnStatus::ErrorInStepComputation.as_int(),
            -3
        );
        assert_eq!(ApplicationReturnStatus::MaximumCpuTimeExceeded.as_int(), -4);
        assert_eq!(ApplicationReturnStatus::MaximumWallTimeExceeded.as_int(), -5);

        assert_eq!(
            ApplicationReturnStatus::NotEnoughDegreesOfFreedom.as_int(),
            -10
        );
        assert_eq!(
            ApplicationReturnStatus::InvalidProblemDefinition.as_int(),
            -11
        );
        assert_eq!(ApplicationReturnStatus::InvalidOption.as_int(), -12);
        assert_eq!(ApplicationReturnStatus::InvalidNumberDetected.as_int(), -13);

        assert_eq!(
            ApplicationReturnStatus::UnrecoverableException.as_int(),
            -100
        );
        assert_eq!(
            ApplicationReturnStatus::NonIpoptExceptionThrown.as_int(),
            -101
        );
        assert_eq!(ApplicationReturnStatus::InsufficientMemory.as_int(), -102);
        assert_eq!(ApplicationReturnStatus::InternalError.as_int(), -199);

        assert_eq!(AlgorithmMode::RegularMode as i32, 0);
        assert_eq!(AlgorithmMode::RestorationPhaseMode as i32, 1);
    }
}
