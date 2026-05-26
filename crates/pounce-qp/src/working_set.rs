//! Working-set representation — the discrete state carried across
//! QP solves to implement parametric warm starting.
//!
//! Each bound slot and each general-constraint slot has a small
//! status enum. The pair `(bounds, constraints)` is the only piece
//! of discrete state the QP solver hands back to the caller (and
//! accepts back as a warm start).

use crate::error::QpError;

/// Status of a single primal-variable bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundStatus {
    /// Not in the working set; `xl < x < xu` strictly.
    Inactive,
    /// Active at the lower bound; `x = xl`, dual `≥ 0`.
    AtLower,
    /// Active at the upper bound; `x = xu`, dual `≤ 0`.
    AtUpper,
    /// `xl = xu`; the variable is fixed and always in the working
    /// set with no sign constraint on the dual.
    Fixed,
}

impl BoundStatus {
    pub fn is_active(self) -> bool {
        !matches!(self, BoundStatus::Inactive)
    }
}

/// Status of a single general constraint `bl ≤ aᵀx ≤ bu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsStatus {
    /// Not in the working set; `bl < aᵀx < bu` strictly.
    Inactive,
    /// Active at the lower bound; `aᵀx = bl`, dual `≥ 0`.
    AtLower,
    /// Active at the upper bound; `aᵀx = bu`, dual `≤ 0`.
    AtUpper,
    /// `bl = bu`; the row is an equality and always in the working
    /// set with no sign constraint on the dual.
    Equality,
}

impl ConsStatus {
    pub fn is_active(self) -> bool {
        !matches!(self, ConsStatus::Inactive)
    }
}

/// The working set for a QP of dimension `n` with `m` general
/// constraints. `bounds.len() == n`, `constraints.len() == m`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkingSet {
    pub bounds: Vec<BoundStatus>,
    pub constraints: Vec<ConsStatus>,
}

impl WorkingSet {
    /// All-inactive working set sized to `(n, m)`. This is the cold-
    /// start seed handed to the phase-1 elastic-mode QP.
    pub fn cold(n: usize, m: usize) -> Self {
        Self {
            bounds: vec![BoundStatus::Inactive; n],
            constraints: vec![ConsStatus::Inactive; m],
        }
    }

    pub fn n(&self) -> usize {
        self.bounds.len()
    }

    pub fn m(&self) -> usize {
        self.constraints.len()
    }

    /// Count of active bounds plus active constraints (the dimension
    /// of the KKT block currently driving the EQP step).
    pub fn active_count(&self) -> usize {
        self.bounds.iter().filter(|s| s.is_active()).count()
            + self.constraints.iter().filter(|s| s.is_active()).count()
    }

    /// Reject working sets whose dimensions disagree with the
    /// problem they will be applied to. Called by the solver before
    /// consuming a user-supplied warm start.
    pub fn validate_dims(&self, n: usize, m: usize) -> Result<(), QpError> {
        if self.bounds.len() != n {
            return Err(QpError::WarmStartDimensionMismatch(format!(
                "bounds.len() = {} but problem n = {n}",
                self.bounds.len()
            )));
        }
        if self.constraints.len() != m {
            return Err(QpError::WarmStartDimensionMismatch(format!(
                "constraints.len() = {} but problem m = {m}",
                self.constraints.len()
            )));
        }
        Ok(())
    }
}
