//! Working-set representation — the discrete state carried across
//! QP solves to implement parametric warm starting.
//!
//! Each bound slot and each general-constraint slot has a small
//! status enum. The pair `(bounds, constraints)` is the only piece
//! of discrete state the QP solver hands back to the caller (and
//! accepts back as a warm start).

use crate::error::QpError;
use crate::options::QpOptions;
use crate::problem::QpProblem;
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};

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

    /// A copy of this working set made **well-formed for `qp`**.
    ///
    /// [`ConsStatus::Equality`] and [`BoundStatus::Fixed`] are not just
    /// "active" markers. They are assertions about the *problem's* bound
    /// topology — `bl == bu`, `xl == xu` — and they carry the semantics that
    /// go with it: the row/bound is always in the working set, its multiplier
    /// is unrestricted in sign, and no drop test can ever remove it (the drop
    /// score is hard-wired to `0.0`, "never drop"). Carried onto a problem
    /// where the row is no longer an equality, that is not a stale *hint*, it
    /// is a false statement the solver cannot walk back.
    ///
    /// The failure it produces is a wrong answer, not a slow one.
    /// `pin_working_set` pins an `Equality` row to `qp.bl[i]`, so a row that
    /// was `x == 1` and is now `x <= 2` gets pinned to the *lower* bound the
    /// new problem does not have — the `-1e20` infinity sentinel. The iterate
    /// lands at `-1e19`, which is perfectly feasible for `x <= 2`, so the M5
    /// audit passes it, no drop test can reconsider it, and the solve reports
    /// `Optimal` at an objective of `1e38`. Same shape through `Fixed` when a
    /// previously-pinned variable is freed. Both reproduced in
    /// `tests/homotopy.rs`; found in review of gh #602.
    ///
    /// What separates the two dangerous statuses from the rest is
    /// **droppability**. An `AtLower` pinned to a lower bound the new problem
    /// does not have lands on the same `-1e20` sentinel — but the dual ratio
    /// test sees a multiplier of the wrong sign and removes it, so the damage
    /// is iterations. `Equality` and `Fixed` have no such escape.
    ///
    /// So: re-derive the two topology-dependent statuses from `qp` itself, and
    /// drop any remaining status that names a bound `qp` does not have. What
    /// survives is the part of the hint that is only ever a guess about *which
    /// constraints bind* — which is what a working set is for, and which costs
    /// at most iterations when it is wrong.
    ///
    /// The predicates match the ones the solver applies when it builds a
    /// working set itself: exact `bl == bu` for rows (as
    /// `is_all_equality_constraints` and `solve_general` both use), and
    /// both-finite-within-`feas_tol` for bounds.
    pub(crate) fn reconciled_with(&self, qp: &QpProblem<'_>, opts: &QpOptions) -> WorkingSet {
        let mut out = self.clone();

        for (i, st) in out.constraints.iter_mut().enumerate() {
            let (bl, bu) = (qp.bl[i], qp.bu[i]);
            if bl == bu {
                // An equality row left `Inactive` is just as broken in the
                // other direction: the ratio test skips `bl == bu` rows, so it
                // could never enter the working set and the audit would have to
                // recover through elastic phase-1.
                *st = ConsStatus::Equality;
                continue;
            }
            *st = match *st {
                // Was an equality, is now a range. The previous status says the
                // row was tight, but not which side of the new range it should
                // sit on, so it carries no usable information.
                ConsStatus::Equality => ConsStatus::Inactive,
                ConsStatus::AtLower if bl <= NLP_LOWER_BOUND_INF => ConsStatus::Inactive,
                ConsStatus::AtUpper if bu >= NLP_UPPER_BOUND_INF => ConsStatus::Inactive,
                keep => keep,
            };
        }

        for (j, st) in out.bounds.iter_mut().enumerate() {
            let (xl, xu) = (qp.xl[j], qp.xu[j]);
            let l_finite = xl > NLP_LOWER_BOUND_INF;
            let u_finite = xu < NLP_UPPER_BOUND_INF;
            if l_finite && u_finite && (xl - xu).abs() <= opts.feas_tol {
                *st = BoundStatus::Fixed;
                continue;
            }
            *st = match *st {
                BoundStatus::Fixed => BoundStatus::Inactive,
                BoundStatus::AtLower if !l_finite => BoundStatus::Inactive,
                BoundStatus::AtUpper if !u_finite => BoundStatus::Inactive,
                keep => keep,
            };
        }

        out
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
