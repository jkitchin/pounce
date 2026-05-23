//! QP problem definition and the solution / warm-start types it
//! pairs with. Storage borrows from `pounce-linalg` types directly;
//! the QP solver never copies the Hessian or Jacobian.

use crate::error::{QpError, QpStatus};
use crate::working_set::WorkingSet;
use pounce_common::Number;
use pounce_linalg::triplet::{GenTMatrix, SymTMatrix};
use std::time::Duration;

/// Caller-supplied hint about the inertia of `H`. Lets a strictly-
/// convex problem skip the inertia-correction probe of §4.5.
/// `Unknown` is always safe (the solver detects indefiniteness from
/// the LDLᵀ factor of the KKT block) and is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HessianInertia {
    /// `H` is symmetric positive semi-definite.
    Psd,
    /// `H` has (potentially) negative eigenvalues; the inertia-
    /// control path is required.
    Indefinite,
    /// Caller offers no claim; solver probes via factor inertia.
    #[default]
    Unknown,
}

/// A convex-or-nonconvex sparse QP:
/// ```text
///     min   ½ xᵀ H x + gᵀ x
///     s.t.  bl ≤ A x ≤ bu
///           xl ≤   x ≤ xu
/// ```
/// Two-sided general bounds (equality is `bl = bu`); two-sided
/// variable bounds (fixed is `xl = xu`; free is `±NLP_*_BOUND_INF`).
/// `H` is symmetric — only the upper triangle is stored.
///
/// The lifetime parameter ties the borrowed problem data to the
/// caller's storage; the solver never copies any of these fields.
pub struct QpProblem<'a> {
    pub n: usize,
    pub m: usize,
    pub h: &'a SymTMatrix,
    pub g: &'a [Number],
    pub a: &'a GenTMatrix,
    pub bl: &'a [Number],
    pub bu: &'a [Number],
    pub xl: &'a [Number],
    pub xu: &'a [Number],
    pub hessian_inertia: HessianInertia,
}

impl<'a> QpProblem<'a> {
    /// Validate every dimension and bound-ordering invariant the
    /// solver relies on. Called once at the top of `solve` before
    /// any work happens.
    pub fn validate(&self) -> Result<(), QpError> {
        if self.h.space().dim() as usize != self.n {
            return Err(QpError::DimensionMismatch(format!(
                "H is {}×{} but n = {}",
                self.h.space().dim(),
                self.h.space().dim(),
                self.n
            )));
        }
        if self.g.len() != self.n {
            return Err(QpError::DimensionMismatch(format!(
                "g.len() = {} but n = {}",
                self.g.len(),
                self.n
            )));
        }
        if self.a.space().n_rows() as usize != self.m
            || self.a.space().n_cols() as usize != self.n
        {
            return Err(QpError::DimensionMismatch(format!(
                "A is {}×{} but expected {}×{}",
                self.a.space().n_rows(),
                self.a.space().n_cols(),
                self.m,
                self.n
            )));
        }
        if self.bl.len() != self.m || self.bu.len() != self.m {
            return Err(QpError::DimensionMismatch(format!(
                "bl.len() = {}, bu.len() = {}, but m = {}",
                self.bl.len(),
                self.bu.len(),
                self.m
            )));
        }
        if self.xl.len() != self.n || self.xu.len() != self.n {
            return Err(QpError::DimensionMismatch(format!(
                "xl.len() = {}, xu.len() = {}, but n = {}",
                self.xl.len(),
                self.xu.len(),
                self.n
            )));
        }
        for (i, (&l, &u)) in self.bl.iter().zip(self.bu.iter()).enumerate() {
            if l > u {
                return Err(QpError::InvertedBounds(format!(
                    "constraint row {i}: bl = {l} > bu = {u}"
                )));
            }
        }
        for (i, (&l, &u)) in self.xl.iter().zip(self.xu.iter()).enumerate() {
            if l > u {
                return Err(QpError::InvertedBounds(format!(
                    "variable {i}: xl = {l} > xu = {u}"
                )));
            }
        }
        Ok(())
    }
}

/// Warm-start seed: previous primal-dual iterate plus working set.
/// Passed to [`crate::QpSolver::solve`] as `Some(ws)`; `None` is the
/// cold-start path through phase-1 elastic mode (§4.3).
#[derive(Debug, Clone)]
pub struct QpWarmStart {
    pub x: Vec<Number>,
    /// Lagrange multipliers for the general constraints, length `m`.
    pub lambda_g: Vec<Number>,
    /// Bound multipliers, length `n`, packed signed
    /// (`z_l − z_u`). Positive ⇒ lower-bound active, negative ⇒
    /// upper-bound active.
    pub lambda_x: Vec<Number>,
    pub working: WorkingSet,
}

/// Solver output. `working` is the new working set, suitable for
/// passing as the next solve's warm start.
#[derive(Debug, Clone)]
pub struct QpSolution {
    pub x: Vec<Number>,
    pub lambda_g: Vec<Number>,
    pub lambda_x: Vec<Number>,
    pub working: WorkingSet,
    pub obj: Number,
    pub status: QpStatus,
    pub stats: QpStats,
}

/// Per-solve counters and timers reported alongside the solution.
/// Phase 5a uses these for the §8.2 scaling-sweep plots and the
/// §8.5 warm-start sweep.
#[derive(Debug, Clone, Default)]
pub struct QpStats {
    /// Total active-set changes (adds + drops) across the solve.
    pub n_working_set_changes: u32,
    /// Number of times the cached factorization was discarded and
    /// the base KKT was refactored (the §4.2 reset cycle).
    pub n_refactor: u32,
    /// Number of Schur-complement rank-1 updates applied (the
    /// bounded-cost path between refactors).
    pub n_schur_updates: u32,
    /// Whether the solve passed through phase-1 elastic mode.
    pub used_phase1: bool,
    /// Wall-clock time spent inside `solve`.
    pub time: Duration,
}
