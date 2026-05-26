//! Linear-solver wrapper used by the QP solver.
//!
//! Owns a `Box<dyn SparseSymLinearSolverInterface>` and exposes a
//! single high-level entry point — `factorize_and_solve` — that
//! drives the backend through its four-step lifecycle (initialize
//! structure → fill values → factor → back-substitute) in one call.
//!
//! ## §4.7 iterative refinement
//!
//! Refinement is *implicit* in this layer: it runs inside the
//! backend's `multi_solve` whenever the backend's `refine` flag is
//! set, which is the default for both pounce-feral and pounce-hsl
//! (MA57). The qp-side code does not need to drive refinement
//! explicitly — every solve we issue is already refined to
//! near-machine-precision provided the factor is non-singular.
//! Reference: Wilkinson 1965 §3.5; Higham 2002 §12 — refinement
//! is the standard practice for sparse-direct symmetric saddle-
//! point solves and is what FERAL implements via
//! `solver.solve_refined`.
//!
//! ## §4.2 Schur-complement update path
//!
//! The Schur-complement update path will extend this module with
//! a cached-factor `resolve` entry point and a state machine
//! tracking which working-set increment is currently absorbed in
//! the Schur complement vs in the base factor. Phase 5a ships the
//! one-shot factor-and-solve at every iteration; Schur updates
//! are a follow-up performance commit (Phase 5a.1).

use crate::error::QpError;
use crate::kkt::KktTriplet;
use pounce_common::{Index, Number};
use pounce_linsol::status::ESymSolverStatus;
use pounce_linsol::{EMatrixFormat, SparseSymLinearSolverInterface};

/// A boxed linear-solver backend (FERAL by default; MA57 when the
/// caller wires the `ma57` feature in pounce-hsl).
///
/// Maintains the last (irn, jcn) so that [`Self::resolve`] can
/// back-substitute against the cached factor without supplying
/// the structure arrays again. This is the wrapper-layer
/// infrastructure on which the §4.2 Schur-complement update path
/// builds.
pub struct LinearSolver {
    backend: Box<dyn SparseSymLinearSolverInterface>,
    cached_irn: Option<Vec<Index>>,
    cached_jcn: Option<Vec<Index>>,
    cached_dim: usize,
    factored: bool,
}

impl LinearSolver {
    pub fn new(backend: Box<dyn SparseSymLinearSolverInterface>) -> Self {
        Self {
            backend,
            cached_irn: None,
            cached_jcn: None,
            cached_dim: 0,
            factored: false,
        }
    }

    /// Factor a fresh KKT and back-substitute against a single RHS.
    /// The solution overwrites `rhs` in place.
    ///
    /// `expected_neg_evals` is the expected count of negative
    /// eigenvalues. For the equality-only KKT `[H Aᵀ; A 0]` with
    /// full-rank `A` and reduced Hessian PD on `null(A)`, this is
    /// exactly `m` (Gould-Hribar-Nocedal 2001 §3.2). Passing
    /// `None` skips the inertia check.
    pub fn factorize_and_solve(
        &mut self,
        kkt: &KktTriplet,
        rhs: &mut [Number],
        expected_neg_evals: Option<i32>,
    ) -> Result<(), QpError> {
        let format = self.backend.matrix_format();
        if format != EMatrixFormat::TripletFormat {
            return Err(QpError::LinearSolverFailure(format!(
                "backend requires {format:?} but pounce-qp emits TripletFormat"
            )));
        }
        if rhs.len() != kkt.dim {
            return Err(QpError::DimensionMismatch(format!(
                "RHS length {} but KKT dim is {}",
                rhs.len(),
                kkt.dim
            )));
        }

        // A factor in progress invalidates the cache before any
        // backend operation, so a partial failure leaves us in a
        // consistent "no cache" state.
        self.factored = false;
        let dim = kkt.dim as Index;
        let nnz = kkt.irn.len() as Index;

        let st = self
            .backend
            .initialize_structure(dim, nnz, &kkt.irn, &kkt.jcn);
        if st != ESymSolverStatus::Success {
            return Err(QpError::LinearSolverFailure(format!(
                "initialize_structure → {st:?}"
            )));
        }

        // Fill values in the order the backend laid out internally
        // (which matches the (irn, jcn) order we just supplied).
        self.backend.values_array_mut().copy_from_slice(&kkt.vals);

        let (check, expected) = match expected_neg_evals {
            Some(e) => (true, e),
            None => (false, 0),
        };

        let st = self.backend.multi_solve(
            true, // new_matrix
            &kkt.irn, &kkt.jcn, 1, rhs, check, expected,
        );
        match st {
            ESymSolverStatus::Success => {
                self.cached_irn = Some(kkt.irn.clone());
                self.cached_jcn = Some(kkt.jcn.clone());
                self.cached_dim = kkt.dim;
                self.factored = true;
                Ok(())
            }
            ESymSolverStatus::Singular => Err(QpError::LinearSolverFailure(
                "KKT matrix is singular (LICQ violation or rank-deficient Jacobian)".into(),
            )),
            ESymSolverStatus::WrongInertia => Err(QpError::LinearSolverFailure(format!(
                "KKT inertia mismatch: expected {} negative eigenvalues, got {}",
                expected,
                self.backend.number_of_neg_evals()
            ))),
            ESymSolverStatus::CallAgain => Err(QpError::LinearSolverFailure(
                "backend requested re-call; not yet supported in pounce-qp".into(),
            )),
            ESymSolverStatus::FatalError => Err(QpError::LinearSolverFailure(
                "backend reported fatal error".into(),
            )),
        }
    }

    /// Back-substitute against the cached factor from the last
    /// successful [`Self::factorize_and_solve`]. Errors if no factor
    /// has succeeded since construction. The structure must not
    /// have changed since the last factor — the caller is
    /// responsible for invariance; this is the building block on
    /// which §4.2 Schur-complement updates layer.
    pub fn resolve(&mut self, rhs: &mut [Number]) -> Result<(), QpError> {
        if !self.factored {
            return Err(QpError::LinearSolverFailure(
                "resolve called before a successful factorize_and_solve".into(),
            ));
        }
        if rhs.len() != self.cached_dim {
            return Err(QpError::DimensionMismatch(format!(
                "resolve RHS length {} but cached factor has dim {}",
                rhs.len(),
                self.cached_dim
            )));
        }
        let irn = self.cached_irn.as_ref().expect("factored ⇒ cache present");
        let jcn = self.cached_jcn.as_ref().expect("factored ⇒ cache present");
        let st = self.backend.multi_solve(
            false, // new_matrix=false ⇒ reuse cached factor
            irn, jcn, 1, rhs, false, 0,
        );
        match st {
            ESymSolverStatus::Success => Ok(()),
            other => Err(QpError::LinearSolverFailure(format!(
                "resolve backend status: {other:?}"
            ))),
        }
    }

    /// Number of negative eigenvalues found in the most recent
    /// factorization. Useful for the §4.5 inertia-control path.
    /// Returns `None` if the backend does not provide inertia.
    pub fn number_of_neg_evals(&self) -> Option<i32> {
        if self.backend.provides_inertia() {
            Some(self.backend.number_of_neg_evals())
        } else {
            None
        }
    }

    /// Whether a cached factor is currently available.
    pub fn has_cached_factor(&self) -> bool {
        self.factored
    }
}
