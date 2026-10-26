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
use pounce_common::timing::{LinearSystemPhase, time_linear_system};
use pounce_common::{Index, Number};
use pounce_linsol::status::ESymSolverStatus;
use pounce_linsol::{EMatrixFormat, SparseSymLinearSolverInterface};
use std::time::{Duration, Instant};

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
    /// Linear-independence rank tests run against this solver; see
    /// [`Self::rank_guard_calls`].
    rank_guard_calls: u32,
    /// Where this solver's time went; see [`Self::time_split`].
    times: LinsolTimes,
}

/// Cumulative wall time a [`LinearSolver`] spent in each kind of linear
/// algebra, and how many back-solves it ran (gh#1061).
///
/// A step QP's debug line reports its wall time; this splits it. A caller
/// takes [`LinearSolver::time_split`] before and after a solve and
/// [`Self::since`] the difference. What none of the buckets hold — assembling
/// the KKT, the ratio test, the drop test, the SMW correction's dense
/// arithmetic — is the solve's wall time minus their sum.
#[derive(Debug, Clone, Copy, Default)]
pub struct LinsolTimes {
    /// Symbolic analysis plus numeric factorization, including every
    /// inertia-control retry.
    pub factor: Duration,
    /// Number of factorizations, inertia-control retries included.
    pub n_factor: u64,
    /// Back-solves against a cached factor: the Schur path's per-iteration
    /// solve, its refinement passes, and the two per rank-2 update.
    pub resolve: Duration,
    /// Number of those back-solves.
    pub n_resolve: u64,
    /// The linear-independence guard, sparse probe or dense fallback.
    pub rank_test: Duration,
}

impl LinsolTimes {
    /// `self − earlier`, bucket by bucket.
    pub fn since(&self, earlier: &LinsolTimes) -> LinsolTimes {
        LinsolTimes {
            factor: self.factor.saturating_sub(earlier.factor),
            n_factor: self.n_factor.saturating_sub(earlier.n_factor),
            resolve: self.resolve.saturating_sub(earlier.resolve),
            n_resolve: self.n_resolve.saturating_sub(earlier.n_resolve),
            rank_test: self.rank_test.saturating_sub(earlier.rank_test),
        }
    }
}

impl LinearSolver {
    pub fn new(backend: Box<dyn SparseSymLinearSolverInterface>) -> Self {
        Self {
            backend,
            cached_irn: None,
            cached_jcn: None,
            cached_dim: 0,
            factored: false,
            rank_guard_calls: 0,
            times: LinsolTimes::default(),
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
        let started = Instant::now();
        let res = self.factorize_and_solve_inner(kkt, rhs, expected_neg_evals);
        self.times.factor += started.elapsed();
        self.times.n_factor += 1;
        res
    }

    fn factorize_and_solve_inner(
        &mut self,
        kkt: &KktTriplet,
        rhs: &mut [Number],
        expected_neg_evals: Option<i32>,
    ) -> Result<(), QpError> {
        let dim = kkt.dim as Index;
        let nnz = kkt.irn.len() as Index;

        // Charged to the convex path's timing sink when one is open (gh
        // #767); a no-op otherwise. Without this the active-set engine's
        // three `LinearSystem*` rows read 0.000s beside a `ConvexSolve` row
        // carrying the whole solve — the same "0% everywhere" report the
        // issue is about, one engine down.
        let st = time_linear_system(LinearSystemPhase::SymbolicFactorization, || {
            self.backend
                .initialize_structure(dim, nnz, &kkt.irn, &kkt.jcn)
        });
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

        let st = time_linear_system(LinearSystemPhase::Factorization, || {
            self.backend.multi_solve(
                true, // new_matrix
                &kkt.irn, &kkt.jcn, 1, rhs, check, expected,
            )
        });
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
        let started = Instant::now();
        let st = time_linear_system(LinearSystemPhase::BackSolve, || {
            self.backend.multi_solve(
                false, // new_matrix=false ⇒ reuse cached factor
                irn, jcn, 1, rhs, false, 0,
            )
        });
        self.times.resolve += started.elapsed();
        self.times.n_resolve += 1;
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

    /// How many rank tests the linear-independence guard has run against
    /// this solver since it was built, sparse or dense (or skipped as over
    /// budget). One guard pass runs a second test when its first answer
    /// dropped a permanent row (gh#974), so this can exceed the passes.
    /// Cumulative, never reset: a QP that spends its time pruning dependent
    /// rows is otherwise indistinguishable from one that spends it pivoting
    /// (gh#1061).
    pub fn rank_guard_calls(&self) -> u32 {
        self.rank_guard_calls
    }

    pub(crate) fn note_rank_guard_call(&mut self) {
        self.rank_guard_calls = self.rank_guard_calls.saturating_add(1);
    }

    /// Cumulative time spent factoring, back-solving and rank-testing since
    /// this solver was built; never reset (gh#1061). See [`LinsolTimes`].
    pub fn time_split(&self) -> LinsolTimes {
        self.times
    }

    pub(crate) fn note_rank_test_time(&mut self, d: Duration) {
        self.times.rank_test += d;
    }

    /// Whether [`Self::determine_dependent_rows`] is available. Always
    /// `true`: a backend without a rank-reveal of its own (MA57) borrows
    /// FERAL's sparse probe, so the QP's linear-independence guard never
    /// depends on `linear_solver` (gh#1061).
    pub fn provides_degeneracy_detection(&self) -> bool {
        true
    }

    /// Rank-reveal the dependent rows of the constraint Jacobian `J`
    /// (1-based triplet, `n_rows × n_cols`); fills `c_deps` with the
    /// 0-based indices of a maximal set of redundant rows. The backend
    /// factors a throwaway system internal to itself, leaving this
    /// wrapper's cached KKT factor untouched.
    ///
    /// A backend that does not rank-reveal (MA57 keeps the trait default)
    /// is answered by FERAL's sparse LU probe instead. Before gh#1061 the
    /// caller fell back to dense modified Gram-Schmidt for it — one length-`n`
    /// normal per active row, `O(k·n)` memory and `O(k²·n)` time — so picking
    /// MA57 for the interior point put that in every SQP QP: on an NLP with
    /// `n ≈ 15 000` and `≈ 10 800` equalities, gigabytes and `~10¹²` flops
    /// per pass. The probe factors a throwaway matrix, so which backend owns
    /// the KKT factor is irrelevant to it.
    pub fn determine_dependent_rows(
        &mut self,
        n_rows: Index,
        n_cols: Index,
        irn: &[Index],
        jcn: &[Index],
        vals: &[Number],
        c_deps: &mut Vec<Index>,
    ) -> ESymSolverStatus {
        if self.backend.provides_degeneracy_detection() {
            self.backend
                .determine_dependent_rows(n_rows, n_cols, irn, jcn, vals, c_deps)
        } else {
            pounce_feral::determine_dependent_rows(n_rows, n_cols, irn, jcn, vals, c_deps)
        }
    }
}
