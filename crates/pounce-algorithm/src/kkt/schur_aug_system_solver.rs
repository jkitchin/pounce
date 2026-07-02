//! Block-triangular / Schur augmented-system solver (pounce#180 item 2, Phase 2).
//!
//! Wraps the standard [`StdAugSystemSolver`] — reusing its exact KKT assembly
//! and RHS packing — but routes the assembled system through a
//! [`FeralSchurSolver`] over a caller-supplied F/S partition (see
//! `crates/pounce-feral/src/schur.rs`). The partition is given as KKT-space
//! indices (`0..dim` in the `x, s, c, d` block order `StdAugSystemSolver`
//! assembles); the `S` block is Schur-complemented out and only the two
//! diagonal blocks are factorized, with inertia recovered a priori via
//! Sylvester's law.
//!
//! **Gate + fallback (first-class, per the scope doc).** The Schur path helps
//! only when `n_schur ≪ n_f`; and it is feral-specific. This wrapper falls back
//! to the plain `StdAugSystemSolver` — transparently, preserving every solve —
//! whenever: the Schur fraction exceeds `max_schur_frac`, the partition is
//! malformed for the current KKT dimension, or the Schur backend reports a
//! `FatalError`. A fallback is permanent for the rest of the solve (the KKT
//! pattern is fixed across IPM iterations, so re-deciding every iterate is
//! pointless).

use std::rc::Rc;

use crate::kkt::aug_system_solver::{AugSysCoeffs, AugSysRhs, AugSysSol, AugSystemSolver};
use crate::kkt::std_aug_system_solver::StdAugSystemSolver;
use pounce_common::diagnostics::DiagnosticsState;
use pounce_common::timing::TimingStatistics;
use pounce_common::types::{Index, Number};
use pounce_feral::{FeralConfig, FeralSchurSolver};
use pounce_linsol::{ESymSolverStatus, FactorPattern};

/// Default upper bound on `n_schur / dim`. Beyond this the dense `S`
/// (`O(n_s²)` store, `O(n_s³)` factor) makes the Schur path lose to a
/// monolithic factorization, so we fall back. Lenient by default — the caller
/// opts in by supplying the block, and knows its structure — but guards against
/// a pathological "Schur block is most of the matrix" request.
const DEFAULT_MAX_SCHUR_FRAC: f64 = 0.5;

pub struct SchurAugSystemSolver {
    /// KKT assembly + the fallback solver.
    inner: StdAugSystemSolver,
    schur: FeralSchurSolver,
    /// Caller-supplied Schur block, KKT-space indices.
    schur_indices: Vec<usize>,
    max_schur_frac: f64,

    /// `None` until the partition has been validated against a concrete KKT
    /// dimension; `Some(dim)` records what it was pinned for (re-decide only if
    /// the dimension changes, which it does not within one solve).
    decided_for_dim: Option<Index>,
    /// After [`Self::decided_for_dim`] is set: whether the Schur path is active.
    /// `false` means permanent fallback to `inner`.
    use_schur: bool,
    have_factor: bool,
    negevals: Index,
    last_status: ESymSolverStatus,
    timing: Option<Rc<TimingStatistics>>,
}

impl SchurAugSystemSolver {
    /// Wrap `inner` with a Schur backend over `schur_indices` (KKT-space).
    /// The Schur block's per-block feral solvers are configured from `cfg`
    /// (same knobs as the monolithic feral backend).
    pub fn new(inner: StdAugSystemSolver, schur_indices: Vec<usize>, cfg: FeralConfig) -> Self {
        Self {
            inner,
            schur: FeralSchurSolver::new(cfg),
            schur_indices,
            max_schur_frac: DEFAULT_MAX_SCHUR_FRAC,
            decided_for_dim: None,
            use_schur: false,
            have_factor: false,
            negevals: 0,
            last_status: ESymSolverStatus::Success,
            timing: None,
        }
    }

    /// Decide (once per KKT dimension) whether the Schur path is usable and, if
    /// so, pin its structure. `irn/jcn` are the assembled lower-triangle
    /// triplet from `inner`.
    fn decide(&mut self, dim: Index) {
        if self.decided_for_dim == Some(dim) {
            return;
        }
        self.decided_for_dim = Some(dim);
        self.use_schur = false;
        let n_s = self.schur_indices.len();
        let d = dim as usize;
        if n_s == 0 || n_s >= d {
            return;
        }
        if (n_s as f64) / (d as f64) > self.max_schur_frac {
            tracing::warn!(
                target: "pounce::kkt",
                n_schur = n_s, dim = d, max_frac = self.max_schur_frac,
                "Schur block too large relative to the KKT; using the standard solver"
            );
            return;
        }
        // Copy the triplet out of `inner` before touching `self.schur`
        // (disjoint fields, but the borrow checker only sees whole-`self`).
        let (irn, jcn) = {
            let (a, b, _v) = self.inner.assembled_triplet();
            (a.to_vec(), b.to_vec())
        };
        let st = self
            .schur
            .initialize_structure(dim, &irn, &jcn, &self.schur_indices);
        if st == ESymSolverStatus::Success {
            self.use_schur = true;
        } else {
            tracing::warn!(
                target: "pounce::kkt",
                "Schur partition rejected by the backend; using the standard solver"
            );
        }
    }

    /// Run the Schur factor + block backsolve for one RHS. Assumes `inner` has
    /// already assembled and `use_schur` is set. Returns the factor status;
    /// on `Success` the solution is written to `sol`.
    fn schur_solve_one(
        &mut self,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
        check_neg_evals: bool,
        num_neg_evals: Index,
    ) -> ESymSolverStatus {
        let dim = self.inner.assembled_dim() as usize;
        // Refill the Schur backend's values from the freshly assembled KKT.
        let vals = self.inner.assembled_triplet().2.to_vec();
        self.schur.values_array_mut().copy_from_slice(&vals);

        let status = {
            let _g = self
                .timing
                .as_deref()
                .map(|t| t.linear_system_factorization.guard());
            self.schur.factor(check_neg_evals, num_neg_evals)
        };
        self.last_status = status;
        match status {
            ESymSolverStatus::Success => {
                self.negevals = self.schur.number_of_neg_evals();
                let mut packed = vec![0.0; dim];
                self.inner.pack_rhs(rhs, &mut packed);
                let bstat = {
                    let _g = self
                        .timing
                        .as_deref()
                        .map(|t| t.linear_system_back_solve.guard());
                    self.schur.backsolve(1, &mut packed)
                };
                if bstat != ESymSolverStatus::Success {
                    self.have_factor = false;
                    self.last_status = bstat;
                    return bstat;
                }
                self.inner.unpack_sol(&packed, sol);
                self.have_factor = true;
                ESymSolverStatus::Success
            }
            ESymSolverStatus::WrongInertia => {
                // Both diagonal blocks factored, but the combined (Sylvester)
                // inertia is wrong: surface the count so the IPM's δ-perturbation
                // loop reacts, exactly as the monolithic path. The perturbation
                // reaches both blocks (δ_x/δ_s → A_FF, δ_c/δ_d → A_SS), so the
                // next re-factor can correct it.
                self.negevals = self.schur.number_of_neg_evals();
                self.have_factor = false;
                status
            }
            // Singular (a diagonal block itself is rank-deficient — the Schur
            // precondition is violated for this iterate) or FatalError/CallAgain:
            // return the sentinel and let `solve` fall back to the monolithic
            // solver, which regularizes the *full* system correctly. Bumping
            // δ_c (where `perturb_for_singular` routes a `Singular`) would not
            // fix a singular A_FF, so we do not surface `Singular` upward.
            other => {
                self.have_factor = false;
                other
            }
        }
    }
}

impl AugSystemSolver for SchurAugSystemSolver {
    fn provides_inertia(&self) -> bool {
        // Both the Schur (feral) path and the fallback backend report inertia.
        self.inner.provides_inertia()
    }

    fn number_of_neg_evals(&self) -> Index {
        if self.use_schur {
            self.negevals
        } else {
            self.inner.number_of_neg_evals()
        }
    }

    fn system_dim(&self) -> Index {
        self.inner.system_dim()
    }

    fn kkt_triplets(&self) -> Option<(Index, Vec<Index>, Vec<Index>, Vec<Number>)> {
        self.inner.kkt_triplets()
    }

    fn l_factor(&self, want_values: bool) -> Option<FactorPattern> {
        // The Schur path has no single monolithic L factor; only the fallback
        // (monolithic) path can expose one.
        if self.use_schur {
            None
        } else {
            self.inner.l_factor(want_values)
        }
    }

    fn increase_quality(&mut self) -> bool {
        self.have_factor = false;
        if self.use_schur {
            self.schur.increase_quality()
        } else {
            self.inner.increase_quality()
        }
    }

    fn last_solve_status(&self) -> ESymSolverStatus {
        if self.use_schur {
            self.last_status
        } else {
            self.inner.last_solve_status()
        }
    }

    fn set_timing_stats(&mut self, timing: Rc<TimingStatistics>) {
        self.timing = Some(Rc::clone(&timing));
        self.inner.set_timing_stats(timing);
    }

    fn set_diagnostics(&mut self, diag: Rc<DiagnosticsState>) {
        self.inner.set_diagnostics(diag);
    }

    fn solve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
        check_neg_evals: bool,
        num_neg_evals: Index,
    ) -> ESymSolverStatus {
        // Assemble the KKT once (reused by whichever path runs).
        let s = self.inner.assemble(coeffs);
        if s != ESymSolverStatus::Success {
            self.last_status = s;
            return s;
        }
        let dim = self.inner.assembled_dim();
        self.decide(dim);

        if self.use_schur {
            let st = self.schur_solve_one(rhs, sol, check_neg_evals, num_neg_evals);
            match st {
                ESymSolverStatus::Success | ESymSolverStatus::WrongInertia => return st,
                // Singular block / FatalError / CallAgain → permanent fallback.
                // Re-run this solve through the monolithic path so the IPM never
                // sees a spurious failure and gets correct full-system
                // regularization for the rest of the run.
                _ => {
                    tracing::warn!(
                        target: "pounce::kkt",
                        status = ?st,
                        "Schur backend could not factor this KKT; falling back to the standard solver"
                    );
                    self.use_schur = false;
                    return self
                        .inner
                        .solve(coeffs, rhs, sol, check_neg_evals, num_neg_evals);
                }
            }
        }
        self.inner
            .solve(coeffs, rhs, sol, check_neg_evals, num_neg_evals)
    }

    fn resolve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
    ) -> ESymSolverStatus {
        if self.use_schur {
            if self.have_factor {
                // Back-substitution only, against the cached Schur factor.
                let dim = self.inner.assembled_dim() as usize;
                let mut packed = vec![0.0; dim];
                self.inner.pack_rhs(rhs, &mut packed);
                let bstat = {
                    let _g = self
                        .timing
                        .as_deref()
                        .map(|t| t.linear_system_back_solve.guard());
                    self.schur.backsolve(1, &mut packed)
                };
                if bstat == ESymSolverStatus::Success {
                    self.inner.unpack_sol(&packed, sol);
                }
                return bstat;
            }
            // No cached factor — do a full solve.
            return self.solve(coeffs, rhs, sol, false, 0);
        }
        self.inner.resolve(coeffs, rhs, sol)
    }
}
