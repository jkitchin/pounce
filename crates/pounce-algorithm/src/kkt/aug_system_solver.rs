//! Augmented-system solver trait — port of `IpAugSystemSolver.hpp`.
//!
//! Solves the symmetric saddle-point system
//!
//! ```text
//!   [ W·factor + Σ_x + δ_x I       0          J_c^T   J_d^T ] [ dx ]   [ rx ]
//!   [          0           Σ_s + δ_s I        0       -I    ] [ ds ] = [ rs ]
//!   [         J_c                  0       -Σ_c-δ_c    0    ] [ dyc]   [ rc ]
//!   [         J_d                 -I            0   -Σ_d-δ_d] [ dyd]   [ rd ]
//! ```
//!
//! See `KKT_SYSTEM.md` §3 for the sign convention. `Σ_x = D_x`, `Σ_s
//! = D_s`, `Σ_c = D_c`, `Σ_d = D_d` are the diagonal weights pulled
//! from `IpoptCalculatedQuantities`. Any of the `D_*` may be `None`,
//! interpreted as zero. `delta_*` are the perturbations driven by the
//! `PerturbationHandler`.

use pounce_common::timing::TimingStatistics;
use pounce_common::types::{Index, Number};
use pounce_linalg::{Matrix, SymMatrix, Vector};
use pounce_linsol::ESymSolverStatus;
use std::rc::Rc;

/// Bundle of the matrices/vectors that define one augmented-system
/// instance. Lives only for the duration of the call. Mirrors the
/// long argument list of upstream `AugSystemSolver::Solve`.
pub struct AugSysCoeffs<'a> {
    /// Hessian-of-Lagrangian block. `None` means W = 0 (used by
    /// `LeastSquareMults` and the resto-NLP equality multiplier
    /// estimate).
    pub w: Option<&'a dyn SymMatrix>,
    /// Multiplier on `W` (typically 1.0; restoration uses ζ).
    pub w_factor: Number,
    /// `D_x`, the (1,1) primal weight diagonal. `None` means zero.
    pub d_x: Option<&'a dyn Vector>,
    pub delta_x: Number,
    /// `D_s`, the (2,2) slack weight diagonal. `None` means zero.
    pub d_s: Option<&'a dyn Vector>,
    pub delta_s: Number,
    /// Equality-constraint Jacobian, `m_c × n_x`.
    pub j_c: &'a dyn Matrix,
    /// `D_c`, the (3,3) diagonal weight. `None` means zero. Goes in
    /// with a *negative* sign, matching upstream.
    pub d_c: Option<&'a dyn Vector>,
    pub delta_c: Number,
    /// Inequality-constraint Jacobian, `m_d × n_x`.
    pub j_d: &'a dyn Matrix,
    /// `D_d`, the (4,4) diagonal weight. `None` means zero. Goes in
    /// with a *negative* sign, matching upstream.
    pub d_d: Option<&'a dyn Vector>,
    pub delta_d: Number,
}

/// Right-hand sides for one solve. All four slices are required;
/// upstream always provides all four (even if some are zero).
pub struct AugSysRhs<'a> {
    pub rhs_x: &'a dyn Vector,
    pub rhs_s: &'a dyn Vector,
    pub rhs_c: &'a dyn Vector,
    pub rhs_d: &'a dyn Vector,
}

/// Solution slots, written in place. Must already be sized to match
/// the corresponding RHS dim.
pub struct AugSysSol<'a> {
    pub sol_x: &'a mut dyn Vector,
    pub sol_s: &'a mut dyn Vector,
    pub sol_c: &'a mut dyn Vector,
    pub sol_d: &'a mut dyn Vector,
}

/// Trait surface mirroring `Ipopt::AugSystemSolver`.
pub trait AugSystemSolver {
    /// Whether the underlying linear solver reports inertia.
    fn provides_inertia(&self) -> bool;

    /// Number of negative eigenvalues observed in the most recent
    /// factorization. Caller checks `provides_inertia()` first.
    fn number_of_neg_evals(&self) -> Index;

    /// Ask the underlying solver for higher-quality pivoting.
    fn increase_quality(&mut self) -> bool;

    /// Status of the most recent `solve` call.
    fn last_solve_status(&self) -> ESymSolverStatus;

    /// Install the shared per-solve `TimingStatistics` so the
    /// linear-system factor/back-solve calls are attributed to
    /// `linear_system_factorization` / `linear_system_back_solve`.
    /// Default impl is a no-op (timing disabled); the standard
    /// solver overrides to record both fields, and composite solvers
    /// (LowRank) forward to their inner solver.
    fn set_timing_stats(&mut self, _timing: Rc<TimingStatistics>) {}

    /// Install the shared per-solve diagnostics state so KKT-dump
    /// sites can consult per-iter gating. Default impl is a no-op
    /// (diagnostics disabled); the standard solver overrides to wire
    /// in the dump path.
    fn set_diagnostics(&mut self, _diag: Rc<pounce_common::diagnostics::DiagnosticsState>) {}

    /// One factor + back-substitution for the full 4×4 block system.
    /// `check_neg_evals=true` asks the linsol to verify that the
    /// observed inertia equals `num_neg_evals`; on mismatch the
    /// status is `WrongInertia` and the solution is left untouched.
    fn solve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
        check_neg_evals: bool,
        num_neg_evals: Index,
    ) -> ESymSolverStatus;

    /// Back-substitution only, reusing the factorization from the most
    /// recent successful `solve`. Caller must guarantee the augmented
    /// matrix is byte-identical to that solve (same W, J_c, J_d, all
    /// diagonals, all perturbations, same pivot tolerance). Used by
    /// `PdFullSpaceSolver`'s iterative-refinement loop and same-matrix
    /// fast path to avoid the per-iter MA57BD refactor that dominates
    /// pounce-ma57 wall time on long-iter problems (e.g. cont5_2_4_l
    /// drops from 97s → ~30s once refactor-per-refinement is gone).
    ///
    /// Default impl falls through to `solve` (correct but slow);
    /// `StdAugSystemSolver` overrides to skip `refill_values` and pass
    /// `new_matrix=false` to the linear solver.
    fn resolve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
    ) -> ESymSolverStatus {
        self.solve(coeffs, rhs, sol, false, 0)
    }

    /// Solve the same KKT system for `nrhs` right-hand sides. Default
    /// impl loops [`solve`]; concrete backends override only when they
    /// can amortize factorization across calls. Mirrors upstream's
    /// `AugSystemSolver::MultiSolve` (`IpAugSystemSolver.hpp:113-150`).
    ///
    /// `rhs_list` and `sol_list` must have the same length; each pair
    /// describes one independent solve. The same `coeffs` are used for
    /// every column.
    fn multi_solve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs_list: &[&AugSysRhs<'_>],
        sol_list: &mut [&mut AugSysSol<'_>],
        check_neg_evals: bool,
        num_neg_evals: Index,
    ) -> ESymSolverStatus {
        debug_assert_eq!(rhs_list.len(), sol_list.len());
        for (rhs, sol) in rhs_list.iter().zip(sol_list.iter_mut()) {
            let status = self.solve(coeffs, rhs, *sol, check_neg_evals, num_neg_evals);
            if status != ESymSolverStatus::Success {
                return status;
            }
        }
        ESymSolverStatus::Success
    }
}
