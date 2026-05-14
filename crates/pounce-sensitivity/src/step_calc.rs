//! `SensStepCalc` trait — orchestrates the sensitivity step computation.
//!
//! Given a converged KKT iterate and a parameter perturbation
//! `Δp`, sIPOPT's step calc produces the first-order primal/dual
//! sensitivity step `(Δx, Δλ, Δz)`. The math factors through two
//! linear systems:
//!
//! 1. **Schur solve**: `S · Δu = −Bᵀ Δp`, with `S` factored by
//!    [`crate::schur_driver::SchurDriver`].
//! 2. **Augmented backsolve**: `K · Δx = −A · Δu`, via the
//!    [`crate::SensBacksolver`].
//!
//! Reference: Pirnay, López-Negrete & Biegler 2012, §3 (DOI:
//! [10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2)).
//! Upstream impl:
//! [`SensStdStepCalc.{hpp,cpp}`](../../../ref/Ipopt/contrib/sIPOPT/src/SensStdStepCalc.cpp).
//!
//! # Phase B.1 scope
//!
//! This file ships the trait + a minimal `StdStepCalc` that exercises
//! the two-step pipeline using the same `SensBacksolver` instance for
//! both the Schur build and the inner augmented backsolve. The
//! algorithm-side wiring that produces the actual `Δp` source vector
//! from a TNLP's parameter-perturbation slot lands in Phase B.2.

use crate::backsolver::SensBacksolver;
use crate::p_calculator::PCalculator;
use crate::schur_driver::SchurDriver;
use pounce_common::types::Number;

/// Compute a sensitivity step `Δu = S⁻¹ · rhs_u` (Schur-space) and
/// `Δx_full = K⁻¹ · A · Δu` (backsolved KKT-space), where the
/// `rhs_u` vector encodes the parameter perturbation.
///
/// Mirrors upstream `SensStepCalc` (abstract,
/// [`SensStepCalc.hpp`](../../../ref/Ipopt/contrib/sIPOPT/src/SensStepCalc.hpp))
/// + `SensStdStepCalc` (concrete, the only implementation upstream
/// ships).
pub trait SensStepCalc {
    /// Run the two-step sensitivity computation. Outputs:
    /// - `du`: length `n_b`, the Schur-space step.
    /// - `dx_full`: length `n_full` (the backsolver's dimension),
    ///   the full primal/dual step `K⁻¹ A · du`. Implementations
    ///   may apply the upstream sign convention internally (the
    ///   exact sign depends on which side of the augmented system
    ///   the perturbation enters; see the upstream reference for
    ///   the parametric-flavor convention).
    ///
    /// Returns `false` if the Schur driver hasn't been built /
    /// factored, the backsolver fails, or the buffers are
    /// mis-sized.
    fn compute_step(
        &self,
        rhs_u: &[Number],
        du: &mut [Number],
        dx_full: &mut [Number],
    ) -> bool;
}

/// Reference implementation that strings together
/// [`SchurDriver::schur_solve`] and a final
/// [`SensBacksolver::solve`] using the `A` data the driver was
/// built with.
///
/// Mirrors upstream
/// [`SensStdStepCalc.cpp:23-282`](../../../ref/Ipopt/contrib/sIPOPT/src/SensStdStepCalc.cpp).
///
/// The Schur-driver type is borrowed for the lifetime of the
/// `StdStepCalc`; this matches upstream's "build once, step many"
/// pattern.
pub struct StdStepCalc<'d, D: SchurDriver + WithBacksolver, P: PCalculator> {
    driver: &'d D,
    /// Borrow of the same `PCalculator` that built the driver, used
    /// to expose the `A` data for the final `A · du` scatter without
    /// requiring the caller to thread it through separately.
    pcalc: &'d P,
}

impl<'d, D: SchurDriver + WithBacksolver, P: PCalculator> StdStepCalc<'d, D, P> {
    /// Construct from references to the driver and the matching
    /// pcalc. Both must already be in the post-factor state.
    pub fn new(driver: &'d D, pcalc: &'d P) -> Self {
        Self { driver, pcalc }
    }
}

/// Bridge trait — exposes a `SensBacksolver`-shaped solve through
/// whatever the driver wraps. Implementations of `SchurDriver` that
/// want to be consumed by `StdStepCalc` opt in by also implementing
/// `WithBacksolver`. This keeps `SchurDriver`'s own surface
/// minimal — most drivers don't need to expose the inner backsolver.
pub trait WithBacksolver {
    /// Apply `K⁻¹ · rhs` and write the result into `out`. Returns
    /// `false` if the inner backsolver fails or buffers don't match.
    fn k_solve(&self, rhs: &[Number], out: &mut [Number]) -> bool;
}

impl<'d, D, P> SensStepCalc for StdStepCalc<'d, D, P>
where
    D: SchurDriver + WithBacksolver,
    P: PCalculator,
{
    fn compute_step(
        &self,
        rhs_u: &[Number],
        du: &mut [Number],
        dx_full: &mut [Number],
    ) -> bool {
        // 1. Schur step: solve S · du = rhs_u.
        if !self.driver.schur_solve(rhs_u, du) {
            return false;
        }
        // 2. Construct the KKT-side rhs: A · du. The trans_multiply
        //    method on a SchurData computes Bᵀ u, so calling
        //    A.trans_multiply(du, scratch) gives Aᵀ Bᵀ = … wait, A
        //    is the row-space matrix, so multiply by du to get the
        //    full-state rhs. The pounce convention: each row of A
        //    selects a single full-state component, so
        //    `A.trans_multiply(du, rhs)` scatters `du[i]` into
        //    `rhs[A_idx[i]]`.
        let a = self.pcalc.data_a();
        let n_full = dx_full.len();
        let mut rhs_full = vec![0.0; n_full];
        if let Err(_) = a.trans_multiply(du, &mut rhs_full) {
            return false;
        }
        // 3. Backsolve K · dx_full = rhs_full.
        self.driver.k_solve(&rhs_full, dx_full)
    }
}

/// Convenience impl: a `DenseGenSchurDriver` parametrized over an
/// `IndexPCalculator<B>` can hand off its inner backsolver via
/// this bridge.
impl<B> WithBacksolver
    for crate::schur_driver::DenseGenSchurDriver<crate::p_calculator::IndexPCalculator<B>, B>
where
    B: SensBacksolver,
{
    fn k_solve(&self, rhs: &[Number], out: &mut [Number]) -> bool {
        // The IndexPCalculator owns the backsolver; reach through.
        self.pcalc().backsolver().solve(rhs, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backsolver::DenseLuBacksolver;
    use crate::p_calculator::IndexPCalculator;
    use crate::schur_data::IndexSchurData;
    use crate::schur_driver::DenseGenSchurDriver;

    /// End-to-end sensitivity step on the 3×3 SPD K + 2-row B = A
    /// setup that other tests reuse.
    ///
    /// S = -K⁻¹ restricted to rows/cols {0, 2}
    ///   = -[[3/4, 1/4], [1/4, 3/4]]
    ///
    /// With rhs_u = (1, 0):
    ///   S · du = (1, 0) ⇒ du = (-3/2, 1/2)  (verified in the
    ///   schur_driver test).
    ///
    /// `A · du` = `e_0 · du[0] + e_2 · du[1]` lifted to full-x:
    ///   rhs_full = (-3/2, 0, 1/2).
    /// K⁻¹ · rhs_full = K⁻¹ · (-3/2, 0, 1/2)
    ///   K⁻¹ = 1/4 · [[3, 2, 1], [2, 4, 2], [1, 2, 3]]
    ///   K⁻¹ · (-3/2, 0, 1/2) = 1/4 · (3·(-3/2)+1·(1/2),
    ///                                 2·(-3/2)+2·(1/2),
    ///                                 1·(-3/2)+3·(1/2))
    ///                        = 1/4 · (-4, -2, 0)
    ///                        = (-1, -1/2, 0).
    #[test]
    fn std_step_calc_runs_two_step_pipeline() {
        #[rustfmt::skip]
        let k = vec![
             2.0, -1.0,  0.0,
            -1.0,  2.0, -1.0,
             0.0, -1.0,  2.0,
        ];
        let backsolver = DenseLuBacksolver::from_dense(3, &k).unwrap();
        let a = IndexSchurData::from_parts(vec![0, 2], vec![1, 1]).unwrap();
        let pc = IndexPCalculator::new(backsolver, a);
        let mut driver = DenseGenSchurDriver::<_, DenseLuBacksolver>::new(pc);
        let b = IndexSchurData::from_parts(vec![0, 2], vec![1, 1]).unwrap();
        assert!(driver.schur_build_and_factor(&b));

        let step = StdStepCalc::new(&driver, driver.pcalc());
        let rhs_u = [1.0, 0.0];
        let mut du = [0.0; 2];
        let mut dx = [0.0; 3];
        assert!(step.compute_step(&rhs_u, &mut du, &mut dx));

        // du = (-3/2, 1/2)
        assert!((du[0] - (-1.5)).abs() < 1e-10, "du[0] = {}", du[0]);
        assert!((du[1] - 0.5).abs() < 1e-10, "du[1] = {}", du[1]);

        // dx = (-1, -1/2, 0)
        assert!((dx[0] - (-1.0)).abs() < 1e-10, "dx[0] = {}", dx[0]);
        assert!((dx[1] - (-0.5)).abs() < 1e-10, "dx[1] = {}", dx[1]);
        assert!((dx[2] - 0.0).abs() < 1e-10, "dx[2] = {}", dx[2]);
    }
}
