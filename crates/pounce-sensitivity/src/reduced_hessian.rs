//! `ReducedHessianCalculator` — port of upstream
//! [`SensReducedHessianCalculator.{hpp,cpp}`](../../../ref/Ipopt/contrib/sIPOPT/src/SensReducedHessianCalculator.cpp).
//!
//! # What this computes
//!
//! Given a converged KKT factor `K` and a parameter-row selector `B`
//! that picks out the **free variables** (post active-set
//! elimination), the reduced Hessian is
//!
//! ```text
//! H_R = obj_scal · B · K⁻¹ · Bᵀ
//! ```
//!
//! per upstream's sign + obj-scaling convention at
//! [`SensReducedHessianCalculator.cpp:90-97`](../../../ref/Ipopt/contrib/sIPOPT/src/SensReducedHessianCalculator.cpp):
//! the raw Schur output is `S = -B K⁻¹ Bᵀ` (with the leading minus
//! from the augmented-system reduction), which is then multiplied
//! by `-obj_scal` to produce the unscaled reduced Hessian.
//!
//! In pounce we default `obj_scal = 1.0` (no NLP-side scaling
//! folded in here) so the operation reduces to `H_R = -S = B K⁻¹ Bᵀ`.
//! Phase D's `SensApplication` will plumb the real obj scaling once
//! the algorithm-side wiring lands.
//!
//! Reference: Pirnay, López-Negrete & Biegler 2012, §5
//! (reduced-Hessian use case), DOI:
//! [10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2).

use crate::p_calculator::PCalculator;
use crate::schur_data::SchurData;
use pounce_common::types::Number;

/// Compute the (column-major) reduced Hessian into the caller-supplied
/// buffer.
///
/// Mirrors `ReducedHessianCalculator::ComputeReducedHessian`
/// ([`SensReducedHessianCalculator.cpp:42-113`](../../../ref/Ipopt/contrib/sIPOPT/src/SensReducedHessianCalculator.cpp)).
///
/// # Arguments
///
/// - `pcalc`: a P-calculator that was built with `data_A = hess_data`
///   (the **same** SchurData on both sides — the reduced-Hessian
///   computation is the diagonal `B = A` case of `schur_matrix`).
///   Must have completed `compute_p` before this call; this function
///   calls `schur_matrix(hess_data, …)` which will run `compute_p`
///   lazily if needed.
/// - `hess_data`: the free-variable selector, in the IndexSchurData
///   format. Conceptually the same matrix as `pcalc.data_a()`.
/// - `obj_scal`: per-NLP objective scaling factor; pounce defaults to
///   `1.0` (no scaling). Mirrors upstream's `apply_obj_scaling(1.0)`
///   ([`SensReducedHessianCalculator.cpp:91`](../../../ref/Ipopt/contrib/sIPOPT/src/SensReducedHessianCalculator.cpp)).
/// - `out`: caller-allocated buffer of length `n_rows × n_rows` where
///   `n_rows = hess_data.nrows()`. Column-major. Overwritten on
///   success.
///
/// Returns `false` if the underlying `schur_matrix` call fails or
/// `out` is mis-sized.
///
/// # Note on scaling-induced warnings
///
/// Upstream prints a J_WARNING block when any of `x` / `c` / `d`
/// scaling is active because "a correct unscaled solution of the
/// reduced hessian cannot be guaranteed"
/// ([`SensReducedHessianCalculator.cpp:64-88`](../../../ref/Ipopt/contrib/sIPOPT/src/SensReducedHessianCalculator.cpp)).
/// Pounce's Phase-C surface takes `obj_scal` as an explicit argument
/// rather than reaching into an `NLP_scaling()` object; users
/// constructing a reduced Hessian outside a configured pounce IPM
/// own the responsibility to pass an `obj_scal` consistent with
/// whatever scaling their `K` factor encodes.
pub fn compute_reduced_hessian<P: PCalculator>(
    pcalc: &mut P,
    hess_data: &dyn SchurData,
    obj_scal: Number,
    out: &mut [Number],
) -> bool {
    let n = hess_data.nrows() as usize;
    if out.len() != n * n {
        return false;
    }
    // Step 1: build the Schur matrix S = -B K⁻¹ Bᵀ via the PCalculator.
    if !pcalc.schur_matrix(hess_data, out) {
        return false;
    }
    // Step 2: H_R = -obj_scal · S. Pounce default obj_scal = 1.0
    // yields H_R = B K⁻¹ Bᵀ.
    let factor = -obj_scal;
    for v in out.iter_mut() {
        *v *= factor;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backsolver::DenseLuBacksolver;
    use crate::p_calculator::IndexPCalculator;
    use crate::schur_data::IndexSchurData;

    /// 3×3 SPD `K = [[2,-1,0], [-1,2,-1], [0,-1,2]]`. Inverse is
    /// `K⁻¹ = 1/4 · [[3, 2, 1], [2, 4, 2], [1, 2, 3]]`.
    ///
    /// Pick `hess_data` selecting rows/cols {0, 2}. The 2×2
    /// submatrix of `K⁻¹` at indices {0, 2} is
    /// `[[3/4, 1/4], [1/4, 3/4]]`. With `obj_scal = 1.0`, that's
    /// the reduced Hessian.
    #[test]
    fn reduced_hessian_matches_kinv_submatrix() {
        #[rustfmt::skip]
        let k = vec![
             2.0, -1.0,  0.0,
            -1.0,  2.0, -1.0,
             0.0, -1.0,  2.0,
        ];
        let backsolver = DenseLuBacksolver::from_dense(3, &k).unwrap();
        let hess_data = IndexSchurData::from_parts(vec![0, 2], vec![1, 1]).unwrap();
        let pcalc_a = IndexSchurData::from_parts(vec![0, 2], vec![1, 1]).unwrap();
        let mut pcalc = IndexPCalculator::new(backsolver, pcalc_a);

        let mut hr = vec![0.0; 2 * 2];
        assert!(compute_reduced_hessian(&mut pcalc, &hess_data, 1.0, &mut hr));

        // Column-major: [j * n + i]
        // H_R[0,0] = 3/4
        // H_R[1,0] = 1/4
        // H_R[0,1] = 1/4
        // H_R[1,1] = 3/4
        assert!((hr[0] - 0.75).abs() < 1e-12, "H_R[0,0] = {}", hr[0]);
        assert!((hr[1] - 0.25).abs() < 1e-12, "H_R[1,0] = {}", hr[1]);
        assert!((hr[2] - 0.25).abs() < 1e-12, "H_R[0,1] = {}", hr[2]);
        assert!((hr[3] - 0.75).abs() < 1e-12, "H_R[1,1] = {}", hr[3]);
    }

    #[test]
    fn reduced_hessian_applies_obj_scal() {
        #[rustfmt::skip]
        let k = vec![
             2.0, -1.0,  0.0,
            -1.0,  2.0, -1.0,
             0.0, -1.0,  2.0,
        ];
        let backsolver = DenseLuBacksolver::from_dense(3, &k).unwrap();
        let hess_data = IndexSchurData::from_parts(vec![1], vec![1]).unwrap();
        let pcalc_a = IndexSchurData::from_parts(vec![1], vec![1]).unwrap();
        let mut pcalc = IndexPCalculator::new(backsolver, pcalc_a);

        // K⁻¹[1, 1] = 1, so H_R = obj_scal · 1.
        let mut hr = vec![0.0; 1];
        assert!(compute_reduced_hessian(&mut pcalc, &hess_data, 2.5, &mut hr));
        assert!((hr[0] - 2.5).abs() < 1e-12, "H_R = {}", hr[0]);
    }

    #[test]
    fn reduced_hessian_rejects_wrong_buffer_size() {
        let backsolver = DenseLuBacksolver::from_dense(2, &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let hd = IndexSchurData::from_parts(vec![0, 1], vec![1, 1]).unwrap();
        let pa = IndexSchurData::from_parts(vec![0, 1], vec![1, 1]).unwrap();
        let mut pc = IndexPCalculator::new(backsolver, pa);
        // hd is 2 rows so output should be 4 entries; pass 3.
        let mut wrong = vec![0.0; 3];
        assert!(!compute_reduced_hessian(&mut pc, &hd, 1.0, &mut wrong));
    }
}
