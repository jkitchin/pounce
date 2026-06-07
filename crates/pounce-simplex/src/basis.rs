//! The basis engine for the revised simplex.
//!
//! The driver talks to the basis through a small trait, [`BasisEngine`]:
//! `ftran` / `btran` apply `B⁻¹` and `B⁻ᵀ`, `update` records a pivot, and
//! `refactor` rebuilds from the current basic columns. Two implementations
//! satisfy it:
//!
//! * [`FaerBasis`] — the production engine. It keeps a faer sparse LU of the
//!   *base* basis `B₀` (as of the last refactorization) and a product-form
//!   (eta) file of the pivots since. `B⁻¹ = E_t · … · E_1 · B₀⁻¹`. faer owns the
//!   numerically-delicate sparse LU with partial pivoting; the rank-1 update —
//!   which no general LU library provides — stays here.
//! * [`DenseBasis`] (test-only) — the explicit dense `B⁻¹` engine that was the
//!   Phase 6.1 baseline. It is retained under `cfg(test)` purely as the oracle
//!   the faer engine is validated against (see the lockstep test below).
//!
//! Every `REFACTOR_INTERVAL` pivots the driver calls `refactor` to shed the
//! accumulated eta chain, bounding round-off growth.

use faer::prelude::Solve;
use faer::sparse::linalg::solvers::Lu as SparseLu;
use faer::sparse::{SparseColMat, Triplet};
use faer::MatMut;

/// Rebuild the base factorization after this many product-form updates,
/// bounding error growth in the eta chain.
pub(crate) const REFACTOR_INTERVAL: usize = 50;

/// The basis-inverse operations the revised simplex needs. `ftran`/`btran` take
/// `&mut self` because the production engine reuses an internal work buffer and
/// applies in-place solves.
pub(crate) trait BasisEngine {
    /// The identity basis (`B = I`); the simplex starts here with the
    /// artificial variables basic.
    fn identity(m: usize) -> Self
    where
        Self: Sized;

    /// FTRAN: `out = B⁻¹ · col`, where `col` is a sparse original
    /// constraint-matrix column `(row, val)`. `out` has length `m`.
    fn ftran(&mut self, col: &[(usize, f64)], out: &mut [f64]);

    /// BTRAN: `out = rowᵀ · B⁻¹` (i.e. `out = B⁻ᵀ · row`). Used to form the
    /// simplex multipliers `y = c_Bᵀ B⁻¹`. `row` and `out` have length `m`.
    fn btran(&mut self, row: &[f64], out: &mut [f64]);

    /// Product-form update for a pivot at basis position `r`, where
    /// `alpha = B⁻¹ A_q` is the FTRAN'd entering column. The pivot element
    /// `alpha[r]` is guaranteed non-negligible by the ratio test.
    fn update(&mut self, r: usize, alpha: &[f64]);

    /// Rebuild the factorization from the sparse basic columns (`cols[r]` is the
    /// original column of the variable basic in row `r`). Returns `false` if the
    /// basis is singular. Resets the update counter on success.
    fn refactor(&mut self, cols: &[&[(usize, f64)]]) -> bool;

    /// Product-form updates applied since the last successful refactorization.
    fn updates_since_refactor(&self) -> usize;
}

/// One product-form (eta) factor `E`, equal to the identity except in column
/// `r`. `vals` is that column: `vals[r] = 1/αᵣ`, `vals[i] = −αᵢ/αᵣ` otherwise,
/// where `α` is the FTRAN'd entering column of the pivot.
struct Eta {
    r: usize,
    vals: Vec<f64>,
}

/// Production basis engine: faer sparse LU of the base basis plus a
/// product-form eta file (see the module docs).
pub(crate) struct FaerBasis {
    m: usize,
    /// LU of the base basis `B₀` as of the last refactor. `None` means the base
    /// is the identity — a fresh `identity()` or the `m == 0` degenerate case —
    /// so the base solve is a no-op and only the eta file applies.
    lu: Option<SparseLu<usize, f64>>,
    /// Pivots since the last refactor, oldest first.
    etas: Vec<Eta>,
    updates: usize,
    /// Dense `m`-length scratch reused by every `ftran`/`btran`.
    work: Vec<f64>,
}

impl BasisEngine for FaerBasis {
    fn identity(m: usize) -> Self {
        FaerBasis {
            m,
            lu: None,
            etas: Vec::new(),
            updates: 0,
            work: vec![0.0; m],
        }
    }

    fn ftran(&mut self, col: &[(usize, f64)], out: &mut [f64]) {
        let m = self.m;
        // Scatter the sparse column into the dense RHS, summing duplicate rows
        // exactly as the basis assembly does.
        self.work.iter_mut().for_each(|v| *v = 0.0);
        for &(i, v) in col {
            self.work[i] += v;
        }
        // Base solve `B₀ x = col` (a no-op when the base is the identity).
        if let Some(lu) = &self.lu {
            let rhs = MatMut::from_column_major_slice_mut(&mut self.work, m, 1);
            lu.solve_in_place(rhs);
        }
        // Apply the eta file forward: `x ← E_t · … · E_1 · x`.
        for eta in &self.etas {
            let vr = self.work[eta.r];
            if vr == 0.0 {
                continue;
            }
            self.work[eta.r] = 0.0;
            for i in 0..m {
                self.work[i] += vr * eta.vals[i];
            }
        }
        out.copy_from_slice(&self.work[..m]);
    }

    fn btran(&mut self, row: &[f64], out: &mut [f64]) {
        let m = self.m;
        self.work[..m].copy_from_slice(&row[..m]);
        // Apply the etas transposed, in reverse order: `w ← E_1ᵀ · … · E_tᵀ · w`.
        // `Eᵀ` changes only component `r`, setting it to `eta · w`.
        for eta in self.etas.iter().rev() {
            let mut s = 0.0;
            for i in 0..m {
                s += eta.vals[i] * self.work[i];
            }
            self.work[eta.r] = s;
        }
        // Base transpose solve `B₀ᵀ y = w`.
        if let Some(lu) = &self.lu {
            let rhs = MatMut::from_column_major_slice_mut(&mut self.work, m, 1);
            lu.solve_transpose_in_place(rhs);
        }
        out.copy_from_slice(&self.work[..m]);
    }

    fn update(&mut self, r: usize, alpha: &[f64]) {
        let m = self.m;
        let piv = alpha[r];
        let mut vals = vec![0.0; m];
        for (i, val) in vals.iter_mut().enumerate() {
            *val = -alpha[i] / piv;
        }
        vals[r] = 1.0 / piv;
        self.etas.push(Eta { r, vals });
        self.updates += 1;
    }

    fn refactor(&mut self, cols: &[&[(usize, f64)]]) -> bool {
        let m = self.m;
        debug_assert_eq!(cols.len(), m);
        if m == 0 {
            self.lu = None;
            self.etas.clear();
            self.updates = 0;
            return true;
        }
        // Column `r` of `B` is the basic column for row `r`; faer sums duplicate
        // `(row, col)` triplets, matching the dense assembly's `+=`.
        let mut trips: Vec<Triplet<usize, usize, f64>> = Vec::new();
        for (r, col) in cols.iter().enumerate() {
            for &(i, v) in col.iter() {
                trips.push(Triplet::new(i, r, v));
            }
        }
        let mat = match SparseColMat::<usize, f64>::try_new_from_triplets(m, m, &trips) {
            Ok(mat) => mat,
            Err(_) => return false,
        };
        match mat.as_ref().sp_lu() {
            Ok(lu) => {
                // faer flags only *structural* singularity (an empty basic
                // column); a structurally-full but numerically-singular basis
                // factors without error but leaves a zero pivot in `U`. Guard it
                // with a probe solve: that zero pivot makes the back-solve divide
                // by zero, so a non-finite result means the basis is unusable.
                for (i, w) in self.work.iter_mut().enumerate() {
                    *w = 1.0 + i as f64;
                }
                let rhs = MatMut::from_column_major_slice_mut(&mut self.work, m, 1);
                lu.solve_in_place(rhs);
                if self.work.iter().any(|v| !v.is_finite()) {
                    return false;
                }
                self.lu = Some(lu);
                self.etas.clear();
                self.updates = 0;
                true
            }
            // Structurally singular (e.g. an empty basic column).
            Err(_) => false,
        }
    }

    fn updates_since_refactor(&self) -> usize {
        self.updates
    }
}

/// Phase 6.1 dense basis engine — an explicit row-major `B⁻¹` with product-form
/// updates and a dense LU refactor (see [`crate::lu`]). Retained only as the
/// correctness oracle for [`FaerBasis`]; not built outside tests.
#[cfg(test)]
pub(crate) struct DenseBasis {
    m: usize,
    /// Explicit `B⁻¹`, row-major `m × m`.
    binv: Vec<f64>,
    updates: usize,
    scratch: Vec<f64>,
}

#[cfg(test)]
impl BasisEngine for DenseBasis {
    fn identity(m: usize) -> Self {
        let mut binv = vec![0.0; m * m];
        for i in 0..m {
            binv[i * m + i] = 1.0;
        }
        DenseBasis {
            m,
            binv,
            updates: 0,
            scratch: vec![0.0; m],
        }
    }

    #[allow(clippy::needless_range_loop)] // flat row-major `binv[i*m+j]` indexing
    fn ftran(&mut self, col: &[(usize, f64)], out: &mut [f64]) {
        let m = self.m;
        out.iter_mut().for_each(|v| *v = 0.0);
        for &(j, v) in col {
            if v == 0.0 {
                continue;
            }
            for i in 0..m {
                out[i] += self.binv[i * m + j] * v;
            }
        }
    }

    #[allow(clippy::needless_range_loop)] // flat row-major `binv[base+k]` indexing
    fn btran(&mut self, row: &[f64], out: &mut [f64]) {
        let m = self.m;
        out.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..m {
            let ri = row[i];
            if ri == 0.0 {
                continue;
            }
            let base = i * m;
            for k in 0..m {
                out[k] += ri * self.binv[base + k];
            }
        }
    }

    #[allow(clippy::needless_range_loop)] // flat row-major `binv` indexing
    fn update(&mut self, r: usize, alpha: &[f64]) {
        let m = self.m;
        let piv = alpha[r];
        let base_r = r * m;
        for k in 0..m {
            self.binv[base_r + k] /= piv;
        }
        for i in 0..m {
            if i == r {
                continue;
            }
            let ai = alpha[i];
            if ai == 0.0 {
                continue;
            }
            let base_i = i * m;
            for k in 0..m {
                self.binv[base_i + k] -= ai * self.binv[base_r + k];
            }
        }
        self.updates += 1;
    }

    #[allow(clippy::needless_range_loop)] // column-major `B X = I` back-solve
    fn refactor(&mut self, cols: &[&[(usize, f64)]]) -> bool {
        use crate::lu::Lu;
        let m = self.m;
        debug_assert_eq!(cols.len(), m);
        let mut b = vec![0.0; m * m];
        for (r, col) in cols.iter().enumerate() {
            for &(i, v) in col.iter() {
                b[i * m + r] += v;
            }
        }
        let lu = Lu::factor(b, m);
        if !lu.is_ok() {
            return false;
        }
        let mut e = vec![0.0; m];
        for col in 0..m {
            e.iter_mut().for_each(|v| *v = 0.0);
            e[col] = 1.0;
            lu.solve(&mut e, &mut self.scratch);
            for row in 0..m {
                self.binv[row * m + col] = e[row];
            }
        }
        self.updates = 0;
        true
    }

    fn updates_since_refactor(&self) -> usize {
        self.updates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(entries: &[(usize, f64)]) -> Vec<(usize, f64)> {
        entries.to_vec()
    }

    #[test]
    fn faer_identity_ftran_btran() {
        let mut b = FaerBasis::identity(3);
        let mut out = vec![0.0; 3];
        b.ftran(&col(&[(0, 2.0), (2, 5.0)]), &mut out);
        assert_eq!(out, vec![2.0, 0.0, 5.0]);
        b.btran(&[1.0, 2.0, 3.0], &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn faer_refactor_inverts_basis() {
        // B columns: diag(2,3,4) → B⁻¹ = diag(1/2,1/3,1/4).
        let mut b = FaerBasis::identity(3);
        let c0 = col(&[(0, 2.0)]);
        let c1 = col(&[(1, 3.0)]);
        let c2 = col(&[(2, 4.0)]);
        assert!(b.refactor(&[&c0, &c1, &c2]));
        let mut out = vec![0.0; 3];
        b.ftran(&col(&[(0, 2.0), (1, 3.0), (2, 4.0)]), &mut out);
        for v in &out {
            assert!((v - 1.0).abs() < 1e-12, "{out:?}");
        }
    }

    #[test]
    fn faer_product_form_update_matches_refactor() {
        // From identity, bring column (1,1)ᵀ into row 0.
        // New B = [[1,0],[1,1]]; B⁻¹ = [[1,0],[-1,1]].
        let mut b = FaerBasis::identity(2);
        let entering = col(&[(0, 1.0), (1, 1.0)]);
        let mut alpha = vec![0.0; 2];
        b.ftran(&entering, &mut alpha); // = (1,1) under identity
        b.update(0, &alpha);
        let mut out = vec![0.0; 2];
        b.ftran(&entering, &mut out); // B⁻¹ · (1,1) should be e0
        assert!(
            (out[0] - 1.0).abs() < 1e-12 && out[1].abs() < 1e-12,
            "{out:?}"
        );
        b.ftran(&col(&[(1, 1.0)]), &mut out); // B⁻¹ · e1 = (0,1)
        assert!(
            out[0].abs() < 1e-12 && (out[1] - 1.0).abs() < 1e-12,
            "{out:?}"
        );
    }

    #[test]
    fn faer_detects_singular_basis() {
        // Two identical columns → singular.
        let mut b = FaerBasis::identity(2);
        let c = col(&[(0, 1.0), (1, 1.0)]);
        assert!(!b.refactor(&[&c, &c]));
    }

    /// Lockstep oracle: drive `FaerBasis` and the dense `DenseBasis` through the
    /// *same* refactor → repeated (ftran, update) sequence and assert their
    /// FTRAN/BTRAN outputs agree. This is the literal realization of the design
    /// promise that the dense engine is the baseline faer is validated against.
    #[test]
    fn faer_matches_dense_oracle_under_pivots() {
        // A small, well-conditioned non-trivial basis and a sequence of pivots
        // that replace one basic column at a time.
        let m = 4;
        // Initial basic columns (dense-ish, diagonally dominant).
        let basis_cols: Vec<Vec<(usize, f64)>> = vec![
            col(&[(0, 4.0), (1, 1.0)]),
            col(&[(1, 5.0), (2, 1.0)]),
            col(&[(0, 1.0), (2, 6.0), (3, 1.0)]),
            col(&[(2, 1.0), (3, 7.0)]),
        ];
        let refs: Vec<&[(usize, f64)]> = basis_cols.iter().map(|c| c.as_slice()).collect();

        let mut faer = FaerBasis::identity(m);
        let mut dense = DenseBasis::identity(m);
        assert!(faer.refactor(&refs));
        assert!(dense.refactor(&refs));

        // A set of entering columns to FTRAN, and the rows they pivot into.
        let entering: Vec<Vec<(usize, f64)>> = vec![
            col(&[(0, 2.0), (3, 1.0)]),
            col(&[(1, 1.0), (2, 3.0)]),
            col(&[(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)]),
        ];
        let pivot_rows = [1usize, 2, 0];

        let assert_close = |a: &[f64], b: &[f64], ctx: &str| {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-9, "{ctx}: {a:?} vs {b:?}");
            }
        };

        let mut fa = vec![0.0; m];
        let mut da = vec![0.0; m];
        for (k, ent) in entering.iter().enumerate() {
            // FTRAN agreement.
            faer.ftran(ent, &mut fa);
            dense.ftran(ent, &mut da);
            assert_close(&fa, &da, &format!("ftran {k}"));

            // BTRAN agreement on a probe row.
            let probe: Vec<f64> = (0..m).map(|i| 1.0 + i as f64).collect();
            let mut fb = vec![0.0; m];
            let mut db = vec![0.0; m];
            faer.btran(&probe, &mut fb);
            dense.btran(&probe, &mut db);
            assert_close(&fb, &db, &format!("btran {k}"));

            // Apply the same pivot to both engines.
            let r = pivot_rows[k];
            faer.update(r, &fa);
            dense.update(r, &da);
        }

        // One more FTRAN after the eta chain to confirm post-update agreement.
        let tail = col(&[(0, 1.0), (2, 2.0)]);
        faer.ftran(&tail, &mut fa);
        dense.ftran(&tail, &mut da);
        assert_close(&fa, &da, "ftran tail");
        assert_eq!(
            faer.updates_since_refactor(),
            dense.updates_since_refactor()
        );
    }
}
