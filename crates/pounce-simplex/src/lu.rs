//! Dense partial-pivot LU for the basis matrix.
//!
//! The revised simplex maintains an explicit dense basis inverse and updates it
//! by product-form (eta) rank-1 steps each pivot (see [`crate::basis`]). Those
//! updates accumulate round-off, so every so often the inverse is rebuilt from
//! scratch by factoring the current basis `B` and solving `B X = I`. This module
//! is that rebuild: a straightforward right-looking LU with partial pivoting.
//!
//! It is deliberately dense and self-contained. Phase 6.2 replaces the whole
//! basis engine with a sparse LU + factorization updates; this is the
//! correctness baseline it will be validated against.

/// An LU factorization `P A = L U` of a square `n × n` matrix, stored
/// row-major. `lu` holds `L` (unit lower, implicit diagonal) and `U` (upper)
/// packed together; `piv` is the row permutation (partial pivoting).
pub(crate) struct Lu {
    n: usize,
    /// Packed `L\U`, row-major `n × n`.
    lu: Vec<f64>,
    /// Row permutation: `piv[i]` is the original row now in position `i`.
    piv: Vec<usize>,
    /// `false` if a zero pivot was hit (singular to working precision).
    ok: bool,
}

impl Lu {
    /// Factor `a` (row-major `n × n`, consumed) with partial pivoting.
    pub(crate) fn factor(mut a: Vec<f64>, n: usize) -> Lu {
        debug_assert_eq!(a.len(), n * n);
        let mut piv: Vec<usize> = (0..n).collect();
        let mut ok = true;
        for k in 0..n {
            // Pivot: largest-magnitude entry in column k at or below the
            // diagonal.
            let mut p = k;
            let mut best = a[k * n + k].abs();
            for i in (k + 1)..n {
                let v = a[i * n + k].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            if best <= 1e-12 {
                ok = false;
                continue;
            }
            if p != k {
                for j in 0..n {
                    a.swap(k * n + j, p * n + j);
                }
                piv.swap(k, p);
            }
            let akk = a[k * n + k];
            for i in (k + 1)..n {
                let f = a[i * n + k] / akk;
                a[i * n + k] = f;
                for j in (k + 1)..n {
                    a[i * n + j] -= f * a[k * n + j];
                }
            }
        }
        Lu { n, lu: a, piv, ok }
    }

    /// `true` if the factorization is non-singular to working precision.
    pub(crate) fn is_ok(&self) -> bool {
        self.ok
    }

    /// Solve `A x = rhs` in place (`rhs` becomes `x`). Applies the row
    /// permutation, then a forward and back substitution.
    #[allow(clippy::needless_range_loop)] // triangular solves index `lu[i*n+j]`
    pub(crate) fn solve(&self, rhs: &mut [f64], scratch: &mut [f64]) {
        let n = self.n;
        // Apply permutation: scratch = P rhs.
        for i in 0..n {
            scratch[i] = rhs[self.piv[i]];
        }
        // Forward solve L y = P rhs (unit lower).
        for i in 0..n {
            let mut s = scratch[i];
            for j in 0..i {
                s -= self.lu[i * n + j] * scratch[j];
            }
            scratch[i] = s;
        }
        // Back solve U x = y.
        for i in (0..n).rev() {
            let mut s = scratch[i];
            for j in (i + 1)..n {
                s -= self.lu[i * n + j] * scratch[j];
            }
            scratch[i] = s / self.lu[i * n + i];
        }
        rhs.copy_from_slice(&scratch[..n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_2x2() {
        // [[4, 3], [6, 3]] x = [10, 12] → x = [1, 2].
        let lu = Lu::factor(vec![4.0, 3.0, 6.0, 3.0], 2);
        assert!(lu.is_ok());
        let mut b = vec![10.0, 12.0];
        let mut scr = vec![0.0; 2];
        lu.solve(&mut b, &mut scr);
        assert!((b[0] - 1.0).abs() < 1e-10, "{b:?}");
        assert!((b[1] - 2.0).abs() < 1e-10, "{b:?}");
    }

    #[test]
    fn solves_3x3_needs_pivot() {
        // A with a zero leading pivot, forcing a row swap.
        // [[0,1,1],[1,0,1],[1,1,0]] x = [2,2,2] → x = [1,1,1].
        let lu = Lu::factor(vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0], 3);
        assert!(lu.is_ok());
        let mut b = vec![2.0, 2.0, 2.0];
        let mut scr = vec![0.0; 3];
        lu.solve(&mut b, &mut scr);
        for v in &b {
            assert!((v - 1.0).abs() < 1e-10, "{b:?}");
        }
    }

    #[test]
    fn detects_singular() {
        // Rank-deficient: row 2 = 2 × row 1.
        let lu = Lu::factor(vec![1.0, 2.0, 2.0, 4.0], 2);
        assert!(!lu.is_ok());
    }
}
