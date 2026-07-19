//! Exact dense rational linear algebra for the witness computation.
//!
//! Certificate-sized systems only, so plain rational Gaussian elimination is
//! ample — and being exact over ℚ there is no pivoting-for-stability concern, we
//! pivot only to dodge a zero pivot. Everything stays in [`BigRational`].

use num_rational::BigRational;
use num_traits::{One, Zero};

/// Exact dot product `row · x`.
pub fn dot(row: &[BigRational], x: &[BigRational]) -> BigRational {
    let mut acc = BigRational::zero();
    for (a, b) in row.iter().zip(x.iter()) {
        acc += a * b;
    }
    acc
}

/// Solve `A y = b` exactly over ℚ. Returns `None` if `A` is singular or the
/// shapes are inconsistent.
#[allow(clippy::needless_range_loop)] // Gaussian elimination indexes m and rhs together
pub fn solve_exact(a: &[Vec<BigRational>], b: &[BigRational]) -> Option<Vec<BigRational>> {
    let n = a.len();
    if n == 0 {
        return Some(Vec::new());
    }
    if a.iter().any(|row| row.len() != n) || b.len() != n {
        return None;
    }

    let mut m: Vec<Vec<BigRational>> = a.to_vec();
    let mut rhs: Vec<BigRational> = b.to_vec();

    // Forward elimination with first-nonzero pivoting.
    for col in 0..n {
        let pivot = (col..n).find(|&r| !m[r][col].is_zero())?;
        m.swap(col, pivot);
        rhs.swap(col, pivot);

        for r in (col + 1)..n {
            if m[r][col].is_zero() {
                continue;
            }
            let factor = &m[r][col] / &m[col][col];
            for k in col..n {
                let sub = &factor * &m[col][k];
                m[r][k] -= sub;
            }
            let sub = &factor * &rhs[col];
            rhs[r] -= sub;
        }
    }

    // Back substitution.
    let mut x = vec![BigRational::zero(); n];
    for i in (0..n).rev() {
        let mut s = rhs[i].clone();
        for k in (i + 1)..n {
            s -= &m[i][k] * &x[k];
        }
        x[i] = &s / &m[i][i];
    }
    Some(x)
}

/// Exact basis of the null space of an `rows × cols` matrix, by reduced row
/// echelon form over ℚ.
///
/// Returns one basis vector per free column; an empty result means the null
/// space is trivial. Exactness means there is no rank-tolerance question — a
/// pivot is zero or it is not.
#[allow(clippy::needless_range_loop)]
pub fn nullspace_exact(m_in: &[Vec<BigRational>], cols: usize) -> Vec<Vec<BigRational>> {
    let rows = m_in.len();
    if cols == 0 {
        return Vec::new();
    }
    // NB: `rows == 0` is *not* an early return. A matrix with no rows constrains
    // nothing, so its null space is all of ℝ^cols and the basis is the identity
    // — which is exactly what the loop below produces (no pivots, every column
    // free). Returning an empty basis here would claim the opposite.

    let mut m: Vec<Vec<BigRational>> = m_in.to_vec();
    let mut pivot_of_col: Vec<Option<usize>> = vec![None; cols];
    let mut r = 0usize;

    for c in 0..cols {
        if r >= rows {
            break;
        }
        let Some(p) = (r..rows).find(|&i| !m[i][c].is_zero()) else {
            continue;
        };
        m.swap(r, p);
        let piv = m[r][c].clone();
        for k in 0..cols {
            m[r][k] = &m[r][k] / &piv;
        }
        for i in 0..rows {
            if i != r && !m[i][c].is_zero() {
                let f = m[i][c].clone();
                for k in 0..cols {
                    let sub = &f * &m[r][k];
                    m[i][k] -= sub;
                }
            }
        }
        pivot_of_col[c] = Some(r);
        r += 1;
    }

    let mut basis = Vec::new();
    for c in 0..cols {
        if pivot_of_col[c].is_some() {
            continue;
        }
        let mut v = vec![BigRational::zero(); cols];
        v[c] = BigRational::one();
        for (cp, piv) in pivot_of_col.iter().enumerate() {
            if let Some(rp) = piv {
                v[cp] = -m[*rp][c].clone();
            }
        }
        basis.push(v);
    }
    basis
}

/// Indices of a maximal linearly independent subset of `rows`, chosen greedily
/// in the order given, by exact forward elimination over ℚ.
///
/// Used to pick a **basis** from a degenerate active set. Real LPs routinely
/// have more constraints active than there are variables, which makes the KKT
/// system rank-deficient; selecting an independent subset restores
/// nonsingularity. Row order is the caller's priority — rows offered earlier
/// win ties.
///
/// Being exact, "increases the rank" is decidable: a pivot is zero or it is not.
/// There is no threshold to tune, which is precisely what makes this safe to do
/// automatically.
pub fn select_independent_rows(rows: &[Vec<BigRational>]) -> Vec<usize> {
    let mut basis: Vec<Vec<BigRational>> = Vec::new();
    let mut chosen: Vec<usize> = Vec::new();

    for (idx, row) in rows.iter().enumerate() {
        // Reduce `row` against the echelon rows accumulated so far.
        let mut v = row.clone();
        for br in &basis {
            let Some(pcol) = br.iter().position(|x| !x.is_zero()) else {
                continue;
            };
            if v[pcol].is_zero() {
                continue;
            }
            let f = &v[pcol] / &br[pcol];
            for (vi, bi) in v.iter_mut().zip(br.iter()) {
                *vi -= &f * bi;
            }
        }
        if v.iter().any(|x| !x.is_zero()) {
            basis.push(v);
            chosen.push(idx);
        }
    }
    chosen
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> BigRational {
        BigRational::new(n.into(), d.into())
    }

    #[test]
    fn solves_a_small_system() {
        // [2 1; 1 3] y = [1; 2]  ->  y = (1/5, 3/5)
        let a = vec![vec![r(2, 1), r(1, 1)], vec![r(1, 1), r(3, 1)]];
        let b = vec![r(1, 1), r(2, 1)];
        let y = solve_exact(&a, &b).unwrap();
        assert_eq!(y, vec![r(1, 5), r(3, 5)]);
    }

    #[test]
    fn needs_pivot_when_leading_zero() {
        // [0 1; 1 0] y = [2; 3] -> y = (3, 2)
        let a = vec![vec![r(0, 1), r(1, 1)], vec![r(1, 1), r(0, 1)]];
        let b = vec![r(2, 1), r(3, 1)];
        let y = solve_exact(&a, &b).unwrap();
        assert_eq!(y, vec![r(3, 1), r(2, 1)]);
    }

    #[test]
    fn singular_returns_none() {
        let a = vec![vec![r(1, 1), r(2, 1)], vec![r(2, 1), r(4, 1)]];
        let b = vec![r(1, 1), r(1, 1)];
        assert!(solve_exact(&a, &b).is_none());
    }

    #[test]
    fn selects_a_basis_from_dependent_rows() {
        // r2 = 2*r0, so it must be dropped; r1 and r3 are independent of r0.
        let rows = vec![
            vec![r(1, 1), r(0, 1)],
            vec![r(0, 1), r(1, 1)],
            vec![r(2, 1), r(0, 1)],
            vec![r(1, 1), r(1, 1)],
        ];
        // Only two independent directions exist in ℚ².
        assert_eq!(select_independent_rows(&rows), vec![0, 1]);
    }

    #[test]
    fn keeps_all_rows_when_independent() {
        let rows = vec![vec![r(1, 1), r(0, 1)], vec![r(0, 1), r(1, 1)]];
        assert_eq!(select_independent_rows(&rows), vec![0, 1]);
    }

    #[test]
    fn earlier_rows_win_ties() {
        // Identical rows: the first is kept, the duplicate dropped.
        let rows = vec![vec![r(3, 1), r(1, 1)], vec![r(3, 1), r(1, 1)]];
        assert_eq!(select_independent_rows(&rows), vec![0]);
    }
}
