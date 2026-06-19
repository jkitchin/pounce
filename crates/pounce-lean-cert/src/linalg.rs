//! Exact dense rational linear algebra for the witness computation.
//!
//! Certificate-sized systems only, so plain rational Gaussian elimination is
//! ample — and being exact over ℚ there is no pivoting-for-stability concern, we
//! pivot only to dodge a zero pivot. Everything stays in [`BigRational`].

use num_rational::BigRational;
use num_traits::Zero;

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
}
