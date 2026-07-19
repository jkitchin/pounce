//! Exact rational Gram matrix from the SDP's floating-point one.
//!
//! This is the SOS analogue of [`crate::refine`] and
//! [`crate::refine_farkas`], and it exists for the same reason: the Lean
//! identity `p − γ = m(x)ᵀ G m(x)` must hold **exactly** over ℚ, coefficient by
//! coefficient, and a float Gram never satisfies it exactly.
//!
//! The pattern is the one used throughout: **the float proposes, the exact
//! arithmetic decides.** Here the float Gram is used only to choose values for
//! the *free* parameters of the coefficient-matching system; the constrained
//! entries are then solved for exactly, and the result is checked --- both the
//! polynomial identity and positive-semidefiniteness --- before it is returned.
//!
//! # Why this can genuinely fail
//!
//! Unlike the KKT and Farkas refinements, success is not guaranteed even in
//! principle. The exact solution set of the coefficient-matching system is an
//! affine subspace; rounding the free parameters moves the point inside it, and
//! nothing guarantees the moved point is still positive semidefinite. When the
//! SDP optimum sits on the boundary of the PSD cone --- which is exactly the
//! case for a *tight* SOS bound --- the feasible set can be a single point, so
//! any rounding at all leaves it. The worked example below is such a case: the
//! constraints plus PSD force `G` uniquely.
//!
//! That is why this routine refuses rather than approximating.

use num_rational::BigRational;
use num_traits::{Signed, Zero};

use crate::ldlt::ldlt;

/// Why an exact Gram could not be produced. Every variant means "refuse".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoundError {
    /// Shapes disagree.
    Shape(&'static str),
    /// The coefficient-matching system has no solution for these free values.
    Inconsistent,
    /// A rounded solution exists but is not positive semidefinite, so it
    /// witnesses nothing. Try a finer rounding grid.
    NotPsd,
    /// Defensive: the assembled Gram failed its own exact identity recheck.
    SelfCheck(&'static str),
}

impl std::fmt::Display for RoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for RoundError {}

/// Round `v` to the nearest multiple of `1/denom`, exactly.
fn snap(v: f64, denom: i64) -> BigRational {
    let scaled = (v * denom as f64).round() as i64;
    BigRational::new(scaled.into(), denom.into())
}

/// Upper-triangle index (i ≤ j) into a packed vector of length `bn(bn+1)/2`.
fn ut_index(bn: usize, i: usize, j: usize) -> usize {
    debug_assert!(i <= j && j < bn);
    i * bn - i * i.saturating_sub(1) / 2 + (j - i)
}

/// Produce an exact rational Gram matrix `G` with
/// `p(x) − γ = m(x)ᵀ G m(x)` as a polynomial identity, and `G ⪰ 0`.
///
/// * `p_terms` — the polynomial as `(exponent vector, coefficient)` pairs.
/// * `gamma` — the claimed lower bound, already exact.
/// * `basis` — the monomial basis `m`, as exponent vectors.
/// * `g_float` — the SDP's Gram matrix; a *hint* only.
/// * `denom` — rounding grid for the free parameters.
pub fn round_gram(
    p_terms: &[(Vec<usize>, BigRational)],
    gamma: &BigRational,
    basis: &[Vec<usize>],
    g_float: &[Vec<f64>],
    denom: i64,
) -> Result<Vec<Vec<BigRational>>, RoundError> {
    let bn = basis.len();
    if bn == 0 || g_float.len() != bn || g_float.iter().any(|r| r.len() != bn) {
        return Err(RoundError::Shape("Gram is not bn × bn"));
    }
    if denom <= 0 {
        return Err(RoundError::Shape("denominator must be positive"));
    }
    let nvars = basis[0].len();
    let nunk = bn * (bn + 1) / 2;

    // --- assemble the coefficient-matching system ---------------------------
    //
    // For each monomial α: Σ_{i≤j, basis_i+basis_j = α} c_ij · G_ij = coeff_α,
    // where c_ij is 1 on the diagonal and 2 off it (G is symmetric, and the
    // packed unknown stands for both G_ij and G_ji).
    let mut rows: std::collections::BTreeMap<Vec<usize>, Vec<BigRational>> =
        std::collections::BTreeMap::new();
    for i in 0..bn {
        for j in i..bn {
            let alpha: Vec<usize> = basis[i].iter().zip(&basis[j]).map(|(a, b)| a + b).collect();
            let coef = if i == j { 1i64 } else { 2i64 };
            rows.entry(alpha)
                .or_insert_with(|| vec![BigRational::zero(); nunk])[ut_index(bn, i, j)] +=
                BigRational::from_integer(coef.into());
        }
    }
    // Right-hand side: coefficients of p − γ.
    let mut rhs_of: std::collections::BTreeMap<Vec<usize>, BigRational> =
        std::collections::BTreeMap::new();
    for (e, c) in p_terms {
        if e.len() != nvars {
            return Err(RoundError::Shape("polynomial arity != basis arity"));
        }
        *rhs_of.entry(e.clone()).or_insert_with(BigRational::zero) += c;
    }
    let zero_exp = vec![0usize; nvars];
    *rhs_of.entry(zero_exp).or_insert_with(BigRational::zero) -= gamma;

    // Every monomial mentioned by either side becomes an equation. A monomial
    // in p that no basis product can reach yields an all-zero row with nonzero
    // rhs, i.e. an inconsistency — caught below rather than silently ignored.
    let mut all: Vec<Vec<usize>> = rows.keys().cloned().collect();
    for k in rhs_of.keys() {
        if !rows.contains_key(k) {
            all.push(k.clone());
        }
    }
    all.sort();
    all.dedup();

    let mut a: Vec<Vec<BigRational>> = Vec::with_capacity(all.len());
    let mut b: Vec<BigRational> = Vec::with_capacity(all.len());
    for alpha in &all {
        a.push(
            rows.get(alpha)
                .cloned()
                .unwrap_or_else(|| vec![BigRational::zero(); nunk]),
        );
        b.push(rhs_of.get(alpha).cloned().unwrap_or_else(BigRational::zero));
    }

    // --- exact RREF of [A | b] ---------------------------------------------
    let m = a.len();
    let mut pivot_col_of_row: Vec<Option<usize>> = vec![None; m];
    let mut pivot_row_of_col: Vec<Option<usize>> = vec![None; nunk];
    let mut r = 0usize;
    for c in 0..nunk {
        if r >= m {
            break;
        }
        let Some(pr) = (r..m).find(|&i| !a[i][c].is_zero()) else {
            continue;
        };
        a.swap(r, pr);
        b.swap(r, pr);
        let piv = a[r][c].clone();
        for k in 0..nunk {
            a[r][k] = &a[r][k] / &piv;
        }
        b[r] = &b[r] / &piv;
        for i in 0..m {
            if i != r && !a[i][c].is_zero() {
                let f = a[i][c].clone();
                for k in 0..nunk {
                    let s = &f * &a[r][k];
                    a[i][k] -= s;
                }
                let s = &f * &b[r];
                b[i] -= s;
            }
        }
        pivot_col_of_row[r] = Some(c);
        pivot_row_of_col[c] = Some(r);
        r += 1;
    }
    // Any all-zero row with a nonzero rhs means no solution exists at all.
    for i in 0..m {
        if a[i].iter().all(BigRational::is_zero) && !b[i].is_zero() {
            return Err(RoundError::Inconsistent);
        }
    }

    // --- free parameters take their (rounded) float values ------------------
    let mut g = vec![BigRational::zero(); nunk];
    for c in 0..nunk {
        if pivot_row_of_col[c].is_none() {
            // recover (i, j) for this packed index
            let (mut ii, mut jj) = (0usize, 0usize);
            'outer: for i in 0..bn {
                for j in i..bn {
                    if ut_index(bn, i, j) == c {
                        ii = i;
                        jj = j;
                        break 'outer;
                    }
                }
            }
            g[c] = snap(g_float[ii][jj], denom);
        }
    }
    // --- solve the pivots exactly against those choices ---------------------
    for i in (0..m).rev() {
        let Some(pc) = pivot_col_of_row[i] else {
            continue;
        };
        let mut acc = b[i].clone();
        for c in (pc + 1)..nunk {
            if !a[i][c].is_zero() {
                acc -= &a[i][c] * &g[c];
            }
        }
        g[pc] = acc;
    }

    // --- unpack to a dense symmetric matrix ---------------------------------
    let mut gm = vec![vec![BigRational::zero(); bn]; bn];
    for i in 0..bn {
        for j in i..bn {
            let v = g[ut_index(bn, i, j)].clone();
            gm[i][j] = v.clone();
            gm[j][i] = v;
        }
    }

    // --- exact self-checks: identity, then PSD ------------------------------
    for (idx, alpha) in all.iter().enumerate() {
        let lhs: BigRational = (0..bn)
            .flat_map(|i| (i..bn).map(move |j| (i, j)))
            .filter(|&(i, j)| {
                let s: Vec<usize> = basis[i].iter().zip(&basis[j]).map(|(x, y)| x + y).collect();
                &s == alpha
            })
            .map(|(i, j)| {
                let c = if i == j { 1i64 } else { 2i64 };
                BigRational::from_integer(c.into()) * &gm[i][j]
            })
            .sum();
        if lhs != b_original(&rhs_of, alpha) {
            let _ = idx;
            return Err(RoundError::SelfCheck("coefficient identity"));
        }
    }
    // PSD via exact LDLᵀ: a factorization with nonnegative diagonal exists iff
    // the matrix is PSD (for the unit-lower form used here).
    match ldlt(&gm) {
        Ok(f) if f.d.iter().all(|v| !v.is_negative()) => Ok(gm),
        _ => Err(RoundError::NotPsd),
    }
}

fn b_original(
    rhs_of: &std::collections::BTreeMap<Vec<usize>, BigRational>,
    alpha: &[usize],
) -> BigRational {
    rhs_of.get(alpha).cloned().unwrap_or_else(BigRational::zero)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn r(n: i64) -> BigRational {
        BigRational::from_integer(n.into())
    }

    /// `p = x⁴ − 2x² + 2`, minimum 1. The SOS certificate is `(x² − 1)²`, i.e.
    /// `G = [[1,0,-1],[0,0,0],[-1,0,1]]` in the basis `(1, x, x²)`.
    ///
    /// PSD forces this uniquely: matching gives `G₁₁ = −2 − 2G₀₂`, so
    /// `G₁₁ ≥ 0` needs `G₀₂ ≤ −1` while the `(0,2)` minor needs `|G₀₂| ≤ 1`.
    /// A float Gram near that point must therefore round to exactly `−1`.
    #[test]
    fn recovers_the_exact_sos_certificate_for_the_quartic() {
        let p = vec![
            (vec![4usize], r(1)),
            (vec![2usize], r(-2)),
            (vec![0usize], r(2)),
        ];
        let basis = vec![vec![0usize], vec![1usize], vec![2usize]];
        // A plausible SDP output: near the exact answer but not equal to it.
        let g_float = vec![
            vec![1.0000000003, 1e-11, -0.9999999997],
            vec![1e-11, 2.4e-10, -3e-11],
            vec![-0.9999999997, -3e-11, 1.0000000002],
        ];
        let g = round_gram(&p, &r(1), &basis, &g_float, 1).unwrap();
        assert_eq!(g[0][0], r(1));
        assert_eq!(g[0][2], r(-1));
        assert_eq!(g[1][1], r(0));
        assert_eq!(g[2][2], r(1));
    }

    /// The identity must hold as an exact polynomial identity, not merely at
    /// the points a test happens to sample.
    #[test]
    fn the_returned_gram_satisfies_the_identity_exactly() {
        let p = vec![
            (vec![4usize], r(1)),
            (vec![2usize], r(-2)),
            (vec![0usize], r(2)),
        ];
        let basis = vec![vec![0usize], vec![1usize], vec![2usize]];
        let g_float = vec![
            vec![1.0, 0.0, -1.0],
            vec![0.0, 0.0, 0.0],
            vec![-1.0, 0.0, 1.0],
        ];
        let g = round_gram(&p, &r(1), &basis, &g_float, 1).unwrap();
        for x in [-3i64, -2, -1, 0, 1, 2, 3] {
            let xv = BigRational::from_integer(x.into());
            let m: Vec<BigRational> = (0..3).map(|k| xv.pow(k as i32)).collect();
            let quad: BigRational = (0..3)
                .flat_map(|i| (0..3).map(move |j| (i, j)))
                .map(|(i, j)| &m[i] * &g[i][j] * &m[j])
                .sum();
            let target = xv.pow(4) - r(2) * xv.pow(2) + r(2) - r(1);
            assert_eq!(quad, target, "identity fails at x = {x}");
        }
    }

    /// A bound *above* the true minimum cannot be certified: the residual
    /// polynomial is negative somewhere, so no PSD Gram exists.
    #[test]
    fn a_bound_above_the_minimum_is_refused() {
        let p = vec![
            (vec![4usize], r(1)),
            (vec![2usize], r(-2)),
            (vec![0usize], r(2)),
        ];
        let basis = vec![vec![0usize], vec![1usize], vec![2usize]];
        let g_float = vec![
            vec![1.0, 0.0, -1.0],
            vec![0.0, 0.0, 0.0],
            vec![-1.0, 0.0, 1.0],
        ];
        // True minimum is 1; claim 2.
        let err = round_gram(&p, &r(2), &basis, &g_float, 1).unwrap_err();
        assert_eq!(err, RoundError::NotPsd, "γ = 2 exceeds the minimum");
    }

    /// A polynomial term the basis cannot reach makes the system inconsistent
    /// rather than silently dropping the term.
    #[test]
    fn unreachable_monomial_is_inconsistent() {
        // x⁶ cannot be formed from products of (1, x, x²).
        let p = vec![(vec![6usize], r(1))];
        let basis = vec![vec![0usize], vec![1usize], vec![2usize]];
        let g_float = vec![vec![0.0; 3]; 3];
        let err = round_gram(&p, &r(0), &basis, &g_float, 1).unwrap_err();
        assert_eq!(err, RoundError::Inconsistent);
    }
}
