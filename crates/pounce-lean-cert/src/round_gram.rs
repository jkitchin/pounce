//! Exact rational Gram matrices from the SDP's floating-point ones.
//!
//! This is the SOS analogue of [`crate::refine`] and
//! [`crate::refine_farkas`], and it exists for the same reason: the Lean
//! identity must hold **exactly** over ℚ, coefficient by coefficient, and a
//! float Gram never satisfies it exactly.
//!
//! Two identities, one machine. Unconstrained:
//!
//! ```text
//!     p(x) − γ = m(x)ᵀ G m(x),                          G ⪰ 0
//! ```
//!
//! and Putinar, for `min p(x)` subject to `gₖ(x) ≥ 0`:
//!
//! ```text
//!     p(x) − γ = σ₀(x) + Σₖ σₖ(x)·gₖ(x),   each σ a Gram form, each G ⪰ 0.
//! ```
//!
//! The second is the first with more blocks and a polynomial attached to each,
//! so [`round_gram_blocks`] is the real routine and [`round_gram`] is the
//! one-block, multiplier-`1` case of it. Coefficient matching stays a single
//! linear system over *all* blocks' entries at once — which it has to be, since
//! the blocks are only jointly constrained: no individual block satisfies
//! anything on its own.
//!
//! The pattern is the one used throughout: **the float proposes, the exact
//! arithmetic decides.** The float Grams are used only to choose values for the
//! *free* parameters of that system; the constrained entries are then solved
//! for exactly, and the result is checked --- both the polynomial identity and
//! positive-semidefiniteness of every block --- before it is returned.
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
    /// Block `block`'s rounded solution is not positive semidefinite, so it
    /// witnesses nothing. Try a finer rounding grid.
    ///
    /// Named per block because in the Putinar case they fail separately and for
    /// different reasons: `σ₀` is pushed off the cone by a tight bound, whereas
    /// a localizing `σₖ` more often goes indefinite because the relaxation
    /// order is too low for that constraint to carry its share.
    NotPsd { block: usize },
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

/// One SOS block of a Putinar certificate: a Gram form `m(x)ᵀ G m(x)` together
/// with the constraint polynomial it multiplies.
///
/// `σ₀` is the block whose `multiplier` is the constant `1`; there is nothing
/// else special about it, which is why it is not a separate parameter.
#[derive(Clone, Copy, Debug)]
pub struct BlockSpec<'a> {
    /// The monomial lift `m`, as exponent vectors.
    pub basis: &'a [Vec<usize>],
    /// The constraint polynomial `g` this block multiplies, as
    /// `(exponent vector, coefficient)` pairs. Pass `[(0…0, 1)]` for `σ₀`.
    pub multiplier: &'a [(Vec<usize>, BigRational)],
    /// The SDP's Gram matrix for this block; a *hint* only.
    pub gram_float: &'a [Vec<f64>],
}

/// Produce an exact rational Gram matrix `G` with
/// `p(x) − γ = m(x)ᵀ G m(x)` as a polynomial identity, and `G ⪰ 0`.
///
/// The unconstrained case: one block, multiplied by the constant `1`. See
/// [`round_gram_blocks`] for what actually happens.
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
    if basis.is_empty() {
        return Err(RoundError::Shape("empty basis"));
    }
    let one = [(
        vec![0usize; basis[0].len()],
        BigRational::from_integer(1.into()),
    )];
    let blocks = [BlockSpec {
        basis,
        multiplier: &one,
        gram_float: g_float,
    }];
    let mut out = round_gram_blocks(p_terms, gamma, &blocks, denom)?;
    Ok(out.remove(0))
}

/// Produce exact rational Gram matrices for a Putinar certificate:
///
/// ```text
///     p(x) − γ = Σₖ mₖ(x)ᵀ Gₖ mₖ(x) · gₖ(x),   every Gₖ ⪰ 0,
/// ```
///
/// as a polynomial identity over ℚ, with `gₖ` the block's `multiplier`.
///
/// The blocks are solved **together**, in one linear system over every block's
/// packed upper triangle. That is not an optimization: a block satisfies
/// nothing on its own, and the only true statement is about their sum. Solving
/// them one at a time would have no meaning to give the intermediate results.
///
/// PSD-ness, by contrast, *is* per block, and is checked per block — with the
/// failing index reported, since a `σ₀` that leaves the cone and a localizing
/// `σₖ` that does are different diagnoses.
pub fn round_gram_blocks(
    p_terms: &[(Vec<usize>, BigRational)],
    gamma: &BigRational,
    blocks: &[BlockSpec<'_>],
    denom: i64,
) -> Result<Vec<Vec<Vec<BigRational>>>, RoundError> {
    if blocks.is_empty() {
        return Err(RoundError::Shape("no SOS blocks"));
    }
    if denom <= 0 {
        return Err(RoundError::Shape("denominator must be positive"));
    }
    for b in blocks {
        let bn = b.basis.len();
        if bn == 0 || b.gram_float.len() != bn || b.gram_float.iter().any(|r| r.len() != bn) {
            return Err(RoundError::Shape("Gram is not bn × bn"));
        }
        if b.multiplier.is_empty() {
            return Err(RoundError::Shape("a block multiplier has no terms"));
        }
    }
    let nvars = blocks[0].basis[0].len();
    if blocks
        .iter()
        .any(|b| b.basis.iter().any(|e| e.len() != nvars))
        || blocks
            .iter()
            .any(|b| b.multiplier.iter().any(|(e, _)| e.len() != nvars))
    {
        return Err(RoundError::Shape(
            "blocks disagree on the number of variables",
        ));
    }

    // Packed layout: block k's upper triangle occupies `offset[k] .. offset[k+1]`.
    let mut offset = Vec::with_capacity(blocks.len() + 1);
    let mut nunk = 0usize;
    for b in blocks {
        offset.push(nunk);
        let bn = b.basis.len();
        nunk += bn * (bn + 1) / 2;
    }
    offset.push(nunk);

    // --- assemble the coefficient-matching system ---------------------------
    //
    // For each monomial α, and each block k, each pair i≤j and each term
    // (β, cβ) of gₖ with basisₖ_i + basisₖ_j + β = α:
    //
    //     Σ  c_ij · cβ · G^k_ij  =  coeff_α(p − γ),
    //
    // where c_ij is 1 on the diagonal and 2 off it (G is symmetric, and the
    // packed unknown stands for both G_ij and G_ji).
    let mut rows: std::collections::BTreeMap<Vec<usize>, Vec<BigRational>> =
        std::collections::BTreeMap::new();
    for (k, b) in blocks.iter().enumerate() {
        let bn = b.basis.len();
        for i in 0..bn {
            for j in i..bn {
                let sym = BigRational::from_integer(if i == j { 1i64 } else { 2i64 }.into());
                let col = offset[k] + ut_index(bn, i, j);
                for (beta, cbeta) in b.multiplier {
                    if cbeta.is_zero() {
                        continue;
                    }
                    let alpha: Vec<usize> = (0..nvars)
                        .map(|v| b.basis[i][v] + b.basis[j][v] + beta[v])
                        .collect();
                    rows.entry(alpha)
                        .or_insert_with(|| vec![BigRational::zero(); nunk])[col] += &sym * cbeta;
                }
            }
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
    // in p that no block can reach yields an all-zero row with nonzero rhs,
    // i.e. an inconsistency — caught below rather than silently ignored.
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
            let (k, i, j) = unpack(&offset, blocks, c);
            g[c] = snap(blocks[k].gram_float[i][j], denom);
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

    // --- unpack to dense symmetric matrices ---------------------------------
    let mut out: Vec<Vec<Vec<BigRational>>> = Vec::with_capacity(blocks.len());
    for (k, blk) in blocks.iter().enumerate() {
        let bn = blk.basis.len();
        let mut gm = vec![vec![BigRational::zero(); bn]; bn];
        for i in 0..bn {
            for j in i..bn {
                let v = g[offset[k] + ut_index(bn, i, j)].clone();
                gm[i][j] = v.clone();
                gm[j][i] = v;
            }
        }
        out.push(gm);
    }

    // --- exact self-checks: identity, then PSD ------------------------------
    //
    // The identity is rechecked from the *assembled* matrices rather than from
    // the solved vector, so a mistake in the packing would be caught too.
    for alpha in &all {
        let mut lhs = BigRational::zero();
        for (k, blk) in blocks.iter().enumerate() {
            let bn = blk.basis.len();
            for i in 0..bn {
                for j in i..bn {
                    let sym = BigRational::from_integer(if i == j { 1i64 } else { 2i64 }.into());
                    for (beta, cbeta) in blk.multiplier {
                        let s: Vec<usize> = (0..nvars)
                            .map(|v| blk.basis[i][v] + blk.basis[j][v] + beta[v])
                            .collect();
                        if &s == alpha {
                            lhs += &sym * cbeta * &out[k][i][j];
                        }
                    }
                }
            }
        }
        if lhs != rhs_of.get(alpha).cloned().unwrap_or_else(BigRational::zero) {
            return Err(RoundError::SelfCheck("coefficient identity"));
        }
    }
    // PSD via exact LDLᵀ: a factorization with nonnegative diagonal exists iff
    // the matrix is PSD (for the unit-lower form used here).
    for (k, gm) in out.iter().enumerate() {
        match ldlt(gm) {
            Ok(f) if f.d.iter().all(|v| !v.is_negative()) => {}
            _ => return Err(RoundError::NotPsd { block: k }),
        }
    }
    Ok(out)
}

/// Recover `(block, i, j)` from a packed column index.
fn unpack(offset: &[usize], blocks: &[BlockSpec<'_>], col: usize) -> (usize, usize, usize) {
    let k = offset.partition_point(|&o| o <= col) - 1;
    let bn = blocks[k].basis.len();
    let local = col - offset[k];
    for i in 0..bn {
        for j in i..bn {
            if ut_index(bn, i, j) == local {
                return (k, i, j);
            }
        }
    }
    (k, 0, 0)
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
        assert_eq!(
            err,
            RoundError::NotPsd { block: 0 },
            "γ = 2 exceeds the minimum"
        );
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

    // --- the constrained (Putinar) case -------------------------------------

    /// `σ₀`'s multiplier: the constant polynomial 1, in `n` variables.
    fn one(n: usize) -> Vec<(Vec<usize>, BigRational)> {
        vec![(vec![0usize; n], r(1))]
    }

    /// `min x² s.t. x − 1 ≥ 0`, minimum 1 at `x = 1`.
    ///
    /// The case that motivates the whole constrained path: `p − 1 = x² − 1` is
    /// **not** a sum of squares — it is negative on `(−1, 1)` — so no
    /// unconstrained certificate for this bound exists at any basis size. Modulo
    /// the constraint it is easy:
    ///
    /// ```text
    ///     x² − 1 = (x − 1)² + 2·(x − 1)
    /// ```
    ///
    /// so `σ₀ = (x−1)²` on the basis `(1, x)` and `σ₁ = 2` on the basis `(1)`.
    #[test]
    fn a_bound_that_is_sos_only_modulo_the_constraint() {
        let p = vec![(vec![2usize], r(1))];
        let g1 = vec![(vec![1usize], r(1)), (vec![0usize], r(-1))]; // x − 1
        let basis0 = vec![vec![0usize], vec![1usize]];
        let basis1 = vec![vec![0usize]];
        // A plausible SDP output: near the exact answer, not equal to it.
        let f0 = vec![
            vec![1.0000000004, -0.9999999996],
            vec![-0.9999999996, 1.0000000002],
        ];
        let f1 = vec![vec![1.9999999997]];
        let out = round_gram_blocks(
            &p,
            &r(1),
            &[
                BlockSpec {
                    basis: &basis0,
                    multiplier: &one(1),
                    gram_float: &f0,
                },
                BlockSpec {
                    basis: &basis1,
                    multiplier: &g1,
                    gram_float: &f1,
                },
            ],
            1,
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        // σ₀ = (x − 1)², i.e. G₀ = [[1,-1],[-1,1]] on (1, x).
        assert_eq!(out[0], vec![vec![r(1), r(-1)], vec![r(-1), r(1)]]);
        // σ₁ = 2.
        assert_eq!(out[1], vec![vec![r(2)]]);
    }

    /// The same bound really is out of reach without the constraint, so the
    /// previous test is measuring the constrained path and not a coincidence.
    #[test]
    fn the_same_bound_is_unreachable_unconstrained() {
        let p = vec![(vec![2usize], r(1))];
        let basis = vec![vec![0usize], vec![1usize]];
        let g_float = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(
            round_gram(&p, &r(1), &basis, &g_float, 1).unwrap_err(),
            RoundError::NotPsd { block: 0 },
            "x² − 1 is negative on (−1, 1), so it is not a sum of squares"
        );
    }

    /// The identity is a *joint* claim, so it must be verified as one: evaluate
    /// `σ₀(x) + σ₁(x)·g(x)` against `p(x) − γ` at sample points. (The routine's
    /// own check is on coefficients; this one is on values, so a packing error
    /// that happened to preserve the coefficient sums would still show up.)
    #[test]
    fn the_putinar_identity_holds_pointwise() {
        let p = vec![(vec![2usize], r(1))];
        let g1 = vec![(vec![1usize], r(1)), (vec![0usize], r(-1))];
        let basis0 = vec![vec![0usize], vec![1usize]];
        let basis1 = vec![vec![0usize]];
        let f0 = vec![vec![1.0, -1.0], vec![-1.0, 1.0]];
        let f1 = vec![vec![2.0]];
        let out = round_gram_blocks(
            &p,
            &r(1),
            &[
                BlockSpec {
                    basis: &basis0,
                    multiplier: &one(1),
                    gram_float: &f0,
                },
                BlockSpec {
                    basis: &basis1,
                    multiplier: &g1,
                    gram_float: &f1,
                },
            ],
            1,
        )
        .unwrap();
        for x in [-3i64, -1, 0, 1, 2, 5] {
            let xv = BigRational::from_integer(x.into());
            let m0 = [r(1), xv.clone()];
            let sigma0: BigRational = (0..2)
                .flat_map(|i| (0..2).map(move |j| (i, j)))
                .map(|(i, j)| &m0[i] * &out[0][i][j] * &m0[j])
                .sum();
            let sigma1 = out[1][0][0].clone();
            let gx = &xv - r(1);
            assert_eq!(
                sigma0 + sigma1 * gx,
                xv.pow(2) - r(1),
                "Putinar identity fails at x = {x}"
            );
        }
    }

    /// A localizing block that goes indefinite is named as **block 1**, so a
    /// user is not sent to inspect the wrong multiplier.
    ///
    /// Same problem as above, but `σ₀`'s basis is cut to the constant `(1)`,
    /// which leaves the system with no freedom at all: matching
    /// `x² − 1 = σ₀ + σ₁·(x − 1)` term by term forces `σ₁ = x + 1` and
    /// `σ₀ = 0`. So `G₁ = [[1, ½], [½, 0]]`, whose determinant is `−¼` — `x + 1`
    /// is not a sum of squares, and no rounding grid can change that. `G₀ = [[0]]`
    /// stays perfectly PSD throughout, which is what makes the index informative.
    #[test]
    fn an_indefinite_localizing_block_is_named() {
        let p = vec![(vec![2usize], r(1))];
        let g1 = vec![(vec![1usize], r(1)), (vec![0usize], r(-1))];
        let basis0 = vec![vec![0usize]];
        let basis1 = vec![vec![0usize], vec![1usize]];
        let f0 = vec![vec![0.0]];
        let f1 = vec![vec![1.0, 0.5], vec![0.5, 0.0]];
        let err = round_gram_blocks(
            &p,
            &r(1),
            &[
                BlockSpec {
                    basis: &basis0,
                    multiplier: &one(1),
                    gram_float: &f0,
                },
                BlockSpec {
                    basis: &basis1,
                    multiplier: &g1,
                    gram_float: &f1,
                },
            ],
            1,
        )
        .unwrap_err();
        assert_eq!(err, RoundError::NotPsd { block: 1 });
    }

    /// Blocks must agree on the number of variables — a mismatch is a caller
    /// bug and is refused rather than silently indexing out of range.
    #[test]
    fn blocks_must_agree_on_arity() {
        let p = vec![(vec![2usize, 0usize], r(1))];
        let basis0 = vec![vec![0usize, 0usize]];
        let basis1 = vec![vec![0usize]]; // one variable, not two
        let f = vec![vec![1.0]];
        let err = round_gram_blocks(
            &p,
            &r(0),
            &[
                BlockSpec {
                    basis: &basis0,
                    multiplier: &one(2),
                    gram_float: &f,
                },
                BlockSpec {
                    basis: &basis1,
                    multiplier: &one(1),
                    gram_float: &f,
                },
            ],
            1,
        )
        .unwrap_err();
        assert!(matches!(err, RoundError::Shape(_)));
    }
}
