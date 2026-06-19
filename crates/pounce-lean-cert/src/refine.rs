//! Mode B: exact rational active-set refinement of a float QP solve.
//!
//! A float `x̃` is feasible/stationary only to ~1e-8, so its lossless rational
//! image is **not** the exact optimizer and a `global-min` proof about it is
//! false. Instead we take the float solve's **active set**, then solve the
//! equality-constrained KKT system *exactly over ℚ*:
//!
//! ```text
//!   [ Q   −A_actᵀ ] [ x* ]   [ −c    ]
//!   [ A_act   0   ] [ λ  ] = [ b_act ]
//! ```
//!
//! The result is the exact rational optimizer of that active face. We then check,
//! exactly, the remaining KKT conditions — dual feasibility (`λ ≥ 0`) and that
//! the *inactive* rows are genuinely satisfied — and refuse (error out) if the
//! active-set guess was wrong, rather than emit a certificate that won't verify.
//!
//! All constraints are pre-normalized by the caller to the one-sided form
//! `A x ≥ b` (the supported v1 slice), so every multiplier is `≥ 0`.

use crate::linalg::{dot, solve_exact};
use num_rational::BigRational;
use num_traits::Zero;
use std::collections::HashSet;

/// The exact rational KKT point (inequalities only).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refined {
    /// The exact optimizer `x*` over ℚ.
    pub x: Vec<BigRational>,
    /// One nonnegative multiplier per constraint (zero for inactive rows),
    /// aligned with the rows of `A`/`b`.
    pub lambda: Vec<BigRational>,
}

/// The exact rational KKT point with equality constraints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefinedEq {
    /// The exact optimizer `x*` over ℚ.
    pub x: Vec<BigRational>,
    /// One nonnegative multiplier per inequality row (zero for inactive),
    /// aligned with `a`/`b`.
    pub lambda: Vec<BigRational>,
    /// One **free-sign** multiplier per equality row, aligned with `e`/`d`.
    pub mu: Vec<BigRational>,
}

/// Why refinement failed — every variant means "do not emit".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RefineError {
    /// The KKT system is singular (degenerate / linearly dependent active set).
    Singular,
    /// Active constraint `constraint` got a negative multiplier — wrong active
    /// set, or the point is not a minimizer.
    NegativeDual { constraint: usize },
    /// Inactive constraint `constraint` is violated at `x*` (`A_i x* < b_i`).
    InactiveViolated { constraint: usize },
    /// Equality row `constraint` is not met exactly at `x*` (`E_j x* ≠ d_j`).
    EqualityResidual { constraint: usize },
    /// Defensive: stationarity did not hold exactly (should be impossible by
    /// construction; guards against a shape bug).
    StationarityResidual,
}

/// Inequality-only refinement — a thin wrapper over [`refine_kkt_eq`] with no
/// equality rows.
pub fn refine_kkt(
    q: &[Vec<BigRational>],
    c: &[BigRational],
    a: &[Vec<BigRational>],
    b: &[BigRational],
    active: &[usize],
) -> Result<Refined, RefineError> {
    let r = refine_kkt_eq(q, c, a, b, active, &[], &[])?;
    Ok(Refined {
        x: r.x,
        lambda: r.lambda,
    })
}

/// Solve the active-set KKT system exactly, **with equality rows**, and validate.
///
/// * `q` — `n×n` symmetric Hessian over ℚ (`M`-of-record; the caller has folded
///   any `½` factor so stationarity reads `Q x* + c = Aᵀλ + Eᵀμ`).
/// * `c` — length-`n` linear objective term.
/// * `a`, `b`, `active` — inequality system `A x ≥ b` (multipliers `λ ≥ 0`) and
///   the indices to treat as active.
/// * `e`, `d` — equality system `E x = d` (free-sign multipliers `μ`); these are
///   always active and `x*` must meet them exactly over ℚ.
pub fn refine_kkt_eq(
    q: &[Vec<BigRational>],
    c: &[BigRational],
    a: &[Vec<BigRational>],
    b: &[BigRational],
    active: &[usize],
    e: &[Vec<BigRational>],
    d: &[BigRational],
) -> Result<RefinedEq, RefineError> {
    let n = c.len();
    let m = a.len();
    let p = e.len();
    let k = active.len();
    let size = n + k + p;

    let mut mat = vec![vec![BigRational::zero(); size]; size];
    let mut rhs = vec![BigRational::zero(); size];

    // Stationarity rows 0..n:  Q x − A_actᵀ λ − Eᵀ μ = −c
    for i in 0..n {
        for j in 0..n {
            mat[i][j] = q[i][j].clone();
        }
        for (ai, &cidx) in active.iter().enumerate() {
            mat[i][n + ai] = -a[cidx][i].clone();
        }
        for (ei, erow) in e.iter().enumerate() {
            mat[i][n + k + ei] = -erow[i].clone();
        }
        rhs[i] = -c[i].clone();
    }
    // Active inequality rows:  A_act x = b_act
    for (ai, &cidx) in active.iter().enumerate() {
        for j in 0..n {
            mat[n + ai][j] = a[cidx][j].clone();
        }
        rhs[n + ai] = b[cidx].clone();
    }
    // Equality rows:  E x = d
    for (ei, erow) in e.iter().enumerate() {
        for j in 0..n {
            mat[n + k + ei][j] = erow[j].clone();
        }
        rhs[n + k + ei] = d[ei].clone();
    }

    let sol = solve_exact(&mat, &rhs).ok_or(RefineError::Singular)?;
    let x = sol[..n].to_vec();

    // Inequality dual feasibility, assemble the full λ.
    let mut lambda = vec![BigRational::zero(); m];
    for (ai, &cidx) in active.iter().enumerate() {
        let lam = &sol[n + ai];
        if lam < &BigRational::zero() {
            return Err(RefineError::NegativeDual { constraint: cidx });
        }
        lambda[cidx] = lam.clone();
    }
    // Equality multipliers are free-sign — no check.
    let mu: Vec<BigRational> = (0..p).map(|ei| sol[n + k + ei].clone()).collect();

    // Inactive inequalities must hold: A_i x* ≥ b_i.
    let active_set: HashSet<usize> = active.iter().copied().collect();
    for (i, row) in a.iter().enumerate() {
        if active_set.contains(&i) {
            continue;
        }
        if dot(row, &x) < b[i] {
            return Err(RefineError::InactiveViolated { constraint: i });
        }
    }
    // Equalities must hold exactly (the Lean theorem's `hxeq` needs this).
    for (ei, erow) in e.iter().enumerate() {
        if dot(erow, &x) != d[ei] {
            return Err(RefineError::EqualityResidual { constraint: ei });
        }
    }

    // Defensive exact stationarity recheck: Q x* + c == Aᵀ λ + Eᵀ μ.
    for i in 0..n {
        let mut lhs = c[i].clone();
        for (j, qx) in q[i].iter().enumerate() {
            lhs += qx * &x[j];
        }
        let mut rhs_i = BigRational::zero();
        for (cidx, row) in a.iter().enumerate() {
            rhs_i += &row[i] * &lambda[cidx];
        }
        for (ei, erow) in e.iter().enumerate() {
            rhs_i += &erow[i] * &mu[ei];
        }
        if lhs != rhs_i {
            return Err(RefineError::StationarityResidual);
        }
    }

    Ok(RefinedEq { x, lambda, mu })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> BigRational {
        BigRational::new(n.into(), d.into())
    }

    #[test]
    fn reference_qp_recovers_exact_optimum() {
        // min x₁²+x₂² (Q=diag(2,2), c=0) s.t. x₁+x₂ ≥ 1, constraint 0 active.
        let q = vec![vec![r(2, 1), r(0, 1)], vec![r(0, 1), r(2, 1)]];
        let c = vec![r(0, 1), r(0, 1)];
        let a = vec![vec![r(1, 1), r(1, 1)]];
        let b = vec![r(1, 1)];
        let refined = refine_kkt(&q, &c, &a, &b, &[0]).unwrap();
        assert_eq!(refined.x, vec![r(1, 2), r(1, 2)], "x* = (1/2, 1/2)");
        assert_eq!(refined.lambda, vec![r(1, 1)], "dual λ = 1");
    }

    #[test]
    fn wrong_active_set_negative_dual_is_rejected() {
        // Unconstrained min is x=0, which is feasible; forcing x₁+x₂≥1 active
        // there would need a negative multiplier. With rhs that makes the
        // active constraint pull the wrong way, λ goes negative.
        // min x₁²+x₂² s.t. x₁+x₂ ≥ −1 (active): KKT gives λ = −1 < 0.
        let q = vec![vec![r(2, 1), r(0, 1)], vec![r(0, 1), r(2, 1)]];
        let c = vec![r(0, 1), r(0, 1)];
        let a = vec![vec![r(1, 1), r(1, 1)]];
        let b = vec![r(-1, 1)];
        assert_eq!(
            refine_kkt(&q, &c, &a, &b, &[0]),
            Err(RefineError::NegativeDual { constraint: 0 })
        );
    }

    #[test]
    fn inactive_violation_is_rejected() {
        // Two constraints; treat only row 0 active, but the exact optimum on
        // row 0's face violates row 1.
        // min x₁²+x₂² s.t. x₁+x₂ ≥ 1 (active), x₁ ≥ 5 (inactive but violated at (1/2,1/2)).
        let q = vec![vec![r(2, 1), r(0, 1)], vec![r(0, 1), r(2, 1)]];
        let c = vec![r(0, 1), r(0, 1)];
        let a = vec![vec![r(1, 1), r(1, 1)], vec![r(1, 1), r(0, 1)]];
        let b = vec![r(1, 1), r(5, 1)];
        assert_eq!(
            refine_kkt(&q, &c, &a, &b, &[0]),
            Err(RefineError::InactiveViolated { constraint: 1 })
        );
    }

    #[test]
    fn equality_constraint_recovers_free_sign_multiplier() {
        // min x₁²+x₂² s.t. x₁+x₂ = 1 (equality). x* = (1/2, 1/2), μ = 1.
        let q = vec![vec![r(2, 1), r(0, 1)], vec![r(0, 1), r(2, 1)]];
        let c = vec![r(0, 1), r(0, 1)];
        let e = vec![vec![r(1, 1), r(1, 1)]];
        let d = vec![r(1, 1)];
        let refined = refine_kkt_eq(&q, &c, &[], &[], &[], &e, &d).unwrap();
        assert_eq!(refined.x, vec![r(1, 2), r(1, 2)], "x* = (1/2, 1/2)");
        assert!(refined.lambda.is_empty());
        assert_eq!(refined.mu, vec![r(1, 1)], "equality multiplier μ = 1");
    }

    #[test]
    fn equality_with_negative_multiplier_is_accepted() {
        // Equalities carry sign-unrestricted multipliers. min x₁²+x₂² s.t.
        // x₁+x₂ = −1 → x* = (−1/2,−1/2), μ = −1 (must NOT be rejected).
        let q = vec![vec![r(2, 1), r(0, 1)], vec![r(0, 1), r(2, 1)]];
        let c = vec![r(0, 1), r(0, 1)];
        let e = vec![vec![r(1, 1), r(1, 1)]];
        let d = vec![r(-1, 1)];
        let refined = refine_kkt_eq(&q, &c, &[], &[], &[], &e, &d).unwrap();
        assert_eq!(refined.x, vec![r(-1, 2), r(-1, 2)]);
        assert_eq!(
            refined.mu,
            vec![r(-1, 1)],
            "μ = −1 is valid for an equality"
        );
    }
}
