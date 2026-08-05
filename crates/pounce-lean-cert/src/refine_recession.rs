//! Exact rational recession direction, refined from the solver's diverging
//! iterate.
//!
//! The `unbounded` verdict rests on a direction `d` satisfying three things at
//! once: `Q d = 0`, `A d ≥ 0`, and `c·d < 0`. Two of those are inequalities,
//! and an inequality satisfied with margin survives the lossless f64→ℚ
//! conversion intact — the exact value is still on the correct side of zero.
//! That is why the LP slice (`Q = 0`) needed no refinement at all: the only
//! *equality* in the list was vacuous.
//!
//! A nonzero `Q` puts it back. `Q d = 0` is an exact equality over ℚ, and a
//! float direction will miss it for the same reason a float Farkas ray misses
//! `Aᵀy = 0` — see [`crate::refine_farkas`], where the residual measured
//! `−103801/262144` on a ray the solver considered converged to 1.7e-11.
//! Copying the solver's direction would produce a certificate that always
//! fails.
//!
//! So the float is used only as a **hint about which flat direction the solve
//! ran away along**, and the direction itself is recomputed exactly: project
//! the hint onto `ker Q` over ℚ, which makes `Q d = 0` true by construction
//! rather than approximately. The two inequalities are then checked exactly on
//! the projected direction, and a failure is a refusal.
//!
//! # What the hint carries, and what it does not
//!
//! The iterate is `x_finite + t·d_true` for a large `t`, so its projection onto
//! `ker Q` is `t·d_true` plus whatever part of `x_finite` also lies in the
//! kernel. That contamination is `O(1)` against a term of size `t`, so after
//! normalization it is `O(1/t)` — invisible to a strict inequality, but able to
//! tip a row that satisfies `A_i d = 0` *exactly* to the wrong side. Such a row
//! is refused rather than nudged: no tolerance is introduced to absorb it,
//! because a tolerance here would be a tolerance in the proof.

use num_rational::BigRational;
use num_traits::{Signed, Zero};

use crate::linalg::{dot, nullspace_exact, solve_exact};

/// Why an exact recession direction could not be produced. Every variant means
/// "refuse", never "emit something weaker".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecessionError {
    /// `Q` has trivial null space — the objective curves in *every* direction,
    /// so no ray can leave the quadratic term unchanged and the objective is
    /// bounded below along every line. Not an obstacle to route around: it says
    /// `unbounded` was the wrong verdict for this problem.
    NoFlatDirection,
    /// The hint projects to zero on `ker Q`: the direction the solve ran along
    /// is entirely curvature, so there is nothing to travel down.
    ZeroDirection,
    /// Row `constraint` fails `A d ≥ 0` — travelling along `d` leaves the
    /// feasible set, so it is not a recession direction of *this* problem.
    LeavesFeasibleSet { constraint: usize },
    /// `c·d ≥ 0` — the direction does not strictly decrease the objective.
    NotDescending,
    /// Defensive: the projected direction failed its own exact recheck.
    SelfCheck(&'static str),
}

impl std::fmt::Display for RecessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for RecessionError {}

/// Refine a hinted direction into an exact rational recession direction.
///
/// * `q` — the `n×n` symmetric Hessian over ℚ, as the Lean statement sees it.
/// * `c` — the linear objective term.
/// * `a_rows` — the inequality system `A x ≥ b`, one row per constraint. Only
///   `A` is needed: `b` plays no part in a recession condition, which is about
///   directions rather than points.
/// * `d_hint` — the solver's diverging iterate, losslessly in ℚ.
///
/// On success the returned `d` satisfies, **exactly over ℚ**: `Q d = 0`,
/// `A d ≥ 0`, and `c·d < 0`, normalized so its largest-magnitude entry is `±1`.
///
/// Normalization is free — all three conditions are homogeneous in `d` and a
/// positive scale factor preserves each — and it is what keeps the emitted
/// witness readable: a direction of `![0, 1]` rather than the 16-digit dyadic
/// fraction an iterate that ran to `1e9` actually is.
pub fn refine_recession(
    q: &[Vec<BigRational>],
    c: &[BigRational],
    a_rows: &[Vec<BigRational>],
    d_hint: &[BigRational],
) -> Result<Vec<BigRational>, RecessionError> {
    let n = c.len();
    if d_hint.len() != n {
        return Err(RecessionError::SelfCheck("hint length != n"));
    }
    if q.len() != n || q.iter().any(|row| row.len() != n) {
        return Err(RecessionError::SelfCheck("Q is not n×n"));
    }
    if a_rows.iter().any(|row| row.len() != n) {
        return Err(RecessionError::SelfCheck("ragged A"));
    }

    let q_is_zero = q.iter().flatten().all(BigRational::is_zero);
    let mut d: Vec<BigRational> = if q_is_zero {
        // ker Q is all of ℚⁿ, so the projection is the identity. Short-circuit
        // it: the general path would build the n standard basis vectors and
        // solve an n×n system to recover the input.
        d_hint.to_vec()
    } else {
        // An orthogonal projection onto ker Q, in the null-space basis `N`:
        // solve (NᵀN) z = N·d_hint, then d = Nᵀ z. `N`'s rows are independent
        // by construction, so the Gram matrix is invertible.
        let basis = nullspace_exact(q, n);
        if basis.is_empty() {
            return Err(RecessionError::NoFlatDirection);
        }
        let k = basis.len();
        let mut gram = vec![vec![BigRational::zero(); k]; k];
        let mut rhs = vec![BigRational::zero(); k];
        for i in 0..k {
            for j in 0..k {
                gram[i][j] = dot(&basis[i], &basis[j]);
            }
            rhs[i] = dot(&basis[i], d_hint);
        }
        let z = solve_exact(&gram, &rhs).ok_or(RecessionError::SelfCheck(
            "null-space Gram matrix is singular",
        ))?;
        let mut d = vec![BigRational::zero(); n];
        for (i, bi) in basis.iter().enumerate() {
            for j in 0..n {
                d[j] += &z[i] * &bi[j];
            }
        }
        d
    };

    // Scale so the largest entry is ±1. Also detects the degenerate projection.
    let scale = d
        .iter()
        .map(BigRational::abs)
        .max()
        .ok_or(RecessionError::SelfCheck("empty direction"))?;
    if scale.is_zero() {
        return Err(RecessionError::ZeroDirection);
    }
    for di in &mut d {
        *di /= &scale;
    }

    // --- exact conditions ---------------------------------------------------
    // `Q d = 0` holds by construction; rechecked because "by construction"
    // rests on the null-space routine being right, and the Lean proof does not
    // get to assume that.
    for row in q {
        if !dot(row, &d).is_zero() {
            return Err(RecessionError::SelfCheck("Q d = 0 failed its recheck"));
        }
    }
    for (i, row) in a_rows.iter().enumerate() {
        if dot(row, &d).is_negative() {
            return Err(RecessionError::LeavesFeasibleSet { constraint: i });
        }
    }
    if !dot(c, &d).is_negative() {
        return Err(RecessionError::NotDescending);
    }

    Ok(d)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> BigRational {
        BigRational::new(n.into(), d.into())
    }

    fn zeros(n: usize) -> Vec<Vec<BigRational>> {
        vec![vec![r(0, 1); n]; n]
    }

    /// The LP slice, unchanged in substance: with `Q = 0` every direction is
    /// flat, so the hint survives the projection and only the normalization
    /// touches it.
    #[test]
    fn an_lp_hint_passes_through_up_to_scale() {
        let d = refine_recession(
            &zeros(2),
            &[r(-1, 1), r(-1, 1)],
            &[vec![r(1, 1), r(1, 1)]],
            &[r(200, 1), r(400, 1)],
        )
        .unwrap();
        assert_eq!(d, vec![r(1, 2), r(1, 1)], "scaled by the largest entry");
    }

    /// The case the LP slice could not reach: `Q = diag(2, 0)` curves in `x₀`
    /// and is flat in `x₁`. The hint has a nonzero `x₀` component — the float
    /// iterate always does — and the projection removes it *exactly*, which is
    /// the whole difference between a certificate that verifies and one that
    /// does not.
    #[test]
    fn a_curved_component_is_projected_out_exactly() {
        let q = vec![vec![r(2, 1), r(0, 1)], vec![r(0, 1), r(0, 1)]];
        let d = refine_recession(
            &q,
            &[r(0, 1), r(-1, 1)],
            &[vec![r(1, 1), r(1, 1)]],
            &[r(3, 7), r(1000, 1)],
        )
        .unwrap();
        assert_eq!(d, vec![r(0, 1), r(1, 1)], "the x₀ component is gone");
        // And gone exactly: Q d is zero, not small.
        assert!(q.iter().all(|row| dot(row, &d).is_zero()));
    }

    /// A positive definite `Q` has no flat direction at all, so the objective
    /// is bounded below along every line. The refusal names that, rather than
    /// reporting a failed inequality further down.
    #[test]
    fn a_positive_definite_hessian_has_no_recession_direction() {
        let q = vec![vec![r(2, 1), r(0, 1)], vec![r(0, 1), r(2, 1)]];
        assert_eq!(
            refine_recession(&q, &[r(-1, 1), r(-1, 1)], &[], &[r(1, 1), r(1, 1)]),
            Err(RecessionError::NoFlatDirection)
        );
    }

    /// The hint lies entirely in the curved subspace: nothing to travel along.
    #[test]
    fn a_hint_orthogonal_to_the_flat_subspace_is_refused() {
        let q = vec![vec![r(2, 1), r(0, 1)], vec![r(0, 1), r(0, 1)]];
        assert_eq!(
            refine_recession(&q, &[r(0, 1), r(-1, 1)], &[], &[r(5, 1), r(0, 1)]),
            Err(RecessionError::ZeroDirection)
        );
    }

    /// A flat direction that leaves the feasible set is not a recession
    /// direction, and the refusal names the row that rules it out.
    #[test]
    fn a_direction_that_exits_the_feasible_set_is_refused() {
        assert_eq!(
            refine_recession(
                &zeros(2),
                &[r(-1, 1), r(-1, 1)],
                &[vec![r(1, 1), r(1, 1)], vec![r(1, 1), r(-1, 1)]],
                &[r(0, 1), r(1, 1)],
            ),
            Err(RecessionError::LeavesFeasibleSet { constraint: 1 })
        );
    }

    /// Feasible to travel along, but uphill — the objective does not run to
    /// −∞ this way.
    #[test]
    fn a_non_descending_direction_is_refused() {
        assert_eq!(
            refine_recession(
                &zeros(2),
                &[r(1, 1), r(1, 1)],
                &[vec![r(1, 1), r(1, 1)]],
                &[r(1, 1), r(1, 1)],
            ),
            Err(RecessionError::NotDescending)
        );
    }

    /// The contamination bound in the module doc, exercised: a hint of
    /// `x_finite + t·d` with `t = 10^6` and a nonzero kernel component in
    /// `x_finite` still yields a direction that clears a *strict* row, because
    /// the pollution is `O(1/t)` after scaling.
    #[test]
    fn finite_contamination_survives_a_strict_row() {
        let q = vec![vec![r(2, 1), r(0, 1)], vec![r(0, 1), r(0, 1)]];
        // x_finite = (1/3, 7), d_true = (0, 1), t = 10^6.
        let hint = [r(1, 3), r(7, 1) + r(1_000_000, 1)];
        let d =
            refine_recession(&q, &[r(0, 1), r(-1, 1)], &[vec![r(1, 1), r(1, 1)]], &hint).unwrap();
        assert_eq!(d, vec![r(0, 1), r(1, 1)]);
    }
}
