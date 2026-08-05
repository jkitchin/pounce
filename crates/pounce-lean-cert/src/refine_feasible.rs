//! Exact rational feasible point, projected from the solver's float iterate.
//!
//! A solve that converges reports an `x̃` feasible to ~1e-9. Its lossless
//! rational image is therefore *in*feasible, exactly and provably, so a
//! certificate about it proves nothing — the same wall the `global-min` path
//! hits, and it is climbed the same way: the float only proposes.
//!
//! Here what it proposes is the **active face**. Project `x̃` onto the affine
//! subspace where the equalities and the tight inequalities hold with equality,
//! exactly over ℚ, then check every original row at the result. The projection
//! is the minimum-norm correction
//!
//! ```text
//!   x̂ = x̃ − Rᵀ (R Rᵀ)⁻¹ (R x̃ − r)
//! ```
//!
//! for the selected rows `R x = r`. Minimum-norm is not an aesthetic choice: `ε`
//! in the emitted certificate is the distance from `x̃` to `x̂`, so a needlessly
//! distant feasible point weakens the very statement being made.
//!
//! What this cannot do is find a feasible point when there is none nearby — a
//! wrong active set, or an actually infeasible problem, produces a projection
//! that violates some row, and the answer is a refusal. That is the intended
//! failure mode: this module is a certificate producer, not a phase-1 solver.

use num_rational::BigRational;
use num_traits::Zero;

use crate::linalg::{dot, select_independent_rows, solve_exact};
use crate::refine::RefineError;

/// Project `x_float`'s rational image onto the active face and verify it.
///
/// * `a`, `b` — the inequality system `A x ≥ b`.
/// * `e`, `d` — the equality system `E x = d`.
/// * `active` — indices into `a` to pin as equalities. A hint: rows omitted
///   here are still checked (as `≥`) at the result, and rows wrongly included
///   simply over-constrain the projection into a refusal.
/// * `x0` — the float iterate's exact rational image.
///
/// Returns a point satisfying **every** row exactly over ℚ, or refuses.
pub fn refine_feasible(
    a: &[Vec<BigRational>],
    b: &[BigRational],
    e: &[Vec<BigRational>],
    d: &[BigRational],
    active: &[usize],
    x0: &[BigRational],
) -> Result<Vec<BigRational>, RefineError> {
    let n = x0.len();

    // Equalities first: they hold regardless, so when the guessed active set is
    // degenerate they are the rows worth keeping. Dependent rows are dropped
    // rather than refused — they are implied by the ones kept, and are verified
    // at the end like every other row.
    let mut rows: Vec<Vec<BigRational>> = Vec::with_capacity(e.len() + active.len());
    let mut rhs: Vec<BigRational> = Vec::with_capacity(e.len() + active.len());
    for (erow, dj) in e.iter().zip(d) {
        rows.push(erow.clone());
        rhs.push(dj.clone());
    }
    for &ci in active {
        rows.push(a[ci].clone());
        rhs.push(b[ci].clone());
    }
    let keep = select_independent_rows(&rows);

    let mut x: Vec<BigRational> = x0.to_vec();
    if !keep.is_empty() {
        // Normal equations for the minimum-norm correction: (R Rᵀ) y = R x₀ − r,
        // then x̂ = x₀ − Rᵀ y. `R` has independent rows, so `R Rᵀ` is invertible
        // and the `SingularUnexpected` arm is unreachable by construction — kept
        // because "unreachable" here rests on `select_independent_rows` being
        // right, which is exactly the kind of assumption worth not betting
        // soundness on.
        let k = keep.len();
        let mut gram = vec![vec![BigRational::zero(); k]; k];
        let mut resid = vec![BigRational::zero(); k];
        for (i, &ri) in keep.iter().enumerate() {
            for (j, &rj) in keep.iter().enumerate() {
                gram[i][j] = dot(&rows[ri], &rows[rj]);
            }
            resid[i] = dot(&rows[ri], x0) - &rhs[ri];
        }
        let y = solve_exact(&gram, &resid).ok_or(RefineError::SingularUnexpected)?;
        for (i, &ri) in keep.iter().enumerate() {
            for j in 0..n {
                x[j] -= &y[i] * &rows[ri][j];
            }
        }
    }

    // Every row, not only the projected ones: a dropped or unlisted row is
    // precisely where a wrong active-set guess shows up.
    for (i, row) in a.iter().enumerate() {
        if dot(row, &x) < b[i] {
            return Err(RefineError::InactiveViolated { constraint: i });
        }
    }
    for (j, erow) in e.iter().enumerate() {
        if dot(erow, &x) != d[j] {
            return Err(RefineError::EqualityResidual { constraint: j });
        }
    }
    Ok(x)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> BigRational {
        BigRational::new(n.into(), d.into())
    }

    /// The float lands just inside the halfspace `x₁ + x₂ ≥ 1`; projecting onto
    /// it lands exactly on the boundary.
    #[test]
    fn projects_onto_a_single_active_row() {
        let a = vec![vec![r(1, 1), r(1, 1)]];
        let b = vec![r(1, 1)];
        let x0 = vec![r(1, 2), r(1, 4)]; // sums to 3/4 — infeasible by 1/4
        let x = refine_feasible(&a, &b, &[], &[], &[0], &x0).unwrap();
        assert_eq!(dot(&a[0], &x), r(1, 1));
        // Minimum-norm: the correction is split evenly, 1/8 each.
        assert_eq!(x, vec![r(5, 8), r(3, 8)]);
    }

    /// Equalities are pinned whether or not they are listed as active.
    #[test]
    fn equalities_are_always_enforced() {
        let e = vec![vec![r(1, 1), r(0, 1)]];
        let d = vec![r(2, 1)];
        let x0 = vec![r(1, 1), r(7, 1)];
        let x = refine_feasible(&[], &[], &e, &d, &[], &x0).unwrap();
        assert_eq!(
            x,
            vec![r(2, 1), r(7, 1)],
            "only the pinned coordinate moves"
        );
    }

    /// A point already feasible is returned untouched, so `ε` collapses to 0
    /// rather than to a tiny-but-nonzero artifact of the projection.
    #[test]
    fn an_exactly_feasible_point_is_left_alone() {
        let a = vec![vec![r(1, 1), r(1, 1)]];
        let b = vec![r(1, 1)];
        let x0 = vec![r(3, 1), r(4, 1)];
        let x = refine_feasible(&a, &b, &[], &[], &[], &x0).unwrap();
        assert_eq!(x, x0);
    }

    /// Projecting onto one row can push the point out of another. The result is
    /// a refusal naming the row, not a certificate that cannot verify.
    #[test]
    fn a_projection_that_breaks_another_row_is_refused() {
        // x₁ ≥ 0 and −x₁ ≥ 0 force x₁ = 0; pinning only the first while the
        // float sits at x₁ = 1 leaves the second violated.
        let a = vec![vec![r(1, 1)], vec![r(-1, 1)]];
        let b = vec![r(1, 1), r(0, 1)];
        let x0 = vec![r(1, 1)];
        let err = refine_feasible(&a, &b, &[], &[], &[0], &x0).unwrap_err();
        assert_eq!(err, RefineError::InactiveViolated { constraint: 1 });
    }

    /// A degenerate active set (more rows than dimensions) is projected on the
    /// independent subset rather than refused — real LPs are routinely
    /// degenerate, and the dropped rows are still checked.
    #[test]
    fn a_degenerate_active_set_projects_on_its_independent_subset() {
        // Two copies of the same row, both listed active.
        let a = vec![vec![r(1, 1), r(1, 1)], vec![r(2, 1), r(2, 1)]];
        let b = vec![r(1, 1), r(2, 1)];
        let x0 = vec![r(1, 4), r(1, 4)];
        let x = refine_feasible(&a, &b, &[], &[], &[0, 1], &x0).unwrap();
        assert_eq!(x, vec![r(1, 2), r(1, 2)]);
    }
}
