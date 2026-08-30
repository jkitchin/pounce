//! Holding the parametric sensitivity step inside the variable bounds.
//!
//! Mirrors upstream
//! [`SensStdStepCalculator::BoundCheck`](https://github.com/coin-or/Ipopt/blob/master/contrib/sIPOPT/src/SensStdStepCalc.cpp),
//! which is what `sens_boundcheck` turns on.
//!
//! A step can point outside the box. Clipping the offending coordinate
//! back to its bound is cheap, but it leaves every other coordinate at
//! its linear-predictor value, so the result satisfies the bounds and
//! no longer satisfies the constraints. On upstream's own parametric
//! example that costs an order of magnitude against a full re-solve.
//!
//! [`refine_step_onto_bounds`] instead adds a row pinning the offending
//! coordinate at its bound and re-solves, so the others move with it.
//! [`worst_violation`] picks which coordinate that is and
//! [`expand_bounds`] puts the bounds in a form both can read.
//!
//! # Both halves, and why each matters
//!
//! Upstream's fix-relax is two cases (Pirnay, Lopez-Negrete and Biegler
//! 2012, section 2.5), and the name refers to both. Its equation 17
//! pins a variable the step carries past a bound, activating it. Its
//! equation 18 sets a bound multiplier to zero when the step drives it
//! negative, deactivating that bound so the variable can move.
//!
//! They fail differently. Without the pin, a crossing variable is
//! clamped and every other one keeps a value computed as though it had
//! not been. Without the release, a variable sitting on a bound stays
//! there however hard the perturbation pulls it off, because the linear
//! step preserves complementarity. Measured against sIPOPT on a model
//! whose bound wants to release, that second case is the difference
//! between returning 0.0 and 1.667.
//!
//! Both are solved the same way: add the row, re-solve the augmented
//! system through the Schur complement over the added rows, which is
//! what the paper's equations 19 through 22 describe.

use crate::schur_data::IndexSchurData;
use pounce_common::types::{Index, Number};
use pounce_linalg::Vector;
use pounce_linalg::expansion_matrix::ExpansionMatrix;
use std::rc::Rc;

/// Expand the compressed bound vectors into full var-x arrays, with
/// infinities where a variable has no bound on that side.
///
/// The compressed form pairs an [`ExpansionMatrix`] with a dense vector
/// holding only the bounded slots. Reading it repeatedly means holding
/// a borrow of the NLP, which a caller that also re-solves cannot do,
/// so this copies once.
pub fn expand_bounds(
    n_x: usize,
    px_l: &Rc<dyn pounce_linalg::Matrix>,
    px_u: &Rc<dyn pounce_linalg::Matrix>,
    x_l: &dyn Vector,
    x_u: &dyn Vector,
) -> (Vec<Number>, Vec<Number>) {
    let mut lo = vec![Number::NEG_INFINITY; n_x];
    let mut hi = vec![Number::INFINITY; n_x];
    for (pm, src, dst) in [(px_l, x_l, &mut lo), (px_u, x_u, &mut hi)] {
        let Some(em) = pm.as_any().downcast_ref::<ExpansionMatrix>() else {
            continue;
        };
        let vals = compressed_values(src);
        for (ci, &full_pos) in em.expanded_pos_indices().iter().enumerate() {
            let i = full_pos as usize;
            if let (true, Some(&v)) = (i < n_x, vals.get(ci)) {
                dst[i] = v;
            }
        }
    }
    (lo, hi)
}

/// Every coordinate whose predicted value leaves its bound, as
/// `(index, the bound it leaves, how far past it)`, worst first.
///
/// This is the half of the bound check that fix-relax keeps. The clamp
/// above answers "put it back", which loses the other coordinates; the
/// refinement needs "which ones, and where do they belong", and then
/// re-solves with those coordinates pinned so the rest respond.
///
/// Upstream's `BoundCheck` collects the whole list in one sweep and
/// its caller pins all of it before re-solving, which is what makes
/// the loop terminate on its own rather than on a pass budget: pinning
/// the single worst one per pass needs as many passes as there are
/// crossings, so on a model with more crossings than passes the budget
/// decides the answer (gh#732).
///
/// The list is ordered by overshoot rather than by index so the pins do
/// not depend on how the model was written, and ties keep index order.
/// `skip` names coordinates already pinned by an earlier pass, which
/// sit ON their bound and would otherwise be picked again.
pub fn bound_violations(
    x_curr: &[Number],
    dx: &[Number],
    lo: &[Number],
    hi: &[Number],
    eps: Number,
    skip: &[usize],
) -> Vec<(usize, Number, Number)> {
    let mut out: Vec<(usize, Number, Number)> = Vec::new();
    for i in 0..x_curr.len().min(dx.len()) {
        if skip.contains(&i) {
            continue;
        }
        let trial = x_curr[i] + dx[i];
        let (bound, over) = if trial < lo[i] {
            (lo[i], lo[i] - trial)
        } else if trial > hi[i] {
            (hi[i], trial - hi[i])
        } else {
            continue;
        };
        if over > eps {
            out.push((i, bound, over));
        }
    }
    // stable, so equal overshoots keep index order
    out.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// The coordinate whose predicted value leaves its bound by the most,
/// as `(index, the bound it leaves)`. The head of
/// [`bound_violations`].
pub fn worst_violation(
    x_curr: &[Number],
    dx: &[Number],
    lo: &[Number],
    hi: &[Number],
    eps: Number,
    skip: &[usize],
) -> Option<(usize, Number)> {
    bound_violations(x_curr, dx, lo, hi, eps, skip)
        .first()
        .map(|&(i, bound, _)| (i, bound))
}

/// Extract dense values from a `dyn Vector` that wraps a `DenseVector`.
/// Returns an empty vector when the downcast fails (and the bound
/// vector is just treated as having no entries — the boundcheck then
/// silently no-ops, matching upstream's behavior when bounds aren't
/// represented as DenseVectors).
fn compressed_values(v: &dyn Vector) -> Vec<Number> {
    use pounce_linalg::dense_vector::DenseVector;
    match v.as_any().downcast_ref::<DenseVector>() {
        // `expanded_values` (not `values`) so a homogeneous bound
        // vector — e.g. every lower bound 0 — materializes its scalar
        // instead of tripping `DenseVector::values`'s
        // `!homogeneous` debug_assert (L16).
        Some(dv) => dv.expanded_values(),
        None => Vec::new(),
    }
}

// Quieter index-typed signature helper for callers that pass usize-
// dimensioned slices but receive Index-counted bound dimensions.
#[doc(hidden)]
pub fn _index_to_usize(i: Index) -> usize {
    i as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_linalg::Vector;
    use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
    use pounce_linalg::expansion_matrix::{ExpansionMatrix, ExpansionMatrixSpace};

    fn make_dv(values: &[Number]) -> DenseVector {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut dv = DenseVector::new(space);
        dv.values_mut().copy_from_slice(values);
        dv
    }

    /// A homogeneous DenseVector of length `dim`, every entry `scalar`.
    /// Built via `Vector::set`, which puts the vector in homogeneous
    /// representation (no materialized storage) — the state under which
    /// `DenseVector::values()` debug_asserts.
    fn make_homogeneous_dv(dim: Index, scalar: Number) -> DenseVector {
        let space = DenseVectorSpace::new(dim);
        let mut dv = DenseVector::new(space);
        dv.set(scalar);
        assert!(dv.is_homogeneous());
        dv
    }

    /// `(px, compressed)` for a bound present on the given positions.
    fn expansion(n: Index, positions: &[Index]) -> Rc<dyn pounce_linalg::Matrix> {
        let space = ExpansionMatrixSpace::new(n, positions.len() as Index, positions, 0);
        Rc::new(ExpansionMatrix::new(space)) as Rc<dyn pounce_linalg::Matrix>
    }

    #[test]
    fn expand_bounds_puts_infinity_where_a_bound_is_absent() {
        // only x1 has a lower bound, only x2 an upper one
        let (lo, hi) = expand_bounds(
            3,
            &expansion(3, &[1]),
            &expansion(3, &[2]),
            &make_dv(&[-2.0]),
            &make_dv(&[7.0]),
        );
        assert_eq!(lo, vec![Number::NEG_INFINITY, -2.0, Number::NEG_INFINITY]);
        assert_eq!(hi, vec![Number::INFINITY, Number::INFINITY, 7.0]);
    }

    #[test]
    fn expand_bounds_materializes_a_homogeneous_vector() {
        // every lower bound 0, stored as a scalar rather than an array
        let (lo, _) = expand_bounds(
            2,
            &expansion(2, &[0, 1]),
            &expansion(2, &[]),
            &make_homogeneous_dv(2, 0.0),
            &make_dv(&[]),
        );
        assert_eq!(lo, vec![0.0, 0.0]);
    }

    #[test]
    fn worst_violation_takes_the_largest_overshoot_not_the_first() {
        let x = [0.5, 0.5, 0.5];
        let dx = [-0.6, -2.0, -0.7];
        let lo = [0.0, 0.0, 0.0];
        let hi = [10.0, 10.0, 10.0];
        // x1 is out by 1.5, x0 by 0.1, x2 by 0.2
        let (i, bound) = worst_violation(&x, &dx, &lo, &hi, 1e-9, &[]).unwrap();
        assert_eq!(i, 1);
        assert_eq!(bound, 0.0);
    }

    #[test]
    fn worst_violation_skips_what_is_already_pinned() {
        let x = [0.5, 0.5];
        let dx = [-0.6, -2.0];
        let lo = [0.0, 0.0];
        let hi = [10.0, 10.0];
        let (i, _) = worst_violation(&x, &dx, &lo, &hi, 1e-9, &[1]).unwrap();
        assert_eq!(i, 0, "the worst one is pinned, so the next is taken");
    }

    #[test]
    fn worst_violation_reports_an_upper_bound_too() {
        let x = [0.5];
        let dx = [3.0];
        let (i, bound) = worst_violation(&x, &dx, &[0.0], &[1.0], 1e-9, &[]).unwrap();
        assert_eq!((i, bound), (0, 1.0));
    }

    #[test]
    fn worst_violation_is_none_inside_the_bounds_and_within_eps() {
        let x = [0.5];
        assert!(worst_violation(&x, &[0.1], &[0.0], &[1.0], 1e-9, &[]).is_none());
        // just outside, but under the tolerance
        assert!(worst_violation(&x, &[0.5 + 1e-12], &[0.0], &[1.0], 1e-9, &[]).is_none());
    }

    #[test]
    fn bound_violations_returns_every_crossing_worst_first() {
        let x = [0.5, 0.5, 0.5, 0.5];
        let dx = [-0.6, -2.0, -0.7, 0.1];
        let lo = [0.0; 4];
        let hi = [10.0; 4];
        let v = bound_violations(&x, &dx, &lo, &hi, 1e-9, &[]);
        // x3 stays inside; the other three are out by 0.1, 1.5 and 0.2
        assert_eq!(
            v.iter().map(|&(i, _, _)| i).collect::<Vec<_>>(),
            vec![1, 2, 0],
            "the whole list, ordered by overshoot",
        );
        assert_eq!(v[0].1, 0.0, "and each carries the bound it left");
    }

    #[test]
    fn bound_violations_leaves_out_what_is_already_pinned() {
        let x = [0.5, 0.5];
        let dx = [-0.6, -2.0];
        let v = bound_violations(&x, &dx, &[0.0, 0.0], &[10.0, 10.0], 1e-9, &[1]);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].0, 0, "the pinned coordinate is not offered again");
    }

    /// `K` for a two-row system where holding row 0 drags row 1 by
    /// `lever`: solving `K y = e0` gives `y = (1, -lever)`.
    fn lever_backsolver(lever: Number) -> crate::backsolver::DenseLuBacksolver {
        crate::backsolver::DenseLuBacksolver::from_dense(2, &[1.0, 0.0, lever, 1.0])
            .expect("nonsingular")
    }

    #[test]
    fn a_refinement_that_ends_further_out_returns_the_unrefined_step() {
        // Row 0 is 0.1 below its bound, and the pin that repairs it
        // throws row 1 a thousand times further out than that. The pass
        // limit stops the loop before it can pin row 1 as well, so what
        // it has to return is worse than what it started from —
        // gh#732's "return the unrefined step" guard, which the pin's
        // own achievement check cannot see, since row 0 lands exactly
        // where it was asked to.
        let bs = lever_backsolver(1000.0);
        let dx_plain = [-0.1, 0.0];
        let (dx, rows, stop) = refine_step_onto_bounds(
            &bs,
            &dx_plain,
            &[0.0, 0.0],
            &[0.0, -1e-3],
            &[Number::INFINITY, 1e-3],
            &[],
            &[0.0, 0.0],
            1e-9,
            1e-9,
            1,
        )
        .expect("refinement");
        assert_eq!(stop, RefineStop::WorseThanPlain);
        assert!(rows.is_empty(), "nothing is reported as constrained");
        assert_eq!(dx, dx_plain.to_vec(), "the unrefined step comes back");
    }

    #[test]
    fn a_pass_whose_correction_is_out_of_scale_is_refused() {
        // The same shape with the lever at 1e10: the pin is achieved to
        // the last digit and the correction is 1e9 times the step it
        // corrects, which is a near-singular solve rather than a
        // repair. A check that reads only the pinned row accepts it.
        let bs = lever_backsolver(1e10);
        let dx_plain = [-0.1, 0.0];
        let (dx, rows, stop) = refine_step_onto_bounds(
            &bs,
            &dx_plain,
            &[0.0, 0.0],
            &[0.0, Number::NEG_INFINITY],
            &[Number::INFINITY, Number::INFINITY],
            &[],
            &[0.0, 0.0],
            1e-9,
            1e-9,
            8,
        )
        .expect("refinement");
        assert_eq!(stop, RefineStop::DegreesOfFreedom);
        assert!(
            rows.is_empty(),
            "the pass was refused, so nothing is pinned"
        );
        assert_eq!(dx, dx_plain.to_vec());
    }

    /// A backsolver whose release half is scripted: the plain solves go
    /// through a `DenseLuBacksolver`, `solve_released_step` answers
    /// from `steps` keyed by how many rows are released — a missing
    /// entry is a factorization that failed — and every call to it is
    /// counted.
    #[derive(Clone)]
    struct ScriptedRelease {
        base: crate::backsolver::DenseLuBacksolver,
        rows: Vec<crate::backsolver::BoundRow>,
        steps: std::collections::BTreeMap<usize, Vec<Number>>,
        calls: Rc<std::cell::Cell<usize>>,
    }

    impl crate::backsolver::SensBacksolver for ScriptedRelease {
        fn dim(&self) -> usize {
            self.base.dim()
        }
        fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
            self.base.solve(rhs, lhs)
        }
        fn bound_rows(&self) -> Option<&[crate::backsolver::BoundRow]> {
            Some(&self.rows)
        }
        fn supports_release(&self) -> bool {
            true
        }
        fn solve_released(&self, _released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
            self.base.solve(rhs, lhs)
        }
        fn solve_released_step(
            &self,
            released: &[usize],
            _rhs: &[Number],
            lhs: &mut [Number],
        ) -> bool {
            self.calls.set(self.calls.get() + 1);
            match self.steps.get(&released.len()) {
                Some(s) => {
                    lhs.copy_from_slice(s);
                    true
                }
                None => false,
            }
        }
    }

    /// `n × n` identity with `lever` at `(1, 0)`, so solving `K y = e0`
    /// gives `y = (1, -lever, 0, …)`: pinning row 0 drags row 1.
    fn lever_matrix(n: usize, lever: Number) -> Vec<Number> {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        a[n] = lever;
        a
    }

    #[test]
    fn a_release_batch_that_makes_the_step_worse_backs_off_to_one() {
        // Two multipliers are negative. Releasing both takes x0 five
        // below its lower bound; releasing the most negative one alone
        // settles. The batch has to earn its place the way the pin
        // batch does — without that, the CSTR of notebook 36 released
        // 56 bounds where 41 were right and the step came back worse
        // than not refining at all (gh#734 review).
        let calls = Rc::new(std::cell::Cell::new(0));
        let bs = ScriptedRelease {
            base: crate::backsolver::DenseLuBacksolver::from_dense(4, &lever_matrix(4, 0.0))
                .expect("nonsingular"),
            rows: vec![
                crate::backsolver::BoundRow {
                    row: 2,
                    var_row: 0,
                    lower: true,
                },
                crate::backsolver::BoundRow {
                    row: 3,
                    var_row: 1,
                    lower: true,
                },
            ],
            steps: [
                (2usize, vec![-5.0, 0.0, 0.0, 0.0]),
                (1usize, vec![0.0, 0.0, 0.0, 0.0]),
            ]
            .into_iter()
            .collect(),
            calls: Rc::clone(&calls),
        };
        let mults = [
            BoundMultiplier { row: 2, base: 1.0 },
            BoundMultiplier { row: 3, base: 1.0 },
        ];
        // z2 = 1 - 2 = -1 and z3 = 1 - 1.5 = -0.5, so both want out
        let (dx, rows, stop) = refine_step_onto_bounds(
            &bs,
            &[0.0, 0.0, -2.0, -1.5],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[Number::INFINITY, Number::INFINITY],
            &mults,
            &[0.0; 4],
            1e-9,
            1e-9,
            8,
        )
        .expect("refinement");
        assert_eq!(rows, vec![2], "the most negative one, alone");
        assert_eq!(stop, RefineStop::Settled);
        assert_eq!(dx, vec![0.0, 0.0, -1.0, 0.0], "and its multiplier is zero");
    }

    /// The primal margin and the release threshold are two numbers.
    /// A caller who says ten is on the bound has said nothing about
    /// whether a multiplier at minus one has changed sign, and with one
    /// number a wide `bound_eps` would stop every release on the model.
    #[test]
    fn a_wide_primal_margin_does_not_stop_a_release() {
        let make = || ScriptedRelease {
            base: crate::backsolver::DenseLuBacksolver::from_dense(4, &lever_matrix(4, 0.0))
                .expect("nonsingular"),
            rows: vec![
                crate::backsolver::BoundRow {
                    row: 2,
                    var_row: 0,
                    lower: true,
                },
                crate::backsolver::BoundRow {
                    row: 3,
                    var_row: 1,
                    lower: true,
                },
            ],
            steps: [
                (2usize, vec![-5.0, 0.0, 0.0, 0.0]),
                (1usize, vec![0.0, 0.0, 0.0, 0.0]),
            ]
            .into_iter()
            .collect(),
            calls: Rc::new(std::cell::Cell::new(0)),
        };
        let mults = [
            BoundMultiplier { row: 2, base: 1.0 },
            BoundMultiplier { row: 3, base: 1.0 },
        ];
        // z2 = 1 - 2 = -1 and z3 = -0.5 both want out. A primal margin
        // of ten is wider than anything here, and both releases still
        // happen. Both, rather than the one the sibling test above
        // settles on: the accept guard compares overshoot against the
        // primal margin, and five below a bound is inside ten, so the
        // batch stands. That is the guard reading the caller's margin,
        // which is the one number a caller who widens it has changed.
        let (_, rows, stop) = refine_step_onto_bounds(
            &make(),
            &[0.0, 0.0, -2.0, -1.5],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[Number::INFINITY, Number::INFINITY],
            &mults,
            &[0.0; 4],
            10.0,
            1e-9,
            8,
        )
        .expect("refinement");
        assert_eq!(rows, vec![2, 3], "the release reads its own threshold");
        assert_eq!(stop, RefineStop::Settled);
        // And the other way: a release threshold of ten is the one
        // thing that stops it, with the primal margin back at its floor.
        let (_, rows, _) = refine_step_onto_bounds(
            &make(),
            &[0.0, 0.0, -2.0, -1.5],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[Number::INFINITY, Number::INFINITY],
            &mults,
            &[0.0; 4],
            1e-9,
            10.0,
            8,
        )
        .expect("refinement");
        assert!(rows.is_empty(), "nothing is negative past ten");
    }

    #[test]
    fn a_release_the_factorization_refuses_is_not_asked_for_twice() {
        // The one negative multiplier cannot be released at all. The
        // pins carry on without it, and the loop neither asks for that
        // factorization again on every later pass nor reports the pass
        // limit for something no budget reaches.
        let calls = Rc::new(std::cell::Cell::new(0));
        let bs = ScriptedRelease {
            base: crate::backsolver::DenseLuBacksolver::from_dense(4, &lever_matrix(4, 1.0))
                .expect("nonsingular"),
            rows: vec![crate::backsolver::BoundRow {
                row: 2,
                var_row: 0,
                lower: true,
            }],
            // no entry for any size: every released factorization fails
            steps: std::collections::BTreeMap::new(),
            calls: Rc::clone(&calls),
        };
        let mults = [BoundMultiplier { row: 2, base: 1.0 }];
        // x0 is 1.0 below its bound and z2 = 1 - 2 = -1 wants out.
        // Pinning x0 drags x1 to -1, under ITS bound of -0.5, so a
        // second pass follows and would ask for the release again.
        let (_dx, rows, stop) = refine_step_onto_bounds(
            &bs,
            &[-1.0, 0.0, -2.0, 0.0],
            &[0.0, 0.0],
            &[0.0, -0.5],
            &[Number::INFINITY, Number::INFINITY],
            &mults,
            &[0.0; 4],
            1e-9,
            1e-9,
            8,
        )
        .expect("refinement");
        assert_eq!(calls.get(), 1, "asked for once, then barred");
        assert_eq!(
            stop,
            RefineStop::DegreesOfFreedom,
            "a bound that cannot leave the active set is not the pass limit",
        );
        assert_eq!(rows, vec![0, 1], "and the pins it could place still stand");
    }
}

/// A bound multiplier the step can drive negative: where it sits in the
/// compound KKT vector, and its value at the base point.
///
/// A negative multiplier means the bound should no longer be active,
/// which is the second half of upstream's fix-relax (its equation 18).
pub struct BoundMultiplier {
    /// Row of the compound KKT vector holding this multiplier.
    pub row: usize,
    /// Its value at the converged point, read raw off `curr.z_l` /
    /// `curr.z_u`, so in the coordinates the solve ran in rather than
    /// the model's own. [`refine_step_onto_bounds`] converts it with
    /// the backsolver's own `F`, so every caller hands over the same
    /// raw value and none of them needs to know the convention.
    pub base: Number,
}

/// Why [`refine_step_onto_bounds`] stopped.
///
/// Only [`RefineStop::Settled`] says the refinement finished: the
/// violation list emptied, which is the loop's own termination
/// condition. Every other value says the step returned is the last one
/// a pass could achieve, and names what stopped it, so a caller can
/// tell a limit it may raise from one it cannot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefineStop {
    /// Nothing is outside a bound and no bound multiplier is negative.
    Settled,
    /// `max_iter` passes were spent with the list still not empty. A
    /// safety limit rather than a budget: a pass now takes every
    /// violation it can see, so reaching this means the conditions kept
    /// moving and the answer is whatever the last pass reached.
    IterationLimit,
    /// A pass could not be solved or could not be achieved: the
    /// conditions have exhausted the problem's degrees of freedom, and
    /// no step holds them all. No budget helps.
    DegreesOfFreedom,
    /// The refinement ended further outside the bounds than the step it
    /// started from, so the unrefined step was returned and no rows are
    /// reported as constrained.
    WorseThanPlain,
}

impl RefineStop {
    /// A stable short name, for a caller reporting this across a
    /// language boundary.
    pub fn as_str(self) -> &'static str {
        match self {
            RefineStop::Settled => "settled",
            RefineStop::IterationLimit => "iteration_limit",
            RefineStop::DegreesOfFreedom => "degrees_of_freedom",
            RefineStop::WorseThanPlain => "worse_than_plain",
        }
    }
}

/// How far a pass's correction may exceed the step it corrects before
/// the pass is refused. The singular case a dense LU does not report
/// comes back around `1e15`, so this only has to sit above the
/// leverage a real pin can have — moving one coordinate onto its bound
/// can legitimately move another by orders of magnitude more.
const CORRECTION_SCALE_LIMIT: Number = 1e4;

/// How much further outside the bounds the refinement may end than the
/// step it started from before the unrefined step is returned instead.
/// A refinement that leaves a coordinate this much further out has not
/// repaired an active set, whatever it achieved on the rows it pinned.
const WORSE_THAN_PLAIN_FACTOR: Number = 10.0;

/// The refinement's release threshold: how far negative the step has to
/// drive a bound multiplier before its bound is released, from the
/// solve's own `bound_relax_factor`.
///
/// Never a caller's `bound_eps`. That is a primal margin, and a
/// multiplier changing sign is not a primal event — reading one number
/// for both is what let a `bound_eps` of `1e-2` stop every release on a
/// model whose multipliers are of order `1e-3`.
///
/// The floor is `1e-9`, which is also what an unset or unreadable
/// `bound_relax_factor` resolves to, since that is the floor by
/// definition. Three callers reach this: [`crate::Solver::bound_context`]
/// off the recorded state, and the CLI and [`crate::SensSolve`] off the
/// options list through [`crate::options::release_floor_from_options`].
/// One derivation, so they cannot drift on what the solve's own margin
/// is.
pub fn release_floor(bound_relax_factor: Number) -> Number {
    bound_relax_factor.abs().max(1e-9)
}

/// Repair the active set the step implies, by pinning and releasing.
///
/// Returns the refined step, the compound rows it constrained, and why
/// it stopped. This is upstream's fix-relax, both cases:
///
/// * a variable the step carries past a bound is pinned AT that bound,
///   which activates it (their equation 17);
/// * a bound multiplier the step drives negative is set to zero, which
///   deactivates that bound and lets the variable move (equation 18).
///
/// Without the second, a variable sitting on a bound at the base point
/// stays there however hard the perturbation pulls it off, because the
/// step holds complementarity. Measured on a model whose bound wants to
/// release, that is the difference between 0.0 and 1.667.
///
/// # One list per pass
///
/// A pass takes EVERY violation it can see — every coordinate outside
/// a bound and every multiplier driven negative — and constrains all of
/// them before re-solving, which is upstream's `BoundCheck` filling one
/// `x_bound_violations_idx` and its caller's `while (bounds_violated)`
/// re-solving over the lot. The loop then ends on its own, when the
/// list comes back empty.
///
/// Taking only the worst one per pass, which this did until gh#732,
/// needs as many passes as there are crossings. On a model with more
/// crossings than passes `max_iter` stopped the loop rather than the
/// violations doing it, so the budget picked the answer: on the CSTR of
/// notebook 36 the pin count equalled the budget at every budget tried,
/// and at 100 pins — half that problem's degrees of freedom — the
/// refined step came back 8.6 times worse than the unrefined one.
/// `max_iter` is a safety limit now, and a stop of
/// [`RefineStop::IterationLimit`] is what says it fired.
///
/// Each pass adds its conditions and re-solves the augmented system
/// carrying all of them, against the original factorization, so its
/// correction is measured from the base step rather than the previous
/// pass. Adding successive corrections counts the earlier ones twice.
/// The Schur complement over those rows is what upstream's equations 19
/// through 22 describe. The factorization is never rebuilt for a pin,
/// which is what makes this cheaper than a re-solve; the Schur
/// complement is rebuilt from scratch each pass, so a pass carrying `k`
/// conditions costs one dense `k × k` solve and `k + 1` back-solves.
/// Collecting the list makes `k` the number of crossings rather than
/// the pass index, so the same repair costs passes instead of pins.
///
/// A release is not a Schur row here, unlike upstream, which puts the
/// multiplier's row in the same list as the primal violations. It
/// re-factors with that bound's `sigma` dropped, because an active
/// bound's `sigma = z / s` grows as the solve converges and destroys
/// the released system's information in the converged factor: computing
/// a release from the held factor gets *worse* the better the solve
/// converged, 2e-4 off at `tol = 1e-10` against 7e-9 at `1e-6`. What
/// gh#732 fixes about a release is that the pins now survive it: their
/// right-hand sides are re-measured against the re-solved base instead
/// of the pin set being cleared, which is where that issue's budget
/// table got its discontinuity. A pin batch that cannot be solved
/// leaves the releases of its own pass standing for the same reason —
/// a release repairs the active set on its own terms.
///
/// The release batch backs off the way the pin batch does, and for a
/// sharper reason. A pin adds a condition, so an over-large batch shows
/// up as an augmented system that cannot be solved. A release REMOVES
/// one: every bound taken out is stiffness that is no longer holding
/// its variable, and a batch that takes too many carries variables off
/// bounds they were sitting on, with nothing left to pin them back.
/// That has no failed solve to report it — on notebook 36's CSTR it was
/// 56 releases where 41 were right, and the step came back worse than
/// not refining at all. So a batch of more than one is kept only when
/// the step it produces is no further outside the bounds than the one
/// in hand.
///
/// `multipliers` carry their base values in the solve's own
/// coordinates. They are converted here, once, with the backsolver's
/// [`SensBacksolver::natural_units_factor`], so they agree with the `z`
/// rows of `dx_plain` before either is used.
///
/// # Two margins
///
/// `eps` is the primal margin: how far outside a bound a coordinate has
/// to end to count as having left it, which decides what a pass pins
/// and what the two guards below compare overshoot against.
/// `release_eps` is the dual one: how far negative the step has to
/// drive a bound multiplier before the bound is released. They are two
/// numbers because a caller who widens the primal margin is saying
/// what counts as on the bound, and that says nothing about whether a
/// multiplier at `-5e-3` has changed sign. With one number, a
/// `bound_eps` of `1e-2` would stop every release on a model whose
/// multipliers are of order `1e-3`, and return the wrong active set
/// without saying so.
///
/// The two guards below stay on `eps`, since they compare primal
/// overshoot and a caller who widened the primal margin has said that
/// about overshoot too. The consequence is worth knowing before you
/// widen it: both guards scale with `eps`, so a margin far above the
/// model's own scale takes them out of the picture — at `eps = 10.0`
/// the second reads `worst_over(dx) > 100.0`, and
/// [`RefineStop::WorseThanPlain`] cannot be reached. A margin wide
/// enough to pin nothing is also wide enough to accept any release
/// batch it produces.
///
/// # Two guards, independent of the loop
///
/// A pass is refused when its correction is out of scale with the step
/// it corrects, not only when a pinned row misses its target. Checking
/// the pinned rows alone is what let gh#732's 100 pins each land within
/// `1e-3` of where they were asked to go while the step as a whole came
/// back unusable: hitting the pinned coordinates says nothing about
/// what the correction did to the other 1300.
///
/// And the unrefined step is returned when the refinement ends further
/// outside the bounds than it started, which costs nothing since
/// `dx_plain` is already in scope. Repairing an active set that leaves
/// the box further out than not repairing it at all has failed on its
/// own terms.
pub fn refine_step_onto_bounds<B>(
    backsolver: &B,
    dx_plain: &[Number],
    x_curr: &[Number],
    lo: &[Number],
    hi: &[Number],
    multipliers: &[BoundMultiplier],
    rhs_plain: &[Number],
    eps: Number,
    release_eps: Number,
    max_iter: usize,
) -> Result<(Vec<Number>, Vec<usize>, RefineStop), String>
where
    B: crate::backsolver::SensBacksolver + Clone,
{
    use crate::sens_app::{SensApplication, SensOptions};

    let n_full = dx_plain.len();
    let mut dx = dx_plain.to_vec();
    // Into the units the step is in, before either is read. `F` is
    // indexed by compound row, the same space `BoundMultiplier::row`
    // lives in.
    let multipliers: Vec<BoundMultiplier> = match backsolver.natural_units_factor() {
        None => multipliers
            .iter()
            .map(|m| BoundMultiplier {
                row: m.row,
                base: m.base,
            })
            .collect(),
        Some(f) => multipliers
            .iter()
            .map(|m| BoundMultiplier {
                row: m.row,
                base: m.base * f[m.row],
            })
            .collect(),
    };
    let multipliers = &multipliers[..];
    let bound_rows = backsolver.bound_rows();
    let can_release = backsolver.supports_release() && rhs_plain.len() == n_full;

    // How far outside its bounds the worst coordinate of a step sits.
    let worst_over = |d: &[Number]| {
        bound_violations(x_curr, d, lo, hi, eps, &[])
            .first()
            .map_or(0.0, |&(_, _, over)| over)
    };

    // Which multiplier rows the step drives negative, most negative
    // first, ignoring any already out of the active set and any the
    // factorization has already refused to release.
    let releasable = |dx: &[Number], released: &[usize], refused: &[usize]| -> Vec<usize> {
        if !can_release {
            return Vec::new();
        }
        let mut v: Vec<(usize, Number)> = multipliers
            .iter()
            .filter(|m| !released.contains(&m.row) && !refused.contains(&m.row))
            .filter(|m| bound_rows.is_some_and(|br| br.iter().any(|b| b.row == m.row)))
            .map(|m| (m.row, m.base + dx[m.row]))
            .filter(|&(_, v)| v < -release_eps)
            .collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        v.into_iter().map(|(r, _)| r).collect()
    };

    // The step under the given conditions, or `None` when the augmented
    // system cannot deliver it. `Err` is reserved for a malformed
    // condition set, which is a caller's bug rather than a refusal.
    let solve_pins = |pins: &[(usize, Number)],
                      released: &[usize],
                      dx_base: &[Number]|
     -> Result<Option<Vec<Number>>, String> {
        if pins.is_empty() {
            return Ok(Some(dx_base.to_vec()));
        }
        let rows: Vec<Index> = pins.iter().map(|&(r, _)| r as Index).collect();
        // Measured from the base step, which moves whenever the
        // released set does, so a pin outlives a release.
        let rhs: Vec<Number> = pins
            .iter()
            .map(|&(r, bound)| (x_curr[r] + dx_base[r]) - bound)
            .collect();
        let signs = vec![1; rows.len()];
        let mk = |r: Vec<Index>| {
            IndexSchurData::from_parts(r, signs.clone()).map_err(|e| format!("{e:?}"))
        };
        let opts = SensOptions {
            run_sens: true,
            ..SensOptions::default()
        };
        // Against the released operator, not the converged one: once a
        // bound is out of the active set, every later condition has to
        // be solved in the system that reflects that.
        let view = ReleasedView {
            base: backsolver.clone(),
            rows: released.to_vec(),
        };
        let mut pin_app = SensApplication::new(mk(rows.clone())?, view, opts);
        let mut du = vec![0.0; rows.len()];
        let mut corr = vec![0.0; n_full];
        if !pin_app.run_sens_step(&mk(rows)?, &rhs, &mut du, &mut corr) {
            // An exactly singular augmented system, where the two
            // guards below catch the near-singular case.
            return Ok(None);
        }

        // A healthy pass lands its conditions within a few parts per
        // million, so this is not an accuracy check: it is for the
        // singular case, where a dense LU returns a solution around
        // 1e15 rather than reporting it.
        let achieved = pins
            .iter()
            .zip(rhs.iter())
            .all(|(&(r, _), &want)| (corr[r] + want).abs() <= 1e-3 * want.abs().max(1.0));
        if !achieved {
            return Ok(None);
        }
        // Achieving every pinned row says nothing about what the
        // correction did to the rest of the vector (gh#732), so the
        // correction's own size is checked too.
        let inf = |v: &[Number]| v.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        let scale = inf(dx_base).max(inf(&rhs)).max(1.0);
        if inf(&corr) > CORRECTION_SCALE_LIMIT * scale {
            return Ok(None);
        }
        Ok(Some(
            dx_base
                .iter()
                .zip(corr.iter())
                .map(|(b, c)| b + c)
                .collect(),
        ))
    };

    // The base step with `set` added to the released bounds, or `None`
    // when the released system cannot be factored.
    let apply_releases = |released: &[usize], set: &[usize]| -> Option<(Vec<usize>, Vec<Number>)> {
        let mut trial = released.to_vec();
        trial.extend_from_slice(set);
        let mut base = vec![0.0; n_full];
        if !backsolver.solve_released_step(&trial, rhs_plain, &mut base) {
            return None;
        }
        // A released bound's multiplier is zero by construction; its
        // own row of the re-solved step is a by-product of the
        // complementarity row the factor still carries.
        for &r in &trial {
            if let Some(m) = multipliers.iter().find(|m| m.row == r) {
                base[r] = -m.base;
            }
        }
        Some((trial, base))
    };

    // (var-x row, the bound it is held at). The right-hand side is
    // re-derived from the base step each pass rather than stored, so a
    // release moves the pins with it instead of clearing them.
    let mut pins: Vec<(usize, Number)> = Vec::new();
    // Multiplier rows taken out of the active set. Unlike a pin these
    // never become a Schur condition: they change the operator, so the
    // step is re-solved against a factorization that does not carry
    // their `sigma` at all.
    let mut released: Vec<usize> = Vec::new();
    // The step corrections are measured from. It moves whenever the
    // released set does, since that is a different system.
    let mut dx_base = dx_plain.to_vec();
    // Bounds whose release the factorization would not deliver. Barred
    // rather than retried: the same factorization would be asked for
    // again every pass until the limit, and the limit is not what
    // stopped it.
    let mut refused_releases: Vec<usize> = Vec::new();
    let mut stop = RefineStop::IterationLimit;

    for _ in 0..max_iter {
        let taken: Vec<usize> = pins.iter().map(|&(r, _)| r).collect();
        let fresh_pins = bound_violations(x_curr, &dx, lo, hi, eps, &taken);
        let fresh_releases = releasable(&dx, &released, &refused_releases);
        if fresh_pins.is_empty() && fresh_releases.is_empty() {
            // A bound whose release was refused is still one the step
            // wants out of the active set. The loop has nothing left to
            // try for it, which is not the same as having settled.
            stop = if releasable(&dx, &released, &[]).is_empty() {
                RefineStop::Settled
            } else {
                RefineStop::DegreesOfFreedom
            };
            break;
        }

        if !fresh_releases.is_empty() {
            // A release is not a condition on the step, it is a
            // different system: re-solve with those bounds' `sigma`
            // gone, and measure the pins from the step that produces.
            //
            // The batch backs off the way the pin batch below does.
            // Taking every negative multiplier at once can release more
            // stiffness than the step wanted: on notebook 36's CSTR, 56
            // releases where 41 were right carried five `v1` intervals
            // off the bound they had been sitting on, with no degrees
            // of freedom left to pin them back (gh#734 review). So the
            // batch is kept only when the step it produces is no
            // further outside the bounds than the one in hand, and
            // otherwise the most negative multiplier goes alone and the
            // next pass re-measures the rest under it.
            let before = worst_over(&dx);
            let mut sets: Vec<&[usize]> = vec![&fresh_releases[..]];
            if fresh_releases.len() > 1 {
                sets.push(&fresh_releases[..1]);
            }
            let mut taken: Option<(Vec<usize>, Vec<Number>, Vec<Number>)> = None;
            for (k, set) in sets.iter().enumerate() {
                let Some((trial, base)) = apply_releases(&released, set) else {
                    continue;
                };
                let Some(step) = solve_pins(&pins, &trial, &base)? else {
                    continue;
                };
                // A single release is the smallest step the loop can
                // take toward a bound that has to leave the active set,
                // so it is taken whether or not it helps: refusing it
                // would leave a negative multiplier with nothing left
                // to do about it. The guard at the end still has the
                // last word on what comes back.
                let alone = k + 1 == sets.len();
                if alone || worst_over(&step) <= before.max(eps) {
                    taken = Some((trial, base, step));
                    break;
                }
            }
            match taken {
                Some((trial, base, step)) => {
                    released = trial;
                    dx_base = base;
                    dx = step;
                }
                None => {
                    // The released system could not be factored, or
                    // could not carry the pins already placed.
                    refused_releases.extend_from_slice(&fresh_releases);
                    if fresh_pins.is_empty() {
                        stop = RefineStop::DegreesOfFreedom;
                        break;
                    }
                }
            }
            if fresh_pins.is_empty() {
                // The release phase already produced this pass's step.
                continue;
            }
        }

        // What the pin batch can undo, snapshotted BELOW the release
        // phase so a release is not among it. Both halves of that
        // matter and they are separable (gh#734 review bisected them):
        // keeping `released` is what leaves the bounds out of the
        // active set at all, and snapshotting `dx` here rather than
        // above is what leaves the STEP the release produced. Roll back
        // only the first and the rows come back while the answer stays
        // the plain step's; roll back both and a sound release is
        // discarded because the pins that came with it did not fit. A
        // release repairs the active set on its own terms.
        let keep_pins = pins.clone();
        let keep_dx = dx.clone();
        pins.extend(fresh_pins.iter().map(|&(i, bound, _)| (i, bound)));
        let mut next = solve_pins(&pins, &released, &dx_base)?;
        if next.is_none() && fresh_pins.len() > 1 {
            // The batch asked for more than the remaining degrees of
            // freedom hold. Keep the worst of the new crossings and let
            // the next pass re-measure the rest under it, which is what
            // the one-at-a-time loop would have done.
            pins.truncate(keep_pins.len());
            pins.push((fresh_pins[0].0, fresh_pins[0].1));
            next = solve_pins(&pins, &released, &dx_base)?;
        }
        match next {
            Some(step) => dx = step,
            None => {
                pins = keep_pins;
                dx = keep_dx;
                stop = RefineStop::DegreesOfFreedom;
                break;
            }
        }
    }

    // The loop can also run out of passes on the one that settled it,
    // which is not the limit firing. And what is left can be a bound
    // the factorization refused to release, which no budget reaches.
    if stop == RefineStop::IterationLimit {
        let taken: Vec<usize> = pins.iter().map(|&(r, _)| r).collect();
        let pins_left = !bound_violations(x_curr, &dx, lo, hi, eps, &taken).is_empty();
        let rel_left = releasable(&dx, &released, &[]);
        if !pins_left && rel_left.is_empty() {
            stop = RefineStop::Settled;
        } else if !pins_left && rel_left.iter().all(|r| refused_releases.contains(r)) {
            stop = RefineStop::DegreesOfFreedom;
        }
    }

    // Whatever stopped it, a refinement that ends further outside the
    // bounds than the step it started from has failed on its own terms.
    let plain_worst = worst_over(dx_plain);
    if worst_over(&dx) > WORSE_THAN_PLAIN_FACTOR * plain_worst.max(eps) {
        return Ok((dx_plain.to_vec(), Vec::new(), RefineStop::WorseThanPlain));
    }

    let mut out = released.clone();
    out.extend(pins.into_iter().map(|(r, _)| r));
    Ok((dx, out, stop))
}

/// The converged backsolver with a set of bounds out of the active set,
/// so the pin machinery can run against the released system without
/// knowing that is what it is doing.
#[derive(Clone)]
struct ReleasedView<B: crate::backsolver::SensBacksolver + Clone> {
    base: B,
    rows: Vec<usize>,
}

impl<B: crate::backsolver::SensBacksolver + Clone> crate::backsolver::SensBacksolver
    for ReleasedView<B>
{
    fn dim(&self) -> usize {
        self.base.dim()
    }
    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        // Nothing released is the converged system, so ask for it
        // directly: routing an empty set through `solve_released` asks
        // a backsolver that cannot release for something it does not
        // need to do, and the ones that can already short-circuit it.
        if self.rows.is_empty() {
            return self.base.solve(rhs, lhs);
        }
        self.base.solve_released(&self.rows, rhs, lhs)
    }
    fn natural_units_factor(&self) -> Option<&[Number]> {
        self.base.natural_units_factor()
    }
    fn bound_rows(&self) -> Option<&[crate::backsolver::BoundRow]> {
        self.base.bound_rows()
    }
    fn supports_release(&self) -> bool {
        self.base.supports_release()
    }
    fn solve_released(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.base.solve_released(released, rhs, lhs)
    }
    fn solve_released_step(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.base.solve_released_step(released, rhs, lhs)
    }
}

/// A bound this far out is the reader's absent-bound sentinel rather
/// than a bound, and a step cannot cross it.
const NO_BOUND_LO: Number = -1e19;
/// Mirror of [`NO_BOUND_LO`].
const NO_BOUND_HI: Number = 1e19;
/// A segment shorter than this has not advanced the path, so the
/// rows changed at its start stay barred from changing back.
const PATH_MIN_SEGMENT: Number = 1e-12;

/// One breakpoint the path stopped at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathSegment {
    /// Fraction of the perturbation applied when this segment ended,
    /// measured from the base point.
    pub at: Number,
    /// Var-x row of the variable whose bound status changed, whatever
    /// the kind of change. A release is detected on the bound's
    /// multiplier row, but it is recorded here by the variable it
    /// frees, so a caller never needs the multiplier layout to read
    /// the record.
    pub var_row: usize,
    /// `true` when the bound involved is the variable's lower bound.
    pub lower: bool,
    /// `true` when the variable reached the bound and is held there
    /// from this fraction on, `false` when it left it: either a bound
    /// active at the base whose multiplier reached zero, or a hold
    /// this path added earlier whose multiplier crossed zero.
    ///
    /// A weakly active bound can be recorded `true` at a fraction of
    /// essentially zero, and that does not contradict the variable
    /// having been on it at the base point: what the working set
    /// gained there is the HOLD. Undecided, the bound sat in the
    /// factorization as an order-one penalty that does not enforce it
    /// (gh#852).
    pub pinned: bool,
}

/// A variable the path holds at a bound it reached, with the
/// accumulated multiplier on its Schur row. The multiplier starts at
/// zero where the hold is added, exactly the crossing, takes a sign on
/// the segment after, and the hold drops where it crosses zero again,
/// which is the "drop" half of add-and-drop.
#[derive(Clone, Copy, Debug)]
struct PathHold {
    /// Var-x row held.
    row: usize,
    /// `true` when the bound held is the variable's lower bound. Only
    /// the record reads this: the drop test does not care which side
    /// the hold is on.
    lower: bool,
    /// Accumulated Schur-row multiplier, in whatever sign convention
    /// the augmented system uses: the drop test only asks when it
    /// crosses zero, so the convention never needs to be named.
    mult: Number,
}

/// Apply the perturbation a little at a time, stopping wherever the
/// active set changes.
///
/// [`refine_step_onto_bounds`] decides every condition at the base
/// point. This advances instead: it takes the fraction of the
/// perturbation that reaches the first breakpoint, applies that one
/// change, and continues from there with the remainder under the new
/// active set. The result is piecewise linear in the parameter, which
/// is the exact solution for a QP, whose solution is piecewise affine
/// in the parameter. For an NLP it stays a predictor, because nothing
/// is re-linearized between breakpoints.
///
/// Three kinds of breakpoint end a segment, all ratio tests on
/// quantities the step already carries. A variable strictly inside its
/// bounds reaches one, and is held there. A bound active at the base
/// has its multiplier reach zero, and the variable leaves it. A hold
/// this path added earlier has its multiplier cross zero, and the
/// variable leaves that bound too: the direction changes at every
/// breakpoint, so a bound reached under one direction may stop binding
/// under a later one.
///
/// Releasing a base-active bound needs no right-hand-side shift,
/// unlike the base-point refinement. The path stops exactly where the
/// multiplier reaches zero, so there is nothing left to drive to zero.
/// Dropping a hold needs no re-factorization at all, since the hold is
/// a Schur row rather than a term in the held factor.
///
/// `weak_rows` names the bound-multiplier rows the activity
/// classifier could not certify as strongly active. Those rows sit in
/// the factorization with an order-one sigma that bends the direction
/// without enforcing the bound, so the walk is allowed to reach one
/// and hold it, releasing the row as it does. Every other base-active
/// bound stays unreachable: its sigma is order `1/mu`, its variable
/// cannot move off the bound, and a Schur hold there would enforce
/// the same bound twice through a near-singular complement.
///
/// Returns the accumulated step and the breakpoints crossed. When
/// `max_iter` segments are used before the target is reached, the
/// remainder is taken in one step under the active set reached, since
/// stopping short would answer a perturbation the caller did not ask
/// for. A returned segment count equal to `max_iter` is what says that
/// happened.
#[allow(clippy::too_many_arguments)]
pub fn step_along_path<B>(
    backsolver: &B,
    rhs_plain: &[Number],
    x_curr: &[Number],
    lo: &[Number],
    hi: &[Number],
    multipliers: &[BoundMultiplier],
    max_iter: usize,
    forced_active: &[usize],
    initial_holds: &[(usize, bool)],
    weak_rows: &[usize],
) -> Result<(Vec<Number>, Vec<PathSegment>), String>
where
    B: crate::backsolver::SensBacksolver + Clone,
{
    let n_full = backsolver.dim();
    let n_x = x_curr.len().min(lo.len()).min(hi.len());
    if rhs_plain.len() != n_full {
        return Err("step_along_path: rhs length is not the KKT dimension".into());
    }
    // The same conversion the refinement makes, for the same reason:
    // these arrive in the solve's coordinates and get compared against
    // the z rows of a step, which are in the model's.
    let mult_nat: Vec<BoundMultiplier> = match backsolver.natural_units_factor() {
        None => multipliers
            .iter()
            .map(|m| BoundMultiplier {
                row: m.row,
                base: m.base,
            })
            .collect(),
        Some(f) => multipliers
            .iter()
            .map(|m| BoundMultiplier {
                row: m.row,
                base: m.base * f[m.row],
            })
            .collect(),
    };
    let bound_rows: Option<Vec<crate::backsolver::BoundRow>> =
        backsolver.bound_rows().map(|b| b.to_vec());
    let can_release = backsolver.supports_release();

    // Which bounds the factorization enforces, decided once. Active
    // means the multiplier dominates the slack. A converged interior
    // point never sits ON a bound: an active bound's slack is order mu
    // over the multiplier, so testing slack against `eps` calls every
    // active bound inactive and the path never releases anything.
    // Complementarity splits the two sides cleanly, z of order one
    // against slack of order mu on the active side and the reverse on
    // the inactive, which is the same split the activity classifier
    // draws.
    //
    // The split is evaluated at the BASE point, which is what makes
    // deciding it here, before the loop, correct rather than a cache:
    // activity of a multiplier row is a property of the factorization,
    // whose sigma for this bound was frozen at the base, and a bound
    // inactive there is represented by a Schur-row hold if the path
    // reaches it, never by its multiplier row. Testing accumulated
    // values instead let a near-bound inactive multiplier drift past
    // its shrinking slack mid-path and "release" a bound that was
    // never held, putting a departure in the record for a variable
    // that was not on that bound.
    //
    // What stays live at every consumer is the released list: a
    // base-active bound whose row has been released is no longer in
    // the factorization, from that fraction on.
    let mut base_active_row: Vec<[Option<usize>; 2]> = vec![[None, None]; n_x];
    if let Some(rows) = bound_rows.as_ref() {
        for br in rows {
            if br.var_row >= n_x {
                continue;
            }
            let slack_base = if br.lower {
                x_curr[br.var_row] - lo[br.var_row]
            } else {
                hi[br.var_row] - x_curr[br.var_row]
            };
            if !slack_base.is_finite() {
                continue;
            }
            if forced_active.contains(&br.row)
                || mult_nat
                    .iter()
                    .any(|m| m.row == br.row && m.base > slack_base)
            {
                let side = if br.lower { 0 } else { 1 };
                base_active_row[br.var_row][side] = Some(br.row);
            }
        }
    }
    let base_active_rows: Vec<usize> = base_active_row
        .iter()
        .flatten()
        .filter_map(|slot| *slot)
        .collect();

    let mut acc = vec![0.0; n_full];
    let mut t = 0.0_f64;
    // Seeded state from the directional-derivative decision at a
    // degenerate base point. A weakly active row the direction holds
    // arrives released, since its order-one sigma is wrong once the
    // direction later changes, and pinned through a Schur hold with
    // zero accumulated multiplier, exactly as a hold added at fraction
    // zero would, so the drop test can end it later like any other. A
    // weakly active row the direction leaves goes into the
    // base-activity table below instead, so the release scan frees it
    // at the fraction where its multiplier actually reaches zero:
    // essentially zero at an exact kink, and partway along the step
    // when the held solve sits inside the ambiguous band, where the
    // bound is genuinely active for the first stretch. Deciding those
    // rows at fraction zero released them a sixth of a step early on
    // the CSTR held at 75% of the breakpoint fraction, and overshot
    // tenfold against the walk's own release. A leaver is not a
    // one-way door, though: `weak_rows` keeps it reachable, so a
    // direction that turns out to press into it is a breakpoint and
    // the walk takes the bound back there (gh#852).
    let mut holds: Vec<PathHold> = initial_holds
        .iter()
        .map(|&(row, lower)| PathHold {
            row,
            lower,
            mult: 0.0,
        })
        .collect();
    let mut released: Vec<usize> = initial_holds
        .iter()
        .filter_map(|&(var_row, lower)| {
            bound_rows.as_ref().and_then(|rows| {
                rows.iter()
                    .find(|b| b.var_row == var_row && b.lower == lower)
                    .map(|b| b.row)
            })
        })
        .collect();
    let mut segments: Vec<PathSegment> = Vec::new();
    // Rows already changed at the fraction the path currently ends at.
    // A zero-length segment is where cycling comes from, so a row that
    // just changed cannot change back at the same fraction. The list
    // clears as soon as the path advances: barring a row any longer
    // makes it miss real breakpoints in the following segment,
    // which showed up as a released variable whose next bound crossing
    // went unrecorded.
    let mut changed_here: Vec<usize> = Vec::new();
    let mut last_beta = 1.0_f64;

    /// What the earliest breakpoint found so far does.
    #[derive(Clone, Copy, PartialEq)]
    enum Event {
        ReachLower,
        ReachUpper,
        ReleaseBase,
        DropHold,
    }

    for _ in 0..max_iter {
        if last_beta > PATH_MIN_SEGMENT {
            changed_here.clear();
        }
        let held: Vec<usize> = holds.iter().map(|h| h.row).collect();
        let (d, du) = path_direction(backsolver, rhs_plain, &released, &held)?;
        let remaining = 1.0 - t;
        if remaining <= 0.0 {
            break;
        }

        let mut best: Option<(Number, usize, Event)> = None;
        let mut offer = |beta: Number, row: usize, ev: Event| {
            if !beta.is_finite() || beta < 0.0 || beta > remaining {
                return;
            }
            match best {
                Some((b, _, _)) if b <= beta => {}
                _ => best = Some((beta, row, ev)),
            }
        };

        // A free variable reaching a bound, or a weakly active one
        // reaching it again. A bound the held factorization actually
        // enforces is not reachable this way: its variable sits
        // essentially on it already, and holding it AGAIN through a
        // Schur row would enforce the same bound twice. Such a bound
        // leaves the active set only through its own multiplier's
        // release below. "Actually enforces" is the distinction the
        // `factor_holds` comment below draws, and it is narrower than
        // "active at the base".
        for i in 0..n_x {
            if holds.iter().any(|h| h.row == i) || changed_here.contains(&i) {
                continue;
            }
            // Base activity was decided once, at the table above; only
            // the released exclusion is live, since a released bound
            // left the factorization mid-path.
            //
            // A weakly active row is the exception, and gh#852 is what
            // it costs to leave it out. Its sigma is order ONE, not
            // order 1/mu: the factorization carries the bound as a
            // finite penalty that bends the direction and does not
            // enforce anything, so a direction that drives the
            // variable outside its bound does exactly that, with no
            // breakpoint to stop it. Excluding it here left the
            // coupled kink's walk with nothing to report and the
            // crossing coordinate outside its box, repaired downstream
            // only by a clamp, which moves that coordinate and leaves
            // every neighbour at the one-sided value.
            let factor_holds = |lower_side: bool| -> bool {
                let side = if lower_side { 0 } else { 1 };
                base_active_row[i][side]
                    .is_some_and(|r| !released.contains(&r) && !weak_rows.contains(&r))
            };
            let v = x_curr[i] + acc[i];
            if d[i] < 0.0 && lo[i] > NO_BOUND_LO && !factor_holds(true) {
                offer((lo[i] - v) / d[i], i, Event::ReachLower);
            }
            if d[i] > 0.0 && hi[i] < NO_BOUND_HI && !factor_holds(false) {
                offer((hi[i] - v) / d[i], i, Event::ReachUpper);
            }
        }
        // A bound active at the base whose multiplier reaches zero.
        // Base activity comes from the table above; which rows have
        // since been released stays a live check.
        if can_release {
            for m in &mult_nat {
                if released.contains(&m.row)
                    || changed_here.contains(&m.row)
                    || !base_active_rows.contains(&m.row)
                {
                    continue;
                }
                let z_curr = m.base + acc[m.row];
                if d[m.row] < 0.0 {
                    offer(-z_curr / d[m.row], m.row, Event::ReleaseBase);
                }
            }
        }
        // A hold this path added whose multiplier crosses zero. The
        // rate is the row's `du` under the current direction. Which
        // sign is the valid side depends on conventions three layers
        // deep, so the test does not choose one: the multiplier took
        // some sign on the segment after the hold was added, and
        // crossing zero from that side is what ends the hold's
        // validity. At creation the multiplier is exactly zero and the
        // product below is zero, so a fresh hold cannot drop before it
        // has accumulated a sign.
        for (k, h) in holds.iter().enumerate() {
            if changed_here.contains(&h.row) {
                continue;
            }
            let rate = du[k];
            if h.mult * rate < 0.0 {
                offer(-h.mult / rate, h.row, Event::DropHold);
            }
        }

        let Some((beta, row, ev)) = best else {
            // Nothing changes before the target, so the rest is one step.
            for (a, dv) in acc.iter_mut().zip(d.iter()) {
                *a += remaining * dv;
            }
            t = 1.0;
            break;
        };

        for (a, dv) in acc.iter_mut().zip(d.iter()) {
            *a += beta * dv;
        }
        for (k, h) in holds.iter_mut().enumerate() {
            h.mult += beta * du[k];
        }
        last_beta = beta;
        t += beta;
        changed_here.push(row);
        let (var_row, lower) = match ev {
            Event::ReachLower | Event::ReachUpper => {
                let lower = ev == Event::ReachLower;
                // A weakly active bound the walk reaches leaves the
                // factorization at the same fraction, which is the
                // treatment `initial_holds` already gets and for the
                // same reason: from here on the Schur hold is what
                // enforces the bound, and the row's order-one sigma is
                // a second, softer copy of it built at a base point
                // whose direction no longer applies. While the hold
                // stands the two are indistinguishable -- the hold
                // takes the coordinate's movement to zero and sigma
                // multiplies exactly that -- so what the release is
                // for is the fraction AFTER the hold drops, where the
                // coordinate moves again and a stale order-one sigma
                // damps it. The base-activity table is not the test
                // here: sigma is in the factor for every bound, and a
                // weak row lands on either side of that table's
                // multiplier-against-slack comparison.
                let reached_row = bound_rows.as_ref().and_then(|rows| {
                    rows.iter()
                        .find(|b| b.var_row == row && b.lower == lower)
                        .map(|b| b.row)
                });
                if can_release
                    && let Some(r) = reached_row
                    && weak_rows.contains(&r)
                    && !released.contains(&r)
                {
                    released.push(r);
                    changed_here.push(r);
                }
                holds.push(PathHold {
                    row,
                    lower,
                    mult: 0.0,
                });
                (row, lower)
            }
            Event::ReleaseBase => {
                // The release scan only offers rows it found bound
                // metadata for, so this lookup cannot miss.
                let Some(br) = bound_rows
                    .as_ref()
                    .and_then(|rows| rows.iter().find(|b| b.row == row))
                else {
                    return Err("step_along_path: released a row with no bound metadata".into());
                };
                // Bar the released variable's own row too: the reach
                // scan works in var rows while the release recorded the
                // multiplier row, and without this the variable can be
                // re-held at the same fraction it was just released.
                changed_here.push(br.var_row);
                released.push(row);
                (br.var_row, br.lower)
            }
            Event::DropHold => {
                // The drop event came from iterating the holds, so the
                // hold is present.
                let Some(h) = holds.iter().find(|h| h.row == row).copied() else {
                    return Err("step_along_path: dropped a hold that does not exist".into());
                };
                holds.retain(|h| h.row != row);
                (row, h.lower)
            }
        };
        segments.push(PathSegment {
            at: t,
            var_row,
            lower,
            pinned: matches!(ev, Event::ReachLower | Event::ReachUpper),
        });
    }

    // The cap bound before the target was reached, so take what is left
    // under the active set reached.
    if t < 1.0 {
        let held: Vec<usize> = holds.iter().map(|h| h.row).collect();
        let (d, _) = path_direction(backsolver, rhs_plain, &released, &held)?;
        for (a, dv) in acc.iter_mut().zip(d.iter()) {
            *a += (1.0 - t) * dv;
        }
    }
    Ok((acc, segments))
}

/// The step for the whole perturbation under the active set the path
/// has reached: released bounds out of the operator with their
/// multipliers constrained to stay at zero, and held variables kept
/// where they are.
///
/// The multiplier constraint is not optional. The re-factored released
/// operator drops the bound's diagonal term, but the factor's
/// complementarity row for that bound still couples the direction
/// through the base slack and multiplier it was built from, and
/// without the constraint the released direction is measurably wrong:
/// on a two-variable QP the free direction after a release came back
/// [1.154, 0.194] against the analytic [1.227, 0.454].
/// A bound the classifier could not call active or inactive at the
/// base point: variable on the bound with a multiplier of the same
/// order as the slack, both order sqrt(mu). The solution map has a
/// kink there, and no single linear step is right for both sides.
#[derive(Clone, Copy, Debug)]
pub struct WeakBound {
    /// Bound-multiplier row in the compound KKT vector.
    pub row: usize,
    /// Var-x row of the variable the bound covers.
    pub var_row: usize,
    /// `true` when the bound is the variable's lower bound.
    pub lower: bool,
}

pub(crate) fn path_direction<B>(
    backsolver: &B,
    rhs_plain: &[Number],
    released: &[usize],
    pinned: &[usize],
) -> Result<(Vec<Number>, Vec<Number>), String>
where
    B: crate::backsolver::SensBacksolver + Clone,
{
    use crate::sens_app::{SensApplication, SensOptions};

    let n_full = backsolver.dim();
    let mut d = vec![0.0; n_full];
    let ok = if released.is_empty() {
        backsolver.solve(rhs_plain, &mut d)
    } else {
        backsolver.solve_released(released, rhs_plain, &mut d)
    };
    if !ok {
        return Err("step_along_path: back-solve failed".into());
    }
    if pinned.is_empty() {
        return Ok((d, Vec::new()));
    }
    // Hold each variable where the path left it, on its bound, by
    // asking the augmented system for the correction that takes its
    // further movement to zero.
    let rows: Vec<Index> = pinned.iter().map(|&r| r as Index).collect();
    let signs = vec![1; rows.len()];
    let mk =
        |r: Vec<Index>| IndexSchurData::from_parts(r, signs.clone()).map_err(|e| format!("{e:?}"));
    let opts = SensOptions {
        run_sens: true,
        ..SensOptions::default()
    };
    let view = ReleasedView {
        base: backsolver.clone(),
        rows: released.to_vec(),
    };
    let mut app = SensApplication::new(mk(rows.clone())?, view, opts);
    let rhs: Vec<Number> = pinned.iter().map(|&i| d[i]).collect();
    let mut du = vec![0.0; rows.len()];
    let mut corr = vec![0.0; n_full];
    if !app.run_sens_step(&mk(rows)?, &rhs, &mut du, &mut corr) {
        return Err(format!(
            "step_along_path: augmented solve failed (holds {pinned:?}, released {released:?})"
        ));
    }
    for (k, v) in d.iter_mut().enumerate() {
        *v += corr[k];
    }
    Ok((d, du))
}
