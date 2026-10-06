//! gh #997: the gh #848 screen refused a **strict local minimum** of an
//! indefinite QP because a better point existed in *another basin*.
//!
//! `refute_indefinite_optimum` walks a negative-curvature direction to the
//! boundary and demotes on any strictly better feasible point. That proves the
//! point is not a **global** minimum, and the engine never claims one on an
//! indefinite `P`. On `min −x² + 0.3x` over `[−1, 2]` the engine stops at
//! `x = −1`, where the bound is strongly active (`f'(−1) = 2.3`) and the
//! critical cone is `{0}`: a strict local minimum. The walk along `+1` leaves
//! the bound **uphill** and ends at `x = 2`, `f = −3.4 < −1.3`, and the CLI
//! reported `Internal_Error` while `pounce.solve_qp(method="active-set")`
//! reported `optimal` on the same point.
//!
//! ## Which branch each test reaches
//!
//! | test | walk slope `gᵀd` | verdict |
//! |---|---|---|
//! | `one_d_strict_local_min_at_a_bound_is_optimal` | `> 0` (uphill) | `Optimal` |
//! | `two_d_strict_local_min_is_optimal_with_and_without_a_row` | `> 0` (uphill) | `Optimal` |
//! | `a_maximum_along_an_active_row_is_still_refused` | `= 0` (critical cone) | refused |
//!
//! The last is the other branch of the new gate, and the reason it is here: a
//! gate that refused nothing would pass the first two.

use pounce_convex::active_set::take_second_order_refusal;
use pounce_convex::{
    ActiveSetOverrides, HessianInertia, QpOptions, QpProblem, QpSolution, QpStatus, Triplet,
    solve_qp_active_set_inertia,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve_indefinite(prob: &QpProblem) -> (QpSolution, bool) {
    let _ = take_second_order_refusal();
    let mut mk = backend;
    let sol = solve_qp_active_set_inertia(
        prob,
        &QpOptions::default(),
        &ActiveSetOverrides::default(),
        HessianInertia::Indefinite,
        &mut mk,
    );
    (sol, take_second_order_refusal())
}

/// `min −x² + 0.3x` over `[−1, 2]` — the issue's reproduction.
#[test]
fn one_d_strict_local_min_at_a_bound_is_optimal() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, -2.0)],
        c: vec![0.3],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![-1.0],
        ub: vec![2.0],
    };
    let (sol, refused) = solve_indefinite(&prob);
    // The engine may legitimately land at either local minimum; what it must
    // not do is refuse one. Both endpoints are strict local minima.
    assert_eq!(sol.status, QpStatus::Optimal, "x = {:?}", sol.x);
    assert!(!refused);
    let x = sol.x[0];
    assert!((x + 1.0).abs() < 1e-8 || (x - 2.0).abs() < 1e-8, "x = {x}");
}

/// `min −x₀² + x₁² + 0.1x₀x₁ + 0.3x₀` on `[−1, 2]²`, optionally with the
/// inactive row `x₀ + x₁ ≤ 3`. The issue's 2-D variant: the engine stops at
/// `(−1, 0.05)`, `g₀ = 2.305 > 0`, reduced Hessian `2 > 0`.
#[test]
fn two_d_strict_local_min_is_optimal_with_and_without_a_row() {
    for with_row in [false, true] {
        let (g, h) = if with_row {
            (
                vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
                vec![3.0],
            )
        } else {
            (vec![], vec![])
        };
        let prob = QpProblem {
            n: 2,
            p_lower: vec![
                Triplet::new(0, 0, -2.0),
                Triplet::new(1, 0, 0.1),
                Triplet::new(1, 1, 2.0),
            ],
            c: vec![0.3, 0.0],
            a: vec![],
            b: vec![],
            g,
            h,
            lb: vec![-1.0, -1.0],
            ub: vec![2.0, 2.0],
        };
        let (sol, refused) = solve_indefinite(&prob);
        assert_eq!(
            sol.status,
            QpStatus::Optimal,
            "with_row = {with_row}, x = {:?}",
            sol.x
        );
        assert!(!refused, "with_row = {with_row}");
    }
}

/// `min x₀x₁ s.t. x₀ + x₁ ≥ 2` over `[0, 4]²` (the corpus's
/// `nonconvex_qp_ineq`). At `(1, 1)` the row is active and `(1, −1)` lies in
/// its null space with `gᵀd = 0` and `dᵀPd = −2`: a maximum along the row, not a
/// local minimum. The local gate must still let this refutation through.
#[test]
fn a_maximum_along_an_active_row_is_still_refused() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(1, 0, 1.0)],
        c: vec![0.0, 0.0],
        a: vec![],
        b: vec![],
        // x₀ + x₁ ≥ 2  ⇔  −x₀ − x₁ ≤ −2
        g: vec![Triplet::new(0, 0, -1.0), Triplet::new(0, 1, -1.0)],
        h: vec![-2.0],
        lb: vec![0.0, 0.0],
        ub: vec![4.0, 4.0],
    };
    let (sol, refused) = solve_indefinite(&prob);
    let f = sol.x[0] * sol.x[1];
    if sol.status == QpStatus::Optimal {
        // An escape to an endpoint is the better outcome and is a genuine
        // local minimum (`f = 0`). Reporting `(1, 1)` optimal is the defect.
        assert!(f.abs() < 1e-8, "certified the row maximum: x = {:?}", sol.x);
    } else {
        assert!(
            refused,
            "refused for a non-second-order reason: {:?}",
            sol.status
        );
    }
}
