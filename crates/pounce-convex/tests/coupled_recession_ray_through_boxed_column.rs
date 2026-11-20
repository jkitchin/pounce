//! The recession test projects boxed columns out of a candidate ray
//! (`sign_projected_direction`, the leo2 fix): a column bounded on both sides
//! has no recession component. That projection leaves the boxed column's
//! finite offset in every equality row it shares with a free column, so a
//! *genuine* ray coupled to a boxed column through an equality is the case it
//! could have hidden.
//!
//! `min −x₁ s.t. x₁ − x₂ + Σ boxed = b`, `x₁, x₂` free (or `x₁ ≥ 0`): the ray
//! `(1, 1, 0, …)` is admissible and drives the objective to −∞. Closed form,
//! so the test checks against the truth. Both bound representations — `lb/ub`
//! vectors and explicit single-coefficient `G` rows — are covered, because the
//! projection reads the latter.

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

const INF: f64 = f64::INFINITY;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `boxes[j] = (lo, hi)` for each boxed column after `x₁, x₂`.
fn coupled(b: f64, x1_lb: f64, boxes: &[(f64, f64)], as_rows: bool) -> QpProblem {
    let n = 2 + boxes.len();
    let mut c = vec![0.0; n];
    c[0] = -1.0;
    let mut a = vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -1.0)];
    for j in 0..boxes.len() {
        a.push(Triplet::new(0, 2 + j, 1.0));
    }
    let mut lb = vec![x1_lb, -INF];
    let mut ub = vec![INF, INF];
    let (mut g, mut h) = (vec![], vec![]);
    for (j, &(lo, hi)) in boxes.iter().enumerate() {
        if as_rows {
            g.push(Triplet::new(h.len(), 2 + j, -1.0));
            h.push(-lo);
            g.push(Triplet::new(h.len(), 2 + j, 1.0));
            h.push(hi);
            lb.push(-INF);
            ub.push(INF);
        } else {
            lb.push(lo);
            ub.push(hi);
        }
    }
    QpProblem {
        n,
        p_lower: vec![],
        c,
        a,
        b: vec![b],
        g,
        h,
        lb,
        ub,
    }
}

#[test]
fn a_ray_coupled_to_boxed_columns_is_still_unbounded() {
    let cases: &[(&str, f64, f64, &[(f64, f64)])] = &[
        ("one box, b=0", 0.0, -INF, &[(0.0, 1.0)]),
        ("one box, b=5", 5.0, -INF, &[(0.0, 1.0)]),
        (
            "x1>=0, three boxes",
            2.0,
            0.0,
            &[(0.0, 1.0), (0.0, 3.0), (0.0, 0.5)],
        ),
    ];
    for &(name, b, x1_lb, boxes) in cases {
        for as_rows in [false, true] {
            let prob = coupled(b, x1_lb, boxes, as_rows);
            let st = solve_qp_ipm(&prob, &QpOptions::default(), backend).status;
            assert_eq!(
                st,
                QpStatus::DualInfeasible,
                "{name}, bounds as rows: {as_rows}"
            );
        }
    }
}

/// Other branch: bound `x₂` too, and the same coupling has no ray — the LP is
/// bounded, so the projection must not manufacture an unbounded verdict.
#[test]
fn the_same_coupling_with_x2_bounded_is_optimal() {
    let mut prob = coupled(0.0, -INF, &[(0.0, 1.0)], false);
    prob.ub[1] = 10.0;
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    // x₁ = x₂ − x₃ is maximised at x₂ = 10, x₃ = 0.
    assert!((sol.x[0] - 10.0).abs() < 1e-6, "x = {:?}", sol.x);
}
