//! Verified infeasibility / unboundedness detection (the HSDE benefit:
//! clean status instead of exhausting the iteration budget).
//!
//! Each declared status is backed by a checked certificate, so these
//! tests also implicitly confirm there are no false positives — the
//! feasible/optimal problems in the rest of the suite must still report
//! `Optimal`, and a couple of those are re-checked here for contrast.

use pounce_convex::{solve_qp_ipm, QpOptions, QpProblem, QpStatus, Triplet};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve(prob: &QpProblem) -> pounce_convex::QpSolution {
    solve_qp_ipm(prob, &QpOptions::default(), backend)
}

/// Primal-infeasible: contradictory equalities x0 = 1 and x0 = 2.
/// (min x0² subject to both.) No x satisfies the constraints.
#[test]
fn primal_infeasible_contradictory_equalities() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 0, 1.0)],
        b: vec![1.0, 2.0],
        g: vec![],
        h: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(
        sol.status,
        QpStatus::PrimalInfeasible,
        "expected primal infeasible, got {:?} after {} iters",
        sol.status,
        sol.iters
    );
}

/// Primal-infeasible via inequalities: x0 ≤ 0 and x0 ≥ 1 (written
/// −x0 ≤ −1). Empty feasible set.
#[test]
fn primal_infeasible_contradictory_inequalities() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),  // x0 ≤ 0
            Triplet::new(1, 0, -1.0), // −x0 ≤ −1  (x0 ≥ 1)
        ],
        h: vec![0.0, -1.0],
    };
    let sol = solve(&prob);
    assert_eq!(
        sol.status,
        QpStatus::PrimalInfeasible,
        "got {:?} after {} iters",
        sol.status,
        sol.iters
    );
}

/// Unbounded LP: min −x0 with x0 ≥ 0 (no upper bound). Objective → −∞
/// along the recession direction d = (1).
#[test]
fn dual_infeasible_unbounded_lp() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![], // LP (P = 0)
        c: vec![-1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, -1.0)], // −x0 ≤ 0  (x0 ≥ 0)
        h: vec![0.0],
    };
    let sol = solve(&prob);
    assert_eq!(
        sol.status,
        QpStatus::DualInfeasible,
        "expected unbounded (dual infeasible), got {:?} after {} iters",
        sol.status,
        sol.iters
    );
}

/// Unbounded QP: a singular Hessian with a recession direction. min x1²
/// − x0 with x0 free, x1 free. The x0 direction has Pd = 0 and cᵀd < 0,
/// so the objective is unbounded below.
#[test]
fn dual_infeasible_unbounded_qp_singular_hessian() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(1, 1, 2.0)], // only x1 is in P
        c: vec![-1.0, 0.0],                     // −x0
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(
        sol.status,
        QpStatus::DualInfeasible,
        "got {:?} after {} iters",
        sol.status,
        sol.iters
    );
}

/// Contrast: a feasible, bounded QP must still report Optimal — the
/// detector must not false-positive. min (x0−1)² + (x1−1)², 0 ≤ x ≤ 5.
#[test]
fn feasible_bounded_still_optimal() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-2.0, -2.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 0, -1.0),
            Triplet::new(3, 1, -1.0),
        ],
        h: vec![5.0, 5.0, 0.0, 0.0],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-6);
    assert!((sol.x[1] - 1.0).abs() < 1e-6);
}
