//! Validation of the convex-QP interior-point solver against problems
//! with analytically known optima (Phase 2). Each test checks the
//! primal solution, the objective, and — where the optimum is interior
//! or the active set is known — the dual/KKT conditions.
//!
//! FERAL backs the augmented-system factorization so the IPM runs
//! end-to-end without an external linear solver.

use pounce_convex::{solve_qp_ipm, QpOptions, QpProblem, QpStatus, Triplet};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve(prob: &QpProblem) -> pounce_convex::QpSolution {
    let opts = QpOptions::default();
    solve_qp_ipm(prob, &opts, backend)
}

/// min ½‖x − x*‖² , i.e. P = I, c = −x*, no constraints. Optimum x = x*.
#[test]
fn unconstrained_quadratic() {
    // min ½(x0² + x1²) − 3 x0 − 4 x1  → optimum (3, 4), f* = −12.5
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-3.0, -4.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 3.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 4.0).abs() < 1e-6, "x1={}", sol.x[1]);
    assert!((sol.obj - (-12.5)).abs() < 1e-6, "obj={}", sol.obj);
}

/// Equality-constrained QP with a closed-form KKT solution.
/// min ½(x0² + x1²) s.t. x0 + x1 = 2.  Optimum (1, 1), f* = 1, y = −1.
#[test]
fn equality_constrained_quadratic() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![0.0, 0.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        b: vec![2.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);
    assert!((sol.obj - 1.0).abs() < 1e-6, "obj={}", sol.obj);
}

/// Inequality-constrained QP where the constraint is active at optimum.
/// min ½(x0² + x1²) s.t. x0 + x1 ≥ 2  (written as −x0 − x1 ≤ −2).
/// Optimum (1, 1), f* = 1, active with z = 1.
#[test]
fn inequality_active_at_optimum() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![0.0, 0.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, -1.0), Triplet::new(0, 1, -1.0)],
        h: vec![-2.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);
    assert!((sol.obj - 1.0).abs() < 1e-6, "obj={}", sol.obj);
    assert!(
        sol.z[0] > 0.5,
        "constraint should be active, z={}",
        sol.z[0]
    );
}

/// Inequality that is *inactive* at optimum: the unconstrained optimum
/// already satisfies it, so z → 0.
/// min ½((x0−3)² + (x1−4)²) s.t. x0 + x1 ≤ 100. Optimum (3, 4), z ≈ 0.
#[test]
fn inequality_inactive_at_optimum() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-3.0, -4.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![100.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 3.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 4.0).abs() < 1e-6, "x1={}", sol.x[1]);
    assert!(
        sol.z[0] < 1e-5,
        "constraint should be inactive, z={}",
        sol.z[0]
    );
}

/// Bound-constrained QP: min ½(x0² + x1²) − 3 x0 − 4 x1 s.t. x0 ≤ 1.
/// Bounds are expressed as inequality rows. Optimum: x0 = 1 (bound
/// active), x1 = 4 (free). f* = ½(1+16) − 3 − 16 = 8.5 − 19 = −10.5.
#[test]
fn bound_constrained_quadratic() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-3.0, -4.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0)], // x0 ≤ 1
        h: vec![1.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 4.0).abs() < 1e-6, "x1={}", sol.x[1]);
    assert!((sol.obj - (-10.5)).abs() < 1e-6, "obj={}", sol.obj);
}

/// LP as the P = 0 case: min −x0 − x1 s.t. x0 ≤ 1, x1 ≤ 1, x ≥ 0.
/// Optimum (1, 1), f* = −2.
#[test]
fn lp_via_empty_hessian() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![], // P = 0  → LP
        c: vec![-1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),  // x0 ≤ 1
            Triplet::new(1, 1, 1.0),  // x1 ≤ 1
            Triplet::new(2, 0, -1.0), // −x0 ≤ 0  (x0 ≥ 0)
            Triplet::new(3, 1, -1.0), // −x1 ≤ 0  (x1 ≥ 0)
        ],
        h: vec![1.0, 1.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);
    assert!((sol.obj - (-2.0)).abs() < 1e-6, "obj={}", sol.obj);
}

/// Coupled Hessian (off-diagonal P term) with an equality constraint.
/// min ½(x0² + x1²) + x0 x1 s.t. x0 + x1 = 2 → wait, P = [[1,1],[1,1]]
/// is only PSD (singular). Use P = [[2,1],[1,2]] (PD): min ½ xᵀP x with
/// x0 + x1 = 2. Optimum is x0 = x1 = 1 by symmetry; f* = ½·(2+2+2)=3.
#[test]
fn coupled_hessian_equality() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 0, 1.0), // off-diagonal (lower)
            Triplet::new(1, 1, 2.0),
        ],
        c: vec![0.0, 0.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        b: vec![2.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);
    assert!((sol.obj - 3.0).abs() < 1e-6, "obj={}", sol.obj);
}
