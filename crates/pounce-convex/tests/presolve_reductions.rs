//! Tests for the LP-oriented presolve reductions (free columns,
//! duplicate rows) and their detections.
//!
//! Duplicate-row multipliers are non-unique, so where a reduction's dual
//! is not uniquely determined we verify that the postsolved point is a
//! *valid KKT point of the original problem* (stationarity, primal
//! feasibility, sign and complementarity of inequality duals) rather
//! than asserting equality with an independent solve.

use pounce_convex::presolve::{presolve, solve_with_presolve, PresolveOutcome};
use pounce_convex::{solve_qp_ipm, QpOptions, QpProblem, QpStatus, Triplet};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn with_presolve(prob: &QpProblem) -> pounce_convex::QpSolution {
    solve_with_presolve(prob, |r| solve_qp_ipm(r, &QpOptions::default(), backend))
}

/// Assert the solution satisfies the original problem's KKT conditions.
fn assert_kkt(prob: &QpProblem, sol: &pounce_convex::QpSolution, tol: f64) {
    let n = prob.n;
    // Stationarity: Px + c + Aᵀy + Gᵀz = 0.
    let mut g = prob.c.clone();
    prob.p_mul(&sol.x, &mut g);
    prob.at_mul(&sol.y, &mut g);
    prob.gt_mul(&sol.z, &mut g);
    for (i, gi) in g.iter().enumerate() {
        assert!(gi.abs() < tol, "stationarity[{i}] = {gi}");
    }
    // Primal equality feasibility: Ax = b.
    let mut ax = vec![0.0; prob.m_eq()];
    prob.a_mul(&sol.x, &mut ax);
    for (i, (&axi, &bi)) in ax.iter().zip(&prob.b).enumerate() {
        assert!((axi - bi).abs() < tol, "Ax=b row {i}: {axi} vs {bi}");
    }
    // Primal inequality feasibility Gx ≤ h, dual sign z ≥ 0, and
    // complementarity z·(h − Gx) ≈ 0.
    let mut gx = vec![0.0; prob.m_ineq()];
    prob.g_mul(&sol.x, &mut gx);
    for i in 0..prob.m_ineq() {
        let slack = prob.h[i] - gx[i];
        assert!(slack > -tol, "Gx≤h row {i}: slack {slack}");
        assert!(sol.z[i] > -tol, "z[{i}] = {} < 0", sol.z[i]);
        assert!(
            (sol.z[i] * slack).abs() < 1e-4,
            "complementarity row {i}: z={} slack={slack}",
            sol.z[i]
        );
    }
}

// --- free / empty columns ---

/// A variable absent from P, A, G with zero cost is irrelevant: presolve
/// pins it to 0 and the rest of the problem solves normally.
#[test]
fn free_column_zero_cost_dropped() {
    // min x0²  s.t. x0 = 2 ; x1 is free with c1 = 0 (irrelevant).
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0, 0.0],
        a: vec![Triplet::new(0, 0, 1.0)], // x0 = 2
        b: vec![2.0],
        g: vec![],
        h: vec![],
    };
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 2.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!(
        sol.x[1].abs() < 1e-9,
        "free x1 should be 0, got {}",
        sol.x[1]
    );
}

/// A free column with nonzero cost makes the problem unbounded below.
#[test]
fn free_column_nonzero_cost_unbounded() {
    // min x0² − x1, x1 free → unbounded (x1 → +∞).
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
    };
    assert!(matches!(presolve(&prob), PresolveOutcome::Unbounded));
    assert_eq!(with_presolve(&prob).status, QpStatus::DualInfeasible);
}

// --- duplicate rows ---

/// Duplicate equality rows with the same rhs are redundant: drop one,
/// solve, recovered point is KKT-valid for the original problem.
#[test]
fn duplicate_equality_rows_redundant() {
    // min x0²+x1² s.t. x0+x1=2 (twice). Optimum (1,1).
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, 1.0), // duplicate of row 0
            Triplet::new(1, 1, 1.0),
        ],
        b: vec![2.0, 2.0],
        g: vec![],
        h: vec![],
    };
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);
    assert_kkt(&prob, &sol, 1e-5);
}

/// Duplicate equality rows with *different* rhs are infeasible.
#[test]
fn duplicate_equality_rows_conflicting_infeasible() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 1.0),
        ],
        b: vec![2.0, 3.0], // x0+x1 can't be both 2 and 3
        g: vec![],
        h: vec![],
    };
    assert!(matches!(presolve(&prob), PresolveOutcome::Infeasible));
    assert_eq!(with_presolve(&prob).status, QpStatus::PrimalInfeasible);
}

/// Duplicate inequality rows: keep the tightest. `x0+x1 ≤ 3` and
/// `x0+x1 ≤ 1` (same lhs) → effective bound is 1.
#[test]
fn duplicate_inequality_keeps_tightest() {
    // min ½‖x−(5,5)‖² (via c=−5·2) s.t. x0+x1 ≤ 3 and x0+x1 ≤ 1.
    // Tightest is x0+x1 ≤ 1; optimum on that line at (0.5, 0.5).
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-10.0, -10.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0), // x0+x1 ≤ 3
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 1.0), // x0+x1 ≤ 1  (tighter)
        ],
        h: vec![3.0, 1.0],
    };
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 0.5).abs() < 1e-5, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 0.5).abs() < 1e-5, "x1={}", sol.x[1]);
    assert_kkt(&prob, &sol, 1e-5);
}

/// A many-duplicate problem exercises the parallel hashing path and must
/// still produce a KKT-valid point.
#[test]
fn many_duplicate_rows_parallel_path() {
    // min Σ x_i²  s.t.  Σ x_i = n  repeated K times. Optimum x = 1.
    let n = 30usize;
    let k = 50usize; // K identical equality rows
    let mut p_lower = Vec::new();
    for i in 0..n {
        p_lower.push(Triplet::new(i, i, 2.0));
    }
    let mut a = Vec::new();
    for row in 0..k {
        for i in 0..n {
            a.push(Triplet::new(row, i, 1.0));
        }
    }
    let prob = QpProblem {
        n,
        p_lower,
        c: vec![0.0; n],
        a,
        b: vec![n as f64; k],
        g: vec![],
        h: vec![],
    };
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    for i in 0..n {
        assert!((sol.x[i] - 1.0).abs() < 1e-5, "x[{i}]={}", sol.x[i]);
    }
    assert_kkt(&prob, &sol, 1e-4);
}
