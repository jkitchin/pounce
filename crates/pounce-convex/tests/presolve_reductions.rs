//! Tests for the LP-oriented presolve reductions (free columns,
//! duplicate rows) and their detections.
//!
//! Duplicate-row multipliers are non-unique, so where a reduction's dual
//! is not uniquely determined we verify that the postsolved point is a
//! *valid KKT point of the original problem* (stationarity, primal
//! feasibility, sign and complementarity of inequality duals) rather
//! than asserting equality with an independent solve.

use pounce_convex::presolve::{PresolveOutcome, presolve, solve_with_presolve};
use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
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

/// Bound-aware KKT check (for reductions that leave a variable at an
/// active box bound, e.g. dominated columns): stationarity carries the
/// bound multipliers, `Px + c + Aᵀy + Gᵀz + z_ub − z_lb = 0`, and both the
/// inequality and the bound complementarities must hold.
fn assert_kkt_bounds(prob: &QpProblem, sol: &pounce_convex::QpSolution, tol: f64) {
    let n = prob.n;
    let mut g = prob.c.clone();
    prob.p_mul(&sol.x, &mut g);
    prob.at_mul(&sol.y, &mut g);
    prob.gt_mul(&sol.z, &mut g);
    for i in 0..n {
        let stat = g[i] + sol.z_ub[i] - sol.z_lb[i];
        assert!(stat.abs() < tol, "stationarity[{i}] = {stat}");
        assert!(
            sol.z_lb[i] > -tol && sol.z_ub[i] > -tol,
            "bound dual sign [{i}]"
        );
        assert!(
            sol.x[i] >= prob.lb_of(i) - tol && sol.x[i] <= prob.ub_of(i) + tol,
            "box [{i}]: {} in [{}, {}]",
            sol.x[i],
            prob.lb_of(i),
            prob.ub_of(i)
        );
        assert!(
            (sol.z_lb[i] * (sol.x[i] - prob.lb_of(i))).abs() < 1e-4,
            "lb comp [{i}]"
        );
        assert!(
            (sol.z_ub[i] * (prob.ub_of(i) - sol.x[i])).abs() < 1e-4,
            "ub comp [{i}]"
        );
    }
    let mut ax = vec![0.0; prob.m_eq()];
    prob.a_mul(&sol.x, &mut ax);
    for (i, (&axi, &bi)) in ax.iter().zip(&prob.b).enumerate() {
        assert!((axi - bi).abs() < tol, "Ax=b row {i}: {axi} vs {bi}");
    }
    let mut gx = vec![0.0; prob.m_ineq()];
    prob.g_mul(&sol.x, &mut gx);
    for i in 0..prob.m_ineq() {
        let slack = prob.h[i] - gx[i];
        assert!(slack > -tol, "Gx≤h row {i}: slack {slack}");
        assert!(sol.z[i] > -tol, "z[{i}] < 0");
        assert!((sol.z[i] * slack).abs() < 1e-4, "ineq comp row {i}");
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
        lb: vec![],
        ub: vec![],
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
        lb: vec![],
        ub: vec![],
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
        lb: vec![],
        ub: vec![],
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
        lb: vec![],
        ub: vec![],
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
        lb: vec![],
        ub: vec![],
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
        lb: vec![],
        ub: vec![],
    };
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    for i in 0..n {
        assert!((sol.x[i] - 1.0).abs() < 1e-5, "x[{i}]={}", sol.x[i]);
    }
    assert_kkt(&prob, &sol, 1e-4);
}

// --- fixpoint cascade ---

/// A chain of fixings that only a *fixpoint* presolve fully unwinds: only
/// one singleton exists initially, but fixing it exposes the next, and so
/// on. Iteration fixes the whole chain (reduced problem empty); a single
/// pass would stop after the first.
#[test]
fn fixpoint_cascades_chain_of_fixings() {
    // x3 = 3 (singleton) → x2 = 5−x3 = 2 → x1 = 7−x2 = 5 → x0 = 9−x1 = 4.
    let prob = QpProblem {
        n: 4,
        p_lower: (0..4).map(|i| Triplet::new(i, i, 2.0)).collect(),
        c: vec![0.0; 4],
        a: vec![
            Triplet::new(0, 2, 1.0),
            Triplet::new(0, 3, 1.0), // x2 + x3 = 5
            Triplet::new(1, 1, 1.0),
            Triplet::new(1, 2, 1.0), // x1 + x2 = 7
            Triplet::new(2, 0, 1.0),
            Triplet::new(2, 1, 1.0), // x0 + x1 = 9
            Triplet::new(3, 3, 1.0), // x3 = 3   (the only initial singleton)
        ],
        b: vec![5.0, 7.0, 9.0, 3.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => {
            // Whole chain fixed ⇒ nothing left to solve.
            assert_eq!(ps.reduced.n, 0, "fixpoint should fix all four variables");
            assert!(ps.stats().fixed_vars >= 4 || ps.stats().free_col_singletons >= 1);
        }
        other => panic!("expected Reduced, got {}", status_of(&other)),
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    let expect = [4.0, 5.0, 2.0, 3.0];
    for i in 0..4 {
        assert!(
            (sol.x[i] - expect[i]).abs() < 1e-6,
            "x[{i}]={} want {}",
            sol.x[i],
            expect[i]
        );
    }
    assert_kkt(&prob, &sol, 1e-5);
}

// --- parallel rows (scalar multiples, not just exact duplicates) ---

/// Parallel equality rows: `x0 + x1 = 2` and `3x0 + 3x1 = 6` are the same
/// constraint scaled by 3. One is dropped; the recovered point is valid.
#[test]
fn parallel_equality_rows_redundant() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0), // x0 + x1 = 2
            Triplet::new(1, 0, 3.0),
            Triplet::new(1, 1, 3.0), // 3x0 + 3x1 = 6  (= 3×row0)
        ],
        b: vec![2.0, 6.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    // One equality row removed by parallel detection.
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => assert_eq!(ps.reduced.m_eq(), 1),
        other => panic!("expected Reduced, got {}", status_of(&other)),
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 1.0).abs() < 1e-6 && (sol.x[1] - 1.0).abs() < 1e-6);
    assert_kkt(&prob, &sol, 1e-5);
}

/// Negatively-scaled parallel equalities: `x0 + x1 = 2` and
/// `−2x0 − 2x1 = −4` are the same constraint. Detected and merged.
#[test]
fn parallel_equality_negative_scale() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, -2.0),
            Triplet::new(1, 1, -2.0), // −2×row0
        ],
        b: vec![2.0, -4.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => assert_eq!(ps.reduced.m_eq(), 1),
        other => panic!("expected Reduced, got {}", status_of(&other)),
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_kkt(&prob, &sol, 1e-5);
}

/// Parallel equalities with inconsistent scaled rhs are infeasible:
/// `x0 + x1 = 2` and `2x0 + 2x1 = 5` (≠ 4).
#[test]
fn parallel_equality_inconsistent_infeasible() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, 2.0),
            Triplet::new(1, 1, 2.0),
        ],
        b: vec![2.0, 5.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    assert!(matches!(presolve(&prob), PresolveOutcome::Infeasible));
}

/// Parallel inequalities (positive multiple): `x0 + x1 ≤ 3` and
/// `2x0 + 2x1 ≤ 2` (⟺ x0 + x1 ≤ 1). The tighter (second) is kept; the
/// optimum lands on x0 + x1 = 1.
#[test]
fn parallel_inequality_keeps_tightest() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-10.0, -10.0], // pull both up; constraint binds
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0), // x0 + x1 ≤ 3
            Triplet::new(1, 0, 2.0),
            Triplet::new(1, 1, 2.0), // 2x0 + 2x1 ≤ 2  ⟺  x0 + x1 ≤ 1
        ],
        h: vec![3.0, 2.0],
        lb: vec![],
        ub: vec![],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => assert_eq!(ps.reduced.m_ineq(), 1),
        other => panic!("expected Reduced, got {}", status_of(&other)),
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] + sol.x[1] - 1.0).abs() < 1e-5, "x={:?}", sol.x);
    assert_kkt(&prob, &sol, 1e-5);
    // Matches the direct solve's primal.
    let d = direct(&prob);
    assert!((sol.x[0] - d.x[0]).abs() < 1e-5 && (sol.x[1] - d.x[1]).abs() < 1e-5);
}

/// Opposite-direction inequalities are *not* merged: `x0 ≤ 3` and
/// `−x0 ≤ −1` (i.e. x0 ≥ 1) form a range, not a duplicate — both kept.
#[test]
fn antiparallel_inequalities_not_merged() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 0, -1.0)],
        h: vec![3.0, -1.0], // x0 ≤ 3 and x0 ≥ 1
        lb: vec![],
        ub: vec![],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => assert_eq!(ps.reduced.m_ineq(), 2, "both kept"),
        other => panic!("expected Reduced, got {}", status_of(&other)),
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_kkt(&prob, &sol, 1e-5);
}

// --- dominated columns ---

/// Dominated column fixed to its lower bound: x2 has no quadratic/equality
/// term, appears only with a nonnegative coefficient in `≤` rows, and has
/// cost c2 ≥ 0 — so pushing it down never hurts. Presolve fixes x2 = lb.
#[test]
fn dominated_column_fixed_to_lower() {
    // min x0² + x1² + 0.5·x2  s.t.  x0 + x1 + x2 ≤ 3,  0 ≤ x ≤ 5.
    // x2: not in P, only in the ≤ row with +1, cost +0.5 ≥ 0 ⇒ x2 = 0.
    let prob = QpProblem {
        n: 3,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-4.0, -4.0, 0.5],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0),
        ],
        h: vec![3.0],
        lb: vec![0.0, 0.0, 0.0],
        ub: vec![5.0, 5.0, 5.0],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => {
            assert_eq!(ps.stats().dominated_cols, 1);
            assert_eq!(ps.reduced.n, 2);
        }
        other => panic!("expected Reduced, got {}", status_of(&other)),
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(sol.x[2].abs() < 1e-6, "x2 fixed to 0: {}", sol.x[2]);
    assert_kkt_bounds(&prob, &sol, 1e-5);
    let d = direct(&prob);
    for i in 0..3 {
        assert!(
            (sol.x[i] - d.x[i]).abs() < 1e-5,
            "x[{i}]: {} vs {}",
            sol.x[i],
            d.x[i]
        );
    }
}

/// Dominated column fixed to its upper bound (mirror): negative `≤`
/// coefficient and nonpositive cost ⇒ pushing it up never hurts.
#[test]
fn dominated_column_fixed_to_upper() {
    // min x0² + x1² − 0.5·x2  s.t.  x0 + x1 − x2 ≤ 1,  0 ≤ x ≤ 4.
    // x2: not in P, coefficient −1 in the ≤ row, cost −0.5 ≤ 0 ⇒ x2 = 4.
    let prob = QpProblem {
        n: 3,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-1.0, -1.0, -0.5],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, -1.0),
        ],
        h: vec![1.0],
        lb: vec![0.0, 0.0, 0.0],
        ub: vec![4.0, 4.0, 4.0],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => assert_eq!(ps.stats().dominated_cols, 1),
        other => panic!("expected Reduced, got {}", status_of(&other)),
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[2] - 4.0).abs() < 1e-6, "x2 fixed to 4: {}", sol.x[2]);
    assert_kkt_bounds(&prob, &sol, 1e-5);
    let d = direct(&prob);
    for i in 0..3 {
        assert!(
            (sol.x[i] - d.x[i]).abs() < 1e-5,
            "x[{i}]: {} vs {}",
            sol.x[i],
            d.x[i]
        );
    }
}

/// A column with *mixed-sign* inequality coefficients is NOT dominated
/// (its effect on feasibility is not sign-definite) — left in place.
#[test]
fn mixed_sign_column_not_dominated() {
    let prob = QpProblem {
        n: 3,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-1.0, -1.0, 0.5],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 2, 1.0),  // +x2 in row 0
            Triplet::new(1, 2, -1.0), // −x2 in row 1  → mixed sign
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 1.0),
        ],
        h: vec![3.0, 3.0],
        lb: vec![0.0, 0.0, 0.0],
        ub: vec![5.0, 5.0, 5.0],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => assert_eq!(ps.stats().dominated_cols, 0),
        // A no-op presolve is also acceptable here.
        _ => {}
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_kkt_bounds(&prob, &sol, 1e-5);
}

/// Dominated column in a pure LP (P = 0), the common case.
#[test]
fn dominated_column_lp() {
    // min −x0 + x1  s.t.  x0 + x1 ≤ 2,  0 ≤ x ≤ 3.
    // x1: cost +1 ≥ 0, coefficient +1 ≥ 0, not in P ⇒ x1 = 0; then x0 = 2.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![-1.0, 1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![2.0],
        lb: vec![0.0, 0.0],
        ub: vec![3.0, 3.0],
    };
    match presolve(&prob) {
        // x1 is dominated; fixpoint iteration then cascades (x0's row
        // becomes redundant, leaving x0 dominated too) — ≥ 1 dominated.
        PresolveOutcome::Reduced(ps) => assert!(ps.stats().dominated_cols >= 1),
        other => panic!("expected Reduced, got {}", status_of(&other)),
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        sol.x[1].abs() < 1e-6 && (sol.x[0] - 2.0).abs() < 1e-6,
        "x={:?}",
        sol.x
    );
    assert_kkt_bounds(&prob, &sol, 1e-5);
}

// --- activity-bound reductions (need the variable box) ---

use pounce_convex::{NEG_INF, POS_INF};

/// Redundant inequality: with x ∈ [0,1]², `x0 + x1 ≤ 5` has max activity
/// 2 ≤ 5, so it is always satisfied → presolve drops it; the recovered
/// point is KKT-valid for the original (un-dropped) problem.
#[test]
fn redundant_inequality_dropped() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-1.0, -1.0], // pull toward (0.5, 0.5), interior
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)], // x0+x1 ≤ 5
        h: vec![5.0],
        lb: vec![0.0, 0.0],
        ub: vec![1.0, 1.0],
    };
    // Presolve should drop the redundant row (0 kept inequalities).
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => {
            assert_eq!(ps.reduced.m_ineq(), 0, "redundant row should be dropped");
        }
        other => panic!("expected Reduced, got {:?}", status_of(&other)),
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 0.5).abs() < 1e-5, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 0.5).abs() < 1e-5, "x1={}", sol.x[1]);
    // The dropped row's dual is 0 — still a valid KKT point.
    assert_kkt(&prob, &sol, 1e-5);
}

/// Activity-infeasible inequality: with x ∈ [2,3], `x0 ≤ 1` has min
/// activity 2 > 1, so no feasible point exists.
#[test]
fn activity_infeasible_inequality() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0)], // x0 ≤ 1
        h: vec![1.0],
        lb: vec![2.0],
        ub: vec![3.0],
    };
    assert!(matches!(presolve(&prob), PresolveOutcome::Infeasible));
    assert_eq!(with_presolve(&prob).status, QpStatus::PrimalInfeasible);
}

/// Activity-infeasible equality: with x ∈ [0,1]², `x0 + x1 = 5` is
/// outside the activity range [0, 2].
#[test]
fn activity_infeasible_equality() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![0.0, 0.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)], // x0+x1 = 5
        b: vec![5.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![1.0, 1.0],
    };
    assert!(matches!(presolve(&prob), PresolveOutcome::Infeasible));
    assert_eq!(with_presolve(&prob).status, QpStatus::PrimalInfeasible);
}

/// A negative-coefficient row exercises the `a < 0` branch of the
/// activity computation: with x ∈ [0,1]², `−x0 − x1 ≤ 0.5` has min
/// activity −2 ≤ 0.5 (not infeasible) and max activity 0 ≤ 0.5
/// (redundant) → dropped.
#[test]
fn redundant_inequality_negative_coeffs() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, -1.0), Triplet::new(0, 1, -1.0)], // −x0−x1 ≤ 0.5
        h: vec![0.5],
        lb: vec![0.0, 0.0],
        ub: vec![1.0, 1.0],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => assert_eq!(ps.reduced.m_ineq(), 0),
        other => panic!("expected Reduced, got {:?}", status_of(&other)),
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_kkt(&prob, &sol, 1e-5);
}

/// Unbounded variables must *not* make a row look redundant: with x0
/// free (no upper bound), `x0 ≤ 1` has max activity +∞, so the row is
/// kept and genuinely binds the solution.
#[test]
fn unbounded_variable_row_not_dropped() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![-10.0], // unconstrained optimum at 5, so x0 ≤ 1 binds
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0)], // x0 ≤ 1
        h: vec![1.0],
        lb: vec![NEG_INF],
        ub: vec![POS_INF],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => {
            assert_eq!(ps.reduced.m_ineq(), 1, "row must be kept (activity +∞)");
        }
        other => panic!("expected Reduced, got {:?}", status_of(&other)),
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 1.0).abs() < 1e-5, "x0={}", sol.x[0]);
}

/// Helper for panic messages: name the non-Reduced outcome.
fn status_of(o: &PresolveOutcome) -> &'static str {
    match o {
        PresolveOutcome::Reduced(_) => "Reduced",
        PresolveOutcome::Infeasible => "Infeasible",
        PresolveOutcome::Unbounded => "Unbounded",
    }
}

// --- free column singleton substitution ---

fn direct(prob: &QpProblem) -> pounce_convex::QpSolution {
    solve_qp_ipm(prob, &QpOptions::default(), backend)
}

/// A free variable in exactly one equality row is substituted out,
/// eliminating both the variable and the row; the recovered (x, y) must
/// match a direct solve.
///
/// min x0² + x1²  s.t.  x0 + x1 + x2 = 3,  with x2 free (no bounds, not
/// in P/G). x2 is a free column singleton in the single equality row; it
/// is substituted as x2 = 3 − x0 − x1. The reduced problem has 2 vars
/// and 0 equality rows. Optimum: x0 = x1 = 0, x2 = 3.
#[test]
fn free_column_singleton_substituted() {
    let prob = QpProblem {
        n: 3,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)], // x2 absent from P
        c: vec![0.0, 0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0),
        ],
        b: vec![3.0],
        g: vec![],
        h: vec![],
        lb: vec![NEG_INF, NEG_INF, NEG_INF],
        ub: vec![POS_INF, POS_INF, POS_INF],
    };
    // Presolve must eliminate the row and the free column.
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => {
            assert_eq!(ps.reduced.n, 2, "x2 should be substituted out");
            assert_eq!(ps.reduced.m_eq(), 0, "the equality row should be consumed");
        }
        other => panic!("expected Reduced, got {:?}", status_of(&other)),
    }
    let d = direct(&prob);
    let p = with_presolve(&prob);
    assert_eq!(p.status, QpStatus::Optimal);
    for i in 0..3 {
        assert!(
            (p.x[i] - d.x[i]).abs() < 1e-5,
            "x[{i}]: presolve {} vs direct {}",
            p.x[i],
            d.x[i]
        );
    }
    assert!((p.x[2] - 3.0).abs() < 1e-5, "x2={}", p.x[2]);
    // The consumed row's multiplier must match the direct solve.
    assert!(
        (p.y[0] - d.y[0]).abs() < 1e-5,
        "y[0]: presolve {} vs direct {}",
        p.y[0],
        d.y[0]
    );
    assert_kkt(&prob, &p, 1e-5);
}

/// Free column singleton with a nonzero objective on the free variable,
/// so the substitution shifts cost onto the surviving variables.
///
/// min x0² + 2·x1  s.t.  x0 + 3·x1 = 6, x1 free (linear-only, not in
/// P/G). x1 = (6 − x0)/3 is substituted; the reduced objective becomes
/// x0² + 2·(6−x0)/3 = x0² − (2/3)x0 + 4. Optimum x0 = 1/3.
#[test]
fn free_column_singleton_shifts_cost() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0, 2.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 3.0)],
        b: vec![6.0],
        g: vec![],
        h: vec![],
        lb: vec![NEG_INF, NEG_INF],
        ub: vec![POS_INF, POS_INF],
    };
    let d = direct(&prob);
    let p = with_presolve(&prob);
    assert_eq!(p.status, QpStatus::Optimal);
    assert!((p.x[0] - (1.0 / 3.0)).abs() < 1e-5, "x0={}", p.x[0]);
    for i in 0..2 {
        assert!(
            (p.x[i] - d.x[i]).abs() < 1e-5,
            "x[{i}]: {} vs {}",
            p.x[i],
            d.x[i]
        );
    }
    assert!(
        (p.obj - d.obj).abs() < 1e-5,
        "obj: presolve {} vs direct {}",
        p.obj,
        d.obj
    );
    assert!(
        (p.y[0] - d.y[0]).abs() < 1e-5,
        "y[0]: {} vs {}",
        p.y[0],
        d.y[0]
    );
    assert_kkt(&prob, &p, 1e-5);
}

/// Regression for the capri LP wrong-answer bug: a free column singleton
/// whose consumed equality row also contains a variable fixed by a
/// *separate* singleton equality row. Postsolve restores the free
/// singleton from the formula `x_col = (b_r − Σ_{j≠col} a_j x_j)/a_col`,
/// which reads the fixed variable's value — so the fixed variable must be
/// restored *before* the free singleton. Naive reverse-LIFO replay (the
/// old code) restored them in push order, leaving the free singleton
/// computed against the fixed var's zero-initialized value and producing a
/// point that violates the consumed row (the silent capri 2625 vs 2690
/// wrong answer).
///
/// min x2²  s.t.  x0 + x1 + x2 = 10,  x1 = 3,  x2 ≥ 0,  x0 free.
/// x1 fixes to 3 (singleton row), the first row becomes x0 + x2 = 7, and
/// x0 (free, now a singleton there) is substituted as x0 = 10 − x1 − x2.
/// Reduced problem: min x2², x2 ≥ 0 → x2 = 0, then x0 = 7, x1 = 3.
#[test]
fn free_singleton_depends_on_fixed_var_postsolve_order() {
    let prob = QpProblem {
        n: 3,
        p_lower: vec![Triplet::new(2, 2, 2.0)], // only x2 in the objective
        c: vec![0.0, 0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0), // x0 + x1 + x2 = 10
            Triplet::new(1, 1, 1.0), // x1 = 3   (singleton → FixedVar)
        ],
        b: vec![10.0, 3.0],
        g: vec![],
        h: vec![],
        lb: vec![NEG_INF, NEG_INF, 0.0], // x0 free; x2 ≥ 0
        ub: vec![POS_INF, POS_INF, POS_INF],
    };
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    // The recovered point must satisfy *both* equality rows. Before the
    // two-pass postsolve fix, row 0 was violated by 3 (x0 restored as 10
    // instead of 7 because x1 was still 0 when the formula was applied).
    let mut ax = vec![0.0; prob.m_eq()];
    prob.a_mul(&sol.x, &mut ax);
    for (i, (&axi, &bi)) in ax.iter().zip(&prob.b).enumerate() {
        assert!((axi - bi).abs() < 1e-6, "Ax=b row {i}: {axi} vs {bi}");
    }
    // x2 only approaches its active bound asymptotically (near-boundary
    // IPM slack), so values are checked to 1e-4; feasibility above is the
    // tight regression guard.
    assert!((sol.x[0] - 7.0).abs() < 1e-4, "x0={} (want 7)", sol.x[0]);
    assert!((sol.x[1] - 3.0).abs() < 1e-4, "x1={} (want 3)", sol.x[1]);
    assert!((sol.x[2] - 0.0).abs() < 1e-4, "x2={} (want 0)", sol.x[2]);
}

/// A bounded variable in one row is *not* a free column singleton (its
/// box can bind), so it must not be substituted.
#[test]
fn bounded_variable_not_substituted() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0, 0.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        b: vec![3.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0], // x1 has a finite lower bound → not free
        ub: vec![POS_INF, POS_INF],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => {
            // Neither var is substituted; the equality row survives.
            assert_eq!(ps.reduced.m_eq(), 1, "bounded var must keep its row");
        }
        other => panic!("expected Reduced, got {:?}", status_of(&other)),
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    // Degenerate vertex (bound and constraint both active), so the IPM
    // converges to looser KKT tolerance — the point of this test is the
    // *non*-substitution above, not solver precision.
    assert_kkt(&prob, &sol, 1e-3);
}

// --- presolve statistics ---

/// `Presolve::stats()` reports the reduction sizes and counts by type.
#[test]
fn presolve_stats_report() {
    // x2 (free singleton) is substituted out → removes a var and a row;
    // x3 (free, zero cost) is dropped as a free column.
    let prob = QpProblem {
        n: 4,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![0.0, 0.0, 0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0), // x2 free singleton in this row
        ],
        b: vec![3.0],
        g: vec![],
        h: vec![],
        lb: vec![NEG_INF, NEG_INF, NEG_INF, NEG_INF],
        ub: vec![POS_INF, POS_INF, POS_INF, POS_INF],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => {
            let s = ps.stats();
            assert!(s.reduced_anything());
            assert_eq!(s.orig_vars, 4);
            assert_eq!(s.orig_rows, 1);
            // x2 substituted (removes var+row), x3 dropped as free column.
            assert_eq!(s.free_col_singletons, 1, "stats={s:?}");
            assert_eq!(s.free_cols_fixed, 1, "stats={s:?}");
            assert_eq!(s.reduced_rows, 0, "the row is consumed; stats={s:?}");
            assert_eq!(s.reduced_vars, 2, "x2,x3 removed; stats={s:?}");
        }
        other => panic!("expected Reduced, got {:?}", status_of(&other)),
    }
}

/// A no-op presolve reports `reduced_anything() == false`.
#[test]
fn presolve_stats_noop() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![1.0],
        lb: vec![0.0, 0.0],
        ub: vec![10.0, 10.0],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => {
            let s = ps.stats();
            assert!(!s.reduced_anything(), "stats={s:?}");
            assert_eq!(s.reduced_vars, s.orig_vars);
            assert_eq!(s.reduced_rows, s.orig_rows);
        }
        other => panic!("expected Reduced, got {:?}", status_of(&other)),
    }
}
