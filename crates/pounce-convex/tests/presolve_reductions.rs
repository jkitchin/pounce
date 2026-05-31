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
