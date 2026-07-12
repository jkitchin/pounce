//! Presolve round-trip exactness (the Phase 3.5 correctness contract):
//! solving with presolve must reproduce the no-presolve `(x, y, z)` to
//! tolerance — primal *and* dual. Also covers presolve-detected
//! infeasibility.
//!
//! Tolerance note: each assertion compares *two independent* IPM solves
//! (direct vs presolved), so the bar is the solvers' own convergence
//! tolerance, not exact equality. We use 1e-5.

use pounce_convex::presolve::{PresolveOutcome, presolve, solve_with_presolve};
use pounce_convex::{NEG_INF, POS_INF, QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

const TOL: f64 = 1e-5;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn direct(prob: &QpProblem) -> pounce_convex::QpSolution {
    solve_qp_ipm(prob, &QpOptions::default(), backend)
}

fn with_presolve(prob: &QpProblem) -> pounce_convex::QpSolution {
    solve_with_presolve(prob, |reduced| {
        solve_qp_ipm(reduced, &QpOptions::default(), backend)
    })
}

fn assert_close(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < TOL, "{what}[{i}]: {x} vs {y}");
    }
}

/// Fixed-variable elimination: `min x0²+x1²+x2² s.t. x0+x1+x2=3, x2=2`.
/// The singleton row `x2=2` fixes x2; presolve substitutes it out.
#[test]
fn fixed_variable_roundtrip_matches_direct() {
    let prob = QpProblem {
        n: 3,
        p_lower: vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 2.0),
        ],
        c: vec![0.0, 0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0),
            Triplet::new(1, 2, 1.0), // singleton → fixes x2 = 2
        ],
        b: vec![3.0, 2.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let d = direct(&prob);
    let p = with_presolve(&prob);
    assert_eq!(d.status, QpStatus::Optimal);
    assert_eq!(p.status, QpStatus::Optimal);
    assert_close(&p.x, &d.x, "x");
    assert_close(&p.y, &d.y, "y");
    assert!((p.obj - d.obj).abs() < TOL, "obj {} vs {}", p.obj, d.obj);
    assert!((p.x[2] - 2.0).abs() < 1e-9, "x2={}", p.x[2]);
}

/// Fixed variable coupling through an off-diagonal Hessian term, so the
/// substitution must move `P` coupling into the linear term:
/// `min x0² + x0 x1 + x1² s.t. x1 = 1`.
#[test]
fn fixed_variable_with_hessian_coupling_roundtrip() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 0, 1.0), // x0 x1 coupling
            Triplet::new(1, 1, 2.0),
        ],
        c: vec![0.0, 0.0],
        a: vec![Triplet::new(0, 1, 1.0)], // x1 = 1
        b: vec![1.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let d = direct(&prob);
    let p = with_presolve(&prob);
    assert_eq!(p.status, QpStatus::Optimal);
    assert_close(&p.x, &d.x, "x");
    assert_close(&p.y, &d.y, "y");
    assert!((p.obj - d.obj).abs() < TOL, "obj {} vs {}", p.obj, d.obj);
}

/// Fixed variable plus an inequality whose RHS must be adjusted by the
/// substitution: `min x0²-6x0 s.t. x1=1, x0+x1 ≤ 3`. After fixing x1=1
/// the inequality becomes `x0 ≤ 2`, which binds (unconstrained x0=3).
#[test]
fn fixed_variable_adjusts_inequality_rhs() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-6.0, 0.0],
        a: vec![Triplet::new(0, 1, 1.0)],
        b: vec![1.0],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)], // x0+x1≤3
        h: vec![3.0],
        lb: vec![],
        ub: vec![],
    };
    let d = direct(&prob);
    let p = with_presolve(&prob);
    assert_eq!(p.status, QpStatus::Optimal);
    assert_close(&p.x, &d.x, "x");
    assert_close(&p.y, &d.y, "y");
    assert_close(&p.z, &d.z, "z");
    assert!((p.obj - d.obj).abs() < TOL, "obj {} vs {}", p.obj, d.obj);
    // The inequality binds with a clearly nonzero multiplier (~2).
    assert!(p.z[0] > 1.0, "inequality should bind, z={}", p.z[0]);
}

/// Empty-row removal must not change the solution and the empty row's
/// dual is 0. (Non-degenerate: the kept constraint is a strict equality.)
#[test]
fn empty_row_roundtrip() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 0.0), // empty row, b=0 → feasible, dropped
            Triplet::new(1, 0, 1.0), // x0 + x1 = 2
            Triplet::new(1, 1, 1.0),
        ],
        b: vec![0.0, 2.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let d = direct(&prob);
    let p = with_presolve(&prob);
    assert_eq!(p.status, QpStatus::Optimal);
    assert_close(&p.x, &d.x, "x");
    assert!(p.y[0].abs() < 1e-9, "empty-row dual={}", p.y[0]);
}

/// Presolve detects trivial primal infeasibility from `0 = 5`.
#[test]
fn empty_row_infeasible_detected() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0],
        a: vec![Triplet::new(0, 0, 0.0)], // 0·x0 = 5
        b: vec![5.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    assert!(matches!(presolve(&prob), PresolveOutcome::Infeasible));
    assert_eq!(with_presolve(&prob).status, QpStatus::PrimalInfeasible);
}

/// Full-KKT check on the *original* problem, carrying every recovered dual
/// (equality `y`, inequality `z`, and bound multipliers `z_lb`/`z_ub`). If
/// postsolve mis-reconstructed any dual on a heavily-reduced problem, the
/// stationarity residual would not vanish — so this validates the *whole*
/// recovered solution, not just the primal.
fn assert_original_kkt(prob: &QpProblem, sol: &pounce_convex::QpSolution, tol: f64) {
    let n = prob.n;
    let mut g = prob.c.clone();
    prob.p_mul(&sol.x, &mut g);
    prob.at_mul(&sol.y, &mut g);
    prob.gt_mul(&sol.z, &mut g);
    for i in 0..n {
        // Stationarity with bound multipliers: ∇L + z_ub − z_lb = 0.
        let stat = g[i] + sol.z_ub[i] - sol.z_lb[i];
        assert!(stat.abs() < tol, "stationarity[{i}] = {stat}");
        assert!(
            sol.z_lb[i] > -tol && sol.z_ub[i] > -tol,
            "bound dual sign [{i}]: z_lb={} z_ub={}",
            sol.z_lb[i],
            sol.z_ub[i]
        );
        assert!(
            sol.x[i] >= prob.lb_of(i) - tol && sol.x[i] <= prob.ub_of(i) + tol,
            "box [{i}]: {} not in [{}, {}]",
            sol.x[i],
            prob.lb_of(i),
            prob.ub_of(i)
        );
        // Complementarity only applies to finite bounds (an infinite bound can
        // never be active, and `0 · ∞` would be NaN).
        if prob.lb_of(i).is_finite() {
            assert!(
                (sol.z_lb[i] * (sol.x[i] - prob.lb_of(i))).abs() < 1e-4,
                "lb complementarity [{i}]"
            );
        }
        if prob.ub_of(i).is_finite() {
            assert!(
                (sol.z_ub[i] * (prob.ub_of(i) - sol.x[i])).abs() < 1e-4,
                "ub complementarity [{i}]"
            );
        }
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
        assert!(sol.z[i] > -tol, "z[{i}] = {} < 0", sol.z[i]);
        assert!(
            (sol.z[i] * slack).abs() < 1e-4,
            "ineq complementarity row {i}: z={} slack={slack}",
            sol.z[i]
        );
    }
}

/// Heavily-reduced problem: a single QP that fires *four distinct* reductions
/// at once — a fixed variable (equality singleton), a free-column singleton
/// (substituted out), a dominated column (fixed to a bound), and a binding
/// inequality — collapsing 6 variables / 2 equalities to a tiny core. Presolve
/// + postsolve must recover the full primal AND dual (equality `y`, inequality
/// `z`, bound `z_lb`/`z_ub`), matching a direct no-presolve solve and the
/// original problem's KKT system.
#[test]
fn heavily_reduced_mixed_reductions_recovers_primal_and_dual() {
    // vars: x0,x1,x2 (in P, solved by the IPM); x3 fixed by `x3 = 1`;
    //       x4 free singleton in `x0+x1+x4 = 4` (substituted); x5 dominated
    //       (only in the ≤ row with +1, cost ≥ 0, box [0,5]) → fixed to 0.
    // The inequality x0 + x2 + x5 ≤ 3 binds at the optimum (nonzero z).
    let prob = QpProblem {
        n: 6,
        p_lower: vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 2.0),
        ],
        //        x0    x1    x2    x3    x4   x5
        c: vec![-8.0, -2.0, -4.0, -3.0, 0.0, 0.5],
        a: vec![
            Triplet::new(0, 3, 1.0), // x3 = 1            (fixed variable)
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(1, 4, 1.0), // x0+x1+x4 = 4      (x4 free singleton)
        ],
        b: vec![1.0, 4.0],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 2, 1.0),
            Triplet::new(0, 5, 1.0), // x0+x2+x5 ≤ 3      (x5 dominated)
        ],
        h: vec![3.0],
        lb: vec![0.0, 0.0, 0.0, 0.0, NEG_INF, 0.0],
        ub: vec![5.0, 5.0, 5.0, 5.0, POS_INF, 5.0],
    };

    // Presolve must fire all three structural reductions and shrink the core.
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => {
            let s = ps.stats();
            assert!(s.fixed_vars >= 1, "expected a fixed var, stats={s:?}");
            assert!(
                s.free_col_singletons >= 1,
                "expected a free-column singleton, stats={s:?}"
            );
            assert!(
                s.dominated_cols >= 1,
                "expected a dominated column, stats={s:?}"
            );
            assert!(
                ps.reduced.n <= 3,
                "core should collapse to ≤3 vars, got {}",
                ps.reduced.n
            );
        }
        PresolveOutcome::Infeasible => panic!("expected Reduced, got Infeasible"),
        PresolveOutcome::Unbounded => panic!("expected Reduced, got Unbounded"),
    }

    let d = direct(&prob);
    let p = with_presolve(&prob);
    assert_eq!(d.status, QpStatus::Optimal);
    assert_eq!(p.status, QpStatus::Optimal);

    // Full primal recovery (all six original variables, incl. substituted x4
    // and the fixed/dominated x3,x5).
    assert_close(&p.x, &d.x, "x");
    assert!((p.obj - d.obj).abs() < TOL, "obj {} vs {}", p.obj, d.obj);
    assert!((p.x[3] - 1.0).abs() < 1e-9, "x3 fixed: {}", p.x[3]);
    assert!(p.x[5].abs() < 1e-6, "x5 dominated to 0: {}", p.x[5]);

    // Full dual recovery: equality multipliers, inequality multiplier, and the
    // bound multipliers all match the direct solve…
    assert_close(&p.y, &d.y, "y");
    assert_close(&p.z, &d.z, "z");
    assert_close(&p.z_lb, &d.z_lb, "z_lb");
    assert_close(&p.z_ub, &d.z_ub, "z_ub");
    // …and the recovered (x, y, z, z_lb, z_ub) is a KKT point of the ORIGINAL.
    assert_original_kkt(&prob, &p, 1e-5);
    // The inequality genuinely binds (a nonzero recovered multiplier).
    assert!(p.z[0] > 1e-3, "inequality should bind, z={}", p.z[0]);
    // The dominated column's bound multiplier is recovered nonzero.
    assert!(
        p.z_lb[5] > 1e-3,
        "dominated-column bound dual should be nonzero, z_lb[5]={}",
        p.z_lb[5]
    );
}

/// Nothing to presolve → identity round-trip. Non-degenerate: the bound
/// that binds (x0 ≤ 1, with unconstrained optimum x0 = 3) has a clearly
/// nonzero multiplier, so the two solves agree well within tolerance.
#[test]
fn noop_presolve_roundtrip() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-6.0, -4.0], // unconstrained opt (3, 2)
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),  // x0 ≤ 1 (binds, mult ~4)
            Triplet::new(1, 1, 1.0),  // x1 ≤ 5 (inactive)
            Triplet::new(2, 0, -1.0), // x0 ≥ 0
            Triplet::new(3, 1, -1.0), // x1 ≥ 0
        ],
        h: vec![1.0, 5.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let d = direct(&prob);
    let p = with_presolve(&prob);
    assert_close(&p.x, &d.x, "x");
    assert_close(&p.z, &d.z, "z");
}
