//! Tests for the explicit variable-bound form: `lb ≤ x ≤ ub` as
//! first-class fields on `QpProblem`, solved by bound expansion in the
//! IPM with the bound multipliers reported in `z_lb` / `z_ub`.
//!
//! Each test cross-checks the bounded form against the equivalent
//! G-row encoding so the two representations agree, and checks the
//! KKT stationarity that includes the bound duals.

use pounce_convex::presolve::solve_with_presolve;
use pounce_convex::{NEG_INF, POS_INF, QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve(prob: &QpProblem) -> pounce_convex::QpSolution {
    solve_qp_ipm(prob, &QpOptions::default(), backend)
}

/// Stationarity with bound duals: Px + c + Aᵀy + Gᵀz − z_lb + z_ub = 0.
fn assert_stationarity(prob: &QpProblem, sol: &pounce_convex::QpSolution, tol: f64) {
    let mut g = prob.c.clone();
    prob.p_mul(&sol.x, &mut g);
    prob.at_mul(&sol.y, &mut g);
    prob.gt_mul(&sol.z, &mut g);
    for i in 0..prob.n {
        g[i] -= sol.z_lb[i];
        g[i] += sol.z_ub[i];
    }
    for (i, gi) in g.iter().enumerate() {
        assert!(gi.abs() < tol, "stationarity[{i}] = {gi}");
    }
}

/// Upper bound binds: min ½(x0−3)²+(x1−4)² with x ≤ (1, +∞).
/// Optimum x0 = 1 (bound active), x1 = 4 (interior). f* = −10.5.
#[test]
fn upper_bound_binds() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-3.0, -4.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![NEG_INF, NEG_INF],
        ub: vec![1.0, POS_INF],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 4.0).abs() < 1e-6, "x1={}", sol.x[1]);
    assert!((sol.obj - (-10.5)).abs() < 1e-6, "obj={}", sol.obj);
    // Upper bound on x0 is active with a positive multiplier (= 2).
    assert!(sol.z_ub[0] > 1.0, "z_ub[0]={}", sol.z_ub[0]);
    assert!(sol.z_lb[0].abs() < 1e-5, "z_lb[0]={}", sol.z_lb[0]);
    assert_stationarity(&prob, &sol, 1e-5);
}

/// Lower bound binds: min ½(x0+3)² with x0 ≥ 0. Optimum x0 = 0.
#[test]
fn lower_bound_binds() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 1.0)],
        c: vec![3.0], // unconstrained optimum at −3
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0],
        ub: vec![POS_INF],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(sol.x[0].abs() < 1e-6, "x0={}", sol.x[0]);
    assert!(sol.z_lb[0] > 1.0, "z_lb[0]={}", sol.z_lb[0]);
    assert_stationarity(&prob, &sol, 1e-5);
}

/// gh #295 (defense-in-depth for #275): a box that admits **no finite point**
/// — a *present* `+∞` lower bound or a *present* `−∞` upper bound — must be
/// certified `PrimalInfeasible` by the core itself, on the raw solve path
/// (bypassing the Python-layer `_validate` guard). Before this fix
/// `expand_bounds` was sign-agnostic and silently dropped such a bound as
/// "absent", returning `Optimal` at a violating point.
#[test]
fn present_infinite_lower_bound_is_primal_infeasible() {
    // min ½x²  s.t.  x ≥ +∞ : impossible.
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 1.0)],
        c: vec![0.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![POS_INF],
        ub: vec![POS_INF],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::PrimalInfeasible);
}

#[test]
fn present_infinite_upper_bound_is_primal_infeasible() {
    // min ½x²  s.t.  x ≤ −∞ : impossible.
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 1.0)],
        c: vec![0.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![NEG_INF],
        ub: vec![NEG_INF],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::PrimalInfeasible);
}

/// The impossible-bound screen must NOT regress the *absent* one-sided `±∞`
/// encoding: `lb = −∞` / `ub = +∞` is the normal way to say "unbounded on
/// that side" and must still solve. `min ½(x+3)²` with `x ∈ (−∞, +∞)` → −3.
#[test]
fn absent_infinite_bounds_still_solve() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 1.0)],
        c: vec![3.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![NEG_INF],
        ub: vec![POS_INF],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - (-3.0)).abs() < 1e-6, "x0={}", sol.x[0]);
}

/// A finite reversed box (`lb > ub`) is the same infeasible class and must
/// still map to `PrimalInfeasible` (regression guard for the pre-existing
/// behavior the #295 screen sits beside).
#[test]
fn finite_reversed_box_is_primal_infeasible() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 1.0)],
        c: vec![0.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![5.0],
        ub: vec![3.0],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::PrimalInfeasible);
}

/// A fixed variable (`lb == ub`) is feasible and must still solve to that
/// value — the screen must not over-reject the degenerate-but-valid box.
#[test]
fn fixed_variable_still_solves() {
    // min ½(x−10)²  s.t.  x = 2  → x* = 2.
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 1.0)],
        c: vec![-10.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![2.0],
        ub: vec![2.0],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 2.0).abs() < 1e-6, "x0={}", sol.x[0]);
}

/// Presolve is the other core entry that must certify an impossible bound
/// infeasible (the fixpoint's first pass), so a presolve-then-solve caller
/// inherits the same rejection.
#[test]
fn presolve_rejects_present_infinite_bound() {
    use pounce_convex::presolve::{PresolveOutcome, presolve};
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 1.0)],
        c: vec![0.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![POS_INF],
        ub: vec![POS_INF],
    };
    assert!(matches!(presolve(&prob), PresolveOutcome::Infeasible));
}

/// Box-constrained LP: min −x0 − x1 with 0 ≤ x ≤ 1. Optimum (1, 1).
#[test]
fn box_constrained_lp() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![-1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![1.0, 1.0],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);
    assert!((sol.obj - (-2.0)).abs() < 1e-6, "obj={}", sol.obj);
    assert_stationarity(&prob, &sol, 1e-5);
}

/// The bounded form must agree with the equivalent G-row encoding.
#[test]
fn bounded_form_matches_g_row_encoding() {
    // min ½‖x‖² + cᵀx, 0 ≤ x ≤ 2.
    let bounded = QpProblem {
        n: 3,
        p_lower: vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 2.0),
        ],
        c: vec![-5.0, 1.0, -0.5],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0, 0.0],
        ub: vec![2.0, 2.0, 2.0],
    };
    // Same problem with bounds written as 2n G rows.
    let mut g = Vec::new();
    let mut h = Vec::new();
    for i in 0..3 {
        g.push(Triplet::new(2 * i, i, 1.0)); // x_i ≤ 2
        h.push(2.0);
        g.push(Triplet::new(2 * i + 1, i, -1.0)); // −x_i ≤ 0
        h.push(0.0);
    }
    let g_form = QpProblem {
        n: 3,
        p_lower: bounded.p_lower.clone(),
        c: bounded.c.clone(),
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![],
        ub: vec![],
    };

    let sb = solve(&bounded);
    let sg = solve(&g_form);
    assert_eq!(sb.status, QpStatus::Optimal);
    assert_eq!(sg.status, QpStatus::Optimal);
    for i in 0..3 {
        assert!(
            (sb.x[i] - sg.x[i]).abs() < 1e-5,
            "x[{i}]: bounded {} vs G-row {}",
            sb.x[i],
            sg.x[i]
        );
    }
    assert!(
        (sb.obj - sg.obj).abs() < 1e-5,
        "obj {} vs {}",
        sb.obj,
        sg.obj
    );
}

/// Presolve respects bounds: a singleton equality that fixes a variable
/// outside its box is infeasible.
#[test]
fn presolve_singleton_fix_violates_bound() {
    // x0 = 5 but x0 ≤ 1 → infeasible.
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0],
        a: vec![Triplet::new(0, 0, 1.0)],
        b: vec![5.0],
        g: vec![],
        h: vec![],
        lb: vec![NEG_INF],
        ub: vec![1.0],
    };
    let sol = solve_with_presolve(&prob, |r| solve_qp_ipm(r, &QpOptions::default(), backend));
    assert_eq!(sol.status, QpStatus::PrimalInfeasible);
}

/// Presolve free-column at a bound: a linear-only variable with positive
/// cost is pushed to its lower bound, and the rest solves normally.
#[test]
fn presolve_free_column_to_lower_bound() {
    // min x0² + x1 (x1 linear-only, c=+1 → pushed to lb) s.t. x0 = 2,
    // with x1 ∈ [3, 10]. Expect x1 = 3.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0, 1.0],
        a: vec![Triplet::new(0, 0, 1.0)], // x0 = 2
        b: vec![2.0],
        g: vec![],
        h: vec![],
        lb: vec![NEG_INF, 3.0],
        ub: vec![POS_INF, 10.0],
    };
    let sol = solve_with_presolve(&prob, |r| solve_qp_ipm(r, &QpOptions::default(), backend));
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 2.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 3.0).abs() < 1e-6, "x1={}", sol.x[1]);
}
