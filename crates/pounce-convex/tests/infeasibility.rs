//! Verified infeasibility / unboundedness detection (the HSDE benefit:
//! clean status instead of exhausting the iteration budget).
//!
//! Each declared status is backed by a checked certificate, so these
//! tests also implicitly confirm there are no false positives — the
//! feasible/optimal problems in the rest of the suite must still report
//! `Optimal`, and a couple of those are re-checked here for contrast.

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
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
        lb: vec![],
        ub: vec![],
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
        lb: vec![],
        ub: vec![],
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
        lb: vec![],
        ub: vec![],
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
        lb: vec![],
        ub: vec![],
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
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-6);
    assert!((sol.x[1] - 1.0).abs() < 1e-6);
}

/// gh #293 — a mixed-scale Hessian must NOT be falsely certified unbounded.
/// `min ½(1e6·x0² + 1e-12·x1²) − x1  s.t.  x ≥ 0` is *bounded*: the unique
/// optimum is `x1* = 1e12`, `f* = −5e11`. The descent ray `x1` has genuine
/// (if tiny) curvature `1e-12 > 0`, so it is not a recession ray. Before #293
/// the `‖Pd‖ ≤ rtol·‖d‖·max|P|` test read the `1e-12` curvature as null
/// relative to the `1e6` block and returned a wrong `DualInfeasible`.
#[test]
fn mixed_scale_hessian_is_bounded_not_dual_infeasible() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1e6), Triplet::new(1, 1, 1e-12)],
        c: vec![0.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY, f64::INFINITY],
    };
    let sol = solve(&prob);
    assert_ne!(
        sol.status,
        QpStatus::DualInfeasible,
        "bounded problem (f* = -5e11) must never get an unboundedness \
         certificate; got a wrong DualInfeasible after {} iters",
        sol.iters
    );
    // The certificate fix also lets it converge to the true optimum.
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "expected Optimal (x1* = 1e12, f* = -5e11), got {:?} after {} iters",
        sol.status,
        sol.iters
    );
    assert!(
        (sol.obj - (-5e11)).abs() <= 1e-3 * 5e11,
        "obj = {} should be ≈ -5e11",
        sol.obj
    );
}

/// gh #293 (symptom 2) — a *uniformly* tiny Hessian must converge, not exhaust
/// the iteration budget. `min ½·1e-12·(x0² + x1²) − x1  s.t.  x ≥ 0` is bounded
/// with the same optimum as above (`x1* = 1e12`, `f* = −5e11`). #290 stopped
/// this from being falsely certified unbounded, but the default HSDE driver
/// then merely ran out of iterations (obj ≈ −4.95e11 at `IterationLimit`)
/// because its per-cone NT scaling never sees the 12-orders-below-O(1)
/// curvature. The fix Ruiz-equilibrates and retries when HSDE hits the cap, so
/// the solve now reports `Optimal` at the true optimum.
#[test]
fn uniform_tiny_hessian_converges_not_iteration_limit() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1e-12), Triplet::new(1, 1, 1e-12)],
        c: vec![0.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY, f64::INFINITY],
    };
    let sol = solve(&prob);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "expected Optimal (x1* = 1e12, f* = -5e11), got {:?} after {} iters",
        sol.status,
        sol.iters
    );
    assert!(
        (sol.obj - (-5e11)).abs() <= 1e-3 * 5e11,
        "obj = {} should be ≈ -5e11",
        sol.obj
    );
}

// --- Status / edge-case honesty (PR70 item C) -----------------------------
//
// A solver that stops early for *any* reason must say so. The danger these
// guard against is a confident `Optimal` (or a spurious infeasible/unbounded)
// on a problem the solver did not actually finish or that is degenerate.

/// Iteration-limit honesty: a real, feasible, bounded QP that needs several
/// IPM iterations must report `IterationLimit` — never a premature `Optimal`,
/// and never a false infeasible/unbounded — when starved of iterations.
#[test]
fn iteration_limit_reported_not_optimal() {
    // The same well-posed box QP as `feasible_bounded_still_optimal`, which
    // converges in several iterations at the default cap. With max_iter = 1 it
    // cannot have converged, so the only honest status is IterationLimit.
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
        lb: vec![],
        ub: vec![],
    };
    let opts = QpOptions {
        max_iter: 1,
        ..QpOptions::default()
    };
    let sol = solve_qp_ipm(&prob, &opts, backend);
    assert_eq!(
        sol.status,
        QpStatus::IterationLimit,
        "1-iteration solve must report IterationLimit, got {:?}",
        sol.status
    );
    assert_ne!(
        sol.status,
        QpStatus::Optimal,
        "must not claim Optimal after a single iteration"
    );
}

/// Degenerate input — a variable fixed by equal bounds (lb == ub) — must
/// solve honestly to `Optimal` at the fixed value, not trip a spurious
/// infeasible/unbounded or numerical failure.
#[test]
fn fixed_variable_equal_bounds_optimal() {
    // min x0² + x1² − 6x0 − 6x1, x0 fixed to 1 (lb==ub==1), x1 ∈ [0, 10].
    // Unconstrained min is (3, 3); with x0 pinned the optimum is (1, 3).
    // obj = 1 + 9 − 6 − 18 = −14.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-6.0, -6.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![1.0, 0.0],
        ub: vec![1.0, 10.0],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 3.0).abs() < 1e-6, "x1={}", sol.x[1]);
    assert!((sol.obj - (-14.0)).abs() < 1e-6, "obj={}", sol.obj);
}

/// Edge input — a fully unconstrained QP (no equalities, no inequalities, no
/// bounds) — must still solve to its stationary point and report `Optimal`.
#[test]
fn unconstrained_qp_optimal() {
    // min x0² + x1² − 6x0 + 4x1  ->  min at (3, −2), obj = 9 + 4 − 18 − 8 = −13.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-6.0, 4.0],
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
    assert!((sol.x[1] - (-2.0)).abs() < 1e-6, "x1={}", sol.x[1]);
    assert!((sol.obj - (-13.0)).abs() < 1e-6, "obj={}", sol.obj);
}
