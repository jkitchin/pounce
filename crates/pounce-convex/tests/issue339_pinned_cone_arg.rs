//! gh #339 regression: the non-symmetric (exp) HSDE driver reported
//! `NumericalFailure` — with a badly wrong objective/iterate, not merely a
//! mislabeled correct one — on a well-posed, essentially trivial exponential
//! cone problem where one cone-triple component is pinned to a large-
//! magnitude value (via an equality constraint elsewhere in the problem)
//! while its companion slack should land near `0`.
//!
//! Root cause: `max_step`'s backtracking line search (the fraction-to-
//! boundary step for the exp/power blocks, which have no closed-form
//! boundary root) tested cone membership with a **fixed absolute** tolerance
//! (`1e-12`). A non-symmetric cone coordinate that legitimately tracks the
//! barrier parameter `μ` down the central path — here the dual coordinate
//! conjugate to the pinned primal argument, driven to `0` as `μ → 0` because
//! the cone-membership slack `ψ` for that triple is comfortably non-tight —
//! shrinks *with* `μ`. Once its magnitude nears the fixed `1e-12` floor, the
//! backtracking started rejecting any further legitimate shrinkage as if the
//! iterate were leaving the cone, collapsing the step length geometrically
//! iteration over iteration until it hit exactly `0` well short of
//! convergence, stranding the iterate on a stalled, incorrect value — not
//! just a mislabeling of an already-correct point (distinct from gh #336's
//! post-loop-adjudication fix). The fix scales that floor by `μ` (capped at
//! the legacy `1e-12` constant, so a well-scaled solve is unaffected):
//! `nscone_mem_tol` in `hsde_nonsym.rs`.
//!
//! This is deliberately a *different* construction than
//! `issue336_scale_status.rs`: #336 scaled a GP's overall objective/cone
//! magnitude (all three coordinates large together); here a single variable
//! is pinned to a large magnitude by an *equality constraint elsewhere in the
//! problem* while its cone-triple companions must land near `0` — an
//! intra-cone magnitude mismatch, not an overall-scale one.

use pounce_convex::{ConeSpec, QpOptions, QpProblem, QpStatus, Triplet, solve_socp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `min t s.t. u = u_val (pinned), t >= exp(scale * u)`, i.e. vars `[u, t]`,
/// one exponential cone `(s0, s1, s2) = (scale*u, 1, t)`. The true optimum is
/// `t* = exp(scale*u_val)`, which underflows to exactly `0.0` in double
/// precision once `scale*u_val < -745` or so — attained, not merely
/// approached, so the correct objective is (numerically) `0`.
fn pinned_exp_problem(scale: f64, u_val: f64) -> QpProblem {
    let g = vec![Triplet::new(0, 0, -scale), Triplet::new(2, 1, -1.0)];
    let h = vec![0.0, 1.0, 0.0];
    let a = vec![Triplet::new(0, 0, 1.0)];
    let b = vec![u_val];
    QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![0.0, 1.0],
        a,
        b,
        g,
        h,
        lb: vec![],
        ub: vec![],
    }
}

fn solve(scale: f64, u_val: f64) -> pounce_convex::QpSolution {
    let prob = pinned_exp_problem(scale, u_val);
    let opts = QpOptions::default();
    solve_socp_ipm(&prob, &[ConeSpec::Exponential], &opts, backend)
}

/// The smallest reproducer from gh #339: `scale=1e3` already triggered
/// `NumericalFailure` with a badly wrong objective (`~8e-5` instead of `~0`)
/// on `main` at the time of the report.
#[test]
fn exp_pinned_arg_scale_1e3_not_numerical_failure() {
    let sol = solve(1000.0, -50.0);
    assert_ne!(
        sol.status,
        QpStatus::NumericalFailure,
        "a correct answer must not be labelled NumericalFailure"
    );
    assert!(
        matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate),
        "expected Optimal/OptimalInaccurate, got {:?}",
        sol.status
    );
    assert!(
        sol.obj.abs() < 1e-4,
        "objective {:.6e} must be ~0 (true optimum underflows to 0)",
        sol.obj
    );
    assert!(
        (sol.x[0] - (-50.0)).abs() < 1e-4,
        "u must stay pinned at -50, got {}",
        sol.x[0]
    );
}

/// Scale sweep at fixed `u = -50`, matching the issue's first table. Every
/// scale must produce a usable status and a near-zero objective — never the
/// wildly wrong, non-monotonic objectives reported in the issue
/// (`8e-5 -> 0.02 -> 1.9 -> 131.8` as `scale` grew).
#[test]
fn exp_pinned_arg_scale_sweep_never_numerical_failure() {
    for &scale in &[1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6] {
        let sol = solve(scale, -50.0);
        assert_ne!(
            sol.status,
            QpStatus::NumericalFailure,
            "scale={scale}: a correct answer must not be labelled NumericalFailure"
        );
        assert!(
            matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate),
            "scale={scale}: expected Optimal/OptimalInaccurate, got {:?}",
            sol.status
        );
        assert!(
            sol.obj.abs() < 1e-3,
            "scale={scale}: objective {:.6e} must be ~0",
            sol.obj
        );
        assert!(
            (sol.x[0] - (-50.0)).abs() < 1e-3,
            "scale={scale}: u must stay pinned at -50, got {}",
            sol.x[0]
        );
    }
}

/// `u` sweep at fixed `scale = 1e6`, matching the issue's second table: even
/// a mild `u = -1` (`scale*u = -1e6`, deep underflow) reportedly failed.
#[test]
fn exp_pinned_arg_u_sweep_never_numerical_failure() {
    for &u_val in &[-1.0, -5.0, -10.0, -20.0, -30.0, -40.0, -50.0] {
        let sol = solve(1e6, u_val);
        assert_ne!(
            sol.status,
            QpStatus::NumericalFailure,
            "u={u_val}: a correct answer must not be labelled NumericalFailure"
        );
        assert!(
            matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate),
            "u={u_val}: expected Optimal/OptimalInaccurate, got {:?}",
            sol.status
        );
        assert!(
            sol.obj.abs() < 1e-3,
            "u={u_val}: objective {:.6e} must be ~0",
            sol.obj
        );
        assert!(
            (sol.x[0] - u_val).abs() < (1e-3_f64).max(1e-6 * u_val.abs()),
            "u={u_val}: pinned variable must stay at {u_val}, got {}",
            sol.x[0]
        );
    }
}

/// Well-scaled end (`scale=1`) must still certify a clean `Optimal` — guards
/// against over-relaxing the interior-membership floor into a general
/// accuracy regression for ordinary, non-extreme problems.
#[test]
fn exp_pinned_arg_well_scaled_is_optimal() {
    let sol = solve(1.0, -5.0);
    assert_eq!(sol.status, QpStatus::Optimal);
    let f_star = (-5.0_f64).exp();
    assert!(
        (sol.obj - f_star).abs() < 1e-6,
        "obj {:.6e} must match f* = {:.6e}",
        sol.obj,
        f_star
    );
}
