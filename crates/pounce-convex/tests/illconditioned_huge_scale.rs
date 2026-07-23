//! Regression for gh #286: a convex QP with an enormous objective magnitude
//! must terminate `Optimal` at the true optimum within the *default* iteration
//! budget, not grind to the cap at a box-violating interior iterate.
//!
//! Root cause: the default HSDE driver skips Ruiz equilibration and relies on
//! its per-cone NT scaling, which conditions the constraint system but does
//! nothing about the sheer *magnitude* of the objective data. With
//! `‖P‖, ‖c‖ ~ 1e18‥1e22` the homogeneous embedding's `τ` collapsed toward the
//! `τ → 0` certificate boundary and primal feasibility crawled: the solve
//! exhausted its budget at a point violating the box by ~0.88 even though the
//! dual and gap had converged in a few dozen steps (and even at
//! `max_iter = 5000` it only reached the optimum by brute force). Normalizing
//! the objective by a scalar `σ = max(‖P‖∞, ‖c‖∞)` — which leaves the minimizer
//! unchanged — restores an `O(1)` objective and the embedding converges in a
//! handful of iterations, the cost scaling Clarabel/OSQP apply as a matter of
//! course.
//!
//! The constructions are diagonal, so each box QP separates per coordinate and
//! the exact optimum is the unconstrained target clamped to the box — an oracle
//! independent of any solver.

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Diagonal box QP `min ½ xᵀP x + cᵀx s.t. -1 ≤ x ≤ 1` with
/// `P = diag(eig)`, `eig_i = scale · cond^(i/(n-1))` and `c_i = -eig_i·tgt_i`.
/// Because `P` is diagonal the objective separates and the exact minimizer is
/// `x*_i = clamp(tgt_i, -1, 1)`.
fn diagonal_box_qp(scale: f64, cond: f64, tgt: &[f64]) -> (QpProblem, Vec<f64>) {
    let n = tgt.len();
    let mut p_lower = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);
    let mut xopt = Vec::with_capacity(n);
    for (i, &t) in tgt.iter().enumerate() {
        let eig = scale * cond.powf(i as f64 / (n - 1) as f64);
        p_lower.push(Triplet::new(i, i, eig));
        c.push(-eig * t);
        xopt.push(t.clamp(-1.0, 1.0));
    }
    let prob = QpProblem {
        n,
        p_lower,
        c,
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![-1.0; n],
        ub: vec![1.0; n],
    };
    (prob, xopt)
}

fn objective(prob: &QpProblem, x: &[f64]) -> f64 {
    let mut px = vec![0.0; prob.n];
    prob.p_mul_add_pub(x, &mut px);
    (0..prob.n)
        .map(|i| 0.5 * x[i] * px[i] + prob.c[i] * x[i])
        .sum()
}

/// Regression for gh #324: a huge Hessian coefficient paired with a *modest*
/// gradient (`‖c‖ ≪ ‖P‖`) must not falsely certify `Optimal` at the cold start.
///
/// `min ½ xᵀP x + cᵀx` with `P = diag(a, b)`, `c = [-1, -1]`, no constraints —
/// unique optimum `x* = [1/a, 1/b]`, `f* = -½(1/a + 1/b)`. The cost
/// normalization (gh #286) divides the objective by `σ ~ max(‖P‖, ‖c‖) ~ ‖P‖`;
/// once `‖P‖ ≳ 1/ε` the scaled cold-start dual residual `‖c/σ‖` underflows below
/// `tol` and the embedding used to report `Optimal` at the untouched start
/// `x = 0` (objective 0 vs the true `-½(1/a+1/b)`), its own `kkt_error` sitting
/// at `‖c‖ = 1`. The relative-KKT re-check now rejects that certificate and the
/// un-normalized solve recovers the true optimum.
#[test]
fn issue_324_huge_hessian_modest_gradient_is_not_false_optimal_at_cold_start() {
    for (a, b) in [(1e-8, 1e8), (1e-10, 1e10), (1e8, 1e8), (1e10, 1e10)] {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, a), Triplet::new(1, 1, b)],
            c: vec![-1.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(
            sol.status,
            QpStatus::Optimal,
            "P=diag({a:.0e},{b:.0e}) must solve, got {:?}",
            sol.status
        );
        // The bug terminated at iteration 0 (the untouched cold start); a
        // genuine solve always steps.
        assert!(
            sol.iters > 0,
            "P=diag({a:.0e},{b:.0e}) certified Optimal at iteration 0 (cold-start \
             false optimum), x={:?}",
            sol.x
        );
        // True optimum x* = [1/a, 1/b]; check the recovered point's own KKT
        // residual is genuinely small (solver-independent oracle).
        let res = sol.kkt_residuals(&prob);
        assert!(
            res.kkt_error() < 1e-6,
            "P=diag({a:.0e},{b:.0e}) kkt_error {:.2e} — not a true optimum (x={:?})",
            res.kkt_error(),
            sol.x
        );
        let obj_exact = -0.5 * (1.0 / a + 1.0 / b);
        let rel = (sol.obj - obj_exact).abs() / obj_exact.abs().max(1.0);
        assert!(
            rel < 1e-6,
            "P=diag({a:.0e},{b:.0e}) objective rel error {rel} (got {}, exact {obj_exact})",
            sol.obj
        );
    }
}

/// A well-conditioned but astronomically-scaled QP (`‖P‖ ~ 1e18`): the pure
/// magnitude drives the `τ` collapse. With the objective normalization it
/// recovers the *exact* clamped optimum in a handful of iterations.
#[test]
fn huge_magnitude_qp_recovers_exact_optimum_at_default_budget() {
    let tgt = [1.5, -1.5, 0.3, -0.7, 2.0, -0.4];
    let (prob, xopt) = diagonal_box_qp(1e18, 1.0, &tgt);

    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);

    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "huge-magnitude QP must converge to Optimal within the default budget \
         (got {:?} after {} iters)",
        sol.status,
        sol.iters
    );
    let x_err = sol
        .x
        .iter()
        .zip(&xopt)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        x_err < 1e-6,
        "x off the exact optimum by {x_err}: {:?}",
        sol.x
    );
    let obj_exact = objective(&prob, &xopt);
    let rel = (sol.obj - obj_exact).abs() / obj_exact.abs().max(1.0);
    assert!(
        rel < 1e-6,
        "objective rel error {rel} (got {}, exact {obj_exact})",
        sol.obj
    );
    // Dual multipliers must return in the *original* objective metric (not the
    // internal scaled one): `Px + c - z_lb + z_ub = 0` relative to ‖grad‖.
    let mut grad = prob.c.clone();
    prob.p_mul_add_pub(&sol.x, &mut grad);
    let gnorm = grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let stat = (0..prob.n)
        .map(|i| (grad[i] - sol.z_lb[i] + sol.z_ub[i]).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        stat / gnorm.max(1.0) < 1e-6,
        "dual multipliers not in the original metric: rel stationarity {}",
        stat / gnorm.max(1.0)
    );
    assert!(
        sol.iters < 60,
        "expected quick convergence, took {} iters",
        sol.iters
    );
}

/// The issue's exact regime: `cond(P) = 1e10` *and* objective magnitude
/// `O(1e22)`. Under such conditioning the lowest-weight coordinates are
/// numerically under-determined at `tol = 1e-8` (their scaled weight falls
/// below the tolerance), so the objective — dominated by the well-determined
/// high-weight terms — is the robust oracle. The status and box feasibility are
/// what regressed in #286.
#[test]
fn issue_286_illconditioned_huge_scale_is_optimal_and_feasible() {
    let tgt = [1.5, -1.5, 0.3, -0.7, 2.0, -0.4];
    let (prob, xopt) = diagonal_box_qp(1e12, 1e10, &tgt);

    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);

    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "badly-scaled QP must report Optimal, not exhaust the budget (got {:?})",
        sol.status
    );
    let box_viol = sol
        .x
        .iter()
        .map(|&xi| (xi.abs() - 1.0).max(0.0))
        .fold(0.0_f64, f64::max);
    assert!(box_viol < 1e-6, "box violated by {box_viol}");
    let obj_exact = objective(&prob, &xopt);
    let rel = (sol.obj - obj_exact).abs() / obj_exact.abs().max(1.0);
    assert!(
        rel < 1e-6,
        "objective rel error {rel} (got {}, exact {obj_exact})",
        sol.obj
    );
    assert!(
        sol.iters < 60,
        "expected quick convergence, took {} iters",
        sol.iters
    );
}

/// A well-scaled box QP: `σ = max(‖P‖, ‖c‖)` is small, so the objective
/// normalization is a no-op (`σ = 1`) and the solve is exactly the historical
/// one — same status, same low iteration count, same answer.
#[test]
fn well_scaled_qp_is_untouched_by_the_normalization() {
    let tgt = [0.9, -0.9, 0.3, -0.7, 0.5, -0.4];
    let (prob, xopt) = diagonal_box_qp(2.0, 1.0, &tgt);

    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);

    assert_eq!(sol.status, QpStatus::Optimal);
    let x_err = sol
        .x
        .iter()
        .zip(&xopt)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(x_err < 1e-6, "x off optimum by {x_err}");
    assert!(
        sol.iters < 30,
        "well-scaled QP iteration count drifted: {}",
        sol.iters
    );
}
