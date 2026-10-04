//! gh #991 at the engine: a convex QP infeasible by a small exact margin must
//! come back `QpStatus::Infeasible`, not `MaxIter`, and carry multipliers that
//! prove it.
//!
//! `x₀ + 2x₁ ≤ 2` and `x₀ + 2x₁ ≥ 2 + ε` on free variables. The elastic
//! phase-1 converges at the minimal violation `ε/2` per row, but the convex
//! feasibility pass that was the only thing allowed to say "infeasible" bounds
//! its penalty bias by ~`(1e3)²/(2γ)` on a free coordinate, which never fell
//! below that residual — so the converged phase-1 was flattened to `MaxIter`
//! after 3 iterations. The fix asks the question directly, through a Farkas
//! certificate from the objective-free phase-1.
//!
//! Oracle: closed form. `y = (1, 1)` combines the rows to `0 ≤ −ε`.

use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver, QpStatus,
};
use std::rc::Rc;

const INF: f64 = 1e20;

fn solve(eps: f64, opts: &QpOptions) -> pounce_qp::QpSolution {
    // P = [[2, 1], [1, 2]] (lower triangle), c = (−1, 2).
    let h_space = SymTMatrixSpace::new(2, vec![1, 2, 2], vec![1, 1, 2]);
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&[2.0, 1.0, 2.0]);
    let a_space = GenTMatrixSpace::new(2, 2, vec![1, 1, 2, 2], vec![1, 2, 1, 2]);
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&[1.0, 2.0, -1.0, -2.0]);
    let g = [-1.0, 2.0];
    let bl = [-INF, -INF];
    let bu = [2.0, -(2.0 + eps)];
    let xl = [-INF, -INF];
    let xu = [INF, INF];
    let qp = QpProblem {
        n: 2,
        m: 2,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    solver.solve(&qp, None, opts).expect("solve")
}

#[test]
fn sliver_gap_is_certified_infeasible_on_both_inner_paths() {
    for schur in [false, true] {
        for max_iter in [200, 100_000] {
            let opts = QpOptions {
                use_schur_updates: schur,
                max_iter,
                ..QpOptions::default()
            };
            let sol = solve(1e-5, &opts);
            assert_eq!(
                sol.status,
                QpStatus::Infeasible,
                "schur={schur} max_iter={max_iter}"
            );
            // The multipliers are the certificate: both rows weighted alike
            // (the combination that cancels `x`), and nonzero.
            let y = &sol.lambda_g;
            assert!(y[0].abs() > 0.0, "y = {y:?}");
            assert!(
                (y[0] - y[1]).abs() <= 1e-9 * y[0].abs(),
                "y must cancel the rows' x-terms: {y:?}"
            );
        }
    }
}

#[test]
fn gap_inside_feas_tol_is_not_certified() {
    // ε/2 = 5e-10 per row is inside the default feas_tol of 1e-9: a point
    // feasible to tolerance exists, so the model is not infeasible.
    let sol = solve(1e-9, &QpOptions::default());
    assert_ne!(sol.status, QpStatus::Infeasible);
}
