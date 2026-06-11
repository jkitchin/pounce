//! Scaling regression: the convex-QP IPM's *iteration count* must stay
//! roughly flat as the problem grows — the defining property of a
//! healthy interior-point method. (Wall-clock growth is the shared
//! pounce-linsol factorization's concern, not the IPM's, so this test
//! guards iterations, not time.)
//!
//! A box-constrained tridiagonal convex QP is solved at sizes spanning
//! three orders of magnitude; the iteration count must not drift upward
//! with n.

use pounce_convex::{solve_qp_ipm, QpOptions, QpProblem, QpStatus, Triplet};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn sparse_box_qp(n: usize) -> QpProblem {
    let mut p_lower = Vec::with_capacity(2 * n);
    for i in 0..n {
        p_lower.push(Triplet::new(i, i, 4.0));
        if i > 0 {
            p_lower.push(Triplet::new(i, i - 1, -1.0));
        }
    }
    let c: Vec<f64> = (0..n).map(|i| -2.0 - (i % 5) as f64).collect();
    let mut g = Vec::with_capacity(2 * n);
    let mut h = Vec::with_capacity(2 * n);
    for i in 0..n {
        g.push(Triplet::new(2 * i, i, 1.0)); // x_i ≤ 1
        h.push(1.0);
        g.push(Triplet::new(2 * i + 1, i, -1.0)); // −x_i ≤ 0
        h.push(0.0);
    }
    QpProblem {
        n,
        p_lower,
        c,
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![],
        ub: vec![],
    }
}

#[test]
fn iteration_count_is_flat_across_sizes() {
    let mut counts = Vec::new();
    for &n in &[100usize, 1_000, 5_000] {
        let sol = solve_qp_ipm(&sparse_box_qp(n), &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "n={n} did not converge");
        counts.push(sol.iters);
    }
    // The iteration count for a well-behaved IPM grows at most very
    // slowly (theoretically ~√n, in practice near-constant on these
    // well-conditioned problems). Assert it never exceeds a small flat
    // bound across 50× growth in n — catches a regression that ties
    // iteration count to problem size.
    for (i, &c) in counts.iter().enumerate() {
        assert!(c <= 20, "size index {i}: {c} iters (expected flat, ≤20)");
    }
    // And that it does not blow up 100→5000: at most a couple extra.
    assert!(
        counts[2] <= counts[0] + 3,
        "iteration count drifted with size: {counts:?}"
    );
}
