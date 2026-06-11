//! Iteration-count comparison: the convex-QP IPM on the same QPs the
//! CLI exposes as builtins, so the counts line up against the NLP path
//! (`pounce --problem <name>` reports "Number of Iterations").
//!
//! Run: `cargo run -p pounce-convex --example iter_compare`

use pounce_convex::{solve_qp_ipm, QpOptions, QpProblem, Triplet};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn report(name: &str, prob: &QpProblem) {
    let sol = solve_qp_ipm(prob, &QpOptions::default(), backend);
    println!(
        "{name:<20} status={:?} iters={} obj={:.6} x={:?}",
        sol.status, sol.iters, sol.obj, sol.x
    );
}

fn main() {
    // `quadratic`: min (x0-3)^2 + (x1-4)^2  ⇒  ½xᵀ(2I)x + (-6,-8)ᵀx + const
    // P = 2I, c = (-6, -8). (constant 25 dropped; affects obj only)
    report(
        "quadratic",
        &QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-6.0, -8.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        },
    );

    // `bounded-quadratic`: same objective, 0 ≤ x ≤ 2 (so optimum at the
    // upper bounds (2,2)). Bounds as four inequality rows.
    report(
        "bounded-quadratic",
        &QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-6.0, -8.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),  // x0 ≤ 2
                Triplet::new(1, 1, 1.0),  // x1 ≤ 2
                Triplet::new(2, 0, -1.0), // x0 ≥ 0
                Triplet::new(3, 1, -1.0), // x1 ≥ 0
            ],
            h: vec![2.0, 2.0, 0.0, 0.0],
            lb: vec![],
            ub: vec![],
        },
    );

    // `eq-quadratic`: min x0² + x1² s.t. x0 + x1 = 1 ⇒ P = 2I, c = 0.
    report(
        "eq-quadratic",
        &QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![1.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        },
    );
}
