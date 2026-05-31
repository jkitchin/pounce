//! Batched / multiple-RHS convex-QP solving: solve a family of QPs that
//! share structure but differ in their data, in parallel via rayon.
//!
//! Run: `cargo run -p pounce-convex --release --example batch_solve`

use pounce_convex::{solve_qp_batch_parallel, solve_qp_multi_rhs, QpOptions, QpProblem, Triplet};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use std::time::Instant;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Box-constrained QP `min ½xᵀ(2I)x + cᵀx, 0 ≤ x ≤ 1` for a given `c`.
fn boxed_qp(c: Vec<f64>) -> QpProblem {
    let n = c.len();
    QpProblem {
        n,
        p_lower: (0..n).map(|i| Triplet::new(i, i, 2.0)).collect(),
        c,
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0; n],
        ub: vec![1.0; n],
    }
}

fn main() {
    let opts = QpOptions::default();

    println!("=== multiple RHS: one structure, many objectives ===");
    let base = boxed_qp(vec![0.0, 0.0]);
    let cs = vec![
        vec![-1.0, -4.0],
        vec![-4.0, 1.0],
        vec![3.0, -2.0],
        vec![0.5, 0.5],
    ];
    let sols = solve_qp_multi_rhs(&base, &cs, &opts, backend);
    for (c, s) in cs.iter().zip(&sols) {
        println!(
            "c={c:?} → x=[{:.3}, {:.3}]  obj={:.4}",
            s.x[0], s.x[1], s.obj
        );
    }

    println!("\n=== batch throughput (parallel via rayon) ===");
    for &count in &[100usize, 1_000, 5_000] {
        // A sweep of distinct small box QPs.
        let probs: Vec<QpProblem> = (0..count)
            .map(|k| {
                let t = (k as f64) / (count as f64);
                boxed_qp(vec![-2.0 * t, -2.0 * (1.0 - t)])
            })
            .collect();

        let t0 = Instant::now();
        let batched = solve_qp_batch_parallel(&probs, &opts, backend);
        let par = t0.elapsed().as_secs_f64() * 1e3;

        // Sequential reference for comparison.
        let t1 = Instant::now();
        let seq: Vec<_> = probs
            .iter()
            .map(|p| pounce_convex::solve_qp_ipm(p, &opts, backend))
            .collect();
        let seq_ms = t1.elapsed().as_secs_f64() * 1e3;

        let all_ok = batched
            .iter()
            .zip(&seq)
            .all(|(b, s)| (b.obj - s.obj).abs() < 1e-9);
        println!(
            "{count:>5} QPs: batch(par) {par:>8.1} ms   sequential {seq_ms:>8.1} ms   \
             speedup {:.2}×   (results match: {all_ok})",
            seq_ms / par,
        );
    }

    println!("\nEach QP solves independently (own factor + iterate), so the");
    println!("batch is embarrassingly parallel; rayon balances uneven iteration");
    println!("counts across instances.");
}
