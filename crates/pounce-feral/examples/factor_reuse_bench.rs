//! Wall-clock comparison: factor-once-solve-many vs refactor-every-time.
//!
//! Demonstrates the speedup `Factorization::solve` (which reuses the
//! cached factor) buys you over rebuilding from scratch each call.
//! Not a criterion bench (the repo has no criterion convention yet);
//! prints raw timings via `std::time::Instant`.
//!
//! Run with `cargo run --release -p pounce-feral --example factor_reuse_bench`.

use pounce_feral::FeralSolverInterface;
use pounce_linsol::Factorization;
use std::time::Instant;

const N_SOLVES: usize = 1000;

fn main() {
    // Symmetric tridiagonal of moderate size — large enough that the
    // factor cost dominates a single back-solve.
    let n = 500;
    let (rows, cols, values) = build_tridiag(n);

    // === Path A: factor once, solve many. ===
    let t = Instant::now();
    let mut fact = Factorization::new(
        n as i32,
        rows.clone(),
        cols.clone(),
        values.clone(),
        Box::new(FeralSolverInterface::new()),
    )
    .expect("initial factor");
    let factor_time = t.elapsed();

    let t = Instant::now();
    let mut rhs = vec![1.0_f64; n];
    for k in 0..N_SOLVES {
        // Vary the RHS slightly so the compiler can't hoist the call.
        rhs[k % n] = (k as f64).sin();
        fact.solve_one(&mut rhs).expect("back-solve");
    }
    let solve_time = t.elapsed();
    let path_a = factor_time + solve_time;

    // === Path B: refactor every time. ===
    let t = Instant::now();
    let mut rhs_b = vec![1.0_f64; n];
    for k in 0..N_SOLVES {
        rhs_b[k % n] = (k as f64).sin();
        let mut fresh = Factorization::new(
            n as i32,
            rows.clone(),
            cols.clone(),
            values.clone(),
            Box::new(FeralSolverInterface::new()),
        )
        .expect("refactor");
        fresh.solve_one(&mut rhs_b).expect("back-solve");
    }
    let path_b = t.elapsed();

    println!("dim = {n}, {N_SOLVES} solves");
    println!();
    println!("Path A: factor once, solve many");
    println!("  initial factor: {:>10.3?}", factor_time);
    println!("  {N_SOLVES} back-solves: {:>10.3?}", solve_time);
    println!("  total:          {:>10.3?}", path_a);
    println!();
    println!("Path B: refactor every solve");
    println!("  total:          {:>10.3?}", path_b);
    println!();
    println!("speedup (B / A): {:>10.1}×", path_b.as_secs_f64() / path_a.as_secs_f64());

    // Sanity: results agree.
    let diff: f64 = rhs
        .iter()
        .zip(rhs_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("max RHS difference between paths: {diff:.2e}");
}

/// Symmetric tridiagonal `tridiag(-1, 2.5, -1)` in 1-based lower-tri
/// triplet format. SPD by construction.
fn build_tridiag(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut rows = Vec::with_capacity(2 * n - 1);
    let mut cols = Vec::with_capacity(2 * n - 1);
    let mut values = Vec::with_capacity(2 * n - 1);
    for i in 0..n {
        rows.push((i + 1) as i32);
        cols.push((i + 1) as i32);
        values.push(2.5);
        if i + 1 < n {
            rows.push((i + 2) as i32);
            cols.push((i + 1) as i32);
            values.push(-1.0);
        }
    }
    (rows, cols, values)
}
