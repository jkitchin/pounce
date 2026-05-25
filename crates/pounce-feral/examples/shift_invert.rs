//! Worked example: factor-once / solve-many for a shift-invert
//! eigenvalue probe.
//!
//! Shift-invert applies `(A - σI)⁻¹` repeatedly to a starting vector
//! to converge on the eigenvalue closest to σ. The matrix `(A - σI)`
//! is factored once; each power-iteration step is a single back-solve
//! against that factor. This is the canonical "factor once, solve
//! many" workload.
//!
//! Run with `cargo run -p pounce-feral --example shift_invert`.

use pounce_feral::FeralSolverInterface;
use pounce_linsol::Factorization;

fn main() {
    // Symmetric 4x4 tridiagonal: A = tridiag(-1, 2, -1) — classic 1-D
    // Laplacian. Eigenvalues are 2 - 2 cos(k π / 5) for k = 1..=4 ≈
    // {0.382, 1.382, 2.618, 3.618}.
    let n = 4;
    let sigma = 0.3; // shift near the smallest eigenvalue
    let (rows, cols, values) = build_shifted_laplacian(n, sigma);

    let mut fact = Factorization::new(
        n as i32,
        rows,
        cols,
        values,
        Box::new(FeralSolverInterface::new()),
    )
    .expect("factor (A - σI)");

    // Power iteration on (A - σI)⁻¹. Each step: solve (A - σI) y = x,
    // then x ← y / ||y||. Converges to the eigenvector for the eigenvalue
    // closest to σ.
    let mut x = vec![1.0_f64; n];
    normalize(&mut x);

    println!("shift-invert with σ = {sigma}, dim = {n}");
    println!("each iteration is one back-solve against the cached factor");
    println!();

    let mut prev_ratio = 0.0;
    for iter in 0..15 {
        let mut y = x.clone();
        fact.solve_one(&mut y).expect("back-solve");
        // Rayleigh quotient against the unfactored A gives the eigenvalue.
        let ax = laplacian_apply(&x);
        let lambda_estimate = dot(&ax, &x) / dot(&x, &x);
        let ratio = norm(&y);
        let delta = (ratio - prev_ratio).abs();
        println!(
            "iter {iter:2}: λ ≈ {lambda_estimate:.6}   ‖(A-σI)⁻¹ x‖ = {ratio:.6}   Δ = {delta:.2e}"
        );
        prev_ratio = ratio;
        x = y;
        normalize(&mut x);
        if delta < 1e-10 && iter > 2 {
            println!("\nconverged after {iter} back-solves; factor was computed once");
            break;
        }
    }

    // Final eigenvalue estimate.
    let ax = laplacian_apply(&x);
    let lambda = dot(&ax, &x) / dot(&x, &x);
    let exact = 2.0 - 2.0 * (std::f64::consts::PI / 5.0).cos();
    println!("\neigenvalue estimate: {lambda:.8}");
    println!("exact (smallest):    {exact:.8}");
    println!("absolute error:      {:.2e}", (lambda - exact).abs());
}

/// Build (A - σI) for the 1-D Laplacian tridiag(-1, 2, -1), in
/// lower-triangle 1-based triplet format.
fn build_shifted_laplacian(n: usize, sigma: f64) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        // Diagonal.
        rows.push((i + 1) as i32);
        cols.push((i + 1) as i32);
        values.push(2.0 - sigma);
        // Sub-diagonal (i+1, i) for i < n-1 — lower triangle only.
        if i + 1 < n {
            rows.push((i + 2) as i32);
            cols.push((i + 1) as i32);
            values.push(-1.0);
        }
    }
    (rows, cols, values)
}

/// Apply A (without shift) to x.
fn laplacian_apply(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = 2.0 * x[i];
        if i > 0 {
            y[i] -= x[i - 1];
        }
        if i + 1 < n {
            y[i] -= x[i + 1];
        }
    }
    y
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn normalize(a: &mut [f64]) {
    let n = norm(a);
    for x in a.iter_mut() {
        *x /= n;
    }
}
