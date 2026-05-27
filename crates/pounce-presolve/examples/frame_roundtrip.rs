//! Worked example for PR 7 of issue #53 — reduction frame +
//! postsolve multiplier recovery.
//!
//! Full problem (3 variables `b1, b2, y`, 3 constraints):
//!
//! ```text
//!   min   f(b1, b2, y) = 10 b1 + 4 b2 + y²
//!   s.t.  c0(b):  2 b1 +   b2       - 3 = 0   ← dropped (equality)
//!         c1(b):    b1 -   b2       + 1 = 0   ← dropped (equality)
//!         c2(b):    b1 +   b2 +  y  - 5 = 0   ← kept    (equality)
//! ```
//!
//! Block solve fixes `(b1, b2) = (2/3, 5/3)` by satisfying `c0` and
//! `c1`. The reduced problem has one variable (`y`) and one row
//! (`c2`), which gives `y* = 8/3`. The IPM then converges to that.
//!
//! Postsolve must:
//! 1. Lift `x` back to full space `(b1*, b2*, y*)`.
//! 2. Recover λ for the dropped rows by solving the full-space KKT
//!    stationarity at the fixed variables.
//! 3. Verify the full-space KKT residual is below tolerance.
//!
//! Run with:
//! ```bash
//! cargo run -p pounce-presolve --example frame_roundtrip
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used)]

use pounce_presolve::reduction_frame::ReductionFrame;

fn main() {
    // Full-space problem shape.
    let n_vars = 3;
    let n_rows = 3;

    // Block solve fixed b1, b2.
    let frame = ReductionFrame::new(
        n_vars,
        n_rows,
        vec![0, 1],                 // fixed_vars: b1, b2
        vec![2.0 / 3.0, 5.0 / 3.0], // their values
        vec![0, 1],                 // dropped_rows: c0, c1
    );

    println!("frame_roundtrip example — PR 7 of pounce#53");
    println!(
        "full vars: {}, reduced vars: {}",
        frame.n_full_vars(),
        frame.n_reduced_vars()
    );
    println!(
        "full rows: {}, reduced rows: {}",
        frame.n_full_rows(),
        frame.n_reduced_rows()
    );

    // Simulated IPM result on the reduced problem.
    let y_star = 8.0 / 3.0;
    let x_reduced = [y_star];
    let lambda_reduced = [2.0 * y_star]; // stationarity at y: 2y - λ_kept = 0

    // Lift x to full space.
    let x_full = frame.lift_x(&x_reduced);
    println!(
        "lifted x:      [{:.6}, {:.6}, {:.6}]",
        x_full[0], x_full[1], x_full[2]
    );

    // Lift λ to full space — zeros at dropped row indices, to be
    // filled in by multiplier recovery.
    let mut lambda_full = frame.lift_lambda(&lambda_reduced);
    println!("lifted λ (pre-recovery): {:?}", lambda_full);

    // Build the full Jacobian at x_full (rows × vars, row-major).
    let jac_full = [
        2.0, 1.0, 0.0, // c0: 2 b1 + b2
        1.0, -1.0, 0.0, // c1: b1 - b2
        1.0, 1.0, 1.0, // c2: b1 + b2 + y
    ];
    // Objective gradient at x_full.
    let grad_f = [10.0, 4.0, 2.0 * x_full[2]];

    // Recover the dropped multipliers.
    let lam_dropped = frame
        .recover_dropped_multipliers(&grad_f, &jac_full, &lambda_full)
        .expect("non-singular");

    // Fill them into the full λ vector.
    for (k, &r) in frame.dropped_rows.iter().enumerate() {
        lambda_full[r] = lam_dropped[k];
    }
    println!("lifted λ (post-recovery): {:?}", lambda_full);

    // Verify full-space stationarity at ALL variables.
    println!("KKT stationarity residual (should be ≈ 0):");
    let mut max_res = 0.0_f64;
    for i in 0..n_vars {
        let mut s = grad_f[i];
        for r in 0..n_rows {
            s -= jac_full[r * n_vars + i] * lambda_full[r];
        }
        println!("  var {i}: {:.3e}", s);
        max_res = max_res.max(s.abs());
    }
    println!("max |residual|: {:.3e}", max_res);
    assert!(max_res < 1e-12);
}
