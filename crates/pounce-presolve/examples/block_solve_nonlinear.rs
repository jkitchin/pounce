//! Worked example for PR 6 of issue #53 — damped-Newton block solve.
//!
//! Solves the 2-variable nonlinear system
//!
//! ```text
//! F(x, y) = [ x² + y - 5;
//!             x  + y² - 3 ]
//! ```
//!
//! starting from `(1.5, 1.5)`. Prints the iteration count and final
//! residual; the residual norm should be far below the default
//! tolerance of `1e-8`.
//!
//! Run with:
//! ```bash
//! cargo run -p pounce-presolve --example block_solve_nonlinear
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used)]

use pounce_common::types::Number;
use pounce_presolve::block_solve::{
    BlockEquations, BlockSolveOptions, BlockSolver, DampedNewtonSolver,
};

struct Demo;
impl BlockEquations for Demo {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&mut self, x: &[Number], f: &mut [Number]) -> bool {
        f[0] = x[0] * x[0] + x[1] - 5.0;
        f[1] = x[0] + x[1] * x[1] - 3.0;
        true
    }
    fn jacobian(&mut self, x: &[Number], j: &mut [Number]) -> bool {
        // Row-major 2x2.
        j[0] = 2.0 * x[0];
        j[1] = 1.0;
        j[2] = 1.0;
        j[3] = 2.0 * x[1];
        true
    }
}

fn main() {
    let mut eqs = Demo;
    let opts = BlockSolveOptions::default();
    let mut solver = DampedNewtonSolver;
    let out = solver
        .solve(&[1.5, 1.5], &mut eqs, &opts)
        .expect("converges");

    println!("block_solve_nonlinear example — PR 6 of pounce#53");
    println!("iterations:    {}", out.iterations);
    println!("residual norm: {:.3e}", out.residual_norm);
    println!("x:             [{:.10}, {:.10}]", out.x[0], out.x[1]);

    // Verify F(x) ≈ 0.
    let mut residual = [0.0; 2];
    eqs.eval(&out.x, &mut residual);
    println!("verify F(x):   [{:.3e}, {:.3e}]", residual[0], residual[1]);
    assert!(out.residual_norm < opts.tol);
}
