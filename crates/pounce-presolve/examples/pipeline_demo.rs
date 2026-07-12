//! End-to-end pipeline worked example — PRs 2-7 of issue #53.
//!
//! Hand-crafted 4-variable problem mixing several elimination
//! patterns and a non-trivial objective:
//!
//! ```text
//!   min   f(a, b, c, d) = (a - 1)² + (b - 2)² + (c - 3)² + d²
//!   s.t.  c0:  a            = 1
//!         c1:  b + c        = 5
//!         c2:  b - c        = -1
//!         c3:  a + b + d    = 4
//! ```
//!
//! All four constraints are equalities, the system is square and
//! non-singular, and BTF decomposes it into three elimination
//! blocks in order:
//!
//! - Block 0: singleton `{row 0 ↔ a}` → a = 1.
//! - Block 1: 2-cycle `{rows 1, 2 ↔ b, c}` → Newton solves (b, c) = (2, 3).
//! - Block 2: singleton `{row 3 ↔ d}` (depends on a and b from
//!   earlier blocks) → d = 1.
//!
//! The reduced problem is empty — presolve solved the entire system
//! by structural elimination. The example then performs multiplier
//! recovery and verifies the full-space KKT residual is zero (to
//! floating-point precision).
//!
//! Run with:
//! ```bash
//! cargo run -p pounce-presolve --example pipeline_demo
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used)]

use pounce_common::types::Number;
use pounce_presolve::matching::hopcroft_karp;
use pounce_presolve::{
    AuxiliaryCouplingClass, BlockEquations, BlockSolveOptions, BlockSolver, BlockTriangularForm,
    DampedNewtonSolver, DulmageMendelsohnPartition, EqualityIncidence, InequalityIncidence,
    ProbeView, ReductionFrame, SquareComponents, classify_block,
};

struct LinearBlockEqs {
    a: Vec<Number>,
    b: Vec<Number>,
    n: usize,
}
impl BlockEquations for LinearBlockEqs {
    fn dim(&self) -> usize {
        self.n
    }
    fn eval(&mut self, x: &[Number], f: &mut [Number]) -> bool {
        for i in 0..self.n {
            let mut s = -self.b[i];
            for j in 0..self.n {
                s += self.a[i * self.n + j] * x[j];
            }
            f[i] = s;
        }
        true
    }
    fn jacobian(&mut self, _x: &[Number], j: &mut [Number]) -> bool {
        j.copy_from_slice(&self.a);
        true
    }
}

fn main() {
    let n_vars = 4; // a, b, c, d
    let n_rows = 4;
    // Jacobian sparsity (row-major).
    let jac_pattern = [
        (0, 0), // c0: a
        (1, 1),
        (1, 2), // c1: b + c
        (2, 1),
        (2, 2), // c2: b - c
        (3, 0),
        (3, 1),
        (3, 3), // c3: a + b + d
    ];
    let jac_irow: Vec<i32> = jac_pattern.iter().map(|(r, _)| *r as i32).collect();
    let jac_jcol: Vec<i32> = jac_pattern.iter().map(|(_, c)| *c as i32).collect();
    let g_l = [1.0, 5.0, -1.0, 4.0];
    let g_u = [1.0, 5.0, -1.0, 4.0]; // all equalities

    let probe = ProbeView {
        n_vars,
        m_rows: n_rows,
        jac_irow: &jac_irow,
        jac_jcol: &jac_jcol,
        jac_values: None,
        g_l: &g_l,
        g_u: &g_u,
        linearity: None,
        one_based: false,
        eq_tol: 1e-12,
        excluded_vars: None,
        excluded_rows: None,
    };

    println!("pipeline_demo — PRs 2-7 of pounce#53");
    println!("======================================");

    // ----- Stage 1: incidence + matching -----
    let inc = EqualityIncidence::from_probe(&probe);
    println!(
        "\n[Stage 1] EqualityIncidence: {} eq rows, {} vars",
        inc.n_eq_rows(),
        inc.n_vars
    );
    let m = hopcroft_karp(&inc);
    println!("[Stage 1] Matching: size = {}", m.size);

    // ----- Stage 2: DM partition -----
    let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
    println!("\n[Stage 2] DM partition:");
    println!("   over rows={:?}, cols={:?}", dm.over_rows, dm.over_cols);
    println!(
        "   square rows={:?}, cols={:?}",
        dm.square_rows, dm.square_cols
    );
    println!(
        "   under rows={:?}, cols={:?}",
        dm.under_rows, dm.under_cols
    );

    // ----- Stage 3: components -----
    let comps = SquareComponents::of_square_part(&inc, &m, &dm);
    println!(
        "\n[Stage 3] Components: {} square block(s)",
        comps.components.len()
    );
    for (i, c) in comps.components.iter().enumerate() {
        println!("   component {i}: rows={:?}, cols={:?}", c.eq_rows, c.cols);
    }

    // ----- Stage 4: BTF -----
    println!("\n[Stage 4] BTF (per component):");
    let ineq = InequalityIncidence::from_probe(&probe);

    // Walk each component's BTF, solve each block, build the frame.
    let mut all_fixed_vars: Vec<usize> = Vec::new();
    let mut all_fixed_values: Vec<Number> = Vec::new();
    let mut all_dropped_rows: Vec<usize> = Vec::new();
    let mut x_running = vec![0.0; n_vars];
    // Full A and b for this problem, used both for block extraction
    // and the final KKT check.
    let a_full = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let b_full = [1.0, 5.0, -1.0, 4.0];

    for comp in &comps.components {
        let btf = BlockTriangularForm::of_component(&inc, &m, comp);
        for (bi, block) in btf.blocks.iter().enumerate() {
            let class = classify_block(block, &ineq, &Default::default());
            println!(
                "   component cols={:?}, block {bi}: rows={:?}, cols={:?}, class={class:?}",
                comp.cols, block.eq_rows, block.cols
            );
            assert_eq!(class, AuxiliaryCouplingClass::PureEquality);

            // Build a small linear system for this block, splicing in
            // the values of already-solved variables.
            let k = block.eq_rows.len();
            let mut a_block = vec![0.0; k * k];
            let mut b_block = vec![0.0; k];
            for (ii, &r) in block.eq_rows.iter().enumerate() {
                let mut rhs = b_full[r];
                for j in 0..n_vars {
                    if let Some(jj) = block.cols.iter().position(|&c| c == j) {
                        a_block[ii * k + jj] = a_full[r][j];
                    } else {
                        rhs -= a_full[r][j] * x_running[j];
                    }
                }
                b_block[ii] = rhs;
            }

            // Solve.
            let mut eqs = LinearBlockEqs {
                a: a_block,
                b: b_block,
                n: k,
            };
            let mut solver = DampedNewtonSolver;
            let opt = BlockSolveOptions::default();
            let out = solver.solve(&vec![0.0; k], &mut eqs, &opt).unwrap();

            // Record the block's solution.
            for (ii, &c) in block.cols.iter().enumerate() {
                x_running[c] = out.x[ii];
                all_fixed_vars.push(c);
                all_fixed_values.push(out.x[ii]);
            }
            for &r in &block.eq_rows {
                all_dropped_rows.push(r);
            }
        }
    }

    // ----- Stage 5: lift via the reduction frame -----
    println!("\n[Stage 5] After block solves:");
    println!("   x_running (a, b, c, d) = {:?}", x_running);
    // For this demo we declare ALL fixed vars and rows as one
    // composite frame — in the real PR 8 orchestrator, each
    // elimination layer becomes its own frame on the stack.
    // Sort by variable index for the frame's contract.
    let mut order: Vec<usize> = (0..all_fixed_vars.len()).collect();
    order.sort_by_key(|&i| all_fixed_vars[i]);
    let fixed_vars_sorted: Vec<usize> = order.iter().map(|&i| all_fixed_vars[i]).collect();
    let fixed_values_sorted: Vec<Number> = order.iter().map(|&i| all_fixed_values[i]).collect();
    let mut dropped_rows_sorted = all_dropped_rows.clone();
    dropped_rows_sorted.sort_unstable();

    let frame = ReductionFrame::new(
        n_vars,
        n_rows,
        fixed_vars_sorted.clone(),
        fixed_values_sorted,
        dropped_rows_sorted.clone(),
    );
    println!(
        "   ReductionFrame: fixed_vars={:?}, dropped_rows={:?}",
        frame.fixed_vars, frame.dropped_rows
    );
    println!(
        "   reduced shape: {} vars × {} rows",
        frame.n_reduced_vars(),
        frame.n_reduced_rows()
    );

    // For this problem BTF decomposed every row into a block — the
    // reduced shape is 0×0 and no IPM step is needed. The lift is
    // trivial: x_full = the values the block solves produced.
    let x_reduced: Vec<Number> = Vec::new();
    println!("\n[Stage 6] Reduced problem is empty (presolve solved it all).");
    let x_full = frame.lift_x(&x_reduced);
    println!("   lifted x_full = {:?}", x_full);

    // Sanity: should match (1, 2, 3, 1).
    let expected = [1.0, 2.0, 3.0, 1.0];
    let max_x_err = x_full
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("   max |x - expected|: {:.3e}", max_x_err);
    assert!(max_x_err < 1e-9);

    // ----- Stage 7: multiplier recovery -----
    // Objective: f = (a-1)² + (b-2)² + (c-3)² + d²
    // ∇f at the optimum (1, 2, 3, 1): (0, 0, 0, 2).
    let grad_f = [
        2.0 * (x_full[0] - 1.0),
        2.0 * (x_full[1] - 2.0),
        2.0 * (x_full[2] - 3.0),
        2.0 * x_full[3],
    ];
    println!("\n[Stage 7] ∇f at the optimum: {:?}", grad_f);

    // No kept rows → reduced λ is empty.
    let lambda_reduced: Vec<Number> = Vec::new();
    let mut lambda_full = frame.lift_lambda(&lambda_reduced);
    println!("   reduced λ = {:?}", lambda_reduced);
    println!("   lifted λ (pre-recovery) = {:?}", lambda_full);

    // Build full Jacobian for the recovery LU.
    let mut jac_full = vec![0.0; n_rows * n_vars];
    for (r, row) in a_full.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            jac_full[r * n_vars + c] = v;
        }
    }
    let lam_dropped = frame
        .recover_dropped_multipliers(&grad_f, &jac_full, &lambda_full)
        .unwrap();
    for (idx, &r) in frame.dropped_rows.iter().enumerate() {
        lambda_full[r] = lam_dropped[idx];
    }
    println!("   recovered λ (full) = {:?}", lambda_full);

    // Final KKT residual at every variable.
    println!("\n[Stage 8] Full-space KKT residual:");
    let mut max_kkt: Number = 0.0;
    for i in 0..n_vars {
        let mut s = grad_f[i];
        for r in 0..n_rows {
            s -= jac_full[r * n_vars + i] * lambda_full[r];
        }
        println!("   ∂L/∂x[{i}] = {:.3e}", s);
        max_kkt = max_kkt.max(s.abs());
    }
    println!("   max |KKT residual| = {:.3e}", max_kkt);
    assert!(max_kkt < 1e-10);
    println!("\n✓ Pipeline end-to-end correct.");
}
