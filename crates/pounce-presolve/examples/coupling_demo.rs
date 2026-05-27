//! Worked example for PR 5 of issue #53 — coupling classification.
//!
//! 3-variable problem:
//!
//! ```text
//! min  x[0]                           (objective grad supports {0})
//! s.t. x[0] + x[1]       = 1   (equality, row 0)
//!      x[1] + x[2]       = 1   (equality, row 1)
//!      x[1]              ≤ 2   (inequality, row 2 — touches x[1])
//! ```
//!
//! After PRs 2–4 we'd get one equality block on vars {0, 1, 2} and
//! rows {0, 1}. Because variable 1 appears in the inequality AND
//! variable 0 appears in the objective gradient, the block is
//! classified as `ObjectiveAndInequalityCoupled`.
//!
//! Run with:
//! ```bash
//! cargo run -p pounce-presolve --example coupling_demo
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used)]

use pounce_presolve::btf::BlockTriangularBlock;
use pounce_presolve::coupling::{
    classify_block, objective_gradient_support, AuxiliaryCouplingClass,
};
use pounce_presolve::incidence::{InequalityIncidence, ProbeView};

fn main() {
    // Jacobian sparsity:
    // row 0 (equality) touches cols 0, 1
    // row 1 (equality) touches cols 1, 2
    // row 2 (inequality) touches col 1
    let irow: [i32; 5] = [0, 0, 1, 1, 2];
    let jcol: [i32; 5] = [0, 1, 1, 2, 1];
    let g_l = [1.0, 1.0, -1e19];
    let g_u = [1.0, 1.0, 2.0];

    let probe = ProbeView {
        n_vars: 3,
        m_rows: 3,
        jac_irow: &irow,
        jac_jcol: &jcol,
        jac_values: None,
        g_l: &g_l,
        g_u: &g_u,
        linearity: None,
        one_based: false,
        eq_tol: 1e-12,
        excluded_vars: None,
        excluded_rows: None,
    };

    let ineq = InequalityIncidence::from_probe(&probe);
    println!("coupling_demo example — PR 5 of pounce#53");
    println!("inequality rows: {}", ineq.n_ineq_rows());
    for k in 0..ineq.n_ineq_rows() {
        println!(
            "  ineq row {} → vars {:?}",
            ineq.ineq_row_inner_idx[k],
            ineq.neighbors(k)
        );
    }

    // Objective grad: ∇f = [1, 0, 0] → support = {0}.
    let grad_f = [1.0, 0.0, 0.0];
    let obj_support = objective_gradient_support(&grad_f, 1e-12);
    println!("objective grad support: {:?}", obj_support);

    let block = BlockTriangularBlock {
        eq_rows: vec![0, 1],
        cols: vec![0, 1, 2],
    };
    println!(
        "candidate block: rows={:?}, cols={:?}",
        block.eq_rows, block.cols
    );

    let class = classify_block(&block, &ineq, &obj_support);
    println!("coupling class: {:?}", class);
    assert_eq!(class, AuxiliaryCouplingClass::ObjectiveAndInequalityCoupled);

    // Try a "clean" block on rows {0} and cols {0} — no inequality
    // touches col 0, but obj does. Should be ObjectiveCoupled.
    let clean = BlockTriangularBlock {
        eq_rows: vec![0],
        cols: vec![0],
    };
    let clean_class = classify_block(&clean, &ineq, &obj_support);
    println!("clean block class: {:?}", clean_class);
    assert_eq!(clean_class, AuxiliaryCouplingClass::ObjectiveCoupled);
}
