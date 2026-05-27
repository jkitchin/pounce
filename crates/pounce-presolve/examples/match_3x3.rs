//! Worked example for PR 2 of issue #53 — equality-row × variable
//! incidence + Hopcroft-Karp matching.
//!
//! Builds a 3 equality-row × 3 variable probe whose Jacobian sparsity
//! is
//!
//! ```text
//! row 0: cols 0, 1
//! row 1: cols 1, 2
//! row 2: cols 0, 2
//! ```
//!
//! and asks Hopcroft-Karp for a maximum matching. The expected output
//! is a perfect matching of size 3.
//!
//! Run with:
//! ```bash
//! cargo run -p pounce-presolve --example match_3x3
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used)]

use pounce_presolve::incidence::{EqualityIncidence, ProbeView};
use pounce_presolve::matching::hopcroft_karp;

fn main() {
    // Jacobian sparsity, in C-style 0-based indexing.
    let irow: [i32; 6] = [0, 0, 1, 1, 2, 2];
    let jcol: [i32; 6] = [0, 1, 1, 2, 0, 2];
    let g_l = [0.0; 3];
    let g_u = [0.0; 3]; // all rows equalities.

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
    let inc = EqualityIncidence::from_probe(&probe);
    println!("match_3x3 example — PR 2 of pounce#53");
    println!("eq_rows: {:?}", inc.eq_row_inner_idx);
    for k in 0..inc.n_eq_rows() {
        println!("  row {k} → {:?}", inc.neighbors(k));
    }

    let m = hopcroft_karp(&inc);
    println!("matching size: {}", m.size);
    for (k, v) in m.row_to_var.iter().enumerate() {
        match v {
            Some(j) => println!("  row {k} ↔ var {j}"),
            None => println!("  row {k} unmatched"),
        }
    }
    assert_eq!(m.size, 3, "expected a perfect matching");
}
