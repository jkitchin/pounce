//! Worked example for PR 4 of issue #53 — Tarjan SCC + block
//! triangular form on the square sub-graph.
//!
//! 4 equality rows × 4 variables with a chain dependency structure:
//!
//! ```text
//! row 0 ↔ col 0
//! row 1 ↔ col 1, uses col 0
//! row 2 ↔ col 2, uses col 1
//! row 3 ↔ col 3, uses col 2
//! ```
//!
//! The dependency DAG is 3 → 2 → 1 → 0, so the BTF emits 4
//! size-1 blocks in order [0], [1], [2], [3].
//!
//! Run with:
//! ```bash
//! cargo run -p pounce-presolve --example btf_chain
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used)]

use pounce_presolve::btf::BlockTriangularForm;
use pounce_presolve::components::SquareComponents;
use pounce_presolve::dulmage_mendelsohn::DulmageMendelsohnPartition;
use pounce_presolve::incidence::{EqualityIncidence, ProbeView};
use pounce_presolve::matching::hopcroft_karp;

fn main() {
    let irow: [i32; 7] = [0, 1, 1, 2, 2, 3, 3];
    let jcol: [i32; 7] = [0, 0, 1, 1, 2, 2, 3];
    let g = [0.0; 4];

    let probe = ProbeView {
        n_vars: 4,
        m_rows: 4,
        jac_irow: &irow,
        jac_jcol: &jcol,
        jac_values: None,
        g_l: &g,
        g_u: &g,
        linearity: None,
        one_based: false,
        eq_tol: 1e-12,
        excluded_vars: None,
        excluded_rows: None,
    };
    let inc = EqualityIncidence::from_probe(&probe);
    let m = hopcroft_karp(&inc);
    let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
    let comps = SquareComponents::of_square_part(&inc, &m, &dm);

    println!("btf_chain example — PR 4 of pounce#53");
    println!("matching size: {}", m.size);
    println!("square components: {}", comps.components.len());
    assert_eq!(
        comps.components.len(),
        1,
        "expected one connected component"
    );

    let btf = BlockTriangularForm::of_component(&inc, &m, &comps.components[0]);
    println!("BTF blocks (elimination order):");
    for (i, b) in btf.blocks.iter().enumerate() {
        println!("  block {i}: rows={:?}, cols={:?}", b.eq_rows, b.cols);
    }
    assert_eq!(btf.blocks.len(), 4, "chain expands to 4 size-1 blocks");
    for (i, b) in btf.blocks.iter().enumerate() {
        assert_eq!(b.eq_rows, vec![i]);
        assert_eq!(b.cols, vec![i]);
    }
}
