//! Worked example for PR 3 of issue #53 — Dulmage-Mendelsohn
//! partition + connected components of the square sub-graph.
//!
//! Builds a 5 equality-row × 5 variable problem that decomposes into
//! three independent square blocks:
//!
//! ```text
//! block A: row 0 ↔ col 0
//! block B: rows 1, 2 share cols 1, 2
//! block C: rows 3, 4 share cols 3, 4
//! ```
//!
//! Runs Hopcroft-Karp, builds the DM partition, then asks for
//! connected components.
//!
//! Run with:
//! ```bash
//! cargo run -p pounce-presolve --example dm_partition
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used)]

use pounce_presolve::components::SquareComponents;
use pounce_presolve::dulmage_mendelsohn::DulmageMendelsohnPartition;
use pounce_presolve::incidence::{EqualityIncidence, ProbeView};
use pounce_presolve::matching::hopcroft_karp;

fn main() {
    let irow: [i32; 9] = [0, 1, 1, 2, 2, 3, 3, 4, 4];
    let jcol: [i32; 9] = [0, 1, 2, 1, 2, 3, 4, 3, 4];
    let g = [0.0; 5];

    let probe = ProbeView {
        n_vars: 5,
        m_rows: 5,
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

    println!("dm_partition example — PR 3 of pounce#53");
    println!("matching size:    {}", m.size);
    println!("over rows: {:?}, cols: {:?}", dm.over_rows, dm.over_cols);
    println!(
        "square rows: {:?}, cols: {:?}",
        dm.square_rows, dm.square_cols
    );
    println!("under rows: {:?}, cols: {:?}", dm.under_rows, dm.under_cols);

    let comps = SquareComponents::of_square_part(&inc, &m, &dm);
    println!("square components: {}", comps.components.len());
    for (i, c) in comps.components.iter().enumerate() {
        println!("  component {i}: rows={:?}, cols={:?}", c.eq_rows, c.cols);
    }

    assert_eq!(comps.components.len(), 3, "expected 3 independent blocks");
}
