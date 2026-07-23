//! Bound-multiplier (`ipopt_zL_out` / `ipopt_zU_out`) `.sol` suffix guard
//! for issue #296.
//!
//! ## Why this test exists
//!
//! pounce advertises drop-in Ipopt compatibility, and Ipopt writes the
//! reduced costs / bound sensitivities into the AMPL `.sol` as the
//! `ipopt_zL_out` / `ipopt_zU_out` variable-suffix blocks (Pyomo surfaces
//! them as `model.ipopt_zL_out[var]`). pounce used to write NO suffix blocks
//! at all, so a user migrating from Ipopt read `None` back with no error
//! (#296). This test pins the presence, the sparse layout, AND — most
//! importantly — the SIGN of those blocks against an analytic reference, on
//! BOTH `.sol`-producing solve paths (the NLP interior-point path and the
//! convex QP path), so a future regression fails loudly.
//!
//! ## Analytic reference (hand-computable)
//!
//! `bound_active_qp.nl`: `min (x−3)² + (y+2)²  s.t.  0 ≤ x ≤ 1, −1 ≤ y ≤ 1`.
//! The unconstrained minimum is `(3, −2)`, so both bounds bind:
//!   * `x* = 1` (upper bound active). `∂f/∂x = 2(x−3) = −4` at `x=1`.
//!   * `y* = −1` (lower bound active). `∂f/∂y = 2(y+2) = +2` at `y=−1`.
//!
//! Ipopt's `.sol` convention (verified numerically against Ipopt 3.14 while
//! building #296): both suffix values equal the objective-gradient component
//! at the bound, i.e.
//!   * `ipopt_zL_out = +z_l` (≥ 0 at an active lower bound) → `zL_out[y] = +2`,
//!   * `ipopt_zU_out = −z_u` (≤ 0 at an active upper bound) → `zU_out[x] = −4`.
//! The inactive side of each variable is ≈ 0 and is sparse-trimmed away.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// One parsed `.sol` suffix block: its name and the sparse `(index, value)`
/// entries (real-valued suffixes only).
struct SolSuffixBlock {
    name: String,
    entries: Vec<(usize, f64)>,
}

/// Extract every real-valued variable-suffix block from an AMPL `.sol`.
///
/// A suffix block is the five-integer header line
/// `suffix <kind> <nvalues> <namelen> 0 0`, the suffix name on the next
/// line, then `<nvalues>` `<index> <value>` pairs. We only need the name +
/// entries here; `kind`/`namelen` are not asserted.
fn parse_sol_suffixes(text: &str) -> Vec<SolSuffixBlock> {
    let mut blocks = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let mut it = line.split_whitespace();
        if it.next() != Some("suffix") {
            continue;
        }
        // kind, nvalues, namelen, tablen, tabline
        let _kind: u32 = it.next().unwrap().parse().unwrap();
        let nvalues: usize = it.next().unwrap().parse().unwrap();
        let name = lines.next().expect("suffix name line").trim().to_string();
        let mut entries = Vec::with_capacity(nvalues);
        for _ in 0..nvalues {
            let e = lines.next().expect("suffix entry line");
            let mut p = e.split_whitespace();
            let idx: usize = p.next().unwrap().parse().unwrap();
            let val: f64 = p.next().unwrap().parse().unwrap();
            entries.push((idx, val));
        }
        blocks.push(SolSuffixBlock { name, entries });
    }
    blocks
}

/// Solve `bound_active_qp.nl` via the CLI with the given `solver_selection`,
/// returning the parsed `.sol` suffix blocks.
fn solve_bound_active(selection: &str, tag: &str) -> Vec<SolSuffixBlock> {
    let dir = std::env::temp_dir().join(format!("pounce_i296_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let nl = dir.join("m.nl");
    std::fs::copy(fixture("bound_active_qp.nl"), &nl).expect("copy fixture");
    let sol = dir.join("m.sol");

    let out = Command::new(pounce_exe())
        .arg(&nl)
        .arg(format!("solver_selection={selection}"))
        .arg("--sol-output")
        .arg(&sol)
        .output()
        .expect("spawn pounce");
    assert_eq!(
        out.status.code(),
        Some(0),
        "solve should succeed ({selection}); stderr:\n{}",
        String::from_utf8_lossy(&out.stderr),
    );

    let sol_text = std::fs::read_to_string(&sol).expect("read .sol");
    parse_sol_suffixes(&sol_text)
}

/// Assert the two bound-multiplier blocks are present with the analytic
/// values: `ipopt_zU_out[x=0] = −4` and `ipopt_zL_out[y=1] = +2`.
fn assert_bound_multipliers(blocks: &[SolSuffixBlock], path: &str) {
    let zl = blocks
        .iter()
        .find(|b| b.name == "ipopt_zL_out")
        .unwrap_or_else(|| panic!("{path}: .sol must contain an ipopt_zL_out block"));
    let zu = blocks
        .iter()
        .find(|b| b.name == "ipopt_zU_out")
        .unwrap_or_else(|| panic!("{path}: .sol must contain an ipopt_zU_out block"));

    // Upper bound active on x (variable index 0): zU_out = −4 (NOT +4).
    let (_, zu_x) = zu
        .entries
        .iter()
        .find(|(i, _)| *i == 0)
        .copied()
        .unwrap_or_else(|| panic!("{path}: ipopt_zU_out must carry an entry for x (index 0)"));
    assert!(
        (zu_x - (-4.0)).abs() < 1e-4,
        "{path}: ipopt_zU_out[x] must be −4 (Ipopt convention: −z_u at an \
         active upper bound); got {zu_x}",
    );

    // Lower bound active on y (variable index 1): zL_out = +2.
    let (_, zl_y) = zl
        .entries
        .iter()
        .find(|(i, _)| *i == 1)
        .copied()
        .unwrap_or_else(|| panic!("{path}: ipopt_zL_out must carry an entry for y (index 1)"));
    assert!(
        (zl_y - 2.0).abs() < 1e-4,
        "{path}: ipopt_zL_out[y] must be +2 (Ipopt convention: +z_l at an \
         active lower bound); got {zl_y}",
    );

    // The inactive sides are ≈ 0 and sparse-trimmed: x has no active lower
    // bound and y no active upper bound, so those slots must be absent (or
    // negligible if present).
    if let Some((_, v)) = zl.entries.iter().find(|(i, _)| *i == 0) {
        assert!(
            v.abs() < 1e-4,
            "{path}: ipopt_zL_out[x] should be ≈0; got {v}"
        );
    }
    if let Some((_, v)) = zu.entries.iter().find(|(i, _)| *i == 1) {
        assert!(
            v.abs() < 1e-4,
            "{path}: ipopt_zU_out[y] should be ≈0; got {v}"
        );
    }
}

/// The NLP interior-point path (`solver_selection=nlp`) must write the
/// bound-multiplier suffixes with the Ipopt sign convention.
#[test]
fn nlp_path_writes_bound_multiplier_suffixes() {
    let blocks = solve_bound_active("nlp", "nlp");
    assert_bound_multipliers(&blocks, "NLP path");
}

/// The convex QP path (`solver_selection=qp-ipm`) must write the same
/// bound-multiplier suffixes — the multipliers are recovered from the folded
/// variable-bound `G` rows — so parity is consistent across solve paths.
#[test]
fn qp_path_writes_bound_multiplier_suffixes() {
    let blocks = solve_bound_active("qp-ipm", "qp");
    assert_bound_multipliers(&blocks, "QP path");
}
