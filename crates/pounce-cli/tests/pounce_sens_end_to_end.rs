//! End-to-end test for the `pounce_sens` binary against the
//! parametric_cpp problem from upstream sIPOPT.
//!
//! Runs the built binary on a hand-crafted
//! `tests/fixtures/parametric.nl` (a transliteration of upstream's
//! [`parametricTNLP.cpp`](../../../../ref/Ipopt/contrib/sIPOPT/examples/parametric_cpp/parametricTNLP.cpp)),
//! then parses the emitted `.sol` and confirms the `sens_sol_state_1`
//! suffix matches the upstream golden output captured in pounce#16
//! (`libsipopt-3.14.19`, run on 2026-05-14). Bypasses AMPL CE — the
//! `.nl` is written by hand, and upstream's `printf` text is the
//! reference rather than an `ipopt_sens`-produced `.sol`.
//!
//! AMPL-tooling-dependent integration (translating `hicks.mod` →
//! `hicks.nl` and diffing against an `ipopt_sens`-produced `.sol`) is
//! a follow-up — see the pounce#17 close-out comment.

use std::path::PathBuf;
use std::process::Command;

const UPSTREAM_X_PERTURBED_NOBC: [f64; 5] = [
    0.576_530_601_168_321_9,
    0.377_551_038_130_684_8,
    -0.045_918_360_700_993_31,
    4.500_000_000_000_000,
    1.000_000_000_000_000,
];

/// Build path to the `pounce_sens` binary in the workspace's
/// `target/<profile>/` directory. `CARGO_BIN_EXE_pounce_sens` is set
/// by cargo for any integration test in the same package as the
/// binary; matches the convention used by other pounce integration
/// tests.
fn pounce_sens_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce_sens"))
}

fn fixtures_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p
}

/// Parse the `sens_sol_state_1` real-var suffix block out of a `.sol`
/// file text. Returns a dense vector of length 5 (the parametric.nl
/// var count), zero-filled for slots not listed.
fn parse_sens_sol_state_1(sol: &str) -> Option<[f64; 5]> {
    let mut lines = sol.lines();
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("suffix ") {
            // header: "<kind> <count> <name>"
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() < 3 {
                continue;
            }
            if parts[2] != "sens_sol_state_1" {
                continue;
            }
            let count: usize = parts[1].parse().ok()?;
            let mut out = [0.0; 5];
            for _ in 0..count {
                let entry = lines.next()?;
                let mut toks = entry.split_whitespace();
                let idx: usize = toks.next()?.parse().ok()?;
                let val: f64 = toks.next()?.parse().ok()?;
                if idx < 5 {
                    out[idx] = val;
                }
            }
            return Some(out);
        }
    }
    None
}

#[test]
fn pounce_sens_matches_upstream_sipopt_on_parametric_cpp() {
    let nl = fixtures_dir().join("parametric.nl");
    assert!(nl.exists(), "fixture missing: {}", nl.display());

    // Write the output `.sol` to a tempfile alongside the fixture (
    // workspace target/ is writeable). Each test invocation gets a
    // fresh name to avoid races across `cargo test --jobs N`.
    let mut out = std::env::temp_dir();
    out.push(format!(
        "pounce_sens_parametric_{}.sol",
        std::process::id()
    ));

    let status = Command::new(pounce_sens_exe())
        .arg(&nl)
        .arg(&out)
        .status()
        .expect("spawn pounce_sens");
    assert!(status.success(), "pounce_sens exited with {status:?}");

    let sol_text = std::fs::read_to_string(&out).expect("read .sol");
    eprintln!("---- emitted .sol ----\n{sol_text}");

    let sens = parse_sens_sol_state_1(&sol_text)
        .expect("sens_sol_state_1 suffix present in .sol");

    eprintln!("pounce sens_sol_state_1  = {:?}", sens);
    eprintln!("upstream sens_sol_state_1 = {:?}", UPSTREAM_X_PERTURBED_NOBC);

    // pounce#16 acceptance bound: per-component agreement to 1e-8.
    for (k, (got, want)) in sens
        .iter()
        .zip(UPSTREAM_X_PERTURBED_NOBC.iter())
        .enumerate()
    {
        let err = (got - want).abs();
        assert!(
            err < 1e-8,
            "x_perturbed[{k}]: pounce={got} upstream={want} |err|={err} not < 1e-8",
        );
    }

    let _ = std::fs::remove_file(&out);
}
