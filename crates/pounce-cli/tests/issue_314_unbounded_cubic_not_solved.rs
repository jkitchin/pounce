//! gh #314 — an unbounded-below NLP whose recession ray increases an
//! *inequality's slack* must not be reported in the AMPL *solved* family.
//!
//! Fixture `unbounded_cubic.nl` is
//!
//! ```text
//! min  -(x1^3 + x2^3)   s.t.  x1 + x2 >= 1,   x1, x2 free
//! ```
//!
//! It is unbounded below: the feasible ray `x1 = t, x2 = 0` (`t >= 1`) drives
//! the objective `-t^3 -> -inf`. This is the same *family* as #274 (an
//! unbounded solve must never be advertised as solved), but a distinct trigger:
//! the recession direction `d = (1, 1)` is feasible, strictly lowers the
//! objective, and *increases the slack* of the inequality `x1 + x2 >= 1`
//! (moves deeper into the feasible region).
//!
//! The core defect this pins: the adversary bot (running a stale pre-fix
//! binary) observed this land in the AMPL solved family
//! (`solve_result_num = 100`, `SolvedToAcceptableLevel`, exit 0), so a
//! Pyomo/AMPL caller silently loaded a diverging iterate
//! (`x ~ 5.8e18`, `obj ~ -3.9e56`) as `optimal`.
//!
//! On current main the "reported as solved" defect is already gone. #314
//! additionally makes the CLI's verdict the *correct* one: `DivergingIterates`
//! (the AMPL 300 "unbounded" range), matching what the library reports for the
//! same model. The #285 checked recession-ray proof already handled a
//! recession ray in `null(A_eq)`; a variable-swap bug in its
//! `recession_blocked_by_inequality` gate inverted the bound semantics and
//! spuriously rejected a ray that *increases* an inequality's slack. Fixing
//! that swap completes the #274 family for the inequality-slack recession
//! shape.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("unbounded_cubic.nl");
    p
}

fn solve_result_num(text: &str) -> i32 {
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("objno ") {
            if let Some(code) = rest.split_whitespace().nth(1) {
                return code.parse().expect("objno code parses");
            }
        }
    }
    panic!("no `objno` line in .sol:\n{text}");
}

/// The core assertion (Deliverable A): the `.sol` must not claim the solve
/// succeeded. A `solve_result_num` in `0..=199` (0..99 solved, 100..199
/// solved-with-warning) makes Pyomo report `TerminationCondition.optimal` and
/// load the diverging iterate.
#[test]
fn unbounded_cubic_is_not_reported_as_solved() {
    let dir = std::env::temp_dir();
    let sol = dir.join("pounce_issue314_unbounded_cubic.sol");
    let _ = std::fs::remove_file(&sol);

    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
        .output()
        .expect("spawn pounce");

    // Under `-AMPL` the termination travels in the file, so exit stays 0.
    assert_eq!(out.status.code(), Some(0), "-AMPL must exit 0");

    let text = std::fs::read_to_string(&sol).expect("read .sol");
    let srn = solve_result_num(&text);

    // Deliverable A — never in the solved family.
    assert!(
        !(0..200).contains(&srn),
        "unbounded NLP reported in the AMPL solved family (solve_result_num={srn}); \
         0..99 = solved and 100..199 = solved-with-warning both make Pyomo report \
         TerminationCondition.optimal and load the diverging iterate:\n{text}"
    );

    // Deliverable B — the correct verdict is unbounded (DivergingIterates),
    // which maps to the AMPL 300..399 "unbounded" range.
    assert!(
        (300..400).contains(&srn),
        "expected the unbounded-family solve_result_num (300..399, \
         DivergingIterates), got {srn}:\n{text}"
    );

    assert!(
        !text.contains("SolvedToAcceptableLevel") && !text.contains("Optimal"),
        "an unbounded solve must not be labelled solved/optimal:\n{text}"
    );
}

/// The CLI's verdict must match what the library reports for the same model,
/// and must exit non-zero outside `-AMPL` mode. Before #314 the CLI wrote
/// `ErrorInStepComputation` (srn 500) while the library reported
/// `Diverging_Iterates` — the two surfaces disagreed about *why* the solve
/// failed.
#[test]
fn cli_reports_diverging_and_exits_nonzero() {
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .output()
        .expect("spawn pounce");

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(
        combined.contains("diverging") || combined.contains("Diverging"),
        "expected the DivergingIterates verdict the library reports:\n{combined}"
    );
    assert!(
        !combined.contains("Solved To Acceptable Level"),
        "must not claim acceptable-level convergence:\n{combined}"
    );
    assert_ne!(
        out.status.code(),
        Some(0),
        "a failed solve must exit non-zero outside -AMPL mode"
    );
}
