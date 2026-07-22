//! gh #274 — an unbounded NLP must not be reported in the AMPL *solved*
//! family.
//!
//! Fixture `unbounded_exp.nl` is `min -exp(x)  s.t.  x >= 0` with `x` free
//! (the bound is a constraint row, not a variable bound). It is unbounded
//! below.
//!
//! The failure mode this pins: the near-feasible restoration re-entry
//! detector in `IpoptAlgorithm::invoke_restoration` declared the point
//! `Solved_To_Acceptable_Level` on the strength of the *primal* residual
//! alone. Here the constraint stays satisfied (`inf_pr ≈ 1.7e-10`) while
//! the iterates run off toward `-inf` (`inf_du ≈ 8.8e+47`), so the detector
//! tripped and the `.sol` carried `solve_result_num = 100`.
//!
//! That lands in AMPL's *solved* range (0..99 solved, 100..199 solved with
//! warning), which Pyomo maps to `TerminationCondition.optimal` — so a
//! Pyomo/AMPL/GAMS caller silently received a diverging iterate
//! (`x ≈ 110.4`, `obj ≈ -8.8e47`) labelled optimal.
//!
//! Note this does not assert the *unbounded* range (300..399). pounce
//! cannot prove unboundedness on this instance — `diverging_iterates_tol`
//! defaults to `1e20` and the iterate only reaches `~110` before the step
//! computation breaks down — so the honest verdict is a failure status.
//! Tightening that classification is gh #273's scope. What matters here is
//! that the result is not advertised as solved.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("unbounded_exp.nl");
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

/// The core assertion: the `.sol` must not claim the solve succeeded.
#[test]
fn unbounded_nlp_is_not_reported_as_solved() {
    let dir = std::env::temp_dir();
    let sol = dir.join("pounce_issue274_unbounded.sol");
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

    assert!(
        !(0..200).contains(&srn),
        "unbounded NLP reported in the AMPL solved family (solve_result_num={srn}); \
         0..99 = solved and 100..199 = solved-with-warning both make Pyomo report \
         TerminationCondition.optimal and load the diverging iterate:\n{text}"
    );

    // It should be a genuine failure verdict: 500..599.
    assert!(
        (500..600).contains(&srn),
        "expected a failure-family solve_result_num (500..599), got {srn}:\n{text}"
    );

    assert!(
        !text.contains("SolvedToAcceptableLevel"),
        "an unbounded solve must not be labelled SolvedToAcceptableLevel:\n{text}"
    );
}

/// The CLI's verdict must match what the library reports for the same model.
/// Before #274 the library said `Error_In_Step_Computation` while the CLI
/// wrote `SolvedToAcceptableLevel` — the two surfaces disagreed about
/// whether the solve had succeeded at all.
#[test]
fn cli_status_matches_library_status_on_unbounded() {
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
        combined.contains("Error in step computation")
            || combined.contains("ErrorInStepComputation"),
        "expected the same Error_In_Step_Computation the library reports:\n{combined}"
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
