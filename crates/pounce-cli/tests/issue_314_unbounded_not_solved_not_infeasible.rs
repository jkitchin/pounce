//! gh #314 — an unbounded-below NLP must land in neither the AMPL *solved*
//! family nor a false *infeasible* verdict.
//!
//! The problem is `min -Σxᵢ³  s.t.  Σxᵢ ≥ 1` with every `xᵢ` free. It is
//! unbounded below (along the ray `x₁ = t, x₂ = … = 0` the objective is
//! `-t³ → -∞`) and *trivially feasible* (`x = (1, 0, …, 0)` satisfies the
//! constraint). So there is nothing for a "solved" status to refer to, and the
//! problem is emphatically not infeasible.
//!
//! This is the same failure family as #274 (an unbounded solve mislabelled
//! solved), but a different trigger. On the summed-cubes shape the unbounded
//! objective drags the iterate out to `|x| ~ 1e16..1e19` with mixed signs,
//! where the restoration sub-IPM stalls and reports *local infeasibility*.
//! Before #314 that surfaced as `Infeasible_Problem_Detected`
//! (`solve_result_num = 200`) — a false certificate: a caller is told the
//! problem has no feasible point when it plainly does. The fix rejects a
//! restoration `LocallyInfeasible` verdict issued from a diverged iterate
//! (descent on the violation still available, or the iterate structurally free
//! to run to infinity at a magnitude no genuine infeasibility point occupies)
//! and reclassifies it to the honest failure family instead.
//!
//! The two fixtures exercise both restoration exits on this shape: `n = 2`
//! reaches restoration's `Failed` arm (an `Error_In_Step_Computation`,
//! `srn = 500`), while `n = 4` reaches its `LocallyInfeasible` arm — the path
//! #314 fixes. Both must exit non-zero with a `solve_result_num` outside both
//! the solved range (0..199) and the infeasible range (200..299).

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

/// Core property: unbounded-below-but-feasible → not solved, not infeasible.
fn assert_not_solved_not_infeasible(fixture_name: &str) {
    let dir = std::env::temp_dir();
    let sol = dir.join(format!("pounce_issue314_{fixture_name}.sol"));
    let _ = std::fs::remove_file(&sol);

    let out = Command::new(pounce_exe())
        .arg(fixture(fixture_name))
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
        "{fixture_name}: unbounded-below NLP reported in the AMPL solved family \
         (solve_result_num={srn}); 0..99 = solved and 100..199 = solved-with-warning \
         both make Pyomo report TerminationCondition.optimal and load the diverging \
         iterate:\n{text}"
    );
    assert!(
        !(200..300).contains(&srn),
        "{fixture_name}: a trivially feasible problem (x = (1, 0, …, 0) satisfies \
         Σxᵢ ≥ 1) reported as infeasible (solve_result_num={srn}); the unbounded \
         objective merely dragged the iterate into a numerically degenerate region \
         where restoration stalled:\n{text}"
    );
    // The honest verdict is the unbounded (300..399) or failure (500..599) family.
    assert!(
        (300..400).contains(&srn) || (500..600).contains(&srn),
        "{fixture_name}: expected an unbounded (300..399) or failure (500..599) \
         solve_result_num, got {srn}:\n{text}"
    );

    assert!(
        !text.contains("SolvedToAcceptableLevel"),
        "{fixture_name}: an unbounded solve must not be labelled \
         SolvedToAcceptableLevel:\n{text}"
    );
}

#[test]
fn sum_cubes_n2_is_not_solved_not_infeasible() {
    // Restoration `Failed` arm.
    assert_not_solved_not_infeasible("unbounded_sum_cubes_n2.nl");
}

#[test]
fn sum_cubes_n4_is_not_solved_not_infeasible() {
    // Restoration `LocallyInfeasible` arm — the path #314 fixes. Before the fix
    // this returned Infeasible_Problem_Detected (srn=200) on a feasible problem.
    assert_not_solved_not_infeasible("unbounded_sum_cubes_n4.nl");
}

/// Outside `-AMPL` mode the failure must also surface a non-zero exit — a
/// downstream shell/CI check keys on it.
#[test]
fn sum_cubes_n4_exits_nonzero_outside_ampl() {
    let out = Command::new(pounce_exe())
        .arg(fixture("unbounded_sum_cubes_n4.nl"))
        .arg("--no-sol")
        .output()
        .expect("spawn pounce");

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    assert_ne!(
        out.status.code(),
        Some(0),
        "an unbounded/failed solve must exit non-zero outside -AMPL mode:\n{combined}"
    );
    assert!(
        !combined.contains("Infeasible Problem Detected")
            && !combined.contains("InfeasibleProblemDetected"),
        "a trivially feasible problem must not be reported infeasible:\n{combined}"
    );
}
