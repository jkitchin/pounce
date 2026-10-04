//! gh #990 item 12: with `solver_selection=lp-ipm qp_crossover=yes` the console
//! log at the default print level had no line that mentioned crossover,
//! simplex or pivots. It now prints one: the engine, whether the purified
//! vertex replaced the interior iterate, the superbasics pushed, the pivots by
//! stage, and the KKT error before and after.
//!
//! Without `qp_crossover` nothing changes -- no line, no cost.

use std::process::Command;

fn run(extra: &[&str]) -> String {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/lp_afiro.nl");
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(path)
        .args(["--no-sol", "--no-options-file", "solver_selection=lp-ipm"])
        .args(extra)
        .output()
        .expect("run pounce");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn crossover_prints_a_one_line_summary() {
    let text = run(&["qp_crossover=yes"]);
    let line = text
        .lines()
        .find(|l| l.starts_with("Crossover:"))
        .unwrap_or_else(|| panic!("no Crossover line in:\n{text}"));
    assert!(line.contains("simplex engine"), "{line}");
    assert!(line.contains("vertex accepted"), "{line}");
    assert!(line.contains("superbasic(s) pushed"), "{line}");
    assert!(line.contains("pivot(s) (push "), "{line}");
    assert!(line.contains("KKT error"), "{line}");
    assert!(text.contains("EXIT: Optimal Solution Found"), "{text}");
}

#[test]
fn no_crossover_line_when_crossover_is_off() {
    let text = run(&[]);
    assert!(!text.contains("Crossover:"), "{text}");
}
