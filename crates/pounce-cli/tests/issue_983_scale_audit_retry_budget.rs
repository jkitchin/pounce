//! The gh#983 scale audit's re-solve is budgeted by the base solve it audits.
//!
//! `scale_audit_retry_meyer3` is CUTE's `meyer3` (Vanderbei's AMPL
//! transliteration): the base solve ends `Solve_Succeeded` in 256 iterations
//! with a small-objective verdict, and the up-scaled re-solve never converges.
//! Before the budget it ran to the caller's `max_iter` -- 3000, then 3000
//! again through its own mu flip -- before being declined. On the `brainpc`
//! family the same shape turned a 1-2 s solve into a timeout under the
//! benchmark's 300 s wall, losing the base answer.
//!
//! The re-solve now gets `max(3 * base iterations, 50)`, capped by the
//! caller's `max_iter`, runs once (no mu flip of its own), and the base answer
//! is kept when it is declined.

use std::path::PathBuf;
use std::process::Command;

fn run(extra: &[&str]) -> String {
    let mut fx = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fx.push("tests");
    fx.push("fixtures");
    fx.push("scale_audit_retry_meyer3.nl");
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fx)
        .arg("--no-options-file")
        .args(extra)
        .output()
        .expect("run pounce");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Every "Number of Iterations" summary, in order: the base solve, then the
/// re-solve.
fn iteration_counts(out: &str) -> Vec<i64> {
    out.lines()
        .filter_map(|l| l.strip_prefix("Number of Iterations....:"))
        .map(|v| v.trim().parse().expect("iteration count"))
        .collect()
}

#[test]
fn a_declined_re_solve_is_capped_at_three_times_the_base_solve() {
    let out = run(&[]);
    let its = iteration_counts(&out);
    assert_eq!(
        its.len(),
        2,
        "expected one base solve and one re-solve:\n{out}"
    );
    let base = its[0];
    assert!(
        base > 50,
        "the floor must not be what binds here: base {base}\n{out}"
    );
    for &r in &its[1..] {
        assert!(
            r <= 3 * base,
            "re-solve ran {r} iterations against base {base}\n{out}"
        );
    }
    assert!(out.contains("gh#983 scale audit: declined"), "{out}");
    assert!(out.contains("EXIT: Optimal Solution Found."), "{out}");
}

#[test]
fn the_caller_max_iter_still_caps_the_re_solve() {
    let out = run(&["max_iter=300"]);
    let its = iteration_counts(&out);
    assert!(
        its.len() >= 2,
        "expected a base solve and a re-solve:\n{out}"
    );
    assert!(its[0] <= 300, "{out}");
    for &r in &its[1..] {
        assert!(
            r <= 300,
            "re-solve ran {r} iterations against max_iter=300\n{out}"
        );
    }
}

/// The declined re-solve's iterations are counted (#1140 review). The answer
/// is the base attempt's, and so is `iteration_count`, but the solve spent
/// both attempts: `retry_total_iterations` must be their sum, read against
/// the two printed summaries, the one place each attempt's count survives.
#[test]
fn a_declined_re_solve_is_counted_in_the_total() {
    let json = std::env::temp_dir().join(format!("pounce-983-total-{}.json", std::process::id()));
    let out = run(&["--json-output", json.to_str().unwrap()]);
    let its = iteration_counts(&out);
    assert_eq!(its.len(), 2, "{out}");
    assert!(out.contains("gh#983 scale audit: declined"), "{out}");
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    let st = &report["statistics"];
    assert_eq!(
        st["iteration_count"], its[0],
        "the base attempt's is reported"
    );
    assert_eq!(
        st["retry_total_iterations"],
        its[0] + its[1],
        "counts {its:?}"
    );
    let _ = std::fs::remove_file(&json);
}
