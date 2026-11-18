//! `nlp_scaling_untagged_constr` through the CLI, on an `.nl` Pyomo wrote.
//!
//! `fixtures/untagged_constr_scaling.nl` has two equality rows at the
//! start `x = (1, 1, 1)`:
//!
//! * `tagged: 2 x0 + x1 = 0`, residual 3, `scaling_factor` 0.5;
//! * `untagged: 1000 x0 + x1 = 5`, residual 996, no `scaling_factor` entry;
//!
//! and `scaling_factor` 2 on `x0`. Stopped at `max_iter=0`, the summary's
//! scaled constraint violation is the larger of the two scaled residuals,
//! which reads off the factor the untagged row got:
//!
//! * `none` leaves it at 1, so the scaled violation is 996;
//! * `gradient-based` takes the row's Jacobian after `x0`'s factor, where
//!   its largest entry is 1000 / 2 = 500, giving 100 / 500 = 0.2 and a
//!   scaled violation of 199.2. Sampling before the variable factor would
//!   give 0.1 and 99.6.
//!
//! With the suffix segments removed, no row has a factor, so the rule
//! scales both: 99.6 for the untagged row (no variable factor now) and 1
//! for the tagged one, whose largest entry is 2.

use std::path::PathBuf;
use std::process::Command;

fn fixture() -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/untagged_constr_scaling.nl");
    std::fs::read_to_string(p).expect("read fixture")
}

/// Run `nl` stopped at the start and return the summary's scaled and
/// unscaled constraint violation.
fn violation(nl: &str, tag: &str, opts: &[&str]) -> (f64, f64) {
    let dir = std::env::temp_dir().join(format!("pounce_untagged_{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("m.nl");
    std::fs::write(&path, nl).expect("write model");
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(&path)
        .arg("--no-sol")
        .args(["solver_selection=nlp", "max_iter=0"])
        .args(opts)
        .output()
        .expect("run pounce");
    let _ = std::fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let line = stdout
        .lines()
        .find(|l| l.starts_with("Constraint violation"))
        .unwrap_or_else(|| panic!("no violation line; stdout=\n{stdout}"));
    let v: Vec<f64> = line
        .split(':')
        .nth(1)
        .expect("a colon")
        .split_whitespace()
        .map(|t| t.parse().expect("a number"))
        .collect();
    (v[0], v[1])
}

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-9 * want.abs()
}

#[test]
fn the_untagged_row_gets_the_gradient_based_factor_after_variable_scaling() {
    let nl = fixture();
    let (scaled, unscaled) = violation(&nl, "none", &["nlp_scaling_method=user-scaling"]);
    assert!(close(unscaled, 996.0), "unscaled {unscaled}");
    assert!(
        close(scaled, 996.0),
        "default leaves the row at 1: {scaled}"
    );

    let (scaled, unscaled) = violation(
        &nl,
        "gb",
        &[
            "nlp_scaling_method=user-scaling",
            "nlp_scaling_untagged_constr=gradient-based",
        ],
    );
    assert!(close(unscaled, 996.0), "unscaled {unscaled}");
    assert!(
        close(scaled, 199.2),
        "factor 0.2 from the scaled column: {scaled}"
    );
}

#[test]
fn a_model_with_no_scaling_factor_suffix_gets_the_rule_on_every_row() {
    // Drop the S4 and S5 segments: each is a header and one entry.
    let text = fixture();
    let lines: Vec<&str> = text.lines().collect();
    let mut kept = Vec::new();
    let mut skip = 0;
    for l in lines {
        if skip > 0 {
            skip -= 1;
            continue;
        }
        if l.starts_with("S4 1 scaling_factor") || l.starts_with("S5 1 scaling_factor") {
            skip = 1;
            continue;
        }
        kept.push(l);
    }
    let nl = kept.join("\n") + "\n";
    assert!(!nl.contains("scaling_factor"));

    let (scaled, _) = violation(&nl, "bare_none", &["nlp_scaling_method=user-scaling"]);
    assert!(
        close(scaled, 996.0),
        "no suffix, no rule: unscaled, {scaled}"
    );
    let (scaled, _) = violation(
        &nl,
        "bare_gb",
        &[
            "nlp_scaling_method=user-scaling",
            "nlp_scaling_untagged_constr=gradient-based",
        ],
    );
    assert!(
        close(scaled, 99.6),
        "factor 0.1 on the untagged row: {scaled}"
    );
}

#[test]
fn the_option_is_inert_under_another_scaling_method() {
    let nl = fixture();
    let base = violation(&nl, "gb_method", &["nlp_scaling_method=gradient-based"]);
    let with = violation(
        &nl,
        "gb_method_opt",
        &[
            "nlp_scaling_method=gradient-based",
            "nlp_scaling_untagged_constr=gradient-based",
        ],
    );
    assert_eq!(base, with);
}
