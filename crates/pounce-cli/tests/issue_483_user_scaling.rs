//! End-to-end guard for gh#483: the `.nl` `scaling_factor` suffixes are
//! the AMPL/Pyomo channel for `nlp_scaling_method=user-scaling`, and
//! pounce used to read none of them.
//!
//! Before this, `NlTnlp` did not implement `get_scaling_parameters` at
//! all, so the default `TNLP` impl answered "no scaling supplied": a
//! Pyomo user who tagged `model.scaling_factor` and set
//! `nlp_scaling_method=user-scaling` — the workflow that works with
//! Ipopt through ASL — got no scaling and no message saying so. The
//! option was accepted and meant "none".
//!
//! Fixtures (both written by Pyomo's NL writer, so the suffix layout is
//! the real one, not a hand-rolled approximation):
//!
//! ```text
//! min (x1 - 2)^4 + (x2 - 3)^2   s.t.  x1*x2 >= 1,  x1 - x2 == 0.5
//! scaling_factor[obj] = 100,  scaling_factor[c1] = 10
//! ```
//!
//! * `user_scaling_suffix.nl` — objective + constraint factors only.
//! * `user_scaling_var_suffix.nl` — the same, plus
//!   `scaling_factor[x1] = 3`, a per-variable factor. It is applied as
//!   a change of variables, so it must not move the answer.

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

/// Run pounce on a fixture with extra key=value options; returns
/// `(success, stdout, stderr)`. The `.nl` is copied into a per-test
/// scratch directory first so the `.sol` the solver writes next to it
/// cannot collide between concurrently running tests.
fn run(fixture_name: &str, tag: &str, opts: &[&str]) -> (bool, String, String) {
    let dir = std::env::temp_dir().join(format!("pounce_gh483_{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join(fixture_name);
    std::fs::copy(fixture(fixture_name), &nl).expect("copy fixture");

    let out = Command::new(pounce_exe())
        .arg(&nl)
        .args(opts)
        .output()
        .expect("run pounce");
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The `(scaled, unscaled)` objective pair from the end-of-run summary
/// block, which is exactly where "was the scaling applied" is visible.
fn objective_columns(stdout: &str) -> (f64, f64) {
    let line = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("Objective."))
        .unwrap_or_else(|| panic!("no Objective summary line in:\n{stdout}"));
    let nums: Vec<f64> = line
        .split_whitespace()
        .filter_map(|t| t.parse::<f64>().ok())
        .collect();
    assert_eq!(nums.len(), 2, "expected scaled + unscaled columns: {line}");
    (nums[0], nums[1])
}

/// With `user-scaling`, the objective the IPM sees is the user's
/// factor times the model's — and the value handed back is still in
/// model units. Pre-fix both columns read the same, because nothing
/// ever read the suffix.
#[test]
fn scaling_factor_suffix_is_applied() {
    let (ok, stdout, stderr) = run(
        "user_scaling_suffix.nl",
        "applied",
        &["nlp_scaling_method=user-scaling"],
    );
    assert!(ok, "solve failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
    let (scaled, unscaled) = objective_columns(&stdout);
    assert!(
        (scaled / unscaled - 100.0).abs() < 1e-6,
        "scaling_factor[obj]=100 should scale the IPM's objective; \
         got scaled={scaled}, unscaled={unscaled}",
    );
}

/// The same file without the option is untouched: the suffix alone
/// changes nothing, matching Ipopt (and every model that carries a
/// `scaling_factor` Suffix for `core.scale_model` rather than for the
/// solver).
#[test]
fn scaling_factor_suffix_is_inert_without_the_option() {
    let (ok, stdout, stderr) = run("user_scaling_suffix.nl", "inert", &[]);
    assert!(ok, "solve failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
    let (scaled, unscaled) = objective_columns(&stdout);
    assert!(
        (scaled - unscaled).abs() < 1e-9,
        "default scaling should leave this objective alone; \
         got scaled={scaled}, unscaled={unscaled}",
    );
}

/// A per-variable factor is applied as a change of variables, which
/// conditions the problem without redefining it. The check is parity
/// against the same model with no variable factor: same objective in
/// the user's own units, and the objective factor still reaching the
/// IPM. Stage 1 refused this file outright.
#[test]
fn variable_scaling_factor_is_applied_without_moving_the_answer() {
    let (ok, stdout, stderr) = run(
        "user_scaling_var_suffix.nl",
        "varfactor",
        &["nlp_scaling_method=user-scaling"],
    );
    assert!(
        ok,
        "solve failed
stdout:
{stdout}
stderr:
{stderr}"
    );
    let (scaled, unscaled) = objective_columns(&stdout);

    let (ok_ref, stdout_ref, stderr_ref) = run(
        "user_scaling_suffix.nl",
        "varfactor_ref",
        &["nlp_scaling_method=user-scaling"],
    );
    assert!(
        ok_ref,
        "reference solve failed
stdout:
{stdout_ref}
stderr:
{stderr_ref}"
    );
    let (_, unscaled_ref) = objective_columns(&stdout_ref);

    assert!(
        (unscaled - unscaled_ref).abs() <= 1e-6 * unscaled_ref.abs().max(1.0),
        "scaling_factor[x1]=3 must not move the objective in user units; \
         got {unscaled} with the factor, {unscaled_ref} without",
    );
    assert!(
        (scaled / unscaled - 100.0).abs() < 1e-6,
        "scaling_factor[obj]=100 should still reach the IPM alongside \
         the variable factor; got scaled={scaled}, unscaled={unscaled}",
    );
}

/// Without `user-scaling` the factors are not consulted at all, so
/// the variable entry is inert and the solve runs unscaled.
#[test]
fn variable_scaling_factor_is_inert_without_the_option() {
    let (ok, stdout, stderr) = run("user_scaling_var_suffix.nl", "varinert", &[]);
    assert!(ok, "solve failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
}

/// A factor that cannot be applied is refused, and the refusal is
/// *readable*.
///
/// The message text is the assertion here, not decoration. It has been
/// wrong twice: once with its line continuations lost, so it printed
/// two runs of eighteen spaces mid-sentence, and once with the
/// continuations replaced by `\n` escapes, which printed real newlines
/// and the source indentation. Nothing caught either, because nothing
/// exercised this path. So this checks the sentence arrives whole, on
/// one line, with no run of source indentation in it.
#[test]
fn an_inapplicable_factor_is_refused_readably() {
    let (ok, stdout, stderr) = run(
        "user_scaling_bad_var_suffix.nl",
        "badfactor",
        &["nlp_scaling_method=user-scaling"],
    );
    assert!(
        !ok,
        "a negative factor must fail the solve\nstdout:\n{stdout}"
    );

    let line = stderr
        .lines()
        .find(|l| l.contains("nlp_scaling_method=user-scaling supplied"))
        .unwrap_or_else(|| panic!("no refusal on stderr:\n{stderr}"));

    // Whole sentences, not fragments separated by swallowed newlines.
    assert!(
        line.contains("supplied per-variable scaling factors that cannot be applied."),
        "the opening clause is broken up: {line}"
    );
    assert!(
        line.contains("Correct the factors, or drop nlp_scaling_method=user-scaling."),
        "the closing clause is broken up: {line}"
    );
    // Source indentation leaking into the output looks like this.
    assert!(
        !line.contains("   "),
        "the message carries a run of source indentation: {line}"
    );
    // The message carries its own terminator: the caller uses
    // `eprint!`, so without it the next line runs straight on.
    assert!(
        stderr.contains(
            "or drop nlp_scaling_method=user-scaling.
"
        ),
        "the refusal is not newline-terminated:
{stderr}"
    );
    // And it says which factor, and why.
    assert!(
        line.contains("-3") && line.contains("finite and positive"),
        "the refusal should name the offending factor and the rule: {line}"
    );
}
