//! End-to-end CLI integration test for the auxiliary-equality
//! preprocessing pass (issue #53).
//!
//! ⚠️ Coverage limitation: the only `.nl` fixture in this repo
//! today (`parametric.nl`) carries sensitivity suffixes, and
//! `crates/pounce-cli/src/main.rs` silently disables presolve when
//! either sensitivity or reduced-Hessian post-processing is active
//! (see `main.rs:306-312`). That means tests against
//! `parametric.nl` cannot exercise the orchestrator's solve path —
//! they can only verify that the CLI plumbing accepts the new
//! options, prints the documented warning, and reaches the same
//! objective as the baseline.
//!
//! The orchestrator's solve path is covered by inline tests in
//! `crates/pounce-presolve/src/lib.rs` (the `phase0_via_tnlp_*`
//! tests). The headline `.nl` acceptance criteria from issue #53
//! require vendoring the ripopt fixtures into
//! `crates/pounce-cli/tests/fixtures/aux_presolve/`; see that
//! directory's `README.md`.

use std::path::PathBuf;
use std::process::Command;

use pounce_cli::solve_report::SolveReport;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture_nl() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("parametric.nl");
    p
}

fn aux_fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("aux_presolve");
    p.push(name);
    p
}

fn tmp_json(suffix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pounce_aux_{}_{suffix}.json", std::process::id()));
    p
}

fn run_for_report(fixture: &PathBuf, extra_args: &[&str]) -> SolveReport {
    let json_path = tmp_json(&format!(
        "e2e_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture)
        .arg("--json-output")
        .arg(&json_path)
        .arg("--json-detail")
        .arg("summary");
    for a in extra_args {
        cmd.arg(a);
    }
    let output = cmd.output().expect("spawn pounce");
    assert!(
        output.status.success(),
        "pounce failed (exit {:?}); stderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let text = std::fs::read_to_string(&json_path).unwrap();
    let _ = std::fs::remove_file(&json_path);
    serde_json::from_str(&text).expect("parse SolveReport JSON")
}

/// `parametric.nl` carries sensitivity suffixes, so the CLI's
/// sensitivity guard kicks in and silently disables presolve before
/// the auxiliary pass would ever run. Confirm:
///   - the binary doesn't panic with the new options;
///   - the documented warning lands on stderr (proving the
///     sens-disable code path is genuinely executed and the test
///     isn't measuring nothing).
#[test]
fn presolve_auxiliary_yes_disabled_by_sensitivity_warning() {
    let output = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg("presolve=yes")
        .arg("presolve_auxiliary=yes")
        .output()
        .expect("spawn pounce");
    let code = output.status.code().unwrap_or(-1);
    assert!(
        code == 0 || code == 1,
        "pounce exited with code {code} (stderr: {})",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panicked at"), "pounce panicked: {stderr}");
    assert!(
        stderr.contains("disabling presolve"),
        "expected the sensitivity-disable warning in stderr; got:\n{stderr}"
    );
}

/// Sanity check: `presolve_auxiliary=no` and the baseline produce
/// the same final objective. With the sens-disable in play above
/// this reduces to "the solver is deterministic given identical CLI
/// args" — but it still guards against a regression where the flag
/// parser would mis-set state.
#[test]
fn presolve_auxiliary_no_matches_baseline_objective() {
    let baseline_json = tmp_json("baseline");
    let aux_off_json = tmp_json("aux_off");

    let baseline_status = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg("--json-output")
        .arg(&baseline_json)
        .arg("--json-detail")
        .arg("summary")
        .status()
        .expect("spawn pounce baseline");
    assert!(baseline_status.success());

    let aux_off_status = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg("--json-output")
        .arg(&aux_off_json)
        .arg("--json-detail")
        .arg("summary")
        .arg("presolve=yes")
        .arg("presolve_auxiliary=no")
        .status()
        .expect("spawn pounce aux-off");
    assert!(aux_off_status.success());

    let baseline_text = std::fs::read_to_string(&baseline_json).unwrap();
    let baseline: SolveReport = serde_json::from_str(&baseline_text).unwrap();
    let aux_off_text = std::fs::read_to_string(&aux_off_json).unwrap();
    let aux_off: SolveReport = serde_json::from_str(&aux_off_text).unwrap();

    let bobj = baseline.statistics.final_objective;
    let aobj = aux_off.statistics.final_objective;
    assert!(
        (bobj - aobj).abs() < 1e-6,
        "baseline obj {bobj} vs aux-off obj {aobj}"
    );

    let _ = std::fs::remove_file(&baseline_json);
    let _ = std::fs::remove_file(&aux_off_json);
}

/// End-to-end acceptance test for issue #53 on the vendored
/// `tutorial_flow_density.nl` fixture (from upstream ripopt, see
/// `tests/fixtures/aux_presolve/README.md`).
///
/// The auxiliary pass should fix every variable up-front so the IPM
/// runs zero iterations and converges to the same objective as the
/// un-presolved baseline.
#[test]
fn presolve_auxiliary_yes_tutorial_zero_ipm_iters() {
    let nl = aux_fixture("tutorial_flow_density.nl");
    if !nl.exists() {
        eprintln!("skipping: {} not present", nl.display());
        return;
    }

    let baseline = run_for_report(&nl, &["presolve=yes", "presolve_auxiliary=no"]);
    let aux_on = run_for_report(&nl, &["presolve=yes", "presolve_auxiliary=yes"]);

    assert_eq!(
        aux_on.statistics.iteration_count, 0,
        "aux on should fix everything pre-IPM; got {} iters",
        aux_on.statistics.iteration_count
    );

    let bobj = baseline.statistics.final_objective;
    let aobj = aux_on.statistics.final_objective;
    assert!(
        (bobj - aobj).abs() < 1e-6,
        "aux objective {aobj} disagrees with baseline {bobj}"
    );
}

/// Same as above on the perturbed-operating-point variant; same
/// expected outcome (0 IPM iters, matching baseline objective).
#[test]
fn presolve_auxiliary_yes_tutorial_perturbed_zero_ipm_iters() {
    let nl = aux_fixture("tutorial_flow_density_perturbed.nl");
    if !nl.exists() {
        eprintln!("skipping: {} not present", nl.display());
        return;
    }

    let baseline = run_for_report(&nl, &["presolve=yes", "presolve_auxiliary=no"]);
    let aux_on = run_for_report(&nl, &["presolve=yes", "presolve_auxiliary=yes"]);

    assert_eq!(
        aux_on.statistics.iteration_count, 0,
        "aux on should fix everything pre-IPM; got {} iters",
        aux_on.statistics.iteration_count
    );

    let bobj = baseline.statistics.final_objective;
    let aobj = aux_on.statistics.final_objective;
    assert!(
        (bobj - aobj).abs() < 1e-6,
        "aux objective {aobj} disagrees with baseline {bobj}"
    );
}

/// Regression test for the aux+bound_tighten interaction bug
/// discovered during the PR #60 review: with `presolve_auxiliary=yes`
/// and the default `presolve_bound_tightening=yes`, the orchestrator
/// used to produce `x_l > x_u` for an aux-fixed variable, causing the
/// IPM to abort with `Invalid Problem Definition`.
///
/// The fixture has a real auxiliary block structure (22 of 24
/// candidate blocks eliminated). Once the bug is fixed the run must
/// reach `Optimal Solution Found` with default options.
///
/// Note: aux-on lands at a different local optimum on this nonconvex
/// problem than the un-presolved path, so we do NOT assert objective
/// parity here — only that the solve completes successfully.
#[test]
fn presolve_auxiliary_yes_gaslib11_solves_with_defaults() {
    let nl = aux_fixture("gaslib11_steady.nl");
    if !nl.exists() {
        eprintln!("skipping: {} not present", nl.display());
        return;
    }

    let report = run_for_report(&nl, &["presolve=yes", "presolve_auxiliary=yes"]);

    assert!(
        report.statistics.final_objective.is_finite(),
        "expected a finite objective; got {}",
        report.statistics.final_objective
    );
}
