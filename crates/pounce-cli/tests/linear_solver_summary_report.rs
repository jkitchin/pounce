//! The solve report's `linear_solver` object across the restoration phase
//! and the second-opinion ladder (structured-KKT Phase 0a).
//!
//! # What this file pins
//!
//! 1. Restoration-phase factorizations are reported, in their own
//!    `linear_solver.restoration` object, and not folded into the main
//!    solve's counts. Before, restoration recorded nothing: on this fixture
//!    that hid 171 of 229 factorizations.
//! 2. A solve that never enters restoration carries no `restoration` object.
//! 3. When the ladder **promotes** a rung, the summary is that rung's: it
//!    equals a direct solve run with the rung's option from the start. The
//!    keep-the-original branch is pinned from Python
//!    (`test_linear_solver_summary_follows_the_kept_verdict`), because no CLI
//!    fixture here keeps its original verdict through the ladder.
//!
//! # What it is *not* evidence about
//!
//! One fixture per branch and FERAL only (MA57 does not self-instrument). The
//! counts are compared between runs of the same binary, never against fixed
//! numbers, so a trajectory change moves them without failing the file.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// Solve `model` with `opts` and return the parsed JSON report. The `.sol`
/// and the report go to a per-test temp directory, never next to the fixture.
fn solve_report(tag: &str, model: &str, opts: &[&str]) -> serde_json::Value {
    let dir = std::env::temp_dir().join(format!("pounce_linsol_summary_{tag}"));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let sol = dir.join("out.sol");
    let json = dir.join("report.json");
    let _ = std::fs::remove_file(&json);
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(fixture(model))
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .args(opts)
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(Path::new(&json)).unwrap_or_else(|e| {
        panic!(
            "no report written ({e}); stderr:\n{}",
            String::from_utf8_lossy(&out.stderr)
        )
    });
    serde_json::from_str(&text).expect("report parses")
}

fn u(v: &serde_json::Value, key: &str) -> u64 {
    v[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} missing or not an integer in {v}"))
}

/// Switches the ladder off, so the restoration-terminating path is what is
/// measured (see `issue_819_restoration_iteration_count.rs` for why both
/// rungs have to be named).
const NO_LADDER: [&str; 2] = [
    "infeasibility_perturbed_start_retry=no",
    "feral_increase_quality_retry=no",
];

#[test]
fn restoration_factorizations_are_reported_separately() {
    let r = solve_report("resto", "square_flowsheet_resto.nl", &NO_LADDER);
    let stats = &r["statistics"];
    assert!(
        u(stats, "restoration_calls") >= 1,
        "fixture must enter restoration"
    );

    let ls = &r["linear_solver"];
    assert!(u(ls, "n_factors") >= 1);
    assert!(ls["total_factor_secs"].as_f64().is_some());
    assert!(ls["last_ordering"].as_str().is_some());

    let resto = &ls["restoration"];
    assert!(
        resto.is_object(),
        "restoration factorizations were not reported"
    );
    // Every inner restoration iteration factors at least once.
    assert!(
        u(resto, "n_factors") >= u(stats, "restoration_inner_iters"),
        "restoration n_factors {} < restoration_inner_iters {}",
        u(resto, "n_factors"),
        u(stats, "restoration_inner_iters"),
    );
    assert!(resto.get("restoration").is_none(), "no nested restoration");
}

#[test]
fn a_solve_without_restoration_reports_none() {
    let r = solve_report("no_resto", "airport.nl", &[]);
    assert_eq!(u(&r["statistics"], "restoration_calls"), 0);
    let ls = &r["linear_solver"];
    assert!(u(ls, "n_factors") >= 1);
    assert!(ls.get("restoration").is_none());
}

/// The ladder promotes `start_point_perturbation=1e-2` on this fixture. The
/// shipped summary must be that rung's, which is the same solve as running
/// with the option from the start and the ladder off.
#[test]
fn a_promoted_rung_reports_its_own_summary() {
    let laddered = solve_report("ladder", "square_flowsheet_resto.nl", &[]);
    let mut direct_opts = vec!["start_point_perturbation=1e-2"];
    direct_opts.extend(NO_LADDER);
    let direct = solve_report("direct", "square_flowsheet_resto.nl", &direct_opts);

    assert_eq!(
        laddered["solution"]["status"], direct["solution"]["status"],
        "the ladder no longer promotes the rung this test is about"
    );
    assert_eq!(
        u(&laddered["statistics"], "iteration_count"),
        u(&direct["statistics"], "iteration_count"),
    );
    let (a, b) = (&laddered["linear_solver"], &direct["linear_solver"]);
    assert_eq!(
        u(a, "n_factors"),
        u(b, "n_factors"),
        "main-solve factorizations"
    );
    assert_eq!(
        u(&a["restoration"], "n_factors"),
        u(&b["restoration"], "n_factors"),
        "restoration factorizations"
    );
}
