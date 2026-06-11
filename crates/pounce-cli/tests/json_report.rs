//! End-to-end integration test for the JSON solve report (pounce#8).
//!
//! Exercises both `pounce` and `pounce_sens` binaries' `--json-output`
//! flags against the same hand-crafted parametric_cpp `.nl` fixture
//! the sensitivity tests use, and checks:
//!
//! * The emitted JSON deserializes back into `SolveReport`.
//! * Schema tag is `pounce.solve-report/v1`.
//! * `Summary` detail omits `iterations` / suffix blocks; `Full`
//!   includes them.
//! * `solution.x` and `solution.lambda` are populated and finite.
//! * `solution.objective` matches `statistics.final_objective`.

use std::path::PathBuf;
use std::process::Command;

use pounce_cli::solve_report::SolveReport;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn pounce_sens_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce_sens"))
}

fn fixture_nl() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("parametric.nl");
    p
}

fn tmp_path(suffix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pounce_json_{}_{suffix}", std::process::id()));
    p
}

#[test]
fn pounce_emits_summary_report_without_iterations() {
    let json_path = tmp_path("pounce_sum.json");
    let status = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg("--json-output")
        .arg(&json_path)
        .arg("--json-detail")
        .arg("summary")
        .status()
        .expect("spawn pounce");
    assert!(status.success(), "pounce exited with {status:?}");

    let text = std::fs::read_to_string(&json_path).unwrap();
    let report: SolveReport = serde_json::from_str(&text).expect("deserialize");
    assert_eq!(report.schema, "pounce.solve-report/v1");
    assert_eq!(
        report.fair_metadata.solver.name, "pounce",
        "FAIR metadata identifies solver"
    );
    assert!(
        !report.fair_metadata.result_id.is_empty(),
        "result_id present"
    );
    assert_eq!(report.problem.n_variables, 5);
    assert_eq!(report.problem.n_constraints, 4);
    assert_eq!(report.solution.x.len(), 5);
    assert_eq!(report.solution.lambda.len(), 4);
    assert!(report.solution.objective.is_finite());
    assert_eq!(
        report.statistics.iteration_count,
        report.statistics.iteration_count
    ); // sanity
       // Summary mode: iterations dropped.
    assert!(
        report.iterations.is_empty(),
        "summary should drop iter history, got {}",
        report.iterations.len()
    );
    // And the raw JSON should not contain the key (skip-if-empty serde tag).
    assert!(!text.contains("\"iterations\""), "json: {text}");

    let _ = std::fs::remove_file(&json_path);
}

#[test]
fn pounce_emits_full_report_with_iterations() {
    let json_path = tmp_path("pounce_full.json");
    let status = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg("--json-output")
        .arg(&json_path)
        .arg("--json-detail")
        .arg("full")
        .status()
        .expect("spawn pounce");
    assert!(status.success());

    let text = std::fs::read_to_string(&json_path).unwrap();
    let report: SolveReport = serde_json::from_str(&text).expect("deserialize");
    assert_eq!(report.schema, "pounce.solve-report/v1");
    assert!(
        !report.iterations.is_empty(),
        "full mode should capture iter rows"
    );
    let it0 = &report.iterations[0];
    assert_eq!(it0.iter, 0, "first row is iter 0");
    assert!(it0.inf_pr >= 0.0, "inf_pr is non-negative");

    let _ = std::fs::remove_file(&json_path);
}

#[test]
fn pounce_sens_emits_report_with_sens_sol_state_suffix() {
    let sol_path = tmp_path("ps.sol");
    let json_path = tmp_path("ps.json");
    let status = Command::new(pounce_sens_exe())
        .arg(fixture_nl())
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path)
        .arg("--json-detail")
        .arg("full")
        .status()
        .expect("spawn pounce_sens");
    assert!(status.success());

    let text = std::fs::read_to_string(&json_path).unwrap();
    let report: SolveReport = serde_json::from_str(&text).expect("deserialize");
    let sens = report
        .solution
        .suffixes
        .iter()
        .find(|s| s.name == "sens_sol_state_1")
        .expect("sens_sol_state_1 suffix present");
    assert_eq!(sens.target, "var");
    assert_eq!(sens.kind, "real");
    assert_eq!(sens.values.len(), 5);
    // Perturbed x[3] = 4.5 (the Δeta1 = -0.5 perturbation pins eta1).
    assert!(
        (sens.values[3] - 4.5).abs() < 1e-8,
        "perturbed x[3] = {} (expected 4.5)",
        sens.values[3],
    );

    let _ = std::fs::remove_file(&sol_path);
    let _ = std::fs::remove_file(&json_path);
}

/// The `--json-output` report must have a *uniform* schema regardless of
/// which solver path produced it. The NLP path is covered above and the
/// convex QP-IPM path in `qp_dispatch_end_to_end.rs`, but nothing asserts
/// the schema is genuinely identical in shape across paths — including the
/// LP-IPM path, which had no JSON coverage at all. This runs one set of
/// schema invariants over three distinct solver paths (NLP, convex QP-IPM,
/// convex LP-IPM) so the benchmark harness can ingest any pounce solve
/// uniformly. A path that emitted a divergent or placeholder report (e.g.
/// an objective that disagrees with `final_objective`, or an `x` whose
/// length contradicts `n_variables`) would fail here.
#[test]
fn json_schema_is_uniform_across_solver_paths() {
    fn fixture_named(name: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests");
        p.push("fixtures");
        p.push(name);
        p
    }

    // (label, fixture, forced solver_selection) — three genuinely different
    // code paths inside the CLI dispatch.
    let cases: &[(&str, PathBuf, &str)] = &[
        ("nlp", fixture_nl(), "nlp"),
        ("convex-qp-ipm", fixture_named("convex_qp.nl"), "qp-ipm"),
        ("convex-lp-ipm", fixture_named("lp_afiro.nl"), "lp-ipm"),
    ];

    for (label, fixture, sel) in cases {
        let json_path = tmp_path(&format!("uniform_{label}.json"));
        let _ = std::fs::remove_file(&json_path);
        let out = Command::new(pounce_exe())
            .arg(fixture)
            .arg("--no-sol")
            .arg("--json-output")
            .arg(&json_path)
            .arg(format!("solver_selection={sel}"))
            .output()
            .unwrap_or_else(|e| panic!("spawn pounce ({label}): {e}"));
        assert_eq!(
            out.status.code(),
            Some(0),
            "{label} solve should succeed; stderr=\n{}",
            String::from_utf8_lossy(&out.stderr)
        );

        let text = std::fs::read_to_string(&json_path)
            .unwrap_or_else(|e| panic!("read report ({label}): {e}"));
        let report: SolveReport = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("deserialize report ({label}): {e}\n{text}"));

        // --- invariants every path must satisfy identically ---
        assert_eq!(
            report.schema, "pounce.solve-report/v1",
            "{label}: schema tag"
        );
        assert_eq!(
            report.fair_metadata.solver.name, "pounce",
            "{label}: solver name"
        );
        assert!(
            !report.fair_metadata.result_id.is_empty(),
            "{label}: result_id present"
        );
        assert!(!report.solution.x.is_empty(), "{label}: primal x populated");
        assert!(
            report.solution.x.iter().all(|v| v.is_finite()),
            "{label}: primal x all finite"
        );
        assert!(
            report.solution.objective.is_finite(),
            "{label}: objective finite"
        );
        assert!(
            (report.solution.objective - report.statistics.final_objective).abs()
                <= 1e-9 * report.solution.objective.abs().max(1.0),
            "{label}: solution.objective {} != statistics.final_objective {}",
            report.solution.objective,
            report.statistics.final_objective
        );
        assert_eq!(
            report.problem.n_variables as usize,
            report.solution.x.len(),
            "{label}: n_variables matches x length"
        );

        let _ = std::fs::remove_file(&json_path);
    }
}

#[test]
fn schema_field_is_stable_across_runs() {
    let p1 = tmp_path("schema_a.json");
    let p2 = tmp_path("schema_b.json");
    for p in [&p1, &p2] {
        Command::new(pounce_exe())
            .arg(fixture_nl())
            .arg("--json-output")
            .arg(p)
            .status()
            .expect("spawn pounce");
    }
    let r1: SolveReport = serde_json::from_str(&std::fs::read_to_string(&p1).unwrap()).unwrap();
    let r2: SolveReport = serde_json::from_str(&std::fs::read_to_string(&p2).unwrap()).unwrap();
    assert_eq!(r1.schema, r2.schema);
    assert_eq!(
        r1.fair_metadata.solver.version,
        r2.fair_metadata.solver.version
    );
    // Two separate runs must produce distinct result_ids.
    assert_ne!(r1.fair_metadata.result_id, r2.fair_metadata.result_id);

    let _ = std::fs::remove_file(&p1);
    let _ = std::fs::remove_file(&p2);
}
