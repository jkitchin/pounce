//! `.nl` complementarity rows (AMPL `complements`, `r`-segment kind 5)
//! through the CLI: read as pairs, solved by
//! `pounce_restoration::mpcc_driver::solve_mpcc`, certified.
//!
//! The fixtures live in `tests/fixtures_mpcc/`, outside `tests/fixtures/`, so
//! the fixture sweep's corpus is unchanged by them.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures_mpcc")
        .join(name)
}

struct Run {
    code: Option<i32>,
    stderr: String,
    json: Option<serde_json::Value>,
}

fn run(name: &str, extra: &[&str]) -> Run {
    // One directory per call: tests run in parallel, and two of them solving
    // the same fixture would otherwise write and read the same report.json.
    static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "pounce-mpcc-{}-{}-{}",
        name,
        std::process::id(),
        call
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let json_path = dir.join("report.json");
    let out = Command::new(pounce_exe())
        .arg(fixture(name))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json_path)
        .args(extra)
        .output()
        .expect("spawn pounce");
    let json = std::fs::read_to_string(&json_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    Run {
        code: out.status.code(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        json,
    }
}

fn x_of(r: &Run) -> Vec<f64> {
    r.json.as_ref().unwrap()["solution"]["x"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

/// `qpec_small` (two pairs, the second biactive at the answer `(1, 1, 0)`):
/// read from `.nl`, solved, and certified strongly stationary.
#[test]
fn a_qpec_written_with_complements_is_solved_and_certified() {
    let r = run("qpec_small.nl", &[]);
    assert_eq!(r.code, Some(0), "{}", r.stderr);
    // 1e-6, not looser: the exact-product stage's own point,
    // (1.000019, 1.0000095, 0), is within 1e-4 of the answer, and the report
    // once carried it in place of the continuation's, which was the one
    // accepted and summarized.
    let x = x_of(&r);
    for (got, want) in x.iter().zip([1.0, 1.0, 0.0]) {
        assert!((got - want).abs() < 1e-6, "x = {x:?}");
    }
    assert!(
        r.stderr.contains("2 pairs, 1 biactive; stationarity: S"),
        "{}",
        r.stderr
    );
    // The structure classifier reads the rows as a QP; with the pairing it is
    // not one, and the solve must not go to a convex engine.
    assert_eq!(
        r.json.as_ref().unwrap()["solution"]["engine"].as_str(),
        Some("nlp"),
        "{:?}",
        r.json
    );
}

/// `ralph1`: the answer, the origin, admits no S-stationary multiplier. The
/// run converges and says so: W, not S.
#[test]
fn a_converged_but_not_strongly_stationary_answer_is_labelled() {
    let r = run("ralph1.nl", &[]);
    assert_eq!(r.code, Some(0), "{}", r.stderr);
    assert!(x_of(&r).iter().all(|v| v.abs() < 1e-4), "{:?}", x_of(&r));
    assert!(r.stderr.contains("stationarity: W only"), "{}", r.stderr);
}

/// A two-sided (mixed) complementarity, `0 <= y <= 1 ⟂ y - x` (`.nl` flag
/// 3), is solved, not refused or dropped. Its optimum is the origin (`f* = 0`;
/// the branches `y = 1` and `y = x` are worse), where the pair is biactive at
/// the lower bound and only weakly stationary, as in `ralph1` it extends.
/// Without the pairing the model is unbounded below in `y`'s direction.
#[test]
fn a_mixed_pair_is_solved() {
    let r = run("mixed_pair.nl", &[]);
    assert_eq!(r.code, Some(0), "{}", r.stderr);
    let x = x_of(&r);
    assert!(x.iter().all(|v| v.abs() < 1e-4), "x = {x:?}");
    assert!(r.stderr.contains("stationarity: W"), "{}", r.stderr);
}

/// Presolve renumbers rows and variables, which the pairs index; asking for
/// it is honoured by declining it, out loud.
#[test]
fn presolve_stands_down_for_complementarity() {
    let r = run("qpec_small.nl", &["presolve=yes"]);
    assert_eq!(r.code, Some(0), "{}", r.stderr);
    assert!(r.stderr.contains("disabling presolve"), "{}", r.stderr);
}

/// The JSON report carries the driver's record (`complementarity`), keyed like
/// Python's `info["complementarity"]`. `statistics.iteration_count` is the
/// reported lowered solve's alone; `total_iterations` is what the run cost,
/// and a report without it read a multi-stage solve as a single fast one.
#[test]
fn the_json_report_carries_the_complementarity_block() {
    let r = run("qpec_small.nl", &[]);
    assert_eq!(r.code, Some(0), "{}", r.stderr);
    let j = r.json.as_ref().expect("report");
    let c = &j["complementarity"];
    assert_eq!(c["strategy"], "both", "{c}");
    assert_eq!(c["reported"], "answer", "{c}");
    assert!(c["accepted_by"].is_string(), "{c}");
    assert_eq!(c["certificate"]["class"], "S", "{c}");
    assert_eq!(c["certificate"]["feasible"], true, "{c}");
    let solves = c["solves"].as_array().expect("solves");
    let sum: u64 = solves
        .iter()
        .map(|s| s["iterations"].as_u64().unwrap())
        .sum();
    assert_eq!(c["total_iterations"].as_u64(), Some(sum), "{c}");
    assert!(
        sum > j["statistics"]["iteration_count"].as_u64().unwrap(),
        "the total must exceed the reported solve's own count: {c}"
    );
    assert!(
        solves
            .iter()
            .all(|s| s["status"].as_str().unwrap().contains('_')),
        "upstream status names: {c}"
    );

    // A weakly stationary fallback is not an answer.
    let r = run("ralph1.nl", &[]);
    let c = &r.json.as_ref().expect("report")["complementarity"];
    assert_eq!(c["reported"], "weak_fallback", "{c}");
    assert!(c.get("accepted_by").is_none(), "{c}");
    assert_eq!(c["certificate"]["class"], "W", "{c}");

    // The strategy is recorded, and only its stages (and the penalty) ran.
    let r = run("qpec_small.nl", &["complementarity_strategy=exact"]);
    let c = &r.json.as_ref().expect("report")["complementarity"];
    assert_eq!(c["strategy"], "exact", "{c}");
    assert!(
        c["solves"]
            .as_array()
            .unwrap()
            .iter()
            .all(|s| s["stage"] != "Scholtes continuation"),
        "{c}"
    );
}

/// A model without complementarity pairs carries no `complementarity` block.
#[test]
fn a_model_without_pairs_has_no_complementarity_block() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("nonconvex_qp.nl");
    let dir = std::env::temp_dir().join(format!("pounce-no-mpcc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let json_path = dir.join("report.json");
    let out = Command::new(pounce_exe())
        .arg(&path)
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json_path)
        .output()
        .expect("spawn pounce");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
    assert!(j.get("complementarity").is_none(), "{j}");
}

/// `x` by `.col` name, for a fixture that ships its Pyomo column names.
fn x_by_name(r: &Run, nl: &str) -> std::collections::HashMap<String, f64> {
    let col = std::fs::read_to_string(fixture(&nl.replace(".nl", ".col"))).unwrap();
    col.split_whitespace()
        .map(String::from)
        .zip(x_of(r))
        .collect()
}

/// The largest `min(r_h, s_h)` over the OPF fixtures' six hinge pairs.
fn natural_residual(x: &std::collections::HashMap<String, f64>) -> f64 {
    (1..=6)
        .map(|h| x[&format!("r[{h}]")].min(x[&format!("s[{h}]")]).abs())
        .fold(0.0, f64::max)
}

fn last_stage(r: &Run) -> (String, String) {
    let s = r.json.as_ref().unwrap()["complementarity"]["solves"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .clone();
    (
        s["stage"].as_str().unwrap().to_string(),
        s["status"].as_str().unwrap().to_string(),
    )
}

/// The polish's accepted branch (`mpcc_driver`, `natural_residual`). At a
/// knot of the volt-var curve the hinge pair is biactive, and every stage
/// stops at a product near the last `τ` with both sides near `sqrt(τ)`:
/// `r = 7.8e-5`, `s = 1.3e-4` at a product of `1e-8`, which put `q`
/// 5.9e-4 off its curve (slope 7.5) while reporting `Solve_Succeeded`. The
/// re-solve on the branch that holds each pair's smaller side at zero
/// removes it.
#[test]
fn a_nearly_biactive_answer_is_re_solved_onto_its_branch() {
    let r = run("opf_vv_lower_knee.nl", &[]);
    assert_eq!(r.code, Some(0), "{}", r.stderr);
    assert_eq!(
        r.json.as_ref().unwrap()["solution"]["status_upstream"],
        "Solve_Succeeded"
    );
    assert_eq!(last_stage(&r).0, "branch polish");
    let x = x_by_name(&r, "opf_vv_lower_knee.nl");
    assert!(
        natural_residual(&x) <= 1e-9,
        "min(r, s) = {:e}",
        natural_residual(&x)
    );
    // q on its volt-var curve, evaluated directly: (0.94, 0.3) to (0.98, 0).
    let (v, q) = (x["v[2]"], x["q[1]"]);
    let vv = if v <= 0.94 {
        0.3
    } else if v >= 0.98 {
        0.0
    } else {
        0.3 * (0.98 - v) / 0.04
    };
    assert!((q - vv).abs() <= 1e-7, "v = {v}, q = {q}, VV(v) = {vv}");
    // The report says so, without the reader recomputing it: the reported
    // point's residual, and the re-solve that produced it.
    let c = &r.json.as_ref().unwrap()["complementarity"];
    let nr = c["natural_residual"].as_f64().expect("natural_residual");
    assert!(nr <= 1e-9, "natural_residual = {nr:e}");
    let p = &c["polish"];
    assert_eq!(p["taken"], true, "{p}");
    assert!(p["residual_before"].as_f64().unwrap() > 1e-5, "{p}");
    assert_eq!(p["residual_after"].as_f64(), Some(nr), "{p}");
    assert!(p["reason"].is_null(), "{p}");
    assert!(c["solves"][0]["seconds"].as_f64().is_some(), "{c}");
}

/// The polish's other branch: `opf_vv_lower_knee_held.nl` adds `r[1] >=
/// 5e-5`, which the stages' near-biactive point satisfies and the branch
/// holding `r[1]` at zero cannot. The re-solve is refused as infeasible and
/// the answer is reported as it was, unpolished; it is never replaced by
/// the failed re-solve's point.
#[test]
fn a_branch_the_answer_cannot_hold_leaves_the_answer_as_it_was() {
    let r = run("opf_vv_lower_knee_held.nl", &[]);
    assert_eq!(r.code, Some(0), "{}", r.stderr);
    let (stage, status) = last_stage(&r);
    assert_eq!(stage, "branch polish");
    assert_eq!(status, "Infeasible_Problem_Detected");
    assert_eq!(
        r.json.as_ref().unwrap()["solution"]["status_upstream"],
        "Solve_Succeeded"
    );
    let x = x_by_name(&r, "opf_vv_lower_knee_held.nl");
    assert!(x["r[1]"] >= 5e-5 - 1e-9, "r[1] = {}", x["r[1]"]);
    assert!(
        natural_residual(&x) > 1e-6,
        "the kept point is the stages' own"
    );
    // The report says the answer is loose, and why it stayed so: a reader
    // of `certificate.class` alone would take it for a tight S answer.
    let c = &r.json.as_ref().unwrap()["complementarity"];
    let nr = c["natural_residual"].as_f64().expect("natural_residual");
    assert!(nr > 1e-6, "natural_residual = {nr:e}");
    let p = &c["polish"];
    assert_eq!(p["taken"], false, "{p}");
    assert_eq!(p["residual_before"].as_f64(), Some(nr), "{p}");
    assert_eq!(
        p["reason"], "the re-solve returned Infeasible_Problem_Detected",
        "{p}"
    );
}

/// A point already complementary to `tol` is not re-solved, and the report
/// says how complementary it is with no `polish` record. Pinned to the
/// schedule: it is this branch's fixture, and under the coupled continuation
/// `qpec_small`'s answer starts at a residual of 1e-6 and is polished (to 0),
/// which is the other branch.
#[test]
fn a_tight_answer_reports_its_residual_and_no_polish() {
    let r = run("qpec_small.nl", &["complementarity_continuation=schedule"]);
    assert_eq!(r.code, Some(0), "{}", r.stderr);
    let c = &r.json.as_ref().unwrap()["complementarity"];
    let nr = c["natural_residual"].as_f64().expect("natural_residual");
    assert!(nr <= 1e-7, "natural_residual = {nr:e}");
    assert!(c.get("polish").is_none(), "{c}");
}

/// The ladder's last branch: no formulation answers (`infeasible_pair.nl`
/// has `x >= 1`, `y >= 1` and `x ⟂ y`), so every stage runs, the l1 penalty
/// included, and the exact-product solve's verdict is reported. Not an
/// answer, so it is neither certified feasible nor polished.
#[test]
fn a_pair_no_point_satisfies_reports_the_verdict() {
    let r = run("infeasible_pair.nl", &[]);
    let j = r.json.as_ref().expect("report");
    assert_eq!(
        j["solution"]["status_upstream"], "Infeasible_Problem_Detected",
        "{}",
        r.stderr
    );
    let c = &j["complementarity"];
    assert_eq!(c["reported"], "verdict", "{c}");
    assert!(
        c.get("accepted_by").is_none() && c.get("fallback").is_none(),
        "{c}"
    );
    assert_eq!(c["certificate"]["feasible"], false, "{c}");
    assert!(c.get("polish").is_none(), "{c}");
    let stages: Vec<&str> = c["solves"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["stage"].as_str().unwrap())
        .collect();
    assert!(stages.contains(&"l1 penalty"), "{stages:?}");
}

fn continuation_taus(r: &Run) -> Vec<Option<f64>> {
    r.json.as_ref().unwrap()["complementarity"]["solves"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["stage"] == "Scholtes continuation")
        .map(|s| s["tau"].as_f64())
        .collect()
}

/// The coupled continuation's other branch (review of #1135, item 2): its
/// answer does not pass the schedule's test, so the schedule resumes at
/// `τ = 0.1` warm from the `τ = 1` solve. `ralph1` under limited memory is
/// the measured case (the sweep's lbfgs leg): the coupled solve reaches the
/// floor, all three finish attempts miss, and the schedule's eight rungs and
/// finish follow. The answer is the schedule's own.
#[test]
fn a_coupled_continuation_that_does_not_answer_hands_over_to_the_schedule() {
    let lbfgs = "hessian_approximation=limited-memory";
    let r = run("ralph1.nl", &[lbfgs]);
    assert!(
        r.stderr.contains("resuming the schedule at τ = 1e-1"),
        "{}",
        r.stderr
    );
    let taus = continuation_taus(&r);
    assert_eq!(taus[0], Some(1.0), "{taus:?}");
    let resumed = taus
        .iter()
        .position(|t| *t == Some(0.1))
        .expect("schedule resumed");
    assert_eq!(
        &taus[resumed..resumed + 8],
        &[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8].map(Some),
        "{taus:?}"
    );
    let s = run(
        "ralph1.nl",
        &[lbfgs, "complementarity_continuation=schedule"],
    );
    let f = |r: &Run| {
        r.json.as_ref().unwrap()["solution"]["objective"]
            .as_f64()
            .unwrap()
    };
    assert!((f(&r) - f(&s)).abs() <= 1e-9, "{} vs {}", f(&r), f(&s));
}

/// The coupling rides the interior-point loop's μ updates; under another
/// engine there are none, so the schedule runs from the start (review of
/// #1135, item 5).
#[test]
fn the_active_set_engine_runs_the_schedule() {
    let r = run("qpec_small.nl", &["algorithm=active-set-sqp"]);
    let mut want: Vec<Option<f64>> = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        .map(Some)
        .to_vec();
    want.push(None);
    assert_eq!(continuation_taus(&r), want, "{}", r.stderr);
}
