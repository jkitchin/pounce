//! gh#1054: `ipm_sqp_handover` — the interior-point method hands its iterate
//! to the active-set SQP before convergence.
//!
//! Three properties, each on a fixture that reaches it:
//!
//! - an accepted handover ends the solve early at the interior method's
//!   answer (`bound_active_qp`, `airport`);
//! - a handover whose point is a KKT point only in the *scaled* space is
//!   declined, and the interior method finishes as if it had never been
//!   tried (`hs13_bigstart`: started at `x₀ = 1e4`, gradient scaling shrinks
//!   the row by ~3e-7, and the SQP's degenerate tail jumps to `x = (2, 0)`,
//!   which violates the declared row by 1 — this shipped as
//!   `Solve_Succeeded` with every printed residual 0 in the first draft);
//! - off by default, and off moves nothing.

use std::path::PathBuf;
use std::process::Command;

use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_solve_report::SolveReport;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn solve(name: &str, tag: &str, opts: &[&str]) -> SolveReport {
    let json = std::env::temp_dir().join(format!(
        "pounce-issue1054-ho-{}-{}-{tag}.json",
        std::process::id(),
        name.replace('.', "-"),
    ));
    let _ = std::fs::remove_file(&json);
    let out = Command::new(pounce_exe())
        .arg(fixture(name))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .arg("solver_selection=nlp")
        .args(opts)
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "no JSON report ({e}); stdout=\n{}\nstderr=\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    let _ = std::fs::remove_file(&json);
    serde_json::from_str(&text).expect("parse JSON report")
}

const HANDOVER: &[&str] = &["ipm_sqp_handover=yes", "ipm_sqp_handover_tol=1e-3"];

fn assert_same_answer(plain: &SolveReport, ho: &SolveReport, rel: f64) {
    let (a, b) = (plain.solution.objective, ho.solution.objective);
    assert!(
        (a - b).abs() <= rel * a.abs().max(1.0),
        "objective moved: plain {a} vs handover {b}"
    );
}

#[test]
fn accepted_handover_ends_early_at_the_same_answer() {
    for name in ["bound_active_qp.nl", "airport.nl"] {
        let plain = solve(name, "plain", &[]);
        let ho = solve(name, "ho", HANDOVER);
        assert_eq!(
            ho.solution.status,
            ApplicationReturnStatus::SolveSucceeded,
            "{name}"
        );
        // Not bit-equal: the handover's point sits *on* the active bounds,
        // the interior one `O(μ)` inside them (5 vs 4.99999994 on
        // `bound_active_qp`).
        assert_same_answer(&plain, &ho, 1e-6);
        assert!(
            ho.statistics.iteration_count < plain.statistics.iteration_count,
            "{name}: handover took {} interior iterations, plain {}",
            ho.statistics.iteration_count,
            plain.statistics.iteration_count
        );
        assert!(
            ho.statistics.final_declared_constr_viol <= 1e-6,
            "{name}: {}",
            ho.statistics.final_declared_constr_viol
        );
    }
}

#[test]
fn a_point_feasible_only_in_scaled_units_is_declined() {
    let plain = solve("hs13_bigstart.nl", "plain", &[]);
    let ho = solve("hs13_bigstart.nl", "ho", HANDOVER);
    assert_eq!(ho.solution.status, ApplicationReturnStatus::SolveSucceeded);
    assert!(
        ho.statistics.final_declared_constr_viol <= 1e-6,
        "the handover's infeasible point was accepted: declared violation {}",
        ho.statistics.final_declared_constr_viol
    );
    assert_same_answer(&plain, &ho, 1e-8);
    assert_eq!(
        ho.statistics.iteration_count, plain.statistics.iteration_count,
        "declined attempts leave the interior trajectory untouched"
    );
}

#[test]
fn off_by_default_and_off_moves_nothing() {
    let plain = solve("airport.nl", "plain", &[]);
    let off = solve("airport.nl", "off", &["ipm_sqp_handover=no"]);
    assert_eq!(
        off.statistics.iteration_count,
        plain.statistics.iteration_count
    );
    assert_eq!(off.solution.objective, plain.solution.objective);
}
