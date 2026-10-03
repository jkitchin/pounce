//! gh#981 — findings from writing textbook chapters on pounce's interior
//! point method. The fixtures are the issue's own reproductions, generated
//! by `tests/fixtures/issue981.py` (Pyomo, no AMPL).
//!
//! Finding 2 (`line_search_method=cg-penalty`) is pinned in
//! `unimplemented_options.rs` and in `pounce_algorithm::unimplemented_options`;
//! finding 3 (the step-character glossary) is documentation. The two
//! findings it fixes with a numerical symptom are pinned here:
//!
//! * **4 — the point returned with `Infeasible_Problem_Detected`.** It must
//!   be the restoration phase's least-infeasible iterate (the certificate),
//!   not the point where the last restoration started.
//! * **5 — restoration rows in scaled units.** A restoration row's `inf_pr`
//!   must be in the same units as the main rows beside it.
//!
//! Finding 1 (the gh#592 `δ_c` walk-back withdrawing `δ_c` on the
//! rank-deficient `issue981_cstr_dup_row`) is **not fixed** and so not
//! pinned here: both candidate fixes cost other models, measured in
//! `dev-notes/issue-981-delta-c-walkback.md`.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::IterPhase;

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

fn tmp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_issue981_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

/// Solve a fixture with the full per-iteration history in the report.
fn solve(name: &str, extra: &[&str]) -> SolveReport {
    let json_path = tmp_path("report.json");
    let sol_path = tmp_path("out.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(name))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path)
        .arg("--json-detail")
        .arg("full")
        .arg("max_wall_time=300");
    for o in extra {
        cmd.arg(o);
    }
    let _ = cmd.output().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}



/// Finding 4. Restoration settles at the local minimizer of the
/// infeasibility, `x1 = -1` with violation 1.5; the report used to return
/// `x1 = -1.0885` at 1.588 — where the last restoration *started*.
#[test]
fn an_infeasibility_verdict_returns_the_least_infeasible_point() {
    let r = solve("issue981_wachter_biegler.nl", &[]);
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::InfeasibleProblemDetected
    );
    let x1 = r.solution.x[0];
    assert!(
        (x1 + 1.0).abs() < 1e-4,
        "returned x1 = {x1}; the certificate is x1 = -1"
    );
    let viol = r.statistics.final_constr_viol;
    assert!(
        (viol - 1.5).abs() < 1e-4,
        "returned violation {viol}; restoration reached 1.5"
    );
    // The returned point is the one the last restoration row describes.
    let last_resto = r
        .iterations
        .iter()
        .rev()
        .find(|it| it.phase == IterPhase::Restoration)
        .expect("restoration ran");
    assert!(
        (last_resto.inf_pr - viol).abs() < 1e-6,
        "last restoration row reads {} but the returned point violates {}",
        last_resto.inf_pr,
        viol
    );
}

/// Finding 5. The water main's main rows print head violations of order
/// 2e5 m; the row-scaled residual is of order 0.2. Restoration rows used to
/// print the latter under the same `inf_pr` heading.
#[test]
fn restoration_rows_report_inf_pr_in_the_main_rows_units() {
    let r = solve("issue981_water_main.nl", &[]);
    let resto: Vec<_> = r
        .iterations
        .iter()
        .filter(|it| it.phase == IterPhase::Restoration)
        .collect();
    assert!(!resto.is_empty(), "restoration did not run");
    for it in &resto {
        assert!(
            it.inf_pr > 1e4,
            "restoration row {} reads inf_pr = {:e}: that is the row-scaled \
             residual, not metres of head",
            it.iter,
            it.inf_pr
        );
    }
    // The main-phase `R` row that takes the restoration iterate back is
    // the same point as the restoration row before it, so the two must
    // agree, not merely share a magnitude.
    for (i, it) in r.iterations.iter().enumerate().skip(1) {
        let prev = &r.iterations[i - 1];
        if it.phase == IterPhase::Main
            && prev.phase == IterPhase::Restoration
            && it.alpha_primal_char == 'R'
        {
            let rel = (it.inf_pr - prev.inf_pr).abs() / it.inf_pr.max(1.0);
            assert!(
                rel < 1e-6,
                "main R row {} reads {:e}, restoration row before it {:e}",
                it.iter,
                it.inf_pr,
                prev.inf_pr
            );
        }
    }
}
