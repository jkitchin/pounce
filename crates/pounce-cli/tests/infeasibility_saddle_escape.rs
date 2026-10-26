//! A local-infeasibility verdict whose point is a saddle of the violation is
//! re-solved from beside it (rung 5 of the second-opinion ladder).
//!
//! # The case
//!
//! Vanderbei `discs` (`fixtures/discs.nl`, 36 variables, 66 rows) is
//! feasible, and pounce solves it at default options. Under
//! `feral_scaling=mc64` the trajectory instead ends with discs 8 and 9 on the
//! same centre. The tangency row between them, `cons1[9,8]`, reads `0 = 4`;
//! its gradient with respect to the centres is exactly zero and the radii are
//! on their lower bound, so the solve reports `Infeasible_Problem_Detected`
//! and every first-order test agrees. Rungs 2–4 end at the same kind of point
//! (rung 1 is `mc64` itself, so it does not open). The same false verdict
//! comes out of main's base solve under `feral_ordering=amd`, `metis` or
//! `scotch` (rung 3 rescues those), and out of upstream Ipopt 3.14.19 with
//! MUMPS at default options.
//!
//! The point is not a local minimizer of the violation: separating the two
//! discs lowers that row's violation to `4 - |delta|^2` in every separating
//! direction. Rung 5 probes for exactly that and restarts from the point it
//! finds; see `crates/pounce-algorithm/src/infeasibility_saddle.rs`.
//!
//! # What is not asserted
//!
//! Which optimum the restart reaches, and its iteration count: any local
//! minimum is a correct answer, and both numbers belong to one platform's
//! trajectory. The base verdict is asserted only through `base_status`.
//!
//! `mc64` is used because it is the setting that defeats every other rung on
//! macOS/aarch64. FERAL's NEON and AVX2 kernels round differently, so on
//! another platform the base solve may solve outright; the arms below then
//! skip the assertions that need the trap, and say so, instead of failing.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

const RUNG: &str = "restart_from_infeasibility_saddle";

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

struct Run {
    out: String,
    report: serde_json::Value,
}

impl Run {
    fn status(&self) -> &str {
        self.report["solution"]["status"].as_str().unwrap_or("")
    }
    fn base_status(&self) -> &str {
        self.report["second_opinion"]["base_status"]
            .as_str()
            .unwrap_or("")
    }
    fn tried(&self) -> Vec<String> {
        self.report["second_opinion"]["tried"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }
    fn promoted_by(&self) -> Option<&str> {
        self.report["second_opinion"]["promoted_by"].as_str()
    }
    /// The base solve fell into the trap, so the arm has something to test.
    fn trapped(&self) -> bool {
        self.base_status() == "Infeasible_Problem_Detected"
    }
}

fn solve(model: &str, extra: &[&str]) -> Run {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let json = dir.join(format!("pounce_saddle_{pid}_{seq}.json"));
    let out = Command::new(pounce_exe())
        .arg(fixture(model))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .args(extra)
        .output()
        .expect("run pounce");
    let text = std::fs::read_to_string(&json).expect("json report written");
    let _ = std::fs::remove_file(&json);
    Run {
        out: format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        report: serde_json::from_str(&text).expect("report parses"),
    }
}

/// The fix: the false verdict is overturned by rung 5, after rungs 2–4 fail.
#[test]
fn a_saddle_of_the_violation_is_escaped() {
    let run = solve("discs.nl", &["feral_scaling=mc64"]);
    if !run.trapped() {
        eprintln!(
            "discs under mc64 solved without the ladder on this platform ({}); \
             nothing to test",
            run.status()
        );
        return;
    }
    assert_eq!(run.status(), "SolveSucceeded", "{}", run.out);
    assert_eq!(run.promoted_by(), Some(RUNG), "{}", run.out);
    assert_eq!(run.tried().last().map(String::as_str), Some(RUNG));
    assert!(
        run.out
            .contains("is not a local minimizer of the constraint violation"),
        "the console says why it restarted: {}",
        run.out
    );
}

/// The switch: with the rung off, the probe does not run and the verdict
/// stands, exactly as before rung 5 existed.
#[test]
fn the_rung_can_be_switched_off() {
    let run = solve(
        "discs.nl",
        &["feral_scaling=mc64", "infeasibility_saddle_escape_retry=no"],
    );
    if !run.trapped() {
        return;
    }
    assert_eq!(run.status(), "InfeasibleProblemDetected", "{}", run.out);
    assert!(!run.tried().iter().any(|t| t == RUNG), "{:?}", run.tried());
    assert!(!run.out.contains("is not a local minimizer"));
}

/// The other branch of the gate. A genuinely infeasible model whose verdict
/// point minimizes the violation produces no escape point, so rung 5 never
/// exists and the ladder is the one it was. `infeasible_equalities` is the
/// fixture that caught the one-sided first draft of the probe: its verdict
/// point minimizes the solver's own measure but not the max norm, which
/// falls on one side at first order.
#[test]
fn a_genuine_infeasibility_verdict_opens_no_saddle_rung() {
    for leg in [&[][..], &["hessian_approximation=limited-memory"][..]] {
        let run = solve("infeasible_equalities.nl", leg);
        assert_eq!(run.status(), "InfeasibleProblemDetected", "{}", run.out);
        assert!(
            !run.tried().iter().any(|t| t == RUNG),
            "{leg:?}: {:?}",
            run.tried()
        );
    }
}
