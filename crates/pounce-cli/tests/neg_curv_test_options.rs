//! `neg_curv_test_reg` reaches the factorization loop (#551 / #677).
//!
//! `neg_curv_test_tol` and `neg_curv_test_reg` configure the inertia-free
//! curvature test of Zavala & Chiang (2014): with a positive tolerance the
//! augmented system is factored without the inertia check, and a
//! factorization is accepted only if the direction it produced clears
//!
//! ```text
//!     dxᵀ W dx + dxᵀ Σ_x dx + dsᵀ Σ_s ds [+ δ_x‖dx‖² + δ_s‖ds‖²]
//!         ≥ neg_curv_test_tol · (‖dx‖² + ‖ds‖²)
//! ```
//!
//! `neg_curv_test_reg` is the bracketed term. Both options were registered
//! and never read — `neg_curv_test_tol` had a field that only ever held its
//! `0.0` default, `neg_curv_test_reg` had no field at all.
//!
//! `crates/pounce-algorithm/tests/barrier_kkt_options.rs` covers the
//! tolerance end to end on a small nonconvex TNLP. The regularization flag
//! needs more than that problem can supply: it only changes the test's
//! verdict once δ_x is nonzero, i.e. on an iterate whose inertia is *still*
//! wrong after the perturbation has been escalated at least once. `csfi2`
//! is such a problem — a nonconvex corpus fixture whose solve spends its
//! middle iterations in exactly that state — so it is what proves the flag
//! is load-bearing rather than parsed and dropped.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;

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
        "pounce_neg_curv_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

fn solve(fixture_name: &str, extra_opts: &[&str]) -> SolveReport {
    let json_path = tmp_path(&format!("{fixture_name}.json"));
    let sol_path = tmp_path(&format!("{fixture_name}.sol"));
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(fixture_name))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path);
    for opt in extra_opts {
        cmd.arg(opt);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// The objective `csfi2` reaches; every configuration below must still
/// land on it, because the curvature test changes which factorizations are
/// accepted, not which problem is being solved.
const CSFI2_OBJECTIVE: f64 = 55.017_5;

fn assert_csfi2_answer(report: &SolveReport, ctx: &str) {
    let obj = report.solution.objective;
    assert!(
        (obj - CSFI2_OBJECTIVE).abs() / CSFI2_OBJECTIVE < 1e-3,
        "{ctx}: objective {obj} is not csfi2's optimum {CSFI2_OBJECTIVE} \
         (status={:?})",
        report.solution.status,
    );
}

/// With the curvature test switched on, dropping the primal
/// regularization out of it must change the solve. Nothing about the
/// *direction* of the change is asserted — `neg_curv_test_reg=no` is
/// upstream's "original IPOPT approach", not a worse one — only that the
/// flag reaches the test that consumes it.
#[test]
fn neg_curv_test_reg_changes_the_solve() {
    let with_reg = solve(
        "csfi2.nl",
        &["neg_curv_test_tol=1e2", "neg_curv_test_reg=yes"],
    );
    let without_reg = solve(
        "csfi2.nl",
        &["neg_curv_test_tol=1e2", "neg_curv_test_reg=no"],
    );
    eprintln!(
        "csfi2 @ neg_curv_test_tol=1e2: reg=yes -> {} iters ({:?}), \
         reg=no -> {} iters ({:?})",
        with_reg.statistics.iteration_count,
        with_reg.solution.status,
        without_reg.statistics.iteration_count,
        without_reg.solution.status,
    );
    assert_csfi2_answer(&with_reg, "csfi2 reg=yes");
    assert_csfi2_answer(&without_reg, "csfi2 reg=no");
    assert_ne!(
        with_reg.statistics.iteration_count, without_reg.statistics.iteration_count,
        "neg_curv_test_reg did not change the solve ({} iterations either way) \
         — the option is parsed but the curvature test is not reading it",
        with_reg.statistics.iteration_count,
    );
}

/// The companion check on the same fixture: the tolerance itself must
/// reach the loop. Off is the default (inertia check, no heuristic).
#[test]
fn neg_curv_test_tol_changes_the_solve() {
    let off = solve("csfi2.nl", &[]);
    let on = solve("csfi2.nl", &["neg_curv_test_tol=1e2"]);
    eprintln!(
        "csfi2: neg_curv_test_tol off -> {} iters ({:?}), 1e2 -> {} iters ({:?})",
        off.statistics.iteration_count,
        off.solution.status,
        on.statistics.iteration_count,
        on.solution.status,
    );
    assert_csfi2_answer(&off, "csfi2 tol=off");
    assert_csfi2_answer(&on, "csfi2 tol=1e2");
    assert_ne!(
        off.statistics.iteration_count, on.statistics.iteration_count,
        "neg_curv_test_tol did not change the solve ({} iterations either way) \
         — the option is parsed but is not reaching `PdFullSpaceSolver`",
        off.statistics.iteration_count,
    );
}
