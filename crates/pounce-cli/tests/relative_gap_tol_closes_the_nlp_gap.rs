//! `relative_gap_tol`: the NLP arm's strict certificate bounds the max-norm of
//! the complementarity products, but the duality gap is their *sum*, ≈ `N·μ`.
//! MIPLIB `neos-4763324-toguru` (160k products) reached `Solve_Succeeded` on
//! the LP → NLP fallback at a relative gap of `3.1e-6` against `tol = 1e-8`.
//! The fallback now sets `relative_gap_tol = tol`, which refuses a strict
//! certificate whose relative gap is above it and lowers the μ floor.
//!
//! `relative_gap_knapsack_500.nl` is a fractional knapsack: 500 variables in
//! `[0, 1]`, random costs in `[-1, 1]` (seed 7), one `≤` row. Its optimum is
//! known in closed form (greedy by cost ratio), so the test measures the
//! answer against the truth rather than against pounce's own residuals. It is
//! solved on the NLP arm with `bound_relax_factor=0`, as the fallback runs it.
//!
//! One test per branch of the gate. Off, the run certifies `1.3e-8` from the
//! optimum (above `tol`): that is the branch the gate refuses, so the fixture
//! reaches it. On, it certifies `4.7e-9` from the optimum, one iteration later.
//!
//! The fallback's own wiring (`main.rs`, which sets the option to `tol` on the
//! LP → NLP re-solve unless the caller set it) is reached by NETLIB TUFF at
//! `tol=1e-12`: the convex arm declines at `1.4e-12`, and the NLP arm, which
//! certified at `3.9e-9` before, now reaches `4.8e-13` and wins the gh#1183
//! comparison. With `relative_gap_tol=0` the old answer comes back, and
//! `issue_1183_fallback_keeps_the_better_answer.rs` uses that to keep its
//! convex-wins branch.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::return_codes::ApplicationReturnStatus;

/// The fixture's optimum, from the greedy solution of the fractional knapsack.
const F_STAR: f64 = -96.08796432222361;

fn run(fixture: &str, opts: &[&str]) -> SolveReport {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("pounce_relgap_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    std::fs::copy(&src, dir.join("m.nl")).expect("copy fixture");
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .current_dir(&dir)
        .args(["m.nl", "--json-output", "r.json"])
        .args(opts)
        .output()
        .expect("run pounce");
    let json = std::fs::read_to_string(dir.join("r.json")).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);
    serde_json::from_str(&json).unwrap_or_else(|e| {
        panic!(
            "no report ({e}); stderr=\n{}",
            String::from_utf8_lossy(&out.stderr)
        )
    })
}

fn solve(gap_tol: &str) -> SolveReport {
    let gap = format!("relative_gap_tol={gap_tol}");
    run(
        "relative_gap_knapsack_500.nl",
        &[
            "solver_selection=nlp",
            "bound_relax_factor=0",
            "tol=1e-8",
            &gap,
        ],
    )
}

fn rel_err(r: &SolveReport) -> f64 {
    (r.solution.objective - F_STAR) / (1.0 + F_STAR.abs())
}

/// Precondition: without the gate the NLP arm certifies a point further than
/// `tol` from the optimum. If this stops holding, the other test no longer
/// reaches the refusal branch and needs a larger fixture.
#[test]
fn without_the_gate_the_certificate_is_above_tol() {
    let r = solve("0");
    assert_eq!(r.solution.engine, "nlp");
    assert_eq!(r.solution.status, ApplicationReturnStatus::SolveSucceeded);
    let e = rel_err(&r);
    assert!(
        e > 1e-8,
        "relative objective error {e:.2e}: the gate has nothing to refuse"
    );
}

#[test]
fn the_gate_holds_the_certificate_to_tol() {
    let r = solve("1e-8");
    assert_eq!(r.solution.engine, "nlp");
    assert_eq!(r.solution.status, ApplicationReturnStatus::SolveSucceeded);
    let e = rel_err(&r);
    assert!(
        (0.0..=1e-8).contains(&e),
        "relative objective error {e:.2e} with relative_gap_tol=1e-8"
    );
}

/// The LP → NLP fallback sets the gate itself. TUFF at `tol=1e-12`: the
/// convex arm declines, and the NLP arm, held to the gap, beats it.
#[test]
fn the_fallback_holds_the_nlp_arm_to_the_gap() {
    let gated = run("lp_tuff.nl", &["tol=1e-12"]);
    let fb = gated
        .convex_fallback
        .as_ref()
        .expect("TUFF reaches the fallback");
    assert_eq!(fb.winner, "nlp", "{fb:?}");
    let nlp = fb.nlp_kkt_error.expect("NLP error measured");
    assert!(nlp < fb.convex_kkt_error, "{fb:?}");

    // The caller's explicit 0 wins over the fallback's default.
    let off = run("lp_tuff.nl", &["tol=1e-12", "relative_gap_tol=0"]);
    let fb_off = off.convex_fallback.as_ref().expect("fallback");
    assert_eq!(fb_off.winner, "cvx-qp", "{fb_off:?}");
    assert!(
        fb_off.nlp_kkt_error.expect("NLP error measured") > 100.0 * nlp,
        "the gate is what moved the NLP answer: {fb_off:?} vs {fb:?}"
    );
}

/// A refused strict certificate must not be swapped for an acceptable-level
/// one at the same point. Before this was blocked, each refusal counted toward
/// `acceptable_iter`, so a gap that did not close within that many iterations
/// ended `Solved_To_Acceptable_Level` on the very point the gate had refused.
/// `acceptable_iter=1` reaches that branch on the first refusal: it exited
/// acceptable at `1.3e-8` from the optimum; it now certifies below `tol`.
#[test]
fn a_refused_certificate_does_not_exit_acceptable() {
    let r = run(
        "relative_gap_knapsack_500.nl",
        &[
            "solver_selection=nlp",
            "bound_relax_factor=0",
            "tol=1e-8",
            "relative_gap_tol=1e-8",
            "acceptable_iter=1",
        ],
    );
    assert_eq!(r.solution.status, ApplicationReturnStatus::SolveSucceeded);
    let e = rel_err(&r);
    assert!(
        (0.0..=1e-8).contains(&e),
        "relative objective error {e:.2e}"
    );
}
