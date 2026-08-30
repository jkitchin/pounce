//! The SQP's rescue paths must be *reachable* (gh #855).
//!
//! `sqp_alg` has two re-solves after a step QP fails — the cold-start fallback
//! (gh #349) and the quasi-Newton reset (gh #358 tail) — and an
//! unbounded-model fallback below them (gh #423) that takes a δ-shifted
//! proximal step. Two things kept the rescues from firing.
//!
//! # 1. A warm solve that errors aborted the whole SQP
//!
//! The cold-start fallback is gated on the warm solve's *status*
//! (`MaxIter` / `NumericalError`), so a warm solve that returned `Err` was
//! propagated by `?` out of the algorithm before any retry ran. That is the
//! **stronger** form of the signal the fallback exists for, and it was the one
//! case it could not see.
//!
//! `eigena2` under `algorithm=active-set-sqp` is the instance. At outer
//! iteration 17 the warm solve fails with
//!
//! ```text
//!   pinned KKT constraint block is rank-deficient (inertia shift masked a
//!   singular constraint block); prune to a linearly-independent subset
//! ```
//!
//! which is a statement about the *pinned set* — exactly what a warm start
//! supplies and a cold start rebuilds. The SQP exited `Internal_Error` /
//! `solve_result_num=500`, "the solver broke, retry", on a model whose
//! objective it can report to four figures.
//!
//! # 2. A retry's `Unbounded` verdict was discarded
//!
//! Both retries accepted `Optimal` on the nose, so an `Unbounded` return —
//! an unblocked negative-curvature direction in the null space of the working
//! set, which is a *finding* — was thrown away and `sol` kept the original
//! failure. The gh #423 fallback is gated on `sol.status == Unbounded`, so the
//! proximal step written for exactly this situation was unreachable from a
//! retry.
//!
//! **That half is not covered by this file, and the source says so where it
//! lives.** No fixture in the corpus reaches it: swept under
//! `active-set-sqp`, the retries fire on three fixtures and return only
//! `MaxIter` or `Optimal`, including with `sqp_qp_max_iter` forced to 2.
//! gh #855 observed it in a build carrying second-order certification for the
//! step subproblem — gh #856's subject, which does not exist yet.

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

/// `(status line, objective)` for one solve.
fn solve(name: &str, extra: &[&str]) -> (String, Option<f64>) {
    let tag: String = format!("{name}{}", extra.join("_"))
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let sol = std::env::temp_dir().join(format!("pounce_855_{tag}.sol"));
    let out = Command::new(pounce_exe())
        .arg(fixture(name))
        .arg(&sol)
        .args(extra)
        .output()
        .expect("run pounce");
    let s = String::from_utf8_lossy(&out.stdout).into_owned();
    let status = s
        .lines()
        .filter_map(|l| l.strip_prefix("Status: "))
        .next_back()
        .unwrap_or("<none>")
        .trim()
        .to_string();
    // The line is `Objective...: <scaled> <unscaled>`; the unscaled one is
    // the model's own, and the two differ here (45.833 against 82.5 on the
    // NLP arm, which applies objective scaling where the SQP arm does not).
    let obj = s
        .lines()
        .find(|l| l.trim_start().starts_with("Objective."))
        .and_then(|l| l.split_whitespace().next_back())
        .and_then(|v| v.parse().ok());
    (status, obj)
}

/// The headline: a warm step-QP failure must not take the SQP down with it.
///
/// `Internal_Error` is the CLI's "the solver broke" band (500) and it is the
/// wrong thing to say about a model the same binary can report an objective
/// for. The assertion is on *that*, not on the replacement status, because the
/// point is the absence of a spurious internal error rather than any
/// particular verdict.
#[test]
fn a_warm_step_qp_failure_does_not_abort_the_sqp() {
    let (status, obj) = solve("eigena2.nl", &["algorithm=active-set-sqp"]);
    assert_ne!(
        status, "Internal_Error",
        "a rank-deficient *pinned* block is a statement about the warm \
         working set, and a cold re-solve is the documented remedy; \
         reporting 500 instead says the solver broke"
    );
    let obj = obj.expect("a solve that reports a status reports an objective");
    assert!(
        obj.is_finite(),
        "the returned objective must be a number, got {obj}"
    );
}

/// And the point it reports is the right neighbourhood, not merely a number.
/// `eigena2`'s optimum is `82.5` — the NLP filter line-search arm reaches it
/// on the same file, which is an independent oracle for this arm. The bar is
/// `1e-3` relative rather than tight because the SQP arm ends
/// `Maximum_Iterations_Exceeded`: it lands at `82.5177`, `2.1e-4` out, which
/// is the right neighbourhood and honestly labelled as not converged.
#[test]
fn the_recovered_solve_lands_near_the_known_optimum() {
    let (_status, sqp) = solve("eigena2.nl", &["algorithm=active-set-sqp"]);
    let (nlp_status, nlp) = solve("eigena2.nl", &["solver_selection=nlp"]);
    let sqp = sqp.expect("sqp objective");
    let nlp = nlp.expect("nlp objective");
    assert!(
        nlp_status == "Solve_Succeeded" && (nlp - 82.5).abs() < 1e-6,
        "premise: the NLP arm is the oracle here and should reach 82.5, got \
         {nlp} ({nlp_status})"
    );
    assert!(
        (sqp - nlp).abs() / nlp.abs().max(1.0) < 1e-3,
        "the SQP arm should land on the same optimum as the NLP arm: \
         {sqp} against {nlp}"
    );
}

/// The other arms must be unmoved: this changes a failure path, so anything
/// that was already solving should solve identically. `auto` does not route
/// here at all, which is why the corpus sweep is silent about this file.
#[test]
fn the_default_route_is_untouched() {
    let (status, obj) = solve("eigena2.nl", &[]);
    assert_eq!(status, "Solve_Succeeded");
    assert!((obj.expect("objective") - 82.5).abs() < 1e-6);
}
