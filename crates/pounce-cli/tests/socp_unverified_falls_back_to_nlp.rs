//! A convex QCQP whose conic solve returns no verified KKT point must be
//! re-solved on the general NLP path, not reported as a failure.
//!
//! The original witness was `airport.nl` (n=84, m=42, from Vanderbei's
//! CUTE-in-AMPL set): it classifies as a convex QCQP, so `auto` routes it to
//! the conic (SOCP) interior-point solver, which *stalled* there at ~9.5e-4
//! complementarity while the NLP filter line-search solved it in 15 iterations.
//! Refusing to certify that point was right, but refusing was previously
//! *terminal*, and the problem was reported `Restoration_Failed`
//! (`solve_result_num` 500).
//!
//! gh #1050 removed that witness: forming the SOC Mehrotra corrector in the
//! NT-scaled space made the conic arm solve `airport` outright (18
//! iterations, the NLP's objective), which `airport_now_solves_on_the_conic_arm`
//! pins. A fallback with no failing input is untested, so the contract below
//! moved to `qcqp_ball_cost_1e22.nl` — `qcqp_ball` with its cost vector scaled
//! to `(1e22, 1e22)` — which the conic arm then failed across `1e15 … 1e24`.
//!
//! gh #1056 removed that one too, and its anisotropic sibling (cost
//! `(1e22, 1)`, `qcqp_ball_cost_1e22_by_1.nl`) with it: the `σ` cost
//! normalization judged its conic answer with orthant residuals, and when it
//! did reject one a cone program had no rung left but the un-normalized
//! re-solve. `cost_scaled_balls_now_solve_on_the_conic_arm` pins both.
//!
//! The next witness, `qcqp_ball_rhs_1e20.nl` (the ball's right-hand side at
//! `1e20`, inactive inside the box, optimum `−70`), went the same way within
//! gh #1056: the rotated cone was emitted as `(s+1, s−1, …)`, and `1e20 ± 1`
//! round to the same double, so the declared cone was empty. Balanced, it
//! solves; `retired_witnesses_now_solve_on_the_conic_arm` pins all three.
//!
//! The contract now runs on `qcqp_ball_cost_1e22.nl` under `qp_hsde=no`. That
//! option selects the conic arm's *direct* driver, which has no cost
//! normalization — `σ` lives in the HSDE driver only — and it degrades from a
//! cost of `3e6` and ends `Numerical failure` from `3e8` (185 iterations
//! here). The NLP path does not read `qp_hsde`, so the rerouted solve returns
//! `−2√2·1e22` exactly as before. Four witnesses in, the lesson is that a
//! fixture-only witness is retired by every fix to the arm it exercises; this
//! one is a documented limit of an opt-out driver, and if that driver learns
//! cost scaling the explicit-`socp` test below fails and this file needs a new
//! witness again — which is the point: the fallback must never sit untested.
//!
//! Extreme *constraint* magnitudes (a `1e300` row) were tried first and
//! rejected: the conic arm fails on them too, but the NLP re-solve either ends
//! `Restoration Failed` or trips FERAL's debug NaN/Inf assertion, so they
//! test the NLP path's limits rather than the routing.
//!
//! That is the same defect shape as gh #413: a specialized fast path displaced
//! a general one, and when the fast path failed there was no fallback left.
//! The dispatcher already routes *large* convex QCQPs to the NLP path before
//! solving, on the reasoning (`dispatch.rs`) that "a convex QCQP is still a
//! valid NLP, so the fallback is sound" — this applies the same reasoning
//! after the fact.
//!
//! The `socp` arm below is the other half of the contract, and the reason the
//! fallback is gated on `auto`: when the user names the engine, its verdict
//! must stand. Silently answering from a different solver would hide exactly
//! the failure this test documents.

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

/// The witness: the cost-scaled ball on the direct conic driver (see the
/// module doc). `qp_hsde=no` goes first so a test's own arguments follow it.
fn run(args: &[&str]) -> (String, String, Option<i32>) {
    let mut all = vec!["qp_hsde=no"];
    all.extend_from_slice(args);
    run_on("qcqp_ball_cost_1e22.nl", &all)
}

fn run_on(name: &str, args: &[&str]) -> (String, String, Option<i32>) {
    let out = Command::new(pounce_exe())
        .arg(fixture(name))
        .arg("--no-sol")
        .args(args)
        .output()
        .expect("spawn pounce");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code(),
    )
}

/// The headline contract: under `auto` the failed conic solve is not the
/// final word. (On `airport`, before the fallback: `Restoration_Failed`,
/// exit 1.)
#[test]
fn unverified_conic_solve_is_re_solved_on_the_nlp_path() {
    let (stdout, stderr, code) = run(&["solver_selection=auto"]);
    assert_eq!(
        code,
        Some(0),
        "must exit 0; stdout=\n{stdout}\nstderr=\n{stderr}"
    );
    let lower = stdout.to_lowercase();
    assert!(
        lower.contains("optimal solution found"),
        "must report an optimum after falling back; stdout=\n{stdout}"
    );
    assert!(
        // Not bare "restoration": the NLP summary's "Number of restoration
        // iterations" line is ordinary output.
        !lower.contains("numerical failure") && !lower.contains("restoration failed"),
        "the conic failure must not be the reported verdict; stdout=\n{stdout}"
    );
}

/// The reroute must be announced. A solve that silently answers from a
/// different engine than the banner named is worse than a slow one — anyone
/// comparing engines would be reading the wrong solver's numbers.
#[test]
fn the_fallback_says_so() {
    let (_stdout, stderr, _) = run(&["solver_selection=auto"]);
    let lower = stderr.to_lowercase();
    assert!(
        lower.contains("no verified kkt point") && lower.contains("nlp"),
        "the reroute must be explained on stderr; stderr=\n{stderr}"
    );
}

/// Exactly one verdict. The fallback decision is taken before the conic path
/// prints its status line, writes the `.sol` or emits the JSON report, so a
/// rerouted solve must not leave a stray conic result behind for a log scraper
/// (or the benchmark harness) to pick up.
#[test]
fn a_rerouted_solve_reports_one_status_not_two() {
    let (stdout, _stderr, _) = run(&["solver_selection=auto"]);
    let conic_lines = stdout
        .lines()
        .filter(|l| l.contains("conic IPM, pounce-convex"))
        .count();
    assert_eq!(
        conic_lines, 0,
        "a rerouted solve must not also print the conic status line; stdout=\n{stdout}"
    );
}

/// An explicitly requested engine keeps its verdict: no reroute, no note. This
/// is what makes the failure observable, and it is how the conic result stays
/// available to anyone debugging the conic solver itself.
#[test]
fn an_explicitly_selected_conic_solve_is_not_rerouted() {
    let (stdout, stderr, _code) = run(&["solver_selection=socp"]);
    assert!(
        stdout.contains("conic IPM, pounce-convex"),
        "solver_selection=socp must report the conic engine's own result; stdout=\n{stdout}"
    );
    assert!(
        stdout.to_lowercase().contains("numerical failure"),
        "the conic verdict must stand when the engine was named; stdout=\n{stdout}"
    );
    assert!(
        !stderr.to_lowercase().contains("re-solved"),
        "an explicit selection must not be rerouted; stderr=\n{stderr}"
    );
}

/// `max_iter=0` must still stop without a solve. The zero-iteration contract
/// (pounce#186) returns `IterationLimit`, which is deliberately *not* a
/// fallback trigger — rerouting it would run a full NLP solve for a request
/// that asked for no iterations at all.
#[test]
fn zero_iteration_contract_is_not_a_fallback_trigger() {
    let (stdout, stderr, _) = run(&["solver_selection=auto", "max_iter=0"]);
    assert!(
        stdout.contains("Maximum iterations exceeded"),
        "max_iter=0 must report the iteration limit; stdout=\n{stdout}"
    );
    assert!(
        !stderr.to_lowercase().contains("re-solved"),
        "max_iter=0 must not trigger the NLP fallback; stderr=\n{stderr}"
    );
}

/// gh #1050: the model that used to witness the fallback now solves on the
/// conic arm itself — under `auto` with no reroute note, and under an explicit
/// `socp` with the conic engine's own `Optimal` at the NLP's objective
/// (`47952.70168`, Ipopt-MA57's to nine figures). Before the NT-scaled SOC
/// corrector the explicit solve ended `Numerical failure` at 16 iterations.
#[test]
fn airport_now_solves_on_the_conic_arm() {
    let (stdout, stderr, code) = run_on("airport.nl", &["solver_selection=socp"]);
    assert_eq!(code, Some(0), "stdout=\n{stdout}\nstderr=\n{stderr}");
    assert!(
        stdout.contains("conic IPM, pounce-convex): Optimal Solution Found."),
        "the conic arm must certify airport itself; stdout=\n{stdout}"
    );
    assert!(
        stdout.contains("obj=47952.7016"),
        "objective must match the NLP/Ipopt optimum; stdout=\n{stdout}"
    );

    let (stdout, stderr, code) = run_on("airport.nl", &["solver_selection=auto"]);
    assert_eq!(code, Some(0), "stdout=\n{stdout}\nstderr=\n{stderr}");
    assert!(
        !stderr.to_lowercase().contains("no verified kkt point"),
        "auto must not need the NLP fallback on airport any more; stderr=\n{stderr}"
    );
}

/// gh #1056: the three previous witnesses are solved by the conic arm itself
/// — `Optimal` at the known optimum under an explicit `socp`, and with no
/// reroute under `auto` — and so is the *active* huge ball (no box, radius
/// `1e10`, optimum `−5e10`), which failed on the conic arm from radius `1e4`
/// up and which the NLP path cannot solve at all (`Search Direction is
/// becoming Too Small`). Before: the isotropic cost ball ended `Numerical
/// failure` (and across `1e16 … 1e19` `IterationLimit`, which `auto` cannot
/// rescue), the anisotropic one `Numerical failure` at 159 iterations, and the
/// inactive huge ball `Numerical failure` at 13.
#[test]
fn retired_witnesses_now_solve_on_the_conic_arm() {
    for (name, f_star) in [
        ("qcqp_ball_cost_1e22.nl", -2.0 * 2.0_f64.sqrt() * 1e22),
        ("qcqp_ball_cost_1e22_by_1.nl", -2e22),
        ("qcqp_ball_rhs_1e20.nl", -70.0),
        ("qcqp_ball_rhs_1e20_active.nl", -5e10),
    ] {
        let (stdout, stderr, code) = run_on(name, &["solver_selection=socp"]);
        assert_eq!(
            code,
            Some(0),
            "{name}: stdout=\n{stdout}\nstderr=\n{stderr}"
        );
        assert!(
            stdout.contains("conic IPM, pounce-convex): Optimal Solution Found."),
            "{name}: the conic arm must certify it; stdout=\n{stdout}"
        );
        let obj: f64 = stdout
            .split("obj=")
            .nth(1)
            .and_then(|t| t.split_whitespace().next())
            .and_then(|t| t.parse().ok())
            .unwrap_or_else(|| panic!("{name}: no obj= on the status line; stdout=\n{stdout}"));
        assert!(
            ((obj - f_star) / f_star).abs() < 1e-8,
            "{name}: objective {obj:e} vs {f_star:e}; stdout=\n{stdout}"
        );

        let (stdout, stderr, code) = run_on(name, &["solver_selection=auto"]);
        assert_eq!(
            code,
            Some(0),
            "{name}: stdout=\n{stdout}\nstderr=\n{stderr}"
        );
        assert!(
            !stderr.to_lowercase().contains("no verified kkt point"),
            "{name}: auto must not need the NLP fallback; stderr=\n{stderr}"
        );
    }
}
