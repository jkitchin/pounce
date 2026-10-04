//! gh #535: an LP the convex interior-point method cannot certify must be
//! re-solved on the general NLP path under `auto`, not reported as the last
//! word.
//!
//! The models that motivate this are NETLIB `gen`/`gen1`. `auto` classifies
//! them LP and routes them to the convex IPM, which exhausts its 200-iteration
//! budget in 190.8 s and exits `Solved_To_Acceptable_Level` with a primal
//! residual of 1.374e-7 against `tol = 1e-8`; `solver_selection=nlp` solves the
//! same model in 19 iterations and 0.982 s to `Solve_Succeeded`, matching
//! Ipopt-3.14.20/MA57's objective to four figures. They are highly degenerate
//! and rank-deficient, strict complementarity fails, and a pure IPM cannot
//! certify the vertex (gh #133); crossover was built to close that and does
//! not. So the router — not the crossover engine — is the lever, and this is
//! the same defect shape `socp_unverified_falls_back_to_nlp.rs` documents for
//! the conic path: a specialized fast path displaced a general one, and when
//! the fast path failed there was no fallback left.
//!
//! `gen.nl` is n=2560 / m=769 / 63085 Jacobian nonzeros and takes three
//! minutes to fail, so it is not a repo fixture. `lp_afiro.nl` under a
//! tolerance no double-precision solver can reach reproduces the *trigger*
//! exactly and deterministically: the convex IPM's KKT error floors at ~5.7e-14
//! and it burns all 199 iterations, which is the `IterationLimit` half of the
//! contract. The gating tests below — a named engine, a user-set budget, a
//! certified solve, a non-LP class — are the half that keeps the fallback from
//! firing anywhere it should not.
//!
//! gh #724 is the third uncertified exit, `NumericalFailure`, which the gate
//! omitted: the same `lp_afiro` at the same unreachable tolerance, with the
//! documented `qp_tau` option raising the fraction-to-boundary parameter,
//! reaches a singular KKT system *before* the budget runs out and was reported
//! `InternalError` with the uncertified iterate as the answer — on a model the
//! NLP path in the same binary solves to afiro's published optimum. The two
//! failures differ only in which one the convex path happens to hit first, so
//! the tests here assert the **reroute**, not the failure-mode string that
//! produced it.

use std::path::PathBuf;
use std::process::Command;

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::return_codes::ApplicationReturnStatus;

/// Well below the ~5.7e-14 KKT error the convex IPM floors at on `afiro`, so
/// the solve provably cannot certify — on any machine, without depending on a
/// fixture that happens to sit on the accuracy knife-edge.
const UNREACHABLE_TOL: &str = "tol=1e-20";

/// The gh #724 trigger. `qp_tau` is the documented fraction-to-boundary
/// parameter; raising it drives the iterates harder against the boundary, so
/// the KKT system goes singular and the post-solve verification refuses the
/// point at iteration ~157 rather than the budget expiring at 199. Nothing
/// about the model changes — only which uncertified exit the convex path
/// reaches, which is exactly the distinction the gate must not make.
const NUMERICAL_FAILURE_TRIGGER: &str = "qp_tau=0.99";

/// Pairs with the trigger above. gh #744 taught the convex extractors to apply
/// `bound_relax_factor`, and on afiro the widened box is enough to keep the
/// KKT system non-singular: every `qp_tau` from `0.9` to `0.9999` now reaches
/// the iteration limit instead, so the trigger stopped triggering. Solving the
/// model exactly as declared restores the singular path at `~157` iterations.
/// This is a knob to hold the *model* fixed at what the trigger was tuned
/// against — it is not part of what these tests assert.
const EXACT_DECLARED_BOUNDS: &str = "bound_relax_factor=0";

/// afiro's published optimum. The NLP path reaches it on this model whichever
/// way the convex attempt failed, which is what makes reporting the convex
/// failure as the final answer a defect and not a limitation.
const AFIRO_OPT: f64 = -464.753_142_857_142_85;

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

fn run(args: &[&str]) -> (String, String, Option<i32>) {
    run_on("lp_afiro.nl", args)
}

/// Same, but capturing the JSON report — the reported *status* is what a
/// caller sees, and it is where gh #724 showed up as `InternalError`.
fn run_json(args: &[&str]) -> (String, String, SolveReport) {
    let json = std::env::temp_dir().join(format!(
        "pounce_535_{}_{}.json",
        std::process::id(),
        args.join("_").replace(['=', '.', '/'], "-")
    ));
    let out = Command::new(pounce_exe())
        .arg(fixture("lp_afiro.nl"))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .args(args)
        .output()
        .expect("spawn pounce");
    let report: SolveReport =
        serde_json::from_str(&std::fs::read_to_string(&json).expect("read JSON report"))
            .expect("parse JSON report");
    let _ = std::fs::remove_file(&json);
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        report,
    )
}

/// The convex status line, which a rerouted solve must not print.
fn convex_verdict_lines(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .filter(|l| l.contains("IPM, pounce-convex"))
        .collect()
}

/// The headline contract: an LP the convex path could not certify continues on
/// the NLP path instead of ending there.
#[test]
fn an_uncertified_lp_is_re_solved_on_the_nlp_path() {
    let (stdout, stderr, _code) = run(&["solver_selection=auto", UNREACHABLE_TOL]);
    assert!(
        stderr
            .to_lowercase()
            .contains("did not certify a kkt point"),
        "the convex solve must decline; stderr=\n{stderr}"
    );
    assert!(
        stdout.contains("EXIT:"),
        "the NLP path must run and report; stdout=\n{stdout}\nstderr=\n{stderr}"
    );
}

/// The reroute must be announced, and must name where the solve went. A run
/// that silently answers from a different engine than the routing banner named
/// misleads anyone comparing the two.
#[test]
fn the_fallback_says_so() {
    let (_stdout, stderr, _) = run(&["solver_selection=auto", UNREACHABLE_TOL]);
    let lower = stderr.to_lowercase();
    assert!(
        lower.contains("nlp interior-point path"),
        "the reroute must say where the solve went; stderr=\n{stderr}"
    );
    assert!(
        lower.contains("solver_selection=qp-ipm"),
        "the note must say how to see the convex result instead; stderr=\n{stderr}"
    );
}

/// Exactly one verdict. The decision is taken above the convex status line, the
/// `.sol` write and the JSON report, so a rerouted solve leaves no stray convex
/// result for a log scraper (or the benchmark harness) to pick up.
#[test]
fn a_rerouted_solve_reports_one_status_not_two() {
    let (stdout, stderr, _) = run(&["solver_selection=auto", UNREACHABLE_TOL]);
    assert!(
        convex_verdict_lines(&stdout).is_empty(),
        "a rerouted solve must not also print the convex status line; \
         stdout=\n{stdout}\nstderr=\n{stderr}"
    );
}

/// …and nothing else from the discarded attempt either. `afiro` reduces under
/// the convex path's presolve, which used to print its summary before the
/// solve, so a rerouted run left one line of a solve that never reported. The
/// summary is now held back until the convex path is known to be the one that
/// reports — a rerouted run's stdout is the NLP solve's and nothing else.
#[test]
fn a_rerouted_solve_leaves_no_trace_of_the_discarded_attempt() {
    let (rerouted, stderr, _) = run(&["solver_selection=auto", UNREACHABLE_TOL]);
    assert!(
        stderr.to_lowercase().contains("did not certify"),
        "precondition: this run must reroute; stderr=\n{stderr}"
    );
    assert!(
        !rerouted.contains("Presolve:"),
        "the declined convex attempt must not leave its presolve summary \
         behind; stdout=\n{rerouted}"
    );

    // The other half: held back, not lost. A convex solve that *does* report
    // still says what presolve did.
    let (kept, _, _) = run(&["solver_selection=auto"]);
    assert!(
        kept.contains("Presolve:"),
        "a convex solve that reports must still print its presolve summary; \
         stdout=\n{kept}"
    );
}

/// The common case is untouched: an LP the convex IPM certifies is still
/// answered by the convex IPM, with no second solve. This is the 369-of-371
/// case in the LP suite and the reason the fallback triggers on a failure to
/// certify rather than on a class.
#[test]
fn a_certified_lp_still_answers_from_the_convex_path() {
    let (stdout, stderr, code) = run(&["solver_selection=auto"]);
    assert_eq!(code, Some(0), "stdout=\n{stdout}\nstderr=\n{stderr}");
    assert!(
        stdout.contains("Optimal Solution Found"),
        "stdout=\n{stdout}"
    );
    assert_eq!(
        convex_verdict_lines(&stdout).len(),
        1,
        "the convex engine must report its own result; stdout=\n{stdout}"
    );
    assert!(
        !stderr.to_lowercase().contains("did not certify"),
        "a certified solve must not reroute; stderr=\n{stderr}"
    );
}

/// A named engine keeps its verdict. This is what makes the convex stall
/// observable at all, and how the convex result stays available to anyone
/// working on the convex solver itself.
///
/// Asserted as "the convex engine reported, and it did not certify" rather
/// than as a particular failure-mode string. Which uncertified exit `afiro`
/// reaches here is a property of the step rule, not of this contract — this
/// test used to hard-code `"Maximum iterations exceeded"` and went red under a
/// step-rule study (gh #690) that changed nothing it exists to check. Both
/// exits are swept: the budget expiring, and the gh #724 numerical failure.
#[test]
fn an_explicitly_selected_convex_solve_is_not_rerouted() {
    for extra in [
        vec![UNREACHABLE_TOL],
        vec![
            UNREACHABLE_TOL,
            NUMERICAL_FAILURE_TRIGGER,
            EXACT_DECLARED_BOUNDS,
        ],
    ] {
        for sel in ["solver_selection=qp-ipm", "solver_selection=lp-ipm"] {
            let mut args = vec![sel];
            args.extend_from_slice(&extra);
            let (stdout, stderr, _code) = run(&args);
            let verdicts = convex_verdict_lines(&stdout);
            assert_eq!(
                verdicts.len(),
                1,
                "{args:?} must report the convex engine's own result; stdout=\n{stdout}"
            );
            assert!(
                !verdicts[0].contains("Optimal Solution Found"),
                "{args:?}: precondition — the convex solve must fail to certify \
                 at this tolerance; stdout=\n{stdout}"
            );
            assert!(
                !stderr.to_lowercase().contains("did not certify"),
                "{args:?} must not be rerouted; stderr=\n{stderr}"
            );
        }
    }
}

/// gh #724: the same LP, failing the other way. `qp_tau=0.99` makes the convex
/// path exit `NumericalFailure` instead of `IterationLimit`, and that status
/// was missing from the reroute gate — so this run reported `InternalError`
/// with the uncertified convex iterate as the answer.
///
/// The precondition is checked structurally: `InternalError` is the return
/// status only `NumericalFailure` maps to (`qp_status_to_ars`), so a
/// named-engine run reporting it is proof this configuration reaches that exit
/// and that the test is still exercising the branch it was written for.
#[test]
fn an_lp_whose_convex_solve_fails_numerically_is_re_solved_on_the_nlp_path() {
    let (named_out, _, named) = run_json(&[
        "solver_selection=lp-ipm",
        UNREACHABLE_TOL,
        NUMERICAL_FAILURE_TRIGGER,
        EXACT_DECLARED_BOUNDS,
    ]);
    assert_eq!(
        named.solution.status,
        ApplicationReturnStatus::ErrorInStepComputation,
        "precondition: this configuration must reach NumericalFailure on the \
         convex path — it is the only status that reports ErrorInStepComputation \
         there (gh#986: it used to be InternalError); \
         stdout=\n{named_out}"
    );

    // gh#986: the convex numerical failure must render as a known status, not
    // the crash-shaped "Unknown SolverReturn value".
    assert!(
        !named_out.contains("Unknown SolverReturn"),
        "convex NumericalFailure must map to a known exit status; stdout=\n{named_out}"
    );

    let (stdout, stderr, report) = run_json(&[
        "solver_selection=auto",
        UNREACHABLE_TOL,
        NUMERICAL_FAILURE_TRIGGER,
        EXACT_DECLARED_BOUNDS,
    ]);
    assert!(
        stderr
            .to_lowercase()
            .contains("did not certify a kkt point"),
        "an LP the convex path could not verify must reroute, whichever way it \
         failed; stdout=\n{stdout}\nstderr=\n{stderr}"
    );
    assert!(
        convex_verdict_lines(&stdout).is_empty(),
        "the discarded convex attempt must not also report; stdout=\n{stdout}"
    );
    // gh#986 review: assert the *route*, not a status. The convex failure
    // now maps to `ErrorInStepComputation`, which the NLP arm can also
    // legitimately return, so "the status is not ErrorInStepComputation" was
    // a proxy that would fail on a correct reroute and pass on a wrong one
    // the day the convex mapping changes again. The report names the engine
    // that produced the verdict.
    assert_eq!(
        report.solution.engine, "nlp",
        "the verdict must come from the NLP arm the reroute hands the model \
         to, not from the discarded convex attempt; stdout=\n{stdout}\nstderr=\n{stderr}"
    );
}

/// …and the answer it reroutes to is afiro's optimum. This is the part that
/// makes gh #724 a defect rather than a naming quibble: the binary reported
/// `InternalError` on a model it solves, and the objective was right there in
/// the same JSON report the status came from.
#[test]
fn the_rerouted_numerical_failure_reaches_the_published_optimum() {
    let (stdout, stderr, report) = run_json(&[
        "solver_selection=auto",
        UNREACHABLE_TOL,
        NUMERICAL_FAILURE_TRIGGER,
        EXACT_DECLARED_BOUNDS,
    ]);
    // Relative, and loose. `tol=1e-20` is unreachable on the NLP path too, so
    // it stops on a vanishing search direction rather than a certificate and
    // lands ~1e-8 relative from the vertex. Eight figures of afiro's optimum
    // is the claim being made here — the reported answer is the model's — not
    // that an unreachable tolerance was somehow met.
    let obj = report.solution.objective;
    assert!(
        ((obj - AFIRO_OPT) / AFIRO_OPT).abs() < 1e-6,
        "rerouted objective {obj} is not afiro's optimum {AFIRO_OPT}; \
         stdout=\n{stdout}\nstderr=\n{stderr}"
    );
}

/// A user-set iteration budget is the question being asked, so its answer
/// stands. `max_iter=0` is the sharp end of this: the zero-iteration contract
/// (pounce#186) must stop without a solve, and rerouting would launch a full
/// NLP solve for a request that asked for no iterations at all.
#[test]
fn a_user_set_iteration_budget_is_not_a_fallback_trigger() {
    for budget in ["max_iter=0", "max_iter=5"] {
        let (stdout, stderr, _) = run(&["solver_selection=auto", budget]);
        assert!(
            stdout.contains("Maximum iterations exceeded"),
            "{budget} must report the iteration limit; stdout=\n{stdout}"
        );
        assert!(
            !stderr.to_lowercase().contains("did not certify"),
            "{budget} must not trigger the NLP fallback; stderr=\n{stderr}"
        );
    }
}

/// The fallback is scoped to `P = 0`. A convex QP is a different (and
/// unmeasured) population — it keeps the engine chosen for it whatever the
/// tolerance.
#[test]
fn a_convex_qp_is_not_rerouted() {
    let (stdout, stderr, _) = run_on("convex_qp.nl", &["solver_selection=auto", UNREACHABLE_TOL]);
    assert!(
        stdout.contains("Problem class: convex QP"),
        "fixture must classify as a convex QP; stdout=\n{stdout}"
    );
    assert_eq!(
        convex_verdict_lines(&stdout).len(),
        1,
        "a convex QP must report from the convex engine; stdout=\n{stdout}"
    );
    assert!(
        !stderr.to_lowercase().contains("did not certify"),
        "a convex QP must not reroute; stderr=\n{stderr}"
    );
}
