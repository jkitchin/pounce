//! The active-set SQP arm classifies the model with the caller's options, not
//! with the adapter's built-in defaults.
//!
//! `IpoptApplication::optimize_sqp_tnlp` built its `TNLPAdapter` with
//! `TNLPAdapter::new`, which hard-codes `MakeParameter` and the two default
//! infinity thresholds. So `fixed_variable_treatment`, `nlp_lower_bound_inf`
//! and `nlp_upper_bound_inf` were accepted and discarded on that arm — the
//! gh#677 shape, and the last of the four option families on this arm that
//! were registered and never read.
//!
//! ## Why it went unnoticed, and why it is worse than a no-op
//!
//! The defaults coincide, so it only diverges for a caller who names one. And
//! when they did, the console *agreed with them*: `emit_problem_stats` reads
//! these three options through its own copy of the same code, so the
//! problem-statistics header reported the classification the caller asked for
//! while the adapter that actually solved used the defaults. The header was
//! describing a model the solve was not solving.
//!
//! That is also why the header cannot be the witness here, and why these tests
//! read the *solve* instead. Checking the printed variable counts would have
//! passed on the broken build.
//!
//! What diverges is not a trajectory but a **classification**: a bound at the
//! caller's own infinity threshold is a bound on one arm and no bound on the
//! other, and a fixed variable is eliminated on one arm and retained on the
//! other. The two arms were answering different questions.
//!
//! The fix is one reader (`IpoptApplication::adapter_options`) for all three
//! call sites — the interior-point arm, this arm, and the statistics block —
//! since three copies is what let them drift in the first place.

use std::process::Command;

fn pounce_exe() -> String {
    let mut p = std::path::PathBuf::from(env!("CARGO_BIN_EXE_pounce"));
    p.set_extension(std::env::consts::EXE_EXTENSION);
    p.to_string_lossy().into_owned()
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// `(iterations, objective)` off the end-of-run summary.
///
/// `solver_selection=nlp` is not decoration: `fixed_var_qp.nl` classifies as a
/// convex QP, so `auto` routes it to `pounce-convex` and never reaches the SQP
/// driver at all — the same trap that kept `issue_900`'s cliff fixture from
/// covering this arm. Forcing the NLP route is what puts the model in front of
/// the engine under test.
fn solve(model: &str, extra: &[&str]) -> (i64, f64) {
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(model))
        .arg("--no-sol")
        .arg("solver_selection=nlp")
        .arg("algorithm=active-set-sqp");
    for o in extra {
        cmd.arg(o);
    }
    let out = cmd.output().expect("spawn pounce");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let iters = text
        .lines()
        .find_map(|l| l.trim_start().strip_prefix("Number of Iterations....:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or_else(|| panic!("no iteration count in:\n{text}"));
    let obj = text
        .lines()
        .find_map(|l| l.trim_start().strip_prefix("Objective...............:"))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no objective in:\n{text}"));
    (iters, obj)
}

/// `fixed_variable_treatment` reaches the arm.
///
/// `fixed_var_qp.nl` has a variable with `x_l == x_u`. Under the default
/// `make_parameter` the adapter eliminates it; under `relax_bounds` it is
/// retained with widened bounds, so the engine is handed a problem one
/// variable larger and pays an iteration to certify it.
///
/// The iteration count is the discriminator and the objective is the control:
/// the two treatments describe the same feasible set, so a change in the
/// ANSWER would mean something else had gone wrong. On the parent commit both
/// runs return `0` iterations — the option is discarded — which is the failure
/// this pins.
#[test]
fn fixed_variable_treatment_changes_what_the_sqp_arm_solves() {
    let (default_iters, default_obj) = solve("fixed_var_qp.nl", &[]);
    let (relaxed_iters, relaxed_obj) = solve(
        "fixed_var_qp.nl",
        &["fixed_variable_treatment=relax_bounds"],
    );

    assert_ne!(
        default_iters, relaxed_iters,
        "`fixed_variable_treatment` must reach the active-set SQP adapter; \
         both treatments took {default_iters} iteration(s), which is what a \
         discarded option looks like"
    );
    assert!(
        (default_obj - relaxed_obj).abs() <= 1e-6 * default_obj.abs().max(1.0),
        "the two treatments describe the same feasible set, so the answer must \
         not move: {default_obj:e} against {relaxed_obj:e}"
    );
}

/// The infinity thresholds reach the arm, on the same principle: a bound above
/// `nlp_upper_bound_inf` is not a bound, so lowering the threshold past a real
/// bound removes it from the model.
///
/// Here the ANSWER is the discriminator rather than the iteration count,
/// because removing a bound that is ACTIVE at the optimum genuinely enlarges
/// the feasible set. `boxed_qp_min.nl` sits against its upper box at
/// `obj = 4`; drop the box and the minimum is `0`. Both runs take one
/// iteration either way, which is why the count cannot carry this claim — the
/// first draft of this test used it and failed on `hs71_obj1e8`, where the arm
/// saturates its 200-iteration budget under both settings and the counts are
/// equal for a reason that has nothing to do with the option.
///
/// On the parent commit both runs return `4`: the option is discarded.
#[test]
fn nlp_upper_bound_inf_changes_what_the_sqp_arm_solves() {
    let (_, boxed_obj) = solve("boxed_qp_min.nl", &[]);
    let (_, unboxed_obj) = solve("boxed_qp_min.nl", &["nlp_upper_bound_inf=1"]);

    assert!(
        (boxed_obj - 4.0).abs() < 1e-9,
        "the fixture must still sit against its box at 4 by default, or this \
         test is no longer contrasting anything; got {boxed_obj:e}"
    );
    assert!(
        unboxed_obj.abs() < 1e-9,
        "`nlp_upper_bound_inf` must reach the active-set SQP adapter: with the \
         upper box above the threshold it is not a bound, and the minimum is \
         0, not {unboxed_obj:e}. Equal objectives here are what a discarded \
         option looks like."
    );
}
