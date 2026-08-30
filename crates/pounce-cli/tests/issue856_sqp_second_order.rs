//! The SQP must not certify a first-order point that is a constrained
//! *maximum* (gh #856).
//!
//! # The instance
//!
//! `nonconvex_qp.nl` is `min x₀x₁ s.t. x₀ + x₁ = 2, 0 ≤ x ≤ 4`. On the
//! feasible segment `f(x₀) = x₀(2 − x₀)` is **concave**, so the `(1, 1)` the
//! SQP converges to at `f = 1` is the constrained maximum and the minimum is
//! `0` at either endpoint. It was reported `Solve_Succeeded`. The NLP filter
//! line-search arm reaches `0` on the same file in the same binary, which is
//! the independent oracle.
//!
//! # Why the check runs at convergence, and not on every step
//!
//! gh #848 gave standalone QP solves a second-order screen, and gh #856
//! explains why handing the same screen to the SQP's **step** subproblem is
//! wrong: that QP is a local model built from the *current* multiplier
//! estimates, and its second-order verdict is not the NLP's. HS071 is the
//! counterexample and it is not exotic — at SQP iteration 0 the multipliers
//! are still zero, so the Lagrangian Hessian is `∇²f`, and started at HS071's
//! own `x*` the step QP's working set leaves a one-dimensional null space on
//! which `dᵀHd = -4.05e-2`. `x*` is refuted, correctly for that model and
//! wrongly for the NLP.
//!
//! At *convergence* the objection disappears, because the multipliers are the
//! converged ones — which is gh #856's own observation ("with the converged
//! multipliers the reduced Hessian is positive"), used here as the design
//! rather than as an obstacle. `sqp_near_solution_start.rs` is the guard that
//! this distinction is real: all four of its HS071 starts stay green.
//!
//! # Refuted by exhibition
//!
//! The direction is only acted on after stepping along it and finding the true
//! objective strictly lower at a point that satisfies the *nonlinear*
//! constraints — the same technique gh #848 uses one layer down. So the
//! curvature search may be approximate: a direction it gets wrong costs two
//! evaluations, not a wrong answer.

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

fn solve(name: &str, extra: &[&str]) -> (String, Option<f64>) {
    let tag: String = format!("{name}{}", extra.join("_"))
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let sol = std::env::temp_dir().join(format!("pounce_856_{tag}.sol"));
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
    // `Objective...: <scaled> <unscaled>`; the unscaled one is the model's.
    let obj = s
        .lines()
        .find(|l| l.trim_start().starts_with("Objective."))
        .and_then(|l| l.split_whitespace().next_back())
        .and_then(|v| v.parse().ok());
    (status, obj)
}

/// The premise, from an arm that is not the one under test: the minimum is
/// `0`, not `1`.
#[test]
fn the_nlp_arm_puts_the_optimum_at_zero() {
    let (status, obj) = solve("nonconvex_qp.nl", &["solver_selection=nlp"]);
    assert_eq!(status, "Solve_Succeeded");
    let obj = obj.expect("objective");
    assert!(obj.abs() < 1e-6, "expected f = 0, got {obj}");
}

/// The headline: the SQP arm must not certify the constrained maximum.
#[test]
fn the_sqp_arm_does_not_certify_the_constrained_maximum() {
    let (status, obj) = solve("nonconvex_qp.nl", &["algorithm=active-set-sqp"]);
    let obj = obj.expect("objective");
    assert!(
        !(status == "Solve_Succeeded" && obj > 1e-6),
        "certified f = {obj} as {status}, but f(x0) = x0(2 - x0) is concave on \
         the feasible segment, so that is the constrained MAXIMUM and (2, 0) \
         is feasible at f = 0"
    );
    assert!(
        obj.abs() < 1e-6,
        "expected the SQP arm to reach f = 0, got {obj} ({status})"
    );
}

/// A second instance the issue does not mention, found by running the fix
/// across the nonconvex fixtures: the SQP arm was returning `0` on a model
/// whose optimum is `-2`. Without this the fix could be tuned to one fixture.
#[test]
fn the_nonconvex_qcqp_reaches_the_same_optimum_as_the_nlp_arm() {
    let (sqp_status, sqp) = solve("nonconvex_qcqp.nl", &["algorithm=active-set-sqp"]);
    let (nlp_status, nlp) = solve("nonconvex_qcqp.nl", &["solver_selection=nlp"]);
    assert_eq!(nlp_status, "Solve_Succeeded", "premise: the oracle solves");
    let (sqp, nlp) = (sqp.expect("sqp obj"), nlp.expect("nlp obj"));
    assert!(
        (nlp + 2.0).abs() < 1e-6,
        "premise: the oracle should reach -2, got {nlp}"
    );
    assert!(
        (sqp - nlp).abs() < 1e-6,
        "the SQP arm should reach the same optimum: {sqp} against {nlp} \
         ({sqp_status})"
    );
}

/// The **limited-memory** leg, which the first version of this fix missed and
/// the SQP-arm sweep caught.
///
/// The escape was gated on `SqpHessianSource::Exact`, on the reasoning that a
/// quasi-Newton Hessian has no negative curvature to find. That is true of the
/// *approximation* and beside the point: a damped-BFGS or L-BFGS matrix is
/// positive definite **by construction**, so the gate was not an optimization,
/// it was the reason the check did not exist under `limited-memory` at all —
/// and that leg went on certifying the same constrained maximum.
///
/// `eval_hess_lag` is a required method of `SqpProblemSpec`, so the exact
/// `∇²L` is always available; it is now taken once at convergence whatever
/// drove the steps. This is not a corner: CLAUDE.md records that the Python
/// frontend and the CasADi plugin both select `limited-memory` on their own
/// whenever no exact Lagrangian Hessian is available, so it is what an
/// embedder gets by default.
///
/// `nonconvex_qp_ineq` is here because it is a *third* wrong answer, one that
/// only the L-BFGS leg was returning: `min x₀x₁ s.t. x₀ + x₁ ≥ 2` over
/// `[0, 4]²` came back at `f = 1`, again the constrained maximum.
#[test]
fn the_limited_memory_leg_is_certified_too() {
    for (name, want) in [
        ("nonconvex_qp.nl", 0.0),
        ("nonconvex_qp_ineq.nl", 0.0),
        ("nonconvex_qcqp.nl", -2.0),
    ] {
        let (status, obj) = solve(
            name,
            &[
                "algorithm=active-set-sqp",
                "hessian_approximation=limited-memory",
            ],
        );
        let obj = obj.expect("objective");
        assert!(
            (obj - want).abs() < 1e-6,
            "{name} under limited-memory: expected f = {want}, got {obj} \
             ({status}) -- a quasi-Newton Hessian is PSD by construction, so \
             the second-order check has to use the exact one"
        );
    }
}

/// A **convex** QP through the same arm is untouched: the escape only looks at
/// an indefinite Lagrangian, and where the reduced Hessian is positive there is
/// no direction to find. Without this the fix could be "always take a step".
#[test]
fn a_convex_model_through_the_sqp_arm_is_unmoved() {
    let (status, obj) = solve("boxed_qp_min.nl", &["algorithm=active-set-sqp"]);
    assert_eq!(status, "Solve_Succeeded");
    let obj = obj.expect("objective");
    let (_s, nlp) = solve("boxed_qp_min.nl", &["solver_selection=nlp"]);
    let nlp = nlp.expect("nlp objective");
    assert!(
        (obj - nlp).abs() / nlp.abs().max(1.0) < 1e-6,
        "convex model moved: {obj} against the NLP arm's {nlp}"
    );
}
