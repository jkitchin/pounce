//! gh #876 — a `NaN` iterate on the active-set SQP arm is not an optimal
//! solution.
//!
//! `unbounded_cubic.nl` under `algorithm=active-set-sqp
//! hessian_approximation=limited-memory` printed, in this order:
//!
//! ```text
//! Objective...............:   nan    nan
//! Overall NLP error.......:   0.0000000000000000e+00    0.0000000000000000e+00
//! EXIT: Optimal Solution Found.
//! Status: Solve_Succeeded
//! ```
//!
//! A perfect zero KKT error beside a `nan` objective, and success reported on
//! the strength of the zero. The mechanism is `f64::max`'s definition: it
//! *ignores* `NaN`, so `fold(0.0, f64::max)` over an all-`NaN` vector is
//! `0.0`, and `0.0 <= tol` passes. `check_kkt` reduced **both** of its
//! residuals that way — the stationarity rows through the fold the issue
//! names, and the constraint violation one layer earlier through
//! `(bl - c).max(0.0)`, which turns a `NaN` row into a tidy "perfectly
//! feasible" `0.0` before the outer `max` ever sees it. Fixing only the
//! reported site would have left `constr_viol` lying.
//!
//! This is the third instance of the shape in this workspace: gh #222 in
//! `pounce-convex` (which has carried a short-circuiting `inf_norm` ever
//! since), gh #845 in `pounce-sensitivity`, and now the SQP arm.
//!
//! The residual-level assertions live next to the code, in
//! `pounce-algorithm`'s `sqp::sqp_alg::kkt_nan_tests`, with the mutation
//! table showing that neither half of the fix subsumes the other. What is
//! only reachable from *here* is the plumbing: that an honest `NaN` residual
//! actually reaches the user as a failure verdict rather than falling into
//! some other gate that rounds it back to success.
//!
//! ## What this file is NOT evidence about
//!
//! * **`Invalid_Number_Detected` is not the same claim as
//!   `Diverging_Iterates`.** `unbounded_cubic.nl` is unbounded, and the
//!   *exact* leg says so — its step QP certifies a recession ray and the arm
//!   returns `SqpStatus::Unbounded` (gh #388). The limited-memory leg cannot:
//!   an L-BFGS matrix is positive definite by construction, so its step QP is
//!   never unbounded, the ray is never certified, and the iterates run off
//!   numerically instead. `Invalid_Number_Detected` is the honest report of
//!   what happened on that leg — "the arithmetic went non-finite and we
//!   stopped" — not a claim about boundedness. Giving the SQP arm its own
//!   `diverging_iterates_tol` guard, so both legs reach the *same* verdict on
//!   this model, is a trajectory change and is deliberately not attempted
//!   here.
//! * **Only the SQP arm.** The interior-point arm has screened this since it
//!   was written (`ipopt_alg.rs`, `if !nlp_err.is_finite()`), and this fix is
//!   modelled on it precisely so the two arms cannot disagree about what a
//!   `NaN` iterate means.
//! * **Not a sweep result.** The default-arm and SQP-arm fixture sweeps run
//!   for this change are a no-collateral-damage check; see the PR. What they
//!   did turn up is worth recording here, because no test in this file
//!   asserts it: the default arm is unmoved on all 180 fixture-legs, and the
//!   SQP arm moves exactly **two**, both on the limited-memory leg and both
//!   on unbounded models. `unbounded_cubic` is the reported defect
//!   (`SolveSucceeded` it=9 → `InvalidNumberDetected` it=8). `unbounded_exp`
//!   is a *second* manifestation the issue does not mention:
//!   `MaximumIterationsExceeded` it=200 → `InvalidNumberDetected` it=3. It
//!   was never falsely optimal, because its iterates went to `±inf` rather
//!   than `NaN` and `inf.abs()` survives a `max`-fold intact — so it failed
//!   the tolerance honestly and then ground out 197 further iterations on an
//!   iterate that had stopped meaning anything. Same defect, other flavour of
//!   non-finite, and the reason the screen tests `is_finite()` rather than
//!   `is_nan()`.

use std::path::PathBuf;
use std::process::Command;

const SQP: &str = "algorithm=active-set-sqp";
const LBFGS: &str = "hessian_approximation=limited-memory";

fn run(name: &str, extra: &[&str]) -> String {
    let mut fx = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fx.push("tests");
    fx.push("fixtures");
    fx.push(name);
    let sol = std::env::temp_dir().join(format!("pounce_876_{name}.sol"));
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(fx)
        .arg(&sol)
        .args(extra)
        .output()
        .expect("run pounce");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn status(stdout: &str) -> String {
    stdout
        .lines()
        .rev()
        .find_map(|l| l.trim().strip_prefix("Status:"))
        .unwrap_or("<none>")
        .trim()
        .to_string()
}

/// The reported defect, verbatim. Both halves are asserted, because the
/// status alone would also be satisfied by the arm failing for some unrelated
/// reason: the printed KKT error must stop being a zero, *and* the verdict
/// must stop being success.
#[test]
fn a_nan_iterate_is_not_reported_as_an_optimal_solution() {
    let out = run("unbounded_cubic.nl", &[SQP, LBFGS]);
    assert_ne!(
        status(&out),
        "Solve_Succeeded",
        "a NaN objective was certified optimal; full output:\n{out}"
    );
    assert_eq!(status(&out), "Invalid_Number_Detected", "output:\n{out}");

    // What went non-finite must show in the report: the KKT error, or the
    // objective. It is asserted as "not finite" and not as the string `nan`,
    // because *which* non-finite value the arm arrives at is not portable:
    // the same fixture reached `nan` in the release build and `inf` in the
    // debug one, since the two take different rounding through the same
    // divergent trajectory. Both are correct answers to the question this
    // test asks, and pinning one of them would be an assertion about the
    // optimizer settings rather than about the fix — the shape of mistake
    // `issue848_sqp_second_order_option.rs` records under "which fixture can
    // carry which claim".
    //
    // Either quantity, not the KKT error alone, since gh#1061. Until then
    // this leg's step QPs stalled at their iteration cap every few
    // iterations (the refactor path lacked the null-iteration guard), and
    // the cold re-solves that followed sent the iterate into a `NaN`
    // residual. With the guard they solve, the SQP follows the cubic down
    // until the objective overflows to `-inf` while the residuals are still
    // finite (`7.4e305`), and the screen fires on the objective. The
    // residual-level half of gh #876 — a `NaN` residual not reducing to `0.0`
    // — no longer has an end-to-end witness in this fixture; it is held by
    // `pounce-algorithm`'s `sqp::sqp_alg::kkt_nan_tests`.
    let reported = |prefix: &str| -> f64 {
        let line = out
            .lines()
            .find(|l| l.trim_start().starts_with(prefix))
            .unwrap_or_default()
            .to_string();
        line.rsplit(':')
            .next()
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("could not parse the {prefix:?} line: {line:?}"))
    };
    let (err, obj) = (reported("Overall NLP error"), reported("Objective"));
    assert!(
        !err.is_finite() || !obj.is_finite(),
        "the report must carry the non-finite quantity the solve stopped on, \
         not a reduction that swallowed it; got KKT error {err}, objective {obj}"
    );
}

/// The **other** branch of the same model, and the reason the claim above is
/// scoped the way it is. The exact leg reaches its verdict through
/// `SqpStatus::Unbounded`, never touches the finiteness screen, and must be
/// bit-for-bit unmoved by this change.
#[test]
fn the_exact_leg_still_reports_the_unboundedness_it_can_certify() {
    let out = run("unbounded_cubic.nl", &[SQP]);
    assert_eq!(status(&out), "Diverging_Iterates", "output:\n{out}");
}

/// The accept branch: the screen must not fire on models that are merely
/// hard. Three fixtures × both legs, all of which converge, so a screen that
/// over-triggers — say, one that tested `viol` for finiteness *before* the
/// `max(0.0)` clamps rather than testing the inputs — turns these red.
#[test]
fn ordinary_models_are_not_newly_rejected_on_either_leg() {
    for fx in [
        "degenerate_start_hs008.nl",
        "boxed_qp_min.nl",
        "nonconvex_qp.nl",
    ] {
        for leg in [vec![SQP], vec![SQP, LBFGS]] {
            let out = run(fx, &leg);
            assert_eq!(
                status(&out),
                "Solve_Succeeded",
                "{fx} on {leg:?} regressed; output:\n{out}"
            );
        }
    }
}
