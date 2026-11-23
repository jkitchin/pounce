//! gh #872: the QP second-order verdict must not depend on the units the user
//! chose for their variables.
//!
//! `units_qp_k1.nl` and `units_qp_k1e5.nl` are **the same model** written in two
//! systems of units, `X = K·x` (see `fixtures/units_qp.py`):
//!
//! ```text
//! min  ½(X₀/K)² + ½(X₁/K)² + 5(X₀/K)(X₁/K)     s.t.  −2K ≤ X ≤ 2K
//! ```
//!
//! `H = K⁻²·[[1,5],[5,1]]`, the box scales with `K`, and the minimum is
//! `obj = −16` at `X = (2K, −2K)` for every `K`. Only the units change — metres
//! to micrometres. `|λ_min|/λ_max = 2/3` at both scales, so the Hessian is
//! *strongly* indefinite throughout; nothing here is near roundoff.
//!
//! Before the fix, `K = 1e5` came back:
//!
//! ```text
//! Problem class: convex QP.
//! Number of Iterations....: 0
//! Objective...............:   0.0000000000000000e+00
//! EXIT: Optimal Solution Found.
//! ```
//!
//! Zero iterations, an exactly zero claimed NLP error, and an answer wrong by
//! 100% of the objective. The causal gate was `dispatch::PSD_TOL`, an absolute
//! `1e-9` used both as the classifier's band *and* as the diagonal shift the
//! inertia certificate factors — so at `‖H‖∞ ~ 1e-10` the certificate was
//! reading `1e-9·I` rather than `H`, and was vacuous. Four more absolute floors
//! downstream (`negcurv.rs`'s ladder start, its witness threshold,
//! `active_set.rs`'s `p_scale`, `qp.py`'s `tol_abs`) each independently blocked
//! the rescue once the classifier was fixed, and each is now relative too.
//!
//! **Not covered here:** the NLP arm's own copy of the constant
//! (`NEG_CURV_DELTA_MIN`, `pd_full_space_solver.rs`). Its shift lands on
//! `W + Σ`, not on `H`, so it is not the same one-line change; and on this
//! model the NLP arm agrees with upstream `ipopt`, which also returns `0.0`.
//! That is why the assertions below name `solver_selection=qp-active-set`.

use std::path::PathBuf;
use std::process::Command;

fn run(name: &str, extra: &[&str]) -> String {
    let mut fx = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fx.push("tests");
    fx.push("fixtures");
    fx.push(name);
    let sol = std::env::temp_dir().join(format!("pounce_872_{name}.sol"));
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(fx)
        .arg(&sol)
        .args(extra)
        .output()
        .expect("run pounce");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn objective(stdout: &str) -> Option<f64> {
    stdout
        .lines()
        .find(|l| l.trim_start().starts_with("Objective."))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
}

/// The control: at `K = 1` the model was always solved correctly. It is here so
/// a failure of the `K = 1e5` leg reads as "the answer moved with the units",
/// not "the fixture is wrong".
#[test]
fn the_model_is_solved_at_unit_scale() {
    let out = run("units_qp_k1.nl", &["solver_selection=qp-active-set"]);
    assert!(
        out.contains("Problem class: nonconvex QP"),
        "K = 1 must classify as nonconvex:\n{out}"
    );
    let obj = objective(&out).expect("an objective line");
    assert!(
        (obj - (-16.0)).abs() < 1e-6,
        "K = 1: expected −16, got {obj}\n{out}"
    );
}

/// The regression: the same model at `K = 1e5`.
#[test]
fn the_verdict_survives_a_change_of_variable_units() {
    let out = run("units_qp_k1e5.nl", &["solver_selection=qp-active-set"]);
    assert!(
        out.contains("Problem class: nonconvex QP"),
        "K = 1e5 is the same strongly indefinite model and must classify the \
         same way; classifying it convex is what made the saddle `Optimal`:\n{out}"
    );
    let obj = objective(&out).expect("an objective line");
    assert!(
        (obj - (-16.0)).abs() < 1e-6,
        "K = 1e5: expected −16 (the objective *value* does not depend on the \
         units), got {obj}\n{out}"
    );
}

/// `sqp_qp_certify_second_order` is documented as not affecting standalone QP
/// solves. It does: the standalone default is `yes`, and an explicit `no`
/// reaches this path and turns the certification off — turning a correct answer
/// into a saddle certified as `Optimal Solution Found`. gh#872's doc half, and
/// the inverse of gh#677: an option read on a path documented as not reading it.
///
/// Measured on gh#871's own fixture (`min −x₀² s.t. x₀+x₁+x₂ = 0` over
/// `[0,1]×[−1,1]²`, true minimum `−1` at `x₀ = 1`), because that is a model
/// whose escape runs *through* the certification. `units_qp_k1e5.nl` above is
/// not: the box-only search in `refute_indefinite_optimum` reaches `−16` there
/// either way, which is why this test does not reuse it — a fixture that takes
/// the other branch would have passed while the claim was false.
///
/// Pinned rather than fixed: the plumbing is deliberate (one override, every
/// path), so the defect was in `upstream_options.rs`'s registered help. This
/// test is what stops that text drifting back.
#[test]
fn certify_second_order_no_does_reach_a_standalone_qp_solve() {
    // Both legs opt into the `bound_relax_factor` widening, because that is
    // what makes the escape OBSERVABLE on this fixture and so what makes the
    // with/without contrast a test of the option rather than of the bounds.
    //
    // The origin has `x0` exactly on its declared lower bound with multiplier
    // zero. Widening that bound gives the curvature search strict room and it
    // escapes to -1. On the model as declared -- the convex arm's default
    // since the widening was found to move answers -- this engine instead
    // REFUSES (`InternalError`, srn=500, the rendering gh #848 documents for
    // `NumericalFailure`), which `the_declared_model_is_refused_not_certified`
    // below pins. It does not certify the saddle, so nothing is reported
    // wrongly; the escape is simply unavailable there. Recovering it on the
    // declared model is a `pounce-qp` matter, tracked separately.
    const RELAX: &str = "bound_relax_factor=1e-8";
    let with = run(
        "nonconvex_qp_eq.nl",
        &["solver_selection=qp-active-set", RELAX],
    );
    let without = run(
        "nonconvex_qp_eq.nl",
        &[
            "solver_selection=qp-active-set",
            "sqp_qp_certify_second_order=no",
            RELAX,
        ],
    );
    let with_obj = objective(&with).expect("an objective line");
    let without_obj = objective(&without).expect("an objective line");
    assert!(
        (with_obj - (-1.0)).abs() < 1e-6,
        "the default (yes) must still reach −1, got {with_obj}\n{with}"
    );
    assert!(
        without_obj > -1e-6,
        "with the certification off the standalone solve must stop at the \
         saddle (≈0); if it now also reaches −1 the option stopped reaching \
         this path and the registered help must be updated back, got \
         {without_obj}\n{without}"
    );
}

/// The other branch of the tightened band, and the reason it exists as a
/// separate fixture.
///
/// `psd_band` acts only where `‖H‖∞ < 1`, and measured on this corpus **48 of
/// the 49** fixtures that reach the classifier sit at `‖H‖∞ ≥ 1` — where the
/// `.min(1.0)` clamp makes the change a literal no-op. So the sweep's clean
/// diff was near-tautological evidence, which is gh#690/#760's lesson restated:
/// a corpus uniform in the dimension a change acts on reports "nothing moved"
/// however large its models are.
///
/// `units_qp_convex.nl` is the fixture that is *not* uniform in it: genuinely
/// positive definite at `‖H‖∞ = 1.2e-10`, differing from `units_qp_k1e5.nl`
/// only in the sign and size of the off-diagonal. A fix that tightened the band
/// into rejecting everything small would move this line off `cvx-qp`, and the
/// indefinite test above would still pass. The exact optimum is `−4.98`, at
/// `X₀` on its upper bound.
#[test]
fn a_convex_model_at_the_same_tiny_scale_still_reaches_the_convex_engine() {
    let out = run("units_qp_convex.nl", &[]);
    assert!(
        out.contains("Problem class: convex QP"),
        "the band tightened, but a PD Hessian must still certify PSD at any \
         scale — roundoff is ε·‖H‖∞, sixteen orders below the band:\n{out}"
    );
    let obj = objective(&out).expect("an objective line");
    assert!(
        (obj - (-4.98)).abs() < 1e-6,
        "expected −4.98, got {obj}\n{out}"
    );
}

/// On the model as declared, the standalone active-set engine cannot make the
/// escape and says so rather than certifying the saddle.
///
/// This is the property that actually matters, and it is gh #871's: before
/// that fix this fixture came back `EXIT: Optimal Solution Found.` at `f = 0`.
/// It now refuses. The refusal is weaker than the escape the widened model
/// gets, and it is honest, which the certified saddle was not.
///
/// Note the routing: at the default `solver_selection=auto` this fixture goes
/// to the NLP arm, which relaxes and reaches `-1`. A caller has to ask for
/// `qp-active-set` by name to see the refusal.
#[test]
fn the_declared_model_is_refused_not_certified() {
    let out = run("nonconvex_qp_eq.nl", &["solver_selection=qp-active-set"]);
    let obj = objective(&out).expect("an objective line");
    assert!(
        obj > -1e-6,
        "on the declared model this engine does not reach -1; if it now does, \
         the escape was recovered and this test should become the positive \
         assertion, got {obj}\n{out}"
    );
    assert!(
        !out.contains("Optimal Solution Found"),
        "the saddle must not be certified as optimal (gh #871)\n{out}"
    );
}
