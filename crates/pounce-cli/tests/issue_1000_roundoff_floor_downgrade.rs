//! gh#1000 (II.22) — the gh#983 item-5 downgrade does not fire on a
//! stationarity residual at the rounding floor of its own terms.
//!
//! Both fixtures are run the way discopt runs an incumbent NLP: the NLP arm,
//! `bound_relax_factor = 0`, `constr_viol_tol = 1e-8`. Without the relaxation
//! an active bound and a row become linearly dependent, the multipliers drift
//! along the unbounded ray of valid ones, and the run carries the gh#884
//! runaway-multiplier signature.
//!
//! * `pump_network_h148_brf0` (the optimization book's pump network, NLP(y)
//!   at `H0_max = 148`): residual `5.5e-3` against terms of `1.57e13` — 1.6
//!   ulps — with the objective right to `5e-9` relative (the convex QCQP arm
//!   finds `130209.66232452`). It was downgraded on the `5.5e-3` alone.
//! * `haverly_multistart_t3_brf0` (Haverly pooling, multistart trial 3): the
//!   control. Residual `9.4e-3` against `6.6e10` (140 ulps), objective `7e-8`
//!   off and complementarity `1.8e-6`; the downgrade is right and stays.
//!   Ipopt 3.14 follows the same 61-iteration trajectory and reports it
//!   optimal. The cause is the `bound_relax_factor = 0` the caller chose, so
//!   the warning names it, and at the default relaxation the same model ends
//!   `Solve_Succeeded` (items II.23 and I.15).
//!
//! The control now runs `haverly_runaway_start_brf0` (gh#1098): the same
//! `.nl` as `haverly_multistart_t3_brf0` with a different starting point (a
//! uniform draw within the bounds) and nothing else changed. The runaway
//! multiplier on trial 3 is a property of the inertia correction's
//! `δ_s = δ_x`: at `perturb_slack_fact=0.03` trial 3 converges `Optimal` in 57
//! iterations at an unscaled dual of `1.25e-11`, with no runaway and nothing to
//! downgrade, so the control no longer reached the branch it controls for.
//! From the new start the runaway is there at every factor measured: residual
//! `1.67e-2` against terms of `3.72e10` (about 2000 ulps), `Solved_To_Acceptable_Level`
//! in 47 iterations at objective `-99.99999262`, the downgrade and the
//! `bound_relax_factor = 0` warning both present — at `perturb_slack_fact` 1,
//! 0.3, 0.1, 0.03 and 0.01, 17/17 under the round-off screen
//! (`mu_init = 0.1·(1 ± k·1e-12)`) at each. At the default relaxation it ends
//! `Optimal` with no `bound_relax_factor` warning, as trial 3 did.

use std::path::PathBuf;
use std::process::Command;

fn run(name: &str) -> String {
    run_with(name, "0")
}

fn run_with(name: &str, bound_relax_factor: &str) -> String {
    let mut fx = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fx.push("tests");
    fx.push("fixtures");
    fx.push(name);
    let brf = format!("bound_relax_factor={bound_relax_factor}");
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fx)
        .args(["solver_selection=nlp", brf.as_str(), "constr_viol_tol=1e-8"])
        .output()
        .expect("run pounce");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn exit_line(out: &str) -> &str {
    out.lines()
        .rfind(|l| l.starts_with("EXIT:"))
        .unwrap_or_else(|| panic!("no EXIT line:\n{out}"))
}

#[test]
fn a_residual_at_the_roundoff_floor_keeps_its_success_verdict() {
    let out = run("pump_network_h148_brf0.nl");
    assert_eq!(exit_line(&out), "EXIT: Optimal Solution Found.", "{out}");
    assert!(!out.contains("status is downgraded"), "{out}");
    // The scale caveat still tells the user what the multipliers did.
    assert!(out.contains("WARNING: large_dual_scale:"), "{out}");
}

#[test]
fn a_residual_above_the_floor_is_still_downgraded() {
    let out = run("haverly_runaway_start_brf0.nl");
    assert_eq!(
        exit_line(&out),
        "EXIT: Solved To Acceptable Level.",
        "{out}"
    );
    assert!(out.contains("status is downgraded"), "{out}");
    assert!(
        out.lines()
            .any(|l| l.starts_with("WARNING: large_dual_scale:")
                && l.contains("bound_relax_factor = 0 is set")),
        "{out}"
    );
}

#[test]
fn the_default_relaxation_certifies_the_same_model() {
    let out = run_with("haverly_runaway_start_brf0.nl", "1e-8");
    assert_eq!(exit_line(&out), "EXIT: Optimal Solution Found.", "{out}");
    assert!(!out.contains("bound_relax_factor = 0 is set"), "{out}");
}
