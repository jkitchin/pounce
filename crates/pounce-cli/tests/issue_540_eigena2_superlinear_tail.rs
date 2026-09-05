//! The inertia test must not be run against a factorization that cannot
//! answer it (gh #540).
//!
//! WHAT WENT WRONG. CUTE `eigena2` stopped at `Solved To Acceptable Level`
//! with the dual infeasibility stuck at 3.3e-7, where Ipopt certifies
//! `Optimal` at 5.2e-9. The objective was already right to twelve digits;
//! only the last two orders of the dual residual were missing. The reported
//! symptom was that `δ_w` re-escalated from `10^-0.8` to `10^1.4` at the
//! second-to-last iteration and damped the Newton step from `1.2e-7` to
//! `8.2e-9`, which is where the superlinear tail went.
//!
//! WHY. Not the `δ_w` update rule — that is an exact port of upstream's
//! `get_deltas_for_wrong_inertia` and it did what it is told. The input was
//! wrong. `eigena2`'s constraint Jacobian degenerates as the iterate
//! converges (45 of its 55 singular values fall to ~1e-8 by iteration 27, on
//! a `‖A‖ ≈ 240` KKT), so every factorization taken with `δ_c = 0` down that
//! tail is singular to working precision, with a smallest pivot at ~1e-16.
//! The negative-eigenvalue count read off such a factor is noise: the same
//! iterate returned 64, 58 and 62 against an expected 55, and a LAPACK
//! eigendecomposition of the dumped matrices agrees with none of those. The
//! `δ_w` ladder then escalated ×8 per retry against a reading that does not
//! respond to `δ_w` at all. The perturbation that *does* repair a
//! rank-deficient constraint block is `δ_c` — and `δ_c` is exactly what the
//! `Singular` verdict reaches for. The fix routes there:
//! `feral_inertia_pivot_floor` reports `Singular` instead of `WrongInertia`
//! when the count is contradicted by a working-precision pivot. Applying
//! `δ_c` lifts the smallest pivot from ~1e-16 to 5.8e-9, and from that point
//! the counts feral reports agree with LAPACK exactly.
//!
//! The trigger only ever fires on a factorization the caller was already
//! going to reject, so it cannot turn a usable factor into a failure; that
//! property is pinned as a unit test in `pounce-feral`
//! (`well_conditioned_inertia_mismatch_still_reports_wrong_inertia`).
//!
//! FIXTURE PROVENANCE. `fixtures/eigena2.nl` is CUTE `EIGENA2` (the
//! symmetric eigenvalue problem posed as an equality-constrained NLP; 110
//! variables, 55 equality constraints) from Vanderbei's CUTE-in-AMPL
//! collection, the same `.nl` the reporter ran, attached to gh #540 because
//! the benchmark archive itself is gitignored and not in the checkout. On
//! this build the pre-fix behaviour reproduces the issue's numbers exactly:
//! 29 iterations, dual infeasibility 3.3144912958031115e-07, constraint
//! violation 2.4123147923660326e-11, `Solved To Acceptable Level`.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::return_codes::ApplicationReturnStatus;

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
        "pounce_issue540_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

fn solve(extra: &[&str]) -> SolveReport {
    solve_with_env(extra, &[])
}

fn solve_with_env(extra: &[&str], env: &[(&str, &str)]) -> SolveReport {
    let json_path = tmp_path("eigena2.json");
    let sol_path = tmp_path("eigena2.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture("eigena2.nl"))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path);
    for o in extra {
        cmd.arg(o);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// The pre-#540 behaviour: every mismatching count is taken at face value and
/// answered with `δ_w`, however small the pivot it was read off.
const NO_TRIGGER: [&str; 1] = ["feral_inertia_pivot_floor=0"];

/// The headline: `eigena2` now earns a strict certificate rather than the
/// acceptable-level fallback.
#[test]
fn eigena2_converges_to_optimal() {
    let r = solve(&[]);
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "eigena2 did not reach a strict certificate (status {:?}, dual inf {:e})",
        r.solution.status,
        r.statistics.final_dual_inf,
    );
}

/// ...and it is the certificate that improved, not the tolerance that moved:
/// the dual residual has to actually clear `tol = 1e-8`, which is the two
/// orders of magnitude the issue is about. Ipopt reaches 9.3e-9 here.
#[test]
fn eigena2_dual_infeasibility_clears_the_strict_tolerance() {
    let dual = solve(&[]).statistics.final_dual_inf;
    assert!(
        dual < 1e-8,
        "dual infeasibility {dual:e} did not clear tol = 1e-8",
    );
}

/// The guard against a vacuous pass: the `#540` trigger must still be doing
/// measurable work on `eigena2`, or the two tests above are passing for a
/// reason they do not describe.
///
/// **This guard was re-derived for gh#693 and is weaker than the one it
/// replaces. Read this before trusting it.**
///
/// It used to assert that disabling the trigger *reproduced the reported
/// failure* — `SolvedToAcceptableLevel` with a stuck ~3.3e-7 dual — under
/// `POUNCE_DBG_NO_QUAD=1`. That is no longer true, and the reason is
/// gh#693: with the least-squares multiplier perturbation removed from the
/// initializer, `y0` changes, the trajectory changes, and the pre-#540
/// route converges on this model instead of stalling. `eigena2` is fixed by
/// two independent routes now, so it can no longer isolate either one.
///
/// The old guard existed precisely to fail in this situation rather than go
/// quiet, and it did — it is being replaced deliberately, not silenced.
/// What is lost is real: **no fixture in this repo now reproduces the gh#540
/// failure**, so the trigger's necessity is pinned by nothing. See gh#693.
///
/// What is still true, and is what this asserts, in two parts.
///
/// **Iteration count, at the default.** With the trigger on `eigena2`
/// converges in 27 iterations; with `feral_inertia_pivot_floor=0` it takes
/// 29. Deterministic across runs and independent of `feral_refine`.
///
/// **Dual residual, at the default.** 3.43e-10 with the trigger against
/// 5.45e-09 without — 16x, deterministic.
///
/// That sentence used to read "under `feral_refine=yes`", and said the
/// margin had been lost at the default when gh#710 turned backend
/// refinement off. **It had not been, because the default never moved on
/// this path** (gh#909): `feral_config_from_options` clears `feral_refine`
/// only under `hessian_approximation=limited-memory`, and this fixture runs
/// with an exact Hessian. So the old test's contrast arm — a second solve
/// passing `feral_refine=yes` — was setting the option to the value it
/// already had, and its two halves measured the same configuration. It
/// passed, and it was pinning one number twice.
///
/// The 2x2 it should have been, measured on `e5f5830b`:
///
/// ```text
///                          trigger ON        trigger OFF     ratio
///   default (refine yes)   3.4317e-10        5.4473e-09      16x
///   feral_refine=no        5.2021e-09        5.4096e-09      1.04x
/// ```
///
/// Iteration counts are 27 and 29 in both rows, so the trigger's two
/// iterations really are independent of refinement. The residual margin is
/// not: it exists at the default and is gone with refinement off. Both
/// rows are asserted below, because the second is the one that makes the
/// first say something — a 16x margin with no configuration that lacks it
/// is not evidence that refinement is what supplies it.
///
/// Both halves of that shift are worth reading. The certificate `eigena2`
/// earns at the default is still strict and still clears `tol = 1e-8`, but
/// by 2x rather than 30x; and the trigger's contribution to it is now the
/// two iterations, not the last order of magnitude.
///
/// **Two caveats a reader has to have, or this test reads stronger than it
/// is.** First, the margin is narrow to one configuration: it is gone under
/// `POUNCE_DBG_NO_QUAD=1`, where the two land at 5.21e-09 and 5.27e-09, and
/// gone again under `feral_refine=no` per the table above. Second, the
/// *ordering* is newly true. On 0.10.0 the trigger made this model's dual
/// residual slightly **worse** — 2.21e-09 with it on against 5.95e-10 with
/// it off, both clearing `tol`:
///
/// ```text
///                          trigger ON        trigger OFF
///   main (0.10.0)          2.212e-09         5.947e-10     <- trigger hurts
///   with gh#693            3.432e-10         5.447e-09     <- trigger helps
/// ```
///
/// So the trigger had already stopped being load-bearing on `eigena2` at the
/// default before gh#693 touched anything; what gh#693 removed was the last
/// configuration (`POUNCE_DBG_NO_QUAD=1`, trigger off) under which the
/// original failure still appeared. This assertion pins a real, reproducible
/// effect of the current build — not a stable invariant of the fix. The
/// property gh#540 actually established, that a count read off a
/// working-precision pivot routes to `delta_c` instead of the `delta_w`
/// ladder, is pinned where it belongs: the `pounce-feral` unit test named in
/// the module header.
#[test]
fn the_trigger_still_improves_the_certificate() {
    // The trigger is worth two iterations, at the default and with
    // backend refinement off alike.
    let on = solve(&[]).statistics;
    let off = solve(&NO_TRIGGER).statistics;
    assert!(
        off.iteration_count > on.iteration_count,
        "the inertia trigger no longer shortens eigena2 at the default \
         (with {} iters, without {} iters), so the tests above are no longer \
         pinning the fix they describe",
        on.iteration_count,
        off.iteration_count,
    );

    // Row 1 of the table in the module doc: at the default -- which is
    // `feral_refine=yes` on this exact-Hessian path, gh#909 -- the
    // certificate margin is 16x.
    let with_trigger = on.final_dual_inf;
    let without = off.final_dual_inf;
    assert!(
        without > with_trigger * 5.0,
        "the inertia trigger no longer improves eigena2's dual residual at \
         the default (with {with_trigger:e}, without {without:e}), so the \
         tests above are no longer pinning the fix they describe",
    );
    assert!(
        with_trigger < 1e-9,
        "expected the trigger to reach ~3.4e-10, got {with_trigger:e}",
    );

    // Row 2, and the reason row 1 says anything. With the backend loop
    // off the margin collapses to ~1.04x, so it is refinement that
    // supplies those digits and not the trigger alone. This arm is also
    // what keeps the assertion honest if the default ever moves: were
    // `feral_refine` to become `no` on this path, row 1 would degrade to
    // this row's numbers and fail loudly rather than quietly measure the
    // same thing twice, which is exactly what the pre-gh#909 version of
    // this test did.
    let unrefined_on = solve(&["feral_refine=no"]).statistics.final_dual_inf;
    let unrefined_off = solve(&["feral_refine=no", NO_TRIGGER[0]])
        .statistics
        .final_dual_inf;
    assert!(
        unrefined_off < unrefined_on * 2.0,
        "with feral_refine=no the trigger's residual margin is supposed to \
         be gone (with {unrefined_on:e}, without {unrefined_off:e}); if it \
         is back, the margin above is no longer attributable to backend \
         refinement and the module doc's 2x2 is stale",
    );
}
