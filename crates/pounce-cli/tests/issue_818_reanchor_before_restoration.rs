//! gh #818 — a line-search failure at an *already feasible* point is not a
//! restoration problem.
//!
//! When the backtracking line search cannot accept any trial step, either
//! the **point** is bad — infeasible, and the restoration phase is exactly
//! the right tool — or the **direction** is, because `W` is a quasi-Newton
//! model carrying curvature the iterate has left behind. Upstream has one
//! answer for both, because restoration is the only fallback it has.
//!
//! At a feasible point that answer is a no-op. The restoration NLP minimizes
//! the constraint violation and there is none to minimize, so it wanders at
//! `theta ~ 1e-13` and reports `Restoration_Failed`. Measured on `deb7` under
//! `limited-memory`: the solve stalls with `inf_pr ~ 1e-12` and `inf_du ~ 1e5`,
//! enters restoration at a point feasible to 8e-13, and spends 340 of its 1242
//! iterations there. On an unconstrained model `theta` is identically zero, so
//! restoration cannot move at all.
//!
//! `IpoptAlgorithm::try_reanchor_before_restoration` puts one rung in front of
//! the hand-off: drop every curvature pair but the newest and retry. It is not
//! a feasibility gate on restoration — that was tried and rejected before (see
//! the `constr_viol_tol` paragraph in `IpoptAlgorithm::invoke_restoration`) —
//! and it runs *after* the acceptable-point decline, so a reportable point is
//! still reported.
//!
//! **The rung ships off.** `limited_memory_ls_failure_restarts` defaults to
//! `0` — upstream's unconditional hand-off. That default was set on a
//! measurement taken at `ALPHA_INTERP_MIN_TRIALS = 5`, where the rung cost
//! iterations on `eigena2` and `infeasible_square_scaled_1em4` to the same
//! verdict and bought only `deb7` and `issue_508_infeasible_gap_1em4`.
//!
//! **That ledger has changed since**, twice (the gate of 6, then gh#1146's
//! restoration fix), and the table in `the_rung_ledger_that_decided_the_default`
//! carries the current numbers: `eigena2` is now a large gain to the same
//! verdict, and the cost has moved to `issue_508_infeasible_gap_1em4`. It
//! stays off regardless, because turning it on is a trajectory change over
//! the whole corpus that needs its own `scripts/sweep-fixtures.sh` run. The
//! case for switching it on is open, not settled.
//!
//! Every test here therefore sets the option explicitly on both sides, and
//! both directions are pinned deliberately: the gain is why the code stays
//! in the tree and stays documented, and the cost is why it is not the
//! default. A change that moves either is a change to that decision.
//!
//! **What this file is not evidence about.** It pins that the rung is wired,
//! fires, is switchable, and which way it moves four named models. It does not
//! pin the corpus — that is `scripts/sweep-fixtures.sh` — and it says nothing
//! about the exact-Hessian arm, which has no curvature history to re-anchor.
//! Note also that setting the option at all, `0` included, opts the solve out
//! of the `Solved_To_Acceptable_Level` re-solve ladder
//! (`TERMINATION_POLICY_OPTIONS`), so every cell below runs in a regime the
//! shipped default never enters.

use pounce_solve_report::SolveReport;
use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture_named(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// Solve `fixture` on the limited-memory arm with `restarts` re-anchors
/// allowed, returning its JSON report. The CLI's exit code is not asserted:
/// `eigena2` fails on this arm either way, and the point of the test is the
/// iteration count, not the status.
fn solve(fixture: &str, tag: &str, restarts: u32) -> SolveReport {
    let json = std::env::temp_dir().join(format!("pounce_issue_818_{tag}.json"));
    let _ = std::fs::remove_file(&json);
    let out = Command::new(pounce_exe())
        .arg(fixture_named(fixture))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .arg("hessian_approximation=limited-memory")
        .arg(format!("limited_memory_ls_failure_restarts={restarts}"))
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "JSON report should be written for {fixture} ({tag}): {e}\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    let _ = std::fs::remove_file(&json);
    serde_json::from_str(&text).expect("deserialize report")
}

fn iters(r: &SolveReport) -> i32 {
    r.statistics.iteration_count
}

/// Run `fixture` and return `(how many times the rung fired, the status line)`.
///
/// The re-anchor is not in the JSON report — it is one `tracing` line at debug
/// level, which is the only place the *decision* is observable rather than its
/// downstream effect on a trajectory. `issue_438_resto_layer2_verdict.rs` reads
/// the restoration layer's verdict the same way.
fn solve_counting_reanchors(fixture: &str, tag: &str, restarts: u32) -> (usize, String) {
    let out = Command::new(pounce_exe())
        .arg(fixture_named(fixture))
        .arg("--no-sol")
        .arg("hessian_approximation=limited-memory")
        .arg(format!("limited_memory_ls_failure_restarts={restarts}"))
        .env("RUST_LOG", "pounce::algorithm=debug")
        .env("NO_COLOR", "1")
        .output()
        .unwrap_or_else(|e| panic!("spawn pounce for {fixture} ({tag}): {e}"));
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let fires = combined
        .lines()
        .filter(|l| l.contains("re-anchoring the limited-memory Hessian"))
        .count();
    (fires, combined)
}

/// `issue_508_infeasible_gap_1em4`: the rung fires on a genuinely infeasible
/// model and the solve reaches the *same* verdict.
///
/// The verdict assertion is the load-bearing half. A rung that reached the
/// answer faster by giving up on a genuine restoration would show here as a
/// changed status, which is precisely the failure mode the ordering against
/// the acceptable-point decline is meant to prevent.
///
/// It used to assert a speed-up as well (79 -> 76). Since gh#1146 confined
/// restoration's limited-memory update to the `x` block, as upstream does,
/// the rung-off run is 73 and the rung-on run 74: the rung no longer pays
/// for itself here, and that is now the *cost* side of the ledger below.
#[test]
fn reanchor_does_not_change_the_infeasibility_certificate() {
    let off = solve("issue_508_infeasible_gap_1em4.nl", "gap_off", 0);
    let on = solve("issue_508_infeasible_gap_1em4.nl", "gap_on", 1);

    assert_eq!(
        off.solution.status, on.solution.status,
        "the rung must not change the verdict, only how long it takes"
    );
    assert!(
        matches!(
            off.solution.status,
            pounce_nlp::return_codes::ApplicationReturnStatus::InfeasibleProblemDetected
        ),
        "expected an infeasibility certificate, got {:?}",
        off.solution.status
    );
}

/// `deb7` is the stall the rung was designed from, and the reason the code
/// stays in the tree. What is pinned here is that the rung **fires** on it,
/// and that firing does not manufacture a success.
///
/// **Neither the iteration count nor the exact number of firings is
/// asserted, because neither is a property of the rung.** An earlier
/// revision asserted that the rung shortens `deb7` (measured 715 without /
/// 610 with); a later one asserted that a budget of 1 is spent exactly once.
/// Both were measurements of a particular gate. At
/// `ALPHA_INTERP_MIN_TRIALS = 6` the rung fires **twice** on `deb7` with
/// `limited_memory_ls_failure_restarts=1`, and that is correct: the budget
/// is per `IpoptAlgorithm`, and a solve that enters restoration builds a
/// second one for the restoration NLP (`resto_inner_solver.rs`), which
/// carries its own counter. A solve gets one re-anchor per algorithm
/// instance it constructs, not one per solve.
///
/// What survives is the direction: the rung either happens or it does not,
/// and that is a property of the code rather than of the arithmetic it runs
/// on or of how many nested solves the trajectory happens to enter.
///
/// The second assertion is the safety half. `deb7` does not solve on this
/// arm under any setting measured, so a `Solve_Succeeded` appearing here
/// would be the rung turning a stall into a spurious success — a wrong
/// answer, and the one outcome that would matter more than any iteration
/// count.
///
/// **It reads the reported verdict, not the log.** An earlier revision
/// asked whether the combined stdout+stderr *contained* the substring
/// `Solve_Succeeded`, which is a proxy, and the proxy broke: the blob is
/// captured at `RUST_LOG=pounce::algorithm=debug` so that the re-anchor
/// decision is observable at all, and the debug narration names status
/// codes. On this arm with the rung on, `deb7` exits `Restoration_Failed`,
/// which is a base verdict the gh#884 dual-divergence retry acts on, so a
/// second `dispatch_standard_solve` runs — and *its* mu-strategy fallback
/// logs `MaximumIterationsExceeded is not Solve_Succeeded`. No success was
/// reported and none was manufactured; a string that names the status
/// matched. The `Status:` line is the verdict, so ask it.
#[test]
fn the_rung_fires_on_the_model_it_was_designed_from() {
    let (off_fires, off_status) = solve_counting_reanchors("deb7.nl", "deb7_off", 0);
    let (on_fires, on_status) = solve_counting_reanchors("deb7.nl", "deb7_on", 1);

    assert_eq!(
        off_fires, 0,
        "the default budget is 0, so nothing may re-anchor; saw {off_fires}"
    );
    assert!(
        on_fires >= 1,
        "a budget of 1 must be spent at least once on deb7 — the model the \
         rung was designed from; saw {on_fires}"
    );

    for (label, out) in [("off", &off_status), ("on", &on_status)] {
        // The last `Status:` line is the verdict finally reported: a solve
        // that retries prints one summary per attempt, and the wrapper
        // restores the base attempt's status after the last of them.
        let verdict = out
            .lines()
            .rev()
            .find_map(|l| l.strip_prefix("Status: "))
            .unwrap_or_else(|| {
                panic!("no `Status:` line in the rung-{label} run's output:\n{out}")
            });
        assert_ne!(
            verdict, "Solve_Succeeded",
            "deb7 does not solve on the limited-memory arm; a success with the \
             rung {label} means the rung manufactured one"
        );
    }
}

/// The other half of the measurement, and the one that decided the default.
///
/// **Re-measured for gh#1146**, which confined restoration's limited-memory
/// update to the `x` block as upstream does. That moved every restoration
/// these fixtures enter. Rung off against rung on, `limited-memory`:
///
/// | fixture | rung off | rung on |
/// |---|---|---|
/// | `eigena2` | `ErrorInStepComputation`/263 | `ErrorInStepComputation`/**163** |
/// | `issue_508_infeasible_gap_1em4` | `InfeasibleProblemDetected`/73 | `InfeasibleProblemDetected`/74 |
/// | `infeasible_square_scaled_1em4` | `InfeasibleProblemDetected`/26 | `InfeasibleProblemDetected`/26 |
/// | `deb7` | `MaximumIterationsExceeded`/3000 | `MaximumIterationsExceeded`/3000 |
///
/// Before gh#1146 (same binary otherwise, `358ecbb`) the rows read 239/161,
/// 79/76, 24/26 and 3000/3000. Earlier still, at `ALPHA_INTERP_MIN_TRIALS = 5`,
/// the rung cost iterations on `eigena2` and `infeasible_square_scaled_1em4`,
/// and that pair is what kept it off. The witnesses have moved since: the
/// cost now shows on `issue_508_infeasible_gap_1em4` (one iteration, same
/// verdict) and `infeasible_square_scaled_1em4` is neutral; the gain is still
/// `eigena2`, now 263 -> 163 to the same verdict.
///
/// The rung stays off for the reason it always has: switching it on is a
/// trajectory change over the whole corpus and needs its own
/// `scripts/sweep-fixtures.sh` run to justify. What is pinned below is only
/// that both directions are still real, so a change to either is visible to
/// the next reader.
#[test]
fn the_rung_ledger_that_decided_the_default() {
    // The cost: same verdict, more iterations.
    let cost_off = solve("issue_508_infeasible_gap_1em4.nl", "gap_cost_off", 0);
    let cost_on = solve("issue_508_infeasible_gap_1em4.nl", "gap_cost_on", 1);
    assert_eq!(
        cost_off.solution.status, cost_on.solution.status,
        "issue_508_infeasible_gap_1em4: the rung must not change the verdict"
    );
    assert!(
        iters(&cost_on) > iters(&cost_off),
        "issue_508_infeasible_gap_1em4: the rung was measured as a cost here \
         ({} on against {} off); a reversal means the ledger above needs \
         re-measuring",
        iters(&cost_on),
        iters(&cost_off)
    );

    // The gain: fewer iterations to the same verdict.
    let gain_off = solve("eigena2.nl", "eigena2_gain_off", 0);
    let gain_on = solve("eigena2.nl", "eigena2_gain_on", 1);
    assert_eq!(
        gain_off.solution.status, gain_on.solution.status,
        "eigena2: the rung must not change the verdict"
    );
    assert!(
        iters(&gain_on) < iters(&gain_off),
        "eigena2: the rung was measured as a gain ({} on against {} off); a \
         reversal means the ledger above needs re-measuring",
        iters(&gain_on),
        iters(&gain_off)
    );
}

/// `0` is the default and is upstream's unconditional hand-off; `1` has to
/// actually reach the algorithm. Without this the option could be registered
/// and ignored — the gh #677 failure mode, where
/// `limited_memory_initialization` was parsed, validated, and read nowhere.
#[test]
fn the_option_reaches_the_algorithm() {
    let off = solve("eigena2.nl", "eigena2_zero", 0);
    let on = solve("eigena2.nl", "eigena2_one", 1);
    assert_ne!(
        iters(&off),
        iters(&on),
        "limited_memory_ls_failure_restarts=1 took the same {} iterations as \
         the default 0 — the option is parsed but not reaching the algorithm",
        iters(&off)
    );
}
