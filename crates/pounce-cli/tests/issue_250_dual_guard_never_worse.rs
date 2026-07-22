//! The dual-divergence guard (`dual_diverging_streak`, pounce#246) is **opt-in**,
//! and when opted into its diversion must never leave the solve worse off than a
//! point it already had in hand (pounce#250 follow-up).
//!
//! WHY IT IS OPT-IN. The guard routes a solve into restoration once the dual
//! infeasibility has grown for N consecutive iterations in an elevated regime. It
//! shipped default-on at N=15 to bound a reported emfl050 bad-warm-start grind,
//! but that justification did not survive being reproduced: the reported
//! measurement was caller-side JAX compilation, and the build predating the guard
//! solves both emfl050 instances to the same optimum in the same time. What
//! remained was an effect on four of 1284 MINLPLib models that is knife-edge and
//! non-monotone in the threshold:
//!
//!   * `deb7`/`deb9` reach a better local optimum (104.95 -> 97.56) at *exactly*
//!     15, and at no other value tried (0, 5, 25, 32, 40, 60) — on macOS/FERAL.
//!     On the Linux CI runner the same setting makes `deb7` *worse*: 97.56 with
//!     the guard off, 127.87 with it on. The effect differs by host in **sign**.
//!   * `pooling_rt2stp` turns from `Solve_Succeeded` into
//!     `Maximum_Iterations_Exceeded` at 10 and 15 only — solving cleanly at 0, 5,
//!     25 and 40.
//!
//! That is basin luck on nonconvex problems, not a property, and the two sides
//! are not commensurate: the upside is a better local optimum on an
//! already-solved problem, the downside is a clean solve becoming a failure. So
//! the guard stays available but is no longer imposed. `pooling_rt2stp_solves_at_default_settings`
//! pins that decision.
//!
//! THE NEVER-WORSE-OFF PROPERTY, for anyone who does opt in. On
//! `autocorr_bern55-06` at N=15 the guard fires at iteration 23; the diverted run
//! reaches the true optimum (-2304.0000278, which Ipopt also finds) and holds it
//! from iteration 57 to 86, but the dual residual sawtooths between 1e-8 and 2e-1
//! there, so it never strings together the `acceptable_iter` consecutive
//! qualifying iterates that would stop the solve. It then entered restoration a
//! second time, wandered into a worse basin, and stopped at -2263.46 — 1.8 %
//! worse, with an overall NLP error of **1.0**, reported as "solved to acceptable
//! level". The better point had already passed the acceptable test; it was simply
//! overwritten, because `store_acceptable_point` keeps the latest rather than the
//! best. POUNCE now records the best acceptable iterate and hands it back if the
//! diverted run ends worse.
//!
//! That fallback bounds the *objective* downside but cannot manufacture a point
//! the solve never reached — which is exactly why `pooling_rt2stp` still needed
//! the default change rather than more machinery.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;

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
        "pounce_issue250_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

fn solve(fixture_name: &str, extra: &[&str]) -> SolveReport {
    let json_path = tmp_path(&format!("{fixture_name}.json"));
    let sol_path = tmp_path(&format!("{fixture_name}.sol"));
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(fixture_name))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path);
    for o in extra {
        cmd.arg(o);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// Ipopt's answer on the identical `.nl`: -2304.0000278027342, reached in 72
/// iterations with an overall NLP error of 3.7e-9.
const AUTOCORR_OPTIMUM: f64 = -2_304.000_027_802_734_2;

/// The guard is opt-in as of the pounce#250 follow-up (`dual_diverging_streak`
/// defaults to 0). Every test below must therefore enable it explicitly — at the
/// former default of 15 — or it would exercise nothing and pass vacuously.
const GUARD_ON: [&str; 2] = ["max_wall_time=10", "dual_diverging_streak=15"];

/// With the guard enabled (streak 15 — its former default; it is now off by
/// default) the diversion must not cost the objective. Pre-fix this returned
/// -2263.4612448099933.
#[test]
fn dual_guard_diversion_does_not_return_a_worse_point() {
    let report = solve("autocorr_bern55-06.nl", &GUARD_ON);
    let obj = report.solution.objective;
    let rel = (obj - AUTOCORR_OPTIMUM).abs() / AUTOCORR_OPTIMUM.abs();
    assert!(
        rel < 1e-6,
        "dual-divergence guard diverted the solve into a worse basin: got {obj}, \
         expected the optimum {AUTOCORR_OPTIMUM} that the same solve already \
         visited (rel err {rel:.3e}; pounce#250 follow-up)",
    );
}

/// The returned point must also be a stationary one. The pre-fix answer was
/// feasible (constraint violation 3.3e-12) but carried an overall NLP error of
/// 1.0, so asserting on the objective alone would not have caught a future
/// regression that returned a different feasible-but-not-KKT point.
#[test]
fn dual_guard_diversion_returns_a_stationary_point() {
    let report = solve("autocorr_bern55-06.nl", &GUARD_ON);
    let err = report.statistics.final_kkt_error;
    assert!(
        err < 1e-4,
        "returned point is not stationary: overall KKT error {err} (pre-fix this \
         was 1.0 — feasible but nowhere near a KKT point; pounce#250 follow-up)",
    );
}

// `deb7` carries NO absolute assertion, deliberately, and the reason is the
// single most decisive measurement in this investigation.
//
// The guard's effect on this model differs by host **in sign**:
//
// | host              | guard off | guard on (streak 15) |
// |-------------------|-----------|----------------------|
// | macOS / FERAL     |    104.95 |            **97.56** |
// | Linux CI runner   |     97.56 |           **127.87** |
//
// It helps on one platform and hurts on the other, from the same source, on the
// same fixture. Two earlier versions of this test tried to pin a value here —
// first "the guard rescues deb7 to 97.56", then the weaker "enabling the guard
// still reaches ~97.56" — and CI falsified both. There is no cross-platform
// claim to make.
//
// That is the finding, not an obstacle to testing it. A heuristic whose sign
// depends on the host is not a property of the algorithm, and it is why
// `dual_diverging_streak` is off by default. It also bounds what the
// best-acceptable fallback can promise: on Linux the guard costs deb7 30 % of
// its objective and the fallback cannot recover it, because 127.87 is the best
// acceptable point the diverted run ever reached — see
// `honour_best_acceptable_after_dual_guard`.
//
// deb7 is still exercised below, under maximum firing pressure — but only for
// invariants that hold on every host (a valid, honestly-labelled point), not the
// cross-host objective comparison an earlier version tried and gh #267 retired.
// The fallback's actual guarantee is a property of the ranking, proven
// host-independently by the `ranks_better_*` unit tests in `ipopt_alg.rs`.

/// The guard must stay off unless asked for. This is the model that decided it:
/// `pooling_rt2stp` solves cleanly with the guard disabled and at streaks 5, 25
/// and 40, but at 10 and 15 the diversion turns `Solve_Succeeded` into
/// `Maximum_Iterations_Exceeded` with an unscaled NLP error of ~18 and a 0.26 %
/// worse objective.
///
/// The best-acceptable fallback cannot rescue this one — the diverted solve never
/// reaches an acceptable point at all, so there is nothing to hand back. That is
/// what makes it a defaults question rather than more machinery.
///
/// Fails if the default is flipped back on.
#[test]
fn pooling_rt2stp_solves_at_default_settings() {
    // Ipopt on the identical .nl: -3273.9549927585640, NLP error ~1e-8.
    const EXPECTED: f64 = -3_273.954_991_374_754_6;
    let report = solve("pooling_rt2stp.nl", &["max_wall_time=10"]);
    let code = report.solution.solve_result_num;
    assert!(
        (0..100).contains(&code),
        "pooling_rt2stp did not converge at default settings \
         (solve_result_num={code}, status={:?}) — has dual_diverging_streak been \
         re-enabled by default? At 15 this model returns \
         Maximum_Iterations_Exceeded (pounce#250 follow-up)",
        report.solution.status,
    );
    let obj = report.solution.objective;
    let rel = (obj - EXPECTED).abs() / EXPECTED.abs();
    assert!(
        rel < 1e-6,
        "pooling_rt2stp: got {obj}, expected {EXPECTED} (rel err {rel:.3e})",
    );
}

/// Under **maximum firing pressure** the guard must still return a valid,
/// honestly-labelled point — but this test does *not* assert it beats the
/// counterfactual of never diverting, which the mechanism cannot promise
/// (gh #267).
///
/// `dual_diverging_streak=1` is a hair trigger: the guard diverts on the first
/// growing dual-infeasibility step in the elevated regime, so it fires early and
/// often on models that would otherwise converge untouched. That makes it the
/// sharpest probe of the record/read machinery — it moves the diversion as early
/// as it can go, exercising the recorder on points at and before the diversion.
///
/// WHAT THIS ASSERTS, AND WHAT IT DELIBERATELY DOES NOT. An earlier version
/// asserted `guard_on_obj <= guard_off_obj + slack` — that a diverted run is no
/// worse than *not diverting*. gh #267 caught that as unsound: the fallback
/// guarantees only that a diverted run is no worse than the best acceptable
/// point *that same run visited*, never that diverting beats the counterfactual
/// solve that never happened. That comparison passed only by a slack ~28,000x
/// the real gap, and on a host whose streak=1 basin flips — as `deb7`'s does at
/// streak 15, 97.56 off vs 127.87 on — it would fail on basin luck, not on any
/// defect. It was host-independent in form, not in outcome.
///
/// The mechanism's real guarantee is a property of the ranking, and it is proven
/// host-independently, by cases, in the `ranks_better_*` unit tests in
/// `ipopt_alg.rs`. Here we assert only what is sound under heavy firing on every
/// host: each arm terminates with a finite objective, and — the gh #267
/// invariant — never reports a success/acceptable status while carrying a
/// grossly-infeasible point. Feasibility is not silently traded for a
/// better-looking objective.
#[test]
fn hair_trigger_guard_returns_a_valid_honestly_labelled_point() {
    for name in ["autocorr_bern55-06.nl", "deb7.nl"] {
        for streak in ["dual_diverging_streak=0", "dual_diverging_streak=1"] {
            let report = solve(name, &["max_wall_time=20", streak]);
            let obj = report.solution.objective;
            assert!(
                obj.is_finite(),
                "{name} ({streak}): non-finite objective {obj} under hair-trigger \
                 firing — the diversion produced garbage (gh #267 / pounce#250)",
            );
            // The gh #267 invariant: a success/acceptable status (AMPL
            // solve_result_num 0..199) must describe a feasible point. At default
            // tolerances the acceptable gate already bounds this; asserting it
            // here guards the fallback + status mapping against a regression that
            // restores an infeasible point under a success status, and it holds
            // on every host (unlike the objective comparison it replaced). The
            // pre-fix gh #267 failure sat at violation 9.94 under this very
            // status; the bound is far below that and far above what a genuine
            // acceptable point reaches.
            let code = report.solution.solve_result_num;
            if (0..200).contains(&code) {
                let viol = report.statistics.final_constr_viol;
                assert!(
                    viol < 1e-2,
                    "{name} ({streak}): reported a success/acceptable status \
                     (solve_result_num={code}) at constraint violation {viol} — \
                     the fallback handed back an infeasible point under a success \
                     status (gh #267)",
                );
            }
        }
    }
}

/// The best-acceptable fallback must not spend feasibility to buy objective
/// (gh #267 — a pounce#259 / PR #259 follow-up).
///
/// The fallback originally ranked recorded acceptable points by scaled
/// objective alone. Being *bounded* by `acceptable_constr_viol_tol` is not the
/// same as *not trading* feasibility within that band, and the band is a user
/// option: widen it and a pure-objective argmax will discard a nearly-feasible
/// point for a lower-objective one that is grossly infeasible, then hand it back
/// under a `Solved_To_Acceptable_Level` status.
///
/// The control is the point of this test. At the same widened tolerances the
/// guard-*off* solve returns a feasible point, so the loose band alone does not
/// cause the infeasible return — it only widens what the fallback can exploit.
/// The fix ranks by `(feasibility, objective)`, so the guard-on solve must also
/// return a feasible point.
///
/// Pre-fix numbers (dev host): guard on returned objective -2307.32 at a
/// constraint violation of **9.94** — below the true optimum -2304.0 precisely
/// because the point is infeasible — while guard off returned -2298.57 at
/// violation 1.06e-4. Both reported `solve_result_num` 100.
#[test]
fn best_acceptable_fallback_does_not_trade_feasibility_for_objective() {
    // A widened acceptable band. `acceptable_constr_viol_tol` alone is not
    // enough — the other acceptable criteria gate first — so the whole triplet
    // is loosened, which is unusual but entirely legal on hard or badly-scaled
    // models. `constr_viol_tol` (the strict feasibility band the fix keys on) is
    // left at its 1e-4 default.
    const BAND: [&str; 5] = [
        "acceptable_constr_viol_tol=1e1",
        "acceptable_tol=1e10",
        "acceptable_dual_inf_tol=1e30",
        "acceptable_compl_inf_tol=1e10",
        "max_wall_time=20",
    ];

    let off = solve(
        "autocorr_bern55-06.nl",
        &[BAND.as_slice(), &["dual_diverging_streak=0"]].concat(),
    );
    let on = solve(
        "autocorr_bern55-06.nl",
        &[BAND.as_slice(), &["dual_diverging_streak=15"]].concat(),
    );

    // The control: the loose band on its own must not produce an infeasible
    // return, or this fixture no longer isolates the fallback's behaviour.
    assert!(
        off.statistics.final_constr_viol < 1e-2,
        "control invalid: guard-off solve is itself infeasible at these \
         tolerances (constr_viol {}) — the fixture no longer isolates the \
         fallback (gh #267)",
        off.statistics.final_constr_viol,
    );

    // The fix: the guard-on solve must hand back a feasible point, not the
    // grossly-infeasible lower-objective one a pure-objective ranking chose.
    // Pre-fix this was 9.94; the threshold sits far below that and far above the
    // ~1e-4 the recovered point actually reaches.
    assert!(
        on.statistics.final_constr_viol < 1e-2,
        "best-acceptable fallback traded feasibility for objective: guard-on \
         solve returned objective {} at constraint violation {} (pre-fix 9.94), \
         while the guard-off control at the same tolerances is feasible at {}. \
         The fallback ranked by objective alone and handed back an infeasible \
         point under a success status (gh #267).",
        on.solution.objective,
        on.statistics.final_constr_viol,
        off.statistics.final_constr_viol,
    );

    // The fix must not over-correct into feasibility-first ranking, which would
    // hand back a trivially-feasible but useless point — the starting iterate
    // sits at objective 0 and violation 0 and would win a feasibility-primary
    // key. Among feasible-enough points objective still decides, so the returned
    // point is the diverted run's own near-optimal endpoint (~-2304), the point
    // gh #267 says should have been kept, not the starting point.
    assert!(
        on.solution.objective < -2000.0,
        "best-acceptable fallback over-corrected: guard-on solve returned \
         objective {} — a feasibility-first ranking handed back a trivially \
         feasible but far-from-optimal point instead of the near-optimal \
         endpoint the diverted run reached (gh #267).",
        on.solution.objective,
    );
}
