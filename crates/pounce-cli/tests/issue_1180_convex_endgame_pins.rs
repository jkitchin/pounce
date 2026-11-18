//! gh #1180 — two Maros–Mészáros iteration regressions, one per defect.
//!
//! Both came in with `fa74ed21` (gh#984 second pass), which made the convex
//! IPM's `Optimal` mean "the returned point's KKT error, read above its own
//! rounding floor, is within `tol`". Before it, both models stopped fast on
//! points that did *not* meet that: QPILOTNO with `1.57e-4` of equality
//! violation, POWELL20 with a primal residual of `3e-5`. The verdict was
//! right to refuse them. What it exposed is two places where the solver's
//! own machinery disagrees with the verdict.
//!
//! * **QPILOTNO — postsolve breaks a certified point.** The reduced problem
//!   solves to `Optimal`; postsolve recovers an eliminated column at
//!   `-2.69e-14`, the box repair clamps it to `0`, and its coefficient of
//!   `5.85e6` in two equality rows turns that into `1.57e-7` of violation.
//!   The CLI demoted the point and re-solved on the NLP arm (42 discarded +
//!   283 iterations). The same IPM without presolve certifies the model in
//!   42. Fix: retry unreduced before giving the convex arm up (84 total).
//!
//! * **POWELL20 — a stop test unreachable in doubles.** Every row is active
//!   and the rows sum to `0 = Σ rhs`, so there is no strict interior and the
//!   multipliers are fixed only up to `t·1`; the IPM drifts to `ẑ ~ 1e11`. The
//!   loop's true-slack complementarity counted the whole product `ẑ·slack`
//!   once the slack cleared its quantum, so a slack at the primal residual's
//!   level (`1e-9`) read `~1e2` against `tol = 1e-8`. The loop ground `μ` to
//!   `1e-63` and stopped at 121 only when that product happened to read `0`.
//!   Fix: read each product above its own noise, as the verdict does, at half
//!   the verdict's width (`hsde::TRUE_COMP_FLOOR_FACTOR`); 121 -> 27.
//!
//! QPILOTNO is the model itself (`benchmarks/qp/generate_nl.py QPILOTNO`).
//! POWELL20 is 1.6 MB as `.nl`, ten times the largest fixture, so the pin is
//! the same family at `n = 2000` (construction on `POWELL_OPTIMUM`). At that size the old rule took 46 iterations and the new one
//! 28; at `n = 3000` the old rule happened to stop at 23, which is the nature
//! of the defect (an exit by luck), so `n` was chosen by measurement.
//!
//! Neither pin is a snapshot: QPILOTNO's optimum is the published one, and
//! POWELL's is exact (all rows active determine `x` up to a shift, and the
//! shift that minimizes `½‖x‖²` is the mean), computed in rationals. The
//! NLP arm is *not* an oracle for POWELL: its `bound_relax_factor` widening
//! buys `2e-5` of row violation and an objective `4e-6` below the optimum.

use std::path::PathBuf;
use std::process::Command;

use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_solve_report::SolveReport;

/// Published optimum, Maros & Mészáros DOC 97/6 (`4.72858690D+06`).
const QPILOTNO_OPTIMUM: f64 = 4.728_586_90e6;

/// `POWELL20` at `n = 2000`: minimize `½‖x‖²` subject to
/// `x[k+1] - x[k] >= (-1)^(k+1)·(2⌊k/2⌋ + 3/2)` for `k < n-1` and
/// `x[0] - x[n-1] >= n - 1/2`. The right-hand sides sum to zero, so every row
/// is active, `x[k] = x[0] + Σ_{j<k} rhs_j`, and the optimal shift is the
/// mean. Exact value of `½‖x*‖²` (computed with `fractions.Fraction`; the
/// same construction gives `52089582812.5` at `n = 10000`, the
/// Maros–Mészáros POWELL20).
const POWELL_OPTIMUM: f64 = 416_916_562.5;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(stem: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(format!("{stem}.nl"));
    p
}

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_gh1180_{}_{seq}_{tag}.{ext}",
        std::process::id()
    ));
    p
}

/// Solve `stem` with `opts`, returning the JSON report and stdout followed by
/// stderr (the failed-retry note is on stderr).
fn solve(stem: &str, opts: &[&str]) -> (SolveReport, String) {
    let json = tmp_path(stem, "json");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(stem))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json);
    for o in opts {
        cmd.arg(o);
    }
    let out = cmd.output().expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "no report for {stem} @ {opts:?} (exit {:?}, {e}); stderr:\n{}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr),
        )
    });
    let _ = std::fs::remove_file(&json);
    (
        serde_json::from_str(&text).expect("parse SolveReport JSON"),
        format!(
            "{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
}

fn assert_convex_optimum(r: &SolveReport, stem: &str, optimum: f64, rel_tol: f64, out: &str) {
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "{stem}: must solve; stdout=\n{out}"
    );
    assert_eq!(
        r.solution.engine, "cvx-qp",
        "{stem}: must be answered by the convex IPM, not rerouted; stdout=\n{out}"
    );
    let rel = ((r.solution.objective - optimum) / optimum).abs();
    assert!(
        rel < rel_tol,
        "{stem}: objective {} is {rel:.2e} from the optimum {optimum}",
        r.solution.objective
    );
}

/// QPILOTNO at the default route stays on the convex arm. The postsolved
/// point fails the verdict (that is the branch under test, so it is asserted
/// on), the unreduced retry certifies it, and the iteration count is both
/// solves'. Measured 42 + 42 = 84; the NLP reroute it replaces was 283 on
/// top of the 42 it threw away.
#[test]
fn qpilotno_postsolve_failure_is_retried_unreduced() {
    let stem = "convex_qp_qpilotno";
    let (r, out) = solve(stem, &[]);
    assert!(
        out.contains("re-solved without presolve"),
        "{stem}: the retry branch was not taken -- if postsolve no longer \
         breaks this point the fixture has stopped testing gh#1180's first \
         defect and needs replacing, not loosening; stdout=\n{out}"
    );
    assert_convex_optimum(&r, stem, QPILOTNO_OPTIMUM, 1e-7, &out);
    let it = r.statistics.iteration_count;
    assert!(
        it <= 120,
        "{stem}: {it} iterations (measured 84; the NLP reroute was 283 + 42)"
    );
}

/// The retry's third branch: taken, but not `Optimal`, so the postsolved point
/// stands. Reached deterministically through the budget: the reduced solve
/// spends 42 of `max_iter = 50`, the unreduced retry gets the remaining 8 and
/// ends `IterationLimit`. The retry spends the caller's budget, never a fresh
/// one (gh#1185 review: it used to get the full `max_iter` and report
/// `Optimal` at 84 iterations under a cap of 50), and its failure is said and
/// counted rather than dropped.
#[test]
fn qpilotno_failed_retry_keeps_the_postsolved_point_within_max_iter() {
    let stem = "convex_qp_qpilotno";
    for cap in [45usize, 50] {
        let arg = format!("max_iter={cap}");
        let (r, out) = solve(stem, &[arg.as_str()]);
        assert!(
            out.contains("re-solving without presolve ended IterationLimit"),
            "{stem} @ {arg}: the failed-retry branch was not reached or not \
             reported; output=\n{out}"
        );
        assert!(
            !out.contains("re-solved without presolve"),
            "{stem} @ {arg}: a retry inside the leftover budget claimed success; \
             output=\n{out}"
        );
        assert_ne!(
            r.solution.status,
            ApplicationReturnStatus::SolveSucceeded,
            "{stem} @ {arg}: the postsolved point fails the KKT check and must \
             not be reported as a clean success; output=\n{out}"
        );
        let it = r.statistics.iteration_count as usize;
        assert_eq!(
            it,
            cap,
            "{stem} @ {arg}: both solves' iterations count, and together they \
             spend exactly the cap (measured 42 + {})",
            cap - 42
        );
    }
}

/// The other branch of the same gate: without presolve there is no postsolve
/// to fail and the retry must not fire. 42 iterations, one solve.
#[test]
fn qpilotno_without_presolve_takes_no_retry() {
    let stem = "convex_qp_qpilotno";
    let (r, out) = solve(stem, &["qp_presolve=no"]);
    assert!(
        !out.contains("re-solved without presolve"),
        "{stem}: retry fired with presolve off; stdout=\n{out}"
    );
    assert_convex_optimum(&r, stem, QPILOTNO_OPTIMUM, 1e-7, &out);
    let it = r.statistics.iteration_count;
    assert!(it <= 60, "{stem}: {it} iterations (measured 42)");
}

/// POWELL20's family stops once the iterate is converged rather than when the
/// complementarity product happens to read zero. Measured 28 (the old rule:
/// 46 here, 121 on POWELL20 itself).
#[test]
fn powell_stops_when_converged_not_by_luck() {
    let stem = "convex_qp_powell20_n2000";
    let (r, out) = solve(stem, &[]);
    assert_convex_optimum(&r, stem, POWELL_OPTIMUM, 1e-9, &out);
    let it = r.statistics.iteration_count;
    assert!(
        it <= 36,
        "{stem}: {it} iterations (measured 28; the binary true-slack gate \
         took 46)"
    );
}
