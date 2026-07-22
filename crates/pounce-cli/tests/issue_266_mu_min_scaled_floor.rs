//! Issue #266 regression: the μ floor's *absolute* term must not block the
//! termination certificate under strong objective scaling.
//!
//! PR #258 (issue #257) converted `compl_inf_tol` — enforced on the
//! **unscaled** complementarity — into μ's scaled space before it enters the
//! dynamic barrier floor. But the floor has a second, independent term:
//! `mu_min`, a raw absolute constant (default `1e-11`) living in scaled
//! space. Once `compl_inf_tol · df / (barrier_tol_factor + 1) < mu_min`, the
//! converted dynamic term stops mattering and μ bottoms out at `mu_min`,
//! leaving an unscaled complementarity of `≈ mu_min / df`. The certificate
//! needs `mu_min / df ≤ compl_inf_tol`, so it becomes unreachable below
//!
//! ```text
//!     df* = mu_min / compl_inf_tol = 1e-11 / 1e-4 = 1e-7
//! ```
//!
//! `hs71_obj1e8.nl` is HS71 with the objective multiplied by `1e8` (the
//! issue's `c1e8.nl`): gradient-based scaling computes `df ≈ 8.3e-8`, just
//! under the cliff. The iterate sits *at* the optimum — same x* as unscaled
//! HS71 to ~1e-9 — yet pre-fix POUNCE exits
//! `Search_Direction_Becomes_Too_Small` (code 400, outside AMPL's 0..99
//! solved band, which discopt's status map reads as UNBOUNDED). Upstream
//! Ipopt certifies the same file Optimal.
//!
//! Fixture provenance: written by Pyomo 6.10 from
//!
//! ```python
//! m.x1..m.x4 = Var(bounds=(1, 5)); x0 = (1, 5, 5, 1)
//! m.obj = Objective(expr=1e8 * (x1*x4*(x1+x2+x3) + x3))
//! m.g1 = Constraint(expr=x1*x2*x3*x4 >= 25)
//! m.g2 = Constraint(expr=x1**2 + x2**2 + x3**2 + x4**2 == 40)
//! ```

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
        "pounce_issue266_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

fn solve(fixture_name: &str, extra_opts: &[&str]) -> SolveReport {
    let json_path = tmp_path(&format!("{fixture_name}.json"));
    let sol_path = tmp_path(&format!("{fixture_name}.sol"));
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(fixture_name))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path);
    for opt in extra_opts {
        cmd.arg(opt);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// HS71's optimum. The fixture's objective is the model's × `OBJ_MULTIPLIER`,
/// and x* does not depend on the multiplier.
const HS71_OPTIMUM: f64 = 17.014_017_140_2;
const OBJ_MULTIPLIER: f64 = 1e8;

/// `solve_result_num` 0..100 is AMPL's "solved" band; anything else is what a
/// B&B driver turns into a spurious UNBOUNDED. Assert on the band, not one
/// code.
fn assert_solved_at_optimum(report: &SolveReport, ctx: &str) {
    let code = report.solution.solve_result_num;
    assert!(
        (0..100).contains(&code),
        "{ctx}: did not converge (solve_result_num={code}, status={:?}); \
         this problem has a finite optimum ~{HS71_OPTIMUM}·{OBJ_MULTIPLIER:e} \
         that Ipopt certifies (issue #266)",
        report.solution.status,
    );
    let obj = report.solution.objective / OBJ_MULTIPLIER;
    assert!(
        (obj - HS71_OPTIMUM).abs() / HS71_OPTIMUM < 1e-6,
        "{ctx}: objective/{OBJ_MULTIPLIER:e} = {obj} is not the known optimum \
         {HS71_OPTIMUM}",
    );
}

/// Confirm the fixture still lands in the regime that triggers #266: the
/// computed objective scaling factor must sit below the `1e-7` cliff.
fn assert_df_below_cliff(report: &SolveReport, ctx: &str) -> f64 {
    let stats = &report.statistics;
    let obj_scale = stats.final_scaled_objective / stats.final_objective;
    assert!(
        obj_scale < 1e-7,
        "{ctx}: expected df < 1e-7 (the #266 cliff; HS71×1e8 computes \
         df≈8.3e-8), got {obj_scale} — the fixture no longer exercises the \
         bug",
    );
    obj_scale
}

/// The issue's headline row: HS71 × 1e8 at stock options must certify. Pre-fix
/// this exits `Search_Direction_Becomes_Too_Small` (400) with the iterate
/// sitting on the optimum, because μ cannot descend below the unconverted
/// `mu_min = 1e-11` while the certificate needs `μ ≤ compl_inf_tol·df ≈
/// 8.3e-12`.
#[test]
fn hs71_obj1e8_certifies_at_default_options() {
    let report = solve("hs71_obj1e8.nl", &[]);
    assert_df_below_cliff(&report, "hs71×1e8 (defaults)");
    assert_solved_at_optimum(&report, "hs71×1e8 (defaults)");
}

/// #266 is tolerance-independent (unlike #257, whose tell was a tol
/// inversion): pre-fix, tol=1e-6 and tol=1e-10 both fail, because the floor is
/// pinned by `mu_min`, not by the `min(tol, …)` dynamic term. The whole band
/// must certify.
#[test]
fn hs71_obj1e8_certifies_across_tolerances() {
    for tol in ["1e-6", "1e-8", "1e-10"] {
        let report = solve("hs71_obj1e8.nl", &[&format!("tol={tol}")]);
        assert_solved_at_optimum(&report, &format!("hs71×1e8 (tol={tol})"));
    }
}

/// Pin the mechanism rather than the symptom: the unscaled complementarity —
/// the quantity `compl_inf_tol` is actually enforced on — must land under
/// `compl_inf_tol` (1e-4). Pre-fix it is held at `mu_min/df ≈ 1.2e-4`, a hard
/// 20% over, and no amount of further iteration can clear it because μ is
/// already at its floor.
#[test]
fn hs71_obj1e8_unscaled_complementarity_clears_compl_inf_tol() {
    let report = solve("hs71_obj1e8.nl", &[]);
    let obj_scale = assert_df_below_cliff(&report, "hs71×1e8 (defaults)");
    let unscaled_compl = report.statistics.final_compl / obj_scale;
    assert!(
        unscaled_compl <= 1e-4,
        "unscaled complementarity {unscaled_compl} exceeds compl_inf_tol=1e-4, \
         so no strict certificate is reachable no matter how long the solve \
         runs (issue #266)",
    );
}

/// A user forcing the pre-fix behaviour (`mu_min` big enough to block the
/// certificate) must still be able to *loosen* `compl_inf_tol` and get their
/// certificate — i.e. the fix must key on the option values, not constants.
/// With `compl_inf_tol=1e-2`, the cliff moves to `df* = 1e-9`, well below this
/// fixture's `df ≈ 8.3e-8`.
#[test]
fn hs71_obj1e8_certifies_with_loosened_compl_inf_tol() {
    let report = solve("hs71_obj1e8.nl", &["compl_inf_tol=1e-2"]);
    assert_solved_at_optimum(&report, "hs71×1e8 (compl_inf_tol=1e-2)");
}

/// The same unconverted `mu_min` lived in the adaptive strategy's clamps
/// (`adaptive.rs`: the fixed-mode reduction, the fixed-mode re-seed, the
/// oracles' internal `[mu_min, mu_max]` bands, and the final band clamp).
/// The consequence there is milder — the acceptable-level fallback rescues
/// the solve into `Solved_To_Acceptable_Level` — but code 100 is still
/// outside AMPL's 0..99 solved band. Post-fix the strict certificate must be
/// issued.
#[test]
fn hs71_obj1e8_certifies_under_adaptive_mu_strategy() {
    let report = solve("hs71_obj1e8.nl", &["mu_strategy=adaptive"]);
    assert_solved_at_optimum(&report, "hs71×1e8 (mu_strategy=adaptive)");
}
