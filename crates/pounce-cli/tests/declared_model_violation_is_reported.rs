//! The convex arm reports how far the returned point sits outside the model
//! **as declared**, not only outside the widened one it was handed.
//!
//! `bound_relax_factor` widens the inequality rows and the variable box by
//! `min(factor, cap)·|b|`. The convex arm no longer does this by default —
//! it moved answers, so it is opt-in — but the NLP arm still does, it is
//! always available here by name, and either way the widened model is the one
//! `final_constr_viol` measures. What the caller cannot otherwise learn is how
//! far the returned point sits outside the model they wrote. The fixture here
//! is that shape stripped to its minimum:
//!
//! ```text
//! min -x - y   s.t.   x + y <= 500,   0 <= x,y <= 1000
//! ```
//!
//! The row is active at the optimum, so the widening buys objective directly.
//! Under `bound_relax_factor=1e-8` the solve prints
//! `Constraint violation....: 0.0` — true of the widened row
//! `x + y <= 500 + 5e-6` — while returning `-500.0000005`, which is *better*
//! than the declared optimum `-500` because the point is `5e-6` outside the
//! declared row. Without the extra number a caller reads that block as their
//! model holding exactly; it holds to `5e-6`.
//!
//! Measured on netlib the same shape reaches `4.99e-06` on `afiro` (declared
//! row `b = 500`, reported `8.68e-13`) and `1.97e-05` on `25fv47` (reported
//! `2.19e-11`).
//!
//! `final_constr_viol` deliberately still measures the widened model: it is
//! what every acceptance gate reads and what gh #712's success-verdict
//! invariant is asserted on. Redefining it broke
//! `issue_689_direct_driver_scaled_feasible.rs`, correctly.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;

/// `min(factor, cap) * |b|` for the fixture's row: `1e-8 * 500`.
const EXPECTED_WIDENING: f64 = 5e-6;
/// The convex arm solves the declared model by default, so a test about the
/// widening has to ask for it.
const RELAX: &str = "bound_relax_factor=1e-8";

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("declared_row_relaxation.nl");
    p
}

fn tmp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_declared_viol_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

fn solve(extra: &[&str]) -> SolveReport {
    let json_path = tmp_path("report.json");
    let sol_path = tmp_path("out.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture())
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path);
    for opt in extra {
        cmd.arg(opt);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

#[test]
fn the_declared_violation_is_reported_beside_the_widened_one() {
    let r = solve(&[RELAX]);
    let stats = &r.statistics;

    // The widened model is satisfied — that is the number the convergence
    // test read, and it is why this solve is `Optimal`.
    assert!(
        stats.final_constr_viol <= 1e-9,
        "final_constr_viol should measure the widened model the solver was \
         handed, and the point satisfies it; got {:e}",
        stats.final_constr_viol
    );

    // The declared model is not, by the width of the widening.
    let declared = stats.final_declared_constr_viol;
    assert!(
        declared.is_finite(),
        "final_declared_constr_viol must be computed when the widening is \
         active; got {declared:e}"
    );
    assert!(
        (declared - EXPECTED_WIDENING).abs() <= 0.05 * EXPECTED_WIDENING,
        "the point should sit one widening outside the declared row \
         ({EXPECTED_WIDENING:e} = 1e-8 * 500); got {declared:e}"
    );
    // The whole point: the two disagree, so reporting only the first hides
    // the second.
    assert!(
        declared > stats.final_constr_viol * 1e3,
        "the two measurements must differ, or this fixture has stopped \
         exercising the gap: declared {declared:e} vs widened {:e}",
        stats.final_constr_viol
    );
}

#[test]
fn the_objective_is_past_the_declared_optimum_by_the_widening() {
    // `min -x-y` over `x+y <= 500` has optimum `-500`. The widened row buys
    // exactly the widening, so a caller comparing against a published optimum
    // sees a solver that beat it — the visible symptom of the invisible
    // violation above.
    let r = solve(&[RELAX]);
    let obj = r.solution.objective;
    assert!(
        obj < -500.0,
        "the widened row should let the objective past the declared optimum; \
         got {obj:e}"
    );
    assert!(
        (obj + 500.0).abs() <= 2.0 * EXPECTED_WIDENING,
        "and by no more than the widening; got {obj:e}"
    );
}

#[test]
fn no_widening_means_nothing_extra_to_report() {
    // The convex arm's DEFAULT. With no widening the two measurements coincide
    // by construction, so the declared field is left uncomputed rather than
    // duplicating a number the caller already has.
    let r = solve(&[]);
    assert!(
        r.statistics.final_declared_constr_viol.is_nan(),
        "expected NaN (nothing to add) with no widening; got {:e}",
        r.statistics.final_declared_constr_viol
    );
    assert!(
        r.statistics.final_constr_viol <= 1e-8,
        "and the declared model should now be satisfied outright; got {:e}",
        r.statistics.final_constr_viol
    );
    assert!(
        (r.solution.objective + 500.0).abs() <= 1e-6,
        "objective should land on the declared optimum -500; got {:e}",
        r.solution.objective
    );
}

// ── the NLP arm ──────────────────────────────────────────────────────────────
//
// The convex arm no longer widens by default, so the arm where this number
// now earns its keep is the NLP one — which must keep widening (a
// feasible-iterate log-barrier needs `x` strictly inside its bounds) and
// whose `final_constr_viol` is the *internal slack* measure, not a statement
// about the user's model at all. On netlib `wood1p` it reports `1.71e-14` at
// a point that is `7.96e-09` outside the declared rows and `9.84e-09` outside
// the declared box, and returns an objective `4.4e-05` from the optimum HiGHS
// reports. Five orders between the number shown and the number that matters.

fn nlp_solve(fixture: &str, extra: &[&str]) -> SolveReport {
    let json_path = tmp_path("nlp.json");
    let sol_path = tmp_path("nlp.sol");
    let mut src = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src.push("tests/fixtures");
    src.push(fixture);
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(&src)
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path)
        .arg("solver_selection=nlp");
    for o in extra {
        cmd.arg(o);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// A binding **row**: the widened row is satisfied, the declared one is not.
#[test]
fn the_nlp_arm_reports_a_declared_row_violation() {
    let r = nlp_solve("bound_relax_row.nl", &[]);
    let d = r.statistics.final_declared_constr_viol;
    assert!(
        (d - 1e-4).abs() < 1e-6,
        "expected the row widening (min(1e-8,1e-4)*1e4 = 1e-4) to show up as \
         the declared violation; got {d:e}"
    );
    assert!(
        r.statistics.final_constr_viol < d / 1e3,
        "and the internal measure should be far smaller, which is the whole \
         point: {:e} vs {d:e}",
        r.statistics.final_constr_viol
    );
}

/// A binding **variable bound**. This is the half that needs the lift out of
/// the compressed bound spaces (`Nlp::declared_box_violation`); without it the
/// number came back `0.0` on this fixture while the point sat `1e-4` outside
/// its declared box.
#[test]
fn the_nlp_arm_reports_a_declared_box_violation() {
    let r = nlp_solve("bound_relax_var.nl", &[]);
    let d = r.statistics.final_declared_constr_viol;
    assert!(
        (d - 1e-4).abs() < 1e-6,
        "expected the box widening to show up as the declared violation; got \
         {d:e}"
    );
}

/// With the widening off the NLP arm still reports the measurement, in the
/// caller's units (gh#987): the field used to read NaN here, so under row
/// scaling the user-unit violation was reported nowhere.
#[test]
fn the_nlp_arm_reports_the_declared_violation_even_without_a_widening() {
    let r = nlp_solve("bound_relax_row.nl", &["bound_relax_factor=0"]);
    let d = r.statistics.final_declared_constr_viol;
    assert!(
        d.is_finite() && d <= 1e-6,
        "expected a finite, converged user-unit violation at bound_relax_factor=0; got {d:e}"
    );
}
