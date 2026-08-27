//! gh #805 — the negative-curvature floor holds the *best* certificate the
//! escapes have left, not the most recent one.
//!
//! gh #797's `neg_curv_escapes` documents an unconditional guarantee: the
//! escape "cannot return a worse answer than leaving it off would have",
//! because the certified stationary point is snapshotted as a floor before the
//! step and handed back unless the continuation comes back with a certificate
//! of its own at a better point. That was enforced at the default
//! `neg_curv_escapes = 1` and nowhere above it: `try_neg_curv_escape` REPLACED
//! the floor on every escape, so escape 2 dropped the point escape 1 had
//! floored and nothing ever compared the two. With `f(B) > f(A)` a lost second
//! bet then reports a point worse than `neg_curv_escapes = 0` returns.
//!
//! The fix ranks instead of replacing, with the same status-dominant order
//! `honour_neg_curv_floor` uses on the way out. Ranking rather than simply
//! keeping the first floor is what these tests are here to justify: keeping
//! the first is the mirror-image bug, and this fixture makes it visible.
//!
//! # The fixture
//!
//! `nonconvex_two_escapes.nl` — see `fixtures/nonconvex_two_escapes.py` for
//! the model, the geometry and why the coefficients are what they are. It is
//! the first fixture in the corpus that places **two** escapes, so it is also
//! the first that reaches any of the multi-escape accounting at all. The three
//! points it walks are
//!
//! ```text
//!     A = (0, 0)         f =     0        certified, maximum along x0
//!     B = (±1, 0)        f =    -0.225    certified, saddle: x1 goes down
//!     C = (±2, ±1.5)     f = -6752.25     the global minimum
//! ```
//!
//! # What these tests do and do not discriminate
//!
//! They pin the ranking against the *keep the first floor* alternative
//! (measured: `a_lost_bet_after_two_escapes_hands_back_the_better_certificate`
//! reports `0` instead of `-0.225` under it — `neg_curv_escapes = 2` coming
//! back worse than `= 1`).
//!
//! They do **not** discriminate the fix from the pre-#805 replace-the-floor
//! code, and cannot: the continuation that reaches B descends from A, so
//! `f(B) < f(A)` here and the most recent certificate *is* the best one. A
//! witness for the other order needs `f(B) > f(A)`, which within one solve
//! takes something non-monotone in `f` — a μ update the escape straddles, or a
//! restoration excursion — and gh #805 records that no such witness could be
//! constructed on the #797 fixtures either. That is what makes the defect
//! latent rather than live, and it is why the fix keeps *both* orders correct
//! instead of only the one a fixture can reach.

use pounce_solve_report::SolveReport;
use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("nonconvex_two_escapes.nl");
    p
}

fn solve(tag: &str, opts: &[&str]) -> SolveReport {
    let json = std::env::temp_dir().join(format!("pounce_issue_805_{tag}.json"));
    let _ = std::fs::remove_file(&json);
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .args(opts)
        .output()
        .expect("spawn pounce");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(0), "must solve:\n{combined}");
    let text = std::fs::read_to_string(&json).expect("JSON report should be written");
    let _ = std::fs::remove_file(&json);
    serde_json::from_str(&text).expect("deserialize report")
}

/// Two escapes are actually placed, and each one is worth its budget. Without
/// this the tests below would be asserting things about a code path the corpus
/// never enters — which was the state of the corpus before this fixture, and
/// is how gh #805 stayed invisible.
#[test]
fn the_budget_buys_a_strictly_better_certificate_each_time() {
    let none = solve("esc0", &["neg_curv_escapes=0"]);
    let one = solve("esc1", &["neg_curv_escapes=1"]);
    let two = solve("esc2", &["neg_curv_escapes=2"]);

    for r in [&none, &one, &two] {
        assert_eq!(r.solution.solve_result_num, 0, "AMPL srn 0 = solved");
    }
    assert!(
        none.solution.objective.abs() < 1e-9,
        "with the escape off the solve certifies the maximum A (obj 0); got {}",
        none.solution.objective
    );
    assert!(
        (one.solution.objective + 0.225).abs() < 1e-6,
        "one escape reaches the saddle B (obj -0.225); got {}",
        one.solution.objective
    );
    assert!(
        (two.solution.objective + 6752.25).abs() < 1e-3,
        "the second escape leaves B along x1 and reaches the global minimum C \
         (obj -6752.25); got {}",
        two.solution.objective
    );
}

/// A third escape is refused: `C` is a genuine minimum, the probe finds no
/// negative curvature there and declines. This is what makes the test above a
/// test of *two* escapes rather than of "more is better" — the ladder stops
/// where the second-order question stops having an answer.
#[test]
fn a_third_escape_finds_nothing_to_spend_itself_on() {
    let two = solve("ladder2", &["neg_curv_escapes=2"]);
    let three = solve("ladder3", &["neg_curv_escapes=3"]);
    assert_eq!(
        two.solution.objective, three.solution.objective,
        "the third escape must be declined, not merely unproductive"
    );
}

/// The gh #805 test. Two escapes are placed and the second one's continuation
/// is cut before it can certify anything; the floor is what gets reported, and
/// it must be `B` — the better of the two certificates the escapes left — not
/// `A`, the one the first bet was placed from.
///
/// `max_iter = 13` sits between the second escape (iteration 9) and the
/// certificate its continuation would otherwise reach (iteration 17), so the
/// continuation is guaranteed to run out of budget and
/// `terminate_at_neg_curv_floor` hands the floor back.
///
/// Measured with the floor kept at the *first* certificate instead of ranked —
/// gh #805's suggested fix — this reports `0`: `neg_curv_escapes = 2` comes
/// back with a worse answer than `= 1` does, which is the same class of defect
/// pointed the other way. `mu_strategy_fallback=no` is pinned for the reason
/// `issue_797_neg_curvature_escape.rs` documents: the retry re-solves under
/// the other μ schedule on exactly this status and can satisfy the assertion
/// without the floor being involved at all.
#[test]
fn a_lost_bet_after_two_escapes_hands_back_the_better_certificate() {
    let report = solve(
        "lost_bet2",
        &[
            "neg_curv_escapes=2",
            "max_iter=13",
            "mu_strategy_fallback=no",
        ],
    );
    assert_eq!(
        report.solution.solve_result_num, 0,
        "the floor is a strict certificate, so the status stays Solve_Succeeded"
    );
    assert!(
        (report.solution.objective + 0.225).abs() < 1e-6,
        "a cut continuation must hand back the best certificate the escapes \
         left (B, obj -0.225), not the one the first bet was placed from \
         (A, obj 0); got {}",
        report.solution.objective
    );
}

/// The guarantee as gh #805 states it, asserted directly: however much budget
/// the option is given, and however the continuation ends, the answer is never
/// worse than the one `neg_curv_escapes = 0` returns.
#[test]
fn no_budget_returns_worse_than_the_escape_being_off() {
    let baseline = solve("guarantee_off", &["neg_curv_escapes=0", "max_iter=13"]);
    for (tag, escapes) in [("g1", "1"), ("g2", "2"), ("g3", "3")] {
        let report = solve(
            tag,
            &[
                &format!("neg_curv_escapes={escapes}"),
                "max_iter=13",
                "mu_strategy_fallback=no",
            ],
        );
        assert_eq!(report.solution.solve_result_num, 0);
        assert!(
            report.solution.objective <= baseline.solution.objective + 1e-9,
            "neg_curv_escapes={escapes} returned {} against the escape-off \
             answer {}",
            report.solution.objective,
            baseline.solution.objective
        );
    }
}
