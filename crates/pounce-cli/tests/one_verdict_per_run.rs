//! A run reports its verdict **once**, and reports the problem-statistics
//! header once, however many attempts it took to get there.
//!
//! Every retry driver — the ℓ₁ fallback, the μ-strategy fallback, the
//! dual-divergence retry, the second-opinion ladder — re-enters the solve
//! routine that emits both blocks. So a retried run used to print the whole
//! Ipopt-style header per attempt and an `EXIT:` / `POUNCE <version>:` pair
//! per attempt, with nothing between the copies saying a second attempt had
//! started. Two distinct complaints came out of that:
//!
//! * it read as the solver printing everything twice for no reason; and
//! * the mid-run verdict read as the final answer. On `csfi2` the terminal
//!   said `POUNCE 0.11.0: Solved To Acceptable Level.` and then went on to
//!   report `Optimal Solution Found.`
//!
//! The second is not only a presentation problem. gh#508 is the same defect
//! read by a machine: `validation/p3_control.py` keeps the last `EXIT:` line
//! it sees and pairs it with the `.sol`, and when no rung promoted, the last
//! banner was the last *rejected* rung's while the `.sol` carried the
//! original verdict. The fix at the time was an arbiter in `main.rs` that
//! re-emitted the true verdict after the fact. Deferring the verdict to the
//! end of the run makes the terminal's final word correct by construction
//! instead, and that arbiter is gone — so this file is what keeps the
//! property it used to enforce.
//!
//! ## Why the header rule is "compare", not "count"
//!
//! The header is suppressed only when it would be *identical* to the one
//! already printed, because a retry can legitimately change the problem: the
//! ℓ₁ wrapper adds one slack per equality row, so its attempt really is a
//! larger problem and reprinting is informative rather than noise.
//! `the_header_reprints_when_the_retry_changes_the_problem` is that case, and
//! it is the guard against "fixing" this by counting attempts.

use std::process::Command;

fn pounce_exe() -> String {
    let mut p = std::path::PathBuf::from(env!("CARGO_BIN_EXE_pounce"));
    p.set_extension(std::env::consts::EXE_EXTENSION);
    p.to_string_lossy().into_owned()
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

struct Run {
    stdout: String,
}

impl Run {
    fn go(model: &str, extra: &[&str]) -> Self {
        let mut cmd = Command::new(pounce_exe());
        cmd.arg(fixture(model)).arg("--no-sol");
        for o in extra {
            cmd.arg(o);
        }
        let out = cmd.output().expect("spawn pounce");
        Self {
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        }
    }

    fn count(&self, prefix: &str) -> usize {
        self.stdout
            .lines()
            .filter(|l| l.starts_with(prefix))
            .count()
    }

    /// The `EXIT:` line's verdict text, which must be unique to be read.
    fn verdict(&self) -> String {
        let mut it = self.stdout.lines().filter_map(|l| l.strip_prefix("EXIT: "));
        let first = it.next().unwrap_or_default().to_string();
        assert!(
            it.next().is_none(),
            "more than one EXIT: line:\n{}",
            self.stdout
        );
        first
    }

    /// How many solve attempts the run actually made. The per-attempt
    /// statistics block is still printed per attempt — only the verdict and
    /// the header are per-run — so this counts them without depending on
    /// either of the things under test.
    fn attempts(&self) -> usize {
        self.count("Number of Iterations....:")
    }
}

/// The μ-strategy fallback: two attempts inside one `optimize_tnlp`, the
/// first ending `Solved_To_Acceptable_Level` and the second certifying.
///
/// This is the run from the bug report. The first attempt's verdict is the
/// one that used to read as the final answer.
#[test]
fn a_retried_run_reports_one_verdict_and_it_is_the_last_attempts() {
    let run = Run::go("csfi2.nl", &[]);
    assert!(
        run.attempts() >= 2,
        "csfi2 should retry under stock options; it made {} attempt(s). \
         If the retry stopped firing this test is no longer measuring \
         anything — pick a model that still retries.\n{}",
        run.attempts(),
        run.stdout
    );
    assert_eq!(
        run.count("EXIT:"),
        1,
        "one verdict per run, not one per attempt:\n{}",
        run.stdout
    );
    assert_eq!(
        run.count("POUNCE 0"),
        1,
        "the version-stamped verdict line is per-run too:\n{}",
        run.stdout
    );
    assert_eq!(
        run.verdict(),
        "Optimal Solution Found.",
        "the verdict must be the promoted attempt's, not the first \
         attempt's Solved_To_Acceptable_Level:\n{}",
        run.stdout
    );
    assert_eq!(
        run.count("Total number of variables"),
        1,
        "the problem is unchanged between attempts, so its header is \
         printed once:\n{}",
        run.stdout
    );
}

/// The second-opinion ladder, promoting. Its rungs are `optimize_tnlp` calls
/// in their own right, so they are outside the funnel that scopes the header
/// memo — a separate mechanism covers them, and this is what checks it.
///
/// The base solve here reports `Infeasible_Problem_Detected` and a rung
/// recovers it, so a verdict printed any earlier than the end of the run is
/// not merely premature but *wrong*.
#[test]
fn a_promoting_ladder_reports_the_promoted_verdict_once() {
    let run = Run::go("degenerate_start_hs008.nl", &[]);
    assert!(
        run.attempts() >= 2,
        "expected a base solve plus at least one rung, got {} attempt(s):\n{}",
        run.attempts(),
        run.stdout
    );
    assert_eq!(
        run.count("EXIT:"),
        1,
        "the ladder's rungs must not each report a verdict:\n{}",
        run.stdout
    );
    assert_eq!(
        run.verdict(),
        "Optimal Solution Found.",
        "the promoted rung's verdict is the run's verdict:\n{}",
        run.stdout
    );
    assert_eq!(
        run.count("Total number of variables"),
        1,
        "one header across the base solve and every rung:\n{}",
        run.stdout
    );
}

/// The second-opinion ladder, *not* promoting — gh#508's shape exactly.
///
/// Every rung is rejected, so the verdict that ships is the base solve's.
/// The failure this pins is the one gh#508 reported: the terminal's last word
/// being the last rejected rung's while the `.sol` carried the original.
#[test]
fn a_ladder_that_promotes_nothing_reports_the_kept_verdict_once() {
    let run = Run::go("infeasible_square_scaled_1em4.nl", &[]);
    assert!(
        run.attempts() >= 2,
        "expected a base solve plus rungs, got {} attempt(s):\n{}",
        run.attempts(),
        run.stdout
    );
    assert_eq!(
        run.count("EXIT:"),
        1,
        "one verdict for the run, not one per rejected rung:\n{}",
        run.stdout
    );
    assert!(
        run.verdict().contains("local infeasibility"),
        "the kept verdict must be the one on the terminal; got {:?}\n{}",
        run.verdict(),
        run.stdout
    );
}

/// A run that never retries is unaffected: one attempt, one verdict, one
/// header. The deferral has to be *released* on this path too, and an
/// unmatched acquire would silently cost the run its `EXIT:` line — which is
/// exactly the failure mode a counter-based scheme has.
#[test]
fn a_single_attempt_run_still_reports_its_verdict() {
    let run = Run::go("hs71_obj1e8.nl", &[]);
    assert_eq!(run.attempts(), 1, "expected one attempt:\n{}", run.stdout);
    assert_eq!(run.count("EXIT:"), 1, "verdict lost:\n{}", run.stdout);
    assert_eq!(run.count("POUNCE 0"), 1, "verdict lost:\n{}", run.stdout);
    assert_eq!(run.verdict(), "Optimal Solution Found.");
}

/// `print_level 0` is a request for silence, and the verdict is no more
/// exempt from it than the summary block it used to end.
#[test]
fn print_level_zero_reports_no_verdict_at_all() {
    let run = Run::go("hs71_obj1e8.nl", &["print_level=0"]);
    assert_eq!(run.count("EXIT:"), 0, "expected silence:\n{}", run.stdout);
    assert_eq!(
        run.count("POUNCE 0"),
        0,
        "expected silence:\n{}",
        run.stdout
    );
}

/// The header is suppressed by comparison, not by counting attempts.
///
/// Under the ℓ₁ auto-fallback the first attempt solves the model as written
/// and the retry solves it with one slack per equality row, so the two
/// attempts genuinely differ in size and the reader needs both headers.
/// `mu_strategy_fallback=no` keeps the run to the ℓ₁ driver alone, so the
/// attempt count is the one this test reasons about.
#[test]
fn the_header_reprints_when_the_retry_changes_the_problem() {
    let run = Run::go(
        "csfi2.nl",
        &[
            "l1_fallback_on_restoration_failure=yes",
            "mu_strategy_fallback=no",
        ],
    );
    let headers = run.count("Total number of variables");
    assert!(
        headers >= 2,
        "the ℓ₁ retry adds slack variables, so its header is a different \
         block and must be printed; saw {headers}:\n{}",
        run.stdout
    );
    // The point of the comparison: the two headers disagree about `n`. If
    // this ever reads equal, the fixture stopped exercising the case and the
    // test above it is no longer a contrast.
    let sizes: Vec<&str> = run
        .stdout
        .lines()
        .filter_map(|l| l.strip_prefix("Total number of variables"))
        .map(|l| l.trim_start_matches(['.', ':', ' ']))
        .collect();
    assert!(
        sizes.windows(2).any(|w| w[0] != w[1]),
        "expected the ℓ₁ attempt to report a different variable count; \
         got {sizes:?}\n{}",
        run.stdout
    );
    // …and the verdict is still reported exactly once across both.
    assert_eq!(
        run.count("EXIT:"),
        1,
        "one verdict per run regardless of how many headers:\n{}",
        run.stdout
    );
}
