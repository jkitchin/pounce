//! `linear_system_scaling` values must change the solve, not just parse
//! (#677).
//!
//! `slack-based` was registered — so `OptionsList` accepted it — but the
//! parser routed it to `LinearSystemScalingChoice::None` through a
//! catch-all arm. `mc19` reached the same fallback through a named arm
//! that warns; `slack-based` warned nothing, while a comment directly
//! above the match claimed both did. Setting it was indistinguishable
//! from not setting it.
//!
//! That is not an obscure value: it is what Ipopt's recommended
//! configuration for large collocation NLPs uses, and it appeared in a
//! user's golden configuration — validated across 25 models — reported
//! on #677. The people most likely to set it were the least likely to
//! find out it did nothing.
//!
//! So the test that matters is not "is it accepted" but "does the solve
//! move". #551 puts it directly: "A read site that parses a value and
//! discards it is the same silent no-op this whole line of work exists
//! to kill, and it is indistinguishable from a real fix by inspection.
//! That test is the deliverable, not the line that reads the field."
//!
//! ## Why this stopped comparing iteration counts (gh#693)
//!
//! The original guard read: `slack-based` must not take the *same number
//! of iterations* as `none`. That is a proxy for the real property, and
//! it turned out to be a bad one in two independent ways.
//!
//! It is **platform-fragile**. Iteration counts are the most
//! platform-sensitive numbers a solver reports, and on macOS at
//! `ac18ba6d` — before gh#693 — this file already failed, on the
//! `slack-based` vs `ruiz` leg: both took 103 iterations on `cresc4`
//! while CI's Linux runners saw them differ. A guard that is red on a
//! developer's machine and green in CI gets read as noise, which is how
//! it stops being a guard.
//!
//! It is **fixture-dependent in a way the property is not**. gh#693
//! left `cresc4` better conditioned, and all four scaling choices now
//! reach the optimum in 69 iterations. Nothing is wrong: `slack-based`
//! still reaches the linear solver and still changes the arithmetic —
//! `cresc4`'s objective is 0.8718975393087962 under `none` and
//! 0.8718975392737567 under `slack-based`. The *option* works; the
//! *proxy* went quiet, and an assertion on the proxy would have gone
//! quiet with it while claiming to guard gh#677.
//!
//! So the comparison is now on the solve's numerical output, which is
//! what gh#677's defect actually looked like — the report above says it
//! in as many words: "`slack-based` produced byte-identical output to
//! `none`, because it *was* `none`." An option routed to a catch-all
//! cannot perturb a single bit, on any platform, so byte-identity is
//! both the exact signal and a portable one.
//!
//! Three relations are pinned, on three fixtures, and they hold
//! identically on `ac18ba6d` and after gh#693:
//!
//! ```text
//!                 slack != none   slack != ruiz   mc19 == none
//!   cresc4              yes             yes            yes
//!   airport             yes             yes            yes
//!   csfi2               yes             yes            yes
//! ```
//!
//! `mc19 == none` is the positive control and it is the more valuable
//! half: it is a genuine fallback, so it proves the comparison can still
//! see a no-op. A test that only ever asserts "these differ" cannot tell
//! a working option from a comparison that has broken open.
//!
//! (`eigenb2` is a fixture where `slack-based` and `none` *do* agree
//! bit for bit, on both builds. It is named here so the next reader
//! knows the relation is a property of the fixture as well as the
//! option, and does not add a fixture to this list without checking.)

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Scratch directories must not collide: the two tests here run
/// concurrently and both solve the same fixtures, so naming the
/// directory after the fixture and the scaling choice is not enough —
/// one test's cleanup would delete the other's output mid-run.
static RUN: AtomicUsize = AtomicUsize::new(0);

/// Solve a fixture and return the raw JSON report as text.
///
/// `linear_scaling_on_demand=no` is essential: the default is `yes`,
/// which computes scaling only on a factorization that already looks
/// troubled. On a fixture that solves cleanly, every scaling choice is
/// then identical — which is exactly how a test could "pass" against an
/// unimplemented option. Forcing scaling on every factorization is what
/// makes the comparison meaningful.
fn solve_output(fixture: &str, scaling: &str) -> String {
    let run = RUN.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("pounce_linscale_{}_{run}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join(fixture);
    let mut src = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src.push("tests/fixtures");
    src.push(fixture);
    std::fs::copy(&src, &nl).expect("copy fixture");
    let json = dir.join("out.json");

    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&nl)
        .arg(dir.join("out.sol"))
        .arg("--json-output")
        .arg(&json)
        .arg(format!("linear_system_scaling={scaling}"))
        .arg("linear_scaling_on_demand=no")
        .arg("print_level=0")
        .output()
        .expect("run pounce");
    assert!(
        json.exists(),
        "no JSON for linear_system_scaling={scaling}; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr),
    );
    let text = std::fs::read_to_string(&json).expect("read json");
    let _ = std::fs::remove_dir_all(&dir);
    text
}

/// The numerical content of a solve, with the parts that cannot be
/// compared across two runs removed: the whole `fair_metadata` block
/// (a result id, timestamps, an elapsed time, and the scratch path the
/// fixture was copied to) and every wall-clock timing. A timing is any key
/// ending in `_secs` — the report's naming convention for seconds — which
/// covers the statistics' two and the linear-solver summary's factorization
/// times without listing each one. Everything left is deterministic for a
/// fixed binary and a fixed set of options.
fn numeric_output(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth: i32 = 0;
    let mut in_metadata = false;
    for line in text.lines() {
        let key = line.trim_start();
        if in_metadata {
            depth += line.matches('{').count() as i32;
            depth -= line.matches('}').count() as i32;
            if depth <= 0 {
                in_metadata = false;
            }
            continue;
        }
        if key.starts_with("\"fair_metadata\"") {
            in_metadata = true;
            depth = line.matches('{').count() as i32 - line.matches('}').count() as i32;
            if depth <= 0 {
                in_metadata = false;
            }
            continue;
        }
        if key.starts_with('"') && key.contains("_secs\":") {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// gh#677's defect, stated as the property it actually was: setting
/// `slack-based` must not be indistinguishable from not setting it.
///
/// An option parsed into a catch-all arm cannot change a single bit of
/// the answer, so comparing the numerical output is exact — and unlike
/// an iteration count it means the same thing on every platform.
#[test]
fn slack_based_scaling_changes_the_solve() {
    for fixture in ["cresc4.nl", "airport.nl", "csfi2.nl"] {
        let none = solve_output(fixture, "none");
        let slack = solve_output(fixture, "slack-based");
        let ruiz = solve_output(fixture, "ruiz");

        assert_ne!(
            numeric_output(&slack),
            numeric_output(&none),
            "linear_system_scaling=slack-based produced output identical to \
             `none` on {fixture} — that is exactly gh#677: the option is \
             parsed but not reaching the linear solver",
        );
        assert_ne!(
            numeric_output(&slack),
            numeric_output(&ruiz),
            "slack-based and ruiz produced identical output on {fixture}, \
             which is suspicious enough to check the wiring before \
             trusting it — slack-based is its own method, not an alias \
             for the one that already worked",
        );
    }
}

/// The positive control for the test above, and the more valuable half.
///
/// `mc19` is still unimplemented and still falls back to no scaling, so
/// its output must be *identical* to `none`. That pins two things at
/// once: the fallback stays deliberate rather than becoming another
/// silent one, and `numeric_output` can still see a no-op — a file that
/// only ever asserted "these differ" would pass just as happily if the
/// comparison itself broke open.
#[test]
fn mc19_still_falls_back_to_no_scaling() {
    for fixture in ["cresc4.nl", "airport.nl", "csfi2.nl"] {
        let mc19 = solve_output(fixture, "mc19");
        let none = solve_output(fixture, "none");
        assert_eq!(
            numeric_output(&mc19),
            numeric_output(&none),
            "mc19 now differs from `none` on {fixture} — if it was \
             implemented, update this test, the CHANGELOG, and \
             `docs/src/options.md`",
        );
    }
}
