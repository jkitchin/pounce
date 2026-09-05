//! gh#616 — the corpus cost of the gh#605 least-square-init safeguard,
//! pinned so it is visible to the suite instead of living in a PR body.
//!
//! Under `least_square_init_primal=yes` (off by default), gh#605's
//! safeguard is a broad win — ten fixtures reach the same answer in
//! fewer iterations, several dramatically — and it cost two tolerance
//! downgrades when gh#616 measured it: `csfi2` and `eigenb2` finished
//! at `SolvedToAcceptableLevel` where the unsafeguarded step reached
//! `SolveSucceeded`.
//!
//! **For one release, one of those two was gone, and nobody went looking
//! for it** (gh#681). gh#588's Q4 evaluates a recognized degree-≤2 row
//! from its constant matrix instead of rebuilding an AD tape every
//! iteration, which reassociates the sums in `eval_g` and `eval_jac_g` —
//! `quad_evaluator_differential.rs` declares those two comparisons
//! non-bitwise in advance, for exactly this reason. `eigenb2` sat close
//! enough to the accept band for the reassociation to carry it across,
//! reaching `SolveSucceeded` in 54 iterations where the tape stalled at
//! `SolvedToAcceptableLevel` in 57. `csfi2` does not move at all, to the
//! bit — the safeguard declines there, so no evaluator change can reach
//! it.
//!
//! gh#702 then compensated that same sum, because the uncompensated
//! association was costing a 1500-variable QCQP 28 extra iterations, and
//! `eigenb2` came back to `SolvedToAcceptableLevel`. So the count is two
//! again, as gh#616 measured it. The lesson is not that either verdict is
//! right: it is that on these two models the status is **downstream
//! round-off**, and the tests below are written to make that visible
//! rather than to defend a particular one.
//!
//! That is a trajectory change and it is pinned as one: the tests below
//! assert both legs, so the fast path's verdict and the tape's are each
//! held in place and a future move is attributable to one of them.
//!
//! gh#616 decided that cost is acceptable and that the accept test must
//! **not** be tightened to chase it. The reasoning is in
//! `docs/src/initialization.md` and the algebra is pinned in
//! `pounce-algorithm/tests/issue_616_ls_init_accept_test.rs`. What this
//! file pins is the measurement, because the failure mode CLAUDE.md
//! warns about is a recorded cost that nothing re-reads: gh#544's
//! `pooling_rt2stp` 206 → 812 sat in a commit message through a
//! release. If someone later retunes the accept test and these statuses
//! move, that is a decision being reversed and it should have to be
//! noticed.
//!
//! Every assertion here is on **status and objective**, plus three
//! iteration-count *relations* (two `==`, one `!=`) that each carry a
//! mechanism claim about which arm of the safeguard ran. No absolute
//! iteration count is asserted: those are the most platform-sensitive
//! numbers in a sweep, and none of gh#616's conclusions rests on a
//! particular one.
//!
//! The measured counts, for the record, under
//! `least_square_init_primal=yes` against `=no` on the parent commit
//! `a44f4e8b`: `csfi2` 35/35, `eigena2` 65/26, `eigenb2` 57/67, `deb7`
//! 202/154, `pooling_rt2stp` 81/298, `unbounded_cubic` 290/290.
//!
//! ## gh#693: the safeguard is unchanged; two of these models stop being
//! ## coin flips
//!
//! gh#693 removed the Tikhonov `δ` from the least-square multiplier
//! initializer. It does **not** change what this safeguard decides. The
//! attribution line is bit-identical across the change on all three
//! models this file drives:
//!
//! ```text
//!   csfi2           violation 1508.554… → 1508.554…  alpha=0   trials=4  declined
//!   eigenb2         violation 1.0 → 0.25000000624999996  alpha=0.5  trials=1  accepted
//!   pooling_rt2stp  violation 4.93000007 → 4.93000007  alpha=0   trials=4  declined
//! ```
//!
//! Same numbers, same arm, same verdict. What moves is the state the
//! initializer's augmented-system solve leaves behind, and it moves the
//! two models that were sitting on the accept band:
//!
//! ```text
//!                  main (0.10.0)              with gh#693
//!   eigenb2 =yes   SolvedToAcceptableLevel 48  SolveSucceeded 17
//!   eigenb2 =no    SolveSucceeded          67  SolveSucceeded 21
//!   eigena2 =yes   SolvedToAcceptableLevel 127 SolveSucceeded 17
//!   csfi2   =yes   SolvedToAcceptableLevel 35  SolvedToAcceptableLevel 35
//! ```
//!
//! The header above warned that a `SolveSucceeded` on `eigenb2` "means
//! something reassociated `eval_g` again and landed on the lucky side of
//! the accept band ... and it is not a fix". That was the right warning
//! and it is worth being precise about why it does not apply here.
//!
//! Each model was re-run at 17 values of `mu_init` at `0.1·(1 ± k·1e-12)`
//! — round-off scale, where a model sitting on a tolerance band scatters
//! and one that is clear of it does not:
//!
//! ```text
//!                          main                       gh#693
//!   eigenb2 =yes   14 Succeeded / 3 Acceptable   17 Succeeded
//!   eigena2 =yes   11 Succeeded / 6 Acceptable   17 Succeeded
//!   csfi2   =yes   17 Acceptable                 17 Acceptable
//! ```
//!
//! So on `main` these two statuses were never stable facts: the values
//! this file pinned were a 3-point island around the default draw, and
//! the majority outcome on `main` itself was already the other one.
//! gh#693 does not carry them across the band, it moves them off it —
//! and `csfi2`, which is genuinely clear of the band, does not move at
//! all, to the bit, in either build. The pins below are updated to the
//! measured outcome, and the round-off screen is the reason they are now
//! stronger than what they replace rather than weaker.
//!
//! This also amends gh#706, which recorded `eigena2`'s status as
//! *platform*-dependent. On `main` it is round-off-dependent on a single
//! platform — 11/6 across a `1e-12` perturbation — which is a simpler
//! and worse explanation. It is deterministic here.
//!
//! `pooling_rt2stp` is the one model in this file that gh#693 does not
//! settle, and the assertion it used to carry was never true in the way
//! the file believed. See the test itself.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use pounce_cli::solve_report::SolveReport;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(format!("{name}.nl"));
    p
}

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_gh616_{}_{seq}_{tag}.{ext}",
        std::process::id()
    ));
    p
}

/// Solve `model` and return its report. `ls_init` picks the route:
/// `true` sets `least_square_init_primal=yes`, `false` leaves the
/// default (`no`).
///
/// The exit status is deliberately not asserted: `unbounded_cubic` is
/// meant to end at `DivergingIterates`, which the CLI reports with a
/// nonzero exit, and that verdict is exactly what one of the tests
/// below is checking. The report file being written and parseable is
/// the success condition here.
fn solve(model: &str, ls_init: bool) -> SolveReport {
    solve_with_env(model, ls_init, &[])
}

/// `solve`, plus environment for the child process. The only caller
/// that needs it sets `POUNCE_DBG_NO_QUAD=1` to force the AD tape,
/// which is how a fixture that moved under gh#588's Q4 is made to say
/// so instead of being asserted around.
fn solve_with_env(model: &str, ls_init: bool, env: &[(&str, &str)]) -> SolveReport {
    solve_with(model, ls_init, env, &[])
}

/// `solve_with_env`, plus extra CLI options appended verbatim. Used by
/// the barrier-independence screen below, which is the same run at two
/// values of `mu_init`.
fn solve_with(model: &str, ls_init: bool, env: &[(&str, &str)], opts: &[&str]) -> SolveReport {
    let json = tmp_path(model, "json");
    // Explicit, so a fixture that solves does not drop a `.sol` beside
    // the `.nl` in the source tree.
    let sol = tmp_path(model, "sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(model))
        .arg("--sol-output")
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .arg("--json-detail")
        .arg("summary")
        .arg("print_level=0")
        // Every test in this file compares two routes that differ only in
        // `least_square_init_primal`, so everything else has to be held
        // fixed. The default-on `mu_strategy_fallback` is not: gh#757 lets
        // it retry a `Solved_To_Acceptable_Level` exit under stock options,
        // and on `csfi2` that retry succeeds. Left on, it would erase the
        // downgrade this file exists to measure -- and erase it on only one
        // of the two arms, since the arms do not both end acceptable. The
        // pin keeps the pins below measuring the safeguard rather than the
        // retry; the retry's own effect on `csfi2` is covered by
        // `issue_757_acceptable_retry`.
        .arg("mu_strategy_fallback=no");
    if ls_init {
        cmd.arg("least_square_init_primal=yes");
    }
    for o in opts {
        cmd.arg(o);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "no report for {model} (exit {:?}, {e}); stderr:\n{}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr),
        )
    });
    let _ = std::fs::remove_file(&json);
    let _ = std::fs::remove_file(&sol);
    serde_json::from_str(&text).expect("parse SolveReport JSON")
}

/// The attribution channel gh#616 added, because taking this
/// measurement without it meant editing `init/default.rs` to print the
/// report and rebuilding the workspace — for every hypothesis. The
/// accessor `IpoptApplication::least_square_init_report` is not
/// reachable from the CLI, and the fixture sweep runs the CLI.
///
/// One line per solve on the `pounce::algorithm` target at `debug`,
/// carrying the same fields as the accessor. Silent at the default log
/// level: this must not become per-iteration noise on a normal run.
#[test]
fn the_safeguard_decision_is_attributable_from_the_cli() {
    let sol = tmp_path("eigenb2_log", "sol");
    let run = |rust_log: Option<&str>| -> String {
        let mut cmd = Command::new(pounce_exe());
        cmd.arg(fixture("eigenb2"))
            .arg("--sol-output")
            .arg(&sol)
            .arg("print_level=0")
            .arg("least_square_init_primal=yes");
        match rust_log {
            Some(v) => cmd.env("RUST_LOG", v),
            None => cmd.env_remove("RUST_LOG"),
        };
        let out = cmd.output().expect("spawn pounce");
        String::from_utf8_lossy(&out.stderr).into_owned()
    };

    let verbose = run(Some("pounce::algorithm=debug"));
    let _ = std::fs::remove_file(&sol);
    assert!(
        verbose.contains("least_square_init_primal safeguard decision"),
        "RUST_LOG=pounce::algorithm=debug did not surface the \
         safeguard's decision; stderr was:\n{verbose}",
    );
    // The decision itself, not just the headline — this is what makes a
    // moving fixture attributable to an arm of the safeguard rather
    // than guessed at.
    for field in [
        "violation_initial",
        "violation_final",
        "alpha",
        "rejected_trials",
        "termination",
    ] {
        assert!(
            verbose.contains(field),
            "attribution line is missing `{field}`:\n{verbose}",
        );
    }
    assert!(
        verbose.contains("accepted"),
        "eigenb2 accepts a backtracked step, so the line should say so:\
         \n{verbose}",
    );

    let quiet = run(None);
    let _ = std::fs::remove_file(&sol);
    assert!(
        !quiet.contains("least_square_init_primal safeguard decision"),
        "the safeguard decision must stay off the default log:\n{quiet}",
    );
}

fn status_of(r: &SolveReport) -> String {
    format!("{:?}", r.solution.status)
}

/// Strip ANSI SGR escapes. `tracing_subscriber`'s fmt layer colours
/// its field names whether or not stderr is a terminal, so a captured
/// attribution line holds `\x1b[3mviolation_initial\x1b[0m\x1b[2m=\x1b[0m1.0`
/// where the eye reads `violation_initial=1.0`.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        if it.peek() == Some(&'[') {
            it.next();
            // CSI runs to the first byte in @..=~.
            for c2 in it.by_ref() {
                if ('\u{40}'..='\u{7e}').contains(&c2) {
                    break;
                }
            }
        }
    }
    out
}

/// The safeguard's attribution line for `model` under
/// `least_square_init_primal=yes`, parsed to field -> value.
///
/// This is the channel gh#616 added, used for the thing it was added
/// for: comparing two models' safeguard decisions without editing
/// `init/default.rs` and rebuilding.
fn safeguard_decision(model: &str) -> BTreeMap<String, String> {
    let sol = tmp_path(model, "sol");
    let out = Command::new(pounce_exe())
        .arg(fixture(model))
        .arg("--sol-output")
        .arg(&sol)
        .arg("print_level=0")
        .arg("least_square_init_primal=yes")
        .env("RUST_LOG", "pounce::algorithm=debug")
        .output()
        .expect("spawn pounce");
    let _ = std::fs::remove_file(&sol);
    let stderr = strip_ansi(&String::from_utf8_lossy(&out.stderr));
    let tail = stderr
        .lines()
        .find_map(|l| l.split_once("safeguard decision"))
        .unwrap_or_else(|| panic!("no safeguard attribution line for {model}; stderr:\n{stderr}"))
        .1;
    // Values are `key=value`, and a value containing spaces is quoted:
    // a declining model reports `termination="no improvement"`, which a
    // plain whitespace split silently truncates to `"no`.
    let mut out = BTreeMap::new();
    let mut pending: Option<(String, String)> = None;
    for tok in tail.split_whitespace() {
        match pending.take() {
            Some((k, mut v)) => {
                v.push(' ');
                v.push_str(tok);
                if tok.ends_with('"') {
                    out.insert(k, v.trim_matches('"').to_string());
                } else {
                    pending = Some((k, v));
                }
            }
            None => {
                let Some((k, v)) = tok.split_once('=') else {
                    continue;
                };
                if v.starts_with('"') && !(v.len() > 1 && v.ends_with('"')) {
                    pending = Some((k.to_string(), v.to_string()));
                } else {
                    out.insert(k.to_string(), v.trim_matches('"').to_string());
                }
            }
        }
    }
    if let Some((k, v)) = pending {
        out.insert(k, v.trim_matches('"').to_string());
    }
    out
}

/// What the safeguard costs, measured on the route users actually get.
///
/// gh#616 recorded two downgrades. On this branch there is one.
///
/// `csfi2` still reaches 55.0176045 at `SolvedToAcceptableLevel`, bit
/// for bit what `=no` reaches, because the safeguard declines every
/// trial there — a decline is not a step, so there is no trajectory for
/// an evaluator change to perturb.
///
/// `eigenb2` downgrades too, which is what gh#616 measured in the first
/// place. It spent one release not doing so: gh#588's Q4 reassociated
/// `eval_g` and nudged this model — which sat close enough to the accept
/// band to be nudged — across it. That was never a better safeguard, and
/// the version of this test written against it said so.
///
/// gh#702 compensated the same sum and put `eigenb2` back where gh#616
/// found it. Three associations of one dot product, three answers:
///
/// | route                       | status                    | iters | objective          |
/// |-----------------------------|---------------------------|-------|--------------------|
/// | tape (`POUNCE_DBG_NO_QUAD`) | `SolvedToAcceptableLevel` |    57 | 1.59999999134715   |
/// | Q4, uncompensated           | `SolveSucceeded`          |    54 | 1.599999999992518  |
/// | Q4, compensated (gh#702)    | `SolvedToAcceptableLevel` |    48 | 1.599999996403372  |
///
/// Two of the three agree with gh#616, and the odd one out is the one
/// nobody designed. Do not read the recovery as progress if it comes
/// back — check which association produced it.
///
/// The `POUNCE_DBG_NO_QUAD=1` leg no longer discriminates, since both
/// evaluator routes agree. It stays as a cross-check: the two routes
/// agreeing on this model is the normal state, and Q4's window was the
/// exception.
///
/// gh#693 adds a fourth row to that table, and it is the one that
/// settles it — `SolveSucceeded` in 17 iterations on both evaluator
/// routes, and 17/17 under the round-off screen in the header. The three
/// rows above disagree because all three sat on the accept band; this
/// one is off it. The test name changed with it: the safeguard's
/// measured cost is now `csfi2` alone.
#[test]
fn the_safeguards_measured_cost_is_now_csfi2_alone() {
    let csfi2 = solve("csfi2", true);
    assert_eq!(
        status_of(&csfi2),
        "SolvedToAcceptableLevel",
        "csfi2 under least_square_init_primal=yes; gh#616 accepted this \
         downgrade deliberately — if it moved, say which change moved it",
    );
    assert!(
        (csfi2.solution.objective - 55.0176045).abs() < 1e-5,
        "csfi2 objective drifted: {}",
        csfi2.solution.objective,
    );

    let eigenb2 = solve("eigenb2", true);
    assert_eq!(
        status_of(&eigenb2),
        "SolveSucceeded",
        "gh#616 measured a downgrade here and gh#693 removed it. Before \
         reading a return to `SolvedToAcceptableLevel` as a regression, \
         re-run the round-off screen in the header: on `main` this model \
         gave 14 `SolveSucceeded` to 3 `SolvedToAcceptableLevel` across \
         17 draws of `mu_init` at 1e-12 scale, so the old pin was an \
         island. If the 14/3 scatter is back, the accept band is being \
         straddled again and that is the finding — not this status",
    );
    assert!(
        (eigenb2.solution.objective - 1.6).abs() < 1e-6,
        "eigenb2 objective drifted: {}",
        eigenb2.solution.objective,
    );

    let tape = solve_with_env("eigenb2", true, &[("POUNCE_DBG_NO_QUAD", "1")]);
    assert_eq!(
        status_of(&tape),
        "SolveSucceeded",
        "with the fast path off, eigenb2 must agree with the fast path. \
         That is the property this leg is for, and it survives gh#693: \
         the tape route went 57 iterations at `SolvedToAcceptableLevel` \
         to 17 at `SolveSucceeded` alongside the fast path, so the two \
         still agree. A disagreement means one of them moved and the \
         diff says which",
    );
    assert!(
        (tape.solution.objective - 1.6).abs() < 1e-6,
        "eigenb2 objective drifted on the tape route: {}",
        tape.solution.objective,
    );
}

/// The fact that decides gh#616: `eigena2` and `eigenb2` hand the
/// safeguard **the same numbers**, and it takes the same decision on
/// both — `theta_0 = 1.0`, one rejected trial, accepted at
/// `alpha = 0.5` on a step of norm 3.2596011939729705.
///
/// gh#616 read that fact off a status *disagreement*: identical
/// decisions, different outcomes, therefore no criterion computed from
/// the safeguard's own inputs separates them, therefore tightening the
/// accept test could not rescue `eigenb2` without also reaching
/// `eigena2`, which needed no rescuing. That disagreement has since
/// opened and closed twice, so this test asserts the premise
/// **directly**, off the attribution channel, instead of inferring it
/// from the outcomes.
///
/// The conclusion now rests on three independent reassociations of one
/// dot product rather than one. Each moves the `violation_final` the
/// safeguard reports by a few ulps, each moves **both models by the
/// same amount**, and nothing else the accept test reads changes at all:
///
/// | route                    | `violation_final`    | `eigena2`  | `eigenb2` |
/// |--------------------------|----------------------|------------|-----------|
/// | tape                     | 0.2500000062500001   | succeeded  | acceptable|
/// | Q4, uncompensated        | 0.2500000062500003   | succeeded  | succeeded |
/// | Q4, compensated (gh#702) | 0.25000000624999996  | *platform* | acceptable|
/// | …plus gh#693 (`main`)    | 0.25000000624999996  | succeeded  | succeeded |
///
/// The last row is the current one, and it is bit-identical to the row
/// above it in every field the safeguard reads — gh#693 changes nothing
/// this test asserts about the decision, only where the iteration after
/// it lands. That is the fourth reassociation and the third verdict on
/// the pair, which is the point: the accept test never decided either.
///
/// Three associations, three verdicts on the pair, off inputs that stay
/// bit-for-bit identical between the two models every time. So neither
/// model's status was ever a property of the accept test: both are
/// decided downstream, by where the iteration after the safeguard lands
/// relative to the acceptable band, and one reassociated sum in `eval_g`
/// is enough to move either. An accept test tightened to chase `eigenb2`
/// would have been tuned against round-off — gh#616's conclusion, now
/// re-derived twice.
///
/// **`eigena2` is why that third row says *platform*.** Under gh#702's
/// compensated sum it reached `SolveSucceeded` on Linux and
/// `SolvedToAcceptableLevel` in 127 iterations on macOS, for an objective
/// correct to 82.50000000000348 either way. That was not a status this
/// file could pin, and the attempt to pin it is what caught the fact: the
/// first version of this assertion asserted the macOS reading and failed
/// on CI. It was filed as gh#706, whose question was not how to get 51
/// iterations back — that number was luck — but why a model this
/// well-behaved sat close enough to the band that libm decided it.
///
/// gh#693 answered that, and not by carrying the model across the band:
/// it took the barrier parameter out of this model's steering
/// altogether. Measured on gh#693's parent (`fe631b0c^`) against `main`,
/// same machine, same fixture, `least_square_init_primal=yes`:
///
/// ```text
///                mu_init screen at 1e-12      mu_init 0.1 -> 100 (1000x)
///   fe631b0c^    11 succeeded / 6 acceptable  n/a: the 1e-12 screen
///                iterations 61 … 127          already moves it 2x
///   main         17 succeeded, 17 iterations  every iterate bit-identical
///                every time                   — objective, inf_pr, ‖d‖,
///                                             both alphas; only the
///                                             printed lg(mu) column moves
/// ```
///
/// A trajectory that swings 61 → 127 iterations on a last-ulp change in
/// `mu_init` is what "sitting on the band" meant here. One that does not
/// move at all when `mu_init` moves three decades is not near one, and
/// the cause was the Tikhonov `δ` in the multiplier initializer rather
/// than gh#702's compensated sum. The tape route now agrees with the
/// compensated one bit-for-bit on the objective, where gh#706 recorded
/// three different values across the three routes above.
///
/// So the assertion below is single-valued again. If it starts failing on
/// one platform's CI leg only, that is gh#706 returning and the two-valued
/// pin was load-bearing after all — say so in the issue rather than
/// widening the `matches!` back out, which is how a defect becomes a
/// shrug.
///
/// The absolute values pinned below are the ones the accept test reads
/// (`violation_initial`, `alpha`, `rejected_trials`, `termination`).
/// `violation_final` and `step_norm` are compared **between** the two
/// models only, never against a literal: cross-model equality is the
/// durable fact, and the last two digits of a reported violation are
/// the kind of number Q4 has already been shown to move.
#[test]
fn eigena2_and_eigenb2_hand_the_safeguard_identical_numbers() {
    let a = safeguard_decision("eigena2");
    let b = safeguard_decision("eigenb2");
    assert!(!a.is_empty(), "eigena2 attribution parsed to nothing");

    for field in [
        "violation_initial",
        "violation_final",
        "alpha",
        "step_norm",
        "rejected_trials",
        "termination",
    ] {
        assert!(
            a.contains_key(field),
            "eigena2 attribution has no `{field}`: {a:?}",
        );
        assert_eq!(
            a.get(field),
            b.get(field),
            "gh#616 rests on eigena2 and eigenb2 handing the safeguard \
             identical numbers, and `{field}` now differs. Re-derive the \
             conclusion in docs/src/initialization.md instead of editing \
             this list",
        );
    }

    // The inputs the accept test actually reads, as literals.
    let at = |k: &str| a.get(k).map(String::as_str);
    assert_eq!(at("violation_initial"), Some("1.0"));
    assert_eq!(at("alpha"), Some("0.5"));
    assert_eq!(at("rejected_trials"), Some("1"));
    assert_eq!(at("termination"), Some("accepted"));

    // Same decision, same verdict — and since gh#693, the same verdict
    // on both models on every platform. Pinned so that the pair moving
    // is a finding rather than a surprise; the premise above is what
    // gh#616 rests on, and these two used to be downstream round-off.
    //
    // Single-valued, and deliberately: this assertion was two-valued
    // while gh#706 was open because gh#702's compensated sum left the
    // model close enough to the band for the platform to pick a side.
    // gh#693 moved it off — 17 iterations, and `mu_init` across three
    // decades does not change a single iterate — so a two-valued pin
    // here would now assert less than is known and let a return to
    // `SolvedToAcceptableLevel` in 127 iterations pass unnoticed, which
    // is the failure mode this whole file exists to prevent.
    let eigena2 = solve("eigena2", true);
    assert_eq!(
        status_of(&eigena2),
        "SolveSucceeded",
        "eigena2 accepts the alpha = 0.5 step and, since gh#693, \
         converges cleanly from it — see the round-off screen in the \
         note above. A `SolvedToAcceptableLevel` here is gh#706 \
         returning, not a tolerance to widen the pin for.",
    );
    assert_eq!(
        status_of(&solve("eigenb2", true)),
        "SolveSucceeded",
        "see the test above — gh#616's originally measured downgrade, \
         removed by gh#693, with the round-off screen that says the old \
         pin was a 3-point island rather than a fact",
    );
    assert!(
        (solve("eigena2", true).solution.objective - 82.5).abs() < 1e-6,
        "eigena2 objective drifted",
    );
}

/// The measurement that makes the single-valued pin above safe, kept as
/// an assertion instead of as a paragraph: `eigena2`'s trajectory no
/// longer depends on the barrier parameter at all.
///
/// `mu_init` is what gh#706's round-off screen perturbed, and on gh#693's
/// parent a change of one part in `1e12` swung this model between 61 and
/// 127 iterations and between both statuses. Here it moves by three
/// decades — a thousandfold, not an ulp — and the run does not notice:
/// same iteration count, same objective to the bit. Only the printed
/// `lg(mu)` column differs.
///
/// A model whose steps are steered by the barrier parameter cannot
/// produce that. So a failure here says the pin above has stopped being
/// safe for the reason it was tightened — the status is a coin flip
/// again, whether or not it has yet landed on the wrong side of the band
/// on this platform.
#[test]
fn eigena2_no_longer_takes_its_trajectory_from_the_barrier_parameter() {
    let default_mu = solve_with("eigena2", true, &[], &[]);
    let big_mu = solve_with("eigena2", true, &[], &["mu_init=100.0"]);

    assert_eq!(
        default_mu.statistics.iteration_count, big_mu.statistics.iteration_count,
        "mu_init 0.1 -> 100 moved eigena2's iteration count, so the \
         barrier parameter steers this model again and gh#706's coin \
         flip is back within reach",
    );
    assert_eq!(
        default_mu.solution.objective.to_bits(),
        big_mu.solution.objective.to_bits(),
        "mu_init 0.1 -> 100 moved eigena2's objective ({} vs {}); the \
         iterates are supposed to be identical, not merely close",
        default_mu.solution.objective,
        big_mu.solution.objective,
    );
}

/// `csfi2` is not a case the accept test could ever have rescued: the
/// safeguard **declines** there — all four trials are worse than
/// `theta_0 = 1508.554...` — so `least_square_init_primal=yes` gives
/// back exactly what `=no` gives, to the bit.
///
/// That is the whole reason no tightening reaches `csfi2`: a tighter
/// test still declines, and the only thing that reaches its old
/// `SolveSucceeded` is accepting a step that makes the violation worse,
/// which is what gh#605 exists to prevent.
#[test]
fn csfi2_declines_the_step_so_the_option_changes_nothing_there() {
    let on = solve("csfi2", true);
    let off = solve("csfi2", false);
    assert_eq!(
        status_of(&on),
        status_of(&off),
        "csfi2 declines every trial, so the two routes must agree",
    );
    assert_eq!(
        on.statistics.iteration_count, off.statistics.iteration_count,
        "csfi2 declines every trial, so the two routes must take the \
         same trajectory",
    );
    assert_eq!(
        on.solution.objective, off.solution.objective,
        "csfi2 declines every trial, so the two routes must reach the \
         same objective",
    );
}

/// But "declined" is **not** "never asked", and that surprises people.
///
/// Declining restores the user's `x` exactly. It does not restore the
/// solver's state: computing the direction already drove the first
/// factorization through the augmented-system solver, on the `W = 0`
/// least-square matrix rather than on the first real KKT matrix.
/// gh#616 isolated this by forcing a decline on either side of that
/// call — declining *before* it is bit-identical to
/// `least_square_init_primal=no` everywhere, declining *after* it is
/// bit-identical to the real safeguard.
///
/// `pooling_rt2stp` is where it shows, as a large iteration difference
/// in the *declined* route's favour. This test pins that the two routes
/// differ, not by how much — the direction is the mechanism claim, the
/// magnitude is a platform detail.
///
/// **This test used to also assert the two routes reach the same local
/// optimum, and that was never a property of this model.** gh#693 made
/// it fail — on the default draw the declined route reaches −4391.826
/// and the `=no` route −3273.955 — which prompted measuring it properly
/// rather than re-pinning it. Across 17 values of `mu_init` at
/// `0.1·(1 ± k·1e-12)` the two routes agree on the optimum at **10 of 17
/// points on `main`** and 8 of 17 here. It was a coin flip that happened
/// to land heads at one draw.
///
/// The repository had in fact already recorded both sides of that flip
/// as fact in different places: `docs/src/initialization.md` still
/// carried gh#616's original measurement, where the two routes reached
/// *different* optima (−4391.826 against −3273.955), while this test
/// asserted they reached the same one. Both were written from a single
/// draw of a bistable nonconvex model, and neither noticed the other.
/// The doc now says so too.
///
/// So the same-optimum assertion is gone rather than inverted, and what
/// is left is the part that survives the screen: the iteration counts
/// differ at **17 of 17 points on both builds**. That is the mechanism
/// claim, and it is the one this test was written to make.
#[test]
fn a_declined_step_is_not_the_same_as_never_asking() {
    let on = solve("pooling_rt2stp", true);
    let off = solve("pooling_rt2stp", false);

    // Only the declined route's status is pinned. The `=no` route is
    // *not*: under the same round-off screen it gives 10 `SolveSucceeded`,
    // 3 `SolvedToAcceptableLevel` and 4 `ErrorInStepComputation` across 17
    // draws, so asserting one of them here would pin a 59% outcome — the
    // same mistake the objective assertion made.
    assert_eq!(
        status_of(&on),
        "SolveSucceeded",
        "the declined route is the stable one on this model (17/17 under \
         the header's round-off screen); if it has started scattering, \
         re-run that screen before treating this as a regression",
    );
    assert_ne!(
        on.statistics.iteration_count, off.statistics.iteration_count,
        "pooling_rt2stp declines every trial, yet the trajectories are \
         known to differ — the augmented-system solve that computed the \
         rejected direction leaves the linear solver's state seeded \
         differently. If these ever agree, the carry-over described in \
         docs/src/initialization.md is gone and that section is stale",
    );
}

/// `unbounded_cubic` starts feasible (`theta_0 = 0`), so the safeguard
/// short-circuits before computing any direction: no violation can be
/// improved on zero. The unsafeguarded path took a step anyway and
/// diverged in 91 iterations against 290 here.
///
/// This is the third, independent arm of the safeguard — neither
/// `csfi2`'s decline nor `eigenb2`'s backtrack — and the issue asked
/// specifically whether the three share a mechanism. They do not. The
/// status is `DivergingIterates` either way: the model is unbounded and
/// the verdict is right on both routes.
#[test]
fn a_feasible_start_short_circuits_and_stays_diverging() {
    let on = solve("unbounded_cubic", true);
    let off = solve("unbounded_cubic", false);
    assert_eq!(
        status_of(&on),
        "DivergingIterates",
        "unbounded_cubic is unbounded; the least-square route must not \
         change that verdict",
    );
    assert_eq!(status_of(&off), "DivergingIterates");
    // theta_0 = 0 means the block returns before the augmented-system
    // solve, so unlike the declined case above this really is a no-op.
    assert_eq!(
        on.statistics.iteration_count, off.statistics.iteration_count,
        "a feasible start short-circuits before the augmented-system \
         solve, so the two routes must be identical",
    );
}

/// The headline that keeps the decision honest: turning the option on
/// does not lose a model. The solved-or-acceptable set is the same size
/// on both routes — gh#605 moved two fixtures between the two solved
/// statuses, it did not drop any.
#[test]
fn no_fixture_stops_solving_when_the_option_is_turned_on() {
    // `deb7` is deliberately absent; see
    // `deb7_is_route_sensitive_in_ipopt_too` below for why it cannot
    // carry this assertion.
    for model in [
        "csfi2",
        "eigena2",
        "eigenb2",
        "pooling_rt2stp",
        "hs71_obj1e8",
        "user_scaling_suffix",
    ] {
        let on = status_of(&solve(model, true));
        let off = status_of(&solve(model, false));
        let solved = |s: &str| s == "SolveSucceeded" || s == "SolvedToAcceptableLevel";
        assert_eq!(
            solved(&on),
            solved(&off),
            "{model}: solved-or-acceptable must not depend on \
             least_square_init_primal (on = {on}, off = {off})",
        );
    }
}

/// `deb7` is the one fixture that cannot carry the invariant above, and
/// the reason is not a POUNCE defect: **reference Ipopt does not satisfy
/// it either.**
///
/// Ipopt 3.14.19, `deb7.nl`, one innocuous option perturbed at a time,
/// solved-or-acceptable on each route:
///
/// | perturbation | `least_square_init_primal=yes` | `=no` |
/// |---|---|---|
/// | `max_soc=0` | Solved To Acceptable Level | Error in step computation |
/// | `tol=1e-6`  | Error in step computation  | Optimal Solution Found    |
///
/// Two of eight sampled perturbations flip Ipopt's own verdict with the
/// option. `deb7` sits on a knife edge here, and which side a binary
/// lands on is set by last-bit arithmetic rather than by anything the
/// option means. POUNCE reproduces that sensitivity: on x86-64, with
/// `feral_refine` off (gh#735 measured it that way; it is not the
/// exact-Hessian default, which is `yes` — gh#909), the `yes` route reaches
/// `ErrorInStepComputation` at it=358 where `no` succeeds at it=131 —
/// while on aarch64 both succeed. `main` is not immune, it is merely on
/// the other side: perturbing `obj_scaling_factor` or `max_soc` flips
/// `main`'s verdict on this fixture and leaves gh#735's unchanged. MA57
/// flips on the same two.
///
/// So this test pins what `deb7` actually guarantees — it solves on the
/// default route — instead of an invariant no implementation holds.
#[test]
fn deb7_is_route_sensitive_in_ipopt_too() {
    let off = status_of(&solve("deb7", false));
    assert!(
        off == "SolveSucceeded" || off == "SolvedToAcceptableLevel",
        "deb7 must still solve on the default route (got {off})",
    );
}
