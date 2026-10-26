//! What the second-opinion ladder and the degeneracy banners actually *do*,
//! end to end, through the CLI.
//!
//! `pounce-algorithm/tests/init_options_wiring.rs` proves the three `*_retry`
//! tags are read; `second_opinion.rs` and `second_opinion_driver.rs` pin the
//! rung table and the driver loop against hand-built option lists. None of
//! that runs a solve. This file is the missing half: a real model, a real
//! degenerate start, and the ladder either recovering it or not.
//!
//! Both fixtures are HS008 (`x² + y² = 25`, `x·y = 9`) started at the origin,
//! where **both** Jacobian rows are identically zero — LICQ fails and the
//! filter line search has no descent direction to find. That is the failure
//! mode `start_point_perturbation` exists for, and the one the KRONOS corpus
//! is full of (a squared slack at zero, an origin start on a homogeneous
//! quadratic). The second fixture flips the first constraint to
//! `x² + y² = -25`, which no real point satisfies, so the model is *both*
//! degenerate at its start and genuinely infeasible — the case where the
//! ladder must spend all three rungs and still keep its verdict.

use std::path::PathBuf;
use std::process::Command;

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

/// Returns (stdout+stderr, `.sol` text).
fn run(model: &str, tag: &str, extra: &[&str]) -> (String, String) {
    let sol = std::env::temp_dir().join(format!("pounce_degen_{tag}.sol"));
    let _ = std::fs::remove_file(&sol);
    let out = Command::new(pounce_exe())
        .arg(fixture(model))
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
        .args(extra)
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "-AMPL must exit 0");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let sol_text = std::fs::read_to_string(&sol).expect("read .sol");
    (log, sol_text)
}

fn objno(sol: &str) -> i32 {
    for line in sol.lines() {
        if let Some(rest) = line.trim().strip_prefix("objno ") {
            if let Some(code) = rest.split_whitespace().nth(1) {
                return code.parse().expect("objno code parses");
            }
        }
    }
    panic!("no `objno` line in .sol:\n{sol}");
}

/// The primal `x` the `.sol` carries, in file order.
fn primal(sol: &str, n: usize) -> Vec<f64> {
    // The AMPL `.sol` body is the primal block: `n` bare numbers, ahead of
    // the `objno` trailer.
    sol.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with("objno") && l.parse::<f64>().is_ok())
        .filter_map(|l| l.parse::<f64>().ok())
        .rev()
        .take(n)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

/// The headline claim: a model upstream IPOPT calls locally infeasible is
/// solved, by moving the starting point and nothing else.
#[test]
fn rung_three_recovers_a_model_that_is_only_degenerate() {
    let (log, sol) = run("degenerate_start_hs008.nl", "hs008", &[]);

    assert!(
        log.contains("start_point_perturbation=1e-2 re-solve recovered the problem"),
        "rung 3 must promote:\n{log}",
    );
    // 0 == `Solve_Succeeded` in the AMPL solve_result_num bands.
    assert_eq!(objno(&sol), 0, "{sol}");

    // And the answer is the real HS008 optimum, not merely a success status:
    // both constraints hold at the point the `.sol` ships.
    let x = primal(&sol, 2);
    assert_eq!(x.len(), 2, "{sol}");
    assert!(
        (x[0] * x[0] + x[1] * x[1] - 25.0).abs() < 1e-6,
        "x²+y²=25 violated at {x:?}",
    );
    assert!((x[0] * x[1] - 9.0).abs() < 1e-6, "xy=9 violated at {x:?}");
}

/// The counterpart, and the reason the ladder is not simply "retry until it
/// works": on a model that really has no solution all three rungs are spent
/// and the verdict stands. A ladder that promoted here would be laundering
/// failures.
#[test]
fn a_genuinely_infeasible_model_survives_all_three_rungs() {
    let (log, sol) = run("degenerate_start_infeasible.nl", "infeas", &[]);

    for rung in [
        "feral_scaling=mc64",
        "mu_strategy=adaptive",
        "start_point_perturbation=1e-2",
    ] {
        assert!(
            log.contains(&format!("{rung} re-solve did not recover")),
            "rung `{rung}` must run and fail:\n{log}",
        );
    }
    // The model is square (2×2) with a constant objective, so the
    // square-system stage runs after the three rungs. The start's Jacobian is
    // identically zero, so it stops at once on stationarity and the verdict
    // still stands — now having survived four attempts.
    assert!(
        log.contains("square_system_lm did not recover (Stationary after 0 iteration(s))"),
        "the square-system stage must run and decline honestly:\n{log}",
    );
    assert!(
        log.contains(
            "keeping the original Infeasible_Problem_Detected verdict; it survived 4 \
             independent re-solve(s)"
        ),
        "the verdict must be kept, and say how many re-solves it survived:\n{log}",
    );
    // 200..=299 is the AMPL "infeasible" band.
    let code = objno(&sol);
    assert!(
        (200..300).contains(&code),
        "expected infeasible band, got {code}"
    );
}

/// The degeneracy banners, which are the *diagnosis* rather than the recovery:
/// they name why the verdict is not evidence about the problem. They print on
/// the run that keeps its verdict, and they name both rows and both columns.
#[test]
fn the_diagnosis_names_the_rank_deficient_rows_and_columns() {
    let (log, _) = run("degenerate_start_infeasible.nl", "diag", &[]);
    assert!(
        log.contains(
            "the constraint Jacobian is rank-deficient there: 2 of 2 constraint rows have \
             an identically zero gradient here (rows 0, 1)"
        ),
        "{log}",
    );
    assert!(
        log.contains("2 of 2 variable columns are identically zero here (variables 0, 1)"),
        "{log}",
    );
    assert!(log.contains("LICQ fails at a point like that"), "{log}");
}

/// `print_level=0` is a request for silence, and the ladder honours it — the
/// rank diagnosis *and* the running narration of which rung is being tried.
/// Worth pinning because both go to stderr through `eprintln!` rather than
/// through the journalist, so nothing else enforces the level for them; the
/// narration was unconditional until it was gated, which meant `print_level=0`
/// still produced ten `pounce:` lines on this fixture.
///
/// The `print_level=1` leg is the positive control: without it this test would
/// keep passing if the ladder stopped running altogether.
#[test]
fn the_ladder_respects_print_level_zero() {
    let (quiet, _) = run(
        "degenerate_start_infeasible.nl",
        "quiet",
        &["print_level=0"],
    );
    for line in [
        "rank-deficient there",
        "re-solving along 3 different trajectories",
        "keeping the original",
    ] {
        assert!(
            !quiet.contains(line),
            "print_level=0 must silence `{line}`:\n{quiet}",
        );
    }

    let (loud, _) = run("degenerate_start_infeasible.nl", "loud", &["print_level=1"]);
    for line in [
        "rank-deficient there",
        "re-solving along 3 different trajectories",
        "keeping the original",
    ] {
        assert!(
            loud.contains(line),
            "print_level=1 must print `{line}`:\n{loud}"
        );
    }
}

/// Turning the third rung off restores upstream IPOPT's behaviour on the
/// recoverable fixture: the verdict it would have shipped. This is what
/// `infeasibility_perturbed_start_retry=no` is *for*, and it is also the
/// mutation guard for the test above — if rung 3 were not the thing doing the
/// work, this assertion would not flip.
///
/// Rung 5 goes off too. HS008 from the origin ends with an exactly zero
/// constraint gradient, which is a saddle of the violation, so with rung 3 off
/// the saddle rung recovers it on its own (`Optimal`, restarting 256 probe
/// steps out); the upstream verdict needs both off.
#[test]
fn turning_the_start_rung_off_gives_the_upstream_verdict_back() {
    let (log, sol) = run(
        "degenerate_start_hs008.nl",
        "hs008_off",
        &[
            "infeasibility_perturbed_start_retry=no",
            "infeasibility_saddle_escape_retry=no",
        ],
    );
    assert!(
        !log.contains("start_point_perturbation=1e-2"),
        "rung 3 must not run:\n{log}",
    );
    let code = objno(&sol);
    assert!(
        (200..300).contains(&code),
        "without rung 3 this model is reported infeasible, got {code}",
    );
}
