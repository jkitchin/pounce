//! gh#1217 — a 2 000-variable maximum-entropy NLP that pounce 0.17.0
//! stopped on with `Error_In_Step_Computation`, where Ipopt + MA57 solves:
//!
//! ```text
//! max   sum_i h[i]
//! s.t.  sum_i p[i] = 1,   sum_i i * p[i] = mean
//!       1 - p[i] * exp(h[i] / p[i]) >= 0        i = 1..N
//!       p[i] >= 0,  h[i] free
//! ```
//!
//! * **Failure 1** — mean 300, started at the solution (every entropy row at
//!   its bound).
//! * **Failure 2** — mean 450, cold start `p = 0.001, h = 0`.
//!
//! The model is generated here, as the `.nl` Pyomo writes for the issue's
//! reproducer (variables `h` are 0..N-1, `p` are N..2N-1; each entropy row
//! is `-(p * exp(h / p)) >= -1`). It needs the issue's own `N = 1000`: at
//! `N <= 300` both runs solved before the fix as well.
//!
//! **Cause.** pounce followed Ipopt 3.14 line for line on failure 1 through
//! iteration 19 (checked against Ipopt + MUMPS, which also solves both runs).
//! At iteration 20 the step has `‖d‖ = 3.7e17`; Ipopt backtracks 75 times
//! and accepts `alpha = 5.2e-23`, because its only floor is the filter
//! acceptor's `CalculateAlphaMin`. pounce clamped that floor to `1e-12` and
//! capped a sweep at 50 trials, so it went to restoration instead and never
//! recovered. The line search now keeps the acceptor's `alpha_min` and a
//! trial budget that reaches it (`pounce_algorithm::line_search::
//! backtracking`), and reproduces Ipopt's 64 iterations. The `1e-12`
//! clamp still holds in the filter's switching region `θ ≤ θ_min` (gh#1226,
//! `pounce-algorithm/tests/issue_1226_nonsmooth_null_steps.rs`). Iteration 20
//! is at `θ = 4e3`, outside that region, so this file pins the other branch.
//!
//! The tests run with every second-opinion rung off: the claim is that the
//! *first* solve succeeds, as it does in Ipopt. The exception is failure 2,
//! whose first-solve outcome depends on platform rounding; see its test.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::IterPhase;

const N: usize = 1000;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn tmp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_issue1217_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

/// `p[i] = exp(-lam*(i-1)) / Z` with `lam` bisected for `mean` — the
/// issue's `exact()`.
fn exact(mean: f64) -> Vec<f64> {
    let weights = |lam: f64| -> Vec<f64> { (0..N).map(|i| (-lam * i as f64).exp()).collect() };
    let mean_of = |lam: f64| -> f64 {
        let w = weights(lam);
        let z: f64 = w.iter().sum();
        w.iter()
            .enumerate()
            .map(|(i, wi)| (i + 1) as f64 * wi)
            .sum::<f64>()
            / z
    };
    let (mut lo, mut hi) = (1e-12, 1.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mean_of(mid) > mean {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let w = weights(0.5 * (lo + hi));
    let z: f64 = w.iter().sum();
    w.iter().map(|wi| wi / z).collect()
}

/// The `.nl` for the model at `mean`, started at `start[i] = (p, h)`.
fn write_nl(mean: f64, start: &[(f64, f64)]) -> PathBuf {
    let mut s = String::new();
    let m = N + 2;
    let _ = writeln!(s, "g3 1 1 0\t# problem max_entropy");
    let _ = writeln!(
        s,
        " {} {} 1 0 2 \t# vars, constraints, objectives, ranges, eqns",
        2 * N,
        m
    );
    let _ = writeln!(
        s,
        " {N} 0 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb"
    );
    let _ = writeln!(s, " 0 0\t# network constraints: nonlinear, linear");
    let _ = writeln!(
        s,
        " {} 0 0 \t# nonlinear vars in constraints, objectives, both",
        2 * N
    );
    let _ = writeln!(
        s,
        " 0 0 0 1\t# linear network variables; functions; arith, flags"
    );
    let _ = writeln!(
        s,
        " 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)"
    );
    let _ = writeln!(s, " {} {N} \t# nonzeros in Jacobian, obj. gradient", 4 * N);
    let _ = writeln!(s, " 0 0\t# max name lengths: constraints, variables");
    let _ = writeln!(s, " 0 0 0 0 0\t# common exprs: b,c,o,c1,o1");
    // Entropy rows: -(p * exp(h / p)).
    for i in 0..N {
        let (h, p) = (i, N + i);
        let _ = write!(s, "C{i}\no16\no2\nv{p}\no44\no3\nv{h}\nv{p}\n");
    }
    let _ = write!(s, "C{N}\nn0\nC{}\nn0\n", N + 1);
    let _ = write!(s, "O0 1\nn0\n");
    let _ = writeln!(s, "x{}", 2 * N);
    for (i, &(p, h)) in start.iter().enumerate() {
        let _ = writeln!(s, "{i} {h:?}");
        let _ = writeln!(s, "{} {p:?}", N + i);
    }
    let _ = writeln!(s, "r");
    for _ in 0..N {
        let _ = writeln!(s, "2 -1");
    }
    let _ = writeln!(s, "4 1");
    let _ = writeln!(s, "4 {mean:?}");
    let _ = writeln!(s, "b");
    for _ in 0..N {
        let _ = writeln!(s, "3");
    }
    for _ in 0..N {
        let _ = writeln!(s, "2 0");
    }
    // Cumulative column counts: h[j] is in one row, p[j] in three.
    let _ = writeln!(s, "k{}", 2 * N - 1);
    for j in 0..2 * N - 1 {
        let c = if j < N { j + 1 } else { N + 3 * (j + 1 - N) };
        let _ = writeln!(s, "{c}");
    }
    for i in 0..N {
        let _ = write!(s, "J{i} 2\n{i} 0\n{} 0\n", N + i);
    }
    let _ = writeln!(s, "J{N} {N}");
    for j in 0..N {
        let _ = writeln!(s, "{} 1", N + j);
    }
    let _ = writeln!(s, "J{} {N}", N + 1);
    for j in 0..N {
        let _ = writeln!(s, "{} {}", N + j, j + 1);
    }
    let _ = writeln!(s, "G0 {N}");
    for j in 0..N {
        let _ = writeln!(s, "{j} 1");
    }
    let path = tmp_path("model.nl");
    std::fs::write(&path, s).expect("write .nl");
    path
}

fn cold() -> Vec<(f64, f64)> {
    vec![(0.001, 0.0); N]
}

fn at_solution(mean: f64) -> Vec<(f64, f64)> {
    exact(mean).into_iter().map(|q| (q, -q * q.ln())).collect()
}

/// Every second-opinion rung off, so a success is the first solve's own.
const NO_LADDER: &[&str] = &[
    "mu_strategy_fallback=no",
    "infeasibility_mu_strategy_retry=no",
    "feral_infeasibility_scaling_retry=no",
    "feral_increase_quality_retry=no",
    "infeasibility_saddle_escape_retry=no",
    "infeasibility_presolve_retry=no",
    "infeasibility_perturbed_start_retry=no",
    "dual_divergence_retry=no",
    "square_system_recovery=no",
];

/// Solve with the full per-iteration history in the report, every
/// second-opinion rung off.
fn solve(nl: &PathBuf, extra: &[&str]) -> SolveReport {
    solve_with(nl, NO_LADDER, extra)
}

/// Solve under the default options — the second-opinion ladder on.
fn solve_default(nl: &PathBuf) -> SolveReport {
    solve_with(nl, &[], &[])
}

fn solve_with(nl: &PathBuf, base: &[&str], extra: &[&str]) -> SolveReport {
    let json_path = tmp_path("report.json");
    let sol_path = tmp_path("out.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(nl)
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path)
        .arg("--json-detail")
        .arg("full")
        .arg("max_wall_time=300");
    for o in base.iter().chain(extra) {
        cmd.arg(o);
    }
    let _ = cmd.output().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    let _ = std::fs::remove_file(nl);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

fn assert_solved(r: &SolveReport, want: f64, label: &str) {
    assert!(
        matches!(
            r.solution.status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "{label}: solve ended {:?}",
        r.solution.status
    );
    let obj = r.solution.objective;
    assert!(
        (obj - want).abs() <= 1e-5,
        "{label}: objective {obj}, want {want}"
    );
}

#[test]
fn control_cold_start_at_mean_300_solves() {
    let r = solve(&write_nl(300.0, &cold()), &[]);
    assert_solved(&r, 6.65357129, "control");
}

/// Failure 2 no longer stops on `Error_In_Step_Computation`, and the
/// default solve reaches the optimum.
///
/// What the *first* solve then reports is not pinned, because it depends on
/// last-bit rounding. From cold, iteration 12 accepts a full step that
/// takes `inf_pr` from 1e-13 to 362; two restoration phases and steps with
/// `‖d‖` up to 1e45 follow. Built from the same source, aarch64 Linux and
/// aarch64 macOS print identical rows through iteration 61, differing
/// only in the last digits of the `inf_pr` round-off, and part at 62:
///
/// | platform        | first solve                           |
/// |-----------------|---------------------------------------|
/// | x86-64 Linux    | `Solve_Succeeded`, 99 (from gh#1218) |
/// | aarch64 Linux   | `Solve_Succeeded`, 124 iterations     |
/// | aarch64 macOS   | `Infeasible_Problem_Detected`, 346    |
///
/// So asserting the first solve's status made this test pass or fail by
/// platform. What holds everywhere: the first solve ends with a verdict the
/// ladder can act on (before gh#1217's fix it ended
/// `Error_In_Step_Computation` on every platform, which nothing re-solves),
/// and the default solve ends at the optimum — on macOS through the
/// `mu_strategy=adaptive` rung, in 61 iterations.
#[test]
fn failure2_cold_start_at_mean_450_reaches_the_optimum() {
    let first = solve(&write_nl(450.0, &cold()), &[]);
    assert!(
        !matches!(
            first.solution.status,
            ApplicationReturnStatus::ErrorInStepComputation
        ),
        "failure 2: first solve ended ErrorInStepComputation again"
    );
    let r = solve_default(&write_nl(450.0, &cold()));
    // Ipopt + MA57 reports 6.89240414; the exact optimum is 6.89240663.
    assert_solved(&r, 6.8924054, "failure 2");
}

/// Failure 1 solves on the first solve, and through the branch the fix is
/// about: a step accepted below the old `1e-12` clamp, after more trials
/// than the old 50-trial cap allowed, with no restoration anywhere in the
/// solve — Ipopt's iteration 20, `5.20e-23f 75`. Before the fix that
/// iteration was `1.79e-12R 40` and restoration ran thirteen times. One
/// solve serves both checks, since the 2000-variable model is the
/// expensive part.
#[test]
fn failure1_solves_through_a_step_below_the_old_alpha_floor() {
    let r = solve(&write_nl(300.0, &at_solution(300.0)), &[]);
    assert_solved(&r, 6.65357129, "failure 1");
    let main: Vec<_> = r
        .iterations
        .iter()
        .filter(|it| it.phase == IterPhase::Main)
        .collect();
    let deep = main
        .iter()
        // A backtracked step, `f` or `h`: iteration 1 is a second-order
        // correction taken at `5.2e-157` (`H`), which Ipopt takes too and
        // which the floor never governed.
        .find(|it| {
            matches!(it.alpha_primal_char, 'f' | 'h')
                && it.alpha_primal > 0.0
                && it.alpha_primal < 1e-12
        })
        .unwrap_or_else(|| panic!("no backtracked step accepted below alpha = 1e-12"));
    assert!(
        deep.ls_trials > 50,
        "iteration {} accepted alpha = {:e} after {} trials; Ipopt takes 75",
        deep.iter,
        deep.alpha_primal,
        deep.ls_trials
    );
    assert!(
        r.iterations
            .iter()
            .all(|it| it.phase != IterPhase::Restoration && it.alpha_primal_char != 'R'),
        "the solve entered restoration; Ipopt's does not"
    );
}

// ---------------------------------------------------------------------------
// The gh#1000 square-certificate counter, away from the float tie.
//
// `issue_508_infeasible_gap_1em4` reaches its verdict through this counter
// only because its least violation lands exactly on `constr_viol_tol = 1e-4`,
// which this PR made inclusive (`reaches_tolerance`). That fixture tests the
// tie, not the rule. The same model with a gap of `3e-4` sits clearly above
// the tolerance and reaches the counter on its first solve before and after
// the change (557 iterations, exact Hessian), so it pins the counter in both
// directions without depending on how the violation rounds.

/// `min (x-5)^2 s.t. x^2 + gap = 0`: the gh#508 probe with its gap replaced.
fn gap_model(gap: &str) -> PathBuf {
    let mut src = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src.push("tests/fixtures/issue_508_infeasible_gap_1em4.nl");
    let text = std::fs::read_to_string(&src).expect("read gh#508 fixture");
    assert!(text.contains("\n4 -0.0001\n"), "fixture's bound line moved");
    let path = tmp_path("gap.nl");
    std::fs::write(
        &path,
        text.replace("\n4 -0.0001\n", &format!("\n4 -{gap}\n")),
    )
    .expect("write gap model");
    path
}

/// Console output of one first solve, with the restoration debug log on.
fn solve_console(nl: &PathBuf, extra: &[&str]) -> String {
    let sol_path = tmp_path("gap.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.env("RUST_LOG", "pounce::restoration=debug")
        .arg(nl)
        .arg(&sol_path)
        .arg("max_wall_time=300");
    for o in NO_LADDER.iter().chain(extra) {
        cmd.arg(o);
    }
    let out = cmd.output().expect("spawn pounce");
    let _ = std::fs::remove_file(&sol_path);
    let _ = std::fs::remove_file(nl);
    String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr)
}

#[test]
fn square_certificate_above_constr_viol_tol_is_honoured() {
    let console = solve_console(&gap_model("0.0003"), &[]);
    assert!(
        console.contains("[PN_SQUARE_CERTIFICATE]"),
        "a 3e-4 violation is above constr_viol_tol = 1e-4; the counter must \
         honour the repeated certificate:\n{console}"
    );
    assert!(
        console.contains("EXIT: Converged to a point of local infeasibility"),
        "expected the infeasibility verdict:\n{console}"
    );
}

#[test]
fn square_certificate_inside_constr_viol_tol_is_not_counted() {
    let console = solve_console(&gap_model("0.0003"), &["constr_viol_tol=1e-3"]);
    assert!(
        !console.contains("[PN_SQUARE_CERTIFICATE]"),
        "a 3e-4 violation is inside constr_viol_tol = 1e-3; no certificate \
         may be honoured:\n{console}"
    );
    assert!(
        !console.contains("local infeasibility"),
        "a violation the user declared feasible must not be claimed \
         infeasible:\n{console}"
    );
}
