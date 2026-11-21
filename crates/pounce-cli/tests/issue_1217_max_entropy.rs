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
//! Two defects, each pinned on the run that reaches it:
//!
//! * **A second-order correction cut to a non-step.** Failure 1's first
//!   trial overflows `exp(h/p)` to ~1e154; the SOC built on it was
//!   fraction-to-the-boundary cut to `alpha = 5e-157`, accepted, and its
//!   dual update took `inf_du` from 1 to 5e4.
//!   `failure1_first_step_is_not_a_degenerate_soc` pins the guard in
//!   `pounce_algorithm::line_search::backtracking`.
//! * **A non-finite KKT solution accepted as a step.** After restoration
//!   resets `y_d = 0`, `h` has no curvature and its only KKT entry is
//!   `exp(h/p) ~ 1e-186`; the factor is accepted and the back-solve returns
//!   inf/NaN, which iterative refinement could not reject (every comparison
//!   with a NaN residual ratio is false). The NaN step cycled through
//!   restoration to `Error_In_Step_Computation`.
//!   `no_iteration_takes_a_nan_step` pins the `δ_x` escalation in
//!   `pounce_algorithm::kkt::pd_full_space_solver`.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::return_codes::ApplicationReturnStatus;

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

/// Solve, returning the report and the console output (which prints the
/// iteration rows of every solve the second-opinion ladder runs, not only
/// the promoted one).
fn solve(nl: &PathBuf, extra: &[&str]) -> (SolveReport, String) {
    let json_path = tmp_path("report.json");
    let sol_path = tmp_path("out.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(nl)
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path)
        .arg("max_wall_time=300");
    for o in extra {
        cmd.arg(o);
    }
    let out = cmd.output().expect("spawn pounce");
    let console =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    let _ = std::fs::remove_file(nl);
    (
        serde_json::from_str(&text).expect("deserialize SolveReport"),
        console,
    )
}

/// The iteration rows of the console log: `iter[r] objective inf_pr inf_du
/// lg(mu) ||d|| ...`, split into fields.
fn iteration_rows(console: &str) -> Vec<Vec<String>> {
    console
        .lines()
        .map(|l| l.split_whitespace().map(str::to_owned).collect::<Vec<_>>())
        .filter(|f| {
            f.len() >= 10
                && f[0].trim_end_matches('r').parse::<usize>().is_ok()
                && f[1].parse::<f64>().is_ok()
        })
        .collect()
}

fn assert_solved(r: &SolveReport, want: f64, label: &str) {
    assert!(
        matches!(
            r.solution.status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "{label}: status {:?}",
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
    let (r, _) = solve(&write_nl(300.0, &cold()), &[]);
    assert_solved(&r, 6.65357129, "control");
}

#[test]
fn failure1_start_at_the_solution_solves() {
    let (r, _) = solve(&write_nl(300.0, &at_solution(300.0)), &[]);
    assert_solved(&r, 6.65357129, "failure 1");
}

#[test]
fn failure2_cold_start_at_mean_450_solves() {
    let (r, _) = solve(&write_nl(450.0, &cold()), &[]);
    // Ipopt + MA57 reports 6.89240414; the exact optimum is 6.89240663.
    assert_solved(&r, 6.8924054, "failure 2");
}

/// Before the fix iteration 1 of failure 1 was the SOC
/// `5.23e-157H` — a step the line search would never have tried itself.
#[test]
fn failure1_first_step_is_not_a_degenerate_soc() {
    let (_, console) = solve(
        &write_nl(300.0, &at_solution(300.0)),
        &["max_iter=1", "mu_strategy_fallback=no"],
    );
    let rows = iteration_rows(&console);
    let it1 = rows
        .iter()
        .find(|f| f[0] == "1")
        .unwrap_or_else(|| panic!("no iteration-1 row in:\n{console}"));
    let alpha: f64 = it1[8]
        .trim_end_matches(|c: char| c.is_ascii_alphabetic())
        .parse()
        .expect("alpha_pr column");
    assert!(
        alpha > 1e-12,
        "iteration 1 took alpha_pr = {alpha} ({}); a correction below the \
         line search's alpha_min must not be accepted",
        it1[8]
    );
}

/// Before the fix both runs printed NaN in the `||d||` column on every
/// outer iteration that left restoration (9 rows on failure 1, 14 on
/// failure 2, counted across all the ladder's solves).
#[test]
fn no_iteration_takes_a_nan_step() {
    for (label, mean, start) in [
        ("failure 1", 300.0, at_solution(300.0)),
        ("failure 2", 450.0, cold()),
    ] {
        let (_, console) = solve(&write_nl(mean, &start), &[]);
        let nan_rows: Vec<String> = iteration_rows(&console)
            .into_iter()
            .filter(|f| f[5].eq_ignore_ascii_case("nan") || f[5] == "inf")
            .map(|f| f.join(" "))
            .collect();
        assert!(
            nan_rows.is_empty(),
            "{label}: iterations took a non-finite step:\n{}",
            nan_rows.join("\n")
        );
    }
}
