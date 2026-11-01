//! gh#1114 — the conic QCQP arm's multipliers, end to end through the CLI.
//!
//! `cvx-qcqp` solves a convex QCQP as an SOCP and maps the cone duals back
//! to the original constraints. The mapping was exact only at exact
//! complementarity, and the point only `~√gap`-accurate, so the duals that
//! reached a `.sol` (Pyomo's `m.dual`) missed stationarity by `~1e-5` while
//! the JSON report claimed `final_dual_inf ~1e-11`. `qcqp_polish` now
//! re-fits them and Newton-polishes the point in the original problem.
//!
//! These run the real CLI on **default options** — `solver_selection=auto`,
//! convex presolve on, the default `bound_relax_factor` — so they cover the
//! `main.rs` wiring the unit tests in `qcqp_polish.rs` cannot: the
//! presolve/postsolve path, the residual swap in the JSON report, the
//! `.sol` duals and the `ipopt_zL_out` bound-multiplier suffix.
//!
//! Every assertion is against a closed form. The first model fails on the
//! parent commit (duals `(1.000006, −1.2e-5)`); the scaled ones failed on
//! the first version of the polish (PR #1120 review: the `x1 ≥ 0`
//! multiplier, `2` against the ball's `1e8`, fell under a multiplier-ratio
//! activity test and `final_dual_inf` read `9.56`).

use pounce_cli::solve_report::SolveReport;
use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

struct Run {
    report: SolveReport,
    sol: String,
    stderr: String,
}

fn solve(name: &str) -> Run {
    let dir = std::env::temp_dir().join(format!("pounce_i1114_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let sol = dir.join("m.sol");
    let json = dir.join("m.json");
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fixture(name))
        .arg("--sol-output")
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .arg("print_level=0")
        .output()
        .expect("spawn pounce");
    assert_eq!(
        out.status.code(),
        Some(0),
        "{name}: solve should succeed; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Run {
        report: serde_json::from_str(&std::fs::read_to_string(&json).expect("read json"))
            .expect("deserialize SolveReport"),
        sol: std::fs::read_to_string(&sol).expect("read .sol"),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    };
    let _ = std::fs::remove_dir_all(&dir);
    run
}

/// The dual block of an AMPL `.sol`: after `Options`, its count and values,
/// the four counts `<n_dual> <m> <n_primal> <n>`, then `n_dual` values.
fn sol_duals(text: &str) -> Vec<f64> {
    let mut rest = text.lines().skip_while(|l| l.trim() != "Options").skip(1);
    let mut int = || -> usize { rest.next().expect(".sol").trim().parse().expect("count") };
    let nopts = int();
    for _ in 0..nopts {
        int();
    }
    let n_dual = int();
    for _ in 0..3 {
        int();
    }
    (0..n_dual)
        .map(|_| {
            rest.next()
                .expect("dual")
                .trim()
                .parse()
                .expect("dual value")
        })
        .collect()
}

/// A real-valued variable suffix block's `(index, value)` entries.
fn sol_suffix(text: &str, name: &str) -> Vec<(usize, f64)> {
    let lines: Vec<&str> = text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let mut it = line.split_whitespace();
        if it.next() != Some("suffix") || lines.get(i + 1).map(|l| l.trim()) != Some(name) {
            continue;
        }
        let _kind = it.next();
        let nvalues: usize = it.next().expect("nvalues").parse().expect("nvalues");
        return lines[i + 2..i + 2 + nvalues]
            .iter()
            .map(|l| {
                let mut p = l.split_whitespace();
                (
                    p.next().unwrap().parse().unwrap(),
                    p.next().unwrap().parse().unwrap(),
                )
            })
            .collect();
    }
    Vec::new()
}

fn assert_close(what: &str, got: f64, want: f64, tol: f64) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, expected {want} (±{tol})"
    );
}

/// The issue's model: `min (x0−2)² + x1²  s.t.  x0² + x1² ≤ 1,
/// x0 + x1 = 1`. LICQ holds at `x* = (1, 0)`, so the multipliers are unique:
/// `μ = 1` on the ball, `ν = 0` on the line.
#[test]
fn issue_model_duals_are_exact_in_sol_and_json() {
    let r = solve("issue_1114_qcqp_equality_row.nl");
    let s = &r.report.solution;
    assert_eq!(
        s.engine, "cvx-qcqp",
        "routing changed; this test pins the conic arm"
    );
    assert_close("x0", s.x[0], 1.0, 1e-10);
    assert_close("x1", s.x[1], 0.0, 1e-10);
    // The `.sol` carries AMPL's sign convention, the negation of the JSON
    // report's (Ipopt's `.sol` on this model reads `−0.99999998501`).
    let duals: Vec<f64> = sol_duals(&r.sol).iter().map(|d| -d).collect();
    for (src, l) in [("json", &s.lambda), (".sol (negated)", &duals)] {
        assert_close(&format!("{src} ball μ"), l[0], 1.0, 1e-8);
        assert_close(&format!("{src} line ν"), l[1], 0.0, 1e-8);
    }
    // The report describes the returned pair: recompute ∇ₓL from it.
    let (x, l) = (&s.x, &s.lambda);
    let grad = [
        2.0 * (x[0] - 2.0) + l[0] * 2.0 * x[0] + l[1],
        2.0 * x[1] + l[0] * 2.0 * x[1] + l[1],
    ];
    let r_inf = grad[0].abs().max(grad[1].abs());
    assert!(r_inf < 1e-10, "|∇L|∞ of the returned pair = {r_inf:.2e}");
    assert!(
        r.report.statistics.final_dual_inf < 1e-10,
        "reported final_dual_inf = {:.2e}",
        r.report.statistics.final_dual_inf
    );
}

/// `min 1e8·(x0−2)² + (x1+1)²  s.t.  ‖x‖² ≤ 1, x1 ≥ 0` as a linear row.
/// `x* = (1, 0)`, with `μ = 1e8` on the ball and `2` on `x1 ≥ 0` — eight
/// orders below the ball's.
#[test]
fn scaled_multipliers_row() {
    let r = solve("issue_1114_scaled_multipliers_row.nl");
    let s = &r.report.solution;
    assert_eq!(s.engine, "cvx-qcqp");
    assert_close("x1", s.x[1], 0.0, 1e-9);
    assert_close("ball μ", s.lambda[0], 1e8, 1e-2);
    // A `≥` row's multiplier is ≤ 0 in the JSON convention; the `.sol` is
    // its negation.
    let duals: Vec<f64> = sol_duals(&r.sol).iter().map(|d| -d).collect();
    for (src, l) in [("json", &s.lambda), (".sol (negated)", &duals)] {
        assert_close(&format!("{src} x1 ≥ 0 row"), l[1], -2.0, 1e-3);
    }
    assert!(
        !r.stderr.contains("misses the original"),
        "polish should have converged:\n{}",
        r.stderr
    );
}

/// The same model with `x1 ≥ 0` as a **variable bound**: its multiplier is
/// the `ipopt_zL_out` suffix (`+z_l`, so `+2`), and the polish's Newton step
/// carries it as a bound row.
#[test]
fn scaled_multipliers_bound() {
    let r = solve("issue_1114_scaled_multipliers_bound.nl");
    let s = &r.report.solution;
    assert_eq!(s.engine, "cvx-qcqp");
    assert_close("x1", s.x[1], 0.0, 1e-7);
    assert_close("ball μ", s.lambda[0], 1e8, 1e-2);
    let zl = sol_suffix(&r.sol, "ipopt_zL_out");
    let z1 = zl
        .iter()
        .find(|(i, _)| *i == 1)
        .map(|e| e.1)
        .unwrap_or_else(|| panic!("ipopt_zL_out must carry x1; got {zl:?}"));
    assert_close("ipopt_zL_out[x1]", z1, 2.0, 1e-3);
    assert!(
        !r.stderr.contains("misses the original"),
        "polish should have converged:\n{}",
        r.stderr
    );
}
