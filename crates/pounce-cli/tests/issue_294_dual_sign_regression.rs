//! Dual-SIGN regression guard for the `.sol` / JSON surfaces (issue #294).
//!
//! ## Why this test exists (the #271 post-mortem)
//!
//! A constraint-dual sign inversion (#271/#272, fixed in #287) flipped every
//! AMPL/Pyomo/GAMS marginal for an unknown span of releases and NO automated
//! check caught it, because the two guards that structurally *could* have —
//! don't:
//!
//!   * the benchmark suite (`benchmarks/benchmark_report.py`) compares
//!     objectives / status / iterations / wall-time and NEVER duals, so a
//!     defect that leaves primals and objectives exact (which a uniform sign
//!     flip does) is invisible to it by construction; and
//!   * `pounce verify` (`crates/pounce-cli/src/verify.rs`) evaluates KKT
//!     stationarity for BOTH `+λ` and `−λ` and keeps the better residual, so
//!     it certifies either sign and cannot guard the convention.
//!
//! The key principle: **agreement between pounce's own surfaces is not a
//! guard** — a uniform flip satisfies it. This test therefore pins each dual
//! surface against an EXTERNAL/ANALYTIC reference with an explicit expected
//! SIGN, so a future uniform flip fails loudly instead of shipping silently.
//!
//! Analytic references used here (both hand-computable in closed form):
//!
//!   * `convex_qp.nl`: `min x0² + x1²  s.t.  x0 + x1 = 2`, optimum (1, 1).
//!     obj*(b) = b²/2 for the RHS `b`, so the AMPL marginal `d obj / d b = b = 2`.
//!     The internal Lagrange multiplier of `L = f + λ(x0+x1−b)` is `λ = −2`
//!     (2·x0 + λ = 0). So: `.sol` marginal = **+2**, JSON `lambda` = **−2**,
//!     and the two must be exact negations.
//!   * `wyndor_min.nl`: the Wyndor Glass Co. LP as a pure minimize,
//!     `min −3x1 − 5x2  s.t.  x1≤4, 2x2≤12, 3x1+2x2≤18, x≥0`, optimum (2, 6).
//!     Textbook shadow prices are (0, 1.5, 1); with the `L = f + Σλᵢ(gᵢ−hᵢ)`,
//!     `λᵢ ≥ 0` convention the KKT multipliers are `λ = [0, 1.5, 1]`, so the
//!     AMPL `.sol` marginals `−λ = [0, −1.5, −1]` and JSON `lambda = [0, 1.5, 1]`.
//!     (The pyomo/GAMS max-vs-min conventions are pinned against IPOPT and
//!     analytically in the Python tests: `python/tests/test_dual_sign_regression.py`
//!     and `python/tests/test_gams_link.py`.)

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

/// Solve `fixture_name` via the CLI, returning the `.sol` marginals and the
/// JSON `solution.lambda`. Duals are read straight from the emitted `.sol`
/// (the AMPL `d obj / d b` marginal convention) and the JSON report (the
/// internal Lagrange-multiplier convention), so both dual-bearing CLI
/// surfaces are checked from one solve.
fn solve_duals(fixture_name: &str, tag: &str) -> (Vec<f64>, Vec<f64>) {
    let dir = std::env::temp_dir().join(format!("pounce_i294_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let nl = dir.join("m.nl");
    std::fs::copy(fixture(fixture_name), &nl).expect("copy fixture");
    let sol = dir.join("m.sol");
    let json = dir.join("m.json");

    let out = Command::new(pounce_exe())
        .arg(&nl)
        .arg("solver_selection=nlp")
        .arg("--sol-output")
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .output()
        .expect("spawn pounce");
    assert_eq!(
        out.status.code(),
        Some(0),
        "solve should succeed for {fixture_name}; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr),
    );

    let sol_text = std::fs::read_to_string(&sol).expect("read .sol");
    let sol_duals = parse_sol_marginals(&sol_text);

    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).expect("read json")).expect("parse");
    let json_lambda = report["solution"]["lambda"]
        .as_array()
        .expect("lambda array")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect();

    (sol_duals, json_lambda)
}

/// Parse the dual (marginal) block out of an AMPL `.sol` file.
///
/// Layout after the free-text banner: a blank line, `Options`, the option
/// count `nopts` and its `nopts` value lines, then the four-integer count
/// block `<n_dual> <m> <n_primal> <n>`, then `n_dual` dual value lines, then
/// the primal block. We consume exactly `n_dual` duals so the parser does not
/// mistake a primal value for a marginal.
fn parse_sol_marginals(text: &str) -> Vec<f64> {
    let mut lines = text.lines();
    // Advance to the "Options" line.
    for line in lines.by_ref() {
        if line.trim() == "Options" {
            break;
        }
    }
    let next_int = |lines: &mut std::str::Lines| -> usize {
        lines
            .next()
            .expect("truncated .sol")
            .trim()
            .parse::<usize>()
            .expect("expected an integer count line in .sol")
    };
    let nopts = next_int(&mut lines);
    for _ in 0..nopts {
        lines.next().expect("truncated option block");
    }
    let n_dual = next_int(&mut lines);
    let _m = next_int(&mut lines);
    let _n_primal = next_int(&mut lines);
    let _n = next_int(&mut lines);

    (0..n_dual)
        .map(|_| {
            lines
                .next()
                .expect("truncated dual block")
                .trim()
                .parse::<f64>()
                .expect("dual value should parse as f64")
        })
        .collect()
}

/// `convex_qp.nl`: single equality `x0 + x1 = 2`.
///
/// Analytic references: `.sol` marginal `d obj / d b = +2`; JSON internal
/// Lagrange multiplier `lambda = −2`. The explicit signs are what make this a
/// guard rather than a self-consistency check — a uniform flip of either
/// surface flips these and fails the exact-value asserts below.
#[test]
fn equality_qp_sol_marginal_and_json_lambda_have_the_right_sign() {
    let (sol_duals, json_lambda) = solve_duals("convex_qp.nl", "eq");

    assert_eq!(sol_duals.len(), 1, ".sol should carry one marginal");
    assert!(
        (sol_duals[0] - 2.0).abs() < 1e-5,
        "AMPL .sol marginal must be d obj/d b = +2 (NOT −2); got {}",
        sol_duals[0],
    );

    assert_eq!(json_lambda.len(), 1, "JSON should carry one multiplier");
    assert!(
        (json_lambda[0] - (-2.0)).abs() < 1e-5,
        "JSON solution.lambda must be the internal Lagrange multiplier −2 \
         (NOT +2); got {}",
        json_lambda[0],
    );

    // The two conventions differ by exactly a sign: marginal = −lambda. A
    // uniform flip that negated BOTH would still satisfy this relation, so it
    // is not the guard on its own — the exact-value asserts above are — but it
    // pins the writer's negation step (nl_writer.rs, the `−v` at the heart of
    // #271) against silent drift.
    assert!(
        (sol_duals[0] + json_lambda[0]).abs() < 1e-9,
        ".sol marginal ({}) must equal −(JSON lambda) ({})",
        sol_duals[0],
        json_lambda[0],
    );
}

/// `wyndor_min.nl`: the Wyndor Glass LP (pure minimize) with two active
/// inequality constraints, textbook shadow prices (0, 1.5, 1).
///
/// Expected signs (analytic): JSON internal multipliers `lambda = [0, 1.5, 1]`
/// (≥ 0, the `L = f + Σλᵢ(gᵢ−hᵢ)` convention), and AMPL `.sol` marginals
/// `−lambda = [0, −1.5, −1]`.
#[test]
fn wyndor_lp_active_inequality_marginals_have_the_right_sign() {
    let (sol_duals, json_lambda) = solve_duals("wyndor_min.nl", "wyndor");

    assert_eq!(json_lambda.len(), 3);
    let expect_lambda = [0.0, 1.5, 1.0];
    for (i, (&got, &want)) in json_lambda.iter().zip(&expect_lambda).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "JSON lambda[{i}] must be {want} (≥ 0 Lagrange convention); got {got}",
        );
    }

    assert_eq!(sol_duals.len(), 3);
    let expect_marginal = [0.0, -1.5, -1.0];
    for (i, (&got, &want)) in sol_duals.iter().zip(&expect_marginal).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "AMPL .sol marginal[{i}] must be {want} (= −lambda for a minimize \
             model); got {got}",
        );
    }
}
