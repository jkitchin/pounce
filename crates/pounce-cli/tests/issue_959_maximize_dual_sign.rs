//! gh#959 — on a `maximize` model the NLP arm returned constraint duals with
//! the **opposite** sign from Ipopt, from the analytic shadow prices, and from
//! POUNCE's own convex arms on the *same* `.nl`.
//!
//! ## Why the sign has to be put back, and where
//!
//! `NlTnlp` negates a maximize objective (`eval_f` / `eval_grad_f` read
//! `prob.minimize`), so everything downstream of the reader solves `min -f`
//! and the multipliers it produces belong to *that* Lagrangian. Upstream
//! carries the sense back out in one place —
//! `AmplTNLP::finalize_solution`, where `obj_sign == -1` selects
//! `lambda_sol = +lambda`, `z_L_sol = -z_L`, `z_U_sol = +z_U` against the
//! minimize branch's `-lambda`, `+z_L`, `-z_U`. POUNCE's convex arms do the
//! same through `qp_extract::recover_duals`'s `sign` and the `ipopt_zL_out`
//! block of `run_convex_qp`. The NLP arm did not, so one POUNCE binary gave
//! two answers for one file.
//!
//! ## What each test is measured against
//!
//! This is #294's file one model over, and the discipline is the same: agreement
//! between POUNCE's own surfaces is **not** a guard, because a uniform flip
//! satisfies it. So every expected value here is external or analytic:
//!
//! * `wyndor_max.nl` — the Wyndor Glass LP (Hillier & Lieberman §3.1) written
//!   as `max 3x₀ + 5x₁`. Textbook shadow prices `d obj*/d rhs = (0, 1.5, 1)`,
//!   and Ipopt 3.14.19 writes exactly that through the identical `.sol` path.
//! * `wyndor_max_boxed.nl` — the same model with `x₀ ∈ [0, 1.5]`, so the
//!   optimum is `(1.5, 6)` with the **variable bound active**: `c₂` prices at
//!   `2.5` and `x₀`'s reduced cost is `∂f/∂x₀ = 3`. That is the arm that
//!   covers `ipopt_zL_out` / `ipopt_zU_out`, which `wyndor_max.nl` cannot —
//!   its optimum is interior in `x` and every bound multiplier there is zero,
//!   so a sign error on them is invisible on that model.
//! * The minimize twins (`wyndor_min.nl`, `wyndor_min_boxed.nl`) are asserted
//!   to be **unmoved**, at the negation of the maximize values. Without them
//!   this file would pass on a build that flipped every model's duals.
//!
//! The cross-arm agreement check (`nlp` against `auto`) is included last and
//! deliberately framed as what it is: the *contradiction* the issue reported,
//! not the evidence that either arm is right.

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

/// What one CLI solve of `fixture_name` reports about duals.
struct Duals {
    /// The `.sol` dual block — AMPL marginals, `d obj / d rhs`.
    sol: Vec<f64>,
    /// `solution.lambda` from the JSON report — internal multipliers.
    json: Vec<f64>,
    /// The `ipopt_zL_out` / `ipopt_zU_out` suffix blocks, as sparse
    /// `(index, value)` pairs exactly as the `.sol` carries them.
    z_l: Vec<(usize, f64)>,
    z_u: Vec<(usize, f64)>,
}

fn solve(fixture_name: &str, selection: &str, tag: &str) -> Duals {
    let dir = std::env::temp_dir().join(format!("pounce_i959_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let nl = dir.join("m.nl");
    std::fs::copy(fixture(fixture_name), &nl).expect("copy fixture");
    let sol = dir.join("m.sol");
    let json = dir.join("m.json");

    let out = Command::new(pounce_exe())
        .arg(&nl)
        .arg(format!("solver_selection={selection}"))
        .arg("--sol-output")
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .output()
        .expect("spawn pounce");
    assert_eq!(
        out.status.code(),
        Some(0),
        "solve should succeed for {fixture_name} ({selection}); stderr:\n{}",
        String::from_utf8_lossy(&out.stderr),
    );

    let sol_text = std::fs::read_to_string(&sol).expect("read .sol");
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).expect("read json")).expect("parse");
    let json_lambda = report["solution"]["lambda"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_f64().expect("f64")).collect())
        .unwrap_or_default();

    let (marginals, rest) = parse_sol_marginals(&sol_text);
    Duals {
        sol: marginals,
        json: json_lambda,
        z_l: parse_suffix(rest, "ipopt_zL_out"),
        z_u: parse_suffix(rest, "ipopt_zU_out"),
    }
}

/// Parse the dual (marginal) block out of an AMPL `.sol`, returning it plus
/// the tail of the file so the suffix blocks can be read from the same text.
/// Layout: banner, blank line, `Options`, the option count and its values,
/// the four-integer count block `<n_dual> <m> <n_primal> <n>`, then `n_dual`
/// dual lines. Consuming exactly `n_dual` keeps a primal value from being read
/// as a marginal.
fn parse_sol_marginals(text: &str) -> (Vec<f64>, &str) {
    let mut lines = text.lines();
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

    let duals = (0..n_dual)
        .map(|_| {
            lines
                .next()
                .expect("truncated dual block")
                .trim()
                .parse::<f64>()
                .expect("dual value should parse as f64")
        })
        .collect();
    (duals, text)
}

/// Read a named `.sol` suffix block as `(index, value)` pairs. The writer
/// emits `suffix <kind> <count> ...`, the name on its own line, then `count`
/// `index value` lines; zero entries are trimmed on write, so an absent index
/// means a zero multiplier.
fn parse_suffix(text: &str, name: &str) -> Vec<(usize, f64)> {
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if !line.trim_start().starts_with("suffix ") {
            continue;
        }
        let count: usize = line
            .split_whitespace()
            .nth(2)
            .and_then(|v| v.parse().ok())
            .expect("suffix header carries a count");
        let this = lines.next().expect("suffix header is followed by a name");
        let mut entries = Vec::new();
        for _ in 0..count {
            let row = lines.next().expect("truncated suffix block");
            let mut it = row.split_whitespace();
            let i: usize = it.next().expect("index").parse().expect("index parses");
            let v: f64 = it.next().expect("value").parse().expect("value parses");
            entries.push((i, v));
        }
        if this.trim() == name {
            return entries;
        }
    }
    Vec::new()
}

fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "{what}[{i}] must be {w}; got {g} (full: {got:?})",
        );
    }
}

fn value_at(entries: &[(usize, f64)], index: usize) -> f64 {
    entries
        .iter()
        .find(|(i, _)| *i == index)
        .map(|(_, v)| *v)
        .unwrap_or(0.0)
}

/// The reported case. `max 3x₀ + 5x₁` subject to `x₀ ≤ 4`, `2x₁ ≤ 12`,
/// `3x₀ + 2x₁ ≤ 18`, `x ≥ 0`. Optimum `(2, 6)`, `obj* = 36`, textbook shadow
/// prices `(0, 1.5, 1)` — which is what Ipopt writes into the `.sol` dual
/// block for this file, and what POUNCE's NLP arm wrote negated.
#[test]
fn a_maximize_lp_reports_ipopts_marginals_on_the_nlp_arm() {
    let d = solve("wyndor_max.nl", "nlp", "max_nlp");
    assert_close(&d.sol, &[0.0, 1.5, 1.0], "maximize .sol marginal");
    // The writer negates `mult_g` on the way out, so the JSON's internal
    // multipliers are the marginals' negation. Pinned explicitly: a build that
    // negated both surfaces would still satisfy `marginal = -lambda`.
    assert_close(&d.json, &[0.0, -1.5, -1.0], "maximize JSON lambda");
}

/// The minimize twin, unmoved. `min -3x₀ - 5x₁` over the same feasible set has
/// the same primal optimum and the opposite marginals — this is #294's own
/// assertion, restated here so a future uniform flip cannot pass this file by
/// flipping everything.
#[test]
fn the_minimize_twin_is_unmoved() {
    let d = solve("wyndor_min.nl", "nlp", "min_nlp");
    assert_close(&d.sol, &[0.0, -1.5, -1.0], "minimize .sol marginal");
    assert_close(&d.json, &[0.0, 1.5, 1.0], "minimize JSON lambda");
}

/// The bound-multiplier arm, which `wyndor_max.nl` cannot reach: its optimum
/// is interior in `x`, so every `ipopt_zL_out` / `ipopt_zU_out` entry there is
/// zero and a sign error on them leaves no trace. Box `x₀ ∈ [0, 1.5]` makes
/// the upper bound active at the optimum `(1.5, 6)`, `obj* = 34.5`.
///
/// Analytic: `c₂` (`2x₁ ≤ 12`) prices at `2.5`, `c₁` and `c₃` are slack, and
/// the reduced cost of `x₀` is `∂f/∂x₀ = 3` — upstream's output convention
/// is that both bound blocks carry the objective-gradient component at the
/// bound, so `ipopt_zU_out[0] = +3` for the maximize model and `-3` for the
/// minimize one.
#[test]
fn a_maximize_model_with_an_active_bound_reports_the_reduced_cost_upstreams_way() {
    let max = solve("wyndor_max_boxed.nl", "nlp", "maxbox_nlp");
    assert_close(&max.sol, &[0.0, 2.5, 0.0], "boxed maximize .sol marginal");
    assert!(
        (value_at(&max.z_u, 0) - 3.0).abs() < 1e-4,
        "ipopt_zU_out[0] must be +3 = ∂f/∂x₀ for the maximize model; got {}",
        value_at(&max.z_u, 0),
    );
    assert!(
        value_at(&max.z_l, 0).abs() < 1e-4,
        "x₀ sits at its UPPER bound, so its lower-bound multiplier is 0; got {}",
        value_at(&max.z_l, 0),
    );

    let min = solve("wyndor_min_boxed.nl", "nlp", "minbox_nlp");
    assert_close(&min.sol, &[0.0, -2.5, 0.0], "boxed minimize .sol marginal");
    assert!(
        (value_at(&min.z_u, 0) - (-3.0)).abs() < 1e-4,
        "ipopt_zU_out[0] must be -3 = ∂f/∂x₀ for the minimize model; got {}",
        value_at(&min.z_u, 0),
    );
}

/// The internal contradiction the issue led with: `solver_selection=nlp` and
/// `solver_selection=auto` (which routes this LP to the convex arm) disagreed
/// in sign on one file, so at most one of them could be right whatever the
/// convention.
///
/// This is the weakest assertion in the file on its own — a uniform flip of
/// both arms satisfies it — and it is here because the *contradiction* is what
/// a user sees. The analytic pins above are what say which way to resolve it.
#[test]
fn the_nlp_and_convex_arms_agree_on_one_file() {
    for fx in ["wyndor_max.nl", "wyndor_max_boxed.nl"] {
        let nlp = solve(fx, "nlp", "agree_nlp");
        let auto = solve(fx, "auto", "agree_auto");
        assert_close(&nlp.sol, &auto.sol, &format!("{fx}: nlp vs auto .sol"));
        assert_close(&nlp.json, &auto.json, &format!("{fx}: nlp vs auto lambda"));
    }
}
