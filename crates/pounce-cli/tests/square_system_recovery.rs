//! The square-system stage of the second-opinion ladder: when a square,
//! equality-only, constant-objective model's interior-point solve fails, and
//! every option rung that applies has failed too, POUNCE solves `c(x) = 0`
//! directly by bounded Levenberg–Marquardt on the constraint Jacobian
//! (`crates/pounce-algorithm/src/square_system.rs`).
//!
//! # The case
//!
//! `square_flowsheet_resto` (32×32, `f ≡ 0`, feasible by construction, its
//! Jacobian nearly singular even at the solution). Under
//! `hessian_approximation=limited-memory` on FERAL PR#222 the IPM runs to the
//! 3000-iteration cap and rung 4 fails as well; the LM solves it in a handful
//! of iterations. On the published FERAL 0.20 the same leg is rescued earlier,
//! by rung 4 (`feral_increase_quality=no`), so the LM never runs there by
//! default — the stage is last on purpose, so that every rescue the ladder
//! already had keeps its answer.
//!
//! # Why the recovery arm turns rung 4 off and caps `max_iter`
//!
//! To reach the stage on **both** FERAL versions with one assertion. With
//! `feral_increase_quality_retry=no` no option rung opens on a budget exit,
//! so the IPM's failure goes straight to the LM wherever it happens. The cap
//! at 500 is `issue857_escalation_gated_quality_rung`'s, for that file's
//! reason: the free-running lbfgs leg exits `Maximum_Iterations_Exceeded` on
//! macOS and `Infeasible_Problem_Detected` on linux, and the cap fires long
//! before the two part company.
//!
//! # The branches the gate takes, each with its own fixture
//!
//! A gate that is only ever seen opening says nothing about the conditions
//! that close it, so each closing condition has a model that *reaches the
//! ladder* (its solve fails) and differs from an opening one in exactly that
//! condition:
//!
//! * **non-square** — 3 free variables, 2 equality rows, constant objective,
//!   infeasible (written inline below);
//! * **non-constant objective** — `infeasible_square_scaled_1em4` and
//!   `infeasible_equalities`, square and equality-only with a real objective;
//! * **the IPM already solves it** — a square, constant-objective 2×2 model
//!   (inline) whose solve succeeds, so the ladder never opens;
//! * **an option rung promotes first** — `degenerate_start_hs008`, square
//!   with a constant objective, whose base solve fails and whose rung 3
//!   recovers it, so the LM never runs.
//!
//! The assertions read the JSON report's `second_opinion.tried`, which names
//! every stage that actually ran, so a gate that opened on a model it should
//! not have is red even where the LM would then have failed honestly.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

const LABEL: &str = "square_system_lm";

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

struct Run {
    out: String,
    report: serde_json::Value,
    sol: String,
}

impl Run {
    fn status(&self) -> &str {
        self.report["solution"]["status"].as_str().unwrap_or("?")
    }
    fn tried(&self) -> Vec<String> {
        self.report["second_opinion"]["tried"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }
    fn promoted_by(&self) -> Option<&str> {
        self.report["second_opinion"]["promoted_by"].as_str()
    }
    fn engine(&self) -> &str {
        self.report["solution"]["engine"].as_str().unwrap_or("")
    }
    fn vec(&self, key: &str) -> Vec<f64> {
        self.report["solution"][key]
            .as_array()
            .unwrap_or_else(|| panic!("solution.{key} in report:\n{}", self.out))
            .iter()
            .map(|v| v.as_f64().expect("number"))
            .collect()
    }
}

fn solve_path(model: &std::path::Path, extra: &[&str]) -> Run {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let stem = model.file_stem().unwrap().to_string_lossy().to_string();
    let sol = dir.join(format!("pounce_sqlm_{pid}_{seq}_{stem}.sol"));
    let json = dir.join(format!("pounce_sqlm_{pid}_{seq}_{stem}.json"));
    let out = Command::new(pounce_exe())
        .arg(model)
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .args(extra)
        .output()
        .expect("run pounce");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&json)
        .unwrap_or_else(|e| panic!("json report written ({e}):\n{out}"));
    let report = serde_json::from_str(&text).expect("valid json report");
    let sol_text = std::fs::read_to_string(&sol).unwrap_or_default();
    let _ = std::fs::remove_file(&sol);
    let _ = std::fs::remove_file(&json);
    Run {
        out,
        report,
        sol: sol_text,
    }
}

fn solve(model: &str, extra: &[&str]) -> Run {
    solve_path(&fixture(model), extra)
}

/// Write an inline `.nl` to the temp dir — kept out of `tests/fixtures/`, so
/// the sweep corpus is unchanged by this file.
fn inline_nl(name: &str, body: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("pounce_sqlm_{}_{name}.nl", std::process::id()));
    std::fs::write(&p, body).expect("write inline .nl");
    p
}

/// An independent evaluation of `square_flowsheet_resto`'s 32 rows, from
/// `dev-notes/square_flowsheet_resto_gen.py` (each row is
/// `expr(x) − expr(x*)`), in the `.nl`'s variable order: `F[0..8]`,
/// `y[8..16]`, `P[16..24]`, `T[24..32]`. Nothing here goes through the
/// solver's `.nl` reader or expression evaluator.
fn flowsheet_residual(x: &[f64]) -> Vec<f64> {
    const NS: usize = 8;
    let star = |i: usize| {
        let i = i as f64;
        (
            3.0e6 - 1.0e4 * i,
            30.0 + 0.7 * i,
            1.0e-6 * (1.0 + 0.3 * i),
            600.0 + 5.0 * i,
        )
    };
    let rows = |p: &dyn Fn(usize) -> f64,
                f: &dyn Fn(usize) -> f64,
                y: &dyn Fn(usize) -> f64,
                t: &dyn Fn(usize) -> f64| {
        let mut r = Vec::with_capacity(4 * NS);
        for i in 0..NS {
            let q = (i + NS - 1) % NS;
            r.push(f(i) - 0.97 * f(q) - 0.5 * y(q) * f(q));
        }
        for i in 0..NS {
            let q = (i + NS - 1) % NS;
            r.push(p(i) - p(q) + 1.5e3 * (f(i) / 30.0).powi(2));
        }
        for i in 0..NS {
            r.push(y(i) * p(i) - 1.0e-3 * (-6000.0 / t(i)).exp() * f(i));
        }
        for i in 0..NS {
            let q = (i + NS - 1) % NS;
            r.push(
                75.0 * f(i) * (t(i) - 298.15) - 75.0 * f(q) * (t(q) - 298.15) + 2.2e5 * y(i) * f(i),
            );
        }
        r
    };
    let at = rows(&|i| x[16 + i], &|i| x[i], &|i| x[8 + i], &|i| x[24 + i]);
    let rhs = rows(&|i| star(i).0, &|i| star(i).1, &|i| star(i).2, &|i| {
        star(i).3
    });
    at.iter().zip(&rhs).map(|(a, b)| a - b).collect()
}

// `mu_strategy_fallback=no`: since gh#1076 the base solve's failure is
// platform-dependent — `Restoration_Failed`/250 on arm64,
// `Maximum_Iterations_Exceeded`/500 on x86_64 — and on x86_64 the monotone
// μ re-solve then recovers the model in 155, ahead of the stage. These tests
// are about the stage, so the fallback is kept out of their way.
const FAILING_LBFGS: &[&str] = &[
    "hessian_approximation=limited-memory",
    "feral_increase_quality_retry=no",
    "mu_strategy_fallback=no",
    "max_iter=500",
];

/// The recovery: the IPM fails, no option rung opens, the LM solves it, and
/// the answer holds up under an evaluation that shares no code with the
/// solver.
#[test]
fn lbfgs_square_flowsheet_is_recovered_by_the_square_system_stage() {
    let r = solve("square_flowsheet_resto.nl", FAILING_LBFGS);
    assert_eq!(r.status(), "SolveSucceeded", "{}", r.out);
    assert_eq!(r.promoted_by(), Some(LABEL), "{}", r.out);
    assert_eq!(r.engine(), "square-lm", "the point is LM's, not the IPM's");
    // gh#1076: the base solve used to run to `max_iter`, which opens no
    // option rung. With a zero pivot read under a reused MC64 scaling now
    // refactored under a fresh one, it fails in restoration at 250 instead
    // (arm64), and `Restoration_Failed` opens the start-point rung first; that
    // rung fails too (62 iterations) and the stage is what recovers it, as
    // before. Whatever opens ahead of it, the stage must be the last rung and
    // the one promoted.
    let tried = r.tried();
    assert_eq!(
        tried.last().map(String::as_str),
        Some(LABEL),
        "the square-system stage must be the rung that recovers this:\n{}",
        r.out
    );
    assert!(
        tried.len() <= 2,
        "at most one option rung ahead of the stage, got {tried:?}:\n{}",
        r.out
    );
    let base = r.report["second_opinion"]["base_status"]
        .as_str()
        .unwrap_or("?");
    assert!(
        matches!(
            base,
            "Maximum_Iterations_Exceeded" | "Infeasible_Problem_Detected" | "Restoration_Failed"
        ),
        "the stage recovers a failed IPM solve, not a converged one: base {base}"
    );
    assert!(
        r.out.contains("EXIT: Optimal Solution Found."),
        "the console verdict is the promoted one:\n{}",
        r.out
    );

    let x = r.vec("x");
    assert_eq!(x.len(), 32);
    let c = flowsheet_residual(&x);
    let worst = c.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    // constr_viol_tol is 1e-4 and tol 1e-8 against gradient-scaled rows;
    // the unscaled 1e-8 is stricter than either and the LM reaches ~3e-10.
    assert!(worst <= 1e-8, "independent max|c| = {worst:e}");

    // Declared box, exactly: F ∈ [1e-3, 1e4], y ∈ [1e-12, 1], P ∈ [1e5, 1e7],
    // T ∈ [300, 900].
    let boxes = [(1e-3, 1e4), (1e-12, 1.0), (1e5, 1e7), (300.0, 900.0)];
    for (k, &v) in x.iter().enumerate() {
        let (lo, hi) = boxes[k / 8];
        assert!(lo <= v && v <= hi, "x[{k}] = {v} outside [{lo}, {hi}]");
    }
    // The multipliers the stage documents: zero, which at ∇f = 0 satisfy
    // stationarity exactly.
    assert!(
        r.vec("lambda").iter().all(|&l| l == 0.0),
        "λ reported as zero"
    );

    // The `.sol` carries the same point the report does (the CLI's capture
    // is overwritten for this promotion — see `square_system_promoted`).
    let tail: Vec<f64> = r
        .sol
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    assert!(tail.len() >= 64, ".sol has a dual and a primal block");
    let xs = &tail[tail.len() - 32..];
    let worst_sol = flowsheet_residual(xs)
        .iter()
        .fold(0.0f64, |a, v| a.max(v.abs()));
    assert!(worst_sol <= 1e-8, ".sol max|c| = {worst_sol:e}");
}

/// The opt-out restores the IPM's verdict untouched.
#[test]
fn the_stage_can_be_switched_off() {
    let mut opts = FAILING_LBFGS.to_vec();
    opts.push("square_system_recovery=no");
    let r = solve("square_flowsheet_resto.nl", &opts);
    assert_ne!(r.status(), "SolveSucceeded", "{}", r.out);
    assert!(!r.tried().iter().any(|t| t == LABEL), "{:?}", r.tried());
}

/// Non-square: 3 free variables, 2 equality rows, constant objective,
/// infeasible (`x0² + x1² + x2² = −1`, `x0 = 0`), so its solve fails and the
/// ladder opens — and the square gate alone keeps the stage out.
#[test]
fn does_not_fire_on_a_non_square_model() {
    let nl = inline_nl(
        "nonsquare",
        "g3 1 1 0\t# 3 vars, 2 equality rows, constant objective, infeasible\n \
         3 2 1 0 2\n 1 0\n 0 0\n 3 0 0\n 0 0 0 1\n 0 0 0 0 0\n 4 0\n 0 0\n 0 0 0 0 0\n\
         C0\no54\n3\no5\nv0\nn2\no5\nv1\nn2\no5\nv2\nn2\n\
         C1\nn0\n\
         O0 0\nn0\n\
         x3\n0 0.5\n1 0.5\n2 0.5\n\
         r\n4 -1\n4 0\n\
         b\n3\n3\n3\n\
         k2\n2\n3\n\
         J0 3\n0 0\n1 0\n2 0\n\
         J1 1\n0 1\n",
    );
    let r = solve_path(&nl, &[]);
    let _ = std::fs::remove_file(&nl);
    assert_ne!(r.status(), "SolveSucceeded", "{}", r.out);
    assert!(
        !r.tried().is_empty(),
        "the ladder must open, or this test says nothing about the gate:\n{}",
        r.out
    );
    assert!(
        !r.tried().iter().any(|t| t == LABEL),
        "{:?}\n{}",
        r.tried(),
        r.out
    );
}

/// Square and equality-only, but with a real objective: the ladder opens on
/// each model and the constant-objective gate keeps the stage out.
///
/// The gate has two halves and each needs its own model. Both fixtures have
/// quadratic objectives, which the `.nl` degree recognizer *proves* varying,
/// so they are stopped by the `derivative_proofs` half before anything is
/// evaluated — measured: with the evaluated half disabled they stay green.
/// The inline model's objective is `exp(x0)`, which the recognizer declines
/// (`Unknown`), so only the evaluated `∇f(x₀) = 0` test can keep it out.
#[test]
fn does_not_fire_on_a_non_constant_objective() {
    let exp_obj = inline_nl(
        "exp_objective",
        "g3 1 1 0\t# square 2x2 infeasible rows, objective exp(x0)\n \
         2 2 1 0 2\n 1 1\n 0 0\n 2 1 1\n 0 0 0 1\n 0 0 0 0 0\n 4 1\n 0 0\n 0 0 0 0 0\n\
         C0\no0\no5\nv0\nn2\no5\nv1\nn2\n\
         C1\nn0\n\
         O0 0\no44\nv0\n\
         x2\n0 0.5\n1 0.5\n\
         r\n4 -1\n4 0\n\
         b\n3\n3\n\
         k1\n2\n\
         J0 2\n0 0\n1 0\n\
         J1 2\n0 1\n1 -1\n\
         G0 1\n0 0\n",
    );
    let models = [
        fixture("infeasible_square_scaled_1em4.nl"),
        fixture("infeasible_equalities.nl"),
        exp_obj.clone(),
    ];
    for model in &models {
        let r = solve_path(model, &[]);
        let name = model.display();
        assert!(
            !r.tried().is_empty(),
            "{name}: ladder should open:\n{}",
            r.out
        );
        assert!(
            !r.tried().iter().any(|t| t == LABEL),
            "{name}: {:?}\n{}",
            r.tried(),
            r.out
        );
    }
    let _ = std::fs::remove_file(&exp_obj);
}

/// Square with a constant objective, and the IPM solves it: no ladder, no
/// stage, the IPM's own answer (`x0·x1 = 2`, `x0 − x1 = 1` from `(1.5, 1.5)`).
#[test]
fn does_not_fire_when_the_ipm_already_solves_the_square_model() {
    let nl = inline_nl(
        "square_ok",
        "g3 1 1 0\t# square 2x2, constant objective, feasible\n \
         2 2 1 0 2\n 1 0\n 0 0\n 2 0 0\n 0 0 0 1\n 0 0 0 0 0\n 4 0\n 0 0\n 0 0 0 0 0\n\
         C0\no2\nv0\nv1\n\
         C1\nn0\n\
         O0 0\nn0\n\
         x2\n0 1.5\n1 1.5\n\
         r\n4 2\n4 1\n\
         b\n3\n3\n\
         k1\n2\n\
         J0 2\n0 0\n1 0\n\
         J1 2\n0 1\n1 -1\n",
    );
    for leg in [&[][..], &["hessian_approximation=limited-memory"][..]] {
        let r = solve_path(&nl, leg);
        assert_eq!(r.status(), "SolveSucceeded", "{leg:?}\n{}", r.out);
        assert!(
            r.report["second_opinion"].is_null(),
            "{leg:?}: no ladder\n{}",
            r.out
        );
        let x = r.vec("x");
        assert!((x[0] * x[1] - 2.0).abs() < 1e-7 && (x[0] - x[1] - 1.0).abs() < 1e-7);
    }
    let _ = std::fs::remove_file(&nl);
}

/// Square with a constant objective, the base solve fails, and an option
/// rung recovers it: the stage is last, so it never runs and the rung's
/// answer stands.
#[test]
fn an_option_rung_that_recovers_the_model_pre_empts_the_stage() {
    let r = solve("degenerate_start_hs008.nl", &[]);
    assert_eq!(r.status(), "SolveSucceeded", "{}", r.out);
    let promoted = r.promoted_by().unwrap_or("-");
    assert_ne!(promoted, LABEL, "{}", r.out);
    assert!(!r.tried().iter().any(|t| t == LABEL), "{:?}", r.tried());
    assert_eq!(r.engine(), "nlp", "a rung's answer is the IPM's");
}

// ---------------------------------------------------------------------------
// The stage's own branches, reached deterministically.
//
// `square_flowsheet_resto` reaches the stage only where the IPM fails, which
// depends on the FERAL version. The models below reach it on any build: the
// root sits **on** a bound (`x0 ≤ 2`), so the barrier cannot converge in the
// two iterations `max_iter=2` gives it, while the LM — which projects onto
// the box instead of staying interior — needs at most two. Each test names
// the branch it exists for and the mutation it was checked against: the
// branch was broken, the test was seen to fail, and the code was restored.
// ---------------------------------------------------------------------------

/// Three variables, the **middle** one fixed (`x1 = 3`, declared start 7),
/// `x0 ≤ 2` active at the root, rows `x0·x2 − x1 = 2` and
/// `x0 + x1/2 − x2 = 1`, objective the constant 5. Root `(2, 3, 2.5)`.
/// `x0`, `x2` are the starting values of the free variables.
fn fixed_mid_slot_nl(x0: f64, x2: f64) -> String {
    format!(
        "g3 1 1 0\t# fixed variable in the middle slot, root on a bound\n \
         3 2 1 0 2\n 1 0\n 0 0\n 3 0 0\n 0 0 0 1\n 0 0 0 0 0\n 6 0\n 0 0\n 0 0 0 0 0\n\
         C0\no2\nv0\nv2\n\
         C1\nn0\n\
         O0 0\nn5\n\
         x3\n0 {x0}\n1 7\n2 {x2}\n\
         r\n4 2\n4 1\n\
         b\n1 2\n4 3\n3\n\
         k2\n2\n4\n\
         J0 3\n0 0\n1 -1\n2 0\n\
         J1 3\n0 1\n1 0.5\n2 -1\n"
    )
}

/// The last `n` numeric lines of a `.sol` before its first `suffix` line —
/// the primal block, which follows the dual block.
fn sol_primal(sol: &str, n: usize) -> Vec<f64> {
    let body: Vec<f64> = sol
        .lines()
        .take_while(|l| !l.starts_with("suffix") && !l.starts_with("objno"))
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    body[body.len() - n..].to_vec()
}

/// `(name, number of nonzero entries)` of every `.sol` suffix block.
fn sol_suffix_counts(sol: &str) -> Vec<(String, usize)> {
    let lines: Vec<&str> = sol.lines().collect();
    let mut out = Vec::new();
    for (k, l) in lines.iter().enumerate() {
        if let Some(rest) = l.strip_prefix("suffix ") {
            let fields: Vec<usize> = rest
                .split_whitespace()
                .filter_map(|f| f.parse().ok())
                .collect();
            out.push((lines[k + 1].trim().to_string(), fields[1]));
        }
    }
    out
}

/// Full and reduced index spaces meet in the stage: the LM iterates over the
/// free variables and scatters into the full vector at every evaluation and
/// at `finalize_solution`. The fixed variable sits in the middle slot, where
/// a free-index-as-full-index slip puts a neighbour's value — plausible and
/// wrong (gh#450's shape). Checked in both the JSON report and the `.sol`.
///
/// Mutation-checked: `TnlpResidual::scatter` writing `x_full[k]` instead of
/// `x_full[j]` fails here (the LM then solves a different system and lands
/// nowhere near `(2, 3, 2.5)`).
#[test]
fn a_fixed_variable_keeps_its_slot_through_the_stage() {
    let nl = inline_nl("fixed_mid", &fixed_mid_slot_nl(1.5, 1.5));
    let r = solve_path(&nl, &["max_iter=2"]);
    let _ = std::fs::remove_file(&nl);
    assert_eq!(r.promoted_by(), Some(LABEL), "{}", r.out);
    assert_eq!(r.engine(), "square-lm");
    let want = [2.0, 3.0, 2.5];
    for (got, which) in [(r.vec("x"), "JSON"), (sol_primal(&r.sol, 3), ".sol")] {
        for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-12,
                "{which} x[{k}] = {g}, want {w}: {got:?}"
            );
        }
    }
}

/// The stage promotes `λ = 0, z = 0`, and the `.sol`'s bound-multiplier
/// suffixes must carry those zeros rather than the failed IPM iterate's `z`.
/// The control is the same model with the stage off, whose `.sol` carries
/// one nonzero entry in each block — so an empty block here is the stage's,
/// not a model with no bound multipliers.
///
/// Mutation-checked: finalizing with `z_l = z_u = 1` in `square_system::run`
/// fails here.
#[test]
fn a_promotion_writes_zero_bound_multiplier_suffixes() {
    let nl = inline_nl("fixed_mid_z", &fixed_mid_slot_nl(1.5, 1.5));
    let on = solve_path(&nl, &["max_iter=2"]);
    let off = solve_path(&nl, &["max_iter=2", "square_system_recovery=no"]);
    let _ = std::fs::remove_file(&nl);
    assert_eq!(on.promoted_by(), Some(LABEL), "{}", on.out);
    let blocks = |r: &Run| -> Vec<(String, usize)> {
        sol_suffix_counts(&r.sol)
            .into_iter()
            .filter(|(n, _)| n == "ipopt_zL_out" || n == "ipopt_zU_out")
            .collect()
    };
    let off_blocks = blocks(&off);
    assert_eq!(
        off_blocks.len(),
        2,
        "control writes both blocks:\n{}",
        off.sol
    );
    assert!(
        off_blocks.iter().all(|(_, k)| *k > 0),
        "the control's failed iterate has nonzero z: {off_blocks:?}"
    );
    assert_eq!(
        blocks(&on),
        [
            ("ipopt_zL_out".to_string(), 0),
            ("ipopt_zU_out".to_string(), 0)
        ],
        "{}",
        on.sol
    );
}

/// The start is already a root, but on the active bound, so the IPM's
/// `bound_push` moves it off and `max_iter=0` stops the IPM there. The
/// stage's start is projected, not pushed: it passes the test before any
/// step, and promotes with **zero** LM iterations — the iterate-0 path of
/// `bounded_levenberg_marquardt`, which no other test reaches end to end.
/// The record keeps the IPM's pairing: one iteration record, the start.
///
/// Mutation-checked: removing the iterate-0 `test.passes` in
/// `bounded_levenberg_marquardt` fails here (the loop's `max_iter` check
/// then fires first and the stage ends `IterationLimit`).
#[test]
fn a_start_that_is_already_a_root_promotes_with_zero_lm_iterations() {
    let nl = inline_nl("at_root", &fixed_mid_slot_nl(2.0, 2.5));
    let r = solve_path(&nl, &["max_iter=0", "--json-detail", "full"]);
    let _ = std::fs::remove_file(&nl);
    assert_eq!(
        r.report["second_opinion"]["base_status"].as_str(),
        Some("Maximum_Iterations_Exceeded"),
        "{}",
        r.out
    );
    assert_eq!(r.promoted_by(), Some(LABEL), "{}", r.out);
    assert_eq!(
        r.report["second_opinion"]["rung_iteration_counts"],
        serde_json::json!([0])
    );
    assert_eq!(r.report["statistics"]["iteration_count"], 0);
    let iters = r.report["iterations"].as_array().map(Vec::len).unwrap_or(0);
    assert_eq!(iters, 1, "one record, iterate 0:\n{}", r.out);
    assert_eq!(r.vec("x"), [2.0, 3.0, 2.5]);
}

/// A promoted verdict's statistics describe the LM in the IPM's
/// conventions: `iterations` holds `iteration_count + 1` records (the start
/// included) whenever iteration history is recorded, and
/// `final_scaled_objective` is `f` times the objective multiplier the IPM
/// would have used — here `obj_scaling_factor=2` on `f ≡ 5`. The IPM's own
/// report on the same model under the same option is the reference for both
/// conventions, so the test reads an outside number rather than restating
/// the code.
///
/// Mutation-checked: `iterations: Vec::new()` in the promoted statistics,
/// and `final_scaled_objective: obj`, each fail here.
#[test]
fn a_promotion_keeps_the_ipms_statistics_conventions() {
    let nl = inline_nl("stats", &fixed_mid_slot_nl(1.5, 1.5));
    let opts = ["obj_scaling_factor=2", "--json-detail", "full"];
    let lm = solve_path(&nl, &[&["max_iter=2"][..], &opts[..]].concat());
    let ipm = solve_path(&nl, &opts);
    let _ = std::fs::remove_file(&nl);
    assert_eq!(lm.promoted_by(), Some(LABEL), "{}", lm.out);
    assert_eq!(ipm.engine(), "nlp", "the reference is an IPM solve");
    for r in [&lm, &ipm] {
        let count = r.report["statistics"]["iteration_count"].as_u64().unwrap();
        let records = r.report["iterations"].as_array().map(Vec::len).unwrap_or(0);
        assert_eq!(records as u64, count + 1, "{}: {}", r.engine(), r.out);
        let st = &r.report["statistics"];
        assert_eq!(st["final_objective"], 5.0, "{}", r.engine());
        assert_eq!(st["final_scaled_objective"], 10.0, "{}", r.engine());
    }
}

/// The objective `exp((x0 − 1.5)²)` has `∇f = 0` at the start `x0 = 1.5`, so
/// the evaluated gate admits the model, and `∇f ≠ 0` at the root `x0 = 2`.
/// The LM converges; the certificate's `∇f(x̂) = 0` check is all that stands
/// between that and a `Solve_Succeeded` with multipliers that do not satisfy
/// stationarity. This is the post-LM rejection branch.
///
/// Mutation-checked: deleting the `eval_grad_f` check in
/// `square_system::run` promotes and fails here.
#[test]
fn a_nonzero_objective_gradient_at_the_root_is_rejected() {
    let nl = inline_nl("grad_at_root", FXG);
    let r = solve_path(&nl, &["max_iter=2"]);
    let _ = std::fs::remove_file(&nl);
    assert_eq!(r.tried(), [LABEL], "the stage must run:\n{}", r.out);
    assert_eq!(r.promoted_by(), None, "{}", r.out);
    assert_eq!(r.status(), "MaximumIterationsExceeded", "{}", r.out);
    assert!(
        r.out
            .contains("objective gradient is nonzero at the returned point"),
        "{}",
        r.out
    );
}

/// `fxg` from the review probes: the root on `x0 ≤ 2`, `x2` fixed, objective
/// `exp((x0 − 1.5)²)`.
const FXG: &str = "g3 1 1 0\t# 3 vars one fixed, objective flat only at the start\n \
     3 2 1 0 2\n 1 0\n 0 0\n 2 1 1\n 0 0 0 1\n 0 0 0 0 0\n 6 1\n 0 0\n 0 0 0 0 0\n\
     C0\no2\nv0\nv1\n\
     C1\nn0\n\
     O0 0\no44\no5\no1\nv0\nn1.5\nn2\n\
     x3\n0 1.5\n1 1.5\n2 7\n\
     r\n4 2\n4 1\n\
     b\n1 2\n3\n4 3\n\
     k2\n2\n4\n\
     J0 3\n0 0\n1 0\n2 -1\n\
     J1 3\n0 1\n1 -1\n2 0.5\n\
     G0 1\n0 0\n";

/// A stage that runs and does not promote must leave everything the user
/// reads exactly as `square_system_recovery=no` leaves it: the `.sol`
/// byte-for-byte, and the JSON statistics and linear-solver summary up to
/// wall-clock fields. Only `second_opinion` may differ — it records that the
/// stage ran.
///
/// Mutation-checked: calling `finalize_solution` on the `∇f`-rejection path
/// of `square_system::run` (so a rejected point still reaches the TNLP)
/// fails here on the `.sol`.
#[test]
fn a_stage_that_does_not_promote_changes_nothing_the_user_reads() {
    let nl = inline_nl("no_promote", FXG);
    let on = solve_path(&nl, &["max_iter=2"]);
    let off = solve_path(&nl, &["max_iter=2", "square_system_recovery=no"]);
    let _ = std::fs::remove_file(&nl);
    assert_eq!(on.tried(), [LABEL], "the stage must run:\n{}", on.out);
    assert!(off.tried().is_empty(), "{}", off.out);
    assert_eq!(on.sol, off.sol, ".sol must be byte-identical");
    fn untimed(v: &serde_json::Value) -> serde_json::Value {
        let mut v = v.clone();
        if let Some(o) = v.as_object_mut() {
            o.retain(|k, _| !k.ends_with("_secs") && !k.contains("time"));
        }
        v
    }
    for block in ["statistics", "linear_solver", "solution"] {
        assert_eq!(
            untimed(&on.report[block]),
            untimed(&off.report[block]),
            "{block} differs"
        );
    }
}

/// Gradient-based row scaling with a row of `‖∇c_0‖∞ = 4000`, its maximum
/// in the **fixed** column, and a start `x0 = 3` **outside** the box
/// `x0 ≤ 2` — the two places a reimplementation of the IPM's scaling can
/// quietly measure somewhere else (the free columns only; the projected
/// start). `a` is the fixed column's coefficient, `b` the curvature of row 1
/// in `x0` (`b = 0`: row 1 linear). `scaling` adds a constraint
/// `scaling_factor` suffix for the `user-scaling` arm.
fn row_scale_nl(a: f64, b: f64, scaling: Option<(f64, f64)>) -> String {
    let (c1, rhs1, j1) = if b != 0.0 {
        (
            format!("C1\no2\nn{b}\no5\nv0\nn2\n"),
            4.0 * b - 1.0,
            "J1 3\n0 0\n1 0.5\n2 -1\n",
        )
    } else {
        ("C1\nn0\n".to_string(), 1.0, "J1 3\n0 1\n1 0.5\n2 -1\n")
    };
    let mut s = format!(
        "g3 1 1 0\t# row 0 scaled by 1000, root on x0 <= 2, start outside the box\n \
         3 2 1 0 2\n {} 0\n 0 0\n 3 0 0\n 0 0 0 1\n 0 0 0 0 0\n 6 0\n 0 0\n 0 0 0 0 0\n\
         C0\no2\nn1000\no2\nv0\nv2\n{c1}O0 0\nn0\n\
         x3\n0 3\n1 7\n2 1.5\n\
         r\n4 {}\n4 {rhs1}\n\
         b\n1 2\n4 3\n3\n\
         k2\n2\n4\n\
         J0 3\n0 0\n1 {}\n2 0\n{j1}",
        if b != 0.0 { 2 } else { 1 },
        5000.0 - 3.0 * a,
        -a,
    );
    if let Some((s0, s1)) = scaling {
        s.push_str(&format!("S5 2 scaling_factor\n0 {s0}\n1 {s1}\n"));
    }
    s
}

/// The acceptance test's row scaling is the IPM's, measured against the
/// IPM: for each arm, the IPM's own scaled/unscaled violation ratio after
/// one iteration (row 0 dominates both, so the ratio is `s_0`) must equal
/// the stage's ratio at its iterate 0 (row 0 dominates there too, `c_0 =
/// −2000`). Neither number is restated from the code.
///
/// * `fixed_column` — row 0's maximum is the fixed variable's coefficient
///   (`s_0 = 100/4000`, not the free columns' `100/3000`);
/// * `projection` — row 1's `∂/∂x0 = 100·x0` and row 0's `1000·x0` are
///   sampled at the raw start `x0 = 3` (`s_0 = 1/30`), not the projected 2
///   (`1/20`);
/// * `equilibration-based` — which the IPM runs as gradient-based, and the
///   stage used to read as unscaled;
/// * `user-scaling` — the `.nl` suffix's `0.01` on row 0.
///
/// Mutation-checked, one per arm: summing the row maximum over the free
/// columns only fails `fixed_column`; sampling `J` at the projected start
/// fails `projection`; mapping `equilibration-based` to unscaled fails that
/// arm; and ignoring `g_scaling` under `user-scaling` fails the last.
#[test]
fn the_acceptance_row_scaling_is_the_ipms() {
    let arms: [(&str, String, &[&str]); 4] = [
        ("fixed_column", row_scale_nl(4000.0, 0.0, None), &[]),
        ("projection", row_scale_nl(1000.0, 50.0, None), &[]),
        (
            "equilibration-based",
            row_scale_nl(4000.0, 0.0, None),
            &["nlp_scaling_method=equilibration-based"],
        ),
        (
            "user-scaling",
            row_scale_nl(1000.0, 0.0, Some((0.01, 4.0))),
            &["nlp_scaling_method=user-scaling"],
        ),
    ];
    for (arm, body, extra) in arms {
        let nl = inline_nl(&format!("rowscale_{}", arm.replace('-', "_")), &body);
        let ipm = solve_path(
            &nl,
            &[&["max_iter=1", "square_system_recovery=no"][..], extra].concat(),
        );
        let lm = solve_path(
            &nl,
            &[&["max_iter=1", "--json-detail", "full"][..], extra].concat(),
        );
        let _ = std::fs::remove_file(&nl);
        let st = &ipm.report["statistics"];
        let ipm_ratio = st["final_constr_viol"].as_f64().unwrap()
            / st["final_declared_constr_viol"].as_f64().unwrap();
        assert_eq!(lm.promoted_by(), Some(LABEL), "{arm}:\n{}", lm.out);
        let rec0 = &lm.report["iterations"][0];
        let lm_ratio = rec0["inf_pr_internal"].as_f64().unwrap() / rec0["inf_pr"].as_f64().unwrap();
        assert!(
            (lm_ratio / ipm_ratio - 1.0).abs() < 1e-9,
            "{arm}: stage s_0 = {lm_ratio}, IPM s_0 = {ipm_ratio}"
        );
    }
}

/// The narration says only what is known. This model's solve never
/// escalated, so "after a factorization escalation" — `describe()`'s
/// wording, true for the option rungs — is false here; and the probe has
/// established `∇f = 0` at the start, not that the objective is constant.
///
/// Mutation-checked: restoring `trigger.describe()` and "the model is square
/// with a constant objective" in the driver's announcement fails here.
#[test]
fn the_stage_narration_claims_nothing_it_has_not_checked() {
    let nl = inline_nl("narration", &fixed_mid_slot_nl(1.5, 1.5));
    let r = solve_path(&nl, &["max_iter=2"]);
    let _ = std::fs::remove_file(&nl);
    assert_eq!(r.promoted_by(), Some(LABEL), "{}", r.out);
    assert_eq!(r.report["statistics"]["quality_escalations"], 0);
    assert!(!r.out.contains("escalation"), "{}", r.out);
    assert!(!r.out.contains("constant objective"), "{}", r.out);
    assert!(
        r.out.contains("∇f exactly zero at the starting point"),
        "{}",
        r.out
    );
}

/// `--compute-red-hessian` on a square-system promotion: the reduced
/// Hessian is computed in `on_converged` from the IPM's KKT factor, which
/// this answer never had. It must not be faked, and it must not vanish
/// silently under a `Solve_Succeeded` either — the CLI says why, on the
/// console and in the report's warnings, instead of "see warnings above"
/// over no warning.
///
/// Mutation-checked: removing the `postopt_withheld` warning in `main.rs`
/// fails here.
#[test]
fn a_promotion_says_why_the_reduced_hessian_is_missing() {
    let nl = inline_nl("red_hessian", &fixed_mid_slot_nl(1.5, 1.5));
    let r = solve_path(&nl, &["max_iter=2", "--compute-red-hessian"]);
    let _ = std::fs::remove_file(&nl);
    assert_eq!(r.promoted_by(), Some(LABEL), "{}", r.out);
    assert!(
        r.out.contains("the reduced Hessian was not produced") && r.out.contains(LABEL),
        "{}",
        r.out
    );
    assert!(!r.out.contains("see warnings above"), "{}", r.out);
    let warnings = r.report["statistics"]["warnings"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        warnings.iter().any(|w| w
            .as_str()
            .is_some_and(|s| s.starts_with("square_system_no_postopt:"))),
        "{warnings:?}"
    );
}
