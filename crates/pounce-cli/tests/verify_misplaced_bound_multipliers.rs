//! `pounce verify --require-optimal` must not accept a bound multiplier on a
//! side that cannot carry one (gh#1202 follow-up).
//!
//! With the `ipopt_zL_out` / `ipopt_zU_out` suffixes present, verify gated
//! only on the exact residual `‖∇f + Jᵀλ − z_L + z_U‖∞`, which counts every
//! suffix entry. A multiplier on a bound the point is not on, or on a side the
//! model does not declare, balances that residual exactly as a real one
//! would. So verify certified `x = 2` as optimal for `min (x − 3)²` on
//! `[0, 10]`, given `ipopt_zU_out = −2` on the bound 8 away. And it passed
//! `presolve=yes`'s rolled-back `qpnstair` duals, whose largest entry is
//! `z_U = 2.8e4` on `rgap1`, a variable with no upper bound.
//!
//! Each test below takes one branch of the new check
//! (`misplaced_bound_multiplier` in `verify.rs`): no bound on that side; a
//! declared bound the point is off; a bound the point is on; the
//! interior-point stand-off `|z|·gap ≤ --compl-tol`, accepted at the default
//! and rejected once `--compl-tol` is tightened below it. The last test pins
//! the suffix-less path, where `--compl-tol` governs the bound-projected
//! residual by the same rule (gh#1209).

use std::path::{Path, PathBuf};
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

/// `min (x − 3)²` over one variable with the `.nl` `b`-section line `bound`
/// (`"0 lo hi"` for a box, `"3"` for free).
fn nl(bound: &str) -> String {
    format!(
        "g3 1 1 0\n 1 0 1 0 0\n 0 1\n 0 0\n 0 1 0\n 0 0 0 1\n 0 0 0 0 0\n 0 1\n 0 0\n \
         0 0 0 0 0\nO0 0\no5\no0\nv0\nn-3\nn2\nx1\n0 0.5\nb\n{bound}\nk0\nG0 1\n0 0\n"
    )
}

/// A `.sol` claiming `x` with the given suffix entries.
fn sol(x: f64, z_l: Option<f64>, z_u: Option<f64>) -> String {
    let mut s = format!("claim\n\nOptions\n3\n1\n1\n0\n0\n0\n1\n1\n{x:e}\nobjno 0 0\n");
    for (name, z) in [("ipopt_zL_out", z_l), ("ipopt_zU_out", z_u)] {
        match z {
            Some(v) => s += &format!("suffix 4 1 13 0 0\n{name}\n0 {v:e}\n"),
            None => s += &format!("suffix 4 0 13 0 0\n{name}\n"),
        }
    }
    s
}

struct Verdict {
    ok: bool,
    receipt: serde_json::Value,
}

fn verify(tag: &str, nl_text: &str, sol_text: &str, extra: &[&str]) -> Verdict {
    let dir = std::env::temp_dir().join(format!("pounce_misplaced_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (nl_path, sol_path, json) = (dir.join("m.nl"), dir.join("m.sol"), dir.join("r.json"));
    std::fs::write(&nl_path, nl_text).unwrap();
    std::fs::write(&sol_path, sol_text).unwrap();
    let out = Command::new(pounce_exe())
        .arg("verify")
        .arg(&nl_path)
        .arg(&sol_path)
        .arg("--require-optimal")
        .arg("--json-output")
        .arg(&json)
        .args(extra)
        .output()
        .expect("run pounce verify");
    let receipt = read_json(&json);
    let _ = std::fs::remove_dir_all(&dir);
    let code = out.status.code();
    assert!(
        matches!(code, Some(0 | 20)),
        "{tag}: verify exited {code:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    Verdict {
        ok: code == Some(0),
        receipt,
    }
}

fn read_json(p: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn misplaced(v: &Verdict) -> f64 {
    v.receipt["optimality"]["misplaced_bound_multiplier"]
        .as_f64()
        .expect("receipt carries misplaced_bound_multiplier")
}

fn exact(v: &Verdict) -> f64 {
    v.receipt["optimality"]["stationarity_residual_with_bound_multipliers"]
        .as_f64()
        .unwrap()
}

/// A declared bound the point is 8 away from cannot carry `z_U = 2`, even
/// though it balances `∇f = −2` exactly. `x = 2` is not optimal.
#[test]
fn a_multiplier_on_a_bound_the_point_is_off_is_rejected() {
    let v = verify("off", &nl("0 0 10"), &sol(2.0, None, Some(-2.0)), &[]);
    assert_eq!(
        exact(&v),
        0.0,
        "the exact residual is balanced by construction"
    );
    assert!(
        !v.ok,
        "x = 2 certified optimal for min (x−3)²: {}",
        v.receipt
    );
    assert_eq!(misplaced(&v), 2.0);
    let w = &v.receipt["optimality"]["worst_misplaced_bound_multiplier"];
    assert_eq!(w["side"], "upper", "{w}");
    assert_eq!(w["bound"].as_f64(), Some(10.0), "{w}");
}

/// No upper bound at all: there is nothing for `z_U` to be a multiplier of.
/// The `qpnstair` shape.
#[test]
fn a_multiplier_on_an_undeclared_side_is_rejected() {
    let v = verify("free", &nl("3"), &sol(2.0, None, Some(-2.0)), &[]);
    assert_eq!(exact(&v), 0.0);
    assert!(!v.ok, "{}", v.receipt);
    assert_eq!(misplaced(&v), 2.0);
    let w = &v.receipt["optimality"]["worst_misplaced_bound_multiplier"];
    assert!(
        w["bound"].is_null(),
        "an undeclared side reports no bound: {w}"
    );
}

/// The true optimum of `min (x − 3)²` on `[0, 1]`: `x = 1`, `z_U = 4`.
#[test]
fn a_multiplier_on_the_bound_the_point_is_on_is_accepted() {
    let v = verify("on", &nl("0 0 1"), &sol(1.0, None, Some(-4.0)), &[]);
    assert!(v.ok, "{}", v.receipt);
    assert_eq!(misplaced(&v), 0.0);
}

/// An interior-point answer stops short of an active bound: here `2e-7`
/// inside, ten times the `1e-8·(1+|b|)` that counts as on it, with
/// `|z|·gap = 8e-7`. That is within the solver's own complementarity
/// tolerance, so the bound carries `z_U`. Tightening `--compl-tol` below
/// `8e-7` takes the other branch.
#[test]
fn a_barrier_stand_off_carries_its_multiplier_up_to_compl_tol() {
    let x = 1.0 - 2e-7;
    let z_u = 2.0 * (x - 3.0);
    let model = nl("0 0 1");
    let v = verify("standoff", &model, &sol(x, None, Some(z_u)), &[]);
    assert!(v.ok, "{}", v.receipt);
    assert_eq!(misplaced(&v), 0.0);
    assert_eq!(
        v.receipt["tolerances"]["complementarity"].as_f64(),
        Some(1e-4)
    );

    let tight = verify(
        "standoff_tight",
        &model,
        &sol(x, None, Some(z_u)),
        &["--compl-tol", "1e-7"],
    );
    assert!(!tight.ok, "{}", tight.receipt);
    assert!((misplaced(&tight) - 4.0).abs() < 1e-6, "{}", tight.receipt);
}

/// The solver's own `.sol` for the box model passes, and its tiny `z_L`
/// at gap 3 (complementarity `~1e-10`) is not called misplaced.
#[test]
fn a_solved_sol_is_accepted() {
    let dir = std::env::temp_dir().join(format!("pounce_misplaced_{}_solve", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (nl_path, sol_path) = (dir.join("m.nl"), dir.join("m.sol"));
    std::fs::write(&nl_path, nl("0 0 10")).unwrap();
    let out = Command::new(pounce_exe())
        .arg(&nl_path)
        .arg(&sol_path)
        .arg("--no-options-file")
        .output()
        .expect("run pounce");
    assert!(out.status.success());
    let text = std::fs::read_to_string(&sol_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let v = verify("solved", &nl("0 0 10"), &text, &[]);
    assert!(v.ok, "{}", v.receipt);
    assert_eq!(misplaced(&v), 0.0);
}

/// A `.sol` claiming `x` with a constraint-dual block but no bound-multiplier
/// suffixes at all: verify must infer the bound multipliers, so it gates on
/// the bound-projected residual.
fn sol_without_suffixes(x: f64) -> String {
    format!("claim\n\nOptions\n3\n1\n1\n0\n0\n0\n1\n1\n{x:e}\nobjno 0 0\n")
}

fn projected(v: &Verdict) -> f64 {
    v.receipt["optimality"]["stationarity_residual"]
        .as_f64()
        .unwrap_or_else(|| panic!("receipt carries stationarity_residual: {}", v.receipt))
}

/// gh#1209: without suffixes, the bound-projected residual decides bound
/// activity by `--compl-tol`, the same rule the misplaced-multiplier check
/// applies when the suffixes are present. `x` stands `2e-6` inside the
/// upper bound with `∇f = −4`, so `|s|·gap = 8e-6`: in the `(1e-6, 1e-4]`
/// band that read as *inactive* (residual 4, rejected) when the projection
/// used `--opt-tol`, while the same point with `ipopt_zU_out` was accepted.
/// Each branch of the threshold is taken once.
#[test]
fn without_suffixes_a_stand_off_is_active_up_to_compl_tol() {
    let x = 1.0 - 2e-6;
    let model = nl("0 0 1");

    let v = verify("nosuf_band", &model, &sol_without_suffixes(x), &[]);
    assert!(
        v.receipt["optimality"]["bound_multipliers_present"] == false,
        "fixture must take the suffix-less path: {}",
        v.receipt
    );
    assert!(v.ok, "{}", v.receipt);
    assert_eq!(projected(&v), 0.0);

    // The same point with the suffix: the two paths now agree.
    let with = verify(
        "suf_band",
        &model,
        &sol(x, None, Some(2.0 * (x - 3.0))),
        &[],
    );
    assert!(with.ok, "{}", with.receipt);

    // `--compl-tol` reaches the projected residual: below 8e-6 the bound no
    // longer absorbs the gradient and the whole of it is residual.
    let tight = verify(
        "nosuf_band_tight",
        &model,
        &sol_without_suffixes(x),
        &["--compl-tol", "1e-6"],
    );
    assert!(!tight.ok, "{}", tight.receipt);
    assert!((projected(&tight) - 4.0).abs() < 1e-5, "{}", tight.receipt);
}
