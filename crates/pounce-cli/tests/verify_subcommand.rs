//! End-to-end integration test for the `pounce verify` subcommand.
//!
//! Drives the real binary against the committed `parametric.nl` fixture
//! (5 vars, 4 cons) and checks the trust contract:
//!
//! * a genuine `.sol` (produced by an actual solve) → `VERIFIED`, exit 0;
//! * a tampered primal → `REJECTED`, exit 20;
//! * an all-zeros fabricated `.sol` → `REJECTED`, exit 20;
//! * a `.sol` whose dimensions don't match the `.nl` → usage error, exit 2;
//! * with `POUNCE_VERIFY_KEY` set, the receipt carries an HMAC-SHA256
//!   signature that re-derives from the documented float-free preimage —
//!   and flipping the key makes the signature change (an agent without the
//!   key can't mint a receipt that validates);
//! * the two complementarity residuals are labelled apart, and the bound one
//!   reads `not checked` rather than a number when the `.sol` carries no
//!   bound multipliers (gh #516).

use std::path::PathBuf;
use std::process::Command;

use pounce_cli::verify::sha256;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture_nl() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("parametric.nl");
    p
}

fn tmp(suffix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pounce_verify_{}_{suffix}", std::process::id()));
    p
}

/// Run a genuine solve to produce a `.sol` next to a temp path.
fn solve_to(sol: &PathBuf) {
    let status = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg(sol)
        .status()
        .expect("spawn pounce solve");
    assert!(status.success(), "solve failed: {status:?}");
    assert!(sol.exists(), "no .sol written");
}

fn verify_exit(nl: &PathBuf, sol: &PathBuf) -> i32 {
    Command::new(pounce_exe())
        .arg("verify")
        .arg(nl)
        .arg(sol)
        .status()
        .expect("spawn pounce verify")
        .code()
        .expect("exit code")
}

fn verify_stdout(nl: &PathBuf, sol: &PathBuf) -> String {
    let out = Command::new(pounce_exe())
        .arg("verify")
        .arg(nl)
        .arg(sol)
        .output()
        .expect("spawn pounce verify");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn genuine_solution_verifies() {
    let sol = tmp("good.sol");
    solve_to(&sol);
    assert_eq!(
        verify_exit(&fixture_nl(), &sol),
        0,
        "genuine .sol should verify"
    );
    let _ = std::fs::remove_file(&sol);
}

#[test]
fn tampered_primal_is_rejected() {
    let sol = tmp("tamper.sol");
    solve_to(&sol);
    // Bump the last primal line by a large amount so at least one
    // constraint residual blows past the feasibility tolerance.
    let text = std::fs::read_to_string(&sol).unwrap();
    let mut lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
    // The last numeric line before `objno` is a primal value.
    let objno_idx = lines.iter().position(|l| l.starts_with("objno")).unwrap();
    let last_primal = objno_idx - 1;
    lines[last_primal] = "9.9e9".to_string();
    std::fs::write(&sol, lines.join("\n")).unwrap();
    assert_eq!(
        verify_exit(&fixture_nl(), &sol),
        20,
        "tampered .sol must be rejected"
    );
    let _ = std::fs::remove_file(&sol);
}

#[test]
fn fabricated_zeros_is_rejected() {
    // A plausible-looking all-zeros solution with a "solved" status.
    let n = 5;
    let m = 4;
    let mut s = String::from("POUNCE 9.9: Optimal Solution Found\n\nOptions\n0\n");
    s.push_str(&format!("{m}\n{m}\n{n}\n{n}\n"));
    for _ in 0..m {
        s.push_str("0.0\n");
    }
    for _ in 0..n {
        s.push_str("0.0\n");
    }
    s.push_str("objno 0 0\n");
    let sol = tmp("fake.sol");
    std::fs::write(&sol, s).unwrap();
    assert_eq!(
        verify_exit(&fixture_nl(), &sol),
        20,
        "fabricated .sol must be rejected"
    );
    let _ = std::fs::remove_file(&sol);
}

#[test]
fn dimension_mismatch_is_usage_error() {
    // 3 primals where the problem has 5 → exit 2.
    let mut s = String::from("msg\n\nOptions\n0\n0\n0\n3\n3\n");
    for _ in 0..3 {
        s.push_str("1.0\n");
    }
    s.push_str("objno 0 0\n");
    let sol = tmp("mismatch.sol");
    std::fs::write(&sol, s).unwrap();
    assert_eq!(
        verify_exit(&fixture_nl(), &sol),
        2,
        "dimension mismatch must be a usage error"
    );
    let _ = std::fs::remove_file(&sol);
}

/// **gh #516.** `verify` printed `complementarity residual` for the *row*
/// quantity, next to a `KKT stationarity residual` line under an
/// `optimality` heading — the exact framing that invites a comparison
/// against a solver's `Complementarity` output, which is the *bound*
/// quantity. On the #505 model the two read `4.529e-2` and `1.179e-11`:
/// nine orders of magnitude, same point, same file, both called
/// "complementarity". Two people read the first as a signal about the
/// solution before tracing it to the definition.
#[test]
fn the_two_complementarity_residuals_are_labelled_apart() {
    let sol = tmp("compl.sol");
    solve_to(&sol);
    let out = verify_stdout(&fixture_nl(), &sol);

    assert!(
        out.contains("constraint complementarity (rows, |λ|·slack)"),
        "the row quantity must say it is over rows:\n{out}"
    );
    assert!(
        out.contains("bound complementarity (vars, |z|·slack)"),
        "the bound quantity must say it is over variables:\n{out}"
    );
    // No line may offer a bare `complementarity residual` for either to be
    // mistaken for. This is the assertion that fails if the label regresses.
    assert!(
        !out.contains("complementarity residual"),
        "an unqualified `complementarity residual` label is the bug:\n{out}"
    );
    let _ = std::fs::remove_file(&sol);
}

/// The bound quantity needs `ipopt_zL_out` / `ipopt_zU_out`, which a `.sol`
/// need not carry. When it doesn't, say *not checked* — reporting nothing
/// leaves the row line to be read as the bound one, and reporting `0.0`
/// would be a fabricated clean bill of health.
#[test]
fn absent_bound_multipliers_read_as_not_checked() {
    // A hand-built .sol with duals but no suffix blocks at all.
    let n = 5;
    let m = 4;
    let mut s = String::from("POUNCE: Optimal Solution Found\n\nOptions\n0\n");
    s.push_str(&format!("{m}\n{m}\n{n}\n{n}\n"));
    for _ in 0..m {
        s.push_str("0.0\n");
    }
    for _ in 0..n {
        s.push_str("0.5\n");
    }
    s.push_str("objno 0 0\n");
    let sol = tmp("nozl.sol");
    std::fs::write(&sol, s).unwrap();

    let out = verify_stdout(&fixture_nl(), &sol);
    let line = out
        .lines()
        .find(|l| l.contains("bound complementarity"))
        .unwrap_or_else(|| panic!("no bound complementarity line:\n{out}"));
    assert!(
        line.contains("not checked"),
        "bound complementarity must report `not checked`, not a number: {line}"
    );
    assert!(
        out.contains("ipopt_zL_out/ipopt_zU_out"),
        "say *why* it is not checked:\n{out}"
    );
    let _ = std::fs::remove_file(&sol);
}

/// The point of naming the bound quantity is that it *is* the solver's:
/// `verify` re-derives it from the `.nl` and the `.sol` alone, so it must
/// land on the same number the solve printed. Pinning the agreement is what
/// makes the comparison the labels invite a legitimate one.
#[test]
fn bound_complementarity_reproduces_the_solvers_own_figure() {
    let sol = tmp("compl_match.sol");
    let out = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg(&sol)
        .output()
        .expect("spawn pounce solve");
    assert!(out.status.success());
    let solve_log = String::from_utf8_lossy(&out.stdout).into_owned();

    // `Complementarity.........:   9.09e-10    9.09e-10` — scaled, unscaled.
    // The fixture is unscaled, so either column serves.
    let from_log = |label: &str| -> f64 {
        let line = solve_log
            .lines()
            .find(|l| l.starts_with(label))
            .unwrap_or_else(|| panic!("no `{label}` line in the solve log:\n{solve_log}"));
        line.rsplit_once(':')
            .and_then(|(_, rest)| rest.split_whitespace().next())
            .and_then(|t| t.parse::<f64>().ok())
            .unwrap_or_else(|| panic!("unparseable `{label}` line: {line}"))
    };
    let solver_compl = from_log("Complementarity");
    let solver_dual_inf = from_log("Dual infeasibility");

    let report = verify_stdout(&fixture_nl(), &sol);
    let from_report = |label: &str| -> f64 {
        let line = report
            .lines()
            .find(|l| l.contains(label))
            .unwrap_or_else(|| panic!("no `{label}` line:\n{report}"));
        line.rsplit_once(':')
            .and_then(|(_, rest)| rest.split_whitespace().next())
            .and_then(|t| t.parse::<f64>().ok())
            .unwrap_or_else(|| panic!("unparseable `{label}` line: {line}"))
    };

    // `verify` prints 3 significant digits, so agreement is to that.
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-3 * a.abs().max(b.abs()).max(1e-300);
    let ours = from_report("bound complementarity (vars");
    assert!(
        close(ours, solver_compl),
        "verify's bound complementarity {ours:.3e} must match the solver's \
         Complementarity {solver_compl:.3e}"
    );
    let ours = from_report("dual infeasibility (with z_L/z_U");
    assert!(
        close(ours, solver_dual_inf),
        "verify's exact dual infeasibility {ours:.3e} must match the solver's \
         {solver_dual_inf:.3e}"
    );

    let _ = std::fs::remove_file(&sol);
}

/// A genuine pounce `.sol` does carry the suffixes, so both quantities are
/// real numbers and the receipt names them apart too.
#[test]
fn receipt_separates_the_two_complementarity_fields() {
    let sol = tmp("compl_receipt.sol");
    solve_to(&sol);
    let receipt = tmp("compl_receipt.json");
    let status = Command::new(pounce_exe())
        .arg("verify")
        .arg(fixture_nl())
        .arg(&sol)
        .arg("--json-output")
        .arg(&receipt)
        .status()
        .expect("spawn pounce verify --json-output");
    assert_eq!(status.code(), Some(0));

    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&receipt).unwrap()).unwrap();
    let opt = &v["optimality"];
    assert_eq!(opt["bound_multipliers_present"], true);
    assert!(
        opt["constraint_complementarity_residual"].is_number(),
        "receipt must carry the row quantity under its own name: {opt}"
    );
    assert!(
        opt["bound_complementarity_residual"].is_number(),
        "a .sol with zL/zU suffixes must report bound complementarity: {opt}"
    );
    // The deprecated alias stays put for v1 consumers, and stays equal to
    // the field it aliases.
    assert_eq!(
        opt["complementarity_residual"],
        opt["constraint_complementarity_residual"]
    );

    let _ = std::fs::remove_file(&sol);
    let _ = std::fs::remove_file(&receipt);
}

#[test]
fn signed_receipt_validates_with_the_key_only() {
    let sol = tmp("signed.sol");
    solve_to(&sol);
    let receipt = tmp("receipt.json");
    let key = "test-secret-key-not-the-agent's";

    let status = Command::new(pounce_exe())
        .arg("verify")
        .arg(fixture_nl())
        .arg(&sol)
        .arg("--json-output")
        .arg(&receipt)
        .env("POUNCE_VERIFY_KEY", key)
        .status()
        .expect("spawn pounce verify --json-output");
    assert_eq!(status.code(), Some(0));

    let text = std::fs::read_to_string(&receipt).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["signature_alg"], "HMAC-SHA256");
    let sig = v["signature"].as_str().expect("signature present");

    // Re-derive the float-free preimage from the receipt fields exactly as
    // documented, and recompute the HMAC with the key. A consumer holding
    // the key accepts; the agent (without the key) cannot forge this.
    let preimage = format!(
        "pounce-verify-receipt/v1\n\
         verify_version=1\n\
         nl_sha256={}\n\
         sol_sha256={}\n\
         n_vars={}\n\
         n_cons={}\n\
         feasible={}\n\
         verified={}\n\
         verdict={}\n",
        v["problem"]["sha256"].as_str().unwrap(),
        v["solution"]["sha256"].as_str().unwrap(),
        v["problem"]["n_vars"].as_u64().unwrap(),
        v["problem"]["n_cons"].as_u64().unwrap(),
        v["feasibility"]["feasible"].as_bool().unwrap(),
        v["verified"].as_bool().unwrap(),
        v["verdict"].as_str().unwrap(),
    );
    let expect = sha256::hmac_hex(key.as_bytes(), preimage.as_bytes());
    assert_eq!(sig, expect, "signature must validate with the real key");

    // A different key produces a different MAC — forgery without the key
    // fails.
    let wrong = sha256::hmac_hex(b"wrong-key", preimage.as_bytes());
    assert_ne!(sig, wrong, "signature must not validate under a wrong key");

    let _ = std::fs::remove_file(&sol);
    let _ = std::fs::remove_file(&receipt);
}

/// Run `verify --json-output` and return the receipt's `optimality` block.
fn receipt_optimality(tag: &str, nl: &PathBuf, sol_text: &str) -> serde_json::Value {
    let sol = tmp(&format!("{tag}.sol"));
    let receipt = tmp(&format!("{tag}.json"));
    std::fs::write(&sol, sol_text).unwrap();
    let status = Command::new(pounce_exe())
        .arg("verify")
        .arg(nl)
        .arg(&sol)
        .arg("--json-output")
        .arg(&receipt)
        .status()
        .expect("spawn pounce verify --json-output");
    assert!(matches!(status.code(), Some(0 | 20)), "{tag}: {status:?}");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&receipt).unwrap()).unwrap();
    let _ = std::fs::remove_file(&sol);
    let _ = std::fs::remove_file(&receipt);
    v["optimality"].clone()
}

/// The receipt's `optimality` block appears exactly when optimality was
/// computed. A model with no rows needs no constraint duals, so a `.sol`
/// without duals or bound-multiplier suffixes still gets a residual, and
/// `--require-optimal` gates on it. The receipt reported
/// `available: false` for that case. A model *with* rows and no duals takes
/// the other branch: nothing is computed, and the block says so.
#[test]
fn receipt_reports_optimality_whenever_it_was_computed() {
    // `min (x − 3)²` on `[0, 1]`, no rows; `x = 1` is the optimum.
    let rowless = tmp("rowless.nl");
    std::fs::write(
        &rowless,
        "g3 1 1 0\n 1 0 1 0 0\n 0 1\n 0 0\n 0 1 0\n 0 0 0 1\n 0 0 0 0 0\n 0 1\n 0 0\n \
         0 0 0 0 0\nO0 0\no5\no0\nv0\nn-3\nn2\nx1\n0 0.5\nb\n0 0 1\nk0\nG0 1\n0 0\n",
    )
    .unwrap();
    let opt = receipt_optimality(
        "rowless",
        &rowless,
        "claim\n\nOptions\n3\n1\n1\n0\n0\n0\n1\n1\n1.0\nobjno 0 0\n",
    );
    let _ = std::fs::remove_file(&rowless);
    assert_eq!(opt["available"], true, "{opt}");
    assert_eq!(opt["bound_multipliers_present"], false, "{opt}");
    assert_eq!(opt["stationarity_residual"].as_f64(), Some(0.0), "{opt}");
    assert_eq!(opt["optimal"], true, "{opt}");

    // `parametric.nl` has 4 rows; a `.sol` with no duals leaves nothing to
    // compute optimality from.
    let mut s = String::from("claim\n\nOptions\n0\n0\n4\n5\n5\n");
    for _ in 0..5 {
        s.push_str("0.0\n");
    }
    s.push_str("objno 0 0\n");
    let opt = receipt_optimality("rows_no_duals", &fixture_nl(), &s);
    assert_eq!(opt["available"], false, "{opt}");
}
