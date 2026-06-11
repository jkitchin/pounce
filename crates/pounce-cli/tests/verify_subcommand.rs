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
//!   key can't mint a receipt that validates).

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
