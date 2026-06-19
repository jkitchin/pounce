//! End-to-end integration test for `pounce certify`.
//!
//! Runs the real binary against a committed convex-QP `.nl`/`.sol` pair
//! (min x₀²+x₁² s.t. x₀+x₁ ≥ 1, free variables) and checks that the emitted
//! `pounce.lean-cert/v1` certificate:
//!
//! * is the supported slice (`qp-convex` / `global-min`),
//! * **snaps the ~1e-9-off float solution to the exact rational optimum**
//!   `x* = (1/2, 1/2)`, `λ = 1`, objective `1/2` (Mode B refinement), and
//! * content-addresses the actual input bytes.
//!
//! Off-slice inputs (the bounded `convex_qp.nl` fixture) must be refused.

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

#[test]
fn certify_emits_exact_certificate() {
    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(fixture("certify_qp.nl"))
        .arg(fixture("certify_qp.sol"))
        .output()
        .expect("run pounce certify");
    assert!(
        out.status.success(),
        "certify failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let cert: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("certify stdout is JSON");

    assert_eq!(cert["schema"], "pounce.lean-cert/v1");
    assert_eq!(cert["verdict"], "global-min");
    assert_eq!(cert["problem_class"], "qp-convex");
    assert_eq!(cert["tolerance"], serde_json::json!({"num":"0","den":"1"}));

    // The float .sol is ~5e-9 off; the cert must carry the EXACT optimum.
    assert_eq!(
        cert["candidate"]["x"][0],
        serde_json::json!({"num":"1","den":"2"})
    );
    assert_eq!(
        cert["candidate"]["x"][1],
        serde_json::json!({"num":"1","den":"2"})
    );
    assert_eq!(
        cert["candidate"]["objective"],
        serde_json::json!({"num":"1","den":"2"})
    );
    assert_eq!(
        cert["witnesses"]["duals"][0],
        serde_json::json!({"num":"1","den":"1"})
    );
    assert_eq!(cert["witnesses"]["active_set"], serde_json::json!([0]));

    // Free variables surface as the infinity sentinels, not 1e19.
    assert_eq!(cert["problem"]["var_bounds"]["lower"][0], "-inf");
    assert_eq!(cert["problem"]["var_bounds"]["upper"][0], "+inf");

    // Content-addressing: 64-hex digests of the actual bytes.
    let nl_hash = cert["binding"]["nl_sha256"].as_str().unwrap();
    assert_eq!(nl_hash.len(), 64);
    assert!(nl_hash.chars().all(|c| c.is_ascii_hexdigit()));
}

fn cert_verify(nl: &str, cert: &str) -> std::process::Output {
    Command::new(pounce_exe())
        .arg("cert-verify")
        .arg(fixture(nl))
        .arg(fixture(cert))
        .output()
        .expect("run pounce cert-verify")
}

#[test]
fn cert_verify_accepts_the_real_certificate() {
    let out = cert_verify("certify_qp.nl", "certify_qp.cert.json");
    assert!(
        out.status.success(),
        "real cert should verify against its .nl: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn cert_verify_rejects_easier_problem_forgery() {
    // certify_qp_fake_easier.cert.json drops the constraint and claims the
    // unconstrained min — a *true* proof of a different problem that PASSES
    // `lake build`, with binding.nl_sha256 still matching certify_qp.nl. The
    // consumer-side re-derivation must catch it.
    let out = cert_verify("certify_qp.nl", "certify_qp_fake_easier.cert.json");
    assert!(
        !out.status.success(),
        "easier-problem forgery must be rejected even though its hash matches"
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("different problem"));
}

#[test]
fn cert_verify_rejects_wrong_nl() {
    // A cert for one problem checked against a different .nl: hash mismatch.
    let out = cert_verify("certify_box.nl", "certify_qp.cert.json");
    assert!(
        !out.status.success(),
        "cert for a different .nl must be rejected"
    );
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn certify_refuses_off_slice() {
    // A maximize objective is outside the v1 slice (global-min verdict only).
    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(fixture("certify_maximize.nl"))
        .arg(fixture("certify_maximize.sol"))
        .output()
        .expect("run pounce certify");
    assert!(!out.status.success(), "off-slice input should be refused");
    assert_eq!(out.status.code(), Some(2), "refusal should exit 2");
}
