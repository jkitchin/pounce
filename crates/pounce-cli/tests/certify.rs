//! End-to-end integration test for `pounce certify`.
//!
//! Runs the real binary against a convex-QP `.nl` (min x₀²+x₁² s.t. x₀+x₁ ≥ 1,
//! free variables), solving it first to produce the `.sol`, and checks that the
//! emitted
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

/// Solve `<stem>.nl` to produce `<stem>.sol` beside it, and return that path.
///
/// `.sol` files are solver byproducts, not fixtures — `tests/fixtures/.gitignore`
/// excludes `*.sol`, so a test naming one that was never generated fails (or,
/// worse, passes for the wrong reason: a missing file also exits 2). Generating
/// it here makes this a genuine end-to-end run: solve in f64, then certify
/// exactly. The float `x*` lands ~4e-9 off the true optimum, which is exactly
/// the input Mode B refinement has to snap.
///
/// The one committed `.sol` (`certify_feasible`) is the exception the gitignore
/// spells out, and tests use it directly rather than through this helper.
fn solve_to_sol(stem: &str) -> PathBuf {
    let out = Command::new(pounce_exe())
        .arg(fixture(&format!("{stem}.nl")))
        .output()
        .expect("run pounce solve");
    assert!(
        out.status.success(),
        "solve of {stem}.nl failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let sol = fixture(&format!("{stem}.sol"));
    assert!(sol.exists(), "solve did not write {}", sol.display());
    sol
}

#[test]
fn certify_emits_exact_certificate() {
    let sol = solve_to_sol("certify_qp");
    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(fixture("certify_qp.nl"))
        .arg(&sol)
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
    // The .sol must be generated, not named: a missing file ALSO exits 2, so
    // naming an uncommittable `.sol` here would make this assertion vacuous.
    let sol = solve_to_sol("certify_maximize");
    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(fixture("certify_maximize.nl"))
        .arg(&sol)
        .output()
        .expect("run pounce certify");
    assert!(!out.status.success(), "off-slice input should be refused");
    assert_eq!(out.status.code(), Some(2), "refusal should exit 2");
    // Distinguish a real refusal from an I/O failure.
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !err.contains("cannot read"),
        "must refuse on slice grounds, not I/O: {err}"
    );
}

#[test]
fn certify_routes_an_infeasible_solve_to_a_farkas_certificate() {
    // The solve exits nonzero (the problem has no solution) but still writes a
    // .sol; solve_to_sol asserts success, so drive the binary directly here.
    let nl = fixture("certify_infeasible.nl");
    let _ = Command::new(pounce_exe()).arg(&nl).output();
    let sol = fixture("certify_infeasible.sol");
    assert!(
        sol.exists(),
        "an infeasible solve must still write its .sol"
    );

    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(&nl)
        .arg(&sol)
        .output()
        .expect("run pounce certify");
    assert!(
        out.status.success(),
        "certifying an infeasible solve should succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let cert: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("certify stdout is JSON");
    assert_eq!(cert["verdict"], "infeasible");
    // Not a claim about a point: the key is absent entirely.
    assert!(
        cert.get("candidate").is_none(),
        "an infeasible certificate carries no candidate"
    );
    // The exact ray, not the ~2.3e10 float one the solver returned.
    assert_eq!(
        cert["witnesses"]["farkas"]["y"],
        serde_json::json!([
            {"num":"1","den":"1"},
            {"num":"1","den":"1"},
            {"num":"1","den":"1"}
        ])
    );
    assert!(
        cert["witnesses"].get("duals").is_none(),
        "KKT witnesses do not belong on an infeasible certificate"
    );
}

#[test]
fn certify_routes_an_unbounded_solve_to_a_recession_certificate() {
    // Like the infeasible case, an unbounded solve exits nonzero while still
    // writing its .sol, so solve_to_sol (which asserts success) cannot be used.
    let nl = fixture("certify_unbounded.nl");
    let _ = Command::new(pounce_exe()).arg(&nl).output();
    let sol = fixture("certify_unbounded.sol");
    assert!(sol.exists(), "an unbounded solve must still write its .sol");

    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(&nl)
        .arg(&sol)
        .output()
        .expect("run pounce certify");
    assert!(
        out.status.success(),
        "certifying an unbounded solve should succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let cert: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("certify stdout is JSON");
    assert_eq!(cert["verdict"], "unbounded");
    assert!(cert.get("candidate").is_none());
    // BOTH witnesses: a direction alone cannot distinguish unbounded from
    // an empty feasible set.
    assert!(cert["witnesses"]["recession"]["x0"].is_array());
    assert!(cert["witnesses"]["recession"]["d"].is_array());
    assert!(cert["witnesses"].get("farkas").is_none());
}

#[test]
fn certify_handles_an_unbounded_solve_with_a_nonzero_hessian() {
    // `min x₀² − x₁ s.t. x₀ + x₁ ≥ 1`: curved in x₀, flat and descending in
    // x₁. The LP fixture above never exercises `Q d = 0`, because for `Q = 0`
    // that condition is vacuous — every recession condition it has is an
    // *inequality*, and inequalities survive the f64→ℚ conversion with their
    // margin intact. A nonzero Q reintroduces an equality, and the solver's
    // diverging iterate misses it: its x₀ coordinate is a small nonzero
    // dyadic, so `Q d` would be nonzero and the Lean goal undischargeable.
    //
    // So `d` here is the exact projection of that iterate onto ker Q, which is
    // what makes the emitted direction the round `[0, 1]` rather than the
    // 16-digit fraction `x0` still is.
    let nl = fixture("certify_unbounded_qp.nl");
    let _ = Command::new(pounce_exe()).arg(&nl).output();
    let sol = fixture("certify_unbounded_qp.sol");
    assert!(sol.exists(), "an unbounded solve must still write its .sol");

    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(&nl)
        .arg(&sol)
        .output()
        .expect("run pounce certify");
    assert!(
        out.status.success(),
        "a nonzero Hessian is no longer refused: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let cert: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("certify stdout is JSON");
    assert_eq!(cert["verdict"], "unbounded");
    assert!(
        !cert["problem"]["objective"]["Q"]["entries"]
            .as_array()
            .expect("Q entries")
            .is_empty(),
        "the point of this fixture is that Q is not empty"
    );
    assert_eq!(
        cert["witnesses"]["recession"]["d"],
        serde_json::json!([{"num":"0","den":"1"}, {"num":"1","den":"1"}]),
        "d is projected onto ker Q exactly, then normalized"
    );
    // x₀ is a different kind of witness and keeps its float provenance: it is
    // a claim about *that* point being feasible, so it is the iterate verbatim.
    let x0 = cert["witnesses"]["recession"]["x0"]
        .as_array()
        .expect("x0 array");
    assert_ne!(
        x0[0]["den"], "1",
        "x₀ is the raw iterate, not a rounded one"
    );
}

#[test]
fn certify_routes_a_nonconvex_polynomial_to_an_sos_global_min() {
    // x⁴ − 2x² + 2 has two global minima at x = ±1 and a local max at 0, so
    // the f64 solve lands on ONE basin and can say nothing global. The SOS
    // route proves 1 ≤ p(x) for every real x without trusting that basin — and
    // since the solve's own point snaps to x = 1, where p is exactly 1, the
    // bound is attained and the verdict is a global minimum.
    let sol = solve_to_sol("certify_sos");
    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(fixture("certify_sos.nl"))
        .arg(&sol)
        .output()
        .expect("run pounce certify");
    assert!(
        out.status.success(),
        "certify failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let cert: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("certify stdout is JSON");
    assert_eq!(cert["verdict"], "global-min");
    assert_eq!(cert["problem_class"], "sos-poly");
    // The SDP's float bound is ~1 − 1e-9; the emitted γ is exactly 1.
    assert_eq!(cert["bound"], serde_json::json!({"num":"1","den":"1"}));
    assert_eq!(cert["tolerance"], serde_json::json!({"num":"0","den":"1"}));
    // The iterate is ~1 − 4e-10; the exhibited minimizer is exactly 1, and its
    // objective is γ itself — that equality is what makes the bound a minimum.
    assert_eq!(
        cert["candidate"]["x"],
        serde_json::json!([{"num":"1","den":"1"}])
    );
    assert_eq!(cert["candidate"]["objective"], cert["bound"]);
    // The QP half is absent: this is not a KKT claim.
    assert!(cert["problem"].get("objective").is_none());
    assert!(cert["witnesses"].get("duals").is_none());

    // p − γ = m(x)ᵀ G m(x) over the basis {1, x, x²}, with G = LDLᵀ exact.
    let sos = &cert["witnesses"]["sos"][0];
    assert_eq!(sos["monomials"], serde_json::json!([[0], [1], [2]]));
    assert!(sos["gram"]["symmetric"].as_bool().unwrap());
    assert!(sos["L"]["unit_lower"].as_bool().unwrap());
    // p − 1 = (x² − 1)², so G has rank one: D = (1, 0, 0). Nonneg, exactly.
    assert_eq!(
        sos["D"],
        serde_json::json!([
            {"num":"1","den":"1"},
            {"num":"0","den":"1"},
            {"num":"0","den":"1"}
        ])
    );
}

/// The other half of the distinction: a polynomial whose minimizer is
/// irrational. `x⁴ − 3x² + 2` minimizes at `±√(3/2)`, so no rational point
/// attains the bound and the verdict must stay a bound — a weaker claim,
/// correctly made, rather than a stronger one made up.
#[test]
fn an_irrational_minimizer_certifies_only_a_bound() {
    let sol = solve_to_sol("certify_sos_bound");
    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(fixture("certify_sos_bound.nl"))
        .arg(&sol)
        .output()
        .expect("run pounce certify");
    assert!(
        out.status.success(),
        "certify failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let cert: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("certify stdout is JSON");
    assert_eq!(cert["verdict"], "global-lower-bound");
    assert!(
        cert.get("candidate").is_none(),
        "a bound names no minimizer"
    );
    // p + 1/4 = (x² − 3/2)², so the tight γ is −1/4 — reachable only on a grid
    // finer than the slack −1 sits on. Anything coarser here is a sharpness
    // regression in the γ ladder, not a soundness one.
    assert_eq!(cert["bound"], serde_json::json!({"num":"-1","den":"4"}));
}

/// The constrained (Putinar) shape, on a problem the unconstrained emitter
/// cannot touch at all: `x³ − 3x` has no lower bound on ℝ, so `p − γ` is never
/// a sum of squares for any γ. On `0 ≤ x ≤ 3` it is one *modulo the
/// constraints*, and that is the whole content of the constrained path.
#[test]
fn a_box_constrained_polynomial_certifies_a_bound_on_the_feasible_set() {
    let sol = solve_to_sol("certify_sos_box");
    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(fixture("certify_sos_box.nl"))
        .arg(&sol)
        .output()
        .expect("run pounce certify");
    assert!(
        out.status.success(),
        "certify failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let cert: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("certify stdout is JSON");
    assert_eq!(cert["problem_class"], "sos-poly");
    // x = 1 is rational and feasible, and p(1) = −2 exactly, so the bound is
    // attained and the verdict is a (constrained) minimum.
    assert_eq!(cert["verdict"], "global-min");
    assert_eq!(cert["bound"], serde_json::json!({"num":"-2","den":"1"}));
    assert_eq!(
        cert["candidate"]["x"],
        serde_json::json!([{"num":"1","den":"1"}])
    );

    // The feasible set travels with the certificate. Without it the same bound
    // would read as a claim about every x — which is false here.
    let g = cert["problem"]["poly_constraints"]
        .as_array()
        .expect("poly_constraints");
    assert_eq!(g.len(), 2, "a box on one variable is two constraints");

    // σ₀ carries no `multiplier` (its multiplier is the constant 1); the
    // localizing blocks name the constraint each multiplies, in that order.
    let sos = cert["witnesses"]["sos"].as_array().expect("sos blocks");
    assert_eq!(sos.len(), 3, "σ₀ plus one block per constraint");
    assert!(sos[0].get("multiplier").is_none());
    assert_eq!(sos[1]["multiplier"], 0);
    assert_eq!(sos[2]["multiplier"], 1);
    // Every block's D is nonnegative — that is what PSD-ness reduces to, and
    // it is checked per block, not just for σ₀.
    for blk in sos {
        for d in blk["D"].as_array().expect("D") {
            let num: i64 = d["num"].as_str().expect("num").parse().expect("integer");
            assert!(num >= 0, "block diagonal must be nonneg, got {d}");
        }
    }
}

/// `--local` on a problem whose local minimum is emphatically NOT global.
///
/// 3x⁴ + 4x³ − 12x² has two minima: x = −2 with p = −32, and x = 1 with p = −5.
/// Started near the second, the solver reports x = 1 — and the global claim
/// "−5 ≤ p(x) everywhere" is simply false, so no certificate for it exists at
/// any relaxation order. Restricted to a ball around x = 1 the claim is true,
/// and the same Putinar machinery proves it with one extra multiplier.
#[test]
fn the_local_flag_certifies_a_minimum_that_is_not_global() {
    let sol = solve_to_sol("certify_sos_local");
    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(fixture("certify_sos_local.nl"))
        .arg(&sol)
        .arg("--local")
        .output()
        .expect("run pounce certify");
    assert!(
        out.status.success(),
        "certify --local failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let cert: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("certify stdout is JSON");
    assert_eq!(cert["problem_class"], "sos-poly");
    // The verdict says `local`, and that word is load-bearing: p(−2) = −32 < −5.
    assert_eq!(cert["verdict"], "local-min");
    assert_eq!(cert["bound"], serde_json::json!({"num":"-5","den":"1"}));
    assert_eq!(
        cert["candidate"]["x"],
        serde_json::json!([{"num":"1","den":"1"}])
    );

    // The ball travels with the certificate — exactly, as a centre and a
    // *squared* radius, because ℚ has no square roots. Without it the bound
    // would read as a global claim.
    let nb = &cert["problem"]["neighborhood"];
    assert_eq!(nb["center"], serde_json::json!([{"num":"1","den":"1"}]));
    let r: i64 = nb["radius_sq"]["num"]
        .as_str()
        .expect("radius_sq num")
        .parse()
        .expect("integer");
    assert!(r > 0, "a non-positive radius² would make the claim vacuous");

    // The global minimizer is outside that ball — otherwise the local claim
    // would be a global one wearing a disguise, and this fixture would be
    // testing nothing.
    assert!(
        (-2f64 - 1.0).powi(2) > r as f64,
        "x = -2 must lie outside the ball for this to be a genuinely local claim"
    );

    // No constraints, so the multiplier family is the ball alone: σ₀ (no
    // `multiplier`) plus one block at index 0, the ball's slot.
    assert!(cert["problem"]["poly_constraints"].is_null());
    let sos = cert["witnesses"]["sos"].as_array().expect("sos blocks");
    assert_eq!(sos.len(), 2, "σ₀ plus the ball's multiplier");
    assert!(sos[0].get("multiplier").is_none());
    assert_eq!(sos[1]["multiplier"], 0);
}

/// `--local` and `--feasible` ask for different claims, and `--radius` means
/// nothing on its own. Both are refused rather than silently resolved: a user
/// who typed both wanted one of them, and guessing which would be a claim they
/// did not ask for.
#[test]
fn the_local_flag_refuses_incoherent_combinations() {
    let sol = solve_to_sol("certify_sos_local");
    for flags in [
        vec!["--local", "--feasible"],
        vec!["--radius", "0.5"], // no --local
    ] {
        let mut cmd = Command::new(pounce_exe());
        cmd.arg("certify")
            .arg(fixture("certify_sos_local.nl"))
            .arg(&sol);
        for f in &flags {
            cmd.arg(f);
        }
        let out = cmd.output().expect("run pounce certify");
        assert_eq!(
            out.status.code(),
            Some(2),
            "{flags:?} should be refused, got: {}",
            String::from_utf8_lossy(&out.stdout)
        );
    }
}

/// `--growth` upgrades the same ball claim from "≤" to "<", by certifying a
/// rational modulus μ > 0 with `p(x) ≥ p(x₀) + μ‖x − x₀‖²` on the ball.
///
/// The point of the extra flag is that this is *strictly more* than a second-
/// order sufficient condition would give: SOSC concludes "is a strict local
/// minimum", while this hands back the modulus itself, exactly, as a rational.
/// It costs one more SOS solve on the shifted objective and no new theory —
/// the whole thing still lives in ℚ, where SOSC could not.
#[test]
fn the_growth_flag_certifies_a_strict_local_minimum() {
    let sol = solve_to_sol("certify_sos_local");
    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(fixture("certify_sos_local.nl"))
        .arg(&sol)
        .arg("--local")
        .arg("--growth")
        .output()
        .expect("run pounce certify");
    assert!(
        out.status.success(),
        "certify --local --growth failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let cert: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("certify stdout is JSON");
    // A distinct verdict, because it is a distinct theorem: `local-min` proves
    // p x₀ ≤ p x, this proves p x₀ < p x away from x₀.
    assert_eq!(cert["verdict"], "local-min-strict");
    assert_eq!(cert["bound"], serde_json::json!({"num":"-5","den":"1"}));

    // μ is a *witness*, so it rides beside `bound` rather than inside
    // `problem` — the problem is unchanged, and `cert-verify` compares
    // `problem`. It must be a positive rational; a non-positive one would
    // prove nothing strict at all.
    let mu = &cert["growth_modulus"];
    let (num, den): (i64, i64) = (
        mu["num"].as_str().expect("μ num").parse().expect("integer"),
        mu["den"].as_str().expect("μ den").parse().expect("integer"),
    );
    assert!(num > 0 && den > 0, "μ must be positive, got {mu}");

    // And it must be *true*, not merely positive. On this fixture the honest
    // ceiling is 5: p(x) + 5 − μ(x−1)² = (x−1)²(3x² + 10x + 3 − μ), which is
    // nonnegative on the ball iff μ ≤ min(3x² + 10x + 3) there. A μ above that
    // is a false claim; the search comes off a fixed ladder, so it lands at or
    // below it rather than exactly on it, and this bound is the invariant.
    let mu_f = num as f64 / den as f64;
    let r: f64 = cert["problem"]["neighborhood"]["radius_sq"]["num"]
        .as_str()
        .expect("radius_sq num")
        .parse()
        .expect("integer");
    let lo = 1.0 - r.sqrt();
    assert!(
        mu_f <= 3.0 * lo * lo + 10.0 * lo + 3.0 + 1e-9,
        "μ = {mu_f} overclaims the growth this problem actually has"
    );
}

/// `--growth` names a modulus measured from the minimizer *within a region*.
/// Without `--local` there is no region, and the unrestricted strict claim is a
/// different theorem — so this is refused rather than quietly promoted.
#[test]
fn the_growth_flag_requires_a_neighborhood() {
    let sol = solve_to_sol("certify_sos_local");
    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(fixture("certify_sos_local.nl"))
        .arg(&sol)
        .arg("--growth")
        .output()
        .expect("run pounce certify");
    assert_eq!(
        out.status.code(),
        Some(2),
        "--growth without --local should be refused, got: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn cert_verify_accepts_the_strict_local_sos_certificate() {
    // `growth_modulus` sits outside `problem`, so the re-derivation must be
    // untouched by it: the strict cert and the plain one describe the *same*
    // problem and differ only in what they claim about it.
    let out = cert_verify(
        "certify_sos_local_strict.nl",
        "certify_sos_local_strict.cert.json",
    );
    assert!(
        out.status.success(),
        "strict local SOS cert should verify against its .nl: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A KKT certificate for a convex QP already proves a *global* minimum, so
/// restricting it to a ball would replace a strong claim with a weaker one for
/// no reason. Refused, with that as the explanation.
#[test]
fn the_local_flag_is_refused_where_the_claim_is_already_global() {
    let sol = solve_to_sol("certify_qp");
    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(fixture("certify_qp.nl"))
        .arg(&sol)
        .arg("--local")
        .output()
        .expect("run pounce certify");
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn cert_verify_accepts_the_local_sos_certificate() {
    // The re-derivation must carry the neighborhood across untouched: it is the
    // certificate's choice, not something the `.nl` determines, so a consumer
    // that re-derived it would either invent one or drop it.
    let out = cert_verify("certify_sos_local.nl", "certify_sos_local.cert.json");
    assert!(
        out.status.success(),
        "real local SOS cert should verify against its .nl: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn cert_verify_accepts_the_constrained_sos_certificate() {
    // The re-derivation has to reproduce the *constraints* too, in the same
    // order — that order is what the `multiplier` indices refer to.
    let out = cert_verify("certify_sos_box.nl", "certify_sos_box.cert.json");
    assert!(
        out.status.success(),
        "real constrained SOS cert should verify against its .nl: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The unconstrained fixture's certificate must not verify against the
/// constrained `.nl`: same problem class, same variable count, different
/// feasible set. If `poly_constraints` were dropped from the re-derivation this
/// would pass, and a bound on a box would read as a bound on all of ℝ.
#[test]
fn cert_verify_rejects_a_constrained_cert_against_an_unconstrained_nl() {
    let out = cert_verify("certify_sos.nl", "certify_sos_box.cert.json");
    assert!(
        !out.status.success(),
        "a constrained cert must be rejected against a different .nl"
    );
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn cert_verify_accepts_the_sos_certificate() {
    // Same re-derivation path as the QP case, but through sos_problem_block:
    // the consumer rebuilds the polynomial from the .nl and must land on the
    // identical canonical problem the producer signed.
    let out = cert_verify("certify_sos.nl", "certify_sos.cert.json");
    assert!(
        out.status.success(),
        "real SOS cert should verify against its .nl: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn cert_verify_rejects_an_sos_certificate_against_the_wrong_nl() {
    let out = cert_verify("certify_qp.nl", "certify_sos.cert.json");
    assert!(
        !out.status.success(),
        "SOS cert for a different .nl must be rejected"
    );
    assert_eq!(out.status.code(), Some(2));
}

/// Run `pounce certify --feasible` on the committed nonconvex fixture.
fn certify_feasible_output() -> std::process::Output {
    Command::new(pounce_exe())
        .arg("certify")
        .arg("--feasible")
        .arg(fixture("certify_feasible.nl"))
        .arg(fixture("certify_feasible.sol"))
        .output()
        .expect("run pounce certify --feasible")
}

/// The fixture's whole point: an indefinite Hessian, so the optimality path has
/// nothing to say — and says so, rather than quietly emitting something weaker.
#[test]
fn a_nonconvex_qp_cannot_be_certified_optimal() {
    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg(fixture("certify_feasible.nl"))
        .arg(fixture("certify_feasible.sol"))
        .output()
        .expect("run pounce certify");
    assert!(
        !out.status.success(),
        "an indefinite Q has no KKT certificate"
    );
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("Indefinite"),
        "refusal should name the indefinite Hessian, got: {err}"
    );
}

/// ...and the same solve certifies `feasible` when that is what is asked for.
#[test]
fn the_feasible_flag_certifies_the_reported_point() {
    let out = certify_feasible_output();
    assert!(
        out.status.success(),
        "--feasible should certify: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let cert: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("certificate should be JSON");

    assert_eq!(cert["verdict"], "feasible");
    // The one verdict with a nonzero tolerance, and it is measured from the
    // reported point rather than copied from a solver setting.
    assert_eq!(
        cert["tolerance"],
        serde_json::json!({"num": "1", "den": "25000000"})
    );
    // The exact vertex (4/3, 4/3) — a point no f64 holds.
    assert_eq!(
        cert["witnesses"]["feasible_witness"]["xhat"],
        serde_json::json!([{"num":"4","den":"3"}, {"num":"4","den":"3"}])
    );
    // Nothing about optimality is claimed, so no optimality witness is shipped.
    assert!(cert["witnesses"]["duals"].is_null());
    assert!(cert["witnesses"]["hessian_psd"].is_null());
}

#[test]
fn cert_verify_accepts_the_feasible_certificate() {
    let out = cert_verify("certify_feasible.nl", "certify_feasible.cert.json");
    assert!(
        out.status.success(),
        "feasible cert should verify against its .nl: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The flag is not a universal weakening. Where feasibility is vacuous or
/// already contradicted, asking for it gets an explanation, not a certificate.
#[test]
fn the_feasible_flag_is_refused_where_it_would_prove_nothing() {
    let out = Command::new(pounce_exe())
        .arg("certify")
        .arg("--feasible")
        .arg(fixture("certify_sos.nl"))
        .arg(solve_to_sol("certify_sos"))
        .output()
        .expect("run pounce certify --feasible");
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("vacuous"),
        "an unconstrained polynomial should be refused as vacuous, got: {err}"
    );
}
