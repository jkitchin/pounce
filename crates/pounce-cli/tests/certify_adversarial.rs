//! Adversarial tests **across** `pounce certify`'s features, rather than within
//! one of them.
//!
//! `tests/certify.rs` checks that each verdict is emitted correctly, and
//! `scripts/check-lean-cert.sh`'s forged fixtures check that each *obligation*
//! a verdict rests on can reject on its own. Both are per-feature. Neither one
//! probes the seams: what happens when two features compose, when a flag is
//! pushed to a boundary where its claim stops being true, or when a certificate
//! is offered a problem it was not written about.
//!
//! Those seams are where a soundness bug would actually live. An obligation
//! that is individually airtight can still be attached to the wrong problem, or
//! silently skipped when a second flag changes the code path. So the tests here
//! are written as *invariants over the whole feature set* — properties that
//! must hold for every fixture and every parameter, not assertions about one
//! expected output.
//!
//! The governing principle throughout: the emitter is allowed to **refuse**,
//! and refusing is never a failure. What it is never allowed to do is emit a
//! certificate whose claim is false, or weaker than the verdict printed on it.

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

/// Solve `<stem>.nl` to produce `<stem>.sol`. Mirrors `certify.rs`'s helper,
/// including its tolerance of a nonzero exit: an INFEASIBLE or UNBOUNDED solve
/// exits nonzero while still writing the `.sol` those verdicts certify.
fn solve_to_sol(stem: &str) -> PathBuf {
    let _ = Command::new(pounce_exe())
        .arg(fixture(&format!("{stem}.nl")))
        .output()
        .expect("run pounce solve");
    let sol = fixture(&format!("{stem}.sol"));
    assert!(sol.exists(), "solve did not write {}", sol.display());
    sol
}

/// Run `pounce certify` with arbitrary flags; return (exit code, stdout).
fn certify(stem: &str, flags: &[&str]) -> (Option<i32>, String) {
    let sol = solve_to_sol(stem);
    let mut cmd = Command::new(pounce_exe());
    cmd.arg("certify");
    for f in flags {
        cmd.arg(f);
    }
    cmd.arg(fixture(&format!("{stem}.nl"))).arg(&sol);
    let out = cmd.output().expect("run pounce certify");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

fn rat(v: &serde_json::Value) -> f64 {
    let num: f64 = v["num"].as_str().expect("num").parse().expect("rational");
    let den: f64 = v["den"].as_str().expect("den").parse().expect("rational");
    num / den
}

/// Evaluate the certificate's own `problem.polynomial` at `x`, in f64.
///
/// Deliberately re-read from the certificate rather than hard-coded per
/// fixture: the property under test is that the *emitted* bound is true of the
/// *emitted* polynomial on the *emitted* ball. A hand-written `p` would test a
/// polynomial the certificate never claimed anything about.
fn eval_poly(cert: &serde_json::Value, x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for t in cert["problem"]["polynomial"]["terms"]
        .as_array()
        .expect("terms")
    {
        // `exponents` is dense: one entry per variable, in variable order.
        let mut term = rat(&t["coeff"]);
        for (i, e) in t["exponents"]
            .as_array()
            .expect("exponents")
            .iter()
            .enumerate()
        {
            term *= x[i].powi(e.as_i64().expect("exponent") as i32);
        }
        acc += term;
    }
    acc
}

/// **Every certificate binds to exactly the problem it was written about.**
///
/// The cross product of every non-forged certificate against every fixture
/// `.nl`. The diagonal must verify; everything off it must be rejected.
///
/// This is the check a matching `nl_sha256` alone cannot make. A hash says the
/// bytes are the ones the emitter saw; it says nothing about whether the
/// `problem` block inside the certificate describes those bytes. `cert-verify`
/// re-derives the problem from the consumer's own `.nl` and compares, and this
/// test is what proves that comparison actually discriminates — a re-derivation
/// that quietly matched everything would pass every single-fixture test in the
/// suite while binding nothing.
///
/// It scales with the fixture set on purpose: a new verdict added without a
/// corresponding tightening of the re-derivation shows up here as an
/// off-diagonal accept, with no new test to write.
#[test]
fn every_certificate_binds_to_exactly_its_own_problem() {
    let dir = fixture("");
    let mut bases: Vec<String> = std::fs::read_dir(&dir)
        .expect("read fixtures")
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            let base = name.strip_suffix(".cert.json")?.to_string();
            // Forged certs are the *other* guard's business: they are supposed
            // to describe their problem correctly and lie about the witnesses.
            (!base.contains("forged") && fixture(&format!("{base}.nl")).exists()).then_some(base)
        })
        .collect();
    bases.sort();
    assert!(
        bases.len() >= 12,
        "expected the full fixture set, got {bases:?}"
    );

    // Two legitimate off-diagonal pairs, and they are asserted rather than
    // tolerated. Each `_strict` fixture's `.nl` is a byte copy of its base's —
    // the strict fixtures exist to hold the *problem* fixed while the claim
    // changes, so that any diff between their goldens is purely the growth
    // construction. Both members of a pair therefore describe one problem, and
    // each cert really is about the other's `.nl`.
    //
    // Worth being precise about what this shows: `cert-verify` binds a
    // certificate to a *problem*, and it does that by re-deriving the problem
    // from the consumer's own `.nl`. The neighborhood is not re-derivable —
    // it is the certifier's choice of where to make its claim, not a fact
    // about the model — so it is deliberately outside what this check compares.
    // `certify_sos_box` (no ball, global) and `certify_sos_box_strict` (a ball)
    // cross-accept for exactly that reason. Coherence *between* the ball and
    // the verdict is a different property, checked by
    // `the_region_and_the_verdict_never_disagree` below on the producer side
    // and by pounce-lean's `check_refusals.py` on the consumer side.
    let identical = |a: &str, b: &str| {
        let pair = |x: &str, y: &str| a == x && b == y;
        pair("certify_sos_local", "certify_sos_local_strict")
            || pair("certify_sos_local_strict", "certify_sos_local")
            || pair("certify_sos_box", "certify_sos_box_strict")
            || pair("certify_sos_box_strict", "certify_sos_box")
    };

    let mut wrong: Vec<String> = Vec::new();
    for c in &bases {
        for n in &bases {
            let ok = Command::new(pounce_exe())
                .arg("cert-verify")
                .arg(fixture(&format!("{n}.nl")))
                .arg(fixture(&format!("{c}.cert.json")))
                .output()
                .expect("run cert-verify")
                .status
                .success();
            let expected = c == n || identical(c, n);
            if ok != expected {
                wrong.push(format!(
                    "cert {c} vs {n}.nl: {} (expected {})",
                    if ok { "ACCEPTED" } else { "rejected" },
                    if expected { "accept" } else { "reject" }
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} pairings bound incorrectly:\n  {}",
        wrong.len(),
        bases.len() * bases.len(),
        wrong.join("\n  ")
    );
}

/// **Widening the ball must weaken the claim, never falsify it.**
///
/// `--radius` is the one knob that changes *what is true*, which makes it the
/// most dangerous flag in the set. `3x⁴ + 4x³ − 12x²` has a local minimum at
/// `x = 1` worth `−5` and the global one at `x = −2` worth `−32`. A ball around
/// `x = 1` big enough to swallow `x = −2` makes the claim `−5 ≤ p` false — so a
/// bug that kept emitting `−5` while widening the neighborhood would produce a
/// certificate that is wrong rather than merely weak.
///
/// The invariant is checked against the certificate's own data: whatever ball
/// it names, its bound must actually hold everywhere in it, and if it claims
/// attainment the candidate must actually attain it and actually be inside.
/// Sampling is f64 with a slack, which is fine — the exact statement is the
/// Lean layer's job. What this catches is the *gross* error, and a bound that
/// is false on a ball is gross.
///
/// Refusal at any radius is a pass. The emitter refusing to certify is the
/// designed response to a relaxation that will not close, and it is what
/// actually happens at radius 3 and 5, where the true bound drops to `−32`.
#[test]
fn no_radius_produces_a_bound_that_is_false_on_its_own_ball() {
    // First the degenerate end. A ball with `r ≤ 0` is empty or a point, and a
    // bound quantified over it says nothing at all while still reading as a
    // `local-min` — the one way this flag can produce a *vacuous* claim rather
    // than a false one. Both must be refused before the flag is parsed as a
    // number the emitter would go on to square.
    for bad in ["0", "-1", "-0.5"] {
        let (code, _) = certify("certify_sos_local", &["--local", "--radius", bad]);
        assert_eq!(code, Some(2), "--radius {bad} must be refused as vacuous");
    }

    let mut certified = 0;
    for radius in ["0.25", "0.5", "1", "1.5", "2", "3", "5"] {
        let (code, stdout) = certify("certify_sos_local", &["--local", "--radius", radius]);
        if code != Some(0) {
            continue; // refusing is always sound
        }
        certified += 1;
        let cert: serde_json::Value =
            serde_json::from_str(&stdout).expect("certify stdout is JSON");

        let gamma = rat(&cert["bound"]);
        let nb = &cert["problem"]["neighborhood"];
        let c = rat(&nb["center"][0]);
        let rsq = rat(&nb["radius_sq"]);
        let r = rsq.sqrt();

        // The bound must hold at every point of the ball it names.
        const N: i32 = 4001;
        for k in 0..=N {
            let x = c - r + 2.0 * r * (k as f64) / (N as f64);
            let p = eval_poly(&cert, &[x]);
            assert!(
                p >= gamma - 1e-9,
                "radius {radius}: certified γ = {gamma} but p({x}) = {p} inside the ball \
                 [{}, {}] — the bound is FALSE where the certificate claims it",
                c - r,
                c + r
            );
        }

        // An attained bound must be attained, and attained *inside*.
        if cert["verdict"] == "local-min" || cert["verdict"] == "local-min-strict" {
            let x0 = rat(&cert["candidate"]["x"][0]);
            assert!(
                (eval_poly(&cert, &[x0]) - gamma).abs() < 1e-9,
                "radius {radius}: verdict claims attainment at {x0}, but p({x0}) ≠ γ"
            );
            assert!(
                (x0 - c) * (x0 - c) <= rsq + 1e-12,
                "radius {radius}: candidate {x0} is outside the ball it is claimed to minimize"
            );
        }
    }
    assert!(
        certified >= 2,
        "every radius was refused — the test proved nothing"
    );
}

/// **A minimum too flat for any modulus degrades to `local-min`.**
///
/// The *other* way `--growth` can fail to deliver, and a different one from the
/// test above. There the bound was never attained, so there was no `x₀` to grow
/// from. Here `x⁴` attains `0` at `x = 0` exactly — the candidate is perfect —
/// and the growth claim is still false: `μx² ≤ x⁴` fails near the origin for
/// every `μ > 0`, so every rung of the ladder must fail and the right answer is
/// the non-strict verdict.
///
/// This is the case where an "almost worked" heuristic would be most tempting
/// and most wrong. The ladder's smallest rung is `1/4096`, and a modulus that
/// small looks like nothing — but `x⁴ ≥ x²/4096` is still false on `|x| < 1/64`,
/// which is inside the ball. A fallback that shipped the last rung tried, or
/// that treated a near-miss in the rounding as success, would emit a strict
/// claim that is plainly false, and it would look entirely reasonable in the
/// JSON. The only thing standing there is that the relaxation genuinely does
/// not close; this pins that the emitter believes it.
///
/// `pounce certify --help` promises exactly this ("A minimum too flat to
/// certify falls back to `local-min`"), which makes it a documented behaviour
/// with no test — the kind that quietly stops being true.
#[test]
fn growth_degrades_on_a_minimum_that_is_attained_but_flat() {
    let (code, stdout) = certify("certify_growth_flat", &["--local", "--growth"]);
    assert_eq!(code, Some(0), "the non-strict claim is still provable here");
    let cert: serde_json::Value = serde_json::from_str(&stdout).expect("JSON");
    assert_eq!(
        cert["verdict"], "local-min",
        "x⁴ has no quadratic growth at 0; the strict verdict would be false"
    );
    assert!(
        cert.get("growth_modulus").is_none(),
        "no modulus may ride along on a non-strict verdict"
    );
    // The candidate really is exact and really does attain — i.e. the
    // degradation is about growth alone, not about a candidate that failed for
    // some unrelated reason and made this test vacuous.
    assert_eq!(rat(&cert["candidate"]["x"][0]), 0.0);
    assert_eq!(rat(&cert["bound"]), 0.0);
    assert_eq!(rat(&cert["candidate"]["objective"]), 0.0);
}

/// **Growth composes with Putinar multipliers.**
///
/// `--growth` was developed against a ball and no constraints, where the
/// feasibility obligation on the candidate is vacuous (`Fin 0`). Adding
/// localizing blocks changes the shape of the identity the shifted objective
/// has to satisfy *and* makes that obligation real. Two features that each work
/// alone are not the same as two that work together, and this is the pairing
/// where a shift applied to the wrong side of the identity would survive the
/// unconstrained fixture untouched.
#[test]
fn growth_composes_with_constraints() {
    let (code, stdout) = certify("certify_sos_box", &["--local", "--growth"]);
    assert_eq!(code, Some(0), "constrained + local + growth should certify");
    let cert: serde_json::Value = serde_json::from_str(&stdout).expect("JSON");
    assert_eq!(cert["verdict"], "local-min-strict");
    assert!(
        rat(&cert["growth_modulus"]) > 0.0,
        "a strict verdict without a positive modulus is the one shape that must not exist"
    );
    let ncon = cert["problem"]["poly_constraints"]
        .as_array()
        .expect("poly_constraints")
        .len();
    assert!(
        ncon >= 2,
        "this fixture is only interesting because it HAS constraints; got {ncon}"
    );
    // …and the multiplier family is still constraints-then-ball, with the ball
    // last. A growth shift that disturbed that ordering would misattribute
    // every localizing block by one.
    let sos = cert["witnesses"]["sos"].as_array().expect("sos blocks");
    assert_eq!(sos.len(), ncon + 2, "σ₀, one per constraint, then the ball");
    assert!(sos[0].get("multiplier").is_none(), "σ₀ has no multiplier");
    for (k, blk) in sos[1..].iter().enumerate() {
        assert_eq!(
            blk["multiplier"], k as u64,
            "multiplier indices must stay dense and ordered; the ball is last"
        );
    }
}

/// **Asking for more than is provable degrades the verdict; it does not
/// fabricate one.**
///
/// `x⁴ − 3x² + 2` minimizes at `±√(3/2)`, which no rational point reaches, so
/// there is no `x₀` for growth to be measured *from*. The honest output is the
/// bound alone. The failure mode this excludes is the tempting one: inventing a
/// nearby rational candidate so the requested stronger verdict can be emitted.
///
/// Note what is asserted — not just that the verdict is weaker, but that the
/// modulus is *absent*. A `local-lower-bound` carrying a stray `growth_modulus`
/// would be internally inconsistent, and the Lean codegen refuses exactly that
/// shape; this keeps the producer from ever constructing it.
#[test]
fn growth_degrades_rather_than_inventing_a_minimizer() {
    let (code, stdout) = certify("certify_sos_bound", &["--local", "--growth"]);
    assert_eq!(code, Some(0));
    let cert: serde_json::Value = serde_json::from_str(&stdout).expect("JSON");
    assert_eq!(
        cert["verdict"], "local-lower-bound",
        "with no rational minimizer the honest verdict is a bound"
    );
    assert!(
        cert.get("growth_modulus").is_none(),
        "a bound-only verdict must not carry a growth modulus: no point to grow from"
    );
    assert!(
        cert.get("candidate").is_none() || cert["candidate"].is_null(),
        "and no candidate either"
    );
}

/// **A refusal leaves nothing behind.**
///
/// Every way `certify` can decline — an unsupported slice, a flag combination
/// that names no claim, a relaxation that will not close — must leave `-o`'s
/// path untouched. A partially written or stale certificate on disk is worse
/// than no certificate: exit codes get lost in scripts, and a file that exists
/// reads as a file that succeeded. The consumer would then run `cert-verify` and
/// `lake build` against a certificate the producer never actually stood behind.
///
/// Checked across the refusal *paths*, not one of them, because they exit from
/// different places in the emitter and only some of them run after the output
/// path has been opened.
#[test]
fn a_refused_certification_writes_no_output_file() {
    let scratch = std::env::temp_dir().join(format!("pounce-cert-adv-out-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).expect("create scratch dir");

    let refusals: &[(&str, &[&str])] = &[
        ("certify_sos_local", &["--growth"]), // flag needs --local
        ("certify_sos_local", &["--radius", "0.5"]), // flag needs --local
        ("certify_sos_local", &["--local", "--radius", "5"]), // relaxation will not close
        ("certify_sos_local", &["--feasible"]), // meaningless on this slice
        ("certify_maximize", &[]),            // slice is refused outright
        ("certify_qp", &["--local"]),         // already global
    ];

    for (i, (stem, flags)) in refusals.iter().enumerate() {
        let out = scratch.join(format!("c{i}.json"));
        assert!(!out.exists());
        let sol = solve_to_sol(stem);
        let mut cmd = Command::new(pounce_exe());
        cmd.arg("certify");
        for f in *flags {
            cmd.arg(f);
        }
        let st = cmd
            .arg("-o")
            .arg(&out)
            .arg(fixture(&format!("{stem}.nl")))
            .arg(&sol)
            .output()
            .expect("run pounce certify");
        assert_eq!(
            st.status.code(),
            Some(2),
            "{stem} {flags:?} was expected to be refused"
        );
        assert!(
            !out.exists(),
            "{stem} {flags:?} was refused but still wrote {}",
            out.display()
        );
    }
    let _ = std::fs::remove_dir_all(&scratch);
}

/// **The flag surface, exhaustively.**
///
/// Every subset of `{--local, --growth, --feasible}` (plus `--radius`, which is
/// only meaningful with `--local`), on both a polynomial and a convex QP. Each
/// combination must either be refused with exit 2 or produce exactly the
/// verdict that combination *names* — never a third thing.
///
/// The specific hazard: flags are parsed independently but interact in the
/// emitter, so a combination nobody wrote a test for can be accepted with one
/// flag silently ignored. That failure is invisible — the output is a valid
/// certificate, just not the one that was asked for, and a user who passed
/// `--growth` and got `local-min` has no way to tell it was dropped rather than
/// unprovable. Enumerating the surface is the only way to see it.
#[test]
fn every_flag_combination_is_refused_or_delivers_what_it_names() {
    // (flags, polynomial fixture verdict, convex-QP verdict); None = must refuse.
    let cases: &[(&[&str], Option<&str>, Option<&str>)] = &[
        (&[], Some("global-lower-bound"), Some("global-min")),
        (&["--local"], Some("local-min"), None),
        (&["--local", "--growth"], Some("local-min-strict"), None),
        // --growth is a *request*, not a demand, but only where its region
        // exists. Without --local there is no ball, so it is refused rather
        // than downgraded — a downgrade would answer a different question.
        (&["--growth"], None, None),
        (&["--feasible"], None, Some("feasible")),
        // Mutually exclusive claims. Guessing which one the user meant would
        // deliver a claim they did not ask for.
        (&["--local", "--feasible"], None, None),
        (&["--growth", "--feasible"], None, None),
        (&["--local", "--growth", "--feasible"], None, None),
        // --radius names the ball; without --local there is no ball to name.
        (&["--radius", "0.5"], None, None),
    ];

    for (flags, want_poly, want_qp) in cases {
        for (stem, want) in [("certify_sos_local", want_poly), ("certify_qp", want_qp)] {
            let (code, stdout) = certify(stem, flags);
            match want {
                None => assert_eq!(
                    code,
                    Some(2),
                    "{flags:?} on {stem} should be refused, got exit {code:?}: {stdout}"
                ),
                Some(verdict) => {
                    assert_eq!(
                        code,
                        Some(0),
                        "{flags:?} on {stem} should certify, got exit {code:?}"
                    );
                    let cert: serde_json::Value = serde_json::from_str(&stdout).expect("JSON");
                    assert_eq!(
                        cert["verdict"], *verdict,
                        "{flags:?} on {stem}: a flag was accepted and then ignored"
                    );
                }
            }
        }
    }
}

/// **A neighborhood and a global verdict can never appear together.**
///
/// These two fields are the certificate's only statement of *where* its claim
/// holds, and they are read by different consumers: `cert-verify` compares the
/// neighborhood, the Lean codegen picks its theorem from the verdict. A
/// certificate carrying a ball under a `global-` verdict would verify against
/// its `.nl` and then lower to a theorem about all of ℝⁿ — the single most
/// damaging inconsistency this schema can express, because both halves are
/// individually well-formed.
///
/// The producer must never construct it; the codegen refuses it on the other
/// side (`pounce-lean`'s `check_refusals.py`). This is the producer half.
#[test]
fn the_region_and_the_verdict_never_disagree() {
    for stem in [
        "certify_sos",
        "certify_sos_bound",
        "certify_sos_box",
        "certify_sos_local",
    ] {
        for flags in [&[][..], &["--local"][..], &["--local", "--growth"][..]] {
            let (code, stdout) = certify(stem, flags);
            if code != Some(0) {
                continue;
            }
            let cert: serde_json::Value = serde_json::from_str(&stdout).expect("JSON");
            let verdict = cert["verdict"].as_str().expect("verdict");
            let has_ball = !cert["problem"]["neighborhood"].is_null();
            assert_eq!(
                has_ball,
                verdict.starts_with("local-"),
                "{stem} {flags:?}: verdict {verdict:?} and neighborhood presence \
                 ({has_ball}) describe different regions"
            );
            let has_mu = cert.get("growth_modulus").is_some_and(|v| !v.is_null());
            assert_eq!(
                has_mu,
                verdict == "local-min-strict",
                "{stem} {flags:?}: verdict {verdict:?} and the growth modulus \
                 disagree about whether this is a strict claim"
            );
        }
    }
}

/// **Certification never rewrites the answer it is certifying.**
///
/// `certify` reads a `.sol` and emits a certificate; it must not change what
/// the solver reported. The concrete risk is the exact refinement path, which
/// *does* move the point — from the float `x*` to a nearby exact rational — and
/// could plausibly be wired back into the reported solution. If it ever were,
/// `pounce solve` would start returning different numbers depending on whether
/// a certificate was requested, which is a silent behavioural change for every
/// existing user of the `.sol` file.
#[test]
fn certifying_does_not_mutate_the_sol_file() {
    // Work on a private copy. Every other test in the suite re-solves fixtures
    // in place, so watching the shared `.sol` would be watching a file three
    // other threads are rewriting — the test would flake on that, not on the
    // property.
    let scratch = std::env::temp_dir().join(format!(
        "pounce-cert-adv-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&scratch).expect("create scratch dir");
    let nl = scratch.join("p.nl");
    let sol = scratch.join("p.sol");
    std::fs::copy(fixture("certify_sos_local.nl"), &nl).expect("copy .nl");
    Command::new(pounce_exe()).arg(&nl).output().expect("solve");
    assert!(sol.exists(), "solve did not write {}", sol.display());

    let before = std::fs::read(&sol).expect("read .sol");
    for flags in [&["--local"][..], &["--local", "--growth"][..]] {
        let mut cmd = Command::new(pounce_exe());
        cmd.arg("certify");
        for f in flags {
            cmd.arg(f);
        }
        let out = cmd.arg(&nl).arg(&sol).output().expect("run pounce certify");
        assert!(out.status.success(), "certify {flags:?} failed");
    }
    let after = std::fs::read(&sol).expect("read .sol");
    assert_eq!(
        before, after,
        "certify rewrote the .sol — the solver's reported answer must be untouched"
    );
    let _ = std::fs::remove_dir_all(&scratch);
}
