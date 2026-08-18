//! Options naming features pounce does not implement are refused
//! (gh#483 follow-up, continuing #191).
//!
//! `upstream_options.rs` registers every name Ipopt registers, so an
//! `ipopt.opt` written for Ipopt parses unchanged — a real compatibility
//! benefit that also turned ~200 knobs into silent no-ops, because
//! registering an option says nothing about implementing it.
//!
//! #191 fixed the half where the feature runs and only the read site was
//! missing, and explicitly scoped out "feature genuinely unimplemented —
//! expected no-ops". This is that other half.
//!
//! The table's membership rules, and why an explicitly-set *default* is
//! still allowed, live in `pounce-algorithm/src/unimplemented_options.rs`
//! alongside the unit tests for the predicate. What is checked here is
//! the CLI's end of it: the exit code, that the message reaches stderr,
//! and — the part only an end-to-end test can show — that the guard runs
//! before solver routing.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn run(fixture_name: &str, tag: &str, opts: &[&str]) -> (Option<i32>, String) {
    let dir = std::env::temp_dir().join(format!("pounce_unimplopt_{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join(fixture_name);
    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixture.push("tests/fixtures");
    fixture.push(fixture_name);
    std::fs::copy(&fixture, &nl).expect("copy fixture");
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .args(opts)
        .arg("print_level=0")
        .output()
        .expect("run pounce");
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// One representative per feature group, so a group dropped from the
/// table fails here rather than going quiet again.
#[test]
fn requesting_an_unimplemented_feature_fails_with_an_explanation() {
    for (i, (opt, needle)) in [
        ("penalty_init_max=42", "CG-penalty"),
        (
            "gradient_approximation=finite-difference-values",
            "finite differences",
        ),
        ("dependency_detector=mumps", "linear-dependency detection"),
        ("check_derivatives_for_naninf=yes", "NaN/Inf"),
        ("magic_steps=yes", "magic steps"),
        ("suppress_all_output=yes", "output controls"),
        ("hsllib=libcoinhsl.so", "HSL loader"),
    ]
    .into_iter()
    .enumerate()
    {
        let (code, err) = run("user_scaling_suffix.nl", &format!("g{i}"), &[opt]);
        assert_eq!(code, Some(2), "`{opt}` should fail; stderr:\n{err}");
        assert!(
            err.contains(needle),
            "`{opt}` should mention `{needle}`; stderr:\n{err}",
        );
        assert!(err.contains("483"), "stderr:\n{err}");
    }
}

/// The message names the option, not just the feature — with ~200
/// registered names, "some option is unsupported" would be useless.
#[test]
fn the_refusal_names_the_offending_option() {
    let (_, err) = run("user_scaling_suffix.nl", "named", &["vartheta=0.9"]);
    assert!(err.contains("`vartheta`"), "stderr:\n{err}");
}

/// Setting an option to its registered default asks for nothing. A
/// generated `ipopt.opt` spells out defaults, and failing on that would
/// break the compatibility the registry exists to provide.
#[test]
fn explicitly_setting_a_default_still_solves() {
    for (i, opt) in ["dependency_detector=none", "magic_steps=no", "recalc_y=no"]
        .into_iter()
        .enumerate()
    {
        let (code, err) = run("user_scaling_suffix.nl", &format!("d{i}"), &[opt]);
        assert_eq!(code, Some(0), "`{opt}` asks for nothing; stderr:\n{err}");
        assert!(!err.contains("does not implement"), "stderr:\n{err}");
    }
}

/// Options whose feature *runs* and only whose read site is missing must
/// keep solving — refusing them would fail solves that are correct
/// today. Wiring them is separate work.
#[test]
fn knobs_on_implemented_features_still_solve() {
    for (i, opt) in [
        "max_resto_iter=17",
        "accept_after_max_steps=3",
        "limited_memory_max_skipping=4",
        "corrector_type=affine",
        // #677: `recalc_y` was refused as unimplemented until the
        // least-square multiplier recalculation landed. It is a real
        // feature now, so asking for it must solve rather than fail.
        "recalc_y=yes",
        "recalc_y_feas_tol=1e-4",
    ]
    .into_iter()
    .enumerate()
    {
        let (code, err) = run("user_scaling_suffix.nl", &format!("b{i}"), &[opt]);
        assert_eq!(
            code,
            Some(0),
            "`{opt}` configures an implemented feature; stderr:\n{err}",
        );
    }
}

/// A caching hint warns and solves: ignoring it costs evaluations, never
/// correctness, so blocking the run would be a worse trade than the
/// silence was.
#[test]
fn a_caching_hint_warns_but_solves() {
    let (code, err) = run("user_scaling_suffix.nl", "hint", &["hessian_constant=yes"]);
    assert_eq!(code, Some(0), "stderr:\n{err}");
    assert!(err.contains("warning"), "stderr:\n{err}");
    assert!(err.contains("hessian_constant"), "stderr:\n{err}");
}

/// A knob for a linear-solver backend pounce does not ship warns and
/// solves, same trade as the caching hint above: pounce factors with
/// `feral` or MA57, so an `ma97_*` value could not have changed the
/// answer even in principle, and refusing it would fail a portable
/// `ipopt.opt` that configures several backends so one file runs
/// everywhere — the compatibility the registry exists to provide.
/// gh#551 section 2.
#[test]
fn a_backend_knob_warns_but_solves() {
    let (code, err) = run("user_scaling_suffix.nl", "backend", &["ma97_order=metis"]);
    assert_eq!(code, Some(0), "stderr:\n{err}");
    assert!(err.contains("warning:"), "stderr:\n{err}");
    assert!(err.contains("`ma97_order`"), "stderr:\n{err}");
    assert!(err.contains("MA97"), "the backend is named; stderr:\n{err}");
    assert!(
        err.contains("result is unaffected"),
        "the warning must say the answer is not at risk; stderr:\n{err}",
    );
    assert!(err.contains("551"), "stderr:\n{err}");
}

/// One line per backend family — an MA97-tuned file sets a dozen
/// `ma97_*` knobs, and a dozen near-identical lines is noise a reader
/// learns to skip, which is silence with extra steps.
///
/// The exact-count assertion also pins the *other* way this could
/// double: the CLI emits before routing (a convex model never reaches
/// `optimize_tnlp`) and `optimize_tnlp` emits for every other frontend,
/// so a CLI run passes both sites and must still print once.
#[test]
fn backend_knobs_warn_once_per_family() {
    let (code, err) = run(
        "user_scaling_suffix.nl",
        "backendgroup",
        &["ma97_order=metis", "ma97_u=1e-4", "pardiso_msglvl=1"],
    );
    assert_eq!(code, Some(0), "stderr:\n{err}");
    assert_eq!(
        err.matches("ma97_order").count(),
        1,
        "one grouped line, not one per option; stderr:\n{err}",
    );
    assert!(err.contains("`ma97_u`"), "stderr:\n{err}");
    assert!(err.contains("Pardiso"), "stderr:\n{err}");
}

/// A backend knob left at its registered default asks for nothing, so
/// it must not even warn — the same gate the refusal table uses.
#[test]
fn a_backend_knob_at_its_default_is_silent() {
    let (code, err) = run("user_scaling_suffix.nl", "backenddef", &["ma97_order=auto"]);
    assert_eq!(code, Some(0), "stderr:\n{err}");
    assert!(!err.contains("warning:"), "stderr:\n{err}");
}

/// The guard runs before routing: a convex-QP model dispatches to
/// `pounce-convex` and never reaches the library-side guard.
#[test]
fn the_refusal_covers_the_convex_route() {
    let (code, err) = run("boxed_qp_min.nl", "convex", &["magic_steps=yes"]);
    assert_eq!(code, Some(2), "stderr:\n{err}");
    assert!(err.contains("magic steps"), "stderr:\n{err}");
}

/// A plain run is untouched — no refusal, no warning.
#[test]
fn a_default_run_is_silent() {
    let (code, err) = run("user_scaling_suffix.nl", "plain", &[]);
    assert_eq!(code, Some(0), "stderr:\n{err}");
    assert!(!err.contains("does not implement"), "stderr:\n{err}");
    assert!(!err.contains("warning:"), "stderr:\n{err}");
}
