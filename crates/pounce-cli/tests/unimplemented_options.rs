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

/// #551 / #677 round 3: the corrector knobs and the three restoration /
/// L-BFGS sub-capabilities. Separate from the loop above because these
/// carry their own tracking issue, and because the point of the last
/// three is the *shape* of the message — a user who set one of them has
/// to be able to tell "restoration does not run here" (which would be
/// false) from "restoration runs, this part of it does not".
#[test]
fn the_corrector_and_resto_sub_capability_refusals_explain_themselves() {
    for (i, (opt, needle)) in [
        ("corrector_type=affine", "TryCorrector"),
        ("skip_corr_if_neg_curv=no", "corrector step"),
        ("corrector_compl_avrg_red_fact=2.0", "corrector step"),
        (
            "expect_infeasible_problem_ctol=1e-4",
            "restoration phase itself runs",
        ),
        (
            "limited_memory_special_for_resto=yes",
            "L-BFGS runs in the restoration sub-solve",
        ),
        (
            "resto_failure_feasibility_threshold=1e-6",
            "restoration runs",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let (code, err) = run("user_scaling_suffix.nl", &format!("c{i}"), &[opt]);
        assert_eq!(code, Some(2), "`{opt}` should fail; stderr:\n{err}");
        assert!(
            err.contains(needle),
            "`{opt}` should mention `{needle}`; stderr:\n{err}",
        );
        assert!(err.contains("551"), "stderr:\n{err}");
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
        // #551/#677: the successive-restoration cap is a real field
        // (`RestoConvCheckAdapter::maximum_resto_iters`) and the option
        // now sets it, so asking for it must solve rather than fail.
        "max_resto_iter=17",
        "accept_after_max_steps=3",
        "limited_memory_max_skipping=4",
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
