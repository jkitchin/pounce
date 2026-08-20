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
    // (option, a phrase from its feature, the issue its group names)
    for (i, (opt, needle, issue)) in [
        ("penalty_init_max=42", "CG-penalty", "483"),
        (
            "gradient_approximation=finite-difference-values",
            "finite differences",
            "483",
        ),
        (
            "dependency_detector=mumps",
            "linear-dependency detection",
            "483",
        ),
        ("check_derivatives_for_naninf=yes", "NaN/Inf", "483"),
        ("magic_steps=yes", "magic steps", "483"),
        // #551: the two line-search knobs whose *feature* is missing.
        // `theta_min` is the CG-penalty acceptor's threshold, not the
        // filter's (the filter derives its own from `theta_min_fact`);
        // `alpha_for_y_tol` only configures the `primal-and-full` /
        // `dual-and-full` multiplier-step rules, which pounce does not
        // have.
        ("theta_min=1e-5", "CG-penalty acceptor", "551"),
        ("alpha_for_y_tol=1e-3", "primal-and-full", "551"),
        ("suppress_all_output=yes", "output controls", "483"),
        ("hsllib=libcoinhsl.so", "HSL loader", "483"),
        // #551/#677: pounce computes the single `sens_state_1`
        // perturbation tier; upstream's further tiers do not exist here.
        ("n_sens_steps=3", "perturbation tier", "677"),
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
        // Every entry names the issue its own group tracks, rather
        // than an `||` over the issues in the table: a wrong-issue
        // message would otherwise pass by borrowing a sibling's number.
        assert!(err.contains(issue), "stderr:\n{err}");
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

/// The four constant-derivative hints used to warn "pounce does not
/// exploit this" and re-evaluate anyway. gh #588 Q6 exploits them, and
/// this pin flips with it: what a hint earns now is decided by the
/// model's own algebra, and what it earns is *measured in evaluations*
/// rather than asserted from a warning string.
///
/// `user_scaling_suffix.nl` has a `∇²L` POUNCE proves is not constant,
/// so asserting `hessian_constant=yes` on it is the case upstream Ipopt
/// honours — reusing a Hessian that genuinely moves — and the case
/// POUNCE refuses. The solve must still succeed: the option is ignored,
/// not fatal.
#[test]
fn a_disproved_caching_hint_is_refused_with_a_warning() {
    let (code, err) = run("user_scaling_suffix.nl", "hint", &["hessian_constant=yes"]);
    assert_eq!(code, Some(0), "stderr:\n{err}");
    assert!(err.contains("warning"), "stderr:\n{err}");
    assert!(err.contains("hessian_constant"), "stderr:\n{err}");
    assert!(
        err.contains("ignoring"),
        "the warning must say the hint was refused, not merely unused; \
         stderr:\n{err}"
    );
}

/// The other half of the flip, and the half a string match cannot reach:
/// a model whose `∇²L` and Jacobian POUNCE *proves* constant is evaluated
/// **once**, with no option set at all.
///
/// `nonconvex_qp.nl` is a quadratic objective over linear rows forced
/// down the NLP path (the convex-QP route would not exercise this), so
/// `∇²L = σ∇²f` and every row's gradient is a constant vector. The
/// assertion is `num_hess_evals == 1` against an iteration count well
/// above 1 — before Q6 both counters tracked the iterations.
#[test]
fn a_proved_constant_hessian_is_evaluated_once() {
    let dir = std::env::temp_dir().join("pounce_unimplopt_constderiv");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join("nonconvex_qp.nl");
    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixture.push("tests/fixtures/nonconvex_qp.nl");
    std::fs::copy(&fixture, &nl).expect("copy fixture");
    let json = dir.join("out.json");
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .arg("--json-output")
        .arg(&json)
        .arg("print_level=0")
        .output()
        .expect("run pounce");
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&json).expect("json report");
    let _ = std::fs::remove_dir_all(&dir);

    // Small, flat JSON: pull the three integers out without a parser
    // dependency this test crate does not otherwise have.
    fn field(text: &str, key: &str) -> i64 {
        let at = text
            .find(&format!("\"{key}\""))
            .unwrap_or_else(|| panic!("`{key}` missing from report:\n{text}"));
        let rest = &text[at + key.len() + 2..];
        let start = rest.find(':').expect("colon") + 1;
        let end = rest[start..]
            .find([',', '}'])
            .map(|e| start + e)
            .unwrap_or(rest.len());
        rest[start..end].trim().parse().expect("integer field")
    }
    let iters = field(&text, "iteration_count");
    let hess = field(&text, "num_hess_evals");
    let jac = field(&text, "num_constr_jac_evals");
    assert!(iters > 1, "expected a multi-iteration solve, got {iters}");
    assert_eq!(
        hess, 1,
        "`∇²L` is provably constant on this model; it must be evaluated \
         once and reused for all {iters} iterations"
    );
    assert_eq!(
        jac, 1,
        "every row is linear; the Jacobian must be evaluated once (got \
         {jac} over {iters} iterations)"
    );
}

/// A knob for a linear-solver backend pounce does not ship warns and
/// solves, same trade as the caching hint above: pounce factors with
/// `feral` or MA57, so an `ma97_*` value could not have changed the
/// answer even in principle, and refusing it would fail a portable
/// `ipopt.opt` that configures several backends so one file runs
/// everywhere — the compatibility the registry exists to provide.
/// gh#551 section 2.
///
/// `tol` is here to make the premise explicit rather than incidental:
/// the warning is what a file with *other content* earns, and a file of
/// nothing but backend knobs is refused instead
/// (`backend_knobs_alone_are_refused`). Without it this test would
/// still pass, on the `print_level=0` the helper injects — which is a
/// harness detail, not a statement about what is being tested.
#[test]
fn a_backend_knob_warns_but_solves() {
    let (code, err) = run(
        "user_scaling_suffix.nl",
        "backend",
        &["ma97_order=metis", "tol=1e-8"],
    );
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
        // `tol` for the same reason as above — this pins the grouping of
        // the *warning*, so the run must be one that warns.
        &[
            "ma97_order=metis",
            "ma97_u=1e-4",
            "pardiso_msglvl=1",
            "tol=1e-8",
        ],
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

/// `run`, but without the injected `print_level=0`.
///
/// The backend-only refusal below turns on what the run set *in total*,
/// and `print_level` is a real pounce option — injecting it would make
/// every case here look like a mixed file and the refusal would never
/// fire. That is not hypothetical: it is why the two warning tests
/// above still pass unchanged, and why they now name a real option of
/// their own rather than leaning on the helper to supply one.
fn run_bare(fixture_name: &str, tag: &str, opts: &[&str]) -> (Option<i32>, String) {
    let dir = std::env::temp_dir().join(format!("pounce_unimplopt_bare_{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join(fixture_name);
    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixture.push("tests/fixtures");
    fixture.push(fixture_name);
    std::fs::copy(&fixture, &nl).expect("copy fixture");
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .args(opts)
        .output()
        .expect("run pounce");
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// A run whose options are *nothing but* backend knobs is refused, not
/// warned.
///
/// The warn-don't-refuse rule protects a file that has other business
/// here from being failed over a knob the run never touches. With
/// nothing else in the file there is no such run to protect: warning
/// and solving would answer "tune MA97" by tuning nothing and reporting
/// success.
#[test]
fn backend_knobs_alone_are_refused() {
    let (code, err) = run_bare("user_scaling_suffix.nl", "onlyknob", &["ma97_order=metis"]);
    assert_eq!(code, Some(2), "stderr:\n{err}");
    assert!(
        err.contains("every option this run sets"),
        "the message must say the whole file is inert, not just this knob; stderr:\n{err}",
    );
    assert!(
        err.contains("linear_solver=feral"),
        "the message must say what to set instead; stderr:\n{err}",
    );
    assert!(err.contains("551"), "stderr:\n{err}");
}

/// Several families, still nothing else: one refusal, not one per
/// family.
#[test]
fn several_backend_families_alone_are_refused_once() {
    let (code, err) = run_bare(
        "user_scaling_suffix.nl",
        "onlyknobs",
        &["ma97_order=metis", "pardiso_msglvl=1", "mumps_pivtol=1e-5"],
    );
    assert_eq!(code, Some(2), "stderr:\n{err}");
    assert_eq!(
        err.matches("every option this run sets").count(),
        1,
        "one refusal for the file, not one per family; stderr:\n{err}",
    );
}

/// The boundary the refusal turns on: add one option pounce reads and
/// the same knobs are a warning again.
///
/// `tol` is deliberately set to its registered default. The gate is
/// *presence*, not "set to a non-default" — a caller who writes `tol`
/// has stated an intention about this solve either way, and a file with
/// real content is the portable-`ipopt.opt` case the warning exists
/// for.
#[test]
fn one_real_option_turns_the_refusal_back_into_a_warning() {
    let (code, err) = run_bare(
        "user_scaling_suffix.nl",
        "knobplusreal",
        &["ma97_order=metis", "tol=1e-8"],
    );
    assert_eq!(code, Some(0), "stderr:\n{err}");
    assert!(
        !err.contains("every option this run sets"),
        "a file with real content must not be refused; stderr:\n{err}",
    );
    assert!(err.contains("warning:"), "stderr:\n{err}");
    assert!(err.contains("`ma97_order`"), "stderr:\n{err}");
}

/// The default gate survives the new refusal: a file spelling out a
/// backend knob's registered default asks for nothing, so it is neither
/// refused nor warned about, even with nothing else in it.
#[test]
fn backend_knobs_at_their_defaults_alone_are_silent() {
    let (code, err) = run_bare(
        "user_scaling_suffix.nl",
        "onlydefknob",
        &["ma97_order=auto"],
    );
    assert_eq!(code, Some(0), "stderr:\n{err}");
    assert!(
        !err.contains("every option this run sets"),
        "stderr:\n{err}"
    );
    assert!(!err.contains("warning:"), "stderr:\n{err}");
}

/// A backend-only options *file* is refused by both routes that
/// deliver one: the implicit `ipopt.opt` in the working directory, and
/// an explicit `option_file_name=`.
///
/// The second is why `option_file_name` is exempt from the "mentions
/// something real" test. It is the mechanism that delivered the list,
/// not a statement about the solve, and counting it as content would
/// have made "the file you pointed me at configures nothing" the one
/// case that could never fire — while being the normal way to point at
/// a file.
#[test]
fn a_backend_only_options_file_is_refused_by_either_route() {
    for (tag, explicit) in [("implicit", false), ("explicit", true)] {
        let dir = std::env::temp_dir().join(format!("pounce_unimplopt_optfile_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let nl = dir.join("user_scaling_suffix.nl");
        let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fixture.push("tests/fixtures/user_scaling_suffix.nl");
        std::fs::copy(&fixture, &nl).expect("copy fixture");
        let opt = dir.join("ipopt.opt");
        std::fs::write(&opt, "ma97_order metis\nma97_u 1e-4\n").expect("write ipopt.opt");

        let mut cmd = Command::new(pounce_exe());
        cmd.arg(&nl).current_dir(&dir);
        if explicit {
            cmd.arg(format!("option_file_name={}", opt.display()));
        }
        let out = cmd.output().expect("run pounce");
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(out.status.code(), Some(2), "{tag} route; stderr:\n{err}");
        assert!(
            err.contains("every option this run sets"),
            "{tag} route; stderr:\n{err}",
        );
    }
}
