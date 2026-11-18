//! `linear_solver` accepted every backend name and silently ran FERAL
//! (gh#483 follow-up).
//!
//! The option's valid-value list is a faithful port of upstream Ipopt's
//! — `ma27`, `ma57`, `ma77`, `ma86`, `ma97`, `mumps`, `pardiso`,
//! `pardisomkl`, `spral`, `wsmp`, `custom`, `feral` — so an `ipopt.opt`
//! written for Ipopt parses here unchanged. pounce implements three of
//! them: `feral`, `ma57` and `custom` (a library the user names, gh#1175). The resolver mapped everything else through a `_ =>` arm to
//! FERAL, so `linear_solver=ma97` "worked": a run that reported success
//! while using a backend the binary does not contain, and a benchmark
//! comparing linear solvers that compared FERAL against itself.
//!
//! Two boundaries this pins, because both are load-bearing:
//!
//! * The **registered default is `feral`** — pounce's own backend, and
//!   one this binary always contains. It diverges from upstream's
//!   `ma57` deliberately: that default advertised a solver a pure-Rust
//!   build does not have, and made an HSL build run MA57 without being
//!   asked. A default run must therefore solve, not be refused.
//! * **Explicit `ma57` on a build without the feature** still falls back.
//!   That fallback is reported in the banner rather than hidden, and
//!   failing a portable `ipopt.opt` over a build flag would cost more
//!   than it buys.
//!
//! The refusal is checked before routing, so it does not depend on
//! whether the model classifies into the NLP path or the convex one —
//! the convex dispatch never reaches `optimize_tnlp`, where the sibling
//! guard for library consumers lives.

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

/// Run `fixture_name` with `opts`; returns `(exit code, stderr)`.
fn run(fixture_name: &str, tag: &str, opts: &[&str]) -> (Option<i32>, String) {
    let dir = std::env::temp_dir().join(format!("pounce_linsolsel_{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join(fixture_name);
    std::fs::copy(fixture(fixture_name), &nl).expect("copy fixture");
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

/// Every backend pounce does not implement is refused, by name, with a
/// message that says what to use instead. Pre-fix each of these exited
/// 0 having run FERAL.
#[test]
fn unimplemented_backends_are_refused() {
    for (i, name) in [
        "ma27",
        "ma77",
        "ma86",
        "ma97",
        "mumps",
        "pardiso",
        "pardisomkl",
        "spral",
        "wsmp",
    ]
    .into_iter()
    .enumerate()
    {
        let (code, err) = run(
            "user_scaling_suffix.nl",
            &format!("bad{i}"),
            &[&format!("linear_solver={name}")],
        );
        assert_eq!(code, Some(2), "linear_solver={name} should fail; {err}");
        assert!(
            err.contains(&format!("linear_solver={name} is not implemented")),
            "the refusal must name the backend; stderr:\n{err}",
        );
        assert!(
            err.contains("linear_solver=feral"),
            "the refusal must say what to use instead; stderr:\n{err}",
        );
    }
}

/// `custom` is implemented (gh#1175): it loads the library named by
/// `custom_linear_solver_lib`. Without one, or with one that cannot be
/// loaded, the run fails and says which -- it never falls back to FERAL.
/// Both branches, since they fail at different points (option check vs
/// `dlopen`).
#[test]
fn custom_needs_a_loadable_library_and_never_falls_back() {
    for (tag, extra, says) in [
        ("custom_unset", None, "custom_linear_solver_lib is not set"),
        (
            "custom_missing",
            Some("custom_linear_solver_lib=/nonexistent/libnope.so"),
            "cannot load custom_linear_solver_lib",
        ),
    ] {
        let mut opts = vec!["linear_solver=custom"];
        opts.extend(extra);
        let (code, err) = run("user_scaling_suffix.nl", tag, &opts);
        assert_ne!(
            code,
            Some(0),
            "{tag}: a custom solver that never loaded solved; {err}"
        );
        assert!(err.contains(says), "{tag}: stderr:\n{err}");
        assert!(
            !err.contains("is not implemented"),
            "{tag}: custom is implemented; stderr:\n{err}"
        );
    }
}

/// The option value is case-insensitive in the registry, so the guard
/// must be too — `MUMPS` cannot be a way around it.
#[test]
fn the_refusal_is_case_insensitive() {
    let (code, err) = run("user_scaling_suffix.nl", "upper", &["linear_solver=MUMPS"]);
    assert_eq!(code, Some(2), "stderr:\n{err}");
}

/// The two pounce implements are accepted. `ma57` is accepted on any
/// build: without the feature it falls back to FERAL and the banner says
/// so, which is a reported substitution rather than a hidden one.
#[test]
fn implemented_backends_are_accepted() {
    for name in ["feral", "ma57"] {
        let (code, err) = run(
            "user_scaling_suffix.nl",
            name,
            &[&format!("linear_solver={name}")],
        );
        assert_eq!(code, Some(0), "linear_solver={name} should solve; {err}");
    }
}

/// The registered default must name a backend this binary contains, or
/// the guard refuses every solve that does not set the option. That is
/// the invariant the `feral` default buys: no explicit-vs-default
/// special case is needed anywhere, because the default is legal.
#[test]
fn the_registered_default_solves() {
    let (code, err) = run("user_scaling_suffix.nl", "default", &[]);
    assert_eq!(code, Some(0), "a default run must not be refused; {err}");
    assert!(
        !err.contains("not implemented"),
        "a default run must not warn either; stderr:\n{err}",
    );
}

/// …and it is FERAL, in every build. Under upstream's `ma57` default an
/// HSL build silently ran MA57 without being asked, and a pure-Rust one
/// banners a solver the option string did not name.
#[test]
fn the_registered_default_is_feral() {
    let dir = std::env::temp_dir().join("pounce_linsolsel_bannerdefault");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join("user_scaling_suffix.nl");
    std::fs::copy(fixture("user_scaling_suffix.nl"), &nl).expect("copy fixture");
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .output()
        .expect("run pounce");
    let _ = std::fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("running with linear solver FERAL"),
        "default banner should name FERAL; stdout:\n{stdout}",
    );
}

/// The guard runs before solver routing: a model that classifies as a
/// convex QP dispatches to `pounce-convex`, which never reaches the
/// library-side guard in `optimize_tnlp`.
#[test]
fn the_refusal_covers_the_convex_route() {
    let (code, err) = run("boxed_qp_min.nl", "convex", &["linear_solver=mumps"]);
    assert_eq!(code, Some(2), "stderr:\n{err}");
    assert!(err.contains("not implemented"), "stderr:\n{err}");
}
