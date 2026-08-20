//! The sIPOPT option names reach the `pounce` driver, not just the
//! `--*` flags (gh#551 / gh#677).
//!
//! `compute_red_hessian`, `rh_eigendecomp`, `run_sens`,
//! `sens_boundcheck` and `sens_bound_eps` are registered so an sIPOPT
//! `ipopt.opt` parses unchanged. Registering them said nothing about
//! reading them: on the command line (or in an options file) every one
//! of them was accepted and then ignored, and the same work was
//! reachable only through `--compute-red-hessian`, `--rh-eigendecomp`
//! and `--sens-boundcheck`. Each test here sets the *option* and checks
//! the driver's output changes.
//!
//! Fixtures: `parametric.nl` is upstream sIPOPT's `parametric_cpp`
//! problem (see `pounce_sens_end_to_end.rs`);
//! `parametric_red_hessian.nl` is the same model plus the AMPL
//! `red_hessian` var-suffix tagging two variables.

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

/// Run the driver on a fixture, returning `(stderr, .sol text)`.
fn run(nl: &str, tag: &str, opts: &[&str]) -> (String, String) {
    let mut sol = std::env::temp_dir();
    sol.push(format!("pounce_sensopt_{tag}_{}.sol", std::process::id()));
    let out = Command::new(pounce_exe())
        .arg(fixture(nl))
        .arg(&sol)
        .args(opts)
        .arg("print_level=0")
        .output()
        .expect("spawn pounce");
    assert!(
        out.status.success(),
        "pounce exited with {:?}; stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    let sol_text = std::fs::read_to_string(&sol).unwrap_or_default();
    let _ = std::fs::remove_file(&sol);
    (String::from_utf8_lossy(&out.stderr).into_owned(), sol_text)
}

/// The `sens_sol_state_1` values out of a `.sol`, as raw text lines —
/// enough to tell "absent", "present", and "present but different"
/// apart without re-implementing the suffix parser.
fn sens_suffix_block(sol: &str) -> Option<Vec<f64>> {
    let mut lines = sol.lines();
    while let Some(line) = lines.next() {
        let Some(rest) = line.strip_prefix("suffix ") else {
            continue;
        };
        let parts: Vec<&str> = rest.split_whitespace().collect();
        if parts.len() < 5 {
            continue;
        }
        let count: usize = parts[1].parse().ok()?;
        let tabline: usize = parts[4].parse().ok()?;
        let name = lines.next()?.trim().to_string();
        for _ in 0..tabline {
            lines.next()?;
        }
        let mut vals = Vec::new();
        for _ in 0..count {
            let entry = lines.next()?;
            let v: f64 = entry.split_whitespace().nth(1)?.parse().ok()?;
            vals.push(v);
        }
        if name == "sens_sol_state_1" {
            return Some(vals);
        }
    }
    None
}

/// `run_sens=no` is upstream's "solve, but skip the step". The `.nl`
/// declares the sIPOPT suffixes, so the step runs by default; the
/// option is the only way to turn it off, and it must actually remove
/// the `sens_sol_state_1` the driver would otherwise write.
#[test]
fn run_sens_no_suppresses_the_suffix_the_nl_asks_for() {
    let (_, on) = run("parametric.nl", "runsens_on", &[]);
    assert!(
        sens_suffix_block(&on).is_some(),
        "the fixture's suffixes ask for the step by default:\n{on}"
    );

    let (err, off) = run("parametric.nl", "runsens_off", &["run_sens=no"]);
    assert!(
        sens_suffix_block(&off).is_none(),
        "`run_sens=no` must not write sens_sol_state_1:\n{off}"
    );
    assert!(err.contains("run_sens=no"), "and must say so: {err}");
}

/// `run_sens=yes` on an input that declares no sIPOPT suffixes has
/// nothing to perturb. Solving and reporting nothing is the silent
/// no-op this work exists to remove, so it warns.
#[test]
fn run_sens_yes_without_the_suffixes_warns() {
    let (err, _) = run(
        "user_scaling_suffix.nl",
        "runsens_nosuffix",
        &["run_sens=yes"],
    );
    assert!(
        err.contains("run_sens=yes") && err.contains("sens_state_1"),
        "stderr:\n{err}"
    );
}

/// `compute_red_hessian=yes` computes and prints the reduced Hessian,
/// exactly as `--compute-red-hessian` does — and without it, nothing
/// is printed.
#[test]
fn compute_red_hessian_option_reaches_the_computation() {
    let (quiet, _) = run("parametric_red_hessian.nl", "rh_off", &[]);
    assert!(
        !quiet.contains("Reduced Hessian"),
        "unset: nothing asks for it:\n{quiet}"
    );

    let (err, _) = run(
        "parametric_red_hessian.nl",
        "rh_on",
        &["compute_red_hessian=yes"],
    );
    assert!(
        err.contains("=== Reduced Hessian"),
        "`compute_red_hessian=yes` must produce it:\n{err}"
    );
    assert!(
        !err.contains("eigenvalues"),
        "...and only it, without `rh_eigendecomp`:\n{err}"
    );
}

/// `rh_eigendecomp=yes` implies the reduced Hessian and adds its
/// eigenvalues to the report, the same implication `--rh-eigendecomp`
/// carries.
#[test]
fn rh_eigendecomp_option_adds_the_eigenvalues() {
    let (err, _) = run(
        "parametric_red_hessian.nl",
        "rh_eig",
        &["rh_eigendecomp=yes"],
    );
    assert!(err.contains("=== Reduced Hessian"), "stderr:\n{err}");
    assert!(
        err.contains("Reduced-Hessian eigenvalues"),
        "stderr:\n{err}"
    );
}

/// `sens_boundcheck=yes` refines the perturbed primal onto the model's
/// own box. On this fixture the unrefined step takes x₂ to −0.0459,
/// below its declared `x_l = 0`; the option is the difference between
/// reporting that and reporting a point on the bound.
#[test]
fn sens_boundcheck_option_refines_the_reported_primal() {
    let (_, plain) = run("parametric.nl", "bc_off", &[]);
    let unrefined = sens_suffix_block(&plain).expect("sens_sol_state_1");
    assert!(
        unrefined[2] < -1e-3,
        "fixture check: the plain step leaves x_l=0, got {}",
        unrefined[2]
    );

    let (_, refined_sol) = run("parametric.nl", "bc_on", &["sens_boundcheck=yes"]);
    let refined = sens_suffix_block(&refined_sol).expect("sens_sol_state_1");
    assert!(
        refined[2] > -1e-3,
        "`sens_boundcheck=yes` must hold x_2 at its bound, got {}",
        refined[2]
    );
    assert!(
        (refined[2] - unrefined[2]).abs() > 1e-6,
        "the refinement must change the reported primal"
    );
}

/// `sens_bound_eps` is the margin the refinement measures against: one
/// wider than the crossing leaves the step alone, a tight one pins it.
/// Same fixture, same flag, two different reported primals.
#[test]
fn sens_bound_eps_option_sets_the_refinement_margin() {
    let (_, plain) = run("parametric.nl", "eps_plain", &[]);
    let (_, slack) = run(
        "parametric.nl",
        "eps_slack",
        &["sens_boundcheck=yes", "sens_bound_eps=10.0"],
    );
    let (_, tight) = run(
        "parametric.nl",
        "eps_tight",
        &["sens_boundcheck=yes", "sens_bound_eps=1e-9"],
    );
    let p = sens_suffix_block(&plain).expect("sens_sol_state_1");
    let s = sens_suffix_block(&slack).expect("sens_sol_state_1");
    let t = sens_suffix_block(&tight).expect("sens_sol_state_1");

    for k in 0..p.len() {
        assert!(
            (s[k] - p[k]).abs() < 1e-9,
            "a 10.0 margin tolerates the crossing and must leave slot {k} alone: {} vs {}",
            s[k],
            p[k]
        );
    }
    assert!(
        (t[2] - s[2]).abs() > 1e-6,
        "a 1e-9 margin must pin what a 10.0 one tolerates: {} vs {}",
        t[2],
        s[2]
    );
}
