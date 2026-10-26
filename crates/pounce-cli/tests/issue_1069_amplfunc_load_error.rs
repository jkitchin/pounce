//! gh#1069: a `.nl` calling an AMPL external function that `$AMPLFUNC`
//! cannot resolve must be an ordinary load error on the CLI — non-zero
//! exit that is not 101, the resolver's message on stderr, no panic.
//!
//! Both the solve path and `pounce verify` used to build the TNLP with the
//! panicking `NlTnlp::new`. The fixture is `hs069` from the Vanderbei set,
//! which imports `myerf` (`F0 1 -1 myerf`).
//!
//! The "function not registered by any loaded library" branch is not
//! covered: reaching it needs a real AMPL function library exporting
//! `funcadd_ASL`, and the repo builds none.

use std::path::PathBuf;
use std::process::{Command, Output};

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures_issue_1069");
    p.push(name);
    p
}

/// Run `pounce` with `args`; `amplfunc` of `None` removes the variable.
fn run(args: &[PathBuf], verify: bool, amplfunc: Option<&str>) -> Output {
    let mut cmd = Command::new(pounce_exe());
    match amplfunc {
        Some(v) => cmd.env("AMPLFUNC", v),
        None => cmd.env_remove("AMPLFUNC"),
    };
    if verify {
        cmd.arg("verify");
    } else {
        cmd.arg("--no-sol");
    }
    cmd.args(args).output().expect("spawn pounce binary")
}

fn assert_clean_error(out: &Output, needle: &str, what: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    let code = out.status.code();
    assert!(
        code.is_some() && code != Some(0) && code != Some(101),
        "{what}: expected a non-zero, non-panic exit, got {code:?}; stderr:\n{stderr}"
    );
    assert!(
        stderr.contains(needle),
        "{what}: stderr should contain {needle:?}, got:\n{stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "{what}: stderr should not report a panic, got:\n{stderr}"
    );
}

#[test]
fn solve_without_amplfunc_is_a_clean_error() {
    let out = run(&[fixture("hs069.nl")], false, None);
    assert_clean_error(&out, "AMPLFUNC is not set", "solve");
}

#[test]
fn verify_without_amplfunc_is_a_clean_error() {
    let out = run(&[fixture("hs069.nl"), fixture("hs069.sol")], true, None);
    assert_clean_error(&out, "AMPLFUNC is not set", "verify");
}

#[test]
fn solve_with_missing_amplfunc_library_is_a_clean_error() {
    let missing = "/nonexistent/pounce-gh1069/libmissing.so";
    let out = run(&[fixture("hs069.nl")], false, Some(missing));
    assert_clean_error(&out, missing, "solve, missing library");
}

#[test]
fn verify_with_missing_amplfunc_library_is_a_clean_error() {
    let missing = "/nonexistent/pounce-gh1069/libmissing.so";
    let out = run(
        &[fixture("hs069.nl"), fixture("hs069.sol")],
        true,
        Some(missing),
    );
    assert_clean_error(&out, missing, "verify, missing library");
}
