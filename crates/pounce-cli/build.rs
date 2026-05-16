//! Stamps build-time metadata into the `pounce` binary so `--about`
//! can print version/build/git/rustc info without runtime introspection.
//!
//! Everything here is best-effort: missing git or `date` just becomes
//! "unknown" in the output. Nothing here changes link behavior.

use std::process::Command;

fn main() {
    // Re-stamp when HEAD moves; otherwise the SHA in the binary is stale.
    // The .git/HEAD path is relative to this crate's manifest dir.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
    println!("cargo:rerun-if-env-changed=PROFILE");
    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rerun-if-env-changed=HOST");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");

    let git_sha = run("git", &["rev-parse", "--short=8", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = run("git", &["status", "--porcelain"])
        .map(|s| if s.is_empty() { "" } else { "+dirty" })
        .unwrap_or("");
    let git = format!("{git_sha}{dirty}");

    // UTC ISO-8601 timestamp. Honor SOURCE_DATE_EPOCH for reproducible
    // builds; otherwise fall back to `date -u` at compile time.
    let build_time = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|s| run("date", &["-u", "-r", &s, "+%Y-%m-%dT%H:%M:%SZ"]))
        .or_else(|| run("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]))
        .unwrap_or_else(|| "unknown".into());

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_version = run(&rustc, &["--version"]).unwrap_or_else(|| "rustc unknown".into());

    println!("cargo:rustc-env=POUNCE_BUILD_GIT={git}");
    println!("cargo:rustc-env=POUNCE_BUILD_TIME={build_time}");
    println!("cargo:rustc-env=POUNCE_BUILD_RUSTC={rustc_version}");
    println!(
        "cargo:rustc-env=POUNCE_BUILD_PROFILE={}",
        std::env::var("PROFILE").unwrap_or_default()
    );
    println!(
        "cargo:rustc-env=POUNCE_BUILD_TARGET={}",
        std::env::var("TARGET").unwrap_or_default()
    );
    println!(
        "cargo:rustc-env=POUNCE_BUILD_HOST={}",
        std::env::var("HOST").unwrap_or_default()
    );
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    Some(s)
}
