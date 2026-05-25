//! Link native libipopt for the `large_scale_ipopt` head-to-head bin.
//! Discovered via pkg-config (same approach as benchmarks/cutest).

use std::process::Command;

fn main() {
    // PKG_CONFIG_PATH is honored: set it (e.g. via the benchmarks Makefile)
    // to point at ref/Ipopt/install-ma57/lib/pkgconfig so the FFI route
    // links against the MA57 build instead of Homebrew's MUMPS Ipopt.
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    let pkg = Command::new("pkg-config")
        .args(["--libs-only-L", "ipopt"])
        .output()
        .expect("pkg-config not found; install ipopt (brew install ipopt)");
    if !pkg.status.success() {
        panic!(
            "pkg-config failed for ipopt: {}",
            String::from_utf8_lossy(&pkg.stderr)
        );
    }
    let lib_path = String::from_utf8(pkg.stdout).unwrap();
    let lib_path = lib_path.trim().trim_start_matches("-L");
    if !lib_path.is_empty() {
        println!("cargo:rustc-link-search=native={}", lib_path);
    }
    println!("cargo:rustc-link-lib=dylib=ipopt");
}
