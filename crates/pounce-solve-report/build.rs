//! Re-exports the build target triple into a crate-readable env var.
//!
//! Cargo sets the `TARGET` env var for **build scripts only** — it is not
//! visible to normal crate compilation, so `option_env!("TARGET")` in the
//! library source is always `None` (and `target_triple` in solve reports
//! came out `"unknown"`). A build script *does* see `TARGET`, so re-export
//! it under our own name for `option_env!("POUNCE_TARGET_TRIPLE")` to read.

fn main() {
    // `TARGET` is always present for build scripts; keep a fallback anyway so
    // an unusual build environment degrades to the old "unknown" rather than
    // breaking the build.
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=POUNCE_TARGET_TRIPLE={target}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TARGET");
}
