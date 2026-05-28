//! Tells cargo where to find `libcoinhsl.dylib` at link- and run-time.
//!
//! Set the env var `COINHSL_DIR` to a CoinHSL install whose `lib/`
//! holds `libcoinhsl.{dylib,a}`. Only consulted when the `ma57`
//! feature is enabled — this crate is left out of the default build.
//!
//! `libcoinhsl.dylib` itself depends on `libopenblas`, `libmetis`,
//! `libgfortran.5`, `libgomp.1`, all of which live next to it under
//! `@rpath`. A single `-rpath` linker arg is enough to satisfy all of
//! them at runtime.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=COINHSL_DIR");

    let Ok(coinhsl_dir) = env::var("COINHSL_DIR").map(PathBuf::from) else {
        // No CoinHSL on this machine — compile pounce-hsl as a regular
        // rlib without emitting any link directives. Downstream crates
        // only pull pounce-hsl into a final binary when their `ma57`
        // feature is enabled, and *that* path needs CoinHSL; if a
        // downstream selects `ma57` here without COINHSL_DIR set, the
        // linker will fail with a clear "library not found: coinhsl"
        // error. The common `cargo build --workspace` (no `ma57`
        // feature) just compiles this crate as an unlinked rlib.
        println!(
            "cargo:warning=COINHSL_DIR not set; pounce-hsl compiled without link directives. \
             Selecting the `ma57` feature in a downstream crate without setting COINHSL_DIR will \
             fail at link time. Build CoinHSL from https://www.hsl.rl.ac.uk/ipopt/ and set \
             COINHSL_DIR to its install root to enable MA57."
        );
        return;
    };

    let lib_dir = coinhsl_dir.join("lib");
    assert!(
        lib_dir.is_dir(),
        "COINHSL lib directory not found: {}",
        lib_dir.display(),
    );

    let Some(lib_dir_str) = lib_dir.to_str() else {
        panic!("COINHSL lib path is not valid UTF-8: {}", lib_dir.display());
    };
    println!("cargo:rustc-link-search=native={lib_dir_str}");
    println!("cargo:rustc-link-lib=dylib=coinhsl");
    // Explicit -lopenblas so `openblas_set_num_threads` resolves at
    // link time. macOS two-level namespace will not pull the symbol
    // transitively through libcoinhsl. The dylib lives in the same
    // lib_dir, so the search path above already finds it.
    println!("cargo:rustc-link-lib=dylib=openblas");
    // libcoinhsl.dylib's @rpath dependencies live in the same lib
    // directory, so this single rpath resolves all of them.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir_str}");
}
