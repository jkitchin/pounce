//! Build script for the POUNCE CUTEst harness.
//!
//! Discovers and links three native dependencies:
//!
//! 1. **CUTEst** (`libcutest_double.a` + the `cutest_trampoline.f90`
//!    routines) under `~/.local/cutest/install/`.
//! 2. **fortran_open_fixed.f90** — small wrapper compiled into the
//!    same trampoline static lib.
//! 3. **libipopt** — discovered via `pkg-config` so the comparison
//!    binary can drive native Ipopt through its C ABI.
//!
//! Everything else (Fortran problem `.dylib`s, `OUTSDIF.d` data) is
//! loaded at runtime by the harness itself.

use std::path::Path;
use std::process::Command;

fn main() {
    // ---- Native Ipopt (libipopt) via pkg-config ------------------------------
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

    // ---- CUTEst trampoline + static library ----------------------------------
    let home = std::env::var("HOME").unwrap();
    let cutest_dir = format!("{}/.local/cutest", home);
    let cutest_lib = format!("{}/install/lib", cutest_dir);
    let cutest_include = format!("{}/install/include", cutest_dir);
    let cutest_modules = format!("{}/install/modules", cutest_dir);

    println!("cargo:rustc-link-search=native={}", cutest_lib);
    println!("cargo:rustc-link-lib=static=cutest_double");

    let trampoline_src = format!("{}/cutest/src/tools/cutest_trampoline.f90", cutest_dir);
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let trampoline_obj = format!("{}/cutest_trampoline.o", out_dir);

    if !Path::new(&trampoline_src).exists() {
        panic!(
            "CUTEst trampoline not found at {}. Install CUTEst first.",
            trampoline_src
        );
    }

    let s = Command::new("gfortran")
        .args([
            "-cpp",
            "-c",
            "-fPIC",
            &format!("-I{}", cutest_include),
            &format!("-I{}", cutest_modules),
            &format!("-J{}", cutest_modules),
            &trampoline_src,
            "-o",
            &trampoline_obj,
        ])
        .status()
        .expect("gfortran not found; install via homebrew (brew install gcc)");
    if !s.success() {
        panic!("Failed to compile cutest_trampoline.f90");
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let fortran_open_src = format!("{}/fortran_open_fixed.f90", manifest_dir);
    let fortran_open_obj = format!("{}/fortran_open_fixed.o", out_dir);
    let s = Command::new("gfortran")
        .args(["-c", "-fPIC", &fortran_open_src, "-o", &fortran_open_obj])
        .status()
        .expect("gfortran not found");
    if !s.success() {
        panic!("Failed to compile fortran_open_fixed.f90");
    }

    let trampoline_lib = format!("{}/libcutest_trampoline.a", out_dir);
    let s = Command::new("ar")
        .args([
            "rcs",
            &trampoline_lib,
            &trampoline_obj,
            &fortran_open_obj,
        ])
        .status()
        .expect("ar not found");
    if !s.success() {
        panic!("Failed to create libcutest_trampoline.a");
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=cutest_trampoline");

    // ---- Fortran runtime -----------------------------------------------------
    let dylib_name = if cfg!(target_os = "macos") {
        "libgfortran.dylib"
    } else {
        "libgfortran.so"
    };
    let out = Command::new("gfortran")
        .arg(format!("-print-file-name={}", dylib_name))
        .output()
        .expect("gfortran not found");
    let gpath = String::from_utf8(out.stdout).unwrap();
    if let Some(parent) = Path::new(gpath.trim()).parent() {
        if let Some(s) = parent.to_str() {
            println!("cargo:rustc-link-search=native={}", s);
        }
    }
    println!("cargo:rustc-link-lib=dylib=gfortran");

    println!("cargo:rerun-if-changed={}", trampoline_src);
    println!("cargo:rerun-if-changed={}/fortran_open_fixed.f90", manifest_dir);
    println!("cargo:rerun-if-changed={}/build.rs", manifest_dir);
}
