//! Every `examples/*.c` compiles against `include/pounce.h` and the
//! crate's cdylib with strict warnings, and runs green.
//!
//! The flags are `-std=c99 -Wall -Wextra -Werror -pedantic`, deliberately:
//! the header has to be clean C99 for any C consumer, and an example that
//! warns is an example a user copies warnings out of. Before this test
//! nothing in CI compiled a line of C against the header (the CasADi plugin
//! is C++ and has its own flags), so a header that stopped compiling, or an
//! example that stopped linking against the library it documents, went
//! unseen.
//!
//! How it finds things:
//!
//! * **The library.** Cargo builds every crate type of the lib target
//!   (`["lib", "cdylib"]`) before an integration test runs, into the
//!   profile directory that also holds `deps/`. This binary lives in
//!   `<target>/<profile>/deps/`, so the cdylib is
//!   `current_exe().parent().parent()/libpounce_cinterface.{so,dylib}`.
//!   That follows `CARGO_TARGET_DIR`, `--release`/`--profile` and
//!   `--target` without reading any of them. If the file is not there the
//!   test fails and says so; it never skips for that.
//! * **The compiler.** `$CC` if set and non-empty, else `cc`.
//! * **Running.** The link carries `-Wl,-rpath,<libdir>`, and the run sets
//!   `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` to `<libdir>` as well.
//!
//! Each example must have an entry in [`EXPECTED_STDOUT`]: the line its
//! successful run prints. An example with no entry fails the test (adding
//! one is part of adding the example), as does an entry naming no file.
//!
//! **`POUNCE_C_EXAMPLES_SKIP=1`.** With no C compiler on `PATH` the test
//! fails, because a silent skip is how CI would stop checking the header
//! without anyone noticing. Set `POUNCE_C_EXAMPLES_SKIP=1` to skip it
//! knowingly on a machine without one. CI (`ubuntu-latest`) ships `cc`,
//! and does not set the variable.
//!
//! Unix only (Linux, macOS): linking a Windows DLL through its import
//! library is a different command line, and no CI job tests the workspace
//! on Windows.
//!
//! Run the same builds outside cargo with `make c-api-examples`.

#![cfg(unix)]
// Test code: a panicking unwrap/expect is the test failing, as intended.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The stdout line each example prints on success.
const EXPECTED_STDOUT: &[(&str, &str)] = &[
    (
        "bounds_update.c",
        "bounds_update example: re-solved to (3/4, 7/8, 1/8) after an in-place update, 2 refusals",
    ),
    (
        "complementarity.c",
        "complementarity example: S-stationary answer (1, 1, 0) certified",
    ),
    (
        "qp_socp.c",
        "qp_socp example: QP (1/2, 1/2), SOC distance 1/sqrt(2), infeasibility certified, 3 refusals",
    ),
    ("sqp_warm_start.c", "all 4 solves succeeded"),
];

/// Warning flags every example is held to.
const STRICT_FLAGS: &[&str] = &["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic"];

const LIB_NAME: &str = "pounce_cinterface";

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn cdylib_file_name() -> String {
    if cfg!(target_os = "macos") {
        format!("lib{LIB_NAME}.dylib")
    } else {
        format!("lib{LIB_NAME}.so")
    }
}

/// The directory holding the cdylib cargo built for this test run.
///
/// `deps/` first: cargo always writes the cdylib there, next to this test
/// binary, and uplifts it to the profile directory only when the lib is a
/// root unit. Under `cargo test --workspace` from a clean target it is not
/// (`pounce-cli` dev-depends on this crate), so only `deps/` has it; the
/// uplifted file, when there, is a hard link to the same output.
fn lib_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    let deps = exe.parent().expect("test binary has a parent directory");
    let profile = deps.parent().expect("deps/ has a parent profile directory");
    for dir in [deps, profile] {
        if dir.join(cdylib_file_name()).is_file() {
            return dir.to_path_buf();
        }
    }
    panic!(
        "expected the cdylib cargo builds for integration tests in {} or {}, \
         but it is in neither (test binary: {}). The lib target must keep \
         crate-type = [\"lib\", \"cdylib\"]; if this toolchain does not build \
         the cdylib for tests, build it first (`cargo build -p \
         pounce-cinterface`) — this test does not skip for a missing library.",
        deps.display(),
        profile.display(),
        exe.display()
    );
}

fn c_compiler() -> OsString {
    match std::env::var_os("CC") {
        Some(cc) if !cc.is_empty() => cc,
        _ => OsString::from("cc"),
    }
}

/// The compiler arguments (everything after the compiler itself) that build
/// `src` into `out` against the header in `include` and the cdylib in
/// `lib_dir`. Libraries come after the source, as the GNU linker requires.
fn compile_args(include: &Path, src: &Path, out: &Path, lib_dir: &Path) -> Vec<OsString> {
    let mut args: Vec<OsString> = STRICT_FLAGS.iter().map(OsString::from).collect();
    args.push(OsString::from("-I"));
    args.push(include.into());
    args.push(src.into());
    args.push(OsString::from("-o"));
    args.push(out.into());
    args.push(OsString::from("-L"));
    args.push(lib_dir.into());
    args.push(OsString::from(format!("-l{LIB_NAME}")));
    args.push(OsString::from("-lm"));
    let mut rpath = OsString::from("-Wl,-rpath,");
    rpath.push(lib_dir);
    args.push(rpath);
    args
}

/// `Ok(())` when a C compiler answers, `Err(why)` when none does.
fn compiler_available(cc: &OsString) -> Result<(), String> {
    match Command::new(cc).arg("--version").output() {
        Ok(o) if o.status.success() => Ok(()),
        Ok(o) => Err(format!(
            "`{} --version` exited {}",
            cc.to_string_lossy(),
            o.status
        )),
        Err(e) => Err(format!("cannot run `{}`: {e}", cc.to_string_lossy())),
    }
}

/// True when the run should skip, false when it should go ahead; panics
/// when there is no compiler and the skip was not asked for.
fn skip_for_missing_compiler(cc: &OsString) -> bool {
    let Err(why) = compiler_available(cc) else {
        return false;
    };
    if std::env::var("POUNCE_C_EXAMPLES_SKIP").as_deref() == Ok("1") {
        eprintln!("skipping C example builds: {why} (POUNCE_C_EXAMPLES_SKIP=1)");
        return true;
    }
    panic!(
        "no C compiler: {why}. Install one, point $CC at it, or set \
         POUNCE_C_EXAMPLES_SKIP=1 to skip this test knowingly."
    );
}

fn work_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("c_examples")
        .join(name);
    std::fs::create_dir_all(&dir).expect("create work dir");
    dir
}

fn examples() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(manifest_dir().join("examples"))
        .expect("read examples/")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "c"))
        .collect();
    v.sort();
    v
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .expect("file name")
        .to_string_lossy()
        .into_owned()
}

fn compile(cc: &OsString, src: &Path, out: &Path, lib_dir: &Path) -> std::process::Output {
    let include = manifest_dir().join("include");
    Command::new(cc)
        .args(compile_args(&include, src, out, lib_dir))
        .output()
        .expect("spawn C compiler")
}

#[test]
fn every_example_has_an_expected_line_and_every_entry_a_file() {
    let names: Vec<String> = examples().iter().map(|p| file_name(p)).collect();
    assert!(!names.is_empty(), "no examples/*.c found");
    for n in &names {
        assert!(
            EXPECTED_STDOUT.iter().any(|(f, _)| f == n),
            "examples/{n} has no entry in EXPECTED_STDOUT"
        );
    }
    for (f, _) in EXPECTED_STDOUT {
        assert!(
            names.iter().any(|n| n == f),
            "EXPECTED_STDOUT names examples/{f}, which does not exist"
        );
    }
}

#[test]
fn every_c_example_builds_strictly_and_runs() {
    let cc = c_compiler();
    if skip_for_missing_compiler(&cc) {
        return;
    }
    let lib_dir = lib_dir();
    let work = work_dir("examples");
    for src in examples() {
        let name = file_name(&src);
        let stem = src
            .file_stem()
            .expect("stem")
            .to_string_lossy()
            .into_owned();
        let out = work.join(&stem);
        let build = compile(&cc, &src, &out, &lib_dir);
        assert!(
            build.status.success(),
            "examples/{name} failed to build with {:?}:\n{}{}",
            STRICT_FLAGS,
            String::from_utf8_lossy(&build.stdout),
            String::from_utf8_lossy(&build.stderr)
        );

        let run = Command::new(&out)
            .env("LD_LIBRARY_PATH", &lib_dir)
            .env("DYLD_LIBRARY_PATH", &lib_dir)
            .output()
            .expect("spawn example binary");
        let stdout = String::from_utf8_lossy(&run.stdout);
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(
            run.status.success(),
            "examples/{name} exited {}:\nstdout:\n{stdout}\nstderr:\n{stderr}",
            run.status
        );
        let expected = EXPECTED_STDOUT
            .iter()
            .find(|(f, _)| *f == name)
            .map(|(_, line)| *line)
            .expect("checked by every_example_has_an_expected_line_and_every_entry_a_file");
        assert!(
            stdout.lines().any(|l| l.contains(expected)),
            "examples/{name} exited 0 but did not print {expected:?}:\n{stdout}"
        );
    }
}

/// The other branch of the build: a C file that warns must *fail* under
/// the assembled flags, or `-Werror -pedantic` is not actually reaching the
/// compiler and the test above proves nothing about warnings.
#[test]
fn a_warning_fails_the_strict_build() {
    let cc = c_compiler();
    if skip_for_missing_compiler(&cc) {
        return;
    }
    let lib_dir = lib_dir();
    let work = work_dir("negative");
    // Clean apart from one unused parameter (-Wextra) — and it includes the
    // header, so the failure is the warning, not the header.
    let warns = work.join("unused_parameter.c");
    std::fs::write(
        &warns,
        "#include \"pounce.h\"\n\
         static int f(int unused) { return 0; }\n\
         int main(void) { return f(0); }\n",
    )
    .expect("write C file");
    let build = compile(&cc, &warns, &work.join("unused_parameter"), &lib_dir);
    assert!(
        !build.status.success(),
        "a C file with an unused parameter built under {STRICT_FLAGS:?}; \
         the strict flags are not taking effect"
    );

    // And the same file without the warning builds, links and runs, so the
    // failure above is the warning and not something else in the command.
    let clean = work.join("clean.c");
    std::fs::write(
        &clean,
        "#include \"pounce.h\"\n\
         static int f(int used) { return used; }\n\
         int main(void) { return f(0); }\n",
    )
    .expect("write C file");
    let out = work.join("clean");
    let build = compile(&cc, &clean, &out, &lib_dir);
    assert!(
        build.status.success(),
        "the warning-free control failed to build:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&out).output().expect("spawn control binary");
    assert!(run.status.success(), "control binary exited {}", run.status);
}

#[test]
fn compile_args_carry_strict_flags_and_link_after_the_source() {
    let args = compile_args(
        Path::new("/inc"),
        Path::new("/src/a.c"),
        Path::new("/out/a"),
        Path::new("/lib dir"),
    );
    let s: Vec<String> = args
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    for f in STRICT_FLAGS {
        assert!(s.iter().any(|a| a == f), "missing {f} in {s:?}");
    }
    let pos = |x: &str| s.iter().position(|a| a == x).unwrap_or(usize::MAX);
    let src = pos("/src/a.c");
    let lib = pos("-lpounce_cinterface");
    let libm = pos("-lm");
    assert!(
        src < lib && lib < libm && libm < s.len(),
        "order wrong: {s:?}"
    );
    assert_eq!(s[pos("-I") + 1], "/inc");
    assert_eq!(s[pos("-o") + 1], "/out/a");
    assert_eq!(s[pos("-L") + 1], "/lib dir");
    // A path with a space stays one argument; nothing is shell-joined.
    assert!(s.iter().any(|a| a == "-Wl,-rpath,/lib dir"), "{s:?}");
}

/// Field names, in declaration order, of `typedef struct { ... } NAME;` in
/// the header (comments stripped; `char x[32]` reads as `x`).
fn header_struct_fields(header: &str, name: &str) -> Vec<String> {
    let mut text = String::new();
    let mut rest = header;
    while let Some(i) = rest.find("/*") {
        text.push_str(&rest[..i]);
        let j = rest[i..].find("*/").expect("unterminated comment") + i + 2;
        rest = &rest[j..];
    }
    text.push_str(rest);
    let close = format!("}} {name};");
    let end = text
        .find(&close)
        .unwrap_or_else(|| panic!("pounce.h has no `typedef struct {{ .. }} {name};`"));
    let open = text[..end]
        .rfind("typedef struct {")
        .expect("typedef struct {")
        + 16;
    text[open..end]
        .split(';')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(|d| {
            let last = d.split_whitespace().last().expect("a declarator");
            last.split('[')
                .next()
                .expect("a name")
                .trim_start_matches('*')
                .to_string()
        })
        .collect()
}

/// `(field, offset, size)` for each named field of a `#[repr(C)]` Rust
/// struct of plain scalars and arrays (all-zero bytes are a valid value).
macro_rules! layout {
    ($t:ty; $($f:ident),* $(,)?) => {{
        // SAFETY: the structs measured here hold only integers, floats
        // and arrays of them.
        let v: $t = unsafe { std::mem::zeroed() };
        (
            stringify!($t),
            std::mem::size_of::<$t>(),
            vec![$((
                stringify!($f),
                std::mem::offset_of!($t, $f),
                std::mem::size_of_val(&v.$f),
            )),*],
        )
    }};
}

/// The `#[repr(C)]` structs the library writes into caller memory must
/// have the header's layout exactly: same fields in the same order, same
/// offsets and widths, same size. The C side is measured by the C compiler,
/// not restated: a program built against `pounce.h` prints `sizeof` of the
/// struct and the `offsetof` and `sizeof` of every field, and each is held
/// to Rust's `size_of` / `offset_of!` / `size_of_val` (a field widened into
/// padding moves no offset, so the widths are checked too). The
/// header's field list is parsed and held to the Rust list too, so a field
/// added to one side only fails here even when it lands in padding.
#[test]
fn repr_c_structs_match_the_header_layout() {
    use pounce_cinterface::{
        PounceComplementaritySolve, PounceComplementarityStats, PounceLinearSolverStats,
        PounceQpCertificate, PounceQpResiduals,
    };
    let structs = [
        layout!(PounceLinearSolverStats;
            solver_name, n_factors, n_pattern_reuse, n_pattern_changes, max_fill_ratio,
            min_abs_pivot, max_abs_pivot, last_inertia_positive, last_inertia_negative,
            last_inertia_zero, last_nnz_a, last_nnz_l),
        layout!(PounceComplementarityStats;
            struct_size, strategy, reported, accepted_by, fallback, n_solves, total_iterations,
            max_abs_product, certificate_class, n_pairs, n_biactive, n_neither,
            multipliers_unique, residual, resid_tol, tight, feasible, primal_violation,
            b_verdict, b_n_biactive, b_branches_solved, b_enumerated, escapes, lifted,
            natural_residual, polish_ran, polish_taken, polish_residual_before,
            polish_residual_after, polish_threshold),
        layout!(PounceComplementaritySolve;
            struct_size, stage, relaxed, tau, status, iterations, max_abs_product, seconds),
        layout!(PounceQpResiduals;
            struct_size, primal_infeasibility, dual_infeasibility, complementarity, kkt_error,
            primal_infeasibility_raw, dual_infeasibility_raw, complementarity_raw,
            kkt_error_raw),
        layout!(PounceQpCertificate; struct_size, kind, scale, tau, kappa),
    ];

    // The extensible output structs (`pounce.h`, "Extensible output
    // structs"): `struct_size` is the first field, at offset 0, and is an
    // `ipindex`; their sizes are this release's, pinned so that a change is
    // a decision (the growth rule only ever adds bytes at the end).
    // `PounceLinearSolverStats` shipped before the convention and has none.
    // The pinned sizes assume an 8-byte-aligned `ipnumber`, which every
    // 64-bit target has; on a 32-bit target (i386 System V aligns `double`
    // to 4) they differ, so there only the Rust-against-C comparison above
    // and the offset-0 check below apply.
    let pin_sizes = cfg!(target_pointer_width = "64");
    for (name, size) in [
        ("PounceComplementarityStats", 224),
        ("PounceComplementaritySolve", 72),
        ("PounceQpResiduals", 72),
        ("PounceQpCertificate", 32),
    ] {
        let (_, rust_size, fields) = structs.iter().find(|(n, _, _)| *n == name).unwrap();
        if pin_sizes {
            assert_eq!(*rust_size, size, "sizeof({name})");
        }
        assert_eq!(
            fields[0],
            (
                "struct_size",
                0,
                std::mem::size_of::<pounce_cinterface::Index>()
            ),
            "{name}: struct_size must be the first field, at offset 0"
        );
    }
    assert!(
        structs[0].2.iter().all(|(f, _, _)| *f != "struct_size"),
        "PounceLinearSolverStats is frozen: no struct_size"
    );

    let header =
        std::fs::read_to_string(manifest_dir().join("include/pounce.h")).expect("read pounce.h");
    for (name, _, fields) in &structs {
        let rust: Vec<&str> = fields.iter().map(|(f, _, _)| *f).collect();
        assert_eq!(
            header_struct_fields(&header, name),
            rust,
            "{name}: the header's fields differ from the Rust struct's (or their order)"
        );
    }

    let cc = c_compiler();
    if skip_for_missing_compiler(&cc) {
        return;
    }
    let mut c = String::from(
        "#include <stddef.h>\n#include <stdio.h>\n#include \"pounce.h\"\nint main(void) {\n",
    );
    for (name, _, fields) in &structs {
        c.push_str(&format!(
            "    printf(\"{name} sizeof %lu\\n\", (unsigned long)sizeof({name}));\n"
        ));
        for (f, _, _) in fields {
            c.push_str(&format!(
                "    printf(\"{name} {f} %lu\\n\", (unsigned long)offsetof({name}, {f}));\n"
            ));
            c.push_str(&format!(
                "    printf(\"{name} sizeof.{f} %lu\\n\", \
                 (unsigned long)sizeof((({name} *)0)->{f}));\n"
            ));
        }
    }
    c.push_str("    return 0;\n}\n");
    let work = work_dir("layout");
    let src = work.join("layout.c");
    std::fs::write(&src, c).expect("write layout.c");
    let lib_dir = lib_dir();
    let out = work.join("layout");
    let build = compile(&cc, &src, &out, &lib_dir);
    assert!(
        build.status.success(),
        "layout.c failed to build:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &lib_dir)
        .env("DYLD_LIBRARY_PATH", &lib_dir)
        .output()
        .expect("spawn layout binary");
    assert!(run.status.success(), "layout binary exited {}", run.status);
    let stdout = String::from_utf8_lossy(&run.stdout);
    let measured: std::collections::HashMap<(String, String), usize> = stdout
        .lines()
        .map(|l| {
            let w: Vec<&str> = l.split_whitespace().collect();
            assert_eq!(w.len(), 3, "unexpected line {l:?}");
            (
                (w[0].to_string(), w[1].to_string()),
                w[2].parse().expect("a number"),
            )
        })
        .collect();
    for (name, size, fields) in &structs {
        let get = |f: &str| measured[&(name.to_string(), f.to_string())];
        assert_eq!(get("sizeof"), *size, "sizeof({name}): C vs Rust");
        for (f, off, width) in fields {
            assert_eq!(get(f), *off, "offsetof({name}, {f}): C vs Rust");
            assert_eq!(
                get(&format!("sizeof.{f}")),
                *width,
                "sizeof({name}.{f}): C vs Rust"
            );
        }
    }
}

/// The parser above is what makes the field-list comparison bite; check it
/// on a header fragment with comments, arrays and pointers.
#[test]
fn header_struct_fields_parses_declarations() {
    let h = "typedef struct { int a; } Other;\n\
             typedef struct {\n  /** doc; with a semicolon */\n  char  s[16];\n  ipindex k; /* c */\n  double *p;\n} Wanted;\n";
    assert_eq!(header_struct_fields(h, "Wanted"), ["s", "k", "p"]);
    assert_eq!(header_struct_fields(h, "Other"), ["a"]);
}
