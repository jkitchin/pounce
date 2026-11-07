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

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The stdout line each example prints on success.
const EXPECTED_STDOUT: &[(&str, &str)] = &[("sqp_warm_start.c", "all 4 solves succeeded")];

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
fn lib_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    let deps = exe.parent().expect("test binary has a parent directory");
    let dir = deps
        .parent()
        .expect("deps/ has a parent profile directory")
        .to_path_buf();
    let lib = dir.join(cdylib_file_name());
    assert!(
        lib.is_file(),
        "expected the cdylib cargo builds for integration tests at {}, \
         but it is not there (test binary: {}). The lib target must keep \
         crate-type = [\"lib\", \"cdylib\"]; if this toolchain does not build \
         the cdylib for tests, build it first (`cargo build -p \
         pounce-cinterface`) — this test does not skip for a missing library.",
        lib.display(),
        exe.display()
    );
    dir
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
