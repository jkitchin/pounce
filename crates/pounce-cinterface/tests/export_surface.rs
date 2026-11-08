//! The exported C surface, the header, the docs and the tests must agree
//! (gh#1147, S1-I2).
//!
//! `header_abi.rs` checks the widths of the scalar typedefs. Nothing checked
//! the *functions*: a `#[no_mangle]` entry point could be added without a
//! prototype in `pounce.h` (so no C caller can reach it), a prototype could
//! outlive the function it declares (so a C caller links against nothing),
//! or the two could disagree on an argument (so every call is undefined
//! behaviour). This test is pure text analysis over the crate's own files —
//! it reads the source and the shipped header rather than restating either,
//! so editing one side alone fails here.
//!
//! What it enforces, for every exported name:
//!
//! 1. **Header.** The set of C exports in `src/lib.rs` + `src/solver.rs` + `src/qp.rs`
//!    equals the set of top-level prototypes in `include/pounce.h`.
//! 2. **Signature.** Same arity; each parameter and the return mapped to a
//!    width class (`i1`, `i32`, `f64`, `ptr`, `fnptr(<typedef>)`, `void`)
//!    and compared positionally. Function pointers keep their typedef name,
//!    so two callbacks swapped in the argument list are caught too.
//! 3. **Docs.** The name appears in backticks in the crate `README.md` and
//!    in `docs/src/c-api.md`. The Fortran shim's `ip*_` symbols (which have
//!    no header) must appear in the README's Fortran section.
//! 4. **Tests.** The name is called (`NAME(`) somewhere under `tests/` or
//!    `examples/`, or in a `#[cfg(test)]` module of `src/`.
//! 5. **Error path.** Some test in those same places carries a line
//!    `// error-path: NAME`, placed in the test that exercises the
//!    function's failure branch (NULL handle, bad index, no prior solve…).
//!
//! Unit tests at the bottom feed the comparison deliberately broken inputs,
//! so the guard is itself shown to fail on each kind of drift.

// Test code: a panicking unwrap/expect is the test failing, as intended.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Pre-existing exports that had no error-path test when this guard was
/// written. **Frozen: this list may only shrink.** Do not add to it — a new
/// entry point lands with its error-path test. The guard asserts every entry
/// is still an export, so a removed symbol cannot linger here.
///
/// Empty on the commit that introduced the guard: every export then had (or
/// was given) an error-path test.
const GRANDFATHERED_NO_ERROR_PATH: &[&str] = &[];

/// Width class of one argument or return value.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Class {
    I1,
    I32,
    F64,
    Ptr,
    FnPtr(String),
    Void,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Sig {
    params: Vec<Class>,
    ret: Class,
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(p: impl AsRef<Path>) -> String {
    let p = p.as_ref();
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

// ---------------------------------------------------------------------------
// Rust side
// ---------------------------------------------------------------------------

/// Map a Rust parameter/return type to its width class.
fn rust_class(ty: &str) -> Result<Class, String> {
    let ty = ty.trim();
    if ty.starts_with("*const") || ty.starts_with("*mut") {
        return Ok(Class::Ptr);
    }
    if let Some(inner) = ty.strip_prefix("Option<").and_then(|r| r.strip_suffix('>')) {
        let inner = inner.trim();
        if inner.ends_with("_CB") {
            return Ok(Class::FnPtr(inner.to_string()));
        }
        return Err(format!("unrecognised Option type `{ty}`"));
    }
    // Fortran callback typedefs are passed bare (non-nullable).
    if ty.ends_with("_CB") {
        return Ok(Class::FnPtr(ty.to_string()));
    }
    Ok(match ty {
        "Bool" => Class::I1,
        "Index" | "c_int" | "i32" | "IpoptBoundStatus" | "IpoptConsStatus" => Class::I32,
        "Number" | "f64" => Class::F64,
        "IpoptProblem" | "IpoptSolver" | "PounceQp" => Class::Ptr,
        "" | "()" => Class::Void,
        _ => return Err(format!("unrecognised Rust type `{ty}`")),
    })
}

/// Split at top-level commas (ignoring those inside `<>` / `()`).
fn split_top_level(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '<' | '(' => depth += 1,
            '>' | ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(ch);
    }
    out.push(cur);
    out.into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Every `#[unsafe(no_mangle)] pub unsafe extern "C" fn NAME(..) -> RET`
/// in `src`, with its signature. A `no_mangle` that is not followed within
/// three lines by such a function is an error (it would be an export this
/// guard cannot see).
fn rust_exports(src: &str) -> Result<BTreeMap<String, Sig>, String> {
    const FN_PREFIX: &str = "pub unsafe extern \"C\" fn ";
    let lines: Vec<&str> = src.lines().collect();
    let mut out = BTreeMap::new();
    for (i, line) in lines.iter().enumerate() {
        if line.trim() != "#[unsafe(no_mangle)]" {
            continue;
        }
        let Some(j) = (i + 1..(i + 4).min(lines.len()))
            .find(|&j| lines[j].trim_start().starts_with(FN_PREFIX))
        else {
            return Err(format!(
                "line {}: `#[unsafe(no_mangle)]` not followed by `{FN_PREFIX}NAME(` \
                 within three lines",
                i + 1
            ));
        };
        // Join from the `fn` line up to the body's `{`.
        let mut decl = String::new();
        for l in &lines[j..] {
            decl.push_str(l.trim());
            decl.push(' ');
            if l.trim_end().ends_with('{') {
                break;
            }
        }
        let rest = &decl[decl.find(FN_PREFIX).unwrap() + FN_PREFIX.len()..];
        let open = rest
            .find('(')
            .ok_or_else(|| format!("no `(` after fn at line {}", j + 1))?;
        let name = rest[..open].trim().to_string();
        // Matching `)`.
        let mut depth = 0i32;
        let mut close = None;
        for (k, ch) in rest[open..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + k);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close = close.ok_or_else(|| format!("unbalanced parens in `{name}`"))?;
        let mut params = Vec::new();
        for p in split_top_level(&rest[open + 1..close]) {
            let ty = p
                .split_once(':')
                .map(|(_, t)| t)
                .ok_or_else(|| format!("`{name}`: parameter `{p}` has no type"))?;
            params.push(rust_class(ty).map_err(|e| format!("`{name}`: {e}"))?);
        }
        let tail = rest[close + 1..].trim();
        let ret_ty = tail
            .strip_prefix("->")
            .map(|r| r.trim_end_matches('{').trim())
            .unwrap_or("");
        let ret = rust_class(ret_ty).map_err(|e| format!("`{name}` return: {e}"))?;
        if out.insert(name.clone(), Sig { params, ret }).is_some() {
            return Err(format!("`{name}` exported twice"));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Header side
// ---------------------------------------------------------------------------

fn strip_c_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
            out.push(' ');
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

/// Drop preprocessor lines, and the `#ifdef __cplusplus` blocks (which hold
/// the `extern "C" {` / `}` wrapper) whole.
fn strip_preprocessor(s: &str) -> String {
    let mut out = String::new();
    let mut in_cpp_block = false;
    for line in s.lines() {
        let t = line.trim();
        if in_cpp_block {
            if t.starts_with("#endif") {
                in_cpp_block = false;
            }
            continue;
        }
        if t.starts_with("#ifdef __cplusplus") {
            in_cpp_block = true;
            continue;
        }
        if t.starts_with('#') {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Map one C parameter declaration (or a bare return type) to its class.
/// `has_name` says whether the last identifier is a parameter name.
fn c_class(decl: &str, has_name: bool) -> Result<Class, String> {
    if decl.contains('*') {
        return Ok(Class::Ptr);
    }
    let mut words: Vec<&str> = decl.split_whitespace().filter(|w| *w != "const").collect();
    if has_name && words.len() > 1 {
        words.pop();
    }
    let ty = words.join(" ");
    if ty.starts_with("enum ") {
        return Ok(Class::I32);
    }
    if ty.ends_with("_CB") {
        return Ok(Class::FnPtr(ty));
    }
    Ok(match ty.as_str() {
        "bool" | "Bool" => Class::I1,
        "int" | "ipindex" | "Index" | "IpoptBoundStatus" | "IpoptConsStatus" => Class::I32,
        "ipnumber" | "Number" | "double" => Class::F64,
        "IpoptProblem" | "IpoptSolver" | "PounceQp" | "UserDataPtr" => Class::Ptr,
        "void" => Class::Void,
        _ => return Err(format!("unrecognised C type `{ty}` in `{decl}`")),
    })
}

/// Every top-level function prototype `RET NAME(ARGS);` in the header.
fn header_prototypes(header: &str) -> Result<BTreeMap<String, Sig>, String> {
    let text = strip_preprocessor(&strip_c_comments(header));
    let mut stmts = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in text.chars() {
        match ch {
            '{' => depth += 1,
            '}' => depth -= 1,
            ';' if depth == 0 => {
                stmts.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(ch);
    }
    let mut out = BTreeMap::new();
    for s in stmts {
        let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
        if s.is_empty() || s.starts_with("typedef") || s.contains('{') || !s.contains('(') {
            continue;
        }
        let open = s.find('(').unwrap();
        let close = s
            .rfind(')')
            .ok_or_else(|| format!("unbalanced prototype `{s}`"))?;
        let head = s[..open].trim();
        let (ret_ty, name) = head
            .rsplit_once([' ', '*'])
            .ok_or_else(|| format!("cannot split return type from name in `{s}`"))?;
        // `rsplit_once` drops the separator; keep a `*` on the return type.
        let ret_ty = if head[..head.len() - name.len()].ends_with('*') {
            format!("{ret_ty}*")
        } else {
            ret_ty.to_string()
        };
        let name = name.trim().to_string();
        let args = s[open + 1..close].trim();
        let params = if args.is_empty() || args == "void" {
            Vec::new()
        } else {
            args.split(',')
                .map(|a| c_class(a, true).map_err(|e| format!("`{name}`: {e}")))
                .collect::<Result<_, _>>()?
        };
        let ret = c_class(&ret_ty, false).map_err(|e| format!("`{name}` return: {e}"))?;
        if out.insert(name.clone(), Sig { params, ret }).is_some() {
            return Err(format!("`{name}` declared twice in the header"));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Every disagreement between the exports and the prototypes, as messages.
fn compare(exports: &BTreeMap<String, Sig>, protos: &BTreeMap<String, Sig>) -> Vec<String> {
    let mut problems = Vec::new();
    for name in exports.keys() {
        if !protos.contains_key(name) {
            problems.push(format!(
                "`{name}` is exported but has no prototype in pounce.h — no C caller can reach it"
            ));
        }
    }
    for name in protos.keys() {
        if !exports.contains_key(name) {
            problems.push(format!(
                "pounce.h declares `{name}` but the library does not export it — \
                 a C caller would fail to link"
            ));
        }
    }
    for (name, rs) in exports {
        let Some(cs) = protos.get(name) else { continue };
        if rs.params.len() != cs.params.len() {
            problems.push(format!(
                "`{name}`: arity differs — Rust takes {}, pounce.h declares {}",
                rs.params.len(),
                cs.params.len()
            ));
            continue;
        }
        for (k, (r, c)) in rs.params.iter().zip(&cs.params).enumerate() {
            if r != c {
                problems.push(format!(
                    "`{name}`: parameter {k} is {r:?} in Rust but {c:?} in pounce.h"
                ));
            }
        }
        if rs.ret != cs.ret {
            problems.push(format!(
                "`{name}`: returns {:?} in Rust but {:?} in pounce.h",
                rs.ret, cs.ret
            ));
        }
    }
    problems
}

// ---------------------------------------------------------------------------
// Docs / tests scanning
// ---------------------------------------------------------------------------

fn mentions_in_backticks(text: &str, name: &str) -> bool {
    text.contains(&format!("`{name}`")) || text.contains(&format!("`{name}("))
}

/// The README's section whose heading mentions Fortran, up to the next
/// heading of the same or higher level.
fn readme_fortran_section(readme: &str) -> Option<String> {
    let mut out = String::new();
    let mut level = None;
    for line in readme.lines() {
        let hashes = line.chars().take_while(|&c| c == '#').count();
        if hashes > 0 && line[hashes..].starts_with(' ') {
            match level {
                None if line.contains("Fortran") => {
                    level = Some(hashes);
                    continue;
                }
                Some(l) if hashes <= l => break,
                _ => {}
            }
        }
        if level.is_some() {
            out.push_str(line);
            out.push('\n');
        }
    }
    level.map(|_| out)
}

/// The text a "tested somewhere" check may look in: every file under
/// `tests/` and `examples/` except this one, plus the `#[cfg(test)]` part of
/// each `src/*.rs` (everything after its first `#[cfg(test)]` line).
fn test_corpus() -> String {
    let root = manifest_dir();
    let mut corpus = String::new();
    for dir in ["tests", "examples"] {
        let mut entries: Vec<_> = std::fs::read_dir(root.join(dir))
            .unwrap_or_else(|e| panic!("read_dir {dir}: {e}"))
            .map(|e| e.unwrap().path())
            .filter(|p| p.is_file())
            .collect();
        entries.sort();
        for p in entries {
            if p.file_name().and_then(|f| f.to_str()) == Some("export_surface.rs") {
                continue;
            }
            corpus.push_str(&read(&p));
            corpus.push('\n');
        }
    }
    for f in ["lib.rs", "solver.rs", "qp.rs", "fortran.rs"] {
        let src = read(root.join("src").join(f));
        if let Some(pos) = src.find("#[cfg(test)]") {
            corpus.push_str(&src[pos..]);
            corpus.push('\n');
        }
    }
    corpus
}

/// Names `N` on lines of the form `// error-path: N`.
fn error_path_markers(corpus: &str) -> BTreeSet<String> {
    let marker = concat!("// error", "-path:");
    corpus
        .lines()
        .filter_map(|l| l.trim().strip_prefix(marker))
        .map(|n| n.trim().to_string())
        .collect()
}

fn called(corpus: &str, name: &str) -> bool {
    let needle = format!("{name}(");
    corpus.match_indices(&needle).any(|(i, _)| {
        // Reject a longer identifier that merely ends in `name`.
        corpus[..i]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'))
    })
}

struct Surface {
    c: BTreeMap<String, Sig>,
    fortran: BTreeMap<String, Sig>,
}

fn surface() -> Surface {
    let root = manifest_dir();
    let mut c = BTreeMap::new();
    for f in ["lib.rs", "solver.rs", "qp.rs"] {
        let ex = rust_exports(&read(root.join("src").join(f)))
            .unwrap_or_else(|e| panic!("src/{f}: {e}"));
        for (k, v) in ex {
            assert!(c.insert(k.clone(), v).is_none(), "`{k}` exported twice");
        }
    }
    let fortran = rust_exports(&read(root.join("src/fortran.rs")))
        .unwrap_or_else(|e| panic!("src/fortran.rs: {e}"));
    assert!(
        !c.is_empty() && !fortran.is_empty(),
        "export scan found nothing"
    );
    Surface { c, fortran }
}

fn all_names(s: &Surface) -> Vec<String> {
    s.c.keys().chain(s.fortran.keys()).cloned().collect()
}

// ---------------------------------------------------------------------------
// The guard
// ---------------------------------------------------------------------------

#[test]
fn every_export_has_a_matching_header_prototype() {
    let s = surface();
    let protos = header_prototypes(include_str!("../include/pounce.h"))
        .unwrap_or_else(|e| panic!("pounce.h: {e}"));
    let problems = compare(&s.c, &protos);
    assert!(
        problems.is_empty(),
        "pounce.h and the exported functions disagree:\n  {}",
        problems.join("\n  ")
    );
    // The scan saw the whole surface, not a prefix of it.
    assert_eq!(s.c.len(), protos.len());
}

#[test]
fn fortran_shim_exports_exactly_the_upstream_symbols() {
    let s = surface();
    let names: Vec<&str> = s.fortran.keys().map(String::as_str).collect();
    assert_eq!(
        names,
        [
            "ipaddintoption_",
            "ipaddnumoption_",
            "ipaddstroption_",
            "ipcreate_",
            "ipfree_",
            "ipsetcallback_",
            "ipsolve_",
            "ipunsetcallback_",
        ],
        "the Fortran shim mirrors IpStdFInterface.c's eight symbols; a change \
         here is an ABI change and needs the README's Fortran section updated"
    );
}

#[test]
fn every_export_is_documented() {
    let s = surface();
    let root = manifest_dir();
    let readme = read(root.join("README.md"));
    let book = read(root.join("../../docs/src/c-api.md"));
    let fortran_section = readme_fortran_section(&readme)
        .expect("README.md has no section whose heading names Fortran");
    let mut missing = Vec::new();
    for name in s.c.keys() {
        if !mentions_in_backticks(&readme, name) {
            missing.push(format!(
                "`{name}` missing from crates/pounce-cinterface/README.md"
            ));
        }
        if !mentions_in_backticks(&book, name) {
            missing.push(format!("`{name}` missing from docs/src/c-api.md"));
        }
    }
    for name in s.fortran.keys() {
        if !mentions_in_backticks(&fortran_section, name) {
            missing.push(format!("`{name}` missing from README.md's Fortran section"));
        }
        if !mentions_in_backticks(&book, name) {
            missing.push(format!("`{name}` missing from docs/src/c-api.md"));
        }
    }
    assert!(
        missing.is_empty(),
        "undocumented exports:\n  {}",
        missing.join("\n  ")
    );
}

#[test]
fn every_export_is_called_by_a_test() {
    let s = surface();
    let corpus = test_corpus();
    let missing: Vec<String> = all_names(&s)
        .into_iter()
        .filter(|n| !called(&corpus, n))
        .collect();
    assert!(
        missing.is_empty(),
        "exports no test or example calls: {missing:?}"
    );
}

#[test]
fn every_export_has_an_error_path_test() {
    let s = surface();
    let names: BTreeSet<String> = all_names(&s).into_iter().collect();
    let markers = error_path_markers(&test_corpus());

    for g in GRANDFATHERED_NO_ERROR_PATH {
        assert!(
            names.contains(*g),
            "GRANDFATHERED_NO_ERROR_PATH names `{g}`, which is no longer exported — remove it"
        );
        assert!(
            !markers.contains(*g),
            "`{g}` now has an error-path test — remove it from GRANDFATHERED_NO_ERROR_PATH"
        );
    }
    let missing: Vec<&String> = names
        .iter()
        .filter(|n| !markers.contains(*n) && !GRANDFATHERED_NO_ERROR_PATH.contains(&n.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "exports with no `// error-path: NAME` marker in any test: {missing:?}"
    );
    let stale: Vec<&String> = markers.iter().filter(|m| !names.contains(*m)).collect();
    assert!(
        stale.is_empty(),
        "error-path markers naming no export (typo or removed symbol): {stale:?}"
    );
}

// ---------------------------------------------------------------------------
// The guard catches what it claims to catch
// ---------------------------------------------------------------------------

const RUST_OK: &str = r#"
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Foo(
    p: IpoptProblem,
    cb: Option<Eval_F_CB>,
    g: Option<Eval_G_CB>,
    x: *mut Number,
    k: *const c_char,
    n: Index,
) -> Bool {
    body()
}
/// docs
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Bar(v: Number) {
}
"#;

const HEADER_OK: &str = r#"
#define X 1
#ifdef __cplusplus
extern "C" {
#endif
/* Foo does a thing; Foo(void) in a comment is not a prototype. */
typedef bool (*Eval_F_CB)(ipindex n, UserDataPtr u);
typedef struct { int a; double b; } S;
enum E { A = 0, B = 1 };
bool Foo(IpoptProblem p, Eval_F_CB cb, Eval_G_CB g, ipnumber* x,
         const char *k, ipindex n);
void Bar(ipnumber v); // trailing
#ifdef __cplusplus
}
#endif
"#;

fn check(rust: &str, header: &str) -> Vec<String> {
    compare(
        &rust_exports(rust).unwrap(),
        &header_prototypes(header).unwrap(),
    )
}

#[test]
fn guard_accepts_an_agreeing_pair() {
    assert_eq!(check(RUST_OK, HEADER_OK), Vec::<String>::new());
    let p = header_prototypes(HEADER_OK).unwrap();
    assert_eq!(p.keys().collect::<Vec<_>>(), ["Bar", "Foo"]);
}

#[test]
fn guard_rejects_an_export_without_a_prototype() {
    let header = HEADER_OK.replace("void Bar(ipnumber v); // trailing", "");
    let p = check(RUST_OK, &header);
    assert!(
        p.iter()
            .any(|m| m.contains("`Bar` is exported but has no prototype")),
        "{p:?}"
    );
}

#[test]
fn guard_rejects_a_phantom_prototype() {
    let header = HEADER_OK.replace("void Bar(", "void Baz(ipindex q);\nvoid Bar(");
    let p = check(RUST_OK, &header);
    assert!(p.iter().any(|m| m.contains("declares `Baz`")), "{p:?}");
}

#[test]
fn guard_rejects_an_arity_mismatch() {
    let header = HEADER_OK.replace("ipindex n);", "ipindex n, ipindex extra);");
    let p = check(RUST_OK, &header);
    assert!(
        p.iter().any(|m| m.contains("`Foo`: arity differs")),
        "{p:?}"
    );
}

#[test]
fn guard_rejects_a_width_mismatch() {
    // `Bool` (1 byte) vs `int` (4) — the gh#624 shape, now on a parameter.
    let header = HEADER_OK.replace("void Bar(ipnumber v)", "void Bar(ipindex v)");
    let p = check(RUST_OK, &header);
    assert!(
        p.iter().any(|m| m.contains("`Bar`: parameter 0 is F64")),
        "{p:?}"
    );
    let header = HEADER_OK.replace("bool Foo(", "int Foo(");
    let p = check(RUST_OK, &header);
    assert!(p.iter().any(|m| m.contains("`Foo`: returns I1")), "{p:?}");
}

#[test]
fn guard_rejects_swapped_callbacks() {
    let header = HEADER_OK.replace("Eval_F_CB cb, Eval_G_CB g", "Eval_G_CB g, Eval_F_CB cb");
    let p = check(RUST_OK, &header);
    assert!(p.iter().any(|m| m.contains("`Foo`: parameter 1")), "{p:?}");
}

#[test]
fn guard_rejects_an_unparseable_export() {
    let rust = "#[unsafe(no_mangle)]\nfn hidden() {}\n";
    assert!(rust_exports(rust).is_err());
    let rust = "#[unsafe(no_mangle)]\npub unsafe extern \"C\" fn W(x: Weird) {}\n";
    assert!(rust_exports(rust).unwrap_err().contains("Weird"));
}

#[test]
fn doc_and_marker_scanners_branch_correctly() {
    assert!(mentions_in_backticks("see `Foo` here", "Foo"));
    assert!(mentions_in_backticks("`Foo(a, b)`", "Foo"));
    assert!(!mentions_in_backticks("Foo without ticks", "Foo"));
    assert!(!mentions_in_backticks("`FooBar`", "Foo"));

    let readme = "# T\n## Fortran shim\n`ipfree_`\n### sub\n`ipsolve_`\n## Other\n`ipcreate_`\n";
    let sec = readme_fortran_section(readme).unwrap();
    assert!(sec.contains("ipfree_") && sec.contains("ipsolve_"));
    assert!(!sec.contains("ipcreate_"));
    assert!(readme_fortran_section("# T\n## C only\n").is_none());

    assert!(called("x = Foo(1);", "Foo"));
    assert!(!called("x = BarFoo(1);", "Foo"));
    assert!(!called("Foo is mentioned", "Foo"));

    let m = error_path_markers("  // error-path: Foo\n// error-path:Bar \n/// error-path: Baz\n");
    assert_eq!(m.into_iter().collect::<Vec<_>>(), ["Bar", "Foo"]);
}
