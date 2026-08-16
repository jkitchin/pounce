//! The header and the implementation must agree about the width of every
//! scalar they exchange (gh#624).
//!
//! `pounce.h` is the contract: a C caller compiles against it and gets
//! whatever it says. Nothing else checks that the Rust side agrees, and
//! for `Bool` it did not — the header said `typedef bool Bool` (one byte,
//! matching Ipopt 3.14) while the implementation used `c_int` (four).
//! Both sides "worked" because x86-64 compilers zero-extend booleans in
//! practice, but the psABI leaves the upper bits of a `bool` return
//! unspecified, so a callback answering `false` — *"I cannot evaluate
//! here"* — was one compiler decision away from being read as success.
//!
//! This test reads the typedefs out of the shipped header rather than
//! restating them, so editing either side alone fails here.

use std::collections::HashMap;

/// Width of each C type the header is allowed to alias, on the platforms
/// pounce targets (LP64 / LLP64 — `int` is 4 bytes on both).
fn c_type_size(c_type: &str) -> Option<usize> {
    Some(match c_type {
        "bool" => 1,
        "int" => 4,
        "double" => 8,
        "float" => 4,
        "int64_t" => 8,
        _ => return None,
    })
}

/// Pull `typedef <c_type> <alias>;` pairs out of the header.
fn header_typedefs() -> HashMap<String, String> {
    let header = include_str!("../include/pounce.h");
    let mut out = HashMap::new();
    for line in header.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("typedef ") else {
            continue;
        };
        let Some(decl) = rest.strip_suffix(';') else {
            continue;
        };
        // `typedef ipnumber Number;` → ("ipnumber", "Number"). Skip
        // function-pointer and struct typedefs, which have punctuation.
        let words: Vec<&str> = decl.split_whitespace().collect();
        if words.len() != 2 || words.iter().any(|w| w.contains(['(', ')', '*'])) {
            continue;
        }
        out.insert(words[1].to_string(), words[0].to_string());
    }
    out
}

/// Resolve an alias chain (`Number` → `ipnumber` → `double`) to a width.
fn declared_size(alias: &str, typedefs: &HashMap<String, String>) -> usize {
    let mut name = alias.to_string();
    for _ in 0..8 {
        if let Some(size) = c_type_size(&name) {
            return size;
        }
        name = typedefs
            .get(&name)
            .unwrap_or_else(|| panic!("`{alias}` resolves to unknown C type `{name}`"))
            .clone();
    }
    panic!("typedef chain for `{alias}` does not terminate");
}

#[test]
fn header_scalars_match_the_implementation() {
    let typedefs = header_typedefs();

    for (alias, rust_size) in [
        ("Bool", size_of::<pounce_cinterface::Bool>()),
        ("Index", size_of::<pounce_cinterface::Index>()),
        ("Number", size_of::<pounce_cinterface::Number>()),
        ("ipindex", size_of::<pounce_cinterface::Index>()),
        ("ipnumber", size_of::<pounce_cinterface::Number>()),
    ] {
        let header_size = declared_size(alias, &typedefs);
        assert_eq!(
            header_size,
            rust_size,
            "pounce.h declares `{alias}` as {header_size} byte(s) via `{}`, \
             but the implementation uses {rust_size}. Every C caller compiled \
             against the header would disagree with this library about every \
             `{alias}` it passes or returns.",
            typedefs.get(alias).map(String::as_str).unwrap_or(alias),
        );
    }
}

/// The specific pairing gh#624 got wrong, spelled out so a future edit to
/// either side names the reason rather than just a size.
#[test]
fn bool_is_the_c99_bool_the_header_promises() {
    let typedefs = header_typedefs();
    assert_eq!(
        typedefs.get("Bool").map(String::as_str),
        Some("bool"),
        "pounce.h must keep `typedef bool Bool` — it is what Ipopt 3.14's \
         IpStdCInterface.h declares, and source-level drop-in compatibility \
         for cyipopt / GAMS / the CasADi plugin depends on it"
    );
    assert_eq!(
        size_of::<pounce_cinterface::Bool>(),
        1,
        "the Rust side must be one byte to match"
    );
}
