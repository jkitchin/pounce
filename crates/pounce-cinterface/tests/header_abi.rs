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

/// `POUNCE_VERSION_*` in the header must name the library it ships with.
///
/// They said `0.1.0` from the first release until 0.16.0 and nothing
/// noticed: a C caller gating on `POUNCE_VERSION_MINOR` was told it had a
/// library fifteen minors older than the one `GetIpoptVersion` reports.
/// `scripts/check-release-consistency.sh` repeats this check so a version
/// bump that forgets the header fails before the tag, not only here.
#[test]
fn header_version_defines_match_the_crate() {
    let header = include_str!("../include/pounce.h");
    let define = |name: &str| -> String {
        let prefix = format!("#define {name} ");
        let mut hits = header
            .lines()
            .filter_map(|l| l.trim().strip_prefix(prefix.as_str()))
            .map(|v| v.trim().to_string());
        let v = hits
            .next()
            .unwrap_or_else(|| panic!("pounce.h has no `#define {name}`"));
        assert!(hits.next().is_none(), "pounce.h defines `{name}` twice");
        v
    };
    let crate_version = env!("CARGO_PKG_VERSION");
    let mut parts = crate_version.split(['.', '-', '+']);
    for (name, want) in [
        ("POUNCE_VERSION_MAJOR", parts.next()),
        ("POUNCE_VERSION_MINOR", parts.next()),
        ("POUNCE_VERSION_PATCH", parts.next()),
    ] {
        assert_eq!(
            Some(define(name).as_str()),
            want,
            "pounce.h `{name}` disagrees with the crate version {crate_version}"
        );
    }
    assert_eq!(
        define("POUNCE_VERSION"),
        format!("\"{crate_version}\""),
        "pounce.h `POUNCE_VERSION` disagrees with the crate version"
    );
}

/// The `POUNCE_CC_*` and `POUNCE_COMPLEMENTS_*` codes are an ABI too: a C
/// caller compares the `int` it gets back against the header's value, the
/// library returns the Rust constant's. Read every `#define POUNCE_CC_` and
/// `#define POUNCE_COMPLEMENTS_` out of the header and hold
/// the two sides to the same names and values, so adding a code to one
/// side only, or renumbering one, fails here (gh#1147).
#[test]
fn header_complementarity_codes_match_the_implementation() {
    use pounce_cinterface::*;
    let rust: HashMap<&str, i64> = [
        ("POUNCE_CC_CLASS_S", POUNCE_CC_CLASS_S),
        ("POUNCE_CC_CLASS_W", POUNCE_CC_CLASS_W),
        ("POUNCE_CC_CLASS_NONE", POUNCE_CC_CLASS_NONE),
        ("POUNCE_CC_CLASS_NOT_SEARCHED", POUNCE_CC_CLASS_NOT_SEARCHED),
        ("POUNCE_CC_CLASS_ABSENT", POUNCE_CC_CLASS_ABSENT),
        ("POUNCE_CC_REPORTED_ANSWER", POUNCE_CC_REPORTED_ANSWER),
        (
            "POUNCE_CC_REPORTED_WEAK_FALLBACK",
            POUNCE_CC_REPORTED_WEAK_FALLBACK,
        ),
        (
            "POUNCE_CC_REPORTED_FEASIBLE_UNCERTIFIED",
            POUNCE_CC_REPORTED_FEASIBLE_UNCERTIFIED,
        ),
        ("POUNCE_CC_REPORTED_VERDICT", POUNCE_CC_REPORTED_VERDICT),
        ("POUNCE_CC_B_VERDICT_B", POUNCE_CC_B_VERDICT_B),
        ("POUNCE_CC_B_VERDICT_NOT_B", POUNCE_CC_B_VERDICT_NOT_B),
        (
            "POUNCE_CC_B_VERDICT_NOT_CHECKED",
            POUNCE_CC_B_VERDICT_NOT_CHECKED,
        ),
        ("POUNCE_CC_B_VERDICT_ABSENT", POUNCE_CC_B_VERDICT_ABSENT),
        ("POUNCE_CC_TEXT_SUMMARY", POUNCE_CC_TEXT_SUMMARY),
        ("POUNCE_CC_TEXT_POLISH_REASON", POUNCE_CC_TEXT_POLISH_REASON),
        ("POUNCE_CC_TEXT_B_REASON", POUNCE_CC_TEXT_B_REASON),
        (
            "POUNCE_COMPLEMENTS_PAIR_MIXED",
            POUNCE_COMPLEMENTS_PAIR_MIXED,
        ),
        (
            "POUNCE_COMPLEMENTS_PAIR_LOWER",
            POUNCE_COMPLEMENTS_PAIR_LOWER,
        ),
        (
            "POUNCE_COMPLEMENTS_PAIR_UPPER",
            POUNCE_COMPLEMENTS_PAIR_UPPER,
        ),
        (
            "POUNCE_COMPLEMENTS_ROW_FIXED_NO_PAIR",
            POUNCE_COMPLEMENTS_ROW_FIXED_NO_PAIR,
        ),
        (
            "POUNCE_COMPLEMENTS_VAR_FIXED_NO_PAIR",
            POUNCE_COMPLEMENTS_VAR_FIXED_NO_PAIR,
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k, i64::from(v)))
    .collect();

    let header = include_str!("../include/pounce.h");
    let mut defined: HashMap<String, i64> = HashMap::new();
    for line in header.lines() {
        let Some(rest) = line.trim().strip_prefix("#define ") else {
            continue;
        };
        let mut words = rest.split_whitespace();
        let Some(name) = words
            .next()
            .filter(|n| n.starts_with("POUNCE_CC_") || n.starts_with("POUNCE_COMPLEMENTS_"))
        else {
            continue;
        };
        let value = words
            .next()
            .unwrap_or_else(|| panic!("`#define {name}` has no value"));
        let value: i64 = value
            .parse()
            .unwrap_or_else(|_| panic!("`#define {name} {value}` is not an integer literal"));
        assert!(
            defined.insert(name.to_string(), value).is_none(),
            "pounce.h defines `{name}` twice"
        );
    }

    for (name, value) in &rust {
        assert_eq!(
            defined.get(*name),
            Some(value),
            "pounce.h `{name}` disagrees with the Rust constant {value} (or is missing)"
        );
    }
    for name in defined.keys() {
        assert!(
            rust.contains_key(name.as_str()),
            "pounce.h defines `{name}` but this test does not know the Rust constant; \
             add it to the list (and to the library, if it is missing there)"
        );
    }
}

/// The matrix-form QP entry's codes (gh#1147 Stage 3): every
/// `#define POUNCE_QP_` and `#define POUNCE_CONE_` in the header against the
/// Rust constant of the same name, both ways, as for the complementarity
/// codes above.
#[test]
fn header_qp_codes_match_the_implementation() {
    use pounce_cinterface::*;
    let rust: HashMap<&str, i64> = [
        ("POUNCE_QP_STATUS_OPTIMAL", POUNCE_QP_STATUS_OPTIMAL),
        (
            "POUNCE_QP_STATUS_OPTIMAL_INACCURATE",
            POUNCE_QP_STATUS_OPTIMAL_INACCURATE,
        ),
        (
            "POUNCE_QP_STATUS_PRIMAL_INFEASIBLE",
            POUNCE_QP_STATUS_PRIMAL_INFEASIBLE,
        ),
        (
            "POUNCE_QP_STATUS_DUAL_INFEASIBLE",
            POUNCE_QP_STATUS_DUAL_INFEASIBLE,
        ),
        (
            "POUNCE_QP_STATUS_ITERATION_LIMIT",
            POUNCE_QP_STATUS_ITERATION_LIMIT,
        ),
        ("POUNCE_QP_STATUS_TIME_LIMIT", POUNCE_QP_STATUS_TIME_LIMIT),
        (
            "POUNCE_QP_STATUS_NUMERICAL_FAILURE",
            POUNCE_QP_STATUS_NUMERICAL_FAILURE,
        ),
        ("POUNCE_CONE_NONNEG", POUNCE_CONE_NONNEG),
        ("POUNCE_CONE_SOC", POUNCE_CONE_SOC),
        ("POUNCE_CONE_EXP", POUNCE_CONE_EXP),
        ("POUNCE_CONE_POW", POUNCE_CONE_POW),
        ("POUNCE_CONE_PSD", POUNCE_CONE_PSD),
        ("POUNCE_QP_CERT_NONE", POUNCE_QP_CERT_NONE),
        (
            "POUNCE_QP_CERT_PRIMAL_INFEASIBLE",
            POUNCE_QP_CERT_PRIMAL_INFEASIBLE,
        ),
        (
            "POUNCE_QP_CERT_DUAL_INFEASIBLE",
            POUNCE_QP_CERT_DUAL_INFEASIBLE,
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k, i64::from(v)))
    .collect();

    let header = include_str!("../include/pounce.h");
    let mut defined: HashMap<String, i64> = HashMap::new();
    for line in header.lines() {
        let Some(rest) = line.trim().strip_prefix("#define ") else {
            continue;
        };
        let mut words = rest.split_whitespace();
        let Some(name) = words
            .next()
            .filter(|n| n.starts_with("POUNCE_QP_") || n.starts_with("POUNCE_CONE_"))
        else {
            continue;
        };
        let value = words
            .next()
            .unwrap_or_else(|| panic!("`#define {name}` has no value"));
        let value: i64 = value
            .parse()
            .unwrap_or_else(|_| panic!("`#define {name} {value}` is not an integer literal"));
        assert!(
            defined.insert(name.to_string(), value).is_none(),
            "pounce.h defines `{name}` twice"
        );
    }
    for (name, value) in &rust {
        assert_eq!(
            defined.get(*name),
            Some(value),
            "pounce.h `{name}` disagrees with the Rust constant {value} (or is missing)"
        );
    }
    for name in defined.keys() {
        assert!(
            rust.contains_key(name.as_str()),
            "pounce.h defines `{name}` but this test does not know the Rust constant; \
             add it to the list (and to the library, if it is missing there)"
        );
    }
}

/// The bound-update codes (gh#1147 Stage 2): every `#define POUNCE_BOUNDS_`
/// in the header against the Rust constant of the same name, both ways.
#[test]
fn header_bounds_codes_match_the_implementation() {
    use pounce_cinterface::*;
    let rust: HashMap<&str, i64> = [
        ("POUNCE_BOUNDS_VIEW_USER", POUNCE_BOUNDS_VIEW_USER),
        ("POUNCE_BOUNDS_VIEW_EFFECTIVE", POUNCE_BOUNDS_VIEW_EFFECTIVE),
    ]
    .into_iter()
    .map(|(k, v)| (k, i64::from(v)))
    .collect();

    let header = include_str!("../include/pounce.h");
    let mut defined: HashMap<String, i64> = HashMap::new();
    for line in header.lines() {
        let Some(rest) = line.trim().strip_prefix("#define ") else {
            continue;
        };
        let mut words = rest.split_whitespace();
        let Some(name) = words.next().filter(|n| n.starts_with("POUNCE_BOUNDS_")) else {
            continue;
        };
        let value: i64 = words
            .next()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("`#define {name}` has no integer value"));
        assert!(
            defined.insert(name.to_string(), value).is_none(),
            "pounce.h defines `{name}` twice"
        );
    }
    for (name, value) in &rust {
        assert_eq!(
            defined.get(*name),
            Some(value),
            "pounce.h `{name}` disagrees with the Rust constant {value} (or is missing)"
        );
    }
    for name in defined.keys() {
        assert!(
            rust.contains_key(name.as_str()),
            "pounce.h defines `{name}` but this test does not know the Rust constant"
        );
    }
}

/// The shared error namespace: every `#define POUNCE_OK` / `POUNCE_ERR_*`
/// in the header against `POUNCE_ERROR_CODES`, the table
/// `PounceErrorString` reads, both ways. So a code defined on one side
/// only, renumbered on one side, or left without a message fails here.
/// Each value must also be unique and sit in its name's block, and each
/// message must be distinct, so the code-to-text map is a bijection.
#[test]
fn header_error_codes_match_the_implementation() {
    use pounce_cinterface::*;
    use std::ffi::CStr;

    let header = include_str!("../include/pounce.h");
    let mut defined: HashMap<String, i64> = HashMap::new();
    for line in header.lines() {
        let Some(rest) = line.trim().strip_prefix("#define ") else {
            continue;
        };
        let mut words = rest.split_whitespace();
        let Some(name) = words
            .next()
            .filter(|n| *n == "POUNCE_OK" || n.starts_with("POUNCE_ERR_"))
        else {
            continue;
        };
        let value: i64 = words
            .next()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("`#define {name}` has no integer value"));
        assert!(
            defined.insert(name.to_string(), value).is_none(),
            "pounce.h defines `{name}` twice"
        );
    }
    assert!(defined.len() > 1, "no POUNCE_ERR_* defines found");

    let rust: HashMap<&str, i64> = POUNCE_ERROR_CODES
        .iter()
        .map(|(n, c, _)| (*n, i64::from(*c)))
        .collect();
    assert_eq!(rust.len(), POUNCE_ERROR_CODES.len(), "a name listed twice");
    for (name, value) in &rust {
        assert_eq!(
            defined.get(*name),
            Some(value),
            "pounce.h `{name}` disagrees with the Rust table's {value} (or is missing)"
        );
    }
    for name in defined.keys() {
        assert!(
            rust.contains_key(name.as_str()),
            "pounce.h defines `{name}` but POUNCE_ERROR_CODES has no entry (no message)"
        );
    }

    let mut values = std::collections::HashSet::new();
    let mut messages = std::collections::HashSet::new();
    for &(name, code, msg) in POUNCE_ERROR_CODES {
        assert!(values.insert(code), "`{name}` reuses the value {code}");
        let text = unsafe { CStr::from_ptr(PounceErrorString(code)) };
        assert_eq!(
            text, msg,
            "PounceErrorString({code}) is not `{name}`'s text"
        );
        assert_ne!(text.to_bytes(), b"unknown error code");
        assert!(messages.insert(msg), "`{name}` shares its message");
        let block = match name {
            "POUNCE_OK" => 0..=0,
            "POUNCE_ERR_NULL"
            | "POUNCE_ERR_COUNT"
            | "POUNCE_ERR_OUT_OF_RANGE"
            | "POUNCE_ERR_DUPLICATE"
            | "POUNCE_ERR_NAN"
            | "POUNCE_ERR_FLAGS"
            | "POUNCE_ERR_ARGUMENT"
            | "POUNCE_ERR_INTERNAL" => -19..=-1,
            "POUNCE_ERR_VAR_ALREADY_PAIRED"
            | "POUNCE_ERR_ROW_ALREADY_PAIRED"
            | "POUNCE_ERR_SHAPE"
            | "POUNCE_ERR_ROW_NOT_FREE"
            | "POUNCE_ERR_VAR_BOUNDS" => -39..=-20,
            "POUNCE_ERR_PAIR_CHANGED"
            | "POUNCE_ERR_COMPLEMENTS_CLASS"
            | "POUNCE_ERR_WORKING_SET" => -59..=-40,
            "POUNCE_ERR_INCOMPLETE"
            | "POUNCE_ERR_CONES"
            | "POUNCE_ERR_OPTIONS"
            | "POUNCE_ERR_INDEFINITE" => -79..=-60,
            other => panic!("`{other}` is in no block: assign it one here"),
        };
        assert!(
            block.contains(&code),
            "`{name}` = {code} is outside its block"
        );
    }
    // error-path: PounceErrorString
    for unknown in [1, 7, -9, -19, -25, -1000, i32::MIN, i32::MAX] {
        let text = unsafe { CStr::from_ptr(PounceErrorString(unknown)) };
        assert_eq!(text.to_bytes(), b"unknown error code", "code {unknown}");
    }
}
