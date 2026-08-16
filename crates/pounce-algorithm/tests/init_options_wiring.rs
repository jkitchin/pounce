//! Cold-start (`DefaultIterateInitializer`) option wiring — gh#604.
//!
//! The eight knobs below had read sites in
//! `IpoptApplication::algorithm_builder_from_options` all along, but no
//! registry entries, so every frontend refused them at the *set* call
//! with `Unknown option "bound_push"`. That is the inverse of gh#551
//! (registered-but-unread): the feature ran, and the only way to reach
//! it was to not use the option system.
//!
//! Three things are asserted here:
//!
//! 1. each option is settable, flows to the builder, and defaults to
//!    the value the builder already hard-coded (so registering them
//!    moved no trajectory);
//! 2. the registered ranges refuse the values upstream refuses, and an
//!    unimplemented *mode* fails with a message instead of quietly
//!    running a different one;
//! 3. the bidirectional registry invariant — no option is read without
//!    being registered, and no Initialization option is registered
//!    without being either consumed or explicitly refused.

use pounce_algorithm::alg_builder::AlgorithmBuilder;
use pounce_algorithm::application::IpoptApplication;
use std::collections::BTreeSet;

fn builder_from(setup: impl FnOnce(&mut IpoptApplication)) -> AlgorithmBuilder {
    let mut app = IpoptApplication::new();
    setup(&mut app);
    app.algorithm_builder_from_options()
}

/// The cold-start options, as the issue lists them.
const COLD_START_OPTIONS: &[&str] = &[
    "bound_push",
    "bound_frac",
    "slack_bound_push",
    "slack_bound_frac",
    "constr_mult_init_max",
    "bound_mult_init_val",
    "bound_mult_init_method",
    "least_square_init_primal",
];

#[test]
fn every_cold_start_option_is_registered() {
    let app = IpoptApplication::new();
    let reg = app.registered_options();
    for name in COLD_START_OPTIONS {
        let opt = reg
            .get_option(name)
            .unwrap_or_else(|| panic!("`{name}` is read by the builder but not registered"));
        assert_eq!(
            opt.category, "Initialization",
            "`{name}` should register under upstream's Initialization category"
        );
    }
}

#[test]
fn cold_start_defaults_match_the_hard_coded_builder_values() {
    // Registering an option whose default disagrees with the struct
    // default silently re-tunes the solver. These must agree exactly.
    let b = builder_from(|_| {}).init;
    assert_eq!(b.bound_push, 1e-2);
    assert_eq!(b.bound_frac, 1e-2);
    assert_eq!(b.slack_bound_push, 1e-2);
    assert_eq!(b.slack_bound_frac, 1e-2);
    assert_eq!(b.constr_mult_init_max, 1e3);
    assert_eq!(b.bound_mult_init_val, 1.0);
    assert_eq!(b.bound_mult_init_method, "constant");
    assert!(!b.least_square_init_primal);

    let app = IpoptApplication::new();
    let reg = app.registered_options();
    for (name, expected) in [
        ("bound_push", 1e-2),
        ("bound_frac", 1e-2),
        ("slack_bound_push", 1e-2),
        ("slack_bound_frac", 1e-2),
        ("constr_mult_init_max", 1e3),
        ("bound_mult_init_val", 1.0),
    ] {
        let opt = reg.get_option(name).expect("registered");
        match opt.default {
            pounce_common::reg_options::DefaultValue::Number(d) => assert_eq!(
                d, expected,
                "registered default for `{name}` must equal the builder's"
            ),
            ref other => panic!("`{name}` should be a number option, got {other:?}"),
        }
    }
}

#[test]
fn numeric_cold_start_overrides_flow_through() {
    let b = builder_from(|app| {
        let o = app.options_mut();
        o.set_numeric_value("bound_push", 0.25, true, false)
            .unwrap();
        o.set_numeric_value("bound_frac", 0.3, true, false).unwrap();
        o.set_numeric_value("slack_bound_push", 0.4, true, false)
            .unwrap();
        o.set_numeric_value("slack_bound_frac", 0.45, true, false)
            .unwrap();
        o.set_numeric_value("constr_mult_init_max", 0.0, true, false)
            .unwrap();
        o.set_numeric_value("bound_mult_init_val", 7.5, true, false)
            .unwrap();
    })
    .init;

    assert_eq!(b.bound_push, 0.25);
    assert_eq!(b.bound_frac, 0.3);
    assert_eq!(b.slack_bound_push, 0.4);
    assert_eq!(b.slack_bound_frac, 0.45);
    // 0 is the documented "discard the least-square guess" value and
    // must survive: the lower bound is non-strict.
    assert_eq!(b.constr_mult_init_max, 0.0);
    assert_eq!(b.bound_mult_init_val, 7.5);
}

#[test]
fn string_cold_start_overrides_flow_through() {
    let b = builder_from(|app| {
        let o = app.options_mut();
        o.set_string_value("least_square_init_primal", "yes", true, false)
            .unwrap();
        o.set_string_value("bound_mult_init_method", "constant", true, false)
            .unwrap();
    })
    .init;
    assert!(b.least_square_init_primal);
    assert_eq!(b.bound_mult_init_method, "constant");
}

/// The Mehrotra cascade sets its own aggressive push defaults with
/// upstream's `SetNumericValueIfUnset` semantics — an *explicit* user
/// value has to win. Before gh#604 there was no way to express one.
#[test]
fn an_explicit_value_beats_the_mehrotra_cascade() {
    let cascade = builder_from(|app| {
        app.options_mut()
            .set_string_value("mehrotra_algorithm", "yes", true, false)
            .unwrap();
    })
    .init;
    assert_eq!(cascade.bound_push, 10.0, "cascade default");
    assert_eq!(cascade.bound_frac, 0.2);
    assert_eq!(cascade.bound_mult_init_val, 10.0);
    assert_eq!(cascade.constr_mult_init_max, 0.0);

    let overridden = builder_from(|app| {
        let o = app.options_mut();
        o.set_string_value("mehrotra_algorithm", "yes", true, false)
            .unwrap();
        o.set_numeric_value("bound_push", 1e-3, true, false)
            .unwrap();
        o.set_numeric_value("bound_mult_init_val", 2.0, true, false)
            .unwrap();
    })
    .init;
    assert_eq!(overridden.bound_push, 1e-3, "explicit user value wins");
    assert_eq!(overridden.bound_mult_init_val, 2.0);
    // Untouched cascade values stay put.
    assert_eq!(overridden.bound_frac, 0.2);
}

#[test]
fn out_of_range_values_are_refused_at_the_set_call() {
    let mut app = IpoptApplication::new();
    let o = app.options_mut();

    // bound_push / slack_bound_push / bound_mult_init_val: lower bound 0, strict.
    for name in ["bound_push", "slack_bound_push", "bound_mult_init_val"] {
        assert!(
            o.set_numeric_value(name, 0.0, true, false).is_err(),
            "`{name}` must refuse 0 (strict lower bound)"
        );
        assert!(
            o.set_numeric_value(name, -1.0, true, false).is_err(),
            "`{name}` must refuse a negative value"
        );
    }

    // bound_frac / slack_bound_frac: (0, 0.5].
    for name in ["bound_frac", "slack_bound_frac"] {
        assert!(
            o.set_numeric_value(name, 0.0, true, false).is_err(),
            "`{name}` must refuse 0"
        );
        assert!(
            o.set_numeric_value(name, 0.51, true, false).is_err(),
            "`{name}` must refuse a value above 0.5"
        );
        assert!(
            o.set_numeric_value(name, 0.5, true, false).is_ok(),
            "`{name}` must accept exactly 0.5 (upper bound is non-strict)"
        );
    }

    // constr_mult_init_max: lower bound 0, NOT strict.
    assert!(
        o.set_numeric_value("constr_mult_init_max", 0.0, true, false)
            .is_ok()
    );
    assert!(
        o.set_numeric_value("constr_mult_init_max", -1.0, true, false)
            .is_err()
    );

    // The string options take their registered values and nothing else.
    assert!(
        o.set_string_value("bound_mult_init_method", "least-square", true, false)
            .is_err(),
        "an unregistered method name must fail at the set call"
    );
    assert!(
        o.set_string_value("least_square_init_primal", "maybe", true, false)
            .is_err()
    );
}

/// `mu-based` parses — an `ipopt.opt` written for Ipopt must still be
/// readable — and is then refused with a message, because pounce
/// implements `constant` only. What it must never do is quietly run
/// `constant` under a name the user chose for something else.
#[test]
fn the_unimplemented_bound_mult_init_method_is_refused_not_served() {
    let mut app = IpoptApplication::new();
    assert_eq!(app.unimplemented_option_value_refusal(), None, "unset");

    app.options_mut()
        .set_string_value("bound_mult_init_method", "constant", true, false)
        .expect("the implemented mode parses");
    assert_eq!(
        app.unimplemented_option_value_refusal(),
        None,
        "the implemented mode is not refused"
    );

    app.options_mut()
        .set_string_value("bound_mult_init_method", "mu-based", true, false)
        .expect("upstream's value must still parse");
    let msg = app
        .unimplemented_option_value_refusal()
        .expect("`mu-based` must be refused");
    assert!(msg.contains("bound_mult_init_method"), "{msg}");
    assert!(msg.contains("mu-based"), "{msg}");
    assert!(msg.contains("604"), "message should name the issue: {msg}");
}

/// `least_square_init_duals` is the one option upstream registers in
/// this class that pounce has no read site for. It is registered (so
/// the options file parses) and refused (so it does not lie).
#[test]
fn least_square_init_duals_parses_and_is_refused_when_requested() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("least_square_init_duals", "no", true, false)
        .expect("parses");
    assert_eq!(
        app.unimplemented_option_refusal(),
        None,
        "`no` is exactly what pounce does — it asks for nothing"
    );

    app.options_mut()
        .set_string_value("least_square_init_duals", "yes", true, false)
        .expect("parses");
    let msg = app
        .unimplemented_option_refusal()
        .expect("`yes` names a feature pounce does not have");
    assert!(msg.contains("least_square_init_duals"), "{msg}");
}

// ---------------------------------------------------------------------
// The bidirectional registry invariant (gh#604 acceptance criterion 3).
// ---------------------------------------------------------------------

/// The literal option tags the solver's own crates pull out of an
/// `OptionsList`, harvested from the sources themselves so a read site
/// added tomorrow is covered without anyone editing a list here.
///
/// Scope is every crate whose reads are served by the registry a bare
/// [`IpoptApplication::new`] builds — which, since gh#604 moved the
/// `qp_*` convex knobs out of `pounce-cli`'s startup path and into the
/// core registry, includes the CLI. Nothing registers options onto that
/// registry from outside it any more, so a name read anywhere here and
/// missing from the registry is a real defect rather than a scoping
/// artifact.
///
/// Only literal tags are visible this way — a call whose tag is a
/// variable (`read_num(key)`) is invisible, which is why the reverse
/// direction below keys off the registry rather than this set alone.
fn option_names_read_by_the_solver_crates() -> BTreeSet<String> {
    const READERS: &[&str] = &[
        "read_num(",
        "read_int(",
        "read_yes(",
        "get_string_value(",
        "get_bool_value(",
        "get_numeric_value(",
        "get_integer_value(",
        "get_enum_value(",
    ];
    const CRATES: &[&str] = &["pounce-algorithm", "pounce-cli", "pounce-presolve"];

    fn collect(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    let crates_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ dir")
        .to_path_buf();
    let mut files = Vec::new();
    for name in CRATES {
        // Only `src` — this test file itself quotes the reader patterns.
        let src_dir = crates_dir.join(name).join("src");
        assert!(
            src_dir.is_dir(),
            "{} must be readable, or this test silently checks nothing",
            src_dir.display()
        );
        collect(&src_dir, &mut files);
    }

    let mut found = BTreeSet::new();
    for file in files {
        let Ok(src) = std::fs::read_to_string(&file) else {
            continue;
        };
        for reader in READERS {
            for (idx, _) in src.match_indices(reader) {
                let rest = src[idx + reader.len()..].trim_start();
                // Only a literal first argument is a tag we can resolve.
                let Some(rest) = rest.strip_prefix('"') else {
                    continue;
                };
                let Some(end) = rest.find('"') else { continue };
                let tag = &rest[..end];
                if tag.is_empty() {
                    continue;
                }
                found.insert(tag.to_string());
            }
        }
    }
    assert!(
        found.len() > 150,
        "the source scan found only {} tags — the reader patterns have \
         drifted and this test is no longer checking anything",
        found.len()
    );
    found
}

/// Forward direction: every option a solver component reads is
/// registered. gh#604's bug was exactly a violation of this — the read
/// site existed and the registry entry did not, so the option was
/// unreachable through every documented path.
#[test]
fn every_option_read_by_the_solver_crates_is_registered() {
    let app = IpoptApplication::new();
    let reg = app.registered_options();
    let mut unregistered: Vec<String> = option_names_read_by_the_solver_crates()
        .into_iter()
        .filter(|tag| reg.get_option(tag).is_none())
        .collect();
    unregistered.sort();
    assert!(
        unregistered.is_empty(),
        "these options are read but never registered, so setting them \
         raises `Unknown option` and the code reading them can never fire: \
         {unregistered:?}"
    );
}

/// Reverse direction: every registered Initialization option is either
/// consumed by a read site or explicitly refused. A registered knob
/// that nothing reads validates, accepts a value, and lies.
#[test]
fn every_registered_initialization_option_is_consumed_or_refused() {
    use pounce_algorithm::unimplemented_options::{UNIMPLEMENTED_FEATURES, UNIMPLEMENTED_VALUES};

    let app = IpoptApplication::new();
    let read = option_names_read_by_the_solver_crates();
    let refused: BTreeSet<&str> = UNIMPLEMENTED_FEATURES
        .iter()
        .flat_map(|g| g.options.iter().copied())
        .chain(UNIMPLEMENTED_VALUES.iter().map(|v| v.option))
        .collect();

    let mut dangling = Vec::new();
    for opt in app.registered_options().registered_options_in_order() {
        if opt.category != "Initialization" {
            continue;
        }
        if read.contains(&opt.name) || refused.contains(opt.name.as_str()) {
            continue;
        }
        dangling.push(opt.name.clone());
    }
    assert!(
        dangling.is_empty(),
        "these Initialization options are registered but neither read nor \
         refused — setting one does nothing, silently: {dangling:?}"
    );

    // The category is non-empty, or the loop above proves nothing.
    let n = app
        .registered_options()
        .registered_options_in_order()
        .iter()
        .filter(|o| o.category == "Initialization")
        .count();
    assert_eq!(
        n,
        COLD_START_OPTIONS.len() + 1,
        "expected the eight cold-start options plus least_square_init_duals"
    );
}
