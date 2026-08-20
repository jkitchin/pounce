//! Every registered sIPOPT option is read by something or refused
//! outright — nothing in the category may be silently inert
//! (gh#551 / gh#677).
//!
//! `register_sipopt_options` registers upstream sIPOPT's seven keys so
//! an `ipopt.opt` written for sIPOPT parses unchanged. That is a real
//! compatibility benefit and it made all seven no-ops: the reduced
//! Hessian, the eigendecomposition and the bound refinement were
//! reachable only through the CLI's own `--*` flags or the
//! `pounce-sensitivity` builder, and setting the option that names
//! them did nothing at all.
//!
//! The invariant below is the one that keeps it fixed: a registered
//! sIPOPT option is either named by a read site in some crate's
//! `src/`, or listed in
//! [`pounce_algorithm::unimplemented_options`] as a feature pounce does
//! not have. Same shape as
//! `every_registered_warm_start_option_is_consumed_or_refused`
//! (gh#606) and `init_options_wiring.rs` (gh#604), one category over.
//!
//! What the *behaviour* of each wired option is belongs to the crate
//! that acts on it —
//! `pounce-sensitivity/tests/sens_options_wiring.rs` and
//! `pounce-cli/tests/sens_options_end_to_end.rs` set each option and
//! check the answer changes. This test only rules out the silent
//! no-op; it cannot tell a read site from a read-and-discard, which is
//! why those tests exist and are the deliverable.

use std::collections::BTreeSet;

use pounce_algorithm::application::IpoptApplication;
use pounce_algorithm::unimplemented_options::{UNIMPLEMENTED_FEATURES, UNIMPLEMENTED_VALUES};

/// Every `.rs` under `crates/*/src/`, minus the two files that merely
/// *register* the sIPOPT names (`upstream_options.rs` and
/// `sens_app.rs`, which hold the duplicated `register_options` blocks).
/// Counting those would make the check vacuous: they mention all seven
/// keys by construction. Test sources are excluded for the same
/// reason — a test that sets an option proves nothing reads it.
fn read_sites() -> Vec<String> {
    fn walk(dir: &std::path::Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs")
                && p.file_name()
                    .is_some_and(|f| f != "upstream_options.rs" && f != "sens_app.rs")
            {
                if let Ok(s) = std::fs::read_to_string(&p) {
                    out.push(s);
                }
            }
        }
    }
    let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/");
    let mut out = Vec::new();
    for e in std::fs::read_dir(crates).expect("read crates/").flatten() {
        let src = e.path().join("src");
        if src.is_dir() {
            walk(&src, &mut out);
        }
    }
    assert!(!out.is_empty(), "found no crate sources to scan");
    out
}

#[test]
fn every_registered_sipopt_option_is_consumed_or_refused() {
    let app = IpoptApplication::new();
    let refused: BTreeSet<&str> = UNIMPLEMENTED_FEATURES
        .iter()
        .flat_map(|g| g.options.iter().copied())
        .chain(UNIMPLEMENTED_VALUES.iter().map(|v| v.option))
        .collect();
    let sources = read_sites();

    let mut dangling = Vec::new();
    let mut seen = 0usize;
    for opt in app.registered_options().registered_options_in_order() {
        if opt.category != "sIPOPT" {
            continue;
        }
        seen += 1;
        if refused.contains(opt.name.as_str()) {
            continue;
        }
        let quoted = format!("\"{}\"", opt.name);
        if sources.iter().any(|s| s.contains(&quoted)) {
            continue;
        }
        dangling.push(opt.name.clone());
    }

    assert!(
        seen >= 7,
        "the sIPOPT category must be non-empty, found {seen}"
    );
    assert!(
        dangling.is_empty(),
        "these sIPOPT options are registered but neither read nor \
         refused — setting one does nothing, silently: {dangling:?}"
    );
}

/// `n_sens_steps` is the one that is refused rather than read: pounce
/// computes the single `sens_state_1` perturbation tier, and upstream's
/// higher tiers (`sens_state_2`, …) do not exist here. Its default
/// still has to parse — a generated options file spells it out.
#[test]
fn n_sens_steps_is_refused_above_its_single_implemented_tier() {
    let mut app = IpoptApplication::new();
    assert_eq!(app.unimplemented_option_refusal(), None, "unset");

    app.options_mut()
        .set_integer_value("n_sens_steps", 1, true, false)
        .expect("the registered default must still parse");
    assert_eq!(
        app.unimplemented_option_refusal(),
        None,
        "`n_sens_steps=1` is the tier pounce computes and asks for nothing"
    );

    app.options_mut()
        .set_integer_value("n_sens_steps", 3, true, false)
        .expect("upstream's values must still parse");
    let msg = app
        .unimplemented_option_refusal()
        .expect("`n_sens_steps=3` must be refused");
    assert!(msg.contains("n_sens_steps"), "{msg}");
    assert!(msg.contains("perturbation tier"), "{msg}");
    assert!(
        msg.contains("677"),
        "the message must name the issue: {msg}"
    );
}
