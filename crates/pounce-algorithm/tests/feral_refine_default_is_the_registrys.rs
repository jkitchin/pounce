//! `feral_refine`'s registered default must be the one the solver runs.
//!
//! WHAT WENT WRONG (gh#909). The option was registered with default
//! `false` and `feral_config_from_options` never consulted the registry.
//! It read the option as `Ok((v, true))` — the tri-state shape used for
//! every other `feral_*` option, which fires only when the user set the
//! value explicitly — so an unset `feral_refine` left `cfg.refine` at
//! `FeralConfig::default()`'s `true`. The clear-to-`false` beside it is
//! scoped to `hessian_approximation=limited-memory`, so on the exact
//! Hessian path — the default path — nothing turned it off.
//!
//! `pounce --print-options` therefore reported `no` while the solver ran
//! `yes`, for every release through 0.11.0. This is the gh#677 shape
//! (registered with one default, read with another) one option family
//! over, and it is the reason that failure mode gets a test of its own
//! here rather than a line in a larger one.
//!
//! WHY THERE WAS NO SYMPTOM, and why this file is not just a unit test of
//! the reader. The mismatch is invisible from inside: every measurement,
//! comment and test in the tree that described "the default" on the exact
//! path was describing `yes` while saying `no`, consistently, so nothing
//! disagreed with anything else. `issue_540_eigena2_superlinear_tail.rs`
//! is the case in point — it solved once at the default and once with an
//! explicit `feral_refine=yes` in order to contrast them, which on that
//! path is the same configuration twice. It passed.
//!
//! WHAT IS PINNED. Not that `refine` is any particular value — a future
//! measurement may well move it, and `feral_config_from_options` carries
//! the two-sided evidence for the current choice. What is pinned is that
//! the *registry* is what decides, so the two cannot drift apart again.
//!
//! THAT DISTINCTION IS THE WHOLE TEST, and getting it wrong is easy.
//! The first draft of this file asserted `refine_under("") ==
//! registered_default()` against the production registry and called it
//! done. It is vacuous: the registered default is now `true` and
//! `FeralConfig::default().refine` is also `true`, so "read from the
//! registry" and "inherited by accident, exactly as in the bug" produce
//! the same answer and the assertion cannot tell them apart. Mutating
//! the reader back to `Ok((v, true))` — the gh#909 defect itself — left
//! all three tests green.
//!
//! So `the_registry_is_what_decides` builds its own registry instead and
//! registers `feral_refine` at each default in turn. If the reader
//! consults the registry the effective value follows it; if it inherits
//! `FeralConfig`'s, it does not move. That one is mutation-checked
//! against the historical reader and goes red. The production-registry
//! assertions below are kept as the statement of the current choice, not
//! as the guard.
//!
//! The registry has to be attached for any of this to mean anything: an
//! `OptionsList` without one answers an unset option from the reader's
//! own fallback, which is the branch no user reaches.
//!
//! MUTATION TABLE. Each applied to the source, run, reverted:
//!
//! | mutation | result |
//! |---|---|
//! | registered default `true` -> `false` (registry and backend disagree again) | `the_limited_memory_carve_out_is_the_only_departure_from_it` red, alone |
//! | reader narrowed back to `Ok((v, true))` -- the gh#909 defect itself | `the_registry_is_what_decides` red, alone |
//!
//! Neither mutation reddens `the_exact_path_runs_the_registered_default`,
//! which is why it is documented above as the statement of the current
//! choice rather than as the guard. Left in deliberately: it is what a
//! reader diffs against when the measurement moves.
#![allow(clippy::unwrap_used)]

use pounce_algorithm::IpoptApplication;
use pounce_algorithm::application::feral_config_from_options;
use pounce_common::{OptionsList, RegisteredOptions};

fn app_with(assignments: &str) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.initialize().expect("registry initializes");
    if !assignments.is_empty() {
        app.options_mut()
            .read_from_str(assignments, true)
            .unwrap_or_else(|e| panic!("{assignments}: {e:?}"));
    }
    app
}

/// The registered default, read the way `pounce --print-options` reports
/// it: an unset option, taking the value and ignoring the found flag.
fn registered_default() -> bool {
    let app = app_with("");
    let (v, found) = app.options().get_bool_value("feral_refine", "").unwrap();
    assert!(
        !found,
        "feral_refine reads as explicitly set on a fresh OptionsList, so \
         this is not the registered default and the test below proves \
         nothing",
    );
    v
}

fn refine_under(assignments: &str) -> bool {
    feral_config_from_options(app_with(assignments).options()).refine
}

#[test]
fn the_exact_path_runs_the_registered_default() {
    // The exact-Hessian path is the default path and takes no carve-out,
    // so it is where the registry must be reproduced exactly. This is
    // the assertion gh#909 was: it failed with `false` registered and
    // `true` running.
    assert_eq!(
        refine_under(""),
        registered_default(),
        "feral_refine runs at a value pounce --print-options does not \
         report; see gh#909",
    );
}

#[test]
fn the_limited_memory_carve_out_is_the_only_departure_from_it() {
    // Scoped, and deliberately so: `feral_config_from_options` carries
    // the measurement. Asserted rather than described, because the
    // wording "the NLP solver turns it off" appeared in three places
    // without this qualifier and was false in all three.
    assert!(
        !refine_under("hessian_approximation limited-memory\n"),
        "the limited-memory carve-out no longer fires",
    );
    assert!(
        refine_under("hessian_approximation exact\n"),
        "an explicit exact Hessian must behave as the unset default does",
    );
}

#[test]
fn an_explicit_setting_beats_the_default_and_the_carve_out() {
    assert!(!refine_under("feral_refine no\n"));
    assert!(refine_under("feral_refine yes\n"));
    // On the limited-memory path an explicit `yes` has to survive the
    // clear-to-false, or the option cannot restore the pre-0.11
    // behaviour there at all.
    assert!(
        refine_under("hessian_approximation limited-memory\nferal_refine yes\n"),
        "an explicit feral_refine=yes must beat the limited-memory carve-out",
    );
    assert!(!refine_under(
        "hessian_approximation limited-memory\nferal_refine no\n"
    ));
}

/// An `OptionsList` whose registry registers `feral_refine` at
/// `default_yes` and nothing else.
///
/// Every other option `feral_config_from_options` reads is absent, so
/// those reads are `Err` and their `if let Ok(..)` arms are skipped,
/// leaving `FeralConfig`'s values — which is what we want: the one thing
/// varying between the two calls below is the registered default of the
/// option under test. `hessian_approximation` is absent too, so this is
/// the exact-Hessian path and no carve-out fires.
fn only_feral_refine(default_yes: bool) -> OptionsList {
    let reg = RegisteredOptions::new();
    reg.add_bool_option("feral_refine", "test registry", default_yes, "")
        .expect("register feral_refine");
    OptionsList::with_registered(reg)
}

#[test]
fn the_registry_is_what_decides() {
    // The guard. `FeralConfig::default().refine` is a fixed `true`, so a
    // registry that says `no` is the only thing that can distinguish a
    // reader consulting the registry from one inheriting the library's
    // value. Under the historical `Ok((v, true))` reader both calls
    // answer `true` and this fails.
    assert!(
        feral_config_from_options(&only_feral_refine(true)).refine,
        "a registry default of yes must produce refine = true",
    );
    assert!(
        !feral_config_from_options(&only_feral_refine(false)).refine,
        "a registry default of no did not reach the backend, so \
         feral_config_from_options is not consulting the registry and \
         pounce --print-options can disagree with the solver again (gh#909)",
    );
}
