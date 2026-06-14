//! Wiring test for the μ-strategy auto-fallback switch (pounce#138).
//!
//! `optimize_tnlp` consults `mu_strategy_fallback` to decide
//! whether to retry a `Solved_To_Acceptable_Level` solve with the opposite
//! `mu_strategy`. This test pins the option's registration, default, and the
//! `"yes"`/`"no"` string round-trip the GAMS link depends on (the link
//! forwards unknown keys via `AddIpoptStrOption`, so the bool must accept the
//! string form). It does not run a solve — the end-to-end promotion is
//! exercised through the GAMS princetonlib instances in the issue.

use pounce_algorithm::application::IpoptApplication;

#[test]
fn fallback_option_defaults_off() {
    let mut app = IpoptApplication::new();
    let (value, _found) = app
        .options_mut()
        .get_bool_value("mu_strategy_fallback", "")
        .expect("option must be registered");
    assert!(!value, "μ-strategy fallback must default to off");
}

#[test]
fn fallback_option_accepts_string_yes() {
    // The GAMS link sets bool options via their string form ("yes"/"no");
    // mirror that exact path rather than set_bool_value.
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("mu_strategy_fallback", "yes", true, false)
        .expect("string 'yes' must round-trip into the bool option");
    let (value, found) = app
        .options_mut()
        .get_bool_value("mu_strategy_fallback", "")
        .unwrap();
    assert!(found && value, "after set 'yes', the switch reads true");

    app.options_mut()
        .set_string_value("mu_strategy_fallback", "no", true, false)
        .unwrap();
    let (value, _) = app
        .options_mut()
        .get_bool_value("mu_strategy_fallback", "")
        .unwrap();
    assert!(!value, "after set 'no', the switch reads false");
}
