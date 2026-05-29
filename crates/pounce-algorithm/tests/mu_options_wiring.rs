//! Verify that the Tier-B adaptive-μ and quality-function option keys
//! parsed by `IpoptApplication::algorithm_builder_from_options` propagate
//! all the way to the `MuOptions` struct on `AlgorithmBuilder`.
//!
//! These are pure wiring tests: they don't run a solve, they just confirm
//! a CLI / `OptionsList` set turns into the corresponding field on the
//! builder. The actual numerics are exercised by the upstream-mirroring
//! unit tests under `crate::mu::oracle::quality_function`.

use pounce_algorithm::alg_builder::MuOptions;
use pounce_algorithm::application::IpoptApplication;
use pounce_algorithm::mu::adaptive::AdaptiveMuKktNorm;
use pounce_algorithm::mu::oracle::quality_function::{BalancingTermType, CentralityType, NormType};

fn mu_options_from(
    setup: impl FnOnce(&mut pounce_algorithm::application::IpoptApplication),
) -> MuOptions {
    let mut app = IpoptApplication::new();
    setup(&mut app);
    app.algorithm_builder_from_options().mu
}

#[test]
fn quality_function_norm_type_flows_through() {
    let mu = mu_options_from(|app| {
        app.options_mut()
            .set_string_value("quality_function_norm_type", "max-norm", true, false)
            .unwrap();
    });
    assert_eq!(mu.quality_function_norm_type, NormType::MaxNorm);
}

#[test]
fn quality_function_centrality_flows_through() {
    let mu = mu_options_from(|app| {
        app.options_mut()
            .set_string_value("quality_function_centrality", "reciprocal", true, false)
            .unwrap();
    });
    assert_eq!(
        mu.quality_function_centrality,
        CentralityType::ReciprocalCenter
    );
}

#[test]
fn quality_function_balancing_term_flows_through() {
    let mu = mu_options_from(|app| {
        app.options_mut()
            .set_string_value("quality_function_balancing_term", "cubic", true, false)
            .unwrap();
    });
    assert_eq!(
        mu.quality_function_balancing_term,
        BalancingTermType::CubicTerm
    );
}

#[test]
fn quality_function_section_knobs_flow_through() {
    let mu = mu_options_from(|app| {
        app.options_mut()
            .set_integer_value("quality_function_max_section_steps", 16, true, false)
            .unwrap();
        app.options_mut()
            .set_numeric_value("quality_function_section_sigma_tol", 5e-3, true, false)
            .unwrap();
        app.options_mut()
            .set_numeric_value("quality_function_section_qf_tol", 1e-4, true, false)
            .unwrap();
    });
    assert_eq!(mu.quality_function_max_section_steps, 16);
    assert!((mu.quality_function_section_sigma_tol - 5e-3).abs() < 1e-12);
    assert!((mu.quality_function_section_qf_tol - 1e-4).abs() < 1e-12);
}

#[test]
fn adaptive_mu_extra_knobs_flow_through() {
    let mu = mu_options_from(|app| {
        app.options_mut()
            .set_numeric_value("adaptive_mu_safeguard_factor", 0.25, true, false)
            .unwrap();
        app.options_mut()
            .set_numeric_value("adaptive_mu_monotone_init_factor", 0.5, true, false)
            .unwrap();
        app.options_mut()
            .set_string_value("adaptive_mu_restore_previous_iterate", "yes", true, false)
            .unwrap();
        app.options_mut()
            .set_integer_value("adaptive_mu_kkterror_red_iters", 7, true, false)
            .unwrap();
        app.options_mut()
            .set_numeric_value("adaptive_mu_kkterror_red_fact", 0.99, true, false)
            .unwrap();
        app.options_mut()
            .set_string_value("adaptive_mu_kkt_norm_type", "max-norm", true, false)
            .unwrap();
    });
    assert!((mu.adaptive_mu_safeguard_factor - 0.25).abs() < 1e-12);
    assert!((mu.adaptive_mu_monotone_init_factor - 0.5).abs() < 1e-12);
    assert!(mu.adaptive_mu_restore_previous_iterate);
    assert_eq!(mu.adaptive_mu_kkterror_red_iters, 7);
    assert!((mu.adaptive_mu_kkterror_red_fact - 0.99).abs() < 1e-12);
    assert_eq!(mu.adaptive_mu_kkt_norm_type, AdaptiveMuKktNorm::MaxNorm);
}

#[test]
fn defaults_match_upstream() {
    // Unconfigured app — defaults should match upstream
    // `IpQualityFunctionMuOracle::RegisterOptions` and
    // `IpAdaptiveMuUpdate::RegisterOptions`.
    let mu = mu_options_from(|_| {});
    assert_eq!(mu.quality_function_norm_type, NormType::TwoNormSquared);
    assert_eq!(mu.quality_function_centrality, CentralityType::None);
    assert_eq!(mu.quality_function_balancing_term, BalancingTermType::None);
    assert_eq!(mu.quality_function_max_section_steps, 8);
    assert!((mu.quality_function_section_sigma_tol - 1e-2).abs() < 1e-12);
    assert_eq!(mu.quality_function_section_qf_tol, 0.0);
    assert_eq!(mu.adaptive_mu_safeguard_factor, 0.0);
    assert!((mu.adaptive_mu_monotone_init_factor - 0.8).abs() < 1e-12);
    assert!(!mu.adaptive_mu_restore_previous_iterate);
    assert_eq!(mu.adaptive_mu_kkterror_red_iters, 4);
    assert!((mu.adaptive_mu_kkterror_red_fact - 0.9999).abs() < 1e-12);
    assert_eq!(
        mu.adaptive_mu_kkt_norm_type,
        AdaptiveMuKktNorm::TwoNormSquared
    );
    // Pounce-specific (pounce#58) probing-oracle iterate-quality guard.
    assert!((mu.probing_iterate_quality_factor - 1e4).abs() < 1e-12);
}

#[test]
fn probing_iterate_quality_factor_flows_through() {
    let mu = mu_options_from(|app| {
        app.options_mut()
            .set_numeric_value("probing_iterate_quality_factor", 5e3, true, false)
            .unwrap();
    });
    assert!((mu.probing_iterate_quality_factor - 5e3).abs() < 1e-12);
}

#[test]
fn probing_iterate_quality_factor_disable_value_flows_through() {
    // 0 means "guard disabled" — the value should still round-trip
    // through the option registry without snap-back to the default.
    let mu = mu_options_from(|app| {
        app.options_mut()
            .set_numeric_value("probing_iterate_quality_factor", 0.0, true, false)
            .unwrap();
    });
    assert_eq!(mu.probing_iterate_quality_factor, 0.0);
}
