//! Wiring tests for algorithmic tuning constants that were registered
//! but never read (#191), so a user override was silently dropped and the
//! solver ran with the hard-coded struct default.
//!
//! Like `mu_options_wiring.rs`, these are pure wiring tests: they assert a
//! `OptionsList` set turns into the corresponding field on
//! `AlgorithmBuilder`. The numerics themselves are exercised by the
//! upstream-mirroring unit tests in `crate::ipopt_alg` /
//! `crate::ipopt_cq`.

use pounce_algorithm::alg_builder::AlgorithmBuilder;
use pounce_algorithm::application::IpoptApplication;

fn builder_from(setup: impl FnOnce(&mut IpoptApplication)) -> AlgorithmBuilder {
    let mut app = IpoptApplication::new();
    setup(&mut app);
    app.algorithm_builder_from_options()
}

#[test]
fn kappa_sigma_default_matches_registered() {
    // Untouched option ⇒ builder carries the upstream default 1e10.
    let b = builder_from(|_| {});
    assert!((b.kappa_sigma - 1e10).abs() <= 1e10 * 1e-12);
}

#[test]
fn kappa_sigma_override_flows_through() {
    let b = builder_from(|app| {
        app.options_mut()
            .set_numeric_value("kappa_sigma", 0.5, true, false)
            .unwrap();
    });
    // The documented `< 1` "disable the correction" value must survive.
    assert!((b.kappa_sigma - 0.5).abs() < 1e-12);
}

#[test]
fn kappa_d_default_matches_registered() {
    let b = builder_from(|_| {});
    assert!((b.kappa_d - 1e-5).abs() < 1e-17);
}

#[test]
fn kappa_d_override_flows_through() {
    let b = builder_from(|app| {
        app.options_mut()
            .set_numeric_value("kappa_d", 0.0, true, false)
            .unwrap();
    });
    assert_eq!(b.kappa_d, 0.0);
}

#[test]
fn tiny_step_and_divergence_defaults_match_registered() {
    let b = builder_from(|_| {});
    assert_eq!(b.tiny_step_tol, 10.0 * f64::EPSILON);
    assert_eq!(b.tiny_step_y_tol, 1e-2);
    assert_eq!(b.diverging_iterates_tol, 1e20);
}

#[test]
fn tiny_step_and_divergence_overrides_flow_through() {
    let b = builder_from(|app| {
        app.options_mut()
            .set_numeric_value("tiny_step_tol", 1e-12, true, false)
            .unwrap();
        app.options_mut()
            .set_numeric_value("tiny_step_y_tol", 1e-3, true, false)
            .unwrap();
        app.options_mut()
            .set_numeric_value("diverging_iterates_tol", 1e8, true, false)
            .unwrap();
    });
    assert_eq!(b.tiny_step_tol, 1e-12);
    assert_eq!(b.tiny_step_y_tol, 1e-3);
    assert_eq!(b.diverging_iterates_tol, 1e8);
}

/// pounce#246: the dual-divergence guard streak defaults to the registered
/// value (15, i.e. enabled) and an explicit override — including `0` to
/// disable — flows through to the builder.
#[test]
fn dual_diverging_streak_default_and_override_flow_through() {
    let b = builder_from(|_| {});
    assert_eq!(
        b.dual_diverging_streak, 0,
        "default must match registered: the guard is OFF by default (pounce#250 follow-up)"
    );

    let off = builder_from(|app| {
        app.options_mut()
            .set_integer_value("dual_diverging_streak", 0, true, false)
            .unwrap();
    });
    assert_eq!(off.dual_diverging_streak, 0, "0 disables the guard");

    let custom = builder_from(|app| {
        app.options_mut()
            .set_integer_value("dual_diverging_streak", 40, true, false)
            .unwrap();
    });
    assert_eq!(custom.dual_diverging_streak, 40);
}

#[test]
fn filter_constants_default_match_registered() {
    let b = builder_from(|_| {}).line_search;
    assert_eq!(b.eta_phi, 1e-8);
    assert_eq!(b.theta_min_fact, 1e-4);
    assert_eq!(b.theta_max_fact, 1e4);
    assert_eq!(b.gamma_phi, 1e-8);
    assert_eq!(b.gamma_theta, 1e-5);
    assert_eq!(b.s_phi, 2.3);
    assert_eq!(b.s_theta, 1.1);
    assert_eq!(b.alpha_min_frac, 0.05);
    assert_eq!(b.obj_max_inc, 5.0);
}

#[test]
fn filter_constants_override_flows_through() {
    let b = builder_from(|app| {
        let o = app.options_mut();
        for (k, v) in [
            ("eta_phi", 2e-8),
            ("theta_min_fact", 2e-4),
            ("theta_max_fact", 2e4),
            ("gamma_phi", 2e-8),
            ("gamma_theta", 2e-5),
            ("s_phi", 2.5),
            ("s_theta", 1.2),
            ("alpha_min_frac", 0.1),
            ("obj_max_inc", 6.0),
        ] {
            o.set_numeric_value(k, v, true, false).unwrap();
        }
    })
    .line_search;
    assert_eq!(b.eta_phi, 2e-8);
    assert_eq!(b.theta_min_fact, 2e-4);
    assert_eq!(b.theta_max_fact, 2e4);
    assert_eq!(b.gamma_phi, 2e-8);
    assert_eq!(b.gamma_theta, 2e-5);
    assert_eq!(b.s_phi, 2.5);
    assert_eq!(b.s_theta, 1.2);
    assert_eq!(b.alpha_min_frac, 0.1);
    assert_eq!(b.obj_max_inc, 6.0);
}

#[test]
fn soc_constants_default_match_registered() {
    let b = builder_from(|_| {}).line_search;
    assert_eq!(b.max_soc, 4);
    assert_eq!(b.kappa_soc, 0.99);
    assert_eq!(b.soc_method, 0);
}

#[test]
fn soc_constants_override_flows_through() {
    let b = builder_from(|app| {
        app.options_mut()
            .set_integer_value("max_soc", 0, true, false)
            .unwrap();
        app.options_mut()
            .set_numeric_value("kappa_soc", 0.5, true, false)
            .unwrap();
        app.options_mut()
            .set_integer_value("soc_method", 1, true, false)
            .unwrap();
    })
    .line_search;
    assert_eq!(b.max_soc, 0);
    assert_eq!(b.kappa_soc, 0.5);
    assert_eq!(b.soc_method, 1);
}

/// The SOC constants must survive `build()` onto the concrete
/// `BacktrackingLineSearch` in the assembled bundle — this covers the
/// builder → line-search assignment (`build_inner`), not just the
/// option → builder step above.
#[test]
fn soc_constants_reach_assembled_line_search() {
    let mut builder = AlgorithmBuilder::new();
    builder.line_search.max_soc = 0;
    builder.line_search.kappa_soc = 0.5;
    builder.line_search.soc_method = 1;
    let bundle = builder.build();
    assert_eq!(bundle.line_search.max_soc, 0);
    assert_eq!(bundle.line_search.kappa_soc, 0.5);
    assert_eq!(bundle.line_search.soc_method, 1);
}

#[test]
fn filter_reset_constants_flow_through() {
    let d = builder_from(|_| {}).line_search;
    assert_eq!(d.max_filter_resets, 5);
    assert_eq!(d.filter_reset_trigger, 5);
    let b = builder_from(|app| {
        app.options_mut()
            .set_integer_value("max_filter_resets", 0, true, false)
            .unwrap();
        app.options_mut()
            .set_integer_value("filter_reset_trigger", 9, true, false)
            .unwrap();
    })
    .line_search;
    assert_eq!(b.max_filter_resets, 0);
    assert_eq!(b.filter_reset_trigger, 9);
}

#[test]
fn refinement_constants_default_match_registered() {
    let r = builder_from(|_| {}).refinement;
    assert_eq!(r.min_refinement_steps, 1);
    assert_eq!(r.max_refinement_steps, 10);
    assert_eq!(r.residual_ratio_max, 1e-10);
    assert_eq!(r.residual_ratio_singular, 1e-5);
    assert_eq!(r.residual_improvement_factor, 0.999_999_999);
}

#[test]
fn refinement_constants_override_flows_through() {
    let r = builder_from(|app| {
        app.options_mut()
            .set_integer_value("min_refinement_steps", 2, true, false)
            .unwrap();
        app.options_mut()
            .set_integer_value("max_refinement_steps", 20, true, false)
            .unwrap();
        app.options_mut()
            .set_numeric_value("residual_ratio_max", 1e-9, true, false)
            .unwrap();
        app.options_mut()
            .set_numeric_value("residual_ratio_singular", 1e-4, true, false)
            .unwrap();
        app.options_mut()
            .set_numeric_value("residual_improvement_factor", 0.5, true, false)
            .unwrap();
    })
    .refinement;
    assert_eq!(r.min_refinement_steps, 2);
    assert_eq!(r.max_refinement_steps, 20);
    assert_eq!(r.residual_ratio_max, 1e-9);
    assert_eq!(r.residual_ratio_singular, 1e-4);
    assert_eq!(r.residual_improvement_factor, 0.5);
}

#[test]
fn perturbation_constants_default_match_registered() {
    let p = builder_from(|_| {}).perturbation;
    assert_eq!(p.max_hessian_perturbation, 1e20);
    assert_eq!(p.min_hessian_perturbation, 1e-20);
    assert_eq!(p.perturb_inc_fact_first, 100.0);
    assert_eq!(p.perturb_inc_fact, 8.0);
    assert_eq!(p.perturb_dec_fact, 1.0 / 3.0);
    assert_eq!(p.first_hessian_perturbation, 1e-4);
    assert_eq!(p.jacobian_regularization_value, 1e-8);
    assert_eq!(p.jacobian_regularization_exponent, 0.25);
    assert!(!p.perturb_always_cd);
}

#[test]
fn perturbation_constants_override_flows_through() {
    let p = builder_from(|app| {
        let o = app.options_mut();
        for (k, v) in [
            ("max_hessian_perturbation", 1e18),
            ("min_hessian_perturbation", 1e-18),
            ("perturb_inc_fact_first", 50.0),
            ("perturb_inc_fact", 4.0),
            ("perturb_dec_fact", 0.5),
            ("first_hessian_perturbation", 1e-3),
            ("jacobian_regularization_value", 1e-7),
            ("jacobian_regularization_exponent", 0.5),
        ] {
            o.set_numeric_value(k, v, true, false).unwrap();
        }
        o.set_string_value("perturb_always_cd", "yes", true, false)
            .unwrap();
    })
    .perturbation;
    assert_eq!(p.max_hessian_perturbation, 1e18);
    assert_eq!(p.min_hessian_perturbation, 1e-18);
    assert_eq!(p.perturb_inc_fact_first, 50.0);
    assert_eq!(p.perturb_inc_fact, 4.0);
    assert_eq!(p.perturb_dec_fact, 0.5);
    assert_eq!(p.first_hessian_perturbation, 1e-3);
    assert_eq!(p.jacobian_regularization_value, 1e-7);
    assert_eq!(p.jacobian_regularization_exponent, 0.5);
    assert!(p.perturb_always_cd);
}

#[test]
fn resto_constants_default_match_registered() {
    let r = builder_from(|_| {}).resto;
    assert_eq!(r.bound_mult_reset_threshold, 1e3);
    assert_eq!(r.constr_mult_reset_threshold, 0.0);
    assert_eq!(r.resto_penalty_parameter, 1e3);
    assert_eq!(r.resto_proximity_weight, 1.0);
}

#[test]
fn resto_constants_override_flows_through() {
    let r = builder_from(|app| {
        let o = app.options_mut();
        for (k, v) in [
            ("bound_mult_reset_threshold", 5e2),
            ("constr_mult_reset_threshold", 3.0),
            ("resto_penalty_parameter", 2e3),
            ("resto_proximity_weight", 2.0),
        ] {
            o.set_numeric_value(k, v, true, false).unwrap();
        }
    })
    .resto;
    assert_eq!(r.bound_mult_reset_threshold, 5e2);
    assert_eq!(r.constr_mult_reset_threshold, 3.0);
    assert_eq!(r.resto_penalty_parameter, 2e3);
    assert_eq!(r.resto_proximity_weight, 2.0);
}
