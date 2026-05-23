//! Scaffolding-level tests for the SQP module. Phase 5b commit 1
//! only verifies that the types compile, the defaults are sane,
//! and `AlgorithmChoice::ActiveSetSqp` is wired into the builder
//! enum. End-to-end optimize tests land in later commits.

use crate::alg_builder::AlgorithmChoice;
use crate::sqp::iterates::SqpIterates;
use crate::sqp::options::{SqpGlobalization, SqpHessianSource, SqpOptions};
use crate::sqp::qp_assembly::{SqpQpData, Triplet};
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_qp::{HessianInertia, ParametricActiveSetSolver, QpOptions, QpSolver, QpStatus};

#[test]
fn algorithm_choice_default_is_interior_point() {
    assert_eq!(AlgorithmChoice::default(), AlgorithmChoice::InteriorPoint);
}

#[test]
fn sqp_options_default_matches_design_note() {
    let opts = SqpOptions::default();
    assert_eq!(opts.globalization, SqpGlobalization::Filter);
    assert_eq!(opts.hessian, SqpHessianSource::Exact);
    assert_eq!(opts.max_iter, 200);
    assert!((opts.tol - 1e-8).abs() < f64::EPSILON);
    assert!((opts.l1_penalty - 1.0).abs() < f64::EPSILON);
}

#[test]
fn sqp_iterates_cold_has_expected_lengths_and_no_working_set() {
    let it = SqpIterates::cold(5, 3);
    assert_eq!(it.n(), 5);
    assert_eq!(it.m(), 3);
    assert_eq!(it.x, vec![0.0; 5]);
    assert_eq!(it.lambda_g, vec![0.0; 3]);
    assert_eq!(it.lambda_x, vec![0.0; 5]);
    assert!(it.working.is_none());
}

// ─────────────────────────────────────────────────────────────────
// QP-from-linearization end-to-end on a closed-form NLP:
//
//     min f(x) = ½(x₁² + x₂²) − x₁ − 2x₂
//     s.t. c(x) = x₁ + x₂ − 1 = 0,  no bounds.
//
// This is a *convex quadratic* NLP, so its SQP linearization at
// any iterate is also the original problem (∇²L = I, ∇f = x − g,
// J_c = (1, 1), c = x₁+x₂ − 1). One SQP iteration from x_0 =
// (0, 0) should produce a step p that lands at the true optimum.
//
// True optimum: minimize ½(x₁²+x₂²) − x₁ − 2x₂ s.t. x₁+x₂ = 1.
// Lagrangian: x₁ − 1 + λ = 0, x₂ − 2 + λ = 0, x₁+x₂ = 1.
// ⇒ x₁ = 1 − λ, x₂ = 2 − λ, sum: 3 − 2λ = 1 ⇒ λ = 1.
// ⇒ x* = (0, 1), λ_g = 1. From x_0 = (0, 0): p = (0, 1).
// ─────────────────────────────────────────────────────────────────
#[test]
fn qp_assembly_one_sqp_iter_solves_convex_eq_nlp() {
    let n = 2;
    let m = 1;
    let x = vec![0.0; n];

    // ∇f(x) = x − (1, 2)
    let grad_f = vec![x[0] - 1.0, x[1] - 2.0];

    // c(x) = x₁ + x₂ − 1 = −1 at x_0
    let c_vals = vec![x[0] + x[1] - 1.0];
    let bl_c = vec![0.0];
    let bu_c = vec![0.0];

    // No variable bounds.
    let xl_orig = vec![NLP_LOWER_BOUND_INF; n];
    let xu_orig = vec![NLP_UPPER_BOUND_INF; n];

    // J_c = [1, 1] — one row, two columns.
    let jac_c = Triplet {
        n_rows: m,
        n_cols: n,
        irow: vec![1, 1],
        jcol: vec![1, 2],
        vals: vec![1.0, 1.0],
    };

    // ∇²L = diag(1, 1) — Lagrangian Hessian of ½‖x‖² is I.
    let hess_lag = Triplet {
        n_rows: n,
        n_cols: n,
        irow: vec![1, 2],
        jcol: vec![1, 2],
        vals: vec![1.0, 1.0],
    };

    let qp_data = SqpQpData::build(
        &x,
        &grad_f,
        &c_vals,
        &bl_c,
        &bu_c,
        &xl_orig,
        &xu_orig,
        jac_c,
        hess_lag,
        HessianInertia::Psd,
    );

    // Spot-check the constructed QP bounds:
    //   bl_qp = bl_c − c_vals = 0 − (−1) = 1
    //   bu_qp = bu_c − c_vals = 0 − (−1) = 1
    assert!((qp_data.bl[0] - 1.0).abs() < 1e-12);
    assert!((qp_data.bu[0] - 1.0).abs() < 1e-12);

    // Solve the QP for the SQP step.
    let qp = qp_data.as_qp();
    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    assert_eq!(sol.status, QpStatus::Optimal);

    // Step p should land at (0, 1) from x_0 = (0, 0).
    assert!((sol.x[0] - 0.0).abs() < 1e-10, "p[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-10, "p[1] = {}", sol.x[1]);
    // QP multiplier on the equality matches the closed-form
    // Lagrange multiplier (sign per our convention):
    // Hx + Aᵀλ = −g ⇒ at x_0 + p = (0, 1): (0, 1) + (λ, λ) = (1, 2),
    // so λ = 1, hence sol.lambda_g[0] should be 1.0.
    assert!(
        (sol.lambda_g[0] - 1.0).abs() < 1e-10,
        "λ_g[0] = {}",
        sol.lambda_g[0]
    );
}

#[test]
fn qp_assembly_preserves_inf_bounds_in_shift() {
    // Variable bounds: xl = -inf, xu = +inf. After shifting by
    // x, they should remain at the sentinel ±inf values.
    let n = 1;
    let m = 0;
    let x = vec![3.7];
    let grad_f = vec![0.0];
    let c_vals: Vec<f64> = vec![];
    let bl_c: Vec<f64> = vec![];
    let bu_c: Vec<f64> = vec![];
    let xl_orig = vec![NLP_LOWER_BOUND_INF];
    let xu_orig = vec![NLP_UPPER_BOUND_INF];

    let jac_c = Triplet {
        n_rows: 0,
        n_cols: n,
        irow: vec![],
        jcol: vec![],
        vals: vec![],
    };
    let hess_lag = Triplet {
        n_rows: n,
        n_cols: n,
        irow: vec![1],
        jcol: vec![1],
        vals: vec![1.0],
    };
    let data = SqpQpData::build(
        &x,
        &grad_f,
        &c_vals,
        &bl_c,
        &bu_c,
        &xl_orig,
        &xu_orig,
        jac_c,
        hess_lag,
        HessianInertia::Psd,
    );

    // After shifting, ±inf is still ±inf — the sentinels survive.
    assert_eq!(data.xl[0], NLP_LOWER_BOUND_INF);
    assert_eq!(data.xu[0], NLP_UPPER_BOUND_INF);

    // Ignore unused-but-clippy-cared-about warning by touching m.
    let _ = m;
}
