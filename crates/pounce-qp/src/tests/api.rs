//! Type-plumbing tests: prove the public surface is wired correctly.
//! These tests do not exercise any QP algorithm; the solver
//! internals land in a follow-up commit.

use crate::error::QpError;
use crate::options::{AntiCyclingChoice, QpAlgorithm, QpOptions};
use crate::problem::{HessianInertia, QpProblem};
use crate::working_set::{BoundStatus, ConsStatus, WorkingSet};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};

fn empty_sym(n: usize) -> SymTMatrix {
    SymTMatrix::new(SymTMatrixSpace::new(n as i32, Vec::new(), Vec::new()))
}

fn empty_gen(m: usize, n: usize) -> GenTMatrix {
    GenTMatrix::new(GenTMatrixSpace::new(
        m as i32,
        n as i32,
        Vec::new(),
        Vec::new(),
    ))
}

#[test]
fn working_set_cold_has_expected_lengths_and_no_active_count() {
    let ws = WorkingSet::cold(5, 3);
    assert_eq!(ws.n(), 5);
    assert_eq!(ws.m(), 3);
    assert_eq!(ws.active_count(), 0);
    assert!(ws.bounds.iter().all(|s| *s == BoundStatus::Inactive));
    assert!(ws.constraints.iter().all(|s| *s == ConsStatus::Inactive));
}

#[test]
fn working_set_active_count_sums_bounds_and_constraints() {
    let mut ws = WorkingSet::cold(4, 4);
    ws.bounds[0] = BoundStatus::AtLower;
    ws.bounds[1] = BoundStatus::Fixed;
    ws.constraints[2] = ConsStatus::Equality;
    ws.constraints[3] = ConsStatus::AtUpper;
    assert_eq!(ws.active_count(), 4);
}

#[test]
fn working_set_status_helpers_classify_correctly() {
    assert!(!BoundStatus::Inactive.is_active());
    assert!(BoundStatus::AtLower.is_active());
    assert!(BoundStatus::AtUpper.is_active());
    assert!(BoundStatus::Fixed.is_active());

    assert!(!ConsStatus::Inactive.is_active());
    assert!(ConsStatus::AtLower.is_active());
    assert!(ConsStatus::AtUpper.is_active());
    assert!(ConsStatus::Equality.is_active());
}

#[test]
fn working_set_validate_dims_rejects_wrong_n() {
    let ws = WorkingSet::cold(3, 2);
    let err = ws.validate_dims(4, 2).unwrap_err();
    assert!(matches!(err, QpError::WarmStartDimensionMismatch(_)));
}

#[test]
fn working_set_validate_dims_rejects_wrong_m() {
    let ws = WorkingSet::cold(3, 2);
    let err = ws.validate_dims(3, 5).unwrap_err();
    assert!(matches!(err, QpError::WarmStartDimensionMismatch(_)));
}

#[test]
fn working_set_validate_dims_accepts_matching() {
    let ws = WorkingSet::cold(3, 2);
    assert!(ws.validate_dims(3, 2).is_ok());
}

#[test]
fn qp_problem_validate_accepts_well_formed() {
    let h = empty_sym(2);
    let a = empty_gen(1, 2);
    let g = [0.0; 2];
    let bl = [-1.0];
    let bu = [1.0];
    let xl = [-10.0; 2];
    let xu = [10.0; 2];
    let qp = QpProblem {
        n: 2,
        m: 1,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::default(),
    };
    qp.validate().unwrap();
}

#[test]
fn qp_problem_validate_rejects_inverted_general_bounds() {
    let h = empty_sym(1);
    let a = empty_gen(1, 1);
    let g = [0.0];
    let bl = [1.0];
    let bu = [-1.0];
    let xl = [-1.0];
    let xu = [1.0];
    let qp = QpProblem {
        n: 1,
        m: 1,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let err = qp.validate().unwrap_err();
    assert!(matches!(err, QpError::InvertedBounds(_)));
}

#[test]
fn qp_problem_validate_rejects_inverted_variable_bounds() {
    let h = empty_sym(2);
    let a = empty_gen(0, 2);
    let g = [0.0; 2];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [1.0, -1.0];
    let xu = [0.0, 1.0];
    let qp = QpProblem {
        n: 2,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Unknown,
    };
    let err = qp.validate().unwrap_err();
    assert!(matches!(err, QpError::InvertedBounds(_)));
}

#[test]
fn qp_problem_validate_rejects_g_dim_mismatch() {
    let h = empty_sym(2);
    let a = empty_gen(0, 2);
    let g = [0.0; 3]; // wrong: should be length 2
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [-1.0; 2];
    let xu = [1.0; 2];
    let qp = QpProblem {
        n: 2,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Unknown,
    };
    let err = qp.validate().unwrap_err();
    assert!(matches!(err, QpError::DimensionMismatch(_)));
}

#[test]
fn qp_options_default_reflects_design_note_pinning() {
    let opts = QpOptions::default();
    assert_eq!(opts.algorithm, QpAlgorithm::ParametricActiveSet);
    assert_eq!(opts.anti_cycling, AntiCyclingChoice::Expand);
    assert_eq!(opts.max_iter, 200);
    assert_eq!(opts.feas_tol, 1e-9);
    assert_eq!(opts.opt_tol, 1e-9);
    assert_eq!(opts.max_schur_updates_before_refactor, 50);
    assert_eq!(opts.elastic_gamma, 1e6);
    assert_eq!(opts.print_level, 0);
}
