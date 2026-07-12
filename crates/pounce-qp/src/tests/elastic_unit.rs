//! Focused unit tests for `crate::elastic` — §8.7 elastic-module
//! contract. The §8.5 / ladder #5 tests in `analytical.rs`
//! exercise the full solver path; these pin the reformulation's
//! shape and initial-seed correctness without involving the
//! factorization.

use crate::elastic::ElasticReformulation;
use crate::kkt::a_times_x;
use crate::problem::{HessianInertia, QpProblem};
use crate::working_set::{BoundStatus, ConsStatus};
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};

fn tiny_qp_with_one_inequality() -> (
    SymTMatrix,
    GenTMatrix,
    [f64; 2],
    [f64; 1],
    [f64; 1],
    [f64; 2],
    [f64; 2],
) {
    // n = 2, m = 1, H = I, A = [1 1], bl = -1, bu = +inf,
    // bounds free.
    let h_space = SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[1.0, 1.0]);

    let a_space = GenTMatrixSpace::new(1, 2, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);

    let g = [1.0, 1.0];
    let bl = [-1.0];
    let bu = [NLP_UPPER_BOUND_INF];
    let xl = [NLP_LOWER_BOUND_INF; 2];
    let xu = [NLP_UPPER_BOUND_INF; 2];
    (h, a, g, bl, bu, xl, xu)
}

#[test]
fn reformulation_dimensions_grow_by_2m() {
    let (h, a, g, bl, bu, xl, xu) = tiny_qp_with_one_inequality();
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
        hessian_inertia: HessianInertia::Psd,
    };
    let reform = ElasticReformulation::build(&qp, 1e6);
    assert_eq!(reform.n_orig, 2);
    assert_eq!(reform.m_orig, 1);
    assert_eq!(reform.n_aug, 4); // 2 vars + 2 slacks per row × 1 row
    assert_eq!(reform.m_aug, 1);
    assert_eq!(reform.g_aug.len(), 4);
    assert_eq!(reform.xl_aug.len(), 4);
    assert_eq!(reform.xu_aug.len(), 4);
}

#[test]
fn reformulation_slack_bounds_are_nonnegative_orthant() {
    let (h, a, g, bl, bu, xl, xu) = tiny_qp_with_one_inequality();
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
        hessian_inertia: HessianInertia::Psd,
    };
    let reform = ElasticReformulation::build(&qp, 1e6);
    // Slacks at positions 2, 3 of the augmented variable list.
    assert_eq!(reform.xl_aug[2], 0.0);
    assert_eq!(reform.xl_aug[3], 0.0);
    assert_eq!(reform.xu_aug[2], NLP_UPPER_BOUND_INF);
    assert_eq!(reform.xu_aug[3], NLP_UPPER_BOUND_INF);
}

#[test]
fn reformulation_gradient_carries_gamma_on_slacks() {
    let (h, a, g, bl, bu, xl, xu) = tiny_qp_with_one_inequality();
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
        hessian_inertia: HessianInertia::Psd,
    };
    let gamma = 1e4;
    let reform = ElasticReformulation::build(&qp, gamma);
    assert_eq!(reform.g_aug[..2], g);
    assert_eq!(reform.g_aug[2], gamma);
    assert_eq!(reform.g_aug[3], gamma);
}

#[test]
fn initial_seed_absorbs_lower_bound_violation_into_v_l() {
    // Original constraint x₁ + x₂ ≥ -1, evaluated at x_orig = (0,0):
    // a·x = 0, no violation. v_l = v_u = 0.
    let (h, a, g, bl, bu, xl, xu) = tiny_qp_with_one_inequality();
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
        hessian_inertia: HessianInertia::Psd,
    };
    let reform = ElasticReformulation::build(&qp, 1e6);
    let (x_aug, working) = reform.initial_seed(&qp, &[0.0, 0.0], 1e-9);
    assert_eq!(x_aug.len(), 4);
    assert_eq!(x_aug[2], 0.0); // v_l = 0 (no violation, a·x = 0 > bl = -1)
    assert_eq!(x_aug[3], 0.0); // v_u = 0
    // Both slacks at their lower bound (= 0).
    assert_eq!(working.bounds[2], BoundStatus::AtLower);
    assert_eq!(working.bounds[3], BoundStatus::AtLower);
}

#[test]
fn initial_seed_pushes_violation_into_slack_and_keeps_qp_feasible() {
    // Strong constraint: x₁ + x₂ ≥ 5 with x_orig = (0, 0) →
    // a·x = 0 < bl = 5 ⇒ v_l = 5.
    let h_space = SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&[1.0, 1.0]);
    let a_space = GenTMatrixSpace::new(1, 2, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);
    let g = [0.0, 0.0];
    let bl = [5.0];
    let bu = [NLP_UPPER_BOUND_INF];
    let xl = [NLP_LOWER_BOUND_INF; 2];
    let xu = [NLP_UPPER_BOUND_INF; 2];

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
        hessian_inertia: HessianInertia::Psd,
    };
    let reform = ElasticReformulation::build(&qp, 1e6);
    let (x_aug, working) = reform.initial_seed(&qp, &[0.0, 0.0], 1e-9);
    assert_eq!(x_aug[2], 5.0); // v_l absorbs the 5-unit violation
    assert_eq!(x_aug[3], 0.0); // v_u = 0
    // Constraint should now be exactly at bl ⇒ AtLower in W.
    assert_eq!(working.constraints[0], ConsStatus::AtLower);
    // v_l interior; v_u at lower.
    assert_eq!(working.bounds[2], BoundStatus::Inactive);
    assert_eq!(working.bounds[3], BoundStatus::AtLower);

    // Verify augmented constraint: a·x + v_l - v_u = 0 + 5 - 0 = 5 = bl.
    let qp_aug = reform.as_qp();
    let aug_lhs = a_times_x(qp_aug.a, &x_aug, 1);
    assert!((aug_lhs[0] - 5.0).abs() < 1e-12);
}

// L15 regression: `original_inertia()` hard-returned `Psd`, so the
// `Indefinite` arm of `as_qp`'s inertia match was dead — an indefinite
// original problem was silently solved through the augmented problem as
// if PSD, skipping the inertia-control path. `build` now captures
// `qp.hessian_inertia`; `as_qp` propagates `Indefinite` and collapses
// `Psd`/`Unknown` to `Psd` (the augmented Hessian is PSD with explicit
// zero slack diagonals).
#[test]
fn as_qp_propagates_original_hessian_inertia() {
    let (h, a, g, bl, bu, xl, xu) = tiny_qp_with_one_inequality();
    let mk = |inertia| QpProblem {
        n: 2,
        m: 1,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: inertia,
    };

    // Indefinite original ⇒ augmented marked Indefinite (the formerly
    // dead arm). Pre-fix this collapsed to Psd.
    let qp_ind = mk(HessianInertia::Indefinite);
    let reform_ind = ElasticReformulation::build(&qp_ind, 1e6);
    assert_eq!(
        reform_ind.as_qp().hessian_inertia,
        HessianInertia::Indefinite
    );

    // Psd and Unknown both collapse to Psd (zero slack diagonals make
    // the augmented Hessian PSD, never indefinite).
    let qp_psd = mk(HessianInertia::Psd);
    assert_eq!(
        ElasticReformulation::build(&qp_psd, 1e6)
            .as_qp()
            .hessian_inertia,
        HessianInertia::Psd
    );
    let qp_unk = mk(HessianInertia::Unknown);
    assert_eq!(
        ElasticReformulation::build(&qp_unk, 1e6)
            .as_qp()
            .hessian_inertia,
        HessianInertia::Psd
    );
}

#[test]
fn is_feasible_zero_slacks_returns_true() {
    let (h, a, g, bl, bu, xl, xu) = tiny_qp_with_one_inequality();
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
        hessian_inertia: HessianInertia::Psd,
    };
    let reform = ElasticReformulation::build(&qp, 1e6);
    let x_aug = [0.5, 0.5, 0.0, 0.0]; // slacks zero
    assert!(reform.is_feasible(&x_aug, 1e-9));
}

#[test]
fn is_feasible_nonzero_slack_returns_false() {
    let (h, a, g, bl, bu, xl, xu) = tiny_qp_with_one_inequality();
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
        hessian_inertia: HessianInertia::Psd,
    };
    let reform = ElasticReformulation::build(&qp, 1e6);
    let x_aug = [0.5, 0.5, 0.001, 0.0]; // v_l > tol
    assert!(!reform.is_feasible(&x_aug, 1e-9));
}
