//! The `PounceQp*` entry, cone by cone and refusal by refusal (gh#1147
//! S3-I3). `tests/qp_handle.rs` (S3-I2) has one model per path and the
//! getters' and report's branches; this file has:
//!
//! * **one tiny model per cone kind** — nonneg, SOC, exp, power, PSD — with
//!   an optimum known by hand, solved through the C handle alone;
//! * **every refusal, each its own fixture**: indefinite `P` (refused under
//!   `ipm`, auto and claimed, on the QP and the conic path; solved under
//!   `active-set`); cones not covering `m_ineq` (short, long, PSD mixed with
//!   exp and with pow); `dim ≤ 0` for every kind; `alpha` outside `(0, 1)`;
//!   NULL arrays, one per argument of each builder; unknown option keys; each
//!   getter before any solve; and each option the conic path refuses
//!   (`method=active-set`, `tau`, `tau_max`, `qp_presolve`, `qp_reg`,
//!   `qp_hsde`, `qp_equilibrate`, `qp_crossover`);
//! * **the passing side of every branch those refusals test** (CLAUDE.md, "A
//!   test is only evidence about the branch it reaches"): a zero-row nonneg
//!   cone, an exponent just inside `(0, 1)`, NULL arrays at `nnz = 0`, the
//!   conic path's accepted options, the switches the active-set engine reads;
//! * **every status a model can reach**: `primal_infeasible` and
//!   `dual_infeasible` with their certificate kinds on the QP and the conic
//!   path, `iteration_limit` on both, and `time_limit`.
//!
//! The same models through Python, compared to the bit, are in
//! `tests/qp_python_parity.rs`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/qp_c.rs"]
mod qp_c;

use pounce_cinterface::*;
use qp_c::{FALSE, Qp, TRUE, close, cs};
use std::ptr::{null, null_mut};

const TOL: f64 = 1e-6;

// ---------------------------------------------------------------------------
// One model per cone kind
// ---------------------------------------------------------------------------

/// Nonnegative orthant as a cone: `min −x0 − x1` s.t. `x0 ≤ 1`, `x1 ≤ 2`
/// (`G = I`, `h = (1, 2)`, one `nonneg(2)`). Optimum `(1, 2)`, `−3`; the
/// multipliers are `z = (1, 1)`.
fn nonneg_model() -> Qp {
    let qp = Qp::new(2, 0, 2, 0);
    assert_eq!(qp.objective(&[], &[-1.0, -1.0]), TRUE);
    assert_eq!(
        qp.inequalities(&[(0, 0, 1.0), (1, 1, 1.0)], &[1.0, 2.0]),
        TRUE
    );
    assert_eq!(qp.cone(POUNCE_CONE_NONNEG, 2, 0.0), 0);
    qp
}

#[test]
fn nonneg_cone() {
    let qp = nonneg_model();
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL);
    let s = qp.solution();
    close("x", &s.x, &[1.0, 2.0], TOL);
    close("obj", &[s.obj], &[-3.0], TOL);
    close("z", &s.z, &[1.0, 1.0], TOL);
}

/// A zero-row nonneg block is accepted and changes nothing (the empty-safe
/// side of the `dim` check; `python/tests/test_socp.py:215`).
#[test]
fn zero_row_nonneg_cone_is_accepted() {
    let qp = Qp::new(2, 0, 2, 0);
    assert_eq!(qp.objective(&[], &[-1.0, -1.0]), TRUE);
    assert_eq!(
        qp.inequalities(&[(0, 0, 1.0), (1, 1, 1.0)], &[1.0, 2.0]),
        TRUE
    );
    assert_eq!(qp.cone(POUNCE_CONE_NONNEG, 0, 0.0), 0);
    assert_eq!(qp.cone(POUNCE_CONE_NONNEG, 2, 0.0), 1);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL);
    close("x", &qp.solution().x, &[1.0, 2.0], TOL);
}

/// SOC with a quadratic objective: the Euclidean projection of `(1, 2, 0)`
/// onto `Q³` (`python/tests/test_socp.py:15`), closed form `(3/2, 3/2, 0)`;
/// `obj = ½‖x‖² − x·(1, 2, 0) = −9/4`.
#[test]
fn second_order_cone_projection() {
    let qp = Qp::new(3, 0, 3, 0);
    let p = [(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)];
    assert_eq!(qp.objective(&p, &[-1.0, -2.0, 0.0]), TRUE);
    let g = [(0, 0, -1.0), (1, 1, -1.0), (2, 2, -1.0)];
    assert_eq!(qp.inequalities(&g, &[0.0; 3]), TRUE);
    assert_eq!(qp.cone(POUNCE_CONE_SOC, 3, 0.0), 0);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL);
    let s = qp.solution();
    close("x", &s.x, &[1.5, 1.5, 0.0], 1e-5);
    close("obj", &[s.obj], &[-2.25], TOL);
}

/// Exponential cone alone: `min z` s.t. `(1, 1, z) ∈ K_exp`, i.e.
/// `1·exp(1/1) ≤ z`: `z = e`. `G = (0, 0, −1)ᵀ`, `h = (1, 1, 0)`.
#[test]
fn exponential_cone() {
    let qp = Qp::new(1, 0, 3, 0);
    assert_eq!(qp.objective(&[], &[1.0]), TRUE);
    assert_eq!(qp.inequalities(&[(2, 0, -1.0)], &[1.0, 1.0, 0.0]), TRUE);
    assert_eq!(qp.cone(POUNCE_CONE_EXP, 3, 0.0), 0);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL);
    let s = qp.solution();
    close("x", &s.x, &[std::f64::consts::E], 1e-5);
    // The non-symmetric driver has no homogeneous scalars to report.
    let c = qp.certificate();
    assert!(c.tau.is_nan() && c.kappa.is_nan(), "{c:?}");
}

/// Power cone, `α = ½`: `max x0` s.t. `(x0, x1, x2) ∈ K_½`, `x1 = 2`,
/// `x2 = ½` (`python/tests/test_socp.py:115`): `x0 = 2^½·(½)^½ = 1`.
#[test]
fn power_cone() {
    let qp = Qp::new(3, 2, 3, 0);
    assert_eq!(qp.objective(&[], &[-1.0, 0.0, 0.0]), TRUE);
    assert_eq!(
        qp.equalities(&[(0, 1, 1.0), (1, 2, 1.0)], &[2.0, 0.5]),
        TRUE
    );
    let g = [(0, 0, -1.0), (1, 1, -1.0), (2, 2, -1.0)];
    assert_eq!(qp.inequalities(&g, &[0.0; 3]), TRUE);
    assert_eq!(qp.cone(POUNCE_CONE_POW, 3, 0.5), 0);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL);
    let s = qp.solution();
    close("x", &s.x, &[1.0, 2.0, 0.5], 1e-5);
    close("obj", &[s.obj], &[-1.0], 1e-5);
}

/// An exponent just inside `(0, 1)` is accepted and solves: the passing side
/// of the `alpha` check. `x0 = 2^α·(½)^(1−α)` at `α = 0.01`.
#[test]
fn power_cone_exponent_near_the_edge() {
    let alpha = 0.01;
    let qp = Qp::new(3, 2, 3, 0);
    assert_eq!(qp.objective(&[], &[-1.0, 0.0, 0.0]), TRUE);
    assert_eq!(
        qp.equalities(&[(0, 1, 1.0), (1, 2, 1.0)], &[2.0, 0.5]),
        TRUE
    );
    let g = [(0, 0, -1.0), (1, 1, -1.0), (2, 2, -1.0)];
    assert_eq!(qp.inequalities(&g, &[0.0; 3]), TRUE);
    assert_eq!(qp.cone(POUNCE_CONE_POW, 3, alpha), 0);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL);
    let want = 2f64.powf(alpha) * 0.5f64.powf(1.0 - alpha);
    close("x0", &qp.solution().x[..1], &[want], 1e-5);
}

/// PSD cone, order 3: `max λ` s.t. `M − λI ⪰ 0` for
/// `M = [[2, 1, 0], [1, 2, 0], [0, 0, 3]]`, so `λ = λ_min(M) = 1`. The slack
/// is `svec(M − λI)` (lower triangle by columns, off-diagonals × √2):
/// `h = svec(M) = (2, √2, 0, 2, 0, 3)`, `G`'s column `svec(I) = (1, 0, 0, 1, 0, 1)`.
#[test]
fn psd_cone() {
    let r2 = std::f64::consts::SQRT_2;
    let qp = Qp::new(1, 0, 6, 0);
    assert_eq!(qp.objective(&[], &[-1.0]), TRUE);
    let g = [(0, 0, 1.0), (3, 0, 1.0), (5, 0, 1.0)];
    assert_eq!(qp.inequalities(&g, &[2.0, r2, 0.0, 2.0, 0.0, 3.0]), TRUE);
    assert_eq!(qp.cone(POUNCE_CONE_PSD, 3, 0.0), 0);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL);
    let s = qp.solution();
    close("x", &s.x, &[1.0], 1e-5);
    close("obj", &[s.obj], &[-1.0], 1e-5);
}

/// `alpha` is read only by the power cone: any value with another kind is
/// ignored, not refused.
#[test]
fn alpha_is_ignored_off_the_power_cone() {
    let qp = Qp::new(1, 0, 9, 0);
    assert_eq!(qp.cone(POUNCE_CONE_NONNEG, 1, 7.0), 0);
    assert_eq!(qp.cone(POUNCE_CONE_SOC, 2, -3.0), 1);
    assert_eq!(qp.cone(POUNCE_CONE_EXP, 3, f64::NAN), 2);
    assert_eq!(qp.cone(POUNCE_CONE_PSD, 2, 2.0), 3);
}

// ---------------------------------------------------------------------------
// Indefinite P
// ---------------------------------------------------------------------------

/// `min ½x0² − ½x1²` over `[−1, 1]²`: local minima `(0, ±1)`, `−½`.
fn indefinite() -> Qp {
    let qp = Qp::new(2, 0, 0, 0);
    assert_eq!(
        qp.objective(&[(0, 0, 1.0), (1, 1, -1.0)], &[0.0, 0.0]),
        TRUE
    );
    assert_eq!(qp.bounds(Some(&[-1.0, -1.0]), Some(&[1.0, 1.0])), TRUE);
    qp
}

// error-path: PounceQpSolve
#[test]
fn indefinite_p_is_refused_under_the_ipm() {
    let qp = indefinite();
    assert_eq!(qp.solve(), POUNCE_QP_ERR_INDEFINITE);
    assert!(!qp.has_result());
}

#[test]
fn indefinite_p_is_refused_under_an_explicit_ipm() {
    let qp = indefinite();
    assert_eq!(qp.str_opt("method", "ipm"), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_INDEFINITE);
}

#[test]
fn indefinite_p_is_solved_under_the_active_set_engine() {
    let qp = indefinite();
    assert_eq!(qp.str_opt("method", "active-set"), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL);
    let s = qp.solution();
    close("obj", &[s.obj], &[-0.5], TOL);
    close("|x|", &[s.x[0], s.x[1].abs()], &[0.0, 1.0], TOL);
}

/// A caller's `indefinite` claim refuses a PSD `P` under the IPM: the claim
/// is used as given, not re-checked.
#[test]
fn claimed_indefinite_is_refused_under_the_ipm() {
    let qp = Qp::new(1, 0, 0, 0);
    assert_eq!(qp.objective(&[(0, 0, 1.0)], &[-1.0]), TRUE);
    assert_eq!(qp.str_opt("hessian_inertia", "indefinite"), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_INDEFINITE);
}

/// `unknown` and `psd` claims pass the IPM screen (the screen refuses only
/// `Indefinite`), and `auto` after a claim restores the verdict.
#[test]
fn claimed_unknown_and_psd_pass_the_ipm_screen() {
    for claim in ["unknown", "psd"] {
        let qp = Qp::new(1, 0, 0, 0);
        assert_eq!(qp.objective(&[(0, 0, 1.0)], &[-1.0]), TRUE);
        assert_eq!(qp.str_opt("hessian_inertia", claim), TRUE);
        assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL, "{claim}");
        close("x", &qp.solution().x, &[1.0], TOL);
    }
    let qp = indefinite();
    assert_eq!(qp.str_opt("hessian_inertia", "psd"), TRUE);
    assert_eq!(qp.str_opt("hessian_inertia", "auto"), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_INDEFINITE);
}

/// The screen also guards the conic path (`solve_socp` runs Python's check
/// too): an indefinite `P` with a cone is refused.
#[test]
fn indefinite_p_is_refused_with_cones() {
    let qp = Qp::new(2, 0, 2, 0);
    assert_eq!(
        qp.objective(&[(0, 0, 1.0), (1, 1, -1.0)], &[0.0, 0.0]),
        TRUE
    );
    assert_eq!(
        qp.inequalities(&[(0, 0, 1.0), (1, 1, 1.0)], &[1.0, 1.0]),
        TRUE
    );
    assert_eq!(qp.cone(POUNCE_CONE_NONNEG, 2, 0.0), 0);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_INDEFINITE);
}

// ---------------------------------------------------------------------------
// Cones not covering m_ineq
// ---------------------------------------------------------------------------

#[test]
fn cones_covering_fewer_rows_than_m_ineq_are_refused() {
    let qp = nonneg_model();
    // A fresh 3-row model with a 2-row cone.
    let short = Qp::new(2, 0, 3, 0);
    assert_eq!(short.objective(&[], &[-1.0, -1.0]), TRUE);
    assert_eq!(
        short.inequalities(&[(0, 0, 1.0), (1, 1, 1.0)], &[1.0, 2.0, 0.0]),
        TRUE
    );
    assert_eq!(short.cone(POUNCE_CONE_NONNEG, 2, 0.0), 0);
    assert_eq!(short.solve(), POUNCE_QP_ERR_CONES);
    assert!(!short.has_result());
    // Control: the covering model solves.
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL);
}

#[test]
fn cones_covering_more_rows_than_m_ineq_are_refused() {
    let qp = nonneg_model();
    assert_eq!(qp.cone(POUNCE_CONE_SOC, 1, 0.0), 1);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_CONES);
}

/// An exp cone is always 3 rows: one declared on 2 rows does not cover them.
#[test]
fn an_exp_cone_on_too_few_rows_is_refused() {
    let qp = Qp::new(1, 0, 2, 0);
    assert_eq!(qp.objective(&[], &[1.0]), TRUE);
    assert_eq!(qp.inequalities(&[], &[0.0, 0.0]), TRUE);
    assert_eq!(qp.cone(POUNCE_CONE_EXP, 3, 0.0), 0);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_CONES);
}

/// A PSD cone of order `k` spans `k(k+1)/2` rows, not `k`.
#[test]
fn a_psd_cone_counts_svec_rows() {
    let qp = Qp::new(1, 0, 2, 0);
    assert_eq!(qp.objective(&[], &[1.0]), TRUE);
    assert_eq!(qp.inequalities(&[], &[0.0, 0.0]), TRUE);
    assert_eq!(qp.cone(POUNCE_CONE_PSD, 2, 0.0), 0);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_CONES);
}

#[test]
fn psd_mixed_with_a_power_cone_is_refused() {
    let qp = Qp::new(1, 0, 4, 0);
    assert_eq!(qp.objective(&[], &[1.0]), TRUE);
    assert_eq!(qp.inequalities(&[], &[0.0; 4]), TRUE);
    assert_eq!(qp.cone(POUNCE_CONE_PSD, 1, 0.0), 0);
    assert_eq!(qp.cone(POUNCE_CONE_POW, 3, 0.5), 1);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_CONES);
}

#[test]
fn psd_mixed_with_an_exp_cone_is_refused() {
    let qp = Qp::new(1, 0, 4, 0);
    assert_eq!(qp.objective(&[], &[1.0]), TRUE);
    assert_eq!(qp.inequalities(&[], &[0.0; 4]), TRUE);
    assert_eq!(qp.cone(POUNCE_CONE_EXP, 3, 0.0), 0);
    assert_eq!(qp.cone(POUNCE_CONE_PSD, 1, 0.0), 1);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_CONES);
}

// ---------------------------------------------------------------------------
// AddCone: dim ≤ 0 and alpha outside (0, 1)
// ---------------------------------------------------------------------------

fn refused(qp: &Qp, kind: i32, dim: Index, alpha: f64) {
    assert_eq!(
        qp.cone(kind, dim, alpha),
        POUNCE_QP_ERR_ARGUMENT,
        "kind {kind} dim {dim} alpha {alpha}"
    );
}

// error-path: PounceQpAddCone
#[test]
fn soc_dim_zero_or_negative_is_refused() {
    let qp = Qp::new(1, 0, 1, 0);
    refused(&qp, POUNCE_CONE_SOC, 0, 0.0);
    refused(&qp, POUNCE_CONE_SOC, -1, 0.0);
    assert_eq!(qp.cone(POUNCE_CONE_SOC, 1, 0.0), 0, "nothing was added");
}

#[test]
fn psd_order_zero_or_negative_is_refused() {
    let qp = Qp::new(1, 0, 1, 0);
    refused(&qp, POUNCE_CONE_PSD, 0, 0.0);
    refused(&qp, POUNCE_CONE_PSD, -2, 0.0);
    assert_eq!(qp.cone(POUNCE_CONE_PSD, 1, 0.0), 0, "nothing was added");
}

#[test]
fn nonneg_negative_dim_is_refused() {
    let qp = Qp::new(1, 0, 1, 0);
    refused(&qp, POUNCE_CONE_NONNEG, -1, 0.0);
    refused(&qp, POUNCE_CONE_NONNEG, Index::MIN, 0.0);
    assert_eq!(qp.cone(POUNCE_CONE_NONNEG, 1, 0.0), 0, "nothing was added");
}

/// The 3-D cones take exactly `dim = 3`, so `0` and negatives are refused
/// with every other value but 3.
#[test]
fn exp_and_power_dims_other_than_three_are_refused() {
    let qp = Qp::new(1, 0, 3, 0);
    for d in [0, -3, 2, 4] {
        refused(&qp, POUNCE_CONE_EXP, d, 0.0);
        refused(&qp, POUNCE_CONE_POW, d, 0.5);
    }
    assert_eq!(qp.cone(POUNCE_CONE_EXP, 3, 0.0), 0, "nothing was added");
}

#[test]
fn power_exponent_outside_the_open_unit_interval_is_refused() {
    let qp = Qp::new(1, 0, 3, 0);
    for a in [
        0.0,
        1.0,
        -0.5,
        1.5,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        refused(&qp, POUNCE_CONE_POW, 3, a);
    }
    assert_eq!(qp.cone(POUNCE_CONE_POW, 3, 0.999), 0, "nothing was added");
}

#[test]
fn unknown_cone_kind_is_refused() {
    let qp = Qp::new(1, 0, 3, 0);
    for k in [-1, 5, 99] {
        refused(&qp, k, 3, 0.5);
    }
}

#[test]
fn add_cone_on_a_null_handle_is_refused() {
    assert_eq!(
        unsafe { PounceQpAddCone(null_mut(), POUNCE_CONE_NONNEG, 1, 0.0) },
        POUNCE_QP_ERR_NULL
    );
}

// ---------------------------------------------------------------------------
// NULL arrays: one fixture per argument, and the nnz = 0 side
// ---------------------------------------------------------------------------

const R: [Index; 1] = [0];
const C: [Index; 1] = [0];
const V: [f64; 1] = [1.0];

/// Call `set(qp, row, col, val, vec)` with each of the four arrays NULL in
/// turn (the triplet ones at `nnz = 1`) and assert each is refused and
/// leaves the handle incomplete.
fn each_null_refused(
    m_eq: usize,
    m_ineq: usize,
    set: unsafe extern "C" fn(
        PounceQp,
        Index,
        *const Index,
        *const Index,
        *const Number,
        *const Number,
    ) -> Bool,
) {
    let vec = [0.0];
    let cases: [(
        &str,
        *const Index,
        *const Index,
        *const Number,
        *const Number,
    ); 4] = [
        ("row", null(), C.as_ptr(), V.as_ptr(), vec.as_ptr()),
        ("col", R.as_ptr(), null(), V.as_ptr(), vec.as_ptr()),
        ("val", R.as_ptr(), C.as_ptr(), null(), vec.as_ptr()),
        ("vector", R.as_ptr(), C.as_ptr(), V.as_ptr(), null()),
    ];
    for (what, r, c, v, b) in cases {
        let qp = Qp::new(1, m_eq, m_ineq, 0);
        assert_eq!(unsafe { set(qp.h, 1, r, c, v, b) }, FALSE, "NULL {what}");
        if m_eq + m_ineq > 0 {
            // The objective is required too; set it so INCOMPLETE is about
            // the refused rows.
            assert_eq!(qp.objective(&[], &[1.0]), TRUE);
        }
        assert_eq!(
            qp.solve(),
            POUNCE_QP_ERR_INCOMPLETE,
            "NULL {what}: nothing set"
        );
    }
    // nnz = 0 with NULL triplet arrays is accepted (nothing is read).
    let qp = Qp::new(1, m_eq, m_ineq, 0);
    assert_eq!(
        unsafe { set(qp.h, 0, null(), null(), null(), vec.as_ptr()) },
        TRUE
    );
}

// error-path: PounceQpSetObjective
#[test]
fn objective_null_arrays_are_refused() {
    each_null_refused(0, 0, PounceQpSetObjective);
}

// error-path: PounceQpSetEqualities
#[test]
fn equality_null_arrays_are_refused() {
    each_null_refused(1, 0, PounceQpSetEqualities);
}

// error-path: PounceQpSetInequalities
#[test]
fn inequality_null_arrays_are_refused() {
    each_null_refused(0, 1, PounceQpSetInequalities);
}

/// `b` / `h` may be NULL only when there are no rows to describe.
#[test]
fn null_right_hand_side_is_accepted_with_no_rows() {
    let qp = Qp::new(1, 0, 0, 0);
    unsafe {
        assert_eq!(
            PounceQpSetEqualities(qp.h, 0, null(), null(), null(), null()),
            TRUE
        );
        assert_eq!(
            PounceQpSetInequalities(qp.h, 0, null(), null(), null(), null()),
            TRUE
        );
    }
}

/// NULL bounds are "no bound on that side", not a refusal: `min x0` over
/// `x0 ≥ 1` with `ub` NULL; then `min −x0` over `x0 ≤ 2` with `lb` NULL.
#[test]
fn null_bounds_mean_unbounded_sides() {
    let qp = Qp::new(1, 0, 0, 0);
    assert_eq!(qp.objective(&[], &[1.0]), TRUE);
    assert_eq!(qp.bounds(Some(&[1.0]), None), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL);
    close("x", &qp.solution().x, &[1.0], TOL);

    let qp = Qp::new(1, 0, 0, 0);
    assert_eq!(qp.objective(&[], &[-1.0]), TRUE);
    assert_eq!(qp.bounds(None, Some(&[2.0])), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL);
    close("x", &qp.solution().x, &[2.0], TOL);

    // Both NULL resets an earlier box: `min −x0` is then unbounded.
    assert_eq!(qp.bounds(None, None), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_DUAL_INFEASIBLE);
}

/// `±1e19` and beyond are no bound (the Ipopt convention), just inside is a
/// bound.
#[test]
fn bound_magnitudes_at_the_infinity_convention() {
    let qp = Qp::new(1, 0, 0, 0);
    assert_eq!(qp.objective(&[], &[-1.0]), TRUE);
    assert_eq!(qp.bounds(None, Some(&[1e19])), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_DUAL_INFEASIBLE);
    let qp = Qp::new(1, 0, 0, 0);
    assert_eq!(qp.objective(&[], &[1.0]), TRUE);
    assert_eq!(qp.bounds(Some(&[-1e19]), None), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_DUAL_INFEASIBLE);
    let qp = Qp::new(1, 0, 0, 0);
    assert_eq!(qp.objective(&[], &[-1.0]), TRUE);
    assert_eq!(qp.bounds(None, Some(&[1e3])), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL);
    close("x", &qp.solution().x, &[1e3], TOL);
}

// error-path: PounceQpSetBounds
#[test]
fn nan_bound_is_refused_on_either_side() {
    let qp = Qp::new(1, 0, 0, 0);
    assert_eq!(qp.bounds(Some(&[f64::NAN]), None), FALSE);
    assert_eq!(qp.bounds(None, Some(&[f64::NAN])), FALSE);
}

// ---------------------------------------------------------------------------
// Incomplete handles: one fixture per missing piece
// ---------------------------------------------------------------------------

#[test]
fn no_objective_is_incomplete() {
    let qp = Qp::new(1, 0, 0, 0);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_INCOMPLETE);
}

#[test]
fn equality_rows_never_set_are_incomplete() {
    let qp = Qp::new(1, 1, 0, 0);
    assert_eq!(qp.objective(&[], &[1.0]), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_INCOMPLETE);
}

#[test]
fn inequality_rows_never_set_are_incomplete() {
    let qp = Qp::new(1, 0, 1, 0);
    assert_eq!(qp.objective(&[], &[1.0]), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_INCOMPLETE);
}

// ---------------------------------------------------------------------------
// Unknown option keys
// ---------------------------------------------------------------------------

// error-path: PounceQpSetNumOption
#[test]
fn unknown_numeric_option_key_is_refused() {
    let qp = Qp::new(1, 0, 0, 0);
    for k in [
        "bogus",
        "",
        "TOL",
        "tol ",
        "max_iters",
        "presolve",
        "method",
    ] {
        assert_eq!(qp.num_opt(k, 0.5), FALSE, "{k:?}");
    }
}

// error-path: PounceQpSetStrOption
#[test]
fn unknown_string_option_key_is_refused() {
    let qp = Qp::new(1, 0, 0, 0);
    for k in [
        "bogus",
        "",
        "Method",
        "presolve",
        "hsde",
        "tol",
        "solver_selection",
    ] {
        assert_eq!(qp.str_opt(k, "yes"), FALSE, "{k:?}");
    }
}

/// A key that is not UTF-8 is an unknown key, not a crash.
#[test]
fn non_utf8_option_key_is_refused() {
    let qp = Qp::new(1, 0, 0, 0);
    let key = [0xffu8, 0xfe, 0];
    let ok = unsafe { PounceQpSetNumOption(qp.h, key.as_ptr().cast(), 1e-8) };
    assert_eq!(ok, FALSE);
    let ok = unsafe { PounceQpSetStrOption(qp.h, key.as_ptr().cast(), cs("yes").as_ptr()) };
    assert_eq!(ok, FALSE);
}

// ---------------------------------------------------------------------------
// Get* before any solve: one fixture per getter
// ---------------------------------------------------------------------------

// error-path: PounceQpGetSolution
#[test]
fn get_solution_before_solve_is_false() {
    let qp = nonneg_model();
    let mut x = [7.0; 2];
    let mut obj = 7.0;
    let mut it: Index = 7;
    let n = null_mut();
    let ok = unsafe { PounceQpGetSolution(qp.h, x.as_mut_ptr(), n, n, n, n, &mut obj, &mut it) };
    assert_eq!(ok, FALSE);
    assert_eq!((x, obj, it), ([7.0; 2], 7.0, 7), "nothing written");
}

// error-path: PounceQpGetResiduals
#[test]
fn get_residuals_before_solve_is_false() {
    let qp = nonneg_model();
    let mut r = PounceQpResiduals {
        kkt_error: 7.0,
        ..Default::default()
    };
    assert_eq!(unsafe { PounceQpGetResiduals(qp.h, &mut r) }, FALSE);
    assert_eq!(r.kkt_error, 7.0);
}

// error-path: PounceQpGetCertificate
#[test]
fn get_certificate_before_solve_is_false() {
    let qp = nonneg_model();
    let mut c = PounceQpCertificate {
        kind: 7,
        ..Default::default()
    };
    assert_eq!(unsafe { PounceQpGetCertificate(qp.h, &mut c) }, FALSE);
    assert_eq!(c.kind, 7);
}

// error-path: PounceQpWriteSolveReport
#[test]
fn write_solve_report_before_solve_is_false() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qp_cones_and_errors");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("before.json");
    let _ = std::fs::remove_file(&path);
    let qp = nonneg_model();
    let p = cs(path.to_str().unwrap());
    assert_eq!(
        unsafe { PounceQpWriteSolveReport(qp.h, p.as_ptr(), null()) },
        FALSE
    );
    assert!(!path.exists());
}

// ---------------------------------------------------------------------------
// Options the conic path refuses: one fixture each
// ---------------------------------------------------------------------------

fn conic_refuses(set: impl Fn(&Qp) -> Bool) {
    let qp = nonneg_model();
    assert_eq!(set(&qp), TRUE, "the option itself is valid");
    assert_eq!(qp.solve(), POUNCE_QP_ERR_OPTIONS);
    assert!(!qp.has_result());
}

#[test]
fn conic_refuses_method_active_set() {
    conic_refuses(|q| q.str_opt("method", "active-set"));
}

#[test]
fn conic_refuses_tau() {
    conic_refuses(|q| q.num_opt("tau", 0.9));
}

#[test]
fn conic_refuses_tau_max() {
    conic_refuses(|q| q.num_opt("tau_max", 0.99));
}

#[test]
fn conic_refuses_qp_presolve() {
    conic_refuses(|q| q.str_opt("qp_presolve", "no"));
}

#[test]
fn conic_refuses_qp_reg() {
    conic_refuses(|q| q.num_opt("qp_reg", 1e-10));
}

/// Set to the engine's own default it is still refused: `solve_socp` takes
/// no such argument at all.
#[test]
fn conic_refuses_qp_hsde() {
    conic_refuses(|q| q.str_opt("qp_hsde", "yes"));
}

#[test]
fn conic_refuses_qp_equilibrate() {
    conic_refuses(|q| q.str_opt("qp_equilibrate", "yes"));
}

#[test]
fn conic_refuses_qp_crossover() {
    conic_refuses(|q| q.str_opt("qp_crossover", "no"));
}

/// The options `solve_socp` does take reach the conic path, and an explicit
/// `method=ipm` is not a refusal.
#[test]
fn conic_accepts_tol_max_iter_time_limit_and_method_ipm() {
    let qp = nonneg_model();
    assert_eq!(qp.str_opt("method", "ipm"), TRUE);
    assert_eq!(qp.num_opt("tol", 1e-10), TRUE);
    assert_eq!(qp.num_opt("max_iter", 100.0), TRUE);
    assert_eq!(qp.num_opt("time_limit", 60.0), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL);
    assert!(qp.residuals().kkt_error < 1e-9);
}

// ---------------------------------------------------------------------------
// Options the active-set engine refuses, and the ones it reads (no cones)
// ---------------------------------------------------------------------------

/// `min ½‖x‖² − x0 − x1` over `0 ≤ x ≤ 10` → `(1, 1)`, `−1`.
fn box_qp() -> Qp {
    let qp = Qp::new(2, 0, 0, 0);
    assert_eq!(
        qp.objective(&[(0, 0, 1.0), (1, 1, 1.0)], &[-1.0, -1.0]),
        TRUE
    );
    assert_eq!(qp.bounds(Some(&[0.0, 0.0]), Some(&[10.0, 10.0])), TRUE);
    assert_eq!(qp.str_opt("method", "active-set"), TRUE);
    qp
}

#[test]
fn active_set_refuses_qp_reg() {
    let qp = box_qp();
    assert_eq!(qp.num_opt("qp_reg", 1e-10), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_OPTIONS);
}

#[test]
fn active_set_refuses_qp_hsde() {
    let qp = box_qp();
    assert_eq!(qp.str_opt("qp_hsde", "no"), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_OPTIONS);
}

#[test]
fn active_set_refuses_qp_crossover() {
    let qp = box_qp();
    assert_eq!(qp.str_opt("qp_crossover", "no"), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_ERR_OPTIONS);
}

#[test]
fn active_set_reads_qp_presolve_and_qp_equilibrate() {
    for (k, v) in [
        ("qp_presolve", "yes"),
        ("qp_presolve", "no"),
        ("qp_equilibrate", "no"),
    ] {
        let qp = box_qp();
        assert_eq!(qp.str_opt(k, v), TRUE);
        assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL, "{k}={v}");
        close("x", &qp.solution().x, &[1.0, 1.0], TOL);
    }
}

/// `tau` / `tau_max` on the QP path: accepted, and the IPM still solves.
#[test]
fn ipm_accepts_tau_and_tau_max() {
    for (k, v) in [("tau", 0.5), ("tau_max", 0.9)] {
        let qp = Qp::new(2, 0, 0, 0);
        assert_eq!(
            qp.objective(&[(0, 0, 1.0), (1, 1, 1.0)], &[-1.0, -1.0]),
            TRUE
        );
        assert_eq!(qp.bounds(Some(&[0.0, 0.0]), Some(&[10.0, 10.0])), TRUE);
        assert_eq!(qp.num_opt(k, v), TRUE);
        assert_eq!(qp.solve(), POUNCE_QP_STATUS_OPTIMAL, "{k}={v}");
        close("x", &qp.solution().x, &[1.0, 1.0], TOL);
    }
}

// ---------------------------------------------------------------------------
// Statuses and certificates, QP path and conic path
// ---------------------------------------------------------------------------

/// `t ≤ −1` (a nonneg row) with `(t, x) ∈ Q²`: no point.
#[test]
fn conic_primal_infeasible_carries_a_farkas_certificate() {
    let qp = Qp::new(2, 0, 3, 0);
    assert_eq!(qp.objective(&[], &[0.0, 0.0]), TRUE);
    let g = [(0, 0, 1.0), (1, 0, -1.0), (2, 1, -1.0)];
    assert_eq!(qp.inequalities(&g, &[-1.0, 0.0, 0.0]), TRUE);
    assert_eq!(qp.cone(POUNCE_CONE_NONNEG, 1, 0.0), 0);
    assert_eq!(qp.cone(POUNCE_CONE_SOC, 2, 0.0), 1);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_PRIMAL_INFEASIBLE);
    let c = qp.certificate();
    let s = qp.solution();
    assert_eq!(c.kind, POUNCE_QP_CERT_PRIMAL_INFEASIBLE);
    let norm =
        s.z.iter()
            .chain(&s.z_lb)
            .chain(&s.z_ub)
            .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert_eq!(c.scale, norm, "the ray's inf-norm");
    assert!(c.scale > 0.0);
    // A Farkas ray: Gᵀz = 0 with hᵀz < 0 (here z = (a, a, 0), hᵀz = −a).
    let (gz0, gz1) = (s.z[0] - s.z[1], -s.z[2]);
    assert!(
        gz0.abs() < 1e-6 * c.scale && gz1.abs() < 1e-6 * c.scale,
        "{:?}",
        s.z
    );
    assert!(-s.z[0] < 0.0);
}

/// `min −x` over `(t, x) ∈ Q²` with `t` free: unbounded along `(1, 1)`.
#[test]
fn conic_dual_infeasible_carries_a_recession_ray() {
    let qp = Qp::new(2, 0, 2, 0);
    assert_eq!(qp.objective(&[], &[0.0, -1.0]), TRUE);
    assert_eq!(
        qp.inequalities(&[(0, 0, -1.0), (1, 1, -1.0)], &[0.0, 0.0]),
        TRUE
    );
    assert_eq!(qp.cone(POUNCE_CONE_SOC, 2, 0.0), 0);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_DUAL_INFEASIBLE);
    let c = qp.certificate();
    let s = qp.solution();
    assert_eq!(c.kind, POUNCE_QP_CERT_DUAL_INFEASIBLE);
    assert_eq!(c.scale, s.x[0].abs().max(s.x[1].abs()));
    // A recession direction: in the cone (t ≥ |x|) and cᵀd = −x < 0.
    assert!(
        s.x[0] >= s.x[1].abs() * (1.0 - 1e-6) && s.x[1] > 0.0,
        "{:?}",
        s.x
    );
}

#[test]
fn conic_iteration_limit() {
    let qp = nonneg_model();
    assert_eq!(qp.num_opt("max_iter", 1.0), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_ITERATION_LIMIT);
    assert_eq!(qp.certificate().kind, POUNCE_QP_CERT_NONE);
    assert_eq!(qp.solution().iters, 1);
}

/// A zero budget: every engine gives up at its first deadline check.
#[test]
fn zero_time_limit_is_a_time_limit_status() {
    let qp = Qp::new(2, 0, 0, 0);
    assert_eq!(
        qp.objective(&[(0, 0, 1.0), (1, 1, 1.0)], &[-1.0, -1.0]),
        TRUE
    );
    assert_eq!(qp.bounds(Some(&[0.0, 0.0]), Some(&[10.0, 10.0])), TRUE);
    assert_eq!(qp.num_opt("time_limit", 0.0), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_TIME_LIMIT);
    assert_eq!(qp.certificate().kind, POUNCE_QP_CERT_NONE);
}

/// The QP path's certificates on the active-set engine (the IPM's are in
/// `qp_handle.rs` and the parity rows): presolve proves both before any
/// iteration.
#[test]
fn presolved_infeasible_and_unbounded_on_the_active_set_engine() {
    let qp = Qp::new(1, 0, 1, 0);
    assert_eq!(qp.objective(&[], &[1.0]), TRUE);
    assert_eq!(qp.inequalities(&[(0, 0, -1.0)], &[-1.0]), TRUE);
    assert_eq!(qp.bounds(None, Some(&[0.0])), TRUE);
    assert_eq!(qp.str_opt("method", "active-set"), TRUE);
    assert_eq!(qp.str_opt("qp_presolve", "yes"), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_PRIMAL_INFEASIBLE);
    assert_eq!(qp.certificate().kind, POUNCE_QP_CERT_PRIMAL_INFEASIBLE);
    assert_eq!(qp.solution().iters, 0);

    let qp = Qp::new(1, 0, 0, 0);
    assert_eq!(qp.objective(&[], &[-1.0]), TRUE);
    assert_eq!(qp.bounds(Some(&[0.0]), None), TRUE);
    assert_eq!(qp.str_opt("method", "active-set"), TRUE);
    assert_eq!(qp.str_opt("qp_presolve", "yes"), TRUE);
    assert_eq!(qp.solve(), POUNCE_QP_STATUS_DUAL_INFEASIBLE);
    assert_eq!(qp.certificate().kind, POUNCE_QP_CERT_DUAL_INFEASIBLE);
}
