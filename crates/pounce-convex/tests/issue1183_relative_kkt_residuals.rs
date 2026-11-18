//! gh #1183: `QpSolution::relative_kkt_residuals`, the ruler the CLI's
//! LP / convex-QP → NLP fallback ranks its two candidates by.
//!
//! The CLI tests (`crates/pounce-cli/tests/issue_1183_*`) check the ranking
//! end to end on NETLIB models. These pin the measure itself on a model whose
//! residuals can be written down by hand, one residual per test, so a change
//! to any one normalization fails here rather than as a moved verdict on a
//! 300-row LP.
//!
//! The model: `min x0 + 2 x1  s.t.  x0 + x1 = 1,  x0 ≤ 0.75,  x ≥ 0`.
//! Optimum `x = (0.75, 0.25)`, `f = 1.25`; with stationarity
//! `c + Aᵀy + Gᵀz − z_lb + z_ub = 0` the multipliers are `y = −2`, `z = 1`.

use pounce_convex::{QpProblem, QpSolution, QpStatus, Triplet};

fn model() -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![1.0, 2.0],
        a: vec![
            Triplet {
                row: 0,
                col: 0,
                val: 1.0,
            },
            Triplet {
                row: 0,
                col: 1,
                val: 1.0,
            },
        ],
        b: vec![1.0],
        g: vec![Triplet {
            row: 0,
            col: 0,
            val: 1.0,
        }],
        h: vec![0.75],
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY, f64::INFINITY],
    }
}

fn point(x: [f64; 2], y: f64, z: f64, z_lb: [f64; 2]) -> QpSolution {
    QpSolution {
        status: QpStatus::Optimal,
        x: x.to_vec(),
        y: vec![y],
        z: vec![z],
        z_lb: z_lb.to_vec(),
        z_ub: vec![0.0, 0.0],
        obj: x[0] + 2.0 * x[1],
        iters: 0,
        iterates: Vec::new(),
    }
}

#[test]
fn the_exact_optimum_reads_zero_on_every_residual() {
    let r = point([0.75, 0.25], -2.0, 1.0, [0.0, 0.0]).relative_kkt_residuals(&model());
    assert!(r.max() < 1e-15, "{r:?}");
}

/// Primal: the equality off by `1e-3`, relative to `1 + max(‖b‖∞, ‖h‖∞) = 2`.
#[test]
fn primal_is_relative_to_the_right_hand_side() {
    let r = point([0.75, 0.251], -2.0, 1.0, [0.0, 0.0]).relative_kkt_residuals(&model());
    assert!((r.primal - 0.5e-3).abs() < 1e-15, "{r:?}");
    assert_eq!(r.primal_cone, 0.0);
}

/// Primal cone: `x1 = −1e-3` sits outside its box, same normalizer.
#[test]
fn a_box_violation_is_the_primal_cone_residual() {
    let r = point([0.75, -1e-3], -2.0, 1.0, [0.0, 0.0]).relative_kkt_residuals(&model());
    assert!((r.primal_cone - 0.5e-3).abs() < 1e-15, "{r:?}");
}

/// Dual: `y` off by `1e-2` leaves `1e-2` of stationarity in both columns,
/// relative to `1 + ‖c‖∞ = 3`.
#[test]
fn dual_is_relative_to_the_cost() {
    let r = point([0.75, 0.25], -2.01, 1.0, [0.0, 0.0]).relative_kkt_residuals(&model());
    assert!((r.dual - 1e-2 / 3.0).abs() < 1e-15, "{r:?}");
}

/// Gap: the *sum* of complementarity, which is what separates this measure
/// from `kkt_residuals_above_floor`'s per-product maximum. A bound multiplier
/// on a variable that is not at its bound is complementarity error on its own,
/// and here it is balanced by `z` so stationarity stays exact: `z_lb0 = 0.1`
/// and `z = 1.1` gives column 0 `1 − 2 + 1.1 − 0.1 = 0`. Its product is
/// `0.1 · 0.75 = 0.075`; then `p = 1.25`, `d = −bᵀy − hᵀz = 2 − 0.825 =
/// 1.175`, and the gap is `0.075 / (1 + 1.25 + 1.175)`.
#[test]
fn the_gap_is_the_relative_duality_gap() {
    let r = point([0.75, 0.25], -2.0, 1.1, [0.1, 0.0]).relative_kkt_residuals(&model());
    assert!(r.dual < 1e-15, "precondition: stationarity is exact; {r:?}");
    assert!((r.gap - 0.075 / 3.425).abs() < 1e-15, "{r:?}");
}

/// Dual cone: a negative inequality multiplier, relative like `dual`.
#[test]
fn a_wrong_signed_multiplier_is_the_dual_cone_residual() {
    let r = point([0.75, 0.25], -2.0, 1.0, [-0.3, 0.0]).relative_kkt_residuals(&model());
    assert!((r.dual_cone - 0.1).abs() < 1e-15, "{r:?}");
}

/// A broken point must never read as a perfect one: `fold(max)` drops a NaN,
/// which is how a NaN iterate would otherwise win the comparison it is used
/// in.
#[test]
fn a_non_finite_point_reads_as_nan_not_zero() {
    let r = point([f64::NAN, 0.25], -2.0, 1.0, [0.0, 0.0]).relative_kkt_residuals(&model());
    assert!(r.max().is_nan(), "{r:?}");
}
