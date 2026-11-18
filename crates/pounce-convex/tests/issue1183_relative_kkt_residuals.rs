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

/// Primal: the equality off by `1e-3`, relative to the largest of the terms
/// it is a difference of, `1 + max(‖b‖∞, ‖h‖∞, ‖Ax‖∞, ‖Gx‖∞) = 1 + 1.001`.
#[test]
fn primal_is_relative_to_the_terms_it_differences() {
    let r = point([0.75, 0.251], -2.0, 1.0, [0.0, 0.0]).relative_kkt_residuals(&model());
    assert!((r.primal - 1e-3 / 2.001).abs() < 1e-15, "{r:?}");
    assert_eq!(r.primal_cone, 0.0);
}

/// Primal cone: `x1 = −1e-3` sits outside its box, same normalizer.
#[test]
fn a_box_violation_is_the_primal_cone_residual() {
    let r = point([0.75, -1e-3], -2.0, 1.0, [0.0, 0.0]).relative_kkt_residuals(&model());
    assert!((r.primal_cone - 0.5e-3).abs() < 1e-15, "{r:?}");
}

/// Dual: `y` off by `1e-2` leaves `1e-2` of stationarity in both columns,
/// relative to `1 + max(‖c‖∞, ‖Aᵀy‖∞, ‖Gᵀz‖∞, …) = 1 + 2.01`.
#[test]
fn dual_is_relative_to_the_terms_it_sums() {
    let r = point([0.75, 0.25], -2.01, 1.0, [0.0, 0.0]).relative_kkt_residuals(&model());
    assert!((r.dual - 1e-2 / 3.01).abs() < 1e-15, "{r:?}");
}

/// Why the normalizer is the terms and not `c` alone: the same model in
/// epigraph form — `min t  s.t.  t − x0 − 2 x1 = 0`, the shape MIPLIB leo2 is
/// published in — has `‖c‖∞ = 1`, and a cost-sized stationarity error would
/// read against `1 + 1` there however large the costs in the row are.
/// Scaling those costs by `100` must leave the relative residual where it was.
#[test]
fn the_dual_residual_is_scale_invariant_in_epigraph_form() {
    let epi = |k: f64| QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![0.0, 0.0, 1.0],
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
            Triplet {
                row: 1,
                col: 2,
                val: 1.0,
            },
            Triplet {
                row: 1,
                col: 0,
                val: -k,
            },
            Triplet {
                row: 1,
                col: 1,
                val: -2.0 * k,
            },
        ],
        b: vec![1.0, 0.0],
        g: vec![Triplet {
            row: 0,
            col: 0,
            val: 1.0,
        }],
        h: vec![0.75],
        lb: vec![0.0, 0.0, f64::NEG_INFINITY],
        ub: vec![f64::INFINITY; 3],
    };
    // The optimum of the scaled model, with `y` off by 1e-6 relative.
    let pt = |k: f64| QpSolution {
        status: QpStatus::Optimal,
        x: vec![0.75, 0.25, 1.25 * k],
        y: vec![-2.0 * k * (1.0 + 1e-6), -1.0],
        z: vec![k],
        z_lb: vec![0.0; 3],
        z_ub: vec![0.0; 3],
        obj: 1.25 * k,
        iters: 0,
        iterates: Vec::new(),
    };
    // Once the terms dominate the `1 +` (k ≫ 1), the reading is fixed: the
    // residual is `2e-6` of terms of size `k`, at any `k`. Normalized by
    // `1 + ‖c‖∞ = 2` instead it would grow a hundredfold between these two.
    let mid = pt(1e3).relative_kkt_residuals(&epi(1e3)).dual;
    let large = pt(1e5).relative_kkt_residuals(&epi(1e5)).dual;
    assert!((mid - 2e-6).abs() < 1e-8, "{mid:e}");
    assert!((large / mid - 1.0).abs() < 1e-2, "{mid:e} vs {large:e}");
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

/// gh #1184 review: reading each point against its *own* scale lets a point
/// shrink its residuals by inflating its own terms. `min x s.t. x = 1, x ≤ 1`:
/// the honest point carries stationarity error `e` with `y = −1 + e, z = 0`.
/// The inflated one carries a 20% larger error, plus `t = 1e6` on `y` and its
/// exact opposite on the active `z`; the two cancel in stationarity and
/// complementarity, but put `‖Aᵀy‖∞ ≈ ‖Gᵀz‖∞ ≈ t` into its own dual scale.
fn inflation_pair() -> (QpProblem, QpSolution, QpSolution) {
    let (e, t) = (1e-6, 1e6);
    let prob = QpProblem {
        n: 1,
        p_lower: vec![],
        c: vec![1.0],
        a: vec![Triplet {
            row: 0,
            col: 0,
            val: 1.0,
        }],
        b: vec![1.0],
        g: vec![Triplet {
            row: 0,
            col: 0,
            val: 1.0,
        }],
        h: vec![1.0],
        lb: vec![],
        ub: vec![],
    };
    let pt = |y: f64, z: f64| QpSolution {
        status: QpStatus::Optimal,
        x: vec![1.0],
        y: vec![y],
        z: vec![z],
        z_lb: vec![0.0],
        z_ub: vec![0.0],
        obj: 1.0,
        iters: 0,
        iterates: Vec::new(),
    };
    let honest = pt(-1.0 + e, 0.0);
    let inflated = pt(-1.0 + 1.2 * e - t, t);
    (prob, honest, inflated)
}

#[test]
fn read_against_its_own_scale_an_inflated_point_wins() {
    let (prob, honest, inflated) = inflation_pair();
    let h = honest.relative_kkt_residuals(&prob).max();
    let i = inflated.relative_kkt_residuals(&prob).max();
    assert!(
        i < h,
        "precondition: the inflated point must outrank the honest one on \
         its own scale, or this pair does not show the defect; {i:e} vs {h:e}"
    );
}

#[test]
fn read_against_a_shared_scale_it_does_not() {
    let (prob, honest, inflated) = inflation_pair();
    let scale = honest
        .relative_kkt_scale(&prob)
        .shared_with(&inflated.relative_kkt_scale(&prob));
    let h = honest.relative_kkt_residuals_at(&prob, &scale).max();
    let i = inflated.relative_kkt_residuals_at(&prob, &scale).max();
    assert!(h < i, "the larger error must read larger; {h:e} vs {i:e}");
}

/// The shared scale is the larger of the two, component by component, and a
/// non-finite component on one side is ignored: a broken candidate already
/// reads `NaN` on its own residuals and must not zero the other's.
#[test]
fn shared_with_takes_the_larger_finite_component() {
    use pounce_convex::RelativeKktScale;
    let a = RelativeKktScale {
        primal: 2.0,
        dual: 5.0,
        gap: f64::INFINITY,
    };
    let b = RelativeKktScale {
        primal: 3.0,
        dual: f64::NAN,
        gap: 4.0,
    };
    let s = a.shared_with(&b);
    assert_eq!((s.primal, s.dual, s.gap), (3.0, 5.0, 4.0));
    assert_eq!(s, b.shared_with(&a));
}

/// A point read against its own scale through `relative_kkt_residuals_at`
/// is exactly `relative_kkt_residuals`: the split changes no reading.
#[test]
fn the_own_scale_reading_is_unchanged() {
    let p = point([0.75, 0.251], -2.01, 1.1, [0.1, 0.0]);
    let own = p.relative_kkt_scale(&model());
    assert_eq!(
        p.relative_kkt_residuals(&model()),
        p.relative_kkt_residuals_at(&model(), &own)
    );
}
