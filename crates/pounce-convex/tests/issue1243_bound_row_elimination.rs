//! gh#1243: folding single-column orthant rows (every finite variable bound)
//! onto the `(x, x)` diagonal of the factored KKT must not change what the
//! solver answers.
//!
//! The elimination is an exact Schur complement, so the two settings of
//! `QpOptions::kkt_row_elim` solve the same Newton systems and differ only
//! in roundoff. These tests hold them to the same verdict, objective and
//! point, on both symmetric drivers (HSDE and direct) and on the conic entry
//! point, over every outcome class a bound can decide: an optimum with bounds
//! active and inactive, fixed columns (`lb == ub`), free columns alongside
//! boxed ones, primal infeasibility caused by the bounds alone, and dual
//! infeasibility along a half-bounded ray.
//!
//! The kernel-level identities (full product, full solve, inertia shift,
//! exact restoration of the value array) are pinned in `ipm.rs`'s
//! `kkt_row_elim_tests`; this file is the end-to-end half.

use pounce_convex::{
    ConeSpec, QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_ipm, solve_socp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// A feasible, bounded random LP (`quadratic = false`) or convex QP.
///
/// Columns cycle through five bound shapes — boxed, lower only, upper only,
/// free, fixed — so every elimination case meets every driver. Feasibility
/// is by construction (`b = A x₀`, `h = G x₀ + slack` for an `x₀` inside the
/// box); boundedness by a positive-definite `P` or, for an LP, by boxing
/// every column whose cost could otherwise run off to infinity.
fn random_problem(seed: u64, n: usize, quadratic: bool) -> QpProblem {
    let mut st = seed;
    let (m_eq, m_ineq) = (n / 4, n / 3);
    let mut lb = vec![f64::NEG_INFINITY; n];
    let mut ub = vec![f64::INFINITY; n];
    let mut x0 = vec![0.0; n];
    for j in 0..n {
        let lo = -1.0 - 2.0 * lcg(&mut st);
        let hi = 1.0 + 2.0 * lcg(&mut st);
        x0[j] = lo + (hi - lo) * (0.25 + 0.5 * lcg(&mut st));
        match j % 5 {
            0 => (lb[j], ub[j]) = (lo, hi),
            1 => lb[j] = lo,
            2 => ub[j] = hi,
            3 if quadratic => {} // free: P makes it bounded
            3 => (lb[j], ub[j]) = (lo, hi),
            _ => (lb[j], ub[j]) = (x0[j], x0[j]),
        }
    }
    let mut a = Vec::new();
    for r in 0..m_eq {
        for k in 0..3 {
            a.push(Triplet::new(r, (r * 3 + k * 5) % n, lcg(&mut st) - 0.5));
        }
    }
    let mut g = Vec::new();
    for r in 0..m_ineq {
        for k in 0..2 {
            g.push(Triplet::new(r, (r * 7 + k * 3 + 1) % n, lcg(&mut st) - 0.5));
        }
    }
    let row = |t: &[Triplet], m: usize, x: &[f64]| {
        let mut v = vec![0.0; m];
        for e in t {
            v[e.row] += e.val * x[e.col];
        }
        v
    };
    let b = row(&a, m_eq, &x0);
    let mut h = row(&g, m_ineq, &x0);
    for v in &mut h {
        *v += 0.1 + lcg(&mut st);
    }
    // Lower-only and upper-only LP columns get a cost that pushes them
    // *toward* their one finite bound, so the LP stays bounded.
    let c: Vec<f64> = (0..n)
        .map(|j| {
            let mag = 0.2 + lcg(&mut st);
            match j % 5 {
                1 => mag,
                2 => -mag,
                _ => mag * if lcg(&mut st) < 0.5 { -1.0 } else { 1.0 },
            }
        })
        .collect();
    let p_lower = if quadratic {
        let mut p: Vec<Triplet> = (0..n)
            .map(|j| Triplet::new(j, j, 0.5 + lcg(&mut st)))
            .collect();
        for j in 1..n {
            p.push(Triplet::new(j, j - 1, 0.1 * (lcg(&mut st) - 0.5)));
        }
        p
    } else {
        Vec::new()
    };
    QpProblem {
        n,
        p_lower,
        c,
        a,
        b,
        g,
        h,
        lb,
        ub,
    }
}

fn opts(hsde: bool, elim: bool) -> QpOptions {
    QpOptions {
        use_hsde: hsde,
        kkt_row_elim: elim,
        ..QpOptions::default()
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// The two settings agree on the verdict and, for an optimum, on the
/// objective and point — and the eliminated solve's point is a KKT point of
/// the caller's problem on its own account, not merely close to the other.
fn assert_agree(what: &str, prob: &QpProblem, on: &QpSolution, off: &QpSolution) {
    assert_eq!(
        on.status, off.status,
        "{what}: verdict moved (iters {} vs {})",
        on.iters, off.iters
    );
    if on.status == QpStatus::Optimal {
        let scale = 1.0 + off.obj.abs();
        assert!(
            (on.obj - off.obj).abs() <= 1e-7 * scale,
            "{what}: objective {} vs {}",
            on.obj,
            off.obj
        );
        let dx = max_abs_diff(&on.x, &off.x);
        assert!(dx <= 1e-5, "{what}: ‖Δx‖∞ = {dx:e}");
        let kkt = on.kkt_residuals_above_floor(prob).kkt_error();
        assert!(kkt <= 1e-6, "{what}: eliminated solve's KKT error {kkt:e}");
        // Not merely within tolerance: no worse than the full system's, by
        // more than an order of magnitude, on the absolute residuals. This is
        // the check that caught a fixed column's two bound rows being folded
        // (4.5e-8 against 8.4e-11 on `lp_crossover_bridge_rescue`'s seed
        // 311), which agreed on verdict, objective and point.
        let (e_on, e_off) = (
            on.kkt_residuals(prob).kkt_error(),
            off.kkt_residuals(prob).kkt_error(),
        );
        assert!(
            e_on <= 10.0 * e_off + 1e-10,
            "{what}: absolute KKT error {e_on:e} with elimination, {e_off:e} without"
        );
        // Bound multipliers are split back out of the same rows either way.
        assert!(
            max_abs_diff(&on.z_lb, &off.z_lb) <= 1e-4,
            "{what}: z_lb moved"
        );
        assert!(
            max_abs_diff(&on.z_ub, &off.z_ub) <= 1e-4,
            "{what}: z_ub moved"
        );
    }
}

#[test]
fn random_bounded_lps_and_qps_agree_on_both_drivers() {
    let mut iters = Vec::new();
    for seed in 1..=12u64 {
        for &quadratic in &[false, true] {
            let prob = random_problem(seed, 30 + 5 * seed as usize, quadratic);
            for &hsde in &[true, false] {
                let on = solve_qp_ipm(&prob, &opts(hsde, true), backend);
                let off = solve_qp_ipm(&prob, &opts(hsde, false), backend);
                let what = format!("seed {seed} qp={quadratic} hsde={hsde}");
                assert_eq!(
                    off.status,
                    QpStatus::Optimal,
                    "{what}: fixture must be solvable"
                );
                assert_agree(&what, &prob, &on, &off);
                iters.push((on.iters as i64 - off.iters as i64).abs());
            }
        }
    }
    // Same Newton systems up to roundoff. Measured when this was written:
    // all 48 pairs took identical iteration counts. The bound leaves room
    // for a platform's different rounding to part a trajectory late.
    let worst = iters.iter().copied().max().unwrap();
    assert!(worst <= 3, "iteration counts diverged by {worst}");
}

/// Every column bounded on both sides and nothing else: the KKT the
/// elimination leaves is `[P + Σ, Aᵀ; A, −δ]`, with no `(z, z)` block at
/// all — the shape the issue's LP has.
#[test]
fn a_pure_box_lp_eliminates_every_inequality_row() {
    let n = 40;
    let mut st = 7u64;
    let prob = QpProblem {
        n,
        p_lower: Vec::new(),
        c: (0..n).map(|_| lcg(&mut st) - 0.5).collect(),
        a: (0..n / 2)
            .flat_map(|r| {
                [
                    Triplet::new(r, 2 * r, 1.0),
                    Triplet::new(r, 2 * r + 1, -1.0),
                ]
            })
            .collect(),
        b: vec![0.0; n / 2],
        g: Vec::new(),
        h: Vec::new(),
        lb: vec![-1.0; n],
        ub: vec![2.0; n],
    };
    for &hsde in &[true, false] {
        let on = solve_qp_ipm(&prob, &opts(hsde, true), backend);
        let off = solve_qp_ipm(&prob, &opts(hsde, false), backend);
        assert_eq!(off.status, QpStatus::Optimal);
        assert_agree(&format!("box LP hsde={hsde}"), &prob, &on, &off);
    }
}

/// `x₀ + x₁ ≥ 5` with `x ∈ [0, 2]²`: infeasible through the bounds alone,
/// so the Farkas ray lives on the eliminated rows.
#[test]
fn infeasibility_decided_by_the_bounds_is_still_detected() {
    let prob = QpProblem {
        n: 2,
        p_lower: Vec::new(),
        c: vec![1.0, 1.0],
        a: Vec::new(),
        b: Vec::new(),
        g: vec![Triplet::new(0, 0, -1.0), Triplet::new(0, 1, -1.0)],
        h: vec![-5.0],
        lb: vec![0.0, 0.0],
        ub: vec![2.0, 2.0],
    };
    for &hsde in &[true, false] {
        let on = solve_qp_ipm(&prob, &opts(hsde, true), backend);
        let off = solve_qp_ipm(&prob, &opts(hsde, false), backend);
        assert_eq!(on.status, off.status, "hsde={hsde}");
        if hsde {
            assert_eq!(on.status, QpStatus::PrimalInfeasible);
        }
    }
}

/// `min −x₀` with `x₀ ≥ 0` only and `x₁ ∈ [0, 1]`: unbounded along a
/// half-bounded column whose bound row is eliminated.
#[test]
fn unboundedness_along_an_eliminated_bound_is_still_detected() {
    let prob = QpProblem {
        n: 2,
        p_lower: Vec::new(),
        c: vec![-1.0, 1.0],
        a: Vec::new(),
        b: Vec::new(),
        g: Vec::new(),
        h: Vec::new(),
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY, 1.0],
    };
    for &hsde in &[true, false] {
        let on = solve_qp_ipm(&prob, &opts(hsde, true), backend);
        let off = solve_qp_ipm(&prob, &opts(hsde, false), backend);
        assert_eq!(on.status, off.status, "hsde={hsde}");
        if hsde {
            assert_eq!(on.status, QpStatus::DualInfeasible);
        }
    }
}

/// The conic entry point with bounds: the bound rows join the orthant and
/// are eliminated; the second-order cone's rows are not.
#[test]
fn a_boxed_socp_agrees() {
    // min −x₀ − x₁ − x₂  s.t.  ‖(x₀, x₁)‖ ≤ x₂ + 1 written as the SOC
    // (x₂ + 1, x₀, x₁), and x ∈ [−3, 1.5]³.
    let prob = QpProblem {
        n: 3,
        p_lower: Vec::new(),
        c: vec![-1.0, -1.0, -1.0],
        a: Vec::new(),
        b: Vec::new(),
        g: vec![
            Triplet::new(0, 2, -1.0),
            Triplet::new(1, 0, -1.0),
            Triplet::new(2, 1, -1.0),
        ],
        h: vec![1.0, 0.0, 0.0],
        lb: vec![-3.0; 3],
        ub: vec![1.5; 3],
    };
    let cones = [ConeSpec::SecondOrder(3)];
    for &hsde in &[true, false] {
        let on = solve_socp_ipm(&prob, &cones, &opts(hsde, true), backend);
        let off = solve_socp_ipm(&prob, &cones, &opts(hsde, false), backend);
        assert_eq!(off.status, QpStatus::Optimal, "hsde={hsde}");
        assert_eq!(on.status, off.status, "hsde={hsde}");
        assert!(
            (on.obj - off.obj).abs() <= 1e-6 * (1.0 + off.obj.abs()),
            "hsde={hsde}"
        );
        assert!(max_abs_diff(&on.x, &off.x) <= 1e-5, "hsde={hsde}");
    }
}

/// `QpFactorization` promises that bound *values* may change between solves
/// on one handle (gh#1243 review). Which rows the elimination folds depends
/// on those values (a fixed or sub-`NARROW_BOX` box keeps its rows), so a
/// handle that eliminated would change its factored pattern when a bound
/// was fixed or released, and answer `NumericalFailure`. Both directions
/// are exercised: built on a wide box then solved with it fixed, and built
/// fixed then solved wide. Each is held to what a fresh one-shot solve of
/// the same instance returns.
#[test]
fn a_reused_factorization_accepts_a_bound_being_fixed_or_released() {
    use pounce_convex::QpFactorization;
    // min ½‖x − (0.7, 0.2)‖² over a box, with x₀ + x₁ ≤ 1.5.
    let base = |lb1: f64, ub1: f64| QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-0.7, -0.2],
        a: Vec::new(),
        b: Vec::new(),
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![1.5],
        lb: vec![0.0, lb1],
        ub: vec![1.0, ub1],
    };
    let wide = base(0.0, 1.0);
    let fixed = base(0.3, 0.3);
    for (built_on, then) in [(&wide, &fixed), (&fixed, &wide)] {
        for elim in [true, false] {
            let o = opts(false, elim);
            let mut handle = QpFactorization::build(built_on, &o, backend).expect("builds");
            for prob in [built_on, then] {
                let reused = handle.solve(prob);
                let fresh = solve_qp_ipm(prob, &o, backend);
                let what = format!(
                    "built on x1 ∈ [{}, {}], solving x1 ∈ [{}, {}], kkt_row_elim={elim}",
                    built_on.lb[1], built_on.ub[1], prob.lb[1], prob.ub[1]
                );
                assert_eq!(reused.status, QpStatus::Optimal, "{what}");
                assert_eq!(fresh.status, QpStatus::Optimal, "{what}: fresh");
                assert!(
                    max_abs_diff(&reused.x, &fresh.x) <= 1e-6,
                    "{what}: {:?} vs {:?}",
                    reused.x,
                    fresh.x
                );
            }
        }
    }
}
