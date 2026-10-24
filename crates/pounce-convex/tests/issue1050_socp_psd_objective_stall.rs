//! gh #1050 — a 4-variable, non-degenerate SOCP with a PSD (rank-3) quadratic
//! objective ran `solve_socp` to its 200-iteration cap (`kkt_error 3e-2`,
//! objective off by `1.4e-3` relative) where Clarabel stops in 7.
//!
//! At the optimum the first SOC is active with a strictly complementary
//! boundary pair (`z_t = 1.26`) and the second is inactive — nothing is
//! degenerate. The trace showed the HSDE loop reaching `μ ≈ 1e-7` by iteration
//! 10 and then locking into a three-step cycle of short steps (α ≈ 0.3, 0.03,
//! 0.6) with large `Δτ > 0`, scaling the whole iterate up along the
//! homogeneous ray and growing `μ` by four orders of magnitude per ten
//! iterations. The direct driver failed the same model with
//! `NumericalFailure` at iteration 14.
//!
//! Cause: the SOC Mehrotra second-order term was formed in **unscaled**
//! space, `Δs_aff∘Δz_aff`. That product is not invariant under the cone's
//! automorphisms, so next to an active boundary (`det s, det z → 0`) it is
//! badly mis-sized and pushes the corrector off the central path. The fix forms
//! it in the NT-scaled space, `(W⁻¹Δs_aff)∘(WΔz_aff)`, as Clarabel, ECOS and
//! CVXOPT do (`SecondOrderCone::comp_residual_corrector`). The predictor and
//! centering terms are identical in both spaces (`W λ⁻¹ = z⁻¹`); only the
//! second-order term moved.
//!
//! Measured on a 600-instance family of this shape (integer rank-3 Gram `P`
//! over 4 variables, dyadic data, one SOC(3) plus an `‖x‖ ≤ 4` ball), default
//! options: before, median 13 iterations, p95 34, **8 not `Optimal`**
//! (7 `IterationLimit`, 1 `OptimalInaccurate`); after, median 9, p95 10, max
//! 13, all `Optimal`. The same family at `n = 6` with rank-2 `P` went from
//! median 14 / max 34 to median 9 / max 18. `family_solves_within_a_bounded_count`
//! pins the first family.

use pounce_convex::{ConeSpec, QpOptions, QpProblem, QpStatus, Triplet, solve_socp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Clarabel at default settings: `Solved`, 7 iterations.
const CLARABEL_OBJ: f64 = -6.352_690_532_4;

/// The issue's model, verbatim. `P = L Lᵀ` with integer `L` (exactly PSD,
/// rank 3); all other data dyadic, so the stored problem is exact.
fn issue_problem() -> QpProblem {
    let p = [
        [9.0, 2.0, -6.0, -3.0],
        [2.0, 1.0, 0.0, -2.0],
        [-6.0, 0.0, 8.0, 0.0],
        [-3.0, -2.0, 0.0, 6.0],
    ];
    let mut p_lower = Vec::new();
    for (i, row) in p.iter().enumerate() {
        for (j, &v) in row.iter().enumerate().take(i + 1) {
            if v != 0.0 {
                p_lower.push(Triplet::new(i, j, v));
            }
        }
    }
    // SOC(3): ‖(1, −0.5) − F x‖ ≤ 5 (active); SOC(5): ‖x‖ ≤ 4 (inactive).
    let mut g = Vec::new();
    for (r, row) in [(1, [1.0, 1.0, 0.75, -0.25]), (2, [-1.25, 1.0, -2.0, -0.75])] {
        for (j, &v) in row.iter().enumerate() {
            g.push(Triplet::new(r, j, v));
        }
    }
    for j in 0..4 {
        g.push(Triplet::new(4 + j, j, -1.0));
    }
    QpProblem {
        n: 4,
        p_lower,
        c: vec![-0.875, 1.625, -2.125, -1.0],
        a: vec![],
        b: vec![],
        g,
        h: vec![5.0, 1.0, -0.5, 4.0, 0.0, 0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    }
}

const CONES: [ConeSpec; 2] = [ConeSpec::SecondOrder(3), ConeSpec::SecondOrder(5)];

fn soc_gap(u: &[f64]) -> f64 {
    u[0] - u[1..].iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn check_optimum(prob: &QpProblem, sol: &pounce_convex::QpSolution, label: &str) {
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "{label}: iters={}",
        sol.iters
    );
    assert!(
        (sol.obj - CLARABEL_OBJ).abs() < 1e-7,
        "{label}: obj {} vs Clarabel {CLARABEL_OBJ}",
        sol.obj
    );
    // Stationarity Px + c + Gᵀz = 0.
    let mut g = prob.c.clone();
    prob.p_mul(&sol.x, &mut g);
    prob.gt_mul(&sol.z, &mut g);
    for (i, gi) in g.iter().enumerate() {
        assert!(gi.abs() < 1e-6, "{label}: stationarity[{i}] = {gi}");
    }
    // s = h − Gx in both cones; cone 1 active, cone 2 inactive.
    let mut gx = vec![0.0; prob.m_ineq()];
    prob.g_mul(&sol.x, &mut gx);
    let s: Vec<f64> = prob.h.iter().zip(&gx).map(|(h, v)| h - v).collect();
    assert!(
        soc_gap(&s[..3]).abs() < 1e-6,
        "{label}: cone 1 slack {:?}",
        &s[..3]
    );
    assert!(
        soc_gap(&s[3..]) > 0.5,
        "{label}: cone 2 slack {:?}",
        &s[3..]
    );
    assert!(
        (sol.z[0] - 1.26).abs() < 1e-2,
        "{label}: z_t = {}",
        sol.z[0]
    );
    assert!(
        sol.z[3].abs() < 1e-6,
        "{label}: inactive cone dual {}",
        sol.z[3]
    );
}

/// The reported solve: default options (HSDE), 200-iteration cap. Was
/// `IterationLimit` at 200; now `Optimal` in 9.
#[test]
fn default_solve_reaches_the_optimum_in_a_few_iterations() {
    let prob = issue_problem();
    let sol = solve_socp_ipm(&prob, &CONES, &QpOptions::default(), backend);
    check_optimum(&prob, &sol, "hsde");
    assert!(sol.iters <= 20, "hsde took {} iterations", sol.iters);
}

/// The direct (non-HSDE) driver shares the cone's corrector. Was
/// `NumericalFailure` at iteration 14; now `Optimal` in 11.
#[test]
fn direct_driver_reaches_the_optimum_too() {
    let prob = issue_problem();
    let opts = QpOptions {
        use_hsde: false,
        ..QpOptions::default()
    };
    let sol = solve_socp_ipm(&prob, &CONES, &opts, backend);
    check_optimum(&prob, &sol, "direct");
    assert!(sol.iters <= 20, "direct took {} iterations", sol.iters);
}

/// Deterministic 64-bit LCG — the family must not depend on a crate's RNG.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn int(&mut self, lo: i64, hi: i64) -> f64 {
        (lo + (self.unit() * (hi - lo + 1) as f64) as i64) as f64
    }
    /// A dyadic rational in [−2, 2] with step 1/8.
    fn dyadic(&mut self) -> f64 {
        self.int(-16, 16) / 8.0
    }
}

/// The issue's family: rank-deficient integer Gram `P`, one SOC(3) row block
/// plus a norm ball. Before the fix 8 of these 600 ended short of `Optimal`
/// (7 at the iteration cap) and the median was 13; after, all solve, median 9,
/// max 13. The bound below leaves room for honest drift while failing on any
/// return of the stall.
#[test]
fn family_solves_within_a_bounded_count() {
    let (n, rank, count) = (4usize, 3usize, 600usize);
    let mut rng = Lcg(12345);
    let mut iters = Vec::with_capacity(count);
    for case in 0..count {
        let l: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..rank).map(|_| rng.int(-2, 2)).collect())
            .collect();
        let mut p_lower = Vec::new();
        for i in 0..n {
            for j in 0..=i {
                let v: f64 = (0..rank).map(|t| l[i][t] * l[j][t]).sum();
                if v != 0.0 {
                    p_lower.push(Triplet::new(i, j, v));
                }
            }
        }
        let c: Vec<f64> = (0..n).map(|_| rng.dyadic()).collect();
        let d = 3;
        let mut g = Vec::new();
        let mut h = vec![0.0; d + n + 1];
        h[0] = 5.0;
        for r in 1..d {
            h[r] = rng.dyadic();
            for j in 0..n {
                g.push(Triplet::new(r, j, rng.dyadic()));
            }
        }
        h[d] = 4.0;
        for j in 0..n {
            g.push(Triplet::new(d + 1 + j, j, -1.0));
        }
        let prob = QpProblem {
            n,
            p_lower,
            c,
            a: vec![],
            b: vec![],
            g,
            h,
            lb: vec![],
            ub: vec![],
        };
        let cones = [ConeSpec::SecondOrder(d), ConeSpec::SecondOrder(n + 1)];
        let sol = solve_socp_ipm(&prob, &cones, &QpOptions::default(), backend);
        assert_eq!(
            sol.status,
            QpStatus::Optimal,
            "case {case}: iters={}",
            sol.iters
        );
        iters.push(sol.iters);
    }
    iters.sort_unstable();
    let median = iters[count / 2];
    let max = iters[count - 1];
    assert!(median <= 12, "median {median} iterations");
    assert!(max <= 25, "max {max} iterations");
}
