//! End-to-end SOCP validation (Phase 2b of the SOCP extension).
//!
//! There's no external reference here: correctness is **intrinsic**. The
//! IPM only reports `Optimal` when the *unregularized* KKT residual
//! (stationarity, `Ax=b`, `s=h−Gx`, `μ=⟨s,z⟩/2 → 0`) is below tolerance,
//! with `s,z` kept inside the cone by the fraction-to-boundary step — so a
//! convergent solve is a verified KKT point. We additionally check the
//! recovered solution against the SOCP KKT conditions and, where the
//! optimum is known in closed form, the primal.

use pounce_convex::{
    ConeSpec, QpOptions, QpProblem, QpStatus, QpWarmStart, Triplet, solve_socp_ipm,
    solve_socp_ipm_warm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve(prob: &QpProblem, cones: &[ConeSpec]) -> pounce_convex::QpSolution {
    let mut opts = QpOptions::default();
    opts.max_iter = 100;
    solve_socp_ipm(prob, cones, &opts, backend)
}

/// In-cone test for a second-order cone block: `u₀ ≥ ‖u_{1..}‖`.
fn in_soc(u: &[f64], tol: f64) -> bool {
    let tail: f64 = u[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
    u[0] + tol >= tail
}

/// Assert the SOCP KKT conditions for a single SOC inequality block (the
/// whole `m_ineq` is one cone here): `s = h−Gx ∈ K`, `z ∈ K`, `sᵀz ≈ 0`,
/// `Ax=b`, and stationarity `Px+c+Aᵀy+Gᵀz = 0`.
fn assert_socp_kkt(prob: &QpProblem, sol: &pounce_convex::QpSolution, tol: f64) {
    let n = prob.n;
    let mi = prob.m_ineq();
    // s = h − Gx.
    let mut gx = vec![0.0; mi];
    prob.g_mul(&sol.x, &mut gx);
    let s: Vec<f64> = (0..mi).map(|i| prob.h[i] - gx[i]).collect();
    assert!(in_soc(&s, tol), "s = h−Gx not in cone: {s:?}");
    assert!(in_soc(&sol.z, tol), "z not in cone: {:?}", sol.z);
    let sz: f64 = s.iter().zip(&sol.z).map(|(a, b)| a * b).sum();
    assert!(sz.abs() < tol, "complementarity sᵀz = {sz}");
    // Ax = b.
    let mut ax = vec![0.0; prob.m_eq()];
    prob.a_mul(&sol.x, &mut ax);
    for (i, (&axi, &bi)) in ax.iter().zip(&prob.b).enumerate() {
        assert!((axi - bi).abs() < tol, "Ax=b row {i}: {axi} vs {bi}");
    }
    // Stationarity Px + c + Aᵀy + Gᵀz = 0.
    let mut g = prob.c.clone();
    prob.p_mul(&sol.x, &mut g);
    prob.at_mul(&sol.y, &mut g);
    prob.gt_mul(&sol.z, &mut g);
    for i in 0..n {
        assert!(g[i].abs() < tol, "stationarity[{i}] = {}", g[i]);
    }
}

/// min t  s.t.  t ≥ ‖x − x*‖  (i.e. minimize the norm to a point), encoded
/// with one second-order cone. Optimum: t* = 0, x = x*. We add the cone
/// rows `(t; x − x*) ∈ K` as `h − G·[t,x] ∈ K`.
#[test]
fn min_norm_to_point_socp() {
    // vars: [t, x0, x1]. Cone: (t, x0 − a, x1 − b) ∈ SOC(3).
    // s = h − G v ∈ K means: s0 = t, s1 = x0 − a, s2 = x1 − b.
    // So G v = (−t, −x0, −x1) and h = (0, −a, −b) ⇒ s = (t, x0−a, x1−b).
    let (a, b) = (2.0, -1.0);
    let prob = QpProblem {
        n: 3,
        p_lower: vec![], // LP objective: minimize t
        c: vec![1.0, 0.0, 0.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, -1.0),
            Triplet::new(1, 1, -1.0),
            Triplet::new(2, 2, -1.0),
        ],
        h: vec![0.0, -a, -b],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob, &[ConeSpec::SecondOrder(3)]);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    // t* = 0, x = (a, b).
    assert!(sol.x[0].abs() < 1e-6, "t={}", sol.x[0]);
    assert!((sol.x[1] - a).abs() < 1e-6, "x0={}", sol.x[1]);
    assert!((sol.x[2] - b).abs() < 1e-6, "x1={}", sol.x[2]);
    assert_socp_kkt(&prob, &sol, 1e-6);
}

/// Minimize a linear cost over a second-order cone with an equality:
/// min −x1  s.t.  x0 = 1,  (x0, x1, x2) ∈ SOC(3).
/// With x0 = 1, the cone is ‖(x1,x2)‖ ≤ 1; minimizing −x1 ⇒ x1 = 1, x2 = 0.
#[test]
fn linear_over_soc_with_equality() {
    // vars [x0, x1, x2]; cone (x0,x1,x2) ∈ K ⇒ s = G·(−I)·x ... encode
    // s = x directly: h = 0, G = −I ⇒ s = −Gx = x. Equality x0 = 1.
    let prob = QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![0.0, -1.0, 0.0],
        a: vec![Triplet::new(0, 0, 1.0)],
        b: vec![1.0],
        g: vec![
            Triplet::new(0, 0, -1.0),
            Triplet::new(1, 1, -1.0),
            Triplet::new(2, 2, -1.0),
        ],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob, &[ConeSpec::SecondOrder(3)]);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);
    assert!(sol.x[2].abs() < 1e-6, "x2={}", sol.x[2]);
    assert_socp_kkt(&prob, &sol, 1e-6);
}

/// A convex-QP objective over a second-order cone: project a point onto
/// the cone. min ½‖x − p‖² s.t. x ∈ SOC(3), with p outside the cone.
#[test]
fn projection_onto_soc_qp() {
    // P = I, c = −p ⇒ ½‖x‖² − pᵀx = ½‖x−p‖² − const. x ∈ K via s = x.
    let p = [1.0, 2.0, 0.0]; // ‖(2,0)‖ = 2 > 1 ⇒ p outside the cone
    let prob = QpProblem {
        n: 3,
        p_lower: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 2, 1.0),
        ],
        c: vec![-p[0], -p[1], -p[2]],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, -1.0),
            Triplet::new(1, 1, -1.0),
            Triplet::new(2, 2, -1.0),
        ],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob, &[ConeSpec::SecondOrder(3)]);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    // The Euclidean projection of (1,2,0) onto the SOC has the closed form
    // for a point with t < ‖x₁‖: scale = (‖x₁‖+t)/(2‖x₁‖); proj =
    // scale·(‖x₁‖, x₁). Here t=1, ‖x₁‖=2 ⇒ scale = 3/4 ⇒ (1.5, 1.5, 0).
    assert!((sol.x[0] - 1.5).abs() < 1e-5, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 1.5).abs() < 1e-5, "x1={}", sol.x[1]);
    assert!(sol.x[2].abs() < 1e-5, "x2={}", sol.x[2]);
    assert_socp_kkt(&prob, &sol, 1e-6);
}

/// SOC warm start: from a nearby SOCP's solution, the warm solve reaches
/// the same KKT point (the projection onto the cone) and takes no more
/// iterations than cold. Exercises the SOC `λ_min` recentering.
#[test]
fn soc_warm_start_matches_cold() {
    let base = QpProblem {
        n: 3,
        p_lower: (0..3).map(|i| Triplet::new(i, i, 1.0)).collect(),
        c: vec![-1.0, -2.0, 0.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, -1.0),
            Triplet::new(1, 1, -1.0),
            Triplet::new(2, 2, -1.0),
        ],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let cones = [ConeSpec::SecondOrder(3)];
    let opts = QpOptions::default();
    let base_sol = solve_socp_ipm(&base, &cones, &opts, backend);
    assert_eq!(base_sol.status, QpStatus::Optimal);

    // Perturb the target slightly.
    let mut pert = base.clone();
    pert.c = vec![-1.1, -1.9, 0.05];
    let cold = solve_socp_ipm(&pert, &cones, &opts, backend);
    let warm = solve_socp_ipm_warm(
        &pert,
        &cones,
        &QpWarmStart::from_solution(&base_sol),
        &opts,
        backend,
    );
    assert_eq!(warm.status, QpStatus::Optimal);
    for i in 0..3 {
        assert!(
            (cold.x[i] - warm.x[i]).abs() < 1e-6,
            "x[{i}]: cold={} warm={}",
            cold.x[i],
            warm.x[i]
        );
    }
    assert_socp_kkt(&pert, &warm, 1e-6);
    // SOC warm restarts the duals centered (stable), so the win is from
    // the primal proximity; it must not regress vs cold.
    assert!(
        warm.iters <= cold.iters,
        "warm {} cold {}",
        warm.iters,
        cold.iters
    );
}

/// A larger second-order cone (dim 12) — exercises the sparse
/// diagonal-plus-rank-1 KKT representation (one auxiliary variable carries
/// the rank-1 update; the `(z,z)` block stays diagonal instead of dense).
/// Projection of a point outside the cone has a known closed form.
#[test]
fn larger_soc_projection_sparse_kkt() {
    let m = 12;
    // p = (t, x₁) with t < ‖x₁‖ ⇒ outside the cone. Project:
    // scale = (‖x₁‖+t)/(2‖x₁‖); proj = scale·(‖x₁‖, x₁).
    let mut p = vec![1.0; m];
    p[0] = 1.0; // t
    let nx: f64 = p[1..].iter().map(|v| v * v).sum::<f64>().sqrt(); // ‖x₁‖
    let scale = (nx + p[0]) / (2.0 * nx);
    let mut expect = vec![0.0; m];
    expect[0] = scale * nx;
    for k in 1..m {
        expect[k] = scale * p[k];
    }

    let prob = QpProblem {
        n: m,
        p_lower: (0..m).map(|i| Triplet::new(i, i, 1.0)).collect(),
        c: p.iter().map(|v| -v).collect(),
        a: vec![],
        b: vec![],
        g: (0..m).map(|i| Triplet::new(i, i, -1.0)).collect(),
        h: vec![0.0; m],
        lb: vec![],
        ub: vec![],
    };
    let opts = QpOptions::default();
    let sol = solve_socp_ipm(&prob, &[ConeSpec::SecondOrder(m)], &opts, backend);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    for k in 0..m {
        assert!(
            (sol.x[k] - expect[k]).abs() < 1e-5,
            "x[{k}]={} want {}",
            sol.x[k],
            expect[k]
        );
    }
    assert_socp_kkt(&prob, &sol, 1e-6);
}

/// Mixed cone: a nonnegative-orthant block and a second-order block in one
/// problem (exercises the composite KKT assembly with both shapes).
/// min −x0 − x1  s.t.  x0 ≤ 1 (orthant),  (1, x1) ∈ SOC(2) ⇒ |x1| ≤ 1.
#[test]
fn mixed_orthant_and_soc() {
    // rows: [orthant] 1 − x0 ≥ 0 ; [soc dim 2] s = (1, x1) with s0=1≥|x1|.
    // s_orth = h0 − G0·x = 1 − x0 (need ≥ 0).
    // s_soc = (h1 − G1 x, h2 − G2 x) = (1, x1): row1 = 0·x + h=1, row2 = −x1+0.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![-1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),  // orthant: 1 − x0 ≥ 0
            Triplet::new(2, 1, -1.0), // soc row 2: s2 = h2 − (−x1) = x1
        ],
        h: vec![1.0, 1.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob, &[ConeSpec::Nonneg(1), ConeSpec::SecondOrder(2)]);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    // max x0 + x1 with x0 ≤ 1, |x1| ≤ 1 ⇒ x0 = 1, x1 = 1.
    assert!((sol.x[0] - 1.0).abs() < 1e-5, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-5, "x1={}", sol.x[1]);
}
