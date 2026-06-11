//! Cone-aware presolve (`presolve_conic`): the orthant/equality reductions
//! apply, second-order-cone rows are preserved, and the reduced cone
//! partition is recovered — so presolve composes with the SOCP solve and
//! the postsolved point is KKT-valid for the original problem.

use pounce_convex::presolve::{presolve_conic, PresolveOutcome};
use pounce_convex::{solve_socp_ipm, ConeSpec, QpOptions, QpProblem, QpStatus, Triplet};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn in_soc(u: &[f64], tol: f64) -> bool {
    let tail: f64 = u[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
    u[0] + tol >= tail
}

/// A mixed problem: projection onto a second-order cone for (x0,x1,x2),
/// plus an orthant bound `x3 ≤ 5` that appears **twice** (a duplicate the
/// presolve should drop) while leaving the SOC rows verbatim.
#[test]
fn conic_presolve_roundtrip_mixed() {
    // min ½‖(x0,x1,x2)‖² − pᵀ(x0,x1,x2) − x3  s.t.
    //   (x0,x1,x2) ∈ SOC(3)         [rows 0,1,2: s = −Gx = x]
    //   x3 ≤ 5                       [row 3, nonneg]
    //   x3 ≤ 5  (duplicate)          [row 4, nonneg]
    let p = [1.0, 2.0, 0.0]; // proj onto SOC = (1.5, 1.5, 0)
    let prob = QpProblem {
        n: 4,
        p_lower: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 2, 1.0),
        ],
        c: vec![-p[0], -p[1], -p[2], -1.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, -1.0),
            Triplet::new(1, 1, -1.0),
            Triplet::new(2, 2, -1.0),
            Triplet::new(3, 3, 1.0), // x3 ≤ 5
            Triplet::new(4, 3, 1.0), // x3 ≤ 5 (duplicate)
        ],
        h: vec![0.0, 0.0, 0.0, 5.0, 5.0],
        lb: vec![],
        ub: vec![],
    };
    let cones = [ConeSpec::SecondOrder(3), ConeSpec::Nonneg(2)];
    let opts = QpOptions::default();

    let ps = match presolve_conic(&prob, &cones) {
        PresolveOutcome::Reduced(ps) => ps,
        other => panic!(
            "expected Reduced, got {:?}",
            matches!(other, PresolveOutcome::Reduced(_))
        ),
    };
    // The duplicate orthant row is dropped; the SOC block survives intact.
    let rc = ps.reduced_cones(&cones);
    assert_eq!(
        rc,
        vec![ConeSpec::SecondOrder(3), ConeSpec::Nonneg(1)],
        "reduced cones {rc:?}"
    );
    assert_eq!(ps.reduced.m_ineq(), 4, "5 → 4 inequality rows");

    // Solve the reduced SOCP and postsolve to the original space.
    let red = solve_socp_ipm(&ps.reduced, &rc, &opts, backend);
    assert_eq!(red.status, QpStatus::Optimal);
    let sol = ps.postsolve(&red);

    // Primal: SOC projection + x3 = 5.
    assert!((sol.x[0] - 1.5).abs() < 1e-5, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 1.5).abs() < 1e-5, "x1={}", sol.x[1]);
    assert!(sol.x[2].abs() < 1e-5, "x2={}", sol.x[2]);
    assert!((sol.x[3] - 5.0).abs() < 1e-5, "x3={}", sol.x[3]);

    // KKT of the original: s = h − Gx, the SOC block ∈ K, z ∈ K, sᵀz ≈ 0,
    // stationarity Px + c + Gᵀz = 0.
    let mut gx = vec![0.0; prob.m_ineq()];
    prob.g_mul(&sol.x, &mut gx);
    let s: Vec<f64> = (0..prob.m_ineq()).map(|i| prob.h[i] - gx[i]).collect();
    assert!(in_soc(&s[0..3], 1e-6), "SOC slack {:?}", &s[0..3]);
    assert!(in_soc(&sol.z[0..3], 1e-6), "SOC dual {:?}", &sol.z[0..3]);
    for i in 3..prob.m_ineq() {
        assert!(s[i] > -1e-6 && sol.z[i] > -1e-6, "orthant feas row {i}");
    }
    let sz: f64 = s.iter().zip(&sol.z).map(|(a, b)| a * b).sum();
    assert!(sz.abs() < 1e-5, "complementarity {sz}");
    let mut g = prob.c.clone();
    prob.p_mul(&sol.x, &mut g);
    prob.gt_mul(&sol.z, &mut g);
    for i in 0..prob.n {
        assert!(g[i].abs() < 1e-5, "stationarity[{i}] = {}", g[i]);
    }
}

/// A pure SOCP: presolve must be a near-no-op on the cone rows (only the
/// objective/equality machinery can act), leaving the partition unchanged.
#[test]
fn conic_presolve_pure_socp_preserves_cone() {
    let prob = QpProblem {
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
    match presolve_conic(&prob, &cones) {
        PresolveOutcome::Reduced(ps) => {
            assert_eq!(ps.reduced.m_ineq(), 3, "SOC rows must all survive");
            assert_eq!(ps.reduced_cones(&cones), vec![ConeSpec::SecondOrder(3)]);
        }
        _ => panic!("expected Reduced"),
    }
}
