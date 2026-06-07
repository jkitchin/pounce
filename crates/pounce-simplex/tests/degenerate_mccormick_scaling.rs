//! Regression: a relaxation LP carrying a *collapsed* McCormick coefficient
//! (numerical noise ~1e-44) must still solve to the true LP optimum.
//!
//! This is the exact `min x0` LP captured from a `pounce-global` simplex-OBBT
//! run on the quartic `x⁴ − 3x²` over the child box `[-2, ~0]`. The box upper
//! bound is a tiny positive `1.9e-15` (a branch artifact), which makes one
//! McCormick secant slope collapse to `2.2e-44`. Geometric-mean equilibration
//! used to let that spurious entry drag column 0's scale up to ~3.4e10, which
//! distorted the reduced-cost tolerances enough that the simplex declared the
//! wrong vertex `x0 = -0.375` optimal instead of the true `-1.8461538`. OBBT
//! then tightened the box to `[-0.375, 0]`, cutting off the global minimizer
//! `x ≈ -1.2247` and certifying `-0.402` instead of the true `-2.25`.
//!
//! Ground truth (`-1.846153845719934`) is from the interior-point solver on the
//! identical polytope. The fix is `EQUILIBRATE_DROP`: entries negligible
//! relative to their row/column max are excluded from the geometric mean.

#![allow(clippy::unwrap_used, clippy::approx_constant)]
use pounce_simplex::{LpProblem, LpStatus, Simplex, Triplet};

fn lp() -> LpProblem {
    LpProblem {
        n: 18,
        m: 15,
        c: vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ],
        a: vec![
            Triplet::new(0, 2, -3.0),
            Triplet::new(0, 3, 1.0),
            Triplet::new(1, 1, -1.0),
            Triplet::new(1, 3, 1.0),
            Triplet::new(1, 4, 1.0),
            Triplet::new(2, 0, -32.0),
            Triplet::new(2, 1, -1.0),
            Triplet::new(3, 0, -13.499999999999988),
            Triplet::new(3, 1, -1.0),
            Triplet::new(4, 0, -3.9999999999999893),
            Triplet::new(4, 1, -1.0),
            Triplet::new(5, 0, -0.499999999999996),
            Triplet::new(5, 1, -1.0),
            Triplet::new(6, 0, 2.2420775429197073e-44),
            Triplet::new(6, 1, -1.0),
            Triplet::new(7, 0, 7.999999999999993),
            Triplet::new(7, 1, 1.0),
            Triplet::new(8, 0, -4.0),
            Triplet::new(8, 2, -1.0),
            Triplet::new(9, 0, -2.999999999999999),
            Triplet::new(9, 2, -1.0),
            Triplet::new(10, 0, -1.9999999999999982),
            Triplet::new(10, 2, -1.0),
            Triplet::new(11, 0, -0.9999999999999973),
            Triplet::new(11, 2, -1.0),
            Triplet::new(12, 0, 3.552713678800501e-15),
            Triplet::new(12, 2, -1.0),
            Triplet::new(13, 0, 1.9999999999999982),
            Triplet::new(13, 2, 1.0),
            Triplet::new(14, 4, 1.0),
            Triplet::new(2, 5, 1.0),
            Triplet::new(3, 6, 1.0),
            Triplet::new(4, 7, 1.0),
            Triplet::new(5, 8, 1.0),
            Triplet::new(6, 9, 1.0),
            Triplet::new(7, 10, 1.0),
            Triplet::new(8, 11, 1.0),
            Triplet::new(9, 12, 1.0),
            Triplet::new(10, 13, 1.0),
            Triplet::new(11, 14, 1.0),
            Triplet::new(12, 15, 1.0),
            Triplet::new(13, 16, 1.0),
            Triplet::new(14, 17, 1.0),
        ],
        b: vec![
            0.0,
            0.0,
            48.0,
            15.187499999999982,
            2.9999999999999893,
            0.187499999999998,
            2.987047333373348e-59,
            1.4210854715202004e-14,
            4.0,
            2.2499999999999987,
            0.9999999999999982,
            0.24999999999999867,
            3.1554436208840472e-30,
            3.552713678800501e-15,
            -5.273616763440499e-28,
        ],
        lb: vec![
            -2.0,
            0.0,
            0.0,
            -5e-324,
            -12.000000000000007,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        ub: vec![
            1.9464907830075622e-15,
            16.000000000000004,
            4.000000000000001,
            12.000000000000005,
            16.000000000000007,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
        ],
    }
}

/// True LP optimum of `min x0` over this polytope, from the interior-point
/// solver. The scaling bug returned `-0.375`.
const TRUE_MIN_X0: f64 = -1.846153845719934;

#[test]
fn cold_min_x0_reaches_true_optimum() {
    // Cold solve (fresh solver), exactly as OBBT's IPM-fallback comparison runs.
    let mut p = lp();
    p.c.iter_mut().for_each(|v| *v = 0.0);
    p.c[0] = 1.0;
    let s = Simplex::new(&p).solve();
    assert_eq!(s.status, LpStatus::Optimal);
    assert!(
        (s.obj - TRUE_MIN_X0).abs() < 1e-5,
        "cold min x0 = {} (want {TRUE_MIN_X0}; the scaling bug returned -0.375)",
        s.obj
    );
}

#[test]
fn warm_sweep_min_x0_reaches_true_optimum() {
    // Warm path, as OBBT drives it: prime once, then flip the objective to
    // min x0 from the primed basis. This is the path that actually mis-solved.
    let lp = lp();
    let mut warm = Simplex::new(&lp);
    assert_eq!(warm.solve().status, LpStatus::Optimal, "prime solve");
    let mut c = vec![0.0; lp.n];
    c[0] = 1.0;
    let smin = warm.solve_objective(&c);
    assert_eq!(smin.status, LpStatus::Optimal);
    assert!(
        (smin.obj - TRUE_MIN_X0).abs() < 1e-5,
        "warm min x0 = {} (want {TRUE_MIN_X0})",
        smin.obj
    );
}
