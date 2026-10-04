//! gh #988 — a warm start that cannot help must degrade to the cold path.
//!
//! The warm leg runs the direct infeasible-start method, which has no
//! infeasibility certificate: on an infeasible neighbour it ends in
//! `NumericalFailure` / `IterationLimit`, where the cold HSDE solve certifies
//! `PrimalInfeasible`. Both tests assert the *status* the cold solve gives,
//! and that a warm start never changes the answer.

use pounce_convex::{
    QpOptions, QpProblem, QpStatus, QpWarmStart, Triplet, solve_qp_ipm, solve_qp_ipm_warm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Discretised CSTR (issue's `mpc14.py`, constants baked in): state `(CA, T)`,
/// input `Tc`. `Ad`/`Bd` are the zero-order-hold matrices at `dt = 0.05`.
const AD: [[f64; 2]; 2] = [
    [0.8954074706996391, -0.0018975850269320378],
    [11.117729114889276, 1.234393572893381],
];
const BD: [f64; 2] = [-9.725211755444644e-05, 0.11658724898071166];

/// The issue's linear MPC over `n_steps` steps. Variables are
/// `[x_0, x_1, .., x_N, u_0, .., u_{N-1}]` (`x_k` is 2-wide). The initial
/// temperature deviation `t0` is the row that makes a neighbour infeasible:
/// the state box `T in [-50, 5]` and the move limit cannot absorb a large one.
fn mpc(n_steps: usize, t0: f64) -> QpProblem {
    let (nn, nx) = (n_steps, 2 * (n_steps + 1));
    let n = 3 * nn + 2;
    let (qx, r, umax, du) = ([100.0, 1.0], 0.1, 10.0, 1.0);
    let (lo, hi) = ([-0.5, -50.0], [0.5, 5.0]);
    let mut p_lower = Vec::new();
    for k in 0..nn {
        p_lower.push(Triplet::new(2 + 2 * k, 2 + 2 * k, 2.0 * qx[0]));
        p_lower.push(Triplet::new(3 + 2 * k, 3 + 2 * k, 2.0 * qx[1]));
        p_lower.push(Triplet::new(nx + k, nx + k, 2.0 * r));
    }
    let mut a = vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)];
    let mut b = vec![-0.02, t0];
    for k in 0..nn {
        for i in 0..2 {
            let row = 2 + 2 * k + i;
            a.push(Triplet::new(row, 2 * (k + 1) + i, 1.0));
            for j in 0..2 {
                a.push(Triplet::new(row, 2 * k + j, -AD[i][j]));
            }
            a.push(Triplet::new(row, nx + k, -BD[i]));
            b.push(0.0);
        }
    }
    let (mut g, mut h) = (Vec::new(), Vec::new());
    let mut push = |col: usize, v: f64, rhs: f64, rows: &mut usize| {
        g.push(Triplet::new(*rows, col, v));
        h.push(rhs);
        *rows += 1;
    };
    let mut rows = 0;
    for i in 0..nx {
        push(i, 1.0, hi[i % 2], &mut rows);
    }
    for i in 0..nx {
        push(i, -1.0, -lo[i % 2], &mut rows);
    }
    for k in 0..nn {
        push(nx + k, 1.0, umax, &mut rows);
    }
    for k in 0..nn {
        push(nx + k, -1.0, umax, &mut rows);
    }
    // Move limits |u_k - u_{k-1}| <= du (u_{-1} = 0).
    for sign in [1.0, -1.0] {
        for k in 0..nn {
            g.push(Triplet::new(rows, nx + k, sign));
            if k > 0 {
                g.push(Triplet::new(rows, nx + k - 1, -sign));
            }
            h.push(du);
            rows += 1;
        }
    }
    QpProblem {
        n,
        p_lower,
        c: vec![0.0; n],
        a,
        b,
        g,
        h,
        lb: vec![],
        ub: vec![],
    }
}

fn warm_vs_cold(n_steps: usize) -> Vec<(f64, QpStatus, QpStatus, usize, usize)> {
    let opts = QpOptions::default();
    let base = solve_qp_ipm(&mpc(n_steps, 3.0), &opts, backend);
    assert_eq!(base.status, QpStatus::Optimal);
    let warm = QpWarmStart::from_solution(&base);
    [3.5, -4.0]
        .iter()
        .map(|&t| {
            let nb = mpc(n_steps, t);
            let cold = solve_qp_ipm(&nb, &opts, backend);
            let w = solve_qp_ipm_warm(&nb, &opts, &warm, backend);
            eprintln!(
                "N={n_steps} t0={t} cold {:?} {} warm {:?} {}",
                cold.status, cold.iters, w.status, w.iters
            );
            (t, cold.status, w.status, cold.iters, w.iters)
        })
        .collect()
}

#[test]
fn infeasible_neighbour_warm_start_matches_cold_verdict() {
    for n_steps in [20, 60, 200] {
        for (t, cold, warm, cold_it, warm_it) in warm_vs_cold(n_steps) {
            if t == 3.5 {
                assert_eq!(cold, QpStatus::PrimalInfeasible, "N={n_steps} t0={t} cold");
            }
            assert_eq!(warm, cold, "N={n_steps} t0={t}: warm must not lose to cold");
            // gh #988 (review): `iters` counts both legs. The stalled warm leg
            // is handed to the cold solve within the stall detector's window
            // (measured: at most 38 iterations over cold, at N=200 t0=-4); it
            // used to run to `max_iter` first (133..219 against 14..19).
            assert!(
                warm_it <= cold_it + 45,
                "N={n_steps} t0={t}: warm {warm_it} vs cold {cold_it}"
            );
        }
    }
}

/// A warm point sitting off a pinned column (`lb == ub`) must not stop the
/// warm solve reaching the cold optimum.
#[test]
fn warm_point_off_fixed_column_still_solves() {
    let opts = QpOptions::default();
    let n_steps = 40;
    let mut prob = mpc(n_steps, 3.0);
    prob.lb = vec![-1e30; prob.n];
    prob.ub = vec![1e30; prob.n];
    let base = solve_qp_ipm(&prob, &opts, backend);
    assert_eq!(base.status, QpStatus::Optimal);
    let nx = 2 * (n_steps + 1);
    let pins: Vec<f64> = (0..5).map(|k| 0.9 * base.x[nx + k]).collect();
    for (k, &v) in pins.iter().enumerate() {
        prob.lb[nx + k] = v;
        prob.ub[nx + k] = v;
    }
    let cold = solve_qp_ipm(&prob, &opts, backend);
    let w = solve_qp_ipm_warm(&prob, &opts, &QpWarmStart::from_solution(&base), backend);
    assert_eq!(w.status, cold.status);
    if cold.status == QpStatus::Optimal {
        assert!((w.obj - cold.obj).abs() <= 1e-6 * (1.0 + cold.obj.abs()));
        for (k, &v) in pins.iter().enumerate() {
            assert!((w.x[nx + k] - v).abs() < 1e-8);
        }
    }
}

/// gh #988 (second pass). A pinned column is substituted out of the warm
/// solve rather than left as the row pair `x <= v`, `-x <= -v`, which has no
/// interior. The lift back to the full problem must hand the caller a KKT
/// point of the *full* problem: pins restored, the pinned column's bound
/// multiplier recovered from stationarity, the multipliers of rows that were
/// all-pinned (dropped as constants) zero, and the objective including what
/// the pin contributes.
///
/// ```text
/// min  x0 + 2 x1 + 3 x2 + 10     s.t.  x0 + x1 + x2 = 4        (equality)
///                                      x1 <= 3                  (inequality)
///                                      2 x0 <= 5                (constant once x0 is pinned)
///      x0 = 1 (lb == ub),  0 <= x1, x2 <= 10
/// ```
#[test]
fn pinned_column_is_eliminated_and_the_lift_is_a_kkt_point() {
    let prob = QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![1.0, 2.0, 3.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0),
        ],
        b: vec![4.0],
        g: vec![Triplet::new(0, 1, 1.0), Triplet::new(1, 0, 2.0)],
        h: vec![3.0, 5.0],
        lb: vec![1.0, 0.0, 0.0],
        ub: vec![1.0, 10.0, 10.0],
    };
    let opts = QpOptions {
        obj_constant: 10.0,
        ..QpOptions::default()
    };
    // A warm point well off the pin and off the optimum.
    let warm = QpWarmStart {
        x: vec![2.5, 1.0, 0.5],
        y: vec![0.5],
        z: vec![0.1, 0.1],
        z_lb: vec![0.1; 3],
        z_ub: vec![0.1; 3],
    };
    let sol = solve_qp_ipm_warm(&prob, &opts, &warm, backend);
    assert_eq!(sol.status, QpStatus::Optimal, "{sol:?}");
    // The pin is restored *bit for bit*: only the elimination writes it back
    // exactly; the full-problem path (pins as a bound pair inside the
    // interior-point iteration) lands within tolerance of it, never on it.
    // So this is the evidence the reduction ran and its answer was returned.
    assert_eq!(sol.x[0], 1.0, "pin restored exactly: {:?}", sol.x);
    // And it ran on its own: a 2-column LP, no cold leg behind it.
    assert!(sol.iters <= 20, "iters = {}", sol.iters);
    assert!(
        (sol.x[1] - 3.0).abs() < 1e-6 && sol.x[2].abs() < 1e-6,
        "{:?}",
        sol.x
    );
    // 1 + 2*3 + 10: the pin's cost and the constant are in the objective.
    assert!((sol.obj - 17.0).abs() < 1e-6, "obj = {}", sol.obj);
    assert_eq!(sol.y.len(), 1);
    assert_eq!(sol.z.len(), 2);
    assert!(
        sol.z[1].abs() < 1e-12,
        "dropped constant row has a zero multiplier"
    );
    let res = sol.kkt_residuals(&prob);
    assert!(
        res.dual_infeasibility < 1e-6 && res.primal_infeasibility < 1e-6,
        "the lifted point must satisfy the FULL problem's KKT conditions: {res:?}"
    );
}

/// gh #988 (review), item 1. Two contradictory equality rows `x0 + x1 = 1`
/// and `x0 + x1 = 1 + gap`, beside an unrelated row `x2 <= big`. The second
/// pass's plateau exit, and the global scale-relative stop before it, read the
/// primal residual against `1 + max(‖b‖, ‖h‖, ‖s‖)`, so the one large
/// right-hand side excused the contradiction in the other two rows and the
/// direct driver returned `Optimal` at `|Ax - b| = gap/2` (up to `0.25`).
fn near_feasible_infeasible(gap: f64, big: f64) -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 2, 1.0),
        ],
        c: vec![1.0, 2.0, -1.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 1.0),
        ],
        b: vec![1.0, 1.0 + gap],
        g: vec![Triplet::new(0, 2, 1.0)],
        h: vec![big],
        lb: vec![],
        ub: vec![],
    }
}

const NEAR_FEASIBLE_CASES: [(f64, f64); 4] = [(1e-4, 1e4), (1e-3, 1e6), (0.5, 1e8), (1e-6, 1e9)];

#[test]
fn near_feasible_infeasible_is_never_optimal_from_a_warm_start() {
    let opts = QpOptions::default();
    for (gap, big) in NEAR_FEASIBLE_CASES {
        let prob = near_feasible_infeasible(gap, big);
        let cold = solve_qp_ipm(&prob, &opts, backend);
        assert_eq!(
            cold.status,
            QpStatus::PrimalInfeasible,
            "gap={gap} big={big} cold"
        );
        // Warm from the feasible neighbour (`gap = 0`).
        let base = solve_qp_ipm(&near_feasible_infeasible(0.0, big), &opts, backend);
        assert_eq!(base.status, QpStatus::Optimal);
        let w = solve_qp_ipm_warm(&prob, &opts, &QpWarmStart::from_solution(&base), backend);
        eprintln!(
            "gap={gap} big={big} warm {:?} {} cold {}",
            w.status, w.iters, cold.iters
        );
        assert_eq!(
            w.status,
            QpStatus::PrimalInfeasible,
            "gap={gap} big={big}: warm must reach the cold verdict"
        );
        // One short warm leg (the stall window) plus the cold leg: measured
        // 10-11 iterations over cold.
        assert!(
            w.iters <= cold.iters + 25,
            "{} vs cold {}",
            w.iters,
            cold.iters
        );
    }
}

#[test]
fn near_feasible_infeasible_is_never_optimal_on_the_direct_driver() {
    let opts = QpOptions {
        use_hsde: false,
        ..QpOptions::default()
    };
    for (gap, big) in NEAR_FEASIBLE_CASES {
        let prob = near_feasible_infeasible(gap, big);
        let d = solve_qp_ipm(&prob, &opts, backend);
        eprintln!("gap={gap} big={big} direct {:?} {}", d.status, d.iters);
        assert_ne!(d.status, QpStatus::Optimal, "gap={gap} big={big}");
        // The direct driver has no certificate of its own for most of these
        // (its Farkas test needs `y` to outgrow the regularized drift), and its
        // final verdict classifies the last point by its true KKT error: a
        // contradiction of `1e-6` leaves `|Ax-b| = 5e-7`, which is "solved to
        // reduced accuracy" by the `1e3·tol` rule and says so. Anything
        // larger must not even be that.
        if gap > 1e-5 {
            assert_ne!(d.status, QpStatus::OptimalInaccurate, "gap={gap} big={big}");
        }
    }
}
