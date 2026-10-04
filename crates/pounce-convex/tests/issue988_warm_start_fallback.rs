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

fn warm_vs_cold(n_steps: usize) -> Vec<(f64, QpStatus, QpStatus)> {
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
            (t, cold.status, w.status)
        })
        .collect()
}

#[test]
fn infeasible_neighbour_warm_start_matches_cold_verdict() {
    for n_steps in [20, 60, 200] {
        for (t, cold, warm) in warm_vs_cold(n_steps) {
            if t == 3.5 {
                assert_eq!(cold, QpStatus::PrimalInfeasible, "N={n_steps} t0={t} cold");
            }
            assert_eq!(warm, cold, "N={n_steps} t0={t}: warm must not lose to cold");
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
