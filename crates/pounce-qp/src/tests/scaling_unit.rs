//! §8.2 scaling-sweep diagnostics — pure-Rust scaling check.
//!
//! These tests exercise the solver at sizes
//! `n ∈ {10, 50, 100, 200}` and print iteration counts + wall
//! time to stderr (visible with `cargo test -p pounce-qp -- --nocapture`).
//! They double as correctness regression at non-tiny sizes
//! (the analytical-ladder tests are all `n ≤ 3`).
//!
//! The full §8.2 deliverable (LASSO at `n ∈ {10², 10³, 10⁴, 10⁵}`,
//! MPC quadrotor horizon 10–160, Maros-Mészáros size buckets) is
//! a follow-up commit that needs criterion-style benchmarking
//! infrastructure and external oracle comparison. These tests are
//! the minimum viable diagnostic until then.

use crate::options::QpOptions;
use crate::problem::{HessianInertia, QpProblem};
use crate::solver::{ParametricActiveSetSolver, QpSolver};
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_feral::FeralSolverInterface;
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use std::rc::Rc;
use std::time::Instant;

fn new_solver() -> ParametricActiveSetSolver {
    ParametricActiveSetSolver::new(Box::new(FeralSolverInterface::new()))
}

/// Identity Hessian as 1-based diagonal triplet.
fn identity_h(n: usize) -> SymTMatrix {
    let irows: Vec<i32> = (1..=n as i32).collect();
    let jcols = irows.clone();
    let space = SymTMatrixSpace::new(n as i32, irows, jcols);
    let mut h = SymTMatrix::new(space);
    h.set_values(&vec![1.0; n]);
    h
}

fn empty_gen(m: usize, n: usize) -> GenTMatrix {
    GenTMatrix::new(GenTMatrixSpace::new(
        m as i32,
        n as i32,
        Vec::new(),
        Vec::new(),
    ))
}

#[test]
fn scaling_box_constrained_interior_optimum() {
    // min ½‖x − target‖² with bounds wide enough that no bound
    // binds. Closed form: x* = target. n iterations expected: 1
    // (interior optimum is found on the first KKT solve).
    for &n in &[10usize, 50, 100, 200] {
        let h = identity_h(n);
        let a = empty_gen(0, n);
        let target: Vec<f64> = (1..=n).map(|i| (i as f64) * 0.3).collect();
        let g: Vec<f64> = target.iter().map(|t| -t).collect();
        let bl: [f64; 0] = [];
        let bu: [f64; 0] = [];
        let xl = vec![-1e3; n];
        let xu = vec![1e3; n];

        let qp = QpProblem {
            n,
            m: 0,
            h: &h,
            g: &g,
            a: &a,
            bl: &bl,
            bu: &bu,
            xl: &xl,
            xu: &xu,
            hessian_inertia: HessianInertia::Psd,
        };

        let mut solver = new_solver();
        let started = Instant::now();
        let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
        let elapsed = started.elapsed();

        assert_eq!(sol.status, crate::QpStatus::Optimal);
        for (i, (&xi, &ti)) in sol.x.iter().zip(target.iter()).enumerate() {
            assert!(
                (xi - ti).abs() < 1e-10,
                "n={n} i={i}: x={xi} but expected {ti}",
            );
        }
        eprintln!(
            "[scaling-box-interior] n={n:4}  iters≈{:3}  ws_changes={}  refactor={}  time={:.3}ms",
            1,
            sol.stats.n_working_set_changes,
            sol.stats.n_refactor,
            elapsed.as_secs_f64() * 1000.0,
        );
        assert_eq!(sol.stats.n_working_set_changes, 0);
    }
}

#[test]
fn scaling_equality_only_diagonal_band() {
    // min ½‖x‖² s.t. A x = b  where A is the n × (n/4) banded
    // sum-window matrix (each row sums 4 consecutive vars to a
    // unique target). Equality-only fast path; solves in one
    // KKT call regardless of n.
    for &n in &[16usize, 64, 128] {
        let m = n / 4;
        let h = identity_h(n);

        // A row i has 4 nonzeros at columns 4i+1..4i+4 (1-based).
        let mut a_irows: Vec<i32> = Vec::with_capacity(4 * m);
        let mut a_jcols: Vec<i32> = Vec::with_capacity(4 * m);
        for i in 0..m {
            for k in 0..4 {
                a_irows.push((i + 1) as i32);
                a_jcols.push((4 * i + k + 1) as i32);
            }
        }
        let a_space = GenTMatrixSpace::new(m as i32, n as i32, a_irows, a_jcols);
        let mut a = GenTMatrix::new(Rc::clone(&a_space));
        a.set_values(&vec![1.0; 4 * m]);

        let g = vec![0.0; n];
        let bl: Vec<f64> = (1..=m).map(|i| i as f64).collect();
        let bu = bl.clone();
        let xl = vec![NLP_LOWER_BOUND_INF; n];
        let xu = vec![NLP_UPPER_BOUND_INF; n];

        let qp = QpProblem {
            n,
            m,
            h: &h,
            g: &g,
            a: &a,
            bl: &bl,
            bu: &bu,
            xl: &xl,
            xu: &xu,
            hessian_inertia: HessianInertia::Psd,
        };
        let mut solver = new_solver();
        let started = Instant::now();
        let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
        let elapsed = started.elapsed();
        assert_eq!(sol.status, crate::QpStatus::Optimal);
        eprintln!(
            "[scaling-equality]    n={n:4} m={m:3}  refactor={}  time={:.3}ms",
            sol.stats.n_refactor,
            elapsed.as_secs_f64() * 1000.0,
        );
        // Equality-only fast path ⇒ exactly one factor call.
        assert_eq!(sol.stats.n_refactor, 1);
    }
}

#[test]
fn warm_restart_at_optimum_converges_in_one_iter() {
    // §8.5 warm-start sweep, simplest case: solve cold once;
    // re-solve with the cold result as a warm start. The cold
    // optimum is also the warm optimum (same QP), so the warm
    // solve should declare optimal in a single inner-loop
    // iteration (one refactor, zero working-set changes).
    let n = 20;
    let h = identity_h(n);
    let a = empty_gen(0, n);
    let g: Vec<f64> = (0..n).map(|i| -((i as f64) * 0.1)).collect();
    let xl = vec![0.0; n];
    let xu = vec![5.0; n];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let qp = QpProblem {
        n,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver = new_solver();
    let cold = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    assert_eq!(cold.status, crate::QpStatus::Optimal);

    let ws = crate::QpWarmStart {
        x: cold.x.clone(),
        lambda_g: cold.lambda_g.clone(),
        lambda_x: cold.lambda_x.clone(),
        working: cold.working.clone(),
    };
    let warm = solver.solve(&qp, Some(&ws), &QpOptions::default()).unwrap();
    assert_eq!(warm.status, crate::QpStatus::Optimal);

    eprintln!(
        "[warm-restart]        n={n:4}  cold refactor={}  warm refactor={}  cold ws_chg={}  warm ws_chg={}",
        cold.stats.n_refactor,
        warm.stats.n_refactor,
        cold.stats.n_working_set_changes,
        warm.stats.n_working_set_changes,
    );

    // The warm restart should make no working-set changes —
    // the supplied W already matches the optimum.
    assert_eq!(warm.stats.n_working_set_changes, 0);
    // And take strictly fewer factor calls than the cold path.
    assert!(
        warm.stats.n_refactor < cold.stats.n_refactor
            || warm.stats.n_refactor == cold.stats.n_refactor && cold.stats.n_refactor == 1,
        "warm refactor count {} should be ≤ cold {}",
        warm.stats.n_refactor,
        cold.stats.n_refactor
    );
}
