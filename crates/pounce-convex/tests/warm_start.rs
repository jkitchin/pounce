//! Warm-start tests for the convex-QP interior-point solver.
//!
//! Warm starting an IPM is subtle: a converged solution sits on the
//! complementarity boundary, the worst place to restart. The solver's
//! Mehrotra-style recentering ([`QpWarmStart`]) keeps the warm primal but
//! pushes the slacks/multipliers back into the interior. These tests check
//! two things:
//!
//! 1. **Correctness** — a warm-started solve reaches the *same* optimum as
//!    a cold solve (the start cannot change the KKT point it converges to).
//! 2. **Benefit** — on a nearby problem, warm starting takes no more
//!    iterations than cold (and typically fewer).

use pounce_convex::{
    QpFactorization, QpOptions, QpProblem, QpStatus, QpWarmStart, Triplet, solve_qp_batch_parallel,
    solve_qp_batch_parallel_warm, solve_qp_ipm, solve_qp_ipm_warm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// A box-constrained QP `min ½·2‖x‖² + cᵀx s.t. 0 ≤ x ≤ 5` (P = 2I).
fn box_qp(c: &[f64]) -> QpProblem {
    let n = c.len();
    QpProblem {
        n,
        p_lower: (0..n).map(|i| Triplet::new(i, i, 2.0)).collect(),
        c: c.to_vec(),
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0; n],
        ub: vec![5.0; n],
    }
}

/// An inequality-constrained QP `min ½·2‖x‖² + cᵀx s.t. Σx ≤ cap`.
fn capped_qp(c: &[f64], cap: f64) -> QpProblem {
    let n = c.len();
    QpProblem {
        n,
        p_lower: (0..n).map(|i| Triplet::new(i, i, 2.0)).collect(),
        c: c.to_vec(),
        a: vec![],
        b: vec![],
        g: (0..n).map(|i| Triplet::new(0, i, 1.0)).collect(),
        h: vec![cap],
        lb: vec![],
        ub: vec![],
    }
}

#[test]
fn warm_start_matches_cold_solution() {
    let opts = QpOptions::default();
    // Solve a base problem, then warm-start a perturbed one from it.
    let base = capped_qp(&[-1.0, -2.0, -0.5], 1.0);
    let base_sol = solve_qp_ipm(&base, &opts, backend);
    assert_eq!(base_sol.status, QpStatus::Optimal);

    let pert = capped_qp(&[-1.2, -1.8, -0.6], 1.1);
    let cold = solve_qp_ipm(&pert, &opts, backend);
    let warm = solve_qp_ipm_warm(
        &pert,
        &opts,
        &QpWarmStart::from_solution(&base_sol),
        backend,
    );

    assert_eq!(cold.status, QpStatus::Optimal);
    assert_eq!(warm.status, QpStatus::Optimal);
    // Same primal, dual, and objective regardless of the start.
    for i in 0..pert.n {
        assert!(
            (cold.x[i] - warm.x[i]).abs() < 1e-6,
            "x[{i}]: cold={} warm={}",
            cold.x[i],
            warm.x[i]
        );
    }
    assert!((cold.obj - warm.obj).abs() < 1e-6);
    assert!((cold.z[0] - warm.z[0]).abs() < 1e-6);
}

#[test]
fn warm_start_matches_cold_with_bounds() {
    let opts = QpOptions::default();
    let base = box_qp(&[-3.0, 6.0, -10.0]); // mixes interior, lower, upper
    let base_sol = solve_qp_ipm(&base, &opts, backend);
    assert_eq!(base_sol.status, QpStatus::Optimal);

    let pert = box_qp(&[-3.5, 5.5, -9.0]);
    let cold = solve_qp_ipm(&pert, &opts, backend);
    let warm = solve_qp_ipm_warm(
        &pert,
        &opts,
        &QpWarmStart::from_solution(&base_sol),
        backend,
    );

    assert_eq!(warm.status, QpStatus::Optimal);
    for i in 0..pert.n {
        assert!(
            (cold.x[i] - warm.x[i]).abs() < 1e-6,
            "x[{i}]: cold={} warm={}",
            cold.x[i],
            warm.x[i]
        );
        assert!((cold.z_lb[i] - warm.z_lb[i]).abs() < 1e-6);
        assert!((cold.z_ub[i] - warm.z_ub[i]).abs() < 1e-6);
    }
}

#[test]
fn warm_start_reduces_iterations_on_nearby_problem() {
    // This test isolates the *warm-start mechanism*, so it holds the problem
    // conditioning fixed by disabling equilibration. Ruiz equilibration is an
    // independent iteration-count improvement; on a problem this small and
    // well-scaled it makes the cold solve converge so well (here, 7 iters) that
    // it absorbs the warm-start margin, conflating the two effects. The
    // equilibrated warm path is exercised by `parallel_batch_warm_*`.
    let opts = QpOptions {
        equilibrate: false,
        ..QpOptions::default()
    };
    // Larger problem so the iteration difference is meaningful.
    let n = 30;
    let c0: Vec<f64> = (0..n).map(|i| -1.0 - (i as f64) * 0.1).collect();
    let base = capped_qp(&c0, 5.0);
    let base_sol = solve_qp_ipm(&base, &opts, backend);
    assert_eq!(base_sol.status, QpStatus::Optimal);

    // A small perturbation of c and the cap.
    let c1: Vec<f64> = c0.iter().map(|v| v * 1.02).collect();
    let pert = capped_qp(&c1, 5.1);

    let cold = solve_qp_ipm(&pert, &opts, backend);
    let warm = solve_qp_ipm_warm(
        &pert,
        &opts,
        &QpWarmStart::from_solution(&base_sol),
        backend,
    );
    assert_eq!(cold.status, QpStatus::Optimal);
    assert_eq!(warm.status, QpStatus::Optimal);

    // The warm start should not need more iterations than cold; for a
    // perturbation this small it should need strictly fewer.
    assert!(
        warm.iters <= cold.iters,
        "warm should not regress: warm={} cold={}",
        warm.iters,
        cold.iters
    );
    assert!(
        warm.iters < cold.iters,
        "warm should beat cold on a nearby problem: warm={} cold={}",
        warm.iters,
        cold.iters
    );
}

#[test]
fn factorization_solve_warm_combines_reuse_and_warm() {
    let opts = QpOptions::default();
    let base = capped_qp(&[-1.0, -2.0, -0.5, -1.5], 2.0);
    let base_sol = solve_qp_ipm(&base, &opts, backend);

    // Build-once / solve-many handle; warm-start a same-structure solve.
    let mut handle = QpFactorization::build(&base, &opts, backend).expect("factor builds");
    let pert = capped_qp(&[-1.1, -1.9, -0.4, -1.6], 2.1);
    let warm = handle.solve_warm(&pert, &QpWarmStart::from_solution(&base_sol));
    let cold = solve_qp_ipm(&pert, &opts, backend);

    assert_eq!(warm.status, QpStatus::Optimal);
    for i in 0..pert.n {
        assert!(
            (cold.x[i] - warm.x[i]).abs() < 1e-6,
            "x[{i}]: cold={} warm={}",
            cold.x[i],
            warm.x[i]
        );
    }
}

#[test]
fn primal_only_warm_start_is_accepted() {
    // A warm start carrying only the primal `x` (cold `y`/`z`) still seeds
    // the solve and reaches the right optimum — this is the mode the JAX
    // differentiable layer uses, where only the primal is returned.
    let opts = QpOptions::default();
    let base = capped_qp(&[-1.0, -2.0, -0.5], 1.0);
    let base_sol = solve_qp_ipm(&base, &opts, backend);

    let pert = capped_qp(&[-1.1, -1.9, -0.55], 1.05);
    let primal_only = QpWarmStart {
        x: base_sol.x.clone(),
        y: Vec::new(),
        z: Vec::new(),
        z_lb: Vec::new(),
        z_ub: Vec::new(),
    };
    let warm = solve_qp_ipm_warm(&pert, &opts, &primal_only, backend);
    let cold = solve_qp_ipm(&pert, &opts, backend);
    assert_eq!(warm.status, QpStatus::Optimal);
    for i in 0..pert.n {
        assert!((cold.x[i] - warm.x[i]).abs() < 1e-6);
    }
}

#[test]
fn parallel_batch_warm_matches_cold_and_helps() {
    let opts = QpOptions::default();
    // A batch of base problems, then a perturbed batch warm-started from
    // the base solutions.
    let base: Vec<QpProblem> = (0..6)
        .map(|k| capped_qp(&[-1.0 - 0.1 * k as f64, -2.0, -0.5], 1.0))
        .collect();
    let base_sols = solve_qp_batch_parallel(&base, &opts, backend);

    let pert: Vec<QpProblem> = (0..6)
        .map(|k| capped_qp(&[-1.05 - 0.1 * k as f64, -1.95, -0.55], 1.05))
        .collect();
    let warms: Vec<QpWarmStart> = base_sols.iter().map(QpWarmStart::from_solution).collect();

    let cold = solve_qp_batch_parallel(&pert, &opts, backend);
    let warm = solve_qp_batch_parallel_warm(&pert, &warms, &opts, backend);

    assert_eq!(cold.len(), 6);
    assert_eq!(warm.len(), 6);
    for k in 0..6 {
        assert_eq!(warm[k].status, QpStatus::Optimal);
        for i in 0..pert[k].n {
            assert!(
                (cold[k].x[i] - warm[k].x[i]).abs() < 1e-6,
                "batch[{k}] x[{i}]: cold={} warm={}",
                cold[k].x[i],
                warm[k].x[i]
            );
        }
        // Per-instance warm start should not regress iterations.
        assert!(
            warm[k].iters <= cold[k].iters,
            "batch[{k}] iters: warm={} cold={}",
            warm[k].iters,
            cold[k].iters
        );
    }
}

#[test]
#[should_panic(expected = "must equal")]
fn parallel_batch_warm_mismatched_lengths_panics() {
    let opts = QpOptions::default();
    let probs = vec![capped_qp(&[-1.0, -2.0, -0.5], 1.0)];
    let warms: Vec<QpWarmStart> = Vec::new(); // wrong length
    let _ = solve_qp_batch_parallel_warm(&probs, &warms, &opts, backend);
}

#[test]
fn stale_warm_start_dims_fall_back_to_cold() {
    let opts = QpOptions::default();
    let prob = capped_qp(&[-1.0, -2.0, -0.5], 1.0);
    // A warm start with the wrong dimensions must be ignored, not crash.
    let bogus = QpWarmStart {
        x: vec![0.0; 7],
        y: vec![],
        z: vec![0.0; 3],
        z_lb: vec![],
        z_ub: vec![],
    };
    let sol = solve_qp_ipm_warm(&prob, &opts, &bogus, backend);
    assert_eq!(sol.status, QpStatus::Optimal);
}
