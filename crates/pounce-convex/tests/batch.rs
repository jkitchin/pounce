//! Batched / multiple-RHS convex-QP solving (pounce#74–#77 analogue at
//! the optimization layer). Each batched solution must match the
//! corresponding single-problem solve, in order.

use pounce_convex::{
    solve_qp_batch, solve_qp_batch_parallel, solve_qp_ipm, solve_qp_multi_rhs, QpOptions,
    QpProblem, QpStatus, Triplet,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Inner-serial backend for the parallel batch path (outer-parallel /
/// inner-serial); feral's parallel and serial drivers are bit-identical, so
/// results match `backend`.
fn serial_backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::serial())
}

/// A simple box-constrained QP `min ½‖x − t‖²·2 ... ` parameterized by a
/// target via the linear term. `c = −2·t` ⇒ unconstrained optimum at `t`,
/// clamped to [0, 1] by the bounds.
fn boxed_qp(c: Vec<f64>) -> QpProblem {
    let n = c.len();
    QpProblem {
        n,
        p_lower: (0..n).map(|i| Triplet::new(i, i, 2.0)).collect(),
        c,
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0; n],
        ub: vec![1.0; n],
    }
}

#[test]
fn batch_matches_individual_solves() {
    let probs = vec![
        boxed_qp(vec![-1.0, -4.0]), // opt clamps to (0.5, 1.0)
        boxed_qp(vec![-4.0, 1.0]),  // opt clamps to (1.0, 0.0)
        boxed_qp(vec![0.0, 0.0]),   // opt at (0, 0)
    ];
    let opts = QpOptions::default();

    let batched = solve_qp_batch(&probs, &opts, backend);
    assert_eq!(batched.len(), probs.len());

    for (i, prob) in probs.iter().enumerate() {
        let single = solve_qp_ipm(prob, &opts, backend);
        assert_eq!(batched[i].status, QpStatus::Optimal);
        assert_eq!(single.status, QpStatus::Optimal);
        for j in 0..prob.n {
            assert!(
                (batched[i].x[j] - single.x[j]).abs() < 1e-9,
                "batch[{i}].x[{j}] {} vs single {}",
                batched[i].x[j],
                single.x[j]
            );
        }
        assert!((batched[i].obj - single.obj).abs() < 1e-9);
    }
}

#[test]
fn multi_rhs_matches_individual_solves() {
    // Same structure (P = 2I, 0 ≤ x ≤ 1), many objectives.
    let base = boxed_qp(vec![0.0, 0.0]);
    let cs = vec![
        vec![-1.0, -4.0],
        vec![-4.0, 1.0],
        vec![3.0, -2.0],
        vec![0.0, 0.0],
    ];
    let opts = QpOptions::default();

    let many = solve_qp_multi_rhs(&base, &cs, &opts, backend);
    assert_eq!(many.len(), cs.len());

    for (i, c) in cs.iter().enumerate() {
        let single = solve_qp_ipm(&boxed_qp(c.clone()), &opts, backend);
        assert_eq!(many[i].status, QpStatus::Optimal);
        for j in 0..base.n {
            assert!(
                (many[i].x[j] - single.x[j]).abs() < 1e-9,
                "multi[{i}].x[{j}] {} vs single {}",
                many[i].x[j],
                single.x[j]
            );
        }
    }

    // Spot-check known clamped optima (IPM tolerance ~1e-4):
    // c=(-1,-4) → unconstrained (0.5, 2.0) clamps to (0.5, 1.0).
    assert!((many[0].x[0] - 0.5).abs() < 1e-4, "x0={}", many[0].x[0]);
    assert!((many[0].x[1] - 1.0).abs() < 1e-4, "x1={}", many[0].x[1]);
    // c=(3,-2) → unconstrained (−1.5, 1.0) clamps to (0.0, 1.0).
    assert!(many[2].x[0].abs() < 1e-4, "x0={}", many[2].x[0]);
    assert!((many[2].x[1] - 1.0).abs() < 1e-4, "x1={}", many[2].x[1]);
}

#[test]
fn batch_preserves_per_instance_status() {
    // Mix a feasible QP with an unbounded one; statuses must line up
    // with the inputs by index.
    let feasible = boxed_qp(vec![-1.0, -1.0]);
    let unbounded = QpProblem {
        n: 1,
        p_lower: vec![], // LP
        c: vec![-1.0],   // min −x0 with x0 ≥ 0, no upper bound
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, -1.0)],
        h: vec![0.0],
        lb: vec![],
        ub: vec![],
    };
    let probs = vec![feasible, unbounded];
    let res = solve_qp_batch(&probs, &QpOptions::default(), backend);
    assert_eq!(res[0].status, QpStatus::Optimal);
    assert_eq!(res[1].status, QpStatus::DualInfeasible);
}

#[test]
fn large_batch_parallel_path() {
    // A batch big enough to exercise the dedicated parallel pool (and the
    // worker-stack / feral-serial handling that prevents the nested-pool
    // stack overflow). Results must match index-wise.
    let opts = QpOptions::default();
    let probs: Vec<QpProblem> = (0..1500)
        .map(|k| {
            let t = (k as f64) / 500.0; // sweeps across the box and beyond
            boxed_qp(vec![-2.0 * t, -2.0 * (1.0 - t)])
        })
        .collect();
    let batched = solve_qp_batch_parallel(&probs, &opts, serial_backend);
    assert_eq!(batched.len(), probs.len());
    // Compare a sample against single solves (full sweep would be slow).
    for k in (0..probs.len()).step_by(97) {
        assert_eq!(batched[k].status, QpStatus::Optimal, "k={k}");
        let single = solve_qp_ipm(&probs[k], &opts, backend);
        for j in 0..probs[k].n {
            assert!((batched[k].x[j] - single.x[j]).abs() < 1e-9, "k={k} j={j}");
        }
    }
}

// --- QpFactorization: build-once / solve-many across instances ---

use pounce_convex::QpFactorization;

#[test]
fn factorization_handle_matches_one_shot() {
    // Fixed structure (P = 2I, 0 ≤ x ≤ 1), many objectives; the handle's
    // reused symbolic factor must give the same answers as one-shot solves.
    //
    // This test is about the *factorization-reuse* mechanism, so it compares
    // against the identical algorithm: the build-once handle path runs the
    // direct (non-HSDE) IPM on a captured factorization and does not
    // Ruiz-equilibrate (it preserves the captured structure across instances),
    // so both `use_hsde` and `equilibrate` are disabled on the one-shot too —
    // otherwise the two would be different solves and only agree to solver
    // tolerance, not the bit-tight match the reuse correctness check wants.
    let base = boxed_qp(vec![0.0, 0.0]);
    let opts = QpOptions {
        use_hsde: false,
        equilibrate: false,
        ..QpOptions::default()
    };
    let mut handle = QpFactorization::build(&base, &opts, backend).expect("build");

    for c in [
        vec![-1.0, -4.0],
        vec![-4.0, 1.0],
        vec![3.0, -2.0],
        vec![0.0, 0.0],
        vec![-2.0, -2.0],
    ] {
        let prob = boxed_qp(c.clone());
        let reused = handle.solve(&prob);
        let one_shot = solve_qp_ipm(&prob, &opts, backend);
        assert_eq!(reused.status, QpStatus::Optimal, "c={c:?}");
        for j in 0..base.n {
            assert!(
                (reused.x[j] - one_shot.x[j]).abs() < 1e-9,
                "c={c:?} x[{j}] reused {} vs one-shot {}",
                reused.x[j],
                one_shot.x[j]
            );
            // Bound duals must match too.
            assert!((reused.z_lb[j] - one_shot.z_lb[j]).abs() < 1e-6);
            assert!((reused.z_ub[j] - one_shot.z_ub[j]).abs() < 1e-6);
        }
        assert!((reused.obj - one_shot.obj).abs() < 1e-9);
    }
}

#[test]
fn factorization_handle_rejects_pattern_mismatch() {
    // Built on a 2-var box QP; solving a 3-var problem must not silently
    // reuse the wrong factor — it returns NumericalFailure.
    let base = boxed_qp(vec![0.0, 0.0]);
    let mut handle = QpFactorization::build(&base, &QpOptions::default(), backend).expect("build");

    let mismatched = boxed_qp(vec![0.0, 0.0, 0.0]); // n = 3
    let sol = handle.solve(&mismatched);
    assert_eq!(sol.status, QpStatus::NumericalFailure);

    // A matching-structure problem still solves fine afterward.
    let ok = handle.solve(&boxed_qp(vec![-1.0, -1.0]));
    assert_eq!(ok.status, QpStatus::Optimal);
}

/// An inequality-constrained QP `min ½·2‖x‖² + cᵀx  s.t.  Σx ≤ 10`, no
/// variable bounds — so `m_ineq = 1` (the explicit `G` row) and the dual
/// vector `z` is non-empty on every code path.
fn ineq_qp(c: Vec<f64>) -> QpProblem {
    let n = c.len();
    QpProblem {
        n,
        p_lower: (0..n).map(|i| Triplet::new(i, i, 2.0)).collect(),
        c,
        a: vec![],
        b: vec![],
        g: (0..n).map(|j| Triplet::new(0, j, 1.0)).collect(),
        h: vec![10.0],
        lb: vec![],
        ub: vec![],
    }
}

#[test]
fn pattern_mismatch_failure_seeds_zero_dual() {
    // Code review L37: failure paths must seed the inequality dual
    // consistently. The QpFactorization pattern-mismatch failure used to
    // return z = 1 (an orthant-era artifact — and not even a member of a
    // general dual cone: the all-ones vector violates an SOC of dimension
    // ≥ 3), while the cone-cover / validation failures return z = 0. They
    // now agree on z = 0: the cone apex, valid in every dual cone and
    // matching the trivial x = 0, y = 0 the same failure returns.
    let base = ineq_qp(vec![-1.0, -1.0]); // n = 2, m_ineq = 1
    let mut handle = QpFactorization::build(&base, &QpOptions::default(), backend).expect("build");

    let mismatched = ineq_qp(vec![-1.0, -1.0, -1.0]); // n = 3 ⇒ pattern mismatch
    let sol = handle.solve(&mismatched);

    assert_eq!(sol.status, QpStatus::NumericalFailure);
    assert_eq!(sol.x, vec![0.0; 3], "failure primal is the trivial point");
    assert_eq!(sol.y, vec![0.0; mismatched.m_eq()]);
    assert_eq!(
        sol.z.len(),
        mismatched.m_ineq(),
        "z spans the inequality rows"
    );
    assert!(
        sol.z.iter().all(|&zi| zi == 0.0),
        "failure inequality dual must be all-zeros (cone apex), got {:?}",
        sol.z
    );
}
