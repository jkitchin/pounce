//! gh#1061: a degenerate apex the refactor path could not leave.
//!
//! The reporter's cold step QP sat at a vertex where thousands of rows were
//! tight and added them one at a time — 466 adds in 300 s, every step
//! `α ≈ 1e-26`, the objective frozen. With Schur updates off it also stalled.
//! Reproduced here on a cone QP that starts at its apex, where every row is
//! tight, the Schur-off half had a defect of its own:
//!
//! **The refactor path's anti-stall latch fired on every solve.** Its
//! improvement test compared against `best_obj = +∞` as `∞ − ∞ = NaN`, so
//! nothing ever counted as progress and Bland's rule latched on the 50th
//! blocking step. Bland has no EXPAND floor: from there each degenerate pivot
//! was a literal `α = 0`, and a 50-variable cone of this family ran its whole
//! 100 000-iteration budget at `x = 0` while the Schur path solved it in 130
//! changes.
//!
//! The one-pivot-per-row cost itself is not fixed here. Adding the other
//! tight rows in batches was tried and measured: it took a 3000-variable
//! cone from 13.4 s to 2.1 s, and it took Maros-Meszaros `QSCFXM1` from 4012
//! to 7829 iterations and the first step QP of `deb7` from a certificate to
//! its iteration cap, depending on when it fired. The reporter's QP is now
//! started from an interior point instead (`pounce-algorithm`,
//! `qp_ipm_cold_start_min_dim`), which gets it to 0 working-set changes.
//!
//! Counts, not times, are asserted: the counts are deterministic and say the
//! same thing.

use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolution, QpSolver,
    QpStatus, QpWarmStart, WorkingSet,
};
use std::rc::Rc;

/// `min ½‖x‖² + gᵀx  s.t.  Ax ≤ 0`, `A` random with 4 entries per row, `g < 0`
/// so the minimizer is pushed into the rows. Started at `x = 0`, the apex,
/// where all `m` rows are tight, with an empty working set — the shape of the
/// reporter's cold QP, which started with the working set short of the rows
/// its point already sat on.
fn cone_solve(n: usize, m: usize, schur: bool, max_iter: u32) -> QpSolution {
    let mut seed: u64 = 12345;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let idx: Vec<i32> = (1..=n as i32).collect();
    let h_space = SymTMatrixSpace::new(n as i32, idx.clone(), idx);
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&vec![1.0; n]);
    let (mut ai, mut aj, mut av) = (vec![], vec![], vec![]);
    for r in 0..m {
        let mut cols = std::collections::BTreeSet::new();
        while cols.len() < 4 {
            cols.insert((rnd() * n as f64) as usize % n);
        }
        for c in cols {
            ai.push(r as i32 + 1);
            aj.push(c as i32 + 1);
            av.push(rnd() * 2.0 - 0.5);
        }
    }
    let a_space = GenTMatrixSpace::new(m as i32, n as i32, ai, aj);
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&av);
    let g: Vec<f64> = (0..n).map(|_| -(rnd() + 0.1)).collect();
    let bl = vec![-1e20; m];
    let bu = vec![0.0; m];
    let xl = vec![-1e20; n];
    let xu = vec![1e20; n];
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
        // What the SQP driver passes for an exact Hessian: drops refactor.
        hessian_inertia: HessianInertia::Indefinite,
    };
    let ws = QpWarmStart {
        x: vec![0.0; n],
        lambda_g: vec![0.0; m],
        lambda_x: vec![0.0; n],
        working: WorkingSet::cold(n, m),
    };
    let opts = QpOptions {
        use_schur_updates: schur,
        max_iter,
        ..QpOptions::default()
    };
    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    solver.solve(&qp, Some(&ws), &opts).unwrap()
}

/// With the latch firing on the 50th blocking step this cone runs
/// out of any budget at the apex (measured: past 120 s, against 0.47 s and
/// 418 working-set changes with the fix); the Schur path, which has no latch,
/// always reached the optimum. Now both do, to the same objective — the two
/// paths agree to ~1e-8 relative, QP-tolerance noise.
#[test]
fn the_refactor_path_leaves_a_degenerate_apex() {
    let refactor = cone_solve(100, 200, false, 3000);
    let schur = cone_solve(100, 200, true, 3000);
    assert_eq!(refactor.status, QpStatus::Optimal, "{:?}", refactor.stats);
    assert_eq!(schur.status, QpStatus::Optimal, "{:?}", schur.stats);
    assert!(
        (refactor.obj - schur.obj).abs() <= 1e-6 * schur.obj.abs(),
        "refactor {} vs schur {}",
        refactor.obj,
        schur.obj
    );
    assert!(refactor.obj < -0.15, "still at the apex: {}", refactor.obj);
}

/// `solve_with_working_set_if_feasible` (gh#1061): a working set whose pinned
/// point violates another row, even after the pin repair, comes back `None`
/// for the caller to seed another way; one that pins feasible is solved as
/// `solve_with_working_set` would. The reporter's second SQP step QP is the
/// first case at scale — warm-started from the previous QP's working set it
/// went to the elastic phase 1 and reached a 300 s cap.
///
/// `min x₁² + x₂²  s.t.  x₁ + x₂ = 1,  x₁ − x₂ ≤ 0,  0 ≤ x ≤ 1`, optimum
/// `(½, ½)`. Pinning `x₁` at its upper bound with the equality puts the point
/// at `(1, 0)`, which violates the inequality, and pinning that too leaves
/// three rows on two variables that no point satisfies.
#[test]
fn an_infeasible_pin_is_reported_instead_of_going_elastic() {
    use pounce_qp::{BoundStatus, ConsStatus};
    let h_space = SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&[2.0, 2.0]);
    let a_space = GenTMatrixSpace::new(2, 2, vec![1, 1, 2, 2], vec![1, 2, 1, 2]);
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&[1.0, 1.0, 1.0, -1.0]);
    let g = [0.0, 0.0];
    let (bl, bu) = ([1.0, -1e20], [1.0, 0.0]);
    let (xl, xu) = ([0.0, 0.0], [1.0, 1.0]);
    let qp = QpProblem {
        n: 2,
        m: 2,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Indefinite,
    };
    let opts = QpOptions::default();
    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));

    let mut bad = WorkingSet::cold(2, 2);
    bad.constraints[0] = ConsStatus::Equality;
    bad.bounds[0] = BoundStatus::AtUpper;
    let out = solver
        .solve_with_working_set_if_feasible(&qp, &bad, &opts)
        .unwrap();
    assert!(
        out.is_none(),
        "an infeasible pin was solved: {:?}",
        out.map(|s| s.x)
    );

    // A dependent working set — the equality plus both bounds it already
    // implies at `(1, 0)` — does not factor. Here it is reported unusable
    // without the rank prune `solve_with_working_set` would run: on
    // gh#1061's correction QP that prune was three sparse rank tests and 40
    // of the 44 s spent before the hint turned out infeasible anyway.
    let mut dependent = WorkingSet::cold(2, 2);
    dependent.constraints[0] = ConsStatus::Equality;
    dependent.bounds[0] = BoundStatus::AtUpper;
    dependent.bounds[1] = BoundStatus::AtLower;
    let guards = solver.rank_guard_calls();
    let factors = solver.linsol_time_split().n_factor;
    let out = solver
        .solve_with_working_set_if_feasible(&qp, &dependent, &opts)
        .unwrap();
    assert!(
        out.is_none(),
        "a dependent pin was solved: {:?}",
        out.map(|s| s.x)
    );
    assert_eq!(solver.rank_guard_calls(), guards, "the pin ran a rank test");
    // Nor the inertia-shift ladder, which cannot repair a singular
    // constraint block and which this caller would reject a shift from
    // anyway: one factorization, not 1 + `inertia_max_shifts`.
    assert_eq!(
        solver.linsol_time_split().n_factor - factors,
        1,
        "the pin ran the shift ladder"
    );

    let mut good = WorkingSet::cold(2, 2);
    good.constraints[0] = ConsStatus::Equality;
    let sol = solver
        .solve_with_working_set_if_feasible(&qp, &good, &opts)
        .unwrap()
        .expect("a feasible pin is solved");
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        (sol.x[0] - 0.5).abs() < 1e-8 && (sol.x[1] - 0.5).abs() < 1e-8,
        "{:?}",
        sol.x
    );
}

/// A full step that nothing blocks leaves the working set, and so the KKT
/// matrix, unchanged; the refactor path back-solves against the factor it
/// has instead of factoring that matrix again (gh#1061). The next iteration
/// is the stationary check that picks the next drop, so before this every
/// drop cost two factorizations: the reporter's warm step QP took 306 for
/// 156 working-set changes, and 157 now, on the same path to the same
/// objective.
///
/// `min ½‖x‖² − cᵀx  s.t.  Σx ≤ 100,  0 ≤ x ≤ 1`, `0 < c < 1`, warm-started
/// at `x = 0` with every lower bound in the working set. Each bound's
/// multiplier has the wrong sign and each drop's step lands inside the box,
/// so the solve is `n` drops and `n` unblocked steps.
#[test]
fn an_unblocked_step_reuses_its_factor() {
    let n = 20;
    let idx: Vec<i32> = (1..=n as i32).collect();
    let h_space = SymTMatrixSpace::new(n as i32, idx.clone(), idx.clone());
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&vec![1.0; n]);
    let a_space = GenTMatrixSpace::new(1, n as i32, vec![1; n], idx);
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&vec![1.0; n]);
    let c: Vec<f64> = (0..n).map(|i| 0.1 + 0.8 * i as f64 / n as f64).collect();
    let g: Vec<f64> = c.iter().map(|v| -v).collect();
    let (bl, bu) = ([-1e20], [100.0]);
    let (xl, xu) = (vec![0.0; n], vec![1.0; n]);
    let qp = QpProblem {
        n,
        m: 1,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Indefinite,
    };
    let mut working = WorkingSet::cold(n, 1);
    working.bounds.fill(pounce_qp::BoundStatus::AtLower);
    let ws = QpWarmStart {
        x: vec![0.0; n],
        lambda_g: vec![0.0],
        lambda_x: vec![0.0; n],
        working,
    };
    let opts = QpOptions {
        use_schur_updates: false,
        ..QpOptions::default()
    };
    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let sol = solver.solve(&qp, Some(&ws), &opts).unwrap();
    assert_eq!(sol.status, QpStatus::Optimal);
    for (xi, ci) in sol.x.iter().zip(&c) {
        assert!((xi - ci).abs() < 1e-9, "{:?}", sol.x);
    }
    assert_eq!(sol.stats.n_working_set_changes, n as u32, "{:?}", sol.stats);
    assert!(
        sol.stats.n_refactor <= n as u32 + 1,
        "{} factorizations for {} drops",
        sol.stats.n_refactor,
        n
    );

    // Through `solve_with_working_set` the hint is pinned first, and that
    // factor is of the matrix the first iteration assembles: it is reused
    // too, so the pin costs no factorization of its own. The one beyond the
    // drops' is the second-order check `solve` runs on an `Indefinite` QP.
    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let sol = solver
        .solve_with_working_set(&qp, &ws.working, &opts)
        .unwrap();
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_eq!(sol.stats.n_working_set_changes, n as u32, "{:?}", sol.stats);
    let factors = solver.linsol_time_split().n_factor;
    assert!(
        factors <= n as u64 + 2,
        "{factors} factorizations, the pin's included, for {n} drops"
    );
}
