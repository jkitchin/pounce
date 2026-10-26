//! gh#1061: one pivot per tight row at a degenerate vertex.
//!
//! The reporter's cold step QP sat at a vertex where thousands of rows were
//! tight and added them one at a time — 466 adds in 300 s, every step
//! `α ≈ 1e-26`, the objective frozen. With Schur updates off it also stalled.
//! Two defects, both reproduced here on a cone QP that starts at its apex,
//! where every row is tight:
//!
//! 1. **The refactor path's anti-stall latch fired on every solve.** Its
//!    improvement test compared against `best_obj = +∞` as `∞ − ∞ = NaN`, so
//!    nothing ever counted as progress and Bland's rule latched on the 50th
//!    blocking step. Bland has no EXPAND floor: from there each degenerate
//!    pivot was a literal `α = 0`, and the 50-variable cone below ran its
//!    whole 100 000-iteration budget at `x = 0` while the Schur path solved it
//!    in 130 changes.
//! 2. **One working-set change per tight row.** After
//!    `DEGENERATE_BATCH_STREAK` degenerate adds in a row, each step now also
//!    adds up to `DEGENERATE_BATCH_MAX` of the other rows it lands on.
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

/// Defect 1. With the latch firing on the 50th blocking step this cone runs
/// out of any budget at the apex (measured: past 120 s, against 0.47 s and
/// 418 working-set changes with the fix); the Schur path, which has no latch,
/// always reached the optimum. Now both do, to the same objective — the two
/// paths agree to ~1e-8 relative, QP-tolerance noise. The batch alone does
/// not hide this one: at `50 × 100` it gets off the apex even with the latch
/// broken, here it does not.
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

/// Defect 2, on the reporter's regime: many tight rows, working set well short
/// of `n`. Without the batch this took 423 refactorizations on the refactor
/// path and 2082 back-solves on the Schur path; with it, 60 and 781, to the
/// same objective (`-92.49386824945` unbatched).
#[test]
fn a_long_degenerate_streak_is_added_in_batches() {
    let refactor = cone_solve(1000, 500, false, 20_000);
    let schur = cone_solve(1000, 500, true, 20_000);
    for sol in [&refactor, &schur] {
        assert_eq!(sol.status, QpStatus::Optimal, "{:?}", sol.stats);
        assert!(
            (sol.obj - -92.49386824945).abs() <= 1e-8 * 92.5,
            "objective {}",
            sol.obj
        );
    }
    assert!(refactor.stats.n_refactor <= 150, "{:?}", refactor.stats);
}
