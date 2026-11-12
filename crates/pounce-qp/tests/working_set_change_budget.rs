//! `QpOptions::max_working_set_changes` bounds a whole solve; `max_iter`
//! bounds one pass.
//!
//! An infeasible start goes through l1-elastic phase 1, and when phase 1 ends
//! with slack left that path runs more passes — penalty escalations, the
//! infeasibility probe — each with its own `max_iter`. On the infeasible LP
//! below, `max_iter = 20` lets the solve make 117 working-set changes. The
//! solve-wide budget stops it at 20, and the solve reports `MaxIter`, not
//! `Infeasible`: a pass cut short is not read as a verdict.

use pounce_common::types::NLP_UPPER_BOUND_INF;
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{
    AntiCyclingChoice, HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver,
    QpStats, QpStatus,
};

const N: usize = 20;
/// The sine rows plus the contradictory pair `Σx ≥ 1`, `−Σx ≥ 0`.
const M: usize = 42;
const BUDGET: u32 = 20;

/// `min gᵀx  s.t.  S x ≥ S·1,  Σx ≥ 1,  −Σx ≥ 0,  −10 ≤ x ≤ 10`, with `S` a
/// dense-ish sine pattern: infeasible by the last two rows, and infeasible at
/// the cold start `x = 0` by the first block too.
fn solve(opts: &QpOptions) -> (QpStatus, QpStats) {
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..M - 2 {
        for j in 0..N {
            let a = ((i * 7 + j * 13) as f64).sin();
            if a.abs() > 0.3 {
                rows.push((i + 1) as i32);
                cols.push((j + 1) as i32);
                vals.push(a);
            }
        }
    }
    let mut bl = vec![0.0; M];
    for (&r, &v) in rows.iter().zip(&vals) {
        bl[(r - 1) as usize] += v;
    }
    for (i, sign, lower) in [(M - 2, 1.0, 1.0), (M - 1, -1.0, 0.0)] {
        for j in 0..N {
            rows.push((i + 1) as i32);
            cols.push((j + 1) as i32);
            vals.push(sign);
        }
        bl[i] = lower;
    }
    let mut a = GenTMatrix::new(GenTMatrixSpace::new(M as i32, N as i32, rows, cols));
    a.set_values(&vals);
    let h = SymTMatrix::new(SymTMatrixSpace::new(N as i32, Vec::new(), Vec::new()));
    let g: Vec<f64> = (0..N).map(|j| 1.0 + ((j * 5) as f64).cos().abs()).collect();
    let bu = vec![NLP_UPPER_BOUND_INF; M];
    let (xl, xu) = (vec![-10.0; N], vec![10.0; N]);
    let qp = QpProblem {
        n: N,
        m: M,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let sol = solver.solve(&qp, None, opts).expect("solve");
    (sol.status, sol.stats)
}

fn options(max_working_set_changes: Option<u32>) -> QpOptions {
    QpOptions {
        anti_cycling: AntiCyclingChoice::Bland,
        max_iter: BUDGET,
        max_working_set_changes,
        ..QpOptions::default()
    }
}

#[test]
fn the_budget_caps_working_set_changes_across_every_elastic_pass() {
    // Precondition: the per-pass cap alone lets the elastic passes add up.
    let (_, per_pass) = solve(&options(None));
    assert!(
        per_pass.used_phase1 && per_pass.n_working_set_changes > BUDGET,
        "precondition: the cold start no longer spans several elastic passes \
         past {BUDGET} changes; pick an LP that does: {per_pass:?}"
    );

    let (status, budgeted) = solve(&options(Some(BUDGET)));
    assert!(budgeted.n_working_set_changes <= BUDGET, "{budgeted:?}");
    assert_eq!(status, QpStatus::MaxIter, "{budgeted:?}");
}
