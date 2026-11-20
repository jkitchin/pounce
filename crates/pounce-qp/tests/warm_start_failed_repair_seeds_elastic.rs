//! gh#1206 B7 — a hint the pin repair gives up on still starts near itself.
//!
//! `solve_with_working_set` pins the hinted rows and, when the pinned point
//! violates other rows, repairs it by pinning those too (#428). The repair
//! declines when that would pin more than `n` rows: a hint that already pins
//! a full vertex. Then the point failed `solve`'s admission pre-check and
//! went to an l1-elastic phase 1 seeded at the *origin* — the hint discarded,
//! and about one working-set change per row spent rebuilding it. The phase is
//! now seeded at the pinned point, where the hint's rows are active.
//!
//! Fixture: `min ½·1e-3‖x‖² − Σ (1 + i)·xᵢ` over `xᵢ ≤ 3` (rows 0..n) and
//! `Σ xᵢ ≤ 3n − 1` (row n), `0 ≤ x ≤ 10`. The optimum gives up one unit on
//! the cheapest variable: `x₀ = 2`, the rest at 3 — the hint (every `xᵢ ≤ 3`
//! active) with one row swapped. Pinned, the hint is `x = 3`, a full vertex
//! violating the sum row, so the repair (`n + 1 > n`) gives up.
//!
//! Opt-in (`QpOptions::elastic_seed_from_hint`, off by default): on #428's
//! MPC sweep the same branch is slower seeded; see the option.

use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::working_set::{ConsStatus, WorkingSet};
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolution, QpSolver, QpStatus,
};

const NEG_INF: f64 = -1e20;

fn run(n: usize, seed_from_hint: bool) -> QpSolution {
    let mut h = SymTMatrix::new(SymTMatrixSpace::new(
        n as i32,
        (1..=n as i32).collect(),
        (1..=n as i32).collect(),
    ));
    h.set_values(&vec![1e-3; n]);
    let (mut ir, mut jc, mut v) = (vec![], vec![], vec![]);
    for i in 0..n {
        ir.push((i + 1) as i32);
        jc.push((i + 1) as i32);
        v.push(1.0);
        ir.push((n + 1) as i32);
        jc.push((i + 1) as i32);
        v.push(1.0);
    }
    let mut a = GenTMatrix::new(GenTMatrixSpace::new((n + 1) as i32, n as i32, ir, jc));
    a.set_values(&v);
    let g: Vec<f64> = (0..n).map(|i| -(1.0 + i as f64)).collect();
    let bl = vec![NEG_INF; n + 1];
    let mut bu = vec![3.0; n + 1];
    bu[n] = 3.0 * n as f64 - 1.0;
    let xl = vec![0.0; n];
    let xu = vec![10.0; n];
    let qp = QpProblem {
        n,
        m: n + 1,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let mut hint = WorkingSet::cold(n, n + 1);
    for c in hint.constraints[..n].iter_mut() {
        *c = ConsStatus::AtUpper;
    }
    let opts = QpOptions {
        elastic_seed_from_hint: seed_from_hint,
        max_iter: 1000,
        ..QpOptions::default()
    };
    let mut s = ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    s.solve_with_working_set(&qp, &hint, &opts).unwrap()
}

#[test]
fn a_hint_the_repair_gives_up_on_seeds_the_elastic_phase() {
    let n = 30;
    let seeded = run(n, true);
    let cold = run(n, false);
    for sol in [&seeded, &cold] {
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 2.0).abs() < 1e-7, "{:?}", sol.x);
        assert!(
            sol.x[1..].iter().all(|&v| (v - 3.0).abs() < 1e-7),
            "{:?}",
            sol.x
        );
    }
    // Both went through phase 1 — the hint is infeasible either way —
    // but only the cold seed rebuilt the vertex row by row. Measured: 2
    // working-set changes seeded, 62 cold.
    assert!(seeded.stats.used_phase1 && cold.stats.used_phase1);
    assert!(
        cold.stats.n_working_set_changes >= n as u32,
        "cold: {} changes",
        cold.stats.n_working_set_changes
    );
    assert!(
        seeded.stats.n_working_set_changes * 5 <= cold.stats.n_working_set_changes,
        "seeded: {} changes, cold: {}",
        seeded.stats.n_working_set_changes,
        cold.stats.n_working_set_changes
    );
}
