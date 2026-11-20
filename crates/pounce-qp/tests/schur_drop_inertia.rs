//! gh#1206 P1 — a Schur-path drop on a Hessian not known PSD keeps the
//! rank-2 update when Haynsworth inertia additivity says the updated KKT has
//! the inertia the new working set needs, and refactorizes otherwise.
//!
//! Before P1 every such drop refactorized to re-run inertia control (gh#1054),
//! so a warm start that releases `k` bounds paid `k` factorizations. Both
//! fixtures are solved with the test on and off; the answers must agree, and
//! the counts say which branch each drop took.

#![allow(clippy::unwrap_used)]
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{
    BoundStatus, HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolution,
    QpSolver, QpStatus, QpWarmStart, WorkingSet,
};

/// Tridiagonal `H` with diagonal `diag` and off-diagonal `0.3`, plus an
/// optional last diagonal entry `last`.
fn tridiag(n: usize, diag: f64, last: f64) -> SymTMatrix {
    let (mut ir, mut jc, mut v) = (vec![], vec![], vec![]);
    for i in 0..n {
        ir.push((i + 1) as i32);
        jc.push((i + 1) as i32);
        v.push(if i + 1 == n { last } else { diag });
        if i > 0 {
            ir.push((i + 1) as i32);
            jc.push(i as i32);
            v.push(0.3);
        }
    }
    let mut h = SymTMatrix::new(SymTMatrixSpace::new(n as i32, ir, jc));
    h.set_values(&v);
    h
}

fn solve(qp: &QpProblem, ws: &QpWarmStart, inertia_test: bool) -> QpSolution {
    let opts = QpOptions {
        use_schur_updates: true,
        schur_drop_inertia_test: inertia_test,
        max_iter: 500,
        ..QpOptions::default()
    };
    let mut s = ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    s.solve(qp, Some(ws), &opts).unwrap()
}

fn all_at_lower(n: usize, xl: &[f64]) -> QpWarmStart {
    let mut w = WorkingSet::cold(n, 0);
    for b in w.bounds.iter_mut() {
        *b = BoundStatus::AtLower;
    }
    QpWarmStart {
        x: xl.to_vec(),
        lambda_g: vec![],
        lambda_x: vec![0.0; n],
        working: w,
    }
}

/// A PD Hessian the caller could not certify (`Unknown`): every drop leaves
/// the reduced Hessian PD, so with the test on every drop keeps its update
/// and the solve factors once, where it used to factor once per drop.
#[test]
fn drops_on_an_uncertified_pd_hessian_take_the_update() {
    let n = 12;
    let h = tridiag(n, 2.0, 2.0);
    let a = GenTMatrix::new(GenTMatrixSpace::new(0, n as i32, vec![], vec![]));
    let g = vec![-1.0; n];
    let xl = vec![0.0; n];
    let xu = vec![10.0; n];
    let qp = QpProblem {
        n,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &[],
        bu: &[],
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Unknown,
    };
    let ws = all_at_lower(n, &xl);
    let on = solve(&qp, &ws, true);
    let off = solve(&qp, &ws, false);
    assert_eq!(on.status, QpStatus::Optimal);
    assert_eq!(off.status, QpStatus::Optimal);
    for (a, b) in on.x.iter().zip(off.x.iter()) {
        assert!((a - b).abs() < 1e-9, "{:?} vs {:?}", on.x, off.x);
    }
    // Every bound is released: the unconstrained minimizer is interior.
    assert!(on.x.iter().all(|&v| v > 0.1), "{:?}", on.x);
    // Off: one factorization per drop. On: drops ride the update layer.
    assert!(
        off.stats.n_refactor >= n as u32,
        "off: {} refactors",
        off.stats.n_refactor
    );
    assert!(
        on.stats.n_refactor * 3 <= off.stats.n_refactor,
        "on: {} refactors / {} updates, off: {} refactors",
        on.stats.n_refactor,
        on.stats.n_schur_updates,
        off.stats.n_refactor
    );
    assert!(on.stats.n_schur_updates >= n as u32 - 2);
}

/// A genuinely indefinite Hessian: releasing the last variable's bound
/// exposes its negative curvature, so that drop's updated KKT carries one
/// negative eigenvalue too many and must refactorize under inertia control;
/// the variable then runs to its upper bound. Same answer either way.
#[test]
fn a_drop_exposing_negative_curvature_still_refactors() {
    let n = 8;
    let h = tridiag(n, 2.0, -1.0);
    let a = GenTMatrix::new(GenTMatrixSpace::new(0, n as i32, vec![], vec![]));
    let mut g = vec![-1.0; n];
    g[n - 1] = -0.5;
    let xl = vec![0.0; n];
    let xu = vec![3.0; n];
    let qp = QpProblem {
        n,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &[],
        bu: &[],
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Indefinite,
    };
    let ws = all_at_lower(n, &xl);
    let on = solve(&qp, &ws, true);
    let off = solve(&qp, &ws, false);
    assert_eq!(on.status, off.status);
    assert_eq!(on.status, QpStatus::Optimal);
    for (a, b) in on.x.iter().zip(off.x.iter()) {
        assert!((a - b).abs() < 1e-8, "{:?} vs {:?}", on.x, off.x);
    }
    assert!((on.x[n - 1] - 3.0).abs() < 1e-12, "{:?}", on.x);
    // Some drops kept the update, and not all of them could.
    assert!(on.stats.n_refactor < off.stats.n_refactor);
    assert!(
        on.stats.n_refactor >= 2,
        "{} refactors",
        on.stats.n_refactor
    );
}
