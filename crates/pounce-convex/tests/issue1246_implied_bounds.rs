//! gh#1246: presolve stops handing the convex solver bound sides that its
//! rows already imply.
//!
//! The interior point turns every finite bound side into a barrier term and a
//! KKT row, so a side the rows imply only costs work. One fixture per branch
//! of `relax_implied_bounds`:
//!
//! * a side bound tightening *derived* (infinite as written) is removed;
//! * a *declared* side the rows imply is removed;
//! * two sides that each imply the other: exactly one goes (no circular proof);
//! * a row that became a bound and was dropped keeps its bound (nothing left
//!   to imply it);
//! * a tiny coefficient does not license removing a declared bound;
//! * a bound nothing implies is kept.
//!
//! Each also solves with and without presolve and compares the answers, so a
//! removal that changed the feasible set fails here rather than in a sweep.

use pounce_convex::presolve::{PresolveOutcome, presolve, solve_with_presolve};
use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

const INF: f64 = f64::INFINITY;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn reduced(prob: &QpProblem) -> pounce_convex::presolve::Presolve {
    match presolve(prob) {
        PresolveOutcome::Reduced(ps) => ps,
        _ => panic!("expected a reduction"),
    }
}

/// Presolved and plain solves agree, and the presolved point is feasible for
/// the *original* box and rows.
fn same_answer(prob: &QpProblem) {
    let o = QpOptions::default();
    let plain = solve_qp_ipm(prob, &o, backend);
    let pre = solve_with_presolve(prob, |q| solve_qp_ipm(q, &o, backend));
    assert_eq!(plain.status, QpStatus::Optimal);
    assert_eq!(pre.status, QpStatus::Optimal);
    assert!(
        (plain.obj - pre.obj).abs() <= 1e-6 * (1.0 + plain.obj.abs()),
        "plain {} vs presolved {}",
        plain.obj,
        pre.obj
    );
    for j in 0..prob.n {
        assert!(
            pre.x[j] >= prob.lb_of(j) - 1e-7,
            "x[{j}]={} below lb",
            pre.x[j]
        );
        assert!(
            pre.x[j] <= prob.ub_of(j) + 1e-7,
            "x[{j}]={} above ub",
            pre.x[j]
        );
    }
}

fn finite_sides(p: &QpProblem) -> usize {
    (0..p.n)
        .map(|j| usize::from(p.lb_of(j) > -1e20) + usize::from(p.ub_of(j) < 1e20))
        .sum()
}

/// `x ≥ 0` and `Σ x = 1` over three columns: bound tightening derives `x ≤ 1`
/// for every column, and before gh#1246 each became a new barrier term. The
/// equality still implies them, so none reaches the solver.
#[test]
fn tightened_sides_the_rows_imply_are_not_handed_to_the_solver() {
    let prob = QpProblem {
        n: 3,
        p_lower: (0..3).map(|j| Triplet::new(j, j, 1.0)).collect(),
        c: vec![1.0, -2.0, 0.5],
        a: (0..3).map(|j| Triplet::new(0, j, 1.0)).collect(),
        b: vec![1.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0; 3],
        ub: vec![INF; 3],
    };
    let ps = reduced(&prob);
    let st = ps.stats();
    assert!(st.tightened_bounds >= 3, "{st:?}");
    for j in 0..ps.reduced.n {
        assert!(ps.reduced.ub_of(j) >= 1e20, "derived ub[{j}] survived");
    }
    assert!(
        finite_sides(&ps.reduced) <= finite_sides(&prob),
        "presolve grew the bound count: {} > {}",
        finite_sides(&ps.reduced),
        finite_sides(&prob)
    );
    same_answer(&prob);
}

/// `0 ≤ x ≤ 1` with `Σ x = 1`: the declared upper bounds are implied (cpo3's
/// `redundant_bounds`). The lower bounds are not — with the upper sides gone
/// nothing bounds the rest of the sum — so they stay.
#[test]
fn declared_sides_the_rows_imply_are_removed_and_the_rest_kept() {
    let prob = QpProblem {
        n: 3,
        p_lower: (0..3).map(|j| Triplet::new(j, j, 1.0)).collect(),
        c: vec![1.0, -2.0, 0.5],
        a: (0..3).map(|j| Triplet::new(0, j, 1.0)).collect(),
        b: vec![1.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0; 3],
        ub: vec![1.0; 3],
    };
    let ps = reduced(&prob);
    assert_eq!(ps.stats().relaxed_bounds, 3, "{:?}", ps.stats());
    for j in 0..ps.reduced.n {
        assert!(ps.reduced.ub_of(j) >= 1e20);
        assert_eq!(ps.reduced.lb_of(j), 0.0);
    }
    same_answer(&prob);
}

/// `x ≤ y` and `y ≤ x` (two inequality rows), `0 ≤ x, y ≤ 1`: `y ≤ 1`
/// implies `x ≤ 1` and `x ≤ 1` implies `y ≤ 1`. Removing both on those two
/// proofs would be circular and leave the pair unbounded above; exactly one
/// upper side must survive. The quadratic objective keeps the columns out of
/// the dominated-column rule, and inequalities keep them out of doubleton
/// aggregation, so the pass sees the pair.
#[test]
fn two_sides_that_imply_each_other_lose_only_one() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-5.0, -5.0], // pushes both up: the surviving bound binds
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, -1.0),
            Triplet::new(1, 0, -1.0),
            Triplet::new(1, 1, 1.0),
        ],
        h: vec![0.0, 0.0],
        lb: vec![0.0; 2],
        ub: vec![1.0; 2],
    };
    let ps = reduced(&prob);
    assert_eq!(ps.reduced.n, 2, "{:?}", ps.stats());
    let ubs = (0..ps.reduced.n)
        .filter(|&j| ps.reduced.ub_of(j) < 1e20)
        .count();
    assert_eq!(ubs, 1, "{:?}: {:?}", ps.stats(), ps.reduced.ub);
    same_answer(&prob);
}

/// A singleton row `x ≤ 1` becomes the bound and is then dropped as
/// redundant: the bound is the constraint now, and no row is left to imply
/// it, so it must reach the solver.
#[test]
fn a_row_that_became_a_bound_keeps_it() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![-10.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0)],
        h: vec![1.0],
        lb: vec![-INF],
        ub: vec![INF],
    };
    let ps = reduced(&prob);
    assert_eq!(ps.reduced.m_ineq(), 0);
    assert_eq!(ps.reduced.ub_of(0), 1.0);
    same_answer(&prob);
}

/// `1e-6·x + y ≤ 1`, `0 ≤ y`, `0 ≤ x ≤ 1e6`: the row implies `x ≤ 1e6`, but
/// only through a coefficient a millionth of the row's largest, so a row
/// residual at solver tolerance would let `x` overshoot the declared bound by
/// a million times that. The bound stays.
#[test]
fn a_tiny_coefficient_does_not_remove_a_declared_bound() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![-1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1e-6), Triplet::new(0, 1, 1.0)],
        h: vec![1.0],
        lb: vec![0.0, 0.0],
        ub: vec![1e6, INF],
    };
    let ps = reduced(&prob);
    assert_eq!(ps.reduced.ub_of(0), 1e6, "{:?}", ps.stats());
    same_answer(&prob);
}

/// `x + y ≤ 1.5`, `0 ≤ x, y ≤ 1`: the row allows either variable up to 1.5,
/// so it implies neither upper bound, and nothing is removed. (Quadratic, so
/// the dominated-column rule does not fix the columns first.)
#[test]
fn a_bound_nothing_implies_is_kept() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-1.0, -2.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![1.5],
        lb: vec![0.0; 2],
        ub: vec![1.0; 2],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => {
            assert_eq!(ps.stats().relaxed_bounds, 0);
            assert_eq!(finite_sides(&ps.reduced), finite_sides(&prob));
        }
        _ => panic!("feasible"),
    }
    same_answer(&prob);
}
