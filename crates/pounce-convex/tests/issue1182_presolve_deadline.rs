//! gh#1182: the presolve fixpoint stops between layers once the caller's
//! deadline has passed, and what it hands back is still a sound reduction.
//!
//! Before this, nothing in the fixpoint looked at the clock. The convex
//! solver's deadline scope opens only once presolve returns, so on a large LP
//! every one of up to 32 layers ran first, each a pass over every nonzero and
//! a clone of the problem. On a synthetic 400 000-column banded LP that was
//! 13.9 s spent before the solver started, against `max_wall_time = 5`.
//!
//! Both branches of the new check are pinned here. `contracting_pair` takes
//! many layers to reach its fixpoint (it is the gh #527 model), so a deadline
//! that has already passed must stop it after the first layer, and one far in
//! the future must leave the reduction exactly as `presolve` makes it.

use std::time::{Duration, Instant};

use pounce_convex::presolve::{FixpointExit, PresolveOutcome, presolve, presolve_until};
use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// The gh #527 model: two rows that tighten each other's upper bound by a
/// factor of 0.999 per round, so the fixpoint takes many layers, plus
/// `min x₂ s.t. x₂ ≥ 10` so the reduced problem has a known optimum.
fn contracting_pair() -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![0.0, 0.0, 1.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, -0.999),
            Triplet::new(1, 1, 1.0),
            Triplet::new(1, 0, -0.999),
            Triplet::new(2, 2, -1.0),
        ],
        h: vec![1.0, 1.0, -10.0],
        lb: vec![0.0, 0.0, f64::NEG_INFINITY],
        ub: vec![1e6, 1e6, f64::INFINITY],
    }
}

/// A deadline that has already passed stops the loop after its first layer,
/// says so, and still returns a reduction that solves and lifts back to the
/// original problem's optimum.
#[test]
fn a_passed_deadline_stops_the_fixpoint_after_one_layer() {
    let prob = contracting_pair();

    // Precondition: unbounded by a deadline, this model takes more than one
    // layer. Otherwise stopping "after one layer" would prove nothing.
    let PresolveOutcome::Reduced(full) = presolve(&prob) else {
        panic!("feasible bounded problem");
    };
    assert!(
        full.stats().rounds > 1,
        "precondition: the fixpoint must need several layers, stats={:?}",
        full.stats()
    );

    let PresolveOutcome::Reduced(ps) = presolve_until(&prob, Some(Instant::now())) else {
        panic!("feasible bounded problem");
    };
    let st = ps.stats();
    assert_eq!(st.exit, FixpointExit::Deadline, "stats={st:?}");
    assert_eq!(st.rounds, 1, "stopped after the first layer, stats={st:?}");

    // The partial reduction is still a reduction of *this* problem.
    let red = solve_qp_ipm(&ps.reduced, &QpOptions::default(), backend);
    let sol = ps.postsolve(&red);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.obj - 10.0).abs() < 1e-6, "objective {}", sol.obj);
    assert_eq!(sol.x.len(), prob.n);
}

/// A deadline nowhere near leaves the reduction exactly as `presolve` makes
/// it: the same exit and the same number of layers.
#[test]
fn a_distant_deadline_changes_nothing() {
    let prob = contracting_pair();
    let PresolveOutcome::Reduced(plain) = presolve(&prob) else {
        panic!("feasible bounded problem");
    };
    let far = Instant::now() + Duration::from_secs(3600);
    let PresolveOutcome::Reduced(timed) = presolve_until(&prob, Some(far)) else {
        panic!("feasible bounded problem");
    };
    let (a, b) = (plain.stats(), timed.stats());
    assert_eq!(a.exit, b.exit);
    assert_eq!(a.rounds, b.rounds);
    assert_eq!(a.reduced_vars, b.reduced_vars);
    assert_eq!(a.reduced_rows, b.reduced_rows);
    assert_eq!(a.tightened_bounds, b.tightened_bounds);
    assert_eq!(plain.reduced.lb, timed.reduced.lb);
    assert_eq!(plain.reduced.ub, timed.reduced.ub);
}
