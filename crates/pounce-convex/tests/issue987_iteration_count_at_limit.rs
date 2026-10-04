//! gh#987 item 5: a run that exhausts `max_iter` reported `max_iter - 1`.
//!
//! The loop recorded `iters = it` at the *top* of each pass, so after the last
//! pass the count was one short; the NLP arm reports the number of iterations
//! actually run and the two engines disagreed on the same model.

use pounce_convex::{QpOptions, QpProblem, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// A small inequality-constrained QP (economic-dispatch shaped) that cannot
/// converge in a couple of interior-point iterations.
fn dispatch() -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: vec![
            Triplet::new(0, 0, 0.010),
            Triplet::new(1, 1, 0.016),
            Triplet::new(2, 2, 0.024),
        ],
        c: vec![12.0, 10.5, 9.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0),
        ],
        b: vec![600.0],
        g: vec![
            Triplet::new(0, 0, 0.95),
            Triplet::new(0, 1, 0.6),
            Triplet::new(0, 2, 0.4),
        ],
        h: vec![390.0],
        lb: vec![100.0, 50.0, 50.0],
        ub: vec![400.0, 300.0, 150.0],
    }
}

#[test]
fn a_run_that_hits_max_iter_reports_max_iter() {
    for k in [1usize, 2, 3] {
        let opts = QpOptions {
            max_iter: k,
            ..QpOptions::default()
        };
        let sol = solve_qp_ipm(&dispatch(), &opts, backend);
        // If the point happens to certify after the fact the status is
        // upgraded, but the count is still the iterations run.
        assert_eq!(
            sol.iters, k,
            "max_iter={k}: reported {} iterations (status {:?})",
            sol.iters, sol.status
        );
    }
}
