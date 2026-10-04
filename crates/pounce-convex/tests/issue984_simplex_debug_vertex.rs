//! gh#984 item 6 (review): `POUNCE_SIMPLEX_DEBUG` must not change the vertex
//! crossover returns. The debug switch used to steer the final pivot (the
//! fix recomputes `x_B` after it), and the item-6 test only checked the
//! vertex's KKT error with the switch off.
//!
//! Its own test binary because it sets a process-wide environment variable:
//! no other test may run concurrently with the flip.
#![allow(clippy::needless_range_loop)]

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// The issue's transportation LP (costs in $/1000 units, so `c ~ 1e3`).
fn transport() -> QpProblem {
    let supply = [120.0, 90.0, 70.0];
    let demand = [60.0, 80.0, 50.0, 70.0];
    let cost = [
        [1.2, 2.1, 3.0, 2.4],
        [1.8, 0.9, 1.6, 2.0],
        [2.6, 1.9, 1.1, 1.5],
    ];
    let mut g = Vec::new();
    let mut h = Vec::new();
    for i in 0..3 {
        for j in 0..4 {
            g.push(Triplet::new(i, 4 * i + j, 1.0));
        }
        h.push(supply[i]);
    }
    for j in 0..4 {
        for i in 0..3 {
            g.push(Triplet::new(3 + j, 4 * i + j, -1.0));
        }
        h.push(-demand[j]);
    }
    QpProblem {
        n: 12,
        p_lower: vec![],
        c: (0..12).map(|k| 1000.0 * cost[k / 4][k % 4]).collect(),
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![0.0; 12],
        ub: vec![],
    }
}

#[test]
fn simplex_debug_switch_does_not_change_the_vertex() {
    let p = transport();
    let opts = QpOptions {
        crossover: true,
        ..QpOptions::default()
    };
    // SAFETY: this binary holds a single test, so nothing reads the
    // environment concurrently with these writes.
    unsafe { std::env::remove_var("POUNCE_SIMPLEX_DEBUG") };
    let off = solve_qp_ipm(&p, &opts, backend);
    unsafe { std::env::set_var("POUNCE_SIMPLEX_DEBUG", "1") };
    let on = solve_qp_ipm(&p, &opts, backend);
    unsafe { std::env::remove_var("POUNCE_SIMPLEX_DEBUG") };
    assert_eq!(off.status, QpStatus::Optimal);
    assert_eq!(on.status, off.status);
    assert_eq!(on.x, off.x, "the debug switch moved the vertex");
    assert_eq!(on.obj.to_bits(), off.obj.to_bits());
    assert!(off.kkt_residuals(&p).kkt_error() < 1e-9);
}
