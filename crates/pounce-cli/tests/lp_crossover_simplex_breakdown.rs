//! A simplex breakdown sends crossover to the active-set bridge, and a vertex
//! that is not finite is a breakdown.
//!
//! `farkas_far_feasible_no_box_c0` (18 columns, 18 rows, every column free,
//! zero objective): the crossover push moves each of the 18 free superbasics
//! along its column until a basic variable blocks. For one of them no basic
//! blocks (the move is a ray of the feasible set), so the step is infinite:
//! that column becomes `inf` and the basics it touches `inf` or NaN. Phases 1
//! and 2 then see no infeasibility and no entering column (every comparison
//! with a NaN is false), and the vertex read back is NaN in all 18
//! coordinates. That unblocked push is a simplex defect of its own, left for a
//! follow-up. On `b0328335` the NaN was returned as a vertex: the never-regress
//! gate's KKT error came out `0` (`f64::max` drops a NaN), the vertex was
//! accepted, and the convex arm reported a numerical failure, so the CLI
//! re-solved the LP on the NLP path. A vertex
//! that is not finite is now a simplex breakdown: the bridge runs from the
//! interior iterate (one solve, one backend), and here its vertex is declined,
//! so the interior iterate is the answer.

use std::path::Path;

use pounce_cli::qp_extract::{BoundRelax, extract_qp};
use pounce_convex::crossover::{clear_report, maybe_crossover, take_report};
use pounce_convex::{QpOptions, QpStatus, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

#[test]
fn a_non_finite_simplex_vertex_is_a_breakdown_and_runs_the_bridge() {
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/farkas_far_feasible_no_box_c0.nl"
    ));
    let nl = pounce_nl::nl_reader::read_nl_file(path).expect("read the fixture");
    let qp = extract_qp(&nl, BoundRelax::NONE).expect("the fixture is an LP");

    let interior = solve_qp_ipm(&qp, &QpOptions::default(), backend);
    assert_eq!(interior.status, QpStatus::Optimal, "precondition");

    let mut backends_built = 0_u32;
    let mut counting_backend = || {
        backends_built += 1;
        backend()
    };
    let crossover_on = QpOptions {
        crossover: true,
        ..QpOptions::default()
    };
    clear_report();
    let out = maybe_crossover(&qp, interior.clone(), &crossover_on, &mut counting_backend);
    let report = take_report().expect("crossover ran");

    assert!(
        out.x.iter().all(|v| v.is_finite()),
        "crossover returned a non-finite point: {report:?}"
    );
    // A breakdown records no simplex counts; the bridge then built one
    // backend for its one solve from the interior iterate.
    assert_eq!(
        report.pivots_push + report.pivots_phase1 + report.pivots_phase2,
        0,
        "the simplex vertex was judged: {report:?}"
    );
    assert_eq!(backends_built, 1, "{report:?}");
    assert_eq!(report.engine, "active-set", "{report:?}");
    assert_eq!(out.status, QpStatus::Optimal);
}
