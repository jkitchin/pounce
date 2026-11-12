//! `qp_crossover=yes` must bound what it spends after declining a vertex.
//!
//! NETLIB SHARE1B (`lp_share1b`, 225 columns, 117 rows), driven through the
//! library as below. The revised simplex reaches its optimal basis (173
//! superbasics pushed, 128 pivots) and the never-regress gate declines the
//! vertex against the interior iterate's KKT error of `1.1e-9`. Crossover then
//! runs the active-set bridge from the interior iterate. On `b0328335` that
//! was two solves (warm, then a cold retry), each bounded only per pass of
//! pounce-qp's l1-elastic phase 1; they built two backends, and the test took
//! 29.6 s. Both returned the interior iterate.
//!
//! Each bridge solve now has a whole-solve cap of `CROSSOVER_MAX_ITER` (1000)
//! working-set changes, and there is no cold retry. Here the simplex basis
//! start does not run (one backend is built), the solve from the interior
//! iterate spends all 1000, the test takes about 2 s, and the interior iterate is
//! still the answer. The cap is the bound, so the test pins it, through the
//! crossover record. The record gained the count and the crate the public
//! constant in the same change, so this test does not build on the parent
//! commit.

use std::path::Path;

use pounce_cli::qp_extract::{BoundRelax, extract_qp};
use pounce_convex::crossover::{CROSSOVER_MAX_ITER, clear_report, maybe_crossover, take_report};
use pounce_convex::{QpOptions, QpStatus, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

#[test]
fn the_bridge_after_a_declined_vertex_stays_within_its_budget() {
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lp_share1b.nl"
    ));
    let nl = pounce_nl::nl_reader::read_nl_file(path).expect("read lp_share1b.nl");
    let qp = extract_qp(&nl, BoundRelax::NONE).expect("SHARE1B is an LP");

    // The interior iterate crossover starts from (crossover is off by default).
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

    // Preconditions: the simplex returned a vertex (its pivot counts are only
    // recorded when it does), the gate declined it, and the bridge then ran.
    // If one fails, this LP no longer exercises the bridge after a decline:
    // find another degenerate LP that does (a declined simplex vertex whose
    // bridge solve needs elastic phase 1) rather than loosening the checks.
    assert!(
        report.pivots_push + report.pivots_phase1 + report.pivots_phase2 > 0,
        "precondition: the simplex returned no vertex: {report:?}"
    );
    assert!(
        !report.accepted,
        "precondition: a vertex was accepted: {report:?}"
    );
    assert!(
        report.bridge_working_set_changes > 0,
        "precondition: the bridge did not run: {report:?}"
    );

    // One backend per bridge solve: the pinned simplex basis when the vertex
    // has one, then one solve from the interior iterate. No cold retry.
    assert!(
        backends_built <= 2,
        "{backends_built} bridge solves: {report:?}"
    );
    // The record sums the solves; each is capped on its own.
    assert!(
        report.bridge_working_set_changes <= backends_built * CROSSOVER_MAX_ITER,
        "the bridge outran its whole-solve budget over {backends_built} solve(s): {report:?}"
    );
    assert_eq!(
        out.x, interior.x,
        "a declined crossover must leave the interior iterate"
    );
    assert_eq!(out.status, interior.status);
}
