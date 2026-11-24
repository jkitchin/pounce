//! gh#1249 / gh#1265: the convex deadline reaches FERAL's analysis.
//!
//! The convex deadline scope (`deadline::with_deadline`) installs
//! `pounce_feral::factor_deadline`, which FERAL's symbolic analysis polls as a
//! `feral::Interrupt` (feral >= 0.25). A deadline that has passed by then
//! stops the analysis, so the numeric factor is never started — with or
//! without the `report_symbolic` pre-analysis. (Before gh#1265 only the
//! pre-analysis checked it, between the phases; without it the factor ran.)
//!
//! The backend factory sleeps past the limit, so the deadline is crossed
//! after the drivers' own pre-factorization check and before FERAL's
//! analysis starts. A summary sink shows whether a numeric factorization ran.
//! Every leg ends `TimeLimit`.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::{FeralConfig, FeralSolverInterface};
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_linsol::summary::LinearSolverSummary;

/// `min Σ x_j  s.t.  Σ x_j = 1,  0 ≤ x ≤ 1`.
fn lp(n: usize) -> QpProblem {
    QpProblem {
        n,
        p_lower: Vec::new(),
        c: vec![1.0; n],
        a: (0..n).map(|j| Triplet::new(0, j, 1.0)).collect(),
        b: vec![1.0],
        g: Vec::new(),
        h: Vec::new(),
        lb: vec![0.0; n],
        ub: vec![1.0; n],
    }
}

const LIMIT: Duration = Duration::from_millis(50);

#[test]
fn a_deadline_crossed_during_the_analysis_skips_the_numeric_factor() {
    for use_hsde in [true, false] {
        for report in [true, false] {
            let sink = Arc::new(Mutex::new(LinearSolverSummary::default()));
            let backend_sink = Arc::clone(&sink);
            let opts = QpOptions {
                use_hsde,
                time_limit: Some(LIMIT),
                ..QpOptions::default()
            };
            let sol = solve_qp_ipm(&lp(200), &opts, move || {
                std::thread::sleep(LIMIT * 3);
                Box::new(
                    FeralSolverInterface::with_config(FeralConfig {
                        report_symbolic: report,
                        ..FeralConfig::default()
                    })
                    .with_summary_sink(Arc::clone(&backend_sink)),
                ) as Box<dyn SparseSymLinearSolverInterface>
            });
            let ctx = format!("use_hsde = {use_hsde}, report_symbolic = {report}");
            assert_eq!(sol.status, QpStatus::TimeLimit, "{ctx}");
            let factored = sink.lock().unwrap().last_nnz_l.is_some();
            assert!(!factored, "{ctx}: a numeric factor ran");
        }
    }
}

#[test]
fn the_pre_analysis_does_not_change_an_unhurried_solve() {
    for use_hsde in [true, false] {
        let opts = QpOptions {
            use_hsde,
            ..QpOptions::default()
        };
        let solve = |report: bool| {
            solve_qp_ipm(&lp(200), &opts, move || {
                Box::new(FeralSolverInterface::with_config(FeralConfig {
                    report_symbolic: report,
                    ..FeralConfig::default()
                })) as Box<dyn SparseSymLinearSolverInterface>
            })
        };
        let (off, on) = (solve(false), solve(true));
        assert_eq!(off.status, QpStatus::Optimal, "use_hsde = {use_hsde}");
        assert_eq!(on.status, off.status);
        assert_eq!(on.iters, off.iters);
        assert_eq!(on.x, off.x, "use_hsde = {use_hsde}");
    }
}
