//! A row emptied by doubleton aggregation must not read as a proof of
//! infeasibility at rounding level.
//!
//! Aggregation substitutes `x_j = mult·y + shift` into every row. When it
//! pins a column outright (`mult = 0`) a row on pinned columns alone empties
//! to `0 (cmp) h − Σ aⱼ·shiftⱼ`, a *computed* difference that carries rounding
//! of the size of the terms that cancelled. The row is handed to the next
//! presolve round, whose empty-row screen compares against exact zero — so a
//! feasible model was rejected at iteration 0 with
//! `empty inequality row (… 0 <= -1.1e-16)`. Found on the MIPLIB 2017 Infeas
//! benchmark, where `dws008-03` and `dws012-02` were "proved" infeasible this
//! way (`0 <= -1.33e-15`); they are infeasible, but not for that reason.
//!
//! `build_rows` already judges a row emptied by *fixing* against
//! `EMPTY_ROW_TOL × (1 + scale)` (gh#496); aggregation now does the same.

use pounce_convex::presolve::{PresolveOutcome, presolve, solve_with_presolve};
use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `x0 − x1 = c1`, `x0 + x1 = c2` pin both columns through aggregation;
/// `x0 + 3·x1 ≤ h` then empties. `x2, x3` keep a nontrivial reduced QP.
fn model(c1: f64, c2: f64, h: f64) -> QpProblem {
    QpProblem {
        n: 4,
        p_lower: vec![Triplet::new(2, 2, 1.0), Triplet::new(3, 3, 1.0)],
        c: vec![0.0, 0.0, -1.0, -1.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, -1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 1.0),
        ],
        b: vec![c1, c2],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 3.0),
            Triplet::new(1, 2, 1.0),
            Triplet::new(1, 3, 1.0),
        ],
        h: vec![h, 5.0],
        lb: vec![],
        ub: vec![],
    }
}

/// `h` is the row's activity at the pinned point, computed the way a
/// modeller's data pipeline would — so the row holds with equality.
fn tight_h(c1: f64, c2: f64) -> f64 {
    let x1 = (c2 - c1) / 2.0;
    let x0 = c1 + x1;
    x0 + 3.0 * x1
}

#[test]
fn rounding_residual_on_an_aggregated_empty_row_is_feasible() {
    // (0.1, 0.7) leaves a residual of −1.1e-16 on the emptied row; before
    // the fix presolve returned `Infeasible` here.
    let prob = model(0.1, 0.7, tight_h(0.1, 0.7));
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => {
            assert!(
                ps.stats().aggregated_vars > 0,
                "the aggregation path must be the one exercised"
            );
            assert!(ps.discarded_infeasibility().is_none());
        }
        PresolveOutcome::Infeasible(t) => panic!("false infeasibility proof: {t}"),
        PresolveOutcome::Unbounded => panic!("unexpected unbounded"),
    }
    let with = solve_with_presolve(&prob, |r| solve_qp_ipm(r, &QpOptions::default(), backend));
    let without = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(with.status, QpStatus::Optimal);
    assert_eq!(without.status, QpStatus::Optimal);
    assert!(
        (with.obj - without.obj).abs() < 1e-6,
        "{} vs {}",
        with.obj,
        without.obj
    );
}

#[test]
fn every_rounding_variant_survives_presolve() {
    let vals = [0.1, 0.2, 0.3, 0.6, 0.7, 1.1, 1.8, 2.9];
    for &c1 in &vals {
        for &c2 in &vals {
            let prob = model(c1, c2, tight_h(c1, c2));
            if let PresolveOutcome::Infeasible(t) = presolve(&prob) {
                panic!("({c1}, {c2}): false infeasibility proof: {t}");
            }
        }
    }
}

/// The other branch: a genuine conflict far above rounding must still be
/// reported, so the snap does not swallow real infeasibility.
#[test]
fn a_real_conflict_on_an_aggregated_empty_row_is_still_infeasible() {
    let prob = model(0.1, 0.7, tight_h(0.1, 0.7) - 1e-3);
    assert!(
        matches!(presolve(&prob), PresolveOutcome::Infeasible(_)),
        "a 1e-3 violation is a real conflict"
    );
}
