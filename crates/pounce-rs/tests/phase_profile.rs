//! Phase-profile telemetry: when does the active set stop moving?
//!
//! The per-iteration record carries two counts —
//! `IterRecord::active_bounds` and `IterRecord::active_set_changes` —
//! whose purpose is to split a solve into "iterations spent finding the
//! active set" and "iterations spent converging on it". These tests pin
//! the properties that reading is built on; what they deliberately do
//! not pin is any particular count, which moves with the model, the
//! scaling and the barrier schedule.

use pounce_rs::prelude::*;

/// min (x0-1)^2 + (x1-2)^2  s.t. x0 + x1 == 3.
///
/// With the bounds used below the unconstrained minimizer of each term
/// is inside the box, so no bound binds at the solution.
struct Quad;
impl Problem for Quad {
    fn objective(&self, x: &[f64]) -> f64 {
        (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
    }
    fn n_constraints(&self) -> usize {
        1
    }
    fn constraints(&self, x: &[f64], g: &mut [f64]) {
        g[0] = x[0] + x[1];
    }
}

/// min (x0-3)^2 + (x1-3)^2, box `[0, 1]^2` — both bounds bind at the
/// solution, and neither binds at the interior start the solver picks.
struct PushedIntoCorner;
impl Problem for PushedIntoCorner {
    fn objective(&self, x: &[f64]) -> f64 {
        (x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2)
    }
    fn n_constraints(&self) -> usize {
        0
    }
    fn constraints(&self, _x: &[f64], _g: &mut [f64]) {}
}

#[test]
fn the_first_captured_iteration_reports_no_change_count() {
    let (sol, iters) = with_iter_capture(|| {
        Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
            .constraint_bounds(&[3.0], &[3.0])
            .solve()
    });
    assert!(sol.success, "status = {:?}", sol.status);
    assert!(!iters.is_empty(), "no iteration records captured");

    // "Nothing to compare against" is a different reading from "nothing
    // changed", and the record keeps them apart: the count is absent on
    // the first row and present on every later one.
    assert_eq!(
        iters[0].active_set_changes, None,
        "first row must not claim a measured change count"
    );
    assert!(
        iters[0].active_bounds.is_some(),
        "the activity count itself is measurable on the first row"
    );
    for row in &iters[1..] {
        assert!(
            row.active_set_changes.is_some(),
            "iteration {} lost its change count",
            row.iter
        );
    }
}

#[test]
fn a_problem_with_no_bounds_reports_an_empty_active_set() {
    // Nothing to classify, so the count is a measured zero rather than
    // an absent field — the guard against a fabricated number.
    let (sol, iters) = with_iter_capture(|| {
        Nlp::new(Quad)
            .x0(&[0.0, 0.0])
            .constraint_bounds(&[3.0], &[3.0])
            .solve()
    });
    assert!(sol.success, "status = {:?}", sol.status);
    for row in &iters {
        assert_eq!(row.active_bounds, Some(0), "iteration {}", row.iter);
        assert!(
            matches!(row.active_set_changes, None | Some(0)),
            "iteration {} invented a change with no bounds to change",
            row.iter
        );
    }
}

#[test]
fn the_counts_stay_inside_the_number_of_bound_indices() {
    // Two variables, two-sided box, no constraint slacks: four bound
    // indices in total, so neither count can exceed four.
    let (sol, iters) = with_iter_capture(|| {
        Nlp::new(PushedIntoCorner)
            .var_bounds(&[0.0, 0.0], &[1.0, 1.0])
            .solve()
    });
    assert!(sol.success, "status = {:?}", sol.status);
    for row in &iters {
        let Some(active) = row.active_bounds else {
            panic!("iteration {} lost its activity count", row.iter);
        };
        assert!(
            (0..=4).contains(&active),
            "iteration {} reports {active} of 4 bound indices active",
            row.iter
        );
        if let Some(changed) = row.active_set_changes {
            assert!(
                (0..=4).contains(&changed),
                "iteration {} reports {changed} of 4 indices moving",
                row.iter
            );
        }
    }
}

#[test]
fn the_active_set_settles_before_the_solve_ends() {
    // The reading the telemetry exists for: churn resolves to zero
    // while iterations continue, so a run divides into an approach and
    // an endgame. Asserted as a property of the tail rather than as a
    // crossover index, which moves with the model and the schedule.
    let (sol, iters) = with_iter_capture(|| {
        Nlp::new(PushedIntoCorner)
            .var_bounds(&[0.0, 0.0], &[1.0, 1.0])
            .solve()
    });
    assert!(sol.success, "status = {:?}", sol.status);
    assert!(
        iters.len() >= 4,
        "too short to have a tail: {}",
        iters.len()
    );

    let Some(last) = iters.last() else {
        panic!("no iteration records captured");
    };
    assert_eq!(
        last.active_set_changes,
        Some(0),
        "the active set was still moving at the final iteration"
    );
    // Both upper bounds bind at (1, 1), and the solver starts strictly
    // inside the box, so the set it settles on is not the one it began
    // with.
    assert_eq!(
        last.active_bounds,
        Some(2),
        "expected both upper bounds active at the solution"
    );
    assert!(
        iters
            .iter()
            .any(|r| r.active_set_changes.is_some_and(|c| c > 0)),
        "no iteration ever moved an index, so the run had no approach phase"
    );
}

#[test]
fn measuring_the_profile_does_not_move_the_solve() {
    // The load-bearing property: the fingerprint is computed inside the
    // per-iteration event, which is skipped entirely when nothing
    // consumes it, and it reads calculated quantities without writing
    // any solver state. So attaching a consumer must not change the
    // trajectory -- otherwise the instrument would be reporting on a
    // run that only exists while it is watching.
    let plain = Nlp::new(PushedIntoCorner)
        .var_bounds(&[0.0, 0.0], &[1.0, 1.0])
        .solve();
    let (watched, iters) = with_iter_capture(|| {
        Nlp::new(PushedIntoCorner)
            .var_bounds(&[0.0, 0.0], &[1.0, 1.0])
            .solve()
    });

    assert!(!iters.is_empty(), "the watched run captured nothing");
    assert_eq!(
        plain.stats.iteration_count, watched.stats.iteration_count,
        "iteration count moved when the profile was measured"
    );
    assert_eq!(
        plain.objective, watched.objective,
        "objective moved when the profile was measured"
    );
    assert_eq!(
        plain.x, watched.x,
        "solution moved when the profile was measured"
    );
}
