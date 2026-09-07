//! Where the convex arm stands on gh#928 -- measured, and not where
//! it was assumed to stand.
//!
//! The convex arm reaches `step_along_path` through
//! `QpSensitivity::parametric_step_path`, and it passes `weak_rows`
//! as `&[]` unconditionally:
//!
//! ```text
//! step_along_path(&bs, .., max_iter, &[], &[], &[], RELEASE_FLOOR)
//! //                                          ^^^ weak_rows
//! ```
//!
//! On the NLP arm an empty `weak_rows` is exactly the gh#928 hazard:
//! the classifier certified nothing, so a bound the walk marks
//! base-active is neither enforced by the factorization nor watched by
//! the reach scan, and the answer runs outside the box. Reading that
//! `&[]` and expecting the same hazard here is the natural inference.
//! It is wrong, and this file is the measurement that says so.
//!
//! # The two regimes, and why neither leaks
//!
//! The arm decides activity with a **hard threshold on the
//! multiplier** -- the `1e-7` handed to `QpSensitivity::build` -- and
//! a bound it calls active is *pinned*, not damped. There is no
//! order-one `Sigma` sitting between "enforced" and "ignored", which
//! is the crack gh#928 falls through. So:
//!
//! * **Pinned** (`z_ub` above the threshold): the variable does not
//!   move at all, `dx0 == 0` to working precision. An answer that does
//!   not move cannot leave the box.
//! * **Free** (`z_ub` below it): the bound is not base-active, the
//!   reach scan watches it, and a step that presses into it records a
//!   breakpoint and stops there. Measured below: the plain step
//!   overshoots to `0.55` and the walk returns `0.5`.
//!
//! The box repair added for gh#928 is therefore **inert on this arm**,
//! and that is a fact worth a test rather than a comment: if the arm
//! ever moves to a barrier-style `Sigma`, the pinned regime stops
//! being pinned and the hazard becomes real here too. The first test
//! below is what would notice.
//!
//! # The cost of pinning, which is not zero
//!
//! Pinning freezes the variable at its **base point**, not at its
//! bound, so across the kink the answer is short by exactly the base
//! slack. Measured over five orders of `delta`, `err/slack` is
//! `1.0000` in every row. That is not a violation and not a wrong
//! active set -- it is the arm answering for the barrier problem at
//! the `mu` its solve stopped at, and the base point really is that
//! far from the bound. It is recorded here because it is invisible to
//! a box check by construction: the error points *into* the feasible
//! region.
//!
//! Note the floor. Below `delta = 1e-3` the base slack stops tracking
//! `delta` and settles at about `5.2e-5`, the distance the interior
//! point method's own convergence leaves; the error settles with it.
//! So the accuracy of a crossing step on this arm is set by the base
//! solve's tolerance, not by the perturbation.
//!
//! # The fixture
//!
//! ```text
//! min  ½(x0² + x1²)   s.t.  x0 + x1 = b,  0 ≤ x0 ≤ ½,  x1 ≥ 0
//! ```
//!
//! Unconstrained by the box the optimum splits `b` evenly, so
//!
//! ```text
//! b <= 1:  x = (b/2, b/2)
//! b >= 1:  x = (1/2, b - 1/2)
//! ```
//!
//! with the kink at `b = 1`, where `x0` reaches its upper bound.
//!
//! # What this file is NOT evidence about
//!
//! - **The conic arm.** Second-order-cone blocks are not `BoundRow`s;
//!   `convex_sens_release.rs` owns what release means there.
//! - **Constraint rows.** `G` rows carry no bound metadata here, so a
//!   limit written as a row is watched by nothing on this arm --
//!   gh#929. The NLP arm's half of that is fixed;
//!   `pounce-sensitivity/tests/issue_928_a_limit_written_as_a_row.rs`
//!   is the map a convex fix would follow.
//! - **Scaling.** Unscaled, like every other convex sensitivity test.
//! - **Magnitude.** Two variables. The largest convex fixture in the
//!   corpus is 534 columns and `benchmarks/qp` reaches 93 263.

use pounce_convex::QpOptions;
use pounce_convex::ipm::solve_qp_ipm;
use pounce_convex::qp::{QpProblem, QpStatus, Triplet};
use pounce_convex::sensitivity::QpSensitivity;
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn tri(row: usize, col: usize, val: f64) -> Triplet {
    Triplet { row, col, val }
}

/// `x0`'s upper bound, and the kink's location in `b`.
const UB0: f64 = 0.5;
const KINK: f64 = 2.0 * UB0;

fn kinked_qp(b: f64) -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0)],
        c: vec![0.0, 0.0],
        a: vec![tri(0, 0, 1.0), tri(0, 1, 1.0)],
        b: vec![b],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![UB0, f64::INFINITY],
    }
}

fn closed_form(b: f64) -> [f64; 2] {
    if b <= KINK {
        [b / 2.0, b / 2.0]
    } else {
        [UB0, b - UB0]
    }
}

/// Solve the perturbed problem outright, two orders tighter than the
/// base solve, and check it against the closed form so a wrong oracle
/// cannot certify a wrong walk.
fn oracle(b: f64) -> Vec<f64> {
    let opts = QpOptions {
        tol: 1e-11,
        ..Default::default()
    };
    let sol = solve_qp_ipm(&kinked_qp(b), &opts, backend);
    assert_eq!(sol.status, QpStatus::Optimal, "the oracle must converge");
    let want = closed_form(b);
    for i in 0..2 {
        assert!(
            (sol.x[i] - want[i]).abs() < 1e-6,
            "the oracle disagrees with the closed form at b={b}: {:?} vs {want:?}",
            sol.x,
        );
    }
    sol.x
}

fn base_sens(b: f64) -> (QpSensitivity, Vec<f64>) {
    let prob = kinked_qp(b);
    let opts = QpOptions::default();
    let sol = solve_qp_ipm(&prob, &opts, backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    let x = sol.x.clone();
    match QpSensitivity::build(&prob, &sol, &opts, 1e-7, backend) {
        Ok(s) => (s, x),
        Err(e) => panic!("the fixture must build a sensitivity, got {e:?}"),
    }
}

/// Bases in the **pinned** regime: `z_ub` above `QpSensitivity`'s
/// activity threshold. Chosen by measurement, not by formula -- the
/// slack floors at the base solve's own convergence below `1e-3`.
const PINNED_DELTAS: [f64; 5] = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6];

/// Bases in the **free** regime, where the reach scan owns the bound.
const FREE_DELTAS: [f64; 4] = [0.15, 0.12, 0.1, 0.08];

/// The step, at each base: large enough that truth is past the kink.
const DP: f64 = 0.2;

/// The threshold handed to `QpSensitivity::build` throughout this file,
/// and the thing that separates the two regimes.
const ACTIVITY_TOL: f64 = 1e-7;

fn slack_and_z(b: f64) -> (f64, f64) {
    let sol = solve_qp_ipm(&kinked_qp(b), &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    (UB0 - sol.x[0], sol.z_ub[0])
}

/// The regimes are real and the grids land in them. Without this every
/// test below could be measuring the same regime twice.
#[test]
fn the_two_grids_straddle_the_activity_threshold() {
    for delta in PINNED_DELTAS {
        let (slack, z) = slack_and_z(KINK - delta);
        assert!(
            z > ACTIVITY_TOL,
            "delta={delta} was meant to be in the pinned regime but \
             z_ub = {z:e} is below the {ACTIVITY_TOL:e} threshold",
        );
        assert!(slack > 0.0, "delta={delta}: base must be inside the box");
    }
    for delta in FREE_DELTAS {
        let (_, z) = slack_and_z(KINK - delta);
        assert!(
            z < ACTIVITY_TOL,
            "delta={delta} was meant to be in the free regime but \
             z_ub = {z:e} is at or above the {ACTIVITY_TOL:e} threshold",
        );
    }
}

/// The pinned regime pins: the variable does not move, so the answer
/// cannot leave the box, so gh#928's repair has nothing to do here.
///
/// The second assertion is the one that matters. It pins the *size* of
/// what pinning costs -- exactly the base slack -- so that a change to
/// a barrier-style sigma on this arm shows up as this test failing
/// rather than as a quiet loss of accuracy nobody measured.
#[test]
fn an_active_bound_is_pinned_and_the_step_is_short_by_the_base_slack() {
    for delta in PINNED_DELTAS {
        let b = KINK - delta;
        let (slack, _) = slack_and_z(b);
        let (mut sens, base) = base_sens(b);
        let (dx, segs) = sens
            .parametric_step_path(&[0], &[DP], 64)
            .expect("parametric_step_path");
        assert!(
            dx[0].abs() < 1e-12,
            "delta={delta}: an active bound on this arm is pinned, so dx0 \
             must be zero; got {:e}. A nonzero value means the arm now \
             damps rather than pins, and gh#928's hazard applies here \
             -- wire `weak_rows` up before relaxing this.",
            dx[0],
        );
        assert_eq!(
            segs.len(),
            0,
            "delta={delta}: a pinned bound is not reached"
        );

        let truth = oracle(b + DP);
        let err = (0..2)
            .map(|i| (base[i] + dx[i] - truth[i]).abs())
            .fold(0.0, f64::max);
        assert!(
            (err / slack - 1.0).abs() < 1e-3,
            "delta={delta}: the pinned step should be short by exactly the \
             base slack; err {err:e} against slack {slack:e} (ratio {})",
            err / slack,
        );
    }
}

/// The free regime: the reach scan catches the crossing, and the walk
/// is exact where the plain step overshoots.
#[test]
fn a_free_bound_that_the_step_presses_into_is_reached_and_held() {
    for delta in FREE_DELTAS {
        let b = KINK - delta;
        let truth = oracle(b + DP);
        let (mut sens, base) = base_sens(b);

        let plain = sens.parametric_step(&[0], &[DP]);
        assert!(
            base[0] + plain[0] > UB0 + 1e-3,
            "delta={delta}: the plain step must overshoot the bound, or the \
             walk has nothing to fix; got {}",
            base[0] + plain[0],
        );

        let (dx, segs) = sens
            .parametric_step_path(&[0], &[DP], 64)
            .expect("parametric_step_path");
        let got = [base[0] + dx[0], base[1] + dx[1]];
        assert_eq!(
            segs.len(),
            1,
            "delta={delta}: crossing the kink is one breakpoint, got {segs:?}",
        );
        let err = (0..2)
            .map(|i| (got[i] - truth[i]).abs())
            .fold(0.0, f64::max);
        assert!(
            err < 1e-6,
            "delta={delta}: walk {got:?} vs re-solve {truth:?}, off by {err:e}",
        );
    }
}

/// Across both regimes, the promise the repair exists to keep.
#[test]
fn no_convex_walk_ends_outside_the_box() {
    for delta in PINNED_DELTAS.iter().chain(FREE_DELTAS.iter()) {
        let b = KINK - delta;
        let (mut sens, base) = base_sens(b);
        let (dx, segs) = sens
            .parametric_step_path(&[0], &[DP], 64)
            .expect("parametric_step_path");
        let x0 = base[0] + dx[0];
        assert!(
            x0 <= UB0 + 1e-9,
            "delta={delta}: the walk returned x0 = {x0} against ub {UB0} \
             (over by {:e}, {} segments)",
            x0 - UB0,
            segs.len(),
        );
    }
}

/// Stepping away from the bound must still reach the re-solve. The
/// segment here is a *release* -- the base-active multiplier reaching
/// zero almost immediately -- not a reach, and asserting zero segments
/// (the first draft of this test) was simply wrong about which event
/// the walk records.
#[test]
fn stepping_away_from_the_bound_reaches_the_resolve() {
    let b = KINK - 1e-5;
    let truth = oracle(b - DP);
    let (mut sens, base) = base_sens(b);
    let (dx, segs) = sens
        .parametric_step_path(&[0], &[-DP], 64)
        .expect("parametric_step_path");
    let got = [base[0] + dx[0], base[1] + dx[1]];
    let err = (0..2)
        .map(|i| (got[i] - truth[i]).abs())
        .fold(0.0, f64::max);
    assert!(
        err < 1e-6,
        "stepping away: walk {got:?} vs re-solve {truth:?}, off by {err:e} \
         over {} segments",
        segs.len(),
    );
    assert!(
        segs.len() <= 1,
        "at most one event stepping away from a single bound, got {segs:?}",
    );
}
