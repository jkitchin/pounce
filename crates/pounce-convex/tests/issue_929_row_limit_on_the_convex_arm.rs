//! A limit written as a **constraint row**, on the convex arm (gh#929).
//!
//! # The gap
//!
//! `boundcheck`'s walk decides everything in *primal KKT rows*: `lo`/`hi`
//! and `x_curr` index the primal prefix of the compound vector, and a
//! `BoundRow` ties a multiplier row to the primal row it constrains. On the
//! NLP arm every constraint has such a row — `dⱼ(x) = sⱼ` puts the
//! constraint's own value in the `s` block — which is what
//! `pounce-sensitivity/tests/issue_928_a_limit_written_as_a_row.rs` exploits.
//!
//! The convex active-set KKT has no `s` block:
//!
//! ```text
//! [ H   Aᵀ  B_aᵀ ] [ dx ]
//! [ A   0   0    ] [ dy ]
//! [ B_a 0   0    ] [ dz ]
//! ```
//!
//! An **inactive** `Gⱼx ≤ hⱼ` appears nowhere at all, so a step that drives
//! it past its limit is not a breakpoint the walk can see. An **active** one
//! has a multiplier row but no primal coordinate, so it could not be reported
//! as a `BoundRow` either, and a perturbation that drives its multiplier
//! negative went on holding a row the solution had left. Both halves are
//! measured below, on the same fixture, against a re-solve.
//!
//! # The fix, and why it costs no factorization
//!
//! `pounce_sens_core::rowlimit::RowLimitView` adjoins `t = G_w x` through a
//! multiplier, immediately after the `x` block so it lands inside the primal
//! prefix the walk indexes. The adjoined block is triangular:
//! `dmu = r_t`, one base back-solve on `r_x + G_wᵀ r_t`, then
//! `dt = G_w dx + r_mu`. The base factor is untouched. Releasing an active
//! row is still the base's own row neutralization — generalized here from one
//! `±1` coupling to a whole `G` row, which is why the fixture's row has **two**
//! nonzeros rather than one.
//!
//! # The fixture
//!
//! ```text
//! min ½‖x‖²   s.t.  x₀+x₁+x₂+x₃ = b,   x₀ + x₂ ≤ 1,   x₁ ≤ 0.8
//! ```
//!
//! One equality (the parameter), one **row** limit, one **variable bound** —
//! so every test here is simultaneously a regression guard that wrapping the
//! backsolver did not take variable-bound handling away. In closed form:
//!
//! ```text
//! b ≤ 2.0        x = (b/4, b/4, b/4, b/4)
//! 2.0 ≤ b ≤ 2.6  x = (0.5, (b−1)/2, 0.5, (b−1)/2)
//! b ≥ 2.6        x = (0.5, 0.8, 0.5, b−1.8)
//! ```
//!
//! The row binds at `b = 2.0` and the bound at `b = 2.6`, in that order going
//! up and in the reverse order coming down, so a single perturbation exercises
//! reach-a-row, reach-a-bound, release-a-row and release-a-bound.
//!
//! # What was measured
//!
//! | direction | before | after | truth |
//! |---|---|---|---|
//! | `b: 0 → 4` | `x₀+x₂ = 2.133`, **1.133 past the limit** | `x₀+x₂ = 1.000` | 1.0 |
//! | `b: 4 → 0` | `(0.5, −0.5, 0.5, −0.5)`, row still pinned | `(0, 0, 0, 0)` | `(0,0,0,0)` |
//! | `b: 2 → 1` | `(0.5, 2.0e-5, 0.5, 2.0e-5)` | `(0.25, 0.25, 0.25, 0.25)` | `(0.25,…)` |
//!
//! and the same two directions through `parametric_step_bounded`, which was
//! augmented in a follow-up commit on the same branch:
//!
//! | direction | before | after | truth |
//! |---|---|---|---|
//! | `b: 0 → 4` | `x₀+x₂ = 2.133`, **1.133 past the limit**, `x₁` pinned correctly on the way | reproduces the re-solve to `< 1e-8` | 1.0 |
//! | `b: 4 → 0` | `(0.5, −0.5, 0.5, −0.5)`, row reported binding | reproduces the re-solve to `< 1e-8` | `(0,0,0,0)` |
//!
//! Both are **exact**, not close: the target's whole active set is reached in
//! one shot, and for a QP fix-relax with the right active set is the re-solve.
//! The "before" column of each is recomputed here by
//! `refined_unaugmented`, which drives `refine_step_onto_bounds` straight
//! against `QpKktBacksolver`, rather than quoted.
//!
//! The third row of the first table is a **different branch**: there the row
//! is in the active set with a multiplier of `4.1e-5`, so the release happens at fraction `8.2e-5`
//! rather than in the middle of the walk. Per the branch rule in CLAUDE.md, a
//! fixture that only ever released a healthy multiplier would say nothing
//! about it.
//!
//! # Mutation table
//!
//! Every row was **run**, not predicted, and the right-hand column is the
//! test list the run printed. Two of them go red only in
//! `pounce-sens-core`'s own unit tests, marked `[core]`: `r_mu` and the
//! observer *ordering* are both unreachable from this arm — `lift_rhs`
//! zeroes the `mu` block and this fixture has a single watched row, so
//! every ordering of one observer is the same ordering. That is the branch
//! rule again, and it is why `rowlimit.rs` carries a two-row unit fixture
//! of its own rather than leaning on this file.
//!
//! | mutation | tests that go red |
//! |---|---|
//! | `primal_box` stops appending the observers, so they fall outside the walk's primal prefix — the effect adjoining them at the *end* of the compound vector would have | `a_row_the_step_would_cross_stops_the_walk`, `an_active_row_whose_multiplier_turns_is_released`, `a_barely_active_row_is_released_at_once`, `the_variable_bound_is_still_released_through_the_view` |
//! | observer box sides swapped (`limit` as the lower bound) | nine of the eighteen, including both inertness controls (`the_observers_are_inert_when_no_limit_is_reached`, `the_observers_add_nothing_to_a_refinement_that_reaches_no_limit`) — the walk stops at fraction ≈ 0 and the refinement pins, on a step that crosses nothing |
//! | `base_value` set to `0` rather than `Gⱼ·x` | `a_barely_active_row_is_released_at_once` |
//! | `RowLimitView::fold` drops the `G_wᵀ r_t` fold | `a_row_the_step_would_cross_stops_the_walk`, `releasing_a_row_moves_its_force_onto_the_observer` |
//! | `RowLimitView::unfold` drops `r_mu` from `dt` | `the_triangular_solve_is_the_augmented_system` `[core]` |
//! | the observers are numbered in reverse, so an active row watches its neighbour | `each_active_row_keeps_its_own_observer` `[core]` |
//! | `WatchedRow::active` left `None` for active rows | `an_active_row_whose_multiplier_turns_is_released`, `a_barely_active_row_is_released_at_once`, `the_variable_bound_is_still_released_through_the_view` |
//! | `release_slots` neutralizes only the first coupling of a row | those three plus `releasing_a_row_moves_its_force_onto_the_observer` — the fixture's row has two nonzeros, so a half-neutralized row is still coupled to `x₂` |
//! | `solve_released_inner`'s shift refuses a non-bound row instead of skipping it | `releasing_a_row_moves_its_force_onto_the_observer` |
//! | base bound rows dropped from `RowLimitView::bound_rows` | `the_variable_bound_is_still_released_through_the_view`, `an_active_row_whose_multiplier_turns_is_released` |
//! | base entries dropped from `RowLimitView::lift_multipliers` | the same two — the walk releases a row only when it is in *both* lists, which is why one type owns both |
//! | the augmentation is applied on the conic path too | `a_conic_model_keeps_the_unaugmented_walk`, on its `boundary-far` leg |
//! | `parametric_step_bounded` keeps the unaugmented backsolver — the refinement's defect restored | `a_row_the_refinement_would_cross_stops_it`, `an_active_row_the_refinement_releases` |
//! | `RowLimitView::lift_step` zero-fills the observers instead of evaluating `dt = G_w dx` | `a_row_the_refinement_would_cross_stops_it`; `lifting_a_step_agrees_with_solving_it` `[core]`. The release direction stays **green** — its condition is read off the multiplier, not the observer's motion — so the reach test is the only one here that sees it |
//! | `refined_row_target` reports every row as a pin, dropping the release lookup | `an_active_row_the_refinement_releases` |
//! | `refined_row_target` maps with `n_t = 0`, the pre-gh#929 layout, so an observer decodes as a variable | `a_row_the_refinement_would_cross_stops_it` |
//!
//! # What this file is NOT evidence about
//!
//! - **Scaling.** Unscaled, like every convex sensitivity test.
//!   `RowLimitView::new` refuses a base reporting a `natural_units_factor`
//!   rather than guessing an entry for the observer rows, and no convex base
//!   reports one, so that refusal is unexercised here.
//! - **The conic arm's own row limits.** `a_conic_model_keeps_the_unaugmented_walk`
//!   pins that the augmentation is *skipped* there; it says nothing about what
//!   watching a cone face would mean. `convex_sens_release.rs` owns release on
//!   that arm. The guard is deliberately all-or-nothing: a **mixed partition**
//!   whose orthant rows are ordinary `Gⱼx ≤ hⱼ` loses its observers too,
//!   because one cone block anywhere sets `cone_kinds`. That is a known
//!   conservative choice, not a covered case — no fixture here has a mixed
//!   partition.
//! - **Magnitude.** Four variables and one row. The largest convex fixture in
//!   the corpus is 534 columns and `benchmarks/qp` reaches 93 263; the
//!   observer block costs `O(nnz(G))` per back-solve and nothing here bounds
//!   that at scale.
//! - **Two-sided rows.** `WatchedRow` is one-sided on purpose; a range row is
//!   two watched rows and no fixture here has one.
//! - **The rest of the convex corpus.** `parametric_step_bounded`'s augmented
//!   branch is reached **3 times across every test in `pounce-convex`**
//!   (measured, by an `eprintln` in the branch), and all three are this
//!   file's own. `convex_sens_release.rs`, `convex_sens_backsolver.rs` and
//!   `convex_soc_sensitivity.rs` reach it **zero** times — they are conic, or
//!   have no inequality rows. So their staying green across that follow-up is
//!   evidence of no collateral damage and *not* evidence about the fix; the
//!   fixture below is the only evidence there is.
//! - **The mixed list's two spaces, beyond one entry each.** The list
//!   `parametric_step_bounded` returns names a release by its multiplier row
//!   and a pin by its primal row, and `refined_row_target` is the only
//!   decoder. Every case here returns exactly two entries of one kind — two
//!   pins going up, two releases coming down. A result mixing a release and a
//!   pin in the same list is decodable by construction (the two ranges do not
//!   overlap) but is not exercised by any fixture.
//! - **Rows with no `x` coefficients**, empty `G`, or `m_ineq = 0` — the
//!   augmentation is skipped for the last of these and the rest are unbuilt.

use pounce_convex::QpOptions;
use pounce_convex::cones::ConeSpec;
use pounce_convex::ipm::{solve_qp_ipm, solve_socp_ipm};
use pounce_convex::qp::{QpProblem, QpSolution, QpStatus, Triplet};
use pounce_convex::sensitivity::{PathTarget, QpSensitivity, RefinedRow};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_sens_core::backsolver::SensBacksolver;
use pounce_sens_core::boundcheck::{
    BoundMultiplier, PathSegment, RefineStop, refine_step_onto_bounds, step_along_path,
};
use pounce_sens_core::rowlimit::{RowLimitView, WatchedRow};

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn tri(row: usize, col: usize, val: f64) -> Triplet {
    Triplet { row, col, val }
}

/// The row's limit, and the variable bound's.
const H0: f64 = 1.0;
const UB1: f64 = 0.8;
/// Activity threshold handed to `QpSensitivity::build`.
const ACTIVITY_TOL: f64 = 1e-7;
/// The same floor `parametric_step_path` walks with.
const EPS: f64 = 1e-8;

fn model(b: f64) -> QpProblem {
    QpProblem {
        n: 4,
        p_lower: (0..4).map(|i| tri(i, i, 1.0)).collect(),
        c: vec![0.0; 4],
        a: (0..4).map(|j| tri(0, j, 1.0)).collect(),
        b: vec![b],
        // Two nonzeros on purpose: a one-nonzero row cannot tell a release
        // that neutralizes the whole row from one that neutralizes its first
        // entry.
        g: vec![tri(0, 0, 1.0), tri(0, 2, 1.0)],
        h: vec![H0],
        lb: vec![f64::NEG_INFINITY; 4],
        ub: vec![f64::INFINITY, UB1, f64::INFINITY, f64::INFINITY],
    }
}

fn closed_form(b: f64) -> [f64; 4] {
    if b <= 2.0 {
        [b / 4.0; 4]
    } else if b <= 2.6 {
        [0.5, (b - 1.0) / 2.0, 0.5, (b - 1.0) / 2.0]
    } else {
        [0.5, UB1, 0.5, b - 1.8]
    }
}

fn solved(b: f64, tol: f64) -> QpSolution {
    let opts = QpOptions {
        tol,
        ..Default::default()
    };
    let sol = solve_qp_ipm(&model(b), &opts, backend);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "the fixture must solve at b={b}"
    );
    sol
}

/// The perturbed problem solved outright, two orders tighter than the base
/// solve, and cross-checked against the closed form so a wrong oracle cannot
/// certify a wrong walk.
fn oracle(b: f64) -> Vec<f64> {
    let sol = solved(b, 1e-13);
    let want = closed_form(b);
    for j in 0..4 {
        assert!(
            (sol.x[j] - want[j]).abs() < 1e-8,
            "the oracle disagrees with the closed form at b={b}: {:?} vs {want:?}",
            sol.x,
        );
    }
    sol.x.clone()
}

/// The walk, and the point it lands on.
fn walk(b0: f64, delta: f64) -> (Vec<f64>, Vec<PathSegment>, QpSensitivity) {
    let prob = model(b0);
    let sol = solved(b0, QpOptions::default().tol);
    let mut sens = QpSensitivity::build(&prob, &sol, &QpOptions::default(), ACTIVITY_TOL, backend)
        .expect("the fixture must build a sensitivity");
    let (dx, segments) = sens
        .parametric_step_path(&[0], &[delta], 64)
        .expect("the walk must answer");
    let x = sol.x.iter().zip(&dx).map(|(a, d)| a + d).collect();
    (x, segments, sens)
}

/// The same walk with the observers taken away: `step_along_path` driven
/// straight against `QpKktBacksolver`, which is exactly what
/// `parametric_step_path` did before gh#929. This is the "before" column of
/// the table in the module docs, recomputed rather than quoted.
fn walk_unaugmented(b0: f64, delta: f64) -> Vec<f64> {
    let prob = model(b0);
    let sol = solved(b0, QpOptions::default().tol);
    let sens = QpSensitivity::build(&prob, &sol, &QpOptions::default(), ACTIVITY_TOL, backend)
        .expect("the fixture must build a sensitivity");
    let bs = sens.backsolver();
    let mut rhs = vec![0.0; bs.dim()];
    rhs[prob.n] = delta;
    let lo: Vec<f64> = (0..prob.n).map(|j| prob.lb[j]).collect();
    let hi: Vec<f64> = (0..prob.n).map(|j| prob.ub[j]).collect();
    let mults: Vec<BoundMultiplier> = bs
        .bound_rows()
        .unwrap_or(&[])
        .iter()
        .map(|br| BoundMultiplier {
            row: br.row,
            base: if br.lower {
                sol.z_lb[br.var_row]
            } else {
                sol.z_ub[br.var_row]
            },
        })
        .collect();
    let (dx, _) = step_along_path(&bs, &rhs, &sol.x, &lo, &hi, &mults, 64, &[], &[], &[], EPS)
        .expect("the unaugmented walk answers too — wrongly, which is the point");
    sol.x.iter().zip(&dx).map(|(a, d)| a + d).collect()
}

fn row_value(x: &[f64]) -> f64 {
    x[0] + x[2]
}

fn max_err(got: &[f64], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .fold(0.0f64, |a, (g, w)| a.max((g - w).abs()))
}

// ---------------------------------------------------------------------------

/// Without this the fixture could be measuring a solver bug rather than a
/// sensitivity one. Every `b` this file uses as an oracle target appears here.
#[test]
fn the_closed_form_and_the_solver_agree_at_every_oracle_target() {
    for b in [0.0, 0.4, 1.0, 3.0, 4.0] {
        let _ = oracle(b);
    }
}

/// The two kinks — `b = 2.0` where the row binds and `b = 2.6` where the
/// bound does — are **bases** here, never oracle targets, and they are where
/// the interior-point method's own convergence is loosest: measured `8.7e-8`
/// and `8.3e-8` at `tol = 1e-13`, against the `1e-8` every non-degenerate
/// target clears. Quoting those numbers is the point — `8.7e-8` is also the
/// accuracy floor of `a_barely_active_row_the_step_presses_into_stays`, which
/// starts at `b = 2.0`.
#[test]
fn the_kink_bases_are_where_the_solver_is_loosest() {
    for b in [2.0, 2.6] {
        let sol = solved(b, 1e-13);
        let err = max_err(&sol.x, &closed_form(b));
        assert!(err < 1e-6, "the kink solve at b={b} is off by {err:e}");
        assert!(
            err > 1e-9,
            "if the kink at b={b} now solves as tightly as the rest, this test \
             has stopped describing anything: err = {err:e}",
        );
    }
}

/// The reach half: a step that would take an **inactive** row past its limit
/// stops there instead.
#[test]
fn a_row_the_step_would_cross_stops_the_walk() {
    let (x, segments, sens) = walk(0.0, 4.0);
    let want = oracle(4.0);
    assert!(
        max_err(&x, &want) < 1e-9,
        "walk {x:?} does not reproduce the re-solve {want:?}",
    );
    assert!(
        row_value(&x) <= H0 + 1e-9,
        "the answer is {:.6} past the row limit",
        row_value(&x) - H0,
    );

    // Both breakpoints, in the order the closed form puts them: the row at
    // b = 2.0 and the bound at b = 2.6.
    let targets: Vec<(PathTarget, f64, bool)> = segments
        .iter()
        .map(|s| (sens.path_segment_target(s), s.at, s.pinned))
        .collect();
    assert_eq!(
        targets.len(),
        2,
        "expected two breakpoints, got {targets:?}"
    );
    assert_eq!(targets[0].0, PathTarget::InequalityRow(0), "{targets:?}");
    assert!((targets[0].1 - 0.5).abs() < 1e-6, "{targets:?}");
    assert!(targets[0].2, "the row is reached and held: {targets:?}");
    assert_eq!(targets[1].0, PathTarget::Variable(1), "{targets:?}");
    assert!((targets[1].1 - 0.65).abs() < 1e-6, "{targets:?}");
    assert!(targets[1].2, "the bound is reached and held: {targets:?}");
}

/// The same step without observers, which is what the arm did before. Kept as
/// a live measurement so the size of the defect is a number this file
/// computes rather than a claim it repeats.
#[test]
fn the_unaugmented_walk_runs_past_the_row() {
    let x = walk_unaugmented(0.0, 4.0);
    let over = row_value(&x) - H0;
    assert!(
        over > 1.0,
        "the pre-gh#929 walk was supposed to overshoot the row limit by ~1.13, \
         got {over:e} from {x:?}",
    );
    assert!((over - 1.1333333).abs() < 1e-5, "measured overshoot {over}");
    // And it is not a small perturbation of the truth either.
    assert!(max_err(&x, &oracle(4.0)) > 1.0, "{x:?}");
}

/// The release half: an **active** row whose multiplier turns negative is let
/// go, and the walk finishes on the re-solve.
#[test]
fn an_active_row_whose_multiplier_turns_is_released() {
    let (x, segments, sens) = walk(4.0, -4.0);
    let want = oracle(0.0);
    assert!(
        max_err(&x, &want) < 1e-9,
        "walk {x:?} does not reproduce the re-solve {want:?}",
    );
    let targets: Vec<(PathTarget, f64, bool)> = segments
        .iter()
        .map(|s| (sens.path_segment_target(s), s.at, s.pinned))
        .collect();
    assert_eq!(
        targets.len(),
        2,
        "expected two breakpoints, got {targets:?}"
    );
    // Coming down, the bound goes first (b = 2.6) and the row second
    // (b = 2.0) — the reverse of the ascending order.
    assert_eq!(targets[0].0, PathTarget::Variable(1), "{targets:?}");
    assert!((targets[0].1 - 0.35).abs() < 1e-6, "{targets:?}");
    assert!(!targets[0].2, "the bound is released: {targets:?}");
    assert_eq!(targets[1].0, PathTarget::InequalityRow(0), "{targets:?}");
    assert!((targets[1].1 - 0.5).abs() < 1e-6, "{targets:?}");
    assert!(!targets[1].2, "the row is released: {targets:?}");
}

/// The bound release is the same test's other half, named so the mutation
/// table can point at it: strip the base's own rows out of the wrapper and
/// `x₁` never leaves `0.8`.
#[test]
fn the_variable_bound_is_still_released_through_the_view() {
    let (x, _, _) = walk(4.0, -4.0);
    assert!(
        x[1].abs() < 1e-9,
        "x₁ should have come off its 0.8 bound and landed at 0, got {}",
        x[1],
    );
}

/// Before the fix the same step kept the row pinned for the whole walk.
#[test]
fn the_unaugmented_walk_holds_a_row_the_solution_has_left() {
    let x = walk_unaugmented(4.0, -4.0);
    assert!(
        (row_value(&x) - H0).abs() < 1e-8,
        "the pre-gh#929 walk was supposed to hold the row at its limit, \
         got {:.9} from {x:?}",
        row_value(&x),
    );
    assert!(
        max_err(&x, &oracle(0.0)) > 0.4,
        "and to be far from the truth: {x:?}",
    );
}

/// A **different branch** of the release rule: the row is in the active set
/// but its multiplier is `4.1e-5`, so the release is immediate rather than
/// mid-walk. A fixture that only ever released a healthy multiplier would be
/// silent about this one.
#[test]
fn a_barely_active_row_is_released_at_once() {
    let base = solved(2.0, QpOptions::default().tol);
    let z = base.z[0];
    assert!(
        z > ACTIVITY_TOL && z < 1e-3,
        "this test needs a barely-active row; z = {z:e}",
    );

    let (x, segments, sens) = walk(2.0, -1.0);
    let want = oracle(1.0);
    assert!(
        max_err(&x, &want) < 1e-8,
        "walk {x:?} does not reproduce the re-solve {want:?}",
    );
    assert_eq!(segments.len(), 1, "{segments:?}");
    assert_eq!(
        sens.path_segment_target(&segments[0]),
        PathTarget::InequalityRow(0),
    );
    assert!(!segments[0].pinned, "{segments:?}");
    assert!(
        segments[0].at < 1e-3,
        "the release should be essentially immediate, got {}",
        segments[0].at,
    );

    // Before the fix: the row stayed pinned and the whole perturbation went
    // into the two free variables.
    let before = walk_unaugmented(2.0, -1.0);
    assert!(
        max_err(&before, &want) > 0.2,
        "the pre-gh#929 answer was supposed to be far off: {before:?}",
    );
}

/// The other side of the same base point: the row is active and truth keeps
/// it active, so the observers must not invent a breakpoint.
#[test]
fn a_barely_active_row_the_step_presses_into_stays() {
    let (x, segments, sens) = walk(2.0, 1.0);
    let want = oracle(3.0);
    // The base solve sits 4.1e-5 inside the row, and a pinned row freezes the
    // answer at the base point rather than at the limit — the cost of pinning
    // that `issue_928_convex_path_leaves_the_box.rs` measures. So the
    // tolerance here is the base solve's own distance from the limit, not the
    // walk's accuracy.
    assert!(max_err(&x, &want) < 1e-3, "walk {x:?} vs {want:?}");
    assert!(
        row_value(&x) <= H0 + 1e-9,
        "the answer must not leave the row: {:.9}",
        row_value(&x),
    );
    let targets: Vec<PathTarget> = segments
        .iter()
        .map(|s| sens.path_segment_target(s))
        .collect();
    assert_eq!(targets, vec![PathTarget::Variable(1)], "{segments:?}");
}

/// A walk that crosses nothing must return the answer it returned before the
/// observers existed — bit for bit, because the adjoined block is exactly
/// triangular and contributes nothing when `r_t` and `r_mu` are zero.
#[test]
fn the_observers_are_inert_when_no_limit_is_reached() {
    let (x, segments, _) = walk(0.0, 0.4);
    assert!(segments.is_empty(), "{segments:?}");
    let before = walk_unaugmented(0.0, 0.4);
    assert_eq!(
        x, before,
        "the observer block changed an answer it does not touch",
    );
    assert!(max_err(&x, &oracle(0.4)) < 1e-9, "{x:?}");
}

/// The conic path keeps the unaugmented walk, on purpose: there `active_rows`
/// is the cone's normal at the boundary point — or the whole block at an apex
/// — not a row of `G`, and `active_ineq` names the rows a block *contributed*
/// rather than the active object. An observer on `Gⱼx ≤ hⱼ` would be watching
/// a limit the solve is not enforcing that way, and would read a multiplier
/// row belonging to something else.
///
/// The fixtures are `convex_soc_sensitivity.rs`'s two, both with closed-form
/// derivatives. The **apex** is the one that catches the mutation: there every
/// `G` row sits exactly at its `h = 0`, so observers would be born at their
/// limits and manufacture breakpoints out of a step that has none. On the
/// boundary fixture the rows are slack and the augmentation would be inert —
/// which is exactly why one conic fixture is not enough.
#[test]
fn a_conic_model_keeps_the_unaugmented_walk() {
    let opts = QpOptions {
        tol: 1e-11,
        ..Default::default()
    };
    let cones = [ConeSpec::SecondOrder(3)];
    let db = 1e-3;

    // Boundary: s = (t, x₀, x₁) = h − Gx with h = 0; dx₀/db = dx₁/db = ½.
    let boundary = QpProblem {
        n: 3,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0), tri(2, 2, 1.0)],
        c: vec![-1.0, -0.2, 0.0],
        a: vec![tri(0, 0, 1.0), tri(0, 1, 1.0)],
        b: vec![1.0],
        g: vec![tri(0, 2, -1.0), tri(1, 0, -1.0), tri(2, 1, -1.0)],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    // Apex: the cost on t crushes the block to its tip, pinning x₀ = x₁ = t = 0,
    // so the equality hands the whole perturbation to x₂: dx/db = (0,0,1,0).
    let apex = QpProblem {
        n: 4,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0), tri(2, 2, 1.0)],
        c: vec![0.0, 0.0, -1.0, 5.0],
        a: vec![tri(0, 0, 1.0), tri(0, 2, 1.0)],
        b: vec![1.0],
        g: vec![tri(0, 3, -1.0), tri(1, 0, -1.0), tri(2, 1, -1.0)],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };

    for (name, prob, db, want) in [
        ("boundary", boundary.clone(), db, vec![0.5, 0.5]),
        // Far enough to drive x₁ (base 0.3) through zero. `-x₁ ≤ 0` is `G`
        // row 2, and reading it as a limit is exactly the mistake: the cone
        // constrains ‖(x₀,x₁)‖, not the sign of x₁, so an observer there
        // manufactures a breakpoint at fraction 0.75 out of a step that has
        // none.
        ("boundary-far", boundary, -0.8, vec![0.5, 0.5]),
        ("apex", apex, db, vec![0.0, 0.0, 1.0, 0.0]),
    ] {
        let sol = solve_socp_ipm(&prob, &cones, &opts, backend);
        assert_eq!(sol.status, QpStatus::Optimal, "{name}");
        let mut sens =
            QpSensitivity::build_conic(&prob, &cones, &sol, &opts, ACTIVITY_TOL, backend)
                .unwrap_or_else(|e| panic!("{name} must build a sensitivity, got {e:?}"));
        let (dx, segments) = sens
            .parametric_step_path(&[0], &[db], 64)
            .unwrap_or_else(|e| panic!("the {name} walk must still answer, got {e:?}"));
        assert!(
            segments.is_empty(),
            "{name}: the conic walk must record no breakpoints, got {segments:?}",
        );
        for (j, w) in want.iter().enumerate() {
            assert!(
                (dx[j] / db - w).abs() < 1e-6,
                "{name}: dx{j}/db = {}, want {w}",
                dx[j] / db,
            );
        }
    }
}

/// The release **shift** — `solve_released_step`, the fix-relax half — routed
/// through the observer.
///
/// The walk never reaches it: along a path a multiplier is already zero where
/// it is released, so there is no force to move. `RowLimitView` still has to
/// get it right, because the shift is what makes a *one-shot* release
/// meaningful, and it is the one place where the base backsolver has to
/// tolerate a released row that is not one of its own `BoundRow`s.
///
/// The check is a number, not a restatement of the code. At `b = 4` the base
/// point is `(0.5, 0.8, 0.5, 2.2)` with the row active at `z = 1.7`. Release
/// the row, keep the bound pinned, perturb nothing: the answer is the optimum
/// of `min ½‖x‖²` subject to `Σx = 4` and `x₁ = 0.8`, which is
/// `(16/15, 0.8, 16/15, 16/15)`. So `dx = (0.5667, 0, 0.5667, −1.1333)`.
#[test]
fn releasing_a_row_moves_its_force_onto_the_observer() {
    let prob = model(4.0);
    let sol = solved(4.0, QpOptions::default().tol);
    let sens = QpSensitivity::build(&prob, &sol, &QpOptions::default(), ACTIVITY_TOL, backend)
        .expect("the fixture must build a sensitivity");
    let bs = sens.backsolver();
    let n = prob.n;
    // The k-th active inequality row's multiplier sits at `n + m_eq + k`, and
    // row 0 is the only active one here.
    let mult_row = n + prob.b.len();
    let view = RowLimitView::new(
        bs,
        n,
        vec![WatchedRow {
            coefficients: vec![(0, 1.0), (2, 1.0)],
            limit: H0,
            base_value: sol.x[0] + sol.x[2],
            active: Some((mult_row, sol.z[0])),
        }],
    )
    .expect("the view must build");

    let released = vec![view.from_base(mult_row).expect("the row lifts")];
    let rhs = vec![0.0; view.dim()];
    let mut lhs = vec![0.0; view.dim()];
    assert!(
        view.solve_released_step(&released, &rhs, &mut lhs),
        "the released step must answer",
    );
    let want = [16.0 / 15.0 - 0.5, 0.0, 16.0 / 15.0 - 0.5, 16.0 / 15.0 - 2.2];
    for j in 0..n {
        assert!(
            (lhs[j] - want[j]).abs() < 1e-6,
            "dx = {:?}, want {want:?}",
            &lhs[..n],
        );
    }
    // And the observer reports the row leaving its limit by the same amount.
    let dt = lhs[n];
    assert!(
        (dt - (want[0] + want[2])).abs() < 1e-6,
        "the observer read {dt}, but dx moved the row by {}",
        want[0] + want[2],
    );
}

// ---------------------------------------------------------------------------
// The one-shot refinement.
//
// `parametric_step_path` walks; `parametric_step_bounded` repairs a single
// linear predictor. gh#929 taught the first about row limits and left the
// second carrying the whole defect, which this section closes. The two
// entry points reach different code — the walk re-forms its direction at each
// breakpoint and never has a multiplier left to move, while the refinement
// decides every condition at the base point and needs
// `solve_released_step`'s shift, the one
// `releasing_a_row_moves_its_force_onto_the_observer` pins at unit level.
// ---------------------------------------------------------------------------

/// The refinement, and the point it lands on — the counterpart of [`walk`].
fn refined(b0: f64, delta: f64) -> (Vec<f64>, Vec<usize>, RefineStop, QpSensitivity) {
    let prob = model(b0);
    let sol = solved(b0, QpOptions::default().tol);
    let mut sens = QpSensitivity::build(&prob, &sol, &QpOptions::default(), ACTIVITY_TOL, backend)
        .expect("the fixture must build a sensitivity");
    let (dx, rows, stop) = sens
        .parametric_step_bounded(&[0], &[delta], EPS, 32)
        .expect("the refinement must answer");
    let x = sol.x.iter().zip(&dx).map(|(a, d)| a + d).collect();
    (x, rows, stop, sens)
}

/// The same refinement with the observers taken away: `refine_step_onto_bounds`
/// driven straight against `QpKktBacksolver`, which is exactly what
/// `parametric_step_bounded` did before this section existed. The "before"
/// column, recomputed rather than quoted — the counterpart of
/// [`walk_unaugmented`].
fn refined_unaugmented(b0: f64, delta: f64) -> Vec<f64> {
    let prob = model(b0);
    let sol = solved(b0, QpOptions::default().tol);
    let sens = QpSensitivity::build(&prob, &sol, &QpOptions::default(), ACTIVITY_TOL, backend)
        .expect("the fixture must build a sensitivity");
    let bs = sens.backsolver();
    let mut rhs = vec![0.0; bs.dim()];
    rhs[prob.n] = delta;
    let mut dx_plain = vec![0.0; bs.dim()];
    assert!(bs.solve(&rhs, &mut dx_plain), "the plain step must answer");
    let lo: Vec<f64> = (0..prob.n).map(|j| prob.lb[j]).collect();
    let hi: Vec<f64> = (0..prob.n).map(|j| prob.ub[j]).collect();
    let mults: Vec<BoundMultiplier> = bs
        .bound_rows()
        .unwrap_or(&[])
        .iter()
        .map(|br| BoundMultiplier {
            row: br.row,
            base: if br.lower {
                sol.z_lb[br.var_row]
            } else {
                sol.z_ub[br.var_row]
            },
        })
        .collect();
    let (dx, _, _) =
        refine_step_onto_bounds(&bs, &dx_plain, &sol.x, &lo, &hi, &mults, &rhs, EPS, EPS, 32)
            .expect("the unaugmented refinement answers too — wrongly, which is the point");
    sol.x.iter().zip(&dx).map(|(a, d)| a + d).collect()
}

/// The reach half. `b: 0 -> 4` crosses the row at `b = 2.0` and the bound at
/// `b = 2.6`; pinning both is the whole active set of the target, so for a QP
/// the one-shot answer is not an approximation — it is the re-solve.
#[test]
fn a_row_the_refinement_would_cross_stops_it() {
    let (x, rows, stop, sens) = refined(0.0, 4.0);
    let want = oracle(4.0);
    assert!(
        max_err(&x, &want) < 1e-8,
        "the refinement {x:?} does not reproduce the re-solve {want:?}",
    );
    assert!(
        row_value(&x) <= H0 + 1e-8,
        "the answer is {:.6} past the row limit",
        row_value(&x) - H0,
    );
    assert_eq!(stop, RefineStop::Settled, "rows {rows:?}");

    // Both conditions are pins, and the row is named as a row.
    let targets: Vec<RefinedRow> = rows
        .iter()
        .map(|&r| {
            sens.refined_row_target(r)
                .unwrap_or_else(|| panic!("row {r} must resolve, from {rows:?}"))
        })
        .collect();
    assert_eq!(targets.len(), 2, "{targets:?}");
    assert!(targets.iter().all(|t| !t.released), "{targets:?}");
    assert!(
        targets
            .iter()
            .any(|t| t.target == PathTarget::InequalityRow(0)),
        "the row limit must be reported as a row: {targets:?}",
    );
    assert!(
        targets.iter().any(|t| t.target == PathTarget::Variable(1)),
        "the variable bound must still be reported: {targets:?}",
    );
}

/// The size of the defect, as a number this file computes.
#[test]
fn the_unaugmented_refinement_runs_past_the_row() {
    let x = refined_unaugmented(0.0, 4.0);
    let over = row_value(&x) - H0;
    assert!(
        (over - 1.1333333).abs() < 1e-5,
        "the pre-fix refinement was supposed to overshoot the row limit by \
         ~1.1333, got {over} from {x:?}",
    );
    // It is not a small perturbation of the truth either — and note that it
    // pinned `x₁` correctly on the way, which is what made it plausible.
    assert!(max_err(&x, &oracle(4.0)) > 0.5, "{x:?}");
    assert!((x[1] - UB1).abs() < 1e-8, "{x:?}");
}

/// The release half, which is the one the walk cannot stand in for: here the
/// row's multiplier is still `1.7` when the decision is taken, so
/// `solve_released_step`'s shift has real force to move.
#[test]
fn an_active_row_the_refinement_releases() {
    let (x, rows, stop, sens) = refined(4.0, -4.0);
    let want = oracle(0.0);
    assert!(
        max_err(&x, &want) < 1e-8,
        "the refinement {x:?} does not reproduce the re-solve {want:?}",
    );
    assert_eq!(stop, RefineStop::Settled, "rows {rows:?}");

    let targets: Vec<RefinedRow> = rows
        .iter()
        .map(|&r| {
            sens.refined_row_target(r)
                .unwrap_or_else(|| panic!("row {r} must resolve, from {rows:?}"))
        })
        .collect();
    assert_eq!(targets.len(), 2, "{targets:?}");
    assert!(
        targets.iter().all(|t| t.released),
        "both limits are released, not pinned: {targets:?}",
    );
    assert!(
        targets
            .iter()
            .any(|t| t.target == PathTarget::InequalityRow(0)),
        "the row must be among the released: {targets:?}",
    );
    assert!(
        targets.iter().any(|t| t.target == PathTarget::Variable(1)),
        "the bound release must survive the augmentation: {targets:?}",
    );
}

/// The reverse direction's "before" column, which is worse than a lost
/// record: the answer is wrong *and* the report says the cap is binding.
#[test]
fn the_unaugmented_refinement_holds_a_row_the_solution_has_left() {
    let x = refined_unaugmented(4.0, -4.0);
    assert!(
        (row_value(&x) - H0).abs() < 1e-8,
        "the pre-fix refinement was supposed to hold the row at its limit, \
         got {:.9} from {x:?}",
        row_value(&x),
    );
    let want = [0.5, -0.5, 0.5, -0.5];
    assert!(
        max_err(&x, &want) < 1e-6,
        "the pre-fix answer was measured at {want:?}, got {x:?}",
    );
    assert!(
        max_err(&x, &oracle(0.0)) > 0.4,
        "and far from the truth: {x:?}"
    );
}

/// A step that reaches nothing must be the plain step, or the observers are
/// manufacturing conditions rather than watching for them. The counterpart of
/// `the_observers_are_inert_when_no_limit_is_reached`.
#[test]
fn the_observers_add_nothing_to_a_refinement_that_reaches_no_limit() {
    let (x, rows, stop, _) = refined(0.0, 0.4);
    assert!(
        rows.is_empty(),
        "no condition should have been added: {rows:?}"
    );
    assert_eq!(stop, RefineStop::Settled);
    let want = oracle(0.4);
    assert!(
        max_err(&x, &want) < 1e-9,
        "{x:?} should be the plain step, which here is exact: {want:?}",
    );
}

/// The conic guard, on the refinement rather than the walk. `row_limit_view`
/// is one helper, but a gate that is only ever exercised through one caller
/// is only evidence about that caller — the apex fixture is the one that
/// would notice, since there every `G` row sits at its `h = 0` and observers
/// would be born at their limits.
#[test]
fn a_conic_model_keeps_the_unaugmented_refinement() {
    let opts = QpOptions {
        tol: 1e-11,
        ..Default::default()
    };
    let cones = [ConeSpec::SecondOrder(3)];
    let apex = QpProblem {
        n: 4,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0), tri(2, 2, 1.0)],
        c: vec![0.0, 0.0, -1.0, 5.0],
        a: vec![tri(0, 0, 1.0), tri(0, 2, 1.0)],
        b: vec![1.0],
        g: vec![tri(0, 3, -1.0), tri(1, 0, -1.0), tri(2, 1, -1.0)],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let db = 1e-3;
    let sol = solve_socp_ipm(&apex, &cones, &opts, backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    let mut sens = QpSensitivity::build_conic(&apex, &cones, &sol, &opts, ACTIVITY_TOL, backend)
        .expect("the apex must build a sensitivity");
    let (dx, rows, _) = sens
        .parametric_step_bounded(&[0], &[db], EPS, 32)
        .expect("the conic refinement must still answer");
    assert!(
        rows.is_empty(),
        "the conic arm must add no row conditions, got {rows:?}",
    );
    for (j, w) in [0.0, 0.0, 1.0, 0.0].iter().enumerate() {
        assert!(
            (dx[j] / db - w).abs() < 1e-6,
            "dx{j}/db = {}, want {w}",
            dx[j] / db,
        );
    }
}
