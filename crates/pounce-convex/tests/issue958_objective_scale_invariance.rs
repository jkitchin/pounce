//! gh#958 — the active-set QP engine returned `numerical_failure` at a point
//! violating `Gx ≤ h` by `3.5e-3`, and multiplying the *whole objective* by
//! `0.01` — which leaves the minimizer untouched — made it return the right
//! answer. The engine was not invariant to objective scaling.
//!
//! ## The chain
//!
//! The model is 4 variables with `P = L Lᵀ` PSD of **rank 2** (a 2-dimensional
//! null space) and `‖P‖ ≈ 2.4e8`, a box and three rows. Three layers:
//!
//! 1. **`SchurState::solve` stopped refining after two passes.** The base
//!    factor's H block is `2.4e8` while the SMW update columns carry the O(1)
//!    constraint rows, so the Schur complement is ill-conditioned and two
//!    passes took the KKT residual only from `1.7e7` to `1.3e3` — a *relative*
//!    residual of `7e-5`. The active-set loop takes that direction as exact.
//!    A direction that is `7e-5` off does not lie in the working set's null
//!    space, so the iterate walks **off its own active rows**: measured, a row
//!    held active moved by `5.4e-6` on one unit step and then by another
//!    `4.7e-3` when `model_step_cap` (legitimately, for a δ-shifted direction)
//!    let the next step run to `α = 1334`. Scaling the objective by `0.01`
//!    takes two orders off the conditioning and the same solve converges.
//!
//! 2. **`ElasticReformulation::is_feasible` was one-sided** (`v > feas_tol`).
//!    Every slack carries the lower bound `0`, so a negative one is outside
//!    the augmented problem's own box; the equilibrated retry came back with
//!    `v_u = -0.765` and this reported the slacks as driven out.
//!
//! and one more, which is why the fixed engine says `Optimal` rather than
//! `OptimalInaccurate`:
//!
//! 3. **`natural_scale` measured `‖Px‖` where the floor is `|P||x|`.** gh#641
//!    made the status bands scale-relative above a data-magnitude crossover;
//!    the gate that decides whether the problem is *above* that crossover read
//!    the norm of the computed product. On a rank-deficient `P` whose optimum
//!    lies near `null(P)`, `‖Px‖` is `23.3` while `|P||x|` is `7.6e8`, so the
//!    relative arm stayed shut and a stationarity residual of `2.0e-8` — below
//!    its own cancellation floor of `1.7e-7` — was banded as inaccurate.
//!
//! ## Mutation table
//!
//! Measured, one revert at a time, against the rest of the fix in place:
//!
//! | revert | what goes red |
//! |---|---|
//! | `natural_scale` back to `‖Px‖` | both `the_unscaled_solve_reaches_the_optimum` (`OptimalInaccurate`) and `the_answer_does_not_depend_on_the_objective_scale` |
//! | `IR_MAX_PASSES` back to 2 | `the_answer_does_not_depend_on_the_objective_scale`, on the `s = 1e2` arm |
//! | layer 2 (`is_feasible`) | **nothing here** — pinned by `pounce-qp`'s `is_feasible_negative_slack_returns_false` |
//!
//! The `s = 1e2` arm is in the invariance test for exactly the reason the
//! reverts show: with the other layers repaired, `s = 1` no longer separates
//! the refinement fix, and a corpus that does not span the dimension a change
//! acts on cannot report on it. The magnitude axis is the dimension here.
//!
//! `IR_MAX_PASSES` itself was swept rather than picked: 2, 4, 6 and 8 all
//! leave an arm red, 10 is the first that passes, and 16 and 24 are
//! bit-identical to 10. 12 is the first round value above where the answer
//! saturates.
//!
//! ## The layer that is still open
//!
//! `audit_and_repair` returns a "no worse" repair with whatever status
//! `solve_elastic` gave it — `Optimal` included, at a point the audit itself
//! just measured as violating the constraints. On this model, before the
//! refinement fix, that was `Optimal` at `max(Gx − h) = 3.5e-3`.
//! `pounce-convex` re-derives its own verdict and saw through it; the SQP
//! outer loop and every direct `pounce-qp` caller do not.
//!
//! Demoting on an absolute `point_is_feasible(.., feas_tol)` is *not* the fix
//! and was measured: that path exists for the near-miss, and
//! `pounce-algorithm/tests/sqp_near_solution_start.rs` is the case — HS071
//! started at `x* + 1e-6·e₁` lands there with a step-QP point missing
//! `feas_tol` by a factor of two (1.95e-9 against 1e-9) on a QP with points
//! feasible to slack 1.66, and the absolute test turns that into
//! `QpIterationLimit` at iteration 0. Telling a hair's-breadth miss from
//! `3.5e-3` needs a scale-relative feasibility test, with its own
//! measurement. The layers above mean this model no longer reaches that code
//! at all, so the gap is recorded rather than closed.
//!
//! ## What this is NOT evidence about
//!
//! The corpus behind it is one model. It says nothing about magnitude at
//! benchmark scale (`benchmarks/qp` reaches 93 263 variables against this
//! model's 4), nothing about the interior-point arm — which solved this
//! instance at both scales before and after — and nothing about a `P` that is
//! large *and* full rank, where `‖Px‖` and `|P||x|` agree and only layer 1
//! applies.

use pounce_convex::{
    ActiveSetOverrides, QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_active_set,
    solve_qp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `L` of `P = L Lᵀ` — 4x2, so `P` is PSD of rank 2 with a 2-dimensional
/// null space, and `‖P‖∞ ≈ 2.4e8`.
const L: [[f64; 2]; 4] = [
    [8911.4, -12808.1],
    [-692.1, 5085.1],
    [6782.7, 3279.6],
    [7510.0, 1454.5],
];
const C: [f64; 4] = [1.787, -10.739, -8.466, 3.796];
const G: [[f64; 4]; 3] = [
    [-0.580, 1.272, 1.292, 1.799],
    [-0.026, 1.384, -0.906, -0.816],
    [0.081, 0.281, -1.599, -1.731],
];
const H: [f64; 3] = [1.276, -2.410, -1.933];
const LB: [f64; 4] = [-1.309, -3.777, 0.308, -2.300];
const UB: [f64; 4] = [1.504, 0.143, 3.864, 0.684];

/// Clarabel's answer on the unscaled instance: `Solved`, `-3.46349211442049`.
/// POUNCE's own interior-point arm agrees to `1e-7`. The optimal *value* is
/// unique even where the minimizer set is not, and it does not depend on `s`.
const F_STAR: f64 = -3.4634921;

/// The problem with the whole objective multiplied by `s`. `s > 0` rescales
/// `½xᵀPx + cᵀx` without moving its minimizer, so every `s` must produce the
/// same `x` and an objective of `s · f*`.
fn build(s: f64) -> QpProblem {
    let mut p_lower = Vec::new();
    for i in 0..4 {
        for j in 0..=i {
            let v = s * (L[i][0] * L[j][0] + L[i][1] * L[j][1]);
            if v != 0.0 {
                p_lower.push(Triplet::new(i, j, v));
            }
        }
    }
    let mut g = Vec::new();
    for (i, row) in G.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                g.push(Triplet::new(i, j, v));
            }
        }
    }
    QpProblem {
        n: 4,
        p_lower,
        c: C.iter().map(|v| s * v).collect(),
        a: Vec::new(),
        b: Vec::new(),
        g,
        h: H.to_vec(),
        lb: LB.to_vec(),
        ub: UB.to_vec(),
    }
}

fn active_set(p: &QpProblem) -> QpSolution {
    let mut mk = backend;
    solve_qp_active_set(
        p,
        &QpOptions::default(),
        &ActiveSetOverrides::default(),
        &mut mk,
    )
}

/// The objective in the *unscaled* sense, so answers at different `s` are
/// directly comparable.
fn f_unscaled(x: &[f64]) -> f64 {
    let mut quad = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            quad += x[i] * x[j] * (L[i][0] * L[j][0] + L[i][1] * L[j][1]);
        }
    }
    0.5 * quad + C.iter().zip(x).map(|(a, b)| a * b).sum::<f64>()
}

/// `max(Gx − h)` and the box violation — positive means the returned point is
/// outside the feasible set the caller posed.
fn max_violation(x: &[f64]) -> f64 {
    let mut worst = f64::NEG_INFINITY;
    for (i, row) in G.iter().enumerate() {
        let ax: f64 = row.iter().zip(x).map(|(a, b)| a * b).sum();
        worst = worst.max(ax - H[i]);
    }
    for i in 0..4 {
        worst = worst.max(LB[i] - x[i]).max(x[i] - UB[i]);
    }
    worst
}

/// The headline: on the problem exactly as the issue poses it — no rescaling
/// — the active-set engine must reach the optimum and say so.
///
/// Before the fix: `NumericalFailure`, `max(Gx − h) = 3.465e-3`, and an
/// objective of `-3.4957`, *below* `f*` because the point is infeasible.
#[test]
fn the_unscaled_solve_reaches_the_optimum() {
    let p = build(1.0);
    let sol = active_set(&p);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "status {:?} at f={} viol={:.3e}",
        sol.status,
        f_unscaled(&sol.x),
        max_violation(&sol.x)
    );
    // Primal feasibility is asserted separately from the status, because the
    // whole defect was a status that did not describe the point.
    assert!(
        max_violation(&sol.x) <= 1e-8,
        "returned point violates its own constraints by {:.3e}",
        max_violation(&sol.x)
    );
    assert!(
        (f_unscaled(&sol.x) - F_STAR).abs() < 1e-6,
        "f = {} against f* = {F_STAR}",
        f_unscaled(&sol.x)
    );
}

/// The invariance the issue names, stated as the property rather than as two
/// separate expectations: `s` rescales the objective without moving its
/// minimizer, so the answer must not depend on it.
///
/// `s = 1e-2` is the value the issue reports as the one that "fixes it", and
/// it solved cleanly before the fix — so this test is only meaningful *with*
/// the `s = 1` arm beside it.
#[test]
fn the_answer_does_not_depend_on_the_objective_scale() {
    let mut reference: Option<Vec<f64>> = None;
    for s in [1.0, 1e-2, 1e2] {
        let p = build(s);
        let sol = active_set(&p);
        assert_eq!(sol.status, QpStatus::Optimal, "s = {s}");
        assert!(
            max_violation(&sol.x) <= 1e-8,
            "s = {s}: violation {:.3e}",
            max_violation(&sol.x)
        );
        assert!(
            (f_unscaled(&sol.x) - F_STAR).abs() < 1e-6,
            "s = {s}: f = {} against f* = {F_STAR}",
            f_unscaled(&sol.x)
        );
        // The reported objective is the *scaled* one, so it tracks `s`.
        assert!(
            (sol.obj - s * F_STAR).abs() < 1e-6 * s.max(1.0),
            "s = {s}: reported obj = {} against s·f* = {}",
            sol.obj,
            s * F_STAR
        );
        match &reference {
            None => reference = Some(sol.x.clone()),
            Some(x0) => {
                // The minimizer set may not be a single point (`P` is rank 2),
                // so this compares the objective value rather than `x`, and the
                // assertions above already pinned it. What is compared here is
                // that the two runs agree *with each other* to tighter than
                // they agree with `f*`.
                assert!(
                    (f_unscaled(&sol.x) - f_unscaled(x0)).abs() < 1e-7,
                    "s = {s}: f = {} against the s = 1 run's {}",
                    f_unscaled(&sol.x),
                    f_unscaled(x0)
                );
            }
        }
    }
}

/// The interior-point arm solved this instance at both scales before the fix
/// and must still. It is the control: a change that fixed the active-set arm
/// by moving the *problem* (equilibration on the way in, say) would show up
/// here, and a regression in the arm that was already right is exactly what
/// the fix must not buy.
#[test]
fn the_interior_point_arm_is_unmoved() {
    for s in [1.0, 1e-2] {
        let p = build(s);
        let sol = solve_qp_ipm(&p, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "ipm at s = {s}");
        assert!(
            (f_unscaled(&sol.x) - F_STAR).abs() < 1e-6,
            "ipm at s = {s}: f = {}",
            f_unscaled(&sol.x)
        );
    }
}
