//! gh #848 — the active-set engine must not certify a saddle point as a
//! solution.
//!
//! Every `QpStatus::Optimal` exit in `pounce-qp` is a *first-order* finding:
//! the projected gradient vanishes and the working set's multipliers carry
//! admissible signs. On a PSD Hessian that is the whole story. On an
//! indefinite one it is not — a saddle point satisfies exactly the same
//! conditions — and the box path's cold start makes the failure the *default*
//! rather than a corner: it begins at the projection of the origin into the
//! box, which on a Hessian with zero linear term is the saddle itself. The
//! gradient there is already zero, so iteration one reports `‖p‖∞ ≤ opt_tol`
//! with an empty working set and returns. `§4.5` inertia control does not
//! intervene: it shifts `H` until the reduced KKT factors, which supplies a
//! descent direction for a nonzero right-hand side, and the right-hand side
//! here is zero.
//!
//! # Which branch each fixture reaches
//!
//! `crates/pounce-qp/src/negcurv.rs` has three verdicts and they fail in
//! opposite directions, so a corpus that reaches only one of them is not
//! evidence about the others. Each test below names its branch:
//!
//! | test | verdict | what it would catch |
//! |---|---|---|
//! | `the_saddle_of_an_indefinite_box_qp_is_not_a_solution` | `NegativeCurvature` → escape | the defect itself |
//! | `an_unblocked_negative_curvature_direction_certifies_unbounded` | `NegativeCurvature` → ray | a missing producer for `ray_certifies_unbounded` (gh #791) |
//! | `a_genuine_minimum_of_an_indefinite_qp_is_still_certified` | `Certified` | a check that rejects everything |
//! | `a_weak_minimum_is_not_rejected` | `NotChecked` | the over-rejection the `δ > 0` shortcut would have caused |
//! | `a_psd_claim_skips_the_check_entirely` | gate | the convex arm paying for this |
//! | `declining_recession_verdicts_keeps_the_point_and_still_reports_the_finding` | `NegativeCurvature` → ray, declined | gh #423's opt-out being ignored, i.e. gh #419 again |
//!
//! `a_weak_minimum_is_not_rejected` is the one that is not a duplicate of
//! anything else here. The tempting cheap test is "`§4.5` needed a shift ⇒
//! the reduced Hessian is not PD ⇒ reject", and it is wrong: the shift ladder
//! fires on `Singular` as well as on `WrongInertia`, and a singular reduced
//! Hessian is a **weak minimum** — a correct answer that the cheap test would
//! have downgraded. That is why the module produces a direction and evaluates
//! `dᵀHd` against `H` rather than inferring from `δ`.

use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver, QpStatus,
    SecondOrderVerdict,
};
use std::rc::Rc;

const NEG_INF: f64 = -1e20;
const POS_INF: f64 = 1e20;

fn new_solver() -> ParametricActiveSetSolver {
    ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()))
}

fn no_rows(n: usize) -> GenTMatrix {
    GenTMatrix::new(GenTMatrixSpace::new(0, n as i32, Vec::new(), Vec::new()))
}

/// Upper triangle of `[[a, b]; [b, c]]`.
fn sym2(a: f64, b: f64, c: f64) -> SymTMatrix {
    let space = SymTMatrixSpace::new(2, vec![1, 1, 2], vec![1, 2, 2]);
    let mut h = SymTMatrix::new(Rc::clone(&space));
    h.set_values(&[a, b, c]);
    h
}

// ─────────────────────────────────────────────────────────────────
// The reproducer.
//
//     min ½ xᵀ [[1, 5]; [5, 1]] x   s.t.  −1 ≤ x ≤ 1
//
// Eigenvalues −4 and 6, eigenvectors (1, −1)/√2 and (1, 1)/√2. The
// objective is `3t² − 2s²` in those coordinates, so the minimum sits at the
// two corners the negative eigendirection points at: `(1, −1)` and `(−1, 1)`,
// both with objective −4.
//
// Pre-fix this returned `Optimal` at `x = (0, 0)` with `obj = 0` — the
// *maximum* along the (1, −1) direction, reported as the answer.
// ─────────────────────────────────────────────────────────────────
#[test]
fn the_saddle_of_an_indefinite_box_qp_is_not_a_solution() {
    let h = sym2(1.0, 5.0, 1.0);
    let a = no_rows(2);
    let g = [0.0, 0.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [-1.0, -1.0];
    let xu = [1.0, 1.0];
    let qp = QpProblem {
        n: 2,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Indefinite,
    };

    let sol = new_solver()
        .solve(&qp, None, &QpOptions::default())
        .expect("indefinite box QP must solve");

    assert_eq!(sol.status, QpStatus::Optimal, "x = {:?}", sol.x);
    assert!(
        (sol.obj + 4.0).abs() < 1e-7,
        "objective {} at x = {:?}; the box minimum is −4 at (1, −1) / (−1, 1). \
         An objective of 0 is the saddle at the origin, which is what gh #848 is.",
        sol.obj,
        sol.x
    );
    // Both corners are optima; either is a correct answer.
    let corner = (sol.x[0] - 1.0).abs() < 1e-7 && (sol.x[1] + 1.0).abs() < 1e-7
        || (sol.x[0] + 1.0).abs() < 1e-7 && (sol.x[1] - 1.0).abs() < 1e-7;
    assert!(corner, "expected a corner of the box, got {:?}", sol.x);
    // Having escaped, the point it *landed* on is certified in its own right:
    // with both bounds active the null space is trivial.
    assert_eq!(sol.stats.second_order, SecondOrderVerdict::Certified);
}

// ─────────────────────────────────────────────────────────────────
// The same defect with nothing to block the escape.
//
//     min ½x₁² − ½x₂²   (unbounded box, no rows)
//
// The origin is stationary and `(0, 1)` has curvature −1 with nothing to stop
// it, so the QP is unbounded below. Pre-fix: `Optimal`, `obj = 0`.
//
// This is the fixture that gives `pounce-convex`'s `ray_certifies_unbounded`
// its `dᵀPd < 0` branch a producer (gh #791). That branch was written for
// "the indefinite Hessians `solve_qp_active_set_inertia` admits" and, until
// this escape existed, nothing in the engine ever emitted such a ray: every
// `unbounded_ray` it produced came from a *zero*-curvature direction.
// ─────────────────────────────────────────────────────────────────
#[test]
fn an_unblocked_negative_curvature_direction_certifies_unbounded() {
    let h = sym2(1.0, 0.0, -1.0);
    let a = no_rows(2);
    let g = [0.0, 0.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [NEG_INF, NEG_INF];
    let xu = [POS_INF, POS_INF];
    let qp = QpProblem {
        n: 2,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Indefinite,
    };

    let sol = new_solver()
        .solve(&qp, None, &QpOptions::default())
        .expect("unbounded indefinite QP must return a verdict, not an error");

    assert_eq!(
        sol.status,
        QpStatus::Unbounded,
        "obj = {}, x = {:?}",
        sol.obj,
        sol.x
    );
    assert_eq!(
        sol.stats.second_order,
        SecondOrderVerdict::NegativeCurvature
    );
    let d = sol
        .unbounded_ray
        .as_ref()
        .expect("an Unbounded verdict must carry its witness");
    // The witness is the negative eigendirection, and it is the *curvature*
    // that certifies here — the gradient is zero, so a zero-curvature ray
    // would certify nothing.
    let hd = [d[0], -d[1]];
    let curv = hd[0] * d[0] + hd[1] * d[1];
    assert!(curv < -1e-8, "ray {d:?} has curvature {curv}, not negative");
    // The witness need not *be* the eigenvector — negative curvature is the
    // whole requirement — but it has to be dominated by the direction that
    // has it, or `x + td` would eventually curve back up.
    assert!(
        d[1].abs() > d[0].abs(),
        "ray {d:?} is not dominated by the negative eigendirection"
    );
}

// ─────────────────────────────────────────────────────────────────
// The `Certified` branch: an indefinite QP whose answer really is a minimum.
//
//     min ½x₁² − ½x₂² − ½x₂   s.t. −1 ≤ x ≤ 1
//
// Optimum `(0, 1)`, objective −1 (this is gh #416's box fixture). At that
// point `x₂` is at its upper bound, so the null space is the `x₁` axis and
// the reduced Hessian is `[1] ≻ 0`. The check must confirm rather than
// disturb it: a second-order test that rejects everything would pass every
// other test in this file.
// ─────────────────────────────────────────────────────────────────
#[test]
fn a_genuine_minimum_of_an_indefinite_qp_is_still_certified() {
    let h = sym2(1.0, 0.0, -1.0);
    let a = no_rows(2);
    let g = [0.0, -0.5];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [-1.0, -1.0];
    let xu = [1.0, 1.0];
    let qp = QpProblem {
        n: 2,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Indefinite,
    };

    let sol = new_solver()
        .solve(&qp, None, &QpOptions::default())
        .expect("indefinite box QP must solve");

    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        sol.x[0].abs() < 1e-9 && (sol.x[1] - 1.0).abs() < 1e-9,
        "expected x* = (0, 1), got {:?}",
        sol.x
    );
    assert!((sol.obj + 1.0).abs() < 1e-9, "obj = {}", sol.obj);
    assert_eq!(sol.stats.second_order, SecondOrderVerdict::Certified);
}

// ─────────────────────────────────────────────────────────────────
// The `NotChecked` branch, and the reason the check produces a witness
// instead of reading `δ`.
//
//     min ½x₁²   s.t. −1 ≤ x ≤ 1        (H = diag(1, 0), g = 0)
//
// Every point with `x₁ = 0` is a global minimum, objective 0 — a *weak*
// minimum, the whole `x₂` axis of them. The reduced Hessian at the origin is
// `diag(1, 0)`: singular, positive semi-definite, and no negative curvature
// anywhere.
//
// The KKT is singular there, so §4.5's ladder shifts and `δ > 0` — the same
// signal the saddle above produces. Rejecting on `δ > 0` would therefore
// downgrade this correct answer, which is a wrong verdict about the user's
// model and strictly worse than the defect being fixed. The probe finds no
// direction with `dᵀHd < 0` because none exists, reports `NotChecked`, and
// leaves the engine's finding alone.
// ─────────────────────────────────────────────────────────────────
#[test]
fn a_weak_minimum_is_not_rejected() {
    let h = sym2(1.0, 0.0, 0.0);
    let a = no_rows(2);
    let g = [0.0, 0.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [-1.0, -1.0];
    let xu = [1.0, 1.0];
    let qp = QpProblem {
        n: 2,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        // Claimed indefinite so the gate opens: this is about what the probe
        // does once it is running, not about whether it runs.
        hessian_inertia: HessianInertia::Indefinite,
    };

    let sol = new_solver()
        .solve(&qp, None, &QpOptions::default())
        .expect("semidefinite box QP must solve");

    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "a weak minimum is still a minimum; x = {:?}, obj = {}",
        sol.x,
        sol.obj
    );
    assert!(sol.obj.abs() < 1e-9, "obj = {}", sol.obj);
    assert!(sol.x[0].abs() < 1e-7, "x₁ = {} should be 0", sol.x[0]);
    assert_ne!(
        sol.stats.second_order,
        SecondOrderVerdict::NegativeCurvature,
        "no direction of negative curvature exists on H = diag(1, 0)"
    );
}

// ─────────────────────────────────────────────────────────────────
// A general row, not a bound, stops the escape.
//
//     min ½x₁² − ½x₂²   s.t.  −1 ≤ x₂ ≤ 1 (as a *row*),  −10 ≤ x ≤ 10
//
// Same saddle at the origin, but the box is slack there and the thing that
// stops the escape is `A x` rather than `x`. Optimum `(0, ±1)`, objective
// −0.5; at that point the row is active, the null space is the `x₁` axis, and
// the reduced Hessian is `[1] ≻ 0`, so the landing point certifies in its own
// right rather than stalling.
//
// This covers `Blocker::Cons`. Without it the ratio test's row arm — the
// half of `feasible_step_along` that has to fetch `Ax` and `Ad` and pick
// between `bl` and `bu` — is never executed by this file.
// ─────────────────────────────────────────────────────────────────
#[test]
fn a_general_row_can_be_the_blocker() {
    let h = sym2(1.0, 0.0, -1.0);
    let space = GenTMatrixSpace::new(1, 2, vec![1], vec![2]);
    let mut a = GenTMatrix::new(space);
    a.set_values(&[1.0]);
    let g = [0.0, 0.0];
    let bl = [-1.0];
    let bu = [1.0];
    let xl = [-10.0, -10.0];
    let xu = [10.0, 10.0];
    let qp = QpProblem {
        n: 2,
        m: 1,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Indefinite,
    };

    let sol = new_solver()
        .solve(&qp, None, &QpOptions::default())
        .expect("row-blocked indefinite QP must solve");

    assert_eq!(sol.status, QpStatus::Optimal, "x = {:?}", sol.x);
    assert!(
        (sol.obj + 0.5).abs() < 1e-6,
        "objective {} at x = {:?}; the minimum is −0.5 at (0, ±1)",
        sol.obj,
        sol.x
    );
    assert!(sol.x[0].abs() < 1e-6, "x₁ = {} should be 0", sol.x[0]);
    assert!(
        (sol.x[1].abs() - 1.0).abs() < 1e-6,
        "x₂ = {} should be ±1 (the row, not a bound, is what stops it)",
        sol.x[1]
    );
    assert_eq!(sol.stats.second_order, SecondOrderVerdict::Certified);
}

// ─────────────────────────────────────────────────────────────────
// The gate. A PSD claim skips the check outright, so the convex arm pays
// nothing for any of the above — no extra factorization, no probe.
//
// Same problem as `a_genuine_minimum_of_an_indefinite_qp_is_still_certified`
// but claimed PSD, which is a *lie* about this `H`. The point is that the
// verdict is `NotChecked` rather than anything else: the engine took the
// caller at its word and never ran the test.
// ─────────────────────────────────────────────────────────────────
#[test]
fn a_psd_claim_skips_the_check_entirely() {
    let h = sym2(1.0, 0.0, -1.0);
    let a = no_rows(2);
    let g = [0.0, -0.5];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [-1.0, -1.0];
    let xu = [1.0, 1.0];
    let qp = QpProblem {
        n: 2,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let sol = new_solver()
        .solve(&qp, None, &QpOptions::default())
        .expect("QP must solve");

    assert_eq!(sol.stats.second_order, SecondOrderVerdict::NotChecked);
}

// ─────────────────────────────────────────────────────────────────
// `certify_second_order = false` restores the pre-#848 behaviour exactly,
// including the defect. Pinned so the knob is known to be a real off switch
// rather than a field nothing reads — which is how gh #677 shipped
// (`limited_memory_initialization` was registered and never read).
// ─────────────────────────────────────────────────────────────────
#[test]
fn the_check_can_be_switched_off_and_then_the_saddle_comes_back() {
    let h = sym2(1.0, 5.0, 1.0);
    let a = no_rows(2);
    let g = [0.0, 0.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [-1.0, -1.0];
    let xu = [1.0, 1.0];
    let qp = QpProblem {
        n: 2,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Indefinite,
    };

    let opts = QpOptions {
        certify_second_order: false,
        ..QpOptions::default()
    };
    let sol = new_solver().solve(&qp, None, &opts).expect("must solve");

    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        sol.obj.abs() < 1e-9,
        "with the check off this is the gh #848 saddle: obj should be 0, got {}",
        sol.obj
    );
    assert_eq!(sol.stats.second_order, SecondOrderVerdict::NotChecked);
}

// ─────────────────────────────────────────────────────────────────
// gh #423's opt-out has to survive gh #848.
//
// `certify_recession_ray = false` means "give me a point, not a verdict".
// The box path has honoured it at its own unblocked-negative-curvature exit
// since gh #423, and the escape has to honour it at the same exit for the
// same reason — the SQP's unbounded-model fallback sets the flag and
// re-solves *because* the caller has nowhere to go with an `Unbounded`. A
// re-solve that comes straight back `Unbounded` leaves the outer loop with no
// step at all, which is gh #419 verbatim.
//
// Measured cost of getting this wrong, on `eigenb2` under
// `algorithm=active-set-sqp` (110 free variables, 55 equalities, nothing that
// can ever block a direction): 200 iterations at f = 1.6013 collapsed to 1
// iteration at f = 24.026, exiting `Search_Direction_Becomes_Too_Small`.
//
// The *finding* still travels — only the action is declined — so a caller
// that reads `stats.second_order` still learns the point is refuted. This
// pairs with `an_unblocked_negative_curvature_direction_certifies_unbounded`
// above: same fixture, the two sides of the same branch.
// ─────────────────────────────────────────────────────────────────
#[test]
fn declining_recession_verdicts_keeps_the_point_and_still_reports_the_finding() {
    let h = sym2(1.0, 0.0, -1.0);
    let a = no_rows(2);
    let g = [0.0, 0.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [NEG_INF, NEG_INF];
    let xu = [POS_INF, POS_INF];
    let qp = QpProblem {
        n: 2,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Indefinite,
    };

    let opts = QpOptions {
        certify_recession_ray: false,
        ..QpOptions::default()
    };
    let sol = new_solver().solve(&qp, None, &opts).expect("must solve");

    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "a caller that declined recession verdicts must still get a usable \
         point (gh #423); obj = {}, x = {:?}",
        sol.obj,
        sol.x
    );
    assert!(
        sol.unbounded_ray.is_none(),
        "no verdict was asked for, so no certificate should be attached"
    );
    assert_eq!(
        sol.stats.second_order,
        SecondOrderVerdict::NegativeCurvature,
        "the action is declined, not the finding"
    );
}
