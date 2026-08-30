//! gh #848 at the convex driver: a first-order-clean saddle must not come back
//! `Optimal`, and the engine's second-order finding is the only channel that
//! can say so.
//!
//! `pounce-qp` refuses the saddle in [`pounce_qp::QpSolver::solve`] (see
//! `pounce-qp/tests/issue848_second_order_certification.rs`). That refusal is
//! not enough on its own, because this crate does not propagate the engine's
//! status — [`verify_status`] re-derives a verdict from the *returned point*,
//! and everything it measures is a first-order residual. A saddle of an
//! indefinite `P` satisfies those conditions exactly, so the re-derivation
//! would overwrite the refusal with `Optimal`: the guard one layer down would
//! be undone by the guard one layer up. `QpStats::second_order` is what stops
//! that, and `the_residual_alone_would_promote_the_refuted_point` is the test
//! that this file's other assertions are actually earning something.
//!
//! It is also where gh #791's `dᵀPd < 0` branch in `ray_certifies_unbounded`
//! gets its first producer. That branch was written for "the indefinite
//! Hessians `solve_qp_active_set_inertia` admits" and, until the escape
//! existed, nothing ever handed it a ray: the engine's `Unbounded` exits all
//! came from zero-curvature directions.
//!
//! ## Which branch each test reaches
//!
//! | test | engine status in | verdict | what it catches |
//! |---|---|---|---|
//! | `a_saddle_of_an_indefinite_qp_is_not_reported_optimal` | `Optimal` after a successful escape | `Certified` | the escape not running under this driver at all |
//! | `an_unbounded_indefinite_qp_certifies_dual_infeasible` | `Unbounded` | `NegativeCurvature` | the `dᵀPd < 0` ray branch losing its producer, or the ray being dropped in translation |
//! | `the_residual_alone_would_promote_the_refuted_point` | `MaxIter` | both | the whole wiring being a no-op because the point is KKT-clean |
//! | `a_convex_qp_is_untouched` | `Optimal` | `NotChecked` | the guard leaking onto the PSD path it must cost nothing on |
//!
//! The first three run on an **indefinite** claim, which is the only claim
//! that reaches any of this: under `HessianInertia::Psd` the engine never runs
//! the test and `second_order` stays `NotChecked`, which `solved_to` treats
//! exactly as it always did.

use pounce_convex::{
    ActiveSetOverrides, HessianInertia, QpOptions, QpProblem, QpSolution, QpStatus,
    SecondOrderVerdict, Triplet, solve_qp_active_set_inertia, verify_status,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
// The engine's own status enum. `verify_status` takes it — it is the verdict
// being adjudicated, not the one being returned — and it is a different type
// from this crate's `QpStatus`, which is what comes back.
use pounce_qp::QpStatus as ActiveSetStatus;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve_indefinite(prob: &QpProblem) -> QpSolution {
    let mut mk = backend;
    solve_qp_active_set_inertia(
        prob,
        &QpOptions::default(),
        &ActiveSetOverrides::default(),
        HessianInertia::Indefinite,
        &mut mk,
    )
}

/// `min ½xᵀPx` over `[−1, 1]²` with `P = [[1, 5], [5, 1]]` and `c = 0`.
///
/// The origin is stationary (`Px + c = 0`) with an empty working set, so it is
/// first-order optimal on every measure this crate applies — and it is a
/// saddle: `P` has eigenvalues `6` and `−4`, and along `(1, −1)/√2` the
/// objective falls. The true minimum is at the corners `±(1, −1)`, where
/// `½(1 − 10 + 1) = −4`.
fn saddle_box_qp() -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 0, 5.0),
            Triplet::new(1, 1, 1.0),
        ],
        c: vec![0.0, 0.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![-1.0, -1.0],
        ub: vec![1.0, 1.0],
    }
}

#[test]
fn a_saddle_of_an_indefinite_qp_is_not_reported_optimal() {
    let sol = solve_indefinite(&saddle_box_qp());
    assert_eq!(sol.status, QpStatus::Optimal, "x = {:?}", sol.x);
    assert!(
        (sol.obj - (-4.0)).abs() < 1e-7,
        "expected the corner minimum −4, got {} at {:?} — 0.0 is the saddle",
        sol.obj,
        sol.x
    );
    // Either corner is a global minimum; both have `|x₀| = |x₁| = 1` with
    // opposite signs.
    assert!(
        (sol.x[0].abs() - 1.0).abs() < 1e-7
            && (sol.x[1].abs() - 1.0).abs() < 1e-7
            && sol.x[0] * sol.x[1] < 0.0,
        "expected a corner ±(1, −1), got {:?}",
        sol.x
    );
}

/// `min ½(x₀² − x₁²)`, both variables free. Nothing blocks travel along the
/// `x₁` axis and the objective falls quadratically: unbounded below, and the
/// only witness is a direction of **negative curvature** — `Pd ≈ 0` is false
/// along it, so the zero-curvature half of `ray_certifies_unbounded` cannot
/// certify this and the `dᵀPd < 0` half must.
fn unbounded_indefinite_qp() -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, -1.0)],
        c: vec![0.0, 0.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    }
}

#[test]
fn an_unbounded_indefinite_qp_certifies_dual_infeasible() {
    let sol = solve_indefinite(&unbounded_indefinite_qp());
    assert_eq!(
        sol.status,
        QpStatus::DualInfeasible,
        "expected the ray to certify unboundedness, got {:?} at {:?} (obj {})",
        sol.status,
        sol.x,
        sol.obj
    );
}

/// The mutation guard for everything above: on the *same point*, the residual
/// band promotes to `Optimal` when the verdict is `NotChecked` and refuses
/// when it is `NegativeCurvature`.
///
/// Without this, a wiring that silently dropped the verdict would still leave
/// the two tests above green for the wrong reason (`pounce-qp` having escaped
/// before the point ever got here), and the one case this crate has to handle
/// on its own — an engine that refuted the point but could not get off it,
/// which is `MaxIter` with a live witness — would be unguarded.
#[test]
fn the_residual_alone_would_promote_the_refuted_point() {
    let prob = saddle_box_qp();
    let opts = QpOptions::default();
    // The saddle, exactly: stationary, feasible, no bound active, so every
    // multiplier is zero and every first-order residual is zero.
    let saddle = QpSolution {
        status: QpStatus::Optimal,
        x: vec![0.0, 0.0],
        y: vec![],
        z: vec![],
        z_lb: vec![0.0; 2],
        z_ub: vec![0.0; 2],
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    };

    assert_eq!(
        verify_status(
            ActiveSetStatus::MaxIter,
            None,
            SecondOrderVerdict::NotChecked,
            &saddle,
            &prob,
            &opts
        ),
        QpStatus::Optimal,
        "the first-order residual at a saddle is clean — this is the defect, \
         and it must stay reproducible or the assertion below proves nothing"
    );
    assert_eq!(
        verify_status(
            ActiveSetStatus::MaxIter,
            None,
            SecondOrderVerdict::NegativeCurvature,
            &saddle,
            &prob,
            &opts
        ),
        QpStatus::IterationLimit,
        "a refuted point is not salvaged by its residual"
    );
    // `Certified` is not a third state here: it must behave exactly like the
    // unrefuted case, or certifying a genuine minimum would cost it its status.
    assert_eq!(
        verify_status(
            ActiveSetStatus::MaxIter,
            None,
            SecondOrderVerdict::Certified,
            &saddle,
            &prob,
            &opts
        ),
        QpStatus::Optimal
    );
}

/// `min (x₀−3)² + (x₁−2)²  s.t.  x₀ + x₁ ≤ 4`, solved under the PSD claim the
/// convex driver always makes. The verdict stays `NotChecked` and the banding
/// is the one it always was.
#[test]
fn a_convex_qp_is_untouched() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-6.0, -4.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![4.0],
        lb: vec![],
        ub: vec![],
    };
    let mut mk = backend;
    let sol = solve_qp_active_set_inertia(
        &prob,
        &QpOptions::default(),
        &ActiveSetOverrides::default(),
        HessianInertia::Psd,
        &mut mk,
    );
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.obj - (-12.5)).abs() < 1e-8, "obj = {}", sol.obj);
}
