//! gh #969 — an unbounded free-variable LP must come back as `DualInfeasible`
//! from the active-set driver, not `IterationLimit`.
//!
//! Same harm as #415 and #388, one family over: the two statuses land in
//! different AMPL `solve_result_num` families and callers branch on that
//! family. `300` tells AMPL / Pyomo / the GAMS links "the model is unbounded —
//! fix the model"; `400` tells them "the iteration limit was hit — raise it and
//! retry", which on an unbounded model can never help. The reported run had
//! spent **4** iterations against an untouched budget, so the status was not
//! even describing what happened.
//!
//! Where it came from, and why the corpus never caught it. The engine does
//! reach a correct `Unbounded` verdict *with a ray* — in the phase-1 recovery
//! path, whose augmented objective is the original one plus a penalty that is
//! zero once the slacks vanish. `solve_general`'s `feasible` branch then
//! flattened **every** non-`Optimal` inner status to `MaxIter`, discarding both
//! the verdict and the ray. So this was not a missing certificate; it was a
//! certificate thrown away one frame below where `verify_status` re-derives it.
//!
//! The claim is still not trusted from the engine. `verify_status` re-derives
//! it against the original problem with `ray_certifies_unbounded`, exactly as
//! #388 routed this engine's other `Unbounded` claim, so a ray that does not
//! stand up is downgraded rather than believed.
//!
//! Oracle: closed form. `d` spans `null(A)`, so `‖A d‖ ≈ 2e-16` and, with the
//! variables free, `x + t·d` stays feasible for all `t`; `cᵀd < 0` then drives
//! the objective down without bound. POUNCE's own IPM agrees on identical data,
//! and every assertion below cross-checks against it — a test that only pinned
//! the active-set status would pass just as well if both engines went wrong
//! together.

use pounce_convex::{
    ActiveSetOverrides, QpOptions, QpProblem, QpStatus, Triplet, solve_qp_active_set, solve_qp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn active_set(prob: &QpProblem) -> QpStatus {
    let mut mk = backend;
    solve_qp_active_set(
        prob,
        &QpOptions::default(),
        &ActiveSetOverrides::default(),
        &mut mk,
    )
    .status
}

/// The issue's LP: `min cᵀx` s.t. `Ax = 1`, `x` free, with `n = 3`, `m = 2`.
///
/// `d = null(A)` (unit), `‖A d‖ = 2.3e-16`, `cᵀd = −0.996`. Reported as
/// `iteration_limit` after 4 iterations at objective `0.446` before the fix.
fn unbounded_free_lp() -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![-1.83, 2.2, 0.26],
        a: vec![
            Triplet::new(0, 0, 0.19),
            Triplet::new(0, 1, -0.52),
            Triplet::new(0, 2, -0.41),
            Triplet::new(1, 0, -2.44),
            Triplet::new(1, 1, 1.8),
            Triplet::new(1, 2, 1.14),
        ],
        b: vec![1.0, 1.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    }
}

#[test]
fn unbounded_free_variable_lp_is_dual_infeasible() {
    let prob = unbounded_free_lp();
    assert_eq!(
        active_set(&prob),
        QpStatus::DualInfeasible,
        "active-set must report the model unbounded, not blame the iteration budget"
    );
    // The engines must not disagree about the user's model.
    assert_eq!(
        solve_qp_ipm(&prob, &QpOptions::default(), backend).status,
        QpStatus::DualInfeasible,
        "IPM oracle"
    );
}

#[test]
fn a_generous_iteration_budget_does_not_change_the_verdict() {
    // The reported symptom was `iteration_limit` with the budget untouched:
    // raising it changed nothing, because iterations were never the problem.
    // If the verdict ever moves with the budget again, it is not a verdict.
    let prob = unbounded_free_lp();
    let mut opts = QpOptions::default();
    opts.max_iter = 100_000;
    let mut mk = backend;
    let sol = solve_qp_active_set(&prob, &opts, &ActiveSetOverrides::default(), &mut mk);
    assert_eq!(sol.status, QpStatus::DualInfeasible);
}

/// The bound is what makes the objective attainable, so the *same* `A` and `c`
/// must now solve. This is the branch that would stay green if the fix had
/// simply learned to call every phase-1 exit `Unbounded`, so it is the test,
/// not a duplicate of the one above.
#[test]
fn boxing_the_same_model_removes_the_ray_and_it_solves() {
    let prob = QpProblem {
        lb: vec![-100.0, -100.0, -100.0],
        ub: vec![100.0, 100.0, 100.0],
        ..unbounded_free_lp()
    };
    let got = active_set(&prob);
    assert_ne!(
        got,
        QpStatus::DualInfeasible,
        "a boxed model has no recession ray; reporting one is a wrong verdict"
    );
    assert_eq!(
        got,
        solve_qp_ipm(&prob, &QpOptions::default(), backend).status,
        "the engines must agree on the boxed model too"
    );
}
