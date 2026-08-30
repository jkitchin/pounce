//! An indefinite QP's claimed optimum must survive a **second-order** check
//! (gh #848), and this file's oracle is arithmetic: an exhibited feasible point
//! with a strictly lower objective.
//!
//! # What regressed
//!
//! gh #112 added `check_psd` precisely because `solve_qp` "accepts an
//! indefinite `P` and returns a silently-wrong `optimal`". gh #786 then scoped
//! that guard away from `method='active-set'` on the premise that the
//! active-set engine gives "a **local** optimum, the same guarantee the NLP
//! filter-IPM gives on a nonconvex NLP". It did not give that guarantee, and a
//! refusal became a confident wrong answer: at `v0.10.0` `dispatch.rs` refused
//! the class outright, at HEAD it dispatched it.
//!
//! ```text
//!   P = [[1, 5], [5, 1]], c = 0, box [-1, 1]^2,  eigvalsh(P) = [-4, 6]
//!   qp-active-set -> Optimal / Solve_Succeeded / success=True, x = [0, 0], f = 0
//!   but x = [1, -1] is feasible with f = -4, and x = [0.1, -0.1] gives -0.04
//! ```
//!
//! Started essentially **at** the global minimum (`x0 = [0.99, -0.99]`,
//! `f = -3.92`) the engine still returned `f ≈ 0` and certified it — it moved
//! uphill and reported success, and the start point was ignored entirely, so no
//! "local optimum from `x0`" reading rescues it. No bound is active at the
//! returned point, so the reduced Hessian **is** `P`: the point is a strict
//! saddle.
//!
//! # Why nothing caught it
//!
//! `verify_status` re-derives only *first-order* KKT residuals. For a convex QP
//! first-order KKT is equivalent to global optimality, so the guard was sound
//! for every input the engine had ever been given; for the indefinite inputs
//! gh #791 newly admits it is necessary and not sufficient. `pounce-qp`'s
//! inertia control shifts the KKT diagonal so the *factorization* has the right
//! inertia — that makes the linear algebra work and is not a second-order test
//! at the returned point, though `dispatch.rs`'s doc comment read the former as
//! the latter.
//!
//! gh #791's negative-curvature branch of `ray_certifies_unbounded` could not
//! fire: its only call site is inside the `Unbounded` arm of `verify_status`,
//! and the engine claims `Optimal`, so the arm is never reached. Its unit tests
//! call the function directly and stayed green regardless. This file reaches it
//! from the `Optimal` side.
//!
//! And `scripts/sweep-fixtures.sh` cannot see any of it: no nonconvex-QP
//! fixture is routed to `qp-active-set`, and `auto` still sends the class to the
//! NLP arm. CLAUDE.md's rule verbatim — a leg is only evidence about the branch
//! its fixture reaches.

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_active_set_inertia};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_qp::{ActiveSetOverrides, HessianInertia};

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve(prob: &QpProblem, inertia: HessianInertia) -> pounce_convex::QpSolution {
    let mut mk = backend;
    solve_qp_active_set_inertia(
        prob,
        &QpOptions::default(),
        &ActiveSetOverrides::default(),
        inertia,
        &mut mk,
    )
}

/// `½xᵀPx + cᵀx`, evaluated by hand so the comparison never routes through the
/// engine that is under test.
fn objective(prob: &QpProblem, x: &[f64]) -> f64 {
    let mut px = vec![0.0; prob.n];
    prob.p_mul(x, &mut px);
    0.5 * (0..prob.n).map(|i| x[i] * px[i]).sum::<f64>()
        + (0..prob.n).map(|i| prob.c[i] * x[i]).sum::<f64>()
}

/// The reported instance: `P = [[1, 5], [5, 1]]`, `c = 0`, box `[-1, 1]²`.
fn saddle_qp() -> QpProblem {
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

/// `min −x₀² + ½x₁²` over `x₀ ≥ 0`: unbounded below along `+x₀`.
fn unbounded_qp() -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, -2.0), Triplet::new(1, 1, 1.0)],
        c: vec![0.0, 0.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0, f64::NEG_INFINITY],
        ub: vec![f64::INFINITY, f64::INFINITY],
    }
}

/// The headline. Whatever the engine returns, it must not be a point that a
/// feasible point exhibited right here beats — that is oracle 1, arithmetic,
/// no solver in the loop.
#[test]
fn the_reported_saddle_is_not_certified_as_optimal() {
    let prob = saddle_qp();
    let sol = solve(&prob, HessianInertia::Indefinite);
    let f_here = objective(&prob, &sol.x);
    for witness in [[1.0, -1.0], [-1.0, 1.0], [0.1, -0.1]] {
        let f_there = objective(&prob, &witness);
        assert!(
            !(sol.status == QpStatus::Optimal && f_there < f_here - 1e-9),
            "status {:?} at x = {:?} (f = {f_here:.6}), but the feasible point \
             {witness:?} gives f = {f_there:.6} (the issue reports Optimal at \
             f = 0 against f = -4)",
            sol.status,
            sol.x
        );
    }
}

/// The start point was ignored, so the defect is reachable from anywhere —
/// including from a point that is already better than what comes back. This is
/// the "it moved uphill and reported success" half of the report, and the
/// engine has no `x0` input, so it is asserted the only way it can be: the
/// returned objective must be at least as good as the best witness.
#[test]
fn the_returned_point_is_no_worse_than_an_exhibited_feasible_one() {
    let prob = saddle_qp();
    let sol = solve(&prob, HessianInertia::Indefinite);
    if sol.status != QpStatus::Optimal {
        return; // an honest non-verdict is allowed; a wrong Optimal is not
    }
    let f_here = objective(&prob, &sol.x);
    assert!(
        f_here <= -4.0 + 1e-6,
        "an Optimal on this box QP has to be one of the two global minima \
         (f = -4); got f = {f_here:.6} at x = {:?}",
        sol.x
    );
}

/// A QP that is genuinely unbounded below must not come back `Optimal` with
/// `obj = 0` and `iters = 0`. The right verdict is the unboundedness
/// certificate, which is what POUNCE's own NLP arm gives on the same model
/// (`Diverging_Iterates`) — oracle 3.
#[test]
fn an_unbounded_indefinite_qp_is_certified_unbounded_not_optimal() {
    let prob = unbounded_qp();
    let sol = solve(&prob, HessianInertia::Indefinite);
    assert_ne!(
        sol.status,
        QpStatus::Optimal,
        "unbounded below, but reported Optimal at x = {:?}, obj = {}",
        sol.x,
        sol.obj
    );
    assert_eq!(
        sol.status,
        QpStatus::DualInfeasible,
        "the ray is a feasible recession direction of negative curvature, \
         which `ray_certifies_unbounded` already knows how to certify — it was \
         only ever unreachable from the `Optimal` side"
    );
}

/// A **convex** QP is untouched: the caller's `HessianInertia::Psd` claim is
/// the gate, so the screen never runs on the arm every existing consumer uses,
/// and the verdict and the point are what they always were.
#[test]
fn the_convex_path_is_not_touched() {
    // `min ½(x₀² + x₁²) − x₀` over `[-1, 1]²`, optimum `(1/1, 0)` → x₀ = 1 is
    // interior to the box at 1? No: the unconstrained optimum is x₀ = 1, on
    // the bound, so the box is active and the answer is exact.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-0.5, 0.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![-1.0, -1.0],
        ub: vec![1.0, 1.0],
    };
    let sol = solve(&prob, HessianInertia::Psd);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 0.5).abs() < 1e-8, "x = {:?}", sol.x);
    assert!(sol.x[1].abs() < 1e-8, "x = {:?}", sol.x);
}

/// An indefinite QP whose returned point **is** a genuine local minimum must
/// keep its `Optimal`. Without this the fix could be "never certify an
/// indefinite QP", which would pass every test above while making the engine
/// useless for the class gh #786 admitted it for.
///
/// `min ½(x₀² − 4x₁²)` over `x₀ ∈ [-1, 1]`, `x₁ ∈ [-1, 1]`: indefinite
/// (`eig = [-4, 1]`), and `x = (0, ±1)` is a genuine local **and** global
/// minimum at `f = -2` — the negative direction is pinned against its bound,
/// so no feasible direction has negative curvature.
#[test]
fn a_genuine_local_minimum_of_an_indefinite_qp_keeps_its_verdict() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, -4.0)],
        c: vec![0.0, 0.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![-1.0, -1.0],
        ub: vec![1.0, 1.0],
    };
    let sol = solve(&prob, HessianInertia::Indefinite);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "a real local minimum must still be certified, got {:?} at x = {:?}",
        sol.status,
        sol.x
    );
    let f = objective(&prob, &sol.x);
    assert!(
        (f + 2.0).abs() < 1e-6,
        "expected the global minimum f = -2, got {f:.6} at x = {:?}",
        sol.x
    );
}

/// The screen builds its direction from the curvature of `P`, and the two
/// searches it runs cover two different geometries. This is the branch rule
/// from CLAUDE.md applied to the fix itself: the saddle above sits in the
/// **interior** of the box, and the unbounded model's negative direction sits
/// **on** a weakly active bound, where an interior-only search sees only the
/// positive curvature of the other coordinate and finds nothing. Both fixtures
/// are needed; neither is evidence about the other's branch.
#[test]
fn both_direction_searches_are_load_bearing() {
    // interior: no bound is active at the returned saddle
    let interior = solve(&saddle_qp(), HessianInertia::Indefinite);
    assert!(
        interior.status != QpStatus::Optimal || objective(&saddle_qp(), &interior.x) <= -4.0 + 1e-6,
        "interior branch"
    );
    // on-bound: x₀ sits at its lower bound with a zero multiplier
    let on_bound = solve(&unbounded_qp(), HessianInertia::Indefinite);
    assert_ne!(on_bound.status, QpStatus::Optimal, "on-bound branch");
}
