//! §4.2 parametric homotopy — does it reach the same answer as the
//! conventional phase-1/phase-2 path?
//!
//! The homotopy is the algorithm this crate is named for and was previously
//! unimplemented (`solve_parametric` was a stub). It is off by default while
//! being evaluated, so these tests drive it explicitly via
//! `QpOptions::use_homotopy` and assert it agrees with the conventional path on
//! problems with closed-form optima.
//!
//! Scope note, so a future reader is not misled by green tests: the homotopy
//! currently starts from the **box-only relaxation**, which means it cannot
//! start when that relaxation is unbounded (`H` singular in a box-unbounded
//! direction — most LP-like instances), and it has no anti-cycling for
//! *coincident* events, so it can stall at a degenerate parameter value. Both
//! cases return `Ok(None)` internally and fall back to the conventional path, so
//! they are invisible in results but real. See `crates/pounce-qp/src/homotopy.rs`.

use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver, QpStatus,
};
use std::rc::Rc;

fn new_solver() -> ParametricActiveSetSolver {
    ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()))
}

struct Case {
    n: usize,
    m: usize,
    h: (Vec<i32>, Vec<i32>, Vec<f64>),
    g: Vec<f64>,
    a: (Vec<i32>, Vec<i32>, Vec<f64>),
    bl: Vec<f64>,
    bu: Vec<f64>,
    xl: Vec<f64>,
    xu: Vec<f64>,
}

fn solve(case: &Case, homotopy: bool) -> (QpStatus, Vec<f64>, f64) {
    let h_space = SymTMatrixSpace::new(case.n as i32, case.h.0.clone(), case.h.1.clone());
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&case.h.2);

    let a_space = GenTMatrixSpace::new(
        case.m as i32,
        case.n as i32,
        case.a.0.clone(),
        case.a.1.clone(),
    );
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&case.a.2);

    let qp = QpProblem {
        n: case.n,
        m: case.m,
        h: &h,
        g: &case.g,
        a: &a,
        bl: &case.bl,
        bu: &case.bu,
        xl: &case.xl,
        xu: &case.xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let opts = QpOptions {
        use_homotopy: homotopy,
        ..QpOptions::default()
    };
    let sol = new_solver().solve(&qp, None, &opts).expect("qp solve");
    (sol.status, sol.x.clone(), sol.obj)
}

/// `min (x₀−3)² + (x₁−2)²  s.t.  x₀ + x₁ ≤ 4,  x ≥ 0`, in `½xᵀHx + gᵀx` form:
/// `H = 2I`, `g = (−6, −4)` (constant 13 dropped).
///
/// `H` is positive definite so the box relaxation is bounded and the homotopy
/// can start. The unconstrained optimum `(3, 2)` violates the row, so it binds:
/// `x* = (2.5, 1.5)`, and `obj = 6.25 + 2.25 − 15 − 6 = −12.5`.
fn projection_case() -> Case {
    Case {
        n: 2,
        m: 1,
        h: (vec![1, 2], vec![1, 2], vec![2.0, 2.0]),
        g: vec![-6.0, -4.0],
        a: (vec![1, 1], vec![1, 2], vec![1.0, 1.0]),
        bl: vec![NLP_LOWER_BOUND_INF],
        bu: vec![4.0],
        xl: vec![0.0, 0.0],
        xu: vec![NLP_UPPER_BOUND_INF, NLP_UPPER_BOUND_INF],
    }
}

/// Two rows, both binding at the optimum, so the path must add more than one
/// constraint on the way to `t = 1`.
///
/// `min ½(x₀² + x₁²) − 4x₀ − 4x₁  s.t.  x₀ + x₁ ≤ 4,  x₀ ≤ 1.5,  x ≥ 0`.
/// Unconstrained optimum is `(4, 4)`. With `x₀ ≤ 1.5` binding, minimizing over
/// `x₁` subject to `x₀ + x₁ ≤ 4` gives `x₁ = 2.5`, so `x* = (1.5, 2.5)` and
/// `obj = ½(2.25 + 6.25) − 6 − 10 = 4.25 − 16 = −11.75`.
fn two_active_case() -> Case {
    Case {
        n: 2,
        m: 2,
        h: (vec![1, 2], vec![1, 2], vec![1.0, 1.0]),
        g: vec![-4.0, -4.0],
        a: (vec![1, 1, 2], vec![1, 2, 1], vec![1.0, 1.0, 1.0]),
        bl: vec![NLP_LOWER_BOUND_INF, NLP_LOWER_BOUND_INF],
        bu: vec![4.0, 1.5],
        xl: vec![0.0, 0.0],
        xu: vec![NLP_UPPER_BOUND_INF, NLP_UPPER_BOUND_INF],
    }
}

#[test]
fn homotopy_solves_projection_qp() {
    let (status, x, obj) = solve(&projection_case(), true);
    assert_eq!(status, QpStatus::Optimal, "status");
    assert!((x[0] - 2.5).abs() < 1e-8, "x0 = {}", x[0]);
    assert!((x[1] - 1.5).abs() < 1e-8, "x1 = {}", x[1]);
    assert!((obj + 12.5).abs() < 1e-8, "obj = {obj}");
}

#[test]
fn homotopy_solves_two_active_rows() {
    let (status, x, obj) = solve(&two_active_case(), true);
    assert_eq!(status, QpStatus::Optimal, "status");
    assert!((x[0] - 1.5).abs() < 1e-8, "x0 = {}", x[0]);
    assert!((x[1] - 2.5).abs() < 1e-8, "x1 = {}", x[1]);
    assert!((obj + 11.75).abs() < 1e-8, "obj = {obj}");
}

/// The homotopy is a *path*, not a different answer: it must agree with the
/// conventional phase-1/phase-2 path everywhere it runs. Any disagreement means
/// one of the two is wrong, which is the property worth pinning.
#[test]
fn homotopy_agrees_with_conventional_path() {
    for (name, case) in [
        ("projection", projection_case()),
        ("two_active", two_active_case()),
    ] {
        let (cs, cx, cobj) = solve(&case, false);
        let (hs, hx, hobj) = solve(&case, true);
        assert_eq!(hs, cs, "{name}: status differs");
        for i in 0..case.n {
            assert!(
                (hx[i] - cx[i]).abs() < 1e-7,
                "{name}: x[{i}] homotopy {} vs conventional {}",
                hx[i],
                cx[i]
            );
        }
        assert!(
            (hobj - cobj).abs() < 1e-7,
            "{name}: obj homotopy {hobj} vs conventional {cobj}"
        );
    }
}

// ---------------------------------------------------------------------------
// Parametric (warm) solves — `QpSolver::solve_parametric`.
//
// This was a stub that discarded both prior arguments and cold-solved, despite
// the crate advertising "true parametric warm starting". It now traces the
// homotopy from the previous problem to the new one, starting from the previous
// solution's working set.
// ---------------------------------------------------------------------------

/// Solve `case` from cold, then re-solve a `g`-perturbed version parametrically
/// from that solution, and return both the warm result and the cold result for
/// the *same* perturbed problem.
fn parametric_vs_cold(
    case: &Case,
    dg: &[f64],
) -> ((QpStatus, Vec<f64>, f64), (QpStatus, Vec<f64>, f64)) {
    let h_space = SymTMatrixSpace::new(case.n as i32, case.h.0.clone(), case.h.1.clone());
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&case.h.2);
    let a_space = GenTMatrixSpace::new(
        case.m as i32,
        case.n as i32,
        case.a.0.clone(),
        case.a.1.clone(),
    );
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&case.a.2);

    let g_new: Vec<f64> = case.g.iter().zip(dg).map(|(a, b)| a + b).collect();
    let opts = QpOptions {
        use_homotopy: true,
        ..QpOptions::default()
    };
    // A closure returning `QpProblem<'_>` cannot tie the borrow of `g` to the
    // returned value's lifetime, so build them explicitly.
    macro_rules! mk {
        ($g:expr) => {
            QpProblem {
                n: case.n,
                m: case.m,
                h: &h,
                g: $g,
                a: &a,
                bl: &case.bl,
                bu: &case.bu,
                xl: &case.xl,
                xu: &case.xu,
                hessian_inertia: HessianInertia::Psd,
            }
        };
    }

    let mut s = new_solver();
    let prev = s.solve(&mk!(&case.g), None, &opts).expect("cold prev");
    let warm = s
        .solve_parametric(&mk!(&case.g), &prev, &mk!(&g_new), &opts)
        .expect("parametric");
    let cold = new_solver()
        .solve(&mk!(&g_new), None, &opts)
        .expect("cold new");
    (
        (warm.status, warm.x.clone(), warm.obj),
        (cold.status, cold.x.clone(), cold.obj),
    )
}

/// A warm parametric solve must land on the same answer as a cold solve of the
/// same target. Warm starting is a route, not a different problem.
#[test]
fn parametric_matches_cold_solve() {
    for (name, case, dg) in [
        ("projection", projection_case(), vec![0.5, -0.25]),
        ("two_active", two_active_case(), vec![-1.0, 0.75]),
    ] {
        let ((ws, wx, wobj), (cs, cx, cobj)) = parametric_vs_cold(&case, &dg);
        assert_eq!(ws, cs, "{name}: status warm {ws:?} vs cold {cs:?}");
        for i in 0..case.n {
            assert!(
                (wx[i] - cx[i]).abs() < 1e-7,
                "{name}: x[{i}] warm {} vs cold {}",
                wx[i],
                cx[i]
            );
        }
        assert!(
            (wobj - cobj).abs() < 1e-7,
            "{name}: obj warm {wobj} vs cold {cobj}"
        );
    }
}

/// Re-solving an **unchanged** QP parametrically must be nearly free: the path
/// has zero length, so no constraint can reach a bound and no multiplier can
/// reach zero along it. This is the property that makes warm starting worth
/// having, and the one a stub silently fails while still returning the right
/// answer — so asserting the answer alone would not catch a regression here.
#[test]
fn parametric_on_unchanged_qp_is_free() {
    for (name, case) in [
        ("projection", projection_case()),
        ("two_active", two_active_case()),
    ] {
        let ((ws, wx, wobj), (_, cx, cobj)) = parametric_vs_cold(&case, &vec![0.0; case.n]);
        assert_eq!(ws, QpStatus::Optimal, "{name}: status");
        for i in 0..case.n {
            assert!(
                (wx[i] - cx[i]).abs() < 1e-9,
                "{name}: x[{i}] moved on an unchanged re-solve"
            );
        }
        assert!(
            (wobj - cobj).abs() < 1e-9,
            "{name}: obj moved on an unchanged re-solve"
        );
    }
}

// ---------------------------------------------------------------------------
// The *ineligible* parametric path — `qp_prev` and `qp_new` differ in a way
// the homotopy cannot trace, so `solve_parametric` never reaches the path.
//
// That branch used to cold-solve, discarding the working set the caller had
// just handed over. It now re-uses it through `solve_with_working_set` (gh
// #602). The two problems below differ in `H`, which the guard rejects on
// because `H` is not interpolated along the path.
// ---------------------------------------------------------------------------

/// `n = 12`, `m = 8`, diagonal PD `H`, every row one-sided and several binding
/// at the optimum.
///
/// Deliberately larger than the two closed-form cases above: they are small
/// enough that a cold solve reaches the optimum in 0-2 working-set changes, so
/// they cannot tell a re-used working set from a discarded one. The whole point
/// of the assertion below is the change *count*.
fn warm_hint_case() -> Case {
    const N: usize = 12;
    const M: usize = 8;
    let mut hi = Vec::new();
    let mut hj = Vec::new();
    let mut hv = Vec::new();
    for i in 0..N {
        hi.push((i + 1) as i32);
        hj.push((i + 1) as i32);
        hv.push(1.0 + (i as f64) / (N as f64));
    }
    let mut ai = Vec::new();
    let mut aj = Vec::new();
    let mut av = Vec::new();
    for i in 0..M {
        for k in 0..3 {
            ai.push((i + 1) as i32);
            aj.push(((i * 5 + k * 3) % N + 1) as i32);
            av.push(0.5 + 0.25 * ((i + k) % 4) as f64);
        }
    }
    Case {
        n: N,
        m: M,
        h: (hi, hj, hv),
        g: (0..N).map(|j| -2.0 - 0.1 * (j % 5) as f64).collect(),
        a: (ai, aj, av),
        bl: vec![NLP_LOWER_BOUND_INF; M],
        bu: (0..M).map(|i| 1.0 + 0.1 * (i % 3) as f64).collect(),
        xl: vec![0.0; N],
        xu: vec![NLP_UPPER_BOUND_INF; N],
    }
}

/// A parametric call the guard *declines* must still land on the cold answer —
/// and must get there on the previous working set rather than from scratch.
///
/// Both halves matter. The answer is what a caller sees, but it was already
/// correct when this branch cold-solved, so asserting it alone cannot detect a
/// regression back to that. The change count is the property that actually
/// moved, and it is exactly the kind of silent trajectory regression that
/// `CLAUDE.md` records shipping in gh #544 — right answer, more iterations,
/// nothing asserting the difference.
#[test]
fn ineligible_parametric_reuses_the_working_set() {
    let case = warm_hint_case();
    let h_space = SymTMatrixSpace::new(case.n as i32, case.h.0.clone(), case.h.1.clone());
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&case.h.2);
    // A 35% Hessian perturbation: `same_h` is false, so the path is declined.
    let mut h_new = SymTMatrix::new(Rc::clone(&h_space));
    let hv_new: Vec<f64> = case.h.2.iter().map(|v| v * 1.35).collect();
    h_new.set_values(&hv_new);

    let a_space = GenTMatrixSpace::new(
        case.m as i32,
        case.n as i32,
        case.a.0.clone(),
        case.a.1.clone(),
    );
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&case.a.2);

    let opts = QpOptions {
        use_homotopy: true,
        ..QpOptions::default()
    };
    macro_rules! mk {
        ($h:expr) => {
            QpProblem {
                n: case.n,
                m: case.m,
                h: $h,
                g: &case.g,
                a: &a,
                bl: &case.bl,
                bu: &case.bu,
                xl: &case.xl,
                xu: &case.xu,
                hessian_inertia: HessianInertia::Psd,
            }
        };
    }

    let mut s = new_solver();
    let prev = s.solve(&mk!(&h), None, &opts).expect("previous solve");
    assert_eq!(prev.status, QpStatus::Optimal, "previous solve");

    let warm = s
        .solve_parametric(&mk!(&h), &prev, &mk!(&h_new), &opts)
        .expect("ineligible parametric solve");
    let cold = new_solver()
        .solve(&mk!(&h_new), None, &opts)
        .expect("cold solve");

    assert_eq!(warm.status, QpStatus::Optimal, "declined parametric status");
    for i in 0..case.n {
        assert!(
            (warm.x[i] - cold.x[i]).abs() < 1e-7,
            "x[{i}]: declined parametric {} vs cold {}",
            warm.x[i],
            cold.x[i]
        );
    }
    assert!(
        (warm.obj - cold.obj).abs() < 1e-7,
        "obj: declined parametric {} vs cold {}",
        warm.obj,
        cold.obj
    );
    assert!(
        warm.stats.n_working_set_changes < cold.stats.n_working_set_changes,
        "declined parametric took {} working-set changes against cold's {} — the \
         previous working set is being discarded again (gh #602)",
        warm.stats.n_working_set_changes,
        cold.stats.n_working_set_changes
    );
}

// ---------------------------------------------------------------------------
// Working-set statuses that assert something about the *problem*.
//
// `ConsStatus::Equality` and `BoundStatus::Fixed` mean `bl == bu` / `xl == xu`,
// and the solver never drops either. Reusing a previous working set across a
// problem whose bound topology changed therefore states something false that
// nothing downstream can walk back: `pin_working_set` pins an `Equality` row to
// `qp.bl[i]`, the M5 audit sees a feasible point, and the solve reports
// `Optimal` at a point ~1e19 away from the optimum.
//
// Found in review of gh #602, against the working-set reuse added there.
// `WorkingSet::reconciled_with` re-derives both statuses from the new problem.
// ---------------------------------------------------------------------------

/// Build a 1-variable, 1-row QP: `min ½·h·x² s.t. bl ≤ x ≤ bu`, `x` free.
fn one_var_case(h_val: f64, bl: f64, bu: f64) -> Case {
    Case {
        n: 1,
        m: 1,
        h: (vec![1], vec![1], vec![h_val]),
        g: vec![0.0],
        a: (vec![1], vec![1], vec![1.0]),
        bl: vec![bl],
        bu: vec![bu],
        xl: vec![NLP_LOWER_BOUND_INF],
        xu: vec![NLP_UPPER_BOUND_INF],
    }
}

/// Solve `prev` cold, then solve `new` parametrically from it and from cold,
/// and assert the two agree. `H` differs between the two in every caller here,
/// so the guard declines and the working-set fallback is what is under test.
fn assert_declined_parametric_agrees(name: &str, prev: &Case, new: &Case) {
    assert_declined_parametric_agrees_with_opts(name, prev, new, false);
}

/// As above, with the homotopy flag under the caller's control — the traced
/// path and the fallback are different routes to the same hazard.
fn assert_declined_parametric_agrees_with_opts(
    name: &str,
    prev: &Case,
    new: &Case,
    homotopy: bool,
) {
    let mk = |c: &Case| {
        let hs = SymTMatrixSpace::new(c.n as i32, c.h.0.clone(), c.h.1.clone());
        let mut h = SymTMatrix::new(hs);
        h.set_values(&c.h.2);
        let asp = GenTMatrixSpace::new(c.m as i32, c.n as i32, c.a.0.clone(), c.a.1.clone());
        let mut a = GenTMatrix::new(asp);
        a.set_values(&c.a.2);
        (h, a)
    };
    let (h_prev, a_prev) = mk(prev);
    let (h_new, a_new) = mk(new);
    macro_rules! qp {
        ($c:expr, $h:expr, $a:expr) => {
            QpProblem {
                n: $c.n,
                m: $c.m,
                h: &$h,
                g: &$c.g,
                a: &$a,
                bl: &$c.bl,
                bu: &$c.bu,
                xl: &$c.xl,
                xu: &$c.xu,
                hessian_inertia: HessianInertia::Psd,
            }
        };
    }
    let opts = QpOptions {
        use_homotopy: homotopy,
        ..QpOptions::default()
    };
    let mut s = new_solver();
    let sol_prev = s
        .solve(&qp!(prev, h_prev, a_prev), None, &opts)
        .expect("previous solve");
    assert_eq!(sol_prev.status, QpStatus::Optimal, "{name}: previous solve");

    let warm = s
        .solve_parametric(
            &qp!(prev, h_prev, a_prev),
            &sol_prev,
            &qp!(new, h_new, a_new),
            &opts,
        )
        .expect("declined parametric solve");
    let cold = new_solver()
        .solve(&qp!(new, h_new, a_new), None, &opts)
        .expect("cold solve");

    assert_eq!(warm.status, cold.status, "{name}: status");
    for i in 0..new.n {
        assert!(
            (warm.x[i] - cold.x[i]).abs() < 1e-7,
            "{name}: x[{i}] declined parametric {} vs cold {}",
            warm.x[i],
            cold.x[i]
        );
    }
    assert!(
        (warm.obj - cold.obj).abs() < 1e-7,
        "{name}: obj declined parametric {} vs cold {}",
        warm.obj,
        cold.obj
    );
}

/// The reported case: `min ½x² s.t. x == 1` re-solved as `min x² s.t. x ≤ 2`.
/// Reusing `Equality` pinned `x` to the new problem's absent lower bound and
/// returned `Optimal` at `x = -1e19`, `obj = 1e38`, against a true optimum of 0.
#[test]
fn declined_parametric_survives_an_equality_becoming_a_range() {
    assert_declined_parametric_agrees(
        "equality -> range",
        &one_var_case(1.0, 1.0, 1.0),
        &one_var_case(2.0, NLP_LOWER_BOUND_INF, 2.0),
    );
}

/// The other direction: a range row that becomes an equality must be marked
/// `Equality`, or the ratio test skips it (`bl == bu` ⇒ `continue`) and it can
/// never enter the working set.
///
/// Unlike the two `Equality`/`Fixed` cases either side of it, this one passes
/// without `reconciled_with` as well — a stale `AtUpper` pins to `bu`, which is
/// the same point when `bl == bu`, and a stale `Inactive` is caught by the M5
/// audit and recovered through elastic phase-1. It is here to pin the rule, not
/// because it currently discriminates.
#[test]
fn declined_parametric_survives_a_range_becoming_an_equality() {
    assert_declined_parametric_agrees(
        "range -> equality",
        &one_var_case(1.0, NLP_LOWER_BOUND_INF, 2.0),
        &one_var_case(2.0, 1.0, 1.0),
    );
}

/// A row active `AtLower` whose new lower bound is `-inf` names a bound the new
/// problem does not have, and gets pinned to the `-1e20` sentinel exactly as
/// the `Equality` case does.
///
/// It comes back with the right answer anyway, which is the distinction worth
/// keeping in view: `AtLower` is *droppable*, so the dual ratio test sees a
/// multiplier of the wrong sign and removes it. Only the statuses no drop test
/// can reconsider — `Equality`, `Fixed` — turn the same bad pin into a wrong
/// answer. So this test guards a rule that currently costs iterations rather
/// than correctness.
#[test]
fn declined_parametric_survives_a_bound_side_disappearing() {
    assert_declined_parametric_agrees(
        "at-lower -> no lower bound",
        &one_var_case(1.0, 1.0, 3.0),
        &one_var_case(2.0, NLP_LOWER_BOUND_INF, 3.0),
    );
}

/// The same failure through a *variable* bound: `BoundStatus::Fixed` carried
/// onto a problem that has freed the variable pinned it to `xl = -1e20`.
#[test]
fn declined_parametric_survives_a_fixed_variable_being_freed() {
    let prev = Case {
        n: 2,
        m: 1,
        h: (vec![1, 2], vec![1, 2], vec![1.0, 1.0]),
        g: vec![0.0, 0.0],
        a: (vec![1, 1], vec![1, 2], vec![1.0, 1.0]),
        bl: vec![NLP_LOWER_BOUND_INF],
        bu: vec![4.0],
        xl: vec![NLP_LOWER_BOUND_INF, 1.0],
        xu: vec![NLP_UPPER_BOUND_INF, 1.0],
    };
    let new = Case {
        h: (vec![1, 2], vec![1, 2], vec![2.0, 2.0]),
        xl: vec![NLP_LOWER_BOUND_INF, NLP_LOWER_BOUND_INF],
        xu: vec![NLP_UPPER_BOUND_INF, NLP_UPPER_BOUND_INF],
        g: prev.g.clone(),
        a: prev.a.clone(),
        bl: prev.bl.clone(),
        bu: prev.bu.clone(),
        n: prev.n,
        m: prev.m,
    };
    assert_declined_parametric_agrees("fixed -> free", &prev, &new);
}

/// The same topology failure reached through the **traced path** rather than
/// the fallback, which is where it survived the first fix (found by
/// @GermanHeim in review of #614).
///
/// With `H` identical the guard admits the pair, so `reconciled_with` on the
/// fallback never runs: `solve_homotopy`'s warm arm clones `sol_prev.working`
/// as-is and hands it to the corrector at `t = 1`, still claiming `Equality`.
/// The row type does not interpolate — a row that is an equality at `t = 0` is
/// a range at every `t > 0` — so the path cannot re-type it and no drop test
/// can either.
///
/// Measured before the topology guard: `Optimal` at `x = -1e19`, objective
/// `5e37`, against a true optimum of 0.
#[test]
fn traced_path_survives_an_equality_becoming_a_range() {
    for homotopy in [false, true] {
        assert_declined_parametric_agrees_with_opts(
            &format!("equality -> range, identical H, homotopy={homotopy}"),
            &one_var_case(1.0, 1.0, 1.0),
            &one_var_case(1.0, NLP_LOWER_BOUND_INF, 2.0),
            homotopy,
        );
    }
}

/// The `Fixed` half of the same hole: identical `H`, a variable pinned in the
/// previous problem and free in the new one. This one was wrong on *both*
/// `use_homotopy` settings before the guard.
#[test]
fn traced_path_survives_a_fixed_variable_being_freed() {
    let prev = Case {
        n: 2,
        m: 1,
        h: (vec![1, 2], vec![1, 2], vec![1.0, 1.0]),
        g: vec![0.0, 0.0],
        a: (vec![1, 1], vec![1, 2], vec![1.0, 1.0]),
        bl: vec![NLP_LOWER_BOUND_INF],
        bu: vec![4.0],
        xl: vec![NLP_LOWER_BOUND_INF, 1.0],
        xu: vec![NLP_UPPER_BOUND_INF, 1.0],
    };
    let new = Case {
        // `H` identical, so the guard admits on `same_h` and only the topology
        // check can decline.
        h: (vec![1, 2], vec![1, 2], vec![1.0, 1.0]),
        xl: vec![NLP_LOWER_BOUND_INF, NLP_LOWER_BOUND_INF],
        xu: vec![NLP_UPPER_BOUND_INF, NLP_UPPER_BOUND_INF],
        g: prev.g.clone(),
        a: prev.a.clone(),
        bl: prev.bl.clone(),
        bu: prev.bu.clone(),
        n: prev.n,
        m: prev.m,
    };
    for homotopy in [false, true] {
        assert_declined_parametric_agrees_with_opts(
            &format!("fixed -> free, identical H, homotopy={homotopy}"),
            &prev,
            &new,
            homotopy,
        );
    }
}

/// A caller handing `solve_with_working_set` a working set from a problem with
/// different bound topology hits the same `-1e19` pin directly, with no
/// `solve_parametric` involved — the public-API half of the hazard, raised by
/// @GermanHeim in review of #614.
#[test]
fn solve_with_working_set_reconciles_a_stale_hint() {
    let prev = one_var_case(1.0, 1.0, 1.0);
    let new = one_var_case(1.0, NLP_LOWER_BOUND_INF, 2.0);
    let mk = |c: &Case| {
        let hs = SymTMatrixSpace::new(c.n as i32, c.h.0.clone(), c.h.1.clone());
        let mut h = SymTMatrix::new(hs);
        h.set_values(&c.h.2);
        let asp = GenTMatrixSpace::new(c.m as i32, c.n as i32, c.a.0.clone(), c.a.1.clone());
        let mut a = GenTMatrix::new(asp);
        a.set_values(&c.a.2);
        (h, a)
    };
    let (h_prev, a_prev) = mk(&prev);
    let (h_new, a_new) = mk(&new);
    macro_rules! qp {
        ($c:expr, $h:expr, $a:expr) => {
            QpProblem {
                n: $c.n,
                m: $c.m,
                h: &$h,
                g: &$c.g,
                a: &$a,
                bl: &$c.bl,
                bu: &$c.bu,
                xl: &$c.xl,
                xu: &$c.xu,
                hessian_inertia: HessianInertia::Psd,
            }
        };
    }
    let opts = QpOptions::default();
    let sol_prev = new_solver()
        .solve(&qp!(prev, h_prev, a_prev), None, &opts)
        .expect("previous solve");
    assert_eq!(sol_prev.status, QpStatus::Optimal);

    // Passed straight in, exactly as an external caller would.
    let out = new_solver()
        .solve_with_working_set(&qp!(new, h_new, a_new), &sol_prev.working, &opts)
        .expect("working-set solve");
    let cold = new_solver()
        .solve(&qp!(new, h_new, a_new), None, &opts)
        .expect("cold solve");

    assert!(
        (out.x[0] - cold.x[0]).abs() < 1e-7,
        "stale Equality hint: x = {} against cold's {}",
        out.x[0],
        cold.x[0]
    );
    assert!(
        (out.obj - cold.obj).abs() < 1e-7,
        "stale Equality hint: obj = {} against cold's {}",
        out.obj,
        cold.obj
    );
}
