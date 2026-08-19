//! gh #697: `Presolve::obj_offset` must aggregate the whole reduction chain.
//!
//! The fixpoint wrapper `presolve` returns holds the composed reduction; the
//! per-layer offsets live in its chain, and only `postsolve` walked that chain.
//! The aggregate accessor was hard-coded to `0.0`, so every multi-layer
//! reduction — which this repo's own notes call the common case — reported
//! "presolve moved no constant into the objective" however large a constant it
//! had actually moved.
//!
//! The reader that made this matter is the CLI's `obj_constant` (gh #689),
//! which normalizes the relative stopping test by the objective's own
//! magnitude. A dropped offset there cannot produce a wrong answer — it makes
//! the gap test too loose, on exactly the models presolve reduces hardest,
//! which is the trajectory/accuracy class that goes unnoticed.

use pounce_convex::presolve::{PresolveOutcome, presolve};
use pounce_convex::{QpOptions, QpProblem, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `½·xᵀPx + cᵀx`, the objective `QpProblem` carries.
fn objective(prob: &QpProblem, x: &[f64]) -> f64 {
    let mut px = vec![0.0; prob.n];
    prob.p_mul(x, &mut px);
    0.5 * px.iter().zip(x).map(|(a, b)| a * b).sum::<f64>()
        + prob.c.iter().zip(x).map(|(a, b)| a * b).sum::<f64>()
}

/// A reduction that takes two layers and moves a constant in each.
///
/// ```text
/// min  10·x₀ + 100·x₁ + ½·(x₁² + x₂²)
/// s.t.     x₀             = 2
///                x₁ + x₂  = 4
/// ```
///
/// Round 1's catalog pass fixes the singleton `x₀ = 2`, moving `10·2` into the
/// objective. The doubleton `x₁ + x₂ = 4` is not a catalog reduction — both of
/// its columns are quadratic — so it falls to the aggregation, which always
/// forms a layer of its own and carries the constant its substitution leaves
/// behind. Two layers, both with a nonzero offset.
fn two_layer_reduction() -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: vec![Triplet::new(1, 1, 1.0), Triplet::new(2, 2, 1.0)],
        c: vec![10.0, 100.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(1, 2, 1.0),
        ],
        b: vec![2.0, 4.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    }
}

#[test]
fn multilayer_presolve_reports_its_composed_objective_offset() {
    let prob = two_layer_reduction();
    let PresolveOutcome::Reduced(ps) = presolve(&prob) else {
        panic!("this reduction is feasible and bounded");
    };
    let stats = ps.stats();
    assert!(
        stats.rounds >= 2,
        "fixture must exercise the multi-layer path, took {} layer(s)",
        stats.rounds
    );
    assert_eq!(stats.fixed_vars, 1, "the singleton should have fired");
    assert_eq!(stats.aggregated_vars, 1, "the doubleton should have fired");

    // The documented contract: reduced objective + offset == original
    // objective, at the point the two describe. Asserted against a solve
    // rather than a literal because which of a doubleton's two columns the
    // aggregation folds decides how much of the linear term lands in the
    // constant — the *contract* is invariant to that choice, the number is
    // not. (It is 428 as this is written.)
    let red = solve_qp_ipm(&ps.reduced, &QpOptions::default(), backend);
    let full = ps.postsolve(&red);
    let want = objective(&prob, &full.x) - objective(&ps.reduced, &red.x);
    assert!(
        want.abs() > 1.0,
        "fixture must move a constant worth measuring, moved {want}"
    );
    assert!(
        (ps.obj_offset - want).abs() < 1e-6 * want.abs().max(1.0),
        "composed offset over {} layers: got {}, want {want}",
        stats.rounds,
        ps.obj_offset
    );
}

/// The single-layer path already reported its offset; keep it reporting the
/// same value, so the fix reads as "the chain case joins it" rather than as a
/// change of meaning.
#[test]
fn single_layer_presolve_still_reports_its_own_offset() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(1, 1, 1.0)],
        c: vec![10.0, 0.0],
        a: vec![Triplet::new(0, 0, 1.0)],
        b: vec![2.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let PresolveOutcome::Reduced(ps) = presolve(&prob) else {
        panic!("this reduction is feasible and bounded");
    };
    assert_eq!(ps.stats().rounds, 1, "fixture must stay a single layer");
    assert!(
        (ps.obj_offset - 20.0).abs() < 1e-12,
        "fixing x₀ = 2 moves 10·2 into the objective, got {}",
        ps.obj_offset
    );
}
