//! What an unbounded solve actually supports, measured rather than assumed.
//!
//! The plan predicted this would be harder than the Farkas case: that the
//! solver returns no feasible witness, so an `unbounded` certificate could not
//! be assembled from a `.sol`. **Both halves of that prediction were wrong**,
//! and the reason is structural rather than incidental.
//!
//! # What the `.sol` carries
//!
//! For `min −x₀ − x₁ s.t. x₀ + x₁ ≥ 1`, POUNCE reports unbounded
//! (`solve_result_num = 300`) and writes its diverging primal iterate,
//! `x ≈ (199.5474, 199.5476)`.
//!
//! That single vector serves **both** roles the certificate needs:
//!
//! * it is exactly feasible (`A x = 399.095… ≥ 1`), so it is the `x₀` witness;
//! * it satisfies the direction conditions exactly (`A d ≥ 0`, `c·d < 0`), so it
//!   is also a `d` witness.
//!
//! # Why no refinement was needed here — and where it became necessary
//!
//! This is the part worth remembering. The Farkas path *requires* exact
//! refinement because its defining condition is an **equality**, `Aᵀy = 0`, and
//! an equality does not survive conversion from floating point: the solver's ray
//! misses it by `−103801/262144`.
//!
//! The recession conditions for an LP are **inequalities** — `A d ≥ 0` and
//! `c·d < 0`. An inequality satisfied with margin *does* survive the conversion,
//! because the exact rational value is still on the correct side of zero. So for
//! `Q = 0` the float witness is already an exact certificate, verbatim.
//!
//! The general QP case reintroduces an equality, `Q d = 0`, and with it a
//! null-space refinement of the same shape as `refine_farkas`. So the split was
//! never LP-vs-QP by accident: it is equality-vs-inequality. That refinement now
//! exists ([`pounce_lean_cert::refine_recession`]) and the emitter covers any
//! `Q`; the last two tests here are the QP case that the LP measurements above
//! could not reach.

#![allow(clippy::unwrap_used)]

use num_rational::BigRational;
use num_traits::Zero;
use pounce_lean_cert::emit::{CertMeta, LinearConstraint, QpInput, emit_unbounded_certificate};
use pounce_lean_cert::refine_recession::RecessionError;
use pounce_lean_cert::{EmitError, Rat};

fn q(v: f64) -> BigRational {
    Rat::from_f64(v).unwrap().inner().clone()
}

fn int(v: i64) -> BigRational {
    BigRational::from_integer(v.into())
}

/// The diverging primal iterate POUNCE writes for `certify_unbounded`.
fn solver_iterate() -> Vec<BigRational> {
    vec![q(199.547370575811897), q(199.547635030014590)]
}

/// `min −x₀ − x₁  s.t.  x₀ + x₁ ≥ 1`. Feasible and unbounded below.
fn system() -> (Vec<Vec<BigRational>>, Vec<BigRational>, Vec<BigRational>) {
    (
        vec![vec![int(1), int(1)]],
        vec![int(1)],
        vec![int(-1), int(-1)],
    )
}

fn row_dot(row: &[BigRational], v: &[BigRational]) -> BigRational {
    row.iter().zip(v).map(|(a, b)| a * b).sum()
}

#[test]
fn the_diverging_iterate_is_exactly_feasible() {
    let (a, b, _c) = system();
    let x = solver_iterate();
    assert!(
        row_dot(&a[0], &x) >= b[0],
        "the iterate must be feasible, or it cannot serve as the x₀ witness"
    );
}

/// The load-bearing contrast with the Farkas case: these conditions are
/// inequalities, and they survive the float→ℚ conversion intact.
#[test]
fn the_float_direction_satisfies_the_recession_conditions_exactly() {
    let (a, _b, c) = system();
    let d = solver_iterate();

    // Q = 0 for an LP, so `Q d = 0` is vacuous — no equality to miss.
    assert!(
        row_dot(&a[0], &d) >= BigRational::zero(),
        "A d ≥ 0 must hold exactly"
    );
    assert!(
        row_dot(&c, &d) < BigRational::zero(),
        "c·d < 0 must hold exactly"
    );
}

/// The contrast, stated as a test so it cannot rot: the Farkas condition is an
/// equality and the float witness misses it; the recession conditions are
/// inequalities and the float witness meets them.
#[test]
fn equalities_need_refinement_and_inequalities_do_not() {
    // Farkas, from the infeasible fixture: Aᵀy = 0 fails exactly.
    let a_inf = [
        vec![int(1), int(1)],
        vec![int(-1), int(0)],
        vec![int(0), int(-1)],
    ];
    let y = [
        q(2.32274114145012817e10),
        q(2.32274114148972511e10),
        q(2.32274114148972511e10),
    ];
    let aty0: BigRational = (0..3).map(|i| &a_inf[i][0] * &y[i]).sum();
    assert!(
        !aty0.is_zero(),
        "the Farkas equality is missed by the float ray — refinement required"
    );

    // Recession: the inequalities hold on the nose.
    let (a, _b, c) = system();
    let d = solver_iterate();
    assert!(row_dot(&a[0], &d) >= BigRational::zero());
    assert!(row_dot(&c, &d) < BigRational::zero());
}

/// A rounded direction also works, and is what a general emitter would prefer
/// for a compact certificate — but note it is a *choice*, not a necessity.
#[test]
fn a_rounded_direction_is_also_exact() {
    let (a, _b, c) = system();
    let d = vec![int(1), int(1)];
    assert_eq!(row_dot(&a[0], &d), int(2));
    assert_eq!(row_dot(&c, &d), int(-2));
}

// --- the QP case: `Q d = 0` is a real obligation ---------------------------

fn meta() -> CertMeta {
    CertMeta {
        nl_sha256: "0".repeat(64),
        sol_sha256: "0".repeat(64),
        solver: "pounce test".to_string(),
    }
}

/// `min x₀² − x₁ s.t. x₀ + x₁ ≥ 1`, with the iterate a solver would diverge to.
///
/// Curved in `x₀`, flat and descending in `x₁`: unbounded below, but not an LP.
fn curved_unbounded_qp(x_float: Vec<f64>) -> QpInput {
    QpInput {
        n: 2,
        q_lower: vec![(0, 0, 2.0)], // ½·2·x₀² = x₀²
        half_quadratic: true,
        c: vec![0.0, -1.0],
        constant: 0.0,
        constraints: vec![LinearConstraint {
            name: "c0".to_string(),
            coeffs: vec![1.0, 1.0],
            lower: 1.0,
            upper: f64::INFINITY,
        }],
        var_lower: vec![f64::NEG_INFINITY; 2],
        var_upper: vec![f64::INFINITY; 2],
        x_float: x_float.clone(),
        active_tol: 1e-7,
    }
}

/// The old emitter refused this outright. It now certifies, and the interesting
/// part is that the two witnesses have **come apart**: `x₀` is the iterate
/// verbatim, while `d` is the exact projection onto `ker Q`, which discards the
/// iterate's `x₀` coordinate entirely.
#[test]
fn a_nonzero_hessian_is_certified_by_projecting_the_direction() {
    // The solver's `x₀` settles near 0 (it is being minimized) while `x₁` runs
    // away. Neither coordinate is a round number, which is the point.
    let iterate = vec![1.0 / 3.0, 4.2e6];
    let cert = emit_unbounded_certificate(&curved_unbounded_qp(iterate.clone()), &meta(), &iterate)
        .expect("a curved-but-flat-along-d QP is certifiable");

    let rec = cert.witnesses.recession.as_ref().unwrap();
    assert_eq!(
        rec.x0[0].inner(),
        Rat::from_f64(1.0 / 3.0).unwrap().inner(),
        "x₀ is the iterate verbatim — feasibility is a claim about that point"
    );
    assert_eq!(
        rec.d
            .iter()
            .map(|r| r.inner().to_string())
            .collect::<Vec<_>>(),
        vec!["0", "1"],
        "d is the exact projection onto ker Q, normalized"
    );
    // The obligation the LP slice never had to meet: Q d = 0, on the nose.
    let qd0 = int(2) * rec.d[0].inner();
    assert!(
        qd0.is_zero(),
        "Q d = 0 must hold exactly, not approximately"
    );
}

/// Copying the float direction would **not** have worked, which is why the
/// projection is not a stylistic choice. The iterate's `x₀` coordinate is
/// `1/3`'s nearest double — a nonzero dyadic — so `Q d` is nonzero and the Lean
/// hypothesis `Q d = 0` is undischargeable.
#[test]
fn the_unprojected_float_direction_would_have_failed() {
    let qd0 = int(2) * q(1.0 / 3.0);
    assert!(
        !qd0.is_zero(),
        "the raw float direction misses Q d = 0 — exactly the Farkas situation"
    );
}

/// A positive definite `Q` curves in every direction, so no ray can leave the
/// quadratic term unchanged. The refusal says that, rather than reporting some
/// downstream inequality that happened to fail.
#[test]
fn a_strictly_convex_qp_is_refused_with_the_reason() {
    let mut input = curved_unbounded_qp(vec![1.0, 1.0e6]);
    input.q_lower = vec![(0, 0, 2.0), (1, 1, 2.0)];
    let err = emit_unbounded_certificate(&input, &meta(), &[1.0, 1.0e6])
        .expect_err("a strictly convex QP has no recession direction");
    assert_eq!(err, EmitError::Recession(RecessionError::NoFlatDirection));
}
