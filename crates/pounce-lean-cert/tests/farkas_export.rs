//! Why exporting the solver's Farkas certificate is *not* plumbing.
//!
//! The plan recorded this as "export the existing certificate from `QpStatus`
//! through the emitter". Two premises in that sentence turned out to be false,
//! and both are pinned here so the correction cannot quietly regress.
//!
//! # `QpStatus` carries nothing
//!
//! `QpStatus::PrimalInfeasible` is a **unit variant**. Its doc comment says a
//! Farkas certificate "was detected and verified", which is true — the check
//! happens inside `detect_infeasibility` in `pounce-convex` — but the status
//! returns no payload. What actually reaches a consumer is the diverging dual
//! iterate, written to the `.sol` as ordinary constraint duals.
//!
//! # The float ray is not an exact certificate
//!
//! This is the substantive one. Farkas requires `y ≥ 0`, `Aᵀy = 0`, and
//! `b·y > 0`. The solver verifies the middle condition to a *relative*
//! tolerance, because the ray diverges: for the instance below `‖y‖ ≈ 2.3e10`
//! and the residual is ~1.7e-11 relative — comfortably "zero" in floating
//! point.
//!
//! Converted losslessly to ℚ, that residual is `−103801/262144`. Not small.
//! **Nonzero.** So `Aᵀy = 0` fails exactly, the Lean hypothesis is
//! undischargeable, and a certificate built by copying the solver's duals
//! could never verify.
//!
//! The fix is the same one the primal path already uses: treat the float ray as
//! a *hint* that identifies the certificate's support, then solve for an exact
//! rational ray. Here that collapses to `y = (1, 1, 1)`. So exporting
//! infeasibility needs a `refine_farkas` analogous to `refine_kkt` — new code,
//! not a field copy.

#![allow(clippy::unwrap_used)]

use num_rational::BigRational;
use num_traits::Zero;
use pounce_lean_cert::Rat;

/// `x₀ + x₁ ≥ 2`, `−x₀ ≥ 0`, `−x₁ ≥ 0` — infeasible, since the last two force
/// `x₀ + x₁ ≤ 0`.
fn system() -> (Vec<Vec<BigRational>>, Vec<BigRational>) {
    let r = |n: i64| BigRational::from_integer(n.into());
    (
        vec![vec![r(1), r(1)], vec![r(-1), r(0)], vec![r(0), r(-1)]],
        vec![r(2), r(0), r(0)],
    )
}

/// The duals POUNCE's LP IPM actually writes for this instance, negated from
/// the AMPL sign convention into the `λ ≥ 0` convention used by the schema.
fn solver_ray() -> Vec<BigRational> {
    [
        2.32274114145012817e10,
        2.32274114148972511e10,
        2.32274114148972511e10,
    ]
    .iter()
    .map(|v| Rat::from_f64(*v).unwrap().inner().clone())
    .collect()
}

fn a_transpose_y(a: &[Vec<BigRational>], y: &[BigRational], n: usize) -> Vec<BigRational> {
    (0..n)
        .map(|j| (0..a.len()).map(|i| &a[i][j] * &y[i]).sum())
        .collect()
}

fn dot(u: &[BigRational], v: &[BigRational]) -> BigRational {
    u.iter().zip(v).map(|(a, b)| a * b).sum()
}

#[test]
fn solver_ray_satisfies_farkas_only_approximately() {
    let (a, b) = system();
    let y = solver_ray();

    // Two of the three conditions hold exactly.
    assert!(y.iter().all(|v| *v >= BigRational::zero()), "y ≥ 0 holds");
    assert!(dot(&b, &y) > BigRational::zero(), "b·y > 0 holds");

    // The third does not.
    let aty = a_transpose_y(&a, &y, 2);
    assert!(
        !aty.iter().all(|v| v.is_zero()),
        "the float ray must NOT satisfy Aᵀy = 0 exactly — if it does, this \
         instance no longer demonstrates the problem and the test is pointless"
    );
    assert_eq!(
        aty[0].to_string(),
        "-103801/262144",
        "the exact residual, for the record"
    );

    // …yet it is tiny relative to the ray's magnitude, which is exactly why the
    // solver accepts it and why the gap is easy to miss.
    let norm = y.iter().max().unwrap().clone();
    let rel = &aty[0] / &norm;
    assert!(
        rel.to_string().len() > 10,
        "relative residual is ~1.7e-11, well inside any float tolerance"
    );
}

#[test]
fn the_refined_ray_is_an_exact_certificate() {
    // Aᵀy = 0 forces y₁ = y₂ = y₀; normalizing y₀ = 1 gives the exact ray.
    let (a, b) = system();
    let one = BigRational::from_integer(1.into());
    let y = vec![one.clone(), one.clone(), one];

    assert!(y.iter().all(|v| *v >= BigRational::zero()), "y ≥ 0");
    assert!(
        a_transpose_y(&a, &y, 2).iter().all(|v| v.is_zero()),
        "Aᵀy = 0 exactly"
    );
    assert!(dot(&b, &y) > BigRational::zero(), "b·y > 0");
}

// --------------------------------------------------------------------------
// The emit path, now that refinement exists.
// --------------------------------------------------------------------------

use pounce_lean_cert::emit::{CertMeta, LinearConstraint, QpInput, emit_infeasible_certificate};

fn meta() -> CertMeta {
    CertMeta {
        nl_sha256: "0".repeat(64),
        sol_sha256: "0".repeat(64),
        solver: "pounce 0.9.0".to_string(),
    }
}

fn con(name: &str, coeffs: Vec<f64>, lower: f64) -> LinearConstraint {
    LinearConstraint {
        name: name.to_string(),
        coeffs,
        lower,
        upper: f64::INFINITY,
    }
}

/// The same infeasible system, as POUNCE would hand it over.
fn infeasible_input() -> QpInput {
    QpInput {
        n: 2,
        q_lower: vec![],
        half_quadratic: true,
        c: vec![1.0, 1.0],
        constant: 0.0,
        constraints: vec![
            con("c0", vec![1.0, 1.0], 2.0),
            con("c1", vec![-1.0, 0.0], 0.0),
            con("c2", vec![0.0, -1.0], 0.0),
        ],
        var_lower: vec![f64::NEG_INFINITY; 2],
        var_upper: vec![f64::INFINITY; 2],
        x_float: vec![0.0, 0.0],
        active_tol: 1e-7,
    }
}

#[test]
fn emits_an_exact_infeasible_certificate_from_the_solver_ray() {
    let y_float = [
        2.32274114145012817e10,
        2.32274114148972511e10,
        2.32274114148972511e10,
    ];
    let cert = emit_infeasible_certificate(&infeasible_input(), &meta(), &y_float, 1e-9)
        .expect("the system is infeasible and the ray refines");

    assert_eq!(cert.verdict, "infeasible");
    // Not a claim about a point.
    assert!(
        cert.candidate.is_none(),
        "infeasible certs carry no candidate"
    );
    assert!(cert.witnesses.duals.is_none());
    assert!(cert.witnesses.hessian_psd.is_none());

    let y = &cert.witnesses.farkas.as_ref().expect("farkas witness").y;
    let got: Vec<String> = y.iter().map(|r| r.inner().to_string()).collect();
    assert_eq!(
        got,
        vec!["1", "1", "1"],
        "the exact ray, not the 2.3e10 float one"
    );
    assert_eq!(cert.tolerance.inner().to_string(), "0");
}

/// A feasible system must not yield an infeasibility certificate, whatever
/// ray is handed in.
#[test]
fn emit_refuses_a_feasible_system() {
    let mut input = infeasible_input();
    // Drop the two rows that make it contradictory.
    input.constraints.truncate(1);
    assert!(
        emit_infeasible_certificate(&input, &meta(), &[1.0], 1e-9).is_err(),
        "x0 + x1 >= 2 alone is feasible; no certificate may be emitted"
    );
}
