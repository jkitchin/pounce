//! Exact rational Farkas ray, refined from the solver's floating-point ray.
//!
//! The primal path never certifies the solver's `x*`; it takes the float active
//! set as a *hint* and re-solves the KKT system exactly ([`crate::refine`]).
//! Infeasibility needs the same treatment, for the same reason.
//!
//! When an interior-point method concludes primal infeasibility, its dual
//! iterate diverges along a Farkas ray. The solver verifies `Aᵀy = 0` only
//! *relative* to that ray's magnitude — which is sound for a float test but not
//! for an exact one. Measured on the `certify_infeasible` fixture: `‖y‖ ≈
//! 2.3e10`, relative residual `1.7e-11`, and the exact residual is
//! `−103801/262144`. Small, nonzero, and fatal to a Lean proof requiring
//! `Aᵀy = 0`.
//!
//! So the float ray is used only to identify **which constraints the
//! certificate rests on**. The ray itself is then recomputed exactly as a null
//! vector of `Aᵀ` restricted to that support.

use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use crate::linalg::nullspace_exact;

/// Why an exact Farkas ray could not be produced. Every variant means "refuse",
/// never "emit something weaker".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FarkasError {
    /// The float ray was all zeros, so there is no support to work from.
    EmptySupport,
    /// `Aᵀ` restricted to the support has trivial null space: no ray exists
    /// there. Usually means the support hint was too small.
    NoRay,
    /// The null space has dimension > 1, so the support does not determine a
    /// ray. Choosing among them needs a rule this does not implement.
    AmbiguousRay { dim: usize },
    /// The ray has mixed signs and so cannot be scaled to `y ≥ 0`.
    SignMixed,
    /// `b·y = 0`, so the ray certifies nothing.
    ZeroCertificate,
    /// Defensive: the assembled ray failed its own exact recheck.
    SelfCheck(&'static str),
}

impl std::fmt::Display for FarkasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for FarkasError {}

/// Refine a floating-point Farkas ray into an exact rational one.
///
/// * `a_rows` — the inequality system `A x ≥ b`, one row per constraint.
/// * `b` — right-hand side, same length as `a_rows`.
/// * `y_float` — the solver's dual ray (already in the `λ ≥ 0` convention).
/// * `support_tol` — relative threshold: constraint `i` is in the support when
///   `|y_float[i]| > support_tol · max|y_float|`.
///
/// On success the returned `y` satisfies, **exactly over ℚ**: `y ≥ 0`,
/// `Aᵀy = 0`, and `b·y > 0`, normalized so its largest entry is 1.
pub fn refine_farkas(
    a_rows: &[Vec<BigRational>],
    b: &[BigRational],
    y_float: &[f64],
    support_tol: f64,
) -> Result<Vec<BigRational>, FarkasError> {
    let m = a_rows.len();
    if m == 0 || b.len() != m || y_float.len() != m {
        return Err(FarkasError::SelfCheck("shape mismatch"));
    }
    let n = a_rows[0].len();
    if a_rows.iter().any(|r| r.len() != n) {
        return Err(FarkasError::SelfCheck("ragged A"));
    }

    // --- support from the float hint ---------------------------------------
    let scale = y_float.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if scale == 0.0 {
        return Err(FarkasError::EmptySupport);
    }
    let support: Vec<usize> = (0..m)
        .filter(|&i| y_float[i].abs() > support_tol * scale)
        .collect();
    if support.is_empty() {
        return Err(FarkasError::EmptySupport);
    }

    // --- exact null space of Aᵀ on the support ------------------------------
    // Aᵀ_S is n × |S|: row j, column k is a_rows[S[k]][j].
    let at_s: Vec<Vec<BigRational>> = (0..n)
        .map(|j| support.iter().map(|&i| a_rows[i][j].clone()).collect())
        .collect();
    let basis = nullspace_exact(&at_s, support.len());
    match basis.len() {
        0 => return Err(FarkasError::NoRay),
        1 => {}
        d => return Err(FarkasError::AmbiguousRay { dim: d }),
    }

    // --- expand to the full constraint set ---------------------------------
    let mut y = vec![BigRational::zero(); m];
    for (k, &i) in support.iter().enumerate() {
        y[i] = basis[0][k].clone();
    }

    // --- orient so that b·y > 0 --------------------------------------------
    let by: BigRational = b.iter().zip(&y).map(|(bi, yi)| bi * yi).sum();
    if by.is_zero() {
        return Err(FarkasError::ZeroCertificate);
    }
    if by.is_negative() {
        for v in &mut y {
            *v = -v.clone();
        }
    }

    // --- a Farkas ray must be nonnegative ----------------------------------
    if y.iter().any(|v| v.is_negative()) {
        return Err(FarkasError::SignMixed);
    }

    // --- normalize so the largest entry is exactly 1 ------------------------
    let peak = y
        .iter()
        .max()
        .cloned()
        .ok_or(FarkasError::SelfCheck("empty ray"))?;
    if peak.is_zero() {
        return Err(FarkasError::SelfCheck("ray collapsed to zero"));
    }
    for v in &mut y {
        *v = &*v / &peak;
    }

    // --- exact self-check: refuse rather than emit something that won't verify
    for (j, _) in (0..n).enumerate() {
        let col: BigRational = (0..m).map(|i| &a_rows[i][j] * &y[i]).sum();
        if !col.is_zero() {
            return Err(FarkasError::SelfCheck("Aᵀy ≠ 0"));
        }
    }
    if y.iter().any(|v| v.is_negative()) {
        return Err(FarkasError::SelfCheck("y ≥ 0"));
    }
    let by_final: BigRational = b.iter().zip(&y).map(|(bi, yi)| bi * yi).sum();
    if !by_final.is_positive() {
        return Err(FarkasError::SelfCheck("b·y > 0"));
    }
    debug_assert!(y.iter().any(BigRational::is_one));

    Ok(y)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn r(v: i64) -> BigRational {
        BigRational::from_integer(v.into())
    }

    /// `x₀ + x₁ ≥ 2`, `−x₀ ≥ 0`, `−x₁ ≥ 0`.
    fn infeasible_system() -> (Vec<Vec<BigRational>>, Vec<BigRational>) {
        (
            vec![vec![r(1), r(1)], vec![r(-1), r(0)], vec![r(0), r(-1)]],
            vec![r(2), r(0), r(0)],
        )
    }

    #[test]
    fn refines_the_solver_ray_to_an_exact_certificate() {
        let (a, b) = infeasible_system();
        // The duals POUNCE's LP IPM actually returns, sign-corrected. Exactly
        // over ℚ these give Aᵀy = −103801/262144, i.e. NOT a certificate.
        let y_float = [
            2.32274114145012817e10,
            2.32274114148972511e10,
            2.32274114148972511e10,
        ];
        let y = refine_farkas(&a, &b, &y_float, 1e-9).unwrap();
        assert_eq!(y, vec![r(1), r(1), r(1)]);
    }

    #[test]
    fn a_wildly_wrong_ray_still_refines_if_the_support_is_right() {
        // The magnitudes are nonsense; only the support matters.
        let (a, b) = infeasible_system();
        let y = refine_farkas(&a, &b, &[1.0, 500.0, 0.003], 1e-9).unwrap();
        assert_eq!(y, vec![r(1), r(1), r(1)]);
    }

    #[test]
    fn too_small_a_support_has_no_ray() {
        let (a, b) = infeasible_system();
        // Only constraint 0 in support: Aᵀ restricted to it has trivial kernel.
        let err = refine_farkas(&a, &b, &[1.0, 0.0, 0.0], 1e-9).unwrap_err();
        assert_eq!(err, FarkasError::NoRay);
    }

    #[test]
    fn all_zero_ray_is_refused() {
        let (a, b) = infeasible_system();
        let err = refine_farkas(&a, &b, &[0.0, 0.0, 0.0], 1e-9).unwrap_err();
        assert_eq!(err, FarkasError::EmptySupport);
    }

    /// A *feasible* system must not yield a certificate. `x₀ ≥ 0`, `x₁ ≥ 0` is
    /// satisfiable, and its only null direction gives `b·y = 0`.
    #[test]
    fn feasible_system_yields_no_certificate() {
        let a = vec![vec![r(1), r(0)], vec![r(0), r(1)]];
        let b = vec![r(0), r(0)];
        let err = refine_farkas(&a, &b, &[1.0, 1.0], 1e-9).unwrap_err();
        assert!(
            matches!(err, FarkasError::NoRay | FarkasError::ZeroCertificate),
            "a feasible system must not produce a Farkas certificate, got {err:?}"
        );
    }

    /// The refined ray is exact where the float one is not — the property that
    /// motivates this module.
    #[test]
    fn the_float_ray_itself_is_not_exact() {
        let (a, _b) = infeasible_system();
        let y: Vec<BigRational> = [
            2.32274114145012817e10,
            2.32274114148972511e10,
            2.32274114148972511e10,
        ]
        .iter()
        .map(|v| crate::Rat::from_f64(*v).unwrap().inner().clone())
        .collect();
        let col0: BigRational = (0..3).map(|i| &a[i][0] * &y[i]).sum();
        assert!(
            !col0.is_zero(),
            "if the raw float ray were already exact, refinement would be unnecessary"
        );
        assert_eq!(col0.to_string(), "-103801/262144");
    }
}
