//! Barrier-parameter update strategies — port of
//! `Algorithm/IpMuUpdate.hpp`, `IpMonotoneMuUpdate.{hpp,cpp}`,
//! `IpAdaptiveMuUpdate.{hpp,cpp}`, and the four oracle files
//! (`IpMuOracle.hpp`, `IpLoqoMuOracle.cpp`, `IpProbingMuOracle.cpp`,
//! `IpQualityFunctionMuOracle.cpp`).
//!
//! Phase 7 ships [`monotone::MonotoneMuUpdate`] (Fiacco-McCormick).
//! Phase 10 adds the adaptive path and all four oracles.

pub mod adaptive;
pub mod monotone;
pub mod oracle;
pub mod r#trait;

pub use r#trait::MuUpdate;

use pounce_common::types::Number;

/// `compl_inf_tol` expressed in the **internally scaled** space that μ lives
/// in (pounce#257). Shared by both μ strategies; see
/// [`monotone::MonotoneMuUpdate::scaled_compl_inf_tol`] for the full story.
///
/// `compl_inf_tol` is enforced on the **unscaled** complementarity, which is
/// the scaled complementarity divided by the objective scaling factor — so in
/// scaled units the tolerance is `compl_inf_tol · |df|`. The factor is signed
/// (`obj_scaling_factor = -1` poses a maximization), so take its magnitude,
/// and fall back to the unconverted tolerance when it is absent or degenerate.
pub(crate) fn scaled_compl_inf_tol(compl_inf_tol: Number, obj_scaling_factor: Number) -> Number {
    let df = obj_scaling_factor.abs();
    if df.is_finite() && df > 0.0 {
        compl_inf_tol * df
    } else {
        compl_inf_tol
    }
}

/// `mu_min` capped so it can never block the termination certificate
/// (pounce#266). Shared by both μ strategies; see
/// [`monotone::MonotoneMuUpdate::certificate_safe_mu_min`] for the full story.
///
/// `mu_min` is an absolute constant in μ's scaled space. Left raw, once
/// `compl_inf_tol·|df|/(barrier_tol_factor+1) < mu_min` the unscaled
/// complementarity is pinned at `mu_min/|df| > compl_inf_tol` and the strict
/// certificate becomes unreachable. The cap keeps `mu_min` inert exactly when
/// it would cost the certificate, with the same headroom the monotone dynamic
/// floor reserves. A floor that is too low only costs iterations; one that is
/// too high costs the certificate.
pub(crate) fn certificate_safe_mu_min(
    mu_min: Number,
    compl_inf_tol: Number,
    barrier_tol_factor: Number,
    obj_scaling_factor: Number,
) -> Number {
    mu_min.min(scaled_compl_inf_tol(compl_inf_tol, obj_scaling_factor) / (barrier_tol_factor + 1.0))
}
