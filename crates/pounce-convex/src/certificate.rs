//! Infeasibility and unboundedness certificates in the caller's own problem
//! (gh#1181).
//!
//! When a convex engine says `PrimalInfeasible` or `DualInfeasible`, it has
//! found a ray: the solution's `(y, z)` (a Farkas ray) or its `x` (a recession
//! direction). This module turns that ray into a certificate **for the
//! problem the caller passed** and checks it independently of the engine:
//! nothing here reads a solver tolerance, a scaling, or the presolved model.
//!
//! # Why a separate check
//!
//! The engine certifies its ray on the problem *it* solved — equilibrated,
//! and after presolve, reduced. Two things then stand between that ray and a
//! certificate the caller can use:
//!
//! * **Presolve.** Its postsolve lifts *optimal* duals: each removed row's
//!   multiplier is recovered from stationarity at the postsolved point, which
//!   involves `c` and `P`. A ray needs those recoveries done with `c = 0`, so
//!   the lifted `(y, z)` of an infeasible solve is not, in general, a ray of
//!   the original problem. And a presolve *verdict* (an empty row, an activity
//!   bound that cannot be met) comes back with no ray at all.
//! * **The variable box.** The engine's box multipliers come back in
//!   `z_lb`/`z_ub` under the engine's own conventions. They are not read
//!   here: given `(y, z)`, the best box multipliers are determined column by
//!   column (below), so they are re-derived.
//!
//! [`certificate`] therefore *checks* whatever ray it is given against the
//! original data and reports whether it passed, and
//! [`certificate_or_resolve`] falls back on an unreduced re-solve when the
//! lifted ray does not pass. A certificate that fails the check is still
//! returned, marked `verified = false`, because that is the signal gh#414's
//! false-*optimal* verdict lacked: a false *infeasible* verdict surfaces as
//! a certificate that does not check.
//!
//! # The primal (Farkas) certificate
//!
//! For `Ax = b`, `Gx + s = h` with `s ∈ K`, `lb ≤ x ≤ ub`, and any
//! `y`, `z ∈ K*`, every feasible `x` satisfies
//!
//! ```text
//! qᵀx ≤ bᵀy + hᵀz,      q = Aᵀy + Gᵀz.
//! ```
//!
//! Write `q = z_lb − z_ub + r` with `z_lb, z_ub ≥ 0`, `z_lb` only on columns
//! with a finite lower bound and `z_ub` only on those with a finite upper one.
//! If `r = 0`, then `qᵀx ≥ lbᵀz_lb − ubᵀz_ub` over the box, so
//!
//! ```text
//! value = bᵀy + hᵀz − lbᵀz_lb + ubᵀz_ub < 0
//! ```
//!
//! means no feasible `x` exists. Given `(y, z)` the choice that minimizes `r`
//! is `z_lb = q⁺` where `lb` is finite, `z_ub = q⁻` where `ub` is finite, so
//! that is what is reported: `r` is the part of `q` on a column whose bound in
//! the needed direction is infinite. On an orthant `z` is clipped to `z⁺`
//! first, so `z ≥ 0` holds exactly rather than to a tolerance.
//!
//! # The dual (recession) certificate
//!
//! A direction `d` with `Pd = 0`, `Ad = 0`, `−Gd ∈ K`, `d_j ≥ 0` where `lb_j`
//! is finite, `d_j ≤ 0` where `ub_j` is finite, and `cᵀd < 0` takes any
//! feasible point to `−∞` objective. Feasibility of *some* point is not part
//! of the certificate (a dual-infeasible problem may also be primal
//! infeasible), exactly as for Clarabel's and SCS's dual-infeasibility
//! certificates.
//!
//! # When a certificate counts as verified
//!
//! A floating-point ray is never exact, so "verified" is a stated margin, the
//! one the engine's own detector applies ([`FARKAS_MARGIN_MIN`], see
//! `ipm::FarkasMargin`): the value must be negative beyond the rounding of
//! the dot product that formed it, and the relative residual must be smaller
//! than the relative value by a factor of `FARKAS_MARGIN_MIN`. For a Farkas
//! ray that factor is the radius, in units of the data's own scale, inside
//! which the ray rules out every feasible point. Every quantity is reported,
//! so a caller who wants a different standard applies it to the numbers.

use crate::cones::ConeSpec;
use crate::ipm::FARKAS_MARGIN_MIN;
use crate::qp::{BOUND_INF, QpProblem, QpSolution, QpStatus};

/// Which engine output a certificate came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CertificateSource {
    /// The ray of the solve that produced the verdict, lifted through
    /// presolve when presolve ran.
    Solve,
    /// The ray of a re-solve of the original, unreduced problem, taken
    /// because the lifted ray did not pass the check (or presolve proved the
    /// verdict and had no ray to lift). The verdict is still the first
    /// solve's.
    UnreducedResolve,
}

impl CertificateSource {
    /// `"solve"` / `"unreduced_resolve"`, the spelling the reports use.
    pub fn as_str(self) -> &'static str {
        match self {
            CertificateSource::Solve => "solve",
            CertificateSource::UnreducedResolve => "unreduced_resolve",
        }
    }
}

/// A primal-infeasibility (Farkas) certificate, normalized to
/// `‖(y, z, z_lb, z_ub)‖∞ = 1`. See the module docs for the algebra.
#[derive(Debug, Clone, PartialEq)]
pub struct FarkasCertificate {
    /// Equality-row multipliers (length `m_eq`), free in sign.
    pub y: Vec<f64>,
    /// Inequality-row multipliers (length `m_ineq`), in the dual cone `K*`
    /// (`≥ 0` on an orthant, where it is exact).
    pub z: Vec<f64>,
    /// Lower-bound multipliers (length `n`), `≥ 0`, zero where `lb` is
    /// infinite.
    pub z_lb: Vec<f64>,
    /// Upper-bound multipliers (length `n`), `≥ 0`, zero where `ub` is
    /// infinite.
    pub z_ub: Vec<f64>,
    /// `bᵀy + hᵀz − lbᵀz_lb + ubᵀz_ub`. Negative for a certificate.
    pub value: f64,
    /// `‖Aᵀy + Gᵀz − z_lb + z_ub‖∞`: the part of `Aᵀy + Gᵀz` no finite
    /// bound absorbs. Zero for an exact certificate.
    pub residual: f64,
    /// How far `z` lies outside `K*`, in the cone's own defining inequality.
    /// Zero on an orthant, where `z` is clipped to `z⁺`.
    pub cone_violation: f64,
    /// `(|value| / Σ|value terms|) / (residual / ‖|A|ᵀ|y| + |G|ᵀ|z|‖∞)`:
    /// the radius, in units of the data's scale, within which the ray rules
    /// out every feasible point. `+∞` when `residual = 0`.
    pub margin: f64,
    /// The checks in the module docs passed.
    pub verified: bool,
    pub source: CertificateSource,
}

/// A dual-infeasibility (recession) certificate, normalized to `‖d‖∞ = 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct RecessionCertificate {
    /// The direction (length `n`).
    pub d: Vec<f64>,
    /// `cᵀd`. Negative for a certificate.
    pub value: f64,
    /// `‖Pd‖∞`.
    pub hessian_residual: f64,
    /// `‖Ad‖∞`.
    pub equality_residual: f64,
    /// How far `−Gd` lies outside `K` (on an orthant, `max(Gd)⁺`).
    pub cone_violation: f64,
    /// `max(max_{lb_j finite} (−d_j)⁺, max_{ub_j finite} d_j⁺)`.
    pub bound_violation: f64,
    /// `(|cᵀd| / |c|ᵀ|d|) / ρ`, where `ρ` is the largest of the residuals
    /// above, each relative to the magnitude it cancels from. `+∞` when every
    /// residual is exactly zero.
    pub margin: f64,
    /// The checks in the module docs passed.
    pub verified: bool,
    pub source: CertificateSource,
}

/// The certificate behind a `PrimalInfeasible` or `DualInfeasible` verdict.
#[derive(Debug, Clone, PartialEq)]
pub enum Certificate {
    PrimalInfeasible(FarkasCertificate),
    DualInfeasible(RecessionCertificate),
}

impl Certificate {
    /// `"primal_infeasible"` / `"dual_infeasible"`.
    pub fn kind(&self) -> &'static str {
        match self {
            Certificate::PrimalInfeasible(_) => "primal_infeasible",
            Certificate::DualInfeasible(_) => "dual_infeasible",
        }
    }

    pub fn verified(&self) -> bool {
        match self {
            Certificate::PrimalInfeasible(c) => c.verified,
            Certificate::DualInfeasible(c) => c.verified,
        }
    }

    pub fn source(&self) -> CertificateSource {
        match self {
            Certificate::PrimalInfeasible(c) => c.source,
            Certificate::DualInfeasible(c) => c.source,
        }
    }

    fn with_source(mut self, s: CertificateSource) -> Self {
        match &mut self {
            Certificate::PrimalInfeasible(c) => c.source = s,
            Certificate::DualInfeasible(c) => c.source = s,
        }
        self
    }
}

/// `(L + 2)·ε / (1 − (L + 2)·ε)`: a bound on the relative rounding error of a
/// length-`L` dot product (Higham's `γ_L`, padded as `ipm::farkas_box_proof`
/// pads it).
fn gamma(len: usize) -> f64 {
    let le = (len as f64 + 2.0) * f64::EPSILON;
    le / (1.0 - le)
}

/// Whether column `j` has a lower bound, by the engine's convention: anything
/// at or past `−BOUND_INF` (`−1e20`) is no bound, as `NEG_INF`'s docs say. A
/// plain `is_finite()` would read a caller's `−1e20` as a real bound, put the
/// ray's rounding leftover on it, and report a false "not verified"
/// (gh#1181 review).
fn has_lb(prob: &QpProblem, j: usize) -> bool {
    prob.lb_of(j) > -BOUND_INF
}

/// [`has_lb`] for the upper bound.
fn has_ub(prob: &QpProblem, j: usize) -> bool {
    prob.ub_of(j) < BOUND_INF
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// The certificate behind `sol`'s verdict, checked against `prob` (and, for a
/// conic problem, against `cones`; empty means the orthant). `None` when the
/// status is neither `PrimalInfeasible` nor `DualInfeasible`, or when the ray
/// is identically zero (a presolve verdict carries none) or non-finite.
pub fn certificate(prob: &QpProblem, cones: &[ConeSpec], sol: &QpSolution) -> Option<Certificate> {
    match sol.status {
        QpStatus::PrimalInfeasible => {
            farkas_certificate(prob, cones, &sol.y, &sol.z).map(Certificate::PrimalInfeasible)
        }
        QpStatus::DualInfeasible => {
            recession_certificate(prob, cones, &sol.x).map(Certificate::DualInfeasible)
        }
        _ => None,
    }
}

/// The certificate behind `sol`'s verdict, from the best ray available.
///
/// Candidates are tried in order and the first that verifies is returned;
/// if none does, the first one present is returned, unverified:
///
/// 1. `farkas_lift` — on a `PrimalInfeasible` verdict reached after
///    presolve, the reduced ray lifted by
///    [`crate::presolve::Presolve::postsolve_farkas`]. Pass `None` when
///    presolve did not run or the verdict is not primal infeasible.
/// 2. `sol`'s own ray ([`certificate`]).
/// 3. `resolve()` — a re-solve of the **original, unreduced** problem, used
///    only when it reaches the same verdict. Its certificate is marked
///    [`CertificateSource::UnreducedResolve`].
///
/// `resolve` is called only when neither ray before it verifies, so a
/// caller pays for a second solve only then; returning `None` from it skips
/// that step (no budget left, or nothing different to solve). The verdict
/// reported beside the certificate stays `sol`'s; a re-solve contributes a
/// ray and nothing else — except when it **contradicts** the verdict, which
/// [`CertificateOutcome::contradicted_by`] reports.
pub fn certificate_or_resolve(
    prob: &QpProblem,
    cones: &[ConeSpec],
    sol: &QpSolution,
    farkas_lift: Option<(&[f64], &[f64])>,
    resolve: impl FnOnce() -> Option<QpSolution>,
) -> CertificateOutcome {
    let done = |certificate| CertificateOutcome {
        certificate,
        contradicted_by: None,
    };
    if !matches!(
        sol.status,
        QpStatus::PrimalInfeasible | QpStatus::DualInfeasible
    ) {
        return done(None);
    }
    let lifted = farkas_lift
        .filter(|_| sol.status == QpStatus::PrimalInfeasible)
        .and_then(|(y, z)| farkas_certificate(prob, cones, y, z))
        .map(Certificate::PrimalInfeasible);
    if lifted.as_ref().is_some_and(Certificate::verified) {
        return done(lifted);
    }
    let own = certificate(prob, cones, sol);
    if own.as_ref().is_some_and(Certificate::verified) {
        return done(own);
    }
    let first = lifted.or(own);
    let resolved = resolve();
    // A re-solve that reaches an optimum contradicts an infeasible or
    // unbounded verdict outright; one that runs out of iterations or time
    // says nothing. An infeasible/unbounded pair is not a contradiction —
    // a model can be both, and the embedding reports whichever ray it
    // reaches first.
    let contradicted_by = resolved
        .as_ref()
        .map(|r| r.status)
        .filter(|s| matches!(s, QpStatus::Optimal | QpStatus::OptimalInaccurate));
    let second = resolved
        .filter(|r| r.status == sol.status)
        .and_then(|r| certificate(prob, cones, &r))
        .map(|c| c.with_source(CertificateSource::UnreducedResolve));
    let certificate = match (first, second) {
        (_, Some(s)) if s.verified() => Some(s),
        (Some(f), _) => Some(f),
        (None, s) => s,
    };
    CertificateOutcome {
        certificate,
        contradicted_by,
    }
}

/// What [`certificate_or_resolve`] found.
#[derive(Debug, Clone, PartialEq)]
pub struct CertificateOutcome {
    /// The certificate, if any ray was found; see [`Certificate::verified`].
    pub certificate: Option<Certificate>,
    /// The status of the unreduced re-solve when it reached an **optimum**
    /// against an infeasible or unbounded verdict. That is the strongest
    /// evidence of a false verdict this code can see — the case gh#414 was on
    /// the optimal side — so a caller must surface it rather than let the
    /// certificate's absence speak for it. `None` when no re-solve ran or it
    /// agreed or was inconclusive.
    pub contradicted_by: Option<QpStatus>,
}

/// Build and check the Farkas certificate from the ray `(y, z)`.
pub fn farkas_certificate(
    prob: &QpProblem,
    cones: &[ConeSpec],
    y: &[f64],
    z: &[f64],
) -> Option<FarkasCertificate> {
    let n = prob.n;
    let (m_eq, m_ineq) = (prob.m_eq(), prob.m_ineq());
    if y.len() != m_eq || z.len() != m_ineq {
        return None;
    }
    if y.iter().chain(z).any(|v| !v.is_finite()) {
        return None;
    }
    // On an orthant the ray's `z` is clipped to `z⁺`, so `z ∈ K*` is exact.
    // A conic `z` is kept as it is and its distance from `K*` reported.
    let orthant = cones.iter().all(|c| matches!(c, ConeSpec::Nonneg(_)));
    let mut z: Vec<f64> = z.to_vec();
    if orthant {
        for v in &mut z {
            *v = v.max(0.0);
        }
    }
    let scale = inf_norm(y).max(inf_norm(&z));
    if scale == 0.0 {
        return None;
    }
    let y: Vec<f64> = y.iter().map(|v| v / scale).collect();
    let z: Vec<f64> = z.iter().map(|v| v / scale).collect();

    // q = Aᵀy + Gᵀz, and the column magnitudes it cancels from.
    let mut q = vec![0.0; n];
    prob.at_mul(&y, &mut q);
    prob.gt_mul(&z, &mut q);
    let mut colmag = vec![0.0; n];
    let mut collen = vec![0usize; n];
    for t in &prob.a {
        colmag[t.col] += (t.val * y[t.row]).abs();
        collen[t.col] += 1;
    }
    for t in &prob.g {
        colmag[t.col] += (t.val * z[t.row]).abs();
        collen[t.col] += 1;
    }

    // Absorb q into the bound multipliers where the bound is finite.
    let mut z_lb = vec![0.0; n];
    let mut z_ub = vec![0.0; n];
    let mut r = vec![0.0; n];
    for j in 0..n {
        let qj = q[j];
        // A column whose `q_j` is below its own rounding is zero: absorbing
        // rounding noise into an infinite bound's multiplier would be a
        // residual of pure noise, and into a finite one would charge noise to
        // the value.
        if qj.abs() <= gamma(collen[j]) * colmag[j] {
            continue;
        }
        if qj > 0.0 {
            if has_lb(prob, j) {
                z_lb[j] = qj;
            } else {
                r[j] = qj;
            }
        } else if has_ub(prob, j) {
            z_ub[j] = -qj;
        } else {
            r[j] = qj;
        }
    }

    let mut value = 0.0;
    let mut value_mag = 0.0;
    let mut terms = 0usize;
    let mut add = |coef: f64, mult: f64| {
        if mult != 0.0 {
            value += coef * mult;
            value_mag += (coef * mult).abs();
            terms += 1;
        }
    };
    for (b, yi) in prob.b.iter().zip(&y) {
        add(*b, *yi);
    }
    for (h, zi) in prob.h.iter().zip(&z) {
        add(*h, *zi);
    }
    for j in 0..n {
        add(-prob.lb_of(j), z_lb[j]);
        add(prob.ub_of(j), z_ub[j]);
    }
    // Normalize the whole certificate to unit inf-norm, bound multipliers
    // included, so the reported vectors and value are on one scale.
    let s2 = scale_of(&[&y, &z, &z_lb, &z_ub]);
    let div = |v: Vec<f64>| -> Vec<f64> { v.into_iter().map(|x| x / s2).collect() };
    let (y, z, z_lb, z_ub) = (div(y), div(z), div(z_lb), div(z_ub));
    let value = value / s2;
    let value_mag = value_mag / s2;
    let residual = inf_norm(&r) / s2;
    let colmag_norm = inf_norm(&colmag) / s2;

    let cone_violation = if orthant {
        0.0
    } else {
        dual_cone_violation(cones, &z)
    };

    let rho_h = if value_mag > 0.0 {
        value.abs() / value_mag
    } else {
        0.0
    };
    let rho_g = if residual == 0.0 {
        0.0
    } else if colmag_norm > 0.0 {
        residual / colmag_norm
    } else {
        f64::INFINITY
    };
    let margin = if rho_g == 0.0 {
        if rho_h > 0.0 { f64::INFINITY } else { 0.0 }
    } else {
        rho_h / rho_g
    };
    // The cone tolerance mirrors the residual's: a `z` outside `K*` by a
    // relative amount `v` is as good as a residual of `v`.
    let cone_ok = cone_violation <= rho_h / FARKAS_MARGIN_MIN;
    let verified = value.is_finite()
        && value + gamma(terms) * value_mag < 0.0
        && margin >= FARKAS_MARGIN_MIN
        && cone_ok;
    Some(FarkasCertificate {
        y,
        z,
        z_lb,
        z_ub,
        value,
        residual,
        cone_violation,
        margin,
        verified,
        source: CertificateSource::Solve,
    })
}

fn scale_of(vs: &[&[f64]]) -> f64 {
    let s = vs.iter().fold(0.0_f64, |m, v| m.max(inf_norm(v)));
    if s > 0.0 { s } else { 1.0 }
}

/// Build and check the recession certificate from the direction `d`.
///
/// An interior-point ray reaches a bound only in the limit, so a component
/// that should be zero at a finite bound comes back a few ulps of `‖d‖` on
/// the wrong side of it (`2.8e-8` measured on a three-column LP). The same
/// direction with those components set to zero is tried as well, and the
/// candidate with the larger margin is the certificate. Zeroing a component
/// moves `Ad`, `Gd`, `Pd` and `cᵀd`, so it is a second candidate measured on
/// its own, not a correction assumed to help.
pub fn recession_certificate(
    prob: &QpProblem,
    cones: &[ConeSpec],
    d: &[f64],
) -> Option<RecessionCertificate> {
    let n = prob.n;
    if d.len() != n || d.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let raw = recession_candidate(prob, cones, d)?;
    if raw.bound_violation == 0.0 {
        return Some(raw);
    }
    let snapped: Vec<f64> = d
        .iter()
        .enumerate()
        .map(|(j, &dj)| {
            let crosses = (dj < 0.0 && has_lb(prob, j)) || (dj > 0.0 && has_ub(prob, j));
            if crosses { 0.0 } else { dj }
        })
        .collect();
    match recession_candidate(prob, cones, &snapped) {
        Some(s) if s.margin > raw.margin || (s.verified && !raw.verified) => Some(s),
        _ => Some(raw),
    }
}

fn recession_candidate(
    prob: &QpProblem,
    cones: &[ConeSpec],
    d: &[f64],
) -> Option<RecessionCertificate> {
    let n = prob.n;
    let scale = inf_norm(d);
    if scale == 0.0 {
        return None;
    }
    let d: Vec<f64> = d.iter().map(|v| v / scale).collect();

    // Each residual entry relative to the magnitude of the terms it is a sum
    // of: ρ = max_i |(Md)_i| / (|M||d|)_i over the rows of `M`.
    let rel = |num: &[f64], den: &[f64]| -> f64 {
        num.iter().zip(den).fold(0.0_f64, |m, (&v, &mag)| {
            if v == 0.0 {
                m
            } else if mag > 0.0 {
                m.max(v.abs() / mag)
            } else {
                f64::INFINITY
            }
        })
    };

    // Pd, from the lower triangle.
    let mut pd = vec![0.0; n];
    prob.p_mul(&d, &mut pd);
    let mut pmag = vec![0.0; n];
    for t in &prob.p_lower {
        pmag[t.row] += (t.val * d[t.col]).abs();
        if t.row != t.col {
            pmag[t.col] += (t.val * d[t.row]).abs();
        }
    }
    let mut ad = vec![0.0; prob.m_eq()];
    prob.a_mul(&d, &mut ad);
    let mut amag = vec![0.0; prob.m_eq()];
    for t in &prob.a {
        amag[t.row] += (t.val * d[t.col]).abs();
    }
    let mut gd = vec![0.0; prob.m_ineq()];
    prob.g_mul(&d, &mut gd);
    let mut gmag = vec![0.0; prob.m_ineq()];
    for t in &prob.g {
        gmag[t.row] += (t.val * d[t.col]).abs();
    }
    let orthant = cones.iter().all(|c| matches!(c, ConeSpec::Nonneg(_)));
    let neg_gd: Vec<f64> = gd.iter().map(|v| -v).collect();
    let (cone_violation, rho_cone) = if orthant {
        let pos: Vec<f64> = gd.iter().map(|v| v.max(0.0)).collect();
        (inf_norm(&pos), rel(&pos, &gmag))
    } else {
        // A conic block's membership is not a per-row property; measure it
        // against the block's own magnitude, the largest `(|G||d|)_i` in it.
        let v = primal_cone_violation(cones, &neg_gd);
        let mag = inf_norm(&gmag);
        let rho = if v == 0.0 {
            0.0
        } else if mag > 0.0 {
            v / mag
        } else {
            f64::INFINITY
        };
        (v, rho)
    };
    let mut bound_violation = 0.0_f64;
    for (j, &dj) in d.iter().enumerate() {
        if has_lb(prob, j) {
            bound_violation = bound_violation.max(-dj);
        }
        if has_ub(prob, j) {
            bound_violation = bound_violation.max(dj);
        }
    }

    let value: f64 = prob.c.iter().zip(&d).map(|(c, d)| c * d).sum();
    let value_mag: f64 = prob.c.iter().zip(&d).map(|(c, d)| (c * d).abs()).sum();
    let terms = prob
        .c
        .iter()
        .zip(&d)
        .filter(|(c, d)| **c * **d != 0.0)
        .count();

    // `‖d‖∞ = 1`, so the bound violation is already relative.
    let rho = rel(&pd, &pmag)
        .max(rel(&ad, &amag))
        .max(rho_cone)
        .max(bound_violation);
    let rho_c = if value_mag > 0.0 {
        value.abs() / value_mag
    } else {
        0.0
    };
    let margin = if rho == 0.0 {
        if rho_c > 0.0 { f64::INFINITY } else { 0.0 }
    } else {
        rho_c / rho
    };
    let verified =
        value.is_finite() && value + gamma(terms) * value_mag < 0.0 && margin >= FARKAS_MARGIN_MIN;
    Some(RecessionCertificate {
        d,
        value,
        hessian_residual: inf_norm(&pd),
        equality_residual: inf_norm(&ad),
        cone_violation,
        bound_violation,
        margin,
        verified,
        source: CertificateSource::Solve,
    })
}

/// Blocks of `v` by `cones` (the orthant when `cones` is empty). Rows past the
/// last block are read as orthant rows.
fn blocks<'a>(cones: &'a [ConeSpec], v: &'a [f64]) -> Vec<(ConeSpec, &'a [f64])> {
    let mut out = Vec::new();
    let mut at = 0;
    for c in cones {
        let k = c.dim().min(v.len().saturating_sub(at));
        out.push((*c, &v[at..at + k]));
        at += k;
    }
    if at < v.len() {
        out.push((ConeSpec::Nonneg(v.len() - at), &v[at..]));
    }
    out
}

/// How far `s` lies outside `K` (zero inside), the largest over blocks.
fn primal_cone_violation(cones: &[ConeSpec], s: &[f64]) -> f64 {
    blocks(cones, s)
        .into_iter()
        .fold(0.0_f64, |m, (c, b)| m.max(crate::qp::cone_violation(&c, b)))
}

/// How far `z` lies outside `K*` (zero inside), the largest over blocks. The
/// orthant, second-order and PSD cones are self-dual; the exponential and
/// power cones are not and have their duals written out here.
fn dual_cone_violation(cones: &[ConeSpec], z: &[f64]) -> f64 {
    let neg = |v: f64| (-v).max(0.0);
    blocks(cones, z).into_iter().fold(0.0_f64, |m, (c, b)| {
        let v = match c {
            ConeSpec::Nonneg(_) | ConeSpec::SecondOrder(_) | ConeSpec::Psd(_) => {
                crate::qp::cone_violation(&c, b)
            }
            // K_exp = cl{(x, y, z) : y > 0, y·e^{x/y} ≤ z}, so
            // K_exp* = cl{(u, v, w) : u < 0, −u·e^{v/u} ≤ e·w}.
            ConeSpec::Exponential => {
                let (u, v, w) = (b[0], b[1], b[2]);
                if u < 0.0 {
                    neg(w).max(-u * (v / u).exp() - std::f64::consts::E * w)
                } else {
                    u.max(neg(v)).max(neg(w))
                }
            }
            // K_α = {(x, y, z) : |x| ≤ y^α z^{1−α}, y, z ≥ 0}, so
            // K_α* = {(u, v, w) : |u| ≤ (v/α)^α (w/(1−α))^{1−α}, v, w ≥ 0}.
            ConeSpec::Power(a) => {
                let (u, v, w) = (b[0], b[1], b[2]);
                let bounds = neg(v).max(neg(w));
                if v < 0.0 || w < 0.0 {
                    bounds.max(u.abs())
                } else {
                    bounds.max(u.abs() - (v / a).powf(a) * (w / (1.0 - a)).powf(1.0 - a))
                }
            }
        };
        m.max(v.max(0.0))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qp::{NEG_INF, POS_INF, Triplet};

    fn sol(status: QpStatus, x: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> QpSolution {
        let n = x.len();
        QpSolution {
            status,
            x,
            y,
            z,
            z_lb: vec![0.0; n],
            z_ub: vec![0.0; n],
            obj: 0.0,
            iters: 0,
            iterates: Vec::new(),
        }
    }

    /// `x₀ ≤ 0`, `x₀ ≥ 1` as rows: `z = (1, 1)` is exact.
    #[test]
    fn exact_farkas_ray_verifies() {
        let p = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![0.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 0, -1.0)],
            h: vec![0.0, -1.0],
            lb: vec![],
            ub: vec![],
        };
        let c = farkas_certificate(&p, &[], &[], &[3.0, 3.0]).unwrap();
        assert!(c.verified, "{c:?}");
        assert_eq!(c.residual, 0.0);
        assert_eq!(c.value, -1.0);
        assert_eq!(c.z, vec![1.0, 1.0]);
    }

    /// The box absorbs `q`: `x₀ ≥ 2` against `x₀ ≤ 1` as a bound.
    #[test]
    fn bound_multipliers_are_rederived() {
        let p = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![0.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0)],
            h: vec![-2.0],
            lb: vec![NEG_INF],
            ub: vec![1.0],
        };
        let c = farkas_certificate(&p, &[], &[], &[1.0]).unwrap();
        assert!(c.verified, "{c:?}");
        assert_eq!(c.z_ub, vec![1.0]);
        assert_eq!(c.value, -1.0);
        // Without the bound nothing absorbs `q`: not a certificate.
        let free = QpProblem {
            ub: vec![POS_INF],
            ..p
        };
        let c = farkas_certificate(&free, &[], &[], &[1.0]).unwrap();
        assert!(!c.verified, "{c:?}");
        assert_eq!(c.residual, 1.0);
    }

    /// A ray whose value is non-negative is no certificate, whatever its
    /// residual.
    #[test]
    fn nonnegative_value_does_not_verify() {
        let p = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![0.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 0, -1.0)],
            h: vec![1.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let c = farkas_certificate(&p, &[], &[], &[1.0, 1.0]).unwrap();
        assert!(!c.verified);
        assert_eq!(c.value, 1.0);
    }

    /// `min −x₀ − x₁` over `x ≥ 0`, `x₀ − x₁ = 0`: `d = (1, 1)`.
    #[test]
    fn exact_recession_verifies_and_curvature_refutes() {
        let p = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![-1.0, -1.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -1.0)],
            b: vec![0.0],
            g: vec![],
            h: vec![],
            lb: vec![0.0, 0.0],
            ub: vec![POS_INF, POS_INF],
        };
        let c = recession_certificate(&p, &[], &[5.0, 5.0]).unwrap();
        assert!(c.verified, "{c:?}");
        assert_eq!(c.value, -2.0);
        assert_eq!(c.equality_residual, 0.0);
        // Against a bound it would cross: not a recession direction.
        let c = recession_certificate(&p, &[], &[-1.0, -1.0]).unwrap();
        assert!(!c.verified);
        assert_eq!(c.bound_violation, 1.0);
        // Curvature along `d`: bounded.
        let curved = QpProblem {
            p_lower: vec![Triplet::new(0, 0, 1.0)],
            ..p
        };
        let c = recession_certificate(&curved, &[], &[1.0, 1.0]).unwrap();
        assert!(!c.verified, "{c:?}");
        assert_eq!(c.hessian_residual, 1.0);
    }

    #[test]
    fn no_certificate_off_the_infeasible_statuses_or_for_a_zero_ray() {
        let p = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![1.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![0.0],
            ub: vec![],
        };
        assert!(certificate(&p, &[], &sol(QpStatus::Optimal, vec![0.0], vec![], vec![])).is_none());
        assert!(
            certificate(
                &p,
                &[],
                &sol(QpStatus::DualInfeasible, vec![0.0], vec![], vec![])
            )
            .is_none()
        );
    }

    /// A re-solve that reaches an optimum against an infeasible verdict is
    /// reported, not silently dropped (gh#1181 review); one that only runs
    /// out of iterations is not a contradiction.
    #[test]
    fn a_resolve_reaching_an_optimum_contradicts_the_verdict() {
        let p = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![1.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![0.0],
            ub: vec![],
        };
        // A presolve-style verdict: no ray at all.
        let verdict = sol(QpStatus::PrimalInfeasible, vec![0.0], vec![], vec![]);
        let out = certificate_or_resolve(&p, &[], &verdict, None, || {
            Some(sol(QpStatus::Optimal, vec![0.0], vec![], vec![]))
        });
        assert_eq!(out.contradicted_by, Some(QpStatus::Optimal));
        assert!(out.certificate.is_none());
        let out = certificate_or_resolve(&p, &[], &verdict, None, || {
            Some(sol(QpStatus::IterationLimit, vec![0.0], vec![], vec![]))
        });
        assert_eq!(out.contradicted_by, None);
        // No re-solve, no claim either way.
        let out = certificate_or_resolve(&p, &[], &verdict, None, || None);
        assert_eq!(out.contradicted_by, None);
    }

    /// SOC dual-cone membership: `(1, 1)` is on the boundary, `(1, 2)` is
    /// outside by `1`.
    #[test]
    fn conic_dual_violation_is_measured() {
        let soc = [ConeSpec::SecondOrder(2)];
        assert_eq!(dual_cone_violation(&soc, &[1.0, 1.0]), 0.0);
        assert_eq!(dual_cone_violation(&soc, &[1.0, 2.0]), 1.0);
        // K_exp*: (−1, 0, w) needs 1 ≤ e·w.
        let e = [ConeSpec::Exponential];
        assert_eq!(dual_cone_violation(&e, &[-1.0, 0.0, 1.0]), 0.0);
        assert!(dual_cone_violation(&e, &[-1.0, 0.0, 0.1]) > 0.0);
    }
}
