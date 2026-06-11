//! Generic 3-D non-symmetric-cone machinery, shared by the exponential and
//! power cones (and any future 3-D [`BarrierCone`]).
//!
//! A non-symmetric cone has no Nesterov–Todd scaling point; the path-following
//! driver instead needs, per iterate, three cone-agnostic ingredients built
//! only from the barrier oracles:
//!
//! - the **conjugate-barrier gradient** `x̃ = −F'_*(z)` (the shadow primal
//!   iterate), computed by a damped Newton solve;
//! - the **dual-aware primal–dual scaling** `M = WᵀW` (the Tunçel scaling
//!   specialized to 3-D, computed by a BFGS update — Dahl & Andersen 2021),
//!   whose defining secants are the `W`-free identities `M·s = z`, `M·x̃ = s̃`;
//! - the **third-order term** `F'''(s)[u, v]` for the nonsymmetric corrector.
//!
//! All three are implemented here once, generic over the cone, so the exp and
//! power cones supply only their barrier oracles (`barrier`, `∇F`, `∇²F`,
//! membership, and an `interior_reference`).

use super::BarrierCone;

// --- small fixed-size 3-vector / 3×3 helpers ------------------------------

#[inline]
fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Symmetric `H` (lower triangle `[h00;h10,h11;h20,h21,h22]`) times a vector.
#[inline]
fn sym_matvec(h: &[f64; 6], v: &[f64; 3]) -> [f64; 3] {
    [
        h[0] * v[0] + h[1] * v[1] + h[3] * v[2],
        h[1] * v[0] + h[2] * v[1] + h[4] * v[2],
        h[3] * v[0] + h[4] * v[1] + h[5] * v[2],
    ]
}

/// Solve the SPD 3×3 system `H x = b`, `H` given by its lower triangle
/// row-major `[h00; h10,h11; h20,h21,h22]`, via Cholesky. `None` if `H` is not
/// numerically positive definite.
pub(crate) fn chol_solve3(h: &[f64; 6], b: &[f64; 3]) -> Option<[f64; 3]> {
    let l00 = h[0];
    if l00 <= 0.0 {
        return None;
    }
    let l00 = l00.sqrt();
    let l10 = h[1] / l00;
    let l11 = h[2] - l10 * l10;
    if l11 <= 0.0 {
        return None;
    }
    let l11 = l11.sqrt();
    let l20 = h[3] / l00;
    let l21 = (h[4] - l20 * l10) / l11;
    let l22 = h[5] - l20 * l20 - l21 * l21;
    if l22 <= 0.0 {
        return None;
    }
    let l22 = l22.sqrt();
    let y0 = b[0] / l00;
    let y1 = (b[1] - l10 * y0) / l11;
    let y2 = (b[2] - l20 * y0 - l21 * y1) / l22;
    let x2 = y2 / l22;
    let x1 = (y1 - l21 * x2) / l11;
    let x0 = (y0 - l10 * x1 - l20 * x2) / l00;
    Some([x0, x1, x2])
}

/// The dual-aware **primal–dual scaling** for a 3-D non-symmetric cone — the
/// Tunçel scaling specialized to 3-D and computed by a BFGS update, exactly as
/// in MOSEK's exp-cone solver (Dahl & Andersen 2021, §5–6). Built from *both*
/// the primal slack `s ∈ K` and the dual `z ∈ K*` (via the shadow iterates),
/// unlike the primal-only Hessian which stalls.
///
/// The driver needs only `M = WᵀW`: Dahl–Andersen's reduced system places `M`
/// in the `(z,z)` cone block, and every RHS term reduces to `M` applied to a
/// vector. The defining double-secant equations (DA eq. 8/29), pre-multiplied
/// by `Wᵀ`, become the exact, `W`-free identities `M·s = z` and `M·x̃ = s̃`.
///
/// pounce convention (`s` primal, `z` dual); the map to Dahl–Andersen's
/// `(x, s)` is `x = s`, `s_DA = z`, so `x̃ = −F'_*(z)` and `s̃ = −∇F(s)`.
#[derive(Debug, Clone)]
pub struct NonsymScaling {
    /// `M = WᵀW`, lower triangle row-major `[m00;m10,m11;m20,m21,m22]` — the
    /// dense `(z,z)` cone block. Satisfies `M·s = z`, `M·x̃ = s̃`.
    pub wtw_lower: [f64; 6],
    /// Shadow primal iterate `x̃ = −F'_*(z)` (∈ int K).
    pub x_tilde: [f64; 3],
    /// Shadow dual iterate `s̃ = −∇F(s)` (∈ int K*).
    pub s_tilde: [f64; 3],
    /// Duality measure `μ = ⟨s,z⟩/ν`.
    pub mu: f64,
    /// Shadow duality measure `μ̃ = ⟨x̃,s̃⟩/ν` (`μ·μ̃ ≥ 1`, `=1` only on path).
    pub mu_tilde: f64,
}

impl NonsymScaling {
    /// Apply `M = WᵀW` to a 3-vector.
    #[inline]
    pub fn apply(&self, v: &[f64; 3]) -> [f64; 3] {
        sym_matvec(&self.wtw_lower, v)
    }

    /// `M⁻¹` as a full symmetric 3×3 — the dense `(z,z)` KKT block is `−M⁻¹`,
    /// and the cone elimination/recovery applies `M⁻¹`. `None` if `M` is not
    /// numerically SPD (should not happen for a valid scaling).
    pub fn minv(&self) -> Option<[[f64; 3]; 3]> {
        let c0 = chol_solve3(&self.wtw_lower, &[1.0, 0.0, 0.0])?;
        let c1 = chol_solve3(&self.wtw_lower, &[0.0, 1.0, 0.0])?;
        let c2 = chol_solve3(&self.wtw_lower, &[0.0, 0.0, 1.0])?;
        Some([
            [c0[0], c1[0], c2[0]],
            [c0[1], c1[1], c2[1]],
            [c0[2], c1[2], c2[2]],
        ])
    }
}

/// The shadow primal iterate `x̃ = −F'_*(d)` for a dual-cone point
/// `d ∈ int K*`: the unique `p ∈ int K` solving `∇F(p) = −d`. The conjugate
/// barrier `F_*` has no closed form for these cones, so `x̃` is computed
/// numerically — it minimizes the strictly convex `G(p) = F(p) + ⟨d, p⟩` over
/// `int K`, solved by **damped Newton** with an Armijo line search guarded by
/// barrier-finiteness (an exact interiority test). Returns `false` if
/// `d ∉ int K*` (no solution) or the iteration fails.
pub(crate) fn conjugate_grad<C: BarrierCone>(cone: &C, d: &[f64], out: &mut [f64]) -> bool {
    // Scaled interior start: along a ray p = t·p̂ the barrier problem has
    // optimum t* = ν/⟨d,p̂⟩ = 3/⟨d,p̂⟩ (from log-homogeneity), which lands the
    // start at the right scale; Newton then corrects the direction.
    let mut phat = [0.0_f64; 3];
    cone.interior_reference(&mut phat);
    let dp = d[0] * phat[0] + d[1] * phat[1] + d[2] * phat[2];
    // NaN-safe: `!(dp > 0.0)` rejects dp <= 0 *and* a NaN dp.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(dp > 0.0) {
        return false; // d ∉ int K* (⟨d,p̂⟩ ≤ 0): no conjugate point.
    }
    let t = 3.0 / dp;
    let mut p = [t * phat[0], t * phat[1], t * phat[2]];

    let gval = |p: &[f64; 3]| cone.barrier(p) + d[0] * p[0] + d[1] * p[1] + d[2] * p[2];
    let mut gp = gval(&p);
    if !gp.is_finite() {
        return false;
    }

    let mut g = [0.0_f64; 3];
    let mut l = [0.0_f64; 6];
    for _ in 0..200 {
        cone.barrier_grad(&p, &mut g);
        let r = [g[0] + d[0], g[1] + d[1], g[2] + d[2]]; // ∇G(p) = ∇F(p)+d
        cone.barrier_hess_lower(&p, &mut l);
        let delta = match chol_solve3(&l, &[-r[0], -r[1], -r[2]]) {
            Some(v) => v,
            None => return false,
        };
        // Newton decrement λ² = rᵀ H⁻¹ r = −rᵀδ.
        let lam2 = -(r[0] * delta[0] + r[1] * delta[1] + r[2] * delta[2]);
        if lam2 <= 1e-24 {
            break; // ∇F(p) ≈ −d.
        }
        let mut step = 1.0_f64;
        loop {
            let pc = [
                p[0] + step * delta[0],
                p[1] + step * delta[1],
                p[2] + step * delta[2],
            ];
            let gc = gval(&pc);
            if gc.is_finite() && gc <= gp - 0.25 * step * lam2 {
                p = pc;
                gp = gc;
                break;
            }
            step *= 0.5;
            if step < 1e-15 {
                return false; // line search collapsed
            }
        }
    }
    out[0] = p[0];
    out[1] = p[1];
    out[2] = p[2];
    true
}

/// Build the dual-aware scaling [`NonsymScaling`] at `(s, z)`. `None` if the
/// iterate is on (or numerically at) the central path — where the scaling
/// degenerates (`YᵀS` singular, `⟨δ_x,δ_s⟩ → 0`) — or if the shadow-iterate
/// solve fails. The driver falls back to the primal Hessian `μ∇²F(s)` then.
pub(crate) fn scaling<C: BarrierCone>(cone: &C, s: &[f64], z: &[f64]) -> Option<NonsymScaling> {
    let nu = 3.0;
    let s3 = [s[0], s[1], s[2]];
    let z3 = [z[0], z[1], z[2]];
    let sz = dot3(&s3, &z3);
    if sz <= 0.0 {
        return None;
    }
    let mu = sz / nu;

    // Shadow iterates: x̃ = −F'_*(z) (conjugate-grad solve), s̃ = −∇F(s).
    let mut xt = [0.0; 3];
    if !conjugate_grad(cone, &z3, &mut xt) {
        return None;
    }
    let mut g = [0.0; 3];
    cone.barrier_grad(&s3, &mut g);
    let st = [-g[0], -g[1], -g[2]];
    let mu_tilde = dot3(&xt, &st) / nu;

    // ⟨δ_x,δ_s⟩ = ⟨s−μx̃, z−μs̃⟩ → 0 on the central path (degenerate).
    let dlt_p = [s3[0] - mu * xt[0], s3[1] - mu * xt[1], s3[2] - mu * xt[2]];
    let dlt_d = [z3[0] - mu * st[0], z3[1] - mu * st[1], z3[2] - mu * st[2]];
    if dot3(&dlt_p, &dlt_d) <= 1e-13 * sz {
        return None;
    }

    // M = Y(YᵀS)⁻¹Yᵀ + t·z_cp z_cpᵀ (DA §5), S = [s, x̃], Y = [z, s̃],
    // z_cp ⊥ {s, x̃} the unit cross product. YᵀS is symmetric by the Euler
    // identities ⟨z,x̃⟩ = ⟨s̃,s⟩ = ν.
    let a00 = dot3(&z3, &s3);
    let a01 = dot3(&z3, &xt);
    let a10 = dot3(&st, &s3);
    let a11 = dot3(&st, &xt);
    let det = a00 * a11 - a01 * a10;
    if det.abs() <= 1e-14 {
        return None;
    }
    let (b00, b01, b10, b11) = (a11 / det, -a01 / det, -a10 / det, a00 / det);

    let zc = cross3(&s3, &xt);
    let zc_norm = dot3(&zc, &zc).sqrt();
    if zc_norm <= 1e-14 {
        return None;
    }
    let z_cp = [zc[0] / zc_norm, zc[1] / zc_norm, zc[2] / zc_norm];

    // BFGS scalar t (DA 32): t = μ·‖ H − s̃s̃ᵀ/ν
    //   − (H x̃ − μ̃ s̃)(H x̃ − μ̃ s̃)ᵀ / (⟨x̃, H x̃⟩ − ν μ̃²) ‖_F .
    let mut hl = [0.0; 6];
    cone.barrier_hess_lower(&s3, &mut hl);
    let hxt = sym_matvec(&hl, &xt);
    let xt_h_xt = dot3(&xt, &hxt);
    let denom_t = xt_h_xt - nu * mu_tilde * mu_tilde;
    if denom_t.abs() <= 1e-14 {
        return None;
    }
    let qv = [
        hxt[0] - mu_tilde * st[0],
        hxt[1] - mu_tilde * st[1],
        hxt[2] - mu_tilde * st[2],
    ];
    let h_full = [
        [hl[0], hl[1], hl[3]],
        [hl[1], hl[2], hl[4]],
        [hl[3], hl[4], hl[5]],
    ];
    let mut fro2 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let m_ij = h_full[i][j] - st[i] * st[j] / nu - qv[i] * qv[j] / denom_t;
            fro2 += m_ij * m_ij;
        }
    }
    let t = mu * fro2.sqrt();
    // NaN-safe: `!(t > 0.0)` rejects t <= 0 *and* a NaN t (which `t <= 0.0`
    // would let through). Bail out rather than build a degenerate factor.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(t > 0.0) {
        return None;
    }

    // M = Y B Yᵀ + t z_cp z_cpᵀ (columns of Y are y0=z, y1=s̃).
    let y0 = z3;
    let y1 = st;
    let mut m_full = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m_full[i][j] = b00 * y0[i] * y0[j]
                + b01 * y0[i] * y1[j]
                + b10 * y1[i] * y0[j]
                + b11 * y1[i] * y1[j]
                + t * z_cp[i] * z_cp[j];
        }
    }
    let wtw = [
        m_full[0][0],
        m_full[1][0],
        m_full[1][1],
        m_full[2][0],
        m_full[2][1],
        m_full[2][2],
    ];

    Some(NonsymScaling {
        wtw_lower: wtw,
        x_tilde: xt,
        s_tilde: st,
        mu,
        mu_tilde,
    })
}

/// The third-order directional term `F'''(s)[u, v]` (a 3-vector) — the
/// ingredient of Dahl–Andersen's nonsymmetric Mehrotra-like corrector
/// (DA eq. 16): `η = −½ F'''(s)[Δxᵃ, (∇²F(s))⁻¹ Δsᵃ]`. Computed as the
/// directional derivative of the Hessian, `F'''(s)[u, v] = d/dt
/// (∇²F(s + t·u)·v)|₀`, by central finite differences of the analytic Hessian
/// (the barrier is smooth). The step `h` is scaled `∝ 1/‖u‖` so the third
/// derivative stays accurate even for a tiny affine step (the endgame). `None`
/// if either perturbed point leaves the cone (then the driver drops the
/// corrector for that block — still a valid centered step).
pub(crate) fn third_dir_apply<C: BarrierCone>(
    cone: &C,
    s: &[f64],
    u: &[f64],
    v: &[f64],
) -> Option<[f64; 3]> {
    let s_scale = 1.0 + s[0].abs().max(s[1].abs()).max(s[2].abs());
    let u_norm = u[0].abs().max(u[1].abs()).max(u[2].abs());
    if u_norm <= 1e-300 {
        return Some([0.0; 3]); // F'''(s)[0, v] = 0
    }
    let h = 1e-6 * s_scale / u_norm;
    let sp = [s[0] + h * u[0], s[1] + h * u[1], s[2] + h * u[2]];
    let sm = [s[0] - h * u[0], s[1] - h * u[1], s[2] - h * u[2]];
    if !cone.in_primal_cone(&sp, 1e-12) || !cone.in_primal_cone(&sm, 1e-12) {
        return None;
    }
    let v3 = [v[0], v[1], v[2]];
    let mut lp = [0.0; 6];
    let mut lm = [0.0; 6];
    cone.barrier_hess_lower(&sp, &mut lp);
    cone.barrier_hess_lower(&sm, &mut lm);
    let hpv = sym_matvec(&lp, &v3);
    let hmv = sym_matvec(&lm, &v3);
    let inv = 1.0 / (2.0 * h);
    Some([
        (hpv[0] - hmv[0]) * inv,
        (hpv[1] - hmv[1]) * inv,
        (hpv[2] - hmv[2]) * inv,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cones::{ExponentialCone, PowerCone};

    /// Validate the generic machinery on one cone: the conjugate-gradient
    /// round-trip, the scaling's defining secants `M·s = z`, `M·x̃ = s̃` (with
    /// `M` SPD), and the third-derivative homogeneity identity
    /// `F'''(s)[s, v] = −2∇²F·v`.
    fn check_machinery<C: BarrierCone>(cone: &C, pts: &[[f64; 3]]) {
        // --- conjugate-gradient round-trip: d = −∇F(p) ⇒ recover p. ---
        for &p in pts {
            let mut g = [0.0; 3];
            cone.barrier_grad(&p, &mut g);
            let d = [-g[0], -g[1], -g[2]];
            assert!(cone.in_dual_cone(&d, 1e-12), "−∇F(p) must be dual-interior");
            let mut xt = [0.0; 3];
            assert!(
                conjugate_grad(cone, &d, &mut xt),
                "conjugate_grad failed at {p:?}"
            );
            for k in 0..3 {
                assert!(
                    (xt[k] - p[k]).abs() < 1e-8,
                    "round-trip[{k}] {} vs {}",
                    xt[k],
                    p[k]
                );
            }
        }

        // --- scaling secants on off-path pairs (s, z = −∇F(s2)), s2 ≁ s. ---
        for i in 0..pts.len() {
            for j in 0..pts.len() {
                if i == j {
                    continue;
                }
                let s = pts[i];
                let mut g = [0.0; 3];
                cone.barrier_grad(&pts[j], &mut g);
                let z = [-g[0], -g[1], -g[2]];
                let sc = match scaling(cone, &s, &z) {
                    Some(sc) => sc,
                    None => continue, // (rare) numerically on-path: skip
                };
                let ms = sc.apply(&s);
                for k in 0..3 {
                    assert!(
                        (ms[k] - z[k]).abs() < 1e-7,
                        "secant M·s=z [{k}]: {} vs {}",
                        ms[k],
                        z[k]
                    );
                }
                let mxt = sc.apply(&sc.x_tilde);
                for k in 0..3 {
                    assert!(
                        (mxt[k] - sc.s_tilde[k]).abs() < 1e-7,
                        "secant M·x̃=s̃ [{k}]: {} vs {}",
                        mxt[k],
                        sc.s_tilde[k]
                    );
                }
                assert!(
                    chol_solve3(&sc.wtw_lower, &[1.0, 0.0, 0.0]).is_some(),
                    "M not SPD: {:?}",
                    sc.wtw_lower
                );
            }
        }

        // --- third-derivative homogeneity: F'''(s)[s, v] = −2∇²F·v. ---
        let vs = [[1.0, 0.0, 0.0], [0.3, -0.7, 1.1], [-2.0, 0.5, 0.4]];
        for &p in pts {
            let mut hl = [0.0; 6];
            cone.barrier_hess_lower(&p, &mut hl);
            for v in vs {
                let hv = sym_matvec(&hl, &v);
                let t3 = third_dir_apply(cone, &p, &p, &v).expect("interior");
                for k in 0..3 {
                    assert!(
                        (t3[k] + 2.0 * hv[k]).abs() < 1e-6,
                        "F'''[s,v][{k}] {} vs −2Hv {}",
                        t3[k],
                        -2.0 * hv[k]
                    );
                }
            }
        }
    }

    #[test]
    fn machinery_on_exponential_cone() {
        use std::f64::consts::E;
        check_machinery(
            &ExponentialCone,
            &[
                [0.0, 1.0, E],
                [-1.0, 2.0, 3.0],
                [0.5, 1.5, 4.0],
                [-2.0, 0.7, 1.2],
            ],
        );
    }

    #[test]
    fn machinery_on_power_cone() {
        let pts = [
            [0.0, 1.0, 1.0],
            [0.3, 2.0, 1.5],
            [-0.5, 1.2, 3.0],
            [0.1, 0.7, 0.9],
        ];
        for alpha in [0.5, 0.3, 0.7] {
            check_machinery(&PowerCone::new(alpha), &pts);
        }
    }
}
