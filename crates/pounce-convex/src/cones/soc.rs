//! Second-order (Lorentz) cone `K = { (t, x) : t ≥ ‖x‖₂ }` for the convex
//! IPM.
//!
//! SOCP extension (see `dev-notes/socp-extension.md`). This module
//! implements the full Nesterov–Todd machinery for a single SOC block:
//!
//! - the Jordan-algebra geometry (`∘`, identity `e`, the `det` quadratic),
//! - the central-path measure `μ = ⟨s, z⟩ / 2` (rank 2, regardless of
//!   dimension),
//! - the fraction-to-boundary `max_step` (the cone-boundary root),
//! - the **Nesterov–Todd scaling Hessian** `W² = η²(2 w̄ w̄ᵀ − J)` that
//!   enters the KKT `(z, z)` block, with its defining identities
//!   (`W² s = z`, symmetric PD, `W² = I` at `s = z`) verified in tests, and
//! - the *reduced-system* methods — `rhs_comp_term` (the `(z)`-row term
//!   `Arw(z)⁻¹ r_comp`), `recover_ds` (`−Arw(z)⁻¹ r_comp − W⁻² dz`), and
//!   the Mehrotra second-order `comp_residual_corrector` — all carrying the
//!   NT scaling/sign conventions, checked against the orthant in the 1-D
//!   limit (`one_dimensional_cone_matches_orthant`).
//!
//! These are fully implemented and **production-wired**: the HSDE solver
//! (`hsde_nonsym.rs`) builds and steps SOC blocks, and the CLI routes a
//! convex QCQP / `solver_selection=socp` to this path
//! (`pounce-cli` dispatch → `SolverChoice::SocpIpm`). The only method left
//! `unimplemented!` is `scaling_diag`, which is deliberately not applicable
//! to SOC (the driver consumes the dense `kkt_block`, not a diagonal).

use super::{Cone, ConeBlock};

/// The second-order cone of a given dimension `m` (`m ≥ 1`):
/// `{ u ∈ ℝᵐ : u₀ ≥ ‖u_{1..}‖₂ }`.
#[derive(Debug, Clone, Copy)]
pub struct SecondOrderCone {
    m: usize,
}

impl SecondOrderCone {
    pub fn new(m: usize) -> Self {
        // A meaningful second-order cone has `m ≥ 1` (`m = 1` degenerates to a
        // single nonnegative coordinate); callers must validate the dimension
        // first (the Python binding does, in `parse_cones` — gh #278). We no
        // longer `assert!` here: a stray `m = 0` yields a trivial, panic-free
        // block instead of aborting across the FFI boundary. Its Jordan/step
        // methods assume `m ≥ 1` and must not be *stepped* at `m = 0`, but
        // constructing and querying such a block never panics.
        SecondOrderCone { m }
    }

    /// `det(u) = u₀² − ‖u_{1..}‖²` — the cone's quadratic form (`uᵀJu`,
    /// `J = diag(1,−1,…,−1)`). Positive in the interior.
    pub fn det(u: &[f64]) -> f64 {
        let tail: f64 = u[1..].iter().map(|v| v * v).sum();
        u[0] * u[0] - tail
    }

    /// Jordan product `s ∘ z = (sᵀz, s₀ z_{1..} + z₀ s_{1..})`.
    pub fn jordan(s: &[f64], z: &[f64], out: &mut [f64]) {
        let dot: f64 = s.iter().zip(z).map(|(a, b)| a * b).sum();
        out[0] = dot;
        for k in 1..s.len() {
            out[k] = s[0] * z[k] + z[0] * s[k];
        }
    }

    /// The Nesterov–Todd scaling: returns `(η, w̄)` with `w̄` the scaling
    /// point (`det(w̄) = 1`, `w̄₀ > 0`) and `η² = √det(s)/√det(z)`. The
    /// scaling Hessian is then `W² = η²(2 w̄ w̄ᵀ − J)`.
    fn nt_scaling(s: &[f64], z: &[f64]) -> (f64, Vec<f64>) {
        let m = s.len();
        let s_det = Self::det(s).max(0.0).sqrt(); // √det(s)
        let z_det = Self::det(z).max(0.0).sqrt();
        // Normalize to the cone's unit-determinant sphere.
        let s_bar: Vec<f64> = s.iter().map(|v| v / s_det).collect();
        let z_bar: Vec<f64> = z.iter().map(|v| v / z_det).collect();
        let sz: f64 = s_bar.iter().zip(&z_bar).map(|(a, b)| a * b).sum();
        let gamma = ((1.0 + sz) / 2.0).sqrt();
        // w̄ = (s̄ + J z̄) / (2γ),  J z̄ = (z̄₀, −z̄_{1..}).
        let mut w_bar = vec![0.0; m];
        w_bar[0] = (s_bar[0] + z_bar[0]) / (2.0 * gamma);
        for k in 1..m {
            w_bar[k] = (s_bar[k] - z_bar[k]) / (2.0 * gamma);
        }
        let eta = (s_det / z_det).sqrt();
        (eta, w_bar)
    }

    /// The block's barrier stiffness `Σ = W⁻²` at `(s, z)`, as the `(η, w̄)`
    /// that [`Self::apply_stiffness`] takes — the second-order analogue of the
    /// orthant's `zᵢ/sᵢ`, which is what the `σ` path's forward-error estimate
    /// needs (gh #1056). `None` where the pair implies nothing usable: either
    /// point not strictly inside the cone (`det ≤ 0` or a non-positive first
    /// coordinate, which a converged active block reaches by rounding), or a
    /// non-finite scaling. Same contract as the orthant's `barrier_ratio`:
    /// the caller declines on `None`, it never substitutes a zero.
    pub(crate) fn stiffness(s: &[f64], z: &[f64]) -> Option<(f64, Vec<f64>)> {
        if s.is_empty() || !(s[0] > 0.0 && z[0] > 0.0) {
            return None;
        }
        if !(Self::det(s) > 0.0 && Self::det(z) > 0.0) {
            return None;
        }
        let (eta, w_bar) = Self::nt_scaling(s, z);
        (eta.is_finite() && eta > 0.0 && w_bar.iter().all(|w| w.is_finite()))
            .then_some((eta, w_bar))
    }

    /// Apply `W⁻² = η⁻²(2 (Jw̄)(Jw̄)ᵀ − J)`, the inverse of the block
    /// [`Self::kkt_block`] assembles (`W̄⁻¹ = J W̄ J` for `det(w̄) = 1`).
    pub(crate) fn apply_stiffness(eta: f64, w_bar: &[f64], v: &[f64], out: &mut [f64]) {
        let inv2 = 1.0 / (eta * eta);
        // (Jw̄)ᵀv = w̄₀v₀ − Σ w̄ₖvₖ.
        let jwv: f64 = w_bar[0] * v[0]
            - w_bar[1..]
                .iter()
                .zip(&v[1..])
                .map(|(w, x)| w * x)
                .sum::<f64>();
        out[0] = inv2 * (2.0 * w_bar[0] * jwv - v[0]);
        for k in 1..w_bar.len() {
            out[k] = inv2 * (-2.0 * w_bar[k] * jwv + v[k]);
        }
    }

    /// Apply the scaling block `W² = η²(2 w̄ w̄ᵀ − J)` to a vector — the
    /// matrix-free form of the dense block returned by [`Self::kkt_block`],
    /// used in `recover_ds` so the recovered slack step is *exactly*
    /// consistent with the assembled KKT block.
    fn apply_w2(eta: f64, w_bar: &[f64], dz: &[f64], out: &mut [f64]) {
        let eta2 = eta * eta;
        let wd: f64 = w_bar.iter().zip(dz).map(|(w, d)| w * d).sum();
        out[0] = eta2 * (2.0 * w_bar[0] * wd - dz[0]); // (J dz)₀ = dz₀
        for k in 1..w_bar.len() {
            out[k] = eta2 * (2.0 * w_bar[k] * wd + dz[k]); // (J dz)_k = −dz_k
        }
    }

    /// Apply `Arw(z)⁻¹` to `b` (solve the arrow system `Arw(z) x = b`),
    /// where `Arw(z) = [[z₀, z₁ᵀ], [z₁, z₀ I]]`. This is the cone's
    /// "division by z"; for a 1-D cone it is `b / z`.
    fn arw_inv(z: &[f64], b: &[f64], out: &mut [f64]) {
        let m = z.len();
        let z1_b1: f64 = z[1..].iter().zip(&b[1..]).map(|(p, q)| p * q).sum();
        let det = Self::det(z);
        let x0 = (z[0] * b[0] - z1_b1) / det;
        out[0] = x0;
        for k in 1..m {
            out[k] = (b[k] - x0 * z[k]) / z[0];
        }
    }
}

impl Cone for SecondOrderCone {
    fn degree(&self) -> usize {
        2 // rank of the second-order cone, independent of dimension
    }

    fn identity(&self, out: &mut [f64]) {
        out.iter_mut().for_each(|v| *v = 0.0);
        if let Some(first) = out.first_mut() {
            *first = 1.0; // e = (1, 0, …, 0); no-op on a degenerate m = 0 block
        }
    }

    fn dim(&self) -> usize {
        self.m
    }

    fn mu(&self, s: &[f64], z: &[f64]) -> f64 {
        let dot: f64 = s.iter().zip(z).map(|(a, b)| a * b).sum();
        dot / 2.0
    }

    fn kkt_block(&self, s: &[f64], z: &[f64]) -> ConeBlock {
        // Diagonal-plus-rank-1 form of W² = η²(2 w̄w̄ᵀ − J)
        //   = diag(η²·(−J)) + (√2 η w̄)(√2 η w̄)ᵀ,
        // so the KKT assembly can keep it sparse via one auxiliary variable.
        let (eta, w_bar) = Self::nt_scaling(s, z);
        let eta2 = eta * eta;
        let mut diag = vec![eta2; self.m];
        diag[0] = -eta2; // −J = diag(−1, 1, …, 1) ⇒ η²·(−J)₀ = −η²
        let scale = (2.0_f64).sqrt() * eta;
        let u: Vec<f64> = w_bar.iter().map(|w| scale * w).collect();
        ConeBlock::DiagPlusRank1 { diag, u }
    }

    fn comp_residual(&self, s: &[f64], z: &[f64], sigma_mu: f64, out: &mut [f64]) {
        // s ∘ z − σμ e.
        Self::jordan(s, z, out);
        out[0] -= sigma_mu;
    }

    fn max_step(&self, v: &[f64], dv: &[f64], tau: f64) -> f64 {
        // Largest α with v + α dv in int(K): det(v+αdv) ≥ 0 and first
        // coordinate ≥ 0. det is the quadratic a α² + b α + c with
        // a = det(dv), c = det(v) > 0, b = 2 (v J dv).
        let a = Self::det(dv);
        let c = Self::det(v);
        let tail: f64 = v[1..].iter().zip(&dv[1..]).map(|(p, q)| p * q).sum();
        let b = 2.0 * (v[0] * dv[0] - tail);

        let mut alpha = f64::INFINITY;
        // Determinant boundary (smallest positive root of a α² + b α + c).
        let disc = b * b - 4.0 * a * c;
        if a.abs() <= 1e-300 {
            if b < 0.0 {
                alpha = alpha.min(-c / b);
            }
        } else if disc >= 0.0 {
            let sq = disc.sqrt();
            for r in [(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)] {
                if r > 0.0 {
                    alpha = alpha.min(r);
                }
            }
        }
        // First-coordinate boundary v₀ + α dv₀ ≥ 0.
        if dv[0] < 0.0 {
            alpha = alpha.min(-v[0] / dv[0]);
        }
        if !alpha.is_finite() {
            return 1.0; // no binding boundary in the step direction
        }
        (tau * alpha).min(1.0)
    }

    fn in_dual_cone(&self, z: &[f64], tol: f64) -> bool {
        // Self-dual: z ∈ K iff z₀ ≥ ‖z₁..‖ − tol.
        let tail: f64 = z[1..self.m].iter().map(|v| v * v).sum::<f64>().sqrt();
        z[0] >= tail - tol
    }

    fn scaling_diag(&self, _s: &[f64], _z: &[f64], _out: &mut [f64]) {
        // SOC's (z,z) block is dense — the driver consumes `kkt_block`, not
        // the orthant's diagonal-only `scaling_diag`.
        unimplemented!("SOC uses kkt_block, not scaling_diag")
    }

    fn comp_residual_corrector(
        &self,
        s: &[f64],
        z: &[f64],
        ds_aff: &[f64],
        dz_aff: &[f64],
        sigma_mu: f64,
        out: &mut [f64],
    ) {
        // Mehrotra second-order term, formed in the **NT-scaled** space
        // (gh #1050), as Clarabel/ECOS/CVXOPT do. With `λ = W z = W⁻¹ s`, the
        // scaled linearization of `s∘z = σμe` is
        //   λ∘(W⁻¹Δs + WΔz) = −r̂,  r̂ = λ∘λ + (W⁻¹Δs_aff)∘(WΔz_aff) − σμe,
        // whose slack step is `Δs = −W(λ \ r̂) − W²Δz`. The reduced system
        // consumes `r_comp` as `Arw(z)⁻¹ r_comp` (`rhs_comp_term`,
        // `recover_ds`), so we return `r_comp = z ∘ W(λ \ r̂)` — the exact
        // unscaled image of the scaled residual.
        //
        // The scaled and unscaled forms agree on `s∘z − σμe` (the identity
        // `W λ⁻¹ = z⁻¹`), so the predictor and the centering term are
        // unchanged; they differ *only* in the second-order term, where the
        // unscaled `Δs_aff∘Δz_aff` is not invariant under the cone's
        // automorphisms. Near an active SOC boundary (`det s, det z → 0`,
        // strictly complementary) that unscaled term is badly mis-sized: on
        // gh #1050 it steered the iterate off the central path into a
        // short-step cycle (α ≈ 0.3, 0.03, 0.6, …) that grew `μ` by four
        // orders of magnitude per ten iterations and ended `IterationLimit`.
        let m = self.m;
        let (eta, w_bar) = Self::nt_scaling(s, z);
        // W = η(2vvᵀ − J), v = (w̄ + e)/√(2(w̄₀ + 1)); W² = η²(2w̄w̄ᵀ − J) is
        // the block `kkt_block` assembles. W⁻¹ = η⁻¹(2(Jv)(Jv)ᵀ − J).
        let nv = (2.0 * (w_bar[0] + 1.0)).sqrt();
        let mut v = vec![0.0; m];
        v[0] = (w_bar[0] + 1.0) / nv;
        for k in 1..m {
            v[k] = w_bar[k] / nv;
        }
        // y = scale·(2 q (qᵀu) − J u), with q = v (W) or q = Jv (W⁻¹).
        let apply = |u: &[f64], inverse: bool, out: &mut [f64]| {
            let scale = if inverse { 1.0 / eta } else { eta };
            let sign = if inverse { -1.0 } else { 1.0 };
            let mut qu = v[0] * u[0];
            for k in 1..m {
                qu += sign * v[k] * u[k];
            }
            out[0] = scale * (2.0 * v[0] * qu - u[0]);
            for k in 1..m {
                out[k] = scale * (2.0 * sign * v[k] * qu + u[k]);
            }
        };
        let mut lambda = vec![0.0; m];
        apply(z, false, &mut lambda);
        let mut a = vec![0.0; m];
        let mut b = vec![0.0; m];
        apply(ds_aff, true, &mut a);
        apply(dz_aff, false, &mut b);
        let mut r_hat = vec![0.0; m];
        let mut second = vec![0.0; m];
        Self::jordan(&lambda, &lambda, &mut r_hat);
        Self::jordan(&a, &b, &mut second);
        for k in 0..m {
            r_hat[k] += second[k];
        }
        r_hat[0] -= sigma_mu;
        // t = λ \ r̂, then u = W t, then r_comp = z ∘ u.
        let mut t = vec![0.0; m];
        Self::arw_inv(&lambda, &r_hat, &mut t);
        let mut u = vec![0.0; m];
        apply(&t, false, &mut u);
        Self::jordan(z, &u, out);
    }

    fn rhs_comp_term(&self, _s: &[f64], z: &[f64], r_comp: &[f64], out: &mut [f64]) {
        // Reduced-KKT (z)-row term: Arw(z)⁻¹ r_comp. Coincides with the NT
        // term −W⁻¹ r̂ via the identity W⁻¹λ⁻¹ = z⁻¹; reduces to r_comp/z in
        // 1-D.
        Self::arw_inv(z, r_comp, out);
    }

    fn recenter_warm(&self, s: &mut [f64], z: &mut [f64], floor: f64) {
        // A *converged* conic warm point sits on the cone boundary
        // (λ_min = u₀ − ‖u₁‖ ≈ 0), where the NT scaling is singular
        // (det → 0). Unlike the orthant, the IPM cannot dwell near that
        // boundary without the factorization blowing up, so seeding the SOC
        // duals there is unstable. We therefore **re-center** each block to
        // a well-conditioned axis point `c·e` (so `s∘z = c²e`, perfectly
        // centered): the warm benefit for SOC comes from the primal `x`
        // (which seeds `s = h − Gx` and the residuals), while the cone duals
        // restart centered. Magnitude is preserved so the scale is sensible.
        let center = |u: &mut [f64]| {
            let mag = u
                .iter()
                .fold(0.0_f64, |m, &v| m.max(v.abs()))
                .max(floor)
                .max(1.0);
            u.iter_mut().for_each(|v| *v = 0.0);
            u[0] = mag;
        };
        center(s);
        center(z);
    }

    fn recover_ds(&self, s: &[f64], z: &[f64], r_comp: &[f64], dz: &[f64], ds: &mut [f64]) {
        // ds = −Arw(z)⁻¹ r_comp − W⁻² dz, exactly consistent with the
        // assembled block (`apply_w2` ≡ `kkt_block` as an operator) and the
        // rhs term above. Reduces to −r_comp/z − (s/z) dz in 1-D.
        let (eta, w_bar) = Self::nt_scaling(s, z);
        let mut rhs = vec![0.0; self.m];
        Self::arw_inv(z, r_comp, &mut rhs);
        let mut w2dz = vec![0.0; self.m];
        Self::apply_w2(eta, &w_bar, dz, &mut w2dz);
        for k in 0..self.m {
            ds[k] = -rhs[k] - w2dz[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn in_interior(u: &[f64]) -> bool {
        u[0] > 0.0 && SecondOrderCone::det(u) > 0.0
    }

    /// Reconstruct the dense symmetric `W² = diag(d) + u uᵀ` from the
    /// cone's diagonal-plus-rank-1 block.
    fn dense(block: &ConeBlock, m: usize) -> Vec<Vec<f64>> {
        let (diag, u) = match block {
            ConeBlock::DiagPlusRank1 { diag, u } => {
                assert_eq!(diag.len(), m);
                (diag, u)
            }
            _ => panic!("expected diag-plus-rank-1 block"),
        };
        let mut w = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                w[i][j] = u[i] * u[j] + if i == j { diag[i] } else { 0.0 };
            }
        }
        w
    }

    fn matvec(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        w.iter()
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// gh #278: a degenerate `m = 0` cone (which a Python caller could reach
    /// via `("soc", 0)`, a negative dim, or a fractional dim rounding to 0)
    /// must *construct and query* without panicking — the raw `assert!` that
    /// used to live in `new` aborted across the FFI boundary. The Python
    /// binding rejects such a dim in `parse_cones`; this is defense in depth.
    #[test]
    fn zero_dimension_cone_does_not_panic() {
        let c = SecondOrderCone::new(0);
        assert_eq!(c.dim(), 0);
        assert_eq!(c.degree(), 2);
        // `identity` on an empty slice is a no-op, not an out-of-bounds write.
        let mut e: Vec<f64> = Vec::new();
        c.identity(&mut e);
        assert!(e.is_empty());
    }

    #[test]
    fn mu_is_half_inner_product() {
        let c = SecondOrderCone::new(3);
        // rank 2 ⇒ μ = ⟨s,z⟩ / 2.
        let s = [2.0, 0.5, 0.5];
        let z = [3.0, -1.0, 0.0];
        let dot = 2.0 * 3.0 + 0.5 * -1.0 + 0.5 * 0.0;
        assert!((c.mu(&s, &z) - dot / 2.0).abs() < 1e-12);
    }

    #[test]
    fn nt_hessian_maps_z_to_s() {
        // The (z,z) scaling block maps z → s, matching the orthant's
        // diag(s/z) (which satisfies diag(s/z)·z = s). For the SOC this is
        // W² = η² Q_{w̄}, with W² symmetric PD. (Equivalently the NT
        // identity z = W² s holds with the inverse scaling; we test the
        // form the KKT block actually uses.)
        let c = SecondOrderCone::new(3);
        let s = [2.0, 0.5, -0.5]; // det = 4 - 0.5 = 3.5 > 0
        let z = [3.0, 1.0, 0.5]; // det = 9 - 1.25 > 0
        assert!(in_interior(&s) && in_interior(&z));
        let w2 = dense(&c.kkt_block(&s, &z), 3);
        let wz = matvec(&w2, &z);
        for k in 0..3 {
            assert!((wz[k] - s[k]).abs() < 1e-9, "W²z[{k}]={} s={}", wz[k], s[k]);
        }
        // Symmetry.
        for i in 0..3 {
            for j in 0..3 {
                assert!((w2[i][j] - w2[j][i]).abs() < 1e-12);
            }
        }
        // Positive definiteness via positive determinant + positive (0,0)
        // leading minor chain on this 3×3 (cheap check: xᵀW²x > 0 on a few
        // probes including the cone axis).
        for x in [[1.0, 0.0, 0.0], [0.3, 0.7, -0.2], [-0.5, 0.1, 0.9]] {
            let q: f64 = x.iter().zip(matvec(&w2, &x)).map(|(a, b)| a * b).sum();
            assert!(q > 0.0, "W² not PD on probe {x:?}: {q}");
        }
    }

    #[test]
    fn nt_hessian_is_identity_at_s_equals_z() {
        let c = SecondOrderCone::new(4);
        let s = [3.0, 1.0, -0.5, 0.5];
        let w2 = dense(&c.kkt_block(&s, &s), 4);
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((w2[i][j] - want).abs() < 1e-9, "W²[{i}][{j}]={}", w2[i][j]);
            }
        }
    }

    /// gh #1056: `apply_stiffness` is the inverse of the assembled `W²`, so
    /// `W⁻²·(W²·v) = v`, and on `W² z = s` it maps `s` back to `z`.
    #[test]
    fn stiffness_inverts_the_nt_block() {
        let c = SecondOrderCone::new(4);
        let s = [3.0, 1.0, -0.5, 0.5];
        let z = [2.0, -0.3, 0.8, 0.1];
        let w2 = dense(&c.kkt_block(&s, &z), 4);
        let (eta, w_bar) = SecondOrderCone::stiffness(&s, &z).expect("interior pair");
        let v = [0.7, -1.1, 0.2, 2.5];
        let w2v: Vec<f64> = (0..4)
            .map(|i| (0..4).map(|j| w2[i][j] * v[j]).sum())
            .collect();
        let mut back = [0.0; 4];
        SecondOrderCone::apply_stiffness(eta, &w_bar, &w2v, &mut back);
        for k in 0..4 {
            assert!(
                (back[k] - v[k]).abs() < 1e-12,
                "k={k}: {} vs {}",
                back[k],
                v[k]
            );
        }
        let mut zz = [0.0; 4];
        SecondOrderCone::apply_stiffness(eta, &w_bar, &s, &mut zz);
        for k in 0..4 {
            assert!((zz[k] - z[k]).abs() < 1e-12, "k={k}: {} vs {}", zz[k], z[k]);
        }
        // Not strictly interior: no stiffness, never a made-up one.
        assert!(SecondOrderCone::stiffness(&[1.0, 1.0, 0.0, 0.0], &z).is_none());
        assert!(SecondOrderCone::stiffness(&s, &[-1.0, 0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn comp_residual_is_jordan_minus_sigma_mu_e() {
        let c = SecondOrderCone::new(3);
        let s = [2.0, 0.5, -0.5];
        let z = [3.0, 1.0, 0.5];
        let mut out = [0.0; 3];
        c.comp_residual(&s, &z, 0.7, &mut out);
        let dot = 2.0 * 3.0 + 0.5 * 1.0 + -0.5 * 0.5;
        assert!((out[0] - (dot - 0.7)).abs() < 1e-12);
        assert!((out[1] - (s[0] * z[1] + z[0] * s[1])).abs() < 1e-12);
        assert!((out[2] - (s[0] * z[2] + z[0] * s[2])).abs() < 1e-12);
    }

    #[test]
    fn max_step_lands_on_the_cone_boundary() {
        let c = SecondOrderCone::new(3);
        let v = [2.0, 0.0, 0.0]; // interior, det = 4
        let dv = [-1.0, 1.0, 0.0]; // heads toward / out of the cone
        // Step to boundary (tau = 1): det(v+αdv) = 0.
        let alpha = c.max_step(&v, &dv, 1.0);
        let p: Vec<f64> = (0..3).map(|k| v[k] + alpha * dv[k]).collect();
        // Either on the determinant boundary or the step was capped at 1.
        assert!(alpha <= 1.0 + 1e-12);
        if alpha < 1.0 - 1e-9 {
            assert!(
                SecondOrderCone::det(&p).abs() < 1e-7,
                "det={}",
                SecondOrderCone::det(&p)
            );
        }
    }

    #[test]
    fn max_step_caps_at_one_when_staying_interior() {
        let c = SecondOrderCone::new(3);
        let v = [5.0, 0.0, 0.0];
        let dv = [1.0, 0.1, -0.1]; // det(dv)=1-0.02>0, b>0 ⇒ stays interior
        assert!((c.max_step(&v, &dv, 0.99) - 1.0).abs() < 1e-12);
    }

    /// `arw_inv` is a genuine inverse: Arw(z)·arw_inv(z,b) = b. This is the
    /// operator the reduced-system rhs / `recover_ds` rely on.
    #[test]
    fn arw_inv_inverts_the_arrow_operator() {
        let z = [3.0, 1.0, -0.5]; // interior
        let b = [0.7, -0.2, 0.4];
        let mut x = [0.0; 3];
        SecondOrderCone::arw_inv(&z, &b, &mut x);
        // Arw(z) x = (z·x, z₀ x₁ + x₀ z₁).
        let zx: f64 = z.iter().zip(&x).map(|(a, c)| a * c).sum();
        assert!((zx - b[0]).abs() < 1e-12);
        for k in 1..3 {
            assert!((z[0] * x[k] + x[0] * z[k] - b[k]).abs() < 1e-12);
        }
    }

    /// `apply_w2` (matrix-free) equals the dense `kkt_block` matrix times
    /// the vector — so `recover_ds`'s `W⁻²dz` is *exactly* the assembled
    /// KKT block, the consistency the reduced system depends on.
    #[test]
    fn apply_w2_matches_dense_kkt_block() {
        let c = SecondOrderCone::new(4);
        let s = [2.0, 0.5, -0.5, 0.3];
        let z = [3.0, 1.0, 0.5, -0.2];
        let w2 = dense(&c.kkt_block(&s, &z), 4);
        let dz = [0.3, -0.7, 0.2, 0.9];
        let want = matvec(&w2, &dz);
        let (eta, w_bar) = SecondOrderCone::nt_scaling(&s, &z);
        let mut got = [0.0; 4];
        SecondOrderCone::apply_w2(eta, &w_bar, &dz, &mut got);
        for k in 0..4 {
            assert!(
                (got[k] - want[k]).abs() < 1e-12,
                "k={k}: {} vs {}",
                got[k],
                want[k]
            );
        }
    }

    /// gh #1050: with no affine step the scaled corrector residual is the
    /// plain `s∘z − σμe` (the predictor/centering terms are frame-free), and
    /// with one it is the *scaled* second-order term — `Arw(z)⁻¹ r_comp`
    /// equals `W(λ \ r̂)` with `r̂ = λ∘λ + (W⁻¹Δs)∘(WΔz) − σμe`, checked here
    /// against an independent dense construction of `W` from `W² = kkt_block`.
    #[test]
    fn corrector_forms_the_second_order_term_in_the_nt_scaled_space() {
        let c = SecondOrderCone::new(4);
        let s = [2.0, 0.5, -0.5, 0.3];
        let z = [3.0, 1.0, 0.5, -0.2];
        let zero = [0.0; 4];
        let (mut plain, mut corr) = ([0.0; 4], [0.0; 4]);
        c.comp_residual(&s, &z, 0.7, &mut plain);
        c.comp_residual_corrector(&s, &z, &zero, &zero, 0.7, &mut corr);
        for k in 0..4 {
            assert!(
                (plain[k] - corr[k]).abs() < 1e-12,
                "k={k}: {} vs {}",
                plain[k],
                corr[k]
            );
        }

        // Dense W as the symmetric PD square root of W² (Denman–Beavers
        // iteration, independent of the closed form under test).
        let w2 = dense(&c.kkt_block(&s, &z), 4);
        let mul = |a: &[Vec<f64>], b: &[Vec<f64>]| -> Vec<Vec<f64>> {
            (0..4)
                .map(|i| {
                    (0..4)
                        .map(|j| (0..4).map(|k| a[i][k] * b[k][j]).sum())
                        .collect()
                })
                .collect()
        };
        let inv = |a: &[Vec<f64>]| -> Vec<Vec<f64>> {
            let mut m: Vec<Vec<f64>> = a.to_vec();
            let mut r: Vec<Vec<f64>> = (0..4)
                .map(|i| (0..4).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                .collect();
            for col in 0..4 {
                let p = (col..4)
                    .max_by(|&x, &y| m[x][col].abs().total_cmp(&m[y][col].abs()))
                    .unwrap();
                m.swap(col, p);
                r.swap(col, p);
                let d = m[col][col];
                for j in 0..4 {
                    m[col][j] /= d;
                    r[col][j] /= d;
                }
                for i in 0..4 {
                    if i != col {
                        let f = m[i][col];
                        for j in 0..4 {
                            m[i][j] -= f * m[col][j];
                            r[i][j] -= f * r[col][j];
                        }
                    }
                }
            }
            r
        };
        let (mut y, mut zz): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (
            w2.clone(),
            (0..4)
                .map(|i| (0..4).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                .collect(),
        );
        for _ in 0..50 {
            let (yi, zi) = (inv(&y), inv(&zz));
            let ny = (0..4)
                .map(|i| (0..4).map(|j| 0.5 * (y[i][j] + zi[i][j])).collect())
                .collect();
            let nz = (0..4)
                .map(|i| (0..4).map(|j| 0.5 * (zz[i][j] + yi[i][j])).collect())
                .collect();
            y = ny;
            zz = nz;
        }
        let w = y;
        let w_inv = inv(&w);
        let ww = mul(&w, &w);
        for i in 0..4 {
            for j in 0..4 {
                assert!((ww[i][j] - w2[i][j]).abs() < 1e-10);
            }
        }
        let lambda = matvec(&w, &z);
        let lambda_s = matvec(&w_inv, &s);
        for k in 0..4 {
            assert!((lambda[k] - lambda_s[k]).abs() < 1e-10, "W z ≠ W⁻¹ s");
        }

        let ds = [0.4, -0.1, 0.3, 0.2];
        let dz = [-0.2, 0.3, 0.1, -0.4];
        let sigma_mu = 0.3;
        let (a, b) = (matvec(&w_inv, &ds), matvec(&w, &dz));
        let (mut r_hat, mut second) = ([0.0; 4], [0.0; 4]);
        SecondOrderCone::jordan(&lambda, &lambda, &mut r_hat);
        SecondOrderCone::jordan(&a, &b, &mut second);
        for k in 0..4 {
            r_hat[k] += second[k];
        }
        r_hat[0] -= sigma_mu;
        let mut t = [0.0; 4];
        SecondOrderCone::arw_inv(&lambda, &r_hat, &mut t);
        let want = matvec(&w, &t);

        c.comp_residual_corrector(&s, &z, &ds, &dz, sigma_mu, &mut corr);
        let mut got = [0.0; 4];
        c.rhs_comp_term(&s, &z, &corr, &mut got);
        for k in 0..4 {
            assert!(
                (got[k] - want[k]).abs() < 1e-10,
                "k={k}: {} vs {}",
                got[k],
                want[k]
            );
        }
    }

    /// Reduced-system triple reduces to the orthant in 1-D: for `m = 1`,
    /// the block is `s/z`, the rhs term is `r/z`, and `recover_ds` is
    /// `−r/z − (s/z)dz`.
    #[test]
    fn one_dimensional_cone_matches_orthant() {
        let c = SecondOrderCone::new(1);
        let s = [2.0];
        let z = [5.0];
        match c.kkt_block(&s, &z) {
            ConeBlock::DiagPlusRank1 { diag, u } => {
                // 1-D: W²[0] = diag + u² = −η² + 2η² = η² = s/z.
                assert!((diag[0] + u[0] * u[0] - s[0] / z[0]).abs() < 1e-12);
            }
            _ => panic!(),
        }
        let r = [0.6];
        let mut term = [0.0];
        c.rhs_comp_term(&s, &z, &r, &mut term);
        assert!((term[0] - r[0] / z[0]).abs() < 1e-12);
        let dz = [0.4];
        let mut ds = [0.0];
        c.recover_ds(&s, &z, &r, &dz, &mut ds);
        assert!((ds[0] - (-r[0] / z[0] - (s[0] / z[0]) * dz[0])).abs() < 1e-12);
    }
}
