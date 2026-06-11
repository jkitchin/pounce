//! Nonnegative-orthant cone — the cone of LP and convex QP.
//!
//! All operations are elementwise. This is the only cone implemented in
//! Phase 2; richer cones (SOC, PSD, exp, pow) plug in behind the same
//! [`Cone`](super::Cone) trait in later phases.

use super::{Cone, ConeBlock};

/// The nonnegative orthant `{ x : x_i ≥ 0 }` of a given dimension.
#[derive(Debug, Clone, Copy)]
pub struct NonnegCone {
    n: usize,
}

impl NonnegCone {
    pub fn new(n: usize) -> Self {
        NonnegCone { n }
    }
}

impl Cone for NonnegCone {
    fn degree(&self) -> usize {
        self.n
    }

    fn identity(&self, out: &mut [f64]) {
        out.iter_mut().for_each(|v| *v = 1.0);
    }

    fn dim(&self) -> usize {
        self.n
    }

    fn mu(&self, s: &[f64], z: &[f64]) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let dot: f64 = s.iter().zip(z).map(|(a, b)| a * b).sum();
        dot / self.n as f64
    }

    fn scaling_diag(&self, s: &[f64], z: &[f64], out: &mut [f64]) {
        for i in 0..self.n {
            out[i] = s[i] / z[i];
        }
    }

    fn comp_residual(&self, s: &[f64], z: &[f64], sigma_mu: f64, out: &mut [f64]) {
        for i in 0..self.n {
            out[i] = s[i] * z[i] - sigma_mu;
        }
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
        for i in 0..self.n {
            out[i] = s[i] * z[i] + ds_aff[i] * dz_aff[i] - sigma_mu;
        }
    }

    fn recover_ds(&self, s: &[f64], z: &[f64], r_comp: &[f64], dz: &[f64], ds: &mut [f64]) {
        for i in 0..self.n {
            ds[i] = -(r_comp[i] / z[i]) - (s[i] / z[i]) * dz[i];
        }
    }

    fn max_step(&self, v: &[f64], dv: &[f64], tau: f64) -> f64 {
        let mut alpha = 1.0_f64;
        for i in 0..self.n {
            if dv[i] < 0.0 {
                let a = -tau * v[i] / dv[i];
                if a < alpha {
                    alpha = a;
                }
            }
        }
        alpha
    }

    fn in_dual_cone(&self, z: &[f64], tol: f64) -> bool {
        // Self-dual: zᵢ ≥ −tol componentwise.
        z[..self.n].iter().all(|&zi| zi >= -tol)
    }

    fn kkt_block(&self, s: &[f64], z: &[f64]) -> ConeBlock {
        ConeBlock::Diagonal((0..self.n).map(|i| s[i] / z[i]).collect())
    }

    fn rhs_comp_term(&self, _s: &[f64], z: &[f64], r_comp: &[f64], out: &mut [f64]) {
        for i in 0..self.n {
            out[i] = r_comp[i] / z[i];
        }
    }

    fn recenter_warm(&self, s: &mut [f64], z: &mut [f64], floor: f64) {
        let n = self.n;
        // Positivity shift: lift s and z off the boundary by ≥ floor.
        let s_min = s.iter().cloned().fold(f64::INFINITY, f64::min);
        let z_min = z.iter().cloned().fold(f64::INFINITY, f64::min);
        let ds = (-1.5 * s_min).max(floor);
        let dz = (-1.5 * z_min).max(floor);
        for i in 0..n {
            s[i] += ds;
            z[i] += dz;
        }
        // Mehrotra centering shift to balance s and z.
        let sz: f64 = s.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
        let sum_s: f64 = s.iter().sum();
        let sum_z: f64 = z.iter().sum();
        let ds2 = 0.5 * sz / sum_z;
        let dz2 = 0.5 * sz / sum_s;
        for i in 0..n {
            s[i] += ds2;
            z[i] += dz2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mu_is_average_complementarity() {
        let c = NonnegCone::new(2);
        // ⟨s,z⟩ = 1*3 + 2*4 = 11, degree 2 → 5.5
        assert!((c.mu(&[1.0, 2.0], &[3.0, 4.0]) - 5.5).abs() < 1e-12);
    }

    #[test]
    fn max_step_caps_at_one_when_all_increasing() {
        let c = NonnegCone::new(2);
        assert!((c.max_step(&[1.0, 1.0], &[1.0, 0.5], 0.99) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn max_step_limited_by_most_negative_ratio() {
        let c = NonnegCone::new(1);
        // v=2, dv=-1, tau=1 → α = -(2)/(-1) = 2, but capped... here it is
        // the boundary at 2 so not capped below 1? -2*? recompute:
        // a = -tau*v/dv = -1*2/(-1) = 2 → α stays min(1,2)=... 2>1 so 1.
        assert!((c.max_step(&[2.0], &[-1.0], 1.0) - 1.0).abs() < 1e-12);
        // v=1, dv=-2, tau=1 → a = -1*1/(-2)=0.5 → α=0.5
        assert!((c.max_step(&[1.0], &[-2.0], 1.0) - 0.5).abs() < 1e-12);
    }
}
