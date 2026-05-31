//! The exponential cone and its self-concordant barrier (Phase H5).
//!
//! The exponential cone is the first **non-symmetric** cone in
//! `pounce-convex` and the gateway to geometric programming, logistic
//! regression, entropy/`log-sum-exp`, and relative-entropy models — the
//! application surface that closes most of the gap with Clarabel.
//!
//! ## The cone
//!
//! In the Clarabel/MOSEK orientation,
//! ```text
//!   K_exp = cl { (x, y, z) : y·exp(x/y) ≤ z,  y > 0 }
//!         = { (x,y,z) : y·log(z/y) ≥ x, y>0, z>0 } ∪ { (x,0,z) : x≤0, z≥0 }.
//! ```
//! Its dual is
//! ```text
//!   K_exp* = cl { (u, v, w) : −u·exp(v/u) ≤ e·w,  u < 0 }.
//! ```
//!
//! ## The barrier
//!
//! The standard degree-3 logarithmically-homogeneous self-concordant
//! barrier (Nesterov) is, with `ψ = y·log(z/y) − x`,
//! ```text
//!   f(x, y, z) = −log(ψ) − log(y) − log(z),   on  ψ > 0, y > 0, z > 0.
//! ```
//! This module provides `f`, `∇f`, `∇²f`, and cone-membership tests. It is
//! deliberately **standalone** (not yet a [`crate::cones::Cone`]): the
//! non-symmetric driver path that consumes these oracles is the next step.
//! The math here is validated both against finite differences and against
//! the exact log-homogeneity identities (`⟨∇f,p⟩ = −3`, `∇²f·p = −∇f`,
//! `f(tp) = f(p) − 3 log t`).

use super::BarrierCone;

/// The 3-dimensional exponential cone `K_exp` and its degree-3 barrier.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExponentialCone;

impl ExponentialCone {
    pub fn new() -> Self {
        ExponentialCone
    }

    /// `ψ = y·log(z/y) − x`, the slack whose positivity (with `y, z > 0`)
    /// defines the open cone. Returns `NaN` if `y` or `z` is non-positive.
    #[inline]
    fn psi(point: &[f64]) -> f64 {
        let (x, y, z) = (point[0], point[1], point[2]);
        y * (z / y).ln() - x
    }
}

impl BarrierCone for ExponentialCone {
    fn barrier_degree(&self) -> f64 {
        3.0
    }

    fn barrier(&self, point: &[f64]) -> f64 {
        let (_, y, z) = (point[0], point[1], point[2]);
        if y <= 0.0 || z <= 0.0 {
            return f64::INFINITY;
        }
        let psi = Self::psi(point);
        if psi <= 0.0 {
            return f64::INFINITY;
        }
        -psi.ln() - y.ln() - z.ln()
    }

    fn barrier_grad(&self, point: &[f64], out: &mut [f64]) {
        let (_, y, z) = (point[0], point[1], point[2]);
        let psi = Self::psi(point);
        let a = (z / y).ln() - 1.0; // ∂ψ/∂y
        // g = −(1/ψ)∇ψ − (0, 1/y, 1/z),  ∇ψ = (−1, a, y/z).
        out[0] = 1.0 / psi;
        out[1] = -a / psi - 1.0 / y;
        out[2] = -(y / z) / psi - 1.0 / z;
    }

    fn barrier_hess_lower(&self, point: &[f64], out: &mut [f64]) {
        let (_, y, z) = (point[0], point[1], point[2]);
        let psi = Self::psi(point);
        let a = (z / y).ln() - 1.0; // ∂ψ/∂y
        let q = y / z; // ∂ψ/∂z
        let ip = 1.0 / psi;
        let ip2 = ip * ip;
        // H = (1/ψ²)∇ψ∇ψᵀ − (1/ψ)∇²ψ + diag(0, 1/y², 1/z²),
        // ∇ψ = (−1, a, q),  ∇²ψ = [[0,0,0],[0,−1/y,1/z],[0,1/z,−y/z²]].
        let h_xx = ip2;
        let h_yx = -a * ip2;
        let h_yy = a * a * ip2 + ip / y + 1.0 / (y * y);
        let h_zx = -q * ip2;
        let h_zy = a * q * ip2 - ip / z;
        let h_zz = q * q * ip2 + ip * y / (z * z) + 1.0 / (z * z);
        // Lower triangle row-major: (0,0);(1,0),(1,1);(2,0),(2,1),(2,2).
        out[0] = h_xx;
        out[1] = h_yx;
        out[2] = h_yy;
        out[3] = h_zx;
        out[4] = h_zy;
        out[5] = h_zz;
    }

    fn in_primal_cone(&self, point: &[f64], tol: f64) -> bool {
        let (_, y, z) = (point[0], point[1], point[2]);
        y > tol && z > tol && Self::psi(point) > tol * (1.0 + y.abs())
    }

    fn in_dual_cone(&self, point: &[f64], tol: f64) -> bool {
        // K_exp* = cl{ (u,v,w) : −u·exp(v/u) ≤ e·w, u<0 }; strict interior
        // uses ψ* = v − u·log(−u/w) > 0 with u<0, w>0  (the conjugate slack).
        let (u, v, w) = (point[0], point[1], point[2]);
        if -u <= tol || w <= tol {
            return false;
        }
        let psi_d = v - u * ((-u) / w).ln();
        psi_d > tol * (1.0 + u.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_hess(point: &[f64]) -> [[f64; 3]; 3] {
        let c = ExponentialCone;
        let mut l = [0.0; 6];
        c.barrier_hess_lower(point, &mut l);
        [
            [l[0], l[1], l[3]],
            [l[1], l[2], l[4]],
            [l[3], l[4], l[5]],
        ]
    }

    /// A handful of interior points (y, z > 0 and ψ > 0).
    fn interior_points() -> Vec<[f64; 3]> {
        vec![
            [0.0, 1.0, std::f64::consts::E], // ψ = 1
            [-1.0, 2.0, 3.0],
            [0.5, 1.5, 4.0],
            [-2.0, 0.7, 1.2],
        ]
    }

    #[test]
    fn membership() {
        let c = ExponentialCone;
        assert!(c.in_primal_cone(&[0.0, 1.0, std::f64::consts::E], 1e-9));
        assert!(c.in_primal_cone(&[-1.0, 2.0, 3.0], 1e-9));
        // y ≤ 0 or z ≤ 0 → outside.
        assert!(!c.in_primal_cone(&[0.0, -1.0, 2.0], 1e-9));
        assert!(!c.in_primal_cone(&[0.0, 1.0, -2.0], 1e-9));
        // ψ < 0: x too large.
        assert!(!c.in_primal_cone(&[5.0, 1.0, std::f64::consts::E], 1e-9));
        // Dual interior: u<0, w>0, ψ* > 0.
        assert!(c.in_dual_cone(&[-1.0, 1.0, 1.0], 1e-9));
        assert!(!c.in_dual_cone(&[1.0, 1.0, 1.0], 1e-9)); // u>0
    }

    #[test]
    fn grad_matches_finite_difference() {
        let c = ExponentialCone;
        let h = 1e-6;
        for p in interior_points() {
            let mut g = [0.0; 3];
            c.barrier_grad(&p, &mut g);
            for k in 0..3 {
                let mut pp = p;
                let mut pm = p;
                pp[k] += h;
                pm[k] -= h;
                let fd = (c.barrier(&pp) - c.barrier(&pm)) / (2.0 * h);
                assert!(
                    (g[k] - fd).abs() < 1e-5,
                    "grad[{k}] at {p:?}: analytic {} vs fd {}",
                    g[k],
                    fd
                );
            }
        }
    }

    #[test]
    fn hess_matches_finite_difference() {
        let c = ExponentialCone;
        let h = 1e-6;
        for p in interior_points() {
            let hess = full_hess(&p);
            for j in 0..3 {
                // FD of the gradient's j-th component.
                let mut pp = p;
                let mut pm = p;
                pp[j] += h;
                pm[j] -= h;
                let mut gp = [0.0; 3];
                let mut gm = [0.0; 3];
                c.barrier_grad(&pp, &mut gp);
                c.barrier_grad(&pm, &mut gm);
                for i in 0..3 {
                    let fd = (gp[i] - gm[i]) / (2.0 * h);
                    assert!(
                        (hess[i][j] - fd).abs() < 1e-4,
                        "H[{i}][{j}] at {p:?}: analytic {} vs fd {}",
                        hess[i][j],
                        fd
                    );
                }
            }
        }
    }

    /// Log-homogeneity of degree ν = 3: f(t·p) = f(p) − 3·log t.
    #[test]
    fn log_homogeneous_degree_three() {
        let c = ExponentialCone;
        for p in interior_points() {
            for &t in &[0.5_f64, 2.0, 3.7] {
                let tp = [t * p[0], t * p[1], t * p[2]];
                let lhs = c.barrier(&tp);
                let rhs = c.barrier(&p) - 3.0 * t.ln();
                assert!((lhs - rhs).abs() < 1e-9, "f(tp)={lhs} vs {rhs}");
            }
        }
    }

    /// Euler identity for a degree-ν log-homogeneous barrier: ⟨∇f(p), p⟩ = −ν.
    #[test]
    fn euler_identity() {
        let c = ExponentialCone;
        for p in interior_points() {
            let mut g = [0.0; 3];
            c.barrier_grad(&p, &mut g);
            let dot = g[0] * p[0] + g[1] * p[1] + g[2] * p[2];
            assert!((dot + 3.0).abs() < 1e-9, "<g,p> = {dot}, expected −3");
        }
    }

    /// Hessian/gradient identity for log-homogeneous barriers: ∇²f(p)·p = −∇f(p).
    #[test]
    fn hessian_times_point_is_neg_grad() {
        let c = ExponentialCone;
        for p in interior_points() {
            let mut g = [0.0; 3];
            c.barrier_grad(&p, &mut g);
            let hess = full_hess(&p);
            for i in 0..3 {
                let hp = hess[i][0] * p[0] + hess[i][1] * p[1] + hess[i][2] * p[2];
                assert!(
                    (hp + g[i]).abs() < 1e-9,
                    "(Hp)[{i}] = {hp} vs −g = {}",
                    -g[i]
                );
            }
        }
    }
}
