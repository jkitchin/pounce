//! The 3-dimensional power cone and its self-concordant barrier (Phase H6).
//!
//! The power cone is the second **non-symmetric** cone in `pounce-convex`,
//! after the exponential cone. It generalizes the (rotated) second-order cone
//! and is the building block for `p`-norm constraints (`‖x‖_p ≤ t`), general
//! geometric-programming monomials, and more.
//!
//! ## The cone
//!
//! For a fixed parameter `α ∈ (0, 1)`,
//! ```text
//!   K_α = { (x, y, z) ∈ ℝ × ℝ₊² : |x| ≤ y^α · z^(1−α) }.
//! ```
//! `α = 1/2` is the rotated quadratic cone; for other `α` it is non-symmetric.
//! Its dual is
//! ```text
//!   K_α* = { (u, v, w) ∈ ℝ × ℝ₊² : |u| ≤ (v/α)^α · (w/(1−α))^(1−α) }.
//! ```
//!
//! ## The barrier
//!
//! The degree-3 logarithmically-homogeneous self-concordant barrier
//! (Chares 2009; Skajaa–Ye 2015), with `ψ = y^{2α} z^{2−2α} − x²`:
//! ```text
//!   F(x, y, z) = −log(ψ) − (1−α)·log y − α·log z,   on ψ > 0, y > 0, z > 0.
//! ```
//! It satisfies the exact log-homogeneity identities (`⟨∇F,p⟩ = −3`,
//! `∇²F·p = −∇F`, `F(tp) = F(p) − 3 log t`) used as validation invariants
//! alongside finite differences.

use super::BarrierCone;

/// The 3-dimensional power cone `K_α` and its degree-3 barrier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerCone {
    /// The exponent `α ∈ (0, 1)` (`y^α z^{1−α}`).
    pub alpha: f64,
}

impl PowerCone {
    /// Build a power cone with exponent `alpha ∈ (0, 1)`.
    pub fn new(alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "power-cone exponent must be in (0, 1), got {alpha}"
        );
        PowerCone { alpha }
    }

    /// `a = y^{2α} z^{2−2α}` — the homogeneous-degree-2 term whose excess over
    /// `x²` defines the cone.
    #[inline]
    fn a_term(&self, y: f64, z: f64) -> f64 {
        y.powf(2.0 * self.alpha) * z.powf(2.0 - 2.0 * self.alpha)
    }

    /// `ψ = y^{2α} z^{2−2α} − x²`, the slack whose positivity (with `y, z > 0`)
    /// defines the open cone.
    #[inline]
    fn psi(&self, p: &[f64]) -> f64 {
        self.a_term(p[1], p[2]) - p[0] * p[0]
    }
}

impl BarrierCone for PowerCone {
    fn barrier_degree(&self) -> f64 {
        3.0
    }

    fn barrier(&self, point: &[f64]) -> f64 {
        let (_, y, z) = (point[0], point[1], point[2]);
        if y <= 0.0 || z <= 0.0 {
            return f64::INFINITY;
        }
        let psi = self.psi(point);
        if psi <= 0.0 {
            return f64::INFINITY;
        }
        -psi.ln() - (1.0 - self.alpha) * y.ln() - self.alpha * z.ln()
    }

    fn barrier_grad(&self, point: &[f64], out: &mut [f64]) {
        let (al, om) = (self.alpha, 1.0 - self.alpha);
        let (x, y, z) = (point[0], point[1], point[2]);
        let a = self.a_term(y, z);
        let psi = a - x * x;
        // ∇ψ = (−2x, 2α·a/y, (2−2α)·a/z); ∇F = −∇ψ/ψ − (0, (1−α)/y, α/z).
        out[0] = 2.0 * x / psi;
        out[1] = -(2.0 * al * a / y) / psi - om / y;
        out[2] = -(2.0 * om * a / z) / psi - al / z;
    }

    fn barrier_hess_lower(&self, point: &[f64], out: &mut [f64]) {
        let (al, om) = (self.alpha, 1.0 - self.alpha);
        let (x, y, z) = (point[0], point[1], point[2]);
        let a = self.a_term(y, z);
        let psi = a - x * x;
        let ip = 1.0 / psi;
        let ip2 = ip * ip;
        // ∇ψ components.
        let p1 = -2.0 * x;
        let p2 = 2.0 * al * a / y;
        let p3 = 2.0 * om * a / z;
        // ∇²ψ components.
        let q11 = -2.0;
        let q22 = 2.0 * al * (2.0 * al - 1.0) * a / (y * y);
        let q23 = 4.0 * al * om * a / (y * z);
        let q33 = 2.0 * om * (1.0 - 2.0 * al) * a / (z * z);
        // H = (1/ψ²)∇ψ∇ψᵀ − (1/ψ)∇²ψ + diag(0, (1−α)/y², α/z²).
        // (∇²ψ has zero (1,·) and (2,·) cross terms with x.)
        let h_xx = p1 * p1 * ip2 - q11 * ip;
        let h_yx = p2 * p1 * ip2;
        let h_yy = p2 * p2 * ip2 - q22 * ip + om / (y * y);
        let h_zx = p3 * p1 * ip2;
        let h_zy = p3 * p2 * ip2 - q23 * ip;
        let h_zz = p3 * p3 * ip2 - q33 * ip + al / (z * z);
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
        y > tol && z > tol && self.psi(point) > tol * (1.0 + y.abs() + z.abs())
    }

    fn in_dual_cone(&self, point: &[f64], tol: f64) -> bool {
        // K_α* = { (u,v,w) : |u| ≤ (v/α)^α (w/(1−α))^(1−α), v,w > 0 }.
        let (al, om) = (self.alpha, 1.0 - self.alpha);
        let (u, v, w) = (point[0], point[1], point[2]);
        if v <= tol || w <= tol {
            return false;
        }
        let bound = (v / al).powf(al) * (w / om).powf(om);
        bound - u.abs() > tol * (1.0 + u.abs())
    }

    fn interior_reference(&self, out: &mut [f64]) {
        // (0, 1, 1) lies in int K_α (|0| < 1) and in int K_α* (for all
        // α ∈ (0,1) the dual bound `(1/α)^α (1/(1−α))^(1−α) > 0`), so it is a
        // valid self-dual start for any α.
        out[0] = 0.0;
        out[1] = 1.0;
        out[2] = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cones() -> Vec<PowerCone> {
        vec![
            PowerCone::new(0.5),
            PowerCone::new(0.3),
            PowerCone::new(0.75),
        ]
    }

    fn full_hess(c: &PowerCone, point: &[f64]) -> [[f64; 3]; 3] {
        let mut l = [0.0; 6];
        c.barrier_hess_lower(point, &mut l);
        [[l[0], l[1], l[3]], [l[1], l[2], l[4]], [l[3], l[4], l[5]]]
    }

    /// Interior points (y, z > 0 and ψ > 0) for each cone.
    fn interior_points() -> Vec<[f64; 3]> {
        vec![
            [0.0, 1.0, 1.0],
            [0.3, 2.0, 1.5],
            [-0.5, 1.2, 3.0],
            [0.1, 0.7, 0.9],
        ]
    }

    #[test]
    fn membership() {
        for c in cones() {
            // (0,1,1) is interior: |0| < 1.
            assert!(c.in_primal_cone(&[0.0, 1.0, 1.0], 1e-9));
            // On/over the boundary: |x| = y^α z^(1-α).
            let b = 1.0_f64.powf(c.alpha) * 1.0_f64.powf(1.0 - c.alpha);
            assert!(!c.in_primal_cone(&[b + 0.1, 1.0, 1.0], 1e-9));
            // y or z ≤ 0 → outside.
            assert!(!c.in_primal_cone(&[0.0, -1.0, 1.0], 1e-9));
            assert!(!c.in_primal_cone(&[0.0, 1.0, -1.0], 1e-9));
        }
    }

    #[test]
    fn dual_membership_via_conjugate_gradient() {
        // For interior `p`, `−∇F(p)` must lie in the dual cone `K_α*`.
        for c in cones() {
            for p in interior_points() {
                let mut g = [0.0; 3];
                c.barrier_grad(&p, &mut g);
                let d = [-g[0], -g[1], -g[2]];
                assert!(
                    c.in_dual_cone(&d, 1e-9),
                    "−∇F(p) must be dual-interior: α={} p={p:?} d={d:?}",
                    c.alpha
                );
            }
        }
    }

    #[test]
    fn grad_matches_finite_difference() {
        let h = 1e-6;
        for c in cones() {
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
                        "grad[{k}] α={} at {p:?}: analytic {} vs fd {}",
                        c.alpha,
                        g[k],
                        fd
                    );
                }
            }
        }
    }

    #[test]
    fn hess_matches_finite_difference() {
        let h = 1e-6;
        for c in cones() {
            for p in interior_points() {
                let hess = full_hess(&c, &p);
                for j in 0..3 {
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
                            "H[{i}][{j}] α={} at {p:?}: analytic {} vs fd {}",
                            c.alpha,
                            hess[i][j],
                            fd
                        );
                    }
                }
            }
        }
    }

    /// Log-homogeneity of degree ν = 3: F(t·p) = F(p) − 3·log t.
    #[test]
    fn log_homogeneous_degree_three() {
        for c in cones() {
            for p in interior_points() {
                for &t in &[0.5_f64, 2.0, 3.7] {
                    let tp = [t * p[0], t * p[1], t * p[2]];
                    let lhs = c.barrier(&tp);
                    let rhs = c.barrier(&p) - 3.0 * t.ln();
                    assert!((lhs - rhs).abs() < 1e-9, "F(tp)={lhs} vs {rhs}");
                }
            }
        }
    }

    /// Euler identity for a degree-ν log-homogeneous barrier: ⟨∇F(p), p⟩ = −ν.
    #[test]
    fn euler_identity() {
        for c in cones() {
            for p in interior_points() {
                let mut g = [0.0; 3];
                c.barrier_grad(&p, &mut g);
                let dot = g[0] * p[0] + g[1] * p[1] + g[2] * p[2];
                assert!((dot + 3.0).abs() < 1e-9, "<g,p> = {dot}, expected −3");
            }
        }
    }

    /// Hessian/gradient identity for log-homogeneous barriers: ∇²F(p)·p = −∇F(p).
    #[test]
    fn hessian_times_point_is_neg_grad() {
        for c in cones() {
            for p in interior_points() {
                let mut g = [0.0; 3];
                c.barrier_grad(&p, &mut g);
                let hess = full_hess(&c, &p);
                for i in 0..3 {
                    let hp = hess[i][0] * p[0] + hess[i][1] * p[1] + hess[i][2] * p[2];
                    assert!(
                        (hp + g[i]).abs() < 1e-9,
                        "(Hp)[{i}] = {hp} vs −g = {} (α={})",
                        -g[i],
                        c.alpha
                    );
                }
            }
        }
    }
}
