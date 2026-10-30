//! Second-order-cone arithmetic for cone slacks in the NLP interior-point
//! method (workstream C, `dev-notes/research/nlp-soc-design.md`).
//!
//! A cone `K = { u : u₀ ≥ ‖u_{1..}‖ }` is declared over free constraint rows;
//! their slacks `s` carry it, and the cone dual is `z = −y_d` on those rows.
//! This module is the per-block arithmetic the IPM needs, with the
//! conventions the design note fixes and the tests below pin:
//!
//! | quantity | value |
//! |---|---|
//! | central path | `s ∘ z = μ e`, `e = (1, 0, …, 0)`, so `⟨s, z⟩ = μ` |
//! | barrier term | `B(s) = −(μ/2)·log det s`, `det s = s₀² − ‖s_{1..}‖²` |
//! | barrier gradient | `∇B = −μ s⁻¹`, `s⁻¹ = J s / det s` |
//! | scaling block | `H` with `H s = z`, symmetric positive definite (NT) |
//!
//! `H` is the cone's analogue of the bound slacks' `Σ = V/S`: it is the
//! `(s, s)` block of the augmented system, and on the central path it equals
//! `μ s⁻²`, the barrier Hessian (`∇²B = μ s⁻²` in the Jordan sense).
//!
//! The Nesterov–Todd point is the one `pounce-convex`'s second-order cone
//! uses (`cones/soc.rs`): `w̄ = (s̄ + J z̄)/(2γ)` with `s̄, z̄` normalized to
//! unit determinant. There `W² = η²(2 w̄ w̄ᵀ − J)` maps `z` to `s` (the
//! orthant's `s/z`); the NLP's block is its inverse, `W⁻²`.

use pounce_common::types::Number;

/// `det u = u₀² − ‖u_{1..}‖²`. Positive in the interior.
pub fn det(u: &[Number]) -> Number {
    u[0] * u[0] - u[1..].iter().map(|v| v * v).sum::<Number>()
}

/// Strictly inside the cone: `u₀ > 0` and `det u > 0`.
pub fn in_interior(u: &[Number]) -> bool {
    !u.is_empty() && u[0] > 0.0 && det(u) > 0.0
}

/// Jordan product `s ∘ z = (⟨s, z⟩, s₀ z_{1..} + z₀ s_{1..})`.
pub fn jordan(s: &[Number], z: &[Number], out: &mut [Number]) {
    out[0] = s.iter().zip(z).map(|(a, b)| a * b).sum();
    for k in 1..s.len() {
        out[k] = s[0] * z[k] + z[0] * s[k];
    }
}

/// `s⁻¹ = J s / det s` (`s ∘ s⁻¹ = e`). Requires `s` in the interior.
pub fn inverse(s: &[Number], out: &mut [Number]) {
    let d = det(s);
    out[0] = s[0] / d;
    for k in 1..s.len() {
        out[k] = -s[k] / d;
    }
}

/// The barrier term `−(μ/2) log det s`; `+∞` outside the interior.
pub fn barrier(s: &[Number], mu: Number) -> Number {
    if !in_interior(s) {
        return Number::INFINITY;
    }
    -0.5 * mu * det(s).ln()
}

/// The barrier gradient `−μ s⁻¹`, written to `out`.
pub fn barrier_gradient(s: &[Number], mu: Number, out: &mut [Number]) {
    inverse(s, out);
    for v in out.iter_mut() {
        *v *= -mu;
    }
}

/// The Nesterov–Todd scaling block `H` (row-major, `m × m`) with `H s = z`:
/// `H = η⁻²(2 (J w̄)(J w̄)ᵀ − J)`, `η² = √det s / √det z`. Symmetric positive
/// definite for `s, z` in the interior. `None` if either is not.
pub fn nt_block(s: &[Number], z: &[Number]) -> Option<Vec<Number>> {
    if !(in_interior(s) && in_interior(z)) {
        return None;
    }
    let m = s.len();
    let sd = det(s).sqrt();
    let zd = det(z).sqrt();
    let sz: Number = s.iter().zip(z).map(|(a, b)| (a / sd) * (b / zd)).sum();
    let gamma = ((1.0 + sz) / 2.0).sqrt();
    // J w̄ = (w̄₀, −w̄_{1..}), w̄ = (s̄ + J z̄)/(2γ).
    let mut jw = vec![0.0; m];
    jw[0] = (s[0] / sd + z[0] / zd) / (2.0 * gamma);
    for k in 1..m {
        jw[k] = -(s[k] / sd - z[k] / zd) / (2.0 * gamma);
    }
    let inv_eta2 = zd / sd;
    let mut h = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            let jij = if i != j {
                0.0
            } else if i == 0 {
                1.0
            } else {
                -1.0
            };
            h[i * m + j] = inv_eta2 * (2.0 * jw[i] * jw[j] - jij);
        }
    }
    h.iter().all(|v| v.is_finite()).then_some(h)
}

/// Largest `α ∈ (0, 1]` with `v + α dv` in the cone, times `tau`, capped at
/// 1 — the cone's fraction-to-boundary rule. Same root as
/// `pounce_convex`'s `SecondOrderCone::max_step`.
pub fn max_step(v: &[Number], dv: &[Number], tau: Number) -> Number {
    use pounce_convex::cones::{Cone, SecondOrderCone};
    SecondOrderCone::new(v.len()).max_step(v, dv, tau)
}

/// Move `u` into the interior in place, so that `u₀ − ‖u_{1..}‖ ≥ margin`:
/// only `u₀` moves. The initialization's analogue of pushing a bounded slack
/// off its bound.
pub fn push_into_interior(u: &mut [Number], margin: Number) {
    let tail = u[1..].iter().map(|v| v * v).sum::<Number>().sqrt();
    if u[0] < tail + margin {
        u[0] = tail + margin;
    }
}

/// `‖s ∘ z − μ e‖∞`: how far the pair is from the central path at `μ`
/// (`μ = 0` for the complementarity error itself).
pub fn complementarity_error(s: &[Number], z: &[Number], mu: Number) -> Number {
    let mut sz = vec![0.0; s.len()];
    jordan(s, z, &mut sz);
    sz[0] -= mu;
    sz.iter().fold(0.0, |a: Number, v| a.max(v.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matvec(h: &[Number], v: &[Number]) -> Vec<Number> {
        let m = v.len();
        (0..m)
            .map(|i| (0..m).map(|j| h[i * m + j] * v[j]).sum())
            .collect()
    }

    fn close(a: &[Number], b: &[Number], tol: Number) -> bool {
        a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
    }

    const S: [Number; 3] = [2.0, 0.6, -0.8];
    const Z: [Number; 3] = [1.5, -0.3, 0.5];

    #[test]
    fn inverse_is_the_jordan_inverse() {
        let mut inv = [0.0; 3];
        inverse(&S, &mut inv);
        let mut e = [0.0; 3];
        jordan(&S, &inv, &mut e);
        assert!(close(&e, &[1.0, 0.0, 0.0], 1e-14), "{e:?}");
    }

    #[test]
    fn nt_block_maps_s_to_z_and_is_spd() {
        let h = nt_block(&S, &Z).unwrap();
        assert!(close(&matvec(&h, &S), &Z, 1e-13), "{:?}", matvec(&h, &S));
        let m = 3;
        for i in 0..m {
            for j in 0..m {
                assert!((h[i * m + j] - h[j * m + i]).abs() < 1e-14);
            }
        }
        // Positive definite: leading principal minors.
        let m1 = h[0];
        let m2 = h[0] * h[4] - h[1] * h[3];
        let m3 = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6])
            + h[2] * (h[3] * h[7] - h[4] * h[6]);
        assert!(m1 > 0.0 && m2 > 0.0 && m3 > 0.0, "{m1} {m2} {m3}");
    }

    /// On the central path (`z = μ s⁻¹`) the NT block is the barrier Hessian
    /// `μ s⁻²`, i.e. the `(s, s)` block a primal barrier would assemble —
    /// checked against the finite-difference Hessian of `B`.
    #[test]
    fn on_the_central_path_the_block_is_the_barrier_hessian() {
        let mu = 0.37;
        let mut z = [0.0; 3];
        inverse(&S, &mut z);
        for v in &mut z {
            *v *= mu;
        }
        let h = nt_block(&S, &z).unwrap();
        let eps = 1e-5;
        let mut fd = vec![0.0; 9];
        for j in 0..3 {
            let (mut sp, mut sm) = (S, S);
            sp[j] += eps;
            sm[j] -= eps;
            let (mut gp, mut gm) = ([0.0; 3], [0.0; 3]);
            barrier_gradient(&sp, mu, &mut gp);
            barrier_gradient(&sm, mu, &mut gm);
            for i in 0..3 {
                fd[i * 3 + j] = (gp[i] - gm[i]) / (2.0 * eps);
            }
        }
        assert!(close(&h, &fd, 1e-8), "{h:?}\n{fd:?}");
    }

    #[test]
    fn central_path_conventions() {
        let mu = 0.37;
        // z = μ s⁻¹ is on the path: s ∘ z = μ e and ⟨s, z⟩ = μ.
        let mut z = [0.0; 3];
        inverse(&S, &mut z);
        for v in &mut z {
            *v *= mu;
        }
        assert!(complementarity_error(&S, &z, mu) < 1e-14);
        let dot: Number = S.iter().zip(&z).map(|(a, b)| a * b).sum();
        assert!((dot - mu).abs() < 1e-14);
        // ∇B = −z there: the barrier gradient is the negative dual.
        let mut g = [0.0; 3];
        barrier_gradient(&S, mu, &mut g);
        assert!(close(&g, &[-z[0], -z[1], -z[2]], 1e-14));
        // and it is the derivative of B.
        let eps = 1e-6;
        for j in 0..3 {
            let (mut sp, mut sm) = (S, S);
            sp[j] += eps;
            sm[j] -= eps;
            let fd = (barrier(&sp, mu) - barrier(&sm, mu)) / (2.0 * eps);
            assert!((fd - g[j]).abs() < 1e-8, "{j}: {fd} vs {}", g[j]);
        }
    }

    /// A one-dimensional cone is a nonnegative slack: the block is `z/s`,
    /// the barrier `−(μ/2) log s²  = −μ log s`, the bound slack's.
    #[test]
    fn a_one_dimensional_cone_is_a_bound() {
        let h = nt_block(&[3.0], &[0.5]).unwrap();
        assert!((h[0] - 0.5 / 3.0).abs() < 1e-15);
        assert!((barrier(&[3.0], 0.2) + 0.2 * (3.0f64).ln()).abs() < 1e-15);
    }

    #[test]
    fn max_step_stops_at_the_boundary() {
        let v = [1.0, 0.0, 0.0];
        let dv = [-1.0, 1.0, 0.0];
        // det(v + α dv) = (1−α)² − α² = 1 − 2α: boundary at α = 1/2.
        assert!((max_step(&v, &dv, 1.0) - 0.5).abs() < 1e-15);
        assert!((max_step(&v, &dv, 0.99) - 0.495).abs() < 1e-15);
        assert_eq!(max_step(&v, &[1.0, 0.0, 0.0], 0.99), 1.0);
    }

    #[test]
    fn push_into_interior_moves_only_the_head() {
        let mut u = [0.1, 3.0, 4.0];
        push_into_interior(&mut u, 1e-2);
        assert_eq!(&u[1..], &[3.0, 4.0]);
        assert!((u[0] - 5.01).abs() < 1e-15);
        assert!(in_interior(&u));
    }

    #[test]
    fn outside_the_interior_there_is_no_block_and_no_finite_barrier() {
        assert!(nt_block(&[1.0, 1.0], &[1.0, 0.0]).is_none());
        assert_eq!(barrier(&[1.0, 2.0], 1.0), Number::INFINITY);
    }
}
