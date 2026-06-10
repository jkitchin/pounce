//! Positive-semidefinite (PSD) cone primitives — Phase H7 foundation.
//!
//! The PSD cone `Sⁿ₊ = { X = Xᵀ ∈ ℝⁿˣⁿ : X ⪰ 0 }` is a **self-scaled**
//! (symmetric) cone, like the orthant and the second-order cone, so it
//! carries a Nesterov–Todd scaling. This module supplies the building
//! blocks the conic IPM needs, all in the symmetric-vectorization (`svec`)
//! coordinates the solver's slack/dual vectors live in:
//!
//! - [`svec`] / [`smat`] — the isometry between a symmetric `n×n` matrix and
//!   `ℝᵐ`, `m = n(n+1)/2`, with off-diagonals scaled by `√2` so that
//!   `⟨X, Y⟩_F = svec(X)·svec(Y)`.
//! - The log-det barrier `F(X) = −log det X`, its gradient `−X⁻¹`, and the
//!   Hessian action `D ↦ X⁻¹ D X⁻¹`.
//! - Membership / fraction-to-boundary via the eigenvalues of `X`.
//! - The **Nesterov–Todd scaling** `W` (symmetric PD, `W Z W = S`), the
//!   matrix the dense `(z,z)` KKT block `W ⊗ₛ W` is built from (driver
//!   integration is Phase H7's next step).
//!
//! Eigendecompositions reuse [`pounce_linalg::symmetric_eigen`] (the
//! cyclic-Jacobi solver shared with the NLP sensitivity path).

use super::{Cone, ConeBlock};
use pounce_linalg::symmetric_eigen;

/// The PSD cone over symmetric `n×n` matrices. Its slack/dual vectors have
/// dimension `n(n+1)/2` in [`svec`] coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PsdCone {
    pub n: usize,
}

impl PsdCone {
    pub fn new(n: usize) -> Self {
        PsdCone { n }
    }

    /// Length of the `svec` vectors this cone owns, `n(n+1)/2`.
    pub fn dim(&self) -> usize {
        self.n * (self.n + 1) / 2
    }

    /// Barrier degree `ν` of `−log det` over `Sⁿ₊` — equal to `n`.
    pub fn degree(&self) -> usize {
        self.n
    }
}

/// `svec` ordering: lower triangle, column by column — `(0,0),(1,0),…,
/// (n−1,0),(1,1),(2,1),…`. Off-diagonals carry a `√2` so the map is an
/// isometry (`‖X‖_F = ‖svec(X)‖₂`). `mat` is row-major `n×n` (symmetric).
pub fn svec(mat: &[f64], n: usize, out: &mut [f64]) {
    let r2 = std::f64::consts::SQRT_2;
    let mut k = 0;
    for j in 0..n {
        for i in j..n {
            out[k] = if i == j {
                mat[i * n + i]
            } else {
                r2 * mat[i * n + j]
            };
            k += 1;
        }
    }
}

/// The `svec` index of the lower-triangle entry `(i, j)` (`i ≥ j`) for an
/// `n×n` matrix, matching [`svec`]'s column-by-column lower-triangle order.
pub fn svec_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i >= j && i < n);
    j * n - j * (j.wrapping_sub(1)) / 2 + (i - j)
}

/// Inverse of [`svec`]: rebuild the symmetric `n×n` matrix (row-major) from
/// its `svec`, dividing off-diagonals by `√2`.
pub fn smat(v: &[f64], n: usize, out: &mut [f64]) {
    let inv_r2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut k = 0;
    for j in 0..n {
        for i in j..n {
            let val = if i == j { v[k] } else { inv_r2 * v[k] };
            out[i * n + j] = val;
            out[j * n + i] = val;
            k += 1;
        }
    }
}

// ---- small dense symmetric-matrix helpers (row-major, modest n) ----

/// `c = a · b` for row-major `n×n` matrices.
fn matmul(a: &[f64], b: &[f64], n: usize, c: &mut [f64]) {
    for i in 0..n {
        for k in 0..n {
            let mut acc = 0.0;
            for j in 0..n {
                acc += a[i * n + j] * b[j * n + k];
            }
            c[i * n + k] = acc;
        }
    }
}

/// Symmetric matrix function `f(A) = Q diag(f(λ)) Qᵀ` for a symmetric `A`
/// (row-major). Returns `None` if the eigensolver fails to converge.
fn sym_apply(a: &[f64], n: usize, f: impl Fn(f64) -> f64) -> Option<Vec<f64>> {
    let mut vals = vec![0.0; n];
    let mut vecs = vec![0.0; n * n];
    if !symmetric_eigen(a, n, &mut vals, &mut vecs) {
        return None;
    }
    // vecs is column-major: eigenvector j has component i at vecs[j*n + i].
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let mut acc = 0.0;
            for j in 0..n {
                acc += f(vals[j]) * vecs[j * n + i] * vecs[j * n + k];
            }
            out[i * n + k] = acc;
        }
    }
    Some(out)
}

impl PsdCone {
    /// The cone identity `e = svec(Iₙ)` — the well-centered cold-start point.
    pub fn identity(&self, out: &mut [f64]) {
        let n = self.n;
        let mut k = 0;
        for j in 0..n {
            for i in j..n {
                out[k] = if i == j { 1.0 } else { 0.0 };
                k += 1;
            }
        }
    }

    /// Smallest eigenvalue of `smat(point)` — `> 0` iff strictly interior.
    pub fn min_eig(&self, point: &[f64]) -> f64 {
        let n = self.n;
        let mut m = vec![0.0; n * n];
        smat(point, n, &mut m);
        let mut vals = vec![0.0; n];
        let mut vecs = vec![0.0; n * n];
        if !symmetric_eigen(&m, n, &mut vals, &mut vecs) {
            return f64::NEG_INFINITY;
        }
        vals[0] // ascending
    }

    /// Whether `smat(point) ⪰ tol·I`.
    pub fn in_cone(&self, point: &[f64], tol: f64) -> bool {
        self.min_eig(point) > tol
    }

    /// The log-det barrier `F = −log det smat(point)` (`+∞` outside the cone).
    pub fn barrier(&self, point: &[f64]) -> f64 {
        let n = self.n;
        let mut m = vec![0.0; n * n];
        smat(point, n, &mut m);
        let mut vals = vec![0.0; n];
        let mut vecs = vec![0.0; n * n];
        if !symmetric_eigen(&m, n, &mut vals, &mut vecs) {
            return f64::INFINITY;
        }
        let mut acc = 0.0;
        for &l in &vals {
            if l <= 0.0 {
                return f64::INFINITY;
            }
            acc += l.ln();
        }
        -acc
    }

    /// Gradient of the barrier, `∇F = −svec(X⁻¹)` (`X = smat(point)`).
    // The eig of a correctly-sized symmetric matrix at a strictly-interior
    // (PD) point always converges, so `sym_apply` cannot return `None` here.
    #[allow(clippy::expect_used)]
    pub fn barrier_grad(&self, point: &[f64], out: &mut [f64]) {
        let n = self.n;
        let mut m = vec![0.0; n * n];
        smat(point, n, &mut m);
        let inv = sym_apply(&m, n, |l| 1.0 / l).expect("barrier_grad: eig failed");
        // out = −svec(X⁻¹).
        svec(&inv, n, out);
        for v in out.iter_mut() {
            *v = -*v;
        }
    }

    /// Hessian action `H[d] = svec(X⁻¹ · smat(d) · X⁻¹)` — the operator
    /// `∇²F(point)` applied to a direction `d` (both in `svec` coordinates).
    // See `barrier_grad`: the interior-point eig always converges.
    #[allow(clippy::expect_used)]
    pub fn barrier_hess_apply(&self, point: &[f64], dir: &[f64], out: &mut [f64]) {
        let n = self.n;
        let mut x = vec![0.0; n * n];
        smat(point, n, &mut x);
        let xinv = sym_apply(&x, n, |l| 1.0 / l).expect("hess: eig failed");
        let mut d = vec![0.0; n * n];
        smat(dir, n, &mut d);
        let mut tmp = vec![0.0; n * n];
        let mut res = vec![0.0; n * n];
        matmul(&xinv, &d, n, &mut tmp); // X⁻¹ D
        matmul(&tmp, &xinv, n, &mut res); // X⁻¹ D X⁻¹
        svec(&res, n, out);
    }

    /// Largest `α ∈ (0, tau]` with `smat(v) + α·smat(dv) ⪰ 0`, scaled by the
    /// fraction-to-boundary parameter `tau`. Computes the most-negative
    /// eigenvalue of `L⁻¹ smat(dv) L⁻ᵀ` where `smat(v) = L Lᵀ` (here via the
    /// symmetric form `V^{-1/2} smat(dv) V^{-1/2}`, `V = smat(v) ≻ 0`).
    pub fn max_step(&self, v: &[f64], dv: &[f64], tau: f64) -> f64 {
        let n = self.n;
        let mut vmat = vec![0.0; n * n];
        smat(v, n, &mut vmat);
        let vinv_half = match sym_apply(&vmat, n, |l| 1.0 / l.max(1e-300).sqrt()) {
            Some(m) => m,
            None => return tau, // can't scale; let the caller's safeguard handle it
        };
        let mut dmat = vec![0.0; n * n];
        smat(dv, n, &mut dmat);
        // M = V^{-1/2} dV V^{-1/2}  (symmetric).
        let mut tmp = vec![0.0; n * n];
        let mut mmat = vec![0.0; n * n];
        matmul(&vinv_half, &dmat, n, &mut tmp);
        matmul(&tmp, &vinv_half, n, &mut mmat);
        let mut vals = vec![0.0; n];
        let mut vecs = vec![0.0; n * n];
        if !symmetric_eigen(&mmat, n, &mut vals, &mut vecs) {
            return tau;
        }
        let min_eig = vals[0]; // ascending
        if min_eig >= 0.0 {
            1.0 // direction keeps PSD for all α ⇒ full step
        } else {
            (tau * (-1.0 / min_eig)).min(1.0)
        }
    }

    /// The Nesterov–Todd scaling matrix `W` (symmetric PD) for the
    /// primal/dual interior pair `(s, z)` (both `svec` of PD matrices):
    /// `W = S^{1/2} (S^{1/2} Z S^{1/2})^{-1/2} S^{1/2}`, which satisfies the
    /// defining identity `W Z W = S`. Returned as a row-major `n×n` matrix.
    /// The dense `(z,z)` KKT scaling block is the symmetric Kronecker
    /// product `W ⊗ₛ W` built from this (Phase H7 driver integration).
    pub fn nt_scaling(&self, s: &[f64], z: &[f64]) -> Option<Vec<f64>> {
        let n = self.n;
        let mut smat_s = vec![0.0; n * n];
        let mut smat_z = vec![0.0; n * n];
        smat(s, n, &mut smat_s);
        smat(z, n, &mut smat_z);
        let s_half = sym_apply(&smat_s, n, |l| l.max(0.0).sqrt())?;
        // M = S^{1/2} Z S^{1/2}.
        let mut tmp = vec![0.0; n * n];
        let mut m = vec![0.0; n * n];
        matmul(&s_half, &smat_z, n, &mut tmp);
        matmul(&tmp, &s_half, n, &mut m);
        let m_inv_half = sym_apply(&m, n, |l| 1.0 / l.max(1e-300).sqrt())?;
        // W = S^{1/2} M^{-1/2} S^{1/2}.
        let mut tmp2 = vec![0.0; n * n];
        let mut w = vec![0.0; n * n];
        matmul(&s_half, &m_inv_half, n, &mut tmp2);
        matmul(&tmp2, &s_half, n, &mut w);
        Some(w)
    }
}

impl PsdCone {
    /// Jordan product `S ∘ Z = (SZ + ZS)/2`, in `svec` coordinates.
    fn jordan(&self, s: &[f64], z: &[f64], out: &mut [f64]) {
        let n = self.n;
        let (mut sm, mut zm) = (vec![0.0; n * n], vec![0.0; n * n]);
        smat(s, n, &mut sm);
        smat(z, n, &mut zm);
        let (mut sz, mut zs) = (vec![0.0; n * n], vec![0.0; n * n]);
        matmul(&sm, &zm, n, &mut sz);
        matmul(&zm, &sm, n, &mut zs);
        let mut j = vec![0.0; n * n];
        for i in 0..n * n {
            j[i] = 0.5 * (sz[i] + zs[i]);
        }
        svec(&j, n, out);
    }

    /// Apply the NT scaling operator `W ⊗ₛ W` to a direction `d`:
    /// `out = svec(W · smat(d) · W)` (`w` is the row-major `n×n` scaling).
    fn apply_scaling(&self, w: &[f64], d: &[f64], out: &mut [f64]) {
        let n = self.n;
        let mut dm = vec![0.0; n * n];
        smat(d, n, &mut dm);
        let (mut tmp, mut res) = (vec![0.0; n * n], vec![0.0; n * n]);
        matmul(w, &dm, n, &mut tmp);
        matmul(&tmp, w, n, &mut res);
        svec(&res, n, out);
    }

    /// Solve the Jordan system `z ∘ D = R` — i.e. the Lyapunov equation
    /// `Z D + D Z = 2·smat(r)` — for symmetric `D`, returning `svec(D)`.
    /// This is `Arw(z)⁻¹ r` for the PSD cone. Via `Z = QΛQᵀ`:
    /// `D = Q [ (Qᵀ(2R)Q)_{ij} / (λᵢ+λⱼ) ] Qᵀ`.
    #[allow(clippy::expect_used)]
    fn lyapunov_solve(&self, z: &[f64], r: &[f64], out: &mut [f64]) {
        let n = self.n;
        let mut zm = vec![0.0; n * n];
        smat(z, n, &mut zm);
        let mut vals = vec![0.0; n];
        let mut q = vec![0.0; n * n]; // column-major eigenvectors
        assert!(
            symmetric_eigen(&zm, n, &mut vals, &mut q),
            "lyapunov: eig failed"
        );
        let mut rm = vec![0.0; n * n];
        smat(r, n, &mut rm);
        // `q` is column-major (q[c*n + i] = Q[i][c]), so reading `q` as
        // row-major IS Qᵀ; transpose it once to get Q row-major. Then the two
        // congruences below are plain matmuls — O(n³) total, not the O(n⁴)
        // quadruple loops this replaced (M23).
        let mut q_rm = vec![0.0; n * n]; // Q row-major: q_rm[i*n+c] = Q[i][c].
        for c in 0..n {
            for i in 0..n {
                q_rm[i * n + c] = q[c * n + i];
            }
        }
        // R̃ = Qᵀ R Q  =  (q · R) · Q_rm.
        let mut tmp = vec![0.0; n * n];
        let mut rtilde = vec![0.0; n * n];
        matmul(&q, &rm, n, &mut tmp);
        matmul(&tmp, &q_rm, n, &mut rtilde);
        // D̃_{ab} = 2 R̃_{ab} / (λ_a + λ_b).
        let mut dtilde = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                dtilde[a * n + b] = 2.0 * rtilde[a * n + b] / (vals[a] + vals[b]);
            }
        }
        // D = Q D̃ Qᵀ  =  (Q_rm · D̃) · q.
        let mut dm = vec![0.0; n * n];
        matmul(&q_rm, &dtilde, n, &mut tmp);
        matmul(&tmp, &q, n, &mut dm);
        svec(&dm, n, out);
    }
}

impl Cone for PsdCone {
    fn degree(&self) -> usize {
        self.n
    }

    fn identity(&self, out: &mut [f64]) {
        PsdCone::identity(self, out);
    }

    fn dim(&self) -> usize {
        PsdCone::dim(self)
    }

    fn mu(&self, s: &[f64], z: &[f64]) -> f64 {
        // ⟨s, z⟩ = svec(S)·svec(Z) = tr(SZ); μ = ⟨s,z⟩ / degree.
        let dot: f64 = s.iter().zip(z).map(|(a, b)| a * b).sum();
        dot / self.n as f64
    }

    fn in_dual_cone(&self, z: &[f64], tol: f64) -> bool {
        // Self-dual: z ∈ K iff λ_min(smat z) ≥ −tol.
        self.min_eig(z) >= -tol
    }

    fn scaling_diag(&self, _s: &[f64], _z: &[f64], _out: &mut [f64]) {
        unimplemented!("PSD uses kkt_block (dense), not scaling_diag")
    }

    fn comp_residual(&self, s: &[f64], z: &[f64], sigma_mu: f64, out: &mut [f64]) {
        // s ∘ z − σμ·svec(I).
        self.jordan(s, z, out);
        let mut e = vec![0.0; self.dim()];
        PsdCone::identity(self, &mut e);
        for k in 0..self.dim() {
            out[k] -= sigma_mu * e[k];
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
        // s∘z + ds_aff∘dz_aff − σμ·svec(I).
        self.jordan(s, z, out);
        let mut second = vec![0.0; self.dim()];
        self.jordan(ds_aff, dz_aff, &mut second);
        let mut e = vec![0.0; self.dim()];
        PsdCone::identity(self, &mut e);
        for k in 0..self.dim() {
            out[k] += second[k] - sigma_mu * e[k];
        }
    }

    // The NT scaling always succeeds at strictly-interior (PD) iterates.
    #[allow(clippy::expect_used)]
    fn recover_ds(&self, s: &[f64], z: &[f64], r_comp: &[f64], dz: &[f64], ds: &mut [f64]) {
        // ds = −Arw(z)⁻¹ r_comp − (W⊗ₛW) dz, consistent with `kkt_block`
        // (the scaling operator) and `rhs_comp_term` (the Lyapunov solve).
        let m = self.dim();
        let mut inv = vec![0.0; m];
        self.lyapunov_solve(z, r_comp, &mut inv);
        let w = self.nt_scaling(s, z).expect("recover_ds: NT scaling");
        let mut hdz = vec![0.0; m];
        self.apply_scaling(&w, dz, &mut hdz);
        for k in 0..m {
            ds[k] = -inv[k] - hdz[k];
        }
    }

    #[allow(clippy::expect_used)]
    fn kkt_block(&self, s: &[f64], z: &[f64]) -> ConeBlock {
        // The (z,z) block is the symmetric Kronecker H = W ⊗ₛ W, an m×m SPD
        // matrix with H·svec(z) = svec(WZW) = svec(s). Build its lower triangle
        // (row-major) directly from a closed form — O(n⁴) total — rather than
        // applying the scaling operator to every unit vector, which costs two
        // O(n³) matmuls per column for O(n²) columns = O(n⁵) (M23).
        //
        // Column b ↔ svec basis vector e_b ↔ the lower-triangle pair (p,q),
        // p ≥ q, for which `smat(e_b)` is E_pp (if p=q) or (E_pq+E_qp)/√2
        // (if p>q). With D := W·smat(e_b)·W (= what `apply_scaling` returns,
        // before the svec scaling), W symmetric gives
        //   p = q:  D_ij = W_ip W_jp
        //   p > q:  D_ij = (W_ip W_jq + W_iq W_jp) / √2
        // and H[a][b] = (i=j ? 1 : √2)·D_ij for the output pair (i,j), i ≥ j.
        let n = self.n;
        let m = self.dim();
        let w = self.nt_scaling(s, z).expect("kkt_block: NT scaling");
        let r2 = std::f64::consts::SQRT_2;
        let inv_r2 = std::f64::consts::FRAC_1_SQRT_2;

        // svec-order lower-triangle pairs (i,j), i ≥ j: column by column.
        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(m);
        for j in 0..n {
            for i in j..n {
                pairs.push((i, j));
            }
        }

        let mut lower = Vec::with_capacity(m * (m + 1) / 2);
        for a in 0..m {
            let (i, j) = pairs[a];
            let row_scale = if i == j { 1.0 } else { r2 };
            for b in 0..=a {
                let (p, q) = pairs[b];
                let d = if p == q {
                    w[i * n + p] * w[j * n + p]
                } else {
                    inv_r2 * (w[i * n + p] * w[j * n + q] + w[i * n + q] * w[j * n + p])
                };
                lower.push(row_scale * d);
            }
        }
        ConeBlock::DenseLower { dim: m, lower }
    }

    fn rhs_comp_term(&self, _s: &[f64], z: &[f64], r_comp: &[f64], out: &mut [f64]) {
        // Arw(z)⁻¹ r_comp — the Lyapunov solve Z D + D Z = 2·smat(r_comp).
        self.lyapunov_solve(z, r_comp, out);
    }

    fn recenter_warm(&self, s: &mut [f64], z: &mut [f64], floor: f64) {
        // Like the SOC: a converged PSD point sits on the boundary (a zero
        // eigenvalue), where the NT scaling is singular. Re-center each block
        // to a well-conditioned multiple of the identity c·I (so S∘Z = c²I),
        // preserving magnitude; the warm benefit comes from the primal x.
        let n = self.n;
        let center = |u: &mut [f64]| {
            let mag = u
                .iter()
                .fold(0.0_f64, |m, &v| m.max(v.abs()))
                .max(floor)
                .max(1.0);
            let mut e = vec![0.0; u.len()];
            PsdCone { n }.identity(&mut e);
            for k in 0..u.len() {
                u[k] = mag * e[k];
            }
        };
        center(s);
        center(z);
    }

    fn max_step(&self, v: &[f64], dv: &[f64], tau: f64) -> f64 {
        PsdCone::max_step(self, v, dv, tau)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul_v(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut c = vec![0.0; n * n];
        matmul(a, b, n, &mut c);
        c
    }

    #[test]
    fn svec_smat_roundtrip_and_isometry() {
        let n = 3;
        // A symmetric matrix (row-major).
        let x = vec![
            2.0, 0.5, -1.0, //
            0.5, 3.0, 0.25, //
            -1.0, 0.25, 1.5,
        ];
        let m = n * (n + 1) / 2;
        let mut v = vec![0.0; m];
        svec(&x, n, &mut v);
        let mut back = vec![0.0; n * n];
        smat(&v, n, &mut back);
        for i in 0..n * n {
            assert!((x[i] - back[i]).abs() < 1e-12, "roundtrip at {i}");
        }
        // Isometry: ⟨X,X⟩_F = ‖svec‖².
        let fro: f64 = x.iter().map(|a| a * a).sum();
        let sv: f64 = v.iter().map(|a| a * a).sum();
        assert!((fro - sv).abs() < 1e-12, "isometry {fro} vs {sv}");
    }

    #[test]
    fn inner_product_preserved() {
        let n = 2;
        let x = vec![1.0, 2.0, 2.0, 3.0];
        let y = vec![0.5, -1.0, -1.0, 4.0];
        let fro: f64 = (0..n * n).map(|i| x[i] * y[i]).sum();
        let m = n * (n + 1) / 2;
        let (mut xv, mut yv) = (vec![0.0; m], vec![0.0; m]);
        svec(&x, n, &mut xv);
        svec(&y, n, &mut yv);
        let dot: f64 = (0..m).map(|i| xv[i] * yv[i]).sum();
        assert!((fro - dot).abs() < 1e-12, "{fro} vs {dot}");
    }

    #[test]
    fn identity_is_in_cone_and_barrier_zero() {
        let c = PsdCone::new(3);
        let mut e = vec![0.0; c.dim()];
        c.identity(&mut e);
        assert!(c.in_cone(&e, 1e-9));
        assert!((c.barrier(&e) - 0.0).abs() < 1e-12); // −log det I = 0
        assert!((c.min_eig(&e) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn barrier_grad_matches_finite_difference() {
        let c = PsdCone::new(2);
        // X = [[2, 0.3],[0.3, 1.5]] ≻ 0.
        let point = {
            let x = vec![2.0, 0.3, 0.3, 1.5];
            let mut v = vec![0.0; c.dim()];
            svec(&x, 2, &mut v);
            v
        };
        let mut g = vec![0.0; c.dim()];
        c.barrier_grad(&point, &mut g);
        let h = 1e-6;
        for k in 0..c.dim() {
            let mut pp = point.clone();
            let mut pm = point.clone();
            pp[k] += h;
            pm[k] -= h;
            let fd = (c.barrier(&pp) - c.barrier(&pm)) / (2.0 * h);
            assert!((g[k] - fd).abs() < 1e-5, "grad[{k}] {} vs fd {fd}", g[k]);
        }
    }

    #[test]
    fn nt_scaling_satisfies_w_z_w_equals_s() {
        let c = PsdCone::new(3);
        // Two distinct PD matrices in svec coords.
        let to_v = |x: &[f64]| {
            let mut v = vec![0.0; c.dim()];
            svec(x, 3, &mut v);
            v
        };
        let smat_s = vec![
            4.0, 1.0, 0.0, //
            1.0, 3.0, 0.5, //
            0.0, 0.5, 2.0,
        ];
        let smat_z = vec![
            2.0, -0.3, 0.2, //
            -0.3, 1.0, 0.1, //
            0.2, 0.1, 1.5,
        ];
        let s = to_v(&smat_s);
        let z = to_v(&smat_z);
        let w = c.nt_scaling(&s, &z).expect("nt scaling");
        // Check W Z W = S.
        let wz = matmul_v(&w, &smat_z, 3);
        let wzw = matmul_v(&wz, &w, 3);
        for i in 0..9 {
            assert!(
                (wzw[i] - smat_s[i]).abs() < 1e-8,
                "W Z W ≠ S at {i}: {} vs {}",
                wzw[i],
                smat_s[i]
            );
        }
        // W is symmetric.
        for i in 0..3 {
            for j in 0..3 {
                assert!((w[i * 3 + j] - w[j * 3 + i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn max_step_lands_on_the_boundary() {
        let c = PsdCone::new(2);
        // v = I; dv = −I ⇒ I − α I ⪰ 0 needs α ≤ 1; with τ=1, step = 1.
        let mut v = vec![0.0; c.dim()];
        c.identity(&mut v);
        let mut dv = vec![0.0; c.dim()];
        c.identity(&mut dv);
        for x in dv.iter_mut() {
            *x = -*x;
        }
        let a = c.max_step(&v, &dv, 1.0);
        assert!((a - 1.0).abs() < 1e-9, "step {a}");
        // At α just below 1 the point is still PD; with τ = 0.99, step ≈ 0.99.
        let a2 = c.max_step(&v, &dv, 0.99);
        assert!((a2 - 0.99).abs() < 1e-9, "step {a2}");
    }

    #[test]
    fn max_step_full_when_direction_keeps_psd() {
        let c = PsdCone::new(2);
        let mut v = vec![0.0; c.dim()];
        c.identity(&mut v);
        // dv = +I ⇒ stays PD for all α ⇒ capped at 1.
        let mut dv = vec![0.0; c.dim()];
        c.identity(&mut dv);
        assert!((c.max_step(&v, &dv, 0.99) - 1.0).abs() < 1e-9);
    }

    fn to_v(c: &PsdCone, x: &[f64]) -> Vec<f64> {
        let mut v = vec![0.0; c.dim()];
        svec(x, c.n, &mut v);
        v
    }

    fn dense_lower_to_full(block: &ConeBlock) -> (usize, Vec<f64>) {
        match block {
            ConeBlock::DenseLower { dim, lower } => {
                let m = *dim;
                let mut full = vec![0.0; m * m];
                let mut k = 0;
                for a in 0..m {
                    for b in 0..=a {
                        full[a * m + b] = lower[k];
                        full[b * m + a] = lower[k];
                        k += 1;
                    }
                }
                (m, full)
            }
            _ => panic!("expected DenseLower"),
        }
    }

    // Build a deterministic PD matrix (row-major n×n) in svec coords: strongly
    // diagonally dominant so it is PD, with off-diagonal structure.
    fn pd_v(c: &PsdCone, scale: f64) -> Vec<f64> {
        let n = c.n;
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] = if i == j {
                    n as f64 + scale
                } else {
                    scale / (1.0 + (i as f64 - j as f64).abs())
                };
            }
        }
        to_v(c, &m)
    }

    /// The closed-form `kkt_block` (M23) must reproduce — entry for entry —
    /// the reference built by applying the NT scaling operator `W ⊗ₛ W` to
    /// every svec unit vector (the previous O(n⁵) construction). Checked over
    /// a range of sizes, including off-diagonal-heavy blocks.
    #[test]
    fn kkt_block_matches_apply_scaling_reference() {
        use crate::cones::Cone;
        for &n in &[1usize, 2, 3, 5, 8] {
            let c = PsdCone::new(n);
            let s = pd_v(&c, 1.3);
            let z = pd_v(&c, 0.7);
            let m = c.dim();
            let w = c.nt_scaling(&s, &z).expect("nt scaling");
            // Reference: column b of H = W⊗ₛW applied to the unit vector e_b.
            let mut e = vec![0.0; m];
            let mut col = vec![0.0; m];
            let mut href = vec![0.0; m * m];
            for b in 0..m {
                e[b] = 1.0;
                c.apply_scaling(&w, &e, &mut col);
                for (a, &v) in col.iter().enumerate() {
                    href[a * m + b] = v;
                }
                e[b] = 0.0;
            }
            let (md, h) = dense_lower_to_full(&c.kkt_block(&s, &z));
            assert_eq!(md, m);
            for a in 0..m {
                for b in 0..m {
                    assert!(
                        (h[a * m + b] - href[a * m + b]).abs() < 1e-9,
                        "n={n} [{a}][{b}]: {} vs {}",
                        h[a * m + b],
                        href[a * m + b]
                    );
                }
            }
        }
    }

    /// The defining NT property of the `(z,z)` block: `H·svec(z) = svec(s)`.
    #[test]
    fn kkt_block_maps_z_to_s() {
        use crate::cones::Cone;
        let c = PsdCone::new(3);
        let s = to_v(&c, &[4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0]);
        let z = to_v(&c, &[2.0, -0.3, 0.2, -0.3, 1.0, 0.1, 0.2, 0.1, 1.5]);
        let (m, h) = dense_lower_to_full(&c.kkt_block(&s, &z));
        for a in 0..m {
            let acc: f64 = (0..m).map(|b| h[a * m + b] * z[b]).sum();
            assert!((acc - s[a]).abs() < 1e-7, "row {a}: {acc} vs {}", s[a]);
        }
    }

    /// `rhs_comp_term` = `Arw(z)⁻¹ r`, so `z ∘ (Arw(z)⁻¹ r) = r`.
    #[test]
    fn lyapunov_inverts_jordan() {
        use crate::cones::Cone;
        let c = PsdCone::new(3);
        let z = to_v(&c, &[2.0, -0.3, 0.2, -0.3, 1.0, 0.1, 0.2, 0.1, 1.5]);
        let r = to_v(&c, &[0.5, 0.1, -0.2, 0.1, 0.3, 0.05, -0.2, 0.05, 0.4]);
        let mut d = vec![0.0; c.dim()];
        c.rhs_comp_term(&z, &z, &r, &mut d);
        let mut zd = vec![0.0; c.dim()];
        c.jordan(&z, &d, &mut zd);
        for k in 0..c.dim() {
            assert!((zd[k] - r[k]).abs() < 1e-9, "{k}: {} vs {}", zd[k], r[k]);
        }
    }

    /// At `s = z = e`, `s∘z = I` and the centered residual is `(1−σμ)·e`.
    #[test]
    fn comp_residual_at_identity() {
        use crate::cones::Cone;
        let c = PsdCone::new(2);
        let mut e = vec![0.0; c.dim()];
        c.identity(&mut e);
        let mut out = vec![0.0; c.dim()];
        Cone::comp_residual(&c, &e, &e, 0.3, &mut out);
        for k in 0..c.dim() {
            assert!((out[k] - 0.7 * e[k]).abs() < 1e-12, "{k}");
        }
    }

    /// `recover_ds` is consistent with the assembled block and rhs term:
    /// it must reproduce `−Arw(z)⁻¹ r − H·dz`.
    #[test]
    fn recover_ds_matches_block_and_rhs() {
        use crate::cones::Cone;
        let c = PsdCone::new(3);
        let s = to_v(&c, &[4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0]);
        let z = to_v(&c, &[2.0, -0.3, 0.2, -0.3, 1.0, 0.1, 0.2, 0.1, 1.5]);
        let r = to_v(&c, &[0.5, 0.1, -0.2, 0.1, 0.3, 0.05, -0.2, 0.05, 0.4]);
        let dz = to_v(&c, &[0.2, 0.0, 0.1, 0.0, -0.1, 0.05, 0.1, 0.05, 0.3]);
        let mut ds = vec![0.0; c.dim()];
        c.recover_ds(&s, &z, &r, &dz, &mut ds);
        // Reference: −rhs_comp_term − H·dz.
        let mut rhs = vec![0.0; c.dim()];
        c.rhs_comp_term(&s, &z, &r, &mut rhs);
        let (m, h) = dense_lower_to_full(&c.kkt_block(&s, &z));
        for a in 0..m {
            let hdz: f64 = (0..m).map(|b| h[a * m + b] * dz[b]).sum();
            assert!((ds[a] - (-rhs[a] - hdz)).abs() < 1e-9, "row {a}");
        }
    }
}
