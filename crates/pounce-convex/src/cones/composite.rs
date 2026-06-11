//! Composite cone — a Cartesian product of cones over which the IPM keeps
//! one stacked slack `s` and dual `z`.
//!
//! The inequality block of a convex program is in general a product
//! `K = R₊^{n₀} × SOC(m₁) × …`. [`CompositeCone`] owns an ordered list of
//! `(offset, ConeKind)` blocks and implements [`Cone`] by dispatching every
//! operation block-wise over the matching slices of `s`/`z`. The IPM driver
//! holds a `CompositeCone` and stays cone-agnostic.
//!
//! Phase 1 of the SOCP extension (see `dev-notes/socp-extension.md`) ships
//! only a single nonnegative-orthant block, so this is bit-identical to the
//! previous bare [`NonnegCone`] path; the seam exists so SOC (and later
//! cones) plug in as new [`ConeKind`] variants without touching the driver.

use super::{Cone, ConeBlock, NonnegCone, PsdCone, SecondOrderCone};

/// Declarative description of one cone block in a problem's inequality
/// partition (the data form; [`ConeKind`] is the runtime form). The blocks
/// stack in order to cover the `m_ineq` inequality rows.
// `Eq` is intentionally not derived: `Power(f64)` carries a float exponent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConeSpec {
    /// Nonnegative orthant of the given number of rows.
    Nonneg(usize),
    /// Second-order cone of the given dimension (`≥ 1`).
    SecondOrder(usize),
    /// 3-dimensional exponential cone. **Non-symmetric** — a problem
    /// containing this routes to the non-symmetric HSDE driver
    /// ([`crate::hsde_nonsym`]), not the symmetric path; it is *not* a
    /// [`ConeKind`] and must be intercepted before [`CompositeCone`] assembly.
    Exponential,
    /// 3-dimensional power cone `K_α = {|x₁| ≤ x₂^α x₃^{1−α}}` with exponent
    /// `α ∈ (0, 1)`. **Non-symmetric** — routes to the non-symmetric HSDE
    /// driver like [`ConeSpec::Exponential`].
    Power(f64),
    /// Positive-semidefinite cone over symmetric `n×n` matrices (the stored
    /// `usize` is the matrix size `n`). Self-scaled, so it stays on the
    /// symmetric driver; it spans `n(n+1)/2` rows in `svec` coordinates.
    Psd(usize),
}

impl ConeSpec {
    /// Number of inequality rows this block spans.
    pub fn dim(&self) -> usize {
        match self {
            ConeSpec::Nonneg(n) | ConeSpec::SecondOrder(n) => *n,
            ConeSpec::Exponential | ConeSpec::Power(_) => 3,
            ConeSpec::Psd(n) => n * (n + 1) / 2,
        }
    }
}

/// A single cone in the product. A closed enum (rather than `dyn Cone`) so
/// dispatch is a cheap match and new cones are added as variants.
#[derive(Debug, Clone)]
pub enum ConeKind {
    /// Nonnegative orthant (LP/QP, and expanded variable bounds).
    Nonneg(NonnegCone),
    /// Second-order (Lorentz) cone.
    SecondOrder(SecondOrderCone),
    /// Positive-semidefinite cone (self-scaled; dense `W⊗ₛW` KKT block).
    Psd(PsdCone),
}

/// Dispatch a `Cone` call to whichever concrete cone this variant wraps.
macro_rules! dispatch {
    ($self:ident, $c:ident => $body:expr) => {
        match $self {
            ConeKind::Nonneg($c) => $body,
            ConeKind::SecondOrder($c) => $body,
            ConeKind::Psd($c) => $body,
        }
    };
}

impl Cone for ConeKind {
    fn degree(&self) -> usize {
        dispatch!(self, c => c.degree())
    }
    fn identity(&self, out: &mut [f64]) {
        dispatch!(self, c => c.identity(out))
    }
    fn dim(&self) -> usize {
        dispatch!(self, c => c.dim())
    }
    fn mu(&self, s: &[f64], z: &[f64]) -> f64 {
        dispatch!(self, c => c.mu(s, z))
    }
    fn scaling_diag(&self, s: &[f64], z: &[f64], out: &mut [f64]) {
        dispatch!(self, c => c.scaling_diag(s, z, out))
    }
    fn comp_residual(&self, s: &[f64], z: &[f64], sigma_mu: f64, out: &mut [f64]) {
        dispatch!(self, c => c.comp_residual(s, z, sigma_mu, out))
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
        dispatch!(self, c => c.comp_residual_corrector(s, z, ds_aff, dz_aff, sigma_mu, out))
    }
    fn recover_ds(&self, s: &[f64], z: &[f64], r_comp: &[f64], dz: &[f64], ds: &mut [f64]) {
        dispatch!(self, c => c.recover_ds(s, z, r_comp, dz, ds))
    }
    fn max_step(&self, v: &[f64], dv: &[f64], tau: f64) -> f64 {
        dispatch!(self, c => c.max_step(v, dv, tau))
    }
    fn kkt_block(&self, s: &[f64], z: &[f64]) -> ConeBlock {
        dispatch!(self, c => c.kkt_block(s, z))
    }
    fn rhs_comp_term(&self, s: &[f64], z: &[f64], r_comp: &[f64], out: &mut [f64]) {
        dispatch!(self, c => c.rhs_comp_term(s, z, r_comp, out))
    }
    fn recenter_warm(&self, s: &mut [f64], z: &mut [f64], floor: f64) {
        dispatch!(self, c => c.recenter_warm(s, z, floor))
    }
    fn in_dual_cone(&self, z: &[f64], tol: f64) -> bool {
        dispatch!(self, c => c.in_dual_cone(z, tol))
    }
}

/// A Cartesian product of cones, the cone of the IPM's stacked `(s, z)`.
#[derive(Debug, Clone)]
pub struct CompositeCone {
    /// `(offset, cone)` for each block; offsets partition `0..dim`.
    blocks: Vec<(usize, ConeKind)>,
    dim: usize,
    degree: usize,
}

impl CompositeCone {
    /// Build from an ordered list of cone blocks. Offsets are assigned by
    /// stacking the blocks in the given order.
    pub fn new(kinds: Vec<ConeKind>) -> Self {
        let mut blocks = Vec::with_capacity(kinds.len());
        let mut dim = 0;
        let mut degree = 0;
        for k in kinds {
            degree += k.degree();
            let d = k.dim();
            blocks.push((dim, k));
            dim += d;
        }
        CompositeCone {
            blocks,
            dim,
            degree,
        }
    }

    /// A single nonnegative-orthant block of dimension `n` — the cone of
    /// LP/QP (and the Phase-1 default for any inequality block).
    pub fn single_nonneg(n: usize) -> Self {
        Self::new(vec![ConeKind::Nonneg(NonnegCone::new(n))])
    }

    /// Build from a declarative [`ConeSpec`] partition of the inequality
    /// rows. An empty `specs` (or `m_ineq == 0`) yields an empty cone; the
    /// common LP/QP case is a single `Nonneg` spec.
    pub fn from_specs(specs: &[ConeSpec]) -> Self {
        let kinds = specs
            .iter()
            .map(|s| match s {
                ConeSpec::Nonneg(n) => ConeKind::Nonneg(NonnegCone::new(*n)),
                ConeSpec::SecondOrder(m) => ConeKind::SecondOrder(SecondOrderCone::new(*m)),
                ConeSpec::Psd(n) => ConeKind::Psd(PsdCone::new(*n)),
                ConeSpec::Exponential | ConeSpec::Power(_) => unreachable!(
                    "non-symmetric cones (exponential/power) must route to \
                     hsde_nonsym before CompositeCone assembly"
                ),
            })
            .collect();
        Self::new(kinds)
    }

    /// The `(offset, cone)` blocks, in row order. Used by the KKT assembly
    /// to place each block's scaling contribution (diagonal or dense).
    pub fn blocks(&self) -> &[(usize, ConeKind)] {
        &self.blocks
    }
}

impl Cone for CompositeCone {
    fn degree(&self) -> usize {
        self.degree
    }

    fn identity(&self, out: &mut [f64]) {
        for (off, k) in &self.blocks {
            let d = k.dim();
            k.identity(&mut out[*off..off + d]);
        }
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn mu(&self, s: &[f64], z: &[f64]) -> f64 {
        if self.degree == 0 {
            return 0.0;
        }
        // μ = ⟨s,z⟩_total / degree_total. Each block's μ is its own
        // ⟨s_b,z_b⟩ / degree_b, so block.mu · block.degree recovers the
        // block dot without a separate inner-product method.
        let mut dot = 0.0;
        for (off, k) in &self.blocks {
            let d = k.dim();
            dot += k.mu(&s[*off..off + d], &z[*off..off + d]) * k.degree() as f64;
        }
        dot / self.degree as f64
    }

    fn scaling_diag(&self, s: &[f64], z: &[f64], out: &mut [f64]) {
        for (off, k) in &self.blocks {
            let d = k.dim();
            k.scaling_diag(
                &s[*off..off + d],
                &z[*off..off + d],
                &mut out[*off..off + d],
            );
        }
    }

    fn comp_residual(&self, s: &[f64], z: &[f64], sigma_mu: f64, out: &mut [f64]) {
        for (off, k) in &self.blocks {
            let d = k.dim();
            k.comp_residual(
                &s[*off..off + d],
                &z[*off..off + d],
                sigma_mu,
                &mut out[*off..off + d],
            );
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
        for (off, k) in &self.blocks {
            let d = k.dim();
            k.comp_residual_corrector(
                &s[*off..off + d],
                &z[*off..off + d],
                &ds_aff[*off..off + d],
                &dz_aff[*off..off + d],
                sigma_mu,
                &mut out[*off..off + d],
            );
        }
    }

    fn recover_ds(&self, s: &[f64], z: &[f64], r_comp: &[f64], dz: &[f64], ds: &mut [f64]) {
        for (off, k) in &self.blocks {
            let d = k.dim();
            k.recover_ds(
                &s[*off..off + d],
                &z[*off..off + d],
                &r_comp[*off..off + d],
                &dz[*off..off + d],
                &mut ds[*off..off + d],
            );
        }
    }

    fn max_step(&self, v: &[f64], dv: &[f64], tau: f64) -> f64 {
        let mut alpha = 1.0_f64;
        for (off, k) in &self.blocks {
            let d = k.dim();
            alpha = alpha.min(k.max_step(&v[*off..off + d], &dv[*off..off + d], tau));
        }
        alpha
    }

    fn rhs_comp_term(&self, s: &[f64], z: &[f64], r_comp: &[f64], out: &mut [f64]) {
        for (off, k) in &self.blocks {
            let d = k.dim();
            k.rhs_comp_term(
                &s[*off..off + d],
                &z[*off..off + d],
                &r_comp[*off..off + d],
                &mut out[*off..off + d],
            );
        }
    }

    fn kkt_block(&self, _s: &[f64], _z: &[f64]) -> ConeBlock {
        // A product cone has *multiple* blocks; the KKT assembly iterates
        // `blocks()` and calls each block's `kkt_block` rather than asking
        // the composite for a single one.
        unimplemented!("use CompositeCone::blocks() for per-block kkt_block")
    }

    fn recenter_warm(&self, s: &mut [f64], z: &mut [f64], floor: f64) {
        for (off, k) in &self.blocks {
            let d = k.dim();
            k.recenter_warm(&mut s[*off..off + d], &mut z[*off..off + d], floor);
        }
    }

    fn in_dual_cone(&self, z: &[f64], tol: f64) -> bool {
        // The dual of a product cone is the product of the duals: every block
        // must lie in its own dual cone.
        self.blocks.iter().all(|(off, k)| {
            let d = k.dim();
            k.in_dual_cone(&z[*off..off + d], tol)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single-nonneg composite reproduces NonnegCone exactly.
    #[test]
    fn single_nonneg_matches_bare_orthant() {
        let n = 4;
        let comp = CompositeCone::single_nonneg(n);
        let bare = NonnegCone::new(n);
        let s = [1.0, 2.0, 0.5, 3.0];
        let z = [3.0, 1.0, 4.0, 0.5];

        assert_eq!(comp.dim(), n);
        assert_eq!(comp.degree(), n);
        assert!((comp.mu(&s, &z) - bare.mu(&s, &z)).abs() < 1e-15);

        let (mut a, mut b) = ([0.0; 4], [0.0; 4]);
        comp.scaling_diag(&s, &z, &mut a);
        bare.scaling_diag(&s, &z, &mut b);
        assert_eq!(a, b);

        comp.comp_residual(&s, &z, 0.7, &mut a);
        bare.comp_residual(&s, &z, 0.7, &mut b);
        assert_eq!(a, b);

        let dv = [-1.0, 0.5, -2.0, 1.0];
        assert!((comp.max_step(&s, &dv, 0.99) - bare.max_step(&s, &dv, 0.99)).abs() < 1e-15);
    }

    /// Two stacked nonneg blocks behave like one orthant of the total size
    /// (μ over the whole vector, min step over blocks). Guards the
    /// block-dispatch arithmetic that SOC will rely on.
    #[test]
    fn two_blocks_compose_like_one_orthant() {
        let comp = CompositeCone::new(vec![
            ConeKind::Nonneg(NonnegCone::new(2)),
            ConeKind::Nonneg(NonnegCone::new(3)),
        ]);
        let whole = NonnegCone::new(5);
        let s = [1.0, 2.0, 3.0, 0.5, 4.0];
        let z = [2.0, 1.0, 0.5, 4.0, 1.0];
        assert_eq!(comp.dim(), 5);
        assert_eq!(comp.degree(), 5);
        assert!((comp.mu(&s, &z) - whole.mu(&s, &z)).abs() < 1e-15);

        let dv = [-0.5, 1.0, -3.0, 0.2, -1.0];
        assert!((comp.max_step(&s, &dv, 0.95) - whole.max_step(&s, &dv, 0.95)).abs() < 1e-15);

        let (mut a, mut b) = ([0.0; 5], [0.0; 5]);
        comp.recover_ds(&s, &z, &[0.1, 0.2, 0.3, 0.4, 0.5], &dv, &mut a);
        whole.recover_ds(&s, &z, &[0.1, 0.2, 0.3, 0.4, 0.5], &dv, &mut b);
        for i in 0..5 {
            assert!((a[i] - b[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn empty_composite_is_inert() {
        let comp = CompositeCone::single_nonneg(0);
        assert_eq!(comp.dim(), 0);
        assert_eq!(comp.degree(), 0);
        assert_eq!(comp.mu(&[], &[]), 0.0);
        assert_eq!(comp.max_step(&[], &[], 0.99), 1.0);
    }
}
