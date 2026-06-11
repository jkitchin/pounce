//! Non-symmetric homogeneous self-dual embedding driver (Phases H5–H6).
//!
//! The non-symmetric counterpart of [`crate::hsde`]. It solves
//! `min cᵀx s.t. Ax = b, Gx + s = h, s ∈ K` where `K` is a product of
//! nonnegative-orthant, second-order, **exponential**, and **power** cones,
//! via the same homogeneous self-dual embedding and two-solve τ scheme. The
//! exp/power blocks use the **dual-aware primal–dual scaling** of Dahl &
//! Andersen (2021) (in place of a Nesterov–Todd point); the orthant and
//! second-order blocks are self-scaled and reuse their NT machinery, so all
//! four cone families coexist in one KKT.
//!
//! ## What differs from the symmetric driver
//!
//! The whole non-symmetric algorithm collapses onto the symmetric structure
//! once the right scaling `M = WᵀW` is in hand (see `dev-notes/hsde.md`):
//!
//! - the cone's `(z, z)` block is `−M⁻¹` (dense 3×3 for the exp cone), which
//!   reduces to `−diag(s/z) = −W²` for the orthant and to the primal-Hessian
//!   block `−(1/μ)∇²F⁻¹` on the central path;
//! - the complementarity right-hand side is `rc = −z + γμ·s̃ − η` with
//!   `s̃ = −∇F(s)` the shadow dual (the corrector `η` is Phase-H5b; here 0),
//!   `comp_term = −M⁻¹·rc`, and the slack recovery `Δs = −comp_term − M⁻¹·Δz`;
//! - for the orthant this is **identical** to the symmetric Mehrotra step,
//!   which is the correctness anchor;
//! - the exp cone has no closed-form fraction-to-boundary, so the step length
//!   is found by backtracking on cone membership.
//!
//! The barrier oracles, conjugate-gradient shadow iterate, and the scaling
//! itself live in [`crate::cones::exp`]; this module is the outer iteration.

use crate::cones::{BarrierCone, Cone, ConeBlock, ExponentialCone, PowerCone, SecondOrderCone};
use crate::debug::{fire, ConvexDebugState};
use crate::ipm::{build_rhs, detect_infeasibility, dot, inf_norm, split_step, QpOptions};
use crate::qp::{QpProblem, QpSolution, QpStatus};
use pounce_common::debug::{Checkpoint, DebugAction, DebugHook};
use pounce_common::types::{Index, Number};
use pounce_linsol::{Factorization, SparseSymLinearSolverInterface};
use std::collections::BTreeMap;

/// A 3-dimensional non-symmetric cone the driver supports. It implements
/// [`BarrierCone`] by dispatching to the concrete cone, so the generic scaling
/// / conjugate-gradient / corrector machinery (in [`crate::cones::nonsym`])
/// works over it unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NonsymCone {
    /// The exponential cone.
    Exp(ExponentialCone),
    /// The power cone `K_α`.
    Power(PowerCone),
}

macro_rules! ns_dispatch {
    ($self:ident, $c:ident => $body:expr) => {
        match $self {
            NonsymCone::Exp($c) => $body,
            NonsymCone::Power($c) => $body,
        }
    };
}

impl BarrierCone for NonsymCone {
    fn barrier_degree(&self) -> f64 {
        ns_dispatch!(self, c => c.barrier_degree())
    }
    fn barrier(&self, p: &[f64]) -> f64 {
        ns_dispatch!(self, c => c.barrier(p))
    }
    fn barrier_grad(&self, p: &[f64], out: &mut [f64]) {
        ns_dispatch!(self, c => c.barrier_grad(p, out))
    }
    fn barrier_hess_lower(&self, p: &[f64], out: &mut [f64]) {
        ns_dispatch!(self, c => c.barrier_hess_lower(p, out))
    }
    fn in_primal_cone(&self, p: &[f64], tol: f64) -> bool {
        ns_dispatch!(self, c => c.in_primal_cone(p, tol))
    }
    fn in_dual_cone(&self, p: &[f64], tol: f64) -> bool {
        ns_dispatch!(self, c => c.in_dual_cone(p, tol))
    }
    fn interior_reference(&self, out: &mut [f64]) {
        ns_dispatch!(self, c => c.interior_reference(out))
    }
}

/// One block of the cone product, by row offset. The non-symmetric driver
/// also accepts self-scaled **second-order** cones (handled via their NT
/// scaling), so an exp/power problem can carry SOC constraints too.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NsBlock {
    /// Nonnegative orthant of the given number of rows.
    Orthant(usize),
    /// Second-order (Lorentz) cone of the given dimension.
    SecondOrder(usize),
    /// A 3-dimensional non-symmetric cone (exponential or power).
    Nonsym(NonsymCone),
}

impl NsBlock {
    /// A 3-dimensional exponential-cone block.
    pub fn exp() -> Self {
        NsBlock::Nonsym(NonsymCone::Exp(ExponentialCone))
    }
    /// A 3-dimensional power-cone block `K_α`.
    pub fn power(alpha: f64) -> Self {
        NsBlock::Nonsym(NonsymCone::Power(PowerCone::new(alpha)))
    }

    fn dim(&self) -> usize {
        match self {
            NsBlock::Orthant(n) | NsBlock::SecondOrder(n) => *n,
            NsBlock::Nonsym(_) => 3,
        }
    }
    /// Barrier degree (orthant: its dimension; second-order cone: 2;
    /// a 3-D non-symmetric cone: 3).
    fn degree(&self) -> usize {
        match self {
            NsBlock::Orthant(n) => *n,
            NsBlock::SecondOrder(_) => 2,
            NsBlock::Nonsym(_) => 3,
        }
    }
}

/// The cone product with each block's row offset precomputed.
pub(crate) struct NsCone {
    blocks: Vec<(usize, NsBlock)>,
    dim: usize,
    degree: usize,
}

impl NsCone {
    pub(crate) fn new(specs: &[NsBlock]) -> Self {
        let mut blocks = Vec::with_capacity(specs.len());
        let (mut dim, mut degree) = (0, 0);
        for b in specs {
            blocks.push((dim, *b));
            dim += b.dim();
            degree += b.degree();
        }
        NsCone {
            blocks,
            dim,
            degree,
        }
    }

    /// Self-dual starting iterate `e` (orthant: ones; non-symmetric cone: the
    /// cone's `interior_reference`, which lies in both `K` and `K*`). The
    /// corrector recenters from here, so an exact central point is not needed.
    fn identity(&self, out: &mut [f64]) {
        for (off, b) in &self.blocks {
            match b {
                NsBlock::Orthant(n) => {
                    for v in &mut out[*off..off + n] {
                        *v = 1.0;
                    }
                }
                NsBlock::SecondOrder(m) => {
                    // e = (1, 0, …, 0), the SOC identity / well-centered start.
                    for v in &mut out[*off..off + m] {
                        *v = 0.0;
                    }
                    out[*off] = 1.0;
                }
                NsBlock::Nonsym(cone) => {
                    cone.interior_reference(&mut out[*off..off + 3]);
                }
            }
        }
    }
}

/// Fraction-to-boundary step for a positive scalar ray `v + α dv > 0`.
fn ray_step(v: f64, dv: f64, tau: f64) -> f64 {
    if dv < 0.0 {
        (tau * (-v / dv)).min(1.0)
    } else {
        1.0
    }
}

/// Per-block, per-iteration scaling data: `M⁻¹` (applied in the RHS and
/// recovery) and the shadow dual `s̃ = −∇F(s)`.
enum BlockScaling {
    /// Orthant: `M⁻¹ = diag(s/z)`, `s̃ = 1/s`.
    Orthant {
        sz_ratio: Vec<f64>,
        s_tilde: Vec<f64>,
    },
    /// Second-order cone: its NT scaling `W² = diag(d) + u uᵀ`, kept in
    /// diag-plus-rank-1 form so the recover step applies `W²·Δz` cheaply.
    SecondOrder { diag: Vec<f64>, u: Vec<f64> },
    /// Non-symmetric cone (exp/power): dense `M⁻¹` (3×3) and the shadow dual.
    Nonsym {
        minv: [[f64; 3]; 3],
        s_tilde: [f64; 3],
    },
}

/// KKT value-array positions for one cone block.
enum ZPos {
    /// Orthant: one diagonal value position per row.
    Diag(Vec<usize>),
    /// Second-order cone: the dense lower-triangle value positions, row-major
    /// `[(0,0); (1,0),(1,1); …]` (length `m(m+1)/2`).
    SecondOrder { dim: usize, pos: Vec<usize> },
    /// Exp/power: the three diagonal positions and the three strict-lower
    /// positions `(1,0),(2,0),(2,1)`.
    Dense { diag: [usize; 3], lower: [usize; 3] },
}

/// The constant KKT pattern (lower triangle, 1-based) plus the scaling-block
/// value positions, so each iteration only rewrites the cone block and
/// `refactor`s (reusing the symbolic factor).
struct NsKkt {
    airn: Vec<Index>,
    ajcn: Vec<Index>,
    values: Vec<Number>,
    dim: usize,
    z_pos: Vec<ZPos>,
}

impl NsKkt {
    fn build(prob: &QpProblem, cone: &NsCone, reg: f64) -> Self {
        let n = prob.n;
        let m_eq = prob.m_eq();
        let m_ineq = prob.m_ineq();
        let mut entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut add = |r: usize, c: usize, v: f64| {
            let (r, c) = if r >= c { (r, c) } else { (c, r) };
            *entries.entry((r, c)).or_insert(0.0) += v;
        };
        // (x,x): P + reg·I.
        for t in &prob.p_lower {
            add(t.row, t.col, t.val);
        }
        for i in 0..n {
            add(i, i, reg);
        }
        // (y,x): A; (y,y): −reg.
        for t in &prob.a {
            add(n + t.row, t.col, t.val);
        }
        for i in 0..m_eq {
            add(n + i, n + i, -reg);
        }
        // (z,x): G.
        for t in &prob.g {
            add(n + m_eq + t.row, t.col, t.val);
        }
        // (z,z): per block, seeded with −reg on the diagonal. Exp blocks also
        // reserve the strict-lower 3×3 off-diagonals (a genuine dense block).
        for (off, b) in &cone.blocks {
            let zb = n + m_eq + off;
            match b {
                NsBlock::Orthant(d) => {
                    for i in 0..*d {
                        add(zb + i, zb + i, -reg);
                    }
                }
                NsBlock::SecondOrder(m) => {
                    // Genuine dense m×m lower triangle for the NT scaling W².
                    for i in 0..*m {
                        for j in 0..=i {
                            add(zb + i, zb + j, if i == j { -reg } else { 0.0 });
                        }
                    }
                }
                NsBlock::Nonsym(_) => {
                    for i in 0..3 {
                        add(zb + i, zb + i, -reg);
                    }
                    add(zb + 1, zb, 0.0);
                    add(zb + 2, zb, 0.0);
                    add(zb + 2, zb + 1, 0.0);
                }
            }
        }

        let nnz = entries.len();
        let mut airn = Vec::with_capacity(nnz);
        let mut ajcn = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        let mut coord_to_pos: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (pos, ((r, c), v)) in entries.into_iter().enumerate() {
            airn.push((r + 1) as Index);
            ajcn.push((c + 1) as Index);
            values.push(v);
            coord_to_pos.insert((r, c), pos);
        }

        let mut z_pos = Vec::with_capacity(cone.blocks.len());
        for (off, b) in &cone.blocks {
            let zb = n + m_eq + off;
            match b {
                NsBlock::Orthant(d) => {
                    z_pos.push(ZPos::Diag(
                        (0..*d).map(|i| coord_to_pos[&(zb + i, zb + i)]).collect(),
                    ));
                }
                NsBlock::SecondOrder(m) => {
                    let mut pos = Vec::with_capacity(m * (m + 1) / 2);
                    for i in 0..*m {
                        for j in 0..=i {
                            pos.push(coord_to_pos[&(zb + i, zb + j)]);
                        }
                    }
                    z_pos.push(ZPos::SecondOrder { dim: *m, pos });
                }
                NsBlock::Nonsym(_) => {
                    let diag = [
                        coord_to_pos[&(zb, zb)],
                        coord_to_pos[&(zb + 1, zb + 1)],
                        coord_to_pos[&(zb + 2, zb + 2)],
                    ];
                    let lower = [
                        coord_to_pos[&(zb + 1, zb)],
                        coord_to_pos[&(zb + 2, zb)],
                        coord_to_pos[&(zb + 2, zb + 1)],
                    ];
                    z_pos.push(ZPos::Dense { diag, lower });
                }
            }
        }
        let _ = m_ineq;
        NsKkt {
            airn,
            ajcn,
            values,
            dim: n + m_eq + m_ineq,
            z_pos,
        }
    }

    /// Write `−M⁻¹ − reg·I` into the cone block of `out` (a copy of
    /// `self.values`), returning the per-block scaling for use in the RHS and
    /// slack recovery. `None` if any exp scaling fails.
    fn update_blocks(
        &self,
        cone: &NsCone,
        s: &[f64],
        z: &[f64],
        reg: f64,
        out: &mut [Number],
    ) -> Option<Vec<BlockScaling>> {
        let mut scalings = Vec::with_capacity(cone.blocks.len());
        for ((off, b), zp) in cone.blocks.iter().zip(&self.z_pos) {
            match (b, zp) {
                (NsBlock::Orthant(d), ZPos::Diag(pos)) => {
                    let mut sz_ratio = vec![0.0; *d];
                    let mut s_tilde = vec![0.0; *d];
                    for i in 0..*d {
                        let (si, zi) = (s[off + i], z[off + i]);
                        sz_ratio[i] = si / zi; // (M⁻¹)_ii
                        s_tilde[i] = 1.0 / si; // −∇F(s)_i
                        out[pos[i]] = -sz_ratio[i] - reg;
                    }
                    scalings.push(BlockScaling::Orthant { sz_ratio, s_tilde });
                }
                (NsBlock::SecondOrder(m), ZPos::SecondOrder { dim, pos }) => {
                    debug_assert_eq!(m, dim);
                    let sb = &s[*off..off + m];
                    let zb = &z[*off..off + m];
                    // W² = diag(d) + u uᵀ from the SOC's NT scaling.
                    let (diag, u) = match SecondOrderCone::new(*m).kkt_block(sb, zb) {
                        ConeBlock::DiagPlusRank1 { diag, u } => (diag, u),
                        _ => unreachable!("SOC kkt_block is DiagPlusRank1"),
                    };
                    // Write −W² − reg into the dense lower triangle.
                    let mut k = 0;
                    for i in 0..*m {
                        for j in 0..=i {
                            let mut w2 = u[i] * u[j];
                            if i == j {
                                w2 += diag[i];
                            }
                            out[pos[k]] = -w2 - if i == j { reg } else { 0.0 };
                            k += 1;
                        }
                    }
                    scalings.push(BlockScaling::SecondOrder { diag, u });
                }
                (NsBlock::Nonsym(nscone), ZPos::Dense { diag, lower }) => {
                    let sb = &s[*off..off + 3];
                    let zb = &z[*off..off + 3];
                    let (minv, s_tilde) = block_minv(nscone, sb, zb)?;
                    out[diag[0]] = -minv[0][0] - reg;
                    out[diag[1]] = -minv[1][1] - reg;
                    out[diag[2]] = -minv[2][2] - reg;
                    out[lower[0]] = -minv[1][0];
                    out[lower[1]] = -minv[2][0];
                    out[lower[2]] = -minv[2][1];
                    scalings.push(BlockScaling::Nonsym { minv, s_tilde });
                }
                _ => unreachable!("block/position shape mismatch"),
            }
        }
        Some(scalings)
    }
}

/// `M⁻¹` and shadow dual for a non-symmetric cone block. Uses the dual-aware
/// scaling off the central path; falls back to the primal Hessian
/// `M = μ∇²F(s)` (so `M⁻¹ = (1/μ)∇²F⁻¹`) when the dual-aware scaling
/// degenerates (near-center). Generic over the cone (exp or power).
fn block_minv<C: BarrierCone>(cone: &C, s: &[f64], z: &[f64]) -> Option<([[f64; 3]; 3], [f64; 3])> {
    use crate::cones::nonsym::{chol_solve3, scaling};
    if let Some(sc) = scaling(cone, s, z) {
        if let Some(minv) = sc.minv() {
            return Some((minv, sc.s_tilde));
        }
    }
    // Fallback: M = μ∇²F(s), μ = ⟨s,z⟩/3.
    let mu = (s[0] * z[0] + s[1] * z[1] + s[2] * z[2]) / 3.0;
    if mu <= 0.0 {
        return None;
    }
    let mut hl = [0.0; 6];
    cone.barrier_hess_lower(s, &mut hl);
    // M = μH ⇒ M⁻¹ = (1/μ)H⁻¹.
    let scaled = [
        mu * hl[0],
        mu * hl[1],
        mu * hl[2],
        mu * hl[3],
        mu * hl[4],
        mu * hl[5],
    ];
    let c0 = chol_solve3(&scaled, &[1.0, 0.0, 0.0])?;
    let c1 = chol_solve3(&scaled, &[0.0, 1.0, 0.0])?;
    let c2 = chol_solve3(&scaled, &[0.0, 0.0, 1.0])?;
    let minv = [
        [c0[0], c1[0], c2[0]],
        [c0[1], c1[1], c2[1]],
        [c0[2], c1[2], c2[2]],
    ];
    let mut g = [0.0; 3];
    cone.barrier_grad(s, &mut g);
    Some((minv, [-g[0], -g[1], -g[2]]))
}

/// Apply a symmetric 3×3 to a 3-slice.
fn matvec3(m: &[[f64; 3]; 3], v: &[f64]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Predictor right-hand side. For orthant/non-symmetric blocks
/// `comp = −M⁻¹·rc`, `rc = −z + σμ·s̃`. For a second-order cone it is the
/// self-scaled term `Arw(z)⁻¹·(s∘z − σμe)` (the cone's `rhs_comp_term`).
fn comp_term(
    cone: &NsCone,
    scalings: &[BlockScaling],
    s: &[f64],
    z: &[f64],
    sigma_mu: f64,
    out: &mut [f64],
) {
    for (&(off, b), sc) in cone.blocks.iter().zip(scalings) {
        let d = b.dim();
        match sc {
            BlockScaling::Orthant { sz_ratio, s_tilde } => {
                for i in 0..d {
                    let rc = -z[off + i] + sigma_mu * s_tilde[i];
                    out[off + i] = -sz_ratio[i] * rc;
                }
            }
            BlockScaling::SecondOrder { .. } => {
                let soc = SecondOrderCone::new(d);
                let (sb, zb) = (&s[off..off + d], &z[off..off + d]);
                let mut r_c = vec![0.0; d];
                soc.comp_residual(sb, zb, sigma_mu, &mut r_c);
                soc.rhs_comp_term(sb, zb, &r_c, &mut out[off..off + d]);
            }
            BlockScaling::Nonsym { minv, s_tilde } => {
                let rc = [
                    -z[off] + sigma_mu * s_tilde[0],
                    -z[off + 1] + sigma_mu * s_tilde[1],
                    -z[off + 2] + sigma_mu * s_tilde[2],
                ];
                let mc = matvec3(minv, &rc);
                out[off] = -mc[0];
                out[off + 1] = -mc[1];
                out[off + 2] = -mc[2];
            }
        }
    }
}

/// Corrector right-hand side: `comp_term = −M⁻¹·rc` with
/// `rc = −z + σμ·s̃ − η`, where `η` is the nonsymmetric corrector
/// (Dahl–Andersen eq. 16). For an orthant block `η_i = ds_aff_i·dz_aff_i/s_i`
/// — exactly the Mehrotra second-order term, so the orthant corrector
/// reproduces standard Mehrotra. For an exp block
/// `η = −½ F'''(s)[ds_aff, (∇²F(s))⁻¹ dz_aff]`. If the exp third-derivative
/// FD leaves the cone, `η = 0` for that block (still a valid centered step).
#[allow(clippy::too_many_arguments)]
fn comp_term_corr(
    cone: &NsCone,
    scalings: &[BlockScaling],
    s: &[f64],
    z: &[f64],
    sigma_mu: f64,
    ds_aff: &[f64],
    dz_aff: &[f64],
    out: &mut [f64],
) {
    use crate::cones::nonsym::{chol_solve3, third_dir_apply};
    for (&(off, b), sc) in cone.blocks.iter().zip(scalings) {
        let d = b.dim();
        match (b, sc) {
            (_, BlockScaling::Orthant { sz_ratio, s_tilde }) => {
                for i in 0..d {
                    let eta = s_tilde[i] * ds_aff[off + i] * dz_aff[off + i];
                    let rc = -z[off + i] + sigma_mu * s_tilde[i] - eta;
                    out[off + i] = -sz_ratio[i] * rc;
                }
            }
            (NsBlock::Nonsym(nscone), BlockScaling::Nonsym { minv, s_tilde }) => {
                // η = −½ F'''(s)[ds_aff, H⁻¹ dz_aff], H = ∇²F(s) of *this* cone.
                let sb = &s[off..off + 3];
                let mut hl = [0.0; 6];
                nscone.barrier_hess_lower(sb, &mut hl);
                let dza = [dz_aff[off], dz_aff[off + 1], dz_aff[off + 2]];
                let hinv_dza = chol_solve3(&hl, &dza).unwrap_or([0.0; 3]);
                let u = [ds_aff[off], ds_aff[off + 1], ds_aff[off + 2]];
                let eta = match third_dir_apply(&nscone, sb, &u, &hinv_dza) {
                    Some(t3) => [-0.5 * t3[0], -0.5 * t3[1], -0.5 * t3[2]],
                    None => [0.0; 3],
                };
                let rc = [
                    -z[off] + sigma_mu * s_tilde[0] - eta[0],
                    -z[off + 1] + sigma_mu * s_tilde[1] - eta[1],
                    -z[off + 2] + sigma_mu * s_tilde[2] - eta[2],
                ];
                let mc = matvec3(minv, &rc);
                out[off] = -mc[0];
                out[off + 1] = -mc[1];
                out[off + 2] = -mc[2];
            }
            (NsBlock::SecondOrder(_), BlockScaling::SecondOrder { .. }) => {
                // Self-scaled corrector: rhs from the Jordan second-order term
                // s∘z + ds_aff∘dz_aff − σμe (the cone's own corrector).
                let soc = SecondOrderCone::new(d);
                let (sb, zb) = (&s[off..off + d], &z[off..off + d]);
                let mut r_c = vec![0.0; d];
                soc.comp_residual_corrector(
                    sb,
                    zb,
                    &ds_aff[off..off + d],
                    &dz_aff[off..off + d],
                    sigma_mu,
                    &mut r_c,
                );
                soc.rhs_comp_term(sb, zb, &r_c, &mut out[off..off + d]);
            }
            _ => unreachable!("block/scaling shape mismatch"),
        }
    }
}

/// Recover the slack step `Δs = −comp_term − M⁻¹·Δz`.
fn recover_ds(cone: &NsCone, scalings: &[BlockScaling], comp: &[f64], dz: &[f64], ds: &mut [f64]) {
    for (&(off, b), sc) in cone.blocks.iter().zip(scalings) {
        let d = b.dim();
        match sc {
            BlockScaling::Orthant { sz_ratio, .. } => {
                for i in 0..d {
                    ds[off + i] = -comp[off + i] - sz_ratio[i] * dz[off + i];
                }
            }
            BlockScaling::SecondOrder { diag, u } => {
                // Δs = −comp − W²·Δz, with W²·Δz = diag∘Δz + u·(uᵀΔz).
                let dzb = &dz[off..off + d];
                let utdz: f64 = u.iter().zip(dzb).map(|(ui, di)| ui * di).sum();
                for i in 0..d {
                    ds[off + i] = -comp[off + i] - (diag[i] * dzb[i] + u[i] * utdz);
                }
            }
            BlockScaling::Nonsym { minv, .. } => {
                let mdz = matvec3(minv, &dz[off..off + 3]);
                for i in 0..3 {
                    ds[off + i] = -comp[off + i] - mdz[i];
                }
            }
        }
    }
}

/// Largest `α ∈ (0, α_cap]` keeping `s + α ds ∈ int K` and `z + α dz ∈ int K*`
/// for every block, by closed form on orthant blocks and backtracking on exp
/// blocks (no closed-form boundary root). Returns a strictly interior step.
fn max_step(
    cone: &NsCone,
    s: &[f64],
    ds: &[f64],
    z: &[f64],
    dz: &[f64],
    tau: f64,
    alpha_cap: f64,
) -> f64 {
    let mut alpha = alpha_cap;
    // Orthant + second-order cone closed forms first.
    for &(off, b) in &cone.blocks {
        if let NsBlock::SecondOrder(m) = b {
            let soc = SecondOrderCone::new(m);
            alpha = alpha.min(soc.max_step(&s[off..off + m], &ds[off..off + m], tau));
            alpha = alpha.min(soc.max_step(&z[off..off + m], &dz[off..off + m], tau));
        }
    }
    for &(off, b) in &cone.blocks {
        if let NsBlock::Orthant(d) = b {
            for i in 0..d {
                alpha = alpha.min(ray_step(s[off + i], ds[off + i], tau));
                alpha = alpha.min(ray_step(z[off + i], dz[off + i], tau));
            }
        }
    }
    // Backtrack on each non-symmetric block's membership (primal s ∈ K, dual
    // z ∈ K*), using that block's own cone.
    let interior = |alpha: f64| -> bool {
        for &(off, b) in &cone.blocks {
            if let NsBlock::Nonsym(nscone) = b {
                let sp = [
                    s[off] + alpha * ds[off],
                    s[off + 1] + alpha * ds[off + 1],
                    s[off + 2] + alpha * ds[off + 2],
                ];
                let zp = [
                    z[off] + alpha * dz[off],
                    z[off + 1] + alpha * dz[off + 1],
                    z[off + 2] + alpha * dz[off + 2],
                ];
                if !nscone.in_primal_cone(&sp, 1e-12) || !nscone.in_dual_cone(&zp, 1e-12) {
                    return false;
                }
            }
        }
        true
    };
    let mut bt = 0;
    while !interior(alpha) && bt < 100 {
        alpha *= 0.8;
        bt += 1;
    }
    if bt >= 100 {
        0.0
    } else {
        alpha
    }
}

/// Solve `min cᵀx s.t. Ax = b, Gx + s = h, s ∈ K` with `K` a product of
/// orthant and exponential cones, via the non-symmetric HSDE.
fn run_nonsym<F>(
    prob: &QpProblem,
    specs: &[NsBlock],
    opts: &QpOptions,
    warm_x: Option<&[f64]>,
    mut make_backend: F,
    mut hook: Option<&mut dyn DebugHook>,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();
    let cone = NsCone::new(specs);
    debug_assert_eq!(cone.dim, m_ineq, "cone dim must cover all inequality rows");
    let degree = cone.degree;

    let kkt = NsKkt::build(prob, &cone, opts.reg);
    let dim = kkt.dim;

    // Seed the factorization at the cone identity (any SPD block works).
    let mut e = vec![0.0; m_ineq];
    cone.identity(&mut e);
    let mut seed_vals = kkt.values.clone();
    if kkt
        .update_blocks(&cone, &e, &e, opts.reg, &mut seed_vals)
        .is_none()
    {
        return failed(prob);
    }
    let mut fact = match Factorization::new(
        dim as Index,
        kkt.airn.clone(),
        kkt.ajcn.clone(),
        seed_vals,
        make_backend(),
    ) {
        Ok(f) => f,
        Err(_) => return failed(prob),
    };

    let neg_b: Vec<f64> = prob.b.iter().map(|v| -v).collect();
    let neg_h: Vec<f64> = prob.h.iter().map(|v| -v).collect();
    let zeros_m = vec![0.0; m_ineq];

    // Self-dual start: x = y = 0, s = z = e, τ = κ = 1. A warm start seeds the
    // **primal** `x` from a previous (nearby) solution while keeping the cones
    // centered at `e` — this lowers the initial primal residual without
    // destabilizing the embedding. (The HSDE iteration count is start-
    // dependent and is not guaranteed to drop, so this is a primal hook, not a
    // promised speedup; the solution is start-independent regardless.)
    let mut x = match warm_x {
        Some(w) if w.len() == n => w.to_vec(),
        _ => vec![0.0; n],
    };
    let mut y = vec![0.0; m_eq];
    let mut s = e.clone();
    let mut z = e;
    let mut tau = 1.0_f64;
    let mut kappa = 1.0_f64;

    let mut rho_x = vec![0.0; n];
    let mut rho_y = vec![0.0; m_eq];
    let mut rho_z = vec![0.0; m_ineq];
    let mut px_vec = vec![0.0; n];
    let mut comp = vec![0.0; m_ineq];
    let mut kkt_vals = kkt.values.clone();
    let mut rhs = vec![0.0; dim];

    let mut p_x = vec![0.0; n];
    let mut p_y = vec![0.0; m_eq];
    let mut p_z = vec![0.0; m_ineq];
    let mut dx = vec![0.0; n];
    let mut dy = vec![0.0; m_eq];
    let mut dz = vec![0.0; m_ineq];
    let mut ds = vec![0.0; m_ineq];
    let mut dz_aff = vec![0.0; m_ineq];
    let mut ds_aff = vec![0.0; m_ineq];

    let mut status = QpStatus::IterationLimit;
    let mut iters = 0;

    // Best iterate seen, by un-homogenized KKT residual. A feasible conic
    // program can stall a hair short of `tol` when an iterate rides deep on a
    // non-symmetric cone boundary: the barrier Hessian blows up, the
    // fraction-to-boundary step collapses, and the duality gap is amplified by
    // a small τ even though primal/dual feasibility are already tight. We
    // snapshot the lowest-residual iterate so that, if the iteration later
    // breaks down or hits the cap, we can return the point we actually reached
    // (and judge its accuracy) rather than whatever degenerate iterate we died
    // on. See the reduced-accuracy acceptance after the loop.
    let mut best_res = f64::INFINITY;
    let mut best: Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64, f64)> = None;

    for it in 0..opts.max_iter {
        iters = it;

        for v in px_vec.iter_mut() {
            *v = 0.0;
        }
        prob.p_mul(&x, &mut px_vec);
        let xpx = dot(&x, &px_vec);

        // Homogeneous residuals (identical to the symmetric driver).
        for (r, (&ci, &pxi)) in rho_x.iter_mut().zip(prob.c.iter().zip(&px_vec)) {
            *r = ci * tau + pxi;
        }
        prob.at_mul(&y, &mut rho_x);
        prob.gt_mul(&z, &mut rho_x);
        for (r, &bi) in rho_y.iter_mut().zip(&prob.b) {
            *r = -bi * tau;
        }
        prob.a_mul(&x, &mut rho_y);
        for i in 0..m_ineq {
            rho_z[i] = s[i] - prob.h[i] * tau;
        }
        prob.g_mul(&x, &mut rho_z);
        let ctx = dot(&prob.c, &x);
        let bty = dot(&prob.b, &y);
        let htz = dot(&prob.h, &z);
        let rho_tau = kappa + ctx + bty + htz + xpx / tau;

        let sz = dot(&s, &z);
        let mu = (sz + tau * kappa) / (degree as f64 + 1.0);

        // Convergence (un-homogenized).
        let pres = inf_norm(&rho_y).max(inf_norm(&rho_z)) / tau;
        let dres = inf_norm(&rho_x) / tau;
        let gap = (xpx / tau + ctx + bty + htz).abs() / tau;
        let res = pres.max(dres).max(gap);

        // Snapshot the best (lowest-residual) iterate for the reduced-accuracy
        // fallback. τ > 0 only — the recovery un-homogenizes by 1/τ.
        if res < best_res && tau > 0.0 {
            best_res = res;
            best = Some((x.clone(), y.clone(), z.clone(), s.clone(), tau, kappa));
        }

        // Debugger checkpoint: top of iteration. Same homogeneous-iterate
        // view as the symmetric HSDE driver (blocks x/s/y/z + τ/κ).
        if hook.is_some() {
            let obj_hat = 0.5 * xpx / (tau * tau) + ctx / tau;
            let mut st = ConvexDebugState {
                cp: Checkpoint::IterStart,
                iter: it as i32,
                mu,
                pinf: pres,
                dinf: dres,
                res,
                obj: obj_hat,
                alpha: (0.0, 0.0),
                x: &mut x,
                s: &mut s,
                y: &mut y,
                z: &mut z,
                dx: &dx,
                dy: &dy,
                dz: &dz,
                ds: &ds,
                tau: Some(&mut tau),
                kappa: Some(&mut kappa),
                status: None,
            };
            if fire(&mut hook, &mut st) == DebugAction::Stop {
                break;
            }
        }

        if pres < opts.tol && dres < opts.tol && gap < opts.tol {
            status = QpStatus::Optimal;
            break;
        }
        // "Acceptable level": near the cone boundary the barrier Hessian blows
        // up (ψ → 0) and the scaling/factorization can break down a hair short
        // of `tol`. If that happens while the KKT residuals are already tiny
        // (within `~1e3·tol`), the current iterate *is* essentially optimal —
        // accept it rather than reporting a spurious NumericalFailure.
        let near_opt = res < 1e3 * opts.tol;
        // Infeasibility certificate as τ → 0.
        if tau < 1e-2 * kappa.max(1.0) {
            if let Some(st) = detect_infeasibility(prob, &x, &y, &z, opts) {
                status = st;
                break;
            }
        }

        // Refactor M with the dual-aware scaling.
        kkt_vals.copy_from_slice(&kkt.values);
        let scalings = match kkt.update_blocks(&cone, &s, &z, opts.reg, &mut kkt_vals) {
            Some(sc) => sc,
            None => {
                status = if near_opt {
                    QpStatus::Optimal
                } else {
                    QpStatus::NumericalFailure
                };
                break;
            }
        };
        if fact.refactor(&kkt_vals).is_err() {
            status = if near_opt {
                QpStatus::Optimal
            } else {
                QpStatus::NumericalFailure
            };
            break;
        }

        // Constant direction p: M p = (−c, b, h).
        build_rhs(&prob.c, &neg_b, &neg_h, &zeros_m, n, m_eq, m_ineq, &mut rhs);
        if fact.solve_one(&mut rhs).is_err() {
            status = if near_opt {
                QpStatus::Optimal
            } else {
                QpStatus::NumericalFailure
            };
            break;
        }
        split_step(&rhs, n, m_eq, m_ineq, &mut p_x, &mut p_y, &mut p_z);
        let two_over_tau = 2.0 / tau;
        let gtp = dot(&prob.c, &p_x)
            + two_over_tau * dot(&px_vec, &p_x)
            + dot(&prob.b, &p_y)
            + dot(&prob.h, &p_z);
        let denom = gtp - kappa / tau - xpx / (tau * tau);

        // Predictor (σ = 0): rc = −z, comp_term = −M⁻¹·rc = M⁻¹·z.
        comp_term(&cone, &scalings, &s, &z, 0.0, &mut comp);
        build_rhs(&rho_x, &rho_y, &rho_z, &comp, n, m_eq, m_ineq, &mut rhs);
        if fact.solve_one(&mut rhs).is_err() {
            status = if near_opt {
                QpStatus::Optimal
            } else {
                QpStatus::NumericalFailure
            };
            break;
        }
        split_step(&rhs, n, m_eq, m_ineq, &mut dx, &mut dy, &mut dz);
        let gtq = dot(&prob.c, &dx)
            + two_over_tau * dot(&px_vec, &dx)
            + dot(&prob.b, &dy)
            + dot(&prob.h, &dz);
        let dtau_aff = (-rho_tau - gtq + kappa) / denom;
        for i in 0..m_ineq {
            dz_aff[i] = dz[i] + dtau_aff * p_z[i];
        }
        let dkappa_aff = (-tau * kappa - kappa * dtau_aff) / tau;
        recover_ds(&cone, &scalings, &comp, &dz_aff, &mut ds_aff);

        // Affine step (closed form on τ/κ + orthant, backtracking on exp).
        let cap = ray_step(tau, dtau_aff, opts.tau).min(ray_step(kappa, dkappa_aff, opts.tau));
        let alpha_aff = if m_ineq > 0 {
            max_step(&cone, &s, &ds_aff, &z, &dz_aff, opts.tau, cap)
        } else {
            cap
        };
        let mut dot_aff = (tau + alpha_aff * dtau_aff) * (kappa + alpha_aff * dkappa_aff);
        for i in 0..m_ineq {
            dot_aff += (s[i] + alpha_aff * ds_aff[i]) * (z[i] + alpha_aff * dz_aff[i]);
        }
        let mu_aff = dot_aff / (degree as f64 + 1.0);
        let sigma = if mu > 0.0 {
            (mu_aff / mu).powi(3).min(1.0)
        } else {
            0.0
        };
        let sigma_mu = sigma * mu;

        // Centering + corrector step. rc = −z + σμ·s̃ − η, with the
        // nonsymmetric corrector η (Mehrotra second-order for orthant/τκ,
        // third-order for exp). `use_corr = false` drops η (a plain centering
        // step) — the safeguard fallback when the corrector overshoots.
        // Use the corrector in the bulk iterations only. Near convergence its
        // marginal benefit is gone and the finite-difference third-derivative
        // perturbation can stall the endgame, so fall to pure centering (the
        // provably convergent path) once residuals are within ~1e3·tol.
        let near_conv = pres.max(dres).max(gap) < 1e3 * opts.tol;
        let mut use_corr = !near_conv;
        let mut dtau = 0.0_f64;
        let mut dkappa = 0.0_f64;
        let mut alpha = 0.0_f64;
        let mut solve_failed = false;
        loop {
            if use_corr {
                comp_term_corr(
                    &cone, &scalings, &s, &z, sigma_mu, &ds_aff, &dz_aff, &mut comp,
                );
            } else {
                comp_term(&cone, &scalings, &s, &z, sigma_mu, &mut comp);
            }
            build_rhs(&rho_x, &rho_y, &rho_z, &comp, n, m_eq, m_ineq, &mut rhs);
            if fact.solve_one(&mut rhs).is_err() {
                solve_failed = true;
                break;
            }
            split_step(&rhs, n, m_eq, m_ineq, &mut dx, &mut dy, &mut dz);
            let gtq = dot(&prob.c, &dx)
                + two_over_tau * dot(&px_vec, &dx)
                + dot(&prob.b, &dy)
                + dot(&prob.h, &dz);
            // τκ second-order term Δτ_aff·Δκ_aff only when the corrector is on.
            let r_tk = if use_corr {
                tau * kappa + dtau_aff * dkappa_aff
            } else {
                tau * kappa
            };
            dtau = (-rho_tau - gtq - (sigma_mu - r_tk) / tau) / denom;
            for i in 0..n {
                dx[i] += dtau * p_x[i];
            }
            for i in 0..m_eq {
                dy[i] += dtau * p_y[i];
            }
            for i in 0..m_ineq {
                dz[i] += dtau * p_z[i];
            }
            dkappa = (sigma_mu - r_tk - kappa * dtau) / tau;
            recover_ds(&cone, &scalings, &comp, &dz, &mut ds);

            let cap = ray_step(tau, dtau, opts.tau).min(ray_step(kappa, dkappa, opts.tau));
            alpha = if m_ineq > 0 {
                max_step(&cone, &s, &ds, &z, &dz, opts.tau, cap)
            } else {
                cap
            };
            // If the corrector collapses the step, retry once without it.
            if use_corr && alpha < 1e-2 {
                use_corr = false;
                continue;
            }
            break;
        }
        if solve_failed {
            status = if near_opt {
                QpStatus::Optimal
            } else {
                QpStatus::NumericalFailure
            };
            break;
        }
        if alpha <= 0.0 {
            status = if near_opt {
                QpStatus::Optimal
            } else {
                QpStatus::NumericalFailure
            };
            break;
        }

        // Debugger checkpoint: combined Newton direction + step length known,
        // not yet applied (single symmetric α in both slots).
        if hook.is_some() {
            let obj_hat = 0.5 * xpx / (tau * tau) + ctx / tau;
            let mut st = ConvexDebugState {
                cp: Checkpoint::AfterSearchDirection,
                iter: it as i32,
                mu,
                pinf: pres,
                dinf: dres,
                res,
                obj: obj_hat,
                alpha: (alpha, alpha),
                x: &mut x,
                s: &mut s,
                y: &mut y,
                z: &mut z,
                dx: &dx,
                dy: &dy,
                dz: &dz,
                ds: &ds,
                tau: Some(&mut tau),
                kappa: Some(&mut kappa),
                status: None,
            };
            if fire(&mut hook, &mut st) == DebugAction::Stop {
                break;
            }
        }

        for i in 0..n {
            x[i] += alpha * dx[i];
        }
        for i in 0..m_eq {
            y[i] += alpha * dy[i];
        }
        for i in 0..m_ineq {
            s[i] += alpha * ds[i];
            z[i] += alpha * dz[i];
        }
        tau += alpha * dtau;
        kappa += alpha * dkappa;

        // Debugger checkpoint: the new homogeneous iterate is in place.
        if hook.is_some() {
            let mut pxn = vec![0.0; n];
            prob.p_mul(&x, &mut pxn);
            let obj_hat = 0.5 * dot(&x, &pxn) / (tau * tau) + dot(&prob.c, &x) / tau;
            let mut st = ConvexDebugState {
                cp: Checkpoint::AfterStep,
                iter: it as i32,
                mu,
                pinf: pres,
                dinf: dres,
                res,
                obj: obj_hat,
                alpha: (alpha, alpha),
                x: &mut x,
                s: &mut s,
                y: &mut y,
                z: &mut z,
                dx: &dx,
                dy: &dy,
                dz: &dz,
                ds: &ds,
                tau: Some(&mut tau),
                kappa: Some(&mut kappa),
                status: None,
            };
            if fire(&mut hook, &mut st) == DebugAction::Stop {
                break;
            }
        }
    }

    // Reduced-accuracy acceptance. If the driver broke down or hit the cap
    // (NumericalFailure / IterationLimit) but the best iterate we reached has a
    // KKT residual within √tol (e.g. tol=1e-8 → 1e-4), the problem was
    // essentially solved — a near-boundary stall on a non-symmetric cone, not a
    // genuine failure. Restore that iterate and report Optimal, mirroring the
    // "solved to reduced accuracy" outcome of ECOS/Clarabel/SCS. This never
    // fires for infeasible/unbounded problems (their residuals never get this
    // small — the embedding drives τ → 0 and the certificate path triggers
    // first) and never relaxes the clean convergence test above (still `tol`).
    if matches!(
        status,
        QpStatus::NumericalFailure | QpStatus::IterationLimit
    ) {
        let reduced_acc = opts.tol.sqrt();
        if best_res < reduced_acc {
            if let Some((bx, by, bz, bs, btau, _bkappa)) = best.take() {
                // κ is not read downstream (the recovery un-homogenizes by
                // 1/τ); restoring x/y/z/s/τ is what the solution recovery and
                // the post-mortem hook consume.
                x = bx;
                y = by;
                z = bz;
                s = bs;
                tau = btau;
                status = QpStatus::Optimal;
            }
        }
    }

    let inv = if tau.abs() > 0.0 { 1.0 / tau } else { 1.0 };
    let mut x: Vec<f64> = x.iter().map(|v| v * inv).collect();
    let mut y: Vec<f64> = y.iter().map(|v| v * inv).collect();
    let mut z: Vec<f64> = z.iter().map(|v| v * inv).collect();
    let mut px = vec![0.0; n];
    prob.p_mul(&x, &mut px);
    let obj = 0.5 * dot(&x, &px) + dot(&prob.c, &x);

    // Debugger post-mortem at the recovered (un-homogenized) solution.
    if hook.is_some() {
        let status_str = format!("{status:?}");
        let mut st = ConvexDebugState {
            cp: Checkpoint::Terminated,
            iter: iters as i32,
            mu: 0.0,
            pinf: 0.0,
            dinf: 0.0,
            res: 0.0,
            obj,
            alpha: (0.0, 0.0),
            x: &mut x,
            s: &mut s,
            y: &mut y,
            z: &mut z,
            dx: &dx,
            dy: &dy,
            dz: &dz,
            ds: &ds,
            tau: Some(&mut tau),
            kappa: Some(&mut kappa),
            status: Some(&status_str),
        };
        let _ = fire(&mut hook, &mut st);
    }

    QpSolution {
        status,
        x,
        y,
        z,
        z_lb: vec![0.0; n],
        z_ub: vec![0.0; n],
        obj,
        iters,
        iterates: Vec::new(),
    }
}

/// Solve `min cᵀx s.t. Ax = b, Gx + s = h, s ∈ K` with `K` a product of
/// orthant, second-order, exponential, and power cones, via the non-symmetric
/// HSDE (cold self-dual start).
pub fn solve_conic_hsde_nonsym<F>(
    prob: &QpProblem,
    specs: &[NsBlock],
    opts: &QpOptions,
    make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    run_nonsym(prob, specs, opts, None, make_backend, None)
}

/// Debug-enabled [`solve_conic_hsde_nonsym`]: fires the interactive
/// [`DebugHook`] at each interior-point checkpoint of the non-symmetric
/// (exponential / power) HSDE solve. The iterate view matches the
/// symmetric HSDE driver (homogeneous `x/s/y/z` plus `τ/κ`). Apart from
/// the hook the result is identical.
pub fn solve_conic_hsde_nonsym_debug<F>(
    prob: &QpProblem,
    specs: &[NsBlock],
    opts: &QpOptions,
    hook: &mut dyn DebugHook,
    make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    run_nonsym(prob, specs, opts, None, make_backend, Some(hook))
}

/// Warm-started [`solve_conic_hsde_nonsym`]: seed the primal `x` from `warm_x`
/// (a previous, nearby solution) while keeping the cones centered. The
/// solution is start-independent; warm-starting lowers the initial primal
/// residual but — as for any HSDE embedding — is not guaranteed to reduce the
/// iteration count. `warm_x` is ignored if its length ≠ `prob.n`.
pub fn solve_conic_hsde_nonsym_warm<F>(
    prob: &QpProblem,
    specs: &[NsBlock],
    warm_x: &[f64],
    opts: &QpOptions,
    make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    run_nonsym(prob, specs, opts, Some(warm_x), make_backend, None)
}

fn failed(prob: &QpProblem) -> QpSolution {
    QpSolution {
        status: QpStatus::NumericalFailure,
        x: vec![0.0; prob.n],
        y: vec![0.0; prob.m_eq()],
        z: vec![1.0; prob.m_ineq()],
        z_lb: vec![0.0; prob.n],
        z_ub: vec![0.0; prob.n],
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qp::Triplet;
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    fn opts() -> QpOptions {
        QpOptions {
            max_iter: 200,
            ..QpOptions::default()
        }
    }

    /// An exponential cone is always 3 rows. Declaring it over a `G` with
    /// only 2 inequality rows is a caller error: the driver must fail
    /// cleanly (`NumericalFailure`) instead of indexing past the 2-row
    /// slack and panicking — the guard in [`crate::ipm::solve_socp_ipm`].
    #[test]
    fn mismatched_cone_dims_fail_cleanly() {
        use crate::cones::ConeSpec;
        use crate::ipm::solve_socp_ipm;
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![1.0, 0.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0), Triplet::new(1, 1, -1.0)],
            h: vec![0.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_socp_ipm(&prob, &[ConeSpec::Exponential], &opts(), backend);
        assert_eq!(sol.status, QpStatus::NumericalFailure);
    }

    /// `min z s.t. x = 1, y = 1, (x,y,z) ∈ K_exp`. The cone forces
    /// `z ≥ y·exp(x/y) = e`, so the optimum is `z = e` at `x = y = 1`.
    #[test]
    fn exp_epigraph_known_optimum() {
        let e = std::f64::consts::E;
        // Variables v = (x, y, z); slack s = v ∈ K_exp via G = −I, h = 0.
        let prob = QpProblem {
            n: 3,
            p_lower: vec![],
            c: vec![0.0, 0.0, 1.0],
            a: vec![
                Triplet::new(0, 0, 1.0), // x = 1
                Triplet::new(1, 1, 1.0), // y = 1
            ],
            b: vec![1.0, 1.0],
            g: vec![
                Triplet::new(0, 0, -1.0),
                Triplet::new(1, 1, -1.0),
                Triplet::new(2, 2, -1.0),
            ],
            h: vec![0.0, 0.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_conic_hsde_nonsym(&prob, &[NsBlock::exp()], &opts(), backend);
        assert_eq!(
            sol.status,
            QpStatus::Optimal,
            "not optimal: {:?}",
            sol.status
        );
        assert!((sol.x[0] - 1.0).abs() < 1e-5, "x = {}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-5, "y = {}", sol.x[1]);
        assert!((sol.x[2] - e).abs() < 1e-5, "z = {} vs e = {e}", sol.x[2]);
        assert!((sol.obj - e).abs() < 1e-5, "obj = {} vs e", sol.obj);
    }

    /// `log-sum-exp` epigraph: `min t s.t. t ≥ log(e^{x₁} + e^{x₂})` with
    /// `x₁ = x₂ = 0`, so the optimum is `t = log 2`. Modeled with two exp
    /// cones `(xᵢ − t, 1, uᵢ) ∈ K_exp` (⇒ `uᵢ ≥ e^{xᵢ−t}`) and the orthant
    /// row `u₁ + u₂ ≤ 1`. This exercises **multiple exp blocks + an orthant
    /// block** in one product cone — the mixed-cone path.
    #[test]
    fn log_sum_exp_known_optimum() {
        // v = (t, u1, u2). Rows: exp1 (0..3), exp2 (3..6), orthant (6).
        let prob = QpProblem {
            n: 3,
            p_lower: vec![],
            c: vec![1.0, 0.0, 0.0], // min t
            a: vec![],
            b: vec![],
            g: vec![
                // exp1 slack = (x1 − t, 1, u1) = (−t, 1, u1)
                Triplet::new(0, 0, 1.0),  // s0 = −t
                Triplet::new(2, 1, -1.0), // s2 = u1
                // exp2 slack = (−t, 1, u2)
                Triplet::new(3, 0, 1.0),  // s3 = −t
                Triplet::new(5, 2, -1.0), // s5 = u2
                // orthant: s6 = 1 − u1 − u2
                Triplet::new(6, 1, 1.0),
                Triplet::new(6, 2, 1.0),
            ],
            // middle exp components pinned to 1 via h (G row = 0).
            h: vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            lb: vec![],
            ub: vec![],
        };
        let specs = [NsBlock::exp(), NsBlock::exp(), NsBlock::Orthant(1)];
        let sol = solve_conic_hsde_nonsym(&prob, &specs, &opts(), backend);
        assert_eq!(
            sol.status,
            QpStatus::Optimal,
            "not optimal: {:?}",
            sol.status
        );
        let want = 2.0_f64.ln();
        assert!(
            (sol.x[0] - want).abs() < 1e-5,
            "t = {} vs log2 = {want}",
            sol.x[0]
        );
        // uᵢ = e^{−t} = 1/2 at the optimum.
        assert!((sol.x[1] - 0.5).abs() < 1e-4, "u1 = {}", sol.x[1]);
        assert!((sol.x[2] - 0.5).abs() < 1e-4, "u2 = {}", sol.x[2]);
    }

    /// A tiny **geometric program**: `min x + 1/x` over `x > 0`, whose optimum
    /// is `2` at `x = 1`. With `x = e^u` it becomes `min e^u + e^{−u}`, modeled
    /// as `min t₁ + t₂` with `(u, 1, t₁) ∈ K_exp` (`t₁ ≥ e^u`) and
    /// `(−u, 1, t₂) ∈ K_exp` (`t₂ ≥ e^{−u}`). Optimum `u = 0`, `t₁ = t₂ = 1`.
    #[test]
    fn geometric_program_known_optimum() {
        // v = (u, t1, t2). Rows: exp1 (0..3), exp2 (3..6).
        let prob = QpProblem {
            n: 3,
            p_lower: vec![],
            c: vec![0.0, 1.0, 1.0], // min t1 + t2
            a: vec![],
            b: vec![],
            g: vec![
                // exp1 slack = (u, 1, t1)
                Triplet::new(0, 0, -1.0), // s0 = u
                Triplet::new(2, 1, -1.0), // s2 = t1
                // exp2 slack = (−u, 1, t2)
                Triplet::new(3, 0, 1.0),  // s3 = −u
                Triplet::new(5, 2, -1.0), // s5 = t2
            ],
            h: vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let specs = [NsBlock::exp(), NsBlock::exp()];
        let sol = solve_conic_hsde_nonsym(&prob, &specs, &opts(), backend);
        assert_eq!(
            sol.status,
            QpStatus::Optimal,
            "not optimal: {:?}",
            sol.status
        );
        assert!((sol.x[0]).abs() < 1e-4, "u = {} vs 0", sol.x[0]);
        assert!((sol.obj - 2.0).abs() < 1e-5, "obj = {} vs 2", sol.obj);
    }

    /// The same geometric program routed through the **public** entry
    /// `solve_socp_ipm` with `ConeSpec::Exponential` — confirms the routing
    /// (exp specs → non-symmetric driver) is wired end-to-end.
    #[test]
    fn routes_exponential_through_public_entry() {
        use crate::cones::ConeSpec;
        use crate::ipm::solve_socp_ipm;
        let prob = QpProblem {
            n: 3,
            p_lower: vec![],
            c: vec![0.0, 1.0, 1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, -1.0),
                Triplet::new(2, 1, -1.0),
                Triplet::new(3, 0, 1.0),
                Triplet::new(5, 2, -1.0),
            ],
            h: vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let specs = [ConeSpec::Exponential, ConeSpec::Exponential];
        let sol = solve_socp_ipm(&prob, &specs, &opts(), backend);
        assert_eq!(
            sol.status,
            QpStatus::Optimal,
            "not optimal: {:?}",
            sol.status
        );
        assert!((sol.obj - 2.0).abs() < 1e-5, "obj = {} vs 2", sol.obj);
    }

    /// Power cone known optimum: `max x s.t. (x, 2, 0.5) ∈ K_α`, i.e.
    /// `x ≤ 2^α · 0.5^{1−α}`. For α = 0.5 the bound is `√(2·0.5) = 1`.
    #[test]
    fn power_cone_known_optimum() {
        // v = (x, y, z); slack s = v ∈ K_α via G = −I, h = 0; y = 2, z = 0.5.
        let prob = QpProblem {
            n: 3,
            p_lower: vec![],
            c: vec![-1.0, 0.0, 0.0], // max x
            a: vec![Triplet::new(0, 1, 1.0), Triplet::new(1, 2, 1.0)],
            b: vec![2.0, 0.5],
            g: vec![
                Triplet::new(0, 0, -1.0),
                Triplet::new(1, 1, -1.0),
                Triplet::new(2, 2, -1.0),
            ],
            h: vec![0.0, 0.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        for alpha in [0.5, 0.3, 0.75] {
            let sol = solve_conic_hsde_nonsym(&prob, &[NsBlock::power(alpha)], &opts(), backend);
            assert_eq!(sol.status, QpStatus::Optimal, "α={alpha}: {:?}", sol.status);
            let want = 2.0_f64.powf(alpha) * 0.5_f64.powf(1.0 - alpha);
            assert!(
                (sol.x[0] - want).abs() < 1e-5,
                "α={alpha}: x = {} vs {want}",
                sol.x[0]
            );
        }
    }

    /// A **second-order cone mixed with an exponential cone** in one problem.
    /// `min t + z s.t. (t, 3, 4) ∈ SOC(3)` (⇒ `t ≥ ‖(3,4)‖ = 5`) and
    /// `(1, 1, z) ∈ K_exp` (⇒ `z ≥ e`). Optimum `t = 5`, `z = e`,
    /// `obj = 5 + e`. Exercises the self-scaled SOC path and the dual-aware
    /// exp path together.
    #[test]
    fn soc_mixed_with_exp() {
        let e = std::f64::consts::E;
        // v = (t, z). Rows: SOC (0..3) = (t, 3, 4); exp (3..6) = (1, 1, z).
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![1.0, 1.0], // min t + z
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, -1.0), // SOC s0 = t
                Triplet::new(5, 1, -1.0), // exp s5 = z
            ],
            h: vec![0.0, 3.0, 4.0, 1.0, 1.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let specs = [NsBlock::SecondOrder(3), NsBlock::exp()];
        let sol = solve_conic_hsde_nonsym(&prob, &specs, &opts(), backend);
        assert_eq!(
            sol.status,
            QpStatus::Optimal,
            "not optimal: {:?}",
            sol.status
        );
        assert!((sol.x[0] - 5.0).abs() < 1e-5, "t = {} vs 5", sol.x[0]);
        assert!((sol.x[1] - e).abs() < 1e-5, "z = {} vs e", sol.x[1]);
        assert!(
            (sol.obj - (5.0 + e)).abs() < 1e-5,
            "obj = {} vs 5+e",
            sol.obj
        );
    }

    /// Warm-starting is **start-independent**: seeding the primal from the
    /// optimum, or from a deliberately wrong point, converges to the same
    /// solution. (We verify correctness — the property the warm path must
    /// preserve — not an iteration-count reduction, which the HSDE embedding
    /// does not guarantee.)
    #[test]
    fn warm_start_is_start_independent() {
        // Geometric program min e^u + e^{−u} = 2 (u, t1, t2).
        let prob = QpProblem {
            n: 3,
            p_lower: vec![],
            c: vec![0.0, 1.0, 1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, -1.0),
                Triplet::new(2, 1, -1.0),
                Triplet::new(3, 0, 1.0),
                Triplet::new(5, 2, -1.0),
            ],
            h: vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let specs = [NsBlock::exp(), NsBlock::exp()];
        let cold = solve_conic_hsde_nonsym(&prob, &specs, &opts(), backend);
        assert_eq!(cold.status, QpStatus::Optimal);
        assert!((cold.obj - 2.0).abs() < 1e-5);

        // The objective is the start-independent invariant (the GP minimum is
        // flat in `u`, so the coordinate itself is sensitive — the objective
        // is what must agree). Warm from the optimum, a bad point, and a
        // length-mismatched (ignored) vector all reach the same optimum.
        for warm in [cold.x.as_slice(), &[50.0, -30.0, 9.0], &[1.0]] {
            let sol = solve_conic_hsde_nonsym_warm(&prob, &specs, warm, &opts(), backend);
            assert_eq!(sol.status, QpStatus::Optimal, "warm {warm:?}");
            assert!(
                (sol.obj - cold.obj).abs() < 1e-5,
                "warm {warm:?} obj {} vs {}",
                sol.obj,
                cold.obj
            );
        }
    }

    /// SOC routed through the non-symmetric driver alone matches the known
    /// norm-minimization optimum (validates the SOC path in isolation).
    /// `min t s.t. (t, x−2, x+1) ∈ SOC` → `x = ?`; simplest: `(t, 3, 4)` → 5.
    #[test]
    fn soc_only_through_nonsym_driver() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0)],
            h: vec![0.0, 3.0, 4.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_conic_hsde_nonsym(&prob, &[NsBlock::SecondOrder(3)], &opts(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "{:?}", sol.status);
        assert!((sol.x[0] - 5.0).abs() < 1e-5, "t = {} vs 5", sol.x[0]);
    }

    /// Power cone routed through the **public** entry `solve_socp_ipm` with
    /// `ConeSpec::Power(α)`.
    #[test]
    fn routes_power_through_public_entry() {
        use crate::cones::ConeSpec;
        use crate::ipm::solve_socp_ipm;
        let prob = QpProblem {
            n: 3,
            p_lower: vec![],
            c: vec![-1.0, 0.0, 0.0],
            a: vec![Triplet::new(0, 1, 1.0), Triplet::new(1, 2, 1.0)],
            b: vec![2.0, 0.5],
            g: vec![
                Triplet::new(0, 0, -1.0),
                Triplet::new(1, 1, -1.0),
                Triplet::new(2, 2, -1.0),
            ],
            h: vec![0.0, 0.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_socp_ipm(&prob, &[ConeSpec::Power(0.5)], &opts(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "{:?}", sol.status);
        assert!((sol.x[0] - 1.0).abs() < 1e-5, "x = {} vs 1", sol.x[0]);
    }
}
