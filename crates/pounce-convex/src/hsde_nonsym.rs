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

use crate::cones::{
    BarrierCone, Cone, ConeBlock, ConeSpec, ExponentialCone, PowerCone, SecondOrderCone,
};
use crate::debug::{ConvexDebugState, fire};
use crate::ipm::{QpOptions, build_rhs, detect_infeasibility_with, dot, inf_norm, split_step};
use crate::qp::{QpProblem, QpSolution, QpStatus, breakdown_status};
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
    ($self:ident, $c:ident => $body:expr_2021) => {
        match $self {
            NonsymCone::Exp($c) => $body,
            NonsymCone::Power($c) => $body,
        }
    };
}

impl NonsymCone {
    /// Whether `p` lies in the **closure** of the primal cone `cl(K)` (to a
    /// relative tolerance `tol`). Unlike [`BarrierCone::in_primal_cone`], which
    /// tests the strict *interior* (`y > tol`), this accepts the boundary /
    /// recession faces of the cone — the points a genuine recession ray lands
    /// on. It is used only by the dual-infeasibility (unboundedness) detector,
    /// where the recession direction `−Gd` must satisfy `−Gd ∈ cl(K)`; a
    /// recession ray legitimately lies on `∂K`, so the strict-interior test
    /// wrongly rejects it (e.g. the exp cone's `y = 0` face, gh #283).
    fn in_primal_closure(&self, p: &[f64], tol: f64) -> bool {
        let (x, y, z) = (p[0], p[1], p[2]);
        let scale = 1.0 + x.abs() + y.abs() + z.abs();
        let t = tol * scale;
        match self {
            NonsymCone::Exp(_) => {
                // cl(K_exp) = { y>0, z>0, y·log(z/y) ≥ x } ∪ { x≤0, y=0, z≥0 }.
                if z < -t {
                    return false;
                }
                if y > t {
                    // Interior/main region: need z > 0 to evaluate ψ.
                    if z <= 0.0 {
                        return false;
                    }
                    y * (z / y).ln() - x >= -t
                } else if y >= -t {
                    // Recession face y ≈ 0: x ≤ 0 and z ≥ 0.
                    x <= t && z >= -t
                } else {
                    false
                }
            }
            NonsymCone::Power(c) => {
                // K_α = { |x| ≤ y^α z^{1−α}, y ≥ 0, z ≥ 0 } is already closed;
                // its boundary (y = 0, z = 0, or |x| = y^α z^{1−α}) is included.
                if y < -t || z < -t {
                    return false;
                }
                let (yc, zc) = (y.max(0.0), z.max(0.0));
                let bound = yc.powf(c.alpha) * zc.powf(1.0 - c.alpha);
                bound - x.abs() >= -t
            }
        }
    }
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

/// Provably-infeasible cone-domain detection at setup, before the HSDE solve.
///
/// A power/exponential cone requires two of its three coordinates to be
/// nonnegative (`y ≥ 0` and `z ≥ 0`) at *every* feasible point — this is the
/// cone's *domain*, independent of the barrier. If the data pin such a
/// coordinate strictly below its domain (e.g. the power cone's `y`-slack is a
/// constant `−1`, or is forced `≤ −1` by another row), the problem is primal
/// infeasible at every point. The HSDE driver's Farkas detector needs the
/// iterate's certificate residual to fall below `FARKAS_RESID_TOL` (~1e-10);
/// on these cone-domain violations the embedding stalls with a small-but-finite
/// residual and never certifies, degrading to `NumericalFailure` (gh #283).
/// This *exact* setup check reports `PrimalInfeasible` directly.
///
/// It is a proof by contradiction: assume a feasible point exists, propagate
/// the resulting variable bounds through the `≥ 0` rows (nonnegative-orthant
/// rows, the leading second-order-cone row, and each exp/power cone's `y`/`z`
/// domain rows) and equality rows by sound interval arithmetic (FBBT), then if
/// any domain row's slack has a **strictly negative upper bound** — or any
/// variable's derived range is empty — feasibility is impossible. Every bound
/// derived is a valid implication of feasibility, so a contradiction proves
/// infeasibility and **no feasible/bounded problem is ever flagged**.
pub(crate) fn detect_cone_domain_infeasible(prob: &QpProblem, blocks: &[NsBlock]) -> bool {
    let n = prob.n;
    let inf = crate::qp::BOUND_INF;
    // Variable ranges, seeded from the problem's own bounds.
    let mut lo = vec![-inf; n];
    let mut hi = vec![inf; n];
    for j in 0..n {
        lo[j] = prob.lb_of(j);
        hi[j] = prob.ub_of(j);
        if lo[j] > hi[j] + 1e-12 {
            return true; // empty variable range
        }
    }

    // Row expressions grouped by inequality-row index: (col, coeff) of G.
    // Slack is `s_r = h_r − Σ g·x`.
    let m_ineq = prob.m_ineq();
    let mut g_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m_ineq];
    for t in &prob.g {
        g_rows[t.row].push((t.col, t.val));
    }
    // Equality rows of A: `Σ a·x = b`.
    let m_eq = prob.m_eq();
    let mut a_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m_eq];
    for t in &prob.a {
        a_rows[t.row].push((t.col, t.val));
    }

    // Inequality rows whose slack is provably `≥ 0` at every feasible point,
    // plus the subset that are *cone-domain* rows we will test for violation.
    let mut nonneg_rows: Vec<usize> = Vec::new();
    let mut domain_rows: Vec<usize> = Vec::new();
    let mut off = 0usize;
    for b in blocks {
        match b {
            NsBlock::Orthant(d) => {
                for r in off..off + d {
                    nonneg_rows.push(r);
                }
            }
            NsBlock::SecondOrder(m) => {
                // `s₀ ≥ ‖s₁..‖ ≥ 0` is a sound `≥ 0` fact for the leading row.
                nonneg_rows.push(off);
                let _ = m;
            }
            NsBlock::Nonsym(_) => {
                // Both the `y` (off+1) and `z` (off+2) coordinates must be ≥ 0.
                nonneg_rows.push(off + 1);
                nonneg_rows.push(off + 2);
                domain_rows.push(off + 1);
                domain_rows.push(off + 2);
            }
        }
        off += b.dim();
    }

    // Min/max of `Σ coeff·x` over the current variable ranges (±inf when a
    // needed bound is absent).
    let expr_bounds = |row: &[(usize, f64)], lo: &[f64], hi: &[f64]| -> (f64, f64) {
        let (mut emin, mut emax) = (0.0_f64, 0.0_f64);
        for &(j, g) in row {
            let (l, u) = (lo[j], hi[j]);
            if g > 0.0 {
                emin += if l <= -inf { -inf } else { g * l };
                emax += if u >= inf { inf } else { g * u };
            } else {
                emin += if u >= inf { -inf } else { g * u };
                emax += if l <= -inf { inf } else { g * l };
            }
        }
        (emin, emax)
    };

    // Sound FBBT: tighten variable ranges from the `≥ 0` and equality rows.
    // Each pass is a valid implication of feasibility; a fixpoint or a small
    // pass cap suffices for the local structure these certificates carry.
    for _ in 0..8 {
        let mut changed = false;
        // Constraints of the form `Σ g·x ≤ rhs` (nonneg rows: `Σ g·x ≤ h_r`;
        // equality rows contribute both `≤ b` and `≥ b`, added below).
        let mut tighten_upper = |row: &[(usize, f64)], rhs: f64, lo: &mut [f64], hi: &mut [f64]| {
            if rhs >= inf {
                return;
            }
            // Min contribution of a single `coeff·x` term over `[lo,hi]`.
            let contrib_min = |j: usize, g: f64, lo: &[f64], hi: &[f64]| -> f64 {
                if g > 0.0 {
                    if lo[j] <= -inf { -inf } else { g * lo[j] }
                } else if hi[j] >= inf {
                    -inf
                } else {
                    g * hi[j]
                }
            };
            for &(j, g) in row {
                // rest_min = Σ_{k≠j} min(g_k·x_k); computed directly (not by
                // subtracting the possibly-infinite own term from the total).
                let mut rest_min = 0.0_f64;
                let mut finite = true;
                for &(k, gk) in row {
                    if k == j {
                        continue;
                    }
                    let c = contrib_min(k, gk, lo, hi);
                    if c <= -inf {
                        finite = false;
                        break;
                    }
                    rest_min += c;
                }
                if !finite {
                    continue;
                }
                // g·x_j ≤ rhs − rest_min.
                let bound = (rhs - rest_min) / g; // g ≠ 0 for stored triplets
                if g > 0.0 {
                    if bound < hi[j] - 1e-12 {
                        hi[j] = bound;
                        changed = true;
                    }
                } else if bound > lo[j] + 1e-12 {
                    lo[j] = bound;
                    changed = true;
                }
            }
        };
        for &r in &nonneg_rows {
            // `s_r = h_r − Σ g·x ≥ 0` ⟺ `Σ g·x ≤ h_r`.
            tighten_upper(&g_rows[r], prob.h[r], &mut lo, &mut hi);
        }
        for r in 0..m_eq {
            // `Σ a·x = b` ⟺ `Σ a·x ≤ b` and `Σ (−a)·x ≤ −b`.
            let b = prob.b[r];
            tighten_upper(&a_rows[r], b, &mut lo, &mut hi);
            let neg: Vec<(usize, f64)> = a_rows[r].iter().map(|&(j, a)| (j, -a)).collect();
            tighten_upper(&neg, -b, &mut lo, &mut hi);
        }
        // Empty variable range ⇒ infeasible.
        for j in 0..n {
            if lo[j] > hi[j] + 1e-9 * (1.0 + lo[j].abs() + hi[j].abs()) {
                return true;
            }
        }
        if !changed {
            break;
        }
    }

    // Cone-domain violation: a domain row whose slack's *upper* bound is
    // strictly negative. `max s_r = h_r − min(Σ g·x)`.
    for &r in &domain_rows {
        let (emin, _) = expr_bounds(&g_rows[r], &lo, &hi);
        if emin <= -inf {
            continue; // unbounded above ⇒ no violation provable
        }
        let s_max = prob.h[r] - emin;
        let margin = 1e-7 * (1.0 + prob.h[r].abs() + emin.abs());
        if s_max < -margin {
            return true;
        }
    }
    false
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

    /// Whether `z` lies in the **dual** cone `K*` (to tolerance `tol`),
    /// block by block. The orthant and SOC are self-dual; exp/power use
    /// their barrier-cone dual test (`K_exp*` requires `u < 0`, so a
    /// componentwise `z ≥ 0` test is wrong in *both* directions). Used to
    /// validate Farkas (primal-infeasibility) multipliers.
    pub(crate) fn in_dual_cone(&self, z: &[f64], tol: f64) -> bool {
        for &(off, b) in &self.blocks {
            let ok = match b {
                NsBlock::Orthant(d) => z[off..off + d].iter().all(|&v| v >= -tol),
                // SOC is self-dual ⇒ its `in_dual_cone` is the membership test.
                NsBlock::SecondOrder(m) => {
                    SecondOrderCone::new(m).in_dual_cone(&z[off..off + m], tol)
                }
                NsBlock::Nonsym(c) => c.in_dual_cone(&z[off..off + 3], tol),
            };
            if !ok {
                return false;
            }
        }
        true
    }

    /// Whether `s` lies in the strict interior of the **primal** cone `K` (to
    /// tolerance `tol`), block by block. For exp/power this is distinct from
    /// `in_dual_cone` (the cones are not self-dual). The recession-membership
    /// test for the dual-infeasibility certificate uses the *closure* variant
    /// [`in_primal_closure`](Self::in_primal_closure) instead, since a
    /// recession ray lands on `∂K` (gh #283); this strict-interior form is
    /// retained as a cone-membership oracle in the unit tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn in_primal_cone(&self, s: &[f64], tol: f64) -> bool {
        for &(off, b) in &self.blocks {
            let ok = match b {
                NsBlock::Orthant(d) => s[off..off + d].iter().all(|&v| v >= -tol),
                // SOC is self-dual ⇒ `in_dual_cone` doubles as the primal test.
                NsBlock::SecondOrder(m) => {
                    SecondOrderCone::new(m).in_dual_cone(&s[off..off + m], tol)
                }
                NsBlock::Nonsym(c) => c.in_primal_cone(&s[off..off + 3], tol),
            };
            if !ok {
                return false;
            }
        }
        true
    }

    /// Whether `s` lies in the **closure** of the primal cone `cl(K)` (to
    /// tolerance `tol`), block by block. This is the recession-membership test
    /// for the dual-infeasibility certificate: a recession ray `−Gd` genuinely
    /// lies on the boundary of `K` (e.g. the exp cone's `y = 0` face), so the
    /// strict-interior [`in_primal_cone`](Self::in_primal_cone) rejects it. The
    /// orthant and second-order cones are already closed, so their closure test
    /// coincides with membership; exp/power use their `cl(K)` face conditions.
    pub(crate) fn in_primal_closure(&self, s: &[f64], tol: f64) -> bool {
        for &(off, b) in &self.blocks {
            let ok = match b {
                NsBlock::Orthant(d) => s[off..off + d].iter().all(|&v| v >= -tol),
                NsBlock::SecondOrder(m) => {
                    SecondOrderCone::new(m).in_dual_cone(&s[off..off + m], tol)
                }
                NsBlock::Nonsym(c) => c.in_primal_closure(&s[off..off + 3], tol),
            };
            if !ok {
                return false;
            }
        }
        true
    }

    /// This cone product as the [`ConeSpec`] list consumed by
    /// [`QpSolution::kkt_residuals_conic`], so the recovered point can be scored
    /// against the same conic optimality measure the direct/symmetric paths use.
    fn cone_specs(&self) -> Vec<ConeSpec> {
        self.blocks
            .iter()
            .map(|&(_, b)| match b {
                NsBlock::Orthant(n) => ConeSpec::Nonneg(n),
                NsBlock::SecondOrder(m) => ConeSpec::SecondOrder(m),
                NsBlock::Nonsym(NonsymCone::Exp(_)) => ConeSpec::Exponential,
                NsBlock::Nonsym(NonsymCone::Power(pc)) => ConeSpec::Power(pc.alpha),
            })
            .collect()
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

/// SOC dimension at or below which the `(z,z)` block is assembled as a dense
/// lower triangle rather than the sparse diagonal+rank-1 auxiliary-variable
/// form. The dense triangle holds `m(m+1)/2` nonzeros vs the aux form's
/// `2m+1` (plus one extra matrix row/col), so for `m <= 3` dense is the
/// smaller — and slightly better-conditioned near the cone boundary —
/// representation. Above this, the aux form avoids the dense `O(m²)` fill that
/// the review (L41) flagged for large SOCs mixed with exp cones.
const SOC_DENSE_MAX_DIM: usize = 3;

/// Multiple of `tol` at which a *scale-relative* true conic KKT residual is
/// tight enough to promote a `near_opt` breakdown / iteration-limit iterate from
/// `OptimalInaccurate` to `Optimal` (see the post-loop adjudication in
/// [`run_nonsym`]). At the default `tol = 1e-8` this is `1e-6`: two orders below
/// the reduced-accuracy salvage band (`√tol = 1e-4`) and one order below the
/// in-loop `near_opt` band (`1e3·tol = 1e-5`), so a genuinely reduced-accuracy
/// point stays `OptimalInaccurate` while a point solved to the exp/power cone's
/// achievable floor (empirically ~1e-7 scale-relative) is certified `Optimal`.
/// Because it multiplies `tol`, a caller-tightened tolerance tightens the
/// promotion gate in lock-step.
const PROMOTE_REL_TOL_FACTOR: f64 = 1e2;

/// Relative slack by which the promotion's dual-feasibility check tolerates the
/// recovered dual `ẑ` lying *outside* `K*`. At a true conic optimum complementary
/// slackness puts `ẑ` on the boundary of `K*` (dual-slack margin → 0), so the
/// membership test must accept the closure `cl(K*)` up to numerical noise rather
/// than demand a strict interior. Kept far below `PROMOTE_REL_TOL_FACTOR·tol` so
/// dual feasibility is verified far more tightly than the overall KKT budget.
const DUAL_CLOSURE_SLACK: f64 = 1e-9;

/// Scale-relative true conic KKT residual of the **un-homogenized** iterate
/// `(x, y, z)/τ`, or `None` when it is not a valid optimality certificate.
///
/// Returns `None` when `τ ≤ 0` (an infeasibility ray, where un-homogenizing is
/// meaningless) or when `ẑ = z/τ` has left the dual cone `K*` — without
/// `ẑ ∈ K*` the residuals below are not a KKT certificate however small they
/// are (this is the safety gate that keeps the promotion off infeasible /
/// unbounded solves). Otherwise it scores the recovered point with
/// [`QpSolution::kkt_residuals_conic`] (each block measured against *its own*
/// cone) and normalizes each residual by the magnitude of its own constituent
/// terms (Clarabel-style), so the certificate is invariant to problem scaling.
fn true_kkt_scale_rel(
    prob: &QpProblem,
    cone: &NsCone,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    tau: f64,
) -> Option<f64> {
    if !(tau > 0.0) {
        return None;
    }
    let inv = 1.0 / tau;
    let z_hat: Vec<f64> = z.iter().map(|v| v * inv).collect();
    // Dual feasibility `ẑ ∈ K*` is required for the residuals below to be a KKT
    // certificate. But at a true optimum complementary slackness places `ẑ` on
    // the *boundary* of `K*` (its dual-slack margin → 0), so the strict-interior
    // `in_dual_cone(·, +tol)` test is the wrong one here — it rejects exactly the
    // duals optimality produces (the p=4 power ball's dual sits ~1e-11 inside the
    // boundary, well under a 1e-9 interior margin). Test membership in the
    // *closure* instead, tolerating a small numerical slack outside: a negative
    // tolerance turns `in_dual_cone`'s interior margin (`margin > tol·(1+‖·‖)`)
    // into the relaxed-closure test `margin > −slack·(1+‖·‖)`, so `ẑ` may lie up
    // to `DUAL_CLOSURE_SLACK` (relative) outside `K*`. That slack is far tighter
    // than the scale-relative promotion budget, so it never admits a dual that
    // is not, to numerical precision, feasible.
    if !cone.in_dual_cone(&z_hat, -DUAL_CLOSURE_SLACK) {
        return None;
    }
    let x_hat: Vec<f64> = x.iter().map(|v| v * inv).collect();
    let y_hat: Vec<f64> = y.iter().map(|v| v * inv).collect();
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();

    // True conic KKT residuals (primal cone-membership, stationarity,
    // per-block complementarity) at the recovered point.
    let candidate = QpSolution {
        status: QpStatus::Optimal,
        x: x_hat.clone(),
        y: y_hat.clone(),
        z: z_hat.clone(),
        z_lb: vec![0.0; n],
        z_ub: vec![0.0; n],
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    };
    let res = candidate.kkt_residuals_conic(prob, &cone.cone_specs());

    // Scale normalizers: each residual by the magnitude of the terms that
    // compose it, so a large-data problem is not held to an unreachable
    // absolute floor and a well-scaled one is not judged loosely.
    let inf = |v: &[f64]| v.iter().fold(0.0f64, |m, &a| m.max(a.abs()));
    let mut px = vec![0.0; n];
    prob.p_mul(&x_hat, &mut px);
    let mut aty = vec![0.0; n];
    prob.at_mul(&y_hat, &mut aty);
    let mut gtz = vec![0.0; n];
    prob.gt_mul(&z_hat, &mut gtz);
    let scale_d = inf(&px).max(inf(&aty)).max(inf(&gtz)).max(inf(&prob.c));
    let mut ax = vec![0.0; m_eq];
    prob.a_mul(&x_hat, &mut ax);
    let mut gx = vec![0.0; m_ineq];
    prob.g_mul(&x_hat, &mut gx);
    let s_hat: Vec<f64> = (0..m_ineq).map(|i| prob.h[i] - gx[i]).collect();
    let scale_p = inf(&ax)
        .max(inf(&gx))
        .max(inf(&s_hat))
        .max(inf(&prob.b))
        .max(inf(&prob.h));
    let obj = 0.5 * dot(&x_hat, &px) + dot(&prob.c, &x_hat);
    let scale_g = obj.abs();

    let pres_rel = res.primal_infeasibility / (1.0 + scale_p);
    let dres_rel = res.dual_infeasibility / (1.0 + scale_d);
    let gap_rel = res.complementarity / (1.0 + scale_g);
    Some(pres_rel.max(dres_rel).max(gap_rel))
}

/// KKT value-array positions for one cone block.
enum ZPos {
    /// Orthant: one diagonal value position per row.
    Diag(Vec<usize>),
    /// Small second-order cone (`m <= SOC_DENSE_MAX_DIM`): the dense
    /// lower-triangle value positions, row-major `[(0,0); (1,0),(1,1); …]`
    /// (length `m(m+1)/2`). For such `m` the dense triangle has *fewer*
    /// nonzeros than the auxiliary-variable form (and is marginally
    /// better-conditioned), so it is preferred.
    SecondOrderDense { dim: usize, pos: Vec<usize> },
    /// Large second-order cone in **diagonal + rank-1** form via one auxiliary
    /// variable `ξ` (the ECOS/Clarabel sparse-SOC trick, matching the
    /// symmetric driver's `ZBlockPos::DiagRank1`): the `(z,z)` diagonal value
    /// positions, the coupling column `(ξ, z_i) = u_i`, and the `(ξ, ξ) = +1`
    /// entry. Eliminating `ξ` reproduces the dense `diag(d) + uuᵀ` block while
    /// keeping the factor sparse — `O(m)` fill, not the dense `O(m²)`.
    SecondOrderSparse {
        diag_pos: Vec<usize>,
        u_pos: Vec<usize>,
        aux_pos: usize,
    },
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
        // (z,z): per block, seeded with −reg on the diagonal. SOC blocks get
        // an auxiliary variable (appended after the base rows) carrying the
        // rank-1 term, so the (z,z) fill is O(m) not the dense O(m²). Exp
        // blocks reserve the strict-lower 3×3 off-diagonals (a genuine dense
        // block).
        let base_dim = n + m_eq + m_ineq;
        let mut aux = base_dim; // next auxiliary-variable index
        for (off, b) in &cone.blocks {
            let zb = n + m_eq + off;
            match b {
                NsBlock::Orthant(d) => {
                    for i in 0..*d {
                        add(zb + i, zb + i, -reg);
                    }
                }
                NsBlock::SecondOrder(m) if *m <= SOC_DENSE_MAX_DIM => {
                    // Small SOC: dense m×m lower triangle for the NT scaling W²
                    // (fewer nonzeros than the aux form at this size).
                    for i in 0..*m {
                        for j in 0..=i {
                            add(zb + i, zb + j, if i == j { -reg } else { 0.0 });
                        }
                    }
                }
                NsBlock::SecondOrder(m) => {
                    // Large SOC: sparse diag-plus-rank-1 via the auxiliary
                    // variable ξ — diagonal −reg (filled per iter), coupling
                    // (ξ, z_i) = u_i, and (ξ, ξ) = +1. Eliminating ξ reproduces
                    // diag(d) + uuᵀ with O(m) fill instead of dense O(m²).
                    for i in 0..*m {
                        add(zb + i, zb + i, -reg);
                        add(aux, zb + i, 0.0);
                    }
                    add(aux, aux, 1.0);
                    aux += 1;
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
        let dim = aux;

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
        let mut aux = base_dim;
        for (off, b) in &cone.blocks {
            let zb = n + m_eq + off;
            match b {
                NsBlock::Orthant(d) => {
                    z_pos.push(ZPos::Diag(
                        (0..*d).map(|i| coord_to_pos[&(zb + i, zb + i)]).collect(),
                    ));
                }
                NsBlock::SecondOrder(m) if *m <= SOC_DENSE_MAX_DIM => {
                    let mut pos = Vec::with_capacity(m * (m + 1) / 2);
                    for i in 0..*m {
                        for j in 0..=i {
                            pos.push(coord_to_pos[&(zb + i, zb + j)]);
                        }
                    }
                    z_pos.push(ZPos::SecondOrderDense { dim: *m, pos });
                }
                NsBlock::SecondOrder(m) => {
                    let diag_pos = (0..*m).map(|i| coord_to_pos[&(zb + i, zb + i)]).collect();
                    let u_pos = (0..*m).map(|i| coord_to_pos[&(aux, zb + i)]).collect();
                    let aux_pos = coord_to_pos[&(aux, aux)];
                    z_pos.push(ZPos::SecondOrderSparse {
                        diag_pos,
                        u_pos,
                        aux_pos,
                    });
                    aux += 1;
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
        debug_assert_eq!(aux, dim, "aux count must match between the two passes");
        NsKkt {
            airn,
            ajcn,
            values,
            dim,
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
                (NsBlock::SecondOrder(m), ZPos::SecondOrderDense { dim, pos }) => {
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
                (
                    NsBlock::SecondOrder(m),
                    ZPos::SecondOrderSparse {
                        diag_pos,
                        u_pos,
                        aux_pos,
                    },
                ) => {
                    let sb = &s[*off..off + m];
                    let zb = &z[*off..off + m];
                    // W² = diag(d) + u uᵀ from the SOC's NT scaling.
                    let (diag, u) = match SecondOrderCone::new(*m).kkt_block(sb, zb) {
                        ConeBlock::DiagPlusRank1 { diag, u } => (diag, u),
                        _ => unreachable!("SOC kkt_block is DiagPlusRank1"),
                    };
                    // (z,z) = −(diag(d) + uuᵀ) − reg, with the rank-1 carried
                    // by the aux variable ξ: diagonal −dᵢ − reg, coupling
                    // (ξ, z_i) = uᵢ, and (ξ, ξ) = +1. Its Schur complement is
                    // −diag(d) − reg − uuᵀ = −W² − reg.
                    for i in 0..*m {
                        out[diag_pos[i]] = -diag[i] - reg;
                        out[u_pos[i]] = u[i];
                    }
                    out[*aux_pos] = 1.0;
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

/// Legacy fixed interior-membership floor for the exp/power backtracking
/// below — the upper bound `nscone_mem_tol` is capped at, so a well-scaled
/// solve (`μ` not yet tiny) sees exactly today's behavior.
const NSCONE_MEM_TOL_CAP: f64 = 1e-12;

/// How far below `μ` the exp/power interior-membership floor sits once `μ`
/// has shrunk past `NSCONE_MEM_TOL_CAP`. Chosen so the floor stays several
/// orders of magnitude under a cone coordinate's own natural central-path
/// scale (empirically `O(μ)`, gh #339's traced repro), while never
/// approaching machine epsilon before `μ` itself does.
const NSCONE_MEM_TOL_MU_FACTOR: f64 = 1e-6;

/// Interior-membership tolerance for [`max_step`]'s exp/power-cone
/// backtracking, scaled by the current barrier parameter `μ` rather than a
/// fixed constant (gh #339).
///
/// A non-symmetric cone coordinate that legitimately tracks `μ` down the
/// central path — e.g. the dual component conjugate to a slack that must
/// vanish at the optimum, which is driven to `0` as `μ → 0` — shrinks *with*
/// `μ`, not independently of it. A fixed absolute floor (the original
/// `1e-12`) is blind to that: once such a coordinate's magnitude drops near
/// the floor, the backtracking line search in `max_step` starts rejecting
/// *any* further legitimate shrinkage as if the point were leaving the cone,
/// collapsing the step length geometrically iteration over iteration until
/// it hits `0` well short of convergence — even though the point remains
/// comfortably interior in a scale-relative sense (its cone-membership slack
/// `ψ`/`ψ*`, tested separately with its own magnitude-relative tolerance, is
/// nowhere near its own zero boundary). This reproduces whenever one
/// cone-triple coordinate is pinned to an extreme value (by data elsewhere in
/// the problem) while a companion coordinate must shrink toward `0`, in any
/// of the three cone positions — not just the `x`-large/`z`-small repro this
/// was diagnosed from.
///
/// Scaling the floor by `μ` (and capping it at the legacy constant) tracks
/// that natural decay rate instead: unaffected while `μ` is not yet tiny
/// (bit-identical to the old fixed `1e-12` for any well-scaled solve), and
/// shrinking in lockstep with `μ` once the central path has driven it far
/// below that — which is exactly the regime where a fixed floor turns into
/// an artificial wall.
#[inline]
fn nscone_mem_tol(mu: f64) -> f64 {
    (mu.max(0.0) * NSCONE_MEM_TOL_MU_FACTOR).min(NSCONE_MEM_TOL_CAP)
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
    mu: f64,
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
    // z ∈ K*), using that block's own cone. The interior-membership floor is
    // μ-relative (gh #339), not the fixed `1e-12` this used to be — see
    // `nscone_mem_tol`.
    let mem_tol = nscone_mem_tol(mu);
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
                if !nscone.in_primal_cone(&sp, mem_tol) || !nscone.in_dual_cone(&zp, mem_tol) {
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
    if bt >= 100 { 0.0 } else { alpha }
}

/// Cone-aware infeasibility detection for the non-symmetric driver.
///
/// Validates the Farkas multiplier `z ∈ K*` and the recession direction
/// `−Gd ∈ K` against the genuine non-symmetric cone instead of the
/// orthant componentwise default. For an exp block the dual cone requires
/// `u < 0`, so the componentwise test both *rejects* genuine exp Farkas
/// certificates (infeasible problems degraded to `IterationLimit`) and
/// *accepts* all-nonnegative `z ∉ K_exp*` (false `PrimalInfeasible`); the
/// recession branch had the analogous flaw via `Gd ≤ 0`.
fn detect_infeasibility_nscone(
    prob: &QpProblem,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    opts: &QpOptions,
    cone: &NsCone,
) -> Option<QpStatus> {
    detect_infeasibility_with(
        prob,
        x,
        y,
        z,
        opts,
        |z, tol| cone.in_dual_cone(z, tol),
        |gd, tol| {
            // `−Gd ∈ K`; the non-symmetric cone is NOT self-dual, so this
            // is the *primal* membership test (distinct from `in_dual_cone`).
            // Use the **closure** test: a recession ray lands on the boundary
            // of `K` (e.g. the exp cone's `y = 0` face), which the strict
            // interior test would reject (gh #283).
            let neg: Vec<f64> = gd.iter().map(|&v| -v).collect();
            cone.in_primal_closure(&neg, tol)
        },
    )
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
        // (within `~1e3·tol`), the iterate is usable — report it as
        // `OptimalInaccurate` (reduced accuracy) rather than discarding it as a
        // spurious NumericalFailure. It is deliberately *not* a bare `Optimal`:
        // the residual sits above `tol`, so callers can distinguish it from a
        // genuinely converged solve (code review 2026-06 item M20).
        let near_opt = res < 1e3 * opts.tol;
        // Infeasibility certificate as τ → 0. Validate the Farkas multiplier
        // and the recession direction against the *actual* (non-symmetric)
        // cone — the orthant-only componentwise test is wrong in both
        // directions for exp/power blocks (`K_exp*` requires `u < 0`).
        if tau < 1e-2 * kappa.max(1.0) {
            if let Some(st) = detect_infeasibility_nscone(prob, &x, &y, &z, opts, &cone) {
                status = st;
                break;
            }
        }

        // Refactor M with the dual-aware scaling.
        kkt_vals.copy_from_slice(&kkt.values);
        let scalings = match kkt.update_blocks(&cone, &s, &z, opts.reg, &mut kkt_vals) {
            Some(sc) => sc,
            None => {
                status = breakdown_status(near_opt);
                break;
            }
        };
        if fact.refactor(&kkt_vals).is_err() {
            status = breakdown_status(near_opt);
            break;
        }

        // Constant direction p: M p = (−c, b, h).
        build_rhs(&prob.c, &neg_b, &neg_h, &zeros_m, n, m_eq, m_ineq, &mut rhs);
        if fact.solve_one(&mut rhs).is_err() {
            status = breakdown_status(near_opt);
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
            status = breakdown_status(near_opt);
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
            max_step(&cone, &s, &ds_aff, &z, &dz_aff, opts.tau, cap, mu)
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
                max_step(&cone, &s, &ds, &z, &dz, opts.tau, cap, mu)
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
            status = breakdown_status(near_opt);
            break;
        }
        if alpha <= 0.0 {
            status = breakdown_status(near_opt);
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
    // genuine failure. Restore that iterate and report `OptimalInaccurate`,
    // mirroring the "solved to reduced accuracy" outcome of ECOS/Clarabel/SCS.
    // It is reported as `OptimalInaccurate`, *not* a bare `Optimal`: the
    // residual is only within `√tol`, so callers can tell it apart from a
    // genuinely converged solve (code review 2026-06 item M20). This never
    // fires for infeasible/unbounded problems (their residuals never get this
    // small — the embedding drives τ → 0 and the certificate path triggers
    // first) and never relaxes the clean convergence test above (still `tol`).
    // Post-loop optimality adjudication. The in-loop test certifies `Optimal`
    // only on an *absolute* KKT residual below `tol`. On a non-symmetric cone
    // (exp/power) the barrier Hessian's conditioning worsens near the boundary
    // (ψ → 0), so that absolute residual can floor a little above `tol` at a
    // genuinely optimal point; the driver then breaks down `near_opt` (labelling
    // the iterate `OptimalInaccurate`) or exhausts its budget. Re-adjudicate the
    // recovered point against the *true, scale-relative* conic KKT residual —
    // the scale-invariant optimality measure for a convex conic program — and
    // promote to `Optimal` only when that certificate is genuinely tight *and*
    // the recovered dual is in `K*`:
    //   * `ẑ ∈ K*` and scale-relative KKT error < `PROMOTE_REL_TOL_FACTOR·tol`
    //       → `Optimal`
    //   * best iterate within the reduced-accuracy band (`√tol`)
    //       → `OptimalInaccurate`
    //   * otherwise the loop's own verdict stands.
    //
    // This never *relaxes* the in-loop `tol` test (that already returned) and is
    // strictly safe: the promotion is gated on `ẑ ∈ K*` (dual feasibility) plus
    // a scale-relative residency two orders below the reduced-accuracy band, so
    // for a convex program — where KKT is sufficient for global optimality — it
    // can only fire at a genuine optimum. Infeasible / unbounded solves
    // terminate with a certificate status (`PrimalInfeasible` / `DualInfeasible`)
    // and never reach here; a τ → 0 ray or an out-of-`K*` dual returns `None`
    // from the certificate and is never promoted.
    if matches!(
        status,
        QpStatus::OptimalInaccurate | QpStatus::NumericalFailure | QpStatus::IterationLimit
    ) {
        let reduced_acc = opts.tol.sqrt();

        // Score both candidate iterates — the loop's final one and the best
        // (lowest in-loop-residual) snapshot — on the *scale-relative* conic KKT
        // residual, the scale-invariant optimality certificate. gh #336: the
        // in-loop residual carries the raw, *unnormalized* complementarity gap
        // `s·z`, whose absolute floor grows with the optimal cone magnitudes, so
        // on extreme-but-legitimate data scaling it stalls well above `tol` even
        // at a point that is primal-feasible, dual-feasible, and objective-
        // correct. The absolute residual therefore must not gate the salvage;
        // `true_kkt_scale_rel` (each residual normalized by its own term
        // magnitudes) can. Adjudicate on whichever iterate certifies tighter.
        let final_rel = true_kkt_scale_rel(prob, &cone, &x, &y, &z, tau);
        let best_rel = best
            .as_ref()
            .and_then(|(bx, by, bz, _bs, bt, _)| true_kkt_scale_rel(prob, &cone, bx, by, bz, *bt));
        let use_best = match (best_rel, final_rel) {
            (Some(b), Some(f)) => b < f,
            (Some(_), None) => true,
            _ => false,
        };
        // Absolute reduced-accuracy fallback (the prior salvage path): a
        // well-scaled solve that stalled a hair short of `tol` still salvages,
        // even when its scale-relative certificate is unavailable (e.g. the
        // recovered dual sits just outside `K*`).
        let abs_salvage = best_res < reduced_acc;
        // Restore the best snapshot when it is the point we adjudicate — either it
        // certifies tighter than the final iterate, or the absolute fallback fires
        // on it. κ is not read downstream (the recovery un-homogenizes by 1/τ);
        // restoring x/y/z/s/τ is what the solution recovery and post-mortem hook
        // consume.
        let restore = use_best || abs_salvage;
        if restore {
            if let Some((bx, by, bz, bs, btau, _bkappa)) = best.take() {
                x = bx;
                y = by;
                z = bz;
                s = bs;
                tau = btau;
            }
        }
        // Certificate of the point we actually return.
        let rel = if restore { best_rel } else { final_rel };
        status = match rel {
            // Genuinely tight scale-relative certificate at a dual-feasible point
            // (the `ẑ ∈ K*` gate lives in `true_kkt_scale_rel`): a clean
            // `Optimal`.
            Some(e) if e < PROMOTE_REL_TOL_FACTOR * opts.tol => QpStatus::Optimal,
            // Primal- and dual-feasible with a *normalized* gap only moderately
            // above `tol` — the accuracy plateau of a boundary-riding
            // non-symmetric cone under extreme scaling. Usable at reduced
            // accuracy: report `OptimalInaccurate` (matching the symmetric SOC
            // driver) rather than discarding a correct answer as a spurious
            // `NumericalFailure` (gh #336).
            Some(e) if e < reduced_acc => QpStatus::OptimalInaccurate,
            // Absolute-residual salvage for the well-scaled near-`tol` stall.
            _ if abs_salvage => QpStatus::OptimalInaccurate,
            _ => status,
        };
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
        // Trivial dual: `z = 0` (the cone apex) is valid in every dual cone,
        // unlike the all-ones vector, which is not a member of an SOC of
        // dimension ≥ 3. Matches `ipm::failed_solution`.
        z: vec![0.0; prob.m_ineq()],
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
        // This exp-cone GP reaches its optimum through the driver's
        // reduced-accuracy fallback (best iterate within √tol), so the status
        // is `OptimalInaccurate` — a usable solve at reduced accuracy, not a
        // failure. The objective check below pins the actual solution quality.
        assert!(
            matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate),
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
            // A bad warm start can land on the reduced-accuracy fallback
            // (`OptimalInaccurate`); both count as a usable solve. The
            // start-independent invariant is the objective, checked next.
            assert!(
                matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate),
                "warm {warm:?}: {:?}",
                sol.status
            );
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

    /// L41: a **large** SOC must be assembled in the sparse diag-plus-rank-1
    /// (auxiliary-variable) form — `O(m)` fill — not the dense `m×m` lower
    /// triangle (`O(m²)`). The KKT dimension grows by exactly one aux variable
    /// for the cone, and the total nnz stays below the dense `(z,z)` count.
    #[test]
    fn large_soc_kkt_block_is_sparse_not_dense() {
        // Pure SOC(m) problem (m > SOC_DENSE_MAX_DIM): n = m, G = -I_m.
        let m = 24;
        let g: Vec<Triplet> = (0..m).map(|i| Triplet::new(i, i, -1.0)).collect();
        let prob = QpProblem {
            n: m,
            p_lower: vec![],
            c: vec![0.0; m],
            a: vec![],
            b: vec![],
            g,
            h: vec![0.0; m],
            lb: vec![],
            ub: vec![],
        };
        let cone = NsCone::new(&[NsBlock::SecondOrder(m)]);
        let kkt = NsKkt::build(&prob, &cone, 1e-10);

        // One auxiliary variable was appended for the single large SOC.
        assert_eq!(
            kkt.dim,
            prob.n + prob.m_eq() + prob.m_ineq() + 1,
            "large SOC must add exactly one auxiliary variable",
        );
        assert!(matches!(
            kkt.z_pos.as_slice(),
            [ZPos::SecondOrderSparse { .. }]
        ));
        // The dense lower triangle alone would be m(m+1)/2 entries in the
        // (z,z) block; the sparse aux form uses ~2m. The full KKT nnz must sit
        // below the dense-(z,z) lower bound.
        let dense_zz = m * (m + 1) / 2;
        assert!(
            kkt.airn.len() < dense_zz,
            "KKT nnz {} should be below the dense (z,z) count {} (sparse SOC fill)",
            kkt.airn.len(),
            dense_zz,
        );
    }

    /// The sparse aux-variable SOC path must also **solve correctly**, not
    /// just assemble sparsely: a large second-order cone routed through the
    /// non-symmetric driver hits the new code path (`m > SOC_DENSE_MAX_DIM`)
    /// and must reach the known norm-minimization optimum.
    /// `min t s.t. (t, 3, 4, 0, 0, 0) ∈ SOC(6)` → `t = ‖(3,4,0,0,0)‖ = 5`.
    #[test]
    fn large_soc_sparse_path_solves() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0)], // SOC s0 = t
            h: vec![0.0, 3.0, 4.0, 0.0, 0.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        // dim 6 > SOC_DENSE_MAX_DIM ⇒ exercises the sparse aux path.
        let sol = solve_conic_hsde_nonsym(&prob, &[NsBlock::SecondOrder(6)], &opts(), backend);
        assert!(
            matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate),
            "status {:?}",
            sol.status
        );
        assert!((sol.x[0] - 5.0).abs() < 1e-5, "t = {} vs 5", sol.x[0]);
    }

    /// A **small** SOC (`m <= SOC_DENSE_MAX_DIM`) stays in the dense
    /// lower-triangle form: fewer nonzeros than the aux form at that size, no
    /// auxiliary variable, and the existing small-SOC solves keep their
    /// (better-conditioned) numerics.
    #[test]
    fn small_soc_kkt_block_stays_dense() {
        let m = SOC_DENSE_MAX_DIM; // 3
        let g: Vec<Triplet> = (0..m).map(|i| Triplet::new(i, i, -1.0)).collect();
        let prob = QpProblem {
            n: m,
            p_lower: vec![],
            c: vec![0.0; m],
            a: vec![],
            b: vec![],
            g,
            h: vec![0.0; m],
            lb: vec![],
            ub: vec![],
        };
        let cone = NsCone::new(&[NsBlock::SecondOrder(m)]);
        let kkt = NsKkt::build(&prob, &cone, 1e-10);

        // No auxiliary variable for a small SOC.
        assert_eq!(
            kkt.dim,
            prob.n + prob.m_eq() + prob.m_ineq(),
            "small SOC must not add an auxiliary variable",
        );
        assert!(matches!(
            kkt.z_pos.as_slice(),
            [ZPos::SecondOrderDense { dim: 3, .. }]
        ));
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

    // ---- H8: non-symmetric Farkas/recession certificates must use the
    // actual cone, not the orthant componentwise test. ----

    /// `NsCone`'s membership tests must agree with the exp cone's own
    /// `K_exp*` test (`u < 0`), where the componentwise `z ≥ 0` disagrees
    /// in both directions.
    #[test]
    fn nscone_exp_membership_disagrees_with_componentwise() {
        let cone = NsCone::new(&[NsBlock::exp()]);
        // Self-dual interior reference: in both K and K* (and has u < 0).
        let mut e = [0.0; 3];
        cone.identity(&mut e);
        assert!(cone.in_dual_cone(&e, 1e-9), "interior ref must be in K*");
        assert!(cone.in_primal_cone(&e, 1e-9), "interior ref must be in K");
        assert!(e[0] < 0.0, "dual exp interior has u < 0, got {}", e[0]);

        // All-nonnegative point: passes componentwise z ≥ 0 but u = 1 > 0
        // ⇒ NOT in K_exp*.
        let allpos = [1.0, 1.0, 1.0];
        assert!(
            allpos.iter().all(|&v| v >= 0.0),
            "componentwise nonneg holds"
        );
        assert!(
            !cone.in_dual_cone(&allpos, 1e-9),
            "(1,1,1) has u>0 ⇒ not in K_exp*"
        );
    }

    /// **False negative** (the headline H8 bug): a genuine exp Farkas
    /// multiplier has `u < 0`, so the orthant componentwise test rejects
    /// it and an infeasible problem degrades to `IterationLimit`. The
    /// cone-aware detector accepts it as `PrimalInfeasible`.
    ///
    /// Setup isolates the primal branch: `A` empty, `y` empty, `G = 0`
    /// (so `Gᵀz = 0` trivially), and `h` chosen so `hᵀz < 0`.
    #[test]
    fn exp_farkas_certificate_rejected_componentwise_accepted_cone_aware() {
        use crate::ipm::detect_infeasibility;
        let cone = NsCone::new(&[NsBlock::exp()]);
        let mut zc = [0.0; 3];
        cone.identity(&mut zc); // in K*, u < 0
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![0.0],
            a: vec![],
            b: vec![],
            g: vec![],              // G = 0 ⇒ Gᵀz = 0
            h: vec![1.0, 0.0, 0.0], // hᵀz = z₀ < 0
            lb: vec![],
            ub: vec![],
        };
        let opts = QpOptions::default();
        let x = [0.0]; // skip the dual-inf branch
        let y: [f64; 0] = [];
        assert!(zc[0] < 0.0 && prob.h[0] > 0.0, "hᵀz = z₀ < 0");

        // Componentwise (orthant) test: z₀ < 0 ⇒ rejects the genuine cert.
        assert_eq!(
            detect_infeasibility(&prob, &x, &y, &zc, &opts),
            None,
            "orthant test wrongly rejects a real exp Farkas certificate"
        );
        // Cone-aware: z ∈ K_exp* ⇒ verified PrimalInfeasible.
        assert_eq!(
            detect_infeasibility_nscone(&prob, &x, &y, &zc, &opts, &cone),
            Some(QpStatus::PrimalInfeasible),
            "cone-aware test must accept the genuine exp Farkas certificate"
        );
    }

    /// **False positive**: an all-nonnegative `z ∉ K_exp*` is accepted by
    /// the componentwise test (bogus `PrimalInfeasible`) but rejected by
    /// the cone-aware one.
    #[test]
    fn nonneg_z_not_in_dual_exp_cone_is_false_positive_componentwise() {
        use crate::ipm::detect_infeasibility;
        let cone = NsCone::new(&[NsBlock::exp()]);
        let z = [1.0, 1.0, 1.0]; // u = 1 > 0 ⇒ ∉ K_exp*, but ≥ 0 componentwise
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![0.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![-1.0, 0.0, 0.0], // hᵀz = −z₀ = −1 < 0
            lb: vec![],
            ub: vec![],
        };
        let opts = QpOptions::default();
        let x = [0.0];
        let y: [f64; 0] = [];
        assert_eq!(
            detect_infeasibility(&prob, &x, &y, &z, &opts),
            Some(QpStatus::PrimalInfeasible),
            "componentwise test FALSE-positives on z=(1,1,1) ∉ K_exp*"
        );
        assert_eq!(
            detect_infeasibility_nscone(&prob, &x, &y, &z, &opts, &cone),
            None,
            "cone-aware test must reject z=(1,1,1): not in K_exp*"
        );
    }

    // ---- gh #283: cone-domain infeasibility and recession/unboundedness ----

    /// **Case 1 (unbounded).** `min u s.t. (u, 1, t) ∈ K_exp` is unbounded
    /// below: the cone forces `t ≥ e^u`, so `u → −∞` (with `t → 0⁺`) stays
    /// feasible while the objective diverges. The recession ray `−Gd = (u<0,
    /// 0, t≥0)` lands on the exp cone's `y = 0` face; the strict-interior
    /// membership test rejected it (→ `NumericalFailure`), the **closure**
    /// test accepts it → `DualInfeasible`.
    #[test]
    fn unbounded_exp_cone_reports_dual_infeasible() {
        use crate::cones::ConeSpec;
        use crate::ipm::solve_socp_ipm;
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![1.0, 0.0], // min u
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0), Triplet::new(2, 1, -1.0)],
            h: vec![0.0, 1.0, 0.0], // slack = (u, 1, t)
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_socp_ipm(&prob, &[ConeSpec::Exponential], &opts(), backend);
        assert_eq!(
            sol.status,
            QpStatus::DualInfeasible,
            "unbounded exp program must certify DualInfeasible, got {:?}",
            sol.status
        );
    }

    /// The **closure** recession test accepts the exp cone's `y = 0` boundary
    /// face (where a recession ray lands) that the strict-interior test rejects.
    #[test]
    fn exp_closure_accepts_recession_face_interior_rejects() {
        let cone = NsCone::new(&[NsBlock::exp()]);
        // (x ≤ 0, y = 0, z ≥ 0) is in cl(K_exp) but not its interior.
        let face = [-2.1, 0.0, 0.49];
        assert!(
            cone.in_primal_closure(&face, 1e-7),
            "recession face must be in cl(K_exp)"
        );
        assert!(
            !cone.in_primal_cone(&face, 1e-9),
            "recession face must NOT be in the strict interior"
        );
        // A point genuinely outside the closure (x > 0 on the y = 0 face) is
        // still rejected — the closure test is not a blanket accept.
        assert!(!cone.in_primal_closure(&[2.0, 0.0, 1.0], 1e-7));
    }

    /// **Cases 2/3 (power-cone domain violation).** `K_0.5` on `(w, t−2, 1)`
    /// with `t ≤ T < 2` forces the cone's `y`-slack `t − 2 ≤ T − 2 < 0`,
    /// violating the `y ≥ 0` domain at every point ⇒ primal infeasible. The
    /// residual-gated Farkas detector stalls short of certifying; the setup
    /// cone-domain screen reports it directly.
    #[test]
    fn power_cone_domain_violation_reports_primal_infeasible() {
        use crate::cones::ConeSpec;
        use crate::ipm::solve_socp_ipm;
        for t_cap in [1.0_f64, 1.999] {
            let prob = QpProblem {
                n: 2,
                p_lower: vec![],
                c: vec![1.0, 0.0], // min w
                a: vec![],
                b: vec![],
                g: vec![
                    Triplet::new(0, 0, -1.0), // s0 = w
                    Triplet::new(1, 1, -1.0), // s1 = t − 2
                    Triplet::new(3, 1, 1.0),  // s3 = T − t ≥ 0
                ],
                h: vec![0.0, -2.0, 1.0, t_cap], // s2 = 1 (const)
                lb: vec![],
                ub: vec![],
            };
            let specs = [ConeSpec::Power(0.5), ConeSpec::Nonneg(1)];
            let sol = solve_socp_ipm(&prob, &specs, &opts(), backend);
            assert_eq!(
                sol.status,
                QpStatus::PrimalInfeasible,
                "power domain violation (T={t_cap}) must be PrimalInfeasible, got {:?}",
                sol.status
            );
        }
    }

    /// A cone-domain coordinate pinned to a **constant** outside the domain
    /// (visible in `h` alone: `(w, −1, 1)` for power, likewise for exp) is
    /// primal infeasible and caught by the setup screen with no solve.
    #[test]
    fn constant_domain_coordinate_reports_primal_infeasible() {
        use crate::cones::ConeSpec;
        use crate::ipm::solve_socp_ipm;
        for spec in [ConeSpec::Power(0.5), ConeSpec::Exponential] {
            let prob = QpProblem {
                n: 1,
                p_lower: vec![],
                c: vec![1.0],
                a: vec![],
                b: vec![],
                g: vec![Triplet::new(0, 0, -1.0)],
                h: vec![0.0, -1.0, 1.0], // slack = (w, −1, 1); y = −1 < 0
                lb: vec![],
                ub: vec![],
            };
            let sol = solve_socp_ipm(&prob, &[spec], &opts(), backend);
            assert_eq!(
                sol.status,
                QpStatus::PrimalInfeasible,
                "constant domain violation ({spec:?}) must be PrimalInfeasible, got {:?}",
                sol.status
            );
        }
    }

    /// **No false positive.** A power program whose cone `y`-slack reaches
    /// `+0.05` (`T = 2.05`, just off the domain boundary) is *feasible* and
    /// bounded — the cone-domain screen must NOT flag it, and it solves to the
    /// known optimum `min w = −√(0.05) ≈ −0.2236`.
    #[test]
    fn feasible_power_near_domain_boundary_stays_optimal() {
        use crate::cones::ConeSpec;
        use crate::ipm::solve_socp_ipm;
        let blocks = [NsBlock::power(0.5), NsBlock::Orthant(1)];
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![1.0, 0.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, -1.0),
                Triplet::new(1, 1, -1.0),
                Triplet::new(3, 1, 1.0),
            ],
            h: vec![0.0, -2.0, 1.0, 2.05],
            lb: vec![],
            ub: vec![],
        };
        assert!(
            !detect_cone_domain_infeasible(&prob, &blocks),
            "feasible near-boundary power program must NOT be flagged infeasible"
        );
        let specs = [ConeSpec::Power(0.5), ConeSpec::Nonneg(1)];
        let sol = solve_socp_ipm(&prob, &specs, &opts(), backend);
        assert!(
            matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate),
            "feasible power program must solve, got {:?}",
            sol.status
        );
        let want = -(0.05_f64).sqrt();
        assert!((sol.obj - want).abs() < 1e-5, "obj = {} vs {want}", sol.obj);
    }

    /// The cone-domain screen must not flag well-posed feasible programs: a
    /// geometric program (`min e^u + e^{−u} = 2`, all exp cones, feasible) and
    /// a feasible power instance both return `false`.
    #[test]
    fn cone_domain_screen_passes_feasible_programs() {
        // GP min e^u + e^{−u} (two exp cones), feasible & bounded.
        let gp = QpProblem {
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
        assert!(!detect_cone_domain_infeasible(
            &gp,
            &[NsBlock::exp(), NsBlock::exp()]
        ));
        // Feasible power: (x, 2, 0.5) ∈ K_α, domain slacks constant & positive.
        let pw = QpProblem {
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
        assert!(!detect_cone_domain_infeasible(&pw, &[NsBlock::power(0.5)]));
    }

    // ---- gh #329: the scale-relative optimality certificate that gates the
    // OptimalInaccurate → Optimal promotion. Its safety rests on two hard gates
    // (`τ > 0` and `ẑ ∈ cl(K*)`); these pin them directly. ----

    /// An infeasibility ray (`τ ≤ 0`) has no meaningful un-homogenized point, so
    /// the certificate returns `None` and can never promote a ray to `Optimal`.
    #[test]
    fn scale_rel_cert_rejects_tau_nonpositive() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0)],
            h: vec![0.0, 1.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let cone = NsCone::new(&[NsBlock::exp()]);
        let z = [0.0, 0.0, 0.0];
        assert_eq!(
            true_kkt_scale_rel(&prob, &cone, &[0.0], &[], &z, 0.0),
            None,
            "τ = 0 must yield no certificate"
        );
        assert_eq!(
            true_kkt_scale_rel(&prob, &cone, &[0.0], &[], &z, -1.0),
            None,
            "τ < 0 must yield no certificate"
        );
    }

    /// A dual `ẑ` well outside `K*` is not a KKT certificate however small the
    /// other residuals; the `ẑ ∈ cl(K*)` gate returns `None` (never promoted).
    /// The exp dual `K_exp*` requires `u < 0`, so `(1, 1, 1)` is firmly outside.
    #[test]
    fn scale_rel_cert_rejects_dual_outside_cone() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![0.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0)],
            h: vec![0.0, 1.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let cone = NsCone::new(&[NsBlock::exp()]);
        let z_bad = [1.0, 1.0, 1.0]; // u = 1 > 0 ⇒ not in K_exp*
        assert!(
            !cone.in_dual_cone(&z_bad, -DUAL_CLOSURE_SLACK),
            "sanity: (1,1,1) is outside cl(K_exp*)"
        );
        assert_eq!(
            true_kkt_scale_rel(&prob, &cone, &[0.0], &[], &z_bad, 1.0),
            None,
            "a dual outside K* must yield no certificate"
        );
    }

    /// The closure gate accepts a dual *on* the boundary of `K*` (where a true
    /// conic optimum's dual sits by complementary slackness) that the old
    /// strict-interior `+1e-9` test rejected — this is exactly what unblocks the
    /// gh #329 promotion — while a dual a finite distance outside stays rejected.
    #[test]
    fn dual_closure_gate_accepts_boundary_rejects_exterior() {
        let cone = NsCone::new(&[NsBlock::power(0.5)]);
        // Boundary of K_{0.5}*: |u| = (v/α)^α (w/(1−α))^(1−α) = sqrt(v·w)·2.
        // Take v = w = 0.5 ⇒ bound = sqrt(0.25)·2 = 1.0; u = 1.0 is on ∂K*.
        let on_boundary = [1.0, 0.5, 0.5];
        assert!(
            !cone.in_dual_cone(&on_boundary, 1e-9),
            "strict-interior test rejects the boundary dual (the gh #329 defect)"
        );
        assert!(
            cone.in_dual_cone(&on_boundary, -DUAL_CLOSURE_SLACK),
            "closure gate must accept the boundary dual"
        );
        // A dual well outside (u = 1.5 > bound = 1.0) is still rejected.
        let exterior = [1.5, 0.5, 0.5];
        assert!(
            !cone.in_dual_cone(&exterior, -DUAL_CLOSURE_SLACK),
            "closure gate must still reject an exterior dual"
        );
    }
}
