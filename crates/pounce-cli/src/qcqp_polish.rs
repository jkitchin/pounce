//! Polishing the conic QCQP arm's answer in the **original** problem's
//! terms (gh #1114).
//!
//! `cvx-qcqp` solves a convex QCQP as an SOCP: each quadratic row becomes a
//! second-order cone (`qp_extract::extract_socp_with_map`). What the conic
//! IPM certifies is the *conic* problem's KKT conditions to its tolerance,
//! and two things about that are weaker than they look from the original
//! QCQP's side:
//!
//! * **The multipliers.** A quadratic row's multiplier is read off its cone
//!   dual as `λ = (z₀ + z₁)/κ` ([`crate::qp_extract::recover_socp_duals`]).
//!   That is the original multiplier only when the cone dual is *aligned*
//!   with the cone slack — exact complementarity, `s ∘ z = 0`. The IPM stops
//!   on the scalar gap `sᵀz`, which is small long before the alignment is,
//!   so the misaligned remainder of `z`, and the part of stationarity it
//!   carried, is dropped. On `min (x₀−2)² + x₁²  s.t.  ‖x‖² ≤ 1,
//!   x₀ + x₁ = 1` the mapped multipliers were `(1.000006, −1.2e-5)` against
//!   the unique `(1, 0)`: a `1.2e-5` Lagrangian-gradient residual reported as
//!   `final_dual_inf = 1.3e-11`.
//! * **The point.** The gap certifies the objective, not `x` to the same
//!   digits: near a curved active constraint `x` is only `~√gap`-accurate.
//!   Measured across the fixture corpus, the original stationarity residual
//!   of the returned `(x, λ)` — even with the best multipliers for that `x`
//!   — was `9e-5` on `airport`, `1e-5` on `qcqp_columns_illcond`, `3e-7` on
//!   `qcqp_ball`, all ending `SolveSucceeded` at `tol = 1e-8`.
//!
//! [`polish_qcqp`] repairs both, in the original problem: it re-fits the
//! multipliers to the original stationarity at the returned `x`
//! ([`refit`]), then takes Newton steps on the original KKT system restricted
//! to the active set the solve identified ([`newton_step`]), keeping each
//! only if the original KKT error falls. The result is reported against the
//! original problem ([`QcqpResiduals`]), so `final_dual_inf` describes the
//! `(x, λ)` actually handed back.

use crate::nl_reader::NlProblem;
use crate::qp_extract::{ConSocpMap, recover_socp_duals};
use pounce_common::types::Index;
use pounce_convex::{ConeSpec, QpProblem, QpSolution, QpStatus};
use pounce_linsol::{Factorization, SparseSymLinearSolverInterface};
use std::collections::BTreeMap;

/// An inequality multiplier at or below this fraction of the largest
/// multiplier is read as inactive: frozen by the re-fit, and left out of the
/// Newton step's active set.
///
/// Freezing, rather than the small weight a multiplier-proportional rule
/// would give it, matters when `x` carries a residual no honest multiplier
/// removes: a minimum-norm fit that may touch every column parks it on an
/// inactive row instead. Measured on a 3-variable ball-plus-plane model, an
/// inactive range row went from `−3e-9` to `−1.6e-5`, "zeroing" a residual
/// the returned `x` genuinely carried.
const ACTIVE_REL: f64 = 1e-6;

/// Newton steps attempted on the active-set KKT system. Each is kept only
/// if it lowers the original KKT error, and the conic point is already close,
/// so convergence is quadratic from the first step; three covers a step
/// spent recovering the regularization's `O(δ)` error.
const MAX_NEWTON: usize = 3;

type SpVec = Vec<(usize, f64)>;

/// One quadratic row's cone block: `G` rows `row0 .. row0 + dim`, of which
/// the first two carry `a_⊥/κ` and the rest `−√2·F` (see
/// `extract_socp_with_map`).
struct Quad {
    con_idx: usize,
    row0: usize,
    dim: usize,
    kappa: f64,
}

/// The original QCQP, read back out of its conic standard form.
struct View<'a> {
    qp: &'a QpProblem,
    a_rows: Vec<SpVec>,
    g_rows: Vec<SpVec>,
    /// Which `G` rows are linear inequalities (the orthant block).
    nonneg: Vec<bool>,
    quads: Vec<Quad>,
}

/// Multipliers of the original problem, in the solver's internal-minimize
/// convention: `Px + c + Aᵀy + Σ zᵢGᵢ + Σ λⱼ∇gⱼ − z_lb + z_ub = 0`. `z` is
/// meaningful only on orthant rows, `quad` is indexed like `View::quads`.
#[derive(Clone, Debug)]
struct Mults {
    y: Vec<f64>,
    z: Vec<f64>,
    quad: Vec<f64>,
    z_lb: Vec<f64>,
    z_ub: Vec<f64>,
}

impl Mults {
    fn scale(&self) -> f64 {
        [&self.y, &self.z, &self.quad, &self.z_lb, &self.z_ub]
            .iter()
            .flat_map(|v| v.iter())
            .fold(1.0_f64, |m, v| m.max(v.abs()))
    }
}

/// The KKT residuals of `(x, multipliers)` measured against the original
/// QCQP (the `bound_relax_factor`-widened one the solver was handed, as the
/// conic arm's own residuals are).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QcqpResiduals {
    /// `max(|Ax − b|, (Gx − h)⁺, g(x)⁺, bound violation)`.
    pub primal_infeasibility: f64,
    /// The variable-box part of [`Self::primal_infeasibility`].
    pub bound_violation: f64,
    /// `‖∇ₓL‖∞`.
    pub dual_infeasibility: f64,
    /// `max |multiplier · constraint value|`.
    pub complementarity: f64,
}

impl QcqpResiduals {
    pub fn kkt_error(&self) -> f64 {
        self.primal_infeasibility
            .max(self.dual_infeasibility)
            .max(self.complementarity)
    }
}

/// The polished point and multipliers ([`polish_qcqp`]).
#[derive(Clone, Debug)]
pub struct QcqpPolish {
    /// The returned point, after any accepted Newton steps.
    pub x: Vec<f64>,
    /// `½xᵀPx + cᵀx` moved by the change in `x`, in the convention of
    /// `QpSolution::obj`.
    pub obj: f64,
    y: Vec<f64>,
    z: Vec<f64>,
    /// Indexed like `con_map`; meaningful at its `Quad` slots.
    quad_by_con: Vec<f64>,
    /// Internal-minimize bound multipliers, length `n`.
    pub z_lb: Vec<f64>,
    pub z_ub: Vec<f64>,
    /// The original problem's residuals at the returned point.
    pub residuals: QcqpResiduals,
    /// Newton steps accepted.
    pub newton_steps: usize,
}

impl QcqpPolish {
    /// Per-`.nl`-constraint duals in POUNCE's `.sol` convention, mapped as
    /// [`recover_socp_duals`] maps them but with a quadratic row's
    /// multiplier read from the polish rather than the cone dual.
    pub fn constraint_duals(&self, prob: &NlProblem, con_map: &[ConSocpMap]) -> Vec<f64> {
        let sign = if prob.minimize { 1.0 } else { -1.0 };
        let mut lambda = recover_socp_duals(prob, con_map, &self.y, &self.z);
        for (i, m) in con_map.iter().enumerate() {
            if matches!(m, ConSocpMap::Quad { .. }) {
                lambda[i] = sign * self.quad_by_con[i];
            }
        }
        lambda
    }
}

fn row_dot(row: &[(usize, f64)], x: &[f64]) -> f64 {
    row.iter().map(|&(j, v)| v * x[j]).sum()
}

impl<'a> View<'a> {
    fn new(qp: &'a QpProblem, cones: &[ConeSpec], con_map: &[ConSocpMap]) -> Self {
        let m_ineq = qp.m_ineq();
        let mut a_rows: Vec<SpVec> = vec![Vec::new(); qp.m_eq()];
        for t in &qp.a {
            a_rows[t.row].push((t.col, t.val));
        }
        let mut g_rows: Vec<SpVec> = vec![Vec::new(); m_ineq];
        for t in &qp.g {
            g_rows[t.row].push((t.col, t.val));
        }
        let mut nonneg = vec![false; m_ineq];
        let mut soc_dim_at: BTreeMap<usize, usize> = BTreeMap::new();
        let mut off = 0usize;
        for spec in cones {
            let dim = spec.dim().min(m_ineq.saturating_sub(off));
            match spec {
                ConeSpec::Nonneg(_) => nonneg[off..off + dim].iter_mut().for_each(|b| *b = true),
                _ => {
                    soc_dim_at.insert(off, dim);
                }
            }
            off += dim;
        }
        // Rows past the cone list are the orthant.
        nonneg[off..].iter_mut().for_each(|b| *b = true);
        let quads = con_map
            .iter()
            .enumerate()
            .filter_map(|(con_idx, m)| match *m {
                ConSocpMap::Quad { z_row0, kappa, .. } => Some(Quad {
                    con_idx,
                    row0: z_row0,
                    dim: soc_dim_at.get(&z_row0).copied().unwrap_or(2).max(2),
                    kappa,
                }),
                _ => None,
            })
            .collect();
        View {
            qp,
            a_rows,
            g_rows,
            nonneg,
            quads,
        }
    }

    fn has_lb(&self, j: usize) -> bool {
        self.qp.lb_of(j) > -1e19
    }

    fn has_ub(&self, j: usize) -> bool {
        self.qp.ub_of(j) < 1e19
    }

    fn slack(&self, r: usize, x: &[f64]) -> f64 {
        self.qp.h[r] - row_dot(&self.g_rows[r], x)
    }

    /// `g(x) = ½‖Fx + w‖² + a_⊥ᵀx + b`, from the cone slacks: the `F` rows
    /// hold `sₖ = √2(Fx + w)ₖ` and `s₀ + s₁ = 2s/κ` with `s = −(a_⊥ᵀx + b)`.
    fn quad_value(&self, q: &Quad, x: &[f64]) -> f64 {
        let s0 = self.slack(q.row0, x);
        let s1 = self.slack(q.row0 + 1, x);
        let sq: f64 = (q.row0 + 2..q.row0 + q.dim)
            .map(|r| self.slack(r, x).powi(2))
            .sum();
        0.25 * sq - 0.5 * q.kappa * (s0 + s1)
    }

    /// `∇g = Fᵀ(Fx + w) + a_⊥ = κ·G₀ − ½ Σₖ sₖ·Gₖ`, `Gₖ = −√2·Fₖ`.
    fn quad_grad(&self, q: &Quad, x: &[f64]) -> SpVec {
        let mut grad: BTreeMap<usize, f64> = BTreeMap::new();
        for &(j, v) in &self.g_rows[q.row0] {
            *grad.entry(j).or_insert(0.0) += q.kappa * v;
        }
        for r in q.row0 + 2..q.row0 + q.dim {
            let s = self.slack(r, x);
            for &(j, v) in &self.g_rows[r] {
                *grad.entry(j).or_insert(0.0) -= 0.5 * s * v;
            }
        }
        grad.into_iter().collect()
    }

    /// Lower triangle of `Q = FᵀF = ½ Σₖ GₖᵀGₖ`, scaled by `weight`, added
    /// into `h`.
    fn add_quad_hessian(&self, q: &Quad, weight: f64, h: &mut BTreeMap<(usize, usize), f64>) {
        for r in q.row0 + 2..q.row0 + q.dim {
            let row = &self.g_rows[r];
            for &(i, a) in row {
                for &(j, b) in row {
                    if i >= j {
                        *h.entry((i, j)).or_insert(0.0) += 0.5 * weight * a * b;
                    }
                }
            }
        }
    }

    /// `Px + c`.
    fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut g = vec![0.0; self.qp.n];
        self.qp.p_mul(x, &mut g);
        for (gi, ci) in g.iter_mut().zip(&self.qp.c) {
            *gi += ci;
        }
        g
    }

    /// `½xᵀPx + cᵀx`.
    fn objective(&self, x: &[f64]) -> f64 {
        let mut px = vec![0.0; self.qp.n];
        self.qp.p_mul(x, &mut px);
        x.iter()
            .zip(&px)
            .zip(&self.qp.c)
            .map(|((xi, pi), ci)| 0.5 * xi * pi + ci * xi)
            .sum()
    }

    fn residuals(&self, x: &[f64], m: &Mults) -> QcqpResiduals {
        let qp = self.qp;
        let n = qp.n;
        let mut r = self.objective_gradient(x);
        let mut primal = 0.0_f64;
        let mut compl = 0.0_f64;
        for (i, row) in self.a_rows.iter().enumerate() {
            primal = primal.max((row_dot(row, x) - qp.b[i]).abs());
            for &(j, v) in row {
                r[j] += m.y[i] * v;
            }
        }
        for (i, row) in self.g_rows.iter().enumerate() {
            if !self.nonneg[i] {
                continue;
            }
            let s = self.slack(i, x);
            primal = primal.max((-s).max(0.0));
            compl = compl.max((m.z[i] * s).abs());
            for &(j, v) in row {
                r[j] += m.z[i] * v;
            }
        }
        for (k, q) in self.quads.iter().enumerate() {
            let g = self.quad_value(q, x);
            primal = primal.max(g.max(0.0));
            compl = compl.max((m.quad[k] * g).abs());
            for (j, v) in self.quad_grad(q, x) {
                r[j] += m.quad[k] * v;
            }
        }
        let mut bound = 0.0_f64;
        for j in 0..n {
            r[j] += -m.z_lb[j] + m.z_ub[j];
            if self.has_lb(j) {
                let s = x[j] - qp.lb_of(j);
                bound = bound.max((-s).max(0.0));
                compl = compl.max((m.z_lb[j] * s).abs());
            }
            if self.has_ub(j) {
                let s = qp.ub_of(j) - x[j];
                bound = bound.max((-s).max(0.0));
                compl = compl.max((m.z_ub[j] * s).abs());
            }
        }
        QcqpResiduals {
            primal_infeasibility: primal.max(bound),
            bound_violation: bound,
            dual_infeasibility: r.iter().fold(0.0_f64, |a, v| a.max(v.abs())),
            complementarity: compl,
        }
    }
}

/// Where a multiplier lives in [`Mults`].
#[derive(Clone, Copy)]
enum Slot {
    Y(usize),
    Z(usize),
    Quad(usize),
    Lb(usize),
    Ub(usize),
}

impl Slot {
    fn get(self, m: &Mults) -> f64 {
        match self {
            Slot::Y(i) => m.y[i],
            Slot::Z(i) => m.z[i],
            Slot::Quad(i) => m.quad[i],
            Slot::Lb(j) => m.z_lb[j],
            Slot::Ub(j) => m.z_ub[j],
        }
    }

    fn set(self, m: &mut Mults, v: f64) {
        match self {
            Slot::Y(i) => m.y[i] = v,
            Slot::Z(i) => m.z[i] = v,
            Slot::Quad(i) => m.quad[i] = v,
            Slot::Lb(j) => m.z_lb[j] = v,
            Slot::Ub(j) => m.z_ub[j] = v,
        }
    }

    fn free(self) -> bool {
        matches!(self, Slot::Y(_))
    }
}

/// One multiplier of the original problem as a column of its stationarity
/// system, with the value of the constraint it multiplies (`≤ 0` / `= 0`
/// form) at the point it was built for.
struct Col {
    slot: Slot,
    grad: SpVec,
    value: f64,
}

/// Every multiplier of the original problem at `x`.
fn columns(view: &View, x: &[f64]) -> Vec<Col> {
    let qp = view.qp;
    let mut cols = Vec::new();
    for (i, row) in view.a_rows.iter().enumerate() {
        cols.push(Col {
            slot: Slot::Y(i),
            grad: row.clone(),
            value: row_dot(row, x) - qp.b[i],
        });
    }
    for (i, row) in view.g_rows.iter().enumerate() {
        if view.nonneg[i] {
            cols.push(Col {
                slot: Slot::Z(i),
                grad: row.clone(),
                value: -view.slack(i, x),
            });
        }
    }
    for (k, q) in view.quads.iter().enumerate() {
        cols.push(Col {
            slot: Slot::Quad(k),
            grad: view.quad_grad(q, x),
            value: view.quad_value(q, x),
        });
    }
    for j in 0..qp.n {
        if view.has_lb(j) {
            cols.push(Col {
                slot: Slot::Lb(j),
                grad: vec![(j, -1.0)],
                value: qp.lb_of(j) - x[j],
            });
        }
        if view.has_ub(j) {
            cols.push(Col {
                slot: Slot::Ub(j),
                grad: vec![(j, 1.0)],
                value: x[j] - qp.ub_of(j),
            });
        }
    }
    cols
}

fn is_active(slot: Slot, m: &Mults, scale: f64) -> bool {
    slot.free() || slot.get(m) > ACTIVE_REL * scale
}

/// Re-fit the multipliers to the original stationarity at a fixed `x`: the
/// smallest weighted correction `min Σ Δᵢ²/wᵢ` with `MΔ = −∇ₓL` (in the
/// least-squares sense), solved by CGLS. An inequality's weight is its own
/// multiplier, so an active one moves in proportion to its size and an
/// inactive one ([`ACTIVE_REL`]) is frozen; equalities are weighted by the
/// largest multiplier and move freely. Whatever part of `∇ₓL` the active
/// gradients cannot reach is left in it — that is the point's residual, for
/// [`newton_step`] to remove. Returns `None` unless it lowers `‖∇ₓL‖∞`.
fn refit(view: &View, x: &[f64], m: &Mults) -> Option<Mults> {
    let n = view.qp.n;
    let cols = columns(view, x);
    let before = view.residuals(x, m).dual_infeasibility;
    let mut r = view.objective_gradient(x);
    for c in &cols {
        let v = c.slot.get(m);
        for &(j, g) in &c.grad {
            r[j] += v * g;
        }
    }
    let scale = m.scale();
    let d: Vec<f64> = cols
        .iter()
        .map(|c| {
            if c.slot.free() {
                scale.sqrt()
            } else if is_active(c.slot, m, scale) {
                c.slot.get(m).sqrt()
            } else {
                0.0
            }
        })
        .collect();
    let delta = cgls_min_norm(&cols, &d, &r, n);
    let mut out = m.clone();
    for (c, dv) in cols.iter().zip(&delta) {
        let v = c.slot.get(m) + dv;
        c.slot
            .set(&mut out, if c.slot.free() { v } else { v.max(0.0) });
    }
    (view.residuals(x, &out).dual_infeasibility < before).then_some(out)
}

/// The minimum-`‖u‖` least-squares solution of `M·D·u = −r` by CGLS,
/// returned as `Δ = D·u`. Sparse mat-vecs only.
fn cgls_min_norm(cols: &[Col], d: &[f64], r: &[f64], n: usize) -> Vec<f64> {
    let k = cols.len();
    let apply = |u: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; n];
        for (i, c) in cols.iter().enumerate() {
            let s = d[i] * u[i];
            if s != 0.0 {
                for &(j, g) in &c.grad {
                    out[j] += s * g;
                }
            }
        }
        out
    };
    let apply_t = |v: &[f64]| -> Vec<f64> {
        cols.iter()
            .zip(d)
            .map(|(c, &di)| di * row_dot(&c.grad, v))
            .collect()
    };
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();

    let mut u = vec![0.0; k];
    let mut res: Vec<f64> = r.iter().map(|v| -v).collect();
    let mut s = apply_t(&res);
    let mut p = s.clone();
    let mut gamma = dot(&s, &s);
    let gamma0 = gamma;
    for _ in 0..(2 * k).clamp(10, 500) {
        if !(gamma > 1e-30 * gamma0) || !gamma.is_finite() {
            break;
        }
        let q = apply(&p);
        let qq = dot(&q, &q);
        if !(qq > 0.0) {
            break;
        }
        let alpha = gamma / qq;
        for (ui, pi) in u.iter_mut().zip(&p) {
            *ui += alpha * pi;
        }
        for (ri, qi) in res.iter_mut().zip(&q) {
            *ri -= alpha * qi;
        }
        s = apply_t(&res);
        let gamma_new = dot(&s, &s);
        let beta = gamma_new / gamma;
        gamma = gamma_new;
        for (pi, si) in p.iter_mut().zip(&s) {
            *pi = si + beta * *pi;
        }
    }
    u.iter().zip(d).map(|(ui, di)| ui * di).collect()
}

/// One Newton step on the original KKT system restricted to the active set
/// `m` identifies: every equality, and each inequality whose multiplier is
/// above [`ACTIVE_REL`] of the largest, held at equality.
///
/// ```text
/// [ H + δ_w·I   J_Aᵀ    ] [ dx ]   [ −(Px + c) ]
/// [ J_A         −δ_c·I  ] [ μ  ] = [ −c_A(x)   ]
/// ```
///
/// with `H = P + Σ λⱼQⱼ` over the active quadratic rows. Solving for the new
/// multipliers `μ` directly, rather than a correction, leaves the inactive
/// ones at zero. Returns `None` if the factorization fails or an active
/// inequality's new multiplier comes out negative beyond the activity
/// threshold — the active set was wrong, and the step is not one to take.
fn newton_step(
    view: &View,
    x: &[f64],
    m: &Mults,
    backend: &dyn Fn() -> Box<dyn SparseSymLinearSolverInterface>,
) -> Option<(Vec<f64>, Mults)> {
    let qp = view.qp;
    let n = qp.n;
    let scale = m.scale();
    let active: Vec<Col> = columns(view, x)
        .into_iter()
        .filter(|c| is_active(c.slot, m, scale))
        .collect();
    let dim = n + active.len();

    // Lower triangle, 0-based, duplicates merged.
    let mut h: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for t in &qp.p_lower {
        let (i, j) = if t.row >= t.col {
            (t.row, t.col)
        } else {
            (t.col, t.row)
        };
        *h.entry((i, j)).or_insert(0.0) += t.val;
    }
    for c in &active {
        if let Slot::Quad(k) = c.slot {
            view.add_quad_hessian(&view.quads[k], m.quad[k], &mut h);
        }
    }
    let diag_mag = (0..n).fold(1.0_f64, |a, j| {
        a.max(h.get(&(j, j)).copied().unwrap_or(0.0).abs())
    });
    let delta_w = 1e-12 * diag_mag;
    for j in 0..n {
        *h.entry((j, j)).or_insert(0.0) += delta_w;
    }
    let delta_c = 1e-12;
    for (i, c) in active.iter().enumerate() {
        for &(j, g) in &c.grad {
            *h.entry((n + i, j)).or_insert(0.0) += g;
        }
        h.insert((n + i, n + i), -delta_c);
    }
    let airn: Vec<Index> = h.keys().map(|&(i, _)| (i + 1) as Index).collect();
    let ajcn: Vec<Index> = h.keys().map(|&(_, j)| (j + 1) as Index).collect();
    let vals: Vec<f64> = h.into_values().collect();

    let mut rhs = Vec::with_capacity(dim);
    rhs.extend(view.objective_gradient(x).iter().map(|v| -v));
    rhs.extend(active.iter().map(|c| -c.value));
    let mut fact = Factorization::new(dim as Index, airn, ajcn, vals, backend()).ok()?;
    fact.solve(&mut rhs, 1).ok()?;
    if !rhs.iter().all(|v| v.is_finite()) {
        return None;
    }

    let x_new: Vec<f64> = x.iter().zip(&rhs[..n]).map(|(a, b)| a + b).collect();
    let mut out = Mults {
        y: vec![0.0; m.y.len()],
        z: vec![0.0; m.z.len()],
        quad: vec![0.0; m.quad.len()],
        z_lb: vec![0.0; n],
        z_ub: vec![0.0; n],
    };
    for (c, &mu) in active.iter().zip(&rhs[n..]) {
        if c.slot.free() {
            c.slot.set(&mut out, mu);
        } else if mu < -ACTIVE_REL * scale {
            return None;
        } else {
            c.slot.set(&mut out, mu.max(0.0));
        }
    }
    Some((x_new, out))
}

/// Polish an optimal conic QCQP solve in the original problem's terms; see
/// the module documentation.
///
/// Returns `None` — the caller keeps the conic answer and its own residuals
/// — when `sol` is not an optimal point of `qp`'s shape: an infeasibility
/// verdict's `(y, z)` is a certificate, not a Lagrange multiplier, and a
/// presolve verdict or zero-iteration stop has no point to polish.
pub fn polish_qcqp(
    qp: &QpProblem,
    cones: &[ConeSpec],
    con_map: &[ConSocpMap],
    sol: &QpSolution,
    backend: &dyn Fn() -> Box<dyn SparseSymLinearSolverInterface>,
) -> Option<QcqpPolish> {
    let n = qp.n;
    let optimal = matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate);
    if !optimal
        || sol.x.len() != n
        || sol.y.len() != qp.m_eq()
        || sol.z.len() != qp.m_ineq()
        || !sol.x.iter().all(|v| v.is_finite())
    {
        return None;
    }
    let view = View::new(qp, cones, con_map);
    let read =
        |v: &[f64]| -> Vec<f64> { (0..n).map(|i| v.get(i).copied().unwrap_or(0.0)).collect() };
    let mut m = Mults {
        y: sol.y.clone(),
        z: sol.z.clone(),
        quad: view
            .quads
            .iter()
            .map(|q| (sol.z[q.row0] + sol.z[q.row0 + 1]) / q.kappa)
            .collect(),
        z_lb: read(&sol.z_lb),
        z_ub: read(&sol.z_ub),
    };
    let mut x = sol.x.clone();

    if let Some(fit) = refit(&view, &x, &m) {
        m = fit;
    }
    let mut res = view.residuals(&x, &m);
    let mut newton_steps = 0;
    for _ in 0..MAX_NEWTON {
        if !(res.kkt_error() > 0.0) {
            break;
        }
        let Some((x_new, m_new)) = newton_step(&view, &x, &m, backend) else {
            break;
        };
        let res_new = view.residuals(&x_new, &m_new);
        if res_new.kkt_error() < res.kkt_error() {
            x = x_new;
            m = m_new;
            res = res_new;
            newton_steps += 1;
        } else {
            break;
        }
    }

    let obj = sol.obj + (view.objective(&x) - view.objective(&sol.x));
    let mut quad_by_con = vec![0.0; con_map.len()];
    for (k, q) in view.quads.iter().enumerate() {
        quad_by_con[q.con_idx] = m.quad[k];
    }
    Some(QcqpPolish {
        x,
        obj,
        y: m.y,
        z: m.z,
        quad_by_con,
        z_lb: m.z_lb,
        z_ub: m.z_ub,
        residuals: res,
        newton_steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::{BinOp, Expr, NlBody};
    use crate::qp_extract::{BoundRelax, extract_socp_with_map};
    use pounce_convex::{QpOptions, solve_socp_ipm};
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    fn sq(e: Expr) -> Expr {
        Expr::Binary(BinOp::Pow, Box::new(e), Box::new(Expr::Const(2.0)))
    }

    fn add(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Add, Box::new(a), Box::new(b))
    }

    /// `min (x0−p0)² + (x1−p1)²  s.t.  x0² + x1² ≤ 1`, plus linear rows
    /// `(coefs, g_l, g_u)` — the issue's model family, built as an
    /// `NlProblem` so the tests run the real extraction. Maximized as the
    /// negation when `!minimize`.
    fn model(minimize: bool, p: [f64; 2], rows: &[(Vec<(usize, f64)>, f64, f64)]) -> NlProblem {
        let shift = |v: usize, c: f64| {
            Expr::Binary(BinOp::Sub, Box::new(Expr::Var(v)), Box::new(Expr::Const(c)))
        };
        let mut obj = add(sq(shift(0, p[0])), sq(shift(1, p[1])));
        if !minimize {
            obj = Expr::Binary(BinOp::Mul, Box::new(Expr::Const(-1.0)), Box::new(obj));
        }
        let mut con_nonlinear = vec![NlBody::Tree(add(sq(Expr::Var(0)), sq(Expr::Var(1))))];
        let mut con_linear = vec![vec![]];
        let mut g_l = vec![-2e19];
        let mut g_u = vec![1.0];
        for (coefs, lo, hi) in rows {
            con_nonlinear.push(NlBody::Tree(Expr::Const(0.0)));
            con_linear.push(coefs.clone());
            g_l.push(*lo);
            g_u.push(*hi);
        }
        let m = g_l.len();
        NlProblem {
            src: None,
            cse_bodies: Vec::new(),
            complementarity: Vec::new(),
            n: 2,
            m,
            num_obj: 1,
            minimize,
            obj_nonlinear: NlBody::Tree(obj),
            obj_linear: vec![],
            obj_constant: 0.0,
            con_nonlinear,
            con_linear,
            x_l: vec![-2e19, -2e19],
            x_u: vec![2e19, 2e19],
            g_l,
            g_u,
            x0: vec![0.0, 0.0],
            lambda0: vec![0.0; m],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            n_discrete: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        }
    }

    /// `‖∇ₓL‖∞` in the minimize form, computed independently of the code
    /// under test from `.sol`-convention duals (`sign` undoes a maximize).
    fn grad_l(
        sign: f64,
        p: [f64; 2],
        x: &[f64],
        lambda: &[f64],
        rows: &[(Vec<(usize, f64)>, f64, f64)],
    ) -> f64 {
        let mut r = [2.0 * (x[0] - p[0]), 2.0 * (x[1] - p[1])];
        for (j, rj) in r.iter_mut().enumerate() {
            *rj += sign * lambda[0] * 2.0 * x[j];
            for (k, (coefs, _, _)) in rows.iter().enumerate() {
                for &(v, c) in coefs {
                    if v == j {
                        *rj += sign * lambda[1 + k] * c;
                    }
                }
            }
        }
        r.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    }

    fn line() -> Vec<(Vec<(usize, f64)>, f64, f64)> {
        vec![(vec![(0, 1.0), (1, 1.0)], 1.0, 1.0)]
    }

    /// The issue's reproduction: `‖x‖² ≤ 1, x0 + x1 = 1` from `p = (2, 0)`.
    /// The mapped multipliers were `(1.000006, −1.2e-5)` with a `1.2e-5`
    /// stationarity residual against a reported `1.3e-11`. Polished, they
    /// are the unique `μ = 1, ν = 0`, and the residual reported is the one
    /// the returned pair has.
    #[test]
    fn issue_1114_equality_row_multipliers_are_stationary() {
        let (p, rows) = ([2.0, 0.0], line());
        let prob = model(true, p, &rows);
        let (qp, con_map, _, cones) =
            extract_socp_with_map(&prob, BoundRelax::NONE).expect("extract");
        let sol = solve_socp_ipm(&qp, &cones, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let raw = recover_socp_duals(&prob, &con_map, &sol.y, &sol.z);
        let raw_r = grad_l(1.0, p, &sol.x, &raw, &rows);

        let pol = polish_qcqp(&qp, &cones, &con_map, &sol, &backend).expect("optimal");
        let lambda = pol.constraint_duals(&prob, &con_map);
        assert!(
            (pol.x[0] - 1.0).abs() < 1e-12 && pol.x[1].abs() < 1e-12,
            "x={:?}",
            pol.x
        );
        assert!((lambda[0] - 1.0).abs() < 1e-10, "ball μ = {}", lambda[0]);
        assert!(lambda[1].abs() < 1e-10, "line ν = {}", lambda[1]);
        let r = grad_l(1.0, p, &pol.x, &lambda, &rows);
        assert!(
            r < 1e-12,
            "|∇L|∞ of the returned pair = {r:.2e} (unpolished {raw_r:.2e})"
        );
        assert!(
            (pol.residuals.dual_infeasibility - r).abs() < 1e-13,
            "reported {:.2e} vs independent {r:.2e}",
            pol.residuals.dual_infeasibility
        );
        assert!(pol.residuals.primal_infeasibility < 1e-12);
    }

    /// Maximize: the internal problem is negated; the polished multipliers
    /// must come back in the caller's sense.
    #[test]
    fn issue_1114_maximize_sense_round_trips() {
        let (p, rows) = ([2.0, 0.0], line());
        let prob = model(false, p, &rows);
        let (qp, con_map, _, cones) =
            extract_socp_with_map(&prob, BoundRelax::NONE).expect("extract");
        let sol = solve_socp_ipm(&qp, &cones, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let pol = polish_qcqp(&qp, &cones, &con_map, &sol, &backend).expect("optimal");
        let lambda = pol.constraint_duals(&prob, &con_map);
        assert!((lambda[0] + 1.0).abs() < 1e-10, "ball = {}", lambda[0]);
        assert!(lambda[1].abs() < 1e-10, "line = {}", lambda[1]);
        let r = grad_l(-1.0, p, &pol.x, &lambda, &rows);
        assert!(r < 1e-12, "|∇L|∞ = {r:.2e}");
    }

    /// The Newton branch: a point off the optimum, which no multiplier
    /// re-fit can make stationary. From `x = (√(1−δ²), δ)` on the circle —
    /// as a `√gap`-accurate conic point can sit — with the ball alone
    /// active, the residual's tangential part is outside the span of `∇g`.
    /// The polish must move `x` to `(1, 0)` and report a residual at
    /// roundoff, and the inactive row `x1 ≥ −5` must keep a zero multiplier
    /// (a re-fit free to touch it would park the tangential residual there
    /// — the row is a *lower* bound so that doing so needs a positive
    /// multiplier the sign clip would not reject).
    #[test]
    fn issue_1114_newton_moves_an_inaccurate_point() {
        let p = [2.0, 0.0];
        let rows = vec![(vec![(1, 1.0)], -5.0, 2e19)];
        let prob = model(true, p, &rows);
        let (qp, con_map, _, cones) =
            extract_socp_with_map(&prob, BoundRelax::NONE).expect("extract");
        let mut sol = solve_socp_ipm(&qp, &cones, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let delta: f64 = 1e-5;
        sol.x = vec![(1.0 - delta * delta).sqrt(), delta];
        let ConSocpMap::Ineq {
            lower: Some(row), ..
        } = con_map[1]
        else {
            panic!("expected a lower-bounded linear row: {:?}", con_map[1]);
        };
        sol.z[row] = 1e-12;
        let raw = recover_socp_duals(&prob, &con_map, &sol.y, &sol.z);
        assert!(
            grad_l(1.0, p, &sol.x, &raw, &rows) > 1e-5,
            "fixture must start off-stationary"
        );

        let pol = polish_qcqp(&qp, &cones, &con_map, &sol, &backend).expect("optimal");
        assert!(pol.newton_steps >= 1, "no Newton step was taken");
        assert!(
            (pol.x[0] - 1.0).abs() < 1e-12 && pol.x[1].abs() < 1e-12,
            "x={:?}",
            pol.x
        );
        let lambda = pol.constraint_duals(&prob, &con_map);
        assert!((lambda[0] - 1.0).abs() < 1e-10, "ball μ = {}", lambda[0]);
        assert_eq!(lambda[1], 0.0, "inactive row multiplier");
        assert!(pol.residuals.kkt_error() < 1e-12, "{:?}", pol.residuals);
    }

    /// The re-fit's own freeze, with Newton unable to help: `refit` alone at
    /// the perturbed point must leave the inactive row's multiplier at zero
    /// and the tangential residual in place.
    #[test]
    fn issue_1114_refit_freezes_an_inactive_row() {
        let p = [2.0, 0.0];
        let rows = vec![(vec![(1, 1.0)], -5.0, 2e19)];
        let prob = model(true, p, &rows);
        let (qp, con_map, _, cones) =
            extract_socp_with_map(&prob, BoundRelax::NONE).expect("extract");
        let view = View::new(&qp, &cones, &con_map);
        let delta: f64 = 1e-5;
        let x = vec![(1.0 - delta * delta).sqrt(), delta];
        let row = qp.m_ineq() - 4 - 1; // the orthant row precedes the 4-row cone
        assert!(view.nonneg[row]);
        let mut m = Mults {
            y: vec![],
            z: vec![0.0; qp.m_ineq()],
            quad: vec![1.0],
            z_lb: vec![0.0; 2],
            z_ub: vec![0.0; 2],
        };
        m.z[row] = 1e-12;
        let fit = refit(&view, &x, &m).unwrap_or(m);
        assert_eq!(fit.z[row], 1e-12, "inactive row multiplier moved");
        assert!(view.residuals(&x, &fit).dual_infeasibility > 1e-5);
    }

    /// An infeasibility verdict's `(y, z)` is a certificate, not a Lagrange
    /// multiplier: nothing is polished.
    #[test]
    fn issue_1114_non_optimal_status_is_not_polished() {
        let prob = model(true, [2.0, 0.0], &line());
        let (qp, con_map, _, cones) =
            extract_socp_with_map(&prob, BoundRelax::NONE).expect("extract");
        let mut sol = solve_socp_ipm(&qp, &cones, &QpOptions::default(), backend);
        sol.status = QpStatus::PrimalInfeasible;
        assert!(polish_qcqp(&qp, &cones, &con_map, &sol, &backend).is_none());
    }
}
