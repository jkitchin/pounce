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
//! to the active set ([`is_active`]: a significant multiplier *or* a
//! constraint at its bound; [`newton_step`] drops any that come back with a
//! negative multiplier). A step is kept only if it lowers the KKT error, each
//! term in its own units, without giving up feasibility beyond `tol`
//! ([`accept_step`]). The result is reported against the original problem
//! ([`QcqpResiduals`]), so `final_dual_inf` describes the `(x, λ)` actually
//! handed back, and a point that still misses `tol` is warned about
//! ([`QcqpPolish::misses_tol`]).

use crate::nl_reader::NlProblem;
use crate::qp_extract::{ConSocpMap, recover_socp_duals};
use pounce_common::types::Index;
use pounce_convex::{ConeSpec, QpProblem, QpSolution, QpStatus};
use pounce_linsol::{Factorization, SparseSymLinearSolverInterface};
use std::collections::BTreeMap;

/// An inequality is read as **active** when its multiplier is above this
/// fraction of the largest multiplier, *or* it is primal-active
/// ([`PRIMAL_ACTIVE_REL`]). An inactive one is frozen by the re-fit and left
/// out of the Newton step's active set.
///
/// Freezing, rather than the small weight a multiplier-proportional rule
/// would give it, matters when `x` carries a residual no honest multiplier
/// removes: a minimum-norm fit that may touch every column parks it on an
/// inactive row instead. Measured on a 3-variable ball-plus-plane model, an
/// inactive range row went from `−3e-9` to `−1.6e-5`, "zeroing" a residual
/// the returned `x` genuinely carried.
///
/// The multiplier ratio alone is not enough (PR #1120 review): on
/// `min W(x0−2)² + (x1+1)²  s.t.  ‖x‖² ≤ 1, x1 ≥ 0` the active `x1 ≥ 0`
/// carries `2` against the ball's `W`, so for `W ≥ 1e7` it fell under the
/// ratio, was frozen, and every Newton step was refused — `final_dual_inf`
/// `9.56` at `W = 1e8`. Hence the primal test beside it.
const ACTIVE_REL: f64 = 1e-6;

/// An inequality whose value is within this fraction of its own magnitude
/// (`1 + |rhs| + Σ|terms|`) of its bound is primal-active, whatever its
/// multiplier. The conic point is only `~√gap`-accurate, so a genuinely
/// active constraint can sit `1e-7` inside its bound (`x1 = 1.8e-7` on the
/// `W = 1e8` model above). Admitting a constraint that is near its bound but
/// not active is safe: held at equality, it comes back with a negative
/// multiplier, and [`newton_step`] drops it and re-solves.
const PRIMAL_ACTIVE_REL: f64 = 1e-6;

/// Iterative-refinement passes on each active-set KKT solve.
const REFINE_STEPS: usize = 2;

/// Constraints [`newton_step`] may drop from its active set, one per
/// re-solve, for coming back with a negative multiplier.
const MAX_DROPS: usize = 8;

/// Newton steps attempted on the active-set KKT system. Each is kept only
/// if it lowers the original KKT error, and the conic point is already close,
/// so convergence is quadratic from the first step; three covers a step
/// spent recovering the regularization's `O(δ)` error.
const MAX_NEWTON: usize = 3;

type SpVec = Vec<(usize, f64)>;

/// Builds a fresh linear-solver backend, as the conic solve's callers do.
pub type BackendFactory = dyn Fn() -> Box<dyn SparseSymLinearSolverInterface>;

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
    /// `max |μ·g| / (1 + |μ|)`: complementarity in the constraint's own
    /// units, which the step-acceptance test compares. Unscaled, a large
    /// multiplier turns roundoff in `g` into an error that refused every
    /// Newton step on an already-stationary model (PR #1120 review, `W = 1e5`).
    compl_scaled: f64,
}

impl QcqpResiduals {
    pub fn kkt_error(&self) -> f64 {
        self.primal_infeasibility
            .max(self.dual_infeasibility)
            .max(self.complementarity)
    }

    /// What a Newton step must lower to be kept: [`Self::kkt_error`] with
    /// each term in its own units — complementarity as
    /// [`Self::compl_scaled`], and the dual residual relative to
    /// `dual_scale`, the objective gradient's magnitude. An absolute dual
    /// residual cannot go below the roundoff in summing gradient terms of
    /// that size: on a `1e8`-weighted objective the exact point reads `3e-8`,
    /// and an unscaled comparison refused the step that reached it in favour
    /// of an inexact one whose terms happened to cancel to `7e-11`.
    fn merit(&self, dual_scale: f64) -> f64 {
        self.primal_infeasibility
            .max(self.dual_infeasibility / dual_scale)
            .max(self.compl_scaled)
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
    /// `max(1, ‖Px + c‖∞)` at the returned point: the scale a dual residual
    /// is judged against ([`Self::misses_tol`]).
    pub dual_scale: f64,
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

    /// Whether the polished point still misses the original KKT conditions
    /// by more than `tol`, with the dual residual read relative to the
    /// objective gradient ([`Self::dual_scale`]) — the caller warns rather
    /// than reporting such a solve as cleanly converged.
    pub fn misses_tol(&self, tol: f64) -> bool {
        let r = &self.residuals;
        r.primal_infeasibility > tol
            || r.dual_infeasibility / self.dual_scale > tol
            || r.compl_scaled > tol
    }
}

fn row_dot(row: &[(usize, f64)], x: &[f64]) -> f64 {
    row.iter().map(|&(j, v)| v * x[j]).sum()
}

impl<'a> View<'a> {
    /// `None` when the cone layout and `con_map` disagree — a cone running
    /// past `G`, a quadratic row whose block is not a second-order cone of
    /// dimension ≥ 2 starting at its `z_row0`, or rows `z_row0`/`z_row1` that
    /// are not adjacent. The polish would otherwise read the wrong rows
    /// without a sign, so it is skipped instead.
    fn new(qp: &'a QpProblem, cones: &[ConeSpec], con_map: &[ConSocpMap]) -> Option<Self> {
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
            let dim = spec.dim();
            if off + dim > m_ineq {
                return None;
            }
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
        let mut quads = Vec::new();
        for (con_idx, m) in con_map.iter().enumerate() {
            if let ConSocpMap::Quad {
                z_row0,
                z_row1,
                kappa,
            } = *m
            {
                let dim = *soc_dim_at.get(&z_row0)?;
                if dim < 2 || z_row1 != z_row0 + 1 || !(kappa > 0.0) {
                    return None;
                }
                quads.push(Quad {
                    con_idx,
                    row0: z_row0,
                    dim,
                    kappa,
                });
            }
        }
        Some(View {
            qp,
            a_rows,
            g_rows,
            nonneg,
            quads,
        })
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
        let mut compl_scaled = 0.0_f64;
        let mut note = |mu: f64, g: f64| {
            compl = compl.max((mu * g).abs());
            compl_scaled = compl_scaled.max((mu * g).abs() / (1.0 + mu.abs()));
        };
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
            note(m.z[i], s);
            for &(j, v) in row {
                r[j] += m.z[i] * v;
            }
        }
        for (k, q) in self.quads.iter().enumerate() {
            let g = self.quad_value(q, x);
            primal = primal.max(g.max(0.0));
            note(m.quad[k], g);
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
                note(m.z_lb[j], s);
            }
            if self.has_ub(j) {
                let s = qp.ub_of(j) - x[j];
                bound = bound.max((-s).max(0.0));
                note(m.z_ub[j], s);
            }
        }
        QcqpResiduals {
            primal_infeasibility: primal.max(bound),
            bound_violation: bound,
            dual_infeasibility: r.iter().fold(0.0_f64, |a, v| a.max(v.abs())),
            complementarity: compl,
            compl_scaled,
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
    /// The magnitude of the terms `value` is a difference of, for the
    /// primal-activity test ([`PRIMAL_ACTIVE_REL`]).
    mag: f64,
}

/// Every multiplier of the original problem at `x`.
fn columns(view: &View, x: &[f64]) -> Vec<Col> {
    let qp = view.qp;
    let abs_dot = |row: &[(usize, f64)]| row.iter().map(|&(j, v)| (v * x[j]).abs()).sum::<f64>();
    let mut cols = Vec::new();
    for (i, row) in view.a_rows.iter().enumerate() {
        cols.push(Col {
            slot: Slot::Y(i),
            grad: row.clone(),
            value: row_dot(row, x) - qp.b[i],
            mag: qp.b[i].abs() + abs_dot(row),
        });
    }
    for (i, row) in view.g_rows.iter().enumerate() {
        if view.nonneg[i] {
            cols.push(Col {
                slot: Slot::Z(i),
                grad: row.clone(),
                value: -view.slack(i, x),
                mag: qp.h[i].abs() + abs_dot(row),
            });
        }
    }
    for (k, q) in view.quads.iter().enumerate() {
        let s0 = view.slack(q.row0, x);
        let s1 = view.slack(q.row0 + 1, x);
        let sq: f64 = (q.row0 + 2..q.row0 + q.dim)
            .map(|r| view.slack(r, x).powi(2))
            .sum();
        cols.push(Col {
            slot: Slot::Quad(k),
            grad: view.quad_grad(q, x),
            value: view.quad_value(q, x),
            mag: 0.25 * sq + 0.5 * q.kappa * (s0.abs() + s1.abs()),
        });
    }
    #[allow(clippy::needless_range_loop)] // `j` names the bound slot, not just `x[j]`
    for j in 0..qp.n {
        if view.has_lb(j) {
            cols.push(Col {
                slot: Slot::Lb(j),
                grad: vec![(j, -1.0)],
                value: qp.lb_of(j) - x[j],
                mag: qp.lb_of(j).abs() + x[j].abs(),
            });
        }
        if view.has_ub(j) {
            cols.push(Col {
                slot: Slot::Ub(j),
                grad: vec![(j, 1.0)],
                value: x[j] - qp.ub_of(j),
                mag: qp.ub_of(j).abs() + x[j].abs(),
            });
        }
    }
    cols
}

/// Every equality, and each inequality whose multiplier is significant
/// ([`ACTIVE_REL`]) or which is primal-active ([`PRIMAL_ACTIVE_REL`]).
fn is_active(c: &Col, m: &Mults, scale: f64) -> bool {
    c.slot.free()
        || c.slot.get(m) > ACTIVE_REL * scale
        || c.value >= -PRIMAL_ACTIVE_REL * (1.0 + c.mag)
}

/// Re-fit the multipliers to the original stationarity at a fixed `x`: the
/// smallest weighted correction `min Σ Δᵢ²/wᵢ` with `MΔ = −∇ₓL` (in the
/// least-squares sense), solved by CGLS. An active inequality ([`is_active`])
/// is weighted by its own multiplier, floored at [`ACTIVE_REL`] of the
/// largest so a primal-active one with a near-zero multiplier can still move;
/// an inactive one is frozen. Equalities are weighted by the largest
/// multiplier and move freely. Whatever part of `∇ₓL` the active
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
            } else if is_active(c, m, scale) {
                c.slot.get(m).max(ACTIVE_REL * scale).sqrt()
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
/// ([`is_active`]), held at equality.
///
/// ```text
/// [ H + δ_w·I   J_Aᵀ    ] [ dx ]   [ −∇ₓL_A(x, μ_A) ]
/// [ J_A         −δ_c·I  ] [ dμ ] = [ −c_A(x)        ]
/// ```
///
/// with `H = P + Σ λⱼQⱼ` over the active quadratic rows, `∇ₓL_A` the
/// Lagrangian gradient with only the active multipliers, and the new active
/// multipliers `μ_A + dμ`; the inactive ones are set to zero.
///
/// For the *correction* `dμ`, not the new `μ`: the regularization then
/// perturbs the constraint rows by `δ_c·dμ`, which is negligible, where for
/// `μ` itself it is `δ_c·μ` — with a ball multiplier of `1e8` that put the
/// step `5e-5` off the constraint and every step was refused.
///
/// An inequality whose new multiplier comes out negative beyond
/// [`ACTIVE_REL`] of the largest does not belong in the active set — a
/// primal-active constraint that is not binding, typically — so the most
/// negative is dropped and the system re-solved, up to [`MAX_DROPS`] times.
/// Returns `None` if the factorization fails or the drops run out.
fn newton_step(
    view: &View,
    x: &[f64],
    m: &Mults,
    backend: &BackendFactory,
) -> Option<(Vec<f64>, Mults)> {
    let scale = m.scale();
    let mut active: Vec<Col> = columns(view, x)
        .into_iter()
        .filter(|c| is_active(c, m, scale))
        .collect();
    for _ in 0..=MAX_DROPS {
        let (x_new, mu) = solve_active_kkt(view, x, m, &active, backend)?;
        let worst = active
            .iter()
            .zip(&mu)
            .enumerate()
            .filter(|(_, (c, v))| !c.slot.free() && **v < -ACTIVE_REL * scale)
            .min_by(|a, b| a.1.1.total_cmp(b.1.1))
            .map(|(i, _)| i);
        if let Some(i) = worst {
            active.remove(i);
            continue;
        }
        let n = view.qp.n;
        let mut out = Mults {
            y: vec![0.0; m.y.len()],
            z: vec![0.0; m.z.len()],
            quad: vec![0.0; m.quad.len()],
            z_lb: vec![0.0; n],
            z_ub: vec![0.0; n],
        };
        for (c, &v) in active.iter().zip(&mu) {
            c.slot
                .set(&mut out, if c.slot.free() { v } else { v.max(0.0) });
        }
        return Some((x_new, out));
    }
    None
}

/// Factor and solve the active-set KKT system of [`newton_step`], returning
/// the new point and the active constraints' new multipliers `μ_A + dμ`.
fn solve_active_kkt(
    view: &View,
    x: &[f64],
    m: &Mults,
    active: &[Col],
    backend: &BackendFactory,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let qp = view.qp;
    let n = qp.n;
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
    for c in active {
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
    let entries: Vec<((usize, usize), f64)> = h.into_iter().collect();
    let airn: Vec<Index> = entries
        .iter()
        .map(|&((i, _), _)| (i + 1) as Index)
        .collect();
    let ajcn: Vec<Index> = entries
        .iter()
        .map(|&((_, j), _)| (j + 1) as Index)
        .collect();
    let vals: Vec<f64> = entries.iter().map(|&(_, v)| v).collect();

    // `∇ₓL` with the active multipliers only.
    let mut grad_l = view.objective_gradient(x);
    for c in active {
        let mu = c.slot.get(m);
        for &(j, g) in &c.grad {
            grad_l[j] += mu * g;
        }
    }
    let mut b = Vec::with_capacity(dim);
    b.extend(grad_l.iter().map(|v| -v));
    b.extend(active.iter().map(|c| -c.value));
    let mut fact = Factorization::new(dim as Index, airn, ajcn, vals, backend()).ok()?;
    let mut rhs = b.clone();
    fact.solve(&mut rhs, 1).ok()?;
    // Iterative refinement against the matrix as assembled, each correction
    // kept only if it shrinks the linear residual: on a near-singular system
    // (a degenerate active set) refinement can diverge, and did on
    // `pump_network_h148_brf0`.
    let residual = |sol: &[f64]| -> Vec<f64> {
        let mut r = b.clone();
        for &((i, j), v) in &entries {
            r[i] -= v * sol[j];
            if i != j {
                r[j] -= v * sol[i];
            }
        }
        r
    };
    let inf = |v: &[f64]| v.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
    let mut r = residual(&rhs);
    for _ in 0..REFINE_STEPS {
        let mut d = r.clone();
        if fact.solve(&mut d, 1).is_err() {
            break;
        }
        let trial: Vec<f64> = rhs.iter().zip(&d).map(|(s, d)| s + d).collect();
        let r_trial = residual(&trial);
        if !(inf(&r_trial) < inf(&r)) {
            break;
        }
        rhs = trial;
        r = r_trial;
    }
    if !rhs.iter().all(|v| v.is_finite()) {
        return None;
    }

    let x_new: Vec<f64> = x.iter().zip(&rhs[..n]).map(|(a, b)| a + b).collect();
    let mu: Vec<f64> = active
        .iter()
        .zip(&rhs[n..])
        .map(|(c, d)| c.slot.get(m) + d)
        .collect();
    Some((x_new, mu))
}

/// Whether a Newton step from residuals `old` to `new` is kept: it must
/// lower the merit ([`QcqpResiduals::merit`]) and must not raise the primal
/// violation above `max(old, tol)` — stationarity is never bought with
/// feasibility the point already had.
fn accept_step(old: &QcqpResiduals, new: &QcqpResiduals, tol: f64, dual_scale: f64) -> bool {
    new.primal_infeasibility <= old.primal_infeasibility.max(tol)
        && new.merit(dual_scale) < old.merit(dual_scale)
}

/// Polish an optimal conic QCQP solve in the original problem's terms; see
/// the module documentation.
///
/// Returns `None` — the caller keeps the conic answer and its own residuals
/// — when `sol` is not an optimal point of `qp`'s shape: an infeasibility
/// verdict's `(y, z)` is a certificate, not a Lagrange multiplier, and a
/// presolve verdict or zero-iteration stop has no point to polish. Also
/// `None` when the cone layout disagrees with `con_map` ([`View::new`]).
///
/// `tol` is the solve's tolerance: a Newton step may not raise the primal
/// violation above `max(current, tol)`, so stationarity is never bought with
/// feasibility the solve had.
pub fn polish_qcqp(
    qp: &QpProblem,
    cones: &[ConeSpec],
    con_map: &[ConSocpMap],
    sol: &QpSolution,
    tol: f64,
    backend: &BackendFactory,
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
    let view = View::new(qp, cones, con_map)?;
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
    // Fixed for the whole loop, so every comparison is in one unit.
    let merit_scale = view
        .objective_gradient(&x)
        .iter()
        .fold(1.0_f64, |a, v| a.max(v.abs()));
    for _ in 0..MAX_NEWTON {
        if !(res.merit(merit_scale) > 0.0) {
            break;
        }
        let Some((x_new, m_new)) = newton_step(&view, &x, &m, backend) else {
            break;
        };
        let res_new = view.residuals(&x_new, &m_new);
        if accept_step(&res, &res_new, tol, merit_scale) {
            x = x_new;
            m = m_new;
            res = res_new;
            newton_steps += 1;
        } else {
            break;
        }
    }

    // The Newton multipliers carry the linear solve's error, which on an
    // ill-conditioned active set exceeds what a re-fit at the final point
    // reaches: `pump_network_h148_brf0`'s dual residual is `7.7e-7` after its
    // accepted step and lower after this.
    if newton_steps > 0 {
        if let Some(fit) = refit(&view, &x, &m) {
            let res_fit = view.residuals(&x, &fit);
            if res_fit.merit(merit_scale) <= res.merit(merit_scale) {
                m = fit;
                res = res_fit;
            }
        }
    }

    let obj = sol.obj + (view.objective(&x) - view.objective(&sol.x));
    let dual_scale = view
        .objective_gradient(&x)
        .iter()
        .fold(1.0_f64, |a, v| a.max(v.abs()));
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
        dual_scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::{BinOp, Expr, NlBody};
    use crate::qp_extract::{BoundRelax, extract_socp_with_map};
    use pounce_convex::{QpOptions, solve_socp_ipm};
    use pounce_feral::FeralSolverInterface;

    /// A linear row `(coefficients, g_l, g_u)`.
    type Row = (Vec<(usize, f64)>, f64, f64);

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
    fn model(minimize: bool, p: [f64; 2], rows: &[Row]) -> NlProblem {
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
    fn grad_l(sign: f64, p: [f64; 2], x: &[f64], lambda: &[f64], rows: &[Row]) -> f64 {
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

    fn line() -> Vec<Row> {
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

        let pol = polish_qcqp(&qp, &cones, &con_map, &sol, 1e-8, &backend).expect("optimal");
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
        let pol = polish_qcqp(&qp, &cones, &con_map, &sol, 1e-8, &backend).expect("optimal");
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

        let pol = polish_qcqp(&qp, &cones, &con_map, &sol, 1e-8, &backend).expect("optimal");
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
        let view = View::new(&qp, &cones, &con_map).expect("layout");
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

    /// `OptimalInaccurate` is an optimal point too, and is polished.
    #[test]
    fn issue_1114_optimal_inaccurate_is_polished() {
        let (p, rows) = ([2.0, 0.0], line());
        let prob = model(true, p, &rows);
        let (qp, con_map, _, cones) =
            extract_socp_with_map(&prob, BoundRelax::NONE).expect("extract");
        let mut sol = solve_socp_ipm(&qp, &cones, &QpOptions::default(), backend);
        sol.status = QpStatus::OptimalInaccurate;
        let pol = polish_qcqp(&qp, &cones, &con_map, &sol, 1e-8, &backend).expect("polished");
        let lambda = pol.constraint_duals(&prob, &con_map);
        assert!((lambda[0] - 1.0).abs() < 1e-10 && lambda[1].abs() < 1e-10);
    }

    /// The active-set drop. `min (x0−2)² + (x1−t)²  s.t.  ‖x‖² ≤ 1, x1 ≥ 0`
    /// with `t = 9e-7`: the optimum is the circle point nearest `(2, t)`,
    /// `x1* = t/‖(2, t)‖ ≈ 4.5e-7`, so `x1 ≥ 0` is *not* binding — but it
    /// is within [`PRIMAL_ACTIVE_REL`] of its bound, so it enters the active
    /// set, comes back with a negative multiplier when held at equality, and
    /// must be dropped and the system re-solved.
    #[test]
    fn issue_1114_primal_active_nonbinding_row_is_dropped() {
        let t = 9e-7;
        let p = [2.0, t];
        let rows = vec![(vec![(1, 1.0)], 0.0, 2e19)];
        let prob = model(true, p, &rows);
        let (qp, con_map, _, cones) =
            extract_socp_with_map(&prob, BoundRelax::NONE).expect("extract");
        let mut sol = solve_socp_ipm(&qp, &cones, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let norm = (4.0 + t * t).sqrt();
        let x_star = [2.0 / norm, t / norm];
        // Start from a point that is off by more than the polish's own
        // accuracy, so the drop is what has to happen for the test to pass.
        sol.x = vec![x_star[0], x_star[1] + 2e-7];
        let pol = polish_qcqp(&qp, &cones, &con_map, &sol, 1e-8, &backend).expect("optimal");
        assert!(pol.newton_steps >= 1, "no Newton step was taken");
        assert!(
            (pol.x[1] - x_star[1]).abs() < 1e-13,
            "x1 = {:e}, expected {:e}",
            pol.x[1],
            x_star[1]
        );
        let lambda = pol.constraint_duals(&prob, &con_map);
        assert_eq!(lambda[1], 0.0, "non-binding row multiplier");
        assert!(
            (lambda[0] - (norm - 1.0)).abs() < 1e-9,
            "ball μ = {}",
            lambda[0]
        );
    }

    /// A cone layout that disagrees with `con_map` skips the polish rather
    /// than reading the wrong rows.
    #[test]
    fn issue_1114_layout_mismatch_skips_the_polish() {
        let prob = model(true, [2.0, 0.0], &line());
        let (qp, mut con_map, _, cones) =
            extract_socp_with_map(&prob, BoundRelax::NONE).expect("extract");
        let sol = solve_socp_ipm(&qp, &cones, &QpOptions::default(), backend);
        assert!(polish_qcqp(&qp, &cones, &con_map, &sol, 1e-8, &backend).is_some());
        let ConSocpMap::Quad { z_row0, kappa, .. } = con_map[0] else {
            panic!("expected the ball as a cone: {:?}", con_map[0]);
        };
        con_map[0] = ConSocpMap::Quad {
            z_row0: z_row0 + 1,
            z_row1: z_row0 + 2,
            kappa,
        };
        assert!(polish_qcqp(&qp, &cones, &con_map, &sol, 1e-8, &backend).is_none());
        assert!(
            polish_qcqp(
                &qp,
                &cones[..cones.len() - 1],
                &con_map,
                &sol,
                1e-8,
                &backend
            )
            .is_none()
        );
    }

    /// Both arms of the acceptance rule.
    #[test]
    fn issue_1114_accept_step_rule() {
        let r = |primal: f64, dual: f64, compl_scaled: f64| QcqpResiduals {
            primal_infeasibility: primal,
            bound_violation: 0.0,
            dual_infeasibility: dual,
            complementarity: compl_scaled,
            compl_scaled,
        };
        let old = r(1e-14, 1e-6, 0.0);
        // Lower merit, feasibility kept: accepted.
        assert!(accept_step(&old, &r(1e-14, 1e-12, 0.0), 1e-8, 1.0));
        // Lower merit bought with feasibility past `tol`: refused.
        assert!(!accept_step(&old, &r(1e-7, 1e-12, 0.0), 1e-8, 1.0));
        // Feasibility worse but within `tol`: accepted.
        assert!(accept_step(&old, &r(1e-9, 1e-12, 0.0), 1e-8, 1.0));
        // Not lower: refused.
        assert!(!accept_step(&old, &r(1e-14, 1e-6, 0.0), 1e-8, 1.0));
        // The dual term is read relative to the gradient scale: `3e-8` on a
        // `2e8` gradient beats a complementarity of `2e-11`.
        let inexact = r(3e-14, 7e-11, 2e-11);
        let exact = r(0.0, 3e-8, 1e-15);
        assert!(accept_step(&inexact, &exact, 1e-8, 2e8));
        assert!(!accept_step(&inexact, &exact, 1e-8, 1.0));
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
        assert!(polish_qcqp(&qp, &cones, &con_map, &sol, 1e-8, &backend).is_none());
    }
}
