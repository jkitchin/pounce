//! [`MpccTnlp`] — lowering of declared complementarity pairs into an NLP.
//!
//! Workstream B of `dev-notes/research/calipso-contact-plan.md`: POUNCE
//! should accept "these two quantities are complementary" as a statement
//! about the model and choose the formulation itself, rather than make the
//! user hand-write `G·H = 0` and pick a route. This module is the first
//! layer: a TNLP wrapper that presents a model with declared pairs as an
//! ordinary NLP under a chosen [`Lowering`]. Which lowering to use, and in
//! what order, is the driver's business, not this module's.
//!
//! ## Pairs
//!
//! A [`ComplementarityPair`] `{ row: i, var: j }` declares
//!
//! ```text
//! 0 <= G(x) := s_G (x_j - b_j)   ⟂   H(x) := s_H (g_i(x) - c_i) >= 0
//! ```
//!
//! where each side takes its orientation from the one finite bound the
//! model states: `x_j` bounded below gives `s_G = +1, b_j = x_L[j]`, bounded
//! above gives `s_G = -1, b_j = x_U[j]`; row `i` likewise from `g_L` / `g_U`.
//! That is AMPL's `g_i(x) complements x_j` in both one-sided forms — the
//! shapes `.nl` complementarity rows encode (kind 5 with one finite variable
//! bound: `x >= l ⟂ body >= 0` or `x <= u ⟂ body <= 0`). A variable with a
//! second finite bound keeps it as an ordinary limit, the row's one finite
//! side naming the bound in the pair (what Pyomo writes for
//! `complements(s >= 0, L >= 0)` with `L <= 1` declared).
//!
//! A **mixed** pair — the variable boxed, the row free: `l <= x_j <= u ⟂
//! g_i`, `g_i >= 0` at `l`, `g_i <= 0` at `u`, `g_i = 0` between (`.nl` flag
//! 3) — lowers to two product rows on the same variable and row,
//! `(x_j - l)·g_i <= 0` and `(u - x_j)·(-g_i) <= 0`. Their products are
//! legitimately nonzero at a bound, so both rows are bounded above under
//! every lowering and their violation is the positive part
//! ([`MpccTnlp::last_violation`]). A boxed row, or a pair with no finite
//! bound on either side, is refused by [`MpccTnlp::new`] with the reason
//! rather than lowered wrongly. A pair between two variables is a row
//! `g_i(x) = x_k` away.
//!
//! ## What the wrapper adds
//!
//! One row per pair, `p_k(x) = G_k(x) · H_k(x)`, appended after the inner
//! model's `m` rows, with bounds set by the lowering:
//!
//! | [`Lowering`] | product row |
//! |---|---|
//! | `Eq` | `G·H = 0` (the exact product; `ncp_eq` in the benchmarks) |
//! | `Ineq` | `G·H <= 0` (same feasible set given `G, H >= 0`; adds no equality) |
//! | `Relaxed(τ)` | `G·H <= τ` (Scholtes' relaxation, for continuation) |
//!
//! The lowering can be changed between solves ([`MpccTnlp::set_lowering`])
//! without rebuilding, so a continuation walks `τ` on one object.
//!
//! ## Derivatives
//!
//! `∇p = s_G H e_j + s_H G ∇g_i`. The Lagrangian Hessian of the lowered
//! model is the inner one evaluated with the row-`i` multiplier shifted by
//! `μ_k s_H G`, plus the cross terms `μ_k s_G s_H (e_j ∇g_iᵀ + ∇g_i e_jᵀ)`
//! (doubled on the diagonal when `x_j` appears in `g_i`). Cross entries the inner Hessian
//! already has are accumulated in place; the rest are appended. No entry is
//! ever duplicated, in the Jacobian or the Hessian.
//!
//! ## Borrow discipline
//!
//! As `pounce-l1penalty`'s wrapper: the inner TNLP is an
//! `Rc<RefCell<dyn TNLP>>`, borrowed for the duration of each call, never
//! re-entered.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use pounce_common::types::{Index, Number};

use crate::alg_types::SolverReturn;
use crate::constant_derivatives::{DerivativeProof, DerivativeProofs};
use crate::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, Linearity, MetaData, NlpInfo,
    ScalingRequest, Solution, SparsityRequest, StartingPoint, TNLP,
};

/// A declared complementarity pair; see the module docs for the semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComplementarityPair {
    /// Inner constraint row `i`: `H(x) = g_i(x) - g_L[i]`.
    pub row: usize,
    /// Inner variable `j`: `G(x) = x_j - x_L[j]`.
    pub var: usize,
}

/// How the product rows are bounded; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lowering {
    Eq,
    Ineq,
    Relaxed(Number),
}

/// A primal-dual point for the next solve to start from, in the lowered
/// model's dimensions (`lambda` covers the product rows too). Set by a
/// driver between continuation stages; see [`MpccTnlp::set_start_override`].
#[derive(Debug, Clone, Default)]
pub struct StartOverride {
    pub x: Vec<Number>,
    pub z_l: Vec<Number>,
    pub z_u: Vec<Number>,
    pub lambda: Vec<Number>,
}

/// A branch of the complementarity: which side of each pair is held at
/// zero, stated as bound overrides on the inner model. Set with
/// [`MpccTnlp::set_branch`] for a branch NLP: the product rows are then
/// unbounded, so the lowered model is the inner one with these bounds — an
/// ordinary NLP, solved through the same wrapper (hooks, deferred finalize,
/// start override) as every other lowered solve.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BranchBounds {
    /// `(variable, lower, upper)`: replaces that variable's bounds.
    pub vars: Vec<(usize, Number, Number)>,
    /// `(row, lower, upper)`: replaces that row's bounds.
    pub rows: Vec<(usize, Number, Number)>,
}

/// Bounds at or beyond this magnitude are infinite, matching the
/// `nlp_lower_bound_inf` / `nlp_upper_bound_inf` defaults (`±1e19`).
const INF: Number = 1e19;

/// The lowered model; see the module docs.
pub struct MpccTnlp {
    inner: Rc<RefCell<dyn TNLP>>,
    /// The pairs as declared, for [`Self::pairs`].
    pairs: Vec<ComplementarityPair>,
    /// The lowered pairs, one per product row: a one-sided declared pair
    /// once, a mixed one twice (see [`MpccTnlp::new`]).
    lowered: Vec<ComplementarityPair>,
    /// Per lowered pair: `H` carries no sign constraint (a mixed pair's free
    /// row), so its product row is bounded above only, never `= 0`, and its
    /// violation is the product's positive part.
    signless: Vec<bool>,
    lowering: Lowering,
    n: usize,
    m_inner: usize,
    index_style: IndexStyle,
    inner_jac_nnz: usize,
    inner_h_nnz: usize,
    /// Per pair: `(s_G, b_j)` and `(s_H, c_i)` — orientation sign and the
    /// finite bound each side is measured from.
    g_side: Vec<(Number, Number)>,
    h_side: Vec<(Number, Number)>,
    /// Per pair: positions in the inner Jacobian value array of row `i`'s
    /// entries, and their (0-based) columns.
    row_entries: Vec<Vec<(usize, usize)>>,
    /// Per pair: the position, within this pair's block of the lowered
    /// Jacobian, of the `x_j` entry (it is appended unless `x_j` is already
    /// a column of row `i`).
    var_slot: Vec<usize>,
    /// Per pair: offset of its block in the lowered Jacobian value array.
    jac_block: Vec<usize>,
    jac_nnz: usize,
    /// Lowered Hessian: inner structure, then appended cross entries
    /// (0-based, lower triangle).
    h_extra: Vec<(usize, usize)>,
    /// (row, col), lower triangle, 0-based -> position in the lowered
    /// Hessian value array.
    h_pos: HashMap<(usize, usize), usize>,
    // Scratch.
    g_inner: Vec<Number>,
    jac_inner: Vec<Number>,
    lam_inner: Vec<Number>,
    // Last finalize, for the driver.
    last_status: Option<SolverReturn>,
    last_x: Vec<Number>,
    last_lambda: Vec<Number>,
    last_z_l: Vec<Number>,
    last_z_u: Vec<Number>,
    last_products: Vec<Number>,
    defer_inner_finalize: bool,
    start_override: Option<StartOverride>,
    branch: Option<BranchBounds>,
}

/// `(sign, bound)` for a side with exactly one finite bound: `(+1, lo)` when
/// only the lower bound is finite, `(-1, hi)` when only the upper is.
fn one_sided(lo: Number, hi: Number) -> Option<(Number, Number)> {
    match (lo > -INF, hi < INF) {
        (true, false) => Some((1.0, lo)),
        (false, true) => Some((-1.0, hi)),
        _ => None,
    }
}

/// A mixed pair: the variable boxed (two finite bounds, `l < u`) and the row
/// free (no finite bound).
pub fn is_mixed(xl: Number, xu: Number, gl: Number, gu: Number) -> bool {
    xl > -INF && xu < INF && xl < xu && gl <= -INF && gu >= INF
}

fn base(style: IndexStyle) -> Index {
    match style {
        IndexStyle::C => 0,
        IndexStyle::Fortran => 1,
    }
}

impl MpccTnlp {
    /// Wrap `inner` with `pairs` under `lowering`. Refuses, with the reason,
    /// a pair whose bounds are not the one-sided shape the module docs state,
    /// an out-of-range index, a variable or row used by two pairs, or an
    /// inner model that cannot report its structure.
    pub fn new(
        inner: Rc<RefCell<dyn TNLP>>,
        pairs: Vec<ComplementarityPair>,
        lowering: Lowering,
    ) -> Result<Self, String> {
        let info = inner
            .borrow_mut()
            .get_nlp_info()
            .ok_or("the inner model did not report its dimensions")?;
        let n = info.n as usize;
        let m = info.m as usize;
        let style = info.index_style;
        let off = base(style) as usize;

        let mut x_l = vec![0.0; n];
        let mut x_u = vec![0.0; n];
        let mut g_l = vec![0.0; m];
        let mut g_u = vec![0.0; m];
        if !inner.borrow_mut().get_bounds_info(BoundsInfo {
            x_l: &mut x_l,
            x_u: &mut x_u,
            g_l: &mut g_l,
            g_u: &mut g_u,
        }) {
            return Err("the inner model did not report its bounds".into());
        }

        let mut seen_row = vec![false; m];
        let mut seen_var = vec![false; n];
        // Each declared pair becomes one product row (one-sided) or two
        // (mixed); the rest of the wrapper works on these lowered pairs.
        let mut lowered = Vec::with_capacity(pairs.len());
        let mut g_side = Vec::with_capacity(pairs.len());
        let mut h_side = Vec::with_capacity(pairs.len());
        let mut signless = Vec::with_capacity(pairs.len());
        for (k, p) in pairs.iter().enumerate() {
            if p.var >= n || p.row >= m {
                return Err(format!(
                    "pair {k}: variable {} / row {} out of range (n = {n}, m = {m})",
                    p.var, p.row
                ));
            }
            if std::mem::replace(&mut seen_var[p.var], true) {
                return Err(format!("pair {k}: variable {} is already in a pair", p.var));
            }
            if std::mem::replace(&mut seen_row[p.row], true) {
                return Err(format!("pair {k}: row {} is already in a pair", p.row));
            }
            let (xl, xu, gl, gu) = (x_l[p.var], x_u[p.var], g_l[p.row], g_u[p.row]);
            if let (Some(gs), Some(hs)) = (one_sided(xl, xu), one_sided(gl, gu)) {
                lowered.push(*p);
                g_side.push(gs);
                h_side.push(hs);
                signless.push(false);
                continue;
            }
            // One-sided with an ordinary second bound on the variable: the
            // row's one finite side names the variable bound in the pair
            // (`g >= c` with `x`'s lower bound, `g <= c` with its upper, as
            // the `.nl` flag does), and the other bound stays a plain limit.
            // `complements(s >= 0, L >= 0)` with `L <= 1` declared writes this.
            if let Some(hs) = one_sided(gl, gu) {
                if xl > -INF && xu < INF {
                    lowered.push(*p);
                    g_side.push(if hs.0 > 0.0 { (1.0, xl) } else { (-1.0, xu) });
                    h_side.push(hs);
                    signless.push(false);
                    continue;
                }
            }
            // Mixed: `l <= x <= u ⟂ g` with the row free -- g >= 0 at l,
            // g <= 0 at u, g = 0 strictly between -- is exactly the box with
            // `(x - l)·g <= 0` and `(u - x)·(-g) <= 0`: two product rows on
            // the same variable and row, g itself unconstrained in sign.
            if is_mixed(xl, xu, gl, gu) {
                lowered.extend([*p, *p]);
                g_side.extend([(1.0, xl), (-1.0, xu)]);
                h_side.extend([(1.0, 0.0), (-1.0, 0.0)]);
                signless.extend([true, true]);
                continue;
            }
            return Err(format!(
                "pair {k}: variable {} has bounds [{xl:e}, {xu:e}] and row {} has \
                 [{gl:e}, {gu:e}]. Two shapes are lowered: one-sided (the row has exactly \
                 one finite bound, and the variable a matching one: `x >= l ⟂ g >= 0`, \
                 either orientation; a second variable bound is kept as an ordinary limit) \
                 and mixed (the variable boxed, the row free: `l <= x <= u ⟂ g`, g >= 0 at l, \
                 g <= 0 at u, g = 0 between). For a box on the row side, put it on a \
                 variable instead (`g(x) = y` as a row, `l <= y <= u ⟂ ...`)",
                p.var, p.row
            ));
        }
        let declared = pairs;
        let pairs = lowered;

        // Inner Jacobian structure, 0-based.
        let jnnz = info.nnz_jac_g as usize;
        let mut irow = vec![0 as Index; jnnz];
        let mut jcol = vec![0 as Index; jnnz];
        if !inner.borrow_mut().eval_jac_g(
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        ) {
            return Err("the inner model did not report its Jacobian structure".into());
        }
        let mut by_row: Vec<Vec<(usize, usize)>> = vec![Vec::new(); m];
        for k in 0..jnnz {
            by_row[irow[k] as usize - off].push((k, jcol[k] as usize - off));
        }
        let mut row_entries = Vec::with_capacity(pairs.len());
        let mut var_slot = Vec::with_capacity(pairs.len());
        let mut jac_block = Vec::with_capacity(pairs.len());
        let mut jac_nnz = jnnz;
        for p in &pairs {
            let ents = by_row[p.row].clone();
            let slot = ents
                .iter()
                .position(|&(_, c)| c == p.var)
                .unwrap_or(ents.len());
            jac_block.push(jac_nnz);
            jac_nnz += ents.len() + usize::from(slot == ents.len());
            var_slot.push(slot);
            row_entries.push(ents);
        }

        // Inner Hessian structure, 0-based, lower triangle.
        let hnnz = info.nnz_h_lag as usize;
        let mut h_pos = HashMap::new();
        if hnnz > 0 {
            let mut hr = vec![0 as Index; hnnz];
            let mut hc = vec![0 as Index; hnnz];
            if !inner.borrow_mut().eval_h(
                None,
                false,
                1.0,
                None,
                false,
                SparsityRequest::Structure {
                    irow: &mut hr,
                    jcol: &mut hc,
                },
            ) {
                return Err("the inner model did not report its Hessian structure".into());
            }
            for k in 0..hnnz {
                let (a, b) = (hr[k] as usize - off, hc[k] as usize - off);
                h_pos.insert((a.max(b), a.min(b)), k);
            }
        }
        let mut h_extra = Vec::new();
        // Cross entries are (j, c) for each column c of row i; the diagonal
        // (j, j) is touched only when x_j itself appears in row i.
        for (p, ents) in pairs.iter().zip(&row_entries) {
            for c in ents.iter().map(|&(_, c)| c) {
                let key = (p.var.max(c), p.var.min(c));
                if let std::collections::hash_map::Entry::Vacant(e) = h_pos.entry(key) {
                    e.insert(hnnz + h_extra.len());
                    h_extra.push(key);
                }
            }
        }

        Ok(Self {
            inner,
            lowering,
            n,
            m_inner: m,
            index_style: style,
            inner_jac_nnz: jnnz,
            inner_h_nnz: hnnz,
            g_side,
            h_side,
            row_entries,
            var_slot,
            jac_block,
            jac_nnz,
            h_extra,
            h_pos,
            g_inner: vec![0.0; m],
            jac_inner: vec![0.0; jnnz],
            lam_inner: vec![0.0; m],
            last_status: None,
            last_x: Vec::new(),
            last_lambda: Vec::new(),
            last_z_l: Vec::new(),
            last_z_u: Vec::new(),
            last_products: Vec::new(),
            defer_inner_finalize: false,
            start_override: None,
            branch: None,
            pairs: declared,
            lowered: pairs,
            signless,
        })
    }

    /// The pairs as declared (a mixed pair once, though it lowers to two
    /// product rows).
    pub fn pairs(&self) -> &[ComplementarityPair] {
        &self.pairs
    }

    pub fn lowering(&self) -> Lowering {
        self.lowering
    }

    /// Change how the product rows are bounded for the next solve.
    pub fn set_lowering(&mut self, lowering: Lowering) {
        self.lowering = lowering;
    }

    /// Solve a branch NLP instead of a lowering (`None` restores the
    /// lowering); see [`BranchBounds`].
    pub fn set_branch(&mut self, branch: Option<BranchBounds>) {
        self.branch = branch;
    }

    /// When set, `finalize_solution` records the result for the driver but
    /// does not forward it to the inner model, so a driver running several
    /// lowered solves forwards only the one it reports.
    pub fn set_defer_inner_finalize(&mut self, defer: bool) {
        self.defer_inner_finalize = defer;
    }

    /// Forget the last solve's captured result. A driver calls this before
    /// each solve, because `finalize_solution` — the only thing that writes
    /// the capture — is never reached by a solve refused before iteration 0
    /// (a degrees-of-freedom gate, a presolve proof, an invalid option): read
    /// without clearing, that solve would report the previous one's point.
    pub fn clear_capture(&mut self) {
        self.last_status = None;
        self.last_x.clear();
        self.last_lambda.clear();
        self.last_z_l.clear();
        self.last_z_u.clear();
        self.last_products.clear();
    }

    /// Start the next solves from `start` instead of the inner model's own
    /// starting point (`None` restores it). Only the parts the solver asks
    /// for are used: `x` always, `z_l`/`z_u` and `lambda` when it requests a
    /// warm start of the duals.
    pub fn set_start_override(&mut self, start: Option<StartOverride>) {
        self.start_override = start;
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn m_inner(&self) -> usize {
        self.m_inner
    }

    pub fn last_status(&self) -> Option<SolverReturn> {
        self.last_status
    }

    /// `x` of the last finalized solve (inner variables only; the wrapper
    /// adds none).
    pub fn last_x(&self) -> &[Number] {
        &self.last_x
    }

    /// Multipliers of the last finalized solve: inner rows, then one per
    /// product row.
    pub fn last_lambda(&self) -> &[Number] {
        &self.last_lambda
    }

    pub fn last_z_l(&self) -> &[Number] {
        &self.last_z_l
    }

    pub fn last_z_u(&self) -> &[Number] {
        &self.last_z_u
    }

    /// `G_k · H_k` at the last finalized `x`, one per product row.
    pub fn last_products(&self) -> &[Number] {
        &self.last_products
    }

    /// How far the last finalized `x` is from complementarity: the largest
    /// `|G·H|` over one-sided product rows and the largest positive part over
    /// a mixed pair's rows (whose products are legitimately negative at a
    /// bound, e.g. `(u - l)·g` with `g < 0` at `x = u`). `+inf` when the last
    /// solve returned no point.
    pub fn last_violation(&self) -> Number {
        if self.last_x.is_empty() {
            return Number::INFINITY;
        }
        self.last_products
            .iter()
            .zip(&self.signless)
            .fold(0.0, |a: Number, (&p, &sl)| {
                a.max(if sl { p.max(0.0) } else { p.abs() })
            })
    }

    /// `(G_k(x), H_k(x))` for every pair at `x`.
    pub fn pair_values(&mut self, x: &[Number]) -> Option<Vec<(Number, Number)>> {
        if !self.inner.borrow_mut().eval_g(x, true, &mut self.g_inner) {
            return None;
        }
        Some(self.pair_values_from(x))
    }

    fn pair_values_from(&self, x: &[Number]) -> Vec<(Number, Number)> {
        self.lowered
            .iter()
            .enumerate()
            .map(|(k, p)| {
                let ((sg, b), (sh, c)) = (self.g_side[k], self.h_side[k]);
                (sg * (x[p.var] - b), sh * (self.g_inner[p.row] - c))
            })
            .collect()
    }

    /// Hand the inner model the solution of a driver-chosen solve (used with
    /// [`Self::set_defer_inner_finalize`]); `lambda` may include the product
    /// rows' multipliers, which are dropped.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_finalize(
        &mut self,
        status: SolverReturn,
        x: &[Number],
        z_l: &[Number],
        z_u: &[Number],
        lambda: &[Number],
        ip_data: &IpoptData,
        ip_cq: &IpoptCq,
    ) {
        let mut inner = self.inner.borrow_mut();
        let f = inner.eval_f(x, true).unwrap_or(Number::NAN);
        let mut g = vec![0.0; self.m_inner];
        if self.m_inner > 0 {
            let _ = inner.eval_g(x, false, &mut g);
        }
        inner.finalize_solution(
            Solution {
                status,
                x,
                z_l,
                z_u,
                g: &g,
                lambda: &lambda[..self.m_inner.min(lambda.len())],
                obj_value: f,
            },
            ip_data,
            ip_cq,
        );
    }
}

impl TNLP for MpccTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as Index,
            m: (self.m_inner + self.lowered.len()) as Index,
            nnz_jac_g: self.jac_nnz as Index,
            nnz_h_lag: (self.inner_h_nnz + self.h_extra.len()) as Index,
            index_style: self.index_style,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        let m = self.m_inner;
        if !self.inner.borrow_mut().get_bounds_info(BoundsInfo {
            x_l: b.x_l,
            x_u: b.x_u,
            g_l: &mut b.g_l[..m],
            g_u: &mut b.g_u[..m],
        }) {
            return false;
        }
        let (lo, hi) = match self.lowering {
            Lowering::Eq => (0.0, 0.0),
            Lowering::Ineq => (-2.0 * INF, 0.0),
            Lowering::Relaxed(tau) => (-2.0 * INF, tau),
        };
        if let Some(br) = &self.branch {
            // A branch NLP: the pinned sides carry the complementarity, and
            // the product rows constrain nothing.
            for &(j, l, u) in &br.vars {
                b.x_l[j] = l;
                b.x_u[j] = u;
            }
            for &(i, l, u) in &br.rows {
                b.g_l[i] = l;
                b.g_u[i] = u;
            }
            for k in 0..self.lowered.len() {
                b.g_l[m + k] = -2.0 * INF;
                b.g_u[m + k] = 2.0 * INF;
            }
            return true;
        }
        for k in 0..self.lowered.len() {
            // A mixed pair's rows are inequalities under every lowering.
            b.g_l[m + k] = if self.signless[k] { -2.0 * INF } else { lo };
            b.g_u[m + k] = hi;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if let Some(st) = &self.start_override {
            if st.x.len() != sp.x.len()
                || (sp.init_z && (st.z_l.len() != sp.z_l.len() || st.z_u.len() != sp.z_u.len()))
                || (sp.init_lambda && st.lambda.len() != sp.lambda.len())
            {
                return false;
            }
            sp.x.copy_from_slice(&st.x);
            if sp.init_z {
                sp.z_l.copy_from_slice(&st.z_l);
                sp.z_u.copy_from_slice(&st.z_u);
            }
            if sp.init_lambda {
                sp.lambda.copy_from_slice(&st.lambda);
            }
            return true;
        }
        let m = self.m_inner;
        let (head, tail) = sp.lambda.split_at_mut(m.min(sp.lambda.len()));
        if !self.inner.borrow_mut().get_starting_point(StartingPoint {
            init_x: sp.init_x,
            x: sp.x,
            init_z: sp.init_z,
            z_l: sp.z_l,
            z_u: sp.z_u,
            init_lambda: sp.init_lambda,
            lambda: head,
        }) {
            return false;
        }
        if sp.init_lambda {
            tail.fill(0.0);
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.inner.borrow_mut().eval_f(x, new_x)
    }

    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        self.inner.borrow_mut().eval_grad_f(x, new_x, grad_f)
    }

    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        let m = self.m_inner;
        if !self.inner.borrow_mut().eval_g(x, new_x, &mut g[..m]) {
            return false;
        }
        self.g_inner.copy_from_slice(&g[..m]);
        for (k, (gk, hk)) in self.pair_values_from(x).into_iter().enumerate() {
            g[m + k] = gk * hk;
        }
        true
    }

    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        let jnnz = self.inner_jac_nnz;
        let m = self.m_inner;
        let off = base(self.index_style);
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                if !self.inner.borrow_mut().eval_jac_g(
                    None,
                    false,
                    SparsityRequest::Structure {
                        irow: &mut irow[..jnnz],
                        jcol: &mut jcol[..jnnz],
                    },
                ) {
                    return false;
                }
                for (k, p) in self.lowered.iter().enumerate() {
                    let b = self.jac_block[k];
                    let row = (m + k) as Index + off;
                    let ents = &self.row_entries[k];
                    for (s, &(_, c)) in ents.iter().enumerate() {
                        irow[b + s] = row;
                        jcol[b + s] = c as Index + off;
                    }
                    if self.var_slot[k] == ents.len() {
                        irow[b + ents.len()] = row;
                        jcol[b + ents.len()] = p.var as Index + off;
                    }
                }
                true
            }
            SparsityRequest::Values { values } => {
                let Some(x) = x else { return false };
                {
                    let mut inner = self.inner.borrow_mut();
                    if !inner.eval_jac_g(
                        Some(x),
                        new_x,
                        SparsityRequest::Values {
                            values: &mut values[..jnnz],
                        },
                    ) {
                        return false;
                    }
                    if !inner.eval_g(x, false, &mut self.g_inner) {
                        return false;
                    }
                }
                let pv = self.pair_values_from(x);
                for (k, &(gk, hk)) in pv.iter().enumerate() {
                    let (sg, sh) = (self.g_side[k].0, self.h_side[k].0);
                    let b = self.jac_block[k];
                    let ents = &self.row_entries[k];
                    // G ∇g_i on row i's columns ...
                    for (s, &(pos, _)) in ents.iter().enumerate() {
                        values[b + s] = gk * sh * values[pos];
                    }
                    // ... plus H on column j, in place or appended.
                    let slot = self.var_slot[k];
                    if slot == ents.len() {
                        values[b + slot] = hk * sg;
                    } else {
                        values[b + slot] += hk * sg;
                    }
                }
                true
            }
        }
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let hnnz = self.inner_h_nnz;
        let m = self.m_inner;
        let off = base(self.index_style);
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                if hnnz > 0
                    && !self.inner.borrow_mut().eval_h(
                        None,
                        false,
                        obj_factor,
                        None,
                        false,
                        SparsityRequest::Structure {
                            irow: &mut irow[..hnnz],
                            jcol: &mut jcol[..hnnz],
                        },
                    )
                {
                    return false;
                }
                for (e, &(a, b)) in self.h_extra.iter().enumerate() {
                    irow[hnnz + e] = a as Index + off;
                    jcol[hnnz + e] = b as Index + off;
                }
                true
            }
            SparsityRequest::Values { values } => {
                let (Some(x), Some(lambda)) = (x, lambda) else {
                    return false;
                };
                {
                    let mut inner = self.inner.borrow_mut();
                    if !inner.eval_g(x, new_x, &mut self.g_inner) {
                        return false;
                    }
                }
                let pv = self.pair_values_from(x);
                // Row-i multipliers shifted by μ_k G_k.
                self.lam_inner.copy_from_slice(&lambda[..m]);
                for (k, p) in self.lowered.iter().enumerate() {
                    self.lam_inner[p.row] += lambda[m + k] * pv[k].0 * self.h_side[k].0;
                }
                values.fill(0.0);
                {
                    let mut inner = self.inner.borrow_mut();
                    if hnnz > 0
                        && !inner.eval_h(
                            Some(x),
                            false,
                            obj_factor,
                            Some(&self.lam_inner),
                            new_lambda || !self.lowered.is_empty(),
                            SparsityRequest::Values {
                                values: &mut values[..hnnz],
                            },
                        )
                    {
                        return false;
                    }
                    if !self.lowered.is_empty()
                        && !inner.eval_jac_g(
                            Some(x),
                            false,
                            SparsityRequest::Values {
                                values: &mut self.jac_inner,
                            },
                        )
                    {
                        return false;
                    }
                }
                // Cross terms μ_k (e_j ∇g_iᵀ + ∇g_i e_jᵀ), lower triangle.
                for (k, p) in self.lowered.iter().enumerate() {
                    let mu = lambda[m + k] * self.g_side[k].0 * self.h_side[k].0;
                    if mu == 0.0 {
                        continue;
                    }
                    for &(pos, c) in &self.row_entries[k] {
                        let d = mu * self.jac_inner[pos];
                        let key = (p.var.max(c), p.var.min(c));
                        let at = self.h_pos[&key];
                        values[at] += if c == p.var { 2.0 * d } else { d };
                    }
                }
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        let m = self.m_inner;
        self.last_status = Some(sol.status);
        self.last_x = sol.x.to_vec();
        self.last_lambda = sol.lambda.to_vec();
        self.last_z_l = sol.z_l.to_vec();
        self.last_z_u = sol.z_u.to_vec();
        self.last_products = sol.g.get(m..).map(<[Number]>::to_vec).unwrap_or_default();
        if self.defer_inner_finalize {
            return;
        }
        self.inner.borrow_mut().finalize_solution(
            Solution {
                status: sol.status,
                x: sol.x,
                z_l: sol.z_l,
                z_u: sol.z_u,
                g: &sol.g[..m.min(sol.g.len())],
                lambda: &sol.lambda[..m.min(sol.lambda.len())],
                obj_value: sol.obj_value,
            },
            ip_data,
            ip_cq,
        );
    }

    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        // Variable metadata passes through unchanged; constraint metadata is
        // extended to the product rows (named `complementarity[k]` where the
        // inner model names its rows), so names stay index-aligned.
        let m_total = self.m_inner + self.lowered.len();
        let mut inner_con = MetaData::default();
        let ok = self
            .inner
            .borrow_mut()
            .get_var_con_metadata(var, &mut inner_con);
        if !ok {
            return false;
        }
        for (key, v) in inner_con.strings {
            let mut v = v;
            if v.len() == self.m_inner {
                v.extend((0..self.lowered.len()).map(|k| format!("complementarity[{k}]")));
            }
            v.resize(m_total, String::new());
            con.strings.insert(key, v);
        }
        for (key, mut v) in inner_con.integers {
            v.resize(m_total, 0);
            con.integers.insert(key, v);
        }
        for (key, mut v) in inner_con.numerics {
            v.resize(m_total, 0.0);
            con.numerics.insert(key, v);
        }
        true
    }

    // ---- Hooks forwarded to the inner model (PR review of #1103: falling
    // back to the trait defaults dropped the caller's intermediate callback,
    // user scaling, linearity and derivative proofs). Anything indexed by
    // constraint row is translated: the product rows are appended after the
    // inner model's `m_inner` rows.

    fn wants_intermediate_callback(&self) -> bool {
        self.inner.borrow().wants_intermediate_callback()
    }

    fn intermediate_callback(
        &mut self,
        stats: IterStats,
        ip_data: &IpoptData,
        ip_cq: &IpoptCq,
    ) -> bool {
        self.inner
            .borrow_mut()
            .intermediate_callback(stats, ip_data, ip_cq)
    }

    fn wants_current_iterate(&self) -> bool {
        self.inner.borrow().wants_current_iterate()
    }

    fn scaling_factors(&self) -> Option<Vec<Number>> {
        self.inner.borrow().scaling_factors()
    }

    /// The inner model's factors for its own rows; the product rows are
    /// unscaled.
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        let ScalingRequest {
            obj_scaling,
            use_x_scaling,
            x_scaling,
            use_g_scaling,
            g_scaling,
        } = req;
        let (inner_g, products) = g_scaling.split_at_mut(self.m_inner.min(g_scaling.len()));
        let ok = self
            .inner
            .borrow_mut()
            .get_scaling_parameters(ScalingRequest {
                obj_scaling,
                use_x_scaling,
                x_scaling,
                use_g_scaling,
                g_scaling: inner_g,
            });
        products.fill(1.0);
        ok
    }

    fn get_objective_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner
            .borrow_mut()
            .get_objective_variables_linearity(types)
    }

    /// The inner model's rows as it reports them; the product rows are
    /// nonlinear.
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        let m = self.m_inner.min(types.len());
        let (inner, products) = types.split_at_mut(m);
        if !self.inner.borrow_mut().get_constraints_linearity(inner) {
            return false;
        }
        products.fill(Linearity::NonLinear);
        true
    }

    /// The inner model's variables as it reports them, with every variable a
    /// product touches (each pair's variable and its row's columns) made
    /// nonlinear.
    fn get_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        if !self.inner.borrow_mut().get_variables_linearity(types) {
            return false;
        }
        for j in self.product_vars() {
            if let Some(t) = types.get_mut(j) {
                *t = Linearity::NonLinear;
            }
        }
        true
    }

    fn get_number_of_nonlinear_variables(&mut self) -> Index {
        match self.nonlinear_vars() {
            Some(v) => v.len() as Index,
            None => -1,
        }
    }

    fn get_list_of_nonlinear_variables(&mut self, pos: &mut [Index]) -> bool {
        let Some(v) = self.nonlinear_vars() else {
            return false;
        };
        for (p, j) in pos.iter_mut().zip(v) {
            *p = j;
        }
        true
    }

    /// The inner proofs; the product rows' gradients vary with `x`, and the
    /// Lagrangian Hessian, which now carries the products' terms, is not
    /// claimed either way.
    fn derivative_proofs(&mut self) -> DerivativeProofs {
        let mut p = self.inner.borrow_mut().derivative_proofs();
        if !p.jac.is_empty() {
            p.jac.resize(self.m_inner, DerivativeProof::Unknown);
            p.jac.extend(std::iter::repeat_n(
                DerivativeProof::Varying,
                self.lowered.len(),
            ));
        }
        p.hessian = DerivativeProof::Unknown;
        p
    }

    /// The constraint metadata truncated to the inner model's rows.
    fn finalize_metadata(&mut self, var: &MetaData, con: &MetaData) {
        let m = self.m_inner;
        let mut inner_con = con.clone();
        for v in inner_con.strings.values_mut() {
            v.truncate(m);
        }
        for v in inner_con.integers.values_mut() {
            v.truncate(m);
        }
        for v in inner_con.numerics.values_mut() {
            v.truncate(m);
        }
        self.inner.borrow_mut().finalize_metadata(var, &inner_con);
    }
}

impl MpccTnlp {
    /// Every variable a product row touches (0-based): each pair's variable
    /// and the columns of its row.
    fn product_vars(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.lowered.iter().map(|p| p.var).collect();
        for entries in &self.row_entries {
            v.extend(entries.iter().map(|&(_, col)| col));
        }
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The inner model's nonlinear variables plus the product-touched ones,
    /// in the index style's numbering; `None` if the inner model reports
    /// every variable nonlinear (or does not say).
    fn nonlinear_vars(&mut self) -> Option<Vec<Index>> {
        let n_inner = self.inner.borrow_mut().get_number_of_nonlinear_variables();
        if n_inner < 0 {
            return None;
        }
        let mut list = vec![0 as Index; n_inner as usize];
        if n_inner > 0
            && !self
                .inner
                .borrow_mut()
                .get_list_of_nonlinear_variables(&mut list)
        {
            return None;
        }
        let off = base(self.index_style);
        list.extend(self.product_vars().into_iter().map(|j| j as Index + off));
        list.sort_unstable();
        list.dedup();
        Some(list)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// min (x0-1)^2 + x1^2 x2 + exp(x2)
    /// s.t. row 0: g0 = x0 + x1^2 - x2 x0   >= 0.5      (paired with x1 >= 0)
    ///      row 1: g1 = sin(x0) + x2^2      >= 0        (paired with x2 >= -1)
    ///      row 2: g2 = x0 x1 x2            in [-10, 10]
    ///
    /// Nonlinear in every argument the pairs touch, and each pair's variable
    /// appears in its own row, so the `x_j` Jacobian entry is accumulated in
    /// place; the appended path is `appended_entries`'s job.
    struct Toy {
        style: IndexStyle,
    }

    impl Toy {
        fn g(x: &[Number]) -> [Number; 3] {
            [
                x[0] + x[1] * x[1] - x[2] * x[0],
                x[0].sin() + x[2] * x[2],
                x[0] * x[1] * x[2],
            ]
        }
        /// Dense Jacobian, rows x cols.
        fn jac(x: &[Number]) -> [[Number; 3]; 3] {
            [
                [1.0 - x[2], 2.0 * x[1], -x[0]],
                [x[0].cos(), 0.0, 2.0 * x[2]],
                [x[1] * x[2], x[0] * x[2], x[0] * x[1]],
            ]
        }
        /// Dense Lagrangian Hessian.
        fn hess(x: &[Number], s: Number, l: &[Number]) -> [[Number; 3]; 3] {
            let mut h = [[0.0; 3]; 3];
            // objective
            h[0][0] += s * 2.0;
            h[1][1] += s * 2.0 * x[2];
            h[1][2] += s * 2.0 * x[1];
            h[2][1] += s * 2.0 * x[1];
            h[2][2] += s * x[2].exp();
            // g0
            h[1][1] += l[0] * 2.0;
            h[0][2] += -l[0];
            h[2][0] += -l[0];
            // g1
            h[0][0] += -l[1] * x[0].sin();
            h[2][2] += l[1] * 2.0;
            // g2
            h[0][1] += l[2] * x[2];
            h[1][0] += l[2] * x[2];
            h[0][2] += l[2] * x[1];
            h[2][0] += l[2] * x[1];
            h[1][2] += l[2] * x[0];
            h[2][1] += l[2] * x[0];
            h
        }
    }

    // Sparse patterns: the Jacobian drops row 1's structural zero at column
    // 1; the Hessian declares the full lower triangle g2 needs, so every
    // cross entry of these pairs already exists and is accumulated in place.
    const JAC: [(usize, usize); 8] = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ];
    const HES: [(usize, usize); 6] = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)];

    impl TNLP for Toy {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 3,
                m: 3,
                nnz_jac_g: JAC.len() as Index,
                nnz_h_lag: HES.len() as Index,
                index_style: self.style,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[-5.0, 0.0, -1.0]);
            b.x_u.copy_from_slice(&[5.0, 1e19, 1e19]);
            b.g_l.copy_from_slice(&[0.5, 0.0, -10.0]);
            b.g_u.copy_from_slice(&[1e19, 1e19, 10.0]);
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x.copy_from_slice(&[0.3, 0.4, 0.5]);
            if sp.init_lambda {
                sp.lambda.fill(7.0);
            }
            true
        }
        fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
            Some((x[0] - 1.0).powi(2) + x[1] * x[1] * x[2] + x[2].exp())
        }
        fn eval_grad_f(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = 2.0 * (x[0] - 1.0);
            g[1] = 2.0 * x[1] * x[2];
            g[2] = x[1] * x[1] + x[2].exp();
            true
        }
        fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
            g.copy_from_slice(&Self::g(x));
            true
        }
        fn eval_jac_g(&mut self, x: Option<&[Number]>, _: bool, mode: SparsityRequest<'_>) -> bool {
            let o = base(self.style);
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    for (k, &(r, c)) in JAC.iter().enumerate() {
                        irow[k] = r as Index + o;
                        jcol[k] = c as Index + o;
                    }
                }
                SparsityRequest::Values { values } => {
                    let j = Self::jac(x.unwrap());
                    for (k, &(r, c)) in JAC.iter().enumerate() {
                        values[k] = j[r][c];
                    }
                }
            }
            true
        }
        fn eval_h(
            &mut self,
            x: Option<&[Number]>,
            _: bool,
            s: Number,
            l: Option<&[Number]>,
            _: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            let o = base(self.style);
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    for (k, &(r, c)) in HES.iter().enumerate() {
                        irow[k] = r as Index + o;
                        jcol[k] = c as Index + o;
                    }
                }
                SparsityRequest::Values { values } => {
                    let h = Self::hess(x.unwrap(), s, l.unwrap());
                    for (k, &(r, c)) in HES.iter().enumerate() {
                        values[k] = h[r][c];
                    }
                }
            }
            true
        }
        fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
    }

    fn pairs() -> Vec<ComplementarityPair> {
        vec![
            ComplementarityPair { row: 0, var: 1 },
            ComplementarityPair { row: 1, var: 2 },
        ]
    }

    fn wrap(style: IndexStyle, lowering: Lowering) -> MpccTnlp {
        MpccTnlp::new(Rc::new(RefCell::new(Toy { style })), pairs(), lowering).unwrap()
    }

    /// Dense lowered Jacobian / Hessian read back through the wrapper's own
    /// sparse interface.
    fn dense_jac(w: &mut MpccTnlp, x: &[Number]) -> Vec<Vec<Number>> {
        let info = w.get_nlp_info().unwrap();
        let (m, nnz, o) = (
            info.m as usize,
            info.nnz_jac_g as usize,
            base(info.index_style) as usize,
        );
        let mut ir = vec![0; nnz];
        let mut jc = vec![0; nnz];
        assert!(w.eval_jac_g(
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut ir,
                jcol: &mut jc
            }
        ));
        let mut v = vec![0.0; nnz];
        assert!(w.eval_jac_g(Some(x), true, SparsityRequest::Values { values: &mut v }));
        let mut d = vec![vec![0.0; 3]; m];
        let mut seen = std::collections::HashSet::new();
        for k in 0..nnz {
            let key = (ir[k] as usize - o, jc[k] as usize - o);
            assert!(seen.insert(key), "duplicate Jacobian entry {key:?}");
            d[key.0][key.1] += v[k];
        }
        d
    }

    fn dense_hess(w: &mut MpccTnlp, x: &[Number], s: Number, l: &[Number]) -> [[Number; 3]; 3] {
        let info = w.get_nlp_info().unwrap();
        let (nnz, o) = (info.nnz_h_lag as usize, base(info.index_style) as usize);
        let mut ir = vec![0; nnz];
        let mut jc = vec![0; nnz];
        assert!(w.eval_h(
            None,
            false,
            1.0,
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut ir,
                jcol: &mut jc
            }
        ));
        let mut v = vec![0.0; nnz];
        assert!(w.eval_h(
            Some(x),
            true,
            s,
            Some(l),
            true,
            SparsityRequest::Values { values: &mut v }
        ));
        let mut d = [[0.0; 3]; 3];
        let mut seen = std::collections::HashSet::new();
        for k in 0..nnz {
            let (a, b) = (ir[k] as usize - o, jc[k] as usize - o);
            assert!(a >= b, "upper-triangle Hessian entry ({a},{b})");
            assert!(seen.insert((a, b)), "duplicate Hessian entry ({a},{b})");
            d[a][b] += v[k];
            if a != b {
                d[b][a] += v[k];
            }
        }
        d
    }

    fn lowered_g(w: &mut MpccTnlp, x: &[Number]) -> Vec<Number> {
        let m = w.get_nlp_info().unwrap().m as usize;
        let mut g = vec![0.0; m];
        assert!(w.eval_g(x, true, &mut g));
        g
    }

    const X: [Number; 3] = [0.7, 0.9, 0.35];

    #[test]
    fn products_are_g_times_h() {
        let mut w = wrap(IndexStyle::C, Lowering::Eq);
        let g = lowered_g(&mut w, &X);
        let gi = Toy::g(&X);
        assert_eq!(&g[..3], &gi);
        assert!((g[3] - (X[1] - 0.0) * (gi[0] - 0.5)).abs() < 1e-15);
        assert!((g[4] - (X[2] + 1.0) * (gi[1] - 0.0)).abs() < 1e-15);
    }

    #[test]
    fn bounds_follow_the_lowering() {
        for (low, lo, hi) in [
            (Lowering::Eq, 0.0, 0.0),
            (Lowering::Ineq, -2e19, 0.0),
            (Lowering::Relaxed(1e-3), -2e19, 1e-3),
        ] {
            let mut w = wrap(IndexStyle::C, low);
            let (mut xl, mut xu, mut gl, mut gu) =
                (vec![0.0; 3], vec![0.0; 3], vec![0.0; 5], vec![0.0; 5]);
            assert!(w.get_bounds_info(BoundsInfo {
                x_l: &mut xl,
                x_u: &mut xu,
                g_l: &mut gl,
                g_u: &mut gu
            }));
            assert_eq!((gl[3], gu[3], gl[4], gu[4]), (lo, hi, lo, hi));
            assert_eq!(gl[..3], [0.5, 0.0, -10.0]);
        }
    }

    /// The lowered Jacobian against central differences of the lowered
    /// constraints, in both index styles.
    #[test]
    fn jacobian_matches_finite_differences() {
        for style in [IndexStyle::C, IndexStyle::Fortran] {
            let mut w = wrap(style, Lowering::Eq);
            let j = dense_jac(&mut w, &X);
            let h = 1e-6;
            for c in 0..3 {
                let (mut xp, mut xm) = (X, X);
                xp[c] += h;
                xm[c] -= h;
                let (gp, gm) = (lowered_g(&mut w, &xp), lowered_g(&mut w, &xm));
                for r in 0..5 {
                    let fd = (gp[r] - gm[r]) / (2.0 * h);
                    assert!(
                        (j[r][c] - fd).abs() < 1e-8,
                        "{style:?} J[{r}][{c}] = {} vs fd {fd}",
                        j[r][c]
                    );
                }
            }
        }
    }

    /// The lowered Lagrangian Hessian against central differences of the
    /// lowered Lagrangian gradient (∇f s + Jᵀλ), with nonzero multipliers on
    /// every row including the products.
    #[test]
    fn hessian_matches_finite_differences() {
        let lam = [0.3, -1.2, 0.8, 2.5, -0.7];
        let s = 1.7;
        for style in [IndexStyle::C, IndexStyle::Fortran] {
            let mut w = wrap(style, Lowering::Eq);
            let hd = dense_hess(&mut w, &X, s, &lam);
            let grad_l = |w: &mut MpccTnlp, x: &[Number]| -> Vec<Number> {
                let mut gf = vec![0.0; 3];
                assert!(w.eval_grad_f(x, true, &mut gf));
                let j = dense_jac(w, x);
                (0..3)
                    .map(|c| s * gf[c] + (0..5).map(|r| lam[r] * j[r][c]).sum::<Number>())
                    .collect()
            };
            let h = 1e-6;
            for c in 0..3 {
                let (mut xp, mut xm) = (X, X);
                xp[c] += h;
                xm[c] -= h;
                let (gp, gm) = (grad_l(&mut w, &xp), grad_l(&mut w, &xm));
                for r in 0..3 {
                    let fd = (gp[r] - gm[r]) / (2.0 * h);
                    assert!(
                        (hd[r][c] - fd).abs() < 1e-7,
                        "{style:?} H[{r}][{c}] = {} vs fd {fd}",
                        hd[r][c]
                    );
                }
            }
        }
    }

    /// `Toy` with row 1 free, so the pair `(row 1, x0)` — `x0` boxed in
    /// `[-5, 5]` — is mixed: `-5 <= x0 <= 5 ⟂ g1`.
    struct MixedToy(Toy);

    impl TNLP for MixedToy {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            self.0.get_nlp_info()
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[-5.0, 0.0, -1.0]);
            b.x_u.copy_from_slice(&[5.0, 1e19, 1e19]);
            b.g_l.copy_from_slice(&[0.5, -1e19, -10.0]);
            b.g_u.copy_from_slice(&[1e19, 1e19, 10.0]);
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            self.0.get_starting_point(sp)
        }
        fn eval_f(&mut self, x: &[Number], n: bool) -> Option<Number> {
            self.0.eval_f(x, n)
        }
        fn eval_grad_f(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            self.0.eval_grad_f(x, n, g)
        }
        fn eval_g(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            self.0.eval_g(x, n, g)
        }
        fn eval_jac_g(&mut self, x: Option<&[Number]>, n: bool, mode: SparsityRequest<'_>) -> bool {
            self.0.eval_jac_g(x, n, mode)
        }
        fn eval_h(
            &mut self,
            x: Option<&[Number]>,
            n: bool,
            s: Number,
            l: Option<&[Number]>,
            nl: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            self.0.eval_h(x, n, s, l, nl, mode)
        }
        fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
    }

    fn wrap_mixed(style: IndexStyle, lowering: Lowering) -> MpccTnlp {
        MpccTnlp::new(
            Rc::new(RefCell::new(MixedToy(Toy { style }))),
            vec![
                ComplementarityPair { row: 0, var: 1 },
                ComplementarityPair { row: 1, var: 0 },
            ],
            lowering,
        )
        .unwrap()
    }

    /// A mixed pair lowers to two product rows, `(x - l)·g <= 0` and
    /// `(u - x)·(-g) <= 0`, bounded above under every lowering (never `= 0`:
    /// the product is legitimately nonzero at a bound), with derivatives that
    /// match finite differences; the declared pairs stay as declared.
    #[test]
    fn a_mixed_pair_lowers_to_two_inequality_rows() {
        let mut w = wrap_mixed(IndexStyle::C, Lowering::Eq);
        assert_eq!(w.pairs().len(), 2, "declared pairs");
        assert_eq!(
            w.get_nlp_info().unwrap().m,
            6,
            "3 rows + 1 one-sided + 2 mixed"
        );

        let gi = Toy::g(&X);
        let g = lowered_g(&mut w, &X);
        assert!((g[3] - X[1] * (gi[0] - 0.5)).abs() < 1e-15);
        assert!((g[4] - (X[0] + 5.0) * gi[1]).abs() < 1e-15);
        assert!((g[5] - (5.0 - X[0]) * (-gi[1])).abs() < 1e-15);

        for (low, lo, hi) in [
            (Lowering::Eq, 0.0, 0.0),
            (Lowering::Ineq, -2e19, 0.0),
            (Lowering::Relaxed(1e-3), -2e19, 1e-3),
        ] {
            let mut w = wrap_mixed(IndexStyle::C, low);
            let (mut xl, mut xu, mut gl, mut gu) =
                (vec![0.0; 3], vec![0.0; 3], vec![0.0; 6], vec![0.0; 6]);
            assert!(w.get_bounds_info(BoundsInfo {
                x_l: &mut xl,
                x_u: &mut xu,
                g_l: &mut gl,
                g_u: &mut gu
            }));
            assert_eq!(
                (gl[3], gu[3]),
                (lo, hi),
                "the one-sided row follows the lowering"
            );
            for k in [4, 5] {
                assert_eq!((gl[k], gu[k]), (-2e19, hi), "mixed row {k} under {low:?}");
            }
        }

        for style in [IndexStyle::C, IndexStyle::Fortran] {
            let mut w = wrap_mixed(style, Lowering::Eq);
            let j = dense_jac(&mut w, &X);
            let lam = [0.3, -1.2, 0.8, 2.5, -0.7, 1.9];
            let s = 1.7;
            let hd = dense_hess(&mut w, &X, s, &lam);
            let grad_l = |w: &mut MpccTnlp, x: &[Number]| -> Vec<Number> {
                let mut gf = vec![0.0; 3];
                assert!(w.eval_grad_f(x, true, &mut gf));
                let j = dense_jac(w, x);
                (0..3)
                    .map(|c| s * gf[c] + (0..6).map(|r| lam[r] * j[r][c]).sum::<Number>())
                    .collect()
            };
            let h = 1e-6;
            for c in 0..3 {
                let (mut xp, mut xm) = (X, X);
                xp[c] += h;
                xm[c] -= h;
                let (gp, gm) = (lowered_g(&mut w, &xp), lowered_g(&mut w, &xm));
                for r in 0..6 {
                    let fd = (gp[r] - gm[r]) / (2.0 * h);
                    assert!((j[r][c] - fd).abs() < 1e-8, "{style:?} J[{r}][{c}]");
                }
                let (lp, lm) = (grad_l(&mut w, &xp), grad_l(&mut w, &xm));
                for r in 0..3 {
                    let fd = (lp[r] - lm[r]) / (2.0 * h);
                    assert!((hd[r][c] - fd).abs() < 1e-7, "{style:?} H[{r}][{c}]");
                }
            }
        }
    }

    /// The violation the driver reads: `|G·H|` on a one-sided row, the
    /// positive part on a mixed one. At `x0 = u` with `g1 < 0` the mixed pair
    /// is satisfied though its first product, `(u - l)·g1`, is far from 0.
    #[test]
    fn a_mixed_pair_at_its_upper_bound_is_not_a_violation() {
        let mut w = wrap_mixed(IndexStyle::C, Lowering::Eq);
        assert_eq!(w.last_violation(), Number::INFINITY, "no point yet");
        // x0 = 5 (its upper bound) and x2 = 0, so g1 = sin 5 + x2² ≈ -0.959.
        let x = [5.0, 0.9, 0.0];
        let gi = Toy::g(&x);
        assert!(gi[1] < 0.0);
        let g = lowered_g(&mut w, &x);
        let zeros = [0.0; 3];
        w.finalize_solution(
            Solution {
                status: SolverReturn::Success,
                x: &x,
                z_l: &zeros,
                z_u: &zeros,
                g: &g,
                lambda: &[0.0; 6],
                obj_value: 0.0,
            },
            &IpoptData::default(),
            &IpoptCq::default(),
        );
        let (p_one, p_lo, p_hi) = (g[3], g[4], g[5]);
        assert!(p_lo < -1.0 && p_hi.abs() < 1e-15, "{p_lo} {p_hi}");
        assert!((w.last_violation() - p_one.abs()).abs() < 1e-15);
    }

    /// A pair whose variable does not appear in its row: the Jacobian entry
    /// for `x_j` is appended rather than accumulated, and the cross Hessian
    /// entries the inner model lacks are appended.
    #[test]
    fn appended_entries() {
        struct Sparse;
        impl TNLP for Sparse {
            fn get_nlp_info(&mut self) -> Option<NlpInfo> {
                Some(NlpInfo {
                    n: 2,
                    m: 1,
                    nnz_jac_g: 1,
                    nnz_h_lag: 1,
                    index_style: IndexStyle::C,
                })
            }
            fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
                b.x_l.copy_from_slice(&[-1e19, 0.0]);
                b.x_u.copy_from_slice(&[1e19, 1e19]);
                b.g_l[0] = 0.0;
                b.g_u[0] = 1e19;
                true
            }
            fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
                sp.x.copy_from_slice(&[1.0, 1.0]);
                true
            }
            fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
                Some(x[0] * x[0])
            }
            fn eval_grad_f(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
                g.copy_from_slice(&[2.0 * x[0], 0.0]);
                true
            }
            // g0 = x0^3, paired with x1: x1 does not appear in row 0.
            fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
                g[0] = x[0].powi(3);
                true
            }
            fn eval_jac_g(
                &mut self,
                x: Option<&[Number]>,
                _: bool,
                mode: SparsityRequest<'_>,
            ) -> bool {
                match mode {
                    SparsityRequest::Structure { irow, jcol } => {
                        irow[0] = 0;
                        jcol[0] = 0;
                    }
                    SparsityRequest::Values { values } => values[0] = 3.0 * x.unwrap()[0].powi(2),
                }
                true
            }
            fn eval_h(
                &mut self,
                x: Option<&[Number]>,
                _: bool,
                s: Number,
                l: Option<&[Number]>,
                _: bool,
                mode: SparsityRequest<'_>,
            ) -> bool {
                match mode {
                    SparsityRequest::Structure { irow, jcol } => {
                        irow[0] = 0;
                        jcol[0] = 0;
                    }
                    SparsityRequest::Values { values } => {
                        values[0] = 2.0 * s + l.unwrap()[0] * 6.0 * x.unwrap()[0];
                    }
                }
                true
            }
            fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
        }
        let mut w = MpccTnlp::new(
            Rc::new(RefCell::new(Sparse)),
            vec![ComplementarityPair { row: 0, var: 1 }],
            Lowering::Eq,
        )
        .unwrap();
        let info = w.get_nlp_info().unwrap();
        assert_eq!(info.nnz_jac_g, 3); // inner (0,0) + product (1,0), (1,1)
        assert_eq!(info.nnz_h_lag, 2); // inner (0,0) + appended cross (1,0)
        let x = [0.6, 0.25];
        // p = x1 * x0^3:  ∂p/∂x0 = 3 x1 x0^2, ∂p/∂x1 = x0^3
        let mut v = vec![0.0; 3];
        assert!(w.eval_jac_g(Some(&x), true, SparsityRequest::Values { values: &mut v }));
        assert!((v[1] - 3.0 * x[1] * x[0] * x[0]).abs() < 1e-15);
        assert!((v[2] - x[0].powi(3)).abs() < 1e-15);
        // L = s f + l0 g0 + mu p: H00 = 2s + 6 x0 (l0 + mu x1), H10 = 3 mu x0^2
        let (s, l0, mu) = (1.3, 0.4, -2.0);
        let mut h = vec![0.0; 2];
        assert!(w.eval_h(
            Some(&x),
            true,
            s,
            Some(&[l0, mu]),
            true,
            SparsityRequest::Values { values: &mut h }
        ));
        assert!((h[0] - (2.0 * s + 6.0 * x[0] * (l0 + mu * x[1]))).abs() < 1e-13);
        assert!((h[1] - 3.0 * mu * x[0] * x[0]).abs() < 1e-13);
    }

    /// The toy model with its bounds flipped so the pairs take the other
    /// orientations: pair (row 0, var 1) has `x_1 <= 2 ⟂ g_0 <= 0.5` (both
    /// sides upper-bounded, `s_G = s_H = -1`); pair (row 1, var 2) has
    /// `x_2 >= -1 ⟂ g_1 <= 3` (mixed, `s_G = +1, s_H = -1`).
    /// `Toy` reporting every row linear and row scaling `[2, 3, 4]`.
    struct Hooked(Toy);

    impl TNLP for Hooked {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            self.0.get_nlp_info()
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            self.0.get_bounds_info(b)
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            self.0.get_starting_point(sp)
        }
        fn eval_f(&mut self, x: &[Number], n: bool) -> Option<Number> {
            self.0.eval_f(x, n)
        }
        fn eval_grad_f(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            self.0.eval_grad_f(x, n, g)
        }
        fn eval_g(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            self.0.eval_g(x, n, g)
        }
        fn eval_jac_g(&mut self, x: Option<&[Number]>, n: bool, mode: SparsityRequest<'_>) -> bool {
            self.0.eval_jac_g(x, n, mode)
        }
        fn eval_h(
            &mut self,
            x: Option<&[Number]>,
            n: bool,
            s: Number,
            l: Option<&[Number]>,
            nl: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            self.0.eval_h(x, n, s, l, nl, mode)
        }
        fn finalize_solution(&mut self, s: Solution<'_>, d: &IpoptData, q: &IpoptCq) {
            self.0.finalize_solution(s, d, q)
        }
        fn get_constraints_linearity(&mut self, t: &mut [Linearity]) -> bool {
            t.fill(Linearity::Linear);
            true
        }
        fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
            *req.use_g_scaling = true;
            req.g_scaling.copy_from_slice(&[2.0, 3.0, 4.0]);
            true
        }
    }

    /// Row-indexed hooks are forwarded with the product rows translated
    /// (PR review of #1103: they fell back to the trait defaults): the inner
    /// rows keep what the model reports, the products are nonlinear and
    /// unscaled.
    #[test]
    fn row_indexed_hooks_are_forwarded_and_extended_to_the_products() {
        let mut w = MpccTnlp::new(
            Rc::new(RefCell::new(Hooked(Toy {
                style: IndexStyle::C,
            }))),
            pairs(),
            Lowering::Eq,
        )
        .unwrap();
        let mut lin = vec![Linearity::NonLinear; 5];
        assert!(w.get_constraints_linearity(&mut lin));
        assert_eq!(
            lin,
            vec![
                Linearity::Linear,
                Linearity::Linear,
                Linearity::Linear,
                Linearity::NonLinear,
                Linearity::NonLinear
            ]
        );
        let (mut obj, mut ux, mut ug) = (1.0, false, false);
        let mut xs = vec![1.0; 3];
        let mut gs = vec![0.0; 5];
        assert!(w.get_scaling_parameters(ScalingRequest {
            obj_scaling: &mut obj,
            use_x_scaling: &mut ux,
            x_scaling: &mut xs,
            use_g_scaling: &mut ug,
            g_scaling: &mut gs,
        }));
        assert!(ug);
        assert_eq!(gs, vec![2.0, 3.0, 4.0, 1.0, 1.0]);
    }

    struct Flipped(Toy);

    impl TNLP for Flipped {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            self.0.get_nlp_info()
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[-5.0, -1e19, -1.0]);
            b.x_u.copy_from_slice(&[5.0, 2.0, 1e19]);
            b.g_l.copy_from_slice(&[-1e19, -1e19, -10.0]);
            b.g_u.copy_from_slice(&[0.5, 3.0, 10.0]);
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            self.0.get_starting_point(sp)
        }
        fn eval_f(&mut self, x: &[Number], n: bool) -> Option<Number> {
            self.0.eval_f(x, n)
        }
        fn eval_grad_f(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            self.0.eval_grad_f(x, n, g)
        }
        fn eval_g(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            self.0.eval_g(x, n, g)
        }
        fn eval_jac_g(&mut self, x: Option<&[Number]>, n: bool, mode: SparsityRequest<'_>) -> bool {
            self.0.eval_jac_g(x, n, mode)
        }
        fn eval_h(
            &mut self,
            x: Option<&[Number]>,
            n: bool,
            s: Number,
            l: Option<&[Number]>,
            nl: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            self.0.eval_h(x, n, s, l, nl, mode)
        }
        fn finalize_solution(&mut self, s: Solution<'_>, d: &IpoptData, q: &IpoptCq) {
            self.0.finalize_solution(s, d, q)
        }
    }

    #[test]
    fn upper_and_mixed_orientations() {
        let mut w = MpccTnlp::new(
            Rc::new(RefCell::new(Flipped(Toy {
                style: IndexStyle::C,
            }))),
            pairs(),
            Lowering::Eq,
        )
        .unwrap();
        // G and H are the oriented slacks, each >= 0 on the feasible side.
        let gi = Toy::g(&X);
        let g = lowered_g(&mut w, &X);
        assert!((g[3] - (2.0 - X[1]) * (0.5 - gi[0])).abs() < 1e-15, "{g:?}");
        assert!((g[4] - (X[2] + 1.0) * (3.0 - gi[1])).abs() < 1e-15, "{g:?}");
        // Jacobian and Hessian against central differences.
        let j = dense_jac(&mut w, &X);
        let lam = [0.3, -1.2, 0.8, 2.5, -0.7];
        let s = 1.7;
        let hd = dense_hess(&mut w, &X, s, &lam);
        let grad_l = |w: &mut MpccTnlp, x: &[Number]| -> Vec<Number> {
            let mut gf = vec![0.0; 3];
            assert!(w.eval_grad_f(x, true, &mut gf));
            let j = dense_jac(w, x);
            (0..3)
                .map(|c| s * gf[c] + (0..5).map(|r| lam[r] * j[r][c]).sum::<Number>())
                .collect()
        };
        let h = 1e-6;
        for c in 0..3 {
            let (mut xp, mut xm) = (X, X);
            xp[c] += h;
            xm[c] -= h;
            let (gp, gm) = (lowered_g(&mut w, &xp), lowered_g(&mut w, &xm));
            for r in 0..5 {
                let fd = (gp[r] - gm[r]) / (2.0 * h);
                assert!(
                    (j[r][c] - fd).abs() < 1e-8,
                    "J[{r}][{c}] = {} vs fd {fd}",
                    j[r][c]
                );
            }
            let (lp, lm) = (grad_l(&mut w, &xp), grad_l(&mut w, &xm));
            for r in 0..3 {
                let fd = (lp[r] - lm[r]) / (2.0 * h);
                assert!(
                    (hd[r][c] - fd).abs() < 1e-7,
                    "H[{r}][{c}] = {} vs fd {fd}",
                    hd[r][c]
                );
            }
        }
    }

    #[test]
    fn refuses_what_it_does_not_lower() {
        let mk = |p: Vec<ComplementarityPair>| {
            MpccTnlp::new(
                Rc::new(RefCell::new(Toy {
                    style: IndexStyle::C,
                })),
                p,
                Lowering::Eq,
            )
        };
        // x0 is boxed and row 0 has only a lower bound: a one-sided pair at
        // x0's lower bound, its upper bound an ordinary limit -- lowered.
        let mut w = mk(vec![ComplementarityPair { row: 0, var: 0 }]).unwrap();
        let g = lowered_g(&mut w, &X);
        let gi = Toy::g(&X);
        assert!((g[3] - (X[0] + 5.0) * (gi[0] - 0.5)).abs() < 1e-15);
        // row 2 has a finite upper bound.
        let e = mk(vec![ComplementarityPair { row: 2, var: 1 }])
            .err()
            .unwrap();
        assert!(e.contains("one-sided"), "{e}");
        // variable used twice.
        let e = mk(vec![
            ComplementarityPair { row: 0, var: 1 },
            ComplementarityPair { row: 1, var: 1 },
        ])
        .err()
        .unwrap();
        assert!(e.contains("already in a pair"), "{e}");
        // out of range.
        let e = mk(vec![ComplementarityPair { row: 9, var: 1 }])
            .err()
            .unwrap();
        assert!(e.contains("out of range"), "{e}");
    }

    #[test]
    fn start_override_replaces_the_inner_start() {
        let mut w = wrap(IndexStyle::C, Lowering::Relaxed(0.1));
        w.set_start_override(Some(StartOverride {
            x: vec![1.0, 2.0, 3.0],
            z_l: vec![0.1; 3],
            z_u: vec![0.2; 3],
            lambda: vec![0.5, 0.6, 0.7, 0.8, 0.9],
        }));
        let (mut x, mut zl, mut zu, mut lam) =
            (vec![0.0; 3], vec![0.0; 3], vec![0.0; 3], vec![0.0; 5]);
        assert!(w.get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x,
            init_z: true,
            z_l: &mut zl,
            z_u: &mut zu,
            init_lambda: true,
            lambda: &mut lam,
        }));
        assert_eq!(
            (x, zl, zu, lam),
            (
                vec![1.0, 2.0, 3.0],
                vec![0.1; 3],
                vec![0.2; 3],
                vec![0.5, 0.6, 0.7, 0.8, 0.9]
            )
        );
        // A point of the wrong dimension is refused, not truncated.
        w.set_start_override(Some(StartOverride {
            x: vec![1.0],
            ..Default::default()
        }));
        let mut x = vec![0.0; 3];
        assert!(!w.get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x,
            init_z: false,
            z_l: &mut [],
            z_u: &mut [],
            init_lambda: false,
            lambda: &mut [],
        }));
    }

    #[test]
    fn starting_point_zeroes_product_multipliers() {
        let mut w = wrap(IndexStyle::C, Lowering::Eq);
        let (mut x, mut zl, mut zu, mut lam) =
            (vec![0.0; 3], vec![0.0; 3], vec![0.0; 3], vec![9.0; 5]);
        assert!(w.get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x,
            init_z: false,
            z_l: &mut zl,
            z_u: &mut zu,
            init_lambda: true,
            lambda: &mut lam,
        }));
        assert_eq!(lam, vec![7.0, 7.0, 7.0, 0.0, 0.0]);
        assert_eq!(x, vec![0.3, 0.4, 0.5]);
    }
}
