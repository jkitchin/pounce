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
//! 0 <= G(x) := x_j - x_L[j]   ⟂   H(x) := g_i(x) - g_L[i] >= 0
//! ```
//!
//! which is AMPL's `g_i(x) complements x_j` with `x_j` bounded below only —
//! the shape `.nl` complementarity rows (kind 5, one finite variable bound)
//! encode — and covers the corpus in `benchmarks/calipso/`. The inner model
//! must state those bounds itself (`x_L[j]` and `g_L[i]` finite, `x_U[j]`
//! and `g_U[i]` infinite); [`MpccTnlp::new`] refuses anything else with the
//! reason, rather than guess at a two-sided (mixed) complementarity it does
//! not lower. A pair between two variables is a row `g_i(x) = x_k` away.
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
//! `∇p = H e_j + G ∇g_i`. The Lagrangian Hessian of the lowered model is the
//! inner one evaluated with the row-`i` multiplier shifted by `μ_k G`, plus
//! the cross terms `μ_k (e_j ∇g_iᵀ + ∇g_i e_jᵀ)` (`2 μ_k ∂g_i/∂x_j` on the
//! diagonal when `x_j` appears in `g_i`). Cross entries the inner Hessian
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
use crate::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, MetaData, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
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

/// Bounds at or beyond this magnitude are infinite, matching the
/// `nlp_lower_bound_inf` / `nlp_upper_bound_inf` defaults (`±1e19`).
const INF: Number = 1e19;

/// The lowered model; see the module docs.
pub struct MpccTnlp {
    inner: Rc<RefCell<dyn TNLP>>,
    pairs: Vec<ComplementarityPair>,
    lowering: Lowering,
    n: usize,
    m_inner: usize,
    index_style: IndexStyle,
    inner_jac_nnz: usize,
    inner_h_nnz: usize,
    /// `x_L[j]` and `g_L[i]` per pair.
    g_offset: Vec<Number>,
    h_offset: Vec<Number>,
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
        let mut g_offset = Vec::with_capacity(pairs.len());
        let mut h_offset = Vec::with_capacity(pairs.len());
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
            if !(xl > -INF && xu >= INF && gl > -INF && gu >= INF) {
                return Err(format!(
                    "pair {k}: only one-sided pairs `0 <= x_j - x_L ⟂ g_i(x) - g_L >= 0` are \
                     lowered (finite x_L and g_L, infinite x_U and g_U); variable {} has bounds \
                     [{xl:e}, {xu:e}] and row {} has [{gl:e}, {gu:e}]",
                    p.var, p.row
                ));
            }
            g_offset.push(xl);
            h_offset.push(gl);
        }

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
            g_offset,
            h_offset,
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
            pairs,
        })
    }

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

    /// When set, `finalize_solution` records the result for the driver but
    /// does not forward it to the inner model, so a driver running several
    /// lowered solves forwards only the one it reports.
    pub fn set_defer_inner_finalize(&mut self, defer: bool) {
        self.defer_inner_finalize = defer;
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

    /// `G_k · H_k` at the last finalized `x`.
    pub fn last_products(&self) -> &[Number] {
        &self.last_products
    }

    /// `(G_k(x), H_k(x))` for every pair at `x`.
    pub fn pair_values(&mut self, x: &[Number]) -> Option<Vec<(Number, Number)>> {
        if !self.inner.borrow_mut().eval_g(x, true, &mut self.g_inner) {
            return None;
        }
        Some(self.pair_values_from(x))
    }

    fn pair_values_from(&self, x: &[Number]) -> Vec<(Number, Number)> {
        self.pairs
            .iter()
            .enumerate()
            .map(|(k, p)| {
                (
                    x[p.var] - self.g_offset[k],
                    self.g_inner[p.row] - self.h_offset[k],
                )
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
            m: (self.m_inner + self.pairs.len()) as Index,
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
        for k in 0..self.pairs.len() {
            b.g_l[m + k] = lo;
            b.g_u[m + k] = hi;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
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
                for (k, p) in self.pairs.iter().enumerate() {
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
                    let b = self.jac_block[k];
                    let ents = &self.row_entries[k];
                    // G ∇g_i on row i's columns ...
                    for (s, &(pos, _)) in ents.iter().enumerate() {
                        values[b + s] = gk * values[pos];
                    }
                    // ... plus H on column j, in place or appended.
                    let slot = self.var_slot[k];
                    if slot == ents.len() {
                        values[b + slot] = hk;
                    } else {
                        values[b + slot] += hk;
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
                for (k, p) in self.pairs.iter().enumerate() {
                    self.lam_inner[p.row] += lambda[m + k] * pv[k].0;
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
                            new_lambda || !self.pairs.is_empty(),
                            SparsityRequest::Values {
                                values: &mut values[..hnnz],
                            },
                        )
                    {
                        return false;
                    }
                    if !self.pairs.is_empty()
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
                for (k, p) in self.pairs.iter().enumerate() {
                    let mu = lambda[m + k];
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
        let m_total = self.m_inner + self.pairs.len();
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
                v.extend((0..self.pairs.len()).map(|k| format!("complementarity[{k}]")));
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
        // x0 has a finite upper bound (two-sided).
        let e = mk(vec![ComplementarityPair { row: 0, var: 0 }])
            .err()
            .unwrap();
        assert!(e.contains("one-sided"), "{e}");
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
