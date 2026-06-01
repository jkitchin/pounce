//! McCormick polyhedral relaxation of a factorable problem over a box.
//!
//! Factorable programming: every tape slot becomes an auxiliary LP variable
//! `w_k`, constrained to the convex/concave envelopes of the operation that
//! produced it given **interval** bounds (from FBBT's forward pass) on its
//! operands. Affine ops are exact equalities; bilinear products get the four
//! McCormick inequalities (the exact convex hull of a single bilinear term);
//! and every univariate atom (`x^n`, `√`, `exp`, `ln`, `sin`, `cos`, `|·|`) is
//! relaxed by the tight polyhedral [`crate::envelope`] — secant + tangent cuts
//! for convex/concave arcs, and the tangent-from-endpoint construction for
//! single-inflection arcs (odd powers across 0, trig over a sub-`π` box). The
//! result is a **linear program** whose optimum is a valid lower bound on the
//! true minimum over the box, exact in the zero-width-box limit (so spatial
//! branch-and-bound converges).
//!
//! The few remaining hard cases — `sin`/`cos` over a box wider than `π`,
//! division by an interval straddling zero, and `Opaque` — fall back to the
//! interval box bound on `w_k`: valid, just weak, which branching then sharpens
//! (and, for trig, narrows below `π` so the envelope engages).

use crate::envelope::{self, Envelope};
use crate::problem::GlobalProblem;
use pounce_convex::{QpProblem, Triplet};
use pounce_nlp::{FbbtOp, FbbtTape};
use pounce_presolve::fbbt::forward_pass;
use std::collections::BTreeMap;

/// Sentinel magnitude past which the convex solver treats a bound as infinite.
const INF: f64 = 1e20;

/// A tape slot's representation in the LP: either an LP column or a constant.
#[derive(Clone, Copy)]
enum Handle {
    Col(usize),
    Const(f64),
}

/// A univariate atom `w = kind(a)` (both LP columns), recorded so the
/// cutting-plane ("sandwich") refinement can add tangent cuts at the LP point.
#[derive(Clone, Copy)]
pub(crate) struct Atom {
    pub w: usize,
    pub a: usize,
    pub kind: AtomKind,
}

/// The univariate operations the relaxation knows how to convexify.
#[derive(Clone, Copy)]
pub(crate) enum AtomKind {
    Pow(u32),
    Exp,
    Ln,
    Sqrt,
    Sin,
    Cos,
}

impl AtomKind {
    pub(crate) fn f(self, t: f64) -> f64 {
        match self {
            AtomKind::Pow(n) => t.powi(n as i32),
            AtomKind::Exp => t.exp(),
            AtomKind::Ln => t.max(1e-12).ln(),
            AtomKind::Sqrt => t.max(0.0).sqrt(),
            AtomKind::Sin => t.sin(),
            AtomKind::Cos => t.cos(),
        }
    }

    fn df(self, t: f64) -> f64 {
        match self {
            AtomKind::Pow(n) => n as f64 * t.powi(n as i32 - 1),
            AtomKind::Exp => t.exp(),
            AtomKind::Ln => 1.0 / t.max(1e-12),
            AtomKind::Sqrt => 0.5 / t.max(1e-300).sqrt(),
            AtomKind::Sin => t.cos(),
            AtomKind::Cos => -t.sin(),
        }
    }

    /// Curvature over `[l, u]`: `Some(true)` convex, `Some(false)` concave,
    /// `None` mixed (a tangent at an interior point would not be globally valid,
    /// so the sandwich step skips it).
    fn curvature(self, l: f64, u: f64) -> Option<bool> {
        match self {
            AtomKind::Pow(n) if n % 2 == 0 => Some(true),
            AtomKind::Pow(_) if l >= 0.0 => Some(true),
            AtomKind::Pow(_) if u <= 0.0 => Some(false),
            AtomKind::Pow(_) => None, // odd power straddling 0
            AtomKind::Exp => Some(true),
            AtomKind::Ln | AtomKind::Sqrt => Some(false),
            AtomKind::Sin | AtomKind::Cos => {
                if u - l > std::f64::consts::PI || self.f(l) * self.f(u) < 0.0 {
                    None // an interior inflection (zero of sin/cos) may exist
                } else if self.f(0.5 * (l + u)) < 0.0 {
                    Some(true)
                } else {
                    Some(false)
                }
            }
        }
    }
}

/// A nonconvex term, tagged with the original variables it depends on, used to
/// score branching candidates by relaxation violation. `w` is the term's LP
/// column; the violation is `|x_w − (true value)|` at the LP solution.
pub(crate) enum BranchTerm {
    Unary {
        w: usize,
        a: usize,
        kind: AtomKind,
        vars: Vec<usize>,
    },
    Bilinear {
        w: usize,
        u: usize,
        v: usize,
        vars: Vec<usize>,
    },
    Ratio {
        w: usize,
        a: usize,
        c: usize,
        vars: Vec<usize>,
    },
}

impl BranchTerm {
    fn vars(&self) -> &[usize] {
        match self {
            BranchTerm::Unary { vars, .. }
            | BranchTerm::Bilinear { vars, .. }
            | BranchTerm::Ratio { vars, .. } => vars,
        }
    }

    /// Relaxation violation `|x_w − (true value of the term)|` at point `x`.
    fn violation(&self, x: &[f64]) -> f64 {
        let v = match *self {
            BranchTerm::Unary { w, a, kind, .. } => x[w] - kind.f(x[a]),
            BranchTerm::Bilinear { w, u, v, .. } => x[w] - x[u] * x[v],
            BranchTerm::Ratio { w, a, c, .. } => {
                if x[c].abs() < 1e-12 {
                    return 0.0;
                }
                x[w] - x[a] / x[c]
            }
        };
        if v.is_finite() {
            v.abs()
        } else {
            0.0
        }
    }
}

/// Per-variable relaxation violation: each nonconvex term credits its violation
/// to every original variable it depends on. The largest score is the
/// most-violation branching variable.
pub(crate) fn branch_scores(terms: &[BranchTerm], x: &[f64], n: usize) -> Vec<f64> {
    let mut s = vec![0.0; n];
    for t in terms {
        let viol = t.violation(x);
        if viol > 0.0 {
            for &i in t.vars() {
                if i < n {
                    s[i] += viol;
                }
            }
        }
    }
    s
}

/// The relaxation LP plus the bookkeeping to read a solution back. The first
/// `n_vars` LP columns are the original problem variables.
pub(crate) struct Relaxation {
    pub qp: QpProblem,
    /// Univariate atoms for cutting-plane refinement.
    pub atoms: Vec<Atom>,
    /// Nonconvex terms for most-violation branching.
    pub branch_terms: Vec<BranchTerm>,
    /// LP column holding the objective value (for an incumbent cutoff row in
    /// OBBT). `None` for a constant/empty objective.
    pub obj_col: Option<usize>,
    /// `true` if a constant constraint was found out of bounds — the box is
    /// then certifiably infeasible and the node can be pruned without solving.
    pub trivially_infeasible: bool,
}

/// Accumulates LP columns and rows while walking the tapes.
struct Builder {
    col_lo: Vec<f64>,
    col_hi: Vec<f64>,
    eq: Vec<Triplet>,
    eq_rhs: Vec<f64>,
    ineq: Vec<Triplet>,
    ineq_rhs: Vec<f64>,
    atoms: Vec<Atom>,
    branch_terms: Vec<BranchTerm>,
    infeasible: bool,
}

fn clamp_inf(v: f64) -> f64 {
    v.clamp(-INF, INF)
}

impl Builder {
    fn new(x_lo: &[f64], x_hi: &[f64]) -> Self {
        Builder {
            col_lo: x_lo.iter().map(|&v| clamp_inf(v)).collect(),
            col_hi: x_hi.iter().map(|&v| clamp_inf(v)).collect(),
            eq: Vec::new(),
            eq_rhs: Vec::new(),
            ineq: Vec::new(),
            ineq_rhs: Vec::new(),
            atoms: Vec::new(),
            branch_terms: Vec::new(),
            infeasible: false,
        }
    }

    fn add_col(&mut self, lo: f64, hi: f64) -> usize {
        let c = self.col_lo.len();
        self.col_lo.push(clamp_inf(lo));
        self.col_hi.push(clamp_inf(hi));
        c
    }

    /// Push `Σ coeff·handle  (=|≤) rhs`, folding constants into the RHS and
    /// summing duplicate column coefficients so each row has unique columns.
    fn row(&self, terms: &[(Handle, f64)], rhs: f64) -> (BTreeMap<usize, f64>, f64) {
        let mut cols: BTreeMap<usize, f64> = BTreeMap::new();
        let mut r = rhs;
        for &(h, coeff) in terms {
            match h {
                Handle::Col(c) => *cols.entry(c).or_insert(0.0) += coeff,
                Handle::Const(v) => r -= coeff * v,
            }
        }
        (cols, r)
    }

    fn emit_eq(&mut self, terms: &[(Handle, f64)], rhs: f64) {
        let (cols, r) = self.row(terms, rhs);
        let row = self.eq_rhs.len();
        for (c, v) in cols {
            if v != 0.0 {
                self.eq.push(Triplet::new(row, c, v));
            }
        }
        self.eq_rhs.push(r);
    }

    fn emit_le(&mut self, terms: &[(Handle, f64)], rhs: f64) {
        let (cols, r) = self.row(terms, rhs);
        if cols.is_empty() {
            // Pure constant inequality: 0 ≤ r. If violated the box is infeasible.
            if r < -1e-9 {
                self.infeasible = true;
            }
            return;
        }
        let row = self.ineq_rhs.len();
        for (c, v) in cols {
            if v != 0.0 {
                self.ineq.push(Triplet::new(row, c, v));
            }
        }
        self.ineq_rhs.push(r);
    }

    /// Four McCormick inequalities for `p = u·v` over `[uL,uU]×[vL,vU]`.
    #[allow(clippy::too_many_arguments)]
    fn bilinear(&mut self, p: Handle, u: Handle, v: Handle, ul: f64, uu: f64, vl: f64, vu: f64) {
        // Skip if any factor range is non-finite (envelope coefficients blow up);
        // the box bound on `p` then carries whatever interval info exists.
        if ![ul, uu, vl, vu].iter().all(|x| x.is_finite()) {
            return;
        }
        // p ≥ uL·v + vL·u − uL·vL
        self.emit_le(&[(p, -1.0), (v, ul), (u, vl)], ul * vl);
        // p ≥ uU·v + vU·u − uU·vU
        self.emit_le(&[(p, -1.0), (v, uu), (u, vu)], uu * vu);
        // p ≤ uU·v + vL·u − uU·vL
        self.emit_le(&[(p, 1.0), (v, -uu), (u, -vl)], -uu * vl);
        // p ≤ uL·v + vU·u − uL·vU
        self.emit_le(&[(p, 1.0), (v, -ul), (u, -vu)], -ul * vu);
    }

    /// Strengthen a trilinear product `w = f₀·f₁·f₂` (the standard grouping
    /// `(f₀f₁)·f₂` is added by the caller). For the other two groupings, create
    /// the pairwise product column and McCormick-relate both it and `w`;
    /// intersecting all three groupings is tighter than any one alone.
    fn trilinear(&mut self, w: usize, f: [(usize, f64, f64); 3]) {
        for &(i, j, k) in &[(0usize, 2usize, 1usize), (1, 2, 0)] {
            let (ci, li, ui) = f[i];
            let (cj, lj, uj) = f[j];
            let cands = [li * lj, li * uj, ui * lj, ui * uj];
            let pl = cands.iter().copied().fold(f64::INFINITY, f64::min);
            let pu = cands.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !pl.is_finite() || !pu.is_finite() {
                continue;
            }
            let p = self.add_col(pl, pu);
            self.bilinear(
                Handle::Col(p),
                Handle::Col(ci),
                Handle::Col(cj),
                li,
                ui,
                lj,
                uj,
            );
            let (ck, lk, uk) = f[k];
            self.bilinear(
                Handle::Col(w),
                Handle::Col(p),
                Handle::Col(ck),
                pl,
                pu,
                lk,
                uk,
            );
        }
    }

    /// Relax a univariate atom `w = f(a)` over `[al, au]` using the tight
    /// polyhedral [`Envelope`] from [`crate::envelope`]. `f` evaluates the atom
    /// (used to pin a constant/degenerate operand); `build` lazily produces the
    /// envelope when the operand is a genuine variable with a bounded range,
    /// returning `None` to decline (e.g. trig over a too-wide interval — the
    /// column's interval box bound then carries the relaxation).
    fn emit_univariate(
        &mut self,
        w: usize,
        a: Handle,
        al: f64,
        au: f64,
        kind: AtomKind,
        build: impl FnOnce() -> Option<Envelope>,
    ) {
        let wh = Handle::Col(w);
        if let Handle::Const(v) = a {
            self.emit_eq(&[(wh, 1.0)], kind.f(v));
            return;
        }
        // Record the atom (operand is a genuine column) for sandwich refinement.
        if let Handle::Col(ac) = a {
            self.atoms.push(Atom { w, a: ac, kind });
        }
        if !al.is_finite() || !au.is_finite() || au - al < 1e-12 {
            if al.is_finite() && au.is_finite() {
                self.emit_eq(&[(wh, 1.0)], kind.f(0.5 * (al + au)));
            }
            return; // unbounded domain: rely on the column's box bound
        }
        let Some(env) = build() else {
            return; // declined: box bound only
        };
        for c in &env.under {
            // w ≥ slope·a + intercept  ⇔  slope·a − w ≤ −intercept
            self.emit_le(&[(wh, -1.0), (a, c.slope)], -c.intercept);
        }
        for c in &env.over {
            // w ≤ slope·a + intercept  ⇔  w − slope·a ≤ intercept
            self.emit_le(&[(wh, 1.0), (a, -c.slope)], c.intercept);
        }
    }
}

/// Sorted union of two ascending, deduplicated index lists.
fn union(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// For each tape slot, the set of original variables (ascending) it depends on.
fn slot_support(tape: &FbbtTape) -> Vec<Vec<usize>> {
    let mut sup: Vec<Vec<usize>> = Vec::with_capacity(tape.ops.len());
    for op in &tape.ops {
        let s = match *op {
            FbbtOp::Const(_) | FbbtOp::Opaque => Vec::new(),
            FbbtOp::Var(i) => vec![i],
            FbbtOp::Add(a, b) | FbbtOp::Sub(a, b) | FbbtOp::Mul(a, b) | FbbtOp::Div(a, b) => {
                union(&sup[a], &sup[b])
            }
            FbbtOp::Neg(a)
            | FbbtOp::Sqrt(a)
            | FbbtOp::Exp(a)
            | FbbtOp::Ln(a)
            | FbbtOp::Abs(a)
            | FbbtOp::Sin(a)
            | FbbtOp::Cos(a)
            | FbbtOp::PowInt(a, _) => sup[a].clone(),
        };
        sup.push(s);
    }
    sup
}

/// Detect a 3-way product `Mul(a, c)` where one operand is itself a `Mul` of
/// two columns and the other operand is a column — returning the three flat
/// factors as `(column, lo, hi)`. `None` if it is an ordinary bilinear product
/// (or any factor is a constant).
fn trilinear_factors(
    handle: &[Handle],
    ivals: &[pounce_presolve::fbbt::Interval],
    ops: &[FbbtOp],
    a: usize,
    c: usize,
) -> Option<[(usize, f64, f64); 3]> {
    let col = |s: usize| match handle[s] {
        Handle::Col(ci) => Some((ci, ivals[s].lo, ivals[s].hi)),
        Handle::Const(_) => None,
    };
    // Inner Mul on the left: (a1·a2)·c.
    if let FbbtOp::Mul(a1, a2) = ops[a] {
        if let (Some(f0), Some(f1), Some(f2)) = (col(a1), col(a2), col(c)) {
            return Some([f0, f1, f2]);
        }
    }
    // Inner Mul on the right: a·(c1·c2).
    if let FbbtOp::Mul(c1, c2) = ops[c] {
        if let (Some(f0), Some(f1), Some(f2)) = (col(a), col(c1), col(c2)) {
            return Some([f0, f1, f2]);
        }
    }
    None
}

/// Process one tape, appending its relaxation to `b`; return the root slot's
/// handle (the LP representation of the whole expression's value).
fn process_tape(
    b: &mut Builder,
    tape: &FbbtTape,
    x_lo: &[f64],
    x_hi: &[f64],
    multilinear: bool,
) -> Option<Handle> {
    if tape.is_empty() {
        return None;
    }
    let ivals = forward_pass(tape, x_lo, x_hi).ok()?;
    let mut handle: Vec<Handle> = Vec::with_capacity(tape.ops.len());

    // A fresh aux column for slot `k`, bounded by its interval.
    macro_rules! new_col {
        ($k:expr) => {{
            let iv = ivals[$k];
            b.add_col(iv.lo, iv.hi)
        }};
    }
    let bounds = |h: Handle, k: usize| -> (f64, f64) {
        match h {
            Handle::Const(v) => (v, v),
            Handle::Col(_) => (ivals[k].lo, ivals[k].hi),
        }
    };

    for (k, op) in tape.ops.iter().enumerate() {
        let h = match *op {
            FbbtOp::Const(c) => Handle::Const(c),
            FbbtOp::Var(i) => Handle::Col(i), // shared original-variable column
            FbbtOp::Add(a, c) => {
                let col = new_col!(k);
                let w = Handle::Col(col);
                b.emit_eq(&[(w, 1.0), (handle[a], -1.0), (handle[c], -1.0)], 0.0);
                w
            }
            FbbtOp::Sub(a, c) => {
                let col = new_col!(k);
                let w = Handle::Col(col);
                b.emit_eq(&[(w, 1.0), (handle[a], -1.0), (handle[c], 1.0)], 0.0);
                w
            }
            FbbtOp::Neg(a) => {
                let col = new_col!(k);
                let w = Handle::Col(col);
                b.emit_eq(&[(w, 1.0), (handle[a], 1.0)], 0.0);
                w
            }
            FbbtOp::Mul(a, c) => {
                let col = new_col!(k);
                let w = Handle::Col(col);
                let (ha, hc) = (handle[a], handle[c]);
                match (ha, hc) {
                    (Handle::Const(va), _) => b.emit_eq(&[(w, 1.0), (hc, -va)], 0.0),
                    (_, Handle::Const(vc)) => b.emit_eq(&[(w, 1.0), (ha, -vc)], 0.0),
                    _ => {
                        let (al, au) = bounds(ha, a);
                        let (cl, cu) = bounds(hc, c);
                        b.bilinear(w, ha, hc, al, au, cl, cu); // grouping (a·b)·c
                                                               // Tighten a 3-way product x·y·z by intersecting the other
                                                               // two groupings (recursive bilinear alone is loose).
                        if multilinear {
                            if let Some(f) = trilinear_factors(&handle, &ivals, &tape.ops, a, c) {
                                b.trilinear(col, f);
                            }
                        }
                    }
                }
                w
            }
            FbbtOp::Div(a, c) => {
                let col = new_col!(k);
                let w = Handle::Col(col);
                let (ha, hc) = (handle[a], handle[c]);
                if let Handle::Const(vc) = hc {
                    if vc != 0.0 {
                        b.emit_eq(&[(w, 1.0), (ha, -1.0 / vc)], 0.0);
                    }
                } else {
                    let (cl, cu) = bounds(hc, c);
                    let (wl, wu) = (ivals[k].lo, ivals[k].hi);
                    // a = w·c, relaxed by McCormick (only sound when c avoids 0).
                    if cl > 1e-12 || cu < -1e-12 {
                        b.bilinear(ha, w, hc, wl, wu, cl, cu);
                    }
                }
                w
            }
            FbbtOp::PowInt(a, n) => {
                let col = new_col!(k);
                let w = Handle::Col(col);
                let ha = handle[a];
                let (al, au) = bounds(ha, a);
                match n {
                    0 => b.emit_eq(&[(w, 1.0)], 1.0),
                    1 => b.emit_eq(&[(w, 1.0), (ha, -1.0)], 0.0),
                    // n ≥ 2 — convex (even / nonneg), concave (nonpos), or the
                    // single-inflection envelope when odd and straddling 0.
                    _ => b.emit_univariate(col, ha, al, au, AtomKind::Pow(n), || {
                        Some(envelope::power(n, al, au))
                    }),
                }
                w
            }
            FbbtOp::Sqrt(a) => {
                let col = new_col!(k);
                let ha = handle[a];
                let (al, au) = bounds(ha, a);
                b.emit_univariate(col, ha, al, au, AtomKind::Sqrt, || {
                    Some(envelope::sqrt(al, au))
                });
                Handle::Col(col)
            }
            FbbtOp::Exp(a) => {
                let col = new_col!(k);
                let ha = handle[a];
                let (al, au) = bounds(ha, a);
                b.emit_univariate(col, ha, al, au, AtomKind::Exp, || {
                    Some(envelope::exp(al, au))
                });
                Handle::Col(col)
            }
            FbbtOp::Ln(a) => {
                let col = new_col!(k);
                let ha = handle[a];
                let (al, au) = bounds(ha, a);
                b.emit_univariate(col, ha, al, au, AtomKind::Ln, || Some(envelope::ln(al, au)));
                Handle::Col(col)
            }
            FbbtOp::Sin(a) => {
                let col = new_col!(k);
                let ha = handle[a];
                let (al, au) = bounds(ha, a);
                b.emit_univariate(col, ha, al, au, AtomKind::Sin, || {
                    envelope::trig(true, al, au)
                });
                Handle::Col(col)
            }
            FbbtOp::Cos(a) => {
                let col = new_col!(k);
                let ha = handle[a];
                let (al, au) = bounds(ha, a);
                b.emit_univariate(col, ha, al, au, AtomKind::Cos, || {
                    envelope::trig(false, al, au)
                });
                Handle::Col(col)
            }
            FbbtOp::Abs(a) => {
                let col = new_col!(k);
                let w = Handle::Col(col);
                let ha = handle[a];
                let (al, au) = bounds(ha, a);
                // |·| is convex: w ≥ a, w ≥ −a, secant overestimator.
                b.emit_le(&[(w, -1.0), (ha, 1.0)], 0.0);
                b.emit_le(&[(w, -1.0), (ha, -1.0)], 0.0);
                if al.is_finite() && au.is_finite() && au - al > 1e-12 {
                    let s = (au.abs() - al.abs()) / (au - al);
                    b.emit_le(&[(w, 1.0), (ha, -s)], al.abs() - s * al);
                }
                w
            }
            // Unsupported: interval box bound on the column only.
            FbbtOp::Opaque => Handle::Col(new_col!(k)),
        };
        handle.push(h);
    }

    // Record nonconvex terms with the original variables they depend on, for
    // most-violation branching. A separate pass over the tape keeps this
    // independent of the relaxation arms above.
    let sup = slot_support(tape);
    let col_of = |s: usize| match handle[s] {
        Handle::Col(c) => Some(c),
        Handle::Const(_) => None,
    };
    for (k, op) in tape.ops.iter().enumerate() {
        let Handle::Col(w) = handle[k] else { continue };
        let unary = |kind: AtomKind, a: usize| {
            col_of(a).map(|ac| BranchTerm::Unary {
                w,
                a: ac,
                kind,
                vars: sup[a].clone(),
            })
        };
        let term = match *op {
            FbbtOp::PowInt(a, n) if n >= 2 => unary(AtomKind::Pow(n), a),
            FbbtOp::Exp(a) => unary(AtomKind::Exp, a),
            FbbtOp::Ln(a) => unary(AtomKind::Ln, a),
            FbbtOp::Sqrt(a) => unary(AtomKind::Sqrt, a),
            FbbtOp::Sin(a) => unary(AtomKind::Sin, a),
            FbbtOp::Cos(a) => unary(AtomKind::Cos, a),
            FbbtOp::Mul(a, c) => match (col_of(a), col_of(c)) {
                (Some(u), Some(v)) => Some(BranchTerm::Bilinear {
                    w,
                    u,
                    v,
                    vars: union(&sup[a], &sup[c]),
                }),
                _ => None,
            },
            FbbtOp::Div(a, c) => match (col_of(a), col_of(c)) {
                (Some(u), Some(v)) => Some(BranchTerm::Ratio {
                    w,
                    a: u,
                    c: v,
                    vars: union(&sup[a], &sup[c]),
                }),
                _ => None,
            },
            _ => None,
        };
        if let Some(t) = term {
            b.branch_terms.push(t);
        }
    }

    handle.last().copied()
}

/// Build the relaxation LP for `prob` over the box `[x_lo, x_hi]`. `multilinear`
/// enables the tighter multi-grouping relaxation of 3-way products.
pub(crate) fn build_relaxation(
    prob: &GlobalProblem,
    x_lo: &[f64],
    x_hi: &[f64],
    multilinear: bool,
) -> Relaxation {
    let mut b = Builder::new(x_lo, x_hi);

    // Objective → LP cost on its root handle.
    let obj_handle = process_tape(&mut b, &prob.objective, x_lo, x_hi, multilinear);

    // Constraints: bracket each root handle by [lo, hi].
    for con in &prob.constraints {
        match process_tape(&mut b, &con.tape, x_lo, x_hi, multilinear) {
            Some(Handle::Col(c)) => {
                let h = Handle::Col(c);
                if con.hi < INF {
                    b.emit_le(&[(h, 1.0)], con.hi); //  g ≤ hi
                }
                if con.lo > -INF {
                    b.emit_le(&[(h, -1.0)], -con.lo); // −g ≤ −lo
                }
            }
            Some(Handle::Const(v)) => {
                if v > con.hi + 1e-9 || v < con.lo - 1e-9 {
                    b.infeasible = true;
                }
            }
            None => {}
        }
    }

    let n_cols = b.col_lo.len();
    let mut c = vec![0.0; n_cols];
    // A constant or empty objective leaves the cost vector zero (the bound is
    // then just the constant / zero, refined by branching).
    let obj_col = match obj_handle {
        Some(Handle::Col(col)) => {
            c[col] = 1.0;
            Some(col)
        }
        _ => None,
    };

    let qp = QpProblem {
        n: n_cols,
        p_lower: Vec::new(),
        c,
        a: b.eq,
        b: b.eq_rhs,
        g: b.ineq,
        h: b.ineq_rhs,
        lb: b.col_lo,
        ub: b.col_hi,
    };
    Relaxation {
        qp,
        atoms: b.atoms,
        branch_terms: b.branch_terms,
        obj_col,
        trivially_infeasible: b.infeasible,
    }
}

/// Valid tangent ("sandwich") cuts at the LP point `x` for atoms whose
/// relaxation value is loose. For a convex atom the tangent at the operand's
/// current value is a global underestimator (and a global overestimator for a
/// concave one); adding it where the LP slack `w` sits on the wrong side of the
/// true atom value tightens the bound without branching. Returned as
/// `(row terms, rhs)` in `Σ coeff·col ≤ rhs` form.
pub(crate) fn sandwich_cuts(
    atoms: &[Atom],
    lb: &[f64],
    ub: &[f64],
    x: &[f64],
    tol: f64,
) -> Vec<(Vec<(usize, f64)>, f64)> {
    let mut cuts = Vec::new();
    for atom in atoms {
        let Some(convex) = atom.kind.curvature(lb[atom.a], ub[atom.a]) else {
            continue;
        };
        let t = x[atom.a];
        if !t.is_finite() {
            continue;
        }
        let slope = atom.kind.df(t);
        let ft = atom.kind.f(t);
        if !slope.is_finite() || !ft.is_finite() {
            continue;
        }
        let intercept = ft - slope * t;
        let w = x[atom.w];
        if convex && w < ft - tol {
            // w ≥ slope·a + intercept  ⇔  slope·a − w ≤ −intercept
            cuts.push((vec![(atom.w, -1.0), (atom.a, slope)], -intercept));
        } else if !convex && w > ft + tol {
            // w ≤ slope·a + intercept  ⇔  w − slope·a ≤ intercept
            cuts.push((vec![(atom.w, 1.0), (atom.a, -slope)], intercept));
        }
    }
    cuts
}

/// Append `≤` cuts (from [`sandwich_cuts`]) to the LP as new inequality rows.
pub(crate) fn append_cuts(qp: &mut QpProblem, cuts: &[(Vec<(usize, f64)>, f64)]) {
    for (terms, rhs) in cuts {
        let row = qp.h.len();
        for &(c, v) in terms {
            qp.g.push(Triplet::new(row, c, v));
        }
        qp.h.push(*rhs);
    }
}
