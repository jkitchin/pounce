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
//! | `Coupled { .. }` | `G·H <= τ(μ)`: the relaxation tightened with the barrier parameter inside one solve |
//!
//! Under `Coupled` the row is written `G·H + (τ₀ - τ) <= τ₀`: the bound
//! stays put and the shift carries `τ`, so a tighter `τ` raises the row's
//! residual and leaves every slack interior. [`TNLP::update_with_barrier`]
//! moves `τ` after each `μ` update to `max(floor, factor·μ)`, never up and
//! never by more than a factor `max_cut` at once.
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
    /// `G·H <= τ` with `τ` coupled to the barrier parameter: it starts at
    /// `tau0` and follows `max(floor, factor·μ)` down, by at most a factor
    /// `max_cut` per `μ` update. See the module docs.
    Coupled {
        tau0: Number,
        floor: Number,
        factor: Number,
        max_cut: Number,
    },
}

/// Whether a pair's row is lifted into a variable; see the module docs
/// ("Lifting").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lift {
    /// Never: every product row is `G(x)·H(x)`.
    No,
    /// A one-sided pair whose row has at least [`LIFT_AUTO_MIN_ENTRIES`]
    /// Jacobian entries.
    Auto,
    /// Every one-sided pair.
    Yes,
}

/// The row width from which [`Lift::Auto`] lifts a pair's row. A product
/// row on a row of width `w` costs `w + 1` Jacobian and `w` Hessian entries
/// unlifted, against one variable, three Jacobian entries and one Hessian
/// entry lifted, so a narrow row gains nothing worth a change of path.
pub const LIFT_AUTO_MIN_ENTRIES: usize = 8;

/// A primal-dual point for the next solve to start from, in the inner
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
    /// The current `τ` under [`Lowering::Coupled`] (its `tau0` until the
    /// first barrier update); unused otherwise.
    coupled_tau: Number,
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
    /// The lowered model's structure as lifting leaves it: the active one.
    layout: Layout,
    /// The other one, when lifting changed anything: unlifted if `layout`
    /// lifts, absent otherwise. [`MpccTnlp::set_lift`] swaps them.
    other_layout: Option<Layout>,
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
    /// Per inner row: the bounds `branch` gives it, or `None`. Built in
    /// [`MpccTnlp::set_branch`], so a lookup by row is O(1).
    branch_rows: Vec<Option<(Number, Number)>>,
}

/// Everything in the lowered model's structure that lifting changes. Built
/// once per setting in [`MpccTnlp::with_lift`]; the rest of the wrapper is
/// the same either way.
#[derive(Debug, Clone)]
struct Layout {
    /// Per lowered pair: the 0-based index of its lifted variable `w_k`
    /// (`n + l`), or `None` when its product row is `G(x)·H(x)`.
    lift_var: Vec<Option<usize>>,
    /// The lifted pairs, `(lowered index, variable index)`, in the order of
    /// their variables.
    lifted: Vec<(usize, usize)>,
    /// Per inner row: its lifted pair `(lowered index, variable index)`, or
    /// `None`. Rows are unique to a pair, so there is at most one.
    lift_of_row: Vec<Option<(usize, usize)>>,
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
}

/// The lowered structure for one lifting setting (module docs,
/// "Lifting"): which pairs are lifted, the lowered Jacobian's blocks and the
/// Hessian entries the products add to the inner model's (`inner_h_pos`).
fn layout(
    pairs: &[ComplementarityPair],
    signless: &[bool],
    row_entries: &[Vec<(usize, usize)>],
    lift: Lift,
    (n, m, jnnz, hnnz): (usize, usize, usize, usize),
    inner_h_pos: &HashMap<(usize, usize), usize>,
) -> Layout {
    // One-sided pairs only: a mixed pair's row is free, so it has no sign to
    // give a variable bound.
    let mut lift_var = Vec::with_capacity(pairs.len());
    let mut lifted = Vec::new();
    let mut lift_of_row = vec![None; m];
    for (k, p) in pairs.iter().enumerate() {
        let on = !signless[k]
            && match lift {
                Lift::No => false,
                Lift::Auto => row_entries[k].len() >= LIFT_AUTO_MIN_ENTRIES,
                Lift::Yes => true,
            };
        if on {
            let w = n + lifted.len();
            lift_var.push(Some(w));
            lifted.push((k, w));
            lift_of_row[p.row] = Some((k, w));
        } else {
            lift_var.push(None);
        }
    }
    // Lowered Jacobian: the inner entries, one `-s_H` entry per lifted row on
    // its variable, then one block per product row.
    let mut var_slot = Vec::with_capacity(pairs.len());
    let mut jac_block = Vec::with_capacity(pairs.len());
    let mut jac_nnz = jnnz + lifted.len();
    for (k, p) in pairs.iter().enumerate() {
        let ents = &row_entries[k];
        jac_block.push(jac_nnz);
        if lift_var[k].is_some() {
            // `G·w`: the entries on x_j and on w_k.
            jac_nnz += 2;
            var_slot.push(0);
        } else {
            let slot = ents
                .iter()
                .position(|&(_, c)| c == p.var)
                .unwrap_or(ents.len());
            jac_nnz += ents.len() + usize::from(slot == ents.len());
            var_slot.push(slot);
        }
    }
    // Cross entries are (j, c) for each column c of row i; the diagonal
    // (j, j) is touched only when x_j itself appears in row i. A lifted
    // pair's product `G·w` has the one cross entry (w_k, j).
    let mut h_pos = inner_h_pos.clone();
    let mut h_extra = Vec::new();
    for (k, (p, ents)) in pairs.iter().zip(row_entries).enumerate() {
        if let Some(w) = lift_var[k] {
            h_pos.insert((w, p.var), hnnz + h_extra.len());
            h_extra.push((w, p.var));
            continue;
        }
        for c in ents.iter().map(|&(_, c)| c) {
            let key = (p.var.max(c), p.var.min(c));
            if let std::collections::hash_map::Entry::Vacant(e) = h_pos.entry(key) {
                e.insert(hnnz + h_extra.len());
                h_extra.push(key);
            }
        }
    }
    Layout {
        lift_var,
        lifted,
        lift_of_row,
        var_slot,
        jac_block,
        jac_nnz,
        h_extra,
        h_pos,
    }
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

/// How a declared pair is lowered, as [`classify_pair`] reads it off the
/// variable's and the row's bounds; see the module docs ("Pairs").
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PairShape {
    /// Both sides have exactly one finite bound. `g` is the variable side
    /// and `h` the row side, each as `(sign, bound)`: `(+1, lo)` for a lower
    /// bound, `(-1, hi)` for an upper.
    OneSided {
        g: (Number, Number),
        h: (Number, Number),
    },
    /// The row has exactly one finite bound and the variable two: the row's
    /// side names the variable bound in the pair (`g >= c` with `x`'s lower
    /// bound, `g <= c` with its upper, as the `.nl` flag does), and the
    /// other bound stays an ordinary limit. `complements(s >= 0, L >= 0)`
    /// with `L <= 1` declared writes this.
    OneSidedPlusBound {
        g: (Number, Number),
        h: (Number, Number),
    },
    /// The variable boxed `l < u`, the row free: lowered to two product rows.
    Mixed { l: Number, u: Number },
}

/// Why [`validate_pairs`] refused a pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairErrorKind {
    /// The variable or the row index is outside `[0, n)` / `[0, m)`.
    OutOfRange,
    /// The variable is already in an earlier pair.
    VarAlreadyPaired,
    /// The row is already in an earlier pair.
    RowAlreadyPaired,
    /// The bounds are none of the shapes [`PairShape`] names.
    UnsupportedShape,
}

/// A refused pair: its position in the declared list, why, and the message
/// [`MpccTnlp::new`] reports (its `Display`).
#[derive(Debug, Clone, PartialEq)]
pub struct PairError {
    /// Index of the refused pair in the list passed to [`validate_pairs`].
    pub pair: usize,
    pub kind: PairErrorKind,
    pub message: String,
}

impl std::fmt::Display for PairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PairError {}

/// The shape of a pair whose variable has bounds `[xl, xu]` and whose row
/// has `[gl, gu]`, or `None` when it is none of the shapes the wrapper
/// lowers. Bounds at or beyond `±1e19` are infinite.
pub fn classify_pair(xl: Number, xu: Number, gl: Number, gu: Number) -> Option<PairShape> {
    let row = one_sided(gl, gu);
    if let (Some(g), Some(h)) = (one_sided(xl, xu), row) {
        return Some(PairShape::OneSided { g, h });
    }
    if let Some(h) = row
        && xl > -INF
        && xu < INF
    {
        let g = if h.0 > 0.0 { (1.0, xl) } else { (-1.0, xu) };
        return Some(PairShape::OneSidedPlusBound { g, h });
    }
    if is_mixed(xl, xu, gl, gu) {
        return Some(PairShape::Mixed { l: xl, u: xu });
    }
    None
}

/// Check `pairs` against a model with `n` variables and `m` rows and the
/// given bounds, in order: per pair, the indices are in range, the variable
/// and then the row are not in an earlier pair, and the bounds have a shape
/// [`classify_pair`] accepts. Returns one shape per pair, or the first
/// refusal. This is the validator [`MpccTnlp::with_lift`] runs, exposed so a
/// frontend can refuse a pair when it is declared rather than at solve time.
pub fn validate_pairs(
    n: usize,
    m: usize,
    x_l: &[Number],
    x_u: &[Number],
    g_l: &[Number],
    g_u: &[Number],
    pairs: &[ComplementarityPair],
) -> Result<Vec<PairShape>, PairError> {
    let mut seen_row = vec![false; m];
    let mut seen_var = vec![false; n];
    let mut shapes = Vec::with_capacity(pairs.len());
    let err = |pair, kind, message| {
        Err(PairError {
            pair,
            kind,
            message,
        })
    };
    for (k, p) in pairs.iter().enumerate() {
        if p.var >= n || p.row >= m {
            return err(
                k,
                PairErrorKind::OutOfRange,
                format!(
                    "pair {k}: variable {} / row {} out of range (n = {n}, m = {m})",
                    p.var, p.row
                ),
            );
        }
        if std::mem::replace(&mut seen_var[p.var], true) {
            return err(
                k,
                PairErrorKind::VarAlreadyPaired,
                format!("pair {k}: variable {} is already in a pair", p.var),
            );
        }
        if std::mem::replace(&mut seen_row[p.row], true) {
            return err(
                k,
                PairErrorKind::RowAlreadyPaired,
                format!("pair {k}: row {} is already in a pair", p.row),
            );
        }
        let (xl, xu, gl, gu) = (x_l[p.var], x_u[p.var], g_l[p.row], g_u[p.row]);
        match classify_pair(xl, xu, gl, gu) {
            Some(shape) => shapes.push(shape),
            None => {
                return err(
                    k,
                    PairErrorKind::UnsupportedShape,
                    format!(
                        "pair {k}: variable {} has bounds [{xl:e}, {xu:e}] and row {} has \
                         [{gl:e}, {gu:e}]. Two shapes are lowered: one-sided (the row has exactly \
                         one finite bound, and the variable a matching one: `x >= l ⟂ g >= 0`, \
                         either orientation; a second variable bound is kept as an ordinary limit) \
                         and mixed (the variable boxed, the row free: `l <= x <= u ⟂ g`, g >= 0 at l, \
                         g <= 0 at u, g = 0 between). For a box on the row side, put it on a \
                         variable instead (`g(x) = y` as a row, `l <= y <= u ⟂ ...`)",
                        p.var, p.row
                    ),
                );
            }
        }
    }
    Ok(shapes)
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
        Self::with_lift(inner, pairs, lowering, Lift::No)
    }

    /// [`Self::new`], lifting the rows `lift` selects (module docs,
    /// "Lifting").
    pub fn with_lift(
        inner: Rc<RefCell<dyn TNLP>>,
        pairs: Vec<ComplementarityPair>,
        lowering: Lowering,
        lift: Lift,
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

        let shapes =
            validate_pairs(n, m, &x_l, &x_u, &g_l, &g_u, &pairs).map_err(|e| e.to_string())?;
        // Each declared pair becomes one product row (one-sided) or two
        // (mixed); the rest of the wrapper works on these lowered pairs.
        let mut lowered = Vec::with_capacity(pairs.len());
        let mut g_side = Vec::with_capacity(pairs.len());
        let mut h_side = Vec::with_capacity(pairs.len());
        let mut signless = Vec::with_capacity(pairs.len());
        for (p, shape) in pairs.iter().zip(&shapes) {
            match *shape {
                PairShape::OneSided { g, h } | PairShape::OneSidedPlusBound { g, h } => {
                    lowered.push(*p);
                    g_side.push(g);
                    h_side.push(h);
                    signless.push(false);
                }
                // Mixed: `l <= x <= u ⟂ g` with the row free -- g >= 0 at l,
                // g <= 0 at u, g = 0 strictly between -- is exactly the box
                // with `(x - l)·g <= 0` and `(u - x)·(-g) <= 0`: two product
                // rows on the same variable and row, g itself unconstrained
                // in sign.
                PairShape::Mixed { l, u } => {
                    lowered.extend([*p, *p]);
                    g_side.extend([(1.0, l), (-1.0, u)]);
                    h_side.extend([(1.0, 0.0), (-1.0, 0.0)]);
                    signless.extend([true, true]);
                }
            }
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
        let row_entries: Vec<Vec<(usize, usize)>> =
            pairs.iter().map(|p| by_row[p.row].clone()).collect();
        // Inner Hessian structure, 0-based, lower triangle.
        let hnnz = info.nnz_h_lag as usize;
        let mut inner_h_pos = HashMap::new();
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
                inner_h_pos.insert((a.max(b), a.min(b)), k);
            }
        }
        let build = |lift: Lift| {
            layout(
                &pairs,
                &signless,
                &row_entries,
                lift,
                (n, m, jnnz, hnnz),
                &inner_h_pos,
            )
        };
        let layout = build(lift);
        // The unlifted structure too, when lifting changed anything, so a
        // stage that must not see the lift (the l1 penalty) can switch.
        let other_layout = (!layout.lifted.is_empty()).then(|| build(Lift::No));

        Ok(Self {
            inner,
            coupled_tau: match lowering {
                Lowering::Coupled { tau0, .. } => tau0,
                _ => 0.0,
            },
            lowering,
            n,
            m_inner: m,
            index_style: style,
            inner_jac_nnz: jnnz,
            inner_h_nnz: hnnz,
            g_side,
            h_side,
            row_entries,
            layout,
            other_layout,
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
            branch_rows: vec![None; m],
            pairs: declared,
            lowered: pairs,
            signless,
        })
    }

    /// Solve the lifted model (`true`, the setting [`Self::with_lift`] was
    /// given) or the unlifted one (`false`). A no-op when nothing was lifted.
    /// The dimensions change with it, and every `optimize_tnlp` re-reads
    /// them; starts, results and multipliers are in the inner model's terms
    /// either way, so a driver switches between solves and nothing else.
    pub fn set_lift(&mut self, on: bool) {
        if on == !self.layout.lifted.is_empty() {
            return;
        }
        if let Some(other) = self.other_layout.take() {
            self.other_layout = Some(std::mem::replace(&mut self.layout, other));
        }
    }

    /// How many pairs are lifted (module docs, "Lifting").
    pub fn n_lifted(&self) -> usize {
        self.layout.lifted.len()
    }

    /// The lowered model's variable count: the inner `n` plus one per
    /// lifted pair.
    pub fn n_lowered(&self) -> usize {
        self.n + self.layout.lifted.len()
    }

    /// `(G_k, H_k)` as the lowered model's product rows use them, at a
    /// lowered `x`: [`Self::pair_values_from`], except that a lifted pair's
    /// `H_k` is its variable `w_k`, which equals `H_k(x)` only to the lifted
    /// row's residual. `g_inner` must hold the inner rows at `x[..n]`.
    fn product_factors(&self, x: &[Number]) -> Vec<(Number, Number)> {
        let mut v = self.pair_values_from(x);
        for &(k, w) in &self.layout.lifted {
            v[k].1 = x[w];
        }
        v
    }

    /// Inner-model duals from a lowered point's: a lifted row's multiplier
    /// is the one its unlifted inequality would carry, `-s_H (z_L - z_U)` of
    /// its variable (module docs, "Lifting"); the rest pass through.
    ///
    /// At a point where `w`'s stationarity holds this equals
    /// `λ_lifted - s_H ν_k G_k`, the mapping that keeps the declared model's
    /// `x`-stationarity exact. Off it the two differ by that residual, and
    /// this one is chosen deliberately. It is sign-correct by construction
    /// and keeps `λ_i (g_i - c)` the lifted solve's own complementarity
    /// `z_w w`. The driver's `gap_at` reads exactly that product, and the
    /// gap decides whether the cold exact solve runs (`answers_agree`). The
    /// other mapping would move the gap by `r_w H`, and with it a trajectory.
    /// Its own error is what is reported in `mult_g`. Measured on three
    /// `relu_nn` networks, the declared model's dual residual from these
    /// multipliers is about 1e-13 at the reported point, against 1e3 to 2e15
    /// for the same models unlifted (the exact product's degenerate
    /// multipliers). At forced `max_iter` exits it is dominated by that same
    /// degeneracy on both formulations.
    fn inner_duals(
        &self,
        z_l: &[Number],
        z_u: &[Number],
        lambda: &[Number],
    ) -> (Vec<Number>, Vec<Number>, Vec<Number>) {
        let n = self.n;
        let mut lam = lambda.to_vec();
        for &(k, w) in &self.layout.lifted {
            let i = self.lowered[k].row;
            if i < lam.len() {
                let zl = z_l.get(w).copied().unwrap_or(0.0);
                let zu = z_u.get(w).copied().unwrap_or(0.0);
                lam[i] = -self.h_side[k].0 * (zl - zu);
            }
        }
        (
            z_l[..n.min(z_l.len())].to_vec(),
            z_u[..n.min(z_u.len())].to_vec(),
            lam,
        )
    }

    /// The box a branch NLP puts on a lifted row's variable: `g_i` in the
    /// branch's `[l, u]` is `w = s_H (g_i - c)` in the image of that box,
    /// infinite ends kept infinite. `None` when row `i` is not lifted or the
    /// branch does not bound it; `w`'s own bounds are then `[0, inf)`.
    fn lifted_box(&self, i: usize) -> Option<(Number, Number)> {
        let (k, _) = self.layout.lift_of_row.get(i).copied().flatten()?;
        let (l, u) = self.branch_rows.get(i).copied().flatten()?;
        let (sh, c) = self.h_side[k];
        let img = |v: Number| {
            if v <= -INF {
                -sh * 2.0 * INF
            } else if v >= INF {
                sh * 2.0 * INF
            } else {
                sh * (v - c)
            }
        };
        let (a, z) = (img(l), img(u));
        Some((a.min(z), a.max(z)))
    }

    /// Fill a lifted start from an inner-shaped one already in `sp`'s inner
    /// slices: `w = H(x)`; its bound multipliers from the row's, `z_L - z_U =
    /// -s_H λ_i`, the negative part on `z_U` when a branch gives `w` a finite
    /// upper bound and clipped at zero otherwise; and the row's multiplier
    /// from the stationarity in `w`, `s_H λ_i = ν_k G_k - (z_L - z_U)`
    /// (module docs, "Lifting"). The inverse of [`Self::inner_duals`] at a
    /// stationary point, wherever `w`'s bounds can carry the sign.
    fn lift_start(&mut self, sp: &mut StartingPoint<'_>) -> bool {
        if self.layout.lifted.is_empty() {
            return true;
        }
        let (n, m) = (self.n, self.m_inner);
        if !self
            .inner
            .borrow_mut()
            .eval_g(&sp.x[..n], true, &mut self.g_inner)
        {
            return false;
        }
        for &(k, w) in &self.layout.lifted {
            let p = self.lowered[k];
            let ((sg, b), (sh, c)) = (self.g_side[k], self.h_side[k]);
            sp.x[w] = sh * (self.g_inner[p.row] - c);
            let lam_i = if sp.init_lambda {
                sp.lambda[p.row]
            } else {
                0.0
            };
            // `z_L - z_U` of `w` is `-s_H λ_i` ([`Self::inner_duals`]). The
            // negative part has somewhere to go only when `w` has a finite
            // upper bound, under a branch; otherwise it is clipped at zero.
            let v = -sh * lam_i;
            let has_upper = self.lifted_box(p.row).is_some_and(|(_, hi)| hi < INF);
            let (zl, zu) = (v.max(0.0), if has_upper { (-v).max(0.0) } else { 0.0 });
            if sp.init_z {
                sp.z_l[w] = zl;
                sp.z_u[w] = zu;
            }
            if sp.init_lambda {
                let nu = sp.lambda[m + k];
                sp.lambda[p.row] = sh * (nu * sg * (sp.x[p.var] - b) - (zl - zu));
            }
        }
        true
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
        if let Lowering::Coupled { tau0, .. } = lowering {
            self.coupled_tau = tau0;
        }
    }

    /// The relaxation `τ` the product rows are held to now: `Relaxed`'s own,
    /// the coupled solve's current one, `None` for the exact lowerings.
    pub fn current_tau(&self) -> Option<Number> {
        match self.lowering {
            Lowering::Relaxed(t) => Some(t),
            Lowering::Coupled { .. } => Some(self.coupled_tau),
            Lowering::Eq | Lowering::Ineq => None,
        }
    }

    /// Solve a branch NLP instead of a lowering (`None` restores the
    /// lowering); see [`BranchBounds`].
    pub fn set_branch(&mut self, branch: Option<BranchBounds>) {
        self.branch_rows = vec![None; self.m_inner];
        for &(i, l, u) in branch.iter().flat_map(|b| &b.rows) {
            if let Some(slot) = self.branch_rows.get_mut(i) {
                *slot = Some((l, u));
            }
        }
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

    /// `(G_k(x), H_k(x))` for every pair at an inner-shaped `x`: the declared
    /// model's sides, `H` from the row itself whether or not it is lifted.
    /// [`Self::product_factors`] is what the lowered rows use instead.
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
            n: self.n_lowered() as Index,
            m: (self.m_inner + self.lowered.len()) as Index,
            nnz_jac_g: self.layout.jac_nnz as Index,
            nnz_h_lag: (self.inner_h_nnz + self.layout.h_extra.len()) as Index,
            index_style: self.index_style,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        let (n, m) = (self.n, self.m_inner);
        if !self.inner.borrow_mut().get_bounds_info(BoundsInfo {
            x_l: &mut b.x_l[..n],
            x_u: &mut b.x_u[..n],
            g_l: &mut b.g_l[..m],
            g_u: &mut b.g_u[..m],
        }) {
            return false;
        }
        // A lifted row is the equality `g_i - s_H w = c_i`, its sign carried
        // by `w >= 0`.
        for &(k, w) in &self.layout.lifted {
            let i = self.lowered[k].row;
            let c = self.h_side[k].1;
            b.g_l[i] = c;
            b.g_u[i] = c;
            b.x_l[w] = 0.0;
            b.x_u[w] = 2.0 * INF;
        }
        let (lo, hi) = match self.lowering {
            Lowering::Eq => (0.0, 0.0),
            Lowering::Ineq => (-2.0 * INF, 0.0),
            Lowering::Relaxed(tau) => (-2.0 * INF, tau),
            Lowering::Coupled { tau0, .. } => (-2.0 * INF, tau0),
        };
        if let Some(br) = &self.branch {
            // A branch NLP: the pinned sides carry the complementarity, and
            // the product rows constrain nothing.
            for &(j, l, u) in &br.vars {
                b.x_l[j] = l;
                b.x_u[j] = u;
            }
            for &(i, l, u) in &br.rows {
                // A lifted row's bounds move to its variable (`lifted_box`);
                // the row stays the equality.
                if let (Some((lo, hi)), Some((_, w))) =
                    (self.lifted_box(i), self.layout.lift_of_row[i])
                {
                    b.x_l[w] = lo;
                    b.x_u[w] = hi;
                    continue;
                }
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

    fn get_starting_point(&mut self, mut sp: StartingPoint<'_>) -> bool {
        // Every attempt starts at `tau0`: a re-solve inside one
        // `optimize_tnlp` call (`mu_strategy_fallback`, the second-opinion
        // ladder) asks for the starting point again, and τ never rises, so
        // without this the retry would start from the first attempt's
        // near-exact τ at a fresh μ: the cold, tight start the coupling
        // exists to avoid.
        if let Lowering::Coupled { tau0, .. } = self.lowering {
            self.coupled_tau = tau0;
        }
        // Both starts are inner-shaped (a lifted variable has no value of
        // its own to start from); `lift_start` completes them.
        let n = self.n;
        if let Some(st) = &self.start_override {
            if st.x.len() != n
                || (sp.init_z && (st.z_l.len() != n || st.z_u.len() != n))
                || (sp.init_lambda && st.lambda.len() != sp.lambda.len())
            {
                return false;
            }
            sp.x[..n].copy_from_slice(&st.x);
            if sp.init_z {
                sp.z_l[..n].copy_from_slice(&st.z_l);
                sp.z_u[..n].copy_from_slice(&st.z_u);
            }
            if sp.init_lambda {
                sp.lambda.copy_from_slice(&st.lambda);
            }
            return self.lift_start(&mut sp);
        }
        let m = self.m_inner;
        {
            let (head, tail) = sp.lambda.split_at_mut(m.min(sp.lambda.len()));
            if !self.inner.borrow_mut().get_starting_point(StartingPoint {
                init_x: sp.init_x,
                x: &mut sp.x[..n],
                init_z: sp.init_z,
                z_l: &mut sp.z_l[..n],
                z_u: &mut sp.z_u[..n],
                init_lambda: sp.init_lambda,
                lambda: head,
            }) {
                return false;
            }
            if sp.init_lambda {
                tail.fill(0.0);
            }
        }
        self.lift_start(&mut sp)
    }

    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.inner.borrow_mut().eval_f(&x[..self.n], new_x)
    }

    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        let n = self.n;
        grad_f[n..].fill(0.0);
        self.inner
            .borrow_mut()
            .eval_grad_f(&x[..n], new_x, &mut grad_f[..n])
    }

    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        let m = self.m_inner;
        if !self
            .inner
            .borrow_mut()
            .eval_g(&x[..self.n], new_x, &mut g[..m])
        {
            return false;
        }
        self.g_inner.copy_from_slice(&g[..m]);
        for &(k, w) in &self.layout.lifted {
            g[self.lowered[k].row] -= self.h_side[k].0 * x[w];
        }
        // Under `Coupled` the bound stays at `tau0` and the row carries the
        // shift (module docs); a branch NLP leaves the rows unbounded.
        let shift = match self.lowering {
            Lowering::Coupled { tau0, .. } if self.branch.is_none() => tau0 - self.coupled_tau,
            _ => 0.0,
        };
        for (k, (gk, hk)) in self.product_factors(x).into_iter().enumerate() {
            g[m + k] = gk * hk + shift;
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
                for (l, &(k, w)) in self.layout.lifted.iter().enumerate() {
                    irow[jnnz + l] = self.lowered[k].row as Index + off;
                    jcol[jnnz + l] = w as Index + off;
                }
                for (k, p) in self.lowered.iter().enumerate() {
                    let b = self.layout.jac_block[k];
                    let row = (m + k) as Index + off;
                    if let Some(w) = self.layout.lift_var[k] {
                        irow[b] = row;
                        jcol[b] = p.var as Index + off;
                        irow[b + 1] = row;
                        jcol[b + 1] = w as Index + off;
                        continue;
                    }
                    let ents = &self.row_entries[k];
                    for (s, &(_, c)) in ents.iter().enumerate() {
                        irow[b + s] = row;
                        jcol[b + s] = c as Index + off;
                    }
                    if self.layout.var_slot[k] == ents.len() {
                        irow[b + ents.len()] = row;
                        jcol[b + ents.len()] = p.var as Index + off;
                    }
                }
                true
            }
            SparsityRequest::Values { values } => {
                let Some(x) = x else { return false };
                {
                    let xi = &x[..self.n];
                    let mut inner = self.inner.borrow_mut();
                    if !inner.eval_jac_g(
                        Some(xi),
                        new_x,
                        SparsityRequest::Values {
                            values: &mut values[..jnnz],
                        },
                    ) {
                        return false;
                    }
                    if !inner.eval_g(xi, false, &mut self.g_inner) {
                        return false;
                    }
                }
                for (l, &(k, _)) in self.layout.lifted.iter().enumerate() {
                    values[jnnz + l] = -self.h_side[k].0;
                }
                let pv = self.product_factors(x);
                for (k, &(gk, hk)) in pv.iter().enumerate() {
                    let (sg, sh) = (self.g_side[k].0, self.h_side[k].0);
                    let b = self.layout.jac_block[k];
                    if self.layout.lift_var[k].is_some() {
                        // ∇(G·w) = s_G w e_j + G e_w.
                        values[b] = hk * sg;
                        values[b + 1] = gk;
                        continue;
                    }
                    let ents = &self.row_entries[k];
                    // G ∇g_i on row i's columns ...
                    for (s, &(pos, _)) in ents.iter().enumerate() {
                        values[b + s] = gk * sh * values[pos];
                    }
                    // ... plus H on column j, in place or appended.
                    let slot = self.layout.var_slot[k];
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
                for (e, &(a, b)) in self.layout.h_extra.iter().enumerate() {
                    irow[hnnz + e] = a as Index + off;
                    jcol[hnnz + e] = b as Index + off;
                }
                true
            }
            SparsityRequest::Values { values } => {
                let (Some(x), Some(lambda)) = (x, lambda) else {
                    return false;
                };
                let xi = &x[..self.n];
                {
                    let mut inner = self.inner.borrow_mut();
                    if !inner.eval_g(xi, new_x, &mut self.g_inner) {
                        return false;
                    }
                }
                let pv = self.product_factors(x);
                // Row-i multipliers shifted by μ_k G_k (an unlifted pair's
                // product carries `G·∇²g_i`; a lifted one's is bilinear).
                self.lam_inner.copy_from_slice(&lambda[..m]);
                for (k, p) in self.lowered.iter().enumerate() {
                    if self.layout.lift_var[k].is_none() {
                        self.lam_inner[p.row] += lambda[m + k] * pv[k].0 * self.h_side[k].0;
                    }
                }
                let any_unlifted = self.layout.lifted.len() < self.lowered.len();
                values.fill(0.0);
                {
                    let mut inner = self.inner.borrow_mut();
                    if hnnz > 0
                        && !inner.eval_h(
                            Some(xi),
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
                    if any_unlifted
                        && !inner.eval_jac_g(
                            Some(xi),
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
                    if let Some(w) = self.layout.lift_var[k] {
                        // ∂²(G·w)/∂x_j∂w = s_G.
                        values[self.layout.h_pos[&(w, p.var)]] += lambda[m + k] * self.g_side[k].0;
                        continue;
                    }
                    let mu = lambda[m + k] * self.g_side[k].0 * self.h_side[k].0;
                    if mu == 0.0 {
                        continue;
                    }
                    for &(pos, c) in &self.row_entries[k] {
                        let d = mu * self.jac_inner[pos];
                        let key = (p.var.max(c), p.var.min(c));
                        let at = self.layout.h_pos[&key];
                        values[at] += if c == p.var { 2.0 * d } else { d };
                    }
                }
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        let (n, m) = (self.n, self.m_inner);
        self.last_status = Some(sol.status);
        // Inner-shaped, the lifted variables mapped back (module docs,
        // "Lifting").
        let (z_l, z_u, lambda) = self.inner_duals(sol.z_l, sol.z_u, sol.lambda);
        self.last_x = sol.x[..n.min(sol.x.len())].to_vec();
        self.last_lambda = lambda;
        self.last_z_l = z_l;
        self.last_z_u = z_u;
        // The products themselves, `G·H`: under `Coupled` the rows carry the
        // shift `tau0 - τ` too (module docs), which is not part of them.
        let shift = match self.lowering {
            Lowering::Coupled { tau0, .. } if self.branch.is_none() => tau0 - self.coupled_tau,
            _ => 0.0,
        };
        self.last_products = sol
            .g
            .get(m..)
            .map(|p| p.iter().map(|v| v - shift).collect())
            .unwrap_or_default();
        // The inner rows at `x`: a lifted row's value is `g_i - s_H w`, and
        // `g_i` is what the inner model is handed and what `H` is read from.
        let mut g = sol.g[..m.min(sol.g.len())].to_vec();
        for &(k, w) in &self.layout.lifted {
            let i = self.lowered[k].row;
            if i < g.len() && w < sol.x.len() {
                g[i] += self.h_side[k].0 * sol.x[w];
            }
        }
        // A lifted product row holds `G·w`, which equals `G·H(x)` only to the
        // lifted row's residual; the acceptance gates read these as the
        // declared model's products, so `H` comes from `g_i` itself.
        for &(k, _) in &self.layout.lifted {
            let (p, i) = (self.lowered[k], self.lowered[k].row);
            if k < self.last_products.len() && i < g.len() && p.var < sol.x.len() {
                let ((sg, b), (sh, c)) = (self.g_side[k], self.h_side[k]);
                self.last_products[k] = sg * (sol.x[p.var] - b) * sh * (g[i] - c);
            }
        }
        if self.defer_inner_finalize {
            return;
        }
        self.inner.borrow_mut().finalize_solution(
            Solution {
                status: sol.status,
                x: &self.last_x,
                z_l: &self.last_z_l,
                z_u: &self.last_z_u,
                g: &g,
                lambda: &self.last_lambda[..m.min(self.last_lambda.len())],
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
        // Lifted variables extend the variable metadata the same way
        // (`complementarity_lift[k]` beside the inner model's names).
        let n_total = self.n_lowered();
        if n_total > self.n {
            let names: Vec<String> = self
                .layout
                .lifted
                .iter()
                .map(|(k, _)| format!("complementarity_lift[{k}]"))
                .collect();
            for v in var.strings.values_mut() {
                if v.len() == self.n {
                    v.extend(names.iter().cloned());
                }
                v.resize(n_total, String::new());
            }
            for v in var.integers.values_mut() {
                v.resize(n_total, 0);
            }
            for v in var.numerics.values_mut() {
                v.resize(n_total, 0.0);
            }
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

    fn update_with_barrier(&mut self, mu: Number) -> bool {
        let inner_moved = self.inner.borrow_mut().update_with_barrier(mu);
        let Lowering::Coupled {
            floor,
            factor,
            max_cut,
            ..
        } = self.lowering
        else {
            return inner_moved;
        };
        if self.branch.is_some() || !(mu.is_finite() && mu > 0.0) {
            return inner_moved;
        }
        let target = (factor * mu).max(floor).max(max_cut * self.coupled_tau);
        if target < self.coupled_tau {
            self.coupled_tau = target;
            return true;
        }
        inner_moved
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
        self.inner.borrow().scaling_factors().map(|mut v| {
            if v.len() == self.n {
                v.resize(self.n_lowered(), 1.0);
            }
            v
        })
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
        let (inner_x, lifts) = x_scaling.split_at_mut(self.n.min(x_scaling.len()));
        let ok = self
            .inner
            .borrow_mut()
            .get_scaling_parameters(ScalingRequest {
                obj_scaling,
                use_x_scaling,
                x_scaling: inner_x,
                use_g_scaling,
                g_scaling: inner_g,
            });
        products.fill(1.0);
        lifts.fill(1.0);
        ok
    }

    fn get_objective_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        let n = self.n.min(types.len());
        let (inner, lifts) = types.split_at_mut(n);
        if !self
            .inner
            .borrow_mut()
            .get_objective_variables_linearity(inner)
        {
            return false;
        }
        lifts.fill(Linearity::Linear);
        true
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
        let n = self.n.min(types.len());
        if !self
            .inner
            .borrow_mut()
            .get_variables_linearity(&mut types[..n])
        {
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
        let (n, m) = (self.n, self.m_inner);
        let mut inner_var = var.clone();
        for v in inner_var.strings.values_mut() {
            v.truncate(n);
        }
        for v in inner_var.integers.values_mut() {
            v.truncate(n);
        }
        for v in inner_var.numerics.values_mut() {
            v.truncate(n);
        }
        let var = &inner_var;
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
    /// and the columns of its row, or its lifted variable.
    fn product_vars(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.lowered.iter().map(|p| p.var).collect();
        for (k, entries) in self.row_entries.iter().enumerate() {
            match self.layout.lift_var[k] {
                Some(w) => v.push(w),
                None => v.extend(entries.iter().map(|&(_, col)| col)),
            }
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

    const COUPLED: Lowering = Lowering::Coupled {
        tau0: 0.1,
        floor: 1e-8,
        factor: 1.0,
        max_cut: 0.1,
    };

    /// Under `Coupled` the bound is `tau0` throughout and the row carries
    /// `tau0 - τ`, so the row reads `G·H <= τ`; the products recorded at
    /// the end are `G·H` alone.
    #[test]
    fn a_coupled_row_carries_the_shift_and_its_product_does_not() {
        let mut w = wrap(IndexStyle::C, COUPLED);
        let (mut xl, mut xu, mut gl, mut gu) =
            (vec![0.0; 3], vec![0.0; 3], vec![0.0; 5], vec![0.0; 5]);
        assert!(w.get_bounds_info(BoundsInfo {
            x_l: &mut xl,
            x_u: &mut xu,
            g_l: &mut gl,
            g_u: &mut gu
        }));
        assert_eq!((gl[3], gu[3]), (-2e19, 0.1));
        let p = lowered_g(&mut wrap(IndexStyle::C, Lowering::Eq), &X);
        assert_eq!(w.current_tau(), Some(0.1));
        assert_eq!(lowered_g(&mut w, &X)[3], p[3], "no shift before τ moves");
        assert!(w.update_with_barrier(1e-3));
        let tau = w.current_tau().unwrap();
        let g = lowered_g(&mut w, &X);
        assert!((g[3] - (p[3] + 0.1 - tau)).abs() < 1e-15);
        let zeros = [0.0; 3];
        w.finalize_solution(
            Solution {
                status: SolverReturn::Success,
                x: &X,
                z_l: &zeros,
                z_u: &zeros,
                g: &g,
                lambda: &[0.0; 5],
                obj_value: 0.0,
            },
            &IpoptData::default(),
            &IpoptCq::default(),
        );
        assert!(
            (w.last_products()[0] - p[3]).abs() < 1e-15,
            "{:?}",
            w.last_products()
        );
    }

    /// A re-solve inside one `optimize_tnlp` call asks for the starting
    /// point again, and starts back at `tau0` (review of #1135, item 1).
    #[test]
    fn a_coupled_resolve_starts_back_at_tau0() {
        let mut w = wrap(IndexStyle::C, COUPLED);
        for _ in 0..20 {
            w.update_with_barrier(1e-12);
        }
        assert_eq!(w.current_tau(), Some(1e-8));
        let (mut x, mut zl, mut zu, mut lam) =
            (vec![0.0; 3], vec![0.0; 3], vec![0.0; 3], vec![0.0; 5]);
        assert!(w.get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x,
            init_z: false,
            z_l: &mut zl,
            z_u: &mut zu,
            init_lambda: false,
            lambda: &mut lam,
        }));
        assert_eq!(w.current_tau(), Some(0.1));
    }

    /// The law, one assertion per branch: proportional to μ, never more than
    /// a factor `max_cut` at once, never below the floor, never up; a move is
    /// reported, no move is not, and the other lowerings never move.
    #[test]
    fn the_coupled_tau_follows_mu_down_within_its_limits() {
        let mut w = wrap(IndexStyle::C, COUPLED);
        assert!(w.update_with_barrier(5e-2));
        assert_eq!(w.current_tau(), Some(5e-2), "factor·μ");
        assert!(w.update_with_barrier(1e-6));
        assert!(
            (w.current_tau().unwrap() - 5e-3).abs() < 1e-18,
            "capped at max_cut"
        );
        assert!(!w.update_with_barrier(1.0), "never up");
        assert!((w.current_tau().unwrap() - 5e-3).abs() < 1e-18);
        for _ in 0..20 {
            w.update_with_barrier(1e-12);
        }
        assert_eq!(w.current_tau(), Some(1e-8), "the floor");
        assert!(!w.update_with_barrier(1e-12), "at the floor nothing moves");
        w.set_lowering(COUPLED);
        assert_eq!(w.current_tau(), Some(0.1), "set_lowering restarts at tau0");
        let mut r = wrap(IndexStyle::C, Lowering::Relaxed(0.1));
        assert!(!r.update_with_barrier(1e-6));
        assert_eq!(r.current_tau(), Some(0.1));
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

    // ---- `classify_pair` / `validate_pairs`: one test per branch. ----

    const F: Number = 1e19;

    /// The refusal text as `MpccTnlp::new` wrote it before the classifier
    /// was factored out, after the `pair k: variable .. row .. has [..]. `
    /// prefix; pinned literally so a reworded message fails here.
    const SHAPE_TAIL: &str = "Two shapes are lowered: one-sided (the row has exactly \
        one finite bound, and the variable a matching one: `x >= l ⟂ g >= 0`, \
        either orientation; a second variable bound is kept as an ordinary limit) \
        and mixed (the variable boxed, the row free: `l <= x <= u ⟂ g`, g >= 0 at l, \
        g <= 0 at u, g = 0 between). For a box on the row side, put it on a \
        variable instead (`g(x) = y` as a row, `l <= y <= u ⟂ ...`)";

    fn one(xl: Number, xu: Number, gl: Number, gu: Number) -> Result<Vec<PairShape>, PairError> {
        validate_pairs(
            1,
            1,
            &[xl],
            &[xu],
            &[gl],
            &[gu],
            &[ComplementarityPair { row: 0, var: 0 }],
        )
    }

    fn refused(r: Result<Vec<PairShape>, PairError>, kind: PairErrorKind, text: &str) {
        let e = r.unwrap_err();
        assert_eq!(e.kind, kind, "{e}");
        assert_eq!(e.to_string(), text);
        assert_eq!(e.message, text);
    }

    #[test]
    fn classify_one_sided_lower_lower() {
        let s = PairShape::OneSided {
            g: (1.0, 2.0),
            h: (1.0, 0.5),
        };
        assert_eq!(classify_pair(2.0, F, 0.5, F), Some(s));
        assert_eq!(one(2.0, F, 0.5, F).unwrap(), vec![s]);
    }

    #[test]
    fn classify_one_sided_upper_upper() {
        let s = PairShape::OneSided {
            g: (-1.0, 3.0),
            h: (-1.0, -1.0),
        };
        assert_eq!(classify_pair(-F, 3.0, -F, -1.0), Some(s));
        assert_eq!(one(-F, 3.0, -F, -1.0).unwrap(), vec![s]);
    }

    #[test]
    fn classify_one_sided_lower_upper() {
        let s = PairShape::OneSided {
            g: (1.0, 0.0),
            h: (-1.0, 4.0),
        };
        assert_eq!(classify_pair(0.0, F, -F, 4.0), Some(s));
        assert_eq!(one(0.0, F, -F, 4.0).unwrap(), vec![s]);
    }

    #[test]
    fn classify_one_sided_upper_lower() {
        let s = PairShape::OneSided {
            g: (-1.0, 1.0),
            h: (1.0, 0.0),
        };
        assert_eq!(classify_pair(-F, 1.0, 0.0, F), Some(s));
        assert_eq!(one(-F, 1.0, 0.0, F).unwrap(), vec![s]);
    }

    #[test]
    fn classify_plus_bound_row_lower_takes_the_variable_lower() {
        let s = PairShape::OneSidedPlusBound {
            g: (1.0, 0.0),
            h: (1.0, 0.0),
        };
        assert_eq!(classify_pair(0.0, 1.0, 0.0, F), Some(s));
        assert_eq!(one(0.0, 1.0, 0.0, F).unwrap(), vec![s]);
    }

    #[test]
    fn classify_plus_bound_row_upper_takes_the_variable_upper() {
        let s = PairShape::OneSidedPlusBound {
            g: (-1.0, 1.0),
            h: (-1.0, 2.0),
        };
        assert_eq!(classify_pair(0.0, 1.0, -F, 2.0), Some(s));
        assert_eq!(one(0.0, 1.0, -F, 2.0).unwrap(), vec![s]);
    }

    /// A fixed variable against a one-sided row has two finite bounds, so
    /// it is the plus-bound shape (not refused): only the free-row case of a
    /// fixed variable is.
    #[test]
    fn classify_plus_bound_with_a_fixed_variable() {
        assert_eq!(
            classify_pair(1.0, 1.0, 0.0, F),
            Some(PairShape::OneSidedPlusBound {
                g: (1.0, 1.0),
                h: (1.0, 0.0),
            })
        );
    }

    #[test]
    fn classify_mixed() {
        let s = PairShape::Mixed { l: -1.0, u: 2.0 };
        assert_eq!(classify_pair(-1.0, 2.0, -F, F), Some(s));
        assert_eq!(one(-1.0, 2.0, -F, F).unwrap(), vec![s]);
    }

    #[test]
    fn refuses_a_boxed_row() {
        assert_eq!(classify_pair(0.0, F, -1.0, 1.0), None);
        refused(
            one(0.0, F, -1.0, 1.0),
            PairErrorKind::UnsupportedShape,
            &format!(
                "pair 0: variable 0 has bounds [0e0, 1e19] and row 0 has [-1e0, 1e0]. {SHAPE_TAIL}"
            ),
        );
    }

    #[test]
    fn refuses_both_sides_free() {
        assert_eq!(classify_pair(-F, F, -F, F), None);
        refused(
            one(-F, F, -F, F),
            PairErrorKind::UnsupportedShape,
            &format!(
                "pair 0: variable 0 has bounds [-1e19, 1e19] and row 0 has [-1e19, 1e19]. {SHAPE_TAIL}"
            ),
        );
    }

    /// Refused today; the C API's MOI helper (gh#1147 S1-I9) maps this case
    /// to "no pair" itself and relies on the wrapper refusing it.
    #[test]
    fn refuses_a_fixed_variable_with_a_free_row() {
        assert_eq!(classify_pair(1.0, 1.0, -F, F), None);
        refused(
            one(1.0, 1.0, -F, F),
            PairErrorKind::UnsupportedShape,
            &format!(
                "pair 0: variable 0 has bounds [1e0, 1e0] and row 0 has [-1e19, 1e19]. {SHAPE_TAIL}"
            ),
        );
    }

    /// A row equality (`gl == gu`) is a boxed row, not one-sided.
    #[test]
    fn refuses_an_equality_row() {
        assert_eq!(classify_pair(0.0, F, 0.0, 0.0), None);
    }

    fn two(pairs: &[ComplementarityPair]) -> Result<Vec<PairShape>, PairError> {
        // n = 3, m = 2, every pair shape-valid, so only the index checks bite.
        validate_pairs(3, 2, &[0.0; 3], &[F; 3], &[0.0; 2], &[F; 2], pairs)
    }

    const fn cp(row: usize, var: usize) -> ComplementarityPair {
        ComplementarityPair { row, var }
    }

    #[test]
    fn refuses_a_variable_out_of_range() {
        refused(
            two(&[cp(0, 0), cp(1, 3)]),
            PairErrorKind::OutOfRange,
            "pair 1: variable 3 / row 1 out of range (n = 3, m = 2)",
        );
        assert_eq!(two(&[cp(0, 0), cp(1, 3)]).unwrap_err().pair, 1);
    }

    #[test]
    fn refuses_a_row_out_of_range() {
        refused(
            two(&[cp(2, 0)]),
            PairErrorKind::OutOfRange,
            "pair 0: variable 0 / row 2 out of range (n = 3, m = 2)",
        );
    }

    #[test]
    fn refuses_a_duplicate_variable() {
        refused(
            two(&[cp(0, 1), cp(1, 1)]),
            PairErrorKind::VarAlreadyPaired,
            "pair 1: variable 1 is already in a pair",
        );
    }

    #[test]
    fn refuses_a_duplicate_row() {
        refused(
            two(&[cp(1, 0), cp(1, 2)]),
            PairErrorKind::RowAlreadyPaired,
            "pair 1: row 1 is already in a pair",
        );
    }

    /// Checks run per pair in order: an earlier pair's bad shape is reported
    /// before a later pair's bad index, as `MpccTnlp::new` always did.
    #[test]
    fn reports_the_first_refusal_in_declaration_order() {
        let e = validate_pairs(
            2,
            2,
            &[-F, 0.0],
            &[F, F],
            &[0.0, 0.0],
            &[F, F],
            &[cp(0, 0), cp(5, 1)],
        )
        .unwrap_err();
        assert_eq!((e.pair, e.kind), (0, PairErrorKind::UnsupportedShape));
    }

    /// `MpccTnlp::new` reports `validate_pairs`'s message unchanged.
    #[test]
    fn the_wrapper_reports_the_classifier_message() {
        let e = MpccTnlp::new(
            Rc::new(RefCell::new(Toy {
                style: IndexStyle::C,
            })),
            vec![cp(2, 1)],
            Lowering::Eq,
        )
        .err()
        .unwrap();
        // Toy's bounds, read back rather than restated.
        let mut t = Toy {
            style: IndexStyle::C,
        };
        let (mut xl, mut xu, mut gl, mut gu) =
            (vec![0.0; 3], vec![0.0; 3], vec![0.0; 3], vec![0.0; 3]);
        assert!(t.get_bounds_info(BoundsInfo {
            x_l: &mut xl,
            x_u: &mut xu,
            g_l: &mut gl,
            g_u: &mut gu,
        }));
        let want = validate_pairs(3, 3, &xl, &xu, &gl, &gu, &[cp(2, 1)]).unwrap_err();
        assert_eq!(want.kind, PairErrorKind::UnsupportedShape);
        assert_eq!(e, want.to_string());
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

    // ---- Lifting ------------------------------------------------------

    fn lifted<T: TNLP + 'static>(inner: T, lowering: Lowering) -> MpccTnlp {
        MpccTnlp::with_lift(Rc::new(RefCell::new(inner)), pairs(), lowering, Lift::Yes).unwrap()
    }

    /// The lowered Jacobian, dense, at any dimension.
    fn jac_n(w: &mut MpccTnlp, x: &[Number]) -> Vec<Vec<Number>> {
        let info = w.get_nlp_info().unwrap();
        let (n, m, nnz) = (info.n as usize, info.m as usize, info.nnz_jac_g as usize);
        let o = base(info.index_style) as usize;
        let (mut ir, mut jc) = (vec![0; nnz], vec![0; nnz]);
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
        let mut d = vec![vec![0.0; n]; m];
        let mut seen = std::collections::HashSet::new();
        for k in 0..nnz {
            let key = (ir[k] as usize - o, jc[k] as usize - o);
            assert!(seen.insert(key), "duplicate Jacobian entry {key:?}");
            d[key.0][key.1] += v[k];
        }
        d
    }

    /// The lowered Lagrangian Hessian, dense (both triangles), at any
    /// dimension.
    fn hess_n(w: &mut MpccTnlp, x: &[Number], s: Number, l: &[Number]) -> Vec<Vec<Number>> {
        let info = w.get_nlp_info().unwrap();
        let (n, nnz) = (info.n as usize, info.nnz_h_lag as usize);
        let o = base(info.index_style) as usize;
        let (mut ir, mut jc) = (vec![0; nnz], vec![0; nnz]);
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
        let mut d = vec![vec![0.0; n]; n];
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

    /// `s ∇f + Jᵀ λ - z_L + z_U` of the lowered model.
    fn stationarity(
        w: &mut MpccTnlp,
        x: &[Number],
        s: Number,
        lam: &[Number],
        z_l: &[Number],
        z_u: &[Number],
    ) -> Vec<Number> {
        let n = x.len();
        let mut gf = vec![0.0; n];
        assert!(w.eval_grad_f(x, true, &mut gf));
        let j = jac_n(w, x);
        (0..n)
            .map(|c| {
                s * gf[c] + (0..lam.len()).map(|r| lam[r] * j[r][c]).sum::<Number>() - z_l[c]
                    + z_u[c]
            })
            .collect()
    }

    /// Lifted, each product row is `G·w`: two Jacobian entries, one Hessian
    /// entry, against central differences of the wrapper's own `eval_g`, in
    /// both orientations of the row and both index styles, at a point with
    /// `w` off `H(x)` (the derivatives may not lean on the equality).
    #[test]
    fn lifted_derivatives_match_finite_differences() {
        for flipped in [false, true] {
            for style in [IndexStyle::C, IndexStyle::Fortran] {
                let mut w = if flipped {
                    lifted(Flipped(Toy { style }), Lowering::Eq)
                } else {
                    lifted(Toy { style }, Lowering::Eq)
                };
                let info = w.get_nlp_info().unwrap();
                assert_eq!((info.n, info.m), (5, 5));
                assert_eq!(w.n_lifted(), 2);
                // Inner 8, one per lifted row, two per product row.
                assert_eq!(info.nnz_jac_g, 8 + 2 + 2 * 2);
                // Inner 6 and one cross entry per product.
                assert_eq!(info.nnz_h_lag, 6 + 2);
                let x = [X[0], X[1], X[2], 0.37, -0.21];
                let j = jac_n(&mut w, &x);
                let lam = [0.3, -1.2, 0.8, 2.5, -0.7];
                let s = 1.7;
                let hd = hess_n(&mut w, &x, s, &lam);
                let grad_l = |w: &mut MpccTnlp, x: &[Number]| {
                    stationarity(w, x, s, &lam, &[0.0; 5], &[0.0; 5])
                };
                let h = 1e-6;
                for c in 0..5 {
                    let (mut xp, mut xm) = (x, x);
                    xp[c] += h;
                    xm[c] -= h;
                    let (gp, gm) = (lowered_g(&mut w, &xp), lowered_g(&mut w, &xm));
                    for r in 0..5 {
                        let fd = (gp[r] - gm[r]) / (2.0 * h);
                        assert!(
                            (j[r][c] - fd).abs() < 1e-8,
                            "flipped={flipped}: J[{r}][{c}] = {} vs fd {fd}",
                            j[r][c]
                        );
                    }
                    let (lp, lm) = (grad_l(&mut w, &xp), grad_l(&mut w, &xm));
                    for r in 0..5 {
                        let fd = (lp[r] - lm[r]) / (2.0 * h);
                        assert!(
                            (hd[r][c] - fd).abs() < 1e-7,
                            "flipped={flipped}: H[{r}][{c}] = {} vs fd {fd}",
                            hd[r][c]
                        );
                    }
                }
            }
        }
    }

    /// At `w = H(x)` the lifted model is the unlifted one: each lifted row
    /// reads its bound `c` and is held there as an equality, `w` carries the
    /// row's sign as a variable bound, and the product rows have the same
    /// values. Unpaired rows are untouched.
    #[test]
    fn a_lifted_row_is_an_equality_and_its_variable_carries_the_sign() {
        for (flipped, c) in [(false, [0.5, 0.0]), (true, [0.5, 3.0])] {
            let (mut w, mut u) = if flipped {
                let st = || {
                    Flipped(Toy {
                        style: IndexStyle::C,
                    })
                };
                (
                    lifted(st(), Lowering::Eq),
                    MpccTnlp::new(Rc::new(RefCell::new(st())), pairs(), Lowering::Eq).unwrap(),
                )
            } else {
                (
                    lifted(
                        Toy {
                            style: IndexStyle::C,
                        },
                        Lowering::Eq,
                    ),
                    wrap(IndexStyle::C, Lowering::Eq),
                )
            };
            let sh = if flipped { -1.0 } else { 1.0 };
            let gi = Toy::g(&X);
            let x = [X[0], X[1], X[2], sh * (gi[0] - c[0]), sh * (gi[1] - c[1])];
            let (gl, gu) = (lowered_g(&mut w, &x), lowered_g(&mut u, &X));
            for i in 0..2 {
                assert!((gl[i] - c[i]).abs() < 1e-15, "{gl:?}");
            }
            assert_eq!(gl[2], gu[2]);
            for k in 3..5 {
                assert!((gl[k] - gu[k]).abs() < 1e-14, "{gl:?} vs {gu:?}");
            }
            let (mut xl, mut xu, mut bl, mut bu) =
                (vec![0.0; 5], vec![0.0; 5], vec![0.0; 5], vec![0.0; 5]);
            assert!(w.get_bounds_info(BoundsInfo {
                x_l: &mut xl,
                x_u: &mut xu,
                g_l: &mut bl,
                g_u: &mut bu
            }));
            assert_eq!((bl[0], bu[0], bl[1], bu[1]), (c[0], c[0], c[1], c[1]));
            assert_eq!((bl[2], bu[2]), (-10.0, 10.0));
            assert_eq!((xl[3], xu[3], xl[4], xu[4]), (0.0, 2e19, 0.0, 2e19));
        }
    }

    /// The duals map between the two models exactly. An inner-shaped start
    /// (the unlifted model's `λ`, `ν`, `z`) becomes a lifted one whose
    /// stationarity residual is the unlifted one's on `x` and zero on every
    /// `w`; and a lifted result maps back to the same inner-shaped duals, so
    /// the driver, the certificate and a warm start never see the lift.
    #[test]
    fn lifted_duals_map_to_the_unlifted_models() {
        for flipped in [false, true] {
            let (mut w, mut u) = if flipped {
                let st = || {
                    Flipped(Toy {
                        style: IndexStyle::C,
                    })
                };
                (
                    lifted(st(), Lowering::Relaxed(0.1)),
                    MpccTnlp::new(Rc::new(RefCell::new(st())), pairs(), Lowering::Relaxed(0.1))
                        .unwrap(),
                )
            } else {
                (
                    lifted(
                        Toy {
                            style: IndexStyle::C,
                        },
                        Lowering::Relaxed(0.1),
                    ),
                    wrap(IndexStyle::C, Lowering::Relaxed(0.1)),
                )
            };
            let sh: Number = if flipped { -1.0 } else { 1.0 };
            // Row multipliers with the sign an active one-sided row has.
            let lam_o = vec![-sh * 0.4, -sh * 1.3, 0.25, 0.9, -0.6];
            let (z_l, z_u) = (vec![0.1, 0.2, 0.3], vec![0.05, 0.0, 0.01]);
            let s = 1.3;
            w.set_start_override(Some(StartOverride {
                x: X.to_vec(),
                z_l: z_l.clone(),
                z_u: z_u.clone(),
                lambda: lam_o.clone(),
            }));
            let (mut x, mut zl, mut zu, mut lam) =
                (vec![0.0; 5], vec![9.0; 5], vec![9.0; 5], vec![0.0; 5]);
            assert!(w.get_starting_point(StartingPoint {
                init_x: true,
                x: &mut x,
                init_z: true,
                z_l: &mut zl,
                z_u: &mut zu,
                init_lambda: true,
                lambda: &mut lam,
            }));
            assert_eq!(&x[..3], &X);
            let gi = Toy::g(&X);
            let c = if flipped { [0.5, 3.0] } else { [0.5, 0.0] };
            for k in 0..2 {
                assert!((x[3 + k] - sh * (gi[k] - c[k])).abs() < 1e-15, "w = H(x)");
                assert!((zl[3 + k] - -sh * lam_o[k]).abs() < 1e-15, "z_w = -s_H λ");
                assert_eq!(zu[3 + k], 0.0);
            }
            let r_u = stationarity(&mut u, &X, s, &lam_o, &z_l, &z_u);
            let r_l = stationarity(&mut w, &x, s, &lam, &zl, &zu);
            for c in 0..3 {
                assert!((r_l[c] - r_u[c]).abs() < 1e-12, "{r_l:?} vs {r_u:?}");
            }
            for c in 3..5 {
                assert!(r_l[c].abs() < 1e-12, "w-stationarity: {r_l:?}");
            }
            // And back.
            let g = lowered_g(&mut w, &x);
            w.finalize_solution(
                Solution {
                    status: SolverReturn::Success,
                    x: &x,
                    z_l: &zl,
                    z_u: &zu,
                    g: &g,
                    lambda: &lam,
                    obj_value: 0.0,
                },
                &IpoptData::default(),
                &IpoptCq::default(),
            );
            assert_eq!(w.last_x(), &X);
            assert_eq!((w.last_z_l(), w.last_z_u()), (&z_l[..], &z_u[..]));
            for (a, b) in w.last_lambda().iter().zip(&lam_o) {
                assert!((a - b).abs() < 1e-15, "{:?} vs {lam_o:?}", w.last_lambda());
            }
            let pu = lowered_g(&mut u, &X);
            for k in 0..2 {
                assert!((w.last_products()[k] - pu[3 + k]).abs() < 1e-14);
            }
            // Off the lifted row (w ≠ H(x), as at any point the solver
            // returns to a nonzero residual) the products are still the
            // declared model's G·H(x), not the row value G·w.
            let mut xo = x.clone();
            xo[3] += 0.37;
            xo[4] -= 0.21;
            let go = lowered_g(&mut w, &xo);
            assert!(
                (go[3] - pu[3]).abs() > 1e-3,
                "the perturbation must move G·w"
            );
            w.finalize_solution(
                Solution {
                    status: SolverReturn::Success,
                    x: &xo,
                    z_l: &zl,
                    z_u: &zu,
                    g: &go,
                    lambda: &lam,
                    obj_value: 0.0,
                },
                &IpoptData::default(),
                &IpoptCq::default(),
            );
            for k in 0..2 {
                assert!(
                    (w.last_products()[k] - pu[3 + k]).abs() < 1e-14,
                    "product {k}: {} vs G·H(x) = {}",
                    w.last_products()[k],
                    pu[3 + k]
                );
            }
            // Off `w`-stationarity the row multiplier is still `-s_H (z_L -
            // z_U)` of `w`: the mapping that keeps `λ_i (g_i - c)` the
            // solve's own complementarity, which `gap_at` reads (see
            // `inner_duals` for why not `λ_lifted - s_H ν G`).
            let mut zo = zl.clone();
            zo[3] += 0.3;
            w.finalize_solution(
                Solution {
                    status: SolverReturn::Success,
                    x: &x,
                    z_l: &zo,
                    z_u: &zu,
                    g: &g,
                    lambda: &lam,
                    obj_value: 0.0,
                },
                &IpoptData::default(),
                &IpoptCq::default(),
            );
            assert_eq!(w.last_lambda()[0], -sh * (zo[3] - zu[3]));
        }
    }

    /// A branch NLP pins a lifted row through its variable: the image of the
    /// row's box under `w = s_H (g - c)`, in either orientation, with
    /// infinite ends kept infinite; the row stays the equality.
    #[test]
    fn a_branch_on_a_lifted_row_moves_to_its_variable() {
        for (flipped, rows, want) in [
            (false, (0.5, 0.5), (0.0, 0.0)),
            (false, (0.5, 1e19), (0.0, 2e19)),
            (true, (0.5, 0.5), (0.0, 0.0)),
            (true, (-1e19, 0.5), (0.0, 2e19)),
            (true, (-1e19, 1e19), (-2e19, 2e19)),
        ] {
            let mut w = if flipped {
                lifted(
                    Flipped(Toy {
                        style: IndexStyle::C,
                    }),
                    Lowering::Eq,
                )
            } else {
                lifted(
                    Toy {
                        style: IndexStyle::C,
                    },
                    Lowering::Eq,
                )
            };
            w.set_branch(Some(BranchBounds {
                vars: vec![],
                rows: vec![(0, rows.0, rows.1)],
            }));
            let (mut xl, mut xu, mut bl, mut bu) =
                (vec![0.0; 5], vec![0.0; 5], vec![0.0; 5], vec![0.0; 5]);
            assert!(w.get_bounds_info(BoundsInfo {
                x_l: &mut xl,
                x_u: &mut xu,
                g_l: &mut bl,
                g_u: &mut bu
            }));
            assert_eq!((xl[3], xu[3]), want, "flipped={flipped} rows={rows:?}");
            assert_eq!((bl[0], bu[0]), (0.5, 0.5));
            assert_eq!((bl[3], bu[3]), (-2e19, 2e19), "the product row is free");
        }
    }

    /// A branch that gives a lifted variable a finite upper bound gives the
    /// negative part of `z_L - z_U` somewhere to go: a start whose row
    /// multiplier has the "wrong" sign for `w >= 0` lands in `z_U`, and maps
    /// back to the same multiplier. Without the branch it is clipped, as `w`
    /// has no upper bound to carry it.
    #[test]
    fn a_lifted_start_under_a_branch_keeps_the_upper_multiplier() {
        let start = |w: &mut MpccTnlp| {
            let lam_o = vec![0.7, -1.3, 0.25, 0.9, -0.6];
            w.set_start_override(Some(StartOverride {
                x: X.to_vec(),
                z_l: vec![0.1, 0.2, 0.3],
                z_u: vec![0.05, 0.0, 0.01],
                lambda: lam_o.clone(),
            }));
            let (mut x, mut zl, mut zu, mut lam) =
                (vec![0.0; 5], vec![9.0; 5], vec![9.0; 5], vec![0.0; 5]);
            assert!(w.get_starting_point(StartingPoint {
                init_x: true,
                x: &mut x,
                init_z: true,
                z_l: &mut zl,
                z_u: &mut zu,
                init_lambda: true,
                lambda: &mut lam,
            }));
            let g = lowered_g(w, &x);
            w.finalize_solution(
                Solution {
                    status: SolverReturn::Success,
                    x: &x,
                    z_l: &zl,
                    z_u: &zu,
                    g: &g,
                    lambda: &lam,
                    obj_value: 0.0,
                },
                &IpoptData::default(),
                &IpoptCq::default(),
            );
            (zl[3], zu[3], w.last_lambda()[0], lam_o[0])
        };
        let mut w = lifted(
            Toy {
                style: IndexStyle::C,
            },
            Lowering::Eq,
        );
        // Unbranched: `-s_H λ = -0.7` has no upper multiplier to go to.
        assert_eq!(start(&mut w), (0.0, 0.0, 0.0, 0.7));
        // Branched to `g_0 = 0.5`, so `w` is fixed at 0 with a finite upper
        // bound: the start carries it, and it maps back exactly.
        w.set_branch(Some(BranchBounds {
            vars: vec![],
            rows: vec![(0, 0.5, 0.5)],
        }));
        let (zl, zu, back, orig) = start(&mut w);
        assert_eq!((zl, zu), (0.0, 0.7));
        assert!((back - orig).abs() < 1e-15, "{back} vs {orig}");
    }

    /// `set_lift(false)` is the unlifted model, structure and values, and
    /// `set_lift(true)` returns to the lifted one: one wrapper serves a stage
    /// that must not see the lift (the l1 penalty) and the rest.
    #[test]
    fn set_lift_switches_between_the_two_structures() {
        let style = IndexStyle::C;
        let mut w = lifted(Toy { style }, Lowering::Eq);
        let mut u = wrap(style, Lowering::Eq);
        let x5 = [X[0], X[1], X[2], 0.37, -0.21];
        let lifted_jac = jac_n(&mut w, &x5);
        assert_eq!(w.get_nlp_info().unwrap().n, 5);

        w.set_lift(false);
        let (iw, iu) = (w.get_nlp_info().unwrap(), u.get_nlp_info().unwrap());
        assert_eq!(
            (iw.n, iw.m, iw.nnz_jac_g, iw.nnz_h_lag),
            (iu.n, iu.m, iu.nnz_jac_g, iu.nnz_h_lag)
        );
        assert_eq!(jac_n(&mut w, &X), jac_n(&mut u, &X));
        let l = [0.3, -0.2, 0.1, 0.7, -0.4];
        assert_eq!(hess_n(&mut w, &X, 1.1, &l), hess_n(&mut u, &X, 1.1, &l));

        w.set_lift(true);
        assert_eq!(w.get_nlp_info().unwrap().n, 5);
        assert_eq!(jac_n(&mut w, &x5), lifted_jac);
        // A wrapper with nothing lifted has nothing to switch.
        u.set_lift(true);
        assert_eq!(u.get_nlp_info().unwrap().n, 3);
    }

    /// `Auto` lifts by row width: the toy's rows (three and two entries) are
    /// below the threshold, and a row at it is lifted; `No` never lifts.
    #[test]
    fn auto_lifts_only_wide_rows() {
        let toy = || {
            Rc::new(RefCell::new(Toy {
                style: IndexStyle::C,
            }))
        };
        let w = MpccTnlp::with_lift(toy(), pairs(), Lowering::Eq, Lift::Auto).unwrap();
        assert_eq!(w.n_lifted(), 0);
        assert_eq!(w.n_lowered(), 3);
        let w = MpccTnlp::with_lift(toy(), pairs(), Lowering::Eq, Lift::No).unwrap();
        assert_eq!(w.n_lifted(), 0);

        /// `y >= 0 ⟂ y - Σ x_i >= 0` over `LIFT_AUTO_MIN_ENTRIES - 1` inputs:
        /// a row of exactly the threshold width.
        struct Wide;
        const K: usize = LIFT_AUTO_MIN_ENTRIES - 1;
        impl TNLP for Wide {
            fn get_nlp_info(&mut self) -> Option<NlpInfo> {
                Some(NlpInfo {
                    n: (K + 1) as Index,
                    m: 1,
                    nnz_jac_g: (K + 1) as Index,
                    nnz_h_lag: 0,
                    index_style: IndexStyle::C,
                })
            }
            fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
                b.x_l.fill(0.0);
                b.x_u.fill(1e19);
                b.g_l[0] = 0.0;
                b.g_u[0] = 1e19;
                true
            }
            fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
                sp.x.fill(1.0);
                true
            }
            fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
                Some(x.iter().sum())
            }
            fn eval_grad_f(&mut self, _: &[Number], _: bool, g: &mut [Number]) -> bool {
                g.fill(1.0);
                true
            }
            fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
                g[0] = x[K] - x[..K].iter().sum::<Number>();
                true
            }
            fn eval_jac_g(
                &mut self,
                _: Option<&[Number]>,
                _: bool,
                mode: SparsityRequest<'_>,
            ) -> bool {
                match mode {
                    SparsityRequest::Structure { irow, jcol } => {
                        irow.fill(0);
                        for (k, c) in jcol.iter_mut().enumerate() {
                            *c = k as Index;
                        }
                    }
                    SparsityRequest::Values { values } => {
                        values.fill(-1.0);
                        values[K] = 1.0;
                    }
                }
                true
            }
            fn eval_h(
                &mut self,
                _: Option<&[Number]>,
                _: bool,
                _: Number,
                _: Option<&[Number]>,
                _: bool,
                _: SparsityRequest<'_>,
            ) -> bool {
                true
            }
            fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
        }
        let pair = vec![ComplementarityPair { row: 0, var: K }];
        let mut w = MpccTnlp::with_lift(
            Rc::new(RefCell::new(Wide)),
            pair.clone(),
            Lowering::Eq,
            Lift::Auto,
        )
        .unwrap();
        assert_eq!(w.n_lifted(), 1);
        let info = w.get_nlp_info().unwrap();
        assert_eq!(
            (info.n, info.nnz_jac_g, info.nnz_h_lag),
            (K as Index + 2, K as Index + 4, 1)
        );
        let mut u = MpccTnlp::new(Rc::new(RefCell::new(Wide)), pair, Lowering::Eq).unwrap();
        let info = u.get_nlp_info().unwrap();
        // Unlifted: the row twice, and a Hessian entry per column.
        assert_eq!(
            (info.n, info.nnz_jac_g, info.nnz_h_lag),
            (K as Index + 1, 2 * (K as Index + 1), K as Index + 1)
        );
    }
}
