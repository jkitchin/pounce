//! Phase 6, planning half — decide which variables a model's linear
//! equality rows determine (issue #487).
//!
//! This is the "aggregation" pass option (b) of gh#487: one
//! implementation in `pounce-presolve` so every frontend — CLI, GAMS, the
//! C interface, Pyomo — gets the same reduction, rather than borrowing
//! Pyomo's NL-v2 writer presolve for the Pyomo path alone.
//!
//! # What it recognises
//!
//! A work list over three shapes, iterated to a fixed point so chains
//! propagate (matching what Pyomo's NL-v2 linear presolve does):
//!
//! 1. **Variables fixed by equal bounds** (`x_l == x_u`) become
//!    constants, so rows mentioning them shed a term.
//! 2. **Singleton linear equality rows** `a·x = b` fix their variable at
//!    `x := b/a`, with a bounds check.
//! 3. **Two-variable linear equality rows** `a₁·x + a₂·y = b` substitute
//!    one variable for the other, `x := α·y + β`, with **no anchoring
//!    requirement** — free/free pairs (arc equalities, `Reference`
//!    aliases, unit-conversion links) aggregate away. This is the shape
//!    [`crate::auxiliary`]'s determined-block pipeline cannot reach, and
//!    the one that closes the gap gh#487 measured.
//!
//! Rows that collapse to `0 = 0` under the accumulated substitutions are
//! structurally redundant and are dropped too (they carry `λ = 0`; see
//! [`crate::linear_eq_elim`] for the dual story).
//!
//! # What it produces
//!
//! An [`EliminationPlan`]: an affine map `x_full = A·y + c` in which every
//! eliminated variable is `α·y_rep + β` for a **single** surviving
//! variable `rep` (or an outright constant). That one-nonzero-per-row
//! shape is what keeps the derivative transforms in
//! [`crate::linear_eq_elim`] to a scaled gather rather than a sparse
//! matrix product.
//!
//! # Failing closed
//!
//! Any contradiction found on the way — a singleton value outside its
//! variable's box, a bound transfer that empties the survivor's box, a row
//! reduced to `0 = b` with `b ≠ 0` — abandons the **whole** plan and
//! returns the identity. Presolve's own certification path (which knows
//! how to withdraw a verdict a witness refutes) then sees the model
//! untouched and decides for itself. An elimination pass is the wrong
//! place to be the first and only voice calling a model infeasible.

use pounce_common::types::{Number, lower_bound_present, upper_bound_present};

/// How one full-space variable is recovered from the reduced solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VarRecovery {
    /// Survives, at the given index in the reduced variable vector.
    Kept(usize),
    /// Eliminated to a constant.
    Constant(Number),
    /// Eliminated to `coeff * x[rep] + offset`, where `rep` is a
    /// **surviving** full-space variable index and `coeff` is non-zero.
    Affine {
        rep: usize,
        coeff: Number,
        offset: Number,
    },
}

/// One accepted elimination, in application order.
///
/// The postsolve multiplier recovery in [`crate::linear_eq_elim`] walks
/// these in reverse, which is what makes the dual system triangular; see
/// that module's `recover_dropped_multipliers` for why.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElimStep {
    /// Full-space row consumed by this step.
    pub row: usize,
    /// Full-space variable it determined.
    pub var: usize,
    /// The row's coefficient on `var` **in the partially substituted
    /// problem at the moment of the step** — i.e. the pivot. Non-zero.
    pub pivot: Number,
}

/// Tunables for [`build_plan`].
#[derive(Debug, Clone, Copy)]
pub struct PlanConfig {
    /// `|g_u - g_l|` at or below this makes a row an equality.
    pub eq_tol: Number,
    /// An accumulated coefficient at or below `coeff_tol * row_scale` is
    /// treated as structurally absent.
    pub coeff_tol: Number,
    /// How far a derived value may sit outside a declared bound before the
    /// plan is abandoned as contradictory. Below this the value is clamped
    /// into the box instead — the same "float noise is not an empty set"
    /// reading `PresolveTnlp` applies to sub-margin crossings.
    pub feas_tol: Number,
    /// Cap on fixed-point sweeps over the candidate rows.
    pub max_passes: usize,
}

impl Default for PlanConfig {
    fn default() -> Self {
        Self {
            eq_tol: 1e-12,
            coeff_tol: 1e-12,
            feas_tol: 1e-8,
            max_passes: 50,
        }
    }
}

/// Everything [`build_plan`] reads about the problem.
#[derive(Debug, Clone, Copy)]
pub struct PlanInput<'a> {
    pub n_vars: usize,
    pub n_rows: usize,
    /// Per-row `(column, coefficient)` lists. Only rows flagged linear and
    /// equality are consulted; the caller may leave the rest empty.
    pub rows: &'a [Vec<(usize, Number)>],
    /// Per-row constant term `c` in `g_r(x) = Σ a_j x_j + c`, so the row
    /// reads `Σ a_j x_j = g_l[r] - c`.
    pub row_const: &'a [Number],
    pub g_l: &'a [Number],
    pub g_u: &'a [Number],
    /// `true` where the row is linear **and** eligible to be consumed.
    pub eligible: &'a [bool],
    pub x_l: &'a [Number],
    pub x_u: &'a [Number],
}

/// A summary of what the pass achieved, for the `Presolve:` console line
/// and for tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinearEqElimReport {
    /// Variables pinned to a constant (by an equal-bounds pair, or by a
    /// singleton row).
    pub n_constant_vars: usize,
    /// Variables folded onto another variable by a two-term row.
    pub n_aggregated_vars: usize,
    /// Rows consumed to determine a variable.
    pub n_rows_eliminated: usize,
    /// Rows that collapsed to `0 = 0` and were dropped as redundant.
    pub n_redundant_rows: usize,
    /// Fixed-point sweeps actually performed.
    pub passes: usize,
    /// The sweep hit [`PlanConfig::max_passes`] with work still to do.
    pub pass_cap_hit: bool,
    /// A contradiction was found; the returned plan is the identity.
    pub infeasible: bool,
}

/// The affine reduction `x_full = A·y + c` plus the bookkeeping postsolve
/// needs.
#[derive(Debug, Clone, Default)]
pub struct EliminationPlan {
    pub n_full: usize,
    pub m_full: usize,
    /// One entry per full-space variable.
    pub recovery: Vec<VarRecovery>,
    /// Full-space index of each surviving variable, ascending.
    pub vars_kept: Vec<usize>,
    /// `true` where the full-space row survives into the reduced problem.
    pub row_kept: Vec<bool>,
    /// Full-space index of each surviving row, ascending.
    pub rows_kept: Vec<usize>,
    /// Reduced variable bounds, aligned with [`Self::vars_kept`]. Carries
    /// every bound transferred off an eliminated variable.
    pub x_l_red: Vec<Number>,
    pub x_u_red: Vec<Number>,
    /// Provenance of each reduced bound: the **full-space column whose own
    /// declared bound** the reduced bound came from, aligned with
    /// [`Self::x_l_red`] / [`Self::x_u_red`]. Equal to `vars_kept[i]` when
    /// the survivor's own bound won, and some eliminated column of the same
    /// cluster when a transferred bound did.
    ///
    /// Postsolve reads this to hand a reduced bound's multiplier back to the
    /// column that actually owns the bound; see
    /// [`crate::linear_eq_elim`]'s `attribute_bound_multiplier` (issue
    /// #493). Which *side* of the origin column's box a reduced bound
    /// corresponds to is not recorded separately: it is the origin's own
    /// side when the composed coefficient `α` of `x_src = α·x_kept + β` is
    /// positive and the opposite side when it is negative, which is exactly
    /// the flip [`transfer_bounds`] performs on the way in.
    pub x_l_src: Vec<usize>,
    pub x_u_src: Vec<usize>,
    /// Accepted eliminations in application order.
    pub steps: Vec<ElimStep>,
    /// Elimination forest: `parent[i] = Some((p, α))` when `i` was folded
    /// onto `p` with `x_i = α·x_p + β` **at that moment** (`p` may itself
    /// be eliminated later). `None` for survivors and for variables pinned
    /// to a constant.
    pub parent: Vec<Option<(usize, Number)>>,
    pub report: LinearEqElimReport,
}

impl EliminationPlan {
    /// The do-nothing plan: every variable and row survives.
    pub fn identity(n_vars: usize, n_rows: usize, x_l: &[Number], x_u: &[Number]) -> Self {
        Self {
            n_full: n_vars,
            m_full: n_rows,
            recovery: (0..n_vars).map(VarRecovery::Kept).collect(),
            vars_kept: (0..n_vars).collect(),
            row_kept: vec![true; n_rows],
            rows_kept: (0..n_rows).collect(),
            x_l_red: x_l.to_vec(),
            x_u_red: x_u.to_vec(),
            x_l_src: (0..n_vars).collect(),
            x_u_src: (0..n_vars).collect(),
            steps: Vec::new(),
            parent: vec![None; n_vars],
            report: LinearEqElimReport::default(),
        }
    }

    /// True when the plan removes nothing, so every wrapper method can take
    /// its forwarding fast path.
    pub fn is_identity(&self) -> bool {
        self.steps.is_empty() && self.report.n_redundant_rows == 0
    }

    pub fn n_reduced_vars(&self) -> usize {
        self.vars_kept.len()
    }

    pub fn n_reduced_rows(&self) -> usize {
        self.rows_kept.len()
    }

    /// Splice a reduced primal back into full space.
    pub fn lift_x(&self, x_red: &[Number], out: &mut [Number]) {
        debug_assert_eq!(out.len(), self.n_full);
        // Survivors first, so the `Affine` arm below can read its
        // representative's value straight out of `out` regardless of index
        // order (`rep` is always a survivor, never another eliminated var).
        for (red, &full) in self.vars_kept.iter().enumerate() {
            out[full] = x_red[red];
        }
        for (i, rec) in self.recovery.iter().enumerate() {
            match *rec {
                VarRecovery::Kept(_) => {}
                VarRecovery::Constant(c) => out[i] = c,
                VarRecovery::Affine { rep, coeff, offset } => out[i] = coeff * out[rep] + offset,
            }
        }
    }

    /// Project a full-space primal down, by reading the survivors.
    pub fn project_x(&self, x_full: &[Number], out: &mut [Number]) {
        for (red, &full) in self.vars_kept.iter().enumerate() {
            out[red] = x_full[full];
        }
    }
}

/// Bounds held with `±f64::INFINITY` for "absent", so arithmetic on them
/// behaves; converted back to the caller's sentinels on the way out.
struct Box2 {
    lo: Vec<Number>,
    hi: Vec<Number>,
    /// Which full-space column's own declared bound each entry came from.
    /// Starts as the identity and follows the bound through every transfer,
    /// so a survivor's final box knows where each of its two sides was born
    /// (gh#493).
    lo_src: Vec<usize>,
    hi_src: Vec<usize>,
}

impl Box2 {
    fn from_declared(x_l: &[Number], x_u: &[Number]) -> Self {
        Self {
            lo_src: (0..x_l.len()).collect(),
            hi_src: (0..x_u.len()).collect(),
            lo: x_l
                .iter()
                .map(|&v| {
                    if lower_bound_present(v) {
                        v
                    } else {
                        Number::NEG_INFINITY
                    }
                })
                .collect(),
            hi: x_u
                .iter()
                .map(|&v| {
                    if upper_bound_present(v) {
                        v
                    } else {
                        Number::INFINITY
                    }
                })
                .collect(),
        }
    }
}

/// Union-find over variables, carrying the affine map to the current root.
struct Substitutions {
    /// `rep[i]` is `i` for a root, else a (possibly stale) ancestor.
    rep: Vec<usize>,
    /// `x_i = to_root[i].0 * x_{rep[i]} + to_root[i].1`.
    to_root: Vec<(Number, Number)>,
    /// Members in each root's cluster, for the union-by-size tie-break.
    cluster_size: Vec<usize>,
    /// Set once a root is pinned to a value.
    root_const: Vec<Option<Number>>,
}

impl Substitutions {
    fn new(n: usize) -> Self {
        Self {
            rep: (0..n).collect(),
            to_root: vec![(1.0, 0.0); n],
            cluster_size: vec![1; n],
            root_const: vec![None; n],
        }
    }

    /// Resolve `i` to its current root, path-compressing on the way back
    /// down. Returns `(root, a, b)` with `x_i = a·x_root + b`.
    fn find(&mut self, i: usize) -> (usize, Number, Number) {
        let mut cur = i;
        let mut path: Vec<usize> = Vec::new();
        while self.rep[cur] != cur {
            path.push(cur);
            cur = self.rep[cur];
        }
        let root = cur;
        let (mut acc_a, mut acc_b) = (1.0, 0.0);
        for &node in path.iter().rev() {
            let (a, b) = self.to_root[node];
            let na = a * acc_a;
            let nb = a * acc_b + b;
            self.rep[node] = root;
            self.to_root[node] = (na, nb);
            acc_a = na;
            acc_b = nb;
        }
        if i == root {
            (root, 1.0, 0.0)
        } else {
            let (a, b) = self.to_root[i];
            (root, a, b)
        }
    }
}

/// Build the reduction. Never panics on a contradictory model: see the
/// module docs on failing closed.
pub fn build_plan(input: &PlanInput<'_>, cfg: &PlanConfig) -> EliminationPlan {
    let n = input.n_vars;
    let m = input.n_rows;
    let identity = || EliminationPlan::identity(n, m, input.x_l, input.x_u);
    if n == 0 {
        return identity();
    }

    let mut subs = Substitutions::new(n);
    let mut bounds = Box2::from_declared(input.x_l, input.x_u);
    let mut parent: Vec<Option<(usize, Number)>> = vec![None; n];
    let mut steps: Vec<ElimStep> = Vec::new();
    let mut row_consumed = vec![false; m];
    let mut redundant_rows: Vec<usize> = Vec::new();
    let mut report = LinearEqElimReport::default();

    // Shape 1: variables the declared box already pins. Folding them in
    // here (rather than leaving them to the algorithm's fixed-variable
    // classification) is what lets a three-term row with one fixed
    // variable become a two-term row the aggregation can consume.
    for j in 0..n {
        let (lo, hi) = (bounds.lo[j], bounds.hi[j]);
        if !lo.is_finite() || !hi.is_finite() {
            continue;
        }
        if hi - lo <= cfg.eq_tol * lo.abs().max(hi.abs()).max(1.0) {
            subs.root_const[j] = Some(0.5 * (lo + hi));
            report.n_constant_vars += 1;
        }
    }

    // Shapes 2 and 3, swept to a fixed point.
    let mut candidates: Vec<usize> = (0..m)
        .filter(|&r| {
            input.eligible[r]
                && lower_bound_present(input.g_l[r])
                && upper_bound_present(input.g_u[r])
                && (input.g_u[r] - input.g_l[r]).abs() <= cfg.eq_tol * input.g_l[r].abs().max(1.0)
        })
        .collect();
    if candidates.is_empty() {
        return finish(
            n,
            m,
            input,
            subs,
            bounds,
            parent,
            steps,
            redundant_rows,
            report,
        );
    }

    let mut terms: Vec<(usize, Number)> = Vec::new();
    for pass in 0..cfg.max_passes.max(1) {
        let mut changed = false;
        report.passes = pass + 1;
        for &r in &candidates {
            if row_consumed[r] {
                continue;
            }
            // Re-express the row over the *current* representatives.
            terms.clear();
            let mut rhs = input.g_l[r] - input.row_const[r];
            let mut row_scale: Number = 0.0;
            let mut ok = true;
            for &(j, a) in &input.rows[r] {
                if a == 0.0 {
                    continue;
                }
                if j >= n {
                    ok = false;
                    break;
                }
                let (root, ra, rb) = subs.find(j);
                row_scale = row_scale.max((a * ra).abs());
                match subs.root_const[root] {
                    Some(c) => rhs -= a * (ra * c + rb),
                    None => {
                        rhs -= a * rb;
                        match terms.iter_mut().find(|(v, _)| *v == root) {
                            Some(slot) => slot.1 += a * ra,
                            None => terms.push((root, a * ra)),
                        }
                    }
                }
            }
            if !ok || !rhs.is_finite() {
                continue;
            }
            let drop_below = cfg.coeff_tol * row_scale.max(1.0);
            terms.retain(|&(_, a)| a.abs() > drop_below);

            match terms.len() {
                0 => {
                    // `0 = rhs`. Zero (to tolerance) means the row carries no
                    // information left and can go; anything else is a
                    // contradiction, and the whole plan stands down.
                    if rhs.abs() <= cfg.feas_tol * row_scale.max(1.0) {
                        row_consumed[r] = true;
                        redundant_rows.push(r);
                        report.n_redundant_rows += 1;
                        changed = true;
                    } else {
                        report.infeasible = true;
                        return abandoned(identity(), report);
                    }
                }
                1 => {
                    let (v, a) = terms[0];
                    let value = rhs / a;
                    if !value.is_finite() {
                        continue;
                    }
                    match clamp_into_box(value, bounds.lo[v], bounds.hi[v], cfg.feas_tol) {
                        Some(pinned) => {
                            subs.root_const[v] = Some(pinned);
                            bounds.lo[v] = pinned;
                            bounds.hi[v] = pinned;
                            row_consumed[r] = true;
                            steps.push(ElimStep {
                                row: r,
                                var: v,
                                pivot: a,
                            });
                            report.n_constant_vars += 1;
                            report.n_rows_eliminated += 1;
                            changed = true;
                        }
                        None => {
                            report.infeasible = true;
                            return abandoned(identity(), report);
                        }
                    }
                }
                2 => {
                    let (v0, a0) = terms[0];
                    let (v1, a1) = terms[1];
                    // Pivot on the larger coefficient so the substitution
                    // multiplier |a_other / a_pivot| never exceeds 1. When the
                    // two are comparable — the `x - y = 0` alias case, which is
                    // most of them — break the tie by cluster size so the
                    // elimination forest stays shallow; postsolve walks its
                    // ancestor chains once per dropped-row nonzero.
                    let (elim, keep, a_elim, a_keep) = if a0.abs() > 4.0 * a1.abs() {
                        (v0, v1, a0, a1)
                    } else if a1.abs() > 4.0 * a0.abs() {
                        (v1, v0, a1, a0)
                    } else if subs.cluster_size[v0] <= subs.cluster_size[v1] {
                        (v0, v1, a0, a1)
                    } else {
                        (v1, v0, a1, a0)
                    };
                    let alpha = -a_keep / a_elim;
                    let beta = rhs / a_elim;
                    if !alpha.is_finite() || !beta.is_finite() || alpha == 0.0 {
                        continue;
                    }
                    // Transfer the eliminated variable's box onto the survivor.
                    if !transfer_bounds(&mut bounds, elim, keep, alpha, beta, cfg.feas_tol) {
                        report.infeasible = true;
                        return abandoned(identity(), report);
                    }
                    subs.rep[elim] = keep;
                    subs.to_root[elim] = (alpha, beta);
                    subs.cluster_size[keep] += subs.cluster_size[elim];
                    parent[elim] = Some((keep, alpha));
                    row_consumed[r] = true;
                    steps.push(ElimStep {
                        row: r,
                        var: elim,
                        pivot: a_elim,
                    });
                    report.n_aggregated_vars += 1;
                    report.n_rows_eliminated += 1;
                    changed = true;
                }
                _ => {}
            }
        }
        candidates.retain(|&r| !row_consumed[r]);
        if !changed || candidates.is_empty() {
            break;
        }
        if pass + 1 == cfg.max_passes.max(1) {
            report.pass_cap_hit = true;
        }
    }

    finish(
        n,
        m,
        input,
        subs,
        bounds,
        parent,
        steps,
        redundant_rows,
        report,
    )
}

fn abandoned(mut plan: EliminationPlan, report: LinearEqElimReport) -> EliminationPlan {
    plan.report = LinearEqElimReport {
        infeasible: report.infeasible,
        passes: report.passes,
        ..LinearEqElimReport::default()
    };
    plan
}

/// Accept `value` as the pinned value of a variable whose box is
/// `[lo, hi]`, or refuse when it sits outside by more than `tol`.
///
/// A sub-tolerance excursion is clamped rather than refused, for the same
/// reason `PresolveTnlp` collapses a sub-margin crossed box to a point:
/// binary float noise around a bound is not an empty feasible set, and
/// treating it as one flips a model POUNCE solves cleanly into a
/// contradiction verdict.
fn clamp_into_box(value: Number, lo: Number, hi: Number, tol: Number) -> Option<Number> {
    let scale = value
        .abs()
        .max(lo.abs().min(1e19))
        .max(hi.abs().min(1e19))
        .max(1.0);
    if lo.is_finite() && value < lo {
        if lo - value > tol * scale {
            return None;
        }
        return Some(lo);
    }
    if hi.is_finite() && value > hi {
        if value - hi > tol * scale {
            return None;
        }
        return Some(hi);
    }
    Some(value)
}

/// Push `elim`'s box through `x_elim = α·x_keep + β` onto `keep`'s box.
/// Returns `false` when the intersection is empty beyond `tol`.
fn transfer_bounds(
    bounds: &mut Box2,
    elim: usize,
    keep: usize,
    alpha: Number,
    beta: Number,
    tol: Number,
) -> bool {
    let (lo_e, hi_e) = (bounds.lo[elim], bounds.hi[elim]);
    // x_elim ∈ [lo_e, hi_e]  ⟺  x_keep ∈ [(lo_e-β)/α, (hi_e-β)/α] (α>0)
    //                            x_keep ∈ [(hi_e-β)/α, (lo_e-β)/α] (α<0)
    let a = (lo_e - beta) / alpha;
    let b = (hi_e - beta) / alpha;
    let (mut derived_lo, mut derived_hi) = if alpha > 0.0 { (a, b) } else { (b, a) };
    if !derived_lo.is_finite() {
        derived_lo = Number::NEG_INFINITY;
    }
    if !derived_hi.is_finite() {
        derived_hi = Number::INFINITY;
    }
    // The same flip the interval carries: with α < 0 it is `elim`'s *upper*
    // bound that becomes the survivor's lower bound. Provenance rides along
    // (gh#493), so a bound that has hopped several times still names the
    // column that declared it.
    let (src_lo, src_hi) = if alpha > 0.0 {
        (bounds.lo_src[elim], bounds.hi_src[elim])
    } else {
        (bounds.hi_src[elim], bounds.lo_src[elim])
    };
    // Strict comparisons, so a tie leaves the incumbent — including the
    // survivor's own declared bound — in place.
    if derived_lo > bounds.lo[keep] {
        bounds.lo[keep] = derived_lo;
        bounds.lo_src[keep] = src_lo;
    }
    if derived_hi < bounds.hi[keep] {
        bounds.hi[keep] = derived_hi;
        bounds.hi_src[keep] = src_hi;
    }
    let (lo, hi) = (bounds.lo[keep], bounds.hi[keep]);
    if lo.is_finite() && hi.is_finite() && lo > hi {
        let scale = lo.abs().max(hi.abs()).max(1.0);
        if lo - hi > tol * scale {
            return false;
        }
        // Float-noise crossing: collapse to a point rather than call the
        // model empty.
        let mid = 0.5 * (lo + hi);
        bounds.lo[keep] = mid;
        bounds.hi[keep] = mid;
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn finish(
    n: usize,
    m: usize,
    input: &PlanInput<'_>,
    mut subs: Substitutions,
    bounds: Box2,
    parent: Vec<Option<(usize, Number)>>,
    steps: Vec<ElimStep>,
    redundant_rows: Vec<usize>,
    report: LinearEqElimReport,
) -> EliminationPlan {
    if steps.is_empty() && redundant_rows.is_empty() {
        let mut plan = EliminationPlan::identity(n, m, input.x_l, input.x_u);
        plan.report = report;
        return plan;
    }

    // Roots that were never pinned survive.
    let mut recovery = vec![VarRecovery::Kept(usize::MAX); n];
    let mut vars_kept: Vec<usize> = Vec::new();
    let mut reduced_of = vec![usize::MAX; n];
    for (j, slot) in reduced_of.iter_mut().enumerate() {
        let (root, _, _) = subs.find(j);
        if root == j && subs.root_const[j].is_none() {
            *slot = vars_kept.len();
            vars_kept.push(j);
        }
    }
    if vars_kept.is_empty() {
        // Every column gone. The reduced problem has no degrees of freedom
        // left, which the IPM has no useful shape for; hand the model back
        // untouched rather than invent one.
        let mut plan = EliminationPlan::identity(n, m, input.x_l, input.x_u);
        plan.report = LinearEqElimReport {
            passes: report.passes,
            ..LinearEqElimReport::default()
        };
        return plan;
    }
    for j in 0..n {
        let (root, a, b) = subs.find(j);
        recovery[j] = match subs.root_const[root] {
            Some(c) => VarRecovery::Constant(a * c + b),
            None if root == j => VarRecovery::Kept(reduced_of[j]),
            None => VarRecovery::Affine {
                rep: root,
                coeff: a,
                offset: b,
            },
        };
    }

    let mut row_kept = vec![true; m];
    for s in &steps {
        row_kept[s.row] = false;
    }
    for &r in &redundant_rows {
        row_kept[r] = false;
    }
    let rows_kept: Vec<usize> = (0..m).filter(|&r| row_kept[r]).collect();

    // Reduced box: keep the caller's own sentinel where nothing was
    // derived, so an "absent" bound stays spelled the way it arrived.
    let mut x_l_red = Vec::with_capacity(vars_kept.len());
    let mut x_u_red = Vec::with_capacity(vars_kept.len());
    let mut x_l_src = Vec::with_capacity(vars_kept.len());
    let mut x_u_src = Vec::with_capacity(vars_kept.len());
    for &j in &vars_kept {
        if bounds.lo[j].is_finite() {
            x_l_red.push(bounds.lo[j]);
            x_l_src.push(bounds.lo_src[j]);
        } else {
            x_l_red.push(input.x_l[j]);
            x_l_src.push(j);
        }
        if bounds.hi[j].is_finite() {
            x_u_red.push(bounds.hi[j]);
            x_u_src.push(bounds.hi_src[j]);
        } else {
            x_u_red.push(input.x_u[j]);
            x_u_src.push(j);
        }
    }

    EliminationPlan {
        n_full: n,
        m_full: m,
        recovery,
        vars_kept,
        row_kept,
        rows_kept,
        x_l_red,
        x_u_red,
        x_l_src,
        x_u_src,
        steps,
        parent,
        report,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        rows: Vec<Vec<(usize, Number)>>,
        row_const: Vec<Number>,
        g_l: Vec<Number>,
        g_u: Vec<Number>,
        eligible: Vec<bool>,
        x_l: Vec<Number>,
        x_u: Vec<Number>,
        n: usize,
    }

    impl Fixture {
        fn new(n: usize) -> Self {
            Self {
                rows: Vec::new(),
                row_const: Vec::new(),
                g_l: Vec::new(),
                g_u: Vec::new(),
                eligible: Vec::new(),
                x_l: vec![-1e19; n],
                x_u: vec![1e19; n],
                n,
            }
        }
        /// `Σ a_j x_j = b`, linear and eligible.
        fn eq(mut self, entries: &[(usize, Number)], b: Number) -> Self {
            self.rows.push(entries.to_vec());
            self.row_const.push(0.0);
            self.g_l.push(b);
            self.g_u.push(b);
            self.eligible.push(true);
            self
        }
        /// A row the pass must not consume (nonlinear, or an inequality).
        fn opaque(mut self, entries: &[(usize, Number)], lo: Number, hi: Number) -> Self {
            self.rows.push(entries.to_vec());
            self.row_const.push(0.0);
            self.g_l.push(lo);
            self.g_u.push(hi);
            self.eligible.push(false);
            self
        }
        fn bounds(mut self, j: usize, lo: Number, hi: Number) -> Self {
            self.x_l[j] = lo;
            self.x_u[j] = hi;
            self
        }
        fn plan(&self) -> EliminationPlan {
            build_plan(
                &PlanInput {
                    n_vars: self.n,
                    n_rows: self.rows.len(),
                    rows: &self.rows,
                    row_const: &self.row_const,
                    g_l: &self.g_l,
                    g_u: &self.g_u,
                    eligible: &self.eligible,
                    x_l: &self.x_l,
                    x_u: &self.x_u,
                },
                &PlanConfig::default(),
            )
        }
    }

    /// Round-trip: lifting the reduced point must reproduce a full-space
    /// point that satisfies every consumed row.
    fn assert_rows_hold(f: &Fixture, plan: &EliminationPlan, y: &[Number]) {
        let mut x = vec![0.0; f.n];
        plan.lift_x(y, &mut x);
        for (r, entries) in f.rows.iter().enumerate() {
            if plan.row_kept[r] || !f.eligible[r] {
                continue;
            }
            let lhs: Number =
                entries.iter().map(|&(j, a)| a * x[j]).sum::<Number>() + f.row_const[r];
            assert!(
                (lhs - f.g_l[r]).abs() < 1e-9,
                "dropped row {r} violated: {lhs} != {}",
                f.g_l[r]
            );
        }
    }

    #[test]
    fn singleton_row_pins_its_variable() {
        let f = Fixture::new(2).eq(&[(0, 2.0)], 6.0);
        let p = f.plan();
        assert_eq!(p.recovery[0], VarRecovery::Constant(3.0));
        assert_eq!(p.recovery[1], VarRecovery::Kept(0));
        assert_eq!(p.vars_kept, vec![1]);
        assert_eq!(p.rows_kept, Vec::<usize>::new());
        assert_eq!(p.report.n_constant_vars, 1);
        assert_rows_hold(&f, &p, &[7.5]);
    }

    #[test]
    fn free_free_pair_aggregates_with_no_anchor() {
        // The shape the determined-block pipeline cannot reach: both
        // columns free and interior, no bound pinning either one.
        let f = Fixture::new(2).eq(&[(0, 1.0), (1, -1.0)], 0.0);
        let p = f.plan();
        assert_eq!(p.n_reduced_vars(), 1);
        assert_eq!(p.n_reduced_rows(), 0);
        assert_eq!(p.report.n_aggregated_vars, 1);
        assert_rows_hold(&f, &p, &[4.25]);
        let mut x = vec![0.0; 2];
        p.lift_x(&[4.25], &mut x);
        assert!((x[0] - x[1]).abs() < 1e-12);
    }

    #[test]
    fn chains_propagate_regardless_of_row_order() {
        // Written back-to-front, so a single forward sweep cannot see the
        // pin until it has already walked past the rows that need it. Only
        // iterating to a fixed point collapses all four columns.
        let f = Fixture::new(5)
            .eq(&[(2, 1.0), (3, -1.0)], 0.0)
            .eq(&[(1, 1.0), (2, -1.0)], 0.0)
            .eq(&[(0, 1.0), (1, -1.0)], 0.0)
            .eq(&[(3, 2.0)], 8.0);
        let p = f.plan();
        // x0..x3 all pin to 4; x4 is untouched and is the only survivor.
        assert_eq!(p.vars_kept, vec![4]);
        assert_eq!(p.n_reduced_rows(), 0);
        let mut x = vec![0.0; 5];
        p.lift_x(&[9.0], &mut x);
        for (j, v) in x.iter().take(4).enumerate() {
            assert!((v - 4.0).abs() < 1e-12, "x{j} = {v}");
        }
        assert_rows_hold(&f, &p, &[9.0]);
    }

    #[test]
    fn a_fully_determined_model_stands_down() {
        // Every column determined would leave the IPM a zero-variable
        // problem. Hand the square system back whole instead.
        let f = Fixture::new(2)
            .eq(&[(0, 1.0), (1, -1.0)], 0.0)
            .eq(&[(1, 2.0)], 8.0);
        let p = f.plan();
        assert!(p.is_identity());
        assert_eq!(p.n_reduced_vars(), 2);
        assert_eq!(p.n_reduced_rows(), 2);
    }

    #[test]
    fn chain_with_a_free_tail_collapses_to_one_column() {
        // x0 = x1, x1 = x2, x2 = x3 with nothing pinning them: four
        // columns become one.
        let f = Fixture::new(4)
            .eq(&[(0, 1.0), (1, -1.0)], 0.0)
            .eq(&[(1, 1.0), (2, -1.0)], 0.0)
            .eq(&[(2, 1.0), (3, -1.0)], 0.0);
        let p = f.plan();
        assert_eq!(p.n_reduced_vars(), 1);
        assert_eq!(p.n_reduced_rows(), 0);
        let mut x = vec![0.0; 4];
        p.lift_x(&[2.5], &mut x);
        for v in &x {
            assert!((v - 2.5).abs() < 1e-12, "{x:?}");
        }
        assert_rows_hold(&f, &p, &[2.5]);
    }

    #[test]
    fn every_recovery_representative_is_a_survivor() {
        // Path compression must leave no `Affine` pointing at a column that
        // was itself eliminated — the wrapper's `lift_x` reads `out[rep]`
        // directly and would otherwise read an unwritten slot.
        let f = Fixture::new(5)
            .eq(&[(0, 1.0), (1, -2.0)], 1.0)
            .eq(&[(1, 1.0), (2, -3.0)], 2.0)
            .eq(&[(2, 1.0), (3, -4.0)], 3.0);
        let p = f.plan();
        for rec in &p.recovery {
            if let VarRecovery::Affine { rep, coeff, .. } = *rec {
                assert!(
                    matches!(p.recovery[rep], VarRecovery::Kept(_)),
                    "representative {rep} is not a survivor"
                );
                assert!(coeff != 0.0);
            }
        }
        assert_rows_hold(&f, &p, &vec![1.0; p.n_reduced_vars()]);
    }

    #[test]
    fn a_fixed_variable_exposes_a_two_term_row() {
        // x2 is pinned by its own box, so the three-term row becomes a
        // two-term one and x0 folds onto x1.
        let f = Fixture::new(3)
            .eq(&[(0, 1.0), (1, 1.0), (2, 1.0)], 10.0)
            .bounds(2, 4.0, 4.0);
        let p = f.plan();
        assert_eq!(p.recovery[2], VarRecovery::Constant(4.0));
        assert_eq!(p.n_reduced_vars(), 1);
        let mut x = vec![0.0; 3];
        p.lift_x(&[1.5], &mut x);
        assert!((x[0] + x[1] + x[2] - 10.0).abs() < 1e-12, "{x:?}");
    }

    #[test]
    fn bounds_transfer_onto_the_survivor() {
        // x0 = 2·x1 with x0 ∈ [4, 10] pins x1 into [2, 5].
        let f = Fixture::new(2)
            .eq(&[(0, 1.0), (1, -2.0)], 0.0)
            .bounds(0, 4.0, 10.0);
        let p = f.plan();
        assert_eq!(p.vars_kept, vec![1]);
        assert!((p.x_l_red[0] - 2.0).abs() < 1e-12, "{:?}", p.x_l_red);
        assert!((p.x_u_red[0] - 5.0).abs() < 1e-12, "{:?}", p.x_u_red);
    }

    #[test]
    fn negative_coefficient_flips_the_transferred_bounds() {
        // x0 = -x1 with x0 ∈ [1, 3] pins x1 into [-3, -1].
        let f = Fixture::new(2)
            .eq(&[(0, 1.0), (1, 1.0)], 0.0)
            .bounds(0, 1.0, 3.0);
        let p = f.plan();
        assert_eq!(p.vars_kept, vec![1]);
        assert!((p.x_l_red[0] + 3.0).abs() < 1e-12, "{:?}", p.x_l_red);
        assert!((p.x_u_red[0] + 1.0).abs() < 1e-12, "{:?}", p.x_u_red);
    }

    /// A transferred bound records the column that declared it, so postsolve
    /// can hand that column's multiplier back (gh#493).
    #[test]
    fn a_transferred_bound_names_the_column_it_came_from() {
        // x0 = 2·x1 with x0 ∈ [-inf, 1] pins x1 ≤ 0.5; x1's own lower bound
        // survives untouched.
        let f = Fixture::new(2)
            .eq(&[(0, 1.0), (1, -2.0)], 0.0)
            .bounds(0, -1e19, 1.0)
            .bounds(1, -4.0, 1e19);
        let p = f.plan();
        assert_eq!(p.vars_kept, vec![1]);
        assert!((p.x_u_red[0] - 0.5).abs() < 1e-12, "{:?}", p.x_u_red);
        assert_eq!(p.x_u_src, vec![0], "the upper bound is x0's");
        assert_eq!(p.x_l_src, vec![1], "the lower bound is x1's own");
    }

    /// The provenance carries the same flip the interval does: with α < 0 the
    /// survivor's *lower* bound is the eliminated column's *upper* one.
    #[test]
    fn a_negative_coefficient_flips_which_side_the_provenance_lands_on() {
        // x0 = -2·x1 with x0 ∈ [-inf, 1] pins x1 ≥ -0.5.
        let f = Fixture::new(2)
            .eq(&[(0, 1.0), (1, 2.0)], 0.0)
            .bounds(0, -1e19, 1.0);
        let p = f.plan();
        assert_eq!(p.vars_kept, vec![1]);
        assert!((p.x_l_red[0] + 0.5).abs() < 1e-12, "{:?}", p.x_l_red);
        assert_eq!(p.x_l_src, vec![0], "x1's lower bound is x0's upper bound");
        assert_eq!(p.x_u_src, vec![1], "nothing tightened x1 from above");
    }

    /// Provenance follows a chain: a bound that hops twice still names the
    /// column that declared it, and each hop composes the sign flip.
    #[test]
    fn provenance_survives_a_chain_of_transfers() {
        // x0 = -x1 (row 0), then x1 = -0.1·x2 (row 1 — the lopsided
        // coefficients make x2 the pivot, so the transfer chains rather than
        // fanning in). x0 ≤ 1 becomes x1 ≥ -1 becomes x2 ≤ 10, still owned by
        // x0's *upper* bound.
        let f = Fixture::new(3)
            .eq(&[(0, 1.0), (1, 1.0)], 0.0)
            .eq(&[(1, 1.0), (2, 0.1)], 0.0)
            .bounds(0, -1e19, 1.0);
        let p = f.plan();
        assert_eq!(p.vars_kept, vec![2]);
        assert!((p.x_u_red[0] - 10.0).abs() < 1e-12, "{:?}", p.x_u_red);
        assert_eq!(p.x_u_src, vec![0]);
        assert_eq!(p.x_l_src, vec![2], "nothing tightened x2 from below");
        // x0 = (-1)·(-0.1)·x2, so the composed α is positive: two flips put
        // the side back where it started, and 0.1·10 is x0's own bound.
        assert_eq!(
            p.recovery[0],
            VarRecovery::Affine {
                rep: 2,
                coeff: 0.1,
                offset: 0.0
            }
        );
    }

    /// A tie leaves the incumbent alone, which is what keeps the degenerate
    /// both-bounds-active case attributed to the survivor.
    #[test]
    fn a_tied_transfer_leaves_the_provenance_on_the_survivor() {
        // x0 = 2·x1, x0 ≤ 1 and x1 ≤ 0.5 are the same constraint.
        let f = Fixture::new(2)
            .eq(&[(0, 1.0), (1, -2.0)], 0.0)
            .bounds(0, -1e19, 1.0)
            .bounds(1, -1e19, 0.5);
        let p = f.plan();
        assert_eq!(p.vars_kept, vec![1]);
        assert!((p.x_u_red[0] - 0.5).abs() < 1e-12, "{:?}", p.x_u_red);
        assert_eq!(p.x_u_src, vec![1]);
    }

    /// Every reduced bound must be the origin's own bound pulled back through
    /// the recovery map — the identity postsolve's rescaling relies on.
    #[test]
    fn provenance_and_the_recovery_map_agree_on_every_reduced_bound() {
        let f = Fixture::new(4)
            .eq(&[(0, 1.0), (1, 3.0)], 6.0)
            .eq(&[(1, 2.0), (2, -0.5)], 1.0)
            .opaque(&[(2, 1.0), (3, 1.0)], 0.0, 10.0)
            .bounds(0, -2.0, 7.0)
            .bounds(1, -5.0, 5.0)
            .bounds(2, -20.0, 20.0);
        let p = f.plan();
        assert!(!p.is_identity());
        for (red, &kept) in p.vars_kept.iter().enumerate() {
            for (src, red_bound, upper) in [
                (p.x_l_src[red], p.x_l_red[red], false),
                (p.x_u_src[red], p.x_u_red[red], true),
            ] {
                if src == kept || !red_bound.is_finite() || red_bound.abs() >= 1e19 {
                    continue;
                }
                let VarRecovery::Affine { rep, coeff, offset } = p.recovery[src] else {
                    panic!(
                        "provenance {src} is not an affine image: {:?}",
                        p.recovery[src]
                    );
                };
                assert_eq!(rep, kept, "provenance {src} names a different survivor");
                // Which side of the origin's box: its own when α > 0, the
                // other one when α < 0.
                let origin = if upper != (coeff < 0.0) {
                    f.x_u[src]
                } else {
                    f.x_l[src]
                };
                let lifted = coeff * red_bound + offset;
                assert!(
                    (lifted - origin).abs() < 1e-12,
                    "reduced bound {red_bound} lifts to {lifted}, not {src}'s {origin}"
                );
            }
        }
    }

    #[test]
    fn absent_bounds_keep_the_callers_sentinel() {
        let f = Fixture::new(2).eq(&[(0, 1.0), (1, -1.0)], 0.0);
        let p = f.plan();
        assert_eq!(p.x_l_red[0], -1e19);
        assert_eq!(p.x_u_red[0], 1e19);
    }

    #[test]
    fn redundant_row_after_substitution_is_dropped() {
        // x0 = x1 and x1 = x2 make x0 = x2 vacuous.
        let f = Fixture::new(3)
            .eq(&[(0, 1.0), (1, -1.0)], 0.0)
            .eq(&[(1, 1.0), (2, -1.0)], 0.0)
            .eq(&[(0, 1.0), (2, -1.0)], 0.0);
        let p = f.plan();
        assert_eq!(p.n_reduced_vars(), 1);
        assert_eq!(p.n_reduced_rows(), 0);
        assert_eq!(p.report.n_redundant_rows, 1);
    }

    #[test]
    fn contradiction_abandons_the_whole_plan() {
        // x0 = x1 and x0 - x1 = 1 cannot both hold.
        let f = Fixture::new(2)
            .eq(&[(0, 1.0), (1, -1.0)], 0.0)
            .eq(&[(0, 1.0), (1, -1.0)], 1.0);
        let p = f.plan();
        assert!(p.report.infeasible);
        assert!(
            p.is_identity(),
            "a contradictory model must be handed back whole"
        );
        assert_eq!(p.n_reduced_vars(), 2);
        assert_eq!(p.n_reduced_rows(), 2);
    }

    #[test]
    fn a_singleton_outside_its_box_abandons_the_plan() {
        let f = Fixture::new(2).eq(&[(0, 1.0)], 5.0).bounds(0, 0.0, 1.0);
        let p = f.plan();
        assert!(p.report.infeasible);
        assert!(p.is_identity());
    }

    #[test]
    fn a_float_noise_excursion_clamps_instead_of_abandoning() {
        // x0 = 0.1 + 0.2 with x0 ≤ 0.3: infeasible by 5.5e-17, which is
        // binary float noise, not an empty set.
        let f = Fixture::new(2)
            .eq(&[(0, 1.0)], 0.1 + 0.2)
            .bounds(0, 0.0, 0.3);
        let p = f.plan();
        assert!(!p.report.infeasible);
        assert_eq!(p.recovery[0], VarRecovery::Constant(0.3));
    }

    #[test]
    fn an_emptied_survivor_box_abandons_the_plan() {
        // x0 = x1 with disjoint boxes.
        let f = Fixture::new(2)
            .eq(&[(0, 1.0), (1, -1.0)], 0.0)
            .bounds(0, 5.0, 6.0)
            .bounds(1, 1.0, 2.0);
        let p = f.plan();
        assert!(p.report.infeasible);
        assert!(p.is_identity());
    }

    #[test]
    fn ineligible_rows_are_never_consumed() {
        let f = Fixture::new(2).opaque(&[(0, 1.0), (1, -1.0)], 0.0, 0.0);
        let p = f.plan();
        assert!(p.is_identity());
        assert_eq!(p.n_reduced_vars(), 2);
    }

    #[test]
    fn an_inequality_row_is_never_consumed() {
        let mut f = Fixture::new(2);
        f.rows.push(vec![(0, 1.0), (1, -1.0)]);
        f.row_const.push(0.0);
        f.g_l.push(0.0);
        f.g_u.push(1.0);
        f.eligible.push(true);
        let p = f.plan();
        assert!(p.is_identity());
    }

    #[test]
    fn a_one_sided_row_at_the_sentinel_is_not_an_equality() {
        // `g_l = g_u = 1e19` is a one-sided row spelled with the absent
        // sentinel on both ends, not an equality at 1e19 (#396 family).
        let mut f = Fixture::new(2);
        f.rows.push(vec![(0, 1.0), (1, -1.0)]);
        f.row_const.push(0.0);
        f.g_l.push(-1e19);
        f.g_u.push(-1e19);
        f.eligible.push(true);
        let p = f.plan();
        assert!(p.is_identity());
    }

    #[test]
    fn the_row_constant_is_honoured() {
        // g(x) = x0 - x1 + 3, constrained to 0 ⇒ x0 - x1 = -3.
        let mut f = Fixture::new(2);
        f.rows.push(vec![(0, 1.0), (1, -1.0)]);
        f.row_const.push(3.0);
        f.g_l.push(0.0);
        f.g_u.push(0.0);
        f.eligible.push(true);
        let p = f.plan();
        let mut x = vec![0.0; 2];
        p.lift_x(&[2.0], &mut x);
        assert!((x[0] - x[1] + 3.0).abs() < 1e-12, "{x:?}");
    }

    #[test]
    fn three_term_rows_are_left_alone() {
        let f = Fixture::new(3).eq(&[(0, 1.0), (1, 1.0), (2, 1.0)], 1.0);
        let p = f.plan();
        assert!(p.is_identity());
    }

    #[test]
    fn steps_are_recorded_in_application_order_with_live_pivots() {
        let f = Fixture::new(3)
            .eq(&[(0, 2.0), (1, -1.0)], 0.0)
            .eq(&[(1, 3.0), (2, -1.0)], 0.0);
        let p = f.plan();
        assert_eq!(p.steps.len(), 2);
        assert_eq!(p.steps[0].row, 0);
        assert!(p.steps[0].pivot != 0.0);
        assert_eq!(p.steps[1].row, 1);
        // Each step's variable is distinct and never a survivor.
        for s in &p.steps {
            assert!(!matches!(p.recovery[s.var], VarRecovery::Kept(_)));
        }
    }

    #[test]
    fn parent_edges_point_at_later_or_never_eliminated_columns() {
        // The postsolve recovery relies on this: a node's parent is a root
        // at the moment of the merge, so it is eliminated strictly later
        // (or not at all). That is what makes the reverse sweep triangular.
        let f = Fixture::new(4)
            .eq(&[(0, 1.0), (1, -1.0)], 0.0)
            .eq(&[(1, 1.0), (2, -1.0)], 0.0)
            .eq(&[(2, 1.0), (3, -1.0)], 0.0);
        let p = f.plan();
        let mut step_of = [usize::MAX; 4];
        for (t, s) in p.steps.iter().enumerate() {
            step_of[s.var] = t;
        }
        for (i, edge) in p.parent.iter().enumerate() {
            if let Some((parent, _)) = *edge {
                let ti = step_of[i];
                let tp = step_of[parent];
                assert!(ti != usize::MAX);
                assert!(tp == usize::MAX || tp > ti, "{i} -> {parent}");
            }
        }
    }

    #[test]
    fn identity_plan_round_trips() {
        let p = EliminationPlan::identity(3, 2, &[-1.0, -1.0, -1.0], &[1.0, 1.0, 1.0]);
        assert!(p.is_identity());
        let mut x = vec![0.0; 3];
        p.lift_x(&[1.0, 2.0, 3.0], &mut x);
        assert_eq!(x, vec![1.0, 2.0, 3.0]);
        let mut y = vec![0.0; 3];
        p.project_x(&x, &mut y);
        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn a_long_alias_chain_stays_shallow() {
        // 400 aliases with equal coefficients: the union-by-size tie-break
        // must keep the elimination forest logarithmic, because postsolve
        // walks ancestor chains per dropped-row nonzero.
        let mut f = Fixture::new(400);
        for j in 0..399 {
            f = f.eq(&[(j, 1.0), (j + 1, -1.0)], 0.0);
        }
        let p = f.plan();
        assert_eq!(p.n_reduced_vars(), 1);
        let mut depth = 0usize;
        for i in 0..400 {
            let mut d = 0usize;
            let mut cur = i;
            while let Some((parent, _)) = p.parent[cur] {
                cur = parent;
                d += 1;
            }
            depth = depth.max(d);
        }
        assert!(
            depth <= 32,
            "elimination forest depth {depth} is not shallow"
        );
        let mut x = vec![0.0; 400];
        p.lift_x(&[7.0], &mut x);
        for v in &x {
            assert!((v - 7.0).abs() < 1e-12);
        }
    }
}
