//! Postsolve dual recovery for bounds presolve *derived* (issue M24).
//!
//! Phase 1 (activity-based bound tightening) and FBBT derive variable bounds
//! from rows: `x0 = 2·x1, x0 ≥ 4` implies `x1 ≥ 2`. The reduced problem
//! carries that bound, and when it ends up active the solver is free to put
//! the multiplier on it — the reduced problem is degenerate there, since the
//! derived bound is implied by the row and the other variables' bounds. The
//! original problem has no such bound, so a multiplier on it is not a
//! multiplier of the original problem at all: written to a `.sol` it fails
//! `pounce verify --require-optimal` (measured on the example above:
//! `λ = 8`, `z_L(x1) = 20` against the true `λ = -2`, `z_L(x0) = 10`).
//!
//! # The rule
//!
//! At the postsolved primal `x*`, every multiplier the solver put on a
//! *derived* side that the original model does not have active there is
//! **spurious**. Zero it, and repair the stationarity residual that leaves by
//! re-solving for the original problem's multipliers on the active set,
//! restricted to the connected component the residual reaches:
//!
//! ```text
//!   min ‖∇f + Jᵀλ − z_L + z_U‖²   over the active rows and declared bounds
//!   s.t. λ_i ≤ 0 (row active at g_l only),  λ_i ≥ 0 (at g_u only),
//!        λ_i free (equality / both sides),   z ≥ 0
//! ```
//!
//! (Ipopt's convention.) Where the original multipliers are unique — LICQ at
//! `x*` on the active set, the usual case — this *is* the `presolve=no`
//! answer, whatever the reduced solve's split was. Where they are not
//! (degenerate), the solve is a proximal-point iteration started from the
//! reduced solve's own multipliers (spurious entries zeroed), so it returns
//! a valid KKT multiplier set of the original problem that keeps as much of
//! the reduced solve's attribution as stays valid. That is deterministic,
//! and it is the easiest rule to defend: every multiplier in the answer
//! belongs to a constraint the original model declares and that is active,
//! with the right sign, and stationarity holds.
//!
//! What this does **not** move: a multiplier on a *declared* bound that is
//! active at `x*`, even when presolve derived a tighter-or-equal bound on the
//! same side (a degenerate pair: the declared bound is the incumbent and
//! keeps it), and anything the residual does not reach.
//!
//! The component is solved dense. It is capped ([`MAX_UNKNOWNS`],
//! [`MAX_COMPONENT_VARS`]); a component past the cap, or a solve that does
//! not reduce the residual, leaves the duals exactly as the solver reported
//! them and says so through [`Outcome`].
//!
//! Activity is geometric, at `x*`, and so are the multipliers: the solve
//! reads `∇f(x*)`. That is right for a point the solver converged to, where
//! complementarity is `~μ` and an active side's slack is far below the
//! [`on_bound`] tolerance. It is not right for a warm start handed back
//! without a step: the interior-point start pushed `x` about `1e-3` off its
//! bounds, so the derived side reads inactive *and* `∇f` there is not the
//! one the supplied duals balance — widening the activity test alone was
//! measured to leave every re-derived multiplier `O(push)` off (CHAIN:
//! `[6.003, -3.003]` against `[6, -3]`). That case never reaches this
//! module: `PresolveTnlp::finalize_solution` recognises the served warm
//! start coming back and reports the supplied duals (`crate::warm_echo`).

use pounce_common::timing::Deadline;
use pounce_common::types::{Number, lower_bound_present, upper_bound_present};

/// Most multipliers one dense recovery solve will take on.
pub(crate) const MAX_UNKNOWNS: usize = 600;
/// Most variables one connected component may span.
pub(crate) const MAX_COMPONENT_VARS: usize = 5000;
/// Most floating-point operations one recovery may spend, all proximal
/// steps and active-set iterations together (gh#1196). A few seconds of
/// dense work at the worst: about forty full refactorizations at
/// [`MAX_UNKNOWNS`], and thousands of active-set changes once the factor
/// is being updated rather than rebuilt.
pub(crate) const MAX_WORK_FLOPS: f64 = 3e9;

/// Read-only view of the original problem at the postsolved point.
pub(crate) struct KktView<'a> {
    pub x: &'a [Number],
    pub g: &'a [Number],
    pub grad_f: &'a [Number],
    /// Jacobian triplets `(row, col, value)`, 0-based.
    pub jac: &'a [(usize, usize, Number)],
    /// The model's *declared* box and row bounds.
    pub x_l: &'a [Number],
    pub x_u: &'a [Number],
    pub g_l: &'a [Number],
    pub g_u: &'a [Number],
    /// Sides on which presolve's bound is strictly tighter than the
    /// declared one (or the declared one is absent).
    pub derived_l: &'a [bool],
    pub derived_u: &'a [bool],
}

/// What [`reattribute`] did.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Outcome {
    /// Spurious multipliers zeroed (non-zero entries on derived sides the
    /// original model does not have active at `x*`).
    pub n_spurious: usize,
    /// Multipliers the dense solve took on (0 when none was needed).
    pub n_unknowns: usize,
    /// `‖stationarity‖∞` over the component, before (with the spurious
    /// entries in place) and after.
    pub residual_before: Number,
    pub residual_after: Number,
    /// `true` when the duals were left untouched because the component was
    /// past the caps, the solve ran out of its work budget, or it did not
    /// reduce the residual.
    pub rolled_back: bool,
    /// `true` when the roll-back was the work budget or the deadline
    /// running out ([`Budget`], gh#1196).
    pub budget_exhausted: bool,
    /// Which branch rolled back, with what it measured (gh#1202); `None`
    /// when nothing was rolled back.
    pub cause: Option<RollBack>,
}

/// Why [`reattribute`] kept the reduced problem's duals (gh#1202). Each
/// cause wants a different fix, so the warning names it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum RollBack {
    /// The component the residual reaches is past [`MAX_UNKNOWNS`] or
    /// [`MAX_COMPONENT_VARS`]: the counts the breadth-first search had
    /// reached when it stopped (lower bounds on the full component).
    Caps { unknowns: usize, vars: usize },
    /// No active row or declared bound the residual reaches can carry it:
    /// there is nothing to solve for.
    NoMultipliers { vars: usize },
    /// The flop allowance or the solve's deadline ran out ([`Budget`]).
    Budget { unknowns: usize },
    /// The solve finished, but its residual missed the acceptance target.
    Residual {
        unknowns: usize,
        attempted: Number,
        target: Number,
    },
}

impl std::fmt::Display for RollBack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            RollBack::Caps { unknowns, vars } => write!(
                f,
                "component past the dense cap (reached {unknowns} multipliers / {vars} \
                 variables; caps {MAX_UNKNOWNS} / {MAX_COMPONENT_VARS})"
            ),
            RollBack::NoMultipliers { vars } => write!(
                f,
                "no active row or declared bound reaches the residual \
                 ({vars} variables)"
            ),
            RollBack::Budget { unknowns } => write!(
                f,
                "work budget or deadline spent ({unknowns} multipliers, \
                 {MAX_WORK_FLOPS:.0e} flops)"
            ),
            RollBack::Residual {
                unknowns,
                attempted,
                target,
            } => write!(
                f,
                "residual not reduced ({unknowns} multipliers; reached {attempted:.3e}, \
                 needed {target:.3e})"
            ),
        }
    }
}

/// Is `v` on the bound `b`, to the tolerance an interior-point solution
/// reaches a bound it is active at (well above `bound_relax_factor`)?
fn on_bound(v: Number, b: Number) -> bool {
    (v - b).abs() <= 1e-6 * b.abs().max(1.0)
}

/// Sign freedom of one unknown multiplier.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Sign {
    Free,
    NonNeg,
    NonPos,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Unknown {
    Row(usize),
    ZLower(usize),
    ZUpper(usize),
}

/// Stationarity residual `∇f + Jᵀλ − z_L + z_U`, all of it.
fn residual(v: &KktView<'_>, lambda: &[Number], z_l: &[Number], z_u: &[Number]) -> Vec<Number> {
    let mut r = v.grad_f.to_vec();
    for &(i, j, a) in v.jac {
        if let (Some(rj), Some(&l)) = (r.get_mut(j), lambda.get(i)) {
            *rj += a * l;
        }
    }
    for (j, rj) in r.iter_mut().enumerate() {
        *rj += -z_l[j] + z_u[j];
    }
    r
}

/// Zero the multipliers presolve-derived bounds carry against the original
/// model, and recover the original problem's multipliers in their place.
/// See the module docs for the rule. `lambda`, `z_l`, `z_u` are in the
/// original problem's row/variable space, Ipopt's sign convention.
pub(crate) fn reattribute(
    v: &KktView<'_>,
    lambda: &mut [Number],
    z_l: &mut [Number],
    z_u: &mut [Number],
    mut budget: Budget<'_>,
) -> Outcome {
    let n = v.x.len();
    let m = v.g.len();
    let mut out = Outcome::default();
    if z_l.len() != n || z_u.len() != n || lambda.len() != m || v.grad_f.len() != n {
        return out;
    }
    // A spurious entry: a derived side the original does not have active.
    let declared_active_l = |j: usize| lower_bound_present(v.x_l[j]) && on_bound(v.x[j], v.x_l[j]);
    let declared_active_u = |j: usize| upper_bound_present(v.x_u[j]) && on_bound(v.x[j], v.x_u[j]);
    let mut spurious: Vec<usize> = Vec::new();
    for j in 0..n {
        let sl = v.derived_l[j] && z_l[j].is_finite() && z_l[j] != 0.0 && !declared_active_l(j);
        let su = v.derived_u[j] && z_u[j].is_finite() && z_u[j] != 0.0 && !declared_active_u(j);
        if sl || su {
            spurious.push(j);
        }
    }
    if spurious.is_empty() {
        return out;
    }
    let (lam0, zl0, zu0) = (lambda.to_vec(), z_l.to_vec(), z_u.to_vec());
    let before = residual(v, lambda, z_l, z_u);
    for &j in &spurious {
        if v.derived_l[j] && !declared_active_l(j) && z_l[j].is_finite() && z_l[j] != 0.0 {
            z_l[j] = 0.0;
            out.n_spurious += 1;
        }
        if v.derived_u[j] && !declared_active_u(j) && z_u[j].is_finite() && z_u[j] != 0.0 {
            z_u[j] = 0.0;
            out.n_spurious += 1;
        }
    }
    let r = residual(v, lambda, z_l, z_u);
    let scale = v.grad_f.iter().fold(
        1.0_f64,
        |a, &b| if b.is_finite() { a.max(b.abs()) } else { a },
    );
    let seed_tol = 1e-10 * scale;
    let seeds: Vec<usize> = spurious
        .iter()
        .copied()
        .filter(|&j| r[j].abs() > seed_tol)
        .collect();
    if seeds.is_empty() {
        // Only negligible mass was on the derived sides: zeroing it is the
        // whole repair.
        return out;
    }

    // Adjacency, both ways.
    let mut by_col: Vec<Vec<(usize, Number)>> = vec![Vec::new(); n];
    let mut by_row: Vec<Vec<(usize, Number)>> = vec![Vec::new(); m];
    for &(i, j, a) in v.jac {
        if i < m && j < n && a != 0.0 && a.is_finite() {
            by_col[j].push((i, a));
            by_row[i].push((j, a));
        }
    }
    let row_sign = |i: usize| -> Option<Sign> {
        let (gl, gu, gi) = (v.g_l[i], v.g_u[i], v.g[i]);
        if !gi.is_finite() {
            return None;
        }
        let lo = lower_bound_present(gl) && on_bound(gi, gl);
        let hi = upper_bound_present(gu) && on_bound(gi, gu);
        match (lo, hi) {
            (true, true) => Some(Sign::Free),
            (true, false) => Some(Sign::NonPos),
            (false, true) => Some(Sign::NonNeg),
            (false, false) => None,
        }
    };

    // Breadth-first over (variable, active row) incidence from the seeds.
    let mut var_slot = vec![usize::MAX; n];
    let mut vars: Vec<usize> = Vec::new();
    let mut row_seen = vec![false; m];
    let mut unknowns: Vec<(Unknown, Sign)> = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    for &j in &seeds {
        if var_slot[j] == usize::MAX {
            var_slot[j] = vars.len();
            vars.push(j);
            queue.push_back(j);
        }
    }
    let mut over_cap = false;
    while let Some(j) = queue.pop_front() {
        if declared_active_l(j) {
            unknowns.push((Unknown::ZLower(j), Sign::NonNeg));
        }
        if declared_active_u(j) {
            unknowns.push((Unknown::ZUpper(j), Sign::NonNeg));
        }
        for &(i, _) in &by_col[j] {
            if row_seen[i] {
                continue;
            }
            row_seen[i] = true;
            let Some(sign) = row_sign(i) else { continue };
            unknowns.push((Unknown::Row(i), sign));
            for &(k, _) in &by_row[i] {
                if var_slot[k] == usize::MAX {
                    var_slot[k] = vars.len();
                    vars.push(k);
                    queue.push_back(k);
                }
            }
        }
        if unknowns.len() > MAX_UNKNOWNS || vars.len() > MAX_COMPONENT_VARS {
            over_cap = true;
            break;
        }
    }
    out.residual_before = vars.iter().fold(0.0_f64, |a, &j| a.max(before[j].abs()));
    let restore = |lambda: &mut [Number], z_l: &mut [Number], z_u: &mut [Number]| {
        lambda.copy_from_slice(&lam0);
        z_l.copy_from_slice(&zl0);
        z_u.copy_from_slice(&zu0);
    };
    if over_cap || unknowns.is_empty() {
        restore(lambda, z_l, z_u);
        out.rolled_back = true;
        out.cause = Some(if over_cap {
            RollBack::Caps {
                unknowns: unknowns.len(),
                vars: vars.len(),
            }
        } else {
            RollBack::NoMultipliers { vars: vars.len() }
        });
        out.residual_after = out.residual_before;
        out.n_spurious = 0;
        return out;
    }
    out.n_unknowns = unknowns.len();

    // Dense A (|vars| × k) in the transformed variable `w = s·y` so every
    // sign constraint reads `w ≥ 0`, and b = r − A·w0 so the component's
    // residual is `A·w + b`.
    let nv = vars.len();
    let k = unknowns.len();
    let mut a = vec![0.0; nv * k];
    let mut w0 = vec![0.0; k];
    let mut constrained = vec![false; k];
    for (c, &(u, sign)) in unknowns.iter().enumerate() {
        let s = if sign == Sign::NonPos { -1.0 } else { 1.0 };
        constrained[c] = sign != Sign::Free;
        match u {
            Unknown::Row(i) => {
                for &(jj, aij) in &by_row[i] {
                    let slot = var_slot[jj];
                    if slot != usize::MAX {
                        a[slot * k + c] += s * aij;
                    }
                }
                w0[c] = s * lambda[i];
            }
            Unknown::ZLower(j) => {
                a[var_slot[j] * k + c] = -1.0;
                w0[c] = z_l[j];
            }
            Unknown::ZUpper(j) => {
                a[var_slot[j] * k + c] = 1.0;
                w0[c] = z_u[j];
            }
        }
        if constrained[c] && !(w0[c] >= 0.0) {
            w0[c] = 0.0;
        }
    }
    let mut b: Vec<Number> = vars.iter().map(|&j| r[j]).collect();
    // r was computed with the current (pre-projection) values; express it
    // against the projected start.
    for (c, &(u, sign)) in unknowns.iter().enumerate() {
        let s = if sign == Sign::NonPos { -1.0 } else { 1.0 };
        let current = match u {
            Unknown::Row(i) => s * lambda[i],
            Unknown::ZLower(j) => z_l[j],
            Unknown::ZUpper(j) => z_u[j],
        };
        let current = if current.is_finite() { current } else { 0.0 };
        for (row, bv) in b.iter_mut().enumerate() {
            *bv -= a[row * k + c] * current;
        }
    }
    // Accept the recovery only when it actually closes stationarity: to the
    // solver's own residual, or to a level no tolerance would notice.
    let target = (10.0 * out.residual_before).max(1e-8 * scale);
    let Some(w) = solve_sign_constrained_ls(&a, &b, nv, k, &constrained, &w0, target, &mut budget)
    else {
        restore(lambda, z_l, z_u);
        out.rolled_back = true;
        out.budget_exhausted = true;
        out.cause = Some(RollBack::Budget { unknowns: k });
        out.residual_after = out.residual_before;
        out.n_spurious = 0;
        out.n_unknowns = 0;
        return out;
    };
    let res_after = (0..nv)
        .map(|row| {
            let mut acc = b[row];
            for c in 0..k {
                acc += a[row * k + c] * w[c];
            }
            acc.abs()
        })
        .fold(0.0_f64, Number::max);
    if !(res_after <= target) {
        restore(lambda, z_l, z_u);
        out.rolled_back = true;
        out.cause = Some(RollBack::Residual {
            unknowns: k,
            attempted: res_after,
            target,
        });
        out.residual_after = out.residual_before;
        out.n_spurious = 0;
        out.n_unknowns = 0;
        return out;
    }
    for (c, &(u, sign)) in unknowns.iter().enumerate() {
        let s = if sign == Sign::NonPos { -1.0 } else { 1.0 };
        match u {
            Unknown::Row(i) => lambda[i] = s * w[c],
            Unknown::ZLower(j) => z_l[j] = w[c],
            Unknown::ZUpper(j) => z_u[j] = w[c],
        }
    }
    out.residual_after = res_after;
    out
}

/// What one recovery may spend: a flop allowance and, optionally, the
/// solve's own deadline (gh#1196).
///
/// The unknowns cap ([`MAX_UNKNOWNS`]) bounds the size of one dense solve,
/// not how many of them run: the proximal-point loop takes up to 200 steps,
/// and each is an active-set QP whose iteration count grows with the number
/// of sign changes. Uncapped, Vanderbei `qpnstair` (`presolve=yes`) spent
/// 838 s here after the interior-point solve had converged, past a 10 s
/// `max_wall_time`. Spending the budget is the same outcome as a component
/// past the caps: the reduced solve's duals are kept, `rolled_back` is set.
pub(crate) struct Budget<'a> {
    flops: f64,
    deadline: Option<&'a Deadline>,
}

impl Default for Budget<'_> {
    fn default() -> Self {
        Self::new(MAX_WORK_FLOPS, None)
    }
}

impl<'a> Budget<'a> {
    pub(crate) fn new(flops: f64, deadline: Option<&'a Deadline>) -> Self {
        Self { flops, deadline }
    }

    /// Charge `flops` of work about to be done; `false` once the allowance
    /// or the deadline is spent. Called at the granularity of one
    /// active-set iteration, so the overshoot is one such iteration.
    fn charge(&mut self, flops: f64) -> bool {
        self.flops -= flops;
        self.flops >= 0.0 && self.deadline.is_none_or(|d| d.exceeded().is_none())
    }
}

/// Proximal steps a residual already under the caller's acceptance target
/// gets to halve before the iteration counts as stalled.
const STALL_STEPS: usize = 3;

/// `min ½‖A·w + b‖²` subject to `w_c ≥ 0` where `constrained[c]`, by a
/// proximal-point iteration `w⁺ = argmin ½‖A·w + b‖² + ½μ‖w − w_k‖²` started
/// at `w0`. Each step is a strictly convex bound-constrained QP, solved by a
/// primal active-set method. On a consistent system with a unique solution
/// this converges to it; on a degenerate one, to a solution near `w0`.
/// `None` when `budget` runs out first.
///
/// It returns the iterate with the smallest residual `‖A·w + b‖∞`, as soon
/// as that is down to the rounding floor `100·ε·‖A‖∞·‖w‖∞`, or once it is
/// under `accept` (the caller's acceptance target) and has not halved in
/// [`STALL_STEPS`] steps — or when the step itself vanishes.
/// The step test alone is not reachable in floating point once the
/// multipliers are large: on Vanderbei `qpnstair` (`‖w‖∞ ≈ 9e4`) the
/// residual contracts ×0.44 per step to a floor of `2e-9` by step 27, while
/// the step stalls near `4e-7` — `4e-12` relative against the `1e-15` the
/// test asks for. The remaining 170 steps iterated rounding noise, and the
/// active-set QP cycled on near-zero entries to its iteration cap, `1e10`
/// flops a step: the whole [`MAX_WORK_FLOPS`] allowance, three times over,
/// on a recovery that had met its acceptance target at step 15.
#[allow(clippy::too_many_arguments)]
fn solve_sign_constrained_ls(
    a: &[Number],
    b: &[Number],
    nrows: usize,
    k: usize,
    constrained: &[bool],
    w0: &[Number],
    accept: Number,
    budget: &mut Budget<'_>,
) -> Option<Vec<Number>> {
    // Normal matrix and AᵀB.
    let nnz = a.iter().filter(|&&x| x != 0.0).count() as f64;
    if !budget.charge(2.0 * nnz * k as f64) {
        return None;
    }
    let mut ata = vec![0.0; k * k];
    let mut atb = vec![0.0; k];
    for row in 0..nrows {
        let ar = &a[row * k..(row + 1) * k];
        for p in 0..k {
            if ar[p] == 0.0 {
                continue;
            }
            atb[p] += ar[p] * b[row];
            for q in 0..k {
                ata[p * k + q] += ar[p] * ar[q];
            }
        }
    }
    let max_diag = (0..k).fold(0.0_f64, |m, p| m.max(ata[p * k + p]));
    let mu = 1e-6 * max_diag.max(1e-300);
    let mut h = ata;
    for p in 0..k {
        h[p * k + p] += mu;
    }
    let norm_a = (0..nrows)
        .map(|row| {
            a[row * k..(row + 1) * k]
                .iter()
                .map(|x| x.abs())
                .sum::<Number>()
        })
        .fold(0.0_f64, Number::max);
    let floor = |w: &[Number]| -> Number {
        100.0 * Number::EPSILON * norm_a * w.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
    };
    let resid = |w: &[Number]| -> Number {
        (0..nrows)
            .map(|row| {
                let ar = &a[row * k..(row + 1) * k];
                (b[row] + ar.iter().zip(w).map(|(x, y)| x * y).sum::<Number>()).abs()
            })
            .fold(0.0_f64, Number::max)
    };
    // One factor of the free block, carried across the proximal steps (H
    // does not change between them, and neither, near convergence, does the
    // active set) and updated, not rebuilt, as the active set moves.
    let mut fac = FreeFactor::new(k);
    let mut w = w0.to_vec();
    let mut best = (resid(&w), w.clone());
    // Residuals of the last few steps, newest last.
    let mut recent: Vec<Number> = Vec::with_capacity(STALL_STEPS + 1);
    for _ in 0..200 {
        // c = Aᵀb − μ w_k
        let c: Vec<Number> = (0..k).map(|p| atb[p] - mu * w[p]).collect();
        let next = box_qp(&h, &c, k, constrained, &w, &mut fac, budget)?;
        let step = next
            .iter()
            .zip(&w)
            .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()));
        let size = next.iter().fold(1.0_f64, |m, p| m.max(p.abs()));
        w = next;
        if !budget.charge(2.0 * (nrows * k) as f64) {
            return None;
        }
        let r = resid(&w);
        if r < best.0 {
            best = (r, w.clone());
        }
        if recent.len() > STALL_STEPS {
            recent.remove(0);
        }
        recent.push(r);
        let stalled = recent.len() > STALL_STEPS && r > 0.5 * recent[0];
        let at_floor = best.0 <= floor(&best.1);
        if step <= 1e-15 * size || at_floor || (best.0 <= accept && stalled) {
            break;
        }
    }
    Some(best.1)
}

/// Cholesky factor `L·Lᵀ = H_FF` of the free block of a `k×k` SPD `H`, for
/// an ordered free set `idx`, kept up to date as indices leave (a column
/// rotation sweep, `O(nf²)`) and join (one appended row, `O(nf²)`), instead
/// of refactored from scratch (`O(nf³)`) on every active-set change.
struct FreeFactor {
    k: usize,
    /// Free indices, in factor order.
    idx: Vec<usize>,
    /// `pos[p]` = position of `p` in `idx`, or `usize::MAX`.
    pos: Vec<usize>,
    /// Row-major lower triangle, stride `k`.
    l: Vec<Number>,
    /// Up/down-dates since the last full factorization.
    updates: usize,
    /// Whether `idx`/`l` describe a valid factor at all.
    valid: bool,
}

impl FreeFactor {
    fn new(k: usize) -> Self {
        Self {
            k,
            idx: Vec::new(),
            pos: vec![usize::MAX; k],
            l: vec![0.0; k * k],
            updates: 0,
            valid: false,
        }
    }

    /// Factor `H_FF` for the free set `free` from scratch.
    fn refactor(&mut self, h: &[Number], free: &[usize], budget: &mut Budget<'_>) -> Option<bool> {
        let (k, nf) = (self.k, free.len());
        if !budget.charge((nf as f64).powi(3) / 3.0 + nf as f64) {
            return None;
        }
        self.idx.clear();
        self.pos.iter_mut().for_each(|p| *p = usize::MAX);
        self.valid = false;
        self.updates = 0;
        for &p in free {
            self.pos[p] = self.idx.len();
            self.idx.push(p);
        }
        let l = &mut self.l;
        for j in 0..nf {
            let mut d = h[free[j] * k + free[j]];
            for q in 0..j {
                d -= l[j * k + q] * l[j * k + q];
            }
            if !(d > 0.0) {
                return Some(false);
            }
            let d = d.sqrt();
            l[j * k + j] = d;
            for i in (j + 1)..nf {
                let mut s = h[free[i] * k + free[j]];
                for q in 0..j {
                    s -= l[i * k + q] * l[j * k + q];
                }
                l[i * k + j] = s / d;
            }
        }
        self.valid = true;
        Some(true)
    }

    /// Drop index `p` from the free set.
    fn remove(&mut self, p: usize) {
        let (k, nf) = (self.k, self.idx.len());
        let r = self.pos[p];
        let l = &mut self.l;
        // Shift rows r+1.. up by one; row t (new) has entries in 0..=t+1.
        for t in r..nf - 1 {
            for q in 0..=(t + 1) {
                l[t * k + q] = l[(t + 1) * k + q];
            }
        }
        // Restore lower-triangularity: rotate columns (t, t+1) to zero the
        // superdiagonal entry of row t. Column rotations keep L·Lᵀ.
        for t in r..nf - 1 {
            let (x, y) = (l[t * k + t], l[t * k + t + 1]);
            let rho = x.hypot(y);
            if rho == 0.0 {
                continue;
            }
            let (c, s) = (x / rho, y / rho);
            for u in t..nf - 1 {
                let (a, b) = (l[u * k + t], l[u * k + t + 1]);
                l[u * k + t] = c * a + s * b;
                l[u * k + t + 1] = -s * a + c * b;
            }
            l[t * k + t + 1] = 0.0;
        }
        self.idx.remove(r);
        self.pos[p] = usize::MAX;
        for (i, &q) in self.idx.iter().enumerate().skip(r) {
            self.pos[q] = i;
        }
        self.updates += 1;
    }

    /// Append index `p` to the free set; `false` when the new pivot is not
    /// numerically positive (the caller refactors).
    fn append(&mut self, h: &[Number], p: usize) -> bool {
        let (k, nf) = (self.k, self.idx.len());
        let l = &mut self.l;
        // Solve L·x = H[idx, p] into row nf.
        for i in 0..nf {
            let mut s = h[self.idx[i] * k + p];
            for q in 0..i {
                s -= l[i * k + q] * l[nf * k + q];
            }
            l[nf * k + i] = s / l[i * k + i];
        }
        let hpp = h[p * k + p];
        let mut d = hpp;
        for q in 0..nf {
            d -= l[nf * k + q] * l[nf * k + q];
        }
        if !(d > 1e-12 * hpp.abs()) {
            return false;
        }
        l[nf * k + nf] = d.sqrt();
        self.pos[p] = nf;
        self.idx.push(p);
        self.updates += 1;
        true
    }

    /// Bring the factor to the free set `!fixed`, by updates when that is
    /// cheaper and safe, else from scratch. `Some(false)`: `H_FF` is not
    /// numerically SPD.
    fn sync(&mut self, h: &[Number], fixed: &[bool], budget: &mut Budget<'_>) -> Option<bool> {
        let free: Vec<usize> = (0..self.k).filter(|&p| !fixed[p]).collect();
        let nf = free.len() as f64;
        if self.valid && self.updates < 64 {
            let leave: Vec<usize> = self.idx.iter().copied().filter(|&p| fixed[p]).collect();
            let join: Vec<usize> = free
                .iter()
                .copied()
                .filter(|&p| self.pos[p] == usize::MAX)
                .collect();
            let changes = (leave.len() + join.len()) as f64;
            if changes == 0.0 {
                return Some(true);
            }
            if 2.0 * changes * nf < nf / 3.0 * nf {
                if !budget.charge(2.0 * changes * nf * nf) {
                    return None;
                }
                for &p in &leave {
                    self.remove(p);
                }
                if join.iter().all(|&p| self.append(h, p)) {
                    return Some(true);
                }
            }
        }
        self.refactor(h, &free, budget)
    }

    /// Solve `H_FF·x = rhs` (in factor order) in place.
    fn solve(&self, rhs: &mut [Number]) {
        let (k, nf) = (self.k, self.idx.len());
        let l = &self.l;
        for i in 0..nf {
            let mut s = rhs[i];
            for q in 0..i {
                s -= l[i * k + q] * rhs[q];
            }
            rhs[i] = s / l[i * k + i];
        }
        for i in (0..nf).rev() {
            let mut s = rhs[i];
            for q in (i + 1)..nf {
                s -= l[q * k + i] * rhs[q];
            }
            rhs[i] = s / l[i * k + i];
        }
    }
}

/// `min ½ wᵀHw + cᵀw` s.t. `w_c ≥ 0` where `constrained[c]`, `H` SPD,
/// started from the feasible `start`. Primal active-set method. `None` when
/// `budget` runs out first.
fn box_qp(
    h: &[Number],
    c: &[Number],
    k: usize,
    constrained: &[bool],
    start: &[Number],
    fac: &mut FreeFactor,
    budget: &mut Budget<'_>,
) -> Option<Vec<Number>> {
    let mut w: Vec<Number> = start.to_vec();
    let mut fixed: Vec<bool> = (0..k).map(|p| constrained[p] && w[p] <= 0.0).collect();
    for p in 0..k {
        if fixed[p] {
            w[p] = 0.0;
        }
    }
    for _ in 0..(10 * k + 50) {
        // Solve H_FF p_F = −c_F (fixed entries are 0).
        if !fac.sync(h, &fixed, budget)? {
            return Some(w);
        }
        let nf = fac.idx.len();
        if !budget.charge(2.0 * (nf * nf + k * k) as f64) {
            return None;
        }
        let mut cand = vec![0.0; k];
        if nf > 0 {
            let mut rhs: Vec<Number> = fac.idx.iter().map(|&p| -c[p]).collect();
            fac.solve(&mut rhs);
            for (r, &p) in fac.idx.iter().enumerate() {
                cand[p] = rhs[r];
            }
        }
        let free = &fac.idx;
        let infeasible = free.iter().any(|&p| constrained[p] && cand[p] < 0.0);
        if !infeasible {
            w = cand;
            // Multipliers of the fixed set: the gradient there.
            let mut worst: Option<(usize, Number)> = None;
            for p in 0..k {
                if !fixed[p] {
                    continue;
                }
                let mut gp = c[p];
                for q in 0..k {
                    gp += h[p * k + q] * w[q];
                }
                let tol = 1e-14 * (h[p * k + p].abs().max(1e-300)).sqrt() * (1.0 + c[p].abs());
                if gp < -tol && worst.is_none_or(|(_, g)| gp < g) {
                    worst = Some((p, gp));
                }
            }
            match worst {
                Some((p, _)) => fixed[p] = false,
                None => return Some(w),
            }
        } else {
            // Step toward the candidate as far as feasibility allows.
            let mut alpha = 1.0_f64;
            let mut block = None;
            for &p in free {
                if constrained[p] && cand[p] < 0.0 {
                    let t = w[p] / (w[p] - cand[p]);
                    if t < alpha {
                        alpha = t;
                        block = Some(p);
                    }
                }
            }
            for &p in free {
                w[p] += alpha * (cand[p] - w[p]);
            }
            if let Some(p) = block {
                fixed[p] = true;
                w[p] = 0.0;
            }
            for &p in free {
                if constrained[p] && w[p] <= 0.0 {
                    fixed[p] = true;
                    w[p] = 0.0;
                }
            }
        }
    }
    Some(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `min x0² + x1²  s.t.  x0 − 2·x1 = 0,  x0 ≥ 4`, at `x* = (4, 2)`, with
    /// presolve's derived `x1 ≥ 2` carrying the reduced solve's whole
    /// multiplier (`λ = −8`, `z_L(x1) = 20`): the original problem's unique
    /// multipliers are `λ = 2`, `z_L(x0) = 10`.
    #[test]
    fn a_derived_bound_multiplier_goes_back_to_the_row_and_the_declared_bound() {
        let jac = [(0, 0, 1.0), (0, 1, -2.0)];
        let v = KktView {
            x: &[4.0, 2.0],
            g: &[0.0],
            grad_f: &[8.0, 4.0],
            jac: &jac,
            x_l: &[4.0, -1e19],
            x_u: &[1e19, 1e19],
            g_l: &[0.0],
            g_u: &[0.0],
            derived_l: &[false, true],
            derived_u: &[false, false],
        };
        let (mut lam, mut zl, mut zu) = (vec![-8.0], vec![0.0, 20.0], vec![0.0, 0.0]);
        let o = reattribute(&v, &mut lam, &mut zl, &mut zu, Budget::default());
        assert!(!o.rolled_back, "{o:?}");
        assert_eq!(o.n_spurious, 1);
        assert!((lam[0] - 2.0).abs() < 1e-10, "λ = {lam:?}");
        assert!((zl[0] - 10.0).abs() < 1e-10, "z_L = {zl:?}");
        assert_eq!(zl[1], 0.0);
        let r = residual(&v, &lam, &zl, &zu);
        assert!(r.iter().all(|x| x.abs() < 1e-10), "{r:?}");
    }

    /// A multiplier on a declared bound the derived one ties with stays put
    /// (the incumbent keeps a degenerate pair), and nothing else moves.
    #[test]
    fn a_declared_active_bound_keeps_its_multiplier() {
        let jac = [(0, 0, 1.0), (0, 1, -2.0)];
        let v = KktView {
            x: &[4.0, 2.0],
            g: &[0.0],
            grad_f: &[8.0, 4.0],
            jac: &jac,
            x_l: &[4.0, 2.0],
            x_u: &[1e19, 1e19],
            g_l: &[0.0],
            g_u: &[0.0],
            derived_l: &[false, true],
            derived_u: &[false, false],
        };
        let (mut lam, mut zl, mut zu) = (vec![-8.0], vec![0.0, 20.0], vec![0.0, 0.0]);
        let o = reattribute(&v, &mut lam, &mut zl, &mut zu, Budget::default());
        assert_eq!(o, Outcome::default());
        assert_eq!((lam, zl), (vec![-8.0], vec![0.0, 20.0]));
    }

    /// Sign constraints bind: an inequality active at its lower side only
    /// takes `λ ≤ 0`, and a degenerate pair of rows resolves to a valid set.
    #[test]
    fn inequality_signs_are_respected() {
        // min (x1 − 3)²  s.t. C0: x0 = 0.5, C1: x0 + x1 ≤ 2. x* = (0.5, 1.5).
        // ∇f = (0, −3). Original: λ1 = 3 (≥ 0, upper-active), λ0 = −3.
        let jac = [(0, 0, 1.0), (1, 0, 1.0), (1, 1, 1.0)];
        let v = KktView {
            x: &[0.5, 1.5],
            g: &[0.5, 2.0],
            grad_f: &[0.0, -3.0],
            jac: &jac,
            x_l: &[-1e19, -1e19],
            x_u: &[1e19, 1e19],
            g_l: &[0.5, -1e19],
            g_u: &[0.5, 2.0],
            derived_l: &[true, false],
            derived_u: &[true, true],
        };
        let (mut lam, mut zl, mut zu) = (vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 3.0]);
        let o = reattribute(&v, &mut lam, &mut zl, &mut zu, Budget::default());
        assert!(!o.rolled_back, "{o:?}");
        assert!(
            (lam[0] + 3.0).abs() < 1e-10 && (lam[1] - 3.0).abs() < 1e-10,
            "{lam:?}"
        );
        assert_eq!((zl, zu), (vec![0.0, 0.0], vec![0.0, 0.0]));
    }

    /// Deterministic pseudo-random numbers in `[-1, 1)`.
    fn lcg(seed: &mut u64) -> Number {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as Number / (1u64 << 53) as Number) * 2.0 - 1.0
    }

    /// gh#1196: a component near [`MAX_UNKNOWNS`] whose sign-constrained
    /// solve needs many active-set changes — the branch `qpnstair` reaches.
    /// `nrow` inequality rows, all active at their upper side (`λ ≥ 0`), over
    /// `nvar` variables, every row touching the seed variable 0 (so one BFS
    /// takes them all in). The reduced solve's `λ` starts strictly positive
    /// everywhere and `∇f` is random, so most of the `λ ≥ 0` have to be
    /// driven onto the bound one blocking step at a time.
    struct Stair {
        nvar: usize,
        jac: Vec<(usize, usize, Number)>,
        grad_f: Vec<Number>,
        x: Vec<Number>,
        g: Vec<Number>,
        x_l: Vec<Number>,
        x_u: Vec<Number>,
        g_l: Vec<Number>,
        g_u: Vec<Number>,
        derived_l: Vec<bool>,
        derived_u: Vec<bool>,
    }

    impl Stair {
        fn new(nvar: usize, nrow: usize) -> Self {
            let mut seed = 1196;
            let mut jac = Vec::new();
            for i in 0..nrow {
                jac.push((i, 0, 1.0 + 0.5 * lcg(&mut seed)));
                for t in 0..4 {
                    let j = 1 + (i * 7 + t * 131 + (i * i) % 17) % (nvar - 1);
                    jac.push((i, j, lcg(&mut seed)));
                }
            }
            let grad_f = (0..nvar).map(|_| 5.0 * lcg(&mut seed)).collect();
            let mut derived_l = vec![false; nvar];
            derived_l[0] = true;
            Self {
                nvar,
                jac,
                grad_f,
                x: vec![0.0; nvar],
                g: vec![1.0; nrow],
                x_l: vec![-1e19; nvar],
                x_u: vec![1e19; nvar],
                g_l: vec![-1e19; nrow],
                g_u: vec![1.0; nrow],
                derived_l,
                derived_u: vec![false; nvar],
            }
        }

        fn view(&self) -> KktView<'_> {
            KktView {
                x: &self.x,
                g: &self.g,
                grad_f: &self.grad_f,
                jac: &self.jac,
                x_l: &self.x_l,
                x_u: &self.x_u,
                g_l: &self.g_l,
                g_u: &self.g_u,
                derived_l: &self.derived_l,
                derived_u: &self.derived_u,
            }
        }

        /// Reduced-solve duals: every `λ` positive, and the spurious mass
        /// on the derived `x0 ≥ ·`.
        fn duals(&self) -> (Vec<Number>, Vec<Number>, Vec<Number>) {
            let mut zl = vec![0.0; self.nvar];
            zl[0] = 50.0;
            (vec![1.0; self.g.len()], zl, vec![0.0; self.nvar])
        }
    }

    /// The slow branch completes inside the default budget — the factor is
    /// updated, not rebuilt, per active-set change — and its answer is a
    /// valid multiplier set: right signs, stationarity no worse than the
    /// acceptance target.
    #[test]
    fn a_component_near_the_cap_with_many_active_set_changes_is_affordable() {
        let p = Stair::new(400, MAX_UNKNOWNS - 10);
        let (mut lam, mut zl, mut zu) = p.duals();
        let o = reattribute(&p.view(), &mut lam, &mut zl, &mut zu, Budget::default());
        assert!(!o.budget_exhausted, "{o:?}");
        assert_eq!(o.n_unknowns, MAX_UNKNOWNS - 10, "{o:?}");
        if !o.rolled_back {
            assert!(lam.iter().all(|&l| l >= 0.0), "a λ ≥ 0 went negative");
            assert_eq!(zl[0], 0.0);
            let zeroed = lam.iter().filter(|&&l| l == 0.0).count();
            assert!(
                zeroed > 50,
                "only {zeroed} λ reached their bound: not the slow branch"
            );
        }
    }

    /// Out of budget, the recovery stops and keeps the reduced solve's
    /// duals exactly, and says why.
    #[test]
    fn a_spent_work_budget_rolls_back_to_the_reduced_duals() {
        let p = Stair::new(400, MAX_UNKNOWNS - 10);
        let (lam0, zl0, zu0) = p.duals();
        let (mut lam, mut zl, mut zu) = p.duals();
        let o = reattribute(
            &p.view(),
            &mut lam,
            &mut zl,
            &mut zu,
            Budget::new(1e6, None),
        );
        assert!(o.rolled_back && o.budget_exhausted, "{o:?}");
        assert!(matches!(o.cause, Some(RollBack::Budget { .. })), "{o:?}");
        assert_eq!(o.n_spurious, 0);
        assert_eq!((lam, zl, zu), (lam0, zl0, zu0));
    }

    /// gh#1202: a roll-back names its branch, with what it measured — each
    /// wants a different fix, and the warning used to list all three.
    #[test]
    fn a_component_past_the_cap_names_the_cap() {
        let p = Stair::new(400, MAX_UNKNOWNS + 50);
        let (lam0, zl0, zu0) = p.duals();
        let (mut lam, mut zl, mut zu) = p.duals();
        let o = reattribute(&p.view(), &mut lam, &mut zl, &mut zu, Budget::default());
        assert!(o.rolled_back && !o.budget_exhausted, "{o:?}");
        match o.cause {
            Some(RollBack::Caps { unknowns, vars }) => {
                assert!(unknowns > MAX_UNKNOWNS, "{o:?}");
                assert!(vars > 1, "{o:?}");
            }
            c => panic!("expected a cap roll-back, got {c:?}"),
        }
        assert!(o.cause.unwrap().to_string().contains("dense cap"));
        assert_eq!((lam, zl, zu), (lam0, zl0, zu0));
    }

    /// One variable, `∇f = 3`, the whole multiplier on a derived `x0 ≥ 0`
    /// the model does not declare. `row` adds `x0 ≤ 1`, active: it can only
    /// take `λ ≥ 0`, which pushes the residual the wrong way.
    fn single(row: bool) -> (Outcome, RollBack) {
        let jac: Vec<(usize, usize, Number)> = if row { vec![(0, 0, 1.0)] } else { vec![] };
        let m = usize::from(row);
        let v = KktView {
            x: &[1.0],
            g: &[1.0][..m],
            grad_f: &[3.0],
            jac: &jac,
            x_l: &[-1e19],
            x_u: &[1e19],
            g_l: &[-1e19][..m],
            g_u: &[1.0][..m],
            derived_l: &[true],
            derived_u: &[false],
        };
        let mut lam = vec![0.0; m];
        let (mut zl, mut zu) = (vec![3.0], vec![0.0]);
        let o = reattribute(&v, &mut lam, &mut zl, &mut zu, Budget::default());
        assert!(o.rolled_back, "{o:?}");
        assert_eq!(zl, vec![3.0], "a roll-back must keep the reduced duals");
        let c = o.cause.expect("a roll-back names its cause");
        (o, c)
    }

    #[test]
    fn a_residual_nothing_can_carry_names_that() {
        let (_, c) = single(false);
        assert_eq!(c, RollBack::NoMultipliers { vars: 1 });
    }

    #[test]
    fn a_residual_the_signs_forbid_closing_names_the_residual() {
        let (o, c) = single(true);
        assert!(!o.budget_exhausted);
        match c {
            RollBack::Residual {
                unknowns: 1,
                attempted,
                target,
            } => assert!(
                attempted > target && (attempted - 3.0).abs() < 1e-9,
                "{c:?}"
            ),
            c => panic!("expected a residual roll-back, got {c:?}"),
        }
    }

    /// The solve's deadline stops the recovery too: a solve that is already
    /// out of time spends none of it in postsolve.
    #[test]
    fn a_crossed_deadline_rolls_back_to_the_reduced_duals() {
        let p = Stair::new(400, MAX_UNKNOWNS - 10);
        let (lam0, zl0, zu0) = p.duals();
        let (mut lam, mut zl, mut zu) = p.duals();
        let d = Deadline::new(0.0, 0.0);
        let o = reattribute(
            &p.view(),
            &mut lam,
            &mut zl,
            &mut zu,
            Budget::new(MAX_WORK_FLOPS, Some(&d)),
        );
        assert!(o.rolled_back && o.budget_exhausted, "{o:?}");
        assert_eq!((lam, zl, zu), (lam0, zl0, zu0));
    }

    /// The up/down-dated factor equals a fresh factorization of the same
    /// free block, through a mix of removals and appends.
    #[test]
    fn factor_updates_match_a_fresh_factorization() {
        let k = 12;
        let mut seed = 7;
        let a: Vec<Number> = (0..k * k).map(|_| lcg(&mut seed)).collect();
        let mut h = vec![0.0; k * k];
        for p in 0..k {
            for q in 0..k {
                for r in 0..k {
                    h[p * k + q] += a[r * k + p] * a[r * k + q];
                }
            }
            h[p * k + p] += 0.1;
        }
        let mut budget = Budget::default();
        let mut fac = FreeFactor::new(k);
        let all: Vec<usize> = (0..k).collect();
        assert_eq!(fac.refactor(&h, &all, &mut budget), Some(true));
        for &p in &[3, 0, 11, 6] {
            fac.remove(p);
        }
        for &p in &[11, 0] {
            assert!(fac.append(&h, p));
        }
        fac.remove(5);
        let order = fac.idx.clone();
        let nf = order.len();
        // L·Lᵀ reproduces H on the free set, in factor order.
        for r in 0..nf {
            for s in 0..nf {
                let llt: Number = (0..=r.min(s))
                    .map(|q| fac.l[r * k + q] * fac.l[s * k + q])
                    .sum();
                let hv = h[order[r] * k + order[s]];
                assert!(
                    (llt - hv).abs() < 1e-10 * (1.0 + hv.abs()),
                    "({r},{s}): {llt} vs {hv}"
                );
            }
            for q in (r + 1)..nf {
                assert_eq!(fac.l[r * k + q], 0.0, "not lower triangular at ({r},{q})");
            }
        }
        // And solves agree with a fresh factor of that block.
        let mut fresh = FreeFactor::new(k);
        assert_eq!(fresh.refactor(&h, &order, &mut budget), Some(true));
        let rhs0: Vec<Number> = (0..nf).map(|_| lcg(&mut seed)).collect();
        let (mut x1, mut x2) = (rhs0.clone(), rhs0);
        fac.solve(&mut x1);
        fresh.solve(&mut x2);
        for (u, v) in x1.iter().zip(&x2) {
            assert!((u - v).abs() < 1e-9 * (1.0 + v.abs()), "{x1:?} vs {x2:?}");
        }
    }
}
