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
//! `λ = 8`, `z_L(x1) = 20` against the true `λ = -2`, `z_L(x0) = 10`). That
//! example fails on its residual. A set that balances the residual from a
//! side the model does not declare fails too, as a misplaced bound multiplier
//! (gh#1202; before that it passed).
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
//! the reduced solve's attribution as stays valid. The exception is an
//! attribution that cannot be kept at working precision — a cancelling
//! `±1e10` pair on constraints with the same gradient (deb7, gh#1202) —
//! where it starts from zero and leans to the smallest valid multipliers
//! instead (`Outcome::restarted`). That is deterministic,
//! and it is the easiest rule to defend: every multiplier in the answer
//! belongs to a constraint the original model declares and that is active,
//! with the right sign, and stationarity holds.
//!
//! What this does **not** move: a multiplier on a *declared* bound that is
//! active at `x*`, even when presolve derived a tighter-or-equal bound on the
//! same side (a degenerate pair: the declared bound is the incumbent and
//! keeps it), and anything the residual does not reach.
//!
//! The component is solved sparse, one strictly convex QP per proximal step
//! through `pounce-qp` ([`StepQp`]), with no cap on its size: the work is
//! bounded by an allowance of active-set changes and the solve's deadline
//! ([`Budget`]). Running out, a failed step, or a solve that does not reduce
//! the residual leaves the duals exactly as the solver reported them and
//! says so through [`Outcome`].
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
use pounce_common::types::{
    NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF, Number, lower_bound_present, upper_bound_present,
};
use pounce_feral::{FeralConfig, FeralSolverInterface};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{
    BoundStatus, ConsStatus, HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem,
    QpSolver, QpStatus, QpWarmStart, WorkingSet,
};

/// Most work one recovery may do, all proximal steps together (gh#1196,
/// gh#1202), in units of one active-set change on a KKT system of
/// dimension one: each step's QP is charged `(changes + 1) × dim`. A change
/// costs a sparse Schur update and the solves around it, which grow with
/// the KKT system, so a count of changes alone does not bound the time.
///
/// Measured (release, one thread): qpnstair uses `1.5e3`, deb7 `6.3e4`.
/// A component with a row touching every multiplier (the adversarial
/// `Stair`) uses `6.4e5` at 800 multipliers (1.0 s) and `9.8e5` at 1000
/// (1.8 s); larger ones run out and roll back, at a few seconds.
pub(crate) const MAX_WORK: f64 = 1e6;

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
    /// Multipliers the solve took on (0 when none was needed).
    pub n_unknowns: usize,
    /// `‖stationarity‖∞` over the component, before (with the spurious
    /// entries in place) and after.
    pub residual_before: Number,
    pub residual_after: Number,
    /// `true` when the duals were left untouched; [`Outcome::cause`] says
    /// why.
    pub rolled_back: bool,
    /// `true` when the roll-back was the work budget or the deadline
    /// running out ([`Budget`], gh#1196).
    pub budget_exhausted: bool,
    /// The solve started from zero, not from the reduced solve's
    /// multipliers, because rounding at those already cost a tenth of the
    /// target (gh#1202).
    pub restarted: bool,
    /// Which branch rolled back, with what it measured (gh#1202); `None`
    /// when nothing was rolled back.
    pub cause: Option<RollBack>,
}

/// Why [`reattribute`] kept the reduced problem's duals (gh#1202). Each
/// cause wants a different fix, so the warning names it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum RollBack {
    /// No active row or declared bound the residual reaches can carry it:
    /// there is nothing to solve for.
    NoMultipliers { vars: usize },
    /// The active-set allowance or the solve's deadline ran out ([`Budget`]).
    Budget { unknowns: usize },
    /// A proximal step's QP ended in a status other than optimal, a time
    /// or iteration limit: its numerics failed. Each step's QP is strictly
    /// convex and feasible by construction, so no other status is valid.
    Solver {
        unknowns: usize,
        status: &'static str,
    },
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
            RollBack::NoMultipliers { vars } => write!(
                f,
                "no active row or declared bound reaches the residual \
                 ({vars} variables)"
            ),
            RollBack::Budget { unknowns } => write!(
                f,
                "work budget or deadline spent ({unknowns} multipliers, \
                 {MAX_WORK:.0e} work units)"
            ),
            RollBack::Solver { unknowns, status } => write!(
                f,
                "a proximal step's QP failed ({unknowns} multipliers; status {status})"
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
    }
    out.residual_before = vars.iter().fold(0.0_f64, |a, &j| a.max(before[j].abs()));
    let restore = |lambda: &mut [Number], z_l: &mut [Number], z_u: &mut [Number]| {
        lambda.copy_from_slice(&lam0);
        z_l.copy_from_slice(&zl0);
        z_u.copy_from_slice(&zu0);
    };
    if unknowns.is_empty() {
        restore(lambda, z_l, z_u);
        out.rolled_back = true;
        out.cause = Some(RollBack::NoMultipliers { vars: vars.len() });
        out.residual_after = out.residual_before;
        out.n_spurious = 0;
        return out;
    }
    out.n_unknowns = unknowns.len();

    // The component's system `A·w + b`, sparse by column, in the
    // transformed variable `w = s·y` so every sign constraint reads `w ≥ 0`,
    // and with `b = r − A·w_current` so the component's residual is
    // `A·w + b`.
    let nv = vars.len();
    let k = unknowns.len();
    let mut cols: Vec<Vec<(usize, Number)>> = Vec::with_capacity(k);
    let mut w0 = vec![0.0; k];
    let mut constrained = vec![false; k];
    let mut current = vec![0.0; k];
    for (c, &(u, sign)) in unknowns.iter().enumerate() {
        let s = if sign == Sign::NonPos { -1.0 } else { 1.0 };
        constrained[c] = sign != Sign::Free;
        let mut col: Vec<(usize, Number)> = Vec::new();
        match u {
            Unknown::Row(i) => {
                for &(jj, aij) in &by_row[i] {
                    let slot = var_slot[jj];
                    if slot != usize::MAX {
                        col.push((slot, s * aij));
                    }
                }
                w0[c] = s * lambda[i];
            }
            Unknown::ZLower(j) => {
                col.push((var_slot[j], -1.0));
                w0[c] = z_l[j];
            }
            Unknown::ZUpper(j) => {
                col.push((var_slot[j], 1.0));
                w0[c] = z_u[j];
            }
        }
        current[c] = if w0[c].is_finite() { w0[c] } else { 0.0 };
        if constrained[c] && !(w0[c] >= 0.0) {
            w0[c] = 0.0;
        }
        cols.push(col);
    }
    let mut sys = LsSystem {
        nrows: nv,
        cols,
        b: vars.iter().map(|&j| r[j]).collect(),
        constrained,
    };
    // `r` was computed with the current (pre-projection) values; express
    // it against them.
    let shift = sys.times(&current);
    for (bv, sv) in sys.b.iter_mut().zip(&shift) {
        *bv -= sv;
    }
    let sys = sys;
    // Accept the recovery only when it actually closes stationarity: to the
    // solver's own residual, or to a level no tolerance would notice.
    let target = (10.0 * out.residual_before).max(1e-8 * scale);
    // Anchor at the reduced solve's multipliers, so a degenerate component
    // keeps as much of their attribution as stays valid — unless the
    // anchor cannot be honoured at working precision: when rounding at the
    // anchor alone already costs a tenth of the target, start from zero,
    // which leans to the smallest valid multipliers instead. deb7 (gh#1202)
    // hands over a cancelling pair `±1.16e10` on two equalities with the
    // same gradient at `x*`; anchored there, the residual crept down ×0.97
    // a step against a floor of `5e-4` and the recovery rolled back, while
    // multipliers of size `1.4e3` close it to `1e-4`.
    let start = if sys.rounding_floor(&w0) > 0.1 * target {
        out.restarted = true;
        vec![0.0; k]
    } else {
        w0
    };
    let w = match solve_sign_constrained_ls(&sys, &start, target, &mut budget) {
        Ok(w) => Some(w),
        Err(Failure::Budget) => None,
        Err(Failure::Solver(status)) => {
            restore(lambda, z_l, z_u);
            out.rolled_back = true;
            out.cause = Some(RollBack::Solver {
                unknowns: k,
                status,
            });
            out.residual_after = out.residual_before;
            out.n_spurious = 0;
            out.n_unknowns = 0;
            return out;
        }
    };
    let Some(w) = w else {
        restore(lambda, z_l, z_u);
        out.rolled_back = true;
        out.budget_exhausted = true;
        out.cause = Some(RollBack::Budget { unknowns: k });
        out.residual_after = out.residual_before;
        out.n_spurious = 0;
        out.n_unknowns = 0;
        return out;
    };
    let res_after = sys
        .times(&w)
        .iter()
        .zip(&sys.b)
        .fold(0.0_f64, |m, (x, y)| m.max((x + y).abs()));
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

/// What one recovery may spend: an allowance of active-set changes and,
/// optionally, the solve's own deadline (gh#1196).
///
/// The proximal-point loop takes up to 200 steps, and each is an active-set
/// QP whose iteration count grows with the number of sign changes. Unbounded,
/// Vanderbei `qpnstair` (`presolve=yes`) spent 838 s here after the
/// interior-point solve had converged, past a 10 s `max_wall_time`. Spending
/// the budget keeps the reduced solve's duals and sets `rolled_back`.
///
/// Counted in working-set changes, which the QP reports exactly, so a
/// recovery's outcome does not depend on the machine it runs on; the
/// deadline is the one input that does.
pub(crate) struct Budget<'a> {
    work: f64,
    deadline: Option<&'a Deadline>,
}

impl Default for Budget<'_> {
    fn default() -> Self {
        Self::new(MAX_WORK, None)
    }
}

impl<'a> Budget<'a> {
    pub(crate) fn new(work: f64, deadline: Option<&'a Deadline>) -> Self {
        Self { work, deadline }
    }

    fn spent(&self) -> bool {
        !(self.work > 0.0) || self.deadline.is_some_and(|d| d.exceeded().is_some())
    }

    /// The QP options one step on a KKT system of dimension `dim` may use:
    /// the changes the remaining allowance pays for as its working-set
    /// budget, and the remaining wall time as its time limit.
    fn options(&self, base: &QpOptions, dim: usize) -> QpOptions {
        let mut o = base.clone();
        let changes = (self.work / dim.max(1) as f64).floor();
        o.max_working_set_changes = Some(changes.clamp(0.0, u32::MAX as f64) as u32);
        // A deadline with no limit reports a finite `1e20` s left
        // (`max_wall_time`'s default), past the `1.8e19` s a `Duration`
        // holds: that is no time limit, not a panic.
        if let Some(d) = self.deadline {
            o.time_limit = std::time::Duration::try_from_secs_f64(d.remaining_wall().max(0.0)).ok();
        }
        o
    }

    /// Charge one step's QP on a KKT system of dimension `dim`: its
    /// working-set changes, and one for the step's factorization.
    fn charge(&mut self, changes: u32, dim: usize) {
        self.work -= (changes as f64 + 1.0) * dim as f64;
    }
}

/// Why [`solve_sign_constrained_ls`] gave up.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Failure {
    /// The [`Budget`] ran out.
    Budget,
    /// A step's QP ended in this status.
    Solver(&'static str),
}

/// The component's least-squares system `A·w + b`, sparse by column:
/// `cols[c]` holds unknown `c`'s `(row, value)` entries, a row being one of
/// the component's variables. `constrained[c]`: `w_c ≥ 0`, else free.
struct LsSystem {
    nrows: usize,
    cols: Vec<Vec<(usize, Number)>>,
    b: Vec<Number>,
    constrained: Vec<bool>,
}

impl LsSystem {
    fn k(&self) -> usize {
        self.cols.len()
    }

    /// `A·w`.
    fn times(&self, w: &[Number]) -> Vec<Number> {
        let mut out = vec![0.0; self.nrows];
        for (col, &wc) in self.cols.iter().zip(w) {
            if wc != 0.0 {
                for &(row, a) in col {
                    out[row] += a * wc;
                }
            }
        }
        out
    }

    /// `A·w + b`.
    fn residual(&self, w: &[Number]) -> Vec<Number> {
        let mut r = self.times(w);
        for (x, y) in r.iter_mut().zip(&self.b) {
            *x += y;
        }
        r
    }

    /// `max_c ‖A_c‖₂`.
    fn max_col_norm(&self) -> Number {
        self.cols
            .iter()
            .map(|col| col.iter().map(|&(_, a)| a * a).sum::<Number>().sqrt())
            .fold(0.0_f64, Number::max)
    }

    /// What rounding alone costs evaluating `A·w + b`:
    /// `100·ε·max_r (|b_r| + Σ_c |a_rc·w_c|)`, row by row. Not
    /// `‖A‖∞·‖w‖∞`, which pairs the densest row with the largest entry
    /// anywhere, so one huge multiplier reads as a floor on every row
    /// (deb7, gh#1202: `3.3e-2` from a `1.16e10` pair, against `5e-4` on the
    /// row that pair sits in).
    fn rounding_floor(&self, w: &[Number]) -> Number {
        let mut acc: Vec<Number> = self.b.iter().map(|x| x.abs()).collect();
        for (col, &wc) in self.cols.iter().zip(w) {
            for &(row, a) in col {
                acc[row] += (a * wc).abs();
            }
        }
        100.0 * Number::EPSILON * acc.iter().fold(0.0_f64, |m, &x| m.max(x))
    }

    /// Is `w` optimal for `min ½‖A·w + b‖²` subject to the sign
    /// constraints, given its residual `res = A·w + b`? The projected
    /// gradient `Aᵀres` — every free entry, and a constrained entry at zero
    /// where it pushes below zero — at most `1e-6` of
    /// `max_c ‖A_c‖₂ · ‖res‖₂`.
    ///
    /// This is the stop for an *inconsistent* system, whose residual stays
    /// above the rounding floor at the optimum: on deb7 (gh#1202) the
    /// optimum is `1.02e-4` (the primal is accurate to about that), and
    /// without this test the iteration kept weakening `μ` past it until an
    /// ill-conditioned active-set QP spent the whole work budget. A
    /// consistent system stops at the floor first: there `‖res‖₂ → 0` and
    /// this ratio does not shrink.
    fn ls_optimal(&self, res: &[Number], w: &[Number], col_norm: Number) -> bool {
        let rn = res.iter().map(|x| x * x).sum::<Number>().sqrt();
        if !(rn > 0.0) || !(col_norm > 0.0) {
            return false;
        }
        let mut pg = 0.0_f64;
        for (c, col) in self.cols.iter().enumerate() {
            let gp: Number = col.iter().map(|&(row, a)| a * res[row]).sum();
            let v = if self.constrained[c] && w[c] <= 0.0 {
                (-gp).max(0.0)
            } else {
                gp.abs()
            };
            pg = pg.max(v);
        }
        pg <= 1e-6 * col_norm * rn
    }
}

/// A row of `A` with more than this many entries, and more than
/// [`DENSE_ROW_FACTOR`] times the median row, keeps its residual as a
/// variable in [`StepQp`] instead of entering `AᵀA`, where it would put
/// `c(c+1)/2` entries. Measured: the densest row of qpnstair's component has
/// 24 entries (median 4), of deb7's 14 (median 3); a variable that every
/// multiplier touches has one per multiplier.
const DENSE_ROW_MIN: usize = 32;
const DENSE_ROW_FACTOR: usize = 8;

/// One proximal step as a sparse QP in the *correction* `d = w⁺ − w_k`:
///
/// ```text
///   min ½‖A_s·d + r_s‖² + ½‖t‖² + ½μ‖d‖²
///   s.t.  t − A_t·d = r_t,   d_c ≥ −w_k,c where constrained
/// ```
///
/// with `r_k = A·w_k + b` evaluated directly and split by rows: the sparse
/// rows `A_s` enter the Hessian as `A_sᵀA_s`, and each dense row of `A_t`
/// (see [`DENSE_ROW_MIN`]) keeps its residual as a variable `t`. That is
/// `min ½‖A·w⁺ + b‖² + ½μ‖w⁺ − w_k‖²` either way. It is posed in
/// correction form, so the KKT solve's error scales with `|d|`, which
/// vanishes as the iteration converges, not with `|w|`: each step is a
/// round of iterative refinement, and that is also what corrects the
/// squared conditioning of `A_sᵀA_s`. Solving for `w⁺` outright left
/// `1.3e-10` of error on a unique `λ = 2`.
///
/// Why split, measured on the recovery alone (release, single thread):
/// with every row augmented (`r` a variable throughout, no `AᵀA`), qpnstair
/// took 13 ms and deb7's final call 347 ms against the dense solve's 90 and
/// 280; with every row eliminated, 8 and 148 ms, but a component with one
/// row touching all 800 multipliers (`AᵀA` dense) took 14.4 s against
/// 1.7 s augmented. The split takes each where it wins.
struct StepQp {
    k: usize,
    /// Rows of the system kept as residual variables, in `t` order.
    dense: Vec<usize>,
    /// Row → its position in `dense`, or `usize::MAX` for an eliminated row.
    dense_at: Vec<usize>,
    h: SymTMatrix,
    /// The Hessian's values without `μ`, and where its `d` diagonal sits.
    base: Vec<Number>,
    diag_at: Vec<usize>,
    a: GenTMatrix,
    xl: Vec<Number>,
    xu: Vec<Number>,
}

impl StepQp {
    fn new(sys: &LsSystem) -> Self {
        let k = sys.k();
        let mut by_row: Vec<Vec<(usize, Number)>> = vec![Vec::new(); sys.nrows];
        for (c, col) in sys.cols.iter().enumerate() {
            for &(row, a) in col {
                by_row[row].push((c, a));
            }
        }
        let mut counts: Vec<usize> = by_row.iter().map(Vec::len).collect();
        counts.sort_unstable();
        let median = counts.get(counts.len() / 2).copied().unwrap_or(0);
        let limit = DENSE_ROW_MIN.max(DENSE_ROW_FACTOR * median);
        let dense: Vec<usize> = (0..sys.nrows)
            .filter(|&r| by_row[r].len() > limit)
            .collect();
        let mut dense_at = vec![usize::MAX; sys.nrows];
        for (t, &r) in dense.iter().enumerate() {
            dense_at[r] = t;
        }
        let n = k + dense.len();
        // `A_sᵀA_s` (lower triangle), plus the diagonal of every variable.
        let mut acc: std::collections::BTreeMap<(usize, usize), Number> = Default::default();
        for c in 0..n {
            acc.insert((c, c), if c < k { 0.0 } else { 1.0 });
        }
        for (r, row) in by_row.iter().enumerate() {
            if dense_at[r] != usize::MAX {
                continue;
            }
            for (i, &(ci, ai)) in row.iter().enumerate() {
                for &(cj, aj) in &row[..=i] {
                    let key = if ci >= cj { (ci, cj) } else { (cj, ci) };
                    *acc.entry(key).or_insert(0.0) += ai * aj;
                }
            }
        }
        let (mut ir, mut jc, mut base) = (Vec::new(), Vec::new(), Vec::new());
        let mut diag_at = Vec::with_capacity(k);
        for (&(p, q), &v) in &acc {
            if p == q && p < k {
                diag_at.push(base.len());
            }
            ir.push(p as i32 + 1);
            jc.push(q as i32 + 1);
            base.push(v);
        }
        let mut h = SymTMatrix::new(SymTMatrixSpace::new(n as i32, ir, jc));
        h.set_values(&base);
        // `t − A_t·d = r_t`.
        let (mut ir, mut jc, mut val) = (Vec::new(), Vec::new(), Vec::new());
        for (t, &r) in dense.iter().enumerate() {
            for &(c, a) in &by_row[r] {
                ir.push(t as i32 + 1);
                jc.push(c as i32 + 1);
                val.push(-a);
            }
            ir.push(t as i32 + 1);
            jc.push((k + t) as i32 + 1);
            val.push(1.0);
        }
        let mut a = GenTMatrix::new(GenTMatrixSpace::new(dense.len() as i32, n as i32, ir, jc));
        a.set_values(&val);
        let mut xl = vec![NLP_LOWER_BOUND_INF; n];
        for (c, &con) in sys.constrained.iter().enumerate() {
            if con {
                xl[c] = 0.0;
            }
        }
        Self {
            k,
            dense,
            dense_at,
            h,
            base,
            diag_at,
            a,
            xl,
            xu: vec![NLP_UPPER_BOUND_INF; n],
        }
    }

    fn n(&self) -> usize {
        self.k + self.dense.len()
    }

    fn set_mu(&mut self, mu: Number) {
        let mut v = self.base.clone();
        for &d in &self.diag_at {
            v[d] += mu;
        }
        self.h.set_values(&v);
    }

    /// The step's linear term `A_sᵀr_s` and right-hand side `r_t`.
    fn rhs(&self, sys: &LsSystem, r_k: &[Number]) -> (Vec<Number>, Vec<Number>) {
        let mut g = vec![0.0; self.n()];
        for (c, col) in sys.cols.iter().enumerate() {
            g[c] = col
                .iter()
                .filter(|&&(row, _)| self.dense_at[row] == usize::MAX)
                .map(|&(row, a)| a * r_k[row])
                .sum();
        }
        let r_t = self.dense.iter().map(|&r| r_k[r]).collect();
        (g, r_t)
    }

    /// Set up the step from `w_k` (whose sign constraints hold): the lower
    /// bounds `d_c ≥ −w_k,c`, and the warm start `(d, t) = (0, r_t)` with
    /// each constrained entry already at zero on its bound. `prev` carries
    /// the previous step's multipliers, which stay a good guess.
    fn start(
        &mut self,
        sys: &LsSystem,
        w: &[Number],
        r_t: &[Number],
        prev: Option<QpWarmStart>,
    ) -> QpWarmStart {
        let n = self.n();
        let m = self.dense.len();
        let mut working = WorkingSet::cold(n, m);
        for (c, &con) in sys.constrained.iter().enumerate() {
            if con {
                self.xl[c] = -w[c];
                if w[c] <= 0.0 {
                    working.bounds[c] = BoundStatus::AtLower;
                }
            }
        }
        working.constraints.fill(ConsStatus::Equality);
        let mut x = vec![0.0; self.k];
        x.extend_from_slice(r_t);
        let (lambda_g, lambda_x) = match prev {
            Some(p) => (p.lambda_g, p.lambda_x),
            None => (vec![0.0; m], vec![0.0; n]),
        };
        QpWarmStart {
            x,
            lambda_g,
            lambda_x,
            working,
        }
    }
}

/// Proximal steps a residual already under the caller's acceptance target
/// gets to halve before the iteration counts as stalled.
const STALL_STEPS: usize = 3;

/// `min ½‖A·w + b‖²` subject to `w_c ≥ 0` where constrained, by a
/// proximal-point iteration `w⁺ = argmin ½‖A·w + b‖² + ½μ‖w − w_k‖²` started
/// at `w0`. Each step is a strictly convex sparse QP ([`StepQp`]) solved by
/// `pounce-qp`'s active-set method, warm-started from the step before. On a
/// consistent system with a unique solution this converges to it; on a
/// degenerate one, to a solution near `w0`.
///
/// When the residual stops halving, `μ` is cut ×100 (to `1e-12·max_c ‖A_c‖²`
/// at the least).
///
/// It returns the iterate with the smallest residual `‖A·w + b‖∞`, as soon
/// as that is down to the rounding floor ([`LsSystem::rounding_floor`]), or
/// the iterate is a least-squares optimum ([`LsSystem::ls_optimal`]), or
/// once it is under `accept` (the caller's acceptance target) and has not
/// halved in [`STALL_STEPS`] steps — or when the step itself vanishes.
/// The step test alone is not reachable in floating point once the
/// multipliers are large: on Vanderbei `qpnstair` (`‖w‖∞ ≈ 9e4`) the
/// residual contracts ×0.44 per step to a floor of `2e-9` by step 27, while
/// the step stalls near `4e-7` — `4e-12` relative against the `1e-15` the
/// test asks for. The remaining 170 steps iterated rounding noise, and the
/// active-set QP of the time cycled on near-zero entries to its iteration
/// cap: the whole work allowance, three times over, on a recovery that had
/// met its acceptance target at step 15.
fn solve_sign_constrained_ls(
    sys: &LsSystem,
    w0: &[Number],
    accept: Number,
    budget: &mut Budget<'_>,
) -> Result<Vec<Number>, Failure> {
    let k = sys.k();
    let col_norm = sys.max_col_norm();
    let max_diag = (col_norm * col_norm).max(1e-300);
    // Small from the start: a Tikhonov step centred at `w_k` tends, as
    // `μ → 0`, to the least-squares solution nearest `w_k`, which is where
    // the proximal iteration was heading anyway, and the correction form
    // absorbs the conditioning (`ε/μ ≈ 2e-6` relative). `1e-6`, which the
    // dense normal-equations solve this replaced needed, took qpnstair 30
    // steps and 222 ms; `1e-10` takes it 15 ms.
    let mut mu = 1e-10 * max_diag;
    let mu_min = 1e-12 * max_diag;
    let mut qp = StepQp::new(sys);
    // `(d, t)` and one multiplier per dense row.
    let kkt_dim = qp.n() + qp.dense.len();
    let base = QpOptions {
        // The work allowance bounds the solve; a pass may take as many
        // changes as it has left.
        max_iter: u32::MAX,
        use_schur_updates: true,
        certify_second_order: false,
        certify_recession_ray: false,
        ..QpOptions::default()
    };
    // Serial, for a deterministic answer. No in-backend refinement: the
    // loop around it is the refinement (each step recomputes `r_k` exactly
    // and solves for a correction), which is the condition `FeralConfig::
    // refine` names for turning it off.
    let mut solver =
        ParametricActiveSetSolver::new(Box::new(FeralSolverInterface::with_config(FeralConfig {
            parallel: Some(false),
            refine: false,
            ..FeralConfig::from_env()
        })));
    let mut w = w0.to_vec();
    let mut res = sys.residual(&w);
    let mut best = (res.iter().fold(0.0_f64, |m, x| m.max(x.abs())), w.clone());
    let mut prev: Option<QpWarmStart> = None;
    // Residuals of the last few steps, newest last.
    let mut recent: Vec<Number> = Vec::with_capacity(STALL_STEPS + 1);
    for _ in 0..200 {
        if budget.spent() {
            return Err(Failure::Budget);
        }
        qp.set_mu(mu);
        let (g, r_t) = qp.rhs(sys, &res);
        let warm = qp.start(sys, &w, &r_t, prev.take());
        let problem = QpProblem {
            n: qp.n(),
            m: qp.dense.len(),
            h: &qp.h,
            g: &g,
            a: &qp.a,
            bl: &r_t,
            bu: &r_t,
            xl: &qp.xl,
            xu: &qp.xu,
            hessian_inertia: HessianInertia::Psd,
        };
        // The step solves for a correction against `r_k`, so its
        // tolerances scale with `r_k`: an absolute `1e-9` read the start as
        // optimal once `‖r_k‖∞` fell below it, stopping a unique `λ = 2` at
        // `1.99999999987`.
        let r_inf = res.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let tol = (1e-6 * r_inf * col_norm.max(1.0)).max(Number::MIN_POSITIVE);
        let mut opts = budget.options(&base, kkt_dim);
        opts.opt_tol = tol;
        opts.feas_tol = tol;
        let sol = solver
            .solve(&problem, Some(&warm), &opts)
            .map_err(|_| Failure::Solver("error"))?;
        budget.charge(sol.stats.n_working_set_changes, kkt_dim);
        match sol.status {
            QpStatus::Optimal => {}
            QpStatus::MaxIter | QpStatus::TimeLimit => return Err(Failure::Budget),
            QpStatus::Infeasible => return Err(Failure::Solver("infeasible")),
            QpStatus::Unbounded => return Err(Failure::Solver("unbounded")),
            QpStatus::NumericalError => return Err(Failure::Solver("numerical error")),
        }
        // `d_c ≥ −w_c` holds to the QP's feasibility tolerance; make the
        // sign constraints exact, so every reported multiplier has its sign.
        // An entry the QP holds on its bound is exactly zero: `w_c + d_c`
        // there is rounding (`1e-17`), and reading it as off the bound made
        // the next step's QP re-add it, 14–25 changes a step on `qpnstair`
        // after the active set had settled.
        let next: Vec<Number> = w
            .iter()
            .zip(&sol.x[..k])
            .zip(&sys.constrained)
            .zip(&sol.working.bounds[..k])
            .map(|(((&wc, &dc), &con), &st)| {
                if con && st.is_active() {
                    0.0
                } else if con {
                    (wc + dc).max(0.0)
                } else {
                    wc + dc
                }
            })
            .collect();
        let step = next
            .iter()
            .zip(&w)
            .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()));
        let size = next.iter().fold(1.0_f64, |m, p| m.max(p.abs()));
        w = next;
        prev = Some(QpWarmStart {
            x: sol.x,
            lambda_g: sol.lambda_g,
            lambda_x: sol.lambda_x,
            working: sol.working,
        });
        res = sys.residual(&w);
        let r = res.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        if r < best.0 {
            best = (r, w.clone());
        }
        if recent.len() > STALL_STEPS {
            recent.remove(0);
        }
        recent.push(r);
        let stalled = recent.len() > STALL_STEPS && r > 0.5 * recent[0];
        let at_floor = best.0 <= sys.rounding_floor(&best.1);
        if step <= 1e-15 * size || at_floor || sys.ls_optimal(&res, &w, col_norm) {
            break;
        }
        if stalled && mu > mu_min {
            // Each step removes only `μ/(σ²+μ)` of the error along a
            // direction of singular value `σ`: where `σ²` is far below `μ`
            // the iteration crawls (deb7, gh#1202: ×0.97 a step, from
            // `σ² ≈ 3e-8·max diag` against `μ = 1e-6·max diag`). Weaken
            // the proximal term rather than stop.
            mu = (mu * 1e-2).max(mu_min);
            recent.clear();
            continue;
        }
        if best.0 <= accept && stalled {
            break;
        }
    }
    Ok(best.1)
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
        assert!(!o.restarted, "{o:?}");
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

    /// Two equalities with the same gradient at `x*` (rows 0 and 1, both
    /// `x0 = 0`), and `x0 − x1 = 0` tying them to `x1`, whose derived lower
    /// bound carries the reduced solve's spurious `z_L = 1`. Only `λ0 + λ1`
    /// is determined (`= −2`, with `λ2 = 1`). `pair` is the reduced
    /// solve's split, which sums to `−1` (it had `λ2 = 0` and the spurious
    /// `z_L = 1` instead), so the reduced point is exactly stationary.
    fn parallel_pair(pair: (Number, Number)) -> (Outcome, Vec<Number>) {
        let jac = [(0, 0, 1.0), (1, 0, 1.0), (2, 0, 1.0), (2, 1, -1.0)];
        let v = KktView {
            x: &[0.0, 0.0],
            g: &[0.0, 0.0, 0.0],
            grad_f: &[1.0, 1.0],
            jac: &jac,
            x_l: &[-1e19, -1e19],
            x_u: &[1e19, 1e19],
            g_l: &[0.0, 0.0, 0.0],
            g_u: &[0.0, 0.0, 0.0],
            derived_l: &[false, true],
            derived_u: &[false, false],
        };
        let mut lam = vec![pair.0, pair.1, 0.0];
        let (mut zl, mut zu) = (vec![0.0, 1.0], vec![0.0, 0.0]);
        let o = reattribute(&v, &mut lam, &mut zl, &mut zu, Budget::default());
        assert!(!o.rolled_back, "{o:?}");
        assert_eq!(zl, vec![0.0, 0.0]);
        let r = residual(&v, &lam, &zl, &zu);
        assert!(r.iter().all(|x| x.abs() < 1e-8), "{r:?} λ = {lam:?}");
        assert!((lam[2] - 1.0).abs() < 1e-8 && (lam[0] + lam[1] + 2.0).abs() < 1e-8);
        (o, lam)
    }

    /// gh#1202 (deb7): a cancelling `±1e10` split cannot be honoured at
    /// working precision — rounding there alone exceeds the target — so the
    /// solve starts from zero and returns small multipliers.
    #[test]
    fn a_split_past_working_precision_is_not_kept() {
        let (o, lam) = parallel_pair((1e10, -1e10 - 1.0));
        assert!(o.restarted, "{o:?}");
        assert!(lam[0].abs() < 10.0 && lam[1].abs() < 10.0, "{lam:?}");
    }

    /// The other branch: a moderate split is kept as nearly as stays valid —
    /// the proximal iteration projects `(3, −4)` onto `λ0 + λ1 = −2`.
    #[test]
    fn a_split_within_working_precision_is_kept() {
        let (o, lam) = parallel_pair((3.0, -4.0));
        assert!(!o.restarted, "{o:?}");
        assert!(
            (lam[0] - 2.5).abs() < 1e-6 && (lam[1] + 4.5).abs() < 1e-6,
            "{lam:?}"
        );
    }

    /// A one-column system `A = col`, `b`, sign-constrained or not.
    fn one_col(col: &[Number], b: &[Number], constrained: bool) -> LsSystem {
        LsSystem {
            nrows: col.len(),
            cols: vec![col.iter().copied().enumerate().collect()],
            b: b.to_vec(),
            constrained: vec![constrained],
        }
    }

    /// [`LsSystem::ls_optimal`]: optimal at an inconsistent least-squares
    /// optimum, and at a sign-constrained one held at zero; not elsewhere;
    /// and never claimed on a zero residual (the floor test's case).
    #[test]
    fn least_squares_optimality_by_branch() {
        // A = [1; 1], b = [−1, 1]: the optimum is w = 0, residual (−1, 1).
        let free = one_col(&[1.0, 1.0], &[-1.0, 1.0], false);
        assert!(free.ls_optimal(&[-1.0, 1.0], &[0.0], 1.0));
        // w = 0.5 is not the optimum.
        assert!(!free.ls_optimal(&[-0.5, 1.5], &[0.5], 1.0));
        // A = [1], b = [1], w ≥ 0: the gradient pushes below zero, so w = 0
        // is optimal for the constrained problem, not for the free one.
        assert!(one_col(&[1.0], &[1.0], true).ls_optimal(&[1.0], &[0.0], 1.0));
        assert!(!one_col(&[1.0], &[1.0], false).ls_optimal(&[1.0], &[0.0], 1.0));
        // A zero residual is left to the floor test.
        assert!(!free.ls_optimal(&[0.0, 0.0], &[1.0], 1.0));
    }

    /// [`LsSystem::rounding_floor`] is per row: a huge entry raises the
    /// floor of the rows it appears in, not of every row.
    #[test]
    fn the_rounding_floor_is_per_row() {
        // Row 0 touches w0, row 1 only w1.
        let sys = LsSystem {
            nrows: 2,
            cols: vec![vec![(0, 1.0)], vec![(1, 1.0)]],
            b: vec![0.0, 0.0],
            constrained: vec![false, false],
        };
        assert_eq!(
            sys.rounding_floor(&[1e10, 1.0]),
            100.0 * Number::EPSILON * 1e10
        );
        assert_eq!(sys.rounding_floor(&[1.0, 1.0]), 100.0 * Number::EPSILON);
    }

    /// Deterministic pseudo-random numbers in `[-1, 1)`.
    fn lcg(seed: &mut u64) -> Number {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as Number / (1u64 << 53) as Number) * 2.0 - 1.0
    }

    /// gh#1196: a large component whose sign-constrained
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

    /// The slow branch completes inside the default budget past the old
    /// dense solve's 600 cap, and its answer is a valid multiplier set:
    /// right signs, stationarity no worse than the acceptance target. Its
    /// one row touching every multiplier is the case [`StepQp`] keeps
    /// augmented (`t`): eliminated, `AᵀA` is dense and this took 14.4 s.
    /// Measured: 1.0 s, against 0.49 s for the dense solve this replaced.
    /// That is the one structure where the dense solve was faster.
    #[test]
    fn a_large_component_with_many_active_set_changes_is_affordable() {
        let p = Stair::new(400, 800);
        let (mut lam, mut zl, mut zu) = p.duals();
        let o = reattribute(&p.view(), &mut lam, &mut zl, &mut zu, Budget::default());
        assert!(!o.rolled_back, "{o:?}");
        assert_eq!(o.n_unknowns, 800, "{o:?}");
        assert!(lam.iter().all(|&l| l >= 0.0), "a λ ≥ 0 went negative");
        assert_eq!(zl[0], 0.0);
        let zeroed = lam.iter().filter(|&&l| l == 0.0).count();
        assert!(
            zeroed > 50,
            "only {zeroed} λ reached their bound: not the slow branch"
        );
    }

    /// Out of budget, the recovery stops and keeps the reduced solve's
    /// duals exactly, and says why.
    #[test]
    fn a_spent_work_budget_rolls_back_to_the_reduced_duals() {
        let p = Stair::new(400, 590);
        let (lam0, zl0, zu0) = p.duals();
        let (mut lam, mut zl, mut zu) = p.duals();
        let o = reattribute(
            &p.view(),
            &mut lam,
            &mut zl,
            &mut zu,
            Budget::new(1.0, None),
        );
        assert!(o.rolled_back && o.budget_exhausted, "{o:?}");
        assert!(matches!(o.cause, Some(RollBack::Budget { .. })), "{o:?}");
        assert_eq!(o.n_spurious, 0);
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

    /// A deadline with no limit (`Deadline::new(1e20, 1e20)`: `1e20` is
    /// `max_wall_time`'s registered default) is no limit: the recovery runs. Its remaining
    /// `1e20` s is past what a `Duration` holds, and converting it panicked
    /// every recovery on a real solve, which no budget-only test reached.
    #[test]
    fn an_unlimited_deadline_does_not_limit_the_recovery() {
        let p = Stair::new(400, 590);
        let (mut lam, mut zl, mut zu) = p.duals();
        let d = Deadline::new(1e20, 1e20);
        let o = reattribute(
            &p.view(),
            &mut lam,
            &mut zl,
            &mut zu,
            Budget::new(MAX_WORK, Some(&d)),
        );
        assert!(!o.rolled_back, "{o:?}");
    }

    /// The solve's deadline stops the recovery too: a solve that is already
    /// out of time spends none of it in postsolve.
    #[test]
    fn a_crossed_deadline_rolls_back_to_the_reduced_duals() {
        let p = Stair::new(400, 590);
        let (lam0, zl0, zu0) = p.duals();
        let (mut lam, mut zl, mut zu) = p.duals();
        let d = Deadline::new(0.0, 0.0);
        let o = reattribute(
            &p.view(),
            &mut lam,
            &mut zl,
            &mut zu,
            Budget::new(MAX_WORK, Some(&d)),
        );
        assert!(o.rolled_back && o.budget_exhausted, "{o:?}");
        assert_eq!((lam, zl, zu), (lam0, zl0, zu0));
    }
}
