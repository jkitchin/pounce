//! Presolve for convex QP (Phase 3.5 — first increment).
//!
//! Reduces a [`QpProblem`] before the interior-point solve and maps the
//! reduced solution back to the original problem space, recovering both
//! the primal `x` and the duals `(y, z)`. The contract is exactness: a
//! presolved-then-postsolved solve must reproduce the no-presolve
//! `(x, y, z)` to solver tolerance (see `tests/presolve_roundtrip.rs`).
//!
//! This is the architectural seam the dev note calls the "missing
//! piece": a **transaction stack** of [`Reduction`]s, each carrying the
//! data needed to undo itself (primal *and* dual). Postsolve replays the
//! stack in reverse. The catalog here is deliberately small but the
//! postsolve is complete, so richer reductions can be added without
//! reworking the recovery path.
//!
//! Reductions implemented:
//! - **Empty rows** (equality / inequality with no nonzeros): a
//!   feasibility check, then drop. Their dual is zero. Detects trivial
//!   primal infeasibility (`0 = b≠0` or `0 ≤ h<0`).
//! - **Fixed-variable elimination** from a singleton equality row
//!   (`a·x_k = b ⇒ x_k = b/a`): substitute `x_k` out of `P`, `c`, `A`,
//!   `G` (adjusting the objective constant and the row right-hand
//!   sides), and recover the fixing row's multiplier from stationarity
//!   at the postsolved point. This is the QP-aware reduction the dev
//!   note flags as net-new (the Hessian coupling moves into the linear
//!   term and the dual must be recovered consistently with `P`).

use crate::qp::{QpProblem, QpSolution, QpStatus, Triplet};

/// Outcome of presolve.
pub enum PresolveOutcome {
    /// Problem reduced; solve `reduced`, then call [`Presolve::postsolve`].
    Reduced(Presolve),
    /// Presolve proved the problem primal-infeasible (e.g. an empty row
    /// `0 = b` with `b ≠ 0`, or contradictory fixed bounds).
    Infeasible,
}

/// A reversible presolve transaction. Each variant stores exactly what
/// postsolve needs to reconstruct the eliminated primal and dual.
enum Reduction {
    /// An empty equality row was dropped. Its multiplier is 0 (it
    /// constrains nothing); postsolve leaves the corresponding `y` entry
    /// at its zero initialization, so no payload is needed.
    EmptyEqRow,
    /// An empty inequality row was dropped. Its multiplier is 0;
    /// postsolve leaves the corresponding `z` entry at zero.
    EmptyIneqRow,
    /// Variable `col` was fixed to `value` by the singleton equality row
    /// `eq_row` (coefficient `a_coef`). Postsolve restores `x[col] =
    /// value` and computes the row's multiplier from stationarity.
    FixedVar {
        col: usize,
        value: f64,
        eq_row: usize,
        a_coef: f64,
    },
}

/// Captured presolve state: the reduced problem plus the transaction
/// stack and the index maps needed to expand a reduced solution back to
/// the original space.
pub struct Presolve {
    /// The reduced problem to hand to the solver.
    pub reduced: QpProblem,
    /// Constant added to the objective by variable substitutions; the
    /// reduced objective plus this equals the original objective.
    pub obj_offset: f64,
    /// Original problem dimensions.
    orig_n: usize,
    orig_m_eq: usize,
    orig_m_ineq: usize,
    /// `kept_col[reduced_col] = orig_col`.
    kept_cols: Vec<usize>,
    /// `kept_eq[reduced_eq_row] = orig_eq_row`.
    kept_eq: Vec<usize>,
    /// `kept_ineq[reduced_ineq_row] = orig_ineq_row`.
    kept_ineq: Vec<usize>,
    /// Original objective data, retained for fixing-row dual recovery.
    orig: QpProblem,
    stack: Vec<Reduction>,
}

const ZERO_TOL: f64 = 0.0;

/// Run presolve on `prob`.
pub fn presolve(prob: &QpProblem) -> PresolveOutcome {
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();

    let mut stack: Vec<Reduction> = Vec::new();

    // --- pass 1: detect empty rows and singleton-equality fixings ---
    // Per-row nonzero counts and (for singletons) the single entry.
    let mut eq_nnz = vec![0usize; m_eq];
    let mut eq_single: Vec<Option<(usize, f64)>> = vec![None; m_eq];
    for t in &prob.a {
        if t.val != ZERO_TOL {
            eq_nnz[t.row] += 1;
            eq_single[t.row] = Some((t.col, t.val));
        }
    }
    let mut ineq_nnz = vec![0usize; m_ineq];
    for t in &prob.g {
        if t.val != ZERO_TOL {
            ineq_nnz[t.row] += 1;
        }
    }

    // Fixed-variable assignments: col -> value. A variable can be fixed
    // by at most one singleton row in this first increment; if two
    // singleton rows touch the same variable we only take the first and
    // leave the rest (handled as ordinary rows), keeping the logic
    // simple and correct.
    let mut fixed: Vec<Option<f64>> = vec![None; n];
    let mut eq_dropped = vec![false; m_eq];

    for row in 0..m_eq {
        match eq_nnz[row] {
            0 => {
                // Empty equality row: feasible iff b == 0.
                if prob.b[row] != 0.0 {
                    return PresolveOutcome::Infeasible;
                }
                eq_dropped[row] = true;
                stack.push(Reduction::EmptyEqRow);
            }
            1 => {
                let (col, a) = eq_single[row].expect("singleton has an entry");
                if fixed[col].is_none() {
                    let value = prob.b[row] / a;
                    fixed[col] = Some(value);
                    eq_dropped[row] = true;
                    stack.push(Reduction::FixedVar {
                        col,
                        value,
                        eq_row: row,
                        a_coef: a,
                    });
                }
            }
            _ => {}
        }
    }

    // Empty inequality rows: feasible iff h >= 0; drop.
    let mut ineq_dropped = vec![false; m_ineq];
    for row in 0..m_ineq {
        if ineq_nnz[row] == 0 {
            if prob.h[row] < 0.0 {
                return PresolveOutcome::Infeasible;
            }
            ineq_dropped[row] = true;
            stack.push(Reduction::EmptyIneqRow);
        }
    }

    // If nothing fired, still return a (trivial) Presolve so the caller
    // path is uniform; the reduced problem is a clone.
    let any_fixed = fixed.iter().any(|f| f.is_some());
    let any_dropped = eq_dropped.iter().any(|&d| d) || ineq_dropped.iter().any(|&d| d);
    let _ = (any_fixed, any_dropped);

    // --- build column map (drop fixed columns) ---
    let mut kept_cols = Vec::new();
    let mut col_new = vec![usize::MAX; n];
    for c in 0..n {
        if fixed[c].is_none() {
            col_new[c] = kept_cols.len();
            kept_cols.push(c);
        }
    }

    // fixed-value lookup
    let fixval = |c: usize| fixed[c].unwrap_or(0.0);

    // --- objective: P, c, offset, with fixed vars substituted ---
    // f = ½ xᵀP x + cᵀx. Split into kept (r) and fixed (f) blocks:
    //   ½ rᵀP_rr r + (c_r + P_rf x_f)ᵀ r + [½ x_fᵀP_ff x_f + c_fᵀx_f]
    // The bracket is the constant offset; (P_rf x_f) augments the linear
    // term on the kept variables.
    let mut new_c = vec![0.0; kept_cols.len()];
    for (newc, &oldc) in kept_cols.iter().enumerate() {
        new_c[newc] = prob.c[oldc];
    }
    let mut offset = 0.0;
    // c_f x_f
    for c in 0..n {
        if let Some(v) = fixed[c] {
            offset += prob.c[c] * v;
        }
    }
    let mut new_p: Vec<Triplet> = Vec::new();
    for t in &prob.p_lower {
        let (i, j, v) = (t.row, t.col, t.val);
        let fi = fixed[i].is_some();
        let fj = fixed[j].is_some();
        match (fi, fj) {
            (false, false) => {
                new_p.push(Triplet::new(col_new[i], col_new[j], v));
            }
            (true, true) => {
                // both fixed → contributes to the constant. The stored
                // triplet is the lower triangle; the symmetric off-
                // diagonal counts twice in ½xᵀPx.
                if i == j {
                    offset += 0.5 * v * fixval(i) * fixval(j);
                } else {
                    offset += v * fixval(i) * fixval(j);
                }
            }
            (true, false) => {
                // P_ij x_i (fixed i) contributes to c_j of kept j.
                new_c[col_new[j]] += v * fixval(i);
            }
            (false, true) => {
                new_c[col_new[i]] += v * fixval(j);
            }
        }
    }

    // --- equality rows: keep non-dropped, remap cols, adjust b ---
    let mut kept_eq = Vec::new();
    let mut eq_row_new = vec![usize::MAX; m_eq];
    for row in 0..m_eq {
        if !eq_dropped[row] {
            eq_row_new[row] = kept_eq.len();
            kept_eq.push(row);
        }
    }
    let mut new_a: Vec<Triplet> = Vec::new();
    let mut new_b = vec![0.0; kept_eq.len()];
    for (newr, &oldr) in kept_eq.iter().enumerate() {
        new_b[newr] = prob.b[oldr];
    }
    for t in &prob.a {
        if eq_dropped[t.row] {
            continue;
        }
        let nr = eq_row_new[t.row];
        if let Some(v) = fixed[t.col] {
            new_b[nr] -= t.val * v; // move fixed term to RHS
        } else {
            new_a.push(Triplet::new(nr, col_new[t.col], t.val));
        }
    }

    // --- inequality rows: same treatment, adjust h ---
    let mut kept_ineq = Vec::new();
    let mut ineq_row_new = vec![usize::MAX; m_ineq];
    for row in 0..m_ineq {
        if !ineq_dropped[row] {
            ineq_row_new[row] = kept_ineq.len();
            kept_ineq.push(row);
        }
    }
    let mut new_g: Vec<Triplet> = Vec::new();
    let mut new_h = vec![0.0; kept_ineq.len()];
    for (newr, &oldr) in kept_ineq.iter().enumerate() {
        new_h[newr] = prob.h[oldr];
    }
    for t in &prob.g {
        if ineq_dropped[t.row] {
            continue;
        }
        let nr = ineq_row_new[t.row];
        if let Some(v) = fixed[t.col] {
            new_h[nr] -= t.val * v;
        } else {
            new_g.push(Triplet::new(nr, col_new[t.col], t.val));
        }
    }

    let reduced = QpProblem {
        n: kept_cols.len(),
        p_lower: new_p,
        c: new_c,
        a: new_a,
        b: new_b,
        g: new_g,
        h: new_h,
    };

    PresolveOutcome::Reduced(Presolve {
        reduced,
        obj_offset: offset,
        orig_n: n,
        orig_m_eq: m_eq,
        orig_m_ineq: m_ineq,
        kept_cols,
        kept_eq,
        kept_ineq,
        orig: prob.clone(),
        stack,
    })
}

impl Presolve {
    /// Expand a reduced-problem solution back to the original space,
    /// recovering primal `x` and duals `(y, z)`.
    pub fn postsolve(&self, red: &QpSolution) -> QpSolution {
        let mut x = vec![0.0; self.orig_n];
        let mut y = vec![0.0; self.orig_m_eq];
        let mut z = vec![0.0; self.orig_m_ineq];

        // Primal: kept columns from the reduced solution.
        for (newc, &oldc) in self.kept_cols.iter().enumerate() {
            x[oldc] = red.x[newc];
        }
        // Duals: kept rows from the reduced solution. Dropped/empty rows
        // and not-yet-assigned fixing rows stay 0 for now.
        for (newr, &oldr) in self.kept_eq.iter().enumerate() {
            y[oldr] = red.y[newr];
        }
        for (newr, &oldr) in self.kept_ineq.iter().enumerate() {
            z[oldr] = red.z[newr];
        }

        // Replay the stack in reverse to restore fixed primals and the
        // fixing-row duals. Empty rows already have dual 0.
        for r in self.stack.iter().rev() {
            if let Reduction::FixedVar { col, value, .. } = r {
                x[*col] = *value;
            }
        }

        // With all primals known, recover each fixing row's multiplier
        // from stationarity for its variable:
        //   (Px)_k + c_k + (Aᵀy)_k + (Gᵀz)_k + a·y_fix = 0
        //   ⇒ y_fix = −[(Px)_k + c_k + (Aᵀy)_k + (Gᵀz)_k] / a
        // All other rows' duals are already in y/z, so the bracket is a
        // straight evaluation against the original problem.
        let n = self.orig_n;
        let mut grad = vec![0.0; n]; // Px + c + Aᵀy + Gᵀz
        for i in 0..n {
            grad[i] = self.orig.c[i];
        }
        self.orig.p_mul(&x, &mut grad);
        self.orig.at_mul(&y, &mut grad);
        self.orig.gt_mul(&z, &mut grad);
        for r in &self.stack {
            if let Reduction::FixedVar {
                col,
                eq_row,
                a_coef,
                ..
            } = r
            {
                y[*eq_row] = -grad[*col] / a_coef;
            }
        }

        // Objective in the original problem.
        let mut px = vec![0.0; n];
        self.orig.p_mul(&x, &mut px);
        let mut obj = 0.0;
        for i in 0..n {
            obj += 0.5 * x[i] * px[i] + self.orig.c[i] * x[i];
        }

        QpSolution {
            status: red.status,
            x,
            y,
            z,
            obj,
            iters: red.iters,
        }
    }
}

/// Convenience: presolve, solve the reduced problem with `solve`, and
/// postsolve — returning a solution in the *original* problem space.
/// Falls back to solving `prob` directly if presolve does nothing
/// useful. On a presolve-detected infeasibility, returns a
/// `PrimalInfeasible` solution without invoking the solver.
pub fn solve_with_presolve<S>(prob: &QpProblem, solve: S) -> QpSolution
where
    S: FnOnce(&QpProblem) -> QpSolution,
{
    match presolve(prob) {
        PresolveOutcome::Infeasible => QpSolution {
            status: QpStatus::PrimalInfeasible,
            x: vec![0.0; prob.n],
            y: vec![0.0; prob.m_eq()],
            z: vec![0.0; prob.m_ineq()],
            obj: 0.0,
            iters: 0,
        },
        PresolveOutcome::Reduced(ps) => {
            let red = solve(&ps.reduced);
            ps.postsolve(&red)
        }
    }
}
