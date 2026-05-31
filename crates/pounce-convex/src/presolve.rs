//! Presolve for convex QP and LP (Phase 3.5).
//!
//! Reduces a [`QpProblem`] before the interior-point solve and maps the
//! reduced solution back to the original problem space, recovering both
//! the primal `x` and the duals `(y, z)`. The contract is correctness of
//! the recovered KKT point: a presolved-then-postsolved solve yields a
//! valid primal–dual solution of the *original* problem (see
//! `tests/presolve_roundtrip.rs` and `tests/presolve_reductions.rs`).
//!
//! This is the architectural seam the dev note calls the "missing
//! piece": a **transaction stack** of [`Reduction`]s, each carrying the
//! data needed to undo itself (primal *and* dual). Postsolve replays the
//! stack in reverse. The catalog is small but the postsolve is complete,
//! so richer reductions can be added without reworking the recovery path.
//!
//! Reductions implemented:
//! - **Empty rows** (equality / inequality with no nonzeros): a
//!   feasibility check, then drop. Their dual is zero. Detects trivial
//!   primal infeasibility (`0 = b≠0` or `0 ≤ h<0`).
//! - **Fixed-variable elimination** from a singleton equality row
//!   (`a·x_k = b ⇒ x_k = b/a`): substitute `x_k` out of `P`, `c`, `A`,
//!   `G` (adjusting the objective constant and the row right-hand
//!   sides), and recover the fixing row's multiplier from stationarity
//!   at the postsolved point. The QP-aware reduction (the Hessian
//!   coupling moves into the linear term and the dual must be recovered
//!   consistently with `P`).
//! - **Empty/free-column elimination**: a variable absent from `P`, `A`,
//!   and `G` is free and unconstrained, so its only objective effect is
//!   `c_k x_k`. If `c_k = 0` the variable is irrelevant (set to 0, drop);
//!   if `c_k ≠ 0` the problem is unbounded below (detected as
//!   [`PresolveOutcome::Unbounded`]).
//! - **Duplicate-row removal** (equality / inequality): rows with an
//!   identical coefficient pattern (after substitution) are redundant or
//!   expose infeasibility. Detection uses rayon-parallel per-row hashing
//!   (PaPILO's hashing-based pairing). Equality duplicates with differing
//!   right-hand sides ⇒ infeasible; inequality duplicates keep the
//!   tightest bound. A dropped duplicate's dual is zero (it is inactive
//!   / its share is carried by the kept row), which is a valid KKT point.
//! - **Free column singleton substitution**: an unbounded variable,
//!   absent from `P` and `G`, that appears in exactly one (multi-entry)
//!   equality row is substituted out via `x_col = (b_r − Σ_{j≠col} a_j
//!   x_j) / a_col`, eliminating both the variable *and* the row. The
//!   substitution shifts cost onto the surviving variables; the consumed
//!   row's multiplier is the unique value `y_r = −c_col / a_col`. This is
//!   a clean PaPILO reduction (uniquely determined dual), unlike forcing
//!   constraints / bound tightening.
//! - **Activity-bound reductions** (need the variable box): for each
//!   inequality `g·x ≤ h`, compute the activity range `[min, max]` over
//!   the box. If `max ≤ h` the row is always satisfied → **redundant**,
//!   drop it (dual 0). If `min > h` the row can never hold →
//!   **infeasible**. For each equality `a·x = b`, infeasible when `b`
//!   lies outside `[min, max]`. (Bound *tightening* — propagating a
//!   row's implied bound back onto a variable — is deferred: when a
//!   tightened bound is active at the optimum its multiplier must be
//!   re-attributed to the source constraint, which is the harder PaPILO
//!   postsolve; see the follow-up note below.)
//!
//! # Relationship to PaPILO
//!
//! [PaPILO](https://github.com/scipopt/papilo) (Gleixner, Gottwald &
//! Hoen; the presolving library SCIP uses) is the reference architecture
//! for this module. It is C++ and Apache-2.0, so POUNCE does **not** wrap
//! it — that would break the pure-Rust guarantee — but ports its ideas:
//!
//! - the **transaction/reduction-stack** model with reversible postsolve
//!   (the [`Reduction`] enum + `stack` + [`Presolve::postsolve`]);
//! - **hashing-based pairing** for duplicate detection, parallelized
//!   (PaPILO uses Intel TBB; we use rayon).
//!
//! PaPILO is the catalog to mine for the next reductions — singleton /
//! doubleton rows, dominated columns, coefficient strengthening, probing
//! — and, importantly, for each one's *postsolve transform*, since the
//! dual recovery is the hard part. Remaining activity-bound work that
//! needs more than the redundancy/feasibility checks above: **bound
//! tightening** (propagate a row's implied bound back onto a variable —
//! the dual of an active tightened bound must be re-attributed to the
//! source row in postsolve), **forcing constraints** (a row at its
//! activity extreme pins every involved variable to a bound), and
//! **dominated columns**. Parallel-row (scalar-multiple) detection, as
//! opposed to the exact-duplicate detection here, is likewise a
//! follow-up.

use crate::qp::{QpProblem, QpSolution, QpStatus, Triplet, BOUND_INF};
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Outcome of presolve.
pub enum PresolveOutcome {
    /// Problem reduced; solve `reduced`, then call [`Presolve::postsolve`].
    Reduced(Presolve),
    /// Presolve proved the problem primal-infeasible (e.g. an empty row
    /// `0 = b` with `b ≠ 0`, contradictory fixed bounds, or duplicate
    /// equality rows with different right-hand sides).
    Infeasible,
    /// Presolve proved the problem unbounded below (a free column with a
    /// nonzero objective coefficient).
    Unbounded,
}

/// A reversible presolve transaction. Each variant stores exactly what
/// postsolve needs to reconstruct the eliminated primal and dual.
///
/// Dropped *rows* (empty rows, duplicate rows) need no stack entry: they
/// are simply absent from the kept-row maps, so postsolve leaves their
/// dual at the zero initialization, which is the correct multiplier.
enum Reduction {
    /// Variable `col` was fixed to `value` by the singleton equality row
    /// `eq_row` (coefficient `a_coef`). Postsolve restores `x[col] =
    /// value` and computes the row's multiplier from stationarity.
    FixedVar {
        col: usize,
        value: f64,
        eq_row: usize,
        a_coef: f64,
    },
    /// A column absent from `P`, `A`, `G` (linear-only) was fixed at
    /// `value` — its optimal box position given the sign of `c_col` —
    /// and dropped. Its reduced cost equals `c_col` (carried by the
    /// active variable bound).
    FreeColumnFixed { col: usize, value: f64 },
    /// A *free column singleton*: variable `col` is unbounded, absent
    /// from `P` and `G`, and appears in exactly one equality row
    /// `eq_row` (coefficient `a_coef`). It is substituted out via
    /// `x_col = (b_r − Σ_{j≠col} a_j x_j) / a_coef`, consuming the row.
    /// Postsolve recovers `x_col` from that expression and sets the
    /// consumed row's multiplier to the unique value `y_r = −c_col / a_coef`.
    FreeColSingleton {
        col: usize,
        eq_row: usize,
        a_coef: f64,
        /// `c_col`, used to recover `y_eq_row = −c_col / a_coef`.
        c_col: f64,
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
    /// `kept_cols[reduced_col] = orig_col`.
    kept_cols: Vec<usize>,
    /// `kept_eq[reduced_eq_row] = orig_eq_row`.
    kept_eq: Vec<usize>,
    /// `kept_ineq[reduced_ineq_row] = orig_ineq_row`.
    kept_ineq: Vec<usize>,
    /// Original problem data, retained for fixing-row dual recovery.
    orig: QpProblem,
    stack: Vec<Reduction>,
}

/// Coefficients are treated as nonzero unless exactly 0.0.
const ZERO_TOL: f64 = 0.0;
/// Slack allowed when checking a fixed value against its variable box.
const BOUND_FEAS_TOL: f64 = 1e-9;
/// Slack allowed in activity-bound comparisons (redundancy / feasibility).
const ACTIVITY_TOL: f64 = 1e-9;

/// Group nonzero entries by row index: `out[row] = [(col, val), …]`.
fn group_by_row(triplets: &[Triplet], m: usize) -> Vec<Vec<(usize, f64)>> {
    let mut out = vec![Vec::new(); m];
    for t in triplets {
        if t.val != ZERO_TOL {
            out[t.row].push((t.col, t.val));
        }
    }
    out
}

/// Minimum and maximum of `Σ a_j x_j` over the variable box, given each
/// variable's effective lower/upper bound. An infinite contribution
/// makes the corresponding extreme `±∞`.
fn activity<L, U>(row: &[(usize, f64)], lb: &L, ub: &U) -> (f64, f64)
where
    L: Fn(usize) -> f64,
    U: Fn(usize) -> f64,
{
    let mut amin = 0.0;
    let mut amax = 0.0;
    for &(c, a) in row {
        let (lo, hi) = (lb(c), ub(c));
        if a > 0.0 {
            amin += a * lo; // a>0: min at lower bound
            amax += a * hi;
        } else {
            amin += a * hi; // a<0: min at upper bound
            amax += a * lo;
        }
    }
    (amin, amax)
}

/// A single constraint row in the reduced column space, tagged with its
/// original row index. Used for duplicate detection and final assembly.
struct Row {
    /// `(reduced_col, value)` pairs, sorted by column, duplicates merged.
    coeffs: Vec<(usize, f64)>,
    rhs: f64,
    orig: usize,
}

/// Run presolve on `prob`.
pub fn presolve(prob: &QpProblem) -> PresolveOutcome {
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();

    let mut stack: Vec<Reduction> = Vec::new();

    // --- per-row / per-column nonzero structure ---
    let mut eq_nnz = vec![0usize; m_eq];
    let mut eq_single: Vec<Option<(usize, f64)>> = vec![None; m_eq];
    // Finer per-column appearance counts: total (`col_nnz`), and split
    // by where the variable appears, so we can recognize a free *column
    // singleton* (a variable in exactly one equality row, nowhere else).
    let mut col_nnz = vec![0usize; n];
    let mut a_col_count = vec![0usize; n];
    let mut g_col_count = vec![0usize; n];
    let mut p_col_present = vec![false; n];
    // For a column singleton: which equality row holds it, with coef.
    let mut col_eq_single: Vec<Option<(usize, f64)>> = vec![None; n];
    for t in &prob.a {
        if t.val != ZERO_TOL {
            eq_nnz[t.row] += 1;
            eq_single[t.row] = Some((t.col, t.val));
            col_nnz[t.col] += 1;
            a_col_count[t.col] += 1;
            col_eq_single[t.col] = Some((t.row, t.val));
        }
    }
    let mut ineq_nnz = vec![0usize; m_ineq];
    for t in &prob.g {
        if t.val != ZERO_TOL {
            ineq_nnz[t.row] += 1;
            col_nnz[t.col] += 1;
            g_col_count[t.col] += 1;
        }
    }
    for t in &prob.p_lower {
        if t.val != ZERO_TOL {
            col_nnz[t.row] += 1;
            p_col_present[t.row] = true;
            if t.row != t.col {
                col_nnz[t.col] += 1;
                p_col_present[t.col] = true;
            }
        }
    }

    // --- empty equality rows + singleton-equality fixings ---
    let mut fixed: Vec<Option<f64>> = vec![None; n];
    let mut eq_dropped = vec![false; m_eq];
    for row in 0..m_eq {
        match eq_nnz[row] {
            0 => {
                if prob.b[row] != 0.0 {
                    return PresolveOutcome::Infeasible;
                }
                eq_dropped[row] = true;
            }
            1 => {
                let (col, a) = eq_single[row].expect("singleton has an entry");
                if fixed[col].is_none() {
                    let value = prob.b[row] / a;
                    // The fixed value must satisfy the variable's box.
                    if value < prob.lb_of(col) - BOUND_FEAS_TOL
                        || value > prob.ub_of(col) + BOUND_FEAS_TOL
                    {
                        return PresolveOutcome::Infeasible;
                    }
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

    // --- free column singletons ---
    // A free variable (unbounded both ways), absent from P and G, that
    // appears in exactly one equality row whose row has ≥ 2 nonzeros, is
    // substituted out: `x_col = (b_r − Σ_{j≠col} a_j x_j) / a_col`. This
    // consumes both the variable and the row. The substitution shifts the
    // cost of the row's other variables (`c_adjust`) and a constant into
    // the objective offset; the consumed row's dual is the unique value
    // `−c_col / a_col`, recovered in postsolve.
    let mut substituted = vec![false; n];
    let mut c_adjust = vec![0.0; n];
    let mut subst_offset = 0.0;
    for col in 0..n {
        if fixed[col].is_some() || substituted[col] {
            continue;
        }
        let free = prob.lb_of(col) <= -BOUND_INF && prob.ub_of(col) >= BOUND_INF;
        let only_in_one_eq = a_col_count[col] == 1 && g_col_count[col] == 0 && !p_col_present[col];
        if !(free && only_in_one_eq) {
            continue;
        }
        let (row, a_col) = col_eq_single[col].expect("column singleton entry");
        // The row must still be live and non-trivial (≥ 2 vars: a plain
        // singleton row was already turned into a FixedVar above).
        if eq_dropped[row] || eq_nnz[row] < 2 {
            continue;
        }
        // Substitute: c_col·x_col = (c_col·b_r/a_col) − Σ_{j≠col}
        // (c_col·a_jr/a_col)·x_j.
        let c_col = prob.c[col];
        subst_offset += c_col * prob.b[row] / a_col;
        for t in &prob.a {
            if t.row == row && t.col != col && t.val != ZERO_TOL {
                c_adjust[t.col] -= c_col * t.val / a_col;
            }
        }
        substituted[col] = true;
        eq_dropped[row] = true;
        stack.push(Reduction::FreeColSingleton {
            col,
            eq_row: row,
            a_coef: a_col,
            c_col,
        });
    }

    // --- empty inequality rows ---
    let mut ineq_dropped = vec![false; m_ineq];
    for row in 0..m_ineq {
        if ineq_nnz[row] == 0 {
            if prob.h[row] < 0.0 {
                return PresolveOutcome::Infeasible;
            }
            ineq_dropped[row] = true;
        }
    }

    // --- activity-bound reductions (need the variable box) ---
    // Effective bounds: a fixed variable contributes its exact value;
    // others contribute their declared box (±∞ when absent).
    let eff_lb = |c: usize| fixed[c].unwrap_or_else(|| prob.lb_of(c));
    let eff_ub = |c: usize| fixed[c].unwrap_or_else(|| prob.ub_of(c));

    // Group nonzeros by row once, reused for inequalities and equalities.
    let g_by_row = group_by_row(&prob.g, m_ineq);
    let a_by_row = group_by_row(&prob.a, m_eq);

    // Inequality `g·x ≤ h`:
    //   max-activity ≤ h  ⇒ redundant (always satisfied) → drop;
    //   min-activity > h   ⇒ infeasible.
    for row in 0..m_ineq {
        if ineq_dropped[row] {
            continue;
        }
        let (amin, amax) = activity(&g_by_row[row], &eff_lb, &eff_ub);
        if amin > prob.h[row] + ACTIVITY_TOL {
            return PresolveOutcome::Infeasible;
        }
        if amax <= prob.h[row] + ACTIVITY_TOL {
            ineq_dropped[row] = true;
        }
    }

    // Equality `a·x = b`: feasible only if `b` lies in the activity
    // range `[min, max]`. Out of range ⇒ infeasible. (A redundant
    // equality whose range is the single point `b` is left in place; its
    // dual is genuine, unlike a dropped inequality's zero multiplier.)
    for row in 0..m_eq {
        if eq_dropped[row] {
            continue;
        }
        let (amin, amax) = activity(&a_by_row[row], &eff_lb, &eff_ub);
        if prob.b[row] < amin - ACTIVITY_TOL || prob.b[row] > amax + ACTIVITY_TOL {
            return PresolveOutcome::Infeasible;
        }
    }

    // --- free / linear-only columns ---
    // A column absent from P, A, G contributes only `c_k x_k`, so its
    // optimum is at a bound dictated by the sign of c_k:
    //   c_k > 0 → minimize by pushing to lb  (unbounded if lb = −∞)
    //   c_k < 0 → push to ub                 (unbounded if ub = +∞)
    //   c_k = 0 → irrelevant; pin to lb if finite else ub if finite else 0
    let mut dropped_col = vec![false; n];
    for c in 0..n {
        if fixed[c].is_some() || substituted[c] {
            dropped_col[c] = true; // fixed / substituted columns are removed
            continue;
        }
        if col_nnz[c] == 0 {
            let (lb, ub) = (prob.lb_of(c), prob.ub_of(c));
            let value = if prob.c[c] > 0.0 {
                if lb <= -BOUND_INF {
                    return PresolveOutcome::Unbounded;
                }
                lb
            } else if prob.c[c] < 0.0 {
                if ub >= BOUND_INF {
                    return PresolveOutcome::Unbounded;
                }
                ub
            } else if lb > -BOUND_INF {
                lb
            } else if ub < BOUND_INF {
                ub
            } else {
                0.0
            };
            dropped_col[c] = true;
            stack.push(Reduction::FreeColumnFixed { col: c, value });
        }
    }

    // --- column map over surviving columns ---
    let mut kept_cols = Vec::new();
    let mut col_new = vec![usize::MAX; n];
    for c in 0..n {
        if !dropped_col[c] {
            col_new[c] = kept_cols.len();
            kept_cols.push(c);
        }
    }
    let fixval = |c: usize| fixed[c].unwrap_or(0.0);

    // --- objective: P, c, offset with fixed vars substituted ---
    // Surviving variables' linear cost is their original `c` plus any
    // cost shifted onto them by a free-column-singleton substitution.
    let mut new_c = vec![0.0; kept_cols.len()];
    for (newc, &oldc) in kept_cols.iter().enumerate() {
        new_c[newc] = prob.c[oldc] + c_adjust[oldc];
    }
    let mut offset = subst_offset;
    for c in 0..n {
        if let Some(v) = fixed[c] {
            offset += prob.c[c] * v;
        }
    }
    // Free/linear-only columns fixed to a bound contribute `c_k · value`.
    for r in &stack {
        if let Reduction::FreeColumnFixed { col, value } = r {
            offset += prob.c[*col] * value;
        }
    }
    let mut new_p: Vec<Triplet> = Vec::new();
    for t in &prob.p_lower {
        let (i, j, v) = (t.row, t.col, t.val);
        match (fixed[i].is_some(), fixed[j].is_some()) {
            (false, false) => new_p.push(Triplet::new(col_new[i], col_new[j], v)),
            (true, true) => {
                // both fixed → constant. Off-diagonal counts twice.
                if i == j {
                    offset += 0.5 * v * fixval(i) * fixval(j);
                } else {
                    offset += v * fixval(i) * fixval(j);
                }
            }
            (true, false) => new_c[col_new[j]] += v * fixval(i),
            (false, true) => new_c[col_new[i]] += v * fixval(j),
        }
    }

    // --- build reduced rows (after substitution), then dedup ---
    let eq_rows = match build_rows(&prob.a, m_eq, &eq_dropped, &prob.b, &fixed, &col_new, true) {
        Ok(rows) => rows,
        Err(()) => return PresolveOutcome::Infeasible,
    };
    let ineq_rows = match build_rows(
        &prob.g,
        m_ineq,
        &ineq_dropped,
        &prob.h,
        &fixed,
        &col_new,
        false,
    ) {
        Ok(rows) => rows,
        Err(()) => return PresolveOutcome::Infeasible,
    };

    let eq_rows = match dedup_rows(eq_rows, true) {
        Ok(rows) => rows,
        Err(()) => return PresolveOutcome::Infeasible,
    };
    let ineq_rows = dedup_rows(ineq_rows, false).expect("ineq dedup never infeasible");

    // --- flatten surviving rows to triplets + kept-row maps ---
    let mut kept_eq = Vec::with_capacity(eq_rows.len());
    let mut new_a = Vec::new();
    let mut new_b = vec![0.0; eq_rows.len()];
    for (newr, row) in eq_rows.iter().enumerate() {
        kept_eq.push(row.orig);
        new_b[newr] = row.rhs;
        for &(c, v) in &row.coeffs {
            new_a.push(Triplet::new(newr, c, v));
        }
    }
    let mut kept_ineq = Vec::with_capacity(ineq_rows.len());
    let mut new_g = Vec::new();
    let mut new_h = vec![0.0; ineq_rows.len()];
    for (newr, row) in ineq_rows.iter().enumerate() {
        kept_ineq.push(row.orig);
        new_h[newr] = row.rhs;
        for &(c, v) in &row.coeffs {
            new_g.push(Triplet::new(newr, c, v));
        }
    }

    // Carry the kept columns' bounds into the reduced problem (empty if
    // none of the kept variables is bounded).
    let (new_lb, new_ub) = if prob.has_bounds() {
        (
            kept_cols.iter().map(|&c| prob.lb_of(c)).collect(),
            kept_cols.iter().map(|&c| prob.ub_of(c)).collect(),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    let reduced = QpProblem {
        n: kept_cols.len(),
        p_lower: new_p,
        c: new_c,
        a: new_a,
        b: new_b,
        g: new_g,
        h: new_h,
        lb: new_lb,
        ub: new_ub,
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

/// Build per-row coefficient lists in the reduced column space,
/// substituting fixed variables into the right-hand side. Rows that
/// become empty after substitution trigger a feasibility check:
/// `0 = rhs` (equality) requires `rhs == 0`; `0 ≤ rhs` (inequality)
/// requires `rhs ≥ 0`. Returns `Err(())` on detected infeasibility.
fn build_rows(
    triplets: &[Triplet],
    m: usize,
    dropped: &[bool],
    base_rhs: &[f64],
    fixed: &[Option<f64>],
    col_new: &[usize],
    is_equality: bool,
) -> Result<Vec<Row>, ()> {
    let mut acc: Vec<Option<Row>> = (0..m)
        .map(|r| {
            if dropped[r] {
                None
            } else {
                Some(Row {
                    coeffs: Vec::new(),
                    rhs: base_rhs[r],
                    orig: r,
                })
            }
        })
        .collect();

    for t in triplets {
        if dropped[t.row] || t.val == ZERO_TOL {
            continue;
        }
        let row = acc[t.row].as_mut().expect("non-dropped row");
        if let Some(v) = fixed[t.col] {
            row.rhs -= t.val * v;
        } else {
            row.coeffs.push((col_new[t.col], t.val));
        }
    }

    let mut out = Vec::new();
    for row in acc.into_iter().flatten() {
        let mut row = row;
        merge_sort_coeffs(&mut row.coeffs);
        if row.coeffs.is_empty() {
            // Row reduced to `0 (cmp) rhs`: a feasibility check.
            if is_equality {
                if row.rhs.abs() > 0.0 {
                    return Err(());
                }
            } else if row.rhs < 0.0 {
                return Err(());
            }
            // Feasible empty row: drop it (no coefficients, no dual).
            continue;
        }
        out.push(row);
    }
    Ok(out)
}

/// Sort coefficients by column and merge any duplicate columns (a
/// variable appearing twice in one row). Drops entries that cancel to 0.
fn merge_sort_coeffs(coeffs: &mut Vec<(usize, f64)>) {
    coeffs.sort_by_key(|&(c, _)| c);
    let mut merged: Vec<(usize, f64)> = Vec::with_capacity(coeffs.len());
    for &(c, v) in coeffs.iter() {
        if let Some(last) = merged.last_mut() {
            if last.0 == c {
                last.1 += v;
                continue;
            }
        }
        merged.push((c, v));
    }
    merged.retain(|&(_, v)| v != 0.0);
    *coeffs = merged;
}

/// Hash a row's coefficient pattern (`(col, value-bits)`), canonicalized
/// by [`merge_sort_coeffs`]. Two rows collide here iff they have the same
/// coefficient pattern (modulo the negligible hash-collision rate, which
/// the caller guards against by comparing patterns directly).
fn row_signature(row: &Row) -> u64 {
    let mut h = DefaultHasher::new();
    row.coeffs.len().hash(&mut h);
    for &(c, v) in &row.coeffs {
        c.hash(&mut h);
        v.to_bits().hash(&mut h);
    }
    h.finish()
}

/// Exact coefficient-pattern equality (values compared bit-for-bit).
fn same_pattern(a: &Row, b: &Row) -> bool {
    a.coeffs.len() == b.coeffs.len()
        && a.coeffs
            .iter()
            .zip(&b.coeffs)
            .all(|(&(ca, va), &(cb, vb))| ca == cb && va.to_bits() == vb.to_bits())
}

/// Remove duplicate rows (identical coefficient pattern). Signatures are
/// computed in parallel (rayon); grouping and the per-group decision are
/// serial and cheap. For `is_equality`, duplicates with differing rhs are
/// infeasible (`Err(())`); otherwise keep the first. For inequalities,
/// keep the tightest (smallest rhs) of each duplicate group.
fn dedup_rows(rows: Vec<Row>, is_equality: bool) -> Result<Vec<Row>, ()> {
    if rows.len() < 2 {
        return Ok(rows);
    }

    // Parallel: one signature per row (PaPILO-style hashing-based pairing).
    let sigs: Vec<u64> = rows.par_iter().map(row_signature).collect();

    // Group row indices by signature (serial; small).
    let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &s) in sigs.iter().enumerate() {
        buckets.entry(s).or_default().push(i);
    }

    let mut keep = vec![true; rows.len()];
    for idxs in buckets.values() {
        if idxs.len() < 2 {
            continue;
        }
        // Within a signature bucket, partition into confirmed-equal
        // pattern groups (guards against hash collisions).
        let mut handled = vec![false; idxs.len()];
        for a in 0..idxs.len() {
            if handled[a] {
                continue;
            }
            // Collect all members sharing the pattern of idxs[a].
            let mut group = vec![idxs[a]];
            handled[a] = true;
            for b in (a + 1)..idxs.len() {
                if !handled[b] && same_pattern(&rows[idxs[a]], &rows[idxs[b]]) {
                    handled[b] = true;
                    group.push(idxs[b]);
                }
            }
            if group.len() < 2 {
                continue;
            }
            if is_equality {
                // Same lhs: all rhs must agree, else infeasible.
                let r0 = rows[group[0]].rhs;
                for &g in &group[1..] {
                    if (rows[g].rhs - r0).abs() > 0.0 {
                        return Err(());
                    }
                }
                // Keep the first, drop the rest.
                for &g in &group[1..] {
                    keep[g] = false;
                }
            } else {
                // Keep the tightest (smallest rhs); drop the rest.
                let tightest = *group
                    .iter()
                    .min_by(|&&p, &&q| rows[p].rhs.partial_cmp(&rows[q].rhs).unwrap())
                    .unwrap();
                for &g in &group {
                    if g != tightest {
                        keep[g] = false;
                    }
                }
            }
        }
    }

    Ok(rows
        .into_iter()
        .zip(keep)
        .filter_map(|(r, k)| if k { Some(r) } else { None })
        .collect())
}

/// Summary of what presolve removed, for logging and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PresolveStats {
    /// Variables in the original problem.
    pub orig_vars: usize,
    /// Variables in the reduced problem.
    pub reduced_vars: usize,
    /// Equality + inequality rows in the original problem.
    pub orig_rows: usize,
    /// Equality + inequality rows in the reduced problem.
    pub reduced_rows: usize,
    /// Variables fixed by a singleton equality row.
    pub fixed_vars: usize,
    /// Free / linear-only columns pinned to a bound and dropped.
    pub free_cols_fixed: usize,
    /// Free column singletons substituted out (each also removes a row).
    pub free_col_singletons: usize,
}

impl PresolveStats {
    /// Did presolve remove anything?
    pub fn reduced_anything(&self) -> bool {
        self.reduced_vars < self.orig_vars || self.reduced_rows < self.orig_rows
    }
}

impl Presolve {
    /// Reduction summary (sizes before/after and counts by reduction).
    pub fn stats(&self) -> PresolveStats {
        let mut s = PresolveStats {
            orig_vars: self.orig_n,
            reduced_vars: self.reduced.n,
            orig_rows: self.orig_m_eq + self.orig_m_ineq,
            reduced_rows: self.reduced.m_eq() + self.reduced.m_ineq(),
            ..Default::default()
        };
        for r in &self.stack {
            match r {
                Reduction::FixedVar { .. } => s.fixed_vars += 1,
                Reduction::FreeColumnFixed { .. } => s.free_cols_fixed += 1,
                Reduction::FreeColSingleton { .. } => s.free_col_singletons += 1,
            }
        }
        s
    }

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
        // Duals: kept rows from the reduced solution. Dropped rows
        // (empty / duplicate) stay 0, which is their correct multiplier.
        for (newr, &oldr) in self.kept_eq.iter().enumerate() {
            y[oldr] = red.y[newr];
        }
        for (newr, &oldr) in self.kept_ineq.iter().enumerate() {
            z[oldr] = red.z[newr];
        }

        // Restore eliminated primals (reverse order, so a substitution's
        // dependencies are already in place). Fixed and free-fixed columns
        // take their stored value; a free-column-singleton is recovered
        // from its consumed equality row using the other variables.
        for r in self.stack.iter().rev() {
            match r {
                Reduction::FixedVar { col, value, .. } => x[*col] = *value,
                Reduction::FreeColumnFixed { col, value } => x[*col] = *value,
                Reduction::FreeColSingleton {
                    col,
                    eq_row,
                    a_coef,
                    ..
                } => {
                    // x_col = (b_r − Σ_{j≠col} a_jr x_j) / a_col.
                    let mut acc = self.orig.b[*eq_row];
                    for t in &self.orig.a {
                        if t.row == *eq_row && t.col != *col {
                            acc -= t.val * x[t.col];
                        }
                    }
                    x[*col] = acc / a_coef;
                }
            }
        }

        // Free-column-singleton consumed-row multipliers have the unique
        // value y_r = −c_col / a_col (from stationarity of the eliminated
        // free variable, which has no P/G terms).
        for r in &self.stack {
            if let Reduction::FreeColSingleton {
                eq_row,
                a_coef,
                c_col,
                ..
            } = r
            {
                y[*eq_row] = -c_col / a_coef;
            }
        }

        // Recover each fixing row's multiplier from stationarity for its
        // variable: with all primals and other duals known,
        //   (Px)_k + c_k + (Aᵀy)_k + (Gᵀz)_k + a·y_fix = 0
        //   ⇒ y_fix = −[(Px)_k + c_k + (Aᵀy)_k + (Gᵀz)_k] / a.
        let n = self.orig_n;
        let mut grad = vec![0.0; n];
        grad[..n].copy_from_slice(&self.orig.c[..n]);
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

        // Bound multipliers: map the reduced kept-column bound duals back,
        // then attribute each free/linear-only fixed column's reduced cost
        // (= c_k) to whichever bound it was pinned at.
        let mut z_lb = vec![0.0; n];
        let mut z_ub = vec![0.0; n];
        for (newc, &oldc) in self.kept_cols.iter().enumerate() {
            if newc < red.z_lb.len() {
                z_lb[oldc] = red.z_lb[newc];
            }
            if newc < red.z_ub.len() {
                z_ub[oldc] = red.z_ub[newc];
            }
        }
        for r in &self.stack {
            if let Reduction::FreeColumnFixed { col, value } = r {
                let ck = self.orig.c[*col];
                // Stationarity for a linear-only var at a bound:
                //   c_k − z_lb + z_ub = 0. At lb: z_lb = c_k (c_k ≥ 0);
                //   at ub: z_ub = −c_k (c_k ≤ 0).
                if (*value - self.orig.lb_of(*col)).abs() <= (*value - self.orig.ub_of(*col)).abs()
                {
                    z_lb[*col] = ck.max(0.0);
                } else {
                    z_ub[*col] = (-ck).max(0.0);
                }
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
            z_lb,
            z_ub,
            obj,
            iters: red.iters,
        }
    }
}

/// Convenience: presolve, solve the reduced problem with `solve`, and
/// postsolve — returning a solution in the *original* problem space. On a
/// presolve-detected infeasibility / unboundedness, returns the matching
/// status without invoking the solver.
pub fn solve_with_presolve<S>(prob: &QpProblem, solve: S) -> QpSolution
where
    S: FnOnce(&QpProblem) -> QpSolution,
{
    let trivial = |status| QpSolution {
        status,
        x: vec![0.0; prob.n],
        y: vec![0.0; prob.m_eq()],
        z: vec![0.0; prob.m_ineq()],
        z_lb: vec![0.0; prob.n],
        z_ub: vec![0.0; prob.n],
        obj: 0.0,
        iters: 0,
    };
    match presolve(prob) {
        PresolveOutcome::Infeasible => trivial(QpStatus::PrimalInfeasible),
        PresolveOutcome::Unbounded => trivial(QpStatus::DualInfeasible),
        PresolveOutcome::Reduced(ps) => {
            let red = solve(&ps.reduced);
            ps.postsolve(&red)
        }
    }
}
