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
//! dual recovery is the hard part. The activity-bound–based reductions
//! (forcing / dominated constraints, bound tightening) require an
//! explicit variable-bound form; the standard form here encodes bounds
//! as `G` rows, so those land once a bounded form is added. Parallel-row
//! (scalar-multiple) detection, as opposed to the exact-duplicate
//! detection here, is likewise a follow-up.

use crate::qp::{QpProblem, QpSolution, QpStatus, Triplet};
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
    /// A free column (absent from `P`, `A`, `G`) with zero objective
    /// coefficient was set to 0 and dropped. Reduced cost is 0.
    FreeColumnZero { col: usize },
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
    let mut col_nnz = vec![0usize; n];
    for t in &prob.a {
        if t.val != ZERO_TOL {
            eq_nnz[t.row] += 1;
            eq_single[t.row] = Some((t.col, t.val));
            col_nnz[t.col] += 1;
        }
    }
    let mut ineq_nnz = vec![0usize; m_ineq];
    for t in &prob.g {
        if t.val != ZERO_TOL {
            ineq_nnz[t.row] += 1;
            col_nnz[t.col] += 1;
        }
    }
    for t in &prob.p_lower {
        if t.val != ZERO_TOL {
            col_nnz[t.row] += 1;
            if t.row != t.col {
                col_nnz[t.col] += 1;
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

    // --- free/empty columns ---
    // A column absent from P, A, G is free; its only effect is c_k x_k.
    let mut dropped_col = vec![false; n];
    for c in 0..n {
        if fixed[c].is_some() {
            dropped_col[c] = true; // fixed columns are removed too
            continue;
        }
        if col_nnz[c] == 0 {
            if prob.c[c] != 0.0 {
                return PresolveOutcome::Unbounded;
            }
            // c_k == 0: variable is irrelevant; pin to 0 and drop.
            dropped_col[c] = true;
            stack.push(Reduction::FreeColumnZero { col: c });
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
    let mut new_c = vec![0.0; kept_cols.len()];
    for (newc, &oldc) in kept_cols.iter().enumerate() {
        new_c[newc] = prob.c[oldc];
    }
    let mut offset = 0.0;
    for c in 0..n {
        if let Some(v) = fixed[c] {
            offset += prob.c[c] * v;
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
        // Duals: kept rows from the reduced solution. Dropped rows
        // (empty / duplicate) stay 0, which is their correct multiplier.
        for (newr, &oldr) in self.kept_eq.iter().enumerate() {
            y[oldr] = red.y[newr];
        }
        for (newr, &oldr) in self.kept_ineq.iter().enumerate() {
            z[oldr] = red.z[newr];
        }

        // Restore eliminated primals: fixed vars and zeroed free columns.
        for r in self.stack.iter().rev() {
            match r {
                Reduction::FixedVar { col, value, .. } => x[*col] = *value,
                Reduction::FreeColumnZero { col } => x[*col] = 0.0,
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
