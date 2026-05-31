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
//! - **Parallel-row removal** (equality / inequality): rows that are
//!   **scalar multiples** of one another (after substitution) — exact
//!   duplicates being the unit-scale case — are redundant or expose
//!   infeasibility. Detection normalizes each row by a canonical pivot and
//!   uses rayon-parallel per-row hashing (PaPILO's hashing-based pairing),
//!   confirming candidates with a tolerance so a wrong merge is
//!   impossible (a quantization split only ever *misses* a pair).
//!   Parallel equalities with inconsistent (scaled) right-hand sides ⇒
//!   infeasible; parallel inequalities (positive multiples — same
//!   direction) keep the most restrictive row. Dual recovery stays
//!   trivial because the *kept* row is an original one in its own frame
//!   and every dropped row's multiplier is zero — a valid KKT point.
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
//!   lies outside `[min, max]`.
//! - **Dominated columns**: a variable absent from `P` and the equalities
//!   that appears in inequalities `Gx ≤ h` with sign-definite coefficients
//!   matching its cost sign is optimal at a bound (pushing it there raises
//!   neither the objective nor any row's activity), so it is fixed and
//!   dropped. Its bound multiplier is its reduced cost `c_k + Σᵢ aᵢₖ zᵢ`,
//!   which the sign conditions make nonnegative — a valid dual by
//!   construction. (PaPILO's dominated-column reduction, restricted to the
//!   clean sign-guaranteed case.)
//! - **Forcing constraints**: when a row's activity range *touches* its
//!   right-hand side it can hold only at one vertex of the box, pinning
//!   every involved variable to a bound (inequality `g·x ≤ h` with
//!   `min = h`; equality `a·x = b` with `min = b` or `max = b`). The row
//!   is dropped and each variable fixed. The dual recovery — the reason
//!   this was the hard PaPILO postsolve — is exact: the forcing row's
//!   multiplier is the tightest value making every pinned variable's bound
//!   multiplier correctly signed (`max`/`min` over `−gradⱼ/coefⱼ`, clamped
//!   `≥ 0` for inequalities), and each pinned variable's bound multiplier
//!   is then its full reduced cost. The multiplier is generally *not
//!   unique* (it ranges over an interval), so postsolve emits a valid
//!   representative; correctness is checked as KKT validity, not dual
//!   equality (`tests/presolve_forcing.rs`). Forcing rows are required to
//!   have disjoint column sets so the recovery stays independent.
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
//! dual recovery is the hard part.
//!
//! Implemented from that catalog so far: the transaction stack, fixed /
//! free / free-singleton columns, empty + duplicate rows, activity-based
//! redundancy/feasibility, and **forcing constraints** (above) — which
//! capture the dual-safe slice of activity/bound reasoning, since a
//! forcing row is exactly a model-changing bound deduction whose dual
//! re-attributes to the source row.
//!
//! - **Bound tightening** (domain propagation): each live row implies
//!   bounds on its variables (`a_k x_k ≤ h − amin_{−k}`, etc.); where one
//!   is strictly tighter than the declared box, the box is shrunk in the
//!   reduced problem (the variable is *kept*). The subtle dual — when a
//!   tightened bound is active at the optimum while the original bound is
//!   slack, its multiplier is not a real bound multiplier but belongs to
//!   the row that implied it — is handled in postsolve by **global bound
//!   recovery**: every row multiplier is recovered first (re-attributing
//!   each active tightened bound to its source row), then every variable's
//!   bound multipliers are read off the complete reduced cost by
//!   complementarity. To keep the re-attributions independent, tightening
//!   sources are restricted to column-disjoint rows untouched by other
//!   reductions (the same conservative rule as forcing). A single pass
//!   (not iterated to a fixpoint), validated by randomized KKT roundtrips
//!   (`tests/presolve_bound_tightening.rs`).
//!
//! The full deferred catalog — forcing constraints, parallel rows,
//! dominated columns, and bound tightening — is implemented, each with a
//! dual recovery proven correct (and KKT-validated in tests).
//!
//! [`presolve`] iterates the single-pass catalog ([`presolve_once`]) to a
//! **fixpoint**, so deductions cascade across rounds (a fixing exposes a
//! new singleton; a tightened bound makes a row forcing; a forcing that
//! shares a column with another — disallowed in one round — fires the next
//! once the shared variable is gone). Because each pass is a correct
//! solution-space transform, the iterate is their composition and reuses
//! every pass's proven dual recovery. The disjoint-source restriction on
//! forcing / tightening within a single round therefore costs little: the
//! fixpoint progressively handles overlaps that a single round defers.

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
    /// A **forcing constraint**: a row whose activity range touches its
    /// right-hand side, so the row can only hold at one vertex of the box,
    /// pinning every involved variable to a bound. The row is dropped and
    /// each variable fixed; postsolve recovers the row's multiplier and the
    /// pinned variables' bound multipliers (see [`Presolve::postsolve`]).
    ForcingRow {
        /// Original row index.
        row: usize,
        /// Equality row? (else inequality.)
        is_equality: bool,
        /// The forced-to vertex is the *max*-activity one (only possible
        /// for equalities); else the min-activity vertex.
        at_max: bool,
        /// Each pinned variable: `(col, coef, value, at_upper)`.
        cols: Vec<(usize, f64, f64, bool)>,
    },
    /// A **dominated column**: a variable absent from `P` and the
    /// equalities, appearing in inequalities `Gx ≤ h` with sign-definite
    /// coefficients that match the sign of its cost, so pushing it to one
    /// bound never hurts the objective *or* feasibility — it is optimal
    /// there. Fixed and dropped; its bound multiplier is its reduced cost,
    /// which the sign conditions make valid by construction (recovered in
    /// the global bound pass from where the variable lands).
    DominatedColumn { col: usize, value: f64 },
    /// A **tightened bound**: row `row` implies a bound on `col` strictly
    /// inside its declared box, so the box is shrunk in the reduced problem
    /// (the variable is *kept*, not removed). Postsolve handles the dual:
    /// if the tightened bound is active at the optimum while the original
    /// bound is slack, its multiplier is re-attributed to the source row
    /// (the multiplier on a non-real bound belongs to the constraint that
    /// implied it). See [`Presolve::postsolve`]'s global bound recovery.
    BoundTightening {
        col: usize,
        row: usize,
        is_equality: bool,
        /// Source-row coefficient `a_{row,col}`.
        coef: f64,
        /// Tightened the upper bound? (else lower.)
        is_upper: bool,
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
    /// For an *iterated* presolve, the ordered single-pass layers
    /// (`L0, L1, …`) whose composition this object represents; empty for a
    /// single pass. `reduced` is then the final layer's reduced problem and
    /// `postsolve` folds the layers in reverse. The single-pass fields
    /// above are unused in that case.
    chain: Vec<Presolve>,
}

/// Coefficients are treated as nonzero unless exactly 0.0.
const ZERO_TOL: f64 = 0.0;
/// Slack allowed when checking a fixed value against its variable box.
const BOUND_FEAS_TOL: f64 = 1e-9;
/// Slack allowed in activity-bound comparisons (redundancy / feasibility).
const ACTIVITY_TOL: f64 = 1e-9;
/// How close `x_i` must be to a box bound to count it *active* when
/// recovering bound multipliers. Looser than [`BOUND_FEAS_TOL`] because an
/// interior-point solve only drives a variable to within ~1e-8 of a bound,
/// not to machine zero; interior variables sit far further away.
const ACTIVE_BOUND_TOL: f64 = 1e-6;

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

/// Run presolve on `prob`, iterating the reduction passes to a **fixpoint**
/// so deductions cascade (a fixing exposes a new singleton, a tightened
/// bound makes a row forcing, …). Each pass is a correct solution-space
/// transform, so the iterate is the composition of the per-pass transforms
/// — postsolve folds them back in reverse — and inherits each pass's proven
/// dual recovery with no new dual math.
pub fn presolve(prob: &QpProblem) -> PresolveOutcome {
    // Cap rounds defensively; in practice it converges in a few.
    const MAX_ROUNDS: usize = 32;
    let mut chain: Vec<Presolve> = Vec::new();
    let mut current = prob.clone();
    loop {
        match presolve_once(&current) {
            PresolveOutcome::Infeasible => return PresolveOutcome::Infeasible,
            PresolveOutcome::Unbounded => return PresolveOutcome::Unbounded,
            PresolveOutcome::Reduced(ps) => {
                if !ps.changed() {
                    // Fixpoint: this round did nothing.
                    if chain.is_empty() {
                        return PresolveOutcome::Reduced(ps); // plain single pass
                    }
                    break;
                }
                current = ps.reduced.clone();
                chain.push(ps);
                if chain.len() >= MAX_ROUNDS {
                    break;
                }
            }
        }
    }
    if chain.len() == 1 {
        return PresolveOutcome::Reduced(chain.pop().unwrap());
    }
    let reduced = chain.last().expect("chain non-empty").reduced.clone();
    PresolveOutcome::Reduced(Presolve {
        reduced,
        obj_offset: 0.0,
        orig_n: prob.n,
        orig_m_eq: prob.m_eq(),
        orig_m_ineq: prob.m_ineq(),
        kept_cols: Vec::new(),
        kept_eq: Vec::new(),
        kept_ineq: Vec::new(),
        orig: prob.clone(),
        stack: Vec::new(),
        chain,
    })
}

/// A single presolve pass (the reduction catalog applied once). [`presolve`]
/// iterates this to a fixpoint.
fn presolve_once(prob: &QpProblem) -> PresolveOutcome {
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

    // --- forcing constraints ---
    // A row whose activity range touches its RHS can hold only at one
    // vertex of the box, pinning every involved variable to a bound:
    //   inequality g·x ≤ h with min-activity == h  ⇒ pin to the min vertex;
    //   equality   a·x = b with min-activity == b  ⇒ pin to the min vertex;
    //   equality   a·x = b with max-activity == b  ⇒ pin to the max vertex.
    // Each pinned variable becomes fixed (substituted out like any fixed
    // var); the row is dropped. Dual recovery (the reason this is subtle)
    // is handled in postsolve. We require each forcing row's columns to be
    // disjoint from every other forcing row's, so the multiplier recovery
    // stays independent (a conservative but always-correct restriction).
    let eff_lb_at = |fixed: &[Option<f64>], c: usize| fixed[c].unwrap_or_else(|| prob.lb_of(c));
    let eff_ub_at = |fixed: &[Option<f64>], c: usize| fixed[c].unwrap_or_else(|| prob.ub_of(c));
    let mut forced_touched = vec![false; n];

    // Pin the variables of one forcing row to `at_max` vertex (or the min
    // vertex when `at_max` is false), recording the reduction. Returns
    // false (skipped) if any column is already fixed/substituted/forced.
    // `row_entries` is the row's `(col, coef)` list, all coefficients nonzero.
    let try_force =
        |row_entries: &[(usize, f64)],
         orig_row: usize,
         is_equality: bool,
         at_max: bool,
         fixed: &mut [Option<f64>],
         forced_touched: &mut [bool],
         stack: &mut Vec<Reduction>|
         -> bool {
            // Every involved column must be free to fix and not shared with
            // another forcing row.
            for &(c, _) in row_entries {
                if fixed[c].is_some() || substituted[c] || forced_touched[c] {
                    return false;
                }
            }
            let mut cols = Vec::with_capacity(row_entries.len());
            for &(c, coef) in row_entries {
                // Vertex bound: min-activity puts coef>0 at lb, coef<0 at
                // ub; max-activity is the mirror.
                let at_upper = if at_max { coef > 0.0 } else { coef < 0.0 };
                let value = if at_upper { prob.ub_of(c) } else { prob.lb_of(c) };
                // A forcing vertex requires finite bounds; guard anyway.
                if !value.is_finite() || value.abs() >= BOUND_INF {
                    return false;
                }
                cols.push((c, coef, value, at_upper));
            }
            for &(c, _, value, _) in &cols {
                fixed[c] = Some(value);
                forced_touched[c] = true;
            }
            stack.push(Reduction::ForcingRow {
                row: orig_row,
                is_equality,
                at_max,
                cols,
            });
            true
        };

    for row in 0..m_ineq {
        if ineq_dropped[row] || g_by_row[row].is_empty() {
            continue;
        }
        let (amin, _) = activity(&g_by_row[row], &|c| eff_lb_at(&fixed, c), &|c| {
            eff_ub_at(&fixed, c)
        });
        if amin.is_finite()
            && (prob.h[row] - amin).abs() <= ACTIVITY_TOL
            && try_force(
                &g_by_row[row],
                row,
                false,
                false,
                &mut fixed,
                &mut forced_touched,
                &mut stack,
            )
        {
            ineq_dropped[row] = true;
        }
    }

    for row in 0..m_eq {
        if eq_dropped[row] || a_by_row[row].len() < 2 {
            continue;
        }
        let (amin, amax) = activity(&a_by_row[row], &|c| eff_lb_at(&fixed, c), &|c| {
            eff_ub_at(&fixed, c)
        });
        let b = prob.b[row];
        let at_max = if amin.is_finite() && (b - amin).abs() <= ACTIVITY_TOL {
            Some(false)
        } else if amax.is_finite() && (amax - b).abs() <= ACTIVITY_TOL {
            Some(true)
        } else {
            None
        };
        if let Some(at_max) = at_max {
            if try_force(
                &a_by_row[row],
                row,
                true,
                at_max,
                &mut fixed,
                &mut forced_touched,
                &mut stack,
            ) {
                eq_dropped[row] = true;
            }
        }
    }

    // --- dominated columns ---
    // A variable absent from P and the equalities, present only in
    // inequalities `Gx ≤ h`, whose live G-coefficients are sign-definite in
    // a way that matches its cost sign, is optimal at a bound: pushing it
    // there never raises the objective nor tightens a `≤` row, so an
    // optimal solution with it at that bound always exists. Fix and drop
    // it. Its bound multiplier is its reduced cost `c_k + Σᵢ aᵢₖ zᵢ`, which
    // the sign conditions (`aᵢₖ ≥ 0, c_k ≥ 0` for the lower bound; mirror
    // for the upper) make nonnegative — so the recovered dual is valid by
    // construction. This is PaPILO's dominated-column reduction, restricted
    // to the case with a clean, sign-guaranteed dual.
    {
        // Per-column G-coefficient sign summary over *live* inequality rows.
        let mut g_all_nonneg = vec![true; n];
        let mut g_all_nonpos = vec![true; n];
        for t in &prob.g {
            if t.val == ZERO_TOL || ineq_dropped[t.row] {
                continue;
            }
            if t.val < 0.0 {
                g_all_nonneg[t.col] = false;
            } else if t.val > 0.0 {
                g_all_nonpos[t.col] = false;
            }
        }
        for col in 0..n {
            if fixed[col].is_some()
                || substituted[col]
                || p_col_present[col]
                || a_col_count[col] != 0
                || g_col_count[col] == 0
            {
                continue;
            }
            let c_k = prob.c[col];
            let lb = prob.lb_of(col);
            let ub = prob.ub_of(col);
            if g_all_nonneg[col] && c_k >= 0.0 && lb > -BOUND_INF {
                fixed[col] = Some(lb);
                stack.push(Reduction::DominatedColumn { col, value: lb });
            } else if g_all_nonpos[col] && c_k <= 0.0 && ub < BOUND_INF {
                fixed[col] = Some(ub);
                stack.push(Reduction::DominatedColumn { col, value: ub });
            }
        }
    }

    // --- bound tightening (domain propagation, single pass) ---
    // From each live row, derive implied bounds on its variables and shrink
    // the box where strictly tighter. The variable is *kept* (only its box
    // changes); the subtle dual — re-attributing an active tightened
    // bound's multiplier to the source row — is handled by postsolve's
    // global bound recovery. A single pass (not iterated to a fixpoint),
    // so it tightens but does not cascade into further reductions here.
    let mut tlb: Vec<f64> = (0..n).map(|c| prob.lb_of(c)).collect();
    let mut tub: Vec<f64> = (0..n).map(|c| prob.ub_of(c)).collect();
    for c in 0..n {
        if let Some(v) = fixed[c] {
            tlb[c] = v;
            tub[c] = v;
        }
    }
    // Source row (and its coef / kind) of each variable's tightened bound.
    let mut ub_src: Vec<Option<(usize, f64, bool)>> = vec![None; n];
    let mut lb_src: Vec<Option<(usize, f64, bool)>> = vec![None; n];

    // Re-attributing an active tightened bound's multiplier to its source
    // row is only *independent* when source rows share no columns (and
    // touch no already-reduced column); otherwise the re-attributions
    // couple. So a row may serve as a tightening source only if all its
    // columns are kept (not fixed/substituted) and disjoint from every
    // other accepted source row — a conservative but always-correct
    // restriction, exactly like forcing's disjoint-column rule.
    let reduction_touched: Vec<bool> =
        (0..n).map(|c| fixed[c].is_some() || substituted[c]).collect();
    let mut bt_col_used = vec![false; n];
    let row_is_clean = |entries: &[(usize, f64)], used: &[bool]| {
        entries
            .iter()
            .all(|&(c, _)| !reduction_touched[c] && !used[c])
    };

    // Tighten variable boxes from one row whose activity lies in `[lo, hi]`
    // (inequality `≤ h`: `lo = −∞, hi = h`; equality: `lo = hi = b`).
    // Returns true on a detected empty domain (infeasible).
    let tighten_from_row =
        |entries: &[(usize, f64)],
         lo: f64,
         hi: f64,
         row_idx: usize,
         is_eq: bool,
         tlb: &mut [f64],
         tub: &mut [f64],
         ub_src: &mut [Option<(usize, f64, bool)>],
         lb_src: &mut [Option<(usize, f64, bool)>]|
         -> bool {
            let (amin, amax) = activity(entries, &|c| tlb[c], &|c| tub[c]);
            // Compute all implied bounds against the row-start state, then
            // apply (so within-row order doesn't matter).
            let mut updates: Vec<(usize, bool, f64, f64)> = Vec::new(); // (col,is_upper,val,coef)
            for &(k, a) in entries {
                if fixed[k].is_some() || a == 0.0 {
                    continue;
                }
                let contrib_min = if a > 0.0 { a * tlb[k] } else { a * tub[k] };
                let contrib_max = if a > 0.0 { a * tub[k] } else { a * tlb[k] };
                let amin_mk = amin - contrib_min;
                let amax_mk = amax - contrib_max;
                if hi.is_finite() {
                    let val = (hi - amin_mk) / a;
                    if val.is_finite() {
                        if a > 0.0 {
                            if val < tub[k] - BOUND_FEAS_TOL {
                                updates.push((k, true, val, a));
                            }
                        } else if val > tlb[k] + BOUND_FEAS_TOL {
                            updates.push((k, false, val, a));
                        }
                    }
                }
                if lo.is_finite() {
                    let val = (lo - amax_mk) / a;
                    if val.is_finite() {
                        if a > 0.0 {
                            if val > tlb[k] + BOUND_FEAS_TOL {
                                updates.push((k, false, val, a));
                            }
                        } else if val < tub[k] - BOUND_FEAS_TOL {
                            updates.push((k, true, val, a));
                        }
                    }
                }
            }
            for (k, is_upper, val, a) in updates {
                if is_upper {
                    if val < tub[k] - BOUND_FEAS_TOL {
                        tub[k] = val;
                        ub_src[k] = Some((row_idx, a, is_eq));
                    }
                } else if val > tlb[k] + BOUND_FEAS_TOL {
                    tlb[k] = val;
                    lb_src[k] = Some((row_idx, a, is_eq));
                }
                if tlb[k] > tub[k] + BOUND_FEAS_TOL {
                    return true;
                }
            }
            false
        };

    for row in 0..m_ineq {
        if ineq_dropped[row] || g_by_row[row].is_empty() || !row_is_clean(&g_by_row[row], &bt_col_used)
        {
            continue;
        }
        if tighten_from_row(
            &g_by_row[row],
            f64::NEG_INFINITY,
            prob.h[row],
            row,
            false,
            &mut tlb,
            &mut tub,
            &mut ub_src,
            &mut lb_src,
        ) {
            return PresolveOutcome::Infeasible;
        }
        for &(c, _) in &g_by_row[row] {
            bt_col_used[c] = true;
        }
    }
    for row in 0..m_eq {
        if eq_dropped[row] || a_by_row[row].is_empty() || !row_is_clean(&a_by_row[row], &bt_col_used)
        {
            continue;
        }
        let b = prob.b[row];
        if tighten_from_row(
            &a_by_row[row],
            b,
            b,
            row,
            true,
            &mut tlb,
            &mut tub,
            &mut ub_src,
            &mut lb_src,
        ) {
            return PresolveOutcome::Infeasible;
        }
        for &(c, _) in &a_by_row[row] {
            bt_col_used[c] = true;
        }
    }

    // Record a reduction for each variable whose box was strictly tightened.
    for k in 0..n {
        if fixed[k].is_some() {
            continue;
        }
        if tub[k] < prob.ub_of(k) - BOUND_FEAS_TOL {
            if let Some((row, coef, is_eq)) = ub_src[k] {
                stack.push(Reduction::BoundTightening {
                    col: k,
                    row,
                    is_equality: is_eq,
                    coef,
                    is_upper: true,
                });
            }
        }
        if tlb[k] > prob.lb_of(k) + BOUND_FEAS_TOL {
            if let Some((row, coef, is_eq)) = lb_src[k] {
                stack.push(Reduction::BoundTightening {
                    col: k,
                    row,
                    is_equality: is_eq,
                    coef,
                    is_upper: false,
                });
            }
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

    // Carry the kept columns' (possibly tightened) bounds into the reduced
    // problem. Emit bounds when the original had them or bound tightening
    // produced a finite bound on a kept variable; otherwise leave empty.
    let need_bounds = prob.has_bounds()
        || kept_cols
            .iter()
            .any(|&c| tlb[c] > -BOUND_INF || tub[c] < BOUND_INF);
    let (new_lb, new_ub) = if need_bounds {
        (
            kept_cols.iter().map(|&c| tlb[c]).collect(),
            kept_cols.iter().map(|&c| tub[c]).collect(),
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
        chain: Vec::new(),
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

/// Relative tolerance for confirming two rows are scalar multiples.
const PARALLEL_TOL: f64 = 1e-9;

/// Canonical pivot used to normalize a row for *parallel* (scalar-
/// multiple) detection: its first coefficient (the rows' coeffs are
/// sorted by column). For inequalities we divide by the pivot's
/// **magnitude** so only *positive* multiples — same inequality direction
/// — normalize alike; for equalities we divide by the **signed** pivot so
/// `±` multiples (the same constraint either way) match.
fn pivot_divisor(row: &Row, is_equality: bool) -> f64 {
    let p = row.coeffs[0].1;
    if is_equality {
        p
    } else {
        p.abs()
    }
}

/// Normalized coefficient values (parallel detection): `coeffs / divisor`.
fn normalized_coeffs(row: &Row, is_equality: bool) -> Vec<(usize, f64)> {
    let d = pivot_divisor(row, is_equality);
    row.coeffs.iter().map(|&(c, v)| (c, v / d)).collect()
}

/// Hash a normalized coefficient pattern. Values are quantized so exact
/// scalar multiples hash together; the hash is only a *filter* (a quantize
/// boundary can split a true pair into different buckets, which merely
/// misses a reduction — never a wrong merge, since membership is confirmed
/// by [`approx_parallel`]).
fn parallel_signature(norm: &[(usize, f64)]) -> u64 {
    let mut h = DefaultHasher::new();
    norm.len().hash(&mut h);
    for &(c, v) in norm {
        c.hash(&mut h);
        ((v / PARALLEL_TOL).round() as i64).hash(&mut h);
    }
    h.finish()
}

/// Confirm two normalized patterns are equal to `PARALLEL_TOL` (same
/// columns, matching values). Conservative: only true scalar multiples
/// pass, so a wrong merge is impossible.
fn approx_parallel(a: &[(usize, f64)], b: &[(usize, f64)]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(&(ca, va), &(cb, vb))| {
            ca == cb && (va - vb).abs() <= PARALLEL_TOL * (1.0 + va.abs().max(vb.abs()))
        })
}

/// Remove **parallel** rows (scalar multiples of one another), the
/// generalization of exact-duplicate removal (PaPILO's parallel-row
/// reduction). Normalized signatures are computed in parallel (rayon);
/// grouping and the per-group decision are serial and cheap.
///
/// Dual recovery stays trivial because we always keep an *original* row in
/// its own frame and set every dropped row's multiplier to 0 (the kept row
/// carries the constraint):
/// - equalities — all scalar multiples represent one constraint; their
///   *normalized* right-hand sides must agree, else the system is
///   infeasible. Keep the first; drop the rest.
/// - inequalities — positive multiples of one direction; keep the **most
///   restrictive** original row (smallest normalized rhs `h / |pivot|`)
///   and drop the looser ones, which it implies.
fn dedup_rows(rows: Vec<Row>, is_equality: bool) -> Result<Vec<Row>, ()> {
    if rows.len() < 2 {
        return Ok(rows);
    }

    // Parallel: normalize + hash each row (PaPILO-style hashing-based
    // pairing, generalized to scalar multiples).
    let norms: Vec<Vec<(usize, f64)>> =
        rows.par_iter().map(|r| normalized_coeffs(r, is_equality)).collect();
    let sigs: Vec<u64> = norms.par_iter().map(|n| parallel_signature(n)).collect();

    // Group row indices by signature (serial; small).
    let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &s) in sigs.iter().enumerate() {
        buckets.entry(s).or_default().push(i);
    }

    // Normalized rhs of a row, for the tightness / consistency decisions.
    let norm_rhs = |i: usize| rows[i].rhs / pivot_divisor(&rows[i], is_equality);

    let mut keep = vec![true; rows.len()];
    for idxs in buckets.values() {
        if idxs.len() < 2 {
            continue;
        }
        // Within a signature bucket, partition into confirmed-parallel
        // groups (guards against quantization collisions).
        let mut handled = vec![false; idxs.len()];
        for a in 0..idxs.len() {
            if handled[a] {
                continue;
            }
            let mut group = vec![idxs[a]];
            handled[a] = true;
            for b in (a + 1)..idxs.len() {
                if !handled[b] && approx_parallel(&norms[idxs[a]], &norms[idxs[b]]) {
                    handled[b] = true;
                    group.push(idxs[b]);
                }
            }
            if group.len() < 2 {
                continue;
            }
            if is_equality {
                // Parallel equalities: normalized rhs must agree, else the
                // two scaled-identical constraints are contradictory.
                let r0 = norm_rhs(group[0]);
                for &g in &group[1..] {
                    if (norm_rhs(g) - r0).abs() > PARALLEL_TOL * (1.0 + r0.abs()) {
                        return Err(());
                    }
                }
                for &g in &group[1..] {
                    keep[g] = false;
                }
            } else {
                // Parallel inequalities: keep the most restrictive original
                // row (smallest normalized rhs); it implies the rest.
                let tightest = *group
                    .iter()
                    .min_by(|&&p, &&q| norm_rhs(p).partial_cmp(&norm_rhs(q)).unwrap())
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
    /// Forcing rows: each pins all its variables to a bound and is dropped.
    pub forcing_rows: usize,
    /// Dominated columns fixed to a bound and dropped.
    pub dominated_cols: usize,
    /// Variable bounds tightened by domain propagation.
    pub tightened_bounds: usize,
}

impl PresolveStats {
    /// Did presolve remove anything?
    pub fn reduced_anything(&self) -> bool {
        self.reduced_vars < self.orig_vars || self.reduced_rows < self.orig_rows
    }
}

impl Presolve {
    /// Did this single pass change anything (a reduction, or a dropped
    /// row)? Used by [`presolve`] to detect the fixpoint.
    fn changed(&self) -> bool {
        !self.stack.is_empty()
            || self.reduced.n < self.orig_n
            || self.reduced.m_eq() + self.reduced.m_ineq() < self.orig_m_eq + self.orig_m_ineq
    }

    /// Reduction summary (sizes before/after and counts by reduction). For
    /// an iterated presolve, counts aggregate over the rounds.
    pub fn stats(&self) -> PresolveStats {
        if self.chain.is_empty() {
            return self.stats_once();
        }
        let mut s = PresolveStats {
            orig_vars: self.orig_n,
            reduced_vars: self.reduced.n,
            orig_rows: self.orig_m_eq + self.orig_m_ineq,
            reduced_rows: self.reduced.m_eq() + self.reduced.m_ineq(),
            ..Default::default()
        };
        for layer in &self.chain {
            let ls = layer.stats_once();
            s.fixed_vars += ls.fixed_vars;
            s.free_cols_fixed += ls.free_cols_fixed;
            s.free_col_singletons += ls.free_col_singletons;
            s.forcing_rows += ls.forcing_rows;
            s.dominated_cols += ls.dominated_cols;
            s.tightened_bounds += ls.tightened_bounds;
        }
        s
    }

    fn stats_once(&self) -> PresolveStats {
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
                Reduction::ForcingRow { .. } => s.forcing_rows += 1,
                Reduction::DominatedColumn { .. } => s.dominated_cols += 1,
                Reduction::BoundTightening { .. } => s.tightened_bounds += 1,
            }
        }
        s
    }

    /// Expand a reduced-problem solution back to the original space,
    /// recovering primal `x` and duals `(y, z)`. For an iterated presolve,
    /// folds the per-round postsolves in reverse.
    pub fn postsolve(&self, red: &QpSolution) -> QpSolution {
        if self.chain.is_empty() {
            return self.postsolve_once(red);
        }
        let mut sol = red.clone();
        for layer in self.chain.iter().rev() {
            sol = layer.postsolve_once(&sol);
        }
        sol
    }

    /// Expand a single pass's reduced solution back to its original space.
    fn postsolve_once(&self, red: &QpSolution) -> QpSolution {
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
                Reduction::ForcingRow { cols, .. } => {
                    // Each forced variable sits at the stored bound value.
                    for &(col, _, value, _) in cols {
                        x[col] = value;
                    }
                }
                Reduction::DominatedColumn { col, value, .. } => x[*col] = *value,
                // The variable is kept; only its box changed, so its primal
                // comes from the reduced solution (already mapped above).
                Reduction::BoundTightening { .. } => {}
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

        // Forcing-row multipliers. `grad` (above, = grad0) is each pinned
        // variable's reduced cost *excluding* the forcing row (its
        // multiplier is still 0). The row multiplier is the tightest value
        // making every pinned variable's bound multiplier correctly signed:
        //   min-vertex  ⇒ mult = maxⱼ(−gradⱼ/coefⱼ)  (clamped ≥ 0 if ≤-row);
        //   max-vertex  ⇒ mult = minⱼ(−gradⱼ/coefⱼ)  (equalities only).
        // (The pinned variables' bound multipliers themselves come out of
        // the global recovery below.)
        for r in &self.stack {
            if let Reduction::ForcingRow {
                row,
                is_equality,
                at_max,
                cols,
            } = r
            {
                let mut mult = if *at_max { f64::INFINITY } else { f64::NEG_INFINITY };
                for &(col, coef, _, _) in cols {
                    let t = -grad[col] / coef;
                    mult = if *at_max { mult.min(t) } else { mult.max(t) };
                }
                if !*is_equality {
                    mult = mult.max(0.0); // inequality multiplier ≥ 0
                }
                if !mult.is_finite() {
                    mult = 0.0;
                }
                if *is_equality {
                    y[*row] = mult;
                } else {
                    z[*row] = mult;
                }
            }
        }

        // Re-attribute active tightened-bound multipliers to their source
        // rows. A tightened bound that is active in the reduced solve while
        // the *original* bound is slack is not a real bound — its
        // multiplier belongs to the row that implied it. Because tightening
        // sources are column-disjoint, these moves are independent.
        let mut col_reduced = vec![usize::MAX; n];
        for (newc, &oldc) in self.kept_cols.iter().enumerate() {
            col_reduced[oldc] = newc;
        }
        for r in &self.stack {
            if let Reduction::BoundTightening {
                col,
                row,
                is_equality,
                coef,
                is_upper,
            } = r
            {
                let newc = col_reduced[*col];
                if newc == usize::MAX {
                    continue;
                }
                let delta = if *is_upper {
                    let m = red.z_ub.get(newc).copied().unwrap_or(0.0);
                    if m > 0.0 && x[*col] < self.orig.ub_of(*col) - BOUND_FEAS_TOL {
                        m / coef
                    } else {
                        0.0
                    }
                } else {
                    let m = red.z_lb.get(newc).copied().unwrap_or(0.0);
                    if m > 0.0 && x[*col] > self.orig.lb_of(*col) + BOUND_FEAS_TOL {
                        -m / coef
                    } else {
                        0.0
                    }
                };
                if *is_equality {
                    y[*row] += delta;
                } else {
                    z[*row] += delta;
                }
            }
        }

        // Global bound-multiplier recovery. With every row multiplier now in
        // place, recompute the full reduced cost and read off each
        // variable's bound multipliers by complementarity against its
        // *original* box: at the lower bound `z_lb = max(0, grad)`, at the
        // upper `z_ub = max(0, −grad)`, interior ⇒ both 0. This single rule
        // subsumes the per-reduction bound recovery (fixed, free-fixed,
        // forcing, dominated — each lands at a real bound or interior with
        // the right reduced cost) and correctly zeroes a tightened
        // variable's bound dual (it sits interior to its real box, the force
        // having moved to the source row above).
        let mut grad = vec![0.0; n];
        grad[..n].copy_from_slice(&self.orig.c[..n]);
        self.orig.p_mul(&x, &mut grad);
        self.orig.at_mul(&y, &mut grad);
        self.orig.gt_mul(&z, &mut grad);
        let mut z_lb = vec![0.0; n];
        let mut z_ub = vec![0.0; n];
        for i in 0..n {
            let lb = self.orig.lb_of(i);
            let ub = self.orig.ub_of(i);
            let at_lb = lb > -BOUND_INF && (x[i] - lb).abs() <= ACTIVE_BOUND_TOL;
            let at_ub = ub < BOUND_INF && (ub - x[i]).abs() <= ACTIVE_BOUND_TOL;
            if at_lb && grad[i] > 0.0 {
                z_lb[i] = grad[i];
            } else if at_ub && grad[i] < 0.0 {
                z_ub[i] = -grad[i];
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
