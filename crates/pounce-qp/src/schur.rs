//! §4.2 sparse Schur-complement parametric active-set updates.
//!
//! Standard SOTA mechanism (Kirches 2011, qpOASES-extended): maintain
//! a cached LDLᵀ factor of a *fixed-dimension* "K_max" matrix in which
//! every potentially-active constraint and bound gets a slot. Active
//! slots carry the actual constraint row; inactive slots carry a
//! `(p,p) = 1` sentinel so the slot decouples from `x` and the
//! corresponding multiplier is forced to zero.
//!
//! Working-set changes (flip a slot active ↔ inactive) are then
//! **symmetric rank-2 updates** of K_max:
//!
//! ```text
//! K_new = K_old + u vᵀ + v uᵀ
//! ```
//!
//! with `u = e_p` (basis vector for the slot's saddle row) and
//! `v` chosen so the row + column flip absorb correctly (the
//! diagonal-correction trick — `v_p = ±½` produces the desired
//! `±1` change at `(p, p)` after symmetrization).
//!
//! Solving against `K_W = K_0 + UVᵀ` (where `U`, `V` accumulate the
//! rank-2 updates since the last refactor) uses the Sherman-
//! Morrison-Woodbury formula:
//!
//! ```text
//! K_W⁻¹ b = K_0⁻¹ b − K_0⁻¹ U · (I + Vᵀ K_0⁻¹ U)⁻¹ · Vᵀ K_0⁻¹ b
//! ```
//!
//! `S = I + Vᵀ K_0⁻¹ U` is the dense Schur block; it grows by 2
//! rows + 2 cols per working-set change. When `S` reaches
//! `max_schur_updates_before_refactor`, we refactor with the
//! current working set as the new base and reset `U`, `V`, `S`.
//!
//! Costs per inner-loop iteration:
//! - One cached `resolve` per right-hand side (the `K_0⁻¹ b`
//!   step).
//! - One small dense `S y = w` solve (size `s_dim` ≤
//!   `max_schur_updates`).
//! - One more cached `resolve` per new flip (to compute
//!   `K_0⁻¹ u` and `K_0⁻¹ v` for the new rank-2 update).
//!
//! References:
//! - Kirches, *Fast Numerical Methods for Mixed-Integer Nonlinear
//!   Model-Predictive Control*, Vieweg+Teubner (2011), Ch. 5-7 —
//!   the canonical reference for the sparse-K_max Schur scheme.
//! - Ferreau, Kirches, Potschka, Bock, Diehl, "qpOASES: a parametric
//!   active-set algorithm for quadratic programming", *Math. Prog.
//!   Comp.* **6** (2014), 327-363 — the dense reference algorithm.
//! - Bartels, Reid lineage — the underlying basis-update theory
//!   (Reid 1982).
//!
//! Phase 5a.2 deliverable. This commit ships the standalone module
//! + opt-in wiring (`QpOptions::use_schur_updates`); the existing
//! refactor-per-iteration path remains the default and is unchanged.

use crate::error::QpError;
use crate::factor::LinearSolver;
use crate::kkt::KktTriplet;
use crate::options::QpOptions;
use crate::problem::QpProblem;
use crate::working_set::WorkingSet;
use pounce_common::{Index, Number};

/// Cached Schur-complement state for one active-set solve.
///
/// Layout: K_max has `dim = n + m + n` (n primals plus `m_total =
/// m + n` slot rows). Slot indices `0..m` correspond to general
/// constraint rows of `A` (in their original order); slot indices
/// `m..m+n` correspond to variable bounds (slot `m + j` is the
/// bound on `x[j]`).
pub struct SchurState {
    pub n: usize,
    pub m: usize,
    pub m_total: usize,
    pub dim: usize,

    /// Active flag per slot at the time of the LAST reset. This is
    /// what the cached factor in `LinearSolver` represents.
    base_active: Vec<bool>,

    /// The rank-2 updates accumulated since the last reset, one per
    /// working-set change, stored **sparse** (gh#1206 P3). Change `k`
    /// contributes columns `2k` and `2k + 1` of `U = [u, v, …]` and of
    /// `V = [v, u, …]`, with `u = e_p` held as its index alone and `v` as
    /// its nonzeros: one row of `A` plus the `(p, ±½)` diagonal correction,
    /// or two entries for a bound. They used to be four dense length-`dim`
    /// vectors per change (`V` held a second copy of `u` and `v`), and every
    /// Schur solve took `O(s_dim · dim)` dense dots over them.
    changes: Vec<SparseChange>,

    /// `K_0⁻¹ U_j` for each column of `U`, cached at apply-time so
    /// the per-iteration solve doesn't re-do them. Genuinely dense, except
    /// `K_0⁻¹ e_p = e_p` for a slot inactive at the last reset (P4), which
    /// is kept as the unit vector it is.
    kinv_u_cols: Vec<KinvCol>,

    /// `S = I + Vᵀ K_0⁻¹ U`, row-major, size `s_dim × s_dim`.
    s_matrix: Vec<Number>,
    s_dim: usize,

    /// The `K_0` triplet **as actually factored** at the last reset —
    /// including any §4.5 inertia shift, so it matches the cached factor
    /// rather than the unshifted ideal.
    ///
    /// Kept solely so [`Self::solve`] can form the residual
    /// `b − (K_0 + UVᵀ)x` for iterative refinement. Without a residual there
    /// is no way to detect that the SMW correction has drifted, and drift is
    /// exactly this path's failure mode (see the refinement note on `solve`).
    base_kkt: Option<KktTriplet>,

    /// The §4.5 inertia shift δ carried by that factor (0.0 when the
    /// unshifted KKT factored cleanly). Rank-2 updates never touch the
    /// H block, so this stays valid until the next reset. Callers need
    /// it to know whether the direction they just solved for is the
    /// true Newton step or a shifted one — see
    /// [`crate::solver::model_step_cap`].
    shift: Number,

    /// Negative eigenvalues of the cached `K_0`: the count its factorization
    /// was checked against at the last reset (zero pivots are a failed
    /// reset, so `K_0` is nonsingular). Read by [`Self::updated_neg_evals`].
    base_neg: i32,

    /// Row index of `A` ([`a_row_index`]), built at the last reset and
    /// reused by every general-row update until the next one. Without it
    /// each such update scanned all of `A` for one row — 1.44 M triplets per
    /// working-set change on the gh#1206 collocation model (P5).
    a_rows: Option<(Vec<usize>, Vec<usize>)>,
}

/// One rank-2 update `e_p vᵀ + v e_pᵀ`, stored sparse: `u = e_p` by its
/// index and `v` by its nonzeros in ascending index order (gh#1206 P3).
///
/// Ascending order is what keeps the sparse dots below bit-identical to the
/// dense ones they replace: a dense dot sums `aᵢbᵢ` in index order, and the
/// terms a sparse one skips are exact zeros.
struct SparseChange {
    p: usize,
    v: Vec<(usize, Number)>,
}

impl SparseChange {
    /// Column `j ∈ {0, 1}` of this change's block of `U` (`[u, v]`).
    fn u_col(&self, j: usize) -> SparseCol<'_> {
        if j == 0 {
            SparseCol::Unit(self.p)
        } else {
            SparseCol::Entries(&self.v)
        }
    }

    /// Column `j ∈ {0, 1}` of this change's block of `V` (`[v, u]`).
    fn v_col(&self, j: usize) -> SparseCol<'_> {
        self.u_col(1 - j)
    }
}

/// A cached `K_0⁻¹ U_j`.
enum KinvCol {
    /// `e_p` itself: `u = e_p` on a slot inactive at the last reset (P4).
    Unit(usize),
    Dense(Vec<Number>),
}

impl KinvCol {
    /// `x ← x − y · self`.
    fn sub_scaled_from(&self, y: Number, x: &mut [Number]) {
        match self {
            KinvCol::Unit(p) => x[*p] -= y * 1.0,
            KinvCol::Dense(c) => {
                for (xi, &ci) in x.iter_mut().zip(c.iter()) {
                    *xi -= y * ci;
                }
            }
        }
    }
}

/// A column of `U` or `V`: `e_p`, or a sparse `v`.
#[derive(Clone, Copy)]
enum SparseCol<'a> {
    Unit(usize),
    Entries(&'a [(usize, Number)]),
}

impl SparseCol<'_> {
    /// `selfᵀ x` for a dense `x`.
    fn dot_dense(self, x: &[Number]) -> Number {
        match self {
            SparseCol::Unit(p) => x[p],
            SparseCol::Entries(e) => e.iter().map(|&(i, v)| v * x[i]).sum(),
        }
    }

    /// `selfᵀ k` for a cached `K_0⁻¹ U_j`.
    fn dot_kinv(self, k: &KinvCol) -> Number {
        match k {
            KinvCol::Dense(d) => self.dot_dense(d),
            KinvCol::Unit(q) => match self {
                SparseCol::Unit(p) => {
                    if p == *q {
                        1.0
                    } else {
                        0.0
                    }
                }
                SparseCol::Entries(e) => e.iter().find(|&&(i, _)| i == *q).map_or(0.0, |&(_, v)| v),
            },
        }
    }

    /// `y ← y + c · self`.
    fn add_scaled_to(self, c: Number, y: &mut [Number]) {
        match self {
            SparseCol::Unit(p) => y[p] += c * 1.0,
            SparseCol::Entries(e) => {
                for &(i, v) in e {
                    y[i] += c * v;
                }
            }
        }
    }
}

impl SchurState {
    pub fn new(n: usize, m: usize) -> Self {
        let m_total = m + n;
        let dim = n + m_total;
        Self {
            n,
            m,
            m_total,
            dim,
            base_active: vec![false; m_total],
            changes: Vec::new(),
            kinv_u_cols: Vec::new(),
            s_matrix: Vec::new(),
            s_dim: 0,
            base_kkt: None,
            shift: 0.0,
            base_neg: 0,
            a_rows: None,
        }
    }

    /// The §4.5 inertia shift δ in the currently cached factor.
    pub fn shift(&self) -> Number {
        self.shift
    }

    /// Active flag for slot `s` derived from the user-facing
    /// `WorkingSet`. Slots `[0, m)` map to general constraint
    /// rows; slots `[m, m+n)` map to variable bounds.
    pub fn slot_active(working: &WorkingSet, slot: usize) -> bool {
        let m = working.m();
        if slot < m {
            working.constraints[slot].is_active()
        } else {
            working.bounds[slot - m].is_active()
        }
    }

    fn slot_is_general(&self, slot: usize) -> bool {
        slot < self.m
    }

    /// [`Self::build_k_max_triplet_indexed`] with the row index of `A`
    /// built here. Test-only: `reset` builds the index once and keeps it.
    #[cfg(test)]
    pub(crate) fn build_k_max_triplet(&self, qp: &QpProblem, slot_active: &[bool]) -> KktTriplet {
        self.build_k_max_triplet_indexed(qp, slot_active, &a_row_index(qp))
    }

    /// Build the K_max triplet for the given active-slot pattern, reading
    /// `A`'s rows through `a_rows` ([`a_row_index`]). Active slots carry the
    /// actual constraint row; inactive slots carry the `(p, p) = 1` sentinel.
    fn build_k_max_triplet_indexed(
        &self,
        qp: &QpProblem,
        slot_active: &[bool],
        a_rows: &(Vec<usize>, Vec<usize>),
    ) -> KktTriplet {
        let n_i = self.n as Index;
        let mut irn = Vec::new();
        let mut jcn = Vec::new();
        let mut vals = Vec::new();

        // H block (lower-triangle 1-based; normalize defensively), with an
        // explicit zero on every diagonal `H` does not store, so an inertia
        // shift changes values and not the pattern (gh#1206 P2).
        crate::kkt::push_h_block(qp, &mut irn, &mut jcn, &mut vals);

        // One row per slot.
        let a_jcols = qp.a.jcols();
        let a_vals = qp.a.values();
        let (row_ptr, row_entries) = (&a_rows.0, &a_rows.1);
        for slot in 0..self.m_total {
            let saddle_row = n_i + (slot as Index) + 1; // 1-based row in K_max
            if slot_active[slot] {
                if self.slot_is_general(slot) {
                    // This row's `A` entries, in triplet order.
                    for &k in &row_entries[row_ptr[slot]..row_ptr[slot + 1]] {
                        // (saddle_row, jcol, value) — saddle_row > n ≥ jcol ⇒ lower-tri.
                        irn.push(saddle_row);
                        jcn.push(a_jcols[k]);
                        vals.push(a_vals[k]);
                    }
                } else {
                    // Variable bound slot: single (saddle_row, var+1, 1).
                    let var = slot - self.m;
                    irn.push(saddle_row);
                    jcn.push((var + 1) as Index);
                    vals.push(1.0);
                }
            } else {
                // Sentinel diagonal at (saddle_row, saddle_row, 1).
                irn.push(saddle_row);
                jcn.push(saddle_row);
                vals.push(1.0);
            }
        }

        KktTriplet {
            dim: self.dim,
            irn,
            jcn,
            vals,
        }
    }

    /// Build the rank-2 update vectors for slot `p` flipping from
    /// its current state in K_max (the BASE state, with all prior
    /// updates already applied) to the OPPOSITE state.
    ///
    /// `going_active = true`: slot transitions inactive ⇒ active.
    /// `going_active = false`: slot transitions active ⇒ inactive.
    ///
    /// Sign convention: with `u = e_p` and the rank-2 update
    /// `u vᵀ + v uᵀ`, an entry `v_p` contributes `2 v_p` to the
    /// `(p, p)` diagonal of the update. To get a `+1` diagonal
    /// change (going inactive: sentinel goes from 0 to 1) we set
    /// `v_p = +½`; for `-1` change (going active: 1 → 0) set
    /// `v_p = -½`.
    fn build_update_vectors(
        &self,
        qp: &QpProblem,
        slot: usize,
        going_active: bool,
    ) -> SparseChange {
        let p = self.n + slot; // 0-based row in K_max
        let sign = if going_active { 1.0 } else { -1.0 };
        let mut v: Vec<(usize, Number)> = Vec::new();
        if self.slot_is_general(slot) {
            let (a_jcols, a_vals) = (qp.a.jcols(), qp.a.values());
            // The row index cached at the last reset, when it still describes
            // this `A`; a full scan otherwise. Both visit the row's triplets
            // in ascending order (the counting sort is stable), so `v` is the
            // same to the bit either way.
            match &self.a_rows {
                Some((ptr, entries)) if entries.len() == a_jcols.len() && ptr.len() == qp.m + 1 => {
                    for &k in &entries[ptr[slot]..ptr[slot + 1]] {
                        v.push(((a_jcols[k] - 1) as usize, sign * a_vals[k]));
                    }
                }
                _ => {
                    let row_1based = (slot + 1) as Index;
                    let a_irows = qp.a.irows();
                    for k in 0..a_irows.len() {
                        if a_irows[k] == row_1based {
                            v.push(((a_jcols[k] - 1) as usize, sign * a_vals[k]));
                        }
                    }
                }
            }
            // Ascending column, duplicates summed in triplet order — the
            // stable sort keeps that order — exactly as the dense
            // `v[col] += …` accumulation they replace.
            v.sort_by_key(|&(i, _)| i);
            let mut merged: Vec<(usize, Number)> = Vec::with_capacity(v.len() + 1);
            for (i, x) in v {
                match merged.last_mut() {
                    Some((j, acc)) if *j == i => *acc += x,
                    _ => merged.push((i, 0.0 + x)),
                }
            }
            v = merged;
        } else {
            v.push((slot - self.m, sign));
        }
        // `p ≥ n` exceeds every column index above, so this stays sorted.
        v.push((p, if going_active { -0.5 } else { 0.5 }));

        SparseChange { p, v }
    }

    /// Discard the update layer (U, V, `K_0⁻¹U`, S).
    fn clear_updates(&mut self) {
        self.changes.clear();
        self.kinv_u_cols.clear();
        self.s_matrix.clear();
        self.s_dim = 0;
    }

    /// Column `j` of `U`.
    fn u_col(&self, j: usize) -> SparseCol<'_> {
        self.changes[j / 2].u_col(j % 2)
    }

    /// Column `j` of `V`.
    fn v_col(&self, j: usize) -> SparseCol<'_> {
        self.changes[j / 2].v_col(j % 2)
    }

    /// Reset: discard any cached SMW state, build the K_max for
    /// the current working set, factor it via `linsol`. Runs
    /// inertia-control retries on `WrongInertia` / `Singular`
    /// per `QpOptions` (same δ-shift policy as the non-Schur
    /// `factorize_with_inertia_control`).
    pub fn reset(
        &mut self,
        linsol: &mut LinearSolver,
        qp: &QpProblem,
        working: &WorkingSet,
        expected_neg: i32,
        opts: &QpOptions,
    ) -> Result<(), QpError> {
        for slot in 0..self.m_total {
            self.base_active[slot] = Self::slot_active(working, slot);
        }
        let a_rows = a_row_index(qp);
        let mut kkt = self.build_k_max_triplet_indexed(qp, &self.base_active, &a_rows);
        self.a_rows = Some(a_rows);
        let mut rhs = vec![0.0; self.dim];

        // First attempt: no shift.
        let mut current = 0.0;
        let mut last_err: Option<QpError>;
        match linsol.factorize_and_solve(&kkt, &mut rhs, Some(expected_neg)) {
            Ok(()) => {
                self.clear_updates();
                self.base_kkt = Some(kkt.clone());
                self.shift = 0.0;
                self.base_neg = expected_neg;
                return Ok(());
            }
            Err(ref e) if e.is_recoverable_factorization_failure() => {
                last_err = Some(e.clone());
            }
            Err(e) => return Err(e),
        }

        let singular_unshifted = linsol.last_factor_singular();
        let mut next = opts.inertia_shift_initial;
        for rung in 0..opts.inertia_max_shifts {
            // gh#1206 B6, as in `factorize_with_inertia_control`: singular
            // unshifted and at the first rung, and a dual shift on the active
            // slot rows (never the `(p, p) = 1` sentinels) factors — a
            // rank-deficient working set, which no rung of this ladder
            // repairs. The caller (`schur_reset_rank_repaired`) prunes it.
            if rung == 1
                && singular_unshifted
                && linsol.last_factor_singular()
                && !crate::deadline::expired()
                && linsol.probe_rank_deficiency(&kkt, self.n, expected_neg, opts)?
            {
                return Err(QpError::rank_deficient_working_set());
            }
            kkt.add_h_diagonal_shift(self.n, next - current);
            current = next;
            let mut rhs_local = vec![0.0; self.dim];
            match linsol.factorize_and_solve(&kkt, &mut rhs_local, Some(expected_neg)) {
                Ok(()) => {
                    self.clear_updates();
                    self.base_kkt = Some(kkt.clone());
                    self.shift = current;
                    self.base_neg = expected_neg;
                    return Ok(());
                }
                Err(ref e) if e.is_recoverable_factorization_failure() => {
                    last_err = Some(e.clone());
                    next *= opts.inertia_shift_factor;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap_or_else(|| {
            QpError::LinearSolverFailure(format!(
                "Schur reset: inertia control exhausted {} shifts (final δ = {:.3e})",
                opts.inertia_max_shifts, current
            ))
        }))
    }

    /// Apply a working-set flip on slot `slot`. Computes the rank-
    /// 2 update vectors, requests `K_0⁻¹ u` and `K_0⁻¹ v` via the
    /// cached `LinearSolver::resolve`, appends them to `U`, `V`,
    /// `K_0⁻¹ U`, and grows the dense Schur block `S` by 2.
    pub fn apply_change(
        &mut self,
        linsol: &mut LinearSolver,
        qp: &QpProblem,
        slot: usize,
        going_active: bool,
    ) -> Result<(), QpError> {
        let change = self.build_update_vectors(qp, slot, going_active);
        let p = change.p;
        // `u = e_p`. A slot inactive at the last reset has the sentinel row
        // `(p, p) = 1` and nothing else in row or column `p` of `K_0` (the
        // inertia shift touches the H block only), so `K_0 e_p = e_p` and
        // `K_0⁻¹ u = u` exactly — no back-solve needed (gh#1206 P4: 0.4 s
        // each at `dim = 448 117`). The base factor is what `resolve` uses,
        // so flips since the reset do not change this.
        let kinv_u = if self.base_active[slot] {
            let mut kinv_u = vec![0.0; self.dim];
            kinv_u[p] = 1.0;
            linsol.resolve(&mut kinv_u)?;
            KinvCol::Dense(kinv_u)
        } else {
            KinvCol::Unit(p)
        };
        let mut kinv_v = vec![0.0; self.dim];
        for &(i, x) in &change.v {
            kinv_v[i] = x;
        }
        linsol.resolve(&mut kinv_v)?;
        let kinv_v = KinvCol::Dense(kinv_v);

        let old_dim = self.s_dim;
        let new_dim = old_dim + 2;
        let mut new_s = vec![0.0; new_dim * new_dim];

        // Copy old S into the top-left old_dim × old_dim block.
        for i in 0..old_dim {
            for j in 0..old_dim {
                new_s[i * new_dim + j] = self.s_matrix[i * old_dim + j];
            }
        }
        // New rows: V_new[i]ᵀ K_0⁻¹ U_j for the two new V rows.
        // V appended columns: [v, u] (we append v first, then u).
        // U appended columns: [u, v].
        // Index convention: new column index `old_dim`  → u
        //                                  `old_dim + 1` → v
        // For V: new row `old_dim`     → vᵀ
        //                `old_dim + 1` → uᵀ
        // Every product below is a sparse `V` column against a cached
        // `K_0⁻¹ U` column: O(nnz(v)), not O(dim) (P3).
        let v_new_rows = [change.v_col(0), change.v_col(1)];
        let u_new_cols = [&kinv_u, &kinv_v];

        // Top-right block: old V rows × new K_0⁻¹ U cols.
        for i in 0..old_dim {
            let vi = self.v_col(i);
            new_s[i * new_dim + old_dim] = vi.dot_kinv(&kinv_u);
            new_s[i * new_dim + old_dim + 1] = vi.dot_kinv(&kinv_v);
        }
        // Bottom-left block: new V rows × old K_0⁻¹ U cols.
        for ii in 0..2 {
            for j in 0..old_dim {
                new_s[(old_dim + ii) * new_dim + j] = v_new_rows[ii].dot_kinv(&self.kinv_u_cols[j]);
            }
        }
        // Bottom-right block: 2×2 from new V rows × new K_0⁻¹ U cols.
        for ii in 0..2 {
            for jj in 0..2 {
                let entry = v_new_rows[ii].dot_kinv(u_new_cols[jj]);
                let identity = if ii == jj { 1.0 } else { 0.0 };
                new_s[(old_dim + ii) * new_dim + old_dim + jj] = entry + identity;
            }
        }

        self.changes.push(change);
        self.kinv_u_cols.push(kinv_u);
        self.kinv_u_cols.push(kinv_v);
        self.s_matrix = new_s;
        self.s_dim = new_dim;

        Ok(())
    }

    /// Solve `K_W [x; λ] = rhs` using SMW. `rhs` is overwritten
    /// with the solution. Requires that `reset` has been called
    /// at least once.
    /// `y = (K_0 + UVᵀ) x` — a matvec against the operator the SMW formula
    /// inverts, used to form the refinement residual.
    ///
    /// `K_0` is stored as **lower-triangle** 1-based triplets, so off-diagonal
    /// entries must be applied to both `(i,j)` and `(j,i)`; getting that wrong
    /// would silently halve the symmetric part and make refinement diverge.
    /// Returns `None` when no base matrix has been recorded (no reset yet), in
    /// which case the caller simply skips refinement.
    fn kw_matvec(&self, x: &[Number]) -> Option<Vec<Number>> {
        let kkt = self.base_kkt.as_ref()?;
        let mut y = vec![0.0; self.dim];
        for k in 0..kkt.irn.len() {
            let i = (kkt.irn[k] - 1) as usize;
            let j = (kkt.jcn[k] - 1) as usize;
            let v = kkt.vals[k];
            y[i] += v * x[j];
            if i != j {
                y[j] += v * x[i];
            }
        }
        // + U (Vᵀ x)
        for j in 0..self.s_dim {
            let c = self.v_col(j).dot_dense(x);
            if c != 0.0 {
                self.u_col(j).add_scaled_to(c, &mut y);
            }
        }
        Some(y)
    }

    /// Solve `(K_0 + UVᵀ) x = rhs` in place, **with iterative refinement**.
    ///
    /// Refinement is not a nicety here; it is what makes this path a
    /// performance switch rather than an accuracy trade. The SMW correction is
    /// computed against a factor of `K_0`, and its error grows with the number
    /// of accumulated rank-2 updates, so the returned step drifts as the update
    /// layer fills. That drift lands directly in the iterate: the active-set
    /// loop takes the (slightly wrong) step, and the final `x` inherits every
    /// step's error. Measured on the `tests/schur_vs_refactor.rs`
    /// singular-Hessian QP, whose exact optimum is `x₀ = 1.5`:
    ///
    /// | `max_schur_updates_before_refactor` | error vs refactor path |
    /// |---|---|
    /// | 1  | 0 (exact) |
    /// | 5  | 0 (exact) |
    /// | 50 (default) | 1.5e-6 |
    ///
    /// A `1e-6` primal error is far above the `1e-9` `opt_tol` this engine
    /// claims, and it was enough to drop the Maros-Mészáros spot check from
    /// 22/24 to 20/24 when the Schur path was enabled. Refining against the
    /// true operator removes the dependence on how full the update layer is.
    ///
    /// Mirrors the convex IPM's `solve_refined`, for the same reason: a single
    /// back-solve near a degenerate optimum loses digits.
    pub fn solve(&self, linsol: &mut LinearSolver, rhs: &mut [Number]) -> Result<(), QpError> {
        /// Refinement passes.
        ///
        /// Two used to be the cap, on the reasoning quoted above — "the drift
        /// is a small relative perturbation, not a conditioning wall". That
        /// holds for the update layer's own drift and does not hold once the
        /// objective is large: with `‖P‖ ≈ 2.4e8` the base factor's H block
        /// dwarfs the O(1) constraint rows the SMW columns carry, `S` is
        /// correspondingly ill-conditioned, and the first two passes take the
        /// residual from `1.7e7` down to only `1.3e3` — a *relative* KKT
        /// residual of `7e-5`, handed to the active-set loop as an exact
        /// direction (gh#958).
        ///
        /// The loop stops on `IR_MIN_GAIN` as soon as a pass buys less than a
        /// factor of two, so a well-scaled solve still converges in one or two
        /// passes and pays nothing for this ceiling; it is reached only where
        /// refinement is still making progress, which is exactly where it is
        /// worth spending. Measured on gh#958's 4-variable rank-deficient QP,
        /// `‖P‖ ≈ 2.4e8`: the passes actually used run 1–12 across the solve,
        /// the residual lands at `1e-10`–`1e-17` against `‖b‖ ≈ 1e6`–`1e7`,
        /// and the engine goes from `numerical_failure` at a point violating
        /// `Gx ≤ h` by `3.5e-3` to the true optimum at `12` iterations.
        ///
        /// **Where the number comes from.** Swept against
        /// `pounce-convex/tests/issue958_objective_scale_invariance.rs`, the
        /// verdict is a step function: caps 2, 4, 6 and 8 all leave at least
        /// one arm red, 10 is the first that passes, and 16 and 24 are
        /// bit-identical to 10 — the answer saturates there. 12 is the first
        /// round value above the saturation point, chosen so the ceiling has
        /// margin over the measurement rather than sitting on its edge. It is
        /// a ceiling, not a target: `IR_MIN_GAIN` decides when to stop, and on
        /// every solve that does not need the passes it stops long before
        /// this.
        const IR_MAX_PASSES: usize = 12;
        /// Minimum fractional residual reduction for a pass to count as
        /// progress. Below this the correction is round-off, not signal.
        const IR_MIN_GAIN: Number = 0.5;

        let b = rhs.to_vec();
        self.solve_once(linsol, rhs)?;

        // Deliberately NOT gated on `‖b‖`. The phase-1 elastic system carries
        // the penalty `γ = 1e6` in its RHS, so a `‖b‖`-relative test treats an
        // absolute residual of ~1e-8 as converged — and on an operator that
        // `γ` has made this ill-conditioned, 1e-8 of residual is ~1e-6 of
        // *solution* error, which is precisely the drift being fixed. Refine
        // while the residual keeps shrinking instead, and stop when it does
        // not: that is scale-free and cannot loop.
        //
        // **Best-so-far**, so refinement can only improve the answer. The loop
        // applies a correction before it can know whether that correction
        // helped, and near the precision floor one does not: the next pass
        // measures the *larger* residual, breaks on `IR_MIN_GAIN`, and leaves
        // the worse iterate behind. Measured on gh#958, a pass took the
        // residual from `4.5e5` to `7.4e5` and that was what came back. It is
        // also what makes the ceiling above safe to raise: an extra pass can
        // now cost time, and nothing else.
        //
        // One buffer, filled by `copy_from_slice`, so this is one allocation
        // per solve — the same order as `b` above — rather than one per pass.
        let mut best_x: Vec<Number> = rhs.to_vec();
        let mut best_norm = Number::INFINITY;
        let mut prev = Number::INFINITY;
        for _ in 0..IR_MAX_PASSES {
            let Some(kx) = self.kw_matvec(rhs) else { break };
            let mut r: Vec<Number> = b.iter().zip(kx.iter()).map(|(&bi, &ki)| bi - ki).collect();
            let r_norm = r.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            if !r_norm.is_finite() {
                break;
            }
            if r_norm < best_norm {
                best_norm = r_norm;
                best_x.copy_from_slice(rhs);
            }
            if r_norm == 0.0 {
                break;
            }
            // No meaningful progress ⇒ further passes are noise.
            if r_norm > prev * (1.0 - IR_MIN_GAIN) {
                break;
            }
            prev = r_norm;
            // A correction that cannot be computed leaves the unrefined answer
            // in place rather than failing the solve — it is still usable, and
            // the caller's own verification decides whether it is good enough.
            if self.solve_once(linsol, &mut r).is_err() {
                break;
            }
            for (xi, &dxi) in rhs.iter_mut().zip(r.iter()) {
                *xi += dxi;
            }
        }
        // The loop measures the residual at the TOP of a pass, so whatever
        // correction it applied last was never weighed. Weigh it here — one
        // matvec, no back-solve — and fall back to the best iterate when the
        // last correction was not an improvement. `is_nan()` is spelled out
        // rather than folded into a negated comparison so a NaN residual takes
        // the fallback explicitly.
        let final_norm = self
            .kw_matvec(rhs)
            .map(|kx| {
                b.iter()
                    .zip(kx.iter())
                    .map(|(&bi, &ki)| (bi - ki).abs())
                    .fold(0.0_f64, f64::max)
            })
            .unwrap_or(Number::INFINITY);
        if final_norm.is_nan() || final_norm > best_norm {
            rhs.copy_from_slice(&best_x);
        }
        Ok(())
    }

    /// One unrefined SMW solve: `x = K_0⁻¹b − K_0⁻¹U · S⁻¹ · VᵀK_0⁻¹b`.
    fn solve_once(&self, linsol: &mut LinearSolver, rhs: &mut [Number]) -> Result<(), QpError> {
        if rhs.len() != self.dim {
            return Err(QpError::DimensionMismatch(format!(
                "Schur solve RHS length {} but K_max dim is {}",
                rhs.len(),
                self.dim
            )));
        }
        // z = K_0⁻¹ rhs
        linsol.resolve(rhs)?;
        if self.s_dim == 0 {
            return Ok(());
        }
        // w = Vᵀ z
        let mut w = vec![0.0; self.s_dim];
        for (j, wj) in w.iter_mut().enumerate() {
            *wj = self.v_col(j).dot_dense(rhs);
        }
        // y = S⁻¹ w
        let y = small_dense_lu_solve(&self.s_matrix, self.s_dim, &w)?;
        // x = z − K_0⁻¹ U y = z − Σ y_j · kinv_u_cols[j]
        for (kinv_uj, &y_j) in self.kinv_u_cols.iter().zip(y.iter()) {
            kinv_uj.sub_scaled_from(y_j, rhs);
        }
        Ok(())
    }

    /// Should the caller refactor? Returns true when the Schur
    /// block has reached the per-options threshold.
    pub fn needs_reset(&self, opts: &QpOptions) -> bool {
        self.s_dim >= opts.max_schur_updates_before_refactor as usize
    }

    /// Whether one more update fits before [`Self::needs_reset`] would ask
    /// for a refactorization anyway — so a caller that refactors on a full
    /// block does not pay the update's back-solves first.
    pub fn has_room_for_update(&self, opts: &QpOptions) -> bool {
        self.s_dim + 2 < opts.max_schur_updates_before_refactor as usize
    }

    pub fn n_schur_updates(&self) -> u32 {
        (self.s_dim / 2) as u32
    }

    /// Negative eigenvalues of `K_W = K_0 + UVᵀ`, from the base factor's
    /// count and the small Schur block alone — no factorization (gh#1206
    /// P1). `None` when the block cannot settle it: `K_W` singular or so
    /// close to it that the count would be a coin flip, or `S` failing the
    /// symmetry it has in exact arithmetic.
    ///
    /// **Derivation.** The update is `UVᵀ = Σ_k (u_k v_kᵀ + v_k u_kᵀ)` with
    /// `U = [u_0, v_0, u_1, v_1, …]` and `V = [v_0, u_0, …] = U C`, where
    /// `C = blockdiag([[0, 1], [1, 0]])` is symmetric with `C² = I`. So
    /// `K_W = K_0 + U C Uᵀ`, a symmetric rank-`2k` update, and the
    /// non-symmetric `S = I + VᵀK_0⁻¹U = I + C UᵀK_0⁻¹U` is `C·G` with
    /// `G = C + UᵀK_0⁻¹U` symmetric. Border `K_0` with
    ///
    /// ```text
    ///     M = ┌ K_0   U  ┐
    ///         └ Uᵀ   −C  ┘
    /// ```
    ///
    /// The Schur complement of `K_0` in `M` is `−C − UᵀK_0⁻¹U = −G`; that of
    /// `−C` (nonsingular, `(−C)⁻¹ = −C`) is `K_0 + U C Uᵀ = K_W`. Haynsworth
    /// inertia additivity applied both ways gives
    /// `In(K_0) + In(−G) = In(−C) + In(K_W)`, and `−C` has `k` eigenvalues of
    /// each sign, so
    ///
    /// ```text
    ///     neg(K_W) = neg(K_0) + pos(G) − k,   zero(K_W) = zero(K_0) + zero(G).
    /// ```
    ///
    /// This is Gill, Murray, Saunders & Wright (1990), "A Schur-complement
    /// method for sparse quadratic programming", for the rank-2 form used
    /// here rather than their bordered one. `G` is `S` with its rows swapped
    /// in pairs (row `2i` of `G` is row `2i + 1` of `S`), so no new data is
    /// needed; its eigenvalues come from a cyclic Jacobi sweep, which is
    /// `O(s_dim³)` on a block of at most `max_schur_updates_before_refactor`.
    pub fn updated_neg_evals(&self) -> Option<i32> {
        let s = self.s_dim;
        if s == 0 {
            return Some(self.base_neg);
        }
        let mut g = vec![0.0; s * s];
        for i in 0..s {
            let src = i ^ 1;
            g[i * s..(i + 1) * s].copy_from_slice(&self.s_matrix[src * s..(src + 1) * s]);
        }
        let g_max = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        if !g_max.is_finite() {
            return None;
        }
        // Symmetric in exact arithmetic; an asymmetry above round-off means
        // the cached columns no longer describe one operator.
        for i in 0..s {
            for j in 0..i {
                let (a, b) = (g[i * s + j], g[j * s + i]);
                if (a - b).abs() > SCHUR_SYMMETRY_REL * g_max.max(1.0) {
                    return None;
                }
                let mean = 0.5 * (a + b);
                g[i * s + j] = mean;
                g[j * s + i] = mean;
            }
        }
        let eig = symmetric_eigenvalues(g, s)?;
        let e_max = eig.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let floor = SCHUR_INERTIA_REL * e_max.max(1.0);
        if eig.iter().any(|v| v.abs() <= floor) {
            return None;
        }
        let pos = eig.iter().filter(|&&v| v > 0.0).count() as i32;
        Some(self.base_neg + pos - (s / 2) as i32)
    }

    /// `(changes, stored v entries)` of the update layer — test-only, for
    /// the P3 storage check.
    #[cfg(test)]
    pub(crate) fn stored_update_entries(&self) -> (usize, usize) {
        (
            self.changes.len(),
            self.changes.iter().map(|c| c.v.len()).sum(),
        )
    }
}

/// Row index of `A`'s triplets: row `i`'s entries are
/// `entries[ptr[i]..ptr[i + 1]]`, each a position in `A`'s triplet arrays,
/// in the order they appear there.
///
/// `SchurState::build_k_max_triplet` used to find an active row's entries by
/// scanning every triplet of `A`, once per active row — `O(m · nnz(A))` per
/// reset, against the `O(nnz(A))` of the refactor path's assembly
/// (`kkt::assemble_active_set_kkt`). On gh#1061's SQP step QP (`m = 13 998`,
/// `nnz(A) = 138 654`) that is ~2e9 comparisons each time the base factor is
/// rebuilt, and with an exact Hessian every drop rebuilds it: the reporter
/// measured 2.2–2.4 s per Schur-path refactorization against 0.57 s on the
/// refactor path, flat across every QP. A counting sort is stable, so the
/// assembled triplet — and therefore the factorization — is unchanged.
fn a_row_index(qp: &QpProblem) -> (Vec<usize>, Vec<usize>) {
    let irows = qp.a.irows();
    let mut ptr = vec![0usize; qp.m + 1];
    for &r in irows {
        ptr[r as usize] += 1;
    }
    for i in 0..qp.m {
        ptr[i + 1] += ptr[i];
    }
    let mut next = ptr.clone();
    let mut entries = vec![0usize; irows.len()];
    for (k, &r) in irows.iter().enumerate() {
        let i = (r - 1) as usize;
        entries[next[i]] = k;
        next[i] += 1;
    }
    (ptr, entries)
}

/// Relative pivot below which the Schur block is treated as singular.
///
/// An exact-zero test let a *numerically* singular block through (gh#974). On
/// the convex `SHARE1B` QP, dropping one row left a zero-curvature direction;
/// the refactor path would have met that with inertia control, but the update
/// layer solved `S` with a smallest pivot of `6.3e-16` against `‖S‖max = 208`
/// and returned `‖p‖∞ = 6e19`. The step that followed left rows violated by
/// 8.6e4, and the solve ended `Unbounded` from there. Over the healthy
/// iterations of the same solve the smallest pivot ratio was `3.3e-9`, so this
/// sits four orders from either side. The error it returns says "singular",
/// which is what makes it recoverable: the loop discards the update layer and
/// refactors under inertia control.
const SCHUR_PIVOT_REL: Number = 1e-13;

/// Relative eigenvalue magnitude of `G = C·S` below which
/// [`SchurState::updated_neg_evals`] declines to count signs (gh#1206 P1).
///
/// An eigenvalue that small makes `K_W` numerically singular, and its sign is
/// round-off. Declining costs one refactorization under inertia control —
/// exactly what every drop paid before P1 — so the floor errs high: three
/// orders above [`SCHUR_PIVOT_REL`], and still below the `3.3e-9` smallest
/// healthy pivot ratio measured on gh#974's solve.
const SCHUR_INERTIA_REL: Number = 1e-10;

/// Relative asymmetry of `G = C·S` (symmetric in exact arithmetic) above
/// which the block is not trusted for an inertia count (gh#1206 P1).
const SCHUR_SYMMETRY_REL: Number = 1e-8;

/// Eigenvalues of the dense symmetric `s × s` matrix `a` (row-major), by
/// cyclic Jacobi rotations. `None` if the sweeps do not converge or a
/// non-finite value appears. For the Schur block (`s` at most
/// `max_schur_updates_before_refactor`, 50 by default) this is microseconds
/// against a back-solve.
fn symmetric_eigenvalues(mut a: Vec<Number>, s: usize) -> Option<Vec<Number>> {
    const MAX_SWEEPS: usize = 100;
    let frob: Number = a.iter().map(|v| v * v).sum::<Number>().sqrt();
    if !frob.is_finite() {
        return None;
    }
    for _ in 0..MAX_SWEEPS {
        let mut off = 0.0;
        for i in 0..s {
            for j in 0..i {
                off += a[i * s + j] * a[i * s + j];
            }
        }
        if off.sqrt() <= Number::EPSILON * frob {
            return Some((0..s).map(|i| a[i * s + i]).collect());
        }
        for p in 0..s {
            for q in (p + 1)..s {
                let apq = a[p * s + q];
                if apq == 0.0 {
                    continue;
                }
                let (app, aqq) = (a[p * s + p], a[q * s + q]);
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * c;
                for k in 0..s {
                    let (akp, akq) = (a[k * s + p], a[k * s + q]);
                    a[k * s + p] = c * akp - sn * akq;
                    a[k * s + q] = sn * akp + c * akq;
                }
                for k in 0..s {
                    let (apk, aqk) = (a[p * s + k], a[q * s + k]);
                    a[p * s + k] = c * apk - sn * aqk;
                    a[q * s + k] = sn * apk + c * aqk;
                }
            }
        }
        if a.iter().any(|v| !v.is_finite()) {
            return None;
        }
    }
    None
}

/// In-place Gauss elimination with partial pivoting for a small
/// dense matrix `s` of size `dim × dim` (row-major). Returns
/// `S⁻¹ b`. For the Schur block (`dim ≤ max_schur_updates`, ≤ a
/// few hundred) this is fast enough.
fn small_dense_lu_solve(
    s_in: &[Number],
    dim: usize,
    b_in: &[Number],
) -> Result<Vec<Number>, QpError> {
    let mut a = s_in.to_vec();
    let mut b = b_in.to_vec();
    let smax = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    // Gaussian elimination with partial pivoting.
    for k in 0..dim {
        // Find pivot.
        let mut piv = k;
        let mut piv_mag = a[k * dim + k].abs();
        for i in (k + 1)..dim {
            let v = a[i * dim + k].abs();
            if v > piv_mag {
                piv_mag = v;
                piv = i;
            }
        }
        if piv_mag <= SCHUR_PIVOT_REL * smax.max(1.0) {
            return Err(QpError::LinearSolverFailure(format!(
                "Schur block is singular at column {k}"
            )));
        }
        if piv != k {
            for j in 0..dim {
                a.swap(k * dim + j, piv * dim + j);
            }
            b.swap(k, piv);
        }
        // Eliminate below.
        let pivot = a[k * dim + k];
        for i in (k + 1)..dim {
            let m = a[i * dim + k] / pivot;
            for j in k..dim {
                a[i * dim + j] -= m * a[k * dim + j];
            }
            b[i] -= m * b[k];
        }
    }
    // Back-substitute.
    let mut x = vec![0.0; dim];
    for k in (0..dim).rev() {
        let mut s = b[k];
        for j in (k + 1)..dim {
            s -= a[k * dim + j] * x[j];
        }
        x[k] = s / a[k * dim + k];
    }
    Ok(x)
}
