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

    /// U columns accumulated since the last reset. Each column is
    /// length `dim`. Per working-set change, two columns appended:
    /// first `u = e_p`, then `v` (the row-difference vector).
    u_cols: Vec<Vec<Number>>,
    v_cols: Vec<Vec<Number>>,

    /// `K_0⁻¹ U_j` for each column of `U`, cached at apply-time so
    /// the per-iteration solve doesn't re-do them.
    kinv_u_cols: Vec<Vec<Number>>,

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
}

/// One half of a rank-2 update vector pair.
struct UpdateVectors {
    u: Vec<Number>,
    v: Vec<Number>,
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
            u_cols: Vec::new(),
            v_cols: Vec::new(),
            kinv_u_cols: Vec::new(),
            s_matrix: Vec::new(),
            s_dim: 0,
            base_kkt: None,
            shift: 0.0,
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

    /// Build the K_max triplet for the given active-slot pattern.
    /// Active slots carry the actual constraint row; inactive
    /// slots carry the `(p, p) = 1` sentinel.
    fn build_k_max_triplet(&self, qp: &QpProblem, slot_active: &[bool]) -> KktTriplet {
        let n_i = self.n as Index;
        let mut irn = Vec::new();
        let mut jcn = Vec::new();
        let mut vals = Vec::new();

        // H block (lower-triangle 1-based; normalize defensively).
        let h_irows = qp.h.irows();
        let h_jcols = qp.h.jcols();
        let h_vals = qp.h.values();
        for k in 0..h_irows.len() {
            let i = h_irows[k];
            let j = h_jcols[k];
            let (lo, hi) = if i >= j { (j, i) } else { (i, j) };
            irn.push(hi);
            jcn.push(lo);
            vals.push(h_vals[k]);
        }

        // One row per slot.
        let a_irows = qp.a.irows();
        let a_jcols = qp.a.jcols();
        let a_vals = qp.a.values();
        for slot in 0..self.m_total {
            let saddle_row = n_i + (slot as Index) + 1; // 1-based row in K_max
            if slot_active[slot] {
                if self.slot_is_general(slot) {
                    // Iterate A entries and pick the row.
                    let slot_1based = (slot + 1) as Index;
                    for k in 0..a_irows.len() {
                        if a_irows[k] == slot_1based {
                            // (saddle_row, jcol, value) — saddle_row > n ≥ jcol ⇒ lower-tri.
                            irn.push(saddle_row);
                            jcn.push(a_jcols[k]);
                            vals.push(a_vals[k]);
                        }
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
    ) -> UpdateVectors {
        let p = self.n + slot; // 0-based row in K_max
        let mut u = vec![0.0; self.dim];
        u[p] = 1.0;

        let mut v = vec![0.0; self.dim];
        if self.slot_is_general(slot) {
            let row_1based = (slot + 1) as Index;
            let sign = if going_active { 1.0 } else { -1.0 };
            for k in 0..qp.a.irows().len() {
                if qp.a.irows()[k] == row_1based {
                    let col_0 = (qp.a.jcols()[k] - 1) as usize;
                    v[col_0] += sign * qp.a.values()[k];
                }
            }
            v[p] = if going_active { -0.5 } else { 0.5 };
        } else {
            let var = slot - self.m;
            let sign = if going_active { 1.0 } else { -1.0 };
            v[var] = sign;
            v[p] = if going_active { -0.5 } else { 0.5 };
        }

        UpdateVectors { u, v }
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
        let mut kkt = self.build_k_max_triplet(qp, &self.base_active);
        let mut rhs = vec![0.0; self.dim];

        // First attempt: no shift.
        let mut current = 0.0;
        let mut last_err: Option<QpError>;
        match linsol.factorize_and_solve(&kkt, &mut rhs, Some(expected_neg)) {
            Ok(()) => {
                self.u_cols.clear();
                self.v_cols.clear();
                self.kinv_u_cols.clear();
                self.s_matrix.clear();
                self.s_dim = 0;
                self.base_kkt = Some(kkt.clone());
                self.shift = 0.0;
                return Ok(());
            }
            Err(ref e) if e.is_recoverable_factorization_failure() => {
                last_err = Some(e.clone());
            }
            Err(e) => return Err(e),
        }

        let mut next = opts.inertia_shift_initial;
        for _ in 0..opts.inertia_max_shifts {
            kkt.add_h_diagonal_shift(self.n, next - current);
            current = next;
            let mut rhs_local = vec![0.0; self.dim];
            match linsol.factorize_and_solve(&kkt, &mut rhs_local, Some(expected_neg)) {
                Ok(()) => {
                    self.u_cols.clear();
                    self.v_cols.clear();
                    self.kinv_u_cols.clear();
                    self.s_matrix.clear();
                    self.s_dim = 0;
                    self.base_kkt = Some(kkt.clone());
                    self.shift = current;
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
        let UpdateVectors { u, v } = self.build_update_vectors(qp, slot, going_active);
        let mut kinv_u = u.clone();
        linsol.resolve(&mut kinv_u)?;
        let mut kinv_v = v.clone();
        linsol.resolve(&mut kinv_v)?;

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
        let v_new_rows: [&[Number]; 2] = [&v, &u];
        let u_new_cols: [&[Number]; 2] = [&kinv_u, &kinv_v];

        // Top-right block: old V rows × new K_0⁻¹ U cols.
        for i in 0..old_dim {
            new_s[i * new_dim + old_dim] = dot(&self.v_cols[i], &kinv_u);
            new_s[i * new_dim + old_dim + 1] = dot(&self.v_cols[i], &kinv_v);
        }
        // Bottom-left block: new V rows × old K_0⁻¹ U cols.
        for ii in 0..2 {
            for j in 0..old_dim {
                new_s[(old_dim + ii) * new_dim + j] = dot(v_new_rows[ii], &self.kinv_u_cols[j]);
            }
        }
        // Bottom-right block: 2×2 from new V rows × new K_0⁻¹ U cols.
        for ii in 0..2 {
            for jj in 0..2 {
                let entry = dot(v_new_rows[ii], u_new_cols[jj]);
                let identity = if ii == jj { 1.0 } else { 0.0 };
                new_s[(old_dim + ii) * new_dim + old_dim + jj] = entry + identity;
            }
        }

        self.u_cols.push(u);
        self.u_cols.push(v.clone());
        self.v_cols.push(v);
        self.v_cols.push(self.u_cols[self.u_cols.len() - 2].clone()); // = the just-pushed `u`
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
            let c = dot(&self.v_cols[j], x);
            if c != 0.0 {
                let uj = &self.u_cols[j];
                for (yi, &uij) in y.iter_mut().zip(uj.iter()) {
                    *yi += c * uij;
                }
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
        let z = rhs.to_vec();
        let mut w = vec![0.0; self.s_dim];
        for j in 0..self.s_dim {
            w[j] = dot(&self.v_cols[j], &z);
        }
        // y = S⁻¹ w
        let y = small_dense_lu_solve(&self.s_matrix, self.s_dim, &w)?;
        // x = z − K_0⁻¹ U y = z − Σ y_j · kinv_u_cols[j]
        for j in 0..self.s_dim {
            let y_j = y[j];
            let kinv_uj = &self.kinv_u_cols[j];
            for i in 0..self.dim {
                rhs[i] -= y_j * kinv_uj[i];
            }
        }
        Ok(())
    }

    /// Should the caller refactor? Returns true when the Schur
    /// block has reached the per-options threshold.
    pub fn needs_reset(&self, opts: &QpOptions) -> bool {
        self.s_dim >= opts.max_schur_updates_before_refactor as usize
    }

    pub fn n_schur_updates(&self) -> u32 {
        (self.s_dim / 2) as u32
    }
}

fn dot(a: &[Number], b: &[Number]) -> Number {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
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
        if piv_mag == 0.0 {
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
