//! Block-triangular / Schur KKT linear solver (pounce#180 item 2, Phase 1).
//!
//! Given a symmetric-indefinite KKT matrix and a caller-supplied partition of
//! its indices into an eliminated "F" block and a small Schur "S" block, this
//! solves the system by the Schur-complement method:
//!
//! ```text
//!   M = [ A_FF  A_FS ]   (symmetric; A_SF = A_FSᵀ)
//!       [ A_SF  A_SS ]
//!
//!   S = A_SS − A_SF · A_FF⁻¹ · A_FS          (dense, n_s × n_s)
//! ```
//!
//! and recovers the full-system inertia **a priori via Sylvester's law of
//! inertia** (Haynsworth additivity): `inertia(M) = inertia(A_FF) + inertia(S)`.
//! Only the two diagonal blocks are factorized — `A_FF` sparsely (feral
//! multifrontal) and the small `S` densely. This is the reduced-space /
//! variable-aggregation path of Parker, Garcia & Bent (arXiv:2602.17968).
//!
//! **Design (validated by the Phase-0 spike, `dev-notes/research/`):** we form
//! `S` ourselves via `n_s` back-solves against a *standalone* `A_FF`
//! factorization, using only feral's proven `Solver::factor` / `solve`. This
//! needs no un-exported partial back-solve, so it carries no FERAL dependency
//! beyond the stable factor/solve API. Correctness and Sylvester inertia were
//! confirmed to machine precision (incl. an indefinite `A_FF`).
//!
//! **Cost / the dense trap:** every `n_s`-dependent cost is confined to the
//! (intended-small) Schur block — dense `S` is `O(n_s²)` storage / `O(n_s³)`
//! factor, forming it costs `n_s` back-solves plus an `O(n_f · n_s)` transient
//! `W = A_FF⁻¹A_FS`. The method beats a monolithic factorization only when
//! `n_s ≪ n_f`; the caller (Phase 2 `SchurAugSystemSolver`) is responsible for
//! gating on that and falling back to the standard solver otherwise.

use feral::{CscMatrix, FactorStatus, Solver};
use pounce_common::types::{Index, Number};
use pounce_linsol::ESymSolverStatus;

use crate::{configure_solver, FeralConfig};

/// Schur-complement KKT solver over a caller-supplied F/S partition.
///
/// Lifecycle mirrors [`crate::FeralSolverInterface`]:
/// [`Self::initialize_structure`] pins the pattern + partition,
/// [`Self::values_array_mut`] receives the KKT nonzeros (same order as the
/// triplet passed to `initialize_structure`), [`Self::factor`] factorizes both
/// blocks and combines inertia, and [`Self::backsolve`] applies the block
/// back-substitution in place.
pub struct FeralSchurSolver {
    cfg: FeralConfig,

    dim: usize,
    n_f: usize,
    n_s: usize,

    /// F-local index → KKT index.
    f_kkt: Vec<usize>,
    /// S-local index → KKT index.
    s_kkt: Vec<usize>,

    // --- A_FF block: f-local lower-triangle triplet; values refilled per factor.
    ff_rows: Vec<usize>,
    ff_cols: Vec<usize>,
    ff_src: Vec<usize>, // input-nnz position feeding each A_FF value
    ff_vals: Vec<Number>,

    // --- coupling A_FS[f_local, s_local]; values refilled per factor.
    coup_f: Vec<usize>,
    coup_s: Vec<usize>,
    coup_src: Vec<usize>,

    // --- A_SS: s-local lower-triangle triplet.
    ss_rows: Vec<usize>,
    ss_cols: Vec<usize>,
    ss_src: Vec<usize>,

    /// Caller-written KKT nonzeros, indexed as the `initialize_structure` triplet.
    values: Vec<Number>,

    ff_solver: Solver,
    s_solver: Solver,
    /// Cached block matrices for iterative-refinement solves (feral needs the
    /// original matrix to compute residuals). Kept when `cfg.refine`.
    ff_matrix: Option<CscMatrix>,
    s_matrix: Option<CscMatrix>,

    // Scratch reused across factors.
    afs: Vec<Number>,  // n_f × n_s column-major (also holds W = A_FF⁻¹A_FS)
    astw: Vec<Number>, // n_s × n_s column-major (A_SFᵀ W), then S base
    ass: Vec<Number>,  // n_s × n_s column-major (dense A_SS, lower filled)

    negevals: Index,
    have_factor: bool,
    initialized: bool,
    last_status: ESymSolverStatus,
}

impl FeralSchurSolver {
    pub fn new(cfg: FeralConfig) -> Self {
        let ff_solver = configure_solver(&cfg);
        let s_solver = configure_solver(&cfg);
        Self {
            cfg,
            dim: 0,
            n_f: 0,
            n_s: 0,
            f_kkt: Vec::new(),
            s_kkt: Vec::new(),
            ff_rows: Vec::new(),
            ff_cols: Vec::new(),
            ff_src: Vec::new(),
            ff_vals: Vec::new(),
            coup_f: Vec::new(),
            coup_s: Vec::new(),
            coup_src: Vec::new(),
            ss_rows: Vec::new(),
            ss_cols: Vec::new(),
            ss_src: Vec::new(),
            values: Vec::new(),
            ff_solver,
            s_solver,
            ff_matrix: None,
            s_matrix: None,
            afs: Vec::new(),
            astw: Vec::new(),
            ass: Vec::new(),
            negevals: 0,
            have_factor: false,
            initialized: false,
            last_status: ESymSolverStatus::Success,
        }
    }

    /// Pin the KKT sparsity pattern and the F/S partition.
    ///
    /// `ia` / `ja` are the **1-based lower-triangle** triplet of the full KKT
    /// (the exact layout `StdAugSystemSolver` assembles). `schur_indices` are
    /// the **0-based KKT indices** (`0..dim`) forming the Schur block `S`; they
    /// need not be contiguous. Returns [`ESymSolverStatus::FatalError`] on a
    /// malformed partition (out-of-range / duplicate index, empty `S`, or
    /// `S == everything` — in which case the caller should use the standard
    /// full-space solver instead).
    pub fn initialize_structure(
        &mut self,
        dim: Index,
        ia: &[Index],
        ja: &[Index],
        schur_indices: &[usize],
    ) -> ESymSolverStatus {
        let dim = dim as usize;
        if ia.len() != ja.len() {
            return self.fail();
        }
        // Validate the Schur set is a strict, non-empty subset of 0..dim.
        let mut is_schur = vec![false; dim];
        for &s in schur_indices {
            if s >= dim || is_schur[s] {
                return self.fail(); // out of range or duplicate
            }
            is_schur[s] = true;
        }
        let n_s = schur_indices.len();
        if n_s == 0 || n_s == dim {
            return self.fail();
        }
        let n_f = dim - n_s;

        // Build F/S local↔KKT maps. F-local and S-local both follow increasing
        // KKT order, which keeps a lower-triangle input entry lower-triangle in
        // each block's local coordinates.
        let mut f_local = vec![usize::MAX; dim];
        let mut s_local = vec![usize::MAX; dim];
        let mut f_kkt = Vec::with_capacity(n_f);
        let mut s_kkt = Vec::with_capacity(n_s);
        for (i, &sch) in is_schur.iter().enumerate() {
            if sch {
                s_local[i] = s_kkt.len();
                s_kkt.push(i);
            } else {
                f_local[i] = f_kkt.len();
                f_kkt.push(i);
            }
        }

        // Split the triplet into A_FF / coupling / A_SS, recording for each the
        // input-nnz position so `factor` can scatter the value buffer.
        let (mut ff_rows, mut ff_cols, mut ff_src) = (Vec::new(), Vec::new(), Vec::new());
        let (mut coup_f, mut coup_s, mut coup_src) = (Vec::new(), Vec::new(), Vec::new());
        let (mut ss_rows, mut ss_cols, mut ss_src) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..ia.len() {
            let i = (ia[k] - 1) as usize;
            let j = (ja[k] - 1) as usize;
            if i >= dim || j >= dim {
                return self.fail();
            }
            match (is_schur[i], is_schur[j]) {
                (false, false) => {
                    ff_rows.push(f_local[i]);
                    ff_cols.push(f_local[j]);
                    ff_src.push(k);
                }
                (true, true) => {
                    ss_rows.push(s_local[i]);
                    ss_cols.push(s_local[j]);
                    ss_src.push(k);
                }
                // Mixed: one endpoint in F, the other in S. Store as
                // A_FS[f_local, s_local] regardless of which was row/col
                // (M is symmetric, input is lower-triangle so only one of
                // (i,j)/(j,i) is present).
                (false, true) => {
                    coup_f.push(f_local[i]);
                    coup_s.push(s_local[j]);
                    coup_src.push(k);
                }
                (true, false) => {
                    coup_f.push(f_local[j]);
                    coup_s.push(s_local[i]);
                    coup_src.push(k);
                }
            }
        }

        self.dim = dim;
        self.n_f = n_f;
        self.n_s = n_s;
        self.f_kkt = f_kkt;
        self.s_kkt = s_kkt;
        self.ff_vals = vec![0.0; ff_src.len()];
        self.ff_rows = ff_rows;
        self.ff_cols = ff_cols;
        self.ff_src = ff_src;
        self.coup_f = coup_f;
        self.coup_s = coup_s;
        self.coup_src = coup_src;
        self.ss_rows = ss_rows;
        self.ss_cols = ss_cols;
        self.ss_src = ss_src;
        self.values = vec![0.0; ia.len()];
        self.afs = vec![0.0; n_f * n_s];
        self.astw = vec![0.0; n_s * n_s];
        self.ass = vec![0.0; n_s * n_s];
        self.ff_matrix = None;
        self.s_matrix = None;
        self.have_factor = false;
        self.initialized = true;
        self.last_status = ESymSolverStatus::Success;
        ESymSolverStatus::Success
    }

    /// Mutable view of the KKT value array (same order as the
    /// `initialize_structure` triplet). The caller writes nonzeros here before
    /// each [`Self::factor`].
    pub fn values_array_mut(&mut self) -> &mut [Number] {
        &mut self.values
    }

    /// Factor both blocks and combine inertia. `check_neg_evals` verifies the
    /// combined negative-eigenvalue count equals `num_neg_evals` (the IPM's
    /// expected `m`); on mismatch the status is [`ESymSolverStatus::WrongInertia`].
    /// A singular `A_FF` or `S` (zero pivot) yields [`ESymSolverStatus::Singular`]
    /// so the IPM's `perturb_for_singular` path fires, exactly as the monolithic
    /// backend does.
    pub fn factor(&mut self, check_neg_evals: bool, num_neg_evals: Index) -> ESymSolverStatus {
        if !self.initialized {
            return self.set_status(ESymSolverStatus::FatalError);
        }
        // 1. Scatter values into A_FF and factor it (sparse).
        for p in 0..self.ff_src.len() {
            self.ff_vals[p] = self.values[self.ff_src[p]];
        }
        let ff_mat =
            match CscMatrix::from_triplets(self.n_f, &self.ff_rows, &self.ff_cols, &self.ff_vals) {
                Ok(m) => m,
                Err(_) => return self.set_status(ESymSolverStatus::FatalError),
            };
        let (neg_ff, singular_ff) = match self.ff_solver.factor(&ff_mat, None) {
            FactorStatus::Success => match self.ff_solver.inertia() {
                Some(i) => (i.negative, i.zero > 0),
                None => (self.ff_solver.num_negative_eigenvalues(), false),
            },
            FactorStatus::Singular => return self.set_status(ESymSolverStatus::Singular),
            FactorStatus::WrongInertia { .. } | FactorStatus::FatalError(_) => {
                return self.set_status(ESymSolverStatus::FatalError)
            }
        };
        self.ff_matrix = Some(ff_mat);
        if singular_ff || self.pivot_below_floor(&self.ff_solver) {
            return self.set_status(ESymSolverStatus::Singular);
        }

        // 2. Form S = A_SS − A_SFᵀ·(A_FF⁻¹·A_FS). One dense back-solve batch.
        for v in self.afs.iter_mut() {
            *v = 0.0;
        }
        for p in 0..self.coup_src.len() {
            self.afs[self.coup_f[p] + self.coup_s[p] * self.n_f] += self.values[self.coup_src[p]];
        }
        // W = A_FF⁻¹ A_FS (n_f × n_s — the transient, gated on small n_s).
        // Refine against the original A_FF when `cfg.refine`, matching the
        // monolithic backend — S's accuracy hinges on W's.
        let w = {
            let r = match (self.cfg.refine, self.ff_matrix.as_ref()) {
                (true, Some(m)) => self.ff_solver.solve_many_refined(m, &self.afs, self.n_s),
                _ => self.ff_solver.solve_many(&self.afs, self.n_s),
            };
            match r {
                Ok(w) => w,
                Err(_) => return self.set_status(ESymSolverStatus::FatalError),
            }
        };
        // astw[i,j] = Σ_f A_FS[f,i] · W[f,j]  (= (A_SFᵀ W)[i,j], symmetric).
        for v in self.astw.iter_mut() {
            *v = 0.0;
        }
        for p in 0..self.coup_src.len() {
            let s = self.coup_s[p];
            let f = self.coup_f[p];
            let val = self.values[self.coup_src[p]];
            for j in 0..self.n_s {
                self.astw[s + j * self.n_s] += val * w[f + j * self.n_f];
            }
        }
        // Dense A_SS (lower-triangle input).
        for v in self.ass.iter_mut() {
            *v = 0.0;
        }
        for p in 0..self.ss_src.len() {
            self.ass[self.ss_rows[p] + self.ss_cols[p] * self.n_s] += self.values[self.ss_src[p]];
        }
        // S lower-triangle = A_SS − astw.
        let n_s = self.n_s;
        let (mut sr, mut sc, mut sv) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n_s {
            for i in j..n_s {
                let val = self.ass[i + j * n_s] - self.astw[i + j * n_s];
                sr.push(i);
                sc.push(j);
                sv.push(val);
            }
        }
        let s_mat = match CscMatrix::from_triplets(n_s, &sr, &sc, &sv) {
            Ok(m) => m,
            Err(_) => return self.set_status(ESymSolverStatus::FatalError),
        };
        let (neg_s, singular_s) = match self.s_solver.factor(&s_mat, None) {
            FactorStatus::Success => match self.s_solver.inertia() {
                Some(i) => (i.negative, i.zero > 0),
                None => (self.s_solver.num_negative_eigenvalues(), false),
            },
            FactorStatus::Singular => return self.set_status(ESymSolverStatus::Singular),
            FactorStatus::WrongInertia { .. } | FactorStatus::FatalError(_) => {
                return self.set_status(ESymSolverStatus::FatalError)
            }
        };
        self.s_matrix = Some(s_mat);
        if singular_s || self.pivot_below_floor(&self.s_solver) {
            return self.set_status(ESymSolverStatus::Singular);
        }

        // 3. Combine inertia via Sylvester's law.
        self.negevals = (neg_ff + neg_s) as Index;
        self.have_factor = true;
        if check_neg_evals && self.negevals != num_neg_evals {
            // Factor is valid and usable; the caller's perturbation loop will
            // bump δ and re-factor. Mirror the monolithic backend's contract.
            return self.set_status(ESymSolverStatus::WrongInertia);
        }
        self.set_status(ESymSolverStatus::Success)
    }

    /// Block back-substitution, in place, for `nrhs` right-hand sides packed
    /// column-major (`dim` per column, in KKT/original index order). Requires a
    /// prior successful [`Self::factor`].
    pub fn backsolve(&self, nrhs: Index, rhs: &mut [Number]) -> ESymSolverStatus {
        let nrhs = nrhs as usize;
        if !self.have_factor || rhs.len() != self.dim * nrhs {
            return ESymSolverStatus::FatalError;
        }
        let (n_f, n_s) = (self.n_f, self.n_s);
        let mut b_f = vec![0.0; n_f];
        let mut b_s = vec![0.0; n_s];
        for c in 0..nrhs {
            let col = &mut rhs[c * self.dim..(c + 1) * self.dim];
            for (fl, &k) in self.f_kkt.iter().enumerate() {
                b_f[fl] = col[k];
            }
            for (sl, &k) in self.s_kkt.iter().enumerate() {
                b_s[sl] = col[k];
            }
            // u = A_FF⁻¹ b_F
            let Some(u) = self.ff_solve(&b_f) else {
                return ESymSolverStatus::FatalError;
            };
            // r_S = b_S − A_SF u
            let mut r_s = b_s.clone();
            for p in 0..self.coup_src.len() {
                r_s[self.coup_s[p]] -= self.values[self.coup_src[p]] * u[self.coup_f[p]];
            }
            // x_S = S⁻¹ r_S
            let Some(x_s) = self.s_solve(&r_s) else {
                return ESymSolverStatus::FatalError;
            };
            // x_F = A_FF⁻¹ (b_F − A_FS x_S)
            let mut rhs_f = b_f.clone();
            for p in 0..self.coup_src.len() {
                rhs_f[self.coup_f[p]] -= self.values[self.coup_src[p]] * x_s[self.coup_s[p]];
            }
            let Some(x_f) = self.ff_solve(&rhs_f) else {
                return ESymSolverStatus::FatalError;
            };
            // Scatter the solution back into the column in KKT order.
            for (fl, &k) in self.f_kkt.iter().enumerate() {
                col[k] = x_f[fl];
            }
            for (sl, &k) in self.s_kkt.iter().enumerate() {
                col[k] = x_s[sl];
            }
        }
        ESymSolverStatus::Success
    }

    pub fn number_of_neg_evals(&self) -> Index {
        self.negevals
    }
    pub fn provides_inertia(&self) -> bool {
        true
    }
    pub fn system_dim(&self) -> Index {
        self.dim as Index
    }
    pub fn schur_dim(&self) -> Index {
        self.n_s as Index
    }
    pub fn last_solve_status(&self) -> ESymSolverStatus {
        self.last_status
    }
    /// Feral delegates all recovery to the IPM's δ-perturbation loop (there is
    /// no higher pivot-quality mode), matching [`crate::FeralSolverInterface`].
    pub fn increase_quality(&mut self) -> bool {
        false
    }

    /// Single-RHS `A_FF⁻¹` solve, iteratively refined against the original
    /// `A_FF` when `cfg.refine` (matching the monolithic backend's default).
    fn ff_solve(&self, rhs: &[Number]) -> Option<Vec<Number>> {
        match (self.cfg.refine, self.ff_matrix.as_ref()) {
            (true, Some(m)) => self.ff_solver.solve_refined(m, rhs),
            _ => self.ff_solver.solve(rhs),
        }
        .ok()
    }
    /// Single-RHS `S⁻¹` solve, refined against the dense `S` when `cfg.refine`.
    fn s_solve(&self, rhs: &[Number]) -> Option<Vec<Number>> {
        match (self.cfg.refine, self.s_matrix.as_ref()) {
            (true, Some(m)) => self.s_solver.solve_refined(m, rhs),
            _ => self.s_solver.solve(rhs),
        }
        .ok()
    }

    fn pivot_below_floor(&self, solver: &Solver) -> bool {
        // MA57 CNTL(2) analog — see `FeralSolverInterface::factor`.
        if self.cfg.singular_pivot_floor > 0.0 {
            if let Some(min_piv) = solver.min_pivot_magnitude() {
                return min_piv < self.cfg.singular_pivot_floor;
            }
        }
        false
    }

    fn fail(&mut self) -> ESymSolverStatus {
        self.initialized = false;
        self.last_status = ESymSolverStatus::FatalError;
        ESymSolverStatus::FatalError
    }
    fn set_status(&mut self, s: ESymSolverStatus) -> ESymSolverStatus {
        self.last_status = s;
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FeralSolverInterface;
    use pounce_linsol::SparseSymLinearSolverInterface;

    /// Build a symmetric-indefinite KKT test matrix as a 1-based lower-triangle
    /// triplet. F = `0..n_f` (tridiagonal, `f_indef` flips the sign of the
    /// second half → indefinite eliminated block); S = `n_f..n_f+n_s` coupled
    /// sparsely to F; A_SS negative diagonal. Returns `(dim, ia, ja, vals)`.
    fn kkt(
        n_f: usize,
        n_s: usize,
        deg: usize,
        f_indef: bool,
    ) -> (Index, Vec<Index>, Vec<Index>, Vec<Number>) {
        let (mut ia, mut ja, mut v) = (Vec::new(), Vec::new(), Vec::new());
        let push = |i: usize,
                    j: usize,
                    x: Number,
                    ia: &mut Vec<Index>,
                    ja: &mut Vec<Index>,
                    v: &mut Vec<Number>| {
            ia.push((i + 1) as Index);
            ja.push((j + 1) as Index);
            v.push(x);
        };
        for i in 0..n_f {
            let d = if f_indef && i >= n_f / 2 { -4.0 } else { 4.0 };
            push(i, i, d, &mut ia, &mut ja, &mut v);
            if i > 0 {
                push(i, i - 1, -1.0, &mut ia, &mut ja, &mut v);
            }
        }
        for s in 0..n_s {
            for k in 0..deg {
                let f = ((s * 7 + k * 101 + 3) * 2_654_435_761usize) % n_f.max(1);
                // lower-triangle: row = n_f+s (larger index) , col = f
                push(n_f + s, f, 0.5 + 0.1 * (k as f64), &mut ia, &mut ja, &mut v);
            }
            push(n_f + s, n_f + s, -1.0, &mut ia, &mut ja, &mut v);
        }
        ((n_f + n_s) as Index, ia, ja, v)
    }

    /// Oracle: solve the same KKT monolithically through `FeralSolverInterface`.
    fn oracle_solve(
        dim: Index,
        ia: &[Index],
        ja: &[Index],
        vals: &[Number],
        rhs: &[Number],
    ) -> (Vec<Number>, Index) {
        let mut s = FeralSolverInterface::with_config(FeralConfig::default());
        assert_eq!(
            s.initialize_structure(dim, ia.len() as Index, ia, ja),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(vals);
        let mut b = rhs.to_vec();
        let st = s.multi_solve(true, ia, ja, 1, &mut b, false, 0);
        assert_eq!(st, ESymSolverStatus::Success);
        (b, s.number_of_neg_evals())
    }

    fn schur_indices_tail(n_f: usize, n_s: usize) -> Vec<usize> {
        (n_f..n_f + n_s).collect()
    }

    fn run_and_check(n_f: usize, n_s: usize, deg: usize, f_indef: bool, schur: &[usize]) {
        let (dim, ia, ja, vals) = kkt(n_f, n_s, deg, f_indef);
        let rhs: Vec<Number> = (0..dim as usize)
            .map(|i| 1.0 + (i % 5) as f64 * 0.25)
            .collect();
        let (x_oracle, neg_oracle) = oracle_solve(dim, &ia, &ja, &vals, &rhs);

        let mut solver = FeralSchurSolver::new(FeralConfig::default());
        assert_eq!(
            solver.initialize_structure(dim, &ia, &ja, schur),
            ESymSolverStatus::Success
        );
        solver.values_array_mut().copy_from_slice(&vals);
        // Check inertia against the oracle's count.
        let st = solver.factor(true, neg_oracle);
        assert_eq!(st, ESymSolverStatus::Success, "factor status");
        assert_eq!(
            solver.number_of_neg_evals(),
            neg_oracle,
            "Sylvester inertia mismatch"
        );

        let mut b = rhs.clone();
        assert_eq!(solver.backsolve(1, &mut b), ESymSolverStatus::Success);
        let err = b
            .iter()
            .zip(&x_oracle)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-8, "solution mismatch: {err:e}");
    }

    #[test]
    fn spd_ff_tail_partition_matches_oracle() {
        run_and_check(6, 2, 2, false, &schur_indices_tail(6, 2));
        run_and_check(200, 8, 4, false, &schur_indices_tail(200, 8));
    }

    #[test]
    fn indefinite_ff_matches_oracle_and_sylvester() {
        // Both A_FF and S carry negative eigenvalues here.
        run_and_check(50, 4, 3, true, &schur_indices_tail(50, 4));
        run_and_check(400, 12, 4, true, &schur_indices_tail(400, 12));
    }

    #[test]
    fn scattered_non_contiguous_schur_set() {
        // Interleaved Schur indices (not a tail) must work: the F/S maps are
        // by KKT order, so a lower-triangle entry stays lower-triangle locally.
        let (n_f_plus_s, n_s) = (60usize, 6usize);
        let n_f = n_f_plus_s - n_s;
        let schur: Vec<usize> = (0..n_s)
            .map(|k| k * 9 + 4)
            .filter(|&x| x < n_f_plus_s)
            .collect();
        run_and_check(n_f, n_s, 3, true, &schur);
    }

    #[test]
    fn multi_rhs_backsolve() {
        let (dim, ia, ja, vals) = kkt(120, 6, 4, true);
        let schur = schur_indices_tail(120, 6);
        let mut solver = FeralSchurSolver::new(FeralConfig::default());
        assert_eq!(
            solver.initialize_structure(dim, &ia, &ja, &schur),
            ESymSolverStatus::Success
        );
        solver.values_array_mut().copy_from_slice(&vals);
        assert_eq!(solver.factor(false, 0), ESymSolverStatus::Success);
        // Two independent RHS packed column-major.
        let n = dim as usize;
        let mut packed = vec![0.0; 2 * n];
        for i in 0..n {
            packed[i] = 1.0 + (i % 3) as f64;
            packed[n + i] = -0.5 + (i % 7) as f64 * 0.1;
        }
        assert_eq!(solver.backsolve(2, &mut packed), ESymSolverStatus::Success);
        for c in 0..2 {
            let rhs_c: Vec<Number> = (0..n)
                .map(|i| {
                    if c == 0 {
                        1.0 + (i % 3) as f64
                    } else {
                        -0.5 + (i % 7) as f64 * 0.1
                    }
                })
                .collect();
            let (x_oracle, _) = oracle_solve(dim, &ia, &ja, &vals, &rhs_c);
            let err = packed[c * n..(c + 1) * n]
                .iter()
                .zip(&x_oracle)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max);
            assert!(err < 1e-8, "col {c} mismatch: {err:e}");
        }
    }

    #[test]
    fn refactor_with_new_values_reuses_pattern() {
        let (dim, ia, ja, mut vals) = kkt(80, 5, 3, true);
        let schur = schur_indices_tail(80, 5);
        let mut solver = FeralSchurSolver::new(FeralConfig::default());
        assert_eq!(
            solver.initialize_structure(dim, &ia, &ja, &schur),
            ESymSolverStatus::Success
        );
        solver.values_array_mut().copy_from_slice(&vals);
        assert_eq!(solver.factor(false, 0), ESymSolverStatus::Success);
        // Perturb the diagonal (same pattern) and re-factor + re-solve.
        for (k, x) in vals.iter_mut().enumerate() {
            if ia[k] == ja[k] {
                *x += 0.3;
            }
        }
        solver.values_array_mut().copy_from_slice(&vals);
        assert_eq!(solver.factor(false, 0), ESymSolverStatus::Success);
        let rhs: Vec<Number> = (0..dim as usize).map(|i| 1.0 + (i % 4) as f64).collect();
        let (x_oracle, _) = oracle_solve(dim, &ia, &ja, &vals, &rhs);
        let mut b = rhs.clone();
        assert_eq!(solver.backsolve(1, &mut b), ESymSolverStatus::Success);
        let err = b
            .iter()
            .zip(&x_oracle)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-8, "post-refactor mismatch: {err:e}");
    }

    #[test]
    fn wrong_inertia_is_flagged() {
        let (dim, ia, ja, vals) = kkt(40, 4, 3, true);
        let schur = schur_indices_tail(40, 4);
        let mut solver = FeralSchurSolver::new(FeralConfig::default());
        assert_eq!(
            solver.initialize_structure(dim, &ia, &ja, &schur),
            ESymSolverStatus::Success
        );
        solver.values_array_mut().copy_from_slice(&vals);
        // Demand the wrong number of negatives → WrongInertia (factor still usable).
        let true_neg = {
            let (_, n) = oracle_solve(dim, &ia, &ja, &vals, &vec![0.0; dim as usize]);
            n
        };
        let st = solver.factor(true, true_neg + 1);
        assert_eq!(st, ESymSolverStatus::WrongInertia);
        assert_eq!(solver.number_of_neg_evals(), true_neg);
    }

    #[test]
    fn malformed_partition_is_rejected() {
        let (dim, ia, ja, _vals) = kkt(10, 2, 2, false);
        let mut solver = FeralSchurSolver::new(FeralConfig::default());
        // Empty Schur set.
        assert_eq!(
            solver.initialize_structure(dim, &ia, &ja, &[]),
            ESymSolverStatus::FatalError
        );
        // Out-of-range index.
        assert_eq!(
            solver.initialize_structure(dim, &ia, &ja, &[dim as usize]),
            ESymSolverStatus::FatalError
        );
        // Duplicate index.
        assert_eq!(
            solver.initialize_structure(dim, &ia, &ja, &[1, 1]),
            ESymSolverStatus::FatalError
        );
        // Whole system as Schur (F empty) → use Std instead.
        let all: Vec<usize> = (0..dim as usize).collect();
        assert_eq!(
            solver.initialize_structure(dim, &ia, &ja, &all),
            ESymSolverStatus::FatalError
        );
    }
}
