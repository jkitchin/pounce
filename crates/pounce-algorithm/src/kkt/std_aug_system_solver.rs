//! Standard augmented-system solver — port of
//! `Algorithm/IpStdAugSystemSolver.{hpp,cpp}`.
//!
//! Flattens the four-block KKT matrix into a single lower-triangular
//! 1-based triplet and hands it to a [`pounce_linsol::TSymLinearSolver`].
//! On the first call the structure is computed (and the linsol's
//! `initialize_structure` is invoked); subsequent calls only refill the
//! values array and call `multi_solve`. Matches the cache/skip logic in
//! upstream `IpStdAugSystemSolver::CreateAugmentedSpace` and
//! `CreateAugmentedSystem`.
//!
//! Sign convention follows upstream:
//!
//! ```text
//!   (1,1) = w_factor·W + diag(D_x + δ_x)
//!   (2,2) = diag(D_s + δ_s)
//!   (3,1) = J_c
//!   (3,3) = -diag(D_c + δ_c)
//!   (4,1) = J_d
//!   (4,2) = -I
//!   (4,4) = -diag(D_d + δ_d)
//! ```
//!
//! Phase-6 first cut: assumes `W` is a [`SymTMatrix`], `J_c`/`J_d` are
//! [`GenTMatrix`], and `D_*` are [`DenseVector`]s — the only concrete
//! types `OrigIpoptNLP` produces. CompoundMatrix/CompoundVector
//! flattening (used by L-BFGS in Phase 8) is deferred.

use crate::kkt::aug_system_solver::{AugSysCoeffs, AugSysRhs, AugSysSol, AugSystemSolver};
use pounce_common::diagnostics::{DiagCategory, DiagnosticsState};
use pounce_common::timing::TimingStatistics;
use pounce_common::types::{Index, Number};
use pounce_linalg::compound_vector::CompoundVector;
use pounce_linalg::dense_vector::DenseVector;
use pounce_linalg::diag_matrix::DiagMatrix;
use pounce_linalg::triplet::{GenTMatrix, SymTMatrix};
use pounce_linalg::Vector;
use pounce_linsol::{ESymSolverStatus, FactorPattern, SymLinearSolver, TSymLinearSolver};
use std::ops::Range;
use std::rc::Rc;

/// Standard augmented-system solver.
pub struct StdAugSystemSolver {
    linsol: TSymLinearSolver,

    /// `true` once the triplet structure has been pinned.
    initialized: bool,
    /// Structural fingerprint `(w_nnz, jc_nnz, jd_nnz, n_x, n_c, n_d)` of
    /// the coefficients the current pinned structure was built from. When
    /// the next solve presents a different fingerprint — e.g. the
    /// limited-memory `LowRankAugSystemSolver` driving this same inner
    /// solver alternately with a Hessian-free `zero_w` block and an
    /// `n`-diagonal quasi-Newton `B0`, which have different W nnz — the
    /// triplet structure (and the backend's symbolic factor) is rebuilt
    /// rather than silently reusing a stale pattern.
    struct_sig: Option<(usize, usize, usize, Index, Index, Index)>,
    n_x: Index,
    n_s: Index,
    n_c: Index,
    n_d: Index,
    /// Total dim = `n_x + n_s + n_c + n_d`.
    dim: Index,

    /// 1-based row indices, length = total triplet nnz.
    irn: Vec<Index>,
    /// 1-based col indices.
    jcn: Vec<Index>,
    /// Working values array reused across calls.
    vals: Vec<Number>,

    // Per-block ranges into `vals` / `irn` / `jcn`.
    w_range: Range<usize>,
    dx_range: Range<usize>,
    ds_range: Range<usize>,
    jc_range: Range<usize>,
    dc_range: Range<usize>,
    jd_range: Range<usize>,
    minus_i_range: Range<usize>,
    dd_range: Range<usize>,

    last_neg_evals: Index,
    last_status: Option<ESymSolverStatus>,

    /// `true` once a successful `solve()` has been completed since the
    /// last reinitialisation or `increase_quality`. Required precondition
    /// for `resolve()` (back-substitution against the cached factor).
    have_factor: bool,

    /// Shared per-solve timing accumulator. `None` until the
    /// algorithm installs it via [`AugSystemSolver::set_timing_stats`];
    /// when `None`, both `solve` and `resolve` skip the timing bumps.
    timing: Option<Rc<TimingStatistics>>,

    /// Shared per-solve diagnostics state. `None` unless the
    /// application requested KKT dumps via the CLI's `--dump` flag.
    /// When set, every successful `solve()` may emit a JSONL record
    /// to `<dump_dir>/iter_NNN/kkt_solve_MMM.jsonl`, gated by the
    /// configured iter-spec for [`DiagCategory::Kkt`].
    diagnostics: Option<Rc<DiagnosticsState>>,
}

impl std::fmt::Debug for StdAugSystemSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StdAugSystemSolver")
            .field("dim", &self.dim)
            .field("nnz", &self.vals.len())
            .field("initialized", &self.initialized)
            .field("last_neg_evals", &self.last_neg_evals)
            .field("last_status", &self.last_status)
            .finish_non_exhaustive()
    }
}

impl StdAugSystemSolver {
    /// Build a solver around a configured [`TSymLinearSolver`].
    pub fn new(linsol: TSymLinearSolver) -> Self {
        Self {
            linsol,
            initialized: false,
            struct_sig: None,
            n_x: 0,
            n_s: 0,
            n_c: 0,
            n_d: 0,
            dim: 0,
            irn: Vec::new(),
            jcn: Vec::new(),
            vals: Vec::new(),
            w_range: 0..0,
            dx_range: 0..0,
            ds_range: 0..0,
            jc_range: 0..0,
            dc_range: 0..0,
            jd_range: 0..0,
            minus_i_range: 0..0,
            dd_range: 0..0,
            last_neg_evals: 0,
            last_status: None,
            have_factor: false,
            timing: None,
            diagnostics: None,
        }
    }

    fn build_structure(&mut self, coeffs: &AugSysCoeffs<'_>) -> ESymSolverStatus {
        let n_x = coeffs.j_c.n_cols();
        let n_c = coeffs.j_c.n_rows();
        let n_d = coeffs.j_d.n_rows();
        debug_assert_eq!(coeffs.j_d.n_cols(), n_x);
        let n_s = n_d;

        let w_nnz = match coeffs.w {
            None => 0_usize,
            Some(w) => w_nonzeros(w),
        };
        let jc_nnz = gen_t_downcast(coeffs.j_c).nonzeros() as usize;
        let jd_nnz = gen_t_downcast(coeffs.j_d).nonzeros() as usize;

        let total = w_nnz
            + (n_x as usize) // dx diagonal
            + (n_s as usize) // ds diagonal
            + jc_nnz
            + (n_c as usize) // dc diagonal (negative)
            + jd_nnz
            + (n_s as usize) // -I block
            + (n_d as usize); // dd diagonal (negative)

        self.irn = Vec::with_capacity(total);
        self.jcn = Vec::with_capacity(total);
        self.vals = vec![0.0; total];

        // ---- (1,1) block: W ----
        let w_start = self.irn.len();
        if let Some(w) = coeffs.w {
            if let Some(t) = w.as_any().downcast_ref::<SymTMatrix>() {
                self.irn.extend_from_slice(t.irows());
                self.jcn.extend_from_slice(t.jcols());
            } else if let Some(dm) = w.as_any().downcast_ref::<DiagMatrix>() {
                // Diagonal W (e.g. the quasi-Newton `B0` substituted by
                // `LowRankAugSystemSolver`): one (i, i) entry per row.
                let n = w_diag_dim(dm);
                for i in 0..n {
                    self.irn.push(i + 1);
                    self.jcn.push(i + 1);
                }
            } else {
                unreachable!("StdAugSystemSolver: W must be a SymTMatrix or DiagMatrix in v1.0")
            }
        }
        self.w_range = w_start..self.irn.len();

        // ---- (1,1) diagonal: D_x + δ_x ----
        let dx_start = self.irn.len();
        for i in 0..n_x {
            self.irn.push(i + 1);
            self.jcn.push(i + 1);
        }
        self.dx_range = dx_start..self.irn.len();

        // ---- (2,2) diagonal: D_s + δ_s ----
        let ds_start = self.irn.len();
        for i in 0..n_s {
            let r = n_x + i + 1;
            self.irn.push(r);
            self.jcn.push(r);
        }
        self.ds_range = ds_start..self.irn.len();

        // ---- (3,1) block: J_c ----
        let jc_start = self.irn.len();
        let j_c = gen_t_downcast(coeffs.j_c);
        let row_off_c = n_x + n_s;
        for (&i, &j) in j_c.irows().iter().zip(j_c.jcols().iter()) {
            // Upstream rows/cols are 1-based already; remap row to the
            // (3,_) compound block.
            self.irn.push(row_off_c + i);
            self.jcn.push(j);
        }
        self.jc_range = jc_start..self.irn.len();

        // ---- (3,3) diagonal: -(D_c + δ_c) ----
        let dc_start = self.irn.len();
        for i in 0..n_c {
            let r = n_x + n_s + i + 1;
            self.irn.push(r);
            self.jcn.push(r);
        }
        self.dc_range = dc_start..self.irn.len();

        // ---- (4,1) block: J_d ----
        let jd_start = self.irn.len();
        let j_d = gen_t_downcast(coeffs.j_d);
        let row_off_d = n_x + n_s + n_c;
        for (&i, &j) in j_d.irows().iter().zip(j_d.jcols().iter()) {
            self.irn.push(row_off_d + i);
            self.jcn.push(j);
        }
        self.jd_range = jd_start..self.irn.len();

        // ---- (4,2) block: -I ----
        let mi_start = self.irn.len();
        for i in 0..n_s {
            self.irn.push(n_x + n_s + n_c + i + 1);
            self.jcn.push(n_x + i + 1);
        }
        self.minus_i_range = mi_start..self.irn.len();

        // ---- (4,4) diagonal: -(D_d + δ_d) ----
        let dd_start = self.irn.len();
        for i in 0..n_d {
            let r = n_x + n_s + n_c + i + 1;
            self.irn.push(r);
            self.jcn.push(r);
        }
        self.dd_range = dd_start..self.irn.len();

        debug_assert_eq!(self.irn.len(), total);
        debug_assert_eq!(self.jcn.len(), total);

        self.n_x = n_x;
        self.n_s = n_s;
        self.n_c = n_c;
        self.n_d = n_d;
        self.dim = n_x + n_s + n_c + n_d;

        let status = self
            .linsol
            .initialize_structure(self.dim, &self.irn, &self.jcn);
        if status == ESymSolverStatus::Success {
            self.initialized = true;
        }
        status
    }

    fn refill_values(&mut self, coeffs: &AugSysCoeffs<'_>) {
        // (1,1) W
        if !self.w_range.is_empty() {
            let Some(w_dyn) = coeffs.w else {
                unreachable!("structure pinned with W; W cannot be None now")
            };
            let dst = &mut self.vals[self.w_range.clone()];
            if let Some(t) = w_dyn.as_any().downcast_ref::<SymTMatrix>() {
                for (d, &v) in dst.iter_mut().zip(t.values().iter()) {
                    *d = coeffs.w_factor * v;
                }
            } else if let Some(dm) = w_dyn.as_any().downcast_ref::<DiagMatrix>() {
                let diag = w_diag_values(dm);
                for (d, &v) in dst.iter_mut().zip(diag.iter()) {
                    *d = coeffs.w_factor * v;
                }
            } else {
                unreachable!("StdAugSystemSolver: W must be a SymTMatrix or DiagMatrix in v1.0")
            }
        }
        // (1,1) diag: D_x + δ_x
        fill_diag(
            &mut self.vals[self.dx_range.clone()],
            coeffs.d_x,
            coeffs.delta_x,
            1.0,
        );
        // (2,2) diag: D_s + δ_s
        fill_diag(
            &mut self.vals[self.ds_range.clone()],
            coeffs.d_s,
            coeffs.delta_s,
            1.0,
        );
        // (3,1) J_c
        {
            let j_c = gen_t_downcast(coeffs.j_c);
            self.vals[self.jc_range.clone()].copy_from_slice(j_c.values());
        }
        // (3,3) diag: -(D_c + δ_c)
        fill_diag(
            &mut self.vals[self.dc_range.clone()],
            coeffs.d_c,
            coeffs.delta_c,
            -1.0,
        );
        // (4,1) J_d
        {
            let j_d = gen_t_downcast(coeffs.j_d);
            self.vals[self.jd_range.clone()].copy_from_slice(j_d.values());
        }
        // (4,2) -I
        for v in &mut self.vals[self.minus_i_range.clone()] {
            *v = -1.0;
        }
        // (4,4) diag: -(D_d + δ_d)
        fill_diag(
            &mut self.vals[self.dd_range.clone()],
            coeffs.d_d,
            coeffs.delta_d,
            -1.0,
        );
    }

    fn pack_rhs(&self, rhs: &AugSysRhs<'_>, packed: &mut [Number]) {
        let n_x = self.n_x as usize;
        let n_s = self.n_s as usize;
        let n_c = self.n_c as usize;
        let n_d = self.n_d as usize;
        copy_vec(rhs.rhs_x, &mut packed[..n_x]);
        copy_vec(rhs.rhs_s, &mut packed[n_x..n_x + n_s]);
        copy_vec(rhs.rhs_c, &mut packed[n_x + n_s..n_x + n_s + n_c]);
        copy_vec(
            rhs.rhs_d,
            &mut packed[n_x + n_s + n_c..n_x + n_s + n_c + n_d],
        );
    }

    fn unpack_sol(&self, packed: &[Number], sol: &mut AugSysSol<'_>) {
        let n_x = self.n_x as usize;
        let n_s = self.n_s as usize;
        let n_c = self.n_c as usize;
        let n_d = self.n_d as usize;
        write_vec(sol.sol_x, &packed[..n_x]);
        write_vec(sol.sol_s, &packed[n_x..n_x + n_s]);
        write_vec(sol.sol_c, &packed[n_x + n_s..n_x + n_s + n_c]);
        write_vec(sol.sol_d, &packed[n_x + n_s + n_c..n_x + n_s + n_c + n_d]);
    }
}

impl AugSystemSolver for StdAugSystemSolver {
    fn provides_inertia(&self) -> bool {
        self.linsol.provides_inertia()
    }

    fn number_of_neg_evals(&self) -> Index {
        self.last_neg_evals
    }

    fn system_dim(&self) -> Index {
        self.dim
    }

    fn kkt_triplets(&self) -> Option<(Index, Vec<Index>, Vec<Index>, Vec<Number>)> {
        if self.irn.is_empty() {
            return None;
        }
        Some((
            self.dim,
            self.irn.clone(),
            self.jcn.clone(),
            self.vals.clone(),
        ))
    }

    fn l_factor(&self, want_values: bool) -> Option<FactorPattern> {
        self.linsol.factor_pattern(want_values)
    }

    fn increase_quality(&mut self) -> bool {
        // Quality bump → pivtol changed → next solve must refactor.
        // `resolve` would silently hand back stale numbers; force the
        // full path by invalidating the cached-factor flag here.
        self.have_factor = false;
        self.linsol.increase_quality()
    }

    fn last_solve_status(&self) -> ESymSolverStatus {
        self.last_status.unwrap_or(ESymSolverStatus::FatalError)
    }

    fn solve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
        check_neg_evals: bool,
        num_neg_evals: Index,
    ) -> ESymSolverStatus {
        // Rebuild the triplet structure (and the backend symbolic factor)
        // whenever the coefficients' structural fingerprint changes, not
        // just on the first call. The limited-memory low-rank solver
        // drives this inner solver with W blocks of different sparsity
        // (`zero_w` for Hessian-free init/multiplier solves vs an
        // `n`-diagonal `B0` for the main solves); reusing a stale pinned
        // structure would drop the W block entirely.
        let sig = {
            let w_nnz = coeffs.w.map(w_nonzeros).unwrap_or(0);
            let jc_nnz = gen_t_downcast(coeffs.j_c).nonzeros() as usize;
            let jd_nnz = gen_t_downcast(coeffs.j_d).nonzeros() as usize;
            (
                w_nnz,
                jc_nnz,
                jd_nnz,
                coeffs.j_c.n_cols(),
                coeffs.j_c.n_rows(),
                coeffs.j_d.n_rows(),
            )
        };
        if !self.initialized || self.struct_sig != Some(sig) {
            let s = self.build_structure(coeffs);
            if s != ESymSolverStatus::Success {
                self.last_status = Some(s);
                return s;
            }
            self.struct_sig = Some(sig);
        }
        self.refill_values(coeffs);

        let mut packed = vec![0.0; self.dim as usize];
        self.pack_rhs(rhs, &mut packed);

        let dump_rhs = packed.clone();

        // Attributes the whole factor+back-solve to
        // `linear_system_factorization` (mirrors upstream
        // `IpStdAugSystemSolver.cpp:155`).
        let _factor_guard = self
            .timing
            .as_deref()
            .map(|t| t.linear_system_factorization.guard());
        let status = self.linsol.multi_solve(
            &self.vals,
            true,
            1,
            &mut packed,
            check_neg_evals,
            num_neg_evals,
        );
        drop(_factor_guard);
        self.last_status = Some(status);
        // Refresh the cached neg-eval count on every outcome where the backend
        // computed an inertia (Success/WrongInertia/Singular), matching IPOPT's
        // `StdAugSystemSolver::NumberOfNegEVals()`, which is a pure pass-through
        // to the linear solver. Refreshing only on Success (as we did before)
        // pins the cache to `num_neg_evals` after the first successful factor,
        // which makes PdFullSpaceSolver's "too-few-negatives → δ_c" routing
        // branch dead code: every WrongInertia then falls through to δ_x, which
        // cannot raise the negative-eigenvalue count. On problems where feral
        // reports too few negatives on the near-singular KKT (e.g. nug12), that
        // sent us thrashing δ_x to its 1e20 ceiling before the δ_c fallback
        // finally engaged. See pounce#99.
        if self.linsol.provides_inertia()
            && matches!(
                status,
                ESymSolverStatus::Success
                    | ESymSolverStatus::WrongInertia
                    | ESymSolverStatus::Singular
            )
        {
            self.last_neg_evals = self.linsol.number_of_neg_evals();
        }
        if status == ESymSolverStatus::Success {
            self.unpack_sol(&packed, sol);
            self.have_factor = true;
        }

        // Diagnostic dump: structured `--dump kkt:...` surface, then
        // the legacy `POUNCE_DUMP_KKT=<path>` env-var fallback. The
        // two paths share `write_kkt_record` so the JSON line is bit-
        // identical regardless of how the dump was requested.
        if let Some(diag) = self.diagnostics.clone() {
            if diag.want(DiagCategory::Kkt) {
                let solve_idx = diag.next_solve_index();
                let filename = format!("kkt_solve_{solve_idx:03}.jsonl");
                // Lift the L pattern off the backend only when the
                // dump variant asks for it AND the factor succeeded —
                // calling `factor_pattern` on a backend that hasn't
                // factored yet returns `None`, but pulling it on
                // every solve when the user only asked for K is pure
                // overhead.
                let variant = diag.config.kkt_variant;
                let factor_pattern =
                    if status == ESymSolverStatus::Success && variant.wants_l_pattern() {
                        self.linsol.factor_pattern(variant.wants_l_values())
                    } else {
                        None
                    };
                if let Some(mut w) = diag.open_writer(&filename) {
                    let _ = write_kkt_record(
                        &mut w,
                        self.dim,
                        &self.irn,
                        &self.jcn,
                        &self.vals,
                        &dump_rhs,
                        &packed,
                        check_neg_evals,
                        num_neg_evals,
                        status,
                        self.last_neg_evals,
                        factor_pattern.as_ref(),
                    );
                }
            }
        }
        if let Ok(path) = std::env::var("POUNCE_DUMP_KKT") {
            use std::sync::atomic::{AtomicBool, Ordering};
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::SeqCst) {
                tracing::warn!(target: "pounce::linsol",
                    "warning: POUNCE_DUMP_KKT is deprecated; prefer `--dump kkt:<iter-spec>` (see pounce --help)"
                );
            }
            dump_kkt(
                &path,
                self.dim,
                &self.irn,
                &self.jcn,
                &self.vals,
                &dump_rhs,
                &packed,
                check_neg_evals,
                num_neg_evals,
                status,
                self.last_neg_evals,
            );
        }

        status
    }

    fn resolve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
    ) -> ESymSolverStatus {
        // Contract: caller has invoked `solve` with byte-identical
        // coefficients since the last `increase_quality`. We trust
        // them and reuse the cached factor. If `have_factor` is false
        // (cold start, or quality was bumped), fall through to a
        // full solve so correctness is preserved even when the call
        // site misjudges the cache state.
        if !self.have_factor {
            return self.solve(coeffs, rhs, sol, false, 0);
        }

        let mut packed = vec![0.0; self.dim as usize];
        self.pack_rhs(rhs, &mut packed);

        // Back-substitution against the cached factor; mirrors upstream
        // `IpStdAugSystemSolver.cpp` `linear_system_back_solve` task.
        let _back_guard = self
            .timing
            .as_deref()
            .map(|t| t.linear_system_back_solve.guard());
        let status = self
            .linsol
            .multi_solve(&self.vals, false, 1, &mut packed, false, 0);
        drop(_back_guard);
        self.last_status = Some(status);
        if status == ESymSolverStatus::Success {
            self.unpack_sol(&packed, sol);
        }
        status
    }

    fn set_diagnostics(&mut self, diag: Rc<DiagnosticsState>) {
        self.diagnostics = Some(diag);
    }

    fn set_timing_stats(&mut self, timing: Rc<TimingStatistics>) {
        self.timing = Some(timing);
    }

    fn try_resolve_many_flat(
        &mut self,
        _coeffs: &AugSysCoeffs<'_>,
        packed_rhs: &mut [Number],
        nrhs: usize,
    ) -> Option<ESymSolverStatus> {
        // Caller must have already populated the cached factor via
        // `solve`. If we're cold (no factor) bail out and let the
        // caller take the per-RHS path — `try_*` semantics, not
        // silent-fallback semantics.
        if !self.have_factor {
            return None;
        }
        if packed_rhs.len() != (self.dim as usize) * nrhs {
            return Some(ESymSolverStatus::FatalError);
        }
        let _back_guard = self
            .timing
            .as_deref()
            .map(|t| t.linear_system_back_solve.guard());
        let status =
            self.linsol
                .multi_solve(&self.vals, false, nrhs as Index, packed_rhs, false, 0);
        drop(_back_guard);
        self.last_status = Some(status);
        Some(status)
    }
}

// ---------------- helpers ----------------

#[allow(clippy::too_many_arguments)]
/// Serialize one KKT solve as a single JSONL record. Shared by the
/// `--dump kkt:...` path (one file per solve under `iter_NNN/`) and
/// the legacy `POUNCE_DUMP_KKT` path (one append-mode file across
/// the whole run).
fn write_kkt_record(
    w: &mut dyn std::io::Write,
    dim: Index,
    irn: &[Index],
    jcn: &[Index],
    vals: &[Number],
    rhs: &[Number],
    sol: &[Number],
    check_neg_evals: bool,
    num_neg_evals: Index,
    status: ESymSolverStatus,
    last_neg_evals: Index,
    factor_pattern: Option<&FactorPattern>,
) -> std::io::Result<()> {
    use std::fmt::Write as _;

    let mut line = String::with_capacity(64 * vals.len());
    line.push('{');
    let _ = write!(line, "\"n\":{dim},");
    let _ = write!(line, "\"check_neg_evals\":{check_neg_evals},");
    let _ = write!(line, "\"num_neg_evals_expected\":{num_neg_evals},");
    let _ = write!(line, "\"num_neg_evals_actual\":{last_neg_evals},");
    let _ = write!(line, "\"status\":\"{status:?}\",");

    line.push_str("\"irn\":[");
    for (i, v) in irn.iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        let _ = write!(line, "{v}");
    }
    line.push_str("],\"jcn\":[");
    for (i, v) in jcn.iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        let _ = write!(line, "{v}");
    }
    line.push_str("],\"vals\":[");
    for (i, v) in vals.iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        let _ = write!(line, "{v:.17e}");
    }
    line.push_str("],\"rhs\":[");
    for (i, v) in rhs.iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        let _ = write!(line, "{v:.17e}");
    }
    line.push_str("],\"sol\":[");
    for (i, v) in sol.iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        let _ = write!(line, "{v:.17e}");
    }
    line.push(']');

    // Optional L pattern + permutation. Pounce#69 schema: emit
    // `L_irn`, `L_jcn`, `perm` whenever a `FactorPattern` is supplied,
    // and emit `L_vals` when the variant included `+Lvals` (the
    // backend populates `l_vals` only in that case).
    if let Some(fp) = factor_pattern {
        line.push_str(",\"L_irn\":[");
        for (i, v) in fp.l_irn.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            let _ = write!(line, "{v}");
        }
        line.push_str("],\"L_jcn\":[");
        for (i, v) in fp.l_jcn.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            let _ = write!(line, "{v}");
        }
        line.push_str("],\"perm\":[");
        for (i, v) in fp.perm.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            let _ = write!(line, "{v}");
        }
        line.push(']');
        if let Some(vals) = fp.l_vals.as_ref() {
            line.push_str(",\"L_vals\":[");
            for (i, v) in vals.iter().enumerate() {
                if i > 0 {
                    line.push(',');
                }
                let _ = write!(line, "{v:.17e}");
            }
            line.push(']');
        }
    }

    line.push_str("}\n");

    w.write_all(line.as_bytes())
}

fn dump_kkt(
    path: &str,
    dim: Index,
    irn: &[Index],
    jcn: &[Index],
    vals: &[Number],
    rhs: &[Number],
    sol: &[Number],
    check_neg_evals: bool,
    num_neg_evals: Index,
    status: ESymSolverStatus,
    last_neg_evals: Index,
) {
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = write_kkt_record(
            &mut f,
            dim,
            irn,
            jcn,
            vals,
            rhs,
            sol,
            check_neg_evals,
            num_neg_evals,
            status,
            last_neg_evals,
            None, // legacy env-var path never carries the L pattern
        );
    }
}

/// Triplet-entry count the (1,1) `W` block contributes, supporting both
/// an explicit [`SymTMatrix`] and a diagonal [`DiagMatrix`] (the latter
/// is what [`crate::kkt::low_rank_aug_system_solver`] substitutes for the
/// limited-memory quasi-Newton `B0`).
fn w_nonzeros(w: &dyn pounce_linalg::SymMatrix) -> usize {
    if let Some(t) = w.as_any().downcast_ref::<SymTMatrix>() {
        t.nonzeros() as usize
    } else if let Some(dm) = w.as_any().downcast_ref::<DiagMatrix>() {
        w_diag_dim(dm) as usize
    } else {
        unreachable!("StdAugSystemSolver: W must be a SymTMatrix or DiagMatrix in v1.0")
    }
}

fn w_diag_dim(dm: &DiagMatrix) -> Index {
    dm.get_diag()
        .expect("DiagMatrix W has no diagonal set")
        .dim()
}

fn w_diag_values(dm: &DiagMatrix) -> Vec<Number> {
    let diag = dm.get_diag().expect("DiagMatrix W has no diagonal set");
    diag.as_any()
        .downcast_ref::<DenseVector>()
        .expect("StdAugSystemSolver: DiagMatrix W diagonal must be DenseVector in v1.0")
        .expanded_values()
}

fn gen_t_downcast(m: &dyn pounce_linalg::Matrix) -> &GenTMatrix {
    let Some(t) = m.as_any().downcast_ref::<GenTMatrix>() else {
        unreachable!("StdAugSystemSolver: J_c / J_d must be GenTMatrix in v1.0")
    };
    t
}

/// Read a vector that is either a [`DenseVector`] or a
/// [`CompoundVector`] of [`DenseVector`]s into a contiguous owned
/// `Vec<Number>`. The resto-side IPM hands us 5-block compound x /
/// D_x; v1.0 originals always arrive as `DenseVector`. Panics on any
/// other layout.
fn flat_read(v: &dyn Vector) -> Vec<Number> {
    if let Some(dv) = v.as_any().downcast_ref::<DenseVector>() {
        return dv.expanded_values();
    }
    if let Some(cv) = v.as_any().downcast_ref::<CompoundVector>() {
        let mut out = Vec::with_capacity(cv.dim() as usize);
        for k in 0..cv.n_comps() {
            let blk = cv.comp(k);
            let dblk = blk
                .as_any()
                .downcast_ref::<DenseVector>()
                .expect("StdAugSystemSolver: CompoundVector blocks must be DenseVectors");
            out.extend_from_slice(&dblk.expanded_values());
        }
        return out;
    }
    unreachable!("StdAugSystemSolver: D_*/rhs/sol must be DenseVector or CompoundVector of DenseVectors in v1.0")
}

/// Inverse of [`flat_read`].
fn flat_write(dst: &mut dyn Vector, src: &[Number]) {
    if let Some(dv) = dst.as_any_mut().downcast_mut::<DenseVector>() {
        dv.set_values(src);
        return;
    }
    if let Some(cv) = dst.as_any_mut().downcast_mut::<CompoundVector>() {
        let mut off = 0usize;
        for k in 0..cv.n_comps() {
            let blk = cv.comp_mut(k);
            let dim = blk.dim() as usize;
            let dblk = blk
                .as_any_mut()
                .downcast_mut::<DenseVector>()
                .expect("StdAugSystemSolver: CompoundVector blocks must be DenseVectors");
            dblk.set_values(&src[off..off + dim]);
            off += dim;
        }
        return;
    }
    unreachable!(
        "StdAugSystemSolver: sol must be DenseVector or CompoundVector of DenseVectors in v1.0"
    )
}

/// Write `sign · (D[i] + delta)` into each slot. `D = None` means
/// the diagonal weight is zero, leaving just `sign · delta`.
fn fill_diag(dst: &mut [Number], d: Option<&dyn Vector>, delta: Number, sign: Number) {
    match d {
        None => {
            for v in dst.iter_mut() {
                *v = sign * delta;
            }
        }
        Some(d) => {
            let xs = flat_read(d);
            debug_assert_eq!(xs.len(), dst.len());
            for (out, &x) in dst.iter_mut().zip(xs.iter()) {
                *out = sign * (x + delta);
            }
        }
    }
}

fn copy_vec(src: &dyn Vector, dst: &mut [Number]) {
    let xs = flat_read(src);
    debug_assert_eq!(xs.len(), dst.len());
    dst.copy_from_slice(&xs);
}

fn write_vec(dst: &mut dyn Vector, src: &[Number]) {
    flat_write(dst, src);
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_common::types::{Index, Number};
    use pounce_linalg::dense_vector::DenseVectorSpace;
    use pounce_linalg::triplet::{GenTMatrixSpace, SymTMatrixSpace};
    use pounce_linsol::sparse_sym_iface::SparseSymLinearSolverInterface;
    use pounce_linsol::EMatrixFormat;

    /// Mock backend: dense LU via tiny Gauss elimination. Used to drive
    /// `StdAugSystemSolver` end-to-end without an MA57 dependency.
    struct DenseMock {
        dim: Index,
        nz: Index,
        a: Vec<Number>,
        last_factor: Vec<Number>, // dense `dim*dim`, lower triangle source
        neg_evals: Index,
    }

    impl DenseMock {
        fn new() -> Self {
            Self {
                dim: 0,
                nz: 0,
                a: Vec::new(),
                last_factor: Vec::new(),
                neg_evals: 0,
            }
        }
    }

    impl SparseSymLinearSolverInterface for DenseMock {
        fn initialize_structure(
            &mut self,
            dim: Index,
            nz: Index,
            _ia: &[Index],
            _ja: &[Index],
        ) -> ESymSolverStatus {
            self.dim = dim;
            self.nz = nz;
            self.a = vec![0.0; nz as usize];
            ESymSolverStatus::Success
        }
        fn values_array_mut(&mut self) -> &mut [Number] {
            &mut self.a
        }
        fn multi_solve(
            &mut self,
            new_matrix: bool,
            ia: &[Index],
            ja: &[Index],
            nrhs: Index,
            rhs_vals: &mut [Number],
            _check: bool,
            _nev: Index,
        ) -> ESymSolverStatus {
            let n = self.dim as usize;
            if new_matrix {
                // Densify the symmetric triplet into row-major full
                // matrix for LU.
                let mut dense = vec![0.0; n * n];
                for k in 0..self.nz as usize {
                    let i = (ia[k] - 1) as usize;
                    let j = (ja[k] - 1) as usize;
                    dense[i * n + j] += self.a[k];
                    if i != j {
                        dense[j * n + i] += self.a[k];
                    }
                }
                self.last_factor = dense;
            }
            // Gauss-eliminate (no pivoting) per column for each rhs.
            for col in 0..nrhs as usize {
                let mut a = self.last_factor.clone();
                let b = &mut rhs_vals[col * n..col * n + n];
                let mut neg = 0_i32;
                for k in 0..n {
                    // Find pivot row by max-abs in col k below k.
                    let mut piv = k;
                    let mut piv_abs = a[k * n + k].abs();
                    for r in (k + 1)..n {
                        let av = a[r * n + k].abs();
                        if av > piv_abs {
                            piv_abs = av;
                            piv = r;
                        }
                    }
                    if piv != k {
                        for c in 0..n {
                            a.swap(k * n + c, piv * n + c);
                        }
                        b.swap(k, piv);
                    }
                    let p = a[k * n + k];
                    if p.abs() < 1e-14 {
                        return ESymSolverStatus::Singular;
                    }
                    if p < 0.0 {
                        neg += 1;
                    }
                    for r in (k + 1)..n {
                        let f = a[r * n + k] / p;
                        for c in k..n {
                            a[r * n + c] -= f * a[k * n + c];
                        }
                        b[r] -= f * b[k];
                    }
                }
                // Back-substitute.
                for k in (0..n).rev() {
                    let mut s = b[k];
                    for c in (k + 1)..n {
                        s -= a[k * n + c] * b[c];
                    }
                    b[k] = s / a[k * n + k];
                }
                self.neg_evals = neg;
            }
            ESymSolverStatus::Success
        }
        fn number_of_neg_evals(&self) -> Index {
            self.neg_evals
        }
        fn increase_quality(&mut self) -> bool {
            false
        }
        fn provides_inertia(&self) -> bool {
            true
        }
        fn matrix_format(&self) -> EMatrixFormat {
            EMatrixFormat::TripletFormat
        }
    }

    /// Hand-built tiny KKT system (n_x=2, n_s=1, n_c=1, n_d=1):
    ///
    /// ```text
    ///   W = diag(2, 3)        D_x = (0, 0)   δ_x = 0
    ///   D_s = (1)             δ_s = 0
    ///   J_c = [1  1]          D_c = (0)      δ_c = 0
    ///   J_d = [1  0]          D_d = (0)      δ_d = 0
    /// ```
    ///
    /// Pick rhs so that the solution is `(dx, ds, dyc, dyd) = (1, 1, 1,
    /// 1, 1)` — five unknowns. Derive rhs from `K · sol`.
    #[test]
    fn solves_5x5_kkt_through_dense_mock() {
        // ---- W ----
        let w_space = SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]);
        let mut w = SymTMatrix::new(w_space);
        w.set_values(&[2.0, 3.0]);

        // ---- J_c (1×2 dense in triplet) ----
        let jc_space = GenTMatrixSpace::new(1, 2, vec![1, 1], vec![1, 2]);
        let mut j_c = GenTMatrix::new(jc_space);
        j_c.set_values(&[1.0, 1.0]);

        // ---- J_d (1×2) ----
        let jd_space = GenTMatrixSpace::new(1, 2, vec![1], vec![1]);
        let mut j_d = GenTMatrix::new(jd_space);
        j_d.set_values(&[1.0]);

        // ---- D_s = 1 (homogeneous) ----
        let s_space = DenseVectorSpace::new(1);
        let mut d_s = s_space.make_new_dense();
        d_s.set_values(&[1.0]);

        // RHS slots — match Ipopt convention: (rhs_x, rhs_s, rhs_c, rhs_d).
        // Compute K · (1,1,1,1,1):
        //   row x1: 2·1 + 0 + 1·1 + 1·1 = 4
        //   row x2: 3·1 + 0 + 1·1 + 0·1 = 4
        //   row s:  1·1 + 0·1·yd + (-1)·1 = 0     (D_s + δ_s) - 1
        //   row c:  1·1 + 1·1     = 2
        //   row d:  1·1 - 1·1     = 0
        let xs = DenseVectorSpace::new(2);
        let mut rx = xs.make_new_dense();
        rx.set_values(&[4.0, 4.0]);
        let mut rs = s_space.make_new_dense();
        rs.set_values(&[0.0]);
        let cs = DenseVectorSpace::new(1);
        let mut rc = cs.make_new_dense();
        rc.set_values(&[2.0]);
        let ds_space = DenseVectorSpace::new(1);
        let mut rd = ds_space.make_new_dense();
        rd.set_values(&[0.0]);

        let mut sx = xs.make_new_dense();
        let mut ss = s_space.make_new_dense();
        let mut sc = cs.make_new_dense();
        let mut sd = ds_space.make_new_dense();

        let linsol = TSymLinearSolver::new(Box::new(DenseMock::new()), None, false);
        let mut solver = StdAugSystemSolver::new(linsol);

        let coeffs = AugSysCoeffs {
            w: Some(&w),
            w_factor: 1.0,
            d_x: None,
            delta_x: 0.0,
            d_s: Some(&d_s),
            delta_s: 0.0,
            j_c: &j_c,
            d_c: None,
            delta_c: 0.0,
            j_d: &j_d,
            d_d: None,
            delta_d: 0.0,
        };
        let rhs = AugSysRhs {
            rhs_x: &rx,
            rhs_s: &rs,
            rhs_c: &rc,
            rhs_d: &rd,
        };
        let mut sol = AugSysSol {
            sol_x: &mut sx,
            sol_s: &mut ss,
            sol_c: &mut sc,
            sol_d: &mut sd,
        };
        let status = solver.solve(&coeffs, &rhs, &mut sol, false, 0);
        assert_eq!(status, ESymSolverStatus::Success);

        for v in sx.values() {
            assert!((v - 1.0).abs() < 1e-10, "sol_x = {v}");
        }
        for v in ss.values() {
            assert!((v - 1.0).abs() < 1e-10, "sol_s = {v}");
        }
        for v in sc.values() {
            assert!((v - 1.0).abs() < 1e-10, "sol_c = {v}");
        }
        for v in sd.values() {
            assert!((v - 1.0).abs() < 1e-10, "sol_d = {v}");
        }
    }

    /// End-to-end equivalence: solving the augmented system with an
    /// explicit dense `W = σ I + v vᵀ − u uᵀ` (a `SymTMatrix`) through
    /// `StdAugSystemSolver` must produce the *same* solution as solving
    /// it with the matching `LowRankUpdateSymMatrix` through
    /// `LowRankAugSystemSolver` wrapping `StdAugSystemSolver`. This is
    /// the integration the limited-memory path relies on: it exercises
    /// the constrained SMW path with both a positive (V) and a negative
    /// (U) curvature column, and the `DiagMatrix`-W branch of
    /// `StdAugSystemSolver`. `DenseMock` gives an exact LU oracle.
    #[test]
    fn lowrank_smw_matches_dense_w_on_constrained_system() {
        use crate::kkt::low_rank_aug_system_solver::LowRankAugSystemSolver;
        use pounce_linalg::diag_matrix::DiagMatrix;
        use pounce_linalg::low_rank_update_sym_matrix::LowRankUpdateSymMatrixSpace;
        use pounce_linalg::multi_vector_matrix::MultiVectorMatrixSpace;

        let n = 4usize;
        let sigma = 2.0;
        // Three V columns and three U columns in R⁴ — six vectors in
        // 4-space, the exact L-BFGS situation (2·history columns) that
        // the single-column-only mock tests never exercised. Magnitudes
        // kept modest so B = σI + Σvvᵀ − Σuuᵀ stays SPD.
        let vcols = [
            vec![0.6, 0.1, -0.2, 0.3],
            vec![0.2, 0.5, 0.1, -0.1],
            vec![-0.1, 0.2, 0.4, 0.2],
            vec![0.3, -0.2, 0.1, 0.4],
            vec![0.15, 0.25, -0.3, 0.1],
            vec![-0.2, 0.1, 0.2, 0.35],
        ];
        let ucols = [
            vec![0.3, -0.1, 0.2, 0.1],
            vec![0.1, 0.3, -0.2, 0.2],
            vec![0.2, 0.1, 0.1, -0.3],
            vec![-0.1, 0.2, 0.15, 0.1],
            vec![0.25, -0.15, 0.1, 0.2],
            vec![0.1, 0.2, -0.25, 0.15],
        ];
        // Dense W = σI + Σ vᵢvᵢᵀ − Σ uᵢuᵢᵀ (full lower triangle triplet).
        let mut wfull = vec![0.0_f64; n * n];
        for i in 0..n {
            wfull[i * n + i] = sigma;
        }
        for c in vcols.iter() {
            for i in 0..n {
                for j in 0..n {
                    wfull[i * n + j] += c[i] * c[j];
                }
            }
        }
        for c in ucols.iter() {
            for i in 0..n {
                for j in 0..n {
                    wfull[i * n + j] -= c[i] * c[j];
                }
            }
        }

        // J_c = [1 1 1 1]; no inequalities (n_d = n_s = 0).
        let make_jc = || {
            let sp = GenTMatrixSpace::new(1, 4, vec![1, 1, 1, 1], vec![1, 2, 3, 4]);
            let mut m = GenTMatrix::new(sp);
            m.set_values(&[1.0, 1.0, 1.0, 1.0]);
            m
        };
        // One inequality row → n_d = n_s = 1 (a slack block), matching
        // HS071's structure. The mock + single-column tests never had a
        // slack block; the SMW s/d-block path went uncovered.
        let make_jd = || {
            let sp = GenTMatrixSpace::new(1, 4, vec![1, 1], vec![1, 3]);
            let mut m = GenTMatrix::new(sp);
            m.set_values(&[1.0, 1.0]);
            m
        };

        let xs = DenseVectorSpace::new(4);
        let cs = DenseVectorSpace::new(1);
        let mk = |sp: &Rc<DenseVectorSpace>, vals: &[Number]| {
            let mut d = sp.make_new_dense();
            d.set_values(vals);
            d
        };

        let solve_with = |w: &dyn pounce_linalg::SymMatrix,
                          aug: &mut dyn AugSystemSolver|
         -> (Vec<Number>, Vec<Number>) {
            let j_c = make_jc();
            let j_d = make_jd();
            let rx = mk(&xs, &[1.0, 2.0, -1.0, 0.5]);
            let rs = mk(&cs, &[0.4]);
            let rc = mk(&cs, &[3.0]);
            let rd = mk(&cs, &[0.7]);
            let mut sx = mk(&xs, &[0.0, 0.0, 0.0, 0.0]);
            let mut ss = mk(&cs, &[0.0]);
            let mut sc = mk(&cs, &[0.0]);
            let mut sd = mk(&cs, &[0.0]);
            let d_s = mk(&cs, &[1.5]);
            let coeffs = AugSysCoeffs {
                w: Some(w),
                w_factor: 1.0,
                d_x: None,
                delta_x: 0.0,
                d_s: Some(&d_s),
                delta_s: 0.0,
                j_c: &j_c,
                d_c: None,
                delta_c: 0.0,
                j_d: &j_d,
                d_d: None,
                delta_d: 0.0,
            };
            let rhs = AugSysRhs {
                rhs_x: &rx,
                rhs_s: &rs,
                rhs_c: &rc,
                rhs_d: &rd,
            };
            let mut sol = AugSysSol {
                sol_x: &mut sx,
                sol_s: &mut ss,
                sol_c: &mut sc,
                sol_d: &mut sd,
            };
            let status = aug.solve(&coeffs, &rhs, &mut sol, false, 1);
            assert_eq!(status, ESymSolverStatus::Success);
            (sx.expanded_values(), sc.expanded_values())
        };

        // Dense reference: full lower-triangle triplet of `wfull`.
        let mut wi = Vec::new();
        let mut wj = Vec::new();
        let mut wv = Vec::new();
        for i in 0..n {
            for j in 0..=i {
                wi.push(i as Index + 1);
                wj.push(j as Index + 1);
                wv.push(wfull[i * n + j]);
            }
        }
        let w_space = SymTMatrixSpace::new(4, wi, wj);
        let mut w_dense = SymTMatrix::new(w_space);
        w_dense.set_values(&wv);
        let mut std_solver = StdAugSystemSolver::new(TSymLinearSolver::new(
            Box::new(pounce_feral::FeralSolverInterface::new()),
            None,
            false,
        ));
        let (ref_x, ref_c) = solve_with(&w_dense, &mut std_solver);

        // Low-rank SMW path: same B as a LowRankUpdateSymMatrix.
        let lr_space = LowRankUpdateSymMatrixSpace::new(4, None, false);
        let mut lr = lr_space.make_new_low_rank();
        let mut diag = xs.make_new_dense();
        diag.set_values(&[sigma; 4]);
        lr.set_diag(Rc::new(diag) as Rc<dyn Vector>);
        let build_mvm = |cols: &[Vec<Number>]| {
            let sp = MultiVectorMatrixSpace::new(cols.len() as Index, Rc::clone(&xs));
            let mut mvm = sp.make_new_multi_vector();
            for (k, c) in cols.iter().enumerate() {
                let mut cv = xs.make_new_dense();
                cv.set_values(c);
                mvm.set_vector(k as Index, Rc::new(cv) as Rc<dyn Vector>);
            }
            mvm
        };
        lr.set_v(Rc::new(build_mvm(&vcols)));
        lr.set_u(Rc::new(build_mvm(&ucols)));
        let _ = DiagMatrix::new(4); // ensure DiagMatrix path is linked

        let mut lr_solver =
            LowRankAugSystemSolver::new(Box::new(StdAugSystemSolver::new(TSymLinearSolver::new(
                Box::new(pounce_feral::FeralSolverInterface::new()),
                None,
                false,
            ))));
        let (lr_x, lr_c) = solve_with(&lr, &mut lr_solver);

        for (a, b) in ref_x.iter().zip(lr_x.iter()) {
            assert!((a - b).abs() < 1e-9, "sol_x mismatch: dense={a} smw={b}");
        }
        for (a, b) in ref_c.iter().zip(lr_c.iter()) {
            assert!((a - b).abs() < 1e-9, "sol_c mismatch: dense={a} smw={b}");
        }
    }
}
