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
            Some(w) => sym_t_downcast(w).nonzeros() as usize,
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
            let w = sym_t_downcast(w);
            self.irn.extend_from_slice(w.irows());
            self.jcn.extend_from_slice(w.jcols());
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
            let w = sym_t_downcast(w_dyn);
            let dst = &mut self.vals[self.w_range.clone()];
            for (d, &v) in dst.iter_mut().zip(w.values().iter()) {
                *d = coeffs.w_factor * v;
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
        if !self.initialized {
            let s = self.build_structure(coeffs);
            if s != ESymSolverStatus::Success {
                self.last_status = Some(s);
                return s;
            }
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
        if status == ESymSolverStatus::Success {
            if self.linsol.provides_inertia() {
                self.last_neg_evals = self.linsol.number_of_neg_evals();
            }
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
                eprintln!(
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

fn sym_t_downcast(m: &dyn pounce_linalg::SymMatrix) -> &SymTMatrix {
    let Some(t) = m.as_any().downcast_ref::<SymTMatrix>() else {
        unreachable!("StdAugSystemSolver: W must be a SymTMatrix in v1.0")
    };
    t
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
}
