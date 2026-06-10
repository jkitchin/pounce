//! MA57 backend — port of `IpMa57TSolverInterface.{hpp,cpp}`.
//!
//! Implements [`SparseSymLinearSolverInterface`] over the Fortran
//! MA57 entry points in [`crate::ffi`]. Memory layout, ICNTL setup,
//! quality escalation, and the `info[0] = -3 / -4` retry loop all
//! match upstream byte-for-byte (cross-reference the line numbers in
//! the comments below against the source in
//! `ref/Ipopt/.../IpMa57TSolverInterface.cpp`).

use crate::ffi::{ma57ad_, ma57bd_, ma57cd_, ma57ed_, ma57id_, openblas_set_num_threads};
use pounce_common::options_list::OptionsList;
use pounce_common::types::{Index, Number};
use pounce_linsol::{EMatrixFormat, ESymSolverStatus, SparseSymLinearSolverInterface};
use std::sync::Once;

/// First-touch OpenBLAS thread-count setup. MA57's internal dgemm
/// calls on small supernodes are dominated by thread spin-up on
/// many-core machines (M1/M2 with 8+ cores): on `elec_400` (1200×1200
/// fully dense Hessian) the default thread count makes each
/// factorization ~5× slower wall-clock and ~30× slower CPU than
/// single-threaded. Honour `OPENBLAS_NUM_THREADS` if the user set it;
/// otherwise force single-threaded BLAS on first MA57 construction.
fn configure_openblas_threads_once() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("OPENBLAS_NUM_THREADS").is_some() {
            return;
        }
        // SAFETY: `openblas_set_num_threads` is a thread-safe setter
        // exported by libopenblas (loaded transitively via libcoinhsl).
        unsafe { openblas_set_num_threads(1) };
    });
}

/// Settings drawn from `OptionsList` at `InitializeImpl` time
/// (`IpMa57TSolverInterface.cpp:287-421`).
#[derive(Debug, Clone, Copy)]
pub struct Options {
    print_level: Index,
    pivtol: Number,
    pivtolmax: Number,
    pre_alloc: Number,
    pivot_order: Index,
    automatic_scaling: bool,
    block_size: Index,
    node_amalgamation: Index,
    small_pivot_flag: Index,
}

impl Options {
    /// Read MA57 options from `opts`, applying `prefix` (e.g.
    /// `"resto."`). Falls back to upstream defaults when an option is
    /// absent.
    pub fn from_options_list(opts: &OptionsList, prefix: &str) -> Self {
        let print_level = opts
            .get_integer_value("ma57_print_level", prefix)
            .ok()
            .map(|(v, _)| v)
            .unwrap_or(0);
        let pivtol = opts
            .get_numeric_value("ma57_pivtol", prefix)
            .ok()
            .map(|(v, _)| v)
            .unwrap_or(1e-8);
        let pivtolmax_default = pivtol.max(1e-4);
        let pivtolmax = opts
            .get_numeric_value("ma57_pivtolmax", prefix)
            .ok()
            .map(|(v, _)| v)
            .unwrap_or(pivtolmax_default)
            .max(pivtol);
        let pre_alloc = opts
            .get_numeric_value("ma57_pre_alloc", prefix)
            .ok()
            .map(|(v, _)| v)
            .unwrap_or(1.05);
        let pivot_order = opts
            .get_integer_value("ma57_pivot_order", prefix)
            .ok()
            .map(|(v, _)| v)
            .unwrap_or(5);
        let automatic_scaling = opts
            .get_bool_value("ma57_automatic_scaling", prefix)
            .ok()
            .map(|(v, _)| v)
            .unwrap_or(false);
        let block_size = opts
            .get_integer_value("ma57_block_size", prefix)
            .ok()
            .map(|(v, _)| v)
            .unwrap_or(16);
        let node_amalgamation = opts
            .get_integer_value("ma57_node_amalgamation", prefix)
            .ok()
            .map(|(v, _)| v)
            .unwrap_or(16);
        let small_pivot_flag = opts
            .get_integer_value("ma57_small_pivot_flag", prefix)
            .ok()
            .map(|(v, _)| v)
            .unwrap_or(0);
        Self {
            print_level,
            pivtol,
            pivtolmax,
            pre_alloc,
            pivot_order,
            automatic_scaling,
            block_size,
            node_amalgamation,
            small_pivot_flag,
        }
    }

    /// Upstream defaults — used when a caller wants an MA57 backend
    /// without going through `OptionsList` (Phase-4 standalone tests).
    pub fn defaults() -> Self {
        Self {
            print_level: 0,
            pivtol: 1e-8,
            pivtolmax: 1e-4,
            pre_alloc: 1.05,
            pivot_order: 5,
            automatic_scaling: false,
            block_size: 16,
            node_amalgamation: 16,
            small_pivot_flag: 0,
        }
    }
}

/// MA57 solver wrapping `libcoinhsl`'s Fortran entry points.
///
/// State machine (see [`SparseSymLinearSolverInterface`] doc):
/// `new` → `initialize_structure` → `values_array_mut` → `multi_solve`
/// → (optionally `increase_quality` → `multi_solve` again).
pub struct Ma57SolverInterface {
    options: Options,
    /// Most-recent factorization's negative-eigenvalue count
    /// (`info[24-1]`).
    negevals: Index,
    /// Set when `increase_quality` was called since the last factor;
    /// triggers `CallAgain` if the next `multi_solve` arrives with
    /// `new_matrix=false`.
    pivtol_changed: bool,
    /// Whether the next `multi_solve` must refactor regardless of
    /// `new_matrix` (set after `CallAgain` was returned).
    refactorize: bool,
    /// Set after a successful `initialize_structure`.
    initialized: bool,

    dim: Index,
    nonzeros: Index,
    /// Numerical values of the matrix nonzeros, in the same triplet
    /// order as `(ia, ja)` from `initialize_structure`.
    a: Vec<Number>,

    /// MA57 ICNTL/CNTL/INFO/RINFO scratch arrays.
    icntl: [Index; 20],
    cntl: [Number; 5],
    info: [Index; 40],
    rinfo: [Number; 20],

    /// Symbolic-factor workspace (see `IpMa57TSolverInterface.hpp:275`).
    lkeep: Index,
    keep: Vec<Index>,
    iwork: Vec<Index>,

    /// Numerical-factor real / integer storage. Grow via MA57E in the
    /// `info[0] = -3 / -4` branches.
    lfact: Index,
    fact: Vec<Number>,
    lifact: Index,
    ifact: Vec<Index>,

    /// Reusable MA57C real workspace (length `n*nrhs`). Kept across
    /// backsolves so the factor-once/solve-many hot path does not
    /// allocate per solve (L11). MA57C uses it as pure scratch — upstream
    /// passes an *uninitialized* `new Number[lwork]` — so it needs no
    /// zeroing on reuse.
    work: Vec<Number>,
}

impl std::fmt::Debug for Ma57SolverInterface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ma57SolverInterface")
            .field("dim", &self.dim)
            .field("nonzeros", &self.nonzeros)
            .field("initialized", &self.initialized)
            .field("negevals", &self.negevals)
            .field("pivtol", &self.options.pivtol)
            .field("pivtolmax", &self.options.pivtolmax)
            .finish_non_exhaustive()
    }
}

impl Ma57SolverInterface {
    /// New backend with `opts` already read from `OptionsList`.
    pub fn with_options(opts: Options) -> Self {
        configure_openblas_threads_once();
        let mut me = Self {
            options: opts,
            negevals: 0,
            pivtol_changed: false,
            refactorize: false,
            initialized: false,
            dim: 0,
            nonzeros: 0,
            a: Vec::new(),
            icntl: [0; 20],
            cntl: [0.0; 5],
            info: [0; 40],
            rinfo: [0.0; 20],
            lkeep: 0,
            keep: Vec::new(),
            iwork: Vec::new(),
            lfact: 0,
            fact: Vec::new(),
            lifact: 0,
            ifact: Vec::new(),
            work: Vec::new(),
        };
        me.apply_icntl();
        me
    }

    /// Default-options factory — primarily for unit tests that
    /// construct an MA57 backend without an `OptionsList`.
    pub fn new() -> Self {
        Self::with_options(Options::defaults())
    }

    /// Build from an `OptionsList` plus prefix, mirroring upstream's
    /// `Ma57TSolverInterface::InitializeImpl(options, prefix)`.
    pub fn from_options_list(opts: &OptionsList, prefix: &str) -> Self {
        Self::with_options(Options::from_options_list(opts, prefix))
    }

    /// Maximum pivot tolerance; useful in tests.
    pub fn pivtol(&self) -> Number {
        self.options.pivtol
    }

    /// Initialise `icntl` / `cntl` from MA57ID and overlay the Ipopt
    /// custom settings (cpp:365-394).
    fn apply_icntl(&mut self) {
        unsafe {
            ma57id_(self.cntl.as_mut_ptr(), self.icntl.as_mut_ptr());
        }
        self.icntl[0] = 0; // error stream
        self.icntl[1] = 0; // warning stream
        self.icntl[3] = 1; // print statistics (unused)
        self.icntl[4] = self.options.print_level;
        self.icntl[5] = self.options.pivot_order;
        self.icntl[6] = 1; // pivoting strategy
        self.icntl[10] = self.options.block_size;
        self.icntl[11] = self.options.node_amalgamation;
        self.icntl[14] = if self.options.automatic_scaling { 1 } else { 0 };
        self.icntl[15] = self.options.small_pivot_flag;
        self.cntl[0] = self.options.pivtol;
    }

    /// Run MA57AD (symbolic phase). Allocates `keep` / `iwork` /
    /// `fact` / `ifact` based on the suggested sizes returned in
    /// `info[8]` and `info[9]`.
    fn symbolic_factorization(&mut self, irn: &[Index], jcn: &[Index]) -> ESymSolverStatus {
        let n = self.dim;
        let ne = self.nonzeros;

        // lkeep >= 5*N + NE + max(N, NE) + 42  (upstream cpp:536).
        // MA57's Fortran interface is 32-bit, so every workspace length must
        // fit in `Index` (i32). Evaluated in i32 this sum overflows near
        // ne ~ 3e8 — wrapping to a negative length that `as usize` turns into
        // an absurd allocation (release) or a panic (debug). Size in i64 and
        // fail cleanly if the problem exceeds MA57's index range.
        let Some((lkeep, liwork)) = ma57_symbolic_sizes(n, ne) else {
            return ESymSolverStatus::FatalError;
        };
        self.lkeep = lkeep;

        self.cntl[0] = self.options.pivtol;
        self.iwork = vec![0; liwork as usize];
        self.keep = vec![0; self.lkeep as usize];

        // SAFETY: pointer/length contract documented in the FFI
        // module; arrays sized to MA57's stated minimums above.
        unsafe {
            ma57ad_(
                &n,
                &ne,
                irn.as_ptr(),
                jcn.as_ptr(),
                &mut self.lkeep,
                self.keep.as_mut_ptr(),
                self.iwork.as_mut_ptr(),
                self.icntl.as_ptr(),
                self.info.as_mut_ptr(),
                self.rinfo.as_mut_ptr(),
            );
        }

        if self.info[0] < 0 {
            return ESymSolverStatus::FatalError;
        }

        // Suggested workspace sizes (cpp:583-584). We grow by
        // `pre_alloc` to avoid the `info[0] = -3 / -4` retries on
        // typical problems.
        let scale = self.options.pre_alloc;
        // `info[8]`/`info[9]` are MA57's suggested workspace sizes; growing by
        // `scale` and rounding up can exceed i32. The float->int cast saturates
        // (so it no longer wraps), but an i32::MAX-element allocation is still
        // absurd — treat an out-of-range suggestion as too large for MA57.
        let (Some(lfact), Some(lifact)) = (
            ma57_scaled_size(self.info[8], scale),
            ma57_scaled_size(self.info[9], scale),
        ) else {
            return ESymSolverStatus::FatalError;
        };
        self.lfact = lfact;
        self.lifact = lifact;

        self.fact = vec![0.0; self.lfact as usize];
        self.ifact = vec![0; self.lifact as usize];

        ESymSolverStatus::Success
    }

    /// Numerical factor (MA57BD) with the `info[0] = -3/-4` grow loop
    /// (cpp:614-727).
    fn factorization(
        &mut self,
        check_neg_evals: bool,
        number_of_neg_evals: Index,
    ) -> ESymSolverStatus {
        self.cntl[0] = self.options.pivtol;
        let n = self.dim;
        let ne = self.nonzeros;

        loop {
            // SAFETY: all arrays sized via `symbolic_factorization`
            // and `Self::values_array_mut`; pointers valid for
            // `n`/`ne`/`lfact`/`lifact`/`lkeep` as required.
            unsafe {
                ma57bd_(
                    &n,
                    &ne,
                    self.a.as_ptr(),
                    self.fact.as_mut_ptr(),
                    &self.lfact,
                    self.ifact.as_mut_ptr(),
                    &self.lifact,
                    &self.lkeep,
                    self.keep.as_mut_ptr(),
                    self.iwork.as_mut_ptr(),
                    self.icntl.as_ptr(),
                    self.cntl.as_ptr(),
                    self.info.as_mut_ptr(),
                    self.rinfo.as_mut_ptr(),
                );
            }
            self.negevals = self.info[24 - 1];

            match self.info[0] {
                0 => break,
                -3 => self.grow_fact(),
                -4 => self.grow_ifact(),
                4 => return ESymSolverStatus::Singular,
                v if v < 0 => return ESymSolverStatus::FatalError,
                // info[0] > 0 is a warning; upstream treats it as
                // fatal (cpp:718-725).
                _ => return ESymSolverStatus::FatalError,
            }
        }

        if check_neg_evals && number_of_neg_evals != self.negevals {
            return ESymSolverStatus::WrongInertia;
        }

        ESymSolverStatus::Success
    }

    /// Grow `fact` via MA57E with `ic = 0` (cpp:644-673).
    fn grow_fact(&mut self) {
        // info[16] is MA57's suggested new lfact.
        let suggested = (self.info[16] as Number * self.options.pre_alloc).ceil() as Index;
        let new_lfact = suggested.max(self.info[16]).max(self.lfact + 1);
        let mut newfac: Vec<Number> = vec![0.0; new_lfact as usize];
        let n = self.dim;
        let ic: Index = 0;
        // info[1] is MA57's reported `lfact` value at failure point;
        // we pass it as `lfact` arg per upstream's call signature.
        let lfact_in = self.info[1];
        let mut idmy: Index = 0;

        // SAFETY: newfac allocated to `new_lfact >= lfact`; integer
        // dummy `idmy` matches upstream's pattern at cpp:669-671.
        unsafe {
            ma57ed_(
                &n,
                &ic,
                self.keep.as_mut_ptr(),
                self.fact.as_mut_ptr(),
                &lfact_in,
                newfac.as_mut_ptr(),
                &new_lfact,
                self.ifact.as_mut_ptr(),
                &lfact_in,
                &mut idmy,
                &new_lfact,
                self.info.as_mut_ptr(),
            );
        }
        self.fact = newfac;
        self.lfact = new_lfact;
    }

    /// Grow `ifact` via MA57E with `ic = 1` (cpp:675-697).
    fn grow_ifact(&mut self) {
        let suggested = (self.info[17] as Number * self.options.pre_alloc).ceil() as Index;
        let new_lifact = suggested.max(self.info[17]).max(self.lifact + 1);
        let mut newifc: Vec<Index> = vec![0; new_lifact as usize];
        let n = self.dim;
        let ic: Index = 1;
        let lifact_in = self.info[1];
        let mut ddmy: Number = 0.0;

        // SAFETY: newifc allocated to `new_lifact >= lifact`; real
        // dummy `ddmy` matches upstream cpp:693-695.
        unsafe {
            ma57ed_(
                &n,
                &ic,
                self.keep.as_mut_ptr(),
                self.fact.as_mut_ptr(),
                &lifact_in,
                &mut ddmy,
                &new_lifact,
                self.ifact.as_mut_ptr(),
                &lifact_in,
                newifc.as_mut_ptr(),
                &new_lifact,
                self.info.as_mut_ptr(),
            );
        }
        self.ifact = newifc;
        self.lifact = new_lifact;
    }

    /// Apply the factorization to `nrhs` right-hand sides packed in
    /// `rhs_vals` (column-major, leading dim `dim`). Solutions
    /// overwrite `rhs_vals`.
    fn backsolve(&mut self, nrhs: Index, rhs_vals: &mut [Number]) -> ESymSolverStatus {
        let n = self.dim;
        let job: Index = 1;
        let lrhs = n;
        // MA57C needs `n*nrhs` reals of workspace; in i32 this overflows for
        // large n*nrhs. Widen to i64 and fail cleanly rather than wrap into a
        // garbage length.
        let lwork_wide = n as i64 * nrhs as i64;
        if lwork_wide > Index::MAX as i64 {
            return ESymSolverStatus::FatalError;
        }
        let lwork = lwork_wide as Index;
        // Reuse the cached workspace (L11): resize to `n*nrhs` (a no-op once
        // it is large enough, so no per-solve allocation in the solve-many
        // hot path). MA57C treats it as scratch, so stale contents are fine.
        self.work.resize(lwork as usize, 0.0);

        // SAFETY: rhs_vals length `n*nrhs` is the caller's contract;
        // `work` sized to `n*nrhs` per MA57C requirement.
        unsafe {
            ma57cd_(
                &job,
                &n,
                self.fact.as_ptr(),
                &self.lfact,
                self.ifact.as_ptr(),
                &self.lifact,
                &nrhs,
                rhs_vals.as_mut_ptr(),
                &lrhs,
                self.work.as_mut_ptr(),
                &lwork,
                self.iwork.as_mut_ptr(),
                self.icntl.as_ptr(),
                self.info.as_mut_ptr(),
            );
        }

        if self.info[0] != 0 {
            return ESymSolverStatus::FatalError;
        }
        ESymSolverStatus::Success
    }
}

impl Default for Ma57SolverInterface {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseSymLinearSolverInterface for Ma57SolverInterface {
    fn initialize_structure(
        &mut self,
        dim: Index,
        nonzeros: Index,
        ia: &[Index],
        ja: &[Index],
    ) -> ESymSolverStatus {
        assert_eq!(ia.len(), nonzeros as usize);
        assert_eq!(ja.len(), nonzeros as usize);
        self.dim = dim;
        self.nonzeros = nonzeros;
        self.a = vec![0.0; nonzeros as usize];

        let status = self.symbolic_factorization(ia, ja);
        if status == ESymSolverStatus::Success {
            self.initialized = true;
        }
        status
    }

    fn values_array_mut(&mut self) -> &mut [Number] {
        debug_assert!(self.initialized);
        &mut self.a
    }

    fn multi_solve(
        &mut self,
        new_matrix: bool,
        _ia: &[Index],
        _ja: &[Index],
        nrhs: Index,
        rhs_vals: &mut [Number],
        check_neg_evals: bool,
        number_of_neg_evals: Index,
    ) -> ESymSolverStatus {
        // Pivot tolerance changed since the last factor: caller has
        // to refill the values and we re-factor (cpp:439-451).
        if self.pivtol_changed {
            self.pivtol_changed = false;
            if !new_matrix {
                self.refactorize = true;
                return ESymSolverStatus::CallAgain;
            }
        }

        if new_matrix || self.refactorize {
            let status = self.factorization(check_neg_evals, number_of_neg_evals);
            if status != ESymSolverStatus::Success {
                return status;
            }
            self.refactorize = false;
        }

        self.backsolve(nrhs, rhs_vals)
    }

    fn number_of_neg_evals(&self) -> Index {
        debug_assert!(self.initialized);
        self.negevals
    }

    fn increase_quality(&mut self) -> bool {
        if self.options.pivtol == self.options.pivtolmax {
            return false;
        }
        self.pivtol_changed = true;
        // Upstream: `pivtol = min(pivtolmax, pivtol^0.75)`
        // (cpp:832).
        self.options.pivtol = self.options.pivtolmax.min(self.options.pivtol.powf(0.75));
        true
    }

    fn provides_inertia(&self) -> bool {
        true
    }

    fn matrix_format(&self) -> EMatrixFormat {
        EMatrixFormat::TripletFormat
    }
}

/// Symbolic-phase workspace sizes for MA57, computed in i64 and validated
/// against MA57's 32-bit `Index`. Returns `(lkeep, liwork)` when both fit, or
/// `None` when the problem is too large for MA57's index range. Sizing follows
/// upstream cpp:536: `lkeep = 5*N + NE + max(N, NE) + 42`, `liwork = 5*N`.
fn ma57_symbolic_sizes(n: Index, ne: Index) -> Option<(Index, Index)> {
    let (n64, ne64) = (n as i64, ne as i64);
    let lkeep = 5 * n64 + ne64 + n64.max(ne64) + 42;
    let liwork = 5 * n64;
    if lkeep > Index::MAX as i64 || liwork > Index::MAX as i64 {
        return None;
    }
    Some((lkeep as Index, liwork as Index))
}

/// Grow a MA57 suggested workspace size `base` by `scale` (>= 1), rounding up,
/// and validate the result fits in `Index`. Returns `max(scaled, base)` (never
/// shrinking below MA57's own minimum), or `None` if the scaled length exceeds
/// the 32-bit index range.
fn ma57_scaled_size(base: Index, scale: Number) -> Option<Index> {
    let scaled = (base as Number * scale).ceil();
    if scaled > Index::MAX as Number {
        return None;
    }
    Some((scaled as Index).max(base))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L10: the symbolic sizing `5*N + NE + max(N,NE) + 42` overflows i32 near
    /// ne ~ 3e8. The guard computes it in i64 and reports `None` (mapped to a
    /// clean `FatalError`) instead of wrapping into a negative/garbage length.
    #[test]
    fn ma57_symbolic_sizes_guards_i32_overflow() {
        // Small problem: exact sizing, lkeep = 5*N + NE + max(N,NE) + 42,
        // liwork = 5*N.
        assert_eq!(
            ma57_symbolic_sizes(10, 20),
            Some((5 * 10 + 20 + 20 + 42, 5 * 10))
        );
        // n = ne = 3e8: 7*3e8 = 2.1e9 < i32::MAX (2.147e9) — still representable.
        assert!(ma57_symbolic_sizes(300_000_000, 300_000_000).is_some());
        // n = ne = 3.5e8: 7*3.5e8 = 2.45e9 > i32::MAX. The i32 expression would
        // wrap to a negative length; the guard returns None.
        assert!(ma57_symbolic_sizes(350_000_000, 350_000_000).is_none());
    }

    /// L10: a suggested size scaled by `pre_alloc` must still fit in i32, and
    /// must never shrink below MA57's own minimum.
    #[test]
    fn ma57_scaled_size_guards_overflow_and_floors_at_base() {
        assert_eq!(ma57_scaled_size(1000, 1.05), Some(1050));
        // scale < 1 must not drop below the MA57-suggested base.
        assert_eq!(ma57_scaled_size(1000, 0.5), Some(1000));
        // base near i32::MAX scaled up overflows the index range -> None.
        assert_eq!(ma57_scaled_size(Index::MAX - 1, 1.05), None);
    }

    /// L11: the MA57C workspace is cached on the struct and reused across
    /// backsolves. Repeated solves against one factorization must stay correct
    /// despite stale scratch, and must not reallocate `work`.
    #[test]
    fn backsolve_reuses_workspace_across_repeated_solves() {
        let mut s = Ma57SolverInterface::new();
        let n: Index = 2;
        let ne: Index = 3;
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        assert_eq!(
            s.initialize_structure(n, ne, &irn, &jcn),
            ESymSolverStatus::Success
        );
        // A = [[2, 1], [1, 3]], det 5, A^-1 = [[3, -1], [-1, 2]] / 5.
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);

        // First solve factors (new_matrix = true); A * (1, 1) = (3, 4).
        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        assert!((rhs[0] - 1.0).abs() < 1e-12 && (rhs[1] - 1.0).abs() < 1e-12);
        // The workspace is now populated and reused, not a per-solve local.
        // Length is n*nrhs with nrhs = 1.
        assert_eq!(s.work.len(), n as usize);
        let cap_after_first = s.work.capacity();

        // Further solves reuse the factor (new_matrix = false) and the cached
        // workspace. Each must be correct; capacity must not grow.
        for &(b0, b1, x0, x1) in &[
            (5.0, 10.0, 1.0, 3.0), // A * (1, 3)
            (2.0, 3.0, 3.0 / 5.0, 4.0 / 5.0),
            (-1.0, 4.0, -7.0 / 5.0, 9.0 / 5.0),
        ] {
            let mut r = [b0, b1];
            assert_eq!(
                s.multi_solve(false, &irn, &jcn, 1, &mut r, false, 0),
                ESymSolverStatus::Success
            );
            assert!((r[0] - x0).abs() < 1e-10, "x0 = {}, want {}", r[0], x0);
            assert!((r[1] - x1).abs() < 1e-10, "x1 = {}, want {}", r[1], x1);
            assert_eq!(
                s.work.capacity(),
                cap_after_first,
                "cached workspace must not reallocate on same-size solves"
            );
        }
    }

    /// 2x2 SPD matrix
    /// ```text
    ///   [ 2  1 ]
    ///   [ 1  3 ]
    /// ```
    /// Lower-triangle triplet (1-based for MA57): (1,1)=2, (2,1)=1, (2,2)=3.
    /// Solving A x = (4, 5)^T gives x = (1, 4/3)... let's pick an
    /// easier RHS: A * (1, 1) = (3, 4), so RHS (3,4) → x (1,1).
    #[test]
    fn factor_and_solve_spd_2x2() {
        let mut s = Ma57SolverInterface::new();
        let n: Index = 2;
        let ne: Index = 3;
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];

        assert_eq!(
            s.initialize_structure(n, ne, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);

        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );

        assert!((rhs[0] - 1.0).abs() < 1e-12, "x0 = {}", rhs[0]);
        assert!((rhs[1] - 1.0).abs() < 1e-12, "x1 = {}", rhs[1]);

        // SPD → 0 negative eigenvalues.
        assert_eq!(s.number_of_neg_evals(), 0);
        assert!(s.provides_inertia());
        assert_eq!(s.matrix_format(), EMatrixFormat::TripletFormat);
    }

    /// 2x2 indefinite matrix
    /// ```text
    ///   [ 1   2 ]
    ///   [ 2   1 ]
    /// ```
    /// Eigenvalues 3, -1: exactly one negative.
    #[test]
    fn detects_one_negative_eigenvalue() {
        let mut s = Ma57SolverInterface::new();
        let n: Index = 2;
        let ne: Index = 3;
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];

        assert_eq!(
            s.initialize_structure(n, ne, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[1.0, 2.0, 1.0]);

        // Matrix has inertia (1 pos, 1 neg). Solve A x = (3, 3)
        // → x = (1, 1).
        let mut rhs = [3.0, 3.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 1),
            ESymSolverStatus::Success
        );
        assert_eq!(s.number_of_neg_evals(), 1);
        assert!((rhs[0] - 1.0).abs() < 1e-12);
        assert!((rhs[1] - 1.0).abs() < 1e-12);
    }

    /// Asking for the wrong inertia returns `WrongInertia` and does
    /// not solve.
    #[test]
    fn check_neg_evals_mismatch_returns_wrong_inertia() {
        let mut s = Ma57SolverInterface::new();
        let n: Index = 2;
        let ne: Index = 3;
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        assert_eq!(
            s.initialize_structure(n, ne, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]); // SPD
        let mut rhs = [3.0, 4.0];

        // Claim 1 negative eigenvalue but matrix is SPD.
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 1),
            ESymSolverStatus::WrongInertia
        );
    }

    /// `increase_quality` raises `pivtol` toward `pivtolmax` and sets
    /// `pivtol_changed`, so the next non-`new_matrix` solve returns
    /// `CallAgain`.
    #[test]
    fn increase_quality_then_resolve_triggers_call_again() {
        let mut s = Ma57SolverInterface::new();
        let n: Index = 2;
        let ne: Index = 3;
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        assert_eq!(
            s.initialize_structure(n, ne, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );

        let pivtol_before = s.pivtol();
        assert!(s.increase_quality());
        assert!(s.pivtol() > pivtol_before);

        // new_matrix=false after pivtol change → CallAgain.
        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(false, &irn, &jcn, 1, &mut rhs, false, 0),
            ESymSolverStatus::CallAgain
        );

        // After CallAgain, caller refills values and retries; the
        // backend re-factorizes and solves.
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(false, &irn, &jcn, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
    }

    /// `increase_quality` returns `false` once `pivtol` has reached
    /// `pivtolmax`.
    #[test]
    fn increase_quality_caps_at_pivtolmax() {
        let mut opts = Options::defaults();
        opts.pivtol = 1e-4;
        opts.pivtolmax = 1e-4;
        let mut s = Ma57SolverInterface::with_options(opts);
        assert!(!s.increase_quality());
    }
}
