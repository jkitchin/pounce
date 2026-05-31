//! Algorithm-side wrapper that drives a sparse symmetric backend.
//!
//! Port of `Algorithm/LinearSolvers/IpTSymLinearSolver.{hpp,cpp}` from
//! Ipopt 3.14.x. This is the layer between an algorithm's "give me a
//! `SymMatrix` plus RHS, return the solution" expectation and the
//! per-backend [`SparseSymLinearSolverInterface`] contract.
//!
//! Responsibilities, mirroring upstream:
//!
//! * Marshal a triplet `(airn, ajcn, vals)` matrix into the layout the
//!   backend declared via [`SparseSymLinearSolverInterface::matrix_format`].
//! * If a [`TSymScalingMethod`] is configured, compute symmetric
//!   scaling factors `s` once per refactor and apply
//!   `A' = diag(s) A diag(s)` / `b' = diag(s) b` / `x = diag(s) x'`.
//! * Drive the backend's `CALL_AGAIN` retry loop (MA57 grow case).
//! * Forward `IncreaseQuality` to the backend, with the upstream
//!   "switch on linear-system scaling on demand" optimization.
//!
//! Tag-based change detection (`SymMatrix::HasChanged`) is *not* part
//! of this wrapper; callers in the Phase-6 KKT layer pass `new_matrix`
//! explicitly. That keeps Phase-4 self-contained while leaving the
//! upstream semantics intact.

use crate::scaling::TSymScalingMethod;
use crate::sparse_sym_iface::{EMatrixFormat, FactorPattern, SparseSymLinearSolverInterface};
use crate::status::ESymSolverStatus;
use crate::sym_solver::SymLinearSolver;
use pounce_common::types::{Index, Number};
use pounce_linalg::triplet_convert::{TriFull, TripletToCsrConverter};

/// Driver wrapping a [`SparseSymLinearSolverInterface`] (and optionally
/// a [`TSymScalingMethod`]).
pub struct TSymLinearSolver {
    backend: Box<dyn SparseSymLinearSolverInterface>,
    scaling_method: Option<Box<dyn TSymScalingMethod>>,
    matrix_format: EMatrixFormat,
    converter: Option<TripletToCsrConverter>,

    /// `true` once [`Self::initialize_structure`] has succeeded.
    initialized: bool,
    /// `true` once the row/column index arrays are populated (= ditto
    /// in this port; see upstream's `have_structure_` for warm-start
    /// semantics).
    have_structure: bool,
    /// `true` if the wrapper should currently apply scaling.
    use_scaling: bool,
    /// Set by [`Self::increase_quality`] when scaling-on-demand fires;
    /// triggers a one-shot scaling-factor recompute on the next solve.
    just_switched_on_scaling: bool,
    /// Mirrors `linear_scaling_on_demand`. `true` keeps scaling off
    /// until `increase_quality` switches it on; `false` scales every
    /// refactor.
    linear_scaling_on_demand: bool,

    dim: Index,
    nonzeros_triplet: Index,
    nonzeros_compressed: Index,

    /// 1-based row indices, one per triplet entry.
    airn: Vec<Index>,
    /// 1-based column indices, one per triplet entry.
    ajcn: Vec<Index>,
    /// Per-row symmetric scaling factors (length `dim`). Empty unless
    /// a scaling method is configured.
    scaling_factors: Vec<Number>,
}

impl std::fmt::Debug for TSymLinearSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TSymLinearSolver")
            .field("matrix_format", &self.matrix_format)
            .field("dim", &self.dim)
            .field("nonzeros_triplet", &self.nonzeros_triplet)
            .field("nonzeros_compressed", &self.nonzeros_compressed)
            .field("use_scaling", &self.use_scaling)
            .field("initialized", &self.initialized)
            .finish_non_exhaustive()
    }
}

impl TSymLinearSolver {
    /// Build a driver around `backend`. Pass `Some(scaling)` to enable
    /// symmetric scaling. `linear_scaling_on_demand=true` matches
    /// upstream's default and keeps scaling off until
    /// [`Self::increase_quality`] turns it on.
    pub fn new(
        backend: Box<dyn SparseSymLinearSolverInterface>,
        scaling_method: Option<Box<dyn TSymScalingMethod>>,
        linear_scaling_on_demand: bool,
    ) -> Self {
        let matrix_format = backend.matrix_format();
        let converter = match matrix_format {
            EMatrixFormat::TripletFormat => None,
            EMatrixFormat::CsrFormat0Offset => {
                Some(TripletToCsrConverter::new(0, TriFull::Triangular))
            }
            EMatrixFormat::CsrFormat1Offset => {
                Some(TripletToCsrConverter::new(1, TriFull::Triangular))
            }
            EMatrixFormat::CsrFullFormat0Offset => {
                Some(TripletToCsrConverter::new(0, TriFull::Full))
            }
            EMatrixFormat::CsrFullFormat1Offset => {
                Some(TripletToCsrConverter::new(1, TriFull::Full))
            }
        };
        let use_scaling = scaling_method.is_some() && !linear_scaling_on_demand;
        Self {
            backend,
            scaling_method,
            matrix_format,
            converter,
            initialized: false,
            have_structure: false,
            use_scaling,
            just_switched_on_scaling: false,
            linear_scaling_on_demand,
            dim: 0,
            nonzeros_triplet: 0,
            nonzeros_compressed: 0,
            airn: Vec::new(),
            ajcn: Vec::new(),
            scaling_factors: Vec::new(),
        }
    }

    /// Pin the triplet sparsity pattern. Must be called once before
    /// the first [`Self::multi_solve`]. `airn` / `ajcn` are 1-based.
    /// Mirrors the bulk of `TSymLinearSolver::InitializeStructure`.
    pub fn initialize_structure(
        &mut self,
        dim: Index,
        airn: &[Index],
        ajcn: &[Index],
    ) -> ESymSolverStatus {
        assert_eq!(airn.len(), ajcn.len());
        let nz = airn.len() as Index;
        self.dim = dim;
        self.nonzeros_triplet = nz;
        self.airn = airn.to_vec();
        self.ajcn = ajcn.to_vec();

        let (ia, ja, nonzeros) = match self.converter.as_mut() {
            None => (&self.airn[..], &self.ajcn[..], self.nonzeros_triplet),
            Some(conv) => {
                let nonzeros_compressed = conv.initialize(self.dim, &self.airn, &self.ajcn);
                self.nonzeros_compressed = nonzeros_compressed;
                (conv.ia(), conv.ja(), nonzeros_compressed)
            }
        };
        let status = self.backend.initialize_structure(dim, nonzeros, ia, ja);
        if status != ESymSolverStatus::Success {
            return status;
        }
        if self.scaling_method.is_some() {
            self.scaling_factors = vec![0.0; dim as usize];
        }
        self.have_structure = true;
        self.initialized = true;
        status
    }

    /// Solve `A x = b` (or multiple RHS).
    ///
    /// `vals` is the new triplet-format value array (length
    /// `nonzeros_triplet`). `new_matrix=true` requests a refactor
    /// (and a fresh scaling-factor computation if scaling is on);
    /// `new_matrix=false` reuses the existing factor and just runs
    /// back-substitution.
    ///
    /// `rhs_vals` packs `nrhs` columns, each length `dim`, in
    /// column-major layout. Solutions overwrite `rhs_vals`.
    #[allow(clippy::too_many_arguments)]
    pub fn multi_solve(
        &mut self,
        vals: &[Number],
        new_matrix: bool,
        nrhs: Index,
        rhs_vals: &mut [Number],
        check_neg_evals: bool,
        number_of_neg_evals: Index,
    ) -> ESymSolverStatus {
        debug_assert!(self.initialized);
        debug_assert_eq!(vals.len(), self.nonzeros_triplet as usize);
        debug_assert_eq!(rhs_vals.len(), (self.dim * nrhs) as usize);

        // One-shot KKT dump for backend-comparison testing. Triggered
        // when POUNCE_DBG_KKT_DUMP is set to a file path; writes one
        // binary record (dim, nnz, nrhs, ia[], ja[], vals[], rhs[]) on
        // the Nth multi_solve call (N = POUNCE_DBG_KKT_DUMP_SKIP, default 0),
        // then disables itself.
        //
        // DEPRECATED: superseded by the unified `--dump kkt:<spec>` CLI
        // surface (see `pounce_common::diagnostics`). Kept for the
        // existing offline FERAL/MA57/LAPACK binary-comparison tool;
        // the env var emits a one-shot warning on first observation.
        {
            use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
            static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
            static WARNED: AtomicBool = AtomicBool::new(false);
            let n_call = CALL_COUNT.fetch_add(1, Ordering::SeqCst);
            let skip: usize = std::env::var("POUNCE_DBG_KKT_DUMP_SKIP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if n_call < skip {
                // not yet
            } else if let Ok(path) = std::env::var("POUNCE_DBG_KKT_DUMP") {
                if !WARNED.swap(true, Ordering::SeqCst) {
                    tracing::warn!(
                        target: "pounce::linsol",
                        "POUNCE_DBG_KKT_DUMP is deprecated; prefer `--dump kkt:<iter-spec>` (see pounce --help)"
                    );
                }
                use std::io::Write;
                if let Ok(mut f) = std::fs::File::create(&path) {
                    let dim = self.dim as u64;
                    let nnz = self.nonzeros_triplet as u64;
                    let nrhs64 = nrhs as u64;
                    let _ = f.write_all(&dim.to_le_bytes());
                    let _ = f.write_all(&nnz.to_le_bytes());
                    let _ = f.write_all(&nrhs64.to_le_bytes());
                    for &i in &self.airn {
                        let _ = f.write_all(&(i as i64).to_le_bytes());
                    }
                    for &j in &self.ajcn {
                        let _ = f.write_all(&(j as i64).to_le_bytes());
                    }
                    for &v in vals {
                        let _ = f.write_all(&v.to_le_bytes());
                    }
                    for &v in &*rhs_vals {
                        let _ = f.write_all(&v.to_le_bytes());
                    }
                    let _ = f.flush();
                }
                // SAFETY: removing an env var is safe in single-threaded
                // setup; this dump fires from the main IPM thread.
                unsafe {
                    std::env::remove_var("POUNCE_DBG_KKT_DUMP");
                }
            }
        }

        // Push values + (optional) scaling into the backend.
        let mut new_matrix = new_matrix;
        if new_matrix || self.just_switched_on_scaling {
            self.give_matrix_to_solver(true, vals);
            new_matrix = true;
        }

        // Apply scaling to RHS columns (multiply by `s_i` per row).
        if self.use_scaling {
            for irhs in 0..nrhs as usize {
                let base = irhs * self.dim as usize;
                for i in 0..self.dim as usize {
                    rhs_vals[base + i] *= self.scaling_factors[i];
                }
            }
        }

        // Backend solve, with `CALL_AGAIN` retry (MA57 grow path).
        // Pre-resolve the index arrays into local pointers so we can
        // hand them to the backend without re-borrowing `self`.
        let status = loop {
            let (ia_ptr, ia_len, ja_ptr, ja_len) = match self.converter.as_ref() {
                None => (
                    self.airn.as_ptr(),
                    self.airn.len(),
                    self.ajcn.as_ptr(),
                    self.ajcn.len(),
                ),
                Some(c) => (c.ia().as_ptr(), c.ia().len(), c.ja().as_ptr(), c.ja().len()),
            };
            // SAFETY: the slices live in `self.airn/ajcn` or in the
            // converter, both owned by `self`; the pointers are valid
            // for the duration of this `multi_solve` call.
            let (ia, ja) = unsafe {
                (
                    std::slice::from_raw_parts(ia_ptr, ia_len),
                    std::slice::from_raw_parts(ja_ptr, ja_len),
                )
            };
            let s = self.backend.multi_solve(
                new_matrix,
                ia,
                ja,
                nrhs,
                rhs_vals,
                check_neg_evals,
                number_of_neg_evals,
            );
            if s == ESymSolverStatus::CallAgain {
                self.give_matrix_to_solver(false, vals);
                continue;
            }
            break s;
        };

        if status == ESymSolverStatus::Success && self.use_scaling {
            // Solution comes back in scaled coordinates `x' = diag(s)
            // x`; restore by another diag(s) multiply (since the
            // scaled system is `(D A D)(D^-1 x) = D b` and we passed
            // `D b`, the backend returns `D^-1 x`, hence multiply by
            // `D` once more — see cpp:286-289).
            for irhs in 0..nrhs as usize {
                let base = irhs * self.dim as usize;
                for i in 0..self.dim as usize {
                    rhs_vals[base + i] *= self.scaling_factors[i];
                }
            }
        }

        status
    }

    /// Push `vals` (triplet-format) into the backend in the right
    /// layout, optionally computing scaling factors and applying the
    /// symmetric scale. Mirrors `TSymLinearSolver::GiveMatrixToSolver`.
    fn give_matrix_to_solver(&mut self, new_matrix: bool, vals: &[Number]) {
        // For triplet-format backends we write directly into the
        // backend's array; for CSR backends we marshal via a temporary
        // and call `convert_values`.
        if self.matrix_format == EMatrixFormat::TripletFormat && !self.use_scaling {
            let pa = self.backend.values_array_mut();
            pa[..self.nonzeros_triplet as usize]
                .copy_from_slice(&vals[..self.nonzeros_triplet as usize]);
            return;
        }

        // Stage values in a local buffer so we can scale before
        // shipping to the backend.
        let mut atriplet: Vec<Number> = vals[..self.nonzeros_triplet as usize].to_vec();

        if self.use_scaling {
            if new_matrix || self.just_switched_on_scaling {
                // `use_scaling` implies the scaling method is set
                // (checked at construction time).
                let Some(method) = self.scaling_method.as_mut() else {
                    unreachable!("use_scaling without a scaling method")
                };
                let ok = method.compute_sym_t_scaling_factors(
                    self.dim,
                    self.nonzeros_triplet,
                    &self.airn,
                    &self.ajcn,
                    &atriplet,
                    &mut self.scaling_factors,
                );
                assert!(ok, "scaling method failed");
                self.just_switched_on_scaling = false;
            }
            for (i, a) in atriplet
                .iter_mut()
                .enumerate()
                .take(self.nonzeros_triplet as usize)
            {
                let r = (self.airn[i] - 1) as usize;
                let c = (self.ajcn[i] - 1) as usize;
                *a *= self.scaling_factors[r] * self.scaling_factors[c];
            }
        }

        if self.matrix_format == EMatrixFormat::TripletFormat {
            let pa = self.backend.values_array_mut();
            pa[..self.nonzeros_triplet as usize].copy_from_slice(&atriplet);
        } else {
            let Some(conv) = self.converter.as_ref() else {
                unreachable!("non-triplet matrix_format requires a converter");
            };
            let pa = self.backend.values_array_mut();
            conv.convert_values(&atriplet, &mut pa[..self.nonzeros_compressed as usize]);
        }
    }

    /// Pass-through to the backend's diagnostic factor-pattern
    /// accessor. Returns `None` when the backend does not expose its
    /// factor data (e.g. MA57). Consumed by the `--dump kkt:*+L` path.
    pub fn factor_pattern(&self, want_values: bool) -> Option<FactorPattern> {
        self.backend.factor_pattern(want_values)
    }
}

impl SymLinearSolver for TSymLinearSolver {
    fn number_of_neg_evals(&self) -> Index {
        self.backend.number_of_neg_evals()
    }

    /// Mirrors upstream's `IncreaseQuality`: switching scaling on at
    /// the wrapper level (`linear_scaling_on_demand=true` path) takes
    /// precedence over asking the backend for tighter pivoting.
    fn increase_quality(&mut self) -> bool {
        if self.scaling_method.is_some() && !self.use_scaling && self.linear_scaling_on_demand {
            self.use_scaling = true;
            self.just_switched_on_scaling = true;
            return true;
        }
        self.backend.increase_quality()
    }

    fn provides_inertia(&self) -> bool {
        self.backend.provides_inertia()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scaling::IdentityScalingMethod;

    /// Mock triplet-format backend that exposes the values array,
    /// records the most-recent solve call, and returns a hand-rolled
    /// solution. Lets us exercise the wrapper without an FFI dep.
    #[derive(Default)]
    struct MockBackend {
        dim: Index,
        nz: Index,
        a: Vec<Number>,
        last_solve_was_new_matrix: bool,
        last_solve_was_scaled_a: Option<Vec<Number>>,
        canned_solution: Vec<Number>,
        neg_evals: Index,
        increase_quality_calls: u32,
        max_increase_quality_calls: u32,
    }

    impl SparseSymLinearSolverInterface for MockBackend {
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
            _ia: &[Index],
            _ja: &[Index],
            nrhs: Index,
            rhs_vals: &mut [Number],
            _check: bool,
            _nev: Index,
        ) -> ESymSolverStatus {
            self.last_solve_was_new_matrix = new_matrix;
            self.last_solve_was_scaled_a = Some(self.a.clone());
            assert_eq!(rhs_vals.len(), (self.dim * nrhs) as usize);
            for irhs in 0..nrhs as usize {
                let base = irhs * self.dim as usize;
                rhs_vals[base..base + self.dim as usize].copy_from_slice(&self.canned_solution);
            }
            ESymSolverStatus::Success
        }
        fn number_of_neg_evals(&self) -> Index {
            self.neg_evals
        }
        fn increase_quality(&mut self) -> bool {
            self.increase_quality_calls += 1;
            self.increase_quality_calls <= self.max_increase_quality_calls
        }
        fn provides_inertia(&self) -> bool {
            true
        }
        fn matrix_format(&self) -> EMatrixFormat {
            EMatrixFormat::TripletFormat
        }
    }

    fn make_2x2_indef_pattern() -> ([Index; 3], [Index; 3]) {
        ([1, 2, 2], [1, 1, 2])
    }

    #[test]
    fn unscaled_triplet_solve_passes_values_through() {
        let backend = MockBackend {
            canned_solution: vec![10.0, 20.0],
            ..Default::default()
        };
        let mut solver = TSymLinearSolver::new(Box::new(backend), None, false);
        let (irn, jcn) = make_2x2_indef_pattern();
        assert_eq!(
            solver.initialize_structure(2, &irn, &jcn),
            ESymSolverStatus::Success
        );

        let vals = [2.0, 1.0, 3.0];
        let mut rhs = [3.0, 4.0];
        assert_eq!(
            solver.multi_solve(&vals, true, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        // Mock writes its canned solution.
        assert_eq!(rhs, [10.0, 20.0]);
        assert!(solver.provides_inertia());
    }

    #[test]
    fn identity_scaling_does_not_change_values() {
        let backend = MockBackend {
            canned_solution: vec![1.0, 1.0],
            ..Default::default()
        };
        // linear_scaling_on_demand=false → scaling active from the
        // first solve.
        let mut solver = TSymLinearSolver::new(
            Box::new(backend),
            Some(Box::new(IdentityScalingMethod)),
            false,
        );
        let (irn, jcn) = make_2x2_indef_pattern();
        solver.initialize_structure(2, &irn, &jcn);

        let vals = [2.0, 1.0, 3.0];
        let mut rhs = [4.0, 5.0];
        assert_eq!(
            solver.multi_solve(&vals, true, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        // Identity scaling: backend should have received the original
        // value array unchanged, and the canned solution survives the
        // unscale step (multiplied by 1.0 twice).
        assert_eq!(rhs, [1.0, 1.0]);
    }

    #[test]
    fn nontrivial_scaling_premultiplies_matrix_and_postmultiplies_solution() {
        // Scaling method that returns s = (2, 3). After scaling, the
        // backend should see (D A D) where D = diag(2,3); solving with
        // RHS (D b) returns (D^-1 x), and the wrapper unscales by D
        // once more to recover x.
        struct DiagTwoThree;
        impl TSymScalingMethod for DiagTwoThree {
            fn compute_sym_t_scaling_factors(
                &mut self,
                _n: Index,
                _nnz: Index,
                _airn: &[Index],
                _ajcn: &[Index],
                _a: &[Number],
                scaling_factors: &mut [Number],
            ) -> bool {
                scaling_factors[0] = 2.0;
                scaling_factors[1] = 3.0;
                true
            }
        }

        let backend = MockBackend {
            // Wrapper passes scaled RHS = (2*4, 3*5) = (8, 15).
            // Mock returns `canned_solution` ignoring the input;
            // wrapper then unscales: x = D · canned = (2 * c0, 3 * c1).
            canned_solution: vec![7.0, 11.0],
            ..Default::default()
        };
        let mut solver =
            TSymLinearSolver::new(Box::new(backend), Some(Box::new(DiagTwoThree)), false);
        let (irn, jcn) = make_2x2_indef_pattern();
        solver.initialize_structure(2, &irn, &jcn);

        let vals = [2.0, 1.0, 3.0];
        let mut rhs = [4.0, 5.0];
        assert_eq!(
            solver.multi_solve(&vals, true, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        assert_eq!(rhs, [2.0 * 7.0, 3.0 * 11.0]);
    }

    #[test]
    fn increase_quality_switches_on_scaling_first() {
        let backend = MockBackend {
            canned_solution: vec![0.0, 0.0],
            max_increase_quality_calls: 5,
            ..Default::default()
        };
        let mut solver = TSymLinearSolver::new(
            Box::new(backend),
            Some(Box::new(IdentityScalingMethod)),
            true, // on demand
        );
        // First IncreaseQuality flips on scaling, does NOT touch the
        // backend.
        assert!(solver.increase_quality());
        // Second IncreaseQuality goes to the backend.
        assert!(solver.increase_quality());
    }

    #[test]
    fn increase_quality_without_scaling_goes_straight_to_backend() {
        let backend = MockBackend {
            max_increase_quality_calls: 1,
            ..Default::default()
        };
        let mut solver = TSymLinearSolver::new(Box::new(backend), None, false);
        assert!(solver.increase_quality());
        // Backend caps at 1; second call returns false.
        assert!(!solver.increase_quality());
    }
}
