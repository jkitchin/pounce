//! `Solver` — value-typed session API that holds an `IpoptApplication`,
//! its TNLP, and the converged KKT factor between calls.
//!
//! This is Phase 3a of the factor-reuse work tracked in
//! [pounce#16](https://github.com/jkitchin/pounce/issues/16). It is
//! the public surface for callers who want to:
//!
//! 1. Run a normal IPM solve, then
//! 2. Issue many cheap operations against the converged factor
//!    (`kkt_solve`, `parametric_step`) without going through the
//!    [`set_on_converged`] callback shape that [`crate::SensSolve`]
//!    requires.
//!
//! [`set_on_converged`]: pounce_algorithm::IpoptApplication::set_on_converged
//!
//! # Usage
//!
//! ```ignore
//! use pounce_sensitivity::Solver;
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! let app = make_configured_app();
//! let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(MyTnlp));
//! let mut solver = Solver::new(app, tnlp);
//!
//! let status = solver.solve();
//! assert!(solver.converged().is_some());
//!
//! // Issue any number of back-solves against the same factor:
//! let dim = solver.kkt_dim().unwrap();
//! let mut lhs = vec![0.0; dim];
//! let rhs = vec![1.0; dim];
//! solver.kkt_solve(&rhs, &mut lhs).unwrap();
//!
//! // Parametric step with respect to a set of pinned equality
//! // constraints (same interpretation as [`crate::SensSolve`]):
//! let dx = solver.parametric_step(&[2, 3], &[-0.5, 0.0]).unwrap();
//! ```
//!
//! # Scope of Phase 3a
//!
//! - **In**: `solve()`, `converged()`, `kkt_solve()`, `parametric_step()`,
//!   `block_dims()` / `kkt_dim()`.
//! - **Deferred to Phase 3b**: `resolve()` (warm-start that reuses the
//!   linear backend pool), `compute_reduced_hessian()` on the Solver
//!   (currently only available through [`crate::SensSolve`]), and the
//!   `parametric_mpc` / `sensitivity_session` example binaries.

use std::cell::{Ref, RefCell};
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::TNLP;

use crate::backsolver::SensBacksolver;
use crate::schur_data::IndexSchurData;
use crate::sens_app::{SensApplication, SensOptions};
use crate::vec_util::dense_to_vec;
use crate::PdSensBacksolver;

/// Errors returned by post-convergence operations on [`Solver`].
#[derive(Debug, Clone)]
pub enum SolverError {
    /// The solver has not yet converged, or the last solve failed
    /// before producing a usable KKT factor.
    NotConverged,
    /// An input slice's length did not match the KKT dimension or the
    /// parameter count.
    BadShape {
        /// Human description of the mismatched buffer.
        what: &'static str,
        /// Length the caller passed.
        got: usize,
        /// Length expected.
        expected: usize,
    },
    /// The underlying back-solve failed (singular factor, numerical
    /// breakdown).
    BacksolveFailed,
    /// The underlying [`SensApplication`] step failed (e.g. row mapping
    /// invalid for the current problem).
    SensComputationFailed(String),
}

/// State captured at convergence: the user-visible iterate plus the
/// `PdSensBacksolver` that wraps the converged KKT factor.
///
/// Read this via [`Solver::converged`].
pub struct ConvergedState {
    /// IPM return status of the most recent solve.
    pub status: ApplicationReturnStatus,
    /// Final primal iterate `x*` (length `n_x`).
    pub x: Vec<Number>,
    /// Final objective value `f(x*)`.
    pub obj_val: Number,
    /// Converged KKT-factor wrapper. Owns `Rc` handles to the
    /// `PdFullSpaceSolver`, the IpoptData / Cq, and the NLP, so it
    /// outlives the IPM call frame.
    backsolver: PdSensBacksolver,
}

impl ConvergedState {
    /// Block dimensions of the compound KKT vector in
    /// `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)` order.
    pub fn block_dims(&self) -> [usize; 8] {
        self.backsolver.block_dims()
    }

    /// Total dimension of the compound KKT vector (sum of `block_dims`).
    pub fn kkt_dim(&self) -> usize {
        self.backsolver.dim()
    }
}

/// Session-style solver: holds an [`IpoptApplication`], its TNLP, and
/// the converged factor between calls.
pub struct Solver {
    app: IpoptApplication,
    tnlp: Rc<RefCell<dyn TNLP>>,
    /// Side channel populated by the `on_converged` callback installed
    /// in [`Self::solve`]. The `RefCell<Option<…>>` shape mirrors the
    /// pattern in [`crate::convenience`] (the callback closure needs
    /// shared mutable access; the `Option` is `None` before the first
    /// solve and gets overwritten on each call).
    state: Rc<RefCell<Option<ConvergedState>>>,
}

impl Solver {
    /// Build a new session. The `app` should already have its options
    /// configured and `initialize()` called.
    pub fn new(app: IpoptApplication, tnlp: Rc<RefCell<dyn TNLP>>) -> Self {
        Self {
            app,
            tnlp,
            state: Rc::new(RefCell::new(None)),
        }
    }

    /// Borrow the underlying `IpoptApplication` (e.g. to read its
    /// options table after a solve). Mutation between `solve` calls is
    /// supported via [`Self::app_mut`].
    pub fn app(&self) -> &IpoptApplication {
        &self.app
    }

    /// Mutable borrow of the underlying `IpoptApplication`. Useful for
    /// reconfiguring options before a follow-up `solve()`. Note that
    /// changing options that affect the KKT linear system between
    /// calls will invalidate the cached factor; the next `solve()`
    /// rebuilds it.
    pub fn app_mut(&mut self) -> &mut IpoptApplication {
        &mut self.app
    }

    /// Run the IPM to convergence. On a successful solve the
    /// [`ConvergedState`] (including the KKT backsolver) is stashed
    /// inside the `Solver` and accessible via [`Self::converged`].
    ///
    /// Each call to `solve()` overwrites the previous converged
    /// state; the previously held factor is dropped.
    pub fn solve(&mut self) -> ApplicationReturnStatus {
        // Clear any previous state so a failed re-solve doesn't leave
        // a stale factor visible.
        self.state.borrow_mut().take();

        let state_cb = Rc::clone(&self.state);
        self.app
            .set_on_converged(Box::new(move |data, cq, nlp, pd| {
                let curr = match data.borrow().curr.clone() {
                    Some(c) => c,
                    None => return,
                };
                let backsolver = match PdSensBacksolver::new(data, cq, nlp, Rc::clone(&pd)) {
                    Ok(b) => b,
                    Err(e) => {
                        // No session state is stored, so post-solve
                        // calls will report NotConverged; at least say
                        // why on stderr rather than failing silently.
                        eprintln!("pounce: Solver could not capture the KKT factor: {e}");
                        return;
                    }
                };
                let x = dense_to_vec(&*curr.x);
                let obj_val = cq.borrow_mut().curr_f();
                // Status is overwritten with the real value after
                // optimize_tnlp returns.
                *state_cb.borrow_mut() = Some(ConvergedState {
                    status: ApplicationReturnStatus::InternalError,
                    x,
                    obj_val,
                    backsolver,
                });
            }));

        let status = self.app.optimize_tnlp(Rc::clone(&self.tnlp));
        if let Some(s) = self.state.borrow_mut().as_mut() {
            s.status = status;
        }
        status
    }

    /// Borrow the converged state, if a successful solve has been
    /// run. Returns `None` if no solve has run or if the most recent
    /// solve failed before reaching convergence.
    pub fn converged(&self) -> Option<Ref<'_, ConvergedState>> {
        let r = self.state.borrow();
        r.as_ref()?;
        Some(Ref::map(r, |o| {
            o.as_ref()
                .unwrap_or_else(|| unreachable!("checked is_some above"))
        }))
    }

    /// Total dimension of the compound KKT vector (sum of
    /// `block_dims`). Returns `None` if no converged factor is held.
    pub fn kkt_dim(&self) -> Option<usize> {
        self.converged().map(|c| c.kkt_dim())
    }

    /// Block dimensions of the compound KKT vector in
    /// `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)` order. Returns `None` if
    /// no converged factor is held.
    pub fn block_dims(&self) -> Option<[usize; 8]> {
        self.converged().map(|c| c.block_dims())
    }

    /// Solve `K · lhs = rhs` against the converged KKT factor. Both
    /// slices must have length `kkt_dim()`; the layout is the flat
    /// `x || s || y_c || y_d || z_l || z_u || v_l || v_u` packing.
    ///
    /// `K` here is the **natural-units** (unscaled) KKT matrix: when
    /// the IPM solved with active NLP scaling, the backsolver scales
    /// the RHS/solution (all eight blocks, including the z/v
    /// bound-multiplier rows) so callers pass and receive data in the
    /// user's own units (pounce#128) — see
    /// [`crate::PdSensBacksolver::solve`]. For the raw scaled-space
    /// back-solve use [`Self::kkt_solve_scaled`].
    pub fn kkt_solve(&self, rhs: &[Number], lhs: &mut [Number]) -> Result<(), SolverError> {
        self.kkt_solve_impl(rhs, lhs, false)
    }

    /// [`Self::kkt_solve`] without the natural-units conjugation: the
    /// back-solve runs against the factor exactly as the IPM holds it
    /// (the solver's internal scaled space). Identical to `kkt_solve`
    /// when no NLP scaling is active.
    pub fn kkt_solve_scaled(&self, rhs: &[Number], lhs: &mut [Number]) -> Result<(), SolverError> {
        self.kkt_solve_impl(rhs, lhs, true)
    }

    fn kkt_solve_impl(
        &self,
        rhs: &[Number],
        lhs: &mut [Number],
        scaled: bool,
    ) -> Result<(), SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let total = state.backsolver.dim();
        if rhs.len() != total {
            return Err(SolverError::BadShape {
                what: "rhs",
                got: rhs.len(),
                expected: total,
            });
        }
        if lhs.len() != total {
            return Err(SolverError::BadShape {
                what: "lhs",
                got: lhs.len(),
                expected: total,
            });
        }
        let ok = if scaled {
            state.backsolver.solve_scaled_space(rhs, lhs)
        } else {
            state.backsolver.solve(rhs, lhs)
        };
        if ok {
            Ok(())
        } else {
            Err(SolverError::BacksolveFailed)
        }
    }

    /// Batched-RHS back-solve. `rhs_flat` and `lhs_flat` are row-major
    /// `(n_rhs, kkt_dim)` buffers; each row is solved against the
    /// same converged factor. Equivalent in result to looping
    /// [`Self::kkt_solve`] but reuses one `IteratesVector` for the
    /// RHS and one for the result across all `n_rhs` calls — see
    /// [`crate::algorithm_backsolver::PdSensBacksolver::solve_many`].
    pub fn kkt_solve_many(
        &self,
        rhs_flat: &[Number],
        lhs_flat: &mut [Number],
        n_rhs: usize,
    ) -> Result<(), SolverError> {
        self.kkt_solve_many_impl(rhs_flat, lhs_flat, n_rhs, false)
    }

    /// [`Self::kkt_solve_many`] without the natural-units
    /// conjugation (the batched sibling of [`Self::kkt_solve_scaled`]).
    pub fn kkt_solve_many_scaled(
        &self,
        rhs_flat: &[Number],
        lhs_flat: &mut [Number],
        n_rhs: usize,
    ) -> Result<(), SolverError> {
        self.kkt_solve_many_impl(rhs_flat, lhs_flat, n_rhs, true)
    }

    fn kkt_solve_many_impl(
        &self,
        rhs_flat: &[Number],
        lhs_flat: &mut [Number],
        n_rhs: usize,
        scaled: bool,
    ) -> Result<(), SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let total = state.backsolver.dim();
        let expected = n_rhs * total;
        if rhs_flat.len() != expected {
            return Err(SolverError::BadShape {
                what: "rhs",
                got: rhs_flat.len(),
                expected,
            });
        }
        if lhs_flat.len() != expected {
            return Err(SolverError::BadShape {
                what: "lhs",
                got: lhs_flat.len(),
                expected,
            });
        }
        let ok = if scaled {
            state
                .backsolver
                .solve_many_scaled_space(rhs_flat, lhs_flat, n_rhs)
        } else {
            state.backsolver.solve_many(rhs_flat, lhs_flat, n_rhs)
        };
        if ok {
            Ok(())
        } else {
            Err(SolverError::BacksolveFailed)
        }
    }

    /// First-order parametric step `Δx ≈ ∂x*/∂p · Δp` for a set of
    /// pinned equality constraints. `pin_constraint_indices` are
    /// 0-based indices into the user's `g(x)`; `deltas` is the
    /// perturbation `Δp` (same length).
    ///
    /// Returns the `n_x`-long primal step. For the full KKT-space
    /// step, use [`Self::kkt_solve`] directly.
    pub fn parametric_step(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
    ) -> Result<Vec<Number>, SolverError> {
        if pin_constraint_indices.len() != deltas.len() {
            return Err(SolverError::BadShape {
                what: "deltas",
                got: deltas.len(),
                expected: pin_constraint_indices.len(),
            });
        }
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;

        // Map user g-indices to y_c rows through the NLP's c/d-split
        // permutation (pounce#128; matches `convenience.rs`).
        let dims = state.backsolver.block_dims();
        let n_x = dims[0];
        let param_rows = state
            .backsolver
            .map_pin_g_to_kkt_rows(pin_constraint_indices)
            .map_err(SolverError::SensComputationFailed)?;
        let signs = vec![1; pin_constraint_indices.len()];
        let a_data = IndexSchurData::from_parts(param_rows, signs)
            .map_err(|e| SolverError::SensComputationFailed(format!("{e:?}")))?;

        let opts = SensOptions {
            run_sens: true,
            ..SensOptions::default()
        };
        let sens_app = SensApplication::new(a_data, state.backsolver.clone(), opts);
        let n_full = state.backsolver.dim();
        let mut dx_full = vec![0.0; n_full];
        if !sens_app.parametric_step(deltas, &mut dx_full) {
            return Err(SolverError::SensComputationFailed(
                "SensApplication::parametric_step failed".into(),
            ));
        }
        dx_full.truncate(n_x);
        Ok(dx_full)
    }

    /// Full KKT-space parametric step for a set of pinned equality
    /// constraints: the same computation as [`Self::parametric_step`],
    /// returned WITHOUT truncating to the primal block. The layout is
    /// the compound KKT vector `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)`;
    /// use [`Self::block_dims`] for the block sizes and
    /// [`Self::g_multiplier_rows`] to locate a constraint's multiplier
    /// row. This exposes the multiplier sensitivities `∂λ*/∂p`
    /// alongside the primal step.
    pub fn parametric_step_full(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
    ) -> Result<Vec<Number>, SolverError> {
        if pin_constraint_indices.len() != deltas.len() {
            return Err(SolverError::BadShape {
                what: "deltas",
                got: deltas.len(),
                expected: pin_constraint_indices.len(),
            });
        }
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;

        let param_rows = state
            .backsolver
            .map_pin_g_to_kkt_rows(pin_constraint_indices)
            .map_err(SolverError::SensComputationFailed)?;
        let signs = vec![1; pin_constraint_indices.len()];
        let a_data = IndexSchurData::from_parts(param_rows, signs)
            .map_err(|e| SolverError::SensComputationFailed(format!("{e:?}")))?;

        let opts = SensOptions {
            run_sens: true,
            ..SensOptions::default()
        };
        let sens_app = SensApplication::new(a_data, state.backsolver.clone(), opts);
        let n_full = state.backsolver.dim();
        let mut dx_full = vec![0.0; n_full];
        if !sens_app.parametric_step(deltas, &mut dx_full) {
            return Err(SolverError::SensComputationFailed(
                "SensApplication::parametric_step failed".into(),
            ));
        }
        Ok(dx_full)
    }

    /// Flat rows of the compound KKT vector holding the equality
    /// multipliers `y_c` for the given 0-based **full-g** constraint
    /// indices. `None` for inequalities (their multipliers live in the
    /// `y_d` block; mapping those is not exposed here). Row `r` of a
    /// [`Self::parametric_step_full`] result is then `∂λ_g/∂p · Δp`.
    pub fn g_multiplier_rows(
        &self,
        g_indices: &[Index],
    ) -> Result<Vec<Option<Index>>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let dims = state.backsolver.block_dims();
        let y_c_offset = (dims[0] + dims[1]) as Index;
        Ok(g_indices
            .iter()
            .map(|&g| {
                state
                    .backsolver
                    .full_g_to_c_block(g)
                    .map(|pos| y_c_offset + pos)
            })
            .collect())
    }

    /// Reduced Hessian `H_R = obj_scal · B K⁻¹ Bᵀ` over the pinned
    /// equality-constraint rows, where `B` selects the
    /// `pin_constraint_indices` rows of the y_c block and `K` is the
    /// **natural-units** (unscaled) KKT matrix — active NLP scaling
    /// is undone by the backsolver, so `−inv(H_R)` is directly the
    /// parameter covariance regardless of `nlp_scaling_method`
    /// (pounce#128). `obj_scal` survives as a plain extra multiplier
    /// (default 1.0); it is no longer needed to recover natural units.
    /// Returns the `n²`-long column-major dense matrix
    /// (`n = pin_constraint_indices.len()`).
    ///
    /// Equivalent to [`crate::SensSolve::with_reduced_hessian`] but
    /// usable post-hoc on a held `Solver`. For the solver-space
    /// (pre-#128) value use [`Self::compute_reduced_hessian_scaled`];
    /// the factors themselves are exposed via [`Self::nlp_scaling`] /
    /// [`Self::pin_g_scaling`].
    pub fn compute_reduced_hessian(
        &self,
        pin_constraint_indices: &[Index],
        obj_scal: Number,
    ) -> Result<Vec<Number>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let n = pin_constraint_indices.len();
        let param_rows = state
            .backsolver
            .map_pin_g_to_kkt_rows(pin_constraint_indices)
            .map_err(SolverError::SensComputationFailed)?;
        let signs = vec![1; n];
        let a_data = IndexSchurData::from_parts(param_rows, signs)
            .map_err(|e| SolverError::SensComputationFailed(format!("{e:?}")))?;
        let opts = SensOptions {
            compute_red_hessian: true,
            obj_scal,
            ..SensOptions::default()
        };
        let mut sens_app = SensApplication::new(a_data, state.backsolver.clone(), opts);
        let mut hr = vec![0.0; n * n];
        if !sens_app.compute_reduced_hessian(&mut hr) {
            return Err(SolverError::SensComputationFailed(
                "SensApplication::compute_reduced_hessian failed".into(),
            ));
        }
        Ok(hr)
    }

    /// The reduced Hessian as the solver's internal **scaled** space
    /// sees it — the value [`Self::compute_reduced_hessian`] returned
    /// before pounce#128: `H̃_ij = (df / (dc_i·dc_j)) · H_ij`.
    /// Identical to `compute_reduced_hessian` when no NLP scaling is
    /// active.
    pub fn compute_reduced_hessian_scaled(
        &self,
        pin_constraint_indices: &[Index],
        obj_scal: Number,
    ) -> Result<Vec<Number>, SolverError> {
        let mut hr = self.compute_reduced_hessian(pin_constraint_indices, obj_scal)?;
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let df = state.backsolver.obj_scaling_factor();
        let dc = state
            .backsolver
            .pin_c_scales(pin_constraint_indices)
            .map_err(SolverError::SensComputationFailed)?;
        crate::reduced_hessian::scale_to_solver_space(&mut hr, df, &dc);
        Ok(hr)
    }

    /// Effective NLP scaling the IPM applied on the most recent
    /// converged solve: `(obj_scaling_factor, c_scale, d_scale)`.
    /// `(1.0, None, None)` ⇔ no scaling was active. The vectors are
    /// per-row factors over the algorithm's equality (`c`) and
    /// inequality (`d`) blocks.
    pub fn nlp_scaling(
        &self,
    ) -> Result<(Number, Option<Vec<Number>>, Option<Vec<Number>>), SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        Ok(state.backsolver.nlp_scaling())
    }

    /// Inertia-correction perturbations `(δ_x, δ_s, δ_c, δ_d)` baked
    /// into the held KKT factor. All zero ⇔ the final factorization
    /// was unregularized and the natural-units back-solves invert the
    /// exact KKT matrix — see
    /// [`crate::PdSensBacksolver::kkt_perturbations`].
    pub fn kkt_perturbations(&self) -> Result<[Number; 4], SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        Ok(state.backsolver.kkt_perturbations())
    }

    /// Per-pin equality-row scaling factors `dc_i` (1.0 entries when
    /// no constraint scaling is active), ordered like
    /// `pin_constraint_indices`.
    pub fn pin_g_scaling(
        &self,
        pin_constraint_indices: &[Index],
    ) -> Result<Vec<Number>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        state
            .backsolver
            .pin_c_scales(pin_constraint_indices)
            .map_err(SolverError::SensComputationFailed)
    }
}
