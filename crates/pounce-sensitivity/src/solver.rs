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
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;

use crate::PdSensBacksolver;
use crate::activity::ActivityReport;
use crate::backsolver::SensBacksolver;
use crate::schur_data::IndexSchurData;
use crate::sens_app::{SensApplication, SensOptions};
use crate::vec_util::dense_to_vec;

/// Sign of the barrier correction term, set from a comparison
/// against sIPOPT rather than derived.
pub const BARRIER_SIGN: Number = -1.0;

/// The bound geometry the bound-aware parametric steps share. See
/// [`Solver::bound_context`].
struct BoundContext {
    /// Length of the primal block.
    n_x: usize,
    /// Lower bounds over the primal block, in the model's own units.
    lo: Vec<Number>,
    /// Upper bounds, likewise.
    hi: Vec<Number>,
    /// The converged primal point, truncated to the primal block.
    x_curr: Vec<Number>,
    /// How far outside a bound still counts as on it.
    eps: Number,
    /// How far negative a bound multiplier has to go before its bound
    /// is released. Always the solve's own margin, whatever `eps` is.
    release_eps: Number,
    /// Bound multipliers at the base point, in the solve's own
    /// coordinates, with the compound row each occupies.
    mults: Vec<crate::boundcheck::BoundMultiplier>,
}

/// Errors returned by post-convergence operations on [`Solver`].
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// An option the requested computation depends on holds an
    /// incompatible value; the message names the option and the value
    /// required.
    BadOptions(String),
}

/// State captured at convergence: the user-visible iterate plus the
/// `PdSensBacksolver` that wraps the converged KKT factor.
///
/// Read this via [`Solver::converged`].
pub struct ConvergedState {
    /// IPM return status of the most recent solve.
    pub status: ApplicationReturnStatus,
    /// Final primal iterate `x*` (length `n_x`), in the user's own
    /// units: a `user-scaling` change of variables is undone here, so
    /// this is `x`, never the algorithm's `x̃ = d ⊙ x` (gh#486).
    pub x: Vec<Number>,
    /// Final objective value `f(x*)`.
    pub obj_val: Number,
    /// `bound_relax_factor` **as the solve that produced this state
    /// ran with it**, not as the application's options read today.
    /// The bounds were relaxed (or not) once, during this solve; a
    /// later `set_numeric_value` cannot change what the held slacks
    /// were measured against, so post-solve calls whose validity
    /// depends on unrelaxed bounds must guard on this value. See
    /// [`Solver::classify_activity`].
    pub bound_relax_factor: Number,
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

        // Snapshot the options this solve will run under, before it
        // runs. `bound_relax_factor` is consumed once, when the NLP
        // relaxes its bounds; reading it back at query time would
        // describe the application's options rather than the state
        // being queried. The registry supplies its own default when
        // the option is unset, so no second copy of the default lives
        // here.
        let brf = self
            .app
            .options()
            .get_numeric_value("bound_relax_factor", "")
            .map(|(v, _)| v)
            .expect("bound_relax_factor is a registered core option");

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
                // The algorithm's iterate is `x̃ = d ⊙ x` when the
                // solve ran under a change of variables (gh#486): this
                // capture reads the iterate, not the
                // `finalize_solution` payload, so it undoes the
                // substitution itself. The backsolver already read the
                // factors off the NLP, in this same var-x space.
                let mut x = dense_to_vec(&*curr.x);
                if let Some(d) = backsolver.variable_scaling() {
                    debug_assert_eq!(x.len(), d.len());
                    for (xi, &di) in x.iter_mut().zip(d.iter()) {
                        *xi /= di;
                    }
                }
                let obj_val = cq.borrow_mut().curr_f();
                // Status is overwritten with the real value after
                // optimize_tnlp returns.
                *state_cb.borrow_mut() = Some(ConvergedState {
                    status: ApplicationReturnStatus::InternalError,
                    x,
                    obj_val,
                    bound_relax_factor: brf,
                    backsolver,
                });
            }));

        let status = crate::optimize_tnlp_for_sensitivity(&mut self.app, Rc::clone(&self.tnlp));
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

    /// Classify every bounded variable and every finite-bounded
    /// inequality row of the converged solve by activity: see
    /// [`crate::activity`] and
    /// `dev-notes/covariance-information-roadmap.md` item 0 (gh #362).
    ///
    /// Requires the held solve to have run with `bound_relax_factor=0`
    /// (the Ipopt default is `1e-8`): with relaxed bounds the solver's
    /// slacks are measured against perturbed bounds, and the
    /// complementarity products the classifier reads no longer track
    /// `μ`.
    ///
    /// The guard reads
    /// [`ConvergedState::bound_relax_factor`] — the value that solve
    /// ran under — not the application's current options. Setting the
    /// option after the fact neither unlocks a state whose bounds were
    /// relaxed nor invalidates one whose bounds were not; re-solve to
    /// change the answer.
    pub fn classify_activity(&self) -> Result<ActivityReport, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let brf = state.bound_relax_factor;
        if brf != 0.0 {
            return Err(SolverError::BadOptions(format!(
                "classify_activity requires bound_relax_factor=0, but the \
                 held solve ran with {brf:e}: relaxed bounds shift the \
                 slacks the classifier reads. Set the option and solve() \
                 again — changing it now does not re-measure the slacks."
            )));
        }
        Ok(crate::activity::compute(&state.backsolver))
    }

    /// The gradient of user constraint row `user_row` at the converged
    /// iterate, in user variable order (length `n_full_x`) and in
    /// **natural (unscaled) units**: the internal Jacobian row carries
    /// the solver's per-row `c_scale`/`d_scale`, which is divided out
    /// here, so this is the gradient of the row as the user wrote it.
    /// Equality and inequality rows alike; entries for fixed
    /// (`make_parameter`-removed) variables are 0 because the solve
    /// dropped their columns. Errors on an out-of-range row.
    ///
    /// Serves the covariance roadmap's item 1: a binding row's normal
    /// restricted to the fitted block is the projection direction.
    pub fn row_normal(&self, user_row: usize) -> Result<Vec<Number>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        crate::activity::row_normal(&state.backsolver, user_row).map_err(|m| {
            SolverError::BadShape {
                what: "row_normal constraint index",
                got: user_row,
                expected: m,
            }
        })
    }

    /// The exact Lagrangian Hessian times a user-space vector, in
    /// user variable order and natural units (see
    /// [`crate::activity::hessian_vec`]). Errors on a length mismatch.
    pub fn hessian_vec(&self, v: &[Number]) -> Result<Vec<Number>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        crate::activity::hessian_vec(&state.backsolver, v).map_err(|n| SolverError::BadShape {
            what: "hessian_vec vector length",
            got: v.len(),
            expected: n,
        })
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
    /// when no NLP scaling is active. "Scaled space" includes a
    /// `user-scaling` change of variables (gh#486), so on such a solve
    /// the `x` and `z` blocks here are in the substituted coordinates
    /// `x̃ = d ⊙ x`, not the model's.
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
        // carry the step from the barrier problem's solution toward the
        // original problem's (the paper's equation 11)
        let corr = self.barrier_correction(state)?;
        for (d, c) in dx_full.iter_mut().zip(corr.iter()) {
            *d += *c * BARRIER_SIGN;
        }
        dx_full.truncate(n_x);
        Ok(dx_full)
        // NOTE: parametric_step_full below applies the same correction,
        // so the two agree on their shared block.
    }

    /// The right-hand side [`Self::parametric_step_full`] answers,
    /// barrier term included. That method adds the term as a correction
    /// to the solution rather than to the right-hand side, which is the
    /// same thing by linearity.
    ///
    /// The parameter rows go through `map_pin_g_to_kkt_rows` exactly as
    /// they do there. Passing the constraint indices raw instead puts
    /// the perturbation on the x rows, where it contributes nothing --
    /// a release then sees only its own multiplier shift and lands on
    /// the wrong answer without failing.
    fn parametric_rhs_full(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
    ) -> Result<Vec<Number>, SolverError> {
        let state = self.converged().ok_or(SolverError::NotConverged)?;
        let state = &*state;
        let dims = state.backsolver.block_dims();
        let n_full = state.backsolver.dim();
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
        let mut rhs = vec![0.0; n_full];
        if !sens_app.parametric_rhs(deltas, &mut rhs) {
            return Err(SolverError::SensComputationFailed(
                "SensApplication::parametric_rhs failed".into(),
            ));
        }
        let mu = state.backsolver.barrier_mu();
        let start = dims[0] + dims[1] + dims[2] + dims[3];
        let end = start + dims[4] + dims[5] + dims[6] + dims[7];
        for r in rhs.iter_mut().take(end).skip(start) {
            *r += mu * BARRIER_SIGN;
        }
        Ok(rhs)
    }

    /// The barrier correction of the parametric step: the paper's
    /// equation 11 term, which carries the step from the solution of
    /// the barrier problem at `mu > 0` toward the one at `mu = 0`.
    ///
    /// [`Self::parametric_step`] is taken against a factorization held
    /// at the final `mu`, so it estimates where the BARRIER problem's
    /// solution moves, not where the original problem's does. The two
    /// differ by `O(mu)`, which is negligible at a tight tolerance and
    /// is not at a loose one. Measured against sIPOPT on a nonlinear
    /// model, the uncorrected step agrees to 2e-9 at `tol = 1e-8` and
    /// differs by 9e-6 at `tol = 1e-3`.
    ///
    /// The term is one more backsolve against the same factor, with
    /// `mu` in the complementarity rows, which are the bound multiplier
    /// blocks of the compound vector.
    ///
    /// Returns the correction over the whole compound vector, to be
    /// added to the step.
    fn barrier_correction(&self, state: &ConvergedState) -> Result<Vec<Number>, SolverError> {
        let dims = state.backsolver.block_dims();
        let n_full = state.backsolver.dim();
        let mu = state.backsolver.barrier_mu();
        // z_l, z_u, v_l, v_u: the rows carrying the complementarity
        // conditions, which are the ones the barrier perturbs
        let start = dims[0] + dims[1] + dims[2] + dims[3];
        let end = start + dims[4] + dims[5] + dims[6] + dims[7];
        let mut rhs = vec![0.0; n_full];
        for r in rhs.iter_mut().take(end).skip(start) {
            *r = mu;
        }
        let mut corr = vec![0.0; n_full];
        if !state.backsolver.solve(&rhs, &mut corr) {
            return Err(SolverError::BacksolveFailed);
        }
        Ok(corr)
    }

    /// Parametric step with the bounds respected by pinning, not by
    /// clamping. Returns the `n_x`-long primal step, the rows it
    /// constrained to reach it, and why the refinement stopped.
    ///
    /// [`Self::parametric_step`] answers where the linear predictor
    /// points, which can be outside the box. Clamping a coordinate
    /// back to its bound leaves every other coordinate at its
    /// predictor value, so the answer is feasible but no longer
    /// consistent with the KKT relations. This instead adds a row
    /// pinning each offending coordinate at its bound and re-solves, so
    /// the others move to stay consistent under the pins, which is the
    /// refinement upstream runs under `sens_boundcheck`.
    ///
    /// A pass takes every crossing it can see, pins them together, and
    /// re-solves, so the loop ends when nothing is left outside rather
    /// than when the passes run out. Each pass rebuilds the Schur
    /// complement over the pins so far, so a pass carrying `k` of them
    /// costs one dense `k × k` solve and `k + 1` back-solves; the
    /// factorization itself is never rebuilt for a pin.
    ///
    /// What counts as outside a bound is the `eps` argument when the
    /// caller passes one, and the solve's own margin when it passes
    /// `None`: the solve was willing to leave a converged point
    /// `bound_relax_factor` outside its bound, so anything within that
    /// is on the bound. An unrelaxed solve gets a roundoff floor.
    ///
    /// Passes stop when nothing is outside its bound by that much, when
    /// a pin cannot be achieved because the pins have exhausted the
    /// problem's degrees of freedom, or at `max_iter`, which is a
    /// safety limit rather than a budget: it took one pin per pass
    /// until gh#732, where a model with more crossings than passes had
    /// its answer picked by the limit. None of those is an error, and
    /// the returned [`crate::boundcheck::RefineStop`] says which
    /// happened.
    pub fn parametric_step_bounded(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_iter: usize,
        bound_eps: Option<Number>,
    ) -> Result<(Vec<Number>, Vec<Index>, crate::boundcheck::RefineStop), SolverError> {
        let dx_full = self.parametric_step_full(pin_constraint_indices, deltas)?;
        let rhs_plain = self.parametric_rhs_full(pin_constraint_indices, deltas)?;
        let ctx = self.bound_context(bound_eps)?;
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let (dx, pinned, stop) = crate::boundcheck::refine_step_onto_bounds(
            &state.backsolver,
            &dx_full,
            &ctx.x_curr,
            &ctx.lo,
            &ctx.hi,
            &ctx.mults,
            &rhs_plain,
            ctx.eps,
            ctx.release_eps,
            max_iter,
        )
        .map_err(SolverError::SensComputationFailed)?;
        Ok((
            dx[..ctx.n_x].to_vec(),
            pinned.into_iter().map(|p| p as Index).collect(),
            stop,
        ))
    }

    /// Parametric step applied a little at a time instead of taken
    /// whole, stopping wherever the active set changes and continuing
    /// from there under the new one. Returns the primal step and the
    /// breakpoints crossed.
    ///
    /// [`Self::parametric_step_bounded`] decides every condition at the
    /// base point, which is upstream's fix-relax. This is past it: the
    /// result is piecewise linear in the parameter, exact for a QP
    /// because a QP's solution is piecewise affine in the parameter,
    /// and still a predictor for an NLP because nothing is
    /// re-linearized between breakpoints.
    ///
    /// `max_iter` caps the breakpoints crossed. It is in practice a
    /// budget on factorizations, since a pin is a back-solve against
    /// the held factor while a release re-factors.
    pub fn parametric_step_path(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_iter: usize,
    ) -> Result<(Vec<Number>, Vec<crate::boundcheck::PathSegment>), SolverError> {
        let rhs_plain = self.parametric_rhs_full(pin_constraint_indices, deltas)?;
        let ctx = self.bound_context(None)?;
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let (dx, segments) = crate::boundcheck::step_along_path(
            &state.backsolver,
            &rhs_plain,
            &ctx.x_curr,
            &ctx.lo,
            &ctx.hi,
            &ctx.mults,
            max_iter,
            &[],
            &[],
        )
        .map_err(SolverError::SensComputationFailed)?;
        Ok((dx[..ctx.n_x].to_vec(), segments))
    }

    /// [`Self::parametric_step_path`] with the weak-row
    /// decision supplied by the caller instead of searched for.
    /// `held_var_rows` names the var-x rows of the weakly active
    /// bounds the direction holds; every other weakly active bound is
    /// forced into the walk's base-activity table as a leaving row.
    /// Study surface for an externally solved eq. 14 QP.
    pub fn parametric_step_path_decided(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_iter: usize,
        held_var_rows: &[Index],
    ) -> Result<(Vec<Number>, Vec<crate::boundcheck::PathSegment>), SolverError> {
        let weak = self.weakly_active_bounds()?;
        if weak.is_empty() {
            return self.parametric_step_path(pin_constraint_indices, deltas, max_iter);
        }
        let mut rhs_plain = self.parametric_rhs_full(pin_constraint_indices, deltas)?;
        for w in &weak {
            rhs_plain[w.row] = 0.0;
        }
        let held: std::collections::HashSet<usize> =
            held_var_rows.iter().map(|&r| r as usize).collect();
        let ctx = self.bound_context(None)?;
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let holds: Vec<(usize, bool)> = weak
            .iter()
            .filter(|w| held.contains(&w.var_row))
            .map(|w| (w.var_row, w.lower))
            .collect();
        let forced_active: Vec<usize> = weak
            .iter()
            .filter(|w| !held.contains(&w.var_row))
            .map(|w| w.row)
            .collect();
        let (dx, segments) = crate::boundcheck::step_along_path(
            &state.backsolver,
            &rhs_plain,
            &ctx.x_curr,
            &ctx.lo,
            &ctx.hi,
            &ctx.mults,
            max_iter,
            &forced_active,
            &holds,
        )
        .map_err(SolverError::SensComputationFailed)?;
        Ok((dx[..ctx.n_x].to_vec(), segments))
    }

    /// Newton iterations on the barrier system, refining a step that
    /// some mode already produced.
    ///
    /// `step` is a full compound step, the shape
    /// [`Self::parametric_step_full`] returns, so any mode's result
    /// can be handed in. Each iteration costs one back-solve against
    /// the held factor and no factorization. Returns the refined step
    /// and a [`CorrectorReport`] saying what the iterations bought.
    ///
    /// The corrector aims at the barrier solution at the μ the solve
    /// finished on, not at a re-solve, so the accuracy it can reach is
    /// bounded by that offset. Where the perturbation needs a bound
    /// the base point held tightly to leave the active set, the held
    /// barrier diagonal cannot represent the change and the iterations
    /// make no progress. `CorrectorReport::improved` reports that
    /// case: the step handed back is then the caller's own.
    ///
    /// The returned point always satisfies the variable bounds, since
    /// the barrier residual is undefined outside them and the
    /// fraction-to-boundary rule keeps every iterate inside. A step
    /// that arrives pointing out of the box is therefore put back in
    /// before the first iteration, which means `max_iter = 0` is not a
    /// no-op: it costs one evaluation, no back-solve, and reports the
    /// residual the caller's step leaves.
    pub fn correct_step(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        step: &[Number],
        max_iter: usize,
    ) -> Result<(Vec<Number>, crate::corrector::CorrectorReport), SolverError> {
        let ctx = self.bound_context(None)?;
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let bs = &state.backsolver;
        let dim = bs.dim();
        if step.len() != dim {
            return Err(SolverError::BadShape {
                what: "step",
                got: step.len(),
                expected: dim,
            });
        }
        // `>= 0`, not `> 0`: `barrier_mu` reports exactly zero for a
        // point whose bound multipliers were zeroed on the way out (see
        // its doc comment), and that is a barrier level, not a missing
        // one. The complementarity rows are then already satisfied
        // where they stand, which is what the corrector should measure.
        let mu = {
            let m = bs.barrier_mu();
            if m >= 0.0 && m.is_finite() {
                m
            } else {
                return Err(SolverError::SensComputationFailed(
                    "corrector: the solve reported no barrier parameter".into(),
                ));
            }
        };
        let base = {
            let mut flat = vec![0.0; dim];
            bs.curr_flat(&mut flat).map_err(|_| {
                SolverError::SensComputationFailed(
                    "corrector: converged iterate unavailable".into(),
                )
            })?;
            flat
        };
        // The pinned equalities' KKT rows and the row scales the
        // algorithm applied to them. A user `g` index is not the KKT
        // row: the two differ once an inequality precedes the pin in
        // `g(x)` (pounce#128), and the residual the corrector measures
        // sits in the algorithm's scaled equality block, so the deltas
        // have to carry the same factors.
        let (pin_rows, pin_scales) = bs
            .pin_rows_and_c_scales(pin_constraint_indices)
            .map_err(SolverError::SensComputationFailed)?;
        let pin_rows: Vec<usize> = pin_rows.iter().map(|&r| r as usize).collect();
        let scaled_deltas: Vec<Number> = deltas
            .iter()
            .zip(&pin_scales)
            .map(|(&d, &c)| d * c)
            .collect();
        crate::corrector::run(
            bs,
            &base,
            step,
            &pin_rows,
            &scaled_deltas,
            &ctx.lo,
            &ctx.hi,
            mu,
            max_iter,
        )
    }

    /// [`Self::parametric_step_bounded`] with the weak-row
    /// decision supplied by the caller instead of searched for. The
    /// direction is computed for the given working set (all weak rows
    /// released, the held variables pinned through Schur rows), then
    /// refined onto the bounds exactly as the searched variant does.
    /// Study surface for an externally solved eq. 14 QP.
    pub fn parametric_step_bounded_decided(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_iter: usize,
        held_var_rows: &[Index],
        bound_eps: Option<Number>,
    ) -> Result<(Vec<Number>, Vec<Index>, crate::boundcheck::RefineStop), SolverError> {
        let weak = self.weakly_active_bounds()?;
        if weak.is_empty() {
            return self.parametric_step_bounded(
                pin_constraint_indices,
                deltas,
                max_iter,
                bound_eps,
            );
        }
        let rhs_plain = self.parametric_rhs_full(pin_constraint_indices, deltas)?;
        let ctx = self.bound_context(bound_eps)?;
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let released: Vec<usize> = weak.iter().map(|w| w.row).collect();
        let pinned_rows: Vec<usize> = held_var_rows.iter().map(|&r| r as usize).collect();
        let (d, _) = crate::boundcheck::path_direction(
            &state.backsolver,
            &rhs_plain,
            &released,
            &pinned_rows,
        )
        .map_err(SolverError::SensComputationFailed)?;
        let (dx, pinned, stop) = crate::boundcheck::refine_step_onto_bounds(
            &state.backsolver,
            &d,
            &ctx.x_curr,
            &ctx.lo,
            &ctx.hi,
            &ctx.mults,
            &rhs_plain,
            ctx.eps,
            ctx.release_eps,
            max_iter,
        )
        .map_err(SolverError::SensComputationFailed)?;
        Ok((
            dx[..ctx.n_x].to_vec(),
            pinned.into_iter().map(|p| p as Index).collect(),
            stop,
        ))
    }

    /// The eq. 14 directional derivative, decided by pounce-qp over
    /// the weak rows the direction engages.
    ///
    /// One released factorization serves the whole decision: the
    /// released `Σ` is built once and every solve passes the same
    /// object, so the factorization cache reuses the factor across the
    /// all-released direction and the basis columns. The decision
    /// itself is the dual of eq. 14 restricted to the weak rows the
    /// direction engages: with `a_k` the signed unit vector of weak
    /// row `k` (positive for a lower bound), `X_k = K_rel^{-1} a_k`,
    /// `S = aᵀX` and `m = aᵀd0`, the pin forces `λ` solve
    ///
    /// ```text
    ///     min  ½ λᵀ S λ + mᵀ λ    s.t.  λ ≥ 0
    /// ```
    ///
    /// whose KKT conditions are eq. 14's complementarity: a released
    /// row moves to its feasible side (the QP gradient `Sλ + m ≥ 0`)
    /// and a held row's pin force is nonnegative. Rows outside the
    /// engaged set are verified against the decided direction and the
    /// set expands until no new row violates, so rows the equalities
    /// already pin (movement exactly zero under every candidate)
    /// never enter and never make anything singular.
    ///
    /// `max_iter` is the total back-solve budget: the all-released
    /// solve, every basis column, and the combined solve that recovers
    /// the direction all count against it. A budget of zero errs
    /// before any work. Any budget above that pays the all-released
    /// factorization first, because which rows engage is only known
    /// once that solve has run, and the shortfall is reported when the
    /// basis columns cannot fit. Either way the caller falls back to
    /// the one-sided step. Returns the direction, the var-x rows held,
    /// and the back-solves spent.
    pub fn parametric_step_directional(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_iter: usize,
    ) -> Result<(Vec<Number>, Vec<usize>, usize), SolverError> {
        use pounce_common::types::NLP_UPPER_BOUND_INF;
        use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
        use pounce_qp::QpStatus;
        use pounce_qp::options::QpOptions;
        use pounce_qp::problem::{HessianInertia, QpProblem};
        use pounce_qp::solver::{ParametricActiveSetSolver, QpSolver};

        const EPS_REL: Number = 1e-9;

        let rhs_plain = self.parametric_rhs_full(pin_constraint_indices, deltas)?;
        let weak = self.weakly_active_bounds()?;
        let ctx = self.bound_context(None)?;
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let bs = &state.backsolver;
        let dim = bs.dim();
        let n_x = ctx.n_x;
        let nw = weak.len();
        let mut work = 0usize;
        // A weak bound's slack and multiplier are both of order
        // sqrt(mu) and their uncertainty equals their magnitude, so a
        // movement below sqrt(mu) of the direction's scale cannot be
        // resolved against the bound and does not warrant an exact
        // complementarity decision. The engagement and expansion
        // tests use this band; acceptance-level roundoff tests keep
        // EPS_REL.
        let band = bs.barrier_mu().max(0.0).sqrt().max(EPS_REL);

        if weak.is_empty() {
            // a clean base point takes the plain step and no decision
            // happens, so the reported decision work is zero
            let mut d = vec![0.0; dim];
            if !bs.solve(&rhs_plain, &mut d) {
                return Err(SolverError::BacksolveFailed);
            }
            return Ok((d[..n_x].to_vec(), Vec::new(), 0));
        }

        // What the caller needs is the number to raise
        // `degeneracy_iter` to, so the message reports the engaged
        // count rather than the weak-set size: engagement is the retry
        // price, and on a model with hundreds of weak bounds the two
        // differ by enough that raising one at a time is dozens of
        // retries. The engaged set can still grow on a later pass, so
        // the figure is a floor and says so.
        //
        // `engaged_now + 2` prices a decision that finishes on one
        // pass. Each expansion round pays another combined solve, so
        // on a multi-pass decision that total is short, and once the
        // engaged set stops growing it stops moving at all: the
        // combined solve of the last round would otherwise be told to
        // raise the budget to the number already spent, which is a
        // retry that buys nothing and reads as self-contradictory.
        // Flooring at `spent + 1` keeps the advice strictly larger
        // than what is gone, so every retry makes progress.
        let budget = |engaged_now: usize, spent: usize| {
            let need = (engaged_now + 2).max(spent + 1);
            SolverError::SensComputationFailed(format!(
                "directional derivative: {spent} of {max_iter} back-solve(s) \
                 spent, and {engaged_now} of {nw} weakly active bound(s) are \
                 engaged so far. Raise degeneracy_iter to at least {need}; \
                 the engaged set can still grow, so that is a floor."
            ))
        };
        let fail = |what: &str| {
            SolverError::SensComputationFailed(format!("directional derivative: {what}"))
        };

        let released: Vec<usize> = weak.iter().map(|w| w.row).collect();
        let sigma = bs
            .released_sigma_x(&released)
            .ok_or_else(|| fail("released sigma unavailable"))?;
        if work + 1 > max_iter {
            // Nothing is engaged before the all-released solve, so
            // this fires only at a budget of zero, and the floor is
            // the one solve the decision cannot start without.
            return Err(SolverError::SensComputationFailed(format!(
                "directional derivative: degeneracy_iter is {max_iter}, and the \
                 decision cannot start without one back-solve over the {nw} \
                 weakly active bound(s). Raise degeneracy_iter to at least 2."
            )));
        }
        let mut d0 = vec![0.0; dim];
        // shift = false, matching `path_direction`'s all-released
        // solve: a weak bound's multiplier is order sqrt(mu) and the
        // released convention holds it at exactly zero, so the step
        // shift's multiplier injection is deliberately omitted.
        if !bs.solve_released_prebuilt(&released, Rc::clone(&sigma), &rhs_plain, &mut d0, false) {
            return Err(SolverError::BacksolveFailed);
        }
        work += 1;

        // movement of weak row k under a direction: positive is the
        // feasible side for that row's bound
        let sign = |k: usize| if weak[k].lower { 1.0 } else { -1.0 };
        let movement = |k: usize, d: &[Number]| -> Number { sign(k) * d[weak[k].var_row] };
        let scale_of = |d: &[Number]| -> Number {
            d[..n_x]
                .iter()
                .fold(0.0_f64, |a, &b| a.max(b.abs()))
                .max(1e-300)
        };

        let tol0 = band * scale_of(&d0);
        let mut engaged: Vec<usize> = (0..nw).filter(|&k| movement(k, &d0) < -tol0).collect();
        if engaged.is_empty() {
            return Ok((d0[..n_x].to_vec(), Vec::new(), work));
        }

        // Each basis column is only ever read at the weak rows' own
        // variables, once to build `S` and never again: the direction
        // it contributes is recovered below in a single solve. So the
        // column is projected onto those `nw` entries and the
        // full-length vector dropped, which bounds this by the weak
        // set rather than by `dim` times the budget. Holding the full
        // columns costs about 114 MB on a 62k model at 230 engaged
        // rows, and grows with `degeneracy_iter`.
        let mut proj: Vec<Option<Vec<Number>>> = vec![None; nw];
        let mut d = d0.clone();
        let held: Vec<usize>;
        loop {
            for &k in &engaged {
                if proj[k].is_some() {
                    continue;
                }
                if work + 1 > max_iter {
                    return Err(budget(engaged.len(), work));
                }
                let mut unit = vec![0.0; dim];
                unit[weak[k].var_row] = sign(k);
                let mut xk = vec![0.0; dim];
                if !bs.solve_released_prebuilt(&released, Rc::clone(&sigma), &unit, &mut xk, false)
                {
                    return Err(SolverError::BacksolveFailed);
                }
                work += 1;
                proj[k] = Some(weak.iter().map(|w| xk[w.var_row]).collect());
            }

            // dense reduced data over the engaged rows, upper triangle
            let ke = engaged.len();
            let mut irows = Vec::new();
            let mut jcols = Vec::new();
            let mut vals = Vec::new();
            for i in 0..ke {
                for j in i..ke {
                    let col_j = proj[engaged[j]].as_ref().expect("column built");
                    let col_i = proj[engaged[i]].as_ref().expect("column built");
                    // S_ij = a_i^T X_j; symmetrize, since S is
                    // symmetric in exact arithmetic. The projection
                    // holds one entry per weak row, so a weak row's
                    // own index is where its `a` picks the column out.
                    let s_ij = 0.5
                        * (sign(engaged[i]) * col_j[engaged[i]]
                            + sign(engaged[j]) * col_i[engaged[j]]);
                    // pounce-linalg triplets are one-based
                    irows.push((i + 1) as Index);
                    jcols.push((j + 1) as Index);
                    vals.push(s_ij);
                }
            }
            // The engine's feasibility and optimality tolerances are
            // absolute and act on the QP's variables, which are the
            // pin forces, so both sides of the problem are scaled to
            // order one: the gradient against the direction's scale
            // (a 1e-10 perturbation must decide the same way a 1e-2
            // one does) and S against its largest entry, which is a
            // compliance in the model's units. The joint scaling maps
            // the solution by g_scale / s_scale exactly, so the
            // scaled solve loses nothing.
            let g_raw: Vec<Number> = engaged.iter().map(|&k| movement(k, &d0)).collect();
            let g_scale = g_raw
                .iter()
                .fold(0.0_f64, |a, &b| a.max(b.abs()))
                .max(1e-300);
            let g: Vec<Number> = g_raw.iter().map(|&v| v / g_scale).collect();
            let s_scale = vals
                .iter()
                .fold(0.0_f64, |a, &b| a.max(b.abs()))
                .max(1e-300);
            let vals_scaled: Vec<Number> = vals.iter().map(|&v| v / s_scale).collect();
            let space = SymTMatrixSpace::new(ke as Index, irows, jcols);
            let mut h = SymTMatrix::new(space);
            h.set_values(&vals_scaled);
            let a_space = GenTMatrixSpace::new(0, ke as Index, Vec::new(), Vec::new());
            let a = GenTMatrix::new(a_space);
            let xl = vec![0.0; ke];
            let xu = vec![NLP_UPPER_BOUND_INF; ke];
            let qp = QpProblem {
                n: ke,
                m: 0,
                h: &h,
                g: &g,
                a: &a,
                bl: &[],
                bu: &[],
                xl: &xl,
                xu: &xu,
                hessian_inertia: HessianInertia::Unknown,
            };
            let opts = QpOptions {
                max_iter: (10 * ke as u32).max(200),
                // the engine's Schur-update path (use_schur_updates)
                // hits MaxIter on a dense reduced problem of hundreds
                // of rows where the refactorizing path terminates
                // Optimal, so the default stays; the heavy-direction
                // exact decision pays engine refactorizations and is
                // priced accordingly in the docs
                ..QpOptions::default()
            };
            let mut engine =
                ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
            let sol = engine
                .solve(&qp, None, &opts)
                .map_err(|e| fail(&format!("reduced QP failed: {e:?}")))?;
            if sol.status != QpStatus::Optimal {
                return Err(fail(&format!(
                    "reduced QP terminated {:?} over {ke} engaged row(s)",
                    sol.status
                )));
            }
            let lambda: Vec<Number> = sol.x.iter().map(|&v| v * (g_scale / s_scale)).collect();

            // plus, not minus: the QP's optimality gradient is
            // S lambda + m, so the direction's movement must be
            // m + lambda S, which is d0 + Σ λ_k X_k here.
            //
            // Each `X_k` is `K_rel⁻¹ a_k`, so that sum is
            // `K_rel⁻¹ (Σ λ_k a_k)` and one solve on the combined
            // right-hand side gives it. That is why the columns above
            // need not be kept: the only thing they were held for is
            // recovered here, in a single back-solve, at the price of
            // one more against the budget per expansion round.
            d.copy_from_slice(&d0);
            if lambda.iter().any(|&l| l != 0.0) {
                if work + 1 > max_iter {
                    return Err(budget(engaged.len(), work));
                }
                let mut comb = vec![0.0; dim];
                for (i, &k) in engaged.iter().enumerate() {
                    comb[weak[k].var_row] += lambda[i] * sign(k);
                }
                let mut corr = vec![0.0; dim];
                if !bs.solve_released_prebuilt(
                    &released,
                    Rc::clone(&sigma),
                    &comb,
                    &mut corr,
                    false,
                ) {
                    return Err(SolverError::BacksolveFailed);
                }
                work += 1;
                for (dv, &cv) in d.iter_mut().zip(corr.iter()) {
                    *dv += cv;
                }
            }

            let tol = band * scale_of(&d);
            let mut grew = false;
            for k in 0..nw {
                if engaged.contains(&k) {
                    continue;
                }
                if movement(k, &d) < -tol {
                    engaged.push(k);
                    grew = true;
                }
            }
            if !grew {
                // relative to the largest pin force, with no absolute
                // floor: a 1e-10-scale perturbation's pins are as real
                // as a 1e-2 one's, and a floor here silently unlabels
                // them while the direction still carries the pin
                let lam_scale = lambda
                    .iter()
                    .fold(0.0_f64, |a, &b| a.max(b.abs()))
                    .max(1e-300);
                held = engaged
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| lambda[*i] > EPS_REL * lam_scale)
                    .map(|(_, &k)| weak[k].var_row)
                    .collect();
                break;
            }
        }

        Ok((d[..n_x].to_vec(), held, work))
    }

    /// The bounds the activity classifier could not call at the base
    /// point: on the bound with a multiplier of the same order as the
    /// slack. Each entry is a bound row present in the held
    /// factorization, with the side taken from the smaller slack,
    /// which is the only side an ambiguous label can come from.
    ///
    /// The classifier reports per user variable, in full-x, while the
    /// bound context and the factor's rows are var-x, and the two
    /// index spaces diverge from the first fixed variable on. Each
    /// var-x row's status is read through the same map the classifier
    /// scattered through, so a fixed variable shifts nothing. Using
    /// the full-x index as a factor row instead returns a NEIGHBORING
    /// variable's answer, plausible and wrong, which is the gh#450
    /// hazard the `primal_row` discipline exists to prevent.
    pub fn weakly_active_bounds(&self) -> Result<Vec<crate::boundcheck::WeakBound>, SolverError> {
        use crate::activity::{AMBIGUOUS, WEAKLY_ACTIVE};

        // A relaxed solve shifts the slacks the classifier reads, so
        // degeneracy is undetectable there: the callers take the plain
        // step, the same choice `estimate_report` makes when it fills
        // `bounds_relaxed` instead of raising.
        let report = match self.classify_activity() {
            Ok(r) => r,
            Err(SolverError::BadOptions(_)) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let ctx = self.bound_context(None)?;
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let Some(rows) = state.backsolver.bound_rows() else {
            return Ok(Vec::new());
        };
        let full_of: Vec<usize> = {
            let (_, _, nlp) = state.backsolver.activity_handles();
            let nl = nlp.borrow();
            (0..ctx.n_x)
                .map(|r| nl.var_x_to_full_x(r as Index) as usize)
                .collect()
        };
        let mut out = Vec::new();
        for var_row in 0..ctx.n_x {
            let Some(&st) = report.var_status.get(full_of[var_row]) else {
                continue;
            };
            if st != WEAKLY_ACTIVE && st != AMBIGUOUS {
                continue;
            }
            let s_lo = ctx.x_curr[var_row] - ctx.lo[var_row];
            let s_hi = ctx.hi[var_row] - ctx.x_curr[var_row];
            let lower = s_lo <= s_hi;
            if let Some(br) = rows
                .iter()
                .find(|b| b.var_row == var_row && b.lower == lower)
            {
                out.push(crate::boundcheck::WeakBound {
                    row: br.row,
                    var_row,
                    lower,
                });
            }
        }
        Ok(out)
    }

    /// The bound geometry both bound-aware steps read: the primal
    /// block's size and base point, its bounds in the model's own
    /// units, the tolerance that decides what counts as on a bound, and
    /// the bound multipliers at the base point.
    ///
    /// Shared rather than assembled twice. The two callers have to
    /// agree on all of it, and the unit and index-space conversions
    /// below are exactly what went wrong when a second caller wrote its
    /// own.
    ///
    /// `bound_eps` overrides the margin. `None` keeps how far outside
    /// the solve itself was willing to settle, floored so an unrelaxed
    /// solve does not pin on roundoff.
    fn bound_context(&self, bound_eps: Option<Number>) -> Result<BoundContext, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let n_x = state.backsolver.block_dims()[0];

        // Expanded once, before any re-solve: reading the compressed
        // form means borrowing the NLP, and the solves below re-borrow
        // it.
        let (mut lo, mut hi) = {
            let (_, _, nlp) = state.backsolver.activity_handles();
            let nl = nlp.borrow();
            crate::boundcheck::expand_bounds(n_x, &nl.px_l(), &nl.px_u(), nl.x_l(), nl.x_u())
        };
        // Those bounds bound the algorithm's `x̃ = d ⊙ x`, while
        // `state.x` and the step are both in the model's own units
        // (gh#486 stage 3). Undo the change of variables on the bounds
        // so all three agree, rather than projecting onto the wrong box.
        // A negative factor reflects the interval, so the sides swap.
        // `variable_scaling`, not `variable_scaling_full`: `lo` / `hi`
        // are var-x length, and the two index spaces diverge from the
        // first fixed variable on.
        if let Some(d) = state.backsolver.variable_scaling() {
            for i in 0..n_x {
                let di = d[i];
                if di == 0.0 || di == 1.0 {
                    continue;
                }
                let (a, b) = (lo[i] / di, hi[i] / di);
                lo[i] = a.min(b);
                hi[i] = a.max(b);
            }
        }

        // What counts as outside a bound is the solve's own answer: it
        // was willing to leave a converged point `bound_relax_factor`
        // outside, so anything within that is on the bound, not past
        // it. A floor keeps an unrelaxed solve from pinning on
        // roundoff.
        let floor = state.bound_relax_factor.abs().max(1e-9);
        let eps = bound_eps.unwrap_or(floor);
        // A caller's `bound_eps` is a primal margin and says nothing
        // about when a multiplier has changed sign, so the release test
        // keeps the solve's own margin.
        let release_eps = floor;
        // The bound multipliers at the base point, with the compound
        // row each one occupies, so a step that drives one negative can
        // release that bound.
        let mults = {
            let dims = state.backsolver.block_dims();
            let (z_l_off, z_u_off) = (
                dims[0] + dims[1] + dims[2] + dims[3],
                dims[0] + dims[1] + dims[2] + dims[3] + dims[4],
            );
            let (data, _, _) = state.backsolver.activity_handles();
            let d = data.borrow();
            let curr = d.curr.as_ref().ok_or(SolverError::NotConverged)?;
            let mut out = Vec::new();
            for (off, v) in [(z_l_off, &curr.z_l), (z_u_off, &curr.z_u)] {
                for (k, &base) in crate::vec_util::dense_to_vec(&**v).iter().enumerate() {
                    out.push(crate::boundcheck::BoundMultiplier { row: off + k, base });
                }
            }
            out
        };
        Ok(BoundContext {
            n_x,
            lo,
            hi,
            x_curr: state.x[..n_x].to_vec(),
            eps,
            release_eps,
            mults,
        })
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
        let corr = self.barrier_correction(state)?;
        for (d, c) in dx_full.iter_mut().zip(corr.iter()) {
            *d += *c * BARRIER_SIGN;
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

    /// Flat rows of the compound KKT vector holding the primal values
    /// `x` for the given 0-based **full-x** variable indices. `None`
    /// where the solve removed the column (`x_l == x_u` under
    /// `fixed_variable_treatment = make_parameter`), which has no row
    /// in the factor at all.
    ///
    /// The `x` counterpart of [`Self::g_multiplier_rows`], and needed
    /// for the same reason: a caller holding user-space indices — from
    /// the `.col` file, from [`Self::classify_activity`], from
    /// [`Self::row_normal`] — cannot index the factor with them
    /// directly. Row `r` of a [`Self::parametric_step_full`] result is
    /// then `∂x/∂p · Δp` for that variable, and `e_r` is the unit
    /// vector selecting its column in a [`Self::kkt_solve`].
    pub fn x_primal_rows(&self, x_indices: &[Index]) -> Result<Vec<Option<Index>>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let n_full = state.backsolver.n_full_x();
        // out of range must not masquerade as "removed as fixed": the
        // NLP map returns None for both, and the caller's whole reason
        // for asking is that it cannot tell the spaces apart itself
        if let Some(&bad) = x_indices.iter().find(|&&i| i < 0 || i >= n_full) {
            return Err(SolverError::BadShape {
                what: "x_primal_rows variable index",
                got: bad as usize,
                expected: n_full as usize,
            });
        }
        // the x block starts at flat index 0, so the var-x position IS
        // the KKT row; the offset stays explicit for the day it is not
        Ok(x_indices
            .iter()
            .map(|&i| state.backsolver.full_x_to_var_x(i))
            .collect())
    }

    /// The user TNLP's variable count: the length of a full-x report
    /// and the domain of [`Self::x_primal_rows`].
    pub fn n_full_x(&self) -> Result<usize, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        Ok(state.backsolver.n_full_x() as usize)
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

    /// The per-variable `user-scaling` factors `d` the held solve ran
    /// under (gh#486), in the user TNLP's **full-x** space, or `None`
    /// when the solve applied no change of variables.
    ///
    /// Every accessor on this type already reports natural units, so
    /// this is diagnostic rather than a correction a caller has to
    /// apply — it answers "was this solve conditioned, and by how
    /// much", the x-axis counterpart of [`Self::nlp_scaling`].
    pub fn variable_scaling(&self) -> Result<Option<Vec<Number>>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        Ok(state.backsolver.variable_scaling_full().map(|d| d.to_vec()))
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
