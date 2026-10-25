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
//! # Where the calculations live
//!
//! Every calculation over the converged factor is
//! [`pounce_sens_core::converged::ConvergedSolve`]'s, written against the
//! `BarrierIterate` trait so it runs over any interior-point engine.
//! `Solver` runs pounce's own solve, holds its factor, and builds a
//! `ConvergedSolve` over it for each call. The two methods that read
//! pounce's own scaled space, [`Solver::kkt_solve_scaled`] and
//! [`Solver::variable_scaling`], stay here.

use std::cell::{Ref, RefCell};
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_sens_core::converged::ConvergedSolve;

use crate::PdSensBacksolver;
use crate::activity::{ActivityReport, ReducedActivityReport, ReducedRowActivityReport};
use crate::backsolver::SensBacksolver;
use crate::boundcheck::{PathOperator, PathSegment, RefineStop, WeakBound};
use crate::corrector::CorrectorReport;
use crate::vec_util::dense_to_vec;

pub use pounce_sens_core::converged::{BARRIER_SIGN, SolverError};

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
    /// Whether the solve computed exact Hessians, **as it ran**. A
    /// `limited-memory` solve's `IpoptData::w` is the quasi-Newton
    /// matrix, and there is no exact Hessian to evaluate at another
    /// point, so the corrector keeps that matrix instead of
    /// refreshing it at the predicted iterate.
    pub exact_hessian: bool,
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
        let exact_hessian = self
            .app
            .options()
            .get_string_value("hessian_approximation", "")
            .map(|(v, _)| v == "exact")
            .expect("hessian_approximation is a registered core option");

        // `set_on_converged` fires once per converged *attempt*, and when a
        // later attempt loses (the mu fallback, the gh#884 retry, the gh#983
        // scale audit) the reported answer is replayed from an earlier one.
        // Keeping only the last firing therefore held the DISCARDED attempt's
        // point and factor beside the winner's status. Every firing is kept
        // under its id until the solve says which one it reported; the KKT
        // state cannot be rewound, but each attempt's own handles keep it
        // alive.
        let attempts: Rc<RefCell<Vec<(u64, ConvergedState)>>> = Rc::new(RefCell::new(Vec::new()));
        let state_cb = Rc::clone(&attempts);
        self.app
            .set_on_converged(Box::new(move |data, cq, nlp, pd| {
                let (curr, id) = {
                    let d = data.borrow();
                    match (d.curr.clone(), d.on_converged_id) {
                        (Some(c), Some(id)) => (c, id),
                        _ => return,
                    }
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
                state_cb.borrow_mut().push((
                    id,
                    ConvergedState {
                        status: ApplicationReturnStatus::InternalError,
                        x,
                        obj_val,
                        bound_relax_factor: brf,
                        exact_hessian,
                        backsolver,
                    },
                ));
            }));

        let status = crate::optimize_tnlp_for_sensitivity(&mut self.app, Rc::clone(&self.tnlp));
        // Hold the firing whose attempt produced the reported answer; drop the
        // rest, which describe attempts the solver discarded.
        let reported = self.app.reported_on_converged_id();
        let held = attempts
            .borrow_mut()
            .drain(..)
            .find(|(id, _)| Some(*id) == reported)
            .map(|(_, mut s)| {
                s.status = status;
                s
            });
        *self.state.borrow_mut() = held;
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

    /// The calculations over the held factor, or
    /// [`SolverError::NotConverged`] when there is none.
    pub fn with_converged<R>(
        &self,
        f: impl FnOnce(&ConvergedSolve<'_, PdSensBacksolver>) -> Result<R, SolverError>,
    ) -> Result<R, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        f(&ConvergedSolve::new(
            &state.backsolver,
            state.bound_relax_factor,
            state.exact_hessian,
        ))
    }

    /// Classify every bounded variable and every finite-bounded
    /// inequality row of the converged solve by activity. See
    /// [`ConvergedSolve::classify_activity`].
    ///
    /// Requires the held solve to have run with `bound_relax_factor=0`.
    /// The guard reads [`ConvergedState::bound_relax_factor`], the value
    /// that solve ran under, not the application's current options.
    /// Setting the option after the fact neither unlocks a state whose
    /// bounds were relaxed nor invalidates one whose bounds were not.
    /// Re-solve to change the answer.
    pub fn classify_activity(&self) -> Result<ActivityReport, SolverError> {
        self.with_converged(|c| c.classify_activity())
    }

    /// The per-variable verdict for `user_vars`, re-measured against the
    /// reduced curvature along each coordinate. See
    /// [`ConvergedSolve::reduced_activity`].
    pub fn reduced_activity(
        &self,
        user_vars: &[usize],
    ) -> Result<ReducedActivityReport, SolverError> {
        self.with_converged(|c| c.reduced_activity(user_vars))
    }

    /// The per-row verdict for `user_rows`, re-measured against the
    /// reduced curvature along each row's gradient. See
    /// [`ConvergedSolve::reduced_row_activity`].
    pub fn reduced_row_activity(
        &self,
        user_rows: &[usize],
    ) -> Result<ReducedRowActivityReport, SolverError> {
        self.with_converged(|c| c.reduced_row_activity(user_rows))
    }

    /// The gradient of user constraint row `user_row`, in user variable
    /// order and natural units. See [`ConvergedSolve::row_normal`].
    pub fn row_normal(&self, user_row: usize) -> Result<Vec<Number>, SolverError> {
        self.with_converged(|c| c.row_normal(user_row))
    }

    /// The exact Lagrangian Hessian times a user-space vector, in natural
    /// units. See [`ConvergedSolve::hessian_vec`].
    pub fn hessian_vec(&self, v: &[Number]) -> Result<Vec<Number>, SolverError> {
        self.with_converged(|c| c.hessian_vec(v))
    }

    /// Solve `K · lhs = rhs` against the converged KKT factor. Both
    /// slices must have length `kkt_dim()`, and the layout is the flat
    /// `x || s || y_c || y_d || z_l || z_u || v_l || v_u` packing.
    ///
    /// `K` here is the **natural-units** (unscaled) KKT matrix: when
    /// the IPM solved with active NLP scaling, the backsolver scales
    /// the RHS/solution (all eight blocks, including the z/v
    /// bound-multiplier rows) so callers pass and receive data in the
    /// user's own units (pounce#128). See
    /// [`crate::PdSensBacksolver::solve`]. For the raw scaled-space
    /// back-solve use [`Self::kkt_solve_scaled`].
    pub fn kkt_solve(&self, rhs: &[Number], lhs: &mut [Number]) -> Result<(), SolverError> {
        self.with_converged(|c| c.kkt_solve(rhs, lhs))
    }

    /// [`Self::kkt_solve`] without the natural-units conjugation: the
    /// back-solve runs against the factor exactly as the IPM holds it
    /// (the solver's internal scaled space). Identical to `kkt_solve`
    /// when no NLP scaling is active. "Scaled space" includes a
    /// `user-scaling` change of variables (gh#486), so on such a solve
    /// the `x` and `z` blocks here are in the substituted coordinates
    /// `x̃ = d ⊙ x`, not the model's.
    pub fn kkt_solve_scaled(&self, rhs: &[Number], lhs: &mut [Number]) -> Result<(), SolverError> {
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
        if state.backsolver.solve_scaled_space(rhs, lhs) {
            Ok(())
        } else {
            Err(SolverError::BacksolveFailed)
        }
    }

    /// Batched-RHS back-solve. `rhs_flat` and `lhs_flat` are row-major
    /// `(n_rhs, kkt_dim)` buffers, each row solved against the same
    /// converged factor. Equivalent in result to looping
    /// [`Self::kkt_solve`] but reuses one `IteratesVector` for the RHS
    /// and one for the result across all `n_rhs` calls. See
    /// [`crate::algorithm_backsolver::PdSensBacksolver::solve_many`].
    pub fn kkt_solve_many(
        &self,
        rhs_flat: &[Number],
        lhs_flat: &mut [Number],
        n_rhs: usize,
    ) -> Result<(), SolverError> {
        self.with_converged(|c| c.kkt_solve_many(rhs_flat, lhs_flat, n_rhs))
    }

    /// [`Self::kkt_solve_many`] without the natural-units conjugation
    /// (the batched sibling of [`Self::kkt_solve_scaled`]).
    pub fn kkt_solve_many_scaled(
        &self,
        rhs_flat: &[Number],
        lhs_flat: &mut [Number],
        n_rhs: usize,
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
        if state
            .backsolver
            .solve_many_scaled_space(rhs_flat, lhs_flat, n_rhs)
        {
            Ok(())
        } else {
            Err(SolverError::BacksolveFailed)
        }
    }

    /// First-order parametric step `Δx ≈ ∂x*/∂p · Δp` for a set of
    /// pinned equality constraints. See
    /// [`ConvergedSolve::parametric_step`].
    pub fn parametric_step(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
    ) -> Result<Vec<Number>, SolverError> {
        check_deltas(pin_constraint_indices, deltas)?;
        self.with_converged(|c| c.parametric_step(pin_constraint_indices, deltas))
    }

    /// Parametric step with the bounds respected by pinning, not by
    /// clamping. See [`ConvergedSolve::parametric_step_bounded`].
    pub fn parametric_step_bounded(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_iter: usize,
        bound_eps: Option<Number>,
    ) -> Result<(Vec<Number>, Vec<Index>, RefineStop), SolverError> {
        self.with_converged(|c| {
            c.parametric_step_bounded(pin_constraint_indices, deltas, max_iter, bound_eps)
        })
    }

    /// Parametric step taken along the path of active-set changes. See
    /// [`ConvergedSolve::parametric_step_path`].
    pub fn parametric_step_path(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_iter: usize,
    ) -> Result<(Vec<Number>, Vec<PathSegment>), SolverError> {
        self.with_converged(|c| c.parametric_step_path(pin_constraint_indices, deltas, max_iter))
    }

    /// [`Self::parametric_step_path`] with the weak-row decision supplied
    /// by the caller. See [`ConvergedSolve::parametric_step_path_decided`].
    pub fn parametric_step_path_decided(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_iter: usize,
        held_var_rows: &[Index],
    ) -> Result<(Vec<Number>, Vec<PathSegment>), SolverError> {
        self.with_converged(|c| {
            c.parametric_step_path_decided(pin_constraint_indices, deltas, max_iter, held_var_rows)
        })
    }

    /// Newton iterations on the barrier system, refining a step that some
    /// mode already produced. See [`ConvergedSolve::correct_step`].
    pub fn correct_step(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        step: &[Number],
        max_iter: usize,
    ) -> Result<(Vec<Number>, CorrectorReport), SolverError> {
        self.with_converged(|c| c.correct_step(pin_constraint_indices, deltas, step, max_iter))
    }

    /// [`Self::parametric_step_bounded`] with the weak-row decision
    /// supplied by the caller. See
    /// [`ConvergedSolve::parametric_step_bounded_decided`].
    pub fn parametric_step_bounded_decided(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_iter: usize,
        held_var_rows: &[Index],
        bound_eps: Option<Number>,
    ) -> Result<(Vec<Number>, Vec<Index>, RefineStop), SolverError> {
        self.with_converged(|c| {
            c.parametric_step_bounded_decided(
                pin_constraint_indices,
                deltas,
                max_iter,
                held_var_rows,
                bound_eps,
            )
        })
    }

    /// [`crate::boundcheck::path_direction`] for a working set the caller
    /// names. See [`ConvergedSolve::path_direction_decided`].
    pub fn path_direction_decided(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        released_bound_rows: &[Index],
        held_primal_rows: &[Index],
        operator: PathOperator,
    ) -> Result<(Vec<Number>, Vec<Number>), SolverError> {
        self.with_converged(|c| {
            c.path_direction_decided(
                pin_constraint_indices,
                deltas,
                released_bound_rows,
                held_primal_rows,
                operator,
            )
        })
    }

    /// The all-released step. See
    /// [`ConvergedSolve::parametric_step_release_all`].
    pub fn parametric_step_release_all(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
    ) -> Result<(Vec<Number>, usize), SolverError> {
        self.with_converged(|c| c.parametric_step_release_all(pin_constraint_indices, deltas))
    }

    /// The eq. 14 directional derivative. See
    /// [`ConvergedSolve::parametric_step_directional`].
    pub fn parametric_step_directional(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_iter: usize,
    ) -> Result<(Vec<Number>, Vec<usize>, usize), SolverError> {
        self.with_converged(|c| {
            c.parametric_step_directional(pin_constraint_indices, deltas, max_iter)
        })
    }

    /// The bounds the activity classifier could not call at the base
    /// point. See [`ConvergedSolve::weakly_active_bounds`].
    pub fn weakly_active_bounds(&self) -> Result<Vec<WeakBound>, SolverError> {
        self.with_converged(|c| c.weakly_active_bounds())
    }

    /// Full KKT-space parametric step. See
    /// [`ConvergedSolve::parametric_step_full`].
    pub fn parametric_step_full(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
    ) -> Result<Vec<Number>, SolverError> {
        check_deltas(pin_constraint_indices, deltas)?;
        self.with_converged(|c| c.parametric_step_full(pin_constraint_indices, deltas))
    }

    /// The right-hand side [`Self::parametric_step_full`] answers,
    /// barrier term included. See [`ConvergedSolve::parametric_rhs_full`].
    pub fn parametric_rhs_full(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
    ) -> Result<Vec<Number>, SolverError> {
        self.with_converged(|c| c.parametric_rhs_full(pin_constraint_indices, deltas))
    }

    /// Flat rows of the equality multipliers `y_c` for the given full-g
    /// constraint indices. See [`ConvergedSolve::g_multiplier_rows`].
    pub fn g_multiplier_rows(
        &self,
        g_indices: &[Index],
    ) -> Result<Vec<Option<Index>>, SolverError> {
        self.with_converged(|c| c.g_multiplier_rows(g_indices))
    }

    /// Flat rows of the inequality multipliers `y_d` for the given full-g
    /// constraint indices. See [`ConvergedSolve::d_multiplier_rows`],
    /// which says when such a row is a derivative.
    pub fn d_multiplier_rows(
        &self,
        g_indices: &[Index],
    ) -> Result<Vec<Option<Index>>, SolverError> {
        self.with_converged(|c| c.d_multiplier_rows(g_indices))
    }

    /// Flat rows of an inequality's slack `s`. See
    /// [`ConvergedSolve::d_slack_rows`].
    pub fn d_slack_rows(&self, g_indices: &[Index]) -> Result<Vec<Option<Index>>, SolverError> {
        self.with_converged(|c| c.d_slack_rows(g_indices))
    }

    /// Flat rows of the primal values `x` for the given full-x variable
    /// indices. See [`ConvergedSolve::x_primal_rows`].
    pub fn x_primal_rows(&self, x_indices: &[Index]) -> Result<Vec<Option<Index>>, SolverError> {
        self.with_converged(|c| c.x_primal_rows(x_indices))
    }

    /// Flat rows of the bound multipliers `(z_l, z_u)` for the given
    /// full-x variable indices. See
    /// [`ConvergedSolve::bound_multiplier_rows`].
    pub fn bound_multiplier_rows(
        &self,
        x_indices: &[Index],
    ) -> Result<Vec<(Option<Index>, Option<Index>)>, SolverError> {
        self.with_converged(|c| c.bound_multiplier_rows(x_indices))
    }

    /// The user TNLP's variable count: the length of a full-x report
    /// and the domain of [`Self::x_primal_rows`].
    pub fn n_full_x(&self) -> Result<usize, SolverError> {
        self.with_converged(|c| c.n_full_x())
    }

    /// The user TNLP's constraint count: the length of a full-g
    /// report and the domain of [`Self::reduced_row_activity`].
    pub fn n_full_g(&self) -> Result<usize, SolverError> {
        self.with_converged(|c| c.n_full_g())
    }

    /// The negated reduced Hessian `−H_R` over the pinned equality rows,
    /// in natural units, so `−inv` of it is the parameter covariance.
    /// See [`ConvergedSolve::compute_reduced_hessian`] for the sign
    /// convention (gh#937), pinned by
    /// `tests/issue_937_reduced_hessian_sign.rs`.
    ///
    /// Equivalent to [`crate::SensSolve::with_reduced_hessian`] but
    /// usable post-hoc on a held `Solver`. For the solver-space value use
    /// [`Self::compute_reduced_hessian_scaled`].
    pub fn compute_reduced_hessian(
        &self,
        pin_constraint_indices: &[Index],
        obj_scal: Number,
    ) -> Result<Vec<Number>, SolverError> {
        self.with_converged(|c| c.compute_reduced_hessian(pin_constraint_indices, obj_scal))
    }

    /// [`Self::compute_reduced_hessian`] plus its eigendecomposition,
    /// `(H_R, eigenvalues, eigenvectors)`. The eigenvalues are those of
    /// `−H_R` in ascending order, so they run stiffest mode first. See
    /// [`ConvergedSolve::compute_reduced_hessian_eigen`].
    pub fn compute_reduced_hessian_eigen(
        &self,
        pin_constraint_indices: &[Index],
        obj_scal: Number,
    ) -> Result<(Vec<Number>, Vec<Number>, Vec<Number>), SolverError> {
        self.with_converged(|c| c.compute_reduced_hessian_eigen(pin_constraint_indices, obj_scal))
    }

    /// The reduced Hessian as the solver's internal scaled space sees it.
    /// See [`ConvergedSolve::compute_reduced_hessian_scaled`].
    pub fn compute_reduced_hessian_scaled(
        &self,
        pin_constraint_indices: &[Index],
        obj_scal: Number,
    ) -> Result<Vec<Number>, SolverError> {
        self.with_converged(|c| c.compute_reduced_hessian_scaled(pin_constraint_indices, obj_scal))
    }

    /// Effective NLP scaling the IPM applied on the most recent
    /// converged solve: `(obj_scaling_factor, c_scale, d_scale)`.
    /// `(1.0, None, None)` means no scaling was active. The vectors are
    /// per-row factors over the algorithm's equality (`c`) and
    /// inequality (`d`) blocks.
    pub fn nlp_scaling(
        &self,
    ) -> Result<(Number, Option<Vec<Number>>, Option<Vec<Number>>), SolverError> {
        self.with_converged(|c| c.nlp_scaling())
    }

    /// The per-variable `user-scaling` factors `d` the held solve ran
    /// under (gh#486), in the user TNLP's **full-x** space, or `None`
    /// when the solve applied no change of variables.
    ///
    /// Every accessor on this type already reports natural units, so
    /// this is diagnostic rather than a correction a caller has to
    /// apply: it answers "was this solve conditioned, and by how much",
    /// the x-axis counterpart of [`Self::nlp_scaling`].
    pub fn variable_scaling(&self) -> Result<Option<Vec<Number>>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        Ok(state.backsolver.variable_scaling_full().map(|d| d.to_vec()))
    }

    /// Inertia-correction perturbations `(δ_x, δ_s, δ_c, δ_d)` baked
    /// into the held KKT factor. All zero means the final factorization
    /// was unregularized and the natural-units back-solves invert the
    /// exact KKT matrix. See
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
        self.with_converged(|c| c.pin_g_scaling(pin_constraint_indices))
    }
}

/// The parametric steps take one perturbation per pinned row, and that is
/// checked before anything reads the held state, as it always was.
fn check_deltas(pins: &[Index], deltas: &[Number]) -> Result<(), SolverError> {
    if pins.len() != deltas.len() {
        return Err(SolverError::BadShape {
            what: "deltas",
            got: deltas.len(),
            expected: pins.len(),
        });
    }
    Ok(())
}
