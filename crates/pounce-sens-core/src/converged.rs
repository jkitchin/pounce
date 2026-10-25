//! The sensitivity calculations over a converged barrier solve, written
//! against [`BarrierIterate`] so they run over any engine that implements
//! it.
//!
//! [`ConvergedSolve`] holds a reference to the engine and the two facts
//! about the solve that the engine's iterate does not carry: the
//! `bound_relax_factor` it ran with and whether it computed exact
//! Hessians. Every calculation reads the rest through the trait.
//! `pounce_sensitivity::Solver` builds one over its own converged factor
//! for each call, so pounce's results come from this code too.

use pounce_common::types::{Index, NLP_UPPER_BOUND_INF, Number};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};

use crate::activity::{
    AMBIGUOUS, ActivityReport, ReducedActivityError, ReducedActivityReport,
    ReducedRowActivityError, ReducedRowActivityReport, WEAKLY_ACTIVE,
};
use crate::barrier::{ActivityData, BarrierIterate, EngineScaling, ModelMaps};
use crate::boundcheck::{BoundMultiplier, PathOperator, PathSegment, RefineStop, WeakBound};
use crate::corrector::{CorrectorError, CorrectorReport};
use crate::index::{FullXSlice, VarToFull, VarX};
use crate::schur_data::IndexSchurData;
use crate::sens_app::{SensApplication, SensOptions};

/// Sign of the barrier correction term, set from a comparison
/// against sIPOPT rather than derived.
pub const BARRIER_SIGN: Number = -1.0;

/// Errors returned by the calculations over a converged solve.
#[derive(Debug, Clone, PartialEq)]
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

/// The bound geometry the bound-aware parametric steps share. See
/// [`ConvergedSolve::bound_context`].
struct BoundContext {
    /// Length of the `x` block: how much of the box below is
    /// variables, and how much of a step is the caller's answer.
    n_x: usize,
    /// Lower bounds over the `(x, s)` prefix of the compound KKT
    /// vector, in the model's own units.
    ///
    /// The `s` half is the limits of the inequality rows, `d_l`. A
    /// limit written as a constraint, `g(x) <= cap`, is a bound like
    /// any other, on the row's slack instead of on a variable, and
    /// leaving it out of the box is what let a step walk straight
    /// through a cap with no breakpoint and no warning (gh#928). The
    /// two blocks are adjacent in the compound vector, so the box is
    /// one contiguous slice and a consumer needs no second index
    /// space. `n_x` is where it changes meaning.
    lo: Vec<Number>,
    /// Upper bounds, likewise: variable `x_u` then row `d_u`.
    hi: Vec<Number>,
    /// The converged point over that same `(x, s)` prefix.
    x_curr: Vec<Number>,
    /// How far outside a bound still counts as on it.
    eps: Number,
    /// How far negative a bound multiplier has to go before its bound
    /// is released. Always the solve's own margin, whatever `eps` is.
    release_eps: Number,
    /// Bound multipliers at the base point, in the solve's own
    /// coordinates, with the compound row each occupies: `z_l`, `z_u`,
    /// then `v_l`, `v_u`. The `v` half is what lets an active
    /// constraint limit be *released* when a step drives its
    /// multiplier through zero, the mirror of the `s` half of the box.
    mults: Vec<BoundMultiplier>,
}

impl BoundContext {
    /// Distance from the base point to each bound, for one var-x row.
    ///
    /// Typed because the callers read a full-x `ActivityReport` in the
    /// same scope: indexing `lo` / `hi` / `x_curr` with a full-x value
    /// is the swap `crate::index` exists to prevent. Those three run
    /// past the `x` block into `s`, so a var-x row is in range for the
    /// whole box and a full-x row that overshoots `n_x` does not run
    /// off the end, it silently reads a slack.
    fn slacks_at(&self, row: VarX) -> (Number, Number) {
        let i = row.get();
        (self.x_curr[i] - self.lo[i], self.hi[i] - self.x_curr[i])
    }
}

/// A converged barrier solve, and the sensitivity calculations over it.
///
/// Built over any [`BarrierIterate`]. `bound_relax_factor` is the value
/// the solve ran with: the classification needs it to be zero, and the
/// bound-aware steps take their margin from it. `exact_hessian` says
/// whether the solve computed exact Hessians, which decides whether the
/// corrector re-evaluates the Hessian at the predicted point.
pub struct ConvergedSolve<'a, E: BarrierIterate + ?Sized> {
    engine: &'a E,
    bound_relax_factor: Number,
    exact_hessian: bool,
    /// What the engine reports about its converged iterate, read from it
    /// once, when first needed. The engine cannot change while this
    /// borrows it, so every calculation uses that one read.
    maps: std::cell::OnceCell<ModelMaps>,
    scaling: std::cell::OnceCell<EngineScaling>,
    data: std::cell::OnceCell<ActivityData>,
}

impl<'a, E: BarrierIterate + ?Sized> ConvergedSolve<'a, E> {
    /// The calculations over `engine`'s converged iterate.
    pub fn new(engine: &'a E, bound_relax_factor: Number, exact_hessian: bool) -> Self {
        Self {
            engine,
            bound_relax_factor,
            exact_hessian,
            maps: std::cell::OnceCell::new(),
            scaling: std::cell::OnceCell::new(),
            data: std::cell::OnceCell::new(),
        }
    }

    /// The engine the calculations run over.
    pub fn engine(&self) -> &'a E {
        self.engine
    }

    /// The `bound_relax_factor` the solve ran with.
    pub fn bound_relax_factor(&self) -> Number {
        self.bound_relax_factor
    }

    /// The engine's model maps, checked against its block lengths.
    fn maps(&self) -> Result<&ModelMaps, SolverError> {
        if let Some(maps) = self.maps.get() {
            return Ok(maps);
        }
        let maps = self
            .engine
            .model_maps()
            .map_err(SolverError::SensComputationFailed)?;
        maps.check(&self.engine.block_dims())
            .map_err(SolverError::SensComputationFailed)?;
        Ok(self.maps.get_or_init(|| maps))
    }

    /// The engine's scaling, checked against its block lengths.
    fn scaling(&self) -> Result<&EngineScaling, SolverError> {
        if let Some(scaling) = self.scaling.get() {
            return Ok(scaling);
        }
        let scaling = self.engine.scaling();
        scaling
            .check(&self.engine.block_dims())
            .map_err(SolverError::SensComputationFailed)?;
        Ok(self.scaling.get_or_init(|| scaling))
    }

    /// The converged quantities the classification reads. The products
    /// below read the Hessian and the Jacobians from them.
    fn data(&self) -> Result<&ActivityData, SolverError> {
        if let Some(data) = self.data.get() {
            return Ok(data);
        }
        let data = self
            .engine
            .activity_data()
            .map_err(SolverError::SensComputationFailed)?;
        Ok(self.data.get_or_init(|| data))
    }

    /// The converged iterate, in the engine frame.
    fn iterate(&self) -> Result<Vec<Number>, SolverError> {
        let mut out = vec![0.0; self.engine.dim()];
        self.engine
            .converged_iterate(&mut out)
            .map_err(SolverError::SensComputationFailed)?;
        Ok(out)
    }

    fn require_unrelaxed(&self, what: &str) -> Result<(), SolverError> {
        let brf = self.bound_relax_factor;
        if brf != 0.0 {
            return Err(SolverError::BadOptions(format!(
                "{what} requires bound_relax_factor=0, but the \
                 held solve ran with {brf:e}: relaxed bounds shift the \
                 slacks the classifier reads. Set the option and solve() \
                 again — changing it now does not re-measure the slacks."
            )));
        }
        Ok(())
    }

    /// Classify every bounded variable and every finite-bounded
    /// inequality row of the converged solve by activity: see
    /// [`crate::activity`] and
    /// `dev-notes/covariance-information-roadmap.md` item 0 (gh #362).
    ///
    /// Requires the solve to have run with `bound_relax_factor=0`
    /// (the Ipopt default is `1e-8`): with relaxed bounds the solver's
    /// slacks are measured against perturbed bounds, and the
    /// complementarity products the classifier reads no longer track
    /// `μ`.
    ///
    /// # Neither classes' `q` is a reduced curvature
    ///
    /// A variable's ratio is `Σ_i/|H_ii|`, and at a kink the
    /// multiplier is generated by the curvature **reduced** along that
    /// coordinate, not by the diagonal. The two agree only where the
    /// coordinate is decoupled, so a genuine kink coupled to a
    /// neighbour reads [`AMBIGUOUS`] here at any tolerance (gh#763).
    /// Do not read that class as "probably not a kink": use
    /// [`Self::reduced_activity`], which normalizes by the reduced
    /// curvature at one back-solve per coordinate.
    ///
    /// A row's ratio divides by the curvature along the row's own
    /// gradient instead, which is a genuine directional curvature but
    /// still not a reduced one, so the same warning and the same
    /// remedy apply there: its ratio is `reduced/directional` and
    /// [`Self::reduced_row_activity`] answers the kink question
    /// (gh#804).
    pub fn classify_activity(&self) -> Result<ActivityReport, SolverError> {
        self.require_unrelaxed("classify_activity")?;
        crate::activity::compute(self.engine).map_err(SolverError::SensComputationFailed)
    }

    /// [`Self::classify_activity`]'s per-variable verdict for
    /// `user_vars`, re-measured against the **reduced** curvature
    /// along each coordinate instead of the Hessian diagonal, one
    /// back-solve against the held factor per variable (gh#763).
    ///
    /// Indices are **user space** (full-x), as the report's are. The
    /// cost is one back-solve per index, so it is a refinement to call
    /// over the entries in question, not over every bounded variable of
    /// a large model. See [`crate::activity::reduced_activity`] for the
    /// algebra and the edge cases.
    ///
    /// Requires `bound_relax_factor=0` for the same reason
    /// [`Self::classify_activity`] does: both read the slacks relaxed
    /// bounds shift.
    pub fn reduced_activity(
        &self,
        user_vars: &[usize],
    ) -> Result<ReducedActivityReport, SolverError> {
        self.require_unrelaxed("reduced_activity")?;
        crate::activity::reduced_activity(self.engine, user_vars).map_err(|e| match e {
            ReducedActivityError::OutOfRange { got, n_full_x } => SolverError::BadShape {
                what: "reduced_activity variable index",
                got,
                expected: n_full_x,
            },
            ReducedActivityError::Backsolve => SolverError::BacksolveFailed,
            ReducedActivityError::Data(m) => SolverError::SensComputationFailed(m),
        })
    }

    /// [`Self::classify_activity`]'s per-row verdict for `user_rows`,
    /// re-measured against the **reduced** curvature along each row's
    /// gradient instead of the directional curvature `∇dᵀH∇d/‖∇d‖²`, one
    /// back-solve against the held factor per row (gh#804).
    ///
    /// Indices are **user space** (full-g), as the report's are,
    /// equality rows included, which report
    /// [`EQUALITY`](crate::activity::EQUALITY) rather than being an
    /// error. See [`crate::activity::reduced_row_activity`] for the
    /// algebra and the edge cases.
    ///
    /// Requires `bound_relax_factor=0` for the same reason
    /// [`Self::classify_activity`] does.
    pub fn reduced_row_activity(
        &self,
        user_rows: &[usize],
    ) -> Result<ReducedRowActivityReport, SolverError> {
        self.require_unrelaxed("reduced_row_activity")?;
        crate::activity::reduced_row_activity(self.engine, user_rows).map_err(|e| match e {
            ReducedRowActivityError::OutOfRange { got, n_full_g } => SolverError::BadShape {
                what: "reduced_row_activity constraint index",
                got,
                expected: n_full_g,
            },
            ReducedRowActivityError::Backsolve => SolverError::BacksolveFailed,
            ReducedRowActivityError::Data(m) => SolverError::SensComputationFailed(m),
        })
    }

    /// The gradient of user constraint row `user_row` at the converged
    /// iterate, in user variable order (length `n_full_x`) and in
    /// **natural (unscaled) units**: the engine's Jacobian row carries
    /// its per-row scale and its change of variables, both divided out
    /// here, so this is the gradient of the row as the user wrote it.
    /// Equality and inequality rows alike. Entries for variables the
    /// engine removed are 0, because the solve dropped their columns.
    /// Errors on an out-of-range row.
    ///
    /// Serves the covariance roadmap's item 1: a binding row's normal
    /// restricted to the fitted block is the projection direction.
    pub fn row_normal(&self, user_row: usize) -> Result<Vec<Number>, SolverError> {
        use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};

        let maps = self.maps()?;
        let n_full_g = maps.n_full_g();
        let bad = || SolverError::BadShape {
            what: "row_normal constraint index",
            got: user_row,
            expected: n_full_g,
        };
        if user_row >= n_full_g {
            return Err(bad());
        }
        let scaling = self.scaling()?;
        let data = self.data()?;
        let (jac, block_pos, row_scale) =
            match (maps.full_g_to_c[user_row], maps.full_g_to_d[user_row]) {
                (Some(p), _) => (
                    &data.jac_c,
                    p,
                    scaling.equality_rows.as_ref().map_or(1.0, |v| v[p]),
                ),
                (None, Some(p)) => (
                    &data.jac_d,
                    p,
                    scaling.inequality_rows.as_ref().map_or(1.0, |v| v[p]),
                ),
                (None, None) => return Err(bad()),
            };
        let n = self.engine.block_dims()[0];
        let m_block = jac.n_rows() as usize;
        let mut e_row = DenseVector::new(DenseVectorSpace::new(m_block as i32));
        let mut grad = DenseVector::new(DenseVectorSpace::new(n as i32));
        e_row.values_mut().fill(0.0);
        e_row.values_mut()[block_pos] = 1.0;
        grad.values_mut().fill(0.0);
        jac.trans_mult_vector(1.0, &e_row, 0.0, &mut grad);

        let d_var = scaling.variables.as_deref();
        let mut full = vec![0.0; maps.n_full_x];
        for (i, slot) in grad.values_mut().iter().enumerate() {
            // `∇g̃ = (∇g ⊘ d) · row_scale`, so both come back out here
            // (gh#486 stage 3).
            let dx = d_var.map_or(1.0, |d| d[i]);
            full[maps.var_x_to_full_x[i]] = *slot * dx / row_scale;
        }
        Ok(full)
    }

    /// The exact Lagrangian Hessian times a user-space vector, in user
    /// variable order and **natural (unscaled) units**: the engine's
    /// Hessian carries the objective scale and the change of variables,
    /// both divided out here. Entries for variables the engine removed
    /// are 0 in and out, since their columns left the solve. Serves the
    /// covariance roadmap's item 2: the tangent-recovered reduced
    /// Hessian is `T^T (H T)`, one product per fitted column. Errors on
    /// a length mismatch.
    pub fn hessian_vec(&self, v_full: &[Number]) -> Result<Vec<Number>, SolverError> {
        use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};

        let maps = self.maps()?;
        if v_full.len() != maps.n_full_x {
            return Err(SolverError::BadShape {
                what: "hessian_vec vector length",
                got: v_full.len(),
                expected: maps.n_full_x,
            });
        }
        let scaling = self.scaling()?;
        let data = self.data()?;
        let n = self.engine.block_dims()[0];
        // `H = H̃ ⊙ (d ⊗ d)` under a change of variables (gh#486 stage 3),
        // so `H v = d ⊙ (H̃ (d ⊙ v))`: the factor goes in with the vector
        // and comes back out of the product.
        let d_var = scaling.variables.as_deref();
        let space = DenseVectorSpace::new(n as i32);
        let mut v_int = DenseVector::new(space.clone());
        let mut hv = DenseVector::new(space);
        {
            let vals = v_int.values_mut();
            vals.fill(0.0);
            for (i, slot) in vals.iter_mut().enumerate() {
                let dx = d_var.map_or(1.0, |d| d[i]);
                *slot = v_full[maps.var_x_to_full_x[i]] * dx;
            }
        }
        data.hessian.mult_vector(1.0, &v_int, 0.0, &mut hv);
        let mut out = vec![0.0; maps.n_full_x];
        for (i, slot) in hv.values_mut().iter().enumerate() {
            let dx = d_var.map_or(1.0, |d| d[i]);
            out[maps.var_x_to_full_x[i]] = *slot * dx / scaling.objective;
        }
        Ok(out)
    }

    /// Solve `K · lhs = rhs` against the converged KKT factor, in
    /// natural units. Both slices must have length `kkt_dim()`, and the
    /// layout is the flat `x || s || y_c || y_d || z_l || z_u || v_l ||
    /// v_u` packing.
    pub fn kkt_solve(&self, rhs: &[Number], lhs: &mut [Number]) -> Result<(), SolverError> {
        let total = self.engine.dim();
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
        if self.engine.solve(rhs, lhs) {
            Ok(())
        } else {
            Err(SolverError::BacksolveFailed)
        }
    }

    /// Batched-RHS back-solve. `rhs_flat` and `lhs_flat` are row-major
    /// `(n_rhs, kkt_dim)` buffers, each row solved against the same
    /// converged factor. Equivalent in result to looping
    /// [`Self::kkt_solve`].
    pub fn kkt_solve_many(
        &self,
        rhs_flat: &[Number],
        lhs_flat: &mut [Number],
        n_rhs: usize,
    ) -> Result<(), SolverError> {
        let total = self.engine.dim();
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
        if self.engine.solve_many(rhs_flat, lhs_flat, n_rhs) {
            Ok(())
        } else {
            Err(SolverError::BacksolveFailed)
        }
    }

    /// The compound KKT rows of the pinned equality rows, and the scale
    /// factor the engine applied to each.
    ///
    /// The KKT row of pin `i` is `n_x + n_s + c`, the matching `y_c` slot,
    /// found through the model maps so the split of the model's rows into
    /// equality and inequality blocks is honored: whenever an inequality
    /// precedes a pin in `g(x)`, `n_x + n_s + g` names the wrong row
    /// (pounce#128). The scales are 1.0 when no row scaling is active.
    /// Errors when a pin index is out of range or refers to an inequality
    /// row.
    fn pin_rows_and_scales(
        &self,
        pins: &[Index],
    ) -> Result<(Vec<Index>, Vec<Number>), SolverError> {
        let dims = self.engine.block_dims();
        let y_c_offset = (dims[0] + dims[1]) as Index;
        let maps = self.maps()?;
        let dc = &self.scaling()?.equality_rows;
        let n_full_g = maps.n_full_g() as Index;
        let mut rows = Vec::with_capacity(pins.len());
        let mut scales = Vec::with_capacity(pins.len());
        for &gi in pins {
            if gi < 0 || (n_full_g > 0 && gi >= n_full_g) {
                return Err(SolverError::SensComputationFailed(format!(
                    "pin constraint index {gi} out of range [0, m={n_full_g})"
                )));
            }
            let Some(ci) = maps.full_g_to_c.get(gi as usize).copied().flatten() else {
                return Err(SolverError::SensComputationFailed(format!(
                    "pin constraint index {gi} is an inequality (not an equality row); \
                     parameter pins must be exact equalities"
                )));
            };
            rows.push(y_c_offset + ci as Index);
            scales.push(dc.as_ref().map(|v| v[ci]).unwrap_or(1.0));
        }
        Ok((rows, scales))
    }

    /// A sIPOPT application over the pinned rows, holding the engine by
    /// reference.
    fn sens_app(
        &self,
        pins: &[Index],
        opts: SensOptions,
    ) -> Result<SensApplication<&'a E>, SolverError> {
        let (param_rows, _) = self.pin_rows_and_scales(pins)?;
        let signs = vec![1; pins.len()];
        let a_data = IndexSchurData::from_parts(param_rows, signs)
            .map_err(|e| SolverError::SensComputationFailed(format!("{e:?}")))?;
        Ok(SensApplication::new(a_data, self.engine, opts))
    }

    /// First-order parametric step `Δx ≈ ∂x*/∂p · Δp` for a set of
    /// pinned equality constraints. `pin_constraint_indices` are 0-based
    /// indices into the user's `g(x)`, and `deltas` is the perturbation
    /// `Δp` (same length).
    ///
    /// Returns the `n_x`-long primal step. For the full KKT-space step,
    /// use [`Self::parametric_step_full`].
    pub fn parametric_step(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
    ) -> Result<Vec<Number>, SolverError> {
        let n_x = self.engine.block_dims()[0];
        let mut dx_full = self.parametric_step_full(pin_constraint_indices, deltas)?;
        dx_full.truncate(n_x);
        Ok(dx_full)
    }

    /// Full KKT-space parametric step for a set of pinned equality
    /// constraints: the same computation as [`Self::parametric_step`],
    /// returned WITHOUT truncating to the primal block. The layout is
    /// the compound KKT vector `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)`.
    /// [`Self::g_multiplier_rows`] locates a constraint's multiplier
    /// row, so this exposes the multiplier sensitivities `∂λ*/∂p`
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
        let opts = SensOptions {
            run_sens: true,
            ..SensOptions::default()
        };
        let sens_app = self.sens_app(pin_constraint_indices, opts)?;
        let n_full = self.engine.dim();
        let mut dx_full = vec![0.0; n_full];
        if !sens_app.parametric_step(deltas, &mut dx_full) {
            return Err(SolverError::SensComputationFailed(
                "SensApplication::parametric_step failed".into(),
            ));
        }
        // carry the step from the barrier problem's solution toward the
        // original problem's (the paper's equation 11)
        let corr = self.barrier_correction()?;
        for (d, c) in dx_full.iter_mut().zip(corr.iter()) {
            *d += *c * BARRIER_SIGN;
        }
        Ok(dx_full)
    }

    /// The right-hand side [`Self::parametric_step_full`] answers,
    /// barrier term included. That method adds the term as a correction
    /// to the solution rather than to the right-hand side, which is the
    /// same thing by linearity.
    ///
    /// The parameter rows go through the model maps exactly as they do
    /// there. Passing the constraint indices raw instead puts the
    /// perturbation on the x rows, where it contributes nothing: a
    /// release then sees only its own multiplier shift and lands on the
    /// wrong answer without failing.
    pub fn parametric_rhs_full(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
    ) -> Result<Vec<Number>, SolverError> {
        let dims = self.engine.block_dims();
        let n_full = self.engine.dim();
        let opts = SensOptions {
            run_sens: true,
            ..SensOptions::default()
        };
        let sens_app = self.sens_app(pin_constraint_indices, opts)?;
        let mut rhs = vec![0.0; n_full];
        if !sens_app.parametric_rhs(deltas, &mut rhs) {
            return Err(SolverError::SensComputationFailed(
                "SensApplication::parametric_rhs failed".into(),
            ));
        }
        let mu = self.engine.barrier_mu();
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
    /// The step is taken against a factorization held at the final
    /// `mu`, so it estimates where the BARRIER problem's solution
    /// moves, not where the original problem's does. The two differ by
    /// `O(mu)`, which is negligible at a tight tolerance and is not at a
    /// loose one. Measured against sIPOPT on a nonlinear model, the
    /// uncorrected step agrees to 2e-9 at `tol = 1e-8` and differs by
    /// 9e-6 at `tol = 1e-3`.
    ///
    /// The term is one more backsolve against the same factor, with
    /// `mu` in the complementarity rows, which are the bound multiplier
    /// blocks of the compound vector. Returns the correction over the
    /// whole compound vector, to be added to the step.
    fn barrier_correction(&self) -> Result<Vec<Number>, SolverError> {
        let dims = self.engine.block_dims();
        let n_full = self.engine.dim();
        let mu = self.engine.barrier_mu();
        // z_l, z_u, v_l, v_u: the rows carrying the complementarity
        // conditions, which are the ones the barrier perturbs
        let start = dims[0] + dims[1] + dims[2] + dims[3];
        let end = start + dims[4] + dims[5] + dims[6] + dims[7];
        let mut rhs = vec![0.0; n_full];
        for r in rhs.iter_mut().take(end).skip(start) {
            *r = mu;
        }
        let mut corr = vec![0.0; n_full];
        if !self.engine.solve(&rhs, &mut corr) {
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
    /// costs one dense `k × k` solve and `k + 1` back-solves. The
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
    /// the returned [`RefineStop`] says which happened.
    pub fn parametric_step_bounded(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_iter: usize,
        bound_eps: Option<Number>,
    ) -> Result<(Vec<Number>, Vec<Index>, RefineStop), SolverError> {
        let dx_full = self.parametric_step_full(pin_constraint_indices, deltas)?;
        let rhs_plain = self.parametric_rhs_full(pin_constraint_indices, deltas)?;
        let ctx = self.bound_context(bound_eps)?;
        let (dx, pinned, stop) = crate::boundcheck::refine_step_onto_bounds(
            &self.engine,
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
    ) -> Result<(Vec<Number>, Vec<PathSegment>), SolverError> {
        let rhs_plain = self.parametric_rhs_full(pin_constraint_indices, deltas)?;
        // Nothing here decides the weak rows, which is what the
        // "decided" variant below is for, but the walk still has to be
        // told which they are, or it reads their order-one sigma as a
        // bound the factorization enforces and lets the variable walk
        // out of its box (gh#852). A relaxed solve shifts the slacks the
        // classifier reads, so this comes back empty there and the walk
        // behaves as it did before, the same silence
        // `weakly_active_bounds` hands every other caller.
        let weak_rows: Vec<usize> = self.weakly_active_bounds()?.iter().map(|w| w.row).collect();
        let ctx = self.bound_context(None)?;
        let (dx, segments) = crate::boundcheck::step_along_path(
            &self.engine,
            &rhs_plain,
            &ctx.x_curr,
            &ctx.lo,
            &ctx.hi,
            &ctx.mults,
            max_iter,
            &[],
            &[],
            &weak_rows,
            ctx.eps,
        )
        .map_err(SolverError::SensComputationFailed)?;
        Ok((dx[..ctx.n_x].to_vec(), segments))
    }

    /// [`Self::parametric_step_path`] with the weak-row decision
    /// supplied by the caller instead of searched for.
    /// `held_var_rows` names the var-x rows of the weakly active bounds
    /// the direction holds, and every other weakly active bound is
    /// forced into the walk's base-activity table as a leaving row. A
    /// row left there is still reachable, so a caller that hands in an
    /// empty held list, every weak row declared a leaver, which is what
    /// an undecided study of the all-released step does, gets the bound
    /// back at the fraction the walk finds the direction pressing into
    /// it, rather than an answer outside the box (gh#852). Study surface
    /// for an externally solved eq. 14 QP.
    pub fn parametric_step_path_decided(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_iter: usize,
        held_var_rows: &[Index],
    ) -> Result<(Vec<Number>, Vec<PathSegment>), SolverError> {
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
        // Every weak row, held or leaving. For a held one the flag is
        // inert, since it arrives released and pinned already, and for
        // a leaving one it is what lets the walk take the bound back
        // when the direction turns out to press into it, which is the
        // whole of an empty held list on a holding perturbation
        // (gh#852).
        let weak_rows: Vec<usize> = weak.iter().map(|w| w.row).collect();
        let (dx, segments) = crate::boundcheck::step_along_path(
            &self.engine,
            &rhs_plain,
            &ctx.x_curr,
            &ctx.lo,
            &ctx.hi,
            &ctx.mults,
            max_iter,
            &forced_active,
            &holds,
            &weak_rows,
            ctx.eps,
        )
        .map_err(SolverError::SensComputationFailed)?;
        Ok((dx[..ctx.n_x].to_vec(), segments))
    }

    /// Newton iterations on the barrier system, refining a step that
    /// some mode already produced.
    ///
    /// `step` is a full compound step, the shape
    /// [`Self::parametric_step_full`] returns, so any mode's result can
    /// be handed in. Every correction pays one derivative evaluation and
    /// one factorization at the predicted point, and each iteration
    /// after that costs one back-solve. Returns the refined step and a
    /// [`CorrectorReport`] saying what the iterations bought.
    ///
    /// The corrector aims at the barrier solution at the μ the solve
    /// finished on, not at a re-solve, so the accuracy it can reach is
    /// bounded by that offset. Its operator is assembled at the
    /// PREDICTED point, every block: the Hessian, the constraint
    /// Jacobians, and the barrier diagonal all evaluated at the stepped
    /// iterate with the step's own multipliers, and the predictor's
    /// active set applied to the diagonal in that frame. A chord
    /// iteration contracts at the rate the distance between its operator
    /// and the true Jacobian sets, and the predicted point is where the
    /// truth is. Under a `limited-memory` solve the quasi-Newton matrix
    /// is kept as is, since no exact Hessian exists to evaluate
    /// elsewhere. Where the perturbation needs a bound to leave the
    /// active set that the step's endpoint does not show, no released
    /// row is applied: the step's clamped multiplier leaves a weak
    /// diagonal entry there, the iterations can move the coordinate
    /// partway off the bound, and the answer is not the re-solve. The
    /// release-deciding modes are the ones that cross exactly.
    /// `CorrectorReport::improved` reports whether the residual fell,
    /// and when it did not, the step handed back is the caller's own.
    ///
    /// The returned point always satisfies the variable bounds, since
    /// the barrier residual is undefined outside them and the
    /// fraction-to-boundary rule keeps every iterate inside. A step that
    /// arrives pointing out of the box is therefore put back in before
    /// the first iteration, which means `max_iter = 0` is not a no-op:
    /// it costs the derivative evaluation and the residual evaluation,
    /// no back-solve, and reports the residual the caller's step leaves.
    ///
    /// Errors with [`SolverError::SensComputationFailed`] when the
    /// barrier residual at that starting point is not finite, which is
    /// what a predicted point outside the domain of one of the model's
    /// functions gives (gh#845). A *declared bound* is protection here,
    /// since the clamp above puts the coordinate back inside it. A
    /// variable held in a function's domain by a **constraint** has no
    /// bound to be put back inside, and an ordinary `log`, `sqrt` or
    /// reciprocal is then reachable by a large enough perturbation.
    /// There is no correction to make from such a point, so it is an
    /// error rather than a report, and never a step full of NaN carrying
    /// `residual = 0.0` and `converged = true`.
    pub fn correct_step(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        step: &[Number],
        max_iter: usize,
    ) -> Result<(Vec<Number>, CorrectorReport), SolverError> {
        let ctx = self.bound_context(None)?;
        let dim = self.engine.dim();
        if step.len() != dim {
            return Err(SolverError::BadShape {
                what: "step",
                got: step.len(),
                expected: dim,
            });
        }
        // The pinned equalities' KKT rows and the row scales the engine
        // applied to them. A user `g` index is not the KKT row: the two
        // differ once an inequality precedes the pin in `g(x)`
        // (pounce#128), and the residual the corrector measures sits in
        // the engine's scaled equality block, so the deltas have to carry
        // the same factors.
        let (pin_rows, pin_scales) = self.pin_rows_and_scales(pin_constraint_indices)?;
        let pin_rows: Vec<usize> = pin_rows.iter().map(|&r| r as usize).collect();
        let scaled_deltas: Vec<Number> = deltas
            .iter()
            .zip(&pin_scales)
            .map(|(&d, &c)| d * c)
            .collect();
        crate::corrector::correct(
            self.engine,
            step,
            &pin_rows,
            &scaled_deltas,
            &ctx.lo,
            &ctx.hi,
            max_iter,
            self.exact_hessian,
        )
        .map_err(|e| match e {
            CorrectorError::Failed(m) => SolverError::SensComputationFailed(m),
            CorrectorError::Backsolve => SolverError::BacksolveFailed,
        })
    }

    /// [`Self::parametric_step_bounded`] with the weak-row decision
    /// supplied by the caller instead of searched for. The direction is
    /// computed for the given working set (all weak rows released, the
    /// held variables pinned through Schur rows), then refined onto the
    /// bounds exactly as the searched variant does. Study surface for an
    /// externally solved eq. 14 QP.
    pub fn parametric_step_bounded_decided(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_iter: usize,
        held_var_rows: &[Index],
        bound_eps: Option<Number>,
    ) -> Result<(Vec<Number>, Vec<Index>, RefineStop), SolverError> {
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
        let released: Vec<usize> = weak.iter().map(|w| w.row).collect();
        let pinned_rows: Vec<usize> = held_var_rows.iter().map(|&r| r as usize).collect();
        let (d, _) =
            crate::boundcheck::path_direction(&self.engine, &rhs_plain, &released, &pinned_rows)
                .map_err(SolverError::SensComputationFailed)?;
        let (dx, pinned, stop) = crate::boundcheck::refine_step_onto_bounds(
            &self.engine,
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

    /// [`crate::boundcheck::path_direction`] for a working set the caller
    /// names, and the force each held row carries under it. Study
    /// surface, and the seam the two pins are measured against each
    /// other through.
    ///
    /// `released_bound_rows` are compound bound-multiplier rows, as
    /// [`Self::weakly_active_bounds`] reports them. `held_primal_rows`
    /// are primal rows: `x` block for a variable, `s` block for a
    /// constraint's own limit (gh#928).
    ///
    /// `operator` picks which operator the walk's single, exact Schur
    /// pin is applied to. See [`PathOperator`]. Both operators give the
    /// same answer: the Schur row enforces `Eᵀ w = 0`, which annihilates
    /// the added diagonal, so the system solved is the released one
    /// either way.
    pub fn path_direction_decided(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        released_bound_rows: &[Index],
        held_primal_rows: &[Index],
        operator: PathOperator,
    ) -> Result<(Vec<Number>, Vec<Number>), SolverError> {
        let rhs_plain = self.parametric_rhs_full(pin_constraint_indices, deltas)?;
        let released: Vec<usize> = released_bound_rows.iter().map(|&r| r as usize).collect();
        let held: Vec<usize> = held_primal_rows.iter().map(|&r| r as usize).collect();
        crate::boundcheck::path_direction_with(&self.engine, &rhs_plain, &released, &held, operator)
            .map_err(SolverError::SensComputationFailed)
    }

    /// The all-released step: the plain parametric step solved with
    /// every weakly active bound's row released, and nothing decided.
    ///
    /// This is [`Self::parametric_step_directional`]'s first back-solve
    /// returned as the answer instead of refined. The caller trades the
    /// engagement's budget for whatever violations the released
    /// direction carries at weak bounds the perturbation actually holds,
    /// which come back as crossings for the mode's clamp, pins, or path
    /// segments, or for a correction, to handle. A clean base point
    /// takes the plain step. Returns the direction over the model's
    /// variables and the number of rows released.
    pub fn parametric_step_release_all(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
    ) -> Result<(Vec<Number>, usize), SolverError> {
        let rhs_plain = self.parametric_rhs_full(pin_constraint_indices, deltas)?;
        let weak = self.weakly_active_bounds()?;
        let ctx = self.bound_context(None)?;
        let released: Vec<usize> = weak.iter().map(|w| w.row).collect();
        let mut d = vec![0.0; self.engine.dim()];
        // solve_released is the whole mechanism: an empty released set
        // is the plain solve, and it does nothing to the right-hand side,
        // matching the directional path's all-released solve.
        if !self.engine.solve_released(&released, &rhs_plain, &mut d) {
            return Err(SolverError::BacksolveFailed);
        }
        Ok((d[..ctx.n_x].to_vec(), released.len()))
    }

    /// The eq. 14 directional derivative, decided by pounce-qp over the
    /// weak rows the direction engages.
    ///
    /// One released operator serves the whole decision, built once from
    /// [`BarrierIterate::released_operator`], so an engine that factors
    /// it once reuses the factor across the all-released direction and
    /// the basis columns. The decision itself is the dual of eq. 14
    /// restricted to the weak rows the direction engages: with `a_k` the
    /// signed unit vector of weak row `k` (positive for a lower bound),
    /// `X_k = K_rel^{-1} a_k`, `S = aᵀX` and `m = aᵀd0`, the pin forces
    /// `λ` solve
    ///
    /// ```text
    ///     min  ½ λᵀ S λ + mᵀ λ    s.t.  λ ≥ 0
    /// ```
    ///
    /// whose KKT conditions are eq. 14's complementarity: a released row
    /// moves to its feasible side (the QP gradient `Sλ + m ≥ 0`) and a
    /// held row's pin force is nonnegative. Rows outside the engaged set
    /// are verified against the decided direction and the set expands
    /// until no new row violates. Nothing reads the perturbation's size,
    /// so the decision is linear in the step.
    ///
    /// An engaged row is decided only when its bound is at a kink, read
    /// off `kappa = sigma * S_kk`, the barrier weight times the row's own
    /// diagonal of the reduced matrix. `sigma` equals the curvature
    /// reduced along the coordinate at an exact kink, and `S_kk` is that
    /// reduced curvature's inverse, so `kappa` is 1 there at any
    /// curvature, coupling, or scaling, and it falls as the squared
    /// ratio of kink width to slack away from one. A row below
    /// `KAPPA_MIN` is dropped from the engaged set and its plain
    /// movement stands: its bound is too far from a kink for a pin force
    /// to decide, and the error of leaving it undecided is bounded by its
    /// own slack, order `sqrt(mu)` at the threshold. A coordinate an
    /// equality pins is the limiting case, `S_kk` exactly zero, dropped
    /// by the same test.
    ///
    /// `max_iter` is the total back-solve budget: the all-released solve,
    /// every basis column, and the combined solve that recovers the
    /// direction all count against it. A budget of zero errs before any
    /// work. Any budget above that pays the all-released factorization
    /// first, because which rows engage is only known once that solve has
    /// run, and the shortfall is reported when the basis columns cannot
    /// fit. Either way the caller falls back to the one-sided step.
    /// Returns the direction, the var-x rows held, and the back-solves
    /// spent.
    pub fn parametric_step_directional(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_iter: usize,
    ) -> Result<(Vec<Number>, Vec<usize>, usize), SolverError> {
        use pounce_qp::QpStatus;
        use pounce_qp::options::QpOptions;
        use pounce_qp::problem::{HessianInertia, QpProblem};
        use pounce_qp::solver::{ParametricActiveSetSolver, QpSolver};

        const EPS_REL: Number = 1e-9;
        /// A row whose `kappa = sigma * S_kk` is below this is not at
        /// a kink and is dropped from the QP. `kappa` is 1 at an
        /// exact kink and equals the squared ratio of kink width to
        /// slack, so a row at the threshold sits about 30 widths from
        /// its bound and the cost of deciding it either way is
        /// bounded by that slack. Measured populations: exact fixture
        /// kinks 1.0, held solves near a release 4e-2 and 6e-3,
        /// genuinely interior rows 3e-8 and below, pin-owned rows at
        /// or below zero.
        const KAPPA_MIN: Number = 1e-3;

        let rhs_plain = self.parametric_rhs_full(pin_constraint_indices, deltas)?;
        let weak = self.weakly_active_bounds()?;
        let ctx = self.bound_context(None)?;
        let bs = self.engine;
        let dim = bs.dim();
        let n_x = ctx.n_x;
        let nw = weak.len();
        let mut work = 0usize;
        // A weak bound's slack and multiplier are both of order
        // sqrt(mu) and their uncertainty equals their magnitude, so a
        // movement below sqrt(mu) of the direction's scale cannot be
        // resolved against the bound and does not warrant an exact
        // complementarity decision. The engagement and expansion
        // tests use this band. Acceptance-level roundoff tests keep
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
        // `weak` comes from `weakly_active_bounds`, which is keyed by a
        // var-x row and so can only ever name a bound in the `x` block.
        // Constraint-row limits (gh#928) live in the `s` block, and the
        // directional decision below reads only `x` rows, so a released
        // row outside the `z` blocks would leave a limit enforced in the
        // system it claims to have released. Checked rather than
        // assumed, since it is a property of two index ranges.
        {
            let dims = bs.block_dims();
            let z_lo = dims[0] + dims[1] + dims[2] + dims[3];
            let z_hi = z_lo + dims[4] + dims[5];
            if released.iter().any(|&r| r < z_lo || r >= z_hi) {
                return Err(fail(
                    "a released bound reached the `s` block. The directional \
                     decision is built on `weakly_active_bounds`, whose rows are \
                     var-x, so this cannot happen without that contract having \
                     changed -- and releasing an `s`-block bound needs the slack \
                     diagonal this arm does not carry.",
                ));
            }
        }
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
        let op = bs
            .released_operator(&released)
            .map_err(|m| fail(&format!("released operator unavailable: {m}")))?;
        let mut d0 = vec![0.0; dim];
        // The released operator does nothing to the right-hand side: a
        // weak bound's multiplier is order sqrt(mu) and the released
        // convention holds it at exactly zero, so the step shift's
        // multiplier injection is deliberately omitted, matching
        // `path_direction`'s all-released solve.
        if !op.solve(&rhs_plain, &mut d0) {
            return Err(SolverError::BacksolveFailed);
        }
        work += 1;

        // movement of weak row k under a direction: positive is the
        // feasible side for that row's bound
        let sign = |k: usize| if weak[k].lower { 1.0 } else { -1.0 };
        let movement = |k: usize, d: &[Number]| -> Number { sign(k) * d[weak[k].var_row] };
        // The barrier weight of each weak row's variable, in natural
        // units to match the natural-units response the back-solves
        // return, so `kappa` below is frame-invariant. The classifier
        // succeeded inside `weakly_active_bounds`, so a nonempty weak
        // set implies this call succeeds too.
        // `var_sigma` is a FULL-x array read from a VAR-x row, and the
        // `unwrap_or(0.0)` below turns a miss into a zero that silently
        // drops the row from the engaged set rather than raising. That
        // is the shape gh#672 finding 1 shipped, so the index is typed:
        // `sigma.at` takes a `FullX` and a bare `w.var_row` will not
        // compile. See `crate::index`.
        let nat_sigma: Vec<Number> = {
            let report = self.classify_activity()?;
            let maps = self.maps()?;
            let sigma = FullXSlice::new(&report.var_sigma);
            let map = VarToFull::build(ctx.n_x, |r| maps.var_x_to_full_x[r.get()]);
            weak.iter()
                .map(|w| {
                    map.full_of(VarX::new(w.var_row))
                        .and_then(|full| sigma.at(full))
                        .unwrap_or(0.0)
                })
                .collect()
        };
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
        // A weak row is decided only when its bound is at a kink, and
        // `kappa = sigma * S_kk` measures exactly that: 1 at an exact
        // kink, falling as the squared ratio of kink width to slack away
        // from one. A row below the threshold is dropped and its plain
        // movement stands, since a pin force there holds the coordinate
        // a full slack from where the bound actually is, and the error
        // of not deciding is bounded by that same slack. The limiting
        // cases fall out of the one test: a coordinate an equality owns
        // has `S_kk` exactly zero (its pin absorbs any bound force, and
        // admitting it puts a zero diagonal beside a nonzero gradient,
        // an unbounded QP), and a negative diagonal, which the QP could
        // not bound either, is likewise below the threshold.
        let mut inert: Vec<usize> = Vec::new();
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
                if !op.solve(&unit, &mut xk) {
                    return Err(SolverError::BacksolveFailed);
                }
                work += 1;
                let col: Vec<Number> = weak.iter().map(|w| xk[w.var_row]).collect();
                let own = sign(k) * col[k];
                if nat_sigma[k] * own < KAPPA_MIN {
                    inert.push(k);
                }
                proj[k] = Some(col);
            }
            engaged.retain(|k| !inert.contains(k));
            if engaged.is_empty() {
                return Ok((d[..n_x].to_vec(), Vec::new(), work));
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
                    // S_ij = a_i^T X_j, symmetrized, since S is
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
                // Optimal, so the default stays. The heavy-direction
                // exact decision pays engine refactorizations and is
                // priced accordingly in the docs
                ..QpOptions::default()
            };
            let mut qp_engine =
                ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
            let sol = qp_engine
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
                if !op.solve(&comb, &mut corr) {
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
                if engaged.contains(&k) || inert.contains(&k) {
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
    /// Both [`WEAKLY_ACTIVE`] and [`AMBIGUOUS`] count as weak here,
    /// deliberately: the ambiguous class contains genuine kinks whose
    /// coordinate is coupled to a neighbour (gh#763), so treating it as
    /// "not a kink" would drop real weak rows. That is why the
    /// mislabeling is not a wrong answer in the step path. See
    /// [`Self::reduced_activity`] for the class itself.
    ///
    /// The classifier reports per user variable, in full-x, while the
    /// bound context and the factor's rows are var-x, and the two index
    /// spaces diverge from the first fixed variable on. Each var-x row's
    /// status is read through the same map the classifier scattered
    /// through, so a fixed variable shifts nothing. Using the full-x
    /// index as a factor row instead returns a NEIGHBORING variable's
    /// answer, plausible and wrong, which is the gh#450 hazard the
    /// `primal_row` discipline exists to prevent.
    pub fn weakly_active_bounds(&self) -> Result<Vec<WeakBound>, SolverError> {
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
        let Some(rows) = self.engine.bound_rows() else {
            return Ok(Vec::new());
        };
        // Three index spaces are live in the loop below: `report` is
        // FULL-x, `ctx` is VAR-x, and `br.row` is a bound row. The
        // first two coincide until the first removed variable and
        // diverge after it, so a swap reads a NEIGHBOURING variable's
        // status, in range, plausible, wrong (gh#450, then gh#672
        // finding 1). The typed indices make that a compile error, and
        // `sens_invariance_legs.rs` leg 3 is what covers the site
        // already written. See `crate::index`.
        let map = {
            let maps = self.maps()?;
            VarToFull::build(ctx.n_x, |r| maps.var_x_to_full_x[r.get()])
        };
        let status = FullXSlice::new(&report.var_status);
        let mut out = Vec::new();
        for row in map.rows() {
            let Some(full) = map.full_of(row) else {
                continue;
            };
            let Some(st) = status.at(full) else {
                continue;
            };
            if st != WEAKLY_ACTIVE && st != AMBIGUOUS {
                continue;
            }
            let (s_lo, s_hi) = ctx.slacks_at(row);
            let lower = s_lo <= s_hi;
            // a table lookup, so a space-swap here would fail loudly
            let var_row = row.get();
            // `rows` carries constraint-row limits too, whose `var_row`
            // is an `s`-block row (gh#928). They can never match here,
            // since `map.rows()` runs over the `x` block, so
            // `var_row < ctx.n_x` and every `s` row is at or above it.
            // That is a property of two index ranges not overlapping,
            // which is exactly the kind of thing that silently stops
            // being true, so it is a filter rather than a coincidence.
            debug_assert!(var_row < ctx.n_x);
            if let Some(br) = rows
                .iter()
                .filter(|b| b.var_row < ctx.n_x)
                .find(|b| b.var_row == var_row && b.lower == lower)
            {
                out.push(WeakBound {
                    row: br.row,
                    var_row,
                    lower,
                });
            }
        }
        Ok(out)
    }

    /// The bound geometry both bound-aware steps read: the primal
    /// block's size and base point, its bounds in the model's own units,
    /// the tolerance that decides what counts as on a bound, and the
    /// bound multipliers at the base point.
    ///
    /// Shared rather than assembled twice. The callers have to agree on
    /// all of it, and the unit and index-space conversions below are
    /// exactly what went wrong when a second caller wrote its own.
    ///
    /// `bound_eps` overrides the margin. `None` keeps how far outside
    /// the solve itself was willing to settle, floored so an unrelaxed
    /// solve does not pin on roundoff.
    fn bound_context(&self, bound_eps: Option<Number>) -> Result<BoundContext, SolverError> {
        let dims = self.engine.block_dims();
        let n_x = dims[0];
        let n_s = dims[1];

        // Both primal blocks: `x` against its bounds, and `s` against the
        // inequality rows' limits. The second half is gh#928's subject:
        // an inequality row's limit is a bound on its slack, and a box
        // that stopped at `n_x` left every such limit unwatched.
        let (lo_all, hi_all) = self
            .engine
            .primal_bounds()
            .map_err(SolverError::SensComputationFailed)?;
        if lo_all.len() != n_x + n_s || hi_all.len() != n_x + n_s {
            return Err(SolverError::BadShape {
                what: "primal bounds of the engine",
                got: lo_all.len().min(hi_all.len()),
                expected: n_x + n_s,
            });
        }
        let (mut lo, mut s_lo) = (lo_all[..n_x].to_vec(), lo_all[n_x..].to_vec());
        let (mut hi, mut s_hi) = (hi_all[..n_x].to_vec(), hi_all[n_x..].to_vec());
        let iterate = self.iterate()?;
        let scaling = self.scaling()?;

        // Those bounds bound the engine's `x̃ = d ⊙ x`, while the base
        // point and the step are both in the model's own units (gh#486
        // stage 3). Undo the change of variables on the bounds so all
        // three agree, rather than projecting onto the wrong box. A
        // negative factor reflects the interval, so the sides swap.
        let mut x_curr = iterate[..n_x].to_vec();
        if let Some(d) = scaling.variables.as_deref() {
            for (xi, &di) in x_curr.iter_mut().zip(d) {
                *xi /= di;
            }
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

        // The `s` block gets the same treatment for the row scaling it
        // was solved under. `F`, the vector every back-solve
        // post-multiplies its answer by, is `1/dd_r` on the `s` rows, so
        // a scaled quantity times `F` is the natural one, which is
        // exactly the conversion the `x` block just made by dividing by
        // `d`. Using `F` rather than re-reading the row scaling keeps
        // this pinned to the same numbers the step arrives in: the point
        // of the conversion is that the base point, the box and the step
        // agree, not that the arithmetic matches a formula.
        //
        // The base slack `s* - d_l` is scale-invariant in sign but not in
        // size, and it is compared against a multiplier that gets the
        // same treatment inside the walk, so both sides move together.
        let mut s_curr = iterate[n_x..n_x + n_s].to_vec();
        if let Some(f) = self.engine.natural_units_factor() {
            for i in 0..n_s {
                let fi = f[n_x + i];
                if fi == 1.0 {
                    continue;
                }
                s_curr[i] *= fi;
                let (a, b) = (s_lo[i] * fi, s_hi[i] * fi);
                s_lo[i] = a.min(b);
                s_hi[i] = a.max(b);
            }
        }

        // What counts as outside a bound is the solve's own answer: it
        // was willing to leave a converged point `bound_relax_factor`
        // outside, so anything within that is on the bound, not past
        // it. A floor keeps an unrelaxed solve from pinning on
        // roundoff.
        let floor = crate::boundcheck::release_floor(self.bound_relax_factor);
        // Rejected here rather than at each entry point, so the pyo3
        // binding and every Rust caller get the check the CLI's
        // `sens_bound_eps` gets from its strict lower bound. Zero
        // reinstates the roundoff pinning the floor prevents, and NaN
        // makes `over > eps` false everywhere, so the refinement pins
        // nothing and still reports settled. Both return a plausible
        // vector rather than failing, which is the worse outcome.
        // `> 0.0` is false for NaN, as `pyomo_pounce`'s own check is.
        let eps = match bound_eps {
            None => floor,
            Some(e) if e > 0.0 => e,
            Some(e) => {
                return Err(SolverError::BadOptions(format!(
                    "bound_eps must be a positive number, got {e}"
                )));
            }
        };
        // A caller's `bound_eps` is a primal margin and says nothing
        // about when a multiplier has changed sign, so the release test
        // keeps the solve's own margin.
        let release_eps = floor;
        // The bound multipliers at the base point, with the compound row
        // each one occupies, so a step that drives one negative can
        // release that bound.
        let z_l_off = dims[0] + dims[1] + dims[2] + dims[3];
        let end = z_l_off + dims[4] + dims[5] + dims[6] + dims[7];
        let mults = (z_l_off..end)
            .map(|row| BoundMultiplier {
                row,
                base: iterate[row],
            })
            .collect();
        lo.extend_from_slice(&s_lo);
        hi.extend_from_slice(&s_hi);
        x_curr.extend_from_slice(&s_curr);
        Ok(BoundContext {
            n_x,
            lo,
            hi,
            x_curr,
            eps,
            release_eps,
            mults,
        })
    }

    /// Flat rows of the compound KKT vector holding the equality
    /// multipliers `y_c` for the given 0-based **full-g** constraint
    /// indices. `None` for inequalities, whose multipliers live in the
    /// `y_d` block that [`Self::d_multiplier_rows`] addresses, and for an
    /// index outside the model's rows. Row `r` of a
    /// [`Self::parametric_step_full`] result is then `∂λ_g/∂p · Δp`.
    pub fn g_multiplier_rows(
        &self,
        g_indices: &[Index],
    ) -> Result<Vec<Option<Index>>, SolverError> {
        let dims = self.engine.block_dims();
        let maps = self.maps()?;
        let y_c_offset = (dims[0] + dims[1]) as Index;
        Ok(g_indices
            .iter()
            .map(|&g| block_of(&maps.full_g_to_c, g).map(|pos| y_c_offset + pos))
            .collect())
    }

    /// Flat rows of the compound KKT vector holding the **inequality**
    /// multipliers `y_d` for the given 0-based **full-g** constraint
    /// indices. `None` for equalities, which are
    /// [`Self::g_multiplier_rows`]'s (gh#910).
    ///
    /// **Reading the row is not the same as the row being a
    /// derivative.** `y_d` holds a number for every inequality, in all
    /// three activity regimes, and only one of them has a two-sided
    /// `∂λ/∂p` at all:
    ///
    /// * **strictly active** (`s ≈ 0`, `λ > 0`): the row behaves as an
    ///   equality over a neighbourhood of the solved point, and this row
    ///   is the same back-solve an equality gets. Well defined.
    /// * **inactive** (`s > 0`, `λ ≈ 0`): the derivative is a structural
    ///   zero over a neighbourhood. The KKT row carries the barrier's
    ///   residue rather than a derivative of anything.
    /// * **weakly active** (a kink: `s ≈ 0` *and* `λ ≈ 0`): the two
    ///   one-sided derivatives differ and no two-sided value exists. The
    ///   entry holds whichever side the factorization landed on.
    ///   [`Self::parametric_step_directional`] is what answers a kink,
    ///   and it needs a direction.
    ///
    /// So a caller reading these rows as `∂λ/∂p` must gate on the regime
    /// first, and the classifier that answers it is
    /// [`Self::reduced_row_activity`], **not**
    /// [`Self::classify_activity`]: a genuine kink whose row couples to
    /// the remaining free space reports `INACTIVE` on the directional
    /// normalizer at strong enough coupling (gh#804), and `INACTIVE` is
    /// the one class whose derivative a caller may legitimately read as a
    /// structural zero (gh#756).
    pub fn d_multiplier_rows(
        &self,
        g_indices: &[Index],
    ) -> Result<Vec<Option<Index>>, SolverError> {
        let dims = self.engine.block_dims();
        let maps = self.maps()?;
        let y_d_offset = (dims[0] + dims[1] + dims[2]) as Index;
        Ok(g_indices
            .iter()
            .map(|&g| block_of(&maps.full_g_to_d, g).map(|pos| y_d_offset + pos))
            .collect())
    }

    /// Flat rows of the compound KKT vector holding an inequality's
    /// **slack** `s`, for the given 0-based full-g row indices, `None`
    /// for a row that is not an inequality.
    ///
    /// The primal counterpart of [`Self::d_multiplier_rows`], and the
    /// discriminator a consumer of [`Self::parametric_step_path`] needs.
    /// A limit written as `g(x) <= cap` bounds this slack rather than any
    /// variable, so a breakpoint on it carries a primal KKT row in the
    /// `s` block (gh#928). Reading such a row as a var-x index returns a
    /// neighbouring variable's answer, the gh#450 hazard, so a caller that
    /// maps rows back to model objects resolves them here.
    pub fn d_slack_rows(&self, g_indices: &[Index]) -> Result<Vec<Option<Index>>, SolverError> {
        let maps = self.maps()?;
        let s_offset = self.engine.block_dims()[0] as Index;
        Ok(g_indices
            .iter()
            .map(|&g| block_of(&maps.full_g_to_d, g).map(|pos| s_offset + pos))
            .collect())
    }

    /// Flat rows of the compound KKT vector holding the primal values
    /// `x` for the given 0-based **full-x** variable indices. `None`
    /// where the engine removed the column, which has no row in the
    /// factor at all.
    ///
    /// The `x` counterpart of [`Self::g_multiplier_rows`], and needed for
    /// the same reason: a caller holding user-space indices cannot index
    /// the factor with them directly. Row `r` of a
    /// [`Self::parametric_step_full`] result is then `∂x/∂p · Δp` for that
    /// variable, and `e_r` is the unit vector selecting its column in a
    /// [`Self::kkt_solve`].
    pub fn x_primal_rows(&self, x_indices: &[Index]) -> Result<Vec<Option<Index>>, SolverError> {
        let maps = self.maps()?;
        let n_full = maps.n_full_x as Index;
        // out of range must not masquerade as "removed as fixed": the
        // map answers None for both, and the caller's whole reason for
        // asking is that it cannot tell the spaces apart itself
        if let Some(&bad) = x_indices.iter().find(|&&i| i < 0 || i >= n_full) {
            return Err(SolverError::BadShape {
                what: "x_primal_rows variable index",
                got: bad as usize,
                expected: n_full as usize,
            });
        }
        let to_var = maps.full_x_to_var_x();
        // the x block starts at flat index 0, so the var-x position IS
        // the KKT row
        Ok(x_indices
            .iter()
            .map(|&i| to_var[i as usize].map(|v| v as Index))
            .collect())
    }

    /// Flat rows of the compound KKT vector holding the bound
    /// multipliers `(z_l, z_u)` for the given 0-based **full-x** variable
    /// indices. Each side is `None` where the variable has no finite
    /// bound on that side, or where the engine removed the column, which
    /// has no row at all.
    ///
    /// The `z_l` block is NOT indexed by variable: it carries one entry
    /// per variable with a finite lower bound, so `z_l_offset + i` names
    /// variable `i`'s multiplier only on a model where every variable
    /// ahead of it is lower-bounded. Any free variable ahead of it shifts
    /// the entry onto a neighbour's, in range and plausible. Errors with
    /// `BadShape` on an out-of-range index, and with
    /// `SensComputationFailed` when the engine does not report its bound
    /// rows.
    pub fn bound_multiplier_rows(
        &self,
        x_indices: &[Index],
    ) -> Result<Vec<(Option<Index>, Option<Index>)>, SolverError> {
        let maps = self.maps()?;
        let n_full = maps.n_full_x as Index;
        if let Some(&bad) = x_indices.iter().find(|&&i| i < 0 || i >= n_full) {
            return Err(SolverError::BadShape {
                what: "bound_multiplier_rows variable index",
                got: bad as usize,
                expected: n_full as usize,
            });
        }
        let dims = self.engine.block_dims();
        let z_lo = dims[0] + dims[1] + dims[2] + dims[3];
        let z_hi = z_lo + dims[4] + dims[5];
        let rows = self.engine.bound_rows().ok_or_else(|| {
            SolverError::SensComputationFailed(
                "bound_multiplier_rows: bound expansions are not recoverable".into(),
            )
        })?;
        // var-x row -> (z_l row, z_u row). Only the z blocks: a v-block
        // entry's `var_row` is an s-block row, a valid index into nothing.
        let mut by_var: Vec<(Option<Index>, Option<Index>)> = vec![(None, None); dims[0]];
        for br in rows {
            if br.row < z_lo || br.row >= z_hi || br.var_row >= dims[0] {
                continue;
            }
            let slot = &mut by_var[br.var_row];
            if br.lower {
                slot.0 = Some(br.row as Index);
            } else {
                slot.1 = Some(br.row as Index);
            }
        }
        let to_var = maps.full_x_to_var_x();
        // a table lookup through the removed-variable map, never `i` itself
        Ok(x_indices
            .iter()
            .map(|&i| to_var[i as usize].map_or((None, None), |v| by_var[v]))
            .collect())
    }

    /// The model's variable count: the length of a full-x report and the
    /// domain of [`Self::x_primal_rows`].
    pub fn n_full_x(&self) -> Result<usize, SolverError> {
        Ok(self.maps()?.n_full_x)
    }

    /// The model's constraint count: the length of a full-g report and
    /// the domain of [`Self::reduced_row_activity`].
    pub fn n_full_g(&self) -> Result<usize, SolverError> {
        Ok(self.maps()?.n_full_g())
    }

    /// Reduced Hessian over the pinned equality-constraint rows:
    /// `obj_scal · B K⁻¹ Bᵀ`, where `B` selects the
    /// `pin_constraint_indices` rows of the y_c block and `K` is the
    /// **natural-units** KKT matrix, so `−inv` of the returned matrix is
    /// directly the parameter covariance whatever scaling the engine
    /// applied (pounce#128). `obj_scal` is a plain extra multiplier.
    /// Returns the `n²`-long column-major dense matrix
    /// (`n = pin_constraint_indices.len()`).
    ///
    /// # Sign convention: this returns `−H_R`, not `H_R` (gh#937)
    ///
    /// The matrix is the **negated** reduced Hessian. On a model whose
    /// objective Hessian is `[[2, 1], [1, 2]]` with both variables
    /// pinned, this returns `[[−2, −1], [−1, −2]]`. So a well-posed
    /// minimum reports an all-*negative* spectrum, which is the
    /// convention, not an indefiniteness or convergence bug.
    ///
    /// The minus is the augmented system's: pin indices map to the `y_c`
    /// multiplier block, and for `K = [[H, Aᵀ], [A, 0]]` the `(y_c, y_c)`
    /// block of `K⁻¹` is `−(A H⁻¹ Aᵀ)⁻¹`, so over pin rows `B K⁻¹ Bᵀ` is
    /// the multiplier sensitivity `∂λ/∂p = −∂²f*/∂p²`. Negate to read
    /// curvature: `−hr` is `H_R`, and `−inv(hr)` is the covariance.
    pub fn compute_reduced_hessian(
        &self,
        pin_constraint_indices: &[Index],
        obj_scal: Number,
    ) -> Result<Vec<Number>, SolverError> {
        let n = pin_constraint_indices.len();
        let opts = SensOptions {
            compute_red_hessian: true,
            obj_scal,
            ..SensOptions::default()
        };
        let mut sens_app = self.sens_app(pin_constraint_indices, opts)?;
        let mut hr = vec![0.0; n * n];
        if !sens_app.compute_reduced_hessian(&mut hr) {
            return Err(SolverError::SensComputationFailed(
                "SensApplication::compute_reduced_hessian failed".into(),
            ));
        }
        Ok(hr)
    }

    /// [`Self::compute_reduced_hessian`] plus its eigendecomposition,
    /// `(H_R, eigenvalues, eigenvectors)`, from
    /// [`pounce_linalg::symmetric_eigen`].
    ///
    /// Eigenvectors are column-major, length `n²`, column `j` belonging
    /// to eigenvalue `j`, and sign-pinned by `symmetric_eigen` so a
    /// column read as a direction reproduces across builds.
    ///
    /// # This is the spectrum of `−H_R`, so ascending runs stiffest first
    ///
    /// The eigenvalues are those of the **negated** reduced Hessian, in
    /// ascending order, which on `−H_R` runs from most negative to least,
    /// i.e. **stiffest mode first and softest last**. On
    /// `H = [[2, 1], [1, 2]]` fully pinned they come back `[−3, −1]`.
    /// Either negate the eigenvalues and reverse the order, or read the
    /// *trailing* columns as the soft modes.
    pub fn compute_reduced_hessian_eigen(
        &self,
        pin_constraint_indices: &[Index],
        obj_scal: Number,
    ) -> Result<(Vec<Number>, Vec<Number>, Vec<Number>), SolverError> {
        let hr = self.compute_reduced_hessian(pin_constraint_indices, obj_scal)?;
        let n = pin_constraint_indices.len();
        let mut vals = vec![0.0; n];
        let mut vecs = vec![0.0; n * n];
        if !pounce_linalg::symmetric_eigen(&hr, n, &mut vals, &mut vecs) {
            return Err(SolverError::SensComputationFailed(
                "the reduced Hessian's eigendecomposition did not converge".into(),
            ));
        }
        Ok((hr, vals, vecs))
    }

    /// The reduced Hessian as the engine's own scaled space sees it:
    /// `H̃_ij = (df / (dc_i·dc_j)) · H_ij`. Identical to
    /// [`Self::compute_reduced_hessian`] when no scaling is active.
    ///
    /// Sign: this is [`Self::compute_reduced_hessian`]'s `−H_R`
    /// multiplied through by `df / (dc_i·dc_j)` (gh#937), so unlike the
    /// natural-units value its orientation is **not** fixed: under
    /// `obj_scaling_factor = −1`, `df` carries the minus that makes a
    /// maximization a minimization.
    pub fn compute_reduced_hessian_scaled(
        &self,
        pin_constraint_indices: &[Index],
        obj_scal: Number,
    ) -> Result<Vec<Number>, SolverError> {
        let mut hr = self.compute_reduced_hessian(pin_constraint_indices, obj_scal)?;
        let df = self.scaling()?.objective;
        let dc = self.pin_g_scaling(pin_constraint_indices)?;
        crate::reduced_hessian::scale_to_solver_space(&mut hr, df, &dc);
        Ok(hr)
    }

    /// The scaling the engine applied: `(objective, equality rows,
    /// inequality rows)`. `(1.0, None, None)` means no scaling was
    /// active.
    pub fn nlp_scaling(
        &self,
    ) -> Result<(Number, Option<Vec<Number>>, Option<Vec<Number>>), SolverError> {
        let s = self.scaling()?;
        Ok((
            s.objective,
            s.equality_rows.clone(),
            s.inequality_rows.clone(),
        ))
    }

    /// The inertia-correction perturbations `(δ_x, δ_s, δ_c, δ_d)` in the
    /// engine's factor, or `None` when the engine does not report them.
    /// See [`BarrierIterate::kkt_perturbations`].
    pub fn kkt_perturbations(&self) -> Option<[Number; 4]> {
        self.engine.kkt_perturbations()
    }

    /// Per-pin equality-row scaling factors `dc_i` (1.0 entries when no
    /// row scaling is active), ordered like `pin_constraint_indices`.
    pub fn pin_g_scaling(
        &self,
        pin_constraint_indices: &[Index],
    ) -> Result<Vec<Number>, SolverError> {
        Ok(self.pin_rows_and_scales(pin_constraint_indices)?.1)
    }
}

/// A model row's position in one row block, `None` for a row in the other
/// block or outside the model.
fn block_of(map: &[Option<usize>], g: Index) -> Option<Index> {
    usize::try_from(g)
        .ok()
        .and_then(|g| map.get(g).copied().flatten())
        .map(|p| p as Index)
}
