//! Per-variable scaling as a TNLP wrapper (issue #486 stage 2).
//!
//! The solver's own scaling handles the objective and the constraint
//! rows but has no representation for a change of variables, so a
//! per-variable `scaling_factor` reached the core and was rejected
//! (issue #483 stage 1). This wrapper supplies the missing piece by
//! substituting variables one level below the algorithm, the same way
//! `PresolveTnlp` transforms coordinates and inverts the transform in
//! `finalize_solution`.
//!
//! # Convention
//!
//! `d_i > 0` is the factor for variable `i`, following Pyomo's
//! `scaling_factor` suffix: the scaled variable is `x̃ = d ⊙ x`, so
//! the algorithm sees `x̃` and the inner TNLP always sees `x = x̃ ⊘ d`.
//!
//! # Transforms
//!
//! Diagonal scaling leaves sparsity unchanged, so every transform is
//! an elementwise multiply or divide:
//!
//! | quantity | transform |
//! |---|---|
//! | starting point | `× d` |
//! | bounds, where present | `× d` |
//! | objective, constraint values | unchanged |
//! | objective gradient | `÷ d` |
//! | Jacobian entry | `÷ d[col]` |
//! | Hessian entry | `÷ (d[row] · d[col])` |
//! | solution `x` | `÷ d` |
//! | bound multipliers `z_L`, `z_U` | `× d` on the way out, `÷ d` in |
//! | constraint multipliers `λ` | unchanged |
//!
//! "Where present" is the whole of the bounds subtlety. An absent
//! bound arrives as the `nlp_lower_bound_inf` / `nlp_upper_bound_inf`
//! sentinel, an ordinary finite number, so scaling it would move it
//! inside the threshold and turn a free variable into a bounded one.
//! Absent bounds are passed through untouched, and a present bound
//! that would scale *past* a threshold is refused in
//! [`wrap_with_scaling`].
//!
//! The multiplier rule follows from stationarity. Writing the scaled
//! condition `∇f̃ - J̃ᵀλ - z̃_L + z̃_U = 0` and substituting
//! `∇f̃ = ∇f ⊘ d`, `J̃ᵀλ = (Jᵀλ) ⊘ d` gives `z_L = d ⊙ z̃_L`, with `λ`
//! untouched because `g` is untouched.
//!
//! # Factors come from the inner TNLP
//!
//! [`wrap_with_scaling`] calls the inner TNLP's
//! [`TNLP::get_scaling_parameters`] once, keeps the variable factors
//! for itself, and forwards the objective and constraint factors
//! upward unchanged, since the core already models those. Without
//! non-trivial variable factors it returns `Ok(None)` and the caller
//! keeps the bare TNLP, so an unscaled solve pays nothing.
//!
//! # Where this sits
//!
//! `optimize_tnlp` installs the wrapper around whatever TNLP it was
//! handed, so with presolve in play the wrapper is *outermost* and the
//! factors it reads are already in reduced variable indices: the CLI
//! builds a `PresolveTnlp` first and hands that over. That is
//! consistent rather than accidental, because the presolve layers
//! project the factors along with everything else. `linear_eq_elim`
//! maps each surviving column's factor through `vars_kept`, and the
//! row-only reducer leaves columns alone. `finalize_solution` then
//! unscales in those same reduced indices, before the presolve layer
//! beneath expands back to the full variable vector.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::types::{Index, Number};

use crate::tnlp::{
    BoundsInfo, IndexStyle, InfeasibilityProof, IpoptCq, IpoptData, IterStats, MetaData, NlpInfo,
    ScalingRequest, Solution, SparsityRequest, StartingPoint, TNLP,
};

/// Smallest variable factor accepted. Anything below this makes the
/// substitution numerically meaningless in the direction it scales.
const MIN_FACTOR: Number = 1e-12;

/// A TNLP presenting `x̃ = d ⊙ x` to the algorithm. See the module
/// documentation for the transform table.
pub struct ScalingTnlp {
    inner: Rc<RefCell<dyn TNLP>>,
    /// Per-variable factors, all finite and `>= MIN_FACTOR`.
    d: Vec<Number>,
    /// Objective and constraint scaling read off the inner TNLP and
    /// passed upward for the core to apply.
    obj_scaling: Number,
    g_scaling: Option<Vec<Number>>,
    /// Column of each Jacobian nonzero, and (row, column) of each
    /// Hessian nonzero, captured on the structure call so the values
    /// call knows which factors to divide by. Zero-based regardless
    /// of the inner TNLP's index style.
    jac_cols: Vec<usize>,
    hess_rc: Vec<(usize, usize)>,
    index_offset: usize,
    /// Scratch for the unscaled `x` handed to the inner TNLP. Moved
    /// out and back by [`ScalingTnlp::with_unscaled`] so it keeps its
    /// capacity across iterations.
    x_scratch: Vec<Number>,
    /// `nlp_lower_bound_inf` / `nlp_upper_bound_inf`. A bound at or
    /// past its threshold means "absent", and an absent bound has to
    /// stay absent under scaling: multiplying the `±1e19` sentinel by
    /// a factor below 1 would move it inside the threshold and hand a
    /// free variable a barrier term it should not have.
    lo_inf: Number,
    up_inf: Number,
}

impl ScalingTnlp {
    /// Fill the Jacobian column cache from the inner TNLP if a values
    /// call arrives before a structure call. The algorithm always asks
    /// for structure first, but a wrapper that panics when it does not
    /// is worse than one that asks for itself.
    fn ensure_jac_cols(&mut self) -> bool {
        if !self.jac_cols.is_empty() {
            return true;
        }
        let Some(info) = self.inner.borrow_mut().get_nlp_info() else {
            return false;
        };
        let nnz = info.nnz_jac_g as usize;
        let (mut irow, mut jcol) = (vec![0; nnz], vec![0; nnz]);
        let ok = self.inner.borrow_mut().eval_jac_g(
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        );
        if ok {
            self.jac_cols = jcol
                .iter()
                .map(|&c| c as usize - self.index_offset)
                .collect();
        }
        ok
    }

    /// The Hessian counterpart of [`Self::ensure_jac_cols`].
    fn ensure_hess_rc(&mut self) -> bool {
        if !self.hess_rc.is_empty() {
            return true;
        }
        let Some(info) = self.inner.borrow_mut().get_nlp_info() else {
            return false;
        };
        let nnz = info.nnz_h_lag as usize;
        let (mut irow, mut jcol) = (vec![0; nnz], vec![0; nnz]);
        let ok = self.inner.borrow_mut().eval_h(
            None,
            true,
            1.0,
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        );
        if ok {
            self.hess_rc = irow
                .iter()
                .zip(jcol.iter())
                .map(|(&r, &c)| {
                    (
                        r as usize - self.index_offset,
                        c as usize - self.index_offset,
                    )
                })
                .collect();
        }
        ok
    }

    /// Call `f` with the inner TNLP seeing `x` in the user's own
    /// coordinates. The scratch buffer is moved out for the duration
    /// so `f` can still borrow `self`, then moved back, which keeps
    /// its capacity instead of allocating once per evaluation.
    ///
    /// Everything about this wrapper rests on `x` and `d` indexing the
    /// same variables, so a length mismatch is a wiring bug and is
    /// asserted rather than papered over: truncating would hand back
    /// quietly wrong derivatives.
    fn with_unscaled<R>(&mut self, x: &[Number], f: impl FnOnce(&mut Self, &[Number]) -> R) -> R {
        assert_eq!(
            x.len(),
            self.d.len(),
            "scaling: got {} variables but {} factors",
            x.len(),
            self.d.len()
        );
        let mut buf = std::mem::take(&mut self.x_scratch);
        buf.clear();
        buf.extend(x.iter().zip(self.d.iter()).map(|(&v, &s)| v / s));
        let out = f(self, &buf);
        self.x_scratch = buf;
        out
    }
}

/// Wrap `inner` when it requests non-trivial per-variable scaling.
///
/// Returns `Ok(None)` when no variable scaling is requested, in which
/// case the caller should use `inner` unchanged. Returns `Err` when a
/// factor is not finite, not positive, or below [`MIN_FACTOR`]: a
/// negative factor would reverse a variable's direction and swap its
/// bounds, which no caller has asked for and which is better refused
/// than silently applied.
pub fn wrap_with_scaling(
    inner: Rc<RefCell<dyn TNLP>>,
    lo_inf: Number,
    up_inf: Number,
) -> Result<Option<Rc<RefCell<dyn TNLP>>>, String> {
    let info = inner
        .borrow_mut()
        .get_nlp_info()
        .ok_or_else(|| "scaling: get_nlp_info failed on the inner problem".to_string())?;
    let n = info.n as usize;
    let m = info.m as usize;

    let mut obj_scaling: Number = 1.0;
    let mut use_x = false;
    let mut x_scaling = vec![1.0; n];
    let mut use_g = false;
    let mut g_scaling = vec![1.0; m];
    let supplied = inner.borrow_mut().get_scaling_parameters(ScalingRequest {
        obj_scaling: &mut obj_scaling,
        use_x_scaling: &mut use_x,
        x_scaling: &mut x_scaling,
        use_g_scaling: &mut use_g,
        g_scaling: &mut g_scaling,
    });
    if !supplied || !use_x {
        return Ok(None);
    }
    if x_scaling.len() != n {
        return Err(format!(
            "scaling: x_scaling has {} entries for {n} variables",
            x_scaling.len()
        ));
    }
    if x_scaling.iter().all(|&s| s == 1.0) {
        return Ok(None);
    }
    for (i, &s) in x_scaling.iter().enumerate() {
        if !s.is_finite() || s <= 0.0 {
            return Err(format!(
                "scaling: variable {i} has scaling factor {s}; factors must be \
                 finite and positive (a negative factor would reverse the \
                 variable and swap its bounds)"
            ));
        }
        if s < MIN_FACTOR {
            return Err(format!(
                "scaling: variable {i} has scaling factor {s}, below the {MIN_FACTOR:e} \
                 floor; such a factor makes the substitution numerically meaningless"
            ));
        }
    }

    // A present bound scaled past its threshold would read as absent
    // and disappear, the mirror of the sentinel problem the wrapper's
    // `get_bounds_info` guards against. It cannot be reported from
    // there, because that callback returns a bare bool, so the bounds
    // are fetched once here where there is a message channel.
    {
        let mut x_l = vec![0.0; n];
        let mut x_u = vec![0.0; n];
        let mut g_l = vec![0.0; m];
        let mut g_u = vec![0.0; m];
        let got = inner.borrow_mut().get_bounds_info(BoundsInfo {
            x_l: &mut x_l,
            x_u: &mut x_u,
            g_l: &mut g_l,
            g_u: &mut g_u,
        });
        if got {
            for (i, &s) in x_scaling.iter().enumerate() {
                if x_l[i] > lo_inf && x_l[i] * s <= lo_inf {
                    return Err(format!(
                        "scaling: variable {i} has lower bound {} and scaling factor \
                         {s}; the scaled bound {} is at or past nlp_lower_bound_inf \
                         ({lo_inf}), where it would be read as no bound at all",
                        x_l[i],
                        x_l[i] * s
                    ));
                }
                if x_u[i] < up_inf && x_u[i] * s >= up_inf {
                    return Err(format!(
                        "scaling: variable {i} has upper bound {} and scaling factor \
                         {s}; the scaled bound {} is at or past nlp_upper_bound_inf \
                         ({up_inf}), where it would be read as no bound at all",
                        x_u[i],
                        x_u[i] * s
                    ));
                }
            }
        }
    }

    let index_offset = match info.index_style {
        IndexStyle::C => 0,
        IndexStyle::Fortran => 1,
    };
    Ok(Some(Rc::new(RefCell::new(ScalingTnlp {
        inner,
        d: x_scaling,
        obj_scaling,
        g_scaling: if use_g && g_scaling.len() == m {
            Some(g_scaling)
        } else {
            None
        },
        jac_cols: Vec::new(),
        hess_rc: Vec::new(),
        index_offset,
        x_scratch: Vec::with_capacity(n),
        lo_inf,
        up_inf,
    }))))
}

/// Undo the substitution on an iterate read *below* the wrapper — the
/// algorithm's own `x̃`, `z̃_L`, `z̃_U`, already lifted to the full
/// variable vector — by the rule [`ScalingTnlp::finalize_solution`]
/// applies: `x = x̃ ⊘ d`, `z = d ⊙ z̃`.
///
/// For readers that hand an iterate to the user without passing through
/// `finalize_solution`: the intermediate-callback window
/// (`GetIpoptCurrentIterate`, the SQP arm's user iterate). `d` is
/// [`factors_of`] / `IpoptNlp::variable_scaling`; `None`, or a length that
/// does not match a vector, leaves that vector alone (an empty slice
/// means "not requested"). Constraint values and multipliers are not
/// moved by the substitution and are not taken.
pub fn unscale_iterate(
    d: Option<&[Number]>,
    x: &mut [Number],
    z_l: &mut [Number],
    z_u: &mut [Number],
) {
    let Some(d) = d else { return };
    if x.len() == d.len() {
        for (v, &s) in x.iter_mut().zip(d) {
            *v /= s;
        }
    }
    for z in [z_l, z_u] {
        if z.len() == d.len() {
            for (v, &s) in z.iter_mut().zip(d) {
                *v *= s;
            }
        }
    }
}

/// The factors a [`ScalingTnlp`] applies, if `tnlp` is one.
///
/// Lets a caller that reads the algorithm's own iterate rather than
/// the `finalize_solution` payload undo the substitution itself.
pub fn factors_of(tnlp: &Rc<RefCell<dyn TNLP>>) -> Option<Vec<Number>> {
    tnlp.borrow().scaling_factors()
}

impl TNLP for ScalingTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.borrow_mut().get_nlp_info()
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        let BoundsInfo { x_l, x_u, g_l, g_u } = b;
        let ok = self
            .inner
            .borrow_mut()
            .get_bounds_info(BoundsInfo { x_l, x_u, g_l, g_u });
        if !ok {
            return false;
        }
        // Positive factors preserve the order of the bounds, so no
        // swap is needed. Absence is directional and decided against
        // the thresholds, matching `classify_bounds`: a lower bound is
        // absent at or below `lo_inf`, an upper bound at or above
        // `up_inf`. Those sentinels are ordinary finite numbers
        // (`±1e19` by default, and the `.nl` reader writes them rather
        // than a true infinity), so scaling one would turn "no bound"
        // into a real bound.
        for (i, &s) in self.d.iter().enumerate() {
            if x_l[i] > self.lo_inf {
                x_l[i] *= s;
            }
            if x_u[i] < self.up_inf {
                x_u[i] *= s;
            }
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        let StartingPoint {
            init_x,
            x,
            init_z,
            z_l,
            z_u,
            init_lambda,
            lambda,
        } = sp;
        let ok = self.inner.borrow_mut().get_starting_point(StartingPoint {
            init_x,
            x,
            init_z,
            z_l,
            z_u,
            init_lambda,
            lambda,
        });
        if !ok {
            return false;
        }
        if init_x {
            for (i, &s) in self.d.iter().enumerate() {
                x[i] *= s;
            }
        }
        if init_z {
            // z_L = d ⊙ z̃_L, so the warm start divides on the way in.
            for (i, &s) in self.d.iter().enumerate() {
                z_l[i] /= s;
                z_u[i] /= s;
            }
        }
        let _ = (init_lambda, lambda); // λ is unchanged by the substitution
        true
    }

    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.with_unscaled(x, |me, xu| me.inner.borrow_mut().eval_f(xu, new_x))
    }

    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        if !self.with_unscaled(x, |me, xu| {
            me.inner.borrow_mut().eval_grad_f(xu, new_x, grad_f)
        }) {
            return false;
        }
        for (i, &s) in self.d.iter().enumerate() {
            grad_f[i] /= s;
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        self.with_unscaled(x, |me, xu| me.inner.borrow_mut().eval_g(xu, new_x, g))
    }

    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let ok = self.inner.borrow_mut().eval_jac_g(
                    x,
                    new_x,
                    SparsityRequest::Structure { irow, jcol },
                );
                if ok {
                    // Remember each nonzero's column so the values
                    // call knows which factor divides it.
                    self.jac_cols = jcol
                        .iter()
                        .map(|&c| c as usize - self.index_offset)
                        .collect();
                }
                ok
            }
            SparsityRequest::Values { values } => {
                let ok = match x {
                    Some(xs) => self.with_unscaled(xs, |me, xu| {
                        me.inner.borrow_mut().eval_jac_g(
                            Some(xu),
                            new_x,
                            SparsityRequest::Values {
                                values: &mut *values,
                            },
                        )
                    }),
                    None => self.inner.borrow_mut().eval_jac_g(
                        None,
                        new_x,
                        SparsityRequest::Values {
                            values: &mut *values,
                        },
                    ),
                };
                if !ok || !self.ensure_jac_cols() {
                    return false;
                }
                for (k, v) in values.iter_mut().enumerate() {
                    *v /= self.d[self.jac_cols[k]];
                }
                true
            }
        }
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let ok = self.inner.borrow_mut().eval_h(
                    x,
                    new_x,
                    obj_factor,
                    lambda,
                    new_lambda,
                    SparsityRequest::Structure { irow, jcol },
                );
                if ok {
                    self.hess_rc = irow
                        .iter()
                        .zip(jcol.iter())
                        .map(|(&r, &c)| {
                            (
                                r as usize - self.index_offset,
                                c as usize - self.index_offset,
                            )
                        })
                        .collect();
                }
                ok
            }
            SparsityRequest::Values { values } => {
                let ok = match x {
                    Some(xs) => self.with_unscaled(xs, |me, xu| {
                        me.inner.borrow_mut().eval_h(
                            Some(xu),
                            new_x,
                            obj_factor,
                            lambda,
                            new_lambda,
                            SparsityRequest::Values {
                                values: &mut *values,
                            },
                        )
                    }),
                    None => self.inner.borrow_mut().eval_h(
                        None,
                        new_x,
                        obj_factor,
                        lambda,
                        new_lambda,
                        SparsityRequest::Values {
                            values: &mut *values,
                        },
                    ),
                };
                if !ok || !self.ensure_hess_rc() {
                    return false;
                }
                for (k, v) in values.iter_mut().enumerate() {
                    let (r, c) = self.hess_rc[k];
                    *v /= self.d[r] * self.d[c];
                }
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        assert_eq!(
            sol.x.len(),
            self.d.len(),
            "scaling: solution has {} variables but {} factors",
            sol.x.len(),
            self.d.len()
        );
        let x: Vec<Number> = sol
            .x
            .iter()
            .zip(self.d.iter())
            .map(|(&v, &s)| v / s)
            .collect();
        let z_l: Vec<Number> = sol
            .z_l
            .iter()
            .zip(self.d.iter())
            .map(|(&v, &s)| v * s)
            .collect();
        let z_u: Vec<Number> = sol
            .z_u
            .iter()
            .zip(self.d.iter())
            .map(|(&v, &s)| v * s)
            .collect();
        self.inner.borrow_mut().finalize_solution(
            Solution {
                status: sol.status,
                x: &x,
                z_l: &z_l,
                z_u: &z_u,
                g: sol.g,
                lambda: sol.lambda,
                obj_value: sol.obj_value,
            },
            ip_data,
            ip_cq,
        );
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        // The variable factors were consumed at construction; the
        // objective and constraint factors belong to the core, which
        // already models them.
        *req.obj_scaling = self.obj_scaling;
        *req.use_x_scaling = false;
        match &self.g_scaling {
            Some(g) if g.len() == req.g_scaling.len() => {
                req.g_scaling.copy_from_slice(g);
                *req.use_g_scaling = true;
            }
            _ => *req.use_g_scaling = false,
        }
        true
    }

    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        self.inner.borrow_mut().get_var_con_metadata(var, con)
    }

    fn finalize_metadata(&mut self, var: &MetaData, con: &MetaData) {
        self.inner.borrow_mut().finalize_metadata(var, con);
    }

    fn get_variables_linearity(&mut self, types: &mut [crate::tnlp::Linearity]) -> bool {
        // A change of variables by a positive diagonal preserves
        // linearity of every row and of the objective.
        self.inner.borrow_mut().get_variables_linearity(types)
    }

    fn get_constraints_linearity(&mut self, types: &mut [crate::tnlp::Linearity]) -> bool {
        self.inner.borrow_mut().get_constraints_linearity(types)
    }

    fn get_objective_variables_linearity(&mut self, types: &mut [crate::tnlp::Linearity]) -> bool {
        self.inner
            .borrow_mut()
            .get_objective_variables_linearity(types)
    }

    fn derivative_proofs(&mut self) -> crate::constant_derivatives::DerivativeProofs {
        // Every transform this wrapper applies is a multiply by a
        // constant factor (`∇f ÷ d`, `J[i,j] ÷ d[j]`, `H[i,j] ÷
        // d[i]·d[j]`), and the row order is untouched — so a derivative
        // that was constant below stays constant above, and one that
        // was proved to vary still varies. Forwarding is sound in both
        // directions, which is the condition the trait states.
        self.inner.borrow_mut().derivative_proofs()
    }

    fn scaling_factors(&self) -> Option<Vec<Number>> {
        Some(self.d.clone())
    }

    fn is_presolve_wrapper(&self) -> bool {
        // Forward rather than claim: this flag asks whether a generic
        // PRESOLVE wrapper sits below, so that the solve entry point
        // does not add a second one. Answering `true` here would make
        // presolve decline every scaled problem.
        self.inner.borrow().is_presolve_wrapper()
    }

    // The forwards below were missing, so the trait defaults answered for
    // the user's problem whenever per-variable factors installed this
    // wrapper: no intermediate callback ever fired (on either arm), a
    // `User_Requested_Stop` from it could not happen, a declared
    // nonlinear-variable subset was dropped from the limited-memory
    // Hessian, and a presolve infeasibility proof below was lost. None of
    // them carries a value the substitution moves — the stats are the
    // algorithm's own, and variable indices are unchanged by a diagonal
    // scaling — so each is a plain forward. The iterate a callback reads
    // through `GetIpoptCurrentIterate` / `with_current_user_iterate` is
    // unscaled by its readers, which see the factors via
    // `scaling_factors`.

    fn get_number_of_nonlinear_variables(&mut self) -> Index {
        self.inner.borrow_mut().get_number_of_nonlinear_variables()
    }

    fn get_list_of_nonlinear_variables(&mut self, pos_nonlin_vars: &mut [Index]) -> bool {
        self.inner
            .borrow_mut()
            .get_list_of_nonlinear_variables(pos_nonlin_vars)
    }

    fn wants_intermediate_callback(&self) -> bool {
        self.inner.borrow().wants_intermediate_callback()
    }

    fn wants_current_iterate(&self) -> bool {
        self.inner.borrow().wants_current_iterate()
    }

    fn update_with_barrier(&mut self, mu: Number) -> bool {
        self.inner.borrow_mut().update_with_barrier(mu)
    }

    fn intermediate_callback(
        &mut self,
        stats: IterStats,
        ip_data: &IpoptData,
        ip_cq: &IpoptCq,
    ) -> bool {
        // The live iterate arrives in this layer's `x̃ = d ⊙ x` coordinates —
        // the presolve layers beneath this wrapper (it is outermost; see the
        // module doc) lifted it in those coordinates — and leaves in the inner TNLP's,
        // by `finalize_solution`'s rule: `x = x̃ ⊘ d`, `z = d ⊙ z̃`. The
        // algorithm's scaled space holds `x̃`, so a column it holds gains
        // the factor `d` in the iterate's scaling. Done here, not where the
        // algorithm builds the iterate: with presolve between the two the
        // algorithm's NLP cannot see these factors, and the user read `x̃`.
        let lifted = crate::live_iterate::current_user_iterate()
            .filter(|it| it.x.len() == self.d.len())
            .map(|mut it| {
                unscale_iterate(Some(&self.d), &mut it.x, &mut it.z_l, &mut it.z_u);
                // `∂L/∂x = d ⊙ ∂L/∂x̃`, the bound multipliers' rule.
                unscale_iterate(Some(&self.d), &mut [], &mut it.grad_lag_x, &mut []);
                // The algorithm's bounds are `x̃`-bounds too.
                for b in [&mut it.x_l_alg, &mut it.x_u_alg] {
                    unscale_iterate(Some(&self.d), b, &mut [], &mut []);
                }
                if let Some(s) = it.scaling.as_mut()
                    && s.x.len() == self.d.len()
                    && s.x_in_alg.len() == self.d.len()
                {
                    for (j, f) in s.x.iter_mut().enumerate() {
                        if s.x_in_alg[j] {
                            *f *= self.d[j];
                        }
                    }
                }
                it
            });
        let _lifted = lifted.map(crate::live_iterate::UserIterateGuard::install);
        self.inner
            .borrow_mut()
            .intermediate_callback(stats, ip_data, ip_cq)
    }

    fn presolve_infeasibility_proof(&self) -> Option<InfeasibilityProof> {
        self.inner.borrow().presolve_infeasibility_proof()
    }
}
