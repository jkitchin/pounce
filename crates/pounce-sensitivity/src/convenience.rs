//! High-level "solve, then run sensitivity" entry point for callers
//! that don't want to write the [`set_on_converged`] callback +
//! [`PdSensBacksolver`] + [`IndexSchurData`] plumbing by hand.
//!
//! ## Parametric continuation with the SQP corrector
//!
//! For parametric NLP sweeps the Phase 5c playbook is:
//!
//! 1. Solve the base problem `min f(x; p₀)` with the IPM. Capture
//!    the converged primal `x*`, constraint multipliers `λ_g`, and
//!    bound multipliers `z_l`, `z_u` via the user TNLP's
//!    `finalize_solution`.
//! 2. Run `SensSolve::with_deltas(Δp)` to get the linear predictor
//!    `Δx ≈ ∂x*/∂p · Δp`.
//! 3. Update the parameter inside the TNLP and construct the
//!    SQP warm-start iterate:
//!    ```ignore
//!    use pounce_algorithm::sqp::{classify_working_set, SqpIterates};
//!    let lambda_x: Vec<f64> = z_l.iter().zip(z_u.iter())
//!        .map(|(l, u)| l - u).collect();
//!    let ws = classify_working_set(
//!        &lambda_x, &lambda_g, m_eq,
//!        &x_predicted, &x_l, &x_u,
//!        &g_at_predicted, &g_l, &g_u,
//!        1e-8, 1e-6,
//!    );
//!    app.set_sqp_warm_start(SqpIterates {
//!        x: x_predicted,
//!        lambda_g,
//!        lambda_x,
//!        working: Some(ws),
//!    });
//!    app.options_mut().set_string_value("algorithm", "active-set-sqp", true, false)?;
//!    let status = app.optimize_tnlp(tnlp);  // SQP corrector
//!    ```
//! 4. The SQP corrector polishes the predictor to first-order KKT
//!    at the new parameter, typically in 0–3 outer iterations
//!    with the warm-started working set.
//!
//! [`set_on_converged`]: pounce_algorithm::IpoptApplication::set_on_converged
//! [`PdSensBacksolver`]: crate::PdSensBacksolver
//! [`IndexSchurData`]: crate::IndexSchurData
//!
//! Usage shape mirrors a builder:
//!
//! ```ignore
//! use pounce_sensitivity::{SensSolve, SensResult};
//!
//! let result: SensResult = SensSolve::new(vec![2, 3])
//!     .with_deltas(vec![0.1, 0.0])
//!     .with_reduced_hessian()
//!     .run(&mut app, tnlp)?;
//! ```
//!
//! Equivalent to ~50 lines of `on_converged` boilerplate (see
//! `pounce-sensitivity/tests/parametric_cpp.rs` for the long form).
//!
//! The pin layout assumed: the user has declared a set of equality
//! constraints in their TNLP of the form `g_i(x) - p_i = 0` and is
//! passing the 0-based indices `i` into `g(x)` as
//! `pin_constraint_indices`. Perturbing `p_i → p_i + Δp_i` produces a
//! first-order estimate of `Δx` that this module returns.

use crate::backsolver::SensBacksolver;
use crate::boundcheck::clamp_with_nlp;
use crate::schur_data::IndexSchurData;
use crate::sens_app::{SensApplication, SensOptions};
use crate::PdSensBacksolver;
use pounce_algorithm::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::TNLP;
use std::cell::RefCell;
use std::rc::Rc;

/// Builder collecting what the post-solve sensitivity step should
/// compute. Construct with [`Self::new`], chain `with_*`, then call
/// [`Self::run`].
pub struct SensSolve {
    pin_constraint_indices: Vec<Index>,
    deltas: Option<Vec<Number>>,
    compute_reduced_hessian: bool,
    rh_eigendecomp: bool,
    obj_scal: Number,
    boundcheck_eps: Option<Number>,
}

/// Output of [`SensSolve::run`]. The `status` field is the same value
/// that [`IpoptApplication::optimize_tnlp`] would have returned on its
/// own; sensitivity outputs are populated only when the solve
/// converged (`SolveSucceeded` or `SolvedToAcceptableLevel`).
#[derive(Debug, Clone)]
pub struct SensResult {
    /// Pounce return status of the underlying solve.
    pub status: ApplicationReturnStatus,
    /// Final primal iterate `x*`. Length `n_x`. None when the solve
    /// failed before convergence.
    pub x: Option<Vec<Number>>,
    /// Final objective `f(x*)`. None when the solve failed.
    pub obj_val: Option<Number>,
    /// Δx for the requested perturbation. Length `n_x`. Only present
    /// when [`SensSolve::with_deltas`] was called and the solve
    /// converged.
    pub dx: Option<Vec<Number>>,
    /// Full KKT-space step (primals + slacks + duals stacked in the
    /// pounce compound-vector layout). Lower-level than `dx`; useful
    /// for cross-checking against upstream sIPOPT outputs. All blocks
    /// — including the bound-multiplier z/v rows — are in natural
    /// (unscaled) units (pounce#128); note upstream sIPOPT reports the
    /// scaled-space step when NLP scaling is active.
    pub dx_full: Option<Vec<Number>>,
    /// Reduced Hessian `H_R`, length `n_params²`, column-major, in
    /// **natural (unscaled) units** — any NLP scaling baked into the
    /// converged KKT factor is undone, so `−inv(H_R)` is directly the
    /// parameter covariance of an estimation problem regardless of
    /// `nlp_scaling_method` (pounce#128). Only present when
    /// [`SensSolve::with_reduced_hessian`] was called and the solve
    /// converged.
    pub reduced_hessian: Option<Vec<Number>>,
    /// The reduced Hessian as the solver's internal scaled space sees
    /// it — `H̃_ij = (df / (dc_i·dc_j)) · H_ij` with `df =`
    /// [`Self::obj_scaling_factor`] and `dc =` [`Self::pin_g_scaling`].
    /// This is the value pounce returned before #128; kept for callers
    /// that calibrated against it. Present iff `reduced_hessian` is.
    pub reduced_hessian_scaled: Option<Vec<Number>>,
    /// Effective objective scaling factor `df` the IPM applied
    /// (`nlp_scaling_method` / `obj_scaling_factor`; 1.0 ⇔ none).
    /// Present whenever the solve converged.
    pub obj_scaling_factor: Option<Number>,
    /// Per-pin equality-row scaling factors `dc_i` (all 1.0 when no
    /// constraint scaling is active), ordered like
    /// `pin_constraint_indices`. Present whenever the solve converged.
    pub pin_g_scaling: Option<Vec<Number>>,
    /// Eigenvalues of `H_R` in ascending order, length `n_params`.
    /// Present only when [`SensSolve::with_reduced_hessian_eigen`] was
    /// called and the solve converged.
    pub reduced_hessian_eigenvalues: Option<Vec<Number>>,
    /// Eigenvectors of `H_R`, length `n_params²`, column-major (column
    /// `j` is the eigenvector for `reduced_hessian_eigenvalues[j]`).
    /// Present only when [`SensSolve::with_reduced_hessian_eigen`] was
    /// called and the solve converged.
    pub reduced_hessian_eigenvectors: Option<Vec<Number>>,
    /// Phase 5c §6 — converged user-space constraint multipliers
    /// `λ_g` (length `n_full_g`), suitable for direct hand-off to
    /// the SQP-corrector via [`pounce_algorithm::sqp::classify_working_set`].
    /// `None` when the solve didn't converge or the underlying NLP
    /// didn't expose `pack_lambda_for_user`.
    pub mult_g: Option<Vec<Number>>,
    /// Converged user-space lower-bound multipliers `z_L`, length
    /// `n_full_x`. Same convention as the C ABI / Python info dict.
    pub mult_x_l: Option<Vec<Number>>,
    /// Converged user-space upper-bound multipliers `z_U`, length
    /// `n_full_x`. Same convention.
    pub mult_x_u: Option<Vec<Number>>,
    /// Converged constraint values `g(x*)` lifted to user-space
    /// (length `n_full_g`). Used by `classify_working_set` to
    /// classify inequality rows against their `[g_l, g_u]` bounds.
    pub g: Option<Vec<Number>>,
}

impl SensSolve {
    /// New builder. `pin_constraint_indices` are 0-based indices into
    /// the user's `g(x)` identifying the parameter-pin equality
    /// constraints (`g_i(x) = p_i`).
    pub fn new(pin_constraint_indices: Vec<Index>) -> Self {
        Self {
            pin_constraint_indices,
            deltas: None,
            compute_reduced_hessian: false,
            rh_eigendecomp: false,
            obj_scal: 1.0,
            boundcheck_eps: None,
        }
    }

    /// Request a first-order sensitivity step `Δx ≈ ∂x*/∂p · Δp` for
    /// the given perturbations. Length must equal
    /// `pin_constraint_indices.len()`.
    pub fn with_deltas(mut self, deltas: Vec<Number>) -> Self {
        self.deltas = Some(deltas);
        self
    }

    /// Request the reduced Hessian `H_R = B K⁻¹ Bᵀ` at the converged
    /// solution, where `B` selects the parameter-pin rows and `K` is
    /// the **natural-units** (unscaled) KKT matrix — any active NLP
    /// scaling is undone by the backsolver (pounce#128). The
    /// solver-space value and the scaling factors are reported
    /// alongside in [`SensResult::reduced_hessian_scaled`] /
    /// [`SensResult::obj_scaling_factor`] /
    /// [`SensResult::pin_g_scaling`].
    pub fn with_reduced_hessian(mut self) -> Self {
        self.compute_reduced_hessian = true;
        self
    }

    /// In addition to the reduced Hessian, also compute its symmetric
    /// eigendecomposition (ascending eigenvalues, column-major
    /// eigenvectors). Implies [`Self::with_reduced_hessian`].
    /// Mirrors upstream `rh_eigendecomp`.
    pub fn with_reduced_hessian_eigen(mut self) -> Self {
        self.compute_reduced_hessian = true;
        self.rh_eigendecomp = true;
        self
    }

    /// Extra constant multiplier applied to the reduced Hessian.
    /// Default 1.0. **Deprecated in spirit since pounce#128**: the
    /// backsolver now undoes the IPM's NLP scaling itself, so this is
    /// no longer needed to recover natural units — it survives as a
    /// plain user-side multiplier for callers that scaled their
    /// objective *outside* pounce.
    pub fn with_obj_scal(mut self, obj_scal: Number) -> Self {
        self.obj_scal = obj_scal;
        self
    }

    /// Enable single-pass bound clamping on the perturbed primal `x*
    /// + Δx`: any coordinate that would exceed its declared
    /// `[x_l, x_u]` by more than `eps` is clipped to the bound.
    /// Mirrors the role of upstream `sens_boundcheck` (without the
    /// iterative Schur-refinement loop — see [`crate::boundcheck`]).
    /// Only applies when [`Self::with_deltas`] is also set.
    pub fn with_boundcheck(mut self, eps: Number) -> Self {
        self.boundcheck_eps = Some(eps);
        self
    }

    /// Solve `tnlp` with `app` and run the requested sensitivity
    /// computations. Returns a [`SensResult`] regardless of solve
    /// success; the `status` field reports the convergence outcome.
    ///
    /// **Side effect**: installs an `on_converged` callback on `app`,
    /// overwriting any previously set callback.
    pub fn run(self, app: &mut IpoptApplication, tnlp: Rc<RefCell<dyn TNLP>>) -> SensResult {
        let n_params = self.pin_constraint_indices.len();
        if let Some(d) = &self.deltas {
            assert_eq!(
                d.len(),
                n_params,
                "deltas.len() ({}) must equal pin_constraint_indices.len() ({})",
                d.len(),
                n_params,
            );
        }
        let want_dx = self.deltas.is_some();
        let want_rh = self.compute_reduced_hessian;
        let want_eigen = self.rh_eigendecomp;
        let pin_indices = self.pin_constraint_indices.clone();
        let deltas = self.deltas.clone();
        let obj_scal = self.obj_scal;
        let boundcheck_eps = self.boundcheck_eps;

        // Side channel: the callback writes here, the outer caller
        // reads after optimize_tnlp returns. RefCell + Rc because the
        // callback closure outlives the call frame.
        let outbox: Rc<RefCell<CallbackOut>> = Rc::new(RefCell::new(CallbackOut::default()));
        let outbox_cb = Rc::clone(&outbox);

        app.set_on_converged(Box::new(move |data, cq, nlp, pd| {
            let curr = match data.borrow().curr.clone() {
                Some(c) => c,
                None => {
                    outbox_cb.borrow_mut().error = Some("no current iterate at convergence".into());
                    return;
                }
            };

            // Always capture x and obj_val so the caller gets them
            // even when only the reduced Hessian was requested.
            outbox_cb.borrow_mut().x = Some(dense_to_vec(&*curr.x));
            outbox_cb.borrow_mut().obj_val = Some(cq.borrow_mut().curr_f());

            // Phase 5c §6 — also capture user-space multipliers +
            // constraint values so callers can wire the parametric
            // corrector via `classify_working_set` without a
            // separate IPM solve. `pack_*_for_user` returns empty
            // when the underlying NLP doesn't implement lifting
            // (defaults on `IpoptNlp` trait); we treat that as
            // "no user-space hand-off available".
            //
            // `curr_c`/`curr_d` cache results into the NLP via
            // `eval_*` if not already computed; pull them BEFORE
            // we hold an immutable borrow on `nlp` so the cache
            // path has access to its own mut borrow.
            let g_curr = cq.borrow_mut().curr_c();
            let d_curr = cq.borrow_mut().curr_d();
            {
                let nlp_borrow = nlp.borrow();
                let lambda = nlp_borrow.pack_lambda_for_user(&*curr.y_c, &*curr.y_d);
                if !lambda.is_empty() {
                    outbox_cb.borrow_mut().mult_g = Some(lambda);
                }
                let z_l = nlp_borrow.pack_z_l_for_user(&*curr.z_l);
                if !z_l.is_empty() {
                    outbox_cb.borrow_mut().mult_x_l = Some(z_l);
                }
                let z_u = nlp_borrow.pack_z_u_for_user(&*curr.z_u);
                if !z_u.is_empty() {
                    outbox_cb.borrow_mut().mult_x_u = Some(z_u);
                }
                let g_user = nlp_borrow.pack_g_for_user(&*g_curr, &*d_curr);
                if !g_user.is_empty() {
                    outbox_cb.borrow_mut().g = Some(g_user);
                }
            }

            let n_x = curr.x.dim() as usize;

            let backsolver = match PdSensBacksolver::new(data, cq, nlp, Rc::clone(&pd)) {
                Ok(b) => b,
                Err(e) => {
                    outbox_cb.borrow_mut().error =
                        Some(format!("PdSensBacksolver::new failed: {e}"));
                    return;
                }
            };
            let n_full = backsolver.dim();

            // Pin constraint indices are user-facing 0-based indices
            // into g(x); the matching y_c slot is found through the
            // NLP's c/d-split row map (pounce#128: a direct
            // `n_x + n_s + i` is wrong once inequalities precede the
            // pins in g).
            let (param_rows, pin_scales) = match backsolver.pin_rows_and_c_scales(&pin_indices) {
                Ok(rs) => rs,
                Err(e) => {
                    outbox_cb.borrow_mut().error = Some(e);
                    return;
                }
            };
            let df = backsolver.obj_scaling_factor();
            outbox_cb.borrow_mut().obj_scaling_factor = Some(df);
            outbox_cb.borrow_mut().pin_g_scaling = Some(pin_scales.clone());
            let signs = vec![1; n_params];

            let a_data = match IndexSchurData::from_parts(param_rows, signs) {
                Ok(a) => a,
                Err(e) => {
                    outbox_cb.borrow_mut().error =
                        Some(format!("IndexSchurData::from_parts failed: {e:?}"));
                    return;
                }
            };

            let opts = SensOptions {
                run_sens: want_dx,
                compute_red_hessian: want_rh,
                rh_eigendecomp: want_eigen,
                obj_scal,
                ..SensOptions::default()
            };
            let mut sens_app = SensApplication::new(a_data, backsolver, opts);

            if let Some(d) = &deltas {
                let mut dx_full = vec![0.0; n_full];
                if !sens_app.parametric_step(d, &mut dx_full) {
                    outbox_cb.borrow_mut().error =
                        Some("SensApplication::parametric_step failed".into());
                    return;
                }
                if let Some(eps) = boundcheck_eps {
                    // x_curr is the first `n_x` slots of the
                    // compound-vector iterate.
                    let x_curr = dense_to_vec(&*curr.x);
                    // clamp only the primal-x block of dx_full; the
                    // rest (s, multipliers) doesn't have primal bounds.
                    let mut dx_primal = dx_full[..n_x].to_vec();
                    let _ = clamp_with_nlp(&*nlp.borrow(), &x_curr, &mut dx_primal, eps);
                    dx_full[..n_x].copy_from_slice(&dx_primal);
                }
                let dx_primal = dx_full[..n_x].to_vec();
                outbox_cb.borrow_mut().dx = Some(dx_primal);
                outbox_cb.borrow_mut().dx_full = Some(dx_full);
            }

            if want_rh {
                let mut hr = vec![0.0; n_params * n_params];
                if want_eigen {
                    let mut w = vec![0.0; n_params];
                    let mut v = vec![0.0; n_params * n_params];
                    if !sens_app.compute_reduced_hessian_eigen(&mut hr, &mut w, &mut v) {
                        outbox_cb.borrow_mut().error =
                            Some("SensApplication::compute_reduced_hessian_eigen failed".into());
                        return;
                    }
                    outbox_cb.borrow_mut().reduced_hessian_eigenvalues = Some(w);
                    outbox_cb.borrow_mut().reduced_hessian_eigenvectors = Some(v);
                } else if !sens_app.compute_reduced_hessian(&mut hr) {
                    outbox_cb.borrow_mut().error =
                        Some("SensApplication::compute_reduced_hessian failed".into());
                    return;
                }
                // Solver-space (pre-#128) value, reconstructed from
                // the natural-units H rather than re-solved.
                let mut hr_scaled = hr.clone();
                crate::reduced_hessian::scale_to_solver_space(&mut hr_scaled, df, &pin_scales);
                outbox_cb.borrow_mut().reduced_hessian = Some(hr);
                outbox_cb.borrow_mut().reduced_hessian_scaled = Some(hr_scaled);
            }
        }));

        let status = app.optimize_tnlp(tnlp);
        let out = outbox.borrow();
        SensResult {
            status,
            x: out.x.clone(),
            obj_val: out.obj_val,
            dx: out.dx.clone(),
            dx_full: out.dx_full.clone(),
            reduced_hessian: out.reduced_hessian.clone(),
            reduced_hessian_scaled: out.reduced_hessian_scaled.clone(),
            obj_scaling_factor: out.obj_scaling_factor,
            pin_g_scaling: out.pin_g_scaling.clone(),
            reduced_hessian_eigenvalues: out.reduced_hessian_eigenvalues.clone(),
            reduced_hessian_eigenvectors: out.reduced_hessian_eigenvectors.clone(),
            mult_g: out.mult_g.clone(),
            mult_x_l: out.mult_x_l.clone(),
            mult_x_u: out.mult_x_u.clone(),
            g: out.g.clone(),
        }
    }
}

#[derive(Default)]
struct CallbackOut {
    x: Option<Vec<Number>>,
    obj_val: Option<Number>,
    dx: Option<Vec<Number>>,
    dx_full: Option<Vec<Number>>,
    reduced_hessian: Option<Vec<Number>>,
    reduced_hessian_scaled: Option<Vec<Number>>,
    obj_scaling_factor: Option<Number>,
    pin_g_scaling: Option<Vec<Number>>,
    reduced_hessian_eigenvalues: Option<Vec<Number>>,
    reduced_hessian_eigenvectors: Option<Vec<Number>>,
    mult_g: Option<Vec<Number>>,
    mult_x_l: Option<Vec<Number>>,
    mult_x_u: Option<Vec<Number>>,
    g: Option<Vec<Number>>,
    #[allow(dead_code)]
    error: Option<String>,
}

fn dense_to_vec(v: &dyn pounce_linalg::Vector) -> Vec<Number> {
    match v
        .as_any()
        .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
    {
        Some(d) => d.values().to_vec(),
        None => vec![0.0; v.dim() as usize],
    }
}
