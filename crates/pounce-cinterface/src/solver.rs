//! Session-style C ABI built on [`pounce_sensitivity::Solver`].
//!
//! Adds an opaque [`IpoptSolver`] handle that captures the converged
//! KKT factor between calls, so C consumers can issue many cheap
//! operations (KKT back-solves, parametric steps, reduced Hessians)
//! against the same factorization without re-running the IPM.
//!
//! ```c
//! IpoptProblem prob = CreateIpoptProblem(...);
//! AddIpoptStrOption(prob, "linear_solver", "feral");
//! IpoptSolver sol = IpoptCreateSolver(&prob);   // consumes prob
//! IpoptSolverSolve(sol, x, NULL, NULL, NULL, NULL, NULL, user_data);
//! IpoptSolverParametricStep(sol, 2, pin_indices, deltas, dx_out);
//! IpoptSolverReducedHessian(sol, 2, pin_indices, 1.0, hr_out);
//! IpoptFreeSolver(sol);
//! ```
//!
//! Ownership: [`IpoptCreateSolver`] takes the IpoptProblem by **pointer
//! to the handle** and nulls it out on success — the IpoptSolver
//! becomes the sole owner. Calling [`crate::FreeIpoptProblem`] on the
//! now-null handle is safe (it null-checks).

use pounce_algorithm::application::{
    default_backend_factory, feral_config_from_options, IpoptApplication,
};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::TNLP;
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    make_default_restoration_factory_provider, InnerBackendFactoryFactory,
};
use pounce_sensitivity::Solver as RustSolver;
use std::cell::RefCell;
use std::ffi::c_void;
use std::rc::Rc;

use crate::{
    Bool, CCallbackTnlp, Index, IpoptProblem, IpoptProblemInfo, LastSolve, Number, FALSE, TRUE,
};

/// Internal owned state for the session-style C handle.
pub struct IpoptSolverInfo {
    /// The session. `None` before the first solve or after a solve
    /// that didn't converge.
    session: Option<RustSolver>,
    /// All the problem state: callbacks, dims, bounds, options. The
    /// IpoptApplication inside is moved out into the `session` on each
    /// successful solve, then restored on the next solve via the
    /// `app_template` field below.
    ///
    /// (Stored as `Option` so `IpoptSolverSolve` can `.take()` the app
    /// to move into the session, then put it back on next call.)
    problem: IpoptProblemInfo,
    /// Number of constraints — cached for cheap shape checks.
    m: Index,
}

/// Opaque session-style handle. Construction via
/// [`IpoptCreateSolver`]; release via [`IpoptFreeSolver`].
pub type IpoptSolver = *mut IpoptSolverInfo;

/// Build an [`IpoptSolver`] session from a configured
/// [`IpoptProblem`]. **Consumes the IpoptProblem** on success: the
/// pointer at `*prob_handle` is set to NULL and ownership transfers
/// to the returned IpoptSolver. The user should not use the original
/// handle again, though calling [`crate::FreeIpoptProblem`] on the
/// now-null pointer is harmless (it null-checks).
///
/// Returns NULL if `prob_handle` is NULL, `*prob_handle` is NULL, or
/// the IpoptProblem hasn't been fully initialized.
///
/// # Safety
///
/// `prob_handle` must be a valid pointer to an [`IpoptProblem`]
/// previously returned by [`crate::CreateIpoptProblem`] (or NULL).
#[no_mangle]
pub unsafe extern "C" fn IpoptCreateSolver(prob_handle: *mut IpoptProblem) -> IpoptSolver {
    if prob_handle.is_null() {
        return std::ptr::null_mut();
    }
    let prob = *prob_handle;
    if prob.is_null() {
        return std::ptr::null_mut();
    }
    // Take ownership of the Box and null out the caller's handle.
    let problem = *Box::from_raw(prob);
    *prob_handle = std::ptr::null_mut();
    let m = problem.m;
    let info = Box::new(IpoptSolverInfo {
        session: None,
        problem,
        m,
    });
    Box::into_raw(info)
}

/// Release an [`IpoptSolver`] and all owned resources, including the
/// IpoptProblem state that was consumed by [`IpoptCreateSolver`].
///
/// # Safety
///
/// `solver` must be a pointer returned by [`IpoptCreateSolver`] and
/// not yet freed, or NULL.
#[no_mangle]
pub unsafe extern "C" fn IpoptFreeSolver(solver: IpoptSolver) {
    if solver.is_null() {
        return;
    }
    drop(Box::from_raw(solver));
}

/// Run the IPM. Same output buffer contract as [`crate::IpoptSolve`]:
/// `x` is in/out (initial guess in, solution out); `g`, `obj_val`,
/// `mult_g`, `mult_x_L`, `mult_x_U` are out-only and may be NULL.
/// `user_data` is threaded into the C callbacks unchanged.
///
/// Returns the same `Index`-cast [`ApplicationReturnStatus`] code as
/// [`crate::IpoptSolve`]. On a converged status the session retains
/// the KKT factor for subsequent [`IpoptSolverKktSolve`],
/// [`IpoptSolverParametricStep`], and [`IpoptSolverReducedHessian`]
/// calls.
///
/// # Safety
///
/// All non-NULL output pointers must be valid for the appropriate
/// length; the C callbacks stored on the underlying IpoptProblem must
/// remain valid through the solve.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn IpoptSolverSolve(
    solver: IpoptSolver,
    x: *mut Number,
    g: *mut Number,
    obj_val: *mut Number,
    mult_g: *mut Number,
    mult_x_L: *mut Number,
    mult_x_U: *mut Number,
    user_data: *mut c_void,
) -> Index {
    if solver.is_null() {
        return ApplicationReturnStatus::InternalError as Index;
    }
    let info = &mut *solver;
    let n = info.problem.n;
    let m = info.m;
    if n < 0 || m < 0 {
        return ApplicationReturnStatus::InvalidProblemDefinition as Index;
    }
    if n > 0 && x.is_null() {
        return ApplicationReturnStatus::InvalidProblemDefinition as Index;
    }
    let n_us = n as usize;
    let m_us = m as usize;
    let initial_x = if n_us > 0 {
        std::slice::from_raw_parts(x, n_us).to_vec()
    } else {
        Vec::new()
    };

    let bridge = Rc::new(RefCell::new(CCallbackTnlp {
        n,
        m,
        nele_jac: info.problem.nele_jac,
        nele_hess: info.problem.nele_hess,
        index_style: info.problem.index_style,
        x_l: info.problem.x_l.clone(),
        x_u: info.problem.x_u.clone(),
        g_l: info.problem.g_l.clone(),
        g_u: info.problem.g_u.clone(),
        initial_x,
        eval_f: info.problem.eval_f,
        eval_grad_f: info.problem.eval_grad_f,
        eval_g: info.problem.eval_g,
        eval_jac_g: info.problem.eval_jac_g,
        eval_h: info.problem.eval_h,
        user_data,
        intermediate_cb: info.problem.intermediate_cb,
        user_scaling: info.problem.user_scaling.clone(),
        final_status: None,
        final_x: vec![0.0; n_us],
        final_z_l: vec![0.0; n_us],
        final_z_u: vec![0.0; n_us],
        final_g: vec![0.0; m_us],
        final_lambda: vec![0.0; m_us],
        final_obj: 0.0,
    }));

    // Re-wire restoration fresh for this solve (same pattern as
    // IpoptSolve). Multi-pass provider so the ℓ₁ wrapper / auto-fallback
    // don't panic on the second inner solve (pounce#10 / pounce#24).
    let feral_cfg = feral_config_from_options(info.problem.app.options());
    let bff_mint = move || -> InnerBackendFactoryFactory {
        let feral_cfg = feral_cfg.clone();
        Box::new(move || default_backend_factory(feral_cfg.clone()))
    };
    let resto_provider = make_default_restoration_factory_provider(
        RestoAlgorithmBuilder::new(),
        info.problem.app.algorithm_builder_from_options(),
        bff_mint,
    );
    info.problem
        .app
        .set_restoration_factory_provider(resto_provider);

    // Move the app out of the problem and into a fresh RustSolver.
    let app = std::mem::replace(&mut info.problem.app, IpoptApplication::new());
    let bridge_for_solver: Rc<RefCell<dyn TNLP>> = bridge.clone();
    let mut rust_solver = RustSolver::new(app, bridge_for_solver);
    let status = rust_solver.solve();
    let bridge_ref = bridge.borrow();
    info.problem.last_solve = Some(LastSolve {
        stats: rust_solver.app().statistics(),
        status,
        linear_solver: rust_solver.app().linear_solver_summary(),
        final_x: bridge_ref.final_x.clone(),
        final_lambda: bridge_ref.final_lambda.clone(),
        final_obj: bridge_ref.final_obj,
    });
    if !x.is_null() && n_us > 0 {
        std::ptr::copy_nonoverlapping(bridge_ref.final_x.as_ptr(), x, n_us);
    }
    if !g.is_null() && m_us > 0 {
        std::ptr::copy_nonoverlapping(bridge_ref.final_g.as_ptr(), g, m_us);
    }
    if !obj_val.is_null() {
        *obj_val = bridge_ref.final_obj;
    }
    if !mult_g.is_null() && m_us > 0 {
        std::ptr::copy_nonoverlapping(bridge_ref.final_lambda.as_ptr(), mult_g, m_us);
    }
    if !mult_x_L.is_null() && n_us > 0 {
        std::ptr::copy_nonoverlapping(bridge_ref.final_z_l.as_ptr(), mult_x_L, n_us);
    }
    if !mult_x_U.is_null() && n_us > 0 {
        std::ptr::copy_nonoverlapping(bridge_ref.final_z_u.as_ptr(), mult_x_U, n_us);
    }

    info.session = Some(rust_solver);
    status as Index
}

/// Total compound-KKT vector dimension. Returns -1 if no converged
/// factor is held.
///
/// # Safety
///
/// `solver` must be a valid [`IpoptSolver`] or NULL.
#[no_mangle]
pub unsafe extern "C" fn IpoptSolverGetKktDim(solver: IpoptSolver) -> Index {
    if solver.is_null() {
        return -1;
    }
    let info = &*solver;
    match info.session.as_ref().and_then(|s| s.kkt_dim()) {
        Some(d) => d as Index,
        None => -1,
    }
}

/// Solve `K · lhs = rhs` against the converged KKT factor. Both
/// `rhs` and `lhs` are flat buffers of length [`IpoptSolverGetKktDim`]
/// in the `x || s || y_c || y_d || z_l || z_u || v_l || v_u` packing.
///
/// Returns `TRUE` on success, `FALSE` if no factor is held or the
/// back-solve fails.
///
/// # Safety
///
/// `rhs` and `lhs` must point to buffers at least
/// [`IpoptSolverGetKktDim`] doubles long.
#[no_mangle]
pub unsafe extern "C" fn IpoptSolverKktSolve(
    solver: IpoptSolver,
    rhs: *const Number,
    lhs: *mut Number,
) -> Bool {
    if solver.is_null() || rhs.is_null() || lhs.is_null() {
        return FALSE;
    }
    let info = &*solver;
    let Some(s) = info.session.as_ref() else {
        return FALSE;
    };
    let Some(dim) = s.kkt_dim() else {
        return FALSE;
    };
    let rhs_slice = std::slice::from_raw_parts(rhs, dim);
    let mut lhs_vec = vec![0.0; dim];
    if s.kkt_solve(rhs_slice, &mut lhs_vec).is_err() {
        return FALSE;
    }
    std::ptr::copy_nonoverlapping(lhs_vec.as_ptr(), lhs, dim);
    TRUE
}

/// First-order parametric step `Δx ≈ ∂x*/∂p · Δp`. `pin_indices` is
/// `n_pins` `Index` values (0-based indices into `g(x)`); `deltas` is
/// the parameter perturbation `Δp` of the same length; `dx_out` is the
/// `n`-long primal step output (length matches the problem's `n`).
///
/// Returns `TRUE` on success, `FALSE` if no converged factor, invalid
/// indices, or the sensitivity computation fails.
///
/// # Safety
///
/// `pin_indices` and `deltas` must point to `n_pins` valid elements;
/// `dx_out` must point to at least `n` `Number` slots (`n` from the
/// underlying IpoptProblem).
#[no_mangle]
pub unsafe extern "C" fn IpoptSolverParametricStep(
    solver: IpoptSolver,
    n_pins: Index,
    pin_indices: *const Index,
    deltas: *const Number,
    dx_out: *mut Number,
) -> Bool {
    if solver.is_null() || n_pins < 0 {
        return FALSE;
    }
    if n_pins > 0 && (pin_indices.is_null() || deltas.is_null()) {
        return FALSE;
    }
    if dx_out.is_null() {
        return FALSE;
    }
    let info = &*solver;
    let Some(s) = info.session.as_ref() else {
        return FALSE;
    };
    let m = info.m;
    let pins_raw = std::slice::from_raw_parts(pin_indices, n_pins as usize);
    let mut pins = Vec::with_capacity(n_pins as usize);
    for &i in pins_raw {
        if i < 0 || i >= m {
            return FALSE;
        }
        pins.push(i as pounce_common::types::Index);
    }
    let deltas_slice = std::slice::from_raw_parts(deltas, n_pins as usize);
    let Ok(dx) = s.parametric_step(&pins, deltas_slice) else {
        return FALSE;
    };
    std::ptr::copy_nonoverlapping(dx.as_ptr(), dx_out, dx.len());
    TRUE
}

/// Reduced Hessian `H_R = obj_scal · B K⁻¹ Bᵀ` over the pinned rows.
/// `hr_out` receives an `n_pins²`-long column-major dense matrix.
///
/// Returns `TRUE` on success, `FALSE` otherwise.
///
/// # Safety
///
/// `pin_indices` must point to `n_pins` valid elements; `hr_out` must
/// point to at least `n_pins²` `Number` slots.
#[no_mangle]
pub unsafe extern "C" fn IpoptSolverReducedHessian(
    solver: IpoptSolver,
    n_pins: Index,
    pin_indices: *const Index,
    obj_scal: Number,
    hr_out: *mut Number,
) -> Bool {
    if solver.is_null() || n_pins < 0 || hr_out.is_null() {
        return FALSE;
    }
    if n_pins > 0 && pin_indices.is_null() {
        return FALSE;
    }
    let info = &*solver;
    let Some(s) = info.session.as_ref() else {
        return FALSE;
    };
    let m = info.m;
    let pins_raw = std::slice::from_raw_parts(pin_indices, n_pins as usize);
    let mut pins = Vec::with_capacity(n_pins as usize);
    for &i in pins_raw {
        if i < 0 || i >= m {
            return FALSE;
        }
        pins.push(i as pounce_common::types::Index);
    }
    let Ok(hr) = s.compute_reduced_hessian(&pins, obj_scal) else {
        return FALSE;
    };
    std::ptr::copy_nonoverlapping(hr.as_ptr(), hr_out, hr.len());
    TRUE
}
