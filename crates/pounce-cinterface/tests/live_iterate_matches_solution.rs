//! `GetIpoptCurrentIterate` at the last intermediate-callback fire reads
//! the solution `IpoptSolve` writes back, on both arms, under every layer
//! between the algorithm and the user's problem: gradient-based scaling
//! with an objective factor, per-variable user scaling, a fixed variable.
//!
//! Upstream (Ipopt 3.14 `TNLP::get_curr_iterate`) reinserts fixed
//! variables (`TNLPAdapter::ResortX`) and recovers their bound multipliers
//! (`ResortBoundMultipliers`, `z_L − z_U = ∇f + Jᵀλ` under
//! `make_parameter`). The interior-point arm here left them at `0`.
//!
//! Model: `min K·[(x0-3)² + (x1-3)² + (x2-1)² + (x3-4)²]`
//! s.t. `g0 = x0 + x1 + x3 ≤ 3.5`, `g1 = x0 − x2 = 0`,
//! `g2 = x1 + x3 ≤ 10` (implied by the box: `x1 ≤ 1`, `x3 = 1`),
//! `x0` free, `x1 ∈ [-10, 1]` (upper active), `x2 ≥ 0`, `x3 = 1` fixed.

use pounce_cinterface::*;
use std::cell::{Cell, RefCell};
use std::ffi::{CString, c_void};

const N: usize = 4;
const M: usize = 3;
const NNZ: usize = 7;

#[derive(Clone, Debug, Default)]
struct It {
    ok: bool,
    /// The `obj_value` the intermediate callback received at this fire.
    obj: Number,
    x: [Number; N],
    z_l: [Number; N],
    z_u: [Number; N],
    g: [Number; M],
    lambda: [Number; M],
    /// `GetIpoptCurrentViolations(scaled = FALSE)` at the same fire.
    viol: Option<Viol>,
    /// `GetIpoptCurrentViolations(scaled = TRUE)` at the same fire.
    viol_scaled: Option<Viol>,
    /// `GetIpoptCurrentIterate(scaled = TRUE)` at the same fire.
    scaled: Option<Box<It>>,
}

unsafe fn iterate(p: IpoptProblem, scaled: Bool) -> It {
    let mut it = It::default();
    it.ok = unsafe {
        GetIpoptCurrentIterate(
            p,
            scaled,
            N as Index,
            it.x.as_mut_ptr(),
            it.z_l.as_mut_ptr(),
            it.z_u.as_mut_ptr(),
            M as Index,
            it.g.as_mut_ptr(),
            it.lambda.as_mut_ptr(),
        )
    } != 0;
    it
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Viol {
    x_l: [Number; N],
    x_u: [Number; N],
    compl_x_l: [Number; N],
    compl_x_u: [Number; N],
    grad_lag_x: [Number; N],
    g: [Number; M],
    compl_g: [Number; M],
}

unsafe fn violations(p: IpoptProblem, scaled: Bool) -> Option<Viol> {
    let mut v = Viol::default();
    let ok = unsafe {
        GetIpoptCurrentViolations(
            p,
            scaled,
            N as Index,
            v.x_l.as_mut_ptr(),
            v.x_u.as_mut_ptr(),
            v.compl_x_l.as_mut_ptr(),
            v.compl_x_u.as_mut_ptr(),
            v.grad_lag_x.as_mut_ptr(),
            M as Index,
            v.g.as_mut_ptr(),
            v.compl_g.as_mut_ptr(),
        )
    };
    (ok != 0).then_some(v)
}

thread_local! {
    static FIRES: RefCell<Vec<It>> = const { RefCell::new(Vec::new()) };
    static K: Cell<Number> = const { Cell::new(1.0) };
    /// `(eval_grad_f, eval_jac_g values)` calls, and whether the callback
    /// reads `GetIpoptCurrentViolations`.
    static EVALS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    static READ_VIOLATIONS: Cell<bool> = const { Cell::new(true) };
    /// `bound_relax_factor` for the next solve, when positive.
    static RELAX: Cell<Number> = const { Cell::new(0.0) };
}

fn k() -> Number {
    K.with(|k| k.get())
}

unsafe extern "C" fn ev_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        *obj = k()
            * ((x[0] - 3.0).powi(2)
                + (x[1] - 3.0).powi(2)
                + (x[2] - 1.0).powi(2)
                + (x[3] - 4.0).powi(2));
        1
    }
}

unsafe extern "C" fn ev_grad_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    grad: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        let g = std::slice::from_raw_parts_mut(grad, N);
        EVALS.with(|e| e.set((e.get().0 + 1, e.get().1)));
        g[0] = 2.0 * k() * (x[0] - 3.0);
        g[1] = 2.0 * k() * (x[1] - 3.0);
        g[2] = 2.0 * k() * (x[2] - 1.0);
        g[3] = 2.0 * k() * (x[3] - 4.0);
        1
    }
}

unsafe extern "C" fn ev_g(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    g: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        let g = std::slice::from_raw_parts_mut(g, M);
        g[0] = x[0] + x[1] + x[3];
        g[1] = x[0] - x[2];
        g[2] = x[1] + x[3];
        1
    }
}

const JAC_ROW: [Index; NNZ] = [0, 0, 0, 1, 1, 2, 2];
const JAC_COL: [Index; NNZ] = [0, 1, 3, 0, 2, 1, 3];
const JAC_VAL: [Number; NNZ] = [1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0];

unsafe extern "C" fn ev_jac_g(
    _n: Index,
    _x: *const Number,
    _new_x: Bool,
    _m: Index,
    _nele: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            std::slice::from_raw_parts_mut(i_row, NNZ).copy_from_slice(&JAC_ROW);
            std::slice::from_raw_parts_mut(j_col, NNZ).copy_from_slice(&JAC_COL);
        } else {
            EVALS.with(|e| e.set((e.get().0, e.get().1 + 1)));
            std::slice::from_raw_parts_mut(values, NNZ).copy_from_slice(&JAC_VAL);
        }
        1
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn ev_h(
    _n: Index,
    _x: *const Number,
    _new_x: Bool,
    of: Number,
    _m: Index,
    _lambda: *const Number,
    _new_lambda: Bool,
    _nele: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            std::slice::from_raw_parts_mut(i_row, N).copy_from_slice(&[0, 1, 2, 3]);
            std::slice::from_raw_parts_mut(j_col, N).copy_from_slice(&[0, 1, 2, 3]);
        } else {
            std::slice::from_raw_parts_mut(values, N).copy_from_slice(&[2.0 * k() * of; N]);
        }
        1
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn on_iter(
    _alg_mod: Index,
    _iter: Index,
    obj: Number,
    _inf_pr: Number,
    _inf_du: Number,
    _mu: Number,
    _d_norm: Number,
    _regu: Number,
    _alpha_du: Number,
    _alpha_pr: Number,
    _ls_trials: Index,
    user_data: *mut c_void,
) -> Bool {
    unsafe {
        let mut it = iterate(user_data as IpoptProblem, 0);
        it.scaled = Some(Box::new(iterate(user_data as IpoptProblem, 1)));
        it.obj = obj;
        if READ_VIOLATIONS.with(|r| r.get()) {
            it.viol = violations(user_data as IpoptProblem, 0);
            it.viol_scaled = violations(user_data as IpoptProblem, 1);
        }
        FIRES.with(|f| f.borrow_mut().push(it));
        1
    }
}

#[derive(Clone, Copy, Debug)]
enum Cfg {
    /// `gradient-based` (engaged: `K = 1000` puts `‖∇f‖∞` far above
    /// `nlp_scaling_max_gradient`) times `obj_scaling_factor = 3`.
    GradientBased,
    /// `user-scaling`: obj 3, x `[2, 0.5, 4, 3]`, g `[0.01, 5, 1]`.
    UserX,
}

/// `(IpoptSolve's output, every fire's iterate)`.
/// [`solve`] with `bound_relax_factor = 1e-3` (a `1e-4` widening, capped
/// by `constr_viol_tol`), so the iterate visibly leaves the declared box and
/// the relaxed and declared bounds are visibly apart.
fn solve_relaxed(algorithm: &str, cfg: Cfg) -> (It, Vec<It>) {
    RELAX.with(|r| r.set(1e-3));
    solve(algorithm, cfg)
}

fn solve(algorithm: &str, cfg: Cfg) -> (It, Vec<It>) {
    FIRES.with(|f| f.borrow_mut().clear());
    K.with(|k| {
        k.set(match cfg {
            Cfg::GradientBased => 1000.0,
            _ => 1.0,
        })
    });
    let x_l = [-2e19, -10.0, 0.0, 1.0];
    let x_u = [2e19, 1.0, 2e19, 1.0];
    let g_l = [-2e19, 0.0, -2e19];
    let g_u = [3.5, 0.0, 10.0];
    unsafe {
        let p = CreateIpoptProblem(
            N as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            M as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            NNZ as Index,
            N as Index,
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            Some(ev_h),
        );
        assert!(!p.is_null());
        let opt = |k: &str, v: &str| {
            let (k, v) = (CString::new(k).unwrap(), CString::new(v).unwrap());
            assert_ne!(AddIpoptStrOption(p, k.as_ptr(), v.as_ptr()), 0);
        };
        opt("algorithm", algorithm);
        let relax = RELAX.with(|r| r.replace(0.0));
        if relax > 0.0 {
            let k = CString::new("bound_relax_factor").unwrap();
            assert_ne!(AddIpoptNumOption(p, k.as_ptr(), relax), 0);
        }
        let kp = CString::new("print_level").unwrap();
        assert_ne!(AddIpoptIntOption(p, kp.as_ptr(), 0), 0);
        match cfg {
            Cfg::GradientBased => {
                opt("nlp_scaling_method", "gradient-based");
                let k = CString::new("obj_scaling_factor").unwrap();
                assert_ne!(AddIpoptNumOption(p, k.as_ptr(), 3.0), 0);
            }
            Cfg::UserX => {
                opt("nlp_scaling_method", "user-scaling");
                let xs = [2.0, 0.5, 4.0, 3.0];
                let gs = [0.01, 5.0, 1.0];
                assert_ne!(SetIpoptProblemScaling(p, 3.0, xs.as_ptr(), gs.as_ptr()), 0);
            }
        }
        assert_ne!(SetIntermediateCallback(p, Some(on_iter)), 0);
        let mut out = It {
            ok: true,
            x: [0.0, 0.0, 0.5, 1.0],
            ..It::default()
        };
        let mut obj = 0.0;
        let st = IpoptSolve(
            p,
            out.x.as_mut_ptr(),
            out.g.as_mut_ptr(),
            &mut obj,
            out.lambda.as_mut_ptr(),
            out.z_l.as_mut_ptr(),
            out.z_u.as_mut_ptr(),
            p as *mut c_void,
        );
        FreeIpoptProblem(p);
        assert_eq!(st, 0, "{algorithm} {cfg:?}: expected Solve_Succeeded");
        (out, FIRES.with(|f| f.borrow().clone()))
    }
}

/// The objective `ev_f` evaluates, at `x`.
fn objective(x: &[Number; N]) -> Number {
    k() * ((x[0] - 3.0).powi(2)
        + (x[1] - 3.0).powi(2)
        + (x[2] - 1.0).powi(2)
        + (x[3] - 4.0).powi(2))
}

fn close(what: &str, got: &[Number], want: &[Number]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9 * w.abs() + 1e-300,
            "{what}[{i}]: callback {g:e}, IpoptSolve {w:e} ({got:?} vs {want:?})"
        );
    }
}

fn check(algorithm: &str, cfg: Cfg) {
    let (out, fires) = solve(algorithm, cfg);
    assert!(
        fires.len() > 1,
        "{algorithm} {cfg:?}: {} fires",
        fires.len()
    );
    assert!(
        fires.iter().all(|f| f.ok),
        "{algorithm} {cfg:?}: GetIpoptCurrentIterate refused"
    );
    // Informative: the fixed x3 carries a multiplier (`2K(1-4) + λ0 < 0`,
    // so z_U), and x1's upper bound binds.
    assert!(out.z_u[3] > 1e-3, "{algorithm} {cfg:?}: {out:?}");
    assert!(out.z_u[1] > 1e-3, "{algorithm} {cfg:?}: {out:?}");
    // `obj_value` is the user's objective at the iterate, as upstream passes
    // `unscaled_curr_f()`: not the algorithm's `curr_f()`, which carries the
    // objective factor (3) and, under gradient-based scaling, the gradient
    // factor too.
    for (i, f) in fires.iter().enumerate() {
        let want = objective(&f.x);
        assert!(
            (f.obj - want).abs() <= 1e-9 * want.abs().max(1.0),
            "{algorithm} {cfg:?}: fire {i} obj_value {:e}, f(x) {want:e}",
            f.obj
        );
    }
    let last = fires.last().unwrap();
    close("x", &last.x, &out.x);
    close("z_l", &last.z_l, &out.z_l);
    close("z_u", &last.z_u, &out.z_u);
    close("g", &last.g, &out.g);
    close("lambda", &last.lambda, &out.lambda);
}

#[test]
fn ipm_iterate_is_the_solution_under_gradient_scaling_and_a_fixed_variable() {
    check("interior-point", Cfg::GradientBased);
}

#[test]
fn ipm_iterate_is_the_solution_under_user_x_scaling_and_a_fixed_variable() {
    check("interior-point", Cfg::UserX);
}

#[test]
fn sqp_iterate_is_the_solution_under_gradient_scaling_and_a_fixed_variable() {
    check("active-set-sqp", Cfg::GradientBased);
}

#[test]
fn sqp_iterate_is_the_solution_under_user_x_scaling_and_a_fixed_variable() {
    check("active-set-sqp", Cfg::UserX);
}

// ---------------------------------------------------------------------
// GetIpoptCurrentViolations
// ---------------------------------------------------------------------

const X_L: [Number; N] = [-2e19, -10.0, 0.0, 1.0];
const X_U: [Number; N] = [2e19, 1.0, 2e19, 1.0];
const G_L: [Number; M] = [-2e19, 0.0, -2e19];
const G_U: [Number; M] = [3.5, 0.0, 10.0];

/// The bounds the algorithm iterates against under `solve_relaxed`, from the
/// declared ones by `bound_relax_factor`'s rule, independently of the
/// solver: the widening `min(1e-3·max(1, |b̃|), constr_viol_tol = 1e-4)` is
/// applied in the coordinates the algorithm sees, `b̃ = d·b` (user scaling's
/// `x̃ = d ⊙ x`; `d = 1` under gradient-based scaling, which scales no
/// variable), and taken back, `b ∓ δ/d`. The fixed `x3` is not in the
/// algorithm: its bound stays its value.
fn relaxed_bounds(cfg: Cfg) -> ([Number; N], [Number; N]) {
    let d = match cfg {
        Cfg::UserX => [2.0, 0.5, 4.0, 3.0],
        Cfg::GradientBased => [1.0; N],
    };
    let (mut l, mut u) = (X_L, X_U);
    for j in 0..N {
        if X_L[j] == X_U[j] {
            continue;
        }
        let delta = |b: Number| (1e-3 * (d[j] * b).abs().max(1.0)).min(1e-4) / d[j];
        if X_L[j] > -1e19 {
            l[j] -= delta(X_L[j]);
        }
        if X_U[j] < 1e19 {
            u[j] += delta(X_U[j]);
        }
    }
    (l, u)
}

/// Upstream `get_curr_violations(scaled = false)` recomputed by hand on the
/// original problem from `(x, z_L, z_U, g, λ)` — none of it read from the
/// solver: the model's own gradient and Jacobian, its declared bounds for
/// the violations, and for `compl_x_*` the relaxed bounds the algorithm's
/// slack is measured to ([`relaxed_bounds`]; upstream's
/// `curr_compl_x_L / obj` at `d = 1` — under user scaling `relaxed_bounds`
/// is pounce's scaled-frame rule, which Ipopt's unscaled-frame relaxation
/// does not share).
fn recompute(
    cfg: Cfg,
    x: &[Number; N],
    z_l: &[Number; N],
    z_u: &[Number; N],
    lambda: &[Number; M],
) -> Viol {
    let (rl, ru) = relaxed_bounds(cfg);
    let kk = k();
    let c = [3.0, 3.0, 1.0, 4.0];
    let g = [x[0] + x[1] + x[3], x[0] - x[2], x[1] + x[3]];
    let mut v = Viol::default();
    for j in 0..N {
        let mut gl = 2.0 * kk * (x[j] - c[j]) - z_l[j] + z_u[j];
        for k in 0..NNZ {
            if JAC_COL[k] as usize == j {
                gl += JAC_VAL[k] * lambda[JAC_ROW[k] as usize];
            }
        }
        // Fixed under make_parameter: upstream reports 0 there.
        v.grad_lag_x[j] = if X_L[j] == X_U[j] { 0.0 } else { gl };
        if X_L[j] > -1e19 {
            v.x_l[j] = (X_L[j] - x[j]).max(0.0);
            v.compl_x_l[j] = (x[j] - rl[j]) * z_l[j];
        }
        if X_U[j] < 1e19 {
            v.x_u[j] = (x[j] - X_U[j]).max(0.0);
            v.compl_x_u[j] = (ru[j] - x[j]) * z_u[j];
        }
    }
    for i in 0..M {
        if G_L[i] == G_U[i] {
            v.g[i] = (g[i] - G_L[i]).abs();
            v.compl_g[i] = -(g[i] - G_L[i]) * lambda[i];
            continue;
        }
        let mut viol: Number = 0.0;
        if G_L[i] > -1e19 {
            viol = viol.max(G_L[i] - g[i]);
            v.compl_g[i] += lambda[i].min(0.0) * (G_L[i] - g[i]);
        }
        if G_U[i] < 1e19 {
            viol = viol.max(g[i] - G_U[i]);
            v.compl_g[i] -= lambda[i].max(0.0) * (g[i] - G_U[i]);
        }
        v.g[i] = viol;
    }
    v
}

fn close_abs(what: &str, got: &[Number], want: &[Number], scale: Number) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9 * (scale + w.abs()),
            "{what}[{i}]: GetIpoptCurrentViolations {g:e}, recomputed {w:e} ({got:?} vs {want:?})"
        );
    }
}

fn close_viol(tag: &str, got: &Viol, want: &Viol) {
    // Entries that vanish at the solution are compared on the problem's
    // own scale (K), not on their own.
    let s = k();
    close_abs(&format!("{tag} x_L_violation"), &got.x_l, &want.x_l, s);
    close_abs(&format!("{tag} x_U_violation"), &got.x_u, &want.x_u, s);
    close_abs(
        &format!("{tag} compl_x_L"),
        &got.compl_x_l,
        &want.compl_x_l,
        s,
    );
    close_abs(
        &format!("{tag} compl_x_U"),
        &got.compl_x_u,
        &want.compl_x_u,
        s,
    );
    close_abs(
        &format!("{tag} grad_lag_x"),
        &got.grad_lag_x,
        &want.grad_lag_x,
        s,
    );
    close_abs(&format!("{tag} constraint violation"), &got.g, &want.g, s);
    close_abs(&format!("{tag} compl_g"), &got.compl_g, &want.compl_g, s);
}

fn check_violations(algorithm: &str, cfg: Cfg) {
    let (out, fires) = solve_relaxed(algorithm, cfg);
    assert!(!fires.is_empty(), "{algorithm} {cfg:?}");
    // Every fire: what the inspector reports is the original problem's
    // violation at the iterate `GetIpoptCurrentIterate` reports. The early
    // fires are where the numbers are large and every scaling factor shows.
    let mut moved = false;
    for (k, f) in fires.iter().enumerate() {
        let v = f.viol.as_ref().unwrap_or_else(|| {
            panic!("{algorithm} {cfg:?}: GetIpoptCurrentViolations refused at fire {k}")
        });
        let want = recompute(cfg, &f.x, &f.z_l, &f.z_u, &f.lambda);
        close_viol(&format!("{algorithm} {cfg:?} fire {k}"), v, &want);
        moved |= want.grad_lag_x.iter().any(|g| g.abs() > 1e-2 * self::k())
            && want.g.iter().any(|g| g.abs() > 1e-3);
    }
    assert!(
        moved,
        "{algorithm} {cfg:?}: no fire with a visible residual"
    );
    // The last fire, against the final x, λ, z IpoptSolve wrote back.
    let last = fires.last().unwrap().viol.clone().unwrap();
    let want = recompute(cfg, &out.x, &out.z_l, &out.z_u, &out.lambda);
    close_viol(
        &format!("{algorithm} {cfg:?} last fire vs IpoptSolve"),
        &last,
        &want,
    );
}

#[test]
fn ipm_violations_are_the_original_problems_under_gradient_scaling() {
    check_violations("interior-point", Cfg::GradientBased);
}

#[test]
fn ipm_violations_are_the_original_problems_under_user_x_scaling() {
    check_violations("interior-point", Cfg::UserX);
}

#[test]
fn sqp_violations_are_the_original_problems_under_gradient_scaling() {
    check_violations("active-set-sqp", Cfg::GradientBased);
}

#[test]
fn sqp_violations_are_the_original_problems_under_user_x_scaling() {
    check_violations("active-set-sqp", Cfg::UserX);
}

/// `scaled = TRUE` keeps upstream's meaning: the algorithm's own space.
/// Under user scaling (obj 3, x `d`, g `s` on the interior-point arm; the
/// SQP arm solves with `d` only) every entry is a fixed factor of its
/// unscaled counterpart: a bound violation `d·viol` (`orig_x_L_violation`),
/// `compl_x = obj·compl_x` (`s̃z̃ = obj·sz`), `∂L̃/∂x̃ = obj·∂L/∂x / d`, a
/// row's violation `s·viol`, `compl_g = obj·compl_g` (`c̃ỹ = s c · obj λ/s`),
/// and at the fixed `x3` (not in the algorithm) factor 1. Checked at every
/// fire, on both arms; the SQP arm used to refuse.
fn check_scaled_violations(algorithm: &str) {
    let d = [2.0, 0.5, 4.0, 1.0];
    let (obj, s) = if algorithm == "interior-point" {
        (3.0, [0.01, 5.0, 1.0])
    } else {
        (1.0, [1.0; M])
    };
    let (_, fires) = solve_relaxed(algorithm, Cfg::UserX);
    let mut seen = [false; 4];
    for (k, f) in fires.iter().enumerate() {
        let (u, sc) = (
            f.viol.clone().unwrap(),
            f.viol_scaled
                .clone()
                .unwrap_or_else(|| panic!("{algorithm} fire {k}: scaled refused")),
        );
        let tag = format!("{algorithm} fire {k} scaled");
        let by = |v: &[Number], f: &dyn Fn(usize) -> Number| -> Vec<Number> {
            v.iter().enumerate().map(|(j, x)| f(j) * x).collect()
        };
        close_abs(&format!("{tag} x_L"), &sc.x_l, &by(&u.x_l, &|j| d[j]), 1e-6);
        close_abs(&format!("{tag} x_U"), &sc.x_u, &by(&u.x_u, &|j| d[j]), 1e-6);
        close_abs(
            &format!("{tag} compl_x_L"),
            &sc.compl_x_l,
            &by(&u.compl_x_l, &|_| obj),
            1e-6,
        );
        close_abs(
            &format!("{tag} compl_x_U"),
            &sc.compl_x_u,
            &by(&u.compl_x_u, &|_| obj),
            1e-6,
        );
        close_abs(
            &format!("{tag} grad_lag_x"),
            &sc.grad_lag_x,
            &by(&u.grad_lag_x, &|j| obj / d[j]),
            1e-6,
        );
        close_abs(
            &format!("{tag} constraint"),
            &sc.g,
            &by(&u.g, &|i| s[i]),
            1e-6,
        );
        close_abs(
            &format!("{tag} compl_g"),
            &sc.compl_g,
            &by(&u.compl_g, &|_| obj),
            1e-6,
        );
        assert_eq!(sc.grad_lag_x[3], 0.0, "fixed variable");
        // Each factor has to show at some fire. The active-set arm's
        // complementarity is exactly 0 at every fire (its multipliers vanish
        // off the working set), so it shows `d` and `s` only.
        seen[0] |= u.x_u[1] > 1e-6;
        seen[1] |= u.grad_lag_x[..3].iter().any(|g| g.abs() > 1e-3);
        seen[2] |= algorithm != "interior-point" || u.compl_x_u[1].abs() > 1e-6;
        seen[3] |= u.g.iter().any(|g| g.abs() > 1e-6);
    }
    assert!(
        seen.iter().all(|&b| b),
        "{algorithm}: factors never show: {seen:?}"
    );
}

#[test]
fn ipm_scaled_violations_are_the_algorithms_own_space() {
    check_scaled_violations("interior-point");
}

#[test]
fn sqp_scaled_violations_are_the_algorithms_own_space() {
    check_scaled_violations("active-set-sqp");
}

// ---------------------------------------------------------------------
// GetIpoptCurrentIterate(scaled = TRUE)
// ---------------------------------------------------------------------

/// `scaled = TRUE` is upstream's `get_curr_iterate(scaled = true)`: the
/// algorithm's iterate in the TNLP's shape. Under user scaling the factors
/// are the ones this test set (obj 3, x `d`, g `s`), so every entry is
/// recomputed from the unscaled read at the same fire: `x̃ = d·x`, `z̃ =
/// 3·z/d`, `g̃ = s·g`, `λ̃ = 3·λ/s` — and at the fixed `x3`, which the
/// algorithm does not hold, upstream's `ResortX` value `x3` and its manual
/// rescale `3·z` (no `d`). It used to ignore `scaled` and return the
/// unscaled iterate, on both arms.
///
/// The active-set SQP arm solves the problem with the per-variable
/// substitution only (it runs neither user nor gradient-based objective /
/// row scaling; only the `obj_scaling_factor` *option* reaches it), so its
/// scaled space has obj 1 and unit rows here: `obj`, `s` are per arm.
fn check_scaled_iterate(algorithm: &str) {
    let d = [2.0, 0.5, 4.0, 3.0];
    let (obj, s) = if algorithm == "interior-point" {
        (3.0, [0.01, 5.0, 1.0])
    } else {
        (1.0, [1.0; M])
    };
    let (_, fires) = solve(algorithm, Cfg::UserX);
    assert!(fires.len() > 1);
    for (k, f) in fires.iter().enumerate() {
        let sc = f.scaled.as_ref().expect("scaled read");
        assert!(sc.ok, "{algorithm} fire {k}: scaled iterate refused");
        let dx = |j: usize| if j == 3 { 1.0 } else { d[j] };
        let want_x: Vec<Number> = (0..N).map(|j| dx(j) * f.x[j]).collect();
        let want_zl: Vec<Number> = (0..N).map(|j| obj * f.z_l[j] / dx(j)).collect();
        let want_zu: Vec<Number> = (0..N).map(|j| obj * f.z_u[j] / dx(j)).collect();
        let want_g: Vec<Number> = (0..M).map(|i| s[i] * f.g[i]).collect();
        let want_l: Vec<Number> = (0..M).map(|i| obj * f.lambda[i] / s[i]).collect();
        let tag = format!("{algorithm} fire {k} scaled");
        close_abs(&format!("{tag} x"), &sc.x, &want_x, 1e-6);
        close_abs(&format!("{tag} z_l"), &sc.z_l, &want_zl, 1e-6);
        close_abs(&format!("{tag} z_u"), &sc.z_u, &want_zu, 1e-6);
        close_abs(&format!("{tag} g"), &sc.g, &want_g, 1e-6);
        close_abs(&format!("{tag} lambda"), &sc.lambda, &want_l, 1e-6);
    }
    // Informative only if the factors show: x̃ ≠ x somewhere, and the fixed
    // variable's multiplier is not zero.
    let last = fires.last().unwrap();
    assert!(last.z_u[3] > 1e-3 && (last.scaled.as_ref().unwrap().x[0] - last.x[0]).abs() > 1e-3);
}

#[test]
fn ipm_scaled_iterate_is_the_algorithms_space_in_the_tnlps_shape() {
    check_scaled_iterate("interior-point");
}

#[test]
fn sqp_scaled_iterate_is_the_algorithms_space_in_the_tnlps_shape() {
    check_scaled_iterate("active-set-sqp");
}

/// `grad_lag_x` comes from the `∇f` and `J` the algorithm evaluated at the
/// iterate (its own `∇L`), not from calling the user's `eval_grad_f` /
/// `eval_jac_g` again: a callback reading every violation at every fire
/// costs exactly the evaluations of one that reads only the iterate. It
/// used to cost one `eval_grad_f` and one `eval_jac_g` per call (two per
/// fire here: scaled and unscaled).
#[test]
fn reading_violations_costs_no_evaluation() {
    for algorithm in ["interior-point", "active-set-sqp"] {
        for cfg in [Cfg::GradientBased, Cfg::UserX] {
            let run = |read: bool| {
                READ_VIOLATIONS.with(|r| r.set(read));
                EVALS.with(|e| e.set((0, 0)));
                let (out, fires) = solve(algorithm, cfg);
                READ_VIOLATIONS.with(|r| r.set(true));
                (EVALS.with(|e| e.get()), out.x, fires.len())
            };
            let (quiet, x0, n0) = run(false);
            let (read, x1, n1) = run(true);
            assert_eq!((n0, x0), (n1, x1), "{algorithm} {cfg:?}: the solve moved");
            assert_eq!(
                quiet, read,
                "{algorithm} {cfg:?}: (grad_f, jac) evaluations, {n1} fires"
            );
        }
    }
}
