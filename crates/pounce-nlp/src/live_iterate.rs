//! The live iterate of the user's problem, for the intermediate-callback
//! window (`GetIpoptCurrentIterate` / `GetIpoptCurrentViolations`,
//! `pounce_algorithm::intermediate::with_current_user_iterate`).
//!
//! Upstream Ipopt's `TNLP::get_curr_iterate` reads the algorithm's iterate
//! and converts it on the spot, through `TNLPAdapter::ResortX` /
//! `ResortBoundMultipliers` / `ResortG`, into the space the user's TNLP
//! declared. pounce cannot do that conversion at read time: the reader runs
//! inside the user's callback, which holds the TNLP mutably, and part of the
//! conversion — the bound multipliers of fixed variables, the rows and
//! columns a presolve layer removed — has to *evaluate* that TNLP. So the
//! algorithm materialises the iterate before the callback, in the space of
//! the TNLP it solves, and each decorator between that TNLP and the user's
//! lifts it one layer on its way through `intermediate_callback`, by the
//! rule its own `finalize_solution` applies. What the user reads at the last
//! fire is then the point `finalize_solution` reports.
//!
//! Lives here rather than in `pounce-algorithm` so the decorators that lift
//! it (`pounce-presolve`'s, which do not depend on the algorithm crate) can
//! reach it.

use pounce_common::types::Number;
use std::cell::RefCell;

/// Iterate of a TNLP in that TNLP's own index spaces (full `n`, full `m`,
/// its `g` order), in the units `finalize_solution` reports: the
/// unscaled-Lagrangian multipliers `∇f + Jᵀλ − z_L + z_U = 0`.
#[derive(Clone, Debug, Default)]
pub struct UserIterate {
    pub x: Vec<Number>,
    pub z_l: Vec<Number>,
    pub z_u: Vec<Number>,
    pub g: Vec<Number>,
    pub lambda: Vec<Number>,
    /// The scaling of the problem the algorithm solves, entry by entry in
    /// this iterate's shape (see [`IterateScaling`]); what
    /// `GetIpoptCurrentIterate` / `GetIpoptCurrentViolations` with
    /// `scaled = TRUE` multiply by. `None`: not known (the scaled readers
    /// then report the unscaled values, scale 1).
    pub scaling: Option<IterateScaling>,
    /// The variable bounds the algorithm measures its slacks to, in this
    /// iterate's units and shape: the declared box widened by
    /// `bound_relax_factor` (and moved by any safe-slack adjustment) for a
    /// column the algorithm holds; the declared bound for one it does not
    /// (a fixed variable: its value; a column presolve removed); `±∞` where
    /// the column has no such bound. Upstream's `compl_x_*` is the
    /// algorithm's `slack·z`, so it is measured to these. Empty: not known
    /// (the declared bounds are used). One residual difference: under
    /// `honor_original_bounds` (off by default) the reported `x` is projected
    /// into the declared box, so `x − x_L^alg` is the projected point's
    /// slack, not the algorithm's.
    pub x_l_alg: Vec<Number>,
    pub x_u_alg: Vec<Number>,
    /// `∇f + Jᵀλ − z_L + z_U` at this iterate, unscaled, in this iterate's
    /// shape, from evaluations the algorithm already made (its own
    /// `∇L` on the interior-point and SQP arms; presolve layers evaluate the
    /// inner problem from the iteration's record). `0` at a column the
    /// algorithm does not hold (upstream `ResortX(…, usefixedvals=false)`).
    /// Empty: not known (`GetIpoptCurrentViolations` then evaluates).
    pub grad_lag_x: Vec<Number>,
}

/// Per-entry factors taking a [`UserIterate`] (and the violations measured
/// at it) to the space the algorithm iterates in — upstream's
/// `NLP_scaling()` applied to the TNLP-shaped quantities
/// `TNLP::get_curr_iterate(scaled = true)` returns:
///
/// * `x̃_j = x_scale_j · x_j`, `x_scale` the per-variable factor (`1` for a
///   column the algorithm does not hold — a fixed variable `make_parameter`
///   removed, a column presolve eliminated: upstream's `ResortX` writes the
///   TNLP's own value there);
/// * `z̃_j = obj · z_j / x_scale_j`, with `z_L` and `z_U` swapped (and
///   negated) at a column the algorithm does not hold when `obj < 0`, as
///   upstream does for fixed variables;
/// * `g̃_i = g_scale_i · g_i`, `λ̃_i = obj · λ_i / g_scale_i` (`1` for a row
///   the algorithm does not hold).
#[derive(Clone, Debug, PartialEq)]
pub struct IterateScaling {
    /// Objective factor (`obj_scaling_factor`, gradient-based times user).
    pub obj: Number,
    /// Per-column factor; length `n`.
    pub x: Vec<Number>,
    /// Whether the algorithm holds column `j` (length `n`).
    pub x_in_alg: Vec<bool>,
    /// Per-row factor; length `m`.
    pub g: Vec<Number>,
}

impl IterateScaling {
    /// Unit scaling for an `n`-column, `m`-row problem whose columns the
    /// algorithm all holds.
    pub fn unit(obj: Number, n: usize, m: usize) -> Self {
        Self {
            obj,
            x: vec![1.0; n],
            x_in_alg: vec![true; n],
            g: vec![1.0; m],
        }
    }

    fn fits(&self, n: usize, m: usize) -> bool {
        self.x.len() == n && self.x_in_alg.len() == n && self.g.len() == m
    }
}

/// `it` in the algorithm's scaled space, by [`IterateScaling`]'s rule —
/// upstream `TNLP::get_curr_iterate(scaled = true)`. Unit scaling when the
/// iterate carries none.
pub fn scaled_iterate(it: &UserIterate) -> UserIterate {
    let (n, m) = (it.x.len(), it.g.len());
    let s = match it.scaling.as_ref().filter(|s| s.fits(n, m)) {
        Some(s) => s.clone(),
        None => IterateScaling::unit(1.0, n, m),
    };
    let mut out = UserIterate {
        x: it.x.iter().zip(&s.x).map(|(v, d)| v * d).collect(),
        z_l: vec![0.0; it.z_l.len()],
        z_u: vec![0.0; it.z_u.len()],
        g: it.g.iter().zip(&s.g).map(|(v, d)| v * d).collect(),
        lambda: it
            .lambda
            .iter()
            .zip(&s.g)
            .map(|(v, d)| s.obj * v / d)
            .collect(),
        scaling: Some(s.clone()),
        // The algorithm's bounds are scaled like `x`.
        x_l_alg: it.x_l_alg.iter().zip(&s.x).map(|(v, d)| v * d).collect(),
        x_u_alg: it.x_u_alg.iter().zip(&s.x).map(|(v, d)| v * d).collect(),
        grad_lag_x: Vec::new(),
    };
    for j in 0..n.min(it.z_l.len()).min(it.z_u.len()) {
        let f = s.obj / s.x[j];
        if !s.x_in_alg[j] && s.obj < 0.0 {
            // Upstream `get_curr_iterate`: "need to swap between z_L and
            // z_U in this case".
            out.z_l[j] = -it.z_u[j] * s.obj;
            out.z_u[j] = -it.z_l[j] * s.obj;
        } else {
            out.z_l[j] = it.z_l[j] * f;
            out.z_u[j] = it.z_u[j] * f;
        }
    }
    out
}

thread_local! {
    static CURRENT: RefCell<Option<UserIterate>> = const { RefCell::new(None) };
}

/// Installs a [`UserIterate`] for the duration of one callback fire and
/// restores whatever was installed before on drop, so a decorator can
/// shadow the iterate with its lifted one while it forwards the callback.
pub struct UserIterateGuard {
    prev: Option<UserIterate>,
}

impl UserIterateGuard {
    pub fn install(it: UserIterate) -> Self {
        let prev = CURRENT.with(|c| c.borrow_mut().replace(it));
        Self { prev }
    }
}

impl Drop for UserIterateGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        CURRENT.with(|c| *c.borrow_mut() = prev);
    }
}

/// Read access to the installed iterate; `None` outside a callback window.
pub fn with_current_user_iterate<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&UserIterate) -> R,
{
    CURRENT.with(|c| c.borrow().as_ref().map(f))
}

/// The objective in the user's own units at the installed iterate, from the
/// `IterStats::obj_value` the algorithm reports alongside it.
///
/// `obj_value` is the algorithm's `curr_f()`: the user's objective times the
/// total objective scale (`obj_scaling_factor`, gradient-based scaling, or a
/// sign flip). The installed iterate carries that scale, so dividing it back
/// out costs no evaluation and gives what upstream passes to
/// `intermediate_callback` (`unscaled_curr_f()`). `None` when no iterate is
/// installed (outside a callback window, during restoration, or for a TNLP
/// whose `wants_current_iterate` is false) or its scale is zero; the caller
/// then decides what to report.
pub fn unscaled_objective(obj_value: Number) -> Option<Number> {
    with_current_user_iterate(|it| it.scaling.as_ref().map_or(1.0, |s| s.obj))
        .filter(|&scale| scale != 0.0)
        .map(|scale| obj_value / scale)
}

/// A copy of the installed iterate, for a decorator about to lift it.
pub fn current_user_iterate() -> Option<UserIterate> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Whether an iterate is installed.
pub fn is_installed() -> bool {
    CURRENT.with(|c| c.borrow().is_some())
}

/// Per-entry violations and complementarities at an iterate, in the shape
/// of upstream `TNLP::get_curr_violations`: full `n` for the first five,
/// full `m` for the last two, zero where a bound or row does not apply.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Violations {
    pub x_l_violation: Vec<Number>,
    pub x_u_violation: Vec<Number>,
    pub compl_x_l: Vec<Number>,
    pub compl_x_u: Vec<Number>,
    pub grad_lag_x: Vec<Number>,
    pub constraint_violation: Vec<Number>,
    pub compl_g: Vec<Number>,
}

/// The declared problem the violations are measured against: the user's
/// own box and row bounds, with the infinity thresholds that mark a side
/// absent.
#[derive(Clone, Copy, Debug)]
pub struct DeclaredBounds<'a> {
    pub x_l: &'a [Number],
    pub x_u: &'a [Number],
    pub g_l: &'a [Number],
    pub g_u: &'a [Number],
    pub lower_inf: Number,
    pub upper_inf: Number,
}

/// `∇f + Jᵀλ − z_L + z_U` at `it`, from the gradient and the Jacobian
/// triplets (0-based) evaluated at `it.x`.
pub fn lagrangian_gradient(
    it: &UserIterate,
    grad_f: &[Number],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[Number],
) -> Vec<Number> {
    let mut s = grad_f.to_vec();
    for ((&r, &c), &v) in jac_rows.iter().zip(jac_cols).zip(jac_vals) {
        if let (Some(&l), Some(sc)) = (it.lambda.get(r), s.get_mut(c)) {
            *sc += v * l;
        }
    }
    for (j, sj) in s.iter_mut().enumerate() {
        *sj += it.z_u.get(j).copied().unwrap_or(0.0) - it.z_l.get(j).copied().unwrap_or(0.0);
    }
    s
}

/// The unscaled violations of upstream `get_curr_violations(scaled=false)`,
/// computed from the user-space iterate against the declared problem:
///
/// * `x_L_violation = max(0, x_L − x)` (upstream's
///   `unscaled_curr_orig_x_L_violation`: the *original*, unrelaxed bound);
/// * `compl_x_L = (x − x_L^alg) · z_L`: upstream's `curr_compl_x_L / obj`,
///   the algorithm's slack — measured to the bound it iterates against, the
///   declared one widened by `bound_relax_factor` ([`UserIterate::x_l_alg`])
///   — times its multiplier. The declared bound when the iterate carries no
///   algorithm bound for the column. **Not upstream's number under per-variable
///   user scaling:** Ipopt relaxes the *unscaled* bound before scaling it
///   (`IpOrigIpoptNLP.cpp`, relax then scale), so its widening is
///   `min(brf·max(1,|b|), cvt)`; pounce relaxes in `ScalingTnlp`'s `x̃ = d ⊙ x`
///   frame, `min(brf·max(1,|d·b|), cvt)/d`. They agree at `d = 1`. This
///   reports pounce's own slack, which is the consistent choice; the
///   relaxation frame itself predates the live iterate;
/// * `grad_lag_x = ∇f + Jᵀλ − z_L + z_U` (`grad_lag`; the caller evaluates
///   it), zeroed at fixed variables when `zero_fixed` — upstream's
///   `ResortX(…, usefixedvals=false)` under `make_parameter`;
/// * equality rows: violation `|g − g_L|`, `compl_g = −(g − g_L) · λ`;
///   inequality rows: violation `max(0, g_L − g, g − g_U)`,
///   `compl_g = min(λ, 0)(g_L − g) − max(λ, 0)(g − g_U)` over the finite
///   sides — upstream's signs, taken from `ResortBoundMultipliers`.
///
/// A side at or beyond the infinity threshold contributes `0`.
pub fn user_violations(
    it: &UserIterate,
    b: &DeclaredBounds<'_>,
    grad_lag: Option<Vec<Number>>,
    zero_fixed: bool,
) -> Violations {
    let n = it.x.len();
    let m = it.g.len();
    let mut v = Violations {
        x_l_violation: vec![0.0; n],
        x_u_violation: vec![0.0; n],
        compl_x_l: vec![0.0; n],
        compl_x_u: vec![0.0; n],
        grad_lag_x: grad_lag.unwrap_or_else(|| vec![0.0; n]),
        constraint_violation: vec![0.0; m],
        compl_g: vec![0.0; m],
    };
    // The bound the algorithm's slack is measured to, where it has one.
    let alg = |v: &[Number], j: usize, declared: Number| {
        v.get(j)
            .copied()
            .filter(|b| b.is_finite() && v.len() == n)
            .unwrap_or(declared)
    };
    for j in 0..n {
        let x = it.x[j];
        let (lo, up) = (b.x_l[j], b.x_u[j]);
        if lo > b.lower_inf {
            v.x_l_violation[j] = (lo - x).max(0.0);
            v.compl_x_l[j] = (x - alg(&it.x_l_alg, j, lo)) * it.z_l[j];
        }
        if up < b.upper_inf {
            v.x_u_violation[j] = (x - up).max(0.0);
            v.compl_x_u[j] = (alg(&it.x_u_alg, j, up) - x) * it.z_u[j];
        }
        if zero_fixed
            && lo == up
            && let Some(gj) = v.grad_lag_x.get_mut(j)
        {
            *gj = 0.0;
        }
    }
    for i in 0..m {
        let (g, lam) = (it.g[i], it.lambda[i]);
        let (lo, up) = (b.g_l[i], b.g_u[i]);
        if lo == up {
            v.constraint_violation[i] = (g - lo).abs();
            v.compl_g[i] = -(g - lo) * lam;
            continue;
        }
        let mut viol: Number = 0.0;
        let mut compl = 0.0;
        if lo > b.lower_inf {
            viol = viol.max(lo - g);
            compl += lam.min(0.0) * (lo - g);
        }
        if up < b.upper_inf {
            viol = viol.max(g - up);
            compl -= lam.max(0.0) * (g - up);
        }
        v.constraint_violation[i] = viol;
        v.compl_g[i] = compl;
    }
    v
}

/// The violations of upstream `get_curr_violations(scaled = true)` from the
/// unscaled ones at `it`, by the iterate's [`IterateScaling`] (unit when it
/// carries none) — every entry is a fixed factor of its unscaled
/// counterpart (`IpTNLP.cpp` 745-1040, `curr_*(…, scaled)`):
///
/// * `x_L_violation`, `x_U_violation`: `x_scale ·` (`orig_x_L_violation`,
///   the declared bound scaled like `x`);
/// * `compl_x_*`, `compl_g`: `obj ·` (upstream's unscaled ones divide the
///   algorithm's `s·z`, `c·y` by `obj`; the `d` and row factors cancel);
/// * `grad_lag_x`: `obj / x_scale ·` (`unapply_grad_obj_scaling`);
/// * constraint violation: `g_scale ·` (`|c̃|`, and the `d` violation is
///   formed scaled and unscaled by `unapply_vector_scaling_d`).
///
/// One rule for both arms and for presolve, whose removed entries carry
/// factor 1 (objective factor for the complementarities and `grad_lag_x`).
pub fn scale_violations(mut v: Violations, it: &UserIterate) -> Violations {
    let (n, m) = (v.x_l_violation.len(), v.constraint_violation.len());
    let s = match it.scaling.as_ref().filter(|s| s.fits(n, m)) {
        Some(s) => s.clone(),
        None => IterateScaling::unit(1.0, n, m),
    };
    for j in 0..n {
        v.x_l_violation[j] *= s.x[j];
        v.x_u_violation[j] *= s.x[j];
        v.compl_x_l[j] *= s.obj;
        v.compl_x_u[j] *= s.obj;
        if let Some(g) = v.grad_lag_x.get_mut(j) {
            *g *= s.obj / s.x[j];
        }
    }
    for i in 0..m {
        v.constraint_violation[i] *= s.g[i];
        v.compl_g[i] *= s.obj;
    }
    v
}

/// What `GetIpoptCurrentViolations` reports: [`user_violations`] at `it`,
/// and with `scaled` [`scale_violations`] of them.
pub fn current_violations(
    it: &UserIterate,
    b: &DeclaredBounds<'_>,
    grad_lag: Option<Vec<Number>>,
    zero_fixed: bool,
    scaled: bool,
) -> Violations {
    let v = user_violations(it, b, grad_lag, zero_fixed);
    if scaled { scale_violations(v, it) } else { v }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn it(x: Number) -> UserIterate {
        UserIterate {
            x: vec![x],
            ..Default::default()
        }
    }

    /// Upstream `get_curr_iterate(scaled = true)` at a fixed variable under
    /// a negative objective factor: `z̃_L = −obj·z_U`, `z̃_U = −obj·z_L`;
    /// at a column the algorithm holds, `obj·z/d` with no swap.
    #[test]
    fn a_negative_objective_factor_swaps_only_a_fixed_variables_multipliers() {
        let it = UserIterate {
            x: vec![2.0, 1.0],
            z_l: vec![0.5, 0.0],
            z_u: vec![0.25, 4.0],
            g: vec![3.0],
            lambda: vec![-1.0],
            scaling: Some(IterateScaling {
                obj: -2.0,
                x: vec![4.0, 1.0],
                x_in_alg: vec![true, false],
                g: vec![10.0],
            }),
            ..Default::default()
        };
        let s = scaled_iterate(&it);
        assert_eq!(s.x, vec![8.0, 1.0]);
        assert_eq!(s.z_l, vec![-0.25, 8.0]);
        assert_eq!(s.z_u, vec![-0.125, 0.0]);
        assert_eq!(s.g, vec![30.0]);
        assert_eq!(s.lambda, vec![0.2]);
    }

    #[test]
    fn a_nested_guard_restores_the_outer_iterate() {
        assert!(!is_installed());
        {
            let _outer = UserIterateGuard::install(it(1.0));
            {
                let _inner = UserIterateGuard::install(it(2.0));
                assert_eq!(with_current_user_iterate(|u| u.x[0]), Some(2.0));
            }
            assert_eq!(with_current_user_iterate(|u| u.x[0]), Some(1.0));
        }
        assert!(!is_installed());
    }
}
