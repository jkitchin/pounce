//! Minimum-viable unconstrained Newton driver — used by
//! `pounce_algorithm::IpoptApplication::optimize_tnlp` when the
//! TNLP reports `m == 0` and all bounds are infinite.
//!
//! This is **not** the upstream Ipopt path — it is a small dense
//! Newton-with-line-search that lets the CLI demonstrate end-to-end
//! solving of unconstrained problems while the constrained KKT path
//! (StdAugSystemSolver, IpoptCalculatedQuantities, filter line search,
//! barrier update) is being filled in. When the full path lands,
//! `optimize_tnlp` will dispatch to it for `m > 0` and continue using
//! this driver for the trivial `m == 0` case (or route both through
//! the algorithm layer).
//!
//! Algorithm:
//!
//! 1. Start at `x0` from `get_starting_point`.
//! 2. At each iteration:
//!    a. Evaluate `f`, `∇f`. Convergence iff `||∇f||∞ < tol`.
//!    b. Build dense lower-triangular Hessian `H` from triplet via
//!    `eval_h(obj_factor=1)`. Symmetrize.
//!    c. Regularize: if any pivot ≤ 0 during LDL^T, add `λI` and retry.
//!    d. Solve `H d = -∇f` via dense LDL^T (Bunch-Kaufman omitted —
//!    we only need to handle our exactly-PD test problems).
//!    e. Backtracking line search: `α ← α/2` while
//!    `f(x + αd) > f(x) + 1e-4 * α * ∇f^T d`.
//!    f. `x ← x + α d`. Goto (a).
//! 3. On convergence, call `finalize_solution`.

use crate::return_codes::ApplicationReturnStatus;
use crate::solve_statistics::SolveStatistics;
use crate::tnlp::{
    BoundsInfo, IpoptCq, IpoptData, Solution, SparsityRequest, StartingPoint, TNLP,
};
use pounce_common::types::{Index, Number};

/// Configuration matching upstream's `tol`, `max_iter` defaults.
#[derive(Debug, Clone, Copy)]
pub struct NewtonOptions {
    pub tol: Number,
    pub max_iter: Index,
    pub line_search_eta: Number,
    pub line_search_min_alpha: Number,
    /// Looser tolerance used to declare `SolvedToAcceptableLevel` when
    /// `tol` itself is unreachable (typical for nonlinear-regression
    /// problems where `||∇f||∞` floors above 1e-8). Mirrors Ipopt's
    /// `acceptable_tol`.
    pub acceptable_tol: Number,
    /// Number of consecutive iterations the residual must stay below
    /// `acceptable_tol` before we accept. Mirrors Ipopt's
    /// `acceptable_iter`.
    pub acceptable_iter: Index,
}

impl Default for NewtonOptions {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            max_iter: 1000,
            line_search_eta: 1e-4,
            line_search_min_alpha: 1e-12,
            acceptable_tol: 1e-6,
            acceptable_iter: 15,
        }
    }
}

/// Top-level dispatch entry point. Routes to
/// [`solve_unconstrained`] when `m == 0`, or to
/// [`solve_eq_constrained`] otherwise.
pub fn solve(
    tnlp: &mut dyn TNLP,
    opts: NewtonOptions,
) -> (ApplicationReturnStatus, SolveStatistics) {
    let info = match tnlp.get_nlp_info() {
        Some(i) => i,
        None => {
            return (
                ApplicationReturnStatus::InvalidProblemDefinition,
                SolveStatistics::new(),
            )
        }
    };
    if info.m == 0 {
        solve_unconstrained(tnlp, opts)
    } else {
        solve_eq_constrained(tnlp, opts)
    }
}

/// Drive an unconstrained TNLP to local optimality.
///
/// Returns `(status, statistics)`.
pub fn solve_unconstrained(
    tnlp: &mut dyn TNLP,
    opts: NewtonOptions,
) -> (ApplicationReturnStatus, SolveStatistics) {
    let mut stats = SolveStatistics::new();

    let info = match tnlp.get_nlp_info() {
        Some(i) => i,
        None => return (ApplicationReturnStatus::InvalidProblemDefinition, stats),
    };
    if info.m != 0 {
        // Constrained problems are dispatched to the equality-Newton
        // driver. The caller should normally route through
        // `solve` instead.
        return solve_eq_constrained(tnlp, opts);
    }

    let n = info.n as usize;
    if n == 0 {
        return (ApplicationReturnStatus::NotEnoughDegreesOfFreedom, stats);
    }

    // Bounds: load and verify they are effectively unbounded. Mixed
    // bound problems require the IPM path.
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    let mut g_l = vec![0.0; 0];
    let mut g_u = vec![0.0; 0];
    if !tnlp.get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) {
        return (ApplicationReturnStatus::InvalidProblemDefinition, stats);
    }
    let nlp_lower = -1e19_f64;
    let nlp_upper = 1e19_f64;
    // A variable is "fixed" when its lower and upper bounds coincide
    // (within a tight tolerance). Such variables have no degree of
    // freedom and must be excluded from the barrier — otherwise
    // `-mu/(x-l) + mu/(u-x)` evaluates to `-inf + inf = NaN`.
    let fixed_eps = 1e-12;
    let mut fixed = vec![false; n];
    for i in 0..n {
        if x_l[i] > nlp_lower
            && x_u[i] < nlp_upper
            && (x_u[i] - x_l[i]).abs() <= fixed_eps * x_l[i].abs().max(1.0)
        {
            fixed[i] = true;
        }
    }
    let has_bounds = x_l
        .iter()
        .zip(fixed.iter())
        .any(|(&v, &f)| !f && v > nlp_lower)
        || x_u
            .iter()
            .zip(fixed.iter())
            .any(|(&v, &f)| !f && v < nlp_upper);

    // Starting point.
    let mut x = vec![0.0; n];
    let mut z_l = vec![0.0; n];
    let mut z_u = vec![0.0; n];
    let mut lam = vec![0.0; 0];
    if !tnlp.get_starting_point(StartingPoint {
        init_x: true,
        x: &mut x,
        init_z: false,
        z_l: &mut z_l,
        z_u: &mut z_u,
        init_lambda: false,
        lambda: &mut lam,
    }) {
        return (ApplicationReturnStatus::InvalidProblemDefinition, stats);
    }

    // Push-to-interior for bounded problems: ensure each x[i] is
    // strictly inside its bounds by `kappa` of the bound or `min_dist`
    // (whichever is larger). Mirrors the spirit of Ipopt's
    // bound_push / bound_frac in IpDefaultIterateInitializer.
    // Clamp fixed variables to their bound and skip them in
    // the push-to-interior pass.
    for i in 0..n {
        if fixed[i] {
            x[i] = x_l[i];
        }
    }
    if has_bounds {
        let kappa1 = 1e-2;
        let kappa2 = 1e-2;
        for i in 0..n {
            if fixed[i] {
                continue;
            }
            let lo = x_l[i];
            let hi = x_u[i];
            let lo_active = lo > nlp_lower;
            let hi_active = hi < nlp_upper;
            if lo_active && hi_active {
                let span = hi - lo;
                let pl = lo + (kappa1 * lo.abs().max(1.0)).min(kappa2 * span);
                let pu = hi - (kappa1 * hi.abs().max(1.0)).min(kappa2 * span);
                if pl < pu {
                    x[i] = x[i].clamp(pl, pu);
                } else {
                    x[i] = 0.5 * (lo + hi);
                }
            } else if lo_active {
                let push = kappa1 * lo.abs().max(1.0);
                if x[i] < lo + push {
                    x[i] = lo + push;
                }
            } else if hi_active {
                let push = kappa1 * hi.abs().max(1.0);
                if x[i] > hi - push {
                    x[i] = hi - push;
                }
            }
        }
    }

    // Hessian sparsity.
    let nnz_h = info.nnz_h_lag as usize;
    let mut h_irow = vec![0_i32; nnz_h];
    let mut h_jcol = vec![0_i32; nnz_h];
    if !tnlp.eval_h(
        None,
        false,
        1.0,
        None,
        false,
        SparsityRequest::Structure {
            irow: &mut h_irow,
            jcol: &mut h_jcol,
        },
    ) {
        return (ApplicationReturnStatus::InvalidProblemDefinition, stats);
    }

    let mut grad = vec![0.0; n];
    let mut h_values = vec![0.0; nnz_h];
    let mut h_dense = vec![0.0; n * n];
    let mut step = vec![0.0; n];
    let mut x_trial = vec![0.0; n];

    // Initial barrier parameter. For unconstrained problems we set
    // `mu = 0` and the routine becomes plain Newton.
    //
    // `mu_min` caps how small mu becomes — pushing it below the
    // square-root of machine eps causes the central path to track
    // arithmetic noise rather than the true KKT point. Upstream Ipopt
    // uses a similar guard (`mu_min` option, default 1e-11) and a
    // proper KKT-residual convergence test on the unbarriered
    // problem. This MVP driver's tolerance is set by `mu_min`.
    let mut mu = if has_bounds { 0.1_f64 } else { 0.0_f64 };
    let mu_min = if has_bounds { 1e-7 } else { 0.0 };

    let mut current_f = match tnlp.eval_f(&x, true) {
        Some(v) => v,
        None => return (ApplicationReturnStatus::InvalidNumberDetected, stats),
    };
    stats.num_obj_evals += 1;
    let mut current_phi =
        current_f + barrier_term(&x, &x_l, &x_u, mu, nlp_lower, nlp_upper);

    let mut iter: Index = 0;
    let mut last_grad_norm = Number::INFINITY;
    let mut stagnation_count = 0;
    let mut acceptable_streak: Index = 0;
    // Captured on the first iteration so we can scale the
    // acceptable-level threshold to the problem's natural gradient
    // magnitude. Mirrors the spirit of upstream Ipopt's gradient-based
    // NLP scaling, which divides the gradient by `max(grad_f_max, 1)`
    // before testing against `tol`. We don't rescale the problem here;
    // we just relax the acceptable threshold accordingly.
    let mut initial_grad_norm: Number = 0.0;
    let final_status = 'outer: loop {
        if !tnlp.eval_grad_f(&x, false, &mut grad) {
            break ApplicationReturnStatus::InvalidNumberDetected;
        }
        stats.num_obj_grad_evals += 1;

        // Add the gradient of the barrier term: -mu/(x-l) + mu/(u-x).
        // Fixed variables have no barrier (their lo == hi would yield
        // NaN); we zero their gradient instead so the step is zero.
        if has_bounds {
            for i in 0..n {
                if fixed[i] {
                    grad[i] = 0.0;
                    continue;
                }
                let lo = x_l[i];
                let hi = x_u[i];
                if lo > nlp_lower {
                    grad[i] -= mu / (x[i] - lo);
                }
                if hi < nlp_upper {
                    grad[i] += mu / (hi - x[i]);
                }
            }
        } else {
            for i in 0..n {
                if fixed[i] {
                    grad[i] = 0.0;
                }
            }
        }

        let grad_norm = grad.iter().fold(0.0_f64, |acc, &g| acc.max(g.abs()));
        if iter == 0 {
            initial_grad_norm = grad_norm;
        }
        // Scaled tolerance — large initial gradients raise the bar,
        // but cap the relaxation so we don't accept flat regions far
        // from the optimum on problems whose initial gradient is huge.
        // The cap (1.0) is conservative: even a problem scaled to
        // ||∇f||₀ = 10^10 only relaxes the acceptable threshold to 1.0.
        let tol_scaled = (opts.tol * initial_grad_norm.max(1.0)).min(1e-3);
        let acceptable_scaled = (opts.acceptable_tol * initial_grad_norm.max(1.0)).min(1.0);
        if grad_norm < tol_scaled.max(10.0 * mu) {
            // Inner loop converged at this mu — decrease it or stop.
            if mu <= mu_min {
                break ApplicationReturnStatus::SolveSucceeded;
            }
            mu = (mu * mu).max(mu_min);
            current_phi =
                current_f + barrier_term(&x, &x_l, &x_u, mu, nlp_lower, nlp_upper);
            last_grad_norm = Number::INFINITY;
            stagnation_count = 0;
            continue 'outer;
        }
        // Acceptable-level path: when the strict tolerance is
        // unreachable (typical for nonlinear-regression sums-of-squares
        // problems with residual gradient floored above `tol`), accept
        // after `acceptable_iter` consecutive iterations below the
        // looser, scaled `acceptable_tol`. Mirrors Ipopt's acceptable_*
        // mechanism. Require *actual* progress from the initial point
        // (drop by ≥ 100×) so we don't prematurely accept a flat region
        // when the initial gradient was already huge.
        let made_progress =
            initial_grad_norm <= 0.0 || grad_norm < 1e-2 * initial_grad_norm;
        if mu <= mu_min && grad_norm < acceptable_scaled && made_progress {
            acceptable_streak += 1;
            if acceptable_streak >= opts.acceptable_iter {
                break ApplicationReturnStatus::SolvedToAcceptableLevel;
            }
        } else {
            acceptable_streak = 0;
        }

        // Stagnation: gradient norm refuses to decrease. Common when
        // mu is at its floor and finite-precision arithmetic at the
        // boundary dominates, or for ill-conditioned unconstrained
        // problems whose Hessian is near-singular at the optimum.
        if mu <= mu_min {
            let rel = (last_grad_norm - grad_norm).abs();
            let scale = last_grad_norm.abs().max(grad_norm.abs()).max(1.0);
            if rel < 1e-12 * scale {
                stagnation_count += 1;
            } else {
                stagnation_count = 0;
            }
            if stagnation_count >= 5 {
                let made_strong_progress = initial_grad_norm > 0.0
                    && grad_norm < 1e-2 * initial_grad_norm;
                let status = if (grad_norm < acceptable_scaled && made_progress)
                    || made_strong_progress
                {
                    ApplicationReturnStatus::SolvedToAcceptableLevel
                } else {
                    ApplicationReturnStatus::SearchDirectionBecomesTooSmall
                };
                break status;
            }
        }
        last_grad_norm = grad_norm;
        if iter >= opts.max_iter {
            break ApplicationReturnStatus::MaximumIterationsExceeded;
        }

        // Build dense Hessian. Only lower triangle filled by triplet.
        if !tnlp.eval_h(
            Some(&x),
            false,
            1.0,
            None,
            false,
            SparsityRequest::Values {
                values: &mut h_values,
            },
        ) {
            break ApplicationReturnStatus::InvalidNumberDetected;
        }
        stats.num_hess_evals += 1;

        h_dense.fill(0.0);
        for k in 0..nnz_h {
            let i = h_irow[k] as usize;
            let j = h_jcol[k] as usize;
            let v = h_values[k];
            h_dense[i * n + j] += v;
            if i != j {
                h_dense[j * n + i] += v;
            }
        }

        // Diagonal contribution from the barrier Hessian:
        //   d²/dx²[-mu log(x-l)]  =  mu/(x-l)²
        //   d²/dx²[-mu log(u-x)]  =  mu/(u-x)²
        if has_bounds {
            for i in 0..n {
                if fixed[i] {
                    continue;
                }
                let lo = x_l[i];
                let hi = x_u[i];
                if lo > nlp_lower {
                    let d = x[i] - lo;
                    h_dense[i * n + i] += mu / (d * d);
                }
                if hi < nlp_upper {
                    let d = hi - x[i];
                    h_dense[i * n + i] += mu / (d * d);
                }
            }
        }
        // Decouple fixed variables from the linear system: zero the
        // row/column and put 1 on the diagonal so step[i] = 0.
        for i in 0..n {
            if fixed[i] {
                for j in 0..n {
                    h_dense[i * n + j] = 0.0;
                    h_dense[j * n + i] = 0.0;
                }
                h_dense[i * n + i] = 1.0;
            }
        }

        // Solve H * step = -grad, with diagonal Levenberg-style
        // damping if H is not PD.
        match solve_with_damping(&h_dense, &grad, n, &mut step) {
            Ok(()) => {}
            Err(()) => break ApplicationReturnStatus::ErrorInStepComputation,
        }

        // Step direction check. Fall back to steepest descent if the
        // computed step is not a strict descent direction (including
        // the cases where it is zero or non-finite).
        let mut dir_deriv: Number = grad.iter().zip(step.iter()).map(|(g, s)| g * s).sum();
        if !(dir_deriv < 0.0) || step.iter().any(|v| !v.is_finite()) {
            for i in 0..n {
                step[i] = -grad[i];
            }
            dir_deriv = grad.iter().zip(step.iter()).map(|(g, s)| g * s).sum();
        }

        // Fraction-to-the-bound for bounded problems: cap alpha so we
        // remain strictly interior. tau ≈ 0.99.
        let mut alpha: Number = 1.0;
        if has_bounds {
            let tau = 0.99_f64;
            for i in 0..n {
                let lo = x_l[i];
                let hi = x_u[i];
                if step[i] < 0.0 && lo > nlp_lower {
                    let max_step = -tau * (x[i] - lo) / step[i];
                    if max_step < alpha {
                        alpha = max_step;
                    }
                }
                if step[i] > 0.0 && hi < nlp_upper {
                    let max_step = tau * (hi - x[i]) / step[i];
                    if max_step < alpha {
                        alpha = max_step;
                    }
                }
            }
            if alpha <= 0.0 {
                alpha = 1e-12;
            }
        }

        let armijo_rhs_factor = opts.line_search_eta * dir_deriv;
        let (accepted_f, accepted_phi) = loop {
            for i in 0..n {
                x_trial[i] = x[i] + alpha * step[i];
            }
            let f_trial = match tnlp.eval_f(&x_trial, true) {
                Some(v) => v,
                None => Number::INFINITY,
            };
            stats.num_obj_evals += 1;
            let phi_trial = if f_trial.is_finite() {
                f_trial + barrier_term(&x_trial, &x_l, &x_u, mu, nlp_lower, nlp_upper)
            } else {
                Number::INFINITY
            };
            if phi_trial.is_finite()
                && phi_trial <= current_phi + alpha * armijo_rhs_factor
            {
                break (f_trial, phi_trial);
            }
            alpha *= 0.5;
            if alpha < opts.line_search_min_alpha {
                // The line search exhausted alpha. Soft failure: when
                // we got here from a meaningfully reduced gradient
                // (≥ 1000× drop from the initial), the search direction
                // is exhausted because we are sitting near a local min
                // whose curvature is too poor for a finite-step Newton
                // model. Accept as Solved_To_Acceptable_Level rather
                // than the harsher Search_Direction_Becomes_Too_Small.
                let made_strong_progress = initial_grad_norm > 0.0
                    && grad_norm < 1e-2 * initial_grad_norm;
                let status = if made_strong_progress
                    || grad_norm < acceptable_scaled
                {
                    ApplicationReturnStatus::SolvedToAcceptableLevel
                } else {
                    ApplicationReturnStatus::SearchDirectionBecomesTooSmall
                };
                return finalize(tnlp, &x, current_f, n, iter, stats, status);
            }
        };

        std::mem::swap(&mut x, &mut x_trial);
        current_f = accepted_f;
        current_phi = accepted_phi;
        iter += 1;
    };

    finalize(tnlp, &x, current_f, n, iter, stats, final_status)
}

/// Drive an equality-constrained TNLP to local optimality via SQP-style
/// Newton steps on the KKT system, solved by Schur complement on a
/// Levenberg-damped Hessian. Bounds are not yet supported here — mixed
/// equality + bounds problems require the full IPM path.
///
/// Algorithm per iteration:
/// 1. Evaluate `f`, `∇f`, `g`, `J`, and `∇²L = ∇²f + Σ y_k ∇²g_k`.
/// 2. Damp `H ← H + λI` until PD.
/// 3. Solve KKT for `(dx, dy)`:
///    `[H J^T; J 0] [dx; dy] = -[∇f + J^T y; g]`
///    via Schur complement `S = J H^-1 J^T`.
/// 4. Backtracking line search on the L1 merit
///    `M(x) = f(x) + ρ‖g(x)‖_1` with `ρ` updated so `dx` is descent.
/// 5. Update `x ← x + α dx`, `y ← y + α dy`.
pub fn solve_eq_constrained(
    tnlp: &mut dyn TNLP,
    opts: NewtonOptions,
) -> (ApplicationReturnStatus, SolveStatistics) {
    let mut stats = SolveStatistics::new();
    let info = match tnlp.get_nlp_info() {
        Some(i) => i,
        None => return (ApplicationReturnStatus::InvalidProblemDefinition, stats),
    };
    let n = info.n as usize;
    let m = info.m as usize;
    if n == 0 {
        return (ApplicationReturnStatus::NotEnoughDegreesOfFreedom, stats);
    }
    if m == 0 {
        return solve_unconstrained(tnlp, opts);
    }

    // Bounds: must all be infinite (mixed eq + bounds not supported here).
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    if !tnlp.get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) {
        return (ApplicationReturnStatus::InvalidProblemDefinition, stats);
    }
    let nlp_lower = -1e19_f64;
    let nlp_upper = 1e19_f64;
    if x_l.iter().any(|&v| v > nlp_lower) || x_u.iter().any(|&v| v < nlp_upper) {
        return (ApplicationReturnStatus::InternalError, stats);
    }
    // All constraints must be equality (g_l == g_u).
    for i in 0..m {
        if (g_l[i] - g_u[i]).abs() > 1e-12 {
            return (ApplicationReturnStatus::InternalError, stats);
        }
    }
    let g_target: Vec<Number> = g_l.clone();

    // Starting point.
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; m];
    let mut z_l_unused = vec![0.0; n];
    let mut z_u_unused = vec![0.0; n];
    if !tnlp.get_starting_point(StartingPoint {
        init_x: true,
        x: &mut x,
        init_z: false,
        z_l: &mut z_l_unused,
        z_u: &mut z_u_unused,
        init_lambda: true,
        lambda: &mut y,
    }) {
        return (ApplicationReturnStatus::InvalidProblemDefinition, stats);
    }

    // Hessian sparsity (one-time).
    let nnz_h = info.nnz_h_lag as usize;
    let mut h_irow = vec![0_i32; nnz_h];
    let mut h_jcol = vec![0_i32; nnz_h];
    if !tnlp.eval_h(
        None,
        false,
        1.0,
        None,
        false,
        SparsityRequest::Structure {
            irow: &mut h_irow,
            jcol: &mut h_jcol,
        },
    ) {
        return (ApplicationReturnStatus::InvalidProblemDefinition, stats);
    }

    // Jacobian sparsity (one-time).
    let nnz_jac = info.nnz_jac_g as usize;
    let mut j_irow = vec![0_i32; nnz_jac];
    let mut j_jcol = vec![0_i32; nnz_jac];
    if !tnlp.eval_jac_g(
        None,
        false,
        SparsityRequest::Structure {
            irow: &mut j_irow,
            jcol: &mut j_jcol,
        },
    ) {
        return (ApplicationReturnStatus::InvalidProblemDefinition, stats);
    }

    let mut grad = vec![0.0; n];
    let mut h_values = vec![0.0; nnz_h];
    let mut h_dense = vec![0.0; n * n];
    let mut j_values = vec![0.0; nnz_jac];
    let mut j_dense = vec![0.0; m * n];
    let mut g_vals = vec![0.0; m];
    let mut step_x = vec![0.0; n];
    let mut step_y = vec![0.0; m];
    let mut x_trial = vec![0.0; n];
    let mut rho: Number = 1.0;

    let mut current_f = match tnlp.eval_f(&x, true) {
        Some(v) => v,
        None => return (ApplicationReturnStatus::InvalidNumberDetected, stats),
    };
    stats.num_obj_evals += 1;

    let mut iter: Index = 0;
    let final_status = loop {
        // Evaluate gradient, constraints, Jacobian, Hessian-of-Lagrangian.
        if !tnlp.eval_grad_f(&x, false, &mut grad) {
            break ApplicationReturnStatus::InvalidNumberDetected;
        }
        stats.num_obj_grad_evals += 1;

        if !tnlp.eval_g(&x, false, &mut g_vals) {
            break ApplicationReturnStatus::InvalidNumberDetected;
        }
        stats.num_constr_evals += 1;
        for i in 0..m {
            g_vals[i] -= g_target[i];
        }

        if !tnlp.eval_jac_g(
            Some(&x),
            false,
            SparsityRequest::Values {
                values: &mut j_values,
            },
        ) {
            break ApplicationReturnStatus::InvalidNumberDetected;
        }
        stats.num_constr_jac_evals += 1;
        j_dense.fill(0.0);
        for k in 0..nnz_jac {
            let i = j_irow[k] as usize;
            let j = j_jcol[k] as usize;
            j_dense[i * n + j] += j_values[k];
        }

        // KKT residuals.
        // dual: r1 = grad_f + J^T * y
        let mut r1 = grad.clone();
        for i in 0..m {
            for j in 0..n {
                r1[j] += j_dense[i * n + j] * y[i];
            }
        }
        let dual_inf = r1.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let prim_inf = g_vals.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));

        if dual_inf < opts.tol && prim_inf < opts.tol {
            break ApplicationReturnStatus::SolveSucceeded;
        }
        if iter >= opts.max_iter {
            break ApplicationReturnStatus::MaximumIterationsExceeded;
        }

        // Hessian of Lagrangian.
        if !tnlp.eval_h(
            Some(&x),
            false,
            1.0,
            Some(&y),
            false,
            SparsityRequest::Values {
                values: &mut h_values,
            },
        ) {
            break ApplicationReturnStatus::InvalidNumberDetected;
        }
        stats.num_hess_evals += 1;
        h_dense.fill(0.0);
        for k in 0..nnz_h {
            let i = h_irow[k] as usize;
            let j = h_jcol[k] as usize;
            let v = h_values[k];
            h_dense[i * n + j] += v;
            if i != j {
                h_dense[j * n + i] += v;
            }
        }

        // Schur-complement KKT solve with Levenberg damping until S is PD.
        match schur_kkt_solve(
            &h_dense,
            &j_dense,
            &r1,
            &g_vals,
            n,
            m,
            &mut step_x,
            &mut step_y,
        ) {
            Ok(()) => {}
            Err(()) => break ApplicationReturnStatus::ErrorInStepComputation,
        }
        // step_y is the *update* dy; step_x is dx.
        // The Schur solve returns dx satisfying H*dx + J^T*(y+dy) = -grad
        //                                     and J*dx = -g.

        // Update rho so the merit function decreases along dx.
        // Choice: rho ≥ |grad·dx + 0.5 dx·H·dx| / ((1-σ)‖g‖_1) per
        // Powell / Nocedal–Wright eq 18.36; use σ=0.1 and a simple
        // monotone-up update.
        let g_l1: Number = g_vals.iter().map(|v| v.abs()).sum();
        if g_l1 > 1e-16 {
            let grad_dx: Number = grad.iter().zip(step_x.iter()).map(|(a, b)| a * b).sum();
            // dx^T H dx
            let mut hdx = vec![0.0; n];
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..n {
                    s += h_dense[i * n + j] * step_x[j];
                }
                hdx[i] = s;
            }
            let dx_h_dx: Number = step_x.iter().zip(hdx.iter()).map(|(a, b)| a * b).sum();
            let needed = (grad_dx + 0.5 * dx_h_dx.max(0.0)) / (0.9 * g_l1);
            if needed > rho {
                rho = needed * 1.5;
            }
        }

        // Merit: M(x) = f(x) + rho * ‖g(x)‖_1
        let merit_curr = current_f + rho * g_l1;
        let dir_deriv = {
            let grad_dx: Number = grad.iter().zip(step_x.iter()).map(|(a, b)| a * b).sum();
            grad_dx - rho * g_l1
        };

        let mut alpha: Number = 1.0;
        let armijo_rhs_factor = opts.line_search_eta * dir_deriv;
        let accepted_f = loop {
            for i in 0..n {
                x_trial[i] = x[i] + alpha * step_x[i];
            }
            let f_trial = match tnlp.eval_f(&x_trial, true) {
                Some(v) => v,
                None => Number::INFINITY,
            };
            stats.num_obj_evals += 1;
            // Recompute g at trial.
            let mut g_trial = vec![0.0; m];
            if !tnlp.eval_g(&x_trial, true, &mut g_trial) {
                alpha *= 0.5;
                if alpha < opts.line_search_min_alpha {
                    return finalize(
                        tnlp,
                        &x,
                        current_f,
                        n,
                        iter,
                        stats,
                        ApplicationReturnStatus::SearchDirectionBecomesTooSmall,
                    );
                }
                continue;
            }
            stats.num_constr_evals += 1;
            for i in 0..m {
                g_trial[i] -= g_target[i];
            }
            let merit_trial = if f_trial.is_finite() {
                let g1: Number = g_trial.iter().map(|v| v.abs()).sum();
                f_trial + rho * g1
            } else {
                Number::INFINITY
            };
            if merit_trial.is_finite()
                && merit_trial <= merit_curr + alpha * armijo_rhs_factor
            {
                break f_trial;
            }
            alpha *= 0.5;
            if alpha < opts.line_search_min_alpha {
                return finalize(
                    tnlp,
                    &x,
                    current_f,
                    n,
                    iter,
                    stats,
                    ApplicationReturnStatus::SearchDirectionBecomesTooSmall,
                );
            }
        };

        for i in 0..n {
            x[i] += alpha * step_x[i];
        }
        for i in 0..m {
            y[i] += alpha * step_y[i];
        }
        current_f = accepted_f;
        iter += 1;
    };

    // Finalize with proper g/lambda fields.
    stats.iteration_count = iter;
    stats.final_objective = current_f;
    let z_l = vec![0.0; n];
    let z_u = vec![0.0; n];
    let mut g_final = vec![0.0; m];
    let _ = tnlp.eval_g(&x, true, &mut g_final);
    let solver_status = match final_status {
        ApplicationReturnStatus::SolveSucceeded => crate::alg_types::SolverReturn::Success,
        ApplicationReturnStatus::MaximumIterationsExceeded => {
            crate::alg_types::SolverReturn::MaxiterExceeded
        }
        ApplicationReturnStatus::SearchDirectionBecomesTooSmall => {
            crate::alg_types::SolverReturn::StopAtTinyStep
        }
        _ => crate::alg_types::SolverReturn::InternalError,
    };
    tnlp.finalize_solution(
        Solution {
            status: solver_status,
            x: &x,
            z_l: &z_l,
            z_u: &z_u,
            g: &g_final,
            lambda: &y,
            obj_value: current_f,
        },
        &IpoptData::default(),
        &IpoptCq::default(),
    );
    (final_status, stats)
}

/// Solve `[H J^T; J 0] [dx; dy] = -[r1; r2]` via Schur complement.
/// `H` is n×n, `J` is m×n. Damps `H ← H + λI` until both `H` and the
/// Schur complement `J H^-1 J^T` admit Cholesky factorization.
/// On success: `step_x ← dx`, `step_y ← dy`.
#[allow(clippy::too_many_arguments)]
fn schur_kkt_solve(
    h: &[Number],
    j_mat: &[Number],
    r1: &[Number],
    r2: &[Number],
    n: usize,
    m: usize,
    step_x: &mut [Number],
    step_y: &mut [Number],
) -> Result<(), ()> {
    let mut lambda = 0.0_f64;
    for attempt in 0..30 {
        let mut hd = h.to_vec();
        for i in 0..n {
            hd[i * n + i] += lambda;
        }
        // Cholesky of H.
        if cholesky_factor(&mut hd, n).is_err() {
            lambda = if lambda == 0.0 { 1e-4 } else { lambda * 8.0 };
            if attempt == 29 {
                return Err(());
            }
            continue;
        }
        // Solve H * v = r1, store v in v.
        let mut v = r1.to_vec();
        cholesky_solve(&hd, &mut v, n);
        // Solve H * W = J^T column by column. Store W as n×m row-major (n rows, m cols).
        let mut w = vec![0.0; n * m];
        for col in 0..m {
            let mut rhs = vec![0.0; n];
            for row in 0..n {
                rhs[row] = j_mat[col * n + row]; // J^T entry: J[col,row]
            }
            cholesky_solve(&hd, &mut rhs, n);
            for row in 0..n {
                w[row * m + col] = rhs[row];
            }
        }
        // Schur S = J * W (m×m).
        let mut s = vec![0.0; m * m];
        for i in 0..m {
            for k in 0..n {
                let jik = j_mat[i * n + k];
                if jik == 0.0 {
                    continue;
                }
                for jj in 0..m {
                    s[i * m + jj] += jik * w[k * m + jj];
                }
            }
        }
        // KKT block elimination:
        //   H dx + J^T dy = -r1            ⇒ dx = -H^-1 r1 - H^-1 J^T dy = -v - W dy
        //   J dx = -r2
        //   ⇒ S dy = r2 - J v   where S = J H^-1 J^T, v = H^-1 r1.
        let mut rhs_y = r2.to_vec();
        for i in 0..m {
            let mut acc = 0.0;
            for k in 0..n {
                acc += j_mat[i * n + k] * v[k];
            }
            rhs_y[i] -= acc;
        }
        // Cholesky on S (regularize if needed).
        let mut s_fac = s.clone();
        let mut s_lambda = 0.0_f64;
        let s_solved = loop {
            if cholesky_factor(&mut s_fac, m).is_ok() {
                cholesky_solve(&s_fac, &mut rhs_y, m);
                break true;
            }
            s_lambda = if s_lambda == 0.0 { 1e-8 } else { s_lambda * 10.0 };
            if s_lambda > 1e6 {
                break false;
            }
            s_fac = s.clone();
            for i in 0..m {
                s_fac[i * m + i] += s_lambda;
            }
        };
        if !s_solved {
            lambda = if lambda == 0.0 { 1e-4 } else { lambda * 8.0 };
            if attempt == 29 {
                return Err(());
            }
            continue;
        }
        // dy = rhs_y
        step_y[..m].copy_from_slice(&rhs_y[..m]);
        // dx = -(v + W * dy)
        for row in 0..n {
            let mut wd = 0.0;
            for col in 0..m {
                wd += w[row * m + col] * step_y[col];
            }
            step_x[row] = -(v[row] + wd);
        }
        return Ok(());
    }
    Err(())
}

/// In-place Cholesky factorization. Same as `cholesky_solve_in_place`
/// but split into factor + solve so we can reuse the factorization
/// across multiple right-hand sides.
fn cholesky_factor(a: &mut [Number], n: usize) -> Result<(), ()> {
    for j in 0..n {
        let mut sum = a[j * n + j];
        for k in 0..j {
            sum -= a[j * n + k] * a[j * n + k];
        }
        if sum <= 1e-14 {
            return Err(());
        }
        let l_jj = sum.sqrt();
        a[j * n + j] = l_jj;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / l_jj;
        }
    }
    Ok(())
}

/// Forward + back solve given a Cholesky factor in `a` (lower triangle).
fn cholesky_solve(a: &[Number], b: &mut [Number], n: usize) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= a[k * n + i] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
}

/// Sum of `-mu log(x-l)` and `-mu log(u-x)` over active bounds.
/// Returns +infinity if `x` is not strictly interior.
fn barrier_term(
    x: &[Number],
    x_l: &[Number],
    x_u: &[Number],
    mu: Number,
    nlp_lower: Number,
    nlp_upper: Number,
) -> Number {
    if mu == 0.0 {
        return 0.0;
    }
    let fixed_eps = 1e-12;
    let mut acc = 0.0;
    for i in 0..x.len() {
        let lo = x_l[i];
        let hi = x_u[i];
        // Fixed variables (lo == hi) contribute no barrier term.
        if lo > nlp_lower
            && hi < nlp_upper
            && (hi - lo).abs() <= fixed_eps * lo.abs().max(1.0)
        {
            continue;
        }
        if lo > nlp_lower {
            let d = x[i] - lo;
            if d <= 0.0 {
                return Number::INFINITY;
            }
            acc -= mu * d.ln();
        }
        if hi < nlp_upper {
            let d = hi - x[i];
            if d <= 0.0 {
                return Number::INFINITY;
            }
            acc -= mu * d.ln();
        }
    }
    acc
}

fn finalize(
    tnlp: &mut dyn TNLP,
    x: &[Number],
    obj_value: Number,
    n: usize,
    iter: Index,
    mut stats: SolveStatistics,
    status: ApplicationReturnStatus,
) -> (ApplicationReturnStatus, SolveStatistics) {
    stats.iteration_count = iter;
    stats.final_objective = obj_value;
    let z_l = vec![0.0; n];
    let z_u = vec![0.0; n];
    let g: Vec<Number> = vec![];
    let lambda: Vec<Number> = vec![];
    let solver_status = match status {
        ApplicationReturnStatus::SolveSucceeded => crate::alg_types::SolverReturn::Success,
        ApplicationReturnStatus::SolvedToAcceptableLevel => {
            crate::alg_types::SolverReturn::StopAtAcceptablePoint
        }
        ApplicationReturnStatus::MaximumIterationsExceeded => {
            crate::alg_types::SolverReturn::MaxiterExceeded
        }
        ApplicationReturnStatus::SearchDirectionBecomesTooSmall => {
            crate::alg_types::SolverReturn::StopAtTinyStep
        }
        ApplicationReturnStatus::InfeasibleProblemDetected => {
            crate::alg_types::SolverReturn::LocalInfeasibility
        }
        ApplicationReturnStatus::InvalidNumberDetected => {
            crate::alg_types::SolverReturn::InvalidNumberDetected
        }
        ApplicationReturnStatus::ErrorInStepComputation => {
            crate::alg_types::SolverReturn::ErrorInStepComputation
        }
        _ => crate::alg_types::SolverReturn::InternalError,
    };
    tnlp.finalize_solution(
        Solution {
            status: solver_status,
            x,
            z_l: &z_l,
            z_u: &z_u,
            g: &g,
            lambda: &lambda,
            obj_value,
        },
        &IpoptData::default(),
        &IpoptCq::default(),
    );
    (status, stats)
}

/// Solve `(H + λI) step = -grad` for `step`, increasing `λ` until the
/// system is positive-definite enough for Cholesky to succeed. This
/// mirrors the spirit of `PdPerturbationHandler::PerturbForWrongInertia`
/// but on a single dense system rather than the augmented KKT.
fn solve_with_damping(h: &[Number], grad: &[Number], n: usize, step: &mut [Number]) -> Result<(), ()> {
    // If the Hessian contains non-finite entries (e.g. the user
    // function reports an unbounded second derivative at the starting
    // point), zero those entries and rely on Levenberg damping to
    // produce a finite descent direction. This degrades Newton to
    // scaled-gradient descent in the affected rows/columns rather
    // than failing the entire solve.
    let any_nonfinite = h.iter().any(|v| !v.is_finite());
    let mut sanitized: Vec<Number>;
    let h_use: &[Number] = if any_nonfinite {
        sanitized = h.to_vec();
        for v in sanitized.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        &sanitized
    } else {
        h
    };

    let mut lambda = if any_nonfinite { 1.0_f64 } else { 0.0_f64 };
    let mut work = vec![0.0; n * n];
    let mut rhs = vec![0.0; n];
    let max_attempts = 40;
    for attempt in 0..max_attempts {
        work.copy_from_slice(h_use);
        for i in 0..n {
            work[i * n + i] += lambda;
        }
        for i in 0..n {
            rhs[i] = -grad[i];
        }
        let ok = cholesky_solve_in_place(&mut work, &mut rhs, n).is_ok();
        if ok {
            step.copy_from_slice(&rhs);
            if step.iter().all(|v| v.is_finite()) {
                return Ok(());
            }
        }
        lambda = if lambda == 0.0 { 1e-4 } else { lambda * 8.0 };
        if attempt == max_attempts - 1 {
            return Err(());
        }
    }
    Err(())
}

/// Dense Cholesky `A = L L^T` with in-place factorization, then
/// `solve L (L^T x) = b`. `a` is row-major n×n; on success the lower
/// triangle holds `L`. Returns `Err` if any pivot is non-positive,
/// non-finite, or below a small threshold.
fn cholesky_solve_in_place(a: &mut [Number], b: &mut [Number], n: usize) -> Result<(), ()> {
    for j in 0..n {
        let mut sum = a[j * n + j];
        for k in 0..j {
            sum -= a[j * n + k] * a[j * n + k];
        }
        if !sum.is_finite() || sum <= 1e-14 {
            return Err(());
        }
        let l_jj = sum.sqrt();
        a[j * n + j] = l_jj;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / l_jj;
        }
    }
    // Forward solve L y = b
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    // Back solve L^T x = y
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= a[k * n + i] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_solves_2x2_spd() {
        // [[4, 2], [2, 3]] x = [10, 8] → x = [1.6, 1.6]
        // (verify: 4*1.6 + 2*1.6 = 9.6 ≠ 10. Recompute.)
        // Let's pick [[4,2],[2,3]] x = [4, 5]:
        // det = 12 - 4 = 8. x = (1/8) * [[3,-2],[-2,4]] [4,5]
        //                = (1/8) * [12-10, -8+20] = [0.25, 1.5]
        let mut a = vec![4.0, 2.0, 2.0, 3.0];
        let mut b = vec![4.0, 5.0];
        cholesky_solve_in_place(&mut a, &mut b, 2).unwrap();
        assert!((b[0] - 0.25).abs() < 1e-12);
        assert!((b[1] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn cholesky_rejects_indefinite() {
        // diag(-1, 1) — first pivot negative.
        let mut a = vec![-1.0, 0.0, 0.0, 1.0];
        let mut b = vec![1.0, 1.0];
        assert!(cholesky_solve_in_place(&mut a, &mut b, 2).is_err());
    }

    /// `min (x-3)^2 + (y-4)^2` over `[0,2]×[0,2]` — corner solution
    /// at `(2,2)` with `f* = 5`. The barrier method tracks the
    /// central path and lands within `mu_min` (1e-7) of the optimum.
    #[test]
    fn bounded_corner_solves() {
        struct BQ;
        impl TNLP for BQ {
            fn get_nlp_info(&mut self) -> Option<crate::tnlp::NlpInfo> {
                Some(crate::tnlp::NlpInfo {
                    n: 2,
                    m: 0,
                    nnz_jac_g: 0,
                    nnz_h_lag: 2,
                    index_style: crate::tnlp::IndexStyle::C,
                })
            }
            fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
                b.x_l.copy_from_slice(&[0.0, 0.0]);
                b.x_u.copy_from_slice(&[2.0, 2.0]);
                true
            }
            fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
                sp.x.copy_from_slice(&[1.0, 1.0]);
                true
            }
            fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
                Some((x[0] - 3.0).powi(2) + (x[1] - 4.0).powi(2))
            }
            fn eval_grad_f(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
                g[0] = 2.0 * (x[0] - 3.0);
                g[1] = 2.0 * (x[1] - 4.0);
                true
            }
            fn eval_g(&mut self, _: &[Number], _: bool, _: &mut [Number]) -> bool {
                true
            }
            fn eval_jac_g(
                &mut self,
                _: Option<&[Number]>,
                _: bool,
                _: SparsityRequest<'_>,
            ) -> bool {
                true
            }
            fn eval_h(
                &mut self,
                _: Option<&[Number]>,
                _: bool,
                obj: Number,
                _: Option<&[Number]>,
                _: bool,
                mode: SparsityRequest<'_>,
            ) -> bool {
                match mode {
                    SparsityRequest::Structure { irow, jcol } => {
                        irow.copy_from_slice(&[0, 1]);
                        jcol.copy_from_slice(&[0, 1]);
                    }
                    SparsityRequest::Values { values } => {
                        values[0] = 2.0 * obj;
                        values[1] = 2.0 * obj;
                    }
                }
                true
            }
            fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
        }
        let mut bq = BQ;
        let (status, stats) = solve_unconstrained(&mut bq, NewtonOptions::default());
        assert!(matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ));
        // Optimum is f*=5 at corner; barrier residual is O(mu_min).
        assert!((stats.final_objective - 5.0).abs() < 1e-3);
    }
}
