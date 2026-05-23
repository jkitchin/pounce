//! The [`QpSolver`] trait and its concrete implementation
//! [`ParametricActiveSetSolver`].
//!
//! Phase 5a commit 2 ships the cold-start equality-only path: KKT
//! assembly via [`crate::kkt`] + one factor-and-solve through a
//! caller-provided linear-solver backend. Working-set machinery,
//! Schur-complement updates, EXPAND anti-cycling, l1-elastic
//! phase-1, and the parametric homotopy land in subsequent commits.

use std::time::Instant;

use crate::error::{QpError, QpStatus};
use crate::factor::LinearSolver;
use crate::kkt::{
    a_times_x, assemble_active_set_kkt, assemble_box_with_active, assemble_equality_plus_bounds,
    h_times_x, is_all_equality_constraints, is_pure_box, is_pure_equality_no_bounds,
    rhs_equality_only, KktTriplet,
};
use crate::options::QpOptions;
use crate::problem::{HessianInertia, QpProblem, QpSolution, QpStats, QpWarmStart};
use crate::working_set::{BoundStatus, ConsStatus, WorkingSet};
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_common::Number;
use pounce_linsol::SparseSymLinearSolverInterface;

/// QP subproblem solver.
///
/// Two entry points: [`solve`](Self::solve) for a single QP with an
/// optional warm-start seed, and [`solve_parametric`](Self::solve_parametric)
/// for the SQP outer-loop case where the new QP is a perturbation of
/// the previous one and the parametric homotopy of §4.2 can reuse
/// the cached factorization across consecutive QPs without
/// rebuilding it.
pub trait QpSolver {
    /// Solve a single QP. `ws == None` ⇒ cold start (phase-1
    /// elastic mode infers the initial working set when the
    /// machinery lands).
    fn solve(
        &mut self,
        qp: &QpProblem,
        ws: Option<&QpWarmStart>,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError>;

    /// Parametric solve: trace the homotopy from `(qp_prev,
    /// sol_prev)` to `qp_new`. Falls back to
    /// [`solve`](Self::solve) when the parametric path detects a
    /// structural change that requires a fresh refactor.
    fn solve_parametric(
        &mut self,
        qp_prev: &QpProblem,
        sol_prev: &QpSolution,
        qp_new: &QpProblem,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError>;
}

/// The sparse parametric active-set QP solver (§4.2 of the design
/// note). Owns a single linear-solver backend; future Schur-
/// complement state lives here too.
pub struct ParametricActiveSetSolver {
    linsol: LinearSolver,
}

impl ParametricActiveSetSolver {
    pub fn new(backend: Box<dyn SparseSymLinearSolverInterface>) -> Self {
        Self {
            linsol: LinearSolver::new(backend),
        }
    }

    /// Primal active-set path for box-constrained QPs
    /// (no general constraints, finite or infinite variable
    /// bounds). Standard add/drop loop with refactor-per-change —
    /// the Schur-complement update path (§4.2) replaces the
    /// refactor in a later commit.
    ///
    /// Each iteration:
    ///   1. assemble `[H Eᵀ_W; E_W 0]` from the current active set;
    ///   2. solve for step `(p, λ_sat)` against RHS `[-(Hx+g); 0]`;
    ///   3. if `‖p‖ < opt_tol`, examine multiplier signs — drop
    ///      one wrong-sign active bound, else declare optimal;
    ///   4. otherwise ratio-test along `p` to the first blocking
    ///      bound, take that step, add the blocker to `W`.
    ///
    /// Sign convention for dropping (with our saddle Lagrangian
    /// `L = ½xᵀHx + gᵀx + λᵀ_sat(E_W x − β_W)` and IPOPT-style
    /// user-facing multipliers `lambda_x = z_l − z_u`):
    ///   * AtLower → `λ_sat ≤ 0` at optimum; drop if `λ_sat > tol`.
    ///   * AtUpper → `λ_sat ≥ 0` at optimum; drop if `λ_sat < -tol`.
    ///   * Fixed → never dropped.
    fn solve_box_constrained(
        &mut self,
        qp: &QpProblem,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        let started = Instant::now();
        let n = qp.n;

        // ---- 1. Initial primal x: project 0 into the box ----
        let mut x = vec![0.0; n];
        for (xi, (&l, &u)) in x.iter_mut().zip(qp.xl.iter().zip(qp.xu.iter())) {
            if l > NLP_LOWER_BOUND_INF && *xi < l {
                *xi = l;
            }
            if u < NLP_UPPER_BOUND_INF && *xi > u {
                *xi = u;
            }
        }

        // ---- 2. Initial working set ----
        let mut working = WorkingSet::cold(n, 0);
        for (i, (status, xi)) in working.bounds.iter_mut().zip(x.iter_mut()).enumerate() {
            let l = qp.xl[i];
            let u = qp.xu[i];
            let l_finite = l > NLP_LOWER_BOUND_INF;
            let u_finite = u < NLP_UPPER_BOUND_INF;
            if l_finite && u_finite && (l - u).abs() <= opts.feas_tol {
                *status = BoundStatus::Fixed;
                *xi = l;
            } else if l_finite && (*xi - l).abs() <= opts.feas_tol {
                *status = BoundStatus::AtLower;
                *xi = l;
            } else if u_finite && (*xi - u).abs() <= opts.feas_tol {
                *status = BoundStatus::AtUpper;
                *xi = u;
            }
        }

        let mut n_refactor: u32 = 0;
        let mut n_changes: u32 = 0;

        for _iter in 0..opts.max_iter {
            // Build active-bound index list (ascending = problem
            // order) and assemble the KKT.
            let active: Vec<usize> = (0..n).filter(|&i| working.bounds[i].is_active()).collect();
            let k = active.len();

            let kkt = assemble_box_with_active(qp, &active);

            // RHS = [ -(H x + g) ; 0_k ]
            let hx = h_times_x(qp.h, &x);
            let mut rhs = vec![0.0; n + k];
            for i in 0..n {
                rhs[i] = -(hx[i] + qp.g[i]);
            }

            // Inertia expectation: k negative eigenvalues for full-
            // rank E_W (always full rank since selection rows pick
            // distinct columns) and PD reduced H.
            let expected_neg = match qp.hessian_inertia {
                HessianInertia::Psd | HessianInertia::Unknown => Some(k as i32),
                HessianInertia::Indefinite => None,
            };
            self.linsol
                .factorize_and_solve(&kkt, &mut rhs, expected_neg)?;
            n_refactor += 1;

            // ---- 3. Check ‖p‖ ----
            let p_inf = rhs[..n].iter().map(|pi| pi.abs()).fold(0.0, f64::max);

            if p_inf <= opts.opt_tol {
                // At KKT-stationary point for current W. Examine
                // multiplier signs.
                let mut worst: Option<(usize, Number)> = None;
                for (j, &i) in active.iter().enumerate() {
                    let lam = rhs[n + j];
                    let viol = match working.bounds[i] {
                        BoundStatus::AtLower => lam,  // want ≤ 0
                        BoundStatus::AtUpper => -lam, // want ≥ 0
                        BoundStatus::Fixed => 0.0,    // never drop
                        BoundStatus::Inactive => unreachable!(),
                    };
                    if viol > worst.map(|(_, v)| v).unwrap_or(opts.opt_tol) {
                        worst = Some((i, viol));
                    }
                }

                if let Some((i_drop, _)) = worst {
                    working.bounds[i_drop] = BoundStatus::Inactive;
                    n_changes += 1;
                    continue;
                }

                // Optimal — pack user-facing multipliers.
                // lambda_x = z_l − z_u = −λ_sat for active i, 0 else.
                let mut lambda_x = vec![0.0; n];
                for (j, &i) in active.iter().enumerate() {
                    lambda_x[i] = -rhs[n + j];
                }

                return Ok(QpSolution {
                    obj: quad_objective(qp, &x),
                    x,
                    lambda_g: Vec::new(),
                    lambda_x,
                    working,
                    status: QpStatus::Optimal,
                    stats: QpStats {
                        n_working_set_changes: n_changes,
                        n_refactor,
                        n_schur_updates: 0,
                        used_phase1: false,
                        time: started.elapsed(),
                    },
                });
            }

            // ---- 4. Ratio test along p ----
            // First snapshot p so the in-place RHS solve doesn't
            // alias the step buffer later.
            let p: Vec<Number> = rhs[..n].to_vec();

            let mut alpha = 1.0_f64;
            let mut blocker: Option<(usize, BoundStatus)> = None;
            for i in 0..n {
                if working.bounds[i].is_active() {
                    continue;
                }
                if p[i] < -opts.feas_tol && qp.xl[i] > NLP_LOWER_BOUND_INF {
                    let r = (x[i] - qp.xl[i]) / -p[i];
                    if r < alpha {
                        alpha = r;
                        blocker = Some((i, BoundStatus::AtLower));
                    }
                }
                if p[i] > opts.feas_tol && qp.xu[i] < NLP_UPPER_BOUND_INF {
                    let r = (qp.xu[i] - x[i]) / p[i];
                    if r < alpha {
                        alpha = r;
                        blocker = Some((i, BoundStatus::AtUpper));
                    }
                }
            }

            if alpha < 0.0 {
                // Defensive: numerical noise shouldn't drive α
                // negative, but clip if it does.
                alpha = 0.0;
            }

            for i in 0..n {
                x[i] += alpha * p[i];
            }

            if let Some((i_block, status)) = blocker {
                // Snap to the exact bound to avoid drift.
                match status {
                    BoundStatus::AtLower => x[i_block] = qp.xl[i_block],
                    BoundStatus::AtUpper => x[i_block] = qp.xu[i_block],
                    _ => unreachable!(),
                }
                working.bounds[i_block] = status;
                n_changes += 1;
            }
        }

        // Hit max_iter.
        Ok(QpSolution {
            obj: quad_objective(qp, &x),
            x,
            lambda_g: Vec::new(),
            lambda_x: vec![0.0; n],
            working,
            status: QpStatus::MaxIter,
            stats: QpStats {
                n_working_set_changes: n_changes,
                n_refactor,
                n_schur_updates: 0,
                used_phase1: false,
                time: started.elapsed(),
            },
        })
    }

    /// Active-set path for QPs with general equality constraints
    /// *and* finite variable bounds. The cold start solves the
    /// equality-relaxed KKT (ignoring bounds) and routes to the
    /// active-set inner loop when that solution is bound-feasible.
    ///
    /// Bound-infeasible equality solutions are rejected with
    /// [`QpError::UnsupportedFeature`] — recovering from that case
    /// requires the §4.3 phase-1 elastic mode, which lands in the
    /// next Phase 5a commit. Once it does, the elastic mode will
    /// replace the rejection branch and produce a bound-and-
    /// equality-feasible starting point.
    ///
    /// In the inner loop the equality rows live permanently in the
    /// working set (`ConsStatus::Equality`) and are never dropped;
    /// only variable bounds add and drop. The KKT layout is
    /// `[H Aᵀ_eq Eᵀ_W; A_eq 0 0; E_W 0 0]` with expected inertia
    /// `(n, m + k, 0)` for full-rank rows and PD reduced H.
    fn solve_equality_plus_bounds(
        &mut self,
        qp: &QpProblem,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        let started = Instant::now();
        let n = qp.n;
        let m = qp.m;

        // ---- 1. Equality-relaxed initial point ----
        let kkt0 = KktTriplet::assemble_equality_only(qp);
        let mut rhs0 = rhs_equality_only(qp);
        let expected_neg0 = match qp.hessian_inertia {
            HessianInertia::Psd | HessianInertia::Unknown => Some(m as i32),
            HessianInertia::Indefinite => None,
        };
        self.linsol
            .factorize_and_solve(&kkt0, &mut rhs0, expected_neg0)?;
        let mut n_refactor: u32 = 1;
        let mut n_changes: u32 = 0;

        let mut x: Vec<Number> = rhs0[..n].to_vec();

        // ---- 2. Bound-feasibility check ----
        for (i, &xi) in x.iter().enumerate() {
            let l = qp.xl[i];
            let u = qp.xu[i];
            if l > NLP_LOWER_BOUND_INF && xi < l - opts.feas_tol {
                return Err(QpError::UnsupportedFeature(format!(
                    "equality-relaxed solution violates lower bound on x[{i}] \
                     (x = {xi:.6e}, xl = {l:.6e}); recovering requires the \
                     phase-1 elastic mode, which lands in the next Phase 5a commit"
                )));
            }
            if u < NLP_UPPER_BOUND_INF && xi > u + opts.feas_tol {
                return Err(QpError::UnsupportedFeature(format!(
                    "equality-relaxed solution violates upper bound on x[{i}] \
                     (x = {xi:.6e}, xu = {u:.6e}); recovering requires the \
                     phase-1 elastic mode, which lands in the next Phase 5a commit"
                )));
            }
        }

        // ---- 3. Initial working set ----
        let mut working = WorkingSet::cold(n, m);
        for c in working.constraints.iter_mut() {
            *c = ConsStatus::Equality;
        }
        for (i, (status, xi)) in working.bounds.iter_mut().zip(x.iter_mut()).enumerate() {
            let l = qp.xl[i];
            let u = qp.xu[i];
            let l_finite = l > NLP_LOWER_BOUND_INF;
            let u_finite = u < NLP_UPPER_BOUND_INF;
            if l_finite && u_finite && (l - u).abs() <= opts.feas_tol {
                *status = BoundStatus::Fixed;
                *xi = l;
            } else if l_finite && (*xi - l).abs() <= opts.feas_tol {
                *status = BoundStatus::AtLower;
                *xi = l;
            } else if u_finite && (*xi - u).abs() <= opts.feas_tol {
                *status = BoundStatus::AtUpper;
                *xi = u;
            }
        }

        // ---- 4. Active-set inner loop ----
        for _iter in 0..opts.max_iter {
            let active: Vec<usize> = (0..n).filter(|&i| working.bounds[i].is_active()).collect();
            let k = active.len();

            let kkt = assemble_equality_plus_bounds(qp, &active);

            let hx = h_times_x(qp.h, &x);
            let mut rhs = vec![0.0; n + m + k];
            for (rhs_i, (hx_i, &g_i)) in rhs[..n].iter_mut().zip(hx.iter().zip(qp.g.iter())) {
                *rhs_i = -(hx_i + g_i);
            }
            // rhs[n..n+m] and rhs[n+m..n+m+k] stay zero.

            let expected_neg = match qp.hessian_inertia {
                HessianInertia::Psd | HessianInertia::Unknown => Some((m + k) as i32),
                HessianInertia::Indefinite => None,
            };
            self.linsol
                .factorize_and_solve(&kkt, &mut rhs, expected_neg)?;
            n_refactor += 1;

            let p_inf = rhs[..n].iter().map(|pi| pi.abs()).fold(0.0, f64::max);

            if p_inf <= opts.opt_tol {
                // Check drop on bound multipliers in rhs[n+m..n+m+k].
                let mut worst: Option<(usize, Number)> = None;
                for (j, &i) in active.iter().enumerate() {
                    let lam = rhs[n + m + j];
                    let viol = match working.bounds[i] {
                        BoundStatus::AtLower => lam,
                        BoundStatus::AtUpper => -lam,
                        BoundStatus::Fixed => 0.0,
                        BoundStatus::Inactive => unreachable!(),
                    };
                    if viol > worst.map(|(_, v)| v).unwrap_or(opts.opt_tol) {
                        worst = Some((i, viol));
                    }
                }

                if let Some((i_drop, _)) = worst {
                    working.bounds[i_drop] = BoundStatus::Inactive;
                    n_changes += 1;
                    continue;
                }

                // Optimal — pack multipliers.
                let lambda_g: Vec<Number> = rhs[n..n + m].to_vec();
                let mut lambda_x = vec![0.0; n];
                for (j, &i) in active.iter().enumerate() {
                    lambda_x[i] = -rhs[n + m + j];
                }

                return Ok(QpSolution {
                    obj: quad_objective(qp, &x),
                    x,
                    lambda_g,
                    lambda_x,
                    working,
                    status: QpStatus::Optimal,
                    stats: QpStats {
                        n_working_set_changes: n_changes,
                        n_refactor,
                        n_schur_updates: 0,
                        used_phase1: false,
                        time: started.elapsed(),
                    },
                });
            }

            // Ratio test along p.
            let p: Vec<Number> = rhs[..n].to_vec();
            let mut alpha = 1.0_f64;
            let mut blocker: Option<(usize, BoundStatus)> = None;
            for i in 0..n {
                if working.bounds[i].is_active() {
                    continue;
                }
                if p[i] < -opts.feas_tol && qp.xl[i] > NLP_LOWER_BOUND_INF {
                    let r = (x[i] - qp.xl[i]) / -p[i];
                    if r < alpha {
                        alpha = r;
                        blocker = Some((i, BoundStatus::AtLower));
                    }
                }
                if p[i] > opts.feas_tol && qp.xu[i] < NLP_UPPER_BOUND_INF {
                    let r = (qp.xu[i] - x[i]) / p[i];
                    if r < alpha {
                        alpha = r;
                        blocker = Some((i, BoundStatus::AtUpper));
                    }
                }
            }
            if alpha < 0.0 {
                alpha = 0.0;
            }
            for (xi, &pi) in x.iter_mut().zip(p.iter()) {
                *xi += alpha * pi;
            }
            if let Some((i_block, status)) = blocker {
                match status {
                    BoundStatus::AtLower => x[i_block] = qp.xl[i_block],
                    BoundStatus::AtUpper => x[i_block] = qp.xu[i_block],
                    _ => unreachable!(),
                }
                working.bounds[i_block] = status;
                n_changes += 1;
            }
        }

        Ok(QpSolution {
            obj: quad_objective(qp, &x),
            x,
            lambda_g: vec![0.0; m],
            lambda_x: vec![0.0; n],
            working,
            status: QpStatus::MaxIter,
            stats: QpStats {
                n_working_set_changes: n_changes,
                n_refactor,
                n_schur_updates: 0,
                used_phase1: false,
                time: started.elapsed(),
            },
        })
    }

    /// Cold-start path for QPs that have only equality constraints
    /// and no variable bounds. Builds the saddle-point KKT and
    /// hands it to the linear solver in one shot.
    fn solve_equality_only(
        &mut self,
        qp: &QpProblem,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        let started = Instant::now();
        let kkt = KktTriplet::assemble_equality_only(qp);
        let mut rhs = rhs_equality_only(qp);

        // Inertia expectation for [H Aᵀ; A 0] with full-rank A and
        // reduced Hessian PD on null(A): exactly m negative
        // eigenvalues (Gould-Hribar-Nocedal 2001 §3.2). Skip the
        // check when the caller declared H indefinite — the
        // §4.5 inertia-control path is required, and Phase 5a
        // commit 2 doesn't ship it yet.
        let expected_neg = match qp.hessian_inertia {
            HessianInertia::Psd | HessianInertia::Unknown => Some(qp.m as i32),
            HessianInertia::Indefinite => None,
        };
        self.linsol
            .factorize_and_solve(&kkt, &mut rhs, expected_neg)?;

        // RHS now holds [x*; λ*].
        let mut x = vec![0.0; qp.n];
        x.copy_from_slice(&rhs[..qp.n]);
        let mut lambda_g = vec![0.0; qp.m];
        lambda_g.copy_from_slice(&rhs[qp.n..]);

        let obj = quad_objective(qp, &x);

        // All general constraints are equalities (precondition of
        // this entry point) — mark them as such in the working set.
        let mut working = WorkingSet::cold(qp.n, qp.m);
        for c in working.constraints.iter_mut() {
            *c = ConsStatus::Equality;
        }

        let _ = opts; // QpOptions reserved for the working-set path.

        Ok(QpSolution {
            x,
            lambda_g,
            lambda_x: vec![0.0; qp.n],
            working,
            obj,
            status: QpStatus::Optimal,
            stats: QpStats {
                n_working_set_changes: 0,
                n_refactor: 1,
                n_schur_updates: 0,
                used_phase1: false,
                time: started.elapsed(),
            },
        })
    }

    /// General-purpose active-set path: handles arbitrary mix of
    /// equality and inequality general constraints, plus variable
    /// bounds, plus optional warm-start. This is the path the
    /// dispatcher routes to whenever a warm start is supplied or
    /// when the problem has at least one one-sided / two-sided
    /// general inequality row.
    ///
    /// Cold-start initial point: solves the equality-relaxed KKT
    /// (only rows with `bl == bu` participate) and accepts the
    /// solution if it is feasible w.r.t. inequality rows and
    /// variable bounds. Bound- or inequality-infeasible cases are
    /// rejected with [`QpError::UnsupportedFeature`] pointing at
    /// the §4.3 elastic-mode commit.
    ///
    /// Warm-start initial point: trusts the caller's `(x, working)`
    /// pair. No correctness check; an infeasible warm start may
    /// diverge or hit max_iter. (Validation is deferred to a
    /// follow-up commit that adds an `OptimalityCheck` audit pass.)
    fn solve_general(
        &mut self,
        qp: &QpProblem,
        ws: Option<&QpWarmStart>,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        let started = Instant::now();
        let n = qp.n;
        let m = qp.m;
        let mut n_refactor: u32 = 0;
        let mut n_changes: u32 = 0;

        // ---- 1. Initial (x, working) — warm-start or cold solve ----
        let (mut x, mut working) = if let Some(w) = ws {
            (w.x.clone(), w.working.clone())
        } else {
            self.cold_general_initial(qp, opts, &mut n_refactor)?
        };

        // Snap primal coordinates of active bounds to their exact
        // bound values; protects against caller drift in warm-start
        // mode and against floating-point noise after the cold-init
        // KKT solve.
        for (i, &status) in working.bounds.iter().enumerate() {
            match status {
                BoundStatus::AtLower | BoundStatus::Fixed => x[i] = qp.xl[i],
                BoundStatus::AtUpper => x[i] = qp.xu[i],
                BoundStatus::Inactive => {}
            }
        }

        // ---- 2. Active-set inner loop ----
        for _iter in 0..opts.max_iter {
            let active_cons: Vec<usize> = (0..m)
                .filter(|&i| working.constraints[i].is_active())
                .collect();
            let active_bounds: Vec<usize> =
                (0..n).filter(|&i| working.bounds[i].is_active()).collect();
            let k_c = active_cons.len();
            let k_b = active_bounds.len();

            let kkt = assemble_active_set_kkt(qp, &active_cons, &active_bounds);

            let hx = h_times_x(qp.h, &x);
            let mut rhs = vec![0.0; n + k_c + k_b];
            for (rhs_i, (hx_i, &g_i)) in rhs[..n].iter_mut().zip(hx.iter().zip(qp.g.iter())) {
                *rhs_i = -(hx_i + g_i);
            }

            let expected_neg = match qp.hessian_inertia {
                HessianInertia::Psd | HessianInertia::Unknown => Some((k_c + k_b) as i32),
                HessianInertia::Indefinite => None,
            };
            self.linsol
                .factorize_and_solve(&kkt, &mut rhs, expected_neg)?;
            n_refactor += 1;

            let p_inf = rhs[..n].iter().map(|pi| pi.abs()).fold(0.0, f64::max);

            if p_inf <= opts.opt_tol {
                // KKT-stationary for current W. Check for wrong-sign
                // multipliers — drop the worst.
                let mut worst: Option<(DropTarget, Number)> = None;
                for (j, &i) in active_cons.iter().enumerate() {
                    let lam = rhs[n + j];
                    let viol = match working.constraints[i] {
                        ConsStatus::AtLower => lam,
                        ConsStatus::AtUpper => -lam,
                        ConsStatus::Equality => 0.0,
                        ConsStatus::Inactive => unreachable!(),
                    };
                    if viol > worst.map(|(_, v)| v).unwrap_or(opts.opt_tol) {
                        worst = Some((DropTarget::Cons(i), viol));
                    }
                }
                for (j, &i) in active_bounds.iter().enumerate() {
                    let lam = rhs[n + k_c + j];
                    let viol = match working.bounds[i] {
                        BoundStatus::AtLower => lam,
                        BoundStatus::AtUpper => -lam,
                        BoundStatus::Fixed => 0.0,
                        BoundStatus::Inactive => unreachable!(),
                    };
                    if viol > worst.map(|(_, v)| v).unwrap_or(opts.opt_tol) {
                        worst = Some((DropTarget::Bound(i), viol));
                    }
                }

                if let Some((target, _)) = worst {
                    match target {
                        DropTarget::Cons(i) => working.constraints[i] = ConsStatus::Inactive,
                        DropTarget::Bound(i) => working.bounds[i] = BoundStatus::Inactive,
                    }
                    n_changes += 1;
                    continue;
                }

                let mut lambda_g = vec![0.0; m];
                for (j, &i) in active_cons.iter().enumerate() {
                    lambda_g[i] = rhs[n + j];
                }
                let mut lambda_x = vec![0.0; n];
                for (j, &i) in active_bounds.iter().enumerate() {
                    lambda_x[i] = -rhs[n + k_c + j];
                }

                return Ok(QpSolution {
                    obj: quad_objective(qp, &x),
                    x,
                    lambda_g,
                    lambda_x,
                    working,
                    status: QpStatus::Optimal,
                    stats: QpStats {
                        n_working_set_changes: n_changes,
                        n_refactor,
                        n_schur_updates: 0,
                        used_phase1: false,
                        time: started.elapsed(),
                    },
                });
            }

            // Ratio test along p — scan inactive constraints AND
            // inactive bounds. For inactive constraint i, the rate
            // of change of `a_iᵀ x` along p is `a_iᵀ p`.
            let p: Vec<Number> = rhs[..n].to_vec();
            let ap = a_times_x(qp.a, &p, m);
            let ax = a_times_x(qp.a, &x, m);

            let mut alpha = 1.0_f64;
            let mut blocker: Option<BlockerTarget> = None;

            // Inactive variable bounds.
            for i in 0..n {
                if working.bounds[i].is_active() {
                    continue;
                }
                if p[i] < -opts.feas_tol && qp.xl[i] > NLP_LOWER_BOUND_INF {
                    let r = (x[i] - qp.xl[i]) / -p[i];
                    if r < alpha {
                        alpha = r;
                        blocker = Some(BlockerTarget::Bound(i, BoundStatus::AtLower));
                    }
                }
                if p[i] > opts.feas_tol && qp.xu[i] < NLP_UPPER_BOUND_INF {
                    let r = (qp.xu[i] - x[i]) / p[i];
                    if r < alpha {
                        alpha = r;
                        blocker = Some(BlockerTarget::Bound(i, BoundStatus::AtUpper));
                    }
                }
            }
            // Inactive general inequality constraints.
            for i in 0..m {
                if working.constraints[i].is_active() {
                    continue;
                }
                // An equality row (bl == bu) ought to be marked
                // ConsStatus::Equality already — defensively skip.
                if qp.bl[i] == qp.bu[i] {
                    continue;
                }
                if ap[i] < -opts.feas_tol && qp.bl[i] > NLP_LOWER_BOUND_INF {
                    let r = (ax[i] - qp.bl[i]) / -ap[i];
                    if r < alpha {
                        alpha = r;
                        blocker = Some(BlockerTarget::Cons(i, ConsStatus::AtLower));
                    }
                }
                if ap[i] > opts.feas_tol && qp.bu[i] < NLP_UPPER_BOUND_INF {
                    let r = (qp.bu[i] - ax[i]) / ap[i];
                    if r < alpha {
                        alpha = r;
                        blocker = Some(BlockerTarget::Cons(i, ConsStatus::AtUpper));
                    }
                }
            }
            if alpha < 0.0 {
                alpha = 0.0;
            }

            for (xi, &pi) in x.iter_mut().zip(p.iter()) {
                *xi += alpha * pi;
            }

            if let Some(blk) = blocker {
                match blk {
                    BlockerTarget::Bound(i, status) => {
                        match status {
                            BoundStatus::AtLower => x[i] = qp.xl[i],
                            BoundStatus::AtUpper => x[i] = qp.xu[i],
                            _ => unreachable!(),
                        }
                        working.bounds[i] = status;
                    }
                    BlockerTarget::Cons(i, status) => {
                        // No primal snap: `α` was chosen so that
                        // a_iᵀ (x + α p) is exactly at the boundary
                        // by construction.
                        working.constraints[i] = status;
                    }
                }
                n_changes += 1;
            }
        }

        Ok(QpSolution {
            obj: quad_objective(qp, &x),
            x,
            lambda_g: vec![0.0; m],
            lambda_x: vec![0.0; n],
            working,
            status: QpStatus::MaxIter,
            stats: QpStats {
                n_working_set_changes: n_changes,
                n_refactor,
                n_schur_updates: 0,
                used_phase1: false,
                time: started.elapsed(),
            },
        })
    }

    /// Build a cold-start `(x, working)` for [`Self::solve_general`]:
    /// solve the equality-relaxed KKT (only rows with `bl == bu`
    /// participate), then reject if the resulting `x` violates any
    /// inequality row or variable bound. Each rejection points at
    /// the §4.3 elastic-mode commit that will replace this branch.
    fn cold_general_initial(
        &mut self,
        qp: &QpProblem,
        opts: &QpOptions,
        n_refactor: &mut u32,
    ) -> Result<(Vec<Number>, WorkingSet), QpError> {
        let n = qp.n;
        let m = qp.m;

        let eq_rows: Vec<usize> = (0..m).filter(|&i| qp.bl[i] == qp.bu[i]).collect();
        let m_eq = eq_rows.len();

        let kkt = assemble_active_set_kkt(qp, &eq_rows, &[]);
        let mut rhs = vec![0.0; n + m_eq];
        for (rhs_i, &g_i) in rhs[..n].iter_mut().zip(qp.g.iter()) {
            *rhs_i = -g_i;
        }
        for (j, &row) in eq_rows.iter().enumerate() {
            rhs[n + j] = qp.bl[row];
        }

        let expected_neg = match qp.hessian_inertia {
            HessianInertia::Psd | HessianInertia::Unknown => Some(m_eq as i32),
            HessianInertia::Indefinite => None,
        };
        self.linsol
            .factorize_and_solve(&kkt, &mut rhs, expected_neg)?;
        *n_refactor += 1;

        let x: Vec<Number> = rhs[..n].to_vec();

        // Inequality-row feasibility check.
        let ax = a_times_x(qp.a, &x, m);
        for i in 0..m {
            if qp.bl[i] == qp.bu[i] {
                continue;
            }
            if qp.bl[i] > NLP_LOWER_BOUND_INF && ax[i] < qp.bl[i] - opts.feas_tol {
                return Err(QpError::UnsupportedFeature(format!(
                    "equality-relaxed solution violates lower bound on constraint row {i} \
                     (a·x = {:.6e}, bl = {:.6e}); requires phase-1 elastic mode \
                     (next Phase 5a commit)",
                    ax[i], qp.bl[i]
                )));
            }
            if qp.bu[i] < NLP_UPPER_BOUND_INF && ax[i] > qp.bu[i] + opts.feas_tol {
                return Err(QpError::UnsupportedFeature(format!(
                    "equality-relaxed solution violates upper bound on constraint row {i} \
                     (a·x = {:.6e}, bu = {:.6e}); requires phase-1 elastic mode \
                     (next Phase 5a commit)",
                    ax[i], qp.bu[i]
                )));
            }
        }
        // Variable-bound feasibility check.
        for (i, &xi) in x.iter().enumerate() {
            if qp.xl[i] > NLP_LOWER_BOUND_INF && xi < qp.xl[i] - opts.feas_tol {
                return Err(QpError::UnsupportedFeature(format!(
                    "equality-relaxed solution violates lower bound on x[{i}] \
                     (x = {xi:.6e}, xl = {:.6e}); requires phase-1 elastic mode",
                    qp.xl[i]
                )));
            }
            if qp.xu[i] < NLP_UPPER_BOUND_INF && xi > qp.xu[i] + opts.feas_tol {
                return Err(QpError::UnsupportedFeature(format!(
                    "equality-relaxed solution violates upper bound on x[{i}] \
                     (x = {xi:.6e}, xu = {:.6e}); requires phase-1 elastic mode",
                    qp.xu[i]
                )));
            }
        }

        // Build the working set: equalities always active; rows /
        // bounds exactly at their boundary value snapped to active.
        let mut working = WorkingSet::cold(n, m);
        for (i, c) in working.constraints.iter_mut().enumerate() {
            if qp.bl[i] == qp.bu[i] {
                *c = ConsStatus::Equality;
            } else if qp.bl[i] > NLP_LOWER_BOUND_INF && (ax[i] - qp.bl[i]).abs() <= opts.feas_tol {
                *c = ConsStatus::AtLower;
            } else if qp.bu[i] < NLP_UPPER_BOUND_INF && (ax[i] - qp.bu[i]).abs() <= opts.feas_tol {
                *c = ConsStatus::AtUpper;
            }
        }
        for (i, status) in working.bounds.iter_mut().enumerate() {
            let l = qp.xl[i];
            let u = qp.xu[i];
            let l_finite = l > NLP_LOWER_BOUND_INF;
            let u_finite = u < NLP_UPPER_BOUND_INF;
            if l_finite && u_finite && (l - u).abs() <= opts.feas_tol {
                *status = BoundStatus::Fixed;
            } else if l_finite && (x[i] - l).abs() <= opts.feas_tol {
                *status = BoundStatus::AtLower;
            } else if u_finite && (x[i] - u).abs() <= opts.feas_tol {
                *status = BoundStatus::AtUpper;
            }
        }

        Ok((x, working))
    }
}

#[derive(Clone, Copy)]
enum DropTarget {
    Cons(usize),
    Bound(usize),
}

#[derive(Clone, Copy)]
enum BlockerTarget {
    Cons(usize, ConsStatus),
    Bound(usize, BoundStatus),
}

impl QpSolver for ParametricActiveSetSolver {
    fn solve(
        &mut self,
        qp: &QpProblem,
        ws: Option<&QpWarmStart>,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        qp.validate()?;
        if let Some(w) = ws {
            w.working.validate_dims(qp.n, qp.m)?;
            if w.x.len() != qp.n {
                return Err(QpError::WarmStartDimensionMismatch(format!(
                    "ws.x.len() = {} but n = {}",
                    w.x.len(),
                    qp.n
                )));
            }
        }

        let has_general_inequality = !is_all_equality_constraints(qp);

        // Any of: caller provided a warm start, or the problem has at
        // least one one-sided / two-sided general inequality row.
        if ws.is_some() || has_general_inequality {
            return self.solve_general(qp, ws, opts);
        }

        // Cold-start fast paths for problems with no general
        // inequalities and no warm-start.
        if is_pure_equality_no_bounds(qp) {
            return self.solve_equality_only(qp, opts);
        }
        if is_pure_box(qp) {
            return self.solve_box_constrained(qp, opts);
        }
        self.solve_equality_plus_bounds(qp, opts)
    }

    fn solve_parametric(
        &mut self,
        _qp_prev: &QpProblem,
        _sol_prev: &QpSolution,
        qp_new: &QpProblem,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        // No parametric path yet — fall back to a fresh cold solve.
        self.solve(qp_new, None, opts)
    }
}

/// Evaluate `½ xᵀ H x + gᵀ x`, walking the symmetric Hessian once
/// and fanning each off-diagonal entry into both halves.
fn quad_objective(qp: &QpProblem, x: &[Number]) -> Number {
    let mut quad = 0.0;
    let irows = qp.h.irows();
    let jcols = qp.h.jcols();
    let vals = qp.h.values();
    for k in 0..irows.len() {
        let i = (irows[k] - 1) as usize;
        let j = (jcols[k] - 1) as usize;
        let v = vals[k];
        if i == j {
            quad += 0.5 * v * x[i] * x[i];
        } else {
            quad += v * x[i] * x[j]; // each off-diag pair contributes once
        }
    }
    let lin: Number = qp.g.iter().zip(x.iter()).map(|(&gi, &xi)| gi * xi).sum();
    quad + lin
}
