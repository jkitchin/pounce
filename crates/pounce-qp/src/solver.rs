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
use crate::options::{AntiCyclingChoice, QpOptions};
use crate::problem::{QpProblem, QpSolution, QpStats, QpWarmStart};
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

    /// §4.5 inertia-controlled factorization. Tries the factor
    /// without shift first; on `WrongInertia` or `Singular`, shifts
    /// the H-block diagonal by progressively larger δ and re-tries.
    /// Returns the final δ used (0.0 when no shift was needed) for
    /// logging / diagnostics.
    ///
    /// `expected_neg` is required (no bypass) so the inertia signal
    /// is always checked. The `HessianInertia::Indefinite` hint
    /// merely tells the caller "shifts may be needed"; the
    /// algorithm decides what to do based on the factor's report.
    fn factorize_with_inertia_control(
        &mut self,
        mut kkt: KktTriplet,
        rhs: &mut [Number],
        expected_neg: i32,
        n_h_rows: usize,
        opts: &QpOptions,
    ) -> Result<Number, QpError> {
        // First attempt: no shift.
        let rhs_snapshot = rhs.to_vec();
        let mut rhs_local = rhs_snapshot.clone();
        match self
            .linsol
            .factorize_and_solve(&kkt, &mut rhs_local, Some(expected_neg))
        {
            Ok(()) => {
                rhs.copy_from_slice(&rhs_local);
                return Ok(0.0);
            }
            Err(QpError::LinearSolverFailure(ref msg))
                if msg.contains("inertia") || msg.contains("singular") => {}
            Err(e) => return Err(e),
        }

        let mut current = 0.0;
        let mut next = opts.inertia_shift_initial;
        for _ in 0..opts.inertia_max_shifts {
            kkt.add_h_diagonal_shift(n_h_rows, next - current);
            current = next;
            let mut rhs_local = rhs_snapshot.clone();
            match self
                .linsol
                .factorize_and_solve(&kkt, &mut rhs_local, Some(expected_neg))
            {
                Ok(()) => {
                    rhs.copy_from_slice(&rhs_local);
                    return Ok(current);
                }
                Err(QpError::LinearSolverFailure(ref msg))
                    if msg.contains("inertia") || msg.contains("singular") =>
                {
                    next *= opts.inertia_shift_factor;
                }
                Err(e) => return Err(e),
            }
        }
        Err(QpError::LinearSolverFailure(format!(
            "inertia control exhausted {} shifts (final δ = {:.3e}); reduced Hessian \
             remains non-PD on null(A_W) — consider an `HessianInertia::Indefinite` \
             problem with no PD reduced direction, or relax `inertia_shift_factor`",
            opts.inertia_max_shifts, current
        )))
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
            // distinct columns) and PD reduced H. Inertia-control
            // retry handles indefinite reduced H via §4.5.
            self.factorize_with_inertia_control(kkt, &mut rhs, k as i32, qp.n, opts)?;
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
        self.factorize_with_inertia_control(kkt0, &mut rhs0, m as i32, qp.n, opts)?;
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

            self.factorize_with_inertia_control(kkt, &mut rhs, (m + k) as i32, qp.n, opts)?;
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
        // eigenvalues (Gould-Hribar-Nocedal 2001 §3.2). The
        // inertia-control retry handles indefinite reduced H via
        // §4.5.
        self.factorize_with_inertia_control(kkt, &mut rhs, qp.m as i32, qp.n, opts)?;

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
            // Try the cheap eq-relaxed cold start first; if it
            // produces an infeasible point, route through §4.3
            // l1-elastic mode instead.
            match self.cold_general_initial(qp, opts, &mut n_refactor)? {
                Some(p) => p,
                None => return self.solve_elastic(qp, opts),
            }
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

            self.factorize_with_inertia_control(kkt, &mut rhs, (k_c + k_b) as i32, qp.n, opts)?;
            n_refactor += 1;

            let p_inf = rhs[..n].iter().map(|pi| pi.abs()).fold(0.0, f64::max);

            if p_inf <= opts.opt_tol {
                // KKT-stationary for current W. Pick a wrong-sign
                // active row to drop.
                //
                // Tie-breaking rule (§4.4): `AntiCyclingChoice::Bland`
                // picks the lowest-indexed violation (Bland 1977 —
                // guarantees finite termination at the cost of slower
                // convergence); the default `Expand`/`None` picks
                // the largest-magnitude violation (Dantzig's
                // steepest-violation rule — faster but not cycle-
                // free under pathological degeneracy).
                //
                // EXPAND (Gill-Murray-Saunders-Wright 1989) is the
                // SOTA default per the design note; its full
                // primal-perturbation machinery is one of the
                // remaining Phase 5a items. Until it lands, the
                // `Expand` enum variant aliases to the steepest-
                // violation behavior, which is correct on every
                // non-cycling problem in the analytical ladder and
                // matches the qpOASES default.
                let use_bland = matches!(opts.anti_cycling, AntiCyclingChoice::Bland);

                let mut worst: Option<(DropTarget, Number)> = None;
                let consider =
                    |worst: &mut Option<(DropTarget, Number)>, target: DropTarget, viol: Number| {
                        if viol <= opts.opt_tol {
                            return;
                        }
                        let take = match *worst {
                            None => true,
                            Some((_, prev_viol)) => {
                                if use_bland {
                                    // Smallest index wins. Compare
                                    // problem-space indices regardless
                                    // of cons-vs-bound; cons indices
                                    // come first.
                                    let new_key = drop_target_key(target);
                                    let prev_key = drop_target_key(worst.unwrap().0);
                                    new_key < prev_key
                                } else {
                                    viol > prev_viol
                                }
                            }
                        };
                        if take {
                            *worst = Some((target, viol));
                        }
                    };

                for (j, &i) in active_cons.iter().enumerate() {
                    let lam = rhs[n + j];
                    let viol = match working.constraints[i] {
                        ConsStatus::AtLower => lam,
                        ConsStatus::AtUpper => -lam,
                        ConsStatus::Equality => 0.0,
                        ConsStatus::Inactive => unreachable!(),
                    };
                    consider(&mut worst, DropTarget::Cons(i), viol);
                }
                for (j, &i) in active_bounds.iter().enumerate() {
                    let lam = rhs[n + k_c + j];
                    let viol = match working.bounds[i] {
                        BoundStatus::AtLower => lam,
                        BoundStatus::AtUpper => -lam,
                        BoundStatus::Fixed => 0.0,
                        BoundStatus::Inactive => unreachable!(),
                    };
                    consider(&mut worst, DropTarget::Bound(i), viol);
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

            // Collect every blocking direction as
            //   (target, ratio, |a·p|).
            // The first pass below populates this list; the second
            // pass selects a winner per the active-cycling rule.
            // For Bland / steepest-violation the selection is the
            // strict-minimum ratio (with index- or step-magnitude
            // tie-break baked into the encounter order); for
            // EXPAND we use a Harris-style two-pass that picks the
            // largest-|a·p| direction among constraints within
            // tolerance of the minimum — this is the "guarantee
            // strict progress at degenerate vertices" half of GMSW
            // EXPAND (Hattingh 1989; Maros 1996 §4.2). The
            // primal-perturbation half (τ-growth + snap-reset) is
            // a follow-up commit.
            let mut candidates: Vec<(BlockerTarget, f64, f64)> = Vec::new();
            for i in 0..n {
                if working.bounds[i].is_active() {
                    continue;
                }
                if p[i] < -opts.feas_tol && qp.xl[i] > NLP_LOWER_BOUND_INF {
                    let r = (x[i] - qp.xl[i]) / -p[i];
                    candidates.push((BlockerTarget::Bound(i, BoundStatus::AtLower), r, p[i].abs()));
                }
                if p[i] > opts.feas_tol && qp.xu[i] < NLP_UPPER_BOUND_INF {
                    let r = (qp.xu[i] - x[i]) / p[i];
                    candidates.push((BlockerTarget::Bound(i, BoundStatus::AtUpper), r, p[i].abs()));
                }
            }
            for i in 0..m {
                if working.constraints[i].is_active() {
                    continue;
                }
                if qp.bl[i] == qp.bu[i] {
                    continue;
                }
                if ap[i] < -opts.feas_tol && qp.bl[i] > NLP_LOWER_BOUND_INF {
                    let r = (ax[i] - qp.bl[i]) / -ap[i];
                    candidates.push((BlockerTarget::Cons(i, ConsStatus::AtLower), r, ap[i].abs()));
                }
                if ap[i] > opts.feas_tol && qp.bu[i] < NLP_UPPER_BOUND_INF {
                    let r = (qp.bu[i] - ax[i]) / ap[i];
                    candidates.push((BlockerTarget::Cons(i, ConsStatus::AtUpper), r, ap[i].abs()));
                }
            }

            let (mut alpha, blocker) = select_blocker(&candidates, opts);

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

    /// Build a cold-start `(x, working)` for [`Self::solve_general`].
    /// Solves the equality-relaxed KKT (only rows with `bl == bu`
    /// participate). Returns `Ok(None)` when the resulting `x`
    /// violates an inequality row or variable bound — the caller
    /// (typically [`Self::solve_general`]) then dispatches to the
    /// §4.3 elastic mode.
    fn cold_general_initial(
        &mut self,
        qp: &QpProblem,
        opts: &QpOptions,
        n_refactor: &mut u32,
    ) -> Result<Option<(Vec<Number>, WorkingSet)>, QpError> {
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

        self.factorize_with_inertia_control(kkt, &mut rhs, m_eq as i32, qp.n, opts)?;
        *n_refactor += 1;

        let x: Vec<Number> = rhs[..n].to_vec();

        // Inequality-row feasibility check — any violation routes
        // the caller to elastic mode.
        let ax = a_times_x(qp.a, &x, m);
        for i in 0..m {
            if qp.bl[i] == qp.bu[i] {
                continue;
            }
            if qp.bl[i] > NLP_LOWER_BOUND_INF && ax[i] < qp.bl[i] - opts.feas_tol {
                return Ok(None);
            }
            if qp.bu[i] < NLP_UPPER_BOUND_INF && ax[i] > qp.bu[i] + opts.feas_tol {
                return Ok(None);
            }
        }
        for (i, &xi) in x.iter().enumerate() {
            if qp.xl[i] > NLP_LOWER_BOUND_INF && xi < qp.xl[i] - opts.feas_tol {
                return Ok(None);
            }
            if qp.xu[i] < NLP_UPPER_BOUND_INF && xi > qp.xu[i] + opts.feas_tol {
                return Ok(None);
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

        Ok(Some((x, working)))
    }

    /// l1-elastic mode — §4.3. Builds an
    /// [`ElasticReformulation`], seeds the augmented problem so
    /// the elastic slacks absorb any infeasibility at the initial
    /// `x`, and routes the augmented problem through
    /// [`Self::solve_general`] via the standard warm-start path.
    /// Unpacks the augmented solution into the original variable
    /// space and reports `QpStatus::Infeasible` when residual
    /// slacks exceed `feas_tol`.
    fn solve_elastic(&mut self, qp: &QpProblem, opts: &QpOptions) -> Result<QpSolution, QpError> {
        let started = Instant::now();
        let n = qp.n;
        let m = qp.m;

        let reform = crate::elastic::ElasticReformulation::build(qp, opts.elastic_gamma);
        let qp_aug = reform.as_qp();

        // Initial `x_orig` for the augmented seed: project 0 into
        // the original variable box. Slacks then absorb any
        // remaining infeasibility.
        let mut x_orig = vec![0.0; n];
        for (xi, (&l, &u)) in x_orig.iter_mut().zip(qp.xl.iter().zip(qp.xu.iter())) {
            if l > NLP_LOWER_BOUND_INF && *xi < l {
                *xi = l;
            }
            if u < NLP_UPPER_BOUND_INF && *xi > u {
                *xi = u;
            }
        }
        let (x_aug, working_aug) = reform.initial_seed(qp, &x_orig, opts.feas_tol);

        let ws = QpWarmStart {
            x: x_aug,
            lambda_g: vec![0.0; reform.m_aug],
            lambda_x: vec![0.0; reform.n_aug],
            working: working_aug,
        };

        // Recursive solve through the standard path.
        let sol_aug = self.solve_general(&qp_aug, Some(&ws), opts)?;

        // Pack the original-space solution.
        let x = sol_aug.x[..n].to_vec();
        let lambda_g = sol_aug.lambda_g.clone();
        let lambda_x = sol_aug.lambda_x[..n].to_vec();
        let mut working = WorkingSet::cold(n, m);
        working
            .constraints
            .copy_from_slice(&sol_aug.working.constraints);
        working.bounds.copy_from_slice(&sol_aug.working.bounds[..n]);

        let feasible = reform.is_feasible(&sol_aug.x, opts.feas_tol);
        let status = if feasible {
            QpStatus::Optimal
        } else {
            QpStatus::Infeasible
        };

        // Objective uses the ORIGINAL `H`, `g`, not the augmented
        // (penalty-inflated) values.
        let obj = quad_objective(qp, &x);

        Ok(QpSolution {
            x,
            lambda_g,
            lambda_x,
            working,
            obj,
            status,
            stats: QpStats {
                n_working_set_changes: sol_aug.stats.n_working_set_changes,
                n_refactor: sol_aug.stats.n_refactor,
                n_schur_updates: sol_aug.stats.n_schur_updates,
                used_phase1: true,
                time: started.elapsed(),
            },
        })
    }

    /// Schur-based variant of [`Self::solve_general`]. Opt-in via
    /// `QpOptions::use_schur_updates`. Replaces the per-iteration
    /// refactor with a cached factor of the fixed-dim K_max
    /// matrix and Sherman-Morrison-Woodbury rank-2 updates per
    /// working-set change. Resets the cached factor when the
    /// Schur block reaches `max_schur_updates_before_refactor`.
    ///
    /// Behavior is algorithmically identical to the refactor-per-
    /// iteration path: same drop / ratio-test logic, same exit
    /// conditions. The difference is the inner-loop cost: one
    /// cached resolve + small dense Schur solve per iteration,
    /// plus two cached resolves per working-set change.
    fn solve_general_schur(
        &mut self,
        qp: &QpProblem,
        ws: Option<&QpWarmStart>,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        let started = Instant::now();
        let n = qp.n;
        let m = qp.m;
        let m_total = m + n;
        let mut n_refactor: u32 = 0;
        let mut n_changes: u32 = 0;
        let mut n_schur_updates: u32 = 0;

        let (mut x, mut working) = if let Some(w) = ws {
            (w.x.clone(), w.working.clone())
        } else {
            match self.cold_general_initial(qp, opts, &mut n_refactor)? {
                Some(p) => p,
                None => return self.solve_elastic(qp, opts),
            }
        };

        for (i, &status) in working.bounds.iter().enumerate() {
            match status {
                BoundStatus::AtLower | BoundStatus::Fixed => x[i] = qp.xl[i],
                BoundStatus::AtUpper => x[i] = qp.xu[i],
                BoundStatus::Inactive => {}
            }
        }

        // Initialize Schur and factor the base K_max.
        let mut schur = crate::schur::SchurState::new(n, m);
        let active_count = active_slot_count(&working);
        schur.reset(&mut self.linsol, qp, &working, active_count as i32, opts)?;
        n_refactor += 1;

        for _iter in 0..opts.max_iter {
            let hx = h_times_x(qp.h, &x);
            let mut rhs = vec![0.0; n + m_total];
            for (rhs_i, (hx_i, &g_i)) in rhs[..n].iter_mut().zip(hx.iter().zip(qp.g.iter())) {
                *rhs_i = -(hx_i + g_i);
            }
            schur.solve(&mut self.linsol, &mut rhs)?;

            let p: Vec<Number> = rhs[..n].to_vec();
            let p_inf = p.iter().map(|pi| pi.abs()).fold(0.0, f64::max);

            if p_inf <= opts.opt_tol {
                let mut worst: Option<(DropTarget, Number)> = None;
                for slot in 0..m_total {
                    if !crate::schur::SchurState::slot_active(&working, slot) {
                        continue;
                    }
                    let lam = rhs[n + slot];
                    let (target, viol) = if slot < m {
                        let v = match working.constraints[slot] {
                            ConsStatus::AtLower => lam,
                            ConsStatus::AtUpper => -lam,
                            ConsStatus::Equality => 0.0,
                            ConsStatus::Inactive => unreachable!(),
                        };
                        (DropTarget::Cons(slot), v)
                    } else {
                        let var = slot - m;
                        let v = match working.bounds[var] {
                            BoundStatus::AtLower => lam,
                            BoundStatus::AtUpper => -lam,
                            BoundStatus::Fixed => 0.0,
                            BoundStatus::Inactive => unreachable!(),
                        };
                        (DropTarget::Bound(var), v)
                    };
                    if viol > worst.map(|(_, w)| w).unwrap_or(opts.opt_tol) {
                        worst = Some((target, viol));
                    }
                }

                if let Some((target, _)) = worst {
                    let slot = match target {
                        DropTarget::Cons(i) => {
                            working.constraints[i] = ConsStatus::Inactive;
                            i
                        }
                        DropTarget::Bound(i) => {
                            working.bounds[i] = BoundStatus::Inactive;
                            m + i
                        }
                    };
                    schur.apply_change(&mut self.linsol, qp, slot, false)?;
                    n_changes += 1;
                    n_schur_updates += 1;
                    if schur.needs_reset(opts) {
                        let ac = active_slot_count(&working);
                        schur.reset(&mut self.linsol, qp, &working, ac as i32, opts)?;
                        n_refactor += 1;
                    }
                    continue;
                }

                // Optimal.
                let mut lambda_g = vec![0.0; m];
                for s in 0..m {
                    if working.constraints[s].is_active() {
                        lambda_g[s] = rhs[n + s];
                    }
                }
                let mut lambda_x = vec![0.0; n];
                for j in 0..n {
                    if working.bounds[j].is_active() {
                        lambda_x[j] = -rhs[n + m + j];
                    }
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
                        n_schur_updates,
                        used_phase1: false,
                        time: started.elapsed(),
                    },
                });
            }

            // Ratio test — identical to solve_general but tracking
            // the slot index of the blocker for apply_change.
            let ap = a_times_x(qp.a, &p, m);
            let ax = a_times_x(qp.a, &x, m);

            let mut candidates: Vec<(BlockerTarget, Number, Number)> = Vec::new();
            for i in 0..n {
                if working.bounds[i].is_active() {
                    continue;
                }
                if p[i] < -opts.feas_tol && qp.xl[i] > NLP_LOWER_BOUND_INF {
                    let r = (x[i] - qp.xl[i]) / -p[i];
                    candidates.push((BlockerTarget::Bound(i, BoundStatus::AtLower), r, p[i].abs()));
                }
                if p[i] > opts.feas_tol && qp.xu[i] < NLP_UPPER_BOUND_INF {
                    let r = (qp.xu[i] - x[i]) / p[i];
                    candidates.push((BlockerTarget::Bound(i, BoundStatus::AtUpper), r, p[i].abs()));
                }
            }
            for i in 0..m {
                if working.constraints[i].is_active() {
                    continue;
                }
                if qp.bl[i] == qp.bu[i] {
                    continue;
                }
                if ap[i] < -opts.feas_tol && qp.bl[i] > NLP_LOWER_BOUND_INF {
                    let r = (ax[i] - qp.bl[i]) / -ap[i];
                    candidates.push((BlockerTarget::Cons(i, ConsStatus::AtLower), r, ap[i].abs()));
                }
                if ap[i] > opts.feas_tol && qp.bu[i] < NLP_UPPER_BOUND_INF {
                    let r = (qp.bu[i] - ax[i]) / ap[i];
                    candidates.push((BlockerTarget::Cons(i, ConsStatus::AtUpper), r, ap[i].abs()));
                }
            }
            let (mut alpha, blocker) = select_blocker(&candidates, opts);
            if alpha < 0.0 {
                alpha = 0.0;
            }
            for (xi, &pi) in x.iter_mut().zip(p.iter()) {
                *xi += alpha * pi;
            }
            if let Some(blk) = blocker {
                let slot = match blk {
                    BlockerTarget::Bound(i, status) => {
                        match status {
                            BoundStatus::AtLower => x[i] = qp.xl[i],
                            BoundStatus::AtUpper => x[i] = qp.xu[i],
                            _ => unreachable!(),
                        }
                        working.bounds[i] = status;
                        m + i
                    }
                    BlockerTarget::Cons(i, status) => {
                        working.constraints[i] = status;
                        i
                    }
                };
                schur.apply_change(&mut self.linsol, qp, slot, true)?;
                n_changes += 1;
                n_schur_updates += 1;
                if schur.needs_reset(opts) {
                    let ac = active_slot_count(&working);
                    schur.reset(&mut self.linsol, qp, &working, ac as i32, opts)?;
                    n_refactor += 1;
                }
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
                n_schur_updates,
                used_phase1: false,
                time: started.elapsed(),
            },
        })
    }
}

fn active_slot_count(working: &WorkingSet) -> usize {
    working.constraints.iter().filter(|s| s.is_active()).count()
        + working.bounds.iter().filter(|s| s.is_active()).count()
}

#[derive(Clone, Copy)]
enum DropTarget {
    Cons(usize),
    Bound(usize),
}

/// Total ordering on `DropTarget` used by Bland's tie-break:
/// constraint indices `0..m` come before bound indices `0..n`.
/// Stable across iterations because the index spaces don't change
/// over the lifetime of a single `solve_general` call.
fn drop_target_key(t: DropTarget) -> (u8, usize) {
    match t {
        DropTarget::Cons(i) => (0, i),
        DropTarget::Bound(i) => (1, i),
    }
}

#[derive(Clone, Copy)]
enum BlockerTarget {
    Cons(usize, ConsStatus),
    Bound(usize, BoundStatus),
}

fn blocker_index_key(b: BlockerTarget) -> (u8, usize) {
    match b {
        BlockerTarget::Cons(i, _) => (0, i),
        BlockerTarget::Bound(i, _) => (1, i),
    }
}

/// Pick a blocking direction from the ratio-test candidate list.
///
/// `AntiCyclingChoice::None` and `AntiCyclingChoice::Bland` both
/// take the strict-minimum ratio. The two differ on the drop
/// path, not on the ratio test — at this point in the loop the
/// difference does not manifest, so both behave identically here.
///
/// `AntiCyclingChoice::Expand` runs the Harris-style two-pass: it
/// finds `α_min`, then among directions whose ratio is within
/// `feas_tol · (1 + |α_min|)` of `α_min`, picks the one with the
/// largest `|a·p|` — the most "expressive" direction. This is
/// the cycling-prevention core of GMSW EXPAND (Gill-Murray-
/// Saunders-Wright 1989); the τ-growth and snap-to-bound
/// machinery is a follow-up commit.
///
/// Returns `(α, blocker)` with `α = 1.0` and `blocker = None`
/// when no direction blocks at less than the full step.
fn select_blocker(
    candidates: &[(BlockerTarget, f64, f64)],
    opts: &QpOptions,
) -> (f64, Option<BlockerTarget>) {
    if candidates.is_empty() {
        return (1.0, None);
    }
    // Pass 1: minimum ratio.
    let mut alpha_min = 1.0_f64;
    for &(_, r, _) in candidates {
        if r < alpha_min {
            alpha_min = r;
        }
    }
    if alpha_min >= 1.0 {
        return (1.0, None);
    }

    match opts.anti_cycling {
        AntiCyclingChoice::None | AntiCyclingChoice::Bland => {
            // Strict-min: pick the first candidate achieving
            // `alpha_min` (encounter order ⇒ lowest index for ties).
            let mut best: Option<(BlockerTarget, f64)> = None;
            for &(target, r, _) in candidates {
                if r > alpha_min + 0.0 {
                    continue;
                }
                if best.is_none() {
                    best = Some((target, r));
                }
            }
            let (target, r) = best.expect("non-empty candidates above");
            (r, Some(target))
        }
        AntiCyclingChoice::Expand => {
            // Harris two-pass: among directions within
            // `tol · (1 + |α_min|)` of `α_min`, pick the largest
            // `|a·p|`. Tie-break by lowest index for reproducibility.
            let tol = opts.feas_tol * (1.0 + alpha_min.abs());
            let mut best: Option<(BlockerTarget, f64, f64)> = None;
            for &(target, r, ap_mag) in candidates {
                if r > alpha_min + tol {
                    continue;
                }
                let take = match best {
                    None => true,
                    Some((prev_target, _, prev_ap)) => {
                        if ap_mag > prev_ap {
                            true
                        } else if ap_mag == prev_ap {
                            blocker_index_key(target) < blocker_index_key(prev_target)
                        } else {
                            false
                        }
                    }
                };
                if take {
                    best = Some((target, r, ap_mag));
                }
            }
            let (target, r, _) = best.expect("non-empty candidates above");
            (r, Some(target))
        }
    }
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
            if opts.use_schur_updates {
                return self.solve_general_schur(qp, ws, opts);
            }
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
