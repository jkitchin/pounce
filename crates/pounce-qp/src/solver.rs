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
    KktTriplet, a_times_x, assemble_active_set_kkt, assemble_box_with_active,
    assemble_equality_plus_bounds, h_times_x, is_all_equality_constraints, is_pure_box,
    is_pure_equality_no_bounds, rhs_equality_only,
};
use crate::options::{AntiCyclingChoice, QpOptions};
use crate::problem::{QpProblem, QpSolution, QpStats, QpWarmStart};
use crate::working_set::{BoundStatus, ConsStatus, WorkingSet};
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_common::{Index, Number};
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_linsol::status::ESymSolverStatus;

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

    /// Warm-start variant that takes ONLY the working set from a
    /// previous solve (not a primal `x`). Useful when the caller
    /// — e.g., the SQP outer loop — has a previous QP's working
    /// set but no compatible primal, because the new QP's
    /// constraint RHS has shifted (each SQP linearization
    /// translates `bl ≤ Ax ≤ bu` by `-c(x_k)`).
    ///
    /// Internally: build the KKT for the active rows of
    /// `working` and solve for a primal that exactly satisfies
    /// those rows. Pass that primal plus the supplied working
    /// set as a regular `QpWarmStart` to
    /// [`Self::solve`].
    ///
    /// Returns the same `QpSolution` shape as
    /// [`Self::solve`].
    fn solve_with_working_set(
        &mut self,
        qp: &QpProblem,
        working: &crate::working_set::WorkingSet,
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
            Err(ref e) if e.is_recoverable_factorization_failure() => {}
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
                Err(ref e) if e.is_recoverable_factorization_failure() => {
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

    /// Assemble and factor the pinned active-set KKT
    /// `[H Aᵀ_W Eᵀ_W; A_W 0 0; E_W 0 0]` with right-hand side
    /// `[-g; cons_targets; bound_targets]`, returning the primal `x`
    /// (the first `n` entries of the KKT solution). `cons_targets` is
    /// parallel to `active_cons`, `bound_targets` to `active_bounds`.
    ///
    /// Shared by the cold-start equality factor and the warm-start
    /// `solve_with_working_set` factor; multipliers are recomputed by
    /// the inner loop, so they are not returned here.
    fn factor_pinned_primal(
        &mut self,
        qp: &QpProblem,
        active_cons: &[usize],
        cons_targets: &[Number],
        active_bounds: &[usize],
        bound_targets: &[Number],
        opts: &QpOptions,
    ) -> Result<Vec<Number>, QpError> {
        let n = qp.n;
        let k_c = active_cons.len();
        let k_b = active_bounds.len();
        let kkt = assemble_active_set_kkt(qp, active_cons, active_bounds);
        let mut rhs = vec![0.0; n + k_c + k_b];
        for (rhs_i, &g_i) in rhs[..n].iter_mut().zip(qp.g.iter()) {
            *rhs_i = -g_i;
        }
        rhs[n..n + k_c].copy_from_slice(cons_targets);
        rhs[n + k_c..n + k_c + k_b].copy_from_slice(bound_targets);
        self.factorize_with_inertia_control(kkt, &mut rhs, (k_c + k_b) as i32, n, opts)?;
        Ok(rhs[..n].to_vec())
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
    /// Bound-infeasible equality solutions fall through to
    /// [`Self::solve_elastic`] — the same §4.3 phase-1 recovery
    /// `solve_general` uses via `cold_general_initial`.
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
        // The cheap equality-relaxed cold start may land outside
        // the box; fall through to the §4.3 elastic mode in that
        // case (same recovery `solve_general` uses; see
        // `cold_general_initial` → `solve_elastic` fall-through).
        for (i, &xi) in x.iter().enumerate() {
            let l = qp.xl[i];
            let u = qp.xu[i];
            if (l > NLP_LOWER_BOUND_INF && xi < l - opts.feas_tol)
                || (u < NLP_UPPER_BOUND_INF && xi > u + opts.feas_tol)
            {
                return self.solve_elastic(qp, opts);
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
        let delta = self.factorize_with_inertia_control(kkt, &mut rhs, qp.m as i32, qp.n, opts)?;

        // RHS now holds [x*; λ*].
        let mut x = vec![0.0; qp.n];
        x.copy_from_slice(&rhs[..qp.n]);
        let mut lambda_g = vec![0.0; qp.m];
        lambda_g.copy_from_slice(&rhs[qp.n..]);

        // H1 / N1: the inertia-control retry solved the *shifted* system
        // `(H+δI)` when `δ > 0`, which it must do whenever the reduced
        // Hessian is not PD on null(A). A `δ > 0` solve is consistent with
        // BOTH a bounded QP (the regularizer merely picks the min-norm
        // point along a flat, gradient-free direction) and an unbounded
        // one — so the shift alone proves nothing.
        //
        // The discriminator is a *certified recession ray*. A QP
        // `min ½xᵀHx + gᵀx  s.t. Ax = b` is unbounded below iff there is a
        // direction `d` with `Hd = 0` (zero curvature — for PSD H
        // equivalent to `dᵀHd = 0`), `Ad = 0` (stays feasible), and
        // `gᵀd < 0` (descent). The shifted solve manufactures exactly
        // this witness when one exists: any descent component of `-g`
        // lying in a zero-curvature, feasible direction is amplified by
        // `1/δ`, so the normalized iterate `d = x/‖x‖` converges to that
        // recession ray as `δ → 0`. We therefore certify the three
        // conditions directly on `d`.
        //
        // This replaces the earlier magnitude heuristic `δ·‖x‖∞ >
        // 1e-3·‖g‖∞`, which fired on any large `‖x‖` and could not
        // distinguish a large-but-finite minimizer in a *curved*
        // direction (e.g. `H = diag(1e-6, 0)`, `g = (-1, 0)`: the curved
        // x₁ runs out to its finite optimum ≈ 1e6) from a genuine blow-up
        // along a *flat* descent ray (N1 false positive). The curvature
        // clause `‖Hd‖∞ ≈ 0` (structural-zero floor, see
        // `ray_is_unbounded_descent`) rejects the former (there `‖Hd‖∞ ≈
        // ‖H‖`) and admits the latter.
        if delta > 0.0 {
            // Feasibility of the candidate ray `d = x/‖x‖`: the saddle
            // solve enforced `Ax = b` exactly, so `Ad = b/‖x‖`, which the
            // blow-up drives to ~0. Verify it explicitly (cheap guard;
            // trivially satisfied in the unconstrained `m = 0` case), then
            // delegate the curvature + descent clauses to the shared test.
            let x_norm = x.iter().map(|v| v * v).sum::<Number>().sqrt();
            let feasible_ray = if x_norm > 0.0 {
                let inv = 1.0 / x_norm;
                let mut ad = vec![0.0; qp.m];
                let mut a_scale: Number = 0.0;
                let irows = qp.a.irows();
                let jcols = qp.a.jcols();
                let vals = qp.a.values();
                for k in 0..irows.len() {
                    let i = (irows[k] - 1) as usize;
                    let j = (jcols[k] - 1) as usize;
                    a_scale = a_scale.max(vals[k].abs());
                    ad[i] += vals[k] * x[j] * inv;
                }
                let ad_inf = ad.iter().map(|v| v.abs()).fold(0.0, f64::max);
                ad_inf <= 1e-6 * (1.0 + a_scale)
            } else {
                false
            };

            if feasible_ray && ray_is_unbounded_descent(qp.h, qp.g, &x, &x) {
                return Ok(QpSolution {
                    x,
                    lambda_g,
                    lambda_x: vec![0.0; qp.n],
                    working: WorkingSet::cold(qp.n, qp.m),
                    obj: Number::NEG_INFINITY,
                    status: QpStatus::Unbounded,
                    stats: QpStats {
                        n_working_set_changes: 0,
                        n_refactor: 1,
                        n_schur_updates: 0,
                        used_phase1: false,
                        time: started.elapsed(),
                    },
                });
            }
        }

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
        // GMSW EXPAND τ — primal-perturbation tolerance.
        // Consumed by `select_blocker` only when
        // `opts.anti_cycling = Expand`; tracked unconditionally
        // so the snap-and-reset logic below is a no-op for the
        // other anti-cycling choices.
        let mut expand_tol = opts.expand_tol_initial;

        // Linear-independence anti-cycling tabu. When the rank guard
        // prunes a linearly-dependent row at a *stationary* (degenerate)
        // vertex, that row is satisfied at `x` and has true `a·p = 0`
        // along every feasible direction — yet numerical drift can give
        // it a tiny `|a·p| > feas_tol`, so the ratio test keeps re-adding
        // it, the factor goes rank-deficient again, and the engine cycles
        // (prune → re-add → prune …). Forbidding a pruned row from
        // re-entering until `x` actually moves breaks that cycle: while
        // the vertex is stationary the active set can only shrink, so the
        // degenerate phase terminates finitely; the tabu is cleared on the
        // first real step (`α > feas_tol`), after which the null space has
        // changed and a previously-dependent row may legitimately re-enter.
        let mut tabu_cons = vec![false; m];
        let mut tabu_bounds = vec![false; n];

        // Anti-stall fallback to Bland's rule (§4.4). The default
        // steepest-violation drop + Harris/largest-pivot add is fast
        // but NOT cycle-free: on a degenerate vertex (notably the
        // elastic phase-1 high-penalty vertices the GEN family and even
        // trivial LPs like `afiro` park at) it can churn the working set
        // without improving the objective until `max_iter`. Bland's rule
        // (lowest-index drop/add) is provably finite. We monitor the
        // objective and, once it fails to improve for `stall_limit`
        // consecutive iterations, latch into Bland selection for the
        // remainder of the solve — the textbook "Bland as anti-cycling
        // fallback after stalling" safeguard. The latch is sticky (never
        // reverts) so it cannot flip-flop, and it is a no-op on problems
        // that make steady progress.
        let mut force_bland = false;
        let mut best_obj = Number::INFINITY;
        let mut stall_iters: u32 = 0;
        // A problem making genuine progress rarely goes this many
        // consecutive iterations without any objective improvement; a
        // degenerate cycle does. Constant (not size-scaled) so it fires
        // well inside the default `max_iter` on large problems too.
        const STALL_LIMIT: u32 = 50;

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

            let delta = match self.factorize_with_inertia_control(
                kkt,
                &mut rhs,
                (k_c + k_b) as i32,
                qp.n,
                opts,
            ) {
                Ok(d) => d,
                Err(e) if e.is_recoverable_factorization_failure() => {
                    // The active set went rank-deficient: at a degenerate
                    // vertex more binding rows than variables can be linearly
                    // dependent, and numerical drift can let a dependent row
                    // (whose `a·p` should be 0) slip past the ratio test's
                    // `feas_tol`. No H-block shift can repair a rank-deficient
                    // *constraint* block, so the inertia loop just exhausted.
                    // Linear-independence guard: prune the active set to a
                    // maximal independent subset, deactivate the redundant
                    // rows (still satisfied at `x` — they are combinations of
                    // the kept ones), and retry on the next iteration.
                    let (kc, kb) = independent_active_subset(
                        &mut self.linsol,
                        qp,
                        &active_cons,
                        &active_bounds,
                    );
                    if kc.len() == active_cons.len() && kb.len() == active_bounds.len() {
                        return Err(e);
                    }
                    let mut keep_c = vec![false; m];
                    for &i in &kc {
                        keep_c[i] = true;
                    }
                    let mut keep_b = vec![false; n];
                    for &i in &kb {
                        keep_b[i] = true;
                    }
                    for &i in &active_cons {
                        if !keep_c[i] {
                            working.constraints[i] = ConsStatus::Inactive;
                            tabu_cons[i] = true;
                            n_changes += 1;
                        }
                    }
                    for &i in &active_bounds {
                        if !keep_b[i] {
                            working.bounds[i] = BoundStatus::Inactive;
                            tabu_bounds[i] = true;
                            n_changes += 1;
                        }
                    }
                    continue;
                }
                Err(e) => return Err(e),
            };
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
                let use_bland =
                    force_bland || matches!(opts.anti_cycling, AntiCyclingChoice::Bland);

                let mut worst: Option<(DropTarget, Number)> = None;
                let consider =
                    |worst: &mut Option<(DropTarget, Number)>, target: DropTarget, viol: Number| {
                        if viol <= opts.opt_tol {
                            return;
                        }
                        let take = match *worst {
                            None => true,
                            Some((prev_target, prev_viol)) => {
                                if use_bland {
                                    // Smallest index wins. Compare
                                    // problem-space indices regardless
                                    // of cons-vs-bound; cons indices
                                    // come first.
                                    let new_key = drop_target_key(target);
                                    let prev_key = drop_target_key(prev_target);
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

                if let Some((target, _viol)) = worst {
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
                // Rank-tabu (rate-aware): a bound pruned as linearly
                // dependent has true `a·p = 0`, so suppress it from the
                // ratio test only while its rate stays in the drift band
                // (`|p[i]| ≤ TABU_DRIFT_REL·‖p‖∞`). If the active set has
                // since evolved and this bound now carries an O(1) rate,
                // it is a GENUINE blocker — let it through so the step is
                // capped (otherwise ‖p‖ overshoots to ~1e14) and Bland's
                // lowest-index rule sees the true candidate set.
                if tabu_bounds[i] && p[i].abs() <= TABU_DRIFT_REL * p_inf {
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
                // Rank-tabu (rate-aware): see the bound loop above — a
                // pruned-dependent row has true `a·p = 0`, so suppress it
                // only while its rate stays in the drift band; a genuine
                // O(1) rate re-admits it so the step is capped and Bland
                // sees the true candidate set.
                if tabu_cons[i] && ap[i].abs() <= TABU_DRIFT_REL * p_inf {
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
            // (The rate-aware tabu skip is applied at the top of each loop
            // above: a pruned dependent row enters `candidates` only once
            // its rate along `p` leaves the linear-dependence drift band.)

            let (mut alpha, blocker) = select_blocker(&candidates, opts, expand_tol, force_bland);

            // F2(a): certified unboundedness on the active-set path. An
            // empty candidate list means NO inactive row or bound blocks
            // along `+p` (and `p` already lies in the active constraints'
            // null space), so `+p` is feasible for every step length — a
            // recession ray if it is also zero-curvature and descent.
            // We only reach for this when the inertia shift fired
            // (`delta > 0`, i.e. the reduced Hessian was singular on the
            // active null space); a PD reduced Hessian gives a finite
            // Newton step and never trips here. Without this the loop
            // takes unbounded full steps until `MaxIter` (δ discarded).
            if candidates.is_empty() && delta > 0.0 && ray_is_unbounded_descent(qp.h, qp.g, &x, &p)
            {
                return Ok(QpSolution {
                    obj: Number::NEG_INFINITY,
                    x,
                    lambda_g: vec![0.0; m],
                    lambda_x: vec![0.0; n],
                    working,
                    status: QpStatus::Unbounded,
                    stats: QpStats {
                        n_working_set_changes: n_changes,
                        n_refactor,
                        n_schur_updates: 0,
                        used_phase1: false,
                        time: started.elapsed(),
                    },
                });
            }

            if alpha < 0.0 {
                alpha = 0.0;
            }

            // A genuine step changes the iterate, so the null space of the
            // active set moves and the rank-tabu list (built at the prior
            // stationary vertex) no longer applies — lift it so legitimately
            // independent rows can re-enter. Degenerate `α ≈ 0` pivots leave
            // the vertex fixed, so the tabu persists and keeps breaking the
            // prune→re-add cycle.
            if alpha > opts.feas_tol {
                tabu_cons.iter_mut().for_each(|t| *t = false);
                tabu_bounds.iter_mut().for_each(|t| *t = false);
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

            // EXPAND τ growth / hard reset. Per Gill-Murray-
            // Saunders-Wright 1989 §3, τ only grows when a
            // constraint actually blocked (α < 1 with a blocker
            // picked). Growing on every iteration regardless
            // (PR #50 review C5) unnecessarily forces the hard
            // reset on non-degenerate problems. No-op when
            // `anti_cycling != Expand` (select_blocker ignores τ).
            if matches!(opts.anti_cycling, AntiCyclingChoice::Expand) && blocker.is_some() {
                expand_tol += opts.expand_tol_growth;
            }
            if expand_tol > opts.expand_tol_max {
                // Cycling-protection hard reset: snap every
                // active-bound primal exactly to its bound to
                // clean out accumulated τ-relaxation drift.
                for (i, &status) in working.bounds.iter().enumerate() {
                    match status {
                        BoundStatus::AtLower | BoundStatus::Fixed => x[i] = qp.xl[i],
                        BoundStatus::AtUpper => x[i] = qp.xu[i],
                        BoundStatus::Inactive => {}
                    }
                }
                expand_tol = opts.expand_tol_initial;
            }

            // Anti-stall monitor: latch into Bland's rule once the
            // objective stops improving for `stall_limit` consecutive
            // iterations. Uses a relative-plus-absolute improvement test
            // so it is scale-invariant (the elastic phase-1 objective is
            // ~γ·infeasibility, often 1e7+). Once latched it stays
            // latched; Bland then guarantees finite termination.
            if !force_bland {
                let obj_now = quad_objective(qp, &x);
                let improved = obj_now < best_obj - 1e-9 * best_obj.abs() - 1e-12;
                if improved {
                    best_obj = obj_now;
                    stall_iters = 0;
                } else {
                    stall_iters += 1;
                    if stall_iters >= STALL_LIMIT {
                        force_bland = true;
                    }
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
        let eq_targets: Vec<Number> = eq_rows.iter().map(|&r| qp.bl[r]).collect();

        // Factor the equality block `[H Aᵀ_eq; A_eq 0]`. If the
        // equalities are rank-deficient — redundant rows, the
        // degenerate case a pure interior-point method hands the
        // LP-crossover bridge — the saddle KKT is singular and no
        // §4.5 H-block shift can rescue a rank-deficient *constraint*
        // block (the shift exhausts and reports a recoverable failure).
        // Linear-independence guard: prune the equalities to a maximal
        // independent subset and retry once. A dropped row is a linear
        // combination of the kept ones, so at the constraint-consistent
        // cold point it is automatically satisfied — the feasible set is
        // unchanged, only the rank deficiency is removed.
        let (x, kept_eq): (Vec<Number>, Vec<usize>) =
            match self.factor_pinned_primal(qp, &eq_rows, &eq_targets, &[], &[], opts) {
                Ok(x) => (x, eq_rows.clone()),
                Err(e) if e.is_recoverable_factorization_failure() => {
                    let (kept, _) = independent_active_subset(&mut self.linsol, qp, &eq_rows, &[]);
                    if kept.len() == eq_rows.len() {
                        // Full row rank already — the failure is not a
                        // rank deficiency this guard can repair.
                        return Err(e);
                    }
                    let kept_targets: Vec<Number> = kept.iter().map(|&r| qp.bl[r]).collect();
                    let x = self.factor_pinned_primal(qp, &kept, &kept_targets, &[], &[], opts)?;
                    (x, kept)
                }
                Err(e) => return Err(e),
            };
        *n_refactor += 1;

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
        let mut kept_eq_flag = vec![false; m];
        for &r in &kept_eq {
            kept_eq_flag[r] = true;
        }
        for (i, c) in working.constraints.iter_mut().enumerate() {
            if qp.bl[i] == qp.bu[i] {
                if kept_eq_flag[i] {
                    *c = ConsStatus::Equality;
                }
                // A redundant equality dropped by the rank-repair guard
                // stays Inactive: the ratio test skips `bl == bu` rows,
                // so it never re-enters the working set, and it remains
                // satisfied as a combination of the kept equalities.
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

        // Recursive solve through the standard path, honoring the
        // same Schur-vs-refactor choice the top-level `solve` makes
        // (L15: this previously hard-called `solve_general`, so an
        // infeasible problem solved with `use_schur_updates = true`
        // silently fell back to the refactor path). Both inner solvers
        // bypass the `solve` feasibility audit, so the recursive solve
        // is still never re-audited and the recovery cannot loop.
        // Phase-1 infeasibility minimization is inherently highly
        // degenerate (many slacks sit exactly at zero), so the
        // steepest-violation default cycles at the elastic vertices the
        // GEN family and even trivial LPs like `afiro` park at. Bland's
        // rule is provably finite; use it for the recovery solve.
        let mut opts_p1 = opts.clone();
        opts_p1.anti_cycling = AntiCyclingChoice::Bland;
        let sol_aug = if opts_p1.use_schur_updates {
            self.solve_general_schur(&qp_aug, Some(&ws), &opts_p1)?
        } else {
            self.solve_general(&qp_aug, Some(&ws), &opts_p1)?
        };

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
        if feasible {
            // Elastic drove every slack to zero ⇒ the recovered `x` is
            // feasible for the original QP and (barring a non-converged
            // phase-1) optimal. Preserve the historical fast path.
            let obj = quad_objective(qp, &x);
            return Ok(QpSolution {
                x,
                lambda_g,
                lambda_x,
                working,
                obj,
                status: QpStatus::Optimal,
                stats: QpStats {
                    n_working_set_changes: sol_aug.stats.n_working_set_changes,
                    n_refactor: sol_aug.stats.n_refactor,
                    n_schur_updates: sol_aug.stats.n_schur_updates,
                    used_phase1: true,
                    time: started.elapsed(),
                },
            });
        }

        // Residual elastic slacks remain. This is *not* automatically an
        // infeasibility certificate: a phase-1 active-set solve can stall
        // at an extremely degenerate vertex — many more active rows than
        // variables and no interior (Slater fails), the m/n ≫ 1 collapsed-
        // cone geometry of #282 — and leave sub-feas_tol residual slacks
        // even though a feasible point plainly exists (e.g. the QP whose
        // feasible set is exactly {0}). Emitting `Infeasible` there is a
        // FALSE certificate: a feasible problem has no Farkas proof.
        //
        // Recovery: an active-set phase-2 solve started from a feasible
        // point of this geometry converges in a handful of pivots (it is
        // the *phase-1 feasibility hunt* that is degenerate, not the
        // phase-2 optimization). Re-solve the ORIGINAL QP, warm-started
        // (via `solve_general`, which bypasses the `solve` feasibility
        // audit and so cannot re-enter elastic) from the near-feasible
        // points phase-1 produced. If any converges to a genuinely
        // feasible optimum, return it — this turns the #282 family from a
        // false `Infeasible` into the correct `x* = 0` solution.
        //
        // Candidate seeds, cheapest-first: the recovered `x`, and the
        // elastic seed `x_orig` (0 projected into the box — feasible
        // whenever the origin is, which is the exact #282 optimum).
        let candidates = [x.clone(), x_orig.clone()];
        for seed in candidates {
            if !self.recovery_seed_usable(qp, &seed) {
                continue;
            }
            let ws_rec = QpWarmStart {
                x: seed,
                lambda_g: vec![0.0; m],
                lambda_x: vec![0.0; n],
                working: WorkingSet::cold(n, m),
            };
            // A recovery re-solve that itself fails (a warm-started active-
            // set solve on this degenerate geometry can hit a non-recoverable
            // factorization) is NON-fatal: this is a best-effort attempt to
            // improve on the phase-1 outcome, so a failure just means "this
            // seed did not pan out" — skip it and fall through to the
            // certificate/honest-status logic. Never let it turn a
            // (correctly) infeasible or honest result into a hard error.
            let rec = if opts.use_schur_updates {
                self.solve_general_schur(qp, Some(&ws_rec), opts)
            } else {
                self.solve_general(qp, Some(&ws_rec), opts)
            };
            let rec = match rec {
                Ok(r) => r,
                Err(_) => continue,
            };
            if rec.status == QpStatus::Optimal
                && self.original_qp_feasible(qp, &rec.x, opts.feas_tol)
            {
                let mut rec = rec;
                rec.stats.used_phase1 = true;
                rec.stats.time = started.elapsed();
                return Ok(rec);
            }
        }

        // Recovery found no feasible point. Only now may we speak to
        // infeasibility — and only when phase-1 actually CONVERGED to its
        // minimal-l1 optimum (a genuine certificate). If phase-1 itself
        // stalled (MaxIter / numerical breakdown) we have no certificate;
        // report that honest, non-committal status instead of asserting a
        // confident `Infeasible` we cannot back up.
        let obj = quad_objective(qp, &x);
        let status = match sol_aug.status {
            QpStatus::Optimal => QpStatus::Infeasible,
            other => other,
        };

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

    /// True when `seed` is a sane warm-start point for the phase-2
    /// recovery re-solve in [`Self::solve_elastic`]: every entry is
    /// finite and inside the variable box (with a small feas_tol slack).
    /// A seed carrying a NaN/inf or grossly out-of-box coordinate would
    /// just send `solve_general` off into its own failure path, so skip
    /// it rather than burn a recovery solve on it.
    fn recovery_seed_usable(&self, qp: &QpProblem, seed: &[Number]) -> bool {
        for (i, &xi) in seed.iter().enumerate() {
            if !xi.is_finite() {
                return false;
            }
            if qp.xl[i] > NLP_LOWER_BOUND_INF && xi < qp.xl[i] - 1e-6 {
                return false;
            }
            if qp.xu[i] < NLP_UPPER_BOUND_INF && xi > qp.xu[i] + 1e-6 {
                return false;
            }
        }
        true
    }

    /// True when `x` satisfies every original general-constraint row and
    /// variable bound to within `feas_tol`. Used to confirm a phase-2
    /// recovery re-solve landed on a genuinely feasible point before its
    /// `Optimal` status is trusted over a false `Infeasible`.
    fn original_qp_feasible(&self, qp: &QpProblem, x: &[Number], feas_tol: Number) -> bool {
        let ax = a_times_x(qp.a, x, qp.m);
        for i in 0..qp.m {
            if qp.bl[i] > NLP_LOWER_BOUND_INF && ax[i] < qp.bl[i] - feas_tol {
                return false;
            }
            if qp.bu[i] < NLP_UPPER_BOUND_INF && ax[i] > qp.bu[i] + feas_tol {
                return false;
            }
        }
        for (i, &xi) in x.iter().enumerate() {
            if qp.xl[i] > NLP_LOWER_BOUND_INF && xi < qp.xl[i] - feas_tol {
                return false;
            }
            if qp.xu[i] < NLP_UPPER_BOUND_INF && xi > qp.xu[i] + feas_tol {
                return false;
            }
        }
        true
    }

    /// Schur-based variant of [`Self::solve_general`]. Opt-in via
    /// `QpOptions::use_schur_updates`. Replaces the per-iteration
    /// refactor with a cached factor of the fixed-dim K_max
    /// matrix and Sherman-Morrison-Woodbury rank-2 updates per
    /// working-set change. Resets the cached factor when the
    /// Schur block reaches `max_schur_updates_before_refactor`.
    ///
    /// Behavior matches the refactor-per-iteration path on every
    /// problem with a positive-definite reduced Hessian: same drop /
    /// ratio-test logic, same exit conditions. The difference is the
    /// inner-loop cost: one cached resolve + small dense Schur solve
    /// per iteration, plus two cached resolves per working-set change.
    ///
    /// Caveat (indefinite reduced Hessian only): the refactor path
    /// runs `factorize_with_inertia_control` — re-checking inertia
    /// and applying a δ-shift — on *every* iteration, whereas this
    /// path only runs inertia control inside `SchurState::reset`
    /// (at init and every `max_schur_updates_before_refactor`
    /// working-set changes). The rank-2 SMW update in `apply_change`
    /// does *not* re-check inertia. A DROP enlarges the active-set
    /// null space and can expose negative curvature that the cached
    /// factor does not regularize until the next reset; an ADD only
    /// shrinks the null space and cannot introduce new negative
    /// curvature. For the convex default (`HessianInertia::Psd`,
    /// which is what the SQP driver feeds) the reduced Hessian is
    /// always PD, so the two paths are identical; the gap is latent
    /// for indefinite inputs on the opt-in `use_schur_updates = true`
    /// path. See code-review item M10.
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

        // GMSW EXPAND τ — same semantics as in solve_general.
        let mut expand_tol = opts.expand_tol_initial;

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
            let (mut alpha, blocker) = select_blocker(&candidates, opts, expand_tol, false);

            // F2(a), Schur path. Same certificate as `solve_general`: an
            // empty candidate list means `+p` is feasible for every step
            // length, so a zero-curvature descent `p` is a recession ray.
            // The Schur driver hides the per-iterate inertia shift inside
            // `SchurState`, so unlike `solve_general` we cannot gate on
            // `delta > 0` here — but `ray_is_unbounded_descent` rejects
            // any direction with measurable curvature (`‖Hp‖∞` above the
            // 1e-10·‖H‖ structural-zero floor), so a PD-reduced-Hessian
            // Newton step never certifies and the unconditional check is
            // still safe.
            if candidates.is_empty() && ray_is_unbounded_descent(qp.h, qp.g, &x, &p) {
                return Ok(QpSolution {
                    obj: Number::NEG_INFINITY,
                    x,
                    lambda_g: vec![0.0; m],
                    lambda_x: vec![0.0; n],
                    working,
                    status: QpStatus::Unbounded,
                    stats: QpStats {
                        n_working_set_changes: n_changes,
                        n_refactor,
                        n_schur_updates,
                        used_phase1: false,
                        time: started.elapsed(),
                    },
                });
            }

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

            // EXPAND τ growth / hard reset (same semantics as in
            // solve_general; PR #50 C5 fix).
            if matches!(opts.anti_cycling, AntiCyclingChoice::Expand) && blocker.is_some() {
                expand_tol += opts.expand_tol_growth;
            }
            if expand_tol > opts.expand_tol_max {
                for (i, &status) in working.bounds.iter().enumerate() {
                    match status {
                        BoundStatus::AtLower | BoundStatus::Fixed => x[i] = qp.xl[i],
                        BoundStatus::AtUpper => x[i] = qp.xu[i],
                        BoundStatus::Inactive => {}
                    }
                }
                expand_tol = opts.expand_tol_initial;
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

/// Relative tolerance for the modified-Gram-Schmidt rank test in
/// [`independent_active_subset`]. A candidate normal whose component
/// orthogonal to the already-accepted normals falls below this
/// fraction of its original norm is judged linearly dependent
/// (redundant) and dropped.
const RANK_REL_TOL: Number = 1e-9;

/// Rate threshold (relative to the step inf-norm) below which a
/// rank-tabu'd row is treated as genuinely linearly dependent and
/// kept out of the ratio test. A row pruned as a linear combination
/// of the kept active rows has true `a·p = 0`, so numerically
/// `|a·p|` sits at the refined-solve residual scale (≈1e-12·‖p‖);
/// anything above `TABU_DRIFT_REL·‖p‖∞` is an O(1) fraction of the
/// step — a *genuine* blocker that the active set's evolution has
/// re-exposed. Suppressing such a row hides it from the ratio test,
/// lets the step overshoot (observed ‖p‖→1e14 on degenerate NETLIB
/// gen), and voids Bland's lowest-index guarantee (it can only rank
/// the surviving candidates). So the tabu suppresses a row only while
/// its rate stays in this drift band; a genuine rate re-admits it.
const TABU_DRIFT_REL: Number = 1e-7;

/// Select a maximal linearly-independent subset of the given active
/// constraint / bound normals by modified Gram-Schmidt with one
/// reorthogonalization pass.
///
/// Returns `(keep_cons, keep_bounds)` — the entries of `active_cons` /
/// `active_bounds`, in their original order, whose normals are
/// linearly independent of the earlier-kept ones. Dependent
/// (redundant) rows are omitted.
///
/// This is the linear-independence guard that lets the active-set
/// engine pin a degenerate / rank-deficient active set. A redundant
/// row is a linear combination of kept rows, so at any
/// constraint-consistent point it is automatically satisfied: dropping
/// it leaves the feasible vertex unchanged while removing the rank
/// deficiency that makes the active-set KKT singular (no H-block shift
/// can rescue a rank-deficient *constraint* block). General-constraint
/// rows are processed before variable bounds, so equality / general
/// rows are preferred over bounds when a tie must be broken.
fn independent_active_subset(
    linsol: &mut LinearSolver,
    qp: &QpProblem,
    active_cons: &[usize],
    active_bounds: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    // Prefer the backend's sparse rank-reveal (feral's `SparseLu`
    // degeneracy probe) when available — it factors a sparse augmented
    // system in O(nnz) instead of the dense O(k²·n) modified-Gram-Schmidt
    // grind, which is the operation that grinds large degenerate LPs
    // (the NETLIB GEN family) to a halt. Fall back to dense MGS for
    // backends that don't rank-reveal (e.g. MA57).
    if linsol.provides_degeneracy_detection() {
        if let Some(kept) = independent_active_subset_sparse(linsol, qp, active_cons, active_bounds)
        {
            return kept;
        }
    }
    independent_active_subset_dense(qp, active_cons, active_bounds)
}

/// Sparse linear-independence guard via the backend's Ipopt-style
/// degeneracy probe. Builds the active-row Jacobian `J` as a 1-based
/// triplet (`n_cols = qp.n`; general rows `0..active_cons.len()`
/// ordered before bound rows, so general rows win ties — matching the
/// dense path), calls `determine_dependent_rows`, and maps the flagged
/// rows back to `(keep_cons, keep_bounds)`. Returns `None` on a probe
/// failure so the caller can fall back to dense MGS.
fn independent_active_subset_sparse(
    linsol: &mut LinearSolver,
    qp: &QpProblem,
    active_cons: &[usize],
    active_bounds: &[usize],
) -> Option<(Vec<usize>, Vec<usize>)> {
    let n_cols = qp.n;
    let n_c = active_cons.len();
    let n_b = active_bounds.len();
    let n_rows = n_c + n_b;
    if n_rows == 0 {
        return Some((Vec::new(), Vec::new()));
    }

    // Each active general row maps to J-row `pos` (its index in
    // `active_cons`); each active bound maps to J-row `n_c + b`.
    let mut j_row_of_con: Vec<Option<usize>> = vec![None; qp.m];
    for (pos, &row) in active_cons.iter().enumerate() {
        j_row_of_con[row] = Some(pos);
    }

    let mut irn: Vec<Index> = Vec::new();
    let mut jcn: Vec<Index> = Vec::new();
    let mut vals: Vec<Number> = Vec::new();

    // General-constraint rows: scatter the sparse Jacobian `A` in one pass.
    let a_irows = qp.a.irows();
    let a_jcols = qp.a.jcols();
    let a_vals = qp.a.values();
    for k in 0..a_irows.len() {
        let row = (a_irows[k] - 1) as usize;
        if let Some(pos) = j_row_of_con[row] {
            let col = (a_jcols[k] - 1) as usize;
            irn.push((pos + 1) as Index);
            jcn.push((col + 1) as Index);
            vals.push(a_vals[k]);
        }
    }

    // Variable-bound rows: a unit entry `(n_c + b, var, 1)`.
    for (b, &var) in active_bounds.iter().enumerate() {
        irn.push((n_c + b + 1) as Index);
        jcn.push((var + 1) as Index);
        vals.push(1.0);
    }

    let mut c_deps: Vec<Index> = Vec::new();
    let st = linsol.determine_dependent_rows(
        n_rows as Index,
        n_cols as Index,
        &irn,
        &jcn,
        &vals,
        &mut c_deps,
    );
    if st != ESymSolverStatus::Success {
        return None;
    }

    let mut dropped = vec![false; n_rows];
    for &d in &c_deps {
        let d = d as usize;
        if d < n_rows {
            dropped[d] = true;
        }
    }

    let mut keep_cons = Vec::with_capacity(n_c);
    for (pos, &row) in active_cons.iter().enumerate() {
        if !dropped[pos] {
            keep_cons.push(row);
        }
    }
    let mut keep_bounds = Vec::with_capacity(n_b);
    for (b, &var) in active_bounds.iter().enumerate() {
        if !dropped[n_c + b] {
            keep_bounds.push(var);
        }
    }

    Some((keep_cons, keep_bounds))
}

/// Dense modified-Gram-Schmidt linear-independence guard — the fallback
/// for backends without a sparse rank-reveal. Allocates a dense normal
/// per active row and orthogonalizes; O(k²·n). Retained byte-for-byte
/// for the MA57 backend.
fn independent_active_subset_dense(
    qp: &QpProblem,
    active_cons: &[usize],
    active_bounds: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let n = qp.n;

    // Gather dense normals for the active general-constraint rows from
    // the sparse Jacobian in one pass.
    let mut pos_of_row: Vec<Option<usize>> = vec![None; qp.m];
    for (pos, &row) in active_cons.iter().enumerate() {
        pos_of_row[row] = Some(pos);
    }
    let mut cons_normals = vec![vec![0.0; n]; active_cons.len()];
    let a_irows = qp.a.irows();
    let a_jcols = qp.a.jcols();
    let a_vals = qp.a.values();
    for k in 0..a_irows.len() {
        let row = (a_irows[k] - 1) as usize;
        if let Some(pos) = pos_of_row[row] {
            let col = (a_jcols[k] - 1) as usize;
            cons_normals[pos][col] += a_vals[k];
        }
    }

    let mut basis: Vec<Vec<Number>> = Vec::new();
    let mut keep_cons = Vec::new();
    let mut keep_bounds = Vec::new();

    for (pos, &row) in active_cons.iter().enumerate() {
        let mut v = std::mem::take(&mut cons_normals[pos]);
        if accept_if_independent(&mut v, &mut basis) {
            keep_cons.push(row);
        }
    }
    for &var in active_bounds {
        let mut v = vec![0.0; n];
        v[var] = 1.0;
        if accept_if_independent(&mut v, &mut basis) {
            keep_bounds.push(var);
        }
    }

    (keep_cons, keep_bounds)
}

/// One modified-Gram-Schmidt step: orthogonalize `v` against `basis`
/// (two passes for numerical robustness against loss of orthogonality).
/// If the residual keeps a non-negligible fraction of `v`'s original
/// norm, normalize it, append it to `basis`, and return `true` (the row
/// is linearly independent); otherwise leave `basis` unchanged and
/// return `false` (linearly dependent / redundant).
fn accept_if_independent(v: &mut [Number], basis: &mut Vec<Vec<Number>>) -> bool {
    let orig = dot(v, v).sqrt();
    if orig == 0.0 {
        return false;
    }
    for _pass in 0..2 {
        for q in basis.iter() {
            let d = dot(q, v);
            if d != 0.0 {
                for (vi, &qi) in v.iter_mut().zip(q.iter()) {
                    *vi -= d * qi;
                }
            }
        }
    }
    let r = dot(v, v).sqrt();
    if r > RANK_REL_TOL * orig {
        let inv = 1.0 / r;
        basis.push(v.iter().map(|&vi| vi * inv).collect());
        true
    } else {
        false
    }
}

fn dot(a: &[Number], b: &[Number]) -> Number {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
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
///
/// `expand_tol` is the current GMSW EXPAND τ (only consumed when
/// `opts.anti_cycling = Expand`; pass 0.0 to disable). Non-zero
/// τ relaxes the Phase-1 minimum ratio by `τ / |a·p|` per
/// candidate, ensuring strictly positive step length even at
/// degenerate vertices where multiple constraints have α = 0
/// under the strict ratio test.
fn select_blocker(
    candidates: &[(BlockerTarget, f64, f64)],
    opts: &QpOptions,
    expand_tol: f64,
    force_bland: bool,
) -> (f64, Option<BlockerTarget>) {
    if candidates.is_empty() {
        return (1.0, None);
    }
    // Pass 1: minimum ratio (strict and τ-relaxed).
    let mut alpha_min = 1.0_f64;
    let mut alpha_min_relaxed = 1.0_f64;
    for &(_, r, ap_mag) in candidates {
        if r < alpha_min {
            alpha_min = r;
        }
        let r_relaxed = if ap_mag > 0.0 {
            r + expand_tol / ap_mag
        } else {
            r
        };
        if r_relaxed < alpha_min_relaxed {
            alpha_min_relaxed = r_relaxed;
        }
    }
    if alpha_min >= 1.0 {
        return (1.0, None);
    }

    // The anti-stall latch forces Bland (strict-min, lowest-index)
    // regardless of the configured rule.
    let effective = if force_bland {
        AntiCyclingChoice::Bland
    } else {
        opts.anti_cycling
    };
    match effective {
        AntiCyclingChoice::None | AntiCyclingChoice::Bland => {
            // Strict-min: pick the first candidate achieving
            // `alpha_min` (encounter order ⇒ lowest index for ties).
            let mut best: Option<(BlockerTarget, f64)> = None;
            for &(target, r, _) in candidates {
                if r > alpha_min {
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
            // Harris two-pass with τ-relaxation. Phase 1 uses
            // `r_relaxed = r + τ/|a·p|`; Phase 2 picks largest
            // `|a·p|` among candidates within `tol · (1 + |α_min_relaxed|)`
            // of `α_min_relaxed`. The step length used is the
            // SELECTED candidate's *true* ratio, clamped from
            // below by `α_min_relaxed` so that even at a
            // degenerate vertex (true ratio = 0) we take a
            // strictly positive step of magnitude ≈ τ/|a·p|.
            let tol = opts.feas_tol * (1.0 + alpha_min_relaxed.abs());
            let mut best: Option<(BlockerTarget, f64, f64)> = None;
            for &(target, r, ap_mag) in candidates {
                let r_relaxed = if ap_mag > 0.0 {
                    r + expand_tol / ap_mag
                } else {
                    r
                };
                if r_relaxed > alpha_min_relaxed + tol {
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
            match best {
                Some((target, r, _)) => {
                    // Floor the step length at the τ-relaxed minimum so
                    // we never freeze at α = 0; cap at 1.0.
                    let alpha = r.max(alpha_min_relaxed).min(1.0).max(0.0);
                    (alpha, Some(target))
                }
                None => {
                    // Pass 2 admitted nothing. This happens when every
                    // candidate's τ-relaxed ratio exceeds the artificial
                    // `α_min_relaxed = 1.0` initialization cap by more than
                    // `tol` — reachable when |a·p| ≈ feas_tol makes
                    // `τ/|a·p|` inflate `r_relaxed` above `1 + tol` for ALL
                    // candidates (so the recorded minimum is the cap, which
                    // no real candidate attains). Fall back to the strict
                    // minimum-ratio blocker (guaranteed to exist since
                    // `α_min < 1.0`) and step exactly `α_min`: never freeze,
                    // panic, or overstep the first blocking constraint.
                    let mut fb: Option<BlockerTarget> = None;
                    for &(target, r, _) in candidates {
                        if r <= alpha_min {
                            fb = Some(target);
                            break;
                        }
                    }
                    (alpha_min, fb)
                }
            }
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

        // Warm-start feasibility pre-check (companion to the M5
        // post-hoc audit below). A warm start whose primal is already
        // infeasible cannot be repaired by the zero-RHS warm inner
        // loop: its ratio test sees already-violated inactive rows,
        // yields negative step lengths that clamp to zero, and freezes
        // the objective until `MaxIter` (observed on degenerate NETLIB
        // `gen`, where the crossover hint pins a rank-deficient vertex
        // that violates ~hundreds of inactive rows). Route such a start
        // straight to l1-elastic phase-1 — the same recovery the cold
        // path takes when `cold_general_initial` returns infeasible, and
        // the M5 audit takes post-hoc. `solve_elastic` seeds a slack-
        // feasible augmented problem and recurses through `solve_general`
        // /`solve_general_schur` *directly*, bypassing this entry, so the
        // recovery cannot loop. A feasible warm start (the common case —
        // a good crossover/SQP hint) passes untouched.
        if let Some(w) = ws {
            if !point_is_feasible(qp, &w.x, opts.feas_tol) {
                return self.solve_elastic(qp, opts);
            }
        }

        let has_general_inequality = !is_all_equality_constraints(qp);

        // Any of: caller provided a warm start, or the problem has at
        // least one one-sided / two-sided general inequality row.
        if ws.is_some() || has_general_inequality {
            let sol = if opts.use_schur_updates {
                self.solve_general_schur(qp, ws, opts)?
            } else {
                self.solve_general(qp, ws, opts)?
            };

            // Feasibility audit (M5): the warm-start inner loop steps
            // with a zero-RHS active-set system, so the residuals of
            // caller-marked-active rows are frozen and an equality row
            // left `Inactive` can never enter the working set — either
            // way the loop can converge to a constraint-violating point
            // and label it `Optimal`. Audit every row + bound; on
            // violation, recover through elastic mode (the same
            // recovery the cold path uses when `cold_general_initial`
            // returns an infeasible point). `solve_elastic` recurses
            // through `solve_general` / `solve_general_schur` *directly*
            // (per `use_schur_updates`), bypassing this entry, and seeds
            // a slack-feasible augmented problem — so the recursive solve
            // is never re-audited and the recovery cannot loop. Feasible
            // warm/cold results pass untouched.
            if matches!(sol.status, QpStatus::Optimal)
                && !point_is_feasible(qp, &sol.x, opts.feas_tol)
            {
                return self.solve_elastic(qp, opts);
            }
            return Ok(sol);
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

    fn solve_with_working_set(
        &mut self,
        qp: &QpProblem,
        working: &crate::working_set::WorkingSet,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        qp.validate()?;
        working.validate_dims(qp.n, qp.m)?;

        // Build the active-row index lists from the supplied
        // working set.
        let active_cons: Vec<usize> = (0..qp.m)
            .filter(|&i| working.constraints[i].is_active())
            .collect();
        let active_bounds: Vec<usize> = (0..qp.n)
            .filter(|&i| working.bounds[i].is_active())
            .collect();

        // The boundary value each active row / bound is pinned to.
        let cons_target = |i: usize| match working.constraints[i] {
            ConsStatus::AtLower | ConsStatus::Equality => qp.bl[i],
            ConsStatus::AtUpper => qp.bu[i],
            ConsStatus::Inactive => unreachable!(),
        };
        let bound_target = |i: usize| match working.bounds[i] {
            BoundStatus::AtLower | BoundStatus::Fixed => qp.xl[i],
            BoundStatus::AtUpper => qp.xu[i],
            BoundStatus::Inactive => unreachable!(),
        };
        let cons_targets: Vec<Number> = active_cons.iter().map(|&i| cons_target(i)).collect();
        let bound_targets: Vec<Number> = active_bounds.iter().map(|&i| bound_target(i)).collect();

        // Factor the pinned KKT for a primal that satisfies the hinted
        // active rows. If the hint is rank-deficient — a degenerate
        // optimum can pin more binding rows than there are variables,
        // and the LP-crossover bridge hands over redundant equality
        // rows — the saddle KKT is singular and the §4.5 H-shift cannot
        // repair a rank-deficient *constraint* block. Linear-
        // independence guard: prune the active set to a maximal
        // independent subset, retry once, and forward the pruned
        // working set so the inner loop starts from a full-rank state.
        // Dropped rows are linear combinations of the kept ones, hence
        // satisfied at the recovered primal.
        let (x_init, fwd_working) = match self.factor_pinned_primal(
            qp,
            &active_cons,
            &cons_targets,
            &active_bounds,
            &bound_targets,
            opts,
        ) {
            Ok(x) => (x, working.clone()),
            Err(e) if e.is_recoverable_factorization_failure() => {
                let (kc, kb) =
                    independent_active_subset(&mut self.linsol, qp, &active_cons, &active_bounds);
                if kc.len() == active_cons.len() && kb.len() == active_bounds.len() {
                    // Full rank already — not a deficiency this repairs.
                    return Err(e);
                }
                let kc_targets: Vec<Number> = kc.iter().map(|&i| cons_target(i)).collect();
                let kb_targets: Vec<Number> = kb.iter().map(|&i| bound_target(i)).collect();
                let x = self.factor_pinned_primal(qp, &kc, &kc_targets, &kb, &kb_targets, opts)?;

                // Forward a pruned working set: dropped active rows /
                // bounds revert to Inactive. A dropped row has `a·p = 0`
                // along every active-set step (it lies in the kept rows'
                // span), so the inner loop never re-adds it and it stays
                // at its boundary.
                let mut fwd = working.clone();
                let mut keep_c = vec![false; qp.m];
                for &i in &kc {
                    keep_c[i] = true;
                }
                let mut keep_b = vec![false; qp.n];
                for &i in &kb {
                    keep_b[i] = true;
                }
                for i in 0..qp.m {
                    if working.constraints[i].is_active() && !keep_c[i] {
                        fwd.constraints[i] = ConsStatus::Inactive;
                    }
                }
                for i in 0..qp.n {
                    if working.bounds[i].is_active() && !keep_b[i] {
                        fwd.bounds[i] = BoundStatus::Inactive;
                    }
                }
                (x, fwd)
            }
            Err(e) => return Err(e),
        };

        // The inner loop recomputes multipliers each iteration from a
        // fresh KKT solve, so the warm-start multipliers are unused;
        // pass zeros and let `solve_general` drive from `(x, working)`.
        let ws = QpWarmStart {
            x: x_init,
            lambda_g: vec![0.0; qp.m],
            lambda_x: vec![0.0; qp.n],
            working: fwd_working,
        };
        self.solve(qp, Some(&ws), opts)
    }
}

/// Evaluate `½ xᵀ H x + gᵀ x`, walking the symmetric Hessian once
/// and fanning each off-diagonal entry into both halves.
/// Feasibility audit for a candidate solution `x` (M5). Checks every
/// general-constraint row — **including equality rows** (`bl == bu`) —
/// and every variable bound against `feas_tol`. Returns `true` iff `x`
/// violates none of them.
///
/// The warm-start path of [`ParametricActiveSetSolver::solve_general`]
/// trusts the caller's `(x, working)` and steps with a zero-RHS active-
/// set system, so the residuals of rows the caller marked active are
/// frozen and never re-checked; an equality row the caller left
/// `Inactive` is skipped by the ratio test (`bl == bu` ⇒ `continue`)
/// and can never enter the working set. Either way the inner loop can
/// reach a KKT-stationary point that violates a constraint and report
/// it as `Optimal`. `solve` runs this audit before trusting an
/// `Optimal` and recovers through elastic mode on failure.
fn point_is_feasible(qp: &QpProblem, x: &[Number], feas_tol: Number) -> bool {
    let ax = a_times_x(qp.a, x, qp.m);
    for i in 0..qp.m {
        if qp.bl[i] > NLP_LOWER_BOUND_INF && ax[i] < qp.bl[i] - feas_tol {
            return false;
        }
        if qp.bu[i] < NLP_UPPER_BOUND_INF && ax[i] > qp.bu[i] + feas_tol {
            return false;
        }
    }
    for (i, &xi) in x.iter().enumerate() {
        if qp.xl[i] > NLP_LOWER_BOUND_INF && xi < qp.xl[i] - feas_tol {
            return false;
        }
        if qp.xu[i] < NLP_UPPER_BOUND_INF && xi > qp.xu[i] + feas_tol {
            return false;
        }
    }
    true
}

/// Two intrinsic clauses of a certified-recession-ray test for QP
/// unboundedness. A QP `min ½xᵀHx + gᵀx s.t. Ax = b` is unbounded
/// below iff there is a direction `d` with `Hd = 0` (zero curvature —
/// for PSD `H` equivalent to `dᵀHd = 0`), `Ad = 0` (stays feasible),
/// and `gᵀd < 0` (descent). This helper checks the two clauses that
/// depend only on `(H, g)` and the current iterate `x_cand`:
///   (i)  zero curvature  `‖Hd‖∞ ≈ 0` relative to `‖H‖`  (H ≡ 0 ⇒ flat),
///   (ii) strict descent of the *local* gradient `(H·x_cand + g)ᵀd < 0`.
///
/// **Feasibility of the ray is the caller's responsibility** — the
/// call sites certify it by different (both locally valid) arguments:
/// the equality-only solve maintains `Ax = b` so `A(x/‖x‖) = b/‖x‖ → 0`
/// as the iterate blows up; the active-set loop reaches its check only
/// when the ratio test finds NO inactive row blocking along `dir` (and
/// `dir` already lies in the active constraints' null space).
///
/// `dir` need not be normalized — the test is scale-invariant.
///
/// The curvature clause is deliberately near-exact (`1e-10·‖H‖`): a
/// false `Unbounded` is the dangerous direction. For PSD `H`, any
/// measurable curvature along `d` means a *finite* minimizer in that
/// direction at `‖∇q‖/λ`, however large — an earlier `dᵀHd ≤ 1e-3·‖H‖`
/// version certified `Unbounded` on bounded QPs whose softest mode sat
/// 3+ orders below the stiffest entry (e.g. `H = diag(1, 1e-4, 0)`,
/// `g = (0, -1, 0)`, true minimum −5000 at `x₂ = 10⁴`). Curvature below
/// `1e-10·‖H‖` is beneath any meaningful precision of the problem data
/// and is treated as structurally zero. Soft-but-real modes therefore
/// fall on the conservative side (reported bounded), never falsely
/// unbounded.
///
/// The descent clause uses the local gradient `H·x_cand + g`, not the
/// origin gradient `g`: with `Hd ≈ 0` enforced only to tolerance, the
/// two can disagree at a large iterate (the earlier `gᵀd` version read
/// "descent" while sitting essentially at the minimizer). For a genuine
/// recession ray they coincide (`xᵀ(Hd) ≈ 0`).
fn ray_is_unbounded_descent(
    h: &pounce_linalg::triplet::SymTMatrix,
    g: &[Number],
    x_cand: &[Number],
    dir: &[Number],
) -> bool {
    let norm = dir.iter().map(|v| v * v).sum::<Number>().sqrt();
    if norm == 0.0 {
        return false;
    }
    let inv = 1.0 / norm;

    // ‖Hd‖∞, H·x_cand, and ‖H‖ (max |stored entry|), using the symmetric
    // triplet convention (off-diagonal pairs stored once ⇒ scatter both
    // (i,j) and (j,i)).
    let n = dir.len();
    let mut hd = vec![0.0; n];
    let mut hx = vec![0.0; n];
    let mut h_scale: Number = 0.0;
    let irows = h.irows();
    let jcols = h.jcols();
    let vals = h.values();
    for k in 0..irows.len() {
        let i = (irows[k] - 1) as usize;
        let j = (jcols[k] - 1) as usize;
        let v = vals[k];
        h_scale = h_scale.max(v.abs());
        hd[i] += v * dir[j] * inv;
        hx[i] += v * x_cand[j];
        if i != j {
            hd[j] += v * dir[i] * inv;
            hx[j] += v * x_cand[i];
        }
    }
    let hd_inf = hd.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    let zero_curvature = if h_scale > 0.0 {
        hd_inf <= 1e-10 * h_scale
    } else {
        true // H ≡ 0: every direction is a zero-curvature ray.
    };

    // Local directional derivative (H·x_cand + g)ᵀd vs ‖g‖₂ — strict
    // (numerically meaningful) descent.
    let slope: Number = g
        .iter()
        .zip(hx.iter())
        .zip(dir.iter())
        .map(|((&gi, &hxi), &di)| (gi + hxi) * di * inv)
        .sum();
    let g_norm = g.iter().map(|v| v * v).sum::<Number>().sqrt();
    let descent = slope < -1e-6 * g_norm.max(1.0);

    zero_curvature && descent
}

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

#[cfg(test)]
mod select_blocker_tests {
    //! Unit tests for the GMSW EXPAND ratio test in `select_blocker`.
    //! These live inside `solver` (not `crate::tests`) so they can reach
    //! the private `select_blocker`/`BlockerTarget` items.
    use super::{BlockerTarget, select_blocker};
    use crate::options::{AntiCyclingChoice, QpOptions};
    use crate::working_set::BoundStatus;

    fn expand_opts(feas_tol: f64) -> QpOptions {
        QpOptions {
            feas_tol,
            anti_cycling: AntiCyclingChoice::Expand,
            ..QpOptions::default()
        }
    }

    /// Regression for H6: the EXPAND branch panicked (`best.expect`)
    /// when every candidate's τ-relaxed ratio `r + τ/|a·p|` exceeded
    /// the artificial `α_min_relaxed = 1.0` initialization cap by more
    /// than `tol`. Reachable with a *single* candidate that has a true
    /// blocking ratio `r < 1` but a tiny `|a·p| ≈ feas_tol`, so
    /// `τ/|a·p|` inflates `r_relaxed` far above `1`. Pre-fix this hits
    /// `best = None → panic`; post-fix it falls back to the strict
    /// minimum-ratio blocker and steps exactly `α_min = r`.
    #[test]
    fn expand_tau_inflation_falls_back_to_strict_min_no_panic() {
        let opts = expand_opts(1e-6);
        // expand_tol (τ) = 1e-3, ap_mag = 1e-9 ⇒ r_relaxed ≈ 0.5 + 1e6.
        let candidates = [(BlockerTarget::Bound(0, BoundStatus::AtLower), 0.5, 1e-9)];
        let (alpha, blocker) = select_blocker(&candidates, &opts, 1e-3, false);
        assert!(
            matches!(blocker, Some(BlockerTarget::Bound(0, BoundStatus::AtLower))),
            "expected the sole candidate as blocker, got {:?}",
            blocker.map(|b| match b {
                BlockerTarget::Bound(i, _) => ("bound", i),
                BlockerTarget::Cons(i, _) => ("cons", i),
            })
        );
        // Step the strict ratio, never the bogus 1.0 floor (which would
        // overstep the constraint).
        assert!(
            (alpha - 0.5).abs() < 1e-12,
            "expected α = 0.5 (strict min), got {alpha}"
        );
    }

    /// Multiple inflated candidates: the fallback must still pick the
    /// strict minimum-ratio one (here index 1, r = 0.25) and step its
    /// ratio, not the larger-index r.
    #[test]
    fn expand_fallback_selects_strict_minimum_among_inflated() {
        let opts = expand_opts(1e-6);
        let candidates = [
            (BlockerTarget::Bound(0, BoundStatus::AtLower), 0.75, 1e-9),
            (BlockerTarget::Bound(1, BoundStatus::AtUpper), 0.25, 1e-9),
        ];
        let (alpha, blocker) = select_blocker(&candidates, &opts, 1e-3, false);
        assert!(
            matches!(blocker, Some(BlockerTarget::Bound(1, BoundStatus::AtUpper))),
            "expected the strict-min candidate (index 1)"
        );
        assert!(
            (alpha - 0.25).abs() < 1e-12,
            "expected α = 0.25, got {alpha}"
        );
    }

    /// Non-degenerate EXPAND still works: a candidate with a healthy
    /// `|a·p|` keeps its τ-relaxed ratio below the cap, so Pass 2
    /// admits it normally (no fallback).
    #[test]
    fn expand_normal_case_admits_in_pass_two() {
        let opts = expand_opts(1e-6);
        let candidates = [(BlockerTarget::Bound(0, BoundStatus::AtLower), 0.5, 1.0)];
        let (alpha, blocker) = select_blocker(&candidates, &opts, 1e-9, false);
        assert!(matches!(
            blocker,
            Some(BlockerTarget::Bound(0, BoundStatus::AtLower))
        ));
        assert!(alpha >= 0.5 && alpha <= 1.0, "α in range, got {alpha}");
    }
}
