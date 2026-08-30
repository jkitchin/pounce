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
use crate::negcurv::SecondOrder;
use crate::options::{AntiCyclingChoice, QpOptions};
use crate::problem::{
    HessianInertia, ParametricSource, QpProblem, QpSolution, QpStats, QpWarmStart,
    SecondOrderVerdict,
};
use crate::working_set::{BoundStatus, ConsStatus, WorkingSet};
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_common::{Index, Number};
use pounce_linalg::triplet::{SymTMatrix, SymTMatrixSpace};
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_linsol::status::ESymSolverStatus;

/// Re-pin rounds [`ParametricActiveSetSolver::repair_pinned_hint`] will spend
/// on an infeasible warm-start primal before giving up on it. Each round adds
/// the rows the current pinned point violates and re-factors, so the cost of a
/// failed repair is bounded by this many pinned-KKT factorizations. One round
/// suffices for the parametric case the repair targets (a hint whose active
/// set has drifted by a few entries); the extra rounds cover a re-pin that
/// exposes a second row behind the first.
const PIN_REPAIR_MAX_ROUNDS: usize = 3;

/// Violated rows [`ParametricActiveSetSolver::repair_pinned_hint`] will always
/// try to re-pin, however small the hint's active set. Beyond this floor the
/// budget scales with the active set (a quarter of it): a hint wrong in a few
/// entries is worth repairing, one wrong in a large fraction of its rows is
/// the badly-wrong hint that l1-elastic phase-1 exists for.
const PIN_REPAIR_MIN_ROWS: usize = 4;

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
    /// sol_prev)` to `qp_new`.
    ///
    /// The path interpolates `g` and the row bounds, and is traced when the
    /// two problems have the same shape and a bit-identical `H`. A pair
    /// that fails that — or a previous solve that did not reach
    /// [`QpStatus::Optimal`], or a path the tracer cannot complete — falls
    /// back to [`solve_with_working_set`](Self::solve_with_working_set) on
    /// `sol_prev`'s working set, and to a cold [`solve`](Self::solve) when
    /// that working set is unusable too. So handing over a previous solve
    /// that turns out ineligible costs nothing beyond the cold solve the
    /// caller would have done anyway.
    ///
    /// **`A` and `xl`/`xu` are not interpolated and not guarded on.** A pair
    /// differing in either is still traced, and the path then extrapolates
    /// about a point that is not on the previous problem's solution
    /// manifold. The result stays correct — the path is a predictor and the
    /// corrector re-solves — but the active-set prediction degrades, and how
    /// much is not predictable from the size of the change. Callers wanting
    /// the path to model what it is given should keep `A` and the variable
    /// bounds fixed and vary `g` and the row bounds, which is the parametric
    /// family this entry point is for. See gh #602.
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
    /// Crate-visible so sibling modules — notably [`crate::homotopy`] — can
    /// reuse the rank-repair helpers, which take the shared linear-solver
    /// backend rather than owning one.
    pub(crate) linsol: LinearSolver,
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
    pub(crate) fn factorize_with_inertia_control(
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
            if crate::deadline::expired() {
                // Cancellation is an *error*, not a value. `Ok(current)` here
                // would hand the caller an `rhs` that was never solved — it
                // still holds `[-g; targets]` — while claiming a shift of
                // `current` succeeded. `solve_equality_only` reads that back as
                // `[x*; λ*]`, sees `delta == 0` (so it also skips the
                // masked-rank-deficiency probe and the recession-ray test), and
                // returns `x = -g` as `QpStatus::Optimal`; `audit_and_repair`
                // only checks primal feasibility, so any such point that
                // happens to satisfy `Ax = b` reaches the user as a certified
                // optimum. Propagating an error instead makes `?` force every
                // caller to deal with it, and the entry points below convert it
                // to the soft `QpStatus::TimeLimit`.
                return Err(QpError::DeadlineExpired);
            }
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
        let delta =
            self.factorize_with_inertia_control(kkt, &mut rhs, (k_c + k_b) as i32, n, opts)?;

        // Masked-rank-deficiency guard. No H-block δ·I shift can repair a
        // rank-deficient *constraint* block — but a large enough δ can grow
        // the H diagonal until the backend stops flagging the singular block
        // and returns a garbage solution instead of a failure (feral masks
        // the null direction around δ≈1e8). The pinned callers
        // (`cold_general_initial`, `solve_with_working_set`, and the #313
        // equality+bounds path) rely on a *reported* recoverable failure to
        // trigger their linear-independence prune; a masked deficiency slips
        // past silently and the solve churns to `MaxIter` (or worse, reports a
        // wrong `Optimal`). A nonzero δ on a pinned KKT is the tell: only then
        // do we rank-reveal the pinned rows, and if any is redundant we
        // convert the spurious success into the recoverable failure the
        // callers already know how to prune. A δ > 0 with a full-rank
        // constraint block is a legitimate indefinite-reduced-Hessian shift
        // and passes through untouched (the common case is δ == 0, which skips
        // the probe entirely).
        if delta > 0.0 {
            let (kc, kb) =
                independent_active_subset(&mut self.linsol, qp, active_cons, active_bounds);
            if kc.len() < k_c || kb.len() < k_b {
                return Err(QpError::LinearSolverFailure(
                    "pinned KKT constraint block is rank-deficient (inertia shift masked a \
                     singular constraint block); prune to a linearly-independent subset"
                        .into(),
                ));
            }
        }
        Ok(rhs[..n].to_vec())
    }

    /// Pin every active row / bound of `working` to its boundary value and
    /// factor that KKT for a primal `x`, returning `x` together with the
    /// working set actually pinned.
    ///
    /// If the hint is rank-deficient — a degenerate optimum can pin more
    /// binding rows than there are variables, and the LP-crossover bridge
    /// hands over redundant equality rows — the saddle KKT is singular and
    /// the §4.5 H-shift cannot repair a rank-deficient *constraint* block.
    /// Linear-independence guard: prune the active set to a maximal
    /// independent subset, retry once, and return the pruned working set so
    /// the inner loop starts from a full-rank state. Dropped rows are linear
    /// combinations of the kept ones, hence satisfied at the recovered primal
    /// — and they stay `Inactive` in the returned set, since the ratio test
    /// skips `bl == bu` rows so a dropped equality can never re-enter.
    fn pin_working_set(
        &mut self,
        qp: &QpProblem,
        working: &WorkingSet,
        opts: &QpOptions,
    ) -> Result<(Vec<Number>, WorkingSet), QpError> {
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

        match self.factor_pinned_primal(
            qp,
            &active_cons,
            &cons_targets,
            &active_bounds,
            &bound_targets,
            opts,
        ) {
            Ok(x) => Ok((x, working.clone())),
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
                Ok((x, fwd))
            }
            Err(e) => Err(e),
        }
    }

    /// Repair a pinned warm-start primal that came out infeasible, rather than
    /// let `solve`'s admission pre-check discard the whole hint (#428).
    ///
    /// When the true active set has moved, the hint still pins a row that
    /// should have been released, so the pinned primal overshoots some *other*
    /// row or bound — by roughly the distance the problem moved. The old
    /// behavior was all-or-nothing: that point failed the admission pre-check
    /// and the entire working set was thrown away for a cold l1-elastic
    /// phase-1, whose recovery re-solve starts from `WorkingSet::cold`. A hint
    /// wrong by *one* entry cost the same as one wrong by hundreds — on a
    /// parametric MPC sweep, roughly one working-set change per constraint row
    /// (issue #428: 403 pivots where 2 were needed, and past `m ≈ max_iter` no
    /// answer at all), while the |A| − 1 entries the hint got *right* were
    /// exactly its value.
    ///
    /// The repair keeps them: the rows the pinned point violates are known, so
    /// add them to the working set at the boundary they overshot and re-pin.
    /// The result satisfies both the hint's rows and the violated ones, and
    /// the inner loop then drops whichever the multiplier signs reject — a
    /// couple of pivots instead of `m`. Nothing here relaxes a tolerance: the
    /// admission pre-check keeps its exact meaning and is simply handed a
    /// feasible point.
    ///
    /// Returns `None` — leaving the caller's original hint, hence the old
    /// elastic recovery — when the hint is not one repair is meant for:
    ///
    ///   * an *active* row is itself violated (re-pinning cannot help);
    ///   * too many rows are violated relative to the hint's active set, the
    ///     badly-wrong-hint case the pre-check was written for (a degenerate
    ///     NETLIB `gen` crossover vertex violating hundreds of inactive rows);
    ///   * the repaired pin set would exceed `n` rows, hence be necessarily
    ///     rank-deficient. A hint that already pins a full vertex therefore
    ///     needs a *drop* the repair cannot choose without a ratio test, and
    ///     keeps the old path;
    ///   * the re-pin fails to factor, or does not reach feasibility within
    ///     [`PIN_REPAIR_MAX_ROUNDS`].
    fn repair_pinned_hint(
        &mut self,
        qp: &QpProblem,
        x: &[Number],
        working: &WorkingSet,
        opts: &QpOptions,
    ) -> Option<(Vec<Number>, WorkingSet)> {
        let mut x_cur = x.to_vec();
        let mut w_cur = working.clone();

        for _ in 0..PIN_REPAIR_MAX_ROUNDS {
            let (cons, bounds) = violated_inactive(qp, &x_cur, &w_cur, opts.feas_tol)?;
            if cons.is_empty() && bounds.is_empty() {
                return Some((x_cur, w_cur));
            }
            let n_violated = cons.len() + bounds.len();
            let active_total = w_cur.active_count();
            if n_violated > (active_total / 4).max(PIN_REPAIR_MIN_ROWS)
                || active_total + n_violated > qp.n
            {
                return None;
            }
            for (i, status) in cons {
                w_cur.constraints[i] = status;
            }
            for (i, status) in bounds {
                w_cur.bounds[i] = status;
            }
            let (x_new, w_new) = self.pin_working_set(qp, &w_cur, opts).ok()?;
            x_cur = x_new;
            w_cur = w_new;
        }

        point_is_feasible(qp, &x_cur, opts.feas_tol).then_some((x_cur, w_cur))
    }

    /// How wrong the caller's working set turns out to be, measured on `qp`
    /// itself rather than predicted from it.
    ///
    /// Pins the hinted active rows and counts how many *other* rows and bounds
    /// the resulting point violates. That count is the cheapest honest answer
    /// available to "is this active set a good guess for this problem": it is a
    /// property of the hint applied to the target, so it needs no model of what
    /// changed between the two problems and no threshold on problem data — the
    /// approach gh #434 refuted when `n_eq / n` failed to discriminate.
    ///
    /// Costs one pinned-KKT factorization, which
    /// [`Self::solve_with_working_set`] already pays, so a caller that goes on
    /// to take the working-set route pays nothing extra for having asked.
    ///
    /// `None` when the pin does not take at all — an active row itself
    /// violated, or a factorization failure. That is a hint too broken to
    /// measure, which callers should read the same way as a large count.
    ///
    /// **Test-only, deliberately.** This measures cleanly and cheaply; what it
    /// does not do is answer the question the solver actually has. See
    /// `tests::hint_signal` for the sweep that declined it, and
    /// `dev-notes/issue-602-parametric-eligibility.md` for why. It stays in the
    /// tree because the next person to reach for this idea should find the
    /// instrument and the negative result, not just the idea.
    #[cfg(test)]
    pub(crate) fn hint_pin_quality(
        &mut self,
        qp: &QpProblem,
        working: &WorkingSet,
        opts: &QpOptions,
    ) -> Option<HintPinQuality> {
        let (x, w) = self.pin_working_set(qp, working, opts).ok()?;
        let (cons, bounds) = violated_inactive(qp, &x, &w, opts.feas_tol)?;
        Some(HintPinQuality {
            active: w.active_count(),
            violated: cons.len() + bounds.len(),
        })
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
            if crate::deadline::expired() {
                return Ok(time_limit_solution(qp, Some(&x), n_refactor));
            }
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
            let delta = self.factorize_with_inertia_control(kkt, &mut rhs, k as i32, qp.n, opts)?;
            n_refactor += 1;
            if crate::deadline::expired() {
                return Ok(time_limit_solution(qp, Some(&x), n_refactor));
            }

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
                        ..Default::default()
                    },
                    unbounded_ray: None,
                });
            }

            // ---- 4. Ratio test along p ----
            // First snapshot p so the in-place RHS solve doesn't
            // alias the step buffer later.
            let p: Vec<Number> = rhs[..n].to_vec();

            // §4.5 companion (gh #416): a δ-shifted direction is not
            // minimized by the unit step — see `model_step_cap`.
            let mut alpha = model_step_cap(qp.h, qp.g, &hx, &p, delta);
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

            if !alpha.is_finite() && !opts.certify_recession_ray {
                // The caller wants a point, not a verdict (gh #423): take
                // the δ-shifted proximal step and keep iterating. See
                // `QpOptions::certify_recession_ray`.
                alpha = 1.0;
            }

            if !alpha.is_finite() {
                // The model falls forever along `p` and no bound blocks:
                // a certified recession ray (same F2 certificate as
                // `solve_general`, with `pᵀHp < 0` in place of `Hp = 0`).
                return Ok(QpSolution {
                    obj: Number::NEG_INFINITY,
                    x,
                    lambda_g: Vec::new(),
                    lambda_x: vec![0.0; n],
                    working,
                    status: QpStatus::Unbounded,
                    stats: QpStats {
                        n_working_set_changes: n_changes,
                        n_refactor,
                        n_schur_updates: 0,
                        used_phase1: false,
                        time: started.elapsed(),
                        ..Default::default()
                    },
                    unbounded_ray: Some(p),
                });
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
                ..Default::default()
            },
            unbounded_ray: None,
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
        // A rank-deficient equality block — redundant / linearly dependent
        // rows, e.g. one row an exact scalar multiple of another — makes the
        // saddle KKT singular, and no §4.5 H-block shift can rescue a
        // rank-deficient *constraint* block. This path pins ALL `m` equality
        // rows in every inner-loop KKT (`assemble_equality_plus_bounds` has no
        // per-row selection), so it cannot prune the dependent rows itself.
        // Factor through `factor_pinned_primal`, which reports a recoverable
        // failure on such a block (whether the backend exhausted the inertia
        // loop or a large shift masked the singular block — see its
        // masked-rank-deficiency guard); on that signal, delegate to
        // `solve_general`, whose `cold_general_initial` + inner-loop
        // linear-independence guard prune the equalities to a maximal
        // independent subset (a dropped row is a linear combination of the
        // kept ones, hence satisfied at any constraint-consistent point) and
        // reach the exact vertex. Without this the solve surfaced to the user
        // as `InternalError` / exit 1, or churned to `MaxIter` (#313).
        let eq_rows: Vec<usize> = (0..m).collect();
        let eq_targets: Vec<Number> = eq_rows.iter().map(|&r| qp.bl[r]).collect();
        let mut x: Vec<Number> =
            match self.factor_pinned_primal(qp, &eq_rows, &eq_targets, &[], &[], opts) {
                Ok(x) => x,
                Err(ref e) if e.is_recoverable_factorization_failure() => {
                    return self.solve_general(qp, None, opts);
                }
                Err(e) => return Err(e),
            };
        let mut n_refactor: u32 = 1;
        let mut n_changes: u32 = 0;

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
            if crate::deadline::expired() {
                return Ok(time_limit_solution(qp, Some(&x), n_refactor));
            }
            let active: Vec<usize> = (0..n).filter(|&i| working.bounds[i].is_active()).collect();
            let k = active.len();

            let kkt = assemble_equality_plus_bounds(qp, &active);

            let hx = h_times_x(qp.h, &x);
            let mut rhs = vec![0.0; n + m + k];
            for (rhs_i, (hx_i, &g_i)) in rhs[..n].iter_mut().zip(hx.iter().zip(qp.g.iter())) {
                *rhs_i = -(hx_i + g_i);
            }
            // rhs[n..n+m] and rhs[n+m..n+m+k] stay zero.

            let delta =
                self.factorize_with_inertia_control(kkt, &mut rhs, (m + k) as i32, qp.n, opts)?;
            n_refactor += 1;
            if crate::deadline::expired() {
                return Ok(time_limit_solution(qp, Some(&x), n_refactor));
            }

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
                        ..Default::default()
                    },
                    unbounded_ray: None,
                });
            }

            // Ratio test along p.
            let p: Vec<Number> = rhs[..n].to_vec();
            // §4.5 companion (gh #416): a δ-shifted direction is not
            // minimized by the unit step — see `model_step_cap`.
            let mut alpha = model_step_cap(qp.h, qp.g, &hx, &p, delta);
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
            if !alpha.is_finite() && !opts.certify_recession_ray {
                // Point, not verdict (gh #423) — see
                // `QpOptions::certify_recession_ray`.
                alpha = 1.0;
            }
            if !alpha.is_finite() {
                // Nonpositive curvature along `p` with no blocking bound:
                // certified recession ray (see `solve_box_constrained`).
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
                        ..Default::default()
                    },
                    unbounded_ray: Some(p),
                });
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
                ..Default::default()
            },
            unbounded_ray: None,
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
        //
        // A rank-deficient equality block — redundant / linearly
        // dependent rows, e.g. three identical rows or one row an
        // integer combination of the others (#326) — makes the saddle
        // KKT singular, and no §4.5 H-block shift can rescue a
        // rank-deficient *constraint* block: the inertia loop exhausts
        // and reports a recoverable failure. This fast path pins ALL `m`
        // equality rows in one shot and has no per-row selection, so it
        // cannot prune the dependent rows itself. On that signal,
        // delegate to the rank-deficiency-aware `solve_general`, whose
        // `cold_general_initial` prunes the equalities to a maximal
        // independent subset (a dropped row is a linear combination of
        // the kept ones, hence satisfied at any constraint-consistent
        // point) and reaches the exact vertex. Without this the solve
        // surfaced to the user as `InternalError` / exit 1 (#326).
        let delta =
            match self.factorize_with_inertia_control(kkt, &mut rhs, qp.m as i32, qp.n, opts) {
                Ok(d) => d,
                Err(ref e) if e.is_recoverable_factorization_failure() => {
                    return self.solve_general(qp, None, opts);
                }
                Err(e) => return Err(e),
            };

        // Masked-rank-deficiency guard (companion to the one in
        // `factor_pinned_primal`). A large enough δ can grow the H
        // diagonal until the backend stops flagging the singular
        // constraint block and returns a solution instead of a failure
        // (feral masks the null direction around δ≈1e8), which would slip
        // a redundant — or worse, *inconsistent* — equality block past the
        // check above. A nonzero δ on this pinned equality KKT is the tell:
        // only then rank-reveal the rows, and if any is redundant delegate
        // to `solve_general` (same recovery as the exact-failure branch).
        // A δ > 0 with a full-rank block is a legitimate indefinite /
        // unbounded reduced-Hessian shift and falls through to the
        // recession-ray test below unchanged (#326).
        if delta > 0.0 {
            let eq_rows: Vec<usize> = (0..qp.m).collect();
            let (kept, _) = independent_active_subset(&mut self.linsol, qp, &eq_rows, &[]);
            if kept.len() < qp.m {
                return self.solve_general(qp, None, opts);
            }
        }

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
        // `certify_recession_ray = false` skips the N1 test outright (gh
        // #423): `x` here IS the δ-shifted proximal point — the exact
        // minimizer of `q(y) + ½δ‖y‖²` over `Ay = b` — which is the step
        // the caller asked for in place of the certificate.
        if delta > 0.0 && opts.certify_recession_ray {
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
                // The witness direction on this path IS the blown-up
                // iterate `x` (see the `d = x/‖x‖` argument above).
                let ray = x.clone();
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
                        ..Default::default()
                    },
                    unbounded_ray: Some(ray),
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
                ..Default::default()
            },
            unbounded_ray: None,
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
            if crate::deadline::expired() {
                return Ok(time_limit_solution(qp, Some(&x), n_refactor));
            }
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
            if crate::deadline::expired() {
                return Ok(time_limit_solution(qp, Some(&x), n_refactor));
            }

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
                // Scope note: EXPAND (Gill-Murray-Saunders-Wright
                // 1989) governs the *ratio test*, and its τ
                // primal-perturbation machinery is implemented —
                // τ-relaxed blocker selection in `select_blocker`,
                // plus the τ-growth and snap-reset below. It does
                // **not** supply a drop rule, so under
                // `AntiCyclingChoice::Expand` this choice is
                // Dantzig's steepest-violation: correct on every
                // non-cycling problem in the analytical ladder, and
                // the qpOASES default, but not cycle-free on its own.
                // The anti-stall Bland latch (`force_bland`) is what
                // bounds the pathological case.
                //
                // (This comment previously said EXPAND's perturbation
                // machinery had not landed and that `Expand` aliased
                // wholesale to steepest-violation. That was true before
                // c20 and stale after it — the aliasing is specific to
                // the drop rule, not to EXPAND as a whole.)
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
                        ..Default::default()
                    },
                    unbounded_ray: None,
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

            // §4.5 companion: a δ-shifted direction is not minimized by
            // the unit step (see `model_step_cap`), so let the ratio test
            // run out to the model's own minimizer along `p`.
            let alpha_cap = model_step_cap(qp.h, qp.g, &hx, &p, delta);

            let (mut alpha, blocker) =
                select_blocker(&candidates, opts, expand_tol, force_bland, alpha_cap);

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
            //
            // F2(b) is the negative-curvature sibling: `alpha_cap` is
            // infinite exactly when the model falls forever along `p`, and
            // `select_blocker` only returns it when nothing blocks — the
            // same recession-ray certificate with `pᵀHp < 0` in place of
            // `Hp = 0`. `ray_is_unbounded_descent` cannot see that case (it
            // demands zero curvature, correctly, since it also serves the
            // PSD paths), so it is checked separately below.
            //
            // Both are suppressed by `certify_recession_ray = false`, which
            // asks for a point rather than a verdict (gh #423); the α clamp
            // below then turns the unblocked direction into the δ-shifted
            // proximal step and the loop carries on.
            if candidates.is_empty()
                && delta > 0.0
                && opts.certify_recession_ray
                && (!alpha.is_finite() || ray_is_unbounded_descent(qp.h, qp.g, &x, &p))
            {
                let ray = p.clone();
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
                        ..Default::default()
                    },
                    unbounded_ray: Some(ray),
                });
            }

            if alpha < 0.0 {
                alpha = 0.0;
            }
            if !alpha.is_finite() {
                // The δ-shifted proximal step. Reached either when
                // `certify_recession_ray` declined the F2 return just above,
                // or — in principle unreachably, since an infinite
                // `alpha_cap` survives `select_blocker` only with an empty
                // candidate list — if a NaN ratio ever gets here; clamping
                // beats propagating a non-finite iterate.
                alpha = 1.0;
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
                ..Default::default()
            },
            unbounded_ray: None,
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
        // The prune is a *loop*, not a single retry. `independent_active_subset`
        // is a numerical rank test, and its answer depends on the shift the
        // factorization settled at — so a subset it called independent at one δ
        // can be rejected at the next. Pruning 4 equality rows to 2 and
        // factoring those 2 hit exactly that: the retry's own masked-deficiency
        // guard found only 1 of them independent, and the single-shot `?`
        // turned a solvable QP into a hard `LinearSolverFailure` for the user.
        //
        // Iterate while the subset keeps shrinking (so termination is
        // guaranteed — it is a strictly decreasing set), and if it still will
        // not factor, return `Ok(None)`. That is this function's existing
        // "fall through to elastic mode" signal, and elastic is precisely the
        // general recovery for a cold start that cannot be formed. An `Err`
        // here is the one outcome that helps nobody: the caller has a
        // perfectly good next thing to try.
        let mut rows: Vec<usize> = eq_rows.clone();
        let mut targets: Vec<Number> = eq_targets.clone();
        let (x, kept_eq): (Vec<Number>, Vec<usize>) = loop {
            match self.factor_pinned_primal(qp, &rows, &targets, &[], &[], opts) {
                Ok(x) => break (x, rows),
                Err(e) if e.is_recoverable_factorization_failure() => {
                    let (kept, _) = independent_active_subset(&mut self.linsol, qp, &rows, &[]);
                    if kept.len() >= rows.len() {
                        // Not shrinking: either genuinely full rank (so the
                        // failure is something this guard cannot repair) or the
                        // rank test disagrees with the factorization. Either
                        // way, hand it to elastic rather than to the user.
                        return Ok(None);
                    }
                    targets = kept.iter().map(|&r| qp.bl[r]).collect();
                    rows = kept;
                }
                Err(e) => return Err(e),
            }
        };
        *n_refactor += 1;

        // Row feasibility check — any violation routes the caller to
        // elastic mode.
        let ax = a_times_x(qp.a, &x, m);

        // Equality rows first. Every equality was *pinned* in the KKT
        // above, so the kept ones are satisfied by construction and
        // this costs nothing — but the rank guard may have PRUNED
        // some, and a pruned equality is only satisfied if it is both
        // linearly dependent on the kept ones *and consistent* with
        // them. Contradictory equalities (`x₀+x₁ = 1` and `x₀+x₁ = 3`)
        // are exactly the dependent-but-inconsistent case: the guard
        // prunes one, `x` satisfies the survivor, and the pruned row
        // is violated by 2.
        //
        // This loop used to `continue` past every `bl == bu` row, so
        // that violation was never seen: the caller took the returned
        // point as feasible, ran phase-2 on an infeasible iterate, and
        // reported `NumericalError` instead of routing to the elastic
        // phase-1 that would have certified the QP infeasible.
        //
        // It stayed hidden because the homotopy masked it — the path
        // reached `t = 1`, the corrector's own warm-start pre-check
        // caught the bad point, and elastic got its chance anyway. Only
        // the *seedless* cold route reaches this loop with a pruned
        // equality, and before #413 added a seeded retry nothing
        // exercised it on an infeasible model.
        for i in 0..m {
            if qp.bl[i] != qp.bu[i] {
                continue;
            }
            if (ax[i] - qp.bl[i]).abs() > opts.feas_tol {
                return Ok(None);
            }
        }

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

    /// Feasibility audit (M5) + elastic repair, applied to whatever a
    /// solve path produced.
    ///
    /// A solve that converged to a constraint-violating point and labelled
    /// it `Optimal` is a wrong answer, however it got there. Two routes
    /// reach that state: the warm-start inner loop steps with a zero-RHS
    /// active-set system, so caller-marked-active residuals are frozen and
    /// an `Inactive` equality can never enter the working set; and the
    /// cold fast paths never run that loop at all, so an inconsistent
    /// equality system passes straight through them.
    ///
    /// On violation, recover through elastic mode. `solve_elastic`
    /// recurses through `solve_general` / `solve_general_schur` *directly*,
    /// bypassing `solve`, and seeds a slack-feasible augmented problem —
    /// so the recursive solve is never re-audited and the recovery cannot
    /// loop.
    fn audit_and_repair(
        &mut self,
        qp: &QpProblem,
        sol: QpSolution,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        if !matches!(sol.status, QpStatus::Optimal) || point_is_feasible(qp, &sol.x, opts.feas_tol)
        {
            return Ok(sol);
        }
        // Never-regress on the recovery. Elastic phase-1 is a *repair* for a
        // solve that converged to a constraint-violating point, but it is not
        // guaranteed to land somewhere better, and when it does not the
        // substitution is destructive: on Maros-Meszaros `QADLITTL` (optimum
        // 480319) the audited iterate sits at 500918 with a small violation,
        // and the elastic result that replaced it was 8.07 — the elastic seed
        // (origin projected into the box), essentially no answer at all. The
        // symptom from outside was a *larger* `max_iter` producing a far worse
        // objective, because only the bigger budget got far enough to reach
        // `Optimal` and trip this audit.
        //
        // Keep whichever point is less infeasible. Elastic still wins whenever
        // it does its job — driving the slacks out — which is the case this
        // path exists for.
        let before = max_violation(qp, &sol.x);
        let repaired = self.solve_elastic(qp, opts)?;
        let after = max_violation(qp, &repaired.x);
        if after <= before {
            return Ok(repaired);
        }
        // Repair regressed feasibility: keep the audited point, but do not
        // dress it up as `Optimal` — it violates constraints, which is exactly
        // what the audit established.
        let mut kept = sol;
        kept.status = QpStatus::MaxIter;
        Ok(kept)
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
        // The caller's iteration budget is left alone. An earlier draft
        // raised it here, on the theory that passes exiting at `MaxIter`
        // near-feasible were short of iterations — the fuzz says
        // otherwise: dropping the bump changes neither the false-
        // certificate count (0) nor the certification rate (108/143). The
        // γ schedule below is what does the work. Overriding the budget
        // was also actively wrong: `sqp_qp_max_iter = 3` asks for a
        // bounded solve, and a phase-1 quietly spending 2000 is not that
        // (`sqp_qp_options_reach_the_active_set_engine` caught it).
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
        if sol_aug.status == QpStatus::TimeLimit || crate::deadline::expired() {
            return Ok(time_limit_solution(qp, Some(&x), sol_aug.stats.n_refactor));
        }

        let feasible = reform.is_feasible(&sol_aug.x, opts.feas_tol);
        if feasible {
            // Elastic drove every slack to zero ⇒ the recovered `x` is
            // feasible for the original QP, and optimal for it iff the
            // phase-1 solve *converged*: past the point where the slacks
            // vanish the augmented objective is the original one plus a
            // zero penalty, so an unconverged phase-1 leaves an ordinary
            // feasible-but-suboptimal iterate. That caveat used to be a
            // parenthetical in this comment while the code labelled the
            // point `Optimal` regardless — a claim contradicted by the
            // returned KKT residual (afiro, `sqp_qp_max_iter=3`: phase-1
            // exits `MaxIter` slack-feasible at objective 440 against a
            // −464.75 optimum, and the driver reported `Optimal` with a
            // KKT error of 10). Carry the inner verdict instead; the point
            // is still returned, just not dressed up.
            let obj = quad_objective(qp, &x);
            return Ok(QpSolution {
                x,
                lambda_g,
                lambda_x,
                working,
                obj,
                status: if sol_aug.status == QpStatus::Optimal {
                    QpStatus::Optimal
                } else {
                    QpStatus::MaxIter
                },
                stats: QpStats {
                    n_working_set_changes: sol_aug.stats.n_working_set_changes,
                    n_refactor: sol_aug.stats.n_refactor,
                    n_schur_updates: sol_aug.stats.n_schur_updates,
                    used_phase1: true,
                    time: started.elapsed(),
                    ..Default::default()
                },
                unbounded_ray: None,
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
        // Candidate seeds, cheapest-first: the recovered `x`, the
        // elastic seed `x_orig` (0 projected into the box — feasible
        // whenever the origin is, which is the exact #282 optimum), and
        // — last, because it costs a third solve — the minimum-norm
        // feasible point from a CONVEX feasibility-only phase-1.
        //
        // That third seed is what makes the certificate below sound on
        // a nonconvex QP. The elastic solve above minimizes
        // `½pᵀHp + gᵀp + γ‖v‖₁`; when `H` is indefinite — which is the
        // default for the SQP's step QP, whose `H` is the exact ∇²L —
        // an active-set method returns a *local* KKT point, so its
        // residual slacks are not the global minimal-l1 violation and
        // prove nothing. Worse, γ = 1e6 turns a slack of ~1e-7 into
        // ~0.1 of apparent objective, so the solve settles at a far box
        // vertex carrying a cancelling `(v_l, v_u)` pair rather than at
        // the small feasible step. HS071 warm-started near its own
        // solution hit exactly this: the recovered point missed
        // `feas_tol` by a factor of two (1.95e-9 against 1e-9) on a QP
        // with points feasible to slack 1.66, and the SQP reported
        // `Infeasible_Problem_Detected` at iteration 0 (gh#484 follow-up).
        let mut candidates = vec![x.clone(), x_orig.clone()];
        // The convex phase-1's point earns a place in the seed list
        // whether or not it cleared `feas_tol`: it is the closest thing
        // to a feasible point anyone here has, and phase-2 warm-started
        // from it routinely polishes the last few ulps that the proximal
        // term's bias left behind. Only `witness` — its *verified*
        // feasibility — is allowed to speak to the certificate.
        let (p1_point, p1_verdict) = self.convex_feasibility_seed(qp, opts);
        if crate::deadline::expired() {
            return Ok(time_limit_solution(
                qp,
                p1_point.as_deref().or(Some(x.as_slice())),
                0,
            ));
        }
        if let Some(seed) = p1_point {
            candidates.push(seed);
        }
        let mut have_feasible_witness = p1_verdict == Some(true);
        for seed in candidates {
            if !self.recovery_seed_usable(qp, &seed) {
                continue;
            }
            if self.original_qp_feasible(qp, &seed, opts.feas_tol) {
                have_feasible_witness = true;
            }
            // Classify the seed rather than handing the inner solve a
            // cold working set. A cold set marks every row `Inactive`,
            // *including equalities* — and the warm inner loop steps
            // with a zero-RHS active-set system, so an equality left
            // Inactive can never enter the working set and is simply
            // never enforced. That is the same M5 failure the `solve`
            // audit exists to catch, and seeding it here made this
            // recovery useless on any QP with an equality row: phase-2
            // reliably converged to `Optimal` at a point violating the
            // equality (by 7.8 on HS071's step QP, against a seed that
            // satisfied it to 1e-12), failed the feasibility check
            // below, and fell through to the certificate.
            let ws_rec = QpWarmStart {
                x: seed.clone(),
                lambda_g: vec![0.0; m],
                lambda_x: vec![0.0; n],
                working: self.working_set_at(qp, &seed, opts.feas_tol),
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
                // "This seed did not pan out" does not apply to cancellation:
                // `continue` would start the next recovery solve on a budget
                // that is already gone.
                Err(QpError::DeadlineExpired) => return Err(QpError::DeadlineExpired),
                Err(_) => continue,
            };
            if rec.status == QpStatus::TimeLimit || crate::deadline::expired() {
                return Ok(time_limit_solution(qp, Some(&rec.x), rec.stats.n_refactor));
            }
            if rec.status == QpStatus::Optimal
                && self.original_qp_feasible(qp, &rec.x, opts.feas_tol)
            {
                let mut rec = rec;
                rec.stats.used_phase1 = true;
                rec.stats.time = started.elapsed();
                return Ok(rec);
            }
        }

        // Recovery found no feasible *optimum*. Only now may we speak to
        // infeasibility, and only when both premises of the certificate
        // hold:
        //
        // 1. Phase-1 actually CONVERGED to its minimal-l1 optimum. If it
        //    stalled (MaxIter / numerical breakdown) we have no
        //    certificate; report that honest, non-committal status
        //    instead of asserting a confident `Infeasible` we cannot
        //    back up.
        // 2. No seed above was itself feasible for the original rows.
        //    Holding a feasible point while announcing infeasibility is
        //    a contradiction in terms — phase-2 failing to *optimize*
        //    from it says nothing about feasibility. Downgrade to the
        //    same non-committal status rather than certify against a
        //    witness we are carrying.
        let obj = quad_objective(qp, &x);
        let status = match sol_aug.status {
            // `p1_verdict == Some(false)` is the only thing here that can
            // carry an infeasibility claim: a *convex* phase-1 that
            // converged to a minimal infeasibility real on this data's
            // scale. The nonconvex elastic solve above cannot — its
            // residual slacks are a local artefact — and neither can
            // "phase-2 failed to improve", which is ignorance.
            QpStatus::Optimal if p1_verdict == Some(false) && !have_feasible_witness => {
                QpStatus::Infeasible
            }
            QpStatus::Optimal => QpStatus::MaxIter,
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
                ..Default::default()
            },
            // Deliberately `None` even when `status` carries an `Unbounded`
            // forwarded from phase-1: that ray lives in the *augmented*
            // (elastic) space, so it is neither dimensioned nor meaningful
            // for the original QP. A caller that needs a witness gets no
            // witness and must not claim unboundedness.
            unbounded_ray: None,
        })
    }

    /// Working set describing which rows and bounds `x` sits on:
    /// equality rows are unconditionally `Equality`, and any inequality
    /// row or variable bound within `feas_tol` of its boundary value is
    /// snapped to the side it touches.
    ///
    /// Mirrors the classification `cold_general_initial` performs on its
    /// own starting point. Marking equalities matters most: the warm
    /// inner loop cannot pull an `Inactive` equality into the working
    /// set, so a warm start that leaves one Inactive is a warm start
    /// whose equality is unenforced for the whole solve.
    fn working_set_at(&self, qp: &QpProblem, x: &[Number], feas_tol: Number) -> WorkingSet {
        let mut working = WorkingSet::cold(qp.n, qp.m);
        let ax = a_times_x(qp.a, x, qp.m);
        for (i, c) in working.constraints.iter_mut().enumerate() {
            if qp.bl[i] == qp.bu[i] {
                *c = ConsStatus::Equality;
            } else if qp.bl[i] > NLP_LOWER_BOUND_INF && (ax[i] - qp.bl[i]).abs() <= feas_tol {
                *c = ConsStatus::AtLower;
            } else if qp.bu[i] < NLP_UPPER_BOUND_INF && (ax[i] - qp.bu[i]).abs() <= feas_tol {
                *c = ConsStatus::AtUpper;
            }
        }
        for (i, status) in working.bounds.iter_mut().enumerate() {
            let l = qp.xl[i];
            let u = qp.xu[i];
            let l_finite = l > NLP_LOWER_BOUND_INF;
            let u_finite = u < NLP_UPPER_BOUND_INF;
            if l_finite && u_finite && (l - u).abs() <= feas_tol {
                *status = BoundStatus::Fixed;
            } else if l_finite && (x[i] - l).abs() <= feas_tol {
                *status = BoundStatus::AtLower;
            } else if u_finite && (x[i] - u).abs() <= feas_tol {
                *status = BoundStatus::AtUpper;
            }
        }
        working
    }

    /// What the convex feasibility phase-1 was able to establish.
    ///
    /// The distinction that matters is between "I found no feasible
    /// point" and "I proved there is none". Only the latter licenses a
    /// Farkas certificate, and it takes two things the first does not:
    /// a *converged* subproblem, and a residual large enough to mean
    /// something on this data.
    fn certify_threshold(qp: &QpProblem, feas_tol: Number) -> Number {
        // Scale-relative. An absolute residual is meaningless without the
        // magnitudes it came from: 8e-8 is enormous on data of order 1
        // and pure rounding on data of order 1e6 — and both occur here.
        let a_inf =
            qp.a.values()
                .iter()
                .map(|v| v.abs())
                .fold(1.0_f64, f64::max);
        let b_inf = qp
            .bl
            .iter()
            .chain(qp.bu.iter())
            .filter(|v| v.abs() < NLP_UPPER_BOUND_INF)
            .map(|v| v.abs())
            .fold(1.0_f64, f64::max);
        let scale = a_inf.max(b_inf);
        // One part per million of the data scale. Below that, "infeasible"
        // and "feasible up to roundoff" are not distinguishable, so the
        // honest verdict is the non-committal one. Genuinely infeasible
        // problems are not close: their minimal infeasibility sits at the
        // scale of the right-hand side that created it.
        (1e-6 * scale).max(feas_tol)
    }

    /// Largest residual a *feasible* instance can leave behind, given the
    /// proximal centre `r` and penalty `γ`.
    ///
    /// If any feasible `x̂` exists then `(x̂, 0)` is admissible for the
    /// phase-1 and costs `½‖x̂ − r‖²` with no penalty, so the optimum's
    /// residual `s*` obeys `γ·s* ≤ ½‖x̂ − r‖²`. Bounding `‖x̂ − r‖` by the
    /// box gives a number that needs no knowledge of `x̂`:
    ///
    /// ```text
    ///     s* ≤ D² / (2γ),    D² = Σ_j max(|xl_j − r_j|, |xu_j − r_j|)²
    /// ```
    ///
    /// A residual *above* this cannot be penalty bias — no feasible point
    /// exists that would have produced it — so it certifies infeasibility.
    /// A residual below it proves nothing either way, which is why the
    /// caller escalates γ (shrinking this bound) rather than concluding.
    ///
    /// A coordinate with an infinite bound has no `D` from the box. Free
    /// variables are ordinary — an SQP step QP inherits them from every
    /// unbounded NLP variable — so returning `INFINITY` there would mean
    /// *never* certifying such a QP infeasible, trading one wrong answer
    /// for a different one. Those coordinates instead take a surrogate
    /// from where the solve actually went, `max(|x_j − r_j|, 1)` with
    /// three orders of headroom. That much is engineering judgment rather
    /// than a theorem, and it is confined to exactly the coordinates
    /// where the theorem has nothing to say.
    fn penalty_bias_bound(qp: &QpProblem, x: &[Number], r: &[Number], gamma: Number) -> Number {
        if gamma <= 0.0 {
            return Number::INFINITY;
        }
        const FREE_HEADROOM: Number = 1e3;
        let mut d_sq = 0.0;
        for j in 0..qp.n {
            let (l, u) = (qp.xl[j], qp.xu[j]);
            let d = if l <= NLP_LOWER_BOUND_INF || u >= NLP_UPPER_BOUND_INF {
                (x[j] - r[j]).abs().max(1.0) * FREE_HEADROOM
            } else {
                (l - r[j]).abs().max((u - r[j]).abs())
            };
            d_sq += d * d;
            if !d_sq.is_finite() {
                return Number::INFINITY;
            }
        }
        d_sq / (2.0 * gamma)
    }

    /// Best point the *feasibility* question alone can produce, and
    /// whether it is actually feasible.
    ///
    /// Solves a convexified elastic phase-1
    ///
    /// ```text
    ///     min  ½‖x − r‖² + γ·Σ(v_l + v_u)
    ///     s.t.  bl ≤ A x + v_l − v_u ≤ bu,   xl ≤ x ≤ xu,   v ≥ 0
    /// ```
    ///
    /// — the same elastic reformulation [`Self::solve_elastic`] uses, with
    /// the caller's objective replaced by a proximal term. That
    /// substitution is the point: it drops `H` and `g`, so the subproblem
    /// is strictly convex however indefinite the caller's `H` is, and an
    /// active-set solve of a strictly convex QP reaches the global
    /// minimum. Feasibility is a property of `A`, `bl`, `bu` and the box
    /// alone, so nothing about the question being asked is lost.
    ///
    /// # Why γ is escalated
    ///
    /// The proximal term is not free: it competes with the penalty. If a
    /// feasible `x̂` exists then `(x̂, 0)` costs `½‖x̂ − r‖²`, so the
    /// optimum's residual `s*` obeys `γ·s* ≤ ½‖x̂ − r‖²`, i.e.
    ///
    /// ```text
    ///     s* ≤ ‖x̂ − r‖² / (2γ)
    /// ```
    ///
    /// With `r = 0`, the default `γ = 1e6` and a box of radius ~6, that
    /// ceiling is ~2e-5 — four orders *above* `feas_tol`. A single solve
    /// can therefore stop several 1e-6 short of feasible while being the
    /// exact optimum of what it was asked. Judging that point against
    /// `feas_tol` and concluding "infeasible" reads a penalty artefact as
    /// a Farkas certificate, which is the defect this function exists to
    /// prevent and, at a fixed γ, quietly reproduced.
    ///
    /// Re-centring `r` alone does not rescue it. The bound is quadratic
    /// in `‖x̂ − r‖`, so it collapses only if the iterate stops moving —
    /// and in the geometry that lands here it does not: successive passes
    /// travel ~1.5 along a near-feasible manifold, holding `‖x̂ − r‖`
    /// roughly constant and the residual with it (measured: 3.2e-6 →
    /// 1.8e-6 → 3.1e-7, linear at best).
    ///
    /// What does work is making γ big enough for the bound to bite, and
    /// the residual itself says how big: a residual `s` at the ceiling
    /// means `γ` is short by about `s / feas_tol`, so scaling γ by that
    /// ratio (with margin) drives the next pass under tolerance. From
    /// `s = 3.2e-6` against `feas_tol = 1e-9` that is a single step.
    ///
    /// γ is escalated rather than the proximal term shrunk, though only
    /// their ratio matters to the bound, because γ is a *linear*
    /// coefficient: it never enters the KKT factorization, only the
    /// right-hand side and the optimality test. Shrinking `H` toward zero
    /// instead would degrade the factorization and hand back the
    /// degenerate-vertex geometry this whole path exists to escape —
    /// which is also why `H = I` and not a pure LP.
    ///
    /// Returns `(x, verdict)`:
    ///
    /// * `Some(true)`  — `x` is feasible for the original rows. A witness;
    ///   it refutes any infeasibility certificate outright.
    /// * `Some(false)` — the phase-1 converged *and* its minimal
    ///   infeasibility is large on this data's scale. That is a real
    ///   certificate: the subproblem is convex, so its optimum is global.
    /// * `None`        — nothing was established. Either the phase-1 did
    ///   not converge, or it did but stopped at a residual too small to
    ///   distinguish from roundoff. Both are ignorance, not proof.
    ///
    /// The point comes back in every case and is worth having even when
    /// the verdict is `None`: it is the closest thing to a feasible point
    /// anyone here has, and it makes a good phase-2 seed.
    fn convex_feasibility_seed(
        &mut self,
        qp: &QpProblem,
        opts: &QpOptions,
    ) -> (Option<Vec<Number>>, Option<bool>) {
        let n = qp.n;
        let idx: Vec<Index> = (1..=n as Index).collect();
        let h_space = SymTMatrixSpace::new(n as Index, idx.clone(), idx);
        let mut h_id = SymTMatrix::new(h_space);
        h_id.set_values(&vec![1.0; n]);

        // Proximal centre, starting at 0 projected into the box.
        let mut r = vec![0.0; n];
        for (xi, (&l, &u)) in r.iter_mut().zip(qp.xl.iter().zip(qp.xu.iter())) {
            if l > NLP_LOWER_BOUND_INF && *xi < l {
                *xi = l;
            }
            if u < NLP_UPPER_BOUND_INF && *xi > u {
                *xi = u;
            }
        }

        // Bland's rule for the reason the caller uses it: a feasibility
        // hunt is inherently degenerate, and Bland's is the pivot rule
        // that provably terminates.
        let mut opts_p1 = opts.clone();
        opts_p1.anti_cycling = AntiCyclingChoice::Bland;

        // Four passes: one to measure the residual, one to act on it,
        // and two of slack for geometries where the first escalation
        // overshoots into a different active set. The loop exits the
        // moment a point clears `feas_tol`, and bails when a pass stops
        // improving — a residual that will not shrink under a γ two
        // orders larger is not penalty bias, and more passes cannot help.
        const MAX_PASSES: usize = 4;
        // γ enters the objective linearly, so a large value costs the
        // factorization nothing — but it still has to be a subproblem the
        // active-set solve can finish, and past ~1e10 it increasingly is
        // not. The cap is that practical ceiling, not an arithmetic one.
        const GAMMA_MAX: Number = 1e10;
        let mut gamma = opts.elastic_gamma;
        let threshold = Self::certify_threshold(qp, opts.feas_tol);
        // Best pass so far: (point, residual, did-it-converge).
        let mut best: Option<(Vec<Number>, Number, bool)> = None;
        for _ in 0..MAX_PASSES {
            // ½‖x − r‖² = ½xᵀx − rᵀx + const, so g = −r.
            let g_prox: Vec<Number> = r.iter().map(|v| -v).collect();
            let qp_feas = QpProblem {
                n,
                m: qp.m,
                h: &h_id,
                g: &g_prox,
                a: qp.a,
                bl: qp.bl,
                bu: qp.bu,
                xl: qp.xl,
                xu: qp.xu,
                hessian_inertia: HessianInertia::Psd,
            };
            let reform = crate::elastic::ElasticReformulation::build(&qp_feas, gamma);
            let qp_aug = reform.as_qp();
            let (x_aug, working_aug) = reform.initial_seed(&qp_feas, &r, opts.feas_tol);
            let ws = QpWarmStart {
                x: x_aug,
                lambda_g: vec![0.0; reform.m_aug],
                lambda_x: vec![0.0; reform.n_aug],
                working: working_aug,
            };
            // Direct inner call, as elsewhere on this path: bypasses the
            // `solve` feasibility audit so this can never re-enter elastic.
            let sol = match if opts_p1.use_schur_updates {
                self.solve_general_schur(&qp_aug, Some(&ws), &opts_p1)
            } else {
                self.solve_general(&qp_aug, Some(&ws), &opts_p1)
            } {
                Ok(sol) => sol,
                // A phase-1 that errors establishes nothing; keep whatever
                // earlier passes found and stop. Never let a best-effort
                // refutation turn into a hard error.
                Err(_) => break,
            };

            let x = sol.x[..n].to_vec();
            if sol.status == QpStatus::TimeLimit || crate::deadline::expired() {
                return (Some(x), None);
            }
            if !x.iter().all(|v| v.is_finite()) {
                break;
            }
            let viol = max_violation(qp, &x);
            let converged = sol.status == QpStatus::Optimal;
            let improved = best.as_ref().is_none_or(|(_, b, _)| viol < *b);
            if improved {
                best = Some((x.clone(), viol, converged));
            }
            if self.original_qp_feasible(qp, &x, opts.feas_tol) {
                return (Some(x), Some(true));
            }
            // A converged pass can answer the question outright, but only
            // once its residual is too big to be penalty bias — and that
            // is not a guess, it is the bound: with `(x̂, 0)` costing at
            // most `½‖x̂ − r‖²` and the box bounding `‖x̂ − r‖`, no
            // feasible instance can leave a residual above
            // `D²/(2γ)`. Above that (and above the noise floor) the
            // convex subproblem's optimum *is* the global minimal
            // infeasibility, so this is a proof.
            //
            // Stopping here matters: escalating γ past a solved problem
            // only makes it harder, and an escalated pass that then runs
            // out of iterations discards the proof. Without this branch
            // the fuzz certified 108/143 instead of 127/143. Without the
            // bias bound in the test, a feasible instance whose bias
            // (3.2e-6) merely exceeded the noise floor (2.7e-6) was
            // certified infeasible — the exact defect being fixed.
            let bias_bound = Self::penalty_bias_bound(qp, &x, &r, gamma);
            if converged && viol > threshold.max(bias_bound) {
                return (Some(x), Some(false));
            }
            if !improved {
                break;
            }
            // Everything left is the ambiguous band — converged, but at a
            // residual that could still be penalty bias. Raise γ, and aim
            // it rather than just cranking it: a bigger γ buys a smaller
            // bias bound at the cost of a harder subproblem, so overshoot
            // is not free. Two targets, whichever is larger:
            //
            //  * enough to push the *bias bound* an order below this
            //    residual, `γ ≥ 10·D²/(2s)`, which is what lets the next
            //    pass certify;
            //  * enough to push the *residual itself* under `feas_tol`,
            //    `γ ≈ γ·s/feas_tol`, which is what lets it find a witness.
            //
            // Whichever fires, the point is that both are computed from
            // measured quantities. Jumping straight to the cap instead —
            // as scaling by `s/feas_tol` alone does on an infeasible
            // instance, where the residual never drops — leaves γ at 1e14
            // and the subproblem too hard to converge, which then throws
            // away the certificate: 107/143 rather than 118/143.
            //
            // Only when the pass converged: the bound describes an
            // optimum, so a residual it explains is one the solver
            // actually reached. A pass that ran out of iterations has no
            // such story, and a larger γ makes it harder, not easier.
            if converged && viol > 0.0 && gamma < GAMMA_MAX {
                // `bias_bound` scales as 1/γ, so the γ that would put it a
                // factor of 10 under this residual is `γ·10·bias/viol`.
                let for_certificate = gamma * 10.0 * bias_bound / viol;
                let for_witness = gamma * (viol / opts.feas_tol.max(Number::MIN_POSITIVE)) * 10.0;
                let target = if for_certificate.is_finite() {
                    for_certificate.min(for_witness)
                } else {
                    for_witness
                };
                // Written as max-then-min rather than `clamp`: once γ is
                // within a decade of the cap the "at least ×10" floor
                // exceeds the ceiling, and `clamp` panics on an inverted
                // range. (It did — the fuzz caught it on the first run.)
                gamma = target.max(gamma * 10.0).min(GAMMA_MAX);
            }
            r = x;
        }

        match best {
            None => (None, None),
            Some((x, viol, converged)) => {
                let bias_bound = Self::penalty_bias_bound(qp, &x, &r, gamma);
                let verdict = if converged && viol > threshold.max(bias_bound) {
                    // Converged, on a convex subproblem, to a minimal
                    // infeasibility that is real on this data. A proof.
                    Some(false)
                } else {
                    // Either it never converged, or it stopped at a
                    // residual indistinguishable from roundoff. Neither
                    // establishes anything; say so rather than guess.
                    None
                };
                (Some(x), verdict)
            }
        }
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
    /// Reset the Schur base factor, **repairing a rank-deficient active set**
    /// rather than failing on it.
    ///
    /// At a degenerate vertex more rows can be binding than there are
    /// variables, and those extra rows are linearly dependent — an LICQ
    /// violation. The resulting active-set KKT is singular, and no §4.5 H-block
    /// shift can repair a rank-deficient *constraint* block, so the inertia
    /// loop simply exhausts and reports failure.
    ///
    /// [`Self::solve_general`] has carried this guard for a long time;
    /// `solve_general_schur` never did. That asymmetry was invisible while the
    /// Schur path was opt-in, and became the dominant failure mode the moment
    /// it was switched on for the convex active-set driver: 27 of 138
    /// Maros-Mészáros problems (`QSHARE2B`, `QSCTAP1`, …) turned into a hard
    /// `LinearSolverFailure("KKT matrix is singular (LICQ violation or
    /// rank-deficient Jacobian)")` where the refactor path had merely failed to
    /// converge.
    ///
    /// The repair is the same one the refactor path and
    /// [`Self::cold_general_initial`] use: prune the active set to a maximal
    /// linearly independent subset and deactivate the rest. A dropped row is a
    /// linear combination of the kept ones, so it stays satisfied at the
    /// current `x` and the feasible set is unchanged — only the rank deficiency
    /// is removed. Deactivating a bound does not move `x`, so the iterate stays
    /// feasible throughout.
    ///
    /// Each repair strictly shrinks the active set, so the inner loop
    /// terminates. `budget` additionally caps how many repairs one *solve* may
    /// perform: the ratio test can re-admit a pruned row on a later iteration,
    /// and without the refactor path's rank-tabu bookkeeping there is nothing
    /// here to stop a prune/re-add cycle. Exhausting the budget surfaces the
    /// original error instead of spinning.
    fn schur_reset_rank_repaired(
        &mut self,
        schur: &mut crate::schur::SchurState,
        qp: &QpProblem,
        working: &mut WorkingSet,
        opts: &QpOptions,
        n_changes: &mut u32,
        budget: &mut u32,
    ) -> Result<(), QpError> {
        loop {
            let ac = active_slot_count(working);
            match schur.reset(&mut self.linsol, qp, working, ac as i32, opts) {
                Ok(()) => return Ok(()),
                Err(e) if e.is_recoverable_factorization_failure() => {
                    if *budget == 0 {
                        return Err(e);
                    }
                    let active_cons: Vec<usize> = (0..qp.m)
                        .filter(|&i| working.constraints[i].is_active())
                        .collect();
                    let active_bounds: Vec<usize> = (0..qp.n)
                        .filter(|&i| working.bounds[i].is_active())
                        .collect();
                    let (kc, kb) = independent_active_subset(
                        &mut self.linsol,
                        qp,
                        &active_cons,
                        &active_bounds,
                    );
                    // Already full rank ⇒ the failure is not a rank deficiency
                    // this guard can repair; do not loop on it.
                    if kc.len() == active_cons.len() && kb.len() == active_bounds.len() {
                        return Err(e);
                    }
                    *budget -= 1;
                    let mut keep_c = vec![false; qp.m];
                    for &i in &kc {
                        keep_c[i] = true;
                    }
                    let mut keep_b = vec![false; qp.n];
                    for &i in &kb {
                        keep_b[i] = true;
                    }
                    for &i in &active_cons {
                        if !keep_c[i] {
                            working.constraints[i] = ConsStatus::Inactive;
                            *n_changes += 1;
                        }
                    }
                    for &i in &active_bounds {
                        if !keep_b[i] {
                            working.bounds[i] = BoundStatus::Inactive;
                            *n_changes += 1;
                        }
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

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
        // Rank repairs allowed for this solve. Generous relative to the number
        // of genuinely dependent rows a degenerate vertex carries, but finite —
        // see `schur_reset_rank_repaired` on why a cap is needed here and not
        // on the refactor path.
        let mut rank_repair_budget: u32 = (qp.n + qp.m).min(1000) as u32;

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
        self.schur_reset_rank_repaired(
            &mut schur,
            qp,
            &mut working,
            opts,
            &mut n_changes,
            &mut rank_repair_budget,
        )?;
        n_refactor += 1;

        // GMSW EXPAND τ — same semantics as in solve_general.
        let mut expand_tol = opts.expand_tol_initial;

        // ---- Null-iteration guard (numerical floor / SMW drift) ----
        //
        // An iteration that takes the full step (`α = 1`, no blocker added)
        // lands, in exact arithmetic, exactly on the minimizer of the current
        // working set — so the very next `‖p‖∞` is zero and the loop moves on
        // to the drop test. When it is *not* zero, the limit is the linear
        // algebra rather than the algorithm, and the loop repeats a literal
        // no-op until `max_iter`. Both variants were measured on
        // Maros-Mészáros `CVXQP3_S` (3650 of its 3750 iterations were such
        // no-ops):
        //
        //   * SMW drift — after 15 accumulated rank-2 updates the direction
        //     no longer lies in the active rows' null space at all
        //     (`‖A_W p‖∞ ≈ 1e-3`), so the "full Newton step" is not one.
        //     Discarding the update layer and refactoring cures this.
        //   * Noise floor — in the `γ = 1e6` elastic phase-1 the active-set
        //     KKT is solved to `‖r‖∞ ≈ 1e-9`, which is all the conditioning
        //     allows, and that leaves `‖p‖∞ ≈ 1.9e-9` permanently above the
        //     `opt_tol = 1e-9` stationarity test. No refactor helps; the
        //     iterate simply *is* stationary to attainable precision.
        //
        // So: on the first no-op, refactor. If a fresh factor still cannot
        // shrink the step, accept the iterate as stationary for this working
        // set and let the drop test run. Accepting is safe — `QpSolver::solve`
        // re-audits feasibility and the convex driver re-measures the KKT
        // error, so a point that is not really optimal is demoted, not
        // believed — and it is strictly better than spending the whole budget
        // re-deriving the same step.
        let mut prev_p_inf = Number::INFINITY;
        let mut prev_was_null_step = false;
        let mut floor_refactored = false;
        /// A genuine Newton step drives `‖p‖∞` to round-off, so anything short
        /// of halving it means the step accomplished nothing.
        const NULL_STEP_GAIN: Number = 0.5;

        let trace = std::env::var("POUNCE_QP_TRACE").is_ok();
        for _iter in 0..opts.max_iter {
            if crate::deadline::expired() {
                return Ok(time_limit_solution(qp, Some(&x), n_refactor));
            }
            let hx = h_times_x(qp.h, &x);
            let mut rhs = vec![0.0; n + m_total];
            for (rhs_i, (hx_i, &g_i)) in rhs[..n].iter_mut().zip(hx.iter().zip(qp.g.iter())) {
                *rhs_i = -(hx_i + g_i);
            }
            // A singular Schur complement is a normal event in SMW updating,
            // not a solver breakdown: the accumulated rank-2 updates can leave
            // the small dense block `S` singular while the underlying
            // active-set KKT is perfectly well conditioned. Recover the way the
            // count-based path already does — discard the update layer and
            // refactor `K_max` against the current working set — then redo the
            // solve. Nothing but the failed `S⁻¹` is thrown away, so the answer
            // is unchanged; this is what keeps the Schur path a pure
            // *performance* switch.
            //
            // Without this recovery the entire solve aborted with
            // `LinearSolverFailure("Schur block is singular …")` on problems the
            // refactor path solves exactly — the reason the Schur path could not
            // be turned on by default. See `tests/schur_vs_refactor.rs`.
            let rhs_backup = rhs.clone();
            if let Err(e) = schur.solve(&mut self.linsol, &mut rhs) {
                if !e.is_recoverable_factorization_failure() {
                    return Err(e);
                }
                self.schur_reset_rank_repaired(
                    &mut schur,
                    qp,
                    &mut working,
                    opts,
                    &mut n_changes,
                    &mut rank_repair_budget,
                )?;
                n_refactor += 1;
                // `solve` writes through `rhs`, so restore it before retrying.
                rhs.copy_from_slice(&rhs_backup);
                schur.solve(&mut self.linsol, &mut rhs)?;
            }
            if crate::deadline::expired() {
                return Ok(time_limit_solution(qp, Some(&x), n_refactor));
            }

            let p: Vec<Number> = rhs[..n].to_vec();
            let p_inf = p.iter().map(|pi| pi.abs()).fold(0.0, f64::max);

            if trace {
                // True residual of the *active-set* KKT system at (p, λ).
                // Row block 1: H p + Σ_active a_i λ_i = -(Hx + g)
                // Row block 2: a_iᵀ p = 0 for every active row / bound.
                let hp = h_times_x(qp.h, &p);
                let hxg = h_times_x(qp.h, &x);
                let mut r1: Vec<Number> = (0..n).map(|i| hp[i] + hxg[i] + qp.g[i]).collect();
                let (ir, jc, av) = (qp.a.irows(), qp.a.jcols(), qp.a.values());
                for k in 0..ir.len() {
                    let i = (ir[k] - 1) as usize;
                    let j = (jc[k] - 1) as usize;
                    if working.constraints[i].is_active() {
                        r1[j] += av[k] * rhs[n + i];
                    }
                }
                for j in 0..n {
                    if working.bounds[j].is_active() {
                        r1[j] += rhs[n + m + j];
                    }
                }
                let ap_dbg = a_times_x(qp.a, &p, m);
                let mut r2 = 0.0_f64;
                for i in 0..m {
                    if working.constraints[i].is_active() {
                        r2 = r2.max(ap_dbg[i].abs());
                    }
                }
                for j in 0..n {
                    if working.bounds[j].is_active() {
                        r2 = r2.max(p[j].abs());
                    }
                }
                let r1n = r1.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
                eprintln!(
                    "[qp] it={_iter} RES stat={r1n:.3e} actrow={r2:.3e} pinf={p_inf:.3e} sdim={} nact={}",
                    schur.n_schur_updates() * 2,
                    active_slot_count(&working)
                );
            }

            // Null-iteration guard — see the note above the loop.
            let mut stationary = p_inf <= opts.opt_tol;
            if !stationary && prev_was_null_step && p_inf > NULL_STEP_GAIN * prev_p_inf {
                if floor_refactored {
                    // A fresh factor already failed to shrink the step: this is
                    // the attainable-accuracy floor, not update drift.
                    stationary = true;
                } else {
                    floor_refactored = true;
                    self.schur_reset_rank_repaired(
                        &mut schur,
                        qp,
                        &mut working,
                        opts,
                        &mut n_changes,
                        &mut rank_repair_budget,
                    )?;
                    n_refactor += 1;
                    prev_was_null_step = false;
                    prev_p_inf = Number::INFINITY;
                    continue;
                }
            }
            prev_p_inf = p_inf;

            if stationary {
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
                    if trace {
                        eprintln!(
                            "[qp] it={_iter} DROP slot={slot} viol={:.3e} obj={:.12e} nact={} pinf={:.2e}",
                            worst.unwrap().1,
                            quad_objective(qp, &x),
                            active_slot_count(&working),
                            p_inf
                        );
                    }
                    // Degenerate rank-2 update ⇒ refactor instead. `working`
                    // already carries this drop, so resetting against it
                    // reaches exactly the state the update was meant to
                    // produce, without the update.
                    if let Err(e) = schur.apply_change(&mut self.linsol, qp, slot, false) {
                        if !e.is_recoverable_factorization_failure() {
                            return Err(e);
                        }
                        self.schur_reset_rank_repaired(
                            &mut schur,
                            qp,
                            &mut working,
                            opts,
                            &mut n_changes,
                            &mut rank_repair_budget,
                        )?;
                        n_refactor += 1;
                    }
                    n_changes += 1;
                    n_schur_updates += 1;
                    if schur.needs_reset(opts) {
                        self.schur_reset_rank_repaired(
                            &mut schur,
                            qp,
                            &mut working,
                            opts,
                            &mut n_changes,
                            &mut rank_repair_budget,
                        )?;
                        n_refactor += 1;
                    }
                    // The working set changed, so the next step is a fresh
                    // Newton step, not a repeat: re-arm the null-iteration
                    // guard.
                    prev_was_null_step = false;
                    floor_refactored = false;
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
                        ..Default::default()
                    },
                    unbounded_ray: None,
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
            // §4.5 companion (gh #416): a δ-shifted direction is not
            // minimized by the unit step — see `model_step_cap`. The shift
            // lives in the cached base factor here rather than in a local,
            // and rank-2 updates never touch the H block, so `schur.shift()`
            // is the δ that produced this `p`.
            let alpha_cap = model_step_cap(qp.h, qp.g, &hx, &p, schur.shift());

            let (mut alpha, blocker) =
                select_blocker(&candidates, opts, expand_tol, false, alpha_cap);

            // F2(a), Schur path. Same certificate as `solve_general`: an
            // empty candidate list means `+p` is feasible for every step
            // length, so a zero-curvature descent `p` is a recession ray.
            // The unconditional (un-gated on δ) check is safe because
            // `ray_is_unbounded_descent` rejects any direction with
            // measurable curvature (`‖Hp‖∞` above the 1e-10·‖H‖
            // structural-zero floor), so a PD-reduced-Hessian Newton step
            // never certifies. F2(b) — the negative-curvature sibling, an
            // infinite `alpha_cap` with nothing to block it — rides along.
            // Both are suppressed by `certify_recession_ray = false` (gh
            // #423); the α clamp below then takes the δ-shifted proximal
            // step instead.
            if candidates.is_empty()
                && opts.certify_recession_ray
                && (!alpha.is_finite() || ray_is_unbounded_descent(qp.h, qp.g, &x, &p))
            {
                let ray = p.clone();
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
                        ..Default::default()
                    },
                    unbounded_ray: Some(ray),
                });
            }

            if alpha < 0.0 {
                alpha = 0.0;
            }
            if !alpha.is_finite() {
                // The δ-shifted proximal step — reached when
                // `certify_recession_ray` declined the F2 return above, or
                // (unreachably in principle, since an infinite cap survives
                // `select_blocker` only with an empty candidate list) if a
                // NaN ratio ever gets here. Clamping beats propagating a
                // non-finite iterate.
                alpha = 1.0;
            }
            if trace && blocker.is_none() {
                eprintln!(
                    "[qp] it={_iter} NOBLOCK alpha={alpha:.3e} pinf={p_inf:.3e} obj={:.12e} nact={} ncand={}",
                    quad_objective(qp, &x),
                    active_slot_count(&working),
                    candidates.len()
                );
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
                if trace {
                    eprintln!(
                        "[qp] it={_iter} ADD  slot={slot} alpha={alpha:.3e} obj={:.12e} nact={} pinf={:.2e}",
                        quad_objective(qp, &x),
                        active_slot_count(&working),
                        p_inf
                    );
                }
                // Same recovery as the drop side: `working` already carries this
                // add, so a reset against it reproduces the intended state.
                if let Err(e) = schur.apply_change(&mut self.linsol, qp, slot, true) {
                    if !e.is_recoverable_factorization_failure() {
                        return Err(e);
                    }
                    self.schur_reset_rank_repaired(
                        &mut schur,
                        qp,
                        &mut working,
                        opts,
                        &mut n_changes,
                        &mut rank_repair_budget,
                    )?;
                    n_refactor += 1;
                }
                n_changes += 1;
                n_schur_updates += 1;
                if schur.needs_reset(opts) {
                    self.schur_reset_rank_repaired(
                        &mut schur,
                        qp,
                        &mut working,
                        opts,
                        &mut n_changes,
                        &mut rank_repair_budget,
                    )?;
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

            // Null-iteration bookkeeping. A blocker means the working set grew,
            // so the next step solves a *different* system and the guard
            // re-arms; no blocker means a full step was taken and the next
            // `‖p‖∞` must be round-off if the linear algebra is sound.
            if blocker.is_some() {
                prev_was_null_step = false;
                floor_refactored = false;
            } else {
                prev_was_null_step = true;
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
                ..Default::default()
            },
            unbounded_ray: None,
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
pub(crate) fn independent_active_subset(
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
/// Returns `(α, blocker)` with `α = alpha_cap` and `blocker = None`
/// when no direction blocks at less than the full step.
///
/// `alpha_cap` is the unconstrained step length the caller wants —
/// the minimizer of the QP model along `p`, which is 1.0 for an
/// unshifted Newton direction and larger (possibly `+∞`) for a
/// δ-shifted one; see [`model_step_cap`]. It is the value returned
/// when nothing blocks, and the ceiling on every value that is.
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
    alpha_cap: f64,
) -> (f64, Option<BlockerTarget>) {
    if candidates.is_empty() {
        return (alpha_cap, None);
    }
    // Pass 1: minimum ratio (strict and τ-relaxed).
    let mut alpha_min = alpha_cap;
    let mut alpha_min_relaxed = alpha_cap;
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
    if alpha_min >= alpha_cap {
        return (alpha_cap, None);
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
                    // we never freeze at α = 0; cap at the model minimizer.
                    let alpha = r.max(alpha_min_relaxed).min(alpha_cap).max(0.0);
                    (alpha, Some(target))
                }
                None => {
                    // Pass 2 admitted nothing. This happens when every
                    // candidate's τ-relaxed ratio exceeds the artificial
                    // `α_min_relaxed = alpha_cap` initialization by more than
                    // `tol` — reachable when |a·p| ≈ feas_tol makes
                    // `τ/|a·p|` inflate `r_relaxed` above `alpha_cap + tol`
                    // for ALL candidates (so the recorded minimum is the cap,
                    // which no real candidate attains). Fall back to the
                    // strict minimum-ratio blocker (guaranteed to exist since
                    // `α_min < alpha_cap`) and step exactly `α_min`: never
                    // freeze, panic, or overstep the first blocking
                    // constraint.
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
        let _deadline = crate::deadline::enter(opts.time_limit);
        // The second-order pass sits *here*, outside `solve_scoped`, for two
        // reasons. It is the one place every route through the engine — box,
        // equality-only, equality-plus-bounds, general, Schur, elastic,
        // homotopy — is guaranteed to pass through exactly once, so no inner
        // loop has to remember it. And its own re-solves call `solve_scoped`
        // directly, which is what stops an escape from recursing into another
        // escape (gh #848).
        let out = self
            .solve_scoped(qp, ws, opts)
            .and_then(|sol| self.escape_negative_curvature(qp, sol, opts));
        soften_deadline(qp, ws.map(|w| w.x.as_slice()), out)
    }

    fn solve_parametric(
        &mut self,
        qp_prev: &QpProblem,
        sol_prev: &QpSolution,
        qp_new: &QpProblem,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        let _deadline = crate::deadline::enter(opts.time_limit);
        let out = self.solve_parametric_scoped(qp_prev, sol_prev, qp_new, opts);
        let mut out = soften_deadline(qp_new, Some(&sol_prev.x), out);
        // Every route through `solve_parametric_scoped` stamps its own source,
        // so the only way to arrive here unstamped is the deadline stub
        // `soften_deadline` just substituted for a cancelled solve. Nothing was
        // reused on that path either, which is what `Cold` says. Stamping it
        // here rather than leaving `None` keeps the field's contract exact:
        // `None` means "not a parametric call", never "a parametric call whose
        // route went unrecorded".
        if let Ok(sol) = &mut out
            && sol.stats.parametric_source.is_none()
        {
            sol.stats.parametric_source = Some(ParametricSource::Cold);
        }
        out
    }

    fn solve_with_working_set(
        &mut self,
        qp: &QpProblem,
        working: &crate::working_set::WorkingSet,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        let _deadline = crate::deadline::enter(opts.time_limit);
        let out = self.solve_with_working_set_scoped(qp, working, opts);
        soften_deadline(qp, None, out)
    }
}

/// Entry-point boundary for the internal cancellation error.
///
/// [`QpError::DeadlineExpired`] exists so `?` propagation forces every
/// internal caller to handle a timeout instead of consuming a half-finished
/// result. It is not part of the crate's contract with its callers, though:
/// a timeout is a *soft* outcome, reported as `QpStatus::TimeLimit` on an
/// `Ok` solution exactly like `MaxIter`. Every public entry point converts it
/// here, so the error can never escape.
/// Record which route a `solve_parametric` call took on the solution it is
/// about to return.
///
/// Applied at each of the three exits rather than inferred by the caller: the
/// guards that choose between them live here, `solve_homotopy` can decline the
/// path after they pass, and a caller re-deriving any of that would be keeping
/// a second copy of this function's control flow (gh #769).
fn stamp(mut sol: QpSolution, source: ParametricSource) -> QpSolution {
    sol.stats.parametric_source = Some(source);
    sol
}

fn soften_deadline(
    qp: &QpProblem,
    hint: Option<&[Number]>,
    out: Result<QpSolution, QpError>,
) -> Result<QpSolution, QpError> {
    match out {
        // `n_refactor = 0`: the cancelled inner solve's counter died with the
        // `Err`, and inventing a number here would be worse than reporting
        // none. `stats.time` still reflects the real budget spent.
        Err(QpError::DeadlineExpired) => Ok(time_limit_solution(qp, hint, 0)),
        other => other,
    }
}

impl ParametricActiveSetSolver {
    fn solve_scoped(
        &mut self,
        qp: &QpProblem,
        ws: Option<&QpWarmStart>,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        qp.validate()?;
        if crate::deadline::expired() {
            return Ok(time_limit_solution(qp, ws.map(|w| w.x.as_slice()), 0));
        }
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

        // Cold + general rows: try the parametric homotopy first. It returns
        // `Ok(None)` when the path cannot be started (no rows, or the box
        // relaxation is unbounded), which is a fall-through signal rather than a
        // verdict, so the conventional path below still handles everything it
        // handled before.
        if ws.is_none() && opts.use_homotopy {
            if let Some(sol) = self.solve_homotopy(qp, None, opts)? {
                return Ok(sol);
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
            return self.audit_and_repair(qp, sol, opts);
        }

        // Cold-start fast paths for problems with no general
        // inequalities and no warm-start.
        //
        // Audited too. These return a point without ever consulting the
        // active-set loop, so nothing else checks them — and an
        // inconsistent equality system is exactly what they cannot see:
        // the smallest possible infeasible QP, `aᵀx = c₁` and `aᵀx = c₂`
        // with `c₁ ≠ c₂` and a box, is all-equality with bounds, lands in
        // `solve_equality_plus_bounds`, and came back `Optimal` at a point
        // violating both rows by 2.9. Every tolerance, every version.
        let sol = if is_pure_equality_no_bounds(qp) {
            self.solve_equality_only(qp, opts)?
        } else if is_pure_box(qp) {
            self.solve_box_constrained(qp, opts)?
        } else {
            self.solve_equality_plus_bounds(qp, opts)?
        };
        self.audit_and_repair(qp, sol, opts)
    }

    fn solve_parametric_scoped(
        &mut self,
        qp_prev: &QpProblem,
        sol_prev: &QpSolution,
        qp_new: &QpProblem,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        if crate::deadline::expired() {
            return Ok(stamp(
                time_limit_solution(qp_new, Some(&sol_prev.x), 0),
                ParametricSource::Cold,
            ));
        }
        // Trace the homotopy from the previous problem to the new one, starting
        // from the previous solution's working set.
        //
        // This is the crate's advertised feature and was a stub that discarded
        // both prior arguments. It is the *easy* direction of the homotopy: the
        // prior solution is already optimal for the prior QP, so the path starts
        // on the solution manifold — there is no `QP_0` to construct and no box
        // relaxation to bound, which is the whole difficulty of the cold case.
        //
        // Guards, in order: the two problems must have the same shape, and the
        // same `H`. `H` is not interpolated along the path (only `g` and the row
        // bounds are), so a changed Hessian would make the traced path solve a
        // different problem than the one requested. Rather than silently
        // mispredict, fall back (see below) — correct, just not warm.
        //
        // `A`, `xl`/`xu` and `hessian_inertia` are **deliberately not** guarded
        // on, though the path does not model them either. gh #602 proposed
        // adding them and the measurement declined it; do not re-add without
        // reading `dev-notes/issue-602-parametric-eligibility.md`.
        //
        // The short version: it is not that tracing an unmodelled change is
        // harmless, it is that declining is not reliably better. Rejecting sends
        // the call to the working-set fallback, and which of the two wins swings
        // with problem size on one synthetic family — at `n = 30` the guard is
        // better or equal in 14 of 14 rows, at `n = 20` it is worse in 9 of 14,
        // by as much as 2 working-set changes against 34. That is #434's
        // situation exactly: a rule that fires on the losses and the gains alike.
        //
        // `hessian_inertia` is the clearest case against, because it has no
        // upside at all: the tracer never reads it and neither does
        // `factorize_with_inertia_control`, so declining on it can only cost —
        // measured at 2 working-set changes becoming 5, for nothing.
        //
        // What would settle it is the discriminator #434 also wanted and did not
        // find: something observable at runtime that says whether the previous
        // active set is a good guess for this problem.
        // Bound *topology* — which rows are equalities, which variables are
        // fixed — is guarded on, and this is a correctness guard rather than a
        // cost one, which is why it stands where the `A` / box guards were
        // declined.
        //
        // `ConsStatus::Equality` and `BoundStatus::Fixed` are claims about the
        // problem, and no drop test can retract either. The tracer starts from
        // `sol_prev.working` and cannot re-type a row mid-path: the row type
        // does not interpolate, so a row that is an equality at `t = 0` and a
        // range at every `t > 0` stays marked `Equality` the whole way and is
        // handed to the corrector still claiming it. That pins it to
        // `qp_new.bl[i]` — the `-1e20` sentinel when the new lower bound is
        // infinite — at a point the feasibility audit accepts, and the solve
        // reports `Optimal`. Measured: `min ½x² s.t. x == 1` re-solved as
        // `min x² s.t. x ≤ 2` returned `Optimal` at `x = -1e19`, and the same
        // through `Fixed` when a pinned variable is freed (gh #602, found in
        // review of #614).
        //
        // Declining sends the pair to the fallback below, which runs the hint
        // through `WorkingSet::reconciled_with` and so cannot make that claim.
        // A genuine parametric family does not trip this: an equality row stays
        // an equality across a sweep, and a fixed variable stays fixed.
        let same_topology = qp_prev.m == qp_new.m
            && qp_prev.n == qp_new.n
            && (0..qp_prev.m)
                .all(|i| (qp_prev.bl[i] == qp_prev.bu[i]) == (qp_new.bl[i] == qp_new.bu[i]))
            && (0..qp_prev.n).all(|j| {
                let fixed = |xl: Number, xu: Number| {
                    xl > NLP_LOWER_BOUND_INF
                        && xu < NLP_UPPER_BOUND_INF
                        && (xl - xu).abs() <= opts.feas_tol
                };
                fixed(qp_prev.xl[j], qp_prev.xu[j]) == fixed(qp_new.xl[j], qp_new.xu[j])
            });

        let same_shape = qp_prev.n == qp_new.n && qp_prev.m == qp_new.m;
        let same_h = qp_prev.h.nonzeros() == qp_new.h.nonzeros()
            && qp_prev.h.values() == qp_new.h.values()
            && qp_prev.h.irows() == qp_new.h.irows()
            && qp_prev.h.jcols() == qp_new.h.jcols();

        if same_shape
            && same_h
            && same_topology
            && sol_prev.status == QpStatus::Optimal
            && sol_prev.x.len() == qp_new.n
            && let Some(sol) = self.solve_homotopy(qp_new, Some((qp_prev, sol_prev)), opts)?
        {
            return Ok(stamp(sol, ParametricSource::Homotopy));
        }

        // The path did not run — the guard declined it, or the tracer returned
        // `Ok(None)`. Neither is a reason to throw away the *working set* the
        // caller just handed us, which is what a cold solve here does.
        //
        // The primal genuinely does not carry over (that is why there is a
        // homotopy at all), but the discrete state does: `solve_with_working_set`
        // pins a fresh primal satisfying the hinted active rows, repairs the pin
        // if some other row is violated (#428), and only then runs the
        // conventional loop. So the hint costs one pinned-KKT factorization and
        // is worth having even when it is stale — which is exactly the SQP
        // driver's standing bet (`sqp_alg.rs` warm-starts this way on every
        // iteration, because each linearization moves `A` and translates the row
        // bounds by `-c(x_k)`).
        //
        // Measured on the synthetic family in
        // `examples/parametric_eligibility_sweep.rs`, over the rejected pairs
        // (`H` perturbed, so `same_h` is false and this branch is the whole
        // behaviour of `solve_parametric`):
        //
        // | H perturbation | cold (was) | working-set hint (now) |
        // |---|---|---|
        // | 1%  | 18 changes | **3** |
        // | 10% | 18 changes | **3** |
        // | 50% | 17 changes | **2** |
        //
        // The hint survives a 50% Hessian perturbation because what it encodes
        // is which constraints bind, and that is far more stable under a change
        // of `H` than the iterate is. See gh #602 and
        // `dev-notes/issue-602-parametric-eligibility.md`.
        //
        // Conditions. The working set must be dimensionally valid for the new
        // problem, or `solve_with_working_set` rejects it with
        // `WarmStartDimensionMismatch` — a hard `Err` out of a call that has a
        // perfectly good cold answer available, which would turn a shape change
        // from "warm start unavailable" into "solve failed". And the previous
        // solve must have reached `Optimal`: a `TimeLimit` result carries
        // `WorkingSet::cold` (all-inactive, i.e. no information) and a `MaxIter`
        // one carries a set that was still moving, neither of which the
        // measurement above covers.
        //
        // `reconciled_with` is not optional. Dimensional validity is not enough
        // to make a working set *meaningful* for another problem: `Equality`
        // and `Fixed` assert `bl == bu` / `xl == xu` about the problem they came
        // from, and the solver never drops either, so carrying one onto a
        // problem where the row is an inequality pins it to a bound that does
        // not exist and reports the result `Optimal`. That is a wrong answer,
        // not a slow one — see `WorkingSet::reconciled_with`, and the two
        // regression tests it names.
        if sol_prev.status == QpStatus::Optimal
            && sol_prev.working.validate_dims(qp_new.n, qp_new.m).is_ok()
        {
            let hint = sol_prev.working.reconciled_with(qp_new, opts);
            return self
                .solve_with_working_set(qp_new, &hint, opts)
                .map(|sol| stamp(sol, ParametricSource::WorkingSet));
        }
        self.solve(qp_new, None, opts)
            .map(|sol| stamp(sol, ParametricSource::Cold))
    }

    fn solve_with_working_set_scoped(
        &mut self,
        qp: &QpProblem,
        working: &crate::working_set::WorkingSet,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        qp.validate()?;
        working.validate_dims(qp.n, qp.m)?;
        if crate::deadline::expired() {
            return Ok(time_limit_solution(qp, None, 0));
        }

        // Make the hint well-formed for `qp` before anything trusts it.
        //
        // This entry point is public API and the hint is caller-supplied, so
        // `Equality` / `Fixed` can arrive on a problem where the row is a range
        // or the variable free. Neither status is droppable, so the pin lands on
        // a bound `qp` does not have — the `-1e20` sentinel — and the feasibility
        // audit accepts it, because such a point genuinely is feasible. The
        // result is `Optimal` at a wrong answer, and no check downstream of here
        // can catch it: the point is optimal for the problem the working set
        // describes, and it is the working set that describes the wrong problem
        // (gh #602, raised in review of #614).
        //
        // It is *close* to a no-op for a well-formed hint — it re-derives
        // statuses that already agree with `qp` — but not exactly one, and the
        // measurement is the honest version of that claim rather than the
        // convenient one. On `benchmarks/warmstart` the conventional arms are
        // bit-identical and the homotopy arms improve (`cold-sqp-hom`
        // 28487 -> 27828, `warm-sqp-hom` 2005 -> 1755), solved counts unchanged.
        // On the CLI fixture sweep five lines move, all on fixtures that were
        // already failing; `jit1` on the homotopy arm goes from
        // `MaximumIterationsExceeded` to a converged solve (dual inf 9.3e-9,
        // constraint violation 6.5e-19). The residual difference comes from
        // hints where a variable's bounds sit within `feas_tol` of each other
        // without being equal, which this promotes to `Fixed`.
        let working = &working.reconciled_with(qp, opts);

        // Factor the pinned KKT for a primal that satisfies the hinted active
        // rows (pruning the hint first if it is rank-deficient).
        let (x_init, fwd_working) = self.pin_working_set(qp, working, opts)?;

        // A pinned primal that is infeasible for some *other* row is the
        // signature of an active set that has moved since the hint was
        // recorded. `solve`'s admission pre-check would drop the hint whole
        // and fall back to a cold l1-elastic phase-1, spending about one
        // working-set change per constraint row to rebuild what the hint
        // already had right. Repair it instead — pin the violated rows too —
        // and hand the pre-check a feasible point (#428). A hint the repair
        // cannot rescue keeps the old path untouched.
        let (x_init, fwd_working) = if point_is_feasible(qp, &x_init, opts.feas_tol) {
            (x_init, fwd_working)
        } else {
            self.repair_pinned_hint(qp, &x_init, &fwd_working, opts)
                .unwrap_or((x_init, fwd_working))
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

    /// Refuse to certify a saddle point as a solution, and get off it when
    /// there is somewhere to go (gh #848).
    ///
    /// Runs after the engine has produced its verdict, at the choke point
    /// every public entry funnels through, so no individual exit in the
    /// engine's five inner loops has to remember to call it. Only
    /// `QpStatus::Optimal` is examined: that is the one status whose meaning
    /// is a *claim about the model* rather than about the solve, and the only
    /// one a second-order finding can falsify.
    ///
    /// The loop is the classical nonconvex active-set move (Nocedal-Wright
    /// §16.4, "if the reduced Hessian is indefinite … a direction of negative
    /// curvature is followed to a new constraint"): at a first-order point
    /// with a witness `d` satisfying `A_W d = 0` and `dᵀHd < 0`,
    ///
    /// * `∇q(x)ᵀd = 0`, because at a first-order point the gradient is a
    ///   combination of the working set's rows and `d` is orthogonal to all of
    ///   them. So `q(x + αd) = q(x) + ½α²·dᵀHd` **decreases in both
    ///   directions**. Neither sign is privileged, and the one taken is
    ///   whichever the feasible set lets travel further — a longer step is
    ///   strictly more objective decrease, since the decrease is quadratic in
    ///   `α` with no linear term to trade against.
    /// * If nothing blocks either sign, `x + αd` is feasible for every `α` and
    ///   the objective falls without bound: a certified recession ray, which
    ///   is the `dᵀPd < 0` branch of `pounce-convex`'s `ray_certifies_unbounded`
    ///   (gh #791) — a branch that had no reachable producer before this.
    ///   That exit alone answers to
    ///   [`QpOptions::certify_recession_ray`](crate::QpOptions::certify_recession_ray):
    ///   a caller that has declined recession verdicts keeps its point and
    ///   its `Optimal`, and reads the refutation off `stats.second_order`.
    /// * Otherwise the step ends on a new row or bound, which joins the
    ///   working set, and the solve resumes from there.
    ///
    /// The resume is a full solve, so it may drop what the escape pinned and
    /// return to where it started; the loop therefore terminates on *measured
    /// objective progress*, not on the working set growing. Without that it
    /// spins the budget on one fixed point — see the guard below, and the
    /// HS071 measurement in it.
    ///
    /// Exhausting the budget, or stalling, with a live witness downgrades to
    /// `QpStatus::MaxIter`. It must not stay `Optimal`: that would be the
    /// original defect with extra steps. `MaxIter` here is the honest reading
    /// — the engine refuted the point and did not reach one it can certify.
    fn escape_negative_curvature(
        &mut self,
        qp: &QpProblem<'_>,
        mut sol: QpSolution,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        if !opts.certify_second_order
            || qp.hessian_inertia == HessianInertia::Psd
            || sol.status != QpStatus::Optimal
        {
            return Ok(sol);
        }

        for _ in 0..opts.neg_curv_max_escapes {
            let d = match self.second_order_verdict(qp, &sol.working, opts)? {
                SecondOrder::Certified => {
                    sol.stats.second_order = SecondOrderVerdict::Certified;
                    return Ok(sol);
                }
                SecondOrder::NotChecked => return Ok(sol),
                SecondOrder::NegativeCurvature(d) => d,
            };

            // Both signs descend; take the one with more room. `partial_cmp`
            // is not needed — an infinite α on either side is unboundedness
            // and is caught before the comparison matters.
            let neg: Vec<Number> = d.iter().map(|v| -v).collect();
            let fwd = feasible_step_along(qp, &sol.x, &d, &sol.working, opts.feas_tol);
            let bwd = feasible_step_along(qp, &sol.x, &neg, &sol.working, opts.feas_tol);
            let (dir, alpha, blocker) = if bwd.0 > fwd.0 {
                (neg, bwd.0, bwd.1)
            } else {
                (d, fwd.0, fwd.1)
            };

            if !alpha.is_finite() {
                if !opts.certify_recession_ray {
                    // The caller wants a point, not a verdict (gh #423) —
                    // the same opt-out the box path honours at its own
                    // unblocked-negative-curvature exit, and it has to be
                    // honoured here for the same reason. The SQP's
                    // unbounded-model fallback sets this flag and re-solves
                    // precisely because the *unblocked* case is a statement
                    // about the linearization: the δ-shifted proximal step
                    // is a real step and certifying recession instead leaves
                    // the outer loop with none at all. Without this arm the
                    // fallback re-solve comes straight back `Unbounded`,
                    // `sol` never becomes `Optimal`, and the SQP exits
                    // `QpStepFailed` at iteration 1 — which is gh #419
                    // verbatim, reached through a door gh #423 did not
                    // close. Measured on `eigenb2` (110 free variables, 55
                    // equalities, nothing that can ever block a direction):
                    // 200 iterations at f = 1.6013 became 1 iteration at
                    // f = 24.026.
                    //
                    // The finding is still reported. It is the *action* that
                    // is declined, not the fact, and a caller that reads
                    // `stats.second_order` (`pounce-convex`'s
                    // `verify_status`) still sees the point refuted. Only
                    // the unblocked branch is gated: a blocked escape ends
                    // on a new row with a strictly lower objective and no
                    // certificate is involved, so it runs either way.
                    sol.stats.second_order = SecondOrderVerdict::NegativeCurvature;
                    return Ok(sol);
                }
                // Feasible for every step length along a direction the
                // objective curves *down* along: the QP is unbounded below,
                // and `dir` is the witness. `obj` follows the convention the
                // engine's other unbounded exits use.
                sol.obj = Number::NEG_INFINITY;
                sol.status = QpStatus::Unbounded;
                sol.stats.second_order = SecondOrderVerdict::NegativeCurvature;
                sol.unbounded_ray = Some(dir);
                return Ok(sol);
            }

            // Step to the blocker and pin it. A degenerate `α = 0` still
            // makes progress: the working set grows, so the next probe sees a
            // strictly smaller null space.
            let mut x = sol.x.clone();
            for (xi, di) in x.iter_mut().zip(dir.iter()) {
                *xi += alpha * di;
            }
            let mut working = sol.working.clone();
            match blocker {
                Some(Blocker::Bound(i, status)) => {
                    // Snap to the bound rather than trusting `α·dᵢ`, the same
                    // drift guard the box path applies to its own ratio test.
                    match status {
                        BoundStatus::AtLower => x[i] = qp.xl[i],
                        BoundStatus::AtUpper => x[i] = qp.xu[i],
                        _ => {}
                    }
                    working.bounds[i] = status;
                }
                Some(Blocker::Cons(i, status)) => working.constraints[i] = status,
                // `α` finite with no blocker is not reachable: `α` starts at
                // infinity and only a blocker lowers it.
                None => return Ok(sol),
            }

            if crate::deadline::expired() {
                // Keep the point — it is feasible and its objective is no
                // worse than the one we arrived with — but not the status.
                // Returning `Optimal` here is exactly the claim the witness
                // just refuted.
                sol.x = x;
                sol.obj = quad_objective(qp, &sol.x);
                sol.working = working;
                sol.status = QpStatus::TimeLimit;
                sol.stats.second_order = SecondOrderVerdict::NegativeCurvature;
                return Ok(sol);
            }

            // Resume from the escaped point. `solve_scoped`, not `solve`:
            // this call must not re-enter the escape, and the deadline scope
            // is already open.
            let escaped_obj = quad_objective(qp, &x);
            let prev_obj = sol.obj;
            let escaped = (x.clone(), working.clone(), escaped_obj);
            let ws = QpWarmStart {
                x,
                lambda_g: vec![0.0; qp.m],
                lambda_x: vec![0.0; qp.n],
                working,
            };
            let stats_so_far = sol.stats.clone();
            let mut next = self.solve_scoped(qp, Some(&ws), opts)?;
            next.stats.n_working_set_changes += stats_so_far.n_working_set_changes + 1;
            next.stats.n_refactor += stats_so_far.n_refactor;
            next.stats.n_schur_updates += stats_so_far.n_schur_updates;
            next.stats.used_phase1 |= stats_so_far.used_phase1;
            sol = next;

            if sol.status != QpStatus::Optimal {
                // The re-solve reached a conclusion of its own — unbounded,
                // out of iterations, out of time. It is not a first-order
                // point being passed off as an optimum, so there is nothing
                // left for the second-order test to falsify.
                return Ok(sol);
            }

            // The escape must not be undone by the solve that follows it.
            //
            // The loop's termination argument is that the working set grows by
            // one per escape, so a run ends within `n + m` of them. The resume
            // is free to *drop* rows again, and on an indefinite `H` it does
            // more than that: the inner loop's steps come from the δ-shifted
            // KKT of §4.5, whose model has the saddle as its *minimum*, so the
            // re-solve walks back uphill to the very point the escape left.
            // That is the same attraction gh #848 reports from the outside
            // ("the start point is ignored entirely — all three starts land on
            // `[0, 0]`"), met here from the inside, and left alone it spins the
            // whole budget on one fixed point: measured on HS071's first step
            // QP, all 20 escapes reported an identical working set, an
            // identical direction, `alpha = 1.4935` and `obj = -1.4116e-7`,
            // having each stepped to `obj = -4.52e-2` and been walked back.
            //
            // So require strict progress, and when there is none stop at the
            // better of the two points rather than re-deriving it 19 more
            // times. The status is not `Optimal` either way — the witness
            // refuted that, and getting off the saddle is what failed here,
            // not the finding.
            if sol.obj >= prev_obj - opts.opt_tol * (1.0 + prev_obj.abs()) {
                let (x_esc, working_esc, obj_esc) = escaped;
                if obj_esc < sol.obj {
                    sol.x = x_esc;
                    sol.obj = obj_esc;
                    sol.working = working_esc;
                }
                sol.status = QpStatus::MaxIter;
                sol.stats.second_order = SecondOrderVerdict::NegativeCurvature;
                return Ok(sol);
            }
        }

        // Out of escapes with the point still un-certified. Report the budget.
        sol.status = QpStatus::MaxIter;
        sol.stats.second_order = SecondOrderVerdict::NegativeCurvature;
        Ok(sol)
    }
}

/// What stopped a step along a negative-curvature direction.
#[derive(Debug, Clone, Copy)]
enum Blocker {
    Bound(usize, BoundStatus),
    Cons(usize, ConsStatus),
}

/// The largest `α ≥ 0` keeping `x + α d` feasible, and what stops it.
///
/// `INFINITY` with `None` means nothing blocks: `d` is a recession direction
/// of the feasible set. Rows and bounds already in `working` are skipped —
/// the witness satisfies `A_W d = 0`, so they neither block nor move, and
/// including them would let floating-point residual in `A_W d` manufacture a
/// spurious zero step.
fn feasible_step_along(
    qp: &QpProblem<'_>,
    x: &[Number],
    d: &[Number],
    working: &WorkingSet,
    feas_tol: Number,
) -> (Number, Option<Blocker>) {
    let mut alpha = Number::INFINITY;
    let mut blocker = None;
    let take = |r: Number, b: Blocker, alpha: &mut Number, blocker: &mut Option<Blocker>| {
        let r = if r.is_finite() { r.max(0.0) } else { r };
        if r < *alpha {
            *alpha = r;
            *blocker = Some(b);
        }
    };

    for i in 0..qp.n {
        if working.bounds[i].is_active() {
            continue;
        }
        if d[i] < -feas_tol && qp.xl[i] > NLP_LOWER_BOUND_INF {
            let r = (x[i] - qp.xl[i]) / -d[i];
            take(
                r,
                Blocker::Bound(i, BoundStatus::AtLower),
                &mut alpha,
                &mut blocker,
            );
        }
        if d[i] > feas_tol && qp.xu[i] < NLP_UPPER_BOUND_INF {
            let r = (qp.xu[i] - x[i]) / d[i];
            take(
                r,
                Blocker::Bound(i, BoundStatus::AtUpper),
                &mut alpha,
                &mut blocker,
            );
        }
    }

    if qp.m > 0 {
        let ax = a_times_x(qp.a, x, qp.m);
        let ad = a_times_x(qp.a, d, qp.m);
        for i in 0..qp.m {
            if working.constraints[i].is_active() {
                continue;
            }
            if ad[i] < -feas_tol && qp.bl[i] > NLP_LOWER_BOUND_INF {
                let r = (ax[i] - qp.bl[i]) / -ad[i];
                let status = if qp.bl[i] == qp.bu[i] {
                    ConsStatus::Equality
                } else {
                    ConsStatus::AtLower
                };
                take(r, Blocker::Cons(i, status), &mut alpha, &mut blocker);
            }
            if ad[i] > feas_tol && qp.bu[i] < NLP_UPPER_BOUND_INF {
                let r = (qp.bu[i] - ax[i]) / ad[i];
                let status = if qp.bl[i] == qp.bu[i] {
                    ConsStatus::Equality
                } else {
                    ConsStatus::AtUpper
                };
                take(r, Blocker::Cons(i, status), &mut alpha, &mut blocker);
            }
        }
    }

    (alpha, blocker)
}

/// The soft outcome for a cancelled solve: the best point we have, clamped
/// into the box so it is at least a usable starting iterate, with
/// `QpStatus::TimeLimit`.
///
/// `n_refactor` is the work done before the deadline hit, and `time` comes
/// from the enclosing deadline scope — a solve that spent the whole budget
/// must not be recorded as having taken no time.
fn time_limit_solution(qp: &QpProblem, hint: Option<&[Number]>, n_refactor: u32) -> QpSolution {
    let mut x = hint.map_or_else(|| vec![0.0; qp.n], ToOwned::to_owned);
    x.resize(qp.n, 0.0);
    for (xi, (&l, &u)) in x.iter_mut().zip(qp.xl.iter().zip(qp.xu.iter())) {
        if !xi.is_finite() {
            *xi = 0.0;
        }
        *xi = xi.clamp(l, u);
    }
    QpSolution {
        obj: quad_objective(qp, &x),
        x,
        lambda_g: vec![0.0; qp.m],
        lambda_x: vec![0.0; qp.n],
        working: WorkingSet::cold(qp.n, qp.m),
        status: QpStatus::TimeLimit,
        stats: QpStats {
            n_working_set_changes: 0,
            n_refactor,
            n_schur_updates: 0,
            used_phase1: false,
            time: crate::deadline::scope_elapsed(),
            ..Default::default()
        },
        unbounded_ray: None,
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
/// Largest constraint / bound violation at `x` (0.0 when feasible).
///
/// What [`ParametricActiveSetSolver::hint_pin_quality`] measured about a
/// working-set hint: how big it is, and how many rows and bounds outside it the
/// pinned point violates.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HintPinQuality {
    /// Rows plus bounds the hint marks active.
    pub(crate) active: usize,
    /// Inactive rows plus bounds the pinned point violates beyond `feas_tol`.
    pub(crate) violated: usize,
}

/// The magnitude behind [`point_is_feasible`]'s boolean, needed so a recovery
/// path can tell whether the point it is about to substitute is actually an
/// improvement on the one it is discarding.
pub(crate) fn max_violation(qp: &QpProblem, x: &[Number]) -> Number {
    let ax = a_times_x(qp.a, x, qp.m);
    let mut worst: Number = 0.0;
    for i in 0..qp.m {
        if qp.bl[i] > NLP_LOWER_BOUND_INF {
            worst = worst.max(qp.bl[i] - ax[i]);
        }
        if qp.bu[i] < NLP_UPPER_BOUND_INF {
            worst = worst.max(ax[i] - qp.bu[i]);
        }
    }
    for (i, &xi) in x.iter().enumerate() {
        if qp.xl[i] > NLP_LOWER_BOUND_INF {
            worst = worst.max(qp.xl[i] - xi);
        }
        if qp.xu[i] < NLP_UPPER_BOUND_INF {
            worst = worst.max(xi - qp.xu[i]);
        }
    }
    worst.max(0.0)
}

/// Rows and bounds that `x` violates by more than `feas_tol`, each paired with
/// the working-set status that pins it back onto the side it overshot.
/// Companion to [`ParametricActiveSetSolver::repair_pinned_hint`].
///
/// Returns `None` when a row or bound the working set already marks *active*
/// is violated. A pinned row is satisfied by construction, so that means the
/// pin did not take (an inconsistent or numerically hopeless hint) — and since
/// the repair only ever *adds* pins, re-pinning such a row cannot fix it.
#[allow(clippy::type_complexity)]
fn violated_inactive(
    qp: &QpProblem,
    x: &[Number],
    working: &WorkingSet,
    feas_tol: Number,
) -> Option<(Vec<(usize, ConsStatus)>, Vec<(usize, BoundStatus)>)> {
    let ax = a_times_x(qp.a, x, qp.m);
    let mut cons = Vec::new();
    for i in 0..qp.m {
        let below = qp.bl[i] > NLP_LOWER_BOUND_INF && ax[i] < qp.bl[i] - feas_tol;
        let above = qp.bu[i] < NLP_UPPER_BOUND_INF && ax[i] > qp.bu[i] + feas_tol;
        if !below && !above {
            continue;
        }
        if working.constraints[i].is_active() {
            return None;
        }
        cons.push((
            i,
            if qp.bl[i] == qp.bu[i] {
                ConsStatus::Equality
            } else if below {
                ConsStatus::AtLower
            } else {
                ConsStatus::AtUpper
            },
        ));
    }

    let mut bounds = Vec::new();
    for (i, &xi) in x.iter().enumerate() {
        let below = qp.xl[i] > NLP_LOWER_BOUND_INF && xi < qp.xl[i] - feas_tol;
        let above = qp.xu[i] < NLP_UPPER_BOUND_INF && xi > qp.xu[i] + feas_tol;
        if !below && !above {
            continue;
        }
        if working.bounds[i].is_active() {
            return None;
        }
        bounds.push((
            i,
            if qp.xl[i] == qp.xu[i] {
                BoundStatus::Fixed
            } else if below {
                BoundStatus::AtLower
            } else {
                BoundStatus::AtUpper
            },
        ));
    }

    Some((cons, bounds))
}

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

/// Step-length cap for the current search direction `p`: the exact
/// minimizer of the QP model along `p`, floored at the unit step.
///
/// With an **unshifted** active-set KKT the unit step already *is* that
/// minimizer. Writing `r = Hx + g`, the system solved is
/// `H p + A_Wᵀ λ = −r` with `A_W p = 0`, so `pᵀr = −pᵀHp` and
///
/// ```text
///     α* = −pᵀr / pᵀHp = 1.
/// ```
///
/// §4.5 inertia control breaks the identity: on an indefinite reduced
/// Hessian it factors `H + δI` instead, giving `pᵀr = −(pᵀHp + δ‖p‖²)`
/// and
///
/// ```text
///     α* = −pᵀr / pᵀHp = 1 + δ‖p‖² / pᵀHp    (> 1)
///     α* = +∞                                (pᵀHp ≤ 0)
/// ```
///
/// so the model asks for a step `1 + δ‖p‖²/pᵀHp` times longer than the
/// one the loop used to take, and asks for an unbounded one whenever the
/// true curvature along `p` is non-positive.
///
/// Taking the unit step anyway is gh #416: with `W` unchanged the inner
/// loop degenerates into proximal-point iteration with parameter δ —
/// `x ← argmin q(y) + ½δ‖y − x‖²` — whose contraction factor is
/// `δ/(λ + δ)` per eigenvalue λ of `H`. Since δ must exceed `|λ_min|` to
/// make the shifted system PD, and it is reached by multiplying by
/// `inertia_shift_factor` (100 by default), δ typically *dominates* the
/// spectrum: the reported Rosenbrock QP has `λ_min = −1.4` and δ = 100,
/// putting every factor within 3 % of 1. The result is a sequence of
/// ~1e-3-long "full steps" that never reach a bound, so 200 iterations
/// pass with zero working-set changes and the QP exits `MaxIter` — the
/// dimension-independent 200-iteration burn in the issue. With the cap
/// the negative-curvature direction runs to its blocking bound instead,
/// which is what an active-set method is supposed to do with it
/// (Nocedal-Wright §16.5).
///
/// `hx` must be `H·x` at the current iterate and `p` the direction just
/// solved for; `delta` is the shift that produced it. `delta == 0`
/// returns 1.0 without touching the data, so every non-shifted solve
/// keeps bit-identical behaviour.
fn model_step_cap(
    h: &pounce_linalg::triplet::SymTMatrix,
    g: &[Number],
    hx: &[Number],
    p: &[Number],
    delta: Number,
) -> Number {
    if delta <= 0.0 {
        return 1.0;
    }

    let mut hp = vec![0.0; p.len()];
    let mut h_scale: Number = 0.0;
    let irows = h.irows();
    let jcols = h.jcols();
    let vals = h.values();
    for k in 0..irows.len() {
        let i = (irows[k] - 1) as usize;
        let j = (jcols[k] - 1) as usize;
        let v = vals[k];
        h_scale = h_scale.max(v.abs());
        hp[i] += v * p[j];
        if i != j {
            hp[j] += v * p[i];
        }
    }

    let curv: Number = p.iter().zip(hp.iter()).map(|(pi, hpi)| pi * hpi).sum();
    let slope: Number = p
        .iter()
        .zip(hx.iter().zip(g.iter()))
        .map(|(pi, (hxi, gi))| pi * (hxi + gi))
        .sum();

    // Relative floor on the curvature. `pᵀHp` below the round-off level
    // of its own accumulation is zero, not a tiny positive number —
    // dividing by it would manufacture an α* of 1e16 and hurl the
    // iterate out of the box. Below the floor the model is (at best)
    // linear along `p`, so the step is bounded only by the ratio test.
    let p_sq: Number = p.iter().map(|v| v * v).sum();
    if curv > 1e-12 * h_scale * p_sq {
        (-slope / curv).max(1.0)
    } else if slope < 0.0 {
        // A successful shifted factorization has `pᵀ(H + δI)p > 0`, hence
        // `pᵀr = −(pᵀHp + δ‖p‖²) < 0`: descent is structural here, and the
        // test only guards against a direction corrupted by round-off.
        Number::INFINITY
    } else {
        1.0
    }
}

pub(crate) fn quad_objective(qp: &QpProblem, x: &[Number]) -> Number {
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
        let (alpha, blocker) = select_blocker(&candidates, &opts, 1e-3, false, 1.0);
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
        let (alpha, blocker) = select_blocker(&candidates, &opts, 1e-3, false, 1.0);
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
        let (alpha, blocker) = select_blocker(&candidates, &opts, 1e-9, false, 1.0);
        assert!(matches!(
            blocker,
            Some(BlockerTarget::Bound(0, BoundStatus::AtLower))
        ));
        assert!(alpha >= 0.5 && alpha <= 1.0, "α in range, got {alpha}");
    }
}
