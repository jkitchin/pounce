//! Active-set QP driver — the [`pounce_qp`] parametric active-set engine
//! wired directly into the convex path, as a peer of [`crate::ipm`].
//!
//! # Why this module exists
//!
//! `solver_selection=qp-active-set` used to reach the active-set engine the
//! long way round: the CLI rewrote it to `algorithm=active-set-sqp` and ran
//! the **full SQP outer loop** over the QP, treating it as a general NLP.
//!
//! That is not mathematically wrong. With an exact Hessian and constraints
//! that are already linear, the first SQP subproblem *is* the original QP, so
//! a healthy run converges in one outer iteration (and did — successful solves
//! reported `n_iter = 1`). But it is an architectural mismatch that forfeits
//! everything the convex path already has, and pays for machinery a QP does
//! not need:
//!
//! * **No presolve.** The SQP path never touches [`crate::presolve`], so the
//!   engine faced every redundant row, fixed variable, and singleton the
//!   IPM never sees. This matters far more for an active-set method than for
//!   an IPM: interior-point iteration counts are nearly problem-size
//!   independent, whereas an active-set pivot count is *combinatorial* in the
//!   size of the active set. Presolve is the difference between a shrunk
//!   problem and one that exhausts its iteration budget.
//! * **No scaling.** No Ruiz equilibration, and `nlp_scaling` is not threaded
//!   through the SQP residuals at all.
//! * **No timing, and misleading statuses**, both of which routed through the
//!   NLP report path rather than the convex one.
//! * **Pointless overhead**: AMPL evaluation callbacks, BFGS storage, and a
//!   filter/line-search globalization, none of which do anything for a
//!   problem whose linearization is exact.
//!
//! This module is the direct route. It hands the QP straight to
//! [`ParametricActiveSetSolver`], inside the same presolve → solve → postsolve
//! wrapper the IPM runs under, so the engine inherits the convex driver's
//! problem reduction, reporting, and status vocabulary.
//!
//! The SQP route is untouched and remains correct for genuine NLPs
//! (`algorithm=active-set-sqp`), where the outer loop is doing real work.
//!
//! # Translation
//!
//! The convex → `pounce-qp` translation (and, critically, the **dual sign
//! transform** on the way back) is the same one [`crate::crossover`] performs;
//! see that module's docs for the derivation. The differences here are:
//!
//! * The Hessian is carried through. Crossover is gated to pure LPs and builds
//!   an empty `H`; a QP driver must translate `p_lower`. Both sides store the
//!   **lower triangle once, 1-based for `pounce-qp` and 0-based for the convex
//!   form**, mirroring off-diagonals implicitly — see
//!   [`QpProblem::p_mul_add`](crate::qp::QpProblem::p_mul_add) against
//!   `pounce_qp`'s `SymTMatrixSpace`, which agree exactly, so the map is a
//!   straight `+1` on both indices with no triangle fixup.
//! * The starting point comes from a **simplex phase-1 feasible vertex**
//!   rather than an IPM iterate (there is no prior iterate here to hint from).
//!   That seed is what makes this path work at all: `pounce-qp`'s own
//!   l1-elastic phase-1 does not terminate on the degenerate netlib-derived
//!   QPs in the Maros-Mészáros set, and handing `solve` a feasible primal
//!   routes it straight to phase-2, which is the part of the method that is
//!   strong. See `crate::simplex::simplex_feasible_vertex`.
//! * Every terminal status is mapped, not just `Optimal` — this is the primary
//!   driver, so it must report `MaxIter` / `NumericalError` / `Infeasible` /
//!   `Unbounded` honestly rather than falling back to a previous solution.

use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_qp::{
    AntiCyclingChoice, BoundStatus, ConsStatus, HessianInertia, ParametricActiveSetSolver,
    QpOptions as ActiveSetOptions, QpProblem as ActiveSetProblem, QpSolver,
    QpStatus as ActiveSetStatus, QpWarmStart, WorkingSet,
};

use crate::ipm::{FARKAS_RESID_TOL, QpOptions, dot, inf_norm};
use crate::qp::{QpProblem, QpSolution, QpStatus};

/// Caller-supplied overrides for the inner `pounce-qp` engine.
///
/// Every field is `None` unless the user set the corresponding option
/// explicitly, so this driver can tell "left at the default" from "asked for
/// the default value" — a distinction it needs, because it deliberately
/// overrides two of these itself (a size-scaled `max_iter` and
/// `use_schur_updates: true`) and an explicit request must win over that.
///
/// Exists because these knobs became unreachable when `qp-active-set` moved off
/// the SQP outer loop: the `sqp_qp_*` option family fed the SQP's QP
/// subproblem, and with no SQP in the picture all seven silently became no-ops.
#[derive(Debug, Clone, Copy, Default)]
pub struct ActiveSetOverrides {
    pub max_iter: Option<u32>,
    pub anti_cycling: Option<AntiCyclingChoice>,
    pub feas_tol: Option<f64>,
    pub opt_tol: Option<f64>,
    pub elastic_gamma: Option<f64>,
    pub use_schur_updates: Option<bool>,
    pub use_homotopy: Option<bool>,
    pub max_schur_updates_before_refactor: Option<u32>,
}

impl ActiveSetOverrides {
    /// True when the caller set nothing.
    pub fn is_empty(&self) -> bool {
        self.max_iter.is_none()
            && self.anti_cycling.is_none()
            && self.feas_tol.is_none()
            && self.opt_tol.is_none()
            && self.elastic_gamma.is_none()
            && self.use_schur_updates.is_none()
            && self.use_homotopy.is_none()
            && self.max_schur_updates_before_refactor.is_none()
    }

    fn apply(&self, o: &mut ActiveSetOptions) {
        if let Some(v) = self.max_iter {
            o.max_iter = v;
        }
        if let Some(v) = self.anti_cycling {
            o.anti_cycling = v;
        }
        if let Some(v) = self.feas_tol {
            o.feas_tol = v;
        }
        if let Some(v) = self.opt_tol {
            o.opt_tol = v;
        }
        if let Some(v) = self.elastic_gamma {
            o.elastic_gamma = v;
        }
        if let Some(v) = self.use_schur_updates {
            o.use_schur_updates = v;
        }
        if let Some(v) = self.use_homotopy {
            o.use_homotopy = v;
        }
        if let Some(v) = self.max_schur_updates_before_refactor {
            o.max_schur_updates_before_refactor = v;
        }
    }
}

/// Clamp a convex lower-bound value to pounce-qp's `±1e19` free convention.
fn to_qp_lower(lb: f64) -> f64 {
    if lb <= NLP_LOWER_BOUND_INF {
        NLP_LOWER_BOUND_INF
    } else {
        lb
    }
}

/// Clamp a convex upper-bound value to pounce-qp's `±1e19` free convention.
fn to_qp_upper(ub: f64) -> f64 {
    if ub >= NLP_UPPER_BOUND_INF {
        NLP_UPPER_BOUND_INF
    } else {
        ub
    }
}

/// A solution carrying no information, returned when the engine reports a
/// status for which no iterate is meaningful. `x` is zero-filled rather than
/// left empty so downstream residual/objective code has the right lengths.
fn empty_solution(n: usize, m_eq: usize, m_ineq: usize, status: QpStatus) -> QpSolution {
    QpSolution {
        status,
        x: vec![0.0; n],
        y: vec![0.0; m_eq],
        z: vec![0.0; m_ineq],
        z_lb: vec![0.0; n],
        z_ub: vec![0.0; n],
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    }
}

/// Solve a convex QP with the [`pounce_qp`] parametric active-set engine.
///
/// Signature-compatible with [`crate::ipm::solve_qp_ipm`] so the CLI driver
/// can select between them without restructuring, including the
/// `make_backend` factory (the active-set engine may need more than one
/// backend instance over a solve).
pub fn solve_qp_active_set<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    engine: &ActiveSetOverrides,
    make_backend: &mut F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    // Ruiz equilibration is applied as a **retry after failure**, not
    // unconditionally, because for this engine it is a genuine trade rather
    // than an improvement. Measured on Maros-Mészáros:
    //
    // | problem  | unscaled            | Ruiz-equilibrated   |
    // |----------|---------------------|---------------------|
    // | LOTSCHD  | MaxIter (2.36)      | **Optimal (2398.4)**|
    // | CVXQP1_S | **Optimal (11590.7)**| MaxIter (17220)    |
    // | CVXQP2_S | **Optimal (8120.9)** | MaxIter (9965)     |
    //
    // Scaling rescues the badly-scaled netlib-derived instances and breaks the
    // well-scaled CVXQP family, so *neither* fixed choice dominates. Solving
    // unscaled first and equilibrating only when that fails takes both columns'
    // wins and cannot regress: we reach the retry only after the first attempt
    // already failed to produce a verified KKT point. This mirrors the IPM's
    // own HSDE-then-Ruiz fallback (`solve_qp_ipm_core`) and its reasoning
    // — "there is nothing left to regress".
    let unscaled_opts = QpOptions {
        equilibrate: false,
        ..*opts
    };
    let sol = solve_translated(
        prob,
        &unscaled_opts,
        engine,
        make_backend,
        FeasibilityProbe::Allowed,
    );
    if is_conclusive(sol.status) {
        return sol;
    }

    let mut best = sol;
    if opts.equilibrate {
        let (scaled, scaling) = crate::equilibrate::equilibrate(prob);
        let mut retry = solve_translated(
            &scaled,
            &unscaled_opts,
            engine,
            make_backend,
            FeasibilityProbe::Allowed,
        );
        scaling.unscale_solution(prob, &mut retry);
        // Re-verify against the ORIGINAL problem: the verdict reached inside the
        // scaled solve certifies a KKT point of the *scaled* QP, and unscaling
        // moves the residuals, so it has to be re-earned here rather than carried
        // over on the strength of the wrong problem's numbers.
        retry.status = reverify_after_unscale(retry.status, &retry, prob, opts);
        // Keep the original failure when the retry also fails, so the reported
        // status describes the attempt made on the problem as the user posed it.
        // `PrimalInfeasible` counts as a win here because `reverify_after_unscale`
        // just re-earned its Farkas certificate against the *original* problem;
        // `DualInfeasible` does not, because its witness (the engine's ray) is not
        // available out here to re-check.
        if is_solved(retry.status) || retry.status == QpStatus::PrimalInfeasible {
            best = retry;
        }
        if is_conclusive(best.status) {
            return best;
        }
    }

    // ---- Last resort: the simplex seed the homotopy displaced (#413) ----
    //
    // Turning the homotopy on also turned the simplex phase-1 vertex seed *off*
    // (see the `seed` binding in `solve_translated`), on the stated premise that
    // the homotopy "is feasible along the whole path" and so is itself the
    // cold-start mechanism. That premise is false — see the path-feasibility
    // guard in `pounce_qp::homotopy` — and when it fails the engine is left with
    // *neither* mechanism: no usable path, and no seed, so it cold-starts the
    // l1-elastic phase-1 that this module's own header records as not
    // terminating on the degenerate netlib-derived QPs in Maros-Mészáros.
    //
    // That is the #413 timeout mode, and it is not a slow corrector. On
    // `QSHARE2B` the seedless route spends its entire iteration budget to return
    // `4854` against a published `11703.7`, still carrying a constraint
    // violation of 20; the seeded route solves it in 52 iterations / 0.03 s.
    //
    // Ordering is load-bearing. This runs *after* the Ruiz retry, not before:
    // both earlier attempts are then bit-identical to what they were, so nothing
    // that used to solve can be displaced — including by the clock, which is the
    // way an added stage can regress a benchmark even when its logic cannot.
    // (Measured: with this stage first, `QPCBOEI1` went from 0.68 s to a 60 s
    // timeout, because the seeded attempt consumed the budget the Ruiz retry
    // needed.) When the caller already asked for `use_homotopy=no`, the first
    // attempt was seeded and there is nothing new to try.
    if engine.use_homotopy != Some(false) {
        let seeded_engine = ActiveSetOverrides {
            use_homotopy: Some(false),
            ..*engine
        };
        let seeded = solve_translated(
            prob,
            &unscaled_opts,
            &seeded_engine,
            make_backend,
            FeasibilityProbe::Allowed,
        );
        if is_solved(seeded.status) || seeded.status == QpStatus::PrimalInfeasible {
            return seeded;
        }
    }

    best
}

/// Did the solve produce a usable, verified KKT point?
fn is_solved(s: QpStatus) -> bool {
    matches!(s, QpStatus::Optimal | QpStatus::OptimalInaccurate)
}

/// Is this verdict final — nothing a second attempt could improve on?
///
/// Either a verified KKT point or a *verified certificate*. Both are earned
/// against the original problem (`verify_status` / `reverify_after_unscale`
/// re-derive every certificate there), so there is no weaker claim here than
/// [`is_solved`] makes: an equilibrated retry cannot overturn a proof, and
/// running one anyway would only burn a second solve before falling back to
/// this same answer.
///
/// `DualInfeasible` is deliberately absent from the retry-acceptance side of
/// this test (see the call site): the unboundedness certificate is re-derived
/// from a ray that only exists inside `solve_translated`, so a retry's claim
/// has no witness left to re-check after unscaling.
fn is_conclusive(s: QpStatus) -> bool {
    is_solved(s) || matches!(s, QpStatus::PrimalInfeasible | QpStatus::DualInfeasible)
}

/// May this attempt spend a second solve on the objective-free feasibility twin
/// to turn an uncertified infeasibility claim into a certified one?
///
/// Exists only to make the recursion finite: the probe re-enters
/// [`solve_translated`] on a problem whose own probe would be the identical
/// solve, so the inner call is handed [`Self::Forbidden`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeasibilityProbe {
    Allowed,
    Forbidden,
}

/// Translate to `pounce-qp` form, solve, and verify — one attempt, no scaling
/// decisions. `opts.equilibrate` is ignored here; the caller owns that choice.
fn solve_translated<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    engine: &ActiveSetOverrides,
    make_backend: &mut F,
    probe: FeasibilityProbe,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();
    let m = m_eq + m_ineq;

    // ---- Hessian: lower triangle, 0-based -> 1-based (no triangle fixup) ----
    let mut h_irow = Vec::with_capacity(prob.p_lower.len());
    let mut h_jcol = Vec::with_capacity(prob.p_lower.len());
    let mut h_val = Vec::with_capacity(prob.p_lower.len());
    for t in &prob.p_lower {
        // Defensive: the convex form documents `row >= col`, but a caller
        // that supplied the upper triangle would otherwise be silently
        // transposed into a different matrix. Normalizing costs nothing and
        // makes the translation total.
        let (r, c) = if t.row >= t.col {
            (t.row, t.col)
        } else {
            (t.col, t.row)
        };
        h_irow.push((r + 1) as i32);
        h_jcol.push((c + 1) as i32);
        h_val.push(t.val);
    }
    let h_space = SymTMatrixSpace::new(n as i32, h_irow, h_jcol);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&h_val);

    // ---- Jacobian A_qp = [A_eq ; G], 1-based ----
    let nnz = prob.a.len() + prob.g.len();
    let mut irows = Vec::with_capacity(nnz);
    let mut jcols = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);
    for t in &prob.a {
        irows.push((t.row + 1) as i32);
        jcols.push((t.col + 1) as i32);
        vals.push(t.val);
    }
    for t in &prob.g {
        irows.push((m_eq + t.row + 1) as i32);
        jcols.push((t.col + 1) as i32);
        vals.push(t.val);
    }
    let mut a_qp = GenTMatrix::new(GenTMatrixSpace::new(m as i32, n as i32, irows, jcols));
    a_qp.set_values(&vals);

    // ---- Row bounds: eq rows bl=bu=b; ineq rows bl=-inf, bu=h ----
    let mut bl = Vec::with_capacity(m);
    let mut bu = Vec::with_capacity(m);
    for &bk in &prob.b {
        bl.push(bk);
        bu.push(bk);
    }
    for &hi in &prob.h {
        bl.push(NLP_LOWER_BOUND_INF);
        bu.push(to_qp_upper(hi));
    }

    // ---- Variable bounds ----
    let mut xl = Vec::with_capacity(n);
    let mut xu = Vec::with_capacity(n);
    for i in 0..n {
        xl.push(to_qp_lower(prob.lb_of(i)));
        xu.push(to_qp_upper(prob.ub_of(i)));
    }

    let g_lin = prob.c.clone();
    let qp = ActiveSetProblem {
        n,
        m,
        h: &h,
        g: &g_lin,
        a: &a_qp,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        // The convex path only accepts problems the class detector has already
        // ruled convex, so the Hessian is PSD by construction.
        hessian_inertia: HessianInertia::Psd,
    };

    let qopts = ActiveSetOptions {
        max_iter: active_set_iter_budget(opts, n, m),
        // Absorb working-set changes as rank-2 Schur updates against a cached
        // factor instead of assembling and factoring a fresh active-set KKT
        // every iteration. This is the lever the wall-clock problem needs: a
        // cold convex-QP solve here runs thousands of iterations (budget
        // `10·(n+m)`), and a per-iteration refactorization is what turns that
        // into a timeout — `QSCTAP1` goes from a 60s timeout to ~2s.
        //
        // Enabling it required fixing two real defects in that path first
        // (`pounce-qp`: singular-Schur recovery, and iterative refinement in
        // `SchurState::solve`); before those it silently lost ~1e-6 of
        // accuracy and aborted outright on singular Schur blocks. See
        // `pounce-qp/tests/schur_vs_refactor.rs`.
        use_schur_updates: true,
        // Trace the §4.2 parametric homotopy rather than the conventional
        // phase-1/phase-2 scheme. Measured on the full Maros-Mészáros set at a
        // 120 s cap, same binary: 71/138 correct against 58/138 for the
        // conventional path, with zero solved-but-wrong in both.
        //
        // The trade is not uniform — 20 problems gained, 7 lost, and six of the
        // losses are large instances that previously solved and now hit the
        // cap (AUG2D, AUG2DC, CONT-050, CONT-100, DTOC3, STADAT3): the homotopy
        // is slower there, not wrong. `sqp_qp_use_homotopy=no` restores the old
        // behaviour for such a workload.
        //
        // Set here rather than in `QpOptions::default()` on purpose: that
        // default is read by every `pounce-qp` consumer including the SQP outer
        // loop's inner subproblems, which this benchmark does not cover.
        use_homotopy: true,
        ..ActiveSetOptions::default()
    };
    // Applied last: the two settings this driver picks for itself (the
    // size-scaled `max_iter` and `use_schur_updates`) are defaults, not
    // mandates, so an explicit user request overrides them.
    let qopts = {
        let mut q = qopts;
        engine.apply(&mut q);
        q
    };

    // Seed from a simplex phase-1 feasible **vertex**, when one is available.
    //
    // `QpSolver::solve` routes a warm start whose primal is feasible directly
    // into phase-2 and never enters l1-elastic mode — which is the point, since
    // that phase-1 does not terminate on the degenerate netlib-derived QPs here
    // (see `simplex::simplex_feasible_vertex`). Phase-2 from a feasible vertex
    // is the part of this method that works well.
    //
    // The working set must come from the simplex **basis**, not from
    // tolerance-snapping the point. `solve_general` trusts the working set it
    // is handed and steps with a zero-RHS active-set system, so:
    //   * a *cold* working set claims nothing is active, and on these low-rank
    //     `P` instances the reduced Hessian is then singular and the engine
    //     certifies a spurious zero-curvature ray (measured: `QSHARE2B`
    //     `-1.2e10`, `QSCAGR7` `-9.9e14`, both `DivergingIterates`);
    //   * snapping every tight row instead names *more than* `n` binding rows
    //     at a degenerate vertex, and no H-block shift repairs a rank-deficient
    //     constraint block (the failure `crossover` documents on the GEN family).
    // The basis avoids both: exactly `n` nonbasic variables, each on a bound,
    // linearly independent because the basis matrix is nonsingular.
    // Skipped entirely when the homotopy is on: the seed exists *because* the
    // parametric homotopy was missing, and the homotopy is itself the cold-start
    // mechanism (it starts from the box relaxation and is feasible along the
    // whole path). Computing one anyway would both pay for a phase-1 solve that
    // is no longer needed and suppress the homotopy, whose hook in
    // `QpSolver::solve` only fires on a genuinely cold call.
    let seed = if qopts.use_homotopy {
        None
    } else {
        crate::simplex::simplex_feasible_vertex(prob).map(|v| {
            use crate::simplex::AtBound;
            let mut working = WorkingSet::cold(n, m);
            for (i, st) in working.bounds.iter_mut().enumerate() {
                // A variable fixed by equal bounds is `Fixed` regardless of which
                // side the basis parked it on.
                let (l, u) = (xl[i], xu[i]);
                let fixed = l > NLP_LOWER_BOUND_INF && u < NLP_UPPER_BOUND_INF && l == u;
                *st = match (fixed, v.struct_at[i]) {
                    (true, _) => BoundStatus::Fixed,
                    (false, AtBound::Lower) => BoundStatus::AtLower,
                    (false, AtBound::Upper) => BoundStatus::AtUpper,
                    (false, AtBound::Free) => BoundStatus::Inactive,
                };
            }
            // Row order matches: the translation above stacks `[A_eq ; G]` in the
            // same order the simplex builds its rows, so index `i` is the same row.
            for (i, st) in working.constraints.iter_mut().enumerate() {
                // Equality rows are ALWAYS active — they are equations, not
                // one-sided constraints that may or may not bind.
                //
                // Deriving their activity from the basis instead (leaving a row
                // inactive when its logical is basic) is tempting: it makes the
                // active count come out at exactly `n` and so avoids the
                // `KKT inertia mismatch: expected 105 …, got 102` that `CVXQP3_S`
                // otherwise hits. But the premise is false. An equality's logical
                // is boxed to `[0, 0]`; a *basic* one is simply a degenerate basic
                // variable sitting at zero, which says nothing about whether the row
                // is a linear combination of the others. Dropping such a row drops a
                // real constraint — the ratio test skips `bl == bu` rows, so it
                // never comes back — and the engine is then free to move in a
                // direction that violates it. On NETLIB `afiro` that produced a
                // phantom `Unbounded` verdict on an LP with a finite optimum of
                // −464.75 (issue #133's regression test caught it).
                //
                // More active rows than variables *is* legitimate at a degenerate
                // vertex; the dependence among them is a rank problem, and
                // `schur_reset_rank_repaired` in `pounce-qp` is what resolves it —
                // by pruning to a maximal independent subset with the algebra to
                // justify each drop, rather than guessing from basis status here.
                *st = if i < m_eq {
                    ConsStatus::Equality
                } else {
                    match v.row_at[i] {
                        // Slack `s = h − Gx` nonbasic at its lower bound 0 means
                        // `Gx = h`: the row sits at its upper bound `bu = h`.
                        AtBound::Lower | AtBound::Upper => ConsStatus::AtUpper,
                        AtBound::Free => ConsStatus::Inactive,
                    }
                };
            }
            QpWarmStart {
                x: v.x,
                lambda_g: vec![0.0; m],
                lambda_x: vec![0.0; n],
                working,
            }
        })
    };

    debug_trace(|| {
        format!(
            "n={n} m_eq={m_eq} m_ineq={m_ineq} seed={} budget={}",
            if seed.is_some() {
                "SIMPLEX-VERTEX"
            } else {
                "NONE (cold)"
            },
            qopts.max_iter,
        )
    });
    let mut solver = ParametricActiveSetSolver::new(make_backend());
    let qsol = match solver.solve(&qp, seed.as_ref(), &qopts) {
        Ok(q) => q,
        // A hard `QpError` (singular factor, dimension mismatch) is a
        // numerical failure, not an infeasibility claim — never assert
        // infeasibility without a certificate.
        //
        // The error text is only surfaced under the debug env var, but it is
        // the *only* record of why the solve died: this arm discards the
        // engine's message and returns a zero-filled solution, so a caller
        // otherwise sees `NumericalFailure` with no cause. Two distinct causes
        // hide here on Maros-Mészáros — `KKT matrix is singular (LICQ
        // violation …)`, the documented rank-detection limitation, and
        // `KKT inertia mismatch`, which was a rank-deficient seed working set.
        Err(e) => {
            debug_trace(|| format!("solver.solve HARD ERROR: {e}"));
            return empty_solution(n, m_eq, m_ineq, QpStatus::NumericalFailure);
        }
    };

    let engine_status = qsol.status;
    // Kept for the `Unbounded` certificate re-check in `verify_status`.
    let engine_ray = qsol.unbounded_ray.clone();

    // ---- Back-translate (sign transform — see crate::crossover docs) ----
    let mut y = vec![0.0; m_eq];
    if qsol.lambda_g.len() >= m_eq {
        y.copy_from_slice(&qsol.lambda_g[..m_eq]);
    }
    let mut z = vec![0.0; m_ineq];
    for i in 0..m_ineq {
        if let Some(&l) = qsol.lambda_g.get(m_eq + i) {
            z[i] = l.max(0.0);
        }
    }
    let mut z_lb = vec![0.0; n];
    let mut z_ub = vec![0.0; n];
    for i in 0..n {
        if let Some(&l) = qsol.lambda_x.get(i) {
            z_lb[i] = l.max(0.0);
            z_ub[i] = (-l).max(0.0);
        }
    }

    // Objective recomputed in convex coordinates (½xᵀPx + cᵀx) rather than
    // taken from the engine, so the two forms cannot silently drift apart.
    let mut px = vec![0.0; n];
    prob.p_mul(&qsol.x, &mut px);
    let obj = (0..n).map(|i| (0.5 * px[i] + prob.c[i]) * qsol.x[i]).sum();

    let mut sol = QpSolution {
        // Provisional — `verify_status` below decides the final verdict.
        status: QpStatus::Optimal,
        x: qsol.x,
        y,
        z,
        z_lb,
        z_ub,
        obj,
        // The active-set engine counts active-set changes rather than
        // interior-point iterations; that is its analogue of "iterations"
        // and the quantity a user tuning `max_iter` is actually spending.
        iters: qsol.stats.n_working_set_changes as usize,
        iterates: Vec::new(),
    };
    sol.status = verify_status(engine_status, engine_ray.as_deref(), &sol, prob, opts);
    // The engine says infeasible and its own multipliers could not prove it.
    // Before demoting that to "the solver broke", spend one more solve on the
    // objective-free twin, whose multipliers *can* — see [`feasibility_probe`].
    if engine_status == ActiveSetStatus::Infeasible
        && sol.status == QpStatus::NumericalFailure
        && probe == FeasibilityProbe::Allowed
        && let Some((y, z)) = feasibility_probe(prob, opts, engine, make_backend)
    {
        // Carry the *certifying* multipliers out, replacing the
        // objective-carrying ones that proved nothing. This keeps the invariant
        // every consumer of a `PrimalInfeasible` here relies on — the returned
        // `(y, z)` verify the status attached to them — which is what lets
        // `reverify_after_unscale` re-earn the verdict against the original
        // problem after an equilibrated retry. Without it that re-check tested
        // the wrong vectors and threw away a proof the driver had just made.
        // The bound duals are dropped rather than left mismatched: the
        // certificate is a statement about `(y, z)` and the box, and stale
        // `z_lb`/`z_ub` from a different multiplier set say nothing about it.
        sol.y = y;
        sol.z = z;
        sol.z_lb = vec![0.0; n];
        sol.z_ub = vec![0.0; n];
        sol.status = QpStatus::PrimalInfeasible;
    }
    debug_trace(|| {
        format!(
            "engine={:?} -> reported={:?} kkt_err={:.3e} obj={:.6e}",
            engine_status,
            sol.status,
            sol.kkt_residuals(prob).kkt_error(),
            sol.obj,
        )
    });
    sol
}

/// Re-solve the **objective-free twin** of `prob` — same `A, b, G, h` and the
/// same box, but `P = 0` and `c = 0` — and return *its* multipliers `(y, z)`
/// when they prove the constraint system has no solution.
///
/// # Why a second solve is the right answer here
///
/// [`certifies_primal_infeasible`] fails on an unbounded variable for a
/// structural reason, not a tolerance one: the elastic phase-1 multipliers carry
/// a residual `Aᵀy + Gᵀz = −(Px + c)` left behind by the original objective, and
/// with no finite bound on that variable there is nothing to bound its
/// contribution with. Deleting the objective deletes the residual at the source:
/// the twin's phase-1 minimizes violation *alone*, so its stationarity is
/// `Aᵀy + Gᵀz = 0` to machine precision — a textbook Farkas pair. Measured on
/// the gh #415 LP with its bounds removed: `q = (0, 0)` exactly against
/// `q = (−1, −1)` from the objective-carrying solve.
///
/// The twin has the *same feasible set* as `prob`, so a certificate for it is a
/// certificate for `prob` — which is why the check below is run against `prob`
/// itself and needs no translation back.
///
/// This costs a whole extra solve, so it is deliberately last: only after the
/// engine has already claimed infeasibility *and* the free check on its own
/// multipliers came up empty. A feasible problem reaches it only via a false
/// `Infeasible` claim (the `DUALC1` hazard), and then the twin solve finds the
/// feasible set is non-empty and certifies nothing — the claim stays demoted.
fn feasibility_probe<F>(
    prob: &QpProblem,
    opts: &QpOptions,
    engine: &ActiveSetOverrides,
    make_backend: &mut F,
) -> Option<(Vec<f64>, Vec<f64>)>
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let twin = QpProblem {
        p_lower: Vec::new(),
        c: vec![0.0; prob.n],
        ..prob.clone()
    };
    let sol = solve_translated(
        &twin,
        opts,
        engine,
        make_backend,
        FeasibilityProbe::Forbidden,
    );
    certifies_primal_infeasible(prob, &sol.y, &sol.z, opts).then_some((sol.y, sol.z))
}

/// Emit a diagnostic line when `POUNCE_AS_DEBUG` is set in the environment.
///
/// This path makes three decisions that are invisible in the reported status —
/// whether a simplex seed was obtained, what iteration budget was chosen, and
/// what the engine's own verdict was before verification either accepted or
/// demoted it. Reconstructing those from the outside is guesswork, and the
/// guesses were wrong twice while this driver was being built. The closure
/// defers formatting so an unset var costs one `env::var` and no allocation.
fn debug_trace(msg: impl FnOnce() -> String) {
    if std::env::var("POUNCE_AS_DEBUG").is_ok() {
        eprintln!("[as] {}", msg());
    }
}

/// Re-earn a status against the original problem after unscaling.
///
/// A success carried out of the scaled solve is only evidence about the scaled
/// QP. Rather than trust it, re-measure: keep a clean verdict only if the
/// unscaled point still earns it, and let a solve that lands in the acceptable
/// band say so. Non-success statuses pass through — they made no claim to
/// re-check.
///
/// A `PrimalInfeasible` carried out of the scaled solve is a claim too, and one
/// with the same problem: equilibration is a diagonal congruence, so it
/// preserves *whether* the QP is infeasible but not the certificate's numbers,
/// which `unscale_solution` has moved. So the Farkas certificate is re-derived
/// here against the original data rather than inherited on the strength of the
/// scaled problem's arithmetic.
fn reverify_after_unscale(
    scaled_status: QpStatus,
    sol: &QpSolution,
    prob: &QpProblem,
    opts: &QpOptions,
) -> QpStatus {
    if scaled_status == QpStatus::PrimalInfeasible {
        return if certifies_primal_infeasible(prob, &sol.y, &sol.z, opts) {
            QpStatus::PrimalInfeasible
        } else {
            QpStatus::NumericalFailure
        };
    }
    if !matches!(
        scaled_status,
        QpStatus::Optimal | QpStatus::OptimalInaccurate
    ) {
        return scaled_status;
    }
    let err = sol.kkt_residuals(prob).kkt_error();
    if !err.is_finite() {
        QpStatus::NumericalFailure
    } else if err <= opts.tol {
        QpStatus::Optimal
    } else if err <= 1e3 * opts.tol {
        QpStatus::OptimalInaccurate
    } else {
        QpStatus::NumericalFailure
    }
}

/// Decide the reported status from the engine's verdict **and** the measured
/// KKT residuals of the point it returned.
///
/// The engine's own status is never propagated unchecked, for two reasons the
/// Maros-Mészáros set demonstrates directly:
///
/// * **A claimed `Optimal` can be wrong.** `QSC205` returns `Optimal` with
///   objective `0.0` against a true optimum of `−5.81e−3`. Routed through the
///   SQP outer loop this was caught downstream — the outer loop re-tested the
///   NLP KKT conditions and refused to converge, reporting an iteration limit.
///   Solving directly removes that accidental safety net, so the check has to
///   be made deliberately here. A *silently wrong* answer is the one failure
///   mode worse than not solving: the baseline active-set column had zero
///   solved-but-wrong across all 138 problems, and that property must survive
///   this refactor.
/// * **A claimed `Infeasible` can be wrong.** `DUALC1` is feasible with a
///   published optimum of `6155.25`, and the phase-1 elastic mode certifies it
///   infeasible. An infeasibility verdict is a *proof obligation*: the IPM
///   only reports one behind a verified Farkas certificate, and the SQP path
///   deliberately refuses to assert infeasibility it cannot back (#282). So the
///   claim is not propagated on the engine's word — it is re-derived here from
///   the phase-1 multipliers ([`certifies_primal_infeasible`], with
///   [`feasibility_probe`] as a second attempt) and only reported when it holds.
///   A claim that cannot be backed is downgraded to an honest "did not solve".
///
/// The `Optimal` / `OptimalInaccurate` / fail banding mirrors the IPM's
/// post-loop verdict (`hsde.rs`): within `tol` is clean, within `1e3·tol` is
/// the "acceptable level" tier, anything worse is not a solve.
fn verify_status(
    engine: ActiveSetStatus,
    ray: Option<&[f64]>,
    sol: &QpSolution,
    prob: &QpProblem,
    opts: &QpOptions,
) -> QpStatus {
    /// Multiple of `tol` inside which a solve is downgraded to "acceptable"
    /// rather than rejected — the IPM's `1e3·tol` band.
    const ACCEPTABLE_FACTOR: f64 = 1e3;

    let err = sol.kkt_residuals(prob).kkt_error();
    let solved_to = |e: f64| {
        if !e.is_finite() {
            None
        } else if e <= opts.tol {
            Some(QpStatus::Optimal)
        } else if e <= ACCEPTABLE_FACTOR * opts.tol {
            Some(QpStatus::OptimalInaccurate)
        } else {
            None
        }
    };

    match engine {
        // Trust, but verify.
        ActiveSetStatus::Optimal => solved_to(err).unwrap_or(QpStatus::NumericalFailure),
        // Unboundedness is a proof obligation too, and the engine's claim does
        // not survive contact with this benchmark: `QSCSD1` (published optimum
        // 8.667, plainly bounded) is reported `Unbounded`. Propagating that
        // unchecked produced "Problem is unbounded (dual infeasible)" on a
        // bounded QP — a *wrong verdict about the user's model*, which is worse
        // than any failure status. So re-derive the certificate here from the
        // returned ray, exactly as the `Infeasible` case is re-derived.
        ActiveSetStatus::Unbounded => match ray {
            Some(d) if ray_certifies_unbounded(prob, d) => QpStatus::DualInfeasible,
            // No ray, or a ray that does not stand up: fall back on the point.
            _ => solved_to(err).unwrap_or(QpStatus::NumericalFailure),
        },
        // Infeasibility is a proof obligation, so the claim is re-derived from
        // the phase-1 multipliers exactly as the `Unbounded` claim is re-derived
        // from the ray. A certificate that stands up is the *right* verdict about
        // the user's model and must be reported as such (gh #415: without this,
        // a two-row LP whose contradiction is visible by inspection came back as
        // `InternalError` / `solve_result_num=500` — "the solver broke, retry" —
        // where every other engine said `200`, "your model is infeasible").
        // A claim that does not stand up is still downgraded: if the returned
        // point satisfies the KKT conditions anyway, report that; otherwise say
        // honestly that we did not solve it.
        ActiveSetStatus::Infeasible => {
            if certifies_primal_infeasible(prob, &sol.y, &sol.z, opts) {
                QpStatus::PrimalInfeasible
            } else {
                solved_to(err).unwrap_or(QpStatus::NumericalFailure)
            }
        }
        // Budget or breakdown: the point may still be good enough to report at
        // the acceptable tier (the IPM salvages solves this way), but it is
        // never promoted to a clean `Optimal`.
        ActiveSetStatus::MaxIter => solved_to(err).unwrap_or(QpStatus::IterationLimit),
        ActiveSetStatus::NumericalError => solved_to(err).unwrap_or(QpStatus::NumericalFailure),
    }
}

/// Does `d` actually certify that `prob` is unbounded below?
///
/// A recession direction of a convex QP must satisfy all four of:
///
/// * **zero curvature**, `Pd ≈ 0` — otherwise `½(x+td)ᵀP(x+td)` grows like `t²`
///   and the objective turns back up;
/// * **equalities preserved**, `Ad ≈ 0`;
/// * **inequalities non-increasing**, `Gd ≤ 0`, so no row is eventually violated;
/// * **box respected directionally** — a component may only move toward an
///   *infinite* bound;
///
/// and then **strict descent**, `(Px + c)ᵀd < 0`, which with `Pd ≈ 0` is `cᵀd`.
/// Together these mean `x + td` stays feasible for every `t ≥ 0` while the
/// objective decreases without bound. Anything less is not a certificate.
///
/// Tolerances are relative to `‖d‖∞` because the ray is not normalized.
fn ray_certifies_unbounded(prob: &QpProblem, d: &[f64]) -> bool {
    if d.len() != prob.n {
        return false;
    }
    let dn = d.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if !dn.is_finite() || dn == 0.0 {
        return false;
    }
    // Scale-relative slack. Deliberately looser than `tol`: the ray comes out of
    // a finite-precision null-space computation, so demanding `tol` here would
    // reject genuine certificates.
    let slack = 1e-7 * dn;

    let mut pd = vec![0.0; prob.n];
    prob.p_mul(d, &mut pd);
    if pd.iter().any(|v| v.abs() > slack) {
        return false;
    }

    let mut ad = vec![0.0; prob.m_eq()];
    for t in &prob.a {
        ad[t.row] += t.val * d[t.col];
    }
    if ad.iter().any(|v| v.abs() > slack) {
        return false;
    }

    let mut gd = vec![0.0; prob.m_ineq()];
    for t in &prob.g {
        gd[t.row] += t.val * d[t.col];
    }
    if gd.iter().any(|&v| v > slack) {
        return false;
    }

    for (i, &di) in d.iter().enumerate() {
        if di < -slack && prob.lb_of(i) > crate::qp::NEG_INF {
            return false;
        }
        if di > slack && prob.ub_of(i) < crate::qp::POS_INF {
            return false;
        }
    }

    // Strict descent. `Pd ≈ 0` was just established, so the directional
    // derivative `(Px + c)ᵀd` reduces to `cᵀd` independently of `x`.
    let slope: f64 = (0..prob.n).map(|i| prob.c[i] * d[i]).sum();
    slope < -slack
}

/// Do the multipliers `(y, z ≥ 0)` prove that `prob` has **no** feasible point?
///
/// # Why not the textbook Farkas test
///
/// The classic certificate for `{x : Ax = b, Gx ≤ h}` is `(y, z ≥ 0)` with
/// `Aᵀy + Gᵀz = 0` and `bᵀy + hᵀz < 0`, and that is what the IPM checks
/// ([`crate::ipm::detect_infeasibility`]). Applied to *these* multipliers it
/// rejects every genuine certificate the active-set engine produces, because of
/// where they come from: the l1-elastic phase-1 minimizes
/// `½xᵀPx + cᵀx + γ·(violation)`, so its stationarity condition is
/// `Px + c + Aᵀy + Gᵀz = 0` — the original objective is still in there, and
/// `q := Aᵀy + Gᵀz` settles at `−(Px + c)`, not at `0`. That residual does not
/// shrink as phase-1 converges; it only shrinks *relative to* `‖(y,z)‖ ∝ γ`.
/// On the two-row LP of gh #415 the relative residual is `1e-6` against a
/// `FARKAS_RESID_TOL` of `1e-10`, so the certificate — which is real — reads as
/// noise, and the engine's correct `Infeasible` verdict was thrown away as a
/// numerical failure.
///
/// # The test used instead
///
/// For **any** feasible `x` and any `(y, z ≥ 0)`, `yᵀ(Ax − b) = 0` and
/// `zᵀ(Gx − h) ≤ 0`, so
///
/// ```text
///     qᵀx  ≤  bᵀy + hᵀz  =:  v          where q := Aᵀy + Gᵀz
/// ```
///
/// A feasible `x` also lies in the box, so `qᵀx ≥ L := min_{lb ≤ x ≤ ub} qᵀx`,
/// which is separable: `L = Σᵢ min(qᵢ·lbᵢ, qᵢ·ubᵢ)`. Therefore
///
/// ```text
///     L > v   ⟹   no feasible point exists.
/// ```
///
/// This is a strict generalization of the Farkas test — with no finite bounds
/// it *is* the Farkas test (`L` is finite only if `q ≡ 0`, and then `L = 0 > v`
/// is exactly `v < 0`) — but on a boxed problem it does not merely *tolerate*
/// the `−(Px + c)` residual, it accounts for it exactly. The bound multipliers
/// `z_lb`/`z_ub` are deliberately not used: minimizing over the box is the same
/// deduction done optimally, and it cannot be thrown off by a phase-1 iterate's
/// noisy bound duals.
///
/// A variable whose *binding* side is infinite makes `L = −∞` and there is
/// nothing to deduce — unless its `qᵢ` is negligible, which is the one place
/// this falls back on [`FARKAS_RESID_TOL`]'s tolerance argument rather than an
/// exact bound.
///
/// Soundness is the property that matters here: a false positive is a wrong
/// verdict about the user's model, which is worse than any failure status (the
/// `DUALC1` hazard this module's docs describe). Every step above is an
/// inequality that holds for *every* feasible point, so a pass is a proof up to
/// floating point, and the `ctol` margin covers that.
fn certifies_primal_infeasible(prob: &QpProblem, y: &[f64], z: &[f64], opts: &QpOptions) -> bool {
    if y.len() != prob.m_eq() || z.len() != prob.m_ineq() {
        return false;
    }
    let dual_norm = inf_norm(y).max(inf_norm(z));
    if !dual_norm.is_finite() || dual_norm == 0.0 {
        return false;
    }
    // `z ≥ 0` is what makes `zᵀ(Gx − h) ≤ 0`; without it there is no deduction.
    let ctol = opts.infeas_tol;
    if z.iter().any(|&zi| zi < -ctol * dual_norm) {
        return false;
    }

    let mut q = vec![0.0; prob.n]; // q = Aᵀy + Gᵀz
    prob.at_mul(y, &mut q);
    prob.gt_mul(z, &mut q);
    let v = dot(&prob.b, y) + dot(&prob.h, z); // v = bᵀy + hᵀz
    if !v.is_finite() {
        return false;
    }

    // L = min over the box of qᵀx, separably.
    let resid_slack = FARKAS_RESID_TOL * dual_norm;
    let mut l = 0.0_f64;
    for (i, &qi) in q.iter().enumerate() {
        if !qi.is_finite() {
            return false;
        }
        if qi == 0.0 {
            continue;
        }
        // The minimizing corner: push `xᵢ` to its lower bound when `qᵢ > 0`, to
        // its upper bound when `qᵢ < 0`.
        let bound = if qi > 0.0 {
            prob.lb_of(i)
        } else {
            prob.ub_of(i)
        };
        if bound > crate::qp::NEG_INF && bound < crate::qp::POS_INF {
            l += qi * bound;
        } else if qi.abs() > resid_slack {
            return false; // qᵀx is unbounded below on the box — no deduction
        }
    }
    if !l.is_finite() {
        return false;
    }

    // Margin against floating-point noise, relative to the magnitudes actually
    // compared. Scaling by `dual_norm` as well keeps a certificate whose two
    // sides are both ~0 next to huge multipliers from passing on rounding.
    let mag = dual_norm.max(v.abs()).max(l.abs());
    l - v > ctol * mag
}

/// Iteration budget for the active-set engine.
///
/// The engine's own default is a flat 200, which is the IPM's budget and the
/// wrong shape for this method: an interior-point iteration count is nearly
/// independent of problem size, whereas an active-set method needs roughly one
/// iteration per active-set change, so its count grows with `n + m`. On the
/// Maros-Mészáros set the flat cap was the single largest failure class —
/// `DUALC1` (n = 9, m = 215) exhausts it and then solves *exactly*, in one
/// outer iteration, once the budget is raised.
///
/// An explicit user `max_iter` always wins; otherwise scale with the problem
/// and keep 200 as a floor for tiny instances.
fn active_set_iter_budget(opts: &QpOptions, n: usize, m: usize) -> u32 {
    const DEFAULT_IPM_MAX_ITER: usize = 200;
    const PER_DIM: usize = 10;
    if opts.max_iter != DEFAULT_IPM_MAX_ITER {
        // User set it explicitly — respect it.
        return opts.max_iter.min(u32::MAX as usize) as u32;
    }
    let scaled = n.saturating_add(m).saturating_mul(PER_DIM);
    scaled.clamp(200, 200_000) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipm::solve_qp_ipm;
    use crate::qp::Triplet;
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    /// `min (x₀−3)² + (x₁−2)²  s.t.  x₀ + x₁ ≤ 4`, written in `½xᵀPx + cᵀx`
    /// form (`P = 2I`, `c = (−6, −4)`, constant 13 dropped).
    ///
    /// Unconstrained optimum `(3, 2)` violates the row, so it binds and the
    /// solution is the projection onto `x₀ + x₁ = 4`: `x* = (2.5, 1.5)`,
    /// objective `−12.5`. Stationarity `Px + c + Gᵀz = 0` gives
    /// `(−1, −1) + z(1, 1) = 0 ⇒ z = 1`, which pins the **dual sign**: a
    /// flipped transform would clamp `z` to 0.
    fn projection_qp() -> QpProblem {
        QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-6.0, -4.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![4.0],
            lb: vec![],
            ub: vec![],
        }
    }

    #[test]
    fn analytic_qp_primal_dual_and_sign() {
        let prob = projection_qp();
        let mut mk = backend;
        let sol = solve_qp_active_set(
            &prob,
            &QpOptions::default(),
            &ActiveSetOverrides::default(),
            &mut mk,
        );

        assert_eq!(sol.status, QpStatus::Optimal, "status");
        assert!((sol.x[0] - 2.5).abs() < 1e-8, "x0 = {}", sol.x[0]);
        assert!((sol.x[1] - 1.5).abs() < 1e-8, "x1 = {}", sol.x[1]);
        assert!((sol.obj + 12.5).abs() < 1e-8, "obj = {}", sol.obj);
        // The sign check: z must be +1, not 0 and not −1.
        assert!(sol.z[0] >= -1e-12, "z must be >= 0: {}", sol.z[0]);
        assert!((sol.z[0] - 1.0).abs() < 1e-7, "z0 = {} (sign!)", sol.z[0]);
    }

    /// The Hessian **off-diagonal** path: `p_lower` stores each pair once and
    /// both forms mirror it implicitly, so a translation that dropped or
    /// double-counted the mirror would change the matrix. Cross-checked
    /// against the IPM on the same problem — the two engines must agree.
    ///
    /// `P = [[2, 1], [1, 2]]`, `c = (−4, −5)`, `x₀ + x₁ ≤ 3`. Stationarity
    /// `Px = −c` gives `x* = (1, 2)` exactly, with `xᵀPx = 14` and
    /// `cᵀx = −14`, so `obj = 7 − 14 = −7`. Getting the mirror wrong changes
    /// `P` and moves `x*` well outside these tolerances.
    ///
    /// Note the optimum sits *exactly on* `x₀ + x₁ = 3`. That is deliberate —
    /// it is the degenerate boundary case — but it means the IPM, which
    /// approaches from the interior, stops a few `1e−5` short while the
    /// active-set engine lands on the vertex exactly. So the analytic values
    /// are asserted tightly and the IPM cross-check is only asked to agree to
    /// interior-point accuracy.
    #[test]
    fn off_diagonal_hessian_matches_ipm() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![
                Triplet::new(0, 0, 2.0),
                Triplet::new(1, 0, 1.0), // the off-diagonal, stored once
                Triplet::new(1, 1, 2.0),
            ],
            c: vec![-4.0, -5.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![3.0],
            lb: vec![],
            ub: vec![],
        };
        let mut mk = backend;
        let asol = solve_qp_active_set(
            &prob,
            &QpOptions::default(),
            &ActiveSetOverrides::default(),
            &mut mk,
        );
        let isol = solve_qp_ipm(&prob, &QpOptions::default(), backend);

        assert_eq!(asol.status, QpStatus::Optimal);
        assert_eq!(isol.status, QpStatus::Optimal);

        // Analytic optimum — tight, and the real test of the mirror.
        assert!((asol.x[0] - 1.0).abs() < 1e-9, "x0 = {}", asol.x[0]);
        assert!((asol.x[1] - 2.0).abs() < 1e-9, "x1 = {}", asol.x[1]);
        assert!((asol.obj + 7.0).abs() < 1e-9, "obj = {}", asol.obj);

        // Cross-check: the IPM must land on the same point, to its own accuracy.
        for i in 0..2 {
            assert!(
                (asol.x[i] - isol.x[i]).abs() < 1e-4,
                "x{i}: active-set {} vs ipm {}",
                asol.x[i],
                isol.x[i]
            );
        }
        assert!(
            (asol.obj - isol.obj).abs() < 1e-6,
            "obj: active-set {} vs ipm {}",
            asol.obj,
            isol.obj
        );
    }

    /// Equality rows and variable bounds both translate, and the bound
    /// multipliers land in the right sign slot.
    /// `min ½(x₀² + x₁²)  s.t.  x₀ + x₁ = 2,  x₀ ≥ 1.5`.
    /// Optimum `(1.5, 0.5)` — the bound binds.
    #[test]
    fn equality_and_bounds_match_ipm() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![2.0],
            g: vec![],
            h: vec![],
            lb: vec![1.5, crate::qp::NEG_INF],
            ub: vec![],
        };
        let mut mk = backend;
        let asol = solve_qp_active_set(
            &prob,
            &QpOptions::default(),
            &ActiveSetOverrides::default(),
            &mut mk,
        );

        assert_eq!(asol.status, QpStatus::Optimal, "status");
        assert!((asol.x[0] - 1.5).abs() < 1e-7, "x0 = {}", asol.x[0]);
        assert!((asol.x[1] - 0.5).abs() < 1e-7, "x1 = {}", asol.x[1]);
        // Lower bound binds ⇒ its multiplier is the positive one.
        assert!(asol.z_lb[0] >= -1e-12, "z_lb must be >= 0");
        assert!(asol.z_ub[0] >= -1e-12, "z_ub must be >= 0");

        let isol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert!(
            (asol.obj - isol.obj).abs() < 1e-6,
            "obj: active-set {} vs ipm {}",
            asol.obj,
            isol.obj
        );
    }

    /// One row `−x ≤ −rhs` (i.e. `x ≥ rhs`) on a single variable in `[0, 10]`.
    /// Infeasible exactly when `rhs > 10`, and infeasible *because of the box* —
    /// the row alone is satisfiable, so the classic zero-residual Farkas test
    /// has nothing to say here and only the box minimization can decide it.
    fn one_row_vs_box(rhs: f64) -> QpProblem {
        QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0)],
            h: vec![-rhs],
            lb: vec![0.0],
            ub: vec![10.0],
        }
    }

    /// The box minimization must be evaluated at the corner that *minimizes*
    /// `qᵀx`, and getting that backwards certifies feasible problems as
    /// infeasible. `x ≥ 5` with `x ≤ 10` is plainly satisfiable; the multiplier
    /// `z = 1` gives `q = (−1)` and `v = −5`, so the true bound
    /// `L = q·ub = −10` correctly fails `L > v`, whereas the wrong corner
    /// (`q·lb = 0`) would "prove" a feasible problem empty.
    #[test]
    fn box_minimum_is_taken_at_the_minimizing_corner() {
        assert!(
            !certifies_primal_infeasible(&one_row_vs_box(5.0), &[], &[1.0], &QpOptions::default()),
            "x ≥ 5, x ≤ 10 is feasible — no certificate may be accepted"
        );
    }

    /// The same shape pushed past the box: `x ≥ 15` with `x ≤ 10` is empty, and
    /// `L = −10 > v = −15` proves it.
    #[test]
    fn box_only_infeasibility_is_certified() {
        assert!(certifies_primal_infeasible(
            &one_row_vs_box(15.0),
            &[],
            &[1.0],
            &QpOptions::default()
        ));
    }

    /// A feasible QP whose feasible set is the single point `{0}` — every row
    /// active, no interior, multipliers wildly non-unique (the #282 hazard in
    /// miniature). Elastic-scale multipliers on it produce `q = 0` and `v = 0`:
    /// no strict descent, so no certificate. A false positive here would be a
    /// wrong statement about the user's model.
    #[test]
    fn spurious_certificate_on_feasible_qp_is_rejected() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![-1.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(1, 0, -1.0),
                Triplet::new(2, 1, 1.0),
                Triplet::new(3, 1, -1.0),
            ],
            h: vec![0.0; 4],
            lb: vec![],
            ub: vec![],
        };
        assert!(!certifies_primal_infeasible(
            &prob,
            &[],
            &[1e6, 1e6, 1e6, 1e6],
            &QpOptions::default()
        ));
    }

    /// gh #415's multipliers, verbatim from the engine: `z = (999999, 1e6)` on
    /// `x₀ + x₁ ≤ 1` / `x₀ + x₁ ≥ 3`. They leave `q = (−1, −1) = −c`, a
    /// *relative* residual of `1e−6` against a `FARKAS_RESID_TOL` of `1e−10`,
    /// so the textbook Farkas test rejects them — which is how a certified
    /// infeasibility became `NumericalFailure`. With the box `[0, 10]²` the
    /// residual is accounted for exactly (`L = −20 > v = −2000001`).
    ///
    /// Delete the box and the same multipliers must stop being a proof: that is
    /// not a regression but the reason [`feasibility_probe`] exists.
    #[test]
    fn box_accounts_for_the_objective_residual_the_farkas_test_rejects() {
        let boxed = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![1.0, 1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 0, -1.0),
                Triplet::new(1, 1, -1.0),
            ],
            h: vec![1.0, -3.0],
            lb: vec![0.0, 0.0],
            ub: vec![10.0, 10.0],
        };
        let z = [999_999.0, 1_000_000.0];
        assert!(
            certifies_primal_infeasible(&boxed, &[], &z, &QpOptions::default()),
            "the box makes these multipliers a proof"
        );

        let free = QpProblem {
            lb: vec![],
            ub: vec![],
            ..boxed
        };
        assert!(
            !certifies_primal_infeasible(&free, &[], &z, &QpOptions::default()),
            "without a box the residual is unaccounted for — must not be trusted"
        );
    }

    /// The budget scales with `n + m` instead of sitting at the IPM's flat
    /// 200, and an explicit user setting still wins.
    #[test]
    fn iteration_budget_scales_and_respects_user() {
        let d = QpOptions::default();
        assert_eq!(d.max_iter, 200, "test assumes the IPM default is 200");
        // Tiny problem keeps the floor.
        assert_eq!(active_set_iter_budget(&d, 2, 1), 200);
        // DUALC1's shape (n=9, m=215) must clear the flat 200 that broke it.
        assert!(
            active_set_iter_budget(&d, 9, 215) > 200,
            "budget must scale past the flat cap for DUALC1's shape"
        );
        // Explicit user value wins.
        let u = QpOptions {
            max_iter: 37,
            ..QpOptions::default()
        };
        assert_eq!(active_set_iter_budget(&u, 9, 215), 37);
    }

    /// **Singular Hessian with general inequality rows** — the geometry that
    /// exposed the seeding contract. `P = diag(0, 1)` has no curvature along
    /// `x₀`, so the reduced Hessian is singular unless the working set pins
    /// that direction. Handed a feasible point with a *cold* working set the
    /// engine sees no active rows, finds zero curvature along a feasible
    /// descent direction, and certifies a spurious unbounded ray; the working
    /// set has to come from the simplex basis for this to solve.
    ///
    /// `min ½x₁² − 2x₀ − x₁  s.t.  x₀ + x₁ ≤ 2,  x₀ ≤ 1.5,  x ≥ 0`.
    /// Both rows bind at `x* = (1.5, 0.5)`: stationarity
    /// `(−2, x₁−1) + z₀(1,1) + z₁(1,0) = 0` gives `z₀ = 0.5, z₁ = 1.5 ≥ 0`,
    /// and `obj = 0.125 − 3 − 0.5 = −3.375`.
    #[test]
    fn singular_hessian_with_inequalities_solves() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(1, 1, 1.0)], // P = diag(0, 1)
            c: vec![-2.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 0, 1.0),
            ],
            h: vec![2.0, 1.5],
            lb: vec![0.0, 0.0],
            ub: vec![],
        };
        let mut mk = backend;
        let sol = solve_qp_active_set(
            &prob,
            &QpOptions::default(),
            &ActiveSetOverrides::default(),
            &mut mk,
        );

        assert_eq!(sol.status, QpStatus::Optimal, "status");
        assert!((sol.x[0] - 1.5).abs() < 1e-7, "x0 = {}", sol.x[0]);
        assert!((sol.x[1] - 0.5).abs() < 1e-7, "x1 = {}", sol.x[1]);
        assert!((sol.obj + 3.375).abs() < 1e-7, "obj = {}", sol.obj);

        let isol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert!(
            (sol.obj - isol.obj).abs() < 1e-6,
            "obj: active-set {} vs ipm {}",
            sol.obj,
            isol.obj
        );
    }
}
