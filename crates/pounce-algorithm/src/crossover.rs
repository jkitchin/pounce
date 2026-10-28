//! NLP crossover: hand a converged interior-point iterate to the
//! active-set path so the solve ends on an **exact** active set
//! (Byrd, Nocedal & Waltz, "KNITRO: An Integrated Package for
//! Nonlinear Optimization", 2006, §7; gh#612).
//!
//! # Why
//!
//! An interior-point method never puts an iterate *on* a constraint: the
//! fraction-to-boundary rule keeps every slack strictly positive, so at
//! termination "which constraints are active" is an inference from a
//! tolerance test, not a fact the solve established. Where **strict
//! complementarity holds** that inference is right and this whole phase is a
//! no-op by design. Where it fails — a weakly active bound whose slack and
//! multiplier are both `O(√μ)` — the interior solve cannot answer the
//! question at all, and three subsystems downstream are already paying for
//! that:
//!
//! 1. [`pounce_sensitivity`]'s `covariance()` classifies activity into
//!    STRONGLY ACTIVE / WEAKLY ACTIVE / **AMBIGUOUS** / UNIDENTIFIED
//!    (`docs/src/sensitivity.md`). The AMBIGUOUS class exists precisely
//!    because a barrier geometry cannot decide.
//! 2. A degenerate solution collapses the reduced Hessian, which has come
//!    back as an inertia problem repeatedly (#540, #541, #544, #592, the
//!    `feral_singular_pivot_floor` knob) and been met each time on the
//!    perturbation side.
//! 3. The active-set SQP's warm start could only come from a previous *SQP*
//!    solve (`docs/src/active-set-sqp.md`), so a sequence whose first solve
//!    wants the IPM had no way to hand off. Crossover is that missing edge:
//!    after it runs, [`crate::application::IpoptApplication::last_sqp_working_set`]
//!    returns a working set the next `algorithm=active-set-sqp` solve can
//!    consume.
//!
//! `pounce-convex` has had the LP form of this for a while
//! ([`pounce_convex::crossover`]); this is the NLP analogue, and it borrows
//! that module's two load-bearing ideas: crossover is a **bridge**, not a new
//! solver, and it is **never-regress** — the crossed-over point replaces the
//! interior one only when it is at least as good a KKT point.
//!
//! # What it does (paper §7)
//!
//! 1. The IPM terminates at `(x, y, z)` within `E_tol`.
//! 2. Estimate the active set `A` by a tolerance test on primal distance and
//!    multiplier magnitude — [`crate::sqp::classify_working_set`], the same
//!    classifier the sensitivity-corrector handoff already uses.
//! 3. Take **one EQP step over `A`** plus a line search on the penalty model.
//!    If the result satisfies the stopping tolerances, terminate. This is the
//!    common path and it solves no LPs, so on a well-behaved problem
//!    crossover costs about one iteration.
//! 4. Otherwise run the full active-set algorithm from the interior iterate,
//!    seeded with `A` and with `ν₀` a little above the largest `|multiplier|`
//!    at the interior solution.
//!
//! # Where this departs from the paper
//!
//! KNITRO's active-set path is **SLQP**: an LP phase picks the working set
//! and an EQP phase computes the step, so its step 3 is a literal EQP solve
//! and its step 4 sizes an LP trust region (their eq. 7.22) to exclude every
//! inactive constraint. POUNCE's active-set path is an ordinary line-search
//! **SQP** over `pounce-qp`'s working-set interface, so:
//!
//! - Step 3 is expressed as one `pounce-qp`
//!   [`QpSolver::solve_with_working_set`] against the NLP linearization at
//!   the interior iterate, warm-started with `A`. That call factorizes the
//!   hinted active set to recover a primal, then pivots — which is exactly
//!   "solve the EQP over `A`, and fix `A` where the tolerance test got it
//!   wrong". The paper's guarantee that step 3 avoids an LP is preserved:
//!   `pounce-qp` solves no LP either.
//! - Step 4's LP trust region has no analogue and is **not** implemented; the
//!   `ν₀` half of that setup is, since the ℓ₁ merit the SQP already carries
//!   takes exactly that parameter.
//!
//! # It runs against the *declared* bounds
//!
//! The caller hands this an
//! [`crate::sqp::IpoptNlpAdapter::new_with_declared_bounds`], not a plain
//! one, and that is load-bearing rather than tidy. `bound_relax_factor`
//! (default `1e-8`) widens every bound before the interior solve starts, so
//! a point sitting exactly on a bound the user declared is a full `1e-8`
//! *inside* the relaxed one. Measured against the relaxed bounds, a pivot
//! that lands precisely on the binding constraint reads as strictly
//! interior, and the identification step then correctly reports an empty
//! active set — crossover would run, succeed, and answer nothing. Worse, the
//! pivot itself would stop `1e-8` shy of each constraint, because against
//! the relaxed problem that point genuinely is optimal.
//!
//! So the whole phase is posed on the model as written. The consequence to
//! be aware of is that the returned point can sit on a declared bound rather
//! than inside it, which is `constr_viol_tol`-legal by construction (the
//! relaxation is capped there) and is the result being asked for.
//!
//! # Never-regress
//!
//! Crossover is a strict refinement of a solve that has already succeeded, so
//! the bar is not "did it solve" but "is this at least as good a KKT point".
//! [`accepts`] applies three gates against the interior iterate — constraint
//! violation, stationarity, and objective — and any failure returns the
//! interior solution untouched. Nothing here can turn a converged solve into
//! a failed one: on every abandonment path the caller keeps what the IPM
//! produced.
//!
//! # Cost and defaults
//!
//! Off by default (`crossover=no`). It runs strictly *after* convergence, so
//! enabling it moves no interior trajectory and needs no baseline fixture
//! sweep (contrast an initial-point or merit-function change, per
//! `CLAUDE.md`).

use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF, Number};
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_qp::{
    BoundStatus, ConsStatus, ParametricActiveSetSolver, QpError, QpOptions, QpSolver, QpStatus,
    WorkingSet,
};

use crate::sqp::iterates::SqpIterates;
use crate::sqp::line_search::l1_merit_line_search;
use crate::sqp::options::{SqpHessianSource, SqpOptions};
use crate::sqp::problem::SqpProblemSpec;
use crate::sqp::qp_assembly::SqpQpData;
use crate::sqp::result::{SqpResult, SqpStatus};
use crate::sqp::sqp_alg::{SqpAlgorithm, check_kkt};
use crate::sqp::warm_start::classify_working_set;

/// Primal tolerance used to read the active set off the **crossed-over**
/// point (as opposed to the interior one).
///
/// This is deliberately far tighter than `crossover_primal_tol`, which has to
/// tolerate the `O(√μ)` standoff of an interior iterate — typically `1e-5`.
/// After crossover the active constraints hold to machine precision, so a
/// tight test is finally meaningful, and that is the whole point of the
/// phase: the same question that could not be answered at the interior point
/// has a definite answer here. `1e-9` is generous by several orders against
/// the `~1e-16` actually observed, while still an order of magnitude below
/// the standoff it replaces.
const IDENTIFIED_PRIMAL_TOL: Number = 1e-9;

/// Slack on the never-regress comparisons, so a crossed-over point is not
/// rejected for a change at the last bit of a residual that is otherwise
/// identical.
const REGRESS_SLACK: Number = 1e-12;

/// Relative slack on the objective gate. The crossed-over point sits on the
/// active constraints exactly rather than `O(μ)` inside them, so the
/// objective is expected to move at the tolerance level; anything beyond this
/// means the active-set phase walked somewhere else and the result is
/// refused.
const OBJ_REL_SLACK: Number = 1e-6;

/// Tuning for the crossover phase. Populated from the `crossover*` options by
/// [`crate::application::IpoptApplication`].
#[derive(Debug, Clone)]
pub struct CrossoverOptions {
    /// Master switch (`crossover`). Default off.
    pub enabled: bool,
    /// Multiplier magnitude above which a row is taken active in the §7
    /// step-2 tolerance test (`crossover_mult_tol`).
    pub mult_tol: Number,
    /// Primal distance to a bound below which a row is taken binding in the
    /// §7 step-2 tolerance test (`crossover_primal_tol`).
    pub primal_tol: Number,
    /// Outer-iteration budget for the §7 step-4 fallback
    /// (`crossover_max_iter`). `0` disables step 4 entirely, leaving
    /// crossover as the one-step refinement of step 3.
    pub max_iter: u32,
}

impl Default for CrossoverOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            mult_tol: 1e-8,
            primal_tol: 1e-6,
            max_iter: 30,
        }
    }
}

/// Which of the paper's two paths produced the returned point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossoverPhase {
    /// §7 step 3 — one EQP-equivalent step over the tolerance-test active
    /// set plus a penalty line search was enough. The common path; no full
    /// active-set run.
    EqpStep,
    /// §7 step 4 — step 3 did not reach the stopping tolerances, so the full
    /// active-set SQP ran from the interior iterate.
    ActiveSet,
}

/// Why crossover did not replace the interior iterate. Reported rather than
/// swallowed: "crossover ran and declined" and "crossover never ran" are
/// different facts about a solve, and the AMBIGUOUS-activity consumers this
/// exists for need to tell them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossoverDecline {
    /// The problem has no bounds and no constraints, so there is no active
    /// set to identify.
    NothingToIdentify,
    /// The EQP-equivalent QP did not solve, and step 4 was disabled or also
    /// failed. [`CrossoverReport::eqp`] says how it ended.
    QpFailed,
    /// The penalty line search could not accept a step from the interior
    /// iterate.
    LineSearchFailed,
    /// The full active-set run did not converge within `crossover_max_iter`.
    ActiveSetNotConverged,
    /// A point was produced but it is not at least as good a KKT point as the
    /// interior iterate (see [`accepts`]).
    Regressed,
    /// The point meets the SQP's tolerances in the scaled space but not the
    /// user's `constr_viol_tol` / `dual_inf_tol` / `compl_inf_tol` on the
    /// model as declared (gh#1054). Scaling can shrink a row's violation by
    /// orders of magnitude, so a point can be a scaled KKT point and an
    /// infeasible one at once; the interior-point verdict reads user units,
    /// and so does this gate.
    UserTolerances,
    /// The solve's `max_wall_time` / `max_cpu_time` ran out before
    /// crossover finished (gh#1054). The interior solution stands, as on
    /// every other decline: crossover refines a converged solve, so running
    /// out of time is never a reason to fail it.
    TimeLimit,
}

/// What the §7 step-3 QP did (gh#1061).
///
/// [`CrossoverDecline::QpFailed`] covers every way that QP can fail to reach
/// `Optimal` — an iteration cap, an infeasible verdict, a hard linear-solver
/// error — and on a large model those call for different responses. This is
/// the detail behind it, kept on accepted reports too, so a slow crossover can
/// be attributed: working-set changes, refactorizations and rank tests against the
/// wall time they cost.
#[derive(Debug, Clone)]
pub struct EqpReport {
    /// The QP's status, or the hard error it returned instead of one.
    pub outcome: Result<QpStatus, QpError>,
    /// Working-set changes (adds + drops). `0` when the QP returned an error.
    pub working_set_changes: u32,
    /// Base-KKT refactorizations. `0` when the QP returned an error.
    pub refactors: u32,
    /// Schur-complement updates between refactorizations. `0` when the QP
    /// returned an error.
    pub schur_updates: u32,
    /// Linear-independence rank tests, counted on errors too.
    pub rank_guard_calls: u32,
    /// Wall time of the QP call.
    pub time: std::time::Duration,
    /// How that wall time split between factorization, back-solves and rank
    /// tests; the rest of `time` went outside the linear solver (gh#1061).
    pub linsol_times: pounce_qp::LinsolTimes,
}

/// What crossover did. Retrieved from
/// [`crate::application::IpoptApplication::crossover_report`].
#[derive(Debug, Clone)]
pub struct CrossoverReport {
    /// The phase that produced the accepted point; `None` when crossover
    /// declined.
    pub phase: Option<CrossoverPhase>,
    /// Why it declined; `None` when it did not.
    pub declined: Option<CrossoverDecline>,
    /// Outer iterations spent in the step-4 fallback. `0` on the step-3 path.
    pub n_iter: u32,
    /// QP subproblems solved across both phases, step 3's included.
    pub n_qp_solves: u32,
    /// Variable bounds in the identified active set (`AtLower`, `AtUpper` or
    /// `Fixed`), read off the returned point where the primal test is exact.
    /// See [`identify_at`].
    pub active_bounds: usize,
    /// Constraint rows in the identified active set (equalities included —
    /// they are unconditionally active).
    pub active_constraints: usize,
    /// Rows and bounds the §7 step-2 tolerance test called active *at the
    /// interior iterate*, before any pivoting.
    ///
    /// Compare against `active_bounds + active_constraints`, which is the
    /// same question answered at the crossed-over point. They differ exactly
    /// where the interior iterate could not support the inference — the
    /// measurement this phase exists to make.
    pub estimated_active: usize,
    /// `max(stationarity, constraint violation)` at the interior iterate.
    pub kkt_before: Number,
    /// The same at the returned point. Never worse than `kkt_before` beyond
    /// the tolerances [`accepts`] allows.
    pub kkt_after: Number,
    /// Max-norm complementarity at the returned point, measured against the
    /// **declared** bounds — see [`complementarity_at`]. `NaN` when crossover
    /// declined.
    ///
    /// This exists because the interior method's own complementarity is
    /// measured against the *relaxed* bounds, and after crossover the two
    /// frames disagree by the entire relaxation: an iterate sitting exactly
    /// on a declared bound is `bound_relax_factor` inside the relaxed one, so
    /// the relaxed reading is `|multiplier| · δ` — around `1e-8` — where the
    /// truth is zero. Reporting that as the solve's complementarity printed a
    /// converged run as `Overall NLP error` above `tol` (#646). The caller
    /// substitutes this figure when the point was accepted.
    pub compl_after: Number,
    /// The §7 step-3 QP, when it ran. `None` when crossover declined before
    /// building it.
    pub eqp: Option<EqpReport>,
}

impl CrossoverReport {
    fn declined(reason: CrossoverDecline) -> Self {
        Self {
            phase: None,
            declined: Some(reason),
            n_iter: 0,
            n_qp_solves: 0,
            active_bounds: 0,
            active_constraints: 0,
            estimated_active: 0,
            kkt_before: Number::NAN,
            kkt_after: Number::NAN,
            compl_after: Number::NAN,
            eqp: None,
        }
    }

    fn with_eqp(mut self, eqp: &EqpReport) -> Self {
        self.eqp = Some(eqp.clone());
        self
    }

    /// This report, turned into a decline for `reason` after the fact,
    /// keeping its counts and residuals — what the attempt cost and found
    /// stays visible when a later gate refuses its point.
    pub(crate) fn decline(mut self, reason: CrossoverDecline) -> Self {
        self.phase = None;
        self.declined = Some(reason);
        self
    }

    /// Did crossover replace the interior iterate?
    pub fn accepted(&self) -> bool {
        self.phase.is_some()
    }
}

/// The converged interior-point iterate, in the algorithm's (compressed,
/// scaled) space — the same space [`crate::sqp::IpoptNlpAdapter`] presents,
/// so no translation is needed between the two engines.
///
/// `lambda_x` is the **packed** bound multiplier `z_l − z_u`, matching
/// [`SqpIterates`] and [`classify_working_set`]; `lambda_g` is `[y_c ; y_d]`.
#[derive(Debug, Clone)]
pub struct CrossoverSeed {
    pub x: Vec<Number>,
    pub lambda_g: Vec<Number>,
    pub lambda_x: Vec<Number>,
}

/// Never-regress gate. The crossed-over point is accepted only when it is at
/// least as good a KKT point of the *original* NLP as the interior iterate,
/// on all three of feasibility, stationarity, and objective.
///
/// Each residual is compared against `max(interior residual, its tolerance)`
/// rather than against the interior residual alone: crossover puts the
/// iterate *on* the active constraints, which can nudge a residual that was
/// `1e-12` up to `1e-10` while the point is unambiguously better identified.
/// Refusing that would make the gate reject exactly the cases the phase
/// exists for. What it still refuses is a residual that crosses its own
/// tolerance, which is the thing that would turn a converged solve into a
/// misreported one.
pub fn accepts(
    before: (Number, Number, Number),
    after: (Number, Number, Number),
    sqp_opts: &SqpOptions,
) -> bool {
    let (stat_b, viol_b, obj_b) = before;
    let (stat_a, viol_a, obj_a) = after;
    if !(stat_a.is_finite() && viol_a.is_finite() && obj_a.is_finite()) {
        return false;
    }
    let stat_tol = sqp_opts.tol.min(sqp_opts.dual_inf_tol);
    if stat_a > stat_b.max(stat_tol) + REGRESS_SLACK {
        return false;
    }
    if viol_a > viol_b.max(sqp_opts.constr_viol_tol) + REGRESS_SLACK {
        return false;
    }
    // Objective: a *decrease* is always fine (crossover found a better point
    // on the same active set); an increase is bounded relative to the
    // interior objective's own magnitude.
    let obj_slack = OBJ_REL_SLACK * obj_b.abs().max(1.0);
    obj_a <= obj_b + obj_slack
}

/// Max-norm complementarity `max_i |slack_i · multiplier_i|` at a
/// crossed-over point, in the frame crossover actually solved in.
///
/// Two things make this different from [`crate::ipopt_cq::IpoptCq`]'s
/// complementarity, and both are deliberate.
///
/// **The bounds are the declared ones.** `nlp` here is the adapter built by
/// `new_with_declared_bounds`, so `xl`/`xu`/`bl_c`/`bu_c` are the box the
/// user wrote rather than the `bound_relax_factor`-widened one the interior
/// iteration ran against. Crossover's whole job is to put the iterate *on*
/// the active constraints of the problem as posed; measured against the
/// relaxed bounds that same point reads `|multiplier| · δ` — the relaxation
/// times the dual, `~1e-8` for a unit multiplier — which is not a residual of
/// anything, just the width of an internal safeguard (#646).
///
/// **The slacks are raw.** The CQ floors a slack that falls below
/// `eps·min(1,μ)` up to about `μ/z`, which keeps the barrier's `Σ = V/S`
/// finite during the iteration. At a purified point the active slacks are
/// *exactly* zero and that floor would put `μ/z ≈ 1e-9` back — reintroducing,
/// as a reporting artifact, the very quantity crossover removed.
///
/// Sign conventions follow the rest of this module: `λ_x = z_l − z_u`
/// (positive at a lower bound), while a row's `λ_g` is **negative** at its
/// lower bound, because the bound block enters stationarity negated.
fn complementarity_at(
    x: &[Number],
    c_vals: &[Number],
    lambda_x: &[Number],
    lambda_g: &[Number],
    xl: &[Number],
    xu: &[Number],
    bl_c: &[Number],
    bu_c: &[Number],
) -> Number {
    let mut worst = 0.0_f64;
    // A point may sit a rounding step outside a bound; that is constraint
    // violation, which `check_kkt` already reports. Clamping at zero here
    // keeps it from re-entering as a *negative* complementarity.
    let mut take = |slack: Number, mult: Number| {
        worst = worst.max((slack.max(0.0) * mult).abs());
    };
    for i in 0..x.len() {
        if xl[i] > NLP_LOWER_BOUND_INF {
            take(x[i] - xl[i], lambda_x[i].max(0.0));
        }
        if xu[i] < NLP_UPPER_BOUND_INF {
            take(xu[i] - x[i], (-lambda_x[i]).max(0.0));
        }
    }
    for i in 0..c_vals.len() {
        if bl_c[i] > NLP_LOWER_BOUND_INF {
            take(c_vals[i] - bl_c[i], (-lambda_g[i]).max(0.0));
        }
        if bu_c[i] < NLP_UPPER_BOUND_INF {
            take(bu_c[i] - c_vals[i], lambda_g[i].max(0.0));
        }
    }
    worst
}

/// Count the active entries of a working set, split bounds / rows.
fn count_active(w: &WorkingSet) -> (usize, usize) {
    let bounds = w
        .bounds
        .iter()
        .filter(|b| !matches!(b, BoundStatus::Inactive))
        .count();
    let rows = w
        .constraints
        .iter()
        .filter(|c| !matches!(c, ConsStatus::Inactive))
        .count();
    (bounds, rows)
}

/// Run the crossover phase (paper §7 steps 2-4).
///
/// `seed` is the converged interior iterate. `make_backend` supplies the
/// sparse symmetric linear solver for the active-set engine — the same
/// factory the IPM used, so crossover inherits the caller's `linear_solver`
/// choice. `make_sqp` builds the step-4 driver; it is a closure rather than a
/// value because step 4 needs its own iteration budget and `ν₀`, and it is
/// never called at all on the step-3 path.
///
/// Returns the report always, and the replacement solution only when it was
/// accepted.
pub fn run<N, B, S>(
    nlp: &mut N,
    seed: &CrossoverSeed,
    opts: &CrossoverOptions,
    sqp_opts: &SqpOptions,
    qp_opts: &QpOptions,
    mut make_backend: B,
    mut make_sqp: S,
) -> (CrossoverReport, Option<SqpResult>)
where
    N: SqpProblemSpec,
    B: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    S: FnMut(SqpOptions) -> Option<SqpAlgorithm>,
{
    let n = nlp.n();
    let m = nlp.m();
    let (xl, xu) = nlp.variable_bounds();
    let (bl_c, bu_c) = nlp.constraint_bounds();

    // Nothing to identify: no general rows and no finite bound anywhere. The
    // interior iterate is already an unconstrained stationary point and its
    // active set is empty by construction.
    let any_bound = xl
        .iter()
        .any(|&v| v > NLP_LOWER_BOUND_INF)
        .then_some(true)
        .or_else(|| xu.iter().any(|&v| v < NLP_UPPER_BOUND_INF).then_some(true))
        .unwrap_or(false);
    if m == 0 && !any_bound {
        return (
            CrossoverReport::declined(CrossoverDecline::NothingToIdentify),
            None,
        );
    }

    // ---- §7 step 1: residuals at the interior iterate ----
    let f_curr = nlp.eval_f(&seed.x);
    let c_vals = nlp.eval_c(&seed.x);
    let grad_f = nlp.eval_grad_f(&seed.x);
    let jac_c = nlp.eval_jac_c(&seed.x);

    let mut iter = SqpIterates {
        x: seed.x.clone(),
        lambda_g: seed.lambda_g.clone(),
        lambda_x: seed.lambda_x.clone(),
        working: None,
    };
    let kkt_before = check_kkt(
        n, m, &iter, &grad_f, &c_vals, &bl_c, &bu_c, &xl, &xu, &jac_c,
    );
    let before = (kkt_before.stationarity, kkt_before.constr_viol, f_curr);

    // ---- §7 step 2: estimate the active set by the tolerance test ----
    let m_eq = m_eq_count(&bl_c, &bu_c);
    let working = classify_working_set(
        &seed.lambda_x,
        &seed.lambda_g,
        m_eq,
        &seed.x,
        &xl,
        &xu,
        &c_vals,
        &bl_c,
        &bu_c,
        opts.mult_tol,
        opts.primal_tol,
    );
    let (est_bounds, est_rows) = count_active(&working);
    let estimated_active = est_bounds + est_rows;

    // `ν₀` a little above the largest |multiplier| at the interior solution
    // (paper §7). The ℓ₁ merit's own Han-Powell update only ever raises ν, so
    // seeding it here is what keeps the first crossover step from being
    // rejected by a penalty that has not yet caught up with the duals the IPM
    // already found.
    let mult_inf = seed
        .lambda_g
        .iter()
        .chain(seed.lambda_x.iter())
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    let nu0 = (mult_inf + sqp_opts.l1_penalty_safety)
        .max(sqp_opts.l1_penalty)
        .min(sqp_opts.l1_penalty_max);

    // ---- §7 step 3: one EQP-equivalent step over the estimated set ----
    let mut n_qp_solves = 0_u32;
    let hessian_inertia = match sqp_opts.hessian {
        SqpHessianSource::Exact => pounce_qp::HessianInertia::Indefinite,
        _ => pounce_qp::HessianInertia::Psd,
    };
    let hess_lag = nlp.eval_hess_lag(&seed.x, &seed.lambda_g);
    let qp_data = SqpQpData::build(
        &seed.x,
        &grad_f,
        &c_vals,
        &bl_c,
        &bu_c,
        &xl,
        &xu,
        jac_c.clone(),
        hess_lag,
        hessian_inertia,
    );
    let qp = qp_data.as_qp();
    // The step-3 QP gets whatever the solve has left (gh#1054); a QP that
    // stops on it comes back `TimeLimit`, which is a decline like any other.
    if sqp_opts
        .deadline
        .as_ref()
        .is_some_and(|d| d.exceeded().is_some())
    {
        return (CrossoverReport::declined(CrossoverDecline::TimeLimit), None);
    }
    let eqp_opts = QpOptions {
        time_limit: sqp_opts
            .deadline
            .as_ref()
            .map(|d| crate::sqp::sqp_alg::deadline_remaining(d).0)
            .map_or(qp_opts.time_limit, |left| {
                Some(qp_opts.time_limit.map_or(left, |own| own.min(left)))
            }),
        // The SQP driver's rule for Schur updates (gh#1054, gh#1083), so the
        // step-3 QP of a large model does not refactor per pivot. It is
        // warm-started from the estimated working set.
        use_schur_updates: sqp_opts
            .step_qp_schur_updates(n + m, Some(hessian_inertia))
            .unwrap_or(qp_opts.use_schur_updates),
        ..qp_opts.clone()
    };
    let mut qp_solver = ParametricActiveSetSolver::new(make_backend());
    let eqp_started = std::time::Instant::now();
    let eqp = qp_solver.solve_with_working_set(&qp, &working, &eqp_opts);
    n_qp_solves += 1;
    let eqp_report = EqpReport {
        outcome: eqp.as_ref().map(|sol| sol.status).map_err(Clone::clone),
        working_set_changes: eqp.as_ref().map_or(0, |s| s.stats.n_working_set_changes),
        refactors: eqp.as_ref().map_or(0, |s| s.stats.n_refactor),
        schur_updates: eqp.as_ref().map_or(0, |s| s.stats.n_schur_updates),
        rank_guard_calls: qp_solver.rank_guard_calls(),
        time: eqp_started.elapsed(),
        linsol_times: qp_solver.linsol_time_split(),
    };
    tracing::debug!(target: "pounce::crossover", "crossover step 3 QP: {eqp_report:?}");

    let mut step3_failure = match &eqp {
        Ok(sol) if sol.status == QpStatus::TimeLimit => CrossoverDecline::TimeLimit,
        _ => CrossoverDecline::QpFailed,
    };
    if let Ok(sol) = eqp
        && sol.status == QpStatus::Optimal
    {
        // Line search on the penalty model (paper §7 step 3). No
        // second-order correction: this is a single refinement step at a
        // point that is already converged, and the Maratos effect the SOC
        // exists for is a *far*-from-solution phenomenon.
        let ls = l1_merit_line_search(
            nlp,
            &seed.x,
            &sol.x,
            &sol.lambda_g,
            &grad_f,
            f_curr,
            &c_vals,
            &bl_c,
            &bu_c,
            &xl,
            &xu,
            nu0,
            sqp_opts,
            None,
        );
        if ls.success {
            let mut cand = SqpIterates {
                x: ls.x_new.clone(),
                lambda_g: seed.lambda_g.clone(),
                lambda_x: seed.lambda_x.clone(),
                working: Some(sol.working.clone()),
            };
            // Interpolate the duals with the accepted step length, exactly
            // as the SQP driver does — the multipliers must describe the
            // step that was actually taken.
            for (l, &lq) in cand.lambda_g.iter_mut().zip(sol.lambda_g.iter()) {
                *l = (1.0 - ls.alpha) * *l + ls.alpha * lq;
            }
            for (l, &lq) in cand.lambda_x.iter_mut().zip(sol.lambda_x.iter()) {
                *l = (1.0 - ls.alpha) * *l + ls.alpha * lq;
            }
            let grad_new = nlp.eval_grad_f(&cand.x);
            let jac_new = nlp.eval_jac_c(&cand.x);
            let kkt_after = check_kkt(
                n, m, &cand, &grad_new, &ls.c_new, &bl_c, &bu_c, &xl, &xu, &jac_new,
            );
            let after = (kkt_after.stationarity, kkt_after.constr_viol, ls.f_new);
            // "If that satisfies the stopping tolerances, terminate" — the
            // paper's step 3 exit, and the reason the common case costs one
            // iteration and no LP.
            let stat_tol = sqp_opts.tol.min(sqp_opts.dual_inf_tol);
            let within_tol = kkt_after.stationarity <= stat_tol
                && kkt_after.constr_viol <= sqp_opts.constr_viol_tol;
            if within_tol && accepts(before, after, sqp_opts) {
                let identified = identify_at(nlp, &cand.x, m_eq, &xl, &xu, &bl_c, &bu_c);
                let (active_bounds, active_constraints) = count_active(&identified);
                let compl_after = complementarity_at(
                    &cand.x,
                    &ls.c_new,
                    &cand.lambda_x,
                    &cand.lambda_g,
                    &xl,
                    &xu,
                    &bl_c,
                    &bu_c,
                );
                return (
                    CrossoverReport {
                        phase: Some(CrossoverPhase::EqpStep),
                        declined: None,
                        n_iter: 1,
                        n_qp_solves,
                        active_bounds,
                        active_constraints,
                        estimated_active,
                        kkt_before: kkt_before.stationarity.max(kkt_before.constr_viol),
                        kkt_after: kkt_after.stationarity.max(kkt_after.constr_viol),
                        compl_after,
                        eqp: Some(eqp_report),
                    },
                    Some(SqpResult {
                        x: cand.x,
                        lambda_g: cand.lambda_g,
                        lambda_x: cand.lambda_x,
                        obj: ls.f_new,
                        status: SqpStatus::Optimal,
                        n_iter: 1,
                        n_qp_solves,
                        n_qp_working_set_changes: sol.stats.n_working_set_changes,
                        final_stationarity: kkt_after.stationarity,
                        final_constr_viol: kkt_after.constr_viol,
                        working_set: Some(identified),
                    }),
                );
            }
            step3_failure = CrossoverDecline::Regressed;
        } else {
            step3_failure = CrossoverDecline::LineSearchFailed;
        }
    }

    // ---- §7 step 4: full active-set run from the interior iterate ----
    if step3_failure == CrossoverDecline::TimeLimit
        || sqp_opts
            .deadline
            .as_ref()
            .is_some_and(|d| d.exceeded().is_some())
    {
        return (
            CrossoverReport::declined(CrossoverDecline::TimeLimit).with_eqp(&eqp_report),
            None,
        );
    }
    if opts.max_iter == 0 {
        return (
            CrossoverReport::declined(step3_failure).with_eqp(&eqp_report),
            None,
        );
    }
    let step4_opts = SqpOptions {
        max_iter: opts.max_iter,
        l1_penalty: nu0,
        ..sqp_opts.clone()
    };
    let Some(mut sqp) = make_sqp(step4_opts) else {
        return (
            CrossoverReport::declined(step3_failure).with_eqp(&eqp_report),
            None,
        );
    };
    iter.working = Some(working);
    let res = match sqp.optimize_with_warm_start(nlp, Some(iter)) {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(target: "pounce::crossover", "crossover step 4 failed: {e:?}");
            return (
                CrossoverReport::declined(step3_failure).with_eqp(&eqp_report),
                None,
            );
        }
    };
    n_qp_solves += res.n_qp_solves;
    if res.status != SqpStatus::Optimal {
        let reason = if matches!(
            res.status,
            SqpStatus::WallTimeExceeded | SqpStatus::CpuTimeExceeded
        ) {
            CrossoverDecline::TimeLimit
        } else {
            CrossoverDecline::ActiveSetNotConverged
        };
        return (
            CrossoverReport::declined(reason).with_eqp(&eqp_report),
            None,
        );
    }
    let after = (res.final_stationarity, res.final_constr_viol, res.obj);
    if !accepts(before, after, sqp_opts) {
        return (
            CrossoverReport::declined(CrossoverDecline::Regressed).with_eqp(&eqp_report),
            None,
        );
    }
    let identified = identify_at(nlp, &res.x, m_eq, &xl, &xu, &bl_c, &bu_c);
    let (active_bounds, active_constraints) = count_active(&identified);
    let compl_after = {
        let c_final = nlp.eval_c(&res.x);
        complementarity_at(
            &res.x,
            &c_final,
            &res.lambda_x,
            &res.lambda_g,
            &xl,
            &xu,
            &bl_c,
            &bu_c,
        )
    };
    let mut res = res;
    res.working_set = Some(identified);
    let report = CrossoverReport {
        phase: Some(CrossoverPhase::ActiveSet),
        declined: None,
        n_iter: res.n_iter,
        n_qp_solves,
        active_bounds,
        active_constraints,
        estimated_active,
        kkt_before: kkt_before.stationarity.max(kkt_before.constr_viol),
        kkt_after: res.final_stationarity.max(res.final_constr_viol),
        compl_after,
        eqp: Some(eqp_report),
    };
    (report, Some(res))
}

/// Read the active set off a crossed-over point.
///
/// Not the same thing as the working set `pounce-qp` returns, and the
/// difference matters. The QP's working set answers "which rows did I have
/// to constrain to compute this step" — a row the step lands *exactly* on
/// without ever being blocked by it is legitimately absent from it. That is
/// precisely the weakly-active case (multiplier zero, constraint binding),
/// i.e. the case crossover exists to resolve, so reporting the QP's set as
/// the identified active set would report `Inactive` for the one row the
/// user ran crossover to ask about.
///
/// It is also not [`classify_working_set`], which is the *interior*-iterate
/// test: there the primal distance is `O(√μ)` and unusable, so multiplier
/// sign carries the decision. Here the situation is inverted. The primal
/// test is exact — that is what the phase bought — while the multiplier at a
/// weakly active constraint is zero to within rounding, so its sign is
/// noise. Deciding activity on a `−1e-17` multiplier would discard exactly
/// the constraints crossover was run to identify. Multiplier sign is
/// consulted only where the primal test genuinely cannot choose a side:
/// a point tight against *both* of two distinct bounds.
///
/// `pounce-qp` treats an incoming working set as a hint and prunes to a
/// maximal linearly independent subset, so publishing this as the
/// warm-start set is safe even where the tight test admits a dependent row.
#[allow(clippy::too_many_arguments)]
fn identify_at<N: SqpProblemSpec>(
    nlp: &mut N,
    x: &[Number],
    m_eq: usize,
    xl: &[Number],
    xu: &[Number],
    bl_c: &[Number],
    bu_c: &[Number],
) -> WorkingSet {
    let c_vals = nlp.eval_c(x);
    // Relative tolerance: a constraint whose bound is `1e6` holds to
    // machine precision at about `1e-10` absolute, which a fixed `1e-9`
    // floor would only just admit and a slightly larger bound would not.
    let tight = |v: Number, bound: Number| -> bool {
        (v - bound).abs() <= IDENTIFIED_PRIMAL_TOL * bound.abs().max(1.0)
    };

    let mut bounds = Vec::with_capacity(xl.len());
    for i in 0..xl.len() {
        let lo_fin = xl[i] > NLP_LOWER_BOUND_INF;
        let up_fin = xu[i] < NLP_UPPER_BOUND_INF;
        let at_lo = lo_fin && tight(x[i], xl[i]);
        let at_up = up_fin && tight(x[i], xu[i]);
        bounds.push(if at_lo && at_up {
            // Both bounds tight: either genuinely fixed, or a box so
            // narrow the two are indistinguishable at this tolerance.
            BoundStatus::Fixed
        } else if at_lo {
            BoundStatus::AtLower
        } else if at_up {
            BoundStatus::AtUpper
        } else {
            BoundStatus::Inactive
        });
    }

    let mut constraints = Vec::with_capacity(bl_c.len());
    for i in 0..bl_c.len() {
        if i < m_eq {
            constraints.push(ConsStatus::Equality);
            continue;
        }
        let lo_fin = bl_c[i] > NLP_LOWER_BOUND_INF;
        let up_fin = bu_c[i] < NLP_UPPER_BOUND_INF;
        let g = c_vals.get(i).copied().unwrap_or(0.0);
        let at_lo = lo_fin && tight(g, bl_c[i]);
        let at_up = up_fin && tight(g, bu_c[i]);
        constraints.push(if at_lo && at_up {
            // A range row pinched to a point, or tight against both ends of
            // a very narrow range: active either way, so report it as the
            // equality it effectively is. The one place a side genuinely
            // cannot be read off the primal.
            ConsStatus::Equality
        } else if at_lo {
            ConsStatus::AtLower
        } else if at_up {
            ConsStatus::AtUpper
        } else {
            ConsStatus::Inactive
        });
    }
    WorkingSet {
        bounds,
        constraints,
    }
}

/// How many leading rows are equalities.
///
/// The IPM-side adapter orders constraints `[c ; d]` — equalities first — so
/// this is a prefix count, and [`classify_working_set`] takes it as such.
/// Counting rather than asking the adapter keeps this function usable against
/// any [`SqpProblemSpec`], including the hand-built ones in the tests.
fn m_eq_count(bl_c: &[Number], bu_c: &[Number]) -> usize {
    bl_c.iter()
        .zip(bu_c.iter())
        .take_while(|(lo, hi)| lo == hi)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> SqpOptions {
        SqpOptions {
            tol: 1e-8,
            dual_inf_tol: 1e-4,
            constr_viol_tol: 1e-6,
            ..SqpOptions::default()
        }
    }

    #[test]
    fn m_eq_count_takes_the_leading_equality_block_only() {
        // [eq, eq, ineq, eq] — the trailing equality is NOT counted: the
        // adapter's layout guarantees equalities are a prefix, and a
        // mid-vector match would mean the caller broke that contract.
        let bl = [0.0, 0.0, -1.0, 2.0];
        let bu = [0.0, 0.0, 1.0, 2.0];
        assert_eq!(m_eq_count(&bl, &bu), 2);
    }

    #[test]
    fn accepts_lets_a_residual_move_inside_its_own_tolerance() {
        let o = opts();
        // Stationarity rises 1e-12 → 1e-10 but stays far inside dual_inf_tol:
        // exactly the "now sitting ON the constraint" case crossover creates.
        assert!(accepts((1e-12, 1e-12, 1.0), (1e-10, 1e-10, 1.0), &o));
    }

    #[test]
    fn accepts_refuses_a_residual_that_crosses_its_tolerance() {
        let o = opts();
        assert!(!accepts((1e-12, 1e-12, 1.0), (1e-3, 1e-12, 1.0), &o));
        assert!(!accepts((1e-12, 1e-12, 1.0), (1e-12, 1e-4, 1.0), &o));
    }

    #[test]
    fn accepts_refuses_an_objective_that_walked_away() {
        let o = opts();
        // A point that is KKT-clean but at a different, worse optimum.
        assert!(!accepts((1e-12, 1e-12, 1.0), (1e-12, 1e-12, 1.5), &o));
        // A decrease is always fine.
        assert!(accepts((1e-12, 1e-12, 1.0), (1e-12, 1e-12, 0.5), &o));
    }

    #[test]
    fn accepts_refuses_non_finite_residuals() {
        let o = opts();
        assert!(!accepts((1e-12, 1e-12, 1.0), (Number::NAN, 1e-12, 1.0), &o));
        assert!(!accepts(
            (1e-12, 1e-12, 1.0),
            (1e-12, 1e-12, Number::INFINITY),
            &o
        ));
    }

    #[test]
    fn count_active_splits_bounds_and_rows() {
        let w = WorkingSet {
            bounds: vec![
                BoundStatus::AtLower,
                BoundStatus::Inactive,
                BoundStatus::Fixed,
            ],
            constraints: vec![
                ConsStatus::Equality,
                ConsStatus::Inactive,
                ConsStatus::AtUpper,
            ],
        };
        assert_eq!(count_active(&w), (2, 2));
    }

    #[test]
    fn declined_report_is_not_accepted() {
        let r = CrossoverReport::declined(CrossoverDecline::NothingToIdentify);
        assert!(!r.accepted());
        assert_eq!(r.declined, Some(CrossoverDecline::NothingToIdentify));
    }
}
