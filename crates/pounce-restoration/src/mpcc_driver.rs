//! Driver for models with declared complementarity pairs: runs the stages
//! [`pounce_algorithm::mpcc`] prescribes on a [`MpccTnlp`] — the Scholtes
//! continuation and the exact product both, keeping the strongly stationary
//! answer with the lower objective; the ℓ₁ penalty only if neither answers —
//! and reports what it found ([`Reported`]).
//!
//! The policy — stage order, the relaxation schedule, the warm-start
//! options, what counts as an answer — lives in `pounce_algorithm::mpcc`.
//! This module runs the solves, the split the second-opinion ladder uses.
//!
//! Three things a library driver has to get right:
//!
//! * **Only the reported answer reaches the caller's model.** The wrapper
//!   defers `finalize_solution`; every stage's result is captured from it,
//!   and the accepted one (or, if none is accepted, the exact-product stage's,
//!   which is what a plain solve of the lowered model would have reported —
//!   never a relaxed stage's, whose success is on the relaxation) is
//!   forwarded once, at the end.
//! * **Options are restored to set-ness.** Every option a stage writes is
//!   snapshotted before the first stage with `OptionsList::snapshot_entry`
//!   and restored after the last, so a knob the caller never set is unset
//!   again rather than written back as whatever it resolved to. That includes
//!   the [`SAFE_DEFAULTS`] every stage runs under (no bound relaxation),
//!   which apply only where the caller did not set the option.
//! * **The caller wires restoration.** No stage changes an option the
//!   restoration sub-solver is built from, so the provider the caller set is
//!   used as is.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_algorithm::mpcc::{
    BCheck, BVerdict, Certificate, CertificateInput, EXACT_STAGE_TAU, MpccStage, Reported,
    SAFE_DEFAULTS, SCHOLTES_SCHEDULE, STAGES, StationarityClass, Strategy, TOUCHED_OPTIONS,
    accepts, answers, branch_bounds, budgeted_max_iter, certify, check_b_stationarity, compared,
    converged, exact_stage_budget, finish_budget, kept_if_nothing_better, warm_start_assignments,
};
use pounce_common::types::{Index, Number};
use pounce_nlp::SolveStatistics;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::mpcc::{ComplementarityPair, Lowering, MpccTnlp, StartOverride};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{BoundsInfo, IndexStyle, IpoptCq, IpoptData, SparsityRequest, TNLP};

/// One lowered solve.
#[derive(Debug, Clone)]
pub struct MpccSolveRecord {
    pub stage: MpccStage,
    /// `None` for an exact-product solve, `Some(τ)` for a relaxed one.
    pub tau: Option<Number>,
    pub status: ApplicationReturnStatus,
    pub iterations: usize,
    pub max_abs_product: Number,
}

/// What the driver reports.
#[derive(Debug, Clone)]
pub struct MpccOutcome {
    /// The status of the solve whose answer was forwarded to the model.
    pub status: ApplicationReturnStatus,
    /// Its statistics (iteration count of that solve alone; see
    /// [`Self::total_iterations`] for the whole run).
    pub statistics: SolveStatistics,
    /// The stage that produced an accepted (strongly stationary) answer, if
    /// any. `None` when the reported point is a fallback: see
    /// [`Self::reported`] and [`Self::fallback`].
    pub accepted_by: Option<MpccStage>,
    /// The stage whose weakly stationary point is reported when no stage
    /// answered, if that is what was reported.
    pub fallback: Option<MpccStage>,
    /// Which point was reported, and on what grounds.
    pub reported: Reported,
    /// The `complementarity_strategy` the run used.
    pub strategy: Strategy,
    /// Every lowered solve, in order.
    pub solves: Vec<MpccSolveRecord>,
    pub total_iterations: usize,
    /// The forwarded point and its complementarity violation (`|G·H|`, or the
    /// positive part on a mixed pair's rows).
    pub x: Vec<Number>,
    pub max_abs_product: Number,
    /// Which stationarity class the forwarded point satisfies
    /// ([`pounce_algorithm::mpcc::certify`]); `None` only if the model could
    /// not be evaluated there.
    pub certificate: Option<Certificate>,
    /// The B-stationarity check of the forwarded point
    /// ([`pounce_algorithm::mpcc::check_b_stationarity`]), run when the point
    /// is feasible and not certified S (S implies B). `None` when it did not
    /// run there.
    pub b_check: Option<BCheck>,
    /// Branch NLPs that moved the point off a non-B-stationary one.
    pub escapes: usize,
}

impl MpccOutcome {
    /// The solve report's `complementarity` block. One conversion, shared by
    /// the CLI's `--json-output` and Python's `report_path`, so the two
    /// frontends cannot describe the same run differently.
    pub fn report_info(&self) -> pounce_solve_report::ComplementarityInfo {
        use pounce_algorithm::mpcc::StationarityClass;
        use pounce_solve_report::{
            ComplementarityCertificateInfo, ComplementarityInfo, ComplementaritySolveInfo,
        };
        // JSON has no infinity; a quantity that was not measured is `null`.
        let finite = |v: Number| v.is_finite().then_some(v);
        ComplementarityInfo {
            strategy: self.strategy.name().to_string(),
            accepted_by: self.accepted_by.map(|s| s.label().to_string()),
            fallback: self.fallback.map(|s| s.label().to_string()),
            reported: match self.reported {
                Reported::Answer(_) => "answer",
                Reported::WeakFallback(_) => "weak_fallback",
                Reported::FeasibleUncertified(_) => "feasible_uncertified",
                Reported::Verdict => "verdict",
            }
            .to_string(),
            summary: pounce_algorithm::mpcc::describe_outcome(
                self.reported,
                self.solves.len(),
                self.total_iterations,
                self.max_abs_product,
                self.certificate.as_ref(),
            ),
            solves: self
                .solves
                .iter()
                .map(|r| ComplementaritySolveInfo {
                    stage: r.stage.label().to_string(),
                    tau: r.tau,
                    status: r.status.upstream_name().to_string(),
                    iterations: r.iterations,
                    max_abs_product: finite(r.max_abs_product),
                })
                .collect(),
            total_iterations: self.total_iterations,
            max_abs_product: finite(self.max_abs_product),
            certificate: self
                .certificate
                .as_ref()
                .map(|c| ComplementarityCertificateInfo {
                    class: match c.class {
                        StationarityClass::S => "S",
                        StationarityClass::W => "W",
                        StationarityClass::None => "none",
                        StationarityClass::NotSearched => "not_searched",
                    }
                    .to_string(),
                    n_pairs: c.n_pairs,
                    n_biactive: c.n_biactive,
                    n_neither: c.n_neither,
                    multipliers_unique: c.multipliers_unique,
                    residual: finite(c.residual),
                    resid_tol: finite(c.resid_tol),
                    tight: c.tight,
                    primal_violation: finite(c.primal_violation),
                    feasible: c.feasible,
                }),
        }
    }
}

#[derive(Clone)]
struct Captured {
    status: ApplicationReturnStatus,
    solver_return: SolverReturn,
    statistics: SolveStatistics,
    x: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    lambda: Vec<Number>,
    max_abs_product: Number,
}

impl Captured {
    fn warm(&self) -> StartOverride {
        StartOverride {
            x: self.x.clone(),
            z_l: self.z_l.clone(),
            z_u: self.z_u.clone(),
            lambda: self.lambda.clone(),
        }
    }
}

/// Solve `inner` with `pairs` declared complementary; see the module docs.
/// `report` receives one line per stage transition (what the console shows
/// at `print_level > 0`). Errors only if the pairs cannot be lowered — the
/// reason is the [`MpccTnlp::new`] refusal.
pub fn solve_mpcc(
    app: &mut IpoptApplication,
    inner: Rc<RefCell<dyn TNLP>>,
    pairs: Vec<ComplementarityPair>,
    report: &mut dyn FnMut(&str),
) -> Result<MpccOutcome, String> {
    let wrapper = Rc::new(RefCell::new(MpccTnlp::new(
        Rc::clone(&inner),
        pairs,
        Lowering::Eq,
    )?));
    wrapper.borrow_mut().set_defer_inner_finalize(true);
    let lowered: Rc<RefCell<dyn TNLP>> = wrapper.clone();

    let tol = app
        .options()
        .get_numeric_value("constr_viol_tol", "")
        .map(|(v, _)| v)
        .unwrap_or(1e-4);
    let snapshot: Vec<_> = TOUCHED_OPTIONS
        .iter()
        .map(|&k| (k, app.options().snapshot_entry(k)))
        .collect();
    let restore = |app: &mut IpoptApplication| {
        for (k, s) in &snapshot {
            app.options_mut().restore_entry(k, s.clone());
        }
    };
    // The caller's options plus the safe defaults it did not set itself:
    // what every stage starts from.
    let user_set: Vec<&str> = SAFE_DEFAULTS
        .iter()
        .map(|(k, _)| *k)
        .filter(|k| app.options().names().any(|n| n.eq_ignore_ascii_case(k)))
        .collect();
    let baseline = |app: &mut IpoptApplication| {
        restore(app);
        for (k, v) in SAFE_DEFAULTS {
            if !user_set.contains(&k) {
                let _ = app.options_mut().read_from_str(&format!("{k} {v}\n"), true);
            }
        }
    };

    // The caller's own iteration limit: a budgeted stage never exceeds it.
    let caller_max_iter = app
        .options()
        .get_integer_value("max_iter", "")
        .map(|(v, _)| v)
        .unwrap_or(3000);
    let set_budget = |app: &mut IpoptApplication, budget: usize| {
        let it = budgeted_max_iter(budget, caller_max_iter);
        let _ = app
            .options_mut()
            .read_from_str(&format!("max_iter {it}\n"), true);
    };

    let mut solves: Vec<MpccSolveRecord> = Vec::new();
    // The exact-product stage's result: what a plain solve of the lowered
    // model says, and so the verdict reported when no stage answers. A
    // relaxed solve's "success" is a success on the relaxation, never on the
    // model.
    let mut exact: Option<Captured> = None;
    // The first stage that returned a point, standing in for `exact` if
    // that stage was refused before iteration 0.
    let mut first: Option<Captured> = None;
    // A lowered solve the caller's intermediate callback stopped: the
    // ladder ends there and that solve is reported. Set by `run` itself, so
    // no solve can stop unnoticed; every caller of `run` leaves on it
    // (`halted()`) before starting another.
    let stopped: RefCell<Option<Captured>> = RefCell::new(None);
    // The best S answer so far, with the model's objective there.
    let mut accepted: Option<(MpccStage, Captured, Option<Certificate>, Number)> = None;
    // The first stage that certified only W: the answer if no later stage
    // certifies S.
    let mut weak: Option<(MpccStage, Captured, Option<Certificate>)> = None;
    // The best S answer whose residual is only inside the `sqrt(tol)` band
    // (not `tight`): an answer if no formulation produces a tight one, but it
    // does not end the ladder -- a later stage may land on the same point
    // more accurately (CALIPSO ball-in-cup: the continuation's acceptable-
    // level point, residual 5e-5, against the penalty's 7e-8).
    let mut loose: Option<(MpccStage, Captured, Option<Certificate>, Number)> = None;
    // The first stage that converged with every product in tolerance but
    // did not certify as stationary: reported only if nothing certifies.
    let mut physical: Option<(MpccStage, Captured, Option<Certificate>)> = None;
    let solver_tol = app
        .options()
        .get_numeric_value("tol", "")
        .map(|(v, _)| v)
        .unwrap_or(1e-8);
    // `obj_scaling_factor < 0` is upstream's spelling of "maximize": the
    // stage comparison and the certificate work on the minimized `σ·f`.
    let sense = app
        .options()
        .get_numeric_value("obj_scaling_factor", "")
        .map_or(1.0, |(v, _)| if v < 0.0 { -1.0 } else { 1.0 });
    let pairs = wrapper.borrow().pairs().to_vec();
    let strategy = app
        .options()
        .get_string_value("complementarity_strategy", "")
        .ok()
        .and_then(|(v, _)| Strategy::parse(&v))
        .unwrap_or(Strategy::Both);

    // One lowered solve: options already set by the caller of `run`.
    let run = |app: &mut IpoptApplication,
               stage: MpccStage,
               lowering: Lowering,
               start: Option<StartOverride>,
               solves: &mut Vec<MpccSolveRecord>|
     -> Captured {
        {
            let mut w = wrapper.borrow_mut();
            // A solve refused before iteration 0 never reaches
            // `finalize_solution`, so without this it would report the
            // previous stage's point and products under its own status.
            w.clear_capture();
            w.set_lowering(lowering);
            w.set_start_override(start);
        }
        let status = app.optimize_tnlp(Rc::clone(&lowered));
        let statistics = app.statistics();
        let w = wrapper.borrow();
        // Complementarity violation: `|G·H|` for one-sided pairs, the positive
        // part for a mixed pair's rows; `+inf` when the solve returned no point.
        let max_abs_product = w.last_violation();
        let cap = Captured {
            status,
            solver_return: w.last_status().unwrap_or(SolverReturn::InternalError),
            statistics: statistics.clone(),
            x: w.last_x().to_vec(),
            z_l: w.last_z_l().to_vec(),
            z_u: w.last_z_u().to_vec(),
            lambda: w.last_lambda().to_vec(),
            max_abs_product,
        };
        solves.push(MpccSolveRecord {
            stage,
            tau: match lowering {
                Lowering::Relaxed(t) => Some(t),
                _ => None,
            },
            status,
            iterations: statistics.spent_iteration_count().max(0) as usize,
            max_abs_product: cap.max_abs_product,
        });
        if status == ApplicationReturnStatus::UserRequestedStop {
            stopped.borrow_mut().get_or_insert_with(|| cap.clone());
        }
        cap
    };
    let halted = || stopped.borrow().is_some();

    // The finish: one exact-product solve warm-started at a relaxed answer
    // (`G·H <= 0` where the equality would over-determine the model -- same
    // feasible set), on `budget` iterations. `None` only if a stop arrived
    // before it ran.
    let finish_from = |app: &mut IpoptApplication,
                       stage: MpccStage,
                       relaxed: &Captured,
                       budget: usize,
                       solves: &mut Vec<MpccSolveRecord>|
     -> Option<Captured> {
        let mut finish = None;
        for lowering in [Lowering::Eq, Lowering::Ineq] {
            baseline(app);
            for a in warm_start_assignments(relaxed.statistics.final_mu) {
                let _ = app.options_mut().read_from_str(&a, true);
            }
            set_budget(app, budget);
            let c = run(app, stage, lowering, Some(relaxed.warm()), solves);
            if halted() {
                return Some(c);
            }
            // Refused before iteration 0, whatever the status: the
            // inequality form.
            let over_determined =
                c.status == ApplicationReturnStatus::NotEnoughDegreesOfFreedom || c.x.is_empty();
            finish = Some(c);
            if !over_determined {
                break;
            }
        }
        finish
    };

    let mut stages_run = 0;
    'ladder: for &stage in STAGES.iter() {
        // An answer from a compared stage ends the ladder before the first
        // stage that is not compared (the penalty).
        if (accepted.is_some() && !compared(stage)) || halted() {
            break;
        }
        if !strategy.runs(stage) {
            continue;
        }
        if stages_run > 0 {
            app.set_in_retry_sequence(true);
        }
        stages_run += 1;
        baseline(app);
        let candidates: Vec<Captured> = match stage {
            MpccStage::Exact => {
                // Two second opinions, both from the model's own start, both
                // compared like any other candidate (lower objective wins):
                //
                // * A relaxed path: `G·H <= EXACT_STAGE_TAU`, one warm step to
                //   the continuation's last `τ`, then the exact finish. A
                //   relaxation keeps an interior; the exact product has none
                //   (every product row is parallel to its pair's active
                //   bound), so this path converges where the cold solve
                //   stalls, and it reaches branches the cold solve misses
                //   (MacMPEC `dempe`, `ex9.2.5`, both past their published
                //   values).
                // * The exact product solved cold, `G·H = 0`. The path least
                //   like the continuation's, and on some models the only one
                //   that reaches the better branch (MacMPEC `water-net`: 927,
                //   where both relaxed paths settle on 953 and 964).
                //
                // Neither dominates on MacMPEC or CALIPSO, so both run. With
                // an answer already in hand (the continuation's, or a converged
                // relaxed path's), the cold solve is a second opinion and runs
                // on a budget proportional to what the stages before it spent
                // ([`EXACT_BUDGET_FACTOR`]); the relaxed path is budgeted only
                // on the continuation's answer.
                let mut cands: Vec<Captured> = Vec::new();
                let budget = exact_stage_budget(
                    accepted.is_some(),
                    iterations_of(&solves, MpccStage::Continuation),
                );
                if let Some(b) = budget {
                    set_budget(app, b);
                }
                let relaxed = run(
                    app,
                    stage,
                    Lowering::Relaxed(EXACT_STAGE_TAU),
                    None,
                    &mut solves,
                );
                if halted() {
                    break;
                }
                // The relaxed path's verdict, if it has one: an exact-product
                // solve (its finish), or the relaxation's own infeasibility
                // (see [`exact_stage_verdict`] for how it ranks).
                let mut path_verdict: Option<Captured> = None;
                if converged(relaxed.status) && !relaxed.x.is_empty() {
                    // One warm step down to the continuation's last `τ`
                    // before the finish: an exact finish from `1e-6` is the
                    // degenerate solve again (`incid-set2-32`: 679 iterations
                    // to `max_iter`). Kept only if it converges.
                    baseline(app);
                    for a in warm_start_assignments(relaxed.statistics.final_mu) {
                        let _ = app.options_mut().read_from_str(&a, true);
                    }
                    if let Some(b) = budget {
                        set_budget(app, b);
                    }
                    let tight = run(
                        app,
                        stage,
                        Lowering::Relaxed(SCHOLTES_SCHEDULE[SCHOLTES_SCHEDULE.len() - 1]),
                        Some(relaxed.warm()),
                        &mut solves,
                    );
                    if halted() {
                        break;
                    }
                    let relaxed = if converged(tight.status) {
                        tight
                    } else {
                        relaxed
                    };
                    let fb = finish_budget(iterations_of(&solves, MpccStage::Exact));
                    let finish = finish_from(app, stage, &relaxed, fb, &mut solves);
                    if halted() {
                        break;
                    }
                    path_verdict = finish.clone();
                    cands.push(match finish {
                        Some(f) if accepts(f.status, f.max_abs_product, tol) => f,
                        _ => relaxed,
                    });
                } else if relaxed.status == ApplicationReturnStatus::InfeasibleProblemDetected
                    && !relaxed.x.is_empty()
                {
                    // The relaxation contains the model, so a relaxation with
                    // no feasible point means the model has none either; the
                    // solver's verdict is local, so the cold solve's own still
                    // ranks first ([`exact_stage_verdict`]).
                    path_verdict = Some(relaxed.clone());
                    cands.push(relaxed);
                }
                // The cold exact product. `G·H <= 0` where the equality would
                // over-determine the model (same feasible set), and whenever
                // the equality solve was refused before iteration 0 (no
                // point), whatever the status: the degrees-of-freedom gate
                // can refuse an over-determined model with a proven verdict
                // (`Infeasible_Problem_Detected`) rather than
                // `Not_Enough_Degrees_Of_Freedom`.
                let answered = accepted.is_some()
                    || cands
                        .first()
                        .is_some_and(|c| accepts(c.status, c.max_abs_product, tol));
                let cold_budget = exact_stage_budget(
                    answered,
                    iterations_of(&solves, MpccStage::Continuation)
                        + iterations_of(&solves, MpccStage::Exact),
                );
                let mut c = None;
                for lowering in [Lowering::Eq, Lowering::Ineq] {
                    baseline(app);
                    if let Some(b) = cold_budget {
                        set_budget(app, b);
                    }
                    let r = run(app, stage, lowering, None, &mut solves);
                    if halted() {
                        break;
                    }
                    let refused = r.status == ApplicationReturnStatus::NotEnoughDegreesOfFreedom
                        || r.x.is_empty();
                    c = Some(r);
                    if !refused {
                        break;
                    }
                }
                if halted() {
                    break;
                }
                exact = exact_stage_verdict(path_verdict, c.clone());
                cands.extend(c);
                cands
            }
            MpccStage::Continuation => {
                let mut last: Option<Captured> = None;
                for &tau in &SCHOLTES_SCHEDULE {
                    baseline(app);
                    let warm = last.as_ref().map(|c| {
                        for a in warm_start_assignments(c.statistics.final_mu) {
                            let _ = app.options_mut().read_from_str(&a, true);
                        }
                        c.warm()
                    });
                    let mut c = run(app, stage, Lowering::Relaxed(tau), warm, &mut solves);
                    if halted() {
                        break;
                    }
                    if !converged(c.status) {
                        // One weaker rung before giving up on the schedule:
                        // the previous stage's x only, no dual warm start.
                        let Some(prev) = &last else { break };
                        baseline(app);
                        // The override carries no duals; a caller's
                        // `warm_start_init_point yes` would ask for them and
                        // refuse the start.
                        let _ = app
                            .options_mut()
                            .read_from_str("warm_start_init_point no\n", true);
                        let primal = StartOverride {
                            x: prev.x.clone(),
                            ..Default::default()
                        };
                        c = run(
                            app,
                            stage,
                            Lowering::Relaxed(tau),
                            Some(primal),
                            &mut solves,
                        );
                        if halted() || !converged(c.status) {
                            break;
                        }
                    }
                    last = Some(c);
                }
                if halted() {
                    break;
                }
                match last {
                    None => {
                        report(&format!(
                            "pounce: complementarity: {} did not solve its first relaxation \
                             (τ = {:.0e}, {:?}); trying the next formulation.",
                            stage.label(),
                            SCHOLTES_SCHEDULE[0],
                            solves.last().map(|s| s.status),
                        ));
                        Vec::new()
                    }
                    Some(relaxed) => {
                        // The finish: exact product from the end of the
                        // schedule; `G·H <= 0` where the equality would
                        // over-determine the model (same feasible set).
                        //
                        // On a budget ([`FINISH_BUDGET_FLOOR`], or what the
                        // schedule spent if more): a finish that has not
                        // converged by then leaves the schedule's answer.
                        let budget = finish_budget(iterations_of(&solves, MpccStage::Continuation));
                        let finish = finish_from(app, stage, &relaxed, budget, &mut solves);
                        if halted() {
                            break;
                        }
                        // With the exact stage not run, the finish -- an
                        // exact-product solve -- is the verdict when nothing
                        // answers, never the relaxed solve below.
                        if !strategy.runs(MpccStage::Exact) {
                            exact = finish.clone();
                        }
                        // A finish that does not converge leaves the
                        // continuation's own answer, which satisfies every
                        // product to the last τ; accepted on the same test.
                        vec![match finish {
                            Some(f) if accepts(f.status, f.max_abs_product, tol) => f,
                            _ => relaxed,
                        }]
                    }
                }
            }
            // Not in STAGES: run only from a reported point, below.
            MpccStage::Branch => Vec::new(),
            MpccStage::Penalty => {
                let _ = app
                    .options_mut()
                    .read_from_str("l1_exact_penalty_barrier yes\n", true);
                let c = run(app, stage, Lowering::Eq, None, &mut solves);
                if halted() {
                    break;
                }
                vec![c]
            }
        };
        for c in &candidates {
            if first
                .as_ref()
                .is_none_or(|f| f.x.is_empty() && !c.x.is_empty())
            {
                first = Some(c.clone());
            }
        }
        for c in candidates {
            if accepts(c.status, c.max_abs_product, tol) {
                // Converged and physical; the certificate decides whether it
                // is an answer. A lowered solve can stall short of a biactive
                // pair and still exit "acceptable" with every product at 0.
                let cert = certify_at(&inner, &pairs, &c.x, solver_tol, tol, sense);
                if cert.as_ref().is_none_or(|ct| answers(ct.class)) {
                    let f = match sense * objective_at(&inner, &c.x) {
                        f if f.is_finite() => f,
                        _ => Number::INFINITY,
                    };
                    if cert.as_ref().is_some_and(|ct| !ct.tight) {
                        report(&format!(
                            "pounce: complementarity: {} converged ({:?}) to a strongly \
                             stationary point within the sqrt(tol) band only (residual {:.1e}); \
                             kept, trying the next formulation for a tighter one.",
                            stage.label(),
                            c.status,
                            cert.as_ref().map_or(Number::NAN, |ct| ct.residual),
                        ));
                        let better = loose.as_ref().is_none_or(|(_, _, _, g)| {
                            f.is_finite() && f < g - 1e-9 * g.abs().max(1.0)
                        });
                        if better {
                            loose = Some((stage, c, cert, f));
                        }
                        continue;
                    }
                    report(&format!(
                        "pounce: complementarity: {} solved the model ({:?}, violation = {:.1e}, \
                         minimized objective {:.6e}).",
                        stage.label(),
                        c.status,
                        c.max_abs_product,
                        f
                    ));
                    // Lower objective wins; a tie (relative 1e-9) keeps the
                    // earlier stage's answer.
                    let better = accepted.as_ref().is_none_or(|(_, _, _, g)| {
                        f.is_finite() && f < g - 1e-9 * g.abs().max(1.0)
                    });
                    if better {
                        accepted = Some((stage, c, cert, f));
                    }
                    if !compared(stage) {
                        break 'ladder;
                    }
                    continue;
                }
                if cert
                    .as_ref()
                    .is_some_and(|ct| kept_if_nothing_better(ct.class))
                {
                    report(&format!(
                        "pounce: complementarity: {} converged ({:?}) to a weakly stationary \
                         point only; kept, trying the next formulation for a strongly \
                         stationary one.",
                        stage.label(),
                        c.status,
                    ));
                    if weak.is_none() {
                        weak = Some((stage, c, cert));
                    }
                    continue;
                }
                report(&format!(
                    "pounce: complementarity: {} converged ({:?}) to a point that is not \
                     stationary (residual {:.1e}); trying the next formulation.",
                    stage.label(),
                    c.status,
                    cert.as_ref().map_or(Number::NAN, |ct| ct.residual),
                ));
                if physical.is_none() {
                    physical = Some((stage, c, cert));
                }
                continue;
            }
            report(&format!(
                "pounce: complementarity: {} did not ({:?}, violation = {:.1e}); trying the next \
                 formulation.",
                stage.label(),
                c.status,
                c.max_abs_product
            ));
        }
    }
    app.set_in_retry_sequence(false);
    restore(app);

    // With no S answer, the first W one; with neither, the first physical
    // one (converged, products in tolerance, not stationary) with its
    // certificate; failing that, the exact-product stage's verdict (or, if
    // it was refused with no point, the first stage that has one).
    let verdict = verdict_of(exact, first);
    // With no tight answer, the best S point inside the band is the answer.
    let accepted = accepted.or(loose);
    let mut selection = match (accepted, weak, physical, verdict) {
        (Some((s, c, cert, _)), _, _, _) => Some((Reported::Answer(s), c, Some(cert))),
        (None, Some((s, c, cert)), _, _) => Some((Reported::WeakFallback(s), c, Some(cert))),
        // Feasible, but the certificate says not stationary: the lowered
        // solve's success is not the declared model's, so it is reported
        // at the acceptable level rather than as `Solve_Succeeded`.
        (None, None, Some((s, c, cert)), _) => {
            Some((Reported::FeasibleUncertified(s), acceptable(c), Some(cert)))
        }
        (None, None, None, Some(c)) => Some((Reported::Verdict, c, None)),
        (None, None, None, None) => None,
    };

    // A feasible reported point the certificate does not call S may still be
    // B-stationary (ralph1's origin is), or may sit where a branch descends.
    // Check, and while a branch descends, re-solve it from the point: an
    // ordinary NLP with each pair's held side pinned. Bounded by
    // MAX_BRANCH_ESCAPES; each accepted escape is feasible and strictly lower.
    let mut b_check: Option<BCheck> = None;
    let mut escapes = 0;
    if let Some((reported, chosen, known_cert)) = selection.as_mut() {
        let eligible = |cert: &Option<Option<Certificate>>, c: &Captured| {
            matches!(cert, Some(Some(ct)) if ct.feasible && ct.class != StationarityClass::S)
                && accepts(c.status, c.max_abs_product, tol)
        };
        while !halted() && escapes < MAX_BRANCH_ESCAPES && eligible(known_cert, chosen) {
            let Some(mp) = model_at(&inner, &chosen.x, sense) else {
                break;
            };
            let inp = mp.input(&pairs, solver_tol, tol);
            let bc = check_b_stationarity(&inp);
            report(&describe_b_check(&bc));
            let BVerdict::NotB { held, .. } = &bc.verdict else {
                b_check = Some(bc);
                break;
            };
            let bounds = branch_bounds(&inp, held);
            wrapper.borrow_mut().set_branch(Some(bounds));
            app.set_in_retry_sequence(true);
            baseline(app);
            let start = StartOverride {
                x: chosen.x.clone(),
                ..Default::default()
            };
            let c = run(
                app,
                MpccStage::Branch,
                Lowering::Ineq,
                Some(start),
                &mut solves,
            );
            wrapper.borrow_mut().set_branch(None);
            app.set_in_retry_sequence(false);
            restore(app);
            b_check = Some(bc);
            if halted() {
                break;
            }
            let f_old = sense * objective_at(&inner, &chosen.x);
            let f_new = sense * objective_at(&inner, &c.x);
            let cert = certify_at(&inner, &pairs, &c.x, solver_tol, tol, sense);
            let lower = f_new < f_old - 1e-9 * f_old.abs().max(1.0);
            if !(accepts(c.status, c.max_abs_product, tol)
                && lower
                && cert.as_ref().is_some_and(|ct| ct.feasible))
            {
                report(&format!(
                    "pounce: complementarity: the branch NLP did not improve the point ({:?}, \
                     minimized objective {f_new:.6e} against {f_old:.6e}); keeping it.",
                    c.status
                ));
                break;
            }
            report(&format!(
                "pounce: complementarity: the branch NLP moved the point off the non-B-stationary \
                 one: minimized objective {f_old:.6e} -> {f_new:.6e}."
            ));
            escapes += 1;
            b_check = None; // a check describes the point it ran at
            let class = cert.as_ref().map(|ct| ct.class);
            (*reported, *chosen) = match class {
                Some(StationarityClass::S) => (Reported::Answer(MpccStage::Branch), c),
                Some(StationarityClass::W) => (Reported::WeakFallback(MpccStage::Branch), c),
                _ => (
                    Reported::FeasibleUncertified(MpccStage::Branch),
                    acceptable(c),
                ),
            };
            *known_cert = Some(cert);
        }
    }

    if let Some(c) = stopped.into_inner() {
        // The caller asked to stop: report the stopped solve, nothing else.
        let total_iterations = solves.iter().map(|s| s.iterations).sum();
        {
            let mut w = wrapper.borrow_mut();
            w.set_start_override(None);
            w.forward_finalize(
                c.solver_return,
                &c.x,
                &c.z_l,
                &c.z_u,
                &c.lambda,
                &IpoptData::default(),
                &IpoptCq::default(),
            );
        }
        return Ok(MpccOutcome {
            status: c.status,
            statistics: c.statistics,
            accepted_by: None,
            fallback: None,
            reported: Reported::Verdict,
            strategy,
            solves,
            total_iterations,
            x: c.x,
            max_abs_product: c.max_abs_product,
            certificate: None,
            b_check: None,
            escapes,
        });
    }
    let (reported, chosen, known_cert) =
        selection.ok_or_else(|| "no exact-product stage ran".to_string())?;
    let accepted_by = match reported {
        Reported::Answer(s) => Some(s),
        _ => None,
    };
    let fallback = match reported {
        Reported::WeakFallback(s) => Some(s),
        _ => None,
    };
    {
        let mut w = wrapper.borrow_mut();
        w.set_start_override(None);
        w.forward_finalize(
            chosen.solver_return,
            &chosen.x,
            &chosen.z_l,
            &chosen.z_u,
            &chosen.lambda,
            &IpoptData::default(),
            &IpoptCq::default(),
        );
    }
    let total_iterations = solves.iter().map(|s| s.iterations).sum();
    let certificate = match known_cert {
        Some(c) => c,
        None => certify_at(&inner, &pairs, &chosen.x, solver_tol, tol, sense),
    };
    Ok(MpccOutcome {
        status: chosen.status,
        statistics: chosen.statistics,
        accepted_by,
        fallback,
        reported,
        strategy,
        solves,
        total_iterations,
        x: chosen.x,
        max_abs_product: chosen.max_abs_product,
        certificate,
        b_check,
        escapes,
    })
}

/// Iterations the solves of `stage` have spent so far.
fn iterations_of(solves: &[MpccSolveRecord], stage: MpccStage) -> usize {
    solves
        .iter()
        .filter(|s| s.stage == stage)
        .map(|s| s.iterations)
        .sum()
}

/// A converged point whose certificate does not say stationary, reported at
/// the acceptable level: the lowered solve's success is not the model's.
fn acceptable(c: Captured) -> Captured {
    Captured {
        status: ApplicationReturnStatus::SolvedToAcceptableLevel,
        solver_return: SolverReturn::StopAtAcceptablePoint,
        ..c
    }
}

/// At most this many branch NLPs from one reported point.
const MAX_BRANCH_ESCAPES: usize = 3;

/// The console line for a B-stationarity check.
fn describe_b_check(bc: &BCheck) -> String {
    let how = if bc.enumerated {
        format!(
            "{} biactive pair{}, {} branch LP{}",
            bc.n_biactive,
            if bc.n_biactive == 1 { "" } else { "s" },
            bc.branches_solved,
            if bc.branches_solved == 1 { "" } else { "s" }
        )
    } else {
        format!(
            "{} biactive pairs, too many to enumerate; heuristic branch",
            bc.n_biactive
        )
    };
    match &bc.verdict {
        BVerdict::B => format!(
            "pounce: complementarity: B-stationarity ({how}): B-stationary -- no branch descends."
        ),
        BVerdict::NotB { decrease, .. } => format!(
            "pounce: complementarity: B-stationarity ({how}): not B-stationary -- a branch \
             descends (∇f·d = {decrease:.2e}); re-solving that branch."
        ),
        BVerdict::NotChecked(why) => {
            format!("pounce: complementarity: B-stationarity ({how}): not decided ({why}).")
        }
    }
}

/// The *inner* model's objective at `x`; `+inf` if it cannot be evaluated,
/// so an unevaluable answer never wins the comparison.
fn objective_at(inner: &Rc<RefCell<dyn TNLP>>, x: &[Number]) -> Number {
    inner
        .borrow_mut()
        .eval_f(x, true)
        .filter(|f| f.is_finite())
        .unwrap_or(Number::INFINITY)
}

/// The exact stage's verdict -- what is reported when nothing answers -- from
/// its two solves: the cold exact product's own whenever it returned a point
/// (it is the plain lowered model), else the relaxed path's (its finish, or
/// its relaxation's infeasibility), else the cold solve's refusal.
///
/// The relaxed path does not override a cold solve that has a point, even
/// when the relaxation reports infeasible. The relaxation contains the
/// model, so a relaxation that truly has no feasible point means the model
/// has none either; but an interior-point `Infeasible_Problem_Detected` is
/// local (a stationary point of the infeasibility measure from one start),
/// not a proof, and preferring it would turn a cold solve's "out of
/// iterations" into a stronger claim than either solve established (review of
/// #1121).
fn exact_stage_verdict(path: Option<Captured>, cold: Option<Captured>) -> Option<Captured> {
    match (path, cold) {
        (_, Some(c)) if !c.x.is_empty() => Some(c),
        (Some(p), c) => Some(p).or(c),
        (None, c) => c,
    }
}

/// The point and status reported when no stage answered: the exact-product
/// stage's. If that stage has no point (refused before iteration 0), the
/// point is borrowed from the first stage that has one — but the status stays
/// the exact stage's. A relaxed solve's `Solve_Succeeded` is a success on the
/// relaxation; carried over whole, it reported success for a model the exact
/// stage had proved infeasible (PR review of #1103).
fn verdict_of(exact: Option<Captured>, first: Option<Captured>) -> Option<Captured> {
    match (exact, first) {
        (Some(e), _) if !e.x.is_empty() => Some(e),
        (Some(e), Some(f)) => Some(Captured {
            status: e.status,
            solver_return: e.solver_return,
            statistics: e.statistics,
            ..f
        }),
        (e, f) => e.or(f),
    }
}

/// The *inner* model's data at a point, as the certificate and the
/// B-stationarity check read it.
struct ModelPoint {
    x: Vec<Number>,
    grad_f: Vec<Number>,
    g: Vec<Number>,
    jac: Vec<(usize, usize, Number)>,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    g_l: Vec<Number>,
    g_u: Vec<Number>,
}

impl ModelPoint {
    fn input<'a>(
        &'a self,
        pairs: &'a [ComplementarityPair],
        solver_tol: Number,
        feas_tol: Number,
    ) -> CertificateInput<'a> {
        CertificateInput {
            x: &self.x,
            grad_f: &self.grad_f,
            g: &self.g,
            jac: &self.jac,
            x_l: &self.x_l,
            x_u: &self.x_u,
            g_l: &self.g_l,
            g_u: &self.g_u,
            pairs,
            solver_tol,
            feas_tol,
        }
    }
}

/// Evaluate the *inner* model at `x`, the objective gradient in the
/// minimized sense. `None` if any callback declines.
fn model_at(inner: &Rc<RefCell<dyn TNLP>>, x: &[Number], sense: Number) -> Option<ModelPoint> {
    let mut m = inner.borrow_mut();
    let info = m.get_nlp_info()?;
    let (n, mm, nnz) = (info.n as usize, info.m as usize, info.nnz_jac_g as usize);
    if x.len() != n {
        return None;
    }
    let off = match info.index_style {
        IndexStyle::C => 0,
        IndexStyle::Fortran => 1,
    };
    let (mut x_l, mut x_u, mut g_l, mut g_u) =
        (vec![0.0; n], vec![0.0; n], vec![0.0; mm], vec![0.0; mm]);
    if !m.get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) {
        return None;
    }
    let mut grad_f = vec![0.0; n];
    let mut g = vec![0.0; mm];
    if !m.eval_grad_f(x, true, &mut grad_f) || !m.eval_g(x, false, &mut g) {
        return None;
    }
    grad_f.iter_mut().for_each(|d| *d *= sense);
    let (mut ir, mut jc, mut v) = (vec![0 as Index; nnz], vec![0 as Index; nnz], vec![0.0; nnz]);
    if !m.eval_jac_g(
        None,
        false,
        SparsityRequest::Structure {
            irow: &mut ir,
            jcol: &mut jc,
        },
    ) || !m.eval_jac_g(Some(x), false, SparsityRequest::Values { values: &mut v })
    {
        return None;
    }
    let jac = (0..nnz)
        .map(|k| ((ir[k] - off) as usize, (jc[k] - off) as usize, v[k]))
        .collect();
    Some(ModelPoint {
        x: x.to_vec(),
        grad_f,
        g,
        jac,
        x_l,
        x_u,
        g_l,
        g_u,
    })
}

/// Evaluate the *inner* model at `x` and certify it. `None` if any callback
/// declines.
fn certify_at(
    inner: &Rc<RefCell<dyn TNLP>>,
    pairs: &[ComplementarityPair],
    x: &[Number],
    solver_tol: Number,
    feas_tol: Number,
    sense: Number,
) -> Option<Certificate> {
    let mp = model_at(inner, x, sense)?;
    Some(certify(&mp.input(pairs, solver_tol, feas_tol)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(status: ApplicationReturnStatus, x: Vec<Number>) -> Captured {
        Captured {
            status,
            solver_return: SolverReturn::InternalError,
            statistics: SolveStatistics::default(),
            max_abs_product: if x.is_empty() { Number::INFINITY } else { 0.25 },
            x,
            z_l: Vec::new(),
            z_u: Vec::new(),
            lambda: Vec::new(),
        }
    }

    /// The exact stage refused before iteration 0 with a proven verdict and
    /// no point, after a relaxed solve that "succeeded": the reported status
    /// is the exact stage's, never the relaxed success; only the point is
    /// borrowed.
    #[test]
    fn a_refused_exact_stage_keeps_its_own_verdict() {
        let exact = cap(
            ApplicationReturnStatus::InfeasibleProblemDetected,
            Vec::new(),
        );
        let first = cap(ApplicationReturnStatus::SolveSucceeded, vec![0.5, 0.5]);
        let v = verdict_of(Some(exact), Some(first)).unwrap();
        assert_eq!(v.status, ApplicationReturnStatus::InfeasibleProblemDetected);
        assert_eq!(v.x, vec![0.5, 0.5]);
    }

    /// Every branch of [`exact_stage_verdict`]: a cold solve with a point is
    /// the verdict even over a relaxed infeasibility (it has a point but did
    /// not converge here); a refused cold solve defers to the relaxed path;
    /// with neither, the refusal stands.
    #[test]
    fn the_cold_solve_with_a_point_is_the_exact_stages_verdict() {
        use ApplicationReturnStatus as S;
        let relaxed_infeasible = || cap(S::InfeasibleProblemDetected, vec![0.5, 0.5]);
        let cold_out_of_iterations = cap(S::MaximumIterationsExceeded, vec![1.0, 0.0]);
        let refused = || cap(S::NotEnoughDegreesOfFreedom, vec![]);

        let v = exact_stage_verdict(Some(relaxed_infeasible()), Some(cold_out_of_iterations));
        assert_eq!(v.map(|v| v.status), Some(S::MaximumIterationsExceeded));

        let v = exact_stage_verdict(Some(relaxed_infeasible()), Some(refused())).unwrap();
        assert_eq!(v.status, S::InfeasibleProblemDetected);
        assert_eq!(v.x, vec![0.5, 0.5]);

        let v = exact_stage_verdict(None, Some(refused()));
        assert_eq!(v.map(|v| v.status), Some(S::NotEnoughDegreesOfFreedom));

        assert!(exact_stage_verdict(None, None).is_none());
    }

    #[test]
    fn an_exact_stage_with_a_point_is_the_verdict() {
        let exact = cap(
            ApplicationReturnStatus::InfeasibleProblemDetected,
            vec![1.0, 0.0],
        );
        let first = cap(ApplicationReturnStatus::SolveSucceeded, vec![0.5, 0.5]);
        let v = verdict_of(Some(exact), Some(first)).unwrap();
        assert_eq!(v.status, ApplicationReturnStatus::InfeasibleProblemDetected);
        assert_eq!(v.x, vec![1.0, 0.0]);
    }
}
