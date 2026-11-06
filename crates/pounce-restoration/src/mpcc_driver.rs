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
    BCheck, BVerdict, Certificate, CertificateInput, EXACT_STAGE_TAU, FINISH_BUDGET_FLOOR,
    MpccStage, POLISH_TRIGGER, RUNG_RETRY_FLOOR, Reported, SAFE_DEFAULTS, SCHOLTES_SCHEDULE,
    STAGES, StationarityClass, Strategy, TOUCHED_OPTIONS, accepts, answers, answers_agree,
    branch_bounds, budgeted_max_iter, certify, check_b_stationarity, compared, complementarity_gap,
    converged, exact_stage_budget, finish_budget, finish_drifted, kept_if_nothing_better,
    natural_residual, polish_class, polish_objective_ok, polish_tightened, relaxation_slack,
    skips_cold_solve, warm_rung_max_iter, warm_start_assignments,
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
    /// Wall-clock seconds of the whole `optimize_tnlp` call, retries and
    /// second opinions inside it included.
    pub seconds: Number,
}

/// What the branch re-solve of a nearly biactive answer did
/// ([`natural_residual`]). Recorded whenever it ran.
#[derive(Debug, Clone)]
pub struct PolishRecord {
    /// The re-solved point replaced the answer.
    pub taken: bool,
    /// The answer's largest `min(|G|, |H|)` before the re-solve.
    pub residual_before: Number,
    /// The same at the re-solved point; `+inf` if it was not measured.
    pub residual_after: Number,
    /// Why the re-solved point was not taken; `None` when it was.
    pub reason: Option<String>,
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
    /// The forwarded point's largest `min(|G|, |H|)` over its pairs
    /// ([`natural_residual`]): how complementary it is, in the units of the
    /// pair sides. `None` when the model could not be evaluated there.
    pub natural_residual: Option<Number>,
    /// The branch re-solve, when it ran.
    pub polish: Option<PolishRecord>,
    /// The `natural_residual` above which the branch re-solve runs
    /// (`POLISH_TRIGGER · tol`), so a reader can tell a residual the driver
    /// left alone on purpose from one it failed to tighten.
    pub polish_threshold: Number,
}

impl MpccOutcome {
    /// The solve report's `complementarity` block. One conversion, shared by
    /// the CLI's `--json-output` and Python's `report_path`, so the two
    /// frontends cannot describe the same run differently.
    pub fn report_info(&self) -> pounce_solve_report::ComplementarityInfo {
        use pounce_algorithm::mpcc::StationarityClass;
        use pounce_solve_report::{
            ComplementarityBStationarityInfo, ComplementarityCertificateInfo, ComplementarityInfo,
            ComplementarityPolishInfo, ComplementaritySolveInfo,
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
                    seconds: finite(r.seconds),
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
            b_stationarity: self.b_check.as_ref().map(|bc| {
                use pounce_algorithm::mpcc::BVerdict;
                let (verdict, reason) = match &bc.verdict {
                    BVerdict::B => ("B", None),
                    BVerdict::NotB { .. } => ("not_B", None),
                    BVerdict::NotChecked(why) => ("not_checked", Some(why.to_string())),
                };
                ComplementarityBStationarityInfo {
                    verdict: verdict.to_string(),
                    reason,
                    n_biactive: bc.n_biactive,
                    branches_solved: bc.branches_solved,
                    enumerated: bc.enumerated,
                }
            }),
            escapes: self.escapes,
            natural_residual: self.natural_residual.and_then(finite),
            polish: self.polish.as_ref().map(|p| ComplementarityPolishInfo {
                taken: p.taken,
                residual_before: finite(p.residual_before),
                residual_after: finite(p.residual_after),
                reason: p.reason.clone(),
            }),
            polish_threshold: finite(self.polish_threshold),
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
    /// What a relaxed point's product slack buys, to first order
    /// ([`relaxation_slack`]): the margin a relaxed point is allowed over a
    /// finish. `0` for an exact-product point (`Eq`/`Ineq`): its products
    /// are residuals within `constr_viol_tol`, not slack it was granted, and
    /// at a point where MFCQ fails their multipliers can be large enough
    /// that `|λ|·p` would let two different branches agree. `+inf` with no
    /// point.
    slack: Number,
    /// The point's [`complementarity_gap`] in the inner model: how far above
    /// its branch's optimum stopping at tolerance can leave it. `+inf` with
    /// no point, or when the model declines an evaluation.
    gap: Number,
    /// Whether the point came from an exact-product solve (`Eq`/`Ineq`),
    /// not a relaxation: a continuation whose finish did not converge keeps
    /// its last relaxed point, which is an answer but not a solution of the
    /// exact product.
    exact_lowering: bool,
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
/// The coupled continuation's law, `τ = max(floor, COUPLED_TAU_PER_MU · μ)`,
/// never cut by more than a factor [`COUPLED_MAX_CUT`] per `μ` update.
/// CCOpt's proportional relaxation update uses the same ratio; the cap keeps a
/// sharp `μ` drop from collapsing the relaxation in one step.
const COUPLED_TAU_PER_MU: Number = 1.0;
/// See [`COUPLED_TAU_PER_MU`].
const COUPLED_MAX_CUT: Number = 0.1;
/// The coupled solve's budget: this many times the first rung's iterations,
/// at least [`COUPLED_BUDGET_FLOOR`]. Where it converges it takes a few dozen
/// iterations (`relu_nn`: 33 to 114); one that reaches the budget is crawling,
/// and the schedule takes over from the first rung.
const COUPLED_BUDGET_FACTOR: usize = 5;
/// See [`COUPLED_BUDGET_FACTOR`].
const COUPLED_BUDGET_FLOOR: usize = 200;

/// Whether finish attempt `a` came closer than `b`: one that returned a
/// point beats one that did not, a converged one beats one that did not, and
/// otherwise the smaller product violation wins.
fn closer_finish(a: Option<&Captured>, b: Option<&Captured>) -> bool {
    let rank = |c: &Captured| (!c.x.is_empty(), converged(c.status));
    match (a, b) {
        (None, _) => false,
        (Some(_), None) => true,
        (Some(a), Some(b)) => {
            let (ra, rb) = (rank(a), rank(b));
            ra > rb || (ra == rb && a.max_abs_product < b.max_abs_product)
        }
    }
}

/// How an exact finish is started from a relaxed point.
#[derive(Debug, Clone, Copy, PartialEq)]
enum FinishStart {
    /// The relaxed point and its duals, the barrier where it ended.
    Warm,
    /// The same, with the barrier restarted at this `μ`.
    WarmAtMu(Number),
    /// The relaxed point alone: fresh duals, the default barrier start.
    Primal,
}

/// The coupled continuation's finish attempts, in order (measured on
/// `relu_nn`: no one of them converges on every model, and each converges
/// where some other fails).
const COUPLED_FINISH_STARTS: [FinishStart; 3] = [
    FinishStart::WarmAtMu(1e-6),
    FinishStart::Warm,
    FinishStart::Primal,
];

/// Each coupled finish attempt's iteration budget. A finish that converges
/// from the coupled endpoint takes a few dozen iterations; one that does not
/// runs to its limit.
const COUPLED_FINISH_BUDGET: usize = 100;

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
    // A budget is the driver's own decision to stop, so a budgeted solve
    // that reaches it runs without `mu_strategy_fallback`, which would
    // otherwise treat the cap as a stall and spend a second budget on a
    // flipped-schedule re-solve. Unless the caller set the option: then it
    // is theirs. Unbudgeted solves (the continuation's rungs) keep it.
    let caller_set_fallback = matches!(
        app.options().get_bool_value("mu_strategy_fallback", ""),
        Ok((_, true))
    );
    let set_budget = |app: &mut IpoptApplication, budget: usize| {
        let it = budgeted_max_iter(budget, caller_max_iter);
        let _ = app
            .options_mut()
            .read_from_str(&format!("max_iter {it}\n"), true);
        if !caller_set_fallback {
            let _ = app
                .options_mut()
                .read_from_str("mu_strategy_fallback no\n", true);
        }
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
    // The coupling rides the interior-point loop's μ updates
    // (`TNLP::update_with_barrier`); under another engine (`algorithm =
    // active-set-sqp`) τ would never move, and the coupled solve would only
    // spend its budget before the schedule took over.
    let interior_point = app
        .options()
        .get_string_value("algorithm", "")
        .map_or(true, |(v, _)| v == "interior-point");
    let coupled = interior_point
        && app
            .options()
            .get_string_value("complementarity_continuation", "")
            .map(|(v, _)| v != "schedule")
            .unwrap_or(true);

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
        // Timed here, around the whole call: the statistics' wall clock
        // describes the last attempt inside it, and a retry or second
        // opinion within the call resets it (ieee300_reserve_q98: 2.9 s of
        // 20 s by the statistics).
        let t0 = std::time::Instant::now();
        let status = app.optimize_tnlp(Rc::clone(&lowered));
        let seconds = t0.elapsed().as_secs_f64();
        let statistics = app.statistics();
        let w = wrapper.borrow();
        // Complementarity violation: `|G·H|` for one-sided pairs, the positive
        // part for a mixed pair's rows; `+inf` when the solve returned no point.
        let max_abs_product = w.last_violation();
        let lam = w.last_lambda();
        let m_inner = w.m_inner();
        let slack = match lowering {
            _ if w.last_x().is_empty() => Number::INFINITY,
            Lowering::Relaxed(_) | Lowering::Coupled { .. } => {
                relaxation_slack(w.last_products(), lam.get(m_inner..).unwrap_or(&[]))
            }
            _ => 0.0,
        };
        let gap = if w.last_x().is_empty() {
            Number::INFINITY
        } else {
            gap_at(
                &inner,
                w.last_x(),
                w.last_z_l(),
                w.last_z_u(),
                lam.get(..m_inner).unwrap_or(&[]),
            )
        };
        let cap = Captured {
            status,
            solver_return: w.last_status().unwrap_or(SolverReturn::InternalError),
            statistics: statistics.clone(),
            x: w.last_x().to_vec(),
            z_l: w.last_z_l().to_vec(),
            z_u: w.last_z_u().to_vec(),
            lambda: w.last_lambda().to_vec(),
            max_abs_product,
            slack,
            gap,
            exact_lowering: matches!(lowering, Lowering::Eq | Lowering::Ineq),
        };
        solves.push(MpccSolveRecord {
            stage,
            // The `τ` the solve ended at: a coupled solve's last.
            tau: w.current_tau(),
            status,
            iterations: statistics.spent_iteration_count().max(0) as usize,
            max_abs_product: cap.max_abs_product,
            seconds,
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
                       start: FinishStart,
                       solves: &mut Vec<MpccSolveRecord>|
     -> Option<Captured> {
        let mut finish = None;
        for lowering in [Lowering::Eq, Lowering::Ineq] {
            baseline(app);
            for a in warm_start_assignments(relaxed.statistics.final_mu) {
                let _ = app.options_mut().read_from_str(&a, true);
            }
            set_budget(app, budget);
            if let FinishStart::WarmAtMu(mu) = start {
                let _ = app
                    .options_mut()
                    .read_from_str(&format!("mu_init {mu:e}\n"), true);
            }
            let point = if start == FinishStart::Primal {
                // The override carries no duals; asking for them would refuse it.
                let _ = app
                    .options_mut()
                    .read_from_str("warm_start_init_point no\n", true);
                StartOverride {
                    x: relaxed.x.clone(),
                    ..Default::default()
                }
            } else {
                relaxed.warm()
            };
            let c = run(app, stage, lowering, Some(point), solves);
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

    // The minimized objective at a captured point (`+inf` if it does not
    // evaluate), as the stage comparison reads it.
    let min_obj = |c: &Captured| -> Number {
        match sense * objective_at(&inner, &c.x) {
            f if f.is_finite() && !c.x.is_empty() => f,
            _ => Number::INFINITY,
        }
    };
    // Item: a finish replaces the relaxed point it started from only if it
    // converged with every product in tolerance *and* did not drift to a
    // worse objective ([`finish_drifted`]); otherwise the relaxed point,
    // which satisfies every product to its `τ`, is the candidate.
    let keep_finish = |finish: Option<Captured>,
                       relaxed: Captured,
                       label: &str,
                       report: &mut dyn FnMut(&str)|
     -> Captured {
        match finish {
            Some(f) if accepts(f.status, f.max_abs_product, tol) => {
                let (ff, fr) = (min_obj(&f), min_obj(&relaxed));
                if finish_drifted(ff, fr, relaxed.slack) {
                    report(&format!(
                        "pounce: complementarity: {label}'s exact finish drifted to a worse \
                         point (minimized objective {ff:.6e} against {fr:.6e} at the last \
                         relaxation, whose products are worth {:.1e}); keeping the relaxed \
                         point.",
                        relaxed.slack
                    ));
                    relaxed
                } else {
                    f
                }
            }
            _ => relaxed,
        }
    };

    // The certificate of a point, computed once per point: the coupled
    // continuation certifies its candidate to decide whether to keep it, and
    // the acceptance loop below certifies the same candidate again (review of
    // #1135, item 8: on MacMPEC's 961-pair models that is not free).
    let cert_cache: RefCell<Vec<(Vec<Number>, Option<Certificate>)>> = RefCell::new(Vec::new());
    let certify_cached = |x: &[Number]| -> Option<Certificate> {
        if let Some((_, c)) = cert_cache.borrow().iter().find(|(p, _)| p.as_slice() == x) {
            return c.clone();
        }
        let c = certify_at(&inner, &pairs, x, solver_tol, tol, sense);
        cert_cache.borrow_mut().push((x.to_vec(), c.clone()));
        c
    };
    let mut stages_run = 0;
    // Whether the coupled continuation supplied the continuation's answer.
    let mut coupled_kept = false;
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
            MpccStage::Exact => 'exact_stage: {
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
                // Neither dominates on MacMPEC or CALIPSO, so both run -- the
                // cold solve only until the continuation and the relaxed path
                // both hold an exact answer (below). With an answer already in hand
                // (the continuation's, or a converged relaxed path's), the cold
                // solve is a second opinion and runs on a budget proportional
                // to what the stages before it spent ([`EXACT_BUDGET_FACTOR`]);
                // the relaxed path is budgeted only on the continuation's
                // answer.
                let mut cands: Vec<Captured> = Vec::new();
                // The budgets are proportional to what the continuation
                // spent, and the coupled continuation spends a fraction of
                // what the schedule does: floored at the finish's own floor
                // then, or the relaxed path is cut short of answers the
                // schedule's budget allowed (`relu_nn` digits_128x3: 285
                // iterations needed, 200 given).
                let exp_floor: usize = if coupled_kept { FINISH_BUDGET_FLOOR } else { 0 };
                let budget = exact_stage_budget(
                    accepted.is_some(),
                    iterations_of(&solves, MpccStage::Continuation),
                )
                .map(|b| b.max(exp_floor));
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
                    break 'ladder;
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
                        break 'ladder;
                    }
                    let relaxed = if converged(tight.status) {
                        tight
                    } else {
                        relaxed
                    };
                    let fb = finish_budget(iterations_of(&solves, MpccStage::Exact));
                    let finish =
                        finish_from(app, stage, &relaxed, fb, FinishStart::Warm, &mut solves);
                    if halted() {
                        break 'ladder;
                    }
                    path_verdict = finish.clone();
                    cands.push(keep_finish(finish, relaxed, stage.label(), report));
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
                // Skipped once two exact answers are in hand: the
                // continuation's certified answer and the relaxed path's, each
                // from an exact-product solve (its finish converged). Whether
                // they agree ([`answers_agree`]: one point reached twice) or
                // not (two branches, of which the comparison below keeps the
                // better), a third route to the same exact problem is a third
                // opinion. It used to run whenever the two disagreed. Skipping
                // it here changed no outcome on MacMPEC (178 models, every
                // status and objective identical to 1e-9; solve time 1595 s ->
                // 1149 s), mpcc_nl, CALIPSO, the OPF, capacity and transmission
                // suites, or `relu_nn` (42; 178 s -> 88 s, where the relaxed
                // path won all 12 problems the exact stage improved). It still
                // runs when either answer is only a
                // relaxed point: CALIPSO bunny-hop seed 1, whose continuation
                // finish fails (`RestorationFailed`) and leaves the `τ = 1e-8`
                // point, is the one case where the cold solve wins (462.605
                // against 463.195, in 58 iterations).
                let path_answer = cands
                    .first()
                    .filter(|c| accepts(c.status, c.max_abs_product, tol));
                if let (Some((_, cont, _, f_cont)), Some(p)) = (&accepted, path_answer) {
                    let f_path = min_obj(p);
                    let agree =
                        answers_agree(*f_cont, f_path, cont.slack + p.slack, cont.gap + p.gap);
                    if skips_cold_solve(agree, cont.exact_lowering, p.exact_lowering) {
                        let how = if agree {
                            "agree"
                        } else {
                            "both reached the exact product"
                        };
                        report(&format!(
                            "pounce: complementarity: the continuation and the exact stage's \
                             relaxed path {how} (minimized objective {f_cont:.6e} and \
                             {f_path:.6e}); skipping the cold exact product."
                        ));
                        exact = exact_stage_verdict(path_verdict, None);
                        break 'exact_stage cands;
                    }
                }
                let answered = accepted.is_some() || path_answer.is_some();
                let cold_budget = exact_stage_budget(
                    answered,
                    iterations_of(&solves, MpccStage::Continuation)
                        + iterations_of(&solves, MpccStage::Exact),
                )
                .map(|b| b.max(exp_floor));
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
                    break 'ladder;
                }
                exact = exact_stage_verdict(path_verdict, c.clone());
                cands.extend(c);
                cands
            }
            MpccStage::Continuation => {
                let mut last: Option<Captured> = None;
                // Iterations the last converged rung took (its retry budget).
                let mut last_its = 0;
                // Iterations of the attempt that produced the last rung's
                // point, not every attempt in its call ([`warm_rung_max_iter`]).
                let mut last_point_its = 0;
                // The coupled continuation (`complementarity_continuation =
                // coupled`): the first rung as an ordinary relaxation, then
                // one solve with τ tied to μ ([`Lowering::Coupled`]) from the
                // second τ to the last, then the exact finish. Kept on the
                // test the schedule's own answer passes; otherwise the
                // schedule resumes at its second rung, warm from the first.
                //
                // The first rung stays ordinary on purpose: solving the
                // smoothest relaxation fully is what reaches the better
                // basin (MacMPEC `bilevel1`: 0 where coupling from the start
                // settles on 5), and the rungs after it are where the
                // schedule spends (on `relu_nn`, τ = 0.1 and 0.01 alone are
                // 60% of its iterations).
                let last_tau = SCHOLTES_SCHEDULE[SCHOLTES_SCHEDULE.len() - 1];
                let mut resume_below = Number::INFINITY;
                let mut coupled_finish: Option<Option<Captured>> = None;
                if coupled {
                    let (factor, max_cut) = (COUPLED_TAU_PER_MU, COUPLED_MAX_CUT);
                    baseline(app);
                    let first = run(
                        app,
                        stage,
                        Lowering::Relaxed(SCHOLTES_SCHEDULE[0]),
                        None,
                        &mut solves,
                    );
                    if halted() {
                        break 'ladder;
                    }
                    if !converged(first.status) {
                        // The schedule stops at a first rung that does not
                        // converge (below: no earlier rung to retry from), so
                        // this does too, rather than solving it again.
                        resume_below = 0.0;
                    } else {
                        baseline(app);
                        for a in warm_start_assignments(first.statistics.final_mu) {
                            let _ = app.options_mut().read_from_str(&a, true);
                        }
                        // The barrier restarts where τ = factor·μ puts it:
                        // warm at the first rung's tiny μ, the coupled solve
                        // crawls (digits_128x3: 251 iterations against 66).
                        let mu0 = (SCHOLTES_SCHEDULE[1] / factor).clamp(1e-9, 1e-1);
                        let _ = app
                            .options_mut()
                            .read_from_str(&format!("mu_init {mu0:e}\n"), true);
                        // On a budget: where the coupled solve crawls (IEEE 300
                        // transmission cases: 3100 iterations to the floor,
                        // against 900 for the whole schedule) the schedule is
                        // the cheaper route, and it resumes from here.
                        let first_its = first.statistics.iteration_count.max(0) as usize;
                        set_budget(
                            app,
                            (COUPLED_BUDGET_FACTOR * first_its).max(COUPLED_BUDGET_FLOOR),
                        );
                        let c = run(
                            app,
                            stage,
                            Lowering::Coupled {
                                tau0: SCHOLTES_SCHEDULE[1],
                                floor: last_tau,
                                factor,
                                max_cut,
                            },
                            Some(first.warm()),
                            &mut solves,
                        );
                        if halted() {
                            break 'ladder;
                        }
                        let at_floor = wrapper
                            .borrow()
                            .current_tau()
                            .is_some_and(|t| t <= last_tau * (1.0 + 1e-9));
                        let mut kept = false;
                        if converged(c.status) && at_floor {
                            // The exact product is degenerate at the coupled
                            // endpoint, and which start converges from it
                            // differs by model: three, each on a short budget,
                            // the first that converges kept.
                            let mut fin = None;
                            for start in COUPLED_FINISH_STARTS {
                                let f = finish_from(
                                    app,
                                    stage,
                                    &c,
                                    COUPLED_FINISH_BUDGET,
                                    start,
                                    &mut solves,
                                );
                                if halted() {
                                    break 'ladder;
                                }
                                let done = f
                                    .as_ref()
                                    .is_some_and(|f| accepts(f.status, f.max_abs_product, tol));
                                // Kept: the attempt that came closest, not the
                                // last one (it becomes the verdict when nothing
                                // answers under `strategy = continuation`).
                                if done || closer_finish(f.as_ref(), fin.as_ref()) {
                                    fin = f;
                                }
                                if done {
                                    break;
                                }
                            }
                            // Kept on the test the schedule's own answer
                            // passes below: the candidate (the finish, or the
                            // relaxed point it falls back to) converged with
                            // every product in tolerance and certifies tight S.
                            let mut sink = |_: &str| {};
                            let cand =
                                keep_finish(fin.clone(), c.clone(), stage.label(), &mut sink);
                            let answers_tight = accepts(cand.status, cand.max_abs_product, tol)
                                && certify_cached(&cand.x)
                                    .is_some_and(|ct| answers(ct.class) && ct.tight);
                            if answers_tight {
                                // The schedule loop below is skipped
                                // (`resume_below = 0`), so its retry budgets
                                // are not read.
                                kept = true;
                                coupled_kept = true;
                                coupled_finish = Some(fin);
                                resume_below = 0.0;
                                last = Some(c);
                            }
                        }
                        if !kept {
                            report(&format!(
                                "pounce: complementarity: the coupled continuation did not reach \
                                 an exact answer; resuming the schedule at τ = {:.0e}.",
                                SCHOLTES_SCHEDULE[1]
                            ));
                            resume_below = SCHOLTES_SCHEDULE[0];
                            last_its = first.statistics.iteration_count.max(0) as usize;
                            last_point_its = last_its;
                            last = Some(first);
                        }
                    }
                }
                for &tau in SCHOLTES_SCHEDULE
                    .iter()
                    .filter(|&&t| t < resume_below * (1.0 - 1e-12))
                {
                    baseline(app);
                    // Read after `baseline`: a budget is set only while the
                    // fallback that rescues a capped rung will run.
                    let fallback_live = app.is_mu_strategy_fallback_enabled();
                    let warm = last.as_ref().map(|c| {
                        for a in warm_start_assignments(c.statistics.final_mu) {
                            let _ = app.options_mut().read_from_str(&a, true);
                        }
                        // `max_iter` only: unlike a finish's budget, this keeps
                        // `mu_strategy_fallback` live, which re-solves a
                        // crawling rung cold and is what rescued the q98 rung.
                        if let Some(it) =
                            warm_rung_max_iter(fallback_live, last_point_its, caller_max_iter)
                        {
                            let _ = app
                                .options_mut()
                                .read_from_str(&format!("max_iter {it}\n"), true);
                        }
                        c.warm()
                    });
                    let mut c = run(app, stage, Lowering::Relaxed(tau), warm, &mut solves);
                    if halted() {
                        break;
                    }
                    if !converged(c.status) {
                        let Some(prev) = &last else { break };
                        // One weaker rung before giving up on the schedule:
                        // the previous stage's x only, no dual warm start,
                        // but the barrier where that stage left it rather
                        // than back at `mu_init`'s 0.1, and a budget of twice
                        // what that stage took (MacMPEC `siouxfls`: the cold
                        // retry of its failed `1e-7` rung cost 727
                        // iterations). Skipping straight to the finish instead
                        // was measured and dropped: on CALIPSO's rocket the
                        // later rungs reach a better branch (1531.73 against
                        // 1533.81) that the skip loses.
                        baseline(app);
                        // The override carries no duals; a caller's
                        // `warm_start_init_point yes` would ask for them and
                        // refuse the start.
                        let _ = app
                            .options_mut()
                            .read_from_str("warm_start_init_point no\n", true);
                        if let Some(mu) = warm_start_assignments(prev.statistics.final_mu)
                            .into_iter()
                            .find(|a| a.starts_with("mu_init"))
                        {
                            let _ = app.options_mut().read_from_str(&mu, true);
                        }
                        set_budget(app, (2 * last_its).max(RUNG_RETRY_FLOOR));
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
                    last_its = solves.last().map_or(0, |s| s.iterations);
                    last_point_its = c.statistics.iteration_count.max(0) as usize;
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
                        let finish = match coupled_finish.take() {
                            Some(f) => f,
                            None => {
                                let budget =
                                    finish_budget(iterations_of(&solves, MpccStage::Continuation));
                                finish_from(
                                    app,
                                    stage,
                                    &relaxed,
                                    budget,
                                    FinishStart::Warm,
                                    &mut solves,
                                )
                            }
                        };
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
                        vec![keep_finish(finish, relaxed, stage.label(), report)]
                    }
                }
            }
            // Not in STAGES: run only from a reported point, below.
            MpccStage::Branch | MpccStage::Polish => Vec::new(),
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
                let cert = certify_cached(&c.x);
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

    // One branch NLP: `bounds` pinned, `stage` recorded, every option back to
    // the caller's afterwards. The polish and the B-escape both run through
    // it, so a change to how a branch solve is set up is made once.
    let branch_solve = |app: &mut IpoptApplication,
                        stage: MpccStage,
                        bounds,
                        start: StartOverride,
                        warm: Option<(Number, usize)>,
                        solves: &mut Vec<MpccSolveRecord>|
     -> Captured {
        wrapper.borrow_mut().set_branch(Some(bounds));
        app.set_in_retry_sequence(true);
        baseline(app);
        if let Some((final_mu, budget)) = warm {
            for a in warm_start_assignments(final_mu) {
                let _ = app.options_mut().read_from_str(&a, true);
            }
            set_budget(app, budget);
        }
        let c = run(app, stage, Lowering::Ineq, Some(start), solves);
        wrapper.borrow_mut().set_branch(None);
        app.set_in_retry_sequence(false);
        restore(app);
        c
    };

    // The polish ([`natural_residual`]). Every stage stops at a product near
    // the last `τ`, so a pair nearly biactive at the answer keeps both sides
    // near `sqrt(τ)`. A reported point whose largest `min(|G|, |H|)` exceeds
    // [`POLISH_TRIGGER`]` · tol` is re-solved once on the branch holding each
    // pair's smaller side at zero. The re-solved point replaces it if it
    // converged, tightened the residual ([`polish_tightened`]), did not raise
    // the objective by more than removing the residual can cost
    // ([`polish_objective_ok`]), and certifies as the ladder accepts; an S
    // answer is demoted to W only where no branch descends from the new
    // point. Otherwise the point is kept as it was.
    // A B-check the polish already ran at the point it reported (see below).
    let mut seeded_check: Option<BCheck> = None;
    let mut polish: Option<PolishRecord> = None;
    if let Some((reported, chosen, known_cert)) = selection.as_mut() {
        let feasible = matches!(known_cert, Some(Some(ct)) if ct.feasible);
        let found = (feasible && !halted() && accepts(chosen.status, chosen.max_abs_product, tol))
            .then(|| model_at(&inner, &chosen.x, sense))
            .flatten()
            .and_then(|mp| {
                let inp = mp.input(&pairs, solver_tol, tol);
                natural_residual(&inp).map(|(r, held)| (r, branch_bounds(&inp, &held)))
            });
        if let Some((r_old, bounds)) = found.filter(|(r, _)| *r > POLISH_TRIGGER * solver_tol) {
            // Warm from the whole point at the barrier it ended on, as a
            // finish is: the branch can be nonconvex, and from `x` alone the
            // re-solve is pushed `bound_push` off its bounds and may settle in
            // another well (a capacity best response from
            // CapacityEquilibriaBenchmarks.jl moved 5.5e-5 above its global
            // optimum to remove a residual of 1.6e-8). The product rows'
            // multipliers are zeroed: they come from whichever lowering
            // produced the answer (an exact product's can take either sign),
            // and on the branch, with one side of each pair held at zero, those
            // rows are inactive and their multiplier is 0.
            let mut start = chosen.warm();
            let m_inner = wrapper.borrow().m_inner();
            if let Some(rows) = start.lambda.get_mut(m_inner..) {
                rows.fill(0.0);
            }
            let budget = finish_budget(solves.iter().map(|s| s.iterations).sum());
            let c = branch_solve(
                app,
                MpccStage::Polish,
                bounds,
                start,
                Some((chosen.statistics.final_mu, budget)),
                &mut solves,
            );
            if !halted() {
                let f_old = sense * objective_at(&inner, &chosen.x);
                let f_new = sense * objective_at(&inner, &c.x);
                let mp = model_at(&inner, &c.x, sense);
                let inp = mp.as_ref().map(|mp| mp.input(&pairs, solver_tol, tol));
                let cert = inp.as_ref().map(certify);
                let r_new = inp
                    .as_ref()
                    .and_then(natural_residual)
                    .map_or(Number::INFINITY, |(r, _)| r);
                let class = cert.as_ref().filter(|ct| ct.feasible).map(|ct| ct.class);
                let stage = match *reported {
                    Reported::Answer(s)
                    | Reported::WeakFallback(s)
                    | Reported::FeasibleUncertified(s) => Some(s),
                    Reported::Verdict => None,
                };
                let was_answer = matches!(*reported, Reported::Answer(_));
                // The demotion's B-check, kept: if the polished point is
                // taken it is the check the B-escape step would run first at
                // the same `x`, so it seeds that step instead of being run
                // again (up to 1024 branch LPs on MacMPEC `qpec2`).
                let mut demotion_check: Option<BCheck> = None;
                // Cheapest and most basic first: the class check can run a
                // B-check (branch LPs), and a re-solve that did not converge
                // should say so rather than describe its point's class.
                let next = if !accepts(c.status, c.max_abs_product, tol) {
                    Err(format!(
                        "the re-solve returned {}",
                        c.status.upstream_name()
                    ))
                } else if !polish_tightened(r_new, r_old, solver_tol) {
                    Err(format!(
                        "its pairs are still complementary only to {r_new:.1e}"
                    ))
                } else if !polish_objective_ok(f_old, f_new, r_old) {
                    Err(format!(
                        "the minimized objective rose from {f_old:.6e} to {f_new:.6e}, more \
                         than removing {r_old:.1e} can cost"
                    ))
                } else {
                    match stage {
                        Some(s) => polish_class(was_answer, class, || {
                            let bc = inp.as_ref().map(check_b_stationarity);
                            let b = bc
                                .as_ref()
                                .is_some_and(|bc| matches!(bc.verdict, BVerdict::B));
                            demotion_check = bc;
                            b
                        })
                        .map(|answer| {
                            if answer {
                                Reported::Answer(s)
                            } else {
                                Reported::WeakFallback(s)
                            }
                        }),
                        None => Err("the reported point is a verdict, not an answer".into()),
                    }
                };
                polish = Some(PolishRecord {
                    taken: next.is_ok(),
                    residual_before: r_old,
                    residual_after: r_new,
                    reason: next.as_ref().err().cloned(),
                });
                match next {
                    Ok(next) => {
                        report(&format!(
                            "pounce: complementarity: the answer's pairs were complementary only \
                             to {r_old:.1e} (min(|G|,|H|)); re-solved on its branch with each \
                             pair's smaller side held at zero: {r_new:.1e}, minimized objective \
                             {f_old:.6e} -> {f_new:.6e}."
                        ));
                        *reported = next;
                        *chosen = c;
                        *known_cert = Some(cert);
                        seeded_check = demotion_check;
                    }
                    Err(why) => report(&format!(
                        "pounce: complementarity: the answer's pairs are complementary only to \
                         {r_old:.1e} (min(|G|,|H|)), and the branch re-solve did not tighten it \
                         ({why}); keeping it."
                    )),
                }
            }
        }
    }

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
        // The polish's demotion check is at this very point and read B, the
        // loop's first verdict: report it, and the loop has nothing to do.
        if let Some(bc) = seeded_check.filter(|_| eligible(known_cert, chosen)) {
            report(&describe_b_check(&bc));
            b_check = Some(bc);
        }
        while b_check.is_none()
            && !halted()
            && escapes < MAX_BRANCH_ESCAPES
            && eligible(known_cert, chosen)
        {
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
            let start = StartOverride {
                x: chosen.x.clone(),
                ..Default::default()
            };
            let c = branch_solve(app, MpccStage::Branch, bounds, start, None, &mut solves);
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
            natural_residual: None,
            polish,
            polish_threshold: POLISH_TRIGGER * solver_tol,
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
    let nat_res = model_at(&inner, &chosen.x, sense)
        .and_then(|mp| natural_residual(&mp.input(&pairs, solver_tol, tol)).map(|(r, _)| r));
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
        natural_residual: nat_res,
        polish,
        polish_threshold: POLISH_TRIGGER * solver_tol,
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

/// The [`complementarity_gap`] of a captured point in the inner model, its
/// bounds and constraint values read back from the model. `+inf` if a
/// callback declines.
fn gap_at(
    inner: &Rc<RefCell<dyn TNLP>>,
    x: &[Number],
    z_l: &[Number],
    z_u: &[Number],
    lambda: &[Number],
) -> Number {
    let mut m = inner.borrow_mut();
    let Some(info) = m.get_nlp_info() else {
        return Number::INFINITY;
    };
    let (n, mm) = (info.n as usize, info.m as usize);
    if x.len() != n {
        return Number::INFINITY;
    }
    let (mut x_l, mut x_u, mut g_l, mut g_u) =
        (vec![0.0; n], vec![0.0; n], vec![0.0; mm], vec![0.0; mm]);
    let mut g = vec![0.0; mm];
    if !m.get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) || !m.eval_g(x, true, &mut g)
    {
        return Number::INFINITY;
    }
    complementarity_gap(x, z_l, z_u, &x_l, &x_u, &g, lambda, &g_l, &g_u)
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
            slack: if x.is_empty() { Number::INFINITY } else { 0.0 },
            gap: if x.is_empty() { Number::INFINITY } else { 0.0 },
            exact_lowering: true,
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
