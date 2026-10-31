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
    Certificate, CertificateInput, MpccStage, Reported, SAFE_DEFAULTS, SCHOLTES_SCHEDULE, STAGES,
    TOUCHED_OPTIONS, accepts, answers, certify, compared, converged, kept_if_nothing_better,
    warm_start_assignments,
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
    /// Every lowered solve, in order.
    pub solves: Vec<MpccSolveRecord>,
    pub total_iterations: usize,
    /// The forwarded point and its largest `|G·H|`.
    pub x: Vec<Number>,
    pub max_abs_product: Number,
    /// Which stationarity class the forwarded point satisfies
    /// ([`pounce_algorithm::mpcc::certify`]); `None` only if the model could
    /// not be evaluated there.
    pub certificate: Option<Certificate>,
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
        let max_abs_product = w
            .last_products()
            .iter()
            .fold(0.0, |a: Number, p| a.max(p.abs()));
        let cap = Captured {
            status,
            solver_return: w.last_status().unwrap_or(SolverReturn::InternalError),
            statistics: statistics.clone(),
            x: w.last_x().to_vec(),
            z_l: w.last_z_l().to_vec(),
            z_u: w.last_z_u().to_vec(),
            lambda: w.last_lambda().to_vec(),
            max_abs_product: if w.last_x().is_empty() {
                Number::INFINITY
            } else {
                max_abs_product
            },
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

    for (k, &stage) in STAGES.iter().enumerate() {
        // An answer from a compared stage ends the ladder before the first
        // stage that is not compared (the penalty).
        if (accepted.is_some() && !compared(stage)) || halted() {
            break;
        }
        if k > 0 {
            app.set_in_retry_sequence(true);
        }
        baseline(app);
        let candidate = match stage {
            MpccStage::Exact => {
                // `G·H <= 0` where the equality would over-determine the
                // model (same feasible set), as the continuation's finish.
                // Also retried whenever the equality solve was refused
                // before iteration 0 (no point), whatever the status: the
                // degrees-of-freedom gate can refuse an over-determined
                // model with a proven verdict (`Infeasible_Problem_Detected`)
                // rather than `Not_Enough_Degrees_Of_Freedom`, and the
                // inequality form then gives a verdict with a point of its
                // own.
                let mut c = run(app, stage, Lowering::Eq, None, &mut solves);
                if halted() {
                    break;
                }
                if c.status == ApplicationReturnStatus::NotEnoughDegreesOfFreedom || c.x.is_empty()
                {
                    baseline(app);
                    c = run(app, stage, Lowering::Ineq, None, &mut solves);
                    if halted() {
                        break;
                    }
                }
                exact = Some(c.clone());
                Some(c)
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
                        None
                    }
                    Some(relaxed) => {
                        // The finish: exact product from the end of the
                        // schedule; `G·H <= 0` where the equality would
                        // over-determine the model (same feasible set).
                        let mut finish = None;
                        for lowering in [Lowering::Eq, Lowering::Ineq] {
                            baseline(app);
                            for a in warm_start_assignments(relaxed.statistics.final_mu) {
                                let _ = app.options_mut().read_from_str(&a, true);
                            }
                            let c = run(app, stage, lowering, Some(relaxed.warm()), &mut solves);
                            if halted() {
                                break;
                            }
                            // Refused before iteration 0, whatever the status:
                            // the inequality form, as in the exact stage.
                            let over_determined = c.status
                                == ApplicationReturnStatus::NotEnoughDegreesOfFreedom
                                || c.x.is_empty();
                            finish = Some(c);
                            if !over_determined {
                                break;
                            }
                        }
                        if halted() {
                            break;
                        }
                        // A finish that does not converge leaves the
                        // continuation's own answer, which satisfies every
                        // product to the last τ; accepted on the same test.
                        match finish {
                            Some(f) if accepts(f.status, f.max_abs_product, tol) => Some(f),
                            _ => Some(relaxed),
                        }
                    }
                }
            }
            MpccStage::Penalty => {
                let _ = app
                    .options_mut()
                    .read_from_str("l1_exact_penalty_barrier yes\n", true);
                let c = run(app, stage, Lowering::Eq, None, &mut solves);
                if halted() {
                    break;
                }
                Some(c)
            }
        };
        if let Some(c) = &candidate {
            if first
                .as_ref()
                .is_none_or(|f| f.x.is_empty() && !c.x.is_empty())
            {
                first = Some(c.clone());
            }
        }
        if let Some(c) = candidate {
            if accepts(c.status, c.max_abs_product, tol) {
                // Converged and physical; the certificate decides whether it
                // is an answer. A lowered solve can stall short of a biactive
                // pair and still exit "acceptable" with every product at 0.
                let cert = certify_at(&inner, &pairs, &c.x, solver_tol, sense);
                if cert.as_ref().is_none_or(|ct| answers(ct.class)) {
                    let f = match sense * objective_at(&inner, &c.x) {
                        f if f.is_finite() => f,
                        _ => Number::INFINITY,
                    };
                    report(&format!(
                        "pounce: complementarity: {} solved the model ({:?}, max |G·H| = {:.1e}, \
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
                        break;
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
                "pounce: complementarity: {} did not ({:?}, max |G·H| = {:.1e}); trying the next \
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
            solves,
            total_iterations,
            x: c.x,
            max_abs_product: c.max_abs_product,
            certificate: None,
        });
    }
    let (reported, chosen, known_cert) = match (accepted, weak, physical, verdict) {
        (Some((s, c, cert, _)), _, _, _) => (Reported::Answer(s), c, Some(cert)),
        (None, Some((s, c, cert)), _, _) => (Reported::WeakFallback(s), c, Some(cert)),
        // Feasible, but the certificate says not stationary: the lowered
        // solve's success is not the declared model's, so it is reported
        // at the acceptable level rather than as `Solve_Succeeded`.
        (None, None, Some((s, c, cert)), _) => (
            Reported::FeasibleUncertified(s),
            Captured {
                status: ApplicationReturnStatus::SolvedToAcceptableLevel,
                solver_return: SolverReturn::StopAtAcceptablePoint,
                ..c
            },
            Some(cert),
        ),
        (None, None, None, Some(c)) => (Reported::Verdict, c, None),
        (None, None, None, None) => return Err("no exact-product stage ran".into()),
    };
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
        None => certify_at(&inner, &pairs, &chosen.x, solver_tol, sense),
    };
    Ok(MpccOutcome {
        status: chosen.status,
        statistics: chosen.statistics,
        accepted_by,
        fallback,
        reported,
        solves,
        total_iterations,
        x: chosen.x,
        max_abs_product: chosen.max_abs_product,
        certificate,
    })
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

/// Evaluate the *inner* model at `x` and certify it. `None` if any callback
/// declines.
fn certify_at(
    inner: &Rc<RefCell<dyn TNLP>>,
    pairs: &[ComplementarityPair],
    x: &[Number],
    solver_tol: Number,
    sense: Number,
) -> Option<Certificate> {
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
    let jac: Vec<(usize, usize, Number)> = (0..nnz)
        .map(|k| ((ir[k] - off) as usize, (jc[k] - off) as usize, v[k]))
        .collect();
    Some(certify(&CertificateInput {
        x,
        grad_f: &grad_f,
        g: &g,
        jac: &jac,
        x_l: &x_l,
        x_u: &x_u,
        g_l: &g_l,
        g_u: &g_u,
        pairs,
        solver_tol,
    }))
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
