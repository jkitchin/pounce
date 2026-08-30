//! Driver for the second-opinion ladder: applies each rung, re-solves, and
//! resolves what the caller is told.
//!
//! The *policy* — which rungs a failure opens, what each changes, and how the
//! outcome resolves — lives in `pounce_algorithm::second_opinion` and is
//! shared. This module is the half that has to run solves, and it lives here
//! rather than beside the policy because every rung must rebuild the
//! restoration sub-IPM's factory provider (the main IPM rereads its options
//! each solve; the sub-IPM uses the provider snapshotted at the *original*
//! options, so without the rebuild the restoration leg would stay on the
//! settings that just failed) and that provider is defined in this crate.
//!
//! Why this is not inlined in the CLI, where it used to live: a caller driving
//! POUNCE from a modelling layer is the one most likely to hand over an
//! uninitialized starting point — `.nl` leaves any variable without a declared
//! guess at zero, and an origin start is where a squared slack, a homogeneous
//! quadratic or an Ackley gradient loses rank or goes non-finite. The embedder
//! surfaces needed this more than the CLI did, and had it least.
//!
//! Two things the CLI's inlined version did not have to get right, and a
//! library one does:
//!
//! * **Options are restored.** Each rung writes `feral_scaling`, `mu_strategy`
//!   and `start_point_perturbation` into the live options list. The CLI could
//!   leave them there because the process was about to exit; a `Problem` that
//!   gets solved twice cannot, or the second solve silently inherits rung 3's
//!   displaced start. Restored to *set-ness*, not to a value: writing back a
//!   resolved `feral_scaling=auto` an env-configured run never set would
//!   override `POUNCE_FERAL_SCALING` on every later solve.
//! * **A rejected rung never reaches the caller's TNLP.** See
//!   [`SecondOpinionTnlp`].

use crate::resto_alg_builder::RestoAlgorithmBuilder;
use crate::resto_inner_solver::{
    InnerBackendFactoryFactory, make_default_restoration_factory_provider,
};
use pounce_algorithm::application::{
    IpoptApplication, default_backend_factory, feral_config_from_options, ma57_config_from_options,
};
use pounce_algorithm::second_opinion::{
    SecondOpinionAvailability, SecondOpinionTrigger, resolve_scaling_retry_outcome,
    scaling_retry_promoted, second_opinion_rungs,
};
use pounce_common::types::{Index, Number};
use pounce_nlp::SolveStatistics;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, InfeasibilityProof, IpoptCq, IpoptData, IterStats, Linearity, MetaData, NlpInfo,
    ScalingRequest, Solution, SparsityRequest, StartingPoint, TNLP,
};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// What a ladder run reports back.
#[derive(Debug, Clone)]
pub struct SecondOpinionOutcome {
    /// The verdict that ships: the promoted rung's, or the original.
    pub status: ApplicationReturnStatus,
    /// Statistics kept in lockstep with `status` — the promoted rung's on
    /// promotion, otherwise the *original* solve's. Pairing the original
    /// verdict with a failed retry's iteration count is how gh #508's report
    /// disagreed with its own `.sol`.
    pub statistics: SolveStatistics,
    /// Rung labels actually run, in order.
    pub tried: Vec<&'static str>,
    /// The rung whose re-solve was promoted, if any.
    pub promoted_by: Option<&'static str>,
    /// The verdict the ladder was opened on — the *base* solve's, before any
    /// rung ran. Equal to [`Self::status`] whenever nothing was promoted.
    ///
    /// Kept because on a promotion it is the only trace left that the base
    /// solver did not converge, and its absence is gh #850: the reported
    /// status and iteration count both become the promoted rung's, so a
    /// fixture that **lost** its baseline solve and is now only rescued by a
    /// retry reads in `scripts/sweep-fixtures.sh` as a large *improvement*.
    /// `square_flowsheet_resto` went `SolveSucceeded`/116 at `v0.10.0` to
    /// `RestorationFailed`/131 at the base solver plus a promoted 54, and the
    /// sweep showed `116 -> 54`, a 2× win.
    pub base_status: ApplicationReturnStatus,
    /// Iterations the base solve spent before the ladder opened.
    ///
    /// [`Self::statistics`] carries the **promoted rung's** count alone, so
    /// this is what the reported number is missing: on the fixture above the
    /// true cost is `131 + 54`, 3.4× what the report says.
    pub base_iteration_count: usize,
    /// Iterations spent by every rung the ladder ran, in the order of
    /// [`Self::tried`]. With [`Self::base_iteration_count`] this is the whole
    /// cost of the solve.
    pub rung_iteration_counts: Vec<usize>,
}

impl SecondOpinionOutcome {
    /// Every iteration the solve actually spent: the base solve's plus each
    /// rung's. [`Self::statistics`]'s count is one rung's.
    pub fn total_iteration_count(&self) -> usize {
        self.base_iteration_count + self.rung_iteration_counts.iter().sum::<usize>()
    }
}

impl SecondOpinionOutcome {
    /// The no-op outcome: no rung applied, nothing changed. For a caller that
    /// skips the ladder on its own grounds and still wants one type back.
    pub fn unchanged(status: ApplicationReturnStatus, statistics: SolveStatistics) -> Self {
        let base_iteration_count = statistics.iteration_count.max(0) as usize;
        Self {
            status,
            statistics,
            tried: Vec::new(),
            promoted_by: None,
            base_status: status,
            base_iteration_count,
            rung_iteration_counts: Vec::new(),
        }
    }

    /// Did the ladder actually run anything?
    pub fn ran(&self) -> bool {
        !self.tried.is_empty()
    }
}

/// Re-solve a failed solve along up to three different trajectories, and
/// promote one only if it converges.
///
/// Call this *after* a solve that returned a failing status, passing that
/// status and `app.statistics()`. Returns the original status and statistics
/// untouched when the verdict opens no ladder, when every rung is disabled or
/// already the baseline, or when no rung promotes.
///
/// `tnlp` must be the same TNLP the original solve ran on. `report` receives
/// one human-readable progress line per event; pass a no-op closure to run
/// silently.
///
/// **Not for multi-start drivers.** A failed start is routine in a multi-start
/// search, and paying up to three extra solves per failed start multiplies
/// cost for no benefit — `solve_nlp_batch` and the CLI's `minima` search
/// deliberately do not call this.
pub fn run_second_opinion_ladder(
    app: &mut IpoptApplication,
    tnlp: Rc<RefCell<dyn TNLP>>,
    status: ApplicationReturnStatus,
    statistics: SolveStatistics,
    report: &mut dyn FnMut(&str),
) -> SecondOpinionOutcome {
    let Some(trigger) = SecondOpinionTrigger::for_status(status) else {
        return SecondOpinionOutcome::unchanged(status, statistics);
    };
    let avail = SecondOpinionAvailability::from_options(app.options(), trigger);
    let rungs = second_opinion_rungs(avail);
    if rungs.is_empty() {
        return SecondOpinionOutcome::unchanged(status, statistics);
    }

    let restore = OptionSnapshot::take(app);
    report(&format!(
        "pounce: {} — re-solving along {} different trajector{} before \
         believing it (second-opinion ladder: {}).",
        trigger.describe(),
        rungs.len(),
        if rungs.len() == 1 { "y" } else { "ies" },
        rungs.iter().map(|r| r.label).collect::<Vec<_>>().join(", "),
    ));

    // Every rung solves through this gate, so a rung that is not promoted
    // never reaches the caller's `finalize_solution`.
    let gate = Rc::new(RefCell::new(SecondOpinionTnlp::new(tnlp)));
    let gated: Rc<RefCell<dyn TNLP>> = gate.clone();

    let mut retry_status = status;
    let mut retry_stats = statistics.clone();
    let mut tried: Vec<&'static str> = Vec::new();
    let mut rung_iteration_counts: Vec<usize> = Vec::new();
    let mut promoted_by = None;
    let base_status = status;
    let base_iteration_count = statistics.iteration_count.max(0) as usize;
    for rung in &rungs {
        report(&format!(
            "pounce: second opinion — re-solving with {}…",
            rung.label
        ));
        // Undo the previous rung before applying this one. The ladder tests
        // one difference from the *baseline* at a time (gh #524), and the
        // baseline includes which knobs the caller left **unset** — writing
        // an unset knob back as its resolved value is a semantic no-op only
        // if nothing branches on set-ness, and things do:
        // `is_mu_strategy_fallback_enabled` is default-on exactly while
        // `mu_strategy` is unset. The snapshot restores set-ness, so this is
        // the true baseline; the rung then sets only its own knob.
        restore.apply(app);
        for assignment in &rung.assignments {
            let _ = app.options_mut().read_from_str(assignment, true);
        }
        // Rebuild the restoration provider at this rung's options — see the
        // module docs.
        let feral_cfg = feral_config_from_options(app.options());
        // Re-read at this rung's options, same as the feral config above: a
        // rung may set `linear_solver` or an `ma57_*` knob, and the provider is
        // rebuilt precisely so the sub-IPM sees the rung and not the baseline.
        let ma57_cfg = ma57_config_from_options(app.options(), "resto.");
        let bff_mint = move || -> InnerBackendFactoryFactory {
            let feral_cfg = feral_cfg.clone();
            let ma57_cfg = ma57_cfg.clone();
            Box::new(move || default_backend_factory(feral_cfg.clone(), ma57_cfg.clone()))
        };
        app.set_restoration_factory_provider(make_default_restoration_factory_provider(
            RestoAlgorithmBuilder::new(),
            app.algorithm_builder_from_options(),
            bff_mint,
        ));

        retry_status = app.optimize_tnlp(Rc::clone(&gated));
        retry_stats = app.statistics();
        tried.push(rung.label);
        rung_iteration_counts.push(retry_stats.iteration_count.max(0) as usize);
        if scaling_retry_promoted(retry_status) {
            report(&format!(
                "pounce: {} re-solve recovered the problem — promoting ({retry_status:?}).",
                rung.label
            ));
            promoted_by = Some(rung.label);
            break;
        }
        report(&format!(
            "pounce: {} re-solve did not recover ({retry_status:?}).",
            rung.label
        ));
    }
    if promoted_by.is_none() {
        report(&format!(
            "pounce: keeping the original {} verdict; it survived {} \
             independent re-solve(s) ({}).",
            status.upstream_name(),
            tried.len(),
            tried.join(", "),
        ));
    }
    restore.apply(app);

    let (status, statistics) =
        resolve_scaling_retry_outcome(status, retry_status, statistics, retry_stats);
    SecondOpinionOutcome {
        status,
        statistics,
        tried,
        promoted_by,
        base_status,
        base_iteration_count,
        rung_iteration_counts,
    }
}

/// The three options the rungs write, captured well enough to put back.
///
/// `Option<String>` / `Option<f64>` is "was it set", not "what does it read
/// as": an unset `feral_scaling` still *reads* as a resolved strategy, and
/// writing that back would pin a value the caller never chose — overriding
/// `POUNCE_FERAL_SCALING` on every subsequent solve of the same application.
struct OptionSnapshot {
    feral_scaling: Option<String>,
    mu_strategy: Option<String>,
    start_point_perturbation: Option<Number>,
}

impl OptionSnapshot {
    fn take(app: &IpoptApplication) -> Self {
        let opts = app.options();
        let string_if_set = |tag: &str| {
            opts.get_string_value(tag, "")
                .ok()
                .and_then(|(v, found)| found.then_some(v))
        };
        Self {
            feral_scaling: string_if_set("feral_scaling"),
            mu_strategy: string_if_set("mu_strategy"),
            start_point_perturbation: opts
                .get_numeric_value("start_point_perturbation", "")
                .ok()
                .and_then(|(v, found)| found.then_some(v)),
        }
    }

    /// Restore the three knobs to the baseline's **set-ness**, not merely
    /// its values: a knob the caller left unset is `unset_value`d, never
    /// written back as whatever it resolved to. Called before every rung
    /// as well as once at the end, so `&self`.
    fn apply(&self, app: &mut IpoptApplication) {
        let opts = app.options_mut();
        match &self.feral_scaling {
            Some(v) => {
                let _ = opts.set_string_value("feral_scaling", v, true, true);
            }
            None => {
                opts.unset_value("feral_scaling");
            }
        }
        match &self.mu_strategy {
            Some(v) => {
                let _ = opts.set_string_value("mu_strategy", v, true, true);
            }
            None => {
                opts.unset_value("mu_strategy");
            }
        }
        match self.start_point_perturbation {
            Some(v) => {
                let _ = opts.set_numeric_value("start_point_perturbation", v, true, true);
            }
            None => {
                opts.unset_value("start_point_perturbation");
            }
        }
    }
}

/// Transparent TNLP decorator that forwards `finalize_solution` **only for a
/// solve that converged**.
///
/// Each rung is a full `optimize_tnlp` through the same TNLP, so each rung's
/// `finalize_solution` overwrites whatever the previous solve recorded — the
/// IPM's `on_converged` capture, `CountingTnlp`'s, and `PyTnlp`'s
/// `state.final_*` alike. Resolving `status` and the statistics afterwards
/// cannot undo that: it never has a handle on the solution vectors. The result
/// was a caller holding the *original verdict* over the **last rejected
/// rung's iterate** — a point the solver had just decided not to believe —
/// with a status line identical either way, so nothing downstream that checks
/// status could see it. Measured on `cresc100`, `discs`, `launch` (rungs 1–2)
/// and `himmelbj` (rung 3).
///
/// Gating here rather than snapshotting in each caller is what makes the fix
/// hold for every embedder, including ones whose solution state this crate
/// cannot reach. It works because the ladder stops at the first promotion, so
/// at most one rung ever converges and it is by construction the winner; every
/// other rung is silently dropped and the caller keeps the answer the original
/// solve gave it.
///
/// The gate reads `Solution::status`, which is the same convergence fact
/// `scaling_retry_promoted` tests one level up —
/// `Success`/`StopAtAcceptablePoint` there are `Solve_Succeeded`/
/// `Solved_To_Acceptable_Level` here. `promoting_solver_return` and
/// `scaling_retry_promoted` are pinned against each other by test.
pub struct SecondOpinionTnlp {
    inner: Rc<RefCell<dyn TNLP>>,
    /// Number of `finalize_solution` calls suppressed, for tests.
    suppressed: Cell<u32>,
}

/// Does a solve that ended in this `SolverReturn` get to speak to the caller?
///
/// The `SolverReturn` mirror of `scaling_retry_promoted`. Kept as a named
/// function rather than an inline `matches!` so the pairing is testable.
pub fn promoting_solver_return(status: SolverReturn) -> bool {
    matches!(
        status,
        SolverReturn::Success | SolverReturn::StopAtAcceptablePoint
    )
}

impl SecondOpinionTnlp {
    fn new(inner: Rc<RefCell<dyn TNLP>>) -> Self {
        Self {
            inner,
            suppressed: Cell::new(0),
        }
    }

    /// How many rejected rungs were kept away from the caller.
    pub fn suppressed_finalizations(&self) -> u32 {
        self.suppressed.get()
    }
}

impl TNLP for SecondOpinionTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.borrow_mut().get_nlp_info()
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        self.inner.borrow_mut().get_bounds_info(b)
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        self.inner.borrow_mut().get_starting_point(sp)
    }

    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.inner.borrow_mut().eval_f(x, new_x)
    }

    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        self.inner.borrow_mut().eval_grad_f(x, new_x, grad_f)
    }

    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        self.inner.borrow_mut().eval_g(x, new_x, g)
    }

    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        self.inner.borrow_mut().eval_jac_g(x, new_x, mode)
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        self.inner
            .borrow_mut()
            .eval_h(x, new_x, obj_factor, lambda, new_lambda, mode)
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        if !promoting_solver_return(sol.status) {
            self.suppressed.set(self.suppressed.get() + 1);
            return;
        }
        self.inner
            .borrow_mut()
            .finalize_solution(sol, ip_data, ip_cq);
    }

    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        self.inner.borrow_mut().get_var_con_metadata(var, con)
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        self.inner.borrow_mut().get_scaling_parameters(req)
    }

    fn get_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.borrow_mut().get_variables_linearity(types)
    }

    fn get_objective_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner
            .borrow_mut()
            .get_objective_variables_linearity(types)
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.borrow_mut().get_constraints_linearity(types)
    }

    fn get_number_of_nonlinear_variables(&mut self) -> Index {
        self.inner.borrow_mut().get_number_of_nonlinear_variables()
    }

    fn derivative_proofs(&mut self) -> pounce_nlp::constant_derivatives::DerivativeProofs {
        self.inner.borrow_mut().derivative_proofs()
    }

    fn get_list_of_nonlinear_variables(&mut self, pos: &mut [Index]) -> bool {
        self.inner.borrow_mut().get_list_of_nonlinear_variables(pos)
    }

    fn intermediate_callback(
        &mut self,
        stats: IterStats,
        ip_data: &IpoptData,
        ip_cq: &IpoptCq,
    ) -> bool {
        self.inner
            .borrow_mut()
            .intermediate_callback(stats, ip_data, ip_cq)
    }

    fn finalize_metadata(&mut self, var: &MetaData, con: &MetaData) {
        self.inner.borrow_mut().finalize_metadata(var, con)
    }

    fn presolve_infeasibility_proof(&self) -> Option<InfeasibilityProof> {
        self.inner.borrow().presolve_infeasibility_proof()
    }

    // The two below are what makes a decorator *transparent* rather than
    // merely forwarding: they are what a caller above uses to see what is
    // underneath. Falling through to the trait defaults reports `false` /
    // `None`, i.e. "there is no presolve wrapper and no scaling under me",
    // which is a lie this type is in no position to tell -- it does not know
    // what it wraps.
    //
    // No live path reaches the failure today: `optimize_tnlp` only calls
    // `wrap_from_options` when `presolve_already_applied` is unset, the CLI
    // sets it, and no other entry point pre-wraps. It bites a caller who
    // calls `wrap_with_presolve` themselves *and* leaves `presolve=yes` --
    // the original solve gets one wrapper and every rung gets two, so the
    // rung stops being "baseline plus one knob", which is the whole contract
    // (gh #524). Forwarding costs two lines and removes the need to re-derive
    // that reachability argument every time something upstream moves.
    fn is_presolve_wrapper(&self) -> bool {
        self.inner.borrow().is_presolve_wrapper()
    }

    fn scaling_factors(&self) -> Option<Vec<Number>> {
        self.inner.borrow().scaling_factors()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_algorithm::second_opinion::scaling_retry_promoted;
    use pounce_common::options_list::OptionsList;

    /// A model that *is* a presolve wrapper and *does* carry scaling factors,
    /// so the decorator has something real to hide.
    struct Underneath;

    impl TNLP for Underneath {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            None
        }
        fn get_bounds_info(&mut self, _b: BoundsInfo<'_>) -> bool {
            false
        }
        fn get_starting_point(&mut self, _sp: StartingPoint<'_>) -> bool {
            false
        }
        fn eval_f(&mut self, _x: &[Number], _n: bool) -> Option<Number> {
            None
        }
        fn eval_grad_f(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool {
            false
        }
        fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool {
            false
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _n: bool,
            _mode: SparsityRequest<'_>,
        ) -> bool {
            false
        }
        fn eval_h(
            &mut self,
            _x: Option<&[Number]>,
            _n: bool,
            _o: Number,
            _l: Option<&[Number]>,
            _nl: bool,
            _mode: SparsityRequest<'_>,
        ) -> bool {
            false
        }
        fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {}
        fn is_presolve_wrapper(&self) -> bool {
            true
        }
        fn scaling_factors(&self) -> Option<Vec<Number>> {
            Some(vec![2.0, 4.0])
        }
    }

    /// The decorator has to answer these two for the model *underneath* it.
    /// Both have trait defaults that are plausible-looking lies (`false` /
    /// `None`) for a decorator, so forgetting to forward one is silent: the
    /// caller above concludes there is no presolve wrapper and no scaling
    /// below, and a caller who pre-wrapped gets a second presolve on every
    /// rung — which is exactly the "one difference from baseline" contract
    /// (gh #524) broken.
    #[test]
    fn the_decorator_is_transparent_about_what_is_underneath_it() {
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Underneath));
        let deco = SecondOpinionTnlp::new(inner);
        assert!(
            deco.is_presolve_wrapper(),
            "the decorator hid a presolve wrapper below it",
        );
        assert_eq!(
            deco.scaling_factors(),
            Some(vec![2.0, 4.0]),
            "the decorator hid the scaling factors below it",
        );
    }

    /// The two promotion predicates sit at different levels — one reads the
    /// `SolverReturn` the TNLP is handed, the other the
    /// `ApplicationReturnStatus` the ladder loop sees — and they have to agree
    /// about which solves count, or the gate suppresses the winner's
    /// `finalize_solution` (caller keeps a stale iterate under a promoted
    /// verdict) or lets a loser's through (the leak this gate exists to close).
    /// Pinned by hand-written pairs because the mapping between the two enums
    /// is private to `pounce-algorithm`.
    #[test]
    fn the_two_promotion_predicates_agree_variant_for_variant() {
        let pairs = [
            (
                SolverReturn::Success,
                ApplicationReturnStatus::SolveSucceeded,
            ),
            (
                SolverReturn::StopAtAcceptablePoint,
                ApplicationReturnStatus::SolvedToAcceptableLevel,
            ),
            (
                SolverReturn::LocalInfeasibility,
                ApplicationReturnStatus::InfeasibleProblemDetected,
            ),
            (
                SolverReturn::InvalidNumberDetected,
                ApplicationReturnStatus::InvalidNumberDetected,
            ),
            (
                SolverReturn::MaxiterExceeded,
                ApplicationReturnStatus::MaximumIterationsExceeded,
            ),
            (
                SolverReturn::RestorationFailure,
                ApplicationReturnStatus::RestorationFailed,
            ),
            (
                SolverReturn::ErrorInStepComputation,
                ApplicationReturnStatus::ErrorInStepComputation,
            ),
            (
                SolverReturn::DivergingIterates,
                ApplicationReturnStatus::DivergingIterates,
            ),
            (
                SolverReturn::UserRequestedStop,
                ApplicationReturnStatus::UserRequestedStop,
            ),
            (
                SolverReturn::FeasiblePointFound,
                ApplicationReturnStatus::FeasiblePointFound,
            ),
        ];
        for (solver, app) in pairs {
            assert_eq!(
                promoting_solver_return(solver),
                scaling_retry_promoted(app),
                "{solver:?} / {app:?} disagree about promotion",
            );
        }
    }

    /// Minimal TNLP that records how many times it was finalized and with what.
    struct Recorder {
        finalized: Vec<(SolverReturn, Vec<Number>)>,
    }

    impl TNLP for Recorder {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            None
        }
        fn get_bounds_info(&mut self, _b: BoundsInfo<'_>) -> bool {
            false
        }
        fn get_starting_point(&mut self, _sp: StartingPoint<'_>) -> bool {
            false
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            None
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
            false
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
            false
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _mode: SparsityRequest<'_>,
        ) -> bool {
            false
        }
        fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
            self.finalized.push((sol.status, sol.x.to_vec()));
        }
    }

    fn finalize_through(gate: &mut SecondOpinionTnlp, status: SolverReturn, x: &[Number]) {
        let empty: &[Number] = &[];
        gate.finalize_solution(
            Solution {
                status,
                x,
                z_l: empty,
                z_u: empty,
                g: empty,
                lambda: empty,
                obj_value: 0.0,
            },
            &IpoptData::default(),
            &IpoptCq::default(),
        );
    }

    /// The gate's whole job: a rung the ladder is about to reject must not
    /// leave its iterate in the caller's solution state. Three rejected rungs
    /// followed by nothing means the caller still holds what it had.
    #[test]
    fn a_rejected_rung_never_reaches_the_callers_tnlp() {
        let inner = Rc::new(RefCell::new(Recorder {
            finalized: Vec::new(),
        }));
        let mut gate = SecondOpinionTnlp::new(inner.clone() as Rc<RefCell<dyn TNLP>>);
        finalize_through(&mut gate, SolverReturn::LocalInfeasibility, &[1.0]);
        finalize_through(&mut gate, SolverReturn::InvalidNumberDetected, &[2.0]);
        finalize_through(&mut gate, SolverReturn::MaxiterExceeded, &[3.0]);
        assert!(inner.borrow().finalized.is_empty());
        assert_eq!(gate.suppressed_finalizations(), 3);
    }

    /// …and the winner does reach it. The ladder stops at the first promotion,
    /// so at most one rung ever gets here, and it is by construction the one
    /// whose verdict ships.
    #[test]
    fn a_promoted_rung_reaches_the_callers_tnlp() {
        for promoting in [SolverReturn::Success, SolverReturn::StopAtAcceptablePoint] {
            let inner = Rc::new(RefCell::new(Recorder {
                finalized: Vec::new(),
            }));
            let mut gate = SecondOpinionTnlp::new(inner.clone() as Rc<RefCell<dyn TNLP>>);
            finalize_through(&mut gate, SolverReturn::LocalInfeasibility, &[1.0]);
            finalize_through(&mut gate, promoting, &[7.0]);
            let seen = inner.borrow().finalized.clone();
            assert_eq!(seen.len(), 1, "{promoting:?}");
            assert_eq!(seen[0].0, promoting);
            assert_eq!(seen[0].1, vec![7.0]);
            assert_eq!(gate.suppressed_finalizations(), 1);
        }
    }

    fn opts_with(pairs: &[(&str, &str)]) -> OptionsList {
        let mut o = OptionsList::new();
        for (k, v) in pairs {
            o.set_string_value(k, v, true, true).unwrap();
        }
        o
    }

    /// The loop's core invariant, and the regression that motivated it.
    ///
    /// Rung N must see the *baseline*, not rung N-1's leftovers, and
    /// "baseline" includes which knobs the caller left **unset**. The rungs
    /// used to undo each other by writing the baseline's resolved value back,
    /// which is a no-op by value and a change by set-ness — and set-ness is
    /// read: `is_mu_strategy_fallback_enabled` is default-on only while
    /// `mu_strategy` is unset, so rung 3 re-asserting a resolved `monotone`
    /// switched pounce's own μ-strategy stall retry off for that rung. On
    /// KRONOS `a18_ackley1` that is `Solve_Succeeded` in 237 iterations
    /// versus `Maximum_Iterations_Exceeded` at 3000.
    ///
    /// So: after the driver's restore-then-assign, exactly the rung's own
    /// knob is set and the other two read as unset.
    #[test]
    fn each_rung_starts_from_the_baseline() {
        let mut app = IpoptApplication::new();
        // A baseline that sets none of the three knobs — the common case, and
        // the only one in which the old write-back was observable.
        let baseline = OptionSnapshot::take(&app);
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            trigger: SecondOpinionTrigger::LocalInfeasibility,
            scaling_retry_enabled: true,
            mu_retry_enabled: true,
            perturbed_start_retry_enabled: true,
            already_mc64: false,
            already_adaptive: false,
            already_perturbed: false,
            baseline_scaling: Some("auto"),
        });
        assert_eq!(rungs.len(), 3, "{:?}", rung_labels(&rungs));

        let expected = [
            ("feral_scaling=mc64", [true, false, false]),
            ("mu_strategy=adaptive", [false, true, false]),
            ("start_point_perturbation=1e-2", [false, false, true]),
        ];
        for (rung, (label, want)) in rungs.iter().zip(expected) {
            assert_eq!(rung.label, label);
            // Exactly what the driver does per rung.
            baseline.apply(&mut app);
            for a in &rung.assignments {
                app.options_mut().read_from_str(a, true).unwrap();
            }
            let opts = app.options();
            let got = [
                matches!(opts.get_string_value("feral_scaling", ""), Ok((_, true))),
                matches!(opts.get_string_value("mu_strategy", ""), Ok((_, true))),
                matches!(
                    opts.get_numeric_value("start_point_perturbation", ""),
                    Ok((_, true))
                ),
            ];
            assert_eq!(
                got, want,
                "rung `{label}`: set-ness of [feral_scaling, mu_strategy, \
                 start_point_perturbation] is wrong",
            );
        }
    }

    fn rung_labels(
        rungs: &[pounce_algorithm::second_opinion::SecondOpinionRung],
    ) -> Vec<&'static str> {
        rungs.iter().map(|r| r.label).collect()
    }

    /// An option the caller never set must come back *unset*, not set to
    /// whatever it read as. `feral_scaling` is the case that matters: unset it
    /// resolves through `FeralConfig::from_env()`, so writing the resolved tag
    /// back would pin `POUNCE_FERAL_SCALING`'s value into the options list and
    /// silently override the environment on every later solve of the same
    /// application.
    #[test]
    fn restoring_an_unset_option_leaves_it_unset() {
        let mut opts = opts_with(&[]);
        // Stand in for what a rung does to the live options list.
        opts.set_string_value("feral_scaling", "mc64", true, true)
            .unwrap();
        opts.set_numeric_value("start_point_perturbation", 1e-2, true, true)
            .unwrap();
        assert!(opts.get_string_value("feral_scaling", "").unwrap().1);
        // …and what the snapshot taken *before* it must put back.
        assert!(opts.unset_value("feral_scaling"));
        assert!(opts.unset_value("start_point_perturbation"));
        assert!(!opts.get_string_value("feral_scaling", "").unwrap().1);
        assert!(
            !opts
                .get_numeric_value("start_point_perturbation", "")
                .unwrap()
                .1
        );
    }

    /// A caller's explicit setting survives the ladder unchanged — the other
    /// half of the round trip.
    #[test]
    fn restoring_a_set_option_puts_the_callers_value_back() {
        let mut opts = opts_with(&[("mu_strategy", "adaptive")]);
        let before = opts
            .get_string_value("mu_strategy", "")
            .ok()
            .and_then(|(v, found)| found.then_some(v));
        assert_eq!(before.as_deref(), Some("adaptive"));
        opts.set_string_value("mu_strategy", "monotone", true, true)
            .unwrap();
        opts.set_string_value("mu_strategy", before.as_ref().unwrap(), true, true)
            .unwrap();
        assert_eq!(
            opts.get_string_value("mu_strategy", "").unwrap().0,
            "adaptive"
        );
    }
}
