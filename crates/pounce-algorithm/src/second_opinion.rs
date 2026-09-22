//! The second-opinion ladder: re-solving a failure along a different
//! trajectory before believing it.
//!
//! A local-infeasibility verdict on a nonconvex problem is a *local*
//! statement, and an `Invalid_Number_Detected` is a statement about the
//! callbacks at one point. Both are frequently artifacts of the trajectory
//! the solve happened to take, or of the point it started from, rather than
//! facts about the model. The ladder re-solves along up to three deliberately
//! different trajectories and promotes a re-solve only if it converges.
//!
//! This module holds the **policy** — which rungs apply to which failure, what
//! each rung changes, and how a ladder's outcome resolves. It reads options
//! and returns descriptions; it runs no solves. The **driver** that applies a
//! rung and calls back into the solver lives in `pounce-restoration`, because
//! each rung has to rebuild the restoration sub-IPM's factory provider and
//! that provider is defined one crate up the dependency graph.
//!
//! Split out of `pounce-cli`'s `main.rs` so the Python, C and Rust embedding
//! surfaces get the same ladder rather than each re-deriving it — the
//! asymmetry mattered, because a caller driving POUNCE from a modelling layer
//! is precisely the one most likely to hand over an uninitialized (and so
//! all-zero, and so possibly rank-deficient) starting point.

use pounce_common::options_list::OptionsList;
use pounce_nlp::SolveStatistics;
use pounce_nlp::return_codes::ApplicationReturnStatus;

/// One rung of the local-infeasibility second-opinion ladder: a label for the
/// console plus the option assignments that define this re-solve's trajectory.
///
/// Assignments are applied on top of the *baseline* options, not on top of the
/// previous rung — see `second_opinion_rungs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecondOpinionRung {
    pub label: &'static str,
    /// The knob this rung varies, and *only* that knob — one
    /// `read_from_str` line per assignment.
    ///
    /// A rung does not carry lines undoing the earlier rungs. It used to,
    /// and that was a defect: the undo was written as the baseline's
    /// *resolved* value, so a knob the caller never set came back **set**.
    /// `mu_strategy` is the one that bites — `is_mu_strategy_fallback_enabled`
    /// is default-on only while `mu_strategy` is unset, so rung 3 writing
    /// back a resolved `monotone` silently switched off pounce's own
    /// μ-strategy stall retry for the duration of the rung. Measured on
    /// KRONOS `a18_ackley1`: the displaced solve stalls at `max_iter`, the
    /// flip that would have certified it in 237 iterations never fires, and
    /// the ladder reports no recovery. Restoring by *value* is not the same
    /// as restoring by *set-ness*.
    ///
    /// The driver restores the baseline with
    /// `OptionSnapshot::apply` before every rung, which does honour
    /// set-ness, so a rung starts from the true baseline by construction.
    pub assignments: Vec<String>,
}

/// Which failure opened the ladder. Not every rung is evidence about every
/// failure: an `Invalid_Number_Detected` is a statement about the *callbacks*
/// at a point, and re-running the same callbacks at the same point under a
/// different linear-solver scaling or a different barrier strategy evaluates
/// the same non-finite quantity again. Only the rung that moves the point
/// applies there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondOpinionTrigger {
    /// `Infeasible_Problem_Detected` — a *local* statement about a nonconvex
    /// problem, which every rung is evidence against.
    LocalInfeasibility,
    /// `Invalid_Number_Detected` — a NaN or infinity out of the model.
    InvalidNumber,
    /// `Restoration_Failed` — the restoration phase could not find a point
    /// the filter would accept. Like the two above and unlike a budget exit,
    /// this is a statement about the trajectory the solve happened to take,
    /// not about the model; see [`SecondOpinionTrigger::for_status`].
    RestorationFailure,
    /// `Error_In_Step_Computation` — the step computation broke down: the
    /// KKT system could not be made to yield a usable step, even after
    /// regularization. Like a restoration failure and unlike a budget exit,
    /// this is a statement about the trajectory the solve happened to take,
    /// so a different barrier schedule is evidence against it.
    ///
    /// Measured on `deb7` (found via gh#960): the wasm32 build, whose factorization
    /// rounds differently from the native one, follows the native trajectory
    /// for 83 iterations, diverges by one ulp in `inf_du` at iteration 84,
    /// and ends `Error_In_Step_Computation` at 160 where native solves in
    /// 147. Under `mu_strategy=adaptive` the same binary and the same model
    /// reach the same optimum as native — `Solve_Succeeded`, 131 iterations,
    /// objective 97.559938 — so the rescue was already in the ladder and
    /// only the trigger was missing. This opens **only** that rung: on the
    /// same case `feral_scaling=mc64` still fails at 154 iterations,
    /// `feral_increase_quality=no` at 201, and `start_point_perturbation`
    /// spends 324 iterations to fail anyway.
    StepComputationError,
    /// `Maximum_Iterations_Exceeded` — **and only when the solve escalated
    /// the linear solver's factorization quality at least once** (gh#857).
    ///
    /// This is the one trigger that is not a property of the verdict alone,
    /// and the exception it carves out of the paragraph on
    /// [`SecondOpinionTrigger::for_status`] is narrow on purpose. A budget
    /// exit normally wants a bigger budget, not a different trajectory —
    /// but a `feral_increase_quality` escalation *changes which pivots are
    /// taken and never steps back down*, so when one fired, the wall the
    /// solve hit may be the escalated trajectory's rather than the model's,
    /// and a bigger budget re-runs the same wall. On
    /// `square_flowsheet_resto`'s lbfgs leg the escalated path reaches 3000
    /// iterations and the un-escalated one converges in 178.
    ///
    /// The escalation count is what keeps this from opening a ladder on
    /// every budget exit; it comes from the `quality_escalations` statistic
    /// and is checked in [`second_opinion_rungs`], not here — `for_status`
    /// only sees a status. A solve that escalated zero times produces an
    /// empty rung list and the driver returns before it narrates anything.
    IterationLimit,
}

/// What the baseline options already provide, so a rung that would be a no-op
/// can be dropped instead of burning a solve to re-derive the same answer.
#[derive(Debug, Clone, Copy)]
pub struct SecondOpinionAvailability {
    pub trigger: SecondOpinionTrigger,
    pub scaling_retry_enabled: bool,
    pub mu_retry_enabled: bool,
    pub perturbed_start_retry_enabled: bool,
    pub already_mc64: bool,
    pub already_adaptive: bool,
    /// The baseline already displaces the start, so there is no displacement
    /// left for the third rung to add that the failing solve did not have.
    pub already_perturbed: bool,
    /// `feral_increase_quality_retry` (gh#857), rung 4's own enable.
    pub increase_quality_retry_enabled: bool,
    /// The baseline already ran with `feral_increase_quality=no`, so rung 4
    /// would re-run the solve that just failed.
    pub already_no_increase_quality: bool,
    /// How many times the failing solve's linear solver actually accepted an
    /// `increase_quality` escalation — the `quality_escalations` statistic.
    ///
    /// The only entry here that is a *measurement of the solve* rather than
    /// a reading of its options, and rung 4 is unimplementable without it:
    /// an escalation moves no field a report carries, so "did this solve
    /// take the rung that has a documented losing direction" cannot be
    /// answered from the verdict. `0` means provably not a candidate, and
    /// rung 4 is dropped — which is what stops a budget exit on a
    /// never-escalating model from paying for an extra solve.
    ///
    /// It is a gate at `>= 1`, deliberately **not** a threshold. `deb7` and
    /// `square_flowsheet_resto`'s base solve escalate exactly twice each on
    /// their exact legs, one gaining the solve and one losing it, so no
    /// count separates them; the verdict does.
    pub baseline_quality_escalations: u64,
    /// `feral_scaling` tag naming the baseline's *resolved* scaling strategy.
    /// `None` under `ScalingStrategy::External`, which drops rungs 2 and 3.
    ///
    /// Now purely a gate. It was the tag those rungs wrote back to undo rung
    /// 1, and `None` meant there was nothing to write; the driver's snapshot
    /// restores by set-ness instead, so no tag is needed and the External
    /// case would in fact be safe to run. The gate is kept because dropping
    /// it adds two rungs on externally-scaled models — a trajectory change,
    /// and a separate decision from this bug fix.
    pub baseline_scaling: Option<&'static str>,
}

/// Build the ladder of second-opinion re-solves for a failing verdict, in the
/// order they should be tried. Each rung varies exactly one knob from the
/// baseline options.
///
/// 1. **`feral_scaling=mc64` — numerical diversity**
///    (`feral_infeasibility_scaling_retry`, on by default). Some KKT
///    trajectories are chaotic: under two equally backward-stable
///    linear-solver scalings the iterates stay bit-identical for many
///    iterations, then diverge by ~1 ULP and fall into different basins — one
///    optimal, the other a spurious stationary point of the constraint
///    violation (`discs.nl`: InfNorm → infeasible, MC64/Identity/MA57/IPOPT →
///    optimal). Sensitive dependence, not a bad solve, so the a-priori
///    scaling router cannot tell the two apart and no per-factor residual
///    flags it; the only reliable signal is the whole-solve verdict.
///
/// 2. **`mu_strategy=adaptive` — algorithmic diversity**
///    (`infeasibility_mu_strategy_retry`, on by default). Rung 1 perturbs only
///    the linear algebra, so it is evidence *only* when the trajectory is
///    ULP-hypersensitive. When it is not, MC64 retraces the same iterates and
///    agrees for the same reason the first solve was wrong — on gh #524
///    (`cresc4`, 6 vars / 8 constraints, feasible, Ipopt solves it in 71
///    iterations) the MC64 re-solve reproduced the original trajectory
///    bit-identically and "corroborated" the false verdict. A different
///    barrier strategy changes the iterate sequence itself, which is what the
///    monotone-µ default gets wrong here: adaptive µ walks to the known
///    optimum. This is also the remedy IPOPT's own documentation gives a user
///    who gets an infeasibility verdict on a problem they believe is feasible;
///    running it automatically just spares them the round trip.
///
/// 3. **`start_point_perturbation=1e-2` — a different starting point**
///    (`infeasibility_perturbed_start_retry`, on by default). The only rung
///    that moves the point rather than the path, and so the only one that is
///    evidence about an `Invalid_Number_Detected` or a `Restoration_Failed`
///    — see [`SecondOpinionTrigger`]. Those two triggers open this rung and
///    nothing else, so they cost exactly one extra solve.
///
/// 4. **`feral_increase_quality=no` — undo the factorization escalation**
///    (`feral_increase_quality_retry`, on by default; gh#857). The only rung
///    whose gate is a *measurement of the failing solve* rather than a
///    reading of its options: it opens on a `Restoration_Failed`, a
///    `Maximum_Iterations_Exceeded` or an `Infeasible_Problem_Detected`
///    **and** only when the solve's `quality_escalations` count is at
///    least 1.
///
///    The infeasibility trigger was added after the other two, and the
///    reason is worth keeping: `square_flowsheet_resto`'s lbfgs leg does
///    **not** exit the same way on every platform. On macOS/arm64 it runs
///    to the 3000-iteration cap and exits `Maximum_Iterations_Exceeded`;
///    on linux/x86_64 the same 3000 iterations with the same 25
///    escalations end `Infeasible_Problem_Detected` instead — a *wrong
///    answer* on a feasible model, and one the three infeasibility rungs
///    above do not recover. The verdict an escalation-rerouted trajectory
///    produces is not a property of the escalation, so pinning the rung to
///    two of the three shapes left the fix firing on one platform and not
///    the other.
///
///    Unlike the other two triggers this one is **not** free, and the
///    difference is worth stating: a `Restoration_Failed` or a budget exit
///    is a failure either way, so the rung's cost lands only on runs that
///    were already going to report one, whereas a *genuine* infeasibility
///    verdict is a correct answer and the rung can only confirm it. Six
///    fixture-legs pay exactly that — one extra solve, e.g.
///    `issue_508_infeasible_gap_1em4` 982 → 1423 total iterations, with no
///    status, objective, iteration count or engine moving. The `>= 1` gate
///    is what bounds it: of the eight NLP-arm infeasibility fixture-legs,
///    four escalated and take the rung and four are untouched.
///    `feral_increase_quality` is
///    on by default and genuinely two-sided — it buys accuracy and 15–25% of
///    the iterations on several fixture-legs and loses whole solves on others
///    — and its losing direction previously had no automatic recovery at all.
///    Measured on `square_flowsheet_resto`: the lbfgs leg escalates 25 times,
///    hits the 3000-iteration cap, and converges in 178 with the rung off.
///
///    It is **appended**, not inserted, so a `Restoration_Failed` that rung 3
///    already recovers (the gh#815 family, and this same fixture's exact leg)
///    reaches promotion first and costs nothing new. The `>= 1` is a gate and
///    not a threshold: `deb7` escalates exactly as many times as
///    `square_flowsheet_resto`'s base solve and *gains* by it, so a count
///    cannot separate the two — only the verdict can, and `deb7`'s is
///    `Optimal`.
///
///    Opening this rung also stands the µ-strategy stall retry down
///    (`Application::run_with_mu_strategy_fallback`), which is what keeps it
///    from being a third solve. That retry fires unconditionally on
///    `Maximum_Iterations_Exceeded`, so before gh#857 this fixture's lbfgs leg
///    paid 3000 capped iterations, then a second full 3000 under the flipped
///    schedule that escalated 25 times again and ended no better, and only
///    then reached this rung's 178. The flip is blind and the escalation is
///    measured; skipping it takes the run from three solves to two, changing
///    no reported number — which is also why the fixture sweep is
///    byte-identical across that change.
///
/// Rungs are **not** cumulative: the driver restores the baseline before each
/// rung, so rung 2 runs without rung 1's scaling and rung 3 without either
/// earlier knob. That reset is load-bearing, not tidiness: on gh #524's
/// `cresc4`, `mu_strategy=adaptive` recovers the optimum but
/// `mu_strategy=adaptive` with `feral_scaling=mc64` still reports local
/// infeasibility, so a cumulative ladder would have discarded the fix.
pub fn second_opinion_rungs(avail: SecondOpinionAvailability) -> Vec<SecondOpinionRung> {
    let mut rungs = Vec::new();
    let infeasible = avail.trigger == SecondOpinionTrigger::LocalInfeasibility;
    if infeasible && avail.scaling_retry_enabled && !avail.already_mc64 {
        rungs.push(SecondOpinionRung {
            label: "feral_scaling=mc64",
            assignments: vec!["feral_scaling mc64\n".to_string()],
        });
    }
    // The barrier-schedule rung is the one rung a step-computation
    // breakdown opens, and the only one measured to recover one: the
    // trajectory walked into a KKT system no regularization could make
    // usable, and a different μ schedule is a different trajectory. See
    // `SecondOpinionTrigger::StepComputationError` for the deb7 numbers.
    if avail.baseline_scaling.is_some()
        && (infeasible || avail.trigger == SecondOpinionTrigger::StepComputationError)
        && avail.mu_retry_enabled
        && !avail.already_adaptive
    {
        rungs.push(SecondOpinionRung {
            label: "mu_strategy=adaptive",
            assignments: vec!["mu_strategy adaptive\n".to_string()],
        });
    }
    // Rung 3 varies exactly one knob from the *baseline*, so both earlier
    // rungs' knobs must be undone first, not inherited — gh #524 is the case
    // where stacking two of them threw the fix away. That undo is the
    // driver's job (`OptionSnapshot::apply` before each rung), not an
    // assignment here; see the note on `assignments`.
    //
    // Not on the iteration-limit trigger. That trigger is gated on a
    // measurement (gh#857, rung 4 below) and exists to test one hypothesis —
    // that the escalation is what walked the solve into the wall. Displacing
    // the start of a solve that ran out of budget tests nothing: it starts a
    // fresh trajectory with the same budget and the same escalating ladder
    // waiting for it. Opening it here would put an extra solve on every
    // escalating budget exit for no reason anyone has measured.
    // Not on the step-computation trigger either, and for a measured
    // reason rather than the reasoning above: on deb7 it spends 324
    // iterations and fails anyway, where the rung above solves in 131.
    if avail.baseline_scaling.is_some()
        && !matches!(
            avail.trigger,
            SecondOpinionTrigger::IterationLimit | SecondOpinionTrigger::StepComputationError
        )
        && avail.perturbed_start_retry_enabled
        && !avail.already_perturbed
    {
        rungs.push(SecondOpinionRung {
            label: "start_point_perturbation=1e-2",
            assignments: vec!["start_point_perturbation 1e-2\n".to_string()],
        });
    }
    // 4. `feral_increase_quality=no` — undo the escalation (gh#857).
    //
    // Appended, never prepended. On a `Restoration_Failed` the rung above
    // already recovers the gh#815 family *and* `square_flowsheet_resto`'s own
    // exact leg, and it promotes and breaks before this one runs, so those
    // solves cost nothing new. This rung is what is left when that fails, and
    // it is the whole ladder on an `Maximum_Iterations_Exceeded`.
    //
    // The `>= 1` is the gate the trigger's doc describes: a solve that never
    // escalated cannot have been rerouted by an escalation, so there is
    // nothing here to test and no solve to spend.
    if matches!(
        avail.trigger,
        SecondOpinionTrigger::RestorationFailure
            | SecondOpinionTrigger::IterationLimit
            | SecondOpinionTrigger::LocalInfeasibility
    ) && avail.increase_quality_retry_enabled
        && !avail.already_no_increase_quality
        && avail.baseline_quality_escalations >= 1
    {
        rungs.push(SecondOpinionRung {
            label: "feral_increase_quality=no",
            assignments: vec!["feral_increase_quality no\n".to_string()],
        });
    }
    rungs
}

/// Whether the ladder's narration should reach the console.
///
/// `print_level 0` is a request for silence, and the ladder's running
/// commentary is no more exempt from it than the `EXIT:` block — the C
/// interface is an Ipopt drop-in, where `print_level=0 sb=yes` is the
/// documented way to get a quiet solve, and up to five unexpected `pounce:`
/// lines on a failing one is exactly what that asks not to happen. The ladder
/// still *runs*; only the console is quiet.
///
/// Lives here rather than at each call site because the CLI and the C
/// interface both need it and a duplicated branch is a branch that can drift
/// — one of them silently losing the gate would look identical to the other
/// keeping it. `pounce-rs` and Python do not call this: they never print,
/// they hand the narration back to the caller.
///
/// Only an *explicit* `print_level` silences: a level nobody set narrates,
/// and so does an unregistered or unreadable one. That is the pre-gate
/// behaviour and the safe direction — too much on a failing solve, never too
/// little.
pub fn narration_is_wanted(options: &OptionsList) -> bool {
    options
        .get_integer_value("print_level", "")
        .map(|(level, found)| !found || level >= 1)
        .unwrap_or(true)
}

/// Did a second-opinion re-solve converge well enough to overturn the original
/// local-infeasibility verdict? Only a clean or acceptable-level solve
/// promotes; everything else (including a second infeasibility verdict) leaves
/// the original verdict standing.
pub fn scaling_retry_promoted(retry_status: ApplicationReturnStatus) -> bool {
    matches!(
        retry_status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    )
}

/// Resolve the final `(status, statistics)` after an MC64 hypersensitivity
/// re-solve (code review L23).
///
/// On promotion the retry is the authoritative solve, so its status **and** its
/// statistics are reported together. Otherwise the original local-infeasibility
/// verdict is kept — and so are the *original* solve's statistics, so the
/// summary / JSON report never pair the original verdict with the failed
/// retry's iteration count or objective. The pre-fix code reverted `status` to
/// `InfeasibleProblemDetected` but read `app.statistics()` *after* the retry,
/// leaking the retry solve's stats into a report labeled with the original
/// verdict.
pub fn resolve_scaling_retry_outcome(
    original_status: ApplicationReturnStatus,
    retry_status: ApplicationReturnStatus,
    original_stats: SolveStatistics,
    retry_stats: SolveStatistics,
) -> (ApplicationReturnStatus, SolveStatistics) {
    if scaling_retry_promoted(retry_status) {
        (retry_status, retry_stats)
    } else {
        (original_status, original_stats)
    }
}

impl SecondOpinionTrigger {
    /// Which ladder, if any, a finished solve's verdict opens.
    ///
    /// Only these three statuses open one. In particular an iteration- or
    /// time-limit exit does not: the answer there is a bigger budget, and a
    /// re-solve from a different trajectory would burn the same budget again
    /// to reach the same wall.
    ///
    /// `Restoration_Failed` is on the list for the same reason the other two
    /// are (gh#815): it is a report about the *path*, not about the model.
    /// The restoration phase failing to find a filter-acceptable point says
    /// the iterate reached somewhere the sub-problem could not work from, and
    /// a different starting point is a different sub-problem. It is not a
    /// budget exit — pounce stops far short of `max_iter` — so "give it more
    /// iterations" is not the available answer, which is precisely the
    /// distinction the paragraph above draws. Measured on the gh#815 square
    /// flowsheet family: both failing members exit `Restoration_Failed`, no
    /// ladder ran, and rung 3 alone recovers both to `Optimal Solution
    /// Found` — one of them (`f100`) to an optimum Ipopt itself misses.
    ///
    /// Only rung 3 opens on this trigger; rungs 1 and 2 stay gated on
    /// [`SecondOpinionTrigger::LocalInfeasibility`], so a restoration failure
    /// costs exactly one extra solve. That is the measured ordering, not
    /// caution for its own sake: over the KRONOS corpus the displaced start
    /// recovered 13 of 15 where `mu_strategy=adaptive` recovered 4 (see
    /// `start_point_retry`'s option text).
    pub fn for_status(status: ApplicationReturnStatus) -> Option<Self> {
        match status {
            ApplicationReturnStatus::InfeasibleProblemDetected => {
                Some(SecondOpinionTrigger::LocalInfeasibility)
            }
            ApplicationReturnStatus::InvalidNumberDetected => {
                Some(SecondOpinionTrigger::InvalidNumber)
            }
            ApplicationReturnStatus::RestorationFailed => {
                Some(SecondOpinionTrigger::RestorationFailure)
            }
            // The exception to the paragraph above, and it is an exception
            // to the *reason* rather than a change of mind about it
            // (gh#857). "A re-solve would burn the same budget to reach the
            // same wall" is true of a budget exit whose trajectory the
            // ladder cannot change — and a `feral_increase_quality`
            // escalation is precisely a trajectory the ladder *can* change,
            // because it persists across every later factorization and one
            // option removes it. Measured: `square_flowsheet_resto`'s lbfgs
            // leg hits the 3000-iteration cap having escalated 25 times, and
            // converges in 178 with the rung off.
            //
            // This returns `Some` for every budget exit; the escalation
            // count is not visible here. `second_opinion_rungs` drops the
            // rung when the count is zero and the driver returns on the
            // empty list *before* narrating, so a non-escalating budget exit
            // is unchanged — same statuses, same statistics, same console.
            // `a_budget_exit_that_never_escalated_opens_no_rung` is the pin.
            ApplicationReturnStatus::MaximumIterationsExceeded => {
                Some(SecondOpinionTrigger::IterationLimit)
            }
            ApplicationReturnStatus::ErrorInStepComputation => {
                Some(SecondOpinionTrigger::StepComputationError)
            }
            _ => None,
        }
    }

    /// The word for this trigger in a console line.
    pub fn describe(self) -> &'static str {
        match self {
            SecondOpinionTrigger::LocalInfeasibility => "local infeasibility",
            SecondOpinionTrigger::InvalidNumber => "invalid number",
            SecondOpinionTrigger::RestorationFailure => "restoration failure",
            SecondOpinionTrigger::StepComputationError => "step computation breakdown",
            SecondOpinionTrigger::IterationLimit => {
                "iteration limit after a factorization escalation"
            }
        }
    }
}

impl SecondOpinionAvailability {
    /// Read everything the ladder needs to know about the baseline solve out
    /// of the options it ran under.
    ///
    /// The scaling and barrier tags are read from the **resolved** strategy,
    /// not from the option strings. `feral_scaling` is applied only when set
    /// explicitly and otherwise `FeralConfig::from_env()` governs via
    /// `POUNCE_FERAL_SCALING`, so the option string reads `auto` for an
    /// env-configured run; writing that back would silently override the
    /// environment on the retry instead of restoring it. `External` is
    /// unreachable from the string option, and if it ever arrives here there
    /// is no tag to write, so the rungs that need one are dropped rather than
    /// guessed at.
    /// `baseline_quality_escalations` is the failing solve's
    /// `SolveStatistics::quality_escalations`, and it is a **required
    /// parameter** rather than a defaulted setter for the reason gh#857
    /// exists at all: the quantity is invisible everywhere else, so a caller
    /// that forgot it would silently pass `0`, rung 4 would never open, and
    /// the recovery would look like a rung that simply does not fire. A
    /// missing argument is a compile error; a forgotten setter is a
    /// regression nobody can see.
    pub fn from_options(
        options: &OptionsList,
        trigger: SecondOpinionTrigger,
        baseline_quality_escalations: u64,
    ) -> Self {
        // Each of the three `*_retry` flags is read with its tag written out
        // as a literal rather than through a shared closure: `init_options_wiring`
        // proves every registered Initialization option is actually consumed by
        // scanning the source for `get_*_value("<tag>"`, and a closure taking
        // the tag as a parameter is invisible to that scan — which would leave
        // `infeasibility_perturbed_start_retry` looking like a knob that
        // validates, accepts a value and does nothing.
        let scaling = crate::application::feral_config_from_options(options).scaling;
        let baseline_scaling = match scaling {
            pounce_feral::ScalingStrategy::Auto => Some("auto"),
            pounce_feral::ScalingStrategy::InfNorm => Some("infnorm"),
            pounce_feral::ScalingStrategy::Mc64Symmetric => Some("mc64"),
            pounce_feral::ScalingStrategy::Identity => Some("identity"),
            pounce_feral::ScalingStrategy::External(_) => None,
        };
        let already_adaptive = options
            .get_string_value("mu_strategy", "")
            .map(|(v, _found)| v == "adaptive")
            .unwrap_or(false);
        Self {
            trigger,
            scaling_retry_enabled: options
                .get_bool_value("feral_infeasibility_scaling_retry", "")
                .map(|(v, _found)| v)
                .unwrap_or(true),
            mu_retry_enabled: options
                .get_bool_value("infeasibility_mu_strategy_retry", "")
                .map(|(v, _found)| v)
                .unwrap_or(true),
            perturbed_start_retry_enabled: options
                .get_bool_value("infeasibility_perturbed_start_retry", "")
                .map(|(v, _found)| v)
                .unwrap_or(true),
            already_mc64: matches!(scaling, pounce_feral::ScalingStrategy::Mc64Symmetric),
            already_adaptive,
            already_perturbed: options
                .get_numeric_value("start_point_perturbation", "")
                .map(|(v, _found)| v > 0.0)
                .unwrap_or(false),
            increase_quality_retry_enabled: options
                .get_bool_value("feral_increase_quality_retry", "")
                .map(|(v, _found)| v)
                .unwrap_or(true),
            // The rung would set `feral_increase_quality` to the value the
            // failing solve already ran under, so it is a no-op re-solve.
            // Read as a *value*, not as set-ness: the default is `yes`, so
            // an unset option is not already-no.
            already_no_increase_quality: options
                .get_bool_value("feral_increase_quality", "")
                .map(|(v, _found)| !v)
                .unwrap_or(false),
            baseline_quality_escalations,
            baseline_scaling,
            // `mu_strategy` has exactly two registered values, so "not
            // adaptive" is "monotone" and there is no third case to guess at.
        }
    }
}

#[cfg(test)]
mod scaling_retry_tests {
    use super::{
        SecondOpinionAvailability, SecondOpinionTrigger, narration_is_wanted,
        resolve_scaling_retry_outcome, scaling_retry_promoted, second_opinion_rungs,
    };
    use pounce_common::options_list::OptionsList;
    use pounce_nlp::SolveStatistics;
    use pounce_nlp::return_codes::ApplicationReturnStatus;

    fn avail() -> SecondOpinionAvailability {
        SecondOpinionAvailability {
            trigger: SecondOpinionTrigger::LocalInfeasibility,
            scaling_retry_enabled: true,
            mu_retry_enabled: true,
            perturbed_start_retry_enabled: true,
            already_mc64: false,
            already_adaptive: false,
            already_perturbed: false,
            increase_quality_retry_enabled: true,
            already_no_increase_quality: false,
            // Zero, so that every test written before gh#857 keeps asserting
            // exactly the ladder it asserted then: rung 4's gate is a count,
            // and a solve that did not escalate cannot reach it. The tests
            // that are about rung 4 raise it explicitly, which is also how
            // they document that the count is the thing doing the work.
            baseline_quality_escalations: 0,
            baseline_scaling: Some("auto"),
        }
    }

    /// The default ladder is three rungs, in increasing order of how much
    /// they change: linear algebra, then barrier trajectory, then the point
    /// the trajectory starts from.
    #[test]
    fn default_ladder_is_scaling_then_barrier_strategy_then_start() {
        let rungs = second_opinion_rungs(avail());
        let labels: Vec<_> = rungs.iter().map(|r| r.label).collect();
        assert_eq!(
            labels,
            [
                "feral_scaling=mc64",
                "mu_strategy=adaptive",
                "start_point_perturbation=1e-2"
            ]
        );
    }

    /// gh #524: the rungs are applied to the *baseline*, not stacked, because
    /// on `cresc4` `mu_strategy=adaptive` recovers the optimum while
    /// `mu_strategy=adaptive` together with `feral_scaling=mc64` still reports
    /// local infeasibility — a cumulative ladder would throw the fix away.
    ///
    /// The barrier rung therefore carries `mu_strategy` and nothing else;
    /// rung 1's scaling is undone by the driver re-applying its snapshot, and
    /// `second_opinion_driver::tests::each_rung_starts_from_the_baseline`
    /// is where the undo itself is pinned.
    #[test]
    fn barrier_rung_varies_only_the_barrier_strategy() {
        for baseline in ["auto", "infnorm"] {
            let rungs = second_opinion_rungs(SecondOpinionAvailability {
                baseline_scaling: Some(baseline),
                ..avail()
            });
            let barrier = rungs
                .iter()
                .find(|r| r.label == "mu_strategy=adaptive")
                .expect("barrier rung present");
            let assigned: Vec<_> = barrier.assignments.iter().map(|a| a.trim()).collect();
            assert_eq!(assigned, ["mu_strategy adaptive"]);
        }
    }

    /// A rung that cannot change anything is dropped rather than burning a
    /// whole solve to re-derive the same answer.
    #[test]
    fn rungs_already_satisfied_at_baseline_are_dropped() {
        let only_barrier = second_opinion_rungs(SecondOpinionAvailability {
            already_mc64: true,
            ..avail()
        });
        assert_eq!(
            only_barrier.iter().map(|r| r.label).collect::<Vec<_>>(),
            ["mu_strategy=adaptive", "start_point_perturbation=1e-2"],
        );

        let only_scaling = second_opinion_rungs(SecondOpinionAvailability {
            already_adaptive: true,
            ..avail()
        });
        assert_eq!(
            only_scaling.iter().map(|r| r.label).collect::<Vec<_>>(),
            ["feral_scaling=mc64", "start_point_perturbation=1e-2"],
        );

        assert!(
            second_opinion_rungs(SecondOpinionAvailability {
                already_mc64: true,
                already_adaptive: true,
                already_perturbed: true,
                ..avail()
            })
            .is_empty(),
            "nothing left to vary means no ladder at all",
        );
    }

    /// A resolved scaling with no `feral_scaling` tag
    /// (`ScalingStrategy::External`) drops the barrier rung; the scaling rung
    /// is unaffected. The gate is now conservatism rather than necessity —
    /// the driver restores by set-ness, so a missing tag no longer strands
    /// the rung under rung 1's scaling — and it is kept because lifting it
    /// would add rungs on externally-scaled models, which is a trajectory
    /// change. See the note on `baseline_scaling`.
    #[test]
    fn barrier_rung_is_dropped_when_the_baseline_scaling_has_no_tag() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            baseline_scaling: None,
            ..avail()
        });
        assert_eq!(
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
            ["feral_scaling=mc64"],
        );
    }

    /// Each rung has its own opt-out, and turning both off restores upstream
    /// IPOPT's behaviour of shipping the first verdict.
    #[test]
    fn each_rung_can_be_disabled_independently() {
        assert_eq!(
            second_opinion_rungs(SecondOpinionAvailability {
                scaling_retry_enabled: false,
                ..avail()
            })
            .iter()
            .map(|r| r.label)
            .collect::<Vec<_>>(),
            ["mu_strategy=adaptive", "start_point_perturbation=1e-2"],
        );
        assert_eq!(
            second_opinion_rungs(SecondOpinionAvailability {
                mu_retry_enabled: false,
                ..avail()
            })
            .iter()
            .map(|r| r.label)
            .collect::<Vec<_>>(),
            ["feral_scaling=mc64", "start_point_perturbation=1e-2"],
        );
        assert_eq!(
            second_opinion_rungs(SecondOpinionAvailability {
                perturbed_start_retry_enabled: false,
                ..avail()
            })
            .iter()
            .map(|r| r.label)
            .collect::<Vec<_>>(),
            ["feral_scaling=mc64", "mu_strategy=adaptive"],
        );
        assert!(
            second_opinion_rungs(SecondOpinionAvailability {
                scaling_retry_enabled: false,
                mu_retry_enabled: false,
                perturbed_start_retry_enabled: false,
                ..avail()
            })
            .is_empty(),
        );
    }

    /// gh #524's lesson applied to the third rung: it varies exactly one thing
    /// from the *baseline*. It undoes the earlier rungs by having the driver
    /// re-apply the snapshot, so its own assignment list is the displacement
    /// and nothing else.
    #[test]
    fn start_rung_assigns_only_the_displacement() {
        for baseline_scaling in ["auto", "infnorm"] {
            let rungs = second_opinion_rungs(SecondOpinionAvailability {
                baseline_scaling: Some(baseline_scaling),
                ..avail()
            });
            let start = rungs
                .iter()
                .find(|r| r.label == "start_point_perturbation=1e-2")
                .expect("start rung present");
            let assigned: Vec<_> = start.assignments.iter().map(|a| a.trim()).collect();
            assert_eq!(assigned, ["start_point_perturbation 1e-2"]);
        }
    }

    /// The regression this file exists to prevent a second time.
    ///
    /// Rungs 2 and 3 used to undo their predecessors by writing the
    /// baseline's *resolved* value back. For a knob the caller never set that
    /// is a no-op by value and a change by set-ness, and
    /// `is_mu_strategy_fallback_enabled` reads set-ness: it is default-on
    /// only while `mu_strategy` is unset. So rung 3 re-asserting a resolved
    /// `monotone` turned pounce's own μ-strategy stall retry off for the
    /// length of the rung. On KRONOS `a18_ackley1` that is the difference
    /// between `Solve_Succeeded` in 237 iterations and
    /// `Maximum_Iterations_Exceeded` at 3000.
    ///
    /// No rung may name a knob it is not there to vary — restoring is the
    /// driver's job, because only the snapshot knows set-ness.
    #[test]
    fn no_rung_writes_back_a_knob_it_does_not_vary() {
        for avail in [
            avail(),
            SecondOpinionAvailability {
                already_adaptive: true,
                ..avail()
            },
            SecondOpinionAvailability {
                trigger: SecondOpinionTrigger::InvalidNumber,
                ..avail()
            },
        ] {
            for rung in second_opinion_rungs(avail) {
                let varies = rung.label.split('=').next().expect("label has a tag");
                for a in &rung.assignments {
                    let tag = a.trim().split_whitespace().next().expect("tag");
                    assert_eq!(
                        tag,
                        varies,
                        "rung `{}` writes `{}`, which it does not vary",
                        rung.label,
                        a.trim(),
                    );
                }
            }
        }
    }

    /// Rung 3 sits behind the same `baseline_scaling` gate as rung 2 and is
    /// dropped with it, for the same reason: kept as conservatism about a
    /// trajectory change on externally-scaled models, not because the rung
    /// needs a tag to put back.
    #[test]
    fn start_rung_is_dropped_when_the_baseline_scaling_has_no_tag() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            baseline_scaling: None,
            ..avail()
        });
        assert!(
            !rungs
                .iter()
                .any(|r| r.label == "start_point_perturbation=1e-2"),
            "{:?}",
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
        );
    }

    /// The console gate, which both printing surfaces share. An explicit `0`
    /// is silence; anything above it, or no setting at all, stays loud — a
    /// caller who did not ask for quiet is not asking for less.
    #[test]
    fn narration_follows_print_level() {
        let mut opts = OptionsList::new();
        // Unset: narrate, the pre-gate behaviour.
        assert!(narration_is_wanted(&opts));
        // …and an unset level narrates whatever the registered default reads
        // as, which is what distinguishes "nobody asked" from "asked for 0".
        for (level, want) in [(0, false), (1, true), (5, true), (12, true)] {
            opts.set_integer_value("print_level", level, true, true)
                .unwrap();
            assert_eq!(narration_is_wanted(&opts), want, "print_level={level}");
        }
    }

    /// An `Invalid_Number_Detected` reaches only the rung that moves the
    /// point. Re-running the same callbacks at the same point under a
    /// different linear-solver scaling or a different barrier strategy
    /// evaluates the same non-finite quantity again, so those two rungs are
    /// not evidence about this failure and would only burn solves.
    #[test]
    fn an_invalid_number_reaches_only_the_start_rung() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            trigger: SecondOpinionTrigger::InvalidNumber,
            ..avail()
        });
        assert_eq!(
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
            ["start_point_perturbation=1e-2"],
        );
    }

    /// gh#815. A restoration failure opens the ladder, and opens exactly the
    /// one rung that is evidence about it. Rungs 1 and 2 vary the *path* from
    /// the same starting point; the restoration sub-problem failed because of
    /// where the iterate got to, and a different path can arrive somewhere
    /// just as bad. Rung 3 moves the point, which makes it a different
    /// sub-problem — and it is the rung the KRONOS measurement ranks first
    /// (13 of 15 against `mu_strategy=adaptive`'s 4).
    #[test]
    fn a_restoration_failure_reaches_only_the_start_rung() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            trigger: SecondOpinionTrigger::RestorationFailure,
            ..avail()
        });
        assert_eq!(
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
            ["start_point_perturbation=1e-2"],
        );
    }

    /// gh#857 rung 4 on a restoration failure, and the ordering claim that
    /// makes it cheap: it is **appended**, so the gh#815 rung still runs
    /// first and still promotes first. `square_flowsheet_resto`'s exact leg
    /// is exactly that case, and is why this rung costs it nothing.
    #[test]
    fn an_escalating_restoration_failure_appends_the_quality_rung() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            trigger: SecondOpinionTrigger::RestorationFailure,
            baseline_quality_escalations: 2,
            ..avail()
        });
        assert_eq!(
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
            ["start_point_perturbation=1e-2", "feral_increase_quality=no",],
        );
    }

    /// The whole ladder on a budget exit is rung 4, and it is there only
    /// because the solve escalated.
    #[test]
    fn an_escalating_budget_exit_reaches_only_the_quality_rung() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            trigger: SecondOpinionTrigger::IterationLimit,
            baseline_quality_escalations: 25,
            ..avail()
        });
        assert_eq!(
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
            ["feral_increase_quality=no"],
        );
        assert_eq!(rungs[0].assignments, ["feral_increase_quality no\n"]);
    }

    /// The third shape an escalation-rerouted trajectory takes, and the one
    /// that is platform-dependent.
    ///
    /// `square_flowsheet_resto`'s lbfgs leg runs 3000 iterations and escalates
    /// 25 times on both macOS/arm64 and linux/x86_64, and then exits
    /// `Maximum_Iterations_Exceeded` on the first and
    /// `Infeasible_Problem_Detected` on the second — a wrong answer on a
    /// feasible model that the un-escalated solve reaches in 178 iterations.
    /// Rung 4 opened on two of the three shapes until that divergence turned
    /// up in CI, which meant the gh#857 fix recovered the model on one
    /// platform and not the other.
    ///
    /// Appended here too, so the three infeasibility rungs still run and still
    /// promote first: this is what is left when they do not.
    #[test]
    fn an_escalating_infeasibility_verdict_appends_the_quality_rung() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            trigger: SecondOpinionTrigger::LocalInfeasibility,
            baseline_quality_escalations: 25,
            ..avail()
        });
        assert_eq!(
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
            [
                "feral_scaling=mc64",
                "mu_strategy=adaptive",
                "start_point_perturbation=1e-2",
                "feral_increase_quality=no",
            ],
        );
    }

    /// And the gate holds on that trigger too: a local-infeasibility verdict
    /// from a solve that never escalated opens the same three rungs it opened
    /// before gh#857, and pays for no fourth.
    #[test]
    fn an_infeasibility_verdict_that_never_escalated_gets_no_quality_rung() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            trigger: SecondOpinionTrigger::LocalInfeasibility,
            baseline_quality_escalations: 0,
            ..avail()
        });
        assert_eq!(
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
            [
                "feral_scaling=mc64",
                "mu_strategy=adaptive",
                "start_point_perturbation=1e-2",
            ],
        );
    }

    /// The other branch of rung 4's gate, and the one that keeps `for_status`
    /// naming a trigger for every budget exit from costing anything. Without
    /// this the change would put an extra solve on every
    /// `Maximum_Iterations_Exceeded` in the corpus.
    ///
    /// An empty ladder is not merely "no rung ran": the driver returns
    /// `unchanged` on an empty list *before* it narrates, so such a solve is
    /// byte-identical to its pre-gh#857 self.
    #[test]
    fn a_budget_exit_that_never_escalated_opens_no_rung() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            trigger: SecondOpinionTrigger::IterationLimit,
            baseline_quality_escalations: 0,
            ..avail()
        });
        assert!(
            rungs.is_empty(),
            "a budget exit with no escalation has nothing for the ladder to \
             test, and must not pay for a solve: {:?}",
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
        );
    }

    /// The displacement rung does **not** open on a budget exit, even an
    /// escalating one. It is evidence about a failed *path*, not about a
    /// budget, and adding it here would double the cost of the recovery
    /// while testing a hypothesis nobody has measured.
    #[test]
    fn a_budget_exit_does_not_reach_the_displacement_rung() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            trigger: SecondOpinionTrigger::IterationLimit,
            baseline_quality_escalations: 7,
            ..avail()
        });
        assert!(
            !rungs
                .iter()
                .any(|r| r.label == "start_point_perturbation=1e-2"),
            "{:?}",
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
        );
    }

    /// Rung 4 turns off with its own option, like every other rung, and is
    /// dropped when the baseline already ran with the escalation disabled —
    /// where it would re-run the solve that just failed.
    #[test]
    fn the_quality_rung_is_droppable_and_never_a_no_op() {
        let escalating = SecondOpinionAvailability {
            trigger: SecondOpinionTrigger::IterationLimit,
            baseline_quality_escalations: 3,
            ..avail()
        };
        assert!(
            second_opinion_rungs(SecondOpinionAvailability {
                increase_quality_retry_enabled: false,
                ..escalating
            })
            .is_empty()
        );
        assert!(
            second_opinion_rungs(SecondOpinionAvailability {
                already_no_increase_quality: true,
                ..escalating
            })
            .is_empty()
        );
    }

    /// The status → trigger map is the whole opt-in surface, so pin both
    /// halves: the verdicts that open a ladder, and a representative exit
    /// that must not. `MaximumIterationsExceeded` is the case the
    /// doc comment argues about — a bigger budget is the answer there, and a
    /// re-solve would burn the same budget to reach the same wall.
    #[test]
    fn only_path_verdicts_open_a_ladder() {
        use ApplicationReturnStatus as A;
        for (status, want) in [
            (
                A::InfeasibleProblemDetected,
                Some(SecondOpinionTrigger::LocalInfeasibility),
            ),
            (
                A::InvalidNumberDetected,
                Some(SecondOpinionTrigger::InvalidNumber),
            ),
            (
                A::RestorationFailed,
                Some(SecondOpinionTrigger::RestorationFailure),
            ),
            // gh#857: a budget exit now names a trigger, but naming one is
            // not opening a ladder — the rung it names is gated on the
            // escalation count, which `for_status` cannot see. The pair of
            // tests below is what says the distinction holds.
            (
                A::MaximumIterationsExceeded,
                Some(SecondOpinionTrigger::IterationLimit),
            ),
            // The wasm32 case from gh#960. This row read `None` until deb7 under the wasm32
            // build produced a step-computation breakdown that
            // `mu_strategy=adaptive` recovers outright — the rescue was in
            // the ladder already and only the mapping was missing. Unlike
            // a budget exit, a breakdown is not a request for more of
            // anything: the trajectory reached a KKT system no
            // regularization could make usable, and only a different
            // trajectory answers it.
            (
                A::ErrorInStepComputation,
                Some(SecondOpinionTrigger::StepComputationError),
            ),
            (A::MaximumCpuTimeExceeded, None),
            (A::SolveSucceeded, None),
            (A::SolvedToAcceptableLevel, None),
        ] {
            assert_eq!(
                SecondOpinionTrigger::for_status(status),
                want,
                "{status:?} opened the wrong ladder"
            );
        }
    }

    /// The wasm32 case from gh#960. A step-computation breakdown opens exactly one rung, and it
    /// is the barrier-schedule one.
    ///
    /// The other three are excluded on measurement, not on principle: on
    /// `deb7` under the wasm32 build — the case that found this — `mc64`
    /// still fails at 154 iterations, `feral_increase_quality=no` at 201,
    /// and `start_point_perturbation` burns 324 to fail anyway, while this
    /// rung solves in 131. A perturbed-start rung here would have doubled
    /// the cost of a failure it cannot fix.
    #[test]
    fn a_step_computation_breakdown_opens_only_the_barrier_rung() {
        let base = SecondOpinionAvailability {
            trigger: SecondOpinionTrigger::StepComputationError,
            // Rung 4's own gate is open, so only the trigger keeps it out;
            // otherwise this would pass for the wrong reason.
            baseline_quality_escalations: 2,
            ..avail()
        };
        assert_eq!(
            second_opinion_rungs(base)
                .iter()
                .map(|r| r.label)
                .collect::<Vec<_>>(),
            vec!["mu_strategy=adaptive"]
        );

        // The option still governs it, and a solve already on the adaptive
        // schedule has nothing left to try.
        for off in [
            SecondOpinionAvailability {
                mu_retry_enabled: false,
                ..base
            },
            SecondOpinionAvailability {
                already_adaptive: true,
                ..base
            },
        ] {
            let rungs = second_opinion_rungs(off);
            assert!(
                rungs.is_empty(),
                "{:?}",
                rungs.iter().map(|r| r.label).collect::<Vec<_>>()
            );
        }
    }

    /// …and disabling that rung leaves an invalid-number run with no ladder at
    /// all, rather than falling back to the two rungs that cannot help.
    #[test]
    fn an_invalid_number_with_the_start_rung_off_has_no_ladder() {
        assert!(
            second_opinion_rungs(SecondOpinionAvailability {
                trigger: SecondOpinionTrigger::InvalidNumber,
                perturbed_start_retry_enabled: false,
                ..avail()
            })
            .is_empty(),
        );
    }

    /// A baseline that already displaces the start has nothing left for rung 3
    /// to add: re-running with the same displacement reproduces the failing
    /// solve.
    #[test]
    fn a_baseline_that_already_perturbs_drops_the_start_rung() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            already_perturbed: true,
            ..avail()
        });
        assert_eq!(
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
            ["feral_scaling=mc64", "mu_strategy=adaptive"],
        );
    }

    /// The verdict a failed ladder keeps is the one the solve actually
    /// shipped. Before the ladder took `Invalid_Number_Detected` as a trigger
    /// this function hard-coded `Infeasible_Problem_Detected`, which for the
    /// new trigger would have reported the wrong failure.
    #[test]
    fn a_failed_ladder_keeps_whichever_verdict_opened_it() {
        for original in [
            ApplicationReturnStatus::InfeasibleProblemDetected,
            ApplicationReturnStatus::InvalidNumberDetected,
        ] {
            let (status, stats) = resolve_scaling_retry_outcome(
                original,
                ApplicationReturnStatus::MaximumIterationsExceeded,
                stats_with_iters(7),
                stats_with_iters(42),
            );
            assert_eq!(status, original);
            assert_eq!(stats.iteration_count, 7);
        }
    }

    fn stats_with_iters(n: i32) -> SolveStatistics {
        SolveStatistics {
            iteration_count: n,
            final_objective: n as f64,
            ..SolveStatistics::default()
        }
    }

    /// Code review L23: when the MC64 hypersensitivity re-solve does **not**
    /// recover, the verdict reverts to the original local-infeasibility status
    /// — and the reported statistics must revert with it, not leak the failed
    /// retry's iteration count / objective.
    #[test]
    fn failed_retry_keeps_original_status_and_stats() {
        let original = stats_with_iters(7);
        let retry = stats_with_iters(42);
        for retry_status in [
            ApplicationReturnStatus::InfeasibleProblemDetected,
            ApplicationReturnStatus::MaximumIterationsExceeded,
            ApplicationReturnStatus::RestorationFailed,
        ] {
            assert!(!scaling_retry_promoted(retry_status));
            let (status, stats) = resolve_scaling_retry_outcome(
                ApplicationReturnStatus::InfeasibleProblemDetected,
                retry_status,
                original.clone(),
                retry.clone(),
            );
            assert_eq!(
                status,
                ApplicationReturnStatus::InfeasibleProblemDetected,
                "a non-promoting retry ({retry_status:?}) keeps the original verdict"
            );
            assert_eq!(
                stats.iteration_count, 7,
                "stats must stay the original solve's, not the failed retry's"
            );
            assert_eq!(stats.final_objective, 7.0);
        }
    }

    /// On promotion the retry is authoritative: its status AND its statistics
    /// are reported together.
    #[test]
    fn promoted_retry_adopts_retry_status_and_stats() {
        let original = stats_with_iters(7);
        let retry = stats_with_iters(42);
        for retry_status in [
            ApplicationReturnStatus::SolveSucceeded,
            ApplicationReturnStatus::SolvedToAcceptableLevel,
        ] {
            assert!(scaling_retry_promoted(retry_status));
            let (status, stats) = resolve_scaling_retry_outcome(
                ApplicationReturnStatus::InfeasibleProblemDetected,
                retry_status,
                original.clone(),
                retry.clone(),
            );
            assert_eq!(status, retry_status, "a promoting retry adopts its verdict");
            assert_eq!(
                stats.iteration_count, 42,
                "promoted: stats must be the retry solve's"
            );
            assert_eq!(stats.final_objective, 42.0);
        }
    }
}
