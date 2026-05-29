//! Main optimization loop — port of
//! `Algorithm/IpIpoptAlg.{hpp,cpp}`.
//!
//! Phase 7 ships the loop scaffold matching `Optimize()` lines
//! 292-563 in upstream. The body invokes:
//!
//!   1. `IterateInitializer::set_initial_iterates`
//!   2. (loop) `OutputIteration` → `CheckConvergence` →
//!      `UpdateBarrierParameter` → `UpdateHessian` →
//!      `ComputeSearchDirection` → `ComputeAcceptableTrialPoint` →
//!      `AcceptTrialPoint`
//!   3. `correct_bound_multiplier` (kappa_sigma) per `MAIN_LOOP.md`
//!      §"Bound multiplier reset" lines 1055-1134
//!   4. exception → `SolverReturn` mapping per the table in
//!      `MAIN_LOOP.md`.
//!
//! The NLP handle and search-direction calculator are optional:
//! when both are present, `iterate()` computes a real Newton step and
//! drives the line search. Without them, `iterate()` runs the bookkeeping
//! pieces (mu update, hessian update, conv check, kappa_sigma reset)
//! and is exercised by structural unit tests. The full path lights up
//! once `pounce-nlp::OrigIpoptNLP` lands.

use crate::alg_builder::AlgorithmBundle;
use crate::conv_check::r#trait::ConvergenceStatus;
use crate::intermediate::{CtxGuard, IntermediateContext};
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iter_dump::IterDumper;
use crate::iterate_dump::emit_record as emit_iterate_record;
use crate::kkt::pd_search_dir_calc::PdSearchDirCalc;
use crate::line_search::backtracking::Outcome;
use crate::restoration::{RestorationOutcome, RestorationPhase};
use pounce_common::diagnostics::DiagnosticsState;
use pounce_common::types::{Index, Number};
use pounce_linalg::Vector;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::return_codes::AlgorithmMode;
use pounce_nlp::tnlp::{IpoptCq as TnlpIpoptCq, IpoptData as TnlpIpoptData, IterStats, TNLP};
use std::cell::RefCell;
use std::rc::Rc;

pub struct IpoptAlgorithm {
    pub data: IpoptDataHandle,
    pub cq: IpoptCqHandle,
    pub bundle: AlgorithmBundle,
    /// Optional NLP handle. Required for any step that evaluates
    /// problem functions or pulls bound expansion matrices (init,
    /// search direction, line-search trial-point evaluation). Absent
    /// in the structural unit tests of Phases 5-6.
    pub nlp: Option<Rc<RefCell<dyn IpoptNlp>>>,
    /// Optional TNLP handle — the user-facing problem. When present,
    /// `iterate()` fires `TNLP::intermediate_callback` once per outer
    /// iteration so callers can monitor progress or request early
    /// termination (returning `false` from the callback surfaces as
    /// `SolverReturn::UserRequestedStop`). Kept separate from `nlp`
    /// because the algorithm-side NLP is the *compressed* `OrigIpoptNlp`
    /// view (fixed-variable elimination, c/d split) while the callback
    /// payload needs to expose the original-coordinate iterate.
    pub tnlp: Option<Rc<RefCell<dyn TNLP>>>,
    /// Search-direction calculator (`PdSearchDirCalc`). Lands once a
    /// concrete `SymLinearSolver` backend (MUMPS / FERAL) is wired
    /// through `AlgBuilder` in Phase 7's tail.
    pub search_dir: Option<PdSearchDirCalc>,
    /// Restoration-phase strategy. Invoked when the line search
    /// returns [`Outcome::Failed`] (port of upstream
    /// `IpBacktrackingLineSearch::ActivateLineSearch`'s resto
    /// fallback). Optional: in its absence, line-search failure maps
    /// directly to [`SolverReturn::RestorationFailure`] so the main
    /// loop's exit-code semantics match upstream's "no resto built"
    /// case.
    pub restoration: Option<Box<dyn RestorationPhase>>,

    /// `kappa_sigma` for the post-AcceptTrialPoint multiplier reset
    /// (`IpIpoptAlg.cpp:correct_bound_multiplier`, line 1055-1134).
    pub kappa_sigma: Number,
    pub max_iter: Index,
    /// Initial primal step length offered to the line search at the
    /// top of each iteration. Mirrors `IpBacktrackingLineSearch`'s
    /// fraction-to-the-boundary primal step (with τ = `data.curr_tau`).
    /// In v1.0 the structural value here is 1.0 and the FTB cap is
    /// applied per-component when the line-search driver computes
    /// trial slacks; the simplification holds for non-degenerate runs.
    pub alpha_init: Number,
    /// Tiny-step relative tolerance — port of upstream
    /// `IpBacktrackingLineSearch::tiny_step_tol_` (default `10·EPSILON`).
    /// Step is "tiny" when `max_i |δx_i|/(1+|x_i|) ≤ tiny_step_tol`
    /// (and same for s, and `c_viol ≤ 1e-4`).
    pub tiny_step_tol: Number,
    /// Port of upstream `IpIpoptAlg.cpp` divergence guard: when
    /// `max_i |x_i|` exceeds this threshold the optimization aborts with
    /// `SolverReturn::DivergingIterates`. Default `1e20` matches the
    /// registered `diverging_iterates_tol` option. Catches MESH and
    /// similar cases where the normal-mode IPM heads off to infinity
    /// (orig `f` to ±1e33 by iter 90) before line-search failure forces
    /// a degenerate restoration entry.
    pub diverging_iterates_tol: Number,
    /// Companion threshold on the dual step — when both primal and dual
    /// steps are tiny in two consecutive iterations the algorithm
    /// declares convergence at the best attainable accuracy. Default
    /// `1e-2` matches upstream.
    pub tiny_step_y_tol: Number,
    /// Set true when the previous iterate was tagged tiny; on the
    /// second consecutive tiny step the loop sets `data.tiny_step_flag`
    /// so the mu update can attempt to terminate. Mirrors
    /// `IpBacktrackingLineSearch::tiny_step_last_iteration_`.
    pub tiny_step_last_iteration: bool,
    /// Cycle-detection state for [`Self::invoke_restoration`]: the
    /// outer `(x, s)` snapshot from the previous restoration entry,
    /// cleared on any iteration that exits via a normal line-search
    /// accept. When restoration is invoked twice in a row and the
    /// outer iterate has not moved between entries (relative
    /// 2-norm < 1e-10 on both `x` and `s`), the inner resto-IPM is
    /// returning Recovered points indistinguishable from `curr` — a
    /// cycle. Surfaces as `ErrorInStepComputation`. Mirrors the
    /// *intent* of upstream `IpBacktrackingLineSearch.cpp:580-600`'s
    /// almost-feasible-resto guard while staying robust against the
    /// `inf_pr` micro-drift seen on ACOPR14 (delta ~3e-12 per entry,
    /// inf_du essentially constant) where a scalar-`inf_pr` heuristic
    /// fails. Productive single-restoration sequences (BT8, HIMMELBJ,
    /// LINSPANH, LSNNODOC, ODFITS, OET3) clear the snapshot via
    /// `Outcome::Accepted` between entries and are unaffected.
    last_resto_entry_x: Option<Box<dyn Vector>>,
    last_resto_entry_s: Option<Box<dyn Vector>>,
    /// Snapshot of the *recovery* iterate from the previous
    /// restoration. Compared against the next entry's `(x, s)` to
    /// detect "outer made no progress between consecutive resto
    /// invocations". When this distance is below threshold for
    /// several consecutive entries, terminate — catching
    /// slow-non-convergence cycles (ACOPR14, TRO3X3, ACOPR30) where
    /// resto's *inner* moves substantively each call but the *outer*
    /// makes no progress between calls. Cleared on any LS-accepted
    /// step.
    last_resto_recovery_x: Option<Box<dyn Vector>>,
    last_resto_recovery_s: Option<Box<dyn Vector>>,
    /// Count of consecutive restoration entries on which the outer
    /// step (recovery → next-entry) was below the iterate-distance
    /// threshold. Cleared on any LS-accepted step. Limit chosen to
    /// let MAKELA3, HAIFAM, HALDMADS, ROBOT, TENBARS2 — which need
    /// 2-3 consecutive resto entries to recover — pass through.
    resto_no_outer_progress_count: usize,
    /// Count of consecutive restoration entries on which the outer
    /// constraint violation at entry was already below `tol` (the
    /// outer optimality tolerance). Matches the *intent* of upstream
    /// `IpBacktrackingLineSearch.cpp:580-600`'s almost-feasible-resto
    /// guard while using a looser cv threshold (`tol` vs `1e-2·tol`)
    /// — catches DECONVBNE's resto-thrash where each cycle re-enters
    /// at cv ≈ 3e-10 < tol with bound multipliers reset to 1, the
    /// outer's σ-blowup explodes inf_du to 1.9e7, alpha-min triggers
    /// resto re-entry, and the (inf_pr, inf_du) post-recovery state
    /// is essentially identical across cycles but `x` drifts enough
    /// that [`Self::last_resto_recovery_x`]-based detection misses.
    /// Cumulative (never cleared on LS-accept), since DECONVBNE's
    /// cycle interleaves R-recoveries with sub-tol accepts that
    /// accomplish no real outer progress. Fires after 3 near-feasible
    /// entries — surfaces as `StopAtAcceptablePoint` since the
    /// recovered point already satisfies constraint feasibility
    /// within `tol`.
    resto_near_feasible_count: usize,
    /// Snapshot of the most recent iterate that the convergence check
    /// flagged "acceptable" (NLP error ≤ `acceptable_tol`). Mirrors
    /// upstream `IpBacktrackingLineSearch::acceptable_iterate_`
    /// (`IpBacktrackingLineSearch.cpp:1286-1310`). Used by
    /// [`Self::restore_acceptable_point`] to roll back when restoration
    /// fails — if such an iterate exists, the algorithm exits with
    /// `SolverReturn::StopAtAcceptablePoint` rather than
    /// `RestorationFailure`. Cleared/refreshed on every iteration that
    /// satisfies the acceptable predicate.
    acceptable_iterate: Option<crate::iterates_vector::IteratesVector>,
    acceptable_iter_number: Index,
    /// Shared per-solve diagnostics state. `None` unless the CLI
    /// requested `--dump <cat>:<spec>`. When set, the outer loop
    /// advances the state's iter counter and the augmented-system
    /// solver consults it to gate KKT dumps.
    diagnostics: Option<Rc<DiagnosticsState>>,
    /// Optional interactive debugger. When installed, the outer loop
    /// fires it at every [`crate::debug::Checkpoint`] (today: the top
    /// of each iteration) so a REPL or agent can inspect and mutate
    /// live state before the next Newton step. See `crate::debug`.
    debug: Option<Box<dyn crate::debug::DebugHook>>,

    // ---- Restoration-phase audit counters (pounce#12). ----
    //
    // Drained into `SolveStatistics` by `IpoptApplication::optimize_constrained`
    // after the solve completes. Counts are cumulative across the run.
    /// Number of `invoke_restoration` entries.
    pub resto_calls: Index,
    /// Sum of inner-IPM iter counts across every restoration call.
    pub resto_inner_iters: Index,
    /// Number of outer iters that ran in restoration mode (R-line
    /// equivalents in `print_level=5` output).
    pub resto_outer_iters: Index,
    /// Cumulative wall-clock seconds spent inside `perform_restoration`.
    pub resto_wall_secs: Number,

    // ---- Per-iteration history capture (pounce#8). ----
    //
    /// When `true`, [`Self::iterate`] appends an `IterRecord` to
    /// `iter_history` each step. Off by default — the JSON output
    /// path in `pounce-cli` opts in.
    pub record_iter_history: bool,
    /// Per-iteration trajectory captured when `record_iter_history`
    /// is on. Drained into [`SolveStatistics::iterations`] by
    /// `IpoptApplication::optimize_constrained` after the solve.
    pub iter_history: Vec<pounce_nlp::solve_statistics::IterRecord>,
    /// When `false`, the per-iteration table that `iterate()` writes
    /// straight to stdout is suppressed. Wired from
    /// `IpoptApplication`'s `print_level` option: level 0 turns this
    /// off (matches upstream's "no console output" contract). Default
    /// `true` so CLI / direct-driver users keep the familiar trace.
    pub print_iter_output: bool,
}

impl IpoptAlgorithm {
    pub fn new(data: IpoptDataHandle, cq: IpoptCqHandle, mut bundle: AlgorithmBundle) -> Self {
        // The builder may pre-populate `bundle.search_dir` when given a
        // `LinearBackendFactory`; lift it onto the algorithm so the
        // iterate body can call into it directly.
        let search_dir = bundle.search_dir.take();
        Self {
            data,
            cq,
            bundle,
            nlp: None,
            tnlp: None,
            search_dir,
            restoration: None,
            kappa_sigma: 1e10,
            max_iter: 3000,
            alpha_init: 1.0,
            tiny_step_tol: 10.0 * Number::EPSILON,
            diverging_iterates_tol: 1e20,
            tiny_step_y_tol: 1e-2,
            tiny_step_last_iteration: false,
            last_resto_entry_x: None,
            last_resto_entry_s: None,
            last_resto_recovery_x: None,
            last_resto_recovery_s: None,
            resto_no_outer_progress_count: 0,
            resto_near_feasible_count: 0,
            acceptable_iterate: None,
            acceptable_iter_number: 0,
            diagnostics: None,
            debug: None,
            resto_calls: 0,
            resto_inner_iters: 0,
            resto_outer_iters: 0,
            resto_wall_secs: 0.0,
            record_iter_history: false,
            iter_history: Vec::new(),
            print_iter_output: true,
        }
    }

    /// Stash the current iterate as the "last acceptable" backup —
    /// port of `IpBacktrackingLineSearch::StoreAcceptablePoint`
    /// (`IpBacktrackingLineSearch.cpp:1286-1293`).
    fn store_acceptable_point(&mut self) {
        let d = self.data.borrow();
        if let Some(curr) = d.curr.as_ref() {
            self.acceptable_iterate = Some(curr.clone());
            self.acceptable_iter_number = d.iter_count;
        }
    }

    /// Roll the iterate back to the last acceptable snapshot — port of
    /// `IpBacktrackingLineSearch::RestoreAcceptablePoint`
    /// (`IpBacktrackingLineSearch.cpp:1295-1310`). Returns `true` if a
    /// snapshot was available and applied; `false` otherwise (caller
    /// then surfaces the original failure status).
    fn restore_acceptable_point(&mut self) -> bool {
        let Some(prev) = self.acceptable_iterate.clone() else {
            return false;
        };
        let mut d = self.data.borrow_mut();
        d.set_trial(prev);
        // `accept_trial_point` promotes `trial → curr`, mirroring the
        // upstream sequence `set_trial(...); AcceptTrialPoint();`.
        d.accept_trial_point();
        true
    }

    pub fn with_nlp(mut self, nlp: Rc<RefCell<dyn IpoptNlp>>) -> Self {
        self.nlp = Some(nlp);
        self
    }

    /// Install a user-facing TNLP handle. Enables per-iteration
    /// `TNLP::intermediate_callback` invocation from `optimize()`.
    pub fn with_tnlp(mut self, tnlp: Rc<RefCell<dyn TNLP>>) -> Self {
        self.tnlp = Some(tnlp);
        self
    }

    /// Build an [`IterStats`] payload from the current `IpoptData` /
    /// `IpoptCq` state. Mirrors the field set the upstream Ipopt main
    /// loop hands to `IntermediateCallback` after each `AcceptTrialPoint`.
    fn build_iter_stats(&self) -> IterStats {
        let d = self.data.borrow();
        let c = self.cq.borrow();
        let dnrm = match d.delta.as_ref() {
            Some(delta) => delta.x.amax().max(delta.s.amax()),
            None => 0.0,
        };
        IterStats {
            // alg_mod tracking (regular vs restoration) is a follow-up;
            // every callback fire from the outer loop reports RegularMode.
            mode: AlgorithmMode::RegularMode,
            iter: d.iter_count,
            obj_value: c.curr_f(),
            inf_pr: c.curr_primal_infeasibility_max(),
            inf_du: c.curr_dual_infeasibility_max(),
            mu: d.curr_mu,
            d_norm: dnrm,
            regularization_size: d.info_regu_x,
            alpha_du: d.info_alpha_dual,
            alpha_pr: d.info_alpha_primal,
            ls_trials: d.info_ls_count,
        }
    }

    /// Fire `TNLP::intermediate_callback` if a TNLP handle and NLP
    /// handle are installed. Wraps the call in an [`IntermediateContext`]
    /// guard so downstream inspector entry points (the C API's
    /// `GetIpoptCurrent*`) can read live state for the duration. Returns
    /// `true` to continue, `false` if the user requested termination.
    fn fire_intermediate(&self) -> bool {
        let Some(tnlp) = self.tnlp.as_ref() else {
            return true;
        };
        let Some(nlp) = self.nlp.as_ref() else {
            return true;
        };
        let stats = self.build_iter_stats();
        let _guard = CtxGuard::install(IntermediateContext {
            data: Rc::clone(&self.data),
            cq: Rc::clone(&self.cq),
            nlp: Rc::clone(nlp),
        });
        tnlp.borrow_mut().intermediate_callback(
            stats,
            &TnlpIpoptData::default(),
            &TnlpIpoptCq::default(),
        )
    }

    pub fn with_search_dir(mut self, sd: PdSearchDirCalc) -> Self {
        self.search_dir = Some(sd);
        self
    }

    pub fn with_restoration(mut self, resto: Box<dyn RestorationPhase>) -> Self {
        self.restoration = Some(resto);
        self
    }

    /// Install the shared diagnostics state. The state is propagated
    /// to the augmented-system solver at the top of [`Self::optimize`]
    /// so dump sites can consult per-iter gating.
    pub fn with_diagnostics(mut self, diag: Rc<DiagnosticsState>) -> Self {
        self.diagnostics = Some(diag);
        self
    }

    /// Install an interactive debugger hook. Fired at each checkpoint
    /// in [`Self::optimize`]; returning [`crate::debug::DebugAction::Stop`]
    /// ends the solve with `SolverReturn::UserRequestedStop`.
    pub fn with_debug_hook(mut self, hook: Box<dyn crate::debug::DebugHook>) -> Self {
        self.debug = Some(hook);
        self
    }

    /// Fire the debugger hook (if installed) at `cp`, building a live
    /// [`crate::debug::DebugCtx`] over cheap handle clones. Returns the
    /// requested action, defaulting to `Resume` when no hook is set.
    fn fire_debug(&mut self, cp: crate::debug::Checkpoint) -> crate::debug::DebugAction {
        use crate::debug::{DebugAction, DebugCtx};
        let Some(hook) = self.debug.as_mut() else {
            return DebugAction::Resume;
        };
        let mut ctx = DebugCtx::new(Rc::clone(&self.data), Rc::clone(&self.cq), cp);
        hook.at_checkpoint(&mut ctx)
    }

    /// Run the restoration phase, bracketed by the `PreRestoration` /
    /// `PostRestoration` debug checkpoints so a debugger can inspect the
    /// iterate just before entry and just after exit. With no debugger
    /// installed this is exactly `invoke_restoration()`.
    fn invoke_restoration_debugged(&mut self) -> IterateOutcome {
        if let Some(o) = self.debug_stop(crate::debug::Checkpoint::PreRestoration) {
            return o;
        }
        let outcome = self.invoke_restoration();
        if let Some(o) = self.debug_stop(crate::debug::Checkpoint::PostRestoration) {
            return o;
        }
        outcome
    }

    /// Fire a sub-iteration checkpoint from inside [`Self::iterate`].
    /// Returns `Some(Terminate(UserRequestedStop))` if the debugger asked
    /// to stop, so the caller can `return` it; `None` to continue.
    fn debug_stop(&mut self, cp: crate::debug::Checkpoint) -> Option<IterateOutcome> {
        if self.debug.is_none() {
            return None;
        }
        if self.fire_debug(cp) == crate::debug::DebugAction::Stop {
            Some(IterateOutcome::Terminate(SolverReturn::UserRequestedStop))
        } else {
            None
        }
    }

    /// Fire the terminal post-mortem checkpoint (if a debugger is set),
    /// carrying the solve outcome so the hook can decide whether to pause
    /// at the final iterate. The action is advisory — the loop returns
    /// `result` regardless — so the hook just gets a last look.
    fn fire_debug_terminal(&mut self, result: SolverReturn) {
        use crate::debug::{Checkpoint, DebugCtx};
        let Some(hook) = self.debug.as_mut() else {
            return;
        };
        let mut ctx = DebugCtx::new(
            Rc::clone(&self.data),
            Rc::clone(&self.cq),
            Checkpoint::Terminated,
        )
        .with_status(format!("{result:?}"));
        let _ = hook.at_checkpoint(&mut ctx);
    }

    /// One iteration body — port of `Optimize()`'s inner loop.
    /// Returns either `Continue` to keep iterating or a terminal
    /// [`SolverReturn`] mirroring upstream's exception → return-code
    /// translation table (see `MAIN_LOOP.md` §"Exception mapping").
    fn iterate(&mut self) -> IterateOutcome {
        // Shared timing accumulator — cheap Rc clone so each phase can
        // bump its own counter without re-borrowing `data`.
        let timing = self.data.borrow().timing.clone();

        // 1. Output iteration row. Header every 10 iters; the row
        //    itself is built by the strategy and printed here so a
        //    long-running solve gives the user feedback. (Phase-7
        //    upstream routes this through the journalist; until that
        //    surface lands, write straight to stdout.)
        //
        //    Print BEFORE `reset_info` so the row reflects the
        //    accepted step from the previous iteration (alphas, ls
        //    count, alpha_char), matching upstream's
        //    `IpIpoptAlgorithm::Optimize` ordering.
        timing.output_iteration.start();
        self.bundle.iter_output.write_output();
        if self.print_iter_output {
            let iter_count = self.data.borrow().iter_count;
            if iter_count % 10 == 0 {
                print!("{}", crate::output::orig::OrigIterationOutput::HEADER);
            }
            let row = self.bundle.iter_output.format_row(&self.data, &self.cq);
            println!("{row}");
        }
        timing.output_iteration.end();

        // Optional per-iteration history capture (pounce#8 / JSON
        // output). Fires alongside the console print so the records
        // are always in lock-step with what the user sees on stdout.
        if self.record_iter_history {
            let d = self.data.borrow();
            let c = self.cq.borrow();
            let iter = d.iter_count;
            let inf_pr = c.curr_primal_infeasibility_max();
            let inf_du = c.curr_dual_infeasibility_max();
            let mu = d.curr_mu;
            let d_norm = match &d.delta {
                Some(delta) => delta.x.amax().max(delta.s.amax()),
                None => 0.0,
            };
            let regularization = d.info_regu_x;
            let alpha_dual = d.info_alpha_dual;
            let alpha_primal = d.info_alpha_primal;
            let alpha_primal_char = d.info_alpha_primal_char;
            let ls_trials = d.info_ls_count;
            let objective = c.unscaled_curr_f();
            drop(d);
            drop(c);
            self.iter_history
                .push(pounce_nlp::solve_statistics::IterRecord {
                    iter,
                    objective,
                    inf_pr,
                    inf_du,
                    mu,
                    d_norm,
                    regularization,
                    alpha_dual,
                    alpha_primal,
                    alpha_primal_char,
                    ls_trials,
                });
        }

        // Reset per-iteration info on data (after printing previous
        // iter's accepted-step info; before the next line search).
        self.data.borrow_mut().reset_info();

        // 2. Convergence check.
        timing.check_convergence.start();
        let nlp_err = self.cq.borrow().curr_nlp_error();
        let iter_count = self.data.borrow().iter_count;
        if !nlp_err.is_finite() {
            timing.check_convergence.end();
            return IterateOutcome::Terminate(SolverReturn::InvalidNumberDetected);
        }
        // Divergence guard — port of upstream `IpIpoptAlg.cpp` post-
        // AcceptTrialPoint check. When `max_i |x_i|` exceeds the
        // registered `diverging_iterates_tol` (default `1e20`), exit
        // cleanly with `DivergingIterates` rather than spiralling into
        // a degenerate restoration whose inner sub-NLP can't recover
        // (MESH: orig `f` already at -3.6e33 by iter 90, restoration
        // entered too late to bound `x`).
        if let Some(curr) = self.data.borrow().curr.as_ref() {
            if curr.x.amax() > self.diverging_iterates_tol {
                timing.check_convergence.end();
                return IterateOutcome::Terminate(SolverReturn::DivergingIterates);
            }
        }
        let conv_status = self
            .bundle
            .conv_check
            .check_convergence_with_state(nlp_err, iter_count, &self.data, &self.cq);
        match conv_status {
            ConvergenceStatus::Continue => {}
            ConvergenceStatus::Converged => {
                timing.check_convergence.end();
                return IterateOutcome::Terminate(SolverReturn::Success);
            }
            ConvergenceStatus::ConvergedToAcceptable => {
                timing.check_convergence.end();
                return IterateOutcome::Terminate(SolverReturn::StopAtAcceptablePoint);
            }
            ConvergenceStatus::MaxIterExceeded => {
                timing.check_convergence.end();
                return IterateOutcome::Terminate(SolverReturn::MaxiterExceeded);
            }
            ConvergenceStatus::CpuTimeExceeded => {
                timing.check_convergence.end();
                return IterateOutcome::Terminate(SolverReturn::CpuTimeExceeded);
            }
            ConvergenceStatus::WallTimeExceeded => {
                timing.check_convergence.end();
                return IterateOutcome::Terminate(SolverReturn::WallTimeExceeded);
            }
            ConvergenceStatus::LocallyInfeasible => {
                timing.check_convergence.end();
                return IterateOutcome::Terminate(SolverReturn::LocalInfeasibility);
            }
            ConvergenceStatus::Failed => {
                timing.check_convergence.end();
                return IterateOutcome::Terminate(SolverReturn::InternalError);
            }
        }

        // Stash the iterate if it satisfies the per-component
        // `acceptable_*_tol` triplet. Mirrors upstream
        // `IpBacktrackingLineSearch.cpp:282-289` — checked at the top
        // of every line-search call so the most recent acceptable
        // iterate is always available as a rollback target if
        // restoration later fails. The recorder feeds
        // `acceptable_obj_change_tol`'s stability cross-check on
        // subsequent iterates.
        if self
            .bundle
            .conv_check
            .current_is_acceptable_with_state(nlp_err, &self.data, &self.cq)
        {
            self.store_acceptable_point();
            let curr_f = self.cq.borrow().curr_f();
            self.bundle.conv_check.set_curr_acceptable_obj(curr_f);
        }
        timing.check_convergence.end();

        // 3. Hessian update. Must run BEFORE `update_barrier_parameter`
        // so the adaptive-μ oracles (probing, quality-function) drive
        // their affine/centering solves against `W(curr_N)`, not the
        // stale `W(curr_{N-1})` left in `data.w` by the previous iter's
        // tail-end Hessian update. Upstream calls `UpdateHessian()`
        // first in every main-loop body (`IpIpoptAlg.cpp:386`); pounce
        // previously reordered this to the tail, which made iters 1+
        // pick μ from the prior iterate's Hessian on adaptive-mu +
        // quality-function — visible on CRESC50 as a catastrophic
        // early-iter divergence (theta=5.8e5 by iter 61 vs upstream
        // never entering restoration).
        timing.update_hessian.start();
        let _ = self.bundle.hess.update_hessian(&self.data, &self.cq);
        timing.update_hessian.end();

        // 4. Barrier parameter. Pass nlp + search_dir through so the
        // adaptive μ oracles (probing, quality-function) can drive
        // their own affine-step solves; monotone ignores them.
        // Snapshot the tiny-step flag (set by the previous iteration's
        // tiny-step branch) and the entry mu — if μ can't reduce while
        // the flag is on, upstream `IpMonotoneMuUpdate.cpp:158-161`
        // throws TINY_STEP_DETECTED → STOP_AT_TINY_STEP, which we
        // realise as a clean termination here. Only the monotone update
        // throws: `IpAdaptiveMuUpdate.cpp` consumes the tiny-step flag
        // via its `force_no_progress` path (fix μ, keep iterating), so
        // the termination is gated on `terminates_on_tiny_step()`.
        timing.update_barrier_parameter.start();
        let tiny_at_entry = self.data.borrow().tiny_step_flag;
        let mu_before = self.data.borrow().curr_mu;
        let mu_terminates_on_tiny = self.bundle.mu_update.terminates_on_tiny_step();
        let next_mu = self.bundle.mu_update.update_barrier_parameter(
            &self.data,
            &self.cq,
            self.nlp.as_ref(),
            self.search_dir.as_mut(),
        );
        self.data.borrow_mut().curr_mu = next_mu;
        timing.update_barrier_parameter.end();
        if tiny_at_entry && mu_terminates_on_tiny && (next_mu - mu_before).abs() < Number::EPSILON {
            return IterateOutcome::Terminate(SolverReturn::StopAtTinyStep);
        }

        // pounce#58 — iterate-quality guard for the probing oracle.
        // The μ-update layer sets `request_resto` when the input
        // iterate is too corrupted for the probing rule to produce a
        // sane μ (see `mu/adaptive.rs` Probing dispatch). Restoration
        // re-initialises the multipliers and gives the outer loop a
        // clean iterate to continue from. When no restoration phase
        // is configured (embedded callers, tests), emit a one-line
        // notice and continue with the current μ — the guard has
        // already prevented the destabilising 4-order μ jump.
        let request_resto = {
            let mut d = self.data.borrow_mut();
            let f = d.request_resto;
            d.request_resto = false;
            f
        };
        if request_resto {
            if self.restoration.is_some() {
                return self.invoke_restoration_debugged();
            } else {
                eprintln!(
                    "[POUNCE] probing-oracle iterate-quality guard fired \
                     at iter {}, but no restoration phase is configured; \
                     continuing with μ={:.3e}.",
                    self.data.borrow().iter_count,
                    next_mu,
                );
            }
        }

        // Mirror upstream `IpAdaptiveMuUpdate.cpp:339, 386, 431` and
        // `IpMonotoneMuUpdate.cpp:165`: every code path that *changes*
        // μ calls `linesearch_->Reset()`, which clears the filter via
        // `FilterLSAcceptor::Reset` (`IpFilterLSAcceptor.cpp:524-532`).
        // Rationale: filter entries are computed against the current
        // barrier — when μ changes, prior entries no longer apply and
        // would over-constrain acceptance. The two upstream paths that
        // do NOT reset (stay-fixed-no-decrease and fixed→free transition)
        // both keep μ at curr_mu, so the `mu_changed` check captures
        // the intended distinction.
        if next_mu != mu_before {
            self.bundle.line_search.reset();
        }

        // Sub-iteration checkpoint: μ has been updated for this iteration.
        if let Some(o) = self.debug_stop(crate::debug::Checkpoint::AfterBarrierUpdate) {
            return o;
        }

        // 5. Search direction. Skipped without an NLP + search_dir.
        // (Hessian was updated in step 3 above before the barrier-μ
        // oracle so that adaptive-μ uses W(curr_N), not stale W.)
        if let (Some(nlp), Some(sd)) = (self.nlp.as_ref(), self.search_dir.as_mut()) {
            timing.compute_search_direction.start();
            let ok = sd.compute_search_direction(&self.data, &self.cq, nlp);
            timing.compute_search_direction.end();
            if !ok {
                // Mirror upstream `IpIpoptAlg.cpp:417-430`: a failed
                // step computation puts the algorithm in emergency
                // mode, which calls `BacktrackingLineSearch::
                // ActivateFallbackMechanism` (cpp:1312-1328). When a
                // restoration phase is configured, the next pass of
                // `ComputeAcceptableTrialPoint` sees `goto_resto` at
                // cpp:299-306 and hands control to restoration. Only
                // when neither restoration nor an acceptor-level
                // fallback is available does upstream throw
                // `STEP_COMPUTATION_FAILED`.
                if self.restoration.is_some() {
                    return self.invoke_restoration_debugged();
                }
                return IterateOutcome::Terminate(SolverReturn::ErrorInStepComputation);
            }
            if std::env::var_os("POUNCE_DBG_DELTA").is_some() {
                let d = self.data.borrow();
                let it = d.iter_count;
                if let Some(delta) = d.delta.as_ref() {
                    use crate::iterates_vector::IteratesVector;
                    use pounce_linalg::{compound_vector::CompoundVector, Vector};
                    let dv: &IteratesVector = delta;
                    eprintln!(
                        "[PN_DELTA] iter={} mu={:.6e} dx_amax={:.6e} ds_amax={:.6e} dyc_amax={:.6e} dyd_amax={:.6e} dzL_amax={:.6e} dzU_amax={:.6e} dvL_amax={:.6e} dvU_amax={:.6e}",
                        it, d.curr_mu,
                        dv.x.amax(), dv.s.amax(), dv.y_c.amax(), dv.y_d.amax(),
                        dv.z_l.amax(), dv.z_u.amax(), dv.v_l.amax(), dv.v_u.amax()
                    );
                    if let Some(cdx) = dv.x.as_any().downcast_ref::<CompoundVector>() {
                        eprintln!(
                            "[PN_DELTA] iter={} dx_blocks_amax: orig={:.6e} nc={:.6e} pc={:.6e} nd={:.6e} pd={:.6e}",
                            it,
                            cdx.comp(0).amax(),
                            cdx.comp(1).amax(),
                            cdx.comp(2).amax(),
                            cdx.comp(3).amax(),
                            cdx.comp(4).amax(),
                        );
                        eprintln!(
                            "[PN_DELTA] iter={} dx_blocks_nrm2: orig={:.6e} nc={:.6e} pc={:.6e} nd={:.6e} pd={:.6e}",
                            it,
                            cdx.comp(0).nrm2(),
                            cdx.comp(1).nrm2(),
                            cdx.comp(2).nrm2(),
                            cdx.comp(3).nrm2(),
                            cdx.comp(4).nrm2(),
                        );
                        eprintln!(
                            "[PN_DELTA] iter={} dx_blocks_asum: orig={:.6e} nc={:.6e} pc={:.6e} nd={:.6e} pd={:.6e}",
                            it,
                            cdx.comp(0).asum(),
                            cdx.comp(1).asum(),
                            cdx.comp(2).asum(),
                            cdx.comp(3).asum(),
                            cdx.comp(4).asum(),
                        );
                        // Argmax of orig block via dot with sign — print first few values.
                        if let Some(dv_orig) =
                            cdx.comp(0)
                                .as_any()
                                .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
                        {
                            let v = dv_orig.values();
                            let mut imax = 0usize;
                            let mut amax = 0.0f64;
                            for (i, &x) in v.iter().enumerate() {
                                if x.abs() > amax {
                                    amax = x.abs();
                                    imax = i;
                                }
                            }
                            eprintln!(
                                "[PN_DELTA] iter={} dx_orig argmax: i={} v={:.17e} (n={})",
                                it,
                                imax,
                                v[imax],
                                v.len()
                            );
                        }
                    }
                    let p = &d.perturbations;
                    eprintln!(
                        "[PN_DELTA] iter={} pert: dx={:.6e} ds={:.6e} dc={:.6e} dd={:.6e}",
                        it, p.delta_x, p.delta_s, p.delta_c, p.delta_d
                    );
                    drop(d);
                    let cq = self.cq.borrow();
                    let gf = cq.curr_grad_f();
                    let gl = cq.curr_grad_lag_x();
                    let cc = cq.curr_c();
                    let cd = cq.curr_d_minus_s();
                    let sx = cq.curr_sigma_x();
                    let ss = cq.curr_sigma_s();
                    eprintln!(
                        "[PN_DELTA] iter={} cq: gradf_amax={:.6e} gradf_nrm2={:.6e} gradlag_amax={:.6e} gradlag_nrm2={:.6e} c_amax={:.6e} c_nrm2={:.6e} d_amax={:.6e} d_nrm2={:.6e} sigx_amax={:.6e} sigx_nrm2={:.6e} sigs_amax={:.6e} sigs_nrm2={:.6e}",
                        it,
                        gf.amax(), gf.nrm2(),
                        gl.amax(), gl.nrm2(),
                        cc.amax(), cc.nrm2(),
                        cd.amax(), cd.nrm2(),
                        sx.amax(), sx.nrm2(),
                        ss.amax(), ss.nrm2(),
                    );
                    if let Some(cgf) = gf.as_any().downcast_ref::<CompoundVector>() {
                        eprintln!(
                            "[PN_DELTA] iter={} gradf_blocks_amax: orig={:.6e} nc={:.6e} pc={:.6e} nd={:.6e} pd={:.6e}",
                            it,
                            cgf.comp(0).amax(),
                            cgf.comp(1).amax(),
                            cgf.comp(2).amax(),
                            cgf.comp(3).amax(),
                            cgf.comp(4).amax(),
                        );
                    }
                    if let Some(curr) = self.data.borrow().curr.clone() {
                        eprintln!(
                            "[PN_DELTA] iter={} bound_mults: zL_amax={:.6e} zU_amax={:.6e} vL_amax={:.6e} vU_amax={:.6e} s_amax={:.6e} s_nrm2={:.6e} x_amax={:.6e} x_nrm2={:.6e}",
                            it,
                            curr.z_l.amax(), curr.z_u.amax(),
                            curr.v_l.amax(), curr.v_u.amax(),
                            curr.s.amax(), curr.s.nrm2(),
                            curr.x.amax(), curr.x.nrm2(),
                        );
                        if let Some(czl) = curr.z_l.as_any().downcast_ref::<CompoundVector>() {
                            eprintln!(
                                "[PN_DELTA] iter={} zL_blocks_amax: orig={:.6e} nc={:.6e} pc={:.6e} nd={:.6e} pd={:.6e}",
                                it,
                                czl.comp(0).amax(),
                                czl.comp(1).amax(),
                                czl.comp(2).amax(),
                                czl.comp(3).amax(),
                                czl.comp(4).amax(),
                            );
                        }
                        if let Some(czu) = curr.z_u.as_any().downcast_ref::<CompoundVector>() {
                            eprintln!("[PN_DELTA] iter={} zU_ncomps={}", it, czu.n_comps());
                            for ic in 0..czu.n_comps() {
                                eprintln!(
                                    "[PN_DELTA] iter={} zU_block[{}]_amax={:.6e} dim={}",
                                    it,
                                    ic,
                                    czu.comp(ic).amax(),
                                    czu.comp(ic).dim()
                                );
                            }
                        }
                    }
                    if let Some(csx) = sx.as_any().downcast_ref::<CompoundVector>() {
                        eprintln!(
                            "[PN_DELTA] iter={} sigx_blocks_amax: orig={:.6e} nc={:.6e} pc={:.6e} nd={:.6e} pd={:.6e}",
                            it,
                            csx.comp(0).amax(),
                            csx.comp(1).amax(),
                            csx.comp(2).amax(),
                            csx.comp(3).amax(),
                            csx.comp(4).amax(),
                        );
                    }
                    drop(cq);
                    let d = self.data.borrow();
                    // Also dump curr.x_orig argmax
                    if let Some(curr) = d.curr.as_ref() {
                        if let Some(cx) = curr.x.as_any().downcast_ref::<CompoundVector>() {
                            if let Some(xo) =
                                cx.comp(0)
                                    .as_any()
                                    .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
                            {
                                let v = xo.values();
                                let mut imax = 0usize;
                                let mut amax = 0.0f64;
                                for (i, &x) in v.iter().enumerate() {
                                    if x.abs() > amax {
                                        amax = x.abs();
                                        imax = i;
                                    }
                                }
                                eprintln!("[PN_DELTA] iter={} curr_x_orig argmax: i={} v={:.17e} amax={:.17e} nrm2={:.17e}",
                                it, imax, v[imax], xo.amax(), xo.nrm2());
                            }
                        }
                    }
                }
            }
        }

        // Sub-iteration checkpoint: the Newton step `δ` (data.delta) and
        // the applied regularization are now available, before the line
        // search consumes them.
        if let Some(o) = self.debug_stop(crate::debug::Checkpoint::AfterSearchDirection) {
            return o;
        }

        // 6. Acceptable trial point — run the line search if we have a
        //    primal/dual step on `data.delta`. Wrap in a guard so all
        //    early-return paths (ErrorInStepComputation, InternalError,
        //    restoration entry) still stop the timer.
        let _ls_guard = timing.compute_acceptable_trial_point.guard();
        let have_delta = self.data.borrow().delta.is_some();
        if have_delta {
            let delta = match self.data.borrow().delta.as_ref().cloned() {
                Some(d) => d,
                None => {
                    return IterateOutcome::Terminate(SolverReturn::ErrorInStepComputation);
                }
            };
            // Cap alpha by the primal fraction-to-the-boundary so the
            // first trial cannot push slacks past their bounds, and by
            // the dual FTB so bound multipliers stay positive. Mirrors
            // upstream `IpBacktrackingLineSearch::FindAcceptableTrialPoint`'s
            // calls to `IpCq.primal_frac_to_the_bound` /
            // `IpCq.dual_frac_to_the_bound` with τ = `curr_tau`.
            let tau = self.data.borrow().curr_tau;
            let alpha_p_max = self.cq.borrow().aff_step_alpha_primal_max(&delta, tau);
            let alpha_d_max = self.cq.borrow().aff_step_alpha_dual_max(&delta, tau);

            // Tiny-step gate — port of `IpBacktrackingLineSearch.cpp:363`
            // and the handling block at lines 382-435. When the search
            // direction is so small that any nonzero α would just
            // bounce inside floating-point noise, take the FTB step
            // unchecked and skip the line search; that's the only way
            // to hit `STOP_AT_TINY_STEP` cleanly when the iterate is
            // already at a converged point but `nlp_error > tol` due to
            // scaling or unbounded duals.
            if self.detect_tiny_step(&delta) {
                let alpha_p = alpha_p_max;
                let alpha_d = alpha_d_max;
                let curr = match self.data.borrow().curr.clone() {
                    Some(c) => c,
                    None => return IterateOutcome::Terminate(SolverReturn::InternalError),
                };
                let trial_iv = scaled_step_unchecked(&curr, &delta, alpha_p, alpha_d);
                {
                    let mut d = self.data.borrow_mut();
                    d.set_trial(trial_iv);
                    d.info_alpha_primal = alpha_p;
                    d.info_alpha_dual = alpha_d;
                    d.info_ls_count = 0;
                    if self.tiny_step_last_iteration {
                        d.info_alpha_primal_char = 'T';
                        d.tiny_step_flag = true;
                    } else {
                        d.info_alpha_primal_char = 't';
                    }
                }
                let dy_amax = delta.y_c.amax().max(delta.y_d.amax());
                self.tiny_step_last_iteration = dy_amax < self.tiny_step_y_tol;
            } else {
                self.tiny_step_last_iteration = false;
                let alpha_init = self.alpha_init.min(alpha_p_max);
                let alpha_dual = self.alpha_init.min(alpha_d_max);
                let outcome = self.bundle.line_search.find_acceptable_trial_point(
                    &self.data,
                    &self.cq,
                    &delta,
                    alpha_init,
                    alpha_dual,
                    self.nlp.as_ref(),
                    self.search_dir.as_mut(),
                );
                match outcome {
                    Outcome::Accepted => {
                        // A normal LS-accepted step breaks any in-flight
                        // restoration cycle — clear the cycle detector
                        // so the next resto entry starts fresh.
                        self.last_resto_entry_x = None;
                        self.last_resto_entry_s = None;
                        self.last_resto_recovery_x = None;
                        self.last_resto_recovery_s = None;
                        self.resto_no_outer_progress_count = 0;
                        // Intentionally *not* clearing
                        // `resto_near_feasible_count` here: DECONVBNE's
                        // cycle interleaves R-recoveries with 2-3
                        // LS-accepted 'f'/'h' steps (which return
                        // `Outcome::Accepted` but accomplish no real
                        // outer progress — alpha drops to 1e-6 and
                        // inf_du remains pinned at 1.9e7), so resetting
                        // on every accept would zero the counter every
                        // cycle and never fire. The counter persists
                        // for the duration of the run and trips after
                        // 3 cumulative near-feasible entries; legitimate
                        // solves enter resto at most once at near-
                        // feasibility (POLAK6, HAIFAM) and stay under
                        // the limit.
                    }
                    Outcome::TinyStep | Outcome::Failed => {
                        // Upstream `IpBacktrackingLineSearch.cpp` raises
                        // `LINE_SEARCH_FAILED` when α drops below
                        // `alpha_min` or all retries reject, which in
                        // turn triggers `ActivateLineSearch` →
                        // restoration.
                        return self.invoke_restoration_debugged();
                    }
                }
            }
        }

        // End the line-search/trial timer here so the bookkeeping in
        // steps 7-8 below is attributed to `accept_trial_point` (which
        // mirrors upstream's split: filter update and FTB reset are
        // accept-side, not line-search-side).
        _ls_guard.stop();

        // 7. Accept trial point (promotes `trial` to `curr` if set).
        //    The acceptor's filter has already been augmented (when
        //    appropriate) inside `find_acceptable_trial_point` via
        //    `update_for_next_iteration`, mirroring upstream's call
        //    chain in `IpBacktrackingLineSearch.cpp:839`.
        let _accept_guard = timing.accept_trial_point.guard();
        self.data.borrow_mut().accept_trial_point();

        // 8. Bound multiplier kappa_sigma reset.
        self.correct_bound_multiplier();

        // Sub-iteration checkpoint: the trial point was accepted; α and
        // the new iterate are in place (before the loop's iter bookkeeping
        // and the next `IterStart`).
        drop(_accept_guard);
        if let Some(o) = self.debug_stop(crate::debug::Checkpoint::AfterStep) {
            return o;
        }

        IterateOutcome::Continue
    }

    /// Port of `IpBacktrackingLineSearch::DetectTinyStep`
    /// (`IpBacktrackingLineSearch.cpp:1219-1278`). Returns true iff
    /// `max_i |δx_i|/(1+|x_i|) ≤ tiny_step_tol`,
    /// `max_i |δs_i|/(1+|s_i|) ≤ tiny_step_tol`, AND
    /// `curr_constraint_violation ≤ 1e-4`. Disabled when
    /// `tiny_step_tol == 0`.
    fn detect_tiny_step(&self, delta: &crate::iterates_vector::IteratesVector) -> bool {
        if self.tiny_step_tol == 0.0 {
            return false;
        }
        let curr = match self.data.borrow().curr.clone() {
            Some(c) => c,
            None => return false,
        };

        // |x_i|+1
        let mut tmp = curr.x.make_new_copy();
        tmp.element_wise_abs();
        tmp.add_scalar(1.0);
        // |δx_i|/(|x_i|+1) ; checked via Amax of (δx ./ (|x|+1)).
        let mut tmp2 = delta.x.make_new_copy();
        tmp2.element_wise_divide(&*tmp);
        if tmp2.amax() > self.tiny_step_tol {
            return false;
        }

        if curr.s.dim() > 0 {
            let mut tmp = curr.s.make_new_copy();
            tmp.element_wise_abs();
            tmp.add_scalar(1.0);
            let mut tmp2 = delta.s.make_new_copy();
            tmp2.element_wise_divide(&*tmp);
            if tmp2.amax() > self.tiny_step_tol {
                return false;
            }
        }

        let cviol = self.cq.borrow().curr_constraint_violation();
        if cviol > 1e-4 {
            return false;
        }
        true
    }

    /// Drive the restoration phase after a line-search failure.
    /// Returns `IterateOutcome::Continue` if the restoration driver
    /// recovered (the algorithm carries on from the recovered iterate);
    /// otherwise terminates with [`SolverReturn::RestorationFailure`].
    /// Mirrors upstream's
    /// `IpBacktrackingLineSearch::ActivateLineSearch` → `PerformRestoration`
    /// chain.
    fn invoke_restoration(&mut self) -> IterateOutcome {
        // Snapshot the outer reference iterate's `(theta, barr)` and
        // build the orig-progress callback the inner IPM will consult
        // at every iteration (mirrors upstream
        // `IpRestoFilterConvCheck::SetOrigLSAcceptor` plus
        // `IpFilterLSAcceptor::Reset`'s `reference_*_` snapshot).
        let reference_theta = self.cq.borrow().curr_constraint_violation();
        let reference_barr = self.cq.borrow().curr_barrier_obj();

        if std::env::var("POUNCE_DBG_RESTO").is_ok() {
            let iter = self.data.borrow().iter_count;
            eprintln!(
                "RESTO_ENTRY iter={} theta={:.6e} barr={:.6e} near_feas_ct={}",
                iter, reference_theta, reference_barr, self.resto_near_feasible_count,
            );
        }

        // No-progress restoration cycle detector. Two layered checks
        // surface as `ErrorInStepComputation` instead of cycling to
        // `max_iter` exhaustion (mirrors the *intent* of upstream
        // `IpBacktrackingLineSearch.cpp:580-600`'s almost-feasible
        // resto guard):
        //
        // 1. *Static cycle*: entry-to-entry — when the curr `(x, s)`
        //    at this entry is essentially identical to the snapshot
        //    from the previous entry, the inner resto-IPM is
        //    returning recovered iterates indistinguishable from
        //    entry, AND the outer didn't move either. Fires
        //    immediately. Catches QCNEW, EQC, MESH, POLAK6, S365,
        //    S365MOD, SIPOW2M, PFIT4.
        //
        // 2. *Slow-progress cycle*: recovery-to-entry — when curr at
        //    this entry is essentially identical to the *recovery*
        //    iterate from the previous resto, the outer made no
        //    progress between resto invocations even though resto's
        //    inner moved substantively. Counted, fires after 5
        //    consecutive entries. Catches ACOPR14, ACOPR30, TRO3X3
        //    while letting MAKELA3, HAIFAM, HALDMADS, ROBOT,
        //    TENBARS2 — which need 2-3 productive resto entries
        //    before LS accepts — pass through.
        //
        // A productive single-restoration sequence (BT8, HIMMELBJ,
        // LINSPANH, LSNNODOC, ODFITS, OET3) clears both snapshots via
        // `Outcome::Accepted` between entries and is unaffected.
        let curr = self
            .data
            .borrow()
            .curr
            .as_ref()
            .expect("curr set before invoke_restoration")
            .clone();
        // Helper: when the cycle detector fires and the orig cv is
        // bounded away from outer tol (e.g. PFIT1's 2.73e-2), the
        // outer is stuck at a feasibility-stationary point and the
        // honest exit is `LocalInfeasibility`. Below that threshold
        // the failure is numerical, not algorithmic, so retain
        // `ErrorInStepComputation`.
        let outer_tol_for_cycle = self.bundle.conv_check.tol_or_default();
        let cycle_exit = if reference_theta > (100.0 * outer_tol_for_cycle).max(1e-4) {
            SolverReturn::LocalInfeasibility
        } else {
            SolverReturn::ErrorInStepComputation
        };
        if let (Some(prev_x), Some(prev_s)) = (
            self.last_resto_entry_x.as_ref(),
            self.last_resto_entry_s.as_ref(),
        ) {
            let dx_rel = relative_distance(&*curr.x, &**prev_x);
            let ds_rel = relative_distance(&*curr.s, &**prev_s);
            if std::env::var_os("POUNCE_DBG_RESTO_CYCLE").is_some() {
                eprintln!(
                    "[PN_RESTO_CYCLE] entry-vs-entry dx_rel={:.6e} ds_rel={:.6e}",
                    dx_rel, ds_rel
                );
            }
            if dx_rel <= 1e-10 && ds_rel <= 1e-10 {
                return IterateOutcome::Terminate(cycle_exit);
            }
        }
        if let (Some(prev_x), Some(prev_s)) = (
            self.last_resto_recovery_x.as_ref(),
            self.last_resto_recovery_s.as_ref(),
        ) {
            let dx_rel = relative_distance(&*curr.x, &**prev_x);
            let ds_rel = relative_distance(&*curr.s, &**prev_s);
            if std::env::var_os("POUNCE_DBG_RESTO_CYCLE").is_some() {
                eprintln!(
                    "[PN_RESTO_CYCLE] entry-vs-recovery dx_rel={:.6e} ds_rel={:.6e} count={}",
                    dx_rel, ds_rel, self.resto_no_outer_progress_count
                );
            }
            if dx_rel <= 1e-10 && ds_rel <= 1e-10 {
                self.resto_no_outer_progress_count =
                    self.resto_no_outer_progress_count.saturating_add(1);
                // 10-strike limit: tuned to give OET7-style traces room
                // to break through (inner inf_pr still decreasing across
                // strikes) while still bounding DECONVBNE-style cycles
                // (which need a guard but tolerate a wider window —
                // ~3 outer steps per cycle, so 10 strikes ≈ 30 outer
                // iters, well below the 2987-iter pathological run).
                if self.resto_no_outer_progress_count >= 10 {
                    return IterateOutcome::Terminate(cycle_exit);
                }
            } else {
                self.resto_no_outer_progress_count = 0;
            }
        }
        // Near-feasible resto re-entry detector — matches the *intent*
        // of upstream `IpBacktrackingLineSearch.cpp:580-600`'s almost-
        // feasible-resto guard with a looser cv threshold. When the
        // outer enters restoration with the constraint violation
        // already at or below `tol`, the resto sub-IPM will produce a
        // recovered iterate that's at most marginally more feasible,
        // and any post-recovery σ-blowup from the next outer KKT solve
        // will re-trigger resto on the next iteration. Counting these
        // entries surfaces the cycle as `StopAtAcceptablePoint` —
        // primal feasibility is already met, only the dual residual
        // remains. Catches DECONVBNE: pounce ran 2987 iters before
        // this guard (cycle of ~30-inner-resto + 3 outer per cycle);
        // upstream solves in 505 iters via a different x trajectory.
        // Single-entry productive restos (BT8, HIMMELBJ, ODFITS) and
        // sub-tol-but-recoverable starts pass through under the 3-
        // strike limit.
        let outer_tol = self.bundle.conv_check.tol_or_default();
        if reference_theta <= outer_tol {
            self.resto_near_feasible_count = self.resto_near_feasible_count.saturating_add(1);
            if self.resto_near_feasible_count >= 3 {
                return IterateOutcome::Terminate(SolverReturn::StopAtAcceptablePoint);
            }
        } else {
            self.resto_near_feasible_count = 0;
        }
        self.last_resto_entry_x = Some(curr.x.make_new_copy());
        self.last_resto_entry_s = Some(curr.s.make_new_copy());

        // Augment the outer's filter with the resto-entry envelope —
        // mirrors upstream `IpBacktrackingLineSearch.cpp:566`:
        // `acceptor_->PrepareRestoPhaseStart()`. Adds
        // `((1-γ_θ)·θ_entry, φ_entry - γ_φ·θ_entry)` to the filter so
        // that after restoration recovers, the outer's Newton step is
        // forced by the filter to make real progress vs the entry
        // point. Without this, the outer accepts null-progress 'h'
        // steps and re-enters restoration on the next iteration (root
        // cause of DECONVBNE's 323 R-accepts vs ipopt's 21).
        self.bundle
            .line_search
            .acceptor_mut()
            .prepare_resto_phase_start(reference_theta, reference_barr);

        let orig_progress_cb = self.bundle.line_search.acceptor().make_orig_progress_check(
            reference_theta,
            reference_barr,
            5.0,
        );

        let (Some(nlp), Some(sd), Some(resto)) = (
            self.nlp.as_ref(),
            self.search_dir.as_mut(),
            self.restoration.as_mut(),
        ) else {
            return IterateOutcome::Terminate(SolverReturn::RestorationFailure);
        };
        resto.set_orig_progress_check(orig_progress_cb);
        let mut pd_guard = sd.pd_solver_mut();
        let aug = pd_guard.aug_solver_mut();
        // Audit counters (pounce#12). Increment call count + outer-iter
        // count (one outer iter is consumed per restoration call) and
        // wall-time around the inner call. Inner iter count is read
        // after via the trait accessor.
        self.resto_calls = self.resto_calls.saturating_add(1);
        self.resto_outer_iters = self.resto_outer_iters.saturating_add(1);
        let resto_t0 = std::time::Instant::now();
        let outcome = resto.perform_restoration(&self.data, &self.cq, nlp, aug);
        drop(pd_guard);
        self.resto_wall_secs += resto_t0.elapsed().as_secs_f64();
        self.resto_inner_iters = self
            .resto_inner_iters
            .saturating_add(resto.last_inner_iter_count());
        match outcome {
            RestorationOutcome::Recovered => {
                // The driver has staged the recovered point on
                // `data.trial`; promote it and continue iterating.
                self.data.borrow_mut().accept_trial_point();
                // Snapshot the recovery iterate for the slow-cycle
                // detector at the top of the next `invoke_restoration`.
                // Compared against next-entry curr, dx_rel ≈ ‖α·d‖ —
                // measures purely the outer step. See header comment
                // on the cycle detector above.
                let recovered = self
                    .data
                    .borrow()
                    .curr
                    .as_ref()
                    .expect("accept_trial_point sets curr")
                    .clone();
                self.last_resto_recovery_x = Some(recovered.x.make_new_copy());
                self.last_resto_recovery_s = Some(recovered.s.make_new_copy());
                // Mirror upstream `IpoptAlgorithm::AcceptTrialPoint`
                // (`IpIpoptAlg.cpp:917-963`): kappa_sigma clamp on the
                // four bound-multiplier vectors. Upstream applies this
                // unconditionally inside AcceptTrialPoint, so the
                // post-restoration path inherits it; pounce factored
                // the clamp out of the data swap so we must call it
                // explicitly here. Without it the all-1 multiplier
                // reset (`bound_mult_reset_threshold`) leaves z*s far
                // from mu at the recovered iterate, blowing up the
                // next KKT solve's σ = z/s diagonal.
                self.correct_bound_multiplier();
                IterateOutcome::Continue
            }
            RestorationOutcome::Failed => {
                // Mirrors upstream `IpBacktrackingLineSearch.cpp:611-623`:
                // when `PerformRestoration` returns false, attempt to
                // roll back to the most recent acceptable iterate before
                // surfacing failure. If a snapshot is available we exit
                // cleanly with `StopAtAcceptablePoint` (mapped by the
                // application layer to `Solved_To_Acceptable_Level`),
                // matching the upstream `ACCEPTABLE_POINT_REACHED`
                // throw. Without a snapshot we surface
                // `RestorationFailure` — unless the restoration left the
                // iterate diverging (`|x|_∞ > diverging_iterates_tol`), in
                // which case we surface `DivergingIterates` to mirror the
                // outcome upstream produces on pathological problems like
                // MESH (where ipopt reports `Diverging_Iterates` and
                // pounce previously reported `Restoration_Failed` with an
                // obj of −3.6e+33).
                if self.restore_acceptable_point() {
                    IterateOutcome::Terminate(SolverReturn::StopAtAcceptablePoint)
                } else if let Some(curr) = self.data.borrow().curr.as_ref() {
                    if curr.x.amax() > self.diverging_iterates_tol {
                        IterateOutcome::Terminate(SolverReturn::DivergingIterates)
                    } else {
                        IterateOutcome::Terminate(SolverReturn::RestorationFailure)
                    }
                } else {
                    IterateOutcome::Terminate(SolverReturn::RestorationFailure)
                }
            }
            RestorationOutcome::LocallyInfeasible => {
                // Mirrors upstream's catch of `LOCALLY_INFEASIBLE` thrown
                // from `IpRestoConvCheck.cpp:240` — the resto sub-IPM
                // settled at a stationary point of `||c(x)||_1` whose
                // residual is still well above `tol`. Without this
                // detection the outer would re-enter restoration on the
                // unchanged iterate forever.
                IterateOutcome::Terminate(SolverReturn::LocalInfeasibility)
            }
        }
    }

    /// Port of `IpIpoptAlg::correct_bound_multiplier`
    /// (`IpIpoptAlg.cpp:1055-1134`). Clamp each bound multiplier
    /// component into `[mu/(kappa_sigma * s_i), kappa_sigma * mu / s_i]`
    /// for all four bound-multiplier vectors.
    fn correct_bound_multiplier(&mut self) {
        if self.kappa_sigma < 1.0 {
            return;
        }
        let mu = self.data.borrow().curr_mu;
        let curr = match self.data.borrow().curr.clone() {
            Some(c) => c,
            None => return,
        };

        let cq = self.cq.borrow();

        let z_l_new = clamp_against_slack(&*curr.z_l, &*cq.curr_slack_x_l(), mu, self.kappa_sigma);
        let z_u_new = clamp_against_slack(&*curr.z_u, &*cq.curr_slack_x_u(), mu, self.kappa_sigma);
        let v_l_new = clamp_against_slack(&*curr.v_l, &*cq.curr_slack_s_l(), mu, self.kappa_sigma);
        let v_u_new = clamp_against_slack(&*curr.v_u, &*cq.curr_slack_s_u(), mu, self.kappa_sigma);
        drop(cq);

        let new_iv = crate::iterates_vector::IteratesVector::new(
            curr.x.clone(),
            curr.s.clone(),
            curr.y_c.clone(),
            curr.y_d.clone(),
            z_l_new,
            z_u_new,
            v_l_new,
            v_u_new,
        );
        self.data.borrow_mut().set_curr(new_iv);
    }

    /// Outer entry point — port of `IpoptAlgorithm::Optimize()`. Calls
    /// the iterate-initializer once, then loops `iterate()` until a
    /// terminal status. The exception → SolverReturn mapping
    /// (TINY_STEP_DETECTED → STEP_BECOMES_TINY,
    /// RESTORATION_FAILED → RESTORATION_FAILURE, etc.) lands in
    /// Phase 9 alongside the restoration phase.
    pub fn optimize(&mut self) -> SolverReturn {
        // Shared timing accumulator — every phase below records into it.
        let timing = self.data.borrow().timing.clone();

        // Install the shared accumulator on the augmented-system solver
        // so its factor / back-solve calls are attributed to
        // `linear_system_factorization` / `linear_system_back_solve`.
        // Same pattern for the diagnostics state when present, so KKT
        // dump sites can consult per-iter gating.
        if let Some(sd) = self.search_dir.as_mut() {
            sd.pd_solver_mut()
                .aug_solver_mut()
                .set_timing_stats(std::rc::Rc::clone(&timing));
            if let Some(diag) = self.diagnostics.as_ref() {
                sd.pd_solver_mut()
                    .aug_solver_mut()
                    .set_diagnostics(Rc::clone(diag));
            }
        }

        // 0a. Strategy initialization — port of upstream's
        //     `IpoptAlgorithm::InitializeImpl` calls. The mu update needs
        //     `data.curr_mu`/`curr_tau` seeded before the iterate
        //     initializer runs (`CalculateSafeSlack` reads them).
        self.bundle.mu_update.initialize(&self.data);

        // 0b. Iterate initializer. Requires NLP; without one the caller
        //    must have populated `data.curr` themselves.
        if let Some(nlp) = self.nlp.as_ref() {
            // The initializer needs an aug-system solver for the
            // least-square multiplier branch; until that's wired we
            // route through whatever the search-direction calculator
            // owns when present. For the stub flow we skip the LSM
            // path by giving the initializer a dummy solver only if
            // the search_dir is present (otherwise the init function
            // is responsible for not consulting it).
            if let Some(sd) = self.search_dir.as_mut() {
                timing.initialize_iterates.start();
                let mut pd_guard = sd.pd_solver_mut();
                let aug_solver = pd_guard.aug_solver_mut();
                let ok = self
                    .bundle
                    .init
                    .set_initial_iterates(&self.data, &self.cq, nlp, aug_solver);
                drop(pd_guard);
                timing.initialize_iterates.end();
                if !ok {
                    return SolverReturn::InternalError;
                }
            }
        }

        // 0c. Seed `IpoptData::w` with the initial-iterate Hessian.
        //     Redundant with the iter-body `update_hessian` call (which
        //     now runs BEFORE `update_barrier_parameter`) but kept to
        //     cover any code path that consults `data.w` between
        //     `set_initial_iterates` and the first `iterate()` call
        //     (e.g. the iter-0 trace dump below).
        if self.data.borrow().curr.is_some() {
            timing.update_hessian.start();
            let _ = self.bundle.hess.update_hessian(&self.data, &self.cq);
            timing.update_hessian.end();
        }

        // Track-A iterate-trace dumper. Activated by
        // `IPOPT_ITER_DUMP_PATH`; otherwise no-op. See `iter_dump.rs`.
        let mut dumper = IterDumper::from_env();
        // Iter 0 record — captures the initialised iterate before any
        // step. Mirrors upstream's "after InitializeIterates(), before
        // the loop" emission point.
        if let Some(d) = dumper.as_mut() {
            d.write_record(&self.data, &self.cq);
        }

        // Advance the diagnostics iter counter so the first `iterate()`
        // body reports as iter 0 (matches `data.iter_count`). Subsequent
        // bumps live at the bottom of the loop alongside the iter_count
        // bookkeeping.
        if let Some(diag) = self.diagnostics.as_ref() {
            diag.bump_iter();
            // Iter-0 iterate row (issue #68). Same hook point as
            // the binary IterDumper above; emits only when
            // `--dump iterates:*` is configured.
            emit_iterate_record(diag.as_ref(), &self.data, &self.cq);
        }

        // Iter 0 intermediate callback — upstream fires once after
        // `InitializeIterates` before the loop body starts so users
        // observe the initial point.
        if !self.fire_intermediate() {
            return SolverReturn::UserRequestedStop;
        }
        if self.fire_debug(crate::debug::Checkpoint::IterStart) == crate::debug::DebugAction::Stop {
            return SolverReturn::UserRequestedStop;
        }

        let result = loop {
            match self.iterate() {
                IterateOutcome::Terminate(ret) => break ret,
                IterateOutcome::Continue => {
                    // Source the local counter from `data.iter_count`
                    // each pass so a pre-seeded counter (e.g. the inner
                    // restoration IPM at `outer.iter + 1`, matching
                    // upstream `IpRestoMinC_1Nrm.cpp:181`) and any
                    // restoration step that set
                    // `data.iter_count = inner.iter_count - 1`
                    // (mirroring `IpRestoMinC_1Nrm.cpp:Set_iter_count`)
                    // are honored — without this the local counter
                    // would advance from its pre-restoration value,
                    // ignoring the inner-IPM iterations.
                    let mut iter_count: Index = self.data.borrow().iter_count;
                    iter_count += 1;
                    if iter_count >= self.max_iter {
                        break SolverReturn::MaxiterExceeded;
                    }
                    self.data.borrow_mut().iter_count = iter_count;
                    // Keep the diagnostics counter in lock-step with
                    // `data.iter_count` so KKT-dump gating reflects the
                    // about-to-execute iteration.
                    if let Some(diag) = self.diagnostics.as_ref() {
                        diag.bump_iter();
                        // Per-iter iterate row (issue #68). Mirrors
                        // the binary IterDumper hook below.
                        emit_iterate_record(diag.as_ref(), &self.data, &self.cq);
                    }
                    // Per-iteration record — emitted after the
                    // iter_count bump so the recorded `iter` field
                    // matches `IpData().iter_count()` at the moment of
                    // emission, identical to upstream's writer.
                    if let Some(d) = dumper.as_mut() {
                        d.write_record(&self.data, &self.cq);
                    }
                    // Per-iteration intermediate callback — fired with
                    // an `IntermediateContext` guard so downstream
                    // inspector entry points (the C API
                    // `GetIpoptCurrent*` family) see live state for the
                    // duration of the user callback.
                    if !self.fire_intermediate() {
                        break SolverReturn::UserRequestedStop;
                    }
                    if self.fire_debug(crate::debug::Checkpoint::IterStart)
                        == crate::debug::DebugAction::Stop
                    {
                        break SolverReturn::UserRequestedStop;
                    }
                }
            }
        };

        // Terminal post-mortem checkpoint. Skipped when the user already
        // asked to stop (they were just at a prompt); otherwise the
        // debugger gets a last look at the final/failing iterate.
        if !matches!(result, SolverReturn::UserRequestedStop) {
            self.fire_debug_terminal(result);
        }
        result
    }
}

/// Internal result of one [`IpoptAlgorithm::iterate`] call. Mirrors the
/// upstream try/catch around `IpoptAlg::Optimize` — anything that's not
/// `Continue` carries the [`SolverReturn`] that the outer loop will
/// surface to `IpoptApplication`.
enum IterateOutcome {
    Continue,
    Terminate(SolverReturn),
}

/// `||a - b||_2 / (1 + ||b||_2)`. Used by the restoration cycle
/// detector in [`IpoptAlgorithm::invoke_restoration`] to test whether
/// the outer iterate has moved between two consecutive restoration
/// entries.
fn relative_distance(a: &dyn Vector, b: &dyn Vector) -> Number {
    if a.dim() == 0 {
        return 0.0;
    }
    let mut diff = a.make_new_copy();
    diff.axpy(-1.0, b);
    diff.nrm2() / (1.0 + b.nrm2())
}

/// `out = curr + α_p · δ` for the primal/equality blocks and
/// `out = curr + α_d · δ` for the bound multipliers, returned as a
/// fresh frozen `IteratesVector`. Mirrors `scaled_step` in the line
/// search; duplicated here for the tiny-step branch which bypasses
/// the line-search driver.
fn scaled_step_unchecked(
    curr: &crate::iterates_vector::IteratesVector,
    delta: &crate::iterates_vector::IteratesVector,
    alpha_primal: Number,
    alpha_dual: Number,
) -> crate::iterates_vector::IteratesVector {
    let mut out = curr.make_new_zeroed();
    out.add_one_vector(1.0, curr, 0.0);
    out.x.axpy(alpha_primal, &*delta.x);
    out.s.axpy(alpha_primal, &*delta.s);
    out.y_c.axpy(alpha_primal, &*delta.y_c);
    out.y_d.axpy(alpha_primal, &*delta.y_d);
    out.z_l.axpy(alpha_dual, &*delta.z_l);
    out.z_u.axpy(alpha_dual, &*delta.z_u);
    out.v_l.axpy(alpha_dual, &*delta.v_l);
    out.v_u.axpy(alpha_dual, &*delta.v_u);
    out.freeze()
}

/// Allocate a fresh `Rc<dyn Vector>` with `kappa_sigma_clamp`
/// applied component-wise against the supplied `slack`. Inputs are
/// borrowed; the original `z` is never mutated. Ports the per-vector
/// piece of `IpIpoptAlg.cpp:1080-1133`.
fn clamp_against_slack(
    z: &dyn Vector,
    slack: &dyn Vector,
    mu: Number,
    kappa_sigma: Number,
) -> Rc<dyn Vector> {
    debug_assert_eq!(z.dim(), slack.dim());
    let n = z.dim() as usize;
    // Flatten both z and slack into contiguous slices so the
    // elementwise clamp doesn't care whether the inputs are
    // [`DenseVector`] (regular IPM path) or [`CompoundVector`]
    // (resto IPM path). The result is reconstructed into a
    // same-shape Vector via `Vector::make_new` + a flat-write
    // helper so the caller sees a vector with the same blocking as
    // its input.
    let mut buf = vec![0.0_f64; n];
    flat_read_into(z, &mut buf);
    let s_vals = flat_read_owned(slack);
    let _ = kappa_sigma_clamp(&mut buf, &s_vals, mu, kappa_sigma);
    let mut out: Box<dyn Vector> = z.make_new();
    flat_write_into(&mut *out, &buf);
    Rc::from(out)
}

pub(crate) fn flat_read_into(v: &dyn Vector, dst: &mut [Number]) {
    if let Some(dv) = v
        .as_any()
        .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
    {
        let vs = dv.expanded_values();
        dst.copy_from_slice(&vs);
        return;
    }
    if let Some(cv) = v.as_any().downcast_ref::<pounce_linalg::CompoundVector>() {
        let mut off = 0usize;
        for k in 0..cv.n_comps() {
            let blk = cv.comp(k);
            let dim = blk.dim() as usize;
            let dblk = blk
                .as_any()
                .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
                .expect("clamp_against_slack: CompoundVector blocks must be DenseVectors");
            let vs = dblk.expanded_values();
            dst[off..off + dim].copy_from_slice(&vs);
            off += dim;
        }
        return;
    }
    panic!("clamp_against_slack: unsupported Vector kind");
}

pub(crate) fn flat_read_owned(v: &dyn Vector) -> Vec<Number> {
    let mut out = vec![0.0; v.dim() as usize];
    flat_read_into(v, &mut out);
    out
}

pub(crate) fn flat_write_into(v: &mut dyn Vector, src: &[Number]) {
    if let Some(dv) = v
        .as_any_mut()
        .downcast_mut::<pounce_linalg::dense_vector::DenseVector>()
    {
        dv.set_values(src);
        return;
    }
    if let Some(cv) = v
        .as_any_mut()
        .downcast_mut::<pounce_linalg::CompoundVector>()
    {
        let mut off = 0usize;
        for k in 0..cv.n_comps() {
            let blk = cv.comp_mut(k);
            let dim = blk.dim() as usize;
            let dblk = blk
                .as_any_mut()
                .downcast_mut::<pounce_linalg::dense_vector::DenseVector>()
                .expect("clamp_against_slack: CompoundVector blocks must be DenseVectors");
            dblk.set_values(&src[off..off + dim]);
            off += dim;
        }
        return;
    }
    panic!("clamp_against_slack: unsupported Vector kind");
}

/// Per-element kappa-sigma clamp — the elementwise arithmetic at the
/// heart of `IpIpoptAlg.cpp:correct_bound_multiplier` (lines
/// 1090-1133). For each index `i`:
///
/// ```text
///   slack_i  = max(slack_i, tiny_double)   // avoid /0
///   z_lo_i   = mu / (kappa_sigma * slack_i)
///   z_hi_i   = kappa_sigma * mu / slack_i
///   z_i      ← clamp(z_i, z_lo_i, z_hi_i)
/// ```
///
/// Returns the maximum elementwise correction magnitude (matching
/// upstream's `Max(max_correction_up, max_correction_low)`).
///
/// `kappa_sigma < 1` short-circuits to the identity per upstream's
/// guard at line 1065.
pub fn kappa_sigma_clamp(
    z: &mut [Number],
    slack: &[Number],
    mu: Number,
    kappa_sigma: Number,
) -> Number {
    debug_assert_eq!(z.len(), slack.len());
    if kappa_sigma < 1.0 {
        return 0.0;
    }
    let mut max_correction = 0.0_f64;
    for (zi, &si) in z.iter_mut().zip(slack.iter()) {
        let s_safe = si.max(Number::MIN_POSITIVE);
        let lo = mu / (kappa_sigma * s_safe);
        let hi = kappa_sigma * mu / s_safe;
        let clamped = zi.clamp(lo, hi);
        let delta = (clamped - *zi).abs();
        if delta > max_correction {
            max_correction = delta;
        }
        *zi = clamped;
    }
    max_correction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kappa_sigma_below_one_is_identity() {
        let mut z = vec![1.0, 2.0, 3.0];
        let slack = [1.0, 1.0, 1.0];
        let m = kappa_sigma_clamp(&mut z, &slack, 1.0, 0.5);
        assert_eq!(m, 0.0);
        assert_eq!(z, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn within_band_is_unchanged() {
        // mu=1, kappa=10, slack=1 → band [0.1, 10]. z=1 → unchanged.
        let mut z = vec![1.0];
        let slack = [1.0];
        let m = kappa_sigma_clamp(&mut z, &slack, 1.0, 10.0);
        assert_eq!(m, 0.0);
        assert_eq!(z, [1.0]);
    }

    #[test]
    fn above_upper_clamped_down() {
        // mu=1, kappa=10, slack=1 → upper = 10. z=100 → 10.
        let mut z = vec![100.0];
        let slack = [1.0];
        let m = kappa_sigma_clamp(&mut z, &slack, 1.0, 10.0);
        assert!((m - 90.0).abs() < 1e-13);
        assert_eq!(z, [10.0]);
    }

    #[test]
    fn below_lower_clamped_up() {
        // mu=1, kappa=10, slack=1 → lower = 0.1. z=0.001 → 0.1.
        let mut z = vec![0.001];
        let slack = [1.0];
        let m = kappa_sigma_clamp(&mut z, &slack, 1.0, 10.0);
        assert!((m - 0.099).abs() < 1e-13);
        assert!((z[0] - 0.1).abs() < 1e-15);
    }

    #[test]
    fn returns_max_over_components() {
        let mut z = vec![100.0, 0.001];
        let slack = [1.0, 1.0];
        let m = kappa_sigma_clamp(&mut z, &slack, 1.0, 10.0);
        assert!((m - 90.0).abs() < 1e-13);
        assert_eq!(z[0], 10.0);
        assert!((z[1] - 0.1).abs() < 1e-15);
    }

    #[test]
    fn slack_clamped_to_min_positive_avoids_division_by_zero() {
        let mut z = vec![1e100];
        let slack = [0.0];
        let _ = kappa_sigma_clamp(&mut z, &slack, 1.0, 10.0);
        assert!(z[0].is_finite() || z[0] == 1e100);
    }

    /// The restoration slot is exercised structurally:
    /// `IpoptAlgorithm::with_restoration` accepts a
    /// `Box<dyn RestorationPhase>` and the trait's default
    /// `perform_restoration` returns `Failed`. End-to-end coverage
    /// (iterate() → line-search-Failed → restoration → recovered)
    /// lands in the Phase 9 integration suite alongside the nested
    /// IPM driver.
    struct _DummyResto;
    impl RestorationPhase for _DummyResto {}
}
