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
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iter_dump::IterDumper;
use crate::kkt::pd_search_dir_calc::PdSearchDirCalc;
use crate::line_search::backtracking::Outcome;
use crate::restoration::{RestorationOutcome, RestorationPhase};
use pounce_common::types::{Index, Number};
use pounce_linalg::Vector;
use pounce_nlp::alg_types::SolverReturn;
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
            search_dir,
            restoration: None,
            kappa_sigma: 1e10,
            max_iter: 3000,
            alpha_init: 1.0,
            tiny_step_tol: 10.0 * Number::EPSILON,
            tiny_step_y_tol: 1e-2,
            tiny_step_last_iteration: false,
        }
    }

    pub fn with_nlp(mut self, nlp: Rc<RefCell<dyn IpoptNlp>>) -> Self {
        self.nlp = Some(nlp);
        self
    }

    pub fn with_search_dir(mut self, sd: PdSearchDirCalc) -> Self {
        self.search_dir = Some(sd);
        self
    }

    pub fn with_restoration(mut self, resto: Box<dyn RestorationPhase>) -> Self {
        self.restoration = Some(resto);
        self
    }

    /// One iteration body — port of `Optimize()`'s inner loop.
    /// Returns either `Continue` to keep iterating or a terminal
    /// [`SolverReturn`] mirroring upstream's exception → return-code
    /// translation table (see `MAIN_LOOP.md` §"Exception mapping").
    fn iterate(&mut self) -> IterateOutcome {
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
        self.bundle.iter_output.write_output();
        {
            let iter_count = self.data.borrow().iter_count;
            if iter_count % 10 == 0 {
                print!("{}", crate::output::orig::OrigIterationOutput::HEADER);
            }
            let row = self.bundle.iter_output.format_row(&self.data, &self.cq);
            println!("{row}");
        }

        // Reset per-iteration info on data (after printing previous
        // iter's accepted-step info; before the next line search).
        self.data.borrow_mut().reset_info();

        // 2. Convergence check.
        let nlp_err = self.cq.borrow().curr_nlp_error();
        let iter_count = self.data.borrow().iter_count;
        if !nlp_err.is_finite() {
            return IterateOutcome::Terminate(SolverReturn::InvalidNumberDetected);
        }
        match self.bundle.conv_check.check_convergence_with_state(
            nlp_err,
            iter_count,
            &self.data,
            &self.cq,
        ) {
            ConvergenceStatus::Continue => {}
            ConvergenceStatus::Converged => {
                return IterateOutcome::Terminate(SolverReturn::Success);
            }
            ConvergenceStatus::MaxIterExceeded => {
                return IterateOutcome::Terminate(SolverReturn::MaxiterExceeded);
            }
            ConvergenceStatus::Failed => {
                return IterateOutcome::Terminate(SolverReturn::InternalError);
            }
        }

        // 3. Barrier parameter. Pass nlp + search_dir through so the
        // adaptive μ oracles (probing, quality-function) can drive
        // their own affine-step solves; monotone ignores them.
        // Snapshot the tiny-step flag (set by the previous iteration's
        // tiny-step branch) and the entry mu — if μ can't reduce while
        // the flag is on, upstream `IpMonotoneMuUpdate.cpp:158-161`
        // throws TINY_STEP_DETECTED → STOP_AT_TINY_STEP, which we
        // realise as a clean termination here.
        let tiny_at_entry = self.data.borrow().tiny_step_flag;
        let mu_before = self.data.borrow().curr_mu;
        let next_mu = self.bundle.mu_update.update_barrier_parameter(
            &self.data,
            &self.cq,
            self.nlp.as_ref(),
            self.search_dir.as_mut(),
        );
        self.data.borrow_mut().curr_mu = next_mu;
        if tiny_at_entry && (next_mu - mu_before).abs() < Number::EPSILON {
            return IterateOutcome::Terminate(SolverReturn::StopAtTinyStep);
        }

        // 4. Hessian update.
        let _ = self.bundle.hess.update_hessian(&self.data, &self.cq);

        // 5. Search direction. Skipped without an NLP + search_dir.
        if let (Some(nlp), Some(sd)) = (self.nlp.as_ref(), self.search_dir.as_mut()) {
            let ok = sd.compute_search_direction(&self.data, &self.cq, nlp);
            if !ok {
                // Upstream's `STEP_COMPUTATION_FAILED` exception →
                // `ErrorInStepComputation` (table in `MAIN_LOOP.md`).
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
                        if let Some(dv_orig) = cdx.comp(0).as_any().downcast_ref::<pounce_linalg::dense_vector::DenseVector>() {
                            let v = dv_orig.values();
                            let mut imax = 0usize;
                            let mut amax = 0.0f64;
                            for (i, &x) in v.iter().enumerate() {
                                if x.abs() > amax { amax = x.abs(); imax = i; }
                            }
                            eprintln!("[PN_DELTA] iter={} dx_orig argmax: i={} v={:.17e} (n={})", it, imax, v[imax], v.len());
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
                                eprintln!("[PN_DELTA] iter={} zU_block[{}]_amax={:.6e} dim={}",
                                    it, ic, czu.comp(ic).amax(), czu.comp(ic).dim());
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
                        if let Some(xo) = cx.comp(0).as_any().downcast_ref::<pounce_linalg::dense_vector::DenseVector>() {
                            let v = xo.values();
                            let mut imax = 0usize;
                            let mut amax = 0.0f64;
                            for (i, &x) in v.iter().enumerate() {
                                if x.abs() > amax { amax = x.abs(); imax = i; }
                            }
                            eprintln!("[PN_DELTA] iter={} curr_x_orig argmax: i={} v={:.17e} amax={:.17e} nrm2={:.17e}",
                                it, imax, v[imax], xo.amax(), xo.nrm2());
                        }
                    }
                    }
                }
            }
        }

        // 6. Acceptable trial point — run the line search if we have a
        //    primal/dual step on `data.delta`.
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
            let alpha_p_max = self
                .cq
                .borrow()
                .aff_step_alpha_primal_max(&delta, tau);
            let alpha_d_max = self
                .cq
                .borrow()
                .aff_step_alpha_dual_max(&delta, tau);

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
                    Outcome::Accepted => {}
                    Outcome::TinyStep | Outcome::Failed => {
                        // Upstream `IpBacktrackingLineSearch.cpp` raises
                        // `LINE_SEARCH_FAILED` when α drops below
                        // `alpha_min` or all retries reject, which in
                        // turn triggers `ActivateLineSearch` →
                        // restoration.
                        return self.invoke_restoration();
                    }
                }
            }
        }

        // 7. Accept trial point (promotes `trial` to `curr` if set).
        //    The acceptor's filter has already been augmented (when
        //    appropriate) inside `find_acceptable_trial_point` via
        //    `update_for_next_iteration`, mirroring upstream's call
        //    chain in `IpBacktrackingLineSearch.cpp:839`.
        self.data.borrow_mut().accept_trial_point();

        // 8. Bound multiplier kappa_sigma reset.
        self.correct_bound_multiplier();

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
        let orig_progress_cb = self
            .bundle
            .line_search
            .acceptor()
            .make_orig_progress_check(reference_theta, reference_barr, 5.0);

        let (Some(nlp), Some(sd), Some(resto)) = (
            self.nlp.as_ref(),
            self.search_dir.as_mut(),
            self.restoration.as_mut(),
        ) else {
            return IterateOutcome::Terminate(SolverReturn::RestorationFailure);
        };
        resto.set_orig_progress_check(orig_progress_cb);
        let aug = sd.pd_solver_mut().aug_solver_mut();
        match resto.perform_restoration(&self.data, &self.cq, nlp, aug) {
            RestorationOutcome::Recovered => {
                // The driver has staged the recovered point on
                // `data.trial`; promote it and continue iterating.
                self.data.borrow_mut().accept_trial_point();
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
                IterateOutcome::Terminate(SolverReturn::RestorationFailure)
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
                let aug_solver = sd.pd_solver_mut().aug_solver_mut();
                let ok = self
                    .bundle
                    .init
                    .set_initial_iterates(&self.data, &self.cq, nlp, aug_solver);
                if !ok {
                    return SolverReturn::InternalError;
                }
            }
        }

        // 0c. Seed `IpoptData::w` with the initial-iterate Hessian so
        //     the first `update_barrier_parameter` call (adaptive mu
        //     oracles drive an affine solve) finds W populated. Matches
        //     upstream's `InitializeImpl` which leaves W set after
        //     `DefaultIterateInitializer::SetInitialIterates`.
        if self.data.borrow().curr.is_some() {
            let _ = self.bundle.hess.update_hessian(&self.data, &self.cq);
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

        loop {
            match self.iterate() {
                IterateOutcome::Terminate(ret) => return ret,
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
                        return SolverReturn::MaxiterExceeded;
                    }
                    self.data.borrow_mut().iter_count = iter_count;
                    // Per-iteration record — emitted after the
                    // iter_count bump so the recorded `iter` field
                    // matches `IpData().iter_count()` at the moment of
                    // emission, identical to upstream's writer.
                    if let Some(d) = dumper.as_mut() {
                        d.write_record(&self.data, &self.cq);
                    }
                }
            }
        }
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

fn flat_read_into(v: &dyn Vector, dst: &mut [Number]) {
    if let Some(dv) = v.as_any().downcast_ref::<pounce_linalg::dense_vector::DenseVector>() {
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

fn flat_read_owned(v: &dyn Vector) -> Vec<Number> {
    let mut out = vec![0.0; v.dim() as usize];
    flat_read_into(v, &mut out);
    out
}

fn flat_write_into(v: &mut dyn Vector, src: &[Number]) {
    if let Some(dv) = v
        .as_any_mut()
        .downcast_mut::<pounce_linalg::dense_vector::DenseVector>()
    {
        dv.set_values(src);
        return;
    }
    if let Some(cv) = v.as_any_mut().downcast_mut::<pounce_linalg::CompoundVector>() {
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
