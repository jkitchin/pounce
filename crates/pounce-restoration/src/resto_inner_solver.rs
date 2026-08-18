//! Nested-IPM driver for the restoration phase.
//!
//! Wires the resto-side bundle ([`crate::resto_alg_builder::RestoAlgorithmBundle`])
//! together with a regular-phase [`pounce_algorithm::alg_builder::AlgorithmBundle`]
//! and runs `optimize()` on the resulting nested
//! [`pounce_algorithm::ipopt_alg::IpoptAlgorithm`]. Returns the recovered
//! `(orig_x, orig_s)` to the calling
//! [`crate::min_c_1nrm::MinC1NormRestoration`] driver via
//! [`crate::min_c_1nrm::RestoSolveResult`].
//!
//! v0.1 scope (Phase 9 — *minimum runnable*):
//!
//! * The inner IPM's `conv_check` / `iter_output` slots are overridden
//!   with [`crate::conv_check::RestoConvCheckAdapter`] (enforces the
//!   resto-side `maximum_iters` / `maximum_resto_iters` caps and
//!   delegates inner-stationarity to a wrapped `OptErrorConvCheck`)
//!   and [`crate::output::RestoIterationOutputAdapter`] (the resto
//!   `iter`-with-`r`-suffix formatter). The kappa-reduction guard and
//!   the outer-filter acceptance test in
//!   [`crate::conv_check::RestoConvCheck`] / `RestoFilterConvCheck`
//!   stay deferred to the outer line search's post-restoration recheck
//!   per the comment below — the v0.1 trait surface
//!   `(nlp_err, iter_count) -> ConvergenceStatus` doesn't expose the
//!   inner iterate's orig-NLP infeasibility.
//! * The init slot is overridden with the resto-side
//!   [`crate::init::RestoIterateInitializer`], threaded with the
//!   [`crate::init::OuterIterateSnapshot`] captured from the outer
//!   `(IpoptData, IpoptCq)` at restoration entry.
//! * Recovery extracts `block 0` of the inner-final compound `x`
//!   ([`crate::resto_nlp::BLOCK_X`]) and clones the inner-final `s`.
//!
//! The restoration-specific termination logic upstream
//! (`IpRestoFilterConvergenceCheck::CheckConvergence`) gates the
//! return to the outer line search on the *outer* filter's acceptance
//! of the recovered iterate; the v0.1 wiring relies on the outer line
//! search re-checking the trial point post-`perform_restoration` so
//! the bit-equivalence behavior is preserved on the entry/exit
//! handshake even though the inner termination is looser.

use crate::init::OuterIterateSnapshot;
use crate::min_c_1nrm::RestoSolveResult;
use crate::resto_alg_builder::RestoAlgorithmBuilder;
use crate::resto_nlp::BLOCK_X;
use pounce_algorithm::alg_builder::{AlgorithmBuilder, LinearBackendFactory};
use pounce_algorithm::ipopt_alg::IpoptAlgorithm;
use pounce_algorithm::ipopt_cq::{IpoptCalculatedQuantities, IpoptCqHandle, unscaled_block_amax};
use pounce_algorithm::ipopt_data::{IpoptData, IpoptDataHandle};
use pounce_algorithm::ipopt_nlp::IpoptNlp;
use pounce_algorithm::iterates_vector::IteratesVector;
use pounce_algorithm::mu::monotone::MonotoneMuUpdate;
use pounce_common::tolerance::is_significant;
use pounce_common::types::Index;
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use pounce_linalg::{CompoundVector, Vector};
use pounce_nlp::alg_types::SolverReturn;
use std::cell::RefCell;
use std::rc::Rc;

/// Factory closure type for the linear-backend factory used by the
/// inner IPM. Re-invoked per restoration entry so each nested
/// [`IpoptAlgorithm`] gets a fresh backend (mirroring upstream's
/// `IpAlgBuilder` re-instantiating the resto sub-algorithm on every
/// trigger).
pub type InnerBackendFactoryFactory = Box<dyn FnMut() -> LinearBackendFactory>;

/// Per-call iteration cap on the restoration sub-IPM
/// (`IpRestoConvCheck.cpp:137` `maximum_iters`). Resto-phase-specific,
/// distinct from the user's outer `max_iter` — the restoration phase is
/// a feasibility-recovery side-solve and gets its own generous budget.
const RESTO_INNER_MAX_ITERS: i32 = 3000;

/// Cap on successive restoration iterations
/// (`IpRestoConvCheck.cpp:144` `maximum_resto_iters`).
const RESTO_MAX_SUCCESSIVE_ITERS: i32 = 3000;

/// How much worse than the violation restoration was *entered* at a
/// recovered point may be and still support a locally-infeasible verdict
/// from one of the reconstructed gates in [`run_inner_resto`] (gh#661).
///
/// Those gates infer "the sub-solve stalled" from a terminal status and
/// then test only that the violation is large. A diverging restoration
/// passes that size test ever more strongly the further it blows up, which
/// is how a model that solves at default options got reported infeasible at
/// a point 105,700x worse than restoration's own starting violation. One
/// order of magnitude lets a plateau wander — the signature these gates
/// describe sits at ~1x entry — without admitting a blow-up.
const RESTO_DIVERGENCE_HEADROOM: f64 = 10.0;

/// Outer iterations spent at an unimprovable constraint violation after
/// which a solve is taken to have demonstrated a floor, independent of
/// how its restoration sub-solve finally exited.
///
/// Measured by [`pounce_algorithm::inf_pr_floor::InfPrFloor`], not by a raw iteration count. The
/// distinction is the whole of gh#664's residual defect: a count says the
/// solve *ran* a long time, which is not the claim the reconstructed
/// gates make. The claim is that it ran out of room, and a solve can
/// burn any number of iterations while still descending.
///
/// The threshold sits in a wide measured gap. Over the fixture sweep the
/// two trajectories that earn the exemption accumulate 943 and 890
/// iterations at their floor. Across all 57 fixtures on three option
/// arms, every row where the guard engages sits at 31 or below —
/// `pooling_rt2stp` at 7, `hs71_obj1e8` at 1, bounded above by their own
/// outer solve lengths. 200 is 6x clear of the largest non-floor case and
/// 4x under the smallest genuine one, so neither side is decided by where
/// exactly the line falls inside that band.
const RESTO_STALL_EVIDENCE_ITERS: Index = 200;

/// Whether a restoration sub-solve ended somewhere enough *worse* than it
/// began that the five reconstructed gates in [`run_inner_resto`] have no
/// basis for a locally-infeasible verdict (gh#661).
///
/// `orig_curr_inf_pr` is the original-NLP violation the outer handed to this
/// restoration call; `orig_inf_pr_scaled` is where the sub-solve left it.
/// Both are in the scaled space (`orig_curr_inf_pr` is the same reference the
/// kappa-reduction guard measures progress against). Restoration exists to
/// *reduce* that number, so a point an order of magnitude worse than entry is
/// not a point restoration converged to.
///
/// Three ways this returns `false`, each deliberate:
///
/// - **No usable reference.** A non-finite or non-positive entry violation
///   gives nothing to measure divergence against, so the gates are left as
///   they were rather than suppressed on a ratio against zero.
/// - **The floor is already evidenced.** `outer_iters_at_floor` is
///   [`pounce_algorithm::inf_pr_floor::InfPrFloor::iters_at_floor`]
///   off the *outer* `IpoptData`: how many outer iterates sat within an
///   order of magnitude of the floor that count is measured against — a
///   reference pinned where the count started, so a solve still working
///   its way down clears it and restarts rather than accumulating.
///   Past [`RESTO_STALL_EVIDENCE_ITERS`] of those
///   the solve has shown it cannot get below that violation, and a
///   restoration blow-up at the end is the tail of that trajectory rather
///   than a description of it. `issue_508_infeasible_gap_1em2` spends 943
///   of its 1016 outer iterations pinned at the `1.0e-2` gap the fixture is
///   built around before its four-iteration sub-solve jumps to `3.19e9`;
///   the final `1e11x` ratio is an artifact of those four iterations.
/// - **Within headroom.** A plateau, which is the signature these gates
///   actually describe, lands at ~1x entry.
///
/// The evidence is read at outer scope on purpose. gh#664 read the *inner*
/// IPM's `iter_count`, which its comments described as the sub-solve's own
/// length; it is not. That counter is seeded from the outer's to mirror
/// upstream `IpRestoMinC_1Nrm.cpp:181`, so the `1019` those comments cited
/// is 1015 outer iterations plus a four-iteration sub-solve. Restoration
/// sub-solves are short — 4 to 12 iterations across every fixture measured
/// — so there was never a thousand-iteration sub-solve to observe, and any
/// per-sub-solve measure of a long stall is measuring a window too small to
/// hold one. The long trajectory lives in the outer loop.
///
/// Never applied to the layer-2 verdict (gh#438): that one is not
/// reconstructed — the sub-solve's own convergence check issued it at a point
/// it certified — so it already carries the stall evidence the other five
/// infer.
fn diverged_from_restoration_entry(
    orig_curr_inf_pr: f64,
    orig_inf_pr_scaled: f64,
    outer_iters_at_floor: Index,
) -> bool {
    let entry_reference_usable = orig_curr_inf_pr.is_finite() && orig_curr_inf_pr > 0.0;
    let floor_already_evidenced = outer_iters_at_floor >= RESTO_STALL_EVIDENCE_ITERS;

    entry_reference_usable
        && !floor_already_evidenced
        && orig_inf_pr_scaled > RESTO_DIVERGENCE_HEADROOM * orig_curr_inf_pr
}

/// Build the restoration convergence-check adapter, threading the inner
/// IPM's *user-derived* stationarity tolerances (`tol`,
/// `acceptable_tol`, `acceptable_iter`) through to the sub-solve.
///
/// Upstream clones the outer `OptionsList` into the restoration
/// sub-options (`IpRestoMinC_1Nrm.cpp`), so the resto IPM inherits the
/// user's `tol`/`acceptable_tol`/`acceptable_iter` unless a `resto.`
/// override is set. Pounce previously hardcoded `(1e-8, 1e-6, 15)`
/// here, so a user asking for e.g. `tol=1e-3` still drove the resto
/// sub-solve to `1e-8` stationarity (and vice-versa). Reading the
/// inner builder's `conv_check` options — which
/// [`pounce_algorithm::application::IpoptApplication::algorithm_builder_from_options`]
/// populates from the same `tol`/`acceptable_tol`/`acceptable_iter`
/// options the outer solve reads — restores upstream's inheritance. The
/// two iteration budgets stay at the resto-phase constants.
fn build_resto_conv_check_adapter(
    conv: &pounce_algorithm::alg_builder::ConvCheckOptions,
) -> crate::conv_check::RestoConvCheckAdapter {
    crate::conv_check::RestoConvCheckAdapter::new(
        conv.tol,
        conv.acceptable_tol,
        conv.acceptable_iter,
        RESTO_INNER_MAX_ITERS,
        RESTO_MAX_SUCCESSIVE_ITERS,
    )
}

/// Build a [`crate::min_c_1nrm::RestoInnerSolver`] closure that
/// constructs and runs the nested IPM on every restoration entry.
///
/// `resto_builder` carries the resto-NLP knobs (`rho`, `eta_factor`,
/// reset thresholds, ...). `inner_alg_builder` is the regular-phase
/// builder template used to assemble the nested bundle (line search,
/// mu update, hessian, scaling). `backend_factory_factory` is invoked
/// once per restoration entry to produce a fresh
/// [`LinearBackendFactory`] (because `build_with_backend` consumes it).
pub fn make_resto_inner_solver(
    resto_builder: RestoAlgorithmBuilder,
    inner_alg_builder: AlgorithmBuilder,
    mut backend_factory_factory: InnerBackendFactoryFactory,
) -> crate::min_c_1nrm::RestoInnerSolver {
    Box::new(
        move |outer_data,
              outer_cq,
              outer_nlp,
              orig_progress_cb,
              print_iter_output,
              debug_hook,
              intermediate_tnlp| {
            run_inner_resto(
                outer_data,
                outer_cq,
                outer_nlp,
                &resto_builder,
                &inner_alg_builder,
                backend_factory_factory(),
                orig_progress_cb,
                print_iter_output,
                debug_hook,
                intermediate_tnlp,
            )
        },
    )
}

/// Build a `Box<dyn RestorationPhase>` that wraps a
/// [`crate::min_c_1nrm::MinC1NormRestoration`] driver with its
/// `inner_solver` hook wired to the nested IPM produced by
/// [`make_resto_inner_solver`]. The closure returned has signature
/// `FnMut() -> Box<dyn RestorationPhase>` so it slots straight into
/// [`pounce_algorithm::application::IpoptApplication::set_restoration_factory`].
///
/// One-shot: the returned closure can only be called once per
/// `optimize_constrained` invocation. Callers that need to run the
/// inner IPM more than once per `optimize_tnlp` — the ℓ₁ outer loop,
/// the ℓ₁-on-restoration-failure auto-fallback — must instead use
/// [`make_default_restoration_factory_provider`] together with
/// [`pounce_algorithm::application::IpoptApplication::set_restoration_factory_provider`].
/// Copy the restoration options the outer [`AlgorithmBuilder`] carries
/// (read from the `OptionsList` in `algorithm_builder_from_options`) into
/// the [`RestoAlgorithmBuilder`], which every frontend constructs with
/// bare defaults and never options-configures (#191). Defaults match, so a
/// run that doesn't set these options is unchanged.
fn apply_outer_resto_options(rb: &mut RestoAlgorithmBuilder, ab: &AlgorithmBuilder) {
    rb.rho = ab.resto.resto_penalty_parameter;
    rb.eta_factor = ab.resto.resto_proximity_weight;
    rb.bound_mult_reset_threshold = ab.resto.bound_mult_reset_threshold;
    rb.constr_mult_reset_threshold = ab.resto.constr_mult_reset_threshold;
    rb.required_infeasibility_reduction = ab.resto.required_infeasibility_reduction;
    rb.evaluate_orig_obj_at_resto_trial = ab.resto.evaluate_orig_obj_at_resto_trial;
    rb.expect_infeasible_problem = ab.resto.expect_infeasible_problem;
    rb.start_with_resto = ab.resto.start_with_resto;
}

/// Resolve the κ_resto the restoration sub-solve's early-exit guard runs
/// with, from the user's `required_infeasibility_reduction` and whether
/// the original NLP is square.
///
/// Upstream applies the square-problem case by *overriding* the
/// sub-option: `IpRestoMinC_1Nrm.cpp:157-163` sets
/// `required_infeasibility_reduction = 0.` on `actual_resto_options`
/// when `IsSquareProblem()`, and `IpRestoConvCheck.cpp:58` then reads
/// that overridden value. So the square-problem case wins over whatever
/// the user asked for — matched here rather than, say, taking the min.
fn effective_kappa_resto(required_infeasibility_reduction: f64, is_square_problem: bool) -> f64 {
    if is_square_problem {
        0.0
    } else {
        required_infeasibility_reduction
    }
}

pub fn make_default_restoration_factory(
    resto_builder: RestoAlgorithmBuilder,
    inner_alg_builder: AlgorithmBuilder,
    backend_factory_factory: InnerBackendFactoryFactory,
) -> Box<dyn FnMut() -> Box<dyn pounce_algorithm::restoration::RestorationPhase>> {
    let mut state = Some((resto_builder, inner_alg_builder, backend_factory_factory));
    Box::new(move || {
        let (mut rb, ab, bff) = state
            .take()
            .expect("restoration factory invoked more than once");
        apply_outer_resto_options(&mut rb, &ab);
        let inner = make_resto_inner_solver(rb, ab, bff);
        let driver = crate::min_c_1nrm::MinC1NormRestoration::new().with_inner_solver(inner);
        Box::new(driver) as Box<dyn pounce_algorithm::restoration::RestorationPhase>
    })
}

/// Multi-pass companion to [`make_default_restoration_factory`].
///
/// Returns a [`pounce_algorithm::application::RestorationFactoryProvider`]:
/// a closure that mints a *fresh* one-shot restoration factory each
/// time it is invoked. `IpoptApplication` re-invokes the provider once
/// per [`pounce_algorithm::application::IpoptApplication::optimize_constrained`]
/// call (see `application.rs:1155`), which is what the ℓ₁ wrapper's
/// BNW outer loop and the `l1_fallback_on_restoration_failure` retry
/// both need — they each run the inner IPM more than once and would
/// otherwise hit the one-shot `restoration factory invoked more than once`
/// panic on the second pass.
///
/// `bff_mint` is the "factory factory factory": invoked once per
/// provider call to produce a fresh
/// [`InnerBackendFactoryFactory`] (FERAL/MA57 backend), so each inner
/// solve gets independent backend state. Callsites that capture a
/// `FeralConfig` (which is `Clone` but not `Copy`, since
/// `ScalingStrategy::External` carries a `Vec`) clone it into each
/// layer, e.g.
/// `move || { let c = feral_cfg.clone(); Box::new(move || default_backend_factory(c.clone())) }`.
pub fn make_default_restoration_factory_provider<F>(
    resto_builder: RestoAlgorithmBuilder,
    inner_alg_builder: AlgorithmBuilder,
    mut bff_mint: F,
) -> Box<dyn FnMut() -> Box<dyn FnMut() -> Box<dyn pounce_algorithm::restoration::RestorationPhase>>>
where
    F: FnMut() -> InnerBackendFactoryFactory + 'static,
{
    Box::new(move || {
        make_default_restoration_factory(
            resto_builder.clone(),
            inner_alg_builder.clone(),
            bff_mint(),
        )
    })
}

/// Single-shot inner-solve driver. Wraps the construction of the
/// nested `IpoptAlgorithm` and the extraction of the recovered
/// `(orig_x, orig_s)` from the inner-final iterate.
pub fn run_inner_resto(
    outer_data: &IpoptDataHandle,
    outer_cq: &IpoptCqHandle,
    outer_nlp: &Rc<RefCell<dyn IpoptNlp>>,
    resto_builder: &RestoAlgorithmBuilder,
    inner_alg_builder: &AlgorithmBuilder,
    backend_factory: LinearBackendFactory,
    orig_progress_cb: Option<pounce_algorithm::restoration::OrigProgressCallback>,
    print_iter_output: bool,
    debug_hook: Option<Rc<RefCell<dyn pounce_algorithm::debug::DebugHook>>>,
    intermediate_tnlp: Option<Rc<RefCell<dyn pounce_nlp::tnlp::TNLP>>>,
) -> Option<RestoSolveResult> {
    // ---- 1. Snapshot outer iterate. ---------------------------------
    let snap = build_outer_snapshot(outer_data, outer_cq)?;

    // ---- 2. Read outer dims and x_ref. ------------------------------
    let (n_orig, m_eq, m_ineq, x_ref_vals) = {
        let curr = outer_data.borrow().curr.clone()?;
        let n_orig = curr.x.dim();
        let m_eq = curr.y_c.dim();
        let m_ineq = curr.y_d.dim();
        let x_ref_vals = expanded_dense_values(&*curr.x, n_orig);
        (n_orig, m_eq, m_ineq, x_ref_vals)
    };

    // ---- 3. Build resto bundle (fresh per call). --------------------
    let mut resto_bundle = resto_builder.build(n_orig, m_eq, m_ineq, &x_ref_vals);
    resto_bundle.nlp.set_orig_nlp(Rc::clone(outer_nlp));
    resto_bundle.init.set_outer_snapshot(snap);

    // Construct the inner IPM's `IpoptData` early so we can wire it
    // into the resto NLP before sealing it inside an `Rc<RefCell<dyn
    // IpoptNlp>>`. This gives the trait-side `eval_f` /
    // `eval_grad_f` / `eval_h` calls live access to `data.curr_mu`
    // — without it `μ` is read as 0.0 (the default) and the
    // proximity term `½·η(μ)·||D_R(x − x_ref)||²` collapses to
    // zero. Mirrors upstream's `RestoIpoptNLP::ip_data_` slot,
    // which `f(x)` reads via `ip_data_->curr_mu()`
    // (`IpRestoIpoptNLP.cpp:485`).
    let inner_data: IpoptDataHandle = Rc::new(RefCell::new(IpoptData::new()));
    // Share the outer solve's wall/CPU-time deadline (pounce#242) so the
    // nested IPM is bounded by the caller's *global* budget. Without this
    // the inner solve ran effectively unbounded by wall time: it builds a
    // fresh `IpoptData` whose `timing.overall_alg` is never started, so
    // the coarse `overall_alg`-based check read 0 elapsed and never
    // tripped — a single restoration entry could then grind for many
    // iterations under one outer "iteration", the dominant source of the
    // observed budget overshoot. The shared `Deadline` measures elapsed
    // time from the outer solve's start, so the inner convergence check
    // trips at the correct global wall time.
    inner_data.borrow_mut().deadline = outer_data.borrow().deadline.clone();
    resto_bundle.nlp.set_inner_data(Rc::clone(&inner_data));

    // Wrap the resto NLP in an Rc<RefCell<dyn IpoptNlp>> for the inner
    // IPM. Move the bundle's nlp out before the bundle is partially
    // consumed below.
    let resto_nlp_rc: Rc<RefCell<dyn IpoptNlp>> = Rc::new(RefCell::new(resto_bundle.nlp));

    // Snapshot the outer-curr orig-NLP `inf_pr` so the inner conv
    // check can run upstream's kappa-reduction early-exit guard
    // (`IpRestoConvCheck.cpp:175`) against a fixed reference.
    let orig_curr_inf_pr = outer_cq.borrow().curr_primal_infeasibility_max();

    // Square-problem kappa override — mirrors upstream
    // `IpRestoMinC_1Nrm.cpp:157-163`: when `IsSquareProblem()` is true
    // (`x.dim() == y_c.dim()`), upstream sets
    // `required_infeasibility_reduction = 0` on the resto sub-options,
    // which the inner conv check (`IpRestoConvCheck.cpp:163`) reads as
    // "the kappa-reduction guard is disabled — keep iterating until the
    // sub-NLP is fully converged". Without this, pounce's resto inner
    // exits on PFIT3/PFIT4 after only a 10% feasibility reduction
    // (kappa_resto=0.9), the outer Newton step from the partially-
    // recovered iterate blows up, and we re-enter resto in a loop.
    //
    // The non-square value is the user's `required_infeasibility_reduction`
    // (#439); it was hardcoded to upstream's 0.9 default here, so setting
    // the registered option did nothing.
    let is_square_problem = n_orig == m_eq;
    let kappa_resto = effective_kappa_resto(
        resto_builder.required_infeasibility_reduction,
        is_square_problem,
    );

    // ---- 4. Build the inner alg bundle and override its init /
    //         conv_check / iter_output slots with resto-side ones. ----
    // The nonlinear-variable mask (gh#624) names positions in the
    // *original* NLP's `x_var` space. The restoration sub-NLP's primal is
    // the 5-block `[orig | n_c | p_c | n_d | p_d]` compound, where those
    // indices mean something else entirely, so the mask never crosses
    // into the inner solve: restoration approximates over its whole
    // space, exactly as it did before the mask existed.
    let mut inner_alg_builder = inner_alg_builder.clone();
    inner_alg_builder.limited_memory_nonlinear_vars = None;
    let inner_alg_builder = &inner_alg_builder;

    let mut alg_bundle = inner_alg_builder.build_with_backend(backend_factory);

    // Wrap the inner `StdAugSystemSolver` with `AugRestoSystemSolver`,
    // which performs the 8-block → 4-block Schur reduction over the
    // four slack pairs (n_c, p_c, n_d, p_d) before delegating to the
    // inner solver. Mirrors upstream `IpAugRestoSystemSolver`
    // (`IpAlgBuilder.cpp::BuildRestoIpoptAlgorithm`).
    // Architectural port toggle: when enabled, wraps the inner
    // `StdAugSystemSolver` with `AugRestoSystemSolver` (Schur-reduction
    // path). Currently disabled while debugging the orig-step
    // computation regression.
    if let Some(search_dir) = alg_bundle.search_dir.as_mut() {
        search_dir.pd_solver_mut().wrap_aug_solver(|inner| {
            Box::new(crate::aug_resto_system_solver::AugRestoSystemSolver::new(
                inner,
            ))
        });
    }
    alg_bundle.init =
        Box::new(resto_bundle.init) as Box<dyn pounce_algorithm::init::r#trait::IterateInitializer>;
    // Thread the user's outer `tol`/`acceptable_tol`/`acceptable_iter`
    // (carried on the inner builder's conv_check options) into the resto
    // sub-solve instead of hardcoded `(1e-8, 1e-6, 15)`. See
    // `build_resto_conv_check_adapter`.
    let outer_tol = outer_data.borrow().tol;
    // The user's `constr_viol_tol`, carried on the same inner conv-check
    // options (upstream reads it with the *original* prefix). It is the
    // violation floor for all five locally-infeasible gates below, scaled by
    // the violated row's own magnitude (`is_significant` — see
    // `pounce_common::tolerance` for why the rejecting direction must be
    // purely relative).
    //
    // That floor used to be `100·outer_tol`: a KKT-error tolerance standing in
    // for a constraint-violation one (gh #508). Sweeping `constr_viol_tol`
    // could not move any of these verdicts, and loosening `tol` — the standard
    // reaction to a struggling solve — raised the floor a hundredfold and
    // withdrew the diagnosis. On `min (x-5)² s.t. x²+δ = 0` at `tol=1e-3` that
    // turned a model infeasible by a full percent into `Restoration_Failed`
    // (AMPL 500, Pyomo `internalSolverError`) while the same model at the
    // default `tol` was correctly diagnosed. Same defect, and the same repair,
    // as the outer cycle exit in `ipopt_alg`.
    let outer_constr_viol_tol = inner_alg_builder.conv_check.constr_viol_tol;
    let mut adapter = build_resto_conv_check_adapter(&inner_alg_builder.conv_check)
        .with_orig_progress_guard(Rc::clone(outer_nlp), orig_curr_inf_pr, kappa_resto)
        // Layer 2 of `IpRestoConvCheck::CheckConvergence` (pounce#438).
        // `constr_viol_tol` is read with the *original* prefix upstream
        // (`IpRestoConvCheck::InitializeImpl`), which is exactly what the
        // inner builder's conv-check options carry here.
        .with_orig_convergence_verdict(
            outer_tol,
            inner_alg_builder.conv_check.constr_viol_tol,
            is_square_problem,
        );
    if let Some(cb) = orig_progress_cb {
        adapter = adapter.with_orig_progress_callback(cb);
    }
    alg_bundle.conv_check =
        Box::new(adapter) as Box<dyn pounce_algorithm::conv_check::r#trait::ConvCheck>;
    alg_bundle.iter_output = Box::new(
        crate::output::RestoIterationOutputAdapter::new().with_orig_nlp(Rc::clone(outer_nlp)),
    ) as Box<dyn pounce_algorithm::output::r#trait::IterationOutput>;

    // Mirror upstream `IpRestoMinC_1Nrm.cpp:91`: set the resto sub-IPM's
    // `theta_max_fact = 1e8` (vs the regular-phase default 1e4). Without
    // this, the inner filter acceptor caps `theta_max = 1e4` on its first
    // line search (resto θ ≈ 0 after slack-init, so
    // `theta_max = 1e4·max(1, 0) = 1e4`); the first non-trivial trial then
    // gets rejected at the `theta_max` gate before reaching f-type/Armijo
    // dispatch — qcqp750-2nc iter 2r α=2e-3 fails this way with
    // θ_trial = 1.5e7 > 1e4, forcing backtracking to α≈3e-5. pounce#21.
    alg_bundle
        .line_search
        .acceptor_mut()
        .set_theta_max_fact(1e8);
    // ...and opt the inner IPM out of pounce's row-count floor on the
    // same reference (`theta_max_row_scale_kappa`, pounce#476). That
    // floor is off by default, so this is unconditional insurance
    // against a user who turns it on globally. It exists to fix exactly
    // the degeneracy described above for the *outer* phase, where
    // upstream left the 1e4 default in place on models that also start
    // feasible. Here upstream already corrected it, and stacking the
    // floor on top would make the inner ceiling `1e8 · m` — no ceiling
    // at all. Keep the resto sub-IPM bit-for-bit upstream at any kappa.
    alg_bundle
        .line_search
        .acceptor_mut()
        .set_theta_max_row_scale_kappa(0.0);
    // ...and out of the adaptive ceiling (`theta_max_adaptive_trigger`,
    // pounce#546) for the same reason. That rule raises `theta_max` when
    // the gate is demonstrably what is refusing the line search, which
    // is the right question for the outer phase. Here the ceiling is
    // already `1e8` by upstream's own hand, so a rule that ratchets it
    // further would be compounding a correction that has already been
    // made. The resto sub-IPM stays bit-for-bit upstream.
    alg_bundle
        .line_search
        .acceptor_mut()
        .set_theta_max_adaptive_trigger(0);

    // Replace the inner-bundle mu update with a resto-configured fresh
    // copy. Upstream `IpAlgBuilder.cpp:929` looks up
    // `options.GetStringValue("mu_strategy", _, "resto." + prefix)` and
    // falls back to the outer `mu_strategy` when no `resto.mu_strategy`
    // override is set — so the inner IPM inherits the outer's adaptive
    // vs. monotone choice. We mirror that by branching on the inner
    // alg builder's `mu_strategy`, which the caller populates from the
    // same `OptionsList` the outer builder reads. The hardcoded
    // monotone path that lived here previously diverged from upstream:
    // when the outer is adaptive and μ has blown up to ~1e6 before
    // entering restoration (ex8_3_10), monotone can only shrink μ by
    // κ_μ per iter and exhausts the resto iter budget before recovery
    // completes; the adaptive path's QF oracle resets μ to ~1.0 in one
    // step.
    //
    // Conservative `mu_min` floor: upstream
    // `IpAdaptiveMuUpdate.cpp:206-211` applies `100 * mu_min` for the
    // restoration phase. Without it, a near-feasible inner iterate
    // (theta ≈ 1e-13) collapses μ to the absolute floor (1e-11) in a
    // single step. With μ at the floor the next direction is dominated
    // by the ρ‖p+n‖₁ penalty and proximity terms instead of the
    // barrier, and the resulting trial blows the orig-NLP infeasibility
    // back up several orders of magnitude — kappa-reduction guard then
    // can never re-fire and the inner runs out of iters
    // (DECONVBNE: 479 iter Restoration_Failed → upstream's resto.mu_min
    // = 1e-9 lets it converge in 484 outer iters). Applied to both
    // branches.
    let outer_mu_min = inner_alg_builder.mu.mu_min;
    let resto_mu_min = 100.0 * outer_mu_min;
    alg_bundle.mu_update = match inner_alg_builder.mu_strategy {
        pounce_algorithm::alg_builder::MuStrategyChoice::Monotone => {
            let mut monotone = MonotoneMuUpdate::new()
                .with_first_iter_resto(true)
                .with_mu_min(resto_mu_min);
            // `tau_min` governs the fraction-to-the-boundary rule, which
            // the restoration IPM applies to its own steps too; upstream
            // reads the same option in the resto sub-algorithm (#551 /
            // #677). Default 0.99 either way.
            monotone.tau_min = inner_alg_builder.mu.tau_min;
            Box::new(monotone) as Box<dyn pounce_algorithm::mu::r#trait::MuUpdate>
        }
        pounce_algorithm::alg_builder::MuStrategyChoice::Adaptive => {
            let mut adaptive = pounce_algorithm::mu::adaptive::AdaptiveMuUpdate::new();
            adaptive.mu_oracle = inner_alg_builder.mu_oracle;
            adaptive.mu_init = inner_alg_builder.mu.mu_init;
            adaptive.mu_max = inner_alg_builder.mu.mu_max;
            adaptive.mu_max_fact = inner_alg_builder.mu.mu_max_fact;
            adaptive.mu_min = resto_mu_min;
            adaptive.mu_linear_decrease_factor = inner_alg_builder.mu.mu_linear_decrease_factor;
            adaptive.mu_superlinear_decrease_power =
                inner_alg_builder.mu.mu_superlinear_decrease_power;
            adaptive.barrier_tol_factor = inner_alg_builder.mu.barrier_tol_factor;
            adaptive.tau_min = inner_alg_builder.mu.tau_min;
            adaptive.sigma_min = inner_alg_builder.mu.sigma_min;
            adaptive.sigma_max = inner_alg_builder.mu.sigma_max;
            adaptive.adaptive_mu_globalization = inner_alg_builder.mu.adaptive_mu_globalization;
            Box::new(adaptive) as Box<dyn pounce_algorithm::mu::r#trait::MuUpdate>
        }
    };

    // ---- 5. Construct inner cq (inner_data already built above). ----
    let inner_cq: IpoptCqHandle = Rc::new(RefCell::new(IpoptCalculatedQuantities::new(
        Rc::clone(&inner_data),
        Rc::clone(&resto_nlp_rc),
    )));

    // Seed inner iter_count = outer.iter_count + 1 to mirror upstream
    // `IpRestoMinC_1Nrm.cpp:181`. The outer transcription block in
    // `min_c_1nrm.rs` uses `result.iter_count - 1` to roll the outer
    // counter forward by `inner_iter_count - outer_iter_count - 1`
    // total iterations spent in restoration; that arithmetic only
    // matches upstream when the inner counter is seeded from the outer.
    //
    // Also propagate the outer's info_* fields onto the inner data so
    // the inner's first OutputIteration row prints the failed-α / 'R'
    // char / ls_count from the outer line search. Mirrors
    // `IpRestoMinC_1Nrm.cpp:182-188`:
    //   resto_ip_data->Set_info_regu_x(IpData().info_regu_x());
    //   resto_ip_data->Set_info_alpha_primal(IpData().info_alpha_primal());
    //   resto_ip_data->Set_info_alpha_primal_char(IpData().info_alpha_primal_char());
    //   resto_ip_data->Set_info_alpha_dual(IpData().info_alpha_dual());
    //   resto_ip_data->Set_info_ls_count(IpData().info_ls_count());
    //   resto_ip_data->Set_info_iters_since_header(IpData().info_iters_since_header());
    //   resto_ip_data->Set_info_last_output(IpData().info_last_output());
    {
        let (
            outer_iter,
            outer_regu_x,
            outer_alpha_primal,
            outer_alpha_primal_char,
            outer_alpha_dual,
            outer_ls_count,
            outer_iters_since_header,
            outer_last_output,
        ) = {
            let d = outer_data.borrow();
            (
                d.iter_count,
                d.info_regu_x,
                d.info_alpha_primal,
                d.info_alpha_primal_char,
                d.info_alpha_dual,
                d.info_ls_count,
                d.info_iters_since_header,
                d.info_last_output,
            )
        };
        let mut inner = inner_data.borrow_mut();
        inner.iter_count = outer_iter + 1;
        inner.info_regu_x = outer_regu_x;
        inner.info_alpha_primal = outer_alpha_primal;
        inner.info_alpha_primal_char = outer_alpha_primal_char;
        inner.info_alpha_dual = outer_alpha_dual;
        inner.info_ls_count = outer_ls_count;
        inner.info_iters_since_header = outer_iters_since_header;
        inner.info_last_output = outer_last_output;
    }

    // Seed `inner_data.curr` with a placeholder iterate matching the
    // resto NLP's compound shape — the init overwrites it on iter 0,
    // but the IteratesVector slot must be `Some` so subsequent
    // accessors don't trip an `expect`.
    inner_data
        .borrow_mut()
        .set_curr(make_placeholder_resto_iv(n_orig, m_eq, m_ineq));

    // ---- 6. Run the nested IPM. -------------------------------------
    //
    // The inner IPM gets its own restoration phase (resto-of-resto):
    // when the inner line search itself fails, upstream's
    // `RestoRestorationPhase` resets the n/p slack feasibility variables
    // in closed form (holding the `x_orig` block and `s` fixed) so the
    // inner can keep iterating. Without this, any inner line-search
    // failure terminates the outer with `RestorationFailure`.
    let resto_of_resto: Box<dyn pounce_algorithm::restoration::RestorationPhase> = Box::new(
        crate::resto_resto::RestoRestorationPhase::new(resto_builder.rho)
            .with_orig_nlp(Rc::clone(outer_nlp)),
    );
    let mut alg = IpoptAlgorithm::new(inner_data, inner_cq, alg_bundle)
        .with_nlp(Rc::clone(&resto_nlp_rc))
        .with_restoration(resto_of_resto);
    // Forward the shared debugger so the same session can step the inner
    // restoration solve (its DebugCtx exposes the resto sub-NLP iterate).
    if let Some(h) = debug_hook {
        alg = alg.with_debug_hook(h);
    }
    // Forward the user's TNLP so `intermediate_callback` fires per inner
    // iteration (gh#645), flagged so those fires carry
    // `AlgorithmMode::RestorationPhaseMode` and skip the live-inspector
    // context — the inner iterate is a compound `(x_orig, n, p)` vector,
    // not a point of the user's NLP. Left unset when the caller
    // installed no callback, so nothing about this path is reachable for
    // them.
    if let Some(t) = intermediate_tnlp {
        alg = alg.with_tnlp(t);
        alg.fires_as_restoration = true;
    }
    // Forward the outer `print_level == 0` gate. Suppresses the
    // restoration `r`-suffixed iter table; the resto-of-resto level
    // also inherits the same flag (its `RestorationPhase` impl is the
    // closed-form `RestoRestorationPhase`, which doesn't print).
    alg.print_iter_output = print_iter_output;
    // The gh #534 deferral is an outer-loop change and stays one. Inside the
    // inner solve "acceptable" is a statement about the *restoration* NLP, whose
    // acceptable-level exit feeds the status mapping below rather than a user's
    // answer, and nothing here was measured in that regime. Off, so the inner
    // solve behaves exactly as it did pre-#534.
    alg.resto_decline_deferrals = 0;
    let status = alg.optimize();

    // ---- 7. Map status & extract orig_x/orig_s. ---------------------
    //
    // We need to recover trial_x / trial_s on BOTH the success path
    // (regular RestoSolveResult return) and the alt-locally-infeasible
    // path (inner exited RestorationFailure / MaxiterExceeded but the
    // resto NLP itself reached stationarity at a point of large
    // orig-NLP `inf_pr`). Hoist the extraction so it runs before the
    // status branch.
    let final_iv = alg.data.borrow().curr.clone()?;
    let xc = final_iv.x.as_any().downcast_ref::<CompoundVector>()?;
    let trial_x = clone_dense_block(xc.comp(BLOCK_X))?;
    let trial_s = clone_to_dense(&*final_iv.s);

    let (inner_iter_count, iters_since_header, last_output) = {
        let d = alg.data.borrow();
        (d.iter_count, d.info_iters_since_header, d.info_last_output)
    };

    // gh#645: the user's callback returned `false` from a restoration
    // fire. Return before the locally-infeasible adjudication below
    // rather than after it: that verdict is a claim about the original
    // NLP's feasibility, and a sub-solve the user interrupted has not
    // finished earning it. The caller maps this to
    // `RestorationOutcome::UserRequestedStop` and drops `trial_x` /
    // `trial_s` on the floor — they are carried here only because the
    // struct is shared with the success path.
    if matches!(status, SolverReturn::UserRequestedStop) {
        return Some(RestoSolveResult {
            trial_x,
            trial_s,
            iter_count: inner_iter_count,
            iters_since_header,
            last_output,
            locally_infeasible: false,
            user_requested_stop: true,
        });
    }

    // Locally-infeasible detection. Mirrors upstream
    // `IpRestoConvCheck.cpp:208-241`: fires when the inner sub-IPM
    // converged via its OWN KKT residual (stationarity of the resto
    // NLP, not via the kappa-reduction early-exit) and the orig-NLP
    // `inf_pr` at the converged iterate is still well above outer
    // `tol`. This is the algorithmic signature of a local
    // infeasibility — the resto sub-problem has driven `||c||_1` to
    // a local minimum that's bounded away from zero.
    //
    // Distinguishing the two `Success` paths matters: when the inner
    // returns via the kappa guard (orig_inf_pr reduced sufficiently),
    // its own KKT residual at termination is whatever happens to be
    // — typically large because we exited early. When the inner
    // returns via stationarity, its KKT residual is tight (≤ inner
    // `tol`). Without this gate, we'd misclassify any kappa-guard
    // exit at exactly the entry `inf_pr` as locally-infeasible
    // (HATFLDF, POLAK6, ROSENMMX, ... regress).
    let (orig_inf_pr_at_final, orig_inf_pr_scaled) =
        eval_orig_inf_pr_at_inner_curr(&*final_iv.x, &*final_iv.s, outer_nlp).unwrap_or((0.0, 0.0));
    // Row magnitude implied by the two measures of the same violation:
    // `orig_inf_pr_scaled = dc * orig_inf_pr_at_final` for the worst row, so
    // their ratio is that row's `1/dc` — its natural magnitude. Using the
    // solver's own scaling rather than a second notion of scale keeps one
    // authority, and it is available here without plumbing the vectors through.
    let violation_scale = if orig_inf_pr_scaled.is_finite()
        && orig_inf_pr_scaled > 0.0
        && orig_inf_pr_at_final.is_finite()
    {
        orig_inf_pr_at_final / orig_inf_pr_scaled
    } else {
        1.0
    };
    let inner_kkt_err = alg.cq.borrow().curr_nlp_error();
    let inner_stationarity_converged = inner_kkt_err <= 10.0 * outer_tol;
    // Square problems: upstream `IpRestoMinC_1Nrm.cpp:357-371` returns
    // the recovered point to the outer unconditionally when the inner
    // succeeds — even if `constr_viol > constr_viol_tol_`. The outer
    // gets another shot at making progress (PFIT4 trace: 190 iters with
    // theta oscillating from 3.77e7 down to 3.42e-11). Pounce previously
    // declared `strict_locally_infeasible` when the inner converged on
    // an infeasible stationary point, which on PFIT3/PFIT4 short-
    // circuited the outer's recovery path. The outer's
    // `resto_no_outer_progress_count` cycle detector (5 consecutive
    // null-progress entries) bounds the worst case if the outer truly
    // can't escape; the cycle exit now surfaces `LocalInfeasibility`
    // when the outer cv at re-entry is at or above `constr_viol_tol`
    // (gh #508 — "is this violation real" is a question about the
    // constraint violation, so it is asked with the constraint-violation
    // tolerance, not with `tol`) and `ErrorInStepComputation` otherwise.
    let strict_locally_infeasible = !is_square_problem
        && matches!(
            status,
            SolverReturn::Success | SolverReturn::StopAtAcceptablePoint
        )
        && inner_stationarity_converged
        && is_significant(orig_inf_pr_at_final, violation_scale, outer_constr_viol_tol);
    // Alt locally-infeasible gate. PFIT2/PFIT3-style: the inner
    // resto NLP is at (or near) a stationary point — `inner_kkt_err`
    // has dropped to a small value — but the inner's own line search
    // can't make the next step (degenerate Hessian / nested
    // resto-of-resto trips), so the inner exits with
    // `RestorationFailure` or `MaxiterExceeded` instead of `Success`.
    // Algorithmically this is the same locally-infeasible signature:
    // the resto sub-problem has driven `||c||_1` as low as the
    // sub-NLP can, the value is bounded above outer `tol`, and the
    // KKT residual is well into the "approaching stationary" regime.
    //
    // Heuristic thresholds:
    //
    //   * `inner_kkt_err <= 1e-2` — loose enough to admit the
    //     PFIT2-style exit (inf_du ≈ 1e-3 in the trace, full nlp_err
    //     of similar magnitude after compl/scaling), tight enough to
    //     reject genuinely-stuck inners that haven't approached
    //     stationarity at all.
    //   * the shared `constr_viol_tol` violation floor — the orig-NLP
    //     `inf_pr` is a violation the user calls a violation (i.e. NOT just a
    //     little above zero — distinguish from the kappa-guard near-feasible
    //     exit).
    //   * `inner_iter_count >= 30` — not a premature failure on the
    //     first few inner iters.
    //
    // Mirrors the spirit of upstream's exception-throw at
    // `IpRestoConvCheck.cpp:240` for the case where the inner happens
    // to exit via line-search failure rather than by clean
    // convergence — upstream avoids this by being more numerically
    // robust in the line search itself (365+ inner iters on PFIT2),
    // pounce currently can't reach that depth so we surface the
    // diagnosis on the failure exit instead.
    let alt_locally_infeasible = matches!(
        status,
        SolverReturn::RestorationFailure
            | SolverReturn::MaxiterExceeded
            | SolverReturn::ErrorInStepComputation
    ) && inner_kkt_err <= 1e-2
        && is_significant(orig_inf_pr_at_final, violation_scale, outer_constr_viol_tol)
        && inner_iter_count >= 30;

    // Cycle locally-infeasible gate (CRESC100-style). The inner has run
    // a very large number of iterations and exited via MaxiterExceeded
    // with orig-NLP `inf_pr` bounded well above outer tol — same
    // user-facing diagnosis (problem is locally infeasible) as the
    // strict / alt gates, but the inner's own KKT residual is still
    // huge because the inner's line search is cycling between basins
    // rather than approaching a stationary point. Upstream solves
    // these via its more robust MUMPS / MA57 backend; for the FERAL
    // backend we surface `LocallyInfeasible` rather than the misleading
    // `Restoration_Failed` once a generous iteration budget has been
    // burned with no exit. Conservative threshold to avoid
    // misclassifying genuinely under-resourced solves.
    let cycle_locally_infeasible = matches!(status, SolverReturn::MaxiterExceeded)
        && inner_iter_count >= RESTO_STALL_EVIDENCE_ITERS
        && is_significant(orig_inf_pr_at_final, violation_scale, outer_constr_viol_tol)
        && orig_inf_pr_at_final.is_finite();

    // Step-failure locally-infeasible gate (qcqp750-2nc-style). The
    // inner ran for a non-trivial number of iterations, the orig-NLP
    // `inf_pr` plateau'd at a finite value well above outer tol, and
    // then the inner step computation diverged (||d|| explodes, line
    // search collapses to alpha ≈ 1e-12). `data.curr` at termination
    // is the last accepted iterate (the pre-explosion plateau), so
    // `trial_x`/`trial_s` extracted above are usable — only
    // `inner_kkt_err` is poisoned by the explosion, which is why the
    // `alt` gate's `<= 1e-2` threshold rejects this signature.
    // Upstream resto's more robust inertia controller avoids the
    // explosion; for pounce we surface `LocallyInfeasible` with the
    // recovered pre-explosion point rather than the misleading
    // `Restoration_Failed`. The `iter >= 30` floor matches the `alt`
    // gate's "not a premature failure".
    let step_failure_locally_infeasible = matches!(status, SolverReturn::ErrorInStepComputation)
        && inner_iter_count >= 30
        && is_significant(orig_inf_pr_at_final, violation_scale, outer_constr_viol_tol)
        && orig_inf_pr_at_final.is_finite();

    // Tiny-step locally-infeasible gate (gh #372). At a tight user `tol`
    // the inner sub-IPM can drive the resto NLP to its stationary point
    // and then be unable to certify it against its own (equally tight)
    // convergence test — every remaining step is below the tiny-step
    // threshold, so it exits `StopAtTinyStep` instead of `Success`. The
    // reproducer is `0 <= x <= 0.6, x >= 0.7` at `tol=1e-10`: the inner
    // stops after 13 iters with `inner_kkt_err = 1.2e-10` and
    // `orig_inf_pr = 1.0e-1`, and every gate above rejects the exit
    // status, so a one-inequality contradiction landed in
    // `Restoration_Failed` (AMPL 500, Pyomo `internalSolverError`) while
    // the identical model at the default `tol=1e-8` exits `Success` and
    // is correctly diagnosed. A step that has shrunk to machine
    // precision IS the numerical stationarity evidence, so unlike
    // `strict` this gate does not additionally demand `inner_kkt_err <=
    // 10*outer_tol` (a threshold the inner cannot reach at tight `tol`,
    // which is exactly how the case arises); it uses the `alt` gate's
    // looser `1e-2` KKT ceiling to reject inners that stalled without
    // ever approaching stationarity. The `constr_viol_tol` violation floor
    // mirrors `strict`.
    //
    // Unlike `strict`, this gate is NOT carved out for square problems
    // (pounce#512). The two carve-outs looked alike but rest on different
    // upstream branches. `strict`'s case is `resto_status == SUCCESS`
    // (`IpRestoMinC_1Nrm.cpp:249-268`), which sets `retval = 0` — return
    // the recovered point, render no verdict — so declaring infeasibility
    // there is pounce-specific and rightly withheld on square problems.
    // A tiny-step exit is upstream's *other* branch, `:278-291`:
    //
    //     else if( resto_status == STOP_AT_TINY_STEP || ... )
    //        ... THROW_EXCEPTION(LOCALLY_INFEASIBLE, ...)
    //
    // which throws on any problem shape; the only square-specific test
    // upstream has in this region (`:269`) is the *feasible* case, a
    // `STOP_AT_ACCEPTABLE_POINT` below `constr_viol_tol`, and the
    // violation floor below already excludes that. Carrying the exclusion
    // here cost the #508 probe model — square, and infeasible by a full
    // percent — its `LocalInfeasibility` verdict whenever the inner
    // reached its stationary point by tiny step rather than by a
    // certified convergence, which is what `tol=1e-12` produces: the
    // honest AMPL 200 degraded to `Restoration_Failed`, AMPL 500.
    let tiny_step_locally_infeasible = matches!(status, SolverReturn::StopAtTinyStep)
        && inner_kkt_err <= 1e-2
        && is_significant(orig_inf_pr_at_final, violation_scale, outer_constr_viol_tol)
        && orig_inf_pr_at_final.is_finite();

    // Layer-2 verdict, rendered from *inside* the sub-solve (pounce#438).
    // The inner convergence check asked, at the moment the restoration
    // sub-problem reached its own KKT point, whether the recovered point is
    // feasible for the ORIGINAL NLP, and answered no — upstream's
    // `LOCALLY_INFEASIBLE` throw at `IpRestoConvCheck.cpp:240`, which
    // `IpoptAlgorithm` turns into `LOCAL_INFEASIBILITY`.
    //
    // Unlike every gate above, this one needs no signature heuristics: the
    // verdict was issued at a point the sub-solve itself certified, against
    // the sub-solve's own (possibly tightened) tolerance, rather than
    // reconstructed after the fact from a terminal status and a KKT residual.
    // The gates above remain because they cover the exits where the inner
    // never gets to render a verdict at all — line-search failure, step
    // explosion, tiny step, iteration cap.
    let verdict_locally_infeasible = matches!(status, SolverReturn::LocalInfeasibility);

    // Admissibility guard over ALL of the gates above, in one place.
    //
    // Never claim infeasibility at a point the solver's own convergence test
    // would accept as feasible. `orig_inf_pr_at_final` is measured *unscaled*
    // (correctly — the floors it is compared against are absolute, user-facing
    // magnitudes), but `inner_kkt_err` and the convergence check are in the
    // *scaled* space. Comparing across the two is the same units mismatch the
    // unscaling was introduced to fix, just moved: on a model whose rows are
    // scaled down, the inner can stop at a point it considers feasible
    // (scaled violation below `tol`) while the unscaled violation still clears
    // a `1e-4` floor, and a gate then reports a feasible model infeasible.
    //
    // Measured: two feasible instances (property-test seeds 99 and 193) turned
    // from `Solve_Succeeded` into `Infeasible_Problem_Detected` exactly this
    // way. Seed 99's numbers make the mechanism plain — `inner_kkt_err`
    // 3.249625e-9 against `orig_inf_pr` 3.249625e-3, the same mantissa scaled
    // by 1e6, the row scaling factor.
    //
    // The guard costs nothing on models that carry no row scaling (the two
    // measures coincide) and nothing on genuinely infeasible ones, whose scaled
    // violation is comfortably above `tol` — `infeasible_equalities.nl`, the
    // model the unscaling was introduced for, sits at 6.7e-7 against a `1e-8`
    // tolerance and still certifies.
    //
    // Applied once to the combination rather than to each disjunct: the two
    // preceding safeguards in this area were each added to one path and not its
    // twin, and a hole survived both times.
    let solver_would_call_it_feasible = orig_inf_pr_scaled <= outer_tol;

    // Divergence guard over the five *reconstructed* gates (gh#661).
    //
    // Each of those gates reconstructs "the sub-solve stalled at a point it
    // could not improve on" after the fact, from a terminal status plus a
    // KKT residual — and every one of them then tests only that the residual
    // violation is *large* (`is_significant`). Large is not the same claim as
    // stalled. A restoration that is actively *diverging* satisfies the size
    // test more emphatically the worse it gets, so the size test alone reads
    // a blow-up as strong evidence for the verdict it most contradicts.
    //
    // `orig_curr_inf_pr` is the original-NLP violation the outer handed to
    // this restoration call, in the same scaled space as `orig_inf_pr_scaled`
    // (it is the reference the kappa-reduction guard measures progress
    // against). Restoration exists to *reduce* that number; the kappa guard
    // will not even call the sub-solve converged until it reaches
    // `kappa_resto` (0.9) of it. A point an order of magnitude *worse* than
    // where restoration started is not a point restoration converged to, and
    // "converged to a point of local infeasibility" — an affirmative claim
    // about the user's model — has no basis there. `Restoration_Failed`, the
    // solver admitting it failed, is the honest answer.
    //
    // Measured on `pooling_rt2stp.nl` under `mehrotra_algorithm=yes`: the
    // `step_failure` gate fired at `entry_inf_pr = 6.957e0`,
    // `orig_inf_pr = 7.355e5` — restoration made feasibility 105,700x worse,
    // with the inner's `inf_pr` climbing monotonically (2.61e2, 5.67e3,
    // 6.24e4, 7.35e5) and `||d||` reaching 1.64e9, and the model was reported
    // infeasible. It is not; at default options the same model solves. The
    // same run on #619's parent commit differed only in that its inner
    // exploded at iteration 19 rather than 32, missing the gate's `iter >= 30`
    // floor — identical divergence, opposite verdict, decided by an iteration
    // count. #619 did not introduce the defect, only a starting point that
    // reaches it.
    //
    // The waiver, the headroom's looseness, and why the layer-2 verdict is
    // exempt are all documented on `diverged_from_restoration_entry`. The
    // short of it: `hs71_obj1e8` and `pooling_rt2stp` are feasible models
    // whose outer solves ran 25 and 20 iterations and never demonstrated a
    // floor at all (1 and 7 iterations at one), while
    // `issue_508_infeasible_gap_1em2` — which reaches a *correct* verdict —
    // spent 943 outer iterations unable to get below the `1.0e-2` gap it is
    // built around.
    //
    // Read off the *outer* `IpoptData`, which is where that trajectory
    // lives: this sub-solve ran four iterations.
    let outer_iters_at_floor = outer_data.borrow().inf_pr_floor.iters_at_floor();
    let diverged_from_entry =
        diverged_from_restoration_entry(orig_curr_inf_pr, orig_inf_pr_scaled, outer_iters_at_floor);

    let reconstructed_locally_infeasible = !diverged_from_entry
        && (strict_locally_infeasible
            || alt_locally_infeasible
            || cycle_locally_infeasible
            || step_failure_locally_infeasible
            || tiny_step_locally_infeasible);

    let locally_infeasible = !solver_would_call_it_feasible
        && (verdict_locally_infeasible || reconstructed_locally_infeasible);

    if std::env::var_os("POUNCE_DBG_RESTO_LOCINF").is_some() {
        tracing::debug!(target: "pounce::restoration",
            "[PN_RESTO_LOCINF] status={:?} iter={} inner_kkt_err={:.6e} orig_inf_pr={:.6e} orig_inf_pr_scaled={:.6e} entry_inf_pr={:.6e} outer_tol={:.6e} verdict={} strict={} alt={} cycle={} step_fail={} tiny_step={} diverged_from_entry={} at_floor={} floor={:.6e} → loc_inf={}",
            status,
            inner_iter_count,
            inner_kkt_err,
            orig_inf_pr_at_final,
            orig_inf_pr_scaled,
            orig_curr_inf_pr,
            outer_tol,
            verdict_locally_infeasible,
            strict_locally_infeasible,
            alt_locally_infeasible,
            cycle_locally_infeasible,
            step_failure_locally_infeasible,
            tiny_step_locally_infeasible,
            diverged_from_entry,
            outer_iters_at_floor,
            outer_data.borrow().inf_pr_floor.floor(),
            locally_infeasible
        );
    }

    // If the inner failed AND we did NOT detect locally-infeasible,
    // fall back to the original Failed path (caller turns this into
    // `RestorationOutcome::Failed`).
    if !is_resto_success(status) && !locally_infeasible {
        return None;
    }

    Some(RestoSolveResult {
        trial_x,
        trial_s,
        iter_count: inner_iter_count,
        // Inner-IPM info_iters_since_header / info_last_output are
        // tracked on the inner data; surface them on best-effort
        // (these only drive header/print spacing on the next outer
        // iteration row).
        iters_since_header,
        last_output,
        locally_infeasible,
        user_requested_stop: false,
    })
}

/// Evaluate `max(||c(x_orig)||∞, ||d(x_orig) − s||∞)` at the inner
/// IPM's converged iterate, in **unscaled** (user) units. Returns `None` on
/// any downcast / dim mismatch (caller treats as `0.0` and the
/// locally-infeasible gate fails closed — i.e. we don't spuriously declare
/// infeasibility on a fixture we can't evaluate).
///
/// The unscaling is load-bearing, not cosmetic. `eval_c` / `eval_d` return the
/// row-scaled residual (`c_scaled = dc ⊙ c_user`), while every gate above
/// compares this value against `constr_viol_tol`, a user-facing magnitude
/// meaning "the constraint violation is meaningfully nonzero", so feeding it a
/// scaled residual would compare two different unit systems.
///
/// On a problem whose constraint scaling is small the mismatch silently
/// disables the gates. `infeasible_equalities.nl` is the worked example: NLP
/// scaling shrinks the rows by ~3e6, so a true violation of `2.0` reads as
/// `6.67e-7` scaled and can never clear a `1e-4` floor. The `strict` gate then
/// fails on that one term alone (its KKT condition passes), the diagnosis is
/// discarded, and a blatantly infeasible model exits `Error_In_Step_Computation`
/// — the AMPL 500 failure range, Pyomo `internalSolverError`. Same user-visible
/// family as gh #372, opposite trigger: this one bites at *loose* `tol`, where
/// the surviving detector that masked it at `tol <= 1e-7` no longer fires.
fn eval_orig_inf_pr_at_inner_curr(
    inner_x: &dyn Vector,
    inner_s: &dyn Vector,
    orig_rc: &Rc<RefCell<dyn IpoptNlp>>,
) -> Option<(f64, f64)> {
    let xc = inner_x.as_any().downcast_ref::<CompoundVector>()?;
    let x_orig = xc.comp(BLOCK_X);
    let mut orig = orig_rc.borrow_mut();
    let m_eq = orig.m_eq();
    let m_ineq = orig.m_ineq();
    let (dc, dd) = (orig.c_scale_vec(), orig.d_scale_vec());
    let c_amax = if m_eq > 0 {
        let mut buf = DenseVectorSpace::new(m_eq).make_new_dense();
        orig.eval_c(x_orig, &mut buf);
        (unscaled_block_amax(&buf, dc.as_deref()), buf.amax())
    } else {
        (0.0, 0.0)
    };
    let d_minus_s_amax = if m_ineq > 0 {
        let mut buf = DenseVectorSpace::new(m_ineq).make_new_dense();
        orig.eval_d(x_orig, &mut buf);
        buf.axpy(-1.0, inner_s);
        (unscaled_block_amax(&buf, dd.as_deref()), buf.amax())
    } else {
        (0.0, 0.0)
    };
    Some((
        c_amax.0.max(d_minus_s_amax.0),
        c_amax.1.max(d_minus_s_amax.1),
    ))
}

/// Capture the pieces of the outer iterate the resto initializer needs.
/// Returns `None` if `outer_data.curr` is unset or the cq layer can't
/// produce a valid `c(x)` / `d(x) − s` for the current iterate.
fn build_outer_snapshot(
    outer_data: &IpoptDataHandle,
    outer_cq: &IpoptCqHandle,
) -> Option<OuterIterateSnapshot> {
    let curr = outer_data.borrow().curr.clone()?;
    let mu = outer_data.borrow().curr_mu;

    let cq_ref = outer_cq.borrow();
    let c_vec = cq_ref.curr_c();
    let d_minus_s_vec = cq_ref.curr_d_minus_s();
    drop(cq_ref);

    Some(OuterIterateSnapshot {
        mu,
        s: curr.s.clone(),
        z_l: curr.z_l.clone(),
        z_u: curr.z_u.clone(),
        v_l: curr.v_l.clone(),
        v_u: curr.v_u.clone(),
        c_vec,
        d_minus_s_vec,
    })
}

/// Build a zeroed placeholder `IteratesVector` for the resto NLP.
/// Shapes:
///
/// * `x` — 5-block compound `[n_orig, m_eq, m_eq, m_ineq, m_ineq]`
/// * `s` — dense `m_ineq`
/// * `y_c` — dense `m_eq`
/// * `y_d` — dense `m_ineq`
/// * `z_l` — 5-block compound `[n_orig, m_eq, m_eq, m_ineq, m_ineq]`
///   (matches the resto NLP's `x_l_resto`)
/// * `z_u` — dense `n_orig` (slacks have no upper bound)
/// * `v_l` — dense `m_ineq`
/// * `v_u` — dense `m_ineq`
///
/// The init's `set_initial_iterates` overwrites every block, so the
/// values here don't matter — the dims do.
fn make_placeholder_resto_iv(n_orig: Index, m_eq: Index, m_ineq: Index) -> IteratesVector {
    use pounce_linalg::CompoundVectorSpace;

    let x_total = n_orig + 2 * m_eq + 2 * m_ineq;
    let x_space = CompoundVectorSpace::new(5, x_total);
    let s0 = DenseVectorSpace::new(n_orig);
    x_space.set_comp(0, n_orig, {
        let s = Rc::clone(&s0);
        move || Box::new(DenseVector::new(Rc::clone(&s)))
    });
    let s_eq = DenseVectorSpace::new(m_eq);
    for i in [1, 2] {
        x_space.set_comp(i, m_eq, {
            let s = Rc::clone(&s_eq);
            move || Box::new(DenseVector::new(Rc::clone(&s)))
        });
    }
    let s_ineq = DenseVectorSpace::new(m_ineq);
    for i in [3, 4] {
        x_space.set_comp(i, m_ineq, {
            let s = Rc::clone(&s_ineq);
            move || Box::new(DenseVector::new(Rc::clone(&s)))
        });
    }
    let mut x_cv = CompoundVector::new(x_space);
    let zero_n = vec![0.0; n_orig as usize];
    let zero_eq = vec![0.0; m_eq as usize];
    let zero_ineq = vec![0.0; m_ineq as usize];
    downcast_dense_mut(x_cv.comp_mut(0)).set_values(&zero_n);
    downcast_dense_mut(x_cv.comp_mut(1)).set_values(&zero_eq);
    downcast_dense_mut(x_cv.comp_mut(2)).set_values(&zero_eq);
    downcast_dense_mut(x_cv.comp_mut(3)).set_values(&zero_ineq);
    downcast_dense_mut(x_cv.comp_mut(4)).set_values(&zero_ineq);

    // z_l: same compound shape.
    let z_l_space = CompoundVectorSpace::new(5, x_total);
    z_l_space.set_comp(0, n_orig, {
        let s = Rc::clone(&s0);
        move || Box::new(DenseVector::new(Rc::clone(&s)))
    });
    for i in [1, 2] {
        z_l_space.set_comp(i, m_eq, {
            let s = Rc::clone(&s_eq);
            move || Box::new(DenseVector::new(Rc::clone(&s)))
        });
    }
    for i in [3, 4] {
        z_l_space.set_comp(i, m_ineq, {
            let s = Rc::clone(&s_ineq);
            move || Box::new(DenseVector::new(Rc::clone(&s)))
        });
    }
    let mut z_l_cv = CompoundVector::new(z_l_space);
    downcast_dense_mut(z_l_cv.comp_mut(0)).set_values(&zero_n);
    downcast_dense_mut(z_l_cv.comp_mut(1)).set_values(&zero_eq);
    downcast_dense_mut(z_l_cv.comp_mut(2)).set_values(&zero_eq);
    downcast_dense_mut(z_l_cv.comp_mut(3)).set_values(&zero_ineq);
    downcast_dense_mut(z_l_cv.comp_mut(4)).set_values(&zero_ineq);

    let mut s = DenseVectorSpace::new(m_ineq).make_new_dense();
    s.set_values(&zero_ineq);
    let mut y_c = DenseVectorSpace::new(m_eq).make_new_dense();
    y_c.set_values(&zero_eq);
    let mut y_d = DenseVectorSpace::new(m_ineq).make_new_dense();
    y_d.set_values(&zero_ineq);
    let mut z_u = DenseVectorSpace::new(n_orig).make_new_dense();
    z_u.set_values(&zero_n);
    let mut v_l = DenseVectorSpace::new(m_ineq).make_new_dense();
    v_l.set_values(&zero_ineq);
    let mut v_u = DenseVectorSpace::new(m_ineq).make_new_dense();
    v_u.set_values(&zero_ineq);

    IteratesVector::new(
        Rc::new(x_cv),
        Rc::new(s),
        Rc::new(y_c),
        Rc::new(y_d),
        Rc::new(z_l_cv),
        Rc::new(z_u),
        Rc::new(v_l),
        Rc::new(v_u),
    )
}

/// Inner-IPM termination → resto-success predicate. Mirrors the upper-
/// half of the `bool MinC_1NrmRestorationPhase::PerformRestoration`
/// return value (`IpRestoMinC_1Nrm.cpp:332-340`): success if the inner
/// converged or hit the user-defined acceptable level; failure on any
/// other terminal status.
fn is_resto_success(status: SolverReturn) -> bool {
    matches!(
        status,
        SolverReturn::Success
            | SolverReturn::StopAtAcceptablePoint
            | SolverReturn::FeasiblePointFound
    )
}

/// Expand a vector block to a dense slice, panicking with a clear
/// diagnostic if it is not a `DenseVector`. Previously a failed downcast
/// silently produced `vec![0.0; fallback_dim]`, masking a non-dense block
/// by corrupting the clone with zeros. The restoration data is all
/// `DenseVector`, so a non-dense block is an invariant violation that
/// must surface, not be papered over. `fallback_dim` is retained only to
/// size the diagnostic.
fn expanded_dense_values(v: &dyn Vector, fallback_dim: Index) -> Vec<f64> {
    v.as_any()
        .downcast_ref::<DenseVector>()
        .map(|d| d.expanded_values())
        .unwrap_or_else(|| {
            panic!(
                "expanded_dense_values: expected a DenseVector for a length-{fallback_dim} block (got a non-dense block)"
            )
        })
}

fn clone_to_dense(template: &dyn Vector) -> Rc<dyn Vector> {
    let n = template.dim();
    let mut v = DenseVectorSpace::new(n).make_new_dense();
    let vals = expanded_dense_values(template, n);
    if !vals.is_empty() {
        v.set_values(&vals);
    }
    Rc::new(v)
}

fn clone_dense_block(v: &dyn Vector) -> Option<Rc<dyn Vector>> {
    let dv = v.as_any().downcast_ref::<DenseVector>()?;
    let mut out = DenseVectorSpace::new(dv.dim()).make_new_dense();
    let vals = dv.expanded_values();
    if !vals.is_empty() {
        out.set_values(&vals);
    }
    Some(Rc::new(out))
}

fn downcast_dense_mut(v: &mut dyn Vector) -> &mut DenseVector {
    v.as_any_mut()
        .downcast_mut::<DenseVector>()
        .expect("expected DenseVector component")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_outer_resto_options_propagates_overrides() {
        // #191: restoration options set on the outer builder must reach
        // the restoration builder (which frontends construct with bare
        // defaults).
        let mut ab = AlgorithmBuilder::new();
        ab.resto.resto_penalty_parameter = 2.5e3;
        ab.resto.resto_proximity_weight = 4.0;
        ab.resto.bound_mult_reset_threshold = 7.0e2;
        ab.resto.constr_mult_reset_threshold = 9.0;
        ab.resto.required_infeasibility_reduction = 0.25;

        let mut rb = RestoAlgorithmBuilder::new();
        apply_outer_resto_options(&mut rb, &ab);

        assert_eq!(rb.rho, 2.5e3);
        assert_eq!(rb.eta_factor, 4.0);
        assert_eq!(rb.bound_mult_reset_threshold, 7.0e2);
        assert_eq!(rb.constr_mult_reset_threshold, 9.0);
        assert_eq!(rb.required_infeasibility_reduction, 0.25);
    }

    /// #439: `required_infeasibility_reduction` was registered but the
    /// κ_resto the guard runs with was hardcoded to upstream's 0.9, so
    /// setting the option was a silent no-op.
    #[test]
    fn effective_kappa_resto_honors_user_option() {
        assert_eq!(effective_kappa_resto(0.9, false), 0.9);
        assert_eq!(effective_kappa_resto(0.25, false), 0.25);
        // `0` disables the guard entirely — the sub-solve then runs to its
        // own convergence.
        assert_eq!(effective_kappa_resto(0.0, false), 0.0);
    }

    /// The square-problem case wins over the user's value, because that is
    /// how upstream applies it: `IpRestoMinC_1Nrm.cpp:157-163` *overwrites*
    /// the sub-option with 0 before `IpRestoConvCheck` reads it.
    #[test]
    fn effective_kappa_resto_square_problem_overrides_user_option() {
        assert_eq!(effective_kappa_resto(0.9, true), 0.0);
        assert_eq!(effective_kappa_resto(0.25, true), 0.0);
    }

    #[test]
    fn apply_outer_resto_options_defaults_are_unchanged() {
        // Default outer options must reproduce the restoration builder's
        // own defaults exactly, so a run that sets nothing is untouched.
        let ab = AlgorithmBuilder::new();
        let mut rb = RestoAlgorithmBuilder::new();
        let baseline = RestoAlgorithmBuilder::new();
        apply_outer_resto_options(&mut rb, &ab);
        assert_eq!(rb.rho, baseline.rho);
        assert_eq!(rb.eta_factor, baseline.eta_factor);
        assert_eq!(
            rb.bound_mult_reset_threshold,
            baseline.bound_mult_reset_threshold
        );
        assert_eq!(
            rb.constr_mult_reset_threshold,
            baseline.constr_mult_reset_threshold
        );
        assert_eq!(
            rb.required_infeasibility_reduction,
            baseline.required_infeasibility_reduction
        );
    }

    #[test]
    fn placeholder_resto_iv_has_correct_shapes() {
        let iv = make_placeholder_resto_iv(2, 1, 1);
        assert_eq!(iv.x.dim(), 2 + 2 * 1 + 2 * 1);
        assert_eq!(iv.s.dim(), 1);
        assert_eq!(iv.y_c.dim(), 1);
        assert_eq!(iv.y_d.dim(), 1);
        // z_l is a 5-block compound matching x_l_resto
        let zl = iv
            .z_l
            .as_any()
            .downcast_ref::<CompoundVector>()
            .expect("z_l compound");
        assert_eq!(zl.n_comps(), 5);
        assert_eq!(zl.comp(0).dim(), 2);
        assert_eq!(zl.comp(1).dim(), 1);
        assert_eq!(iv.z_u.dim(), 2);
        assert_eq!(iv.v_l.dim(), 1);
        assert_eq!(iv.v_u.dim(), 1);
    }

    /// **L18 regression.** The restoration sub-solve must inherit the
    /// user's outer `tol`/`acceptable_tol`/`acceptable_iter` (carried on
    /// the inner builder's `conv_check` options) rather than the
    /// previously-hardcoded `(1e-8, 1e-6, 15)`. Upstream clones the outer
    /// options into the resto sub-options, so a user `tol=1e-3` drives the
    /// resto IPM to `1e-3` stationarity too.
    #[test]
    fn resto_conv_check_adapter_inherits_user_tolerances() {
        let mut conv = pounce_algorithm::alg_builder::ConvCheckOptions::default();
        // Deliberately not the old hardcoded defaults.
        conv.tol = 1e-3;
        conv.acceptable_tol = 1e-2;
        conv.acceptable_iter = 7;

        let adapter = build_resto_conv_check_adapter(&conv);
        assert_eq!(adapter.inner_tol(), 1e-3, "outer tol must propagate");
        assert_eq!(adapter.inner_acceptable_tol(), 1e-2);
        assert_eq!(adapter.inner_acceptable_iter(), 7);
    }

    #[test]
    fn is_resto_success_only_accepts_successful_terminations() {
        assert!(is_resto_success(SolverReturn::Success));
        assert!(is_resto_success(SolverReturn::StopAtAcceptablePoint));
        assert!(is_resto_success(SolverReturn::FeasiblePointFound));
        assert!(!is_resto_success(SolverReturn::MaxiterExceeded));
        assert!(!is_resto_success(SolverReturn::RestorationFailure));
        assert!(!is_resto_success(SolverReturn::InternalError));
        assert!(!is_resto_success(SolverReturn::LocalInfeasibility));
    }
}

#[cfg(test)]
mod issue_661_divergence_guard {
    use super::*;

    /// gh#661, the case the guard exists for. `pooling_rt2stp.nl` under
    /// `mehrotra_algorithm=yes`: restoration entered at a violation of
    /// 6.957e0 and left at 7.355e5 — 105,700x worse — and a reconstructed
    /// gate read that as "converged to a point of local infeasibility".
    /// The model solves at default options, and its outer trajectory
    /// spent 7 iterations at a floor: no evidence of being out of room.
    #[test]
    fn a_blow_up_without_floor_evidence_is_divergence() {
        assert!(diverged_from_restoration_entry(6.957_464e0, 7.354_646e5, 7));
    }

    /// A plateau — the signature the reconstructed gates actually describe —
    /// keeps its verdict. `qcqp750-2nc`, the fixture the `step_failure` gate
    /// is named for, sat pinned at `inf_pr = 1.05e6` for 25+ iterations,
    /// i.e. ~1x entry. The guard never engages, whatever the floor evidence.
    #[test]
    fn a_plateau_is_not_divergence() {
        assert!(!diverged_from_restoration_entry(1.05e6, 1.05e6, 27));
        // and wandering within the headroom is still a plateau
        assert!(!diverged_from_restoration_entry(1.05e6, 9.9e6, 27));
    }

    /// The headroom is a strict `>`, so exactly 10x entry is not yet
    /// divergence.
    #[test]
    fn the_headroom_boundary_is_exclusive() {
        assert!(!diverged_from_restoration_entry(
            1.0,
            RESTO_DIVERGENCE_HEADROOM,
            5
        ));
        assert!(diverged_from_restoration_entry(
            1.0,
            RESTO_DIVERGENCE_HEADROOM * 1.000_001,
            5
        ));
    }

    /// Restoration that *improved* on entry is the ordinary case and must
    /// never be read as divergence.
    #[test]
    fn progress_is_not_divergence() {
        assert!(!diverged_from_restoration_entry(1.0e3, 1.0e-4, 40));
    }

    /// `issue_508_infeasible_gap_1em2`, which reaches a *correct*
    /// infeasibility verdict: 943 of its 1016 outer iterations sat at the
    /// `1.0e-2` gap the fixture is built around, and only then did a
    /// four-iteration restoration sub-solve jump from `1.04e-2` to
    /// `3.19e9`. The final ratio is ~1e11 — far past the headroom — but
    /// the floor was already demonstrated, so the verdict stands.
    #[test]
    fn a_demonstrated_floor_keeps_its_verdict_despite_a_terminal_blow_up() {
        assert!(!diverged_from_restoration_entry(1.04e-2, 3.19e9, 943));
    }

    /// The point of measuring a floor rather than counting iterations,
    /// and the gh#664 residual this replaces. A solve may run for any
    /// number of iterations while still descending — it has not run out
    /// of room, and the reconstructed gates' premise does not hold for it.
    /// Under the old proxy an elapsed-iteration count alone bought the
    /// exemption; here the count that matters is time *at a floor*, which
    /// such a solve never accumulates.
    #[test]
    fn a_long_run_without_a_floor_is_still_divergence() {
        assert!(diverged_from_restoration_entry(1.04e-2, 3.19e9, 0));
        assert!(diverged_from_restoration_entry(
            1.04e-2,
            3.19e9,
            RESTO_STALL_EVIDENCE_ITERS - 1
        ));
    }

    /// The waiver boundary is inclusive, on `>=`.
    #[test]
    fn the_floor_waiver_boundary_is_inclusive() {
        let (entry, final_) = (1.0, 1.0e9);
        assert!(diverged_from_restoration_entry(
            entry,
            final_,
            RESTO_STALL_EVIDENCE_ITERS - 1
        ));
        assert!(!diverged_from_restoration_entry(
            entry,
            final_,
            RESTO_STALL_EVIDENCE_ITERS
        ));
    }

    /// Without a usable entry reference there is nothing to measure
    /// divergence against, so the gates are left exactly as they were
    /// rather than suppressed on a ratio against zero or a NaN.
    #[test]
    fn an_unusable_entry_reference_suppresses_nothing() {
        assert!(!diverged_from_restoration_entry(0.0, 1.0e9, 5));
        assert!(!diverged_from_restoration_entry(f64::NAN, 1.0e9, 5));
        assert!(!diverged_from_restoration_entry(f64::INFINITY, 1.0e9, 5));
        assert!(!diverged_from_restoration_entry(-1.0, 1.0e9, 5));
    }
}
