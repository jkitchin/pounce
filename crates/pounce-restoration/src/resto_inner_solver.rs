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
use pounce_algorithm::ipopt_cq::{IpoptCalculatedQuantities, IpoptCqHandle};
use pounce_algorithm::ipopt_data::{IpoptData, IpoptDataHandle};
use pounce_algorithm::ipopt_nlp::IpoptNlp;
use pounce_algorithm::iterates_vector::IteratesVector;
use pounce_algorithm::mu::monotone::MonotoneMuUpdate;
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
        move |outer_data, outer_cq, outer_nlp, orig_progress_cb, print_iter_output| {
            run_inner_resto(
                outer_data,
                outer_cq,
                outer_nlp,
                &resto_builder,
                &inner_alg_builder,
                backend_factory_factory(),
                orig_progress_cb,
                print_iter_output,
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
pub fn make_default_restoration_factory(
    resto_builder: RestoAlgorithmBuilder,
    inner_alg_builder: AlgorithmBuilder,
    backend_factory_factory: InnerBackendFactoryFactory,
) -> Box<dyn FnMut() -> Box<dyn pounce_algorithm::restoration::RestorationPhase>> {
    let mut state = Some((resto_builder, inner_alg_builder, backend_factory_factory));
    Box::new(move || {
        let (rb, ab, bff) = state
            .take()
            .expect("restoration factory invoked more than once");
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
/// `FeralConfig` (which is `Copy`) can pass
/// `move || Box::new(move || default_backend_factory(feral_cfg))`.
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
    let is_square_problem = n_orig == m_eq;
    let kappa_resto = if is_square_problem { 0.0 } else { 0.9 };

    // ---- 4. Build the inner alg bundle and override its init /
    //         conv_check / iter_output slots with resto-side ones. ----
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
    let mut adapter = crate::conv_check::RestoConvCheckAdapter::new(1e-8, 1e-6, 15, 3000, 3000)
        .with_orig_progress_guard(Rc::clone(outer_nlp), orig_curr_inf_pr, kappa_resto);
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
        pounce_algorithm::alg_builder::MuStrategyChoice::Monotone => Box::new(
            MonotoneMuUpdate::new()
                .with_first_iter_resto(true)
                .with_mu_min(resto_mu_min),
        )
            as Box<dyn pounce_algorithm::mu::r#trait::MuUpdate>,
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
    // Forward the outer `print_level == 0` gate. Suppresses the
    // restoration `r`-suffixed iter table; the resto-of-resto level
    // also inherits the same flag (its `RestorationPhase` impl is the
    // closed-form `RestoRestorationPhase`, which doesn't print).
    alg.print_iter_output = print_iter_output;
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
    let outer_tol = outer_data.borrow().tol;
    let orig_inf_pr_at_final =
        eval_orig_inf_pr_at_inner_curr(&*final_iv.x, &*final_iv.s, outer_nlp).unwrap_or(0.0);
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
    // when the outer cv at re-entry is bounded above `max(100*tol, 1e-4)`
    // and `ErrorInStepComputation` otherwise.
    let strict_locally_infeasible = !is_square_problem
        && matches!(
            status,
            SolverReturn::Success | SolverReturn::StopAtAcceptablePoint
        )
        && inner_stationarity_converged
        && orig_inf_pr_at_final > (100.0 * outer_tol).max(1e-4);

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
    //   * `orig_inf_pr_at_final > max(100*outer_tol, 1e-3)` — the
    //     orig-NLP `inf_pr` floor is non-trivial (i.e. NOT just a
    //     little above outer tol — distinguish from the kappa-guard
    //     near-feasible exit).
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
        && orig_inf_pr_at_final > (100.0 * outer_tol).max(1e-3)
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
        && inner_iter_count >= 1000
        && orig_inf_pr_at_final > (100.0 * outer_tol).max(1e-3)
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
        && orig_inf_pr_at_final > (100.0 * outer_tol).max(1e-3)
        && orig_inf_pr_at_final.is_finite();

    let locally_infeasible = strict_locally_infeasible
        || alt_locally_infeasible
        || cycle_locally_infeasible
        || step_failure_locally_infeasible;

    if std::env::var_os("POUNCE_DBG_RESTO_LOCINF").is_some() {
        tracing::debug!(target: "pounce::restoration",
            "[PN_RESTO_LOCINF] status={:?} iter={} inner_kkt_err={:.6e} orig_inf_pr={:.6e} outer_tol={:.6e} strict={} alt={} cycle={} step_fail={} → loc_inf={}",
            status,
            inner_iter_count,
            inner_kkt_err,
            orig_inf_pr_at_final,
            outer_tol,
            strict_locally_infeasible,
            alt_locally_infeasible,
            cycle_locally_infeasible,
            step_failure_locally_infeasible,
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
    })
}

/// Evaluate `max(||c(x_orig)||∞, ||d(x_orig) − s||∞)` at the inner
/// IPM's converged iterate. Returns `None` on any downcast / dim
/// mismatch (caller treats as `0.0` and the locally-infeasible gate
/// fails closed — i.e. we don't spuriously declare infeasibility on
/// a fixture we can't evaluate).
fn eval_orig_inf_pr_at_inner_curr(
    inner_x: &dyn Vector,
    inner_s: &dyn Vector,
    orig_rc: &Rc<RefCell<dyn IpoptNlp>>,
) -> Option<f64> {
    let xc = inner_x.as_any().downcast_ref::<CompoundVector>()?;
    let x_orig = xc.comp(BLOCK_X);
    let mut orig = orig_rc.borrow_mut();
    let m_eq = orig.m_eq();
    let m_ineq = orig.m_ineq();
    let c_amax = if m_eq > 0 {
        let mut buf = DenseVectorSpace::new(m_eq).make_new_dense();
        orig.eval_c(x_orig, &mut buf);
        buf.amax()
    } else {
        0.0
    };
    let d_minus_s_amax = if m_ineq > 0 {
        let mut buf = DenseVectorSpace::new(m_ineq).make_new_dense();
        orig.eval_d(x_orig, &mut buf);
        buf.axpy(-1.0, inner_s);
        buf.amax()
    } else {
        0.0
    };
    Some(c_amax.max(d_minus_s_amax))
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

fn expanded_dense_values(v: &dyn Vector, fallback_dim: Index) -> Vec<f64> {
    v.as_any()
        .downcast_ref::<DenseVector>()
        .map(|d| d.expanded_values())
        .unwrap_or_else(|| vec![0.0; fallback_dim as usize])
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
