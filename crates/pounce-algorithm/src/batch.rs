//! Batched NLP solving (pounce#126) — N independent NLPs on a rayon
//! pool.
//!
//! The NLP analog of `pounce-convex`'s `solve_qp_batch` /
//! `solve_qp_batch_parallel`: solve a batch of independent problems
//! (parametric sweeps, multi-start, branch-and-bound node relaxations)
//! and return one result per input, in input order. Each instance runs
//! the full filter-IPM end-to-end via its own [`IpoptApplication`].
//!
//! # Parallelism model: outer-parallel, inner-serial
//!
//! Same rationale as the QP batch (`pounce-convex/src/batch.rs`): each
//! NLP solve is fully independent, so the batch is embarrassingly
//! parallel *across instances* — but the default FERAL factorization
//! backend is itself rayon-parallel *within* a factor, and running
//! both levels oversubscribes the cores. [`solve_nlp_batch_parallel`]
//! therefore builds everything per worker, and the `configure` hook
//! should install an **inner-serial** linear-solver backend;
//! [`install_serial_feral_backend`] does exactly that (it honors the
//! app's `feral_*` options, forcing only `parallel = off` — a
//! per-backend setting, not global state). Unlike the QP batch, NLP
//! instances converge in wildly different iteration counts; that is
//! fine for thread-per-instance rayon (a converged instance frees its
//! worker — no lockstep), and no cross-instance KKT-structure sharing
//! is attempted because general-NLP sparsity may differ per instance.
//!
//! # Construction happens on the worker
//!
//! `pounce-nlp`'s solver plumbing is single-threaded
//! (`Rc<RefCell<…>>`), so nothing solver-side crosses a thread
//! boundary: each worker receives one owned `T: TNLP + Send` (e.g. a
//! `pounce_nl::nl_reader::NlTnlp`, which is `Send` since its CSE nodes
//! went `Arc`), constructs its own [`IpoptApplication`] *inside* the
//! worker closure, and only the plain-data [`NlpBatchResult`] comes
//! back.
//!
//! # The `configure` hook
//!
//! `pounce-algorithm` cannot depend on `pounce-restoration` (the dep
//! edge runs the other way), so the per-worker application is handed
//! to a caller-supplied `configure` closure that should, at minimum,
//! mirror what single-solve drivers do: set options, then install a
//! linear-backend factory and a restoration-factory provider. Without
//! a restoration provider an instance that needs the restoration
//! phase returns `RestorationFailure` instead of recovering — same as
//! a bare `IpoptApplication`. See `pounce-cli`'s solve setup or
//! `pounce-py`'s `Problem::prepare` for the full recipe.

use crate::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_linsol::{EMatrixFormat, ESymSolverStatus, SparseSymLinearSolverInterface};
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{
    BoundsInfo, IpoptCq, IpoptData, IterStats, Linearity, MetaData, NlpInfo, ScalingRequest,
    Solution, SparsityRequest, StartingPoint, TNLP,
};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};

/// Final iterate of one batch instance, captured from the
/// `finalize_solution` callback (owned copies of the borrowed
/// buffers).
#[derive(Debug, Clone)]
pub struct NlpBatchSolution {
    /// Algorithm-level termination status (finer-grained than the
    /// application-level [`NlpBatchResult::status`]).
    pub solver_status: SolverReturn,
    pub x: Vec<Number>,
    /// Lower / upper bound multipliers.
    pub z_l: Vec<Number>,
    pub z_u: Vec<Number>,
    /// Constraint values `g(x)` at the final iterate.
    pub g: Vec<Number>,
    /// Constraint multipliers.
    pub lambda: Vec<Number>,
    pub obj: Number,
}

/// Outcome of one instance of a batched solve.
#[derive(Debug, Clone)]
pub struct NlpBatchResult {
    pub status: ApplicationReturnStatus,
    /// `None` when the solve aborted before `finalize_solution` ran
    /// (e.g. `InvalidProblemDefinition`).
    pub solution: Option<NlpBatchSolution>,
    /// Per-instance solve statistics (iteration count, final KKT
    /// error, timings, …).
    pub stats: SolveStatistics,
}

/// Delegating wrapper that records the `finalize_solution` payload so
/// the batch driver can return it as owned data. Every other `TNLP`
/// method forwards to the wrapped instance unchanged.
struct CaptureTnlp<T: TNLP> {
    inner: T,
    captured: Option<NlpBatchSolution>,
}

impl<T: TNLP> TNLP for CaptureTnlp<T> {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.get_nlp_info()
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        self.inner.get_bounds_info(b)
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        self.inner.get_starting_point(sp)
    }
    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.inner.eval_f(x, new_x)
    }
    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        self.inner.eval_grad_f(x, new_x, grad_f)
    }
    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        self.inner.eval_g(x, new_x, g)
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        self.inner.eval_jac_g(x, new_x, mode)
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
            .eval_h(x, new_x, obj_factor, lambda, new_lambda, mode)
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        self.captured = Some(NlpBatchSolution {
            solver_status: sol.status,
            x: sol.x.to_vec(),
            z_l: sol.z_l.to_vec(),
            z_u: sol.z_u.to_vec(),
            g: sol.g.to_vec(),
            lambda: sol.lambda.to_vec(),
            obj: sol.obj_value,
        });
        self.inner.finalize_solution(sol, ip_data, ip_cq);
    }
    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        self.inner.get_var_con_metadata(var, con)
    }
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        self.inner.get_scaling_parameters(req)
    }
    fn get_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.get_variables_linearity(types)
    }
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.get_constraints_linearity(types)
    }
    fn get_number_of_nonlinear_variables(&mut self) -> pounce_common::types::Index {
        self.inner.get_number_of_nonlinear_variables()
    }
    fn get_list_of_nonlinear_variables(
        &mut self,
        pos_nonlin_vars: &mut [pounce_common::types::Index],
    ) -> bool {
        self.inner.get_list_of_nonlinear_variables(pos_nonlin_vars)
    }
    fn intermediate_callback(
        &mut self,
        stats: IterStats,
        ip_data: &IpoptData,
        ip_cq: &IpoptCq,
    ) -> bool {
        self.inner.intermediate_callback(stats, ip_data, ip_cq)
    }
    fn finalize_metadata(&mut self, var: &MetaData, con: &MetaData) {
        self.inner.finalize_metadata(var, con)
    }
}

/// Install an inner-serial FERAL backend on `app`, honoring any
/// `feral_*` options already set (read the options *before* calling
/// this if `configure` sets them — i.e. set options first, then call
/// this). The `parallel = off` toggle is per-backend; concurrent
/// solves on other threads are unaffected.
pub fn install_serial_feral_backend(app: &mut IpoptApplication) {
    let mut cfg = crate::application::feral_config_from_options(app.options());
    cfg.parallel = Some(false);
    app.set_linear_backend_factory(Box::new(move |_choice| {
        Box::new(pounce_feral::FeralSolverInterface::with_config(cfg.clone()))
    }));
}

/// Per-thread pool of serial FERAL backends for **identical-sparsity**
/// batches (the issue-#126 "optional later optimization": reuse the
/// symbolic analysis across instances).
///
/// FERAL's `Solver` keys its symbolic factorization (fill-reducing
/// ordering + supernode structure) on a pattern fingerprint and reuses
/// it across `factor()` calls — that is what already amortizes the
/// symbolic cost over a single instance's IPM iterations. A fresh
/// backend per instance throws that cache away between instances; this
/// pool instead parks each worker thread's backend when its solve
/// finishes and hands it to the next instance scheduled on that
/// thread. When the next instance's KKT pattern is identical, its
/// first factorization hits the cached symbolic; when it differs, the
/// fingerprint mismatch triggers a fresh analysis — so pooling is
/// *always correct*, just only *profitable* for shared-structure
/// sweeps.
///
/// **Determinism caveat (why this is opt-in):** the pooled solver
/// carries value-adjacent state across instances — most notably the
/// MC64 scaling cache (reused only inside its validity bound) and any
/// pivot-quality escalation a previous instance triggered. Solutions
/// still satisfy the same tolerances, but are not guaranteed
/// bit-identical to fresh-backend solves. The default batch entry
/// points keep one fresh backend per instance.
///
/// Construct once per batch ([`Self::serial`]), share the `Arc` with
/// each worker's `configure` via
/// [`install_pooled_serial_feral_backend`]. Restoration-phase inner
/// solves keep their own fresh backends. Pooled solvers (and their
/// cached factors) free when the `Arc` drops.
pub struct FeralBackendPool {
    cfg: pounce_feral::FeralConfig,
    slots: Mutex<HashMap<ThreadId, pounce_feral::FeralSolverInterface>>,
}

impl FeralBackendPool {
    /// A pool minting inner-serial backends with `cfg` (its `parallel`
    /// field is forced off — pooling exists for the outer-parallel /
    /// inner-serial batch shape).
    pub fn serial(mut cfg: pounce_feral::FeralConfig) -> Arc<Self> {
        cfg.parallel = Some(false);
        Arc::new(Self {
            cfg,
            slots: Mutex::new(HashMap::new()),
        })
    }

    /// Take this thread's parked backend (or mint a fresh one). The
    /// returned guard parks it back on drop, on the same thread.
    fn acquire(self: &Arc<Self>) -> PooledFeralBackend {
        let recycled = self
            .slots
            .lock()
            .ok()
            .and_then(|mut s| s.remove(&thread::current().id()));
        let inner = recycled
            .unwrap_or_else(|| pounce_feral::FeralSolverInterface::with_config(self.cfg.clone()));
        PooledFeralBackend {
            inner: Some(inner),
            pool: Arc::clone(self),
        }
    }
}

impl std::fmt::Debug for FeralBackendPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parked = self.slots.lock().map(|s| s.len()).unwrap_or(0);
        f.debug_struct("FeralBackendPool")
            .field("parked", &parked)
            .finish_non_exhaustive()
    }
}

/// RAII guard around a pooled [`pounce_feral::FeralSolverInterface`]:
/// delegates the backend trait verbatim and parks the solver back into
/// its pool slot when the solve's ownership chain drops it. If the
/// slot is already occupied (two live backends on one thread — e.g. a
/// nested factory call), the extra solver is simply dropped: pooling
/// degrades to fresh-per-instance, never to shared-mutable state.
struct PooledFeralBackend {
    inner: Option<pounce_feral::FeralSolverInterface>,
    pool: Arc<FeralBackendPool>,
}

impl PooledFeralBackend {
    fn get(&mut self) -> &mut pounce_feral::FeralSolverInterface {
        // Invariant: `inner` is `Some` from construction until drop.
        #[allow(clippy::expect_used)]
        self.inner.as_mut().expect("pooled backend already taken")
    }
}

impl Drop for PooledFeralBackend {
    fn drop(&mut self) {
        if let (Some(solver), Ok(mut slots)) = (self.inner.take(), self.pool.slots.lock()) {
            slots.entry(thread::current().id()).or_insert(solver);
        }
    }
}

impl SparseSymLinearSolverInterface for PooledFeralBackend {
    fn initialize_structure(
        &mut self,
        dim: Index,
        nonzeros: Index,
        ia: &[Index],
        ja: &[Index],
    ) -> ESymSolverStatus {
        self.get().initialize_structure(dim, nonzeros, ia, ja)
    }
    fn values_array_mut(&mut self) -> &mut [Number] {
        self.get().values_array_mut()
    }
    fn multi_solve(
        &mut self,
        new_matrix: bool,
        ia: &[Index],
        ja: &[Index],
        nrhs: Index,
        rhs_vals: &mut [Number],
        check_neg_evals: bool,
        number_of_neg_evals: Index,
    ) -> ESymSolverStatus {
        self.get().multi_solve(
            new_matrix,
            ia,
            ja,
            nrhs,
            rhs_vals,
            check_neg_evals,
            number_of_neg_evals,
        )
    }
    fn number_of_neg_evals(&self) -> Index {
        match &self.inner {
            Some(s) => s.number_of_neg_evals(),
            None => 0,
        }
    }
    fn increase_quality(&mut self) -> bool {
        self.get().increase_quality()
    }
    fn provides_inertia(&self) -> bool {
        self.inner.as_ref().is_some_and(|s| s.provides_inertia())
    }
    fn matrix_format(&self) -> EMatrixFormat {
        match &self.inner {
            Some(s) => s.matrix_format(),
            None => EMatrixFormat::TripletFormat,
        }
    }
    fn provides_degeneracy_detection(&self) -> bool {
        self.inner
            .as_ref()
            .is_some_and(|s| s.provides_degeneracy_detection())
    }
    fn determine_dependent_rows(
        &mut self,
        ia: &[Index],
        ja: &[Index],
        c_deps: &mut Vec<Index>,
    ) -> ESymSolverStatus {
        self.get().determine_dependent_rows(ia, ja, c_deps)
    }
    fn factor_pattern(&self, want_values: bool) -> Option<pounce_linsol::FactorPattern> {
        self.inner
            .as_ref()
            .and_then(|s| s.factor_pattern(want_values))
    }
}

/// Install a backend factory drawing from `pool` — the
/// identical-sparsity batch optimization (see [`FeralBackendPool`] for
/// the reuse semantics and the determinism caveat). Call from
/// `configure` *instead of* [`install_serial_feral_backend`].
pub fn install_pooled_serial_feral_backend(
    app: &mut IpoptApplication,
    pool: &Arc<FeralBackendPool>,
) {
    let pool = Arc::clone(pool);
    app.set_linear_backend_factory(Box::new(move |_choice| Box::new(pool.acquire())));
}

/// Per-instance warm-start iterate for [`solve_nlp_batch_parallel_warm`]:
/// primal point plus the three dual vectors the IPM's warm-start
/// initializer consumes via `TNLP::get_starting_point`. Build one from
/// a previous batch's [`NlpBatchSolution`] (the `From` impl) for MPC /
/// sequential-chaining workloads.
#[derive(Debug, Clone, Default)]
pub struct NlpWarmStart {
    pub x: Vec<Number>,
    /// Constraint multipliers λ.
    pub lambda: Vec<Number>,
    /// Lower / upper bound multipliers.
    pub z_l: Vec<Number>,
    pub z_u: Vec<Number>,
    /// Barrier parameter μ to resume from — typically the previous
    /// solve's final μ ([`SolveStatistics::final_mu`]). When `Some`,
    /// the warm batch sets `mu_init` to it (floored at `1e-9`, and
    /// only if the caller's `configure` didn't set `mu_init`
    /// explicitly), so the IPM continues near the converged barrier
    /// instead of recentering at the cold default `0.1` — without
    /// this, a warm start from a near-optimal point can *gain*
    /// iterations walking back out to the central path.
    pub mu: Option<Number>,
}

impl From<&NlpBatchSolution> for NlpWarmStart {
    fn from(sol: &NlpBatchSolution) -> Self {
        Self {
            x: sol.x.clone(),
            lambda: sol.lambda.clone(),
            z_l: sol.z_l.clone(),
            z_u: sol.z_u.clone(),
            mu: None,
        }
    }
}

/// Build a warm start from a full batch result: iterate + duals from
/// the captured solution, μ from the statistics. An instance whose
/// solve produced no solution yields an empty warm start, which the
/// next solve treats as a cold start (dimension-mismatch fallback).
impl From<&NlpBatchResult> for NlpWarmStart {
    fn from(r: &NlpBatchResult) -> Self {
        match &r.solution {
            Some(sol) => Self {
                mu: Some(r.stats.final_mu),
                ..Self::from(sol)
            },
            None => Self::default(),
        }
    }
}

/// Floor for warm-start `mu_init` threading (see [`NlpWarmStart::mu`]).
const WARM_MU_FLOOR: Number = 1e-9;

/// Apply the warm-start options to a configured per-worker app:
/// always force `warm_start_init_point=yes` (the point of the warm
/// entry), and thread the instance's resumed μ into `mu_init` unless
/// the caller's `configure` already set one explicitly.
fn apply_warm_options(app: &mut IpoptApplication, mu: Option<Number>) {
    let _ = app
        .options_mut()
        .set_string_value("warm_start_init_point", "yes", true, false);
    if let Some(mu) = mu {
        let user_set_mu = matches!(
            app.options().get_numeric_value("mu_init", ""),
            Ok((_, true))
        );
        if !user_set_mu {
            let _ =
                app.options_mut()
                    .set_numeric_value("mu_init", mu.max(WARM_MU_FLOOR), true, false);
        }
    }
}

/// Delegating wrapper that serves a [`NlpWarmStart`] from
/// `get_starting_point` (primal always; duals when the initializer
/// asks, i.e. under `warm_start_init_point=yes`). Any length mismatch
/// against the wrapped problem falls back to the inner TNLP's own
/// starting point — a per-instance cold start, mirroring the QP batch's
/// warm-start contract. Every other method forwards unchanged.
struct WarmStartTnlp<T: TNLP> {
    inner: T,
    warm: NlpWarmStart,
}

impl<T: TNLP> TNLP for WarmStartTnlp<T> {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.get_nlp_info()
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        self.inner.get_bounds_info(b)
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        let dims_ok = self.warm.x.len() == sp.x.len()
            && (!sp.init_lambda || self.warm.lambda.len() == sp.lambda.len())
            && (!sp.init_z
                || (self.warm.z_l.len() == sp.z_l.len() && self.warm.z_u.len() == sp.z_u.len()));
        if !dims_ok {
            return self.inner.get_starting_point(sp);
        }
        if sp.init_x {
            sp.x.copy_from_slice(&self.warm.x);
        }
        if sp.init_z {
            sp.z_l.copy_from_slice(&self.warm.z_l);
            sp.z_u.copy_from_slice(&self.warm.z_u);
        }
        if sp.init_lambda {
            sp.lambda.copy_from_slice(&self.warm.lambda);
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.inner.eval_f(x, new_x)
    }
    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        self.inner.eval_grad_f(x, new_x, grad_f)
    }
    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        self.inner.eval_g(x, new_x, g)
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        self.inner.eval_jac_g(x, new_x, mode)
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
            .eval_h(x, new_x, obj_factor, lambda, new_lambda, mode)
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        self.inner.finalize_solution(sol, ip_data, ip_cq)
    }
    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        self.inner.get_var_con_metadata(var, con)
    }
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        self.inner.get_scaling_parameters(req)
    }
    fn get_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.get_variables_linearity(types)
    }
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.get_constraints_linearity(types)
    }
    fn get_number_of_nonlinear_variables(&mut self) -> pounce_common::types::Index {
        self.inner.get_number_of_nonlinear_variables()
    }
    fn get_list_of_nonlinear_variables(
        &mut self,
        pos_nonlin_vars: &mut [pounce_common::types::Index],
    ) -> bool {
        self.inner.get_list_of_nonlinear_variables(pos_nonlin_vars)
    }
    fn intermediate_callback(
        &mut self,
        stats: IterStats,
        ip_data: &IpoptData,
        ip_cq: &IpoptCq,
    ) -> bool {
        self.inner.intermediate_callback(stats, ip_data, ip_cq)
    }
    fn finalize_metadata(&mut self, var: &MetaData, con: &MetaData) {
        self.inner.finalize_metadata(var, con)
    }
}

/// Solve one instance with a fresh, caller-configured application.
///
/// Panic-isolated: a Rust panic raised *during* the solve — a user
/// `eval_*`/`intermediate_callback` that panics, or a backend assertion
/// tripping on malformed structure — is caught here and reported as an
/// [`ApplicationReturnStatus::InternalError`] row. Without this guard the
/// panic would unwind out of the rayon `map`/`collect` (or the sequential
/// `map`) and discard *every other instance's* result, so a single bad
/// instance would poison the whole batch. (Solver-detected failures —
/// infeasibility, invalid numbers, eval callbacks that *return* failure —
/// are already reported as ordinary statuses and never reach this path.)
fn solve_nlp_one<T, C>(index: usize, tnlp: T, configure: &mut C) -> NlpBatchResult
where
    T: TNLP + 'static,
    C: FnMut(usize, &mut IpoptApplication),
{
    let mut app = IpoptApplication::new();
    configure(index, &mut app);
    let cap = Rc::new(RefCell::new(CaptureTnlp {
        inner: tnlp,
        captured: None,
    }));
    // `AssertUnwindSafe`: on a caught panic we discard `app`/`cap` without
    // observing them, so any broken interior-mutability invariant cannot
    // leak out — only the freshly-built `InternalError` row is returned.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let status = app.optimize_tnlp(Rc::clone(&cap) as Rc<RefCell<dyn TNLP>>);
        let stats = app.statistics();
        let solution = cap.borrow_mut().captured.take();
        NlpBatchResult {
            status,
            solution,
            stats,
        }
    }));
    outcome.unwrap_or_else(|_| NlpBatchResult {
        status: ApplicationReturnStatus::InternalError,
        solution: None,
        stats: SolveStatistics::default(),
    })
}

/// Solve a batch of independent NLPs **sequentially**, returning one
/// result per input in input order. `configure` is called once per
/// instance (receiving the instance index, so per-instance options are
/// possible) on a fresh [`IpoptApplication`] (set options / backend /
/// restoration there — see the module docs). Predictable and
/// contention-free; the right choice when each instance is large
/// enough that the linear-solver backend parallelizes internally.
pub fn solve_nlp_batch<T, C>(problems: Vec<T>, mut configure: C) -> Vec<NlpBatchResult>
where
    T: TNLP + 'static,
    C: FnMut(usize, &mut IpoptApplication),
{
    problems
        .into_iter()
        .enumerate()
        .map(|(i, t)| solve_nlp_one(i, t, &mut configure))
        .collect()
}

/// Solve a batch of independent NLPs **in parallel across instances**
/// on rayon's global pool, returning one result per input in input
/// order regardless of completion order. Best for many small / medium
/// instances where cross-instance throughput beats parallelizing each
/// factor internally.
///
/// Each worker owns its instance end-to-end: the `T: TNLP + Send`
/// moves in, and the application, backend, and restoration strategy
/// are all constructed *inside* the worker via `configure` (which must
/// therefore be `Sync` — it is shared by reference and called once per
/// instance, with the instance index so per-instance options are
/// possible). For the outer-parallel / inner-serial win, have
/// `configure` install an inner-serial backend —
/// [`install_serial_feral_backend`] after setting options.
pub fn solve_nlp_batch_parallel<T, C>(problems: Vec<T>, configure: C) -> Vec<NlpBatchResult>
where
    T: TNLP + Send + 'static,
    C: Fn(usize, &mut IpoptApplication) + Sync,
{
    problems
        .into_par_iter()
        .enumerate()
        .map(|(i, t)| {
            solve_nlp_one(i, t, &mut |i: usize, app: &mut IpoptApplication| {
                configure(i, app)
            })
        })
        .collect()
}

/// Warm-started **sequential** batch — [`solve_nlp_batch`] seeded per
/// instance from `warms`. See [`solve_nlp_batch_parallel_warm`] for
/// the warm-start contract (option forcing, dimension-mismatch
/// fallback).
///
/// # Panics
/// Panics if `warms.len() != problems.len()`.
pub fn solve_nlp_batch_warm<T, C>(
    problems: Vec<T>,
    warms: Vec<NlpWarmStart>,
    mut configure: C,
) -> Vec<NlpBatchResult>
where
    T: TNLP + 'static,
    C: FnMut(usize, &mut IpoptApplication),
{
    assert_eq!(
        warms.len(),
        problems.len(),
        "warms.len() ({}) must equal problems.len() ({})",
        warms.len(),
        problems.len()
    );
    let mus: Vec<Option<Number>> = warms.iter().map(|w| w.mu).collect();
    let wrapped: Vec<WarmStartTnlp<T>> = problems
        .into_iter()
        .zip(warms)
        .map(|(inner, warm)| WarmStartTnlp { inner, warm })
        .collect();
    solve_nlp_batch(wrapped, |i, app: &mut IpoptApplication| {
        configure(i, app);
        apply_warm_options(app, mus[i]);
    })
}

/// Warm-started parallel batch: like [`solve_nlp_batch_parallel`] but
/// each instance is seeded from the corresponding entry of `warms` —
/// typically the previous step's [`NlpBatchSolution`]s for a sequence
/// of nearby batches (receding-horizon MPC, sequential parameter
/// continuation, B&B dives). The NLP analog of the QP batch's
/// `solve_qp_batch_parallel_warm`.
///
/// Each worker forces `warm_start_init_point=yes` *after* `configure`
/// runs (that option is the point of this entry; pair it with
/// `mu_init` / `warm_start_target_mu` via `configure` for the full
/// re-optimization effect), then serves the warm iterate through
/// `get_starting_point`. A warm start only affects an instance's
/// iteration count, not its solution; a per-instance dimension
/// mismatch falls back to that instance's own (cold) starting point.
///
/// # Panics
/// Panics if `warms.len() != problems.len()`.
pub fn solve_nlp_batch_parallel_warm<T, C>(
    problems: Vec<T>,
    warms: Vec<NlpWarmStart>,
    configure: C,
) -> Vec<NlpBatchResult>
where
    T: TNLP + Send + 'static,
    C: Fn(usize, &mut IpoptApplication) + Sync,
{
    assert_eq!(
        warms.len(),
        problems.len(),
        "warms.len() ({}) must equal problems.len() ({})",
        warms.len(),
        problems.len()
    );
    let mus: Vec<Option<Number>> = warms.iter().map(|w| w.mu).collect();
    let wrapped: Vec<WarmStartTnlp<T>> = problems
        .into_iter()
        .zip(warms)
        .map(|(inner, warm)| WarmStartTnlp { inner, warm })
        .collect();
    solve_nlp_batch_parallel(wrapped, |i, app: &mut IpoptApplication| {
        configure(i, app);
        apply_warm_options(app, mus[i]);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_nlp::tnlp::IndexStyle;

    /// `min (x0 - a)^2 + (x1 - b)^2  s.t. x0 + x1 = s` — a tiny
    /// per-instance-parameterized equality-constrained QP with an
    /// analytic solution: `x = (a, b) + ((s - a - b)/2) * (1, 1)`.
    struct ShiftedQuad {
        a: f64,
        b: f64,
        s: f64,
        /// Optional variable bounds (default free).
        x_l: [f64; 2],
        x_u: [f64; 2],
    }

    impl ShiftedQuad {
        fn new(a: f64, b: f64, s: f64) -> Self {
            Self {
                a,
                b,
                s,
                x_l: [-1e19; 2],
                x_u: [1e19; 2],
            }
        }
        fn expected(&self) -> [f64; 2] {
            let t = (self.s - self.a - self.b) / 2.0;
            [self.a + t, self.b + t]
        }
    }

    impl TNLP for ShiftedQuad {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 2,
                m: 1,
                nnz_jac_g: 2,
                nnz_h_lag: 2,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&self.x_l);
            b.x_u.copy_from_slice(&self.x_u);
            b.g_l[0] = self.s;
            b.g_u[0] = self.s;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 0.0;
            sp.x[1] = 0.0;
            true
        }
        fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
            Some((x[0] - self.a).powi(2) + (x[1] - self.b).powi(2))
        }
        fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
            grad_f[0] = 2.0 * (x[0] - self.a);
            grad_f[1] = 2.0 * (x[1] - self.b);
            true
        }
        fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g[0] = x[0] + x[1];
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow.copy_from_slice(&[0, 0]);
                    jcol.copy_from_slice(&[0, 1]);
                }
                SparsityRequest::Values { values } => {
                    values.copy_from_slice(&[1.0, 1.0]);
                }
            }
            true
        }
        fn eval_h(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            obj_factor: Number,
            _lambda: Option<&[Number]>,
            _new_lambda: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow.copy_from_slice(&[0, 1]);
                    jcol.copy_from_slice(&[0, 1]);
                }
                SparsityRequest::Values { values } => {
                    values.copy_from_slice(&[2.0 * obj_factor, 2.0 * obj_factor]);
                }
            }
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    fn configure(_i: usize, app: &mut IpoptApplication) {
        let _ = app
            .options_mut()
            .set_integer_value("print_level", 0, true, false);
        install_serial_feral_backend(app);
    }

    fn batch(k: usize) -> Vec<ShiftedQuad> {
        (0..k)
            .map(|i| ShiftedQuad::new(1.0 + i as f64, 2.0, 1.0 + (i % 3) as f64))
            .collect()
    }

    /// A `ShiftedQuad` that panics inside `eval_f` when `boom` — stands in
    /// for any Rust panic raised mid-solve (a buggy user callback, a backend
    /// assertion). Used to prove panic isolation: the panicking instance must
    /// fail only itself, not unwind the whole batch.
    struct BoomQuad {
        inner: ShiftedQuad,
        boom: bool,
    }

    impl TNLP for BoomQuad {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            self.inner.get_nlp_info()
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            self.inner.get_bounds_info(b)
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            self.inner.get_starting_point(sp)
        }
        fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
            if self.boom {
                panic!("boom: simulated mid-solve panic in eval_f");
            }
            self.inner.eval_f(x, new_x)
        }
        fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
            self.inner.eval_grad_f(x, new_x, grad_f)
        }
        fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
            self.inner.eval_g(x, new_x, g)
        }
        fn eval_jac_g(
            &mut self,
            x: Option<&[Number]>,
            new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            self.inner.eval_jac_g(x, new_x, mode)
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
                .eval_h(x, new_x, obj_factor, lambda, new_lambda, mode)
        }
        fn finalize_solution(&mut self, sol: Solution<'_>, d: &IpoptData, q: &IpoptCq) {
            self.inner.finalize_solution(sol, d, q)
        }
    }

    #[test]
    fn empty_batch_returns_empty() {
        let out = solve_nlp_batch_parallel(Vec::<ShiftedQuad>::new(), configure);
        assert!(out.is_empty());
    }

    #[test]
    fn single_element_batch_solves() {
        let probs = batch(1);
        let expected = probs[0].expected();
        let out = solve_nlp_batch_parallel(probs, configure);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].status, ApplicationReturnStatus::SolveSucceeded);
        let sol = out[0].solution.as_ref().expect("solution captured");
        assert!((sol.x[0] - expected[0]).abs() < 1e-6);
        assert!((sol.x[1] - expected[1]).abs() < 1e-6);
    }

    #[test]
    fn parallel_results_in_input_order_and_match_sequential() {
        let k = 8;
        let expected: Vec<[f64; 2]> = batch(k).iter().map(|p| p.expected()).collect();
        let par = solve_nlp_batch_parallel(batch(k), configure);
        let seq = solve_nlp_batch(batch(k), configure);
        assert_eq!(par.len(), k);
        for i in 0..k {
            assert_eq!(
                par[i].status,
                ApplicationReturnStatus::SolveSucceeded,
                "instance {i}"
            );
            let ps = par[i].solution.as_ref().expect("parallel solution");
            let ss = seq[i].solution.as_ref().expect("sequential solution");
            // Input order: each instance's analytic optimum.
            assert!(
                (ps.x[0] - expected[i][0]).abs() < 1e-6 && (ps.x[1] - expected[i][1]).abs() < 1e-6,
                "instance {i}: got {:?}, expected {:?}",
                ps.x,
                expected[i]
            );
            // Parallel and sequential agree (same algorithm, same
            // serial backend — bit-for-bit).
            assert_eq!(ps.x, ss.x, "instance {i}");
            assert_eq!(
                par[i].stats.iteration_count, seq[i].stats.iteration_count,
                "instance {i}"
            );
        }
    }

    #[test]
    fn infeasible_instance_mixed_in_does_not_poison_batch() {
        // Middle instance has contradictory bounds vs. the equality
        // row: x0 + x1 = 10 with x <= 1 componentwise.
        let mut probs = batch(3);
        probs[1].s = 10.0;
        probs[1].x_u = [1.0; 2];
        let out = solve_nlp_batch_parallel(probs, configure);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].status, ApplicationReturnStatus::SolveSucceeded);
        assert_eq!(out[2].status, ApplicationReturnStatus::SolveSucceeded);
        assert_ne!(
            out[1].status,
            ApplicationReturnStatus::SolveSucceeded,
            "infeasible instance must not report success"
        );
    }

    #[test]
    fn panicking_instance_does_not_poison_batch() {
        // Middle instance panics inside `eval_f`; the surrounding good
        // instances must still solve, the batch must keep input order, and
        // the panicking row must surface as `InternalError` — not unwind the
        // whole `collect()`. The default panic hook still prints the message;
        // that is fine (and useful) — only the unwind is contained.
        let good = batch(3);
        let expected: Vec<[f64; 2]> = good.iter().map(|p| p.expected()).collect();
        let probs: Vec<BoomQuad> = good
            .into_iter()
            .enumerate()
            .map(|(i, inner)| BoomQuad {
                inner,
                boom: i == 1,
            })
            .collect();
        let out = solve_nlp_batch_parallel(probs, configure);
        assert_eq!(out.len(), 3);
        assert_eq!(out[1].status, ApplicationReturnStatus::InternalError);
        assert!(
            out[1].solution.is_none(),
            "a panicked instance carries no captured solution"
        );
        for i in [0, 2] {
            assert_eq!(out[i].status, ApplicationReturnStatus::SolveSucceeded);
            let sol = out[i].solution.as_ref().expect("solution");
            assert!(
                (sol.x[0] - expected[i][0]).abs() < 1e-6
                    && (sol.x[1] - expected[i][1]).abs() < 1e-6,
                "instance {i}: got {:?}, expected {:?}",
                sol.x,
                expected[i]
            );
        }
        // The sequential path is panic-isolated too (same `solve_nlp_one`).
        let probs_seq: Vec<BoomQuad> = batch(3)
            .into_iter()
            .enumerate()
            .map(|(i, inner)| BoomQuad {
                inner,
                boom: i == 1,
            })
            .collect();
        let seq = solve_nlp_batch(probs_seq, configure);
        assert_eq!(seq[1].status, ApplicationReturnStatus::InternalError);
        assert_eq!(seq[0].status, ApplicationReturnStatus::SolveSucceeded);
        assert_eq!(seq[2].status, ApplicationReturnStatus::SolveSucceeded);
    }

    /// Warm-start chain (the MPC shape): solve a batch cold, perturb
    /// each instance's data slightly, re-solve warm-seeded from the
    /// previous solutions. Solutions must be correct, and the warm
    /// solves must not iterate more than the cold solves of the same
    /// perturbed instances.
    #[test]
    fn warm_started_batch_chains() {
        let k = 4;
        let cold = solve_nlp_batch_parallel(batch(k), configure);
        let warms: Vec<NlpWarmStart> = cold.iter().map(NlpWarmStart::from).collect();

        // Perturb each instance: shift the equality target slightly.
        let perturbed = || -> Vec<ShiftedQuad> {
            batch(k)
                .into_iter()
                .map(|mut p| {
                    p.s += 0.01;
                    p
                })
                .collect()
        };
        let warm = solve_nlp_batch_parallel_warm(perturbed(), warms, configure);
        let cold2 = solve_nlp_batch_parallel(perturbed(), configure);
        for i in 0..k {
            assert_eq!(warm[i].status, ApplicationReturnStatus::SolveSucceeded);
            let expect = perturbed()[i].expected();
            let sol = warm[i].solution.as_ref().expect("warm solution");
            assert!(
                (sol.x[0] - expect[0]).abs() < 1e-5 && (sol.x[1] - expect[1]).abs() < 1e-5,
                "instance {i}: warm solve must reach the perturbed optimum"
            );
            assert!(
                warm[i].stats.iteration_count <= cold2[i].stats.iteration_count,
                "instance {i}: warm start took {} iters vs cold {}",
                warm[i].stats.iteration_count,
                cold2[i].stats.iteration_count
            );
        }
    }

    /// A dimension-mismatched warm entry falls back to that instance's
    /// own cold start instead of failing.
    #[test]
    fn warm_start_dimension_mismatch_falls_back_cold() {
        let probs = batch(2);
        let expected: Vec<[f64; 2]> = probs.iter().map(|p| p.expected()).collect();
        let warms = vec![NlpWarmStart::default(), NlpWarmStart::default()];
        let out = solve_nlp_batch_parallel_warm(probs, warms, configure);
        for (i, r) in out.iter().enumerate() {
            assert_eq!(r.status, ApplicationReturnStatus::SolveSucceeded);
            let sol = r.solution.as_ref().expect("solution");
            assert!(
                (sol.x[0] - expected[i][0]).abs() < 1e-6
                    && (sol.x[1] - expected[i][1]).abs() < 1e-6
            );
        }
    }

    /// The pool moves parked backends between threads (HashMap behind
    /// the pool's mutex), so the backend must be `Send`; the pool
    /// itself is shared by reference from the `Sync` configure hook.
    #[test]
    fn backend_pool_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FeralBackendPool>();
        fn assert_send<T: Send>() {}
        assert_send::<pounce_feral::FeralSolverInterface>();
    }

    /// Pooled backends on an identical-structure batch: results match
    /// the fresh-backend batch within solver tolerance, and the pool
    /// ends up holding at most one parked solver per worker thread.
    #[test]
    fn pooled_backends_match_fresh_on_identical_structure() {
        let k = 8;
        let fresh = solve_nlp_batch_parallel(batch(k), configure);

        let pool = FeralBackendPool::serial(pounce_feral::FeralConfig::default());
        let pool_for_cfg = Arc::clone(&pool);
        let pooled = solve_nlp_batch_parallel(batch(k), move |_i, app| {
            let _ = app
                .options_mut()
                .set_integer_value("print_level", 0, true, false);
            install_pooled_serial_feral_backend(app, &pool_for_cfg);
        });
        assert!(
            pool.slots.lock().map(|s| !s.is_empty()).unwrap_or(false),
            "at least one worker must have parked its backend"
        );
        for i in 0..k {
            assert_eq!(pooled[i].status, ApplicationReturnStatus::SolveSucceeded);
            let pf = fresh[i].solution.as_ref().expect("fresh");
            let pp = pooled[i].solution.as_ref().expect("pooled");
            for j in 0..2 {
                assert!(
                    (pf.x[j] - pp.x[j]).abs() < 1e-6,
                    "instance {i} x[{j}]: pooled {} vs fresh {}",
                    pp.x[j],
                    pf.x[j]
                );
            }
        }
    }

    /// Ragged convergence: mix trivially-easy instances with ones that
    /// start far away; per-instance iteration counts may differ and
    /// every result still lands at its own optimum in input order.
    #[test]
    fn ragged_iteration_counts_keep_order() {
        let probs: Vec<ShiftedQuad> = (0..6)
            .map(|i| ShiftedQuad::new(10f64.powi(i - 3), 2.0, 1.0))
            .collect();
        let expected: Vec<[f64; 2]> = probs.iter().map(|p| p.expected()).collect();
        let out = solve_nlp_batch_parallel(probs, configure);
        for (i, r) in out.iter().enumerate() {
            assert_eq!(r.status, ApplicationReturnStatus::SolveSucceeded);
            let sol = r.solution.as_ref().expect("solution");
            assert!(
                (sol.x[0] - expected[i][0]).abs() < 1e-5
                    && (sol.x[1] - expected[i][1]).abs() < 1e-5,
                "instance {i}: got {:?}, expected {:?}",
                sol.x,
                expected[i]
            );
        }
    }
}
