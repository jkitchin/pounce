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
use pounce_common::types::Number;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{
    BoundsInfo, IpoptCq, IpoptData, IterStats, Linearity, MetaData, NlpInfo, ScalingRequest,
    Solution, SparsityRequest, StartingPoint, TNLP,
};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

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

/// Solve one instance with a fresh, caller-configured application.
fn solve_nlp_one<T, C>(tnlp: T, configure: &mut C) -> NlpBatchResult
where
    T: TNLP + 'static,
    C: FnMut(&mut IpoptApplication),
{
    let mut app = IpoptApplication::new();
    configure(&mut app);
    let cap = Rc::new(RefCell::new(CaptureTnlp {
        inner: tnlp,
        captured: None,
    }));
    let status = app.optimize_tnlp(Rc::clone(&cap) as Rc<RefCell<dyn TNLP>>);
    let stats = app.statistics();
    let solution = cap.borrow_mut().captured.take();
    NlpBatchResult {
        status,
        solution,
        stats,
    }
}

/// Solve a batch of independent NLPs **sequentially**, returning one
/// result per input in input order. `configure` is called once per
/// instance on a fresh [`IpoptApplication`] (set options / backend /
/// restoration there — see the module docs). Predictable and
/// contention-free; the right choice when each instance is large
/// enough that the linear-solver backend parallelizes internally.
pub fn solve_nlp_batch<T, C>(problems: Vec<T>, mut configure: C) -> Vec<NlpBatchResult>
where
    T: TNLP + 'static,
    C: FnMut(&mut IpoptApplication),
{
    problems
        .into_iter()
        .map(|t| solve_nlp_one(t, &mut configure))
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
/// instance). For the outer-parallel / inner-serial win, have
/// `configure` install an inner-serial backend —
/// [`install_serial_feral_backend`] after setting options.
pub fn solve_nlp_batch_parallel<T, C>(problems: Vec<T>, configure: C) -> Vec<NlpBatchResult>
where
    T: TNLP + Send + 'static,
    C: Fn(&mut IpoptApplication) + Sync,
{
    problems
        .into_par_iter()
        .map(|t| solve_nlp_one(t, &mut |app: &mut IpoptApplication| configure(app)))
        .collect()
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

    fn configure(app: &mut IpoptApplication) {
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
