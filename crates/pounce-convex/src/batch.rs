//! Batched convex-QP solving (multiple right-hand sides / scenarios).
//!
//! Companion to the single-problem [`solve_qp_ipm`](crate::solve_qp_ipm),
//! mirroring the batched / build-once-solve-many capability the JAX and
//! sensitivity layers grew in pounce#74–#77 (parallel `batched_solve`,
//! `kkt_solve_many`): solve a *family* of convex QPs that share the same
//! structure but differ in their data, reusing one backend factory and
//! running the instances in parallel with rayon.
//!
//! Two entry points cover the two shapes that matter:
//!
//! - [`solve_qp_batch`] — a slice of independent [`QpProblem`]s (same
//!   dimensions, typically the same `P`/`A`/`G` with varying `c`/`b`/`h`/
//!   bounds, as in scenario sweeps or MPC). Each is solved end-to-end;
//!   instances run concurrently.
//! - [`solve_qp_multi_rhs`] — one fixed QP *structure* with many linear
//!   objectives `c` (the classic "multiple RHS" case: same `P`/`A`/`G`/
//!   `b`/`h`/bounds, different `c`). A thin convenience over
//!   [`solve_qp_batch`] that builds the per-`c` problems for you.
//!
//! Parallelism. Each QP solve is fully independent (its own factorization
//! and iterate), so the batch is embarrassingly parallel *across
//! instances*. There is an important interaction, though: the default
//! factorization backend (feral) is itself recursive and rayon-parallel
//! *within* a single factor. Running many instances on rayon while each
//! also parallelizes internally oversubscribes the cores (and can
//! overflow a worker stack on large batches), so it is typically *slower*
//! than either level of parallelism alone.
//!
//! The right model for a batch of many smallish QPs is **outer-parallel,
//! inner-serial**: parallelize across instances and make each factor
//! serial. [`solve_qp_batch_parallel`] does exactly that — it runs the
//! instances on a dedicated rayon pool (ample worker stack) with feral's
//! internal parallelism disabled for the duration. The default
//! [`solve_qp_batch`] is sequential: predictable, contention-free, and
//! the right choice when each individual factor is large enough to
//! parallelize on its own. The `make_backend` factory is shared by
//! reference and called once per instance, so it must be `Sync`.

use crate::ipm::{solve_qp_ipm, solve_qp_ipm_warm, QpOptions, QpWarmStart};
use crate::qp::{QpProblem, QpSolution};
use pounce_linsol::SparseSymLinearSolverInterface;
use rayon::prelude::*;

/// Run `run` under the batch's outer-parallel / inner-serial regime: a
/// dedicated rayon pool with a 64 MiB worker stack and feral's internal
/// parallelism disabled (via the process-wide `FERAL_PARALLEL`, saved and
/// restored). Shared by the cold and warm parallel batch entry points.
fn with_outer_parallel<R, G>(run: G) -> R
where
    G: Fn() -> R + Send,
    R: Send,
{
    let prev = std::env::var("FERAL_PARALLEL").ok();
    std::env::set_var("FERAL_PARALLEL", "off");
    let out = match rayon::ThreadPoolBuilder::new().stack_size(64 << 20).build() {
        Ok(pool) => pool.install(run),
        Err(_) => run(),
    };
    match prev {
        Some(v) => std::env::set_var("FERAL_PARALLEL", v),
        None => std::env::remove_var("FERAL_PARALLEL"),
    }
    out
}

/// Solve a batch of convex QPs in parallel, returning one solution per
/// input in the same order.
///
/// Solves the instances **sequentially**, reusing the one `make_backend`
/// factory. Predictable and contention-free; the right choice when each
/// individual factor is large enough to parallelize on its own (feral
/// does that internally). For many small QPs where cross-instance
/// parallelism wins, use [`solve_qp_batch_parallel`].
///
/// The problems are independent — each is solved cold. When the
/// instances share a *fixed structure* (same `A`/`G`/`P` sparsity and the
/// same set of finite bounds, varying only `c`/`b`/`h`/bound values),
/// [`QpFactorization`](crate::QpFactorization) builds the KKT symbolic
/// factor once and reuses it across solves, avoiding repeated AMD
/// ordering / symbolic analysis.
pub fn solve_qp_batch<F>(
    probs: &[QpProblem],
    opts: &QpOptions,
    mut make_backend: F,
) -> Vec<QpSolution>
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    probs
        .iter()
        .map(|prob| solve_qp_ipm(prob, opts, &mut make_backend))
        .collect()
}

/// Solve a batch in parallel **across instances**, with each instance's
/// factor run **serially** (feral's internal parallelism disabled for the
/// duration) to avoid oversubscription. Best for many small / medium QPs.
///
/// Runs on a dedicated rayon pool with a 64 MiB worker stack (feral's
/// recursive multifrontal factor can overflow the default ~2 MiB worker
/// stack on large batches). `make_backend` must be `Sync`; it is called
/// once per instance on the worker that runs it.
///
/// Results are returned in input order regardless of completion order.
///
/// **Note:** to enforce inner-serial factors this sets the process-wide
/// `FERAL_PARALLEL` environment variable for the duration of the call and
/// restores it afterward. That is a process-global side effect, so do not
/// run another feral-backed solve on a *different* thread concurrently
/// with this call if it relies on feral's internal parallelism — the two
/// would race on the variable. Sequential or batched-only use is fine.
pub fn solve_qp_batch_parallel<F>(
    probs: &[QpProblem],
    opts: &QpOptions,
    make_backend: F,
) -> Vec<QpSolution>
where
    F: Fn() -> Box<dyn SparseSymLinearSolverInterface> + Sync,
{
    with_outer_parallel(|| {
        probs
            .par_iter()
            .map(|prob| solve_qp_ipm(prob, opts, &make_backend))
            .collect()
    })
}

/// Warm-started parallel batch: like [`solve_qp_batch_parallel`] but each
/// instance is seeded from the corresponding entry of `warms` (typically
/// the previous step's solutions for a sequence of nearby batches, as in
/// receding-horizon / training-loop solving). See [`QpWarmStart`] for the
/// recentering strategy; a warm start only affects an instance's iteration
/// count, not its solution, and a per-instance dimension mismatch falls
/// back to that instance's cold start.
///
/// # Panics
/// Panics if `warms.len() != probs.len()`.
pub fn solve_qp_batch_parallel_warm<F>(
    probs: &[QpProblem],
    warms: &[QpWarmStart],
    opts: &QpOptions,
    make_backend: F,
) -> Vec<QpSolution>
where
    F: Fn() -> Box<dyn SparseSymLinearSolverInterface> + Sync,
{
    assert_eq!(
        warms.len(),
        probs.len(),
        "warms.len() ({}) must equal probs.len() ({})",
        warms.len(),
        probs.len()
    );
    with_outer_parallel(|| {
        probs
            .par_iter()
            .zip(warms.par_iter())
            .map(|(prob, warm)| solve_qp_ipm_warm(prob, opts, warm, &make_backend))
            .collect()
    })
}

/// Solve one QP structure against many linear objectives `c`
/// (sequentially; see [`solve_qp_batch`]).
///
/// All of `P`, `A`, `b`, `G`, `h`, and the bounds come from `base`; each
/// entry of `cs` (each length `base.n`) replaces `base.c`. Returns one
/// solution per `c`, in order.
///
/// This is the convex-solver analogue of the sensitivity layer's
/// `kkt_solve_many` "multiple RHS" call, but at the optimization level:
/// each RHS is a different objective, so each is a full QP solve (the KKT
/// system changes with the iterate), not a shared back-substitution.
///
/// # Panics
/// Panics if any `c` in `cs` does not have length `base.n`.
pub fn solve_qp_multi_rhs<F>(
    base: &QpProblem,
    cs: &[Vec<f64>],
    opts: &QpOptions,
    make_backend: F,
) -> Vec<QpSolution>
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let probs = multi_rhs_problems(base, cs);
    solve_qp_batch(&probs, opts, make_backend)
}

/// Parallel counterpart of [`solve_qp_multi_rhs`] (see
/// [`solve_qp_batch_parallel`] for the parallelism model).
///
/// # Panics
/// Panics if any `c` in `cs` does not have length `base.n`.
pub fn solve_qp_multi_rhs_parallel<F>(
    base: &QpProblem,
    cs: &[Vec<f64>],
    opts: &QpOptions,
    make_backend: F,
) -> Vec<QpSolution>
where
    F: Fn() -> Box<dyn SparseSymLinearSolverInterface> + Sync,
{
    let probs = multi_rhs_problems(base, cs);
    solve_qp_batch_parallel(&probs, opts, make_backend)
}

/// Build the per-objective problem list for the multi-RHS entry points.
fn multi_rhs_problems(base: &QpProblem, cs: &[Vec<f64>]) -> Vec<QpProblem> {
    for (k, c) in cs.iter().enumerate() {
        assert_eq!(
            c.len(),
            base.n,
            "cs[{k}] has length {}, expected n = {}",
            c.len(),
            base.n
        );
    }
    cs.iter()
        .map(|c| QpProblem {
            c: c.clone(),
            ..base.clone()
        })
        .collect()
}
