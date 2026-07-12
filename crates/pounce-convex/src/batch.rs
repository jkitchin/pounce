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
//! serial. [`solve_qp_batch_parallel`] runs the instances on rayon's global
//! pool and each worker builds its **own serial backend** from the supplied
//! `make_backend` factory. The factory is therefore expected to produce an
//! inner-serial backend (e.g. `pounce_feral::FeralSolverInterface::serial`);
//! the toggle is a per-backend setting, not global state. The serial feral
//! driver factorizes supernodes in a flat postorder loop (bounded stack),
//! so the batch needs no oversized worker stacks — unlike feral's *parallel*
//! driver, which climbs the elimination tree recursively and was the reason
//! an earlier version provisioned a custom 64 MiB-stack pool. The default
//! [`solve_qp_batch`] is sequential: predictable, contention-free, and the
//! right choice when each individual factor is large enough to parallelize
//! on its own. The `make_backend` factory is shared by reference and called
//! once per instance, so it must be `Sync`.

use crate::ipm::{QpOptions, QpWarmStart, solve_qp_ipm, solve_qp_ipm_warm};
use crate::qp::{QpProblem, QpSolution};
use pounce_linsol::SparseSymLinearSolverInterface;
use rayon::prelude::*;

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

/// Solve a batch in parallel **across instances**. Best for many small /
/// medium QPs, where cross-instance throughput beats parallelizing each
/// factor internally.
///
/// Runs on rayon's global pool. `make_backend` must be `Sync`; it is called
/// once per instance on the worker that runs it, so each worker gets its
/// **own** backend.
///
/// For the outer-parallel / inner-serial win, pass a `make_backend` that
/// builds an *inner-serial* backend (e.g.
/// `pounce_feral::FeralSolverInterface::serial`) — that keeps the only
/// parallelism across instances, avoiding the oversubscription that makes a
/// parallel-over-parallel batch slower. The toggle is a per-backend setting
/// with no global state, so concurrent feral solves on other threads are
/// unaffected. The serial feral factor uses a flat (bounded-stack)
/// supernode loop, so no oversized worker stacks are needed.
///
/// Results are returned in input order regardless of completion order.
pub fn solve_qp_batch_parallel<F>(
    probs: &[QpProblem],
    opts: &QpOptions,
    make_backend: F,
) -> Vec<QpSolution>
where
    F: Fn() -> Box<dyn SparseSymLinearSolverInterface> + Sync,
{
    probs
        .par_iter()
        .map(|prob| solve_qp_ipm(prob, opts, &make_backend))
        .collect()
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
    probs
        .par_iter()
        .zip(warms.par_iter())
        .map(|(prob, warm)| solve_qp_ipm_warm(prob, opts, warm, &make_backend))
        .collect()
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
