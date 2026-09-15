"""gh #942: a retained factor pins OS threads, so holding 128 of them
leaks ~``ncores`` native threads per differentiable solve.

# What actually leaks

Not the pinned ``ThreadPoolExecutor`` the issue suspected — that is one
thread per :class:`~pounce.jax.JaxProblem`, and the reporter's growth was
``+ncores`` *per solve*. It is the held :class:`pounce.Solver`: FERAL's
backend lazily builds its **own** rayon ``ThreadPool`` on the first
parallel ``factor()`` and parks those workers for the ``Solver``'s whole
lifetime (feral 0.17 ``numeric/solver.rs::ensure_parallel_pool``, kept
warm on purpose by its issue #19). ``pounce.jax`` retains up to
``_solver_registry_capacity`` converged solvers so the backward can reuse
the factor, so the process holds ``capacity × ncores`` parked threads.

Measured with pounce 0.11.0 on a 4-core Linux box, one ``JaxProblem``,
``jax.jit(jax.value_and_grad(...))`` over ``batched_solve``: +4 threads
per gradient step, flat at 544 once the registry filled at 128 — i.e.
``28 baseline + 129 × 4``. The issue reports the same shape on a 10-core
macOS host: +10 per step, flat at 1316. The plateau is the LRU doing its
job, not a bound anyone chose; on a 64-core host it is past 8000 threads
from a single problem, and the reporter's sweep (one problem per fit)
died at ``RuntimeError: can't start new thread``.

Reproduced without JAX at all, which is what pinned the mechanism: hold
six ``pounce.Solver`` objects built from one ``pounce.Problem`` and the
thread count climbs 7 → 31 in lockstep, drops back to 11 when they are
dropped, and does not move at all under ``FERAL_PARALLEL=0``. Under
``RAYON_NUM_THREADS=2`` it climbs by 2 per solver instead of 4.

# The three guards here, and why each one is separate

1. ``test_training_loop_registry_reaches_steady_state`` — the soft
   eviction tier. Deliberately asserts on the *registry size* rather
   than on ``/proc``: the count is the quantity the fix controls and is
   deterministic, while the thread count carries JAX's own workers.
2. ``test_soft_tier_spares_factors_no_backward_has_read`` — the branch
   the soft tier must **not** touch, per CLAUDE.md's branch rule. A
   plain count cap that ignored consumption would evict exactly the
   entries ``grad(vmap_solve)`` still needs, and the legs above would
   stay green while doing it, because a training loop's entries are all
   consumed. Mutation-checked below.
3. ``test_clear_solver_cache_releases_on_the_owning_thread`` /
   ``test_close_hands_back_the_pinned_executor`` — the teardown paths,
   where the failure was not "too many threads" but "the documented
   escape hatch silently did nothing".

# Mutation table

Each row was run; the named test is the only one that goes red.

* soft tier evicts by age alone (drop both the ``and
  self._solver_consumed`` short-circuit **and** the ``if k in
  self._solver_consumed`` guard in ``_register_solver``) →
  ``test_soft_tier_spares_factors_no_backward_has_read`` fails with
  ``missing Solver for backward (id=5)``. Note it takes *both*: with
  only the inner guard removed the trim short-circuits on an empty
  consumed set, which is why that test primes the registry with
  consumed entries first.
* ``_solver_registry_target = _solver_registry_capacity`` (no soft
  tier) → ``test_thread_budget_bounds_the_default_target`` fails. That
  is the test that pins the *default*;
  ``test_training_loop_registry_reaches_steady_state`` sets the target
  explicitly and pins the mechanism, so it survives this row by design
  — a loop long enough to exercise the derived default would be 65+
  steps on a 4-core host and longer on a bigger one.
* ``clear_solver_cache`` clears on the caller's thread (the pre-gh#942
  body) → ``test_clear_solver_cache_releases_on_the_owning_thread`` and
  ``test_sequential_problems_do_not_accumulate_threads`` fail on the
  unraisable count, three per held entry, *not* on the registry length:
  the dict empties either way, which is exactly why this shipped.
* ``close`` no longer calls ``clear_solver_cache`` →
  ``test_close_hands_back_the_pinned_executor`` fails on the registry
  length and ``test_sequential_problems_do_not_accumulate_threads`` on
  a rising thread count (measured: 97, 109, 120, 132 across four
  problems).
"""

import os
import platform
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from pounce.jax import JaxProblem  # noqa: E402
from pounce.jax._problem import (  # noqa: E402
    _HELD_FACTOR_THREAD_BUDGET,
    _held_factor_threads,
)

N_VARS = 3
M_CONS = 1
B = 4


def _f(x, p):
    return jnp.sum((x - p) ** 2)


def _g(x, p):
    return jnp.array([jnp.sum(x) - 1.0])


def _make(**kw):
    return JaxProblem(
        f=_f, g=_g, n=N_VARS, m=M_CONS, p_example=jnp.zeros(N_VARS),
        lb=jnp.zeros(N_VARS), cl=jnp.zeros(M_CONS), cu=jnp.zeros(M_CONS),
        options={"tol": 1e-8, "print_level": 0, "sb": "yes"},
        **kw,
    )


def _x0():
    return jnp.broadcast_to(jnp.full((N_VARS,), 1.0 / N_VARS), (B, N_VARS))


def _p_batch(k):
    return jnp.asarray(
        0.2 + 0.1 * ((np.arange(B * N_VARS) + k) % 5),
        dtype=jnp.float64,
    ).reshape(B, N_VARS)


def _os_thread_count():
    """Live OS threads in this process, or ``None`` where unavailable.

    ``threading.active_count()`` is useless here — it reported a flat ~4
    through the whole of the issue's 2500-step run, because the leaked
    threads are Rust-side and the ``threading`` module has never heard of
    them.
    """
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return None


_needs_proc = pytest.mark.skipif(
    platform.system() != "Linux" or not os.path.isdir("/proc/self/task"),
    reason="needs /proc/<pid>/task to count native threads",
)


def test_thread_budget_bounds_the_default_target():
    """The steady-state target is derived from an OS-thread budget.

    The pre-gh#942 rationale for 128 priced the cache in factor memory
    alone ("128 × sizeof(factor)"), which is why the thread cost went
    unnoticed. The hard ceiling stays 128 so no AD shape regresses; the
    target is what bounds the threads.
    """
    with _make() as jp:
        assert jp._solver_registry_capacity == 128
        per_factor = _held_factor_threads()
        assert per_factor >= 1
        assert jp._solver_registry_target <= jp._solver_registry_capacity
        assert jp._solver_registry_target >= 2
        # The budget is the point: steady-state held threads stay inside
        # it (barring the floor of 2, which only binds above a
        # 128-thread machine).
        held = jp._solver_registry_target * per_factor
        assert held <= max(_HELD_FACTOR_THREAD_BUDGET, 2 * per_factor)


def test_training_loop_registry_reaches_steady_state():
    """A ``jit(value_and_grad)`` loop stops accumulating held factors.

    Every step's factor is dead the moment its backward has run, and the
    soft tier is what notices. Before gh#942 the registry grew one entry
    per step to the full 128.
    """
    with _make() as jp:
        jp._solver_registry_target = 3

        @jax.jit
        @jax.value_and_grad
        def loss(p):
            return jnp.sum(jp.batched_solve(p, _x0()) ** 2)

        for k in range(12):
            jax.block_until_ready(loss(_p_batch(k)))
            # `+ 1`: the step's own entry is registered before it is read.
            assert len(jp._solver_registry) <= jp._solver_registry_target + 1

        assert len(jp._solver_registry) <= 4
        assert len(jp._solver_consumed) <= len(jp._solver_registry)


def test_soft_tier_spares_factors_no_backward_has_read():
    """``grad(vmap_solve)`` registers B factors before any backward runs.

    This is the other branch of the eviction rule, and the one a plain
    count cap gets wrong. Verified against the shipped behaviour:
    forcing ``_solver_registry_capacity = 2`` (the *hard* ceiling, which
    is blind to consumption by design) raises ``missing Solver for
    backward``, while ``_solver_registry_target = 2`` — the soft tier —
    completes and agrees with the dense backward.
    """
    p_batch = _p_batch(0)

    with _make() as jp:
        jp._solver_registry_target = 2

        @jax.jit
        @jax.value_and_grad
        def warmup(p):
            return jnp.sum(jp.batched_solve(p, _x0()) ** 2)

        # Prime the registry with *consumed* entries first. Without this
        # the soft tier never fires on the vmap shape at all (the
        # consumed set is empty, so the whole trim short-circuits) and
        # the test would pass against an eviction rule that ignores
        # consumption entirely — the exact defect it is here to catch.
        for k in range(3):
            jax.block_until_ready(warmup(_p_batch(k)))
        assert jp._solver_consumed

        @jax.jit
        @jax.value_and_grad
        def loss(p):
            return jnp.sum(jp.vmap_solve(p, _x0()) ** 2)

        val, grad = loss(p_batch)
        jax.block_until_ready(grad)
        # All B forwards ran before any backward, so no entry the vmap
        # still needed was evictable — while the primed entries, which
        # were dead, are gone.
        assert len(jp._solver_registry) <= jp._solver_registry_target + B

    with _make(factor_reuse=False) as ref:
        @jax.jit
        @jax.value_and_grad
        def loss_dense(p):
            return jnp.sum(ref.vmap_solve(p, _x0()) ** 2)

        val_d, grad_d = loss_dense(p_batch)

    assert np.isclose(float(val), float(val_d), rtol=1e-9, atol=1e-9)
    assert np.allclose(np.asarray(grad), np.asarray(grad_d),
                       rtol=1e-6, atol=1e-7)


def _record_unraisable(monkeypatch):
    """Capture unraisable exceptions instead of letting them print.

    The unsendable-drop failure happens inside a decref, where there is
    no return path to raise on, so it surfaces here and nowhere else.
    """
    seen = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)
    return seen


def test_clear_solver_cache_releases_on_the_owning_thread(monkeypatch):
    """The documented escape hatch has to actually free the factors.

    ``pounce.Solver`` is ``#[pyclass(unsendable)]``, so clearing the
    registry from the caller's thread panicked inside ``dict.clear``'s
    decref. That is unraisable: the dict emptied, every Rust-side factor
    leaked permanently, and not one thread came back.
    """
    seen = _record_unraisable(monkeypatch)

    with _make() as jp:
        @jax.jit
        @jax.value_and_grad
        def loss(p):
            return jnp.sum(jp.batched_solve(p, _x0()) ** 2)

        for k in range(3):
            jax.block_until_ready(loss(_p_batch(k)))

        before = _os_thread_count()
        jp.clear_solver_cache()
        after = _os_thread_count()

        assert len(jp._solver_registry) == 0
        assert len(jp._solver_consumed) == 0
        assert seen == [], (
            "clear_solver_cache() dropped held Solvers off their owning "
            f"thread: {[repr(a.exc_value) for a in seen]}"
        )
        if before is not None and after is not None:
            assert after <= before

    assert seen == []


@_needs_proc
def test_training_loop_thread_count_plateaus():
    """The issue's own symptom, in the units the reporter measured.

    Runs past the steady-state target and asserts the native thread
    count stops moving. Asserted as a plateau rather than an absolute
    number because JAX brings its own workers (LLVM compile threads
    appear during the first few steps, and the XLA Eigen pool is sized
    by the host).
    """
    with _make() as jp:
        jp._solver_registry_target = 3

        @jax.jit
        @jax.value_and_grad
        def loss(p):
            return jnp.sum(jp.batched_solve(p, _x0()) ** 2)

        counts = []
        for k in range(16):
            jax.block_until_ready(loss(_p_batch(k)))
            counts.append(_os_thread_count())

    # Let JAX finish standing its own machinery up before comparing.
    # Tolerance is one factor's worth of threads, floored at 4 so a
    # single-core runner's ordinary jitter cannot trip it — the leak
    # this catches is ~one factor per step across the whole tail, ten
    # times larger either way.
    settled = counts[6:]
    tol = max(4, _held_factor_threads())
    assert max(settled) - min(settled) <= tol, (
        f"native thread count still moving across the tail: {counts}"
    )


def test_close_hands_back_the_pinned_executor():
    """``close`` / the context manager release what a finished problem
    still holds, and are terminal and idempotent.

    This is the path the reporter's crash needed: their sweep built one
    projection layer — hence one ``JaxProblem``, hence one executor and
    one registry of held factors — per fit, and a dropped Python
    reference reclaims neither. There is no GC safety net behind it
    (measured: a ``JaxProblem`` that has solved once is pinned by JAX's
    callback caches and survives ``jax.clear_caches()`` plus repeated
    ``gc.collect()``), so ``close`` is the mechanism, not a convenience.
    """
    jp = _make()

    @jax.jit
    @jax.value_and_grad
    def loss(p):
        return jnp.sum(jp.batched_solve(p, _x0()) ** 2)

    for k in range(3):
        jax.block_until_ready(loss(_p_batch(k)))

    assert not jp.closed
    assert len(jp._solver_registry) > 0

    jp.close()

    assert jp.closed
    assert len(jp._solver_registry) == 0
    assert len(jp._pinned_solvers) == 0

    with pytest.raises(Exception) as excinfo:
        jp.batched_solve(_p_batch(0), _x0())
    assert "closed" in str(excinfo.value)

    # Idempotent, and a second close must not raise through the
    # already-shut-down executor.
    jp.close()
    assert jp.closed


@_needs_proc
def test_sequential_problems_do_not_accumulate_threads(monkeypatch):
    """The sweep shape: many short-lived problems in one process.

    gh#942's crash was ~7 fits in one process, each with its own
    ``JaxProblem``; the traceback came from the *next* problem failing to
    start its own executor worker. Closed problems must give their
    threads back.
    """
    seen = _record_unraisable(monkeypatch)
    baseline = _os_thread_count()
    after_each = []

    for k in range(4):
        with _make() as jp:
            @jax.jit
            @jax.value_and_grad
            def loss(p, jp=jp):
                return jnp.sum(jp.batched_solve(p, _x0()) ** 2)

            for j in range(3):
                jax.block_until_ready(loss(_p_batch(k + j)))
        after_each.append(_os_thread_count())

    assert seen == [], [repr(a.exc_value) for a in seen]
    # Flat across problems, not merely bounded: a per-problem leak shows
    # up as a rising sequence here (measured with the drain removed from
    # close(): 97, 109, 120, 132).
    assert max(after_each) - min(after_each) <= max(4, _held_factor_threads()), (
        f"threads accumulated across closed problems: baseline={baseline} "
        f"after_each={after_each}"
    )
