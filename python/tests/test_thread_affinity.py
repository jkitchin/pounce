"""Cross-thread use of the evaluator objects (issue #477).

``NlProblem``, ``DenseLU`` and ``SparseLU`` used to be
``#[pyclass(unsendable)]``, which gave them pyo3's per-object thread
affinity. That default was not merely restrictive, it was *unsafe for the
host*:

* touching an instance from a thread other than its creator raised a Rust
  ``PanicException``, which derives from ``BaseException`` and therefore
  slips past every ordinary ``except Exception`` handler. In a
  branch-and-bound host this did not surface as an error at all — it
  surfaced as a wrong answer (a false ``infeasible``, which from a global
  optimizer is a wrong certificate);
* the *drop* path tripped separately, and fired even for code that never
  used an instance cross-thread: whichever thread happens to run the
  garbage collector inherits the object, and freeing it there wrote an
  unraisable ``RuntimeError`` and leaked the payload.

None of these types has any real thread affinity — ``NlTnlp`` is ``Send``,
and the LU factors are owned matrix data — so the marker is gone and the
tests below pin that. Concurrency is still the GIL's: the evaluators do
not release it, so these calls serialize rather than overlap.
"""

import gc
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import pounce
from pounce._pounce import DenseLU, SparseLU


def _quadratic():
    """``min x0² + 3·x1·x2 + sin(x0)`` s.t. ``x0 + x1 + x2 = 1``."""
    x = [pounce.NlExpr.var(i) for i in range(3)]
    obj = x[0] * x[0] + 3.0 * x[1] * x[2] + pounce.NlExpr.sin(x[0])
    return pounce.build_nl_problem(
        3, obj, constraints=[x[0] + x[1] + x[2]], g_l=[1.0], g_u=[1.0]
    )


def _run_on_thread(fn):
    """Call ``fn()`` on a fresh thread, re-raising whatever it raised.

    ``BaseException`` on purpose: a ``PanicException`` is exactly what this
    module exists to keep out, and an ``except Exception`` here would drop
    it on the floor the same way the reporting host did.
    """
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - see docstring
            box["error"] = exc

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def test_nl_problem_evaluates_on_a_foreign_thread():
    """The reproduction from the issue, method by method."""
    p = _quadratic()
    x = [1.0, 2.0, 3.0]
    expected = (
        p.objective(x),
        p.gradient(x).tolist(),
        p.constraints(x).tolist(),
        p.jacobian(x).tolist(),
        p.hessian(x).tolist(),
        p.hessian_vector_product(x, [1.0, 0.0, 0.0]).tolist(),
        repr(p),
    )

    def evaluate():
        return (
            p.objective(x),
            p.gradient(x).tolist(),
            p.constraints(x).tolist(),
            p.jacobian(x).tolist(),
            p.hessian(x).tolist(),
            p.hessian_vector_product(x, [1.0, 0.0, 0.0]).tolist(),
            repr(p),
        )

    assert _run_on_thread(evaluate) == expected


def test_nl_problem_built_on_one_thread_variant_on_another():
    """``variant`` clones the tapes, so it moves the model too."""
    p = _run_on_thread(_quadratic)
    v = _run_on_thread(lambda: p.variant(x_l=[0.0, 0.0, 0.0]))
    assert v.x_l.tolist() == [0.0, 0.0, 0.0]
    assert v.objective([1.0, 2.0, 3.0]) == p.objective([1.0, 2.0, 3.0])


def test_nl_problem_drops_cleanly_on_a_foreign_thread():
    """Collecting on a thread that did not build the model is silent.

    The old behavior wrote an unraisable ``RuntimeError`` ("is unsendable,
    but is being dropped on another thread") and leaked the tapes — for
    code that had merely let the GC run somewhere else.
    """
    holder = {"p": _run_on_thread(_quadratic)}
    unraisable = []
    previous = sys.unraisablehook
    sys.unraisablehook = unraisable.append
    try:
        del holder["p"]
        gc.collect()
    finally:
        sys.unraisablehook = previous
    assert unraisable == []


def test_shared_nl_problem_agrees_across_workers():
    """The host's actual usage: one evaluator, a pool of workers.

    Exact equality, not a tolerance — the same tape on the same input has
    no license to differ by a bit across threads.
    """
    p = _quadratic()
    x = [1.0, 2.0, 3.0]
    reference = (p.objective(x), p.gradient(x).tolist(), p.hessian(x).tolist())

    def hammer(_):
        for _ in range(500):
            got = (p.objective(x), p.gradient(x).tolist(), p.hessian(x).tolist())
            if got != reference:
                return got
        return reference

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(hammer, range(8)))
    assert results == [reference] * 8


def test_nl_problem_built_on_a_worker_solves_in_a_batch():
    """`solve_nlp_batch` clones the tape out; the pyclass itself is now
    free to have been built anywhere."""
    p = _run_on_thread(_quadratic)
    results = pounce.solve_nlp_batch([p, p.variant(x0=[0.5, 0.5, 0.5])], parallel=True)
    assert len(results) == 2
    for x, _info in results:
        assert x.shape == (3,)


@pytest.mark.parametrize("factory", ["dense", "sparse"])
def test_lu_factor_crosses_threads(factory):
    """A factor built on one thread back-solves on another, and is
    collected there without complaint."""
    values = np.array([4.0, 1.0, 1.0, 3.0])
    if factory == "dense":
        build = lambda: DenseLU(2)  # noqa: E731
    else:
        build = lambda: SparseLU(2, [0, 0, 1, 1], [0, 1, 0, 1])  # noqa: E731

    lu = _run_on_thread(build)
    lu.factor(values)
    b = np.array([1.0, 2.0])
    expected = lu.solve(b).tolist()

    assert _run_on_thread(lambda: lu.solve(b).tolist()) == expected
    assert np.allclose(np.array(expected) @ np.array([[4.0, 1.0], [1.0, 3.0]]).T, b)

    unraisable = []
    previous = sys.unraisablehook
    sys.unraisablehook = unraisable.append
    try:
        del lu
        gc.collect()
    finally:
        sys.unraisablehook = previous
    assert unraisable == []
