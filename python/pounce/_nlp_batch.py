"""Batched NLP solving (pounce#126): solve N independent NLPs, one
result per input, in input order.

Two kinds of input, two execution paths:

* ``NlProblem`` instances (from :func:`pounce.read_nl` /
  :meth:`NlProblem.variant`) are **native-Rust** evaluators — reverse-
  mode AD tapes with no Python callbacks. These solve in parallel on a
  Rayon thread pool with the GIL released (outer-parallel across
  instances, inner-serial linear-solver factor per worker).
* Callback-based :class:`pounce.Problem` objects evaluate through
  Python, so every objective/gradient/constraint call needs the GIL —
  parallelizing them buys nothing and risks deadlock. They are solved
  **sequentially**, documented here rather than discovered the hard
  way. (True parallel callback batching is pounce#126 phase 2.)
"""

from __future__ import annotations

from . import _pounce


def solve_nlp_batch(problems, x0s=None, options=None, parallel=True):
    """Solve a batch of independent NLPs; one ``(x, info)`` per input.

    Parameters
    ----------
    problems : sequence of NlProblem, or sequence of Problem
        The instances to solve. All entries must be of the same kind.
        ``NlProblem`` entries (native, from :func:`read_nl` /
        :meth:`NlProblem.variant`) run in parallel; ``Problem``
        entries (Python callbacks) run sequentially — see the module
        docstring.
    x0s : sequence of array-like, optional
        Per-instance starting points. Required for ``Problem`` inputs
        (their ``solve`` takes ``x0`` explicitly). For ``NlProblem``
        inputs this overrides each model's ``.nl`` starting point
        (entries may be ``None`` to keep a model's own ``x0``).
    options : dict, optional
        IPOPT-style options applied identically to every instance,
        with the same value coercion as ``Problem.add_option``
        (``True``/``False`` become ``"yes"``/``"no"``). The native
        path defaults ``print_level`` to 0 — interleaved per-iteration
        tables from concurrent workers are noise — but an explicit
        ``print_level`` wins.
    parallel : bool, default True
        Native path only: solve instances concurrently on the Rayon
        pool (each worker using an inner-serial factorization). With
        ``False``, instances solve one at a time and each
        factorization may parallelize internally — better for a few
        large instances. Ignored for ``Problem`` inputs (always
        sequential).

    Returns
    -------
    list of (numpy.ndarray, dict)
        Per instance, in input order: the final iterate ``x`` and an
        info dict matching ``Problem.solve``'s layout (``status``,
        ``status_msg``, ``obj_val``, ``g``, ``mult_g``, ``mult_x_L``,
        ``mult_x_U``, ``iter_count``, ...).
    """
    problems = list(problems)
    if not problems:
        return []
    if x0s is not None:
        x0s = list(x0s)
        if len(x0s) != len(problems):
            raise ValueError(
                f"solve_nlp_batch: got {len(problems)} problems but "
                f"{len(x0s)} starting points"
            )

    native = [isinstance(p, _pounce.NlProblem) for p in problems]
    if all(native):
        if x0s is not None:
            problems = [
                p if x0 is None else p.variant(x0=x0)
                for p, x0 in zip(problems, x0s)
            ]
        return _pounce.solve_nlp_batch(
            problems, options=options, parallel=parallel
        )
    if any(native):
        raise TypeError(
            "solve_nlp_batch: mixed NlProblem and Problem inputs are not "
            "supported; split the batch by kind"
        )

    # Callback-based Problem objects: sequential fallback (the GIL
    # serializes every callback, so there is no parallel win to offer;
    # see pounce#126 phase 2).
    if not all(isinstance(p, _pounce.Problem) for p in problems):
        raise TypeError(
            "solve_nlp_batch: expected a sequence of pounce.NlProblem or "
            "pounce.Problem instances"
        )
    if x0s is None:
        raise ValueError(
            "solve_nlp_batch: x0s is required for Problem inputs (their "
            "solve() takes an explicit starting point)"
        )
    results = []
    for p, x0 in zip(problems, x0s):
        if options:
            for name, value in options.items():
                p.add_option(name, value)
        results.append(p.solve(x0))
    return results
