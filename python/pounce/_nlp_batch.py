"""Batched NLP solving (pounce#126): solve N independent NLPs, one
result per input, in input order.

Two kinds of input, one entry point:

* ``NlProblem`` instances (from :func:`pounce.read_nl` /
  :meth:`NlProblem.variant`) are **native-Rust** evaluators — reverse-
  mode AD tapes with no Python callbacks. These solve in parallel on a
  Rayon thread pool with the GIL fully released (outer-parallel across
  instances, inner-serial linear-solver factor per worker).
* Callback-based :class:`pounce.Problem` objects also solve in
  parallel (phase 2): each worker owns its solve and re-acquires the
  GIL transiently for every ``objective`` / ``gradient`` /
  ``constraints`` / ``jacobian`` / ``hessian`` call. The GIL
  serializes the *Python* share of the work, so the speedup scales
  with the Rust/Python work ratio — large instances whose KKT
  factorizations dominate parallelize well; tiny problems whose
  callbacks dominate won't. Native ``NlProblem`` batches don't have
  this ceiling.

Both paths support warm-start chaining: pass a previous batch's
results as ``warms=`` to seed each instance (receding-horizon MPC,
parameter continuation, B&B dives).
"""

from __future__ import annotations

from . import _pounce


def solve_nlp_batch(problems, x0s=None, options=None, parallel=True,
                    warms=None, share_structure=False):
    """Solve a batch of independent NLPs; one ``(x, info)`` per input.

    Parameters
    ----------
    problems : sequence of NlProblem, or sequence of Problem
        The instances to solve. All entries must be of the same kind
        (mixed batches raise ``TypeError``). See the module docstring
        for the two execution models.
    x0s : sequence of array-like, optional
        Per-instance starting points. Required for ``Problem`` inputs
        (their ``solve`` takes ``x0`` explicitly). For ``NlProblem``
        inputs this overrides each model's ``.nl`` starting point
        (entries may be ``None`` to keep a model's own ``x0``).
    options : dict, optional
        IPOPT-style options applied to every instance, with the same
        value coercion as ``Problem.add_option`` (``True``/``False``
        become ``"yes"``/``"no"``). For ``Problem`` inputs each
        instance's own ``add_option`` settings are applied first, then
        this overlay. ``print_level`` defaults to 0 for the batch —
        interleaved per-iteration tables from concurrent workers are
        noise — but an explicit ``print_level`` wins.
    parallel : bool, default True
        Solve instances concurrently on the Rayon pool (each worker
        using an inner-serial factorization). With ``False``,
        instances solve one at a time and each factorization may
        parallelize internally — better for a few large instances.
    warms : sequence of (x, info), optional
        Previous results (as returned by this function), one per
        instance, to warm-start from: the warm ``x`` and the
        ``mult_g`` / ``mult_x_L`` / ``mult_x_U`` duals seed each
        solve, the previous barrier parameter (``info["mu"]``) is
        threaded into ``mu_init``, and ``warm_start_init_point=yes``
        is forced. A warm start only affects iteration counts, never
        the solution.
    share_structure : bool, default False
        Opt-in for batches whose instances share their KKT sparsity
        (parametric sweeps, multi-start, B&B siblings): each worker
        keeps its factorization backend across instances, so the
        symbolic analysis (fill-reducing ordering, supernode
        structure) runs once per worker instead of once per instance.
        Always correct — a sparsity change just triggers a fresh
        analysis — but solver state carried across instances means
        results are within tolerance of, not bit-identical to, the
        default fresh-backend solves.

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
            problems, options=options, parallel=parallel, warms=warms,
            share_structure=share_structure,
        )
    if any(native):
        raise TypeError(
            "solve_nlp_batch: mixed NlProblem and Problem inputs are not "
            "supported; split the batch by kind"
        )

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
    return _pounce.solve_problem_batch(
        problems, x0s, options=options, parallel=parallel, warms=warms,
        share_structure=share_structure,
    )
