"""Predictor--corrector continuation over a Pyomo model (pounce#608).

The Pyomo adapter for :class:`pounce.Continuation`. Everything the
driver needs already exists on this side; this module is the wiring, not
a second implementation:

* the **model update** is writing the declared mutable ``Param``s;
* the **tangent predictor** is :func:`pyomo_pounce.estimate`, which is
  the sIPOPT parametric step against the retained KKT factor
  (:func:`pyomo_pounce.retain_kkt`);
* the **warm transfer** is the solved model's own ``Var`` values plus
  the ``dual`` / ``ipopt_zL_in`` / ``ipopt_zU_in`` suffixes, which the
  solver plugin already reads under ``warm_start_init_point=yes``;
* the **corrector** is an ordinary ``SolverFactory("pounce").solve``.

Usage -- an MPC horizon shift, the case pounce#608 names::

    import pyomo.environ as pyo
    from pyomo_pounce import declare_sens_param, continuation

    m.x0 = pyo.Param([0, 1], initialize=0.0, mutable=True)
    declare_sens_param(m.x0)

    trace = continuation(
        m, [m.x0],
        [{m.x0: measured_state(k)} for k in range(20)],
        transfer=shift_horizon,          # see `shift_map` below
    )
    print(trace.report())

Read ``docs/src/continuation.md`` first. On an interior-point method a
warm-started solve along a continuation path already converges in about
one iteration, so the tangent predictor has almost no iterations left to
remove; measured on pounce's own warm-start corpus it is a wash against
a plain previous-solution warm start. What continuation buys on this
frontend is the orchestration -- transfer, seeding, event detection, and
the counters -- not a speedup.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pyomo.environ as pyo

from pounce import ContinuationStep, ContinuationTrace, StepController

from pyomo_pounce.sens import (
    declare_sens_param,
    estimate,
    release_kkt,
    retain_kkt,
)

__all__ = ["continuation", "shift_map"]

_OK = ("optimal", "locallyOptimal", "feasible")


def _param_data(param):
    """Every ``_ParamData`` under a scalar or indexed ``Param``."""
    try:
        return [param[i] for i in param]
    except TypeError:
        return [param]


def _assign(param, value):
    """Write `value` (scalar or per-index mapping/sequence) into `param`."""
    data = _param_data(param)
    if np.isscalar(value):
        for d in data:
            d.value = float(value)
        return
    if isinstance(value, dict):
        for key, val in value.items():
            param[key].value = float(val)
        return
    vals = np.asarray(value, float).ravel()
    if vals.size != len(data):
        raise ValueError(
            f"continuation: {param.name} has {len(data)} entries but the path "
            f"supplied {vals.size} values for this step"
        )
    for d, v in zip(data, vals):
        d.value = float(v)


def _flatten(params, point) -> np.ndarray:
    """The step's parameter values as one vector, for the trace record."""
    out: List[float] = []
    for p in params:
        val = point[p] if p in point else [d.value for d in _param_data(p)]
        if np.isscalar(val):
            out.append(float(val))
        elif isinstance(val, dict):
            out.extend(float(val[k]) for k in sorted(val, key=str))
        else:
            out.extend(float(v) for v in np.asarray(val, float).ravel())
    return np.asarray(out, float)


def shift_map(model, blocks, *, shift=1, fill="last"):
    """A ready-made horizon-shift transfer for a receding-horizon model.

    Returns a callable suitable as `continuation`'s `transfer`: it moves
    each indexed ``Var`` in `blocks` back by `shift` stages, so stage
    ``k`` of the next solve starts from stage ``k + shift`` of the last
    one -- the prolongation an MPC controller does by hand.

    `fill` says what the `shift` freed stages at the end get: ``"last"``
    repeats the final stage (a steady-state tail, the usual choice), or
    pass a float for a constant.

    Args:
        model: The Pyomo model.
        blocks: Indexed ``Var`` components whose leading index is the
            stage.
        shift: Stages to advance per step.
        fill: ``"last"`` or a float.
    """
    def transfer():
        for var in blocks:
            keys = sorted(var, key=lambda k: (k if isinstance(k, tuple) else (k,)))
            vals = [pyo.value(var[k], exception=False) for k in keys]
            if any(v is None for v in vals):
                continue
            n = len(keys)
            tail = vals[-1] if fill == "last" else float(fill)
            moved = [vals[min(i + shift, n - 1)] if fill == "last"
                     else (vals[i + shift] if i + shift < n else tail)
                     for i in range(n)]
            for k, v in zip(keys, moved):
                var[k].value = float(v)
    return transfer


def continuation(
    model,
    params: Sequence,
    path: Sequence[Dict],
    *,
    transfer: Optional[Callable] = None,
    predictor: str = "tangent",
    solver: str = "pounce",
    options: Optional[dict] = None,
    controller: Optional[StepController] = None,
    tee: bool = False,
) -> ContinuationTrace:
    """Trace a Pyomo model over a prescribed parameter path.

    Args:
        model: The Pyomo model. Its `params` must already be declared
            with :func:`pyomo_pounce.declare_sens_param` when
            ``predictor="tangent"``; this function declares any that are
            not, so the common case needs no extra call.
        params: The mutable ``Param`` components carrying the parameter.
        path: One entry per point: a ``{param: value}`` mapping, where
            `value` is a scalar, a ``{index: value}`` dict, or a sequence
            in the component's index order.
        transfer: Optional zero-argument callable run **after** the
            predictor writes its values and **before** the next solve.
            Use it for a horizon shift or a remesh -- anything that moves
            state between differently-shaped stages. :func:`shift_map`
            builds the receding-horizon one.
        predictor: ``"tangent"`` for the sIPOPT parametric step against
            the retained factor, or ``"zero"`` for the plain
            previous-solution transfer (the documented fallback for a
            model with no declared parameters).
        solver: Solver name for ``SolverFactory``.
        options: Extra solver options. ``warm_start_init_point`` is set
            to ``yes`` from the second point on unless overridden.
        controller: Unused for a prescribed path; accepted so the
            signature matches the generic driver's.
        tee: Stream solver output.

    Returns:
        :class:`pounce.ContinuationTrace` -- the same record type the
        generic driver returns, so a Pyomo trace and a ``Problem`` trace
        are read the same way.
    """
    if predictor not in ("tangent", "zero"):
        raise ValueError(
            f"continuation: predictor must be 'tangent' or 'zero', "
            f"got {predictor!r}"
        )
    params = list(params)
    if predictor == "tangent":
        if not params:
            raise ValueError(
                "continuation: predictor='tangent' needs at least one "
                "declared Param; pass predictor='zero' for a model without "
                "parameter sensitivities"
            )
        declare_sens_param(*params)
        retain_kkt(model)

    opts = dict(options or {})
    trace = ContinuationTrace()
    prev_active = None
    prev_point = None

    try:
        for k, point in enumerate(path):
            kind = "cold"
            if k > 0:
                kind = "zero"
                if predictor == "tangent":
                    # PREDICT: the sIPOPT parametric step against the
                    # factor the previous solve retained. `estimate`
                    # measures the perturbation from the *solve* point,
                    # so it is correct to call it before writing the new
                    # parameter values.
                    perturb = []
                    for p in params:
                        if p not in point:
                            continue
                        val = point[p]
                        for d in _param_data(p):
                            perturb.append((d, _value_for(p, d, val)))
                    try:
                        predicted = estimate(model, perturb)
                    except Exception:
                        predicted = None
                    if predicted is not None:
                        for vd, val in predicted.items():
                            if not vd.fixed:
                                vd.value = float(val)
                        kind = "tangent"
                if transfer is not None:
                    transfer()
                    kind = "transfer" if kind == "zero" else kind
                opts.setdefault("warm_start_init_point", "yes")

            for p in params:
                if p in point:
                    _assign(p, point[p])

            t0 = time.perf_counter()
            results = pyo.SolverFactory(solver).solve(
                model, tee=tee, options=opts, load_solutions=True
            )
            elapsed = time.perf_counter() - t0

            tc = str(results.solver.termination_condition)
            step = ContinuationStep(
                index=k, s=float(k), theta=_flatten(params, point),
                predictor=kind, corrected=True, solve_time=elapsed,
                status=0 if tc in _OK else -1, status_msg=tc,
                iters=int(getattr(results.solver, "statistics", {})
                          .get("iterations", -1))
                if isinstance(getattr(results.solver, "statistics", None), dict)
                else -1,
                obj=float(pyo.value(next(model.component_data_objects(
                    pyo.Objective, active=True)))),
            )

            active = _bound_activity(model)
            step.active_set_event = (
                prev_active is not None and active is not None
                and bool(np.any(active != prev_active))
            )
            prev_active = active
            trace.steps.append(step)
            trace.x.append(_primal_vector(model))

            if tc not in _OK:
                trace.status = f"solve_failed at step {k}: {tc}"
                break
            prev_point = point
    finally:
        if predictor == "tangent":
            release_kkt(model)

    del prev_point
    return trace


def _value_for(param, data, val):
    if np.isscalar(val):
        return float(val)
    if isinstance(val, dict):
        return float(val[data.index()])
    idx = list(param)
    return float(np.asarray(val, float).ravel()[idx.index(data.index())])


def _primal_vector(model) -> np.ndarray:
    return np.asarray(
        [pyo.value(v, exception=False) or 0.0
         for v in model.component_data_objects(pyo.Var, active=True)],
        dtype=float,
    )


def _bound_activity(model, tol=1e-6):
    """Boolean at-a-bound fingerprint over the model's Vars.

    The Pyomo-side counterpart of the driver's multiplier fingerprint:
    the plugin does not hand bound multipliers back on every path, so
    activity is read off the primal's distance to its bounds instead.
    """
    flags = []
    for v in model.component_data_objects(pyo.Var, active=True):
        x = pyo.value(v, exception=False)
        if x is None:
            return None
        lo, hi = v.lb, v.ub
        flags.append(lo is not None and abs(x - lo) <= tol)
        flags.append(hi is not None and abs(x - hi) <= tol)
    return np.asarray(flags, dtype=bool) if flags else None
