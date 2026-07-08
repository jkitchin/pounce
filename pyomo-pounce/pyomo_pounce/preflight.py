"""Starting-point preflight for Pyomo models solved with POUNCE.

The Pyomo-shaped twin of ``pounce check-x0`` / ``pounce.preflight``
(see the POUNCE docs' initialization chapter). The headline trap it
catches: a ``Var`` whose ``.value`` was never set is written as **0**
into the ``.nl`` file, so a model initialized "nowhere" is actually
initialized at the origin — outside many models' meaningful range and
a domain error for ``log``, division, and friends.

Usage::

    import pyomo_pounce

    report = pyomo_pounce.preflight(model)
    if report.fatal:
        raise ValueError(str(report))
    pyomo.environ.SolverFactory("pounce").solve(model)

The report evaluates the model exactly as the NL writer will see it
(unset values temporarily treated as 0, then restored).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

__all__ = ["preflight", "PyomoPreflightReport", "initialize_missing_values"]


def _finite(b: Optional[float]) -> bool:
    return b is not None and math.isfinite(b) and abs(b) < 1e19


def _box_violation(v: float, lo: Optional[float], hi: Optional[float]) -> float:
    if v is None or not math.isfinite(v):
        return math.inf
    below = (lo - v) if _finite(lo) else -math.inf
    above = (v - hi) if _finite(hi) else -math.inf
    return max(below, above, 0.0)


@dataclass
class PyomoPreflightReport:
    """Result of :func:`preflight`. ``str(report)`` renders a text
    report; ``report.fatal`` is True when a POUNCE solve of the model
    as written would abort at the starting point."""

    n_vars: int
    n_cons: int
    unset: List[str]  # capped list of names
    n_unset: int
    bound_violations: List[Tuple[str, float, Optional[float], Optional[float], float]]
    n_bound_violations: int
    n_on_bounds: int
    con_violations: List[Tuple[str, float, Optional[float], Optional[float], float]]
    n_con_violations: int
    max_con_violation: float
    non_evaluable: List[str]  # constraint/objective names, capped
    n_non_evaluable: int
    objective: Optional[float]
    warnings: List[str] = field(default_factory=list)
    fatal: bool = False
    verdict: str = "CLEAN"

    @property
    def ok(self) -> bool:
        return not self.fatal

    def __str__(self) -> str:
        lines = ["pyomo-pounce preflight — starting-point check"]
        lines.append(f"  model      : {self.n_vars} vars, {self.n_cons} constraints")
        lines.append(
            f"  unset vars : {self.n_unset} (become 0 in the .nl file)"
            + (f"  e.g. {', '.join(self.unset)}" if self.unset else "")
        )
        obj = (
            f"{self.objective:.10e}"
            if self.objective is not None and math.isfinite(self.objective)
            else "NOT EVALUABLE"
        )
        lines.append(f"  objective  : {obj}")
        lines.append(
            f"  bounds     : {self.n_bound_violations} violated, "
            f"{self.n_on_bounds} on-bound (the solver's interior clamp moves these)"
        )
        for name, v, lo, hi, viol in self.bound_violations:
            lines.append(f"      {name}: value {v} outside [{lo}, {hi}] by {viol:.3e}")
        lines.append(
            f"  constraints: {self.n_con_violations} violated at the start "
            f"(max {self.max_con_violation:.3e}), {self.n_non_evaluable} not evaluable"
        )
        for name, v, lo, hi, viol in self.con_violations:
            lines.append(f"      {name}: body {v} vs [{lo}, {hi}], violation {viol:.3e}")
        for name in self.non_evaluable:
            lines.append(f"      {name}: NOT EVALUABLE at the starting point")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        lines.append(f"  VERDICT: {self.verdict}")
        return "\n".join(lines)


def preflight(model, feas_tol: float = 1e-6, max_list: int = 5) -> PyomoPreflightReport:
    """Check a Pyomo model's starting point as POUNCE will see it.

    Evaluates every active constraint and the objective at the current
    variable values, with unset values temporarily treated as 0 —
    exactly what Pyomo's NL writer sends to the solver — and restores
    the model untouched. Reports unset variables, values at/outside
    their bounds, per-constraint violations, and expressions that fail
    to evaluate (NaN/inf/exception: these are fatal, the solve would
    abort with ``Invalid_Number_Detected``).
    """
    import pyomo.environ as pyo
    from pyomo.environ import value

    variables = [
        v
        for v in model.component_data_objects(pyo.Var, active=True, descend_into=True)
        if not v.fixed
    ]
    constraints = list(
        model.component_data_objects(pyo.Constraint, active=True, descend_into=True)
    )

    unset_vars = [v for v in variables if v.value is None]

    # Evaluate as-written: unset -> 0, restore afterwards.
    try:
        for v in unset_vars:
            v.set_value(0.0, skip_validation=True)

        n_bound_violations = 0
        n_on_bounds = 0
        bound_violations = []
        for v in variables:
            val = v.value
            lo, hi = v.lb, v.ub
            viol = _box_violation(val, lo, hi)
            if viol > feas_tol:
                n_bound_violations += 1
                bound_violations.append((v.name, val, lo, hi, viol))
            if val is not None and math.isfinite(val):
                at_lo = _finite(lo) and abs(val - lo) <= 1e-8 * (1.0 + abs(lo))
                at_hi = _finite(hi) and abs(hi - val) <= 1e-8 * (1.0 + abs(hi))
                if at_lo or at_hi:
                    n_on_bounds += 1
        bound_violations.sort(key=lambda t: -t[4])
        del bound_violations[max_list:]

        con_violations = []
        non_evaluable = []
        n_con_violations = 0
        n_non_evaluable = 0
        max_con_violation = 0.0
        for c in constraints:
            body = value(c.body, exception=False)
            if body is None or not math.isfinite(body):
                n_non_evaluable += 1
                if len(non_evaluable) < max_list:
                    non_evaluable.append(c.name)
                continue
            lo = value(c.lower, exception=False) if c.lower is not None else None
            hi = value(c.upper, exception=False) if c.upper is not None else None
            viol = _box_violation(body, lo, hi)
            if viol > feas_tol:
                n_con_violations += 1
                if math.isfinite(viol):
                    max_con_violation = max(max_con_violation, viol)
                con_violations.append((c.name, body, lo, hi, viol))
        con_violations.sort(key=lambda t: -t[4])
        del con_violations[max_list:]

        objective = None
        obj = next(
            model.component_data_objects(pyo.Objective, active=True, descend_into=True),
            None,
        )
        obj_bad = False
        if obj is not None:
            objective = value(obj.expr, exception=False)
            obj_bad = objective is None or not math.isfinite(objective)
    finally:
        for v in unset_vars:
            v.set_value(None, skip_validation=True)

    warnings: List[str] = []
    fatal = n_non_evaluable > 0 or obj_bad
    if unset_vars:
        warnings.append(
            f"{len(unset_vars)} variable(s) have no value and will be written "
            "as 0 in the .nl file; set Var values (or run "
            "initialize_missing_values / block_initialize) before solving"
        )
    if fatal:
        warnings.append(
            "the model does not evaluate at its starting point (as written, "
            "unset values = 0); a POUNCE solve would abort with "
            "Invalid_Number_Detected"
        )
    if n_bound_violations:
        warnings.append(
            f"{n_bound_violations} variable value(s) violate their bounds; "
            "the solver clamps them inside before iterating"
        )
    if max_con_violation > 1e4:
        warnings.append(
            f"very large initial infeasibility (max constraint violation "
            f"{max_con_violation:.3e}); consider better initial values"
        )

    verdict = "FATAL" if fatal else ("WARNINGS" if warnings else "CLEAN")
    return PyomoPreflightReport(
        n_vars=len(variables),
        n_cons=len(constraints),
        unset=[v.name for v in unset_vars[:max_list]],
        n_unset=len(unset_vars),
        bound_violations=bound_violations,
        n_bound_violations=n_bound_violations,
        n_on_bounds=n_on_bounds,
        con_violations=con_violations,
        n_con_violations=n_con_violations,
        max_con_violation=max_con_violation,
        non_evaluable=non_evaluable,
        n_non_evaluable=n_non_evaluable,
        objective=objective,
        warnings=warnings,
        fatal=fatal,
        verdict=verdict,
    )


def initialize_missing_values(model, strategy: str = "midpoint") -> int:
    """Give every valueless, unfixed ``Var`` a sane starting value.

    ``strategy="midpoint"`` uses the midpoint of two finite bounds, one
    unit inside a one-sided bound, and 0 for free variables — a
    bounds-aware version of the NL writer's silent 0. ``strategy="zero"``
    makes the silent default explicit. Returns the number of variables
    initialized. Set values survive untouched either way.
    """
    import pyomo.environ as pyo

    if strategy not in ("midpoint", "zero"):
        raise ValueError(f"unknown strategy {strategy!r}")
    count = 0
    for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if v.fixed or v.value is not None:
            continue
        val = 0.0
        if strategy == "midpoint":
            lo, hi = v.lb, v.ub
            if _finite(lo) and _finite(hi):
                val = 0.5 * (lo + hi)
            elif _finite(lo):
                val = lo + 1.0
            elif _finite(hi):
                val = hi - 1.0
        v.set_value(val, skip_validation=True)
        count += 1
    return count
