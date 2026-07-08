"""Starting-point repair for Pyomo models: projection and the
fill -> repair -> block-solve pipeline.

:func:`pyomo_pounce.initialize_missing_values` fills each valueless
variable independently, so the fill can be internally inconsistent
(mole fractions that do not sum to one, flows that violate a balance).
:func:`project_to_feasible` repairs that: it moves the current point
the minimum distance onto the model's own feasible set, writing the
repaired values back into ``Var.value``. Unlike the NumPy-level
``pounce.project_to_feasible`` (which projects onto *linearized*
constraints), this solves the full nonlinear projection with POUNCE.

:func:`initialize` chains the whole story::

    report = pyomo_pounce.initialize(model, decisions=[m.feed, m.reflux])
    # fill missing values -> project onto the constraints -> solve the
    # equality blocks in calculation order (block_initialize)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pyomo_pounce.block_init import BlockInitReport, _flatten_vars, block_initialize
from pyomo_pounce.preflight import initialize_missing_values

__all__ = ["project_to_feasible", "initialize", "InitializeReport"]


def project_to_feasible(
    model,
    solver=None,
    *,
    options: Optional[dict] = None,
    tee: bool = False,
) -> str:
    """Move the current point the minimum distance onto the feasible set.

    Temporarily replaces the model's objective with
    ``min sum((v - v0)**2)`` over every unfixed variable that has a
    value ``v0``, solves against the model's own (active) constraints
    and bounds with POUNCE, and restores the original objective(s). The
    repaired point lands in ``Var.value``. Valueless variables get a
    bounds-aware seed and are free to move (they carry no anchor term).

    Args:
        model: A Pyomo model. Modified in place: variable values only;
            objectives/constraints are restored exactly.
        solver: A Pyomo solver; default ``SolverFactory("pounce")``.
        options: Solver options dict (e.g. ``{"tol": 1e-8}``).
        tee: Echo solver output.

    Returns the solver termination condition as a string (``"optimal"``
    / ``"locallyOptimal"`` on success). Raises ``ValueError`` when no
    unfixed variable has a value (nothing to anchor; run
    ``initialize_missing_values`` first).
    """
    import pyomo.environ as pyo
    from pyomo_pounce.block_init import _seed_var

    variables = [
        v
        for v in model.component_data_objects(pyo.Var, active=True, descend_into=True)
        if not v.fixed
    ]
    anchored = [(v, float(v.value)) for v in variables if v.value is not None]
    if not anchored:
        raise ValueError(
            "project_to_feasible: no unfixed variable has a value to anchor "
            "the projection; run initialize_missing_values(model) first"
        )
    for v in variables:
        if v.value is None:
            _seed_var(v)

    if solver is None:
        solver = pyo.SolverFactory("pounce")

    deactivated = []
    for obj in model.component_data_objects(
        pyo.Objective, active=True, descend_into=True
    ):
        obj.deactivate()
        deactivated.append(obj)
    model._pounce_projection_objective = pyo.Objective(
        expr=sum((v - v0) ** 2 for v, v0 in anchored)
    )
    try:
        results = solver.solve(model, tee=tee, options=dict(options or {}))
        return str(results.solver.termination_condition)
    finally:
        model.del_component(model._pounce_projection_objective)
        for obj in deactivated:
            obj.activate()


@dataclass
class InitializeReport:
    """What :func:`initialize` did, stage by stage."""

    n_decisions_fixed: int = 0
    n_filled: int = 0
    #: Termination condition of the projection solve, or None when the
    #: projection stage was skipped.
    projection: Optional[str] = None
    block: Optional[BlockInitReport] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        proj_ok = self.projection in (None, "optimal", "locallyOptimal", "feasible")
        return proj_ok and (self.block is None or self.block.ok)

    def __str__(self) -> str:
        lines = [
            "pyomo-pounce initialize (fill -> repair -> block-solve)",
            f"  decisions held: {self.n_decisions_fixed}",
            f"  values filled : {self.n_filled}",
            f"  projection    : {self.projection or 'skipped'}",
        ]
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        if self.block is not None:
            lines.extend("  " + line for line in str(self.block).splitlines())
        return "\n".join(lines)


def initialize(
    model,
    decisions=None,
    solver=None,
    *,
    fill: str = "midpoint",
    project: bool = True,
    options: Optional[dict] = None,
    tee: bool = False,
) -> InitializeReport:
    """Fill, repair, and block-solve a model's starting point.

    ``decisions`` are held (fixed) at their current values for the
    **whole** pipeline — the projection must not drift the feed or the
    reflux you just specified — and released at the end. The three
    stages, each skippable:

    1. **Fill** — :func:`initialize_missing_values` gives every
       valueless unfixed variable a bounds-aware value
       (``fill="midpoint"`` or ``"zero"``; ``fill=None`` skips).
    2. **Repair** — :func:`project_to_feasible` moves the (possibly
       internally inconsistent) filled point the minimum distance onto
       the model's constraints (``project=False`` skips).
    3. **Block-solve** — :func:`block_initialize` solves the square
       equality system in calculation order, overwriting the repaired
       values with the consistent profile.

    Returns an :class:`InitializeReport`; ``report.block.square`` and
    the name lists tell you what the model is still missing.
    """
    report = InitializeReport()

    # Hold the decisions across ALL stages: filling must not paper over
    # a valueless decision, and the projection must not drift them.
    fixed_by_us = []
    if decisions is not None:
        for vd in _flatten_vars(decisions):
            if vd.fixed:
                continue
            if vd.value is None:
                raise ValueError(
                    f"decision variable {vd.name!r} has no value: a decision "
                    "must be held at a concrete value during initialization"
                )
            vd.fix()
            fixed_by_us.append(vd)
    report.n_decisions_fixed = len(fixed_by_us)

    try:
        if fill is not None:
            report.n_filled = initialize_missing_values(model, strategy=fill)
        if project:
            try:
                report.projection = project_to_feasible(
                    model, solver=solver, options=options, tee=tee
                )
                if report.projection not in ("optimal", "locallyOptimal", "feasible"):
                    report.warnings.append(
                        f"projection ended {report.projection}; continuing with "
                        "the unrepaired point"
                    )
            except ValueError as e:
                report.warnings.append(str(e))
        # Decisions are already fixed, so block_initialize sees them as
        # plain inputs; passing them again is harmless.
        report.block = block_initialize(
            model, decisions=decisions, solver=solver, tee=tee
        )
    finally:
        for vd in fixed_by_us:
            vd.unfix()
    return report
