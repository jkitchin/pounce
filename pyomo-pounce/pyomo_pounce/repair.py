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

from pyomo_pounce.block_init import (
    BlockInitReport,
    BlockRepairPlan,
    _flatten_vars,
    _preview,
    _seed_pin,
    _seed_var,
    block_initialize,
    block_repair_plan,
)
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
    #: Pinned variables held for the pipeline (repair="auto" only);
    #: not counted in ``n_decisions_fixed``.
    n_pinned: int = 0
    n_filled: int = 0
    #: Termination condition of the projection solve, or None when the
    #: projection stage was skipped.
    projection: Optional[str] = None
    block: Optional[BlockInitReport] = None
    #: The :class:`BlockRepairPlan` applied, when the specification
    #: actually needed repair; None when the decisions were used as-is.
    repair: Optional[BlockRepairPlan] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        proj_ok = self.projection in (None, "optimal", "locallyOptimal", "feasible")
        return proj_ok and (self.block is None or self.block.ok)

    def __str__(self) -> str:
        lines = [
            "pyomo-pounce initialize (fill -> repair -> block-solve)",
            f"  decisions held: {self.n_decisions_fixed}",
        ]
        if self.n_pinned:
            lines.append(f"  pins held     : {self.n_pinned}")
        if self.repair is not None:
            lines.append(
                f"  spec repair   : {len(self.repair.pruned)} pruned "
                f"({_preview(self.repair.pruned) or 'none'}), "
                f"{len(self.repair.pinned)} pinned "
                f"({_preview(self.repair.pinned) or 'none'})"
            )
        lines += [
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
    repair: str = "auto",
    fill: str = "midpoint",
    project: bool = True,
    options: Optional[dict] = None,
    tee: bool = False,
) -> InitializeReport:
    """Fill, repair, and block-solve a model's starting point.

    ``decisions`` are held (fixed) at their current values for the
    **whole** pipeline — the projection must not drift the feed or the
    reflux you just specified — and released at the end. With
    ``repair="auto"`` (default) the specification is checked first:
    when holding the decisions as given leaves the equality system
    square, they are used as-is. When it does not, they become the
    candidate pool of :func:`block_repair_plan`: conflicting decisions
    are pruned (solved for instead of held), variables the equalities
    provably cannot determine are pinned automatically, and
    ``report.repair`` records the plan. The repair is call-scoped like
    the decisions themselves. ``repair="off"`` holds the decisions
    exactly as given — nothing pruned or pinned, every decision needs a
    value, and a non-square system is reported, not repaired. The three
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
    if repair not in ("auto", "off"):
        raise ValueError(
            f"initialize: repair must be 'auto' or 'off', got {repair!r}"
        )

    report = InitializeReport()

    # Check the specification first, on the untouched model, then hold
    # it across ALL stages: filling must not paper over a valueless
    # held decision, and the projection must not drift them. A pruned
    # decision stays free (it gets solved for); a valueless pinned
    # variable gets a nonzero bounds-aware seed before being held.
    plan = None
    if repair == "auto":
        plan = block_repair_plan(model, decision_candidates=decisions)
        if plan.pruned or plan.pinned:
            report.repair = plan
        if plan.redundant_constraints:
            report.warnings.append(
                f"{len(plan.redundant_constraints)} redundant/conflicting "
                "equalities no specification can satisfy: "
                + _preview(plan.redundant_constraints)
            )
        if plan.loose_variables:
            report.warnings.append(
                f"{len(plan.loose_variables)} variables undetermined by the "
                "equalities and not repairable: "
                + _preview(plan.loose_variables)
            )
    fixed_by_us = []
    try:
        # Fixing happens inside the try so a mid-loop ValueError cannot
        # leave earlier decisions fixed on the model.
        if plan is not None:
            to_hold = plan.decisions
        else:
            to_hold = [
                vd for vd in _flatten_vars(decisions or []) if not vd.fixed
            ]
        for vd in to_hold:
            if vd.value is None:
                raise ValueError(
                    f"decision variable {vd.name!r} has no value: a decision "
                    "must be held at a concrete value during initialization"
                )
            vd.fix()
            fixed_by_us.append(vd)
        report.n_decisions_fixed = len(fixed_by_us)
        if plan is not None:
            for vd in plan.pinned:
                if vd.value is None:
                    _seed_pin(vd)
                vd.fix()
                fixed_by_us.append(vd)
            report.n_pinned = len(plan.pinned)

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
        # The held decision set is already fixed, so block_initialize
        # sees it as plain inputs; in auto mode its own plan is then a
        # no-op, and repair="off" passes through so nothing gets pinned.
        report.block = block_initialize(
            model, solver=solver, tee=tee, repair=repair
        )
    finally:
        for vd in fixed_by_us:
            vd.unfix()
    return report
