"""Block-sequential initialization for Pyomo models (experimental).

IDAES-style initialization without hand-written initialization routines:
hold the *decision* variables at their current values, take the model's
active **equality** constraints, extract the square (well-determined)
part of the variable/constraint incidence graph (Dulmage-Mendelsohn,
via ``pyomo.contrib.incidence_analysis``), and solve it block by block
in topological order, writing the solution into ``Var.value``. The
block-by-block solve itself is delegated to Pyomo's
``solve_strongly_connected_components`` (1x1 blocks by Newton, larger
blocks by POUNCE); this module contributes the decision handling, the
square-part extraction, the seeding, and the diagnostics.

The distillation-column shape of the workflow::

    report = pyomo_pounce.block_initialize(
        model, decisions=[m.feed, m.reflux, m.boilup])
    if not report.square:
        print(report)   # names of what you forgot to specify

set the decisions, solve for a physical profile with them held
constant, then let the optimizer move them.

**Experimental.** Variables in the square subsystem are (re)computed in
place, using any existing values as Newton starting guesses. Variables
in the under- or over-determined parts (degrees of freedom you did not
flag as decisions, redundant specifications) are left untouched and
reported **by name** — pair with
:func:`pyomo_pounce.initialize_missing_values` /
:func:`pyomo_pounce.project_to_feasible` to handle the remainder (or
use the :func:`pyomo_pounce.initialize` pipeline).

:func:`block_analyze` is the analysis-only sibling: the same decision
handling and the same Dulmage-Mendelsohn partition, but nothing is
seeded, projected, or solved, and the full partition is returned as
**component objects with nothing capped** — for diagnosing a large
model, and for tooling that builds on the partition rather than on a
display-sized name list.

:func:`block_repair_plan` is the planner: given the candidate
decisions, it plans a valid specification — which candidates a square
system can hold (selected), which the equalities claim (pruned), and
which variables nothing can determine (pinned, identified
automatically) — touching nothing. :func:`block_initialize` runs the
same check on its ``decisions`` and applies the plan when the
specification needs it, so a badly specified model initializes anyway
and the report says what was repaired.

Requires ``pyomo.contrib.incidence_analysis`` (needs ``networkx`` and
``scipy``); raises ``ImportError`` with instructions otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from pyomo.core.expr.numeric_expr import DivisionExpression, NegationExpression
from pyomo.environ import value

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyomo.core.base.constraint import ConstraintData
    from pyomo.core.base.var import VarData

__all__ = [
    "block_analyze",
    "block_initialize",
    "block_repair_plan",
    "BlockAnalysisReport",
    "BlockInitReport",
    "BlockRepairPlan",
]


@dataclass
class BlockInitReport:
    """What :func:`block_initialize` did (and could not do)."""

    #: True when the equality system (after fixing decisions) is exactly
    #: square: no unmatched/underconstrained variables and no
    #: unmatched/overconstrained constraints.
    square: bool = True
    n_decisions_fixed: int = 0
    #: Pinned variables held during the solve (repair="auto" only);
    #: not counted in ``n_decisions_fixed``.
    n_pinned: int = 0
    n_blocks: int = 0
    n_1x1: int = 0
    n_subsystem_solves: int = 0
    n_vars_initialized: int = 0
    skipped_underdetermined: int = 0
    skipped_overdetermined: int = 0
    #: Names of unmatched/underconstrained variables (capped): the
    #: things you probably forgot to specify or flag as decisions.
    underconstrained_variables: List[str] = field(default_factory=list)
    #: Names of unmatched/overconstrained constraints (capped):
    #: redundant or conflicting specifications.
    overconstrained_constraints: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    #: The :class:`BlockRepairPlan` applied, when the specification
    #: actually needed repair (something pruned or pinned); None when
    #: the given decisions were used as-is.
    repair: Optional["BlockRepairPlan"] = None

    @property
    def ok(self) -> bool:
        return not self.failures

    def __str__(self) -> str:
        lines = [
            "pyomo-pounce block_initialize",
            f"  decisions fixed   : {self.n_decisions_fixed}",
        ]
        if self.n_pinned:
            lines.append(f"  pins held         : {self.n_pinned}")
        if self.repair is not None:
            lines.append(
                f"  spec repair       : {len(self.repair.decisions)} decisions "
                f"kept, {len(self.repair.pruned)} pruned, "
                f"{len(self.repair.pinned)} pinned"
            )
            if self.repair.pruned:
                lines.append(
                    "    pruned (solved for): " + _preview(self.repair.pruned)
                )
            if self.repair.pinned:
                lines.append(
                    "    pinned (held): " + _preview(self.repair.pinned)
                )
        lines += [
            f"  system square     : {self.square}",
            f"  blocks solved     : {self.n_blocks} "
            f"({self.n_1x1} by Newton 1x1, {self.n_subsystem_solves} subsystem solves)",
            f"  vars initialized  : {self.n_vars_initialized}",
            f"  left untouched    : {self.skipped_underdetermined} underdetermined, "
            f"{self.skipped_overdetermined} overdetermined",
        ]
        if self.underconstrained_variables:
            lines.append(
                "  underconstrained vars (specify or flag as decisions): "
                + ", ".join(self.underconstrained_variables)
            )
        if self.overconstrained_constraints:
            lines.append(
                "  overconstrained cons (redundant/conflicting specs): "
                + ", ".join(self.overconstrained_constraints)
            )
        for f in self.failures:
            lines.append(f"  FAILED: {f}")
        return "\n".join(lines)


def _preview(components, cap: int = 10) -> str:
    """Display-sized name list; the underlying data is never capped."""
    names = [c.name for c in components[:cap]]
    extra = len(components) - len(names)
    return ", ".join(names) + (f", ... and {extra} more" if extra > 0 else "")


@dataclass
class BlockAnalysisReport:
    """The full Dulmage-Mendelsohn partition from :func:`block_analyze`.

    Every list holds the Pyomo **component data objects** themselves
    (``VarData`` / ``ConstraintData``), in DM order, with nothing
    capped; ``str(report)`` shows a display-sized preview.
    """

    #: True when the equality system (after fixing decisions) is exactly
    #: square: no underconstrained part and no overconstrained part.
    square: bool = True
    n_decisions_fixed: int = 0
    #: Size of the analyzed system: active equality constraints and the
    #: unfixed variables appearing in them.
    n_constraints: int = 0
    n_variables: int = 0
    #: The underconstrained subsystem: variables the equalities cannot
    #: determine (the things to specify or flag as decisions), and the
    #: constraints entangled with them.
    underconstrained_variables: List[VarData] = field(default_factory=list)
    underconstrained_constraints: List[ConstraintData] = field(default_factory=list)
    #: The overconstrained subsystem: redundant or conflicting
    #: specifications, and the variables they fight over.
    overconstrained_constraints: List[ConstraintData] = field(default_factory=list)
    overconstrained_variables: List[VarData] = field(default_factory=list)
    #: The square (well-determined) part, and its block-triangular
    #: calculation order: ``variable_blocks[k]`` is solved from
    #: ``constraint_blocks[k]``, in sequence.
    square_variables: List[VarData] = field(default_factory=list)
    square_constraints: List[ConstraintData] = field(default_factory=list)
    variable_blocks: List[List[VarData]] = field(default_factory=list)
    constraint_blocks: List[List[ConstraintData]] = field(default_factory=list)

    @property
    def n_extra_degrees_of_freedom(self) -> int:
        """How many more specifications would square the under part."""
        return len(self.underconstrained_variables) - len(
            self.underconstrained_constraints
        )

    @property
    def n_extra_specifications(self) -> int:
        """How many redundant/conflicting rows the over part carries."""
        return len(self.overconstrained_constraints) - len(
            self.overconstrained_variables
        )

    @property
    def n_blocks(self) -> int:
        return len(self.variable_blocks)

    @property
    def n_1x1(self) -> int:
        return sum(1 for blk in self.variable_blocks if len(blk) == 1)

    def __str__(self) -> str:
        lines = [
            "pyomo-pounce block_analyze",
            f"  decisions fixed   : {self.n_decisions_fixed}",
            f"  equality system   : {self.n_constraints} constraints, "
            f"{self.n_variables} variables",
            f"  system square     : {self.square}",
            f"  square part       : {len(self.square_variables)} variables in "
            f"{self.n_blocks} blocks ({self.n_1x1} 1x1)",
        ]
        if self.underconstrained_variables:
            lines.append(
                f"  underconstrained  : {len(self.underconstrained_variables)} "
                f"variables, {len(self.underconstrained_constraints)} constraints "
                f"({self.n_extra_degrees_of_freedom} more specifications needed)"
            )
            lines.append(
                "    vars (specify or flag as decisions): "
                + _preview(self.underconstrained_variables)
            )
            if self.underconstrained_constraints:
                lines.append(
                    "    cons: " + _preview(self.underconstrained_constraints)
                )
        if self.overconstrained_constraints:
            lines.append(
                f"  overconstrained   : {len(self.overconstrained_constraints)} "
                f"constraints, {len(self.overconstrained_variables)} variables "
                f"({self.n_extra_specifications} redundant/conflicting)"
            )
            lines.append(
                "    cons (redundant/conflicting specs): "
                + _preview(self.overconstrained_constraints)
            )
            if self.overconstrained_variables:
                lines.append(
                    "    vars: " + _preview(self.overconstrained_variables)
                )
        return "\n".join(lines)


def _flatten_vars(vars_like):
    """Accept VarData, indexed Var containers, or iterables of either."""
    out = []
    for v in vars_like:
        if hasattr(v, "values") and callable(v.values):  # indexed container
            out.extend(v.values())
        else:
            out.append(v)
    return out


@dataclass
class BlockRepairPlan:
    """A valid specification planned by :func:`block_repair_plan`.

    A plan, not an action: nothing on the model is touched. Every list
    holds component data objects, uncapped. Applying the plan to a model
    you intend to solve means fixing ``decisions`` and ``pinned`` and
    leaving ``pruned`` free.
    """

    #: True when holding ``decisions`` and ``pinned`` makes the equality
    #: system exactly square: no loose variables, no redundant rows.
    square: bool = True
    n_constraints: int = 0
    n_variables: int = 0
    #: The candidates selected as decisions: the equalities do not
    #: contest them, so they can be held.
    decisions: List[VarData] = field(default_factory=list)
    #: The candidates the equalities claim: holding these too would
    #: overconstrain the system, so the plan solves for them instead.
    pruned: List[VarData] = field(default_factory=list)
    #: Variables the equalities provably cannot determine, identified
    #: automatically: they appear in the system, but only through edges
    #: that cannot determine them (an equation ``0 == f/g`` cannot
    #: determine a variable appearing only in ``g``). Hold them at a
    #: value of your choosing — for a flowsheet these are the loose
    #: integrators, e.g. drum levels with no weir feedback.
    pinned: List[VarData] = field(default_factory=list)
    #: Undetermined variables the plan cannot square away: a genuine
    #: modeling defect (or a missing specification).
    loose_variables: List[VarData] = field(default_factory=list)
    #: Equalities no specification can satisfy independently: redundant
    #: or conflicting rows, a model defect.
    redundant_constraints: List[ConstraintData] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "pyomo-pounce block_repair_plan",
            f"  equality system   : {self.n_constraints} constraints, "
            f"{self.n_variables} variables",
            f"  repaired square   : {self.square}",
            f"  decisions         : {len(self.decisions)} selected, "
            f"{len(self.pruned)} pruned",
        ]
        if self.pruned:
            lines.append(
                "    pruned (solved for): " + _preview(self.pruned)
            )
        if self.pinned:
            lines.append(
                f"  pinned            : {len(self.pinned)} undetermined by "
                "the equalities, hold at chosen values: "
                + _preview(self.pinned)
            )
        if self.loose_variables:
            lines.append(
                f"  loose variables   : {len(self.loose_variables)} "
                "undetermined (model defect or missing specification): "
                + _preview(self.loose_variables)
            )
        if self.redundant_constraints:
            lines.append(
                f"  redundant rows    : {len(self.redundant_constraints)} no "
                "specification can satisfy: "
                + _preview(self.redundant_constraints)
            )
        return "\n".join(lines)


def _usable_incident(con, incident):
    """The incident variables an equality can actually determine.

    An equation ``0 == f/g`` cannot determine a variable that appears
    only in the denominator ``g``: its sensitivity there vanishes
    whenever the equation is satisfied, so a matching through that edge
    is singular at every solution. This is the shape substituting
    ``dx/dt = 0`` into ``dx/dt == f/M`` produces, which is how loose
    integrators (drum levels) hide in steady-state reductions.

    The rule is deliberately shallow and conservative: only a division
    at the top of the body qualifies, so a nested ``0 == (a/b)/c``
    keeps the ``b`` edge even though ``b`` is equally undeterminable.
    False negatives only — do not make this recursive without thinking
    through the false-positive direction.
    """
    if con.lower is None or con.upper is None:
        return incident
    if value(con.lower) != 0 or value(con.upper) != 0:
        return incident
    body = con.body
    while isinstance(body, NegationExpression):
        body = body.args[0]
    if not isinstance(body, DivisionExpression):
        return incident
    from pyomo.contrib.incidence_analysis import get_incident_variables

    numerator_ids = {id(v) for v in get_incident_variables(body.args[0])}
    return [v for v in incident if id(v) in numerator_ids]


def _seed_pin(v) -> None:
    """Seed a pinned variable, never at exactly zero.

    A pin appears only in denominators of ``0 == f/g`` rows — that is
    what made every edge unusable — so zero is the one value guaranteed
    to break every equation it touches. Falls back from the bounds-aware
    seed to a nonzero in-bounds point.
    """
    _seed_var(v)
    if v.value != 0.0:
        return
    lo, hi = v.lb, v.ub
    finite = lambda b: b is not None and abs(b) < 1e19  # noqa: E731
    if finite(lo) and finite(hi):
        for frac in (0.75, 0.6):  # midpoint was zero; try off-center
            cand = lo + frac * (hi - lo)
            if cand != 0.0:
                v.set_value(cand, skip_validation=True)
                return
    elif finite(lo):
        v.set_value(lo + 2.0, skip_validation=True)  # lo + 1 was zero
    elif finite(hi):
        v.set_value(hi - 2.0, skip_validation=True)  # hi - 1 was zero
    else:
        v.set_value(1.0, skip_validation=True)


def _tiered_matching(var_adj, tiers):
    """Maximum matching, augmenting from variables in tier order.

    Greedy augmentation in priority order is lexicographically optimal
    over the transversal matroid: tier-1 coverage is maximized first,
    then tier-2 given tier-1, and so on. ``var_adj[v]`` lists the
    equation indices variable ``v`` appears in. Returns ``(eq_match,
    var_match)`` as dicts (eq index -> var index, var index -> eq
    index). Iterative, so deep alternating paths cannot hit the
    recursion limit.
    """
    eq_match = {}
    var_match = {}
    for tier in tiers:
        for v0 in tier:
            pred = {}
            seen = set()
            queue = [v0]
            free_eq = None
            while queue and free_eq is None:
                v = queue.pop()
                for e in var_adj[v]:
                    if e in seen:
                        continue
                    seen.add(e)
                    pred[e] = v
                    w = eq_match.get(e)
                    if w is None:
                        free_eq = e
                        break
                    queue.append(w)
            e = free_eq
            while e is not None:  # flip the alternating path
                v = pred[e]
                prev = var_match.get(v)
                eq_match[e] = v
                var_match[v] = e
                e = prev
    return eq_match, var_match


def block_repair_plan(model, decision_candidates=None) -> BlockRepairPlan:
    """Plan a valid specification; touch nothing, solve nothing.

    ``decision_candidates`` are the variables you would like to hold
    (for a flowsheet, the flow controls). The plan selects the subset a
    valid specification can hold — those are the ``decisions`` — and
    prunes the rest, which the equalities claim and solve for. Matching
    prefers plain variables over candidates, which provably minimizes
    the number pruned; among candidates, **earlier-listed ones are
    preferentially kept**, so the listing order is an implicit priority
    when a pruning tie could go either way. Variables the equalities
    provably cannot determine are identified automatically and come
    back ``pinned``: hold them at a value of your choosing. On a
    well-specified system the plan is a no-op: every candidate
    selected, nothing pruned or pinned.

    Args:
        model: A Pyomo model (Block). Only active equality constraints
            and unfixed variables participate.
        decision_candidates: Variables (VarData or indexed Var
            containers) you would like held. Purely structural, so
            values are not needed. Already-fixed variables are inputs,
            not part of the plan.

    Returns a :class:`BlockRepairPlan`; fix ``plan.decisions`` and
    ``plan.pinned`` (and leave ``plan.pruned`` free) to define a square
    system.
    """
    try:
        # Probe networkx explicitly: pyomo defers its optional imports, so
        # `pyomo.contrib.incidence_analysis` imports fine without it and
        # would only blow up (DeferredImportError) at first use.
        import networkx  # noqa: F401

        from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "block_repair_plan requires pyomo.contrib.incidence_analysis "
            "and its optional dependencies (pip install networkx scipy)"
        ) from e

    plan = BlockRepairPlan()

    candidates = []
    candidate_ids = set()
    for vd in _flatten_vars(decision_candidates or []):
        if not vd.fixed and id(vd) not in candidate_ids:
            candidate_ids.add(id(vd))
            candidates.append(vd)

    # TODO: block_repair_plan, block_analyze, and the initialize pipeline
    # each build their own IncidenceGraphInterface; on very large models
    # a single shared graph would remove the repeated structural pass.
    igraph = IncidenceGraphInterface(model, include_inequality=False)
    eqs = list(igraph.constraints)
    gvars = list(igraph.variables)
    plan.n_constraints = len(eqs)
    plan.n_variables = len(gvars)
    if not eqs:
        # nothing to determine: every candidate is simply an input
        plan.decisions = candidates
        return plan

    vindex = {id(v): i for i, v in enumerate(gvars)}
    raw_degree = [0] * len(gvars)
    var_adj = [[] for _ in gvars]
    for e, con in enumerate(eqs):
        # raw incidence comes from the graph already built above; the
        # expression body is only inspected on 0 == f/g shaped rows
        incident = [
            v for v in igraph.get_adjacent_to(con) if id(v) in vindex
        ]
        for v in incident:
            raw_degree[vindex[id(v)]] += 1
        for v in _usable_incident(con, incident):
            var_adj[vindex[id(v)]].append(e)

    # pinned: present in the equalities, but every edge is unusable —
    # nothing can determine these, under any matching
    pinned_idx = [
        i for i in range(len(gvars))
        if raw_degree[i] > 0 and not var_adj[i] and id(gvars[i]) not in candidate_ids
    ]

    # candidates augment in reverse listing order: greedy augmentation
    # preferentially matches (prunes) the earliest-processed vertex, so
    # reversing makes earlier-listed candidates preferentially kept
    tiers = (
        [i for i, v in enumerate(gvars)
         if id(v) not in candidate_ids and var_adj[i]],
        [vindex[id(v)] for v in reversed(candidates) if id(v) in vindex],
    )
    eq_match, var_match = _tiered_matching(var_adj, tiers)

    pruned_ids = {id(gvars[i]) for i in tiers[1] if i in var_match}
    # candidates off the equality graph are inputs nothing can contest
    plan.decisions = [v for v in candidates if id(v) not in pruned_ids]
    plan.pruned = [v for v in candidates if id(v) in pruned_ids]
    plan.pinned = [gvars[i] for i in pinned_idx]
    plan.loose_variables = [gvars[i] for i in tiers[0] if i not in var_match]
    plan.redundant_constraints = [
        eqs[e] for e in range(len(eqs)) if e not in eq_match
    ]
    plan.square = not plan.loose_variables and not plan.redundant_constraints
    return plan


def block_analyze(model, decisions=None) -> BlockAnalysisReport:
    """Partition the equality system; touch nothing, solve nothing.

    The analysis half of :func:`block_initialize` on its own: hold the
    decisions fixed, decompose the active equality constraints
    (Dulmage-Mendelsohn), and return the **full** partition — the
    underconstrained, overconstrained, and square parts as component
    objects, plus the square part's block-triangular calculation order —
    with nothing capped for display and no values read or written.

    Args:
        model: A Pyomo model (Block). Only active equality constraints
            and unfixed variables participate.
        decisions: Variables (VarData or indexed Var containers) to hold
            fixed during the analysis, then release. Purely structural,
            so unlike :func:`block_initialize` they do **not** need
            values. Already-fixed variables may be listed and stay
            fixed.

    Returns a :class:`BlockAnalysisReport`.
    """
    try:
        # Probe networkx explicitly: pyomo defers its optional imports, so
        # `pyomo.contrib.incidence_analysis` imports fine without it and
        # would only blow up (DeferredImportError) at first use.
        import networkx  # noqa: F401

        from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "block_analyze requires pyomo.contrib.incidence_analysis "
            "and its optional dependencies (pip install networkx scipy)"
        ) from e

    report = BlockAnalysisReport()

    fixed_by_us = []
    if decisions is not None:
        for vd in _flatten_vars(decisions):
            if vd.fixed:
                continue  # already an input; leave as the user set it
            vd.fix()
            fixed_by_us.append(vd)
    report.n_decisions_fixed = len(fixed_by_us)

    try:
        igraph = IncidenceGraphInterface(model, include_inequality=False)
        if not igraph.constraints:
            return report
        report.n_constraints = len(igraph.constraints)
        report.n_variables = len(igraph.variables)

        var_dm, con_dm = igraph.dulmage_mendelsohn()
        report.underconstrained_variables = list(var_dm.unmatched) + list(
            var_dm.underconstrained
        )
        report.underconstrained_constraints = list(con_dm.underconstrained)
        report.overconstrained_constraints = list(con_dm.unmatched) + list(
            con_dm.overconstrained
        )
        report.overconstrained_variables = list(var_dm.overconstrained)
        report.square_variables = list(var_dm.square)
        report.square_constraints = list(con_dm.square)
        report.square = (
            not report.underconstrained_variables
            and not report.overconstrained_constraints
        )

        if report.square_variables:
            var_blocks, con_blocks = igraph.block_triangularize(
                variables=report.square_variables,
                constraints=report.square_constraints,
            )
            report.variable_blocks = [list(blk) for blk in var_blocks]
            report.constraint_blocks = [list(blk) for blk in con_blocks]
    finally:
        for vd in fixed_by_us:
            vd.unfix()

    return report


def block_initialize(
    model,
    decisions=None,
    solver=None,
    *,
    repair: str = "auto",
    max_list: int = 10,
    tee: bool = False,
) -> BlockInitReport:
    """Fill ``Var.value`` by solving equality blocks in calculation order.

    Args:
        model: A Pyomo model (Block). Only active equality constraints
            and unfixed variables participate.
        decisions: Variables (VarData or indexed Var containers) to hold
            at their **current values** during the initialization solve,
            then release — the degrees of freedom the optimizer will
            move later. Each decision that stays held must have a value
            (``ValueError`` otherwise). Already-fixed variables may be
            listed and stay fixed. Equivalent to fixing them yourself,
            but scoped and self-documenting.
        solver: A Pyomo solver (from ``SolverFactory``) for blocks
            larger than 1x1. Default: ``SolverFactory("pounce")``,
            constructed only when such a block exists.
        repair: ``"auto"`` (default) checks and repairs the
            specification as described below; ``"off"`` holds the
            decisions exactly as given — nothing pruned, nothing
            pinned, every decision needs a value, and a non-square
            system is reported (``report.square``) instead of repaired.
        max_list: Cap on the reported name lists.
        tee: Echo block-solver output.

    With ``repair="auto"`` the specification is checked before anything
    is held. When holding the given decisions would leave the equality
    system exactly square, they are used as-is. When it would not, they
    are treated as the candidate pool of :func:`block_repair_plan`:
    conflicting decisions are pruned (solved for instead of held),
    variables the equalities provably cannot determine are pinned
    automatically, at their current values or a nonzero bounds-aware
    seed if they have none, and ``report.repair`` records the plan
    (None when the decisions were used as-is). The repair is scoped to
    this call like the decisions themselves — flags are restored, so it
    never alters the model's own specification.

    Returns a :class:`BlockInitReport`. ``report.square`` is False when
    the equality system (after the repair) is still not exactly square;
    the offending variable/constraint **names** are reported, and the
    square part is still solved best-effort. ``report.failures``
    is non-empty when the block solve itself failed (Pyomo's
    ``solve_strongly_connected_components`` fails fast, so variables in
    blocks downstream of a failure keep their seed values).
    """
    import pyomo.environ as pyo

    try:
        # Probe networkx explicitly: pyomo defers its optional imports, so
        # `pyomo.contrib.incidence_analysis` imports fine without it and
        # would only blow up (DeferredImportError) at first use.
        import networkx  # noqa: F401

        from pyomo.contrib.incidence_analysis import (
            solve_strongly_connected_components,
        )
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "block_initialize requires pyomo.contrib.incidence_analysis "
            "and its optional dependencies (pip install networkx scipy)"
        ) from e
    from pyomo.util.subsystems import TemporarySubsystemManager, create_subsystem_block

    if repair not in ("auto", "off"):
        raise ValueError(
            f"block_initialize: repair must be 'auto' or 'off', got {repair!r}"
        )

    report = BlockInitReport()

    # --- check the specification, then hold it -------------------------
    # repair="auto": on a well-specified system the plan is a no-op
    # (every decision selected), which is exactly the shipped behavior;
    # a broken one is repaired. A pruned decision needs no value (it
    # gets solved for); a pinned variable without one gets a nonzero
    # bounds-aware seed (the pin is the repair's choice, not the user's,
    # so erroring on it would be hostile). repair="off" holds the
    # decisions exactly as given and reports instead of repairing.
    plan = None
    if repair == "auto":
        plan = block_repair_plan(model, decision_candidates=decisions)
        if plan.pruned or plan.pinned:
            report.repair = plan
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

        # The square (well-determined) part of the equality system: the
        # DM decomposition separates it from remaining degrees of
        # freedom and redundant specifications — and names them. The
        # decisions are already fixed above, so none are passed on.
        analysis = block_analyze(model)
        under_vars = analysis.underconstrained_variables
        over_cons = analysis.overconstrained_constraints
        report.skipped_underdetermined = len(under_vars)
        report.skipped_overdetermined = len(over_cons)
        report.underconstrained_variables = [v.name for v in under_vars[:max_list]]
        report.overconstrained_constraints = [c.name for c in over_cons[:max_list]]
        report.square = analysis.square

        square_vars = analysis.square_variables
        square_cons = analysis.square_constraints
        if not square_vars:
            return report

        # Solve-plan statistics (the SCC solve below follows exactly
        # the analysis' block structure).
        report.n_blocks = analysis.n_blocks
        report.n_1x1 = analysis.n_1x1
        n_large = report.n_blocks - report.n_1x1

        for v in square_vars:
            if v.value is None:
                _seed_var(v)

        if n_large > 0 and solver is None:
            solver = pyo.SolverFactory("pounce")

        # Delegate the block-by-block solve to Pyomo's own machinery:
        # 1x1 blocks via calculate_variable_from_constraint, larger
        # blocks via `solver`. Variables that appear in the square
        # constraints but belong to the non-square part are temporarily
        # fixed at their current values.
        blk = create_subsystem_block(square_cons, square_vars)
        try:
            with TemporarySubsystemManager(to_fix=list(blk.input_vars.values())):
                solve_strongly_connected_components(
                    blk,
                    solver=solver,
                    solve_kwds={"tee": tee},
                )
            report.n_subsystem_solves = n_large
            report.n_vars_initialized = len(square_vars)
        except Exception as e:  # noqa: BLE001 - report, don't raise
            report.failures.append(
                f"strongly-connected-component solve failed: {e}"
            )
    finally:
        for vd in fixed_by_us:
            vd.unfix()

    return report


def _seed_var(v) -> None:
    """Bounds-aware Newton seed for a valueless variable."""
    lo, hi = v.lb, v.ub
    finite = lambda b: b is not None and abs(b) < 1e19  # noqa: E731
    if finite(lo) and finite(hi):
        v.set_value(0.5 * (lo + hi), skip_validation=True)
    elif finite(lo):
        v.set_value(lo + 1.0, skip_validation=True)
    elif finite(hi):
        v.set_value(hi - 1.0, skip_validation=True)
    else:
        v.set_value(0.0, skip_validation=True)
