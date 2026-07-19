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
model, and for tooling that builds on the partition (automated
specification repair) rather than on a display-sized name list.

Requires ``pyomo.contrib.incidence_analysis`` (needs ``networkx`` and
``scipy``); raises ``ImportError`` with instructions otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyomo.core.base.constraint import ConstraintData
    from pyomo.core.base.var import VarData

__all__ = [
    "block_analyze",
    "block_initialize",
    "BlockAnalysisReport",
    "BlockInitReport",
]


@dataclass
class BlockInitReport:
    """What :func:`block_initialize` did (and could not do)."""

    #: True when the equality system (after fixing decisions) is exactly
    #: square: no unmatched/underconstrained variables and no
    #: unmatched/overconstrained constraints.
    square: bool = True
    n_decisions_fixed: int = 0
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

    @property
    def ok(self) -> bool:
        return not self.failures

    def __str__(self) -> str:
        lines = [
            "pyomo-pounce block_initialize",
            f"  decisions fixed   : {self.n_decisions_fixed}",
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
            move later. Each must have a value (``ValueError``
            otherwise). Already-fixed variables may be listed and stay
            fixed. Equivalent to fixing them yourself, but scoped and
            self-documenting.
        solver: A Pyomo solver (from ``SolverFactory``) for blocks
            larger than 1x1. Default: ``SolverFactory("pounce")``,
            constructed only when such a block exists.
        max_list: Cap on the reported name lists.
        tee: Echo block-solver output.

    Returns a :class:`BlockInitReport`. ``report.square`` is False when
    the equality system (with decisions fixed) is not exactly square;
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

    report = BlockInitReport()

    # --- hold the decisions -------------------------------------------
    fixed_by_us = []
    if decisions is not None:
        for vd in _flatten_vars(decisions):
            if vd.fixed:
                continue  # already an input; leave as the user set it
            if vd.value is None:
                raise ValueError(
                    f"decision variable {vd.name!r} has no value: a decision "
                    "must be held at a concrete value during initialization"
                )
            vd.fix()
            fixed_by_us.append(vd)
    report.n_decisions_fixed = len(fixed_by_us)

    try:
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
