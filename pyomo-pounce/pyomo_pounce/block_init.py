"""Block-sequential initialization for Pyomo models (experimental).

IDAES-style initialization without hand-written initialization routines:
take the model's active **equality** constraints, find the square
(well-determined) part of the variable/constraint incidence graph, order
it into diagonal blocks (Dulmage-Mendelsohn / block triangularization,
via ``pyomo.contrib.incidence_analysis``), and solve the blocks in
topological order, writing each block's solution into ``Var.value``.
1x1 blocks are solved with Pyomo's Newton one-liner
(``calculate_variable_from_constraint``); larger blocks become square
subsystem solves with POUNCE.

The result is a starting point that satisfies the model's sequential
"calculation order" structure — usually a dramatically better start
than zeros for flowsheet-shaped models.

**Experimental.** The API and semantics may change; in particular:
variables in the square subsystem are (re)computed in place, using any
existing values as Newton starting guesses. Variables in the under- or
over-determined parts (degrees of freedom, redundant specifications)
are left untouched — pair with
:func:`pyomo_pounce.initialize_missing_values` to fill the remainder.

Requires ``pyomo.contrib.incidence_analysis`` (needs ``networkx`` and
``scipy``); raises ``ImportError`` with instructions otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["block_initialize", "BlockInitReport"]


@dataclass
class BlockInitReport:
    """What :func:`block_initialize` did."""

    n_blocks: int = 0
    n_1x1: int = 0
    n_subsystem_solves: int = 0
    n_vars_initialized: int = 0
    skipped_underdetermined: int = 0
    skipped_overdetermined: int = 0
    failures: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def __str__(self) -> str:
        lines = [
            "pyomo-pounce block_initialize",
            f"  blocks solved     : {self.n_blocks} "
            f"({self.n_1x1} by Newton 1x1, {self.n_subsystem_solves} subsystem solves)",
            f"  vars initialized  : {self.n_vars_initialized}",
            f"  left untouched    : {self.skipped_underdetermined} underdetermined, "
            f"{self.skipped_overdetermined} overdetermined",
        ]
        for f in self.failures:
            lines.append(f"  FAILED: {f}")
        return "\n".join(lines)


def block_initialize(
    model,
    solver=None,
    *,
    max_block_solver_iter: int = 100,
    tee: bool = False,
) -> BlockInitReport:
    """Fill ``Var.value`` by solving equality blocks in calculation order.

    Args:
        model: A Pyomo model (Block). Only active equality constraints
            and unfixed variables participate; fix a variable to treat
            it as a known input.
        solver: A Pyomo solver (from ``SolverFactory``) for blocks
            larger than 1x1. Default: ``SolverFactory("pounce")``.
            Never invoked when every block is 1x1.
        max_block_solver_iter: ``max_iter`` passed to the block solver.
        tee: Echo block-solver output.

    Returns a :class:`BlockInitReport`; ``report.failures`` lists blocks
    whose solve did not converge (their variables keep whatever values
    they had).
    """
    import pyomo.environ as pyo

    try:
        # Probe networkx explicitly: pyomo defers its optional imports, so
        # `pyomo.contrib.incidence_analysis` imports fine without it and
        # would only blow up (DeferredImportError) at first use.
        import networkx  # noqa: F401

        from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "block_initialize requires pyomo.contrib.incidence_analysis "
            "and its optional dependencies (pip install networkx scipy)"
        ) from e
    from pyomo.util.calc_var_value import calculate_variable_from_constraint
    from pyomo.util.subsystems import TemporarySubsystemManager, create_subsystem_block

    report = BlockInitReport()

    igraph = IncidenceGraphInterface(model, include_inequality=False)
    if not igraph.constraints:
        return report

    # The square (well-determined) part of the equality system: DM
    # decomposition separates it from degrees of freedom and redundant
    # specifications.
    var_dm, con_dm = igraph.dulmage_mendelsohn()
    report.skipped_underdetermined = len(var_dm.unmatched) + len(
        var_dm.underconstrained
    )
    report.skipped_overdetermined = len(con_dm.unmatched) + len(con_dm.overconstrained)
    square_vars = list(var_dm.square)
    square_cons = list(con_dm.square)
    if not square_vars:
        return report

    var_blocks, con_blocks = igraph.block_triangularize(
        variables=square_vars, constraints=square_cons
    )

    for vars_blk, cons_blk in zip(var_blocks, con_blocks):
        report.n_blocks += 1
        if len(vars_blk) == 1 and len(cons_blk) == 1:
            var, con = vars_blk[0], cons_blk[0]
            if var.value is None:
                _seed_var(var)
            try:
                calculate_variable_from_constraint(var, con)
                report.n_1x1 += 1
                report.n_vars_initialized += 1
            except Exception as e:  # noqa: BLE001 - collect, keep going
                report.failures.append(f"{con.name} -> {var.name}: {e}")
            continue

        # k x k block: square subsystem solve. Other variables appearing
        # in these constraints are temporarily fixed at their current
        # values (they were computed by earlier blocks).
        if solver is None:
            solver = pyo.SolverFactory("pounce")
        blk = create_subsystem_block(cons_blk, vars_blk)
        for v in vars_blk:
            if v.value is None:
                _seed_var(v)
        blk._obj = pyo.Objective(expr=0.0)
        try:
            with TemporarySubsystemManager(to_fix=list(blk.input_vars.values())):
                results = solver.solve(
                    blk, tee=tee, options={"max_iter": max_block_solver_iter}
                )
            cond = str(results.solver.termination_condition)
            if cond not in ("optimal", "locallyOptimal", "feasible"):
                report.failures.append(
                    f"block of {len(vars_blk)} vars "
                    f"({vars_blk[0].name}, ...): termination {cond}"
                )
            else:
                report.n_subsystem_solves += 1
                report.n_vars_initialized += len(vars_blk)
        except Exception as e:  # noqa: BLE001
            report.failures.append(
                f"block of {len(vars_blk)} vars ({vars_blk[0].name}, ...): {e}"
            )
        finally:
            blk.del_component(blk._obj)

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
