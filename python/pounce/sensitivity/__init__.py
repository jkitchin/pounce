"""Post-optimal sensitivity analysis against a held KKT factorization.

One converged solve, and every question below is a back-solve against
the factorization it left behind -- no re-solve, no finite differences.

    import pounce
    from pounce.sensitivity import SensSession, solution, covariance

    nl = pounce.read_nl("model.nl")
    solver = pounce.Solver(problem)
    solver.solve(x0)

    sess = SensSession(nl, solver, nl.var_names, nl.con_names,
                       pins={"p": 4})
    x_new = solution(sess, [4], [0.05])          # the moved solution
    rep   = solution_report(sess, [4], [0.05])   # what it did about bounds

A parameter is addressed by the full-g row of the defining equality it
is pinned by; a fitted parameter or a residual by its full-x column or
full-g row. Nothing here knows about a modelling layer:
:mod:`pyomo_pounce` is one caller, and builds the same session from
Pyomo components.

See `docs/src/sensitivity.md`.
"""
from ._session import (
    NlBridge,
    SensSession,
    objective_sign,
    row_index,
    solve_for_sensitivity,
    user_row_names,
)
from ._stats import (
    DEFAULT_HINTS,
    Covariance,
    Information,
    covariance,
    information,
)
from ._step import (
    ActiveSetChange,
    SolutionReport,
    active_set_changes,
    check_margins,
    refuse_on_pdpert,
    solution,
    solution_report,
    weakly_active,
)

__all__ = [
    "SensSession",
    "solve_for_sensitivity",
    "NlBridge",
    "row_index",
    "objective_sign",
    "user_row_names",
    "solution",
    "solution_report",
    "active_set_changes",
    "SolutionReport",
    "ActiveSetChange",
    "covariance",
    "information",
    "Covariance",
    "Information",
    "DEFAULT_HINTS",
    "weakly_active",
    "check_margins",
    "refuse_on_pdpert",
]
