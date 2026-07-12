"""Pyomo solver plugin for the POUNCE interior-point NLP solver.

Usage:
    import pyomo_pounce  # registers 'pounce' with SolverFactory
    from pyomo.environ import *
    solver = SolverFactory('pounce')

Initialization helpers (see the POUNCE docs' initialization chapter):
    report = pyomo_pounce.preflight(model)         # starting-point check
    pyomo_pounce.initialize(model, decisions=[...])  # fill -> repair -> block-solve
    # ... or the individual stages:
    pyomo_pounce.initialize_missing_values(model)  # fill unset Var values
    pyomo_pounce.project_to_feasible(model)        # min-norm repair onto constraints
    pyomo_pounce.block_initialize(model, decisions=[...])  # DM-ordered equality solve

Parametric sensitivity (see pyomo_pounce.sens):
    declare_sens_param(m.p)      # flag parameters when building the model
    SolverFactory('pounce').solve(m)   # normal solve keeps the KKT factor
    gradient(m.x, wrt=m.p)       # then sensitivities are cheap backsolves
    estimate(m, [(m.p, 2.5)])
    covariance(m, n_data=len(y)) # parameter covariance for least squares
"""
from pyomo_pounce.block_init import BlockInitReport, block_initialize
from pyomo_pounce.pounce_solver import POUNCE
from pyomo_pounce.sens import (
    Covariance,
    Gradient,
    covariance,
    declare_fitted,
    declare_residual,
    declare_sens_param,
    estimate,
    gradient,
)
from pyomo_pounce.preflight import (
    PyomoPreflightReport,
    initialize_missing_values,
    preflight,
)
from pyomo_pounce.repair import InitializeReport, initialize, project_to_feasible

__all__ = [
    "POUNCE",
    "declare_sens_param",
    "declare_fitted",
    "declare_residual",
    "covariance",
    "Covariance",
    "gradient",
    "estimate",
    "Gradient",
    "preflight",
    "PyomoPreflightReport",
    "initialize_missing_values",
    "project_to_feasible",
    "initialize",
    "InitializeReport",
    "block_initialize",
    "BlockInitReport",
]
