"""Pyomo solver plugin for the POUNCE interior-point NLP solver.

Usage:
    import pyomo_pounce  # registers 'pounce' with SolverFactory
    from pyomo.environ import *
    solver = SolverFactory('pounce')

The same solver is registered against Pyomo's newer `pyomo.contrib.solver`
interface as well, carrying all of the extras below (see pyomo_pounce.v2):
    from pyomo.contrib.solver.common.factory import SolverFactory as SF2
    solver = SF2('pounce')             # v2 interface
    solver = SolverFactory('pounce_v2')  # v2 engine, legacy API

Initialization helpers (see the POUNCE docs' initialization chapter):
    report = pyomo_pounce.preflight(model)         # starting-point check
    pyomo_pounce.initialize(model, decisions=[...])  # fill -> repair -> block-solve
    # ... or the individual stages:
    pyomo_pounce.initialize_missing_values(model)  # fill unset Var values
    pyomo_pounce.project_to_feasible(model)        # min-norm repair onto constraints
    pyomo_pounce.block_initialize(model, decisions=[...])  # DM-ordered equality solve
    pyomo_pounce.block_analyze(model, decisions=[...])     # the DM partition only:
    #     full component lists, nothing solved, no values needed or written
    pyomo_pounce.block_repair_plan(model, decision_candidates=[...])
    #     plan a valid specification: which candidates to hold as the
    #     decisions, which to prune, and what gets pinned automatically

Predictor--corrector continuation over a parameter path (pounce#608):
    trace = pyomo_pounce.continuation(m, [m.p], path)   # repeated NLPs
    pyomo_pounce.shift_map(m, [m.x])                    # horizon-shift transfer

Parametric sensitivity (see pyomo_pounce.sens):
    declare_sens_param(m.p)      # flag parameters when building the model
    SolverFactory('pounce').solve(m)   # normal solve keeps the KKT factor
    gradient(m.x, wrt=m.p)       # then sensitivities are cheap backsolves
    estimate(m, [(m.p, 2.5)])
    covariance(m, n_data=len(y)) # parameter covariance for least squares
"""
from pyomo_pounce.block_init import (
    BlockAnalysisReport,
    BlockInitReport,
    BlockRepairPlan,
    block_analyze,
    block_initialize,
    block_repair_plan,
    structural_incidence,
)
from pyomo_pounce.continuation import continuation, shift_map
from pyomo_pounce.pounce_solver import POUNCE, check_binary
from pyomo_pounce.sens import (
    Covariance,
    EstimateReport,
    Gradient,
    covariance,
    declare_fitted,
    declare_residual,
    declare_sens_param,
    estimate,
    estimate_report,
    gradient,
    information,
    Information,
    release_kkt,
    retain_kkt,
)
from pyomo_pounce.preflight import (
    PyomoPreflightReport,
    initialize_missing_values,
    preflight,
)
from pyomo_pounce.repair import InitializeReport, initialize, project_to_feasible

# The v2 interface needs Pyomo 6.10.1+; this package supports pyomo>=6.0
# through the legacy plugin and must keep importing cleanly there. So the
# v2 registration is optional: on an older Pyomo, `import pyomo_pounce`
# still works and SolverFactory('pounce') behaves exactly as before --
# only the v2 names are absent, and `HAVE_V2_INTERFACE` says so.
#
# The probe is on Pyomo, not on `pyomo_pounce.v2`: wrapping the latter in
# try/except would also swallow a genuine ImportError raised by a bug
# inside it and report the interface as merely "unavailable". Once Pyomo
# is known to have the API, v2 is imported unguarded so real breakage is
# loud.
#
# Probe `SolutionLoader` specifically, NOT the `pyomo.contrib.solver.
# common` package. The package exists from 6.9.2, but 6.9.2-6.10.0 ship
# the older `SolutionLoaderBase`/`get_primals` API that `v2.py` does not
# target. A package-level probe therefore passes across all five of those
# releases and lets v2.py's ImportError escape -- which took `import
# pyomo_pounce` down with it, and the legacy plugin with that. This name
# is the one that actually tracks the API v2.py uses.
try:
    from pyomo.contrib.solver.common.solution_loader import (  # noqa: F401
        SolutionLoader as _probe,
    )
except ImportError:  # pragma: no cover - depends on the Pyomo version
    HAVE_V2_INTERFACE = False
else:
    del _probe
    from pyomo_pounce.v2 import LegacyPounceSolver, Pounce, PounceConfig
    HAVE_V2_INTERFACE = True

__all__ = [
    "POUNCE",
    "HAVE_V2_INTERFACE",
    "check_binary",
    "declare_sens_param",
    "declare_fitted",
    "declare_residual",
    "release_kkt",
    "retain_kkt",
    "covariance",
    "Covariance",
    "gradient",
    "estimate",
    "estimate_report",
    "continuation",
    "shift_map",
    "EstimateReport",
    "Gradient",
    "preflight",
    "PyomoPreflightReport",
    "initialize_missing_values",
    "project_to_feasible",
    "initialize",
    "InitializeReport",
    "block_initialize",
    "BlockInitReport",
    "block_analyze",
    "BlockAnalysisReport",
    "block_repair_plan",
    "BlockRepairPlan",
    "information",
    "Information",
    "structural_incidence",
]

if HAVE_V2_INTERFACE:
    __all__ += ["Pounce", "PounceConfig", "LegacyPounceSolver"]
