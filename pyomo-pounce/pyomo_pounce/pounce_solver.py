"""POUNCE solver plugin for Pyomo.

Registers 'pounce' with Pyomo's SolverFactory. POUNCE speaks the AMPL
NL/SOL protocol, so Pyomo drives it through the AMPL Solver Library
interface exactly as it drives IPOPT.

The `pounce` binary is provided by the `pounce-solver` dependency,
which ships a per-platform wheel that drops the executable into the
active environment under `<venv>/bin/pounce`. Falls back to any
`pounce` already on PATH for system installs or local dev builds
(`cargo install --path crates/pounce-cli`).

Usage:
    import pyomo_pounce
    from pyomo.environ import *
    solver = SolverFactory('pounce')
    result = solver.solve(model)
"""

import shutil

from pyomo.opt import SolverFactory
from pyomo.solvers.plugins.solvers.ASL import ASL


@SolverFactory.register("pounce", doc="The POUNCE interior-point NLP solver")
class POUNCE(ASL):
    """Pyomo solver interface for POUNCE via the AMPL Solver Library protocol."""

    def __init__(self, **kwds):
        kwds["type"] = "pounce"
        super().__init__(**kwds)
        self._metasolver = False
        self.options.solver = "pounce"

    def _default_executable(self):
        # Prefer the binary bundled in the installed ``pounce-solver`` wheel,
        # whose location is deterministic (``pounce/bin/pounce`` inside the
        # package) and independent of PATH. ``shutil.which("pounce")`` alone
        # finds only the ``<venv>/bin/pounce`` console-script shim, which is
        # invisible to non-activated-environment runs (cron, IDE runners,
        # Jupyter kernels) and can be shadowed by a stale system binary. Fall
        # back to PATH for system installs and local cargo dev builds where
        # ``pounce-solver`` is not installed.
        try:
            from pounce._cli import _bundled_binary

            bundled = _bundled_binary()
            if bundled.is_file():
                return str(bundled)
        except Exception:
            pass
        return shutil.which("pounce")
