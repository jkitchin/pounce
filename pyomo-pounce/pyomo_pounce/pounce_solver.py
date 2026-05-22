"""POUNCE solver plugin for Pyomo.

Registers 'pounce' with Pyomo's SolverFactory. POUNCE speaks the AMPL
NL/SOL protocol, so Pyomo drives it through the AMPL Solver Library
interface exactly as it drives IPOPT.

Usage:
    import pyomo_pounce
    from pyomo.environ import *
    solver = SolverFactory('pounce')
    result = solver.solve(model)
"""
import os
import platform
import shutil

from pyomo.opt import SolverFactory
from pyomo.solvers.plugins.solvers.ASL import ASL


def _bundled_binary():
    """Path to the pounce binary bundled inside this wheel, if any."""
    name = "pounce.exe" if platform.system() == "Windows" else "pounce"
    path = os.path.join(os.path.dirname(__file__), "bin", name)
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return None


@SolverFactory.register("pounce", doc="The POUNCE interior-point NLP solver")
class POUNCE(ASL):
    """Pyomo solver interface for POUNCE via the AMPL Solver Library protocol."""

    def __init__(self, **kwds):
        kwds["type"] = "pounce"
        super().__init__(**kwds)
        self._metasolver = False
        self.options.solver = "pounce"

    def _default_executable(self):
        # Prefer the binary bundled in the wheel, fall back to PATH.
        return _bundled_binary() or shutil.which("pounce")
