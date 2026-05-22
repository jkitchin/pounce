"""Pyomo solver plugin for the POUNCE interior-point NLP solver.

Usage:
    import pyomo_pounce  # registers 'pounce' with SolverFactory
    from pyomo.environ import *
    solver = SolverFactory('pounce')
"""
from pyomo_pounce.pounce_solver import POUNCE

__all__ = ["POUNCE"]
