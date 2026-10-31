"""CALIPSO.jl examples/nonconvex/ (thowell/CALIPSO.jl @ 0e3bdcf): three small
problems with known solutions, the cheapest correctness check in the corpus.

  complementarity  8 variables, 3 NCP pairs written as product equalities
                   (CALIPSO's own transcription); x* = [1, 0, 2, 0, 0, 0, 3, 6]
  maratos          the Maratos-effect example; no complementarity; x* = [1, 0]
  wachter          the Wachter-Biegler counterexample, on which a line-search
                   IPM without a feasibility restoration converges to a
                   spurious point; no complementarity; x* = [1, 0, 0.5]

`cone(x) = x` in CALIPSO is x >= 0, written here as bounds.
"""

import jax.numpy as jnp
import numpy as np

from mpcc_nlp import Model, generic_report

EMPTY = jnp.zeros(0)


def _complementarity():
    def f(x):
        return (x[0] - 5) ** 2 + (2 * x[1] + 1) ** 2

    def eq(x):
        return jnp.array([
            2 * (x[1] - 1) - 1.5 * x[1] + x[2] - 0.5 * x[3] + x[4],
            3 * x[0] - x[1] - 3.0 - x[5],
            -x[0] + 0.5 * x[1] + 4.0 - x[6],
            -x[0] - x[1] + 7.0 - x[7],
        ])

    def comp(x):
        return jnp.array([x[2] * x[5], x[3] * x[6], x[4] * x[7]])

    m = Model("complementarity", 8, f, eq, lambda x: EMPTY, comp,
              np.zeros(8), np.full(8, np.inf))
    m.pairs = lambda x: (x[2:5], x[5:8])
    m.report = generic_report(m)
    m.known_solution = np.array([1.0, 0, 2, 0, 0, 0, 3, 6])
    return m, lambda seed: np.zeros(8)


def _maratos():
    def f(x):
        return 2.0 * (x[0] ** 2 + x[1] ** 2 - 1.0) - x[0]

    m = Model("maratos", 2, f, lambda x: jnp.array([x[0] ** 2 + x[1] ** 2 - 1.0]),
              lambda x: EMPTY, lambda x: EMPTY, np.full(2, -np.inf), np.full(2, np.inf))
    m.report = generic_report(m)
    m.known_solution = np.array([1.0, 0.0])
    return m, lambda seed: np.array([2.0, 1.0])


def _wachter():
    m = Model("wachter", 3, lambda x: x[0],
              lambda x: jnp.array([x[0] ** 2 - x[1] - 1.0, x[0] - x[2] - 0.5]),
              lambda x: EMPTY, lambda x: EMPTY,
              np.array([-np.inf, 0.0, 0.0]), np.full(3, np.inf))
    m.report = generic_report(m)
    m.known_solution = np.array([1.0, 0.0, 0.5])
    return m, lambda seed: np.array([-2.0, 3.0, 1.0])


MODELS = {"complementarity": _complementarity, "maratos": _maratos, "wachter": _wachter}
