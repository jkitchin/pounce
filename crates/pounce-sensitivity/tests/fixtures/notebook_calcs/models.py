"""Write the models tests/notebook_calcs_over_engines.rs runs the
statistics and the steps of notebooks 26, 31 and 36 over.

    python models.py [name ...]

needs Pyomo and NumPy. Each model named, or every model when none is, is
written as <name>.nl with its .row and .col files.

decay: y = A exp(-k t) fitted to twelve points in the residual-variable
idiom, the model of notebook 26. A and k are the fitted parameters and
r the residuals, which equality rows define.

bounded: y = a exp(-k t) + b fitted to twelve points, with the limit
a + b <= cap binding at the fit and k held at its lower bound, so a
member is pinned and a row pins a combination of the others.

parametric: a model in x1, x2 and x3 with two parameters p and q held by
pin rows. At the solve x1 sits on its upper bound with a vanishing
multiplier, a kink, and the row x2 + x3 <= cap is inactive and becomes
active as q grows: the cases notebook 36's steps meet.

coupled: min 0.5 k^2 + c k y + 0.5 y^2 - 1.1 p k with p pinned at 0 and
0 <= k <= 10. At the solve k sits on its bound with a vanishing
multiplier, a kink, and c = sqrt(1 - 1e-3) couples it to y so strongly
that the classifier's ratio along k alone is 1e-3, so the classifier
returns ambiguous for k. With the curvature reduced over y, the ratio
classifies k as weakly active.

The .sol and Ipopt output excerpt beside decay, bounded and parametric
come from

    ipopt <name>.nl -AMPL linear_solver=ma57 bound_relax_factor=0 print_level=8

unidentified: the decay fit with a third fitted parameter u in
[-0.3, 0.3] that the residuals do not depend on. Its only curvature is the
objective's 1e-30 u^2, far below the noise scale, and the interior-point
method leaves it at the centre of its box, where its barrier weight
2 mu / 0.3^2 is about the curvature floor sqrt(eps) * 3.87.

held: bounded without the limit row. k still sits on its lower bound,
and no row binds, so the information of a alone conditions on k and
profiles out b.

concave: the decay fit with a third fitted parameter u in
[-0.014, 0.014] and an objective term -1.3e-7 u^2. The barrier keeps u at
the centre of its box, where its weight 2 mu / 0.014^2 = 2.6e-5 outweighs
the negative curvature, so the factor needs no inertia correction. With
that weight removed the curvature along u is -2.6e-7, and u's variance is
negative.

kinkrow: min (x - p)^2 with p pinned at 1 and the row x <= 1. At the
solve x sits on the row with a vanishing multiplier, so the row is weakly
active and its multiplier has no one derivative to read.

coupled, unidentified, held, concave and kinkrow run over pounce's solve
only.
"""
import sys

import numpy as np
import pyomo.environ as pyo


def write(m, name):
    m.write(name, format="nl", io_options={"symbolic_solver_labels": True})


def decay():
    t = np.linspace(0.0, 3.0, 12)
    y = 2.0 * np.exp(-1.3 * t) + 0.05 * np.random.default_rng(7).standard_normal(12)
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 11)
    m.A = pyo.Var(initialize=1.0)
    m.k = pyo.Var(initialize=1.0)
    m.r = pyo.Var(m.I, initialize=0.0)

    @m.Constraint(m.I)
    def res(mm, i):
        return mm.r[i] == float(y[i]) - mm.A * pyo.exp(-mm.k * float(t[i]))

    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
    return m


def bounded():
    t = np.linspace(0.0, 3.0, 12)
    y = (1.5 * np.exp(-0.6 * t) + 0.8
         + 0.03 * np.random.default_rng(3).standard_normal(12))
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 11)
    m.a = pyo.Var(initialize=1.0)
    m.b = pyo.Var(initialize=0.5)
    m.k = pyo.Var(bounds=(0.9, None), initialize=1.0)
    m.r = pyo.Var(m.I, initialize=0.0)

    @m.Constraint(m.I)
    def res(mm, i):
        return mm.r[i] == float(y[i]) - mm.a * pyo.exp(-mm.k * float(t[i])) - mm.b

    m.cap = pyo.Constraint(expr=m.a + m.b <= 2.0)
    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
    return m


def parametric():
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(bounds=(0.0, 1.0), initialize=0.5)
    m.x2 = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.x3 = pyo.Var(bounds=(-5.0, 5.0), initialize=0.0)
    m.p = pyo.Var(initialize=1.19)
    m.q = pyo.Var(initialize=0.5)
    m.pin_p = pyo.Constraint(expr=m.p == 1.19)
    m.pin_q = pyo.Constraint(expr=m.q == 0.5)
    m.cap = pyo.Constraint(expr=m.x2 + m.x3 <= 2.3)
    m.obj = pyo.Objective(
        expr=(m.x1 - m.p) ** 2 + (m.x2 - 1.5 - m.q) ** 2 + 0.2 * m.x1 * m.x2
        + (m.x3 - 0.3 * m.p) ** 2 + 0.05 * pyo.exp(0.5 * m.x3))
    return m


def coupled():
    c = float(np.sqrt(1.0 - 1e-3))
    m = pyo.ConcreteModel()
    m.k = pyo.Var(bounds=(0.0, 10.0), initialize=0.3)
    m.y = pyo.Var(initialize=0.0)
    m.p = pyo.Var(initialize=0.0)
    m.pin_p = pyo.Constraint(expr=m.p == 0.0)
    m.obj = pyo.Objective(
        expr=0.5 * m.k ** 2 + c * m.k * m.y + 0.5 * m.y ** 2 - 1.1 * m.p * m.k)
    return m


def held():
    m = bounded()
    m.del_component(m.cap)
    return m


def concave():
    m = decay()
    m.u = pyo.Var(bounds=(-0.014, 0.014), initialize=0.0)
    m.obj.set_value(m.obj.expr - 1.3e-7 * m.u ** 2)
    return m


def kinkrow():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=0.5)
    m.p = pyo.Var(initialize=1.0)
    m.pin_p = pyo.Constraint(expr=m.p == 1.0)
    m.cap = pyo.Constraint(expr=m.x <= 1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2)
    return m


def unidentified():
    m = decay()
    m.u = pyo.Var(bounds=(-0.3, 0.3), initialize=0.0)
    m.obj.set_value(m.obj.expr + 1e-30 * m.u ** 2)
    return m


MODELS = {"decay": decay, "bounded": bounded, "parametric": parametric,
          "coupled": coupled, "unidentified": unidentified, "held": held,
          "concave": concave, "kinkrow": kinkrow}
for name in sys.argv[1:] or MODELS:
    write(MODELS[name](), f"{name}.nl")
