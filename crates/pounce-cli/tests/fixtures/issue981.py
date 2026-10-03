"""Regenerate the three gh #981 fixtures.

    python3 issue981.py            # writes the .nl files next to this script

Needs only Pyomo (its `.nl` writer is native — no AMPL involved). The models
are the reproductions from gh #981, which were filed against discopt's
`Model.solve(solver="pounce")`; they are transcribed term for term so the
`.nl` is checkable against the issue text.

* `issue981_cstr_dup_row.nl` — a CSTR (A -> B -> C, Arrhenius rates) with
  one constraint duplicated: `balA_again` is exactly twice `balA`, so the
  equality Jacobian is rank-deficient at every point. `delta_c` is the
  right perturbation for that, and gh #981 finding 1 is the `delta_c`
  walk-back (gh #592) withdrawing it anyway.
* `issue981_wachter_biegler.nl` — Wächter & Biegler's infeasible example,
  `min x1  s.t.  x1^2 - x2 - 1 = 0,  x1 - x3 - 0.5 = 0,  x2, x3 >= 0`, from
  `(-2, 1, 1)`. The infeasibility measure has its local minimizer at
  `x1 = -1` with violation 1.5; gh #981 finding 4 is the returned point.
* `issue981_water_main.nl` — a three-pipe water main written with flows in
  L/s instead of m^3/s, which makes it badly scaled and infeasible. Its main
  rows print head violations of order 1e5 m while the row-scaled residual is
  of order 0.2; gh #981 finding 5 is the restoration rows printing the
  latter.
* `issue981_opf3_overload.nl` — a 3-bus polar AC OPF (lines 0.05+0.25j,
  generators at buses 0 and 1, a 3.6 + 0.5j p.u. load at bus 2) from
  V = 0.95, θ = (0, 0.4, −0.4), Pg = 1.2, Qg = 0. The load exceeds what the
  network can carry, so the model is infeasible; the filter line search says
  so in 22 iterations. The issue's own 3-bus case was not published, so this
  one was found by a grid search for the symptom it reported: under
  `line_search_method=penalty` an unbroken run of `S` soft-restoration steps
  to `max_iter` with no restoration call.
* `issue981_circle_parabola.nl` — `(x-3)² + y² = 1`, `y = x²`,
  `min x² + y²`, from the origin. Infeasible (the parabola comes no closer
  than 2.24 to the circle's centre). Restoration recovers to the same
  stationary point of the infeasibility 66 times (1361 iterations) before the
  verdict — the issue's "126 restoration calls" observation. NOT fixed: a
  "same recovery three times running" exit cut this to 3 calls, but the same
  signature appears on the feasible `square_flowsheet_resto` (L-BFGS), and
  with `feral_increase_quality_retry=no` it turned that model's honest
  `Maximum_Iterations_Exceeded` into a false `Infeasible_Problem_Detected`.
  Kept as the reproduction for whoever takes it up.
"""

import math
import os

from pyomo.environ import (
    ConcreteModel, Constraint, ConstraintList, Objective, Var, cos, exp,
    maximize, sin,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def cstr_dup_row():
    cA0, k10, E1, k20, E2, R = 2.0, 1.0e6, 50_000.0, 5.0e3, 35_000.0, 8.314
    m = ConcreteModel()
    m.cA = Var(bounds=(0, 2), initialize=1.0)
    m.cB = Var(bounds=(0, 2), initialize=1.0)
    m.tau = Var(bounds=(0.1, 60), initialize=10.0)
    m.T = Var(bounds=(300, 360), initialize=300.6)
    k1 = k10 * exp(-E1 / (R * m.T))
    k2 = k20 * exp(-E2 / (R * m.T))
    m.balA = Constraint(expr=cA0 - m.cA - m.tau * k1 * m.cA == 0)
    m.balB = Constraint(expr=-m.cB + m.tau * (k1 * m.cA - k2 * m.cB) == 0)
    # The redundant copy, with k1 spelled out again (as in the issue).
    m.balA_again = Constraint(
        expr=2 * (cA0 - m.cA - m.tau * k10 * exp(-E1 / (R * m.T)) * m.cA) == 0
    )
    m.obj = Objective(expr=10 * m.cB - 0.02 * m.tau, sense=maximize)
    return m


def wachter_biegler():
    m = ConcreteModel()
    m.x1 = Var(bounds=(-10, 10), initialize=-2.0)
    m.x2 = Var(bounds=(0, 10), initialize=1.0)
    m.x3 = Var(bounds=(0, 10), initialize=1.0)
    m.obj = Objective(expr=m.x1)
    m.c1 = Constraint(expr=m.x1**2 - m.x2 - 1 == 0)
    m.c2 = Constraint(expr=m.x1 - m.x3 - 0.5 == 0)
    return m


def water_main():
    CHW = 130.0
    Lp = [1500.0, 1000.0, 1200.0]
    Qp = [1000 * q for q in (0.14, 0.08, 0.06)]  # the L/s mistake
    HR, zA, zB, zC = 100.0, 60.0, 72.0, 55.0
    m = ConcreteModel()
    m.D = Var(range(3), bounds=(0.10, 0.60), initialize=0.3)
    m.H = Var(range(3), bounds=(0, HR), initialize=dict(enumerate([95.0, 90.0, 90.0])))
    hf = [10.67 * Lp[i] * Qp[i] ** 1.852 / CHW ** 1.852 * m.D[i] ** (-4.87) for i in range(3)]
    m.c = ConstraintList()
    m.c.add(HR - m.H[0] == hf[0])
    m.c.add(m.H[0] - m.H[1] == hf[1])
    m.c.add(m.H[0] - m.H[2] == hf[2])
    m.c.add(m.H[1] - zB >= 27.0)
    m.c.add(m.H[2] - zC >= 20)
    m.c.add(m.H[0] - zA >= 20)
    for i in range(3):
        m.c.add(4 * Qp[i] / math.pi * m.D[i] ** (-2) >= 0.6)
    m.obj = Objective(expr=sum(1.5e-3 * Lp[i] * m.D[i] ** 1.4 for i in range(3)))
    return m


def opf3_overload():
    lines = [(0, 1), (0, 2), (1, 2)]
    y = 1 / complex(0.05, 0.25)
    G = {(i, j): 0.0 for i in range(3) for j in range(3)}
    Bm = dict(G)
    for (i, j) in lines:
        G[i, j] -= y.real; G[j, i] -= y.real; Bm[i, j] -= y.imag; Bm[j, i] -= y.imag
        G[i, i] += y.real; G[j, j] += y.real; Bm[i, i] += y.imag; Bm[j, j] += y.imag
    gens, Pd, Qd = [0, 1], (0.0, 0.0, 3.6), (0.0, 0.0, 0.5)
    m = ConcreteModel()
    m.V = Var(range(3), bounds=(0.9, 1.1), initialize=0.95)
    m.th = Var(range(3), bounds=(-3.14, 3.14), initialize={0: 0.0, 1: 0.4, 2: -0.4})
    m.th[0].fix(0)
    m.Pg = Var(gens, bounds=(0, 2), initialize=1.2)
    m.Qg = Var(gens, bounds=(-1, 1), initialize=0.0)

    def P(m, i):
        return m.V[i] * sum(m.V[j] * (G[i, j] * cos(m.th[i] - m.th[j]) + Bm[i, j] * sin(m.th[i] - m.th[j])) for j in range(3))

    def Q(m, i):
        return m.V[i] * sum(m.V[j] * (G[i, j] * sin(m.th[i] - m.th[j]) - Bm[i, j] * cos(m.th[i] - m.th[j])) for j in range(3))

    m.pb = Constraint(range(3), rule=lambda m, i: (m.Pg[i] if i in gens else 0) - Pd[i] == P(m, i))
    m.qb = Constraint(range(3), rule=lambda m, i: (m.Qg[i] if i in gens else 0) - Qd[i] == Q(m, i))
    cost = {0: (0.11, 5, 150), 1: (0.085, 1.2, 600)}
    m.obj = Objective(expr=sum(cost[g][0] * (100 * m.Pg[g]) ** 2 + cost[g][1] * 100 * m.Pg[g] + cost[g][2] for g in gens))
    return m


def circle_parabola():
    m = ConcreteModel()
    m.x = Var(initialize=0.0)
    m.y = Var(initialize=0.0)
    m.c1 = Constraint(expr=(m.x - 3) ** 2 + m.y ** 2 == 1)
    m.c2 = Constraint(expr=m.y == m.x ** 2)
    m.obj = Objective(expr=m.x ** 2 + m.y ** 2)
    return m


if __name__ == "__main__":
    for name, build in [
        ("issue981_cstr_dup_row", cstr_dup_row),
        ("issue981_wachter_biegler", wachter_biegler),
        ("issue981_water_main", water_main),
        ("issue981_opf3_overload", opf3_overload),
        ("issue981_circle_parabola", circle_parabola),
    ]:
        build().write(os.path.join(HERE, name + ".nl"))
