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
"""

import math
import os

from pyomo.environ import (
    ConcreteModel, Constraint, ConstraintList, Objective, Var, exp, maximize,
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


if __name__ == "__main__":
    for name, build in [
        ("issue981_cstr_dup_row", cstr_dup_row),
        ("issue981_wachter_biegler", wachter_biegler),
        ("issue981_water_main", water_main),
    ]:
        build().write(os.path.join(HERE, name + ".nl"))
