"""Generator for infeasible_pair.nl: a declared pair with no feasible point.

    min  x + y   s.t.  x >= 1,  y >= 1  (rows),  0 <= x  _|_  y >= 0

The rows, not the variable bounds, carry the 1: AMPL reads a pair against
its variable's bound, so `x >= 1` as a bound would make the pair
`(x - 1) _|_ y` and the model feasible at (1, 1). As rows, x*y >= 1 at
every point that satisfies them, so every formulation of the driver fails
and it reports the exact-product solve's verdict.

Written with Pyomo's `mpec.nl` transformation, which adds an auxiliary
variable for the pair's expression side (`cc.bv`). Regenerate with

    python infeasible_pair.py <outdir>
"""
import sys

import pyomo.environ as pe
from pyomo.mpec import Complementarity, complements

m = pe.ConcreteModel()
m.x = pe.Var(bounds=(0, None), initialize=2)
m.y = pe.Var(bounds=(0, None), initialize=2)
m.obj = pe.Objective(expr=m.x + m.y)
m.lx = pe.Constraint(expr=m.x >= 1)
m.ly = pe.Constraint(expr=m.y >= 1)
m.cc = Complementarity(expr=complements(m.x >= 0, m.y >= 0))
pe.TransformationFactory("mpec.nl").apply_to(m)
for v in m.component_data_objects(pe.Var):
    if v.value is None:
        v.value = 0.5
m.write(f"{sys.argv[1]}/infeasible_pair.nl", format="nl",
        io_options={"symbolic_solver_labels": True})
