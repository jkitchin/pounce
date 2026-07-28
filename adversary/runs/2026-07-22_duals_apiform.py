"""Follow-up: is the sign defect in the .sol WRITER or in the internal
convention? Solve the IDENTICAL .nl through the Python library API and
compare mult_g against the .sol bytes.

Also: probe how pounce.minimize signs multipliers for the g(x)>=0 form.
"""
import os
import subprocess
import tempfile

import numpy as np
import pyomo.environ as pyo

import pounce

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
WORK = tempfile.mkdtemp(prefix="adv_apiform_")


def build(eq=False):
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(bounds=(0, None), initialize=1.0)
    m.x2 = pyo.Var(bounds=(0, None), initialize=1.0)
    m.obj = pyo.Objective(expr=-3 * m.x1 - 5 * m.x2, sense=pyo.minimize)
    m.c1 = pyo.Constraint(expr=m.x1 <= 4)
    m.c2 = pyo.Constraint(expr=2 * m.x2 <= 12)
    m.c3 = (
        pyo.Constraint(expr=3 * m.x1 + 2 * m.x2 == 18)
        if eq
        else pyo.Constraint(expr=3 * m.x1 + 2 * m.x2 <= 18)
    )
    return m


for eq in (False, True):
    tag = "EQ" if eq else "INEQ"
    nl = os.path.join(WORK, f"w_{tag}.nl")
    build(eq).write(nl, io_options={"symbolic_solver_labels": True})

    # --- Python library API on the exact same .nl ---
    nlp = pounce.read_nl(nl)

    class W:
        def objective(self, x):
            return nlp.objective(x)

        def gradient(self, x):
            return nlp.gradient(x)

        def constraints(self, x):
            return nlp.constraints(x)

        def jacobian(self, x):
            return nlp.jacobian(x)

        def jacobianstructure(self):
            return nlp.jacobian_structure()

    prob = pounce.Problem(
        n=nlp.n,
        m=nlp.m,
        problem_obj=W(),
        lb=nlp.x_l,
        ub=nlp.x_u,
        cl=nlp.g_l,
        cu=nlp.g_u,
    )
    prob.add_option("print_level", 0)
    x, info = prob.solve(np.array([1.0, 1.0]))
    mg = np.asarray(info.get("mult_g"))
    print(f"[{tag}] read_nl().solve  x={np.round(x,6)}  mult_g={np.round(mg,6)}")
    print(f"        info keys: {sorted(info.keys())}")
    for k in ("mult_x_L", "mult_x_U"):
        if k in info:
            print(f"        {k} = {np.round(np.asarray(info[k]),6)}")

    # --- CLI .sol on the exact same .nl ---
    base = nl[:-3]
    subprocess.run([POUNCE, base, "-AMPL"], capture_output=True, cwd=WORK, timeout=30)
    lines = [ln.strip() for ln in open(base + ".sol")]
    i = lines.index("Options")
    j = i + 1 + 1 + int(lines[i + 1])
    counts = [int(lines[j + k]) for k in range(4)]
    j += 4
    sol_duals = np.array([float(lines[j + k]) for k in range(counts[0])])
    print(f"[{tag}] .sol duals      = {np.round(sol_duals,6)}")
    print(f"[{tag}] .sol == mult_g? {np.allclose(sol_duals, mg, atol=1e-6)}")
    print(f"[{tag}] correct (AMPL)  = [0, -1.5, -1.0]\n")

# --- pounce.minimize with g(x) >= 0 form (its own constraint convention) ---
print("=== pounce.minimize, constraints in the g(x)>=0 / h(x)==0 form ===")
res = pounce.minimize(
    lambda z: -3 * z[0] - 5 * z[1],
    np.array([1.0, 1.0]),
    jac=lambda z: np.array([-3.0, -5.0]),
    bounds=[(0, None), (0, None)],
    constraints=[
        {"type": "ineq", "fun": lambda z: 4 - z[0], "jac": lambda z: np.array([[-1.0, 0.0]])},
        {"type": "ineq", "fun": lambda z: 12 - 2 * z[1], "jac": lambda z: np.array([[0.0, -2.0]])},
        {"type": "ineq", "fun": lambda z: 18 - 3 * z[0] - 2 * z[1],
         "jac": lambda z: np.array([[-3.0, -2.0]])},
    ],
)
print("minimize x =", np.round(res.x, 6), " mult_g =", np.round(np.asarray(res.mult_g), 6))
print("  constraints are g>=0 (negated vs '<= b'), so the AMPL-correct")
print("  d obj/d b duals [0,-1.5,-1] map to mult_g of [0,+1.5,+1] here IF the")
print("  convention were AMPL's, and [0,-1.5,-1] if it is flipped.")
print(f"\nworkdir: {WORK}")
