#!/usr/bin/env python3
"""The hello-world: solve a small NLP with POUNCE through `nlpsol`.

    CASADIPATH=/path/to/pounce/casadi python3 01_rosenbrock.py

Everything here is ordinary CasADi. The only POUNCE-specific parts are
the plugin name and the `pounce` option dict, which takes any
Ipopt-compatible option name.
"""

import casadi as ca

x = ca.MX.sym("x", 2)
p = ca.MX.sym("p")

f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
g = x[0] ** 2 + x[1] ** 2 - p  # stay inside a circle of radius sqrt(p)

nlp = {"x": x, "p": p, "f": f, "g": g}

solver = ca.nlpsol(
    "solver",
    "pounce",
    nlp,
    {
        "print_time": False,
        "pounce": {
            "print_level": 5,     # POUNCE's own iteration table
            "tol": 1e-9,
            "max_iter": 200,
        },
    },
)

sol = solver(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)

print()
print("x*      =", sol["x"].full().ravel())
print("f(x*)   =", float(sol["f"]))
print("lam_g   =", sol["lam_g"].full().ravel())
print("lam_x   =", sol["lam_x"].full().ravel())

stats = solver.stats()
print("status  =", stats["return_status"])
print("success =", stats["success"])
print("iters   =", stats["iter_count"])

# `stats()["iterations"]` carries the full per-iteration trace, the same
# columns POUNCE prints: use it for convergence plots without parsing
# stdout.
inf_pr = stats["iterations"]["inf_pr"]
print("primal infeasibility, first→last:", inf_pr[0], "→", inf_pr[-1])
