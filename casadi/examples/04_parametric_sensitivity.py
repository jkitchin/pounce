#!/usr/bin/env python3
"""Differentiating through a POUNCE solve.

A CasADi `nlpsol` object is a `Function`, so it composes into larger
expression graphs and can be differentiated: CasADi's `Nlpsol` base class
implements forward and adjoint derivatives of the solution map by
linearizing the KKT system at the returned point. That machinery is
inherited by every plugin, POUNCE included — nothing here is special-cased.

Two consequences worth knowing:

  * `jacobian(sol['x'], p)` is exact (implicit function theorem), not a
    finite difference — but it is only as good as the multipliers, so
    a loose `tol` shows up here first;
  * the solve is embedded in the graph, so an outer optimizer or a
    least-squares fit can be built on top of it (bilevel / parameter
    estimation).
"""

import casadi as ca

x = ca.MX.sym("x", 2)
p = ca.MX.sym("p", 2)

# min (x0 - p0)^2 + (x1 - p1)^2   s.t.   x0^2 + x1^2 <= 1
f = (x[0] - p[0]) ** 2 + (x[1] - p[1]) ** 2
g = x[0] ** 2 + x[1] ** 2 - 1

solver = ca.nlpsol(
    "solver", "pounce", {"x": x, "p": p, "f": f, "g": g},
    {"print_time": False, "pounce": {"print_level": 0, "tol": 1e-12}},
)

sol = solver(x0=[0.1, 0.1], p=p, lbg=-ca.inf, ubg=0)

x_of_p = ca.Function("x_of_p", [p], [sol["x"]])
dx_dp = ca.Function("dx_dp", [p], [ca.jacobian(sol["x"], p)])

p_val = ca.DM([2.0, 1.0])
print("x*(p)   =", x_of_p(p_val).full().ravel())
print("dx*/dp  =")
print(dx_dp(p_val))

# Cross-check with a central difference on the solution map.
eps = 1e-6
fd = ca.DM.zeros(2, 2)
for j in range(2):
    d = ca.DM.zeros(2)
    d[j] = eps
    fd[:, j] = (x_of_p(p_val + d) - x_of_p(p_val - d)) / (2 * eps)
print("finite difference =")
print(fd)
print("max |analytic - FD| =", float(ca.norm_inf(dx_dp(p_val) - fd)))

# Because the solve is differentiable, it can sit inside another
# optimization. Here: choose p so that the solution lands on a target.
target = ca.DM([0.6, 0.8])
q = ca.MX.sym("q", 2)
inner = solver(x0=[0.1, 0.1], p=q, lbg=-ca.inf, ubg=0)["x"]
outer = ca.nlpsol(
    "outer", "pounce", {"x": q, "f": ca.sumsqr(inner - target)},
    {"print_time": False, "pounce": {"print_level": 0}},
)
r = outer(x0=[2.0, 1.0])
print("\nbilevel: p minimizing ||x*(p) - target||^2 =", r["x"].full().ravel())
print("resulting x*(p) =", x_of_p(r["x"]).full().ravel(), " target =", target.full().ravel())
