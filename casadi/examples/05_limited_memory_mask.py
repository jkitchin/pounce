#!/usr/bin/env python3
"""Limited-memory Hessians over the nonlinear variables only.

Many models are mostly linear: slacks, inventory balances, flows that
appear in the objective and constraints with a constant coefficient.
Under `hessian_approximation=limited-memory` the L-BFGS update spans the
whole variable vector by default, which spends storage and curvature
information on variables whose second derivatives are identically zero.

`pass_nonlinear_variables=True` asks CasADi to work out which variables
actually enter nonlinearly (`which_depends`) and hand that set to POUNCE,
which then approximates over the subspace and leaves the Hessian exactly
zero elsewhere. It is an approximation-space restriction, not a different
problem: the KKT point is the same, which is what this script checks.

Whether it is *faster* is a different question, and the honest answer is
"measure it". On this model scaled up to 2000 linear variables the
restriction costs time rather than saving it (4.6 s against 0.9 s
unmasked): zeroing the quasi-Newton diagonal on the linear block leaves
those KKT rows carrying only the barrier term, and the linear solve pays
more than the smaller update saves. That is a property of the
formulation — the same model through CasADi's Ipopt plugin goes from
0.4 s to 399 s — not of POUNCE. Switch it on for a model where you have
measured a win.

Give the set yourself with `nonlinear_variables=[bool]*nx` when you know
the structure better than the symbolic analysis can (for example when an
expression is nonlinear on paper but linear on the feasible set).

The corresponding POUNCE-side entry point is
`IpoptSetNonlinearVariables` in the C API; `num_linear_variables` is the
Ipopt-compatible contiguous-prefix fallback.
"""

import casadi as ca

N_LIN = 200      # variables that appear only linearly
N = 2 + N_LIN

x = ca.MX.sym("x", N)
nonlin, lin = x[:2], x[2:]

f = (1 - nonlin[0]) ** 2 + 100 * (nonlin[1] - nonlin[0] ** 2) ** 2 + ca.sum1(lin)
g = ca.vertcat(
    nonlin[0] ** 2 + nonlin[1] ** 2 - 1.5,   # nonlinear
    ca.sum1(lin) - 1,                        # linear
)
nlp = {"x": x, "f": f, "g": g}

args = dict(
    x0=[0.5, 0.5] + [0.1] * N_LIN,
    lbx=-5, ubx=5,
    lbg=[-ca.inf, 0], ubg=[0, 0],
)

base = {
    "print_time": False,
    "pounce": {"print_level": 0, "hessian_approximation": "limited-memory"},
}
masked = dict(base, pass_nonlinear_variables=True)

full = ca.nlpsol("full", "pounce", nlp, base)
part = ca.nlpsol("part", "pounce", nlp, masked)

r_full = full(**args)
r_part = part(**args)

print(f"{N} variables, {N_LIN} of them linear\n")
print(f"full space   : f = {float(r_full['f']):.10f}   "
      f"iters = {full.stats()['iter_count']}")
print(f"masked       : f = {float(r_part['f']):.10f}   "
      f"iters = {part.stats()['iter_count']}")
print(f"max |Δx|     : {float(ca.norm_inf(r_full['x'] - r_part['x'])):.3e}")
print("(same KKT point — the mask restricts the approximation, not the problem)")

# Declaring the subset by hand — same answer, and the way to do it when
# the structural analysis is not what you want.
by_hand = dict(base, nonlinear_variables=[True, True] + [False] * N_LIN)
r_hand = ca.nlpsol("hand", "pounce", nlp, by_hand)(**args)
print(f"explicit mask: f = {float(r_hand['f']):.10f}")
