#!/usr/bin/env python3
"""`Opti` + POUNCE: a minimum-effort rocket ascent.

`Opti` is CasADi's modelling front end, and it only accepts an `nlpsol`
*plugin* name — which is exactly what POUNCE registers, so
`opti.solver("pounce", ...)` works with no other change.

The two dicts `opti.solver` takes are (plugin options, solver options):
the second lands in POUNCE's own option list, so anything you would put
in an `ipopt.opt` goes there.
"""

import casadi as ca

N = 40          # control intervals
T = 10.0        # horizon (s)
dt = T / N
m = 1.0         # mass
u_max = 3.0     # thrust limit

opti = ca.Opti()

pos = opti.variable(N + 1)
vel = opti.variable(N + 1)
u = opti.variable(N)

# Explicit Euler dynamics: ṗ = v, v̇ = u/m − g
g_acc = 1.0
for k in range(N):
    opti.subject_to(pos[k + 1] == pos[k] + dt * vel[k])
    opti.subject_to(vel[k + 1] == vel[k] + dt * (u[k] / m - g_acc))

opti.subject_to(opti.bounded(0, u, u_max))
opti.subject_to(pos[0] == 0)
opti.subject_to(vel[0] == 0)
opti.subject_to(pos[N] == 10)
opti.subject_to(vel[N] == 0)

# Minimum control effort.
opti.minimize(dt * ca.sumsqr(u))

opti.set_initial(u, 1.0)

opti.solver(
    "pounce",
    {"print_time": False},            # CasADi-level options
    {"print_level": 0, "tol": 1e-8},  # POUNCE options
)

sol = opti.solve()

print("status      :", sol.stats()["return_status"])
print("iterations  :", sol.stats()["iter_count"])
print("effort      :", float(sol.value(dt * ca.sumsqr(u))))
print("final pos/vel:", float(sol.value(pos[N])), float(sol.value(vel[N])))
print("u[0:5]      :", sol.value(u)[:5])

# `opti.debug` still works: POUNCE reports the same failure modes CasADi
# expects, so `opti.debug.value(...)` after a failed solve behaves as it
# does with any other plugin.
