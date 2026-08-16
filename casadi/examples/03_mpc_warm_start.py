#!/usr/bin/env python3
"""Receding-horizon loop: one solver object, re-solved per step, warm
started from the previous solution.

The pattern is the CasADi standard one — build `nlpsol` once, call it in
the loop with a new parameter value and the previous `x`, `lam_g`,
`lam_x` — with two POUNCE options that make the warm start count:

    warm_start_init_point = yes     use the supplied multipliers
    mu_init               = 1e-6    start near the previous barrier level

Without them the multipliers you pass are ignored (they stay strictly
outputs), which is Ipopt's contract too.
"""

import casadi as ca

N = 20      # horizon
dt = 0.1

# --- Build the parametric OCP once. x0_par is the measured state. ---
X = ca.MX.sym("X", 2, N + 1)
U = ca.MX.sym("U", 1, N)
x0_par = ca.MX.sym("x0_par", 2)

cost = 0
cons = [X[:, 0] - x0_par]
for k in range(N):
    pos, vel = X[0, k], X[1, k]
    nxt = ca.vertcat(pos + dt * vel, vel + dt * (U[0, k] - 0.1 * vel * ca.fabs(vel)))
    cons.append(X[:, k + 1] - nxt)
    cost += pos**2 + 0.1 * vel**2 + 0.01 * U[0, k] ** 2
cost += 10 * (X[0, N] ** 2 + X[1, N] ** 2)

nlp = {
    "x": ca.vertcat(ca.vec(X), ca.vec(U)),
    "p": x0_par,
    "f": cost,
    "g": ca.vertcat(*cons),
}

nx = 2 * (N + 1) + N
ng = 2 * (N + 1)

cold_opts = {"print_time": False, "pounce": {"print_level": 0, "tol": 1e-8}}
warm_opts = {
    "print_time": False,
    "pounce": {
        "print_level": 0,
        "tol": 1e-8,
        "warm_start_init_point": "yes",
        "mu_init": 1e-6,
    },
}

cold = ca.nlpsol("cold", "pounce", nlp, cold_opts)
warm = ca.nlpsol("warm", "pounce", nlp, warm_opts)

lbx = [-ca.inf] * (2 * (N + 1)) + [-2.0] * N
ubx = [ca.inf] * (2 * (N + 1)) + [2.0] * N

state = ca.DM([1.0, 0.0])
prev = None
cold_iters, warm_iters = [], []

for step in range(8):
    args = dict(p=state, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    # Reference: what a cold solve costs at this state.
    cold(x0=ca.DM.zeros(nx), **args)
    cold_iters.append(cold.stats()["iter_count"])

    if prev is None:
        sol = cold(x0=ca.DM.zeros(nx), **args)
    else:
        sol = warm(x0=prev["x"], lam_g0=prev["lam_g"], lam_x0=prev["lam_x"], **args)
        warm_iters.append(warm.stats()["iter_count"])

    prev = sol
    # Apply the first control and step the plant.
    u0 = float(sol["x"][2 * (N + 1)])
    pos, vel = float(state[0]), float(state[1])
    state = ca.DM([pos + dt * vel, vel + dt * (u0 - 0.1 * vel * abs(vel))])
    print(f"step {step}: u0={u0:+.4f}  state=[{float(state[0]):+.4f}, {float(state[1]):+.4f}]")

print()
print("cold-start iterations per step :", cold_iters)
print("warm-start iterations per step :", warm_iters)
print(
    f"mean: {sum(cold_iters)/len(cold_iters):.1f} cold vs "
    f"{sum(warm_iters)/max(len(warm_iters),1):.1f} warm"
)
