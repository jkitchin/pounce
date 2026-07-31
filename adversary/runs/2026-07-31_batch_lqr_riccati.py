"""Adversary cross-check: batch of finite-horizon LQR QPs vs Riccati recursion
Family: batch   Class: batch of independent equality-constrained QPs sharing
        structure (dynamics), differing only in the initial-state equality
        RHS -- a standard batch-MPC data-parallel pattern.
Source: Discrete-time finite-horizon LQR, e.g. Bertsekas, "Dynamic
        Programming and Optimal Control" Vol 1, or Boyd, EE363 lecture notes
        ("Linear Quadratic Regulator: Discrete-Time Finite Horizon"). The
        Riccati recursion
            P_N = Qf
            K_t = (R + B'P_{t+1}B)^{-1} B'P_{t+1}A
            P_t = Q + A'P_{t+1}A - A'P_{t+1}B K_t
        gives the EXACT optimal cost x0' P_0 x0 for
            min sum_{t=0}^{N-1} x_t'Qx_t + u_t'Ru_t + x_N'Qf x_N
            s.t. x_{t+1}=Ax_t+Bu_t, x_0 given.
        This is a completely independent algorithm (dynamic programming, not
        an LP/QP/conic solver) and serves as the oracle -- no cvxpy involved.
Known optimal: cost_b = x0_b' P_0 x0_b for each of 6 different initial states,
        computed by the Riccati recursion below.
"""
import numpy as np
import time

# double integrator
A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[0.0], [1.0]])
Q = np.eye(2)
R = np.array([[1.0]])
Qf = 10.0 * np.eye(2)
N = 4  # horizon
nx, nu = 2, 1

# --- Riccati recursion (independent oracle, no LP/QP/conic solver at all) ---
P = [None] * (N + 1)
K = [None] * N
P[N] = Qf
for t in range(N - 1, -1, -1):
    Pn = P[t + 1]
    M = R + B.T @ Pn @ B
    Kt = np.linalg.solve(M, B.T @ Pn @ A)
    K[t] = Kt
    P[t] = Q + A.T @ Pn @ A - A.T @ Pn @ B @ Kt
P0 = P[0]

x0_batch = [
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([2.0, -1.0]),
    np.array([-3.0, 2.0]),
    np.array([0.5, 0.5]),
    np.array([-1.0, -2.0]),
]
# x0'P0 x0 is the cost-to-go from t=0 INCLUDING the x_0'Q x_0 stage term; the
# QP below has no x_0 decision variable (x_0 is fixed data), so its objective
# omits that constant term -- subtract it for an apples-to-apples comparison.
known_costs = [float(x0 @ P0 @ x0 - x0 @ Q @ x0) for x0 in x0_batch]

# --- build the QP: z = (x_1..x_N, u_0..u_{N-1}), stacked ---
nz = N * nx + N * nu


def xidx(t):  # x_1..x_N -> t=1..N
    return (t - 1) * nx


def uidx(t):  # u_0..u_{N-1} -> t=0..N-1
    return N * nx + t * nu


# objective: sum_{t=1}^{N-1} x_t'Qx_t + x_N'Qf x_N + sum_{t=0}^{N-1} u_t'Ru_t
# pounce's solve_qp minimizes (1/2) z'Pz + c'z (see qp.py docstring), so the
# block-diagonal P must be TWICE the (Q,Qf,R) blocks to reproduce the
# textbook sum-of-quadratic-forms cost.
Pmat = np.zeros((nz, nz))
for t in range(1, N):
    Pmat[xidx(t) : xidx(t) + nx, xidx(t) : xidx(t) + nx] = 2.0 * Q
Pmat[xidx(N) : xidx(N) + nx, xidx(N) : xidx(N) + nx] = 2.0 * Qf
for t in range(N):
    Pmat[uidx(t) : uidx(t) + nu, uidx(t) : uidx(t) + nu] = 2.0 * R
c = np.zeros(nz)

# equality constraints: x_1 = A x0 + B u0 ; x_{t+1} = A x_t + B u_t (t=1..N-1)
neq = N * nx
Aeq = np.zeros((neq, nz))
for t in range(N):
    row = t * nx
    Aeq[row : row + nx, xidx(t + 1) : xidx(t + 1) + nx] = np.eye(nx)
    if t == 0:
        pass  # x0 term goes into b (RHS), varies per instance
    else:
        Aeq[row : row + nx, xidx(t) : xidx(t) + nx] = -A
    Aeq[row : row + nx, uidx(t) : uidx(t) + nu] = -B


def beq_for(x0):
    b = np.zeros(neq)
    b[0:nx] = A @ x0  # from x_1 - B u0 = A x0
    return b


# --- pounce: solve_qp_batch ---
from pounce import solve_qp_batch, solve_qp

problems = [{"P": Pmat, "c": c, "A": Aeq, "b": beq_for(x0)} for x0 in x0_batch]

t0 = time.perf_counter()
results_batch = solve_qp_batch(problems)
t_batch = time.perf_counter() - t0

t0 = time.perf_counter()
results_single = [solve_qp(**pr) for pr in problems]
t_single = time.perf_counter() - t0

print("=== pounce solve_qp_batch vs per-item solve_qp vs Riccati (oracle) ===")
max_err_riccati = 0.0
max_err_batch_vs_single = 0.0
all_ok = True
for i, (rb, rs, known) in enumerate(zip(results_batch, results_single, known_costs)):
    ob, os_ = rb.obj, rs.obj
    err_r = abs(ob - known) / max(1.0, abs(known))
    err_bs = abs(ob - os_)
    max_err_riccati = max(max_err_riccati, err_r)
    max_err_batch_vs_single = max(max_err_batch_vs_single, err_bs)
    ok = rb.status == "optimal" and err_r < 1e-6
    all_ok = all_ok and ok
    print(
        f"item {i}: status={rb.status} obj_batch={ob:.8e} obj_single={os_:.8e} "
        f"known(Riccati)={known:.8e} rel_err_riccati={err_r:.2e} ok={ok}"
    )

print(f"t_batch={t_batch:.4f}s t_singles={t_single:.4f}s")
print(f"max_rel_err_vs_riccati={max_err_riccati:.2e} max_abs_err_batch_vs_single={max_err_batch_vs_single:.2e}")
print("VERDICT: PASS" if all_ok else "VERDICT: FAIL")
