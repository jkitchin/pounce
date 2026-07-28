"""Adversary cross-check: L2-regularized logistic regression (exp cone)
Family: exp   Class: logistic regression / logsumexp epigraph, analytic optimum
Source: Boyd & Vandenberghe, "Convex Optimization" (2004) sec. 7.1.1 (logistic
        regression MLE) + MOSEK Modeling Cookbook v3.3 sec. 5.2.5 "Log-sum-exp"
        and 5.3 "Logistic regression" for the exp-cone epigraph
        softplus(-u) <= t  <=>  exp(-u-t) + exp(-t) <= 1.
        Instance constructed so the optimum is available in CLOSED FORM.

Problem:
    min_w  sum_i log(1 + exp(-y_i w'x_i)) + (lam/2)||w||^2

Instance: two orthogonal feature directions a (m copies, y=+1) and
b (k copies, y=+1), rotated by R(30 deg) so the solver sees a coupled 2-D
problem.  Because a _|_ b the objective separates in u_a = a'w, u_b = b'w:

    g(u) = n_g log(1+e^{-u}) + (lam/2) u^2/||a||^2
    g'(u) = -n_g/(1+e^u) + lam u/||a||^2 = 0

Choosing ||a||^2 = (1+e^{u*}) lam u* / n_g pins u* exactly.
We pin u_a* = log 3 (m=2) and u_b* = log 4 (k=3), lam = 1:
    ||a||^2 = 4 log 3 / 2 = 2 log 3
    ||b||^2 = 5 log 4 / 3
    f* = 2 log(4/3) + 3 log(5/4) + (1/2)[ (log3)^2/||a||^2 + (log4)^2/||b||^2 ]

CONE CONVENTION (verified against crates/pounce-convex/src/cones/exp.rs and
python/pounce/qp.py::solve_socp docstring):
    pounce K_exp = {(x,y,z) : y exp(x/y) <= z, y>0}   (Clarabel/MOSEK order)
    slack s = h - G x must lie in the cone, block by block.
cvxpy's cp.constraints.ExpCone(x,y,z) uses the SAME order (y e^{x/y} <= z).
"""

import time

import numpy as np

# ---------------------------------------------------------------- instance
LAM = 1.0
m, k = 2, 3
ua_star, ub_star = np.log(3.0), np.log(4.0)
na2 = (1.0 + np.exp(ua_star)) * LAM * ua_star / m  # ||a||^2
nb2 = (1.0 + np.exp(ub_star)) * LAM * ub_star / k  # ||b||^2

th = np.pi / 6.0
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
a = R @ np.array([np.sqrt(na2), 0.0])
b = R @ np.array([0.0, np.sqrt(nb2)])
assert abs(a @ b) < 1e-12

KNOWN_OPTIMAL = (
    m * np.log(1.0 + np.exp(-ua_star))
    + k * np.log(1.0 + np.exp(-ub_star))
    + 0.5 * LAM * (ua_star**2 / na2 + ub_star**2 / nb2)
)
W_STAR = (ua_star / na2) * a + (ub_star / nb2) * b

# --------------------------------------------------------------- pounce
# vars: w0 w1 ta tb z1 z2 z3 z4      (n = 8)
# obj:  (lam/2)||w||^2 + m ta + k tb
# exp1: (-a'w - ta, 1, z1)   exp2: (-ta, 1, z2)   z1+z2 <= 1
# exp3: (-b'w - tb, 1, z3)   exp4: (-tb, 1, z4)   z3+z4 <= 1
n = 8
IW0, IW1, ITA, ITB, IZ1, IZ2, IZ3, IZ4 = range(8)
P = np.zeros((n, n))
P[IW0, IW0] = LAM
P[IW1, IW1] = LAM  # pounce obj = 1/2 x'Px + c'x
c = np.zeros(n)
c[ITA], c[ITB] = float(m), float(k)

rows, h = [], []


def add(coefs, hv):
    """append row with s = hv - coefs.x"""
    r = np.zeros(n)
    for i, v in coefs.items():
        r[i] = v
    rows.append(r)
    h.append(hv)


def softplus_block(uvec, it, iz1, iz2):
    # (-u - t, 1, z1)
    add({IW0: uvec[0], IW1: uvec[1], it: 1.0}, 0.0)  # s = -(a'w + t)
    add({}, 1.0)  # s = 1
    add({iz1: -1.0}, 0.0)  # s = z1
    # (-t, 1, z2)
    add({it: 1.0}, 0.0)
    add({}, 1.0)
    add({iz2: -1.0}, 0.0)


softplus_block(a, ITA, IZ1, IZ2)
softplus_block(b, ITB, IZ3, IZ4)
# z1+z2 <= 1, z3+z4 <= 1  (nonneg slacks)
add({IZ1: 1.0, IZ2: 1.0}, 1.0)
add({IZ3: 1.0, IZ4: 1.0}, 1.0)

G = np.array(rows)
h = np.array(h)
cones = [("exp", 3)] * 4 + [("nonneg", 2)]

from pounce import solve_socp  # noqa: E402

t0 = time.perf_counter()
r = solve_socp(P=P, c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x_p = np.asarray(r.x)
w_pounce = x_p[[IW0, IW1]]
obj_pounce = float(r.obj)
status = r.status

# --------------------------------------------------------------- oracles
import cvxpy as cp  # noqa: E402


def cvx(solver):
    w = cp.Variable(2)
    ua, ub = a @ w, b @ w
    obj = (
        m * cp.logistic(-ua) + k * cp.logistic(-ub) + 0.5 * LAM * cp.sum_squares(w)
    )
    pr = cp.Problem(cp.Minimize(obj))
    t0 = time.perf_counter()
    pr.solve(solver=solver)
    return pr.value, np.asarray(w.value), time.perf_counter() - t0, pr.status


obj_o1, w_o1, t_o1, st1 = cvx(cp.SCS)
obj_o2, w_o2, t_o2, st2 = cvx(cp.CLARABEL)


def rel(x, y):
    return abs(x - y) / max(1.0, abs(y))


print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s iters={getattr(r,'iterations',None)}")
print(f"w={w_pounce}")
print("=== oracle SCS ===")
print(f"status={st1} obj={obj_o1:.10e} t={t_o1:.4f}s w={w_o1}")
print("=== oracle CLARABEL ===")
print(f"status={st2} obj={obj_o2:.10e} t={t_o2:.4f}s w={w_o2}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e}  w*={W_STAR}")
print(f"rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"obj_err_vs_SCS={rel(obj_pounce, obj_o1):.2e} obj_err_vs_CLARABEL={rel(obj_pounce, obj_o2):.2e}")
print(f"w_inf_err_vs_known={np.max(np.abs(w_pounce - W_STAR)):.2e}")
print(f"w_inf_err_vs_CLARABEL={np.max(np.abs(w_pounce - w_o2)):.2e}")

print(f"pounce kkt_error={r.kkt_error:.2e} residuals={r.residuals} iters={r.iters}")
ok = (
    (status.startswith("optimal") or r.success)
    and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4
    and rel(obj_pounce, obj_o2) < 1e-4
)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status})")
