"""Adversary cross-check: analytic center of a box via exp-cone log barrier
Family: exp   Class: exponential-cone (log-barrier analytic center)
    (fresh class for exp -- prior exp runs covered GP/posynomial problems,
    entropy maximization x*log(x), KL-divergence/relative-entropy
    projection, logistic-regression softplus, and log-sum-exp; none used
    the analytic-center barrier -log(affine expression) directly, which
    is a distinct exp-cone shape: t_i <= log(w_i) with w_i an AFFINE
    function of x, not log of the variable itself.)
Source: Boyd & Vandenberghe, "Convex Optimization", Section 8.5.3
    ("Analytic center of a set of linear inequalities"): the analytic
    center of {x : a_i^T x <= b_i, i=1..m} maximizes sum_i log(b_i - a_i^T x).

Problem: analytic center of the box x1 in [-2, 1], x2 in [-1, 3], i.e. the
4 inequalities  1-x1>=0, x1-(-2)>=0, 3-x2>=0, x2-(-1)>=0.

    maximize   log(1-x1) + log(x1+2) + log(3-x2) + log(x2+1)

Because the barrier is additively SEPARABLE across coordinates (a box is a
Cartesian product), each coordinate's analytic center is found independently:
for scalar t in [l,u], maximize log(u-t)+log(t-l); d/dt: 1/(t-l) = 1/(u-t)
=> t = (l+u)/2, the midpoint. This is a standard textbook fact independent
of both pounce and cvxpy.

    x1* = (-2+1)/2 = -0.5,   x2* = (-1+3)/2 = 1.0
    KNOWN_OPTIMAL (as a MINIMIZE of -sum log(...), pounce's sense)
        = -[log(1.5) + log(1.5) + log(2) + log(2)]
        = -2*log(1.5) - 2*log(2)

CONE LAYOUT (pounce, solve_socp): variables x = [x1, x2, t1, t2, t3, t4]
(t1<=log(1-x1), t2<=log(x1+2), t3<=log(3-x2), t4<=log(x2+1)). objective:
minimize -(t1+t2+t3+t4). Per term, a triple (t_j, 1, w_j) in Kexp where
Kexp = {(a,b,c): b*exp(a/b) <= c, b>0}; with b=1 this is exp(t_j) <= w_j,
i.e. t_j <= log(w_j).
"""
import time
import numpy as np
import pounce
import cvxpy as cp

l1, u1 = -2.0, 1.0
l2, u2 = -1.0, 3.0

x1_star, x2_star = (l1 + u1) / 2, (l2 + u2) / 2
KNOWN_OPTIMAL = -(np.log(u1 - x1_star) + np.log(x1_star - l1)
                  + np.log(u2 - x2_star) + np.log(x2_star - l2))
print(f"closed-form (separable box barrier): x*=({x1_star},{x2_star}) "
      f"min_obj={KNOWN_OPTIMAL:.10f}")

# ---- pounce exp-cone encoding ----------------------------------------------
# vars: x1(0) x2(1) t1(2) t2(3) t3(4) t4(5)
nv = 6
c = np.zeros(nv)
c[2:] = -1.0  # minimize -(t1+t2+t3+t4)

rows = 12
G = np.zeros((rows, nv))
h = np.zeros(rows)

# term1: t1 <= log(u1 - x1)
G[0, 2] = -1.0            # s0 = t1
h[1] = 1.0                # s1 = 1
G[2, 0] = 1.0; h[2] = u1  # s2 = u1 - x1
# term2: t2 <= log(x1 - l1)
G[3, 3] = -1.0            # s3 = t2
h[4] = 1.0
G[5, 0] = -1.0; h[5] = -l1  # s5 = -l1 - (-x1) = x1 - l1
# term3: t3 <= log(u2 - x2)
G[6, 4] = -1.0
h[7] = 1.0
G[8, 1] = 1.0; h[8] = u2
# term4: t4 <= log(x2 - l2)
G[9, 5] = -1.0
h[10] = 1.0
G[11, 1] = -1.0; h[11] = -l2

cones = [("exp", 3)] * 4

t0 = time.perf_counter()
res = pounce.solve_socp(c=c, G=G, h=h, cones=cones, tol=1e-9)
t_pounce = time.perf_counter() - t0
x = np.asarray(res.x)
x_pounce = x[:2]
obj_pounce = res.obj
status = res.status

# ---- oracle: cvxpy (exp-cone via cp.log), CLARABEL + SCS -------------------
def solve_cvxpy(solver):
    xv = cp.Variable(2)
    terms = [cp.log(u1 - xv[0]), cp.log(xv[0] - l1),
             cp.log(u2 - xv[1]), cp.log(xv[1] - l2)]
    prob = cp.Problem(cp.Maximize(cp.sum(terms)), [])
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    return -prob.value, time.perf_counter() - t0, np.asarray(xv.value)  # min sense

obj_cl, t_cl, x_cl = solve_cvxpy(cp.CLARABEL)
obj_scs, t_scs, x_scs = solve_cvxpy(cp.SCS)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


err_known = rel(obj_pounce, KNOWN_OPTIMAL)
err_cl = rel(obj_pounce, obj_cl)
err_scs = rel(obj_pounce, obj_scs)
x_err_known = float(np.linalg.norm(x_pounce - np.array([x1_star, x2_star]), np.inf))

print(f"=== pounce ===")
print(f"status={status} obj={obj_pounce:.10f} x={np.round(x_pounce, 6)} t={t_pounce:.4f}s")
print("=== oracle cvxpy ===")
print(f"CLARABEL obj={obj_cl:.10f} x={np.round(x_cl,6)} t={t_cl:.4f}s")
print(f"SCS      obj={obj_scs:.10f} t={t_scs:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10f}")
print(f"rel_err vs known={err_known:.2e}  vs CLARABEL={err_cl:.2e}  vs SCS={err_scs:.2e}  "
      f"x_inf_err_vs_known={x_err_known:.2e}")

ok = (status == "optimal" or res.success) and err_known < 1e-5 and err_cl < 1e-5 and x_err_known < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, err_known={err_known:.2e})")
