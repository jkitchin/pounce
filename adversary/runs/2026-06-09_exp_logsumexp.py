"""Adversary cross-check: log-sum-exp minimization over an affine hyperplane.
Family: exp   Class: exponential-cone (log-sum-exp epigraph)

Problem:  minimize   log( sum_i exp(x_i) )
          s.t.        sum_i x_i = c0

SOURCE: Standard convex-analysis result. The log-sum-exp function is convex and
  Schur-convex; minimizing it over the hyperplane {sum_i x_i = c0} is solved at
  the symmetric point x_i = c0/n (the gradient softmax(x) is proportional to the
  constraint normal 1 only when all components are equal). Then
        min LSE = log( n * exp(c0/n) ) = log(n) + c0/n.
  This is an exact closed-form known optimum.

KNOWN_OPTIMAL: log(n) + c0/n.  Here n=4, c0=4  ->  log(4) + 1 = 2.386294361119891.
  Optimal x* = (1,1,1,1).

N_VARIABLES (pounce): n (x) + 1 (t epigraph) + n (u_i) = 9.

CONE LAYOUT:
  Epigraph trick:  log(sum exp(x_i)) <= t  <=>  exists u_i >= 0 with
      sum_i u_i <= 1   and   exp(x_i - t) <= u_i  for each i.
  exp(x_i - t) <= u_i is the cone (x = x_i - t, y = 1, z = u_i) in Kexp
  since Kexp = {(x,y,z): y*exp(x/y) <= z} gives 1*exp(x_i-t) <= u_i.
  Objective: minimize t.
  Constraints: sum x_i = c0 (equality), sum u_i <= 1 (nonneg slack).
"""
import time
import numpy as np
import pounce
import cvxpy as cp

n = 4
c0 = 4.0
KNOWN_OPTIMAL = float(np.log(n) + c0 / n)

# variables: [x_0..x_{n-1}, t, u_0..u_{n-1}]   length 2n+1
xt = lambda i: i              # index of x_i
T = n                         # index of t
ui = lambda i: n + 1 + i      # index of u_i
N = 2 * n + 1

c = np.zeros(N)
c[T] = 1.0  # minimize t

# Cone rows: n exp cones (3 rows each) then one nonneg block (1 row) for sum u <= 1
rows_exp = 3 * n
rows_nn = 1
G = np.zeros((rows_exp + rows_nn, N))
h = np.zeros(rows_exp + rows_nn)
for i in range(n):
    r = 3 * i
    # cone triple slacks (s0,s1,s2) = (x_i - t, 1, u_i)
    # s0 = x_i - t  -> -G x = x_i - t  => G[r, x_i]=-1, G[r, t]=+1, h0=0
    G[r, xt(i)] = -1.0
    G[r, T] = 1.0
    # s1 = 1        -> h1 = 1
    h[r + 1] = 1.0
    # s2 = u_i      -> G[r+2, u_i] = -1
    G[r + 2, ui(i)] = -1.0
# nonneg: sum_i u_i <= 1  -> slack = 1 - sum u_i >= 0
rnn = rows_exp
for i in range(n):
    G[rnn, ui(i)] = 1.0
h[rnn] = 1.0

# equality: sum x_i = c0
A = np.zeros((1, N))
for i in range(n):
    A[0, xt(i)] = 1.0
b = np.array([c0])

cones = [("exp", 3)] * n + [("nonneg", 1)]

t0 = time.perf_counter()
res = pounce.solve_socp(c=c, A=A, b=b, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
xv = np.asarray(res.x)
x_pounce = xv[:n]
obj_pounce = res.obj
status = res.status

# ---- oracle: cvxpy with TWO solvers ---------------------------------------
def solve_cvxpy(solver):
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.log_sum_exp(x)), [cp.sum(x) == c0])
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    return prob.value, time.perf_counter() - t0, np.asarray(x.value)

obj_ecos, t_ecos, x_ecos = solve_cvxpy(cp.ECOS)
obj_scs, t_scs, x_scs = solve_cvxpy(cp.SCS)

def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))

err_known = rel(obj_pounce, KNOWN_OPTIMAL)
err_ecos = rel(obj_pounce, obj_ecos)
err_scs = rel(obj_pounce, obj_scs)

print(f"=== pounce (n={n}, c0={c0}) ===")
print(f"status={status} LSE={obj_pounce:.10f} t={t_pounce:.4f}s")
print(f"x*={np.round(x_pounce,5)}")
print("=== oracle cvxpy ===")
print(f"ECOS LSE={obj_ecos:.10f} t={t_ecos:.4f}s")
print(f"SCS  LSE={obj_scs:.10f} t={t_scs:.4f}s")
print(f"known_optimal=log({n})+{c0}/{n}={KNOWN_OPTIMAL:.10f}")
print(f"rel_err vs known={err_known:.2e}  vs ECOS={err_ecos:.2e}  vs SCS={err_scs:.2e}")

ok = (status == "optimal" or res.success) and err_known < 1e-5 and err_ecos < 1e-5 and err_scs < 1e-5
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, err_known={err_known:.2e})")
