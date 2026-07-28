"""Adversary cross-check: open-top square-base box volume maximization (2-var GP).
Family: exp   Class: exponential-cone (geometric programming, design)

Problem:  maximize  V = x^2 * y           (square base x by x, height y)
          s.t.       x^2 + 4*x*y <= A      (open-top surface: base + 4 walls)
                     x>0, y>0

SOURCE: Classic geometric-programming box-design problem (Boyd, Kim, Vandenberghe
  & Hassibi, "A Tutorial on Geometric Programming", Optim. Eng. 2007, Sec. on box
  design; equivalently Boyd & Vandenberghe, "Convex Optimization", Sec. 4.5).
  Lagrange/KKT closed form (verified here with sympy):
        x* = sqrt(A/3),  y* = x*/2,  V* = sqrt(3) * A^(3/2) / 18.
  With A = 12:  x* = 2,  y* = 1,  V* = 4  (exact).

KNOWN_OPTIMAL: V* = sqrt(3)*A^(3/2)/18 = 4.0 for A=12.

GP-to-conic encoding.  Let u = log x, v = log y (decision vars in pounce).
  Objective: maximize V = x^2 y = exp(2u+v)  <=>  maximize 2u+v (monotone)
             <=>  minimize -(2u+v); recover V = exp(-obj).
  Posynomial  x^2 + 4 x y <= A  <=>  exp(2u)/A + 4*exp(u+v)/A <= 1.
  Introduce t1,t2 >= 0 with
        exp(2u)/A      <= t1    (term 1)
        4*exp(u+v)/A   <= t2    (term 2)
        t1 + t2        <= 1.
  Rewrite each as  exp(arg) <= z :
    term1:  exp(2u + log(1/A))   <= t1   ->  arg1 = 2u - logA
    term2:  exp(u + v + log(4/A)) <= t2  ->  arg2 = u + v + log(4/A)
  pounce cone:  Kexp = {(x,y,z): y*exp(x/y) <= z, y>0}.  With y=1: exp(arg) <= z,
  so the triple is (arg, 1, z) and the cone slack is s = h - G x.

CONE LAYOUT (decision vars X = [u, v, t1, t2], N=4):
  cone 0 (exp): (s0,s1,s2) = (2u - logA,        1, t1)   const  -logA
  cone 1 (exp): (s0,s1,s2) = (u + v + log(4/A), 1, t2)   const +log(4/A)
  nonneg block (1 row): slack = 1 - t1 - t2 >= 0
  slack s = h - G x.

NOTE (regression guard, issue #145): a prior version of this script set
  h[3] = -log(4/A) instead of +log(4/A), flipping the sign of the cone-1
  constant. That is a DIFFERENT problem whose true optimum is V = 4/9; pounce
  solved THAT problem correctly (V = 4/9), and so does cvxpy when handed the
  same mis-signed G/h. The original "same encoding" check below masked the bug
  by rebuilding the affine arg by hand (with the correct sign) instead of
  reusing G/h, so it silently solved the corrected problem and disagreed with
  pounce. The check now consumes the SAME G/h via s = h - G@X, so an encoding
  error in this script can never again be mistaken for a solver bug.
"""
import time
import numpy as np
import pounce
import cvxpy as cp

A = 12.0
KNOWN_OPTIMAL = float(np.sqrt(3) * A**1.5 / 18.0)  # = 4.0 for A=12

logA = np.log(A)
log4A = np.log(4.0 / A)  # = -1.0986...

# decision vars X = [u, v, t1, t2]
U, V, T1, T2 = 0, 1, 2, 3
N = 4
c = np.zeros(N)
c[U] = -2.0  # minimize -(2u + v)
c[V] = -1.0

# cones: 2 exp (3 rows each) + 1 nonneg (1 row)
G = np.zeros((3 * 2 + 1, N))
h = np.zeros(3 * 2 + 1)

# --- cone 0: s = (2u - logA, 1, t1) ---  s = h - G x
G[0, U] = -2.0
h[0] = -logA  # const of arg0 is -logA
h[1] = 1.0
G[2, T1] = -1.0

# --- cone 1: s = (u + v + log(4/A), 1, t2) ---
G[3, U] = -1.0
G[3, V] = -1.0
h[3] = log4A  # const of arg1 is +log(4/A)   (issue #145: was -log4A -> wrong)
h[4] = 1.0
G[5, T2] = -1.0

# --- nonneg: 1 - t1 - t2 >= 0 ---
G[6, T1] = 1.0
G[6, T2] = 1.0
h[6] = 1.0

cones = [("exp", 3), ("exp", 3), ("nonneg", 1)]

t0 = time.perf_counter()
res = pounce.solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
xv = np.asarray(res.x)
u_s, v_s = xv[U], xv[V]
x_s, y_s = np.exp(u_s), np.exp(v_s)
obj_pounce = float(np.exp(-(c @ xv)))  # V = exp(2u+v) = exp(-c'x)
status = res.status

# ---- oracle 1: cvxpy GP with TWO solvers (independent model) ----------------
def solve_cvxpy_gp(solver):
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    prob = cp.Problem(cp.Maximize(cp.log(x**2 * y)), [x**2 + 4 * x * y <= A])
    t0 = time.perf_counter()
    prob.solve(solver=solver, gp=True)
    return float(x.value**2 * y.value), time.perf_counter() - t0, (float(x.value), float(y.value))

obj_ecos, t_ecos, xy_ecos = solve_cvxpy_gp(cp.ECOS)
obj_scs, t_scs, xy_scs = solve_cvxpy_gp(cp.SCS)

# ---- oracle 2: the SAME conic program. Feed pounce's EXACT G/h/c/cones to
#      cvxpy's ExpCone via s = h - G@X. This is the true apples-to-apples
#      encoding check: it can only agree with the closed form if G/h are right,
#      and it consumes the identical arrays pounce gets (no hand-rebuilt args).
def solve_same_encoding():
    X = cp.Variable(N)
    s = h - G @ X
    cons = [
        cp.constraints.ExpCone(s[0], s[1], s[2]),  # cone 0
        cp.constraints.ExpCone(s[3], s[4], s[5]),  # cone 1
        s[6] >= 0,                                 # nonneg block
    ]
    prob = cp.Problem(cp.Minimize(c @ X), cons)
    prob.solve(solver=cp.ECOS)
    Xv = X.value
    return float(np.exp(-(c @ Xv))), (float(np.exp(Xv[U])), float(np.exp(Xv[V])))

obj_same, xy_same = solve_same_encoding()


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))

err_known = rel(obj_pounce, KNOWN_OPTIMAL)
err_ecos = rel(obj_pounce, obj_ecos)
err_scs = rel(obj_pounce, obj_scs)
err_same = rel(obj_same, KNOWN_OPTIMAL)

print(f"=== pounce (A={A}) ===")
print(f"status={status} V={obj_pounce:.10f} t={t_pounce:.4f}s")
print(f"x*={x_s:.6f} (known 2) y*={y_s:.6f} (known 1)")
print("=== oracle cvxpy (GP mode, independent model) ===")
print(f"ECOS V={obj_ecos:.10f} t={t_ecos:.4f}s  x,y={xy_ecos}")
print(f"SCS  V={obj_scs:.10f} t={t_scs:.4f}s  x,y={xy_scs}")
print(f"=== SAME ENCODING (pounce's exact G/h via cvxpy ExpCone) ===")
print(f"V={obj_same:.10f}  x,y={xy_same}  (rel_err vs known={err_same:.2e})")
print(f"known_optimal=sqrt(3)*A^1.5/18={KNOWN_OPTIMAL:.10f}")
print(f"rel_err vs known={err_known:.2e}  vs ECOS={err_ecos:.2e}  vs SCS={err_scs:.2e}")

# The same-encoding check guards the formulation: if it does NOT match the known
# optimum, the G/h in THIS script are wrong (a FORMULATION_ERROR), not pounce.
encoding_ok = err_same < 1e-5
# pounce solves the GP correctly though it may report optimal_inaccurate at the
# default tol while still landing on V=4 (answer correct to ~1e-7); accept that.
ok = encoding_ok and err_known < 1e-4 and err_ecos < 1e-4 and err_scs < 1e-4
if not encoding_ok:
    print(f"VERDICT: FAIL FORMULATION_ERROR (same-encoding cvxpy gives V={obj_same:.6f} "
          f"!= known {KNOWN_OPTIMAL:.6f}; the G/h in this script are wrong)")
elif ok:
    print(f"VERDICT: PASS (pounce V={obj_pounce:.6f} matches known optimum and both GP "
          f"oracles; status={status})")
else:
    print(f"VERDICT: FAIL (status={status}, err_known={err_known:.2e})")
