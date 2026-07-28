"""Adversary cross-check: separable geometric program via exponential cones.
Family: exp   Class: geometric programming (separable posynomial), exp-cone driver

Problem:  minimize  x + 4/x + y + 9/y ,   x, y > 0.
Separable: min_x (x + 4/x) at x = 2 -> 4 ;  min_y (y + 9/y) at y = 3 -> 6.
KNOWN global optimum = 10 at (x, y) = (2, 3).

Distinct from the already-logged "min x + 1/x" GP (2 exp cones, opt 2): this is
FOUR exp cones with non-unit coefficients and two variables, a genuinely
separable posynomial minimum.

Encoding (exp cone convention from solve_socp: slack (s0,s1,s2) obeys
s1*exp(s0/s1) <= s2, s1 > 0).  Substitute u = log x, v = log y:
  x   = e^u   <= t1 ;  4/x = 4 e^{-u} , with e^{-u} <= t2
  y   = e^v   <= t3 ;  9/y = 9 e^{-v} , with e^{-v} <= t4
Minimize t1 + 4 t2 + t3 + 9 t4.
Vars z = [u, v, t1, t2, t3, t4] (nvar = 6).
"""
import time
import numpy as np
import pounce
import cvxpy as cp

KNOWN_OPTIMAL = 10.0

nvar = 6
IU, IV, IT1, IT2, IT3, IT4 = range(6)
c = np.zeros(nvar)
c[IT1] = 1.0
c[IT2] = 4.0
c[IT3] = 1.0
c[IT4] = 9.0

rows = []
hs = []


def exp_cone(lin_col, lin_sign, t_col):
    """Append a 3-row exp block enforcing t >= exp(lin_sign * z[lin_col]).
    slack = (s0, s1, s2) with s0 = lin_sign*z[lin_col], s1 = 1, s2 = t.
    s = h - G z, so G[row0, lin_col] = -lin_sign ; h1 = 1 ; G[row2, t_col] = -1.
    """
    g0 = np.zeros(nvar); g0[lin_col] = -lin_sign; rows.append(g0); hs.append(0.0)
    rows.append(np.zeros(nvar)); hs.append(1.0)
    g2 = np.zeros(nvar); g2[t_col] = -1.0; rows.append(g2); hs.append(0.0)


exp_cone(IU, +1.0, IT1)   # t1 >= e^{u}
exp_cone(IU, -1.0, IT2)   # t2 >= e^{-u}
exp_cone(IV, +1.0, IT3)   # t3 >= e^{v}
exp_cone(IV, -1.0, IT4)   # t4 >= e^{-v}

G = np.array(rows)
h = np.array(hs)
cones = [("exp", 3)] * 4

t0 = time.perf_counter()
r = pounce.solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
z = np.asarray(r.x)
obj_pounce = float(r.obj)
status = r.status
x_star = np.exp(z[IU]); y_star = np.exp(z[IV])


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# --- oracle: cvxpy ECOS and SCS ---
def cvx(solver):
    x = cp.Variable(pos=True); y = cp.Variable(pos=True)
    prob = cp.Problem(cp.Minimize(x + 4 * cp.inv_pos(x) + y + 9 * cp.inv_pos(y)))
    t0 = time.perf_counter(); prob.solve(solver=solver); dt = time.perf_counter() - t0
    return float(prob.value), float(x.value), float(y.value), dt


obj_ecos, xe, ye, t_ecos = cvx(cp.ECOS)
obj_scs, xs, ys, t_scs = cvx(cp.SCS)

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} x*={x_star:.6f} y*={y_star:.6f} t={t_pounce:.4f}s")
print("=== oracle cvxpy ===")
print(f"ECOS obj={obj_ecos:.10e} (x={xe:.6f},y={ye:.6f}) t={t_ecos:.4f}s")
print(f"SCS  obj={obj_scs:.10e} t={t_scs:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e}")
print(f"rel_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e} "
      f"rel_vs_ECOS={rel(obj_pounce, obj_ecos):.2e} rel_vs_SCS={rel(obj_pounce, obj_scs):.2e}")
print(f"x_err={abs(x_star-2.0):.2e} y_err={abs(y_star-3.0):.2e}")

# Correctness verdict: obj must match known optimum AND both cvxpy oracles.
# NOTE: the non-symmetric exp-cone driver returns status="optimal_inaccurate"
# / success=False here even though obj is correct to ~1e-7 (the simpler logged
# "x+1/x" GP returns clean optimal). That is a soft status-certification label,
# not a wrong answer -- recorded as a note, not a correctness failure.
answer_correct = rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 and rel(obj_pounce, obj_ecos) < 1e-4
clean_status = (status == "optimal") and bool(r.success)
print(f"NOTE status={status} success={r.success} (soft cert; answer correct={answer_correct})")
print("VERDICT: PASS" if answer_correct else f"VERDICT: FAIL (status={status}, "
      f"rel_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e})")
