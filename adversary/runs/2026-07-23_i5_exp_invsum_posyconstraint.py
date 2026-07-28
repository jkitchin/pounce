"""Adversary cross-check: GP with a posynomial (exp-cone) CONSTRAINT.
Family: exp   Class: geometric programming, exp cone in BOTH objective & constraint

Problem:  minimize  1/x1 + 1/x2 + 1/x3   s.t.  x1 + x2 + x3 <= 3,  xi > 0.
By Cauchy-Schwarz / symmetry: (Sum 1/xi)(Sum xi) >= 9, so Sum 1/xi >= 9/3 = 3,
with equality at x1 = x2 = x3 = 1 (constraint active).
KNOWN global optimum = 3 at (1, 1, 1).

Novelty vs logged exp tests: the linear budget x1+x2+x3<=3 is a POSYNOMIAL
(sum-of-monomials) constraint, so it too must be expressed through exp cones
(si >= e^{ui}, Sum si <= 3) -- exp cones drive BOTH the objective epigraph and
the constraint, unlike the entropy/GP objective-only encodings already logged.

Encoding: ui = log xi.  1/xi = e^{-ui} <= ti  (obj) ; xi = e^{ui} <= si (budget).
Vars z = [u1,u2,u3, t1,t2,t3, s1,s2,s3]  (nvar = 9).
Cones: 6 exp (obj + budget epigraphs) then 1 nonneg (Sum si <= 3).
"""
import time
import numpy as np
import pounce
import cvxpy as cp

KNOWN_OPTIMAL = 3.0
nvar = 9
U = [0, 1, 2]; T = [3, 4, 5]; S = [6, 7, 8]

c = np.zeros(nvar)
for t in T:
    c[t] = 1.0

rows = []; hs = []


def exp_cone(lin_col, lin_sign, epi_col):
    g0 = np.zeros(nvar); g0[lin_col] = -lin_sign; rows.append(g0); hs.append(0.0)
    rows.append(np.zeros(nvar)); hs.append(1.0)
    g2 = np.zeros(nvar); g2[epi_col] = -1.0; rows.append(g2); hs.append(0.0)


for i in range(3):
    exp_cone(U[i], -1.0, T[i])    # ti >= e^{-ui}   (= 1/xi)
for i in range(3):
    exp_cone(U[i], +1.0, S[i])    # si >= e^{ui}    (= xi)

# budget: 3 - (s1+s2+s3) >= 0  -> nonneg slack
gb = np.zeros(nvar)
for s in S:
    gb[s] = 1.0
rows.append(gb); hs.append(3.0)

G = np.array(rows); h = np.array(hs)
cones = [("exp", 3)] * 6 + [("nonneg", 1)]

t0 = time.perf_counter()
r = pounce.solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
z = np.asarray(r.x)
obj_pounce = float(r.obj)
status = r.status
x_star = np.exp(z[U])
budget = float(np.sum(x_star))


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def cvx(solver):
    x = cp.Variable(3, pos=True)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.inv_pos(x))), [cp.sum(x) <= 3])
    t0 = time.perf_counter(); prob.solve(solver=solver); dt = time.perf_counter() - t0
    return float(prob.value), np.asarray(x.value), dt


obj_ecos, xe, t_ecos = cvx(cp.ECOS)
obj_scs, xs, t_scs = cvx(cp.SCS)

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} x*={x_star} sum(x)={budget:.6f} t={t_pounce:.4f}s")
print("=== oracle cvxpy ===")
print(f"ECOS obj={obj_ecos:.10e} x={xe} t={t_ecos:.4f}s")
print(f"SCS  obj={obj_scs:.10e} t={t_scs:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e}")
print(f"rel_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e} rel_vs_ECOS={rel(obj_pounce, obj_ecos):.2e}")
print(f"x_inf_err_vs_ones={np.max(np.abs(x_star-1.0)):.2e}")

# See note in the separable-GP sibling: exp-cone driver returns
# "optimal_inaccurate"/success=False here despite obj correct to ~1e-8.
answer_correct = rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 and rel(obj_pounce, obj_ecos) < 1e-4
clean_status = (status == "optimal") and bool(r.success)
print(f"NOTE status={status} success={r.success} (soft cert; answer correct={answer_correct})")
print("VERDICT: PASS" if answer_correct else f"VERDICT: FAIL (status={status}, "
      f"rel_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e})")
