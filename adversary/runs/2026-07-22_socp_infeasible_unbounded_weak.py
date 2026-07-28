"""Adversary cross-check: conic status reporting — infeasible / unbounded / WEAKLY infeasible SOCP
Family: socp   Class: status reporting (infeasibility & unboundedness certificates)

Sources / theory:
  - Farkas lemma for conic programs: Ben-Tal & Nemirovski, *Lectures on Modern
    Convex Optimization* (2001), Thm 1.4.1 (conic duality / infeasibility).
  - Weak infeasibility in SOCP: Lourenço, Muramatsu & Tsuchiya, "Solving SDP
    completely with an interior point oracle" / "Weak infeasibility in second
    order cone programming", Optim. Lett. 10 (2016) 1743-1755, Example 1.
    Also Pataki, "Bad semidefinite programs: they all look the same".

Standard form used by pounce.solve_socp:
    min c'x   s.t.  A x = b,   s = h - G x  in  K

--------------------------------------------------------------------------
(a) STRONGLY (primal) INFEASIBLE
    min x1 + x2  s.t.  ||x||_2 <= 1,  x1 + x2 = 3
    Infeasible because x1+x2 <= sqrt(2)*||x||_2 <= sqrt(2) < 3.
    Farkas certificate (improving ray): y = -1, z = (sqrt(2), -1, -1) in SOC_3
      A'y + G'z = 0 and b'y + h'z = -3 + sqrt(2) = -1.5858 < 0.

(b) UNBOUNDED (primal feasible, objective -> -inf; dual infeasible)
    min -x1  s.t.  (x1,x2,x3) in SOC_3,  x2 - x3 = 1
    Feasible point (1,1,0). Recession direction d = (sqrt(2), 1, 1):
      d in SOC_3 (boundary), A d = 0, c'd = -sqrt(2) < 0  =>  unbounded.

(c) WEAKLY INFEASIBLE  (the pathology)
    min 0  s.t.  (x1,x2,x3) in SOC_3,  x1 - x3 = 0,  x2 = 1
    Infeasible: needs x1 >= sqrt(1 + x1^2) > x1.
    But inf of the infeasibility measure is 0 and NOT attained:
      x = (T, 1, T) has dist(x, SOC_3) -> 0 as T -> inf.
    NO Farkas certificate exists:  A'y + G'z = 0 forces z = (y1, y2, -y1),
    and z in SOC_3 forces y2 = 0, whence b'y + h'z = y2 = 0, never < 0.
    So the problem is infeasible with no improving ray -> weakly infeasible.
"""

import time

import numpy as np

import cvxpy as cp
from pounce import solve_socp

SQ2 = np.sqrt(2.0)


def banner(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def run_pounce(**kw):
    t0 = time.perf_counter()
    try:
        r = solve_socp(**kw)
        dt = time.perf_counter() - t0
        return dict(status=str(r.status), obj=r.obj, x=None if r.x is None else np.asarray(r.x),
                    iters=r.iters, t=dt, err=None)
    except Exception as e:  # noqa: BLE001
        return dict(status=f"EXC:{type(e).__name__}", obj=None, x=None, iters=None,
                    t=time.perf_counter() - t0, err=str(e))


def run_cvxpy(build, solver):
    prob = build()
    t0 = time.perf_counter()
    try:
        prob.solve(solver=solver)
        return dict(status=prob.status, obj=prob.value, t=time.perf_counter() - t0, err=None)
    except Exception as e:  # noqa: BLE001
        return dict(status=f"EXC:{type(e).__name__}", obj=None,
                    t=time.perf_counter() - t0, err=str(e))


def show(tag, d):
    print(f"  {tag:<22} status={d['status']:<28} obj={d['obj']}  t={d['t']:.4f}s"
          + (f"  ({d['err'][:80]})" if d.get("err") else ""))


results = {}

# ---------------------------------------------------------------- (a)
banner("(a) STRONGLY INFEASIBLE SOCP:  min x1+x2  s.t. ||x||<=1, x1+x2=3")

# s = (1, x1, x2) in SOC_3   <=>  h - Gx  with h=(1,0,0), G=[[0,0],[-1,0],[0,-1]]
Ga = np.array([[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]])
ha = np.array([1.0, 0.0, 0.0])
Aa = np.array([[1.0, 1.0]])
ba = np.array([3.0])
ca = np.array([1.0, 1.0])

# --- analytic Farkas certificate check ---
y = np.array([-1.0])
z = np.array([SQ2, -1.0, -1.0])
stat = Aa.T @ y + Ga.T @ z
cone_ok = z[0] >= np.linalg.norm(z[1:]) - 1e-12
val = float(ba @ y + ha @ z)
print(f"  certificate: ||A'y+G'z||={np.linalg.norm(stat):.2e}  z in SOC_3: {cone_ok}  "
      f"b'y+h'z = {val:.6f} (<0 required)")
assert np.linalg.norm(stat) < 1e-12 and cone_ok and val < -1e-9
print("  -> ANALYTIC TRUTH: primal INFEASIBLE (strongly; improving ray exists)")

pa = run_pounce(c=ca, A=Aa, b=ba, G=Ga, h=ha, cones=[("soc", 3)])
show("pounce", pa)


def build_a():
    x = cp.Variable(2)
    return cp.Problem(cp.Minimize(cp.sum(x)), [cp.norm(x, 2) <= 1, cp.sum(x) == 3])


for s in (cp.CLARABEL, cp.SCS, cp.ECOS):
    d = run_cvxpy(build_a, s)
    show(f"cvxpy/{s}", d)
    results[("a", s)] = d
results[("a", "pounce")] = pa

# ---------------------------------------------------------------- (b)
banner("(b) UNBOUNDED SOCP:  min -x1  s.t. (x1,x2,x3) in SOC_3, x2-x3=1")

Gb = -np.eye(3)
hb = np.zeros(3)
Ab = np.array([[0.0, 1.0, -1.0]])
bb = np.array([1.0])
cb = np.array([-1.0, 0.0, 0.0])

x_feas = np.array([1.0, 1.0, 0.0])
d_rec = np.array([SQ2, 1.0, 1.0])
print(f"  feasible pt (1,1,0): cone ok={x_feas[0] >= np.linalg.norm(x_feas[1:]) - 1e-12}, "
      f"Ax-b={float((Ab @ x_feas - bb)[0]):.1e}")
print(f"  recession dir (sqrt2,1,1): in cone={d_rec[0] >= np.linalg.norm(d_rec[1:]) - 1e-12}, "
      f"A d={float((Ab @ d_rec)[0]):.1e}, c'd={float(cb @ d_rec):.6f} (<0 required)")
assert cb @ d_rec < 0 and np.abs(Ab @ d_rec).max() < 1e-12
print("  -> ANALYTIC TRUTH: primal FEASIBLE and UNBOUNDED (obj -> -inf)")

pb = run_pounce(c=cb, A=Ab, b=bb, G=Gb, h=hb, cones=[("soc", 3)])
show("pounce", pb)


def build_b():
    x = cp.Variable(3)
    return cp.Problem(cp.Minimize(-x[0]), [cp.norm(x[1:], 2) <= x[0], x[1] - x[2] == 1])


for s in (cp.CLARABEL, cp.SCS, cp.ECOS):
    d = run_cvxpy(build_b, s)
    show(f"cvxpy/{s}", d)
    results[("b", s)] = d
results[("b", "pounce")] = pb

# ---------------------------------------------------------------- (c)
banner("(c) WEAKLY INFEASIBLE SOCP:  min 0  s.t. (x1,x2,x3) in SOC_3, x1-x3=0, x2=1")

Gc = -np.eye(3)
hc = np.zeros(3)
Ac = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
bc = np.array([0.0, 1.0])
cc = np.zeros(3)

print("  infeasibility measure along x=(T,1,T):")
for T in (1.0, 1e1, 1e3, 1e6):
    xT = np.array([T, 1.0, T])
    gap = np.linalg.norm(xT[1:]) - xT[0]  # >0 means violated
    print(f"    T={T:9.0e}  ||(x2,x3)|| - x1 = {gap:.3e}  (>0 => infeasible, ->0 as T->inf)")
# no-certificate proof, numerically corroborated: maximize b'y+h'z s.t. A'y+G'z=0, z in K*
yv = cp.Variable(2)
zv = cp.Variable(3)
cert = cp.Problem(cp.Minimize(bc @ yv + hc @ zv),
                  [Ac.T @ yv - zv == 0, cp.norm(zv[1:], 2) <= zv[0],
                   cp.norm(yv, 2) <= 1])
cert.solve(solver=cp.CLARABEL)
print(f"  best improving-ray value over ||y||<=1 : {cert.value:.3e} "
      "(must be < 0 for a Farkas certificate; it is exactly 0 analytically)")
print("  -> ANALYTIC TRUTH: primal INFEASIBLE, but WEAKLY (no improving ray)")

pc = run_pounce(c=cc, A=Ac, b=bc, G=Gc, h=hc, cones=[("soc", 3)])
show("pounce", pc)
if pc["x"] is not None:
    xr = pc["x"]
    print(f"    returned x = {xr}")
    print(f"    cone violation ||(x2,x3)||-x1 = {np.linalg.norm(xr[1:]) - xr[0]:.3e}")
    print(f"    eq residual ||Ax-b||_inf     = {np.abs(Ac @ xr - bc).max():.3e}")


def build_c():
    x = cp.Variable(3)
    return cp.Problem(cp.Minimize(0),
                      [cp.norm(x[1:], 2) <= x[0], x[0] - x[2] == 0, x[1] == 1])


for s in (cp.CLARABEL, cp.SCS, cp.ECOS):
    d = run_cvxpy(build_c, s)
    show(f"cvxpy/{s}", d)
    results[("c", s)] = d
results[("c", "pounce")] = pc

# ---------------------------------------------------------------- grade
banner("GRADE")

INFEAS_WORDS = ("infeasible", "primal_infeasible")
UNBD_WORDS = ("unbounded", "dual_infeasible")


def is_infeas(s):
    s = s.lower()
    return "infeasible" in s and "dual_infeasible" not in s and "unbounded" not in s


def is_unbd(s):
    s = s.lower()
    return "unbounded" in s or "dual_infeasible" in s


def is_confident_optimal(s):
    s = s.lower()
    return s in ("optimal", "success", "solved", "optimal_inaccurate")


verdicts = []

# (a) must be infeasible (allow inaccurate variants)
sa = pa["status"]
ok_a = is_infeas(sa) or "infeasible" in sa.lower()
print(f"  (a) strongly infeasible : pounce={sa!r} -> {'OK' if ok_a else 'WRONG'}")
verdicts.append(ok_a)

sb = pb["status"]
ok_b = is_unbd(sb)
print(f"  (b) unbounded           : pounce={sb!r} -> {'OK' if ok_b else 'WRONG'}")
verdicts.append(ok_b)

sc = pc["status"]
# For weak infeasibility: anything except a CONFIDENT optimal claim is acceptable.
confident_wrong_c = is_confident_optimal(sc)
print(f"  (c) weakly infeasible   : pounce={sc!r} -> "
      f"{'CONFIDENT-WRONG (finding)' if confident_wrong_c else 'acceptable (non-optimal status)'}")
verdicts.append(not confident_wrong_c)

print("\n  oracle summary:")
for case in ("a", "b", "c"):
    row = "  ".join(f"{s}={results[(case, s)]['status']}" for s in (cp.CLARABEL, cp.SCS, cp.ECOS))
    print(f"    ({case}) {row}")

print("\nVERDICT: PASS" if all(verdicts) else "\nVERDICT: FAIL")
