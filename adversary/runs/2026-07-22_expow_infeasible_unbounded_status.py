"""Adversary cross-check: status reporting on non-conic-feasible exp/power instances.

Family: exp + power   Class: infeasibility / unboundedness / status reporting
Source: analytic construction; GP log-transform per Boyd, Kim, Vandenberghe &
Hassibi, "A tutorial on geometric programming", Optim Eng 8(1):67-127 (2007),
sec. 2.4 (posynomial -> convex form u = log x); Farkas' lemma (Boyd &
Vandenberghe, Convex Optimization (2004), sec. 5.8.3); MOSEK Modeling Cookbook
v3.3 sec. 5.2 (exp cone) and 4.1 (power cone).

CASES
  A-INF  infeasible GP     (delta = 4.0  >> delta* = 2 - 2 ln 2)
  A-NEAR infeasible GP     (delta = 0.63 > delta*, near-miss)
  A-FEAS feasible control  (delta = 0.60 < delta*, near-miss)
  B-UNB  unbounded exp-cone GP (min u, recession direction u -> -inf)
  B-BND  bounded control       (min -u, optimum -(1 - ln 2))
  C-INF  power cone domain violation: y-component forced <= -1 at every point
  C-FEAS feasible control: y-component reaches +0.05, optimum -sqrt(0.05)

ANALYTIC PROOFS  (all on the log-transformed / linear system)

Family A.  Variables u, v (u = log x, v = log y).
  C1:  exp(u + v - 1) + exp(u - v - 1) <= 1
  C2:  exp(-u - v + d - 1) + exp(-u + v + d - 1) <= 1
  Both LHS equal 2 e^{+-u - 1} cosh(v), minimized over v at v = 0, so the
  feasible set is nonempty iff it is nonempty on v = 0.  There:
       u <= 1 - ln 2      and      u >= d - 1 + ln 2 .
  Nonempty  <=>  d <= d* = 2 - 2 ln 2 = 0.6137056389.
  Farkas certificate for d > d*: multiply the two constraints,
       4 e^{d - 2} cosh^2(v) <= 1,  and cosh(v) >= 1
   =>  4 e^{d-2} <= 1  =>  d <= 2 - 2 ln 2.  Contradiction.
  In log space that product is exactly the nonnegative combination
  (1, 1) . [ (1,1;1,-1) u <= (1-ln..) ; ... ] summing to  0 <= -(d - d*) < 0,
  i.e. a Farkas vector y >= 0 with A^T y = 0 and b^T y < 0.

Family B.  min u  s.t.  exp(u + v - 1) + exp(u - v - 1) <= 1.
  From any feasible (u, v), the ray (u - t, v), t >= 0, is feasible (both
  exponentials strictly decrease) and drives the objective to -inf.  So the
  recession cone contains a direction of strictly negative cost => the primal
  is UNBOUNDED (dual infeasible).  Control: min -u has optimum -(1 - ln 2).

Family C.  K_alpha = {(x,y,z) : |x| <= y^alpha z^(1-alpha), y, z >= 0}.
  Cone triple is (w, t - 2, 1) with the extra linear constraint t <= T.
  Membership forces t >= 2.  With T = 1 the linear system {t <= 1, t >= 2} has
  the Farkas certificate (1, 1): 0 <= -1 < 0.  Infeasible at every point.
  Control T = 2.05: y = 0.05 attainable, min w = -sqrt(0.05 * 1).
"""

import time
import math
import numpy as np

LN2 = math.log(2.0)
DELTA_STAR = 2.0 - 2.0 * LN2  # 0.6137056389

from pounce import solve_socp  # noqa: E402

results = []  # (case, expect, pounce_status, oracle statuses, note)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# ---------------------------------------------------------------------------
# Family A: GP feasibility.  Variables z = (u, v, t1, t2, t3, t4)
#   ti are exp-cone epigraph vars:  ti >= exp(a_i^T (u,v) + b_i)
#   constraints  t1 + t2 <= 1  and  t3 + t4 <= 1
# Cone rows: for each i, slack triple (x, y, z) = (a_i^T z + b_i, 1, t_i).
# ---------------------------------------------------------------------------
def build_gp(delta, drop_c2=False):
    """drop_c2=True keeps only the C1 posynomial -> min u is UNBOUNDED below."""
    A_exp = [
        (1.0, 1.0, -1.0),  # u + v - 1
        (1.0, -1.0, -1.0),  # u - v - 1
    ]
    if not drop_c2:
        A_exp += [
            (-1.0, -1.0, delta - 1.0),  # -u - v + delta - 1
            (-1.0, 1.0, delta - 1.0),  # -u + v + delta - 1
        ]
    k = len(A_exp)
    n = 2 + k
    n_nn = k // 2
    G = np.zeros((3 * k + n_nn, n))
    h = np.zeros(3 * k + n_nn)
    for i, (au, av, b) in enumerate(A_exp):
        r = 3 * i
        # s_r = au*u + av*v + b  ->  h - G z, so G row = -(au, av), h = b
        G[r, 0] = -au
        G[r, 1] = -av
        h[r] = b
        h[r + 1] = 1.0  # s_{r+1} = 1
        G[r + 2, 2 + i] = -1.0  # s_{r+2} = t_i
    for j in range(n_nn):
        G[3 * k + j, 2 + 2 * j] = 1.0
        G[3 * k + j, 3 + 2 * j] = 1.0
        h[3 * k + j] = 1.0
    cones = [("exp", 3)] * k + [("nonneg", n_nn)]
    return G, h, cones, n


def run_gp(tag, delta, objective, expect, drop_c2=False, **kw):
    G, h, cones, n = build_gp(delta, drop_c2)
    c = np.zeros(n)
    if objective == "feas":
        pass  # pure feasibility (zero objective)
    elif objective == "min_u":
        c[0] = 1.0
    elif objective == "max_u":
        c[0] = -1.0
    t0 = time.perf_counter()
    r = solve_socp(c=c, G=G, h=h, cones=cones, **kw)
    t = time.perf_counter() - t0
    return r, t


# ---------------------------------------------------------------------------
# cvxpy oracles
# ---------------------------------------------------------------------------
import cvxpy as cp  # noqa: E402


def oracle_gp(delta, objective, solvers, drop_c2=False):
    out = {}
    for s in solvers:
        u = cp.Variable()
        v = cp.Variable()
        cons = [cp.exp(u + v - 1) + cp.exp(u - v - 1) <= 1]
        if not drop_c2:
            cons.append(
                cp.exp(-u - v + delta - 1) + cp.exp(-u + v + delta - 1) <= 1
            )
        if objective == "feas":
            obj = cp.Minimize(0)
        elif objective == "min_u":
            obj = cp.Minimize(u)
        else:
            obj = cp.Minimize(-u)
        p = cp.Problem(obj, cons)
        try:
            t0 = time.perf_counter()
            p.solve(solver=s)
            dt = time.perf_counter() - t0
            out[str(s)] = (p.status, p.value, dt)
        except Exception as e:  # noqa: BLE001
            out[str(s)] = (f"ERROR:{type(e).__name__}", None, 0.0)
    return out


# ---------------------------------------------------------------------------
# Family C: power cone domain violation.  z = (w, t)
#   pow cone alpha=0.5 on triple (w, t-2, 1);  nonneg: t <= T
# ---------------------------------------------------------------------------
def run_pow(T, alpha=0.5, **kw):
    n = 2
    G = np.zeros((4, n))
    h = np.zeros(4)
    G[0, 0] = -1.0
    h[0] = 0.0  # s0 = w
    G[1, 1] = -1.0
    h[1] = -2.0  # s1 = t - 2
    h[2] = 1.0  # s2 = 1
    G[3, 1] = 1.0
    h[3] = T  # s3 = T - t >= 0
    cones = [("pow", alpha), ("nonneg", 1)]
    c = np.array([1.0, 0.0])  # min w
    t0 = time.perf_counter()
    r = solve_socp(c=c, G=G, h=h, cones=cones, **kw)
    return r, time.perf_counter() - t0


def run_pow_const(**kw):
    """Degenerate probe: y-component is the CONSTANT -1 (no variable at all).
    (w, -1, 1) is in no power cone; infeasibility is visible in h alone."""
    G = np.zeros((3, 1))
    G[0, 0] = -1.0
    h = np.array([0.0, -1.0, 1.0])
    t0 = time.perf_counter()
    r = solve_socp(c=[1.0], G=G, h=h, cones=[("pow", 0.5)], **kw)
    return r, time.perf_counter() - t0


def run_exp_const(**kw):
    """Same shape but on the EXP cone: (x, y, z) = (w, -1, 1), y > 0 required.
    Localizes whether the detection gap is power-cone-specific."""
    G = np.zeros((3, 1))
    G[0, 0] = -1.0
    h = np.array([0.0, -1.0, 1.0])
    t0 = time.perf_counter()
    r = solve_socp(c=[1.0], G=G, h=h, cones=[("exp", 3)], **kw)
    return r, time.perf_counter() - t0


def oracle_pow(T, solvers):
    out = {}
    for s in solvers:
        w = cp.Variable()
        t = cp.Variable()
        # cvxpy PowCone3D(u, v, ww, alpha):  u^alpha v^(1-alpha) >= |ww|
        # pounce (x, y, z) = cvxpy (ww, u, v)
        cons = [cp.constraints.PowCone3D(t - 2, 1.0, w, 0.5), t <= T]
        p = cp.Problem(cp.Minimize(w), cons)
        try:
            t0 = time.perf_counter()
            p.solve(solver=s)
            dt = time.perf_counter() - t0
            out[str(s)] = (p.status, p.value, dt)
        except Exception as e:  # noqa: BLE001
            out[str(s)] = (f"ERROR:{type(e).__name__}", None, 0.0)
    return out


# ---------------------------------------------------------------------------
# Orientation sanity check FIRST: a known-feasible exp-cone GP with known answer
# min x + 1/x  = 2  (docstring example) -- confirms our exp encoding orientation.
# ---------------------------------------------------------------------------
print("=== orientation sanity check (exp cone, known optimum 2) ===")
G0 = np.zeros((6, 3))
G0[0, 0] = -1.0
G0[2, 1] = -1.0
G0[3, 0] = 1.0
G0[5, 2] = -1.0
r0 = solve_socp(c=[0, 1, 1], G=G0, h=[0, 1, 0, 0, 1, 0], cones=[("exp", 3), ("exp", 3)])
print(f"  status={r0.status} obj={r0.obj:.10f}  (expect optimal / 2.0)")
ORIENT_OK = r0.status.startswith("optimal") and abs(r0.obj - 2.0) < 1e-5

# power-cone orientation: min w s.t. (w, 4, 1) in K_0.5  -> w* = -2
Gp = np.zeros((3, 1))
Gp[0, 0] = -1.0
rp = solve_socp(c=[1.0], G=Gp, h=[0.0, 4.0, 1.0], cones=[("pow", 0.5)])
print(f"  power: status={rp.status} obj={rp.obj:.10f}  (expect optimal / -2.0)")
ORIENT_OK = ORIENT_OK and rp.status.startswith("optimal") and abs(rp.obj + 2.0) < 1e-5
print(f"  ORIENTATION_OK={ORIENT_OK}")
print()

SOLVERS = [cp.ECOS, cp.SCS, cp.CLARABEL]
POW_SOLVERS = [cp.SCS, cp.CLARABEL]  # ECOS has no power cone

print(f"delta* = 2 - 2 ln 2 = {DELTA_STAR:.12f}")
print()

CASES = [
    ("A-INF   GP delta=4.0", "primal_infeasible", 4.0, "feas", False),
    ("A-NEAR  GP delta=0.63", "primal_infeasible", 0.63, "feas", False),
    ("A-FEAS  GP delta=0.60", "optimal", 0.60, "feas", False),
    ("A-FEAS2 GP delta=0.60 min u", "optimal", 0.60, "min_u", False),
    ("B-UNB   C1 only, min u", "dual_infeasible", 0.0, "min_u", True),
    ("B-BND   C1 only, max u", "optimal", 0.0, "max_u", True),
    ("B-BND2  C1+C2, min u", "optimal", 0.0, "min_u", False),
]

for tag, expect, delta, obj, drop in CASES:
    r, t = run_gp(tag, delta, obj, expect, drop_c2=drop)
    orc = oracle_gp(delta, obj, SOLVERS, drop_c2=drop)
    orc_str = "  ".join(f"{k.split('.')[-1]}={v[0]}" for k, v in orc.items())
    val = f"{r.obj:.8e}" if r.obj is not None else "None"
    print(f"[{tag}]")
    print(f"  expect     : {expect}")
    print(f"  pounce     : status={r.status:22s} obj={val}  t={t:.4f}s")
    print(f"  oracles    : {orc_str}")
    ok = r.status == expect
    if expect == "optimal":
        ok = r.status.startswith("optimal")
    results.append((tag, expect, r.status, orc_str, ok, t))
    print(f"  MATCH      : {ok}")
    print()

# analytic optima:
#   max u under C1 at v=0 -> u = 1 - ln 2, so min(-u) = -(1 - ln 2)
#   min u under C1+C2 (delta=0) at v=0 -> u = ln 2 - 1
print(f"  [B-BND  analytic min(-u) under C1     = {-(1.0 - LN2):.12f}]")
print(f"  [B-BND2 analytic min(u)  under C1+C2  = {LN2 - 1.0:.12f}]")
print()

for tag, T, expect in [
    ("C-INF   pow T=1.0", 1.0, "primal_infeasible"),
    ("C-NEAR  pow T=1.999", 1.999, "primal_infeasible"),
    ("C-FEAS  pow T=2.05", 2.05, "optimal"),
]:
    r, t = run_pow(T)
    orc = oracle_pow(T, POW_SOLVERS)
    orc_str = "  ".join(f"{k.split('.')[-1]}={v[0]}" for k, v in orc.items())
    val = f"{r.obj:.8e}" if r.obj is not None else "None"
    print(f"[{tag}]")
    print(f"  expect     : {expect}")
    print(f"  pounce     : status={r.status:22s} obj={val}  t={t:.4f}s")
    print(f"  oracles    : {orc_str}")
    if T > 2.0:
        known = -math.sqrt((T - 2.0) * 1.0)
        print(f"  known opt  : {known:.12f}  rel_err={rel(r.obj, known):.2e}"
              if r.obj is not None else f"  known opt  : {known:.12f}")
    ok = r.status.startswith("optimal") if expect == "optimal" else r.status == expect
    results.append((tag, expect, r.status, orc_str, ok, t))
    print(f"  MATCH      : {ok}")
    print()

# ---------------------------------------------------------------------------
# Localization probes for the power-cone infeasibility (only if it mismatched)
# ---------------------------------------------------------------------------
print("=== localization probes: constant-y domain violation (w, -1, 1) ===")
rc, tc = run_pow_const()
print(f"  pow alpha=0.5 : status={rc.status:22s} obj={rc.obj}")
re_, te = run_exp_const()
print(f"  exp cone      : status={re_.status:22s} obj={re_.obj}")
print()

print("=== option sweep on C-INF (T=1.0, alpha=0.5) ===")
for tol in [None, 1e-6, 1e-10]:
    for mi in [None, 500, 2000]:
        r, t = run_pow(1.0, tol=tol, max_iter=mi)
        print(f"  tol={str(tol):8s} max_iter={str(mi):6s} -> {r.status:22s} "
              f"iters={r.iters}")
print()

print("=== alpha sweep on C-INF (T=1.0) ===")
for a in [0.1, 0.25, 0.5, 0.75, 0.9]:
    r, t = run_pow(1.0, alpha=a)
    print(f"  alpha={a:<5} -> {r.status}")
print()

# ---------------------------------------------------------------------------
# Driver baseline: does the SYMMETRIC (LP/SOC) path report these statuses?
# ---------------------------------------------------------------------------
from pounce import solve_qp  # noqa: E402

print("=== driver baseline: symmetric cones (nonneg / soc) ===")
rb = solve_qp(c=[1.0, 1.0], G=np.array([[1.0, 1.0], [-1.0, -1.0]]), h=[0.0, -1.0])
print(f"  LP  x1+x2<=0, -x1-x2<=-1 (Farkas (1,1))   -> {rb.status}")
rb = solve_qp(c=[-1.0], G=np.array([[-1.0]]), h=[0.0])
print(f"  LP  min -x s.t. x>=0     (recession ray)  -> {rb.status}")
rb = solve_socp(c=[-1.0, 0.0, 0.0], G=-np.eye(3), h=[0.0, 0.0, 0.0], cones=[("soc", 3)])
print(f"  SOC min -t s.t. t>=||x|| (recession ray)  -> {rb.status}")

# minimal unbounded exp-cone probe: (u, 1, t) in Kexp, min u  -> u -> -inf
Gm = np.zeros((3, 2))
Gm[0, 0] = -1.0
Gm[2, 1] = -1.0
rm = solve_socp(c=[1.0, 0.0], G=Gm, h=[0.0, 1.0, 0.0], cones=[("exp", 3)])
print(f"  EXP min u s.t. (u,1,t) in Kexp           -> {rm.status}  obj={rm.obj:.3e}")
print()

print("=== SUMMARY ===")
for tag, expect, got, orc, ok, t in results:
    print(f"{'OK ' if ok else 'BAD'} {tag:30s} expect={expect:20s} got={got}")

n_bad = sum(1 for r in results if not r[4])
print()
print(f"orientation_ok={ORIENT_OK}  mismatches={n_bad}/{len(results)}")
print("VERDICT: PASS" if (ORIENT_OK and n_bad == 0) else f"VERDICT: FAIL ({n_bad} status mismatches)")
