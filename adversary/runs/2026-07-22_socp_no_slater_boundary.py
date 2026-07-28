"""Adversary cross-check: SOCP with NO Slater point (feasible set on the cone boundary)
Family: socp   Class: degenerate / constraint-qualification failure / non-unique optimum

Construction (self-derived; the "no relative interior" pathology is the standard
Slater-failure example, cf. Ben-Tal & Nemirovski, *Lectures on Modern Convex
Optimization* (SIAM 2001) Sec. 2.4 on conic duality and the Slater condition,
and Boyd & Vandenberghe, *Convex Optimization* (2004) Sec. 5.2.3):

    variables z = (t, x1, x2, y)
    minimize   c'z
    s.t.       (t, x1, x2) in SOC3     i.e.  t >= sqrt(x1^2 + x2^2)
               t - x1 = 0                    <-- kills the cone interior
               x1 + y = 3
               y >= 0

The equality t = x1 combined with t >= ||(x1,x2)|| forces x2 = 0 and x1 >= 0, so
    t^2 - x1^2 - x2^2 == 0
identically on the feasible set: the SOC is active with EXACTLY ZERO slack at
every feasible point.  There is no strictly feasible point => Slater fails =>
the central path is degenerate and the dual optimal set is unbounded.

Feasible set is the segment { (a, a, 0, 3-a) : a in [0,3] }.

Variant A (unique optimum):   c = (0, -2, 0, 3)
    obj(a) = -2a + 3(3-a) = 9 - 5a, minimized at a = 3.
    ANALYTIC OPTIMUM = -6 at z* = (3, 3, 0, 0)   (y=0 also active: doubly degenerate)

Variant B (non-unique optimal face):  c = (0, 1, 0, 1)
    obj(a) = a + (3-a) = 3 for ALL a in [0,3].
    ANALYTIC OPTIMUM = 3, attained on the ENTIRE feasible segment.
"""

import time

import numpy as np

# ---------------------------------------------------------------- problem data
# z = (t, x1, x2, y)
A = np.array(
    [
        [1.0, -1.0, 0.0, 0.0],  # t - x1 = 0
        [0.0, 1.0, 0.0, 1.0],  # x1 + y = 3
    ]
)
b = np.array([0.0, 3.0])

# s = h - G z must lie in the cones.  Want s = (t, x1, x2 | y).
G = np.zeros((4, 4))
G[0, 0] = -1.0  # s0 = t
G[1, 1] = -1.0  # s1 = x1
G[2, 2] = -1.0  # s2 = x2
G[3, 3] = -1.0  # s3 = y
h = np.zeros(4)
CONES = [("soc", 3), ("nonneg", 1)]

VARIANTS = {
    "A_unique": (np.array([0.0, -2.0, 0.0, 3.0]), -6.0),
    "B_nonunique_face": (np.array([0.0, 1.0, 0.0, 1.0]), 3.0),
}


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# ------------------------------------------------------------- sanity: encoding
# Re-derive the cone layout before trusting anything (rule: encoding is the trap).
z_probe = np.array([2.0, 2.0, 0.0, 1.0])  # t=2, x1=2, x2=0, y=1  -> feasible-ish
s_probe = h - G @ z_probe
assert np.allclose(s_probe, [2.0, 2.0, 0.0, 1.0]), s_probe
assert s_probe[0] >= np.linalg.norm(s_probe[1:3]) - 1e-12  # SOC block ok
assert s_probe[3] >= 0  # nonneg block ok
assert np.allclose(A @ np.array([3.0, 3.0, 0.0, 0.0]), b)  # z* is feasible

# Slater check: max over feasible z of (t - ||(x1,x2)||) is identically 0.
for a in (0.0, 0.7, 1.5, 3.0):
    z = np.array([a, a, 0.0, 3.0 - a])
    assert abs(A @ z - b).max() < 1e-14
    assert abs(z[0] - np.linalg.norm(z[1:3])) < 1e-14  # zero slack everywhere
print("encoding + no-Slater sanity checks: OK (SOC slack == 0 on all of F)")

from pounce import solve_socp  # noqa: E402

import cvxpy as cp  # noqa: E402

results = {}
for name, (c, known) in VARIANTS.items():
    # --- pounce ---
    t0 = time.perf_counter()
    r = solve_socp(c=c, A=A, b=b, G=G, h=h, cones=CONES)
    t_p = time.perf_counter() - t0
    zp = np.asarray(r.x, dtype=float)
    obj_p = float(r.obj)

    # --- oracle 1: cvxpy / CLARABEL ---
    z = cp.Variable(4)
    cons = [
        A @ z == b,
        cp.SOC(z[0], cp.hstack([z[1], z[2]])),
        z[3] >= 0,
    ]
    prob = cp.Problem(cp.Minimize(c @ z), cons)
    t0 = time.perf_counter()
    prob.solve(solver=cp.CLARABEL)
    t_cl = time.perf_counter() - t0
    obj_cl, z_cl = prob.value, z.value

    # --- oracle 2: cvxpy / SCS (independent algorithm: ADMM, not IPM) ---
    prob2 = cp.Problem(cp.Minimize(c @ z), cons)
    t0 = time.perf_counter()
    prob2.solve(solver=cp.SCS, eps=1e-10, max_iters=200_000)
    t_scs = time.perf_counter() - t0
    obj_scs = prob2.value

    # --- oracle 3: cvxpy / ECOS ---
    try:
        prob3 = cp.Problem(cp.Minimize(c @ z), cons)
        t0 = time.perf_counter()
        prob3.solve(solver=cp.ECOS, abstol=1e-11, reltol=1e-11)
        t_ecos = time.perf_counter() - t0
        obj_ecos = prob3.value
    except Exception as exc:  # pragma: no cover
        obj_ecos, t_ecos = None, float("nan")
        print(f"  ECOS unavailable: {exc}")

    # primal feasibility of pounce's point, checked independently of pounce
    eq_res = float(np.abs(A @ zp - b).max())
    soc_res = float(np.linalg.norm(zp[1:3]) - zp[0])  # <= 0 required
    nn_res = float(-min(zp[3], 0.0))

    print(f"\n=== variant {name} ===")
    print(f"pounce   status={r.status} obj={obj_p:.12e} t={t_p:.4f}s")
    print(f"         z={np.array2string(zp, precision=9)}")
    print(f"         eq_res={eq_res:.2e} soc_res={soc_res:.2e} nonneg_res={nn_res:.2e}")
    print(f"CLARABEL obj={obj_cl:.12e} t={t_cl:.4f}s")
    print(f"SCS      obj={obj_scs:.12e} t={t_scs:.4f}s")
    if obj_ecos is not None:
        print(f"ECOS     obj={obj_ecos:.12e} t={t_ecos:.4f}s")
    print(f"analytic obj={known:.12e}")

    # Oracles must agree with EACH OTHER before we trust them.
    oracle_vals = [v for v in (obj_cl, obj_scs, obj_ecos) if v is not None]
    oracle_spread = max(oracle_vals) - min(oracle_vals)
    print(f"oracle_spread={oracle_spread:.2e}")
    ok_oracles = oracle_spread < 1e-5 and all(rel(v, known) < 1e-5 for v in oracle_vals)

    err_known = rel(obj_p, known)
    err_oracle = rel(obj_p, obj_cl)
    print(f"pounce rel_err_vs_analytic={err_known:.2e} rel_err_vs_CLARABEL={err_oracle:.2e}")

    if name == "A_unique":
        z_star = np.array([3.0, 3.0, 0.0, 0.0])
        x_err = float(np.abs(zp - z_star).max())
        print(f"x_inf_err_vs_analytic_argmin={x_err:.2e}")
    else:
        # non-unique face: only the objective and feasibility are well defined
        x_err = 0.0
        a = zp[0]
        print(f"(non-unique face; pounce landed at a={a:.6f} in [0,3])")

    feas_ok = eq_res < 1e-7 and soc_res < 1e-7 and nn_res < 1e-9
    ok = (
        (r.status in ("optimal", "success") or getattr(r, "success", False))
        and err_known < 1e-6
        and err_oracle < 1e-6
        and feas_ok
        and (x_err < 1e-5)
    )
    results[name] = (ok, ok_oracles, err_known, err_oracle, t_p, t_cl, r.status, obj_p, obj_cl)
    print(f"variant {name}: {'PASS' if ok else 'FAIL'} (oracles_self_consistent={ok_oracles})")

all_ok = all(v[0] for v in results.values())
all_or = all(v[1] for v in results.values())
print()
if not all_or:
    print("VERDICT: INCONCLUSIVE (oracles disagreed with each other / analytic value)")
else:
    print("VERDICT: PASS" if all_ok else "VERDICT: FAIL")
