"""Adversary cross-check: min sum |x_i|^p s.t. a'x = b  at EXTREME power-cone alpha.
Family: power   Class: ill-conditioning / bad scaling -- power-cone barrier at
                       alpha -> 0 (p = 100) and alpha -> 1 (p = 1.01).
Source: Analytic Hoelder/Lagrange solution (Boyd & Vandenberghe, "Convex
        Optimization" (2004), Ex. 4.7 / App. A.1.6 dual-norm characterization;
        MOSEK Modeling Cookbook v3.3 sec. 4.2 "Power cone" for the
        |x|^p <= s  <=>  (x, s, 1) in P_{1/p}  epigraph).

Closed form.  With q = p/(p-1) (Hoelder conjugate) and S = sum_i |a_i|^q:
        x_i^* = sign(a_i) |a_i|^{q-1} (b/S),      f^* = (b/S)^p S.
Derivation: stationarity p|x_i|^{p-1}sgn(x_i) = lam a_i gives
x_i = sgn(a_i)|lam a_i/p|^{1/(p-1)}; a'x = b fixes |lam/p|^{1/(p-1)} = b/S.

WHY THIS IS THE ILL-CONDITIONING PROBE.  The epigraph |x|^p <= s uses the power
cone with alpha = 1/p, so:
  * p = 100    -> alpha = 0.01     (cone degenerates toward the z-axis half-line)
  * p = 1.01   -> alpha = 0.990099 (cone degenerates toward the y-axis; the
                  solution concentrates on the largest |a_i|, an L1-like vertex)
In both regimes the barrier F = -log(y^{2a} z^{2-2a} - x^2) - (1-a)log y - a log z
has wildly asymmetric curvature in y vs z, and the Hessian condition number of
the scaling blows up.  Both instances are scaled so the ANALYTIC optimum stays
O(1) -- any error is the barrier/IPM, not float overflow in the reference.

CONE CONVENTION -- verified read-only against
  crates/pounce-convex/src/cones/power.rs:
      K_alpha = {(x, y, z) : |x| <= y^alpha z^(1-alpha), y, z >= 0},  alpha in (0,1)
      (PowerCone::new asserts 0 < alpha < 1; 0.01 and 0.990099 are both legal)
  and python/pounce/qp.py::solve_socp: ("pow", alpha) -- second element is the
  EXPONENT, not a dimension; slack s = h - Gx must lie in the cone.
  So |x_i|^p <= s_i  <=>  |x_i| <= s_i^{1/p} * 1^{1-1/p}
      <=>  (x_i, s_i, 1) in K_{1/p}, i.e. alpha = 1/p.
  NOTE cvxpy's PowCone3D(u,v,w,alpha) is u^alpha v^{1-alpha} >= |w|, i.e. the
  triple order is PERMUTED relative to pounce: pounce (x,y,z) = cvxpy (w,u,v).
  Oracles below are built BOTH via the DCP cp.power atom and via the explicit
  permuted PowCone3D, so a cone-order mistake cannot hide.
"""

import time

import numpy as np

import cvxpy as cp

from pounce import solve_socp


def analytic(p, a, bval):
    q = p / (p - 1.0)
    S = float(np.sum(np.abs(a) ** q))
    x_star = np.sign(a) * np.abs(a) ** (q - 1.0) * (bval / S)
    f_star = (bval / S) ** p * S
    return q, S, x_star, float(f_star)


def solve_pounce(p, a, bval):
    """min sum s_i  s.t. a'x = b, (x_i, s_i, 1) in K_{1/p}."""
    nvar = a.size
    n = 2 * nvar
    c = np.zeros(n)
    c[nvar:] = 1.0
    A = np.zeros((1, n))
    A[0, :nvar] = a
    rows, h = [], []
    for i in range(nvar):
        r0 = np.zeros(n)
        r0[i] = -1.0
        rows.append(r0)
        h.append(0.0)  # slack = x_i
        r1 = np.zeros(n)
        r1[nvar + i] = -1.0
        rows.append(r1)
        h.append(0.0)  # slack = s_i
        rows.append(np.zeros(n))
        h.append(1.0)  # slack = 1
    t0 = time.perf_counter()
    r = solve_socp(
        c=c,
        A=A,
        b=np.array([bval]),
        G=np.array(rows),
        h=np.array(h),
        cones=[("pow", 1.0 / p)] * nvar,
    )
    t = time.perf_counter() - t0
    return r, np.asarray(r.x)[:nvar], float(r.obj), t


def cvx_dcp(p, a, bval, solver):
    x = cp.Variable(a.size)
    pr = cp.Problem(cp.Minimize(cp.sum(cp.power(cp.abs(x), p))), [a @ x == bval])
    t0 = time.perf_counter()
    try:
        pr.solve(solver=solver)
    except Exception as exc:  # oracle may itself choke at extreme p
        return None, None, time.perf_counter() - t0, f"error: {type(exc).__name__}"
    return pr.value, np.asarray(x.value), time.perf_counter() - t0, pr.status


def cvx_powcone(p, a, bval):
    """Same model via the explicit permuted PowCone3D encoding."""
    nvar = a.size
    x = cp.Variable(nvar)
    s = cp.Variable(nvar, nonneg=True)
    cons = [a @ x == bval, cp.constraints.PowCone3D(s, np.ones(nvar), x, 1.0 / p)]
    pr = cp.Problem(cp.Minimize(cp.sum(s)), cons)
    t0 = time.perf_counter()
    try:
        pr.solve(solver=cp.CLARABEL)
    except Exception as exc:
        return None, None, time.perf_counter() - t0, f"error: {type(exc).__name__}"
    return pr.value, np.asarray(x.value), time.perf_counter() - t0, pr.status


def rel(u, v):
    return abs(u - v) / max(1.0, abs(v))


# -- two extreme-alpha instances, both scaled so the analytic optimum is O(1) --
CASES = [
    # p = 100 -> alpha = 0.01.  b = S makes b/S = 1, so f* = S and
    # x_i* = sign(a_i)|a_i|^{1/99} (all ~1), keeping everything O(1).
    ("p=100 (alpha=0.010000)", 100.0, np.array([2.0, -3.0, 1.0, 5.0]), None),
    # p = 1.01 -> alpha = 0.990099.  |a_i| <= 1 keeps S = sum|a_i|^101 finite;
    # two near-equal leading entries avoid a pure single-coordinate vertex.
    ("p=1.01 (alpha=0.990099)", 1.01, np.array([1.0, -0.98, 0.5]), 1.0),
]

overall_ok = True
for label, p, a, bval in CASES:
    q0 = p / (p - 1.0)
    if bval is None:  # choose b = S so that b/S = 1 exactly
        bval = float(np.sum(np.abs(a) ** q0))
    q, S, x_star, f_star = analytic(p, a, bval)
    assert abs(a @ x_star - bval) < 1e-9 * max(1.0, abs(bval)), "analytic x* infeasible"

    r, x_p, obj_p, t_p = solve_pounce(p, a, bval)
    o1, x1, t1, st1 = cvx_dcp(p, a, bval, cp.CLARABEL)
    o2, x2, t2, st2 = cvx_dcp(p, a, bval, cp.SCS)
    o3, x3, t3, st3 = cvx_powcone(p, a, bval)

    print(f"\n================ {label} ================")
    print(f"a={a} b={bval:.12g} q={q:.6f} S={S:.12g}")
    print(f"known_optimal={f_star:.12e}")
    print(f"x*          ={x_star}")
    print("=== pounce ===")
    print(f"status={r.status} obj={obj_p:.12e} t={t_p:.4f}s iters={r.iters} kkt={r.kkt_error:.2e}")
    print(f"x={x_p}")
    print(f"feas |a'x-b|={abs(float(a @ x_p) - bval):.2e}")
    for nm, o, xo, to, st in (
        ("CLARABEL (cp.power)", o1, x1, t1, st1),
        ("SCS      (cp.power)", o2, x2, t2, st2),
        ("CLARABEL (PowCone3D)", o3, x3, t3, st3),
    ):
        if o is None:
            print(f"=== oracle {nm} === status={st} (unavailable)")
        else:
            print(f"=== oracle {nm} === status={st} obj={o:.12e} t={to:.4f}s")
            print(f"    x={xo}")
            print(f"    rel_err_vs_known={rel(o, f_star):.2e}")

    e_known = rel(obj_p, f_star)
    print(f"rel_err_vs_known={e_known:.2e}")
    print(f"x_inf_err_vs_known={np.max(np.abs(x_p - x_star)):.2e}")
    for nm, o in (("CLARABEL", o1), ("SCS", o2), ("PowCone3D", o3)):
        if o is not None:
            print(f"obj_err_vs_{nm}={rel(obj_p, o):.2e}")

    ok = (r.status.startswith("optimal") or r.success) and e_known < 1e-4
    overall_ok &= ok
    print(f"CASE VERDICT: {'PASS' if ok else f'FAIL (status={r.status}, err={e_known:.2e})'}")


# ---------------------------------------------------------------------------
# CHARACTERIZATION: attainable KKT accuracy as a function of alpha = 1/p.
# Both arms use data scaled so the analytic optimum stays O(1), so any trend is
# the barrier/IPM and not overflow in the problem data.
#   alpha -> 0 arm: a = (2,-3,1,5), b = S  (=> b/S = 1, f* = S, x* ~ 1)
#   alpha -> 1 arm: a = (1,-0.98,0.5), b = 1, |a_i| <= 1 keeps S = sum|a_i|^q O(1)
# NOTE: the alpha -> 1 arm MUST use |a_i| <= 1.  Reusing a = (2,-3,1,5) there
# gives q = 101 and S ~ 4e70, and the resulting numerical_failure is a property
# of 1e70-scaled data, not of the cone -- a trap this script avoids on purpose.
# ---------------------------------------------------------------------------
print("\n================ alpha sweep (KKT accuracy vs alpha) ================")
print(f"{'p':>8} {'alpha':>9} {'pounce status':>21} {'it':>3} {'kkt':>9} {'relerr':>9} {'clarabel':>9}")
SWEEP = [(p, np.array([1.0, -0.98, 0.5]), 1.0) for p in (1.001, 1.01, 1.05)] + [
    (p, np.array([2.0, -3.0, 1.0, 5.0]), None) for p in (1.5, 2.0, 3.0, 10.0, 50.0, 100.0, 200.0)
]
for p, a, bv in SWEEP:
    if bv is None:
        bv = float(np.sum(np.abs(a) ** (p / (p - 1.0))))
    _, _, _, f_star = analytic(p, a, bv)
    r, _, obj_p, _ = solve_pounce(p, a, bv)
    o, _, _, st = cvx_dcp(p, a, bv, cp.CLARABEL)
    print(
        f"{p:>8} {1.0 / p:>9.6f} {r.status:>21} {r.iters:>3} {r.kkt_error:>9.2e} "
        f"{rel(obj_p, f_star):>9.2e} {st:>9}"
    )

print(f"\nVERDICT: {'PASS' if overall_ok else 'FAIL'}")
