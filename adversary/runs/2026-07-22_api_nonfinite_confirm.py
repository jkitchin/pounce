"""Confirmation probes for the three findings from
2026-07-22_api_nonfinite_input.py.

F1: solve_qp(lb=+inf) / solve_qp(ub=-inf) -> 'optimal', bound ignored
    (correct answer is primal_infeasible / a ValueError).
F2: pounce.minimize with NaN in x0 -> success=True, fun=nan, x contains nan.
"""
import numpy as np
import pounce
from pounce import minimize, solve_qp

print("pounce", pounce.__version__)
np.set_printoptions(precision=6)


def base():
    P = np.array([[4.0, 1, 0, 0], [1, 3, 1, 0], [0, 1, 2, 1], [0, 0, 1, 5]])
    c = np.array([-1.0, -2, -3, -4])
    A = np.array([[1.0, 1, 1, 1]])
    b = np.array([2.0])
    G = np.array([[1.0, 0, 0, 0], [0, 1, -1, 0]])
    h = np.array([0.5, 0.3])
    return P, c, A, b, G, h


P, c, A, b, G, h = base()

print("\n--- F1: lb=+inf (x >= +inf is INFEASIBLE) ---")
for label, lb, ub in [
    ("lb=+inf (all)", np.full(4, np.inf), None),
    ("ub=-inf (all)", None, np.full(4, -np.inf)),
    ("lb[0]=+inf only", np.array([np.inf, -5, -5, -5]), None),
    ("ub[0]=-inf only", None, np.array([-np.inf, 5, 5, 5])),
    ("lb=+1e12 (finite analogue)", np.full(4, 1e12), None),
    ("ub=-1e12 (finite analogue)", None, np.full(4, -1e12)),
]:
    r = solve_qp(P=P, c=c, A=A, b=b, G=G, h=h, lb=lb, ub=ub)
    print("  %-28s status=%-20s obj=%s" % (label, r.status, r.obj))
    if r.status == "optimal" and lb is not None:
        viol = float(np.max(np.asarray(lb) - np.asarray(r.x)))
        print("      max lb violation = %s   x=%s" % (viol, np.asarray(r.x)))
    if r.status == "optimal" and ub is not None:
        viol = float(np.max(np.asarray(r.x) - np.asarray(ub)))
        print("      max ub violation = %s   x=%s" % (viol, np.asarray(r.x)))

print("\n--- F2: NaN in x0 through pounce.minimize ---")


def f(x):
    return float((x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2)


def g(x):
    return np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)])


for label, x0 in [
    ("x0=[1, nan]", np.array([1.0, np.nan])),
    ("x0=[nan, nan]", np.array([np.nan, np.nan])),
    ("x0=[nan, 2]", np.array([np.nan, 2.0])),
    ("x0=[0,0] control", np.array([0.0, 0.0])),
]:
    r = minimize(f, x0, jac=g)
    x = np.asarray(r.x)
    print(
        "  %-20s success=%-5s status=%-4s msg=%-42s fun=%-10s x=%s nfev=%s nit=%s"
        % (
            label,
            r.success,
            getattr(r, "status", "?"),
            str(getattr(r, "message", ""))[:42],
            r.fun,
            x,
            getattr(r, "nfev", "?"),
            getattr(r, "nit", "?"),
        )
    )

print("\n--- F2b: NaN in x0 with bounds/constraints ---")
r = minimize(f, np.array([1.0, np.nan]), jac=g, bounds=[(-10, 10), (-10, 10)])
print("  bounded : success=%s fun=%s x=%s" % (r.success, r.fun, np.asarray(r.x)))
r = minimize(
    f,
    np.array([1.0, np.nan]),
    jac=g,
    constraints=[{"type": "eq", "fun": lambda x: np.array([x[0] + x[1] - 1.0])}],
)
print("  eq-con  : success=%s fun=%s x=%s" % (r.success, r.fun, np.asarray(r.x)))

print("\n--- F2c: NaN in minimize bounds ---")
for label, bnds in [
    ("lb nan", [(np.nan, 10), (-10, 10)]),
    ("ub nan", [(-10, np.nan), (-10, 10)]),
    ("lb=+inf", [(np.inf, 10), (-10, 10)]),
    ("ub=-inf", [(-10, -np.inf), (-10, 10)]),
]:
    try:
        r = minimize(f, np.array([0.0, 0.0]), jac=g, bounds=bnds)
        print("  %-10s success=%s fun=%s x=%s" % (label, r.success, r.fun, np.asarray(r.x)))
    except BaseException as e:
        print("  %-10s raised %s: %s" % (label, type(e).__name__, str(e)[:100]))

print("\n--- F2d: NaN returned by the objective / gradient ---")


def fnan(x):
    return np.nan


r = minimize(fnan, np.array([0.0, 0.0]), jac=lambda x: np.zeros(2))
print("  f->nan   : success=%s status=%s fun=%s x=%s" % (r.success, r.status, r.fun, np.asarray(r.x)))


def gnan(x):
    return np.array([np.nan, np.nan])


r = minimize(f, np.array([0.0, 0.0]), jac=gnan)
print("  jac->nan : success=%s status=%s fun=%s x=%s" % (r.success, r.status, r.fun, np.asarray(r.x)))
