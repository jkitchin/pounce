"""Adversary check: POUNCE reports local infeasibility on HS13, which is feasible.

Family: nlp   Class: false LocalInfeasibility verdict
Source: Hock & Schittkowski (1981), problem 13. Known optimum f* = 1 at x* = (1, 0).
Started from x0 = (1e4, 1e4).

    min  (x1 - 2)^2 + x2^2
    s.t. (1 - x1)^3 - x2 >= 0
         x1 >= 0, x2 >= 0

HS13 is the classic LICQ/MFCQ-failure problem: the active constraint's gradient
vanishes at x*, so it is genuinely hard. But it is FEASIBLE, and Ipopt converges
to f = 0.98492872 in 29 iterations from this start. POUNCE stops after 12 with
`Converged to a point of local infeasibility`.

`LocalInfeasibility` is a strong claim. Ipopt's own semantics (and POUNCE's doc
comment) are that the iterate has converged to a *stationary point of the
constraint violation* whose residual is bounded away from zero — i.e. no local
move reduces the infeasibility. That is a checkable statement, and this script
checks it at the exact point POUNCE returns:

  theta(x)  = max(0, x2 - (1 - x1)^3)          (violation of the >= constraint)
  grad theta = (3 (1 - x1)^2, 1)               where theta > 0

If grad theta is far from zero, and no active bound blocks the descent
direction, the point is not stationary for the infeasibility and the verdict is
false. As a second, assumption-free check, the script simply walks downhill from
POUNCE's point and reports whether theta drops.
"""

import numpy as np

# The point POUNCE returns (from the .sol), and Ipopt's for comparison.
X_POUNCE = np.array([1.569_799_743_337_755_96, 3.174_354_251_066_263_12e-1])
LB = np.array([0.0, 0.0])


def theta(x):
    """Constraint violation of (1 - x1)^3 - x2 >= 0."""
    return max(0.0, x[1] - (1.0 - x[0]) ** 3)


def grad_theta(x):
    """Gradient of theta where it is positive."""
    return np.array([3.0 * (1.0 - x[0]) ** 2, 1.0])


def obj(x):
    return (x[0] - 2.0) ** 2 + x[1] ** 2


def main():
    x = X_POUNCE
    t, g = theta(x), grad_theta(x)
    print(f"POUNCE's returned point:  x = {x}")
    print(f"  objective                 f = {obj(x):.10f}")
    print(f"  constraint violation  theta = {t:.10f}")
    print(f"  grad theta                  = {g}  (norm {np.linalg.norm(g):.6f})")

    # Is any bound active in a way that blocks the descent direction -grad?
    d = -g
    at_lb = np.isclose(x, LB, atol=1e-8)
    blocked = at_lb & (d < 0)
    print(f"  at lower bound              = {at_lb}")
    print(f"  descent direction blocked   = {blocked}")

    print()
    print("Walking downhill on theta from POUNCE's point:")
    print(f"  {'step':>10} {'theta':>16} {'feasible?':>10}")
    best = (t, x)
    for step in (1e-3, 1e-2, 1e-1, 0.3, 0.5, 1.0):
        y = np.maximum(x + step * d / np.linalg.norm(d), LB)
        ty = theta(y)
        print(f"  {step:>10.3f} {ty:>16.10f} {str(ty <= 1e-12):>10}")
        if ty < best[0]:
            best = (ty, y)

    # Assumption-free: a short projected-gradient descent on theta alone.
    y = x.copy()
    for _ in range(200):
        if theta(y) <= 1e-14:
            break
        y = np.maximum(y - 0.05 * grad_theta(y) / max(1.0, np.linalg.norm(grad_theta(y))), LB)
    print()
    print(f"After 200 projected-gradient steps on theta alone:")
    print(f"  x = {y}   theta = {theta(y):.3e}   f = {obj(y):.6f}")

    print()
    stationary = np.linalg.norm(np.where(blocked, 0.0, g)) < 1e-6
    # Reachability is decided by the line search above, which lands exactly on
    # theta = 0. The projected-gradient run is only illustrative: it crawls,
    # because theta flattens cubically as x1 -> 1, so its residual after a fixed
    # 200 steps says nothing about whether a feasible point is reachable.
    reachable = best[0] <= 1e-12
    if not stationary and reachable:
        print("VERDICT: FAIL — POUNCE's point is NOT a stationary point of the")
        print("         infeasibility (grad theta is O(1), no bound blocks descent),")
        print("         and plain gradient descent on theta reaches a FEASIBLE point")
        print("         from it. The LocalInfeasibility verdict is false.")
    else:
        print("VERDICT: PASS / INCONCLUSIVE — the verdict is consistent with the point.")


if __name__ == "__main__":
    main()
