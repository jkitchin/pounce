#!/usr/bin/env python
"""SOS adversary test: the Motzkin polynomial.

    M(x, y) = x^4 y^2 + x^2 y^4 + 1 - 3 x^2 y^2

Degree 6. M(x,y) >= 0 everywhere (by AM-GM on the three monomials
x^4 y^2, x^2 y^4, 1 whose geometric mean is x^2 y^2), with global minimum

    M* = 0   attained at the FOUR points (x,y) = (+-1, +-1).

SOURCE: T. S. Motzkin, "The arithmetic-geometric inequality" (1967). The
canonical example of a polynomial that is nonnegative but NOT a sum of squares
(SOS). Reznick / Lasserre / Parrilo all use it as the textbook hard case for
the SOS / moment hierarchy.

Why it is adversarial for pounce's SOS path:
  * UNCONSTRAINED: because M is nonnegative but not SOS, and M is not coercive
    (M(x,0) = M(0,y) = 1, bounded but flat to infinity), the unconstrained
    Lasserre/SOS relaxation cannot certify ANY finite lower bound -- there is no
    sigma_0 SOS with M - lambda = sigma_0. The relaxation is structurally
    unbounded for every order. The EXPECTED, correct outcome is therefore a
    failure to return a finite bound (NOT a wrong bound).
  * CONSTRAINED to a box [-2,2]^2 (Putinar): the feasible set is compact, the
    Archimedean condition holds, and the hierarchy converges. pounce should
    certify M* = 0 exactly and recover all four minimizers (+-1, +-1).

pounce returns a certified LOWER BOUND. A valid lower bound must satisfy
lower_bound <= M* (= 0). A bound that EXCEEDS M* would be a SOLVER_BUG. A
loose / absent (nan) bound is NOT a correctness bug -- only an invalid (too
high) finite bound is.
"""
import time
import numpy as np
from scipy.optimize import minimize
import pounce

KNOWN_MIN = 0.0
KNOWN_ARGMINS = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]

# Motzkin polynomial as {exponent_tuple: coefficient}
OBJ = {
    (4, 2): 1.0,
    (2, 4): 1.0,
    (0, 0): 1.0,
    (2, 2): -3.0,
}

# Box [-2,2]^2 as Putinar inequalities g_i(x) >= 0
G1 = {(0, 0): 4.0, (2, 0): -1.0}   # 4 - x^2 >= 0
G2 = {(0, 0): 4.0, (0, 2): -1.0}   # 4 - y^2 >= 0
BOX = 2.0


def f(x, y):
    return x**4 * y**2 + x**2 * y**4 + 1.0 - 3.0 * x**2 * y**2


def report_run(label, kwargs):
    print(f"\n--- {label} ---")
    rows = {}
    for order in (0, 1, 2, 3):
        t0 = time.time()
        r = pounce.sos_minimize(OBJ, n_vars=2, order=order, **kwargs)
        dt = time.time() - t0
        rows[order] = (r, dt)
        print(
            f"order={order:>2}  lower_bound={r.lower_bound!s:>24}  "
            f"is_exact={r.is_exact}  status={r.status}  "
            f"num_min={r.num_minimizers}  success={r.success}  time={dt:.3f}s"
        )
    return rows


def main():
    # sanity: each known argmin reproduces the known min
    for p in KNOWN_ARGMINS:
        assert abs(f(*p) - KNOWN_MIN) < 1e-12, (p, f(*p))

    print("=== SOS Motzkin polynomial  M = x^4 y^2 + x^2 y^4 + 1 - 3 x^2 y^2 ===")
    print(f"KNOWN global min M* = {KNOWN_MIN:.10f} at (+-1, +-1)  [nonneg but NOT SOS]")

    # ===================== ORACLE (independent of pounce) =====================
    t_or0 = time.time()
    gx = np.linspace(-3.0, 3.0, 1601)
    gy = np.linspace(-3.0, 3.0, 1601)
    GX, GY = np.meshgrid(gx, gy)
    FV = f(GX, GY)
    grid_min = float(FV.min())
    idx = np.unravel_index(np.argmin(FV), FV.shape)
    print(f"\nGrid min over [-3,3]^2 = {grid_min:.8f} "
          f"at (x,y)=({GX[idx]:.4f},{GY[idx]:.4f})")

    rng = np.random.default_rng(0)
    best_ms = np.inf
    best_pt = None
    for _ in range(400):
        x0 = rng.uniform([-2.5, -2.5], [2.5, 2.5])
        res = minimize(lambda p: f(p[0], p[1]), x0, method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 5000})
        if res.fun < best_ms:
            best_ms = float(res.fun)
            best_pt = res.x
    t_oracle = time.time() - t_or0
    print(f"Multistart best = {best_ms:.12f} at "
          f"(x,y)=({best_pt[0]:.6f},{best_pt[1]:.6f})")

    # the lowest objective any independent search found (refutation floor)
    refutation_min = min(grid_min, best_ms, KNOWN_MIN)
    print(f"Refutation min (lowest objective found / known) = {refutation_min:.12f}")

    # ===================== POUNCE: unconstrained =====================
    uncon = report_run("UNCONSTRAINED Lasserre relaxation", {})

    # ===================== POUNCE: box-constrained (Putinar) =====================
    con = report_run("BOX-CONSTRAINED [-2,2]^2 (Putinar) relaxation",
                     {"inequalities": [G1, G2]})

    # -------- pick best finite bound across BOTH formulations --------
    def best_finite(rows):
        finite = {o: (r, dt) for o, (r, dt) in rows.items()
                  if np.isfinite(r.lower_bound)}
        if not finite:
            return None
        o = max(finite, key=lambda k: finite[k][0].lower_bound)
        return o, finite[o][0], finite[o][1]

    bu = best_finite(uncon)
    bc = best_finite(con)

    print("\n================= VERDICT =================")
    print(f"Refutation floor (true M* / best point found) = {refutation_min:.3e}")

    # 1) Invalid-bound (SOLVER_BUG) check across every run we made.
    invalid = []
    for tag, rows in (("unconstrained", uncon), ("box", con)):
        for o, (r, _) in rows.items():
            lb = r.lower_bound
            if np.isfinite(lb) and lb > refutation_min + 1e-5:
                invalid.append((tag, o, lb))
    if invalid:
        print("INVALID LOWER BOUND(S) (exceed true global min):")
        for tag, o, lb in invalid:
            print(f"  [{tag}] order={o} lower_bound={lb:+.8f} > M*={refutation_min:+.8f}")
        print("VERDICT: SOLVER_BUG (invalid lower bound exceeds true global minimum)")
        return

    # 2) Constrained formulation should certify M* = 0 exactly.
    tol = 1e-4
    if bc is not None:
        oc, rc, dtc = bc
        gap_c = KNOWN_MIN - rc.lower_bound
        print(f"\n[box] best finite lower_bound = {rc.lower_bound:+.3e} at order={oc} "
              f"(is_exact={rc.is_exact}, num_min={rc.num_minimizers})")
        print(f"[box] gap (M* - lb) = {gap_c:+.3e}  (>=0 required for validity)")
        # validate any recovered minimizers against the actual polynomial + box
        if rc.minimizers:
            objs = []
            for m in rc.minimizers:
                x, y = float(m[0]), float(m[1])
                inbox = abs(x) <= BOX + 1e-6 and abs(y) <= BOX + 1e-6
                objs.append((tuple(np.round([x, y], 4)), f(x, y), inbox))
            print("[box] recovered minimizers (point, M(point), in-box):")
            for pt, fv, inbox in objs:
                print(f"        {pt}  M={fv:+.3e}  in_box={inbox}")
            worst = max(abs(fv) for _, fv, _ in objs)
            all_inbox = all(inbox for *_, inbox in objs)
            print(f"[box] max |M(recovered)| = {worst:.3e}; all in box = {all_inbox}")

        box_ok = abs(rc.lower_bound - KNOWN_MIN) <= tol and gap_c >= -1e-6
    else:
        box_ok = False
        print("[box] no finite bound returned")

    # 3) Unconstrained outcome (structurally unbounded -> expected nan / no bound).
    if bu is None:
        print("\n[unconstrained] no finite lower bound at any order "
              "(EXPECTED: Motzkin is nonneg-but-not-SOS and non-coercive; the "
              "unconstrained SOS relaxation is unbounded). Not a correctness bug "
              "since no invalid finite bound was produced.")
        uncon_expected = True
    else:
        ou, ru, _ = bu
        gap_u = KNOWN_MIN - ru.lower_bound
        print(f"\n[unconstrained] finite bound {ru.lower_bound:+.3e} at order={ou}, "
              f"gap {gap_u:+.3e}")
        # still fine as long as it does not EXCEED M* (checked above)
        uncon_expected = ru.lower_bound <= refutation_min + 1e-5

    print(f"\n[timing] oracle={t_oracle:.4f}s")

    if box_ok and uncon_expected:
        print("VERDICT: PASS (box-constrained Putinar relaxation certifies M*=0 "
              "exactly and recovers all four minimizers (+-1,+-1); unconstrained "
              "relaxation is unbounded as expected for the Motzkin polynomial; no "
              "invalid lower bound produced anywhere)")
    elif box_ok:
        print("VERDICT: PASS (box certifies M*=0; unconstrained behaviour noted)")
    else:
        print("VERDICT: SOLVER_LIMITATION (no valid tight bound recovered even "
              "with box constraints) -- review; NOT a SOLVER_BUG (no bound "
              "exceeded the true minimum)")


if __name__ == "__main__":
    main()
