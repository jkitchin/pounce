"""Diagnostics for the monotonicity drops seen in
2026-07-22_sos_duality_invariants.py.

Question: is the non-monotone bound sequence (a) a defect in the raw SDP
relaxation value (a mathematical impossibility -> SOLVER_BUG), or (b) an
artifact of the *certification* correction, which subtracts a measured miss
that grows with the moment-matrix size (-> TOLERANCE)?

Two probes:
  P1  drop the box constraints -> certified=False -> the RAW SDP bound.
      All three objectives are coercive so the unconstrained relaxation is
      well posed.  If the raw sequence is monotone, the drop lives in the
      certification correction.
  P2  tighten tol on the certified (boxed) runs.  A correction-driven drop
      should shrink with tol; a genuine relaxation defect should not.
"""

import numpy as np

from pounce.sos import sos_minimize

def box(n, lo, hi):
    g = []
    for i in range(n):
        e = [0] * n
        ei = tuple(e[:i] + [1] + e[i + 1:])
        z = tuple(e)
        g.append({ei: 1.0, z: -float(lo)})
        g.append({ei: -1.0, z: float(hi)})
    return g


BOOTH = {(2, 0): 5.0, (0, 2): 5.0, (1, 1): 8.0, (1, 0): -34.0, (0, 1): -38.0,
         (0, 0): 74.0}
ROSEN = {(4, 0): 100.0, (2, 1): -200.0, (0, 2): 100.0, (2, 0): 1.0,
         (1, 0): -2.0, (0, 0): 1.0}
DIXON = {(0, 4): 8.0, (1, 2): -8.0, (2, 0): 3.0, (1, 0): -2.0, (0, 0): 1.0}

PROBS = [("booth", BOOTH, 2, [1, 2, 3]),
         ("rosenbrock2d", ROSEN, 2, [2, 3, 4]),
         ("dixon_price2", DIXON, 2, [2, 3, 4])]

print("=" * 74)
print("P1: RAW (uncertified, no box) bound vs order")
print("=" * 74)
for name, poly, n, orders in PROBS:
    print(f"\n{name}:")
    prev = None
    for k in orders:
        r = sos_minimize(poly, n_vars=n, order=k)
        lb = r.lower_bound
        d = "" if prev is None else f"  delta={lb - prev:+.3e}"
        bad = " <-- DROP" if (prev is not None and lb < prev - 1e-6) else ""
        print(f"  order={k} status={r.status:<10s} cert={r.certified} "
              f"lb={lb: .12e}{d}{bad}")
        if np.isfinite(lb):
            prev = lb

print()
print("=" * 74)
print("P2: CERTIFIED (boxed) bound vs order, at several tol")
print("=" * 74)
BOXES = {"booth": (-10.0, 10.0), "rosenbrock2d": (-2.0, 2.0), "dixon_price2": (-2.0, 2.0)}
for name, poly, n, orders in PROBS:
    lo, hi = BOXES[name]
    g = box(n, lo, hi)
    print(f"\n{name} (box [{lo},{hi}]):")
    for tol in [1e-8, 1e-10, 1e-12]:
        seq = []
        for k in orders:
            r = sos_minimize(poly, inequalities=g, n_vars=n, order=k,
                             tol=tol, max_iter=2000)
            seq.append((k, r.lower_bound, r.certified, r.status))
        s = "  ".join(f"o{k}={lb: .4e}" for k, lb, _, _ in seq)
        drops = [seq[i - 1][1] - seq[i][1] for i in range(1, len(seq))
                 if np.isfinite(seq[i][1]) and np.isfinite(seq[i - 1][1])
                 and seq[i][1] < seq[i - 1][1]]
        maxdrop = max(drops) if drops else 0.0
        print(f"  tol={tol:.0e}  {s}   max_drop={maxdrop:.3e}  "
              f"cert={[c for _, _, c, _ in seq]}")
