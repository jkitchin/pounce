"""Adversary cross-check: curve_fit one-sided (None) bounds after PR #263.

Family: nlp (python surface)   Class: API / bounds normalisation
Target: PR #263 "Read curve_fit's 2-tuple bounds as scipy (lower, upper), not pairs"
Oracle: scipy.optimize.curve_fit + the equivalent *list* pair-list form of the
        same box (which PR #263 documents as the surviving pounce API), plus a
        noiseless problem whose exact answer is known analytically.

Model:  f(x; A, k) = A * exp(-k x), data generated noiselessly at A=2, k=1.
Known optimum: popt = (2.0, 1.0) exactly, for any box containing it.

The box under test is A in [-inf, 10], k in [0, +inf] -- expressed as a
per-parameter pair list with ``None`` for the open sides:

    bounds = ((None, 10.0), (0.0, None))

It contains (2, 1), so the constrained optimum IS the unconstrained one.

PR #263 removed the ``_is_pair_list`` guard so that *any* length-2 tuple is read
as scipy's ``(lower, upper)``. For n == 2 that reinterprets the pair list above
as lower=(None, 10.0), upper=(0.0, None); ``_to_array(None)`` is NaN, so the box
becomes lb=(nan, 10.0), ub=(0.0, nan). The reversed-bound guard in
``_minimize._normalize_bounds`` is ``lb > ub``, and every comparison against NaN
is False, so the NaN box passes validation silently and reaches the solver.
"""

import numpy as np
from scipy import optimize as scipy_optimize

import pounce
from pounce._curve_fit import _normalize_bound_arg
from pounce._minimize import _normalize_bounds

KNOWN_OPTIMAL = np.array([2.0, 1.0])


def model(x, a, b):
    return a * np.exp(-b * x)


x = np.linspace(0.0, 3.0, 60)
y = model(x, *KNOWN_OPTIMAL)  # noiseless
p0 = [1.0, 2.0]

TUPLE_PAIRS = ((None, 10.0), (0.0, None))  # pounce pair-list, tuple flavour
LIST_PAIRS = [(None, 10.0), (0.0, None)]  # pounce pair-list, list flavour
SCIPY_EQUIV = ([-np.inf, 0.0], [10.0, np.inf])  # same box, scipy 2-tuple form

print("=== bounds normalisation ===")
for label, b in (("tuple pair-list", TUPLE_PAIRS), ("list pair-list", LIST_PAIRS)):
    pairs = _normalize_bound_arg(b, 2)
    lb, ub = _normalize_bounds(pairs, 2)
    print(f"{label:16s} {str(b):28s} -> pairs={pairs} lb={lb} ub={ub}")
print(f"NaN-blind guard check: np.float64('nan') > 0.0 -> {np.float64('nan') > 0.0}")

print("\n=== solves ===")
r_tuple = pounce.curve_fit(model, x, y, p0=p0, bounds=TUPLE_PAIRS, jac="fd")
r_list = pounce.curve_fit(model, x, y, p0=p0, bounds=LIST_PAIRS, jac="fd")
s_popt, _ = scipy_optimize.curve_fit(model, x, y, p0=p0, bounds=SCIPY_EQUIV)

for label, popt, status in (
    ("pounce tuple pair-list", r_tuple.popt, r_tuple.status),
    ("pounce list  pair-list", r_list.popt, r_list.status),
    ("scipy  (oracle)       ", s_popt, 0),
):
    err = float(np.linalg.norm(np.asarray(popt) - KNOWN_OPTIMAL, np.inf))
    print(f"{label}  status={status}  popt={np.asarray(popt)}  inf_err_vs_known={err:.3e}")

tuple_err = float(np.linalg.norm(np.asarray(r_tuple.popt) - KNOWN_OPTIMAL, np.inf))
list_err = float(np.linalg.norm(np.asarray(r_list.popt) - KNOWN_OPTIMAL, np.inf))

print()
if list_err < 1e-6 and tuple_err > 1e-3 and r_tuple.status == 0:
    print(
        "VERDICT: FAIL (REGRESSION) -- the tuple pair-list form returns a badly "
        f"wrong fit (inf err {tuple_err:.3e}) while still reporting "
        f"status={r_tuple.status} (Solve_Succeeded). The identical box as a list "
        f"is correct (inf err {list_err:.3e})."
    )
else:
    print("VERDICT: PASS")
