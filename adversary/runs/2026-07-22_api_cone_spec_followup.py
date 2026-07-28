"""Follow-up probe for 2026-07-22_api_cone_spec_contract.py

1. Corrected exp / pow sanity checks (the originals were MY formulation errors:
   a free variable in the exp case, cone-apex degeneracy in the pow case).
2. Pin down the zero-/negative-dimension panic surface: which cone kinds panic,
   and does the row-sum validation pass first (i.e. is the panic reachable from
   a spec that survives every documented check)?
"""

import numpy as np

from pounce import solve_socp


def probe(label, fn, expected=None, tol=1e-5):
    try:
        r = fn()
    except BaseException as exc:  # noqa: BLE001
        tag = "PANIC" if "Panic" in type(exc).__name__ else "clean"
        print(f"{label:52s} -> {tag} {type(exc).__name__}: {str(exc)[:110]}")
        return None
    if expected is None:
        print(f"{label:52s} -> NO ERROR status={r.status} obj={r.obj:.10e}")
    else:
        err = abs(r.obj - expected) / max(1.0, abs(expected))
        flag = "OK" if err < tol else "<<< MISMATCH"
        print(f"{label:52s} -> status={r.status} obj={r.obj:.10e} "
              f"exp={expected:.10e} err={err:.2e} {flag}")
    return r


print("=== corrected exp sanity: min t s.t. (u,1,t) in Kexp, u = 1 -> t = e ===")
Gexp = np.zeros((3, 2))
Gexp[0, 0] = -1.0
Gexp[2, 1] = -1.0
hexp = np.array([0.0, 1.0, 0.0])
probe("('exp',3) min t, u==1", lambda: solve_socp(
    c=[0.0, 1.0], A=[[1.0, 0.0]], b=[1.0], G=Gexp, h=hexp, cones=[("exp", 3)]),
    np.e)

print("\n=== corrected pow sanity: max x s.t. |x| <= sqrt(y z), y=4, z=1 -> x=2 ===")
Gpow = -np.eye(3)
hpow = np.zeros(3)
probe("('pow',0.5) max x, y=4 z=1", lambda: solve_socp(
    c=[-1.0, 0.0, 0.0], A=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], b=[4.0, 1.0],
    G=Gpow, h=hpow, cones=[("pow", 0.5)]), -2.0)
# alpha = 1/3: |x| <= y^(1/3) z^(2/3), y=8, z=1 -> x <= 2
probe("('pow',1/3) max x, y=8 z=1", lambda: solve_socp(
    c=[-1.0, 0.0, 0.0], A=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], b=[8.0, 1.0],
    G=Gpow, h=hpow, cones=[("pow", 1.0 / 3.0)]), -2.0)

print("\n=== zero-dimension cones: does row-sum validation pass, then panic? ===")
# 5-row G from the base problem; a 0-dim block contributes 0 rows so the
# sum still equals 5 -> every documented check passes.
G5 = np.array([[0., 0., -1.], [-1., 0., 0.], [0., -1., 0.],
               [-1., 0., 0.], [0., -1., 0.]])
h5 = np.array([0., 1., -2., 0., 0.])
c5 = np.array([0., 0., 1.])
for kind in ["nonneg", "soc", "psd"]:
    probe(f"('{kind}',0) + soc3 + nonneg2 (rows sum = 5)", lambda k=kind: solve_socp(
        c=c5, G=G5, h=h5, cones=[(k, 0), ("soc", 3), ("nonneg", 2)]), 1.0)

print("\n=== negative dimensions saturate to 0 (Rust f64->usize) -> same panic ===")
# ('soc', -3) contributes SecondOrder(0) == 0 rows, so the sum is still 5.
for kind in ["nonneg", "soc", "psd"]:
    probe(f"('{kind}',-3) + soc3 + nonneg2 (rows sum = 5)", lambda k=kind: solve_socp(
        c=c5, G=G5, h=h5, cones=[(k, -3), ("soc", 3), ("nonneg", 2)]), 1.0)

print("\n=== non-integer dimensions (rounded) ===")
probe("('soc',3.4)+('nonneg',2) rounds to 3", lambda: solve_socp(
    c=c5, G=G5, h=h5, cones=[("soc", 3.4), ("nonneg", 2)]), 1.0)
probe("('soc',0.4)+... rounds to 0 -> panic?", lambda: solve_socp(
    c=c5, G=G5, h=h5, cones=[("soc", 0.4), ("soc", 3), ("nonneg", 2)]), 1.0)

print("\n=== case-insensitivity is documented/by design (parse_cones lowercases) ===")
for k in ["SOC", "Q", "NonNeg", "SDP", "BANANA"]:
    probe(f"kind='{k}'", lambda kk=k: solve_socp(
        c=c5, G=G5, h=h5, cones=[(kk, 3), ("nonneg", 2)]), 1.0)
