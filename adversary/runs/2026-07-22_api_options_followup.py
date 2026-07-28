"""Focused follow-up on the two signals from 2026-07-22_api_options_contract.py:

 F1. `max_iter` integer truncation on the NLP path: values > 2^31 wrap.
     Does a wrap that lands on a VALID positive int get silently accepted?
 F2. `solve_qp`/`solve_socp` do not validate `tol` at all, while
     `pounce.minimize` rejects tol<=0 with OPTION_INVALID. Consequences?
"""
import warnings

import numpy as np
import pounce


def hs071_min(**kw):
    def fun(x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def jac(x):
        return np.array([x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]),
                         x[0] * x[3], x[0] * x[3] + 1.0,
                         x[0] * (x[0] + x[1] + x[2])])

    cons = [{"type": "ineq",
             "fun": lambda x: x[0] * x[1] * x[2] * x[3] - 25.0,
             "jac": lambda x: np.array([x[1] * x[2] * x[3], x[0] * x[2] * x[3],
                                        x[0] * x[1] * x[3], x[0] * x[1] * x[2]])},
            {"type": "eq", "fun": lambda x: np.sum(x ** 2) - 40.0,
             "jac": lambda x: 2.0 * x}]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return pounce.minimize(fun, np.array([1.0, 5.0, 5.0, 1.0]), jac=jac,
                               bounds=[(1.0, 5.0)] * 4, constraints=cons, **kw)


print("=" * 74)
print("F1. max_iter integer truncation on the NLP path (pounce.minimize)")
print("=" * 74)
print("  reference: max_iter=3 ->", end=" ")
r = hs071_min(max_iter=3)
print(f"nit={r.nit} status={r.status} f={r.fun:.8f} success={r.success}")

# 2**32 + 3 truncates to 3 in int32; 2**32 truncates to 0.
for v, wrap in [(2 ** 32 + 3, 3), (2 ** 32 + 1, 1), (2 ** 33 + 7, 7), (2 ** 32, 0)]:
    try:
        r = hs071_min(max_iter=v)
        print(f"  max_iter={v} (int32 wrap -> {wrap}): NO ERROR "
              f"nit={r.nit} status={r.status} f={r.fun:.8f} success={r.success}"
              f"   <== wrapped? {'YES' if r.nit == wrap else 'no'}")
    except Exception as e:
        print(f"  max_iter={v} (wrap -> {wrap}): "
              f"{type(e).__name__}: {str(e).splitlines()[0][:110]}")

print()
print("  CLI parity check (same wrap through the KEY=VALUE surface):")
import subprocess, os
CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
NL = ("/private/tmp/claude-501/-Users-jkitchin-projects-pounce/"
      "671a5f76-82be-4f1a-bac6-59f0cb187d8b/scratchpad/hs071.nl")
for arg in ["max_iter=3", f"max_iter={2**32 + 3}", f"max_iter={10**12}"]:
    p = subprocess.run([CLI, NL, "print_level=0", arg], capture_output=True,
                       text=True, timeout=20)
    txt = (p.stdout + p.stderr)
    it = [l for l in txt.splitlines() if "teration" in l or "not a valid" in l
          or "Option" in l]
    print(f"    CLI {arg:<22} rc={p.returncode}  {' / '.join(it[-2:])[:120]!r}")

print()
print("=" * 74)
print("F2. tol validation gap: minimize rejects tol<=0, solve_qp accepts it")
print("=" * 74)
# min 1/2 x'x - [1,1]'x  =>  x* = (1,1), obj* = -1
P = np.eye(2)
c = np.array([-1.0, -1.0])
TRUE_X, TRUE_OBJ = np.ones(2), -1.0
for tol in [None, 1e-9, 0.0, -1.0, -1e300, 1e300, float("inf"), float("nan")]:
    try:
        r = pounce.solve_qp(P=P, c=c, tol=tol)
        err = float(np.linalg.norm(np.asarray(r.x) - TRUE_X, np.inf))
        flag = ""
        if r.status == "optimal" and err > 1e-4:
            flag = "  <== 'optimal' but WRONG (x_err %.2e)" % err
        print(f"  solve_qp(tol={tol!r:>8}) status={r.status:<16} iters={r.iters:<4} "
              f"obj={r.obj:+.6g} x={np.round(r.x, 6)}{flag}")
    except Exception as e:
        print(f"  solve_qp(tol={tol!r:>8}) {type(e).__name__}: {str(e)[:90]}")

print()
print("  Same values through pounce.minimize (NLP path), for contrast:")
for tol in [0.0, -1.0, 1e300, float("nan")]:
    try:
        r = hs071_min(tol=tol)
        print(f"    minimize(tol={tol!r:>8}) NO ERROR nit={r.nit} f={r.fun:.8f}")
    except Exception as e:
        print(f"    minimize(tol={tol!r:>8}) {type(e).__name__}: "
              f"{str(e).splitlines()[0][:90]}")

print()
print("  And through the routed convex path minimize(solver_selection='qp-ipm'):")


def qf(x):
    return float(0.5 * x @ x - c @ x * -1)


def qf2(x):
    return float(0.5 * x @ x + c @ x)


def qg2(x):
    return x + c


for tol in [0.0, -1.0, 1e300]:
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = pounce.minimize(qf2, np.zeros(2), jac=qg2,
                                solver_selection="qp-ipm", tol=tol)
        err = float(np.linalg.norm(np.asarray(r.x) - TRUE_X, np.inf))
        print(f"    minimize(qp-ipm, tol={tol!r:>7}) success={r.success} "
              f"nit={r.nit} f={r.fun:+.6g} x_err={err:.2e}"
              + ("   <== success but WRONG" if r.success and err > 1e-4 else ""))
    except Exception as e:
        print(f"    minimize(qp-ipm, tol={tol!r:>7}) {type(e).__name__}: "
              f"{str(e).splitlines()[0][:90]}")

print()
print("  CLI convex route, tol=-1 / tol=0:")
for arg in ["tol=0", "tol=-1"]:
    p = subprocess.run([CLI, NL, "print_level=0", arg], capture_output=True,
                       text=True, timeout=20)
    line = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()]
    print(f"    CLI {arg:<8} rc={p.returncode} {' | '.join(line[-2:])[:130]!r}")
