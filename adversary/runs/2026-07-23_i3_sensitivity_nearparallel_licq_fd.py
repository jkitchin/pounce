"""i3 Test 8 — QpSensitivity dx/db on NEAR-LICQ from nearly-parallel active
EQUALITY constraints, cross-checked against a central finite-difference
re-solve (#284/#305).

#284/#305 fixed QpSensitivity silently OVER-DAMPING dx/db on near-LICQ problems.
This probes near-LICQ produced NOT by an ill-conditioned Hessian (that was the
prior Hilbert test) but by two nearly-parallel active equality-constraint
gradients — the direct #284 geometry.

QP:  min 0.5||x||^2  s.t.  A x = b,   A = [[1, 0], [1, eps]],  b = [1, 1].
Rows (1,0) and (1,eps) are nearly parallel for small eps -> near-LICQ.
The solution is x = A^{-1} b, so the EXACT sensitivity is
    dx/db = A^{-1} = (1/eps) [[eps, 0], [-1, 1]]  =  [[1, 0], [-1/eps, 1/eps]].
parametric_step([i],[delta]) must equal column i of A^{-1} times delta.

Oracle: (1) closed-form A^{-1}; (2) central FD re-solve of the QP at b +/- delta
with an independent solve_qp call. Over-damping = parametric_step magnitude
significantly BELOW the FD / closed-form magnitude.
Tested for eps in {1e-3, 1e-5, 1e-7} (worsening LICQ).
"""
from __future__ import annotations
import numpy as np
import pounce


def solve(eps, b):
    A = np.array([[1.0, 0.0], [1.0, eps]])
    P = np.eye(2)
    c = np.zeros(2)
    r = pounce.solve_qp(P=P, c=c, A=A, b=b)
    return np.asarray(r.x)


def run_eps(eps):
    A = np.array([[1.0, 0.0], [1.0, eps]])
    b = np.array([1.0, 1.0])
    Ainv = np.linalg.inv(A)                    # exact dx/db
    sens = pounce.QpSensitivity(P=np.eye(2), c=np.zeros(2), A=A, b=b)
    x0 = np.asarray(sens.x)

    # central FD re-solve for each parameter (column of dx/db)
    fd = np.zeros((2, 2))
    delta = 1e-6
    for j in range(2):
        db = np.zeros(2); db[j] = delta
        xp = solve(eps, b + db)
        xm = solve(eps, b - db)
        fd[:, j] = (xp - xm) / (2 * delta)

    # QpSensitivity predicted step for a unit-delta on each parameter
    step = np.zeros((2, 2))
    for j in range(2):
        d = np.zeros(2); d[j] = 1.0
        step[:, j] = np.asarray(sens.parametric_step([j], [1.0]))

    err_exact = np.max(np.abs(step - Ainv))
    err_fd = np.max(np.abs(step - fd))
    # relative (the sensitive entry is ~1/eps)
    scale = np.max(np.abs(Ainv))
    rel = err_exact / scale
    print(f"[eps={eps:.0e}] |A^-1|max={scale:.3e}  x={x0}")
    print(f"   dx/db exact  =\n{Ainv}")
    print(f"   dx/db pounce =\n{step}")
    print(f"   dx/db FD     =\n{fd}")
    print(f"   err_vs_exact={err_exact:.3e} (rel {rel:.3e})  err_vs_FD={err_fd:.3e}  "
          f"ill_conditioned={sens.ill_conditioned}")
    ok = rel < 1e-3 and (err_fd / scale) < 1e-3
    return ok, rel


def main():
    allok = True
    worst = 0.0
    for eps in (1e-3, 1e-5, 1e-7):
        ok, rel = run_eps(eps)
        allok = allok and ok
        worst = max(worst, rel)
        print()
    if allok:
        print(f"VERDICT: PASS (dx/db matches closed-form A^-1 and central FD to "
              f"rel<1e-3 across near-LICQ; worst rel={worst:.2e})")
    else:
        print(f"VERDICT: FAIL (QpSensitivity dx/db over-damped/wrong vs closed-form "
              f"A^-1 & central FD on near-parallel active equalities; worst rel="
              f"{worst:.2e} — #284 residual)")


if __name__ == "__main__":
    main()
