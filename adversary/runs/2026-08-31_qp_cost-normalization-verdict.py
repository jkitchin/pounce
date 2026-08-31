"""Adversary probe: does the convex arm's cost normalization buy a false verdict?
Family: qp   Class: huge-scale / cost-normalized (sigma) path
Targets: PR #835 (gh#414 "Stop the cost normalization from buying an Optimal verdict")
         PR #781 (primal (x,x) regularization escalation on wrong inertia)

CLAUDE.md records that the sigma path engages only when
    max(||P||_inf, ||c||_inf) * eps > tol
which 1 of 79 fixtures and 0 of 138 Maros-Meszaros problems satisfy -- so the
corpora are nearly blind here and this is the arm's least-covered branch.

Oracle is CLOSED FORM, not another solver: with P = s*M (M PSD) and
c = -s*M*x_star the unconstrained minimizer is exactly x_star at every scale s,
objective -0.5*s*x_star' M x_star. A second-opinion cvxpy/Clarabel run is
reported alongside but is not the pass criterion, since the oracle itself
degrades at 1e14.

The contract under test is VERDICT HONESTY, which needs no oracle at all:
a returned status of "optimal" asserts the KKT system was solved. Reporting
`optimal` on a point far from the closed-form optimum is the gh#414 defect;
reporting a non-optimal status on a hard-scaled model is merely a limitation.
"""
import numpy as np, warnings
warnings.simplefilter("ignore")
import pounce
try:
    import cvxpy as cp
    HAVE_CVXPY = True
except Exception:
    HAVE_CVXPY = False

rng = np.random.default_rng(414)
n = 6
Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
M = Q @ np.diag(np.linspace(1.0, 4.0, n)) @ Q.T
M = (M + M.T) / 2
x_star = rng.normal(size=n)

SCALES = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14]
TOLX = 1e-4

print(f"{'scale':>8} {'sigma?':>7} {'status':>12} {'||x-x*||inf':>12} {'rel obj err':>12} "
      f"{'cvxpy err':>11}  verdict")
bad = []
for s in SCALES:
    P = s * M
    c = -s * (M @ x_star)
    engages = max(np.max(np.abs(P)), np.max(np.abs(c))) * np.finfo(float).eps > 1e-8
    obj_star = 0.5 * x_star @ P @ x_star + c @ x_star
    try:
        r = pounce.solve_qp(P=P, c=c)
        st, x = r.status, np.asarray(r.x)
        xe = float(np.max(np.abs(x - x_star)))
        oe = abs(r.obj - obj_star) / max(1.0, abs(obj_star))
    except Exception as e:
        st, xe, oe = f"raised {type(e).__name__}", np.nan, np.nan
    ce = np.nan
    if HAVE_CVXPY:
        try:
            v = cp.Variable(n)
            pr = cp.Problem(cp.Minimize(0.5*cp.quad_form(v, cp.psd_wrap(P)) + c @ v))
            pr.solve(solver=cp.CLARABEL)
            ce = float(np.max(np.abs(v.value - x_star)))
        except Exception:
            pass
    claims_optimal = (st == "optimal")
    wrong = claims_optimal and (not np.isfinite(xe) or xe > TOLX)
    verdict = "FALSE OPTIMAL" if wrong else ("ok" if claims_optimal else f"declined ({st})")
    if wrong:
        bad.append((s, st, xe, oe))
    print(f"{s:>8.0e} {str(engages):>7} {str(st):>12} {xe:>12.3e} {oe:>12.3e} "
          f"{ce:>11.3e}  {verdict}")

print("\n-- same sweep with an equality constraint and bounds (regularization path, PR #781) --")
A = rng.normal(size=(2, n)); b = A @ x_star
for s in (1e0, 1e6, 1e10, 1e14):
    P = s * M; c = -s * (M @ x_star)
    try:
        r = pounce.solve_qp(P=P, c=c, A=A, b=b)
        x = np.asarray(r.x)
        xe = float(np.max(np.abs(x - x_star)))       # x_star is feasible & optimal here
        res = float(np.max(np.abs(A @ x - b)))
        wrong = r.status == "optimal" and xe > TOLX
        if wrong: bad.append((s, r.status, xe, np.nan))
        print(f"{s:>8.0e} status={r.status:<10} ||x-x*||inf={xe:.3e} ||Ax-b||inf={res:.3e}"
              f"  {'FALSE OPTIMAL' if wrong else ''}")
    except Exception as e:
        print(f"{s:>8.0e} raised {type(e).__name__}: {str(e)[:70]}")

print(f"\nfalse-optimal verdicts: {len(bad)}")
for b_ in bad: print("   scale=%.0e status=%s ||x-x*||=%.3e" % (b_[0], b_[1], b_[2]))
print("VERDICT: PASS" if not bad else f"VERDICT: FAIL ({len(bad)} false-optimal verdicts)")
