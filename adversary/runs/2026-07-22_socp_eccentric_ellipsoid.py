"""Adversary cross-check: linear program over an EXTREMELY ECCENTRIC ellipsoid.
Family: socp   Class: ill-conditioned / near-zero cone curvature at the optimum

Source: Boyd & Vandenberghe, *Convex Optimization* (CUP 2004), Sec. 4.4.2 /
Ex. 4.21 -- minimizing a linear function over an ellipsoid, the canonical SOCP
    minimize    c'x
    subject to  || D x ||_2 <= 1                (ellipsoid E = {x : ||Dx|| <= 1})
This is B&V's "SOCP with a single second-order cone constraint"; the support
function of an ellipsoid has the closed form  inf_{x in E} c'x = -||D^{-T} c||_2.

ADVERSARIAL TWIST (dimension: ill-conditioning & bad scaling):
D = diag(d) Q with Q orthogonal and d = logspace(k/2, -k/2, 6), i.e. the
ellipsoid semi-axis lengths are 1/d spanning 10^k -- an axis ratio (and
cond(D)) of 10^k, swept over k = 4, 6, 8, 10.
c = Q' 1 is O(1)-scaled, so D^{-T} c = 1/d spans 10^k. Consequently the OPTIMAL
VALUE is -||1/d|| ~ 10^{k/2} and ||x*|| ~ 10^{k/2}, while the data c is O(1) and
the cone bound is 1: the objective row, the primal iterate and the cone slacks
live on wildly different scales simultaneously. The contact point sits far out
on the LONG axis, where the cone boundary has normal curvature ~10^-k of the
ambient scale -- the regime where an unscaled IPM loses digits, stalls short of
the boundary, or (as ECOS and SCS do here at k=10) mistakes the nearly-flat
boundary for a recession direction and declares the problem unbounded.

KNOWN OPTIMAL -- exact closed form (no solver involved):
    Lagrangian/Cauchy-Schwarz:  c'x = (D^{-T}c)'(Dx) >= -||D^{-T}c|| ||Dx||
                                    >= -||D^{-T}c||,
    with equality iff Dx = -D^{-T}c / ||D^{-T}c||.  Hence
        OPT = -|| D^{-T} c ||_2,      x* = -D^{-1} D^{-T} c / || D^{-T} c ||_2.
    The SOC is TIGHT at x* (||Dx*|| = 1 exactly).
    With D = diag(d) Q:  D^{-1} = Q' diag(1/d),  D^{-T} = diag(1/d) Q.

Cone encoding (re-derived; pounce convention s = h - G x must lie in K):
    ("soc", 7): s = (s0, s1..s6) with s0 >= ||s1..s6||_2.
      s0   = 1     -> h[0] = 1,  G[0,:] = 0
      s1:6 = D x   -> h[1:] = 0, G[1:,:] = -D
    => 1 >= ||Dx||.  No equality constraints (A = b = None).
"""
import time
import numpy as np

np.set_printoptions(precision=6, suppress=False)

# ---------------- data: eccentric ellipsoid, cond(D) = 10^k ----------------
n = 6
rng = np.random.default_rng(20260722)
# fixed orthogonal Q via QR of a reproducible random matrix (sign-normalized)
Qraw, Rraw = np.linalg.qr(rng.standard_normal((n, n)))
Q = Qraw * np.sign(np.diag(Rraw))
assert np.allclose(Q @ Q.T, np.eye(n), atol=1e-13)

import pounce  # noqa: E402
import cvxpy as cp  # noqa: E402


def rel(u, v):
    return abs(u - v) / max(1.0, abs(v))


def relvec(u, v):
    # scale-aware componentwise error (x spans many decades)
    if u is None or np.asarray(u, dtype=float).shape != np.shape(v):
        return float("inf")
    u = np.asarray(u, dtype=float)
    if not np.all(np.isfinite(u)):
        return float("inf")
    return float(np.max(np.abs(u - v) / np.maximum(1e-300, np.abs(v))))


def build(k):
    """Return (D, c, x_star, KNOWN_OPTIMAL) for axis ratio 10^k.

    c = Q' 1 is chosen so the reference is computable in a NUMERICALLY STABLE
    way despite cond(D) = 10^k:  D^{-T} c = diag(1/d) Q c = diag(1/d) 1 = 1/d,
    where Q c = 1 holds to ~eps ABSOLUTE and every entry of 1 is O(1), so each
    w_i = 1/d_i carries only ~eps RELATIVE error.  (The naive route -- pick w,
    set c = D'w, recover w = D^{-T}c -- loses eps*cond(D) = 1e-8 digits at
    k=8 and is not a trustworthy reference.  Verified: that route's round-trip
    error is ~1e-8 at k=8, which is why it is not used here.)
    """
    d = np.logspace(k / 2.0, -k / 2.0, n)
    D = np.diag(d) @ Q
    c = Q.T @ np.ones(n)                 # O(1)-scaled objective row
    Dinv = Q.T @ np.diag(1.0 / d)        # D^{-1}
    w = (1.0 / d) * (Q @ c)              # = D^{-T} c, stable; ~= 1/d
    assert relvec(w, 1.0 / d) < 1e-12    # stability of the reference itself
    nw = float(np.linalg.norm(w))
    known = -nw                          # OPT = -||D^{-T} c||
    x_star = -Dinv @ (w / nw)
    # self-checks on the closed form, before any solver runs. The residual of an
    # explicitly formed D @ D^{-1} product is inherently ~eps*cond(D), so the
    # tolerance is conditioning-aware; the reference VALUES above are not.
    tol = max(1e-12, 1e-15 * 10.0 ** k)
    assert abs(np.linalg.norm(D @ x_star) - 1.0) < tol         # SOC TIGHT
    assert abs(float(c @ x_star) - known) / nw < tol
    return D, c, x_star, known


def run_pounce(D, c):
    G = np.zeros((n + 1, n))
    G[1:, :] = -D
    h = np.zeros(n + 1)
    h[0] = 1.0
    t0 = time.perf_counter()
    res = pounce.solve_socp(c=c, G=G, h=h, cones=[("soc", n + 1)])
    t = time.perf_counter() - t0
    x = np.asarray(res.x, dtype=float)
    return res.status, float(c @ x), t, x


def run_cvxpy(D, c, solver, **kw):
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(c @ x), [cp.norm(D @ x, 2) <= 1])
    t0 = time.perf_counter()
    try:
        prob.solve(solver=solver, **kw)
    except Exception as e:                                   # pragma: no cover
        return f"error:{type(e).__name__}", float("nan"), 0.0, np.full(n, np.nan)
    return prob.status, float(prob.value), time.perf_counter() - t0, np.asarray(x.value)


all_ok = True
rows = []
for k in (4, 6, 8, 10):
    D, c, x_star, known = build(k)
    condD = np.linalg.cond(D)
    st_p, obj_p, t_p, x_p = run_pounce(D, c)
    norm_p = float(np.linalg.norm(D @ x_p))
    st_cl, obj_cl, t_cl, x_cl = run_cvxpy(D, c, cp.CLARABEL)
    st_ec, obj_ec, t_ec, x_ec = run_cvxpy(D, c, cp.ECOS)
    st_sc, obj_sc, t_sc, x_sc = run_cvxpy(D, c, cp.SCS, eps=1e-10, max_iters=100000)

    print(f"\n================ axis ratio 10^{k}  (cond(D) = {condD:.3e}) ================")
    print(f"||x*|| = {np.linalg.norm(x_star):.6e}   |c|_inf = {np.max(np.abs(c)):.3e}"
          f"   semi-axis ratio = 1e{k}")
    print(f"known_optimal = {known:.14e}")
    print(f"pounce           status={st_p:<20s} obj={obj_p:.14e} t={t_p:.4f}s  ||Dx||={norm_p:.12f}")
    print(f"cvxpy/CLARABEL   status={st_cl:<20s} obj={obj_cl:.14e} t={t_cl:.4f}s")
    print(f"cvxpy/ECOS       status={st_ec:<20s} obj={obj_ec:.14e} t={t_ec:.4f}s")
    print(f"cvxpy/SCS        status={st_sc:<20s} obj={obj_sc:.14e} t={t_sc:.4f}s")
    print(f"oracle agreement CLARABEL vs ECOS = {rel(obj_cl, obj_ec):.2e}  "
          f"CLARABEL vs SCS = {rel(obj_cl, obj_sc):.2e}")
    print(f"rel_err vs known:  pounce={rel(obj_p, known):.2e}  "
          f"CLARABEL={rel(obj_cl, known):.2e}  ECOS={rel(obj_ec, known):.2e}  "
          f"SCS={rel(obj_sc, known):.2e}")
    print(f"x rel err vs closed form:  pounce={relvec(x_p, x_star):.2e}  "
          f"CLARABEL={relvec(x_cl, x_star):.2e}  ECOS={relvec(x_ec, x_star):.2e}")

    # Two INDEPENDENT references must agree before either is trusted. The
    # stably-computed closed form counts as one such reference (it involves no
    # solver at all), so validity = any two of {closed form, CLARABEL, ECOS,
    # SCS} agreeing. At k=10 both ECOS and SCS break down and the surviving
    # pair is {closed form, CLARABEL} -- which still pins the answer.
    refs = {"closed_form": known, "CLARABEL": obj_cl, "ECOS": obj_ec, "SCS": obj_sc}
    names = [a for a in refs if np.isfinite(refs[a])]
    agreeing = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                if rel(refs[a], refs[b]) < 1e-6]
    oracles_agree = len(agreeing) > 0
    print(f"agreeing reference pairs: {agreeing}")
    ok = (st_p == "optimal") and rel(obj_p, known) < 1e-4 and norm_p <= 1.0 + 1e-6
    rows.append((k, condD, st_p, obj_p, rel(obj_p, known), relvec(x_p, x_star),
                 rel(obj_cl, known), rel(obj_ec, known), oracles_agree, t_p, t_cl, t_ec, ok,
                 st_ec, rel(obj_sc, known)))
    all_ok = all_ok and ok and oracles_agree

print("\n=== summary (k, cond, pounce status, obj_err, x_err, clarabel_err, ecos_err) ===")
for r in rows:
    print(f"k={r[0]:2d} cond={r[1]:.1e} {r[2]:<9s} obj_err={r[4]:.2e} x_err={r[5]:.2e} "
          f"CLA={r[6]:.2e} ECOS={r[7]:.2e}({r[13]}) SCS={r[14]:.2e} agree={r[8]} "
          f"t_p={r[9]:.4f} t_cla={r[10]:.4f}")

print("VERDICT: PASS" if all_ok else "VERDICT: FAIL (see per-k rows above)")
