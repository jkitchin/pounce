"""MPCC stationarity for the CALIPSO corpus.

The same question `benchmarks/mpcc/stationarity.py` answers -- *which*
stationarity concept, if any, does a returned point satisfy -- for models
given as JAX functions instead of that harness's quadratic forms, and at
trajectory scale. A converged lowered NLP says only that the reformulation's
residuals are small; on `nonconvex/complementarity` two routes return
feasible points with objectives 25 and 85 while the optimum is 17, and the
residuals alone cannot tell those apart from it.

What is tested
--------------

With pairs ``0 <= G_i ⟂ H_i >= 0``, find multipliers with

    grad f = sum lam_k grad c_k + sum mu_k grad h_k + sum nu_i grad G_i
             + sum w_i grad H_i + zL - zU

(``c`` equalities, ``h >= 0`` inequalities with ``mu >= 0`` on the active
ones and zero on the rest, ``zL, zU >= 0`` on active bounds), where
``nu_i`` is free on a pair with only ``G`` active, zero where ``G`` is
inactive, and on a **biactive** pair:

  S (strong)  ``nu_i >= 0`` and ``w_i >= 0`` -- a KKT point of the
              tightened NLP; B-stationary under MPCC-LICQ
  W (weak)    no sign condition

Each is a bounded least-squares feasibility problem
(`scipy.optimize.lsq_linear`, BVLS). C and M need an enumeration over the
biactive pairs' sign branches (3**b); trajectory models have dozens of
biactive pairs (every rocket step outside the keep-out zones is one), so
they are reported only when ``b <= MAX_ENUM``. The verdict is the strongest
class established: ``S``, else ``C``/``M`` when enumerated, else ``W``,
else ``none``.

A model with no pairs is a plain NLP and the test is ordinary KKT; its
verdict is reported as ``KKT``. A model whose complementarity is between
second-order cones (drifting, quadruped) is reported ``n/a``: the classes
above are defined for NCP pairs, and the conic analogues are not
implemented here.

Thresholds, and why each is relative
------------------------------------

* A pair side counts as zero at ``sqrt(solver_tol) * side_scale``, the
  rule `benchmarks/mpcc/spec.py:pair_activity` derives: ``G*H`` is flat at
  the corner, so a product driven to ``tol`` pins each side only to
  ``sqrt(tol)``. ``side_scale = max(1, max_i |G_i|)`` (and likewise for H)
  makes it relative to the family's own magnitude: bunny-hop's impulses
  live near 100 while its gaps live near 1.
* Inequality rows and bounds count as active within ``sqrt(solver_tol)``
  too. An interior-point answer sits ``mu / z`` from an active bound, not
  ``mu``: on `complementarity`, POUNCE's ``direct`` answer -- the optimum
  -- has ``x[1] = 1.4e-6`` against its bound, and the fixed ``1e-6`` the
  harness uses for rows read that bound as inactive and the point as not
  stationary at all (S residual 0.56). Counting a near-active row as
  active only *allows* a sign-constrained multiplier there, so the cost
  of the looser rule is certifying a point within ``sqrt(tol)`` of
  stationary, never a far one.
* The stationarity residual is judged against
  ``max(1e-6, solver_tol) * dual_scale``, the dual scale being the largest
  entry of ``grad f`` and of every column a multiplier may use -- the same
  argument as `dual_inf_scale_kappa`; an absolute threshold would grade a
  point by its units.

The verdict is about stationarity only, never feasibility: the test asks
whether ``grad f`` is a combination of the constraint gradients, which an
infeasible point can satisfy (`wachter`'s infeasible stopping point passes
the KKT test). Read it together with the source-level violation;
`summarize.py` counts a run as certified only when it is also physical.

``solver_tol`` is what the solver was asked for: 1e-8 for the POUNCE and
Ipopt routes, 1e-4 for CALIPSO.jl's defaults. Judging CALIPSO's answers at
1e-8 would fail them for being what they were asked to be.
"""

from __future__ import annotations

import itertools
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import lsq_linear

MAX_ENUM = 8


#: Above this many free columns BVLS (exact, active-set) takes minutes on a
#: dense system -- the rocket's 856 x 714 had not finished one solve in five
#: -- so the interior-reflective `trf` method is used instead: 10-20 s per
#: solve there, at a residual accuracy well inside `resid_tol`.
BVLS_MAX_COLS = 300


def _bvls(A, b, lo, hi):
    fixed = lo == hi
    free = ~fixed
    if not free.any():
        return float(np.max(np.abs(b))) if b.size else 0.0
    method = "bvls" if free.sum() <= BVLS_MAX_COLS else "trf"
    res = lsq_linear(A[:, free], b, bounds=(lo[free], hi[free]), method=method)
    return float(np.max(np.abs(A[:, free] @ res.x - b)))


def classify(model, z, solver_tol: float = 1e-8) -> Dict[str, object]:
    if getattr(model, "soc_complementarity", False):
        return {"klass": "n/a", "reason": "second-order-cone complementarity"}
    pairs = getattr(model, "pairs", None)
    zj = jnp.asarray(z, dtype=float)
    zn = np.asarray(z, dtype=float)
    n = zn.size

    gf = np.asarray(jax.grad(model.f)(zj))
    Je = np.asarray(jax.jacfwd(model.eq)(zj)).reshape(-1, n)
    hv = np.asarray(model.ineq(zj))
    Jh = np.asarray(jax.jacfwd(model.ineq)(zj)).reshape(-1, n)

    row_tol = np.sqrt(solver_tol)
    cols, lo, hi = [], [], []

    def add(block, l, h):
        for k in range(block.shape[0]):
            cols.append(block[k])
            lo.append(l[k])
            hi.append(h[k])

    add(Je, np.full(Je.shape[0], -np.inf), np.full(Je.shape[0], np.inf))
    h_act = hv <= row_tol
    add(Jh, np.zeros(hv.size), np.where(h_act, np.inf, 0.0))
    lb_act = np.isfinite(model.lb) & (zn - model.lb <= row_tol)
    ub_act = np.isfinite(model.ub) & (model.ub - zn <= row_tol)
    eye = np.eye(n)
    add(eye[lb_act], np.zeros(lb_act.sum()), np.full(lb_act.sum(), np.inf))
    add(-eye[ub_act], np.zeros(ub_act.sum()), np.full(ub_act.sum(), np.inf))

    out: Dict[str, object] = {"solver_tol": solver_tol}
    if pairs is None:
        g_act = h2_act = np.zeros(0, bool)
        JG = JH = np.zeros((0, n))
        nb = 0
    else:
        G, H = (np.asarray(v) for v in pairs(zj))
        JG = np.asarray(jax.jacfwd(lambda x: pairs(x)[0])(zj)).reshape(-1, n)
        JH = np.asarray(jax.jacfwd(lambda x: pairs(x)[1])(zj)).reshape(-1, n)
        corner = np.sqrt(solver_tol)
        g_act = np.abs(G) <= corner * max(1.0, float(np.max(np.abs(G))))
        h2_act = np.abs(H) <= corner * max(1.0, float(np.max(np.abs(H))))
        nb = int(np.sum(g_act & h2_act))
        out.update(n_pairs=int(G.size), n_biactive=nb,
                   n_neither=int(np.sum(~g_act & ~h2_act)))
    base = len(cols)
    q = g_act.size
    for J, act in ((JG, g_act), (JH, h2_act)):
        add(J, np.where(act, -np.inf, 0.0), np.where(act, np.inf, 0.0))

    A = np.array(cols).T if cols else np.zeros((n, 0))
    lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
    free = lo != hi
    scale = max(1.0, float(np.max(np.abs(gf))) if gf.size else 1.0,
                float(np.max(np.abs(A[:, free]))) if free.any() else 0.0)
    tol = max(1e-6, solver_tol) * scale
    out.update(dual_scale=scale, resid_tol=tol)

    if pairs is None:
        r = _bvls(A, gf, lo, hi)
        out["residuals"] = {"KKT": r}
        out["klass"] = "KKT" if r <= tol else "none"
        return out

    bi = np.flatnonzero(g_act & h2_act)
    nu = base + bi
    w = base + q + bi

    def with_signs(signs):
        l, h = lo.copy(), hi.copy()
        for a, c, s in zip(nu, w, signs):
            if s == "pp":
                l[a] = l[c] = 0.0
            elif s == "nn":
                h[a] = h[c] = 0.0
            elif s == "nu0":
                l[a] = h[a] = 0.0
            elif s == "w0":
                l[c] = h[c] = 0.0
        return l, h

    res = {"S": _bvls(A, gf, *with_signs(["pp"] * len(bi))),
           "W": _bvls(A, gf, lo, hi)}
    if len(bi) <= MAX_ENUM:
        for klass, opts in (("M", ("pp", "nu0", "w0")), ("C", ("pp", "nn"))):
            res[klass] = min(_bvls(A, gf, *with_signs(c))
                             for c in itertools.product(opts, repeat=len(bi)))
    out["residuals"] = res
    for klass in ("S", "M", "C", "W"):
        if klass in res and res[klass] <= tol:
            out["klass"] = klass
            break
    else:
        out["klass"] = "none"
    if out["klass"] == "W" and len(bi) > MAX_ENUM:
        out["note"] = f"{len(bi)} biactive pairs > MAX_ENUM={MAX_ENUM}; C/M not enumerated"
    return out


def selftest():
    """Solver-free checks on exact points of `nonconvex/complementarity`
    and `maratos`. Each branch the verdict can take is reached: S with no
    biactive pair (x*, f = 17), S with one (f = 25), M-but-not-S (f = 85,
    the point solver defaults return), KKT, and none."""
    import models

    m, _, _, _ = models.get("complementarity")
    cases = [
        (np.array([1.0, 0, 2, 0, 0, 0, 3, 6]), "S", 0),
        (np.array([5.0, 2, 0, 0, 1, 10, 0, 0]), "S", 1),
        (np.array([3.0, 4, 0, 0, 0, 2, 3, 0]), "M", 1),
    ]
    for x, want, nb in cases:
        assert float(np.max(np.abs(np.asarray(m.eq(jnp.asarray(x)))))) < 1e-12, x
        c = classify(m, x)
        assert c["klass"] == want and c["n_biactive"] == nb, (x, c)
    assert classify(m, cases[2][0])["residuals"]["S"] > 1.0  # not S, by a margin

    m, _, _, _ = models.get("maratos")
    assert classify(m, np.array([1.0, 0.0]))["klass"] == "KKT"
    assert classify(m, np.array([0.0, 1.0]))["klass"] == "none"  # feasible, not stationary
    print("stationarity selftest: ok")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    selftest()
