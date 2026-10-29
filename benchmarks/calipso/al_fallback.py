"""Prototype augmented-Lagrangian (AL) fallback rung over POUNCE.

Experiment 1 of dev-notes/research/calipso-contact-plan.md: does an AL outer
loop rescue problems on which POUNCE's IPM stops at a point of local
infeasibility? CALIPSO treats equalities with a primal-dual augmented
Lagrangian and inequalities with an interior point; this prototype makes the
same split *around* POUNCE rather than inside it:

    outer k:  x_k = argmin  f(x) + lam_k . c_E(x) + rho_k/2 |c_E(x)|^2
                     s.t.   gl_I <= g_I(x) <= gu_I,  xl <= x <= xu       (POUNCE)
              lam_{k+1} = lam_k + rho_k c_E(x_k)
              rho_{k+1} = 10 rho_k  unless |c_E| fell by 4x

then one ordinary POUNCE solve of the original problem warm-started from the
AL point, so the answer carries POUNCE's own convergence verdict and
multipliers. It is a measurement device, not a design: the outer loop,
update rule and constants are the textbook ones (Nocedal & Wright, alg.
17.4) and nothing is tuned.

The problem is anything with the `pounce.NlProblem` callback surface
(`n, m, x0, x_l, x_u, g_l, g_u, objective, gradient, constraints,
jacobian[_structure], hessian[_structure]`): an `.nl` file through
`pounce.read_nl`, or `NlpAdapter` below for a JAX model.
"""
from __future__ import annotations

import numpy as np

BIG = 1e20
OK = (0, 1)


class _ALSubproblem:
    """cyipopt-style callbacks for one AL subproblem."""

    def __init__(self, nl, eq, lam, rho):
        self.nl, self.eq, self.lam, self.rho = nl, eq, lam, rho
        self.ineq = ~eq
        jr, jc = (np.asarray(a) for a in nl.jacobian_structure())
        self.jr, self.jc = jr, jc
        keep = self.ineq[jr]
        rows_I = np.cumsum(self.ineq) - 1
        self.jkeep = keep
        self.sub_jr, self.sub_jc = rows_I[jr[keep]], jc[keep]
        self.eq_entries = ~keep
        hr, hc = (np.asarray(a) for a in nl.hessian_structure())
        # Gauss-Newton term rho J_E^T J_E: its pattern joins the Hessian's.
        n = nl.n
        pat = set(zip(hr.tolist(), hc.tolist()))
        Jr_E, Jc_E = jr[self.eq_entries], jc[self.eq_entries]
        by_row = {}
        for r, c in zip(Jr_E.tolist(), Jc_E.tolist()):
            by_row.setdefault(r, []).append(c)
        for cols in by_row.values():
            for a in cols:
                for b in cols:
                    if a >= b:
                        pat.add((a, b))
        pat = sorted(pat)
        self.hr = np.array([p[0] for p in pat], dtype=int)
        self.hc = np.array([p[1] for p in pat], dtype=int)
        self.hpos = {p: k for k, p in enumerate(pat)}
        self.orig_h = np.array([self.hpos[(r, c)] for r, c in zip(hr.tolist(), hc.tolist())], dtype=int)
        self.n, self.m = n, int(self.ineq.sum())

    def _cE(self, x):
        g = np.asarray(self.nl.constraints(x))
        return g[self.eq] - np.asarray(self.nl.g_l)[self.eq]

    def objective(self, x):
        c = self._cE(x)
        return float(self.nl.objective(x) + self.lam @ c + 0.5 * self.rho * c @ c)

    def gradient(self, x):
        c = self._cE(x)
        w = np.zeros(self.nl.m)
        w[self.eq] = self.lam + self.rho * c
        J = np.asarray(self.nl.jacobian(x))
        g = np.asarray(self.nl.gradient(x)).copy()
        np.add.at(g, self.jc, w[self.jr] * J)
        return g

    def constraints(self, x):
        return np.asarray(self.nl.constraints(x))[self.ineq]

    def jacobianstructure(self):
        return self.sub_jr, self.sub_jc

    def jacobian(self, x):
        return np.asarray(self.nl.jacobian(x))[self.jkeep]

    def hessianstructure(self):
        return self.hr, self.hc

    def hessian(self, x, lagrange, obj_factor):
        c = self._cE(x)
        full = np.zeros(self.nl.m)
        full[self.eq] = obj_factor * (self.lam + self.rho * c)
        full[self.ineq] = lagrange
        h = np.zeros(self.hr.size)
        h[self.orig_h] = np.asarray(self.nl.hessian(x, full, obj_factor))
        # + obj_factor * rho * J_E^T J_E, lower triangle
        J = np.asarray(self.nl.jacobian(x))
        rows = {}
        for k in np.flatnonzero(self.eq_entries):
            rows.setdefault(int(self.jr[k]), []).append((int(self.jc[k]), float(J[k])))
        s = obj_factor * self.rho
        for ents in rows.values():
            for a, va in ents:
                for b, vb in ents:
                    if a >= b:
                        h[self.hpos[(a, b)]] += s * va * vb
        return h


def _bounds(v):
    v = np.asarray(v, dtype=float)
    return np.where(np.isfinite(v), v, np.sign(v) * BIG)


class _Cy:
    """NlProblem's method names -> the cyipopt names pounce.Problem calls."""

    def __init__(self, nl):
        self.nl = nl

    def objective(self, x):
        return self.nl.objective(x)

    def gradient(self, x):
        return self.nl.gradient(x)

    def constraints(self, x):
        return self.nl.constraints(x)

    def jacobianstructure(self):
        return self.nl.jacobian_structure()

    def jacobian(self, x):
        return self.nl.jacobian(x)

    def hessianstructure(self):
        return self.nl.hessian_structure()

    def hessian(self, x, lagrange, obj_factor):
        return self.nl.hessian(x, lagrange, obj_factor)


def solve_plain(nl, x0, options=None, warm=None):
    import pounce

    prob = pounce.Problem(n=nl.n, m=nl.m, problem_obj=_Cy(nl), lb=_bounds(nl.x_l), ub=_bounds(nl.x_u),
                          cl=_bounds(nl.g_l), cu=_bounds(nl.g_u))
    for k, v in {"print_level": 0, **(options or {})}.items():
        prob.add_option(k, v)
    kw = {"x0": np.asarray(x0, dtype=float)}
    if warm is not None:
        kw["warm_start"] = pounce.WarmStart(x=np.asarray(x0, dtype=float), mu=warm)
    x, info = prob.solve(**kw)
    return np.asarray(x), info


def constraint_violation(nl, x):
    g = np.asarray(nl.constraints(x))
    gl, gu = np.asarray(nl.g_l), np.asarray(nl.g_u)
    v = np.maximum(np.maximum(gl - g, g - gu), 0)
    xv = np.maximum(np.maximum(np.asarray(nl.x_l) - x, x - np.asarray(nl.x_u)), 0)
    return float(max(v.max(initial=0.0), xv.max(initial=0.0)))


def solve_al(nl, x0, options=None, rho0=10.0, max_outer=30, tol=1e-8):
    """AL outer loop, then a warm POUNCE polish. Returns (x, info, record)."""
    import pounce

    gl, gu = np.asarray(nl.g_l), np.asarray(nl.g_u)
    eq = np.isclose(gl, gu) & np.isfinite(gl)
    lam = np.zeros(int(eq.sum()))
    rho = rho0
    x = np.asarray(x0, dtype=float)
    inner_iters, last_c = 0, np.inf
    log = []
    for k in range(max_outer):
        sub = _ALSubproblem(nl, eq, lam, rho)
        prob = pounce.Problem(n=sub.n, m=sub.m, problem_obj=sub, lb=_bounds(nl.x_l), ub=_bounds(nl.x_u),
                              cl=_bounds(gl[~eq]), cu=_bounds(gu[~eq]))
        for key, v in {"print_level": 0, "max_iter": 500, **(options or {})}.items():
            prob.add_option(key, v)
        x, info = prob.solve(x0=x)
        x = np.asarray(x)
        inner_iters += int(info["iter_count"])
        c = sub._cE(x)
        cn = float(np.max(np.abs(c))) if c.size else 0.0
        log.append((k, int(info["status"]), int(info["iter_count"]), rho, cn))
        if cn <= tol and int(info["status"]) in OK:
            break
        lam = lam + rho * c
        if cn > 0.25 * last_c:
            rho *= 10.0
        last_c = cn
    xp, infop = solve_plain(nl, x, options)
    return xp, infop, {"outer": log, "inner_iters": inner_iters, "polish_iters": int(infop["iter_count"]),
                       "al_point_viol": constraint_violation(nl, x)}


class NlpAdapter:
    """The NlProblem callback surface for a model from this directory's
    registry (equality rows, `ineq >= 0` rows and bounds; no complementarity
    block -- for plain NLPs such as `wachter`)."""

    def __init__(self, model, x0):
        import jax
        import jax.numpy as jnp

        self.model = model
        z = jnp.asarray(x0, dtype=float)
        ne, ni = int(model.eq(z).shape[0]), int(model.ineq(z).shape[0])
        cons = lambda v: jnp.concatenate([model.eq(v), model.ineq(v)])
        self.n, self.m = model.n, ne + ni
        self.x0 = np.asarray(x0, dtype=float)
        self.x_l, self.x_u = np.asarray(model.lb, float), np.asarray(model.ub, float)
        self.g_l = np.zeros(self.m)
        self.g_u = np.concatenate([np.zeros(ne), np.full(ni, np.inf)])
        self._f = jax.jit(model.f)
        self._g = jax.jit(jax.grad(model.f))
        self._c = jax.jit(cons)
        self._J = jax.jit(jax.jacfwd(cons))
        self._H = jax.jit(jax.hessian(lambda v, lam, s: s * model.f(v) + jnp.dot(lam, cons(v))))
        self._jr, self._jc = np.nonzero(np.ones((self.m, self.n), bool))
        self._hr, self._hc = np.tril_indices(self.n)

    def objective(self, x):
        return float(self._f(x))

    def gradient(self, x):
        return np.asarray(self._g(x))

    def constraints(self, x):
        return np.asarray(self._c(x))

    def jacobian_structure(self):
        return self._jr, self._jc

    def jacobian(self, x):
        return np.asarray(self._J(x)).ravel()

    def hessian_structure(self):
        return self._hr, self._hc

    def hessian(self, x, lam, obj_factor=1.0):
        return np.asarray(self._H(x, np.asarray(lam, float), obj_factor))[self._hr, self._hc]

