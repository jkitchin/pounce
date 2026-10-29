"""Generic JAX-differentiated MPCC -> NLP lowering and the route drivers.

A model supplies, as JAX functions of the full variable vector z:
  f(z)          objective
  eq(z)         rows that must be 0
  ineq(z)       rows that must be >= 0
  comp(z)       one scalar per complementarity pair, >= 0 by construction
                (G*H for an NCP pair, beta.eta for an SOC pair); the MPCC
                asks for comp(z) = 0
  lb, ub        variable bounds
  extra[name]   optional alternative complementarity row blocks
                (e.g. the paper's full Jordan-product equality rows)

Lowerings of the complementarity block:
  prod_eq    comp = 0
  prod_ineq  comp <= 0
  scholtes   comp <= tau
  jordan_eq  model.extra["jordan"](z) = 0   (drifting only; CALIPSO's rows)

Sparse Jacobian / Hessian patterns, detected structurally from JAX at random
points (the first dense attempt made each drifting solve ~450 s: the
factorization cost of a dense 631x631 KKT dominated).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

BIG = 1e20


@dataclass
class Model:
    name: str
    n: int
    f: Callable
    eq: Callable
    ineq: Callable
    comp: Callable
    lb: np.ndarray
    ub: np.ndarray
    extra: Dict[str, Callable] = field(default_factory=dict)
    # source-level diagnostics: z -> dict of floats
    report: Optional[Callable] = None


def generic_report(model: Model) -> Callable:
    """Source-level violation for a model whose complementarity function is
    a plain product (so |comp| is the residual) and whose cones are all
    nonnegative orthants: max of |eq|, |comp|, inequality and bound
    shortfall."""

    def report(z):
        zj = jnp.asarray(z)
        zn = np.asarray(z, dtype=float)
        e = np.asarray(model.eq(zj))
        c = np.asarray(model.comp(zj))
        h = np.asarray(model.ineq(zj))
        eqv = float(np.max(np.abs(e))) if e.size else 0.0
        cv = float(np.max(np.abs(c))) if c.size else 0.0
        iv = float(np.max(np.maximum(-h, 0))) if h.size else 0.0
        bv = float(np.max(np.maximum(np.maximum(model.lb - zn, zn - model.ub), 0)))
        return {"obj": float(model.f(zj)), "eq_viol": eqv, "compl": cv,
                "ineq_viol": iv, "bound_viol": bv, "violation": max(eqv, cv, iv, bv)}

    return report


# Compiled callbacks and sparsity, shared by every lowering that uses the
# same constraint function (prod_eq / prod_ineq / scholtes differ only in
# row bounds), so a 10-stage continuation compiles once, not ten times.
_CACHE: Dict[tuple, tuple] = {}


def _compiled(model: Model, kind: str, cons: Callable, m: int):
    key = (id(model), kind)
    if key in _CACHE:
        return _CACHE[key]
    f = jax.jit(model.f)
    g = jax.jit(jax.grad(model.f))
    c = jax.jit(cons)
    J = jax.jit(jax.jacfwd(cons))

    def lag(z, lam, sigma):
        return sigma * model.f(z) + jnp.dot(lam, cons(z))

    H = jax.jit(jax.hessian(lag))

    # Structural sparsity: union of nonzeros at a few random points
    # (random lam for the Hessian). A structural zero is zero
    # everywhere; a numerical coincidence at three independent
    # Gaussian points has probability zero.
    n = model.n
    rng = np.random.default_rng(12345)
    jmask = np.zeros((m, n), bool)
    hmask = np.zeros((n, n), bool)
    for _ in range(3):
        zr = jnp.asarray(rng.standard_normal(n))
        lr = jnp.asarray(rng.standard_normal(m))
        jmask |= np.asarray(J(zr)) != 0
        hmask |= np.asarray(H(zr, lr, 1.0)) != 0
    hmask = np.tril(hmask | hmask.T)
    _CACHE[key] = (f, g, c, J, H, np.nonzero(jmask), np.nonzero(hmask))
    return _CACHE[key]


class LoweredNlp:
    def __init__(self, model: Model, lowering: str, tau: float = 0.0):
        self.model = model
        self.lowering = lowering
        self.tau = tau
        z0 = jnp.zeros(model.n)
        neq = int(model.eq(z0).shape[0])
        nin = int(model.ineq(z0).shape[0])
        if lowering == "jordan_eq":
            compf = model.extra["jordan"]
        else:
            compf = model.comp
        ncp = int(compf(z0).shape[0])

        def cons(z):
            return jnp.concatenate([model.eq(z), model.ineq(z), compf(z)])

        self.m = neq + nin + ncp
        cl = np.concatenate([np.zeros(neq), np.zeros(nin), np.zeros(ncp)])
        cu = np.concatenate([np.zeros(neq), np.full(nin, BIG), np.zeros(ncp)])
        if lowering == "prod_ineq":
            cl[neq + nin:] = -BIG
            cu[neq + nin:] = 0.0
        elif lowering == "scholtes":
            cl[neq + nin:] = -BIG
            cu[neq + nin:] = tau
        elif lowering in ("prod_eq", "jordan_eq"):
            pass
        else:
            raise ValueError(lowering)
        self.cl, self.cu = cl, cu
        self.n = model.n
        self.lb = np.where(np.isfinite(model.lb), model.lb, -BIG)
        self.ub = np.where(np.isfinite(model.ub), model.ub, BIG)

        kind = "jordan" if lowering == "jordan_eq" else "comp"
        (self._f, self._g, self._c, self._J, self._H,
         self._jidx, self._hidx) = _compiled(model, kind, cons, self.m)

    # cyipopt-style callbacks
    def objective(self, x):
        return float(self._f(jnp.asarray(x)))

    def gradient(self, x):
        return np.asarray(self._g(jnp.asarray(x)))

    def constraints(self, x):
        return np.asarray(self._c(jnp.asarray(x)))

    def jacobian(self, x):
        return np.asarray(self._J(jnp.asarray(x)))[self._jidx]

    def jacobianstructure(self):
        return self._jidx

    def hessianstructure(self):
        return self._hidx

    def hessian(self, x, lagrange, obj_factor):
        h = np.asarray(self._H(jnp.asarray(x), jnp.asarray(lagrange), obj_factor))
        return h[self._hidx]


BASE_OPTIONS = {
    "print_level": 0,
    "tol": 1e-8,
    "max_iter": 3000,
    "bound_relax_factor": 0.0,
    "honor_original_bounds": "yes",
}

TAU_SCHEDULE = (1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
OK = (0, 1)


#: Which solver runs the lowered NLP. ``pounce`` (default) or ``ipopt``
#: (cyipopt, e.g. conda-forge's Ipopt 3.14). For Ipopt,
#: ``IPOPT_LINEAR_SOLVER`` picks the backend (``mumps``, ``spral``, or
#: ``ma57`` with ``IPOPT_HSLLIB`` pointing at a licensed libhsl/libcoinhsl).
#: For POUNCE, ``POUNCE_LINEAR_SOLVER=ma57`` needs a ``--features ma57``
#: build of the extension; ``feral`` is the default.
SOLVER = os.environ.get("CALIPSO_SOLVER", "pounce")

#: Options only POUNCE has. A route that needs one cannot run on Ipopt.
POUNCE_ONLY = {"l1_exact_penalty_barrier"}


def warm_options(mu):
    """The options ``pounce.WarmStart.options()`` sets (python/pounce/
    _warm_start.py), all of which are upstream Ipopt options: applying the
    same set by hand is what makes the two continuations comparable."""
    p = 1e-9
    return {
        "warm_start_init_point": "yes",
        "mu_init": float(np.clip(mu, 1e-9, 1e-1)) if mu else 1e-6,
        "warm_start_bound_push": p,
        "warm_start_bound_frac": p,
        "warm_start_slack_bound_push": p,
        "warm_start_slack_bound_frac": p,
        "warm_start_mult_bound_push": p,
    }


#: Ipopt's ApplicationReturnStatus, by the names POUNCE reports, so the two
#: solvers' results group under one vocabulary.
IPOPT_STATUS = {
    0: "Solve_Succeeded", 1: "Solved_To_Acceptable_Level",
    2: "Infeasible_Problem_Detected", 3: "Search_Direction_Becomes_Too_Small",
    4: "Diverging_Iterates", 5: "User_Requested_Stop", 6: "Feasible_Point_Found",
    -1: "Maximum_Iterations_Exceeded", -2: "Restoration_Failed",
    -3: "Error_In_Step_Computation", -4: "Maximum_CpuTime_Exceeded",
    -5: "Maximum_WallTime_Exceeded", -10: "Not_Enough_Degrees_Of_Freedom",
    -11: "Invalid_Problem_Definition", -12: "Invalid_Option",
    -13: "Invalid_Number_Detected", -100: "Unrecoverable_Exception",
    -101: "NonIpopt_Exception_Thrown", -102: "Insufficient_Memory",
    -199: "Internal_Error",
}


class _IpoptCallbacks:
    """``nlp`` plus an ``intermediate`` that records the iteration count and
    the barrier parameter, which cyipopt's ``info`` does not return."""

    def __init__(self, nlp):
        self._nlp = nlp
        self.iter_count = 0
        self.mu = None

    def __getattr__(self, name):
        return getattr(self._nlp, name)

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        self.iter_count = int(iter_count)
        self.mu = float(mu)
        return True


def _solve_ipopt(nlp, opts, x0, warm):
    import cyipopt

    cb = _IpoptCallbacks(nlp)
    prob = cyipopt.Problem(n=nlp.n, m=nlp.m, problem_obj=cb,
                           lb=nlp.lb, ub=nlp.ub, cl=nlp.cl, cu=nlp.cu)
    opts = dict(opts)
    if warm is not None:
        opts.update(warm_options(warm["mu"]))
    ls = os.environ.get("IPOPT_LINEAR_SOLVER")
    if ls:
        opts["linear_solver"] = ls
    if os.environ.get("IPOPT_HSLLIB"):
        opts["hsllib"] = os.environ["IPOPT_HSLLIB"]
    for k, v in opts.items():
        prob.add_option(k, v)
    kw = {}
    if warm is not None:
        kw = {"lagrange": warm["mult_g"], "zl": warm["mult_x_L"], "zu": warm["mult_x_U"]}
    x0 = warm["x"] if warm is not None else np.asarray(x0, dtype=float)
    t0 = time.perf_counter()
    x, info = prob.solve(np.asarray(x0, dtype=float), **kw)
    wall = time.perf_counter() - t0
    st = int(info["status"])
    info = dict(info, iter_count=cb.iter_count, mu=cb.mu or 0.0,
                status_msg=IPOPT_STATUS.get(st, f"status_{st}"))
    return np.asarray(x), info, wall


def solve_once(nlp: LoweredNlp, opts: dict, x0, warm=None):
    if SOLVER == "ipopt":
        return _solve_ipopt(nlp, opts, x0, warm)
    import pounce

    prob = pounce.Problem(n=nlp.n, m=nlp.m, problem_obj=nlp,
                          lb=nlp.lb, ub=nlp.ub, cl=nlp.cl, cu=nlp.cu)
    opts = dict(opts)
    if os.environ.get("POUNCE_LINEAR_SOLVER"):
        opts["linear_solver"] = os.environ["POUNCE_LINEAR_SOLVER"]
    for k, v in opts.items():
        prob.add_option(k, v)
    kw = {"x0": np.asarray(x0, dtype=float)}
    if warm is not None:
        kw["warm_start"] = pounce.WarmStart(
            x=warm["x"], lagrange=warm["mult_g"], zl=warm["mult_x_L"],
            zu=warm["mult_x_U"], mu=warm["mu"])
    t0 = time.perf_counter()
    x, info = prob.solve(**kw)
    return np.asarray(x), info, time.perf_counter() - t0


def _warm(x, info):
    return {"x": np.array(x), "mult_g": np.array(info["mult_g"]),
            "mult_x_L": np.array(info["mult_x_L"]), "mult_x_U": np.array(info["mult_x_U"]),
            "mu": float(info.get("mu", 0.0)) or None}


def run_route(model: Model, route: str, x0, extra_opts=None):
    """Returns dict(x, status, status_msg, iters, wall, stages)."""
    opts = dict(BASE_OPTIONS)
    if route == "defaults_eq":
        # closest to what the paper handed Ipopt: product equality,
        # solver defaults (bound_relax_factor=1e-8 etc.)
        opts = {"print_level": 0, "max_iter": 3000}
    if extra_opts:
        opts.update(extra_opts)

    single = {
        "direct": ("prod_ineq", {}),
        "ncp_eq": ("prod_eq", {}),
        "defaults_eq": ("prod_eq", {}),
        "ncp_eq_l1": ("prod_eq", {"l1_exact_penalty_barrier": "yes"}),
        "jordan_eq": ("jordan_eq", {}),
    }
    if route in single:
        low, o = single[route]
        if SOLVER != "pounce" and POUNCE_ONLY & set(o):
            raise NotImplementedError(f"{route} needs a POUNCE-only option")
        nlp = LoweredNlp(model, low)
        x, info, wall = solve_once(nlp, {**opts, **o}, x0)
        return {"x": x, "status": int(info["status"]), "status_msg": str(info["status_msg"]),
                "iters": int(info["iter_count"]), "wall": wall, "stages": 1}

    if route in ("scholtes_then_ncp", "scholtes_then_ineq", "scholtes_only"):
        finish = {"scholtes_then_ncp": "prod_eq", "scholtes_then_ineq": "prod_ineq",
                  "scholtes_only": None}[route]
        warm = None
        xcur = np.asarray(x0, float)
        iters, wall, stages = 0, 0.0, 0
        last_ok = None
        log = []
        for tau in TAU_SCHEDULE:
            nlp = LoweredNlp(model, "scholtes", tau)
            x, info, w = solve_once(nlp, opts, xcur, warm)
            iters += int(info["iter_count"]); wall += w; stages += 1
            st = int(info["status"])
            if st not in OK and warm is not None:
                # restart rung: primal-only
                x, info, w = solve_once(nlp, opts, xcur, None)
                iters += int(info["iter_count"]); wall += w; stages += 1
                st = int(info["status"])
            log.append((tau, st, int(info["iter_count"])))
            if st not in OK:
                break
            xcur, warm, last_ok = x, _warm(x, info), (x, info)
        if last_ok is None:
            return {"x": xcur, "status": st, "status_msg": str(info["status_msg"]),
                    "iters": iters, "wall": wall, "stages": stages, "log": log}
        if finish is None:
            x, info = last_ok
            return {"x": x, "status": int(info["status"]), "status_msg": str(info["status_msg"]),
                    "iters": iters, "wall": wall, "stages": stages, "log": log}
        nlp = LoweredNlp(model, finish)
        x, info, w = solve_once(nlp, opts, xcur, warm)
        iters += int(info["iter_count"]); wall += w; stages += 1
        st = int(info["status"])
        log.append(("finish", st, int(info["iter_count"])))
        if st not in OK:
            # keep the continuation's answer, flag it
            x = xcur
        return {"x": x, "status": st, "status_msg": str(info["status_msg"]),
                "iters": iters, "wall": wall, "stages": stages, "log": log,
                "finish_failed": st not in OK}
    raise ValueError(route)
