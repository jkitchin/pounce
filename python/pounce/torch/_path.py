"""Numerical-continuation / predictor–corrector path following on the
held KKT factor (pounce#90), PyTorch frontend (pounce#109).

PyTorch mirror of :mod:`pounce.jax._path`. Traces a solution path of a
parametric NLP

    min_x  f(x, θ)   s.t.  g(x, θ) = 0,  lb ≤ x ≤ ub

by *composing* the post-solve sensitivity primitives rather than
re-solving the NLP at every step:

* **predict** — extrapolate along the held-factor sensitivity
  ``∂x*/∂θ`` (:meth:`TorchProblem.jvp_from_state`);
* **monitor** (no solve) — the KKT residual at the predicted point plus
  the active-set margin (:meth:`TorchProblem.active_set_margin`);
* **correct** — only when the monitor trips: a warm-started, barrier-μ
  seeded re-solve that re-anchors the factor in one solve
  (:meth:`TorchProblem.warm_anchor`).

Because PyTorch is eager, the inverse-map RHS is a plain Python callable
(no ``jax.pure_callback`` wrapper needed) — drop it straight into a
``scipy.integrate`` / ``torchdiffeq`` vector field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.func import grad, jacrev

from ._build import _DT, _t, _to_np

_OK_STATUS = ("Solve_Succeeded", "Solved_To_Acceptable_Level")


def _require_equality_constraints(tp, where):
    """Guard: path continuation here assumes a fixed active set in which
    every general constraint is an equality (``cl == cu``)."""
    if tp._m > 0:
        cl = np.asarray(tp._cl_for_classify)
        cu = np.asarray(tp._cu_for_classify)
        if np.any(cl != cu):
            raise ValueError(
                f"{where} supports equality constraints (cl == cu) and "
                "variable bounds only; this problem has two-sided inequality "
                "constraints (cl != cu), for which the active-set monitor and "
                "KKT residual are not yet valid. Reformulate the inequalities "
                "with slack equalities, or remove them."
            )


def inverse_map_rhs(tp, dy_ds, *, output=None, x0=None, warm=False):
    """Build the right-hand side of the inverse / uncertainty-mapping ODE
    (Alves–Kitchin–Lima):

    .. math::  \\frac{d\\theta}{ds} = \\Big(\\frac{\\partial y}{\\partial
        \\theta}\\Big)^{-1} \\frac{dy}{ds},

    where ``y = output(x*(θ), θ)`` is an output of the embedded optimizer
    ``x*(θ) = argmin_x f(x, θ) s.t. …``. With ``J = ∂x*/∂θ`` from the held
    KKT factor and the output Jacobians ``∂h/∂x``, ``∂h/∂θ`` by autodiff,

    .. math::  \\frac{\\partial y}{\\partial \\theta}
        = \\frac{\\partial h}{\\partial x} J + \\frac{\\partial h}{\\partial \\theta},
        \\qquad \\frac{d\\theta}{ds}
        = \\Big(\\frac{\\partial y}{\\partial \\theta}\\Big)^{-1} \\frac{dy}{ds}.

    The output and parameter dimensions must match (square ``∂y/∂θ``).

    Parameters mirror :func:`pounce.jax.inverse_map_rhs`. Returns a plain
    callable ``f(s, theta) -> dtheta_ds`` of shape ``(p,)`` — drop-in for a
    ``scipy.integrate`` / ``torchdiffeq`` RHS.
    """
    if len(tp._p_shape) != 1:
        raise ValueError(
            f"inverse_map_rhs requires a 1-D θ (p_shape={tp._p_shape})."
        )
    n = tp._n
    p = tp._p_shape[0]
    x0_arr = torch.zeros(n, dtype=_DT) if x0 is None else _t(x0)
    h = output if output is not None else (lambda x, theta: x)

    # ∂y/∂θ must be square (k == p). Probe the output dimension up front.
    if output is None:
        k = n
    else:
        out = h(torch.zeros(n, dtype=_DT), torch.zeros(p, dtype=_DT))
        if out.ndim != 1:
            raise ValueError(
                f"inverse_map_rhs output h(x, θ) must be 1-D; got {tuple(out.shape)}."
            )
        k = out.shape[0]
    if k != p:
        detail = (
            " (the default identity output ⇒ k = n, so this needs n == p)"
            if output is None else ""
        )
        raise ValueError(
            "inverse_map_rhs requires a square output sensitivity ∂y/∂θ: the "
            f"output dimension k={k} must equal the parameter dimension "
            f"p={p}{detail}."
        )

    cache = {"x": _to_np(x0_arr), "duals": None, "mu": None}

    def _solve(theta):
        if not warm:
            x_star, _duals, J = tp.solve_with_jacobian(theta, x0_arr)
            return x_star, J
        state, info = tp.warm_anchor(
            theta, torch.as_tensor(cache["x"], dtype=_DT),
            duals=cache["duals"], mu=cache["mu"],
        )
        try:
            x_star = state.x_star[0]
            J = tp.sensitivity(state)
            cache["x"] = _to_np(x_star)
            cache["duals"] = tuple(_to_np(d[0]) for d in state.duals)
            cache["mu"] = float(info["mu"])
        finally:
            state.close()
        return x_star, J

    def f(s, theta):
        theta_t = _t(theta)
        x_star, J = _solve(theta_t)
        h_x = jacrev(lambda xx: h(xx, theta_t))(x_star)       # (k, n)
        h_th = jacrev(lambda th: h(x_star, th))(theta_t)      # (k, p)
        dy_dtheta = h_x @ J + h_th
        v = dy_ds(s) if callable(dy_ds) else dy_ds
        dtheta_ds = torch.linalg.solve(dy_dtheta, _t(v))
        return dtheta_ds

    return f


@dataclass
class PathTrace:
    """Result of a path-following run (mirror of the JAX dataclass)."""

    s: np.ndarray
    theta: np.ndarray
    x: np.ndarray
    lam: np.ndarray
    n_steps: int
    n_correctors: int
    n_accepts: int
    active_set_changes: list = field(default_factory=list)
    turning_points: list = field(default_factory=list)
    status: str = "ok"


class PathFollower:
    """Predictor–corrector path follower over a :class:`TorchProblem`
    (pounce#90). Mirror of :class:`pounce.jax.PathFollower`."""

    def __init__(
        self,
        tp,
        *,
        monitor_tol: float = 1e-6,
        active_margin_tol: float = 1e-4,
        ds0: float = 0.05,
        ds_min: float = 1e-4,
        ds_max: float = 0.25,
        grow: float = 1.5,
        shrink: float = 0.5,
        max_steps: int = 100_000,
    ):
        self._tp = tp
        self._monitor_tol = float(monitor_tol)
        self._active_margin_tol = float(active_margin_tol)
        self._ds0 = float(ds0)
        self._ds_min = float(ds_min)
        self._ds_max = float(ds_max)
        self._grow = float(grow)
        self._shrink = float(shrink)
        self._max_steps = int(max_steps)

    # ----- monitors -----

    def _kkt_residual(self, x, theta, lam, zL, zU) -> float:
        tp = self._tp
        m = tp._m
        gradf = grad(lambda xx: tp._f(xx, theta))(x)
        stat = gradf - zL + zU
        cviol = 0.0
        if m > 0:
            gx = tp._g(x, theta)
            Jg = jacrev(lambda xx: tp._g(xx, theta))(x)
            stat = stat + Jg.T @ lam
            cviol = float(torch.max(torch.abs(gx)))
        r = float(torch.max(torch.abs(stat))) + cviol
        if tp._lb is not None:
            r += float(torch.clamp(_t(tp._lb) - x, min=0.0).max())
        if tp._ub is not None:
            r += float(torch.clamp(x - _t(tp._ub), min=0.0).max())
        return r

    def _margin_at(self, x, lam, zL, zU, theta) -> float:
        r = self._tp._margin_arrays(
            x[None], lam[None], zL[None], zU[None], _t(theta)[None], 1,
        )
        return float(r["margin"][0])

    def _active_signature(self, lam, zL, zU, active_tol=1e-6):
        tp = self._tp
        sig = (
            tuple(bool(v) for v in _to_np(zL > active_tol))
            + tuple(bool(v) for v in _to_np(zU > active_tol))
        )
        if tp._m > 0:
            cl = np.asarray(tp._cl_for_classify)
            cu = np.asarray(tp._cu_for_classify)
            is_ineq = cl != cu
            lam_np = _to_np(lam)
            sig = sig + tuple(
                bool(is_ineq[i] and abs(lam_np[i]) > active_tol)
                for i in range(tp._m)
            )
        return sig

    # ----- parameter continuation -----

    def follow(self, theta_of_s, s_span, x0) -> PathTrace:
        """Trace ``x*(θ(s))`` for a prescribed path ``θ(s)`` (pounce#90)."""
        tp = self._tp
        _require_equality_constraints(tp, "PathFollower.follow")
        s0, s1 = float(s_span[0]), float(s_span[1])
        if not (s1 > s0):
            raise ValueError("s_span must have s1 > s0")

        def th(s):
            return _t(theta_of_s(s))

        state, info = tp.warm_anchor(th(s0), _t(x0))
        if info["status_msg"] not in _OK_STATUS:
            state.close()
            raise RuntimeError(
                f"pounce.torch: anchor solve failed at s0 ({info['status_msg']})."
            )
        x = state.x_star[0]
        lam, zL, zU = (d[0] for d in state.duals)
        mu = float(info["mu"])
        sig = self._active_signature(lam, zL, zU)

        S = [s0]; TH = [_to_np(th(s0))]; X = [_to_np(x)]; LAM = [_to_np(lam)]
        n_corr = n_acc = n_steps = 0
        as_changes: list = []
        status = "ok"

        s = s0
        ds = self._ds0
        while s < s1 - 1e-12 and n_steps < self._max_steps:
            n_steps += 1
            ds = min(ds, s1 - s)
            s_new = s + ds
            th_cur = th(s)
            th_new = th(s_new)
            dth = th_new - th_cur

            if tp._m > 0:
                dx, dlam = tp.jvp_from_state(state, dth, with_duals=True)
                lam_pred = lam + dlam
            else:
                dx = tp.jvp_from_state(state, dth)
                lam_pred = lam
            x_pred = x + dx

            r = self._kkt_residual(x_pred, th_new, lam_pred, zL, zU)
            margin = self._margin_at(x_pred, lam_pred, zL, zU, th_new)

            if r <= self._monitor_tol and margin > self._active_margin_tol:
                x = x_pred
                lam = lam_pred
                s = s_new
                n_acc += 1
                S.append(s); TH.append(_to_np(th_new)); X.append(_to_np(x)); LAM.append(_to_np(lam))
                ds = min(ds * self._grow, self._ds_max)
                continue

            new_state, cinfo = tp.warm_anchor(
                th_new, x_pred, duals=(lam_pred, zL, zU), mu=mu,
            )
            if cinfo["status_msg"] not in _OK_STATUS:
                new_state.close()
                ds *= self._shrink
                if ds < self._ds_min:
                    status = "corrector_failed"
                    break
                continue

            state.close()
            state = new_state
            x = state.x_star[0]
            lam, zL, zU = (d[0] for d in state.duals)
            mu = float(cinfo["mu"])
            n_corr += 1
            s = s_new

            new_sig = self._active_signature(lam, zL, zU)
            if new_sig != sig:
                as_changes.append(s)
                sig = new_sig
                ds = min(self._ds0, max(ds * self._shrink, self._ds_min))
            else:
                iters = int(cinfo["iter_count"])
                if iters <= 3:
                    ds = min(ds * self._grow, self._ds_max)
                elif iters >= 10:
                    ds = max(ds * self._shrink, self._ds_min)

            S.append(s); TH.append(_to_np(th_new)); X.append(_to_np(x)); LAM.append(_to_np(lam))

        state.close()
        if n_steps >= self._max_steps and s < s1 - 1e-12:
            status = "max_steps"
        return PathTrace(
            s=np.asarray(S), theta=np.asarray(TH), x=np.asarray(X), lam=np.asarray(LAM),
            n_steps=n_steps, n_correctors=n_corr, n_accepts=n_acc,
            active_set_changes=as_changes, turning_points=[], status=status,
        )

    # ----- pseudo-arclength continuation (folds) -----

    def trace_arclength(
        self, x0, theta0, *, ds: float = 0.05, n_steps: int = 200,
        direction: float = 1.0, newton_tol: float = 1e-9, newton_max: int = 40,
    ) -> PathTrace:
        """Pseudo-arclength continuation of the solution curve for a
        **scalar** parameter ``θ``, tracing *past folds* (pounce#90)."""
        tp = self._tp
        n, m = tp._n, tp._m
        if len(tp._p_shape) != 0 and tp._p_shape != (1,):
            raise ValueError(
                "trace_arclength supports a scalar parameter only "
                f"(p_shape={tp._p_shape}); use follow(...) for a path."
            )
        _require_equality_constraints(tp, "PathFollower.trace_arclength")
        d = n + m

        def _theta_arg(theta_scalar):
            return theta_scalar if tp._p_shape == () else theta_scalar[None]

        def R(u):
            x = u[:n]
            theta = _theta_arg(u[d])
            gradf = grad(lambda xx: tp._f(xx, theta))(x)
            if m > 0:
                lam = u[n:d]
                Jg = jacrev(lambda xx: tp._g(xx, theta))(x)
                return torch.cat([gradf + Jg.T @ lam, tp._g(x, theta)])
            return gradf

        dR = jacrev(R)

        u = torch.cat([
            _t(x0), torch.zeros(m, dtype=_DT), torch.tensor([float(theta0)], dtype=_DT),
        ])

        for _ in range(newton_max):
            res = R(u)
            if float(torch.max(torch.abs(res))) < newton_tol:
                break
            A = dR(u)[:, :d]
            u = u.clone()
            u[:d] = u[:d] - torch.linalg.solve(A, res)

        def tangent(u_at, t_prev):
            A = dR(u_at)
            _, _, Vh = torch.linalg.svd(A)
            t = Vh[-1]
            t = t / torch.linalg.norm(t)
            if t_prev is None:
                if float(t[d]) * direction < 0:
                    t = -t
            elif float(torch.dot(t, t_prev)) < 0:
                t = -t
            return t

        t = tangent(u, None)
        X = [_to_np(u[:n])]; TH = [float(u[d])]; LAM = [_to_np(u[n:d])]
        turning: list = []
        status = "ok"

        for _ in range(n_steps):
            u_pred = u + ds * t
            uc = u_pred
            ok = False
            for _ in range(newton_max):
                res = R(uc)
                arc = torch.dot(t, uc - u_pred)
                F = torch.cat([res, arc.reshape(1)])
                if float(torch.max(torch.abs(F))) < newton_tol:
                    ok = True
                    break
                A = dR(uc)
                Aaug = torch.cat([A, t[None, :]], dim=0)
                uc = uc - torch.linalg.solve(Aaug, F)
            if not ok:
                status = "corrector_failed"
                break
            u = uc
            t_new = tangent(u, t)
            if float(t[d]) * float(t_new[d]) < 0:
                turning.append(float(u[d]))
            t = t_new
            X.append(_to_np(u[:n])); TH.append(float(u[d])); LAM.append(_to_np(u[n:d]))

        K = len(X)
        return PathTrace(
            s=np.arange(K) * ds, theta=np.asarray(TH), x=np.asarray(X), lam=np.asarray(LAM),
            n_steps=K - 1, n_correctors=K - 1, n_accepts=0,
            active_set_changes=[], turning_points=turning, status=status,
        )
