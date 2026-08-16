"""The scale axis: one MPC problem, several horizons.

Every other family in this suite is small and fixed-size, because the
suite measures *warm-start behavior* and small problems measure it
cleanly. That leaves one question unanswered: how the answers move with
problem size. The active-set path's per-QP cost grows with the active
set, and the sparse KKT and Schur machinery only start to matter at
size, so "warm-sqp wins" measured at n = 47 does not automatically hold
at n = 482.

This module is the same linear MPC problem at horizons N = 10, 20, 40
and 80 — variables 3N+2, dynamics rows 2N+2, block-banded Jacobian.
Only N changes between the families, so reading down the horizon column
isolates scale from every other property. It is a linear-quadratic MPC
on purpose: a convex QP, so all three solvers can take it, and its
sparsity is the block-banded structure real MPC codes exploit rather
than an artificial pattern.

The parameter is the initial state, walked around a circle in state
space. Constant ‖θ‖ keeps each step about as hard as the last (unlike a
closed-loop path, which converges and makes late steps trivial), while
the set of saturated controls rotates — so the active set moves at
every horizon in the same way, and the horizons stay comparable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

import numpy as np

from ..spec import Bounds, ParametricFamily

_INF = 1e20

#: Horizons swept. Variables run 32 → 242, dynamics rows 22 → 162.
MPC_HORIZONS = (10, 20, 40, 80)


class LinearMpcBase(ParametricFamily):
    """Linear-quadratic MPC of a damped oscillator, horizon ``_NH``.

    ``z = [x₀ … x_N, u₀ … u_{N−1}]`` with

        ``x⁺ = [[1, h], [−ah, 1 − bh]] x + [0, h]ᵀ u``

    minimizing a quadratic stage cost with a terminal weight, subject to
    the dynamics as equality rows, the initial state as the parameter,
    and ``|u| ≤ u_max`` — the bound that gives the problem an active set
    worth carrying.

    Everything is linear or quadratic, so the Jacobian and Hessian are
    constant and exact, and the instance is a convex QP.
    """

    quadratic = True
    n_steps = 20

    #: `c[:2] = X0 - theta` with `cl == cu == 0`, so stepping theta is
    #: exactly stepping these two rows' right-hand side -- the sIPOPT
    #: `deltas` convention the tangent predictor takes (pounce#608).
    pin_rows = (0, 1)

    _NH = 10  # overridden per horizon
    _H = 0.1
    _A = 1.0  # stiffness
    _B = 0.1  # damping
    _U_MAX = 0.5
    _Q = np.array([1.0, 0.1])
    _R = 0.05
    _QT = 10.0
    _RADIUS = 1.5
    _DPHI = 0.05  # radians per step, before the scale multiplier

    def __init__(self):
        self._theta = self._theta_at(0.0)

    # -- layout ----------------------------------------------------

    @property
    def n(self) -> int:
        return 2 * (self._NH + 1) + self._NH

    @property
    def m(self) -> int:
        return 2 + 2 * self._NH

    @property
    def _u_off(self) -> int:
        return 2 * (self._NH + 1)

    def _theta_at(self, phi: float) -> np.ndarray:
        return self._RADIUS * np.array([np.cos(phi), np.sin(phi)])

    def bounds(self) -> Bounds:
        lb = np.full(self.n, -10.0)
        ub = np.full(self.n, 10.0)
        lb[self._u_off :] = -self._U_MAX
        ub[self._u_off :] = self._U_MAX
        return Bounds(lb=lb, ub=ub, cl=np.zeros(self.m), cu=np.zeros(self.m))

    def cold_x0(self) -> np.ndarray:
        return np.zeros(self.n)

    def set_theta(self, theta: np.ndarray) -> None:
        self._theta = np.asarray(theta, dtype=float).ravel().copy()

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        return [
            self._theta_at(scale * self._DPHI * k) for k in range(self.n_steps)
        ]

    # -- functions -------------------------------------------------

    def _split(self, z):
        return z[: self._u_off].reshape(self._NH + 1, 2), z[self._u_off :]

    def objective(self, z):
        X, U = self._split(z)
        stage = float(np.sum(self._Q * X[:-1] ** 2))
        terminal = float(self._QT * np.sum(self._Q * X[-1] ** 2))
        return stage + terminal + float(self._R * (U @ U))

    def gradient(self, z):
        X, U = self._split(z)
        g = np.zeros(self.n)
        gx = g[: self._u_off].reshape(self._NH + 1, 2)
        gx[:-1] = 2.0 * self._Q * X[:-1]
        gx[-1] = 2.0 * self._QT * self._Q * X[-1]
        g[self._u_off :] = 2.0 * self._R * U
        return g

    def constraints(self, z):
        X, U = self._split(z)
        h, a, b = self._H, self._A, self._B
        c = np.empty(self.m)
        c[:2] = X[0] - self._theta
        x1, x2 = X[:-1, 0], X[:-1, 1]
        c[2::2] = X[1:, 0] - (x1 + h * x2)
        c[3::2] = X[1:, 1] - (x2 + h * (-a * x1 - b * x2 + U))
        return c

    def jacobian_dense(self, z):
        h, a, b = self._H, self._A, self._B
        j = np.zeros((self.m, self.n))
        j[0, 0] = 1.0
        j[1, 1] = 1.0
        for k in range(self._NH):
            r1, r2 = 2 + 2 * k, 3 + 2 * k
            i1, i2 = 2 * k, 2 * k + 1
            n1, n2 = 2 * (k + 1), 2 * (k + 1) + 1
            j[r1, n1] = 1.0
            j[r1, i1] = -1.0
            j[r1, i2] = -h
            j[r2, n2] = 1.0
            j[r2, i1] = a * h
            j[r2, i2] = -1.0 + b * h
            j[r2, self._u_off + k] = -h
        return j

    def hessian_dense(self, z, lagrange, obj_factor):
        # Constant and diagonal: the objective is a separable quadratic
        # and every constraint is linear.
        diag = np.zeros(self.n)
        dx = diag[: self._u_off].reshape(self._NH + 1, 2)
        dx[:-1] = 2.0 * self._Q
        dx[-1] = 2.0 * self._QT * self._Q
        diag[self._u_off :] = 2.0 * self._R
        return obj_factor * np.diag(diag)


    # -- sparse path -----------------------------------------------
    #
    # The dense methods above stay (the self-test finite-differences
    # them at the small horizons), but nothing calls them during a
    # solve: at N = 800 the dense Hessian alone would be 2402² doubles,
    # 46 MB, rebuilt at every iteration. The structure below is the
    # block-banded pattern an MPC transcription actually has —
    # 7 nonzeros per dynamics row pair, and a diagonal Hessian.

    def sparse_structure(self):
        jr, jc = [], []
        jr += [0, 1]
        jc += [0, 1]
        for k in range(self._NH):
            r1, r2 = 2 + 2 * k, 3 + 2 * k
            i1, i2 = 2 * k, 2 * k + 1
            n1, n2 = 2 * (k + 1), 2 * (k + 1) + 1
            u = self._u_off + k
            jr += [r1, r1, r1, r2, r2, r2, r2]
            jc += [n1, i1, i2, n2, i1, i2, u]
        idx = np.arange(self.n)
        return (
            np.array(jr, dtype=np.int64),
            np.array(jc, dtype=np.int64),
            idx.copy(),  # Hessian is diagonal
            idx.copy(),
        )

    def jacobian_values(self, z):
        h, a, b = self._H, self._A, self._B
        vals = np.empty(2 + 7 * self._NH)
        vals[0] = 1.0
        vals[1] = 1.0
        block = np.array([1.0, -1.0, -h, 1.0, a * h, -1.0 + b * h, -h])
        vals[2:] = np.tile(block, self._NH)
        return vals

    def hessian_values(self, z, lagrange, obj_factor):
        diag = np.zeros(self.n)
        dx = diag[: self._u_off].reshape(self._NH + 1, 2)
        dx[:-1] = 2.0 * self._Q
        dx[-1] = 2.0 * self._QT * self._Q
        diag[self._u_off :] = 2.0 * self._R
        return obj_factor * diag


def _horizon_family(nh: int, tier: str = "default") -> Type[LinearMpcBase]:
    return type(
        f"LinearMpc{nh}",
        (LinearMpcBase,),
        {
            "name": f"mpc_horizon_{nh}",
            "_NH": nh,
            "tier": tier,
            # The large tier walks fewer steps: one active-set solve at
            # N = 800 is seconds, and the per-step numbers are what
            # matter, not the path length.
            "n_steps": 20 if tier == "default" else 8,
            "tags": {
                "regime": "saturation",
                "channel": "rhs",
                "curvature": "convex",
                "horizon": str(nh),
            },
        },
    )


#: Opt-in tier: n = 602 → 2402, where the sparse KKT and the Schur
#: machinery start to be what the cost is made of. Not in the default
#: sweep because a single active-set solve here takes seconds.
MPC_LARGE_HORIZONS = (200, 400, 800)

HORIZON_FAMILIES: List[Type[LinearMpcBase]] = [
    _horizon_family(nh) for nh in MPC_HORIZONS
] + [_horizon_family(nh, tier="large") for nh in MPC_LARGE_HORIZONS]

HORIZON_BY_NAME: Dict[str, Type[LinearMpcBase]] = {
    f.name: f for f in HORIZON_FAMILIES
}
