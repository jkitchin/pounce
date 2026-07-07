"""One warm-start object for every solve path.

POUNCE's warm-start machinery is spread over several knobs whose
interplay is easy to get wrong (see ``docs/src/initialization.md``):
``warm_start_init_point=yes`` alone saves nothing, because the default
``mu_init`` (0.1) re-walks the barrier schedule and the default
``warm_start_bound_push``/``_frac`` (1e-3) shove an at-the-bound
solution back off its bounds. :class:`WarmStart` packages the whole
recipe into a single argument::

    x, info = prob.solve(x0=x0)                     # cold solve
    ws = pounce.WarmStart.from_info(x, info)        # capture everything
    x2, info2 = prob.solve(warm_start=ws)           # warm re-solve

    ws.save("state.npz")                            # ... and across processes
    ws = pounce.WarmStart.load("state.npz")

``warm_start=`` is accepted by :meth:`pounce.Problem.solve` and
:func:`pounce.minimize`. On the interior-point path it seeds the primal
and dual iterates and sets the five enabling options; on the active-set
SQP path (``algorithm=active-set-sqp``) it forwards the captured
working set, which is that path's warm-start payload.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np

from . import _pounce

__all__ = ["WarmStart"]


def _opt_array(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    a = np.asarray(v, dtype=float).ravel()
    return a if a.size else None


@dataclasses.dataclass
class WarmStart:
    """A captured solve state, usable as ``solve(warm_start=...)``.

    Attributes:
        x: Primal point (used as ``x0`` unless an explicit ``x0`` is
            passed to ``solve``).
        lagrange: Constraint multipliers (``info["mult_g"]``).
        zl / zu: Lower / upper bound multipliers.
        mu: Barrier parameter at capture (``info["mu"]``); seeds
            ``mu_init``. ``None`` or ``<= 0`` (e.g. captured from the
            SQP path) falls back to ``mu_init_fallback``.
        working_set: The SQP ``(bounds, constraints)`` working-set pair,
            forwarded on the ``algorithm=active-set-sqp`` path.
        bound_push: Value applied to the five ``warm_start_*_push`` /
            ``_frac`` options. The tight default (1e-9) keeps an
            at-the-bound solution essentially where it is; raise it if
            the next problem's solution may sit elsewhere.
        mu_init: Explicit ``mu_init`` override; default derives it from
            ``mu`` (clamped to ``[1e-9, 1e-1]``).
        mu_init_fallback: ``mu_init`` used when ``mu`` is unknown.
    """

    x: np.ndarray
    lagrange: Optional[np.ndarray] = None
    zl: Optional[np.ndarray] = None
    zu: Optional[np.ndarray] = None
    mu: Optional[float] = None
    working_set: Optional[Tuple[np.ndarray, np.ndarray]] = None
    bound_push: float = 1e-9
    mu_init: Optional[float] = None
    mu_init_fallback: float = 1e-6

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float).ravel()
        self.lagrange = _opt_array(self.lagrange)
        self.zl = _opt_array(self.zl)
        self.zu = _opt_array(self.zu)
        if self.working_set is not None:
            b, c = self.working_set
            self.working_set = (
                np.asarray(b, dtype=np.int8).ravel(),
                np.asarray(c, dtype=np.int8).ravel(),
            )

    # -- construction --------------------------------------------------

    @classmethod
    def from_info(cls, x, info, **overrides) -> "WarmStart":
        """Capture a warm start from a solve's ``(x, info)`` result.

        Works with :meth:`Problem.solve`'s ``info`` dict and with
        :func:`pounce.minimize`'s ``result.info``. Keyword overrides
        (e.g. ``bound_push=1e-6``) are forwarded to the constructor.
        """
        mu = info.get("mu")
        mu = float(mu) if mu is not None and float(mu) > 0.0 else None
        return cls(
            x=np.asarray(x, dtype=float),
            lagrange=info.get("mult_g"),
            zl=info.get("mult_x_L"),
            zu=info.get("mult_x_U"),
            mu=mu,
            working_set=info.get("working_set"),
            **overrides,
        )

    # -- persistence ----------------------------------------------------

    def save(self, path) -> None:
        """Serialize to a NumPy ``.npz`` archive (portable across
        processes; the file-based analog of the GAMS ``sqp_state_file``)."""
        payload = {"x": self.x, "_meta": np.array(
            [self.mu if self.mu is not None else np.nan,
             self.bound_push,
             self.mu_init if self.mu_init is not None else np.nan,
             self.mu_init_fallback]
        )}
        for key in ("lagrange", "zl", "zu"):
            v = getattr(self, key)
            if v is not None:
                payload[key] = v
        if self.working_set is not None:
            payload["ws_bounds"], payload["ws_constraints"] = self.working_set
        np.savez(path, **payload)

    @classmethod
    def load(cls, path) -> "WarmStart":
        """Inverse of :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            meta = data["_meta"]
            working_set = None
            if "ws_bounds" in data.files:
                working_set = (data["ws_bounds"], data["ws_constraints"])
            return cls(
                x=data["x"],
                lagrange=data["lagrange"] if "lagrange" in data.files else None,
                zl=data["zl"] if "zl" in data.files else None,
                zu=data["zu"] if "zu" in data.files else None,
                mu=None if np.isnan(meta[0]) else float(meta[0]),
                working_set=working_set,
                bound_push=float(meta[1]),
                mu_init=None if np.isnan(meta[2]) else float(meta[2]),
                mu_init_fallback=float(meta[3]),
            )

    # -- application ----------------------------------------------------

    def options(self) -> dict:
        """The enabling solver options this warm start implies.

        ``warm_start_init_point=yes`` makes the solver honor the seeds;
        ``mu_init`` skips the barrier walk-down; the tightened
        ``warm_start_*`` pushes keep an at-the-bound point where it is.
        All are ignored (harmlessly) on the SQP path.
        """
        if self.mu_init is not None:
            mu_init = self.mu_init
        elif self.mu is not None:
            mu_init = float(np.clip(self.mu, 1e-9, 1e-1))
        else:
            mu_init = self.mu_init_fallback
        p = self.bound_push
        return {
            "warm_start_init_point": "yes",
            "mu_init": mu_init,
            "warm_start_bound_push": p,
            "warm_start_bound_frac": p,
            "warm_start_slack_bound_push": p,
            "warm_start_slack_bound_frac": p,
            "warm_start_mult_bound_push": p,
        }

    def solve_kwargs(self) -> dict:
        """The seed keyword arguments for :meth:`Problem.solve`."""
        kw = {}
        if self.lagrange is not None:
            kw["lagrange"] = self.lagrange
        if self.zl is not None:
            kw["zl"] = self.zl
        if self.zu is not None:
            kw["zu"] = self.zu
        if self.working_set is not None:
            kw["working_set"] = self.working_set
        return kw


# ---------------------------------------------------------------------------
# Problem.solve(warm_start=...) — wrap the native method once at import.
# The pyo3 class cannot be subclassed, so the ergonomic entry point is a
# thin wrapper that translates a WarmStart into the native seed kwargs +
# enabling options and otherwise passes through unchanged.
# ---------------------------------------------------------------------------

_native_solve = _pounce.Problem.solve


def _solve_with_warm_start(
    self,
    x0=None,
    lagrange=None,
    zl=None,
    zu=None,
    working_set=None,
    warm_start: Optional[WarmStart] = None,
):
    if warm_start is None:
        if x0 is None:
            raise TypeError("Problem.solve() missing required argument: 'x0'")
        return _native_solve(
            self, x0=x0, lagrange=lagrange, zl=zl, zu=zu, working_set=working_set
        )
    ws = warm_start
    for k, v in ws.options().items():
        self.add_option(k, v)
    kw = ws.solve_kwargs()
    # Explicit per-call seeds win over the WarmStart's captured ones.
    for key, val in (
        ("lagrange", lagrange),
        ("zl", zl),
        ("zu", zu),
        ("working_set", working_set),
    ):
        if val is not None:
            kw[key] = val
    return _native_solve(self, x0=ws.x if x0 is None else x0, **kw)


_solve_with_warm_start.__name__ = "solve"
_solve_with_warm_start.__qualname__ = "Problem.solve"
_solve_with_warm_start.__doc__ = (_native_solve.__doc__ or "") + (
    "\n\n"
    "warm_start : pounce.WarmStart, optional\n"
    "    A captured solve state (see ``WarmStart.from_info``). Applies the\n"
    "    enabling options (``warm_start_init_point=yes``, ``mu_init``, the\n"
    "    tightened ``warm_start_*`` pushes — note they persist on this\n"
    "    Problem like any ``add_option``), seeds the duals, forwards the\n"
    "    SQP working set when present, and defaults ``x0`` to the captured\n"
    "    point. Explicit ``x0``/seed arguments override the captured ones.\n"
)

_pounce.Problem.solve = _solve_with_warm_start
