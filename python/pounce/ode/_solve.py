"""SciPy-signature ``solve_ivp`` on top of pounce's Radau IIA(5) integrator.

:func:`solve_ivp` matches the call signature and return shape of
:func:`scipy.integrate.solve_ivp` for the implicit ``Radau`` method, so
stiff-ODE code ports by changing only the import. Beyond SciPy it accepts a
**mass matrix** ``M`` (``M y' = f``), which turns the same solver into an
index-1 **DAE** integrator — something ``scipy.integrate.solve_ivp`` cannot
do.

Only ``method="Radau"`` is implemented (the implicit, stiff-capable method —
pounce's niche). Non-stiff explicit methods are better served by SciPy /
diffrax; this raises for them rather than silently substituting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import _radau
from .._result import ResultMixin


@dataclass
class OdeResult(ResultMixin):
    """Result of :func:`solve_ivp`, mirroring SciPy's ``Bunch``.

    ``t`` ``(n_points,)``, ``y`` ``(n, n_points)``, ``sol`` (a dense-output
    callable when ``dense_output=True``), ``nfev`` / ``njev`` / ``nlu``,
    ``status`` (0 = reached the end), ``message``, ``success``.
    """

    t: np.ndarray
    y: np.ndarray
    sol: Any
    nfev: int
    njev: int
    nlu: int
    status: int
    message: str
    success: bool
    nstep: int = 0
    nrej: int = 0
    info: dict = field(default_factory=dict, repr=False)


def mesh_initial_guess(fun_np, t_np, y0_np, n, m):
    """Cheap explicit trajectory on a fixed mesh, to seed collocation Newton.

    Shared by the differentiable ``pounce.jax.odeint`` / ``pounce.torch.odeint``
    frontends. Runs the adaptive Radau solver on the concrete RHS sampled at
    the mesh nodes; init-guess quality only affects Newton convergence, never
    the converged solution or its gradient, so any reasonable trajectory works
    (it falls back to holding the initial state if the explicit solve fails).
    """
    try:
        res = solve_ivp(
            fun_np, (float(t_np[0]), float(t_np[-1])), y0_np,
            method="Radau", t_eval=t_np, rtol=1e-3, atol=1e-6,
        )
        Y = np.asarray(res.y, dtype=np.float64)
        if Y.shape == (n, m) and np.all(np.isfinite(Y)):
            return Y
    except Exception:
        pass
    return np.broadcast_to(y0_np[:, None], (n, m)).copy()


def solve_ivp(
    fun,
    t_span,
    y0,
    method="Radau",
    t_eval=None,
    dense_output=False,
    events=None,
    vectorized=False,
    args=None,
    *,
    mass=None,
    rtol=1e-3,
    atol=1e-6,
    jac=None,
    first_step=None,
    max_step=np.inf,
    **options,
):
    """Solve an initial value problem ``M y' = f(t, y)`` with pounce.

    Drop-in for :func:`scipy.integrate.solve_ivp` with ``method="Radau"``.

    Parameters
    ----------
    fun : callable
        ``fun(t, y)`` (or ``fun(t, y, *args)`` when ``args`` is given)
        returning ``dy/dt`` as an ``(n,)`` array.
    t_span : 2-tuple
        ``(t0, tf)``. Integration may run forward or backward.
    y0 : array (n,)
        Initial state.
    method : str
        Only ``"Radau"`` (implicit, stiff/DAE-capable) is supported.
    t_eval : array or None
        Times at which to store the solution (interpolated from the dense
        output). If ``None``, the solver's own steps are returned.
    dense_output : bool
        If ``True``, attach a continuous solution ``res.sol(t)``.
    args : tuple or None
        Extra arguments passed to ``fun`` / ``jac``.
    mass : array (n, n) or None
        Constant mass matrix ``M`` (``M y' = f``). A singular ``M`` makes
        this an index-1 DAE solve — a pounce extension beyond SciPy.
    rtol, atol : float
        Relative / absolute error tolerances.
    jac : callable or None
        ``jac(t, y)`` returning ``df/dy``. Estimated by finite differences
        if omitted.
    first_step, max_step : float
        Initial / maximum step size.

    Returns
    -------
    OdeResult
    """
    if method != "Radau":
        raise NotImplementedError(
            f"pounce.ode.solve_ivp implements the stiff/DAE 'Radau' method; "
            f"got method={method!r}. For non-stiff explicit integration use "
            "scipy.integrate.solve_ivp or diffrax."
        )
    if events is not None:
        raise NotImplementedError("event detection is not yet supported.")
    # gh #165: don't silently no-op SciPy parameters.
    if vectorized:
        warnings.warn(
            "pounce.ode.solve_ivp ignores vectorized=True; the RHS and its "
            "finite-difference Jacobian are evaluated point-wise. Pass an "
            "analytic jac= to avoid the per-column RHS evaluations.",
            UserWarning, stacklevel=2,
        )
    if options:
        warnings.warn(
            f"pounce.ode.solve_ivp received unrecognized options "
            f"{sorted(options)} and ignored them.",
            UserWarning, stacklevel=2,
        )

    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=float).ravel()

    if args is not None:
        _fun = fun
        fun = lambda t, y: _fun(t, y, *args)
        if jac is not None:
            _jac = jac
            jac = lambda t, y: _jac(t, y, *args)

    res = _radau.integrate(
        fun, t0, t1, y0, rtol=rtol, atol=atol, first_step=first_step,
        max_step=max_step, mass=mass, jac=jac, t_eval=t_eval,
        dense_output=dense_output or t_eval is not None,
    )
    # Like SciPy's solve_ivp, a numerical failure (step underflow, step cap)
    # is reported as status < 0 / success = False with the partial trajectory
    # accumulated so far — never raised.
    return OdeResult(
        t=res["t"],
        y=res["y"],
        sol=res.get("sol"),
        nfev=res["nfev"],
        njev=res["njev"],
        nlu=res["nlu"],
        status=res["status"],
        message=res["message"],
        success=res["success"],
        nstep=res["nstep"],
        nrej=res["nrej"],
    )


def solve_dae(
    F,
    t_span,
    y0,
    yp0=None,
    *,
    consistent="project",
    rtol=1e-3,
    atol=1e-6,
    jac=None,
    first_step=None,
    max_step=np.inf,
    t_eval=None,
    dense_output=False,
    args=None,
):
    """Solve a fully-implicit index-1 DAE ``F(t, y, y') = 0`` with pounce.

    A pounce extension beyond SciPy (which has no fully-implicit DAE solver),
    using the same Radau IIA(5) collocation as :func:`solve_ivp` in residual
    form. Index-1 only.

    Parameters
    ----------
    F : callable
        ``F(t, y, yp)`` (or ``F(t, y, yp, *args)``) returning the ``(n,)``
        residual; a solution has ``F(t, y, y') == 0``.
    t_span, y0 : as in :func:`solve_ivp`.
    yp0 : array (n,) or None
        Guess for the initial derivative. With ``consistent="project"``
        (default) it is projected onto ``F(t0, y0, y'0) = 0`` (differential
        ``y`` and algebraic ``y'`` held fixed, IDA ``IDA_YA_YDP_INIT`` style),
        so an approximate guess (or ``None`` → zeros) is fine. With
        ``consistent="assume"`` it is used as given and must already satisfy
        ``F(t0, y0, yp0) == 0``.
    jac : callable or None
        ``jac(t, y, yp)`` returning ``(dF/dy, dF/dy')``; finite-differenced
        (``2n`` evals) if omitted.

    Returns
    -------
    OdeResult
    """
    from . import _dae

    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=float).ravel()
    if yp0 is not None:
        yp0 = np.asarray(yp0, dtype=float).ravel()

    if args is not None:
        _F = F
        F = lambda t, y, yp: _F(t, y, yp, *args)
        if jac is not None:
            _jac = jac
            jac = lambda t, y, yp: _jac(t, y, yp, *args)

    if consistent == "project":
        prob = _dae._DaeProblem(F, y0.size, jac=jac)
        y0, yp0 = _dae.consistent_initial_conditions(prob, t0, y0, yp0)
    elif consistent == "assume":
        if yp0 is None:
            raise ValueError("consistent='assume' requires an explicit yp0.")
    else:
        raise ValueError("consistent must be 'project' or 'assume'.")

    res = _dae.integrate_dae(
        F, t0, t1, y0, yp0, rtol=rtol, atol=atol, jac=jac,
        first_step=first_step, max_step=max_step, t_eval=t_eval,
        dense_output=dense_output or t_eval is not None,
    )
    return OdeResult(
        t=res["t"], y=res["y"], sol=res.get("sol"),
        nfev=res["nfev"], njev=res["njev"], nlu=res["nlu"],
        status=res["status"], message=res["message"], success=res["success"],
        nstep=res["nstep"], nrej=res["nrej"],
    )
