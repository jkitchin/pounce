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
    t_events: Any = None
    y_events: Any = None
    info: dict = field(default_factory=dict, repr=False)


def _project_output_points(res, prob, t0, y0, t_eval):
    """gh #216: Newton-polish the algebraic components of the requested output
    (``res['sol']`` and, when ``t_eval`` is given, ``res['y']``) onto the
    constraint manifold, in place. No-op when there are no algebraic variables
    or the algebraic rows are affine (exact under the cubic dense output)."""
    from . import _dae

    sol = res.get("sol")
    if sol is None:
        return
    y0 = np.asarray(y0, dtype=float)
    zp = np.zeros(y0.size)
    _, Fyp = prob.jacs(t0, y0, zp, prob.F(t0, y0, zp))
    alg_var = _dae._algebraic_mask(Fyp)
    if not alg_var.any():
        return
    alg_eq = _dae._algebraic_equation_mask(Fyp)
    if _dae._algebraic_rows_affine(prob, t0, y0, alg_eq):
        return
    psol = _dae.project_output(sol, prob, alg_eq, alg_var)
    res["sol"] = psol
    if t_eval is not None:
        res["y"] = psol(np.asarray(t_eval, dtype=float))


def _wrap_events_args(events, args):
    """Bind ``*args`` into each event ``g(t, y)`` (SciPy passes args to events
    too), preserving ``terminal`` / ``direction`` attributes."""
    evs = [events] if callable(events) else list(events)
    out = []
    for e in evs:
        w = (lambda _f: (lambda t, y: _f(t, y, *args)))(e)
        w.terminal = getattr(e, "terminal", False)
        w.direction = getattr(e, "direction", 0.0)
        out.append(w)
    return out[0] if callable(events) and len(out) == 1 else out


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
    consistent="project",
    project_output=False,
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
    events : callable, list of callables, or None
        SciPy-style event functions ``g(t, y)`` whose zero crossings are
        located (root-found on the dense output) and returned in
        ``res.t_events`` / ``res.y_events``. Each may carry ``terminal``
        (``bool`` or a positive ``int`` count — stops the integration with
        ``status=1``) and ``direction`` (``>0`` rising, ``<0`` falling, ``0``
        either) attributes.
    args : tuple or None
        Extra arguments passed to ``fun`` / ``jac`` / ``events``.
    mass : array (n, n), callable, or None
        Mass matrix ``M`` in ``M y' = f`` — either a constant array or a
        callable ``M(t, y)`` (``M(t, y, *args)`` with ``args``) for a
        state/time-dependent mass. A singular ``M`` makes this an index-1 DAE
        solve (a pounce extension beyond SciPy); a callable ``M`` is routed
        through the fully-implicit DAE engine (:func:`solve_dae`).
    consistent : {"project", "assume"}
        Only meaningful for a singular (DAE) ``mass``. ``"project"`` (default)
        projects ``y0`` onto the algebraic manifold ``0 = f`` before
        integrating — matching :func:`solve_dae` — so ``res.y[:, 0]`` is a
        point the model admits even when the given algebraic components are
        only a rough guess. ``"assume"`` trusts ``y0`` verbatim (the pre-0.x
        behavior); use it if you rely on ``res.y[:, 0]`` echoing your input and
        know it is already consistent. Ignored for a non-singular ``mass``
        (a plain ODE has no algebraic manifold to project onto).
    project_output : bool
        Only meaningful for a singular (DAE) ``mass``. When ``True``, the
        algebraic components of every *requested output* point (``res.sol(t)``
        and ``res.y`` at ``t_eval``) are Newton-polished onto ``0 = f`` before
        being returned. Radau IIA is stiffly accurate, so the constraint holds
        at the solver's own accepted steps, but the dense-output polynomial
        only *interpolates* it between them. For a **linear** conservation law
        (mass / atom / charge / site balance, ``sum(x) = 1``) the cubic
        satisfies it exactly and this is skipped automatically; it matters only
        for a **nonlinear** algebraic constraint whose interpolated residual is
        large enough to care about (gh #216). Off by default; does not change
        the trajectory, step sequence, or error control — only what you read.
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
        if callable(mass):
            _mass = mass
            mass = lambda t, y: _mass(t, y, *args)
        if events is not None:
            events = _wrap_events_args(events, args)

    if consistent not in ("project", "assume"):
        raise ValueError("consistent must be 'project' or 'assume'.")

    if callable(mass):
        # State/time-dependent mass M(t, y) y' = f(t, y): route through the
        # fully-implicit DAE engine as F(t, y, y') = M(t, y) y' - f(t, y).
        from . import _dae

        def _F(t, y, yp):
            return np.asarray(mass(t, y), dtype=float) @ yp - np.asarray(fun(t, y), dtype=float)

        prob = _dae._DaeProblem(_F, y0.size)
        if consistent == "project":
            y0, yp0 = _dae.consistent_initial_conditions(prob, t0, y0, None)
        else:
            # 'assume': trust y0; derive y'0 with y0 held fixed (min-norm solve
            # of M(t0, y0) y'0 = f(t0, y0)). Correct when y0 is consistent.
            M0 = np.asarray(mass(t0, y0), dtype=float)
            f0 = np.asarray(fun(t0, y0), dtype=float)
            yp0 = np.linalg.lstsq(M0, f0, rcond=None)[0]
        res = _dae.integrate_dae(
            _F, t0, t1, y0, yp0, rtol=rtol, atol=atol, first_step=first_step,
            max_step=max_step, t_eval=t_eval,
            dense_output=dense_output or t_eval is not None, events=events,
        )
        if project_output:
            _project_output_points(res, prob, t0, y0, t_eval)
    else:
        dae_prob = None
        if mass is not None and (consistent == "project" or project_output):
            # A singular constant mass is an index-1 DAE: 0 = f on the algebraic
            # rows. The fast mass path never projected the IC (gh #215) nor the
            # requested output points (gh #216); build the residual form once
            # and reuse it for both.
            from . import _dae

            M = np.asarray(mass, dtype=float)
            if M.ndim == 2 and _dae._algebraic_mask(M).any():
                dae_jac = None if jac is None else (
                    lambda t, y, yp: (-np.asarray(jac(t, y), dtype=float), M))
                _F = lambda t, y, yp: M @ yp - np.asarray(fun(t, y), dtype=float)
                dae_prob = _dae._DaeProblem(_F, y0.size, jac=dae_jac)
                if consistent == "project":
                    y0, _ = _dae.consistent_initial_conditions(
                        dae_prob, t0, y0, None)
        res = _radau.integrate(
            fun, t0, t1, y0, rtol=rtol, atol=atol, first_step=first_step,
            max_step=max_step, mass=mass, jac=jac, t_eval=t_eval,
            dense_output=dense_output or t_eval is not None, events=events,
        )
        if project_output and dae_prob is not None:
            _project_output_points(res, dae_prob, t0, y0, t_eval)
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
        t_events=res.get("t_events"),
        y_events=res.get("y_events"),
    )


def solve_dae(
    F,
    t_span,
    y0,
    yp0=None,
    *,
    consistent="project",
    project_output=False,
    rtol=1e-3,
    atol=1e-6,
    jac=None,
    first_step=None,
    max_step=np.inf,
    t_eval=None,
    dense_output=False,
    events=None,
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
    project_output : bool
        When ``True``, Newton-polish the algebraic components of every
        requested output point (``res.sol(t)`` and ``res.y`` at ``t_eval``)
        onto ``0 = F_alg`` — the constraint holds at the solver's accepted
        steps but the dense output only interpolates it between them (gh #216).
        Skipped automatically for affine constraints (exact under the cubic).
        Off by default; does not affect the trajectory or step control.

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
        if events is not None:
            events = _wrap_events_args(events, args)

    prob = _dae._DaeProblem(F, y0.size, jac=jac)
    if consistent == "project":
        y0, yp0 = _dae.consistent_initial_conditions(prob, t0, y0, yp0)
    elif consistent == "assume":
        if yp0 is None:
            raise ValueError("consistent='assume' requires an explicit yp0.")
    else:
        raise ValueError("consistent must be 'project' or 'assume'.")

    res = _dae.integrate_dae(
        F, t0, t1, y0, yp0, rtol=rtol, atol=atol, jac=jac,
        first_step=first_step, max_step=max_step, t_eval=t_eval,
        dense_output=dense_output or t_eval is not None, events=events,
    )
    if project_output:
        _project_output_points(res, prob, t0, y0, t_eval)
    return OdeResult(
        t=res["t"], y=res["y"], sol=res.get("sol"),
        nfev=res["nfev"], njev=res["njev"], nlu=res["nlu"],
        status=res["status"], message=res["message"], success=res["success"],
        nstep=res["nstep"], nrej=res["nrej"],
        t_events=res.get("t_events"), y_events=res.get("y_events"),
    )
