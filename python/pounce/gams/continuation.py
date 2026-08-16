"""Continuation over a GAMS parametric path (pounce#608).

pounce#608 asks for the continuation driver to reach the GAMS frontend.
For the **pip link** that is wiring rather than a second implementation:
:func:`pounce.gams.link.problem_from_view` already turns a GMO view into
an ordinary :class:`pounce.Problem` with the link's own option defaults,
and :class:`pounce.Continuation` drives ``Problem`` handles. So a GAMS
path traced in one process gets the whole driver -- the tangent
predictor included, when the parameter enters through pin-equality rows
-- and not the degraded zero-order transfer the CLI path is limited to.

The distinction that matters is **who owns the process**:

* ``option nlp = pounce;`` inside a GAMS loop is one link invocation per
  solve. GAMS owns the process, the KKT factor dies between points, and
  the best available transfer is whatever GAMS itself carries in the
  ``.l`` / ``.m`` levels -- a zero-order primal-dual warm start.
* Driving the views from Python, as :func:`trace` does, keeps every
  point in one process. The factor survives, so the sensitivity
  predictor is available.

What the native C link's state file carries
-------------------------------------------

``gams/gams_pounce.c`` has a ``sqp_state_file`` warm start the pip link
does not reproduce, and it is worth being precise about what it is: the
file holds the **discrete working set only** -- one byte of
``bound_status`` per variable and one of ``cons_status`` per constraint,
behind a magic string and a checksum over ``(n, m, bounds)``. It carries
no primal point, no multipliers and no barrier parameter, and it feeds
``IpoptSetWarmStartWorkingSet`` on the active-set SQP path.

For interior-point continuation that is strictly less than what the pip
link already has in memory, not more. And because the checksum is taken
over the bounds, any change of problem *shape* -- a horizon shift, a
remesh -- invalidates the file, which is exactly the case pounce#608's
transfer map exists to serve. So the state file is not a route to
carrying continuation state across GAMS solves, and this module does
not try to make it one.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

__all__ = ["trace"]


def trace(
    view_of_theta: Callable[[object], object],
    thetas: Sequence,
    *,
    pins: Optional[Sequence[int]] = None,
    options: Optional[dict] = None,
    max_iter: Optional[int] = None,
    max_wall_time: Optional[float] = None,
    transfer=None,
    monitor=None,
    bounds=None,
    x0=None,
    **kwargs,
):
    """Trace a GAMS parametric path through :class:`pounce.Continuation`.

    Args:
        view_of_theta: ``theta -> GmoView``. Whatever installing the
            parameter means on the GAMS side -- rewriting a scalar,
            re-reading a control file, regenerating the model -- happens
            here, which is what keeps the driver frontend-neutral.
        thetas: The parameter path.
        pins: 0-based indices of the pin-equality rows through which
            ``theta`` enters (``g_i(x) = theta_i``). Supplying them
            enables the tangent predictor; omitting them falls back to
            the zero-order warm transfer.
        options: POUNCE options, as a ``pounce.opt`` file would give
            them.
        max_iter, max_wall_time: The GAMS environment's ``gevIterLim`` /
            ``gevResLim``, applied as the link applies them.
        transfer, monitor, bounds, x0, **kwargs: Forwarded to
            :class:`pounce.Continuation` / :meth:`Continuation.run`
            unchanged -- including ``subdivide`` and ``subdivide_tol``.

    Returns:
        :class:`pounce.ContinuationTrace`, the same record the generic
        and Pyomo frontends return.
    """
    import pounce

    from .link import problem_from_view

    def update(theta):
        view = view_of_theta(theta)
        if view is None:
            raise TypeError(
                "gams.continuation.trace: view_of_theta(theta) must return "
                "the GmoView at theta; it returned None"
            )
        _gp, prob = problem_from_view(
            view, options=options, max_iter=max_iter,
            max_wall_time=max_wall_time,
        )
        return prob

    run_keys = ("subdivide", "subdivide_tol", "max_subdivisions", "counter")
    run_kw = {k: kwargs.pop(k) for k in run_keys if k in kwargs}

    if x0 is None and len(thetas):
        # The anchor starts from the GAMS model's own variable levels,
        # the way an ordinary `option nlp = pounce;` solve does. The
        # driver's generic default is zeros clipped into the box, which
        # for a model whose box excludes zero is not merely a worse
        # guess but a different basin: HS071 started from zeros converges
        # to a stationary point at 32.94 rather than the 17.01 the link
        # reaches from its declared levels.
        first = view_of_theta(thetas[0])
        init = getattr(first, "var_init", None)
        if callable(init):
            x0 = list(init())

    driver = pounce.Continuation(
        update, pins=pins, transfer=transfer, monitor=monitor,
        bounds=bounds, **kwargs,
    )
    return driver.run(thetas, x0=x0, **run_kw)
