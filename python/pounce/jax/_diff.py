"""Differentiate through the solver via implicit differentiation.

Setup. For a parametric NLP

    min_x  f(x, p)
    s.t.   g(x, p) = 0
           x_L <= x <= x_U,

the KKT conditions at the optimum ``x*(p)`` are

    ∇_x L(x*, λ*, p) = 0      with  L = f + λᵀ g    (active-set L)
              g(x*, p)  = 0

(plus complementarity on the bound multipliers — we treat the active
set as locally fixed; this is the standard implicit-function
assumption). Differentiating w.r.t. ``p`` and grouping into a 2×2 KKT
block,

    ⎡ H_xx   J_gxᵀ ⎤ ⎡ dx/dp ⎤     ⎡ ∂_p ∇_x L ⎤
    ⎣ J_gx     0   ⎦ ⎣ dλ/dp ⎦ = - ⎣ ∂_p g     ⎦.

For a cotangent ``v`` on ``x*``, the VJP w.r.t. ``p`` is computed by
solving the *transposed* KKT system, which is symmetric here:

    ⎡ H_xx   J_gxᵀ ⎤ ⎡ u_x ⎤   ⎡ v ⎤
    ⎣ J_gx     0   ⎦ ⎣ u_λ ⎦ = ⎣ 0 ⎦,

then    dL/dp = - u_xᵀ (∂_p ∇_x L) - u_λᵀ (∂_p g).

We assemble the dense KKT matrix from the JAX-AD Hessian and Jacobian
and solve it with ``jax.scipy.linalg.solve``. This keeps the
backward fully traced and itself differentiable (so you can take
second derivatives through the solver if you ever need to). For large
sparse problems the right move is to expose a Rust-side
sensitivity-solve via ``pounce-sensitivity``; that lands as a follow-up
once the JAX prototype is exercising the surface.

Bounds. Active variable bounds reduce dx/dp to zero on the active
coordinates. We detect activity from the optimizer's bound multipliers
``info['mult_x_L']`` / ``info['mult_x_U']`` (above
``active_tol``) and project the cotangent / right-hand-side onto the
inactive set before the KKT solve, then scatter back.

General inequalities. A two-sided constraint row ``cl[i] <= g_i(x)
<= cu[i]`` is *active* at the optimum iff (a) it is an equality
(``cl[i] == cu[i]``) or (b) ``|mult_g_i| > active_tol`` (binding at
one side). Slack inequality rows are dropped from the KKT block
(zeroed and identity-augmented on the multiplier diagonal) so the
sensitivity is taken over the active manifold only — including a
slack row as if it were ``g_i(x) = 0`` over-constrains ``dx*/dp``
and silently returns the wrong gradient (pounce#73).
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ._build import _JaxProblem
from .._pounce import Problem

from .._ad_common import ACTIVE_TOL as _ACTIVE_TOL  # single source of truth (DiffHandoff contract)


_BOUND_NAMES = ("lb", "ub", "cl", "cu")


def _split_bounds(lb, ub, cl, cu):
    """Split the four bound arrays into a static half and a traced half.

    A bound built with ``jnp.*`` *inside* a traced function is a tracer,
    not a concrete array — and that is exactly how the docs' examples
    write them (``lb=jnp.full(n, -10.0)`` in the body of a jitted loss).
    Closing such a value over and then dereferencing it inside a
    ``pure_callback`` host body, or inside a ``custom_vjp`` rule, leaks
    it out of the trace; JAX reports the leak as a side effect *in the
    user's function*, which is near-undiagnosable from the message
    (gh#740). Traced bounds are therefore threaded through explicitly —
    as ``pure_callback`` operands on the way to the host, and as a
    ``custom_vjp`` argument so nothing is closed over.

    Everything else — ``None``, Python scalars, lists, numpy arrays,
    already-concrete ``jax.Array`` values — stays static and is closed
    over verbatim, so the common case keeps its old behaviour exactly.

    Returns ``(static, dynamic)``: ``static`` is the 4-tuple with
    ``None`` punched into every traced slot, and ``dynamic`` maps those
    slot names to the arrays to thread through.
    """
    static = []
    dynamic = {}
    for name, b in zip(_BOUND_NAMES, (lb, ub, cl, cu)):
        if isinstance(b, jax.core.Tracer):
            static.append(None)
            dynamic[name] = jnp.asarray(b, dtype=jnp.float64)
        else:
            static.append(b)
    return tuple(static), dynamic


def _merge_bounds(static, dynamic):
    """Re-join :func:`_split_bounds`'s halves into ``(lb, ub, cl, cu)``.

    Used both inside ``pure_callback`` host bodies (where ``dynamic``
    holds the concrete numpy arrays JAX materialized) and inside the
    backward rules (where it holds the residual's traced copies).
    """
    merged = dict(zip(_BOUND_NAMES, static))
    merged.update(dynamic)
    return tuple(merged[name] for name in _BOUND_NAMES)


def _trip(b, mask):
    """NaN where ``mask`` holds, zero elsewhere, shaped like the bound ``b``."""
    zeros = jnp.zeros(jnp.shape(b), dtype=jnp.float64)
    if jnp.shape(mask) != jnp.shape(b):
        # A scalar (or otherwise broadcast) bound: the transpose of the
        # broadcast sums the coordinates, so one tripped coordinate is
        # enough to poison it.
        mask = jnp.any(mask)
    return jnp.where(mask, jnp.nan, zeros)


def _active_bound_masks(need, g, m, cl, cu, p, x_star, lam, mult_xL, mult_xU):
    """Per-bound mask: "the term we are dropping is nonzero here".

    A variable bound's side is read straight off its multiplier — the
    same ``_ACTIVE_TOL`` rule :func:`_kkt_implicit_backward` uses,
    except split by side, since ``dx*/dlb`` is nonzero only where the
    *lower* bound binds.

    A constraint row needs its side too, and the multiplier's magnitude
    does not carry it, so we evaluate ``g(x*, p)`` and ask which bound
    it is sitting on. An equality row (``cl == cu``) reads as both,
    which is right: it pins ``x*`` from either side. ``need`` holds the
    bounds actually being asked about, so that extra ``g`` evaluation
    only happens when a *constraint* bound was traced.
    """
    masks = {
        "lb": mult_xL > _ACTIVE_TOL,
        "ub": mult_xU > _ACTIVE_TOL,
    }
    if g is not None and m > 0 and ("cl" in need or "cu" in need):
        gx = g(x_star, p)
        cl_v = jnp.asarray(cl, dtype=jnp.float64)
        cu_v = jnp.asarray(cu, dtype=jnp.float64)
        active = (cl_v == cu_v) | (jnp.abs(lam) > _ACTIVE_TOL)
        masks["cl"] = active & ((gx - cl_v) <= (cu_v - gx))
        masks["cu"] = active & ((cu_v - gx) <= (gx - cl_v))
    else:
        masks["cl"] = jnp.zeros((m,), dtype=bool)
        masks["cu"] = jnp.zeros((m,), dtype=bool)
    return masks


def _bound_cotangents(dyn_bounds, masks):
    """Cotangent for the traced bounds: NaN exactly where the drop bites.

    Bounds are constants of the problem, so ``solve`` has no gradient to
    give back for them. Handing back a plain zero would be a silent lie
    in one case: a bound *derived from* ``p`` that is active at ``x*``
    genuinely moves ``x*``, so dropping its term yields a confidently
    wrong ``jax.grad`` — the failure mode pounce#73 was filed for.

    We cannot tell a p-derived bound from a constant one at trace time;
    under ``jit`` both are tracers. We do not have to. A cotangent
    handed back to a bound that *is* a constant transposes into nothing
    and is discarded, so NaN on the coordinates where the dropped term
    is nonzero is inert for the ordinary ``lb=jnp.full(n, -10.0)`` case
    (gh#740) and unmissable for the case that would otherwise be wrong.
    Slack coordinates keep an honest zero — there the dropped term
    really is zero, so a p-derived slack bound still differentiates.
    """
    return {k: _trip(v, masks[k]) for k, v in dyn_bounds.items()}


def _kkt_implicit_backward(f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, v):
    """Shared implicit-function-theorem backward for the NLP custom_vjp.

    This is the single, solver-agnostic core that every NLP backward in
    this module routes through (the plain ``solve``, ``solve_with_warm``,
    and the per-element ``vmap`` of ``vmap_solve_parallel``). It is the
    Python embodiment of the ``solve → DiffHandoff`` contract's backward
    (see ``dev-notes/diff-handoff-contract.md``): the active set is
    derived in exactly one place, and the KKT block is assembled and
    solved once.

    Given the converged primal ``x_star``, constraint multipliers
    ``lam``, bound multipliers ``mult_xL`` / ``mult_xU`` (the handoff
    duals), and the output cotangent ``v`` (w.r.t. ``x*``), returns the
    cotangent ``dL_dp`` w.r.t. the parameter ``p``.

    Active set (the rule lives *here*, not in each caller):
      * a variable bound is active when ``mult_xL`` or ``mult_xU`` exceeds
        ``_ACTIVE_TOL`` → that variable is pinned (``dx/dp = 0``);
      * a constraint row is active when it is an equality
        (``cl[i] == cu[i]``) or its multiplier exceeds ``_ACTIVE_TOL``;
        inactive (slack) rows drop out of the KKT block via the
        identity-augment trick (pounce#73).
    """
    active = (mult_xL > _ACTIVE_TOL) | (mult_xU > _ACTIVE_TOL)

    def lagrangian(x, p_):
        base = f(x, p_)
        if g is not None and m > 0:
            base = base + jnp.dot(lam, g(x, p_))
        return base

    H = jax.hessian(lagrangian, argnums=0)(x_star, p)
    # ∂_p ∇_x L — partial Jacobian of grad-L w.r.t. p.
    grad_L_of_p = lambda p_: jax.grad(lagrangian, argnums=0)(x_star, p_)
    dgradL_dp = jax.jacrev(grad_L_of_p)(p)

    if g is not None and m > 0:
        J = jax.jacrev(g, argnums=0)(x_star, p)
        dg_dp = jax.jacrev(lambda p_: g(x_star, p_))(p)
        cl_arr = jnp.asarray(cl, dtype=H.dtype)
        cu_arr = jnp.asarray(cu, dtype=H.dtype)
        is_equality = cl_arr == cu_arr
        cons_active = is_equality | (jnp.abs(lam) > _ACTIVE_TOL)
        cons_inactive = ~cons_active
    else:
        J = jnp.zeros((0, n))
        dg_dp = jnp.zeros((0,) + jnp.shape(p))
        cons_inactive = jnp.zeros((0,), dtype=bool)

    # Augment with identity on the active set: zero rows/cols belonging to
    # active vars, put 1 on their diagonal, zero the matching RHS rows.
    active_mat = jnp.diag(active.astype(H.dtype))
    H_eff = jnp.where(active[:, None] | active[None, :], 0.0, H) + active_mat
    J_eff = jnp.where(cons_inactive[:, None] | active[None, :], 0.0, J)
    v_eff = jnp.where(active, 0.0, v)

    # Assemble [[H, Jᵀ], [J, D]] u = [v; 0] with D = diag(cons_inactive),
    # so each slack row reads `1 · u_lam[i] = 0` and drops out.
    if m > 0:
        cons_inactive_diag = jnp.diag(cons_inactive.astype(H.dtype))
        top = jnp.concatenate([H_eff, J_eff.T], axis=1)
        bot = jnp.concatenate([J_eff, cons_inactive_diag], axis=1)
        K = jnp.concatenate([top, bot], axis=0)
        rhs = jnp.concatenate([v_eff, jnp.zeros(m, dtype=H.dtype)])
        u = jnp.linalg.solve(K, rhs)
        u_x, u_lam = u[:n], u[n:]
    else:
        u_x = jnp.linalg.solve(H_eff, v_eff)
        u_lam = jnp.zeros(0)

    # Contract with the parameter sensitivities. The minus sign comes from
    # rearranging dKKT/dp = 0 into the form above.
    dL_dp = -jnp.tensordot(u_x, dgradL_dp, axes=1)
    if m > 0:
        dL_dp = dL_dp - jnp.tensordot(u_lam, dg_dp, axes=1)
    return dL_dp


def _kkt_active_sets(m, cl, cu, lam, mult_xL, mult_xU):
    """``(active_vars (n,), inactive_rows (m,))`` — the same rule as
    :func:`_kkt_implicit_backward`: a variable is pinned when either bound
    multiplier exceeds ``_ACTIVE_TOL``; a row drops out unless it is an
    equality or its multiplier exceeds ``_ACTIVE_TOL``."""
    active = (mult_xL > _ACTIVE_TOL) | (mult_xU > _ACTIVE_TOL)
    if m > 0:
        cl_arr = jnp.asarray(cl, dtype=jnp.float64)
        cu_arr = jnp.asarray(cu, dtype=jnp.float64)
        cons_inactive = ~((cl_arr == cu_arr) | (jnp.abs(lam) > _ACTIVE_TOL))
        cons_inactive = jnp.broadcast_to(cons_inactive, (m,))
    else:
        cons_inactive = jnp.zeros((0,), dtype=bool)
    return active, cons_inactive


def _kkt_param_rhs(f, g, m, p, x_star, lam, p_dot):
    """``(∂_p ∇_x L · ṗ, ∂_p g · ṗ)`` at ``(x*, λ*)`` — one JVP each.

    Linear in ``p_dot``, written in plain JAX, so JAX can transpose it
    (reverse mode through a ``custom_jvp``) and differentiate it again
    (second order: ``x*`` and ``λ*`` are themselves outputs of the
    ``custom_jvp`` solve, so their tangents flow).
    """
    def lagrangian(x, p_):
        base = f(x, p_)
        if g is not None and m > 0:
            base = base + jnp.dot(lam, g(x, p_))
        return base

    def grad_L_of_p(p_):
        return jax.grad(lagrangian, argnums=0)(x_star, p_)

    r_x = jax.jvp(grad_L_of_p, (p,), (p_dot,))[1]
    if g is not None and m > 0:
        r_g = jax.jvp(lambda p_: g(x_star, p_), (p,), (p_dot,))[1]
    else:
        r_g = jnp.zeros((0,), dtype=r_x.dtype)
    return r_x, r_g


def _kkt_implicit_tangent(
    f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, p_dot,
):
    """Forward-mode twin of :func:`_kkt_implicit_backward` (gh#1010).

    Differentiating the KKT conditions along ``ṗ`` gives

        ⎡ H_xx   J_gxᵀ ⎤ ⎡ ẋ ⎤     ⎡ ∂_p ∇_x L · ṗ ⎤
        ⎣ J_gx     0   ⎦ ⎣ λ̇ ⎦ = - ⎣ ∂_p g · ṗ     ⎦,

    on the same active set and with the same identity-augment trick as
    the backward: pinned variables read ``ẋ_i = 0`` and slack rows read
    ``λ̇_i = 0``. Returns ``(ẋ (n,), λ̇ (m,))``.

    The whole rule is plain JAX and linear in ``ṗ`` — the matrix is
    symmetric, so its transpose is the backward's system — which is what
    lets a ``custom_jvp`` built on it serve ``jvp`` / ``jacfwd``,
    ``grad`` / ``jacrev`` (by transposition) and ``hessian`` (by
    differentiating the rule) from one definition.
    """
    active, cons_inactive = _kkt_active_sets(m, cl, cu, lam, mult_xL, mult_xU)

    def lagrangian(x, p_):
        base = f(x, p_)
        if g is not None and m > 0:
            base = base + jnp.dot(lam, g(x, p_))
        return base

    H = jax.hessian(lagrangian, argnums=0)(x_star, p)
    r_x, r_g = _kkt_param_rhs(f, g, m, p, x_star, lam, p_dot)

    H_eff = (jnp.where(active[:, None] | active[None, :], 0.0, H)
             + jnp.diag(active.astype(H.dtype)))
    r_x = jnp.where(active, 0.0, r_x)
    if m > 0:
        J = jax.jacrev(g, argnums=0)(x_star, p)
        J_eff = jnp.where(cons_inactive[:, None] | active[None, :], 0.0, J)
        top = jnp.concatenate([H_eff, J_eff.T], axis=1)
        bot = jnp.concatenate(
            [J_eff, jnp.diag(cons_inactive.astype(H.dtype))], axis=1
        )
        K = jnp.concatenate([top, bot], axis=0)
        r_g = jnp.where(cons_inactive, 0.0, r_g)
        w = jnp.linalg.solve(K, -jnp.concatenate([r_x, r_g]))
        return w[:n], w[n:]
    return jnp.linalg.solve(H_eff, -r_x), jnp.zeros((0,), dtype=H.dtype)


def _bound_tangent_poison(dyn_bounds, bounds_dot, g, m, cl, cu, p,
                          x_star, lam, mult_xL, mult_xU):
    """Forward-mode form of :func:`_bound_cotangents` (gh#1010).

    Returns a scalar that is NaN when a *perturbed* traced bound is
    active at ``x*`` and zero otherwise, as ``Σ ḃ · trip(b)``: linear in
    the bound tangents, so it transposes to exactly the NaN-where-active
    cotangent the ``custom_vjp`` handed back. Bounds JAX knows to be
    constant arrive as ``SymbolicZero`` and contribute nothing, which is
    what keeps the ordinary ``lb=jnp.full(n, -10.0)`` case clean.
    """
    from jax.custom_derivatives import SymbolicZero

    live = {
        k: t for k, t in bounds_dot.items() if not isinstance(t, SymbolicZero)
    }
    if not live:
        return None
    masks = _active_bound_masks(
        live, g, m, cl, cu, p, x_star, lam, mult_xL, mult_xU
    )
    trips = _bound_cotangents({k: dyn_bounds[k] for k in live}, masks)
    return sum(jnp.sum(live[k] * trips[k]) for k in live)


# Ipopt-style statuses that count as a converged forward solve:
# Solve_Succeeded, Solved_To_Acceptable_Level, Feasible_Point_Found.
_OK_STATUSES = (0, 1, 6)


def _check_forward_status(info, on_failure) -> None:
    """Surface a non-converged forward solve (gh#985).

    ``on_failure``: ``"warn"`` (default; ``RuntimeWarning``), ``"raise"``
    (``RuntimeError``) or ``"ignore"``. Never changes the return value, so
    existing callers keep working. The check runs in the host callback,
    so ``"raise"`` surfaces as ``jax.errors.JaxRuntimeError`` (a
    ``RuntimeError`` subclass), with JAX's callback traceback on stderr.
    """
    if on_failure == "ignore":
        return
    status = int(info["status"])
    if status in _OK_STATUSES:
        return
    msg = (
        "pounce.jax forward solve did not converge: "
        f"{info.get('status_msg', status)} (status {status}); the returned x* "
        "is not an optimum and its gradients are meaningless"
    )
    if on_failure == "raise":
        raise RuntimeError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _report_batch_failures(results, on_failure) -> None:
    """One aggregated warning / error for a batch (gh#985 review).

    ``results`` is a per-element sequence of ``(status, status_msg)``.  A
    per-element ``warnings.warn`` from the same line collapses to a single
    message under Python's default filter, so five failing elements used to
    read as one; this names the count and the failing indices instead.
    """
    if on_failure == "ignore":
        return
    bad = [
        (i, int(s), msg) for i, (s, msg) in enumerate(results)
        if int(s) not in _OK_STATUSES
    ]
    if not bad:
        return
    shown = ", ".join(f"{i} ({msg or s})" for i, s, msg in bad[:10])
    more = f", ... and {len(bad) - 10} more" if len(bad) > 10 else ""
    msg = (
        f"pounce.jax batched forward solve: {len(bad)} of {len(results)} "
        f"element(s) did not converge -- index (status): {shown}{more}; "
        "those rows of x* are not optima and their gradients are meaningless"
    )
    if on_failure == "raise":
        raise RuntimeError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _batch_failure_sink(on_failure):
    """``(sink, flush)`` for a sequential (``lax.map``) batch under
    ``on_failure="warn"``: each element's host callback calls ``sink(info)``
    instead of warning, and ``flush(x_batch)`` -- attached with
    ``jax.debug.callback`` on the batch output, so it runs after every
    element -- emits the single aggregated warning.  ``"raise"`` keeps the
    per-element raise (the first failure stops the batch); ``"ignore"``
    needs nothing.  Returns ``(None, None)`` then."""
    if on_failure != "warn":
        return None, None
    results = []

    def sink(info):
        results.append((int(info["status"]), info.get("status_msg")))

    def flush(_x):
        items = list(results)
        results.clear()
        _report_batch_failures(items, "warn")

    return sink, flush


def _check_on_failure(on_failure) -> None:
    if on_failure not in ("warn", "raise", "ignore"):
        raise ValueError(
            f"on_failure must be 'warn', 'raise' or 'ignore', got {on_failure!r}"
        )


def _solve_once(
    f: Callable,
    g: Callable | None,
    p: jnp.ndarray,
    x0: jnp.ndarray,
    n: int,
    m: int,
    lb,
    ub,
    cl,
    cu,
    options: dict | None,
    extra: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Forward solve. ``p`` is closed over by ``f`` / ``g`` via partial."""

    def f_of_x(x):
        return f(x, p)

    if g is not None:
        def g_of_x(x):
            return g(x, p)
    else:
        g_of_x = None

    extra = extra or {}
    obj = _JaxProblem(
        f=f_of_x, g=g_of_x, n=n, m=m,
        jac_pattern=extra.get("jac_pattern"),
        hess_pattern=extra.get("hess_pattern"),
        lb=lb, ub=ub, x0=x0,
    )
    problem = Problem(n=n, m=m, problem_obj=obj, lb=lb, ub=ub, cl=cl, cu=cu)
    if options:
        for k, v in options.items():
            problem.add_option(k, v)
    x_np, info = problem.solve(x0=np.asarray(x0))
    sink = extra.get("_sink")
    if sink is not None:
        sink(info)  # a batch reports its failures once, aggregated
    else:
        _check_forward_status(info, extra.get("on_failure", "warn"))
    return np.asarray(x_np), info


def _make_solve_custom_jvp(
    f: Callable,
    g: Callable | None,
    n: int,
    m: int,
    static_bounds,
    options: dict | None,
    extra: dict | None = None,
):
    """The solve as a ``jax.custom_jvp`` (gh#1010).

    It used to be a ``custom_vjp``, which JAX refuses to push a tangent
    through: ``jvp`` / ``jacfwd`` / ``hessian`` of anything calling
    :func:`solve` raised. The tangent rule here is
    :func:`_kkt_implicit_tangent`, plain JAX and linear in ``ṗ``, so
    one definition serves every transform — forward mode directly,
    reverse mode by transposing the rule (the KKT matrix is symmetric,
    so the transposed solve is the old backward's), and higher orders
    by differentiating the rule. That last step needs ``λ*`` to carry a
    tangent, because the Hessian of the Lagrangian depends on it, so
    the primitive returns ``(x*, λ*, z_L, z_U)`` and the rule re-enters
    it for the primal (the same construction as :func:`solve_bvp`'s
    ``differentiable_to_second_order``). The bound multipliers only
    select the active set, which is locally constant, so their tangent
    is zero.
    """
    @jax.custom_jvp
    def solve_full(p, x0, dyn_bounds):
        x_star, info = _pure_callback_solve(
            f, g, p, x0, n, m, static_bounds, dyn_bounds, options, extra
        )
        return x_star, info["mult_g"], info["mult_x_L"], info["mult_x_U"]

    def solve_jvp(primals, tangents):
        from jax.custom_derivatives import SymbolicZero

        p, x0, dyn_bounds = primals
        p_dot, _x0_dot, bounds_dot = tangents
        out = solve_full(p, x0, dyn_bounds)
        x_star, lam, mult_xL, mult_xU = out
        _lb, _ub, cl, cu = _merge_bounds(static_bounds, dyn_bounds)
        # x0 has no sensitivity through x* (the solver is deterministic
        # at the optimum), so its tangent is dropped.
        if isinstance(p_dot, SymbolicZero):
            dx = jnp.zeros((n,), dtype=jnp.float64)
            dlam = jnp.zeros((m,), dtype=jnp.float64)
        else:
            dx, dlam = _kkt_implicit_tangent(
                f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, p_dot,
            )
        # The bounds are held constant — see :func:`_bound_cotangents`.
        poison = _bound_tangent_poison(
            dyn_bounds, bounds_dot, g, m, cl, cu, p, x_star, lam,
            mult_xL, mult_xU,
        )
        if poison is not None:
            dx = dx + poison
            dlam = dlam + poison
        return out, (dx, dlam, jnp.zeros_like(mult_xL), jnp.zeros_like(mult_xU))

    solve_full.defjvp(solve_jvp, symbolic_zeros=True)

    def solve_fn(p, x0, dyn_bounds):
        return solve_full(p, x0, dyn_bounds)[0]

    return solve_fn


def _pure_callback_solve(
    f, g, p, x0, n, m, static_bounds, dyn_bounds, options, extra=None,
):
    """JAX pure_callback wrapper around :func:`_solve_once`.

    Returns ``(x_star, info)`` where ``info`` is a dict of arrays.
    The shapes are static so JAX can trace through cleanly.
    """
    result_shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        {
            "obj_val": jax.ShapeDtypeStruct((), jnp.float64),
            "status": jax.ShapeDtypeStruct((), jnp.int32),
            "iter_count": jax.ShapeDtypeStruct((), jnp.int32),
            "g": jax.ShapeDtypeStruct((m,), jnp.float64),
            "mult_g": jax.ShapeDtypeStruct((m,), jnp.float64),
            "mult_x_L": jax.ShapeDtypeStruct((n,), jnp.float64),
            "mult_x_U": jax.ShapeDtypeStruct((n,), jnp.float64),
        },
    )

    def host_call(p_h, x0_h, bounds_h):
        lb, ub, cl, cu = _merge_bounds(static_bounds, bounds_h)
        x_np, info = _solve_once(
            f=f, g=g,
            p=jnp.asarray(p_h),
            x0=jnp.asarray(x0_h),
            n=n, m=m, lb=lb, ub=ub, cl=cl, cu=cu,
            options=options,
            extra=extra,
        )
        info_out = {
            "obj_val": np.float64(info["obj_val"]),
            "status": np.int32(info["status"]),
            "iter_count": np.int32(info["iter_count"]),
            "g": np.asarray(info["g"], dtype=np.float64),
            "mult_g": np.asarray(info["mult_g"], dtype=np.float64),
            "mult_x_L": np.asarray(info["mult_x_L"], dtype=np.float64),
            "mult_x_U": np.asarray(info["mult_x_U"], dtype=np.float64),
        }
        return np.asarray(x_np, dtype=np.float64), info_out

    # ``sequential``: under ``jax.vmap`` each batch element is its own
    # host solve (gh#1010). The solver is stateful and single-threaded,
    # so a looped callback is the only batching rule that is correct.
    return jax.pure_callback(
        host_call, result_shapes, p, x0, dyn_bounds, vmap_method="sequential",
    )


def _check_array_p(p, who: str):
    """``p`` must be one array (or scalar); a pytree is not supported.

    The host callback converts ``p`` with ``asarray`` and the implicit
    backward contracts ``jacrev`` over it with ``tensordot``, so a tuple or
    dict otherwise fails deep inside a ``pure_callback`` with an unrelated
    shape error. Say what to do instead.
    """
    leaves, treedef = jax.tree_util.tree_flatten(p)
    if len(leaves) != 1 or treedef.num_nodes != 1:
        raise TypeError(
            f"pounce.jax.{who}: `p` must be a single array or scalar, got a "
            f"{type(p).__name__} pytree. Flatten it first, e.g. "
            "`q, unravel = jax.flatten_util.ravel_pytree(p)`, solve with `q`, "
            "and call `unravel(q)` inside `f` / `g`."
        )


def solve(
    p,
    *,
    f: Callable,
    g: Callable | None = None,
    x0,
    n: int,
    m: int = 0,
    lb=None,
    ub=None,
    cl=None,
    cu=None,
    options: dict | None = None,
    jac_pattern=None,
    hess_pattern=None,
    on_failure: str = "warn",
):
    """Parametric solve. ``x* = solve(p, f=..., g=..., x0=..., ...)``.

    Differentiable w.r.t. ``p`` via the implicit-function rule on the
    KKT system at ``x*(p)``. Not differentiable w.r.t. ``x0`` or the
    bounds. ``x0``'s cotangent is an exact zero — the solver is
    deterministic at the optimum. The bounds are treated as constants
    of the problem, so building one out of ``p`` will *not* produce the
    bound-sensitivity term; where such a bound is *active* at ``x*``
    that term is a real part of ``dL/dp``, and rather than silently
    drop it ``solve`` returns NaN for those coordinates. The gradient
    goes visibly bad instead of quietly wrong. Fold the bound into a
    row of ``g`` if you need to differentiate through it.

    The bounds may be built with ``jnp.*`` inside a traced function —
    they are threaded through the callback and the ``custom_vjp``
    rather than closed over, so ``jax.jit`` works either way (gh#740).
    A constant bound is unaffected by the NaN above: its cotangent
    transposes into nothing and is discarded.

    ``f`` and ``g`` must take ``(x, p)`` and be JAX-traceable.

    The solve is a ``jax.custom_jvp`` (gh#1010), so it composes with every
    JAX transform: ``jvp`` / ``jacfwd``, ``grad`` / ``jacrev`` (by
    transposing the tangent rule), ``hessian`` and higher orders, ``jit``,
    and ``vmap`` (one host solve per batch element, in order). In forward
    mode the NaN rule above is applied per bound *array*: a traced bound
    array with a ``p``-dependent coordinate makes the whole tangent NaN
    if any of its coordinates is active.

    ``jac_pattern`` / ``hess_pattern`` : ``(rows, cols)`` or ``None``
        Known sparsity patterns, as in :func:`pounce.jax.from_jax`
        (Jacobian ``(m, n)``; lower triangle of the Lagrangian Hessian).
        Supplying one skips detection; it must be a superset of the true
        structure. Without them the pattern is read off the jaxpr of
        ``f`` / ``g`` (structural, value-independent; see
        :func:`pounce.jax.from_jax`'s ``pattern_detection``), falling back
        per matrix to the union of probes at ``x0``, inside ``[lb, ub]``
        and at standard-normal points for a model the jaxpr analysis
        cannot bound (gh#985).

    ``on_failure`` : ``"warn"`` (default), ``"raise"`` or ``"ignore"``
        What to do when the forward solve does not converge (anything
        other than solved / acceptable / feasible point). The return
        value is unchanged; ``"warn"`` emits a ``RuntimeWarning`` and
        ``"raise"`` a ``RuntimeError`` from the host callback. Because
        that error crosses a ``jax.pure_callback`` it reaches the caller
        as ``jax.errors.JaxRuntimeError`` (a ``RuntimeError`` subclass
        whose message carries the original text), raised when the result
        is materialised (``block_until_ready`` / ``np.asarray``), and JAX
        also logs the callback's traceback to stderr. Catch
        ``RuntimeError``, not a narrower type.
    """
    _check_array_p(p, "solve")
    static_bounds, dyn_bounds = _split_bounds(lb, ub, cl, cu)
    _check_on_failure(on_failure)
    extra = dict(jac_pattern=jac_pattern, hess_pattern=hess_pattern,
                 on_failure=on_failure)
    fn = _make_solve_custom_jvp(f, g, n, m, static_bounds, options, extra)
    return fn(p, x0, dyn_bounds)


def _solve_once_warm(
    f: Callable,
    g: Callable | None,
    p: jnp.ndarray,
    x0: jnp.ndarray,
    n: int,
    m: int,
    lb,
    ub,
    cl,
    cu,
    options: dict | None,
    lam_warm: np.ndarray,
    zL_warm: np.ndarray,
    zU_warm: np.ndarray,
    mu_warm: float,
    extra: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """Forward solve with user-supplied dual (and optional μ) warm-start.

    ``mu_warm`` seeds the interior-point barrier parameter for the
    corrector in predictor–corrector path following (pounce#86): when
    finite it is fed to both ``mu_init`` (the monotone-strategy initial
    μ) and ``warm_start_target_mu`` (the warm-start initializer's
    target). ``NaN`` means "no μ seed" — fall back to the solver's
    default initial barrier.
    """

    def f_of_x(x):
        return f(x, p)

    if g is not None:
        def g_of_x(x):
            return g(x, p)
    else:
        g_of_x = None

    extra = extra or {}
    obj = _JaxProblem(
        f=f_of_x, g=g_of_x, n=n, m=m,
        jac_pattern=extra.get("jac_pattern"),
        hess_pattern=extra.get("hess_pattern"),
        lb=lb, ub=ub, x0=x0,
    )
    problem = Problem(n=n, m=m, problem_obj=obj, lb=lb, ub=ub, cl=cl, cu=cu)
    merged = dict(options or {})
    merged.setdefault("warm_start_init_point", "yes")
    if np.isfinite(mu_warm):
        # Seed the barrier from the previous solve's converged μ so the
        # corrector resumes near the central path instead of re-walking
        # the homotopy from the default initial μ (pounce#86).
        merged.setdefault("mu_init", float(mu_warm))
        merged.setdefault("warm_start_target_mu", float(mu_warm))
    for k, v in merged.items():
        problem.add_option(k, v)
    x_np, info = problem.solve(
        x0=np.asarray(x0),
        lagrange=np.asarray(lam_warm),
        zl=np.asarray(zL_warm),
        zu=np.asarray(zU_warm),
    )
    _check_forward_status(info, extra.get("on_failure", "warn"))
    return (
        np.asarray(x_np, dtype=np.float64),
        np.asarray(info["mult_g"], dtype=np.float64),
        np.asarray(info["mult_x_L"], dtype=np.float64),
        np.asarray(info["mult_x_U"], dtype=np.float64),
        float(info["mu"]),
        info,
    )


def _pure_callback_warm_solve(
    f, g, p, x0, n, m, static_bounds, dyn_bounds, options,
    lam_warm, zL_warm, zU_warm, mu_warm, extra=None,
):
    """Pure-callback wrapper around :func:`_solve_once_warm`.

    Returns ``(x*, lam_out, zL_out, zU_out, mu_out)`` — the arrays the
    bwd needs (lam_out, the bound multipliers), the warm-state triple
    the user threads into the next call, and the converged barrier
    parameter ``mu_out`` (pounce#86).
    """
    result_shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((m,), jnp.float64),
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.float64),
    )

    def host_call(p_h, x0_h, lam_h, zL_h, zU_h, mu_h, bounds_h):
        lb, ub, cl, cu = _merge_bounds(static_bounds, bounds_h)
        x_np, lam_out, zL_out, zU_out, mu_out, _info = _solve_once_warm(
            f=f, g=g,
            p=jnp.asarray(p_h),
            x0=jnp.asarray(x0_h),
            n=n, m=m, lb=lb, ub=ub, cl=cl, cu=cu,
            options=options,
            lam_warm=lam_h, zL_warm=zL_h, zU_warm=zU_h,
            mu_warm=float(np.asarray(mu_h)),
            extra=extra,
        )
        return x_np, lam_out, zL_out, zU_out, np.float64(mu_out)

    # ``sequential``: under ``jax.vmap`` each batch element is its own
    # host solve, as in :func:`_pure_callback_solve`. The backward is plain
    # JAX over the per-lane residuals, so it batches without help.
    return jax.pure_callback(
        host_call, result_shapes, p, x0, lam_warm, zL_warm, zU_warm, mu_warm,
        dyn_bounds, vmap_method="sequential",
    )


def _make_solve_with_warm_custom_vjp(
    f: Callable,
    g: Callable | None,
    n: int,
    m: int,
    static_bounds,
    options: dict | None,
    extra: dict | None = None,
):
    @jax.custom_vjp
    def solve_fn(p, x0, lam_warm, zL_warm, zU_warm, mu_warm, dyn_bounds):
        x_star, lam_out, zL_out, zU_out, mu_out = _pure_callback_warm_solve(
            f, g, p, x0, n, m, static_bounds, dyn_bounds, options,
            lam_warm, zL_warm, zU_warm, mu_warm, extra,
        )
        return x_star, lam_out, zL_out, zU_out, mu_out

    def fwd(p, x0, lam_warm, zL_warm, zU_warm, mu_warm, dyn_bounds):
        x_star, lam_out, zL_out, zU_out, mu_out = _pure_callback_warm_solve(
            f, g, p, x0, n, m, static_bounds, dyn_bounds, options,
            lam_warm, zL_warm, zU_warm, mu_warm, extra,
        )
        return (
            (x_star, lam_out, zL_out, zU_out, mu_out),
            (p, x_star, lam_out, zL_out, zU_out, dyn_bounds),
        )

    def bwd(residuals, cotangents):
        p, x_star, lam, mult_xL, mult_xU, dyn_bounds = residuals
        _lb, _ub, cl, cu = _merge_bounds(static_bounds, dyn_bounds)
        # Only the x* cotangent contributes a gradient w.r.t. p.
        # Cotangents on (lam_out, zL_out, zU_out, mu_out) are dropped —
        # same pattern existing `solve` uses for x0: warm dual / barrier
        # outputs don't carry differentiable info back to p in the
        # implicit rule (they're consequences of the active set and the
        # barrier homotopy, not inputs).
        v = cotangents[0]

        dL_dp = _kkt_implicit_backward(
            f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, v
        )

        return (
            dL_dp,
            jnp.zeros((n,), dtype=jnp.float64),
            jnp.zeros((m,), dtype=jnp.float64),
            jnp.zeros((n,), dtype=jnp.float64),
            jnp.zeros((n,), dtype=jnp.float64),
            jnp.zeros((), dtype=jnp.float64),
            _bound_cotangents(
                dyn_bounds,
                _active_bound_masks(
                    dyn_bounds, g, m, cl, cu, p, x_star, lam, mult_xL, mult_xU
                ),
            ),
        )

    solve_fn.defvjp(fwd, bwd)
    return solve_fn


def solve_with_warm(
    p,
    *,
    f: Callable,
    g: Callable | None = None,
    x0,
    n: int,
    m: int = 0,
    lb=None,
    ub=None,
    cl=None,
    cu=None,
    options: dict | None = None,
    warm_start: tuple | None = None,
    jac_pattern=None,
    hess_pattern=None,
    on_failure: str = "warn",
):
    """Parametric solve that consumes and returns dual warm-state.

    Like :func:`solve`, but:

    * ``warm_start=(lam, zL, zU)`` seeds the solver's dual variables
      via IPOPT's ``warm_start_init_point=yes`` machinery. Pass
      ``None`` to start from zeros (still warm-starts the option,
      but with no informative duals — useful for the *first* call
      in a sequence where you want a uniform code path).
    * ``warm_start=(lam, zL, zU, mu)`` *additionally* seeds the
      interior-point barrier parameter μ (pounce#86): the corrector
      resumes near the central path at the supplied μ rather than
      re-walking the barrier homotopy from the default initial μ.
      Pass ``mu=None`` inside the 4-tuple to skip the μ seed but still
      receive the converged μ on output (report-only).
    * Returns ``(x*, (lam_out, zL_out, zU_out))`` for a 3-tuple /
      ``None`` warm-start, or ``(x*, (lam_out, zL_out, zU_out,
      mu_out))`` for a 4-tuple warm-start — the returned warm-state
      arity matches the input, so threading μ in gives μ back out.

    ``jac_pattern`` / ``hess_pattern`` / ``on_failure`` behave as in
    :func:`solve` (gh#985): explicit sparsity patterns skip detection, and
    a non-converged forward solve warns (default), raises or is ignored.

    The forward call is differentiable w.r.t. ``p`` only — cotangents
    on the warm-output duals/μ and the warm-input duals/μ are dropped
    (zero), matching how :func:`solve` handles ``x0``. This is the
    implicit-function-theorem fix point: at the optimum the duals and
    the barrier are functions of ``p`` and the active set, not
    independent inputs feeding into ``dx*/dp``.

    Typical use (predictor–corrector corrector, with μ threading)::

        x0, lam, zL, zU, mu = init_state(...)
        for p_k in trajectory:
            x_star, (lam, zL, zU, mu) = solve_with_warm(
                p_k, f=f, g=g, x0=x0, n=n, m=m,
                lb=lb, ub=ub, cl=cl, cu=cu,
                warm_start=(lam, zL, zU, mu),
            )
            x0 = x_star  # primal warm-start for free
    """
    _check_on_failure(on_failure)
    extra = dict(jac_pattern=jac_pattern, hess_pattern=hess_pattern,
                 on_failure=on_failure)
    want_mu = warm_start is not None and len(warm_start) == 4
    if warm_start is None:
        lam_warm = jnp.zeros(m, dtype=jnp.float64)
        zL_warm = jnp.zeros(n, dtype=jnp.float64)
        zU_warm = jnp.zeros(n, dtype=jnp.float64)
        mu_warm = jnp.asarray(jnp.nan, dtype=jnp.float64)
    else:
        if want_mu:
            lam_warm, zL_warm, zU_warm, mu_seed = warm_start
        else:
            lam_warm, zL_warm, zU_warm = warm_start
            mu_seed = None
        lam_warm = jnp.asarray(lam_warm, dtype=jnp.float64)
        zL_warm = jnp.asarray(zL_warm, dtype=jnp.float64)
        zU_warm = jnp.asarray(zU_warm, dtype=jnp.float64)
        # A ``None`` μ inside a 4-tuple means "report μ out, don't seed
        # it in" — pass NaN to the host so it skips the mu_init seed.
        mu_warm = jnp.asarray(
            jnp.nan if mu_seed is None else mu_seed, dtype=jnp.float64
        )

    _check_array_p(p, "solve_with_warm")
    static_bounds, dyn_bounds = _split_bounds(lb, ub, cl, cu)
    fn = _make_solve_with_warm_custom_vjp(
        f, g, n, m, static_bounds, options, extra
    )
    x_star, lam_out, zL_out, zU_out, mu_out = fn(
        p, x0, lam_warm, zL_warm, zU_warm, mu_warm, dyn_bounds,
    )
    if want_mu:
        return x_star, (lam_out, zL_out, zU_out, mu_out)
    return x_star, (lam_out, zL_out, zU_out)


def vmap_solve(
    p_batch,
    *,
    f: Callable,
    g: Callable | None = None,
    x0,
    n: int,
    m: int = 0,
    lb=None,
    ub=None,
    cl=None,
    cu=None,
    options: dict | None = None,
    jac_pattern=None,
    hess_pattern=None,
    on_failure: str = "warn",
):
    """Batched solve over the leading axis of ``p_batch``.

    Runs the solves one at a time with ``jax.lax.map``, preserving
    differentiability via :func:`solve`'s ``custom_jvp``. Since gh#1010 a
    plain ``jax.vmap`` of :func:`solve` does the same (its callback is
    batched sequentially); this helper adds the aggregated failure report.

    ``on_failure`` behaves as in :func:`solve`, except that ``"warn"``
    emits **one** ``RuntimeWarning`` for the whole batch naming how many
    elements did not converge and which (index and status), after the
    batch has run; ``"raise"`` raises at the first failing element.
    """
    p_batch = jnp.asarray(p_batch)
    _check_on_failure(on_failure)
    sink, flush = _batch_failure_sink(on_failure)
    extra = dict(jac_pattern=jac_pattern, hess_pattern=hess_pattern,
                 on_failure=on_failure, _sink=sink)
    static_bounds, dyn_bounds = _split_bounds(lb, ub, cl, cu)
    fn = _make_solve_custom_jvp(f, g, n, m, static_bounds, options, extra)

    def one(p_i):
        return fn(p_i, x0, dyn_bounds)

    # ``jax.lax.map`` runs sequentially under the hood (one element at
    # a time), which is exactly what we want for an impure callback.
    out = jax.lax.map(one, p_batch)
    if flush is not None:
        jax.debug.callback(flush, out)
    return out


def _solve_batch_threadpool(
    f, g, p_batch_np, x0_np, n, m, lb, ub, cl, cu, options, workers,
    extra=None,
):
    """Dispatch ``B`` independent solves across a ``ThreadPoolExecutor``.

    Each worker builds its own ``Problem`` (no shared state) and runs
    ``Problem.solve``. Genuine parallelism is unlocked by the
    ``py.allow_threads`` block around ``optimize_tnlp`` in
    ``pounce-py`` — the GIL is released across the IPM iteration so
    threads actually run concurrently on the Rust side. JAX-traced
    ``f`` / ``g`` callbacks reacquire the GIL the usual way; that's
    serialized but the per-step cost is small relative to the linear
    algebra.
    """
    B = p_batch_np.shape[0]
    n_workers = workers or min(B, 8)
    x_out = np.empty((B, n), dtype=np.float64)
    lam_out = np.empty((B, m), dtype=np.float64)
    zL_out = np.empty((B, n), dtype=np.float64)
    zU_out = np.empty((B, n), dtype=np.float64)
    extra = dict(extra or {})
    on_failure = extra.get("on_failure", "warn")
    results = [(0, None)] * B

    def one(i):
        def sink(info):
            results[i] = (int(info["status"]), info.get("status_msg"))

        x_np, info = _solve_once(
            f=f, g=g,
            p=jnp.asarray(p_batch_np[i]),
            x0=jnp.asarray(x0_np[i]) if x0_np.ndim == 2 else jnp.asarray(x0_np),
            n=n, m=m, lb=lb, ub=ub, cl=cl, cu=cu,
            options=options,
            extra=dict(extra, _sink=sink),
        )
        x_out[i] = x_np
        lam_out[i] = np.asarray(info["mult_g"], dtype=np.float64)
        zL_out[i] = np.asarray(info["mult_x_L"], dtype=np.float64)
        zU_out[i] = np.asarray(info["mult_x_U"], dtype=np.float64)

    if n_workers <= 1 or B <= 1:
        for i in range(B):
            one(i)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(one, range(B)))
    _report_batch_failures(results, on_failure)
    return x_out, lam_out, zL_out, zU_out


def _make_vmap_solve_parallel_custom_vjp(
    f: Callable,
    g: Callable | None,
    n: int,
    m: int,
    static_bounds,
    options: dict | None,
    workers: int | None,
    extra: dict | None = None,
):
    @jax.custom_vjp
    def solve_fn(p_batch, x0_batch, dyn_bounds):
        x_star, *_ = _pure_callback_parallel_solve(
            f, g, p_batch, x0_batch, n, m, static_bounds, dyn_bounds,
            options, workers, extra,
        )
        return x_star

    def fwd(p_batch, x0_batch, dyn_bounds):
        x_star, lam, mult_xL, mult_xU = _pure_callback_parallel_solve(
            f, g, p_batch, x0_batch, n, m, static_bounds, dyn_bounds,
            options, workers, extra,
        )
        return x_star, (p_batch, x_star, lam, mult_xL, mult_xU, dyn_bounds)

    def bwd(residuals, cotangent_x_batch):
        (p_batch, x_star_batch, lam_batch, mult_xL_batch, mult_xU_batch,
         dyn_bounds) = residuals
        # The bounds are per-problem, not per-batch-element, so they stay
        # in ``bwd_single``'s closure rather than joining the vmap.
        _lb, _ub, cl, cu = _merge_bounds(static_bounds, dyn_bounds)

        def bwd_single(p, x_star, lam, mult_xL, mult_xU, v):
            return _kkt_implicit_backward(
                f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, v
            )

        dL_dp_batch = jax.vmap(bwd_single)(
            p_batch, x_star_batch, lam_batch, mult_xL_batch, mult_xU_batch,
            cotangent_x_batch,
        )
        # x0_batch carries no gradient (matches `solve`). The bounds are
        # shared across the batch, so a coordinate trips if it is active
        # in *any* element — that element's dropped term is nonzero.
        masks_batch = jax.vmap(
            lambda p_i, x_i, lam_i, zL_i, zU_i: _active_bound_masks(
                dyn_bounds, g, m, cl, cu, p_i, x_i, lam_i, zL_i, zU_i
            )
        )(p_batch, x_star_batch, lam_batch, mult_xL_batch, mult_xU_batch)
        masks = {k: jnp.any(v, axis=0) for k, v in masks_batch.items()}
        return (
            dL_dp_batch,
            jnp.zeros_like(x_star_batch),
            _bound_cotangents(dyn_bounds, masks),
        )

    solve_fn.defvjp(fwd, bwd)
    return solve_fn


def _pure_callback_parallel_solve(
    f, g, p_batch, x0_batch, n, m, static_bounds, dyn_bounds, options, workers,
    extra=None,
):
    B = p_batch.shape[0]
    result_shapes = (
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, m), jnp.float64),
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, n), jnp.float64),
    )

    def host_call(p_h, x0_h, bounds_h):
        lb, ub, cl, cu = _merge_bounds(static_bounds, bounds_h)
        return _solve_batch_threadpool(
            f, g, np.asarray(p_h), np.asarray(x0_h),
            n, m, lb, ub, cl, cu, options, workers, extra,
        )

    return jax.pure_callback(
        host_call, result_shapes, p_batch, x0_batch, dyn_bounds,
    )


def vmap_solve_parallel(
    p_batch,
    *,
    f: Callable,
    g: Callable | None = None,
    x0,
    n: int,
    m: int = 0,
    lb=None,
    ub=None,
    cl=None,
    cu=None,
    options: dict | None = None,
    workers: int | None = None,
    jac_pattern=None,
    hess_pattern=None,
    on_failure: str = "warn",
):
    """Parallel batched solve. Drop-in for :func:`vmap_solve`.

    Each of the ``B`` elements of ``p_batch`` is dispatched to a worker
    in a ``ThreadPoolExecutor`` of size ``workers`` (default:
    ``min(B, 8)``). Each worker owns an independent ``Problem`` so
    there's no shared state. The ``py.allow_threads`` block around
    ``optimize_tnlp`` in ``pounce-py`` releases the GIL across the
    IPM iteration, so threads actually run concurrently on the Rust
    side — the only cross-thread serialization is the Python
    callbacks for ``f`` / ``g``, which reacquire the GIL the usual
    way (typically a small fraction of total solve time for
    JAX-jitted callables).

    Differentiable w.r.t. ``p_batch`` via per-element implicit
    function theorem. The backward pass vectorizes naturally via
    ``jax.vmap`` because the KKT solve is pure JAX.

    ``x0`` may be a single ``(n,)`` vector (broadcast to all batch
    elements) or a ``(B, n)`` batch.

    ``jac_pattern`` / ``hess_pattern`` / ``on_failure`` behave as in
    :func:`solve` (gh#985); the patterns are shared by every batch
    element, so they must cover the structure at all of them. Failures
    are reported once for the whole batch, after every element has run:
    one ``RuntimeWarning`` (or ``RuntimeError`` under ``"raise"``) naming
    how many elements did not converge and which (index and status).
    """
    p_batch = jnp.asarray(p_batch)
    B = p_batch.shape[0]
    x0_arr = jnp.asarray(x0)
    if x0_arr.ndim == 1:
        x0_arr = jnp.broadcast_to(x0_arr, (B, n))
    static_bounds, dyn_bounds = _split_bounds(lb, ub, cl, cu)
    _check_on_failure(on_failure)
    extra = dict(jac_pattern=jac_pattern, hess_pattern=hess_pattern,
                 on_failure=on_failure)
    fn = _make_vmap_solve_parallel_custom_vjp(
        f, g, n, m, static_bounds, options, workers, extra,
    )
    return fn(p_batch, x0_arr, dyn_bounds)
