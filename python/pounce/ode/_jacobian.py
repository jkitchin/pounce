"""Jacobian strategy for the ``pounce.ode`` integrators.

For each problem the Jacobian is obtained, in order of preference:

1. from a **user-supplied analytic Jacobian** (fastest, exact);
2. by **JAX forward-mode autodiff** (:func:`jax.jacfwd`) when the RHS /
   residual is JAX-traceable — exact, with *no* truncation error;
3. by a **central finite difference** for opaque callables (raw NumPy, C
   extensions, spline lookups, …) that JAX cannot trace.

Why this matters — and why *central*, not forward, differences: near a
singular-Jacobian steady state (e.g. the Robertson DAE on its slow manifold,
pounce#175) a forward difference's O(d) truncation error is large enough to
corrupt the simplified-Newton contraction, which drives the step-size
controller to ~45x more steps than SciPy on the same problem. A central
difference has O(d^2) error — and is *exact* for the quadratic stiff terms
that dominate there — which restores step-count parity. Exact JAX autodiff
sidesteps the issue entirely, which is why it is preferred whenever the RHS
can be traced. (The rest of POUNCE is JAX-native; the ODE module defaulting
to a hand-rolled finite difference was an oversight this module corrects.)

The JAX-vs-FD decision is *probed once* per problem — on the first Jacobian
evaluation — and cached: a single trial trace either succeeds (a jitted
``jacfwd`` is reused for every subsequent step) or raises (fall back to
central differences for the remainder of the solve). This keeps opaque
callables working with zero JAX overhead while giving traceable ones exact
derivatives.
"""
import numpy as np

_SQRT_EPS = float(np.sqrt(np.finfo(float).eps))


def _central(evalfn, x):
    """Central-difference Jacobian of ``evalfn: R^n -> R^m`` at ``x``.

    Returns ``(m, n)``. Costs ``2n`` evaluations. The perturbation
    ``sqrt(eps)*max(1, |x_j|)`` matches the forward-difference scaling this
    replaces, but the symmetric stencil gives O(d^2) truncation error.
    """
    n = x.size
    cols = []
    for j in range(n):
        d = _SQRT_EPS * max(1.0, abs(x[j]))
        xp = x.copy(); xp[j] += d
        xm = x.copy(); xm[j] -= d
        cols.append((np.asarray(evalfn(xp), float)
                     - np.asarray(evalfn(xm), float)) / (2.0 * d))
    return np.stack(cols, axis=1)


def _try_jax_jacfwd(argnums):
    """Return ``jax.jit(jax.jacfwd(g, argnums))`` builder, or ``None`` if JAX
    is unavailable. ``g`` wraps the callable so list/tuple RHS outputs are
    stacked into an array before differentiation."""
    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        return None

    # pounce is a double-precision solver; a float32 autodiff Jacobian is too
    # coarse to drive the stiff simplified-Newton iteration reliably. Enable
    # x64 globally (a no-op if already on), matching ``pounce.jax``.
    jax.config.update("jax_enable_x64", True)

    def build(fn):
        g = lambda *a: jnp.asarray(fn(*a))
        return jax.jit(jax.jacfwd(g, argnums=argnums))

    return build


def make_ode_jac(fun_raw, f_counting=None):
    """Return ``jac(t, y, f0) -> (n, n)`` for an explicit ODE ``y' = f(t, y)``.

    ``fun_raw`` is the untouched user RHS (traced by JAX). ``f_counting``, if
    given, is a wrapper that tallies evaluations (used only on the
    finite-difference path, so ``nfev`` still reflects FD work; the JAX path
    performs no host-side evaluations).
    """
    f_cd = f_counting if f_counting is not None else fun_raw
    state = {"mode": None, "jf": None}

    def jac(t, y, f0):
        if state["mode"] is None:
            _probe_ode(fun_raw, t, y, state)
        if state["mode"] == "jax":
            return np.asarray(state["jf"](t, y), dtype=float)
        return _central(lambda yy: f_cd(t, yy), y)

    return jac


def _probe_ode(fun_raw, t, y, state):
    build = _try_jax_jacfwd(argnums=1)
    if build is not None:
        try:
            jf = build(fun_raw)
            J = np.asarray(jf(t, y), dtype=float)
            if J.shape == (y.size, y.size) and np.all(np.isfinite(J)):
                state["jf"] = jf
                state["mode"] = "jax"
                return
        except Exception:
            pass
    state["mode"] = "cd"


def make_dae_jac(F_raw, F_counting=None):
    """Return ``jacs(t, y, yp, F0) -> (F_y, F_y')`` for an implicit DAE
    ``F(t, y, y') = 0``. JAX differentiates w.r.t. ``y`` (argnums 1) and ``y'``
    (argnums 2); the fallback central-differences each separately."""
    F_cd = F_counting if F_counting is not None else F_raw
    state = {"mode": None, "jfy": None, "jfp": None}

    def jacs(t, y, yp, F0):
        if state["mode"] is None:
            _probe_dae(F_raw, t, y, yp, state)
        if state["mode"] == "jax":
            Fy = np.asarray(state["jfy"](t, y, yp), dtype=float)
            Fyp = np.asarray(state["jfp"](t, y, yp), dtype=float)
            return Fy, Fyp
        Fy = _central(lambda yy: F_cd(t, yy, yp), y)
        Fyp = _central(lambda pp: F_cd(t, y, pp), yp)
        return Fy, Fyp

    return jacs


def _probe_dae(F_raw, t, y, yp, state):
    by = _try_jax_jacfwd(argnums=1)
    bp = _try_jax_jacfwd(argnums=2)
    if by is not None and bp is not None:
        try:
            jfy = by(F_raw)
            jfp = bp(F_raw)
            Fy = np.asarray(jfy(t, y, yp), dtype=float)
            Fyp = np.asarray(jfp(t, y, yp), dtype=float)
            n = y.size
            if (Fy.shape == (n, n) and Fyp.shape == (n, n)
                    and np.all(np.isfinite(Fy)) and np.all(np.isfinite(Fyp))):
                state["jfy"] = jfy
                state["jfp"] = jfp
                state["mode"] = "jax"
                return
        except Exception:
            pass
    state["mode"] = "cd"
