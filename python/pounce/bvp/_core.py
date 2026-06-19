"""Backend-agnostic collocation core for the differentiable BVP solver.

A boundary value problem

    dy/dx = f(x, y, p, theta),    a <= x <= b
    bc(y(a), y(b), p, theta) = 0

is discretised on a **fixed** mesh ``x = [x_0, ..., x_{m-1}]`` with the
4th-order Lobatto IIIA (Hermite--Simpson) collocation formula — the same
scheme SciPy's :func:`scipy.integrate.solve_bvp` uses on each mesh
interval. For an interval ``[x_i, x_{i+1}]`` with width ``h_i`` and node
states ``y_i``, ``y_{i+1}``::

    f_i      = f(x_i, y_i)
    f_{i+1}  = f(x_{i+1}, y_{i+1})
    y_mid    = (y_i + y_{i+1})/2 - h_i/8 * (f_{i+1} - f_i)
    f_mid    = f(x_i + h_i/2, y_mid)
    r_i      = y_{i+1} - y_i - h_i/6 * (f_i + 4 f_mid + f_{i+1})   (= 0)

Stacking the ``n*(m-1)`` collocation residuals with the ``n + k`` boundary
residuals (``k = len(p)`` unknown parameters) gives a **square** root-find
of size ``N = n*m + k`` in the unknowns ``z = [vec(Y); p]``.

The same residual drives every backend:

* the NumPy SciPy-compatible :func:`pounce.bvp.solve_bvp` (root-find posed
  as a pounce feasibility NLP),
* the differentiable ``pounce.jax.solve_bvp`` / ``pounce.torch.solve_bvp``
  layers, where ``theta`` is the autodiff knob and ``dz*/dtheta`` falls out
  of the implicit-function theorem on the converged KKT system.

Nothing here imports NumPy/JAX/Torch at module load: the array operations
are taken as arguments (``concat`` / a row-major ``reshape``) so the one
implementation serves all three. ``f`` and ``bc`` passed in are already
*normalised* to the ``(x, Y, p)`` / ``(ya, yb, p)`` signature with
``theta`` closed over (see :func:`normalize_fun` / :func:`normalize_bc`).
"""

from __future__ import annotations


def collocation_residual(nfun, nbc, x, Y, p, concat):
    """Flat Hermite--Simpson collocation + boundary residual.

    Parameters
    ----------
    nfun : callable
        Normalised RHS ``nfun(x, Y, p) -> (n, m)`` (vectorised over the
        mesh, SciPy convention).
    nbc : callable
        Normalised boundary residual ``nbc(ya, yb, p) -> (n + k,)``.
    x : array (m,)
        The fixed mesh.
    Y : array (n, m)
        State values at the mesh nodes.
    p : array (k,)
        Unknown parameters (may be length 0).
    concat : callable
        ``concat([a, b]) -> array`` for the active backend
        (``np.concatenate`` / ``jnp.concatenate`` / a ``torch.cat`` shim).

    Returns
    -------
    array (n*(m-1) + n + k,)
        ``[vec(col_res); bc_res]`` — zero at a solution.
    """
    h = x[1:] - x[:-1]                                  # (m-1,)
    f = nfun(x, Y, p)                                   # (n, m)
    y_mid = 0.5 * (Y[:, 1:] + Y[:, :-1]) - 0.125 * h * (f[:, 1:] - f[:, :-1])
    x_mid = x[:-1] + 0.5 * h                            # (m-1,)
    f_mid = nfun(x_mid, y_mid, p)                       # (n, m-1)
    col = (Y[:, 1:] - Y[:, :-1]) - (h / 6.0) * (f[:, :-1] + 4.0 * f_mid + f[:, 1:])
    bc_res = nbc(Y[:, 0], Y[:, -1], p)                  # (n + k,)
    return concat([col.reshape(-1), bc_res])


def residual_of_z(z, nfun, nbc, x, n, m, k, concat):
    """Collocation residual as a function of the flat unknown ``z``.

    ``z = [vec(Y); p]`` with ``vec`` the row-major (state-major) flatten of
    the ``(n, m)`` state array. Returns the length-``N`` residual.
    """
    Y = z[: n * m].reshape(n, m)
    p = z[n * m :]
    return collocation_residual(nfun, nbc, x, Y, p, concat)


def pack_z(Y, p, concat):
    """Flatten ``(Y, p)`` into the unknown vector ``z = [vec(Y); p]``."""
    return concat([Y.reshape(-1), p])


def unpack_z(z, n, m):
    """Inverse of :func:`pack_z`: return ``(Y, p)`` views of ``z``."""
    return z[: n * m].reshape(n, m), z[n * m :]


def num_unknowns(n, m, k):
    """Size ``N = n*m + k`` of the collocation root-find."""
    return n * m + k


def _make_normalized(fun, bc, theta, uses_p):
    """Return ``(nfun, nbc)`` with ``theta`` closed in and a uniform
    ``(x, Y, p)`` / ``(ya, yb, p)`` call shape.

    ``theta is None`` selects the SciPy-style no-``theta`` signatures;
    otherwise ``theta`` is threaded as the trailing argument. ``uses_p``
    decides whether ``p`` is forwarded to the user callables.
    """
    if theta is None:
        if uses_p:
            nfun = lambda x, Y, p: fun(x, Y, p)
            nbc = lambda ya, yb, p: bc(ya, yb, p)
        else:
            nfun = lambda x, Y, p: fun(x, Y)
            nbc = lambda ya, yb, p: bc(ya, yb)
    else:
        if uses_p:
            nfun = lambda x, Y, p: fun(x, Y, p, theta)
            nbc = lambda ya, yb, p: bc(ya, yb, p, theta)
        else:
            nfun = lambda x, Y, p: fun(x, Y, theta)
            nbc = lambda ya, yb, p: bc(ya, yb, theta)
    return nfun, nbc
