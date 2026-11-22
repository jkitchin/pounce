"""Implicit-function ``custom_jvp`` for a square root-find (gh#1229).

``odeint``, ``daeint`` and ``solve_bvp(method="newton")`` all solve a square
system ``R(z, *args) = 0`` on the host and differentiate the root with the
implicit-function theorem, ``ż = -R_z⁻¹ (∂R/∂args · ȧrgs)``. They used to wrap
that in a ``jax.custom_vjp``, which JAX refuses to push a tangent through:
``jax.jvp`` / ``jax.jacfwd`` raised "can't apply forward-mode autodiff (jvp)
to a custom_vjp function". For parameter fitting — a few parameters, many
residuals — forward mode is the cheap direction (one tangent solve per
parameter instead of one adjoint solve per residual), so that was the wrong
limitation to have.

:func:`make_ift_root` builds the root as a ``jax.custom_jvp`` instead, the
same construction gh#1010 used for :func:`pounce.jax.solve`. The tangent rule
is plain JAX apart from the linear solve, which goes through
``jax.lax.custom_linear_solve`` with a host ``solve`` (``R_z w = b``) and
``transpose_solve`` (``R_zᵀ u = c``) against a sparse LU of ``R_z`` at the
converged ``z*``. One definition then serves every transform: forward mode
directly, reverse mode by JAX transposing the rule (the transposed linear
solve is exactly the old backward's back-solve), and ``vmap`` through the
callbacks' ``broadcast_all`` batching, which groups lanes by primal so a
``jacfwd`` / ``jacrev`` is one factorization and a multi-RHS solve.
"""

from __future__ import annotations

import jax


def make_ift_root(forward, residual, lin_solve):
    """Return ``solve_fn(*args) -> z*`` as an implicit-function ``custom_jvp``.

    Parameters
    ----------
    forward : callable
        ``forward(*args) -> z*`` — the primal root (typically a
        ``pure_callback``). Only ever evaluated on primal values.
    residual : callable
        ``residual(z, *args) -> R`` in traceable JAX, square in ``z``.
        Arguments it ignores (an initial guess, say) get an exact zero
        sensitivity, which is the right answer: a root is a root.
    lin_solve : callable
        ``lin_solve(z, args, b, transpose) -> w`` solving ``R_z w = b``
        (``transpose=False``) or ``R_zᵀ w = b`` (``transpose=True``) at
        ``z``. Must batch under ``vmap`` (a ``broadcast_all`` callback).
    """

    @jax.custom_jvp
    def solve_fn(*args):
        return forward(*args)

    @solve_fn.defjvp
    def _solve_jvp(primals, tangents):
        z = solve_fn(*primals)
        _, r = jax.jvp(lambda *a: residual(z, *a), primals, tangents)

        def matvec(w):
            return jax.jvp(lambda zz: residual(zz, *primals), (z,), (w,))[1]

        z_dot = jax.lax.custom_linear_solve(
            matvec, -r,
            solve=lambda _mv, b: lin_solve(z, primals, b, False),
            transpose_solve=lambda _vm, c: lin_solve(z, primals, c, True),
        )
        return z, z_dot

    return solve_fn
