"""Compiled host-side evaluation of user callables (gh#1241).

``odeint``, ``daeint`` and ``solve_bvp(method="newton")`` run their Newton
loop on the host, inside a ``jax.pure_callback``, and evaluate the user's
RHS / residual there with concrete numpy operands — dozens to hundreds of
times per solve (residuals, plus a finite-difference column Jacobian). That
evaluation sits *outside* any enclosing ``jax.jit`` trace, so calling the
user's function as-is dispatches every JAX primitive in it one at a time.
For RHS code written the usual way (one term per reaction, ``.at[].add()``
updates) that measured 50x slower than the same solve with ``jax.jit(rhs)``.

:func:`host_jit` compiles the callable once per input shape and reuses it
for every host evaluation of a solve. It is built per solve call, not cached
globally on the function object, so a function closing over mutable state
sees the state as of each call — exactly what eager evaluation saw.

Eager evaluation also accepted functions that branch in Python on a concrete
value (``if t > 0.5:``), which ``jax.jit`` rejects. Such a function falls
back to eager evaluation for the rest of the solve rather than failing.
"""

from __future__ import annotations

import jax

# Errors raised when a function that worked eagerly cannot be traced:
# Python control flow on a value, or numpy conversion of a traced array.
_UNTRACEABLE = (
    jax.errors.ConcretizationTypeError,      # incl. TracerBoolConversionError
    jax.errors.TracerArrayConversionError,
    jax.errors.TracerIntegerConversionError,
)


def host_jit(fn):
    """Return ``fn`` compiled with ``jax.jit``, falling back to ``fn`` itself
    (eagerly, from then on) if it turns out not to be traceable."""
    jitted = jax.jit(fn)
    eager = False

    def call(*args):
        nonlocal eager
        if not eager:
            try:
                return jitted(*args)
            except _UNTRACEABLE:
                eager = True
        return fn(*args)

    return call
