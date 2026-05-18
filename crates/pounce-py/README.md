# pounce-py

PyO3 bindings for POUNCE. Builds the `_pounce` Python extension module that
backs the `pounce` Python package (see `python/`).

The Python-facing surface is a cyipopt-compatible `Problem` class plus a
scipy-style `minimize()` facade and a `pounce.jax` subpackage providing:

- AD-built objective gradient, constraint Jacobian, and Lagrangian Hessian
  from JAX-traced `f(x)` and `g(x)` functions.
- A `custom_vjp` wrapper around the solver that differentiates `x*(p)` via
  the post-optimal KKT sensitivity rule (uses `pounce-sensitivity`).
- A `vmap` batching rule that loops the solver over a parameter batch.

Build:

```sh
# Develop install (needs maturin):
cd python && maturin develop --release

# Wheel:
cd python && maturin build --release
```

`cargo build` in the workspace builds this crate as a regular rlib for
type-checking; the wheel build adds `--features extension-module` so PyO3
links against the right Python ABI.
