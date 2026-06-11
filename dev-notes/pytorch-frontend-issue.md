## Summary

Add a **PyTorch frontend** for pounce's differentiable solver, mirroring the
existing `pounce.jax` subpackage. The goal is a `pounce.torch` namespace where a
solve is a `torch.autograd.Function` you can drop inside a learned model and
backprop through, with the same constraint-satisfaction guarantee the JAX path
gives today.

This is a **frontend/adapter**, not a second solver. The numerical core (the
Rust IPM, exposed via `pounce._pounce.Problem`) and the implicit-function-theorem
backward math are autodiff-framework-agnostic. PyTorch needs only a thin wrapper
layer — and because PyTorch is eager, that layer is *smaller* than the JAX one
(no `pure_callback` / `ShapeDtypeStruct` machinery).

## Motivation / positioning

pounce's differentiable layer is one Rust IPM with a KKT-based implicit backward.
JAX is the first frontend; making PyTorch a thin binding turns "a JAX library"
into "one numerical backbone under any autodiff frontend" — the same "one roof"
thesis extended from problem classes to autodiff frameworks. Precedent:
cvxpylayers ships JAX + PyTorch + TF bindings off one core (`diffcp`); theseus is
PyTorch-native for this class of layer. A large share of the ML/research audience
is PyTorch-first, so this widens reach materially for relatively contained effort.

## What is already framework-agnostic (reuse as-is)

1. **The solver core** — `pounce._pounce.Problem`. The boundary is already NumPy
   (`_diff.py::_solve_once` / `host_call` do `np.asarray`). PyTorch CPU tensors
   are zero-copy to/from NumPy, so the Rust side does not change at all.
2. **The implicit-function-theorem backward** — assemble the KKT block
   `[[H, Jᵀ], [J, D]]`, solve against the cotangent, contract with the parameter
   sensitivities (`_diff.py:128-208`). Pure linear algebra; reimplement with
   `torch.linalg.solve` instead of `jnp.linalg.solve`. The active-set logic
   (bound multipliers → `dx/dp = 0` on active coords; slack inequality rows
   dropped via the identity-augment trick, pounce#73) ports line-for-line.

## What is JAX-specific (needs a PyTorch equivalent)

| JAX piece (file) | PyTorch equivalent | Notes |
|---|---|---|
| `jax.grad/jacrev/hessian` on user `f,g` (`_build.py`) | `torch.func.grad/jacrev/hessian` | `torch.func` mirrors JAX's API; near-mechanical |
| `@jax.custom_vjp` + `fwd`/`bwd` (`_diff.py`) | `torch.autograd.Function` + `forward`/`backward` | same split |
| `jnp.linalg.solve`, `jnp.where`, `jnp.diag` (KKT bwd) | `torch.linalg.solve`, `torch.where`, `torch.diag` | line-for-line |
| `jax.pure_callback` + `ShapeDtypeStruct` (`_diff.py::_pure_callback_solve`) | **dropped** | eager mode calls `problem.solve(...)` directly inside `forward` |
| global `jax_enable_x64` (`jax/__init__.py`) | `torch.float64` tensors | no global flag; validate float32 path is rejected/guarded |
| `jax.lax.map` / threadpool batching (`_diff.py::vmap_solve*`) | Python loop or `torch.func.vmap`; reuse the *same* `ThreadPoolExecutor` | parallel path is pure Python + Rust GIL-release — identical |
| sparse colored AD (`_build.py`, CPR coloring) | rebuild on `torch.func.jvp/vjp` | one JVP/HVP per color; biggest non-mechanical port |

## Surface to port (parity target with `pounce.jax`)

Map the public API in `python/pounce/jax/__init__.py`:

- `from_jax` → `from_torch` (`_build.py`) — build a `Problem` from traced
  `f(x)`, `g(x)`; gradient/Jacobian (with detected sparsity)/Lagrangian Hessian.
- `solve`, `solve_with_warm` → `_diff.py` — the `custom_vjp` → `autograd.Function`
  port, incl. dual + μ warm-start threading (pounce#86).
- `vmap_solve`, `vmap_solve_parallel` → batched solves (loop + threadpool).
- `JaxProblem`, `AnchorState` → `TorchProblem` (`_problem.py`) — stateful builder
  that caches the compiled AD artefacts, sparsity, and active-set masks for
  iterative use.
- `PathFollower`, `PathTrace`, `inverse_map_rhs` → `_path.py` — predictor–corrector
  path following.
- `QpLayer`, `solve_qp`, `solve_qp_batch`, `solve_socp` → `_qp.py` — the
  differentiable conic layers (the headline "feasible-by-construction" layer).

## Technical design

- **Package:** `python/pounce/torch/` mirroring `python/pounce/jax/` file split
  (`_build.py`, `_diff.py`, `_problem.py`, `_path.py`, `_qp.py`, `__init__.py`).
- **Optional dependency:** add `torch = ["torch>=2.2"]` to
  `[project.optional-dependencies]` in `python/pyproject.toml` (alongside the
  existing `jax` extra); import-guard with a useful error like the JAX path does.
  `torch.func` (functorch, merged into core) requires torch ≥ 2.0; pin ≥ 2.2 for
  a stable `torch.func` surface.
- **dtype:** require/validate float64 inputs (Newton + KKT solve need it). Either
  cast internally or raise on float32, matching the JAX x64 rationale.
- **Differentiable backward:** keep the `backward` itself differentiable where
  cheap (so double-backward works), as the JAX bwd does by staying in-framework.
- **Shared core, no duplication:** factor the framework-neutral solve/KKT-assembly
  helpers so JAX and Torch adapters call common code where practical (the active-set
  masking + KKT assembly is identical; only the array namespace differs). Consider
  an array-API/duck-typed inner helper to avoid two copies of the backward.

## Plan / phases

**Phase 0 — scaffolding (small).**
Create `python/pounce/torch/__init__.py` with the import guard + `torch` extra in
`pyproject.toml`. CI: add a `torch` test job (CPU wheel).

**Phase 1 — `solve` MVP (the proof point).**
Port `solve` (`from_torch` build + single `autograd.Function`). Validate
`torch.autograd.gradcheck` against finite differences and cross-check the gradient
numerically against the JAX `solve` on a shared fixture (e.g. `hs071`,
`inverse_map`). This phase alone demonstrates the whole thesis.

**Phase 2 — batching + warm starts.**
`vmap_solve`, `vmap_solve_parallel` (reuse the threadpool), `solve_with_warm`
(dual + μ threading, pounce#86). Verify `autograd.Function` vmap protocol or fall
back to a loop.

**Phase 3 — `TorchProblem` + sparse colored AD.**
Stateful builder caching AD artefacts; rebuild CPR coloring on `torch.func.jvp/vjp`.
This is the largest port — benchmark against `bench_sparse_ad_83`.

**Phase 4 — conic layers.**
`QpLayer`, `solve_qp/_batch`, `solve_socp` — the feasible-by-construction layer
that most directly competes with cvxpylayers/theseus.

**Phase 5 — path following + docs + parity tests.**
`PathFollower`/`inverse_map_rhs`; a docs page mirroring the JAX integration guide;
a parity test matrix asserting JAX and Torch agree to tolerance on shared fixtures.

## Testing strategy

- `torch.autograd.gradcheck` / `gradgradcheck` (float64) on every layer.
- **JAX↔Torch parity fixtures:** same `f,g,p` → assert `x*` and `dL/dp` match to
  tolerance. Port the existing `python/tests/test_jax.py`, `test_qp_jax.py`,
  `test_socp_jax.py`, `test_solver_session.py` as `test_*_torch.py`.
- Active-set edge cases that motivated pounce#73 (slack inequalities) — keep the
  regression in the Torch suite too.

## Open questions / risks

- **`autograd.Function` + `vmap`:** the newer functorch vmap protocol needs a
  `setup_context`/`vmap` staticmethod, or we loop. Decide per-layer.
- **GIL / threadpool parity:** confirm the `py.allow_threads` GIL-release around
  `optimize_tnlp` benefits Torch callbacks the same way (it should — it's below
  the Python layer).
- **Code reuse vs. duplication:** how much of the backward to share via a neutral
  inner helper vs. two readable copies. Lean toward one shared helper if it stays
  legible.
- **Dense KKT in the backward:** the current backward assembles a dense KKT and
  uses `linalg.solve` (noted as a follow-up in `_diff.py:30-36` to move to the
  Rust-side `pounce-sensitivity` sparse solve). That follow-up is
  framework-independent — both frontends benefit once it lands; don't block the
  Torch port on it.

## References

- `python/pounce/jax/__init__.py` — public surface to mirror.
- `python/pounce/jax/_diff.py` — `custom_vjp` + KKT backward (the core to port).
- `python/pounce/jax/_build.py` — model AD + sparsity detection.
- `python/pounce/jax/_qp.py`, `_path.py`, `_problem.py` — remaining surface.
- `python/pyproject.toml` — optional-dependency extras pattern.
- pounce#73 (slack-inequality active set), pounce#86 (μ warm-start).
- Prior art: cvxpylayers (`diffcp`), theseus.
