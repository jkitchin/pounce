# QA campaign — LP/QP stack + PyTorch frontend

_Hands-on validation of the LP/QP/conic stack and the PyTorch frontend (#111):
scaling, robustness, error-message quality, and the new-user onboarding path.
Existing unit suites are green; this campaign targets what they don't cover.
Findings are filed as GitHub issues (linked below); coverage gaps land as
committed regression tests._

## Summary of findings

Eight findings, all filed as GitHub issues; no product code changed this pass
(per the campaign decision). Coverage gaps landed as committed regression tests
in `python/tests/test_qa_lp_qp_torch.py` (green for verified-correct behavior,
`xfail` pinned to the issue for each bug). The stack is **numerically sound** —
LP/QP match scipy to ~1e-11, JAX↔Torch parity is exact, gradchecks (incl.
double-backward) pass, warm-start is correct, and known SOS/SOCP optima are hit.
The findings are about **error legibility, input validation, docs, and a
dense-input scaling cliff**, not about wrong answers on well-posed convex input.

| # | Sev | Finding | Issue |
|---|---|---|---|
| F1 | S2 | `solve_qp` accepts an indefinite `P`, returns silently-wrong `optimal` | [#112](https://github.com/jkitchin/pounce/issues/112) |
| F2+F8 | S2 | `solve_qp` no input validation: shape mismatch→`primal_infeasible`, NaN/Inf→`iteration_limit` | [#113](https://github.com/jkitchin/pounce/issues/113) |
| F3 | S2 | docs QCQP routing snippet routes to NLP (KeyError) when run verbatim | [#114](https://github.com/jkitchin/pounce/issues/114) |
| F4 | S2 | `minimize()` verbose by default (unlike scipy); log bypasses Python stdout | [#115](https://github.com/jkitchin/pounce/issues/115) |
| F6 | S2 | dense `P`/`G` inputs hit a time+memory cliff (12 GB / 17 s at n=5000 vs 0.27 s sparse) | [#116](https://github.com/jkitchin/pounce/issues/116) |
| F5+F7 | S3 | `sos_minimize`: cryptic error on SymPy input; garbage `lower_bound` on `numerical_failure` | [#117](https://github.com/jkitchin/pounce/issues/117) |

Observations (by-design / not filed): O1 torch deprecation warning; O2/O5
float32 coercion vs rejection split; O3 QP routing needs analytic `jac`; O4
modest `vmap_solve_parallel` speedup; O6 extreme ill-conditioning →
`dual_infeasible`.

### Artifacts

- Findings report: this file.
- Scaling harness: `python/benchmarks/scaling_lp_qp_torch.py` (not in CI).
- Regression tests: `python/tests/test_qa_lp_qp_torch.py`
  (13 green + 5 `xfail` pinned to the issues above).

## Environment

| component | version |
|---|---|
| python | 3.11.13 (arm64, macOS) |
| pounce | 0.4.0 (editable, `maturin develop --release`) |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| torch | 2.12.0 (CPU) |
| jax / jaxlib | 0.10.1 |

Built from `main` at the #111 PyTorch landing (`899405e`). feral sibling at
`../feral`.

## Phase 0 — baseline (green)

- **Rust:** `cargo test -p pounce-convex -p pounce-qp -p pounce-cli` — all
  suites pass (0 failed, 0 ignored).
- **Python:** the LP/QP + conic + jax + torch + parity suites —
  **110 passed** (`test_qp`, `test_socp`, `test_sos`,
  `test_minimize_autoroute`, `test_minimize_socp_autoroute`, `test_qp_jax`,
  `test_socp_jax`, `test_torch`, `test_qp_torch`, `test_socp_torch`,
  `test_parity_jax_torch`).

Baseline is fully green, so any failure surfaced below is attributable to this
campaign's probing, not pre-existing breakage.

## Findings

_Severity: **S1** broken/incorrect · **S2** rough edge / bad UX · **S3** polish /
nice-to-have. Each filed finding links its issue._

### Phase 1 — new-user experience

#### Confirmed findings

- **F1 — `solve_qp` accepts an indefinite (nonconvex) `P` and returns a
  silently-wrong `status="optimal"`.** Severity **S2**.
  Repro:
  ```python
  import numpy as np
  from pounce.qp import solve_qp
  P = np.array([[1.0, 0.0], [0.0, -1.0]])   # indefinite
  r = solve_qp(P, np.zeros(2))
  # -> r.status == "optimal", r.x == [0, 0], r.obj == 0.0
  ```
  The true problem `min ½(x₀² − x₁²)` is **unbounded below**, yet the solver
  reports a clean optimum at the origin. The module docstring does scope the API
  to *convex* QP and the IPM's "verified unboundedness detection" only applies to
  convex inputs, so this is a documented-precondition violation rather than a
  contract break — hence S2, not S1. But there is **no PSD/indefiniteness guard
  on `P`**: a cheap symmetric-eigenvalue (or factorization) check could turn a
  silent wrong answer into a clear error, which is exactly the kind of mistake a
  new user makes (hand-built `P` that isn't PSD). Sensible fix: detect a
  non-PSD `P` and either raise or return a `nonconvex`/`invalid` status.

- **F2 — `solve_qp` does no `A`/`b` (or `G`/`h`) shape validation; a row-count
  mismatch silently yields `status="primal_infeasible"`.** Severity **S2**.
  Repro:
  ```python
  import numpy as np
  from pounce.qp import solve_qp
  P = np.diag([2.0, 2.0]); c = np.zeros(2)
  A = np.array([[1.0, 1.0]])     # 1 equality row
  b = np.array([1.0, 2.0])       # 2 RHS entries  <- mismatch
  r = solve_qp(P, c, A=A, b=b)
  # -> r.status == "primal_infeasible"  (not a shape error)
  ```
  A 1×2 `A` with a length-2 `b` is a user error that should surface as a clear
  `ValueError` ("A has 1 row but b has length 2"), but instead it is consumed
  and reported as an infeasible *problem* — a misdiagnosis that sends a new user
  hunting for a modelling bug that isn't there. Sensible fix: validate
  `A.shape[0] == b.shape[0]` and `G.shape[0] == h.shape[0]` (and column counts
  against `n`) in the Python wrapper before the solve.

- **F3 — the QCQP routing snippet in `docs/src/python.md` is broken when run
  verbatim: it routes to the NLP solver, not `socp`, and `print(res.info["solver"])`
  raises `KeyError`.** Severity **S2** (documented code that errors on copy-paste).
  The snippet (≈L171):
  ```python
  ball = {"type": "ineq", "fun": lambda x: 1.0 - x @ x}   # x·x ≤ 1
  res = minimize(lambda x: -x[0] - x[1], [0.1, 0.1], constraints=[ball])
  print(res.info["solver"])          # docs claim 'socp'
  ```
  Actual: an NLP filter-IPM iteration log prints, then `KeyError: 'solver'`
  (the key is absent on the NLP path — as the *earlier* routing snippet at
  ≈L160 correctly notes). The answer is right (`x≈[0.707, 0.707]`); only the
  routing claim is wrong. Root cause = O3: with no analytic `jac` the constraint
  Hessian is recovered from a finite-difference-of-finite-difference Jacobian,
  too noisy to confirm the quadratic, so detection conservatively falls to NLP.
  Adding `"jac": lambda x: -2*np.asarray(x)` to the constraint (and a `jac` to
  the objective) does route to `socp` (verified). Fix options: (a) give the
  doc's constraint/objective analytic `jac`s; (b) use `solve_socp(...)` with
  explicit cones in the snippet; or (c) replace the unconditional
  `print(res.info["solver"])` with `print(res.info.get("solver"))` and a note
  that derivative-free QCQP detection may defer to NLP.

- **F4 — `minimize()` prints a full IPM iteration table by default (unlike
  `scipy.optimize.minimize`, which is silent), and the log bypasses Python's
  `sys.stdout` so `contextlib.redirect_stdout` cannot suppress it.** Severity
  **S2**. The headline scipy-style snippet (`docs/src/python.md` ≈L62) prints a
  3-row iteration table on a trivial problem. A user porting from scipy expects
  silence unless `disp=True`. Worse, the log is emitted from Rust to OS fd 1, so
  the standard Python idioms to capture/silence output do **not** work — only
  `options={"print_level": 0}` does (which every *other* doc snippet quietly
  passes, a tell that the authors know it is noisy). For a "thin facade shaped
  after `scipy.optimize.minimize`", a silent-by-default `print_level` would match
  the advertised contract; at minimum the headline snippet should show how to
  quiet it. Repro:
  ```python
  import io, contextlib, numpy as np
  from pounce import minimize
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
      minimize(lambda x: (x-1)@(x-1)+1, x0=np.zeros(3))
  # table still prints to the terminal; buf is empty (Rust writes to fd 1)
  ```

#### Positive results (Phase 1)

- **Import guards are excellent.** With the `[torch]` extra absent,
  `import pounce.torch` raises a clear, actionable `ImportError`:
  `pounce.torch requires PyTorch; install with `pip install pounce[torch]`.`
  (verified by blocking the `torch` import). `pounce.jax`/`qp`/`sos` import
  cleanly with extras present.
- **Docs snippets that *are* self-contained run correctly** (objective values
  and gradients verified): the scipy-style `minimize` (snippet 1), the torch
  differentiable `solve` with an equality constraint (`x*=[0.3,0.7]`,
  `p.grad=[-0.4,0.4]`), and the torch `solve_qp` layer (`x=[0.25,0.25]`, active
  constraint ⇒ `c.grad≈0`, correct).
- **CLI first-contact is clean.** `pounce --help` is thorough; auto-routing
  prints a legible `Problem class: … Selected solver: …` line and solves the
  convex-QP and LP fixtures correctly (`obj=2.0` / `obj=-464.753`). Error paths
  are graceful, not panics: an unknown flag (`--solver`, which does not exist —
  the CLI forces a solver via the `solver_selection=` KEY=VALUE pair) exits 2
  with `unrecognized argument '--solver'` + usage; forcing
  `solver_selection=qp-ipm` on a nonconvex QP exits 2 with a descriptive
  *"problem class nonconvex QP does not match forced solver qp-ipm (expected an
  LP or convex QP)"*. The editable `maturin develop` install (no bundled
  binary) prints a clear, self-correcting message pointing at `cargo run` or the
  wheel.

#### Observations (by design / not pounce bugs)

- **O1 — `torch.func` emits a `torch.jit.script` DeprecationWarning under torch
  2.12.** `pounce.torch.from_torch` triggers it (pounce's own source contains no
  `jit.script`; it comes from inside `torch.func.grad/jacrev/hessian`).
  DeprecationWarnings are hidden from end users by default (only pytest surfaces
  them), so impact is low. Watch as torch evolves; not actionable in pounce
  today. Severity S3.

- **O2 — `pounce.torch.solve_qp`/`solve_socp` silently upcast float32 inputs to
  float64** (the returned tensor is float64 regardless of input dtype). This is
  *documented, deliberate* behavior — `python/pounce/torch/_qp.py` states "All
  inputs are coerced to float64", and the QP layer owns all the linear algebra
  so the upcast is safe. It does differ from the AD-traced `from_torch`/`solve`
  path, which *requires* float64 (Newton stalls in float32), but the two are
  consistent in spirit (both want float64; the QP layer can coerce because it
  controls the math, the traced path cannot coerce the user's function). Only
  surprise is the silent dtype change of the output. Severity S3.

- **O3 — `minimize()` auto-routing to `qp-ipm` requires a user-supplied `jac`
  (or `jac`+`hess`); pure derivative-free probing falls back to the NLP
  solver.** Confirmed against `python/pounce/_route.py`: without `jac`, the QP
  Hessian is a finite-difference *of a finite-difference* gradient, too noisy to
  pass the constant-Hessian / held-out-validation gates, so detection
  conservatively returns `None` (→ NLP). This is **by design** — the module
  docstring is explicit that detection is "deliberately conservative" and that a
  convex QP routed to the NLP solver is "merely *slower*", with the *correct*
  answer. (LP routing succeeds derivative-free because it only needs a constant
  *gradient*, one finite difference, not two.) The only cost is a silent
  discoverability gap: a user passing an opaque quadratic callable gets the
  right answer but never learns they missed the specialized fast path. Severity
  S3 (a doc note — "supply `jac` to unlock the QP fast path" — would close it).

### Phase 2 — correctness cross-checks

All correctness checks pass; the stack is numerically sound well beyond the
existing fixtures.

#### Positive results

- **Convex QP vs `scipy.optimize` SLSQP** — 20 random SPD problems with a random
  inequality + box: worst `|obj_pounce − obj_scipy| = 1.5e-11`.
- **LP vs `scipy.optimize.linprog` (HiGHS)** — 20 random feasible LPs: worst
  `|obj_pounce − obj_highs| = 4.2e-11`.
- **Routing transparency** — for convex QPs with analytic `jac`+`hess`,
  `minimize()` auto-routes to `qp-ipm` and its `x*` matches the forced-NLP
  solve to `≤1.4e-8` (the "routing never changes the answer" contract holds).
- **Torch `gradcheck` + `gradgradcheck`** (float64) pass on `solve_qp`
  (incl. double-backward), `solve_socp` (one SOC cone), and the NLP implicit-diff
  `solve` (equality-constrained).
- **JAX ↔ Torch parity** on `solve_qp` `x*` *and* `dL/dc` across four shapes
  (unconstrained, equality, active-inequality, box+inequality): agreement is
  **exact (0.0)** — both frontends wrap the same Rust solve and the same
  dense-KKT backward.
- **SOS/Lasserre known optima** — `(x−1)²+2 → 2.0`, `x²+y²−2x−4y+10 → 5.0`,
  `min x s.t. x²≤4 → −2.0` (all exact). **SOCP known optimum** —
  `min cᵀx s.t. ‖x‖≤1` with `c=[3,4]` gives `x*=[−0.6,−0.8]`, `obj=−5.0` (= −‖c‖).

#### Finding

- **F5 — `sos_minimize` raises a cryptic `TypeError: 'Add' object is not
  iterable` when handed a SymPy expression** (instead of the required
  `{exponent_tuple: coefficient}` dict). Severity **S3**. Passing a SymPy poly
  to a *polynomial* optimizer is the natural first attempt; the failure surfaces
  deep in `_infer_n_vars` rather than as a clear "objective must be a dict of
  `{exponent_tuple: coefficient}` (got a SymPy `Add`); see the module docstring"
  message. The dict format *is* documented, so this is purely an error-legibility
  gap. Sensible fix: a type check at the top of `sos_minimize` (and optionally a
  `sympy`-poly → dict convenience adapter).

### Phase 3 — scaling

Harness: `python/benchmarks/scaling_lp_qp_torch.py` (not run in CI). Times are
best-of-N wall seconds on the Phase-0 machine (macOS arm64); `maxrss` is the
process peak-RSS high-water mark.

#### Convex QP (`solve_qp`) — n sweep, box + thin inequality block

| n | dense P+G time | dense P+G maxrss | **sparse P+G** time | **sparse P+G** maxrss |
|---:|---:|---:|---:|---:|
| 10 | 0.001s | 54 MB | — | — |
| 100 | 0.003s | 55 MB | — | — |
| 1000 | 0.44s | 319 MB | — | — |
| 5000 | **16.7s** | **12.3 GB** | **0.27s** | **146 MB** |

All solves return `optimal` with correct objectives; iteration counts stay flat
(9→17 as n grows). The story is entirely **dense vs sparse inputs** → **F6**.

#### SOCP (`solve_socp`) — scales excellently

| shape | size | time | status |
|---|---|---:|---|
| many small cones | 2048 cones / 4096 rows | 0.045s | optimal |
| one big cone | 4096-dim SOC | 0.093s | optimal |

Sub-0.1s at 4096 dimensions; no memory growth of note. No findings.

#### Torch differentiable `solve_qp` — forward vs dense-KKT backward

| n | forward | backward | bwd/fwd | maxrss |
|---:|---:|---:|---:|---:|
| 100 | 0.002s | 0.002s | 1.15 | 211 MB |
| 250 | 0.012s | 0.012s | 1.02 | 219 MB |
| 500 | 0.044s | 0.047s | 1.06 | 252 MB |

The backward (dense KKT + `torch.linalg.solve`, flagged as a follow-up in
`torch/_qp.py`) costs ~the same as the forward through n=500 and stays light on
memory — the dense-KKT concern is real but not yet painful at moderate n.
(At large n it inherits the same dense-input cliff as F6, since the torch layer
takes dense tensors.)

#### Batch / vmap

| batch | `solve_qp_batch` | `vmap_solve` | `vmap_solve_parallel` | parallel speedup |
|---:|---:|---:|---:|---:|
| 64 | 0.003s | 0.114s | 0.112s | 1.02× |
| 256 | 0.012s | 0.456s | 0.441s | 1.03× |

`solve_qp_batch` is ~**40× faster** than the vmap paths (one batched native call
vs per-item Python AD tracing). See **O4** on the modest parallel speedup.

#### SOS/Lasserre (`sos_minimize`) — the practical ceiling

Convex bowl `Σ(xᵢ−1)²+1` (true min **1.0**), n_vars × relaxation order:

| n_vars \ order | 1 | 2 | 3 |
|---:|---:|---:|---:|
| 2 | 0.001s | 0.026s | 0.015s |
| 3 | 0.001s | 0.013s | 0.19s |
| 4 | 0.001s | 0.058s | **6.3s** |
| 5 | 0.001s | 0.24s | **66.4s → `numerical_failure`** |

The combinatorial blow-up of the moment matrix is expected (inherent to the
Lasserre hierarchy); now quantified — order-3 is the practical wall by ~5 vars.
See **F7** for the garbage-bound-on-failure issue.

#### Findings

- **F6 — `solve_qp`/`solve_socp` with *dense* `P`/`G` inputs hits a severe,
  silent time+memory cliff at a few thousand variables; the sparse path is
  60–80× better and most new users won't discover it.** Severity **S2**.
  Same n=5000 box-constrained QP, measured in isolation:
  - dense `P` + dense `G`: **16.7 s**, peak RSS **12.3 GB**
  - scipy-sparse `P` + sparse `G`: **0.27 s**, peak RSS **146 MB**
  Both return the correct `optimal`. A user who builds `P`/`G` as ordinary NumPy
  arrays (the obvious thing) silently gets the dense path and will OOM or appear
  to hang around n≈5k, never learning that wrapping the inputs in
  `scipy.sparse` solves the *same* problem in a quarter-second. **Both** matrices
  must be sparse — a sparse `P` with a dense `G` barely helps (the dense `G`
  forces densification). The docstring notes sparse is *accepted* but does not
  warn that dense is O(n³)/heavy-memory. Sensible fixes: (a) document the cliff
  and recommend sparse inputs for n ≳ 1000; (b) emit a one-time warning when a
  large dense `P`/`G` is passed; (c) auto-convert/treat obviously-sparse dense
  inputs as sparse. (The 12 GB is well beyond a dense 5000×5000 factorization's
  ~200 MB, so the dense path's internal fill/representation is itself worth a
  maintainer look — but the user-facing fix is "pass sparse".)

- **F7 — on a numerically failed SOS relaxation, `sos_minimize` returns a wildly
  wrong `lower_bound` (≈5e9 for a problem whose minimum is 1.0) alongside
  `status="numerical_failure"`.** Severity **S3**. The status *does* flag the
  failure (so it is not silently wrong for a user who checks `.status`), but a
  user who reads `.lower_bound` directly gets meaningless garbage rather than
  `nan`/`None`. Repro: the 5-var order-3 bowl above. Sensible fix: when the
  relaxation does not converge, set `lower_bound` to `nan`/`None` so a bad bound
  cannot be mistaken for a real one.

#### Observations

- **O4 — `vmap_solve_parallel` delivers only a modest ~1.0–1.2× over
  `vmap_solve`, not the multiple-of-`workers` speedup a threadpool suggests.**
  This is *consistent with the documented design*: `torch.func` transforms are
  process-global and not thread-safe, so the parallel path serializes the
  (GIL-bound) Python derivative callbacks under a lock and only the Rust IPM
  linear algebra runs concurrently. For small/medium problems the Python AD
  tracing dominates, so the win is small. Measured 1.02–1.03× at n=8, rising to
  ~1.2× when per-item work grows (n=60 with an exp term). Not a bug — the
  docstring is honest — but worth setting expectations: reach for
  `solve_qp_batch` (one batched native call, ~40× faster here) when the per-item
  problem is a QP/SOCP rather than an arbitrary differentiable `f`.

### Phase 4 — robustness / edge cases

The solver is robust on most degenerate inputs; the one real gap is **no
NaN/Inf input validation** (F8), which pairs with the shape gap (F2).

#### Positive results

- **Degenerate / redundant constraints handled correctly**: duplicate equality
  rows → `optimal`; redundant inequalities → `optimal`; contradictory equalities
  (`x₀=0 ∧ x₀=1`) → `primal_infeasible`.
- **Tiny / empty problems**: single-variable → `optimal`; unconstrained QP →
  `optimal`; **zero-size `n=0` → `optimal` with `x=[]`** (no crash).
- **Status reporting correct**: infeasible (`x≥2 ∧ x≤1`) → `primal_infeasible`;
  unbounded LP (`min −x, x≥0`) → `dual_infeasible`.
- **Warm-start works**: perturbing `c` slightly, a cold solve takes 13 IPM
  iterations, a warm solve from the previous `QpResult` takes **7**, and the two
  optima agree to `5.5e-13` (fewer iters, identical answer).
- **Equilibration helps badly-scaled data**: a problem with a 1e8-scaled linear
  term reaches the right neighborhood (`x₀≈−1e8`), though it flags
  `iteration_limit` at the extreme.

#### Finding

- **F8 — `solve_qp` does not validate NaN/Inf in its inputs; garbage propagates
  into the solve and surfaces as `iteration_limit`/`numerical_failure` with a
  `nan` solution, instead of a clear up-front error.** Severity **S2**.
  Observed:
  | input | result |
  |---|---|
  | `NaN` in `c` | `status="iteration_limit"`, `x=[nan, nan]` |
  | `NaN` in `b` | `status="iteration_limit"`, `x=[nan, nan]` |
  | `Inf` in `P` | `status="numerical_failure"`, `x=[0, 0]` |
  A `NaN` in the data reported as `iteration_limit` actively misleads — it tells
  the user to raise `max_iter` when the real problem is a bad input. This is the
  same class of gap as **F2** (no shape validation) and would naturally be fixed
  by the same up-front validation pass in the Python wrapper:
  `np.isfinite(...).all()` on every array, raising `ValueError` on failure.
  (Recommend filing F2 + F8 together as one "`solve_qp` input validation" issue.)

#### Observations

- **O5 — float32 handling is path-dependent (extends O2 to the native API).**
  The AD-traced NLP path (`pounce.torch.solve`) **rejects** float32 with a clear,
  excellent error (`p must be float64 (got torch.float32); the Newton and KKT
  …`). The QP/SOCP layers — **both** `pounce.torch.solve_qp` and the native
  `pounce.qp.solve_qp` — **silently upcast** float32 → float64 (output dtype is
  float64). This is internally defensible (the QP layer owns all the math and can
  safely coerce; the AD path cannot coerce the user's traced function) and the
  QP coercion is documented, but the *reject-vs-coerce* split across entry points
  is a small inconsistency a user may trip on. Severity S3.
- **O6 — extreme ill-conditioning is reported as `dual_infeasible` even when a
  (huge, finite) minimizer exists.** `min ½(1e-12·x₀² + 1e12·x₁²) + x₀ + x₁` has
  a finite optimum at `x₀≈−1e12`, but the solver returns `dual_infeasible`
  (≈"unbounded"). Genuinely hard numerically (condition number ~1e24); noted for
  completeness, not filed — the status is at least a non-`optimal` flag, not a
  silently wrong answer.

<!-- more findings appended as phases run -->
