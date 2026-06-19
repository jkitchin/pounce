# Changelog

All notable changes to POUNCE are tracked here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `1.0.0`. Pre-1.0 minor bumps may include breaking
changes.


## [Unreleased]

### Added — boundary value problems (`pounce.bvp`)

A `scipy.integrate.solve_bvp`-compatible boundary value problem solver, plus
differentiable JAX/PyTorch frontends:

- **`pounce.solve_bvp(fun, bc, x, y, p=None, ...)`** — drop-in for
  `scipy.integrate.solve_bvp`. Discretises the BVP with 4th-order
  Hermite–Simpson collocation on a fixed mesh and solves the square
  collocation root-find as a pounce feasibility NLP. Returns a SciPy-shaped
  bunch (`sol`, `x`, `y`, `yp`, `p`, `rms_residuals`, `niter`, `status`,
  `message`, `success`). Accuracy matches SciPy (same collocation scheme).
  The default `method="newton"` factors the exact **sparse** `N×N`
  collocation Jacobian (analytic per-node blocks from `fun_jac`/`bc_jac`,
  else a vectorised finite difference) with FERAL's unsymmetric sparse LU,
  using a **modified (frozen-Jacobian) Newton** that reuses the factor
  across steps and refactors only on stall — so it is **typically faster
  than `scipy.integrate.solve_bvp`** at equal mesh (≈0.6–1.0×), including
  large nonlinear problems. `method="ipm"` solves it as a pounce
  feasibility NLP.
  Adaptive mesh refinement is **on by default** (`adaptive=True`, like
  SciPy — a faithful port of SciPy's Lobatto residual estimator + refinement
  rule that reproduces its mesh sequence node-for-node); `adaptive=False`
  solves the given mesh as-is. The collocation system is solved to round-off
  independent of the mesh `tol` (the latter only gates refinement).
  `verbose` mirrors SciPy (1 = termination report, 2 = per-iteration
  progress). Result `status` codes: 0 converged, 1 max nodes, 2 singular
  Jacobian, 3 bc_tol unmet, 4 Newton non-convergence, 5 IPM acceptable-only.
- **`pounce._pounce.SparseLU`** — new PyO3 binding exposing FERAL's
  unsymmetric sparse LU (`factor` / `solve` / `solve_transpose`) for direct
  `A x = b` on general sparse matrices.
- **`pounce.solve_bvp_constrained`** — constrained / optimal-control BVPs
  (state & parameter bounds, inequality path constraints, optional
  objective over an under-determined system), solved with the interior-point
  method on the collocation NLP. This is unique to pounce —
  `scipy.integrate.solve_bvp` cannot express bounds, path constraints, or an
  objective.
- **`pounce.jax.solve_bvp` / `pounce.torch.solve_bvp`** — the same solve made
  differentiable w.r.t. a `theta` parameter threaded into `fun` / `bc`, via
  the implicit-function theorem on the collocation system. Supports
  gradients/Jacobians w.r.t. ODE/BC coefficients, boundary values, and the
  sensitivity of solved-for unknown parameters `p*`. The default
  `method="newton"` is the fast path (FERAL sparse-LU forward + sparse
  `R_zᵀ` backward, first-order). `method="ipm", second_order=True` wraps the
  solve in a `custom_jvp` that re-applies the implicit-function theorem,
  enabling `jax.grad(jax.grad(...))` / `jax.hessian` to arbitrary order.
- Docs: `docs/src/bvp.md`; worked accuracy/speed/differentiability comparison
  in `python/examples/bvp_scipy_compare.py`.

#### Scope and positioning

Honest framing of where this sits relative to other BVP solvers:

- **Algorithm class.** Fixed 4th-order Hermite–Simpson collocation — the same
  family as MATLAB `bvp4c` and `scipy.integrate.solve_bvp` (itself a
  bvp4c-style port). At equal mesh we match SciPy's accuracy and are
  typically a bit faster; this is "competitive with a widely-used production
  solver," **not** the numerical state of the art. Higher-order /
  variable-order collocation (COLNEW/COLSYS), 5th-order `bvp5c`, and
  deferred-correction / continuation codes (TWPBVP, ACDC) need fewer nodes
  per digit of accuracy and are more robust on stiff / singularly-perturbed
  boundary-layer problems.
- **Where it genuinely leads.** End-to-end **differentiability** of the
  solution (`∂y/∂θ`, Jacobians, second order) via implicit differentiation in
  JAX/PyTorch, and **integrated bound / path constraints and objectives**
  (optimal control) through the IPM — capabilities classical BVP solvers do
  not offer. (For heavy constrained optimal control, mature direct-collocation
  stacks such as CasADi and Pyomo.DAE + IPOPT remain more complete.)
- **Not yet covered:** variable/high-order collocation; continuation /
  deferred correction for stiff boundary layers; multipoint boundary
  conditions; DAEs; the singular term `S`; complex-valued problems. A
  credible "SOTA" claim would also require benchmarking against COLNEW /
  `bvp5c` / SciPy on a standard suite (e.g. the Cash–Mazzia test set) for
  accuracy-vs-nodes and robustness, not just speed-vs-SciPy.


## [0.5.0] - 2026-06-14

### Added — broader `scipy.optimize.minimize` compatibility

`pounce.minimize` now covers much more of the SciPy surface, so it works as a
drop-in `method=` callable for `scipy.optimize.minimize` and ports existing
SciPy code with fewer changes:

- **`args=(...)`** — extra positional arguments forwarded to `fun` / `jac`.
- **`jac=True`** — `fun` returns `(value, gradient)` in one call; the pair is
  cached so the gradient is not recomputed.
- **`callback`** — invoked each iteration; both SciPy signatures are accepted
  (`callback(xk)` and `callback(intermediate_result)`).
- **scipy `Bounds` and `LinearConstraint` objects** — accepted alongside
  `(lo, hi)` pairs and constraint dicts; a `LinearConstraint` may carry a
  sparse `A`, which is honored. When all constraints are linear the objective
  Hessian is the Lagrangian Hessian, so an exact `hess` is used (no L-BFGS
  fallback).
- **scipy option spellings** as synonyms — `maxiter`→`max_iter`,
  `gtol`/`ftol`/`xtol`→`tol`, `disp`→`print_level`, `maxcor`→
  `limited_memory_max_history`; options may be passed as `**kwargs` (the legacy
  `options={…}` dict still works).
- The result is now a genuine `scipy.optimize.OptimizeResult` carrying the
  `nfev` / `njev` / `nhev` evaluation counters, with pounce extras under
  `res.info` and a back-compat shim so a key absent at the top level falls back
  to `res.info`.

**Changed:** the `solver_selection` default is now `"nlp"` (no structure
probing) — automatic LP/QP/QCQP routing is opt-in via `solver_selection="auto"`,
so a general NLP or an expensive `fun` pays no probe overhead. The `args`
argument is now the third positional parameter (matching SciPy), ahead of `jac`.

### Fixed — `obj_scaling_factor` was silently ignored (maximization diverged)

The `obj_scaling_factor` option was registered but never read: every solve
constructed the NLP with no-op scaling, so the documented behavior — a
constant multiplier on the objective, negative to **maximize** — was a silent
no-op and maximization problems diverged (the IPM minimized the unscaled
objective). The option value is now carried into `OrigIpoptNlp` on both the
IPM and SQP paths (`ConstObjScaling`), combining with gradient-based /
user scaling exactly as documented. Sensitivity analysis works under a
negative factor too: the natural-units correction from #128 below uses a
two-sided scaling with no square root, so `solve_with_sens` /
`Solver.reduced_hessian` return the declared problem's reduced Hessian for
maximization problems as well.

### Added — KKT regularization reported alongside sensitivity outputs

The IPM's inertia-correction perturbations are baked into the converged
factor in scaled space, so a regularized final factorization makes the
natural-units sensitivity outputs (covariance in particular) inexact and not
perfectly scaling-invariant. The final `(δ_x, δ_s, δ_c, δ_d)` are now
reported so workflows can check for the all-zero (exact) case:
`info["kkt_perturbations"]` and `Solver.kkt_perturbations` (Python),
`SensResult::kkt_perturbations` and `Solver::kkt_perturbations` (Rust).

### Fixed — sensitivity back-solves now return natural (unscaled) units (#128)

The reduced Hessian from `solve_with_sens(compute_reduced_hessian=True)` /
`Solver.reduced_hessian`, the parametric step `dx`, and the raw
`Solver.kkt_solve` were returned in the IPM's internally **scaled** space
whenever NLP scaling was active (the default
`nlp_scaling_method = "gradient-based"` fires when an objective gradient or a
constraint row exceeds 100 at the starting point). For a parameter-estimation
NLP this made `-inv(reduced_hessian)` differ from the true covariance by
`df / (dc_i·dc_j)` — the discretization-tracking "≈ nfe" fudge factor reported
in #128. The same scaled factor silently corrupted the factor-reuse VJP/JVP of
**both** differentiable frontends (`pounce.jax` `JaxProblem(factor_reuse=True)`
and `pounce.torch` `TorchProblem`) on badly-scaled problems.

The scaled primal-dual system is the two-sided diagonal scaling
`K_scaled = E·K_natural·F` (per-block: `E = (df, df/dd, dc, dd, df, df)` and
`F = (1, 1/dd, dc/df, dd/df, 1/df, dd/df)` over `x, s, y_c, y_d, z, v`), so
every held-factor back-solve now computes `K_natural⁻¹ = F·K_scaled⁻¹·E`: all
eight KKT blocks — including the bound-multiplier z/v rows in `dx_full` —
come back in the user's own units regardless of scaling method, and a
negative `obj_scaling_factor` is handled (no square root involved). The CLI
sIPOPT mode inherits the same correction: the `red_hessian` var-suffix output
is now natural-units where upstream sIPOPT prints a scaled value it warns
about. The pre-fix solver-space values and the factors stay accessible:
`info["reduced_hessian_scaled"]` / `info["obj_scaling_factor"]` /
`info["pin_g_scaling"]`, `Solver.reduced_hessian(..., scaled=True)`,
`Solver.kkt_solve(..., scaled=True)` / `kkt_solve_many(..., scaled=True)`,
the `Solver.nlp_scaling` dict, the C ABI's `IpoptSolverKktSolveScaled`, and
the matching Rust surfaces (`SensResult` fields,
`Solver::{compute_reduced_hessian_scaled, kkt_solve_scaled,
kkt_solve_many_scaled, nlp_scaling}`, `PdSensBacksolver::solve_scaled_space`).

Also fixed in the same change: `SensSolve` / `Solver` pin-constraint indices
are now mapped to KKT rows through the equality/inequality split
(`full_g_to_c_block`), so pins are selected correctly when inequality
constraints precede them in `g(x)` (previously the wrong row was used
silently; the CLI sIPOPT path already mapped correctly and now shares the
same helper). `pounce.curve_fit` no longer requires scaling to be off to
trust the converged factor for its covariance / data-sensitivity reads.

### Changed — C ABI: sensitivity entry points now return natural units (breaking)

Behavior change for C callers of the sensitivity ABI (`pounce-cinterface`).
`IpoptSolverReducedHessian`, `IpoptSolverParametricStep`, and
`IpoptSolverKktSolve` now return values in **natural (unscaled) units** as part
of the #128 fix above — previously, on a badly-scaled NLP (where the default
`gradient-based` method fires), they returned the IPM's internally scaled
values. A C caller that was compensating for the old behavior — e.g. passing a
non-`1.0` `obj_scal` to `IpoptSolverReducedHessian` to undo the `df / dc²`
factor by hand — will now get a doubly-corrected (wrong) result and must drop
that workaround; `obj_scal` is once again only the plain extra multiplier its
docs describe. Callers that want the old scaled values back have an escape
hatch **only** for the raw KKT solve: `IpoptSolverKktSolveScaled(..., scaled =
true)`. There is intentionally no scaled variant of `IpoptSolverReducedHessian`
or `IpoptSolverParametricStep` — the natural-units reduced Hessian and
parametric step are the only correct answers for a covariance / predictor read,
so the scaled forms are not re-exposed across the ABI (the Rust
`Solver::compute_reduced_hessian_scaled` remains for in-process calibrated
callers).

### Added — Batched NLP solving (`solve_nlp_batch`) (#126)

Solve N independent NLPs in parallel on a Rayon pool — the general-NLP
analog of `solve_qp_batch_parallel`, for parametric sweeps, multi-start,
MPC chains, and branch-and-bound node relaxations (each sibling node
differing only in tightened bounds).

- **Rust** — `pounce_algorithm::solve_nlp_batch` /
  `solve_nlp_batch_parallel`: one fully-equipped `IpoptApplication` per
  instance, built *inside* the worker via a `Sync` configure hook that
  receives the instance index (outer-parallel / inner-serial, like the
  QP batch; `install_serial_feral_backend` sets up the per-worker
  serial factor). Results return in input order with the captured
  final iterate, multipliers, and per-instance `SolveStatistics`.
- **`pounce-nl`** — CSE `Expr` sharing switched from `Rc` to `Arc`, so
  `NlProblem` / `NlTnlp` are `Send` and an owned evaluator can move to a
  worker. `NlTnlp` is now `Clone`, and `NlTnlp::variant` /
  `NlVariation` build per-instance bound / starting-point overrides on
  one parsed model (tapes are cheap to clone).
- **Python** — `pounce.solve_nlp_batch(problems, x0s=, options=,
  parallel=, warms=, share_structure=)`: native `NlProblem` inputs
  (from `read_nl` or the new `NlProblem.variant(...)`) solve in
  parallel with the GIL released. One `(x, info)` pair per input,
  `info` matching `Problem.solve`'s layout; `print_level` defaults to
  0 for the batch.
- **Phase 2: parallel callback batching** — callback-based
  `pounce.Problem` inputs also solve in parallel: each instance's
  bridge (Python callables + pre-resolved sparsity) moves to a rayon
  worker that owns the whole solve, re-acquiring the GIL transiently
  per `eval_*` callback. The GIL serializes only the Python share, so
  the speedup scales with the Rust/Python work ratio (~4x on 4 cores
  for an n=800 banded NLP with NumPy-vectorized callbacks; tiny
  callback-dominated problems won't speed up). Per-instance
  `add_option` settings are honored, with `options=` as a batch-level
  overlay; a raising callback degrades to that instance's failure
  without poisoning the batch.
- **Warm-started batches** — `solve_nlp_batch_parallel_warm` /
  `solve_nlp_batch_warm` (Rust) and `warms=` (Python, both input
  kinds): seed each instance from a previous result's iterate + duals
  and thread the converged barrier μ into `mu_init`
  (`warm_start_init_point=yes` forced; dimension mismatch falls back
  to a cold start). Re-solving a perturbed 24-instance `.nl` sweep
  warm cut total iterations 482 → 120.
- **Identical-sparsity structure sharing** — `FeralBackendPool` /
  `install_pooled_serial_feral_backend` (Rust) and
  `share_structure=True` (Python): opt-in per-worker backend pooling
  so FERAL's pattern-fingerprint symbolic cache (ordering + supernode
  structure) carries across batch instances instead of being rebuilt
  per instance. Always correct (pattern changes re-analyze); results
  are within solver tolerance of — not guaranteed bit-identical to —
  fresh-backend solves, which is why it is opt-in. Cross-*thread*
  symbolic sharing stays future work (needs the `BackendPool`
  ownership refactor documented in `dev-notes/backend-pool-resolve.md`
  or a feral-side symbolic export API).

### Added — PyTorch frontend for the differentiable solver (`pounce.torch`)

A PyTorch frontend mirroring `pounce.jax`: a solve is a
`torch.autograd.Function` you can drop inside a learned model and backprop
through, with the same constraint-satisfaction guarantee. This is a thin
adapter, not a second solver — the Rust IPM core and the
implicit-function-theorem backward are framework-agnostic; only the array
namespace differs. Because PyTorch is eager, the adapter is smaller than the
JAX one (no `pure_callback` / `ShapeDtypeStruct`, no host-callback registry or
single-thread executor pin), and float64 is requested per tensor rather than
via a global flag.

Surface (parity with `pounce.jax`):

- `from_torch` — build a `Problem` from `torch.func`-traced `f` / `g`
  (`grad` / `jacrev` / `jacfwd` / `hessian`; CPR colored AD for `sparse=`,
  via the shared detection/coloring helpers now in `pounce._ad_common`).
- `solve` / `solve_with_warm` — `autograd.Function` + KKT implicit-diff
  backward, with dual + barrier-μ warm-start threading.
- `vmap_solve` / `vmap_solve_parallel` — sequential / threadpool batches.
- `TorchProblem` — build-once handle with k_aug-style factor-reuse backward
  (`Solver.kkt_solve_many`), stacked block-diagonal batched solve, and the
  anchor / sensitivity / jvp_from_state / vjp_from_state / active_set_margin
  post-solve API.
- `solve_qp` / `solve_qp_batch` / `solve_socp` / `QpLayer` — OptNet-style
  differentiable conic layers (feasible-by-construction).
- `PathFollower` / `inverse_map_rhs` — predictor–corrector path following.

PyTorch is an optional dependency (`pip install pounce-solver[torch]`,
torch ≥ 2.2). See the [PyTorch integration guide](docs/src/python.md).

### Added — Convex QCQP auto-routes to the conic (SOCP) solver

The `auto` router now recognizes a convex **quadratically-constrained QP** and
sends it to the `pounce-convex` conic interior-point solver instead of the
general NLP path. Each convex-quadratic inequality `½xᵀHx + aᵀx + b ≤ 0`
(`H ⪰ 0`) is reformulated to one second-order cone (`H = FᵀF`, via a pivoted
rank-revealing Cholesky so a rank-deficient `H` yields the minimal cone), solved
alongside the QP objective and linear constraints, and its dual is mapped back
to a per-constraint multiplier. Works on both surfaces:

- **CLI** — a convex-QCQP `.nl`/Pyomo model routes automatically; force it with
  `solver_selection=socp` (errors if the problem is not a convex QCQP).
- **Python** — `minimize()` probes each constraint's Hessian at an anchor plus
  held-out points, validates the fitted quadratic before trusting it, and routes
  only when the feasible set is provably convex (a scipy `ineq` `g(x) ≥ 0` must
  be concave); otherwise it falls back to NLP. `options={"solver_selection":
  "socp"}` forces the conic path.

This closes the long-standing "conic solver: future" gap in the routing docs —
the conic solver shipped in 0.4.0 but was not reachable from either router for
quadratic *constraints*. See [LP / QP Solver Routing](docs/src/lp-qp-routing.md)
and [Choosing a Solver](docs/src/choosing-a-solver.md).

### Added — CLI knobs for the convex IPM and active-set QP solvers (#134)

Previously hard-coded solver defaults are now registered CLI options:

- **Convex IPM** (`solver_selection=lp-ipm` / `qp-ipm` / `socp`): `qp_tau`,
  `qp_reg`, `qp_infeas_tol`, `qp_hsde`, `qp_equilibrate`, `qp_crossover`.
  Each is forwarded only when explicitly set, so engine defaults are otherwise
  preserved.
- **Active-set QP** (`solver_selection=qp-active-set`): `sqp_qp_feas_tol`,
  `sqp_qp_opt_tol`, `sqp_qp_max_iter`, `sqp_qp_elastic_gamma`,
  `sqp_qp_anti_cycling` (`expand` / `bland` / `none`).

Both families are documented in `dev-notes/lp-qp-routing.md`. The LP crossover
default is now **off** (opt in with `qp_crossover=yes`): crossover-on regressed
the LP suite 3×–800× without reaching an exact vertex on the GEN family — the
machinery is unchanged, only the default flips.

### Fixed — `solver_selection=socp` was rejected; debugger silently no-op'd on `qp-active-set`

Two reachability gaps around the new convex solvers, found while confirming the
interactive debugger works on every backend:

- `solver_selection=socp` was a documented value (the conic IPM that a convex
  QCQP reaches under `auto`) but was missing from the option's registered
  allow-list, so forcing it failed with `Invalid value "socp"`. It is now
  accepted and routes a convex LP/QP/QCQP to the SOCP conic IPM.
- The interactive debugger (`--debug` / `--debug-script` / `--debug-json`) is a
  pdb-for-the-IPM. It engages correctly on `lp-ipm`, `qp-ipm`, and `socp`, but
  the active-set SQP engine (`qp-active-set`) has no such hook, so a debug
  request there used to run to completion without ever pausing — a silent
  no-op. It now prints an explicit note that the debugger is IPM-only and points
  at `qp-ipm` for interactive convex-QP debugging.

### Added — exact-vertex LP crossover (revised simplex)

`pounce-convex` gains a revised-simplex crossover that purifies a near-optimal
interior LP iterate to an exact optimal vertex, pivoting one variable at a time
on feral's unsymmetric sparse LU with Bland's anti-cycling rule (so it walks
through the highly degenerate NETLIB GEN vertices where the active-set bridge
stalls). It is tried first and accepted only when the KKT error does not
regress; on any breakdown it falls back to the legacy active-set bridge. Opt-in
via `qp_crossover=yes`.

### Fixed — convex LP/QP robustness on rank-deficient and large-scale data

Interior-point hardening in the dedicated convex (`lp-ipm` / `qp-ipm`) path:

- **Adaptive equality-block regularization (δ_c)** (#133). The
  equality-multiplier `(y,y)` block was frozen at a static value; on
  rank-deficient equality Jacobians that leaves a near-singular saddle and the
  solve plateaus to max_iter. δ_c now seeds from a μ-scaled base
  (`1e-8·μ^0.25`) and escalates on singular factorization / wrong KKT inertia /
  un-refinable direction probes, resetting each iteration so one hard iterate
  never inflates regularization for the rest of the solve. Regression-clean on
  NETLIB; drops the rank-deficient GEN constraint violation 9.3e-5 → 3.1e-8.
  (GEN still floors just above the 1e-8 tolerance, so #133 stays open.)
- **Scale-gated relative stopping + ratio-based infeasibility ray**. The
  absolute KKT stop is unreachable once the data scale pushes the
  finite-precision residual floor above tol — POWELL20 / BOYD1 / BOYD2 /
  QFORPLAN / QSHELL (scale 7e9–4e12) ran to max_iter despite being optimal to
  ~1e-9 relative. A Clarabel-style scale-relative residual now relaxes the
  absolute test, but only once roundoff sits below tol, so well-scaled problems
  are unaffected. Infeasibility now triggers on the ratio κ/τ→∞ rather than a
  bare τ floor, fixing a false `PrimalInfeasible` on feasible large-norm-x QPs.

### Fixed — active-set QP cycling on degenerate phase-1 (#133)

The active-set QP elastic phase-1 recovery (ℓ₁-infeasibility minimization,
γ=1e6) is inherently highly degenerate and could cycle under the default
anti-cycling rule (NETLIB `afiro` bailed at iteration 0). The phase-1 solve now
runs under Bland's rule (provably finite), and `solve_general` latches into
Bland after 50 consecutive non-improving iterations as a sticky,
scale-invariant anti-stall safety net.

### Changed — large sparse convex QPs are now recognized as convex (faster routing)

Problem classification certifies a coupled quadratic Hessian as PSD via feral's
sparse LDLᵀ inertia (~O(nnz·fill)) instead of a dense Jacobi eigensolve, so
large-but-sparse convex QPs (the CVXQP family, n≈1000) that previously fell back
to the general NLP solver are now sent to the dedicated convex path.

### Added — limited-memory update type & history honored on the IPM path (#131, #132)

`limited_memory_update_type` and `limited_memory_max_history` were registered
but read nowhere on the interior-point path (the updater was hard-wired to
Powell-damped BFGS). Both are now threaded through to the limited-memory
updater. The default is unchanged (`bfgs`, history 6 — bit-exact with Ipopt), so
there is no behavior change unless set; `sr1` (which can represent negative
curvature) is now selectable and rescues ill-conditioned nonconvex objectives
where damped BFGS hides indefiniteness from the inertia check.


## [0.4.0] — 2026-06-05

### Added — Convex / conic solver (`pounce-convex`; `solve_qp` / `solve_socp`)

POUNCE is no longer NLP-only: a new pure-Rust convex interior-point solver
(`pounce-convex`) handles **LP, convex QP, SOCP, and PSD / exp / power cones**,
solving each to a **global** optimum (a convex problem has no other kind). It
uses a homogeneous self-dual embedding (HSDE) — symmetric for the self-dual
cones and a non-symmetric driver for the exponential/power cones — over a
`Cone` abstraction (`nonneg`, `soc`, `psd`, `exp`, `power`, plus composite and
chordal decompositions for sparse SDPs). Convex solvers extract the constant
`P`, `A`, `c`, `b` data once at setup rather than re-evaluating per iteration,
and share the `pounce-linsol` / `pounce-linalg` factorization substrate with the
NLP path. Python entry points are typed (not SciPy-shaped, by necessity — a cone
program is *data*, not a callable): `solve_qp(P, c, A, b, G, h, lb, ub, …)`,
`solve_socp(…, cones=…)`, plus `solve_qp_batch` / `solve_qp_multi_rhs` for
batched factor reuse, and a reduced-Hessian sensitivity API. The CLI reads conic
instances from CBLIB / `.cbf` (including PSDCON / HCOORD / DCOORD SDP blocks).

### Fixed — Convex LP/QP reported objective dropped tree-folded constant

The convex LP/QP path (`solver_selection=lp-ipm` / `qp-ipm`) reported an
objective off by the objective's constant term whenever AMPL/Pyomo folded that
constant into the **nonlinear objective tree** (the `+9` of `(x-3)²`) rather
than the `.nl` linear-section constant. The quadratic-form extractor
(`analyze_quadratic_full`) discarded the degree-0 term — correct for the
*minimizer*, wrong for the *reported value* — so e.g. `HS21` reported `0.04`
instead of `−99.96` and `HS35` `−8.889` instead of `0.111`. The extractor now
returns that constant and the convex driver adds it to the reported objective
alongside `obj_constant`; the optimal point was always correct. Caught by a
head-to-head NLP-vs-convex run over the Maros-Mészáros QP and NETLIB LP suites
(`benchmarks/nl_compare_nlp_vs_convex.md`).

### Fixed — Convex LP/QP IPM stalled on badly-scaled NETLIB LPs

The static KKT regularization `δ` (added on the reduced KKT diagonal so the
LDLᵀ has a stable inertia) was `1e-8`, large enough to **floor the achievable
primal residual** at `δ·‖dy‖`: with a full Newton step `A·dx = −r_p + δ·dy`, so
on instances with large equality multipliers the primal infeasibility cannot
fall below `δ·‖dy‖`. On NETLIB `adlittle` (`‖dy‖ ≈ 4e8`) this froze `inf_pr`
near 4 and the LP IPM ran to its iteration cap, returning a wrong objective
(`439665` vs the published `225494.96`). Lowering the default `δ` to `1e-10` —
still strictly positive, so the system stays quasi-definite — clears the floor:
`adlittle` now converges in ~57 iterations to the optimum, `stocfor1` speeds up
(139 → 71 iters), and the rest of the LP/QP suites are unchanged (the QP suite
is bit-identical). The whole `1e-9‥1e-11` band converges the benchmark suites;
`1e-10` is centered in it.

Also: the convex IPM's opt-in iteration trace now records a **terminal record at
the converged iterate** (the NLP path's N+1 convention), so the trace always
ends at the optimum instead of at the last pre-step state — previously a solve
that converged in a single step left only the cold-start record in the trace.

### Added — SOS polynomial global optimization (`sos_minimize`)

`sos_minimize(objective, *, inequalities, equalities, …)` computes **certified
global** lower bounds for polynomial optimization via a sum-of-squares /
Lasserre relaxation (Putinar localizing multipliers for constraints), built on
the new PSD cone. When the relaxation is exact it extracts the global
minimizer(s) with an exactness certificate (multi-atom extraction without a
non-symmetric eig, plus facial reduction for degenerate solves).

### Added — Multi-backend interactive debugger (convex/conic IPM)

The interactive debugger was generalized over a `DebugState` trait so one REPL
drives the convex solver as well as the NLP loop. New backend: a
**convex/conic** debugger (`pounce_cblib --debug`, wired through the symmetric
and non-symmetric HSDE drivers), exposing the same checkpoints and commands as
the NLP path. This composes with the 0.4.0 debugger features below (quote-aware
tokenization, `ask` provider presets, `--debug-json` protocol, Ctrl-C escape
hatch).

### Added — `pounce.curve_fit` (Python)

A `scipy.optimize.curve_fit`-style nonlinear fitter on top of the
interior-point solver, returning much more than `(popt, pcov)`:

- parameter covariance, standard errors, and Student-t confidence intervals
  read pounce-natively from the converged factor's reduced Hessian
  (`pcov = 2·s²·inv(H_S) = s²·(JᵀJ)⁻¹`; matches scipy / `pycse.nlinfit`). The
  t-quantiles use scipy when present and an accurate scipy-free inverse-t
  (via the inverse regularized incomplete beta) otherwise, so the CIs are
  correct on a numpy-only install even for small samples;
- a smooth (C²) loss family — ordinary/weighted least squares plus robust
  Cauchy and a smooth pseudo-Huber, exposed under both `soft_l1` and `huber`
  (the same C² loss: a true piecewise Huber is only C¹, which the IPM can't
  use), with a sandwich covariance estimator (non-smooth L1/MAE is
  intentionally out of scope for the IPM);
- parameter constraints scipy can't express — positivity/negativity/ranges
  via `bounds`, and relations between parameters via `constraints=`; an active
  bound/constraint yields a covariance projected onto the free subspace;
- data sensitivity `dpopt/ddata` (∂params/∂data) from a single batched
  back-solve against the same factor (`Solver.kkt_solve_many`);
- a `CurveFitResult` with `predict()`, `confidence_band()` (both `confidence`
  and `prediction` kinds, heteroscedastic-aware), `correlation`, R²/χ²/dof,
  and `summary()`.

Derivatives resolve analytic `jac` → JAX autodiff (the default for
`jax.numpy` models) → a finite-difference fallback; exact derivatives let the
solve converge cleanly with scaling off, which is what makes the
factor-based covariance and sensitivity exact. Docs:
`docs/src/curve-fitting.md`; notebook `python/notebooks/18_curve_fit.ipynb`.

`p0` is now optional even without bounds: when omitted, the parameter count is
read from the model signature and the starting point is chosen data-drivenly
(a bound-aware, data-scale candidate sweep scored by the objective) instead of
defaulting to a flat vector of ones — so badly-scaled problems get a far better
seed, while `ones` (clipped into the bounds) is always among the scored
candidates so the choice is never worse than the old default.

### Added — `pounce.curve_fit_minima` (Python)

`curve_fit_minima` finds **multiple** parameter sets that each explain the
data, for the non-convex problems where one fit isn't the whole story
(peak-assignment ambiguity, frequency aliasing in sinusoids, amplitude/decay
trade-offs in sums of exponentials, sign/label symmetry, …).

- drives `pounce.find_minima` over the *very same* fitting objective as
  `curve_fit` — identical `sigma` weighting, robust `loss`, `f_scale`,
  `constraints`, and resolved Jacobian — so the enumerated minima are true
  optima of the actual fit, not a separate surrogate;
- reuses the model Jacobian as the search **gradient** and the Gauss-Newton
  matrix as the search **Hessian**, which sharpens the basin escapes and lets
  `find_minima` certify each point as a minimum (rejecting saddles);
- refines every distinct minimum into a full `CurveFitResult` (covariance,
  CIs, optional `dpopt/ddata`) and returns them ranked by SSE, best first;
- the `method`, `n_minima`, `max_solves`, `patience`, `dedup`, `seed`, and
  `find_minima_kw` arguments pass straight through to `find_minima`; finite
  `bounds` define the box it samples / repels within. Docs:
  `docs/src/curve-fitting.md`.

### Added — `pounce verify` subcommand + signed receipts

A `verify` subcommand that re-derives feasibility from the canonical `.nl`
rather than trusting a `.sol`'s status line or the solver/agent that produced
it — the trust anchor when pounce is a tool an agent calls: the agent
proposes a solution, a small deterministic checker disposes.

- `pounce verify <problem.nl> <claim.sol>` evaluates `g(x*)` and bounds
  against the canonical model, reporting the worst constraint/bound violation
  and (when the `.sol` carries duals) a bound-projected KKT stationarity
  residual. Exit 0 = VERIFIED, 20 = REJECTED, 2 = usage/IO. Feasibility
  gates; optimality is informational unless `--require-optimal`.
- The JSON receipt content-addresses both inputs by SHA-256 (zero new deps);
  with `POUNCE_VERIFY_KEY` set it signs the receipt with HMAC-SHA256 over a
  float-free preimage so any language can re-derive it.
- MCP `verify_solution` tool plus dependency-free `verify_sig` helpers and a
  stdlib reference signer service.

The check itself (recompute feasibility against the model + a content-addressed
receipt) is ready to use and needs no secrets; the signing / remote-authority
layer is an explicit proof of concept. Docs: `docs/src/verify.md`.

### Added — Debugger `load` / `sweep` / `multistart`

The interactive solver debugger gained three commands for seeding solves
from externally-computed points and for initialization-sensitivity
diagnostics:

- `load <file> [block]` — the inverse of `save`. Reads a block (default
  `x`) into the live iterate from either a `save` artifact (JSON; every
  block present is loaded) or a plain numeric file
  (comma/whitespace/newline-separated). The many-variable escape hatch:
  generate a start once (`numpy.savetxt`) and `load` it instead of typing
  it. A loaded `x` becomes the seed for the next step / `resolve`.
- `sweep <file>` — run one full solve per start in a file (one per line),
  then tabulate each terminal status / objective, count distinct minima
  (objectives clustered to a relative `1e-6`), and flag the best solve.
- `multistart <N> [rel]` — `N` solves from sampled restarts: each variable
  with a finite box `[x_Lᵢ, x_Uᵢ]` is drawn **uniformly in that box**;
  unbounded variables fall back to a relative jitter `±rel·(|xᵢ|+1)`
  around `x`. Start 0 is the unperturbed point; deterministic (fixed-seed
  PRNG), so runs reproduce. Backed by a new `DebugCtx::var_bounds()` that
  reconstructs full-length algorithm-space bounds (post-scaling, with `±∞`
  for absent bounds) from the NLP's reduced bound vectors + expansion
  matrices.

Tab completion now also covers **filesystem paths** (after
`load`/`sweep`/`save`/`source`, with a trailing `/` on directories) and
block names for `load`'s optional second argument — available both at the
REPL Tab key and via the programmatic `complete` command.

**Ctrl-C at the prompt** is now a working escape hatch: the first press
cancels the current input line (readline convention), a second in a row
stops the solve (a clean `UserRequestedStop`) — mirroring the running-mode
double-tap, so two Ctrl-Cs always exit whether running or paused.

And a little something for the 2am debugging sessions: an undocumented
`coffee` command at the prompt. ☕

Both sweep commands build on the existing re-solve machinery and keep each
solve's trajectory observable (breakpoints/events still fire inside a
sweep). JSON mode emits `sweep_result` per solve and a final
`sweep_summary`; `hello.capabilities` advertises `load` and `sweep`. For
automated global search with dedup and minimum certification, the Python
`find_minima` remains the production path. Docs: `docs/src/debugger.md`
(new "Multi-start and initialization sensitivity" section + scripting
examples).

### Added — Sparse (colored) AD for the JAX front-ends (`sparse=`)

`from_jax` and `JaxProblem` gained a `sparse=True` flag that computes the
constraint Jacobian and the Lagrangian Hessian with CPR-style colored AD
— one JVP/HVP per color (`k ≪ n` colors) scattered back to the detected
nonzeros — instead of materializing the dense matrix and slicing it
(pounce#83). Per-iteration derivative cost drops from `O(n)` to `O(k)`
AD passes on genuinely sparse problems; benchmarked on a banded family at
~560× (Jacobian) / ~200× (Hessian) per eval and 7.6× faster full solve
by `n=2000`. When the sparsity pattern is **value-independent** (any
composition of smooth pointwise ops) the reported structure, values, and
solutions are identical to the dense path; the differentiable backward is
unaffected. For **value-dependent** structure (`where` / `abs` / branches) a
random probe can miss a nonzero, and under compression a missed entry aliases
into a same-colored reported entry — silently wrong derivatives — so such
models should hand-specify the pattern via the `Problem` API or stay on the
dense path. Dense problems see a small bounded overhead, so the flag is opt-in.

- Forward/reverse mode selection (`jacfwd` when `n < m`, else `jacrev`)
  for the dense path / sparsity probe.
- Multi-probe sparsity detection (`n_probes=`, default 3 under
  `sparse=True`, 1 otherwise) unions several random probes to harden
  against value-dependent structure.
- Benchmark: `python/benchmarks/bench_sparse_ad_83.py`. Docs:
  `docs/src/python.md` (JAX integration → "Sparse Jacobian/Hessian
  compression").

### Added — Interactive solver debugger (`--debug` / `--debug-json`)

A "pdb for the interior-point loop." `pounce <problem> --debug` opens a
branded REPL that pauses the solve to inspect and **mutate** live state;
`--debug-json` speaks a newline-delimited JSON protocol so an LLM agent,
a script, or a visual debugger (VS Code DAP / webview) can drive it.
Full guide in `docs/src/debugger.md`. Zero effect on the solve when not
attached.

- **Checkpoints & stepping:** pauses at `iter_start`, the sub-iteration
  phases (`after_mu` / `after_search_dir` / `after_step`), `step_rejected`
  (line search gave up, before restoration), around restoration
  (`pre_/post_restoration_entry/exit`), and `terminated`.
  `step` / `stepi` / `continue` / `run N` / `stop-at <cp>` / `detach` /
  `quit`. The same debugger **steps into the restoration inner IPM**
  (pauses flagged `in_restoration`).
- **Breakpoints:** by iteration (`break N`, one-shot `tbreak N`),
  conditional with `&&`/`||`, on a solver **event** (`break on
  regularized|resto_entered|tiny_step|ls_rejected|mu_stalled|nan`), and
  **watchpoints** (`watchpoint x[3]`). `commands N …` auto-runs a list on
  hit.
- **Inspect:** `info`; `print` of blocks, search-direction blocks (`dx`),
  scalars (`mu obj inf_pr inf_du err compl iter`), `kkt` (inertia +
  regularization), and `active`; `watch`/`display`; `diff`.
- **Named-equation diagnostics:** `print residuals` labels primal/dual
  residuals with their original `.nl` constraint/variable names; `print
  equation <name|row>` renders the source algebra of a named constraint
  (by model name or `.nl` row index); `print rank` reports the SVD
  numerical rank of the equality Jacobian J_c and names the implicated
  rows. `diagnose` (alias `diag`) runs a panel of heuristics over the
  current iterate and emits a **named** health report — *"the worst
  constraint residual is c[mass_balance]"* rather than *"row 13 is
  infeasible"* — the live counterpart to the `pounce-studio` `diagnose`
  tool.
- **Mutate / what-if:** `set mu`, `set x[i]`, `set opt`; `goto`/`restart`
  (soft rewind) and `resolve` (re-solve from the current point).
- **Visualize:** `viz kkt`/`viz L`/`viz <block>` open via `pounce-dbg-viz`
  — an interactive Plotly viewer (spy/heatmap for the KKT matrix & LDLᵀ
  factor, bars for vectors); `save` dumps the iterate. `pip install
  'pounce-solver[viz]'`.
- **Attach & drive:** `--debug-on-error` (post-mortem), `--debug-on-
  interrupt` / Ctrl-C / in-band `{"cmd":"pause"}` (async pause),
  `--debug-script` / `source`, option discovery + Tab completion, `ask`
  (consult an LLM about the paused state; provider-selectable via
  `$POUNCE_DBG_LLM` = `claude` / `codex` / `gemini` / `llm` or a custom
  command template, default Claude Code), and a branded REPL banner
  reusing the project wordmark with a command cheat-sheet.
- **JSON protocol:** `hello` → `pause` → `result` (with `request_id`) →
  `progress` → `terminated`. Engine in `pounce-algorithm::debug`; front
  end in `pounce-cli::debug_repl`.
- **MCP live-debug proxy:** `pounce-studio` exposes the debugger over the
  Model Context Protocol (`debug_start` / `debug_command` / `debug_state`
  / `debug_sessions` / `debug_close`), proxying the `--debug-json`
  protocol so an MCP client can start, drive, and inspect a live solve.

### Added — `read_nl` / `NlProblem` (Python)

`pounce.read_nl(path)` loads an AMPL `.nl` file through pounce's own reader
and returns an `NlProblem` exposing the model's `objective`, `gradient`,
`hessian`, and constraint `jacobian` at any point — the same evaluation
pipeline the solver uses, available standalone for inspection, finite-
difference checks, or feeding another tool. Exported from `pounce`
(`read_nl`, `NlProblem` are in `__all__`).

### Added — Expanded `.nl` opcode coverage

The `.nl` reader now handles conditional/logical opcodes (`if-then-else`,
comparisons), the n-ary list reducers `o11` (MINLIST) / `o12` (MAXLIST), and
the remaining smooth transcendentals (inverse and hyperbolic trig). Models
that previously failed to load with an "unsupported opcode" error now parse,
with FD-verified first/second derivatives on the smooth interior.

> `min`/`max`/`if-then-else` are **non-smooth**: at a kink the gradient is a
> subgradient and the Hessian misses the kink curvature, so an iterate landing
> on or oscillating across the switch can stall the interior-point solve. The
> inverse-trig opcodes (`asin`/`acos`/`atanh`/`acosh`) have **bounded domains**
> whose derivatives blow up at the edge — bound such variables away from the
> boundary. The reader accepts these models; convergence is on you.

### Added — `pounce --cite` and `--minima`

- `pounce --cite [REPORT.json]` lists the citations to use for pounce (and,
  when a solve report is given, any method-specific references it triggered,
  e.g. the Byrd restoration paper). `--bibtex` emits ready-to-paste entries.
- `pounce <problem> --minima` runs the multistart global search from the CLI
  with full `find_minima` parity (method, `n_minima`, dedup, seed).

### Changed

- **Default solver trajectory** moved on several fronts as the interior-point
  method was brought closer to IPOPT. These change which iterates are visited
  (and, on a few problems, the iteration count) but not the math being solved:
  - the barrier parameter `μ` is now updated *inside* the monotone reduction
    loop, so the relaxed-complementarity error reflects the current `μ`. Net
    +2 problems reach Optimal on the internal `.nl` sweep, at a ~2.7% total
    iteration-count cost and a regression on `deconvb` / `gausselm`;
  - under the watchdog, the line search bypasses the acceptor's `alpha_min`
    floor (mirrors IPOPT) so the full-step watchdog trial actually runs;
  - the IPOPT safe-slack bound-adjustment mechanism (`slack_move`) is ported
    and active by default;
  - NLP gradient-based scaling now lifts fixed variables to their value before
    sampling, so the computed scale factors match the operating point.
- **Auto-retry on local infeasibility (default on).** New option
  `feral_infeasibility_scaling_retry` (default `yes`): when a solve ends in
  `Infeasible_Problem_Detected` under a non-MC64 effective scaling, pounce
  re-solves once with `feral_scaling=mc64` (main IPM and restoration sub-IPM).
  This rescues problems where a backward-stable scaling choice lands in a
  spurious infeasible basin under sensitive dependence (`discs.nl` is the
  canonical case); every individual solve along both trajectories is itself
  backward-stable, so an a-priori scaling router can't distinguish them. Set
  to `no` to restore the single-solve behavior.
- **New option `feral_scaling`** (default `auto`, mirrors `feral_ordering`):
  pins FERAL's diagonal KKT scaling strategy; also settable via the
  `POUNCE_FERAL_SCALING` env var.
- **Dependency:** `feral` pinned to crates.io `0.10.0` (was a git rev),
  bringing AMF ordering by default and MC64 inertia-guided scaling fallback.
- **Internal:** the `.nl` pipeline was extracted into a new leaf crate
  `pounce-nl` (re-exported from `pounce-cli`; no public API change).
- **`pounce-studio-mcp` → 0.1.0** (versioned independently of the `0.4.0`
  core): the MCP server graduated from its `0.0.1` spike to its first
  functional release — analyze / run / explain / citations tools, GAMS
  problem tools, a live debug-session proxy, and PyO3 backing via
  `pounce-studio-core`.

### Fixed

- **Windows build:** the debugger's `SIGINT`-to-break handler referenced
  `nix::sys` / `nix::libc`, which the (Unix-only) `nix` crate does not expose
  on Windows, breaking the `pounce-cli` build there. The POSIX handler is now
  `#[cfg(unix)]`-gated with a no-op `install()` stub elsewhere; the rustyline
  prompt's Ctrl-C double-tap remains the cross-platform escape hatch.
- **`.sol` banner no longer goes stale:** the `parse_sol` round-trip test
  fixture derived its `POUNCE <version>:` message from a hardcoded literal,
  which silently drifted on each release (it was still `0.3.1`). It now
  reads `CARGO_PKG_VERSION`, like the production writer always has, so the
  fixture self-updates and never needs a manual bump.
- **Restoration:** the limited-memory (L-BFGS) Hessian is now built in the
  iterates' native space, fixing a space mismatch on compound problems (#102);
  the cycle detector rolls back to the last acceptable point instead of
  erroring out when a usable iterate exists.
- **KKT:** the negative-eigenvalue cache is refreshed on `WrongInertia` /
  `Singular` outcomes (not only `Success`), matching IPOPT's inertia
  pass-through so δ_c regularization routing stays live near a singular KKT (#99).
- **`find_minima`:** the in-bounds test uses a bound-magnitude-relative
  tolerance so large-scale boxes aren't spuriously rejected (#101); MLSL is
  bounded by a sample budget so it always terminates instead of looping when
  its clustering filter rejects every sample (#103).
- **Bounds length is validated up front** across `minimize`, `find_minima`,
  `find_saddles`, `find_critical_points`, `reaction_network`, and `curve_fit`.
  A `bounds` list whose length didn't match the variable/parameter count used
  to fail silently — a too-short list left trailing variables unbounded, and in
  the sampling-based searches a length-1 box could *broadcast* across every
  dimension (sampling all of them from variable 0's interval). It now raises a
  clear `ValueError` immediately, like scipy; `curve_fit`'s scipy-style
  `(lo, hi)` tuple form is likewise checked so array sides must be scalar or
  length-`n_params`.
- **Input validation hardened** so imperfect-but-plausible arguments raise a
  clear `ValueError` up front instead of failing cryptically deep in the solve:
  - `minimize` / `find_minima` / `find_saddles` now **promote a scalar / 0-d
    `x0` to 1-D** (matching scipy), so `minimize(f, 1.5)` works instead of
    raising `iteration over a 0-d array`;
  - a **reversed bound** (`low > high`) is rejected instead of silently
    producing an infeasible box (a fixed `low == high` is still allowed);
  - **malformed constraint dicts** (not a dict, or missing `type` / `fun`, or a
    non-callable `fun`) raise a descriptive error instead of a bare `KeyError`;
  - `curve_fit` validates its data and weights: `xdata`/`ydata` length must
    match and be non-empty and finite, `sigma` must be positive and finite,
    `f_scale` must be positive and finite, and an explicit `p0` must have one
    start per model parameter — each previously surfaced as a `LinAlgError`,
    `ZeroDivisionError`, back-solve `RuntimeError`, broadcast error, or a
    silently wrong fit;
  - a model with **keyword-only parameters** (`f(x, *, a, b)`) — which
    `curve_fit` cannot call positionally as `f(x, *params)` — is rejected with
    a clear message instead of a downstream `TypeError`;
  - `CurveFitResult.confidence_band` checks that `x` has the same
    dimensionality as the fitted `xdata` and that a prediction-band `sigma` is
    scalar or matches `x`, replacing a cryptic einsum/broadcast error;
  - `find_minima` / `find_saddles` reject a sub-1 `n_minima` / `n_saddles` /
    `patience` / `max_solves`, and `find_saddles` rejects a Morse `index`
    outside `[1, n]` (which previously sliced the step vector wrong and found
    the wrong critical points).

## [0.3.0] — 2026-06-02

### Added — Multiple-minima & critical-point global search (PR #94)

`pounce.find_minima(fun, x0, n_minima=..., method=...)` returns several
distinct local minima from a single call. Methods: `flooding` /
`deflation` (add a repulsive Gaussian / pole "hump" — with analytic
gradient and Hessian — at each found minimum and re-solve), `multistart`,
`mlsl`, and `basinhopping`. Anisotropic auto bump widths and a
curvature-derived auto amplitude; Hessian-based saddle rejection; global
box restarts; bounds and constraints pass through `minimize` untouched.
The six-hump-camel demo recovers all six minima.

- `pounce.find_critical_points` / `pounce.find_saddles` — stationary
  points via the squared-gradient merit `½‖∇f‖²`, classified by Morse
  index from the Hessian eigenvalues (degenerate / non-Morse points are
  *flagged*, not mislabeled); eigenvector-following saddle search with
  box-clipped steps.
- `pounce.reaction_network` — minima, transition states, and barriers
  (Müller–Brown example).
- Robustness: non-finite candidates and objectives are rejected before
  acceptance; the de-duplication metric is the same per-dimension scaled
  distance across the minima and saddle routes.
- Examples (`gaussian_hump_minima.py`, `critical_points.py`,
  `reaction_barrier.py`), notebooks 15–17, docs (`find-minima.md`,
  `find-minima-choosing.md`), and 29 unit tests.

### Fixed

- **Acceptable-point termination now rejects a non-finite objective.** A
  near-feasible iterate whose objective evaluates to NaN/Inf (e.g. CUTE
  `himmelbj`) exits `Invalid_Number_Detected` instead of a spurious
  `Solved_To_Acceptable_Level` carrying a `nan` objective; the benchmark
  driver's objective scrape is null-safe to match.
- **No spurious `jacobian()` call on unconstrained problems**
  (`pounce-py`). `eval_jac_g` short-circuits when there are no constraint
  entries (mirroring the Hessian guard), so the unconstrained `minimize`
  facade — which legitimately omits `jacobian` — no longer logs an
  `AttributeError` at ERROR level on every iteration.

### Benchmarks & docs

- The benchmark suite now runs **single-threaded and sequential** by
  default (`OMP/VECLIB/RAYON=1`) so POUNCE and Ipopt solve times are
  directly comparable on one host; the ipopt-ma57 reference was
  regenerated and `BENCHMARK_REPORT.md` carries a threading/timing note
  (POUNCE's `faer`/`rayon` dense linear algebra is up to ~2× faster
  multi-threaded, so the single-threaded numbers are a controlled lower
  bound).
- Example notebooks re-executed against the current solver and corrected
  where the prose had drifted: warm-starting now demonstrates the
  `mu_init` + tight `warm_start_*_bound_push` tuning that actually cuts
  iterations (11 → 5 on HS071); the scaling example uses a `1e12`
  constraint where gradient scaling visibly helps (31 → 11 iters); the
  FBBT notebook shows the domain-safety and false-infeasibility wins.
- mdBook restyled in the POUNCE tiger / cream brand palette.

### Added — Inverse-map ODE recipe over a sensitivity RHS (pounce#91)

`pounce.jax.inverse_map_rhs(jp, dy_ds, *, output=None, x0=None)` builds
the right-hand side of the Alves–Kitchin–Lima inverse / uncertainty
mapping ODE (pounce#84, Eq. 3):

```
dθ/ds = (∂y/∂θ)^{-1} · dy/ds
```

where ``y = output(x*(θ), θ)`` is an output of the embedded optimizer.
POUNCE supplies the RHS; an off-the-shelf adaptive integrator (diffrax,
scipy) does the stepping — *no NLP inversion*.

- The inverse map is a **linear solve against** the total output
  sensitivity ``∂y/∂θ = (∂h/∂x) J + ∂h/∂θ`` (with ``J = ∂x*/∂θ`` from
  the held factor), not a Jacobian-vector product — so it wants the full
  ``J`` and ``jnp.linalg.solve``.
- The whole evaluation rides one `jax.pure_callback`, so the RHS is
  JAX-traceable and composes under `jax.jit` and diffrax (which
  jit-compiles the vector field).
- Worked example `python/examples/inverse_map_diffrax.py` integrates a
  closed output boundary with diffrax Dopri5 and round-trips back through
  the optimizer onto the boundary (~1e-7). `diffrax` is an optional
  extra (`pip install pounce[diffrax]`); the example falls back to RK4
  if it's absent.
- `inverse_map_rhs(..., warm=True)` warm-starts each inner solve from the
  previous evaluation's primal/duals/μ (pounce#86). Result-invariant (up
  to solver tolerance); a *modest* lever — ~1.4-1.7× fewer IPM iterations,
  ~1.3× wall-clock, roughly flat in problem size (interior-point
  warm-start ceiling + per-eval Jacobian-build overhead). Benchmark:
  `python/benchmarks/inverse_map_warm.py`. For a real speedup on a smooth
  map, prefer `PathFollower` (it skips solves, not just cheapens them).
- Switch to `PathFollower` when the path folds or the active set changes.
- Worked notebook `python/notebooks/14_path_following.ipynb` tours the
  whole family (sensitivity → margin → continuation → fold → inverse map).

Also fixes the build-once / stacked path for **unconstrained** problems
(``g=None``, ``m=0``): the constraint callbacks no longer dereference the
(``None``) constraint-Jacobian jit, so `JaxProblem.solve` /
`solve_with_jacobian` / the batched solves now work with no constraints.

### Added — Predictor–corrector path-following engine (pounce#90)

`pounce.jax.PathFollower` traces a solution path of a parametric NLP by
*composing* the post-solve sensitivity primitives instead of re-solving
at every step:

```python
from pounce.jax import PathFollower
pf = PathFollower(jp, monitor_tol=1e-6, ds0=0.05)
trace = pf.follow(theta_of_s, (0.0, 1.0), x0)   # parameter continuation
# trace.x, trace.theta, trace.s, trace.lam,
# trace.n_correctors, trace.n_accepts, trace.active_set_changes
```

- **predict** — extrapolate primal *and duals* along the held-factor
  sensitivity (`jvp_from_state(..., with_duals=True)`); **monitor**
  (no solve) — KKT residual + active-set margin (#89) at the predicted
  point; **correct** — only when the monitor trips, a warm-μ re-solve
  that also re-anchors the factor in one solve (`warm_anchor`, #86).
- Adaptive step size; detects and records active-set changes and
  re-anchors on the new active set.
- `PathFollower.trace_arclength(...)` — pseudo-arclength continuation for
  a scalar-parameter, equality/unconstrained family, tracing **past
  folds** where `∂x*/∂θ` is singular (parameter continuation cannot).
  Reports turning points. Bifurcation/branch-switching and
  inequality-active folds are out of scope for v1.
- On a linear-response NLP the predictor is exact, so the whole path is
  traced with **zero correctors** (one anchor solve vs one cold solve
  per step); nonlinear paths correct adaptively and still trace to
  tolerance.

New supporting public surface:

- `JaxProblem.warm_anchor(p, x0, *, duals=None, mu=None)` — a warm-started,
  μ-seeded re-solve that pins the converged factor and returns a `B=1`
  `AnchorState` (the corrector + anchor in one solve). Threads μ through
  the reusable build-once path (the #86 follow-up).
- `JaxProblem.jvp_from_state(..., with_duals=True)` /
  `batched_jvp_from_state(..., with_duals=True)` — also return the dual
  sensitivity `∂λ*/∂θ · dp` from the same held-factor back-solve.

### Added — Active-set-proximity monitor (pounce#89)

`JaxProblem.active_set_margin(state)` reports the distance to an
active-set change at the anchor point — the "predictor is about to
become invalid" signal for predictor–corrector path following. The
post-solve sensitivity is a derivative on a *fixed* active set; this
flags when a bound / inequality is about to cross its critical-region
boundary (where the sensitivity is discontinuous).

```python
r = jp.active_set_margin(state)
# r["margin"], r["min_mult"], r["min_slack"]  — each (B,)
```

- By complementarity: an **active** bound/inequality (multiplier `>
  active_tol`) is about to leave the set — its *multiplier* heads to
  zero; an **inactive** one is about to enter — its *slack* heads to
  zero. `min_mult` / `min_slack` track each; `margin = min(min_mult,
  min_slack)`.
- Equalities (`cl == cu`) are excluded (always active); `±inf` bounds
  and the slack side of a one-sided inequality drop out naturally.
  An unconstrained interior point returns `inf`.
- Pure-JAX reduction over state the `AnchorState` already holds — no
  solve, no back-solve. Pairs with the caller-side KKT-residual
  (smooth-drift) monitor: re-anchor when either trips.

### Added — Single-problem ergonomic sensitivity wrappers (pounce#88)

Thin un-batched wrappers over the `batched_*` post-solve sensitivity
methods, for the scalar / path-following user (one NLP at a time):

```python
x_star, (lam, zL, zU), J = jp.solve_with_jacobian(theta, x0)   # J: (n, p)
state = jp.anchor(theta, x0)             # un-batched point → B=1 state
J  = jp.sensitivity(state)               # (n, p) from the held factor
dx = jp.jvp_from_state(state, dtheta)    # J @ dtheta  -> (n,)
dp = jp.vjp_from_state(state, x_bar)     # J^T @ x_bar -> (p,)
```

- `solve_with_jacobian` / `sensitivity` / `jvp_from_state` /
  `vjp_from_state` accept and return un-batched shapes, delegating to
  the batched methods with `B=1` and squeezing — no new numerics.
- `anchor` now accepts a single un-batched point (`p_shape`) in addition
  to a batch (`(B,) + p_shape`); a single point yields a `B=1`
  `AnchorState`. The single-problem from-state wrappers reject a `B>1`
  state rather than silently mis-shaping.
- Implemented as `JaxProblem` methods (mirroring the `batched_*` names)
  rather than free functions, for consistency with the existing surface.

### Added — Exact post-solve sensitivity at a supplied point (pounce#87)

`JaxProblem.sensitivity_at(x_star, theta, duals, *, wrt_cols=None)`
returns the exact primal sensitivity `∂x*/∂θ` evaluated at a
caller-supplied primal-dual point, by re-assembling and factoring the
KKT system *there* — no IPM re-solve.

```python
J = jp.sensitivity_at(x_star, theta, (lam, zL, zU))   # (n, p_dim)
```

- **Re-factor, not reuse.** A held FERAL factor encodes the anchor
  point's `H` / `J`, so back-solving it at a moved `x_star` gives a
  first-order-stale sensitivity. `sensitivity_at` assembles the dense
  `(n+m)×(n+m)` KKT at the supplied point, which is exact there
  (assuming a KKT point for `theta`). The cheap-but-local reuse path
  stays as the predictor `batched_jvp_from_state`; this is its
  exact-refresh complement.
- Active set is read from the supplied bound multipliers `(zL, zU)`,
  exactly like the `custom_vjp` backward — the caller passes the duals
  the anchoring solve / `solve_with_warm` returned at this point.
- Pure-JAX, so itself differentiable (second-order sensitivities work);
  matches `jax.jacobian` over a fresh solve to ~1e-6 at every point
  along a swept path, including a binding bound.

This is the exact-refresh primitive for the inverse map, where `x*`
traces a known output boundary and the sensitivity must be evaluated at
the known point without paying a full re-solve per RK stage.

### Added — Barrier-μ warm start for predictor–corrector correctors (pounce#86)

The interior-point barrier parameter μ is now reported on every solve and
can be threaded into a warm-started re-solve, so a predictor–corrector
corrector resumes near the central path instead of re-walking the barrier
homotopy from the default initial μ.

- **`info["mu"]`** — every `Problem.solve` / `Solver.solve` /
  `solve_with_sens` info dict now carries the converged barrier parameter
  (`0.0` on the barrier-free SQP path).
- **`pounce.jax.solve_with_warm`** accepts a 4-element warm-state
  `(lam, zL, zU, mu)` that seeds `mu_init` / `warm_start_target_mu`, and
  returns the converged μ in a matching 4-tuple. The 3-tuple form is
  unchanged; passing `mu=None` inside a 4-tuple reports μ out without
  seeding it in. Differentiability w.r.t. `p` is preserved (the μ
  input/output are stop-gradient, like the duals).

On a small parametric NLP, seeding μ from the previous solve's converged
barrier cut a warm-started corrector from 5 interior-point iterations to
1 (same optimum). The `mu_init` / `warm_start_target_mu` algorithm
options already existed; this exposes the converged μ needed to drive
them along a path.

### Added — Post-solve Jacobian / sensitivity API from the held KKT factor (pounce#82)

`JaxProblem` now exposes a first-class post-solve sensitivity surface
that reuses the held FERAL stacked KKT factor instead of round-tripping
through `jax.vjp` / `jax.jacrev`:

```python
x_star, (lam, zL, zU), J, state = jp.batched_solve_with_jacobian(
    p_batch, x0,
    wrt_cols=slice(0, ny),   # optional parameter-column selection (1-D p)
    return_state=True,
)
dp_bar = jp.batched_vjp_from_state(state, x_bar)   # J^T @ x_bar
state.close()
```

- **`batched_solve_with_jacobian(...)`** returns the full per-block
  primal Jacobian `J` of shape `(B, n, p_dim)` (or `(B, n, len(wrt_cols))`)
  alongside `x_star` and the `(lam, zL, zU)` duals (same contract as
  `batched_solve_with_warm`). The Jacobian is assembled by evaluating the
  existing factor-reuse backward over the `n×n` identity output basis —
  one multi-RHS `Solver.kkt_solve_many` against the held LDLᵀ factor, no
  NLP re-solve.
- **`anchor(p_batch, x0, *, wrt_cols=None)`** solves once and pins the
  factor, returning an **`AnchorState`** handle for reuse across several
  post-solve sensitivity calls (linear-update pattern).
- **`batched_vjp_from_state(state, x_bar)`** is the public reverse-mode
  product `Jᵀ x̄` against a held factor.
- **`batched_jvp_from_state(state, dp)`** is the forward-mode product
  `J @ dp` — the cheap path for linear updates that never materialise the
  full `J`. It assembles the parameter-side RHS `[∂²L/∂x∂p · dp;
  ∂g/∂p · dp]` into the compound x- and constraint-blocks and back-solves
  once against the held factor. Accepts a reduced `dp` when the state was
  anchored with `wrt_cols`.
- **`AnchorState`** lifetime: works as a context manager
  (`with jp.anchor(...) as state:`) *and* supports explicit ownership
  (`state.close()`, `state.reanchor(...)`) for handles that outlive a
  lexical block. Pinned factors are exempt from the LRU but capped
  (`_pinned_capacity`, default 16) with a loud overflow error, and a
  `weakref` finalizer reclaims the factor if a handle is dropped without
  `close()`.

### Added — Structured logging + colored iteration table (pounce#71)

POUNCE now emits diagnostics through the
[`tracing`](https://docs.rs/tracing) ecosystem and renders the
per-iteration table in a tiger/rust branded color theme.

- **Colored iteration table.** Restoration lines take a background that
  varies by restoration kind (soft-stay → tan, soft-exit → amber, hard →
  deep rust); the row text shades smoothly from black toward red as the
  primal step length `alpha` shrinks (a visual stalling cue, shifted to
  cream → bright-yellow on the dark restoration backgrounds). Color is
  emitted only when stdout is a terminal — redirected output and
  `NO_COLOR` get plain text with identical column alignment.
- **Structured logs.** Solver-internal diagnostics, warnings, and
  developer instrumentation are now `tracing` events under namespaced
  targets (`pounce::algorithm`, `pounce::linsol`, `pounce::mu`,
  `pounce::sqp`, `pounce::linesearch`, `pounce::restoration`,
  `pounce::presolve`, `pounce::py`). Logs go to **stderr**; program
  output (iteration table, summary, `--dump`) stays on **stdout**.
- **Spans.** `solve`, `iteration`, `linear_solve`, and `restoration`
  spans tag nested events with context.
- **New environment variables:** `RUST_LOG` (verbosity / per-target
  filtering, default `info`), `POUNCE_LOG_FORMAT=text|json` (JSON sink on
  stderr, including the per-iteration `pounce::iteration` stream for
  Studio / CI), `NO_COLOR` / `CLICOLOR_FORCE` (color policy). Documented
  in `docs/src/options.md` and `docs/src/troubleshooting.md`.
- New `pounce-observability` crate (subscriber install + iteration
  collector) and a `pounce-common::style` palette module.
- A `log` → `tracing` bridge (`tracing_log::LogTracer`) so any remaining
  `log::*` call sites — chiefly transitive dependencies — surface through
  the subscriber and obey `RUST_LOG`.
- **Branded CLI header.** The `pounce` banner now renders a molten
  tiger/rust POUNCE logo (terminal-only; `NO_COLOR` / non-TTY get plain
  text).

### Changed

- Per-iteration JSON solve-report data is now sourced from the
  `pounce::iteration` tracing event (via an in-process collector layer)
  rather than an in-loop accumulation; the report contents are
  unchanged. Capturing iteration history requires the tracing subscriber
  installed by the CLI / Python / C frontends (or
  `pounce_observability::init_for_tests()` in tests).
- Bumped the `feral` linear-algebra dependency from 0.8.0 to 0.9.0.

### Removed

- Dropped the direct `log` crate dependency in favor of `tracing`.

### Added — Active-set SQP with working-set warm start (Phase 5b + 5c + 5d)

A new sequential-quadratic-programming driver sits alongside the
existing interior-point method, opt-in via a single option flip.
Designed for **warm-started NLP sequences** (MPC, parametric
continuation, homotopy sweeps), where the previous solve's active
set is a strong starting point.

**Tutorial:** `docs/src/active-set-sqp.md`.
**Python notebook:** `python/notebooks/06_sqp_parametric_continuation.ipynb`.
**C example:** `crates/pounce-cinterface/examples/sqp_warm_start.c`.
**GAMS example:** `gams/examples/parametric_sqp_warm_start.gms`.
**Design note:** `docs/src/active-set-sqp-warm-start.md`.

#### Algorithm selection (cross-cutting)

- New top-level option `algorithm`, values `interior-point`
  (default; existing IPM path) and `active-set-sqp` (new SQP driver).
  Settable through every interface — `add_option` in Rust /
  Python, `AddIpoptStrOption` in C, `pounce.opt` in GAMS — exactly
  like `linear_solver` already is.

#### SQP suboptions (`sqp_*` namespace)

`sqp_globalization` (`filter` | `l1-elastic`),
`sqp_hessian` (`exact` | `damped-bfgs` | `lbfgs`),
`sqp_max_iter`, `sqp_tol`, `sqp_constr_viol_tol`,
`sqp_dual_inf_tol`, `sqp_l1_penalty`, `sqp_l1_penalty_safety`,
`sqp_l1_penalty_max`, `sqp_bt_reduction`, `sqp_bt_min_alpha`,
`sqp_print_level`, `sqp_lbfgs_max_history`. Defaults mirror
`SqpOptions::default()`. Each is "only consulted when `algorithm`
is `active-set-sqp`"; the IPM path ignores them silently.

#### Python — `pounce.Problem`

New keyword argument and methods:

```python
prob.add_option("algorithm", "active-set-sqp")
x, info = prob.solve(x0, working_set=ws)
ws = info["working_set"]      # always present; None on the IPM path
ws = prob.get_working_set()
prob.set_working_set(ws)
prob.clear_working_set()
```

The `working_set` value is a 2-tuple `(bounds, constraints)` of
numpy int8 arrays with status codes 0..=3 (Inactive / AtLower /
AtUpper / Fixed-or-Equality). Module-level helper
`pounce.classify_working_set(x, x_l, x_u, g, g_l, g_u, lambda_g,
z_l, z_u, m_eq, ...)` classifies an IPM-converged iterate
into a WS suitable for `Problem.solve(working_set=…)`.

#### C ABI — four new entry points

```c
Bool IpoptGetWorkingSet(IpoptProblem, IpoptBoundStatus*, IpoptConsStatus*);
Bool IpoptSetWarmStartWorkingSet(IpoptProblem, const IpoptBoundStatus*, const IpoptConsStatus*);
Bool IpoptClearWarmStartWorkingSet(IpoptProblem);
enum ApplicationReturnStatus IpoptSolveWarmStart(
    IpoptProblem, ipnumber *x, *g, *obj_val, *mult_g, *mult_x_L, *mult_x_U,
    const IpoptBoundStatus *bound_in,
    const IpoptConsStatus  *cons_in,
    IpoptBoundStatus       *bound_out,
    IpoptConsStatus        *cons_out,
    UserDataPtr user_data);
```

Plus typedefs `IpoptBoundStatus`, `IpoptConsStatus` and the four
status constants `POUNCE_WS_INACTIVE` (= 0), `POUNCE_WS_AT_LOWER`
(= 1), `POUNCE_WS_AT_UPPER` (= 2), `POUNCE_WS_FIXED_OR_EQ` (= 3).
**No existing C entry-point signature changed** — cyipopt / JuMP /
AMPL clients link unchanged.

#### GAMS solver link

Two mechanisms ship in tandem:

- **§7.4(a) marginal-based reconstruction** (default, no
  configuration). The solver link reads variable and equation
  marginals (`x.m`, `con.m`) at the top of every `pouCallSolver`
  invocation and reconstructs the SQP working set automatically.
  Lossy at degenerate active sets — same idiom as CONOPT, IPOPT,
  KNITRO under GAMS.
- **§7.4(b) persistent state file** (opt-in via
  `sqp_state_file <path>` in `pounce.opt`). A small binary blob
  with FNV-1a checksum keyed by `(n, m, x_l, x_u, g_l, g_u)` so
  structural changes invalidate cleanly. Falls back to §7.4(a) on
  any read failure.

#### Sensitivity (`pounce-sensitivity`)

`SensResult` now carries the converged user-space multipliers
(`mult_g`, `mult_x_L`, `mult_x_U`) and constraint values (`g`),
so the parametric "predictor + SQP corrector" pattern is a single
`SensSolve::run` followed by one `classify_working_set` call.

#### Hessian sources

The `sqp_hessian` option selects between three implementations:

- `exact` — uses `eval_h`; pounce-qp's inertia control handles
  indefiniteness via diagonal-shift retry (§4.5).
- `damped-bfgs` — Powell-damped rank-2 BFGS, dense `n×n`,
  guaranteed PSD (Powell 1978).
- `lbfgs` — limited-memory BFGS with circular history, default
  6 pairs (matches IPOPT's `limited_memory_max_history`),
  materialized to dense Triplet at QP-solve time.

#### Globalizations

`sqp_globalization` selects the SQP outer-loop step-acceptance
test:

- `filter` (default) — Fletcher-Leyffer 2002 Pareto-frontier
  filter on `(constraint violation, objective)`. No penalty
  parameter; recommended general default.
- `l1-elastic` — Han-Powell merit `φ(x; ν) = f(x) + ν · violation(x)`
  with adaptive ν clamped by `sqp_l1_penalty_safety` /
  `sqp_l1_penalty_max`. SNOPT-style behaviour.

### Added — `feral_ordering` option (FERAL fill-reducing ordering)

User-facing knob for the FERAL backend's fill-reducing ordering. New
string option `feral_ordering` accepts `auto` (default; feral's
adaptive dispatcher — picks AMD / AMF / MetisND from cheap pattern
features), `auto_race` (runs symbolic factorization on AMD, MetisND,
ScotchND, KahipND and keeps the smallest factor_nnz; ~4× a single
symbolic pass, amortized across numeric refactorizations), and the
concrete methods `amd`, `amf`, `metis`, `scotch`, `kahip`. Settable
through every interface that consumes `pounce.opt` /
`OptionsList` — Rust, Python, C, GAMS, CLI — and also via the
`POUNCE_FERAL_ORDERING` environment variable for option-free
callers. Reuses the same explicit-set semantics as the other
`feral_*` options: leaving it unset keeps the `FeralConfig::from_env`
default (`Auto`).

The motivating case is `pinene_3200_0009`, where the cheap `Auto`
heuristic picks MetisND (88 s numeric) but AMD factors in 19.5 s on
the same matrix; `feral_ordering auto_race` measures both and lands
on the winner without per-problem manual tuning. See
`docs/src/options.md` "FERAL backend tuning" and
`docs/src/troubleshooting.md` for guidance.

### Added — AMPL imported (external) function support (issue #49)

`.nl` files that declare imported functions in their `F` segments
and call them via `f<id> <nargs>` tokens are now solved end-to-end.
Set `AMPLFUNC` to a newline-separated list of shared-library paths;
pounce loads each library via the standard AMPL `funcadd_ASL` ABI,
binds every referenced funcall id to a `(library, name)` pair, and
emits `TapeOp::Funcall` nodes that participate in full forward /
reverse / Hessian sweeps (first- and second-derivative requests
are issued back through the library on demand, with the packed
upper-triangular Hessian indexed as `hes[lo + hi*(hi+1)/2]`).

Tested against the IDAES `general_helmholtz_external.dylib`
fixture from the issue report — pounce reaches
`EXIT: Optimal Solution Found` on the 3-variable Helmholtz
problem. Without `AMPLFUNC` set, problems that need external
functions fail with a clear error naming the offending function
and pointing at `AMPLFUNC`.

Limitations: only the `Tape` (default) AD path supports external
functions. The `HybridTape` partial-separability path and the
JIT-style `HessianProgram` path panic on `TapeOp::Funcall` — both
are alternative routes not on `NlTnlp::new`'s critical path, so
the current production flow is unaffected.

### Added — Phase 5a `pounce-qp` crate

Standalone sparse parametric active-set QP solver. Drives the
SQP subproblem solves; also exposed as a standalone crate
(`pounce_qp::ParametricActiveSetSolver`). Implements
Gill-Murray-Saunders elastic mode (§4.3), full GMSW EXPAND
anti-cycling (§4.4), Bunch-Kaufman inertia control via
diagonal-shift retry (§4.5), iterative refinement (§4.7), and
Sherman-Morrison-Woodbury Schur-complement factor updates (§4.2,
opt-in via `QpOptions::use_schur_updates`).

### Added — In-repo regression fixtures

- `crates/pounce-algorithm/tests/hock_schittkowski_subset.rs` —
  10 HS problems with published closed-form optima.
- `crates/pounce-qp/tests/mm_published_optima.rs` —
  Maros-Mészáros-flavoured framework with 5 fixtures + reusable
  `compare_qps_to_published(text, x*, f*, …)` helper.
- `crates/pounce-algorithm/tests/parametric_sqp_corrector.rs` —
  IPM → classify_working_set → SQP corrector end-to-end.
- `crates/pounce-algorithm/tests/sqp_filter_vs_l1_elastic.rs` —
  parity between the two globalizations.

### Added — Auxiliary-equality preprocessing (Phase 0 presolve, issue #53)

A 14-PR series that scaffolds an opt-in *Phase 0* presolve pass:
detects block-triangular structure in the equality system, solves
the dependent blocks ahead of the IPM, and substitutes the
recovered variables back into the user TNLP. Targets gas-network,
power-flow, and process-design problems where a few hundred
algebraic state variables eliminate cleanly.

The algorithm and reference implementation are a port of
[ripopt PR #32](https://github.com/jkitchin/ripopt/pull/32) by
**David Bernal Neira** ([@bernalde](https://github.com/bernalde)).
The ripopt work also vendored the
`tutorial_flow_density{,_perturbed}.nl` and `gaslib11_steady.nl`
fixtures we now use for end-to-end testing.

- Hopcroft-Karp incidence matching, Dulmage-Mendelsohn decomposition,
  Tarjan SCC → block-triangular form.
- Coupling classification (linear / nonlinear / inequality-coupled)
  plus a damped-Newton block solver with large-block fallback.
- Trivial-elimination pre-pass; inequality-coupled blocks handled
  by projection.
- Reduction-frame bookkeeping with full multiplier recovery so
  `final_zL` / `final_zU` round-trip back to the user space.
- Orchestrator wired into `PresolveTnlp`, gated by
  `presolve_auxiliary` (default off). Diagnostics surfaced via
  `presolve_auxiliary_diagnostics`.
- Design note: `dev-notes/auxiliary-equality-preprocessing.md`;
  user docs in `docs/src/auxiliary-presolve.md`.

### Added — FBBT (Feasibility-Based Bound Tightening, #62)

Three-commit landing of FBBT inside `pounce-presolve`:

- `pounce-presolve::interval` — outward-rounded interval arithmetic
  on `f64`, with `Interval::div` reciprocal endpoints rounded
  outward (fixes a subtle near-zero straddle case discovered in
  review).
- `ExpressionProvider` trait + forward pass walks each constraint
  expression and tightens variable bounds from the constraint's
  `g_l`/`g_u` envelope.
- Reverse propagation + orchestrator wired through `PresolveTnlp`
  end-to-end. New options: `presolve_fbbt` (master switch,
  default off), `fbbt_tol`, `fbbt_max_iter`, `fbbt_max_constraints`.
- Docs: `docs/src/fbbt.md`; demo notebook
  `python/notebooks/08_fbbt.ipynb`.

### Added — Problem and KKT-system scaling (#61, f00c1f9)

End-to-end wiring of the upstream `nlp_scaling_*` and
`linear_system_scaling` option families:

- `nlp_scaling_method`: `none` / `user-scaling` (new — pulled from
  `set_problem_scaling` Python API or `SetIpoptProblemScaling`
  C API) / `gradient-based` (existing, now with target-gradient
  knobs `nlp_scaling_obj_target_gradient` and
  `nlp_scaling_constr_target_gradient`).
- `linear_system_scaling`: `none` / `mc19` / `ruiz` (iterative
  symmetric infinity-norm equilibration, new) / `slack-based`.
  Applied to the augmented system independent of NLP-level
  scaling.
- Python `Problem.set_problem_scaling(obj_scaling, x_scaling=None,
  g_scaling=None)` plus a worked example in
  `python/notebooks/07_scaling.ipynb`.
- Documentation: `docs/src/scaling.md`.

### Added — Mehrotra adaptive-μ defaults and init cascade (upstream parity)

- `mehrotra_algorithm` option routed through `PdSearchDirCalc`
  (previously parsed but inert).
- `adaptive_mu_globalization` cascade finished per upstream Ipopt;
  `bound_push` / `bound_frac` / `bound_mult_init_val` / `alpha_for_y`
  cascade from `mehrotra_algorithm yes`.
- `least_square_init_primal` implemented in
  `DefaultIterateInitializer`.
- `accept_every_trial_step` honored in the line search and
  cascaded from `mehrotra_algorithm` (matches upstream
  initialization behavior).

### Added — FERAL backend tunables and 0.8.0 bump

- `feral_pivtol` exposed as an `OptionsList` option with
  `FERAL_PIVTOL` environment-variable fallback.
- Tri-state `cascade_break` (#55): `auto` / `on` / `off`, inheriting
  the FERAL Phase B default unless explicitly set.
- Workspace bump to `feral 0.8.0`, which ships the SSIDS-aligned
  strict-zero-pivot inertia policy (feral gh#54 / pounce gh#52,
  *nuffield2_trap*). The temporary `[patch.crates-io]` block
  pointing at the local feral checkout has been removed.

### Added — `pounce-solve-report` crate + `IpoptWriteSolveReport` C API

- New publishable crate `pounce-solve-report` (first crates.io
  release) emits the machine-readable `pounce.solve-report/v1`
  JSON shared by the CLI, the C ABI, and the GAMS driver.
- C ABI: `IpoptWriteSolveReport(IpoptProblem, const char *path)`
  writes the report to disk after `IpoptSolve`.
- GAMS driver now emits `pounce.solve-report/v1` alongside the
  `.lst` so studio tooling can consume it directly.

### Added — Diagnostics dumps

- `--dump iterates:{summary,full}` (#68) — per-iteration trajectory
  artefacts the studio can replay. `summary` writes one JSON line
  per outer iteration; `full` adds the primal/dual vectors and
  KKT residuals.
- `--dump kkt:*+L` (#69) — augments the existing KKT-system dump
  with the LDLᵀ factor pattern (block structure, fill-in, pivot
  signs) for inertia post-mortems.
- `print_options_documentation yes` now actually walks the
  registered options and emits a categorized dump (previously a
  registered-but-inert toggle).

### Added — Studio Claude-skill and MCP GAMS tools

- `studio/skill/` — Claude-skill front-end as an alternative to the
  MCP server. Lighter-weight install path for users who just want
  the studio prompts and don't need an MCP runtime.
- `studio/mcp` — new GAMS problem tools (`run_gams_problem`,
  `analyze_gams_problem`, `parse_gams_listing`,
  `list_gams_examples`) plus an install script.

### Added — Parallel batched `pounce.jax.vmap_solve_parallel` + GIL release (pounce#74)

`pounce_py::Problem::solve` now releases the Python GIL across the
`optimize_tnlp` call (every TNLP callback was already
`Python::with_gil`-wrapped, so this is a localized
`py.allow_threads` block in `crates/pounce-py/src/problem.rs`).
That unlocks true concurrent IPM iteration across independent
`Problem` instances on different OS threads — Python-level
`f` / `g` callbacks still serialize on the GIL but the linear-algebra
heart of the solver runs in parallel.

`pounce.jax.vmap_solve_parallel` rides that change: a drop-in
replacement for `vmap_solve` that dispatches the batch over a
`ThreadPoolExecutor` of independent `Problem` instances. Forward
is parallel via the threadpool; backward is `jax.vmap` over the
per-element KKT solve (pure JAX, vectorizes naturally).

```python
from pounce.jax import vmap_solve_parallel

x_batch = vmap_solve_parallel(
    p_batch, f=f, g=g, x0=x0, n=n, m=m,
    lb=lb, ub=ub, cl=cl, cu=cu,
    workers=8,  # default: min(B, 8)
)
```

Microbench (`n=30`, `B=16`, nonlinear unconstrained, M1 8-core):
`vmap_solve` 1.00s → `vmap_solve_parallel(workers=8)` 0.37s
(~2.75×). Speedup grows with per-element solve cost. Numerically
identical to the sequential reference.

### Added — `pounce.jax.solve_with_warm` (pounce#74)

Companion to `pounce.jax.solve` that threads the previous solve's
dual triple `(mult_g, mult_x_L, mult_x_U)` into the next call via
IPOPT's `warm_start_init_point=yes` machinery.

```python
from pounce.jax import solve_with_warm

x_star, warm = solve_with_warm(
    p, f=f, g=g, x0=x0, n=n, m=m,
    lb=lb, ub=ub, cl=cl, cu=cu,
    warm_start=None,                # cold first call
)
for p_k in trajectory[1:]:
    x_star, warm = solve_with_warm(
        p_k, f=f, g=g, x0=x_star, n=n, m=m,
        lb=lb, ub=ub, cl=cl, cu=cu,
        warm_start=warm,            # threaded duals
    )
```

Differentiable w.r.t. `p` via the same implicit-function rule as
`solve`. Cotangents on the warm-state outputs and the warm-state
inputs are dropped (zero) — at the optimum the duals are a
function of `p` and the active set, not an independent input to
`dx*/dp`. `solve` itself is unchanged (non-breaking).

### Added — `pounce.jax.JaxProblem` build-once/solve-many handle (pounce#75)

Iterative outer loops (differentiable constrained layers in a
training step, parametric sweeps) were paying a ~45ms rebuild on
every call to `pounce.jax.solve` / `vmap_solve_parallel` /
`solve_with_warm` — re-JIT of `jax.grad`/`jacrev`/`hessian`, the
one-shot random sparsity probe, plus a fresh `pounce.Problem`
construction — versus a ~3ms underlying solve. On `n=5, m=6`
problems that's a ~14× wrapper overhead.

`JaxProblem` is a build-once handle: do the JIT and sparsity probe
in `__init__`, then expose `.solve(p, x0)`, `.solve_with_warm(p, x0,
warm)`, `.vmap_solve(p_batch, x0)`, and `.vmap_solve_parallel(p_batch,
x0, workers=)` as methods that reuse the prebuilt state across
calls. Each worker thread in `vmap_solve_parallel` keeps its own
cached `pounce.Problem` via `threading.local` so the build cost is
paid at most once per worker (typically `min(B, 8)` total) rather
than `B` times per batch.

```python
from pounce.jax import JaxProblem

jp = JaxProblem(
    f=f, g=g, n=n, m=m, p_example=p0,
    lb=lb, ub=ub, cl=cl, cu=cu,
    options={"tol": 1e-9, "print_level": 0},
)
for p_k in trajectory:
    x_star = jp.solve(p_k, x0=x_prev)
    x_prev = x_star
```

Microbench on the issue's `n=5, m=6` shape — 20 sequential solves at
different `p`:

```
top-level solve   (20 calls): 1.914s  → 95.7ms/solve
JaxProblem.solve  (20 calls): 0.136s  → 6.8ms/solve
speedup: 14.1x
```

Existing top-level `solve` / `vmap_solve` / `vmap_solve_parallel` /
`solve_with_warm` are unchanged (non-breaking) — `JaxProblem` is a
new surface for performance-sensitive iterative use.

### Added — `JaxProblem` factor-reuse backward (k_aug-style; pounce#76)

The `custom_vjp` backward of `JaxProblem.solve` /
`solve_with_warm` no longer assembles a dense
`(n+m) × (n+m)` KKT block in JAX and runs `jnp.linalg.solve` on it.
Instead it reuses the IPM's converged compound KKT factor through
`pounce.Solver.kkt_solve` — the same factor [k_aug] uses for
parametric sensitivity. Two wins:

* **Perf.** The dense back-solve is O((n+m)³) on every bwd call;
  reusing the held LDLᵀ factor makes it O(nnz(L)). For modest `n`
  the absolute savings are small; for `n+m` in the hundreds-to-
  thousands it dominates the bwd.
* **Correctness.** The compound block's bound-multiplier rows
  `(z_l, z_u)` already encode active-set behaviour — at convergence
  active bounds have unbounded `z` (forces `Δx_i = 0` in the
  back-solve), inactive bounds have `z ≈ 0` (leaves `Δx_i` free).
  Slack inequality rows in the user's `g` are handled the same way
  by `(v_l, v_u)`. The factor-reuse path therefore drops the
  explicit active-set masking the dense path does on `H` / `J` / `v`;
  accuracy is `O(μ)` at the IPM barrier parameter, well below `tol`
  after convergence.

Behaviour change: `JaxProblem(factor_reuse=True)` is the default. Set
`factor_reuse=False` for a verbatim fallback to the pre-#76 dense
backward (useful for higher-order differentiation, since the dense
backward stays JAX-traced and is itself differentiable).

Plumbing:

* `pounce.Solver` exposes a new `block_dims` getter returning the
  `(n_x, n_s, n_y_c, n_y_d, n_z_l, n_z_u, n_v_l, n_v_u)` layout of
  the compound KKT vector so the JAX bwd can pack a partial RHS
  (just the x-block) and unpack `u_x` / `u_y_c` / `u_y_d`.
* Each fwd registers its converged `Solver` in a bounded-LRU cache
  on the `JaxProblem` (default capacity 128, exposed as
  `clear_solver_cache()` for early eviction). LRU rather than
  pop-on-read because `jax.jacobian` calls the bwd N times per
  fwd; pop semantics would crash from the second direction onward.
* The back-solve `pure_callback` uses
  `vmap_method="sequential"` so `jax.jacobian` / `jax.vmap` of a
  loss-gradient correctly iterate one cotangent at a time across
  the impure host call.

The standalone `pounce.jax.solve` / `vmap_solve_parallel` /
`solve_with_warm` keep the dense backward for now.

[k_aug]: https://github.com/dthierry/k_aug

### Added — `JaxProblem.batched_solve` stacked block-diagonal solve (pounce#76 (A))

`JaxProblem.batched_solve(p_batch, x0)` runs one IPM solve over a
single NLP whose variables are `[x^(1); ...; x^(B)]`, constraints are
`concat(g(x^(k), p^(k)))`, and objective is `Σ_k f(x^(k), p^(k))`.
The Jacobian and Lagrangian Hessian are block-diagonal (no
cross-block coupling, since each block-`k` constraint touches only
the block-`k` slice of `X` and the objective is a pure sum), so the
IPM sees one big sparse problem but spends linear-system work
proportional to `B × (per-block factor cost)`.

Complementary to the existing batched surfaces:

* `vmap_solve` — sequential `jax.lax.map`, one solve per iterate.
* `vmap_solve_parallel` — B independent IPMs in a
  `ThreadPoolExecutor` (GIL released per solve). Wins when batch
  elements have very different convergence behaviour.
* `batched_solve` — one stacked IPM. Wins when blocks have similar
  convergence behaviour (shared barrier homotopy and shared
  symbolic factorisation amortise across the batch) and when B is
  large enough that the per-call Python overhead of B fwd
  dispatches becomes visible — one Rust crossing instead of B.

`custom_vjp`-wrapped: `jax.grad` / `jax.jacobian` through
`batched_solve` work end-to-end. The bwd vmaps the per-element
dense KKT back-solve, which is exact because the block-diagonal
coupling means `∂x^(k)*/∂p^(j) = 0` for `k ≠ j`.

Plumbing:

* `_StackedJaxNlp` lifts the per-block sparsity pattern (cached on
  the parent `JaxProblem` from the one-shot probe) to the stacked
  problem's block-diagonal pattern at construction time, so the
  per-solve `jacobianstructure` / `hessianstructure` callbacks are
  O(1).
* Stacked Problems are built per (thread, B) with a tiny LRU on
  the `JaxProblem` (cap 4) keyed by batch size — guards against
  cycling between a couple of sizes (e.g. eval batch ≠ train
  batch).
* Per-block bounds `lb`/`ub`/`cl`/`cu` are tiled across the batch;
  per-block bounds aren't exposed on this surface.

### Changed

- `pounce-qp::ParametricActiveSetSolver::solve_equality_plus_bounds`
  now falls through to `solve_elastic` when the equality-relaxed
  cold start violates a variable bound. Previously returned
  `UnsupportedFeature`.
- `optimize_sqp_tnlp` now populates `SolveStatistics`
  (`iteration_count`, `final_dual_inf`, `final_constr_viol`,
  `final_objective`) so `GetIpoptIterCount`, `info["iter_count"]`,
  etc. report SQP-side numbers on the SQP path.

### Fixed

- SQP `check_kkt` stationarity formula: was `∇f + Jᵀ λ_g + λ_x`,
  must be `∇f + Jᵀ λ_g − λ_x` (pounce-qp packs
  `λ_x = z_l − z_u = −λ_sat`). Latent — only triggered by problems
  with an active variable bound. Discovered on a 3-D simplex
  projection.
- `fix(mu): guard probing oracle against corrupted iterate (#58)`
  — the probing oracle no longer dereferences fields of an
  iterate that the line-search rejected mid-update.
- `fix(mu/probing)`: σ denominator uses `curr_avrg_compl`, not
  `data.curr_mu`, matching upstream.
- `fix(mu-oracle)`: allow inexact affine predictor solves to feed
  the quality-function oracle (upstream parity).
- `fix(l1-wrapper): use multi-pass restoration factory provider
  (#24)` — the ℓ₁ penalty wrapper now nests a restoration sub-IPM
  whose own restoration provider is the multi-pass factory,
  matching the outer IPM path.
- `fix(restoration)`: restoration sub-IPM inherits the outer
  `mu_strategy` rather than resetting to `monotone`.
- `fix(feral)`: zero-pivot factorizations on LP-shape KKT
  systems route to `Singular` instead of bubbling up as
  `Internal`.
- `fix(fbbt)`: outward-round reciprocal endpoints in
  `Interval::div` for the near-zero straddle case.
- `fix(presolve)`: auxiliary preprocessing + `presolve_bound_tightening`
  infeasibility paths (#60).
- `fix(init/ls)`: perturb `delta_c`/`delta_d` by 1e-8 in the
  least-squares-init augmented system to avoid exact rank
  deficiency.
- `fix(scaling)`: scale `d_l` / `d_u` in step with `d(x)` under
  gradient-based scaling.
- `fix(hsl)`: HSL build script is a no-op when `COINHSL_DIR` is
  unset, so `cargo build` works on machines without HSL
  installed even with the `ma57` feature off.
- `fix(benchmark-report)`: composite report now globs the newest
  `pounce_*.json` under `benchmarks/mittelmann/results/` instead
  of hard-coding `pounce_v0.1.0.json`.
- `fix(jax)`: `pounce.jax.solve` backward pass now respects the
  constraint active set, not just variable bounds. Slack inequality
  rows are dropped from the implicit-function-theorem KKT block via
  the same identity-augment trick used for active bounds; previously
  they were kept as equalities, silently returning the wrong
  `dx*/dp` whenever an inequality was inactive at the optimum
  (pounce#73).

### Docs

- `docs: adaptive-μ option tables, scaling worked example,
  troubleshooting guide` — `docs/src/options.md`,
  `docs/src/scaling.md`, `docs/src/troubleshooting.md` refreshed.
- FBBT reference page (`docs/src/fbbt.md`) and Pyomo demo
  notebook `python/notebooks/08_fbbt.ipynb` (#62).
- Scaling docs page (`docs/src/scaling.md`) + Python demo notebook
  `python/notebooks/07_scaling.ipynb` (#61).
- `studio/skill` README: corrected `POUNCE_BIN` claim,
  `inspect --json`, sibling-feral layout.
- README badges: PyPI version + downloads for `pounce-solver` and
  `pyomo-pounce`; Zenodo DOI
  `10.5281/zenodo.20387011` published.

### Compatibility

- All existing IPM users (`IpoptSolve`, `Problem.solve(x0=…)`,
  `option nlp = pounce` without `algorithm` set) continue to
  behave identically. Every Phase 5 addition is opt-in.
- The C ABI is strictly additive — four new symbols, no signature
  changes.
- The Python `Problem.solve` signature gained one optional kwarg
  (`working_set=None`); positional callers are unaffected.


### Algorithm-path isolation guarantees

The IPM and active-set SQP paths share the TNLP layer, options
registry, linear-solver backend, and `finalize_solution`, but are
otherwise isolated. Toggling `algorithm` is always safe:

- The default (`algorithm = interior-point`) runs zero Phase 5
  code. Users who never set `active-set-sqp` are unaffected.
- `sqp_*` options are silently ignored on the IPM path.
- IPM warm-start options (`warm_start_init_point`, `bound_push`,
  `bound_frac`, `slack_bound_push`, `mult_init_max`, `mu_init`,
  `mu_target`, …) are silently ignored on the SQP path.
- Warm-start payloads are path-local:
  `set_sqp_warm_start(SqpIterates)` /
  `Problem.solve(working_set=…)` / `IpoptSetWarmStartWorkingSet`
  feed the SQP loop only; `lagrange=` / `zl=` / `zu=` paired with
  `warm_start_init_point=yes` feed the IPM only.
- `info["working_set"]` is always present in the Python info
  dict but is `None` on the IPM path.
- Callers can flip between paths across solves on the same
  problem handle — the parametric corrector pattern in the
  tutorial uses this for cold IPM warm-up followed by an SQP
  corrector.

These guarantees are exercised by the test suite: see
`application_default_does_not_select_sqp`,
`application_sqp_warm_start_auto_clears_after_use`,
`application_sqp_warm_start_round_trip`, and
`test_get_working_set_returns_none_on_ipm_path` (Python).

## [0.2.0] — 2026-05-25

First tagged release. The `0.1.0` work-in-progress version was never
tagged; everything below summarizes the state of `main` as of this
release.

### Solver core

- **Full Ipopt-parity C ABI**: `CreateIpoptProblem`, `IpoptSolve`,
  `AddIpoptStrOption` / `AddIpoptNumOption` / `AddIpoptIntOption`,
  `OpenIpoptOutputFile`, `SetIpoptProblemScaling`,
  `SetIntermediateCallback`, `GetIpoptCurrentIterate`,
  `GetIpoptCurrentViolations`, plus a new `IpoptSolver` session
  handle (`IpoptSolverSolve`, `IpoptSolverResolve`,
  `IpoptSolverKktSolve`, `IpoptSolverParametricStep`).
- **Restoration phase** wired through `IpoptSolve` with the soft
  restoration line search; nested IPM honors the parent's
  `print_iter_output` gate.
- **Rapid infeasibility detection** in the main loop; convergence
  statuses certified against upstream Ipopt.
- **Option-parity (tier-A waves 1-4)**: convergence options
  (`tol`, `acceptable_tol`, etc.), mu/watchdog/output toggles,
  iteration-output flags, warm-start machinery,
  `fixed_variable_treatment`, `nlp_*_bound_inf`,
  `barrier_tol_factor`, `sigma_min` / `sigma_max` for the adaptive
  quality-function oracle.
- **Sensitivity (sIPOPT)**: Phase D landed — convenience API,
  eigendecomposition, fixed-variable lifting, boundcheck. New
  `Solver` session API on top: value-typed `Factorization` handle
  in `pounce-linsol` enables factor-once / solve-many; `Solver`
  exposes `kkt_solve`, `parametric_step`, and
  `compute_reduced_hessian` without callback shapes.
- **Presolve** crate (`pounce-presolve`) as an opt-in TNLP wrapper.

### Backends and bindings

- **Python** (`pounce-solver`): PyO3 bindings with a cyipopt-style
  `Problem` class and a scipy-style `minimize()` facade. The wheel
  bundles the `pounce` CLI executable.
- **Python session API** (`pounce.Solver`): pyclass that wraps the
  Rust `Solver`, enabling warm-start sequences (MPC / parametric /
  B&B) and many-RHS sensitivity workflows without the
  callback-based shape.
- **pyomo-pounce** (`pyomo-pounce`): Pyomo SolverFactory plugin
  that drives the `pounce` CLI on the user's PATH.
- **GAMS link**: native solver link (`libGamsPounce`) for GAMS;
  Jacobian eval skips dense memsets and pure-linear rows.
- **CLI**: bundled `pounce` binary writes AMPL `.sol` solution
  output; new `--about` prints version / build / features / paths;
  `--dump` writes per-iteration KKT artefacts; the sIPOPT
  sensitivity step is folded in.

### Linear-solver layer

- **Public `Factorization`** in `pounce-linsol`: factor once,
  back-solve many RHS, refactor with new values reusing the
  symbolic factor / AMD ordering.
- **MA57** backend (`pounce-hsl`) honors the `linear_solver`
  option default (`"ma57"`).
- **Feral** backend: cascade-break and FMA default off (opt-in via
  env); near-singular factorizations are flagged via an absolute
  pivot floor; explicit-zero stripping before KKT factor; skips
  refactor on same-matrix back-solve.

### Numerical robustness

- TNLP `eval_*` user-callback failures surface as NaN instead of
  panicking.
- Round-off-tolerant `Compare_le` in the Armijo line-search test.
- Unconstrained problems routed through the IPM (no degenerate
  paths).
- `push_x_into_interior` uses `dim()` (not `values().len()`),
  fixing a subtle off-by-one on partially-filled vectors.
- `OrigIpoptNlp::eval_h` always uses the `h_entry_in_full`
  mapping; closes the panic when an entire Hessian row sits on a
  fixed variable.

### Benchmarks

- **Composite report** (`make benchmark` →
  `benchmarks/BENCHMARK_REPORT.md`) covering 9 suites: CUTEst (727
  curated; 1542 full sweep), Mittelmann LP/QP, water-network
  design, gas-network, electrolyte, grid, CHO, large-scale, and
  the GAMS link.
- **Incremental per-suite targets**: `make benchmark-<suite>`
  skips when `results.json` is fresh; `make benchmark-<suite>-rerun`
  forces a rebuild.
- **MA57 baseline** integrated into the composite report.

### Studio & tooling

- **studio/mcp** MCP server (`pounce-studio-mcp`) with
  `analyze`, `run`, `explain`, `citations` tools and an embedded
  glossary; backed by `pounce-studio-core` via PyO3.
- **Linear-solver post-mortem** aggregated end-to-end and
  surfaced through the studio.

### Infrastructure

- CI workflow with format / clippy / build / test, plus
  wheel-smoke for `pounce-solver` and `pyomo-pounce`.
- mdbook documentation built and deployed to GitHub Pages via the
  new `docs.yml` workflow.
- Zenodo metadata (`.zenodo.json`) and `CITATION.cff` for
  archival on every GitHub Release.

[0.3.0]: https://github.com/jkitchin/pounce/releases/tag/v0.3.0
[0.2.0]: https://github.com/jkitchin/pounce/releases/tag/v0.2.0
