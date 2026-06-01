# Changelog

All notable changes to POUNCE are tracked here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `1.0.0`. Pre-1.0 minor bumps may include breaking
changes.


## Unreleased

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

## [0.3.0] — 2026-05-29

### Added — Active-set SQP with working-set warm start (Phase 5b + 5c + 5d)

A new sequential-quadratic-programming driver sits alongside the
existing interior-point method, opt-in via a single option flip.
Designed for **warm-started NLP sequences** (MPC, parametric
continuation, homotopy sweeps), where the previous solve's active
set is a strong starting point.

**Tutorial:** `docs/tutorials/active-set-sqp.md`.
**Python notebook:** `python/notebooks/06_sqp_parametric_continuation.ipynb`.
**C example:** `crates/pounce-cinterface/examples/sqp_warm_start.c`.
**GAMS example:** `gams/examples/parametric_sqp_warm_start.gms`.
**Design note:** `dev-notes/research/active-set-sqp-warm-start.md`.

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
