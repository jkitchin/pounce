# Changelog

All notable changes to POUNCE are tracked here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `1.0.0`. Pre-1.0 minor bumps may include breaking
changes.


## Unreleased

### Added — Interactive solver debugger (`--debug` / `--debug-json`)

A "pdb for the interior-point loop." The CLI can now pause the solve at
each outer iteration, drop into a prompt to inspect and *mutate* live
state, then continue — driven either by a human or by an LLM agent /
program.

- **Two front ends, one command engine.** `pounce <problem> --debug`
  gives a line REPL (`pounce-dbg>` prompt on stderr). `--debug-json`
  speaks newline-delimited JSON on stdin/stdout — each pause emits one
  `{"event":"pause",…}` object and each command one
  `{"event":"result",…}` object — so an agent can drive the loop
  programmatically (JSON mode forces `print_level 0` to keep the channel
  clean). Commands accept either a bare string or
  `{"cmd":"…","args":[…]}`.
- **Inspect:** `info`, `print x|s|y_c|y_d|z_l|z_u|v_l|v_u`, `print dx`
  (search-direction blocks), `print mu|obj|inf_pr|inf_du|err|iter`.
- **Flow:** `step`, `continue`, `run N`, `break [N|clear|del N]`,
  `detach`, `quit`. Pauses at the first checkpoint so you start in
  control.
- **Conditional breakpoints:** `break if <metric><op><value>` pauses
  when a live quantity crosses a threshold — metrics
  `mu|inf_pr|inf_du|obj|err|iter`, operators `< <= > >= ==`
  (e.g. `break if inf_pr<1e-6`). `break` lists them, `break clear cond`
  removes them; the firing condition is reported in the pause
  banner / `"reason"` field.
- **Mutate:** `set mu <v>`, `set x[i] <v>` (single component),
  `set x <v0,v1,…>` (whole block). Iterate edits rebuild the
  `IteratesVector` with a fresh tag so the tag-keyed CQ caches invalidate
  correctly — the new point feeds straight back into the solve.
- **Option discovery + completion:** `opt [filter]` lists registered
  options (name/type/default/short-desc, long-desc on exact match),
  `complete <prefix>` returns command/option candidates, and
  `set opt <name> <value>` validates against the registry.
- **Line editing (rustyline):** on an interactive TTY the `--debug` REPL
  gets persistent history (`~/.pounce_dbg_history`), Ctrl-R search, and
  context-sensitive Tab completion — command verbs, block names, metric
  names (after `break if`), and option names (after `set opt`/`opt`).
  Piped input and `--debug-json` fall back to a plain line reader.
- **Drop in on failure (`--debug-on-error`):** run freely and pause only
  at a new terminal checkpoint, and only if the solve didn't succeed —
  a post-mortem at the failing iterate. The terminal checkpoint also
  fires in normal `--debug` (final-point inspect); JSON `pause` events
  gain `checkpoint:"terminated"` + `status`.
- **Attach via Ctrl-C (`--debug-on-interrupt`):** run normally but install
  a SIGINT handler that drops into the debugger at the next iteration; a
  second Ctrl-C aborts. Ctrl-C also breaks into any other debug mode
  mid-`continue` (at a rustyline prompt Ctrl-C stays a line-cancel).
- **Soft rewind (`goto <k>` / `restart`):** the debugger snapshots the
  primal-dual state (iterate + μ + τ) every iteration (cheap — the
  iterate is an immutable `Rc`; capped at 2000, oldest evicted), and
  `goto`/`restart` rewinds to a captured iteration so you can re-tune and
  resume. Primal-dual only — strategy history (filter, adaptive-μ,
  quasi-Newton memory) is not restored, so it's "resume from here," not a
  bit-exact replay.
- **Save artifacts (`save [path]`):** write the current iterate (all
  primal/dual blocks + the search-direction blocks) and residual scalars
  to a JSON file for external analysis; defaults to a temp path keyed by
  iteration.
- **Re-solve from a saved point (`resolve`):** capture the current primal
  `x` and the `set opt` edits staged this session, then re-run the solve
  from that point with the new options applied (a primal warm start). The
  CLI loops: apply staged options, seed the next solve via a starting-point
  wrapper, re-install a fresh debugger, run again. Because each solve
  rebuilds its strategies from options, option changes do take effect on
  the re-solve. The primal seed is dropped (with a fall-back to the
  problem's own start) if presolve / fixed-variable elimination changed
  the coordinate count.
- **Visualization:** `viz <block|dx>` writes the vector and opens it in
  an external viewer (`POUNCE_DBG_VIEWER`, else `xdg-open`/`open`).
- **Visual-debugger protocol (`--debug-json`).** stdout is now a *pure*
  JSON channel — the banner, problem-stats block, and final summary are
  routed to stderr — so a GUI / IDE front end can consume it line by
  line. The session is framed by lifecycle events: `hello` (protocol
  version + capabilities + command/metric/block vocabulary), `pause`,
  `result` (echoing the client's `request_id` for async correlation),
  and `terminated` (final status, iteration count, objective, eval
  counts). Exit model: `quit` stops now (`UserRequestedStop`),
  `continue`/`detach` run to completion, and a closed JSON stdin pipe
  aborts the solve (REPL Ctrl-D detaches and finishes); every non-kill
  path ends with a `terminated` event.

Engine lives in `pounce-algorithm`'s `debug` module (`DebugHook` /
`DebugCtx` / `Checkpoint`); the REPL/agent front end is
`pounce-cli::debug_repl`. Only the `IterStart` checkpoint is wired today;
the `Checkpoint` enum is open for finer-grained stops.

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
**Design note:** `docs/research/active-set-sqp-warm-start.md`.

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

[0.2.0]: https://github.com/jkitchin/pounce/releases/tag/v0.2.0
