# Solver Options

POUNCE accepts options the same way upstream Ipopt does. Option names
and semantics follow Ipopt's, so an existing Ipopt options file or
`KEY=VALUE` invocation works unchanged.

## Setting options

**On the command line** — append `KEY=VALUE` pairs after the input:

```sh
pounce problem.nl tol=1e-10 max_iter=500 print_level=8
```

**From an options file** — upstream `ipopt.opt` format:

```sh
pounce problem.nl --options-file ipopt.opt
```

Command-line `KEY=VALUE` pairs override values loaded from the options
file.

## Commonly used options

| Option          | Meaning                                                              |
|-----------------|----------------------------------------------------------------------|
| `tol`           | Overall convergence tolerance on the KKT error.                      |
| `max_iter`      | Maximum number of outer iterations.                                  |
| `print_level`   | Console verbosity, 0 (silent) – 12 (maximum debug).                  |
| `linear_solver` | KKT linear-solver backend. `ma57` requires the `ma57` feature build. |
| `mu_strategy`   | Barrier-parameter update strategy (`monotone` / `adaptive`).         |
| `solver_selection` | Route LP/convex-QP to the specialized convex IPM. See [LP/QP Routing](lp-qp-routing.md). |
| `qp_presolve`   | Presolve on the convex LP/QP path (`yes` / `no`, default `yes`). See [LP/QP Routing](lp-qp-routing.md#presolve). |

For the full upstream option catalogue, see the
[Ipopt options reference](https://coin-or.github.io/Ipopt/OPTIONS.html);
POUNCE reuses those names.

For scaling-specific options (`nlp_scaling_method`, target-gradient
overrides, `linear_system_scaling`), see the [Scaling](scaling.md)
reference page. For nonlinear bound tightening (`presolve_fbbt`,
`fbbt_tol`, `fbbt_max_iter`, `fbbt_max_constraints`), see the
[FBBT](fbbt.md) reference page.

## Barrier-parameter (μ) strategy

The barrier parameter μ controls the inner subproblem's relaxation of
complementarity. The two strategies are `monotone` (default — geometric
schedule) and `adaptive` (quality-function oracle picks each μ from the
current iterate's complementarity). See
[μ-strategy](troubleshooting.md#μ-strategy) for when to switch.

| Option                                  | Default            | Meaning                                                                                       |
|-----------------------------------------|--------------------|-----------------------------------------------------------------------------------------------|
| `mu_strategy`                           | `monotone`         | `monotone` (Fiacco–McCormick schedule) or `adaptive` (oracle-driven).                         |
| `mu_oracle`                             | `quality-function` | Adaptive oracle: `quality-function` / `loqo` / `probing`.                                     |
| `mu_init`                               | `0.1`              | Seed value for μ at the first iterate.                                                        |
| `mu_min`                                | `1e-11`            | Floor on μ; the solver stops decreasing past this.                                            |
| `mu_max`                                | `1e5`              | Cap on μ (adaptive mode). When set explicitly it overrides the `mu_max_fact` initialization.  |
| `mu_max_fact`                           | `1e3`              | Initializes `mu_max` as `mu_max_fact · curr_avrg_compl` at the first iterate (adaptive mode). |
| `mu_target`                             | `0.0`              | Stop target for μ in monotone mode.                                                           |
| `mu_linear_decrease_factor`             | `0.2`              | κ_μ in `μ ← min(κ_μ · μ, μ^θ_μ)`.                                                             |
| `mu_superlinear_decrease_power`         | `1.5`              | θ_μ in the same formula.                                                                      |
| `barrier_tol_factor`                    | `10.0`             | Inner-subproblem tolerance scales as `barrier_tol_factor · μ`.                                |
| `sigma_max`                             | `1e2`              | Upper clamp on σ chosen by the quality-function oracle.                                       |
| `sigma_min`                             | `1e-6`             | Lower clamp on σ (raising this to `1e-2` can break a stair-stepping stall on some problems).  |
| `adaptive_mu_globalization`             | `obj-constr-filter`| Adaptive-mode globalization: `kkt-error`, `obj-constr-filter`, or `never-monotone-mode`.      |

### Quality-function oracle (adaptive-μ details)

These are only consumed when `mu_strategy=adaptive` and
`mu_oracle=quality-function`. Defaults mirror upstream
`IpQualityFunctionMuOracle::RegisterOptions`.

| Option                                  | Default          | Meaning                                                                                       |
|-----------------------------------------|------------------|-----------------------------------------------------------------------------------------------|
| `quality_function_norm_type`            | `2-norm-squared` | Norm used to aggregate KKT components inside `q(σ)`: `1-norm`, `2-norm`, `2-norm-squared`, `max-norm`. |
| `quality_function_centrality`           | `none`           | Centrality penalty term: `none`, `log`, `reciprocal`, `cubed-reciprocal`.                     |
| `quality_function_balancing_term`       | `none`           | Balancing penalty when complementarity ≪ infeasibilities: `none` or `cubic`.                  |
| `quality_function_max_section_steps`    | `8`              | Cap on golden-section iterations when picking σ.                                              |
| `quality_function_section_sigma_tol`    | `1e-2`           | Width tolerance in σ-space terminating the golden-section search.                             |
| `quality_function_section_qf_tol`       | `0.0`            | Relative flatness tolerance on `q(σ)` terminating golden section.                             |

### Adaptive-μ globalization

Tuning the safeguards that fall back to monotone-μ mode when the
adaptive oracle stops making progress. Defaults mirror upstream
`IpAdaptiveMuUpdate::RegisterOptions`.

| Option                                  | Default | Meaning                                                                                       |
|-----------------------------------------|---------|-----------------------------------------------------------------------------------------------|
| `adaptive_mu_safeguard_factor`          | `0.0`   | LOQO safeguard floor on the oracle's μ candidate.                                             |
| `adaptive_mu_monotone_init_factor`      | `0.8`   | Multiplier on `avrg_compl` when seeding monotone mode after a bailout.                        |
| `adaptive_mu_restore_previous_iterate`  | `no`    | Restore the latest free-mode iterate when switching to fixed mode.                            |
| `adaptive_mu_kkterror_red_iters`        | `4`     | Window length for the `kkt-error` globalization history.                                      |
| `adaptive_mu_kkterror_red_fact`         | `0.9999`| Required relative KKT-error reduction over that window.                                       |
| `adaptive_mu_kkt_norm_type`             | `2-norm-squared` | Norm used to score the iterate in adaptive globalization decisions.                  |

## ℓ₁ penalty-barrier wrapper options

These tune the degenerate-NLP wrapper described in
[Running Solves](cli.md). All are default-tuned and rarely need
overriding:

| Option                               | Default | Meaning                                                    |
|--------------------------------------|---------|------------------------------------------------------------|
| `l1_exact_penalty_barrier`           | `no`    | Run the ℓ₁-exact penalty-barrier wrapper unconditionally.  |
| `l1_fallback_on_restoration_failure` | `no`    | Retry with the wrapper only when the standard solve fails. |
| `l1_penalty_init`                    | `1.0`   | Initial penalty weight ρ.                                  |
| `l1_penalty_max`                     | `1e6`   | Maximum penalty weight before declaring infeasibility.     |
| `l1_penalty_increase_factor`         | `8.0`   | Multiplier applied to ρ each outer iteration.              |
| `l1_penalty_max_outer_iter`          | `8`     | Maximum penalty outer iterations.                          |
| `l1_slack_tol`                       | `1e-6`  | Slack tolerance for "constraints satisfied".               |
| `l1_steering_factor`                 | `10.0`  | Steering-rule factor for ρ escalation.                     |

## NLP Presolve

POUNCE's TNLP-wrapper presolve pipeline runs *before* the IPM
starts. It tightens variable bounds, drops redundant rows, and
(optionally) eliminates square auxiliary-equality sub-systems
structurally. All are off by default — set the master switch first:

| Option                                  | Default | Meaning                                                                        |
|-----------------------------------------|---------|--------------------------------------------------------------------------------|
| `presolve`                              | `no`    | Master switch for the whole presolve layer. Off → wrapper is a no-op.          |
| `presolve_bound_tightening`             | `yes`   | Phase 1 — Andersen-style bound propagation from linear rows.                   |
| `presolve_redundant_constraint_removal` | `yes`   | Phase 2 — drop linear constraints already implied by current bounds.           |
| `presolve_linear_eq_reduction`          | `no`    | Phase ≥2 — eliminate fixed singleton variables exposed by linear equalities.   |
| `presolve_licq_check`                   | `yes`   | Phase 3 — detect rank-deficient equality blocks before the IPM starts.         |
| `presolve_licq_action`                  | `warn`  | What to do on degeneracy: `warn` (just report) or `auto_l1` (turn on ℓ₁).      |
| `presolve_warm_z_bounds`                | `yes`   | Phase 4 — warm-start bound multipliers when bounds get tightened by Phase 1.   |
| `presolve_bound_mult_init_val`          | `1.0`   | Value used by Phase 4 for those warm-start hints.                              |
| `presolve_max_passes`                   | `3`     | Fixed-point iteration cap across the bound-tightening passes.                  |
| `presolve_print_level`                  | `0`     | Per-pass verbosity (0 silent, 5 per-pass, 8 per-transformation).               |

### Feasibility-based bound tightening (Phase 1b)

Interval-arithmetic propagation through nonlinear constraint
expression DAGs (see [FBBT](fbbt.md)). Available today for
`.nl`-loaded problems via `NlTnlp`; other TNLP sources opt out
silently.

| Option                  | Default | Meaning                                                                                  |
|-------------------------|---------|------------------------------------------------------------------------------------------|
| `presolve_fbbt`         | `no`    | Master switch. Requires `presolve=yes` and an `ExpressionProvider`.                      |
| `fbbt_tol`              | `1e-6`  | Minimum per-variable bound improvement to keep iterating.                                |
| `fbbt_max_iter`         | `10`    | Outer-sweep cap.                                                                         |
| `fbbt_max_constraints`  | `0`     | Per-sweep cap on constraints inspected (`0` = unlimited).                                |

### Auxiliary-equality preprocessing (Phase 0)

A separate set of options controls the structural elimination pass
documented in [Auxiliary-Equality Preprocessing](auxiliary-presolve.md):

| Option                                   | Default | Meaning                                                                                  |
|------------------------------------------|---------|------------------------------------------------------------------------------------------|
| `presolve_auxiliary`                     | `no`    | Master switch for the Phase-0 structural elimination pass.                               |
| `presolve_auxiliary_coupling`            | `safe`  | Which coupling classes are eligible: `none` / `safe` / `aggressive`.                     |
| `presolve_auxiliary_tol`                 | `1e-8`  | Residual tolerance for accepting a candidate block solve.                                |
| `presolve_auxiliary_max_block_dim`       | `8`     | Largest block the lightweight Newton solver will attempt (larger blocks rejected in v1). |
| `presolve_auxiliary_wall_time_fraction`  | `0.1`   | Fraction of the solver's wall-time budget the pass is allowed to spend.                  |
| `presolve_auxiliary_diagnostics`         | `no`    | Emit the diagnostics summary via the journalist after Phase 0 runs.                      |

## FERAL backend tuning

`linear_solver=feral` (the default — see
[Commonly used options](#commonly-used-options)) is configurable
through six `feral_*` options. Defaults are tuned for the IPM
workload and rarely need changing; reach for these when profiling a
specific problem.

| Option                       | Default | Meaning                                                                                                                                                                                  |
|------------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `feral_ordering`             | `auto`  | Fill-reducing ordering method (see table below). `auto` lets feral's adaptive dispatcher pick per-matrix; `auto_race` measures the actual symbolic outcome and keeps the best.            |
| `feral_pivtol`               | `1e-8`  | Relative Bunch-Kaufman partial-pivoting threshold `u`. Analog of `ma27_pivtol` / `ma57_pivtol`. Smaller → sparser `L`, faster, less stable; larger → more 2×2 blocks, denser, more stable. LAPACK's textbook maximum-stability value is `0.5`. |
| `feral_refine`               | `yes`   | Iterative refinement on every back-solve. Closes the residual floor from cascade-break's `L`-factor perturbation; disable only when timing the bare factor + back-solve in isolation.     |
| `feral_cascade_break`        | (unset) | Tri-state. Unset → inherit feral's Phase B default (CB on with bounded delayed-pivot catchment). `yes` records explicit intent (no behavioural change). `no` reproduces pre-Phase-B behaviour by surfacing `DelayBudgetExceeded` on non-root cascade victims.  |
| `feral_fma`                  | `no`    | Dispatch dense kernels through fused multiply-add intrinsics. Roughly 2× throughput on aarch64 / x86_v3, at the cost of per-pivot rounding drift that trips more `WrongInertia` checks. Turn on when kernel throughput dominates and the IPM tolerates a noisier inertia signal. |
| `feral_singular_pivot_floor` | `1e-20` | Pounce's analog of MA57's `CNTL(2)`. After a successful factor, the smallest accepted `D`-block pivot magnitude (scaled space) is compared against this absolute floor; if it falls below, the factor is reported `Singular` so the IPM bumps `δ_w`. `0` disables. |

### `feral_ordering` variants

All six concrete and adaptive options live under the same string
option. `feral_ordering` also falls back to the
`POUNCE_FERAL_ORDERING` environment variable when not set on the
OptionsList.

| Value       | Strategy                                                                                                                                                                                                                                                  |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `auto`      | **Default.** Adaptive dispatcher: picks a concrete method per matrix from cheap pattern features. Branches: very-large-and-sparse (`n > 100 000`, avg degree < 5) → AMD; `n ≤ 10 000` → AMF; otherwise → MetisND. One symbolic pass; right when the heuristic shape rules apply (the common case). |
| `auto_race` | Race-based dispatcher: runs full symbolic factorization on AMD, MetisND, ScotchND, KahipND and keeps the smallest `factor_nnz`. ~4× a single symbolic pass, paid once per problem (symbolic factorization is cached across numeric refactorizations with the same pattern). Use when the cheap dispatcher's guess is suspect — e.g. `pinene_3200_0009`, where `auto` picks MetisND (88 s numeric factor) but `amd` factors in 19.5 s on the same matrix. |
| `amd`       | Approximate Minimum Degree (Amestoy/Davis/Duff). Pins AMD regardless of problem shape; robust default for IPM workloads. Best for very-large-and-sparse cases that the adaptive dispatcher already routes here.                                            |
| `amf`       | Approximate Minimum Fill (HAMF4 variant of Amestoy 1999). Strong on small-and-sparse populations (`n ≤ 10 000`); aggregate fill ≈ 0.87× AMD on feral's IPM small-sparse inventory.                                                                          |
| `metis`     | feral-metis multilevel nested dissection. Tends to produce squarer fronts than AMD on banded / nearly-1D structure; preferred for large structured matrices.                                                                                              |
| `scotch`    | feral-scotch nested dissection. Similar regime to METIS; alternative when METIS is unavailable or for cross-validation.                                                                                                                                   |
| `kahip`     | feral-kahip flow-based nested dissection with K1 preprocessing. Ties METIS on fill geomean at 4–6× per-call symbolic cost. Reach for it only when ND fill matters and per-call cost is amortized.                                                          |

When in doubt: leave `feral_ordering` at the default. When a hard
problem looks linear-solver-bound, try `feral_ordering auto_race`
before per-variant manual sweeping — it's the safe choice when the
per-problem winner is uncertain.

#### Caller-supplied ordering (`External`)

Beyond the string variants above, a structure-aware caller can inject a
**precomputed permutation** the generic AMD/METIS pass cannot see — a
block-triangular / Schur ordering (Parker, Garcia & Bent,
arXiv:2602.17968) or a tearing ordering from equation-oriented
decomposition. Because a permutation is a vector it cannot travel through
the string `feral_ordering` option; supply it programmatically instead:

- **Python:** `Problem.set_ordering(perm)` (and `get_ordering()` /
  `clear_ordering()`) — see [the Python guide](./python.md#caller-supplied-kkt-ordering-set_ordering).
- **Rust:** `IpoptApplication::set_external_ordering(perm)`.

`perm` is a 0-based, new-to-old permutation (`perm[k]` is the original
index that becomes index `k`) whose length must equal the **augmented KKT
system dimension** (variables + slacks + constraint duals), not the
problem's `n`. FERAL validates it as a bijection and fails the
factorization with an error on a wrong length or duplicate — a valid but
poor ordering only costs fill/time, never correctness. This maps to
FERAL's `OrderingMethod::External` (feral#107) and honors only the default
FERAL backend.

## Logging and colored output

POUNCE emits structured logs and a colored iteration table through the
[`tracing`](https://docs.rs/tracing) ecosystem. Behavior is governed by
environment variables (not solver options), so they apply to the `pounce`
CLI, the C/Python frontends, and anything embedding the library.

| Variable | Values | Effect |
|---|---|---|
| `RUST_LOG` | e.g. `info`, `debug`, `pounce::restoration=debug` | Log verbosity / per-target filtering. Default `info`. Logs go to **stderr**. |
| `POUNCE_LOG_FORMAT` | `text` (default) · `json` | `json` emits line-delimited JSON on stderr (incl. the per-iteration `pounce::iteration` stream) for Studio / CI ingestion. |
| `NO_COLOR` | set to any value | Disables ANSI color in the iteration table **and** logs (see <https://no-color.org>). |
| `CLICOLOR_FORCE` | set to any value | Forces color even when stdout is not a terminal. |

**Filtering by subsystem.** Solver internals log under namespaced targets
— `pounce::algorithm`, `pounce::linsol`, `pounce::mu`, `pounce::sqp`,
`pounce::linesearch`, `pounce::restoration`, `pounce::presolve`,
`pounce::py`. For example, to trace only the restoration phase:

```sh
RUST_LOG=pounce::restoration=debug pounce problem.nl
```

**Program output vs. logs.** The iteration table, the final summary, and
`--dump` diagnostics are *program output* on **stdout**; diagnostic and
progress messages are *logs* on **stderr**. Redirecting one does not
affect the other:

```sh
pounce problem.nl > result.txt 2> solve.log
```

**Color.** The iteration table is colored with a tiger/rust theme:
restoration lines take a background that varies by restoration kind
(soft-stay → tan, soft-exit → amber, hard → deep rust), and the row text
shades from black toward red as the primal step length `alpha` shrinks
(stalling). Color is emitted only when stdout is a terminal; redirected
output and `NO_COLOR` get plain text with identical column alignment.

**Machine-readable iterations.** `POUNCE_LOG_FORMAT=json` turns the
per-iteration records into JSON on stderr:

```sh
POUNCE_LOG_FORMAT=json pounce problem.nl 2> iters.jsonl
```
