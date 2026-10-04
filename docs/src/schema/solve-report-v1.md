# POUNCE solve-report schema, v1

**Schema tag:** `pounce.solve-report/v1`

This document is the canonical reference for the JSON solve report
emitted by `pounce --json-output PATH` and `pounce_sens --json-output
PATH`. The report carries everything an AMPL `.sol` file holds —
status, primal `x`, dual `lambda`, suffix blocks — plus FAIR-aligned
provenance metadata and (optionally) the per-iteration trajectory.

Implementation: the serde structs live in [`crates/pounce-solve-report/src/lib.rs`](https://github.com/jkitchin/pounce/blob/main/crates/pounce-solve-report/src/lib.rs) (per-iteration `IterRecord` in `crates/pounce-nlp/src/solve_statistics.rs`); [`crates/pounce-cli/src/solve_report.rs`](https://github.com/jkitchin/pounce/blob/main/crates/pounce-cli/src/solve_report.rs) wires them to the CLI.

## Why a structured solve report?

Production NLP workflows often need to (a) capture which solve
produced which numbers for audit / reproducibility, (b) feed solver
output into downstream tooling (notebooks, dashboards, ML pipelines)
that don't want to parse a free-form `.sol` file, and (c) compare
runs across versions of pounce. Both upstream Ipopt's stdout summary
and AMPL's `.sol` were designed for human consumption and AMPL's
reader respectively — neither carries provenance metadata, neither is
schema-versioned, and neither is trivially machine-parseable across
ecosystems.

A versioned JSON schema with FAIR-aligned provenance solves all three.

## FAIR alignment

The `fair_metadata` block maps onto the four FAIR principles
(Wilkinson et al. 2016, "The FAIR Guiding Principles for scientific
data management and stewardship", *Scientific Data* **3**, 160018, DOI
[10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18); citation
verified via Crossref on 2026-05-14):

| Principle | Mapping in this schema |
|---|---|
| **F**indable | `result_id` (`<unix_nanos>-<pid>`, globally unique and time-ordered), `created_at_iso`, `created_at_unix_nanos`. |
| **A**ccessible | Plain-text JSON on disk; no protocol gating; UTF-8. Same trust model as the `.sol` file. |
| **I**nteroperable | Schema-versioned (`pounce.solve-report/v1`); JSON primitives only (no binary blobs); units documented per-field below; `solution.status` is the enum-variant string for cross-language consumption, beside `solution.status_upstream` in IPOPT's own enumerator spelling. |
| **R**eusable | `solver` (name + version + git commit + target triple), `license`, `input` (kind + path + size), and `environment` (solve-affecting env-var overrides in force) capture enough provenance to reproduce a solve. |

## Versioning policy

`schema` is the version tag. Compatibility rules:

* **Adding fields** is non-breaking. Consumers MUST tolerate unknown
  fields. New optional fields land between versions; the major version
  doesn't bump.
* **Removing or renaming fields** bumps the major version (`v1` →
  `v2`). Consumers should pin against a major version (`schema
  starts_with "pounce.solve-report/v1"`).
* **Changing field semantics** without a rename is forbidden. If
  semantics need to change, add a new field and deprecate the old.

The pre-1.0 phase of POUNCE itself does NOT relax this rule for the
schema. Once a solve-report version ships, its field set is frozen
even while the rest of the solver is under churn.

## Top-level shape

```json
{
  "schema": "pounce.solve-report/v1",
  "fair_metadata": { ... },
  "problem":       { ... },
  "solution":      { ... },
  "statistics":    { ... },
  "iterations":    [ ... ],  // optional, omitted when empty
  "linear_solver": { ... }   // optional, omitted when backend did not report
}
```

## Fields

### `schema` (string, required)

Identifier for this schema version. Always
`"pounce.solve-report/v1"` for v1. Major-version bumps change the
prefix; minor / patch (additive) changes do not.

### `fair_metadata` (object, required)

| Field | Type | Notes |
|---|---|---|
| `result_id` | string | Format: `<unix_nanos>-<process_id>`. Monotonically ordered within a process, globally unique across processes. No external UUID library needed. |
| `created_at_iso` | string | Solve start time as ISO-8601 UTC: `YYYY-MM-DDTHH:MM:SS.sssZ`. |
| `created_at_unix_nanos` | integer | Same instant as Unix nanoseconds since 1970-01-01 UTC. Provided alongside the ISO string for consumers that prefer integer arithmetic. |
| `elapsed_seconds` | float | Wallclock seconds the solve took (matches `statistics.total_wallclock_time_secs` modulo float precision). |
| `solver` | object | See below. |
| `license` | string | SPDX identifier. Always `"EPL-2.0"` for this version. |
| `input` | object | See `Input descriptor` below. |
| `environment` | array \| omitted | Solve-affecting environment overrides in force. Omitted when none are set. See `Environment overrides` below. |

#### `solver` sub-object

| Field | Type | Notes |
|---|---|---|
| `name` | string | Always `"pounce"`. |
| `version` | string | Crate version (e.g. `"0.1.0"`). Read from `CARGO_PKG_VERSION` at build time. |
| `git_commit` | string \| omitted | Build-time git revision. Omitted when the build environment did not set `POUNCE_GIT_COMMIT` (e.g. development builds). Set via `POUNCE_GIT_COMMIT=$(git rev-parse HEAD) cargo build` to populate. |
| `target_triple` | string | Build target triple (e.g. `"x86_64-apple-darwin"`); falls back to `"unknown"` when Cargo did not expose `TARGET` at build time. |

#### `Input descriptor` (`input`)

Tagged enum keyed on `kind`. Possible shapes:

```json
{ "kind": "nl-file", "path": "/path/to/foo.nl", "size_bytes": 366 }
{ "kind": "builtin", "name": "rosenbrock" }
{ "kind": "tnlp-direct" }
```

* `nl-file` — the input came from `.nl` file at `path`. `size_bytes`
  is present when the file's metadata is readable; consumers that want
  bit-exact provenance can hash the file themselves.
* `builtin` — the input was a built-in problem named by `name` (e.g.
  `pounce --problem rosenbrock`).
* `tnlp-direct` — used by library callers building a TNLP in-process
  without a `.nl` round-trip.

#### `Environment overrides` (`environment`)

An array of `{ "name", "value" }` objects, one per solve-affecting
environment variable set in the process at report time:

```json
"environment": [
  { "name": "POUNCE_FERAL_PIVTOL", "value": "1e-6" }
]
```

The whole array is omitted when none are set (the common case). Only the
variables that change pounce's numerics or parallelism are captured — the
`POUNCE_FERAL_*` linear-solver knobs and the legacy `FERAL_PIVTOL` /
`FERAL_PARALLEL`. These alter the factorization and can otherwise silently
differ a run between two machines (e.g. one with `POUNCE_FERAL_PIVTOL`
exported in a shell profile) with nothing in the report saying so. The
`POUNCE_DBG_*` debug gates are deliberately **not** captured — they only
add diagnostic output and never change the result.

Presence records that the variable was *set*, not that it took effect: an
explicit `OptionsList` setting (e.g. `feral_pivtol` in an options file)
takes precedence over the env fallback. See
[Options › Environment overrides](../options.md#environment-overrides-feral-and-debug-gates)
for the option each variable maps to.

### `problem` (object, required)

Problem dimensions reported by the TNLP at `get_nlp_info()`.

| Field | Type | Notes |
|---|---|---|
| `n_variables` | integer | Number of primal variables. |
| `n_constraints` | integer | Number of constraints (equalities + inequalities). |
| `n_objectives` | integer | Number of objectives. The IPM uses objective 0; extras are read but ignored. |
| `minimize` | boolean | `true` for minimization (the AMPL default), `false` for a `maximize` model. Read it to interpret the sign of `solution.objective` and `solution.lambda`, both of which are in the sense this field names. Reported by every arm since 0.12.0; the general NLP arm left it at its `true` default before that, so a maximize model reads `true` in an older report ([#959](https://github.com/jkitchin/pounce/issues/959)). |
| `nnz_jac_g` | integer \| omitted | Number of declared non-zeros in the constraint Jacobian. |
| `nnz_h_lag` | integer \| omitted | Number of declared non-zeros in the lower triangle of the Lagrangian Hessian. |

### `solution` (object, required)

| Field | Type | Notes |
|---|---|---|
| `status` | string | `ApplicationReturnStatus` enum variant name verbatim (e.g. `"SolveSucceeded"`, `"MaximumIterationsExceeded"`). |
| `status_upstream` | string | The same verdict in upstream IPOPT's C enumerator spelling from `IpReturnCodes_inc.h` (e.g. `"Solve_Succeeded"`, `"Infeasible_Problem_Detected"`) — the spelling CUTEst status tables, the CLI's own `Status:` line and the reference JSONs under `benchmarks/*/ipopt_ma57.json` all use. Derived from `status`, so the two can never disagree. Compare against this field, not `status`, when your consumer already keys off IPOPT's names. Added after `pounce.solve-report/v1` shipped; absent from reports written by pounce ≤ 0.10.0. |
| `solve_result_num` | integer | AMPL-style solve-result code (Gay 2005, "Hooking Your Solver to AMPL" §5, p. 23 table): 0 = solved, 100-range = warning, 200-range = infeasible, 400-range = limit reached, 500-range = failure. Within the solved range, `0` is `SolveSucceeded` and `1` is `SolvedToAcceptableLevel` — IPOPT's codes ([solution output](../solution-output.md#solved-strict-acceptable-and-square)). Identical to the `objno` code in the `.sol`. |
| `objective` | float | Final unscaled objective value, **in the sense the model declares** — a `maximize` model reports the value of the objective as written, not of the internal minimization. `0.0` (not NaN) when the solve never completed; check `statistics.iteration_count > 0` to distinguish. |
| `x` | array of float \| empty | Primal vector, length `problem.n_variables`. Empty when the binary doesn't capture the final iterate (currently: `pounce` on the `newton_driver` fast-path). Omitted from JSON when empty. |
| `lambda` | array of float \| empty | Constraint multipliers, length `problem.n_constraints`. Same omission convention as `x`. **Convention:** `σ·λ`, where `λ` is the multiplier of the internal minimization (the `λ` of `L = f + λᵀg − z_L(x−x_L) + z_U(x−x_U)`) and `σ = +1` for a `minimize` model, `−1` for `maximize` (gh#959), i.e. the multipliers in the model's declared sense. They are Lagrange multipliers, **not** the `.sol` shadow prices, which carry the opposite sign (see [Dual sign conventions](../cli.md#dual-sign-conventions)). |
| `suffixes` | array of object \| empty | sIPOPT-style suffix blocks; emitted only at `--json-detail full`. See below. |

#### Suffix entries

```json
{
  "name": "sens_sol_state_1",
  "target": "var",
  "kind": "real",
  "values": [0.576..., 0.378..., -0.046..., 4.5, 1.0]
}
```

| Field | Type | Notes |
|---|---|---|
| `name` | string | AMPL suffix name. |
| `target` | string | One of `"var"`, `"con"`, `"obj"`, `"problem"`. Matches AMPL's `Sufkind_*` enum. |
| `kind` | string | `"real"` or `"int"`. Selects which payload array is populated. |
| `values` | array of float | Dense values (length = target dimension). Present when `kind = "real"`. |
| `int_values` | array of integer | Present when `kind = "int"`. |

### `statistics` (object, required)

Projection of `pounce_nlp::solve_statistics::SolveStatistics` minus
the per-iteration history (which lives at the top level when present).

| Field | Type | Notes |
|---|---|---|
| `iteration_count` | integer | Number of accepted outer iterations. On a multi-pass solve (the ℓ₁ exact-penalty ρ loop, or the plain attempt plus the ℓ₁ retry) it is the **total across all passes** (gh#987); `passes` gives the split. |
| `final_objective` | float \| null | Unscaled. Matches `solution.objective`. `null` if never computed — see below. |
| `final_scaled_objective` | float \| null | Scaled by the IPM's internal NLP scaling. Equal to `final_objective` when no scaling is in effect. `null` if never computed. |
| `final_dual_inf` | float \| null | `||∇L||∞` at termination. `null` if never computed — see below. |
| `final_constr_viol` | float \| null | Primal infeasibility `‖(c(x), d(x) − s)‖∞` in the **internally scaled** slack form — the residual the convergence test reads, the same quantity as the last row's `inf_pr_internal`, after a restoration exit too. On a badly scaled model it can be orders below the printed `inf_pr`; the violation in the model's own units at the returned point is `final_declared_constr_viol`, which the NLP arm fills on every solve (gh#981; since gh#987 also at `bound_relax_factor=0`, where it was `null`). The convex arm fills it on every solve too (gh#987 second pass; it is `final_constr_viol` itself when no widening was applied, and the declared-model measurement when one was). `null` if never computed. |
| `final_compl` | float \| null | Max complementarity over the four bound blocks. `null` if never computed. |
| `final_kkt_error` | float \| null | Overall KKT error reported by the convergence check. `null` if never computed. |

> **`null` values.** The four residuals are produced by the convergence check at the
> end of a solve. A solve the solver *refused* — rejected during setup
> (`NotEnoughDegreesOfFreedom`, `InvalidProblemDefinition`), aborted, or caught
> by the batch panic handler — never reaches it, and these slots are emitted as
> `null` rather than `0.0`. A zero there is indistinguishable from a perfect
> solve, and consumers acted on it: it was enough to make `pounce.minimize`
> report `success=True` for a problem the solver had declined to attempt.
>
> The two objective fields follow the same rule for the same reason: `0.0` is
> an ordinary objective value, so it cannot signal "never evaluated". They are
> seeded from the current iterate whenever one exists, so they are `null` only
> when the solve produced no point at all.
>
> Consumers should treat `null` as "not computed", not as zero. pounce's own
> readers map it to NaN, which fails closed against any `value <= tol` test.

> **Objective sense.** Both objective fields are in the sense `problem.minimize`
> names, on every engine. POUNCE solves a `maximize` model internally as
> `min −f`, so the algorithm's own numbers are negated; the report carries them
> back. The **console** residual table does not, and deliberately: that block is
> diffed against IPOPT's own output, and upstream prints the internal value
> there (`AmplTNLP::eval_f` returns `obj_sign · objval`). On a maximize model
> the CLI prints the declared-sense value on its own line below the table
> instead. Before 0.12.0 the general NLP arm reported the internal value in the
> report as well, so one binary gave `−36` and `+36` for the same file
> depending on which engine answered
> ([#959](https://github.com/jkitchin/pounce/issues/959)).
| `num_obj_evals` | integer | `eval_f` call count. |
| `num_constr_evals` | integer | `eval_g` call count. |
| `num_obj_grad_evals` | integer | `eval_grad_f` count. |
| `num_constr_jac_evals` | integer | `eval_jac_g` count. |
| `num_hess_evals` | integer | `eval_h` count. |
| `total_wallclock_time_secs` | float | Wall time spent inside `optimize_*`. |
| `restoration_calls` | integer | Number of restoration-phase entries (pounce#12). |
| `restoration_inner_iters` | integer | Cumulative inner-IPM iterations across all restoration calls. |
| `restoration_outer_iters` | integer | Outer iterations that ran in restoration mode (`R`-line equivalents). |
| `restoration_wall_secs` | float | Wall time spent inside `perform_restoration`. |
| `quality_escalations` | integer | Times the linear solver escalated its factorization (`IncreaseQuality`) during this solve, restoration sub-solves included (pounce#857). `0` on every backend that cannot escalate and on any solve that never stalled. Present since the field was added; reads `0` on reports written before it, since it deserializes with a default. It is the only trace an escalation leaves — status, objective, iteration count and engine are all unchanged by one, and the console's `q` info-string flag misses every escalation taken inside restoration, because those rows carry no info column. **On a laddered run this is the promoted solve's count, not the base solve's** — the same rule `iteration_count` follows. It is a sharp edge here because `feral_increase_quality_retry` promotes a re-solve that by construction escalated zero times, so a run whose base solve escalated twenty-five times reports `0` once the recovery lands; `second_opinion.base_status` records what it recovered from, and `feral_increase_quality_retry=no` reproduces the base solve outright. |
| `warnings` | array of string | Structured solve-quality warnings about the **returned point** (gh#983), each `"<code>: <text>"`. Codes: `objective_scale_small` (the objective gradient is tiny and nothing scales it up, so the absolute complementarity floor is a visible fraction of the objective's scale), `unscaled_stationarity_above_tol` (gradient-based scaling froze a tiny factor at a huge-gradient start and a re-solve with the scaling re-evaluated did not fix it), `large_dual_scale` (the stationarity residual is made of terms of `1e8` or more: failed-constraint-qualification symptom), `unscaled_dual_inf_above_acceptable` (the unscaled dual infeasibility exceeds `acceptable_tol` although the strict gate, which tolerates `dual_inf_tol`, passed). `integer_relaxation` (the CLI solved a `.nl` that declares binary / integer variables as its continuous relaxation, so `status` `solved` is a bound on the MIP optimum, not an integer-feasible point; `solve_result_num` stays in the solved band because Pyomo and AMPL read the `0-99` band as "load this solution", and gh#987). Empty on a clean run; never changes `status`. Reads `[]` on reports written before the field existed. On a laddered run these describe the run that produced the returned point. |
| `passes` | array of object \| omitted | Per-pass summary of a multi-pass solve (gh#987), omitted on an ordinary single-pass solve. One entry per ℓ₁ exact-penalty ρ pass, preceded by an entry with `rho` omitted for the plain attempt when `l1_fallback_on_restoration_failure` ran it first. Entry fields: `rho` (penalty parameter), `iterations` (that pass), `first_row` (index into the concatenated `iterations` array where the pass's rows start; each pass restarts its own `iter` at 0), `slack_sum` (`Σ(p+n)` of the augmented slacks, the BNW steering signal), `constraint_violation` (max violation of the **model's own** constraints at the pass's end, model units; omitted when unmeasurable), `status` (the pass's `ApplicationReturnStatus` variant). Every other `statistics` field describes the last pass, which produced the returned point; `iteration_count` and `iterations` cover all passes. |

Eval counters (`num_*_evals`) populate only on the `.nl`-file path
because the `pounce` binary's `CountingTnlp` wrapper tracks them.
Library callers using `IpoptApplication::optimize_tnlp` directly see
zeros there; the underlying counts are still available through
upstream's `IpoptCalculatedQuantities` if needed.

### `iterations` (array of object, optional)

Per-iteration trajectory. Emitted only at `--json-detail full` (when
`IpoptApplication::enable_iter_history()` was called). Omitted from
JSON entirely when empty.

Each row is one line of the console iteration table, in order, and
carries the numbers that line prints — restoration-phase rows (the
`r`-suffixed lines) included, tagged `"phase": "restoration"`. On those
rows `objective` and `inf_pr` are the *original* problem's at the
restoration iterate, exactly as printed, and in the same units as on the
main rows — unscaled objective, and the user's constraints in user units
against their declared bounds (gh#981; they used to be the row-scaled
residual, so a badly scaled model's restoration rows read orders below
the main rows beside them). The remaining columns belong to the
restoration sub-solve. A restoration row's `iter` continues the outer
count, so the main-phase `R` row that leaves restoration repeats the last
restoration row's index, as the console does (gh#979). Fields:

| Field | Type | Notes |
|---|---|---|
| `iter` | integer | 0-based iteration index. |
| `phase` | string | `"main"` or `"restoration"`. Absent in reports from 0.12.x and earlier — read it as `"main"`. |
| `objective` | float | `f(x_k)` at the start of iter `k` (unscaled), as printed. |
| `inf_pr` | float | Primal infeasibility as printed in the `inf_pr` column: under the default `inf_pr_output=original`, the unscaled max-norm violation of the user's constraints. |
| `inf_pr_internal` | float | The algorithm's internal residual `‖(c(x), d(x) − s)‖∞` in the scaled slack form — what the filter, the convergence test and the `intermediate` callback's `inf_pr` read. Differs from `inf_pr` while a slack sits off `d(x)` (e.g. 0.955 vs 0.94 at a start where `g = 2.44 > c_u = 1.5`); on restoration rows it is the restoration problem's own residual. |
| `inf_du` | float | Dual infeasibility `||∇L_k||∞`. |
| `mu` | float | Barrier parameter μ_k (not log10; consumers can take `log10` if they want the console format). |
| `d_norm` | float | `||d_xs||∞` of the search step taken at iter `k-1` to land at iter `k`. `0.0` at iter 0. |
| `regularization` | float | Hessian regularization `δ_w` applied this iter; `0.0` when none was needed. |
| `alpha_dual` | float | Dual step length. |
| `alpha_primal` | float | Primal step length. |
| `alpha_primal_char` | string (1 char) | Single-character tag matching the alpha-primal column of upstream's iter table. See [the step characters](#step-characters-alpha_primal_char) below. |
| `ls_trials` | integer | Number of backtracking line-search trials this iter. |

#### Step characters (`alpha_primal_char`)

The letter says how the step was accepted. What it can tell you depends
on `line_search_method` (gh#981):

| Char | Meaning |
|---|---|
| `f` | **Filter only.** An *f-type* step: the switching condition held and the step passed the Armijo test on the barrier objective, so the filter was not augmented. |
| `h` | **Filter:** an *h-type* step, which made sufficient progress on θ or φ against the filter and augmented it. **Penalty:** *every* accepted step. The penalty acceptor has no f/h distinction, so under `line_search_method=penalty` the letter carries no information. |
| `F` / `H` | As `f` / `h`, but the step that was accepted is a **second-order correction**. Upper case is the SOC marker (`backtracking.rs`, `mode.to_ascii_uppercase()`). |
| `s` | A **soft-restoration** step (Ipopt's `in_soft_resto_phase_`), accepted on primal-dual error reduction; the line search stays in soft restoration. |
| `S` | A soft-restoration step that is also acceptable to the original acceptor, so soft restoration ends. Not a second-order correction: SOC only ever produces `F` / `H`. |
| `w` | **Watchdog** accept-anyway: the last trial was accepted despite the acceptor rejecting it, and the filter was not augmented. |
| `t` / `T` | Tiny step: the search direction was below `tiny_step_tol`, so the full step was taken without a line search. `T` means the previous iteration was tiny too. |
| `R` | The row where the main phase hands off to, or takes back from, the restoration phase. The restoration phase's own rows are tagged `"phase": "restoration"` (printed with an `r` suffix). |
| ` ` (blank) | No acceptor was consulted: iteration 0, or `accept_every_trial_step=yes`. |

A run of `s` with no `R` means soft restoration is making progress on
the primal-dual error without ever handing off to full restoration. Under
`line_search_method=penalty` that can continue until `max_iter`; see
gh#981.

### `linear_solver` (object, optional)

Aggregate post-mortem from the symmetric-indefinite linear backend
that solved the KKT systems. Populated only when the backend
self-instruments (the default FERAL backend does; HSL MA57 and
custom backends plugged through `set_linear_backend_factory` do not).
Omitted from JSON when no backend reported.

| Field | Type | Notes |
|---|---|---|
| `solver_name` | string | Backend identifier (e.g. `"feral"`). |
| `n_factors` | integer | Total numeric factorizations performed. |
| `n_pattern_reuse` | integer | Factor calls that reused the existing symbolic pattern. |
| `n_pattern_changes` | integer | Factor calls that triggered a re-analysis. |
| `max_fill_ratio` | float \| omitted | Peak `nnz(L) / nnz(A)` observed across all factorizations. |
| `min_abs_pivot` | float \| omitted | Smallest absolute pivot magnitude seen across all factorizations (diagnostic for near-singularity). Measured in FERAL's **equilibrated** space, not the model's units — compare runs, do not read it as a curvature scale. |
| `max_abs_pivot` | float \| omitted | Largest absolute pivot magnitude, in the same equilibrated space: it reads `3.0` whether a Hessian entry is `2` or `2e6` (gh#990), so it says nothing about the model's scale. |
| `last_inertia` | `[int, int, int]` \| omitted | `(positive, negative, zero)` inertia of the final factor, **after** regularization: it describes the *corrected* matrix `K + diag(δ_w, −δ_c)`, not the Hessian the model supplied. A solve that ends at a local maximum or saddle with a large `δ_w` therefore still reads `(n, m, 0)`; the wrongly-signed curvature is visible only as the last `iterations[*].regularization` entry (`δ_w`), so read the two together (gh#987). Should match `(n, m, 0)` at a regular KKT optimum with `δ_w = 0`. |
| `last_inertia_unregularized` | `[int, int, int]` \| omitted | `(positive, negative, zero)` inertia of the most recent factorization attempted with **no** regularization (`δ_w = δ_c = 0`) — the KKT matrix's own curvature verdict, tried first on every iteration. At a local maximum where `last_inertia` reads `(n, m, 0)` after the shift, this reads e.g. `(0, 2, 0)`. Omitted when no unperturbed trial was recorded (a backend that reports no inertia, or a solve that never factored). Additive, gh#987. |
| `last_nnz_a` | integer \| omitted | Non-zero count of the assembled KKT matrix at the final factor. |
| `last_nnz_l` | integer \| omitted | Non-zero count of the L-factor at the final factor. |
| `total_factor_secs` | float | Wall-clock seconds inside the numeric factor call, summed over every factorization (regularization retries included). Always measured, independent of `timing_statistics`. On the Schur path, includes forming and factoring `S`. Divide by `statistics.iteration_count` for a per-iteration figure. |
| `total_factor_flops` | float | Sum of the backend's a-priori work proxy (FERAL: `Σ ncol·nrow²` over supernodes). For comparing orderings and sizes; not a time. |
| `total_delayed_cols` | integer | Delayed-column entries summed over factorizations. A column delayed up `k` tree levels counts `k` times. |
| `total_two_by_two` | integer | 2×2 pivot blocks summed over factorizations. |
| `total_n_tiny` | integer | Pivots statically perturbed to the pivot floor, summed. |
| `last_factor_flops` | float \| omitted | Work proxy of the final factor. |
| `last_peak_bytes` | integer \| omitted | Predicted peak memory (factor plus transient contribution blocks) of the final factor. |
| `last_n_supernodes` | integer \| omitted | Supernodes in the final factor's elimination tree. |
| `last_max_front_rows` | integer \| omitted | Rows in the final factor's largest frontal matrix. |
| `last_delayed_cols` | integer \| omitted | Delayed-column entries in the final factor. |
| `last_two_by_two` | integer \| omitted | 2×2 pivot blocks in the final factor. |
| `last_n_tiny` | integer \| omitted | Statically perturbed pivots in the final factor. |
| `last_ordering` | string \| omitted | Concrete ordering the final factor used (`amd`, `amf`, `metis`, `scotch`, `kahip`, `external`), never `auto`. Can differ from the requested `feral_ordering`: below `amd_switch` (120) FERAL uses an AMD leaf, so a pinned `scotch` / `metis` / `kahip` reports `amd` on a small matrix. |
| `schur` | object \| omitted | Present only when the block-triangular / Schur path (`set_kkt_schur_block`) actually factored; absent when it was not requested **or** fell back to the standard solver. Fields: `n_eliminated`, `n_schur`, `n_factors`, `eliminated_factor_secs`, `form_schur_secs`, `schur_factor_secs`. The eliminated block's factorizations are also counted in the fields above. |

| `restoration` | object \| omitted | The restoration phase's factorizations, in the same shape as this object (without a nested `restoration`). Present only when restoration factored. The fields above describe the main solve alone, so total factorization work is the sum of the two. Recorded by the CLI, Python and C frontends, which wire the restoration backend; a custom restoration factory records nothing. |

## Detail levels

The `--json-detail LEVEL` flag selects how much detail is emitted.
Levels map to verbosity in the same spirit as upstream's `print_level`
(0 silent → 12 maximum debug):

| Level | What's emitted | What's omitted |
|---|---|---|
| `summary` (default) | FAIR metadata, problem, solution scalars + arrays, aggregate statistics | `iterations`, `solution.suffixes` |
| `full` | All of the above plus per-iteration trajectory and suffix blocks | nothing — full detail |

`summary` is the right choice for production logs and batch runs.
`full` is the debugging equivalent of upstream's `print_level=8`.

## Worked example

`pounce_sens crates/pounce-cli/tests/fixtures/parametric.nl out.sol --json-output result.json --json-detail full` produces (truncated for brevity):

```json
{
  "schema": "pounce.solve-report/v1",
  "fair_metadata": {
    "result_id": "1778777029606881000-76543",
    "created_at_iso": "2026-05-14T16:43:49.606Z",
    "created_at_unix_nanos": 1778777029606881000,
    "elapsed_seconds": 0.011,
    "solver": {
      "name": "pounce",
      "version": "0.1.0",
      "target_triple": "x86_64-apple-darwin"
    },
    "license": "EPL-2.0",
    "input": {
      "kind": "nl-file",
      "path": "crates/pounce-cli/tests/fixtures/parametric.nl",
      "size_bytes": 366
    }
  },
  "problem": { "n_variables": 5, "n_constraints": 4, "n_objectives": 1, "minimize": true },
  "solution": {
    "status": "SolveSucceeded",
    "status_upstream": "Solve_Succeeded",
    "solve_result_num": 0,
    "objective": 0.5510204081632656,
    "x":      [0.6326530575201161, 0.3877551079678144, 0.020408165487930466, 5.0, 1.0],
    "lambda": [-0.16326530000405073, -0.28571431357898697, -0.16326530000405073, 0.18075803406303625],
    "suffixes": [{
      "name": "sens_sol_state_1",
      "target": "var",
      "kind": "real",
      "values": [0.5765305974643309, 0.3775510440570709, -0.04591835847859835, 4.5, 1.0]
    }]
  },
  "statistics": { "iteration_count": 9, "final_dual_inf": 2.89e-14, "...": "..." },
  "iterations": [
    { "iter": 0, "objective": 0.0451, "inf_pr": 5.0, "inf_pr_internal": 5.0,
      "inf_du": 0.407, "mu": 0.1, "d_norm": 0.0, "regularization": 0.0,
      "alpha_dual": 0.0, "alpha_primal": 0.0, "alpha_primal_char": " ",
      "ls_trials": 0, "phase": "main" },
    { "iter": 1, "objective": 0.957, "inf_pr": 0.212, "...": "..." }
  ]
}
```

## Consumer guidance

* **Pin the major version.** Check `schema.startswith("pounce.solve-report/v1")` before consuming.
* **Tolerate unknown fields.** New optional fields will land between minor versions of pounce. Use `serde(default)` / equivalent.
* **Distinguish "no solve" from "solve produced zero".** Pre-solve, scalar fields are `0.0` (not `NaN`, because JSON has no NaN literal). `statistics.iteration_count == 0` is the signal that no solve occurred.
* **`solution.x` / `solution.lambda` may be empty.** When the binary couldn't capture the final iterate (currently: the `pounce` binary on its `newton_driver` fast-path for `m=0, n≤1000` problems), the arrays are empty and the keys are omitted from JSON entirely. `pounce_sens` always populates them.

## References

* Wilkinson et al. (2016). "The FAIR Guiding Principles for scientific data management and stewardship." *Scientific Data* **3**, 160018. DOI [10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18). (Verified via Crossref 2026-05-14.)
* Gay (2005). "Hooking Your Solver to AMPL." <https://ampl.com/REFS/hooking2.pdf>. §5 (Returning Results to AMPL) for the `.sol` baseline this schema is structured around.
* SPDX license identifiers: <https://spdx.org/licenses/>.
