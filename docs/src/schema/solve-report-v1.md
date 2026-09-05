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
| `minimize` | boolean | `true` for minimization (the AMPL default). |
| `nnz_jac_g` | integer \| omitted | Number of declared non-zeros in the constraint Jacobian. |
| `nnz_h_lag` | integer \| omitted | Number of declared non-zeros in the lower triangle of the Lagrangian Hessian. |

### `solution` (object, required)

| Field | Type | Notes |
|---|---|---|
| `status` | string | `ApplicationReturnStatus` enum variant name verbatim (e.g. `"SolveSucceeded"`, `"MaximumIterationsExceeded"`). |
| `status_upstream` | string | The same verdict in upstream IPOPT's C enumerator spelling from `IpReturnCodes_inc.h` (e.g. `"Solve_Succeeded"`, `"Infeasible_Problem_Detected"`) — the spelling CUTEst status tables, the CLI's own `Status:` line and the reference JSONs under `benchmarks/*/ipopt_ma57.json` all use. Derived from `status`, so the two can never disagree. Compare against this field, not `status`, when your consumer already keys off IPOPT's names. Added after `pounce.solve-report/v1` shipped; absent from reports written by pounce ≤ 0.10.0. |
| `solve_result_num` | integer | AMPL-style solve-result code (Gay 2005, "Hooking Your Solver to AMPL" §5, p. 23 table): 0 = solved, 100-range = warning, 200-range = infeasible, 400-range = limit reached, 500-range = failure. Within the solved range, `0` is `SolveSucceeded` and `1` is `SolvedToAcceptableLevel` — IPOPT's codes ([solution output](../solution-output.md#solved-strict-acceptable-and-square)). Identical to the `objno` code in the `.sol`. |
| `objective` | float | Final unscaled objective value. `0.0` (not NaN) when the solve never completed; check `statistics.iteration_count > 0` to distinguish. |
| `x` | array of float \| empty | Primal vector, length `problem.n_variables`. Empty when the binary doesn't capture the final iterate (currently: `pounce` on the `newton_driver` fast-path). Omitted from JSON when empty. |
| `lambda` | array of float \| empty | Constraint multipliers, length `problem.n_constraints`. Same omission convention as `x`. |
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
| `iteration_count` | integer | Number of accepted outer iterations. |
| `final_objective` | float \| null | Unscaled. Matches `solution.objective`. `null` if never computed — see below. |
| `final_scaled_objective` | float \| null | Scaled by the IPM's internal NLP scaling. Equal to `final_objective` when no scaling is in effect. `null` if never computed. |
| `final_dual_inf` | float \| null | `||∇L||∞` at termination. `null` if never computed — see below. |
| `final_constr_viol` | float \| null | `||c(x)||∞` (primal infeasibility). `null` if never computed. |
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

Eval counters (`num_*_evals`) populate only on the `.nl`-file path
because the `pounce` binary's `CountingTnlp` wrapper tracks them.
Library callers using `IpoptApplication::optimize_tnlp` directly see
zeros there; the underlying counts are still available through
upstream's `IpoptCalculatedQuantities` if needed.

### `iterations` (array of object, optional)

Per-iteration trajectory. Emitted only at `--json-detail full` (when
`IpoptApplication::enable_iter_history()` was called). Omitted from
JSON entirely when empty.

Each row maps to one line of the upstream-formatted console iter
table. Fields:

| Field | Type | Notes |
|---|---|---|
| `iter` | integer | 0-based iteration index. |
| `objective` | float | `f(x_k)` at the start of iter `k` (unscaled). |
| `inf_pr` | float | Primal infeasibility `||c(x_k)||∞`. |
| `inf_du` | float | Dual infeasibility `||∇L_k||∞`. |
| `mu` | float | Barrier parameter μ_k (not log10; consumers can take `log10` if they want the console format). |
| `d_norm` | float | `||d_xs||∞` of the search step taken at iter `k-1` to land at iter `k`. `0.0` at iter 0. |
| `regularization` | float | Hessian regularization `δ_w` applied this iter; `0.0` when none was needed. |
| `alpha_dual` | float | Dual step length. |
| `alpha_primal` | float | Primal step length. |
| `alpha_primal_char` | string (1 char) | Single-character tag (`f`, `h`, `r`, etc.) matching the alpha-primal column of upstream's iter table. |
| `ls_trials` | integer | Number of backtracking line-search trials this iter. |
| `active_bounds` | integer, optional | Bound and slack indices the barrier currently treats as active. Omitted on paths that do not measure it (the convex IPM rows). |
| `active_set_changes` | integer, optional | How many of those indices changed class since the previous row. Omitted on the first row — there is no predecessor, which is not the same statement as a measured zero — and on paths that do not measure it. |

#### The phase profile

`active_bounds` and `active_set_changes` exist to answer one question:
how much of a run is spent *finding* the active set versus *converging*
on one. `active_set_changes` is large while the solve is still deciding
which constraints bind and falls to zero once it has decided; where that
happens divides a run into an approach and an endgame, which is what
decides whether a staged solve — coarse-to-fine, or a cheap approach
phase handed off to a more accurate endgame — can pay at all.

An interior-point iterate never sits *on* a constraint, so "active" here
is an inference: index `i` is counted active when its bound multiplier
exceeds its slack, the split of `s_i · z_i ≈ mu` between its two
factors. The barrier drives a strongly active bound to `s = O(mu)`
against `z = O(1)` and an inactive one to the reverse, so the comparison
separates them by the method's own geometry rather than by a tolerance.

**Read it within one run only.** The partition is frame-dependent: a
per-variable rescaling carries the slack by `d` and the multiplier by
`d⁻¹`, so the boundary moves while the product does not, and no
classifier built from `(s_i, z_i, mu)` alone can be scale-free — their
only dimensionless combination is `s_i z_i / mu ≈ 1`. The comparison is
taken in the solver's internal scaled frame, the one in which
`kappa_sigma` and the fraction-to-boundary rule already compare these
quantities. Do not compare counts across models or across scalings, and
do not read them as the scale-invariant activity classification: that is
`sens_covariance()`'s `classify_activity`, which normalizes by curvature
and costs a back-solve per index in its reduced form
(see [Sensitivity Analysis](../sensitivity.md)).

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
| `min_abs_pivot` | float \| omitted | Smallest absolute pivot magnitude seen across all factorizations (diagnostic for near-singularity). |
| `max_abs_pivot` | float \| omitted | Largest absolute pivot magnitude. |
| `last_inertia` | `[int, int, int]` \| omitted | `(positive, negative, zero)` inertia of the final factor. Should match `(n, m, 0)` at a regular KKT optimum. |
| `last_nnz_a` | integer \| omitted | Non-zero count of the assembled KKT matrix at the final factor. |
| `last_nnz_l` | integer \| omitted | Non-zero count of the L-factor at the final factor. |

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
    { "iter": 0, "objective": 0.0451, "inf_pr": 5.0, "inf_du": 0.407, "mu": 0.1,
      "d_norm": 0.0, "regularization": 0.0, "alpha_dual": 0.0, "alpha_primal": 0.0,
      "alpha_primal_char": " ", "ls_trials": 0, "active_bounds": 2 },
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
