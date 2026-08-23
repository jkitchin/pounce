# The CLI is the only complete frontend — a capability audit

## Why this note exists

Ten PRs from an outside contributor (@GermanHeim, who maintains `oximo` on
top of `pounce-rs`) turned out to be eight instances of one defect class:

> Capability either lives inside `crates/pounce-cli/`, or is wired only on
> the `.nl` path. Every other entry point silently gets less.

#207 (option registration), #363 (presolve bypassed by `optimize_tnlp`),
#583 (convex engines discarded `max_wall_time`), #595 (SOCP bounds),
#743 (PSD classifier), #754 (convex option parsing) are all that shape.
They were found one release at a time, by a downstream user, because the
test suite and the fixture corpus are `.nl`-shaped and run through the CLI:
a hole on the library path is invisible to every guard we have.

This note is the result of sweeping for the remaining instances *before*
they are reported, so that a per-route guard has a clean baseline to hold.

## The route map

The split that matters is not language, it is **in-process vs. subprocess**:

| Frontend | Route | Gets CLI-local behavior? |
|---|---|---|
| `pounce` CLI | native | yes |
| `pyomo-pounce` | writes `.nl`, execs the `pounce` binary | yes |
| `pounce-rs` | `IpoptApplication::optimize_tnlp` | no |
| `python/pounce` (`Problem`/`Solver`) | pyo3 → `optimize_tnlp` | no |
| `pounce-cinterface` | → `optimize_tnlp` | no |
| `pounce-wasm` | → `optimize_tnlp` | no |
| GAMS pip link | built on the Python `Problem` | no |

Only two route-asymmetry markers exist today: `convex_routing_available`
(default `false`, CLI sets it `true`) and `unhonored_option_file_name`.
Everything else that is asymmetric is asymmetric *silently*.

## Finding 1 — the local-infeasibility second-opinion ladder is CLI-only

`crates/pounce-cli/src/main.rs:1580-1660` implements a two-rung ladder that
re-solves on a different trajectory before shipping an
`Infeasible_Problem_Detected` verdict (rung 1 `feral_scaling=mc64`, rung 2
`mu_strategy=adaptive`). **Both rungs are on by default.**

Both gate options — `feral_infeasibility_scaling_retry` and
`infeasibility_mu_strategy_retry` — are registered core-side in
`upstream_options.rs`, so every frontend parses and accepts them, and only
the CLI reads them. The registry docstring already says so: *"Driven by the
pounce CLI; library embedders that inject their own restoration provider
implement the retry themselves."*

This is the highest-severity item in the audit because it is not a missing
knob — it is a different **verdict** for the same model. A false
`Infeasible_Problem_Detected` is the dangerous kind of wrong answer for a
branch-and-bound driver (see `false_local_infeasibility.rs`): a false
*unbounded* is loud and retryable, a false *infeasible* silently prunes a
node that may contain the optimum. `oximo` is exactly such a driver.

**Extraction is cheap.** The decision logic is already pure and factored:
`second_opinion_rungs(SecondOpinionAvailability) -> Vec<SecondOpinionRung>`,
`scaling_retry_promoted(status) -> bool`, and
`resolve_scaling_retry_outcome(...)`. They are private `fn`s in the binary's
`main.rs`, so not even `pounce_cli`-the-library can reach them. Only the
"apply the assignments and re-run" driver is orchestration.

### Caveat, measured: it does not currently diverge on any repo fixture

Running all **71** `.nl` fixtures in `crates/pounce-cli/tests/fixtures/`
twice — once at defaults, once with both retries disabled, which is exactly
what an in-process frontend gets — produced **zero** status differences.

Two consequences, and the second is arguably a finding of its own:

1. The practical impact of the gap is currently unmeasured, not "known bad".
   Do not describe library callers as getting wrong verdicts today.
2. **The ladder has no end-to-end coverage.** Its unit tests exercise
   `second_opinion_rungs` and `resolve_scaling_retry_outcome` in isolation;
   no fixture makes the retry actually fire. `cresc4.nl` (gh#524's case) now
   solves at baseline — the conditioning fix in `linear_system_scaling.rs`
   overtook it — and `discs.nl`, the canonical motivating model, is not in
   the repo at all. A ladder that never runs in CI can rot silently.

## Finding 2 — `check_x0::check_tnlp` is library-shaped and stuck in the CLI

`crates/pounce-cli/src/check_x0.rs:616` is generic over `&mut dyn TNLP`,
with no `.nl` and no CLI coupling, and its own doc says it is public "so the
debugger / tests can reuse it without going through a file." Nothing but
crate placement stops a `pounce-rs` user from preflighting a starting point
— non-finite evaluations at x0, bound violations, interior-clamp
displacement, scaling spread.

Same one-move extraction as #743 and #754. Best effort-to-value ratio here.

## Finding 3 — convex routing exists twice, and is missing from the third frontend

`crates/pounce-cli/src/qp_extract.rs` (2,169 lines, `&NlProblem`-coupled)
and `python/pounce/_route.py` (617 lines, callable-coupled) independently
implement the same classify-and-extract decision. `pounce-rs` has neither:
`application.rs:1241` refuses `lp-ipm`/`qp-ipm`/`socp` outright.

Two implementations of one numerical decision in two languages is the #755
drift shape at ~20x the scale. A TNLP-based classifier in Rust would serve
`pounce-rs` and could back the Python one.

## Finding 4 — `verify.rs` is CLI-only, which undercuts its own rationale

`crates/pounce-cli/src/verify.rs` argues in its header that independent
verification matters because "pounce is a *tool an agent calls*" and trust
belongs to a small deterministic checker rather than the solver's own exit
string. But an agent driving `pounce-rs` — which is what `oximo` is — cannot
reach it. The core check is `g_l <= g(x*) <= g_u` over a TNLP.

## Finding 5 — smaller CLI-trapped generics

- `counting_tnlp.rs` (184 lines) — evaluation-count wrapper, generic over
  TNLP. "How many f/g/H evaluations did that take" is a reasonable library
  question.
- `seeded_tnlp.rs` (134 lines) — primal warm-start-from-iterate wrapper,
  generic over TNLP.
- `builtin.rs` (818 lines) — self-contained `impl TNLP` test problems; would
  serve library consumers testing their own integration.
- `cbf.rs` (867 lines) — a self-contained Conic Benchmark Format parser with
  no CLI or `.nl` coupling at all. Niche, but it is pure library code.

## Finding 6 — a stale pointer in a user-facing message

`application.rs:1417` tells users hitting the convex-option refusal
"Tracking issue: https://github.com/jkitchin/pounce/issues/604". Issue #604
is closed and is about cold-start initialization options — unrelated.

## Explicitly checked and NOT a gap

- **Presolve** — #363's fix holds; `optimize_tnlp` materializes the presolve
  wrapper at the public entry point (`application.rs:880`).
- **Diagnostics / `--dump`** — `DiagnosticsState` lives in `pounce-common`
  and is library-reachable; only argument parsing is CLI-local.
- **`pounce-sensitivity`** — its own crate. `sens.rs` is only the `.nl`
  suffix plumbing.
- **`nl_hessian_program.rs`** — a `Tape` optimization, and `Tape` exists only
  for `.nl` input, so library callers supplying their own derivatives lose
  nothing.
- **The option-registry sweep is otherwise clean.** Of 435 registered
  options with 284 read sites, exactly three are read only inside
  `pounce-cli`: the two retry gates above, plus `sb` (suppress-banner, where
  CLI-only is arguably correct).

## The guard this baseline is for

`no_silent_options` is excellent for what it does, but it is **per-option,
not per-(option x route)**: `read_all_rust_sources(&crates)` scans the whole
tree for *any* read site. `max_wall_time` was read by the NLP path
throughout the #583 bug, so the guard stayed green while both convex engines
discarded it. Same for `solver_selection` before #207.

Extending it to a route matrix is what stops this class from recurring —
but only once the findings above are cleared, or the guard just codifies
them as permanent exceptions.

## Method

Route classification by grepping entry points for `optimize_tnlp` vs.
subprocess invocation. Option sweep by replicating `no_silent_options`'s
accessor grammar and partitioning read sites by crate
(`scratchpad/routescan.py`, reproduced in this note's history). Divergence
sweep by running the debug binary over all 71 fixtures with the retry gates
on and off and diffing `Status:`. Module classification by counting
`NlProblem` vs `dyn TNLP` vs `ExitCode` references per file.

Not covered: the C interface and wasm surfaces were classified by route but
not swept for their own capability gaps; #482's thread-local half is
unverified (its rayon half appears resolved upstream in feral#156).
