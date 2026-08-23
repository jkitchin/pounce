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

### Caveat, measured: it does not currently change an outcome in-tree

Running all **71** `.nl` fixtures in `crates/pounce-cli/tests/fixtures/`
twice — once at defaults, once with both retries disabled, which is exactly
what an in-process frontend gets — produced **zero** status differences.

**That result is expected, and the repo already says so.** An earlier draft
of this note read it as "the ladder has no end-to-end coverage". That is
wrong on both halves, and the correction matters:

- The ladder *is* exercised end-to-end, on the CLI path, by
  `issue_508_infeasibility_gap_status.rs` (a re-solve runs and is not
  promoted; the `.sol` keeps the original verdict),
  `issue_693_relative_infeasibility_stash.rs` (reached through the
  `feral_scaling=mc64` rung), and `presolve_certified_infeasibility.rs`
  (the ladder is correctly suppressed when presolve has already certified
  infeasibility).
- The sweep's null result is the *asserted* state.
  `false_local_infeasibility.rs::cresc4_no_longer_needs_either_rung_of_the_ladder`
  pins it deliberately: gh#693 removed the damped `y0` that put cresc4's
  base trajectory in the wrong basin, so the monotone-mu path now reaches
  the optimum unaided and the test fails if the ladder starts carrying
  gh#524 again.

What is genuinely missing is narrower, and that test states it plainly in
its own docstring: nothing in-tree demonstrates that the ladder *changes an
outcome* on a real model. Rung 2's diversity property has no witness here,
and `discs.nl` — rung 1's canonical case — is not a fixture. The repo
records that as a known, tracked gap rather than shrugging it off.

So the honest reading of the sweep is: the practical impact of the
CLI-only-ness is unmeasured, because the corpus contains no model on which
the ladder changes anything for *anyone*. Do not describe library callers as
getting wrong verdicts today. The structural gap below stands on its own.

## Finding 0 (found while fixing Finding 1): the facade had no restoration phase

`pounce-rs` did not depend on `pounce-restoration` at all. Neither did
`pounce-wasm`. The CLI, `pounce-cinterface` and `pounce-py` all did.

`IpoptApplication` runs a restoration phase only if a caller installed one —
`pounce-algorithm` cannot build the provider itself, because
`pounce-restoration` depends on *it* rather than the reverse. So the wiring
lived as the same ten lines pasted into four frontends, and the two that
never pasted it silently solved worse: a model needing restoration stopped at
`Restoration_Failed` where the CLI returned a real verdict.

This is the sharpest form of the pattern this audit is about, and it was
invisible from the option registry — there is no option to grep for, only a
missing dependency edge.

**Measured.** 10 of the 71 `.nl` fixtures in the CLI corpus invoke
restoration, and most of them *succeed* through it: `cresc4`, `deb7`,
`eigena2`, `eigmaxa`, `pooling_rt2stp`. Reduced to one line:
`min x s.t. x² = 2` from `x₀ = 1e-8` — where `∇g = 2x ≈ 0` makes the
linearised constraint useless — solves through the wired application and
returns `Restoration_Failed` through a bare one.

**Fixed** by removing the duplication that caused it.
`pounce_restoration::install::install_default_restoration` is the whole
wiring behind one call; all six call sites use it, including the two that
had none. `pounce-rs` gained `pounce_rs::application()`, a constructor
returning a wired application, and its own doc examples now use it. A
`_configured` variant carries the batch path's `parallel = false` override
through provider re-mints.

The residual is deliberate and documented: a bare `IpoptApplication::new()`
still has no restoration, because `pounce-algorithm` genuinely cannot install
it. `restoration_surface.rs` pins both halves, so the facade can never
silently hand back the unwired application again.

## Finding 2 — `check_x0::check_tnlp` is library-shaped and stuck in the CLI

`crates/pounce-cli/src/check_x0.rs:616` is generic over `&mut dyn TNLP`,
with no `.nl` and no CLI coupling, and its own doc says it is public "so the
debugger / tests can reuse it without going through a file." Nothing but
crate placement stops a `pounce-rs` user from preflighting a starting point
— non-finite evaluations at x0, bound violations, interior-clamp
displacement, scaling spread.

Same one-move extraction as #743 and #754. Best effort-to-value ratio here.

## Finding 3 — convex routing is unreachable from every in-process frontend

`application.rs:1241` refuses `lp-ipm` / `qp-ipm` / `socp` outside the CLI,
and `unhonored_convex_option` refuses the `qp_*` knobs on the same paths.

**Correction to an earlier reading of this gap.** It is tempting to describe
`crates/pounce-cli/src/qp_extract.rs` (2,169 lines, Rust) and
`python/pounce/_route.py` (617 lines, Python) as two implementations of one
decision, i.e. the #755 drift shape at scale. They are not, and treating
them that way would send someone off to "extract the Rust one and delete the
Python one", which cannot work:

- `qp_extract` calls `prob.obj_nonlinear.analyze_quadratic_full()`. It reads
  the **symbolic** `.nl` expression tree, so its routing is *certain*.
- `_route.py` says in its own header that it cannot read structure, because
  `minimize` takes opaque Python callables. It **probes** the callables,
  fits a linear/quadratic model, and validates that model at held-out points
  before trusting it.

Two different algorithms, because they have different information. A TNLP is
opaque in exactly the way a Python callable is: it exposes `eval_f`,
`eval_grad_f`, `eval_jac_g`, `eval_h` — numerical callbacks — and no
expression tree. You cannot symbolically certify a degree-<=2 objective from
numerical evaluations.

So closing this gap means **porting the probe-and-validate router to Rust**
over `dyn TNLP` — new numerical code, not a move. That is buildable and it
would serve `pounce-rs`, `pounce-cinterface` and `pounce-wasm` at once, but
it carries the asymmetric risk `_route.py` documents: a convex problem sent
to the NLP solver is merely slower, while a nonconvex problem sent to a
convex solver is **silently wrong**. The held-out validation gate is the
load-bearing part and must be ported with it.

This one wants its own PR and its own review, not a slot in a batch.

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
  serve library consumers testing their own integration. **Moved** to
  `pounce-nlp` (`pounce_nlp::builtin`). Its problems are known-good models
  with known answers, which is what someone wiring up their first `TNLP`
  wants to check their plumbing against before trusting it with a real model.
- `cbf.rs` (867 lines) — a self-contained Conic Benchmark Format parser with
  no CLI or `.nl` coupling at all. Niche, but it is pure library code.
  **Moved** to `pounce-convex` (`pounce_convex::cbf`) — not `pounce-nlp`,
  because it builds `QpProblem` / `ConeSpec` and belongs beside the conic
  solver that consumes them.

Both were pure relocations: the only edits were `pounce_nlp::tnlp` → `crate::tnlp`
and `pounce_convex::` → `crate::`. Their 6 and 10 inline tests travelled with
them (the CLI suite drops 606 → 590 by exactly that 16). Following the
precedent already set when the `.nl` pipeline moved to `pounce-nl`,
`pounce-cli` re-exports both under their historical names, so every existing
`crate::builtin::…` / `pounce_cli::cbf::…` path resolves unchanged — the move
cost zero call-site churn.

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
