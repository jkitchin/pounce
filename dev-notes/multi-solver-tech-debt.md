# Multi-solver maintenance: technical-debt audit

_Written when reconciling PR #70, which took POUNCE from one solver to three._

## Why this note exists

Until the 0.4.0 line, POUNCE was effectively **one solver**: the Ipopt-derived
filter-line-search interior-point method for general NLPs (`pounce-algorithm`).
PR #70 adds two more solver families:

- **`pounce-convex`** — a convex/conic interior-point solver (LP, convex QP,
  SOCP, PSD, exp/power cones) over a homogeneous self-dual embedding (HSDE),
  with SOS polynomial optimization layered on the PSD cone.
- **`pounce-global`** — a spatial branch-and-bound global optimizer for
  factorable nonconvex NLPs.

Going from one solver to three is a capability win, but it permanently changes
the maintenance surface: several things that used to have exactly one
implementation now have N, and a few abstractions were introduced to span them.
This note records the debt so it stays visible and is paid down deliberately
rather than discovered painfully.

## What is NOT debt (so we don't "fix" the wrong thing)

- **The two interior-point implementations are not duplicated linear algebra.**
  Both `pounce-algorithm` (NLP filter-IPM) and `pounce-convex` (conic HSDE-IPM)
  depend on `pounce-linsol` + `pounce-linalg` and share that sparse-symmetric
  factorization/KKT substrate. Only the *outer loops* differ — filter line
  search vs. HSDE — which is correct: they are genuinely different algorithms,
  not two copies of one. Merging them would be the mistake.
- **Separate typed entry points per solver are partly intrinsic.** A cone
  program is *data* (matrices + cone list); a certified global optimum needs a
  *symbolic* objective to relax. Neither fits `minimize(fun, x0, …)`. Some API
  divergence is the nature of the problem, not sloppiness. The debt is the
  *absence of a router on top* (see area 2), not the existence of typed entries.

## The four debt areas

### 1. Debugger trait fan-out

**State.** The interactive debugger was generalized over a `DebugState` trait
(`crates/pounce-common/src/debug.rs`) so one REPL (`debug_repl.rs`) drives all
iteration-loop solvers via `&mut dyn DebugState`. A *second*, parallel hierarchy
— `TreeDebugState` / `TreeDebugHook` (`crates/pounce-cli/src/tree_debug.rs`) —
exists for the branch-and-bound tree, bridged to the IPM REPL by a shared
command queue for `into` step-into. NLP-only commands (rank, sweep, resolve)
reach the concrete `DebugCtx` through `as_nlp()` / `as_nlp_mut()` downcasts.

**Debt.**
- Every new debugger command must decide its behavior on **all three** backends,
  or silently degrade on the ones it doesn't handle. Downcast-and-branch
  (`as_nlp`) is the smell: it compiles even when a command is a no-op on conic /
  tree states, so coverage gaps are invisible.
- Two trait hierarchies (`DebugState` + `TreeDebugState`) plus a bridge is real
  surface area; a fourth solver would likely add a third.
- The `--debug-json` **metric vocabulary** is a cross-cutting contract
  (`iter, mu, objective, inf_pr, inf_du, nlp_error, complementarity`) consumed by
  the MCP proxy and its tests. It already needed a consistency pass once
  (`727d088`). Each backend maps its native quantities onto this NLP-centric set
  (e.g. convex reports `nlp_error = max(pinf, dinf, μ)`; the name no longer means
  "NLP error"). More backends → more semantic stretching of fixed field names.

**Recommendation.**
- Maintain a **capability matrix** (command × backend) in `docs/src/debugger.md`,
  and make "unsupported on this backend" an explicit, uniform REPL/JSON response
  rather than a silent no-op.
- Keep a **single source of truth** for the JSON metric set and assert in a test
  that every `DebugState` impl populates (or explicitly NaNs) each field, so a
  new backend can't quietly drift the protocol.
- Re-evaluate whether `TreeDebugState` can fold into `DebugState` (or a shared
  supertrait) once a second tree-like solver is on the horizon.

**Status (issue #105).** `pounce-global` and its `TreeDebugState`/`TreeDebugHook`
tree debugger were stripped from this release (commits `36f50bf`, `9a4c908`), so
there are now exactly two backends behind one `DebugState` trait + REPL — the NLP
filter-IPM and the convex/conic IPM — and the second-trait-hierarchy debt is moot
until a tree-like solver returns. The remaining recommendations are now done:
the [capability matrix](../docs/src/debugger.md) lists every backend-conditional
command, unsupported commands return an explicit error (`nlp_only`) rather than a
silent no-op, and the streamed metric set is built from one `METRICS` source via
`metric_fields` with a test (`metric_fields_match_advertised_vocabulary`) pinning
the emitted fields to the advertised `hello.metrics` vocabulary.

### 2. Python routing facade (designed, not built)

**State.** `dev-notes/lp-qp-routing.md` (this PR's headline design doc) specifies
a `ProblemClass`-driven router, and `crates/pounce-cli/src/dispatch.rs` already
classifies and routes on the **CLI** (`solver_selection=auto`). But the **Python**
surface exposes parallel, hand-picked entry points: `minimize` (NLP),
`solve_qp`, `solve_socp`, `sos_minimize`, `minimize_global` — with no unifying
dispatch.

**Debt.**
- Users must know solver theory to pick the right entry point; the CLI can
  auto-route from a parsed `.nl`, but Python callers get no equivalent.
- Two divergent dispatch stories (CLI classifier vs. Python explicit) will drift
  in behavior and documentation.
- `minimize` deliberately *cannot* route (it only sees an opaque callable) — so a
  Python router can't just live behind `minimize`; it needs structured input.
  That design question is unresolved and compounds with each new solver.

**Recommendation.** Decide explicitly: either (a) build a Python router that
takes structured problems and dispatches by `ProblemClass` (mirroring
`dispatch.rs`), or (b) commit to explicit entry points and document the choice
prominently (a "Choosing a Solver" page already exists — make it the front door).
Track the routing facade as the designed-but-unbuilt piece it is, so it isn't
mistaken for shipped.

### 3. Release / publish surface

**State.** The workspace grew **16 → 18 published crates** across **three**
registries (PyPI `pounce-solver`, PyPI `pyomo-pounce`, crates.io). Per
`CLAUDE.md`, the crates.io publish has historically been manual and "easy to
forget." This PR adds `pounce-convex` and `pounce-global` to the topological
publish order (`publish-crates.sh`, `dev-notes/cargo-release.md`); both are **new
crate names**, so they hit the crates.io new-crate rate limit on first publish.
(Note: main recently added `.github/workflows/release-crates.yml`, which begins
automating the crates.io publish on `v*` tags — partially mitigating the manual
step.)

**Debt.**
- More crates = more topological-order maintenance and more first-publish
  rate-limit exposure on each new-crate release.
- Three registries must reach the same `X.Y.Z`; a long-lived feature branch
  (like this one) silently accrues version skew against a fast-moving release
  line — exactly the conflict this reconciliation had to clean up.

**Recommendation.** Finish automating the crates.io publish via the new
`release-crates.yml` so the manual step disappears; keep the publish list and the
layered dependency order in `cargo-release.md` as the single source the script
derives from; consider a CI check that the three registries' target versions
agree before tagging.

### 4. Docs / CHANGELOG drift

**State.** The PR's major features (convex/conic, SOS, global) were **absent from
its own CHANGELOG** until this reconciliation backfilled them. The book
(`docs/src/SUMMARY.md`) and the solver-landscape material must now present three
solvers coherently rather than one.

**Debt.** With multiple solvers shipping independently, "the feature exists but
isn't documented anywhere a user looks" becomes the default failure mode, and the
gap compounds across releases.

**Recommendation.** Adopt a lightweight "**one feature → CHANGELOG entry + book
section**" definition-of-done, and name an owner for the cross-solver
landscape/choosing-a-solver docs so they're updated as a unit when a solver
lands or changes class coverage.

## Suggested follow-ups

Each area should become a tracked issue linking back to this note. None blocks
the PR #70 merge — they are the deliberate paydown plan for the maintenance cost
of becoming a multi-solver project.
