# pounce — notes for Claude

Topic-specific notes live where they load only when relevant:

- **Releasing / publishing** (PyPI, crates.io, ghcr.io, GitHub Release,
  checking what is published, the `github.repository` gate on publishing
  workflows): the `release` skill, `.claude/skills/release/SKILL.md`.
- **GAMS solver link** (pip link vs native C link, `gamsconfig.yaml`
  registration): the `gams-link` skill, `.claude/skills/gams-link/SKILL.md`.
- **Sensitivity pre-merge checks** (invariance legs, re-solve oracle, index
  spaces, `/sens-review`): `crates/pounce-sensitivity/CLAUDE.md`, which also
  governs `crates/pounce-sens-core`. It loads when you work in those crates;
  read it before reviewing or merging a sensitivity change from elsewhere.

## Coverage is measured combined, never Rust-only

`.github/workflows/coverage.yml` runs `scripts/coverage-combined.sh` (i.e.
`make coverage`) and uploads to Codecov. It runs on **pushes to `main` only** —
plus `workflow_dispatch` for an on-demand run against a branch. It was on
`pull_request` too until the wait proved indefensible: 33–40 minutes against
ci.yml's ~10, holding every PR visibly un-green long after the checks that can
reject it had passed, for a number nothing blocks on. The cost of that choice
is the per-PR patch-coverage status, which was `informational` anyway; the
other cost is that a PR editing `coverage.yml` no longer exercises it, so
dispatch such a branch manually before merging. It does **not** run `cargo
llvm-cov --workspace`: large parts of POUNCE are reachable only through the
`pounce._pounce` extension module or the CLI, and a Rust-only report shows
those at 0% — inventing gaps in code that is well covered. The job asserts the
attribution worked before uploading, by checking that `crates/pounce-py/*`
does not read ~0% (it can only be reached through the `.so`). Requires the
`CODECOV_TOKEN` repository secret. Full rationale: CONTRIBUTING.md,
"Measuring coverage".

## The docs site has a fourth input: the wiki

`.github/workflows/docs.yml` clones **jkitchin/pounce.wiki** — a separate
repository — and `scripts/build-versioned-docs.sh` indexes it alongside
`docs/src` into `ask-index.json`, the corpus the in-browser docs assistant
(`docs/src/ask.md`) searches. So the deployed site now depends on a repo that
is not this one. The clone step is `continue-on-error` and the build treats a
missing `POUNCE_WIKI_DIR` as book-only, deliberately: an unreachable wiki must
degrade the assistant, never take the docs site down. A run where that step is
amber is a real signal — the wiki pages stopped being answerable — not noise.

Two things about the assistant that are easy to get wrong:

* **It is injected per page, not listed in `book.toml`.** `additional-js`
  would reach `dev/` and nothing else until the next release, because the
  site root is built from the newest *tag*, whose source predates any change
  you just made. It rides the version selector's injection instead, so
  editing `docs/assets/ask.js` changes every version's pages on the next
  deploy. The corollary is that a plain `mdbook build docs` (`make book`) has
  no **Ask** button at all — use `make book-site`, and serve it under the real
  base path, or relative-path bugs stay invisible above the top level.
* **One index, built from `main`, shared by every version.** Citations
  therefore resolve against the *site root*, not the current version
  (`data-root`, distinct from `data-p2r`). Resolving them against an archived
  book 404s on every page and anchor added since that tag — measured, four of
  six hits on one v0.2.0 page.

Retrieval fails silently by construction: a broken ranker returns the wrong
passage and the only symptom is a reader not finding the page. `make
ask-check` (`docs/tests/ask_retrieval.mjs`, also in `ci.yml`) runs the
**shipped** `ask.js` through a test seam against a labelled query set — a test
of a reimplemented scorer would stay green while the real one regressed. Its
thresholds sit deliberately below the measured score because it is a corpus
test and honest doc edits move rankings; it exists to catch a scoring change
that drops several queries at once. Change anything in the tokenizer, the
stemmer, or the scorer and re-measure rather than reasoning about it — every
one of the four ranking defects fixed while building it looked fine on
inspection.

## Trajectory changes need the fixture sweep

A change that reroutes **which** correction the solver reaches for, or that
reorders/rescales the steps it takes, is a *trajectory* change. Run
`scripts/sweep-fixtures.sh` against a baseline binary and diff, **before
merge**, and be able to explain every line that moves.

The sweep runs **two legs** per fixture — `exact` (the default path) and
`lbfgs` (`hessian_approximation=limited-memory`) — each line prefixed with the
leg name. Both run by default; do not diff only one. The L-BFGS leg exists
because the corpus was exact-Hessian only, and that gap shipped gh#677: the
initial Hessian scalar used `scalar2` where Ipopt uses `scalar1`, because
`limited_memory_initialization` was registered and never read. L-BFGS is not
a rare opt-in — the Python frontend and the CasADi plugin both select it
automatically when no exact Lagrangian Hessian is available.

**The convex arm is covered — do not skip the sweep for a convex-path
change.** Both legs run at the default `solver_selection=auto`, which routes
to the most specialized engine available, so 42 of the 79 fixtures never touch
the NLP arm at all: 37 reach the convex QP interior-point and 5 the convex
QCQP conic one (counted off the engine column on `fdea82b5`; recount rather
than trust it, the split moves whenever a fixture is added). gh#760 is the
case for saying so explicitly — `4c02817d` skipped the sweep on the reasoning
that "this is a trajectory change on the convex path, not the NLP path, so
`scripts/sweep-fixtures.sh` does not cover it", and substituted an
objective-parity check over the convex corpus. Objective parity is blind to
trajectory by construction. Run across that commit the sweep moves 52
fixture-legs and flips `scaled_feasible_a` on the lbfgs leg from
`MaximumIterationsExceeded`/199 to `SolveSucceeded`/69.

Each line also records **which engine solved the model** (`cvx-qp`,
`cvx-qcqp`, `qp-active-set`, `sqp-active-set`, `square-lm`, `nlp`). Status, objective and
iteration count can all be unchanged while a model silently changes arms, so a
routing regression used to leave no trace in the diff. A line whose only moving
field is the engine is a routing change, and is as reportable as a moved
iteration count.

`sqp-active-set` is the active-set SQP outer loop (`algorithm=active-set-sqp`);
`qp-active-set` is `pounce_convex::active_set`, reached by
`solver_selection=qp-active-set` on an LP or convex QP. Different engines, not
synonyms — both solve their subproblems with `pounce-qp`, which is why the
names are close. The SQP arm reported `nlp` until it was fixed, so a sweep
taken against a binary older than that shows every SQP line moving in this
column and nothing else; that is the reporting fix, not a reroute.

`square-lm` is not routed to: it is the last stage of the second-opinion
ladder (`pounce_algorithm::square_system`), which runs only after the NLP arm
and every option rung have failed on a square, constant-objective model, and
answers from Levenberg–Marquardt on `c(x) = 0`. A line moving `nlp` →
`square-lm` means the IPM stopped solving that model, which is a regression
to explain even though the status reads `SolveSucceeded`.

The engine is read from the JSON report's `solution.engine`. When a fixture
produces no JSON at all — `deb7` on the L-BFGS leg under
`algorithm=active-set-sqp` times out — the sweep falls back to scraping the
`Selected solver:` banner, so a change to that banner's wording moves the
column on exactly those lines, with `NO_JSON` unchanged either side.

The `mu=` column (gh#1042 C1) records the `mu_strategy_fallback` re-solve,
read from the JSON's `statistics.mu_strategy_fallback`: `-`, or
`<strategy>@<first status>/<first iters>,tot=N` when the re-solve was
promoted — then `it=` is the re-solve's alone — or
`kept(<strategy>:<retry status>/<retry iters>),tot=N` when it lost. A first
attempt that gets worse and is rescued by the fallback moves only this column
and reads as a speed-up in `it=`: gh#1042's R2b reported `pooling_rt2stp`
492 → 109 while its first attempt went Optimal/492 → Acceptable/494. To diff
against a sweep from before the column, drop it with
`sed -E 's/ mu=[^ ]+ +/ /'`.

What the corpus still cannot give you is **magnitude at benchmark scale**. The
largest fixture of any class is `deb7` at 813 variables and the largest convex
one is `lp_degen2` at 534, while `benchmarks/qp` reaches 93 263 (`BOYD2`). A
moved convex line is a signal to go measure `benchmarks/qp`, not a bound on
what you will find there.

The *degenerate* half of that gap is closed, and how it was closed is the
lesson. Nothing in the corpus predicted the 4.4× iteration cost `4c02817d`
carried on the Maros-Meszaros QSCFXM family (38 → 168) — not because the
fixtures were small (gh#690 had already added 142–534-column convex ones) but
because **not one of them was a model on which relaxing the box costs
anything**. Swept twice, default versus `bound_relax_factor=0`, the convex
lines move by tens of percent in both directions and `lp_degen2` moves 18 → 15.
`convex_qp_qscfxm1` is QSCFXM1 itself, moves 131 → 30 on both legs, and is
pinned by `crates/pounce-cli/tests/issue_760_convex_bound_relax_magnitude.rs`.
The general rule it instances is the branch rule (next section): a corpus that is
uniform in the dimension a change acts on reports "small and mixed" no matter
how large its models are. See `dev-notes/qp-bound-relax-iteration-cost.md` and
`dev-notes/convex-fixture-corpus.md`.

Nor does it cover the convex arm's **cost-normalization** (`σ`) path, and
neither does `benchmarks/qp` — measured, not assumed, while fixing gh#414
reopened: exactly **1 of 79** fixtures reaches it (`qcqp_columns_illcond`, on
both legs) and **0 of 138** Maros-Meszaros problems do. `σ` engages only when
`max(‖P‖∞, ‖c‖∞)·ε > tol`, which almost nothing in either corpus satisfies. So
an empty sweep *and* an empty `benchmarks/qp` diff are together compatible with
having changed that path's verdict on every input it sees — which is what
gh#414's second fix did. A change to `crates/pounce-convex/src/ipm.rs`'s
`normalized_optimum_is_genuine` / `hsde_cost_scale` needs
`crates/pounce-convex/tests/issue414_cost_normalized_false_optimal.rs` and
`illconditioned_huge_scale.rs`, which reach the accept and reject branches on
purpose; the corpora are evidence of no *collateral* damage and nothing more.

The same hole exists on the **acceptable-level dual gate**, and it is the one
case where the blind spot is a *verdict* rather than a trajectory (gh#884). No
fixture in the corpus has an acceptable-level exit whose unscaled dual residual
exceeds `dual_inf_tol`, so a change to that gate is measured against a corpus
that is uniform in exactly the dimension it acts on. Measured: a proposed
ceiling referenced to `norm_inf(grad f)` diffs **empty across all 182
fixture-legs, both legs**, while tightening `acceptable_dual_inf_tol` from
`1e10` to `dual_inf_tol` for the entire unconstrained and bound-only class —
turning a `Solved_To_Acceptable_Level` at `3.7e-6` into a `RestorationFailed`
15 orders from `f*` on an unconstrained ill-conditioned quadratic. The model
that caught it is not in the corpus at all. So a gate-touching change needs a
purpose-built fixture that reaches an acceptable-level exit above
`dual_inf_tol`; an empty sweep is not evidence about it. Full account:
`dev-notes/mpcc-biactive-dual-divergence.md`.

**The `--minima` search has its own sweep.** `scripts/sweep-fixtures.sh` never
passes `--minima`, so a change to `crates/pounce-minima` or
`crates/pounce-cli/src/minima` diffs it empty by construction. Run
`scripts/sweep-minima.sh` against a baseline binary instead: six methods over a
fixture set chosen per branch (constrained models for dual recovery, saddle
models whose every line reads `found=0`, a model whose every candidate reaches
the `kkt_tol` veto), recording status, solves, every minimum and the best
point's duals. A pure move must diff empty; gh#966's extraction did, 54
lines, both when the files moved and when the driver went behind `LocalSolve`.
Python's `find_minima` runs the same crate, and
`python/tests/test_minima_cli_parity.py` holds the two frontends to the same
status, solve count and values for one seed across all six methods — a
frontend that builds the problem, the options or the seed differently fails
there, where no test of the crate alone can see it.

"It cannot produce a wrong answer" is **not** the relevant safety property
here, and that exact argument is what shipped gh#544 in 0.10.0: a trajectory
regression produces the *right* answer, slowly — or a differently-wrong
tolerance-legal one. gh#544 took `pooling_rt2stp` from 206 to 812 iterations;
the suite asserts status and objective, not iteration count, so nothing saw it.
It surfaced as a CI wall-clock timeout, was misattributed, the cap was raised
for good reasons, and the underlying defect shipped and came back as gh#592.

Related: a measured regression that gets recorded as "an accepted cost of the
fix" needs an issue and an owner — or, when nobody plans to work on it, an
entry in `dev-notes/known-limitations.md` with the measurements, what was
tried, and how to tell whether a fix works. The tracker holds planned work
only. Without either it is indistinguishable from noise to the next reader —
which is how 206 → 812 sat in a commit message through a release.

Full post-mortem: `dev-notes/trajectory-regressions-and-the-fixture-sweep.md`.

## A test is only evidence about the branch it reaches

If the rule under test *branches* — on an activity class, on a status, on
which side of a threshold a quantity falls — a fixture that always takes one
branch says nothing about the other, and it stays green while the other
branch is broken. Before trusting a green test or an empty sweep on a change,
check which branch its fixture takes, and add a fixture for each branch the
rule can take: the second fixture is the test, not a duplicate of the first.
The same holds for a corpus: one that is uniform in the dimension a change
acts on reports "small and mixed" no matter how large its models are.

The worked example (gh#756, whose defect stayed live under a green leg
because that leg's fixture certified the class the PR's rule returned early
on) and its classifier corollary are in `crates/pounce-sensitivity/CLAUDE.md`.

## Working GitHub issues

When opening a PR that fixes a filed issue, the PR **body** (not just the
title) must contain an actual GitHub closing keyword tied to the issue
number — `Fixes #123`, `Closes #123`, or `Resolves #123`. Putting the issue
number only in the PR title (e.g. `Fix foo (#123)`) does **not** trigger
GitHub's auto-close on merge — the issue is left open, dangling, even
though the fix is merged. Confirmed missing on PR #342 (fixed #339, no
closing keyword, issue had to be closed by hand after merge); PR #344 (#341)
did it correctly.
