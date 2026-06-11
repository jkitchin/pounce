# Contributing to POUNCE

Thanks for working on POUNCE. This file captures the few conventions that keep
a multi-solver, multi-registry project from drifting. The release mechanics
live in `dev-notes/cargo-release.md` and `dev-notes/pypi-release.md`; this file
is about getting a change *merge-ready*.

## Enable the git hooks (one-time)

```sh
git config core.hooksPath .githooks
```

The `pre-commit` hook runs `cargo fmt --all -- --check`, mirroring CI so
formatting drift never reaches `main`.

## Definition of done for a user-facing change

POUNCE is a family of solvers (NLP filter-IPM, active-set SQP, convex/conic
IPM, SOS/Lasserre), and the recurring failure mode is *"the feature exists but
isn't documented where a user looks."* To avoid it, a change that adds or
changes user-visible behavior is not done until **all three** of these land in
the same PR:

1. **Code + test.** The behavior, with a test that pins it (Rust `cargo test`
   and/or Python `pytest`).

2. **CHANGELOG entry.** Add a bullet under the `## [Unreleased]` section of
   `CHANGELOG.md` (create that section if it is absent — it sits directly above
   the most recent released version). One entry per feature, in the user's
   terms, naming the surface(s) it affects (CLI / Python / Pyomo). At release
   time the section is renamed to the version and dated.

3. **Book section.** Update the rendered book under `docs/src/`. A brand-new
   page **must** be linked from `docs/src/SUMMARY.md` — an unlinked page is
   invisible in the book, and `scripts/check-docs-consistency.sh` (run in CI)
   will fail the build until it is wired in.

   In particular, anything that changes **which solver handles which problem
   class** — a new solver, a new routing rule, a new `solver_selection` value,
   a class moving from local to global — must update the cross-solver landscape
   docs as a unit (see ownership below), not just the page for the one solver.

## Cross-solver documentation ownership

The cross-solver "landscape" docs are easy to update piecemeal and leave
inconsistent. These three pages must always agree and are owned, as a unit, by
the maintainer (**@jkitchin**) — flag a reviewer on any PR that touches solver
routing or problem-class coverage:

- `docs/src/choosing-a-solver.md` — the solver-landscape map and the
  "at a glance" table.
- `docs/src/lp-qp-routing.md` — how `auto` classifies a problem and the
  `solver_selection` values.
- `docs/src/python.md` — the Python entry points and auto-routing behavior.

When a solver lands or changes its problem-class coverage, update all three in
the same change so a reader gets one coherent story regardless of which page
they land on.

## CI guards worth knowing about

These run on every PR; run them locally before pushing to get fast feedback:

- `scripts/check-release-consistency.sh` — the three registry versions agree
  and the crates.io publish list matches the workspace in topological order.
- `scripts/check-docs-consistency.sh` — every `docs/src` page is reachable
  from `SUMMARY.md` and every TOC link resolves.
- `cargo fmt --all -- --check`, `cargo clippy`, `cargo test`, and the Python
  test suite (see `.github/workflows/ci.yml`).
