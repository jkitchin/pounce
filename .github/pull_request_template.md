<!--
Delete any section that does not apply. A one-line typo fix does not need a
verification table; a solver behaviour change does.

The prose sections matter more than the checklist. POUNCE PRs are read later as
the record of *why* a numerical decision was made — the CHANGELOG says what
changed, this says what was tried and rejected.
-->

## Problem

<!--
What breaks, in the user's terms, with numbers. If there is an oracle (Ipopt,
a published optimum, a closed-form answer), compare against it here.

| | status | objective | NLP error | iters |
|---|---|---|---|---|
| before | | | | |
| after | | | | |
| oracle | | | | |
-->

## Cause

<!-- The actual mechanism, not the symptom. -->

## Fix

<!--
What changed and why this approach. If you tried something else first and
rejected it, say so and say what ruled it out — a future maintainer retuning
this will otherwise repeat the experiment. Prefer "measured over N models,
every threshold that fires also introduces X" over "this seemed better".
-->

## Blast radius

<!--
Who else is affected. For solver changes: what does the corpus sweep say, and
against which baseline commit? If the answer is "bit-identical except the
targeted case", say that — it is the most useful sentence in the PR.
-->

## Tests

<!--
What pins this, and how you know the tests bite: state that they fail on the
parent commit, and for the right reason. A regression test that passes pre-fix
is not a regression test.

If a behaviour is deliberately NOT pinned (platform-dependent, needs a fixture
too large to vendor), say so and why, so the gap is a known one.
-->

---

- [ ] Tests fail on the parent commit for the stated reason
- [ ] `CHANGELOG.md` `[Unreleased]` entry, in the user's terms
- [ ] Book page under `docs/src/` updated, and linked from `SUMMARY.md` if new
- [ ] `cargo fmt --all -- --check`, `cargo clippy`, `cargo test` clean
- [ ] Every claim in this PR body and in the code comments is true of the code
      as it stands now — no test named that does not exist, no doc describing a
      design that was revised mid-review, no measurement whose baseline has
      since moved

<!--
That last box is not boilerplate. The recurring failure here is a PR body or
doc comment that was accurate three commits ago: a claim that a test pins some
behaviour when that test was later dropped, a comment describing the first
attempt rather than what shipped, or a corpus sweep measured before a sibling
PR landed. Re-read the body against the final diff before requesting review.

See CONTRIBUTING.md for the full definition of done and the cross-solver doc
ownership rules (solver routing / problem-class coverage changes need
@jkitchin as reviewer).
-->
