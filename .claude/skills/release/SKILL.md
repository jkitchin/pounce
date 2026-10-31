---
name: release
description: How to cut a POUNCE release - the three registry surfaces (PyPI pounce-solver, PyPI pyomo-pounce, 22 crates.io crates), their tag prefixes and version files, the pre-tag consistency guard, the container-image surface, creating the GitHub Release by hand, and how to check what is actually published. Use when tagging, releasing, publishing, or auditing published versions.
---

# Releasing POUNCE

pounce ships to **three** registries on each release, all three automated by
GitHub Actions (each on its own tag prefix). The crates.io publish used to be
manual; it is now automated by `release-crates.yml` (tag-triggered).

## Surfaces (all must reach the same X.Y.Z)

A pre-tag guard, `scripts/check-release-consistency.sh` (run in CI on every
PR), fails the build unless all three versions below agree **and** the
crates.io publish list matches the workspace's publishable crates in
topological order. Run it before tagging.

1. **PyPI `pounce-solver`** — `.github/workflows/release-pounce.yml`, triggered
   by pushing a `python-vX.Y.Z` tag. Builds wheels (incl. Windows) + sdist,
   publishes to PyPI. Version: `python/pyproject.toml`.
2. **PyPI `pyomo-pounce`** — `.github/workflows/release-pyomo-pounce.yml`,
   triggered by a `pyomo-pounce-vX.Y.Z` tag. Version: `pyomo-pounce/pyproject.toml`.
3. **crates.io — 22 workspace crates** — automated by
   `.github/workflows/release-crates.yml`, triggered by a `vX.Y.Z` tag push
   (or run manually via `workflow_dispatch`, which defaults to a dry run). It
   runs `scripts/publish-crates.sh`, which publishes in topological order and
   is idempotent (skips any crate already live at the target version), so a
   re-run or resumed run is safe; resume a mid-batch failure with
   `--start-from <crate>`. New-crate rate limits apply on first publish only.
   Crates with `publish = false` (pounce-py, pounce-studio-pyo3, iter-diff)
   are intentionally excluded. Full procedure: `dev-notes/cargo-release.md`.
   Version: root `Cargo.toml` `[workspace.package]` (all crates inherit it via
   `version.workspace = true`).

   The CLI binary is also bundled inside the PyPI wheels, so an end user
   `pip install pounce-solver` does not require the crates.io publish — but the
   crates.io publish is still part of a complete release.

## Container images build for stable releases only

A fourth surface, `ghcr.io/jkitchin/pounce`, is **not** covered by the
version guard above and ships **only on a `v*` tag** (gh#599).
`.github/workflows/release-docker.yml` has no `main` trigger: the
source-built image no longer goes out as `:edge` / `:sha-<short>` on every
merge, so those tags sit frozen at whatever commit last published them and
must not be read as "tip of `main`". `make docker` is how you get an image
of an unreleased fix; a manual workflow run (`variant=source`,
`dry_run=false`) publishes one if a bug report needs it.

Every publishing workflow — the three registry releases, this one, and the
docs deploy — gates its first job on `github.repository ==
'jkitchin/pounce'`. Forks receive tags when they sync, and none of these can
succeed there, so without the gate a contributor's fork sync is a failed run
and an email. Keep the gate on the job every other job reaches through
`needs`; `pull_request` runs in the base repo's context, so PR checks are
unaffected. `coverage.yml` carries the same gate for the same reason — it is
not a publishing workflow, but it needs `secrets.CODECOV_TOKEN`, which no fork
has, and it fails loudly on a bad upload.

## GitHub Release

Created **by hand** (`gh release create vX.Y.Z --notes-file <file>`); no workflow
makes it. Body has historically been the matching `## [X.Y.Z]` section of
CHANGELOG.md. A git tag alone does NOT create a Release, and creating a Release
does NOT trigger any workflow (nothing has an `on: release` trigger).

## Checking what's published (don't get this wrong)

crates.io API needs a User-Agent or it silently looks unpublished:

    curl -s -H "User-Agent: pounce-release-check (jkitchin@andrew.cmu.edu)" \
      https://crates.io/api/v1/crates/<name> | python3 -c \
      "import sys,json; c=json.load(sys.stdin).get('crate'); print(c['max_version'] if c else 'NOT PUBLISHED')"

Sanity-check against `serde` first; if serde reads NOT PUBLISHED your request is
being rejected, not the crate missing.
