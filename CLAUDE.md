# pounce — release / publishing facts

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
3. **crates.io — 19 workspace crates** — automated by
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
