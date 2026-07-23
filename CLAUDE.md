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

## Working GitHub issues

When opening a PR that fixes a filed issue, the PR **body** (not just the
title) must contain an actual GitHub closing keyword tied to the issue
number — `Fixes #123`, `Closes #123`, or `Resolves #123`. Putting the issue
number only in the PR title (e.g. `Fix foo (#123)`) does **not** trigger
GitHub's auto-close on merge — the issue is left open, dangling, even
though the fix is merged. Confirmed missing on PR #342 (fixed #339, no
closing keyword, issue had to be closed by hand after merge); PR #344 (#341)
did it correctly.

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

## GAMS solver link — two routes

POUNCE registers with GAMS (`option nlp = pounce;`) two independent ways:

1. **pip (pure-Python, recommended for users)** — `pip install
   pounce-solver[gams]` + `pounce-gams register`. Lives in
   `python/pounce/gams/` (`gmo_translate.py`, `link.py`, `register.py`) with the
   `pounce-gams` CLI in `python/pounce/_gams_cli.py`. Built on GAMS's own
   `gamsapi[core]` PyPI bindings (which `dlopen` the user's GAMS libs) — **we
   redistribute nothing GAMS-owned**. POUNCE is a local NLP solver, so the link
   wires GMO's numerical evaluators straight into the cyipopt-style `Problem`
   callbacks (no opcode translator, unlike discopt's global solver). Registers a
   script solver via a `gamsconfig.yaml` `solverConfig` entry — no `sudo`, no
   system-dir writes, survives GAMS upgrades. The per-user config dir is
   OS-specific and **NOT XDG on macOS**: macOS `~/Library/Preferences/GAMS`,
   Linux `~/.config/GAMS`, Windows `%LOCALAPPDATA%\GAMS` (verify with `gamsinst
   -listdirs`). License-free unit tests in `python/tests/test_gams_link.py`
   drive a fake `GmoView`; the live `gamsapi` adapter is the only
   CI-untestable surface.
2. **native C link** — `gams/gams_pounce.c` + `make -C gams && sudo make -C gams
   install`. The authoritative reference for GMO call sequence, sign
   conventions, option keywords, and status mapping. Adds active-set-SQP
   working-set / state-file warm starts the pip link does not yet reproduce.

Docs: `docs/src/gams.md` (user-facing), `gams/README.md` (C link).
