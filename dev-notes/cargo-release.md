# crates.io release

POUNCE ships **20** Rust crates to crates.io. This file is the procedure.
For the PyPI side (`pounce-solver` + `pyomo-pounce`), see `pypi-release.md`.

The publish list and its dependency order live in
`scripts/publish-crates.sh` (the `CRATES=(...)` array). That list is the one
the release uses, and `scripts/check-release-consistency.sh` (run in CI on
every PR) fails the build if it ever drifts from the workspace's actual
publishable crates or stops being topologically ordered — so the tables below
are documentation, not a second source of truth to keep hand-synced.

## What publishes, what does not

The publishable set is exactly "every workspace member without
`publish = false`". As of this writing that is these 20:

| Crate                  | Publishes? | Role                                         |
| ---------------------- | ---------- | -------------------------------------------- |
| `pounce-common`        | yes        | foundation: types, exceptions, journal       |
| `pounce-linalg`        | yes        | dense/sparse linear-algebra primitives       |
| `pounce-linsol`        | yes        | symmetric linear-solver trait layer          |
| `pounce-feral`         | yes        | pure-Rust sparse LDLᵀ backend                |
| `pounce-hsl`           | yes        | optional HSL/MA57 backend (user supplies HSL)|
| `pounce-nlp`           | yes        | NLP-side glue: TNLP trait, IpoptApplication  |
| `pounce-l1penalty`     | yes        | ℓ₁-exact penalty-barrier TNLP wrapper        |
| `pounce-observability` | yes        | tracing/metrics install (pounce#71)          |
| `pounce-presolve`      | yes        | NLP preprocessing TNLP wrapper               |
| `pounce-qp`            | yes        | active-set QP subproblem solver              |
| `pounce-algorithm`     | yes        | IPM core (Ipopt `src/Algorithm` port)        |
| `pounce-restoration`   | yes        | restoration phase                            |
| `pounce-sensitivity`   | yes        | sIPOPT port / parametric warm-start          |
| `pounce-solve-report`  | yes        | `pounce.solve-report/v1` JSON writer         |
| `pounce-cinterface`    | yes        | C ABI (CreateIpoptProblem / IpoptSolve)      |
| `pounce-convex`        | yes        | LP/QP/SOCP/SDP conic IPM                      |
| `pounce-nl`            | yes        | `.nl` reader + AD tape; pounce-cli depends   |
| `pounce-studio-core`   | yes        | solve-report parsers; pounce-cli dep (0.4.0+)|
| `pounce-cli`           | yes        | `pounce` and `pounce_sens` binaries          |
| `pounce-rs`            | yes        | single-crate Rust facade (re-exports TNLP + driver) |
| `pounce-py`            | **no**     | ships on PyPI as `pounce-solver` via maturin |
| `pounce-studio-pyo3`   | **no**     | PyO3 wrapper; ships on PyPI                   |
| `iter-diff`            | **no**     | internal Track-A validation tool             |

Each `publish = false` crate has that flag in its `Cargo.toml`. The publish
script's list is derived from this same rule, and the consistency check
enforces that they agree.

## Dependency order

cargo refuses to publish a crate before the crates it depends on are live, so
the list is topologically sorted. The script publishes one crate at a time in
this order, not in parallel — each crate must be live (and visible in the
index, which is not instantaneous) before any dependent can publish. The
layered view (each layer depends only on earlier layers):

Layer 0: `pounce-common`, `pounce-studio-core` (leaves)
Layer 1: `pounce-linalg`
Layer 2: `pounce-linsol`, `pounce-nlp`
Layer 3: `pounce-convex`, `pounce-feral`, `pounce-hsl`, `pounce-l1penalty`,
         `pounce-nl`, `pounce-observability`, `pounce-presolve`, `pounce-qp`,
         `pounce-solve-report`
Layer 4: `pounce-algorithm`
Layer 5: `pounce-restoration`, `pounce-sensitivity`
Layer 6: `pounce-cinterface`, `pounce-cli`

`pounce-convex` (LP/QP/SOCP/SDP conic IPM) depends only on `pounce-common` +
`pounce-linsol` + `pounce-linalg`, so it sits in layer 3. `pounce-nl` depends
on `pounce-common` + `pounce-nlp`, also layer 3. `pounce-cli` is the sink:
it depends (transitively) on nearly everything, so it always publishes last.

The exact order the script uses can be re-derived at any time with:

```sh
cargo metadata --format-version 1 | python3 -c '...'   # see check-release-consistency.sh
```

## Rate limits — new crate names

crates.io rate-limits **new crate names** to 5 publishes burst, then 1 per
~10 minutes. New *versions* of *existing* crates are 1/min (burst 30), so a
routine release is unaffected.

As of 0.6.0, **all 19** crates are published (the last new name,
`pounce-convex`, first published in 0.5.0). Every release since is a version
bump of existing crates — 1/min, burst 30 — so the new-crate rate limit does
not apply to a routine release.

The new-crate burst limit only matters if a future release introduces **new**
crate names. If one does and you would exceed the 5-burst limit, either set
`SLEEP=600` on the publish script or email **help@crates.io** ahead of time
for a temporary exemption
(they typically respond within a business day); list the new crate names and
note they all share the `pounce-` prefix under account `jkitchin`.

## Cutting a release

### Pre-flight

1. Make sure `cargo login` is set up (one-time: `cargo login <token>` from
   https://crates.io/me) — only needed for a local publish; CI uses the
   `CARGO_REGISTRY_TOKEN` secret (see Automation below).
2. Bump the workspace version in `Cargo.toml` (root `[workspace.package]` and
   every entry in `[workspace.dependencies]` that points at one of our crates
   — they must all match the new version) **and** the two PyPI projects
   (`python/pyproject.toml`, `pyomo-pounce/pyproject.toml`) to the same
   X.Y.Z. `scripts/check-release-consistency.sh` verifies all three agree; run
   it before tagging. If the version bump is non-trivial, do it as its own
   commit.
3. Bump `CITATION.cff` to match: set `version:` to the new release version and
   `date-released:` to the release date. GitHub's "Cite this repository"
   widget reads these. (The `doi:` is the Zenodo *concept* DOI and stays put.)
4. Run `scripts/publish-crates.sh --dry-run` to catch missing metadata, broken
   links, or dirty-tree errors. This dry-runs every crate end-to-end, so any
   breakage appears here, not three crates into the real release.

### Real release

The crates.io publish is automated by `.github/workflows/release-crates.yml`
(triggered on a `v*` tag push, or manually via `workflow_dispatch` — which
defaults to a dry run). It runs `scripts/publish-crates.sh`, which is
idempotent (skips any crate already live at the target version), so a re-run
or a resumed run is safe.

To publish locally instead:

```sh
# Option A: no new-crate rate-limit concern (the common case):
scripts/publish-crates.sh

# Option B: several new crate names this release — space publishes out:
SLEEP=600 scripts/publish-crates.sh
```

If a publish fails part-way through (network blip, transient 5xx), fix the
issue and resume:

```sh
scripts/publish-crates.sh --start-from pounce-algorithm
```

### After publish

Tag the release in git so the release point is reproducible:

```sh
git tag vX.Y.Z && git push origin vX.Y.Z
```

(The Python distributions use their own tag prefixes — `python-v*` and
`pyomo-pounce-v*` — so the bare `v*` tag namespace is reserved for the Rust
crates and drives `release-crates.yml`.)

## Automation

| Surface                 | Workflow                              | Trigger                  |
| ----------------------- | ------------------------------------- | ------------------------ |
| crates.io (20 crates)   | `release-crates.yml`                  | `v*` tag / manual        |
| PyPI `pounce-solver`    | `release-pounce.yml`                  | `python-v*` tag          |
| PyPI `pyomo-pounce`     | `release-pyomo-pounce.yml`            | `pyomo-pounce-v*` tag    |

`release-crates.yml` needs the `CARGO_REGISTRY_TOKEN` secret and runs in the
`crates-io` environment. The GitHub Release itself is still created by hand
(`gh release create vX.Y.Z --notes-file <file>`); no workflow makes it.

## Yanking

If a release is broken, yank individual crates with
`cargo yank --version X.Y.Z -p pounce-common`. Yanking is reversible
(`cargo yank --undo …`) and does **not** delete the artifact — it just
prevents new builds from picking that version. There is no "yank the whole
workspace" command; iterate over the crate list manually if needed.

## HSL note

`pounce-hsl` is publishable but does **not** ship HSL source — users must
license MA57 separately from STFC and set `COINHSL_DIR`. The published crate
is a thin FFI wrapper. The README spells this out; flagging here so we do not
accidentally pull HSL source into a future release and create a licensing
problem.
