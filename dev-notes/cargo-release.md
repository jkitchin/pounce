# crates.io release

POUNCE ships **22** Rust crates to crates.io. This file is the procedure.
For the PyPI side (`pounce-solver` + `pyomo-pounce`), see `pypi-release.md`.

The publish list and its dependency order live in
`scripts/publish-crates.sh` (the `CRATES=(...)` array). That list is the one
the release uses, and `scripts/check-release-consistency.sh` (run in CI on
every PR) fails the build if it ever drifts from the workspace's actual
publishable crates or stops being topologically ordered — so the tables below
are documentation, not a second source of truth to keep hand-synced.

## What publishes, what does not

The publishable set is exactly "every workspace member without
`publish = false`". As of this writing that is these 22:

| Crate                  | Publishes? | Role                                         |
| ---------------------- | ---------- | -------------------------------------------- |
| `pounce-common`        | yes        | foundation: types, exceptions, journal       |
| `pounce-linalg`        | yes        | dense/sparse linear-algebra primitives       |
| `pounce-sens-core`     | yes        | engine-agnostic sensitivity core (`SensBacksolver`) |
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
| `pounce-minima`        | yes        | multistart / find-minima search (gh#966)     |
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
Layer 2: `pounce-linsol`, `pounce-nlp`, `pounce-sens-core`
Layer 3: `pounce-feral`, `pounce-hsl`, `pounce-l1penalty`,
         `pounce-nl`, `pounce-observability`, `pounce-presolve`, `pounce-qp`,
         `pounce-solve-report`
Layer 4: `pounce-convex`
Layer 5: `pounce-algorithm`
Layer 6: `pounce-minima`, `pounce-restoration`, `pounce-sensitivity`
Layer 7: `pounce-cinterface`, `pounce-cli`, `pounce-rs`

`pounce-convex` (LP/QP/SOCP/SDP conic IPM) depends on `pounce-qp`,
`pounce-presolve` and `pounce-sens-core` besides the layer-0–2 crates, so it
sits in layer 4, and `pounce-algorithm` above it: the SQP seeds a large cold
step QP with the convex interior point (gh#1061). `pounce-nl` depends
on `pounce-common` + `pounce-nlp`, also layer 3. `pounce-cli` and the
`pounce-rs` facade are the sinks: they depend (transitively) on nearly
everything, so they publish last.

The exact order the script uses can be re-derived at any time with:

```sh
cargo metadata --format-version 1 | python3 -c '...'   # see check-release-consistency.sh
```

## Rate limits — new crate names

crates.io rate-limits **new crate names** to 5 publishes burst, then 1 per
~10 minutes. New *versions* of *existing* crates are 1/min (burst 30), so a
routine release is unaffected.

As of 0.12.0, 21 of the 22 are published. The exception is
`pounce-minima` (gh#966), a new name that first ships in the release after
0.12.0 — one new crate, well inside the 5-burst limit, so it needs no
exemption. Everything else is a version bump of an existing crate — 1/min,
burst 30. Confirm with the `curl` check in the root `CLAUDE.md` rather than
trusting this paragraph.

The new-crate burst limit only matters if a future release introduces **new**
crate names. If one does and you would exceed the 5-burst limit, either set
`SLEEP=600` on the publish script or email **help@crates.io** ahead of time
for a temporary exemption
(they typically respond within a business day); list the new crate names and
note they all share the `pounce-` prefix under account `jkitchin`.

## Cutting a release

### Benchmarks — on a clean `main`, current with `origin`

Do this first, once the release's content is frozen (nothing solver-facing
left to merge). A release report must describe the commit being tagged, and
the last three releases each had to learn that separately: the August
baseline was regenerated off `perf/feral-refine-target @ …-dirty` (a branch
that never merged, flattering CUTEst by 19 points of "POUNCE faster"), 0.11.0
was first swept 27 commits behind its tag, and 0.12.0 ran from a
`bench/v0.12.0-sweep` branch.

1. **Run from a worktree at `origin/main`, not the working checkout.** A
   sweep takes hours; anything that edits the main checkout in that time
   (another session, a branch switch) dirties the tree and the provenance
   line records `-dirty`.

   ```sh
   git fetch origin
   git worktree add --detach ../pounce-bench-main origin/main
   cd ../pounce-bench-main
   echo "$POUNCE_BENCH_DATA"          # the .nl data root; a fresh worktree has no .nl files
   make -C benchmarks benchmark-rerun  # builds target/release/pounce, then every suite
   ```

   `benchmark-rerun`, not `benchmark`: the incremental target skips any
   suite whose `pounce.json` is newer than the binary and its inputs, which
   in a reused tree is a stale result silently carried forward.
2. **Check what ran.** The report's "POUNCE binary" row (from
   `target/release/pounce --about`) must name the `origin/main` commit,
   and its POUNCE row must read a tree with no `-dirty`. The two rows can
   name different commits legitimately: a report regenerated after a
   harness or reference fix is generated from the fix's commit, while the
   numbers still come from the swept binary. Every suite's `<suite>/pounce.env.json` records
   the time limit and thread settings it ran under; keep them the same as
   the previous release's (0.12.0: 300 s, threads unpinned) or say in the
   commit which changed — otherwise timing deltas are not comparable.
3. **Diff against the previous release, not just read the headline.**
   `BENCHMARK_REPORT.json` is the status/objective regression baseline:
   compare the new one with the committed one problem by problem, and
   account for every status flip and every objective that moves. Timing
   is only comparable on the machine that produced the Ipopt reference
   (see `benchmarks/README.md`).
4. **Run the fixture sweep against the previous release binary.**
   `scripts/sweep-fixtures.sh`, both legs, previous tag's `pounce` vs this
   one. The benchmark report asserts status and objective; the sweep is
   what shows iteration counts and engine routing moving — the gh#544
   206 → 812 regression shipped through a release because nothing at
   release time compared iteration counts. If `crates/pounce-minima` or
   `crates/pounce-cli/src/minima` changed since the last tag, run
   `scripts/sweep-minima.sh` too.

   **Sweep on x86_64 as well as on the Mac.** FERAL has separate NEON
   (aarch64) and AVX2+FMA (x86_64) dense kernels, so on nonconvex models
   the two architectures take different trajectories. That is legitimate
   rounding, not a defect, but it means an arm64 sweep says nothing about
   x86, which is what CI and most users run. The FERAL 0.21 bump (#1038)
   is the case. On arm64, `pooling_rt2stp` looked better (−4391.826 in 82
   iterations). On x86 the base solve stalled for 700 iterations behind a
   δ_x-ladder latch and needed a rescue (904 iterations against 0.20's
   162). Only the first CI run showed it. An amd64 container reproduces
   CI's kernel path: colima's Rosetta for Linux reports `avx2` and `fma`.

   ```sh
   docker run --rm --platform linux/amd64 -v "$PWD":/src -w /src \
     -e CARGO_TARGET_DIR=/src/target-x86 rust:1-bookworm \
     bash -c 'cargo build --release --bin pounce &&
              scripts/sweep-fixtures.sh target-x86/release/pounce sweep-x86.txt'
   ```

   Build the previous tag's binary the same way. colima shares only
   `$HOME` with the VM, so keep the worktree and its output under `$HOME`.
   The benchmark suites still run on the Mac, so their per-model objective
   and iteration moves on nonconvex models describe arm64 trajectories:
   say so in the report.
5. Copy the regenerated `benchmarks/BENCHMARK_REPORT.{md,json}` and
   `benchmarks/figures/` into the release PR, with a commit message naming
   the swept SHA and what moved. If anything solver-facing merges after the
   sweep, re-run it — the version bump below does not count, since it
   touches no solve path.

### Pre-flight

1. Make sure `cargo login` is set up (one-time: `cargo login <token>` from
   https://crates.io/me) — only needed for a local publish; CI uses the
   `CARGO_REGISTRY_TOKEN` secret (see Automation below).
2. Move the CHANGELOG's `## [Unreleased]` entries under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading. That section becomes the GitHub
   Release body.
3. Bump the workspace version in `Cargo.toml` (root `[workspace.package]` and
   every entry in `[workspace.dependencies]` that points at one of our crates
   — they must all match the new version) **and** the two PyPI projects
   (`python/pyproject.toml`, `pyomo-pounce/pyproject.toml`) to the same
   X.Y.Z. `scripts/check-release-consistency.sh` verifies all three agree; run
   it before tagging. If the version bump is non-trivial, do it as its own
   commit.
4. Bump `ARG POUNCE_VERSION` in `docker/Dockerfile.release` to the same
   X.Y.Z. `scripts/check-release-consistency.sh` fails if you forget — and
   it is worth knowing why it is checked rather than trusted: the release
   image *pip-installs* that pinned version, so a stale ARG produces an
   image **tagged** X.Y.Z that **contains** the previous release. The build
   succeeds and the smoke test passes, because the old wheel works fine.
5. Bump `CITATION.cff` to match: set `version:` to the new release version and
   `date-released:` to the release date. GitHub's "Cite this repository"
   widget reads these. (The `doi:` is the Zenodo *concept* DOI and stays put.)
6. Rebuild the browser demo's Pyodide wheel with
   `crates/pounce-wasm/build-wheel.sh` and commit what it stages under
   `crates/pounce-wasm/web-python/wheels/`. The wheel is committed and its
   filename carries the `python/pyproject.toml` version, so after step 3 the
   staged one is stale: `build-wheel.sh --check`, run by CI's `WebAssembly
   build + smoke` job, fails the release PR with `staged wheel is
   pounce_solver-<old>-… but python/pyproject.toml is at <new>`.
   `check-release-consistency.sh` does not check it. Renaming the file is not
   a fix — the wheel embeds the solver, so the demo would run the previous
   release under the new version's name. The script installs its own
   Pyodide toolchain under `target/` on first run (needs CPython 3.13).
7. Run `cargo package --workspace --exclude pounce-py --exclude
   pounce-studio-pyo3 --exclude iter-diff --exclude pounce-wasm` to catch
   missing metadata, broken links, or dirty-tree errors. It packages and then
   compiles every publishable crate, so breakage appears here rather than
   three crates into the real release.

   **Do not use `scripts/publish-crates.sh --dry-run` for this**, even though
   it exists and reads like the obvious choice. It stops at the *second*
   crate, always, on a release commit:

   ```
   [2/20] cargo publish -p pounce-linalg --dry-run
   error: failed to select a version for the requirement `pounce-common = "^0.11.0"`
     candidate versions found which didn't match: 0.10.0, 0.9.0, 0.8.0, ...
   ```

   That is not a defect in the tree — it is what step 3 just did. `cargo
   publish --dry-run` resolves each crate's dependencies against the **live**
   crates.io index, and the whole point of a release commit is that the new
   version is not there yet. Only the leaves (`pounce-common`,
   `pounce-studio-core`) can ever pass. The failure is therefore expected and
   carries no information, which is worse than useless: it trains you to skip
   the step.

   `cargo package --workspace` is the one that works, because it resolves
   intra-workspace dependencies against the versions it is packaging locally
   instead of against the index. Drop `--no-verify` — that flag skips the
   compile, which is most of the value.

   The same limitation applies to `release-crates.yml`'s `workflow_dispatch`
   dry-run mode, which runs this same script: it will fail at crate 2 on a
   release commit for the same reason. Its dry run is meaningful only when
   run against a tree whose version is *already* published.

   **Give this step its own `CARGO_TARGET_DIR`.** `cargo package` verifies a
   crate by compiling the *packaged* tarball, whose dependencies resolve to a
   local registry rather than to the workspace paths. Those registry builds
   land in the ordinary `target/` directory, and cargo fingerprints a registry
   dependency by `name@version` — which does not move between the bump commit
   and the tag. An rlib built by an earlier packaging run of the *same*
   version is therefore reused, however much the source has changed since.

   Cutting 0.11.0 hit it. Verifying `pounce-solve-report` failed with `E0609:
   no field final_declared_box_viol on type &SolveStatistics`, in a tree where
   `cargo check -p pounce-solve-report` passed and the packaged
   `pounce-nlp-0.11.0.crate` — checksum matching the one cargo had just written
   into the packaged lock — contained the field. The giveaway is in the error
   itself: the struct the compiler saw had 36 fields against the source's 52.
   `libpounce_nlp-6b0dfeb5fac168ba.rlib` had been built three days earlier,
   before gh#900 added the field, and was reused rather than rebuilt.
   `rm -rf target/package` does **not** clear it — the stale artifact is in
   `target/debug/deps`, not under `target/package`.

   So run the step as

       CARGO_TARGET_DIR=$(mktemp -d) cargo package --workspace \
         --exclude pounce-py --exclude pounce-studio-pyo3 \
         --exclude iter-diff --exclude pounce-wasm

   The loud direction is the one above: a real-looking compile error in a tree
   that is fine. The quiet direction is worse, and is the reason this note
   exists — with a warm `target/`, a defect introduced *after* an earlier
   packaging run of the same version passes this step, because the step never
   compiles the code carrying it.

   CI is unaffected: `release-crates.yml` runs on a fresh checkout with no
   `target/`. This bites local pre-flight only, which is exactly where the
   step is supposed to earn its keep.

### Real release — the tags are the release

Merge the release PR, then tag the merge commit on `main`. Pushing the tags
*is* the publish: each prefix drives its own workflow, so there is no
separate "publish, then tag" step.

```sh
git checkout main && git pull --ff-only
scripts/check-release-consistency.sh
git tag python-vX.Y.Z && git tag pyomo-pounce-vX.Y.Z && git tag vX.Y.Z
git push origin python-vX.Y.Z pyomo-pounce-vX.Y.Z   # PyPI first
git push origin vX.Y.Z                              # crates.io + container images
```

Push the two PyPI tags **at or before** `v*`: `release-docker.yml` fires on
`v*` and pip-installs the release from PyPI (see Container images below).

`release-crates.yml` runs `scripts/publish-crates.sh`, which is idempotent
(skips any crate already live at the target version), so a re-run or a
resumed run is safe. To publish locally instead:

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

1. **Verify every surface reached X.Y.Z.** crates.io with the `curl` check
   in the root `CLAUDE.md` (sanity-check `serde` first; a request without a
   User-Agent reads as NOT PUBLISHED), `pip index versions pounce-solver`
   and `pyomo-pounce`, and the container commands below.
2. **Create the GitHub Release by hand.** No workflow does it, and a tag
   alone does not make one:
   `gh release create vX.Y.Z --notes-file <the ## [X.Y.Z] CHANGELOG section>`.
3. Add a fresh `## [Unreleased]` heading to the CHANGELOG, close the release
   tracking issue, and remove the benchmark worktree
   (`git worktree remove ../pounce-bench-main`).

## Automation

| Surface                 | Workflow                              | Trigger                  |
| ----------------------- | ------------------------------------- | ------------------------ |
| crates.io (22 crates)   | `release-crates.yml`                  | `v*` tag / manual        |
| PyPI `pounce-solver`    | `release-pounce.yml`                  | `python-v*` tag          |
| PyPI `pyomo-pounce`     | `release-pyomo-pounce.yml`            | `pyomo-pounce-v*` tag    |
| ghcr.io container images| `release-docker.yml`                  | `v*` tag / manual        |
| wasm demo pages         | `docs.yml`                            | push to main             |

`release-crates.yml` needs the `CARGO_REGISTRY_TOKEN` secret and runs in the
`crates-io` environment. The GitHub Release itself is still created by hand
(`gh release create vX.Y.Z --notes-file <file>`); no workflow makes it.

## Container images and the wasm demos

Two surfaces ship on every release but are **not** covered by bumping a
version number, so they need their own attention.

### Container images (`release-docker.yml` -> `ghcr.io/<owner>/pounce`)

- **Tag ordering matters.** The release image pip-installs from PyPI, but
  the `v*` tag it fires on does not publish to PyPI — `python-v*` does. Push
  `python-v*` and `pyomo-pounce-v*` **at or before** `v*`. The workflow polls
  PyPI for ~20 minutes to absorb the normal lag; if it times out nothing is
  half-published, so re-run it from the Actions tab with `dry_run=false` once
  the wheels are live.
- **What each tag means.** `v*` publishes `:X.Y.Z`, `:X.Y`, `:latest` from
  the PyPI-based image, and that is the whole of the automatic publishing.
  The source-built image no longer goes out as `:edge` / `:sha-<short>` on
  every push to `main` (gh#599) — those tags are frozen at whatever commit
  last published them. Cut a fresh one from the Actions tab with
  `variant=source`, `dry_run=false` if a bug report needs an image of an
  unreleased commit; otherwise `make docker` builds it locally.
- **After the release, check the base image pin.** `docker/Dockerfile.release`
  was pinned to Debian trixie because the wheels published through 0.9.0
  bundled a CLI needing glibc 2.39 (#452, fixed in #456).  **Discharged for
  0.11.0**: 0.10.0 is on PyPI and its bundled CLI floors at `GLIBC_2.16`
  (`objdump -T pounce/bin/pounce` on the published manylinux wheel), so the
  base is now bookworm. The general rule survives the specific pin — the
  image installs a *published* wheel, so what constrains the base is that
  wheel's binary, not the tree's. If a release ever raises the CLI's floor,
  `scripts/check-cli-portability.sh` fails in CI first; if one somehow ships,
  re-measure on the artifact before trusting this base.
- **Verify the published image, do not assume.** The images smoke-test
  themselves at build time, but that proves the build, not the publish:

  ```sh
  docker run --rm ghcr.io/<owner>/pounce:X.Y.Z --version
  docker run --rm --entrypoint python ghcr.io/<owner>/pounce:X.Y.Z \
    -c "import pounce; print(pounce.__version__)"
  ```

### wasm demos (`docs.yml` -> GitHub Pages)

The `pounce-wasm` crate is `publish = false` and inherits
`version.workspace`, so it cannot drift on version — the risk is different:
the demo pages are rebuilt and deployed from `main` on every docs
deployment, so a change that breaks the wasm build breaks the *published
demo*, not a release artifact you can yank.

- CI runs `WebAssembly build + smoke` on every PR, and the wasm/native
  parity check covers the whole `crates/pounce-cli/tests/fixtures` set. If
  you add a solver feature reachable from a `.nl` file, that check is what
  tells you whether it survives the wasm boundary.
- A feature that needs threads, the filesystem, or process spawning will
  **not** work in the browser build (single-threaded, no
  `SharedArrayBuffer`, WASI shim only). Better to find that in the parity
  check than in a demo page.
- After a docs deployment, the pages are live immediately at `/demo/` and
  `/demo/python/` — there is no release gate in front of them.

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
