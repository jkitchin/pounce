# POUNCE container images

Two Dockerfiles, one published image. Both produce the same three
interfaces — the `pounce` CLI, `import pounce`, and Pyomo's
`SolverFactory('pounce')` — so a job script written against one runs
unchanged on the other. They differ only in where the solver comes from.

| File | Solver comes from | Build time | Published as |
|---|---|---|---|
| `Dockerfile.release` | PyPI wheels, pinned to `X.Y.Z` | seconds | `:X.Y.Z`, `:X.Y`, `:latest` on a `v*` tag |
| `Dockerfile` | compiled from the tree you hand it | minutes | nothing automatic; `:edge`, `:sha-<short>` on a manual run |

Only the release image publishes on its own (gh#599). The source image used
to go out as `:edge` on every commit to `main`; that trigger is gone, so
`:edge` in the registry is stale and `make docker` is how you get an image of
an unreleased fix. It still builds on every PR that touches `docker/**`, so
it cannot rot, and `release-docker.yml` → *Run workflow* with
`variant=source`, `dry_run=false` cuts one on demand.

User-facing documentation lives in [`docs/src/docker.md`](../docs/src/docker.md)
— pull commands, Apptainer/Singularity usage, bind mounts. This file covers
the build side.

## Building locally

Both must be built **from the repository root**, because the source build's
context has to contain `Cargo.toml`, `crates/`, `python/`, and
`pyomo-pounce/`. The Makefile targets fill in the version and commit for you:

```sh
make docker          # source build -> pounce:dev
make docker-release  # released build -> pounce:<version from Cargo.toml>
```

The equivalents by hand:

```sh
docker build -f docker/Dockerfile -t pounce:dev \
  --build-arg POUNCE_BUILD_GIT="$(git rev-parse --short=8 HEAD)" .

docker build -f docker/Dockerfile.release -t pounce:0.12.0 \
  --build-arg POUNCE_VERSION=0.11.0 .
```

Each image runs its own smoke test as the final build step — CLI solve,
`import pounce`, Pyomo plugin lookup — so a build that succeeds is an image
that works. There is no separate test to run.

## Things that will bite you

**The `.dockerignore` is an allowlist.** A full POUNCE working tree is tens
of gigabytes (`target/` alone reaches ~40G), so the root `.dockerignore`
excludes everything and re-adds the handful of paths the build reads. **If
you add a new top-level directory the build needs, you must re-add it
there** — otherwise it is silently absent from the context and the build
fails with a missing-file error that points nowhere near the cause.

Two of its entries are load-bearing beyond size:

- `.cargo/config.toml` is gitignored and machine-specific — it hard-codes a
  local `COINHSL_DIR`. In the context it would make `pounce-hsl/build.rs`
  assert on a path that does not exist in the image.
- `.git` is excluded for size (~90M of history the compile does not read).
  That is why the commit SHA is passed in as `POUNCE_BUILD_GIT` instead:
  `crates/pounce-cli/build.rs` normally shells out to `git` to stamp
  `pounce --about`, and without either the SHA *or* the build arg an image
  cannot say which commit it contains.

**The source build's artifacts must be copied out inside the compiling
`RUN`.** It uses `--mount=type=cache` for the cargo registry and `target/`,
and cache-mount contents do not survive into the next layer. Splitting that
step in two produces an image with no binaries in it and no error to explain
why.

**Both images are based on Debian trixie, and the release image has no
choice.** The published Linux wheels are tagged `manylinux2014` (glibc 2.17)
but bundle a `pounce` CLI built on the release runner's host — Ubuntu 24.04,
glibc 2.39 — because `release-pounce.yml` runs `cargo build` outside the
manylinux container that maturin builds the extension module in. On bookworm
(glibc 2.36) the smoke test fails outright:

```
pounce/bin/pounce: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.39' not found
```

on both amd64 and arm64 — a bug in the wheels, not in the image, and one that
also broke `pip install pounce-solver` on most HPC distributions.

That is **fixed in main** (#452 / #456): the CLI is now built inside the
manylinux container and `scripts/check-cli-portability.sh` fails CI if its
glibc floor ever exceeds the wheel's manylinux tag. But this image installs
from PyPI, and the fix only lands there when a `python-v*` tag rebuilds the
wheels. **Once the next release ships, drop `DEBIAN_RELEASE` to bookworm or
lower** — the floor is now 2.16 — and delete the note in
`Dockerfile.release`. The source image is self-contained and does not care;
it tracks the release image only so there is one base to reason about.

**Neither image enables `--features ma57`.** CoinHSL is license-restricted
and cannot be redistributed. The pure-Rust FERAL backend is the default and
needs no external libraries; `linear_solver=ma57` in a container will not
work.

## Publishing

`.github/workflows/release-docker.yml` pushes the release image to
`ghcr.io/jkitchin/pounce` on a `v*` tag. Read the header comment there for
the trigger matrix; two points matter when cutting a release:

1. **Tag order.** The release image installs from PyPI, but the `v*` tag
   this workflow fires on does not publish to PyPI — `python-v*` does. Push
   the `python-v*` and `pyomo-pounce-v*` tags at or before `v*`. The
   workflow polls PyPI for ~20 minutes to absorb the normal lag; if it times
   out, nothing is half-published — re-run it from the Actions tab with
   `dry_run=false` once the wheels are live.

2. **Check the package visibility once, after the first publish.** For this
   repo the package inherited the repository's public visibility and
   anonymous pulls worked immediately — no manual step was needed. That is
   worth verifying rather than assuming, because if a package ever does come
   out private, `docker pull` and `apptainer pull` fail for everyone but you
   with an auth error that reads like the image does not exist. The honest
   check is an unauthenticated one:

   ```sh
   docker logout ghcr.io
   curl -s "https://ghcr.io/token?scope=repository:<owner>/pounce:pull" \
     | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])' \
     | xargs -I{} curl -so /dev/null -w '%{http_code}\n' -H "Authorization: Bearer {}" \
         https://ghcr.io/v2/<owner>/pounce/manifests/edge
   ```

   `200` means public. If it is not, flip it under *Packages → pounce →
   Package settings → Change visibility*.
