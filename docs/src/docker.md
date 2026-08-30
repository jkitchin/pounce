# Docker & Containers

Prebuilt images carry the whole POUNCE surface — the `pounce` CLI, the
Python API, and the Pyomo plugin — with no Rust toolchain and no `pip
install` on your side. This is the path of least resistance on a cluster,
where you often cannot install a toolchain anyway, and the quickest way to
try POUNCE without touching your environment.

```sh
docker pull ghcr.io/jkitchin/pounce:latest
docker run --rm ghcr.io/jkitchin/pounce:latest --version
```

## Which tag

| Tag | Contains | Use it when |
|---|---|---|
| `latest` | the newest release | you just want POUNCE |
| `X.Y.Z` (e.g. `0.9.0`) | exactly that release, forever | reproducibility — papers, cluster job scripts, CI |
| `X.Y` (e.g. `0.9`) | newest patch of that minor series | you want fixes but not feature changes |

Pin `X.Y.Z` for anything you intend to re-run months later — `latest` and
`X.Y` both move under you.

Published images are cut from releases only. There are `edge` and
`sha-<short>` tags in the registry from when every commit to `main` was
built, but they are **frozen at whatever commit last published them** — do
not read `edge` as "tip of `main`". To run an unreleased fix, build the
image yourself from a checkout; it takes minutes and needs no Rust
toolchain on your side beyond Docker:

```sh
git clone https://github.com/jkitchin/pounce && cd pounce
make docker          # -> pounce:dev, compiled from the tree you checked out
docker run --rm pounce:dev --version
```

## Running solves

The entrypoint is the `pounce` CLI, so arguments after the image name go
straight to it:

```sh
docker run --rm ghcr.io/jkitchin/pounce:latest --list-problems
docker run --rm ghcr.io/jkitchin/pounce:latest --problem rosenbrock
```

Your own problems live on the host, so mount a directory. The working
directory inside the image is `/work`:

```sh
docker run --rm -v "$PWD:/work" ghcr.io/jkitchin/pounce:latest \
  problem.nl print_level=5 tol=1e-10
```

That writes `problem.sol` next to `problem.nl` in the mounted directory —
see [Running Solves](cli.md) for the full option surface. The image runs as
UID 1000 rather than root, so files it creates in a bind mount are not
root-owned on the host. If your host UID differs, add `--user "$(id -u):$(id
-g)"`.

## Python and Pyomo

Override the entrypoint to get a shell or an interpreter:

```sh
docker run --rm -it --entrypoint python ghcr.io/jkitchin/pounce:latest
docker run --rm -it --entrypoint bash -v "$PWD:/work" ghcr.io/jkitchin/pounce:latest
```

Both `import pounce` (with numpy and scipy) and Pyomo's
`SolverFactory('pounce')` work out of the box:

```sh
docker run --rm -v "$PWD:/work" --entrypoint python \
  ghcr.io/jkitchin/pounce:latest my_model.py
```

The optional extras are *not* installed — no JAX, no PyTorch, no `plotly`,
no GAMS bindings. Add what you need in a derived image:

```dockerfile
FROM ghcr.io/jkitchin/pounce:0.10.0
USER root
RUN pip install --no-cache-dir "pounce-solver[jax]==0.10.0"
USER pounce
```

## On a cluster (Apptainer / Singularity)

Most HPC sites run Apptainer (formerly Singularity) rather than Docker,
because it needs no daemon and no root. It pulls Docker images directly:

```sh
apptainer pull pounce.sif docker://ghcr.io/jkitchin/pounce:0.10.0
apptainer run pounce.sif problem.nl
```

Two differences from `docker run` are worth knowing before you write a job
script:

- **You are yourself.** Apptainer runs the container as your own user, not
  the image's, so output files land with your ownership and no `--user`
  flag is needed.
- **`$HOME` and `$PWD` are already there.** Apptainer bind-mounts them by
  default, so a `.nl` file in your submit directory is usually visible with
  no `-B` flag at all. Add `-B /scratch:/scratch` (or your site's
  equivalent) for anything outside them.

Build the `.sif` once on a login node and reuse it — pulling on every array
task hammers the registry and will get rate-limited. In a Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=pounce
#SBATCH --cpus-per-task=4

apptainer exec $HOME/images/pounce.sif \
  pounce $SLURM_SUBMIT_DIR/problem.nl --json-output result.json
```

Pin the digest rather than the tag if the run has to be reproducible years
later — a tag can be re-pushed, a digest cannot:

```sh
apptainer pull pounce.sif docker://ghcr.io/jkitchin/pounce@sha256:<digest>
```

## Building your own

You do not need the images to be published — both Dockerfiles are in the
repository and build from a clone. From the repository root:

```sh
make docker          # compiles the current working tree -> pounce:dev
make docker-release  # installs the released wheels -> pounce:<version>
```

`make docker` is the one to reach for when testing a branch: it compiles
whatever is checked out, and stamps the commit into `pounce --about` so the
image can say what it contains. `make docker-release` needs no Rust
toolchain and takes seconds. See [`docker/README.md`](https://github.com/jkitchin/pounce/blob/main/docker/README.md)
for build arguments and the `.dockerignore` caveat.

## What is not in the image

The **HSL MA57 backend** is absent, and cannot be added by us: CoinHSL is
license-restricted and not redistributable. Passing `linear_solver=ma57`
will not work in a container. The pure-Rust FERAL backend is the default
everywhere and needs no external libraries — see
[Installation](installation.md#hsl-ma57-backend-optional) if you have a
CoinHSL license and want a local build with it.

The **GAMS link** is likewise absent, since it needs your own GAMS install
and license on the host. See [GAMS](gams.md).
