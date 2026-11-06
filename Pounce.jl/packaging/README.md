# Local JLL release preparation

`build_tarballs.jl` describes Pounce_jll 0.15.0, pinned to native source
`b4e5ffb641990738fd69cb53c02ae5966fd68edc`. It builds the pure-Rust FERAL C
interface, installs its public header and EPL-2.0 license, and requests
BinaryBuilder's library audit. Optional HSL/MA57 binaries are not included.
Linux declares `CompilerSupportLibraries_jll` for Rust's `libgcc_s` unwinder;
the initial local audit identified this dependency explicitly.
The four candidate targets are Linux glibc and macOS, each x86_64/aarch64.
Listing a candidate is not evidence that it passed a build or test. Windows
remains outside the advertised platform set until its binary and suite pass.

This preparation directory follows BinaryBuilder's
[build recipe](https://docs.binarybuilder.org/stable/building/) and
[local JLL generation](https://docs.binarybuilder.org/stable/building/#Building-a-custom-JLL-package-locally)
interfaces. It permits `--deploy=local` and rejects remote deployment and
registration flags. No publication is part of the commands below.

Create a separate environment containing BinaryBuilder (the recipe is being
validated with 0.6.6), and use a Linux runner or Docker. On macOS the Docker VM
must be able to read Julia's artifact paths and write the temporary build tree.
Use a dedicated build directory; BinaryBuilder writes `products` and `build`
there. Set `JULIA_PKG_DEVDIR` to a task-specific directory for generated wrappers.

```sh
mkdir -p /tmp/pounce-jll-build
cd /tmp/pounce-jll-build
# Select the intended local Docker context before running, if using Docker.
BINARYBUILDER_RUNNER=docker JULIA_PKG_DEVDIR=/tmp/pounce-jll-dev \
  julia --project=/path/to/binarybuilder-environment \
  -e 'using BinaryBuilder, Pkg; BinaryBuilder.devdir[] = Pkg.devdir(); include(popfirst!(ARGS))' \
  /path/to/pounce/Pounce.jl/packaging/build_tarballs.jl \
  --deploy=local aarch64-linux-gnu
```

Use additional triplets (or the recipe's default matrix) to build other targets.
The launcher resets BinaryBuilder 0.6.6's cached `devdir` to the runtime Pkg
setting; otherwise a precompiled BinaryBuilder can retain its old development
directory despite `JULIA_PKG_DEVDIR`. This is a version-specific tooling detail.
BinaryBuilder produces archive/tree hashes, audit output, and a generated
`Pounce_jll` package. Save these with source revision, BinaryBuilder/toolchain
versions, and test logs. Inspect dynamic dependencies; an unaudited host build
does not establish the binary's minimum supported Linux/macOS version.

## Stage the Julia release candidate

The working source-checkout package remains installable while the JLL is
unpublished. The staging script makes a separate, reviewable candidate copy,
adds the generated JLL's actual UUID as a runtime dependency, and changes the
default library path to the JLL product. It preserves custom-library overrides,
the optional MOI extension, and all API/tests. It never edits the working package
or publishes a repository. Choose a new output directory.

```sh
julia /path/to/pounce/Pounce.jl/packaging/stage_release.jl \
  /tmp/pounce-jll-dev/Pounce_jll /tmp/pounce-release-candidate
julia --project=/tmp/pounce-release-candidate --startup-file=no \
  -e 'using Pkg; Pkg.test("Pounce")'
```

Compare the staged `Project.toml` and `src/Pounce.jl` with the source package
before adopting them for release. The generated README explains binary-backed
installation. Keep the candidate and its generated JLL together until publication;
the temporary local artifact URLs are not suitable for public distribution.
`Candidate.toml` records the source revision/worktree state and generated JLL
identity. On a clean target machine or container, use the stricter validator:

```sh
julia --startup-file=no /path/to/pounce/Pounce.jl/packaging/validate_release.jl \
  /tmp/pounce-release-candidate --no-rust
```

It requires no `cargo`/`rustc` on PATH, no custom-library override, core-only
loading without MOI, and a default library inside Julia's artifact store before
running the complete package suite. Run in a fresh process after staging.

## Release gates

- Build and audit each advertised platform; record archive and tree hashes.
- Run the full package suite on Julia 1.10 and current Julia, with both artifact
  selection and a custom-library override. Test alongside Ipopt.jl.
- Install into a clean environment with no Rust toolchain or source-checkout
  library available, and verify `Pounce.library_path()` is in the JLL artifact.
- Verify core-only installation does not install/load MOI; test both load orders.
- Run the pinned BMOPFTools scenarios and retain independent residual checks.
- Review native compatibility (currently 0.15.x), licenses/source attribution,
  generated dependency URLs, and platform minimum OS requirements.
- After explicit authorization, publish the candidate branch/artifacts and run
  hosted CI before release/registration. Use the generated JLL UUID and actual
  published artifact hashes/URLs; do not invent a registry identity or URLs.

The roadmap and validation notes record which gates actually have evidence.
The current recipe and staging scripts alone do not prove release readiness.
