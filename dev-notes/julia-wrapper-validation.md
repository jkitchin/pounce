# Julia wrapper validation notes

The first wrapper was developed against native POUNCE commit
`b4e5ffb641990738fd69cb53c02ae5966fd68edc` (0.15.0).
The Rust solver is unchanged by the wrapper contribution.
Ipopt.jl sources and adapted tests are pinned in `Pounce.jl/NOTICE.md`.

The local refactoring and release-preparation gates are complete. The staged
JLL candidate passes all **16** platform/version/library-selection runs, each
with **3,662 assertions**. The pinned BMOPFTools validation, startup measurements,
minimum-dependency tests, and native C tests are detailed below. Exact binary
hashes and runtime environments are in the
[artifact inventory](julia-wrapper-artifacts.json). Hosted CI, publication, and
registration remain separately authorized release actions; none is claimed here.

## An inherited test exposed a native limited-memory limitation

The Ipopt.jl test `test_manually_evaluated_primal_status` constructs:

- feasibility objective (zero);
- one variable, `0 <= x <= 1`;
- an affine interval `0.1 <= x <= 1.1`;
- a vector nonlinear oracle imposing `0.25 <= x <= 0.75`;
- a scalar nonlinear constraint `-1 <= log(x) <= 1`.

The oracle supplies no Hessian, so the adapter selects limited-memory.
The original test supplies no start and the wrapper starts at zero, projected
onto the variable bounds. On the native commit above, this returns
`Restoration_Failed`. Its result-classification assertions assume a feasible
solve without checking the status, and therefore fail.

The kill-switch control on that exact model disabled
`infeas_stationarity_tol`, `presolve`, `acceptable_iter`,
`acceptable_progress_kappa`, `obj_scale_certificate_threshold`,
`primal_noise_floor_kappa`, `dual_divergence_retry`, and NLP scaling.
It returned the same failure and point.

The same functions passed directly to the C API, bypassing MOI, reproduce:

| Hessian | Starting x | Native status |
|---|---:|---|
| limited-memory | 0 | Restoration_Failed |
| limited-memory | 0.5 | Solve_Succeeded |
| exact | 0 | Solve_Succeeded |
| exact | 0.5 | Solve_Succeeded |

Reproduce with:

```sh
julia --project=Pounce.jl dev-notes/julia-feasibility-repro.jl
```

This establishes a native limited-memory limitation, not its internal cause.
No solver fix is included here. The result-classification test now starts
at the known strictly feasible point `x=0.5` and asserts successful termination
before manipulating the stored results. Other tests exercise limited-memory
solves. This should not be read as a general robustness or OPF benchmark.

## Test adaptations

Ipopt-specific SPRAL/HSL library tests and its extension alias test are omitted.
The vector-oracle timing test uses an infinity sentinel to verify timers reset
between solves, instead of sleeping and assuming Ipopt's evaluation counts.
Callback exceptions are rethrown only after the C API returns, including
interrupts, so the interrupt test expects an exception and no usable result.

Native `check_derivatives_for_naninf=yes` is refused by POUNCE (gh #483);
the wrapper instead checks callback outputs for finiteness.

At the initial wrapper revision, binary distribution, Windows CI, and BMOPFTools
benchmarks were follow-up work. Later application and binary results are recorded
below; Windows remains outside the advertised binary platform set.

## Wrapper design study and model updates

The [Ipopt.jl and MadNLP study](julia-wrapper-design-study.md) identified stale
nonlinear bounds after a solve–modify–solve sequence. The fix, regression tests,
and comparative reproducer are committed at
`8b142ab0adb8752d9925bd35e781a6484f3e792d`.

On that commit, using the unchanged native library from
`b4e5ffb641990738fd69cb53c02ae5966fd68edc`, the full package suite passed
3,453 assertions on each of Julia 1.10.12 and 1.12.6 (macOS arm64). This includes
82 new assertions for nonlinear bound updates, fresh-model comparisons,
cached/direct JuMP models, parameter updates, and consistent result invalidation.
The runs used MathOptInterface 1.54.0, JuMP 1.32.1, and Ipopt.jl 1.16.0.

The committed reproducer also ran against registered MadNLP 0.11.0 and
Ipopt.jl 1.16.0. Updated Pounce and its fresh-model control both returned
`1.3862943686119529` for the new bound. Both reference packages retained their
old result after the bound update; their fresh-model controls returned the
expected new optimum. Package identities and the pre-fix values are recorded
in the study. No upstream issue or PR was published as part of this work.

## Statistics and callback refactor

Implementation commit: `cc42dec83da1f1a1e95cdf29b50f3b905b4bc2c4`.
Native library: unchanged POUNCE 0.15.0 built from
`b4e5ffb641990738fd69cb53c02ae5966fd68edc`.

On macOS arm64, the full package suite passed 3,586 assertions on each of
Julia 1.10.12 and 1.12.6, with MathOptInterface 1.54.0, JuMP 1.32.1, and
Ipopt.jl 1.16.0. The 133 additional assertions cover native statistics,
callback installation/replacement/removal, snapshot freshness and task scope,
restoration restrictions, exceptions/recovery, recursive solve rejection, and
direct-model parameter introspection. The callback tests exposed missing
parameter constraint getters; these are now implemented too.

The documentation consistency check passed (54 reachable pages, no dead links),
as did whitespace checks. No Rust source changed, so the native test suite was
not rerun for this Julia-only refactor. Hosted CI remains unrun/unpublished.

The allocation probe used Julia 1.12.6 and the same native library/MOI version
before and after the change:

| Variables queried | Baseline `85cb9d43` | Refactor `cc42dec8` |
|---|---:|---:|
| 128 | 155,648 bytes | 0 bytes |
| 512 | 2,179,072 bytes | 0 bytes |

These are Julia allocations for a warmed scalar-query sweep inside one regular
callback. Warmup populates the new cache, so the first snapshot copy is excluded.
They are not total callback/solver allocations or a wall-time speedup measurement.
Both probes record wrapper and native source revisions, Julia/MOI/native versions,
library path, and working-tree state. The baseline wrapper code was unchanged;
only the new roadmap document was untracked.

Reproduce from the repository root in an environment containing Pounce and
MathOptInterface, setting the native revision to the commit actually used to
build the selected library:

```sh
POUNCE_NATIVE_REVISION=b4e5ffb641990738fd69cb53c02ae5966fd68edc \
  julia --project=/path/to/environment --startup-file=no dev-notes/julia-callback-allocation-probe.jl
```

The same script can be copied to a baseline checkout for comparison. The probe's
diagonal quadratic model is self-contained and uses no external benchmark data.

## Explicit model/cache/result lifecycle

At wrapper commit `b8a9f4e3`, the full suite passed 3,642 assertions on each of
Julia 1.10.12 and 1.12.6 on macOS arm64, using the unchanged native library from
`b4e5ffb641990738fd69cb53c02ae5966fd68edc`. The 56 new lifecycle assertions check
derivative-cache reuse across bound/parameter updates, rebuilds after structural
edits, result retention for settings/starts, option resets, failed-attempt
invalidation/recovery, and rejection of callback mutations before data changes.
Existing tests continue to compare updated nonlinear models with fresh models.

Public MOI/JuMP entry points and low-level C bindings are retained. Tests that
inspect implementation details now use the explicit native/data containers.
These fields are not a supported user API; model/result queries use MOI.

## Optional MOI and startup

Wrapper commit `7527a8fe2d2771a01c8f9091202959152d545f21` passes **3,653 assertions**
on each of Julia 1.10.12 and 1.12.6 (macOS arm64), with MOI 1.54.0, JuMP 1.32.1,
and Ipopt.jl 1.16.0. Native POUNCE remains 0.15.0 from
`b4e5ffb641990738fd69cb53c02ae5966fd68edc`. Tests cover an installation with no
MOI dependency, both import orders, and native-library overrides after
precompilation, including a solve through the extension.

The objective-transition probe exposed an inherited Hessian assembly defect:
after replacing `x²` with `x³`, the Hessian at `x=2` summed to 14 instead of 12.
Commit `9247e5a3` weights only the active objective and suppresses objective
curvature in feasibility mode while retaining constraint curvature. Five
regression assertions cover these transitions. This is an analytic derivative
contract check, not a diagnosis of a reported native solver failure.

The [startup probe](julia-startup-probe.jl) uses one fresh Julia process per
sample, warm package caches, and the same HS071 model/environment/native library.
Three samples compare lifecycle baseline `b8a9f4e3` against the extension code
at `7527a8fe`. Two after-samples carry commit `06f3d312`, which adds only the
application harness and does not change `Pounce.jl`. Raw selected observations
are retained in [julia-wrapper-measurements.json](julia-wrapper-measurements.json).
Initial cache-building samples and exploratory after-samples overlapping image
downloads are excluded; the retained samples are baseline 2–4 and extension 4–6.

| Phase | Before median (range), seconds | After median (range), seconds |
|---|---:|---:|
| `import Pounce` | 0.651 (0.598–0.663) | 0.0054 (0.0044–0.0095) |
| `using JuMP` after Pounce | 0.077 (0.075–0.081) | 0.844 (0.819–1.103) |
| First model construction | 1.138 (1.088–1.152) | 0.526 (0.522–0.540) |
| First `optimize!` | 4.467 (4.290–4.479) | 1.913 (1.906–1.936) |
| Sum through first solve, median of per-process sums | 6.345 | 3.277 |
| Subsequent model construction | 0.000491 | 0.000484 |
| Subsequent `optimize!` | 0.001845 | 0.001853 |

MOI loading now occurs with JuMP, so that individual import phase increases.
The combined startup measurement decreases by about 48% on this probe.
Precompilation shifts work into cache creation (the new extension took about
7 s to precompile on this host); this table excludes installation/cache creation.
It is neither a steady-state solver speedup nor a broad performance guarantee.
At `06f3d312`, the callback allocation probe still reports zero Julia bytes for
warmed sweeps of both 128 and 512 scalar queries; the initial snapshot is excluded.

## BMOPFTools application validation

Harness/wrapper revision: `06f3d31259bc81239c2c8b45ce2bdf0889112b0a`, clean worktree
for all measured application runs. Native revision is unchanged as above.
BMOPFTools 0.1.0 is pinned to `542a48ba7f4c95e50b8acdc790a5323a4ef24930`.
The [source record](../benchmarks/julia_bmopf/SOURCE.md) identifies the author,
CC BY 4.0 teaching fixture, modifications, and limits. The committed benchmark
Manifest captures the entire dependency graph, including Julia 1.12.6,
JuMP 1.32.1, MOI 1.54.0, Ipopt.jl 1.16.0/native 3.14.19, and native dependencies.

Three fresh-process samples per solver (samples 2–4) each pass ten scenarios:
first/repeat, explicit primal/dual starts, demand parameters 0.5/1.4/1.0 with
fresh-model controls, and recovery after `max_iter=0`. Sample 1 is retained only
as exploratory correctness evidence because an allocation probe overlapped it.
All 30 cross-solver objective comparisons pass `rtol=atol=1e-6`. The 77-variable,
149-constraint model uses the unchanged four-bus fixture, per-unit formulation,
exact derivatives, `tol=1e-8`, and `max_iter=500`. Pounce uses FERAL; Ipopt uses
MUMPS. This is a small adapter/latency exercise with fixed demand, not evidence
about large OPF cases or matched-backend algorithm performance.

All solutions have zero physical-profiler errors. Because BMOPFTools explicitly
leaves branch/device equations and terminal KCL unassessed, the harness also
reconstructs SI line currents from the impedance matrix and bus voltages, checks
returned branch currents, recomputes load currents from specified complex powers,
and checks downstream KCL, source/operating voltages, thermal limits, and losses.
Worst residuals across the selected scenarios:

| Check | Pounce | Ipopt | Acceptance |
|---|---:|---:|---:|
| Model constraint violation, model units | 7.79e-13 | 1.13e-10 | ≤1e-6 |
| Independently reconstructed terminal KCL, A | 3.65e-9 | 9.12e-7 | ≤1e-5 |
| Returned vs reconstructed branch current, A | 1.13e-12 | 2.08e-8 | ≤1e-5 |

Timing medians and ranges over the three selected processes:

| Phase | Pounce seconds | Ipopt seconds |
|---|---:|---:|
| First model construction | 6.535 (6.507–6.667) | 7.388 (6.455–11.256) |
| First JuMP `optimize!` | 2.149 (2.139–2.161) | 2.339 (2.053–3.259) |
| Repeated JuMP `optimize!` | 0.00181 (0.00180–0.00193) | 0.00111 (0.00109–0.00126) |
| Explicit-start JuMP `optimize!` | 0.00188 (0.00185–0.00255) | 0.000827 (0.000776–0.000946) |
| Recovery JuMP `optimize!` | 0.01127 (0.01107–0.01137) | 0.00584 (0.00542–0.00593) |

First/repeat solves take two iterations, explicit starts one; recovery takes
31 Pounce / 30 Ipopt iterations. Parameter-update counts and all timer boundaries
are preserved in the measurements JSON. Pounce's median native timer is 1.611 s
on the first solve and 0.00154 s on the repeated solve; first-call compilation
can occur inside that timer. Ipopt has no equivalent native timer exposed by
this harness. A separate warmed diagnostic initializes AD/sparsity in 16.3 μs
(Pounce) / 19.0 μs (Ipopt) on disposable models; it is not an additive breakdown
of the timed solves. The sizeable first-call ranges and different backends limit
what can be inferred from these timings. No solver-wide ranking is claimed.

## Native compatibility and Linux validation

At wrapper `1d985b52d4901479967b01cce91c476e17bf09a9`, the complete package suite
passes **3,662 assertions** on each of macOS arm64 Julia 1.10.12 and 1.12.6.
The additional nine assertions cover rejection of an unrelated shared library
and native version boundaries. The loader now validates every C symbol used by
the bindings and accepts POUNCE 0.15.x. Other minor versions require an explicit
ABI review; matching Ipopt symbol names alone do not establish compatibility.

The earlier extension revision `7527a8fe` passes **3,653 assertions** on Linux
arm64 Julia 1.12.6, in the official `julia:1.12.6-bookworm` image
(`sha256:0d9424d38430320596424b22a7d03add630ea24313e139eaff5b521093f74f17`).
Its native library was built from the same pinned `b4e5ffb6` source using Rust
1.91.1 in `rust:1.91-bookworm`, whose pulled manifest digest was
`sha256:c1e5f19e773b7878c3f7a805dd00a495e747acbdc76fb2337a4ebf0418896b33`.
All 95 `cargo test --locked -p pounce-cinterface` tests also pass in that Linux
environment. These are local container checks on a dedicated Colima VM, not
hosted CI and not evidence of an audited portable JLL artifact. The existing
user VM and default Docker context were left unchanged.

After the complete Linux run, the updated `1d985b52` loader and its library tests
were copied into the isolated Linux test checkout. All 15 library-selection and
native-version assertions pass there too, including extension solves through
file/directory overrides and rejection of `libjulia` as an unrelated library.

The declared dependency floors were tested separately at wrapper
`0e81791fb82decde80b262e4c396de95500f2149`: Julia 1.10.12, MOI **1.48.0**, JuMP
**1.29.0**, and Ipopt.jl 1.16.0, with those three packages pinned and
`Pkg.test("Pounce"; allow_reresolve=false)`. All **3,610 assertions** pass on
macOS arm64 with the same native `b4e5ffb6` library. The count is lower because
the older MOI conformance suite has fewer tests; no wrapper tests were disabled.

A final macOS arm64 C-interface check at branch commit
`75c206dd142f6d148734c1976f500a72b8f7b6ed` passes all **95 tests** with Rust
**1.91.1**, using `cargo +1.91.1 test --locked -p pounce-cinterface`. Native
Rust sources and Cargo files still match `b4e5ffb6`; only validation notes were
uncommitted during this check. Test-log SHA-256:
`a86f4c306c376c1a2fd3e452534ef0dc248c866e329295ac5ef5e6f6a93f4246`.

## Local binary release preparation — local gates complete

The current runtime matrix is below. Each passed cell is the complete
**3,662-assertion** suite at staged wrapper `add2e6e3`, with the four audited
native `b4e5ffb6` artifacts. Exact environment settings, artifact/library hashes,
build-audit log hashes, and test log hashes are retained in the
[artifact inventory](julia-wrapper-artifacts.json). Intel runs use Rosetta;
the Linux 1.13.3 setup workaround is detailed below. Override tests select a copy
of the audited library outside its artifact directory; they test path selection,
not compatibility with an independently built native binary.

| Platform | Artifact 1.10.13 | Artifact 1.13.3 | Override 1.10.13 | Override 1.13.3 |
|---|---|---|---|---|
| Linux arm64 | Passed | Passed | Passed | Passed |
| Linux x86_64 | Passed | Passed | Passed | Passed |
| macOS arm64 | Passed | Passed | Passed | Passed |
| macOS x86_64 | Passed | Passed | Passed | Passed |

The earlier measurements deliberately retain their Julia 1.10.12/1.12.6
versions. On 2026-11-06 the official [Julia downloads page](https://julialang.org/downloads/)
lists stable **1.13.3** and LTS **1.10.13**; the local `release` channel still
points to 1.12.6. Current-release validation must therefore also cover 1.13.3
and 1.10.13. The results below and the artifact inventory track that matrix;
no prior test counts are attributed to the newer versions. The manual binary workflow exposes all four
versions for reproducibility and defaults to 1.13.3.

The local-only BinaryBuilder recipe first built Linux arm64 from native
`b4e5ffb6` using BinaryBuilder 0.6.6 and Rust 1.91.0. Its audit found the Linux
`libgcc_s.so.1` runtime dependency unresolved. Recipe correction `350f204c`
declares `CompilerSupportLibraries_jll`, but its completed rebuild still could
not resolve the dependency: the default GCC 5 compiler selected libgfortran3,
while the runtime package supplies libgfortran5 artifacts. Both archives are
exploratory evidence, not accepted release artifacts. BinaryBuilder's default
allows audit failures without a failing exit code, so exit status alone is
insufficient evidence.

At `08fc113f` plus the GCC 10 recipe change, a dependency-resolution probe with
BinaryBuilder 0.6.6 confirms libgfortran5 tags for both Linux architectures and
matching CompilerSupportLibraries artifact trees: arm64
`3cd2fb12f766ee959993b19d1c546f9b1e3aa868`, x86_64
`e0ae8e1988425737cec3f6c615497f3284d9e771`. The recipe also explicitly sets
`ignore_audit_errors=false`. This metadata probe alone did not establish binary
compatibility; the subsequent complete builds supply the audit evidence.

The complete Linux arm64 rebuild at recipe
`b892f46a51ce42fd5192c6f9633b14bd24200d4c` now **passes the fatal audit**.
BinaryBuilder resolves `libgcc_s.so.1` through CompilerSupportLibraries_jll,
sets the library SONAME, and finds the EPL license. The generated runtime wrapper
initializes CompilerSupportLibraries_jll before POUNCE. The archive contains only
the POUNCE library, public header, and license; it does not duplicate the runtime
dependency's libraries. Native source remains `b4e5ffb6`, with Rust 1.91.0 and
GCC 10.2.0. Artifact tree: `a796ceec2106222a27debcc6bc909869492c09b6`;
archive SHA-256: `1b0d2946e91b4893641af83468496479585d76da4fd01a7cae3b025e9f94bd03`.
Runtime tests were pending when this artifact was built; combined-JLL results
are recorded below.

ELF inspection records `$ORIGIN` as the runpath and library SHA-256
`2c7b6c155b0ae50611d5c43ed5676fdc59b50c5b3e836ace92199a9e54da6d9e`.
The library has a weak `__cxa_thread_atexit_impl@GLIBC_2.18` reference.
An actual `ctypes.CDLL` load on CentOS 7 arm64 fails for that versioned symbol,
even though `ldd -v` finds the version namespace. CentOS 7 is therefore not
covered by this artifact. The build's GCC 10 shard
`d5a9463d95b3ebf75ac06d8ab163ee9fe43e3d0d` contains glibc **2.19**, verified
from its libc binary; this is the conservative Linux arm64 build baseline.
The CentOS image used was arm64 manifest
`73f11afcbb50d8bc70eab9f0850b3fa30e61a419bc48cf426e63527d14a8373b`
from the official `centos:7` manifest list. This loader check is separate from
the complete Julia suites on Debian 12 recorded below.

The earlier Linux x86_64 build at recipe `350f204c` also finished and confirmed
the unavailable libgfortran3 runtime mapping. Its archive is exploratory and
was superseded by the GCC 10 build below. The auditor also identifies CPUID-based instruction
dispatch and therefore cannot statically establish a minimum instruction set;
x86 runtime checks remain necessary.

The GCC 10 x86_64 rebuild at recipe `73f45e96` reached the final link, where
Rust 1.91's bundled LLD rejected compressed debug sections in
`libc_nonshared.a` because it lacks zlib support. A small control using the
same explicit `x86_64-linux-gnu-cc` linker reproduces that error; adding
`-C linker-features=-lld` produces a library that passes the control's audit.
The first control using `gcc` instead of Cargo's `cc` did not reproduce the
failure and is not counted as that evidence. The recipe now selects the
toolchain's GNU linker for Linux x86_64 using the
[documented Rust opt-out](https://blog.rust-lang.org/2025/09/18/Rust-1.90.0/).
The following full POUNCE build verifies this linker change on the solver.

That complete Linux x86_64 build now **passes the fatal audit** at recipe
`28297b0b`, with the GNU linker and runtime dependency resolved. The CPUID
inspection warning remains; it is not an unresolved library dependency.
Intel macOS also **passes the fatal audit** at recipe `73f45e96`, with only
libSystem linked and a minimum macOS load command of **10.12**. Julia itself
may impose a newer OS requirement. Neither audit substitutes for runtime tests.

The [artifact inventory](julia-wrapper-artifacts.json) records all four accepted
archives, tree hashes, and exact recipe commits. Their archive SHA-256 values
were checked before copying to one local product directory. BinaryBuilder's
`--skip-build --deploy=local` then generated a combined Pounce_jll 0.15.0+0,
recomputing all four matching tree hashes and locating the library products.
This assembly step does not rerun the audits; the accepted individual builds
above supply that evidence. All generated download URLs are local file URLs.

At staged wrapper `add2e6e3388d0ed5d27d72ceaf4265bc49f21524`, the combined JLL
passes **3,662 assertions** on Julia **1.13.3** on each of Linux arm64 and
macOS arm64, with MOI 1.54.0, JuMP 1.32.1, and Ipopt.jl 1.16.0. Each run uses
a fresh depot and validates core-only artifact loading before the complete
suite. The Linux container has no Rust toolchain and mounts only the candidate,
generated JLL, archives, and validation scripts, so no source-build library is
available. On macOS, Cargo and rustc are excluded from PATH. Both resolve the
expected artifact library and leave MOI unloaded until requested. These runs
also pass all **3,662 assertions** on Julia **1.10.13** on each arm64 platform,
with the same staged wrapper and dependency versions. These four runs cover the
default artifact path. Intel results follow; all override runs also pass, as
shown in the matrix above. Log hashes and exact environments are in the inventory.

Intel macOS also passes **3,662 assertions** on each of Julia **1.10.13** and
**1.13.3** with the same
staged wrapper and dependencies. This runs the x86_64 Julia executable and native
artifact under Rosetta on macOS **26.7.2**; it is not a physical Intel-machine run.
Each fresh depot, no-Rust check, artifact path, and log hash is recorded separately.

The first Intel Linux Julia 1.13.3 run under the dedicated VM's Rosetta translator
failed before loading Pounce: Julia's zstd executable selected system libzstd
1.5.4 instead of bundled 1.5.7. Direct ELF-loader invocation of that same
executable succeeds. A container-local shim applies that invocation only to
the decompressor, without changing Julia/Pounce's library search environment.
The next run received SIGSEGV in Julia's GC safepoint during `Pkg.instantiate`,
also before loading Pounce. A subsequent run explicitly sets
`JULIA_NUM_THREADS=1` and `JULIA_NUM_GC_THREADS=1`; it **passes all 3,662
assertions** at staged wrapper `add2e6e3`, with MOI 1.54.0, JuMP 1.32.1, and
Ipopt.jl 1.16.0. The selected library is the accepted Intel Linux artifact. These exploratory failures are not passing
tests, nor evidence of a Pounce native failure. The artifact inventory records
the translation mode, settings, and successful log hash; native hosted Linux CI remains
a separate release gate.

Intel Linux also passes **3,662 assertions** on Julia **1.10.13** with the same
staged wrapper, native artifact, and package dependency versions. This run uses
a fresh depot under Rosetta with one Julia worker and GC thread. Julia 1.10's
decompressor needs no shim. Together these runs complete the eight-cell default
artifact matrix. The complete override matrix now also passes: each of the eight
platform/version pairs runs the full suite against an explicit copy outside the
artifact store. Every copy's SHA-256 matches its accepted native artifact.

The staged wrapper's `src`, `ext`, and `test` files remain identical to the
current branch except for the staging tool's documented JLL import/default-path
changes. The manual CI definition uses Julia 1.12.6 for BinaryBuilder, stages
with the chosen test Julia, and mounts only the candidate, generated JLL, product
archives, and validator scripts into its clean test container. It runs both
selection modes. YAML and shell syntax checks pass; hosted execution has not
been launched and remains a release gate after authorization to publish.

The combined macOS arm64 artifact also passes the pinned BMOPFTools application
checks on Julia **1.12.6**, at staged wrapper
`204c80e9f57c5649ec38a0197800706510237a34` and harness
`11a215aca2d2b20e3f669f663853d20412b9e4b4`, both clean. Each solver passes all ten
scenarios, and all ten cross-solver objective comparisons pass. Maximum model
residuals are **7.79e-13** for Pounce and **1.13e-10** for Ipopt; independent
terminal KCL errors are **3.65e-9 A** and **9.12e-7 A**. The loaded Pounce library
has the accepted artifact's SHA-256
`347e1edb48c44f35317bc1a51e72e65009f42ad25d870d680c55c86ccf78b339`.
Adding the staged package/JLL with `Pkg.PRESERVE_ALL` leaves every original
application dependency record unchanged except Pounce, adding only Pounce_jll.
The [measurement data](julia-wrapper-measurements.json) retain source/dependency
identities, per-scenario correctness results, and report/manifest hashes. These
are correctness checks; their timings are excluded because a separate Linux VM
validation was active. The earlier three-process performance evidence stands
on its own.

The first macOS arm64 build (recipe `350f204c`, native `b4e5ffb6`, Rust 1.91.0)
completed, but its audit reported an unwanted `/usr/lib/libiconv.2.dylib`
dependency. Inspection of the actual library's 93 undefined symbols found no
iconv references. A minimal Rust program did not reproduce the extra linkage,
so it does not establish where that link was added. A separate linker control
at wrapper `b892f46a` explicitly requests an unused iconv library; inspection
and audit output confirm that `-Wl,-dead_strip_dylibs` removes that dependency
while retaining libSystem. The recipe applies this flag on Darwin; the subsequent full solver build and
combined-JLL runs supply its audit and runtime evidence. The first macOS archive is exploratory
(tree `95e88077236cbcc37b850a0b26af149421cb2ec1`, SHA-256
`d898dd529763ad0ebe59accdcf8a147ea682672389d0e09f5ea5044b7237b94a`).

The corrected macOS arm64 artifact at recipe
`73f45e96039651570b7197e4868f369945be292f` now **passes the fatal audit**.
Inspection confirms only `/usr/lib/libSystem.B.dylib` remains as a dependency,
with dylib ID `@rpath/libpounce_cinterface.dylib` and minimum macOS **11.0**.
Native source and Rust version remain `b4e5ffb6` / 1.91.0. Artifact tree:
`3eb5a4b359039259aa9db75130616c897e893fb7`; archive SHA-256:
`3d1569f1c4410a03e5ba4a73194f9486a27f1ca6e4045945f50257c6633ea757`;
library SHA-256:
`347e1edb48c44f35317bc1a51e72e65009f42ad25d870d680c55c86ccf78b339`.
The later combined-JLL arm64 runtime results are recorded above. An earlier corrected macOS
compile was deliberately stopped to switch the dedicated VM from QEMU to its
supported Rosetta mode; this accepted archive comes from a fresh complete build.

An exploratory full-suite run of the earlier Linux artifact at staged wrapper
`350f204c` failed during the core-only installation subprocess: the VM kernel
killed Julia's Pkg precompiler for lack of memory while three native builds
were active. The run was then stopped deliberately (exit 143). Its test and
kernel logs are retained locally; it is not a passing validation result or
evidence of a wrapper defect. Final suites are scheduled separately from the
native builds.

The staging tool at wrapper `350f204c586fe9de625fbc77dbe6cb5d8e940c8d` generated a
separate Pounce package with the actual generated Pounce_jll UUID
`96a3a933-3cb5-5901-bb57-278b133f379d`, version 0.15.0+0. Its source loader diff
consists of importing Pounce_jll and selecting its product as the default;
custom-library override behavior and the optional MOI extension are preserved.
`Candidate.toml` records the clean source revision and local JLL identity.

An exploratory install into a fresh `julia:1.12.6-bookworm` Linux arm64 container
with a new depot, no Cargo/Rust on PATH, and no `POUNCE_LIBRARY_PATH` successfully
loads native 0.15.0 from
`/depot/artifacts/3e988b8297986233f5da66668143e7ca91b684ee/lib/libpounce_cinterface.so`.
The optimizer/extension remains unloaded in this core-only process. This proves
the staging/default-artifact mechanism. It did not waive the runtime-dependency
audit correction or replace the later full artifact-backed package suites.
All generated JLLs and archives remain local; nothing has been uploaded,
registered, pushed, or submitted as a PR.
