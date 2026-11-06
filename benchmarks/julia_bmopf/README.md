# Julia / BMOPFTools adapter validation

This application harness tests the pinned four-bus teaching feeder described in
[SOURCE.md](SOURCE.md). It is separate from the `.nl` composite report.
No external private benchmark data are needed. Build the native C library first.

```sh
julia --startup-file=no benchmarks/julia_bmopf/setup.jl /tmp/pounce-bmopf-env
POUNCE_NATIVE_REVISION=<commit-used-to-build-the-library> \
  julia --project=/tmp/pounce-bmopf-env --startup-file=no \
  benchmarks/julia_bmopf/run.jl pounce /tmp/pounce-bmopf.json
POUNCE_NATIVE_REVISION=<commit-used-to-build-the-library> \
  julia --project=/tmp/pounce-bmopf-env --startup-file=no \
  benchmarks/julia_bmopf/run.jl ipopt /tmp/ipopt-bmopf.json
julia --project=/tmp/pounce-bmopf-env --startup-file=no \
  benchmarks/julia_bmopf/compare.jl /tmp/pounce-bmopf.json /tmp/ipopt-bmopf.json
```

Keep the generated environment's `Project.toml` and `Manifest.toml` with the JSON
results. Each output records package versions/tree hashes, model/source commits,
fixture and native-library SHA-256 hashes, Julia/CPU, options, and worktree state.
Use a clean committed checkout for reportable measurements. Run sequentially on
an otherwise idle host, with warm package caches, and repeat in fresh processes
for timing distributions. The first row still includes first-call compilation;
the repeat row uses the same model. No automatic solution warm start is assumed.

Both arms use identical per-unit JuMP models, exact sparse derivatives, `tol=1e-8`
and `max_iter=500`. Pounce uses FERAL; Ipopt uses its bundled MUMPS. This is an
adapter comparison with different linear backends, not a controlled comparison
of interior-point algorithms. A MadNLP arm is deferred until the same parameter
and start contracts are validated for it; the design study covers that wrapper.

Each arm checks ten solves: first/repeat, explicit primal and constraint/bound
dual starts, demand parameter factors 0.5/1.4/1.0 with fresh-model controls, and
recovery after an intentionally exhausted iteration limit. The parameter sweep
uses normal `optimize!` to exercise the wrapper's update path; BMOPFTools'
`resolve_opf!` would intentionally reset the cached optimizer instead.

Acceptance requires successful status **and** model residual at most `1e-6`
and no errors from BMOPFTools' independent physical solution profiler. Checks
use the scenario's actual SI demand; fresh controls and cross-solver comparisons
require matching local objectives within `rtol=atol=1e-6`. This does not prove
global optimality. JSON retains the profiler's residuals, tolerances, and warnings.
Because that profiler explicitly leaves terminal KCL and branch equations
unassessed, `physics.jl` additionally reconstructs complex series currents from
the fixture's SI impedance matrices and returned bus voltages. It checks those
against returned branch currents, recomputes load currents from the specified
complex powers, and verifies downstream KCL, thermal limits, voltage limits,
source voltage, and nonnegative independently reconstructed losses. Current and
voltage tolerances are `1e-5 A` and `1e-5 V`. This checker is specific to the
pinned series-only feeder; it is not a general BMOPF network verifier.

Timing boundaries are explicit:

- `build_seconds`: model/device/KCL expression construction, before optimizer
  attachment or AD initialization; null when no model was built.
- `optimize_elapsed_seconds`: wall time around JuMP `optimize!`, including cache
  copying, AD initialization, and compilation if needed.
- `moi_solve_seconds`: each wrapper's own timer, retained under its own name.
- `native_solve_seconds`: Pounce's C API timer; null for Ipopt because this harness
  has no equivalent public native timer. Do not compare unlike boundaries.
- `diagnostic_copy_seconds` / `diagnostic_ad_setup_seconds`: separate warmed,
  disposable model diagnostics, after the solves. The AD diagnostic invokes the
  pinned wrappers' private `_setup_model` helpers (AD and sparsity initialization).
  It is not an additive phase breakdown of the timed solves and does not prewarm
  them. Keep this diagnostic synchronized when changing either wrapper version.

First-call overhead and native callback compilation can dominate this small
example. It supports reproducible correctness and latency observations, not
claims about utility-scale OPF performance or persistent factorization reuse.
