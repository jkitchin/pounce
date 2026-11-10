# Julia native complementarity integration checks

This harness exercises the original MPCCModels volt-var/volt-watt cases through
Pounce.jl, including its legacy NLPBlock and native MOI.Complements paths. It
checks the original NLP and variable bounds, `min(G,H)`, direct volt-var errors
and volt-watt cap violations. Solver status alone is never the acceptance test.
This is an integration check, not a performance comparison or a global-optimum
claim. See [SOURCE.md](SOURCE.md) for source, revision, license and citation.

Use an isolated Julia environment containing this checkout's Pounce.jl, JuMP,
ADNLPModels, NLPModels and the pinned MPCCModels checkout. For example:

```julia
using Pkg
Pkg.activate("/tmp/pounce-mpcc-validation"; shared=false)
Pkg.develop(path="/path/to/pounce/Pounce.jl")
Pkg.develop(path="/path/to/MPCCModels.jl")
Pkg.add(["JuMP", "ADNLPModels", "NLPModels"])
```

Check out MPCCModels revision `19a4b275c8ebb3fb5f7a051cfe36c45ad1a948de` first.
The harness checks that revision and rejects tracked model-source edits. Then:

```sh
POUNCE_LIBRARY_PATH=/path/to/libpounce_cinterface.dylib \
POUNCE_NATIVE_REVISION=<exact-native-source-commit> \
julia --project=/tmp/pounce-mpcc-validation \
  benchmarks/julia_mpcc_opf/run.jl /path/to/MPCCModels.jl
```

Use `.so` for Linux. Preserve the log, environment Manifest, native revision and
binary hash with results. The script prints wrapper/model revisions, worktree
state, binary hash, Julia/dependency versions and per-case outcomes. Dependencies
for these application checks are intentionally outside the package test target.
