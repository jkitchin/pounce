# Julia wrapper design study

Studied on 2026-11-06, starting from wrapper commit
`e0c2e03ccce1d2936f3a951573f37932ad845b79`. Native POUNCE sources and the
tested 0.15.0 library are from `b4e5ffb641990738fd69cb53c02ae5966fd68edc`.
This study does not change the Rust solver.

## Sources and scope

- Ipopt.jl: source commit
  [`592aa741adaad85377c72aafa9bc08fdf0cc908e`](https://github.com/jump-dev/Ipopt.jl/tree/592aa741adaad85377c72aafa9bc08fdf0cc908e),
  the same adapter revision used by the initial Pounce.jl contribution.
- MadNLP.jl: source commit
  [`53f1fa00ea257644148416bc8305a8f17d724b0c`](https://github.com/madsuite-org/MadNLP.jl/tree/53f1fa00ea257644148416bc8305a8f17d724b0c).
- Runtime probes: Julia 1.12.6 on macOS arm64, MathOptInterface 1.54.0,
  Ipopt.jl 1.16.0 with native Ipopt 3.14.19, MadNLP 0.11.0.
  Registered package tree hashes were
  `d5a4fb77d030c56258a33a7046195506055ce5cc` (Ipopt) and
  `b82ad9ba469372af51e67fc64f289be387b7a760` (MadNLP).
  Source inspection uses the commit links above; runtime observations use these
  registered releases, not an assertion that their trees equal repository HEAD.

The review covered package boundaries, C callbacks, nonlinear evaluators,
model mutation, warm starts, statistics, precompilation, and tests. This is not
a solver speed or OPF robustness comparison.

## Architecture to retain

Ipopt.jl is the closer implementation model: Julia-owned buffers and callbacks
feed a native solver through a C ABI, with a separate MOI adapter. MadNLP is a
Julia-native solver: its MOI adapter creates an
[`NLPModels.AbstractNLPModel`](https://github.com/madsuite-org/MadNLP.jl/blob/53f1fa00ea257644148416bc8305a8f17d724b0c/ext/MadNLPMOI/MOI_wrapper.jl)
and invokes `MadNLPSolver` / `solve!`. Its NLPModels interface is useful for
solver-independent experiments, but adding a second modeling interface is not
necessary for our first JuMP contribution.

Both support the MOI nonlinear evaluator approach. Keep the adapter responsible
for MOI indices, bounds, objective sense, dual conventions, starts, sparse
derivatives, and result classification; keep the C wrapper responsible for
ABI, ownership, and exception containment. Do not port MadNLP's numerical
solver, GPU array handling, or linear solver dispatch into this wrapper.

The actual POUNCE header remains the authority for the C ABI. In particular,
POUNCE callback booleans are one byte; copying Ipopt.jl's C declarations
unchanged would be inappropriate. Preserve the existing GC rooting, null-safe
structural callbacks, and deferred exception propagation.

## A confirmed solve–modify–solve defect

Before the fix, solve `max x` subject to `0 <= x <= 3` and `exp(x) <= 2`,
change the nonlinear upper bound to `4`, then solve again. The second optimum
must be `log(4)`, approximately `1.3862943611198906`.

| Wrapper/runtime | First x | Second x after update |
|---|---:|---:|
| Pounce at `e0c2e03c`, native `b4e5ffb6` | 0.6931471880488821 | 0.6931471880488821 |
| Ipopt.jl 1.16.0 / Ipopt 3.14.19 | 0.6931471880488821 | 0.6931471880488821 |
| MadNLP 0.11.0 | 0.6931471880488822 | 0.6931471880488822 |

All three reported `LOCALLY_SOLVED` after the second solve. This is an adapter
state issue: the nonlinear model's constraint changes, but `NLPBlockData`
retains a snapshot of the old bounds. Rebuilding only the solver handle or NLP
wrapper consumes that old snapshot. A freshly constructed model with the new
bound gives the expected answer. Reproduce the comparison with
`julia --project=<environment> dev-notes/julia-bound-update-repro.jl Ipopt MadNLP`;
the optional packages must already be installed in that environment.

Pounce now invalidates the nonlinear setup when its bounds change. This
conservative fix rebuilds the evaluator as well as the native handle. It can
later be optimized to refresh only bounds, with the regression tests retained.
Parameter values still update the evaluator in place, but invalidate the old
solution. Status getters now agree with `ResultCount` when a new native handle
is pending.

`Pounce.jl/test/updates.jl` tests lower, upper, equality, and interval nonlinear
bounds, compares results and duals with a fresh model, covers cached and direct
JuMP use, and checks affine/quadratic/variable bounds and parameters. The initial
44-assertion nonlinear regression had 24 failures before the fix; the completed
82-assertion update suite passes after it. This illustrates why conformance
tests and agreement with another wrapper are insufficient by themselves.

## Warm starts and benchmarking

MadNLP's [warm-start tutorial](https://github.com/madsuite-org/MadNLP.jl/blob/53f1fa00ea257644148416bc8305a8f17d724b0c/docs/src/tutorials/warmstart.md)
distinguishes copying a previous primal/dual solution from retaining a solver
whose symbolic factorization can be reused. The latter assumes unchanged
problem structure. Its MOI `optimize!` still constructs a new `MadNLPSolver`
on each call; ordinary repeated JuMP solves do not imply this retained-solver
behavior.

Make the same distinction for Pounce. Explicit primal/dual starts are supported;
keeping a C problem handle does not promise reuse of the Rust solver's
factorization. A future persistent session interface should have its own
contract. No automatic previous-solution reuse is promised by this package.

For BMOPFTools comparisons, record model construction/AD setup, first-call
compilation, subsequent wall time, native solver time, linear solver choices,
and options separately. Pounce's current `MOI.SolveTimeSec` measures the wrapper
call including setup; MadNLP returns its solver counter. These should not be
treated as identical timing boundaries. Validate the returned OPF point using
the application's residual checks, not only termination status or objective.

## POUNCE-specific performance follow-ups

Two costs are visible in the source; neither has been benchmarked in this study:

1. The inherited adapter registers an intermediate callback even without a
   user callback, just to count iterations. In POUNCE, registering one requests
   a live iterate snapshot every iteration (`CCallbackTnlp::wants_current_iterate`).
   Use the existing native `GetIpoptIterCount` after solving, and register an
   intermediate callback only when needed. `GetIpoptSolveTime` can expose native
   timing alongside the wrapper timer. Test callback installation after an
   initial solve and callback replacement when making this change.
2. `CallbackVariablePrimal` currently copies the entire primal vector for each
   scalar query. Reading all n variables therefore copies O(n²) elements.
   Cache one snapshot per callback invocation, clearing it before every new
   invocation and on callback exit. Preserve restoration-phase restrictions.

These are adaptation opportunities specific to the POUNCE native contract,
not reasons to copy more upstream code mechanically.

## Packaging, latency, and testing follow-ups

Both [Ipopt](https://github.com/jump-dev/Ipopt.jl/blob/592aa741adaad85377c72aafa9bc08fdf0cc908e/Project.toml)
and [MadNLP](https://github.com/madsuite-org/MadNLP.jl/blob/53f1fa00ea257644148416bc8305a8f17d724b0c/Project.toml)
make MOI an optional Julia extension. That is worth adopting if low-level-only
Pounce users matter. Keeping a direct MOI dependency is reasonable for the first
JuMP-focused PR; extension loading adds API/type-inference considerations for
the exported `Optimizer` alias and needs load-order tests.

Ipopt uses `Ipopt_jll` for ordinary installation. A Pounce JLL, automatic artifact
selection, and a documented custom-library override are the path to convenient
public installation. The current source-checkout build is an explicit first-PR
limit. Do not copy OpenBLAS/LBT initialization merely because both reference
packages use it: POUNCE's default native backend has different dependencies.

Ipopt's [extension workload](https://github.com/jump-dev/Ipopt.jl/blob/592aa741adaad85377c72aafa9bc08fdf0cc908e/ext/IpoptMathOptInterfaceExt/IpoptMathOptInterfaceExt.jl)
precompiles representative MOI model construction and solves. MadNLP also uses
PrecompileTools. Measure Pounce startup and first-solve latency before adding a
small workload; avoid making package precompilation accidentally depend on a
developer's absolute native-library path.

MadNLP's [MOI tests](https://github.com/madsuite-org/MadNLP.jl/blob/53f1fa00ea257644148416bc8305a8f17d724b0c/test/MOI/MOI_interface_test.jl)
combine MOI conformance with additional nonlinear problem tests. Keep our C
boundary tests separate from MOI conformance and application scenarios. Before
claiming OPF readiness, add a small reproducible BMOPFTools case with physical
feasibility checks. Hosted CI, Windows binaries, and that application validation
remain outside the local checks completed so far.
