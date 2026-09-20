# Benchmarks

Comparison harnesses that exercise POUNCE against upstream Ipopt across
several NLP test suites. Each suite lives in its own subdirectory with
its own README explaining the problems, prerequisites, and how to run
it. All suites feed a single composite report at
[`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md), with run metadata (POUNCE
version, git SHA, Ipopt version, linear solver) emitted in the report
header.

The benchmark *inputs* (large `.nl` exports, compiled SIF problem
libraries) and the per-run *outputs* (logs, JSON results) are not tracked
in the repository. The per-suite README files, the source harnesses,
`benchmarks/scripts/`, the saved Ipopt references and the composite report
are tracked.

### The inputs are local and not publicly available

The maintainers keep the full set of `.nl` inputs locally, in the data root
described below, and do not publish it. Some of those files carry license
restrictions that do not allow redistribution. For example, the Mittelmann
`.nl` files are AMPL translations of Mittelmann's models, and `gams/nlpbench`
is a private GAMS-licensed clone. So the published results cannot be
reproduced from this repository alone. Three suites can be regenerated from
public sources: `make qp-generate`, `make lp-generate` and
`make large-scale-generate` download or build their inputs. The
`robot_a/b/c` Mittelmann instances can also be written without AMPL
(`mittelmann/gen_robot_nl.py`). For every other suite, the tracked README
says where the problems come from.

## Where the `.nl` live (`POUNCE_BENCH_DATA`)

The `.nl` inputs are large (~2 GB) and gitignored, so they don't have to
sit inside the checkout. Set `POUNCE_BENCH_DATA` to a stable, syncable
path and the harness reads (and regenerates) every suite's `.nl` there
instead — convenient for sharing one dataset across machines (e.g. via a
Dropbox folder) and for running from any working directory:

```sh
export POUNCE_BENCH_DATA="$HOME/Dropbox/pounce-bench-data"
```

When unset it defaults to the in-repo `benchmarks/` tree, so nothing
changes until you opt in. The data root mirrors the in-repo layout:

```
$POUNCE_BENCH_DATA/
  vanderbei/nl/*.nl
  electrolyte/nl/*.nl
  grid/nl/*.nl
  cho/nl_export_results/*.nl
  large_scale/nl/*.nl
  gas/*.nl
  water/*.nl
  mittelmann/nl/*.nl
  qp/nl/*.nl
  lp/nl/*.nl
  lpopt/nl/*.nl
```

Only the `.nl` inputs (and the `.sol` scratch the solver writes beside
them) live under `POUNCE_BENCH_DATA`. The `pounce.json` results, `logs/`,
the committed `ipopt_ma57.json` references, and the download/source caches
(`mps/`, `data/`, `source/`) stay in the repo.

**Migrating an existing checkout** — copy the current `.nl` to the new
root (mirroring the layout above), then point the env var at it. From the
repo root, with the var exported:

```sh
mkdir -p "$POUNCE_BENCH_DATA"
cd benchmarks
# Suites with an nl/ subdir (plus cho's nl_export_results) — rsync --relative
# reproduces the path under the new root:
rsync -a --relative \
  vanderbei/nl electrolyte/nl grid/nl cho/nl_export_results \
  large_scale/nl mittelmann/nl qp/nl lp/nl lpopt/nl \
  "$POUNCE_BENCH_DATA/"
# gas/ and water/ keep their .nl flat alongside a tracked README — copy just
# the .nl:
mkdir -p "$POUNCE_BENCH_DATA/gas" "$POUNCE_BENCH_DATA/water"
rsync -a gas/*.nl   "$POUNCE_BENCH_DATA/gas/"
rsync -a water/*.nl "$POUNCE_BENCH_DATA/water/"
cd ..
```

Verify a suite resolves (`make -C benchmarks -n vanderbei-run` should show
the `$POUNCE_BENCH_DATA/...` path), then the old in-repo copies can be
deleted — they're gitignored either way. The GAMS nlpbench harness under
`gams/` is unaffected by this and keeps its own layout.

## Suites

Every suite is `.nl`-driven: each is a directory of AMPL `.nl` files run
through the shared driver (`scripts/run_nl_bench.sh`). A release reruns
POUNCE (`<suite>/pounce.json`); the Ipopt-MA57 side is a committed
reference (`<suite>/ipopt_ma57.json`) regenerated only on request. The
composite loader merges the two — see [POUNCE runs vs the Ipopt
reference](#pounce-runs-vs-the-ipopt-reference) below.

| Suite | Problem class | Size | Notes |
|-------|---------------|------|-------|
| [`vanderbei/`](vanderbei/README.md) | Canonical NLP collection | 733 problems | Vanderbei's CUTE-in-AMPL transliteration; the `.nl` replacement for the retired compiled CUTEst suite |
| [`cho/`](cho/README.md)             | Parameter estimation, kinetic ODE | 1 problem, medium dense | CHO cell kinetics from Pyomo `parmest`; stress test for restoration |
| [`electrolyte/`](electrolyte/README.md) | Gibbs free-energy minimization | 13 problems | Aqueous electrolyte equilibrium; ill-conditioned by construction |
| [`gas/`](gas/README.md)             | Gas-pipeline network NLP | 4 problems, up to 21k vars | Finite-volume Euler discretization, GasLib networks |
| [`grid/`](grid/README.md)           | AC optimal power flow | MATPOWER cases | Polar-form ACOPF; canonical nonconvex grid NLP |
| [`large_scale/`](large_scale/README.md) | Synthetic large sparse NLPs | up to 100k vars | Rosenbrock, Bratu, OptControl, PoissonControl, SparseQP — `.nl` from a Pyomo generator; stresses sparse linsol |
| [`mittelmann/`](mittelmann/README.md) | Mittelmann ampl-nlp | 47 problems, up to 261k vars | Standard public NLP benchmark |
| [`qp/`](qp/README.md)               | Convex QP (Maros-Mészáros) | 138 problems, up to ~93k vars | Standard convex-QP benchmark; `.nl` from a generator that converts the qpsolvers `.mat` mirror |
| [`lp/`](lp/README.md)               | Linear programs (Netlib + small Mészáros) | small, known optima | LP *validation* set — small classic LPs pounce solves to optimality; `.nl` converted from MPS via `generate_nl.py` |
| [`lpopt/`](lpopt/README.md)         | Hard linear programs (Mittelmann lpopt) | large/degenerate subset | LP *stress* tier from the plato lpopt benchmark (even HiGHS/ipopt time out at short limits); run with a long limit |
| [`water/`](water/README.md)         | Water-network design | 6 problems | MINLPLib instances, signomial nonlinearities |
| [`warmstart/`](warmstart/README.md) | Parametric NLP **sequences** | 6 families × 3 step sizes | The one suite that is not a cold-solve set: measures warm starting (cold vs warm × IPM vs active-set SQP) over paths of related NLPs. Python-driven, not `.nl` |
| [`mpcc/`](mpcc/README.md)           | Complementarity-constrained programs (MPCC) | 11 cases, ≤ 3 vars | gh#794 Gate 0: compares the direct, exact-product/NCP, ℓ₁ exact-penalty and Scholtes-continuation routes on a corpus with known optima and known MPCC stationarity classes. Python-driven, not `.nl` |

The `warmstart/` and `mpcc/` suites sit outside the `.nl` machinery entirely — they
has no Ipopt reference and does not feed the composite report, because
it measures a different thing (the cost of a *sequence* of related
solves, cold vs warm) and needs an in-process solver handle to carry a
working set between steps. It has its own report at
`warmstart/results.md`; see [`warmstart/README.md`](warmstart/README.md).

`mpcc/` is the same shape for a different reason: an MPCC has no Ipopt
reference to compare against because the quantity under test is not
"did the NLP converge" but *which MPCC stationarity class the returned
point attains*, which no `.nl` runner reports. Its own report is
`mpcc/results-full.md`, and the Gate 0 decision it feeds is
[`dev-notes/mpcc-gate0-report.md`](../dev-notes/mpcc-gate0-report.md).

The GAMS nlpbench harness ([`gams/nlpbench/`](../gams/nlpbench/)) is no
longer aggregated into the composite report — its problem coverage
duplicated the `.nl` suites. `make gams-bench` now runs only a small
liveness smoke check of the GAMS solver-link path; see its `Makefile` for
the on-demand fuller runs.

## POUNCE runs vs the Ipopt reference

The Ipopt-MA57 reference and the per-release POUNCE run are **decoupled**:

- **Ipopt-MA57 reference** — run *once* with `make -C benchmarks
  ipopt-reference`. It runs Ipopt on every suite and writes the committed
  `benchmarks/<suite>/ipopt_ma57.json` plus a provenance stamp
  (`benchmarks/ipopt_ma57.provenance.json`: which Ipopt, which machine,
  when). Ipopt is slow and its results don't change between POUNCE
  iterations, so this is regenerated only when you ask — e.g. on a new
  machine, a new Ipopt build, or after adding problems
  (`make -C benchmarks ipopt-ref-<suite>` refreshes one suite). Commit
  these files.

- **POUNCE run** — every release runs `make -C benchmarks benchmark`,
  which reruns *only* POUNCE on each suite (`<suite>/pounce.json`,
  gitignored) and regenerates the report, comparing against the saved
  reference. If a suite has no committed reference, it is reported
  POUNCE-only with a note.

Because Ipopt solve *times* in the reference come from whatever machine
generated it, timing comparisons are only meaningful when the release
report is produced on that same machine; status / objective / iteration
counts are machine-independent.

## Running everything

All targets are driven through this directory's `Makefile`. The
top-level `Makefile` provides three convenience shims (`make benchmark`,
`make benchmark-report`, `make benchmark-<suite>`) that delegate here.

All `*-run` targets are **incremental** — they skip work if the suite's
`pounce.json` is newer than its inputs (the `.nl` files or the `pounce`
binary). To force a rerun, use the corresponding `*-rerun` target, which
wipes the suite's `pounce.json` first.

```sh
# One-time (per machine / ipopt build): generate + commit the reference
make -C benchmarks ipopt-reference

# Release sweep — rerun POUNCE on each suite (incremental), then report
make -C benchmarks benchmark

# Force every POUNCE run: wipes all pounce.json then rebuilds
make -C benchmarks benchmark-rerun

# Just regenerate the composite report (no suite reruns)
make -C benchmarks benchmark-report

# One suite at a time — POUNCE (incremental)
make -C benchmarks vanderbei-run
make -C benchmarks water-run
make -C benchmarks gas-run
make -C benchmarks electrolyte-run
make -C benchmarks grid-run
make -C benchmarks cho-run
make -C benchmarks large-scale
make -C benchmarks mittelmann-run
make -C benchmarks gams-bench

# Force a POUNCE rerun of one suite (wipes its pounce.json first)
make -C benchmarks water-rerun
make -C benchmarks vanderbei-rerun

# Refresh the Ipopt reference for one suite (rare)
make -C benchmarks ipopt-ref-water

# Or from the repo root via the shim
make benchmark
make benchmark-water
```

`make -C benchmarks help` lists every target.

## How the comparison runs

Every suite uses the **AMPL solver protocol** through the shared driver
`benchmarks/scripts/run_nl_bench.sh`. Its final argument selects which
solver(s) to run: a release run invokes it in `pounce` mode (`pounce
<file.nl>` → `<suite>/pounce.json`), and `make ipopt-reference` invokes
it in `ipopt` mode (`ipopt <file.nl> -AMPL` → `<suite>/ipopt_ma57.json`).
Both write the same `{solver,name,n,m,status,objective,iterations,
solve_time}` schema, which `benchmark_report.py` merges per suite. The
`ipopt` binary is the locally-built MA57 install at
`ref/Ipopt/install-ma57/bin/ipopt` (`make -C benchmarks build-ipopt-ma57`),
so POUNCE-MA57 is compared against the same linear-solver family.

(The earlier compiled CUTEst and large_scale Rust harnesses, which linked
`libipopt` via `pkg-config` FFI, have been retired in favour of the `.nl`
suites — there is no longer an FFI path.)

## Head-to-head solver comparisons

Alongside the per-suite runs above, two scripts grade *multiple solvers on the
same problem* against published optima rather than against each other:

| script | set | solvers | output |
|---|---|---|---|
| `scripts/compare_pounce_clarabel.py` | Netlib LP + Maros-Mészáros QP | pounce (`lp-ipm`/`qp-ipm`) vs Clarabel | `clarabel_compare.md` |
| `scripts/compare_qp_four_way.py` | Maros-Mészáros QP (138) | pounce `qp-ipm`, `nlp`, `qp-active-set`; Clarabel | `qp_four_way.md` |

```sh
python3 benchmarks/scripts/compare_qp_four_way.py [--limit N] [--time-limit S]
```

The `qp-active-set` column was added because that engine had almost no
benchmark exposure: it was graded on 3 hand-constructed problems in
`adversarial_convex.md`, while the IPM path was graded on all 138 here. It is
the engine behind `algorithm=active-set-sqp` — the one that matters for
warm-started sequences (MPC, branch-and-bound nodes, homotopy continuation) —
and a degeneracy-rich published set with ground truth is where it should be
held to account.

Two things to know before reading its numbers:

- **The timings are cold-start**, which is the case an active-set method is
  least suited to; its design point is carrying the working set *across*
  solves. Read the speed line as a floor, not a verdict.
- **`solved-but-wrong` is the number that matters.** A solver reporting success
  with an objective off ground truth is far worse than one that fails visibly,
  so the report breaks those out separately for the active-set column.

Both scripts need the release binary to be current — they invoke
`target/release/pounce` directly, so rebuild before running or you will
benchmark whatever was there last (`cargo build --release --bin pounce`; check
with `pounce --about`, which prints the commit).

## Adding a new suite

1. Create `benchmarks/<suite>/` with a `README.md` describing the
   problem class, sources, prerequisites, and how to run.
2. Add a `<suite>-run` target (POUNCE mode → `<suite>/pounce.json`) and
   the suite name to the `nldir_*` map and `REF_SUITES` in
   `benchmarks/Makefile`, so `make ipopt-reference` covers it too. The
   composite loader (`load_suite`) picks up `<suite>/pounce.json` +
   `<suite>/ipopt_ma57.json` automatically once you add the suite to the
   loop in `benchmark_report.py:main()`. Then run `make ipopt-ref-<suite>`
   once and commit the reference.
3. The `.gitignore` whitelists every `benchmarks/*/README.md`,
   `benchmarks/*/ipopt_ma57.json`, and `benchmarks/scripts/`. Everything
   else under `benchmarks/` is ignored by default (including the
   per-release `pounce.json`) — add explicit `!benchmarks/<suite>/<file>`
   lines for any other tracked source (e.g. a problem generator script).
