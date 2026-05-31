# Large-Scale Synthetic Suite

Large, sparse, synthetic NLPs designed to stress the sparse linear algebra
path and workspace sizing of both POUNCE and Ipopt. Problems are
parameterised by a size and scaled up to around 100K variables. They are
emitted as AMPL `.nl` files by `generate_nl.py` (Pyomo) and run through the
same dual-solver `.nl` driver (`benchmarks/scripts/run_nl_bench.sh`) as
every other suite — there is no compiled Rust harness and no libipopt FFI.

The five problems cover the main structural patterns POUNCE needs to handle
efficiently:

- **rosenbrock** — generalized/chained Rosenbrock (CUTE `GENROSE`),
  unconstrained, tridiagonal Hessian, `f* = 1` at `x = 1`. Default `n = 2000`
  (kept small: it is fundamentally O(n) Newton iterations).
- **bratu** — 1-D Bratu BVP `-u'' = λ e^u` with a 3-point stencil; pure
  feasibility (objective ≡ 0), nonlinear equality constraints. Default
  `n = 10000`.
- **optcontrol** — discretised linear-quadratic optimal control; quadratic
  objective, block-tridiagonal linear dynamics. Default `T = 50000`
  (`n = 100001`, `m = 50001`).
- **poisson** — 2-D Poisson boundary control on a K×K grid; quadratic
  objective, 5-point-stencil linear constraints. Default `K = 200`
  (`n = 80000`, `m = 40000`).
- **sparseqp** — convex sparse QP, tridiagonal `Q`, cyclic three-term
  inequality rows, box bounds. Default `n = 50000`.

These are intentionally synthetic rather than drawn from a public library so
the size can be scaled freely without shipping giant fixtures, and so both
solvers see the exact same problem.

## Contents

- `generate_nl.py` — Pyomo generator; writes one `.nl` (plus matching
  `.row`/`.col` name maps) per problem into `nl/`
- `nl/` — generated `.nl` files (gitignored; regenerated locally)
- `pounce.json` / `ipopt_ma57.json` — latest POUNCE and Ipopt/MA57 results

## Prerequisites

- `pyomo` (for `generate_nl.py`)
- `ipopt` (MA57 build) for the comparison side, same as the other `.nl`
  suites

## How to run

From the repo root:

```bash
make -C benchmarks large-scale            # generate .nl if missing, then run
make -C benchmarks large-scale-rerun      # force a rerun
make -C benchmarks large-scale-generate   # (re)generate the .nl files only
```

Regenerate at a different scale, or generate a single problem:

```bash
python3 generate_nl.py --scale 0.1          # 10% of every default size
python3 generate_nl.py optcontrol --optcontrol-t 1000
```

## Output

- `nl/*.nl` — generated problems
- `pounce.json` / `ipopt_ma57.json` — POUNCE and Ipopt per-problem results
  in the canonical
  `{solver,name,n,m,status,objective,iterations,solve_time}` schema

This suite feeds the composite `benchmarks/BENCHMARK_REPORT.md` via
`load_domain_results()` in `benchmark_report.py`.
