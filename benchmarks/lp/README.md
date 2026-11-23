# LP suite — NETLIB + small Csaba Meszaros (validation LPs)

Small, **tractable** linear programs intended for *validation*: an
interior-point solver should reach optimality quickly on essentially all
of them. This is deliberately not a stress test — the hard,
multi-million-row Mittelmann `lpopt` instances live in
`benchmarks/lpopt/`. Here the point is breadth of small, well-understood
LPs whose optima are known.

Like every other suite, this is `.nl`-driven through
`benchmarks/scripts/run_nl_bench.sh`, so each problem is recorded in the
standard schema — `{solver, name, n, m, status, objective, iterations,
solve_time}` — in `lp/pounce.json` and merged into the composite
`BENCHMARK_REPORT.md`.

## Two sources

1. **NETLIB LP** test set — <https://www.netlib.org/lp/data/>. The
   canonical ~100-instance LP collection (Gay, 1985). Files are stored in
   netlib's *compressed* format and must be expanded with the netlib
   `emps` program; `generate_nl.py` fetches and compiles `emps.c` (old
   K&R C, built with `cc -w`) and pipes each instance through it to get
   standard MPS. **89 instances converted.**

   Deferred (over the size cap): `dfl001`, `fit2d`, `fit2p`. Not present
   as plain `emps` files in `lp/data/` and therefore skipped: `qap8`,
   `qap12`, `qap15` (404, "see NOTES"), `stocfor3` and `truss` (Fortran
   shar bundles), `standgub` (GUB `'MARKER'` lines; no reference optimum).

2. **Csaba Meszaros LP test set** (size-filtered subset) —
   <https://old.sztaki.hu/~meszaros/public_ftp/lptestset/> (the live
   mirror of the sztaki FTP). Subdirs `misc/`, `problematic/`,
   `stochlp/`, `New/` hold gzip'd netlib-format files; `generate_nl.py`
   `gunzip`s then `emps`-expands them. The infeasible `infeas/` subdir is
   excluded. A **size filter** keeps only the small instances.
   **282 instances converted** out of 375 candidate files; the remaining
   93 are deferred as over the cap (the well-known large Meszaros LPs:
   `stormg2-*`, `stat96v*`, `dbir*`, `nsct*`, `model*`, `world`, `mod2`,
   `lpl*`, `nug15`, …) plus one non-LP archive (`stoprobs.zip`).

**Total: 371 `.nl` (89 NETLIB + 282 Meszaros).**

## Size cap

In `generate_nl.py` (override with `--max-vars/--max-cons/--max-nnz`):

    MAX_VARS = 10_000
    MAX_CONS = 10_000
    MAX_NNZ  = 200_000

Each instance is dimension-screened (via `highspy`) *before* conversion;
anything over the cap is reported `deferred(too large)` and not written.

## How the `.nl` are produced

Both sources yield standard MPS after `emps` expansion. Each MPS is
converted to `.nl` by `mps_to_nl.py` — the proven converter copied
verbatim from `benchmarks/lpopt/` (parse with HiGHS `highspy`, rebuild as
a Pyomo `ConcreteModel`, write `.nl` with Pyomo's ASL writer — the same
`.nl` pipeline as `large_scale/generate_nl.py`).

Convention (matches MPS and HiGHS):

    minimize   c' x + offset
    subject to row_lower <= A x <= row_upper,  col_lower <= x <= col_upper

The converter **preserves the MPS objective constant/offset**, so the
emitted objective matches the published optimum (an `.nl` would otherwise
silently drop the constant).

```sh
python3 generate_nl.py                 # fetch + convert all (under the cap)
python3 generate_nl.py --netlib-only   # NETLIB only
python3 generate_nl.py --meszaros-only # Meszaros subset only
python3 generate_nl.py --validate      # also solve each NETLIB instance
                                       #   and compare obj vs published opt
python3 generate_nl.py afiro blend     # only the named instances
```

`nl/afiro.nl` (the smallest instance) is the Make stamp the suite's
generate target depends on.

## Published-optima cross-check (validation)

`generate_nl.py` embeds the published NETLIB optima parsed from the
`lp/data/readme` PROBLEM SUMMARY TABLE. With `--validate` it runs POUNCE
on each converted NETLIB instance and prints the relative error vs the
published value. Result of the full sweep:

- **~85 of 89 NETLIB instances match the published optimum** to
  ≤ 1e-5 relative error (most to 1e-8). Spot checks against the values
  named in the task:

  | instance | published | POUNCE | rel.err |
  |---|---|---|---|
  | `afiro`    | −4.6475314286e+02 | −4.6475314761e+02 | 1.0e−08 |
  | `adlittle` |  2.2549496316e+05 |  2.2549496160e+05 | 6.9e−09 |
  | `blend`    | −3.0812149846e+01 | −3.0812150421e+01 | 1.9e−08 |
  | `share2b`  | −4.1573224074e+02 | −4.1573224775e+02 | 1.7e−08 |
  | `sc50a`    | −6.4575077059e+01 | −6.4575077642e+01 | 9.0e−09 |
  | `stocfor1` | −4.1131976219e+04 | −4.1131976244e+04 | 6.0e−10 |
  | `brandy`   |  1.5185098965e+03 |  1.5185098913e+03 | 3.4e−09 |
  | `bandm`    | −1.5862801845e+02 | −1.5862801946e+02 | 6.4e−09 |

- The few apparent mismatches were investigated and are **not conversion
  bugs**:
  - `e226` — POUNCE returns −11.6389, published is −18.7519. The
    difference is exactly the MPS objective **constant** (offset 7.113):
    −18.7519 + 7.113 = −11.6389. HiGHS solving the same MPS returns the
    identical −11.6389, i.e. POUNCE and HiGHS agree; the readme table
    value omits the constant. The conversion correctly *includes* it.
    `--validate` therefore compares against *published + offset*, with
    the offset read off each MPS at conversion time, so `e226` passes
    (rel.err 1.5e−09) while a conversion that dropped the constant would
    still be flagged (rel.err 0.61 on `e226`). `e226` is the only NETLIB
    instance in the table with a nonzero objective constant.
  - `agg`, `agg2`, `degen3` — POUNCE terminates `Solved To Acceptable
    Level` (not full convergence) on these harder/degenerate LPs, so the
    objective is slightly off; not a data error.
  - `greenbea` — POUNCE hits `Maximum Number of Iterations Exceeded`.

  A non-optimal termination is a legitimate, recorded benchmark outcome.

A separate spot check confirmed several Meszaros instances (`cep1`,
`sc205-2r-4`, `fxm2-6`, `aa01`, `cari`, `air02`, `pgp2`) all reach
`Optimal Solution Found`.

## Reproducibility / layout

- `generate_nl.py` — fetch + (emps-)expand + size-screen + convert driver,
  with embedded NETLIB optima for the `--validate` cross-check.
- `mps_to_nl.py` — the MPS→`.nl` converter (copied from `benchmarks/lpopt/`).
- `nl/*.nl` — the generated suite (regenerated locally; gitignored).
- `mps/` — the compiled `emps` binary + transient MPS (gitignored).
- `data/netlib/`, `data/meszaros/` — the raw downloaded source files
  (gitignored).

Only `generate_nl.py`, `mps_to_nl.py`, and this `README.md` are tracked,
so the suite is fully reproducible from the two upstream sources.

## Running

```sh
# run POUNCE on the suite -> benchmarks/lp/pounce.json
make -C benchmarks lp-run

# refresh the ipopt-ma57 reference for this suite (rare)
make -C benchmarks ipopt-ref-lp
```
