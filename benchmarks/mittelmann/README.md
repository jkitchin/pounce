# Mittelmann ampl-nlp benchmark

Harness for running the [Mittelmann ampl-nlp benchmark](https://plato.asu.edu/ftp/ampl-nlp.html)
against pounce and ipopt. 47 medium-to-large NLP instances, sizes 500 to 261k variables.

## Layout

```
mittelmann/
├── Makefile          fetch + translate only (tracked)
├── problems.txt      the 47 problem names (tracked)
├── gen_robot_nl.py   robot_a/b/c straight to .nl, no AMPL needed (tracked)
├── ipopt_ma57.json   saved Ipopt-MA57 reference (tracked)
├── source/           .mod files from plato.asu.edu (untracked)
├── nl/               .nl translations (untracked; see NLDIR below)
└── pounce.json       per-release POUNCE results (untracked)
```

Running the solvers and building the report are not done here. Like every
suite, Mittelmann runs through `benchmarks/scripts/run_nl_bench.sh` and is
reported by `benchmarks/benchmark_report.py`.

## Running it

From the repository root:

```
make -C benchmarks mittelmann-run           # POUNCE, part of the release sweep
make -C benchmarks ipopt-ref-mittelmann     # refresh the saved Ipopt reference
```

Both first run `make -C benchmarks/mittelmann fetch translate NLDIR=<dir>`, where
`<dir>` is `$POUNCE_BENCH_DATA/mittelmann/nl` (or `benchmarks/mittelmann/nl`).
Both steps skip every problem whose `.nl` is already there, so with a translated
set in place no network and no AMPL are needed. Only a missing `.nl` is fetched
from plato.asu.edu and translated.

## Prerequisites for translating

Only needed when a `.nl` is missing:

1. **AMPL Community Edition** at `.venv-ampl/` in the repository root, or pass
   `AMPL=<path to the ampl binary>`:
   ```
   uv venv .venv-ampl --python 3.12
   uv pip install --python .venv-ampl/bin/python amplpy
   .venv-ampl/bin/python -m amplpy.modules install ampl
   .venv-ampl/bin/python -m amplpy.modules activate <CE-UUID>
   ```
   Register for a UUID at https://ampl.com/ce. **The activation step matters.**
   Without it AMPL runs as a demo, limited to 300 variables and 300
   constraints. Almost every problem here is larger, so translation fails
   with "a demo license for AMPL is limited to 300 variables".

2. For the Ipopt reference, an Ipopt built with CoinHSL and the AMPL interface.
   Point the harness at it with `IPOPT_MA57_BIN=<path>/bin/ipopt`. No build
   recipe is tracked, so `make build-ipopt-ma57` just says so.

## `robot_a` / `robot_b` / `robot_c` without AMPL

`gen_robot_nl.py` writes those three instances straight to `.nl`, so the
per-iteration cost work in pounce#476 is reproducible with no AMPL licence and
no `make translate`:

```
curl -O http://plato.asu.edu/ftp/ampl-nlp-source/robot_a.mod
python3 gen_robot_nl.py robot_a.mod robot_a.nl     # n=1001 m=52013 nzJ=196781
pounce robot_a.nl max_iter=200 --no-sol
ipopt  robot_a -AMPL                               # same file, for comparison
```

It emits the `V` segments for the model's `SUM`/`SUM1`/`SUM2` defined variables
rather than inlining them, so the tape the solver builds matches what AMPL would
hand it — which matters here, because those 12003 shared subexpressions *are* the
performance story (see `dev-notes/research/robot-abc-per-iteration-cost.md`).
Sizes match the published `n`/`m` exactly; see the script header for the one
AMPL presolve step (six duplicate linear rows per collocation point collapsing to
one) that it reproduces by hand. `--no-merge` emits the unpresolved 18-family
form for comparison.

## Licensing & redistribution

The .mod sources are fetched from Mittelmann's public mirror and **not redistributed
in this repo** (gitignored). The .nl translations are AMPL artifacts and also gitignored.
Only the harness (`Makefile`, `problems.txt`, `gen_robot_nl.py`) and the saved
reference results are in git.

The bundled AMPL CE license is non-commercial and tied to your registration; it
is not redistributable.
