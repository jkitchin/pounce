# Mittelmann ampl-nlp benchmark

Harness for running the [Mittelmann ampl-nlp benchmark](https://plato.asu.edu/ftp/ampl-nlp.html)
against pounce and ipopt. 47 medium-to-large NLP instances, sizes 500 to 261k variables.

## Layout

```
mittelmann/
├── Makefile          orchestration (fetch / translate / run / report)
├── problems.txt      the 47 problem names
├── run_solver.sh     per-instance solve wrapper
├── make_report.py    SGM table generator
├── profiles/         per-problem overrides (env files); see profiles/README.md
├── notes/            per-problem diagnostic write-ups
├── source/           cached .mod files from plato.asu.edu (gitignored)
├── nl/               cached .nl files produced by ampl (gitignored)
├── logs/             per-instance solver stdout (gitignored)
├── results/          per-version JSON results (gitignored)
└── reports/          per-version markdown reports (checked in)
```

## Prerequisites

1. **AMPL Community Edition** installed at `../.venv-ampl/`. To install:
   ```
   uv venv ../.venv-ampl --python 3.12
   ../.venv-ampl/bin/python -m ensurepip
   uv pip install --python ../.venv-ampl/bin/python amplpy
   ../.venv-ampl/bin/python -m amplpy.modules install ampl
   ../.venv-ampl/bin/python -m amplpy.modules activate <CE-UUID>
   ```
   Register for a UUID at https://ampl.com/ce.

2. **ipopt** binary on PATH (e.g. `brew install ipopt`).

3. **pounce** built: `cargo build --release --bin pounce` (run from repo root).

## Usage

```
make fetch              # download .mod files from plato (cached, ~1 min)
make translate          # ampl: .mod -> .nl (cached, ~30 s)
make run-pounce         # run pounce on every .nl, write results/pounce_<version>.json
make run-ipopt-feral    # run ipopt (FERAL) on every .nl  (also: run-ipopt-mumps, run-ipopt-ma57)
make run-all            # run every solver variant
make report             # regenerate reports/BENCHMARK_REPORT_<version>.md

make all         # fetch + translate + run-all + report
```

Default per-instance timeout is 7200s, matching Mittelmann's published convention.
Override with `make run-pounce TIMELIMIT=300`.

## Licensing & redistribution

The .mod sources are fetched from Mittelmann's public mirror and **not redistributed
in this repo** (gitignored). The .nl translations are AMPL artifacts and also gitignored.
Only the harness files and the markdown reports are in git.

The bundled AMPL CE license is non-commercial and tied to your registration; it
is not redistributable.
