# CBLIB suite — conic (exponential / power cone) tier

The **conic** benchmark tier: instances from the Conic Benchmark Library
(CBLIB, <https://cblib.zib.de>) in Conic Benchmark Format (`.cbf`). Unlike
every other suite here — which is `.nl`-driven through the main `pounce`
NLP binary — these are *conic programs* (geometric programs and power-cone
models) solved through POUNCE's convex conic driver (`pounce-convex`'s
non-symmetric HSDE path) via the dedicated `pounce_cblib` binary.

Each instance is recorded in the same schema as the other suites —
`{solver, name, n, m, status, objective, iterations, solve_time}` — in
`cblib/pounce.json`, so it merges into the composite `BENCHMARK_REPORT.md`.

## What runs

By default the runner solves the small instances **vendored with the
repo** (under `crates/pounce-cli/tests/data/cblib/`), so it works offline:

| Instance | Class | Cones |
|---|---|---|
| `demb761`, `beck751`, `fang88` | geometric programs (Demberg / Beck / Fang) | exponential |
| `pow3_synthetic` | hand-authored power-cone problem | power (`POWCONES`) |

These are also the cross-check tests in
`crates/pounce-cli/tests/cblib_vs_nlp.rs`, where each conic solve is
validated against an **independent** smooth-NLP solve (the two agree on the
objective to ~1e-8). Published CBLIB reference objectives are unavailable
(the solution files 404), so that conic-vs-NLP cross-check *is* the
correctness reference.

## Running

```sh
python3 benchmarks/cblib/run_cblib.py            # vendored instances
python3 benchmarks/cblib/run_cblib.py --detail full   # + per-iteration trace
python3 benchmarks/cblib/run_cblib.py --dir /path/to/cblib   # more instances
```

`--dir` points at a folder of additional `.cbf` files — e.g. a local CBLIB
checkout. The reader supports the cone kinds `F`/`L=`/`L+`/`L-`/`EXP`/`Q`
and the 3-D power cone (`POWCONES` / `@k:POW`); instances using PSD
(`DCOORD`), rotated SOC (`QR`), or dual power cones are skipped with a
clear error. The large power-cone instances (`2013_fir*`, ~120 MB) are not
vendored; fetch them into a `--dir` to include them.

The underlying `pounce_cblib <file.cbf> --json-output <out>` emits a full
`pounce.solve-report/v1` JSON (the same schema the `.nl` path writes, with
an input descriptor of kind `cbf-file`); the runner projects each into the
suite record schema.

## Full corpus + conic-robustness regression watch

Beyond the 5 vendored instances, a 132-instance corpus (exp-cone GPs,
power-cone, SOC families from <https://cblib.zib.de>) lives in the bench-data
tree at `pounce-bench-data/cblib/cbf/` (307 M). Run it with:

```sh
python3 benchmarks/cblib/run_cblib.py \
  --dir "$HOME/Dropbox/projects/pounce-bench-data/cblib/cbf"
```

A stress sweep (60 s/instance, 2026-06-07) over that corpus originally
classified **71 pass · 34 `NumericalFailure` · 10 timeout · 17
unsupported-cone**. The 34 failures seeded a tracked **conic-robustness
regression set** (`MANIFEST.tsv` in the bench-data dir). 27 of them fail *with a
usable objective already in hand*
— several provably match a sibling formulation that passed (`flay02m`==`flay02h`,
`slay04h`==`slay04m`, `clay020{3,4,5}h`==`..m`).

This is **not** an ill-conditioned-input problem: both HSDE drivers can discard
a converged-enough iterate when the KKT factorization degrades near the cone
boundary (`s∘z → 0` ⇒ NT scaling blows up) a hair short of `tol` (1e-8). The
**non-symmetric** driver (`hsde_nonsym.rs`, exp/power cones) already carried an
Ipopt-style "acceptable level" tier — accept the iterate when the
*unregularized* KKT residual is already `< 1e3·tol`. The **symmetric** driver
(`hsde.rs`, SOC/orthant/PSD) did **not**, so it discarded iterates the
non-symmetric one would have kept. Porting that same tier into the symmetric
driver (the principled fix — **not** porting the orthant path's Ruiz
equilibration fallback) recovers **12 of the 34** (all SOC/orthant, with
byte-identical objectives), taking the corpus to **83 pass · 22
`NumericalFailure`**. The remaining 22 are genuine: 9 exp-cone gap-laggards
(would need a composite pres/dres/mu criterion), `slay06h`/`slay06m` (true
divergence), and the `expdesign_D_*` 0-iteration structural failures. Re-run the
corpus after any conic-driver change to track the count.

> Note: the raw solve report renders `QpStatus::NumericalFailure` as
> `InternalError` (`pounce_cblib.rs:33`); classify on the stderr banner, not
> the JSON `status` field.
