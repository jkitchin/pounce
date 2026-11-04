# Electrical Grid Suite (AC Optimal Power Flow)

AC Optimal Power Flow (AC OPF) is the canonical nonconvex NLP of the power
systems community: minimise generation cost subject to nonlinear AC power
flow equations, voltage limits, line thermal limits, and generator output
bounds. Problems are PGLib-OPF cases (Babaeinejadsarookolaee et al., "The
Power Grid Library for Benchmarking AC Optimal Power Flow Algorithms",
arXiv:1908.02788; data CC BY 4.0, v23.07), in the MATPOWER case format
(Zimmerman, Murillo-Sánchez, Thomas, IEEE Trans. Power Syst. 2011).
`case3_lmbd` and `case5_pjm` exist only in PGLib, not in MATPOWER's own
case library. Cite PGLib's version and each file header's source, as PGLib
asks; `$POUNCE_BENCH_DATA/SOURCES.md` has the record. POUNCE uses the polar
form with explicit bus voltage magnitude and angle variables.

The 22 problems range from `case3_lmbd` (3 buses, 12 variables) to
`pglib_case10000_goc` (10 000 buses). These are good
stress tests for dense KKT paths because the constraint Jacobian is
moderately dense and the Hessian has many non-convex blocks from the
voltage–angle coupling.

This suite lives under `benchmarks/grid/`. See `benchmarks/README.md` for
an overview of all suites. Like the other suites, problems are solved
through the AMPL `.nl` interface by the shared NL driver
(`benchmarks/scripts/run_nl_bench.sh`), which runs both POUNCE and Ipopt
(linked against MA57) and feeds the composite `benchmarks/BENCHMARK_REPORT.md`.

## Contents

- `nl/` (data root) — the AMPL `.nl` files solved by the harness, one per case
- `grid_nl_export.py` — the generator that wrote them; it is not in this
  repository (nor in its predecessors') and is recorded as lost
- `pounce.json` — latest POUNCE per-problem results
- `ipopt_ma57.json` — Ipopt/MA57 reference results
- `grid_benchmark_report.md` — per-problem analysis
- `problems.rs` — **retired** pure-Rust `NlpProblem` definitions, kept for
  reference; no longer compiled (the suite is now `.nl`-driven)

## How to run

From the repo root:

```bash
make -C benchmarks grid-run     # writes benchmarks/grid/pounce.json
make -C benchmarks grid-rerun   # force a fresh run
```

Or solve a single case directly:

```bash
pounce benchmarks/grid/nl/case30_ieee.nl print_level=5
```

## Re-exporting

If the MATPOWER cases or export logic change, regenerate the `.nl` files:

```bash
python benchmarks/grid/grid_nl_export.py
```
