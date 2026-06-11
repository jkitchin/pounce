# POUNCE Benchmark Report

Generated: 2026-06-04 19:15:02

## Provenance

| Component | Version / Detail |
|-----------|------------------|
| POUNCE | v0.3.1-dev (main @ 37133a7-dirty) |
| POUNCE linear solver | feral (default) |
| Ipopt | Ipopt 3.14.20 (Darwin arm64), ASL(20241202) |
| Ipopt linear solver | ma57 (via ref/Ipopt/install-ma57) |
| Platform | Darwin 25.5.0 arm64 |

POUNCE results were produced this run by `make -C benchmarks
<suite>-run` (pounce only). The Ipopt column is a saved reference
(`make -C benchmarks ipopt-reference`), rerun only when explicitly
regenerated — generated 2026-06-04 19:15:02 EDT on Johns-Mac-mini.local (Darwin 25.5.0 arm64), git 37133a7, timelimit 300s. Ipopt solve *times* are
from that reference machine and only comparable to POUNCE when this
report is generated on the same host.

The GAMS solver-link path is exercised separately as a liveness
smoke check (`make -C benchmarks gams-bench`) and is not aggregated here.

> **Threading & timing.** The reference and POUNCE runs are pinned to a
> single compute thread (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
> `VECLIB_MAXIMUM_THREADS`, `RAYON_NUM_THREADS` all = 1) and run
> sequentially so pounce and Ipopt solve times are directly comparable
> on one host.
> POUNCE's dense linear algebra (via `faer`/`rayon`) parallelizes across
> cores, so its *multi-threaded* wall-clock is up to ~2x faster on the
> larger dense problems (e.g. Mittelmann `cont*`/`qcqp*`, QP); the
> single-threaded times reported here are therefore a controlled lower
> bound, not pounce's real-world speed, and should not be compared
> against multi-threaded runs of this report.

## Executive Summary

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Optimal (strict) | **1271/1326** (95.9%) | **1233/1326** (93.0%) |
| Acceptable (informational, *not* counted as solved) | 15 | 25 |
| Solved exclusively (strict Optimal) | 44 | 6 |
| Both Optimal | 1227 | |
| Matching objectives (< 0.01%) | 1216/1227 | |

> **Note:** All headline counts use strict Optimal status only. `Acceptable`
> means the iterate met relaxed tolerances but not the requested tolerance —
> per CLAUDE.md's "Honesty in Benchmarks" rule it is reported separately and
> never folded into the pass rate. See the "Acceptable (not Optimal)" and
> "Different Local Minima" sections below.

## Performance Profiles

[Dolan & Moré (2002)](https://doi.org/10.1007/s101070100263) performance profiles pooled over every suite with an Ipopt reference. ρ_s(τ) is the fraction of problems a solver solves within a factor τ of the fastest solver on each problem: the **height at τ=1** is how often it was the quickest, and the **right-hand plateau** is its overall robustness (fraction solved at all). A problem counts as solved only at strict/acceptable success; failures and timeouts are charged infinite cost. Regenerate or slice these with `python3 scripts/perf_profile.py <suite…> [--metric iters] [--mode data]`.

![**Performance profile by wall-clock time.** Valid because POUNCE and Ipopt-MA57 were run interleaved on this host (see Provenance).](figures/profile_performance_time.png)

**Performance profile by wall-clock time.** Valid because POUNCE and Ipopt-MA57 were run interleaved on this host (see Provenance).
  
_1287 problems; solvers: pounce, ipopt._

![**Performance profile by iteration count** — machine-independent, so it stays comparable across hosts and reruns.](figures/profile_performance_iters.png)

**Performance profile by iteration count** — machine-independent, so it stays comparable across hosts and reruns.
  
_1287 problems; solvers: pounce, ipopt._

![**Data profile (absolute-time ECDF).** Fraction of problems solved within a given wall-clock budget, without best-solver normalization — reads directly as “how many by 1 s? by 10 s?”.](figures/profile_data_time.png)

**Data profile (absolute-time ECDF).** Fraction of problems solved within a given wall-clock budget, without best-solver normalization — reads directly as “how many by 1 s? by 10 s?”.
  
_1326 problems; solvers: pounce, ipopt._

## Per-Suite Summary

| Suite | Problems | POUNCE Optimal | Ipopt Optimal | POUNCE only | Ipopt only | Both Optimal | Match |
|-------|----------|---------------|--------------|-------------|------------|--------------|-------|
| Vanderbei | 733 | 703 (95.9%) | 684 (93.3%) | 21 | 2 | 682 | 673/682 |
| Electrolyte | 13 | 13 (100.0%) | 13 (100.0%) | 0 | 0 | 13 | 13/13 |
| Grid | 4 | 4 (100.0%) | 4 (100.0%) | 0 | 0 | 4 | 4/4 |
| CHO | 1 | 1 (100.0%) | 1 (100.0%) | 0 | 0 | 1 | 1/1 |
| Water | 6 | 6 (100.0%) | 6 (100.0%) | 0 | 0 | 6 | 4/6 |
| Gas | 4 | 3 (75.0%) | 3 (75.0%) | 0 | 0 | 3 | 3/3 |
| LargeScale | 5 | 5 (100.0%) | 5 (100.0%) | 0 | 0 | 5 | 5/5 |
| Mittelmann | 47 | 41 (87.2%) | 33 (70.2%) | 9 | 1 | 32 | 32/32 |
| QP | 138 | 136 (98.6%) | 132 (95.7%) | 4 | 0 | 132 | 132/132 |
| LP | 371 | 358 (96.5%) | 352 (94.9%) | 9 | 3 | 349 | 349/349 |
| LPopt | 4 | 1 (25.0%) | 0 (0.0%) | 1 | 0 | 0 | 0/1 |

## Vanderbei Reference Cross-Check

Per-problem status from R. Vanderbei's `cute_table.pdf` (`vanderbei/cute_table_status.json`). The meaningful denominator is the **expected-solvable** set — problems with a documented finite optimum — not all 733: the CUTE collection deliberately includes unbounded, infeasible, and no-solver-finishes problems.

| cute_table status | problems | POUNCE solved | meaning |
|---|---|---|---|
| optimum | 684 | 668 | finite reference optimum exists (expected-solvable) |
| hard | 14 | 8 | in table, but SNOPT+NITRO+LOQO all hit time/iter limits |
| infeasible | 3 | 0 | a reference solver declared infeasibility |
| unbounded | 1 | 0 | unbounded below |
| untabulated | 31 | 27 | not in cute_table — no reference datum |

**POUNCE solved 668 / 684 expected-solvable (97.7%).** The hard / infeasible / unbounded / untabulated rows above are excluded from this denominator — a POUNCE failure there is shared with the commercial reference solvers and is not counted as a miss.

**Genuine misses — expected-solvable but POUNCE did not reach Optimal (16):**

> brainpc0 britgas coshfun cresc100 cresc132 cresc50 deconvb flosp2hh grouping himmelbj nonmsqrt orthrds2 palmer5e sineali ssebnln steenbrc

**Objective disagreements vs. cute_table reference (133)** — POUNCE converged but to a different value than the agreed reference optimum (possible wrong basin or misread problem):

| Problem | POUNCE obj | reference obj | rel. diff |
|---|---|---|---|
| broydn7d | 3.450050e+02 | 3.823419e+00 | 8.9e+01 |
| liswet9 | 1.899426e+03 | 2.499976e+01 | 7.5e+01 |
| liswet8 | 6.509716e+02 | 2.499977e+01 | 2.5e+01 |
| liswet7 | 3.911520e+02 | 2.499979e+01 | 1.5e+01 |
| eigenbco | 5.617567e-26 | 9.000000e+00 | 1.0e+00 |
| scurly10 | -1.003163e-02 | -1.003163e+06 | 1.0e+00 |
| scurly20 | -1.003163e-02 | -1.003163e+06 | 1.0e+00 |
| scurly30 | -1.003163e-02 | -1.003163e+06 | 1.0e+00 |
| meyer3 | 8.794586e-07 | 8.794586e+01 | 1.0e+00 |
| djtl | -8.951545e-05 | -8.951545e+03 | 1.0e+00 |
| hs099 | -3.456347e+02 | -8.310799e+08 | 1.0e+00 |
| spanhyd | 5.193068e-04 | 2.397380e+02 | 1.0e+00 |
| himmelbd | 6.699286e-05 | 5.922563e+00 | 1.0e+00 |
| arglinb | 1.469851e-04 | 4.634146e+00 | 1.0e+00 |
| hs084 | -2.872406e+02 | -5.280335e+06 | 1.0e+00 |
| brownden | 4.823391e+00 | 8.582220e+04 | 1.0e+00 |
| arglinc | 3.679793e-04 | 6.135135e+00 | 1.0e+00 |
| lakes | 6.298140e+01 | 3.505248e+05 | 1.0e+00 |
| optctrl3 | 5.120041e-01 | 2.048017e+03 | 1.0e+00 |
| optctrl6 | 5.120041e-01 | 2.048017e+03 | 1.0e+00 |
| bdqrtic | 1.333272e+00 | 3.983818e+03 | 1.0e+00 |
| penalty2 | 3.848429e+01 | 9.709608e+04 | 1.0e+00 |
| flosp2hm | 1.598325e-02 | 3.887126e+01 | 1.0e+00 |
| flosp2th | 4.111842e-03 | 1.000000e+01 | 1.0e+00 |
| flosp2tm | 4.111842e-03 | 1.000000e+01 | 1.0e+00 |
| flosp2tl | 4.111842e-03 | 1.000000e+01 | 1.0e+00 |
| flosp2hl | 1.598295e-02 | 3.887054e+01 | 1.0e+00 |
| bratu1d | -4.242482e-03 | -8.518927e+00 | 1.0e+00 |
| dallasl | -1.048696e+02 | -2.026041e+05 | 1.0e+00 |
| hs064 | 4.375194e+00 | 6.299842e+03 | 1.0e+00 |
| scosine | -8.553258e+00 | -9.999000e+03 | 1.0e+00 |
| dallasm | -5.386822e+01 | -4.819819e+04 | 1.0e+00 |
| jensmp | 1.422873e-01 | 1.243622e+02 | 1.0e+00 |
| errinros | 4.833189e-02 | 3.990415e+01 | 1.0e+00 |
| dallass | -4.053642e+01 | -3.239323e+04 | 1.0e+00 |
| cvxbqp1 | 4.286142e+03 | 2.250225e+06 | 1.0e+00 |
| ncvxbqp1 | -3.781988e+07 | -1.985544e+10 | 1.0e+00 |
| cvxqp2 | 1.558904e+05 | 8.184246e+07 | 1.0e+00 |
| cvxqp3 | 2.204021e+05 | 1.157111e+08 | 1.0e+00 |
| palmer7e | 2.398370e-02 | 1.015390e+01 | 1.0e+00 |
| lch | -1.649122e-02 | -4.318289e+00 | 1.0e+00 |
| avion2 | 4.115108e+05 | 9.468017e+07 | 1.0e+00 |
| chainwoo | 2.788601e-01 | 6.362471e+01 | 1.0e+00 |
| hs119 | 1.313276e+00 | 2.448997e+02 | 9.9e-01 |
| palmer1b | 2.409219e-02 | 3.447355e+00 | 9.9e-01 |
| 3pk | 1.677022e-02 | 1.720119e+00 | 9.9e-01 |
| hs062 | -2.624873e+02 | -2.627251e+04 | 9.9e-01 |
| himmelbf | 3.640806e+00 | 3.185717e+02 | 9.9e-01 |
| model | 7.764926e+01 | 5.742163e+03 | 9.9e-01 |
| cragglvy | 2.988096e+01 | 1.688215e+03 | 9.8e-01 |
| steenbre | 5.197370e+02 | 2.745916e+04 | 9.8e-01 |
| ncvxqp4 | -1.789977e+06 | -9.397879e+07 | 9.8e-01 |
| cvxqp1 | 2.071451e+04 | 1.087512e+06 | 9.8e-01 |
| hs107 | 1.027441e+02 | 5.055012e+03 | 9.8e-01 |
| ncvxqp2 | -1.541020e+06 | -5.781269e+07 | 9.7e-01 |
| steenbrd | 2.614067e+02 | 9.030082e+03 | 9.7e-01 |
| steenbrb | 2.627318e+02 | 9.075855e+03 | 9.7e-01 |
| palmer4b | 2.097297e-01 | 6.835139e+00 | 9.7e-01 |
| palmer3b | 1.373521e-01 | 4.227647e+00 | 9.7e-01 |
| orthregd | 1.523900e+03 | 4.245801e+04 | 9.6e-01 |
| palmer1 | 4.262081e+02 | 1.175460e+04 | 9.6e-01 |
| lsnnodoc | 4.808004e+00 | 1.231124e+02 | 9.6e-01 |
| hs020 | 1.670770e+00 | 4.019873e+01 | 9.6e-01 |
| hs017 | 4.156277e-02 | 1.000000e+00 | 9.6e-01 |
| eigenc2 | 4.011484e+01 | 7.718095e+02 | 9.5e-01 |
| orthrgds | 1.523900e+03 | 2.603509e+04 | 9.4e-01 |
| palmer5d | 5.193696e+00 | 8.733940e+01 | 9.4e-01 |
| sseblin | 1.078040e+06 | 1.617060e+07 | 9.3e-01 |
| palmer2 | 2.646634e+02 | 3.651090e+03 | 9.3e-01 |
| freuroth | 4.458645e+04 | 6.081592e+05 | 9.3e-01 |
| bt4 | -3.704768e+00 | -4.551055e+01 | 9.2e-01 |
| explin | -6.031302e+04 | -7.237563e+05 | 9.2e-01 |
| expquad | -3.020500e+05 | -3.624600e+06 | 9.2e-01 |
| explin2 | -6.037160e+04 | -7.244591e+05 | 9.2e-01 |
| qrtquad | -3.065620e+05 | -3.648088e+06 | 9.2e-01 |
| powell20 | 5.211962e+06 | 5.214578e+07 | 9.0e-01 |
| palmer5c | 2.632140e-01 | 2.128087e+00 | 8.8e-01 |
| hs103 | 7.378044e+01 | 5.436680e+02 | 8.6e-01 |
| hs102 | 1.239180e+02 | 9.118806e+02 | 8.6e-01 |
| hs101 | 2.461046e+02 | 1.809765e+03 | 8.6e-01 |
| hs019 | -1.157377e+03 | -6.961814e+03 | 8.3e-01 |
| camel6 | -2.154638e-01 | -1.031628e+00 | 7.9e-01 |
| catena | -5.175437e+03 | -2.307775e+04 | 7.8e-01 |
| hong | 3.964576e-01 | 1.347307e+00 | 7.1e-01 |
| fccu | 3.716370e+00 | 1.114911e+01 | 6.7e-01 |
| hs083 | -1.059902e+04 | -3.066554e+04 | 6.5e-01 |
| palmer1d | 1.550473e-06 | 6.526826e-01 | 6.5e-01 |
| palmer2b | 1.952286e-02 | 6.233947e-01 | 6.0e-01 |
| palmer7c | 1.385267e-05 | 6.019857e-01 | 6.0e-01 |
| batch | 1.036721e+05 | 2.591804e+05 | 6.0e-01 |
| gpp | 5.783505e+03 | 1.440093e+04 | 6.0e-01 |
| liswet10 | 3.930236e+01 | 2.499967e+01 | 5.7e-01 |
| harkerp2 | -3.697727e-05 | -5.000000e-01 | 5.0e-01 |
| hs114 | -9.211388e+02 | -1.768807e+03 | 4.8e-01 |
| hs105 | 5.918833e+02 | 1.136361e+03 | 4.8e-01 |
| eigena2 | 4.583333e+01 | 8.250000e+01 | 4.4e-01 |
| fletcher | 1.165685e+01 | 1.952537e+01 | 4.0e-01 |
| eigenb2 | 1.000000e+00 | 1.600000e+00 | 3.7e-01 |
| dixmaand | 6.503642e-01 | 1.000000e+00 | 3.5e-01 |
| dixmaanh | 6.560532e-01 | 1.000000e+00 | 3.4e-01 |
| dixmaanl | 6.599015e-01 | 1.000000e+00 | 3.4e-01 |
| liswet12 | -3.379107e+03 | -5.026353e+03 | 3.3e-01 |
| smmpsf | 7.321578e+05 | 1.046985e+06 | 3.0e-01 |
| discs | 1.444952e+01 | 1.200008e+01 | 2.0e-01 |
| cliff | 2.135747e-09 | 1.997866e-01 | 2.0e-01 |
| engval1 | 4.474733e+03 | 5.548668e+03 | 1.9e-01 |
| hs093 | 1.102206e+02 | 1.350760e+02 | 1.8e-01 |
| hairy | 1.639999e+01 | 2.000000e+01 | 1.8e-01 |
| qudlin | -6.000000e+03 | -7.200000e+03 | 1.7e-01 |
| palmer8c | 1.415888e-05 | 1.597681e-01 | 1.6e-01 |
| hs044 | -1.300000e+01 | -1.500000e+01 | 1.3e-01 |
| hs113 | 2.170197e+01 | 2.430621e+01 | 1.1e-01 |
| palmer1c | 1.984316e-08 | 9.759799e-02 | 9.8e-02 |
| palmer1a | 4.213860e-04 | 8.988363e-02 | 8.9e-02 |
| avgasb | -4.483219e+00 | -4.132819e+00 | 8.5e-02 |
| liswet1 | 2.712029e+01 | 2.500304e+01 | 8.5e-02 |
| mwright | 2.312853e+01 | 2.497881e+01 | 7.4e-02 |
| palmer8a | 1.927299e-03 | 7.400970e-02 | 7.2e-02 |
| palmer6a | 2.800103e-03 | 5.594884e-02 | 5.3e-02 |
| palmer4c | 4.753927e-07 | 5.031070e-02 | 5.0e-02 |
| palmer4a | 7.908980e-04 | 4.060614e-02 | 4.0e-02 |
| expfitc | 1.159858e-04 | 2.330257e-02 | 2.3e-02 |
| palmer3a | 4.343676e-04 | 2.043142e-02 | 2.0e-02 |
| palmer3c | 1.843055e-07 | 1.953764e-02 | 2.0e-02 |
| palmer2a | 3.838581e-04 | 1.716074e-02 | 1.7e-02 |
| palmer6c | 1.644281e-06 | 1.638742e-02 | 1.6e-02 |
| palmer2c | 3.935677e-08 | 1.442139e-02 | 1.4e-02 |
| palmer5b | 6.582649e-06 | 9.752493e-03 | 9.7e-03 |
| penalty1 | 6.439498e-08 | 9.686175e-03 | 9.7e-03 |
| palmer8e | 5.501854e-05 | 6.339307e-03 | 6.3e-03 |
| twirism1 | -1.001289e+00 | -1.006758e+00 | 5.4e-03 |
| expfitb | 1.249092e-04 | 5.019366e-03 | 4.9e-03 |
| trainh | 1.231200e+01 | 1.236996e+01 | 4.7e-03 |

## Vanderbei Suite — Performance

On 682 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 37.6ms | 37.8ms |
| Total time | 350.83s | 252.00s |
| Mean iterations | 50.9 | 47.2 |
| Median iterations | 16 | 16 |

- **Geometric mean speedup**: 0.8x
- **Median speedup**: 1.0x
- POUNCE faster: 332/682 (49%)
- POUNCE 10x+ faster: 0/682
- Ipopt faster: 350/682

## Electrolyte Suite — Performance

On 13 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 30.3ms | 35.9ms |
| Total time | 412.2ms | 476.5ms |
| Mean iterations | 12.5 | 12.2 |
| Median iterations | 10 | 10 |

- **Geometric mean speedup**: 1.2x
- **Median speedup**: 1.2x
- POUNCE faster: 12/13 (92%)
- POUNCE 10x+ faster: 0/13
- Ipopt faster: 1/13

## Grid Suite — Performance

On 4 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 38.5ms | 39.4ms |
| Total time | 321.7ms | 179.1ms |
| Mean iterations | 15.5 | 15.5 |
| Median iterations | 17 | 17 |

- **Geometric mean speedup**: 0.8x
- **Median speedup**: 1.2x
- POUNCE faster: 3/4 (75%)
- POUNCE 10x+ faster: 0/4
- Ipopt faster: 1/4

## CHO Suite — Performance

On 1 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 6.54s | 1.77s |
| Total time | 6.54s | 1.77s |
| Mean iterations | 43.0 | 33.0 |
| Median iterations | 43 | 33 |

- **Geometric mean speedup**: 0.3x
- **Median speedup**: 0.3x
- POUNCE faster: 0/1 (0%)
- POUNCE 10x+ faster: 0/1
- Ipopt faster: 1/1

## Water Suite — Performance

On 6 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 111.8ms | 78.1ms |
| Total time | 724.6ms | 499.2ms |
| Mean iterations | 209.3 | 205.2 |
| Median iterations | 183 | 209 |

- **Geometric mean speedup**: 0.7x
- **Median speedup**: 0.6x
- POUNCE faster: 1/6 (17%)
- POUNCE 10x+ faster: 0/6
- Ipopt faster: 5/6

## Gas Suite — Performance

On 3 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 92.4ms | 67.1ms |
| Total time | 301.7ms | 221.3ms |
| Mean iterations | 39.7 | 39.7 |
| Median iterations | 20 | 20 |

- **Geometric mean speedup**: 0.8x
- **Median speedup**: 0.7x
- POUNCE faster: 1/3 (33%)
- POUNCE 10x+ faster: 0/3
- Ipopt faster: 2/3

## LargeScale Suite — Performance

On 5 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 631.4ms | 518.2ms |
| Total time | 5.00s | 3.13s |
| Mean iterations | 306.6 | 305.6 |
| Median iterations | 1 | 2 |

- **Geometric mean speedup**: 0.8x
- **Median speedup**: 0.9x
- POUNCE faster: 2/5 (40%)
- POUNCE 10x+ faster: 0/5
- Ipopt faster: 3/5

## Mittelmann Suite — Performance

On 32 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 6.53s | 3.05s |
| Total time | 931.38s | 754.83s |
| Mean iterations | 142.9 | 86.9 |
| Median iterations | 34 | 34 |

- **Geometric mean speedup**: 0.6x
- **Median speedup**: 0.6x
- POUNCE faster: 11/32 (34%)
- POUNCE 10x+ faster: 0/32
- Ipopt faster: 21/32

## QP Suite — Performance

On 132 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 118.1ms | 103.7ms |
| Total time | 112.81s | 120.58s |
| Mean iterations | 76.0 | 76.1 |
| Median iterations | 24 | 24 |

- **Geometric mean speedup**: 0.9x
- **Median speedup**: 0.9x
- POUNCE faster: 61/132 (46%)
- POUNCE 10x+ faster: 0/132
- Ipopt faster: 71/132

## LP Suite — Performance

On 349 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 211.1ms | 153.1ms |
| Total time | 1116.35s | 390.90s |
| Mean iterations | 109.7 | 106.9 |
| Median iterations | 56 | 56 |

- **Geometric mean speedup**: 0.7x
- **Median speedup**: 0.7x
- POUNCE faster: 56/349 (16%)
- POUNCE 10x+ faster: 1/349
- Ipopt faster: 293/349

## Failure Analysis

### Vanderbei Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 4 | 6 |
| Infeasible_Problem_Detected | 4 | 4 |
| Invalid_Number_Detected | 1 | 3 |
| Maximum_CpuTime_Exceeded | 2 | 7 |
| Maximum_Iterations_Exceeded | 13 | 16 |
| Restoration_Failed | 0 | 3 |
| Search_Direction_Becomes_Too_Small | 1 | 1 |
| Solver_Error | 5 | 2 |
| Unknown_Error | 0 | 7 |

### Gas Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Infeasible_Problem_Detected | 1 | 1 |

### Mittelmann Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 1 | 0 |
| Maximum_CpuTime_Exceeded | 5 | 10 |
| Maximum_Iterations_Exceeded | 0 | 3 |
| Solver_Error | 0 | 1 |

### QP Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 1 | 4 |
| Maximum_CpuTime_Exceeded | 1 | 2 |

### LP Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 9 | 15 |
| Infeasible_Problem_Detected | 1 | 1 |
| Maximum_Iterations_Exceeded | 1 | 1 |
| Restoration_Failed | 0 | 1 |
| Solver_Error | 2 | 0 |
| Unknown_Error | 0 | 1 |

### LPopt Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Maximum_CpuTime_Exceeded | 3 | 4 |

## Regressions (Ipopt Optimal, POUNCE not Optimal)

| Problem | Suite | n | m | POUNCE status | Ipopt obj |
|---------|-------|---|---|--------------|-----------|
| model3 | LP | 3840 | 1609 | Acceptable | 1.750657e+03 |
| model7 | LP | 8007 | 3358 | Acceptable | 4.942951e+03 |
| orthrds2 | Vanderbei | 203 | 100 | Acceptable | 1.544297e+03 |
| p0201 | LP | 201 | 133 | Solver_Error | 7.161458e+01 |
| qcqp1000-1nc | Mittelmann | 1000 | 154 | Acceptable | -1.841514e+06 |
| ssebnln | Vanderbei | 194 | 96 | Acceptable | 1.078040e+06 |

## Wins (POUNCE Optimal, Ipopt not Optimal) — 44 problems

| Problem | Suite | n | m | Ipopt status | POUNCE obj |
|---------|-------|---|---|-------------|------------|
| BOYD1 | QP | 93261 | 18 | Acceptable | -6.415515e+05 |
| CVXQP3_L | QP | 10000 | 7500 | Maximum_CpuTime_Exceeded | 1.102011e+06 |
| NARX_CFy | Mittelmann | 43973 | 48256 | Maximum_CpuTime_Exceeded | 8.581943e-03 |
| QPILOTNO | QP | 2172 | 975 | Acceptable | 6.433450e+05 |
| QSCORPIO | QP | 358 | 388 | Acceptable | 9.357630e+02 |
| brainpc1 | Vanderbei | 6905 | 6900 | Restoration_Failed | 4.371251e-04 |
| brainpc2 | Vanderbei | 13805 | 13800 | Maximum_CpuTime_Exceeded | 4.330683e-04 |
| brainpc5 | Vanderbei | 6905 | 6900 | Maximum_CpuTime_Exceeded | 3.750403e-04 |
| brainpc7 | Vanderbei | 6905 | 6900 | Maximum_CpuTime_Exceeded | 3.927065e-04 |
| bt8 | Vanderbei | 5 | 2 | Acceptable | 1.000000e+00 |
| complex | LP | 1408 | 1023 | Acceptable | -9.966667e+01 |
| coolhans | Vanderbei | 9 | 0 | Unknown_Error | 0.000000e+00 |
| csfi2 | Vanderbei | 5 | 4 | Acceptable | 5.501760e+01 |
| dallasl | Vanderbei | 906 | 667 | Invalid_Number_Detected | -1.048696e+02 |
| dallasm | Vanderbei | 196 | 151 | Invalid_Number_Detected | -5.386822e+01 |
| dallass | Vanderbei | 46 | 31 | Invalid_Number_Detected | -4.053642e+01 |
| de063155 | LP | 1488 | 852 | Restoration_Failed | 9.883094e+08 |
| dirichlet120 | Mittelmann | 53881 | 241 | Maximum_CpuTime_Exceeded | 3.503609e-02 |
| drcav2lq | Vanderbei | 10816 | 816 | Maximum_CpuTime_Exceeded | 1.555870e-03 |
| drcavty2 | Vanderbei | 10816 | 816 | Maximum_CpuTime_Exceeded | 1.555870e-03 |
| eigenc2 | Vanderbei | 462 | 231 | Unknown_Error | 4.011484e+01 |
| elec_400 | Mittelmann | 1200 | 400 | Maximum_CpuTime_Exceeded | 1.001292e+02 |
| finnis | LP | 614 | 497 | Acceptable | 4.288360e+03 |
| flosp2th | Vanderbei | 691 | 0 | Maximum_Iterations_Exceeded | 4.111842e-03 |
| greenbeb | LP | 5405 | 2389 | Acceptable | -4.302260e+06 |
| henon120 | Mittelmann | 32401 | 241 | Maximum_CpuTime_Exceeded | 1.332947e+02 |
| lane_emden120 | Mittelmann | 57721 | 241 | Maximum_CpuTime_Exceeded | 9.340251e+00 |
| manne | Vanderbei | 1094 | 730 | Acceptable | -9.741479e-01 |
| maros | LP | 1443 | 845 | Acceptable | -4.720630e+04 |
| nql180 | Mittelmann | 129601 | 130080 | Solver_Error | -9.277211e-01 |
| palmer7e | Vanderbei | 8 | 0 | Maximum_Iterations_Exceeded | 2.398370e-02 |
| pilotnov | LP | 2172 | 975 | Acceptable | -4.497276e+03 |
| polak3 | Vanderbei | 12 | 10 | Maximum_Iterations_Exceeded | 5.933003e+00 |
| polak6 | Vanderbei | 5 | 4 | Unknown_Error | -4.400000e+01 |
| qap15 | LPopt | 22275 | 6330 | Maximum_CpuTime_Exceeded | 8.674941e+02 |
| qcqp1000-2c | Mittelmann | 1000 | 5107 | Maximum_CpuTime_Exceeded | 1.112949e+04 |
| qcqp1500-1c | Mittelmann | 1500 | 10508 | Maximum_CpuTime_Exceeded | 1.751829e+04 |
| qcqp1500-1nc | Mittelmann | 1500 | 10508 | Maximum_CpuTime_Exceeded | 1.537714e+04 |
| scfxm1-2r-27 | LP | 6189 | 4088 | Acceptable | 2.886965e+03 |
| scorpion | LP | 358 | 388 | Acceptable | 9.345764e+02 |
| scrs8-2r-256 | LP | 9765 | 7196 | Acceptable | 2.156604e+01 |
| steenbre | Vanderbei | 540 | 126 | Acceptable | 5.197370e+02 |
| steenbrf | Vanderbei | 468 | 108 | Acceptable | 2.315828e+02 |
| steenbrg | Vanderbei | 540 | 126 | Acceptable | 5.006421e+02 |

## Acceptable (not Optimal) — 15 problems

These problems converged within relaxed tolerances but not strict tolerances.

| Problem | Suite | n | m | Ipopt status | POUNCE obj | Ipopt obj |
|---------|-------|---|---|-------------|------------|-----------|
| QRECIPE | QP | 180 | 91 | Acceptable | -2.666160e+02 | -2.666160e+02 |
| aa4 | LP | 7195 | 426 | Acceptable | 9.659420e+02 | 9.659420e+02 |
| air05 | LP | 7195 | 426 | Acceptable | 9.659420e+02 | 9.659420e+02 |
| bore3d | LP | 315 | 233 | Acceptable | 4.094410e+02 | 4.094410e+02 |
| co5 | LP | 7993 | 5715 | Acceptable | 5.075138e+03 | 5.075138e+03 |
| coshfun | Vanderbei | 61 | 20 | Maximum_Iterations_Exceeded | 4.186680e-01 | -1.169474e+11 |
| cq5 | LP | 7530 | 5025 | Acceptable | 2.842296e+03 | 2.842296e+03 |
| model3 | LP | 3840 | 1609 | Optimal | 1.750657e+03 | 1.750657e+03 |
| model7 | LP | 8007 | 3358 | Optimal | 4.942951e+03 | 4.942951e+03 |
| orthrds2 | Vanderbei | 203 | 100 | Optimal | 1.544296e+03 | 1.544297e+03 |
| pilot.ja | LP | 1988 | 940 | Acceptable | -6.113137e+03 | -6.113137e+03 |
| qcqp1000-1nc | Mittelmann | 1000 | 154 | Optimal | -1.841514e+06 | -1.841514e+06 |
| recipe | LP | 180 | 91 | Acceptable | -2.666160e+02 | -2.666160e+02 |
| ssebnln | Vanderbei | 194 | 96 | Optimal | 1.079313e+06 | 1.078040e+06 |
| steenbrc | Vanderbei | 540 | 126 | Unknown_Error | 1.040910e+04 | 1.258424e+04 |

## POUNCE-Only Suite Details

These suites currently run POUNCE only — no Ipopt-side comparison is captured in their result files. Per-problem timing and iteration counts are shown so users can inspect the whole picture.

### LPopt

| Problem | n | m | Status | Objective | Iters | Time |
|---------|---|---|--------|-----------|-------|------|
| ex10 | 17,680 | 69,608 | Maximum_CpuTime_Exceeded | N/A | 30 | 600.05s |
| irish-electricity | 61,728 | 104,259 | Maximum_CpuTime_Exceeded | N/A | 1630 | 600.06s |
| qap15 | 22,275 | 6,330 | Optimal | 8.6749e+02 | 49 | 102.48s |
| supportcase10 | 14,630 | 165,684 | Maximum_CpuTime_Exceeded | N/A | 26 | 600.09s |

POUNCE: **1/4 Optimal** in 1902.69s total

---
*Generated by benchmark_report.py*