# lp_infeas — LPs with no optimum (infeasibility / unboundedness detection)

Every instance here is either **primal-infeasible** (no point satisfies the
constraints) or **dual-infeasible / unbounded** (the objective decreases
without bound). Nothing has an optimum, so the suite measures one thing:
does the solver reach the right verdict, how quickly, and does it ever reach
a *wrong* one?

The other LP suites (`lp/`, `lpopt/`) are feasible by construction. `lp/`
explicitly excludes the Mészáros `infeas/` directory. So this is the only
place the convex path's infeasibility and unboundedness certificates
(`docs/src/lp-qp-routing.md`, "Infeasible and unbounded problems") are
exercised on real models rather than unit fixtures.

## Sources

1. **NETLIB `lp/infeas`** (J. W. Chinneck, 1993) —
   <https://www.netlib.org/lp/infeas/>. 29 primal-infeasible LPs from
   petrochemical, forestry, network and BP operations models, plus the
   original infeasible GREENBEA. Up to 10.7k columns (`gosh`). Stored in
   netlib's compressed format and expanded with `emps`. CPLEX2 is documented
   as "almost feasible".
2. **MIPLIB 2017 Infeas** (H. Lu and B. Tang, 2026) —
   <https://mit-lu-lab.github.io/MIPLIB2017-Infeas/>. 334 LPs derived from
   MIPLIB 2017 with integrality dropped. The 229 `*_if` instances are
   primal-infeasible, each one side of a branch-and-bound split. The 105
   `*_ub` instances are unbounded: bounds and rows were dropped from a
   bounded root relaxation until the objective became unbounded. Every
   instance has at least ~100k nonzeros and the largest has tens of
   millions, so `generate_nl.py` keeps those with **≤ 300 000 nonzeros**
   (`--max-nnz`; 0 disables the cap). That leaves 93 `_if` + 54 `_ub` =
   147. Downloads are checked against the published SHA-256. Cite as Lu &
   Tang, "MIPLIB 2017 Infeas" (2026), and Gleixner et al., MIPLIB 2017,
   doi:10.1007/s12532-020-00194-3.

`generate_nl.py` converts each MPS with `../lp/mps_to_nl.py`, the same
HiGHS → Pyomo → ASL pipeline as `lp/`, and writes `nl/labels.json`. That
file maps each name to `{source, expected, n, m, nnz, highs}`. `expected`
is the source's own label. `highs` is HiGHS 1.15's verdict on the original
MPS, run with 1 thread and a 60 s limit, as an independent check of the
label and the reference column.

## Scoring

`score.py` puts every run in one of three classes:

| class | meaning |
|---|---|
| **correct** | `Infeasible_Problem_Detected` on a `primal_infeasible` instance; `Diverging_Iterates` (pounce's unbounded status) on a `dual_infeasible` one |
| **wrong** | any other *definite* verdict — a success status (an optimum of a model that has none) or the opposite class |
| **unresolved** | no verdict: a time limit, iteration limit, numerical failure or crash |

The convex path documents its verdicts as certificate-backed and never
reported in error, so `score.py` exits non-zero on a wrong verdict. Unresolved
instances measure speed and robustness and do not fail it.

## Running

```sh
pip install highspy pyomo            # generation only
make lp-infeas-generate              # downloads ~130 MB, converts, records HiGHS verdicts
make lp-infeas-run                   # pounce over every .nl, then score.md
make lp-infeas-score                 # re-score an existing pounce.json
```

`lp-infeas-run` uses the standard serial driver (`scripts/run_nl_bench.sh`)
at `BENCH_TIMELIMIT` (300 s). A full serial run takes roughly two hours, most
of it on the few MIPLIB instances that reach the limit. `score.py` accepts
several result files, so the run can be sharded: run the driver on disjoint
subsets of `nl/` and pass every shard's JSON to `score.py`.

There is no Ipopt reference. Ipopt's verdict on an infeasible LP is a local
`Infeasible_Problem_Detected`, not a certificate, so it does not measure the
same thing. The suite is also not in the composite `BENCHMARK_REPORT.md`,
whose loader compares solved counts against that reference.

## Results

Measured 2026-11-01 on `174a84f`, the first commit carrying the Farkas box
proof and the aggregation emptied-row fix. Default options (`solver_selection=auto`;
every instance routed to the convex LP interior point). 300 s limit on a
4-core container; NETLIB ran serially and the MIPLIB instances in three
concurrent shards, so their times carry some contention. The HiGHS column
uses HiGHS's default LP method with 1 thread and a **60 s** limit, so its
unresolved count is not directly comparable.

| source | class | n | correct | wrong | unresolved | median time, correct (s) | HiGHS correct / wrong / unresolved |
|---|---|---:|---:|---:|---:|---:|---:|
| netlib-infeas | primal_infeasible | 29 | 29 | 0 | 0 | 0.1 | 29 / 0 / 0 |
| miplib2017-infeas | primal_infeasible | 93 | 90 | 0 | 3 | 11.0 | 92 / 0 / 1 |
| miplib2017-infeas | dual_infeasible | 54 | 48 | 0 | 6 | 24.0 | 49 / 0 / 5 |
| **all** | | 176 | **167** | **0** | 9 | | |

- **No wrong verdicts.** Every label HiGHS reached agrees with the source's.
- **NETLIB is decided almost at once.** Presolve proves 15 of the 29 and
  the IPM certifies the other 14; the slowest is `klein3` at 0.8 s.
- **The 9 unresolved are all the time limit**, not an iteration limit or a
  stall on an uncertified ray. They are `neos-1367061_if`,
  `neos-4409277-trave_if`, `satellites2-25_if`, `eilC76-2_ub`,
  `gmut-76-40_ub`, `n2seq36q_ub`, `neos-1593097_ub`, `s55_ub` and
  `thor50dday_ub`. Detection takes few iterations (a median of 22 over the 138 certified),
  but on ~200k-nonzero models one IPM iteration can cost 10–44 s:
  in an earlier 600 s run `n2seq36q_ub` was certified unbounded in 9
  iterations and 396 s. Speed, not detection, is what this suite currently measures as the
  gap. HiGHS has a median of 0.5 s on the same models.
- HiGHS hit its 60 s limit on 6 instances. pounce resolves 5 of them within
  its 300 s (`lr1dr04vc05v17a-t360_if`/`_ub`, `neos-3372571-onahau_ub`,
  `neos-4954274-beardy_ub`, `opm2-z8-s0_ub`); `s55_ub` defeats both.

**Before the two fixes** (`0ae32e5`'s parent, MIPLIB only, 120 s limit,
4-way parallel): 82 of 93 primal-infeasible instances were certified, and
there were no wrong verdicts. Three instances (`cmflsp50-24-8-8_if`,
`momentum1_if`, `neos-662469_if`) ran the IPM's 200-iteration budget out on
a converged Farkas ray that the ratio test rejected. They then fell back to
the NLP path. At 600 s that path rescued `cmflsp50` after 197 more
iterations, and the other two still timed out. `dws008-03_if` and
`dws012-02_if` were "proved" infeasible by presolve from a `−1.3e-15`
rounding residual: the right verdict, but for the wrong reason. Both defects
are fixed in `0ae32e5`. See `docs/src/lp-qp-routing.md`, "Infeasible and
unbounded problems".
