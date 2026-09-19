# KKT scaling, Phase 0b: GasLib-40 transient, horizon sweep

Structured-KKT decomposition, Phase 0b: fit an exponent in the number of time
blocks to every quantity that could carry superlinear solve time, and place the
growth in one of four rows — fill, pivoting, iterations, other per-iteration
work. Only fill and pivoting are things a block-structured KKT solve can change.

Harness: `benchmarks/kkt_scaling/sweep.py` (the report fields it reads were
added in the same branch: factorization seconds, work proxy, delayed / 2×2 /
tiny pivots, largest front, and restoration factorizations reported apart).

## Family

GasLib-40 transient (public GasLib topology): hourly backward-Euler finite
volumes, initial state fixed to the steady state, daily sinusoidal demand,
**no periodic terminal constraint** (that would close the chain into a ring),
cold start. Horizon `H` = 6, 12, 24, 48, 96, 192 hours → 7 to 193 time points,
5 788 to 162 772 free variables (≈ 844 per hour, exactly linear). pounce at
`cecadc3d`, release build, default options except `feral_ordering`. All 18
solves converged (`Solve_Succeeded`).

## Result

Exponent in `H` (log-log least squares over all six sizes):

| metric | auto | metis | auto_race |
|---|---|---|---|
| iterations | 0.05 | 0.05 | 0.09 |
| nnz(L) | 1.19 | 1.13 | 1.14 |
| work proxy per factorization | **1.71** | **1.54** | **1.54** |
| delayed columns per factorization | 0.86 | 0.82 | 0.88 |
| factorization s / iteration | 1.23 | 1.10 | 1.07 |
| other s / iteration | 1.11 | 1.13 | 1.10 |
| wall | **1.24** | **1.16** | **1.17** |

| H | n | wall (auto) | iterations | largest front auto / metis | factorization share |
|---|---|---|---|---|---|
| 6 | 5 788 | 1.1 s | 152 | 95 / 112 | 64% |
| 24 | 20 980 | 6.5 s | 177 | 256 / 320 | 55% |
| 96 | 81 748 | 34.2 s | 180 | 698 / 683 | 68% |
| 192 | 162 772 | 67.8 s | 189 | 1 002 / 773 | 72% |

## Reading

1. **This family does not reproduce quadratic scaling.** Wall time grows as
   `H^1.16–1.24` over a 28× range of problem size. Whatever produced the
   observed near-`N²` behaviour is not present in this configuration.
2. **Iterations are flat** (exponent ≤ 0.09, 152–219). The iteration row is
   empty here.
3. **Pivoting is not the source.** Delayed columns per factorization grow
   slightly *less* than linearly, i.e. the delayed work per unit of problem is
   constant or falling.
4. **Fill is the one superlinear signal, and it is moderate.** The work proxy
   per factorization grows as `H^1.5–1.7` and the largest front grows with the
   horizon under every ordering (metis 112 → 773). For a chain of fixed-size
   time blocks, a nested dissection that cuts *between time points* has a top
   separator of one interface, independent of `H`; a front that keeps growing
   means the partitioner is cutting elsewhere. That is the case Phase 2
   (structure-derived block-level ordering) exists to test. It is bounded,
   though: factorization time per iteration grows only as `H^1.1–1.2`, so a
   perfect temporal ordering would buy roughly `32^0.2 ≈ 2×` at the largest
   size, not an exponent change from 2 to 1.
5. **`metis` is no better than the default here** in wall time, despite lower
   fill at 96–192 h: it takes more iterations at 12 h and 96 h. Ordering
   changes the trajectory (it is also a pivoting decision), so fill alone is
   not the wall-time predictor.
6. **Non-factorization work is ~30% of the time and grows ~linearly** (1.1),
   so it is not a hidden quadratic either.

## What this does not settle

The observed quadratic scaling came from other configurations. Candidates this
family deliberately left out, each of which changes the structure:

- **periodic / cyclic-steady-state terminal constraints** (a ring, and in the
  CSS computation the initial state is free);
- the **linepack lower bound** and terminal targets of the production runs;
- a **finer discretization** (the documented 24 h GasLib-40 transient runs are
  56k variables, ≈ 2.7× this model's per-hour size) or a 30-minute time step;
- sizes beyond 163k (the documented 72 h case is 503k variables).

Next: rerun the sweep on the configuration that actually showed near-`N²`
growth, then decide the §38.2 row from that, not from this family.
