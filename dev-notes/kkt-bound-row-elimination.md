# Bound rows folded onto the KKT diagonal (gh#1243)

## What changed

The convex QP/conic interior point expands every finite variable bound into
an inequality row (`expand_bounds`: `x_j ≤ u` becomes `e_jᵀx + s = u`, and
`x_j ≥ l` becomes `−e_jᵀx + s = −l`). Each such row used to be one row of the
factored augmented system. On Mittelmann's `Primal2_1000` that made the HSDE
KKT 8.84 M rows and 19.3 M nonzeros, against 3.86 M rows for the same model on
the NLP arm. Ipopt instead carries bounds as `Σ = z/s` on the `(x, x)`
diagonal.

`KktStructure::build(.., eliminate = true)` now does the same. It is an exact
Schur complement, not an approximation, and it applies to **every
single-column row of a nonnegative-orthant block**, whether it came from a
bound or the user wrote it:

* An eliminated row `i`, with entry `a` in column `j` and `(z, z)` diagonal
  `v = −(s_i/z_i) − δ < 0`, adds `−a²/v` to `(j, j)` and leaves the factored
  matrix.
* The right-hand side gets `r_j − a·r_i/v`, and the solution's
  `u_i = (r_i − a·u_j)/v` is recovered after the back-solve.

The drivers do not see the difference:

* **Layout.** `KktStructure::dim` is still the full `[x | y | z | aux]`
  length. The driver writes an eliminated row's scaling through
  `update_blocks` exactly as before, into a trailing value slot.
  `refactor`, `solve` and `matvec` take and return full-length vectors.
* **Refinement.** The HSDE's iterative refinement still forms `rhs − K u` for
  the *full* `K` (`KktStructure::matvec`). So the refinement residual that its
  dynamic regularization keys on measures the same quantity it always did.
* **Inertia.** The HSDE's inertia test expects one negative eigenvalue fewer
  per eliminated row. This is exact by Haynsworth inertia additivity: the
  eliminated block is all negative on the orthant.
* **Value array.** `refactor` folds the terms into the value array, factors,
  and restores the `(x, x)` entries bit for bit. Later updates and the matvec
  therefore see the full KKT's own values.
* **Guard.** If an eliminated `v` is not a finite negative number, or a fold
  overflows, `refactor` reports `Singular`. The HSDE answers that by raising
  the `(z, z)` regularization, which moves every `v` away from zero.

Paths covered: the symmetric HSDE (`hsde.rs`), the direct driver (`run_ipm`)
and the reusable `QpFactorization`. The nonsymmetric-cone HSDE
(`hsde_nonsym.rs`, exponential and power cones) assembles its own KKT and
still factors bound rows. `assemble_kkt_for_bench` builds the full system on
purpose.

`qp_kkt_row_elim=no` (`QpOptions::kkt_row_elim = false`) factors the full
system as before. Measured: it reproduces `main` exactly across the fixture
sweep (below).

## The exception: fixed and sub-tolerance boxes

A column whose two bounds are both finite and at most `NARROW_BOX = 1e-8`
(relative) apart, or crossed, keeps both of its rows in the matrix. At
feasibility `s_u + s_l = u − l ≈ 0`, so both slacks vanish together. Both
folded `z/s` terms then grow in lockstep, and each `dz` is recovered as a
quotient of two vanishing numbers.

This was found, not predicted. `lp_crossover_bridge_rescue`'s seed 311 has one
fixed column. With it folded, the HSDE's final absolute dual residual went
from `8.4e-11` to `4.5e-8`, with the same verdict, objective and iteration
count. The crossover gate, which reads that residual, then accepted a simplex
vertex the test needs declined. With the exception in place, the same solve
reaches `3.1e-11`, better than with elimination off.
`tests/issue1243_bound_row_elimination.rs` now requires the eliminated solve's
absolute KKT error to be within 10× of the full system's on every instance. A
verdict-and-objective check alone could not have caught this.

## Evidence

* `ipm.rs` `kkt_row_elim_tests`:
  * Detection is pinned on a fixture that reaches every branch: bound pairs,
    a lone bound, a scaled singleton, duplicate triplets, a two-column row, an
    SOC singleton, a fixed column, a narrow column and a wide scaled box.
  * The full product is unchanged (`< 1e-14`).
  * The reduced factor solves the full system (`< 1e-10`, with the residual
    checked against the full matrix).
  * The negative-eigenvalue count drops by exactly the eliminated count.
  * The value array is restored bit for bit, and the guard's branch is
    reached.
* `tests/issue1243_bound_row_elimination.rs`: elimination on versus off, on
  both drivers. It covers 24 random LPs/QPs with every bound shape (boxed,
  one-sided, free, fixed), a pure box LP with no `(z, z)` block left, a
  boxed SOCP, bound-decided infeasibility and unboundedness along an
  eliminated bound. When written, all 48 random pairs took identical
  iteration counts.
* Fixture sweep (`scripts/sweep-fixtures.sh`, both legs, 304 fixture-legs),
  against `main` at `0c9dacc`:
  * `qp_kkt_row_elim=no`: **empty diff.**
  * Default: 6 fixtures move, identically on both legs (the convex arm
    ignores the Hessian approximation). No status, engine or objective moves
    beyond `issue745_netlib_problem`'s `6.021872e-10 → 6.021890e-10`, an
    absolute `2e-15` on a zero optimum.

    | fixture | iterations | where the trajectories part |
    |---|---|---|
    | `convex_qp_qpilotno` | 83 → 82 | trace iteration 34, `μ ≈ 3e-9` |
    | `scaled_feasible_b` | 36 → 38 | iteration 17, `μ ≈ 3e-14` |
    | `qcqp_ball_rhs_1e20` | 23 → 24 | iteration 2, `μ` differing by 2 ppm (a `1e20` right-hand side) |
    | `feasible_x0_sentinel_bound` | 26 → 31 | iteration 3, at a primal residual of `1e16` (a `−1e30` coefficient, a `5e20` rhs) |
    | `feasible_x0_extreme_row` | 34 → 36 | iteration 1 (`±1e30` coefficients next to boxes of width `1e6`) |

    The last two are adversarial-scaling fixtures. Their first Newton systems
    are conditioned far beyond `1/ε` in both forms, so the two factorizations
    give different, each refinement-checked, steps from the start. Both end
    `SolveSucceeded` at a final KKT error of `0` (`extreme_row` was `3.8e-11`
    without elimination). `feasible_x0_wide_scale`, the same family at `1e8`,
    does not move.
* Maros-Meszaros (`benchmarks/qp`, all 138, `.nl` from `generate_nl.py`),
  `RAYON_NUM_THREADS=1`, `max_wall_time=100`, `main` versus this change. Both
  binaries ran side by side on a 4-core Linux x86_64 container:
  * Verdicts: no problem gets worse. `BOYD2` goes `MaximumWallTimeExceeded`
    (43 iterations, 102 s) → `SolveSucceeded` (77 iterations, 86 s). `BOYD1`
    still hits the wall, after 28 iterations instead of 9. No engine or other
    objective moves.
  * Iterations: 126 unchanged. `QSHIP12S` −2; `HUES-MOD`, `HUESTIS`,
    `QADLITTL`, `QFFFFF80`, `QPILOTNO` and `QSIERRA` −1; `AUG2DCQP`,
    `PRIMALC1` and `QSHIP04S` +1.
  * Peak RSS: down by up to 45%. `CVXQP1_L` 401 → 221 MB, `CVXQP3_L`
    519 → 296 MB, `CONT-300` 1.92 → 1.21 GB, `BOYD1` 1.07 → 0.73 GB. It is
    never up by more than 0.5% (13 MB problems, which is noise).
  * Suite wall time: 522 → 397 s.
* A synthetic boxed LP (400 k columns, 208 k rows, 1.6 M nonzeros; banded,
  with every 50th column pinned by a singleton row), solved with the default
  convex HSDE and single-threaded. All three runs take 20 iterations to the
  same objective (to `3e-11` relative):

  | binary | peak RSS | wall |
  |---|---|---|
  | `main` | 6.24 GB | 100 s |
  | this change, `qp_kkt_row_elim=no` (the copy fixes only) | 4.97 GB | 100 s |
  | this change | 3.94 GB | 69 s |

## What is still open from gh#1243

* Items 3 and 4 (the `permute_pattern` copies in FERAL's symbolic phase, and
  MC64 on LP-shaped KKTs) are in the `feral` crate, not here.
* `expand_bounds` still builds the expanded `QpProblem` (a copy of `G` plus
  one triplet per bound) beside the problem it came from. The factored KKT
  no longer carries those rows, but the problem copy remains.
* The `.nl` row/column names (0.51 GiB on `Primal2_1000`) are read up front.
