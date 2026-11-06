# `relu_nn`: where pounce's time goes against CCOpt (gh#1136)

Measured 2026-11-06 on `main` + #1137 (`70a5da38`), macOS arm64. CCOpt
0.1.0 on MadNLP 0.10.3 with MUMPS 5.9.1, the versions the `mpcc_compare`
environment pins. The numbers come from counters read out of both solvers.
The code citations are to the installed sources
(`~/.julia/packages/{MadNLP,CCOpt}`).

## Suite totals (42 problems, `mpcc_compare --suites relu_nn`)

| arm | iterations | wall | ms / iteration |
|---|---|---|---|
| pounce (declared pairs) | 16,632 | 92.6 s | 5.6 |
| CCOpt relaxation | 1,945 | 1.8 s | 0.9 |
| CCOpt penalty | 10,790 | 8.9 s | 0.8 |

Pounce's count is the corrected one. Before the gh#884 retry was counted,
the same run reported 15,884.

The suite-wide ms/iteration is misleading. CCOpt's iterations sit mostly on
small models, and its large ones take very few. Per model, below.

On pounce's side, solves that enter restoration take 45% of the time (65 of
315 lowered solves, 41.4 s, 2,833 restoration iterations). The exact-product
stage takes 56%. It supplies the answer on 13 of 42 models.

## The largest model, `digits_128x3_img372_eps25`

Pounce takes 942 iterations (856 as reported before the gh#884 retry was
counted) and 20–25 s. CCOpt's relaxation takes 54
iterations and 0.22 s.

### The linear algebra is not the gap

Both factor the same KKT matrix: order 1,984, nnz 45,248.

| per factorization | FERAL | MUMPS (via MadNLP) |
|---|---|---|
| flops | 33.4 M | 90.4 M (`RINFOG(3)`) |
| factor entries | 166k | 264k (`INFOG(29)`), 398 delayed pivots |
| time | 3.7 ms | 2.9 ms mean, 4.6 ms bare on the final KKT |

MadNLP builds its default `SparseKKTSystem` (`IPM/options.jl:225`) and
solves it with sequential MUMPS (`default_sparse_solver = MumpsSolver`,
`options.jl:240`). It runs with AMD ordering (`ICNTL(7)=0`), diagonal
scaling (`ICNTL(8)=1`), pivot tolerance 1e-6 (`LinearSolvers/mumps.jl:193–203`)
and single-thread OpenBLAS. Its `linear_solver_time` covers factorization
and refined back-solves, not KKT assembly (`IPM/factorization.jl:4,24`).

On a plain NLP with the same lifted structure, FERAL beats HSL. Iterations
in each run, then search-direction time per iteration: FERAL 412, 10.2 ms;
MA57 (IDAES Ipopt 3.13) 463, about 19 ms; MA27 407, about 57 ms; Ipopt's own
MUMPS 400, 14.3 ms. Changing `feral_ordering` (amd, metis, scotch) only
makes FERAL slower.

### What differs

1. **CCOpt solves the problem in one solve.** `ProportionalRelaxationUpdate`
   sets `σ = 1.0·μ¹` from `μ_init` (CCOpt `Solvers/relaxation/Relaxation.jl`).
   There is no `τ = 1` stage, no exact finish and no exact stage. Pounce's
   coupled solve uses the same law (`COUPLED_TAU_PER_MU = 1.0`,
   `mpcc_driver.rs`). Around it, pounce also runs:
   - a `τ = 1` relaxation (45 iterations);
   - the coupled solve itself (109 iterations);
   - an exact finish (86 iterations plus 71 in restoration, then a gh#884
     retry);
   - the exact stage (about 695 iterations, its answer discarded on this
     model).

2. **Pounce factors more often per iteration: 2.0 against 1.3.** It needs
   a Hessian perturbation on 79 of the coupled solve's 110 iterations.
   CCOpt needs 70 factorizations for 54 iterations. Both use the same test
   and ladder. MadNLP's `is_inertia_correct` is `num_zero == 0 &&
   num_pos == num_variables` (`KKT/KKTsystem.jl:242`). CCOpt's
   `inertia_correction!` (`Solvers/relaxation/inertia_correction.jl:79`) is
   Ipopt's scheme: the same first perturbation, ×8 growth, and restoration
   past the maximum. So pounce's iterates have the wrong inertia far more
   often. Instrumenting pounce's factor loop on the continuation path: 406
   of 720 factorizations were rejected, 325 of them with **exactly one**
   negative eigenvalue too many. Changing `feral_pivtol` (1e-8 to 1e-2)
   leaves 48–56% rejected, so the eigenvalue is real at these iterates, not
   a misread sign.

   CCOpt's Vicente–Wright regularization does not explain it. It is zero
   until `μ ≤ 5e-6` (`solver.jl:2–10`).

3. **The two solvers receive different models.** Pyomo's `mpec.nl`
   declares each pair as `y ⟂ row .c` with `.c = bv` and `bv` **free**, so
   `bv ≥ 0` is a row with its own slack: 1,152 rows. CCOpt's
   `write_vertical` (`mpcc_compare/solvers.py`) passes the pair as two
   variables, `y ⟂ w` with `w ≥ 0` a bound: 768 rows. Pounce in CCOpt's
   form (Python API, `complementarity_lift=yes`) rejects 39% of its
   factorizations rather than 56%, but takes 243 iterations rather than 161
   and reaches a worse local answer (−17.2542 against −17.2870). So the
   formulation is part of the story, not the explanation.

## Open

- Why pounce's coupled-solve iterates lose inertia by one eigenvalue so
  often. The next measurement is the inertia and the smallest pivots at
  matched iterates of pounce's and MadNLP's solves of the vertical-form
  model.
- Whether the `τ = 1` pre-solve and the exact finish can be skipped when the
  coupled endpoint already certifies tightly. That would be
  answer-preserving, but it is a trajectory change and needs the full MPCC
  corpora.
- The exact stage (#1136 item 2) and restoration's cost per iteration
  (item 3) are the larger terms. Cutting the exact stage gives up answers:
  it supplies 13 of 42 here.
