# Acknowledgments

POUNCE's nonlinear-programming core is a Rust port of
[Ipopt](https://github.com/coin-or/Ipopt), the interior-point nonlinear
programming solver by Andreas Wächter, Lorenz T. Biegler, and the COIN-OR
community. Its algorithm, console output, and option semantics are modeled
directly on that codebase, which is released under the EPL-2.0.

It is a sibling of [ripopt](https://github.com/jkitchin/ripopt), an
earlier memory-safe interior-point NLP optimizer in Rust by the same
author (DOI
[10.5281/zenodo.19542664](https://doi.org/10.5281/zenodo.19542664)).

## Convex solver inspiration

The specialized convex conic solver (`pounce-convex`; see
[Convex Solver](convex-solver.md)) is a pure-Rust port of ideas — not a
wrapper — from two reference projects, gratefully acknowledged:

- [**Clarabel**](https://github.com/oxfordcontrol/Clarabel.rs) by Paul
  Goulart and Yuwen Chen (University of Oxford). POUNCE's
  homogeneous-free conic interior-point design — a quadratic objective
  handled directly over a product of symmetric cones, with
  Nesterov–Todd scaling for the second-order cone and a
  diagonal-plus-rank-1 sparse KKT representation — follows Clarabel's
  approach. Clarabel is itself a pure-Rust solver; POUNCE shares the
  spirit but is an independent implementation.
- [**PaPILO**](https://github.com/scipopt/papilo), the presolving
  library of [**SCIP**](https://www.scipopt.org/) (the Zuse Institute
  Berlin optimization suite). POUNCE's transaction-stack presolve with
  full primal **and dual** postsolve — forcing constraints, dominated
  columns, bound tightening with global dual recovery, parallel/duplicate
  rows, iterated to a fixpoint — is modeled on PaPILO's catalog and
  postsolve discipline.

## Contributors

- **David Bernal Neira** ([@bernalde](https://github.com/bernalde))
  designed and prototyped the auxiliary-equality preprocessing pass
  in [ripopt PR #32](https://github.com/jkitchin/ripopt/pull/32).
  POUNCE's `pounce-presolve::auxiliary` Phase-0 orchestrator (issue
  [#53](https://github.com/jkitchin/pounce/issues/53)) is a port of
  that work — Hopcroft-Karp matching, Dulmage-Mendelsohn partition,
  Tarjan SCC, block-triangular reduction, damped-Newton block
  solver, reduction frame with multiplier recovery — and ships
  with the `tutorial_flow_density{,_perturbed}.nl` and
  `gaslib11_steady.nl` test fixtures David vendored.
- **Milan Rother** ([@milanofthe](https://github.com/milanofthe))
  suggested the boundary value problem solver and the tritium
  gas-liquid-contactor (GLC) test problem behind
  [`docs/src/bvp.md`](bvp.md) and
  `python/examples/glc_feral_vs_scipy.py`. The GLC model is adapted from
  [pathsim-chem](https://github.com/pathsim/pathsim-chem)
  (`src/pathsim_chem/tritium/glc.py`, MIT License, Copyright (c) 2025
  PathSim).

## Key references

- Wächter, A., Biegler, L.T. "On the implementation of an
  interior-point filter line-search algorithm for large-scale
  nonlinear programming." *Mathematical Programming* 106(1), 25–57
  (2006). DOI
  [10.1007/s10107-004-0559-y](https://doi.org/10.1007/s10107-004-0559-y)
  — the algorithm POUNCE implements.
- Wächter, A., Biegler, L.T. "Line search filter methods for nonlinear
  programming: Motivation and global convergence." *SIAM Journal on
  Optimization* 16(1), 1–31 (2005). DOI
  [10.1137/S1052623403426556](https://doi.org/10.1137/S1052623403426556)
- Wächter, A., Biegler, L.T. "Line search filter methods for nonlinear
  programming: Local convergence." *SIAM Journal on Optimization*
  16(1), 32–48 (2005). DOI
  [10.1137/S1052623403426544](https://doi.org/10.1137/S1052623403426544)
- Fletcher, R., Leyffer, S. "Nonlinear programming without a penalty
  function." *Mathematical Programming* 91(2), 239–269 (2002). DOI
  [10.1007/s101070100244](https://doi.org/10.1007/s101070100244) — the
  filter concept underlying the line search.
- Pirnay, H., López-Negrete, R., Biegler, L.T. "Optimal sensitivity
  based on IPOPT." *Mathematical Programming Computation* 4(4),
  307–331 (2012). DOI
  [10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2)
  — the sIPOPT method behind `pounce-sensitivity`.
- Duff, I.S. "MA57—a code for the solution of sparse symmetric
  definite and indefinite systems." *ACM Transactions on Mathematical
  Software* 30(2), 118–144 (2004). DOI
  [10.1145/992200.992202](https://doi.org/10.1145/992200.992202) — the
  optional `ma57` linear-solver backend.
- Goulart, P.J., Chen, Y. "Clarabel: An interior-point solver for
  conic programs with quadratic objectives." (2024).
  [arXiv:2405.12762](https://arxiv.org/abs/2405.12762) /
  [Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs) — the
  conic interior-point design behind `pounce-convex`.
- Gleixner, A., Gottwald, L., Hoen, A. "PaPILO: A Parallel Presolving
  Library for Integer and Linear Optimization with Multiprecision
  Support." *INFORMS Journal on Computing* 35(6), 1329–1341 (2023). DOI
  [10.1287/ijoc.2022.0171](https://doi.org/10.1287/ijoc.2022.0171) —
  the presolve catalog and dual-postsolve model behind
  `pounce-convex::presolve`.
- Domahidi, A., Chu, E., Boyd, S. "ECOS: An SOCP solver for embedded
  systems." *European Control Conference* (2013), 3071–3076. DOI
  [10.23919/ECC.2013.6669541](https://doi.org/10.23919/ECC.2013.6669541)
  — the sparse second-order-cone KKT representation.
- Amos, B., Kolter, J.Z. "OptNet: Differentiable Optimization as a
  Layer in Neural Networks." *ICML* (2017), 136–145.
  [arXiv:1703.00443](https://arxiv.org/abs/1703.00443) — the implicit
  differentiation behind the `pounce.jax` convex layers.
- Wilkinson, M.D. et al. "The FAIR Guiding Principles for scientific
  data management and stewardship." *Scientific Data* 3, 160018
  (2016). DOI
  [10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18) — the
  provenance model behind the [JSON solve report](json-output.md).
