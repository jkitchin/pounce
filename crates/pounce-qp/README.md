# pounce-qp

Sparse parametric active-set quadratic programming subproblem solver
for POUNCE. The QP subproblem solver inside the active-set SQP NLP
path (Phase 5b) and the corrector inside `pounce-sensitivity`
(Phase 5c).

Pure Rust, no FFI. Layered on `pounce-common` + `pounce-linalg` +
`pounce-linsol`; tests use `pounce-feral` as the LDLᵀ backend.

## Status

**Phase 5a — feature-complete on correctness.** Every problem
class identified in the design note (`dev-notes/research/active-set-sqp-warm-start.md`)
solves end-to-end; infeasibility is certified honestly; warm-start
is wired. Five of the six analytical-ladder problems (§8.0) pass;
problem #4 (LICQ-violating redundant equality) is a documented
limitation that needs rank-detection beyond inertia control.

**Phase 5a.1 — performance and tooling refinements landed.** The
Harris-style two-pass ratio test (cycling-prevention core of
GMSW EXPAND), the QPS RANGES section, the cached-factor `resolve`
infrastructure (the building block on which full Schur layers),
and basic §8.2 scaling-sweep diagnostics all ship.

**Phase 5a.2 — algorithmic completion landed.** §4.2 sparse
Schur-complement active-set updates (c18 standalone module +
c19 wired into `solve_general` behind `use_schur_updates`) and
§4.4 full GMSW EXPAND with τ-growth + snap-reset (c20) are both
done. The Schur path is opt-in; correctness verified by Schur-
vs-refactor cross-checks. EXPAND degrades to Harris-only
behavior on non-cycling problems and only kicks in on
pathological degeneracy. The remaining items (Maros-Mészáros
oracle comparison, large-n scaling benchmarks) require FFI or
benchmark infrastructure that fall outside the pure-Rust
constraint.

**Phase 5b — SQP NLP integration core landed** in
`pounce-algorithm::sqp`. The crate is now wired into the
`SqpAlgorithm` outer loop via `ParametricActiveSetSolver`; SQP
consumes any `IpoptNlp` and converges on convex equality NLPs
(1 iter) and nonlinear NLPs (l1-merit line search, demonstrated
on a circle-projection problem). `AlgorithmBuilder` exposes the
SQP path via `algorithm = ActiveSetSqp` and the sister
`build_sqp_with_backend` method.

## What works

| Problem class | Path | Tested |
|---|---|---|
| Unconstrained QP | `solve_equality_only` fast path (m=0, no bounds) | ladder #1 |
| Equality-only (free vars) | `solve_equality_only` | ladder #2 |
| Box-constrained (m=0) | `solve_box_constrained` | ladder #3 |
| Equality + bounds (bound-feasible eq solution) | `solve_equality_plus_bounds` | dedicated tests |
| General inequalities (cold-feasible) | `solve_general` | dedicated tests |
| Warm-start with arbitrary working set | `solve_general` | drop / one-iter tests |
| Bound- or inequality-infeasible cold start | `solve_elastic` (l1-elastic mode, §4.3) | dedicated tests |
| Certified infeasibility detection | `solve_elastic` + `is_feasible` check | ladder #5 |
| Indefinite H with PD reduced Hessian | inertia-controlled factor (§4.5) | ladder #6 |
| Indefinite H requiring shift | `factorize_with_inertia_control` retry loop | dedicated test |

| Algorithmic feature | Status |
|---|---|
| Sparse triplet KKT assembly over `SymTMatrix` / `GenTMatrix` | done |
| §4.5 inertia control via diagonal-shift retry | done |
| §4.3 l1-elastic mode | done |
| §4.7 iterative refinement (inherited from FERAL) | done |
| §4.4 anti-cycling: Bland's rule (`AntiCyclingChoice::Bland`) | done (c8) |
| §4.4 anti-cycling: Harris two-pass (`AntiCyclingChoice::Expand`) | done (c14) |
| §4.4 anti-cycling: full GMSW EXPAND (τ-growth + snap-reset) | done (c20) |
| §4.2 cached-factor `resolve` infrastructure | done (c16) |
| §4.2 sparse Schur-complement update standalone (`schur::SchurState`) | done (c18) |
| §4.2 sparse Schur wired into `solve_general` (`use_schur_updates`) | done (c19) |
| §8.1 Maros-Mészáros .qps reader (incl. RANGES) | done (c11, c13) |
| §8.1 Maros-Mészáros oracle comparison (qpOASES / OSQP) | **deferred** (FFI; not pure-Rust) |
| §8.2 basic scaling-sweep diagnostics | done (c15) |
| §8.2 large-n scaling (LASSO at 10²–10⁵, MPC horizon 10–160) | **Phase 5a.2** |
| §8.7 per-module unit tests for `kkt`, `elastic`, `refinement`, `qps` | done |

## Public API at a glance

```rust
use pounce_qp::{ParametricActiveSetSolver, QpProblem, QpSolver, QpOptions, QpWarmStart};
use pounce_feral::FeralSolverInterface;

let mut solver = ParametricActiveSetSolver::new(Box::new(FeralSolverInterface::new()));
let sol = solver.solve(&qp, Some(&ws), &QpOptions::default())?;
assert_eq!(sol.status, pounce_qp::QpStatus::Optimal);
```

For the QPS / Maros-Mészáros on-ramp:

```rust
use pounce_qp::parse_qps;
let model = parse_qps(qps_text)?;
// ... wrap model.h_irow / model.a_irow / model.g / model.bl ... into
// pounce-linalg SymTMatrix / GenTMatrix and pass to solver.solve.
```

## Tests

76 tests: 70 unit tests across 8 modules under `src/tests/`, plus 6
integration tests under `tests/`.

- `src/tests/analytical.rs` (26) — §8.0 ladder + integration tests for
  every problem class, with hand-derived expected values.
- `src/tests/api.rs` (11) — type-plumbing invariants for `WorkingSet`,
  `QpProblem::validate`, default `QpOptions`.
- `src/tests/kkt_unit.rs` (5) — §8.7 unit tests for
  `KktTriplet::add_h_diagonal_shift`.
- `src/tests/elastic_unit.rs` (7) — §8.7 unit tests for
  `ElasticReformulation::build` and `initial_seed`.
- `src/tests/refinement_unit.rs` (4) — §4.7 pin that FERAL's iterative
  refinement is on by default and delivers near-machine-precision;
  plus c16 cached-`resolve` contract tests.
- `src/tests/qps_unit.rs` (8) — QPS parser + round-trip solve + RANGES
  semantics for each row sense.
- `src/tests/scaling_unit.rs` (3) — §8.2 scaling-sweep diagnostics at
  `n ∈ {10, 50, 100, 200}` and a warm-restart speedup test
  (cold: ~30 refactors / ~20 ws changes; warm at optimum:
  1 refactor / 0 ws changes — the §8.5 payoff in microcosm).
- `src/tests/schur_unit.rs` (6) — Schur-complement update unit tests.
- `tests/` (6) — `cold_general_initial_regression.rs` and
  `mm_published_optima.rs` integration regressions.

Run with: `cargo test -p pounce-qp`. For per-test timings,
`cargo test -p pounce-qp --release -- --nocapture`.

## Design reference

The full design note, including literature pinning for every
algorithmic choice, integration plan, and per-workload notes, is at
[`dev-notes/research/active-set-sqp-warm-start.md`](../../dev-notes/research/active-set-sqp-warm-start.md).
