# pounce-convex

Interior-point solvers for POUNCE's convex problem classes: **LP and
convex QP** today, with cone-generic scaffolding for the conic family
(SOCP, exponential/power cones, SDP) planned.

This crate is Phase 2 of the LP/QP routing plan
(`dev-notes/lp-qp-routing.md`). It provides a bare primal-dual
interior-point method for convex QP in standard form:

```text
minimize    ½ xᵀP x + cᵀx
subject to  A x = b
            G x ≤ h
```

LP is the `P = 0` case and is solved by the same driver.

## Design

- **Cone-generic.** The interior-point iteration is built over a
  [`cones::Cone`] trait with only the nonnegative orthant
  (`cones::nonneg`) implemented. Later phases add SOC / PSD / exp / pow
  cones behind the same trait, so the driver is extended, not rewritten.
- **Shared factorization.** The symmetric indefinite KKT system is solved
  through `pounce_linsol::Factorization` — the same factor-once /
  solve-many handle the NLP path uses (feral by default, MA57 optional).
  No new linear-algebra dependency.
- **Bare method now, Mehrotra next.** The current iteration uses a fixed
  centering parameter and fraction-to-boundary step control. Mehrotra
  predictor-corrector and the homogeneous self-dual embedding are Phase 3
  and slot into this same scaffolding.

## Status

Phase 2, first increment: correct convex-QP solves validated against
problems with analytically known optima (unconstrained, equality-,
inequality-, and bound-constrained). Not yet wired into the CLI dispatch
(`auto` still routes to NLP-IPM); not yet performance-tuned.
