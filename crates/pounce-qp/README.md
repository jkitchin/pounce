# pounce-qp

Sparse parametric active-set quadratic programming subproblem solver
for POUNCE. Used as the QP subproblem solver inside the active-set
SQP NLP path, and (in later phases) as the corrector inside
`pounce-sensitivity`.

**Status.** Phase 5a scaffold. Types, trait surface, and unit-test
plumbing are in place; the solver algorithm is not yet implemented.
The crate compiles, its smoke tests pass, and downstream crates can
already depend on it for type-level integration work in parallel.

See [`docs/research/active-set-sqp-warm-start.md`](../../docs/research/active-set-sqp-warm-start.md)
for the full design note, algorithm pinning, integration plan, and
phasing.
