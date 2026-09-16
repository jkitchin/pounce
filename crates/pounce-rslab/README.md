# pounce-rslab

Experimental [RSLAB](https://github.com/milanofthe/rslab) backend for POUNCE's
symmetric linear-solver layer. Implements `pounce_linsol::SparseSymLinearSolverInterface`
alongside `pounce-feral` (FERAL, the shipping default) and `pounce-hsl` (MA57).

This crate exists to answer one question — *is RSLAB a numerically credible KKT
backend for POUNCE?* — not to ship a third production backend. The assessment it
was built to support is `dev-notes/rslab-backend-assessment.md`.

## It is not wired into the optimizer, and cannot be

`pounce-algorithm` does not know this crate exists. That is not caution, it is a
constraint: RSLAB has no crates.io release under its own name (the registry
`rslab` is an unrelated slab allocator, `os-module/rslab` 0.2.1), so it is a
**git** dependency, and cargo refuses to publish a crate carrying one — even an
optional one behind a disabled feature. Adding `pounce-rslab = { optional = true }`
to `pounce-algorithm` would therefore break the crates.io release of
`pounce-algorithm` and of every crate above it, which
`scripts/check-release-consistency.sh` guards. Hence `publish = false` here, and
no edge into the publishable graph.

Nothing is lost for the evaluation. The backend is a
`Box<dyn SparseSymLinearSolverInterface>` like any other, so it plugs into:

* `pounce_linsol::Factorization::new(dim, irn, jcn, vals, backend)` — the
  factor-once / solve-many handle the cross-solver harness drives;
* `IpoptApplication::set_linear_backend_factory(..)` — a **full NLP solve**
  through the real IPM, with RSLAB under it and no change to POUNCE's core.
  `examples/rslab_nlp_solve.rs` does exactly that.

`pounce-rslab` is a workspace member but **not** a default-member, exactly like
`pounce-hsl`, so a plain `cargo build` / `cargo test` never fetches RSLAB. Build
it explicitly:

```sh
cargo test -p pounce-rslab
cargo run -p pounce-rslab --release --example rslab_bench
```

## What the adapter actually does

| POUNCE expects | RSLAB offers | Adapter |
| --- | --- | --- |
| triplet, 1-based, lower triangle | CSC, 0-based, lower triangle | index shift + `from_triplets`, then a cached triplet→slot scatter on every refill |
| column-major multi-RHS block | **row-major** `n × nrhs` | transpose in `backsolve`, fused with the equilibration |
| symbolic reuse across refactors | `analyze_with` → `factor_numeric` | analysis held for the life of the pattern |
| `number_of_neg_evals` | `Inertia { positive, negative, zero }` | reported unchanged |
| `min |pivot|` for the gh#540 trust gate | *nothing* | recovered from `D` in one O(n) pass (`inertia::scan_d`), 2×2 blocks by eigenvalue |
| `Singular` on rank deficiency | `Err(NumericallyRankDeficient)` | mapped |
| equilibration | private to `LdltSolver` | the same one-pass inf-norm step, reimplemented in `scaling.rs` |

The adapter uses RSLAB's low-level entry points rather than its ergonomic
`LdltSolver` handle for exactly one reason: `LdltSolver` keeps its `LdltFactors`
private and RSLAB has no `min_pivot_magnitude()` anywhere, so POUNCE's
inertia-trust gate could not be preserved through it. See the module docs on
`lib.rs` for what that costs.
