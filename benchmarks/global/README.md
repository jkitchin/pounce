# Global-optimization benchmark (`pounce-global`)

A graduated suite of **verifiable** nonconvex problems for the spatial
branch-and-bound global solver — from quick 2-D classics to instances that
branch into the thousands. Every instance has a known global optimum, so the
harness checks the *certified* value against ground truth (not just that it
returned something).

Unlike the other tiers (which drive the CLI on AMPL `.nl` files), the global
solver is Rust-native and needs finite variable bounds, so the harness is a
self-contained Rust example — no Pyomo / `.nl` generation:

```sh
cargo run --release -p pounce-global --example benchmark
```

It prints the Markdown table below. The source is
`crates/pounce-global/examples/benchmark.rs`.

## What the instances exercise

| instance | what it stresses |
|---|---|
| **six-hump camel** | the classic 2-D nonconvex case (two global minima); envelopes + OBBT + most-violation branching |
| **himmelblau** | quartic with four global minima; the relaxation prunes it almost immediately |
| **bukin-6** | `\|·\|` + `√` (non-smooth, the Hessian sweep declines) — forces branching |
| **allpairs bilinear** `Σ_{i<j} xᵢxⱼ` | scalable McCormick stress; the relaxation is loose in the box interior, so node count grows fast with `n` |
| **double camel** (4-D) | two coupled camels — high node count; run serial **and** on the parallel node pool |

## Results

Apple M4 Pro (14 cores), `--release`, tolerances `abs_gap = rel_gap = 1e-4`,
`max_nodes = 500_000`. Every row certified the known global optimum (`✓`).

| instance | n | threads | status | objective | known | gap | nodes | peak frontier | est. peak mem | time (s) |
|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|
| six-hump camel | 2 | 1 | Optimal ✓ | -1.03163 | -1.03163 | 0.0e0 | 49 | 16 | 1.9 KiB | 1.55 |
| himmelblau | 2 | 1 | Optimal ✓ | +0.00000 | +0.00000 | 0.0e0 | 5 | 2 | 240 B | 0.26 |
| bukin-6 | 2 | 1 | Optimal ✓ | +0.00000 | +0.00000 | 0.0e0 | 187 | 15 | 1.8 KiB | 4.32 |
| allpairs bilinear | 4 | 1 | Optimal ✓ | -2.00000 | -2.00000 | 0.0e0 | 11 | 4 | 608 B | 0.54 |
| allpairs bilinear | 6 | 1 | Optimal ✓ | -3.00000 | -3.00000 | 0.0e0 | 139 | 45 | 8.1 KiB | 9.00 |
| allpairs bilinear | 8 | 1 | Optimal ✓ | -4.00000 | -4.00000 | 0.0e0 | 4039 | 1781 | 375.7 KiB | 381.49 |
| double camel | 4 | 1 | Optimal ✓ | -2.06326 | -2.06326 | 0.0e0 | 1749 | 312 | 46.3 KiB | 207.54 |
| double camel | 4 | 8 | Optimal ✓ | -2.06326 | -2.06326 | 0.0e0 | 1681 | 310 | 46.0 KiB | 47.03 |

## Reading the numbers

**Correctness at scale.** All eight certified the true global optimum, including
the non-smooth bukin-6 and the 4039-node `allpairs n=8`. The relaxation suite
(tight envelopes + αBB + OBBT + RLT) is strong: textbook problems like
himmelblau close in a handful of nodes. The instances that branch are the ones
where McCormick is genuinely loose in the box interior (`allpairs`) or the
objective is non-smooth (bukin-6).

**Parallel scaling.** The double camel has enough nodes to saturate the node
pool: 207.5 s serial → 47.0 s on 8 threads, a **4.4×** wall-clock speedup. (The
node count differs slightly — 1749 vs 1681 — because the parallel best-first is
non-deterministic, as documented; the certified optimum and gap do not change.)
This is the first scaling measurement on a problem large enough to be credible,
versus the earlier ~40-node toy.

**Memory.** The best-first frontier is the dominant resident-memory term, and
it stays small here: the heaviest instance (`allpairs n=8`) peaked at 1781 open
nodes ≈ **376 KiB**. Each frontier node costs ≈ `size_of(Node) + 2·n·8` bytes
(≈ 216 B at `n = 8`); `pounce_global::estimate_node_bytes` reports the figure.

Because every processed node pushes at most two children and pops one, the
frontier can hold at most `max_nodes + 1` open nodes, so the **worst-case**
frontier memory is `(max_nodes + 1) × bytes/node` — here ≈ 103 MiB at
`max_nodes = 500k`, `n = 8`. In practice pruning keeps the actual peak three
orders of magnitude below that. The library exposes both:

- `GlobalProblem::estimated_peak_memory_bytes(opts)` — the a-priori worst case,
  used by the CLI to **warn before solving** when a large `max_nodes` × wide
  problem could exhaust memory;
- `GlobalSolution::{peak_frontier, peak_memory_bytes}` — the measured peak after
  the solve (the CLI prints `peak_frontier=… (~…)` in its summary line).
