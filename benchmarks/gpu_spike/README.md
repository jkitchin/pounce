# gpu_spike — Phase-0 GPU spike harnesses

Turnkey measurement harnesses for the GPU batched-differentiable-layers
roadmap (`dev-notes/research/gpu-batched-layers.md`, §5 Phase 0). They
answer the two questions a CPU box cannot: the **GPU↔CPU throughput
crossover** and the **on-device f32 accuracy / determinism**.

Standalone crate (its own `[workspace]`) — build and run from this
directory. It does **not** depend on the pounce solver crates; the
batched dense-QP / Cholesky kernels here mirror the condensed-KKT solve
the GPU path would run.

## On / off switches

- **Compile-time:** `--features gpu` pulls in `wgpu`. Without it the
  crate builds pure-Rust CPU-only (no GPU dependency at all).
- **Runtime:** `--device cpu|gpu|both` forces a side even when built
  with the GPU feature — so you can A/B the *same* binary. `--backend
  vulkan|metal|dx12|gl|all` selects the wgpu backend for cross-platform
  comparison.

If no usable GPU adapter exists (or init fails), the GPU side
probe-and-verifies, prints a fallback line, and runs CPU-only — the
runtime-selection contract from the design note's §11.

## Steps

| step | what it measures |
|---|---|
| `baseline` | Step 0 — CPU batched-QP throughput (solves/sec): the bar the GPU must beat |
| `microbench` | Step 1 — GPU vs CPU batched dense Cholesky+solve; finds the crossover `(batch, n)` |
| `accuracy` | Step 2 — f32 vs f64 residual, the f64-refinement-tail recovery, and GPU run-to-run variation (determinism) |
| `all` | baseline + microbench + accuracy |

## Examples

```sh
# CPU throughput bar (run this on the target Mac first)
cargo run --release -- baseline -b 1024 -n 32 -m 32

# crossover sweep on Apple Silicon (Metal)
cargo run --release --features gpu -- microbench \
    --batches 256,1024,4096,16384 --dims 16,32,64 --backend metal

# on-device f32 accuracy + determinism on an ill-conditioned batch
cargo run --release --features gpu -- accuracy \
    -n 48 --jitter 1e-3 -r 20 --device gpu

# force CPU to A/B against the GPU run above, same binary
cargo run --release --features gpu -- accuracy -n 48 --jitter 1e-3 -r 20 --device cpu
```

## Flags

```
-b/--batch/--batches   comma list of batch sizes      (default 1024)
-n/--dim/--dims        comma list of system sizes n    (default 32)
-m/--cons              # inequality constraints (baseline QP)  (default 32)
-t/--threads           CPU worker threads              (default: all cores)
-r/--repeats           timing repeats; best is kept    (default 5)
--jitter               SPD diagonal jitter; smaller = worse-conditioned (default 1.0)
--device               cpu | gpu | both               (default: both if gpu feature, else cpu)
--backend              vulkan | metal | dx12 | gl | all  (default all)
```

## Notes / caveats

- The Step-1 GPU kernel is an **unoptimized spike** (global-memory,
  one invocation per batch element, no tiling or shared memory). Its
  throughput is a **lower bound** — a tuned kernel (threadgroup tiling,
  subgroup ops) would do better. The crossover it reports is therefore
  conservative.
- `--jitter` is the conditioning knob: lower values raise κ and lower
  the f32 accuracy floor — use it to probe where f32 needs the f64 tail.
- f32 results differ across vendors/backends; the f64 refinement tail
  is what makes the final answer agree to f64 tolerance everywhere
  (design note §11). Step 2 reports both the raw f32 residual and the
  post-tail residual so you can see the gap close.
