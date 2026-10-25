# EXPERIMENT: feral#228 `OnStale` × pounce #1046, end to end

**Branch `exp/feral-onstale` — never merge.** It patches `feral` to a git rev
(jkitchin/feral#228 head `58e2a467`), which blocks the crates.io publish.

## Question

1. Is `ReanalysisPolicy::OnStale` worth shipping and making pounce's default?
   Feral's evidence so far is replayed KKT *sequences* (qcqp1000-2c 0.75×
   flops, qcqp1000-2nc kkt-first 0.55×, others 1.00×). A replay holds the
   trajectory fixed; this measures the solve.
2. Does #1046 (forget the analysis built on the multiplier-init system) still
   earn its keep once `OnStale` is on? Replays say no on qcqp1000-2c/2nc;
   clnlbeam's 2465 → 514 iterations under #1046 cannot be replayed.

## The switches (one binary, four legs)

| env var | values | effect |
|---|---|---|
| `POUNCE_FERAL_REANALYSIS` | `never` (default) / `on_stale` | `Solver::with_reanalysis(OnStale(default))` on every FERAL backend |
| `POUNCE_EXP_FORGET_INIT` | `0` (default) / `1` | `1` = #1046's forget; `0` = `main` behaviour |

Checked on x86_64 before pushing: with `never`, the fixture sweep (both legs)
is identical line-for-line to pounce `main` on feral 0.21.0 (`=0`) and to
#1046 on feral 0.21.0 (`=1`). So the git feral and the switches change
nothing by themselves.

## Running it

```
cargo build --release -p pounce-cli
scripts/onstale-ab.sh target/release/pounce ab-out <model.nl>...
```

Four legs per model, serial; `results.tsv` has status, iterations,
second-opinion and mu-fallback columns (a rescued solve otherwise reads as a
speed-up), quality escalations, wall, `total_factor_secs`, flops, delayed
columns, final nnz(L), preprocess and objective. Interrupted runs resume.

Already measured here (x86_64, 4 cores), fixture corpus: `OnStale` moves only
`lbfgs eigena2` (still `ErrorInStepComputation`, 478 → 306 total iterations),
identically with and without #1046; #1046 moves `pooling_rt2stp` 492 → 111
and `deb7` 114 → 225 (+ rescue, 621 total), identically under both policies.
`laptime` (58k vars): `OnStale` no change in iterations or objective.
