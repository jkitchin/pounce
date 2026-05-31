# Preprocessing benchmark

`run_preprocessing_benchmark.py` runs a single `.nl` problem twice
— once with `presolve_auxiliary=no` (baseline), once with
`presolve_auxiliary=yes` — and compares iteration counts, final
objective, and wall time.

## Usage

```sh
python benchmarks/preprocessing/run_preprocessing_benchmark.py \
    crates/pounce-cli/tests/fixtures/parametric.nl
```

The script discovers a `pounce` binary at
`target/release/pounce` or `target/debug/pounce`, falling back to
`cargo run --release -p pounce-cli` if neither is built.

## Expected results on the auxiliary-presolve fixtures

The acceptance criteria from
[pounce#53](https://github.com/jkitchin/pounce/issues/53) target:

| Problem                                | Baseline | With `presolve_auxiliary=yes`              |
|----------------------------------------|----------|--------------------------------------------|
| `tutorial_flow_density.nl`             | 6–7 iters | 0 iters                                    |
| `tutorial_flow_density_perturbed.nl`   | 6–7 iters | 0 iters                                    |
| `gaslib11_steady.nl`                   | converges | same objective ± tol, 140/136 vars/cons    |

Those fixtures are vendored in
`crates/pounce-cli/tests/fixtures/aux_presolve/`; run:

```sh
python benchmarks/preprocessing/run_preprocessing_benchmark.py \
    crates/pounce-cli/tests/fixtures/aux_presolve/tutorial_flow_density.nl
```

## Output

```
Problem: crates/pounce-cli/tests/fixtures/parametric.nl

config                  iters         final_obj       wall
baseline                    8     2.500000000e+00     0.012s
auxiliary=yes               8     2.500000000e+00     0.014s

iteration delta:    +0   (0.0%)
auxiliary-preprocessing: 0 of 0 candidate block(s) eliminated, fixing 0 variable(s) and dropping 0 row(s) in 0 ms
```

A zero delta on `parametric.nl` is expected: it doesn't have the
structure auxiliary preprocessing exploits. The real wins show up
on the ripopt fixtures listed above.
