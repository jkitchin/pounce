# Auxiliary-presolve fixtures

This directory is where the upstream-ripopt benchmark fixtures
should be vendored to unlock the full acceptance criteria for
[pounce#53](https://github.com/jkitchin/pounce/issues/53).

Until those land, the CLI smoke test in
`../../auxiliary_presolve.rs` runs against the existing
`../parametric.nl` to verify the `presolve_auxiliary=yes` path
compiles, links, and doesn't panic. That covers correctness but
not the headline reductions ripopt reports.

## Fixtures to vendor

Copy from ripopt with their MIT license headers preserved:

| Source (ripopt)                          | Destination (pounce)                              | Expected behaviour with `presolve_auxiliary=yes` |
|------------------------------------------|---------------------------------------------------|--------------------------------------------------|
| `tests/fixtures/tutorial_flow_density.nl`           | `tutorial_flow_density.nl`           | 0 IPM iterations                                 |
| `tests/fixtures/tutorial_flow_density_perturbed.nl` | `tutorial_flow_density_perturbed.nl` | 0 IPM iterations                                 |
| `tests/fixtures/gaslib11_steady.nl`                 | `gaslib11_steady.nl`                 | Reduces 204/200 → 140/136 vars/cons              |

Once vendored, extend `crates/pounce-cli/tests/auxiliary_presolve.rs`
with tests that:

1. Run `pounce` on `tutorial_flow_density.nl` with `presolve=yes
   presolve_auxiliary=yes` and assert
   `report.statistics.iteration_count == 0`.
2. Run on `gaslib11_steady.nl` and check the diagnostics struct
   reports `vars_eliminated >= 60` (the rough delta from 200 → 136).
3. Verify the same objective is reached as the un-presolved path
   on each, to within `tol`.

## Provenance and license

Both pounce and ripopt are released under the Eclipse Public License v2.0,
so no license conflict arises from vendoring these fixtures.

| Fixture                                | Origin                                                              | Upstream path                                          |
|----------------------------------------|---------------------------------------------------------------------|--------------------------------------------------------|
| `tutorial_flow_density.nl`             | ripopt issue-23 tutorial (`incidence_examples` model export)        | `ripopt/tests/fixtures/issue_23/tutorial_flow_density.nl`           |
| `tutorial_flow_density_perturbed.nl`   | same, shifted operating point                                       | `ripopt/tests/fixtures/issue_23/tutorial_flow_density_perturbed.nl` |
| `gaslib11_steady.nl`                   | ripopt gas-network benchmark; derived from GasLib gaslib-11 (CC-BY-SA 3.0) | `ripopt/benchmarks/gas/gaslib11_steady.nl`                          |

`.nl` files are machine-generated AMPL output and carry no header — provenance
is documented here. GasLib upstream: https://gaslib.zib.de/.
