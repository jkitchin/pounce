# pounce-restoration

Restoration phase for POUNCE. Port of Ipopt's `Algorithm/Resto*`.

## What it does

When the [filter line search](../pounce-algorithm) cannot accept any
step from the current iterate, the regular IPM hands control here. The
restoration phase solves a relaxed feasibility problem:

```
min  ρ * ‖p - n‖_1 + 0.5 * ζ * ‖D_R (x - x_R)‖_2^2
s.t. g(x) - p + n = 0,
     x_L ≤ x ≤ x_U,   p ≥ 0,  n ≥ 0
```

so that constraint violation is driven down. On success it returns to
the regular IPM with a new iterate that the filter is willing to
accept. See `ref/Ipopt/AGENT_REFERENCE/RESTORATION.md` for the full
write-up.

## Modules

| Module                   | Upstream counterpart                       |
|--------------------------|---------------------------------------------|
| `r#trait`                | `IpRestorationPhase.hpp`                    |
| `resto_nlp`              | `IpRestoIpoptNLP.{hpp,cpp}`                 |
| `min_c_1nrm`             | `IpRestoMinC_1Nrm.{hpp,cpp}`                |
| `conv_check`             | `IpRestoFilterConvCheck.{hpp,cpp}`          |
| `aug_resto_system_solver`| `IpRestoAugSystemSolver.{hpp,cpp}`          |
| `init`                   | `IpRestoIterateInitializer.{hpp,cpp}`       |
| `resto_resto`            | `IpRestoRestorationPhase.{hpp,cpp}`         |
| `resto_inner_solver`     | inner IPM driver for the resto sub-problem  |
| `resto_alg_builder`      | strategy wire-up for the inner solver       |
| `output`                 | iteration output                            |

## Status

Phase 9 of the port. The trait surface and strategy skeletons are in
place and exercised on CUTEst / Mittelmann problems; the cycle
detector, μ-min widening, and almost-feasible gates have all landed
recently (see `git log --oneline crates/pounce-restoration`). Edge-case
bug-hunt continues in tandem with the CUTEst sweep.

## License

EPL-2.0.
