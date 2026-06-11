# Finding Multiple Minima from the CLI

The `pounce` command line solves one problem from one starting point. The
`--minima` family turns that single solve into a **global search**: it drives
the same interior-point solver in a loop, escaping each minimum it finds, and
collects the distinct local minima into a deduplicated archive. It is the
pure-Rust counterpart of the Python [`find_minima`](find-minima.md) API and
needs no Python — it works on built-in problems and on AMPL `.nl` files alike.

```console
$ pounce model.nl --minima flooding --n-minima 10
```

## Methods

`--minima <method>` selects one of six strategies. They share the same local
solver and acceptance test and differ only in how they leave a minimum once
found:

| method | how it escapes a found minimum | reference |
|---|---|---|
| `multistart` | independent random / Sobol' starts across the box | — |
| `mlsl` | Multi-Level Single-Linkage clustering of sampled starts | Rinnooy Kan & Timmer (1987) |
| `basinhopping` | Metropolis random walk over minima | Wales & Doye (1997) |
| `flooding` | repulsive Gaussian bumps added at found minima (filled-function) | Ge (1990) |
| `deflation` | softened `1/‖x−x*‖^p` poles added at found minima | Farrell, Birkisson & Funke (2015) |
| `tunneling` | equal-height tunnel term between descents | Levy & Montalvo (1985) |

`--multistart` is shorthand for `--minima multistart`. For help choosing,
see [Choosing a Method](find-minima-choosing.md) — the guidance there applies
unchanged to the CLI.

## Shared options

| flag | default | meaning |
|---|---|---|
| `--n-minima <N>` | 10 | target number of distinct minima (a stop condition) |
| `--max-solves <N>` | `8 × n-minima` | hard cap on solver calls |
| `--patience <N>` | 8 | stop after `N` solves in a row that find nothing new |
| `--dedup <d>` | 1e-4 | minima within this per-dimension-scaled distance are the same |
| `--psd-tol <t>` | 1e-6 | smallest Hessian eigenvalue tolerated by the saddle-rejection check |
| `--seed <S>` | 0 | seed for sampling / Sobol' scramble (runs are reproducible) |
| `--sobol` / `--no-sobol` | on | use a scrambled Sobol' sequence for box sampling |

A candidate is **accepted** when its solve converged, the point is finite and
inside the bounds, its objective Hessian is positive semidefinite within
`--psd-tol` (saddle rejection; skipped when no Hessian is available or the
problem is large), and it is not already within `--dedup` of an archived
minimum. The dedup distance is measured in a per-dimension-scaled space
(`‖(a−b)/L‖`, with `L` the box width per variable), so a single tolerance is
scale-free.

The search stops at the first of: **`target_reached`** (`--n-minima` found),
**`converged`** (`--patience` consecutive empty solves), or
**`budget_exhausted`** (`--max-solves` reached).

## Strategy knobs

Each is optional and used only by the relevant method; omit them to take the
defaults (which mirror `find_minima`). `"auto"` widths are sized per dimension
from the bounds.

| flags | method |
|---|---|
| `--sigma`, `--sigma-frac`, `--amplitude`, `--amp-margin` | `flooding` |
| `--eta`, `--power`, `--soft`, `--length`, `--length-frac` | `deflation`, `tunneling` |
| `--gamma`, `--samples-per-round` | `mlsl` |
| `--step`, `--temperature` | `basinhopping` |
| `--restart-jitter` | all (perturbation scale for restart fallbacks) |

> The repulsion methods (`flooding`, `deflation`, `tunneling`) run each escape
> solve under `hessian_approximation = limited-memory` — the analytic penalty
> term is added to the objective and its gradient, and the quasi-Newton update
> supplies curvature, so the dense augmented Hessian is never assembled. Each
> accepted point is then **polished** by re-solving the clean objective with
> the exact Hessian, so the reported minima sit on the true problem.

## Output

The console prints a ranked table of the distinct minima (rank, objective, and
scaled distance to the best), followed by the stop status and the number of
solves:

```text
find-minima: 6 distinct minima in 17 solves (target_reached)
  rank        objective     dist-to-best
     0      -1.03162845e0       0.000000e0
     1      -1.03162845e0      4.772232e-1
     ...
```

**Solution files.** With `.sol` output enabled, the **global best** minimum is
written to the usual `<stub>.sol` (preserving the AMPL contract), and the
remaining minima, ranked by objective, to siblings `<stub>.min001.sol`,
`<stub>.min002.sol`, … (so `min001` is the second-best point).

**JSON report.** `--json-output` writes the standard single-solve report for
the best minimum, plus a backward-compatible `minima` section:

```json
"minima": {
  "method": "multistart",
  "status": "target_reached",
  "n_solves": 17,
  "n_minima": 6,
  "minima": [{ "x": [...], "objective": -1.0316 }, ...],
  "values": [-1.0316, -1.0316, -0.2155, ...]
}
```

Omitting `--minima` leaves the default single-solve output completely
unchanged.

## Example

The six-hump camel function has six local minima (two global at
`f ≈ −1.0316`). Searching for all of them from an `.nl` model:

```console
$ pounce sixhump.nl --minima multistart --n-minima 6 \
        --max-solves 120 --patience 40 --dedup 1e-3 --seed 0
```

## References

* Ge, R. "A filled function method for finding a global minimizer of a
  function of several variables." *Mathematical Programming* **46**, 191–204
  (1990).
* Rinnooy Kan, A.H.G. & Timmer, G.T. "Stochastic global optimization methods
  part II: Multi level methods." *Mathematical Programming* **39**, 57–78
  (1987).
* Levy, A.V. & Montalvo, A. "The tunneling algorithm for the global
  minimization of functions." *SIAM J. Sci. Stat. Comput.* **6**(1), 15–29
  (1985). [doi:10.1137/0906002](https://doi.org/10.1137/0906002).
* Wales, D.J. & Doye, J.P.K. "Global optimization by basin-hopping and the
  lowest energy structures of Lennard-Jones clusters containing up to 110
  atoms." *J. Phys. Chem. A* **101**(28), 5111–5116 (1997).
* Farrell, P.E., Birkisson, Á. & Funke, S.W. "Deflation techniques for finding
  distinct solutions of nonlinear partial differential equations." *SIAM J.
  Sci. Comput.* **37**(4), A2026–A2045 (2015).
  [doi:10.1137/140984798](https://doi.org/10.1137/140984798).
