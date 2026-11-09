# Diagnosing a Model

```sh
pounce diagnose <problem.nl> [--x0-file x0.txt | --sol result.sol] [--json]
```

`pounce diagnose` checks a model **without solving it**. It reads the `.nl`,
proves what the algebra and bounds alone can prove, then evaluates the model
at one point and reports what a solve from there would run into. Each finding
says what kind of evidence it rests on, so a contradiction proved from the
bounds is never confused with a rank estimate at one point.

Use it when a solve ended in `Infeasible_Problem_Detected`,
`Invalid_Number_Detected` or `Restoration_Failed`, when a model is new, or
before you hand a model to someone else. It is read-only: it never changes the
model, and it never runs the solver.

## What it checks

The checks run in this order. The first group needs no point; the second runs
at one point.

| Check | Evidence | What it finds |
|---|---|---|
| `bounds` | proof | variable or row bounds that are crossed or NaN |
| `linear_rows` | proof | rows whose body is affine: constant rows that can never hold; single-variable rows that are really bounds, and contradict the variable's bounds or each other; parallel rows that contradict each other or duplicate an equality |
| `incidence` | proof | variables that appear nowhere; objective terms that improve without limit along a variable no constraint touches |
| `expression_domains` | proof / heuristic | operators (`log`, `sqrt`, `/`, fractional powers, `acos`, …) whose argument the bounds do not keep inside the operator's domain; `atan(y/x)` where `x` can be 0 |
| `row_enclosures` | proof | rows that cannot take a value inside their bounds anywhere in the variable box (`x^2 == -1`, `exp(x) <= 0`) |
| `coefficient_ranges` | proof | ranges of linear coefficients, quadratic curvature, right-hand sides and finite bounds, with the extremes named; wide ranges; finite bounds so large they stand in for "no bound" |
| `start_point` | numerical | everything `pounce check-x0` reports (bound violations, interior-clamp moves, initial violation, Hessian NaN/Inf); its full report is embedded as `sections.check_x0` |
| `nonfinite_localization` | numerical | for every row or derivative that is NaN/Inf at the point, the operator that produced it from finite inputs, and why (outside the domain, a pole, overflow) |
| `structural_rank` | structural | equality rows that cannot be matched to distinct free variables (Dulmage–Mendelsohn over-determined block) |
| `numerical_rank` | numerical | the rank of the equality Jacobian at the point, and the **smallest sets of dependent equations** |
| `jacobian_scaling` | numerical | spread of the Jacobian's row and column scales at the point |

## Reading a finding

```text
  [WARNING] dependent_equality_rows  (numerical, high confidence)
    At the model's starting point, equalities bal_a, bal_b, bal_total are
    linearly dependent, and removing any one of them leaves the rest
    independent (a minimal dependent set). The structural rank does not show
    this loss, so it comes from the values at this point.
    why: LICQ fails at this point: ...
    next: Check whether one of these equations is implied by the others; ...
```

Every finding carries:

- **`code`** — a stable identifier you can filter on.
- **`severity`** — `error` (the model as written cannot be solved, or a solve
  from this point aborts), `warning` (likely trouble), `info`.
- **`basis`** — what the finding rests on:
  - `proof`: follows from the algebra and bounds; true at every point.
  - `structural`: follows from the sparsity pattern; also true at every point.
  - `numerical`: observed at the named point, under the stated tolerance.
    Says nothing about other points.
  - `heuristic`: a pattern that often signals trouble but proves nothing.
- **`confidence`**, **`why_it_matters`**, **`suggested_next_check`** — the
  next step is an investigation, never an instruction to change the model.
- **`affected`** variables and constraints, by name, and **`evidence`**:
  intervals, values, the point and the tolerance used.

A check that could not run is listed under "checks that did not run" with the
reason (for example, an equality Jacobian too large for a dense SVD), so an
empty report can be told apart from an unchecked one.

## Choosing the point

The numerical checks evaluate at the model's starting point unless you give
another one:

- `--x0-file x0.txt` — `n` whitespace-separated values.
- `--sol result.sol` — the primal values of an AMPL `.sol`, for example the
  point a failed solve stopped at.

Comparing the two is often the fastest diagnosis: a rank loss present at both
the start and the final point is likely a property of the formulation; one
present at only one of them is a property of that point.

## Possible versus certain domain violations

The domain checks enclose each operator's argument in an interval over the
variable bounds. Interval arithmetic over-approximates (it cannot see that
`x - x` is zero), so:

- an argument whose enclosure lies **entirely** outside the domain is a
  proof: every point in the box fails (`expression_domain_violation`);
- an enclosure that only **reaches** outside is a risk, not a certainty
  (`possible_expression_domain_violation`): the bounds do not exclude it;
- an enclosure that **touches** the edge — `log(x)` with `x >= 0` — is
  reported too (`domain_boundary_reachable`), because the solver relaxes
  bounds by `bound_relax_factor` (default 1e-8) and can evaluate just past
  them.

Operators inside an `if-then-else` branch are reported at reduced severity,
because the condition may guard them.

## The smallest dependent sets

When the equality Jacobian is rank deficient at the point, `diagnose` does not
stop at "rank deficient". It takes the rows that take part in the null space
and removes them one at a time, keeping a removal only when the remaining rows
are still dependent, all under one fixed singular-value threshold. What is
left is **inclusion-minimal**: removing any one of its rows leaves the rest
independent. The evidence gives the combination of rows that cancels. This is
the irreducible degenerate set idea of Dowling and Biegler's *Degeneracy
Hunter*, computed numerically at one point and one threshold. If the rank
deficiency is larger than one, up to `--max-dependent-sets` sets are found.

When the numerical rank is below the structural rank
(`numerical_rank_below_structural_rank`), the dependence comes from the values
at this point, not from the sparsity pattern, and may disappear elsewhere.

The singular-value threshold defaults to `max(m, n)·ε·σ_max`, the
LAPACK/NumPy convention, on the **unscaled** Jacobian; change it with
`--rank-tol`. Because the SVD is dense it is skipped (and the check reported
as not run) above `--max-dense-entries` / `--max-dense-dim`.

## Exit code and JSON

The exit code is `0` when no finding is an error, `22` when at least one is,
and `2` on a usage or I/O error. `--json` prints the report as
`pounce.diagnose/v1`; `--json-output <path>` writes it to a file as well. The
JSON keeps every finding (up to `--max-per-code` per code; the number dropped
is in `truncated`) and the supporting tables under `sections`.

From Python:

```python
import pounce
report = pounce.diagnose("model.nl")
for f in report.errors():
    print(f["code"], f["observation"])
```

## What it does not do

- It does not decide what the model *means*. A proved contradiction says the
  rows as written cannot all hold; which row is wrong is your call.
- A numerical finding is about one point. Rank, scaling and NaN findings at
  the start say nothing about the solution, and vice versa.
- It does not search for a minimal infeasible subsystem (IIS); see discopt.
- Constraint-qualification checks at a solution (LICQ/MFCQ with multipliers),
  rank robustness across tolerances and points, and convexity evidence are
  planned (gh#1160, phase 2).

The idea list draws on [NLPDiagnostics.jl](https://github.com/frederikgeth/NLPDiagnostics.jl)
and the JuMP model-debugging initiative; the dependent-set search on
Dowling and Biegler's *Degeneracy Hunter*.
