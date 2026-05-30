# LP / QP Solver Routing

POUNCE can route **linear programs (LP)** and **convex quadratic
programs (QP)** to a specialized interior-point solver
(`pounce-convex`) instead of the general nonlinear (NLP) filter-IPM.
The specialized path uses Mehrotra predictor-corrector and reaches the
solution in materially fewer iterations on these problem classes —
typically 30–50% fewer than the general NLP path on bound- or
inequality-constrained convex QPs.

Routing is **automatic and transparent**: you do not change how you
call POUNCE. The same `pounce problem.nl`, the same
`SolverFactory('pounce')` in Pyomo, and the same AMPL `solve` all work
unchanged — POUNCE inspects the problem and picks the solver.

## How routing works

When POUNCE loads a problem it classifies it into one of:

| Class            | Routed to                              |
|------------------|----------------------------------------|
| **LP**           | convex IPM (`pounce-convex`)           |
| **convex QP**    | convex IPM (`pounce-convex`)           |
| **convex QCQP**  | NLP filter-IPM *(conic solver: future)*|
| **nonconvex QP** | NLP filter-IPM (finds a local minimum) |
| **NLP**          | NLP filter-IPM                         |

The classifier is **conservative**: a problem is sent to the convex
solver only when POUNCE can *prove* it is an LP or a convex QP (the
objective is a degree-≤2 polynomial with a positive-semidefinite
Hessian and the constraints are linear). Anything it cannot prove
convex — transcendental terms, an indefinite Hessian, quadratic
constraints — falls back to the general NLP solver, which always
produces a correct (locally optimal) answer. You never get a wrong
"optimum" from a misclassification.

> **Note on QP detection.** The AMPL `.nl` format has no dedicated
> quadratic section: a QP's quadratic terms are written into the
> nonlinear expression tree. POUNCE walks that tree to recover the
> Hessian and test convexity, the same way QP-capable AMPL solvers do.

## Choosing the solver explicitly

The `solver_selection` option overrides the automatic choice. It is a
normal POUNCE option, so it works on the command line, in an options
file, or through Pyomo's `solver.options`.

| Value           | Behavior                                                            |
|-----------------|---------------------------------------------------------------------|
| `auto`          | **Default.** Route by detected class (table above).                 |
| `nlp`           | Always use the NLP filter-IPM, regardless of class.                 |
| `lp-ipm`        | Force the convex IPM; **errors** if the problem is not an LP.        |
| `qp-ipm`        | Force the convex IPM; **errors** if the problem is not LP/convex-QP. |
| `qp-active-set` | Reserved for the active-set QP track; currently falls back to NLP.  |

```sh
# Let POUNCE decide (default):
pounce model.nl

# Force the NLP path even on a convex QP (e.g. to compare):
pounce model.nl solver_selection=nlp

# Insist the problem is a convex QP — fail loudly if it is not:
pounce model.nl solver_selection=qp-ipm
```

A forced value that does not match the detected class is rejected with
a clear message rather than silently ignored:

```text
pounce: problem class NLP does not match forced solver qp-ipm
        (expected an LP or convex QP)
```

### From Pyomo

```python
solver = SolverFactory('pounce')
solver.options['solver_selection'] = 'qp-ipm'   # or 'auto', 'nlp', ...
solver.solve(model)
```

## What you get back

The convex IPM reports the same way as the NLP path: an optimal-status
banner, the objective value (in your original sense — a `maximize`
objective and any constant term are reported correctly), and a `.sol`
file with the primal solution when one is requested.

```text
POUNCE (convex QP IPM, pounce-convex): Optimal Solution Found.
        obj=2.00000000  iters=2
```

## Scope and limitations

- **Convex QP only.** Nonconvex (indefinite-Hessian) QPs are solved by
  the NLP path to a *local* minimum; POUNCE does not do global
  optimization.
- **Convex QCQP** (quadratic constraints) is detected as its own class
  but currently routes to the NLP path; a second-order-cone solver is
  planned.

Both the primal solution and the constraint duals are written to the
`.sol` file, in the same sign convention as POUNCE's NLP path (so Pyomo
and AMPL read them identically regardless of which solver ran).

### Infeasible and unbounded problems

The convex solver detects infeasibility and unboundedness directly,
reporting a clean status instead of exhausting the iteration budget:

- **Primal infeasible** — no point satisfies the constraints. Reported
  with AMPL `solve_result_num` 200.
- **Unbounded** (dual infeasible) — the objective decreases without
  bound along a feasible direction. Reported with `solve_result_num`
  300.

Each verdict is backed by a *verified* certificate (a Farkas
infeasibility proof or an unbounded recession direction that is checked,
not merely inferred), so these statuses are never reported in error; a
problem the solver cannot certify simply runs to the iteration limit.

The design and roadmap live in
[`dev-notes/lp-qp-routing.md`](https://github.com/jkitchin/pounce/blob/main/dev-notes/lp-qp-routing.md).
