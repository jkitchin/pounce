# LP / QP Solver Routing

POUNCE can route **linear programs (LP)**, **convex quadratic
programs (QP)**, and **convex quadratically-constrained QPs (QCQP)** to a
specialized interior-point solver (`pounce-convex`) instead of the general
nonlinear (NLP) filter-IPM. The specialized path uses Mehrotra
predictor-corrector and reaches the solution in materially fewer iterations
on these problem classes — typically 30–50% fewer than the general NLP path
on bound- or inequality-constrained convex QPs.

Routing is **automatic and transparent**: you do not change how you
call POUNCE. The same `pounce problem.nl`, the same
`SolverFactory('pounce')` in Pyomo, and the same AMPL `solve` all work
unchanged — POUNCE inspects the problem and picks the solver.

## How routing works

When POUNCE loads a problem it classifies it into one of:

| Class            | Routed to                                  |
|------------------|--------------------------------------------|
| **LP**           | convex IPM (`pounce-convex`)               |
| **convex QP**    | convex IPM (`pounce-convex`)               |
| **convex QCQP**  | conic IPM (`pounce-convex`, SOCP)          |
| **nonconvex QP** | NLP filter-IPM (finds a local minimum)     |
| **NLP**          | NLP filter-IPM                             |

The classifier is **conservative**: a problem is sent to the convex
solver only when POUNCE can *prove* it is convex — an LP or convex QP
(degree-≤2 objective with a positive-semidefinite Hessian, linear
constraints), or a convex QCQP (additionally allowing convex-quadratic
inequality constraints, each with a positive-semidefinite Hessian and a
one-sided `≤` bound, which are reformulated to second-order cones).
Anything it cannot prove convex — transcendental terms, an indefinite
objective Hessian, a quadratic *equality*, or a quadratic inequality whose
feasible set is nonconvex — falls back to the general NLP solver, which
always produces a correct (locally optimal) answer. You never get a wrong
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
| `socp`          | Force the conic IPM; **errors** if the problem is not a convex QCQP. |
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

Before solving, POUNCE prints a one-line **routing banner** naming the
detected class, the solver it selected, and the effective
`solver_selection` — so it is always clear which of POUNCE's solvers ran
and why:

```text
Problem class: LP. Selected solver: convex QP interior-point (pounce-convex) [solver_selection=auto].
```

(The banner is suppressed alongside the startup banner — `sb yes` or
JSON-debug protocol mode — to keep stdout clean for machine consumers.)

The convex IPM then reports the same way as the NLP path: an
optimal-status line, the objective value (in your original sense — a
`maximize` objective and any constant term are reported correctly), and a
`.sol` file with the primal solution when one is requested.

```text
POUNCE (LP IPM, pounce-convex): Optimal Solution Found.
        obj=2.00000000  iters=2
```

> **Driver.** The convex path uses the **homogeneous self-dual embedding
> (HSDE)** interior-point driver — the same self-dual formulation
> Clarabel/ECOS use. It is self-starting, returns verified
> infeasibility/unboundedness certificates, and conditions the KKT system
> internally through its per-cone scaling, so it solves even badly-scaled
> LPs (e.g. NETLIB `nl`, `‖c‖ ~ 1e6`) without external pre-scaling.

## Presolve

Before the convex interior-point solve, POUNCE runs a **presolve** pass
that shrinks the problem and can detect trivial infeasibility or
unboundedness without solving. It removes empty, duplicate, and
activity-redundant rows; fixes and substitutes structural columns
(singleton-row fixings, free columns, free column singletons); and
recovers both the primal and dual of the eliminated pieces so the
reported solution is for your original problem. When it reduces the
model, it logs a one-line summary:

```text
Presolve: 40 → 32 vars, 12 → 8 rows (fixed 3, free-fixed 2, substituted 3)
```

Presolve is on by default. Turn it off with `qp_presolve=no` (e.g. to
compare timings or isolate a solver issue):

```sh
pounce model.nl qp_presolve=no
```

## Scope and limitations

- **Convex problems only.** Nonconvex (indefinite-Hessian) QPs, quadratic
  equalities, and quadratic inequalities whose feasible set is nonconvex are
  solved by the NLP path to a *local* minimum; POUNCE does not do global
  optimization.
- **Convex QCQP** (convex-quadratic constraints) routes to the conic IPM:
  each convex-quadratic inequality `½xᵀQx + aᵀx + b ≤ 0` (with `Q ⪰ 0`) is
  reformulated to one second-order cone (`Q = FᵀF`, so `‖Fx‖² = xᵀQx`) and
  solved alongside the QP objective and linear constraints.

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
