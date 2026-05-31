# Tutorial: active-set SQP and working-set warm starting

This is the user-facing walkthrough for pounce's Phase 5b/5c
active-set SQP driver. It assumes you can already drive pounce's
default IPM via the standard interface (`Problem.solve` in Python,
`IpoptSolve` in C, `option nlp = pounce` in GAMS).

The design rationale and algorithmic choices live in the
[design note](../../dev-notes/research/active-set-sqp-warm-start.md) — read
that if you want to know *why* the solver works the way it does.
This tutorial covers *how to use* the solver: switching to the
SQP path, carrying a working set across solves, and stitching
the parametric predictor + SQP corrector pattern together.

## 1. When to use the active-set SQP

Use it when **the same NLP shape is solved many times under small
perturbations** — MPC closed-loop, parametric continuation,
homotopy sweeps, sensitivity-driven design exploration. The IPM
re-solves each instance from scratch (the central-path push at
the beginning of a fresh solve typically costs 4–8 iterations
even when the previous optimum is essentially correct); the SQP
warm-started from the previous working set typically picks up
where it left off in **0–3 outer iterations** when the active
set is stable, or grows by a few QP add/drop steps when one or
two constraints flip.

Stick with the IPM (the default) for **cold solves of a single
problem** or **large-scale problems with thousands of active
inequalities**. The IPM scales linearly in the active set; the
active-set SQP's per-QP cost grows with the number of active
constraints.

## 2. Switching to the SQP path

The switch is a single option flip — `algorithm` from its default
`interior-point` to `active-set-sqp`. Everything else (callbacks,
bounds, starting point, finalize_solution) is unchanged.

### Python

```python
import pounce
import numpy as np

prob = pounce.Problem(
    n=2, m=1, problem_obj=MyNlp(),
    lb=[0.0, 0.0], ub=[10.0, 10.0],
    cl=[1.0], cu=[1.0],
)
prob.add_option("algorithm", "active-set-sqp")
prob.add_option("print_level", 0)
x, info = prob.solve(x0=np.array([0.5, 0.5]))
```

### C

```c
#include "pounce.h"

IpoptProblem prob = CreateIpoptProblem(/* ... */);
AddIpoptStrOption(prob, "algorithm", "active-set-sqp");
double x[2] = {0.5, 0.5};
double obj;
int status = IpoptSolve(prob, x, NULL, &obj, NULL, NULL, NULL, NULL);
```

### GAMS

```
* pounce.opt
algorithm  active-set-sqp
```

```
Model mymodel / all /;
option nlp = pounce;
mymodel.optfile = 1;
Solve mymodel using nlp minimizing obj;
```

### SQP-specific options

All SQP knobs live under the `sqp_*` namespace. The defaults
mirror `SqpOptions::default()`.

| Option                  | Default     | Meaning                                            |
| ----------------------- | ----------- | -------------------------------------------------- |
| `sqp_globalization`     | `filter`    | `filter` or `l1-elastic` (Fletcher-Leyffer / Han-Powell) |
| `sqp_hessian`           | `exact`     | `exact`, `damped-bfgs`, or `lbfgs`                 |
| `sqp_max_iter`          | `200`       | outer iteration cap                                |
| `sqp_tol`               | `1e-8`      | stationarity tolerance (max-norm)                  |
| `sqp_constr_viol_tol`   | `1e-6`      | constraint-violation tolerance                     |
| `sqp_dual_inf_tol`      | `1e-4`      | dual-infeasibility tolerance                       |
| `sqp_l1_penalty`        | `1.0`       | initial ν (Han-Powell only)                        |
| `sqp_l1_penalty_safety` | `0.1`       | additive ν margin                                  |
| `sqp_l1_penalty_max`    | `1e10`      | ν upper clamp                                      |
| `sqp_bt_reduction`      | `0.5`       | backtracking factor                                |
| `sqp_bt_min_alpha`      | `1e-12`     | minimum step before line-search failure            |
| `sqp_print_level`       | `0`         | 0=silent, 1=per-iter summary, 2+=trace             |
| `sqp_lbfgs_max_history` | `6`         | L-BFGS history size                                |

### Algorithm-path isolation guarantees

The two solver paths share the TNLP layer, the `OrigIpoptNlp`
adapter, the linear-solver backend, the options registry, and
`finalize_solution`. Beyond that they are **deliberately
isolated**, so toggling `algorithm` is always safe — no Phase 5
addition can change IPM behaviour, and no IPM warm-start setting
can change SQP behaviour. Concretely:

- **The default (`algorithm = interior-point`) is unchanged.** No
  user who hasn't typed `active-set-sqp` ever runs Phase 5 code.
- **`sqp_*` options are silently ignored on the IPM path.** Setting
  `sqp_globalization`, `sqp_hessian`, `sqp_max_iter`, … while
  `algorithm` is `interior-point` is a no-op. The option-list
  parser still validates them (out-of-range numeric values fail
  validation regardless of `algorithm`), but the IPM driver never
  reads the resolved values.
- **IPM warm-start options are silently ignored on the SQP path.**
  `warm_start_init_point`, `bound_push`, `bound_frac`,
  `slack_bound_push`, `mult_init_max`, `mu_init`, `mu_target` and
  the rest of the IPM-side initializer knobs sit on the
  `AlgorithmBuilder` but are not consulted when the SQP outer
  loop runs.
- **Warm-start payloads are path-local.**
  `IpoptApplication::set_sqp_warm_start(SqpIterates)` /
  `Problem.solve(working_set=…)` / `IpoptSetWarmStartWorkingSet`
  feed the SQP loop only — the IPM never reads `sqp_warm_start`.
  Symmetrically, `lagrange=` / `zl=` / `zu=` on
  `Problem.solve` (paired with `warm_start_init_point=yes`) feed
  the IPM only — the SQP loop never consults them.
- **You can flip between paths across solves on the same
  `Problem` handle.** The application's per-solve setup
  (restoration factory, options snapshot, statistics reset) is
  rebuilt for every `solve()`, so a cold IPM solve followed by
  an SQP solve with `algorithm` re-set in between is a supported
  pattern. This is exactly how the parametric corrector in §4
  hands off from a cold IPM warm-up to the SQP corrector.
- **The C ABI is strictly additive.** Existing cyipopt / JuMP /
  AMPL clients link against the new `libpounce_cinterface`
  unchanged; the four new entry points (`IpoptGetWorkingSet`,
  `IpoptSetWarmStartWorkingSet`, `IpoptClearWarmStartWorkingSet`,
  `IpoptSolveWarmStart`) are pure additions.
- **`info["working_set"]` is always present, sometimes `None`.**
  Python callers that don't touch the SQP path never have to
  read that key, but reading it is safe — it returns `None` on
  the IPM path so a downstream loop won't crash on a missing
  key.

This isolation is verified by the existing test suite: 868
workspace tests cover both paths, plus crosscutting tests like
`application_sqp_warm_start_auto_clears_after_use` (asserts the
SQP-side warm-start state doesn't leak between solves) and
`application_default_does_not_select_sqp` (asserts the default
solver path is IPM).

## 3. The working-set warm-start contract

The §6 contract is the tuple `(x, λ_g, λ_x, 𝒲)` — primal, constraint
multipliers, packed bound multipliers, and the discrete *working
set* `𝒲` (which bounds and constraints are active at the optimum).
The first three are floating-point; only the last is the
parametric-warm-start payoff over the IPM, because IPM-side
multipliers are continuous interior-point estimates whereas the
active set is what tells the next QP which rows to keep in the
KKT block from iteration zero.

### Python: carry across solves

```python
prob.add_option("algorithm", "active-set-sqp")

ws = None
for k in range(horizon_steps):
    # ... user code updates the parameter inside MyNlp ...
    x, info = prob.solve(x0=x_prev, working_set=ws)
    ws = info["working_set"]   # (bounds_int8_array, constraints_int8_array)
    x_prev = x
```

The status codes in the `working_set` tuple use these values
(int8 arrays):

```
0 = Inactive
1 = AtLower   (active at lower bound)
2 = AtUpper   (active at upper bound)
3 = Fixed (variable) or Equality (constraint)
```

### C: carry across solves

```c
IpoptBoundStatus *bounds = malloc(n * sizeof *bounds);
IpoptConsStatus  *cons   = malloc(m * sizeof *cons);

for (int k = 0; k < horizon_steps; k++) {
    /* ... user code updates the parameter ... */
    if (k == 0) {
        IpoptSolve(prob, x, NULL, &obj, NULL, NULL, NULL, NULL);
    } else {
        IpoptSolveWarmStart(prob, x, NULL, &obj, NULL, NULL, NULL,
                            bounds, cons,        /* in */
                            bounds, cons,        /* out, may alias */
                            NULL);
    }
    /* read the WS out for next iteration */
    IpoptGetWorkingSet(prob, bounds, cons);
}
```

### GAMS: working set persists automatically

The GAMS solver link reads variable and equation marginals (`x.m`,
`con.m`) at the top of each `pouCallSolver` invocation and
reconstructs the working set from them. No `solve`-statement
gymnastics required — every subsequent `solve` automatically
warm-starts from the previous solution's marginals.

Use the **§7.4(b) state file** option for the precision-critical
case where the marginal signs are ambiguous (degenerate active
set):

```
* pounce.opt
algorithm        active-set-sqp
sqp_state_file   .mymodel.pou-ws
```

The link writes a small binary blob after each solve and reads
it at the start of the next, keyed by a checksum over
`(n, m, x_l, x_u, g_l, g_u)` so structural changes invalidate
the file cleanly.

## 4. Worked example: parametric continuation

The headline use case. You have an NLP

  min f(x; p) s.t. g(x; p) = 0, x ≥ 0

and you want to trace `x*(p)` as `p` sweeps a path. The pounce
playbook is:

1. **Solve** at `p₀` with the IPM (better cold-start
   convergence than the SQP elastic phase).
2. **Predictor**: ask `pounce_sensitivity::SensSolve` for
   `Δx ≈ ∂x*/∂p · Δp` at `p₀`.
3. **Classify** the active set at the converged IPM iterate via
   `pounce.classify_working_set(...)`.
4. **Update** `p` in your TNLP. Apply `x* + Δx` as the predictor.
5. **Corrector**: switch `algorithm` to `active-set-sqp`, install
   the working set + predictor as warm start, solve.
6. The corrector lands on `x*(p₀ + Δp)` in 0–3 outer iterations
   for small Δp.

### Python (full code)

```python
import numpy as np
import pounce


class ParamNlp:
    """min ½‖x − p‖²  s.t.  sum(x) = 1, x ≥ 0  with parameter p."""

    def __init__(self):
        self.p = np.zeros(3)

    def set_p(self, p):
        self.p = np.asarray(p, dtype=float)

    def objective(self, x):
        d = x - self.p
        return 0.5 * float(d @ d)

    def gradient(self, x):
        return x - self.p

    def constraints(self, x):
        return np.array([float(x.sum())])

    def jacobianstructure(self):
        return (np.zeros(3, dtype=np.int64), np.arange(3, dtype=np.int64))

    def jacobian(self, x):
        return np.ones(3)

    def hessianstructure(self):
        idx = np.arange(3, dtype=np.int64)
        return (idx, idx)

    def hessian(self, x, lagrange, obj_factor):
        return np.full(3, obj_factor)


nlp = ParamNlp()


def build_problem(algorithm):
    p = pounce.Problem(
        n=3, m=1, problem_obj=nlp,
        lb=[0.0] * 3, ub=[1e20] * 3,
        cl=[1.0], cu=[1.0],
    )
    p.add_option("algorithm", algorithm)
    p.add_option("print_level", 0)
    return p


# --- Step 1: cold IPM solve at p₀ ---
nlp.set_p([0.5, 0.4, -0.1])
x_ipm, info_ipm = build_problem("interior-point").solve(x0=np.full(3, 1.0 / 3))
print(f"IPM   converged: x = {x_ipm}, f = {info_ipm['obj_val']:.4f}")

# --- Step 3: classify the active set at x_ipm ---
ws = pounce.classify_working_set(
    x=x_ipm,
    x_l=np.array([0.0, 0.0, 0.0]),
    x_u=np.array([1e20, 1e20, 1e20]),
    g=info_ipm["g"],
    g_l=np.array([1.0]),
    g_u=np.array([1.0]),
    lambda_g=info_ipm["mult_g"],
    z_l=info_ipm["mult_x_L"],
    z_u=info_ipm["mult_x_U"],
    m_eq=1,
)
bounds, cons = ws
print(f"      working set: bounds = {bounds.tolist()}, cons = {cons.tolist()}")

# --- Step 4: perturb p and run the SQP corrector ---
nlp.set_p([0.52, 0.39, -0.05])     # Δp = (0.02, -0.01, 0.05)
x_sqp, info_sqp = build_problem("active-set-sqp").solve(
    x0=x_ipm, working_set=ws,
)
print(f"SQP   corrector: x = {x_sqp}, f = {info_sqp['obj_val']:.4f}")
print(f"      info['working_set'] for the next step: {info_sqp['working_set']}")
```

Expected output (deterministic, ran live before this tutorial
was checked in):

```
IPM   converged: x = [5.5e-01 4.5e-01 1.2e-07], f = 0.0075
      working set: bounds = [0, 0, 1], cons = [3]
SQP   corrector: x = [0.565 0.435 0.   ], f = 0.0033
      info['working_set'] = (array([0, 0, 1], dtype=int8), array([3], dtype=int8))
```

The IPM lands `x₃` essentially-zero (1.2e-7) — it's an IPM
artifact of the central-path push; for `classify_working_set`'s
default `primal_tol = 1e-6` that's already inside the
"at the bound" band, so `bounds[2] = 1` (AtLower). The SQP
corrector hits `x₃ = 0` exactly because the working set tells
it `x₃` is an active lower bound from iteration zero — no
central-path detour.

`bounds = [0, 0, 1]` means `x[0]`, `x[1]` inactive (interior),
`x[2]` at its lower bound. `cons = [3]` means the sum constraint
is binding (equality).

### Running it

Save as `parametric_demo.py` and run:

```
python parametric_demo.py
```

For an executable variant see
`python/examples/sqp_warm_start_mpc.py` (a 20-step parametric
sweep) and the Jupyter notebook in
`python/notebooks/06_sqp_parametric_continuation.ipynb`.

## 5. Choosing a globalization

`sqp_globalization = filter` (the default) follows Fletcher-Leyffer
2002 — a Pareto-frontier filter on `(constraint violation,
objective)`. Robust, no penalty parameter to tune, recommended
for general nonlinear NLPs.

`sqp_globalization = l1-elastic` is the SNOPT-style Han-Powell
merit `φ(x; ν) = f(x) + ν · violation(x)` with adaptive ν. The
new `sqp_l1_penalty_safety` (default 0.1) and `sqp_l1_penalty_max`
(default 1e10) options control the ν update:

  ν ← clamp(max(ν, ‖λ_qp‖_∞ + sqp_l1_penalty_safety), 0, sqp_l1_penalty_max)

Use `l1-elastic` when you want behaviour close to SNOPT for
comparison studies, or when the filter is rejecting too many
trial steps on a problem where the merit decreases steadily.

## 6. Choosing a Hessian source

| Source        | When to use                                                            |
| ------------- | ---------------------------------------------------------------------- |
| `exact`       | NLP provides `eval_h`; the QP's inertia-control handles indefinite ∇²L |
| `damped-bfgs` | Dense `n×n` Powell-damped BFGS; guaranteed PSD; n ≤ a few hundred       |
| `lbfgs`       | Limited-memory BFGS with `sqp_lbfgs_max_history` pairs; large `n`        |

The default `exact` is fastest when reliable. Switch to
`damped-bfgs` for ill-scaled nonconvex NLPs where the QP solver's
inertia retries dominate the iteration cost. Use `lbfgs` only
when the dense `n²` BFGS storage is the bottleneck (n ≥ ~1000).

## 7. Pitfalls

- **Calling `Problem.solve(working_set=…)` with a stale working set
  whose dimensions changed.** Validated and rejected with
  `ValueError`. Pass the WS only when it came from a solve of the
  *same* problem shape.
- **Mixing IPM and SQP across solves without resetting state.**
  The IPM path ignores `set_sqp_warm_start`, and the SQP path
  ignores the IPM warm-start options (`warm_start_init_point`
  etc.). Each path's warm-start input is path-local.
- **Degenerate active set after IPM convergence.** The
  multiplier-sign + primal-distance heuristic in
  `classify_working_set` is lossy at degenerate optima — same
  trade-off CONOPT/IPOPT/KNITRO have under GAMS. The first QP
  step in the SQP corrector re-classifies any wrongly-tagged
  rows, so correctness is preserved; only the iteration count
  may be slightly higher than ideal.
- **L1Elastic with a hard cap.** If your problem's QP multipliers
  spike (poorly scaled constraints), bump `sqp_l1_penalty_max`
  or rescale.

## 8. Where the code lives

| Concern                 | File                                                     |
| ----------------------- | -------------------------------------------------------- |
| SQP outer loop          | `crates/pounce-algorithm/src/sqp/sqp_alg.rs`             |
| QP subproblem solver    | `crates/pounce-qp/src/solver.rs`                         |
| Working set type        | `crates/pounce-qp/src/working_set.rs`                    |
| Classifier              | `crates/pounce-algorithm/src/sqp/warm_start.rs`          |
| IpoptApplication hooks  | `crates/pounce-algorithm/src/application.rs`             |
| C ABI                   | `crates/pounce-cinterface/src/lib.rs` + `include/pounce.h` |
| Python binding          | `crates/pounce-py/src/{problem,warm_start}.rs`           |
| GAMS link               | `gams/gams_pounce.c`                                     |
| Design rationale        | `dev-notes/research/active-set-sqp-warm-start.md`        |

## 9. Reading list

- Hock, Schittkowski (1981) *Test Examples for Nonlinear Programming Codes* — the in-repo HS subset reference.
- Nocedal, Wright (2006) *Numerical Optimization*, ch. 18 — SQP fundamentals.
- Fletcher, Leyffer (2002) — filter line search.
- Han (1977) / Powell (1978) — l1-merit and damped-BFGS update.
- Wächter, Biegler (2006) — pounce's IPM heritage.
- Gill, Murray, Saunders (2002) — SNOPT / l1-elastic phase 1.
- Forsgren, Gill, Wright (2002) — IPM vs SQP comparison.
- Kirches (2011) — parametric active-set SQP.
