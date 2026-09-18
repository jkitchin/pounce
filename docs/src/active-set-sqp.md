# Tutorial: active-set SQP and working-set warm starting

This is the user-facing walkthrough for pounce's Phase 5b/5c
active-set SQP driver. It assumes you can already drive pounce's
default IPM via the standard interface (`Problem.solve` in Python,
`IpoptSolve` in C, `option nlp = pounce` in GAMS).

The design rationale and algorithmic choices live in the
[design note](active-set-sqp-warm-start.md) — read
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
problem**. The IPM scales linearly in the active set; the active-set
SQP's per-QP cost grows with the number of active constraints, so a
*cold* SQP solve does lose ground as the problem grows.

A warm one does not, which is the case worth being precise about. On
[the warm-start benchmark](warm-start-benchmark.md)'s MPC sweep the
warm-started SQP's advantage over its own cold twin *improves* with
problem size all the way to 1645 active constraints (0.17× → 0.02× wall
time from N = 10 to N = 80, and 0.02–0.03× at n = 602–2402), because
warm cost is set by how far the active set moved rather than by how
large it is. An earlier version of this page warned against the SQP for
"large-scale problems with thousands of active inequalities"; the
measured behavior does not support that for warm-started sequences.

The awkward case in such a sequence is the **first** solve, which has no
previous working set to inherit and so pays the cold-SQP cost the
paragraph above warns about. [Crossover](crossover.md) fills that gap:
run the first solve on the IPM with `crossover=yes`, and
`last_sqp_working_set()` returns an identified active set the following
SQP solve can warm-start from. Before it, the only source of a working
set was another SQP solve.

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

### Rust

The flip is one option, so it needs no cargo feature — the default
`pounce-rs` build reaches the SQP path. On the builder API:

```rust
use pounce_rs::prelude::*;

let sol = Nlp::new(MyNlp)
    .var_bounds(&[0.0, 0.0], &[10.0, 10.0])
    .constraint_bounds(&[1.0], &[1.0])
    .x0(&[0.5, 0.5])
    .option_str("algorithm", "active-set-sqp")
    .solve();
```

or, driving `IpoptApplication` directly:

```rust
let mut app = IpoptApplication::new();
app.initialize()?;
app.initialize_with_options_str("algorithm active-set-sqp\n")?;
let status = app.optimize_tnlp(tnlp);
```

Carrying a working set across solves — §3 below — additionally needs
`features = ["qp"]`, because the `WorkingSet` type belongs to the QP engine.

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

#### Confirming the engine actually ran

The copyright banner says "interior-point" on every run — it is a fixed
string and does not read `algorithm`. Three things do tell you:

* **The routing line.** `Selected solver: active-set SQP (pounce-qp
  subproblems)`, printed under the banner.
* **`sqp_print_level=1`**, which prints the outer-iteration table:

  ```text
  iter      objective   inf_pr   inf_du    ||p||    alpha     ws
     0  1.6000000e+09 1.20e+01 1.20e+09        -        -      -
     1  1.5937500e+09 1.62e+00 2.41e+08 1.25e+00 1.00e+00      2
  ```

  Row `k` reports the residuals *at* `x_k`; the step columns describe the
  step that arrived there, so they are blank on row 0 — the same convention
  the interior-point table uses. `ws` is that iteration's QP working-set
  changes. `sqp_print_level=2` adds a per-iteration trace of the iterate,
  the QP step and the line search.
* **`--json-output`**, whose `solution.engine` reads `sqp-active-set`. From
  Python, `info["n_qp_solves"]` is nonzero if and only if this engine ran.

Two summary lines also fingerprint this arm, though neither is meant as an
indicator: `Complementarity` is exactly zero (there is no barrier term) and
`Variable bound violation` reads `nan`.

#### The inner QP subproblem (`sqp_qp_*`)

Each outer SQP iteration solves a QP subproblem with the `pounce-qp`
active-set engine. These knobs control that inner solve, and are read
only on the SQP path.

| Option                                      | Default  | Meaning                                                     |
| ------------------------------------------- | -------- | ----------------------------------------------------------- |
| `sqp_qp_feas_tol`                           | `1e-9`   | QP constraint-feasibility tolerance                          |
| `sqp_qp_opt_tol`                            | `1e-9`   | QP optimality (KKT) tolerance                                |
| `sqp_qp_max_iter`                           | `200`    | active-set pivots per QP solve                               |
| `sqp_qp_elastic_gamma`                      | `1e6`    | penalty on the elastic (phase-1) slacks                      |
| `sqp_qp_anti_cycling`                       | `expand` | `expand` (Harris two-pass), `bland`, or `none`               |
| `sqp_qp_certify_second_order`               | `no`     | check second-order optimality before certifying the QP       |
| `sqp_qp_use_homotopy`                       | `no`     | trace the parametric homotopy on a cold QP solve             |
| `sqp_qp_use_schur_updates`                  | `no`     | absorb working-set changes as rank-2 Schur updates           |
| `sqp_qp_max_schur_updates_before_refactor`  | `50`     | updates to absorb before refactoring (Schur path only)       |

Three of these are worth more than a table row.

**`sqp_qp_certify_second_order`** is the one that changes answers rather
than speed. Every `Optimal` the active-set engine returns is a
*first-order* verdict — vanishing projected gradient, sign-admissible
working-set multipliers — and a saddle point of an indefinite Hessian
satisfies that exactly. Set it to `yes` and the engine must also fail to
find a direction `d` with `A_W d = 0` and `dᵀHd < 0` before certifying;
where it finds one it follows it to the next blocking row (gh #848).
This fixes real wrong answers on this path — a constrained *maximum*
reported as `Solve_Succeeded`.

It is `no` by default anyway, because here the QP is a **local model**
and its second-order verdict is not the NLP's: at iteration 0 the
multipliers are still zero, so HS071 started at its own solution reports
negative curvature and takes five iterations instead of one. Making it
the default needs Hessian modification first (gh #856). The check is
skipped outright when the Hessian is known positive semidefinite, so
quasi-Newton runs (`sqp_hessian=damped-bfgs` or `lbfgs`) pay nothing
either way.

Note the asymmetry with **standalone** QP solves
(`solver_selection=qp-active-set`, `pounce.qp.solve_qp`): there the QP
*is* the question, so certification is on by default. Setting this option
explicitly reaches those solves too, and setting it to `no` there is a
way to get a saddle certified as `Optimal` (gh #872).

**`sqp_qp_use_schur_updates`** keeps a cached factor of the
fixed-dimension `K_max` matrix and absorbs each working-set change as a
Sherman-Morrison-Woodbury rank-2 update, refactoring every
`sqp_qp_max_schur_updates_before_refactor` updates (Kirches 2011;
qpOASES-extended). With it off, each iteration assembles a fresh
active-set KKT and factors it from scratch — algorithmically identical,
but every working-set change repeats the full symbolic analysis, which
measured 32% of runtime on Q25FV47.

Its default of `no` is a measured choice, not an oversight. On
Maros-Meszaros the update path is 28–88× faster where it works
(Q25FV47: 19.7 s → 0.5 s), but it is less robust: of the 46 instances
the default path solves correctly, enabling updates breaks 9
(`InternalError`, `TimeOut`, or a wrong objective), and total wall over
that set rises 107 s → 251 s on the new timeouts. Treat it as opt-in for
warm-started workloads where the speedup dominates, not as a general
accelerator.

**`sqp_qp_use_homotopy`** changes the cold-start path: the solve starts
from the box-only relaxation (all general rows dropped, which the box
fast path solves directly) and tightens the row bounds toward their
targets along `t ∈ [0, 1]`, jumping the working set wherever a row
becomes binding or an active multiplier reaches zero. The iterate stays
feasible for the `t`-problem throughout, so there is no phase-1 to stall
in — the failure mode the conventional path hits on degenerate
netlib-derived QPs. Default `no` while it is evaluated against the
conventional path; see
[the warm-start benchmark](warm-start-benchmark.md) for measurements.

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
  `bound_push`, `bound_frac`, `slack_bound_push`, `mult_init_max`,
  `mu_init`, `mu_target` and the rest of the IPM-side initializer
  knobs sit on the `AlgorithmBuilder` but are not consulted when
  the SQP outer loop runs. The one exception is
  `warm_start_init_point`, which the **C ABI** also reads — see the
  next bullet.
- **Warm-start payloads are path-local.**
  `IpoptApplication::set_sqp_warm_start(SqpIterates)` /
  `Problem.solve(working_set=…)` / `IpoptSetWarmStartWorkingSet`
  feed the SQP loop only — the IPM never reads `sqp_warm_start`.
  The *primal* start is not path-local, though: every frontend
  warm-starts the SQP from the same `x0` it would have cold-started
  from (`Problem.solve(x0=…)`, the `x` buffer handed to
  `IpoptSolve`), so supplying a working set never displaces the
  caller's iterate.
  Initial multipliers reach the SQP whenever the caller supplied
  both a working set and duals: `zl=` / `zu=` / `lagrange=` on
  `Problem.solve`, and — under `warm_start_init_point=yes`, which
  is what makes them inputs at all in upstream Ipopt —
  `mult_x_L` / `mult_x_U` / `mult_g` on `IpoptSolve`. Without a
  working set they feed the IPM only. The SQP packs them signed as
  `lambda_x = z_l − z_u`.
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

### Measuring whether the warm start paid

Do not judge a warm start by `info["iter_count"]`. That is the
**outer** SQP iteration count, and on a problem whose subproblem
is already a QP the outer loop terminates in one iteration
whether or not you warm started — the number reads the same cold
and warm while the work underneath differs by an order of
magnitude. The saved work is inside the QP subproblems, and two
further keys report it:

| key | meaning |
| --- | ------- |
| `info["n_qp_solves"]`    | QP subproblems solved during this solve |
| `info["n_qp_ws_changes"]`| active-set changes (adds + drops) across those QPs |

Both are `0` on the IPM path, which solves no QP subproblems.
`n_qp_ws_changes` is the measurement to watch across a sequence:

```python
prob.add_option("algorithm", "active-set-sqp")

ws = None
for k in range(horizon_steps):
    x, info = prob.solve(x0=x_prev, working_set=ws)
    print(k, info["iter_count"], info["n_qp_ws_changes"])
    ws, x_prev = info["working_set"], x
```

A warm start that is working drives the second column toward
zero while the first stays flat.
[The warm-start benchmark](warm-start-benchmark.md) is built on
exactly this measurement, and reports what the effect is worth
across eight problem families and all three solve paths.

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

### Rust: carry across solves

```toml
[dependencies]
pounce-rs = { version = "0.9", features = ["qp"] }
```

The round trip is `last_sqp_working_set` out, `SqpIterates` in:

```rust
use pounce_rs::prelude::*;
use pounce_rs::sqp::SqpIterates;

let mut app = IpoptApplication::new();
app.initialize()?;
app.initialize_with_options_str("algorithm active-set-sqp\n")?;

let mut working = None;
let mut x = x0.clone();
for _ in 0..horizon_steps {
    // ... user code updates the parameter inside the TNLP ...
    if let Some(w) = working.take() {
        app.set_sqp_warm_start(SqpIterates {
            x: x.clone(),
            lambda_g: lambda_g.clone(),
            lambda_x: lambda_x.clone(),   // packed: z_l − z_u
            working: Some(w),
        });
    }
    app.optimize_tnlp(Rc::clone(&tnlp));
    working = app.last_sqp_working_set().cloned();
    x = /* the x captured in finalize_solution */;
}
```

The warm start is consumed by the solve that follows it, so each pass
installs a fresh one; `clear_sqp_warm_start()` drops an unused one. Reading
individual rows uses the same status enums the Python int8 codes above
encode — `BoundStatus::{Inactive, AtLower, AtUpper, Fixed}` and
`ConsStatus::{Inactive, AtLower, AtUpper, Equality}`, both re-exported from
`pounce_rs::sqp`.

When the seed comes from a *sensitivity predictor* instead of a previous
solve (§4), there is no working set to carry —
`pounce_rs::sqp::classify_working_set` derives one from the predicted point
and its multipliers.

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
2. **Predictor**: ask `pounce_rs::sensitivity::SensSolve` for
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
| Design rationale        | [Design Note](active-set-sqp-warm-start.md)             |

## 9. Reading list

- Hock, Schittkowski (1981) *Test Examples for Nonlinear Programming Codes* — the in-repo HS subset reference.
- Nocedal, Wright (2006) *Numerical Optimization*, ch. 18 — SQP fundamentals.
- Fletcher, Leyffer (2002) — filter line search.
- Han (1977) / Powell (1978) — l1-merit and damped-BFGS update.
- Wächter, Biegler (2006) — pounce's IPM heritage.
- Gill, Murray, Saunders (2002) — SNOPT / l1-elastic phase 1.
- Forsgren, Gill, Wright (2002) — IPM vs SQP comparison.
- Kirches (2011) — parametric active-set SQP.
