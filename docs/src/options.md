# Solver Options

POUNCE accepts options the same way upstream Ipopt does. Option names
and semantics follow Ipopt's, so an existing Ipopt options file or
`KEY=VALUE` invocation works unchanged.

## Setting options

**On the command line** — append `KEY=VALUE` pairs after the input:

```sh
pounce problem.nl tol=1e-10 max_iter=500 print_level=8
```

**From an options file** — upstream `ipopt.opt` format:

```sh
pounce problem.nl --options-file ipopt.opt
```

Command-line `KEY=VALUE` pairs override values loaded from the options
file.

## Commonly used options

| Option          | Meaning                                                              |
|-----------------|----------------------------------------------------------------------|
| `tol`           | Overall convergence tolerance on the KKT error.                      |
| `max_iter`      | Maximum number of outer iterations.                                  |
| `print_level`   | Console verbosity, 0 (silent) – 12 (maximum debug).                  |
| `linear_solver` | KKT linear-solver backend: `feral` (default) or `ma57` (needs a `--features ma57` build). Any other registered name is **refused**. See below. |
| `mu_strategy`   | Barrier-parameter update strategy (`monotone` / `adaptive`).         |
| `solver_selection` | Route LP/convex-QP to the specialized convex IPM. See [LP/QP Routing](lp-qp-routing.md). |
| `qp_presolve`   | Presolve on the convex LP/QP path (`yes` / `no`, default `yes`). See [LP/QP Routing](lp-qp-routing.md#presolve). |
| `obj_scaling_factor` | Constant multiplier on the objective; **negative maximizes**. See below. |
| `bound_relax_factor` | Relaxation applied to variable/constraint bounds before the solve (default `1e-8`). |
| `honor_original_bounds` | Project the reported point back into the un-relaxed bounds (`yes` / `no`, default `no`). See below. |

For the full upstream option catalogue, see the
[Ipopt options reference](https://coin-or.github.io/Ipopt/OPTIONS.html);
POUNCE reuses those names.

For scaling-specific options (`nlp_scaling_method`, target-gradient
overrides, `linear_system_scaling`), see the [Scaling](scaling.md)
reference page. For nonlinear bound tightening (`presolve_fbbt`,
`fbbt_tol`, `fbbt_max_iter`, `fbbt_max_constraints`), see the
[FBBT](fbbt.md) reference page.

## Options POUNCE does not implement

POUNCE's option registry is a faithful port of Ipopt's: every name Ipopt
registers is registered here, so an `ipopt.opt` written for Ipopt parses
unchanged. Registering an option is not the same as implementing it,
though — and for a long time setting an unimplemented one did nothing at
all, silently.

Options naming a feature POUNCE does not have now **fail the solve**,
naming the option, the feature, and what to use instead:

```
$ pounce model.nl dependency_detector=mumps
pounce: `dependency_detector` configures linear-dependency detection on the
equality constraints, which pounce does not implement. It is registered so an
ipopt.opt written for Ipopt still parses, but setting it used to do nothing at
all — silently — so it is refused instead. Instead: pounce's presolve removes
structurally redundant rows; see `presolve`. Remove it to run.
```

The features in question: the Chen-Goldfarb (CG-penalty) / inexact-Newton
line search, derivative approximation by finite differences,
linear-dependency detection, the per-iteration NaN/Inf derivative check,
multiplier recalculation by least squares, a selectable
constraint-violation norm, magic steps, bound replacement, the L-BFGS
augmented-system variants, reading options from a file, skipping the
finalize callback, the dynamic HSL loader, and `suppress_all_output` /
`debug_print_level`.

Two deliberate exceptions:

* **Setting an option to its registered default is allowed.** A generated
  `ipopt.opt` spells out defaults, and `dependency_detector=none` asks for
  nothing. Only a value that differs from the default is a request POUNCE
  cannot honour.
* **Caching hints warn instead of failing.** `grad_f_constant`,
  `hessian_constant`, `jac_c_constant` and `jac_d_constant` tell the
  solver a quantity does not change between iterations. POUNCE
  re-evaluates regardless, so ignoring them costs evaluations and never
  correctness — failing the solve would be a worse trade.

Options whose *feature* runs and whose value simply is not read yet are
**not** in this category; they still solve, with the default in effect.
Wiring those is tracked on
[#191](https://github.com/jkitchin/pounce/issues/191) and
[#483](https://github.com/jkitchin/pounce/issues/483).

## Derivative checker

Wrong analytic derivatives are the most common cause of an NLP that
stalls, cycles, or converges to something that is not a solution — and
they are invisible from the iteration log. `derivative_test` compares
what your TNLP returns against finite differences at the (bound-projected)
starting point, before the solve:

```
pounce problem.nl derivative_test=first-order
```

| Option | Default | Effect |
|---|---|---|
| `derivative_test` | `none` | `none` / `first-order` / `second-order` / `only-second-order`. |
| `derivative_test_perturbation` | `1e-8` | Relative finite-difference step: `perturbation · max(1, |xᵢ|)`. |
| `derivative_test_tol` | `1e-4` | Flag an entry when `\|analytic − fd\| > tol · max(1, \|fd\|)`. |
| `derivative_test_first_index` | `-2` (all) | First **variable** for the first-order test; first **constraint** for the second-order one, where `-1` is the objective's Hessian. |
| `derivative_test_print_all` | `no` | List every entry, not just the suspicious ones. |

`first-order` checks `eval_grad_f` and `eval_jac_g`; `second-order` adds
`eval_h`; `only-second-order` checks the Hessian alone. The Hessian is
checked one multiplier block at a time — `obj_factor = 1, λ = 0` against
differences of `eval_grad_f`, then `obj_factor = 0, λ = eⱼ` against
differences of row `j` of `eval_jac_g`.

Entries that look wrong are marked `*`:

```
Derivative checker: first derivatives at the starting point (perturbation 1.0e-8, tolerance 1.0e-4).
* grad_f[    1]       =    3.5000000000000000e0    ~    3.0000000119209290e0  [  1.667e-1]
1 suspicious derivative(s) and 0 missing sparsity entrie(s) out of 6 checked (8 evaluations).
```

Two checks beyond upstream Ipopt's, because both catch a class of bug no
value-by-value comparison can:

* A Jacobian or Hessian entry whose finite difference is nonzero but
  which the **sparsity structure omits** (`!` in the report). A missing
  structural entry is not a wrong number — it is a derivative the solver
  can never see.
* The perturbation is taken **downward** when stepping up would leave a
  variable's box, so a model using `sqrt`, `log`, or `1/x` is not
  evaluated outside its own domain by the checker.

The test is advisory: it reports and the solve continues. It is written
to **stderr**, so it survives `print_level=0` and never mixes into
`--json-output`'s stdout. It is slow — the second-order test costs
roughly `(m+1)·n` evaluations — so leave it off for production runs.

> `check_derivatives_for_naninf` is a separate upstream option, for a
> per-iteration NaN/Inf guard, and is **not implemented**.

## Choosing a linear solver

POUNCE implements two KKT backends:

* **`feral`** — pure-Rust sparse symmetric indefinite solver. The
  effective default; no Fortran toolchain, no HSL licence.
* **`ma57`** — HSL MA57, available only in a `cargo build --features
  ma57` build.

The option's *registered* value list is a faithful port of upstream
Ipopt's (`ma27`, `ma77`, `ma86`, `ma97`, `mumps`, `pardiso`,
`pardisomkl`, `spral`, `wsmp`, `custom`), so an `ipopt.opt` written for
Ipopt parses here unchanged. Selecting one of those **fails the solve**
with a message naming it. They used to fall through to FERAL silently,
which meant `linear_solver=ma97` "worked" and a benchmark comparing
backends compared FERAL with itself.

The **registered default is `feral`**, which diverges from upstream's
`ma57` on purpose: a default has to name a solver the binary actually
contains. Under the upstream default a pure-Rust build advertised MA57 to
every `print_user_options` dump while running FERAL, and an
HSL-enabled build used MA57 without being asked. **If you build
`--features ma57` and want it, select it explicitly** — that is the one
behavioural change here.

Not a failure: **explicit `ma57` on a build without the feature** falls
back to FERAL and says so in the banner (`FERAL (ma57 requested but not
compiled)`). That substitution is reported rather than hidden, and
failing a portable `ipopt.opt` over a build flag would cost more than it
buys.

The per-backend tuning options (`ma97_scaling`, `mumps_pivtolmax`,
`pardiso_*`, `wsmp_*`, `spral_*`, …) remain registered for the same
`ipopt.opt`-compatibility reason. They are unreachable now that their
backend cannot be selected.

## Bound relaxation and `honor_original_bounds`

Before the solve, POUNCE widens every variable and constraint bound by
`bound_relax_factor` (default `1e-8`, capped by `constr_viol_tol`),
exactly as upstream Ipopt does — it keeps the interior-point iterates
strictly feasible without the user's bounds becoming numerically
degenerate. The consequence is that a solution *pinned to a bound* is
reported just past it:

```
min (x − 3)²  s.t.  0 ≤ x ≤ 1     →     x = 1.00000000937
```

`honor_original_bounds=yes` projects the reported point back into the
bounds you declared, so that solve returns exactly `x = 1`. Reach for it
whenever the value flows somewhere that cares about the domain — a
`sqrt(1 − x)`, a domain assertion, or a Pyomo `Var` the value is loaded
back into.

The default is `no`, matching upstream. As upstream also documents, the
constraint-violation and complementarity figures in the end-of-run
summary are for the **non-projected** point; only the reported `x` (and
the objective and constraint values evaluated at it) move.

## Objective sense and `obj_scaling_factor`

`obj_scaling_factor` multiplies the objective the IPM minimizes, so a
**negative** value maximizes — upstream's documented spelling for a
maximization problem stated as a minimization. Because it changes what is
being optimized rather than just its conditioning, it is honored only by
the general NLP interior-point path: a model that would otherwise route
to the specialized convex solvers (LP / convex QP / SOCP, see
[LP/QP Routing](lp-qp-routing.md)) is re-routed under
`solver_selection=auto`, and an explicit convex `solver_selection` is
**refused** rather than silently answering with the minimizer.

A positive factor is a pure conditioning knob; the convex path reports
natural units either way, so it keeps the fast path.

## Barrier-parameter (μ) strategy

The barrier parameter μ controls the inner subproblem's relaxation of
complementarity. The two strategies are `monotone` (default — geometric
schedule) and `adaptive` (quality-function oracle picks each μ from the
current iterate's complementarity). See
[μ-strategy](troubleshooting.md#μ-strategy) for when to switch.

| Option                                  | Default            | Meaning                                                                                       |
|-----------------------------------------|--------------------|-----------------------------------------------------------------------------------------------|
| `mu_strategy`                           | `monotone`         | `monotone` (Fiacco–McCormick schedule) or `adaptive` (oracle-driven).                         |
| `mu_oracle`                             | `quality-function` | Adaptive oracle: `quality-function` / `loqo` / `probing`.                                     |
| `mu_init`                               | `0.1`              | Seed value for μ at the first iterate.                                                        |
| `mu_min`                                | `1e-11`            | Floor on μ; the solver stops decreasing past this. In both μ strategies the effective floor is capped at `compl_inf_tol·|df|/(barrier_tol_factor+1)` (df = objective scaling factor) so a strongly scaled-down objective can still certify. |
| `mu_max`                                | `1e5`              | Cap on μ (adaptive mode). When set explicitly it overrides the `mu_max_fact` initialization.  |
| `mu_max_fact`                           | `1e3`              | Initializes `mu_max` as `mu_max_fact · curr_avrg_compl` at the first iterate (adaptive mode). |
| `mu_target`                             | `0.0`              | Stop target for μ in monotone mode.                                                           |
| `mu_linear_decrease_factor`             | `0.2`              | κ_μ in `μ ← min(κ_μ · μ, μ^θ_μ)`.                                                             |
| `mu_superlinear_decrease_power`         | `1.5`              | θ_μ in the same formula.                                                                      |
| `barrier_tol_factor`                    | `10.0`             | Inner-subproblem tolerance scales as `barrier_tol_factor · μ`.                                |
| `sigma_max`                             | `1e2`              | Upper clamp on σ chosen by the quality-function oracle.                                       |
| `sigma_min`                             | `1e-6`             | Lower clamp on σ (raising this to `1e-2` can break a stair-stepping stall on some problems).  |
| `adaptive_mu_globalization`             | `obj-constr-filter`| Adaptive-mode globalization: `kkt-error`, `obj-constr-filter`, or `never-monotone-mode`.      |

### Quality-function oracle (adaptive-μ details)

These are only consumed when `mu_strategy=adaptive` and
`mu_oracle=quality-function`. Defaults mirror upstream
`IpQualityFunctionMuOracle::RegisterOptions`.

| Option                                  | Default          | Meaning                                                                                       |
|-----------------------------------------|------------------|-----------------------------------------------------------------------------------------------|
| `quality_function_norm_type`            | `2-norm-squared` | Norm used to aggregate KKT components inside `q(σ)`: `1-norm`, `2-norm`, `2-norm-squared`, `max-norm`. |
| `quality_function_centrality`           | `none`           | Centrality penalty term: `none`, `log`, `reciprocal`, `cubed-reciprocal`.                     |
| `quality_function_balancing_term`       | `none`           | Balancing penalty when complementarity ≪ infeasibilities: `none` or `cubic`.                  |
| `quality_function_max_section_steps`    | `8`              | Cap on golden-section iterations when picking σ.                                              |
| `quality_function_section_sigma_tol`    | `1e-2`           | Width tolerance in σ-space terminating the golden-section search.                             |
| `quality_function_section_qf_tol`       | `0.0`            | Relative flatness tolerance on `q(σ)` terminating golden section.                             |

### Adaptive-μ globalization

Tuning the safeguards that fall back to monotone-μ mode when the
adaptive oracle stops making progress. Defaults mirror upstream
`IpAdaptiveMuUpdate::RegisterOptions`.

| Option                                  | Default | Meaning                                                                                       |
|-----------------------------------------|---------|-----------------------------------------------------------------------------------------------|
| `adaptive_mu_safeguard_factor`          | `0.0`   | LOQO safeguard floor on the oracle's μ candidate.                                             |
| `adaptive_mu_monotone_init_factor`      | `0.8`   | Multiplier on `avrg_compl` when seeding monotone mode after a bailout.                        |
| `adaptive_mu_restore_previous_iterate`  | `no`    | Restore the latest free-mode iterate when switching to fixed mode.                            |
| `adaptive_mu_kkterror_red_iters`        | `4`     | Window length for the `kkt-error` globalization history.                                      |
| `adaptive_mu_kkterror_red_fact`         | `0.9999`| Required relative KKT-error reduction over that window.                                       |
| `adaptive_mu_kkt_norm_type`             | `2-norm-squared` | Norm used to score the iterate in adaptive globalization decisions.                  |

## ℓ₁ penalty-barrier wrapper options

These tune the degenerate-NLP wrapper described in
[Running Solves](cli.md). All are default-tuned and rarely need
overriding:

| Option                               | Default | Meaning                                                    |
|--------------------------------------|---------|------------------------------------------------------------|
| `l1_exact_penalty_barrier`           | `no`    | Run the ℓ₁-exact penalty-barrier wrapper unconditionally.  |
| `l1_fallback_on_restoration_failure` | `no`    | Retry with the wrapper only when the standard solve fails. |
| `l1_penalty_init`                    | `1.0`   | Initial penalty weight ρ.                                  |
| `l1_penalty_max`                     | `1e6`   | Maximum penalty weight before declaring infeasibility.     |
| `l1_penalty_increase_factor`         | `8.0`   | Multiplier applied to ρ each outer iteration.              |
| `l1_penalty_max_outer_iter`          | `8`     | Maximum penalty outer iterations.                          |
| `l1_slack_tol`                       | `1e-6`  | Slack tolerance for "constraints satisfied".               |
| `l1_steering_factor`                 | `10.0`  | Steering-rule factor for ρ escalation.                     |

## NLP Presolve

POUNCE's TNLP-wrapper presolve pipeline runs *before* the IPM
starts. It tightens variable bounds, drops redundant rows, and
(optionally) eliminates square auxiliary-equality sub-systems
structurally. All are off by default — set the master switch first:

`presolve=yes` applies equally to CLI solves and to every
`IpoptApplication::optimize_tnlp` library solve; callers no longer need to
wrap a callback TNLP manually.
The wrapper postsolves before `finalize_solution`, so callback payloads remain
in the submitted TNLP's original variable and constraint space. Bare callback
TNLPs do not expose an expression provider, so `presolve_fbbt=yes` remains a
no-op for this library entry point.

| Option                                  | Default | Meaning                                                                        |
|-----------------------------------------|---------|--------------------------------------------------------------------------------|
| `presolve`                              | `no`    | Master switch for the whole presolve layer. Off → wrapper is a no-op.          |
| `presolve_bound_tightening`             | `yes`   | Phase 1 — Andersen-style bound propagation from linear rows.                   |
| `presolve_redundant_constraint_removal` | `yes`   | Phase 2 — drop linear constraints already implied by current bounds.           |
| `presolve_linear_eq_reduction`          | `no`    | Phase 6 — eliminate variables determined by linear equality rows (see below).   |
| `presolve_licq_check`                   | `yes`   | Phase 3 — detect rank-deficient equality blocks before the IPM starts.         |
| `presolve_licq_action`                  | `warn`  | What to do on degeneracy: `warn` (just report) or `auto_l1` (turn on ℓ₁).      |
| `presolve_warm_z_bounds`                | `yes`   | Phase 4 — warm-start bound multipliers when bounds get tightened by Phase 1.   |
| `presolve_bound_mult_init_val`          | `1.0`   | Value used by Phase 4 for those warm-start hints.                              |
| `presolve_max_passes`                   | `3`     | Fixed-point iteration cap across the bound-tightening passes.                  |
| `presolve_print_level`                  | `0`     | Per-pass verbosity (0 silent, 5 per-pass, 8 per-transformation).               |

### Linear-equality variable elimination (Phase 6)

`presolve_linear_eq_reduction=yes` is the only pass that removes
**columns**. It reads the model's linear equality rows and eliminates the
variables they determine, iterating to a fixed point so chains propagate:

* a variable whose declared bounds are equal becomes a constant;
* a singleton row `a·x = b` pins its variable at `b/a`;
* a two-variable row `a₁·x + a₂·y = b` substitutes one variable for the
  other, `x := α·y + β`. There is **no anchoring requirement**: a row
  linking two otherwise-free interior variables — an arc equality, a
  `Reference` alias, a unit-conversion link — aggregates away, which is
  the case the [auxiliary-equality pass](auxiliary-presolve.md) cannot
  reach because it only solves *determined* square blocks.

Rows that collapse to `0 = 0` under the accumulated substitutions are
dropped as structurally redundant.

A row written with a constant on the left — `x0 − 2·x1 + 3 = 3` — is
eligible on the same terms as `x0 − 2·x1 = 0`. The `.nl` reader folds a
constant row body into the row's bounds when the file is read, so the pass
sees an ordinary linear equality.

Every eliminated variable's bounds are transferred onto its survivor, so
the reduced box is never looser than the original. `finalize_solution`
lifts the primal back to the original variable order and recovers a
multiplier for each consumed row, so `.sol` / JSON solution blocks keep the
original model's shape and can still be read positionally by AMPL or Pyomo.

Three things to know before turning it on:

* **Dual attribution.** A transferred bound's multiplier comes back on the
  variable that *declared* the bound, not on the survivor that inherited
  it. The plan records where each reduced bound came from, and postsolve
  rescales the multiplier by the substitution's coefficient `α` — and moves
  it to the other side of the box when `α < 0`, since a negative
  coefficient turns a lower bound into an upper one. On a model with a
  single active transferred bound the reported duals match a no-presolve
  solve exactly.

  Where the survivor's *own* bound and a transferred bound are active at
  the same point, the split between the two multipliers is genuinely
  non-unique — the reduced problem has one where the full problem has two —
  and the pass leaves the whole multiplier on the survivor. That is a valid
  KKT point, but it is not the split a no-presolve solve happens to report.
  The same holds for a variable pinned by a singleton row `a·x = b` whose
  value lands on one of its own bounds: the row multiplier absorbs it.

  A practical consequence for `.sol` readers, unchanged by any of the
  above: the writer omits exact zeros from suffix blocks (it always has),
  so a variable whose bound multiplier is zero gets **no**
  `ipopt_zL_out` / `ipopt_zU_out` entry at all rather than an entry of
  zero. Code that indexes those suffixes must treat a missing index as
  zero — as it already must for any variable whose bound multiplier lands
  exactly on zero. Row multipliers are unaffected: the dual block is dense
  and comes back at the original row count.
* **Bounds the reduced problem never saw.** Re-attribution can only move a
  multiplier the solver reported, and sometimes there is none. The
  transfers can leave a survivor's reduced box as a single point, and a
  variable with equal bounds is a *fixed* variable, which the solver drops
  — so it comes back with no bound multiplier at all even though the
  cluster it stands for is sitting on a bound that needs one. Postsolve
  fills that in: whatever stationarity residual the recovered row
  multipliers cannot close is a bound multiplier that was never reported,
  and it goes on the declared bound the point is actually resting on — the
  survivor's own where that is the active one, otherwise the column the
  bound was borrowed from, through the same `α` rescale as above. A
  residual with no active declared bound to carry it is left alone rather
  than parked somewhere that would break complementarity.

  One column is deliberately outside this: one the *model* declares fixed
  (`x_l == x_u`). The solver drops those as parameters whether or not the
  reduction runs, and reports no multiplier for them either way, so nothing
  here changes what they report.
* **Failing closed.** If the equality system is contradictory, the pass
  stands down entirely and hands the model to the solver untouched, rather
  than being the first and only voice to call a model infeasible. The same
  goes for a model whose every column is determined: a zero-variable problem
  is not a shape worth handing the IPM.

It is off by default because it changes the variable count, which the
sensitivity and reduced-Hessian paths index against the original `.nl`.
(The CLI already disables presolve entirely when those are requested.)

**LP and convex QP take a different route to the same reduction.** Those
models never reach Phase 6 — the CLI dispatches them to `pounce-convex`
before any presolve wrapper is built — but they are not left unreduced.
`pounce-convex` has its own presolve, on by default, and it now performs
the two-variable aggregation as part of that catalog, sharing this
planner rather than restating it. So the reduction is the same; only the
switch differs (`qp_presolve=no` / `presolve=no` turns it off there, and
`presolve_linear_eq_reduction` does not apply). See
[LP/QP routing](lp-qp-routing.md#two-variable-equality-rows-aggregation).

The two agree on dual attribution as well: a transferred bound's
multiplier is reported on the column that *declared* the bound, not on the
survivor that inherited it. They get there differently — Phase 6 records
during planning where each reduced bound came from, while the convex path
reads the leftover reduced cost at postsolve and hands it to whichever
column is sitting on its own bound, because it also has inequality rows
and its own bound-tightening layer to account for. Where the survivor's
own bound is active as well, both leave the multiplier on the survivor;
the split is genuinely non-unique there and either answer is a valid KKT
point.

### Feasibility-based bound tightening (Phase 1b)

Interval-arithmetic propagation through nonlinear constraint
expression DAGs (see [FBBT](fbbt.md)). Available today for
`.nl`-loaded problems via `NlTnlp`; other TNLP sources opt out
silently.

| Option                  | Default | Meaning                                                                                  |
|-------------------------|---------|------------------------------------------------------------------------------------------|
| `presolve_fbbt`         | `no`    | Master switch. Requires `presolve=yes` and an `ExpressionProvider`.                      |
| `fbbt_tol`              | `1e-6`  | Minimum per-variable bound improvement to keep iterating.                                |
| `fbbt_max_iter`         | `10`    | Outer-sweep cap.                                                                         |
| `fbbt_max_constraints`  | `0`     | Per-sweep cap on constraints inspected (`0` = unlimited).                                |

### Auxiliary-equality preprocessing (Phase 0)

A separate set of options controls the structural elimination pass
documented in [Auxiliary-Equality Preprocessing](auxiliary-presolve.md):

| Option                                   | Default | Meaning                                                                                  |
|------------------------------------------|---------|------------------------------------------------------------------------------------------|
| `presolve_auxiliary`                     | `no`    | Master switch for the Phase-0 structural elimination pass.                               |
| `presolve_auxiliary_coupling`            | `safe`  | Which coupling classes are eligible: `none` / `safe` / `aggressive`.                     |
| `presolve_auxiliary_tol`                 | `1e-8`  | Residual tolerance for accepting a candidate block solve.                                |
| `presolve_auxiliary_max_block_dim`       | `8`     | Largest block the lightweight Newton solver will attempt (larger blocks rejected in v1). |
| `presolve_auxiliary_wall_time_fraction`  | `0.1`   | Fraction of the solver's wall-time budget the pass is allowed to spend.                  |
| `presolve_auxiliary_diagnostics`         | `no`    | Emit the diagnostics summary via the journalist after Phase 0 runs.                      |

## FERAL backend tuning

`linear_solver=feral` (the default — see
[Commonly used options](#commonly-used-options)) is configurable
through seven `feral_*` options. Defaults are tuned for the IPM
workload and rarely need changing; reach for these when profiling a
specific problem. Each also falls back to a matching `POUNCE_FERAL_*`
environment variable when left unset on the OptionsList (see
[Environment overrides](#environment-overrides-feral-and-debug-gates)).

| Option                       | Default | Meaning                                                                                                                                                                                  |
|------------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `feral_ordering`             | `auto`  | Fill-reducing ordering method (see table below). `auto` lets feral's adaptive dispatcher pick per-matrix; `auto_race` measures the actual symbolic outcome and keeps the best.            |
| `feral_pivtol`               | `1e-8`  | Relative Bunch-Kaufman partial-pivoting threshold `u`. Analog of `ma27_pivtol` / `ma57_pivtol`. Smaller → sparser `L`, faster, less stable; larger → more 2×2 blocks, denser, more stable. LAPACK's textbook maximum-stability value is `0.5`. |
| `feral_refine`               | `yes`   | Iterative refinement on every back-solve. Closes the residual floor from cascade-break's `L`-factor perturbation; disable only when timing the bare factor + back-solve in isolation.     |
| `feral_cascade_break`        | (unset) | Tri-state. Unset → inherit feral's Phase B default (CB on with bounded delayed-pivot catchment). `yes` records explicit intent (no behavioural change). `no` reproduces pre-Phase-B behaviour by surfacing `DelayBudgetExceeded` on non-root cascade victims.  |
| `feral_fma`                  | `no`    | Dispatch dense kernels through fused multiply-add intrinsics. Roughly 2× throughput on aarch64 / x86_v3, at the cost of per-pivot rounding drift that trips more `WrongInertia` checks. Turn on when kernel throughput dominates and the IPM tolerates a noisier inertia signal. |
| `feral_singular_pivot_floor` | `1e-20` | Pounce's analog of MA57's `CNTL(2)`. After a successful factor, the smallest accepted `D`-block pivot magnitude (scaled space) is compared against this absolute floor; if it falls below, the factor is reported `Singular` so the IPM bumps `δ_w`. `0` disables. |
| `feral_min_par_flops`        | `1e8`   | Flop threshold above which a supernode subtree is dispatched to a parallel worker (feral#19). Lower → dispatch more aggressively (`0` fires on every multi-child tree at/above `N_PAR_MIN` supernodes); a very large value rejects all tree-level parallelism. Only matters when feral's internal parallelism is active; no effect on a serial factor. |
| `feral_static_pivoting`      | (unset) | Tri-state. Factor with static pivoting (SSIDS-style delayed pivots disabled). Unset → inherit feral's delayed-pivot default. `yes` runs every supernode as the root does — a failing pivot is force-accepted in place with iterative refinement recovering the residual — breaking the delayed-pivot cascade that can turn one factorization into tens of seconds (feral#8; the emfl050 case in #254). feral's analog of MA57's `cntl[4]`. `no` keeps delayed pivoting on. Deliberately **not** coupled to `max_wall_time` — the accuracy/speed trade is the caller's to set per solve. |

### `feral_ordering` variants

All six concrete and adaptive options live under the same string
option. `feral_ordering` also falls back to the
`POUNCE_FERAL_ORDERING` environment variable when not set on the
OptionsList.

| Value       | Strategy                                                                                                                                                                                                                                                  |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `auto`      | **Default.** Adaptive dispatcher: picks a concrete method per matrix from cheap pattern features. Branches: very-large-and-sparse (`n > 100 000`, avg degree < 5) → AMD; `n ≤ 10 000` → AMF; otherwise → MetisND. One symbolic pass; right when the heuristic shape rules apply (the common case). |
| `auto_race` | Race-based dispatcher: runs full symbolic factorization on AMD, MetisND, ScotchND, KahipND and keeps the smallest `factor_nnz`. ~4× a single symbolic pass, paid once per problem (symbolic factorization is cached across numeric refactorizations with the same pattern). Use when the cheap dispatcher's guess is suspect — e.g. `pinene_3200_0009`, where `auto` picks MetisND (88 s numeric factor) but `amd` factors in 19.5 s on the same matrix. |
| `amd`       | Approximate Minimum Degree (Amestoy/Davis/Duff). Pins AMD regardless of problem shape; robust default for IPM workloads. Best for very-large-and-sparse cases that the adaptive dispatcher already routes here.                                            |
| `amf`       | Approximate Minimum Fill (HAMF4 variant of Amestoy 1999). Strong on small-and-sparse populations (`n ≤ 10 000`); aggregate fill ≈ 0.87× AMD on feral's IPM small-sparse inventory.                                                                          |
| `metis`     | feral-metis multilevel nested dissection. Tends to produce squarer fronts than AMD on banded / nearly-1D structure; preferred for large structured matrices.                                                                                              |
| `scotch`    | feral-scotch nested dissection. Similar regime to METIS; alternative when METIS is unavailable or for cross-validation.                                                                                                                                   |
| `kahip`     | feral-kahip flow-based nested dissection with K1 preprocessing. Ties METIS on fill geomean at 4–6× per-call symbolic cost. Reach for it only when ND fill matters and per-call cost is amortized.                                                          |

When in doubt: leave `feral_ordering` at the default. When a hard
problem looks linear-solver-bound, try `feral_ordering auto_race`
before per-variant manual sweeping — it's the safe choice when the
per-problem winner is uncertain.

#### Caller-supplied ordering (`External`)

Beyond the string variants above, a structure-aware caller can inject a
**precomputed permutation** the generic AMD/METIS pass cannot see — a
block-triangular / Schur ordering (Parker, Garcia & Bent,
arXiv:2602.17968) or a tearing ordering from equation-oriented
decomposition. Because a permutation is a vector it cannot travel through
the string `feral_ordering` option; supply it programmatically instead:

- **Python:** `Problem.set_ordering(perm)` (and `get_ordering()` /
  `clear_ordering()`) — see [the Python guide](./python.md#caller-supplied-kkt-ordering-set_ordering).
- **Rust:** `IpoptApplication::set_external_ordering(perm)`.

`perm` is a 0-based, new-to-old permutation (`perm[k]` is the original
index that becomes index `k`) whose length must equal the **augmented KKT
system dimension** (variables + slacks + constraint duals), not the
problem's `n`. FERAL validates it as a bijection and fails the
factorization with an error on a wrong length or duplicate — a valid but
poor ordering only costs fill/time, never correctness. This maps to
FERAL's `OrderingMethod::External` (feral#107) and honors only the default
FERAL backend.

## Environment overrides (FERAL and debug gates)

A handful of knobs are reachable through environment variables. The
`feral_*` numerics knobs read their `POUNCE_FERAL_*` variable **only as a
fallback** when the matching option is left unset on the OptionsList — set
the option (per solve, recordable, discoverable via the debugger's `opt`
command) in preference to the env var (process-wide, invisible to the solve
report). The debug gates below have no option equivalent; they exist purely
to switch on extra diagnostic output.

### FERAL numerics fallbacks

Each maps one-to-one to a registered option in
[FERAL backend tuning](#feral-backend-tuning). Prefer the option; the env
var is the fallback for callers with no OptionsList (some tests, legacy
embeddings).

| Variable                            | Option                       |
|-------------------------------------|------------------------------|
| `POUNCE_FERAL_ORDERING`             | `feral_ordering`             |
| `POUNCE_FERAL_SCALING`              | `feral_scaling`              |
| `POUNCE_FERAL_PIVTOL`               | `feral_pivtol` (deprecated bare `FERAL_PIVTOL` also accepted) |
| `POUNCE_FERAL_REFINE`               | `feral_refine`               |
| `POUNCE_FERAL_CASCADE_BREAK`        | `feral_cascade_break`        |
| `POUNCE_FERAL_FMA`                  | `feral_fma`                  |
| `POUNCE_FERAL_SINGULAR_PIVOT_FLOOR` | `feral_singular_pivot_floor` |
| `POUNCE_FERAL_MIN_PAR_FLOPS`        | `feral_min_par_flops`        |
| `POUNCE_FERAL_STATIC_PIVOTING`      | `feral_static_pivoting`      |

`FERAL_PARALLEL=0` (legacy, no `POUNCE_` prefix) forces feral's internal
factor serial process-wide; the first-class per-backend lever is the
solver API, not an option.

### Debug and diagnostic gates

These switch on extra diagnostic emission for a specific subsystem. Most
emit at **debug** level under a `pounce::*` [tracing](#logging-and-colored-output)
target, so setting the gate alone is not enough — pair it with a matching
`RUST_LOG` (e.g. `RUST_LOG=pounce::mu=debug`) or the output stays filtered.
Presence-only unless a value is noted; they are diagnostic aids, not part
of the stable interface, and may change between releases.

| Variable | Subsystem (`RUST_LOG` target) | Emits |
|---|---|---|
| `POUNCE_DBG_AMU` | `pounce::mu` | Adaptive-μ per-iteration state (θ, f, oracle inputs). |
| `POUNCE_DBG_ORACLE` | `pounce::mu` | μ-oracle probe-guard decisions (probe-Newton → restoration requests). |
| `POUNCE_DBG_QF` | `pounce::mu` | Quality-function μ-oracle σ search (floor, current μ). |
| `POUNCE_DBG_QF_AGGR` | `pounce::mu` | Quality-function aggregate step/complementarity terms per σ. |
| `POUNCE_DBG_QF_SWEEP=<iter>` | `pounce::mu` | Dumps the full quality-function σ sweep at the given iteration number. |
| `POUNCE_DBG_DELTA` | `pounce::algorithm` | Primal-dual search direction `δ` per iteration. |
| `POUNCE_DBG_LS=1` | `pounce::linesearch` | Filter line-search / backtracking acceptance trace (must equal `1`). |
| `POUNCE_DBG_PERT` | `pounce::linsol` | Inertia-perturbation handler decisions (`WRONG_INERTIA`, `δ_w` escalation). |
| `POUNCE_DBG_PD_TAGS` | `pounce::linsol` | Primal-dual full-space solver dependent-block tag changes. |
| `POUNCE_DBG_KKT_DUMP=<path>` | `pounce::linsol` | Writes the tagged KKT matrix to `<path>`. |
| `POUNCE_DBG_KKT_DUMP_SKIP=<n>` | — | Skip the first `<n>` factorizations before honoring `POUNCE_DBG_KKT_DUMP`. |
| `POUNCE_DUMP_KKT=<path>` | `pounce::linsol` | Writes the standard augmented-system KKT matrix to `<path>`. **Deprecated** — prefer `--dump kkt:<iter-spec>` (see `pounce --help`). |
| `POUNCE_DBG_RESTO` | `pounce::algorithm`, `pounce::restoration` | Restoration entry trace **and** the augmented restoration-system stats. Canonical spelling; the legacy `POUNCE_RESTO_DBG` (restoration-system stats only) is a deprecated alias. |
| `POUNCE_DBG_RESTO_CYCLE` | `pounce::algorithm` | Restoration no-progress cycle-detector relative-step metrics. |
| `POUNCE_DBG_RESTO_INIT` | `pounce::restoration` | Restoration initial-point vectors. |
| `POUNCE_DBG_RESTO_KAPPA` | `pounce::restoration` | Restoration `κ_resto` convergence-guard evaluation. |
| `POUNCE_DBG_RESTO_LOCINF` | `pounce::restoration` | Restoration local-infeasibility verdict inputs. |
| `POUNCE_DBG_TAPE_STATS` | — (stderr) | AD tape counts after parsing an `.nl` model. Printed straight to stderr; no `RUST_LOG` needed. |
| `POUNCE_SIMPLEX_DEBUG` | — (stderr) | Convex/LP-QP simplex pivoting trace. Printed straight to stderr; no `RUST_LOG` needed. |

Two already-documented gates round out the set: `POUNCE_DBG_LLM` and
`POUNCE_DBG_VIEWER` (see [the debugger guide](./debugger.md)).

## Logging and colored output

POUNCE emits structured logs and a colored iteration table through the
[`tracing`](https://docs.rs/tracing) ecosystem. Behavior is governed by
environment variables (not solver options), so they apply to the `pounce`
CLI, the C/Python frontends, and anything embedding the library.

| Variable | Values | Effect |
|---|---|---|
| `RUST_LOG` | e.g. `info`, `debug`, `pounce::restoration=debug` | Log verbosity / per-target filtering. Default `info`. Logs go to **stderr**. |
| `POUNCE_LOG_FORMAT` | `text` (default) · `json` | `json` emits line-delimited JSON on stderr (incl. the per-iteration `pounce::iteration` stream) for Studio / CI ingestion. |
| `NO_COLOR` | set to any value | Disables ANSI color in the iteration table **and** logs (see <https://no-color.org>). |
| `CLICOLOR_FORCE` | set to any value | Forces color even when stdout is not a terminal. |

**Filtering by subsystem.** Solver internals log under namespaced targets
— `pounce::algorithm`, `pounce::linsol`, `pounce::mu`, `pounce::sqp`,
`pounce::linesearch`, `pounce::restoration`, `pounce::presolve`,
`pounce::py`. For example, to trace only the restoration phase:

```sh
RUST_LOG=pounce::restoration=debug pounce problem.nl
```

**Program output vs. logs.** The iteration table, the final summary, and
`--dump` diagnostics are *program output* on **stdout**; diagnostic and
progress messages are *logs* on **stderr**. Redirecting one does not
affect the other:

```sh
pounce problem.nl > result.txt 2> solve.log
```

**Color.** The iteration table is colored with a tiger/rust theme:
restoration lines take a background that varies by restoration kind
(soft-stay → tan, soft-exit → amber, hard → deep rust), and the row text
shades from black toward red as the primal step length `alpha` shrinks
(stalling). Color is emitted only when stdout is a terminal; redirected
output and `NO_COLOR` get plain text with identical column alignment.

**Machine-readable iterations.** `POUNCE_LOG_FORMAT=json` turns the
per-iteration records into JSON on stderr:

```sh
POUNCE_LOG_FORMAT=json pounce problem.nl 2> iters.jsonl
```
