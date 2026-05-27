# NLP and Linear-System Scaling

Optimization problems whose objective, constraints, or KKT system span
many orders of magnitude often converge poorly — or not at all — without
some form of rescaling. pounce inherits two independent scaling layers
from Ipopt and adds a third option at the linear-system level
(see [issue #61](https://github.com/jkitchin/pounce/issues/61)).

The two layers are conceptually separate:

| Layer | Option | What it touches |
|---|---|---|
| **NLP scaling** | `nlp_scaling_method` | The objective `f` and each constraint row `c_i`, before the IPM sees them. Changes algorithmic behavior (filter, `tol`, μ). |
| **Linear-system scaling** | `linear_system_scaling` | Symmetric scaling of the KKT augmented system `D K D` for the factorization. Purely numerical — the IPM sees the same iterates. |

You can configure them independently. Defaults match upstream Ipopt:
`nlp_scaling_method = gradient-based`, `linear_system_scaling = none`.

## NLP-level scaling

| Option | Default | Effect |
|---|---|---|
| `nlp_scaling_method` | `gradient-based` | `none` / `gradient-based` / `user-scaling`. |
| `nlp_scaling_max_gradient` | `100.0` | Cutoff above which gradient-based scaling applies. Per-row scale = `min(1, max_gradient / ‖∇c_i‖_∞)`. |
| `nlp_scaling_min_value` | `1e-8` | Floor on computed scale factors — prevents inverting near-zero gradients. |
| `nlp_scaling_obj_target_gradient` | `0.0` | When `> 0`, *pins* the scaled objective gradient ∞-norm to this value. Overrides the `max_gradient` cutoff. |
| `nlp_scaling_constr_target_gradient` | `0.0` | Same as above, per constraint row. |
| `obj_scaling_factor` | `1.0` | Constant multiplier on the objective, applied after the automatic factor. |

### `gradient-based` (default)

Evaluates `∇f` and `∇c_i` *once* at the starting point `x_0` and
chooses per-row scales that pull each gradient ∞-norm into a
reasonable band. Single-shot is mandatory — recomputing per iteration
would invalidate the filter's history (Wächter, 2013).

The clamp at 1.0 means scaling never *amplifies* a small row; it only
damps large ones.

### `user-scaling`

The TNLP is asked for `obj_scaling`, a per-variable `x_scaling`, and a
per-constraint `g_scaling` via the `get_scaling_parameters` callback.
Use this when you know the natural units of your problem (e.g. mass in
kg vs. distance in mm) and can supply better scales than the
gradient-based heuristic.

> **Note**: pounce's `OrigIpoptNlp` currently honors `obj_scaling` and
> per-constraint `g_scaling`. The `x_scaling` request channel is
> accepted but not yet acted on. Mirrors the design in
> [issue #61](https://github.com/jkitchin/pounce/issues/61).

If the TNLP's `get_scaling_parameters` returns false (the default),
pounce falls back to no automatic scaling.

#### Setting user scaling

* **From C** — call `SetIpoptProblemScaling(problem, obj, x_scaling,
  g_scaling)` then `AddIpoptStrOption("nlp_scaling_method",
  "user-scaling")`. See `crates/pounce-cinterface/include/pounce.h`.
* **From Rust** — implement
  [`TNLP::get_scaling_parameters`](https://github.com/jkitchin/pounce/blob/main/crates/pounce-nlp/src/tnlp.rs)
  on your problem type.
* **From Python** — `pounce.Problem.set_problem_scaling(obj_scaling,
  x_scaling=None, g_scaling=None)`, followed by
  `add_option("nlp_scaling_method", "user-scaling")`. Walked through
  end-to-end in
  [`python/notebooks/07_scaling.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/07_scaling.ipynb).

### Target-gradient overrides

`nlp_scaling_obj_target_gradient` and
`nlp_scaling_constr_target_gradient` are subtle. When set to a
positive value, they *override* the `max_gradient` cutoff and the 1.0
clamp: the scaling is computed unconditionally as
`target / max_gradient_norm`, so the scaled gradient ∞-norm becomes
exactly the target. Useful when you have a specific numeric range you
want the IPM to see.

The default `0.0` means "use the cutoff path" — i.e. only scale rows
that are above `nlp_scaling_max_gradient`.

## Linear-system-level scaling

| Option | Default | Effect |
|---|---|---|
| `linear_system_scaling` | `none` | `none` / `mc19` / `ruiz` / `slack-based`. |
| `linear_scaling_on_demand` | `yes` | Defer scaling computation until a linear solve is poor; reduces overhead for well-conditioned KKT systems. |

The KKT augmented system is symmetric; all linear-system scalers in
pounce use the symmetric form `D K D` (single diagonal) to preserve
that structure for the downstream factorization (MA57, MUMPS,
FERAL/SSIDS).

* **`none`** — first-class choice. The inner linear solver (MA57,
  MUMPS, FERAL) often does its own scaling under some configurations;
  stacking pounce-level scaling on top can *hurt*. Default. Use
  `ma57_automatic_scaling=yes` to get MA57's internal scaling instead.
* **`mc19`** — HSL MC19 row/column scaling (Curtis-Reid 1972;
  minimizes Σ log²|a_ij|). FFI to `libcoinhsl`; requires the
  `pounce-hsl` build.
* **`ruiz`** — iterative symmetric ∞-norm equilibration (Ruiz,
  CERFACS TR/PA/01/14). Pure Rust, no Fortran dependency. Converges
  geometrically; capped at 10 iterations. Recommended starting point
  when MA57's internal scaling is off and you can't link MC19.
* **`slack-based`** — slack-aware scaling driven by the current
  barrier slacks. Used internally by the inexact algorithm; rarely
  user-selected.

> **Heads-up**: `linear_system_scaling` is registered through the
> options machinery, and each method's per-row factor computation is
> tested in isolation, but the runtime dispatch in `alg_builder.rs`
> currently passes `None` to `TSymLinearSolver`. Wiring the option
> through end-to-end is tracked separately from
> [issue #61](https://github.com/jkitchin/pounce/issues/61).

## Reporting

All scaling effects are undone before the solve report (final
objective, multipliers, dual residuals, KKT termination metric) is
handed back to the user. You always see quantities in the natural
units of your TNLP.

Internally, the IPM operates in scaled space: stopping criteria
(`tol`, `acceptable_tol`) compare scaled values, the barrier parameter
μ is in scaled units, and the filter's history is built from scaled
function values.

## When to override the defaults

Reach for non-default scaling when:

* The constraint Jacobian has entries spanning many orders of magnitude
  (chemistry, power-flow, mixed-unit mechanics). Try `mc19` or `ruiz`
  at the linear-system level, after disabling MA57's internal scaling.
* The IPM stalls with small step sizes but no clear infeasibility.
  Worth turning `nlp_scaling_method=none` to see whether the default
  gradient scaling is doing the wrong thing; then re-enable with
  problem-specific target gradients.
* You know the natural units of your problem better than the solver
  can infer from gradients at `x_0`. Wire `user-scaling`.

Otherwise the upstream-Ipopt-style defaults (`gradient-based` at the
NLP level, `none` at the linear-system level with MA57's internal
scaling on) are a reasonable starting point.

## References

* Wächter, A. *On the effects of scaling on the performance of Ipopt.*
  arXiv:1301.7283 (2013). <https://arxiv.org/abs/1301.7283>
* Ruiz, D. *A scaling algorithm to equilibrate both rows and columns
  norms in matrices.* CERFACS TR/PA/01/14.
  <https://cerfacs.fr/wp-content/uploads/2017/06/14_DanielRuiz.pdf>
* Curtis, A. R. and Reid, J. K. *On the Automatic Scaling of Matrices
  for Gaussian Elimination.* (1972). HSL MC19 reference.
* pounce issue [#61](https://github.com/jkitchin/pounce/issues/61).
