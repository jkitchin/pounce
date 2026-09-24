# `bound_relax_factor` × a large objective coefficient (gh#967)

`pounce.minimize` returns `Solve_Succeeded` on an LP whose reported objective
is wrong by `1e4` and **negative**, for an objective that is non-negative
everywhere on the declared box. This note records what the number is, why the
obvious fixes were rejected, and what a caller should do instead.

## The model

```
min  C·x + y/C     s.t.  x + y >= 1,  x, y ∈ [0, 1]
```

The optimum is `x=0, y=1` with value `1/C` for **every** `C`, so the exact
answer is known at every scale. Measured on 0.12.0, NLP arm:

| C | true | `res.fun` | abs err | status | `x[0]` |
|---|---|---|---|---|---|
| 1e2 | 1.0000e-02 | 9.999005e-03 | 9.949e-07 | 0 | -9.949212e-09 |
| 1e6 | 1.0000e-06 | **-9.948213e-03** | 9.949e-03 | 0 | -9.949213e-09 |
| 1e12 | 1.0000e-12 | **-1.000000e+04** | 1.000e+04 | 0 | -1.000000e-08 |

`x[0]` is ≈ `-1e-8` **regardless of C** — a constant absolute box violation,
the `bound_relax_factor` signature. Multiplying it by `C` reproduces the error
column exactly. The whole error is the widening; `C` only amplifies it.

## The one-line diagnosis

`bound_relax_factor` perturbs the declared feasible set by an **absolute**
amount (`min(relax·max(1,|b|), constr_viol_tol)`), so its effect on the
reported objective is unbounded in `|∇f|`. At `|∇f| = 1e12` a `1e-8` widening
is a `1e4` objective error. Nothing in the convergence test sees this: the
run terminates with `final_kkt_error = 1.59e-14` (scaled) while
`final_unscaled_kkt_error = 2.66e-05`, and `final_declared_box_viol` correctly
reads `1.0e-08` — the measurement is already there, nothing gates on it.

**The two arms disagree on this model today.** Same binary, same model:

```
solver_selection=nlp      f = -1.0000000e+04   x[0] = -1.0000e-08
solver_selection=auto     f = +7.8126070e-09   x[0] = +7.8116e-21
solver_selection=lp-ipm   f = +7.8126070e-09   x[0] = +7.8116e-21
```

The convex arm is already immune because gh#760 made `bound_relax_factor`
opt-in there ("the convex arm solves the model you declared"). This is the NLP
arm lacking a property the convex arm has.

## Rejected: default `honor_original_bounds` to `yes`

It is exact on this family — `f = 1.000000e-12`, `x[0] = 0.0` at every `C` —
and the fixture sweep looks clean: 27 of 194 legs move, **objective column
only**, with status, iteration count, engine and escalation counts
byte-identical on all 194. All 27 are `nlp`; the 82 `cvx-qp` and 8 `cvx-qcqp`
lines do not move.

That sweep is misleading, and the way it misleads is the point. It records
status, objective, iterations and engine — **not feasibility** — so it is
uniform in the dimension the change acts on. Re-measured by running every
returned `.sol` back through `pounce verify --feas-tol 1e-12`:

| | baseline | `honor_original_bounds=yes` |
|---|---|---|
| VERIFIED | 52 | 61 |
| REJECTED | 45 | 36 |

9 fixtures move REJECTED → VERIFIED and none the other way — but **declared
row violation increases on 10 of 97**:

```
autocorr_bern55-06        1.48e-08 -> 2.78e-05    (1900x)
mu_fallback_point_floor   7.76e-08 -> 1.71e-06
csfi2                     2.00e-08 -> 2.59e-07
deb7                      3.61e-08 -> 1.62e-07
linear_eq_collapsed_box   4.44e-16 -> 9.99e-09
mpcc_worse_local_solution 3.33e-16 -> 9.54e-09
nonconvex_qp_eq           0        -> 8.75e-09
nonconvex_qp              2.22e-16 -> 8.75e-09
```

Bound violation increases on none. So the projection is a straight **trade of
box feasibility for row feasibility**: clamping `x` into the box pushes it off
the equality manifold, taking residuals that were exactly zero or at machine
epsilon to ~1e-8. It is a post-hoc clamp that knows nothing about the rows —
which is what `crossover_issue612.rs:337` means by "what `honor_original_bounds`
exists to **paper over**".

It also leaves the report incoherent: under `honor_original_bounds=yes` the run
reports `final_declared_box_viol = 1.00e-08` while the returned `x[0]` is
exactly `0.0`. As upstream documents and `orig_ipopt_nlp.rs:1518` repeats, the
residuals describe the **non-projected** point. That is very likely why
upstream moved this default to `no` in 3.14.

Keep it opt-in. It is the right answer for a caller who knows their box matters
more than their rows — a Pyomo `Var` whose value is loaded back into its own
bounds, a downstream `sqrt(1-x)` — and the wrong default for everyone else.

## Rejected: cap the relaxation by objective sensitivity

Prototyped on `prototype/967-bound-relax-obj-cap` (commit `013f7cb0`, do not
merge). `bound_relax_obj_cap` caps coordinate `i`'s widening at
`cap / |∂f/∂xᵢ|` at the lifted starting point. Off by default it is
bit-identical — byte-identical sweep on all 194 legs, all five `relax_bounds`
unit tests pass.

On the reproducer it works: absolute error bounded at ~6e-06 across
`C = 1e2..1e12` instead of growing as `C·1e-8`, never negative, `+1` iteration
at `C = 1e12`. It is still disqualified, three ways:

1. **Vacuous where it is most needed.** `units_qp_k1` has `∇f(x₀) = [0, 0]`,
   so the cap is `INFINITY` and the option is inert even at `1e-30`. The
   sensitivity that matters is at the *solution*; the relaxation is applied
   *before* the solve. Only near-linear objectives are protected — the
   reproducer's class, not the general one.

2. **It reaches a point the suite already names as a regression, and does so
   non-monotonically.** `mpcc_worse_local_solution`: default `-13.005680`,
   `cap=1e-8` → `-1.207234`, which
   `issue884_promotion_gate_reads_the_answer.rs:32` lists as "a strictly worse
   feasible point". `cap=1e-6` and `cap=1e-10` both give `-13.005680`. A
   safeguard whose harm is not monotone in its own knob cannot be tuned around.

3. **3.8× iterations on the canonical gh#544 fixtures.** `pooling_rt2stp`
   exact `109 → 418`, `deb7` lbfgs `715 → 2732`, and `pooling_rt2stp`'s two
   legs *swap* answers (exact `-3273.95 → -4391.83`, lbfgs
   `-4391.83 → -3273.95`) — it perturbs which basin is reached rather than
   improving anything.

Zero status changes across all 194 legs. That is the trap: the suite asserts
status and objective, so this would have passed review unseen, exactly as
gh#544 did.

## Rejected: gate the status on it

`kkt_fidelity_tol` (gh#173) already does this and is off by default — it
downgrades a `Solve_Succeeded` whose `final_unscaled_kkt_error` exceeds the
threshold. Measured at `kkt_fidelity_tol=1e-8` the reproducer downgrades to
`Solved_To_Acceptable_Level` from `C = 1e6` up. It does **not** close the
complaint, for two reasons:

- It relabels without fixing the number: `res.fun` is still `-1.0e+04`.
- `Solved_To_Acceptable_Level` maps to `success=True` in the Python wrapper,
  so the consumer that trusted `success` still trusts it.

Combining it with `honor_original_bounds=yes` is worse than either: the answer
is exact and the status reads `Solved_To_Acceptable_Level`, because the
residuals are measured at the un-projected point.

The gh#200 masked-certificate guard is *not* the lever either, though its log
line looks like it should be. Its contract is "never worse off": it refuses the
masked certificate, continues, and — when continuing achieves nothing, which
here it cannot, the residual being an irreducible widening artifact —
`honour_refused_certificate` restores the refused point *with the status it
would originally have had*. Working as specified.

## What a caller should do

| situation | remedy |
|---|---|
| LP or convex QP | let `solver_selection=auto` route it; that arm solves the declared model. The Python frontend defaults to `nlp`, so say `auto` explicitly. |
| need a vertex | `crossover=yes`. Exact at `C ≤ 1e6` on this family (`5.2e-18` at `1e2`, `5.1e-11` at `1e6`); does not engage at `1e12`. |
| the box matters more than the rows | `honor_original_bounds=yes`. Exact here at every `C`; read the trade above first. |
| want to know it happened | `final_declared_box_viol` in the `info` dict / solve report, and the bold red `Violation of the model as declared` line in the CLI. |

## What `Solve_Succeeded` guarantees

It is a statement about the **scaled** KKT residuals of the **relaxed** model.
It is not a bound on the unscaled objective error, and where `nlp_scaling`
deflates the objective it can be many orders from one:
`final_kkt_error = 1.59e-14` and `final_unscaled_kkt_error = 2.66e-05` on the
same exit. A consumer that needs an unscaled guarantee should read
`final_unscaled_kkt_error` and `final_declared_box_viol`, or set
`kkt_fidelity_tol`, rather than branching on the status alone.
