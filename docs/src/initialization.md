# Initialization and Warm Starts

POUNCE is a local NLP solver: every solve starts from a point, and that
point often decides whether the solve takes 15 iterations or 150, or
whether it converges at all. This page collects the initialization
story in one place: where the starting point comes from on each
frontend, what the solver does with it (the part that surprises
people), how to warm-start each algorithm path, and how to diagnose a
bad start. The per-algorithm details live in their own pages; this is
the map.

## Where the starting point comes from

| Frontend | Primal starting point |
|---|---|
| Python `Problem.solve(x0=...)` | the `x0` argument |
| Python `minimize(fun, x0, ...)` | the `x0` argument |
| CLI / AMPL | the `.nl` file's initial-guess segment; zeros for variables without one |
| Pyomo | each `Var`'s `.value`, serialized into the `.nl` by Pyomo's writer |
| GAMS | variable levels (`x.L`) via GMO |
| Rust | `Nlp::new(problem).x0(&[...])`, or `TNLP::get_starting_point` |

Two silent-zero traps hide in that table:

* **Pyomo:** a `Var` whose `.value` was never set is written as `0`
  in the `.nl` file. A model initialized "nowhere" is actually
  initialized at the origin, which for many process models is outside
  every variable's meaningful range (and a domain error for `log`,
  `/`, and friends).
* **GAMS:** levels default to `0` unless assigned. Set `x.L` before
  the `solve` statement.

Dual estimates can be seeded too: `Problem.solve` accepts
`lagrange=`, `zl=`, `zu=` keyword arguments, and the `.nl` format
carries constraint-dual guesses when the modeling layer writes them.
Dual seeds are ignored unless you opt into a warm start (below). The
scipy-style `minimize` facade does not expose dual seeding; use
`pounce.Problem` directly when you need it.

## What the solver does with your point (cold start)

The default interior-point path ports Ipopt's iterate initializer
(`crates/pounce-algorithm/src/init/default.rs`). The sequence:

1. **The primal point is pushed into the interior of the bounds.**
   Per component, with bounds `lo <= x <= hi`:
   `p_l = min(bound_push * max(|lo|, 1), bound_frac * (hi - lo))`,
   likewise `p_u`, and `x` is clamped into `[lo + p_l, hi - p_u]`.
   One-sided bounds use the `bound_push` term alone; free variables
   are untouched. With the defaults (`bound_push = bound_frac =
   1e-2`), a variable sitting exactly on its lower bound `1.0` starts
   at `1.01` instead. Your point is honored *approximately*, and the
   deliberately-at-a-bound part of it is not honored at all. This is
   the single most common reason a "perfect" starting point does not
   behave like one.
2. **Slacks** are set to `s = d(x)` and pushed into the slack bounds
   the same way.
3. **Duals** get fixed defaults: constraint multipliers start at
   `y = 0` and are then replaced by a least-square estimate, unless
   that estimate exceeds `constr_mult_init_max` (in which case it is
   discarded and `y` stays at zero); bound multipliers are
   `z = v = bound_mult_init_val = 1.0`.
4. **The barrier parameter** starts at `mu_init = 0.1` (monotone
   `mu_strategy`, the default) regardless of how good your point is.

The knobs, all Ipopt-compatible, and all settable through every
frontend's option path (`Problem.solve(options={...})`, `pounce
model.nl bound_push=0.1`, an `ipopt.opt` line, `IpoptApplication::
options_mut`):

| Option | Default | Meaning |
|---|---|---|
| `bound_push` | `1e-2` | Absolute push off each bound (relative to `max(|bound|, 1)`). |
| `bound_frac` | `1e-2` | Cap on the push as a fraction of the bound interval. |
| `slack_bound_push` / `slack_bound_frac` | `1e-2` | Same, for inequality slacks. |
| `bound_mult_init_val` | `1.0` | Initial bound-multiplier value. |
| `bound_mult_init_method` | `constant` | `constant` is the only implemented mode; upstream's `mu-based` parses and is then refused rather than silently served as `constant`. |
| `constr_mult_init_max` | `1e3` | Cap on the least-square constraint-multiplier estimate; `0` keeps `y = 0`. |
| `least_square_init_primal` | `no` | Replace the starting `x` with the min-norm solution of the linearized constraints before the interior push — but only if that actually reduces the true nonlinear violation (see [Safeguarding the least-square start](#safeguarding-the-least-square-start)). |
| `mu_init` | `0.1` | Initial barrier parameter (monotone strategy). |
| `start_with_resto` | `no` | Jump straight into feasibility restoration at iteration 1 (aborts if the start is already feasible). |

An *infeasible* starting point is fine: the IPM does not require
feasibility, and `least_square_init_primal=yes` can cheaply reduce
iteration-0 infeasibility on mostly-linear models (the
`mehrotra_algorithm` LP/QP cascade turns it on for you, along with
more aggressive `bound_push` / `bound_frac` / `bound_mult_init_val`).
A point where a function *fails to evaluate* is not fine; see
[Diagnosing a bad start](#diagnosing-a-bad-start).

### Safeguarding the least-square start

The min-norm solution of the *linearized* constraints is a local model
step, not automatically a better starting point. Where the Jacobian is
small relative to the residual, the linearization asks for a very large
correction and the true nonlinear violation at the far end can be far
worse than where it started. On `x₀² + x₁² = 1` from `(0.05, 0.05)` the
Jacobian is `(0.1, 0.1)`, the linearized correction is about 7 units
long, and the violation at the far end is `48.5` against the `0.995` it
started with.

So the step is scored before it is taken. Writing `θ(x)` for the
unscaled max-norm nonlinear violation —
`max(‖c(x)‖∞, ‖max(d_l − d(x), d(x) − d_u, 0)‖∞)`, the same quantity the
CLI reports as the model's constraint violation — the initializer:

1. evaluates `θ₀` at your point, after the interior push;
2. computes the least-square direction `d = x_ls − x₀` once;
3. tries `α = 1, ½, ¼, …` (at most `least_square_init_max_trials`,
   default 4), pushing each candidate into the bound interior *before*
   measuring it, so the accepted merit is the merit of the point the
   algorithm will really start from;
4. accepts the first `α` with `θ(α) ≤ (1 − η·α)·θ₀`, where
   `η = least_square_init_accept_ratio` (default `1e-2`). The linear
   model predicts `θ → 0` at `α = 1`, so this is exactly "the actual
   feasibility reduction is at least `η` times the predicted one";
5. keeps your original `x` if no trial qualifies.

`least_square_init_max_trials` and `least_square_init_accept_ratio` are
fields on `DefaultIterateInitializer`, not registered options: unlike
every knob in the table above they are not settable from a frontend,
and setting them by name is rejected with `Unknown option`. They are
named here because the safeguard's behaviour is defined in terms of
them, not because you can tune them.

Each trial costs one constraint evaluation; none costs a Jacobian or a
KKT solve, because only the length of the step changes. A point that is
already feasible is left alone — no step can improve a violation of
zero.

The decision is readable after the solve:

```rust
if let Some(r) = app.least_square_init_report() {
    println!("{} -> {} (alpha {}, {} rejected, {})",
             r.violation_initial, r.violation_final,
             r.alpha, r.rejected_trials, r.termination);
}
```

From the CLI, where that accessor is not reachable, the same fields go
out once per solve at `debug` level:

```sh
RUST_LOG=pounce::algorithm=debug pounce model.nl model.sol \
    least_square_init_primal=yes
# DEBUG pounce::algorithm: pounce: least_square_init_primal safeguard
#   decision violation_initial=1.0 violation_final=0.25 alpha=0.5
#   step_norm=3.2596 rejected_trials=1 termination="accepted"
```

### What the safeguard costs, and why it is not tuned away

The guarantee above is about **the starting point's violation** — the
only quantity the test measures. It says nothing about the trajectory
that follows. A different, more feasible starting point on a nonconvex
model is entitled to reach a different local minimum and to converge
into a different tolerance band, and on this corpus two models do.

Sweeping the 57 CLI fixtures with `least_square_init_primal=yes`, with
the safeguard against without it (gh#616, measured on `a44f4e8b`):

| fixture | unsafeguarded | safeguarded | what changed |
|---|---|---|---|
| `csfi2` | `SolveSucceeded`, 53 it | `SolvedToAcceptableLevel`, 35 it | objective bit-identical at 55.0176045 |
| `eigenb2` | `SolveSucceeded`, 55 it | `SolvedToAcceptableLevel`, 57 it | 1.6 → 1.599999991 |
| `pooling_rt2stp` | −4391.826, 134 it | −3273.955, 81 it | **not a stable fact — see below** |
| `deb7` | 249.746, 479 it | 97.560, 202 it | different local optimum, much better |
| `eigena2` | `SolveSucceeded`, 78 it | `SolveSucceeded`, 65 it | |
| `unbounded_cubic` | `DivergingIterates`, 91 it | `DivergingIterates`, 290 it | unbounded either way |

Fourteen fixtures move in total; `SolveSucceeded` goes 46 → 44 and the
solved-or-acceptable set is unchanged at 46. Under **default options**
the two routes are bit-identical, because `least_square_init_primal`
defaults to `no`. Under `mehrotra_algorithm=yes` — which turns the
option on as part of its cascade — the same 27 fixtures solve to the
same objectives on both sides, at 2475 against 2463 total iterations.
Twelve fixtures move there. Ten of them fail on both sides
(restoration failure, detected infeasibility, a step-computation
error), so only the failure label and the meaningless objective it
carries change; the other two are `eigena2` and `eigenb2`, which solve
to the same objectives either way and differ by a single iteration.

Across the 57 fixtures the safeguard engages on 29: 16 accept, 8
decline every trial, and 5 start feasible and short-circuit. It is
inert on the other 28 — 26 are LP or convex-QP models the CLI
dispatches to `pounce-convex`, which does not run this initializer at
all, and 2 have no constraints for the step to act on.

The two downgrades are deliberate, and they are **not** a defect in the
accept test. Attributing every moving fixture through the report above
puts them in three different arms of the safeguard, which do not share
a mechanism:

* **`theta_0 = 0`, short-circuit.** `unbounded_cubic`, `unbounded_exp`,
  `boxed_qp_fixed_var` start feasible, so no direction is computed at
  all. `unbounded_cubic`'s 91 → 290 is the unsafeguarded path having
  taken a step from an already-feasible point; both routes return
  `DivergingIterates` on a model that is genuinely unbounded.
* **Declined.** `csfi2`, `deb7`, `pooling_rt2stp`,
  `linear_eq_aggregation`, `linear_eq_aggregation_row_constant`,
  `issue_372_infeasible_bounds`: every trial is worse than `theta_0`,
  so the user's point is kept.
* **Backtracked accept.** `eigena2`, `eigenb2`, `hs71_obj1e8`,
  `user_scaling_suffix`, `user_scaling_var_suffix` accept at
  `alpha < 1`.

`csfi2` is in the declined group. Its old `SolveSucceeded` came from
taking a step that raises the true violation above `theta_0 = 1508.55`
— exactly the step the safeguard exists to refuse. A *tighter* accept
test still declines it, so no tuning reaches it; only removing the
safeguard does. With the step declined, `csfi2` under
`least_square_init_primal=yes` now matches `=no` to the bit, which is
the least surprising thing an off-by-default option can do.

`eigenb2` is in the accepted group, and it is paired with `eigena2`:
the safeguard sees **bit-identical numbers** on both — `theta_0 = 1.0`,
accepted `theta = 0.2500000062500001`, `alpha = 0.5`, one rejected
trial, step norm `3.2596` — and `eigena2` improves while `eigenb2`
drops a tolerance band. Any criterion computed from the safeguard's own
inputs necessarily treats the two the same, so none can keep one and
drop the other. Two specific proposals were measured and rejected:

* **Retuning `least_square_init_accept_ratio`.** Acceptance is
  `theta_0 − theta >= eta·alpha·theta_0`, so `eigenb2`'s trial survives
  every `eta <= 1.5`, and `eta > 1` is meaningless (it would demand a
  negative violation at `alpha = 1`). No reachable setting rejects it.
* **A band that prefers the untouched point when the improvement is
  marginal.** `eigenb2`'s step is not marginal: it cuts the violation
  4×, the median of the sixteen accepted steps in the corpus and the
  same ratio as `airport`, `cresc4` and both
  `issue_508_infeasible_gap_*` fixtures, all of which are wins.
* **Requiring the accepted point not to degrade the dual residual.**
  Measured: iteration-0 `inf_du` *improves* on both, 100 → 13.9 on
  `eigena2` and 100 → 47.7 on `eigenb2`. The gate accepts the step.

So the downgrades are accepted as the cost of a route that is off by
default, and the corpus measurement is pinned by
`crates/pounce-cli/tests/issue_616_ls_init_downgrades.rs` rather than
left in a PR body.

#### One of the two downgrades has since gone away (gh#681)

Everything above is the measurement on `a44f4e8b` and is left as it was
taken. On current `main` plus gh#588's quadratic-structure work, the
cost is **one** downgrade, not two: `eigenb2` reaches `SolveSucceeded`
at 1.5999999999925176 in 54 iterations. `csfi2` does not move at all,
to the bit — it is in the declined group, and a decline is not a step,
so there is no trajectory for anything downstream to perturb.

Nothing about the safeguard changed. gh#588's Q4 evaluates a recognized
degree-≤2 row from its stored constant matrix instead of rebuilding an
AD tape each iteration, and that reassociates the sums in `eval_g` and
`eval_jac_g` — a difference that phase declares non-bitwise in advance,
because the tape adds one summand at a time in file order while the
matvec adds a merged row. `eigenb2` sat close enough to the acceptable
band for the reassociation to carry it across.

The pairing argument above is **strengthened**, not weakened, and it is
worth being explicit about why, because the obvious reading is that
gh#616 lost its evidence. `eigena2` and `eigenb2` still hand the
safeguard bit-identical numbers, and the safeguard still takes the same
decision on both: `theta_0 = 1.0`, `alpha = 0.5`, step norm
`3.2596011939729705`, one rejected trial, accepted. The only thing that
moved in that report is **two ulps** of the reported `violation_final`
(`0.2500000062500001` → `0.2500000062500003`), on both models together,
and it is a diagnostic rather than an input to the accept test.

So `eigenb2` crossed a tolerance band while every number the accept
test reads stayed put. Its downgrade was never a property of that test:
it was decided downstream, by where the iteration *after* the safeguard
landed relative to the acceptable band, and one reassociated sum was
enough to move it. An accept test retuned to chase `eigenb2` — either
of the two proposals rejected above — would have been tuned against
round-off. The conclusion is the same one gh#616 reached, now resting
on a mechanism rather than on a two-model coincidence.

The test file pins both legs: the fast path's verdict, and the tape's
under `POUNCE_DBG_NO_QUAD=1`, which still reproduces gh#616's
downgrade exactly. That is what makes a future move attributable to one
of them instead of being absorbed as noise.

#### And gh#693 removed the other one, by moving both models off the band

The section above argues that `eigenb2`'s downgrade "was never a
property of that test: it was decided downstream, by where the iteration
*after* the safeguard landed relative to the acceptable band". gh#693 —
which removed the Tikhonov `δ` from the least-square multiplier
initializer — is a clean test of that claim, and confirms it.

The safeguard's decision is bit-identical across gh#693 on every model
here: `csfi2` and `pooling_rt2stp` still decline with four rejected
trials, `eigenb2` still accepts at `alpha = 0.5` on a step of norm
3.2596011939729705 after one rejected trial. Nothing the accept test
reads moved. The outcomes did:

| fixture | 0.10.0 | with gh#693 |
|---|---|---|
| `eigenb2`, `=yes` | `SolvedToAcceptableLevel`, 48 it | `SolveSucceeded`, 17 it |
| `eigenb2`, `=no` | `SolveSucceeded`, 67 it | `SolveSucceeded`, 21 it |
| `eigena2`, `=yes` | `SolvedToAcceptableLevel`, 127 it | `SolveSucceeded`, 17 it |
| `csfi2`, `=yes` | `SolvedToAcceptableLevel`, 35 it | `SolvedToAcceptableLevel`, 35 it |

So the safeguard's measured cost is now `csfi2` alone.

The reason to trust that as a real improvement rather than another lucky
landing — which is what gh#588's Q4 turned out to be — is that it
survives a round-off screen. Re-running each model at 17 values of
`mu_init` at `0.1·(1 ± k·1e-12)`:

| fixture, `=yes` | 0.10.0 | with gh#693 |
|---|---|---|
| `eigenb2` | 14 `SolveSucceeded` / 3 `SolvedToAcceptableLevel` | 17 `SolveSucceeded` |
| `eigena2` | 11 `SolveSucceeded` / 6 `SolvedToAcceptableLevel` | 17 `SolveSucceeded` |
| `csfi2` | 17 `SolvedToAcceptableLevel` | 17 `SolvedToAcceptableLevel` |

On 0.10.0 neither status was a stable fact — `eigenb2`'s pinned
`SolvedToAcceptableLevel` was a three-point island around the default
draw, and the majority outcome at neighbouring draws was already the
other one. gh#693 does not carry these models across the band; it moves
them off it. `csfi2`, which is genuinely clear of the band, does not
move at all in either build, which is the control.

This also amends gh#706. That issue recorded `eigena2`'s status as
*platform*-dependent; it is round-off-dependent on a single platform,
6 draws in 17 at a `1e-12` perturbation, which is a simpler and worse
explanation. It is deterministic after gh#693.

### A declined step is not the same as never asking

Worth knowing before you read `least_square_init_primal=yes` results:
declining restores your `x` exactly, but it does not restore the
solver's state. Computing the direction has by then driven the first
factorization through the augmented-system solver, on the `W = 0`
least-square matrix rather than on the first real KKT matrix.

gh#616 isolated this by forcing a decline on either side of that call.
Declining *before* the augmented-system solve is bit-identical to
`least_square_init_primal=no` on every fixture; declining *after* it is
bit-identical to the real safeguard. So the carrier is that one solve,
not the staging or the trial evaluations — those are free.

It shows on two of the eight declining fixtures: `pooling_rt2stp` takes
298 iterations with the option off and 81 with it on and declined, and
`deb7` takes 154 against 202. Everywhere else declining and `=no` agree
exactly.

**Which local optimum each route reaches on `pooling_rt2stp` is not a
stable fact, and this page previously reported it as one.** The table
above records the two routes reaching *different* optima (−4391.826
against −3273.955); the test that pins this behaviour,
`issue_616_ls_init_downgrades.rs`, simultaneously asserted they reach
the *same* one. Both were written from a single draw, and neither
noticed the other. Re-measured across 17 values of `mu_init` at
`0.1·(1 ± k·1e-12)` — a perturbation at round-off scale — the two routes
agree on the optimum at **10 of 17 points** on 0.10.0 and 8 of 17 after
gh#693. This model is bistable between −3273.955 and −4391.826 and a
single run picks a side essentially at random.

What *is* stable, at 17 of 17 points on both builds, is that the two
routes take different numbers of iterations — which is the carry-over
this section is about. The test now asserts that and nothing more. Making the decline a
true no-op would need a separate augmented-system solver for the
initializer; it was not done, because it is a trajectory change that
costs `pooling_rt2stp` 81 → 298 iterations to buy a tidier contract on
an off-by-default option.

## Warm-starting the interior-point path

From Python, the packaged form is one object:

```python
x, info = prob.solve(x0=x0)                  # cold solve
ws = pounce.WarmStart.from_info(x, info)     # captures x, duals, mu
x2, info2 = prob.solve(warm_start=ws)        # warm re-solve
ws.save("state.npz")                         # reuse across processes
```

`warm_start=` is accepted by `Problem.solve` and `pounce.minimize`,
seeds the primal and dual iterates, applies the enabling options
below, and forwards the SQP working set when the state was captured
from that path. The rest of this section is what it does under the
hood (and the only route from the CLI or an options file).

The enabling options are **scoped to the call**: they are installed for
that one solve and taken back afterwards, including when the solve
raises. A warm solve therefore never changes what the next ordinary
`solve` on the same `Problem` does. (Before pounce#607 it did, and the
cost was invisible: on HS071 an ordinary cold solve went from 17
iterations to 24 on a `Problem` that had served one warm solve, with the
same objective to ten digits.)

Passing a previous solution as `x0` is **not** a warm start by
itself. The IPM warm start is a package of three things, and skipping
any one of them silently degrades to (roughly) a cold solve:

1. **Opt in and seed the duals.** Set `warm_start_init_point=yes` and
   pass the previous multipliers.
2. **Lower `mu_init`.** The default `0.1` makes the solver walk the
   barrier schedule down from scratch even when started at the
   optimum. Seed it near the converged complementarity (e.g. `1e-7`
   after a `tol=1e-8` solve). Since #606 this is a *floor*: the solver
   measures the point you supplied and raises `mu` above it if the
   point cannot support that barrier (see below).
3. **Tighten the warm-start pushes.** The warm initializer applies
   its own interior clamp with `warm_start_bound_push` / `_frac`
   (default `1e-3`), which shoves an at-the-bound solution back off
   its bounds. Tighten them to keep the point.

```python
x, info = make_problem().solve(x0=x0_cold)      # cold solve

warm = make_problem()
warm.add_option("warm_start_init_point", "yes")
warm.add_option("mu_init", 1e-7)
for k in ("warm_start_bound_push", "warm_start_bound_frac",
          "warm_start_slack_bound_push", "warm_start_slack_bound_frac",
          "warm_start_mult_bound_push"):
    warm.add_option(k, 1e-9)

x2, info2 = warm.solve(
    x0=x,
    lagrange=np.asarray(info["mult_g"]),
    zl=np.asarray(info["mult_x_L"]),
    zu=np.asarray(info["mult_x_U"]),
)
```

On HS071 this takes the re-solve from 11 iterations to 5, while
`warm_start_init_point=yes` alone saves nothing; the full runnable
comparison is `python/examples/hs071_warm_start.py`. On the CLI the
same options apply as `KEY=VALUE` pairs, with dual seeds coming from
the `.nl` file's dual segment when present.

| Option | Default | Meaning |
|---|---|---|
| `warm_start_init_point` | `no` | Master switch: honor supplied primal *and* dual seeds. |
| `warm_start_bound_push` / `warm_start_bound_frac` | `1e-3` | Interior clamp used instead of `bound_push` / `bound_frac`. |
| `warm_start_slack_bound_push` / `warm_start_slack_bound_frac` | `1e-3` | Same, for slacks. |
| `warm_start_mult_bound_push` | `1e-3` | Floor on seeded bound multipliers (a carried-in `z = 0` must not start on the barrier's boundary). |
| `warm_start_mult_init_max` | `1e6` | Cap on seeded equality multipliers. |
| `warm_start_recentering` | `residual` | Reconstruct the multiplier blocks the caller did not supply, and raise `mu` when the supplied point cannot support it. `none` restores the pre-#606 constants. |

### What the solver does with a partial warm start

Two things about the list above are worth knowing before you tune it.

**You cannot seed every multiplier block.** `TNLP::get_starting_point`
— which is what `lagrange` / `zl` / `zu` reach — carries the equality
multipliers and the *variable*-bound multipliers. The interior-point
method also needs a multiplier for each inequality row's slack
(`v_L` / `v_U` internally), and there is no field for those on any
frontend. On every warm start ever run they arrived as zero and were
floored at `warm_start_mult_bound_push`.

**A constant is the wrong fill.** `warm_start_mult_bound_push` is a
number chosen with no reference to the slacks it is paired against, so
the "warm" point it produces is not a stationary point of anything.

Under `warm_start_recentering=residual` (the default since #606) the
initializer instead rebuilds what it was not given:

- a bound-multiplier entry that arrives as exactly `0` (or `NaN`) is
  not a legal barrier multiplier, so it was never a seed; it is
  re-derived from the stationarity identity
  `P_L z_L − P_U z_U = ∇f + J_c^T y_c + J_d^T y_d` (and its slack-block
  twin `P_L v_L − P_U v_U = −y_d`), floored at `μ / slack` so an
  inactive bound still gets the value complementarity implies and
  capped at ten times that floor, so a stationarity *miss* cannot be
  laundered into a multiplier (#617);
- an equality-multiplier block that is identically zero goes through
  the same regularized least-squares augmented solve the cold path
  uses, now with real bound multipliers in its right-hand side;
- `mu` is raised to the point's measured average complementarity when
  that exceeds `mu_init` **by more than a factor of ten**, so a stale
  seed gets a looser barrier instead of being trusted while a merely
  imperfect one keeps the barrier it asked for. Moving `mu` reroutes
  the whole trajectory, so a near miss is not worth what the reroute
  costs. The measurement is clamped to `[1e-11, 0.1]`; `mu_init`
  itself is not, so an explicit setting outside that band is a floor
  and is never capped. The primal and dual residuals deliberately do
  **not** move `mu`: a warm point at a moved parameter carries both by
  construction, and reacting to them discards the warm start to pay for
  a Newton step that was about to happen anyway.

`warm_start_target_mu`, when set, still pins `mu` outright.

### A seed the solver will not believe

Everything above derives the blocks you did not supply *from* the ones
you did, so a supplied block that does not describe your point gets
propagated rather than caught. Two guards bound that (#617, #618). Both
are as conservative as the `mu` rule above, and for the same reason —
refusing a seed reroutes the trajectory exactly as much as trusting a
bad one does.

**A dual block that cannot belong to this primal point is refused.**
Each seeded bound-multiplier block is measured on the quantity the
barrier *is*: `|z_i| · s_i`, averaged over the entries you actually
seeded. A point on any central path — converged, mid-solve, or stale —
carries that at the order of its own barrier, and a point that misses
feasibility by `inf_pr` may carry it at that order too. A block reading
ten times above **both** cannot have come from a solve of this problem,
so it takes the pre-#606 constant fill and stops being an input to
anything. The equality block gets the matching test — a `y` whose
stationarity residual dwarfs `∇f` and the multipliers *you* supplied is
not this point's `y` — and while the block itself is left where you put
it (there is no constant to fall back to), the split no longer runs off
it.

**A slack the point's own infeasibility swamps is not a measurement.**
Both halves of the bound reconstruction read a small slack as "this
bound is active". On a point that misses feasibility by more than the
slack itself, that reading is not available, so those entries keep the
pre-#606 constant instead. It is a per-entry test, so a partly-stale
seed keeps the reconstruction exactly where its slacks still outrun the
residual. The comparison is made against `inf_pr` only once `inf_pr`
clears the barrier by a factor of ten — a converged solve leaves
`inf_pr` at its own tolerance, routinely above the pushed slacks, and
comparing the two unguarded would throw away the reconstruction on the
exact restarts it exists for.

Neither guard fires on a good seed: an exact same-model restart is
bit-identical to what #606 shipped.

What happened is reported back. From Python it is `info["warm_start"]`:

```python
x2, info2 = warm.solve(x0=x, zl=..., zu=...)
info2["warm_start"]
# {'primal_residual': 1.6e-09, 'dual_residual': 3.5e-10,
#  'complementarity': 4.2e-09, 'mu_in': 2.5e-09, 'mu_out': 4.2e-09,
#  'bound_duals': 'reconstructed', 'eq_duals': 'accepted',
#  'bound_duals_reconstructed': 1, 'bound_duals_rejected': 0,
#  'eq_duals_rejected': False, 'stationarity_split': True,
#  'recentering_disabled': False}
```

`bound_duals` reads `rejected` when a seeded block was refused, and
`bound_duals_rejected` counts the entries; the verdicts are per block,
so a model can refuse the blocks you seeded and still reconstruct the
slack-bound blocks nobody can seed, and the two counters keep that
legible.

From Rust it is `IpoptApplication::warm_start_diagnostics()`. At
`print_level=5` the iteration line carries `wz` (bound multipliers
rebuilt), `wz!` (a seeded bound block was refused), `wy` (equality
multipliers rebuilt), `wy0` (a reconstruction was discarded), `wy!`
(the seeded `y` was refused as an input) and `wmu` (the barrier was
loosened).

### Two options that are refused

`warm_start_same_structure` and `warm_start_entire_iterate` are
registered — an `ipopt.opt` written for Ipopt parses unchanged — but
both name Ipopt's `TNLP::GetWarmStartIterate` surface, which pounce
does not expose. Setting either to `yes` used to parse, set a field
nothing read, and change nothing at all. Since #606 it fails with a
message instead. `warm_start_init_point=yes` is the supported route
and carries the primal point and every multiplier block the TNLP
surface has.

### Which model does this warm start belong to?

A warm start is a point in *one* model's variable space, with
multipliers in that model's constraint space. Replay it against a model
whose variables have been reordered, whose bounds have moved, or which
is simply a different model of the same shape, and the arrays are still
the right *length* — so nothing objects. What comes back is a wrong
answer, or the right answer down a much longer trajectory.

Pass `problem=` when you capture, and the object records a
**signature** of the model as well: dimensions, the bound signature, the
declared sparsity, the scaling convention, the algorithm/backend, the
model-defining options, and an order-sensitive probe of the model
itself.

```python
ws = pounce.WarmStart.from_info(x, info, problem=prob)
ws.save("state.npz")

# ... later, possibly in another process
ws = pounce.WarmStart.load("state.npz")
x2, info2 = prob.solve(warm_start=ws)     # checked before the solver runs
```

A mismatch is refused *before the solver is entered*, with a report
naming every facet that moved:

```text
warm start is not compatible with this problem (1 mismatch,
exact-structure replay, schema v2):
  - bounds: captured '51e5c8cd33c97b92', target '42ae305673e91939'
resolve it by one of:
  - re-capture against this problem: WarmStart.from_info(x, info, problem=prob)
  - transfer it explicitly: ws.transfer(prob, mapper) or, with stable IDs
    on both sides, ws.reindex(prob)
  - assert it transfers as-is: ws.migrate(prob)
  - downgrade the check: compat='warn' or compat='unsafe'
```

`compat` picks how hard that is enforced — `"strict"` (the default)
raises, `"warn"` emits the same report as a warning and proceeds,
`"unsafe"` skips the comparison. Set it on the object, on `load()`, or
per call: `prob.solve(warm_start=ws, compat="warn")`.
`ws.describe_compatibility(prob)` returns the report as a string without
raising, which is the dry run for a replay you are unsure of.

#### Reordered variables

Every facet listed above is a digest of what the model *declares*, and
none of them can see a **reordering**: permuting a model with a uniform
box and a dense jacobian leaves the bound digest and the sparsity digest
bit-identical. Replaying through one produced objective 16.0909 against
a true 17.0140 on permuted HS071, with nothing raised (#621).

So the signature also records a **probe**: the model evaluated once at a
fixed point inside the bounds, summarized order-sensitively. A
permutation moves those numbers, so it is refused with no help from you:

```python
ws = pounce.WarmStart.from_info(x, info, problem=prob)
reordered_prob.solve(warm_start=ws)        # refused — no var_ids needed
```

```text
warm start is not compatible with this problem (1 mismatch,
exact-structure replay, schema v2):
  - probe: this problem's model does not evaluate to the same numbers as
    the one the warm start was captured against (a reordering of the
    variables looks exactly like this; so does a different model of the
    same shape)
```

The probe costs one model evaluation at capture, and one at replay only
when the artifact carries a probe to compare against — 0.15 ms on a
4-variable model and 1.7 ms at 10 000 variables, or 1.2% and 0.002% of a
cold solve of the same model. It is a fixed 20 floats in the artifact
whatever the problem size. Decline it with `probe=False` for a model
whose evaluation is expensive or has side effects:

```python
ws = pounce.WarmStart.from_info(x, info, problem=prob, probe=False)
```

The comparison is to a **relative tolerance** (`PROBE_RTOL`, 1e-9), not a
hash equality: a model does not have to be bitwise reproducible to
replay. Re-associating a model's internal sums — what a different BLAS
or thread count does — moves the probe by 5e-18 relative and is
accepted.

**Stable IDs remain the rigorous answer**, for two reasons. The probe
infers ordering from arithmetic, so a model that is genuinely symmetric
under the permutation looks unchanged to it; and the probe can only
*refuse* a reordering, where IDs let `reindex` repair it:

```python
ws = pounce.WarmStart.from_info(x, info, problem=prob,
                                var_ids=names, con_ids=con_names)
...
prob2.solve(warm_start=ws, var_ids=names_in_prob2_order)   # refused, by name
ws.reindex(prob2, var_ids=names_in_prob2_order)            # ...or repaired
```

The probe is best-effort, and unavailable in three cases: an artifact
captured with `probe=False`, an artifact written before #621, and a
model that will not evaluate at an arbitrary interior point or answers
with a NaN. Each leaves the facet unrecorded, which reads as
*unverifiable* rather than incompatible — the replay still proceeds. When
neither the probe nor IDs were available on both sides, the report says
so rather than claiming a clean bill of health:

```text
warm start is compatible with this problem
  (note: neither a model probe nor stable IDs were available on both
  sides, so a pure reordering of the variables would not have been
  caught here (pounce#621). ...)
```

`describe_compatibility()` is the dry run, though, and you have to know
to call it. The enforcing path — `check_compatible()`, which
`solve(warm_start=...)` takes for you — says the same thing as a
`WarmStartOrderingUnverifiedWarning` (#660), so a replay that could not
have ruled a reordering out is never silent. Nothing disagreed, so it is
a warning and not a refusal; if you would rather refuse, promote it:

```python
import warnings
warnings.simplefilter("error", pounce.WarmStartOrderingUnverifiedWarning)
```

### Transferring a warm start: horizon shifts and reindexing

When the model *has* changed and you know how, say so. `transfer()`
takes a mapper and produces a **mapped** replay — labelled as such, and
still refused on any problem other than the one it was mapped to:

```python
def shift(ctx):                       # ctx: source, target, problem
    m = ctx.index_map("var")          # target-indexed source positions, -1 = new
    return {"x": ..., "lagrange": ..., "zl": ..., "zu": ...}

moved = ws.transfer(next_prob, shift, var_ids=next_ids, con_ids=next_con_ids)
```

With stable IDs on both sides, `reindex` writes that mapper for you —
entries the target shares with the source move to their new positions,
and entries only the target has are the freshly-entered stage of a
receding horizon:

```python
moved = ws.reindex(next_prob, var_ids=next_ids, con_ids=next_con_ids)
x, info = next_prob.solve(warm_start=moved)
```

That covers both cases: a reordering, where the ID sets are equal, and a
receding horizon, where they overlap.

#### What goes in the new stage

The two blocks of a prolongated stage are answered differently, and both
answers were measured (pounce#622).

Its **multipliers** are left *unseeded* — `NaN`, which the warm
initializer reads as "you decide" — rather than fabricated. Since
pounce#606 the solver reconstructs each unseeded *bound* multiplier
from `μ̂ / slack`, the complementarity relation it is about to enforce,
which is a better number than anything this side can invent. That
reconstruction needs no dual to work from, only the slacks the point
already determines, so it runs however much or little you seeded
(pounce#622).

The *equality* multipliers are the asymmetric half. Completing those
takes a least-squares solve that a partial seed can support and a bare
point cannot, so a state carrying no duals at all gets them reported
`unseeded` and left at the constant fill — deliberately, with its own
measurement behind it (deriving them from a primal-only seed cost
1102 → 1211 iterations across the 27 parametric paths in
`benchmarks/warmstart`). `info["warm_start"]` reports the split
directly: `eq_duals: accepted` for a mapped replay against `unseeded`
for a values-only one.

So what the carried multipliers buy is that equality block, not the
bound blocks. Dropping them is close to a wash on iteration count here
— 44/55/55/47 against 45/50/54/46 over the eight-step loop tabulated
below — and the reason to carry them is that they are the only thing
that *can* carry `y` across the shift.

Its **primal values** are the `fill_x` argument, and they matter more
than they look:

| `fill_x` | what lands in the new stage |
|---|---|
| `"prolong"` (default) | Repeat the last stage. When the identifier map is a pure shift — every matched entry the same distance from its counterpart, which is what a receding horizon *is* — that distance is the layout's own period, and each new entry takes the value one period behind it, clipped into its box. One variable per stage or `(p, v, u)` interleaved, the value lands in the same *kind* of slot; a tail longer than one stage repeats the terminal stage. Not a shift (a reordering, an interpolation) means no stage to repeat, and this degrades to `"zero"`. |
| `"zero"` | Zero clipped into the variable's box: independent of the point, and of the model. The pre-#622 default. |
| an array or scalar | Your values, used as-is — nothing is prolongated on top of an explicit answer. |

The default is worth what it costs to state. On a chain with a slew
limit, `"zero"` enters the new stage 2.25 away from feasible — the new
variable starts at `0` next to a neighbour at `2.75` under a limit of
`0.5` — and the filter's first iterations go on walking that back: 11
iterations against 7 for a cold solve. `"prolong"` enters it *feasible*
(primal residual 1.7e-10) and costs 8, and over the closed loop 21
against cold's 22.

Where the transfer pays properly is over a sequence, at a horizon long
enough to have something to carry. Eight steps of a receding horizon on
the sinusoidal tracking family, total iterations, transferred against
cold:

| horizon | transferred | cold |
|---|---|---|
| 5 | 45 | 67 |
| 10 | 50 | 75 |
| 20 | 54 | 77 |
| 40 | 46 | 76 |

`"zero"` runs the same loop in 42 / 47 / 55 / 53 — ahead at the two
short horizons, behind at the two long ones, and 27 against 21 on the
slew fixture. The default is not the one that wins every row; it is the
one that never hands the solver a point the model itself rejects. When
your own prolongation is better than repeating a stage — a simulation
step, a tangent predictor — `transfer()` with an explicit mapper is
where it goes.

Those numbers are `66cc1d4` + pounce#622, and they are the *reverse* of
what this page said before pounce#620: a transferred start used to lose
to a cold solve by more the longer the horizon got. Residual-adaptive
recentering (pounce#606/#620) is what turned that around; the fill
policy above is what fixed the case it did not reach.

#### Could a better transfer do better? (pounce#622)

Yes, by about 2x — and not by any of the obvious routes, so the
measurements are recorded here rather than left for the next person to
re-run. Bound the question with oracles no transfer can beat: seed each
window with the *next* window's converged answer. Eight-step receding
horizon, total iterations:

| horizon | cold | shipped | perfect primal | perfect primal+dual |
|---|---|---|---|---|
| 5 | 67 | 45 | 21 | 9 |
| 10 | 75 | 50 | 24 | 12 |
| 20 | 77 | 54 | 26 | 18 |
| 40 | 76 | 46 | 23 | 18 |

So the barrier's own floor is about one iteration per warm step, and
roughly half of what the shipped transfer spends is the zero-order
prediction rather than the interior-point method. Two ways of
collecting it were measured and neither works:

**A finite-difference (secant) predictor** — each variable stepped by
its own drift across the last two solves, which stable identifiers make
directly observable — is *worse than zero-order everywhere*: 59/75/66/60
against 45/50/54/46, and at horizon 10 no better than a cold solve. The
prolongated point is feasible to 1e-10; extrapolating pushes it off the
constraint manifold and breaks the pairing between the carried
multipliers and the new slacks. This is the same failure
`docs/src/continuation.md` records for the predictor at horizon 80.

**The KKT tangent** (`pounce.Solver.parametric_step`, the machinery
behind the `pred-ipm` arm) cannot be pointed at a horizon shift at all,
for a reason worth stating plainly: **a receding horizon is not a
parametric perturbation.** On the stages two consecutive windows share,
theta does not move — the same physical targets are in force. What
changes is *which stages exist*: one leaves, one enters. Fed a shift,
`parametric_step` is handed a delta vector of exact zeros and correctly
returns a zero step, so the "predictor" is bit-identical to the
zero-order transfer. Give the same family a parameter that genuinely
moves — an MPC initial-condition pin — and the tangent becomes
non-trivial and then degrades with horizon: 52/87/95/137 against the
zero-order 54/78/81/93 at horizons 5/10/20/40, ahead only at the
shortest, and worst where there are the most active-set events per step.

What is left, then, is the part no first-order step can supply: the
freshly-entered stage has no history to extrapolate *from*. Closing the
gap means predicting it from the model — a dynamics rollout, which
`transfer()`'s mapper already lets you supply and which only you can
write — or changing method, which is what the active-set SQP path is
for.

Two things it still does not buy you. A single hand-off across a *large*
parameter step on a *small* model is not where warm starting wins —
on the five-variable slew fixture, whose targets move by 2.0 per stage,
the transferred point costs 8 iterations against a cold solve's 7 to 9
(the cold arm's own spread over where you put the guess). And the
underlying barrier/active-set limit described just below has not gone
anywhere; it is still the reason the SQP path exists.

### Artifacts written before pounce#607

Archives from earlier releases carry no signature. They are
*unverifiable*, not incompatible, so they still load and still replay;
what you get is one `WarmStartLegacyWarning` and a dimension check
(the only facet their own arrays witness). Two ways to clear it:

```python
ws = pounce.WarmStart.load("old.npz")      # warns on replay
ws = ws.migrate(prob)                      # re-sign it against this problem
ws.save("old.npz")                         # ... and it is a v2 artifact now
```

`migrate` is an **assertion**, not a conversion: it re-signs the arrays
without touching them, so use it only when they really do belong to this
problem. When they need rearranging, that is `reindex` / `transfer`. An
unsigned warm start held only in memory — `from_info(x, info)` with no
`problem=` — behaves exactly as it always has, and says nothing.

Even a well-executed IPM warm start has a structural limit: the
barrier pushes iterates off the bounds, so the active-set information
in a converged solution cannot be fully exploited. When you are
solving a *sequence* of related NLPs (MPC steps, branch-and-bound
nodes, homotopy paths), that limit is the reason the active-set SQP
path exists.

## Warm-starting the active-set SQP path

With `algorithm=active-set-sqp`, the warm-start payload is different:
alongside the primal/dual seeds it carries the **working set** (which
bounds and constraints are active), and an unchanged working set means
the next solve converges in a handful of QP iterations.

```python
prob.add_option("algorithm", "active-set-sqp")

ws = None
for k in range(horizon_steps):
    x, info = prob.solve(x0=x_prev, working_set=ws)
    ws = info["working_set"]
    x_prev = x
```

The two paths' warm-start inputs are deliberately path-local: the
IPM-side options above (`warm_start_init_point`, `mu_init`,
`bound_push`, ...) are silently ignored on the SQP path, and
`working_set=` is ignored on the IPM path. Details, the
`classify_working_set` helper for reconstructing a working set from
multipliers, and the GAMS `sqp_state_file` / marginal-based routes are
in [Active-Set SQP & Warm Starts](active-set-sqp.md). Note the GAMS
warm-start features currently live in the native C link only, not the
pip link (see [GAMS](gams.md)).

## Sequences of solves: batch chaining and sessions

For MPC chains, parametric sweeps, and B&B node relaxations from
Python, `solve_nlp_batch` packages the whole IPM warm-start recipe
for you:

```python
results = pounce.solve_nlp_batch(batch_t)                   # cold
results = pounce.solve_nlp_batch(batch_t1, warms=results)   # warm
```

Each instance is seeded with the previous primal and duals, the
converged `mu` is threaded into `mu_init`, and
`warm_start_init_point=yes` is forced; see
[Batched NLP solving](python.md#batched-nlp-solving-solve_nlp_batch).
For post-solve sensitivity queries against the converged KKT factor
(a different kind of reuse, no re-solve at all), see
[Sessions](sessions.md). JAX users get warm-start hand-off along a
parameter trajectory via `JaxProblem`; see
[the Python guide](python.md).

## Diagnosing a bad start

The first stop is the preflight check, which evaluates the model once
at its starting point (no solve) and reports everything this page has
warned about: NaN/inf evaluations, bound violations, how far the
interior clamp will move the point, initial constraint violation,
derivative scale spread, and the factors automatic scaling will pick
here.

```sh
pounce check-x0 model.nl              # text report; --json for tools
pounce check-x0 model.nl --x0-file candidate.txt
pounce check-x0 model.nl --scaling-max-gradient 10   # preview another cutoff
```

```python
report = pounce.preflight(problem_obj, x0, lb=lb, ub=ub, cl=cl, cu=cu)
print(report)          # report.fatal, report.warnings, report.to_dict()
```

Exit code 0 means the model evaluates cleanly at x0 (warnings allowed);
21 means a solve from this point would abort. The other diagnostics:

* **`Invalid_Number_Detected`** means an evaluator returned NaN/inf,
  and the very first evaluation at the starting point is the usual
  culprit (`log(0)` or a division at an all-zeros default start).
  The interior clamp only repairs bound violations; it cannot fix
  domain errors on free variables. Move the start into the domain,
  or add bounds that keep the clamp inside it.
* **The `automatic scaling at x0` section** shows what
  `nlp_scaling_method=gradient-based` will actually do here: the
  objective factor, whether each row block clears the
  `nlp_scaling_max_gradient` cutoff at all, and — for a `.nl` model —
  the coefficient magnitudes of its quadratic rows. That last part
  exists because the automatic scaler is a *point sample*: a row like
  `x'Qx <= b` written about the origin has a zero Jacobian at
  `x0 = 0`, so it is left unscaled no matter how far `Q` and `b`
  disagree. See [Scaling](scaling.md#quadratic-rows-the-sampler-cannot-see).
* **`derivative_test=first-order`** runs the derivative checker at
  the starting point; wrong derivatives look exactly like a bad
  start (immediate restoration, tiny steps).
* **The [interactive debugger](debugger.md)** (`--debug`) breaks at
  iteration 0, so you can inspect the initial objective, `inf_pr`,
  and `inf_du` before a single step is taken, and `resolve` from an
  edited iterate.
* **Presolve** (`presolve=yes`) reports structural trouble that no
  starting point can fix, like rank-deficient equality blocks
  (LICQ check), and its bound tightening shrinks the box the
  interior clamp places you in. See
  [Troubleshooting Recipes](troubleshooting.md) and [FBBT](fbbt.md).
* **`pounce-studio analyze-nl`** gives a structural pre-flight of a
  model file without solving.

## Conditioning the starting point

Two options displace the point *before* the first barrier iteration.
Both are off by default and both are trajectory changes: they alter
which local solution the run converges to, not just how fast it gets
there. Neither is set by anything automatic except the third rung of
the [local-infeasibility ladder](troubleshooting.md), which only runs
after a solve has already failed.

They exist because of a measurement. Over a 244-problem corpus taken
from the KRONOS benchmark set (Ahmed & Hasan 2026 — see
[Acknowledgments](acknowledgments.md#starting-point-conditioning-kronos)),
fifteen models ended `Infeasible_Problem_Detected` or
`Invalid_Number_Detected` from their bundled starting point. Ten of
those are models an independent solver proves feasible to 2.4e-7 or
better, so the verdict was wrong. What recovered them:

| Remedy | Recovered (of 15) |
|---|---|
| default | 0 |
| `start_with_resto=yes` | 0 |
| `expect_infeasible_problem=yes` (a no-op then; refused since #981) | 0 |
| `mu_strategy=adaptive` | 4 |
| Adam warm-up | 3 |
| **one displaced start** | **13** |
| restoration + displaced start | 14 |

That ordering is the diagnosis. The iterate does not need to be
*better*, it needs to be *non-degenerate*. The common failure is a
start at which the constraint Jacobian is rank-deficient — a squared
slack sitting at zero, or an origin start on a homogeneous quadratic —
where LICQ fails and the filter line search has no descent direction
to find, whatever else it is given. Restoration does not help because
restoration inherits the same degenerate point.

### Jitter (`start_point_perturbation`)

```sh
pounce model.nl start_point_perturbation=1e-2
pounce model.nl start_point_perturbation=1e-2 start_point_perturbation_seed=7
```

Each variable is displaced by `scale * (1 + |x_i|) * u_i`, with `u_i`
drawn uniformly from `[-1, 1)`, then clipped back inside any bound it
has. The `(1 + |x_i|)` factor is the part that matters: a purely
relative perturbation is identically zero at `x_i = 0`, and a start at
the origin is the single most common degenerate start in the corpus
above.

Non-finite entries are repaired *first* — replaced by the midpoint of a
two-sided box, one unit inside a one-sided bound, or zero if the
variable is free — because NaN plus noise is NaN, and without that step
the displacement reproduces the original `Invalid_Number_Detected`
exactly.

The draw is SplitMix64 seeded by `start_point_perturbation_seed` and
nothing else — no clock, no address, no thread identity — so the same
seed and the same incoming point give the same displaced point on every
platform and every run. Vary the seed to drive a multistart by hand.

Only `x` is displaced; a warm-started `z` or `lambda` is passed through
untouched. Read that as a caveat rather than a safeguard: under
`warm_start_init_point=yes` the displacement does pair a moved primal
point with multipliers certified at the old one. There is no meaningful
"same displacement" for a dual, so moving them alongside is not on
offer, and declining to displace at all would switch the third retry rung
off for exactly the warm-started runs. That is the wrong trade — the rung
only ever runs *after* a solve has already failed, so the stale duals are
being weighed against a verdict, not against a solution. The barrier's
first iteration re-derives `z` from the bounds in any case; `lambda` is
what actually carries over.

The conditioned point is computed once per incoming start and cached, so
a warm start does not pay for the Adam warm-up twice.

### Adam warm-up (`start_point_conditioner=adam`)

```sh
pounce model.nl start_point_conditioner=adam
pounce model.nl start_point_conditioner=adam \
       adam_warmup_iters=500 adam_warmup_penalty=1.0
```

Runs Adam on the penalised merit

```
f(x) + rho * || violation(x) ||^2
```

and starts the barrier solve from where it lands. This is stage 0 of
the KRONOS algorithm, generalised from that paper's equality-only
`rho*||h(x)||^2` so it applies to an arbitrary NLP rather than only to
a squared-slack reformulation: the violation of a row is its distance
outside `[g_l, g_u]` and zero inside, which reduces to `g - b` on an
equality row. Iterates are clipped into the variable bounds at every
step, so the warm-up never hands back a point outside the box.

It changes only *where* the solve starts. No algorithm, no derivative
and no option below it moves, so the barrier solve that follows is
exactly the solve POUNCE would have run had the conditioned point been
passed in by hand.

| Option | Default | Meaning |
|---|---|---|
| `adam_warmup_iters` | 200 | Iteration budget |
| `adam_warmup_learning_rate` | 5e-2 | Step size |
| `adam_warmup_penalty` | 10.0 | `rho` on the squared violation |

The defaults are KRONOS's published stage-0 values. Two properties are
worth knowing before tuning them. Adam normalises by its second-moment
estimate, so the learning rate is very nearly the per-coordinate step
length *in the model's own units* — it wants setting against the size
of the variables, not the size of the derivatives. And because the step
is size-capped near the learning rate regardless of how large the
gradient is, the budget buys roughly `iters * learning_rate` units of
travel per coordinate: about 10 units at the defaults. A start that is
1000 units from anywhere useful will not arrive in 200 iterations.

The warm-up is **guarded**: if it does not reduce the merit it restores
the original point and reports zero iterations, so enabling it can
never cost more than the function evaluations it spent.

#### Why it is not a default

It is a real preconditioner with a fat tail. Measured over 40 problems
POUNCE already solves, it broke none of them and cut the iteration
count on 22, sometimes hard:

| Model | before | after |
|---|---|---|
| `rk23` | 82 | 11 |
| `bt5` | 45 | 9 |
| `chnrosnb` | 40 | 10 |
| `hs056` | 42 | 12 |

Median 0.83x, geometric mean 0.79x — and yet the **total rose 1.62x**,
3030 to 4900 iterations, driven by two models: `palmer1c` 71 -> 1023
and `biggs6` 1906 -> 2938. Excluding those two the ratio is 0.89x.

`palmer1c` is the case to understand. A fixed, unscaled penalty against
a badly-scaled model walks the iterate somewhere that the barrier
method then has to walk back from, and it is `adam_warmup_penalty` that
is wrong there, not the idea. If the warm-up hurts a particular model,
that is the first knob to move.

A median win with a 14x tail is an option, not a default.

## No good starting point at all?

Three composable primitives cover the "generate or repair a point"
workflows from Python:

```python
# N diverse starts (the sampler behind find_minima): sobol / uniform /
# jitter / bounds midpoint. Feed them to solve_nlp_batch or race them.
starts = pounce.generate_starts(16, bounds=bounds, seed=0)

# Safeguarded sparse elastic repair of a candidate onto the constraints
# + bounds (the standalone form of least_square_init_primal). Never
# returns a point whose true nonlinear violation is worse than the one
# you gave it; pass return_report=True for the diagnostics.
x0 = pounce.project_to_feasible(problem_obj, x0, lb=lb, ub=ub, cl=cl, cu=cu)
x0, rep = pounce.project_to_feasible(problem_obj, x0, lb=lb, ub=ub,
                                     cl=cl, cu=cu, return_report=True)
# rep.violation_initial / .violation_final / .step_norm /
# .rejected_trials / .elastic_total / .termination

# Cheap tournament: a few iterations from each start, ranked; continue
# the winner at full effort with a WarmStart.
best = pounce.race_starts(fun, starts, bounds=bounds, iters=10)[0]
res = pounce.minimize(fun, best.x,
                      warm_start=pounce.WarmStart.from_info(best.x, best.info))
```

### Racing starts: the successive-halving ladder

The default policy, `policy="fixed"`, spends the same budget on every
candidate from a cold start and ranks the field once at the end. That
keeps most of multistart's cost — the candidate that was hopeless after
two iterations is still charged for ten — and throws away the solver
state between rounds. pounce#610 adds an opt-in alternative,
`policy="halving"`, an adaptive **successive-halving ladder**:

1. every candidate runs for a small budget;
2. the field is ranked on five signals (below);
3. the weakest fraction is eliminated;
4. the survivors are **resumed from their held solver state** with a
   budget `eta` times larger, and the ladder repeats.

The winner ends up with about the effort `iters` would have given it
under the fixed policy; what changes is what the losers cost.

**It is opt-in, and the reason is measured — read this before using
it.** The ladder's early rungs rank the field on a handful of
iterations. On a strongly multimodal model that ranking carries almost
no information about which basin ends lowest, so rung 0 discards the
eventual winner. On 2-D Ackley from 27 Sobol starts with `iters=40` —
so rung 0 is four iterations and cuts 27 candidates to 9 — the start
that reaches the global minimum at full effort is cut at rung 0 in
every seed tried, ranked 19th, 13th and 24th of 27. The fixed policy
returns 4e-16 on all three seeds; the ladder returns 3.57, 5.38 and
3.57. Across an independent five-model set the ladder was 30% cheaper
overall and returned a worse answer in 13 of 45 configurations, and the
gap *widened* with more starts, because a larger field is culled harder
on the same weak signal.

Nor is that a tuning accident. `explore` does not help — it retains the
candidate *farthest* from those kept, which is not the winner — and the
only setting that recovered the answer, `min_rung_iters=20` (half the
total budget, i.e. a single cut), cost slightly more than the fixed
policy on both models. On a genuinely multimodal problem the ladder's
saving *is* the quality loss.

Reach for `policy="halving"` when a solver iteration is expensive and
the basins are few or well separated, and check the answer against the
default before relying on it. `python/tests/test_starts_racing.py::`
`test_the_ladder_can_cut_the_winner_at_rung_zero` pins the failure
mode, so if the rung-0 ranking ever becomes informative on that model
the test fails and the default is worth revisiting.

```python
best, race = pounce.race_starts(fun, starts, jac=jac, bounds=bounds,
                                constraints=cons, iters=20,
                                policy="halving", return_report=True)
print(race.report())
# race: policy=halving eta=3 candidates=16 rungs=2
#   rung 0: budget=37 evals entrants=16 -> survivors=7 spent=530 evals / 112 iters (0 resumed, 16 started)
#       - #10: duplicate of candidate 6 (scaled distance 0.000606 <= 0.001)
#       - #14: below halving cut (rank 7 of 15, keep 6)
#       - #5: below halving cut (rank 8 of 15, keep 6)
#       ... seven more
#   rung 1: budget=111 evals entrants=7 -> survivors=7 spent=272 evals / 61 iters (7 resumed, 0 started)
#   total 834 evals / 173 iters, 7 resumes
```

(HS71, 16 Sobol starts, `iters=20` — the `hs71` row of the benchmark
table below. The fixed policy spends 259 iterations on the same field.)

`RaceReport` carries the per-rung resource spend and a reason for every
candidate's exit; `RaceCandidate` carries each one's evaluations,
iterations, resumes, restoration calls and final residuals.

**What "resumed" means, precisely.** POUNCE has no API for suspending an
IPM mid-iteration and re-entering the same algorithm object — every
`Solver.solve` builds its application afresh. What a pause carries is the
whole interior-point *iterate*: the primal point, the constraint
multipliers, both bound-multiplier blocks, and the barrier parameter μ,
replayed through the warm-start machinery above so that pounce#606's
recentering measures the point it is actually handed. That is materially
not a cold restart. Measured on the `rastrigin_eq` fixture in
`python/tests/test_starts_racing.py`:

| paused at | resumed (state + point) | restarted (point only) |
|---|---|---|
| 3 iterations | **32 iters** / 330 evals | 43 iters / 368 evals |
| 5 iterations | **17 iters** / 250 evals | 43 iters / 376 evals |
| 8 iterations | **0 iters** / 80 evals | 43 iters / 372 evals |

Both arms start from the identical iterate and reach the identical
objective, start for start. The last row is the clearest: by 8
iterations every candidate has converged, the resumed solve recognises
it immediately because the carried duals and μ satisfy the convergence
check on entry, and the restarted solve — handed the same point and
nothing else — needs 5 to 8 iterations each to re-derive the same
certificate.

The size of that gap is model-dependent. On HS71 the same comparison is
a wash (98/92/77 iterations resumed against 102/87/79 restarted), which
is the regime pounce#608 warns about: a warm-started IPM often converges
in one iteration per step, and where it does a resume has nothing left
to remove. What a pause does *not* carry is the filter history and the
line-search state; that would need a `Solver.resolve()`, which does not
exist yet.

**Ranking.** Eliminations are decided on a weighted sum of five
rank-normalized signals — rank-normalized so that a violation in mol/s
and a dimensionless KKT residual can be combined without an invented
scale factor:

| signal | what it reads | default weight |
|---|---|---|
| `violation` | how infeasible the iterate is now | 3.0 |
| `feasibility_progress` | how much of its *initial* violation it has removed | 1.0 |
| `kkt` | the scaled first-order residual, in log units | 1.5 |
| `objective_progress` | objective removed per evaluation spent, damped while infeasible | 1.0 |
| `health` | restoration share, non-finite objective, failed exit | 1.0 |

Feasibility carries the most weight because an infeasible candidate's
objective is not a number about the problem being solved. Pass
`weights=` to re-balance. Diversity is protected two ways: survivors
within `cluster_tol` of each other in scaled units are collapsed to the
best of the group, and `explore` candidates from *outside* the cut are
retained anyway, chosen farthest-first from those already kept.

**Evaluations, not iterations, are the resource.** Rung 0 has no
evaluation budget — it *is* the calibration — and every later rung's
budget is a multiple of what rung 0 actually cost. Each candidate then
converts its remaining budget into an iteration cap through *its own*
measured evaluations-per-iteration, so a candidate whose iterations are
expensive (a dozen line-search trials, a restoration excursion) is
granted fewer of them for the same resource. A cumulative iteration
ceiling rising to `iters` bounds the other side.

**When not to use it.** A rung boundary costs a fresh solver application
and a re-evaluation at the seed. On a model where that fixed cost is a
large fraction of the whole solve — one variable, no constraints, a
handful of evaluations per iteration — the ladder cuts iterations but
comes out level or slightly up on evaluations. Measured over
`benchmarks/scripts/race_starts_bench.py` (six multi-basin models × three
field sizes): **17.9% fewer** user-callable evaluations overall with no
quality regression *on that set*, ranging from **43.8% fewer** on HS71
with 27 starts to **5.5% more** on the two-variable `himmelblau_disc`
with 16. Iterations fall in every one of the eighteen configurations.
That set is not a promise about your model — see the quality caveat
above. Where the ladder does not pay, the default `policy="fixed"` is
the pre-#610 policy, kept verbatim and reproducing its old answers
exactly:

```python
best = pounce.race_starts(fun, starts, bounds=bounds, iters=10)  # fixed
best = pounce.race_starts(fun, starts, bounds=bounds, iters=10,
                          policy="halving")                      # the ladder
```

`policy="halving"` runs on the NLP path only — it holds a
`pounce.Solver` session per candidate, which is what a pause suspends —
and refuses a non-`"nlp"` `solver_selection` rather than silently losing
the session it needs.

When the model has many local minima and you want *all* of them (or a
managed search rather than a tournament), the
[global search drivers](find-minima.md) (`multistart`, `mlsl`,
`deflation`, `flooding`, `tunneling`, `basinhopping`) manage
populations of starting points and warm-start bookkeeping for you,
from Python (`pounce.find_minima`) or the CLI (`--minima`).
