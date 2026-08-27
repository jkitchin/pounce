# Solver Options

POUNCE accepts options the same way upstream Ipopt does. Option names
and semantics follow Ipopt's, so an existing Ipopt options file or
`KEY=VALUE` invocation works unchanged.

## Setting options

**On the command line** — append `KEY=VALUE` pairs after the input:

```sh
pounce problem.nl tol=1e-10 max_iter=500 print_level=8
```

**From an options file** — upstream `ipopt.opt` format (one `name value`
pair per line, `#` comments):

```sh
pounce problem.nl --options-file tuned.opt
pounce problem.nl option_file_name=tuned.opt   # the Ipopt spelling; same thing
```

**From an options file nobody named.** With neither of the above, POUNCE
looks in the working directory for `pounce.opt`, then `ipopt.opt`, and
reads the first one it finds — the way `ipopt` picks up `./ipopt.opt`:

```sh
printf 'max_iter 5\n' > ipopt.opt
pounce problem.nl        # → Using option file "ipopt.opt".
```

The run says which file configured it (on the same `sb` gate as the
banner). If both default names are present, `pounce.opt` wins and the
other is named in a warning rather than left to look applied.
`--no-options-file` skips the lookup entirely — the escape hatch for a
directory holding an options file written for some other run.

Command-line `KEY=VALUE` pairs — and `$pounce_options`, AMPL's
`<solver>_options` channel — override values loaded from the options
file, never the reverse.

An options file that was *named* but cannot be read is an error, not a
shrug:

```
$ pounce problem.nl option_file_name=typo.opt
pounce: failed to load options file: options file "typo.opt" does not exist. …
```

Upstream opens a named file with a bare `ifstream` and reads nothing if
that fails, so a typo there runs at stock defaults without a word. That
silence is the one thing this path exists to remove
([#518](https://github.com/jkitchin/pounce/issues/518)): a run configured
entirely through an options file that quietly ran at defaults still
*reported success*, which invalidates any benchmark set up that way.

One upstream quirk is inherited: `option_file_name` set *inside* an
options file chains nowhere, because the file has already been chosen by
the time it is read. POUNCE warns about that rather than ignoring it.

## Commonly used options

| Option          | Meaning                                                              |
|-----------------|----------------------------------------------------------------------|
| `tol`           | Overall convergence tolerance on the KKT error.                      |
| `max_iter`      | Maximum number of outer iterations.                                  |
| `print_level`   | Console verbosity, 0 (silent) – 12 (maximum debug).                  |
| `linear_solver` | KKT linear-solver backend: `feral` (default) or `ma57` (needs a `--features ma57` build). Any other registered name is **refused**. See below. |
| `mu_strategy`   | Barrier-parameter update strategy (`monotone` / `adaptive`).         |
| `solver_selection` | Route LP/convex-QP to the specialized convex IPM. See [LP/QP Routing](lp-qp-routing.md). |
| `qp_presolve`   | Presolve on the convex LP/QP path, and on the conic (convex QCQP) path with the cone rows protected (`yes` / `no`, default `yes`). See [LP/QP Routing](lp-qp-routing.md#presolve). |
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
multiplier recalculation by least squares, least-square initialization of
*all* duals (`least_square_init_duals`), oracle-driven μ on the switch
into fixed mode (`fixed_mu_oracle`; POUNCE implements that option's
default, `average_compl`), a selectable constraint-violation norm, magic
steps, bound replacement, the L-BFGS augmented-system variants, skipping
the finalize callback, the dynamic HSL loader, and `suppress_all_output`
/ `debug_print_level`.

Two line-search knobs joined that list with
[#551](https://github.com/jkitchin/pounce/issues/551). `theta_min` is the
CG-penalty acceptor's threshold, not the filter's — the filter line
search derives its own `theta_min` from `theta_min_fact * max(1, θ₀)`, as
upstream does, and never takes it directly, so set `theta_min_fact` if
that is what you meant. `alpha_for_y_tol` configures only the
`primal-and-full` / `dual-and-full` multiplier-step rules, which POUNCE
does not have; `alpha_for_y` supports `primal` (the default),
`bound-mult`, `min`, `max` and `full`.

Four **corrector** knobs joined it too. `corrector_type` and its
safeguards `skip_corr_if_neg_curv`, `skip_corr_in_monotone_mode` and
`corrector_compl_avrg_red_fact` select and gate the corrector step Ipopt
tries *inside the line search* (`FilterLSAcceptor::TryCorrector`).
POUNCE's line search takes no corrector trial at all. The
predictor-corrector POUNCE does implement is Mehrotra's, applied to the
search-direction right-hand side: `mehrotra_algorithm=yes` is read and
honoured, and it also selects `mu_strategy=adaptive` and
`mu_oracle=probing`.

And three **sub-capabilities of features that do run**, where the
refusal message has to say so or it reads as "restoration is missing":
`expect_infeasible_problem_ctol` / `_ytol` steer
`IpBacktrackingLineSearch`'s `count_successive_shortened_steps_`
machinery, which POUNCE does not have — the restoration phase itself
runs, and `expect_infeasible_problem` and
`required_infeasibility_reduction` are read;
`resto_failure_feasibility_threshold` asks for a threshold below which a
stopped restoration is reclassified as a failure, and POUNCE has no such
reclassification (`max_resto_iter`, below, is what bounds a restoration);
and `limited_memory_special_for_resto=yes` asks for the special
quasi-Newton update Ipopt dropped in Nov 2010 — L-BFGS runs in the
restoration sub-solve with the regular update, which is what upstream's
own default `no` asks for.

The same rule applies one level down, to a single *value* of an option
that otherwise works. `bound_mult_init_method` is read and honoured, but
only its default `constant` is implemented; `mu-based` parses (an
`ipopt.opt` written for Ipopt still loads) and is then refused, because
serving it as `constant` would run a different initialization than the
one you asked for under the name you asked for.

And it applies per *entry point*, for an option whose feature exists but
sits somewhere this caller cannot reach. `option_file_name` is refused by
a library caller that reads no options file, and the convex `qp_*` knobs
(see [LP/QP Routing](lp-qp-routing.md#tuning-the-convex-ipm)) are refused
by one that cannot route a model to the convex engines. Both are
registered centrally, so they parse everywhere and the message can say
which surface honours them — rather than the caller getting `Unknown
option` for a name that exists.

`option_file_name` was on that list until
[#518](https://github.com/jkitchin/pounce/issues/518) implemented it;
refusing an option is the cheap half of "implement it or fail loudly",
and an entry leaves this table by getting the other half.

Two deliberate exceptions:

* **Setting an option to its registered default is allowed.** A generated
  `ipopt.opt` spells out defaults, and `dependency_detector=none` asks for
  nothing. Only a value that differs from the default is a request POUNCE
  cannot honour.
* **Caching hints are checked, not trusted.** `grad_f_constant`,
  `hessian_constant`, `jac_c_constant` and `jac_d_constant` tell the
  solver a derivative does not change between iterations. Ipopt takes
  such a hint on faith and silently returns a wrong answer if it is
  false. POUNCE asks the model first, and there are three cases:

  * **POUNCE proves the derivative constant** — from an `.nl` model's own
    algebra — and reuses it across iterations *whether or not you set the
    option*. Setting it is harmless and unnecessary.
  * **POUNCE proves the derivative is not constant** and you set the
    option anyway: the option is **ignored, with a warning**. A QCQP's
    `∇²L = σQ₀ + Σᵢλᵢ Qᵢ` genuinely varies with the multipliers, so
    `hessian_constant=yes` there is not a hint but a false statement, and
    honouring it would trade a correct answer for a fast wrong one.
  * **POUNCE cannot tell** — every callback front end (the C interface,
    the Python `Problem` callbacks, both GAMS links) hands POUNCE numbers
    rather than algebra — and your assertion is **honoured on trust**,
    exactly as upstream. "Unproved" is not "disproved"; overriding you
    here would be its own silent wrong answer.

  `POUNCE_DBG_CONSTDERIV=1` prints which of the three fired for each of
  the four options.

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
backend cannot be selected, so setting one alongside options POUNCE does
read **warns and solves** — it does not fail the run:

```
$ pounce model.nl ma97_order=metis tol=1e-8
pounce: warning: `ma97_order` configures the HSL MA97 sparse symmetric linear
solver, which pounce does not implement, so it is ignored — as is every other
`ma97_*` option. pounce factors the KKT system with `feral` (pure Rust, the
default) or MA57 (`linear_solver=ma57`, in a `--features ma57` build); no
setting written for another backend transfers to either. The name is registered
so an `ipopt.opt` written for Ipopt still parses unchanged — which is why this
is a warning and not an error: the solve runs, and its result is unaffected.
```

A warning rather than a refusal, unlike the options above, because a
portable `ipopt.opt` routinely carries settings for several backends at
once so that one file runs everywhere. Refusing would fail that file over
knobs the run never touches — for a user who is not using MA97 and never
asked POUNCE to. One line is printed per backend family, listing the
options it saw, and only for a value that differs from the registered
default: a file that spells out defaults asks for nothing and gets
nothing said about it. `pardisolib` warns with the `pardiso_*` family;
`hsllib` is still refused, because POUNCE *has* an HSL backend (MA57) and
the refusal points you at `--features ma57` rather than leaving you to
believe a library was loaded.

#### …unless they are all you set

That reasoning assumes the file has other business here. If the backend
knobs are *everything* the run sets, nothing in the file survives, and
warning-then-solving would answer "tune the linear solver" by tuning
nothing and reporting success. That case is **refused**:

```
$ pounce model.nl ma97_order=metis
pounce: error: every option this run sets configures a linear-solver backend
pounce does not implement, so there is nothing left for it to act on. […] Set
`linear_solver=feral` (or `ma57`) if the defaults are what you want.
```

The rule in full:

| what the run sets | result |
|---|---|
| backend knobs only | **error**, exit 2 |
| backend knobs + any option POUNCE reads | warning, solve continues |
| backend knobs at their registered defaults | silent, solve continues |
| no backend knobs | silent, solve continues |

Two details worth knowing. The second row counts an option's *presence*,
not whether you changed it — writing `tol` at its default is still a
statement about this solve, and it is enough to put you back on the
warning path. And `option_file_name` does not count as content: it says
where the options came from, not what to solve, so pointing at a
backend-only `ipopt.opt` is refused exactly as passing the same knobs on
the command line would be.

A file that *selects* the backend it tunes never reaches this: `linear_solver=ma97`
is refused on its own, as [above](#choosing-a-linear-solver).

### Inertia-free curvature test (`neg_curv_test_tol`)

By default every KKT factorization is checked for the right inertia — as
many negative eigenvalues as there are constraints — and the primal
regularization δ_x is escalated until it has it. The inertia-free
alternative of Zavala & Chiang (2014) factors without that check and
instead asks whether the direction the system produced actually curves
upward:

```
dxᵀ W dx + dxᵀ Σ_x dx + dsᵀ Σ_s ds [+ δ_x‖dx‖² + δ_s‖ds‖²]
    ≥ neg_curv_test_tol · (‖dx‖² + ‖ds‖²)
```

| Option              | Default | Meaning                                                                                     |
|---------------------|---------|---------------------------------------------------------------------------------------------|
| `neg_curv_test_tol` | `0.0`   | `0` keeps the inertia check. Positive is the test's α_n: the factorization is accepted only if the direction clears the bound above, and otherwise δ_x is escalated exactly as a wrong inertia would. Upstream recommends `1e-12`–`1e-11`. |
| `neg_curv_test_reg` | `yes`   | Whether the bracketed primal-regularization term counts toward the curvature. `no` is the original Ipopt form that ignores it. Only read when `neg_curv_test_tol > 0`. |

**This is a heuristic, and turning it on is not free — it can change
the answer, not just the path to it.** Measured over POUNCE's fixture
corpus at the recommended `1e-11` (`scripts/sweep-fixtures.sh`, both
legs), 11 of 59 models move:

| model | default | `neg_curv_test_tol=1e-11` |
|---|---|---|
| `csfi2` | `Solved_To_Acceptable_Level`, 35 it | **`Solve_Succeeded`, 27 it** |
| `unbounded_cubic` | `Diverging_Iterates`, 290 it | **`Diverging_Iterates`, 61 it** |
| `cresc4` | 81 it | 90 it |
| `infeasible_equalities` | `Infeasible_Problem_Detected`, 28 it | same, 37 it |
| `unbounded_exp` | `Error_In_Step_Computation`, 27 it | same, 32 it |
| `eigena2` | 26 it | **421 it** |
| `eigenb2` | 67 it | **960 it** |
| `autocorr_bern55-06` | `Solve_Succeeded`, 72 it, obj `-2304.000028` | **1042 it, obj `-2288.000022`** |
| `pooling_rt2stp` | `Solve_Succeeded`, 298 it, obj `-3273.954992` | **`Solved_To_Acceptable_Level`, 537 it, obj `-3085.16078`** |
| `deb7` | `Solve_Succeeded`, 154 it | **`Error_In_Step_Computation`, 183 it** |
| `eigenb2` (L-BFGS leg) | `Solve_Succeeded`, 56 it | **`Error_In_Step_Computation`, 76 it** |

The last four rows are the reason to read this before switching it on.
`deb7` and `eigenb2`-under-L-BFGS stop converging at all;
`autocorr_bern55-06` and `pooling_rt2stp` still report success but land
on a **worse objective** — a tolerance-legal wrong answer, which is the
failure mode that is invisible to a suite asserting status and
objective-to-a-tolerance. Accepting a factorization whose inertia is
wrong is exactly the kind of change that produces it.

It is off by default (`neg_curv_test_tol=0` keeps the inertia check),
and nothing above happens to a solve that leaves it alone. If you turn
it on, measure your own model.

### Escaping a stationary point that is not a minimum (`neg_curv_escapes`)

The convergence test is a **first-order** test. On a nonconvex model that
is strictly weaker than "local minimum": at a point where the reduced
Hessian on `null(A)` is negative definite every KKT residual is zero, so
the test has nothing to object to, and the point reported as
`Solve_Succeeded` can be a constrained **maximum**.

The CLI fixture `nonconvex_qp.nl` is that case in three lines:

```
min x₀·x₁   s.t.   x₀ + x₁ = 2,   0 ≤ x ≤ 4
```

On the feasible segment the objective is `f(x₀) = x₀(2 − x₀)`, which is
*concave* — maximized at `(1, 1)` with `f = 1`, minimized at the endpoints
`(0, 2)` and `(2, 0)` with `f = 0`. From the bound-pushed start
`(0.01, 0.01)` the first Newton step lands exactly on `(1, 1)`, and every
iteration after it takes a step of size `1e-14`.

Inertia correction does not prevent this and never could. It *engages* —
the iteration log shows `lg(rg)` from the second iteration on — but `δ_x I`
is symmetric, the model and the iterate are symmetric under `x₀ ↔ x₁`, and
a symmetric correction applied to a zero gradient gives a zero step however
indefinite the reduced Hessian is. Regularization makes the **step**
well-posed; nothing else in the algorithm asks whether the point it
converged to is a minimum.

| Option | Default | Meaning |
|---|---|---|
| `neg_curv_escapes` | `1` | How many times a certified stationary point with an indefinite reduced Hessian may be *left* along a direction of negative curvature instead of reported. `0` reports the first-order certificate whatever its curvature. |

With this on, a point about to be certified is first tested for
second-order necessity: one extra factorization of the augmented system
with the inertia check on and **no** perturbation, whose correct inertia is
exactly the statement that `W + Σ` is positive definite on `null(A)`. A
point that passes costs that one factorization and nothing else. A point
that fails gets `δ_x` escalated until the inertia is right, and a few
inverse-iteration back-solves against that factor recover the
most-negative-curvature direction — which is then *measured*, not trusted.
The solve steps along it (capped by the fraction-to-the-boundary rule,
backtracked against the second-order decrease model, refused outright if it
raises the constraint violation past `constr_viol_tol`) and continues.

**It cannot return a worse answer than leaving it off would have.** The
stationary point is snapshotted before the step and is restored and
reported unless the continuation comes back with a certificate of its own
at a better point — the same floor-and-deadline accounting as
`resto_decline_deferrals`, and the continuation is cut after 30 iterations
either way. Raising the option above `1` does not weaken that (gh #805):
the floor holds the **best** certificate the escapes have left, not the
most recent one, so every bet is placed against the same baseline the
first one was — the point a `neg_curv_escapes = 0` build reports. Each
escape does buy its continuation its own 30 iterations, so the *cost*
scales with the option and the guarantee does not. On `nonconvex_qp.nl` — and on `nonconvex_qp_ineq.nl`, the same
model with its row relaxed to `x₀ + x₁ ≥ 2` — it turns `Solve_Succeeded` at
`obj = 1` into `Solve_Succeeded` at `obj = 0`; across the rest of the
fixture corpus (`scripts/sweep-fixtures.sh`, both legs, 152 fixture-legs) it
moves nothing.

Two limits are worth knowing:

* **It is still a local method.** An escape finds a point that is
  second-order suspect and leaves it; it does not certify global
  optimality, and a stationary point whose reduced Hessian is positive
  definite is never touched.
* **Under `hessian_approximation=limited-memory` it does nothing.** The
  curvature it reads is `B`, and BFGS maintains `B` positive definite by
  construction, so the inertia test passes at `δ_x = 0` and the escape
  declines. The L-BFGS leg of the fixture sweep still reports `obj = 1` on
  both nonconvex-QP fixtures.

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

Note what `honor_original_bounds` does *not* do: it moves the reported
point, not the solve. The iterate still stopped where the relaxed bounds
put it, and any question of the form "is this constraint active" is still
being asked about a point `~1e-8` shy of the bound. Projection cannot
recover that, because the projection has no way to tell "pinned to the
bound" from "genuinely `1e-8` inside it". [Crossover](crossover.md)
answers that question instead: it re-solves against the bounds you
declared, so the returned point sits *on* them and the active set is
established rather than inferred.

## Large constraint values and `primal_noise_floor_kappa`

On a model whose constraint values run to `~1e7` and beyond, a converged
solve could exit `Search_Direction_Becomes_Too_Small` **while holding the
correct optimum**. The cause is arithmetic, not the model.

The KKT error the convergence test compares against `tol` is

```
max( ‖∇L‖∞ / s_d ,  max(‖c‖∞, ‖d − s‖∞) ,  ‖compl‖∞ / s_c )
```

The dual and complementarity terms are normalised; the primal one — like
upstream Ipopt's — is a bare absolute residual. But `c_i = g_i(x) − b_i`
and `d_i − s_i` are each a *difference of quantities the row's own size*,
so they are quantised in units of `eps · |b_i|`. At `|b| ~ 1e8` the
smallest **nonzero** value the primal term can take is one ulp,
`1.5e-8` — already larger than the default `tol = 1e-8`. Asking for
`nlp_err <= tol` there is asking the residual to land on a bitwise-exact
`0` rather than on one ulp, which is arithmetic luck rather than a
property of the iterate.

POUNCE therefore judges the primal term in the **strict** test against
each row's own floating-point resolution: a row's residual counts only
where it exceeds `max(placement floor, kappa · eps · |row magnitude|)`,
with `kappa = primal_noise_floor_kappa` (default `64`). Verdicts are flat
across kappa from 8 to 1024 on the measured set.

Three things bound what this can do:

* **Only the strict test reads it.** `constr_viol` is still checked
  against `constr_viol_tol` (default `1e-4`) on the full, unfloored
  residual, so nothing the floor forgives can exceed the feasibility
  tolerance you set — however large your data grows.
* **The acceptable-level band keeps the raw error.** It sits two decades
  above `tol`, clear of any realistic quantum.
* **It cannot rescue an infeasible model.** On a model with no feasible
  point the filter and restoration phase reach a verdict on their own
  criteria; the floor only ever participates at a point the rest of the
  algorithm already believes is converged.

Set `primal_noise_floor_kappa = 0` to switch the floor off and restore
upstream Ipopt's bare-absolute primal term exactly.

When the floor changes the reported picture, the end-of-run summary says
so — a large-`|b|` solve prints the tested value under the raw one:

```
Overall NLP error.......:   2.3841857910156250e-07    2.3841857910156250e-07
  ...above the per-row floating-point noise floor:   0.0000000000000000e+00
```

Solves where the two agree — every model whose data is `O(1)` — print the
usual block unchanged.

One case this does **not** paper over: tightening `constr_viol_tol` below
a row's own ulp (say `1e-8` on data at `1e8`) still will not certify. That
is the tolerance gate doing what you asked — the residual you requested is
not representable at that scale.

## `Solved_To_Acceptable_Level` and `acceptable_progress_kappa`

`Solved_To_Acceptable_Level` is the fallback verdict for a solve that
cannot reach `tol`: after `acceptable_iter` (default `15`) consecutive
iterates with an NLP error under `acceptable_tol` (default `1e-6`), the
solver stops and hands back the point it has. That criterion is a count of
iterates inside a band, and on its own it asks only *is the error small* —
never *has anything stopped moving*.

Those come apart. An interior-point iterate can be near-stationary for the
current **barrier subproblem** — a much weaker statement than near-KKT for
the NLP — for fifteen iterations running while the solve is still
descending. Two measured cases: the `kissing` model stopped with objective
`1.00000108` where continuing reaches `0.84544259` and a strict
certificate, 18% lower; `NARX_CFy` stopped with both residuals near `1e-7`
where sixty more iterations collapse them by five orders.

POUNCE therefore also requires the streak to have **flattened**. Across the
`acceptable_iter` iterates that made it up:

* the spread (`max − min`) of the NLP error must be within
  `acceptable_progress_kappa · acceptable_tol`; and
* the spread of the objective within the same fraction of
  `acceptable_tol · max(1, |f|)`.

`acceptable_progress_kappa` defaults to `0.1`, so at default tolerances
both quantities must have stayed inside a tenth of the acceptable band over
the whole streak.

It is a **spread**, not a trend, and either signal alone is enough to keep
solving. Both choices are deliberate: `kissing`'s error was an order of
magnitude *worse* at the iterate it stopped on than at one it had already
reached inside the same streak — it was wandering across the band, not
converging inside it — while its objective was flat to all eight printed
figures over the same iterates.

Three things bound what this can do:

* **It cannot lose a verdict.** The refused termination is *recorded*, and
  a run that fails to do better ends at exactly that iterate under exactly
  that status. A misfire costs iterations, never the answer — you will not
  see `Maximum_Iterations_Exceeded` where the count alone would have said
  `Solved_To_Acceptable_Level`.
* **It never looks at a solve that converges.** A solve that reaches `tol`
  never completes an acceptable-level streak, so nothing here runs.
* **A genuine stall flattens.** When the iterate, the objective and the
  error are all pinned — the case the acceptable-level exit exists for —
  the window is flat and termination happens as before.

Set `acceptable_progress_kappa = 0` to switch the progress test off and
restore upstream Ipopt's bare consecutive-count criterion. Widening
`acceptable_tol` widens the flat bar with it, so asking for a looser band
still gets you the early exit.
## Big models that start feasible and the `theta_max` ceiling

The filter has a hard ceiling. Any trial iterate whose constraint
violation `θ` exceeds

```
theta_max = theta_max_fact · max(1, θ₀)
```

is rejected outright, before any of the filter's usual tests run. It is a
global-convergence safeguard: it keeps the line search from wandering
arbitrarily far from feasibility.

The trouble is the `1`. POUNCE's `θ` is a **1-norm over constraint rows** —
`‖c‖₁ + ‖d − s‖₁`, a *sum* of `m` residuals — so a ceiling of `T` really
says "a mean per-row violation of `T/m`", and that allowance shrinks as the
model grows. And on a problem started at a **feasible** point, `θ₀ = 0`,
the `max` collapses and the ceiling is the bare constant `theta_max_fact`
however large the model is.

`robot_a` is the measured case: 52 013 constraint rows, a feasible start,
so `theta_max` locked at `1e4` — a mean per-row allowance of `0.19` — while
the route to the optimum passes through `θ ≈ 9.4e7`. Every step toward the
solution was refused at the gate, and the solve ground to its iteration
limit at objective `8.173304` instead of the true `1.0431952`.

POUNCE's answer is `theta_max_adaptive_trigger` (default `3`), described
below. `theta_max_row_scale_kappa` (default `0`, off) is an earlier,
static attempt at the same problem, kept because it is occasionally the
more direct lever; it floors the reference at the row count instead:

```
theta_max = theta_max_fact · max(θ₀, theta_max_row_scale_kappa · rows, 1)
```

so the ceiling means a mean per-row violation of `theta_max_fact`
regardless of `m`. Measured under defaults otherwise, against Ipopt 3.14
on the same machine:

| model | POUNCE default | POUNCE `kappa = 1` | Ipopt (default) |
|---|---|---|---|
| `robot_a` | `Maximum_CpuTime_Exceeded`, 8.173304 | **Optimal, 1.0431952, 112 it** | `Maximum_Iterations_Exceeded`, 8.173304 |
| `robot_b` | `Maximum_CpuTime_Exceeded`, 15.484684 | **Optimal, 2.3330990, 252 it** | `Maximum_Iterations_Exceeded`, 15.484684 |
| `robot_c` | `Maximum_CpuTime_Exceeded`, 29.039906 | **Optimal, 1.4059756, 109 it** | `Maximum_Iterations_Exceeded`, 29.039906 |

Ipopt has the same defect and no correction for it; all three solve under
`theta_max_fact = 1e8` set by hand, which is the blunt version of the same
move.

### The adaptive rule: `theta_max_adaptive_trigger`

**On by default.** Rather than guessing from problem size whether a model
needs headroom, POUNCE measures whether the ceiling is what is refusing
the line search, and raises it only then.

A trial refused because `θ_trial > theta_max` takes a distinct early exit,
before the filter and Armijo tests run at all. So the acceptor can count
those refusals and compare them against the number of trials attempted.
When **every** trial of a line search was refused at the gate, for
`theta_max_adaptive_trigger` **consecutive** line searches, the ceiling is
demonstrably the binding constraint — not the filter — and it is
multiplied by `theta_max_adaptive_factor` (default `100`), at most
`theta_max_adaptive_max_raises` times per solve (default `4`).

| option | default | meaning |
|---|---|---|
| `theta_max_adaptive_trigger` | `3` | consecutive fully gate-refused line searches before a raise; `0` disables |
| `theta_max_adaptive_factor` | `100` | geometric factor per raise |
| `theta_max_adaptive_max_raises` | `4` | cap on raises per solve |

Three properties follow, and they are what the static floor could not
offer:

* **A converging model cannot trip it.** Converging means trials are
  getting past the gate; a model accepting steps never accumulates the
  streak. `brainpc1/3/5/7` — the family the static floor damaged at every
  `kappa` — are untouched *by construction*, not by a lucky constant.
* **A blocked model trips it immediately.** `robot_a` is refused at the
  gate from its first line search onward.
* **The ceiling stays finite.** Wächter–Biegler's global-convergence
  argument (Thm. 2) needs `theta_max` finite, not fixed. A bounded number
  of bounded raises keeps it finite, so a solve cannot ratchet the
  safeguard away one line search at a time.

Requiring a streak rather than a single line search is deliberate: one
Newton direction that overshoots into a huge `θ` can legitimately have
all its trials refused, and backtracking is the right response to that.
Only a model that cannot get past the gate *repeatedly* is one whose route
needs the headroom.

Measured, defaults otherwise:

| model | rule off (`trigger = 0`) | rule on (default) |
|---|---|---|
| `robot_a` | `Maximum_Iterations_Exceeded`, 14.23 | **Optimal, 1.0432009, 190 it** |
| `robot_b` | max time, 15.484684 | **Optimal, 2.3330990, 269 it** |
| `robot_c` | max time, 29.039906 | **Optimal, 1.4059756, 222 it** |
| `brainpc1` | Optimal, 64 it | Optimal, 64 it — *identical* |
| `brainpc3` | Optimal, 43 it | Optimal, 43 it — *identical* |
| `brainpc5` | Optimal, 982 it | Optimal, 982 it — *identical* |
| `brainpc7` | Optimal, 43 it | Optimal, 43 it — *identical* |
| `bt4` | Optimal, 9 it, −3.7047681836394486 | *identical* |

Across the whole Vanderbei corpus (733 problems) the rule changes four
outcomes: `britgas` goes from its iteration limit to `Optimal` in 16
iterations, `catenary` solves to the same objective in 50 iterations
instead of 56, and `coshfun` and `brainpc0` fail either way — `coshfun`
now reporting diverging iterates, which is what Ipopt 3.14 also does on
it. Net `Optimal` count 702 → 703.

Note `brainpc0` *does* trip the rule while `brainpc1/3/5/7` do not,
despite identical row counts. That is precisely the distinction a
size-based floor cannot draw.

The restoration sub-IPM always runs with the rule disabled. Upstream
already corrects the resto phase's instance of this degeneracy by
hard-coding `resto.theta_max_fact = 1e8` (`IpRestoMinC_1Nrm.cpp:91`), so a
rule that ratchets further would be compounding a correction already made.

Set `theta_max_adaptive_trigger = 0` to restore upstream Ipopt's fixed
ceiling exactly.

### When to reach for the static floor instead

Symptoms, all three together:

* a **large** number of constraint rows (thousands upward);
* a **feasible or near-feasible starting point** — the iteration log's
  first `inf_pr` is `0` or very small;
* the solve stalls with `inf_pr` flat and `alpha_pr` tiny, and raising
  `max_iter` does not help.

The quick confirmation is to set `theta_max_fact = 1e8` by hand. If that
unsticks the model, `theta_max_row_scale_kappa = 1` is the principled
version of it — it scales the ceiling to the model rather than to a
constant you picked.

### Why the static floor is off by default

Because raising the ceiling unconditionally is not free. It relaxes a global-convergence
safeguard, and a model that was **not** being blocked by it can wander
instead. On the Vanderbei corpus, `brainpc1/3/5/7` (`m = 6900`,
`θ₀ = 1e-2`) all regress — `brainpc1` from `Optimal` in 64 iterations to
divergent, objective `3.7e3` against the correct `4.4e-04`.

A scan over `kappa` showed why this cannot be tuned away. The damage is a
**step function, not a gradient**:

| kappa | `robot_a` | `brainpc1` | `brainpc3` | `brainpc7` |
|---|---|---|---|---|
| **0** (default) | max time, 616 it | **Optimal, 64 it** | **Optimal, 43 it** | **Optimal, 43 it** |
| 0.01 | Optimal, 287 it | max time, 3.8e8 | Acceptable, 149 it | Acceptable, 552 it |
| 0.05 | Optimal, 153 it | max time | Acceptable, 149 it | Acceptable, 552 it |
| 0.2 | Optimal, 127 it | max time | Acceptable, 149 it | Acceptable, 552 it |
| 1.0 | Optimal, 112 it | max time | Acceptable, 149 it | Acceptable, 552 it |

`brainpc3` and `brainpc7` land on the *identical* worse answer at every
nonzero `kappa`, even `0.01` — where the ceiling moves only from `1e4` to
`6.9e5`. The instant it rises at all, they break. `robot_a` meanwhile
improves monotonically all the way to `kappa = 1`. There is no separating
value.

That is a verdict on the *design*, not on the tuning: the real question is
whether a model's route to the optimum **needs** the extra headroom, and
the row count does not answer it. A static floor cannot know — which is
why the adaptive rule above, which asks the question directly, is the
default and this one is not.

What still bounds the option when you do turn it on:

* **It only ever raises the reference.** A model whose own `θ₀` already
  exceeds `kappa · rows` gets exactly upstream's ceiling.
* **It reduces to upstream on a single-row problem**, where the floor is
  `max(kappa · 1, 1) = 1` — upstream's constant.
* **`theta_max` is still finite**, and still fixed for the whole solve
  after its first line search. This rescales the safeguard; it does not
  remove it.
* **The restoration sub-IPM is untouched**, at any `kappa`. Upstream
  already fixes its own instance of this by hard-coding
  `resto.theta_max_fact = 1e8` (`IpRestoMinC_1Nrm.cpp:91`) — the resto NLP
  is also initialised feasible, so it hit the same degeneracy. Stacking
  the row floor on top would push that inner ceiling to `1e8 · m`, i.e.
  remove it, so the sub-IPM always runs with `kappa = 0`.

## Large gradients and `dual_inf_scale_kappa`

The dual side of the same story. `dual_inf_tol` (default `1.0`) is a bare
**absolute** bound on `‖∇L‖∞` — but the aggregate above **normalises**
that quantity, dividing it by `s_d`, which grows with the mean magnitude
of the multipliers. On a model whose gradients live at `1e10` the two
gates are judging one number by standards ten orders apart.

Vanderbei's `orthrds2` is the reported case: `s_d ≈ 1.6e10` with
`‖∇L‖∞ = 89.7`, so the aggregate's dual term is `5.6e-09` — comfortably
inside the default `tol = 1e-8`, i.e. stationary to nine digits relative
to the size of the gradients involved — while the component gate refused
it against `1.0`. The solve exited `Solved_To_Acceptable_Level` holding
the answer, and `dual_inf_tol=1e3` alone turned it into
`Optimal Solution Found` at the same objective.

The simplest statement of the defect: multiply an objective by a positive
constant. Same feasible set, same solution, same active set, same Newton
step — and every multiplier, `s_d` and `‖∇L‖∞` scale with it, so a large
enough constant costs the certificate.

The **strict** test therefore judges the unscaled dual infeasibility
against

```
max( dual_inf_tol ,  kappa · tol · dual_scale )
```

with `kappa = dual_inf_scale_kappa` (default `1`) and `dual_scale` the
magnitude of the largest single term `∇L` is assembled from (`∇f`,
`Jᵀy`, the bound multipliers). Since `∇L` is the *sum* of those terms,
`‖∇L‖∞ / dual_scale` is the fraction of them that failed to cancel — a
scale-invariant statement of stationarity, and the thing the absolute
bound was standing in for.

What bounds it:

* **It cannot forgive non-stationarity.** A point where nothing cancelled
  has `‖∇L‖∞ ≈ dual_scale`, a ratio of `1` against a bar of `1e-8`.
  `min −exp(x) s.t. x >= 0` running away to `inf_du = 8.8e+47` is refused
  by eight orders, because its `∇f` runs away by exactly the same factor.
* **The aggregate still has to pass.** `nlp_err <= tol` is tested on the
  same iterate; this only removes the second, inconsistent standard.
* **It is inert on ordinary models.** At the defaults the floor does not
  rise above `dual_inf_tol` until `dual_scale` exceeds
  `dual_inf_tol / tol = 1e8`, so every model with `O(1)` gradients keeps
  upstream's comparison bit for bit.
* **Only the strict gate reads it.** `acceptable_dual_inf_tol` (`1e10`) is
  untouched.

Set `dual_inf_scale_kappa = 0` to switch the floor off and restore
upstream Ipopt's bare-absolute bound. That is also the setting to reach
for if you *tighten* `dual_inf_tol` and want that absolute standard
honoured unconditionally — the floor is a floor, so it can override a
tightened `dual_inf_tol` on a large-gradient model.

### `s_max` — where `s_d` and `s_c` come from

The two normalising factors above are built from the multipliers
themselves, capped by `s_max` (default `100`, upstream's):

```
s_d = max( s_max , (‖y_c‖₁+‖y_d‖₁+‖z‖₁+‖v‖₁) / (their total dimension) ) / s_max
s_c = max( s_max , (‖z‖₁+‖v‖₁) / (their dimension) ) / s_max
```

Both are exactly `1` while the multipliers average below the cap — which
is every well-scaled problem, and why the option is invisible there — and
grow as `mean / s_max` once the average passes it. Raising `s_max`
therefore delays the normalisation (the KKT error stays closer to the raw
residuals); lowering it applies the normalisation sooner and makes the
scaled error smaller for the same iterate, so the solve certifies
earlier. The scaled and unscaled numbers are both reported: `--json-output`
carries `final_kkt_error` and `final_unscaled_kkt_error`, and their ratio
is exactly what `s_max` controls.

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

## Starting-point conditioning

Three options displace the starting point before the barrier solve, and
one turns the automatic retry that uses them on or off. All are off by
default except the retry, which fires only after a solve has already
failed. Full rationale and the measurements behind the defaults:
[Conditioning the starting point](initialization.md#conditioning-the-starting-point).

| Option | Default | Meaning |
|---|---|---|
| `infeasibility_perturbed_start_retry` | `yes` | Rung 3 of the [second-opinion ladder](troubleshooting.md#the-second-opinion-ladder-what-those-extra-solves-in-your-log-are): on `Infeasible_Problem_Detected` or `Invalid_Number_Detected`, re-solve once from a displaced start. Promoted only on `Solve_Succeeded` / `Solved_To_Acceptable_Level`. |
| `start_point_perturbation` | `0.0` | Relative displacement `scale·(1 + \|x_i\|)·u_i`, `u_i` uniform on `[-1, 1)`, clipped into bounds. `0` disables. Non-finite entries are repaired to a finite in-bounds value first. |
| `start_point_perturbation_seed` | `0` | SplitMix64 seed for that displacement — no clock, no address, no thread identity, so the same seed gives the same point on every platform. |
| `start_point_conditioner` | `none` | `none`, or `adam` to run a first-order warm-up on `f(x) + ρ‖violation(x)‖²` and start from where it lands. |

`start_point_conditioner=adam` reads three more. They are ignored unless
it is set, and their defaults are KRONOS's published stage-0 values
(Ahmed & Hasan 2026, see
[Acknowledgments](acknowledgments.md#starting-point-conditioning-kronos)).

| Option | Default | Meaning |
|---|---|---|
| `adam_warmup_iters` | `200` | Iteration budget. Adam's step is size-capped near the learning rate, so this buys about `iters × learning_rate` units of travel per coordinate. |
| `adam_warmup_learning_rate` | `5e-2` | Step size. Because Adam normalizes by its second-moment estimate this is nearly the per-coordinate step *in the model's units*, so set it against the size of the variables, not the derivatives. |
| `adam_warmup_penalty` | `10.0` | ρ on the squared violation. Trades objective against feasibility during the warm-up only. The fixed, unscaled default is the likeliest cause of the measured tail on badly-scaled models — the first knob to move if the warm-up hurts. |

The warm-up is guarded: if it does not reduce the merit it hands back
the original point unchanged. It is off by default because across 40
problems POUNCE already solves it cut the median iteration count to
0.83× while raising the **total** 1.62×, on the strength of two
outliers (`palmer1c` 71 → 1023). A median win with a 14× tail is an
option, not a default.

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
| `tau_min`                               | `0.99`             | Floor on the fraction-to-the-boundary parameter τ = max(`tau_min`, 1 − μ); a step may cover at most τ of the distance to a bound. Read by both μ strategies (and by the restoration sub-solve). |
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
| `adaptive_mu_max_free_returns`          | `-1`    | Cap on returns to free-μ mode after entering monotone mode; `-1` is unlimited (upstream). POUNCE extension (#749). |
| `adaptive_mu_budget_pin_fraction`       | `0.75`  | Fraction of an explicitly set `max_cpu_time`/`max_wall_time` after which the strategy finishes monotone; `1` disables. Inert without a time budget. POUNCE extension (#753). |

## Limited-memory Hessian (L-BFGS) initialization

Under `hessian_approximation=limited-memory` the Hessian model is
`B = σ I + V Vᵀ − U Uᵀ`. The rank-2 corrections come from the curvature
history; `σ` is the diagonal they are built on, and
`limited_memory_initialization` chooses the formula for it.

| Option | Default | Meaning |
|---|---|---|
| `limited_memory_initialization` | `scalar2` | Formula for `σ`: `scalar1` (σ = sᵀy/sᵀs), `scalar2` (σ = yᵀy/sᵀy), `scalar3` (arithmetic mean of the two), `scalar4` (geometric mean), `constant` (σ = `limited_memory_init_val`). |
| `limited_memory_init_val` | `1.0` | `σ` on the first iteration, before any curvature pair exists — and every iteration under `constant`. |
| `limited_memory_init_val_min` / `_max` | `1e-8` / `1e8` | Clamp applied to `σ` however it was computed. |

The default matches Ipopt's `scalar1`. Note that it changed in #677:
every earlier release used `scalar2` and ignored this option entirely —
it was registered but never read, so setting it had no effect and no
warning. The two differ by σ_scalar2/σ_scalar1 = (yᵀy·sᵀs)/(sᵀy)², which
is ≥ 1 by Cauchy–Schwarz and grows without bound as the curvature pair
becomes ill-conditioned, so on a badly scaled problem they are far
apart. If you are reproducing results from an older POUNCE, set
`limited_memory_initialization scalar2`.

### `recalc_y` under L-BFGS

A quasi-Newton dual step is computed from an approximate Hessian, so an
L-BFGS solve can settle a feasible primal and still fail to drive dual
infeasibility to tolerance. `recalc_y yes` re-estimates the equality and
inequality multipliers by least squares on every iteration whose
constraint violation is below `recalc_y_feas_tol` (default `1e-6`),
side-stepping the approximation. Each firing costs one extra
augmented-system solve.

Ipopt's option text says this is used by default with a quasi-Newton
Hessian. **POUNCE does not enable it by default**, because doing so
regressed 7 of 57 fixtures on the L-BFGS leg — re-estimating `y` every
iteration also overwrites Newton multipliers that were converging
perfectly well. Reach for it when dual infeasibility oscillates without
descending while the objective and the primal have already settled; that
is the shape it fixes.

`σ` cannot be observed directly, but the symptom of a badly chosen one is
recognisable: a search direction much larger than the problem's scale,
primal step sizes collapsing to `1e-3` or below, primal infeasibility
that barely moves, and dual infeasibility climbing by orders of magnitude
while the objective drifts.

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
through ten `feral_*` options. Defaults are tuned for the IPM
workload and rarely need changing; reach for these when profiling a
specific problem. Each also falls back to a matching `POUNCE_FERAL_*`
environment variable when left unset on the OptionsList (see
[Environment overrides](#environment-overrides-feral-and-debug-gates)).

| Option                       | Default | Meaning                                                                                                                                                                                  |
|------------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `feral_ordering`             | `auto`  | Fill-reducing ordering method (see table below). `auto` lets feral's adaptive dispatcher pick per-matrix; `auto_race` measures the actual symbolic outcome and keeps the best.            |
| `feral_pivtol`               | `1e-8`  | Relative Bunch-Kaufman partial-pivoting threshold `u`. Analog of `ma27_pivtol` / `ma57_pivtol`. Smaller → sparser `L`, faster, less stable; larger → more 2×2 blocks, denser, more stable. LAPACK's textbook maximum-stability value is `0.5`. |
| `feral_refine`               | `no`    | Whether FERAL runs its own iterative refinement inside every back-solve. **Off by default for the NLP solver (gh#710, reported as gh#698 observation 5), as on every direct linear solver Ipopt ships and on POUNCE's own MA57. Note this is the *option*'s default; `FeralConfig`'s own stays `yes`, for callers such as `pounce-convex`'s SOS/QP solvers that refine their own system but never call `increase_quality`.** Refinement belongs on the *unreduced* Newton system — that is `PdFullSpaceSolver`'s loop, capped at `max_refinement_steps` and accepting at `residual_ratio_max = 1e-10` — not on the condensed system a backend factorized, because the condensation destroys information as `mu -> 0` (Wachter-Biegler 3.10). Turning it on nests FERAL's loop inside that one, and FERAL's convergence target is hard-wired to `eps*sqrt(n)`; on a large ill-conditioned KKT that target is unreachable, so the inner loop runs to its cap on every back-solve chasing digits the caller discards. It was on through 0.10.0 because FERAL's `ZeroPivotAction::ForceAccept` can leave real residual against the system it factorized, and without it the gh#590 badly-scaled LP exits `RestorationFailed` — but Ipopt's answer to a factorization that cannot deliver is `IncreaseQuality` (escalate the pivot threshold and refactorize), and that rung was unimplemented in the FERAL backend. It is now, so refinement no longer has to stand in for it. On the 126028-dimension `laptime` KKT under limited-memory, one binary, three runs back to back: 68.9 s on, 18.8 s off, against MA57's 10.7 s (back-solve 54.6 s -> 8.2 s). Set `yes` to restore pre-0.11 behaviour on a problem that needs it. |
| `feral_refine_steps`         | `10`    | Maximum correction steps FERAL's inner iterative refinement may take on a single back-solve, when `feral_refine` is on. An **upper bound**, not a step count: refinement still exits early on its own convergence test, so lowering this only truncates the solves that were going to run long. `0` leaves refinement enabled but caps it at zero corrections — that still costs the residual evaluation, so use `feral_refine=no` to switch refinement off outright. Reach for a small cap (`1`) on very large, badly conditioned KKT systems where the interior-point tail spends most of its wall clock inside refinement rather than the factor (gh#710) — but check the answer, not just the clock: sweeping the fixture corpus at `1` moves 15 of 118 legs and loses two, `deb7` (exact) from `SolveSucceeded` to `ErrorInStepComputation` and `cresc4` (limited-memory) from `SolveSucceeded` to `InfeasibleProblemDetected`, while others improve. A per-problem lever, not a global one. Ignored when `feral_refine=no`, which is now the default — set `feral_refine=yes` before either knob has any effect. |
| `feral_refine_target`        | `0`     | Residual level at which FERAL's inner refinement is **skipped entirely**, as a relative 2-norm `‖b − A·x‖₂ / ‖b‖₂` on the unrefined solve. `0` (the default) disables the check, so every back-solve refines. Where `feral_refine_steps` truncates the refinement on every solve alike, this one decides *whether* it runs, per solve — which is the difference that matters, because FERAL's `RefineOptions` carries a step cap and no target and so converges to `eps*sqrt(n)`, the tightest residual the arithmetic admits, while `PdFullSpaceSolver` accepts a step at `residual_ratio_max = 1e-10` on the unreduced system. On the 126028-dimension `laptime` KKT the unrefined solve already lands in the 1e-11 band against a hard-wired target of 7.9e-14, and setting `1e-8` takes the solve from 67.2 s to 28.7 s (back-solve 53.4 s → 18.6 s). Unlike `feral_refine=no` this still refines the solves that need it: the gh#590 noise-floor LP (data scale 1e11) keeps its certificate at `1e-8`, which no `feral_refine_steps` value achieves. It is still a **per-problem lever, not a global one** — at `1e-8` the fixture corpus moves 17 of 118 legs, losing `eigena2` (limited-memory) from `SolvedToAcceptableLevel` to `ErrorInStepComputation`, `eigenb2` from `SolveSucceeded` to `SolvedToAcceptableLevel`, and taking `pooling_rt2stp` (exact) from 128 to 413 iterations for the same objective, while `autocorr_bern55-06` and `cresc4` improve. Reach for it when the timing report shows `LinearSystemBackSolve` dominating on a large KKT, and check the answer. Ignored when `feral_refine=no`, which is now the default — set `feral_refine=yes` before either knob has any effect. Upstream fix: feral#190. |
| `feral_cascade_break`        | (unset) | Tri-state. Unset → inherit feral's Phase B default (CB on with bounded delayed-pivot catchment). `yes` records explicit intent (no behavioural change). `no` reproduces pre-Phase-B behaviour by surfacing `DelayBudgetExceeded` on non-root cascade victims.  |
| `feral_fma`                  | `no`    | Dispatch dense kernels through fused multiply-add intrinsics. Roughly 2× throughput on aarch64 / x86_v3, at the cost of per-pivot rounding drift that trips more `WrongInertia` checks. Turn on when kernel throughput dominates and the IPM tolerates a noisier inertia signal. |
| `feral_singular_pivot_floor` | `1e-20` | Pounce's analog of MA57's `CNTL(2)`. After a successful factor, the smallest accepted `D`-block pivot magnitude (scaled space) is compared against this absolute floor; if it falls below, the factor is reported `Singular` so the IPM bumps `δ_w`. `0` disables. |
| `feral_inertia_pivot_floor`  | `1e-12` | Pivot magnitude below which a *mismatching* inertia count is treated as noise rather than as evidence (#540). Consulted only once the negative-eigenvalue count already disagrees with what the IPM asked for: if the smallest accepted pivot (scaled space) is under this floor, the factor is reported `Singular` instead of `WrongInertia`, so `δ_c` — the perturbation that repairs a rank-deficient constraint block — is applied before the `δ_w` ladder starts multiplying by 8 per retry. Because it only ever fires on a factor the caller was already going to reject, it cannot turn a usable factorization into a failure. Necessarily larger than `feral_singular_pivot_floor`, which governs factors that are unusable outright. `0` disables. |
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
| `POUNCE_FERAL_REFINE_STEPS`         | `feral_refine_steps`         |
| `POUNCE_FERAL_REFINE_TARGET`        | `feral_refine_target`        |
| `POUNCE_FERAL_CASCADE_BREAK`        | `feral_cascade_break`        |
| `POUNCE_FERAL_FMA`                  | `feral_fma`                  |
| `POUNCE_FERAL_SINGULAR_PIVOT_FLOOR` | `feral_singular_pivot_floor` |
| `POUNCE_FERAL_INERTIA_PIVOT_FLOOR`  | `feral_inertia_pivot_floor`  |
| `POUNCE_FERAL_MIN_PAR_FLOPS`        | `feral_min_par_flops`        |
| `POUNCE_FERAL_STATIC_PIVOTING`      | `feral_static_pivoting`      |

These variables are parsed by `feral::env` (feral#176), which accepts the
same spellings the option parser does — including scientific notation, so
`POUNCE_FERAL_MIN_PAR_FLOPS=1e8` sets the documented default rather than
being silently discarded, as it was before feral 0.17.0 when pounce parsed
these with a bare `str::parse`. A value that is out of range for the
target type is clamped to the maximum; a value that cannot be parsed at
all, or that fails the knob's stated requirement (e.g. a negative pivot
floor), is refused with a one-time warning on stderr and the default is
used. A refused variable never silently changes numerics.

`FERAL_PARALLEL` (legacy, no `POUNCE_` prefix) forces feral's internal
factor serial or parallel process-wide — `0`/`off`/`false`/`no` to force
serial, `1`/`on`/`true`/`yes` to force parallel, and unset to leave
feral's own platform-derived default alone. The force-on direction is the
only override available to CLI, Python and NL callers on a host where
that autodetection is wrong (feral falls back to sequential when the
rayon pool fails to build); the first-class per-backend lever,
`FeralConfig.parallel`, is the Rust solver API, not an option.

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
| `POUNCE_DBG_CLASSIFY` | — (stderr) | The detected problem class and the finding that produced it, next to the `.nl` header's own nonlinearity census. This is the line to read when a model routed to a solver you did not expect. No `RUST_LOG` needed. |
| `POUNCE_DBG_CONSTDERIV` | — (stderr) | Which of the three [constant-derivative](#options-pounce-does-not-implement) cases fired for each of the four `*_constant` hints — the proof, whether you asserted it, and whether the derivative is reused. No `RUST_LOG` needed. |
| `POUNCE_DBG_GONDZIO` | — (stderr) | One line per convex (LP/QP/conic) solve: which driver ran, its iteration count, and how many Gondzio centrality correctors were attempted, how many accepted, and their mean step-length gain. Read it to tell whether `qp_gondzio_corr` is doing anything on your model — an `attempted=0` line means the cone is not a pure orthant, or the option is `0`. No `RUST_LOG` needed. |
| `POUNCE_DBG_NO_QUAD` | — (no output) | **Changes what runs, rather than emitting.** Turns off quadratic recognition, so every `.nl` body keeps its expression tree and is evaluated through the AD tape rather than from stored constant structure — both the expanded read-out `½xᵀHx + aᵀx + c` and, since gh#673, the factored one `Σ wₖ(bₖᵀx + dₖ)²` a sum of squared residuals keeps. This is the A/B switch the quadratic evaluator is measured with: if a model's numbers move when it is set, the evaluator is the difference. Slower by construction, and larger in memory. It is **not** a general "pre-quadratic" switch — in particular the constant-derivative proofs behind the four `*_constant` hints read the same recognizer through the tree, so they resolve identically either way and `POUNCE_DBG_CONSTDERIV=1` prints the same verdicts with it set. |
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
