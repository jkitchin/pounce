# Crossover: identifying an exact active set

An interior-point method never puts an iterate *on* a constraint. The
fraction-to-boundary rule keeps every slack strictly positive, so when the
solve converges, "which constraints are active" is something you **infer from
a tolerance test**, not something the solve established.

Usually the inference is right. Where it is not — and the case where it is
not is sharply defined — POUNCE can run an opt-in **crossover** phase: after
the interior-point solve converges, it pivots to the active-set path and
returns a point at which a linearly independent set of constraints is
satisfied to *equality*, with multipliers that certify stationarity against
exactly that set.

```
crossover                yes | no     (default: no)
crossover_max_iter       integer      (default: 30)
crossover_mult_tol       number       (default: 1e-8)
crossover_primal_tol     number       (default: 1e-6)
```

> This is the NLP feature. The convex LP path has its own, separate
> `qp_crossover` option, which purifies an LP iterate to an exact *vertex*
> — different engine, different problem class. Setting one does not affect
> the other.

## When you need it

The discriminating property is **failure of strict complementarity**: a
constraint that is active but whose multiplier is zero. At such a point the
barrier's own geometry places the iterate `O(√μ)` from the constraint. With
`μ` around `1e-9` at termination, that is a distance of about `1e-5` — four
orders of magnitude *larger* than the `1e-8` tolerance the solve reports
converging at. No tolerance test applied afterwards can recover the answer,
because the information is not in the iterate.

Three parts of POUNCE already pay for this:

1. **Sensitivity.** `sens_covariance()` classifies each constraint as STRONGLY
   ACTIVE / WEAKLY ACTIVE / **AMBIGUOUS (loosely converged)** / UNIDENTIFIED
   (see [Sensitivity Analysis](sensitivity.md)). The AMBIGUOUS class exists
   because the interior iterate cannot decide. Crossover collapses it.
2. **Degeneracy.** A degenerate solution collapses the reduced Hessian, and
   that has repeatedly surfaced as an inertia problem met with a
   perturbation-side heuristic. Crossover attacks the same thing
   structurally: it produces a *linearly independent* active set.
3. **Warm starts.** The active-set SQP could previously only warm-start from
   a previous *SQP* solve ([Active-Set SQP & Warm
   Starts](active-set-sqp.md)). After crossover,
   `last_sqp_working_set()` returns the identified set, so a sequence whose
   first solve wants the interior method — MPC with a cold first solve,
   parameter continuation — can hand off to the active-set path.

Symptoms that point here: a parameter estimate sitting exactly at a bound
with a confidence interval you do not trust; shadow prices off a degenerate
model; an over-modeled engineering model where several constraints bind at
once.

## When you do not

If your solution satisfies strict complementarity — the bulk of well-posed
models — crossover has nothing to correct. It will run, take one step, find
the point already satisfies the stopping tolerances, and return it. That
costs roughly one extra iteration and changes nothing. Leaving it off is the
right default.

It also does nothing useful on a solve that did not converge: an unconverged
interior point is not a KKT point, so there is no active set at it worth
identifying. Crossover is skipped unless the solve reached `Solve_Succeeded`
or `Solved_To_Acceptable_Level`.

## What it does

The phase follows Byrd, Nocedal & Waltz, *KNITRO: An Integrated Package for
Nonlinear Optimization* (2006), §7.

1. The interior-point method terminates at `(x, y, z)` within its tolerance.
2. **Estimate the active set** by a tolerance test on primal distance
   (`crossover_primal_tol`) and multiplier magnitude
   (`crossover_mult_tol`).
3. **Take one EQP-equivalent step** over that set, plus a line search on the
   ℓ₁ penalty model with `ν₀` set just above the largest `|multiplier|` at
   the interior solution. If the result satisfies the stopping tolerances,
   stop. This is the common path, and it solves no LPs.
4. Otherwise **run the full active-set SQP** from the interior iterate,
   seeded with the estimated set and the same `ν₀`, for at most
   `crossover_max_iter` outer iterations.

### It works against the bounds you declared

The interior method widens every bound by `bound_relax_factor` (default
`1e-8`) before it starts — that widening is what lets an iterate approach a
bound without ever being pinned to it. Crossover undoes it: the pivot and
the activity test both run against the box and row bounds as *written*.

This is not a detail. A point sitting exactly on a declared bound is a full
`1e-8` **inside** the relaxed one, so measured against the relaxed bounds
every binding constraint reads as inactive and the pivot stops just short of
each one. Crossover would run, succeed, and report an empty active set.

A consequence worth knowing: the crossed-over point can sit `~1e-8` closer
to a bound than the interior solution did, and on the bound rather than
inside it. That is the intended result, and it is inside `constr_viol_tol`
by construction — the relaxation is capped there. It also makes
`honor_original_bounds` a no-op on a crossed-over solution: the point is
already in the declared box.

### Where this departs from the paper

KNITRO's active-set path is SLQP — an LP phase picks the working set, an EQP
phase computes the step. POUNCE's is an ordinary line-search SQP over
`pounce-qp`'s working-set interface. So:

- Step 3 is one `pounce-qp` solve against the NLP linearization at the
  interior iterate, warm-started with the estimated set. That call
  factorizes the hinted set to recover a primal and then pivots, which is
  what "solve the EQP over `A`, and fix `A` where the tolerance test got it
  wrong" amounts to here. The paper's property that step 3 avoids an LP is
  preserved.
- Step 4's LP trust region (the paper's eq. 7.22, sized to exclude every
  inactive constraint) has **no analogue** in a line-search SQP and is not
  implemented. Only the `ν₀` half of that setup is reproduced.

## It cannot make a solve worse

Crossover is a refinement of a solve that already succeeded, so the bar is
not "did it solve" but "is this at least as good a KKT point". The
crossed-over point replaces the interior one only if **all three** hold
against the interior iterate:

- constraint violation no worse, allowing movement within
  `sqp_constr_viol_tol`;
- stationarity no worse, allowing movement within the stationarity
  tolerance;
- the objective did not increase beyond a small relative slack.

It must also be a solution of **your** model, in your units: the
crossed-over point has to meet `constr_viol_tol`, `dual_inf_tol` and
`compl_inf_tol` unscaled, against the bounds and rows as you declared them
(gh#1054). The three comparisons above read the SQP's residuals, which live in
POUNCE's internally scaled space, and gradient-based scaling can shrink a row's
violation by orders of magnitude: on `hs13_bigstart` (started at `x₀ = 1e4`)
the SQP's degenerate tail jumped to a point that violates the declared row by
`1` and read `3.3e-7` in the scaled space. The interior-point verdict is issued
in user units, so the point that replaces it is held to the same gate; a point
that fails it is declined with `UserTolerances`.

Any failure returns the interior solution untouched. The tolerances rather
than the raw residuals are the comparison point on purpose: crossover puts
the iterate *on* the active constraints, which can move a residual from
`1e-12` to `1e-10` while the point is unambiguously better identified.
Refusing that would reject exactly the cases the phase exists for. What the
gate still refuses is a residual crossing its own tolerance.

Because it runs strictly *after* convergence and is off by default, enabling
it moves no interior trajectory.

## Which bounds the reported residuals are measured against

`bound_relax_factor` (default `1e-8`) widens every bound by `δ` before the
solve. That is invisible during the interior iteration, which never touches
even the widened bound — but crossover puts the iterate *exactly* on the
declared one, which is `δ` **inside** the relaxed one. Measured in the
relaxed frame the returned point therefore has slack `δ` at every active
constraint, and the complementarity term reads `v·δ` rather than `~μ`.

For a unit multiplier and the default relaxation that is `1e-8` — which is
`tol`. So the summary printed a converged, strictly better point as having
an `Overall NLP error` at or above the tolerance it converged at, and the
opt-in `kkt_fidelity_tol` gate (applied *after* crossover) downgraded
`Solve_Succeeded` on it. That was [#646](https://github.com/jkitchin/pounce/issues/646),
and it is fixed: **when crossover is accepted, the reported complementarity
and the two KKT aggregates are measured against the declared bounds** — the
frame crossover solved in, and the frame the never-regress gate already
judged the point in.

Measured on HS14 (strictly complementary, `v ≈ 1.85`):

| | without crossover | with crossover |
|---|---|---|
| Dual infeasibility | `1.9e-12` | `8.9e-16` |
| Constraint violation | `2.9e-13` | `2.2e-16` |
| Complementarity | `2.5e-09` | `3.5e-16` |
| Overall NLP error | `2.5e-09` | `8.9e-16` |

Two details of the substitution are worth stating, because they are the
places it could have been done wrong:

- **Only complementarity moves.** Stationarity involves no bounds, and the
  crossed-over point is strictly *interior* to the relaxed box, so its
  constraint violation is zero under either reading.
- **The slacks are raw.** The interior machinery floors a slack that falls
  below `eps·min(1,μ)` up to about `μ/z`, which is part of what keeps the
  barrier's `Σ = V/S` finite while the iteration runs (the other part is
  the representability floor `s ≥ max_i z_i / (f64::MAX/4)`, which is what
  covers a subnormal `μ` — see below). At a purified point the active
  slacks are *exactly* zero, and that floor would put `μ/z ≈ 1e-9` straight
  back — reintroducing as a reporting artifact the very quantity crossover
  removed. The declared-frame measurement does not apply the `μ/z`
  correction. It does carry the representability floor, for the reason
  given below.

This is a change to *reporting* only. It runs after the exit status is
already decided, and it applies solely to a point the never-regress gate
accepted on its declared-bound residuals, so it cannot dress up a worse
iterate — the reading it replaces is the artifact, not the point.

## What it does to a downstream sensitivity result

Crossover moves the iterate onto its active bounds, which changes the
barrier diagonal `Σ = z/s` the sensitivity path factorizes. That was
expected to be a hazard — a slack driven to zero divides badly — and it
is the opposite. `Σ` is the stiffness with which the barrier pins a
bounded variable, and a reduced Hessian read off the held KKT factor
carries a residual error of exactly `O(1/Σ)`: the leftover of that pin
being finite rather than exact. A **larger** `Σ` is a sharper pin and a
more accurate answer.

Measured on `min ½xᵀQx − qᵀx` with two parameters held by pin rows and a
third variable capped by a bound that binds with multiplier `4.5`. The
reduced Hessian over the pins has an `O(1)` gap between the
bound-pinned answer and the free one, so drift is unmissable:

| | `Σ` at the active bound | reduced-Hessian error |
|---|---|---|
| `crossover=no` | `8.1e+09` | `4.95e-10` |
| `crossover=yes`, `bound_relax_factor=0` | `2.0e+16` | **`4.44e-16`** |
| `crossover=yes`, default relaxation | `2.0e+16` | **`4.44e-16`** |

**Against the bounds as declared, crossover sharpens the result by
exactly the factor `Σ` grew.** The error is `Q_aw²/Σ`, the bound block's
Schur complement, and it holds to every printed digit until `Σ` grows
large enough that the prediction drops below the roundoff of the answer
itself — which is where the two crossover rows above sit. With the point
*on* its bound the pin is as exact as double precision expresses.

**The two crossover rows are identical, and that is recent.** Until
[#654](https://github.com/jkitchin/pounce/issues/654) the second one read
`4.5e+08` / `8.89e-09` — 18× *worse* than not crossing over at all,
rising toward 400× as the bound's multiplier grew. The crossed-over point
sits exactly `δ = bound_relax_factor` inside the live relaxed bound, so
the barrier saw a slack of `δ` where an interior iterate would have
carried `μ/z`, making `Σ = z/δ` instead of `z²/μ` and *loosening* the pin
by `z·δ/μ`. That was the same frame mismatch as
[#646](https://github.com/jkitchin/pounce/issues/646) reaching the
numerics rather than the printed residuals, and it is fixed the same way:
**when crossover is accepted, `Σ` is re-measured against the declared
bounds** — for variable bounds and inequality-row bounds alike — before
the sensitivity path factors with it or classifies against it.

The correction is applied at the consumer boundary, not on the live
iterate: the relaxed bounds are still what the algorithm ran against, and
nothing about the solve moves. It covers `sens_covariance()`,
`sens_information()`,
`classify_activity()`, `compute_reduced_hessian`, the parametric steps,
and the `SensSolve` builder, because all of them read the one held
factor.

So `crossover=yes` and `bound_relax_factor = 0` are now independent
choices: a crossed-over solve reports the same downstream numbers either
way. You may still want `bound_relax_factor = 0` for
`classify_activity()`, which requires it for an unrelated reason — the
central-path checks it makes read the barrier's own slacks, which the
relaxation shifts.

`Σ` never becomes infinite, in either frame — but the two frames get
there by different floors, and it is worth knowing which supplies the
guarantee where.

On the **live interior path**, `CalculateSafeSlack` carries two. The
first, `eps·min(1, μ)`, is a threshold on the *barrier* term; nothing in
it mentions the multiplier, so it bounds `Σ` only because a normal `μ`
makes `μ/z` a usable slack. Push `μ` into the subnormal range and it
stops covering anything — at `μ = 9.1e-308` the threshold is `2.0e-323`,
small enough that a slack of `2.0e-308` clears it untouched and `z/s`
overflows ([#655](https://github.com/jkitchin/pounce/issues/655)). What
makes `Σ` finite unconditionally is the second, `s ≥ max_i z_i /
(f64::MAX/4)`, which is stated in terms of the quantity that has to stay
representable rather than in terms of `μ`.

The **declared frame** does not go through that function at all — the
`max(μ/z, s_min)` correction is exactly the standoff crossover exists to
remove, and applying it would put back as an artifact what the phase
just took out. So it carries its own pair: `eps·max(1,|bound|)`, the
distance at which the point *is* the bound, which covers a pivot landing
on it exactly; and the same `max_i z_i / (f64::MAX/4)` as above, because
a representability bound is about what a double can hold rather than
about where the barrier would have put the point, and it would otherwise
stop at the frame boundary.

Neither floor is reached in ordinary use. On the fixture in the table
above the crossed-over slack does reach the first one; the slack measured
in [#653](https://github.com/jkitchin/pounce/issues/653) bottomed out at
`1.8e-12` — the residual of the QP step plus line search — and reached
neither. The representability half matters for solves at pathological
tolerances, not for these.

There is a bound at the other end too, and it is not symmetric with
these. A floor keeps `Σ` from leaving the double range; the ceiling
[#737](https://github.com/jkitchin/pounce/issues/737) added keeps it
from swamping the constraint rows the same variable sits in, which
happens at a `Σ` that is perfectly representable. It applies to the
sensitivity system in either frame, and only to a variable that appears
in a constraint row — never to the bound-pinned, otherwise-unconstrained
variable of the table above, whose stiffness is the accuracy being
measured. See [A Param pinned to exactly a
bound](sensitivity.md#a-param-pinned-to-exactly-a-bound).

## Reading the result

The returned solution — `x`, the objective, `g`, and every multiplier — is
the crossed-over point, reported through the same path as any other solve.
There is no separate "crossover solution" to fetch.

Beyond that, from Rust:

```rust
let status = app.optimize_tnlp(tnlp);

// None  ⇒ crossover never ran (option off, or the solve did not converge).
// Some  ⇒ it ran; `accepted()` says whether it replaced the interior point.
if let Some(r) = app.crossover_report() {
    println!("accepted: {}", r.accepted());
    println!("phase:    {:?}", r.phase);          // EqpStep | ActiveSet
    println!("declined: {:?}", r.declined);       // why, when it did not
    println!("active:   {} bounds, {} rows", r.active_bounds, r.active_constraints);
    println!("estimated {} active before pivoting", r.estimated_active);
    println!("KKT {:e} -> {:e}", r.kkt_before, r.kkt_after);
    println!("complementarity (declared frame): {:e}", r.compl_after);
}

// The identified set, ready to seed an `algorithm=active-set-sqp` solve.
let ws = app.last_sqp_working_set();
```

"Crossover never ran" and "crossover ran and declined" are different facts
about a solve, and the reason `crossover_report()` distinguishes them: a
consumer reasoning about active-set certainty must not read a declined
crossover as a confirmed active set.

`estimated_active` versus `active_bounds + active_constraints` is the
measurement the phase exists to make. They differ exactly where the
tolerance test on the interior iterate was wrong.

Crossover spends the solve's own `max_wall_time` / `max_cpu_time`: what the
interior-point phase left, not a fresh budget. When it runs out, crossover
declines with `TimeLimit` and the interior solution stands. Before gh#1054 it
had no budget at all, and a large model could sit in crossover's first QP for
as long as it took.

## Handing over before convergence (`ipm_sqp_handover`)

```
ipm_sqp_handover               yes | no   (default: no)
ipm_sqp_handover_tol           number     (default: 1e-4)
ipm_sqp_handover_stall_iter    integer    (default: 0, off)
ipm_sqp_handover_max_attempts  integer    (default: 3)
```

Crossover runs after the interior method has converged, so it cannot shorten
a long interior-point tail — and on large optimal-control models that tail is
where the time goes: the reporter of gh#1054 measured the objective within
0.1 % of final after ~40 % of the iterations, the rest spent with the dual
infeasibility oscillating and the filter backtracking. `ipm_sqp_handover=yes`
offers the *unconverged* iterate to the same bridge:

- **When**: once the overall NLP error is at or below `ipm_sqp_handover_tol`
  (a loosened `tol`), or — with `ipm_sqp_handover_stall_iter = k > 0` — once
  the best error has not halved in `k` iterations, whatever its value.
- **What**: the active set is read off the multipliers against the slacks.
  At an interior iterate `s·z ≈ μ`, so `√μ` separates active from inactive and
  the primal tolerance is `max(crossover_primal_tol, √μ)`; the equality rows
  are always in. One QP step is taken over that set, and if it does not meet
  the stopping tolerances the active-set SQP runs from that working set for
  up to `crossover_max_iter` iterations — the same two phases as above.
- **Accepting**: only a point that meets the stopping tolerances, in the
  scaled space *and* in user units on the declared model (see
  [It cannot make a solve worse](#it-cannot-make-a-solve-worse)). The solve
  then ends `Solve_Succeeded` at that point, and its working set is published
  through `last_sqp_working_set()` for the next SQP solve.
- **Falling back**: a declined attempt changes nothing the interior method
  carries, so it simply continues from the iterate it was at. The next attempt
  waits until the error is ten times smaller, or another stall window has
  passed, and at most `ipm_sqp_handover_max_attempts` are made.

Swept over the CLI fixture corpus at the defaults (both legs, 234
fixture-legs), against the same binary with it off: 73 finish in fewer
interior-point iterations, 3 report more, 158 are unchanged, and no status gets
worse. The three that report more are models whose plain solve is re-run by the
quality audit (`csfi2`: 35 + 21 iterations; the sweep records the last solve
only) and finish in one solve with the handover (30). Two move from
`Solved_To_Acceptable_Level` to `Solve_Succeeded` (`mu_fallback_point_floor`,
55 → 5 iterations). Every objective that moves goes *down* — the handover lands
on the declared bounds rather than `O(μ)` inside them, and on nonconvex models
it sometimes reaches a better point (`mpcc_worse_local_solution`: −13.0 →
−84.3) — except `jit1`, where it is `4e-6` higher, because the interior method
solved the problem with its bounds widened by `bound_relax_factor` and the
handover solves the problem as declared. `app.handover_report()` (Rust) says
how many attempts were made, at which iteration one was accepted, and the
crossover report of the last one.

An accepted point is also put through the interior method's own
negative-curvature escape before it is certified, exactly as its convergence
exit does: the SQP's one-step path has no curvature test, and without the
escape the handover ended at the stationary point the interior method leaves
on `nonconvex_qp`, `nonconvex_qp_ineq` and `nonconvex_two_escapes`. And each
attempt may spend at most the interior-point time used so far (at least one
second), within the solve's own budget — on `deb7` an unbounded attempt ran the
full-SQP fallback for minutes against a 114-iteration interior solve.

What it does not change is the cost of the QP itself. An attempt is one
active-set QP over the whole model warm-started from the identified set, so it
costs about one sparse factorization plus one pivot per misidentified row: it
pays where the active set is settled and the interior tail is long, and an
attempt at a point where the set is still moving is a QP solve thrown away.
That is what the back-off and the attempt cap bound.

## Scope

Crossover is an opt-in post-convergence phase, and the handover an opt-in
use of the same bridge before convergence. Neither adds SLQP as an
algorithm, neither is a default, and with both off nothing about the
interior iteration changes.
