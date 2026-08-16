# Changelog

All notable changes to POUNCE are tracked here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `1.0.0`. Pre-1.0 minor bumps may include breaking
changes.


## [Unreleased]

- **`WarmStart` artifacts carry the model they belong to, and warm-start
  options no longer outlive the call that asked for them** (#607). Two
  independent silences in `pounce.WarmStart`, both of the shape gh#544
  taught us to distrust — the answer is fine, everything else is not.

  *Option scoping.* Applying a warm start called `add_option` on the
  **persistent** `Problem`, and `add_option` is append-only, so the seven
  enabling options (`warm_start_init_point`, `mu_init`, the four
  `warm_start_*_push` / `_frac` and `warm_start_mult_bound_push`) stayed
  set for every later solve on that object. Measured on HS071 at the
  parent commit `70bf53de`: an ordinary cold solve that takes 17
  iterations on a fresh `Problem` takes **24** on one that has served a
  warm solve — the same objective to ten digits, 41% more iterations, and
  nothing anywhere saying why. A warm solve that *raised* left the same
  residue. They are now installed as a scoped overlay and taken back in a
  `finally`, so both cases come back to 17. The overlay snapshots and
  restores the whole option list rather than unsetting a list of names,
  so an option added to `WarmStart.options()` later is scoped by
  construction.

  *Persistence schema.* A serialized warm start recorded arrays and four
  floats — no dimensions, ordering, sparsity, bounds, scaling, algorithm
  or model fingerprint — so an incompatible replay was simply attempted.
  Replayed against the same model with its variables reordered it
  returned objective **16.3801** where the truth is 17.0140, `x` wrong by
  0.257, status `Error_In_Step_Computation`, no exception and no warning;
  against a changed box, 15.8436 and `Restoration_Failed`; against a
  different model of the same shape, a clean `Solve_Succeeded` for the
  wrong reason. The archive schema is now versioned (v2) and carries a
  `ProblemSignature`: dimensions, optional stable variable/constraint
  IDs, and digests of the bound signature, declared sparsity, scaling
  convention, algorithm/backend and the model-defining options. Capture
  it with `WarmStart.from_info(x, info, problem=prob)`; a mismatch is
  refused **before the solver is entered**, with a report naming every
  facet that moved and the ways forward.

  `compat="strict"` (default) raises, `"warn"` warns and proceeds,
  `"unsafe"` skips the check. Replay is labelled `exact` or `mapped`:
  `ws.transfer(prob, mapper)` is the explicit hook for a horizon shift or
  a changed discretization, `ws.reindex(prob, var_ids=…)` writes the
  mapper for you when both sides carry stable IDs, and a mapped artifact
  is still refused on a third problem. Unmapped multiplier entries are
  seeded `NaN`, the native "unseeded" contract, rather than fabricated.

  *Migration.* Version-1 archives — everything written before this — load
  and replay unchanged; they are *unverifiable*, not incompatible, and
  refusing them outright would be a migration cost with no safety return.
  What they get is a one-line `WarmStartLegacyWarning` naming
  `ws.migrate(prob)`, plus the one check their own arrays support: the
  dimensions. An unsigned *in-memory* `from_info(x, info)` stays silent,
  as before. A v2 archive stays readable by a v1 loader — the new keys
  are additive and `_meta` is untouched.

  Known limitations, stated because one of them is a case the issue
  names: a permutation of a model with a uniform box and a dense
  jacobian leaves every structural digest bit-identical, so a reordering
  is detectable only when the caller supplies `var_ids` on both sides.
  And a mapped interior-point warm start is a correctness mechanism, not
  a speed-up — on a receding-horizon fixture the transferred point costs
  12 iterations against 7 cold, widening to 30 against 10 at horizon 40,
  which is the barrier/active-set limit `docs/src/initialization.md`
  already describes.

  Python-side, plus four additive `Problem` accessors
  (`options_snapshot` / `restore_options`, `get_bounds`,
  `get_problem_scaling`, `problem_obj`); no solver step moved. The
  fixture sweep is identical across all 57 models — status, objective and
  iteration count.

- **The cold-start initialization options are settable** (#604).
  `bound_push`, `bound_frac`, `slack_bound_push`, `slack_bound_frac`,
  `constr_mult_init_max`, `bound_mult_init_val`, `bound_mult_init_method`
  and `least_square_init_primal` were read by the algorithm builder but
  never registered, so every frontend rejected them at the *set* call
  with `Unknown option "bound_push"` — the documentation described knobs
  that no supported path could reach. This is the inverse of #551
  (registered-but-unread), and the same class of silence.

  They now register under upstream's `Initialization` category with
  upstream's types, defaults and ranges. The registered defaults equal
  the values the builder already hard-coded and the read sites still fire
  only when a caller sets the key explicitly, so nothing moves unless you
  ask it to: the fixture sweep is identical across all 57 models, status,
  objective and iteration count.

  Two of upstream's knobs name behaviour POUNCE does not have, and both
  now say so instead of doing something else. `bound_mult_init_method=
  mu-based` used to fall through to a third path — the NLP's own `y`
  guess — that is neither documented mode; it is refused, as is
  `least_square_init_duals=yes`. Both values still *parse*, so an
  `ipopt.opt` written for Ipopt loads unchanged.

  A registry invariant test now runs in CI in both directions: no option
  is read without being registered, and no `Initialization` option is
  registered without being either consumed or explicitly refused.

- **The convex LP/QP knobs are registered core-side and refused where
  they configure nothing** (#604). `qp_tau`, `qp_tau_max`, `qp_reg`,
  `qp_infeas_tol`, `qp_hsde`, `qp_equilibrate` and `qp_crossover` were
  registered by the `pounce` CLI at startup, onto the same registry the
  core builds. Setting one from Python or the C interface therefore
  failed with `Unknown option "qp_tau"` — naming an option that exists —
  and adding any `qp_*` name to the core registry would have aborted the
  CLI binary at startup with `OPTION_ALREADY_REGISTERED`, a hazard
  documented in a comment and tested by nothing. #360 moved the
  `sqp_qp_*` block out of the CLI for those two reasons; this is the
  other half.

  They now live beside `solver_selection` and `qp_presolve` in the core
  registry, so every frontend parses them. Parsing is not honouring: only
  the CLI classifies a `.nl` model and routes it to the convex engines,
  so a library solve now **refuses** a non-default `qp_*` (including
  `qp_presolve`, previously a documented silent no-op there) with a
  message naming the surfaces that do honour it — `option_file_name`'s
  treatment from #518, one feature over. Explicitly-set defaults still
  pass, as everywhere else. CLI behaviour is unchanged, including the
  fallback where a convex attempt hands the model to the NLP path having
  genuinely used those values.

- **A feasible NLP whose constraint rows sit at their own floating-point
  resolution is no longer refused a certificate — or convicted of local
  infeasibility** (#590). Reported against LyoPRONTO's pseudosteady-limit
  continuation study: `pounce-solver 0.10.0` returned
  `Infeasible_Problem_Detected` on the Problem 1 `f = 0.02` rung, a
  known-feasible optimal-control problem that Ipopt 3.14.16 solves. Through
  Pyomo that status is indistinguishable from a genuine infeasibility proof,
  which makes it the most damaging verdict the solver can get wrong.

  The model is written in Landau coordinates, so its conduction rows carry
  `1/(H − S)²` and reach magnitudes near `1e8`. At the point POUNCE stalled on,
  the scaled KKT error was `4.29e-10` against `tol = 1e-6`, and **no constraint
  row's residual rose above its own noise floor** — yet the unscaled violation
  read `1.62e-2`. One ulp of those rows is `~1e-2`, so that number is the
  quantum the rows are measured in, not a distance from feasibility. Ipopt
  lands on the same point with `8.06e-3` and calls it `Solved To Acceptable
  Level`; which side of `1e-2` a run falls on is arithmetic luck.

  gh #528 built the per-row noise floor for exactly this effect but gave it
  only to the strict **aggregate**, leaving the per-component `constr_viol`
  test on the raw residual. That was incoherent on its own terms — the
  component test is a refinement of the aggregate, so an unfloored component
  could veto a certificate the floored aggregate had already granted, which is
  precisely what happened here — and it left the rapid-infeasibility detector's
  absolute arm free to convict on a quantum. Both now consult the same floor,
  in both cases **only when no row rises above it**: one resolvable row
  anywhere and `constr_viol_tol` is back in charge. A genuinely infeasible
  model is untouched, its violation being pinned at its infeasibility gap,
  orders above `eps ·` the row's own magnitude.

  On the reported rung POUNCE now exits `Optimal Solution Found` in 20
  iterations at an endpoint of 6.14809 h, matching the issue's expected
  6.1481 h; before the fix it was `Infeasible_Problem_Detected` on 0.10.0 and a
  10000-iteration `Maximum_Iterations_Exceeded` on `main`. Ipopt needs 58
  iterations to reach only "acceptable" on the same rung. The full continuation
  ladder now walks past the reported failure to `f = 0.01`, every rung
  `converged_to_tolerance`. On the scale-equivariant LP family from gh #528,
  extended to the data scales where the quantum clears the issue's
  `constr_viol_tol = 1e-6`, the change moved 23 verdicts and every one of them
  improved — `Solved_To_Acceptable_Level`, `Search_Direction_Becomes_Too_Small`
  and `Maximum_Iterations_Exceeded` all becoming `Solve_Succeeded`, with no
  verdict degraded anywhere. `primal_noise_floor_kappa = 0` opts out of both
  gates, as it already did for the aggregate.

- **SOCP warm starts now support first-class variable bounds.** Bounds are
  expanded and bound duals normalized internally; warm solves force the direct
  driver for symmetric cones and may fall back to cold HSDE when enabled.
  Exponential/power cones use their dedicated cold HSDE route.

### Fixed — a released bound is now accurate at a tight tolerance (#587 follow-up)

`sens_boundcheck` / `estimate(mode="fix_relax")` could release a bound
onto the wrong answer, and did so more badly the *better* the solve had
converged. Asking for `tol = 1e-10` instead of `1e-6` made the released
coordinate about 30000x less accurate, with no warning: the check that
guards the refinement watches the condition it imposed, which a release
does satisfy, while the error lands in the other variables.

The cause is that an active bound contributes `sigma = z / s` to the
KKT's `x` diagonal. That term grows as the solve converges, and it
destroys the released system's information in the converged factor, so
no amount of rearranging recovers a release from it. Releases are now
computed by dropping the bound's `sigma` and re-factoring — one
factorization per released set, still well under a re-solve, and none
at all for a step that releases nothing.

Measured on the release model, error against the exact answer: at
`tol = 1e-8`, 2.4e-7 → 2.2e-12; at `tol = 1e-10`, 2.2e-4 → 9.6e-15. The
same holds under `obj_scaling_factor`, which reaches the defect by
moving where the solve stops: flat at ~1.5e-10 across six decades,
against 8.7e-4 at `obj_scaling_factor = 1000` before.

Pinning is unchanged, as is every path that releases no bound.

### Added — `estimate(mode="fix_relax")` bends the estimate around a bound instead of clipping it (#587)

`estimate()` takes the linear step, and where that step leaves a
variable's bound it clips the value and warns. Clipping costs more than
the one variable. Every other variable keeps the value the step gave it,
computed on the assumption that the clipped one was free to move where
the step said, so the result satisfies the bounds and no longer
satisfies the constraints. On a model where `y = 2x + 1` and `x` hits
its lower bound, the clipped answer is `y = -5`, which is not on the
constraint at all.

`mode="fix_relax"` repairs the active set the step implies, which is
upstream sIPOPT's strategy of that name (Pirnay, Lopez-Negrete and
Biegler 2012, section 2.5) and both of its cases. A variable the step
carries past a bound is pinned there, activating it. A bound multiplier
the step drives negative is set to zero, deactivating that bound so the
variable can move. Each adds a row to the held factorization and
re-solves through the Schur complement, so the others move with it.

Both halves matter and they fail differently. On the model above,
pinning returns `y = 1` against the clamp's `y = -5`. On a model whose
bound wants to release, the plain step is stuck at `x = 0` where the
answer is `x = 1.667`, because the step preserves complementarity and
nothing but the release lets the variable off the bound.

Checked against sIPOPT 3.14.19 itself, driven through
`pyomo.contrib.sensitivity_toolbox`, on cases built to separate the two:
pounce and sIPOPT agree to 2e-8 on the pin case and to 1e-6 on the
release case, and both match a full re-solve. On upstream's own
parametric example, at its own perturbation, the refinement lands within
6e-9 of a re-solve where clipping the crossing coordinate is off by
0.12.

    estimate(m, [(m.p, 3.0)])                       # clips
    estimate(m, [(m.p, 3.0)], mode="fix_relax")     # pins and re-solves

`mode="linear"` is the default and is unchanged.

Each pass rebuilds the Schur complement over the conditions so far,
so pass `k` costs one dense `k × k` solve and `k + 1` back-solves and
the total grows quadratically. The default `max_passes` of 16 is 136
back-solves. The factorization itself is never rebuilt, which is what
keeps this cheaper than re-solving. `max_passes` bounds that work and
is a budget rather than a safeguard: the refinement is only worth
running while it stays cheaper than the re-solve it replaces.

Two limits stop it short of holding every bound. The pass budget, which
a caller can raise. And the problem's degrees of freedom, which no
budget helps: each pin consumes one, and past that no step holds every
bound at once, so the augmented system is singular. A dense LU does not
report that, it returns a solution around 1e15, so each pass checks that
it achieved the displacement it asked for and drops the pin when it did
not. `estimate()` warns in both cases, names the variables still
outside, and says which limit was reached, since only the first can be
fixed by asking for more.

**Testing.** `crates/pounce-sensitivity/tests/parametric_cpp.rs` checks
the refinement against a full re-solve on upstream's example, on a
three-degree-of-freedom model where three coordinates cross at once and
all three pins hold, and for the refusal when the pins would exceed the
degrees of freedom. `pyomo-pounce/tests/test_fix_relax.py` covers the
Pyomo surface, including under a `user-scaling` change of variables, and
that the two modes agree exactly where nothing crosses.

### Changed — every parametric step now carries the barrier correction (#587)

`estimate()`, `gradient()`, `Solver.parametric_step` and the `SensSolve`
builder take their step against a factorization held at the solve's
final `mu`. On its own that estimates where the BARRIER problem's
solution moves, not where the original problem's does, and the two
differ by `O(mu)`. The paper's equation 11 closes the gap with one more
term, and it is now applied on every path.

This moves existing answers. At a converged tolerance the shift is
below anything a caller would notice: measured against sIPOPT on a
nonlinear model, agreement improves from 2e-9 to 4e-10 at `tol = 1e-8`.
Where the solve leaves `mu` loose it is the point of the change: at
`tol = 1e-3` the same comparison improves from 9e-6 to 2.4e-7. A caller
comparing against a value recorded from a loosely converged solve will
see a difference, and the new value is the better one.

There is no option for it. The barrier problem's answer is not one a
caller has a reason to want, and upstream applies the term
unconditionally as well.

### Changed — `sens_boundcheck` refines instead of clamping (#587)

The option is named after upstream sIPOPT's, and upstream's runs an
iterative Schur refinement. Pounce's ran a single-pass clamp, so the
behavior under a shared option name differed from the solver it names.
It now refines, which is the same computation `mode="fix_relax"` uses,
across `--sens-boundcheck`, `SensSolve::with_boundcheck`, and
`solve_with_sens(sens_boundcheck=True)`.

This changes what the option guarantees. The clamp always returned a
point inside the declared box. The refinement does not, because pins are
limited by the problem's degrees of freedom, and the CLI's help text no
longer promises it. The message it prints on stderr now reports pinned
coordinates rather than clamped ones.

What counts as outside a bound is no longer a separate tolerance. Three
numbers answered that question, disagreeing across surfaces: 1e-3 on the
CLI flag, 1e-9 in the Python binding, and a third invented inside
`estimate()`. It now comes from the solve's own `bound_relax_factor`,
which is the value that says how far outside a bound the solve was
willing to leave a converged point.

- **A premature `Solve_Succeeded` on a badly-scaled model, caused by two
  defects in the inertia-correction path** (#592). On a fixed-policy NLP from
  LyoPRONTO's Problem 2 GDP, POUNCE returned `Solve_Succeeded`, and restarting
  it from the returned primal point improved the objective twice — 25.096 s,
  0.079 %, in total — landing on the point IPOPT 3.14.16 reaches in one solve.

  The convergence test was not at fault: the certified point clears IPOPT's
  own unscaled component gates too (dual `9.994e-3 <= 1`, violation
  `2.844e-6 <= 1e-4`, complementarity `9.091e-7 <= 1e-4`). What differed was
  the *trajectory*, and two things in the inertia correction were sending it
  somewhere IPOPT does not go.

  **The inertia-trust floor was dimension-blind.** `feral_inertia_pivot_floor`
  (#540) decides when a mismatching inertia count is noise, by comparing the
  smallest equilibrated pivot against a floor. Its rationale is the backward
  error `n · eps`, but the value shipped was the constant `1e-12` — `n · eps`
  at `n ≈ 4500`, while these KKT systems are order 165–311, where `n · eps` is
  3.7e-14 … 6.9e-14. It now defaults to `n · eps` for the system actually being
  factored. Setting the option explicitly still pins an absolute floor for
  every dimension, and `0` still disables the trigger, so #540's opt-out is
  unchanged. Across the 57-fixture CLI corpus exactly two models moved — both
  the #540 models, both to *fewer* iterations, no status changes.

  **`δ_c` could be spent on a full-rank Jacobian and never withdrawn.** When
  the noise trigger fires, the handler raises `δ_c`. If the small pivot came
  from the Hessian block rather than a rank-deficient Jacobian, `δ_c` is the
  wrong medicine — but the handler kept it and climbed the `δ_x` ladder on top
  of it, reaching `δ_w = 1e2` on a system IPOPT regularises at `1e-4`. The
  resulting over-damped step froze the objective for eight iterations, and the
  loose-tolerance exit then fired at the reported point.

  Which block owns the smallest pivot would settle it directly, but that index
  is not exposed by the linear-solver backend. What does separate the two
  populations is how far the `δ_x` ladder climbs while `δ_c` is up: when `δ_c`
  is right it is right within one rung (#540's models never exceed one), and
  when it is wrong the ladder climbs without limit (4 rungs here, 14 on
  `pooling_rt2stp`). So a **`δ_c` walk-back** was added — after
  `perturb_delta_c_max_rungs` rungs (new option, default `3`) all four deltas
  are withdrawn, the degeneracy probe is reset, and `δ_c` is latched off for
  the remainder of that iterate. `w` marks the iteration line when it fires.
  `perturb_delta_c_max_rungs = 0` restores the previous behaviour exactly.

  With both fixed, the reported first solve reaches `31785.744274` in 27
  iterations — the reporter's IPOPT answer — so the restart has nothing left to
  improve. On the stricter criterion the reporter named — the *original cold
  GDP pipeline*, where every option-level workaround had failed — the cold
  solve now lands on IPOPT's phase-switch times (1.575762165 h / 3.917595809 h
  against 1.925104405 h / 3.924408024 h before) and two successive restarts
  reproduce it to twelve digits. `pooling_rt2stp`, which #544 had cost 812
  iterations against a pre-#544 206, comes back to 298.

  The reproducer is not vendored: it encodes LyoPRONTO's model equations and
  LyoPRONTO is GPL-3.0 against POUNCE's EPL-2.0. The behaviour is pinned by
  unit tests on the walk-back state machine and the dimension-aware floor, plus
  end-to-end tests on the already-vendored `pooling_rt2stp` fixture, which
  exhibits the same pattern. Investigation and evidence:
  `dev-notes/issue-592-restart-non-idempotence.md`.

  *Breaking (Rust API):* `pounce_feral::FeralConfig::inertia_pivot_floor` is
  now `Option<f64>`, where `None` selects the dimension-aware default.

- **A failed solve no longer raises on one Pyomo route and returns on the
  other** (#589). `pyomo_pounce`'s two status tables — `sens._STATUS_RESULT`
  for the legacy route and `v2._V2_STATUS` for `pyomo.contrib.solver` — each
  listed nine of the engine's twenty exits. The other eleven — among them
  `Restoration_Failed` — took the default, and on the v2 side that default was
  `SolutionStatus.noSolution`. `noSolution` is not a severity there; it is the
  switch that turns the solution loader off, so under the default
  `load_solutions=True` the same failed solve returned a results object from
  `SolverFactory("pounce")` and raised `NoSolutionError` from
  `SolverFactory("pounce_v2")`:

  ```text
  SolverFactory("pounce")     -> results.solver.termination_condition = error
  SolverFactory("pounce_v2")  -> NoSolutionError
  ```

  A restoration failure is an ordinary numerical exit: the engine stops at an
  iterate and reports it, and `sens_solve` captures that iterate before its
  non-converged early return — so the v2 route was declining to hand over a
  point it was holding. Both tables now cover every `ApplicationReturnStatus`,
  and no exit maps to `noSolution`, matching the rule the `.sol` route already
  follows ("a primal vector came back"). The fallback for an unrecognized
  status is `unknown` too, so a status added to the engine later cannot
  silently reintroduce the asymmetry, and a new test holds both tables to the
  Rust enum.

  Termination conditions get more specific with the added rows. On the legacy
  route the eleven exits reported plain `TerminationCondition.error` and now
  agree with the severity the `.sol` route gives the same solve, while naming
  the outcome more precisely than AMPL's bands can. Eight report
  `internalSolverError` (the 500 failure band verbatim); the two definition
  errors report `invalidProblem`, which `.sol` cannot distinguish from an
  internal failure; and `Search_Direction_Becomes_Too_Small` reports
  `minStepLength` where `.sol` says `maxIterations` for the whole 400 band.

  One of the eleven changes severity. `Search_Direction_Becomes_Too_Small` is
  now `SolverStatus.warning` rather than the default's `error`, matching the
  400 limit band — a stalled solve is a limit case, not a failure. A legacy
  caller branching on `status == error` for that exit will see `warning`. The
  other ten stay `error`.

  Callers on the v2 route were affected on every restoration failure; `drto`
  in particular, since its `dynamic_optimization` transform declares
  sensitivity parameters and so routes every model through `_sens_solve`.

- **The GAMS links report the same thing as each other, on every exit**
  (#589). Both links carry the same table, and checking them against the
  engine's enum turned up the same class of gap plus one of its own.

  Three exits — `Insufficient_Memory`, `Unrecoverable_Exception`,
  `NonIpopt_Exception_Thrown` — were in neither link's table and took the
  `default` arm, so a solve killed for memory was reported to GAMS as an
  internal POUNCE error. All three are mapped now, and both tables cover the
  enum, so `default` is reserved for a status POUNCE does not have yet.

  `Restoration_Failed` and `Invalid_Number_Detected` now set the objective
  row. `gmoSetSolution2` publishes the iterate as `x.l` for every exit, so
  leaving these two out of the has-a-solution set did not hide the point — it
  showed the point with an objective of `0` beside it. The report is guarded
  on `isfinite(obj_val)` in both links, which is what makes it safe: POUNCE
  leaves the objective at NaN when it refused the solve, and
  `Invalid_Number_Detected` is by definition an exit where something went
  non-finite. `Diverging_Iterates` and `Insufficient_Memory` stay out
  deliberately.

  The pip link's `gmoSolveStat_*` constants were wrong in three places:
  `SOLVESTAT_EVAL_ERR` was `11` (`gmoSolveStat_InternalErr`) and
  `SOLVESTAT_INTERNAL_ERR` was `12` (`gmoSolveStat_Skipped`), and four exits
  used `gmoSolveStat_SolverErr` where the C link uses `gmoSolveStat_Solver`.
  So the two links disagreed on four statuses and the pip link reported two
  more under names it did not mean. A wrong integer there is invisible without
  GAMS in the loop, so the values are now checked against GAMS's own
  `gams.core.gmo` — `gamsapi[core]` is pure Python and needs no license, and
  CI installs it for exactly this test.

- **An accepted solve no longer loads into Pyomo as a warning** (#591).
  `Solved_To_Acceptable_Level` is written into the `.sol` as AMPL
  `solve_result_num = 1` — IPOPT's own code for the same outcome
  (`STOP_AT_ACCEPTABLE_POINT`) — instead of `100`. Nothing reads that number
  in isolation; consumers key on the *band*, and the two bands are not
  interchangeable. Pyomo's legacy `.sol` reader maps `0`–`99` to
  `SolverStatus.ok` and `100`–`199` to `SolverStatus.warning`, both with
  `TerminationCondition.optimal`, so an accepted POUNCE solve arrived as

  ```text
  solver.status          warning
  termination_condition  optimal
  message                POUNCE 0.10.0: SolvedToAcceptableLevel
  ```

  and Pyomo logged "Loading a SolverResults object with a warning status"
  while the equivalent IPOPT solve ("Solved To Acceptable Level.") loaded
  clean. Any solver-swappable application whose accepted-solve contract
  includes `status == ok` had to special-case POUNCE.

  On the **v2** route the same code was worse than a warning: Pyomo's v2
  `.sol` reader maps `100`–`199` to `TerminationCondition.error`, so an
  accepted solve raised `NoOptimalSolutionError` under the default
  `raise_exception_on_nonoptimal_result=True`. That is fixed by the same
  change.

  The fix is in the emitted code, so it covers both routes into Pyomo: the
  `SolverFactory("pounce")` plugin and driving the POUNCE binary through
  Pyomo's generic `ipopt` ASL interface. The in-process sensitivity route
  (`pyomo_pounce.sens`) does not read a `.sol`, and its own table reported
  `SolverStatus.warning` for the same status; it now reports `ok`, agreeing
  with the `.sol` route and with the v2 interface (which already mapped it to
  `convergenceCriteriaSatisfied` / `SolutionStatus.optimal`). The convex
  engines' `OptimalInaccurate`, which maps onto the same NLP status, moves
  from `100` to `1` with it.

  Reduced accuracy is not swept under the rug: the distinction stays in the
  code (`1`, not `0`), in the `.sol` message line, in the JSON report's
  `status`, and in the console summary. `Feasible_Point_Found` — a usable
  point that did *not* meet the convergence criteria, which POUNCE's own
  interfaces do not call a success — deliberately stays in the `100`
  "solved, with a warning" band.

- **The convex wall-clock budget is now reachable from Python** (#585).
  `pounce.qp.solve_qp`, `solve_socp`, `solve_qp_batch`, and
  `solve_qp_multi_rhs` take `time_limit=` — seconds as a `float`, `None` (the
  default) meaning unbounded. Until now the deadline machinery below existed
  only for the CLI's `max_wall_time` and for Rust callers setting
  `QpOptions::time_limit` directly; no Python caller could ask a convex solve
  to stop at a budget, even though `QpResult.status` could already come back
  `"time_limit"`. `max_iter` was never a substitute — one IPM iteration may be
  a single KKT solve or a factorization plus several inertia-controlled
  refactorizations (plus a possible simplex crossover on the LP route), so
  per-iteration cost varies by more than an order of magnitude within a single
  solve. This is what a receding-horizon controller with a fixed control
  period, a scenario sweep that must not stall on one pathological instance,
  or any solve behind a request needs.

  The budget is **per instance, not per batch**: each solve opens its own
  deadline scope, so `time_limit=10` over 100 problems permits 1000 s of wall
  clock. That matches the Rust `pounce_convex::batch` semantics and is the only
  machine-independent reading, since a shared clock would make *which*
  instances get cancelled depend on rayon's scheduling. A verdict still
  outranks the clock (see below), so a status of `"time_limit"` always means
  the solve concluded nothing — never a wrong `"optimal"`. Both
  `method="ipm"` and `method="active-set"` honour it, as do both conic drivers.
  A negative, NaN, or infinite budget raises `ValueError` rather than being
  read as "no limit"; `None` is how that is spelled.

  Deliberately not plumbed: `pounce.jax` (and the torch layer), whose
  `_check_status` raises on `"time_limit"` because a non-KKT iterate makes the
  implicit-function gradient meaningless — a differentiable layer returning
  quietly wrong gradients under load is worse than one that runs long — and
  the build-once `QpFactorization` / `QpSensitivity` handles, which have no
  clear per-call budget semantics yet.

- Added solve-wide convex-engine wall-clock deadlines. Both
  `pounce_convex::QpOptions` and `pounce_qp::QpOptions` now expose
  `time_limit: Option<Duration>` and their status enums expose `TimeLimit`.
  Automatic LP/QP/active-set/SOCP routing forwards an explicitly set
  `max_wall_time`, shares it across setup and retry stages, reports AMPL result
  400 / `MaximumWallTimeExceeded`, and never reroutes a timed-out solve to NLP.
  This is source-breaking for exhaustive option literals and enum matches.

  A deadline changes when a solve stops, never what it answers. Two rules
  carry that:

  * **Cancellation is an error, not a value.** In `pounce-qp` it travels as
    the internal `QpError::DeadlineExpired`, so `?` propagation forces every
    caller to handle it, and only the `QpSolver` entry points turn it back
    into the soft `QpStatus::TimeLimit`. Without this,
    `factorize_with_inertia_control` — whose "success" is a right-hand side
    solved in place — could return `Ok` on a timeout, and
    `solve_equality_only` would read the *un-solved* `[-g; b]` back as
    `[x*; λ*]`, skip the guards a zero shift makes moot, and report a
    feasible-but-wrong point as `Optimal`. Covered by a regression test that
    reproduces the window deterministically.
  * **A verdict outranks the clock.** Every deadline check in `pounce-convex`
    runs after some inner solve has already returned, so the crossing can land
    between convergence and the check. `Optimal`, `OptimalInaccurate`, and the
    two infeasibility certificates now survive it; only `IterationLimit` /
    `NumericalFailure`, which conclude nothing, are relabelled. A problem that
    converges a millisecond past its budget hands back the optimum it computed
    instead of a report that nothing was solved. `TimeLimit` also joins each
    driver's reduced-accuracy salvage, so a cancelled solve whose last iterate
    satisfies the KKT conditions is still allowed to say so.

  Both rules apply on the LP / convex-QP **active-set** route as well as the
  IPM one: `solve_qp_active_set` shares the IPM's relabelling policy rather
  than stamping `TimeLimit` over whatever the inner solve returned, and a
  cancelled engine result now goes through the same verified KKT check as an
  iteration-limited one — reaching the answer and then running out of seconds
  used to be reported as reaching nothing, while running out of *iterations*
  at the identical point reported `Optimal`.

  A cancelled active-set solve also reports the wall-clock time it actually
  spent rather than zero. `pounce_convex::batch`'s `time_limit` is documented
  as per-instance, which is what it has always been.

- **`max_wall_time` now bounds the whole CLI solve, not each engine that gets
  a turn at it.** When a convex attempt declines the problem and hands it to
  the NLP path (the gh #535 LP→NLP reroute, or `socp_nlp_fallback`), the NLP
  solve built its `Deadline` from the option value — which still named the
  full budget — so a run that spent most of its seconds convex-side was
  granted them all again and the cap could buy nearly twice the wall clock it
  promises. The declined attempt is now charged against the option before the
  handover, the same deduction the convex path already applies internally for
  extraction and presolve.


### Added — `pyomo_pounce.estimate_report()` says what the estimate's step did about the bounds (#584)

`estimate()` takes the linear step, clips any variable it carries past a
bound, warns, and returns. The warning names those variables and stops
there, so a caller cannot tell how far along the perturbation the active
set changed, whether a constraint became active before any variable did,
or whether a gap against a re-solve comes from the barrier parameter,
from a regularized factor, or from relaxed bounds.

`estimate_report(model, perturb)` takes the same perturbation argument
and returns an `EstimateReport`:

    import pyomo_pounce
    pyomo_pounce.declare_sens_param(m.setpoint)
    pyo.SolverFactory("pounce").solve(m)

    r = pyomo_pounce.estimate_report(m, [(m.setpoint, 3.0)])
    r.alpha        # 0.0297, the fraction of the move that fits
    r.first        # 'u[0]', the control that saturates there
    r.crossed      # {u[0]: 4.22, u[1]: 0.68, ...}, what estimate() clamps
    r.violation    # 4.4e-16, the constraint violation at the prediction
    r.activity     # per variable: inactive, weakly_active, strongly_active

`estimate()` is unchanged. The classification is passed through from
`Solver.classify_activity`, the same classifier the covariance and
information accessors use, so this adds no second opinion about what is
active. Alongside it the report carries the three quantities that
separate the predictor from the exact value at the perturbed active
set: the barrier parameter `mu`, the factor's inertia-correction
`perturbations`, and `bounds_relaxed`.

A coordinate the full step carries past a bound always sets `alpha`,
off the same predicate and the same tolerance that put it in
`crossed`, so the two halves of the report cannot disagree about what
the step did.

Everything else on a bound is skipped, since the slack left there is
what the barrier leaves rather than room to move, and dividing it by a
step of the same size would become the minimum on any model carrying an
active bound. Because that exclusion only ever reaches coordinates that
do not cross, it cannot cost a crossing. The classification drives it
where the classifier commits, applied per side rather than per
coordinate, because a coordinate held at one bound can still be carried
across its other one. For the coordinates the classifier declines to
rule on, the size of the remaining gap decides: it is `O(mu)` at a
strongly active bound and `O(sqrt(mu))` at a weakly active one against
`O(1)` room in the interior, so the threshold scales with `sqrt(mu)`,
measured four orders of magnitude clear of the interior case, and is
capped because it is applied relative to the coordinate's own magnitude
and a loose `mu` at termination would otherwise widen it without limit.
The `ambiguous` verdict cannot decide this on its own, since it covers
both a coordinate on its bound and one near it with room left.

A solve that ran with a non-zero `bound_relax_factor` is reported
through `bounds_relaxed` rather than raised on. The classifier declines
such a solve, since relaxed bounds shift the slacks it reads, but
everything else in the report is still measured, and a caller reaches
for a diagnostic precisely when the estimate and a re-solve disagree.

Reading that condition needs `Solver.bound_relax_factor`, new on the
Python solver: the value the held solve actually ran under, or `None`
with no converged factor. `classify_activity` guards on it and raises,
so without the getter the only way to tell a relaxed solve from any
other bad-options failure is to provoke the guard and match the option
name in the message, which turns back into a re-raise the day that
message is reworded.

This is item 0 of the sensitivity roadmap (#255).

**Testing.** `pyomo-pounce/tests/test_estimate_report.py` checks the
step fraction against a brute-force scan of the unclamped step, on the
same coordinate exactly and to 1e-12 relative, including with a fixed
variable present, which the solve removes and which shifts every later
factor column. The violation is checked against direct evaluation at
the predicted point to 1e-12, hand-computed crossings to 1e-8 for both
a variable bound and a constraint, and the report is checked identical
through `SolverFactory('pounce')`, `SolverFactory('pounce_v2')` and the
contrib `SolverFactory('pounce')`. An invariant across every fixture
holds the two halves of the report consistent: a non-empty crossed set
means the reported fraction is below one. The ratio test is also tested
directly on constructed arrays, which is the only way to reach the
no-bound sentinel, a coordinate left outside its bound by a relaxed
solve, and the per-side exclusion at chosen values.

### Fixed — the no-session error named only the legacy solver route (#584)

`estimate`, `gradient`, `covariance` and `information` raise a
`RuntimeError` naming the declaration to make and the solve to run when
no sensitivity session exists. All four messages told the reader to
solve with `SolverFactory('pounce')`, which has been incomplete since
the v2 interface was added in 0.10.0: `SolverFactory('pounce_v2')` and
the contrib `SolverFactory('pounce')` build the same session, because
`Pounce.solve` sends a model carrying declarations down the same
in-process route. The messages now name all three.

### Fixed — publishing workflows no longer fail in forks (#599)

Syncing a fork failed three checks — `Deploy docs` and both architectures
of `release-docker`'s source-image build — every time, because both
workflows fired on a push to `main` and neither can succeed anywhere but
this repository: the fork has no Pages environment, and its `GITHUB_TOKEN`
cannot write to `ghcr.io/jkitchin/pounce`. Contributors got a failed run
and an email for work that was never theirs to publish.

The container images are now cut from releases only. `release-docker.yml`
has no `main` trigger at all: the source-built image was what that trigger
published, as `:edge` and `:sha-<short>`, and it existed to hand out an
image of an unreleased fix — which `make docker` builds locally in about
the time the pull would take. The image itself is unchanged and still
builds on every PR touching `docker/**`, so it cannot rot, and a manual
run of the workflow (`variant=source`, `dry_run=false`) still publishes one
when a bug report needs it. `:edge` and `:sha-*` already in the registry
stay where they are and no longer move; `docs/src/docker.md` says so rather
than promising a tag that tracks `main`.

Tags reach forks too, so trigger changes alone would not have covered the
`v*`, `python-v*` and `pyomo-pounce-v*` paths. All four publishing
workflows and the docs deploy now gate their first job on
`github.repository == 'jkitchin/pounce'`. Pull requests are unaffected —
`pull_request` runs in the base repository's context, so a contributor's PR
still gets the full CI and the Docker rot guard.

## [0.10.0] - 2026-08-11

### Fixed — an ill-scaled dense Hessian column no longer costs the solve its certificate

Dense-column peeling — added earlier in this same release, under "a single
dense Hessian row made `.nl` setup and every `eval_h` O(n²)" (#552) below, so
no published version ever carried the defect described here — gives a dense
Hessian column its own singleton color and then recovers every entry in that
column's *row* from the same pass, by symmetry. That is exact in real
arithmetic. In floating point the pass is accumulated at the scale of the
whole dense column, so each entry read out of it carries an absolute
roundoff floor of about `eps * ||H(:, d)||` where
the ordinary path — column `j`'s own pass — would have left about
`eps * |H[d, j]|`.

For a well-scaled dense column the two are the same to the last bit:
checked against a fully uncompressed reference Hessian, every peel-firing
model in the benchmark corpus but one recovers bit-identical values. The
exception is `cho_parmest`, a 12-parameter kinetic fit whose peeled columns
span from `2.8e5` down to `5.6e-4`; its small entries came back with a
relative error near `1e-5`. The primal solution absorbs that — `x` stayed
good to 13 digits — but the multipliers come out of the KKT system the
Hessian sits in, so `inf_du` picked up a jitter floor near `1e-6` and the
solve stalled at `Solved To Acceptable Level` on a problem POUNCE had
certified `Optimal` for its whole history.

Nothing structural separates the two cases — it is a property of the values
— so POUNCE now measures it. On the models that peel anything at all (56 of
2,014 in the corpus), one extra `eval_h` at `x0`, at the peeled color count,
leaves each peeled column's exact pass in hand; a column whose smallest
recovered entry would take more than a `1e-8` relative hit is dropped from
the peel set and colored the ordinary way. `cho_parmest` vetoes 7 of its 12
and is back to `Optimal` in 34 iterations at an overall NLP error of
`4.0308123324089062e-09` — bit-identical to the last release that had not
yet grown the optimization.

51 of the 56 veto nothing, so their coloring, seeds and decode tables are
untouched and their arithmetic is unchanged — including every model the
optimization was written for (`rocket_12800` 6.9×, `steering_12800` 6.4×,
`gasoil_3200`, `pinene_3200`, `robot_1600`, `marine_1600`). The remaining
four are `orth*` models that veto one column each and do not need to: the
test is a worst-case bound, and it is loose by an amount that varies per
column, so it is possible to bound high and measure exact. They still solve
`Optimal`, and pay 2–3.5× on solves of 0.04–0.33 s (worst case +0.32 s
absolute) for the recolor.

That tradeoff is deliberate rather than tuned away. Over the corpus the
highest bound on a column that recovers to machine precision is `2.9e-8`
and the lowest on one of `cho_parmest`'s harmful columns is `8.3e-8` — a
gap of 2.8×, measured on two model families. `1e-8` sits below both,
because the errors are asymmetric: a false veto costs one model a coloring,
a missed veto costs a solve its certificate.

### Changed — FERAL 0.15.1, which stops assuming the host has threads

The workspace pin moves `feral` 0.15.0 → 0.15.1. It is a patch release with
no API change and no source change anywhere in the workspace, but two of its
items are the upstream half of the `FERAL_PARALLEL` fix directly below.

**`Solver` derives `use_parallel` from the platform (feral#154).**
`Solver::new()` / `Solver::with_params()` previously hardcoded
`use_parallel = true` and built a rayon `ThreadPool` on every `factor()`.
That is wrong wherever the host has no usable threads, and POUNCE has such a
host: feral is in the `pounce-wasm` dependency tree, and on
`wasm32-wasip1` both `available_parallelism()` and `thread::spawn` report
`Unsupported`, while a threads-enabled wasm host whose worker pool has not
been stood up makes `build()` wait for workers that never arrive. The default
is now `available_parallelism() > 1`.

**A failed pool falls back to the sequential driver (feral#156).** When
`use_parallel` is on but the `Solver`-owned pool could not be built, `factor()`
(including the MC64 retry), `solve_refined` and `solve_many_refined` previously
ran the parallel driver with no `install` — i.e. on rayon's *global* pool,
defeating the per-`Solver` isolation exactly when nesting is least welcome
(compare the latent nested-rayon self-deadlock in feral#102). All four now run
the sequential driver. `Solver` also no longer initializes rayon's global
registry on any path. No numerical change: the two drivers carry a bit-exact
per-supernode contract.

Together these are why `FERAL_PARALLEL`'s force-on arm matters, and this bump
is what makes the comment at `crates/pounce-feral/src/lib.rs:424` true rather
than aspirational.

The release also carries three items we get for free:

- **MC64 scaling-cache gate fixed.** The value-bound gate that decides whether
  a cached MC64 scaling may be reused compared the current *minimum* scaled
  diagonal against the baseline *mean* — different statistics, so it behaved as
  an absolute property of the matrix rather than the drift measure it was meant
  to be, and rejected reuse on matrices that had not moved. The fix is a strict
  widening. Across 53 gate evaluations on the seven corpus families that route
  to MC64, exactly two decisions change, both on `robot_1600`: factorization
  time −14.3%, scaling time −39.8%. Inertia is unchanged on every iterate of
  every family, and a genuine diagonal collapse is still rejected.
- **MC64 Hungarian matching 4–5% faster** on large matchings (the min-heap now
  stores the key inline instead of a random access into the distance array per
  comparison). Verified bit-identical on 51 matrices across 39 families.
- **`Supernode.nrow` corrected after amalgamation.** `find_supernodes` used
  `col_counts[first_col].max(ncol)`, exact for a *fundamental* supernode but
  wrong after a size-based merge; undercounts reached 40%. Numeric factors and
  inertia are unaffected — `build_row_indices` always recomputed the true row
  set — but everything that *estimates* from `nrow` was working from a wrong
  number. The one edge that reaches us: `estimate_assembly_flops` rises 2–3×,
  so matrices sitting just under `PAR_MIN_FLOPS` can now route to the parallel
  driver where they previously fell to sequential. `FeralConfig.min_par_flops`
  defaults to `None`, so we inherit feral's threshold — which was itself
  calibrated against the understated estimate and may now sit in the wrong
  place. `feral_min_par_flops` / `POUNCE_FERAL_MIN_PAR_FLOPS` is the override
  if it needs re-tuning.

### Fixed — `FERAL_PARALLEL` is now bidirectional and speaks the same grammar as every other knob

`FERAL_PARALLEL` was the one boolean environment variable in
`crates/pounce-feral` that did not accept the `1|on|true|yes` /
`0|off|false|no` grammar the rest of the file uses. It parsed
`0|false|off` only, which left two gaps.

**No force-on at all.** `FERAL_PARALLEL=1` silently did nothing. That was
harmless while feral's internal parallelism was simply on by default, but
upstream feral#156 makes the default *platform-derived*, with a fallback
to sequential when the rayon pool fails to build. On a host where that
autodetection is wrong — threaded wasm is the motivating case — an
explicit force-on is the escape hatch, and POUNCE swallowed it. The
first-class lever, `FeralConfig.parallel`, is reachable only from the
Rust API: there is no `feral_parallel` OptionsList option, so CLI,
Python, NL and GAMS callers had no override whatsoever. (feral#156 is
merged upstream but not in the pinned 0.15.0, so this is a fix landing
ahead of the pin that needs it rather than a live regression.)

**`no` was missing from the off list**, even though feral's own C-ABI
shim accepts it (`feral/src/capi.rs`) — so `FERAL_PARALLEL=no` disabled
the parallel driver when a caller went through the C API and was ignored
when the same caller went through POUNCE.

The grammar is now a single pure `parse_bool_env` helper that
`POUNCE_FERAL_CASCADE_BREAK`, `POUNCE_FERAL_FMA`, `POUNCE_FERAL_REFINE`,
`POUNCE_FERAL_STATIC_PIVOTING` and `FERAL_PARALLEL` all share, so the
vocabulary cannot drift per knob again. Unset and unrecognized values
still mean "leave the default alone" rather than a silent `false`.
Behaviour of the four `POUNCE_FERAL_*` knobs is unchanged — they already
had the full grammar; only `FERAL_PARALLEL` gains tokens.

### Added — a Rust chapter in the book, and the SQP warm-start contract on the facade (#561 follow-ups)

Two gaps left open by #561, both about the Rust surface being reachable but
undocumented.

**The book had no Rust chapter.** It has never had one — the Rust snippets
lived inside the per-solver pages, so a Rust user had no entry point
equivalent to `docs/src/python.md`. `docs/src/rust.md` is now that page:
install, the `Problem` + `Nlp` builder, the `TNLP` trait for exact
Hessians / custom sparsity / scaling, iteration capture, the feature-flag
table, and a worked snippet per feature module. Linked from `SUMMARY.md`
under Integrations, next to the Python API.

**`docs/src/active-set-sqp.md` had Python / C / GAMS but no Rust**, in both
§2 (switching to the SQP path) and §3 (carrying a working set across solves).
Both now have a Rust section. Writing §3 turned up a real hole rather than a
prose one: the round trip needs `SqpIterates` and `classify_working_set`,
which lived in `pounce-algorithm`'s internals and were reachable only as
`pounce_rs::pounce_algorithm::sqp::…`. A new Rust chapter telling readers to
spell that would have re-created exactly the coupling #561 removed, so the
facade grew a curated **`pounce_rs::sqp`** (feature `qp`): `SqpIterates`,
`classify_working_set`, `SqpOptions` / `SqpGlobalization` /
`SqpHessianSource`, and the `WorkingSet` / `BoundStatus` / `ConsStatus`
status types.

The split is worth stating because it decides which feature a reader needs:
**flipping `algorithm` to `active-set-sqp` needs no feature at all** — it is
one option string on the default build — while *carrying a working set* needs
`features = ["qp"]`, because `WorkingSet` is the QP engine's type.

`tests/sqp_surface.rs` pins the round trip through the facade alone
(`last_sqp_working_set` → `SqpIterates` → `set_sqp_warm_start`, same answer
warm as cold), that `classify_working_set` derives a seed from a predicted
point, and that the algorithm flip needs none of those types. Nothing in CI
compiles book snippets, so every runnable snippet in `rust.md` — and every
API name its prose mentions — was compile-checked against
`pounce-rs --all-features` before landing.

Also swept up: the user-facing prose references #561 missed, because that
pass fixed `use` lines only. `docs/src/sessions.md`'s "which layer do I want"
table and its verification list, plus `active-set-sqp.md` §4, still pointed
at `pounce_sensitivity::` / `pounce_linsol::` paths; all now name the facade.

### Added — `pounce-rs` covers the convex, active-set QP, and sensitivity paths (#561)

The `pounce-rs` facade exists so Rust users depend on one crate and are
insulated from churn in the internal crate layout (#168) — but its
dependencies were `pounce-common` + `pounce-nlp` + `pounce-algorithm` +
`pounce-observability`, which is the NLP path and nothing else. Anything past
a single cold NLP solve — batched QPs, parametric sweeps, sensitivities —
meant depending directly on the internal crates the facade exists to hide,
which is exactly the coupling it was introduced to remove. The Python side
never had this gap: `import pounce` reaches all of it.

Three feature-gated modules close it. The default build is unchanged:

| feature | module | covers |
|---|---|---|
| `convex` | `pounce_rs::convex` | LP, convex QP, SOCP / exponential / power / PSD cones, SOS; `solve_qp_batch{,_parallel,_parallel_warm}`, `solve_qp_multi_rhs*`, `QpFactorization` symbolic reuse; `QpSensitivity` / `ReducedHessian` |
| `qp` | `pounce_rs::qp` | sparse parametric active-set QP: `ParametricActiveSetSolver`, `QpSolver::solve_parametric`, `WorkingSet` warm starts, the `.qps` reader |
| `sensitivity` | `pounce_rs::sensitivity` | `SensSolve` / `SensResult`, `compute_reduced_hessian`, and the long-form `SensApplication` / Schur stack |
| `full` | — | all three |

Separate modules rather than a flat re-export because `pounce-convex` and
`pounce-qp` are different solver families that both name their types
`QpProblem` / `QpSolution` / `QpStatus` / `QpOptions` / `QpWarmStart`; one
flat surface cannot carry both.

Two things beyond plain re-exports were needed for the modules to be usable
from `pounce-rs` alone — without them a caller would still have had to name
the internal crates, and the gap would have been closed only on paper:

* **`pounce_rs::linsol`** (on with `convex` or `qp`). Both QP entry points are
  parameterized over the factorization backend — `solve_qp_ipm` takes a
  `FnMut() -> Box<dyn SparseSymLinearSolverInterface>`,
  `ParametricActiveSetSolver::new` takes a boxed backend — so the solvers
  alone are not callable. `backend()` and `serial_backend()` are the two
  factories every in-tree caller writes by hand (parallel FERAL, and the
  inner-serial one that keeps `solve_qp_batch_parallel` from
  oversubscribing).
* **the triplet storage** (`SymTMatrix` / `GenTMatrix` and their spaces),
  re-exported through `pounce_rs::qp`, since `pounce_qp::QpProblem` *borrows*
  its Hessian and Jacobian in that form.

Enabling a feature widens the public surface, not the build: the default NLP
path already pulls `pounce-qp`, `pounce-linsol`, and `pounce-feral`
transitively, so only `convex` and `sensitivity` add crates to compile.

The book's Rust snippets moved onto the facade too. `docs/src/sensitivity.md`,
`docs/src/sessions.md`, and `docs/src/global-optimization.md` were telling
readers to import `pounce_sensitivity` / `pounce_convex` / `pounce_linsol` /
`pounce_feral` directly — the exact coupling this change removes, printed as
the recommended way — and the SOS example hand-rolled the very backend factory
`pounce_rs::linsol::backend()` now supplies. Each snippet names the feature it
needs. The `pounce-qp` listings in `docs/src/active-set-sqp-warm-start.md` are
left alone deliberately: they are annotated with the source path they quote
(`// crates/pounce-qp/src/problem.rs`) and show that crate's own definitions,
not user-facing imports.

Each module is covered by an integration test that imports **only**
`pounce_rs` — the property the issue is about, mechanically checked rather
than asserted in prose. The sensitivity test pins the same upstream sIPOPT
3.14.19 `parametric_cpp` golden Δx that `pounce-sensitivity`'s own tests do,
to 1e-8. CI grew a `cargo clippy + cargo test -p pounce-rs --all-features`
step, because the workspace build compiles default features only and would
never have touched the new modules; `[package.metadata.docs.rs] all-features`
keeps the published docs from showing the NLP path alone.

### Added — `pyomo-pounce` registers a native `pyomo.contrib.solver` (v2) interface (#558)

`pyomo_pounce` previously registered one solver: `POUNCE`, a subclass of
Pyomo's *legacy* `ASL` plugin. #553 made the POUNCE binary drivable by
Pyomo's newer `pyomo.contrib.solver` interface, but that is not the same
as having one — a user had to point `ipopt_v2` at the executable by hand,
and doing so silently dropped everything `pyomo-pounce` adds.

`pyomo_pounce.v2.Pounce` is now registered alongside the legacy plugin:

    import pyomo_pounce
    from pyomo.contrib.solver.common.factory import SolverFactory as SF2
    SF2('pounce')                        # v2 interface
    SolverFactory('pounce_v2')           # v2 engine, legacy API
    SolverFactory('pounce')              # legacy plugin, unchanged

The split follows Pyomo's own `ipopt` / `ipopt_v2` convention, so the
registration is purely additive: `SolverFactory('pounce')` still returns
the legacy plugin and behaves exactly as before.

**The extras come with it.** That is the point of the class, and each
needed translating onto the v2 lifecycle rather than copying:

- the guard that refuses a model with live integer variables instead of
  solving the continuous relaxation and reporting a fractional value as
  optimal (#341);
- `scaling_factor` Suffix handling (#483 / #486);
- bundled-binary resolution, which keeps a stale `pounce` on `PATH` from
  being picked up silently (#315);
- the sensitivity path (`declare_sens_param` → in-process
  `pounce.Solver`). This one is a real translation, not an adaptation:
  v2 returns a `Results` and hands the solution back through a solution
  loader the caller may decline to load, where the legacy path writes
  values onto the model as a side effect of solving. The new
  `PounceSensSolutionLoader` serves primals, duals and reduced costs out
  of the in-process solve, converting the engine's internal multipliers
  to the AMPL marginal convention `dual` carries and to Ipopt's
  `ipopt_zU_out` sign — the same two conversions the warm-start reader
  applies in the other direction.

**Inherited deliberately** from Pyomo's `Ipopt` v2 class: the `.nl`
write, the `.sol` read, option splitting between the command line and the
`.opt` file, and the solver-log parse. POUNCE is ASL/Ipopt-compatible on
all of them, including the `Number of Iterations....:` and
`Total seconds in POUNCE =` lines the log parser reads. The one place it
is not is the version banner — `pounce --version` prints `pounce X.Y.Z`,
and Pyomo's Ipopt parser requires a leading `ipopt` by design, so that
other ASL executables are not mistaken for Ipopt. Without the override
here, `version()` is `None` and `available()` reports `NotFound`. Solves
still run (nothing gates on the version), but every availability check
lies — including the `if not solver.available(): skip` pattern that
guards most Pyomo test suites, and Pyomo tooling that selects a solver by
availability.

**Why it matters beyond API parity.** #552 measured the same POUNCE
binary through both interfaces on drto/IDAES collocation models: 0.553 s
of non-solve time per solve through the legacy path against 0.301 s
through v2, ~1.8×. (On a plain `pyomo.dae` model the same comparison is
1.05×, so this is model-shape dependent.) Since `SolverFactory('pounce')`
is the documented route, users of that model class were paying the legacy
path's cost with no supported alternative that kept the extras.

**Requirements, and only for the v2 route** (`pip install
pyomo-pounce[pyomo-v2]`): **Pyomo ≥ 6.10.1**, where the `SolutionLoader` /
`get_vars` API this builds on landed — `pyomo.contrib.solver.common`
exists from 6.9.2, but 6.9.2–6.10.0 ship the older `SolutionLoaderBase` /
`get_primals` — and **pounce-solver > 0.9.0**, because Pyomo's
`asl_sol_reader` asserts `n_opts >= 2` where the legacy reader is lenient
and so needs the per-model `.sol` `Options` echo added after 0.9.0.
Neither applies to `SolverFactory('pounce')`: on an older Pyomo the
legacy plugin behaves exactly as before and
`pyomo_pounce.HAVE_V2_INTERFACE` reports `False`.

**Testing.** `pyomo-pounce/tests/test_v2.py` solves the same model
through both interfaces and compares primals, objective, duals and
reduced costs, and does the same for the ordinary route against the
sensitivity route — including that they agree on `solution_status` for a
limit-stopped solve, which decides whether `solve()` raises. That closes
the gap noted in #558: the interface table in `docs/src/pyomo.md` was
produced by hand and nothing in CI re-checked it, and the #553
ASL-compatibility fixes (quoted option values, the per-model `.sol`
`Options` echo) were pinned by Rust unit tests only — the v2 route
exercises both on every solve. A CI leg pinned to a Pyomo *below* the
floor checks that the legacy plugin still imports there, so the floor is
a tested property rather than a claim.

### Changed — FERAL 0.15.0, which fixes the parallel-driver task granularity behind the #552 factorization gap

The workspace pin moves `feral` 0.14.0 → 0.15.0. The change that matters for
POUNCE is upstream feral#148: the parallel multifrontal driver spawned one
boxed rayon task *per supernode* — roughly 1.8M allocations per solve on a
chain-structured problem — so a chain-shaped KKT paid allocator and scheduling
cost for a tree that offers no parallelism. It now spawns one task per subtree,
and a chain-shaped tree collapses to a single task and takes the sequential
driver outright.

This lands on the path POUNCE actually ships: `FeralConfig.parallel` defaults
to `None` and is only disabled by an explicit `FERAL_PARALLEL=0`
(`crates/pounce-feral/src/lib.rs:399`), so every default solve was on the
parallel driver. Paired A/B on the six KKT matrices dumped from real solves
(alternating process launches, 12 pairs, `min` of 5 warm factorizations):

| matrix | 0.14.0 | 0.15.0 | speedup | wins | p |
|---|---:|---:|---:|---:|---:|
| clnlbeam | 53.07 ms | 23.18 ms | **2.29×** | 12/12 | 0.0005 |
| steering_12800 | 39.11 | 22.59 | **1.73×** | 12/12 | 0.0005 |
| dtoc2 | 112.42 | 96.70 | 1.16× | 12/12 | 0.0005 |
| rocket_12800 | 18.11 | 15.65 | 1.16× | 12/12 | 0.0005 |
| dtoc1nd | 13.73 | 12.02 | 1.14× | 11/12 | 0.006 |
| marine_1600 | 29.10 | 28.73 | 1.01× | 8/12 | 0.39 |

The release also makes the x86_64 `pulp` dispatch actually vectorize — every
pulp kernel had been executing its lane operations as outlined calls, roughly a
10× kernel slowdown, invisible on aarch64 where NEON is baseline — and turns
the packed BLAS-3 trailing update into an explicit SIMD kernel.

All of it is scheduling- and codegen-only. Factors are byte-identical, pinned
upstream by `tests/task_plan_parity.rs` and hardcoded `tests/golden_bits.rs`
digests, and confirmed here by identical solve-vector hashes across both
versions on all six matrices. 0.15.0's one breaking change is diagnostic-only
(`SupernodeTiming` / `BucketStats` / `ProfileReport` moved from `*_us` fields
to `*_ns`, with `.us()` accessors kept); POUNCE consumes none of that API, so
no source change was needed anywhere in the workspace.

This narrows but does not close the gap reported in #552. It is also the
counterpart to the #562 change below: that one removes a POUNCE-side cost on
the same code path.

### Changed — the FERAL backend stops rebuilding the CSC matrix on every factorization (#562)

`pounce-feral` handed FERAL a `CscMatrix` built by
`CscMatrix::from_triplets` on **every** `factor()` call, even though
`initialize_structure` fixes the sparsity pattern and only the values
change between IPM iterations. Each rebuild re-ran an identical
bucket-count / place / sort / sum-duplicates pass to arrive at the same
`col_ptr` and `row_idx` it had computed the iteration before — and that
pass allocates a fresh `Vec<(usize, f64)>` *per column* and sorts it, so
on a 100k-column KKT it is ~100k short-lived allocations and ~100k sorts
to reproduce a permutation that was already known. MA57 has no equivalent
step (its triplet is consumed in place by `ma57bd_`), so this was pure
overhead in the FERAL arm of every A/B comparison.

The first factorization after an `initialize_structure` still goes
through `from_triplets`, and now also records the triplet → CSC slot
permutation. Later factorizations zero the retained matrix's values and
scatter the caller's triplets through that permutation — one O(nnz) pass
with no allocation. The scatter accumulates with `+=` so duplicate
triplets landing in one slot are still summed, and because it reuses the
slots the first build created, the explicit structural zeros on the
(2,2) multiplier diagonal survive a refill (dropping them is the
FERAL-side cascade that the "do not strip zeros" rule exists to avoid).
`initialize_structure` — the only place the pattern can change —
invalidates the permutation and the matrix.

On a clnlbeam-shaped KKT (n = 99,999, nnz = 299,994) the rearrangement
drops from ~4.4 ms to ~0.34 ms per factorization, a 13× reduction, and
the saving grows with the matrix: the issue measures 16.5 ms of rebuild
per factorization on `dtoc2`. Under `debug_assertions` each refill is
cross-checked against a fresh `from_triplets` — exact on `col_ptr` and
`row_idx`, to a relative tolerance on the values — which doubles as the
assertion that nothing mutates the pattern behind
`initialize_structure`'s back.

### Changed — `eval_h` routes through the shared-CSE tape, amortizing prelude second-order work across summands (#557)

The follow-on to #476 (`eval_g`) and #553 (`eval_jac_g`): the Lagrangian
Hessian — the largest of the three evaluators by a wide margin, 66–69% of
AD time on shared-CSE chain models — now takes the `HybridTape` path too,
above its own measured op-ratio gate.

Unlike the Jacobian, the Hessian can share **both** second-order sweeps of
the prelude, which is why its crossover sits lower. The coloring hands
every summand of a color the same seed vector `s_c`, so per color the
prelude forward tangent runs once for the whole constraint block; and
because reverse-over-tangent is linear in its adjoint seeds, each summand
folds its row multiplier `λ_k` into the adjoints it deposits at `Shared`
boundaries, and one unit-weight prelude reverse sweep per color replaces
the per-summand sweeps. The per-summand local ops keep the exact
arithmetic of the flat tape (`ror_dir_step` mirrors
`Tape::hessian_directional`'s arms, writing into the dense per-color
`compressed` buffer — no hashing), so the local contributions are
bit-identical; the folded prelude sweep is mathematically identical and
agrees to rounding, which the tests pin as exact equality on
all-dyadic-arithmetic models and a few-ULP band on transcendental ones.

Measured on chain models with CSE redundancy 40, varying the shared body
size (`eval_h`, flat → hybrid, n ≈ 21,000, m = 20,000). Median of 5
interleaved flat/hybrid pairs per point, with the observed range:

| flat/shared op ratio | 1.94 | 2.54 | 3.12 | 3.69 | 4.24 | 5.29 | 6.76 | 8.53 |
|---|---|---|---|---|---|---|---|---|
| median speedup | 1.00× | 1.04× | 1.18× | 1.23× | 1.31× | 1.44× | 1.36× | 1.49× |
| min–max | 0.91–1.11 | 0.36–1.14 | 1.10–1.33 | 1.16–1.30 | 1.23–1.32 | 1.19–1.54 | 1.27–1.50 | 1.36–1.65 |

The medians are what the threshold is set from: single runs on a shared
machine scatter too widely to place a gate on — the 2.54 column spans
0.36× to 1.14×. The gate (`HYBRID_HESS_MIN_OP_RATIO`) sits at 3.0, the
lowest ratio where every sample wins by at least 10%; below it `eval_h`
stays on the flat path bit-identically, so a gate set high is merely
conservative while one set low risks a real regression. With all default
gates the whole AD stack at ratio 4.24 speeds up 1.25× (median of 5
interleaved pairs, range 1.14–1.44). Models without shared CSE bodies —
including everything Pyomo writes, which has no `V` segments — never
build the hybrid tape and are untouched.

The two prelude sweeps run once per Hessian color, so they walk the union
of that color's summands' `prelude_reach` rather than the whole prelude.
Walking all of it would cost `n_colors × |prelude|` where the op-ratio
gate assumes `|prelude|` — a discrepancy the gate cannot see, unbounded
in `n_colors`. On the chain models above this is neutral (every color
reaches nearly every body there, so there is nothing to skip: 1.02× at
ratio 4.24, 1.12× at 8.53); it is insurance for models where a color
reaches only part of the prelude, which a unit test on two shared bodies
of differing width exercises directly.

The dormant `HybridTape::hessian_summand` is **removed**. It was the
wrong tool for this and had never had a caller: it re-walked
`prelude_reach` once per seed variable (sharing no prelude work at all)
and scattered through a `HashMap` lookup per emitted pair. Its
replacements are `prelude_tangent` / `hessian_summand_directional` /
`prelude_reverse_directional`.

Instrumentation and coverage: `POUNCE_DBG_FORCE_HYBRID_HESS=1` forces the
Hessian gate on (the crossover table's measurement instrument, paired
with the existing `POUNCE_DBG_NO_HYBRID=1` flat forcing);
`POUNCE_DBG_TAPE_STATS=1` now prints both gates' on/off state. The
shared-CSE paths also gain their first end-to-end coverage — no
repository fixture had a CSE shared across constraints, all 62 checked —
via a generated model whose one defined variable feeds 16 rows, solved
through the CLI both hybrid and flat to the same analytic optimum.

### Fixed — a single dense Hessian row made `.nl` setup and every `eval_h` O(n²) (#552)

`NlTnlp` recovers the Lagrangian Hessian by graph coloring: columns whose
nonzero rows are pairwise disjoint share a color, and one directional
product `H·s_c` per color recovers all of them at once. The seed and
result buffers are `n_colors × n` dense arrays, which is fine while the
coloring works.

It does not work when the Hessian has one **dense row**. A variable that
multiplies a sum over all the others — a total-cost variable, a shared
design parameter, a scalar entering every constraint — puts a nonzero in
*every* column at its own row, so no two columns may share a color and
the greedy walk hands out one color per variable. `seeds` and
`compressed` then become O(n²), on a model whose Hessian is otherwise
perfectly sparse.

A second, independent O(n²) sat behind it in `Tape::hessian_sparsity`,
which kept a `BTreeSet` of dependent variables alive for *every* tape
slot. A long additive chain — exactly what a dense row is built from —
left one full-size set per partial sum and copied the whole running set
at each union. (`split_top_sums` keeps top-level sums out of a single
tape, so this only bit when the sum fed an enclosing operator.)

**What is new.** Columns whose nonzero-row count exceeds
`DENSE_COL_FACTOR` (16) times the average — never fewer than
`DENSE_COL_MIN` (32) entries, at most `MAX_PEELED_COLS` (256) of them —
become *candidates* for peeling, and those that pay for themselves are
peeled out of the coloring and given a singleton color each. One
product with seed `e_d` recovers the whole of column `d`, and because
`H[d, j] == H[j, d]` that same pass also supplies every entry in row `d`.
Row `d` therefore stops constraining anyone else's color and drops out of
the conflict structure, leaving the rest to color on their real sparsity.
`hessian_sparsity` additionally computes slot lifetimes up front and
moves a dying operand's set into its consumer instead of copying it.

Measured on `min Σ(xⱼ−1)² + x₀·Σxⱼ` discretized alongside a sparse
constraint chain — one dense row, `nnz_h ≈ 2n`:

| n = 8,010 | before | after |
|---|---|---|
| colors | 8,010 | 6 |
| peak RSS through setup | 1,038 MB | 59 MB |
| setup wall clock | 2.69 s | 0.18 s |

| n = 4,010, 10 iterations | before | after |
|---|---|---|
| `LagrangianHessianEvaluations` | 2.069 s | 0.019 s |
| `OverallAlgorithm` | 2.385 s | 0.284 s |

Memory is now linear in `n` where it was quadratic, so what used to
extrapolate to ~6.4 GB at n = 20,000 (and an OOM at flowsheet sizes) is a
few tens of MB. Models without a dense row are untouched: a banded
Hessian peels nothing and colors by its bandwidth exactly as before, and
every one of the repository's 62 `.nl` fixtures returns a bit-identical
objective and exit status.

**Which candidates actually get peeled.** Each peeled column costs
exactly one color, so peeling only pays when it removes more conflict
than it adds — and simply truncating an over-long candidate list to
`MAX_PEELED_COLS` does not bound the damage. The columns that miss the
cut stay in the conflict structure, so the base color count is unchanged
and the singleton colors are pure addition. On disjoint 50x50 blocks
scattered through a 200,000-variable Hessian that colors to `50 + 256`
where a plain walk needs 50 — a 6x regression in the very quantity
peeling exists to reduce, and 8.5x on 34x34 blocks.

The candidate set is therefore chosen by estimating the result: for
peeling nothing and for the top `k` candidates by degree over a doubling
ladder, `|peeled| + max surviving row degree` is a lower bound on the
resulting color count (a row with `d` surviving entries makes those `d`
columns pairwise conflict), computable in one O(nnz) pass. The best
scoring option wins, ties going to fewer peels. On the block patterns
above every `k > 0` now scores strictly worse than peeling nothing, so
nothing is peeled and the coloring matches the plain walk exactly; the
one-dense-row case still collapses n = 20,000 to 2 colors. Note the plain
coloring cannot serve as a comparison baseline here — on that same
one-dense-row case the plain walk is itself O(n²), which is the blowup
peeling exists to avoid — so the choice has to be made from a bound.

### Fixed — Pyomo's modern solver interface could not drive POUNCE at all (#552)

`SolverFactory('ipopt_v2', executable=<pounce>)` — the v2 interface that
is becoming Pyomo's default `ipopt` — failed on every model. Two
independent gaps, both places where POUNCE diverged from what the ASL
does and every reader therefore expects:

1. **Quoted option values.** The v2 interface builds each option as
   `option_file_name="/tmp/…/x.opt"` and passes it as a single `argv`
   entry. With no shell in between, the quotes arrive as literal
   characters. Ipopt's ASL option parser strips them; POUNCE did not, so
   it looked for a file whose name began with `"`, failed to load it, and
   aborted the run. `key=value` now drops one matching pair of surrounding
   `"` or `'` quotes. A quote on one side only is left as content.

2. **The `.sol` `Options` block.** POUNCE wrote a count of `0`. AMPL and
   Pyomo's *legacy* reader accept that, but no ASL solver emits it: the
   v2 reader reads the first two option words unconditionally (to detect
   the documented quirk where a second word of `3` means two extra words
   follow the `z` block) and so asserts `n_opts >= 2`. A POUNCE `.sol`
   could not be parsed through the modern interface at all.

   POUNCE now echoes the model's own option words, which is what Ipopt
   and the ASL's `writesol.c` do — they are AMPL's flags, passed through
   rather than interpreted, and they are per-model: `.nl` header line 0
   is `g<count> <opt0> …`, so `bearing_400` (`g3 1 1 0`) gets
   `3 / 1 / 1 / 0` while `arki0003` and `camshape_6400` (`g3 10 1 0`) get
   `3 / 10 / 1 / 0`. Verified byte-for-byte against Ipopt 3.14.20 `-AMPL`
   output on all three. Where there is no originating header — a problem
   built through `NlProblem::from_expressions`, the WASM entry point, or
   a header whose declared count does not match the words present — a
   generic `3 / 1 / 1 / 0` goes out instead, which satisfies the v2
   reader without claiming to be the model's own.

With both fixed, the same model solved through `SolverFactory('pounce')`
and through `SolverFactory('ipopt_v2', executable=<pounce>)` returns
identical objectives and identical primal values. This also makes it
possible to benchmark POUNCE against Ipopt through *one* Pyomo interface,
which #552's overhead comparison could not do.

### Changed — `eval_jac_g` takes the shared-CSE path when the shared bodies are big enough to pay for it (#476)

`HybridTape` — the constraint tape that evaluates a CSE body once for the
whole block instead of once per referencing summand — has driven `eval_g`
since #476, while `eval_jac_g` stayed on the flat per-summand tapes whose
CSE bodies are duplicated. The Jacobian now uses it too, but only above a
measured threshold, because unlike `eval_g` it is not a free win.

The asymmetry is structural. `eval_g` needs only values, so the prelude
is swept once for the entire constraint block and the saving is the full
op-count ratio. The Jacobian needs a gradient *per row*, so only the
forward sweep can be shared — the reverse sweep still walks each
summand's `prelude_reach` separately, and it pays a per-op cost the flat
tape does not: a nested `SummandOp` dispatch and an indirected walk over
a reach list rather than a straight loop over a contiguous `Vec<TapeOp>`.

Measured on chain models with CSE redundancy 40, varying the shared body
size (`eval_jac_g`, flat → hybrid, n ≈ 20,000):

| flat/shared op ratio | 1.94 | 2.20 | 2.84 | 3.53 | 4.20 | 5.16 | 6.35 | 8.00 |
|---|---|---|---|---|---|---|---|---|
| speedup | 0.77× | 0.63× | 0.88× | 1.21× | 1.21× | 1.18× | 1.50× | 1.32× |

The crossover sits near 3, so the gate is set at 4 to keep a margin: a
model that does not clearly benefit stays on the flat path bit-for-bit.
With the gate in place the same sweep shows 1.00–1.03× below it (i.e.
unchanged) and 1.13–1.30× above it. `robot_a` (#476) measures 4.03×.

`eval_h` is **not** changed and remains the larger prize — it is ~69% of
AD time on these models against the Jacobian's ~19%. The dormant
`HybridTape::hessian_summand` is not the tool for it: it re-walks
`prelude_reach` once per seed variable, so it shares no prelude work at
all, and it scatters through a `HashMap` lookup per emitted pair where
the coloring path writes into a dense buffer. Amortizing prelude
second-order work across summands needs a new directional routine, which
is its own piece of work.

`POUNCE_DBG_TAPE_STATS=1` now also prints the flat-versus-shared op counts
for the constraint block, and `POUNCE_DBG_NO_HYBRID=1` forces the flat
path, so the trade above is reproducible on any model.

### Changed — the `.nl` reader's parse and setup paths allocate far less (#552)

Reading a `.nl` put two heap allocations on every data line: one for the
`String` the reader handed over, another for the `Vec<&str>` that
`parse_var_coef` / `parse_bound_line` collected from it. `parse_expr`
allocated a `String` per expression token — one per node of every
expression tree in the file, ~620k of them for a 20k-variable model. All
of these now borrow from the source buffer. Setup also replaced the
global Hessian-pair `BTreeSet`, the per-row Jacobian column sets, and the
per-summand color sets with `Vec` + `sort_unstable` + `dedup`, and
dropped a `HashMap` that was built with one entry per Hessian nonzero,
read once, and discarded — the decode tables are now built straight from
the sorted pair list, which also makes the `eval_h` scatter walk `values`
forward instead of in `HashMap` order.

This is a constant-factor cleanup, not a complexity change: total setup
on ordinary sparse models moves 1.00–1.07×. The dense-row case above is
where the compounding shows up.

### Fixed — the filter's `theta_max` ceiling is now raised on demand instead of blocking a solve forever (#476, #546)

The filter rejects any trial iterate whose constraint violation exceeds

```
theta_max = theta_max_fact · max(1, θ₀)          (Wächter–Biegler Eqn. 21)
```

outright, before its usual tests run. The `1` in that `max` is
dimensionally wrong for POUNCE: `θ` is a **1-norm over constraint rows**
(`‖c‖₁ + ‖d − s‖₁`), a *sum* of `m` residuals, so a fixed ceiling means a
mean per-row allowance of `theta_max/m` that shrinks as the model grows.
On a model started at a **feasible** point (`θ₀ = 0`) the `max` collapses
entirely and the ceiling is the bare constant `1e4` however large the
model is. `robot_a` (52 013 rows, feasible start) has to pass through
`θ ≈ 9.4e7` to reach its optimum, so every productive step was refused at
the gate and the solve ran to its iteration limit at objective `14.23`.

**What is new: `theta_max_adaptive_trigger` (default `3`).** POUNCE now
measures whether the gate is what is refusing the line search instead of
guessing from problem size. A trial refused because `θ_trial > theta_max`
takes a distinct early exit, so the acceptor counts those and compares
against the trials attempted. When **every** trial of a line search was
refused at the gate, for this many consecutive line searches, the ceiling
is multiplied by `theta_max_adaptive_factor` (default `100`), at most
`theta_max_adaptive_max_raises` times per solve (default `4`).

| model | before | after |
|---|---|---|
| `robot_a` | `Maximum_Iterations_Exceeded`, 14.23 | **Optimal**, 1.0432009, 190 it |
| `robot_b` | max time, 15.484684 | **Optimal**, 2.3330990, 269 it |
| `robot_c` | max time, 29.039906 | **Optimal**, 1.4059756, 222 it |

**Nothing that was converging changes.** A converging model is accepting
steps and therefore not being refused at the gate, so it never accumulates
the streak. `brainpc1/3/5/7` and `bt4` — the models that #545's static
row-count floor regressed at every `kappa` tried — are bit-for-bit
unchanged (64 / 43 / 982 / 43 iterations; `bt4` 9 iterations at
`−3.7047681836394486`). That is a property of the design rather than of a
chosen constant, which is exactly what the static floor could not offer:
it asked "does this model have many rows?", a proxy, and a kappa scan
showed the proxy was wrong with no separating value.

Requiring a *streak* rather than one line search is deliberate — a single
overshooting Newton direction can legitimately have all of its trials
refused, and backtracking is the correct response. The raise cap is what
keeps `theta_max` finite, which is what Wächter–Biegler's global
convergence argument (Thm. 2) needs; it needs the ceiling finite, not
fixed, so a solve cannot ratchet the safeguard away one line search at a
time.

Set `theta_max_adaptive_trigger = 0` for upstream Ipopt's fixed ceiling
exactly. The restoration sub-IPM always runs with the rule disabled, since
upstream already corrects the resto phase's instance of this degeneracy by
hard-coding `resto.theta_max_fact = 1e8` (`IpRestoMinC_1Nrm.cpp:91`).

`theta_max_row_scale_kappa` (#545) is unchanged and still defaults to `0`.

**Full Vanderbei sweep (733 problems), against `main` on the same
machine: 702 → 703 `Optimal`, and four problems move.**

| model | main | with the rule |
|---|---|---|
| `britgas` | `Maximum_Iterations_Exceeded`, 3000 it, 13943.55 | **`Optimal`, 16 it, −1.59e-07** |
| `catenary` | `Optimal`, 56 it | `Optimal`, **50 it**, same objective to all 16 digits |
| `coshfun` | `Error_In_Step_Computation`, 677 it | `Diverging_Iterates`, 1022 it, −6.6e10 |
| `brainpc0` | max_iter, 0.37934 | max_iter, 0.35545 |

`coshfun` is not a regression: it fails either way, and Ipopt 3.14 runs it
to `Maximum_Iterations_Exceeded` at −1.17e11, so "iterates diverging,
problem might be unbounded" is the more accurate diagnosis of the same
underlying behaviour. `brainpc0` fails either way too, at a better
objective. `catenary` matches Ipopt's answer exactly
(`−348403.1570810291`) in both columns.

`drcav3lq` and `drcavty3` also show small iteration-count differences, but
the rule never fires on either (verified directly) — both are
CPU-time-limited, so that is wall-clock noise, not a behaviour change.

Note `brainpc0` *is* gate-blocked and does trip the rule, while
`brainpc1/3/5/7` do not. That is the distinction the static row-count
floor could not draw: all five have the same row count.



### Added — per-variable scaling reaches the solver

- `nlp_scaling_method=user-scaling` now applies per-variable
  `scaling_factor` entries, closing the last of the three factor kinds
  (gh #486 stage 2, after #485 delivered the objective and constraint
  halves). A new `ScalingTnlp` wrapper substitutes variables one level
  below the algorithm, mirroring how `PresolveTnlp` transforms
  coordinates and inverts the transform at `finalize_solution`. The
  substitution is diagonal, so sparsity is untouched and every
  transform is an elementwise multiply or divide: bounds and the
  starting point scale, the objective gradient and Jacobian columns
  divide, Hessian entries divide by both their row's and column's
  factor, and on the way out the solution divides back while bound
  multipliers multiply, with constraint multipliers unchanged.
- Installed at `optimize_tnlp`, the single entry point every path
  funnels through, so the CLI, the Python bindings, and the batch
  solvers all get it from one place. Only `user-scaling` consults the
  problem for factors, so no other scaling method is affected and an
  unscaled solve pays nothing.
- Consumers that read the algorithm's own iterate rather than the
  `finalize_solution` payload — the CLI's `on_converged` hook, which
  feeds the `.sol` file, the JSON report, and the bound-multiplier
  suffixes — now undo the substitution explicitly, so every output
  carries the model's units. A caught defect, not a hypothetical: the
  `.sol` reported the scaled iterate before this.
- Factors must be positive and finite. A negative factor reverses a
  variable and swaps its bounds, so it raises rather than being
  applied silently.
- Absent bounds survive scaling. A variable with no bound arrives
  carrying the `nlp_lower_bound_inf` / `nlp_upper_bound_inf` sentinel,
  which is an ordinary finite number (`±1e19`) and not an infinity, so
  scaling it by a factor below 1 would move it inside the threshold
  and hand a free variable a barrier term and bound multipliers it
  never had. Bounds are scaled only where present, and a present bound
  that would scale *past* a threshold, where it would read as no bound
  at all, is refused with a message naming the threshold it crossed.
- `Problem.set_problem_scaling` accepts `x_scaling` from Python, and
  the C `SetIpoptProblemScaling` applies it, both landing in the
  caller's own units. Each previously refused any non-unit entry.
- The sensitivity accessors were refused at this stage: they read the
  KKT factorization directly and did not yet carry the substitution,
  so they would have reported scaled-space numbers under a
  natural-units contract. Stage 3, below, carries the factors through
  and removes the refusal — within this same release, so no version
  ever shipped with it.

### Added — the sensitivity layer carries the variable factors, and the last refusals lift

- Every sensitivity accessor now answers in the model's own units on a
  variable-scaled solve (gh #486 stage 3), and the four refusals stage
  2 left in place — on pyomo-pounce's `gradient`, `estimate`,
  `covariance` and `information` — are gone. Nothing about user
  scaling is refused any more on any axis.
- The change of variables joins the objective and per-row factors in
  the held factor's natural-units conjugation (`PdSensBacksolver`), as
  a `1/d` on both sides of the `x` block and a `d` on the way out of
  the bound-multiplier rows — the `z`-row `d` cancels against the
  slack diagonal, so it appears in `F` alone. It is diagonal like the
  other two, so the three compose by elementwise product and
  `kkt_solve`, `parametric_step`, `parametric_step_full` and the
  reduced Hessian all follow without further work.
- The accessors that read the model's matrices rather than the factor
  carry it separately: `hessian_vec` (`H = H̃ ⊙ (d ⊗ d)`),
  `row_normal` (`∇g = ∇g̃ ⊙ d`), and `classify_activity`, whose
  exported `Σ` gains the `d²` it was missing. Classification itself is
  run on the unscaled geometry rather than only corrected on export:
  the per-entry ratio `Σ/q` absorbs any `d`, but the identification
  floor is one number shared across entries, so a *non-uniform* `d`
  would otherwise move entries across it and change a status.
- The two consumers that capture the algorithm's iterate — the
  `Solver` session's `ConvergedState::x` and the `SensSolve` builder's
  `x` / `mult_x_L` / `mult_x_U` — undo the substitution, the same
  predicate that caught the CLI's `on_converged` hook in stage 2. The
  `sens_boundcheck` projection now scales its step into the solve's
  own coordinates before clamping against the solve's own bounds,
  which the natural-units step no longer shared.
- The factors a solve ran under are readable back:
  `Solver.nlp_scaling["x_scaling"]` (Python),
  `Solver::variable_scaling` (Rust), reported in the user's full-x
  space. Diagnostic rather than a correction to apply, since every
  output already carries it.
- pyomo-pounce's in-process sensitivity path applies variable factors
  too. It builds no `.nl` for the solver, so it had no suffix segments
  to read; `problem_scaling` now translates the Suffix's variable
  entries into `set_problem_scaling(x_scaling=…)` alongside the row
  ones, by name against the written model's columns.
- `nlp_scaling_method` note unchanged by any of this: `tol` still
  compares scaled quantities, matching upstream.

### Fixed — the inertia test was answered with `δ_w` even when the factorization could not measure inertia (#540)

- CUTE `eigena2` stopped at `Solved_To_Acceptable_Level` with the dual
  infeasibility stuck at `3.31e-07`, where Ipopt certifies `Optimal` at
  `9.31e-09`. The objective was already correct to twelve digits; only the
  last two orders of the dual residual were missing. It now converges to
  `Solve_Succeeded` in **27 iterations at `5.73e-10`** — the same iteration
  count as Ipopt, at a residual an order of magnitude tighter, with the
  constraint violation at `2.2e-16`.
- The reported symptom was `δ_w` re-escalating from `10^-0.8` to `10^1.4` at
  the second-to-last iteration, damping the Newton step from `1.2e-7` to
  `8.2e-9`. The **`δ_w` update rule is not at fault** — it is an exact port of
  upstream's `get_deltas_for_wrong_inertia` and it did what its input told it
  to. The input was wrong.
- `eigena2`'s constraint Jacobian degenerates as the iterate converges: 45 of
  its 55 singular values fall to `~1e-8` by iteration 27, so every KKT
  factorization taken with `δ_c = 0` down that tail is **singular to working
  precision** (smallest pivot `~1e-16` on a `‖A‖ ≈ 240` matrix). The
  negative-eigenvalue count read off such a factor is noise, not a
  measurement: the same iterate returns 64, 58 and 62 against an expected 55,
  and an exact LAPACK eigendecomposition of the dumped matrices agrees with
  none of them. The ladder then multiplied the Hessian perturbation by 8 per
  retry against a reading that does not respond to `δ_w` at all.
- **The perturbation that repairs a rank-deficient constraint block is
  `δ_c`**, which is what the `Singular` verdict reaches for. A mismatching
  count that is contradicted by a pivot at the working-precision floor is now
  reported `Singular` rather than `WrongInertia`, so `δ_c` is applied first.
  It lifts the smallest pivot from `~1e-16` to `5.8e-09`, and from there the
  counts the backend reports agree with LAPACK exactly. Upstream's MA27 /
  MA57 / MUMPS interfaces likewise test singularity *before* comparing the
  count.
- **It cannot cost a usable factorization.** The trigger is consulted only
  once the count has already mismatched — i.e. on a factorization the caller
  was going to reject either way — so it never turns a successful factor into
  a failure; it only changes which perturbation is reached for first.
- New option **`feral_inertia_pivot_floor`** (default `1e-12`, the middle of
  the `n·eps` range over which an equilibrated pivot loses its sign) sets the
  threshold; **`0` disables the trigger**, restoring the previous routing.
- `PDPerturbationHandler::PerturbForSingularity` no longer assumes the
  degeneracy probe is in its pristine state when a `Singular` verdict
  arrives. Upstream encodes that as `DBG_ASSERT`s, but every real linear
  solver can report singularity from any rung of the `δ_w` ladder (MUMPS
  `INFO(1) = -10`, MA27 `IFLAG = 3`); the probe is now abandoned and the
  determined-state path taken, which keeps the `δ_w` rung already paid for.
### Added — `theta_max_row_scale_kappa`, an opt-in rescue for large models that stall from a feasible start (#476)

- **No default behaviour changes.** The new option defaults to `0`, which is
  upstream Ipopt's ceiling bit-for-bit.
- The filter rejects outright any trial whose constraint violation exceeds
  `theta_max = theta_max_fact · max(1, theta_0)` (Eqn. (21)). That `1` is
  dimensionally wrong for POUNCE, because `theta` is a **1-norm over
  constraint rows** — a *sum* of `m` residuals — so a fixed ceiling means a
  mean per-row violation of `theta_max/m`, which shrinks as the model grows.
  On a model started at a **feasible** point (`theta_0 = 0`) the `max`
  collapses entirely and the ceiling becomes the bare constant `1e4` no matter
  how many rows there are.
- `robot_a` (Vanderbei, `m = 52013`) starts feasible, so `theta_max` locked at
  `1e4` — a mean per-row allowance of `0.19` — while the route to the optimum
  passes through `theta ≈ 9.4e7`. Every step toward the solution was refused at
  the gate and the solve ground to `max_iter` at objective `8.173304`.
- Setting `theta_max_row_scale_kappa = 1` floors the reference at the row
  count, so the ceiling means a mean per-row violation of `theta_max_fact`
  independent of `m`. Measured against Ipopt 3.14 on the same machine:

  | model | POUNCE default | POUNCE `kappa = 1` | Ipopt (default) |
  |---|---|---|---|
  | `robot_a` | `Maximum_CpuTime_Exceeded`, 8.173304 | **Optimal, 1.0431952061, 112 it** | `Maximum_Iterations_Exceeded`, 8.173304 |
  | `robot_b` | `Maximum_CpuTime_Exceeded`, 15.484684 | **Optimal, 2.3330990188, 252 it** | `Maximum_Iterations_Exceeded`, 15.484684 |
  | `robot_c` | `Maximum_CpuTime_Exceeded`, 29.039906 | **Optimal, 1.4059755771, 109 it** | `Maximum_Iterations_Exceeded`, 29.039906 |

  Ipopt is affected identically and needs `theta_max_fact = 1e8` set by hand to
  solve any of the three.
- **Why it is not the default.** Raising the ceiling relaxes a
  global-convergence safeguard, and a model that was not blocked by it can
  wander instead: on the Vanderbei corpus `brainpc1/3/5/7` (`m = 6900`,
  `theta_0 = 1e-2`) regress, `brainpc1` from `Optimal` in 64 iterations to
  divergent (objective `3.7e3` against `4.4e-04`). A kappa scan showed the
  damage is a **step function, not a gradient** — `brainpc3` and `brainpc7`
  land on the identical worse answer at every kappa in `{0.01, 0.05, 0.2,
  1.0}`, while `robot_a` improves monotonically (287 → 153 → 127 → 112
  iterations) — so no single multiplier separates the two families. A *static*
  floor cannot: the question is whether a model's route to the optimum needs
  the headroom, which the row count does not answer. Raising the ceiling
  adaptively, only when trials are demonstrably being rejected at the
  `theta_max` gate, is tracked as follow-up work on #476.
- Upstream papers over the one instance of this it noticed by hard-coding
  `resto.theta_max_fact = 1e8` for the restoration sub-IPM
  (`IpRestoMinC_1Nrm.cpp:91`) — the same degeneracy, since the resto NLP is
  also initialised feasible. That sub-IPM runs with `kappa = 0` regardless of
  this option, so it stays bit-for-bit upstream.

### Fixed — the acceptable-level streak had no progress test, so it stopped solves that were still descending (#533)

- Acceptable-level termination fired after `acceptable_iter` (default 15)
  consecutive iterates under `acceptable_tol`, with **no test for whether the
  solve was still making progress**. On two corpus models that streak
  completed at a point that is near-KKT for the *barrier subproblem* while the
  NLP solve was still descending, and POUNCE returned a worse answer under a
  weaker status than it would have reached by continuing.
- `kissing` (Vanderbei) stopped at iteration 103 with objective `1.00000108`
  and `Solved_To_Acceptable_Level`. Continuing reaches `0.84544259` with a
  strict certificate at 550 — the reported answer was **18% high**, and the
  lower value matches Ipopt's `0.845442591227744` to eight figures. The
  iterate it stopped on was near-stationary for the barrier subproblem at
  `lg(mu) = -8.6`, with `‖d‖` oscillating around `1e-6` rather than shrinking,
  and its `inf_du` (`4.15e-07`) was an order of magnitude *worse* than one it
  had already reached inside the same streak (`3.35e-08`).
- `NARX_CFy` (Mittelmann) stopped at iteration 565 with both residuals near
  `1e-7`. Sixty more iterations and 25 more seconds collapse them by five
  orders and reach an objective (`8.6445195e-03`) that beats both its own
  acceptable answer and Ipopt's — and at `275.7 s` that is inside the
  benchmark's `300 s` limit, so the streak, and only the streak, stopped it.
- **The streak must now also have flattened.** Across the `acceptable_iter`
  iterates that made it up, the *spread* (`max − min`) of the KKT error must
  sit within `acceptable_progress_kappa · acceptable_tol`, and the spread of
  the objective within the same fraction of `acceptable_tol · max(1, |f|)`.
  Spread rather than trend, because both models were *wandering* across the
  band rather than converging inside it, and either signal alone is enough to
  keep solving, because `kissing`'s objective was flat to all eight printed
  figures over the iterates in question while the continued run moved it by
  15%.
- **Never worse, by construction.** Like the masked-certificate veto (#200),
  this is a bet that is *tested*, not predicted: the refused termination is
  recorded, and a run that fails to do better ends at exactly that iterate
  under exactly that status (`Solved_To_Acceptable_Level`) instead of
  surfacing `Maximum_Iterations_Exceeded` or a bare failure. The cost of a
  misfire is bounded extra iterations, never a lost verdict.
- A refused acceptable-level termination can now also be paid off by a
  *better acceptable point*, not only by a strict certificate: a continued run
  that itself exits `Solved_To_Acceptable_Level` at a better-ranking point
  keeps that point (same status, better answer) rather than rolling back. Only
  an acceptable-level refusal admits this candidate, so a refused strict
  certificate's `Success` is never reported at a point that qualified only at
  the acceptable level.
- The count itself is untouched: the streak still advances on the band test
  alone, so the refused iterate is exactly the one the unvetoed run would have
  returned. Solves that reach `tol` never complete a streak and are entirely
  unaffected; a solve that stalls to a genuine standstill sees a flat window
  and terminates as before.
- New option **`acceptable_progress_kappa`** (default `0.1`) sets the
  fraction of the band; **`0` switches the progress test off entirely**,
  restoring upstream Ipopt's bare consecutive-count criterion bit-for-bit.
### Fixed — `auto` sent the NETLIB GEN family to the convex IPM, which cannot certify it, while the NLP path solves it in a second (#535)

- `solver_selection=auto` routes every detected LP to the convex
  interior-point method. On `gen` / `gen1` that costs **194× in wall clock
  and the certificate**: 199 of a 200-iteration budget, 190.8 s, and
  `Solved_To_Acceptable_Level` at a primal residual of `1.374e-7` against
  `tol = 1e-8`. The general NLP filter-IPM — the same binary, the default
  for every other class — solves the same model in **19 iterations and
  0.982 s** to `Solve_Succeeded`, matching Ipopt-3.14.20/MA57's objective
  to four figures.
- The root cause is #133 and still stands: the family is highly degenerate
  and rank-deficient, strict complementarity fails, the
  fraction-to-boundary step collapses, and a pure IPM cannot certify the
  vertex. Crossover was built to close exactly this and does not (it is off
  by default because it regressed LP-suite solve times 3×–800× while still
  not reaching an exact vertex on GEN). This is not the #528 noise floor
  either — the largest right-hand side in `gen.nl` is `65.16`, so #531's
  `κ = 64` floor sits at `9.2e-13` there, five orders below the violation.
- **The routing is now the lever, not the crossover engine.** Under `auto`,
  an LP whose convex solve finishes *without a certificate* —
  `OptimalInaccurate` or `IterationLimit` — is re-solved on the general NLP
  interior-point path, which owns the whole verdict. An LP is also a valid
  NLP, and the fallback is the same discipline the conic path has used
  since the `airport` stall: the decision is taken above the status line,
  the `.sol` write and the JSON report, so a rerouted run still reports
  exactly one status.
- It is also the *faster* answer here — one second of NLP after a failed
  convex attempt is nothing against the 190.8 s the convex attempt costs.
- Narrow by construction. It does not fire on a convex QP (`P ≠ 0`), on a
  solve that certified, on a *verified* infeasible / unbounded verdict, when
  `solver_selection` names an engine, when `max_iter` was set explicitly
  (a user-set budget is the question being asked, and `max_iter=0` must
  still stop without a solve, #186), or with the interactive debugger
  attached. A tightened `tol` deliberately does **not** suppress it: that is
  an accuracy request, so trying the engine that can meet it is the right
  response.
- With this the `Ipopt only` column of the benchmark report loses its last
  two entries that POUNCE was thought unable to reach at all.
### Fixed — `dual_inf_tol` was an absolute bound on a normalized quantity, so large-gradient models could not certify (#532)

- The dual twin of #528. The KKT error the strict test compares against
  `tol` is `max(‖∇L‖_∞/s_d, max(‖c‖_∞, ‖d − s‖_∞), ‖compl‖_∞/s_c)`, and its
  dual term carries `s_d`, which grows with the mean magnitude of the
  multipliers. The per-component gate then tested that same `‖∇L‖_∞`
  against `dual_inf_tol`, default `1.0` — one quantity, two standards, and
  on a model whose gradients live at `1e10` they sit ten orders apart.
- Vanderbei's `orthrds2` at default options: `s_d ≈ 1.6e10` with
  `‖∇L‖_∞ = 89.7`, an aggregate dual term of `5.6e-09` against
  `tol = 1e-8` — stationary to nine digits relative to the size of the
  gradients involved — and `89.7 > 1.0` refused it. The solve exited
  `Solved_To_Acceptable_Level` holding the answer; `dual_inf_tol=1e3` alone
  turned it into `Optimal Solution Found` at the same objective, with
  nothing else about the solve changing.
- The map that exposes it is as simple as they come: multiplying an
  objective by a positive constant changes no feasible point, no solution
  and no active set, but multiplies `∇f`, every multiplier, `s_d` and
  `‖∇L‖_∞` — so a large enough constant costs the certificate.
- **The strict gate now judges the unscaled dual infeasibility against
  `max(dual_inf_tol, kappa · tol · dual_scale)`**, where `dual_scale`
  (`IpoptCalculatedQuantities::curr_unscaled_dual_infeasibility_scale_max`)
  is the magnitude of the largest single term `∇L` is assembled from —
  `∇f`, `Jᵀy`, the bound multipliers. `∇L` is the sum of exactly those
  terms, so `‖∇L‖_∞ / dual_scale` is the fraction of them that failed to
  cancel: a scale-invariant statement of stationarity, and one `s_d` cannot
  make, since it is built from multiplier magnitudes alone and never sees
  `∇f`.
- The relaxation only ever forgives a residual small *relative to the
  problem's own scale*, and never admits a non-stationary point:
  `min -exp(x) s.t. x >= 0` reaching `inf_du = 8.8e+47` has `∇f = −8.8e47`
  with no multiplier to meet it, so nothing cancelled and it stays refused
  by eight orders. It is bounded twice over — the aggregate `nlp_err <= tol`
  gate must still pass on the same iterate, and at the default `kappa = 1`
  the floor does not rise above `dual_inf_tol` until `dual_scale` exceeds
  `dual_inf_tol / tol = 1e8`, leaving every `O(1)` model on upstream's
  comparison bit for bit. `acceptable_dual_inf_tol` is untouched.
- New option **`dual_inf_scale_kappa`** (default `1`) sets the safety factor
  on the floor; **`0` switches it off entirely**, restoring upstream Ipopt's
  bare-absolute bound. That is also the setting to use if you tighten
  `dual_inf_tol` and want that absolute standard honoured unconditionally.
### Fixed — the restoration-declining guard stopped solves that were still converging (#534)

- When the line search fails at a point that already passes the
  acceptable-level tolerances, POUNCE declines to enter restoration and
  reports that point (upstream `IpBacktrackingLineSearch.cpp`'s
  `ACCEPTABLE_POINT_REACHED`). The reasoning is sound — restoration reduces
  the constraint violation, and from an acceptable point it has nothing to
  reduce and a reportable solution to lose — but the guard read the entry
  point and **nothing about the trajectory that reached it**, so it stopped
  a contracting endgame and a dead stall with equal confidence. On CUTE
  `eigena2` it fires while the dual infeasibility is quartering every
  iteration on unit steps (`1.19e-5 → 2.96e-6 → 7.38e-7 → 1.84e-7`), three
  iterations short of a strict certificate at a point already feasible to
  `2.4e-11`.
- **The guard now asks whether the solve is still converging.** The decline
  is deferred — at most `resto_decline_deferrals` times per solve, default
  1 — when the overall NLP error contracted by at least
  `resto_decline_progress_ratio` (default `0.5`) on each of the last three
  outer iterations. The solve then continues for up to ten more iterations.
- **A deferral that does not pay off is free.** The point the guard would
  have returned is snapshotted as a floor; if the continuation does not
  reach a strict certificate, or ends anywhere that is not at least as good
  an answer, the floor is restored and reported. The deadline is also
  clamped below `max_iter`, so a lost bet cannot turn
  `Solved_To_Acceptable_Level` into `Maximum_Iterations_Exceeded`. Worst
  case is the old behaviour plus a bounded handful of iterations, never a
  worse reported point.
- Where the answer *is* a stall the guard fires exactly as before: on
  `csfi2` — the other model the issue names as reaching this guard, added
  as a fixture — the window at the decline is
  `[3.267e0, 1.845e-6, 8.468e-8, 8.524e-8]`, flat on its last step, and the
  default build's answer is bit-identical to `resto_decline_deferrals=0`.
  Forcing the deferral there costs 11 iterations and returns the same
  primal vector, bit for bit.
- Two new options, both pounce additions: `resto_decline_deferrals`
  (`0` restores the pre-#534 behaviour) and
  `resto_decline_progress_ratio` (set it very large to drop the progress
  requirement entirely — the "bypass the guard and see how far the solve
  gets" experiment, previously only reachable by patching the source). The
  `pounce::algorithm` debug line for a decline now carries the NLP-error
  window and the verdict, so a trace answers "why did it not defer?" on its
  own.
- **Not verified end-to-end on `eigena2`/`eigenb2`.** Those `.nl` live in
  the gitignored benchmark archive and were not reproducible from the
  published `.mod`; what the progress test does on their recorded traces is
  pinned as a unit test over the numbers in the issue instead.

### Fixed — feasible, bounded LPs exited `Search_Direction_Becomes_Too_Small` once the data reached `~1e7` (#528)

- On plain LPs with `O(1)` matrix entries but right-hand sides of magnitude
  `~1e7` and above, the NLP filter-IPM exited code 3
  (`Search_Direction_Becomes_Too_Small`) while *holding the correct
  optimum* — matching scipy/HiGHS to eight significant figures. 9 of 48
  runs in the reporter's sweep, none at `1e6`, more of them at `1e8`.
- The KKT error the convergence test compares against `tol` is
  `max(‖∇L‖_∞/s_d, max(‖c‖_∞, ‖d − s‖_∞), ‖compl‖_∞/s_c)`. The dual and
  complementarity terms are normalised; the primal one is a bare absolute
  residual — and `c_i` and `d_i − s_i` are each a difference of quantities
  the row's own size, so they are quantised in units of `eps ·` that
  magnitude. At `|b| ~ 1e8` the smallest **nonzero** value `‖d − s‖_∞` can
  take is one ulp, `1.5e-8`, already above the default `tol = 1e-8`. So
  `nlp_err <= tol` stopped being a statement about the iterate and became a
  bet on the residual landing on an exact `0` rather than on one ulp —
  which is why the failures were scattered across seeds instead of
  uniform. An iterate that lost the bet could never certify, the solve kept
  recomputing a point it could not improve, and it exited on the collapsed
  search direction.
- **The strict gate now judges the primal term against the finest residual
  each row can represent**
  (`IpoptCalculatedQuantities::curr_primal_infeasibility_above_noise`,
  built on the same per-row `row_noise_floor` model #390 / #446 already use
  to decide when a residual is too fine to be real). Only that gate reads
  it: `constr_viol` is still tested against `constr_viol_tol` on the full,
  unfloored residual, and the scale-relative feasibility veto still sees it
  too — so the floor cannot admit a violation the user's own feasibility
  tolerance would reject, it only stops an unrepresentable one from vetoing
  a certificate. The acceptable-level band keeps the raw error.
- The reporter's 48-run sweep now returns `Solve_Succeeded` on all 48, at
  the scipy optimum; the same LP family solves cleanly out to a data scale
  of `1e10`.
- **It fixes considerably more than the LP family it was opened for.** A
  corpus sweep against the parent commit found 8 status changes across 733
  Vanderbei NLPs, every one an improvement and no regressions: `orthrege`
  goes from 2652 iterations and `Solved_To_Acceptable_Level` to **84
  iterations** and a real certificate at the same objective; `steenbrf`
  from a 3000-iteration `Maximum_Iterations_Exceeded` to `Solve_Succeeded`
  in 481; `cresc4` from `Infeasible_Problem_Detected` to `Solve_Succeeded`.
  Each is the same pathology as the issue — thousands of iterations spent
  at a point the solver could not improve because the certificate was
  unreachable.
- **On LPs and QPs nothing moves at all.** 371 netlib/Meszaros LPs and 138
  Maros-Mészáros QPs are bit-identical to the parent commit, model by
  model, in both the objective bit pattern and the iteration count: on data
  that is `O(1)` no row is near its resolution limit, so the floored
  aggregate and the raw one are the same number and the gate is upstream's.
  On *nonlinear* problems iterate trajectories do change — a solve that
  previously spent nine iterations re-rolling until the residual happened
  to land on an exact `0` now stops when it has converged, so iteration
  counts and final points differ even on models that already succeeded.
- Locally-infeasible models can now cost *more*: the ℓ₁ fallback's outer
  loop stops escalating ρ when an inner solve fails numerically, and the
  old false `Search_Direction_Becomes_Too_Small` was being read as exactly
  that. With the inner solve certifying honestly the loop runs the
  escalation it was always meant to (`cresc50`: 2 attempts → 3, same
  `Infeasible_Problem_Detected`). The earlier saving was a false economy,
  not a cost this removed.
- New option **`primal_noise_floor_kappa`** (default `64`) sets the safety
  factor on the per-row floor; **`0` switches it off entirely**, restoring
  upstream Ipopt's bare-absolute primal term bit-for-bit.
- When the floor changes the verdict the end-of-run summary now says so,
  printing the tested value under `Overall NLP error`. Solves where the two
  agree — every `O(1)` model — keep the summary block unchanged.

### Fixed — `pounce-convex` presolve documented a fixpoint it never reached, and the layer cap silently decided the reduction (#527)

- `presolve` describes itself as iterating the reduction passes to a
  **fixpoint**. On netlib `bore3d` it never reached one: it exited on the
  `MAX_ROUNDS` layer cap on every solve, at every cap tried (32, 64, 200).
  An arbitrary defensive constant — not the algorithm — was choosing which
  of two different reduced problems the solver was handed, and nothing
  anywhere recorded that it had. Follow-up from #523 / #525, which fixed the
  correctness bug this mechanism set up.
- **The cause is that bound tightening is the one reduction that consumes
  nothing.** A fixing removes its column and an aggregation removes its
  row, so those can fire at most `n + m` times however long the loop runs.
  Narrowing a box leaves the column *and* the row in place, so rows that
  mutually imply ever-tighter bounds fire every round, converging toward a
  limit they never reach. A `MAX_BOX_REFINEMENTS` budget (12) now bounds how
  many times the iteration may refine the *same* box side, carried across
  rounds and renumbered onto each layer's surviving columns. Termination
  follows from the algorithm; with the cap lifted, `bore3d` reaches a real
  fixpoint for the first time (238 layers).
- The budget is a **count, not a magnitude**, so no scale-dependent constant
  of the kind #523 came out of enters. That matters because a
  minimum-improvement threshold does not work here, and the absolute and
  relative versions fail on different models: an absolute floor
  (`BOUND_FEAS_TOL`) already exists and stops a cascade collapsing toward
  *zero*, but not one converging to a limit of `1e3`, where the improvements
  stay far above any floor; and a relative floor fails on `bore3d` itself,
  whose cascade shrinks each bound to `3.887e-2` of its previous value — a
  96% relative improvement, every round. Exhausting the budget costs at most
  a looser box, never a wrong one.
- **The exit reason is now recorded and reported.** `PresolveStats` carries
  `exit: FixpointExit` (`Fixpoint` or `RoundCap`) and `rounds`, and the CLI
  prints a line whenever the loop was truncated. A reduction that came out
  of the cap is no longer indistinguishable from one that came out of a
  fixpoint — which is why this went unnoticed through three releases.
- **Presolve got faster.** Duplicate/parallel-row merging reads coefficients
  and right-hand sides only, never a variable box, so a round that follows a
  bounds-only round and itself fixes, substitutes and drops nothing is
  re-deduping rows it already deduped. Skipping it there is a memoization,
  not a heuristic. That hashing pass was ~70% of presolve's cost on a
  cascade: `bore3d` presolve goes from ~14 ms to ~5 ms for the identical
  reduced problem (128 vars / 77 rows).
- `MAX_ROUNDS` is left at 32 deliberately. It is no longer the termination
  argument and no longer silent, and raising it was measured rather than
  assumed: 128 layers cost ~16 ms and reduced `bore3d` to exactly the same
  problem 32 layers reach, while multiplying the chain's retained memory
  (every layer holds a clone of its input). `bore3d` still stops on the cap,
  for a second and independent reason now documented in the module: bound
  propagation is *serialized* by the disjoint-source rule that keeps the
  dual re-attributions independent, so one round advances the propagation
  graph by roughly one edge per column — about three tightenings per layer
  on `bore3d`, with variables still receiving their first finite bound at
  layer 320. Measured, that extra depth changes only boxes: 32 layers and
  330 layers give the same 128 variables, 77 rows, 61 fixings, 17 forcing
  rows and 64 aggregations.

### Fixed — `pounce-convex` reported false `Infeasible_Problem_Detected` at iteration 0 on `bore3d` / `QBORE3D` (#523)

- Netlib `bore3d` and its Maros-Mészáros quadratic twin `QBORE3D` (both
  n=315, m=233, both feasible) came back from the convex path as
  `Infeasible_Problem_Detected` in ~5 ms with `iters=0` and no diagnostic,
  while the NLP path (`solver_selection=nlp`) solved both to the committed
  Ipopt-MA57 reference objectives. They were the only non-matching failure
  in the `lp` suite and the only one in the `qp` suite.
- The claim came from presolve's **forcing-constraint** reduction. Bound
  tightening propagated a group of nonnegative variables' upper bounds
  geometrically toward their true limit of zero; by round 21 of the
  presolve fixpoint those boxes were ~1e-8 wide, and one equality row's
  activity range `[-6.29e-10, 4.87e-8]` came within `ACTIVITY_TOL` (1e-9)
  of its right-hand side `0`. Forcing read that as "this row can hold only
  at its min vertex" and pinned all six of its variables to bounds — two of
  them, with coefficients `-1.14e-1` and `-5.7e-3`, to *upper* bounds
  `4.8e-9` and `1.5e-8` from zero. A gap of `6.29e-10` had licensed a
  displacement of `1.5e-8`. Substituted into the next row those two
  appeared in, the residual was `2.0e-8` against a tolerance of `1.0e-9`,
  and a feasible problem was declared infeasible.
- **Forcing now requires the pin to be tight.** A row whose activity range
  only *approximately* touches its right-hand side may still spend the
  residual gap, moving each variable up to `gap / |coefⱼ|` (capped by its
  box width) off the bound the pin claims it must occupy. Forcing fires
  only when that displacement is within `FORCING_PIN_TOL` for **every**
  variable in the row, so a pin is a deduction rather than a guess. A row
  that genuinely touches (gap 0) is unaffected.
- **An infeasibility verdict is now re-derived before it is emitted.**
  Two reductions — forcing constraints and dominated columns — fix a
  variable at a value chosen from a tolerance judgment, and a wrong one is
  substituted into every row that variable touches until some row reads as
  contradictory. When any screen reports infeasible, presolve re-runs from
  the original problem with those two withheld; only a verdict that pass
  reaches on its own is returned as `PrimalInfeasible`. Otherwise the
  reduction is solved normally and the discarded claim is kept on the
  handle (`Presolve::discarded_infeasibility`). Everything that only
  *reports* — empty rows, activity ranges, parallel rows, emptied-row
  residuals — is retained, so no infeasibility class detectable before is
  lost. Verified: with the forcing fix reverted, the guard alone still
  solves both problems to the reference objective.
- **The screen that fired is now named.** `PresolveOutcome::Infeasible`
  carries an `InfeasibleTrigger` (the screen plus the row / column / bound
  and the compared values), and the CLI prints it — `Presolve: proved
  primal infeasible — <screen> (<detail>)`, or `Presolve: discarded an
  unconfirmed infeasibility claim — …; solving normally`. Previously even
  `print_level=8` emitted nothing on the way out.
- **Breaking (pounce-convex API):** `PresolveOutcome::Infeasible` is now a
  tuple variant; `matches!(…, PresolveOutcome::Infeasible)` becomes
  `PresolveOutcome::Infeasible(_)`.

### Fixed — the local-infeasibility re-solve corroborated its own mistake; it now varies the barrier trajectory too (#524)

- A local-infeasibility verdict was second-guessed by exactly one re-solve,
  `feral_scaling=mc64` (`feral_infeasibility_scaling_retry`). That rung
  perturbs the *linear algebra* only, so it is evidence only when the
  trajectory is ULP-hypersensitive the way `discs.nl` is. When it is not,
  MC64 retraces the same iterates and agrees for the same reason the first
  solve was wrong — and the CLI then printed "now corroborated by a second
  scaling", asserting confidence that had not been earned.
- The worked case is CUTE `cresc4` (6 variables, 8 constraints, nonconvex,
  feasible; LOQO, SNOPT and Ipopt-MA57 all reach `0.8718976`, Ipopt in 71
  iterations). POUNCE on defaults converged to a point with constraint
  violation 0.51 and objective `~2e-09` — a degenerate zero-area crescent —
  and reported `Infeasible_Problem_Detected`. The MC64 re-solve reproduced
  the failing trajectory character-for-character through iteration 15,
  diverged at iteration 16 in the eighth significant digit — the exact
  hypersensitivity signature the guard was written for — and landed in the
  same basin anyway. Whether that ULP-scale perturbation escapes is luck.
- The retry is now a two-rung ladder, and the new second rung
  (`infeasibility_mu_strategy_retry`, default `yes`) re-solves with
  `mu_strategy=adaptive`, which changes the iterate sequence itself rather
  than the arithmetic underneath it. `cresc4` now returns
  `Optimal Solution Found` at `0.8718975`, matching the reference to eight
  significant figures. Retrying a different barrier strategy is also the
  remedy IPOPT's own documentation gives a user who gets an infeasibility
  verdict on a problem they believe is feasible.
- Rungs apply to the *baseline* options, not on top of each other: the
  barrier rung re-asserts the baseline `feral_scaling` first. This is
  load-bearing — on `cresc4`, `mu_strategy=adaptive` recovers the optimum
  and `mu_strategy=adaptive` together with `feral_scaling=mc64` still
  reports local infeasibility, so a cumulative ladder would have discarded
  the fix.
- Safety is unchanged in the direction that matters: a rung is promoted
  only when it returns `Solve_Succeeded` / `Solved_To_Acceptable_Level`, so
  an overturned verdict is always backed by the retry's own convergence
  check rather than by trusting the strategy. Genuinely infeasible models
  still report infeasible (the ladder simply runs out of rungs), a rung
  that would be a no-op at the current options is skipped rather than
  burning a solve, presolve-certified infeasibility remains exempt, and the
  extra solve is spent only on runs that would otherwise report failure.
  Set both options to `no` for upstream IPOPT's behaviour of shipping the
  first verdict.
- What it costs: nothing on a successful solve (a 733-problem Vanderbei
  sweep shows no objective drift and no status change other than `cresc4`),
  and up to two extra solves on one that reports infeasible. On the suite's
  hardest infeasible case, `cresc132`, that is 3.65s → 73.4s; across the
  whole suite, +7 %.

### Fixed — a successful restoration phase left the watchdog's shortened-step counter running, arming the watchdog where IPOPT would not (#524)

- `steenbrf` (n=468, m=108) ran to `max_iter` at 3000 iterations under the
  monotone barrier default while Ipopt-MA57 solved the same file. It now
  converges in **481 iterations** to `Solve_Succeeded` — to full tolerance,
  at a lower local minimum (282.678) than the reference's acceptable-level
  point (1321.65 at 1846 iterations).
- The watchdog arms after `watchdog_shortened_iter_trigger` (default 10)
  **consecutive** shortened line-search steps. Upstream clears that counter
  when the restoration phase succeeds
  (`IpBacktrackingLineSearch.cpp:624-631`), because an iterate that came
  back from restoration is a different point and the run of shortened steps
  before it did not continue through it. POUNCE had no equivalent: upstream
  calls `PerformRestoration()` from inside `FindAcceptableTrialPoint`, so
  those assignments sit in scope there, whereas POUNCE returns
  `Outcome::Failed` and the main loop runs restoration — and its recovery
  path never told the line search it had happened.
- So runs of shortened steps on either side of a restoration accumulated as
  if consecutive. On `steenbrf`: five shortened steps, restoration, five
  more — exactly the trigger. The watchdog armed, spent its three trial
  iterations, reverted to the pre-watchdog iterate (the reverted iteration's
  objective and `inf_pr` are bit-identical to the snapshot's), and the line
  search then collapsed to `alpha` ~1e-08 with 20+ backtracks. That cycle
  repeated 105 times. Upstream's longest run of consecutive shortened steps
  on this problem is 6, so its watchdog never arms at all.
- `in_soft_resto_phase` and `soft_resto_counter` are reset alongside it, as
  upstream does at the same site.
- Corpus effect (733-problem Vanderbei sweep, same host): **two problems
  fixed** — `steenbrf` and `brainpc2` (`Maximum_Iterations_Exceeded` at 3000
  → `Solved_To_Acceptable_Level` at 1003) — **none broken**, no objective
  drift on any problem both runs solve, and the suite gets *faster*: total
  solve time −6.5 %, total iterations −7.3 %.

### Added — the CLI prints a machine-readable `Status:` line

- Every NLP solve now ends with `Status: <upstream_name>` —
  `Status: Infeasible_Problem_Detected`, using the upstream IPOPT
  enumerator spelling that CUTEst tables and the benchmark references
  already use. Printed once, after the verdict is final, and suppressed at
  `print_level 0` and under `--json-debug` (whose stdout is a protocol
  channel).
- This exists because free-form banners are not a status channel and the
  second-opinion ladder above proved it. The engine prints one `EXIT:`
  banner per *solve*, so a laddered run prints several, and a consumer that
  scans the log for known phrases gets whichever phrase it ranks first
  rather than the verdict that shipped. `benchmarks/scripts/run_nl_bench.sh`
  ranks the iteration-limit phrase above the local-infeasibility one, so on
  `cresc100` — barrier rung hits `max_iter`, original infeasibility verdict
  then stands — it recorded `Maximum_Iterations_Exceeded` for a run that
  shipped `Infeasible_Problem_Detected`. Wrong status, no error, straight
  into `BENCHMARK_REPORT.md`. That driver already preferred a `Status:`
  line and only fell back to phrase-ranking because nothing emitted one.
- `ApplicationReturnStatus::upstream_name()` is public, for embedders that
  need the same spelling.

### Fixed — tightening `constr_viol_tol` made POUNCE more likely to report local infeasibility (#519)

- Rapid infeasibility detection counted an iterate toward its streak when
  the constraint violation exceeded `infeas_viol_kappa · constr_viol_tol`
  (default `1e2 · 1e-4 = 1e-2`), unclamped. `constr_viol_tol` is a
  *feasibility* tolerance, so tightening it lowered the bar for "bounded
  away from feasible": at `constr_viol_tol = 1e-6` a violation of `1e-4`
  counted, at `1e-9` a violation of `1e-7` did. Asking for a stricter
  feasibility standard widened the set of points the solver was willing to
  call infeasible — the wrong direction.
- On the model behind #505, whose run plateaus at an unscaled violation of
  `1.943e-4` with a scaled NLP error of `4.89e-10`, the flip tracked
  `100 · constr_viol_tol` to three significant figures: `constr_viol_tol`
  of `1e-4` / `1e-5` reached Solved To Acceptable Level at iteration 37,
  while `1.92e-6` and `1.94e-6` exited at 26–27 with `objno 200`. The
  reporter's stack set `constr_viol_tol = 1e-6` by default, so every solve
  through it carried the lowered floor.
- Both arms of the detector are now floored at a shared
  `MIN_INFEAS_VIOL_FLOOR = 1e-2` — the relative arm already was, with a doc
  comment stating the rule ("too-loose withholds a verdict, too-tight
  fabricates one"); the absolute arm was the same expression without the
  clamp. `1e-2` is also the default `acceptable_constr_viol_tol`, so the
  detector can no longer convict a point whose violation sits inside the
  band the defaults call acceptable. The two coincide at the defaults,
  which is why the defect was invisible out of the box.
- Behaviour at the default tolerances is unchanged, and genuine
  infeasibility detection is untouched: `infeas_viol_kappa` still raises
  the floor, and the disable switches (`infeas_stationarity_tol = 0`,
  `infeas_max_streak = 0`) are unchanged.

### Fixed — options files were silently ignored: `option_file_name` configured nothing and `./ipopt.opt` was never read (#518)

- `option_file_name=` was *refused* (it named a feature POUNCE did not
  implement) and there was no implicit options-file lookup at all, so the
  only way to configure a run from a file was `--options-file`. On 0.9.0,
  as released, `option_file_name=` was accepted and did nothing: a run
  configured entirely through an options file executed at stock defaults
  and still reported success. Nothing in the output said so, which
  silently invalidates any benchmark set up that way — and the POUNCE side
  looks *better* precisely because it dropped the stricter settings.
- `option_file_name=<path>` is now implemented, and `--options-file` is
  the same thing spelled as a flag (the flag wins if both are given).
- With no file named, POUNCE probes the working directory for
  `pounce.opt`, then `ipopt.opt`, and reads the first that exists — the
  `ipopt` behaviour the issue found most visibly missing. Both present:
  `pounce.opt` wins and the shadowed file is named in a warning. The run
  prints `Using option file "<path>".`, on the same `sb` gate as the
  banner, so a discovered file cannot steer a solve anonymously.
- `--no-options-file` skips the lookup, for a directory holding an options
  file meant for some other run.
- Precedence is unchanged and now uniform: command-line `KEY=VALUE` and
  `$pounce_options` override the file, never the reverse.
- A file that was *named* but cannot be read is an error rather than a
  shrug — upstream's `ifstream` reads nothing and says nothing there,
  which is the same silence in a smaller box. `option_file_name` set
  *inside* an options file still chains nowhere (the file is chosen before
  it is read), but now warns instead of being ignored.
- Naming two different files at once — `--options-file a.opt` beside
  `option_file_name=b.opt`, or either beside `--no-options-file` — is an
  error. A precedence rule there would mean one of the files the user
  named was quietly not read, which is the reported failure reintroduced
  by the fix for it.
- **Library callers still cannot set it.** `option_file_name` reaches a
  file only through the CLI's resolution path; Python, the C interface and
  WASM set their options directly and read no files, so there the option
  is refused (`Invalid_Option`) naming the alternative, rather than
  accepted and dropped. It was the blanket #483 refusal that covered this
  before; the guard now sits exactly where the feature still does not
  reach, and keeps that machinery's default gate — `option_file_name` set
  to its registered `ipopt.opt` asks for nothing and still solves, so a
  caller replaying a full option dump is unaffected. Those entry points
  deliberately do *not* gain the implicit
  `./ipopt.opt` lookup: action at a distance under Python or the GAMS C
  link would be worse than not having it, and `pounce.opt` already means
  something else to GAMS.

### Fixed — `pounce verify` printed "complementarity residual" for a quantity that is not the solver's complementarity (#516)

- `verify` computed *constraint* complementarity — `max_i |λ_i| · dist(g_i,
  nearest finite side)` over **rows**, from the `.sol`'s constraint duals —
  and printed it as a bare `complementarity residual`, on the line below
  `KKT stationarity residual`, under an `optimality` heading. A solver's
  `Complementarity` is *bound* complementarity — `max_j max(|z_L·(x−x_L)|,
  |z_U·(x_U−x)|)` over **variables**, from the bound multipliers. On the
  #505 model at Ipopt's solution the two read `4.529e-2` and `1.1786e-11`:
  nine orders of magnitude, same point, same file, both labelled
  "complementarity". Two people read the first as a signal about the
  solution before tracing it to the definition.
- Both are now named by what they range over — `constraint complementarity
  (rows, |λ|·slack)` and `bound complementarity (vars, |z|·slack)` — in the
  console report, the `--help` text, and the JSON receipt
  (`constraint_complementarity_residual` /
  `bound_complementarity_residual`; the old `complementarity_residual` stays
  as a deprecated alias so v1 consumers keep parsing). No line offers an
  unqualified `complementarity residual` for either to be mistaken for.
- `verify` now *computes* bound complementarity, by reading the
  `ipopt_zL_out` / `ipopt_zU_out` variable suffixes out of the `.sol` — the
  only route by which bound multipliers reach one. Where the suffixes are
  absent the line reads `not checked`, with the reason, rather than `0.0`.
  On the `parametric.nl` fixture the reported `9.091e-10` reproduces the
  solve's own `Complementarity` line exactly; a test pins the agreement.
- The same suffixes sharpen stationarity. The existing residual is
  bound-*projected*: it projects out exactly the component a bound
  multiplier would carry, so it reads `0.0` on a point whose multiplier is
  missing or wrong (#495). With the suffixes in hand `verify` also reports
  the exact `‖∇f + Jᵀλ − (zL_out + zU_out)‖∞`, which matches the solver's
  `Dual infeasibility`, and `--require-optimal` gates on that whenever it is
  available — the projection can only understate the residual. A `.sol`
  whose `ipopt_zU_out` sign is flipped now scores `8.0` where the projected
  residual still reads `0.000e0`.
- Also: a bounds-only model (no rows) legitimately has an empty dual block,
  so `∇f` alone is its Lagrangian gradient. `verify` no longer skips the
  whole optimality section on those.

### Fixed — adaptive μ floored the fixed-mode barrier decrease at `mu_min` instead of upstream's tolerance-derived floor (#511)

- In the adaptive strategy's **fixed-mode** (monotone-mode) decrease, POUNCE
  floored the reduced μ at `mu_min` (`1e-11`). Upstream
  (`IpAdaptiveMuUpdate.cpp:325-329`) floors it at
  `min(apply_obj_scaling(compl_inf_tol), tol) / (barrier_tol_factor + 1)` —
  `9.09e-10` at the default `tol=1e-8`, `compl_inf_tol=1e-4`,
  `barrier_tol_factor=10`. The two are not interchangeable: `mu_min` is the
  *free*-mode clamp, and once the strategy has switched to fixed mode upstream
  deliberately stops μ at the accuracy the termination test asks for, which is
  the point of the switch. POUNCE also skipped upstream's objective-scaling
  conversion of `compl_inf_tol`, so the paths additionally disagreed whenever
  objective scaling was active.
- Effect: whenever the strategy switched to fixed mode and the barrier
  subproblem kept solving, POUNCE went on pushing μ to `1e-11` where upstream
  stops ~91× higher, running the Newton system far past the point where the
  extra digits buy anything — a plausible source of degenerate search
  directions on an ill-conditioned Jacobian.
- The floor now matches upstream, with `compl_inf_tol` converted into μ's
  scaled space (#257) and the certificate-safe `mu_min` cap (#266) still
  `max`ed in so the restoration sub-builder's `100 · outer_mu_min` safeguard
  keeps applying — the same composition the monotone floor already used.
- Behavioural check across the 60 `.nl` fixtures under
  `mu_strategy=adaptive`, at both `tol=1e-8` and `tol=1e-4`: every exit
  status, iteration count and objective is unchanged. The branch binds on
  seven fixtures; on two of them (`jit1_node`, `nonconvex_qp`) the new floor
  changes the μ trajectory, and both still land `Optimal Solution Found` in
  the same iteration count with the same objective — μ simply stops at the
  tolerance-derived floor instead of `1e-11`. On the rest the pre-existing
  objective-scaling cap had already pulled both floors to the same value.
- Found by the adversary agent while dissecting #505; not causal there.
  Coverage in `crates/pounce-algorithm/src/mu/adaptive.rs`
  (`fixed_mode_floor_matches_upstream_not_mu_min`,
  `fixed_mode_floor_scales_compl_inf_tol`,
  `fixed_mode_floor_keeps_resto_mu_min_safeguard`).

### Fixed — adaptive μ reset the line-search filter on "μ changed" instead of upstream's every-free-mode-iteration, so stale entries forced spurious restorations (#510)

- Ipopt's μ updates hold a line-search handle and call `linesearch_->Reset()`
  themselves — which clears the entire filter — at four points:
  `IpAdaptiveMuUpdate.cpp:339` (fixed-mode decrease), `:386` (free→fixed
  switch), `:431` (**unconditionally, every free-mode iteration**), and
  `IpMonotoneMuUpdate.cpp:165` (after a monotone reduction). POUNCE's
  `MuUpdate` trait carries no line-search handle, so the main loop inferred
  the reset from `next_mu != mu_before`. That proxy is right for the monotone
  update and wrong for the adaptive one: upstream's line 431 does not consult
  μ at all.
- The gap was documented in `mu/adaptive.rs` as "primarily affects the
  watchdog counter, not convergence". Measurement says otherwise. On the
  unscaled reproducer from #505 (`nlp_scaling_method=none`, adaptive μ), at
  internal iteration 69 — after a restoration that returned at the μ it left
  with, so no reset fired — the filter still held the pre-restoration entries
  `(θ, φ) = (0.0141, 22157.56)` and `(0.0220, 22157.56)`. Every trial step
  from `α = 2.4e-6` down to `1e-12` was rejected on the filter alone, forcing
  a third entry into restoration. Those entries were computed against a
  barrier parameter and an iterate the algorithm had long since left.
- Each μ update now raises `IpoptData::request_ls_reset` at exactly upstream's
  call sites and the main loop honours it — the same plumbing the pounce#58
  probing guard already uses for `request_resto`. The affected paths are the
  adaptive ones where μ can stay numerically fixed across an iteration: the
  whole free-mode endgame, every iteration following a restoration that
  returns at the same μ, and the two clamp-flattened decreases. Monotone runs
  are unchanged — its reduction loop only ever exits with a strictly smaller
  μ, so flag and proxy agree.
- Not a fix for #505: that model's first restoration is entered on
  iterate-acceptability rejections, not filter rejections, so filter state is
  irrelevant at that point. This removes the later spurious restorations only.

### Fixed — `mu_strategy=adaptive` kept iterating at a frozen point instead of stopping (#512)

- When the line search takes two consecutive steps so small that any nonzero
  step length is floating-point noise, the barrier update gets one chance to
  move μ. If it cannot, nothing else can change either: every later iteration
  recomputes the same point. IPOPT stops there and reports *search direction
  is becoming too small* — "solved to the best accuracy this problem allows".
  With `mu_strategy=adaptive`, POUNCE did not stop; it kept going until it hit
  `max_iter` and then reported *maximum iterations exceeded*, which reads as
  "your model was too hard" when the truth is "we finished 280 iterations
  ago". A code comment asserted that only the monotone update terminates this
  way; upstream's adaptive update does too, at two places
  (`IpAdaptiveMuUpdate.cpp:330-333` and `:377-380`), and both are now ported.
- Measured on the existing fixture corpus with `mu_strategy=adaptive`.
  `airport.nl` at `tol=1e-12` went from 300 iterations and *maximum
  iterations exceeded* to 16 and a clean tiny-step exit — with the final
  objective, dual infeasibility and constraint violation agreeing to every
  digit printed, so the 284 extra iterations moved nothing. Twelve other
  models in the same sweep stop between 7 and 20 iterations where they
  previously burned the whole budget. At **default** tolerances the same
  models keep their status and reach it 4–6× faster: `hs71_obj1e8.nl`
  certifies the same optimum in 11 iterations rather than 70, the `jit1`
  family in 15–20 rather than 73–77.
- Nothing changes for the default `mu_strategy` (monotone) — the fixture
  corpus is bit-identical there — and no model in the sweep traded a better
  status for a worse one.
- Restoration runs the same inner solve, so it gained the same exit, and that
  exposed a hole underneath it: `resto_inner_solver`'s tiny-step
  locally-infeasible gate excluded square problems, so a square model that is
  genuinely infeasible lost its AMPL 200 verdict to *restoration failed*
  (AMPL 500, Pyomo `internalSolverError`) whenever the inner solve reached its
  stationary point by tiny step. Upstream throws local-infeasibility on that
  branch (`IpRestoMinC_1Nrm.cpp:278-291`) with no test on problem shape, and
  POUNCE now matches. This also fixes the same wrong verdict on the default
  monotone path, where it was already reachable: #508's own probe model at
  `tol=1e-12` reported 500 before this change and reports 200 now.
- Found by source comparison against upstream rather than by a failing model;
  the reproducers above were found afterwards, in the fixtures already in the
  repository. Regression coverage in
  `crates/pounce-cli/tests/issue_512_adaptive_tiny_step.rs`.

### Fixed — infeasible models reported `internalSolverError` because the infeasibility threshold was built from `tol` (#508)

- On a model with no solution, POUNCE chooses between *converged to a point
  of local infeasibility* (AMPL 200 — "your model has no solution") and
  *error in step computation* / *restoration failed* (AMPL 500, Pyomo
  `internalSolverError` — "your solver broke"). That choice was made against
  a threshold built from `tol`, a tolerance on the **KKT error**, and never
  consulted `constr_viol_tol`, the option that declares what a violated
  constraint is. Different quantity, different units.
- Two measured consequences, on `min (x−5)² s.t. x²+δ = 0` — infeasible for
  every `δ > 0`, with the reported violation exactly `δ`, so the threshold is
  visible to the digit. Sweeping `constr_viol_tol` across four orders of
  magnitude moved the boundary **not at all**: at `constr_viol_tol=1e-3` a
  violation of `1e-4`, comfortably inside the user's own declared feasibility
  tolerance, still came back 500. And sweeping `tol` moved it a great deal,
  in the wrong direction — the failure band tracked `max(100·tol, 1e-4)`, so
  at `tol=1e-4` every gap from `3e-4` to `1e-2` answered "internal error",
  including a model infeasible by a full percent. Loosening `tol` is the
  standard reaction to a struggling solve, so the failure widened exactly
  when the user tried to help.
- Every one of these sites now tests the constraint violation against
  `constr_viol_tol`, and at the boundary with `>=` rather than `>` — a
  violation *at* the tolerance the user declared too large is a violation.
  Six thresholds changed: the restoration-cycle exit in `ipopt_alg`, which is
  the only locally-infeasible safety net square problems have, and the five
  locally-infeasible gates in `resto_inner_solver` that shared the same
  `100·tol` floor. The invariant now holds across the whole sweep: a
  violation at or above the user's declared feasibility tolerance lands in
  the AMPL infeasible range at every `tol`, and `constr_viol_tol` is what
  moves it.
- Where the remaining violation is *below* `constr_viol_tol`, POUNCE still
  declines to certify infeasibility — the iterate is primal-feasible by the
  user's own declaration, and claiming otherwise is the false-infeasibility
  failure from the opposite direction.
- Also fixed alongside: after a non-promoted MC64 hypersensitivity re-solve,
  the terminal's last `EXIT:` banner was the retry's while the `.sol`, the
  summary and the JSON report all carried the kept first-solve verdict. Two
  banners is expected and announced; the two disagreeing is not, and
  `validation/p3_control.py` reads the log exactly that way. The CLI now
  re-emits the verdict that actually shipped as the final banner.
- Reported by the adversary agent. Regression coverage in
  `crates/pounce-cli/tests/issue_508_infeasibility_gap_status.rs`. The banner
  ordering is covered by an invariant guard rather than a bite-on-parent pin:
  every non-promoted MC64 retry reachable from the fixture corpus happens to
  return the same verdict the `.sol` keeps, so the two banners agree there by
  luck; the mismatch is still a reachable state of the code. Noted in the test
  so the gap is a known one.

### Fixed — false primal-infeasible certificate on redundant equality rows that disagree by one ULP (#496)

- The convex presolve fixes a variable from a singleton equality row and
  substitutes it into the remaining rows. A row that empties in the process
  becomes the feasibility test `0 = rhs`, and that test was written against
  **exact zero**. But `rhs` at that point is `b − Σ aⱼ vⱼ`, a computed
  difference — so it carries the subtraction's rounding error, and a second
  equality that pins the same variable to the same value fails the test by
  one ULP.
- Found by the adversary agent while probing #492; unrelated to that change
  and reproduced on `main`. Two rows on one column, `−2.830268 x₀ =
  0.13596324445199998` and `2.470924 x₀ = −0.11870071803600002`, imply
  values of `x₀` that differ by `6.9e-18`. POUNCE answered *primal
  infeasible* at iteration 0; HiGHS solves the model it was reduced from.
  Making the right-hand sides bit-exact, or duplicating a row verbatim, made
  it optimal again — so the redundancy was never the trigger, only its
  inexactness.
- The residual is now compared against a tolerance scaled by the magnitude
  of the terms that cancelled — the scale the rounding error actually lives
  on — so the verdict does not change when the same rows are multiplied
  through by 1000. Real conflicts are unaffected: rows that disagree by
  anything a solve could act on are still certified infeasible at presolve
  time, and the empty rows present in the *input* (no substitution, no
  rounding) keep their exact check. The emptied-inequality test `0 ≤ rhs`
  gets the same treatment.
- This is the worst failure mode available to a presolver: a redundant
  balance, an alias plus its defining equation, or a unit conversion stated
  twice is ordinary in real models, and its right-hand sides are usually
  computed rather than typed. Regression coverage in
  `crates/pounce-convex/tests/issue496_ulp_inconsistent_equalities.rs`.

### Fixed — two of three local-infeasibility exits discarded an acceptable point (#505)

- Three routes in `ipopt_alg.rs` conclude `LocalInfeasibility`: the
  convergence check's rapid detection, restoration layer 2, and the
  slow-cycle exits. Only the cycle exits consulted the acceptable-point
  stash before surfacing the verdict. The other two built the terminate
  outcome inline, so a solve that had already passed through an acceptable
  iterate — stashed, un-vetoed, sitting there precisely as a rollback
  target — discarded it and reported the hard failure instead.
- The two were found independently, months apart, because nothing tied them
  together. All three now go through `terminate_local_infeasibility`, and a
  tripwire test fails if a new site rebuilds the outcome inline.
- Upstream Ipopt does the same thing at the equivalent point: its
  `CurrentIsAcceptable()` check intercepts restoration entry and throws
  `ACCEPTABLE_POINT_REACHED` rather than proceeding from a good point.
- Inert on genuinely infeasible models: nothing is stashed unless the whole
  acceptable triplet passed, so the restore falls through to the verdict
  unchanged.

### Fixed — `solve_qp(method="active-set")` panicked across the FFI on a reversed box (#491)

- `pounce.solve_qp(..., method="active-set")` with a crossed bound
  (`lb > ub`) **panicked out of Rust into Python** — `pyo3_runtime.
  PanicException: min > max, or either was NaN` — instead of returning
  `primal_infeasible` the way `method="ipm"` does on the identical input.
  Because `PanicException` derives from `BaseException`, a caller that had
  defensively wrapped the solve in `except Exception` still lost its loop.
  Found by the nightly adversary run.
- The path: the engine's own `validate` rejects `xl > xu`, so the driver's
  first two attempts returned `NumericalFailure` and it fell through to the
  last-resort simplex-seeded attempt — where the seed was clamped into the
  inverted interval and `f64::clamp` panicked. `clamp` panics on
  `min > max`; the seed builder now uses `max`-then-`min`, so it can no
  longer be the thing that takes the process down.
- The real fix is a single screen on the variable box, `screen_variable_box`,
  run at **every** convex solve entry point — both engines' alike. Whether a
  box is empty is an input-domain question, and the answer must not depend on
  which engine the caller selected. It lives in the Rust core rather than in
  `qp.py`'s validation pass, so it holds for the raw `_pounce` bindings, the
  CLI, and direct Rust callers too. A reversed box is `PrimalInfeasible` by
  inspection — `x_i ≥ lb_i > ub_i ≥ x_i` has no solution — which is the one
  infeasibility claim the active-set driver may make without re-deriving a
  certificate. The gh #295 impossible-bound class (a *present* `+∞` lower /
  `−∞` upper) now runs through the same screen; the active-set path had no
  check for it at all.
- A crossing of `1e-9` or less is repaired rather than rejected: presolve's
  bound tightening tolerates exactly that much and can hand a driver a
  reduced problem carrying one, so the variable is fixed at its box midpoint
  and solved.
- **The IPM's own answer on a reversed box was not uniform either**, which
  the same screen fixes. Measured on `min ½x² s.t. 0 ≤ x ≤ −gap` before the
  change: crossings up to `1e-9` converged to the box midpoint and reported
  `Optimal`, crossings of `1e-6` and wider reported `PrimalInfeasible`, and
  the band between them — `1e-8`, `1e-7` — reported `NumericalFailure` at a
  `NaN` iterate. The screen keeps the two outer bands (the snap-to-midpoint
  repair *is* the answer the iteration was already reaching) and replaces the
  `NaN` band with the verdict the bands on both sides of it imply. Both
  methods now agree at every crossing width.
- Well-formed boxes are untouched: the screen returns the caller's problem by
  reference and every solve downstream of it is bit-identical.
- The comment at `python/pounce/qp.py:397` claimed the finite reversed case
  "is correctly reported `primal_infeasible` by the solver". That was true of
  the IPM only — and only outside its `NaN` band — and is what left this gap
  unexamined; it now says what each path does.

### Changed — `.nl` variable bounds reach the convex solvers as a box, not as rows (#491)

- `qp_extract` emitted each finite `.nl` variable bound as a `G` row
  (`x ≤ x_u`, `−x ≤ −x_l`) and left the solver's `lb`/`ub` empty. That was
  never *wrong* — the IPM re-expands finite bounds into exactly those rows
  internally — but it discarded the one thing only the box can carry: that
  these rows **are** a box. Bounds now go in the box on both the QP and SOCP
  extraction paths, and their multipliers come back in `z_lb`/`z_ub` instead
  of being decoded out of `z` by row position.
- Three consequences, all real. The empty-box screen above reads `lb`/`ub`, so
  a reversed bound in a model arrived as a pair of contradictory *rows* — an
  infeasibility that has to be certified numerically rather than seen. The
  active-set engine handles a box with bound statuses rather than constraint
  rows, so every `.nl` QP carried up to `2n` rows it did not need, in the one
  dimension an active-set method pays combinatorially for. And presolve
  reasons over `tlb`/`tub`, so bounds hidden in rows had to be rediscovered
  before any box reduction could fire.

### Fixed — presolve could not read a singleton row as the bound it is (#491)

- Two defects, both of which had to go before a variable box written as a pair
  of rows could be recognized as one:
  - **`−∞ − (−∞) = NaN`.** The implied bound on a column needs the row's
    activity *without* that column, and it was computed by subtracting the
    column's own contribution from the total. The moment that contribution was
    the infinite one the result was `NaN`, `val.is_finite()` was false, and the
    tightening silently did nothing — for *every* column of any row holding a
    variable unbounded on the relevant side, the singleton row `x ≤ u`
    included. Activity is now accumulated as a finite part plus a count of
    infinite terms, so the leave-one-out is exact and the rest is infinite only
    if some *other* column is.
  - **Whole-column source disjointness.** A tightening source row claimed its
    entire column, so of the pair `x ≤ u` / `−x ≤ −l` only the first was ever
    used and only one side of the box was derived. A **singleton** row now
    claims just the `(column, side)` it implies. Multi-column rows keep the
    conservative whole-column claim: their `Gᵀz` credit lands in every column
    they touch, and relaxing it there produces a postsolved point with a
    nonzero reduced cost at an interior variable (caught by
    `randomized_overlapping_tightening_roundtrip`).
- Net effect: a contradictory pair is now presolved to `Infeasible` — the
  `1e-8` case reported `Numerical failure … obj=NaN` after 169 iterations and
  now reports `Problem is primal infeasible` in 0 — and a consistent pair
  becomes the box it describes, which also drops both rows as redundant.
  Two reduction tests asserted the *old mechanism* ("the row is kept") for
  cases that now become bounds; they assert the invariant they were named for
  instead (the constraint still binds; both sides of a range survive).

### Fixed — a `NaN` iterate could leave the convex solvers (#491)

- `finite_or_failed` gates every caller-facing convex entry point
  (`solve_qp_ipm`, `solve_qp_ipm_warm`, `solve_socp_ipm`,
  `solve_qp_active_set`): a non-finite entry in `x`, `y`, `z`, `z_lb`, `z_ub`,
  or `obj` is replaced by a zero-filled `NumericalFailure`. A `NaN` in a
  returned iterate is never information — it cannot be checked against a bound,
  printed into a `.sol`, or warm-started from, and it turns every arithmetic
  downstream into another `NaN`, converting one solver's failure into the
  caller's several steps removed from the cause. The status was already
  `NumericalFailure` in the case this was written for; `obj=NaN` still reached
  the CLI's summary line. The `*_debug` entry points deliberately pass the raw
  iterate through — it is what the hook was attached to see.

### Added — LP and convex-QP presolve folds away two-variable equality rows (#494)

- A row `a₁·x + a₂·y = b` linking two variables says one of them *is* the
  other, up to a scale and a shift — an arc equality, a `Reference` alias,
  a unit conversion. Neither variable is determined by it, so the convex
  presolve's catalog had nothing that could act on it, even though it
  already had singleton-row fixing and free-column-singleton substitution.
  On a flowsheet those rows are most of the model. LP and convex QP now
  substitute one variable for the other and drop the row, iterating to a
  fixed point so **chains** of aliases collapse to a single column.
- This is the reduction #490 added for the general NLP path, reaching the
  models that path never sees: the CLI dispatches LP and convex QP to
  `pounce-convex` several hundred lines before any presolve wrapper is
  built. Both now share one planner, so they agree on what can go.
- On by default, as the rest of the convex presolve is. `qp_presolve=no`
  (or `presolve=no`) turns the whole pass off. The one-line presolve
  summary gained an `aggregated` count.
- Any bound on an eliminated variable is carried onto the survivor, so the
  reduced problem describes the same feasible set — and the postsolved
  duals still describe the *original* one. Where a reduced solve reports a
  bound force on a variable that has no such bound of its own (it inherited
  one), the force is re-attributed to the variable that actually declares
  it. Convexity is preserved by construction: the substitution is a
  congruence `P' = MᵀPM`.
- Fails closed, like Phase 6: a contradictory alias system, or one that
  would remove every column, stands the pass down and hands the model over
  untouched rather than being the first and only voice calling it
  infeasible. The conic path (SOCP, exponential/power cones, SDP, SOS) opts
  out entirely — those rows are coupled in fixed-size blocks.

### Fixed — the convex presolve's iteration trace reported a reduced-problem objective

- `QpIterate::objective` is documented to be in the original problem's
  coordinates, but the trace comes back from a solve of the *reduced*
  problem, and presolve never added back the constant its substitutions
  moved into the objective. Any `--json-detail full` report on a model
  where presolve fixed or substituted a variable carried an `iterations`
  array offset by that constant. The final objective, the solution, and
  the duals were always right; only the trace was affected.


### Fixed — a constant left in a `.nl` row body made an affine row look nonlinear (#492)

- A `.nl` writer may leave a constant on the left of a constraint —
  `x0 + x1 + 3 <= 6` rather than `x0 + x1 <= 3` — and that constant arrives
  in the row's *nonlinear-part* expression segment. POUNCE read any
  non-empty expression segment as "this row is nonlinear", which is an
  identity check, not a linearity test: the row is affine.
- The reader now folds a constant row body into the row bounds at parse.
  The body drops by `c` and each bound drops by `c` with it, so the
  feasible set, the active set, the objective, and every multiplier are
  unchanged — nothing reports a raw row body to the user, and both AMPL and
  `pounce verify` re-derive bodies from the `.nl`. The fold is by
  *evaluation*, not syntax, so a computed constant (`sqrt(9)`, `1 + 2`) is
  caught too.
- Two things this unblocks:
  - **Presolve.** The new linear-equality reduction (Phase 6, #487) only
    consumes rows tagged linear, so it declined every row with a constant
    body. A model whose equalities carry offsets now reduces like the same
    model written with the offsets in the bounds.
  - **Routing.** A model that is a plain LP apart from a *computed* row
    constant classified NLP and never reached `pounce-convex`; forcing
    `solver_selection=lp-ipm` on it was a hard error. It now classifies LP.
    (A bare literal already classified LP — the classifier's polynomial
    walk absorbed it — so only the computed form was misrouted.)
- Absent-bound sentinels are left alone. Presence is directional (#401), so
  shifting a ±1e19 sentinel would turn "no bound" into a real one; the fold
  moves a bound only when that side is actually bounded. A non-finite
  constant, or one inside an imported-function call, is left in the
  expression rather than pushed into a bound.

### Added — presolve eliminates variables determined by linear equality rows (#487)

- `presolve_linear_eq_reduction` was a registered option with no
  implementation behind it. It now runs a **variable-elimination phase**
  (Phase 6) in `pounce-presolve`, the first pass in the layer that removes
  *columns* rather than only rows. Off by default; requires `presolve=yes`.
- The measurement in #487: a solvent-extraction NMPC flowsheet declares
  23,681 variables and 23,138 equality rows, and Pyomo's NL-v2 writer — with
  the linear presolve its `ipopt_v2` interface turns on by default — hands
  the solver 10,284 columns and 9,741 rows. POUNCE reached the solver at
  full size on every path. It now recognises the same three shapes, iterated
  to a fixed point so chains propagate: variables fixed by equal bounds,
  singleton rows `a·x = b`, and two-variable rows `a₁·x + a₂·y = b`.
- The two-variable shape is the one that mattered and the one nothing in
  POUNCE could reach. The auxiliary-equality pipeline (#53) eliminates
  *determined* square blocks; a row linking two otherwise-free interior
  variables — an arc equality, a `Reference` alias, a unit-conversion link —
  determines neither of them and so survived. Phase 6 substitutes one for
  the other with no anchoring requirement.
- Implemented in `pounce-presolve` (option (b) of #487's two candidate
  homes) rather than by switching the Pyomo path onto NL-v2's presolve, so
  the CLI, GAMS, the C interface, and Pyomo all get the same reduction from
  one implementation — and so the sensitivity session is not asked to
  thread a third variable space through index maps whose correctness rests
  on there being exactly two (the #450 hazard).
- Duals for the consumed rows are **recovered, not zeroed**. Taken as a
  whole the recovery is a square sparse solve; taken one elimination at a
  time in reverse order it is triangular, because every other row of the
  problem-as-it-stood-then survives that step. So it is a linear sweep over
  the elimination forest, not a factorization. `.sol` and JSON solution
  blocks come back at the original model's length, in the original order,
  and can still be read positionally by AMPL and Pyomo.
- A transferred bound's multiplier is reported on the variable that
  **declared** the bound, not on the survivor that inherited it (#493). The
  plan records where each reduced bound came from; postsolve rescales the
  multiplier by the substitution coefficient `α` and sends it to the other
  side of the box when `α < 0`, so `ipopt_zL_out` / `ipopt_zU_out` match a
  no-presolve solve. Where the survivor's own bound is active *as well*, the
  split is genuinely non-unique and the multiplier stays on the survivor —
  still a valid KKT point, and documented as such in `docs/src/options.md`.
- A bound multiplier the reduced solve never *produced* is now recovered
  too (#495). Re-attribution can only move a multiplier that exists, and
  accumulated bound transfers can leave a survivor's reduced box as a
  single point — a fixed variable, which the solver drops as a parameter,
  so it reports `z_l = z_u = 0` even though the cluster it stands for is
  sitting on a bound carrying load. Stationarity in the original space was
  off by exactly the missing multiplier, and `ipopt_zL_out` /
  `ipopt_zU_out` came back empty where Pyomo and AMPL expect a value.
  Postsolve now reads the residual its row-multiplier sweep cannot close —
  which is precisely that multiplier — and places it on the declared bound
  the point is resting on, the survivor's own or, through the same `α`
  rescale, the column the bound was borrowed from. A residual with no
  active declared bound to carry it is left alone: trading a stationarity
  error for a complementarity error is not a repair. A column the *model*
  declares fixed is untouched, since the solver drops those with or without
  the reduction and a no-presolve solve reports nothing for them either.
- One documented caveat remains, in `docs/src/options.md`: a contradictory
  equality system makes the pass stand down entirely rather than be the
  first and only voice calling a model infeasible.


### Fixed — the C working-set API validated nothing structural, and reported the wrong row order (#484 follow-up, round 4)

- `IpoptSetWarmStartWorkingSet` range-checked its status codes and stopped
  there. `POUNCE_WS_FIXED_OR_EQ` asserts `x_L == x_U` for a variable, or
  `g_L == g_U` for a row; `POUNCE_WS_AT_LOWER` / `_AT_UPPER` assert the
  bound being sat on is finite. Those are claims about the *model*, not
  guesses about the active set, and the model is right there to check them
  against. Passing a false one — `FIXED` for a variable whose bounds differ
  — was accepted, `TRUE` was returned, the solve over-constrained itself,
  and a **convex** program came back with the wrong optimum. Found by the
  property-based probe in `adversary/fuzz/`; such codes are now rejected
  with `FALSE`, which callers already handle.
- Adding that validation immediately rejected pounce's *own*
  `IpoptGetWorkingSet` output, which exposed the larger defect underneath:
  **the statuses were reported in the SQP's internal row order, not the
  caller's.** The SQP works on a reordered constraint vector — equalities
  first, inequalities after — and both entry points copied that vector
  positionally against the caller's indices. On HS071, whose rows are
  `[x₀x₁x₂x₃ ≥ 25, Σxᵢ² = 40]`, `IpoptGetWorkingSet` returned
  `[Equality, AtLower]`: exactly reversed. The documented
  get → set round-trip was therefore feeding every warm start a working set
  with its rows swapped. (Silently: a wrong working set is a wrong *hint*,
  and hints are answer-preserving, so nothing downstream ever complained.)
- Both entry points now translate between the caller's indexing and the
  internal one. The permutation is a pure function of the bounds — a row is
  an equality iff both sides are finite and equal, a variable is dropped iff
  `x_L == x_U` — so it is reconstructed in the C layer with no plumbing of
  `BoundClassification` out of `pounce-nlp`. Fixed variables, absent from
  the internal problem, are reported as `POUNCE_WS_FIXED_OR_EQ`.
- The GAMS link's state-file warm start (`gams/gams_pounce.c`) goes through
  these two calls and was reading back mis-ordered statuses; it is fixed by
  the same change, with no edit on its side.
- Regression coverage asserts the *values*, not just that the codes are in
  range: HS071's converged set must read `bounds = [1,0,0,0]`,
  `constraints = [1,3]`. Confirmed to fail (`[3,1]`) without the change.

### Fixed — `Optimal` at infeasible points, and a hard error, on the QP cold paths (#484 follow-up, round 3)

- Two defects the property-based probe (`adversary/fuzz/`) had flagged as
  open. Both predate #484 — they reproduce identically on the pre-#484
  code — and both are now closed, taking the probe to a clean run: **zero
  invariant violations over 400 generated instances**, against 18 before
  this series began.
- **The cold fast paths were never audited.** `solve` runs a feasibility
  audit (M5) on solves that converge to a constraint-violating point and
  label it `Optimal`, but the audit guarded only the `solve_general`
  branch. `solve_equality_only`, `solve_box_constrained` and
  `solve_equality_plus_bounds` returned straight to the caller. The
  smallest possible infeasible QP falls in that gap: `aᵀx = c₁` and
  `aᵀx = c₂` with `c₁ ≠ c₂` is all-equality with a box, so it routes to
  `solve_equality_plus_bounds` — and came back `Optimal` at a point
  violating both rows by 2.9, at every tolerance from 1e-9 to 1e-2. The
  audit is now a helper (`audit_and_repair`) applied to every path.
- **The rank-deficiency prune was single-shot.** `cold_general_initial`
  prunes a rank-deficient equality set to an independent subset and
  retries, but `independent_active_subset` is a *numerical* rank test
  whose answer depends on the shift the factorization settled at, so the
  pruned subset can itself be rejected at the next δ. It was: four
  equality rows pruned to two, and the retry's own masked-deficiency guard
  then found only one of those two independent. The retry's `?` propagated
  `LinearSolverFailure("pinned KKT constraint block is rank-deficient …")`
  to the caller — a hard error on a QP the elastic path handles. The prune
  now iterates while the subset strictly shrinks (so it terminates), and
  on persistent failure returns the function's existing "fall through to
  elastic" signal instead of an error. A caller with a good next thing to
  try should not be handed a linear-algebra failure.
- Regression tests use the probe's *exact* instances, not tidied-up
  equivalents. Hand-built versions of both do not reproduce: the free
  variables, the indefinite Hessian and (for the rank case) the 1e6 row
  scaling are all load-bearing, and a uniformly-scaled bounded version with
  a PSD Hessian passes on the unfixed solver. Both were confirmed to fail
  without the change.

### Fixed — penalty bias was read as an infeasibility certificate (#484 follow-up, round 2)

- The previous round stopped the l1-elastic path certifying infeasibility
  off a *nonconvex* phase-1, by adding a convex feasibility phase-1 whose
  verdict is sound. Property-based probing (`adversary/fuzz/`) showed that
  fix was incomplete — 6/257 feasible instances were still certified
  infeasible — and that the reason was a flaw in the new code, not a
  leftover of the old one.
- The convex phase-1 minimizes `½‖x − r‖² + γ‖v‖₁`. The proximal term
  competes with the penalty, so a *feasible* instance still leaves a
  residual: `(x̂, 0)` costs `½‖x̂ − r‖²`, hence `s* ≤ ‖x̂ − r‖²/(2γ)`. With
  `γ = 1e6` and an ordinary box that ceiling is ~1e-5 — four orders above
  `feas_tol`. The phase-1 was converging correctly and stopping a few 1e-6
  short; the code then judged it against `feas_tol`, found it wanting, and
  issued the certificate. A penalty artefact was being read as a Farkas
  proof.
- The bias is now computed rather than tripped over. `penalty_bias_bound`
  evaluates `D²/(2γ)` from the box, and a residual may only certify once
  it exceeds *both* that bound and a scale-relative noise floor
  (one part per million of `max(‖A‖∞, ‖b‖∞)` — below which "infeasible"
  and "feasible up to roundoff" are not distinguishable). Where the
  residual sits under the bound, γ is escalated — aimed at the measured
  shortfall rather than cranked, since overshoot leaves the subproblem too
  hard to converge and throws the certificate away — and the phase-1
  re-solves from the previous iterate.
- Certifying now also *requires* the convex phase-1 to have converged.
  Nonconvex residual slacks and "phase-2 failed to improve" are both
  ignorance, not proof; neither may speak to infeasibility.
- Free variables are handled explicitly. An unbounded box gives no `D`, and
  returning infinity there would mean never certifying a QP with a free
  variable — trading one wrong answer for another. Those coordinates take a
  surrogate from the distance the solve actually travelled. That much is
  judgment rather than theorem, and it is confined to the coordinates where
  the theorem has nothing to say; the fuzz shows the answer is insensitive
  to it across three orders of magnitude.
- Measured over 400 generated instances (65% feasible-by-construction with
  an attached witness, 35% infeasible by exact arithmetic; every instance
  independently re-decided by `scipy.optimize.linprog`/HiGHS):

  | | before #484 | round 1 | now |
  |---|---|---|---|
  | feasible QPs certified infeasible | 14 | 9 | **0** |
  | genuine infeasibilities certified | 118/143 | 118/143 | 108/143 |

  The 10 lost certifications are the price of demanding a sound proof: they
  become a non-committal status rather than a confident one. That is the
  right direction — a false certificate is a wrong answer, a non-committal
  status is a weaker one — but it is a real cost, and it is not tunable:
  raising the phase-1 iteration budget 5× recovers one of them at 3× the
  wall time.
- Regression test: `penalty_bias_is_not_an_infeasibility_certificate`, built
  from an instance the probe found, with its feasibility witness checked
  arithmetically in the test rather than assumed.

### Fixed — active-set SQP declared feasible problems infeasible when started near a solution (#484 follow-up)

- HS071 cold-starts fine from `(1,5,5,1)`, but nudge the start to
  `x* + 1e-6·e₁` and the SQP died at iteration 0 with
  `Infeasible_Problem_Detected` — on a problem whose feasible set had
  not moved. Starting near a solution is the entire premise of warm
  starting, so this stood directly behind the C-API defect above: fixing
  the discarded iterate let callers reach the solution's neighbourhood,
  which is where this waited.
- The verdict came from the step QP's l1-elastic phase-1 in
  `pounce_qp::solver::solve_elastic`, and two independent defects had to
  line up to produce it.
- **The residual-slack certificate was applied to nonconvex problems.**
  It is a *global* claim — that the minimal l1 infeasibility is positive
  — but an active-set solve of a nonconvex elastic problem returns a
  local KKT point. The SQP's default Hessian is the exact ∇²L, indefinite
  here (three zeros on its diagonal), and γ = 1e6 amplifies a ~1e-7 slack
  into ~0.1 of apparent objective, so phase-1 settled at a far box vertex
  carrying a cancelling `(v_l, v_u)` pair and missed `feas_tol` by a
  factor of two — 1.95e-9 against 1e-9 — on a QP with points feasible to
  slack 1.66. Before certifying, `solve_elastic` now runs a
  feasibility-only phase-1 with the caller's objective replaced by
  `½‖x‖²`. That subproblem is strictly convex however indefinite `H` is,
  so its verdict is sound, and feasibility depends only on `A`, the row
  bounds and the box, so nothing about the question is lost. A feasible
  point it finds serves as both a phase-2 seed and a witness; the solver
  no longer announces infeasibility while holding one.
- **The phase-2 recovery seeded a cold working set.** A cold set marks
  every row `Inactive`, equalities included, and the warm inner loop
  cannot pull an Inactive equality into the working set — so the row was
  never enforced. The recovery that exists to prevent false certificates
  was therefore inert on any QP with an equality row: it converged to
  `Optimal` at a point violating HS071's equality by 7.8, failed its own
  feasibility check, and fell through to the certificate. Recovery seeds
  are now classified (equalities active, rows and bounds at their
  boundary snapped) instead of handed a cold set.
- Genuine infeasibility is still certified — the existing certificate
  tests are unchanged — and truly infeasible QPs now pay one extra convex
  phase-1 before the verdict.
- Regression tests: `crates/pounce-algorithm/tests/sqp_near_solution_start.rs`
  sweeps 8 perturbation sizes × 4 coordinates around HS071's solution
  (a single size is not enough — `1e-6` and `1e-3` failed while `1e-8`
  and `1e-1` happened to survive), and
  `nonconvex_step_qp_near_nlp_solution_not_false_infeasible` in
  `pounce-qp` pins the exact step QP with an arithmetically verified
  feasibility witness.

### Fixed — `IpoptSetWarmStartWorkingSet` discarded the caller's iterate (#484)

- `IpoptSetWarmStartWorkingSet` eagerly built a full `SqpIterates` and
  hard-coded its primal to `x = 0`. `SqpAlgorithm::optimize_with_warm_start`
  treats a supplied iterate as *the* starting point — it only consults the
  NLP's `get_starting_x` on the cold branch — so those zeros silently
  replaced the `x` buffer passed to `IpoptSolve`. The one call documented as
  "supply a warm-start working set" was also, invisibly, resetting the
  iterate to the origin.
- The effect is not a slower warm start, it is a wrong answer: on any
  problem whose bounds exclude the origin the warm solve returned
  `Infeasible_Problem_Detected` at iteration 0 and wrote zeros back into
  `x`. On HS071 (`1 ≤ x ≤ 5`), warm-starting *at the solution with the
  exact active set* failed where the same solve without the call converged
  in one iteration. An all-inactive working set — semantically a cold start
  — failed identically, so the working set's contents never mattered;
  making the call at all was the defect.
- The C layer now stages the working set alone and merges it with the real
  starting point inside `IpoptSolve`, which is the first moment the iterate
  is known. This is what `pounce.h` already claimed, so no new symbols and
  no API change. `IpoptSolveWarmStart`, which delegates to the same path,
  is fixed with it — as is the native GAMS link's state-file warm start.
- Initial multipliers now reach the SQP too, under
  `warm_start_init_point=yes` — upstream Ipopt's contract for `mult_g` /
  `mult_x_L` / `mult_x_U` being inputs rather than pure outputs. With the
  option off (the default) they stay strictly outputs, so callers passing
  uninitialized buffers cannot seed the solver with garbage. The SQP packs
  them signed as `lambda_x = z_l − z_u`, matching the Python path.
- Regression tests in `crates/pounce-cinterface/tests/warm_start_iterate.rs`
  reproduce the reporter's four-case HS071 driver (cold / control / warm /
  warm-with-inactive-set) and assert the converged iterate, not merely a
  status code. The Rust and Python APIs were never affected — Rust callers
  populate `SqpIterates.x` themselves, and the Python path was corrected in
  gh#57.
### Changed — options naming unimplemented features are refused, not ignored (#483 follow-up, continuing #191)

- The option registry is a faithful port of Ipopt's, so an `ipopt.opt`
  written for Ipopt parses unchanged — and ~200 of the registered knobs
  were silent no-ops, because registering an option says nothing about
  implementing it. #191 fixed the half where the *feature* runs and only
  the read site was missing; it explicitly scoped out "feature genuinely
  unimplemented — expected no-ops". This closes that half.
- Setting an option that configures a feature POUNCE does not have now
  fails the solve — exit 2 from the CLI, `Invalid_Option` for library
  callers — naming the option, the feature, and the alternative. Covered:
  the Chen-Goldfarb (CG-penalty) / inexact-Newton line search, derivative
  approximation by finite differences, linear-dependency detection, the
  per-iteration NaN/Inf derivative check, multiplier recalculation,
  a selectable constraint-violation norm, magic steps, bound replacement,
  the L-BFGS augmented-system variants, the linear-variable hint, reading
  options from a file, skipping the finalize callback, the dynamic HSL
  loader, `suppress_all_output` and `debug_print_level`.
- **An explicitly-set default is still allowed.** A generated `ipopt.opt`
  spells out defaults and `dependency_detector=none` asks for nothing;
  only a value differing from the registered default is refused. Without
  this the change would break the compatibility the registry provides.
- **Caching hints warn instead of failing.** `grad_f_constant`,
  `hessian_constant`, `jac_c_constant`, `jac_d_constant` are hints POUNCE
  does not exploit; ignoring them costs evaluations, never correctness, so
  blocking the solve would be the worse trade.
- Membership was established per option, not inferred: the name must
  appear in no crate source outside the registry (whole-word — so
  `penalty_max` is not counted present because `l1_penalty_max` exists)
  *and* the feature itself must be absent. Options whose feature runs and
  whose read site is merely missing — the restoration knobs, the
  `limited_memory_*` tail, the corrector selectors — are deliberately
  excluded and still solve; refusing them would fail solves whose answers
  are correct today. A unit test pins that boundary.
- The check runs before solver routing, so a model that classifies as a
  convex QP gets the same verdict.

### Fixed — `fast_step_computation` is wired, not refused (#483 follow-up, #191 round 2)

- `PdSearchDirCalc` has owned this flag — skip the search-direction
  residual check, allow an inexact linear solve — since it landed, and
  consumes it at two sites; only the option's read site was missing, so
  setting it did nothing. It now reaches the builder, default unchanged
  (`no`).
- It briefly sat in the refusal table above, added by hand against that
  table's own membership rule (its name *does* appear in the sources), so
  it would have failed a solve POUNCE can serve. The boundary test now
  pins it.

### Added — the derivative checker (`derivative_test`) now exists (#483 follow-up)

- All five `derivative_test*` options were registered and none was ever
  read, so `derivative_test=first-order` ran no test and printed nothing.
  That is the worst shape an unimplemented option can take: a *checker*
  that silently checks nothing reports success by omission — a user with
  a hand-written `eval_grad_f` turns it on, sees no complaints, and
  concludes the gradient is right.
- Implemented as a port of upstream's `TNLPAdapter::CheckDerivatives`:
  `first-order` compares `eval_grad_f` / `eval_jac_g` against finite
  differences at the bound-projected starting point, `second-order` adds
  `eval_h` (checked one multiplier block at a time), `only-second-order`
  does the Hessian alone. `derivative_test_perturbation`,
  `derivative_test_tol`, `derivative_test_first_index` and
  `derivative_test_print_all` are all honored.
- Two checks upstream does not make, because neither is reachable by a
  value-by-value comparison: an entry whose finite difference is nonzero
  but which the **sparsity structure omits** (a derivative the solver can
  never see), and taking the perturbation **downward** when stepping up
  would leave a variable's box, so a model using `sqrt`/`log`/`1/x` is
  not evaluated outside its own domain by its own checker.
- Advisory, like upstream: suspicious entries are reported and the solve
  continues. The report goes to stderr, so it survives `print_level=0`
  and stays out of `--json-output`'s stdout. It runs on both solver
  routes — the convex dispatch never reaches the NLP path's copy, and the
  check is about the model, not the engine.
- `check_derivatives_for_naninf` (a per-iteration NaN/Inf guard) remains
  unimplemented and is documented as such.

### Fixed — `linear_solver` accepted every backend name and silently ran FERAL (#483 follow-up)

- The option's registered value list is a faithful port of upstream
  Ipopt's — `ma27`, `ma57`, `ma77`, `ma86`, `ma97`, `mumps`, `pardiso`,
  `pardisomkl`, `spral`, `wsmp`, `custom`, `feral` — so an `ipopt.opt`
  written for Ipopt parses unchanged. POUNCE implements two of them, and
  the resolver mapped everything else through a `_ =>` arm to FERAL. So
  `linear_solver=ma97` "worked": a successful run using a backend the
  binary does not contain, and a benchmark comparing linear solvers that
  compared FERAL against itself.
- Selecting an unimplemented backend is now refused — exit 2 from the CLI,
  `Invalid_Option` for library callers — with a message naming the backend
  and what to use instead. The names stay registered, so an Ipopt
  `ipopt.opt` still parses and gets a precise complaint rather than
  "invalid value". Their per-backend tuning knobs (`ma97_scaling`,
  `mumps_pivtolmax`, `pardiso_*`, …) are registered for the same reason
  and are now unreachable.
- **The registered default is now `feral`**, diverging from upstream's
  `ma57` on purpose: a default has to name a solver the binary actually
  contains. Under the old default a pure-Rust build advertised MA57 to
  every `print_user_options` dump and banner-adjacent consumer while
  running FERAL, and — the behavioural half — an **HSL-enabled build used
  MA57 without being asked**. If you build `--features ma57` and want it,
  select it explicitly with `linear_solver=ma57`. This also removes the
  explicit-vs-default special case the banner, the wheel's banner, and
  the refusal guard each had to carry.
- Not a failure: **explicit `ma57` without the feature** falls back to
  FERAL with the banner saying so (`FERAL (ma57 requested but not
  compiled)`) — a reported substitution, not a hidden one.
- The check runs before solver routing, so it does not depend on whether
  the model classifies into the NLP path or the convex one.

### Fixed — a maximize request was silently dropped on the convex LP/QP route (#483 follow-up)

- `obj_scaling_factor` is upstream's spelling for maximization (the IPM
  minimizes `factor·f`, so a negative factor maximizes `f`). The convex
  solvers in `pounce-convex` equilibrate internally and never read the
  option — and every LP / convex-QP model routes to them by default. So
  `min (x−3)²` over `x ∈ [0,1]` with `obj_scaling_factor=-1` returned
  `x = 1`, the **minimizer** of the objective the user asked to maximize,
  reported as `Optimal Solution Found`. The same file under
  `solver_selection=nlp` answered correctly with `x = 0`.
- A negative factor now declines the convex fast path under
  `solver_selection=auto` (routing to the NLP path, which honors it) and
  is **refused** — exit 2, with an explanation — under an explicit convex
  `solver_selection`, where the alternative is not a skipped extra but a
  wrong answer.
- A *positive* factor is unaffected: it only rescales conditioning, the
  convex path reports natural units either way, and both paths agree.

### Fixed — `honor_original_bounds` was registered but never read (#483 follow-up)

- `bound_relax_factor` (default `1e-8`) widens the variable box before the
  solve, so a solution pinned to a bound is reported just outside it:
  `min (x−3)² + (y+2)²` over `x ∈ [0,1]`, `y ∈ [−1,1]` returned
  `x = 1.00000000937`, `y = −1.00000000875`. Upstream registers
  `honor_original_bounds` to project that back; pounce registered it and
  never read it, so there was no way to get a point inside the declared
  box — and the value flows on into a downstream `sqrt(1−x)`, a domain
  assertion, or a Pyomo `Var` whose bounds it is loaded back into.
- The option is now honored on both routes a caller can read the solution
  from: the `.sol` / JSON primal (via the CLI's converged-iterate hook)
  and `TNLP::finalize_solution` (pounce-py, the C interface, any Rust
  TNLP). The default stays `no`, matching upstream, and — as upstream
  documents — the summary's constraint-violation and complementarity
  figures remain those of the non-projected point.

### Fixed — user scaling was a silent no-op from Pyomo, and the core silently discarded `x_scaling` (#483)

- **The `scaling_factor` Suffix now reaches the solver.** Tagging the
  standard Pyomo Suffix and setting `nlp_scaling_method=user-scaling` —
  the workflow that works with Ipopt through ASL — produced *no scaling
  at all* and no message saying so. `NlTnlp` never implemented
  `get_scaling_parameters`, so the `.nl`'s `scaling_factor` suffix
  segments were parsed and then ignored, and pyomo-pounce had no scaling
  code of its own. The option was accepted and meant "none". Objective
  and constraint factors now apply on both paths: the ASL/subprocess
  solve reads the suffix segments, and pyomo-pounce's in-process
  sensitivity path installs them via `Problem.set_problem_scaling`.
  Untagged components, and components tagged `0` (AMPL's suffix
  default), are unscaled; entries on inactive constraints and fixed
  variables are skipped; a container entry expands to its members.
- **Per-variable factors are refused instead of dropped.** POUNCE models
  objective and constraint scaling only, and `scale_user_supplied` ended
  in `let _ = use_x_scaling;` — so a caller who supplied variable factors
  got a converged answer to a differently-conditioned problem than the
  one described, with the objective and constraint factors applied and
  the variable ones gone. A non-unit `x_scaling` now fails the solve with
  `Invalid_Option` and an explanation, `pounce.Problem.set_problem_scaling`
  raises, and pyomo-pounce raises naming the variables. An all-ones
  request asks for nothing and still solves. Modelling variable scaling
  in the core is staged work tracked on #483.
- **No silence anywhere else on the path.** `nlp_scaling_method=user-scaling`
  on a model that classifies as LP/QP/SOCP used to route to
  `pounce-convex`, which equilibrates internally and never reads the
  scaling callback; `solver_selection=auto` now declines that fast path
  so the scaling is honored, and an explicit `solver_selection` warns.
  pyomo-pounce warns when `user-scaling` is requested with no
  export-enabled `scaling_factor` Suffix to apply.
- The sensitivity accessors needed no code: `natural_units_conj` already
  translates `df`/`dc`/`dd` for any scaling method. That is now proven
  rather than assumed — `covariance`, `information`, `gradient`,
  `estimate`, `wrt=` blocks, the retain-only path, the classifier's
  statuses, and the fixed-variable composition are each checked against
  unscaled ground truth with user scaling engaged.

### Fixed — `inf_pr` reported the internal reformulation, not the original NLP (#476)

- The `inf_pr` iteration column printed `max(‖c‖∞, ‖d − s‖∞)` — the
  violation of Ipopt's internal *slack* reformulation — in **both**
  `inf_pr_output` modes. Upstream's default, `original`, prints the true
  violation of the user's own rows. POUNCE registered the option and then
  ignored it.
- The two diverge exactly when the slack drifts from `d(x)`: `s` is
  confined to `[d_l, d_u]`, so `d = s + (d − s)` clears a lower bound
  however large the gap grows. On a model that is all inequalities the gap
  *is* the whole number. On Mittelmann's `robot_a` POUNCE reported a
  constraint violation of `2.79e+04` at iterates where every original row
  was satisfied and Ipopt printed `0.00e+00` — a feasible point that read
  as badly infeasible.
- The column now matches `ipopt` 3.14.19 digit for digit on `robot_a`,
  including the two spikes (`7.15e+04`, `2.40e+04`) where the rows really
  are violated. `inf_pr_output=internal` still selects the old quantity.
- Display only. The filter's `theta`, the barrier-parameter strategies and
  the convergence test keep the internal measure — that split is upstream's.
  The end-of-run "Constraint violation" summary line is **deliberately not
  changed**: it is bound to "Overall NLP error", which *is* the convergence
  gate, so reconciling it means deciding whether convergence should be
  judged on the original NLP — a behaviour change for every model rather
  than a reporting fix. Tracked in #476.

### Added — pyomo-pounce: `release_kkt()`, the exit of the retention story (#475)

- The held KKT factorization can now be dropped on demand:
  `release_kkt(model)` frees the factor's memory immediately and
  returns whether anything was held. Declarations and a prior
  `retain_kkt()` are untouched, so the next solve keeps its factor
  again; after release the accessors raise their no-session error.
  Release drops the model's hold, not a result's: a `Covariance` or
  `Information` with a pending `conditioned_on`, and a `Gradient`,
  each hold their own reference and keep working across the release.
  The retention policy (three ways to keep, one way to release) is
  stated in one place in the docs.

### Performance — shared `.nl` subexpressions are evaluated once per sweep (#476)

- Constraint values (`eval_g`) now go through a problem-wide tape with a
  **shared prelude**: every `.nl` defined variable (`V` segment) referenced
  by two or more summands is evaluated once per sweep instead of once per
  reference. Previously each summand got an independent flat tape, so a
  defined variable feeding many rows was re-evaluated for each of them.
- Invisible on most models, dominant on constraint-heavy ones. Mittelmann's
  `robot_a` (n = 1001, m = 52013, 12003 defined variables each feeding 13
  rows) went from 3.6M to 894k tape ops per constraint sweep — 4.0x less
  arithmetic, on the line search's inner loop. End to end: **22 % faster**
  on `robot_a`/`robot_c`, with a bit-identical iteration trajectory.
- `eval_g` / `eval_f` also stopped allocating a `Vec` per summand (~148k
  allocations per call on `robot_a`, ~20 % of `eval_g`); they reuse the
  scratch arena `eval_jac_g` / `eval_grad_f` already used.
- Both are transparent: models using opcodes the shared-prelude path does
  not support (comparisons, AND/OR/NOT, if-then-else, min/max lists,
  external functions), and models with nothing shared to hoist, keep the
  previous path. All 42 `.nl` fixtures in the tree produce identical
  status, objective and iteration counts.
- `benchmarks/mittelmann/gen_robot_nl.py` reproduces `robot_a`/`b`/`c` as
  `.nl` without an AMPL licence. Full investigation, including why the
  iteration count is inherited from Ipopt rather than a POUNCE weakness:
  `dev-notes/research/robot-abc-per-iteration-cost.md`.

### Fixed — `NlProblem` can be shared across threads (#477)

- `NlProblem`, `DenseLU` and `SparseLU` were `#[pyclass(unsendable)]`,
  which gave each instance pyo3's per-object thread affinity. Touching
  one from a thread other than its creator raised a Rust
  `PanicException`, and merely *collecting* one on another thread wrote
  an unraisable `RuntimeError` and leaked the payload — the latter fires
  for code that never used an instance cross-thread at all, since
  whichever thread runs the GC inherits every object it frees.
- `PanicException` derives from `BaseException`, not `Exception`, so it
  slipped past every ordinary `except Exception` in a host application.
  In a threaded branch-and-bound host this did not surface as an error:
  it surfaced as a **wrong answer** — a shared `NlProblem`-backed
  evaluator reported `infeasible` on a model the reference evaluator
  solved to optimality, and a false `infeasible` from a global optimizer
  is a wrong certificate.
- The marker was a conservative default, not a physical constraint:
  `NlTnlp` is `Send` (its CSE nodes went `Arc` for #126's batched
  solving) and the LU factors are owned matrix data. All three classes
  are now sendable, so one evaluator can be built on one thread and used
  or dropped on any other. Concurrency is still the GIL's — these calls
  serialize rather than overlap — so the win is one shared tape instead
  of one per worker, and the end of the `threading.local` ceremony hosts
  needed to work around it. `#[test]`s pin `Send` on all three, and the
  Python suite covers cross-thread evaluation, `variant`, batch solving,
  and foreign-thread collection.
- `Solver`, `QpFactorization` and `QpSensitivity` stay `unsendable`:
  those hold `Rc`-based Ipopt state and non-`Send` linear-solver trait
  objects, so their affinity is real rather than defensive. Each class
  now says so at its definition, and `docs/src/python.md` documents which
  objects cross threads and which do not.

### Fixed — every eigendecomposition now returns sign-pinned eigenvectors (#471)

- An eigenvector's sign is arbitrary: `v` and `-v` are equally valid,
  and which one came back was decided by the arithmetic that produced
  it — LAPACK's build convention in `pyomo-pounce`, the Jacobi rotation
  order in the Rust kernel. The same model, data, and POUNCE version
  could report `v` on one machine and `-v` on another. Since the
  documented use is reading the eigenvector as a *direction* ("the
  parameter combination the data cannot pin down"), the reported
  direction — and anything that steps along it on a nonlinear model —
  was not reproducible.
- One convention now holds everywhere: **the largest-magnitude
  component of each eigenvector is positive**, ties broken by the
  earliest row. Applied at both sources — `pounce_linalg::symmetric_eigen`
  (the single Rust eigensolver, so every surface fed by it inherits it:
  `SensResult::reduced_hessian_eigenvectors`,
  `info["reduced_hessian_eigenvectors"]`, the CLI's
  `_red_hessian_eigenvectors` JSON block, `ReducedHessian.eigenvectors`
  from both the Rust and Python QP sensitivity APIs) and
  `pyomo-pounce`'s `Covariance.eigen()` / `Information.eigen()`.
- Eigenvalues, spans, and any use of the eigenvectors as a subspace are
  unchanged; internal uses (matrix reconstruction, projectors, Rayleigh
  quotients, null-space bases) were already sign-invariant. The sign is
  all this fixes — a repeated eigenvalue still leaves the basis within
  its eigenspace arbitrary, which the API docs and
  `docs/src/sensitivity.md` now say.
- Notebook 31 pinned the sign by hand to keep its committed output
  stable; that workaround is gone, since the library now guarantees it.

### Added — pyomo-pounce: `retain_kkt()`, factor retention without declarations (covariance roadmap item 4, #262)

- `retain_kkt(model)` keeps the solve's KKT factorization with nothing
  declared, so `covariance(model, wrt=block)` and
  `information(model, wrt=block)` work on any block without a declared
  default: the MHE case, where the arrival state and the parameters
  are each queried by `wrt=` and neither is THE fitted set.
  `covariance(model)` with no block stays an error (no default to
  reduce onto), an undeclared solve without the call pays nothing
  exactly as before, and retain beside declarations changes nothing
  (all four rows of the roadmap's table are tested). The retain intent
  follows `model.clone()` through the registry's deepcopy.
- The retention policy is now stated in one place (docs and the
  `retain_kkt` docstring): declarations keep the factor, `retain_kkt()`
  keeps it without them, and a result object's unread lazy
  `conditioned_on` keeps the session alive until first access.
- The no-session errors from both accessors name `retain_kkt()` as the
  declaration-free route.
- Demo notebooks for the whole roadmap, written for a reader new to
  NLP: 31 (information and identifiability: the poorly identified
  fit, `eigen()` naming the combination the data cannot determine,
  and zero variance versus finite information at a bound) and 32 (one
  solve, many questions: `wrt=` marginals, confidence and prediction
  bands on undeclared prediction variables, `conditioned_on`, and
  `retain_kkt()` with
  nothing declared), both committed executed.

### Added — Python: in-memory model construction, Hessian-vector products, and an `erf` tape op (#469)

Everything below already existed in the Rust core; the gap was that none
of it was reachable from Python, which forced a frontend with its own
expression DAG (discopt) to round-trip through a temporary `.nl` file.

- **`pounce.build_nl_problem(...)` + `pounce.NlExpr`** — build the
  expression DAG directly and hand it to the AD tape, with no `.nl` file
  anywhere in the loop. `NlExpr` supports the Python arithmetic operators
  (numbers accepted on either side) plus method-form transcendentals
  (`sqrt exp log log10 sin cos tan asin acos atan sinh cosh tanh asinh
  acosh atanh erf`) and static-method nodes for the multi-argument /
  control-flow cases (`sum`, `atan2`, `min`, `max`, `compare`, `select`,
  `logical_and` / `logical_or` / `logical_not`). The result is the same
  `NlProblem` class `read_nl` returns, with the same evaluator surface,
  and it feeds `solve_nlp_batch`.
  The round trip this replaces was not just slower but *lossy*: `.nl`
  writers commonly refuse `atan2` (no two-argument funcall path) and
  `min`/`max` (they force a DNLP model type), and AMPL has no `erf`
  opcode at all — yet the tape differentiates all three natively.
- **`pounce.parse_nl_text(text, var_names=None, con_names=None)`** — the
  same parser `read_nl` uses, fed a string instead of a path, for a
  frontend that generates `.nl` in memory. Names are passed explicitly
  since there are no sibling `.col` / `.row` files to read.
- **`NlProblem.hessian_vector_product(x, v, lam=None, obj_factor=1.0)`**
  — matrix-free `(obj_factor·∇²f + Σᵢ lamᵢ·∇²gᵢ)·v`, one
  forward-over-reverse AD pass per tape seeded with `v` directly.
  `hessian(...)` runs one such pass *per Hessian color* and decodes the
  compressed columns, so the matrix-free call is cheaper by roughly the
  chromatic number of the coloring — the operator a Newton–Krylov /
  truncated-CG step wants on a model where `∇²L` is impractical to form.
  Available on every `NlProblem` regardless of how it was built.
  `v` may be a dense length-`n` vector (ndarray of any dtype or stride,
  list, sequence), a dense `(n, k)` block of directions, or any SciPy
  sparse vector / `(n, k)` matrix; the result matches the input's shape.
  A sparse `v` is densified on the way in and all-zero directions are
  skipped, so a mostly-empty block costs only the live columns — the
  sparsity that pays is the model's, and every pass is O(tape ops), never
  O(n²), whichever way `v` arrives. The block form shares one forward
  sweep per tape across all `k` directions rather than repeating it, which
  is what makes `p.hessian_vector_product(x, np.eye(n))` a reasonable way
  to densify the Hessian. The result is always dense: `∇²L·v` is dense in
  general even when both factors are sparse, and the sparse Hessian itself
  is already available as `hessian_structure()` + `hessian(x)`.
  Backed by `NlTnlp::hessian_vector_products` on the Rust side.
- **`Erf` tape op** (`TapeOp::Erf` / `UnaryOp::Erf`), the one operator in
  the gap analysis that is not decomposable into ops pounce already had.
  Value via `libm` (the rust-lang port of musl's libm — the classic
  Abramowitz–Stegun series is only ~1e-7 accurate, which would cap the
  achievable KKT residual), derivatives closed-form:
  `erf'(u) = 2/√π·exp(-u²)`, `erf''(u) = -2u·erf'(u)`, evaluated as
  `-2·(u·erf'(u))` so the Hessian arms stay finite at magnitudes where
  `-2u` would overflow while `erf'` has already underflowed. Reachable
  only through the in-memory path, since no `.nl` opcode maps to it.
  `HessianProgram`'s `program_supports_op` allowlist does not include it,
  as it does not include the other transcendentals.

New Rust API alongside the bindings: `NlProblem::from_expressions` /
`NlProblemParts`, `NlTnlp::hessian_vector_product` /
`hessian_vector_products`, and `nl_reader::render_expression`.

### Changed — `TapeOp` / `UnaryOp` gained an `Erf` variant (#469)

Source-breaking for any crate outside this workspace that matches on
either enum exhaustively; nothing in-repo is affected, and no existing
API changed behavior. Called out under its own heading because a reader
scanning for breakage should not have to find it inside an `Added` entry.

### Fixed — expression-DAG walks that were exponential in sharing depth (#469)

`collect_vars` and `collect_funcall_ids` re-entered a shared `Expr::Cse`
body once per reference rather than once per body, making them Θ(2^depth)
on a subexpression-sharing DAG. Both now memoize on `Arc` pointer
identity, which cannot change the answer (each collects into a set). This
was reachable before — `get_variables_linearity` calls `collect_vars`, and
presolve calls that on every solve — but `build_nl_problem` is a new door
that makes such a DAG trivially constructible from a frontend. Measured
on a balanced share-DAG: depth 26 took 3.0 s through
`get_variables_linearity` and now completes at depth 30 instantly.

### Fixed — a deeply nested expression no longer kills the interpreter (#472)

Every consumer of an `Expr` recurses once per level of nesting — the tape
builder, the problem assembler, freeing one, and the `.nl` parser that
produces one — so a deep enough model overran the thread's stack. The
result was a SIGSEGV rather than an exception: no traceback, nothing to
catch, and an interactive session lost with it. Both doors reached it, and
one of them predates the `NlExpr` surface entirely: `read_nl` on a
generated `.nl` file whose objective is a long `o0` chain died at ~4 000
levels, inside the parser, before any tape existed.

- **The walks now run on a worker thread with a 64 MB stack**, so what is
  survivable no longer depends on the calling thread — 8 MB on a
  macOS/Linux main thread, 1 MB on Windows, less on a `threading.Thread`.
  That covers the tape build, the assembly, the parse, and the teardown of
  a problem that holds deep trees.
- **One depth limit, enforced at both doors.** `NlExpr.max_depth` is now
  10 000 and `read_nl` / `parse_nl_text` enforce the same number on what
  they parsed — a model that arrives already built cannot be capped as it
  is constructed. Past it, a `ValueError` naming the flat alternative
  (`NlExpr.sum`, or `.nl`'s `o54`). Previously `NlExpr` refused depth
  1 001 while the parser accepted 3 000 and crashed on 4 000; the limit is
  now a property of the machinery rather than of the door, and it sits
  ~3x inside what the worker stack holds even in a debug build.
- Wide models are unaffected either way: an n-ary sum is one level however
  many terms it has.

**Building an expression is no longer quadratic.** Every operator used to
deep-copy both operands, so accumulating in a Python loop copied the whole
chain on every iteration: 5 000 terms took over five seconds, 40 000 over
forty. Operands are now *referenced* rather than copied, making an
operator O(1) whatever it is applied to — that same 5 000-term loop is
about three milliseconds.

- Reusing a Python name now reuses the subexpression rather than
  duplicating it. `t = x[0] * x[1]` used ten times is one shared body on
  the tape — evaluated once per sweep, with its adjoint summing the ten
  contributions, which is the same value and the same derivatives off a
  tape a tenth the size. Expressions that are only tractable *because* of
  the sharing now work at all: `for _ in range(40): e = e * e` describes
  `x ** 2**40` in 40 nodes and builds, tapes, and differentiates in under
  a millisecond, where copying would have needed a trillion nodes.
- What reaches the solver is unchanged for anything not genuinely shared.
  References that exist only to hand an operand to an operator are inlined
  when the model is assembled, so `from_expressions` sees the same plain
  tree it always did — which is what keeps each term of a sum its own
  tape, and each tape its own small set of Hessian colors.

New Rust API: `NlTnlp::problem_mut`, which is how the Python binding takes
the expression trees out of a problem to tear them down on a stack sized
for them.

### Added — pyomo-pounce: `wrt=` block selection on both accessors (covariance roadmap item 3, #262)

- `covariance(m, wrt=...)` and `information(m, wrt=...)` reduce onto
  any block of the solve's variables off the held factor, post-solve:
  a Var, an indexed slice, a `(Var, iterable)` pair, data objects, or
  a mixed list. The declared fitted block is the default, so the
  no-wrt behavior is untouched (existing suites pass unchanged). Each
  call re-reduces onto its own argument: one solve, many blocks, each
  getting that block's marginal.
- Sigma estimation divides by the FIT's degrees of freedom, a
  property of the solve, not the block, so sub-block marginals agree
  exactly with the corresponding entries of the default answer.
- A rank-deficient block (more coordinates than the fit's degrees of
  freedom: the prediction-band case, e.g. `wrt=m.r` giving
  `sigma^2 X(X'X)^-1 X'`) returns the homoscedastic Lagrangian
  marginal with membership handling bypassed; `information()` refuses
  it toward `covariance()`. Gated by the count plus a rank test on the
  block (a within-count but linearly dependent block, e.g. a
  duplicated design point, takes the same routes with its own
  message), not by LAPACK, which does not reliably raise on
  structurally singular blocks; a dedicated `_SingularBlock` exception
  stands as the last resort. The rank test runs on the diagonally
  scaled block, so the verdict tracks collinearity and not the unit
  spread between coordinates: a covariance block carries the square
  of that spread, and numpy's default tolerance is relative to the
  largest singular value, so two well-determined parameters far apart
  in magnitude would otherwise be refused for their units alone.
- `information()` routes: a manifold-parameterizing block (size equal
  to the degrees of freedom) gets the exact tangent construction; a
  sub-block of the fitted set gets its marginal by Schur complement
  of the exact tangent R over the fitted block, never inverting a
  covariance, so a pinned member costs no digits; any other block
  reduces off the held factor with the item-1 corrections, benign for
  free coordinates (no barrier term in the slice).
- Strongly active variables OUTSIDE the block come back on the result
  as `.conditioned_on` (both accessors): the matrix is conditional on
  those bounds, not marginal over them. Identification is item 1's
  reduced-level rule applied per candidate as a singleton block (one
  backsolve gives `(K^-1)_ii`, effective curvature
  `|1/(K^-1)_ii - Sigma|`, the shipped ratio edges call it):
  scale-invariant, same theory as the block members, after a cheap
  `Sigma > sqrt(mu)` prefilter only near-bound variables pay. Computed
  lazily on first access, so calls that never read it cost nothing
  extra. Diagnostics stay "fitted parameter"-worded on the default
  path and speak block-relative only under an explicit wrt=.

### Changed — pyomo-pounce: `information()` rank-gates the free block before the conditional-information solve

- Computing `S`, the information a bound-pinned parameter carries
  conditional on the rest of the pinned set, takes a Schur complement
  through the free block. That step previously relied on
  `np.linalg.solve` raising on a singular free block, which is
  BLAS- and machine-dependent: the same refusal test flipped
  pass/fail across adjacent CI runs on the same numpy with no numeric
  change, because whether the pivot lands on exact zero varies by
  build. The free block is now rank-tested first (diagonally scaled,
  as the `wrt=` gates are), so the verdict is deterministic; the
  exception clause stays as the last resort.
- This applies on the DEFAULT path, not only under `wrt=`: a
  numerically dependent free block now raises where it could
  previously return a large-but-meaningless `S`. Well-conditioned and
  merely ill-scaled blocks are unaffected — that is what the diagonal
  scaling buys.

### Fixed — `sparse=True` no longer materializes a dense Jacobian/Hessian to detect the pattern (#464)

- Sparsity detection ran *before* the `if sparse:` branch and evaluated
  the dense `(m, n)` Jacobian and `(n, n)` Lagrangian Hessian at each
  probe point, so building a sparse problem was `O(n²)` in memory no
  matter what `sparse` was set to — and `n_probes` defaults to 3 under
  `sparse=True`, so the sparse path allocated *more* dense matrices than
  the dense one. A collocation system at `n = 18144` was killed at ten
  minutes and 5.5 GB; the pattern for the size actually wanted
  (`n = 91584`) would have been a 67 TB matrix.
- Detection now sweeps a *block* of rows (VJPs) or columns (JVPs/HVPs)
  at a time under a fixed byte budget, reducing each block to index
  pairs before allocating the next. The AD pass count is unchanged —
  `jacfwd`/`jacrev` are themselves vmapped JVPs/VJPs over the identity
  basis — but peak memory drops from `O(n²)` to `O(block·n + nnz)`. The
  detected pattern is bit-identical to the dense probe's, including
  row-major ordering. On the banded family above: `n = 18144` now builds
  in 3.3 s at flat RSS, and every size from 2268 up holds the same peak.
- **New: `jac_pattern=(rows, cols)` / `hess_pattern=(rows, cols)`** on
  `from_jax`, `JaxProblem`, `from_torch`, and `TorchProblem`. Supplying
  either skips detection for that matrix entirely — no probe
  evaluations, and no probabilistic-detection risk. For a
  full-discretization method the structure is known in closed form
  before any numbers exist, so rediscovering it by probing inverts the
  method's main structural advantage. `n = 91584` with supplied patterns
  builds in 0.9 s and 0.21 GB. Either may be given alone; the other is
  still probed. The pattern must be a superset of the true structure —
  extra entries only report zeros, but a missing entry is silently
  wrong, and nothing evaluates the model to check. Upper-triangle
  entries in `hess_pattern` are folded onto their mirror, since `H` is
  symmetric.
- The CPR column coloring (`_color_columns`), the other half of the
  build cost, is now vectorized over CSR/CSC gathers instead of
  per-column Python dict/set loops: 52 s → 2.5 s on a banded `n = 9072`
  pattern, with entry-for-entry identical output (tested against the
  previous implementation as an oracle).

### Added — pyomo-pounce: `information()`, the un-inverted sibling of `covariance()` (covariance roadmap item 2, #262)

- The reduced Hessian over the declared fitted block from the same
  single solve, natural units, no `sigma^2`: for a homoscedastic
  Lagrangian fit, `covariance()` equals `2*sigma^2*inv(information())`
  on the free block (tested). Same `hessian=` selector as
  `covariance()`.
- The Lagrangian form is built by TANGENT RECOVERY, not by inverting
  the covariance back or subtracting the barrier off the factor: the
  K-inverse columns' x-blocks are `T*M`, so `T = Zx*inv(M)` exactly
  and `R = T'HT` with the exact Lagrangian Hessian. The barrier
  weight cancels multiplicatively; measured on a pinned model with
  `Sigma/q ~ 3e10`, the route is exact to machine precision where the
  subtraction route loses ten digits. A new session primitive,
  `Solver.hessian_vec(v)` (user-space, natural units), supplies the
  products.
- Dispositions per item 1's table, with the sign of one flipped by
  design: a strongly active parameter returns `S`, the reduction onto
  the pinned set, NOT a zero row (zero information is the opposite of
  what a pinned parameter carries), conditional on the rest of the
  pinned set, zero cross blocks. Binding rows project the free block
  on both sides, the pseudo-inverse of the projected covariance
  (tested against `pinv`). The Gauss-Newton product is formed over
  ALL fitted parameters and sliced last, so the pinned rows exist to
  build `S` from. An indefinite Lagrangian block returns as computed
  with a warning naming Gauss-Newton; the detector is unit-pinned
  since a genuine minimum is PSD by necessity.
- `covariance()`'s binding-row conditional-information scalar now
  comes from the same tangent construction (lazily, only solves with
  a binding row pay the Hessian products): accurate to ~1e-6 where
  the factor subtraction lost ten digits, the residue being the
  binding row's own finite slack-barrier weight in the recovery.
  Bound and equality activity stays machine-exact; the residue is
  specific to binding inequality rows, which couple through the
  slack block.
- Membership, row handling, and their warnings are shared with
  `covariance()` (`_classify_fitted_block`), so the two accessors
  cannot drift.
- Factor indexing follows the full-x vs var-x discipline of gh #450
  throughout: fitted and residual rows route through `primal_row`,
  and the tangent recovery slices the factor's var-x block and
  scatters back to full-x for `hessian_vec`. Regression: one inert
  fixed variable ahead of the fitted block changes nothing, both
  forms.

### Added — POUNCE runs in the browser (WebAssembly)

- The default build has no C or Fortran dependency, so the whole solver —
  `.nl` reader, AD tape, sparse LDL^T, interior-point algorithm — compiles
  to `wasm32-wasip1` and runs client-side. New crate `pounce-wasm`
  (`publish = false`) exposes a four-function C ABI (allocate, load, solve,
  free) that takes `.nl` bytes and `ipopt.opt`-format options and returns
  JSON: a problem summary from `pounce_load`, a solution from
  `pounce_solve`. Both catch panics, so a malformed model returns
  `{"error": …}` instead of trapping the instance.
- A demo page ships with it (`crates/pounce-wasm/web`, built by
  `crates/pounce-wasm/build.sh` or `make wasm`): drop a `.nl` file — with
  its `.col` / `.row` name files if you have them — to see the model's
  size, degrees of freedom, equality/inequality split, nonlinear fraction,
  and Jacobian/Hessian sparsity, then solve it with the iteration table
  streaming into the page. It is static files plus a ~60-line WASI shim; no
  `wasm-bindgen`, no npm, no server. Nothing is uploaded.
- A second page, `/demo/python/`, runs **Pyomo** in the browser: Pyodide
  supplies CPython, `micropip` installs Pyomo's pure-Python wheel, you write
  an ordinary model, and `pounce_browser.solve(m)` writes the `.nl`, solves
  it with the same wasm module, and loads the `.sol` back so `x.value` and
  `model.dual[c]` read as after any local solve. Variables and rows are
  matched by Pyomo's own writer ordering, pinned by
  `crates/pounce-wasm/tests/pyomo_roundtrip.py` — a model with a closed-form
  optimum and multipliers, run in CI with Node standing in for the browser.
  It is Pyomo's modelling layer rather than POUNCE's Python API: the model
  crosses as a file, so there are no Python callbacks mid-solve.
- The Python page's script box is an editor rather than a bare textarea:
  syntax highlighting, line numbers, Tab / Shift-Tab block indent, and
  indentation carried across Enter (one level deeper after a colon). No
  editor library — a highlighted `<pre>` under a transparent `<textarea>`,
  so the caret, selection, and undo stay the browser's, and the page keeps
  working offline where a CDN-loaded editor would not.
- The demos are published with the docs: `docs.yml` builds the module and
  stages both pages into the Pages site
  ([`/demo/`](https://jkitchin.github.io/pounce/demo/) and
  [`/demo/python/`](https://jkitchin.github.io/pounce/demo/python/)) on
  every deployment from `main`. Neither needs Pages configuration —
  single-threaded, so no
  `SharedArrayBuffer` and no COOP/COEP headers (which Pages cannot set),
  and all URLs are relative so the `/pounce/` base path just works.
- Numerics are unchanged. Over all 37 `.nl` fixtures in
  `crates/pounce-cli/tests/fixtures`, wasm and native agree on exit status
  and iteration count on every one, and the objective is bit-identical on
  34 of the 36 that return one; the two exceptions differ by one ulp, at
  objectives of 1e-10 and 1e-11. Documented in `docs/src/wasm.md`; CI builds the module and solves
  a model under Node's WASI on every PR.
- Downloads off a finished solve: an AMPL `.sol` (byte-identical to the one
  `pounce model.nl` writes, reduced-cost suffixes included, so AMPL and
  Pyomo read a browser solve back unchanged), a CSV of every variable and
  constraint with names, bounds, and multipliers, and the solver log. Both
  solution files are formatted in wasm from the full solution rather than
  from the on-screen tables, which truncate at 2,000 rows.
- The AMPL `.sol` writer moved from `pounce-cli` to `pounce-nl`
  (`sol_writer`, re-exported under its old name) so every frontend emits the
  same file — same dual sign convention, same suffix headers. `NlTnlp` now
  keeps the converged multipliers (`final_lambda`, `final_bound_multipliers`)
  alongside `final_x`, so writing that file needs no second derivation of
  the duals.
- Dropping a file resets the page: it discards the worker and starts a fresh
  wasm instance, so no parsed model, solver state, or grown heap survives
  into the next model.
- Payloads returned from wasm are length-prefixed rather than
  NUL-terminated. The reader no longer scans for a terminator, so a bad
  pointer or length is reported as itself instead of surfacing later as an
  unrelated JSON parse error.
- `pounce-nl` now builds for targets with no dynamic loader: `libloading`
  became a `cfg(any(unix, windows))` dependency and `ExternalLibrary::load`
  reports that AMPL imported functions are unavailable there rather than
  failing to compile. No change on unix/windows.
### Fixed — debugger: convex backend answered `print kkt` with a self-referential hint (#462)

- On the convex / conic IPM, `print kkt` replied *"no KKT factorization yet —
  stop at `after_search_dir`"* — and repeated it verbatim once you were
  stopped at `after_search_dir`. The advice could never pay off: that backend
  has no augmented system to report inertia for at any checkpoint, so the
  suggestion sent a user (or an agent driving `--debug-json`) round a loop
  with no exit. Diagnostics only; no numerical result was affected.
- The backend-conditional commands now reject on a **capability** basis
  before any timing check, matching the documented contract and their
  siblings (`print rank`, `diagnose`, `resolve`): `print kkt`, `print
  residuals`, `print active` / `inactive`, `viz kkt`, and `viz L` answer
  *"only available for the NLP solver (not the convex/conic solver)"*.
  `print active` previously reported "no bounded variables or inequality
  slacks" there — a silent no-op the docs explicitly rule out. The NLP
  filter-IPM is unchanged, including the genuine "not factored yet" hint at
  a pre-factorization checkpoint.
- The `hello` handshake is now answered for the backend that is running, so
  a JSON client feature-detecting off `capabilities` is told the truth:
  `kkt_inspect`, `diagnose`, `mutate_mu`, `resolve`, `sweep`, `load`, and
  `structural_diagnose` are `false` on the convex path, `viz` drops to
  `["block","delta"]`, and `blocks` lists that solver's own iterate blocks
  (`x`/`s`/`y`/`z`, plus `tau`/`kappa` on HSDE) instead of the NLP names.

### Fixed — Linux wheels shipped a CLI that could not start on most clusters (#452)

- The published Linux wheels were tagged `manylinux2014` (glibc 2.17) but
  bundled a `pounce` CLI requiring **glibc 2.39**. On anything older the
  binary refused to exec:

      pounce/bin/pounce: /lib/x86_64-linux-gnu/libc.so.6:
        version `GLIBC_2.39' not found

  `import pounce` was unaffected, but everything driving the CLI —
  the `pounce` console script and all of pyomo-pounce, which shells out
  to it — failed on Debian 12, Ubuntu 22.04, RHEL/Rocky/Alma 8 and 9, and
  most HPC images. `pip install` reported success; the failure came at
  exec time.
- Cause: `release-pounce.yml` built the CLI on the runner host (Ubuntu
  24.04, glibc 2.39) and only then handed the tree to maturin, which built
  the extension module inside the manylinux container and labelled the
  wheel accordingly. auditwheel never inspects `pounce/bin/pounce` — to the
  wheel format it is opaque data, not a linked object — so the mismatch
  shipped silently. Exactly two symbols set the floor, `pidfd_spawnp` and
  `pidfd_getpid`, from Rust std's process-spawn path; everything else the
  binary needed topped out at glibc 2.34.
- Fix: build the CLI **inside** the manylinux container, via maturin's
  `before-script-linux`, so the artifact and the compatibility promise come
  from the same place. No source change — POUNCE is pure Rust, so the
  container only has to supply a linker. The resulting floor is glibc 2.16,
  and wall-clock on the large_scale `sparseqp` / `optcontrol` / `rosenbrock`
  problems is unchanged (same allocator, same codegen).
- Guard: `scripts/check-cli-portability.sh` asserts the bundled binary's
  highest referenced glibc symbol version stays within the wheel's
  manylinux floor, and CI now installs the built wheel in a glibc-2.17
  container and solves with it. Neither check existed before, and the
  pre-existing wheel smoke test could not have caught this: it ran on the
  same host that built the binary, the one platform where the floor is
  invisible.
### Added — container images (#449)

- Two Dockerfiles under `docker/`, both producing the same three
  interfaces (the `pounce` CLI, `import pounce`, and Pyomo's
  `SolverFactory('pounce')`) so a job script written against one runs
  unchanged on the other. `Dockerfile.release` pip-installs the
  published wheels for a pinned `X.Y.Z` and builds in seconds;
  `Dockerfile` compiles the working tree you hand it, for testing an
  unreleased commit. `make docker` / `make docker-release` build them
  locally with the version and commit filled in.
- Published to `ghcr.io/jkitchin/pounce` by
  `.github/workflows/release-docker.yml`: `:X.Y.Z`, `:X.Y` and
  `:latest` from the release image on a `v*` tag, `:edge` and
  `:sha-<short>` from the source image on every push to `main`. Both
  are multi-arch (linux/amd64 + linux/arm64, each built natively).
  Primarily aimed at clusters, where Apptainer/Singularity can pull
  them directly — see the new [Docker & Containers](docs/src/docker.md)
  page.
- Each image runs a smoke test as its final build step (a CLI solve,
  `import pounce`, and a Pyomo plugin lookup), so a build that succeeds
  is an image that works.
- Neither image enables the `ma57` feature: CoinHSL is
  license-restricted and not redistributable. The pure-Rust FERAL
  backend is the default and needs nothing external.
- Both images sit on Debian trixie rather than bookworm, which the
  release image has no choice about *yet*: the wheels published through
  0.9.0 are tagged `manylinux2014` but bundle a CLI needing glibc 2.39,
  so `pounce --version` fails outright on bookworm. The build-time smoke
  test is what surfaced that, and it is fixed in #452 / #456 — but this
  image installs from PyPI, so the pin stays until a `python-v*` tag
  rebuilds the wheels, after which it can drop to bookworm or lower.
- `crates/pounce-cli/build.rs` now honors a `POUNCE_BUILD_GIT`
  override for the SHA it stamps into `pounce --about`. The source
  image keeps `.git` out of its build context (~90M the compile never
  reads), which previously left every such image reporting `unknown`
  with no way to tell which commit it contained.

### Fixed — pyomo-pounce: a fixed variable shifted every sensitivity result

- A variable whose bounds are equal is removed from the solve
  (`fixed_variable_treatment = make_parameter`), so the KKT factor's
  `x` block is shorter than the user's variable list and every later
  column moves. The `.col`-order rows the sensitivity session hands
  around are user-space, and were used to index the factor directly:
  one fixed variable anywhere made every later variable read its
  NEIGHBOUR's row. `gradient()` returned a plausible wrong number with
  no warning (`d(y)/dp` where `d(x)/dp` was asked for); `estimate()`
  failed on a shape mismatch; `covariance()` raised a bogus
  "structurally unidentifiable". Models with no fixed variable — every
  model in the test suite — were never affected, since the two spaces
  coincide exactly then.
- `Solver.primal_rows(x_indices)`: the factor rows of user-space
  variable indices, `None` for a removed fixed variable. The `x`
  counterpart of the existing `multiplier_rows`, which had always done
  this translation for the `y_c` block. Every index the session takes
  into the factor now routes through it, and asking for a removed
  variable's sensitivity is an explicit refusal rather than a
  neighbouring column.
- Regression tests add one inert fixed variable to an existing model
  and require every answer to be unchanged, plus a finite-difference
  check so they pin the right value and not merely a consistent one.

### Performance — pyomo-pounce: `initialize()` walked the same incidence three times per call (#444)

- One `initialize()` call built whole-model incidence three times and
  re-derived each constraint's incident variables six times over: the
  plan pass, a re-plan inside `block_initialize(repair="auto")` that
  was structurally a no-op (every candidate already fixed) yet paid a
  full graph construction and denominator sweep, and the analyze
  pass. On a 25,276-constraint collocation model that was 345.7 s of
  almost purely symbolic work (fill and projection off).
- The call now performs one structural incidence walk
  (`structural_incidence`, fixed variables included) and each pass
  filters it to the currently-unfixed variables through
  `IncidenceGraphInterface.subgraph`, which copies the stored graph
  without re-inspecting constraints. The plan filters pre-fixing and
  the analyze pass post-fixing. The shared view is structural where a
  fresh build substitutes fixed values: the edge sets can differ when
  that substitution cancels a term — a fixed zero is the obvious way
  but not the only one, since values can cancel across terms with no
  fixed variable being zero (`a*x - b*x` with `a` and `b` fixed
  equal). Every row adjacent to a fixed variable is therefore
  re-derived and compared, and a genuine cancellation falls back to a
  fresh build for that pass; the variable order within rows may differ on
  models where fixed values change the linear/nonlinear split; order
  feeds tie-breaks, so equally-valid diagnostics (which variable is
  reported loose) may resolve differently between the two views.
  `initialize` passes `repair="off"` downstream since its own plan
  already ran. On pyomo older than 6.7.1 (no
  `IncidenceGraphInterface.subgraph`) every pass falls back to the
  fresh build: old speed, same behavior.
  `block_repair_plan`, `block_analyze`, and `block_initialize` accept
  the shared graph as an optional `igraph=` argument and behave as
  before when it is omitted.
- Regressions: a fresh-versus-shared analysis identity test (same
  plan, same square decomposition, same block order), and a counter
  test pinning exactly one model-walking incidence construction per
  `initialize()` call so the redundancy cannot quietly return
  (`subgraph` re-enters `__init__` with a prebuilt graph tuple and is
  not counted).

### Changed — pyomo-pounce: `covariance()` membership from the solve's barrier geometry (#362, covariance roadmap item 1)

- Bound and constraint activity on the fitted parameters now classifies
  through the item-0 rule applied at the reduced fitted block, where a
  fitted parameter's curvature actually lives (its raw Lagrangian diagonal
  is zero in the residual-variable idiom, so the per-coordinate report
  alone cannot decide membership there). The slack-only test
  (`tol = 1e-6 * (1 + |x|)`) and its moved-bounds re-injection are gone.
- Dispositions per the item-1 table: strongly active projects (zero
  variance, conditional, warned); weakly active is KEPT at its full
  finite variance, where the slack test deleted it and the raw factor
  would have halved it; ambiguous and unidentified are kept and warned.
- The value correction subtracts the fitted rows' retained barrier
  diagonal (`var_sigma`, new in the activity report along with
  `row_sigma`) from the factor's reduced Hessian: a weakly active kept
  parameter reports its true curvature `q`, not the factor's `2q`, and
  inactive rows shed an O(μ) drift. On kept rows Σ is at most the
  curvature's own size, so nothing cancels; pinned coordinates and
  binding directions never enter (excluded by the free restriction and
  annihilated by the projection, which also annihilates the binding
  rows' huge barrier weight exactly).
- A strongly active inequality row whose normal touches the fitted
  block pins a combination, not a coordinate: both accessor paths
  (Lagrangian and Gauss-Newton) project on the null space of the
  binding normals, the matrix goes singular by one per binding row,
  and the warning names the constraint, the pinned combination, and
  its conditional information. A bound moved onto a row
  (jkitchin/pounce#357) is the single-coordinate case and returns
  exactly the variable-bound disposition, so the two spellings of one
  limit agree in the matrices (#362), not only in the classification.
- The restricted normal is honest only when the row's support outside
  the fitted block is pinned (declared-parameter pin columns do not
  count as outside: they cannot move). A binding row reaching the
  fitted parameters through FREE eliminated variables pins a
  direction the restricted normal cannot represent (`a + r_1 <= cap`
  with `r_1 = y_1 - a - b x_1` pins a `b`-direction while the
  restricted normal reads `e_a`), and the reduced-level ratio is
  equally blind, so such a row takes item 0's raw classification, is
  kept unprojected, and warns explicitly. The general treatment is
  the row's reduced normal through the elimination, item 2 territory.
- The report's `var_sigma`/`row_sigma` and `row_normal` follow the
  documented natural-units contract like every other sensitivity
  output: classification runs on the solver's scaled quantities (the
  ratio is scale-invariant), the exported values are unscaled at the
  boundary. Tested by pinning the weak scalar row's `Σ = 1/c²` under
  both scaling modes and `row_normal` returning the user's own
  coefficient.
- Declaration-triggered solves set `bound_relax_factor=0`: the
  classifier reads slacks as distances to the user's own bounds.
- `Solver.row_normal(j)`: a constraint row's gradient at the converged
  iterate in user variable order, serving the projection direction.
- Tests pin the analytics: the weakly active kept variance equals the
  unconstrained `σ²(XᵀX)⁻¹` entry; a binding `a + b <= cap` matches
  restricted least squares with corr −1 and a rank drop; bound and row
  spellings agree to 1e-6; an inactive far bound changes nothing to
  1e-7; Gauss-Newton matches on the linear model, projection included.
- `_classify_ratio` mirrors the Rust rule in Python, so a drift test
  checks the two against each other on real solves: every classified
  entry of a `classify_activity()` report must re-derive its own
  status from the reported `(ratio, mu)` through the Python rule.

### Fixed — feasible convex QPs reported locally infeasible on the NLP path

- With `solver_selection=nlp`, POUNCE reported `Converged to a point of
  local infeasibility` on 15 Maros-Mészáros convex QPs that are feasible
  and that POUNCE's own `qp-ipm` path solves to optimality (#446). A
  confident wrong answer, not a failure: Pyomo maps
  `Infeasible_Problem_Detected` into the infeasible family, so a caller
  reads it as a modelling error. QSCSD1 is the clearest case — the verdict
  was rendered at a converged KKT point whose constraint violation was
  `9.2e-15` and dual infeasibility `2.8e-14`, with an objective agreeing
  with the `qp-ipm` optimum to six figures.
- One cause for all 15, in the scale-relative feasibility measure added by
  #385 and extended to equality rows by #390. It divides a row's violation
  by that row's *declared* magnitude — the pre-fold RHS, or the declared
  bounds — and abstained only when that magnitude was **exactly** zero.
  Every one of the 15 carries rows whose declared magnitude is `1e-17` to
  `1e-16`: rounding residue from the netlib→`.nl` conversion, `2^-53`
  written where the model says `0`. The row's residual cannot be driven
  below its own floating-point noise floor either, so the ratio was noise
  over noise. QSCSD1 read 81× violated, which vetoed its success
  certificate at iteration 15, kept the solver grinding to iteration 77
  past a solution it already had, and then armed the rapid-infeasibility
  pre-filter that issued the verdict.
- "Zero" is now read numerically. A row abstains once its declared
  magnitude sinks under its own noise floor, `κ · eps · max_j |∂g_i/∂x_j| ·
  ‖x‖_∞` — the finest residual the solver could drive that row to, which is
  what makes a declared magnitude below it residue rather than data. The
  comparison keeps the scale invariance that is the whole point of the
  measure: under a row scaling both the floor and the magnitude carry
  `dc_i`, so the verdict is the same however the row is written. An
  absolute cutoff on the magnitude would have thrown that away.
- The floor models **how finely the solver can place `x`**, not how
  accurately a row evaluates. A Newton step comes from a linear solve with
  norm-wise backward error, so every component of `x` is positioned to
  about `eps · ‖x‖_∞`: a variable at `1e-8` inside a vector of norm `2.7`
  is still only resolved to `~6e-16`. The dependence on the *global*
  `‖x‖_∞` is therefore deliberate — `x` is one vector solved for jointly.
  The seemingly sharper per-row alternative, the exact term sum
  `Σ_j |a_ij x_j|`, was implemented and measured: it models the row's
  evaluation error, which is not what limits the residual, and it regressed
  QETAMACR, QSCORPIO and QPILOTNO. QSCORPIO's row 93 parks its variables
  near a zero bound at `~1e-8`, so the term sum drops its floor to `8.5e-22`
  and `−5.6e-17` of rounding residue reads as real data again, at an iterate
  resolved only to `~6e-16`. A unit test pins the `‖x‖_∞` dependence.
- A row whose Jacobian is empty abstains outright. Every variable it
  mentions has been fixed and substituted out, leaving a constant `0 = b`
  that no iterate can move — a statement about the model, which presolve
  certifies up front, rather than a residual to judge an iterate by.
  QPILOTNO's row 150 reduces to `0 = −2.22e-16` this way and pinned the
  relative measure at 100% for all 375 iterations.
- Both blocks are fixed, not just the equality one: QPILOTNO also carries
  43 inequality bounds at `1e-17`–`1e-15`, and one of them — a row sitting
  at exactly `d(x) = 0` against a declared bound of `1.1e-16` — drove the
  verdict from the inequality side.
- Measured on the `qp_convex` head-to-head (`make benchmark-qp-convex`),
  the NLP arm goes from **113/138 solved in 525.7 s to 137/138 in 504.4 s**
  — more problems solved, in less total time. All 15 wrong
  infeasibility verdicts are gone, and so are nine further failures with
  the same origin — five `Search_Direction_Becomes_Too_Small` and four
  `Solver_Error`, all cases of the veto pushing the solver past a solution
  it had already reached until the step computation broke down. No problem
  regressed, and the iteration counts of the newly-solved problems mostly
  fall (QSCSD1 77→15, QSCSD8 121→20). The one remaining failure, BOYD2's
  300 s CPU-limit timeout, is pre-existing and unrelated.
- Masked in default use, because `solver_selection=auto` routes convex QPs
  to `qp-ipm`. Anyone driving POUNCE as a general NLP solver on a convex QP
  hit it.

### Fixed — restoration had no verdict when its sub-problem converged

- `IpRestoConvCheck::CheckConvergence` is two layers, and pounce ported
  only the first. Layer 1 — the κ_resto reduction guard, the
  square-problem fast path, and the outer-filter acceptance test —
  answers *can the trial point leave restoration?*. Layer 2
  (`IpRestoConvCheck.cpp:200-240`) answers *has the restoration
  sub-problem itself converged?*, and it is the only thing that bounds a
  restoration which can never satisfy layer 1 (#438).
- Without it, a sub-problem sitting at its own KKT point — the moment it
  has provably done everything it can — was indistinguishable from one
  still making progress. When the κ_resto target is out of reach (the
  outer enters restoration at a nearly-feasible point, restoration moves
  the iterate away, and the reduction test then fails at every subsequent
  iteration), the only remaining exits were the `maximum_iters` /
  `max_resto_iter` caps: the wrong status, reported after orders of
  magnitude more work than the diagnosis needed.
- Layer 2's four-way verdict is now rendered inside the restoration
  sub-solve: tighten the sub-problem tolerance once and continue,
  converged-to-acceptable on a feasible square problem,
  converged-to-a-feasible-point, or `LOCALLY_INFEASIBLE`. The tightening
  arm keeps upstream's `tol > 1e-1 · orig_tol` guard, without which it is
  an infinite loop.
- Two deliberate deviations from upstream, both scoped so that #438
  changes only the arm that had no verdict at all: the
  converged-to-a-feasible-point arm hands the recovered point back to the
  outer phase instead of throwing (upstream reports a restoration
  failure), and square problems are exempt from the locally-infeasible
  arm, matching the carve-out the post-hoc detection in
  `resto_inner_solver` already made for them.
- Set `POUNCE_DBG_RESTO_LAYER2=1` with `RUST_LOG=pounce::restoration=debug`
  to trace the verdict.

### Fixed — layer 2 tightened at points that were already feasible

- The layer-2 port above regressed four Vanderbei problems from `Optimal`:
  `dallasm` and `dallasl` to `Error_In_Step_Computation`, `eigmaxa` and
  `eigmina` to `Restoration_Failed`.
- One cause for all four. Upstream states the tightening arm's premise in
  its own comment — it tightens "in case the problem is only very
  slightly infeasible" — and the arm exists to spend the tolerance budget
  chasing a residual constraint violation down to zero. It was firing on
  points that had no violation left to chase: `eigmaxa`/`eigmina` reach
  `inf_pr = 7.5e-15` against an outer `tol` of `1e-8`, `dallasm` reaches
  `1.5e-10`. Tightening there drives the restoration sub-solve *past* the
  point the outer asked for, into a tiny-step or step-computation failure
  where handing the point straight back solves the problem.
- The arm now checks its own premise (`orig_trial_inf_pr > orig_tol`). A
  point at or under the original NLP's `tol` falls through to the
  converged-to-a-feasible-point arm, which is the verdict that describes
  it. Restorations that are genuinely slightly infeasible — #438's own
  case, `qcqp1000-1nc`, sits at ~5e-3, five orders above the gate — are
  untouched.
- Upstream never has to make this distinction, because it cannot reach
  layer 2 at a feasible point: `IpBacktrackingLineSearch.cpp:578` refuses
  to enter restoration once the violation is under `1e-2 · tol`, and
  layer 1's reduction target is floored at `min(tol, constr_viol_tol)`
  (`IpRestoConvCheck.cpp:162`) so a feasible trial is released before
  layer 2 is consulted. Pounce has neither guard, so it arrives by both
  routes and the premise has to be tested rather than assumed.
- Adding upstream's layer-1 floor was tried and rejected: it is faithful,
  but it releases the sub-solve earlier at points far from the
  sub-problem's own KKT point (`dallasl` exits at `inner_kkt_err = 4.5e-1`
  where it previously ran one iteration further to `4.2e-9`) and regresses
  `dallasl` on its own. Left unfloored deliberately, with a note at the
  call site; it is a separate question from #438.
- Verified against a same-machine `8c81cf4a` baseline: Vanderbei returns
  to 697 optimal with identical statuses on all 733 problems and no
  objective changes.

### Fixed — port gap: no `ACCEPTABLE_POINT_REACHED` at the restoration doorway

- **Upstream refuses to enter restoration from an acceptable point; pounce did
  not.** `IpBacktrackingLineSearch.cpp:557-570`, in the `if (!accept)` arm that
  hands the iterate to restoration:

  ```cpp
  if( CurrentIsAcceptable() )
  {
     THROW_EXCEPTION(ACCEPTABLE_POINT_REACHED,
                     "Restoration phase called at acceptable point.");
  }
  ```

  Restoration reduces the constraint violation, so from a point that already
  passes the acceptable-level tolerances it has nothing to reduce and can only
  put a reportable solution at risk. That check had no counterpart in
  `invoke_restoration`, so pounce entered anyway.
- **What the gap cost.** On Maros-Mészáros/mittelmann `qcqp1000-1nc` (n=1000)
  the line search fails at iteration 187 holding the published optimum —
  `-2.6628866e+07`, matching ipopt-ma57 to nine significant figures — with an
  overall NLP error of `6.0e-8`, two orders inside `acceptable_tol`.
  Restoration walked that iterate to `theta 5e-3` and ground out **2780 further
  iterations** without recovering, so a solved problem reported
  `Maximum_CpuTime_Exceeded` at the 300 s cap. With the check restored it stops
  at iteration 187 and reports the optimum: **300 s → 22 s**, relative
  objective difference against ipopt-ma57 `4.2e-12`.
- **The predicate is upstream's, unmodified.** A strict `constr_viol_tol` gate
  was tried on top: it deviates and does not discriminate — at their
  restoration entries `qcqp1000-1nc` sits at `theta = 6.0e-8`, `csfi2` at
  `1.5e-7`, `eigena2` at `2.1e-10`, all strictly feasible, one by six orders.
  Nothing observable at the doorway separates a restoration that recovers from
  one that does not, which is why upstream does not try to. A
  bounded-restoration budget was also considered and rejected: it would invent
  a mechanism upstream has no equivalent of (see #438 for what upstream uses
  instead, and what pounce is still missing).
- **Acceptability is the full triplet, never `theta` alone** — gh #274: a
  perfectly feasible point can be arbitrarily far from stationary
  (`min -exp(x) s.t. x >= 0` reaches this branch with `inf_pr = 1.7e-10` and
  `inf_du = 8.8e+47`), and `current_is_acceptable_with_state` carries
  `acceptable_dual_inf_tol` to reject it. The finiteness precondition is
  upstream's too; CUTE `himmelbj` reaches a near-feasible point where `f` is
  NaN and must stay `Invalid_Number_Detected`, which it does.
- **One documented deviation.** Upstream runs `PrepareRestoPhaseStart()` before
  the check; this sits ahead of pounce's own restoration cycle detectors, which
  upstream has no equivalent of — an acceptable point should be reported
  whatever the cycle state, and the skipped filter augmentation is immaterial
  on a path that terminates.
- **Measured** (60 s/300 s caps, ipopt-ma57 reference, same host): mittelmann
  42/47 → **43/47**; vanderbei 702/733, unchanged. Three vanderbei problems move
  `Solve_Succeeded` → `Solved_To_Acceptable_Level` with objectives unchanged to
  `9.8e-15`, `2.9e-12` and `1.1e-14`, in fewer iterations. `csfi2` thereby
  matches ipopt-ma57 exactly, which itself reports `Solved_To_Acceptable_Level`
  at the same 35 iterations; `eigena2`/`eigenb2` are honest divergences, where
  ipopt's line search never fails and so never reaches this branch at all.
  All 16 problems the restoration code names as depending on productive
  restoration (`bt8`, `odfits`, `linspanh`, `lsnnodoc`, `oet3`, `makela3`,
  `haifam`, `haldmads`, `robot`, `polak6`, `s365mod`, `sipow2m`, `pfit4`,
  `quartc`, `oet7`, `himmelbj`) are unchanged.

### Fixed — the conic QCQP path had no fallback when it returned no verified KKT point

- **A convex QCQP whose conic solve fails verification is now re-solved on the
  general NLP path instead of being reported as a failure.** `solver_selection=
  auto` routes a convex QCQP to the SOCP conic IPM, and `main.rs` returned that
  result unconditionally — the status was never inspected. The dispatcher
  already falls back to the NLP filter-IPM for *large* convex QCQPs before
  solving, on the reasoning that "a convex QCQP is still a valid NLP, so the
  fallback is sound"; this applies the same reasoning after the fact.
- On Vanderbei `airport` (n=84) the conic solver **stalls** — identically at
  `max_iter=200` and `max_iter=1000`, so a lack of progress rather than a
  budget — at 31 iterations with a complementarity of `9.55e-4`, five orders
  above tolerance, while dual infeasibility (`2.4e-10`), constraint violation
  (`1.1e-8`) and bound violation (`0`) are all converged and the objective
  agrees with ipopt-ma57 to nine significant figures. The post-solve
  verification is right to refuse to certify it, but refusing was terminal: the
  NLP path solves the same model in 15 iterations, matching ipopt exactly.
  `Restoration_Failed` (`solve_result_num` 500) → `Solve_Succeeded`.
- **Gated on `auto`, and only on `NumericalFailure`.** Under an explicit
  `solver_selection` the named engine's verdict stands — silently answering
  from a different solver would hide the stall. `PrimalInfeasible` /
  `DualInfeasible` are verdicts the conic solver *did* verify, and
  `IterationLimit` is the requested budget, which is also what `max_iter=0`
  returns (the zero-iteration contract, pounce#186).
- The decision is taken **before** the status line, the `.sol` write and the
  JSON report, so a rerouted solve emits exactly one verdict and leaves no
  stray conic result for a log scraper or the benchmark harness.
- vanderbei 701/733 → **702/733**, no other problem changed.

### Fixed — `required_infeasibility_reduction` was registered but never read

- The option appeared in the options list with upstream's default and
  help text, but nothing consumed it: the κ_resto the restoration
  sub-solve's early-exit guard runs with was hardcoded to upstream's
  `0.9` in `run_inner_resto`. Setting the option was a silent no-op —
  no error, no warning, no effect (#439).
- The value now flows from the options list through
  `AlgorithmBuilder::resto` into `RestoAlgorithmBuilder` alongside the
  other restoration knobs wired up in #191, and is read at the guard.
  Upstream's square-problem override is preserved and keeps its
  precedence: `IpRestoMinC_1Nrm.cpp:157-163` applies that case by
  *overwriting* the sub-option with `0` before `IpRestoConvCheck` reads
  it, so a square problem still disables the guard regardless of what
  the user asked for.
- Behavior is unchanged for anyone who leaves the option alone — the
  new default equals the previously-hardcoded `0.9`.

### Fixed — pyomo-pounce: constraint-row lookups included a non-row name

- Pyomo's `.row` file lists the `m` constraint rows and then appends
  the objective's name, and the declared-parameter surgery aliases the
  objective along with the constraints it rewrites. The name-to-row
  index built from that file therefore mapped the objective to row
  `m`, one past the last constraint — a name that indexes past the end
  of every per-row array the index is used with. The warm-start suffix
  reader got a bounds check for this in #432; the row names are now
  trimmed where they are read instead, so the pin map and the
  multiplier lookups hold the same invariant structurally rather than
  one call site at a time.
- No public API routed an objective into those lookups (`gradient()`
  sends an objective target to the variable lookup, which rejects it
  by name), so this is an invariant repair, not a user-visible
  behavior change.

### Fixed — warm start: NaN seeds resolved only in densely-stored multiplier blocks

- `warm_start_init_point=yes` treats NaN in a `lagrange` / `zl` / `zu`
  seed as "unseeded, use the solver's own default" (#432). That
  substitution ran only for densely-stored multiplier blocks. A
  compound block kept the NaN, which would then propagate through the
  warm-start clamps into the iterate; the guard against it was a
  `debug_assert!`, compiled out of the release builds that ship.
- NaN resolution now recurses into compound blocks, so the documented
  contract holds for every multiplier-block layout rather than only
  the one the seeding path happens to build. The seeding path builds
  dense blocks today, so no shipped configuration reached the gap —
  it is the contract that is now true, not a solve that changed.

### Fixed — pyomo-pounce: the sens path dropped solver options (#432)

- `SolverFactory("pounce").solve(m, options={...})` and factory-level
  options were silently ignored the moment a model carried
  declarations: the reroute to the in-process session forwarded
  nothing. `max_iter`, tolerances, scaling, everything ran at defaults
  with no signal, so a model stopped being tunable the day it gained a
  declaration.
- Options now flow: factory options first, per-call `options=` on top,
  applied after the tee default so an explicit `print_level` wins. The
  ASL layer's bookkeeping `solver` key is excluded.
- Regression: `max_iter=1` must stop a declared model's solve, from
  both option sources.

### Added — pyomo-pounce: in-process warm starts from the model's suffixes (#432)

- With `warm_start_init_point=yes` (or `True`; `add_option` maps them
  alike) among the options, the sens path reads the model's `dual` /
  `ipopt_zL_in` / `ipopt_zU_in` suffixes into the session's initial
  multipliers, matched by component name, with a constraint replaced
  by the declared-parameter surgery reached through its clone alias.
  Both external sign conventions are crossed on the way in: `dual`
  holds the AMPL marginal `-λ` (#271) and `ipopt_zU_in` Ipopt's
  negative-at-upper `z_u` (#296); the session wants the internal
  `+λ` and non-negative `z_u`.
- Entries the user did not supply are seeded NaN, a new "unseeded"
  marker in the warm-start contract of `Problem.solve` /
  `Solver.solve`'s `lagrange`/`zl`/`zu` arguments: the warm-start
  initializer substitutes its own resolved defaults
  (`bound_mult_init_val`, including the Mehrotra override, for bound
  multipliers; for equality duals the warm path's existing 0, which
  is not the cold path's least-squares estimate), so the defaults
  live in one place. The contract covers the warm-start initializer
  only: the batched solver's multiplier seeds and the SQP
  working-set arrays do not route through it and must not carry NaN. Through Ipopt's ASL interface an
  absent entry reads as a zero multiplier because a dense array
  cannot say "unknown", and a zero bound multiplier on an active
  bound is a contradictory certificate the solver must first recover
  from; a suffix knows which entries exist, so an explicit zero is
  honored and absence means "initialize normally". (POUNCE's own CLI
  reads no `ipopt_zL_in`/`ipopt_zU_in` at all today, so the
  comparison is with Ipopt-via-ASL, not this project's binary.)
- The sens path's results object now reports the iteration count
  (`statistics.black_box.number_of_iterations`).
- Regression: a plain solve's exported multipliers, fed back as
  suffixes, make the declared warm re-solve beat the cold one; the
  reader's fallback semantics are unit-tested (explicit zero kept,
  absent entry defaulted).

### Added — post-solve activity classification on `Solver` (#362, covariance roadmap item 0)

- `Solver.classify_activity()` (Rust: `pounce_sensitivity::activity`)
  classifies every bounded variable and every finite-bounded inequality row
  of the held converged solve as `inactive`, `weakly_active`,
  `strongly_active`, `ambiguous`, or `unidentified`, from the ratio `r = Σ/q`
  of barrier curvature (`Σ = z/s`, summed over the sides that exist) to the
  model's own curvature (`q = |H_ii|` for a variable; for a row,
  `|∇dᵀH∇d|/‖∇d‖⁴`, whose fourth power makes the ratio invariant to
  rescaling the row and to the solver's per-row scaling, so both
  spellings of one limit classify identically at any row coefficient)
  at the converged iterate. `r` is O(μ)
  inactive, O(1) weakly active, O(1/μ) strongly active, so one ratio
  separates the regimes where no fixed threshold on a slack or multiplier
  alone can: both are O(√μ) at weak activity. Edges sit at √μ and 1/√μ with
  a fixed inner band [1e-1, 1e1]; gaps report `ambiguous`; at μ > 1e-4 only
  the two clear calls are made. Curvature below `√ε·max(1, max_j|H_jj|)`
  reports `unidentified`, with the sign of the raw value.
- Inequality rows classify through the same rule as variable bounds, which
  is the gap behind #362: a bound moved onto a row disappears from the
  bound-multiplier view but not from this one. The tests walk the same
  scalar geometry through both formulations and require identical statuses.
- The report is indexed in user space: `var_*` follow the user's `n`
  variables and `row_*` the user's `m` constraints, in their order. A
  variable removed internally by `fixed_variable_treatment = make_parameter`
  (`lb == ub`) reports `fixed` at its own index and an equality constraint
  reports `equality`, so user indices never shift.
- Per-entry honesty flags on variables and rows alike: `off_central_path`
  (`s·z` differs from `μ` by more than 10× on some side) and `contaminated`
  (classified inactive yet `r > 100μ`: `inactive` means `r = O(μ)`, so
  the threshold is μ-relative and reachable).
  `classify_activity` refuses to run with `bound_relax_factor != 0` (the
  Ipopt default is `1e-8`): relaxed bounds shift the slacks the classifier
  reads. The guard tests the value **the held solve ran under**, now
  snapshotted as `ConvergedState::bound_relax_factor`, not the
  application's live options: the bounds were relaxed (or not) once,
  during that solve, so setting the option afterwards neither unlocks a
  relaxed state nor invalidates an unrelaxed one. Re-solve to change the
  answer.
- Documented in `docs/src/sensitivity.md` ("Activity classification", plus
  the scale-invariance note under "Units and NLP scaling") and listed with
  the other session entry points in `docs/src/sessions.md`.
- Item 0 of `dev-notes/covariance-information-roadmap.md` (#262).
  Everything the classifier reads was already retained at convergence
  (`Σ`, the solver's slacks, the bound multipliers, `μ`, and the exact
  Lagrangian Hessian), so the change is exposure plus the rule, not new
  computation. Items 1-4 build on these statuses.

### Fixed — the QP homotopy stepped over crossings its own ratio test had found (#434)

- The §4.2 parametric path's two ratio tests selected the next event with
  `t + dt < t_next - T_EPS`. That margin reads as a don't-bother-for-a-hair
  guard but is not one: it makes a crossing that happens *earlier* than the
  incumbent, by less than `T_EPS = 1e-12`, lose to it, so the step knowingly
  overshoots the earlier crossing. Measured on `QSHARE2B` in #413: row 132
  crossed at `dt = 2.9e-16` and lost to a step of `1.1e-14`.
- Overshooting is not a rounding-level mistake, because violation is
  absorbing. The primal ratio test only ever *prevents* a violation and can
  never repair one, so a row stepped over stays inactive and violated for the
  rest of the path while the direction solve pushes it further out
  (`QSHARE2B` row 7 went `8e-2 -> 0.4 -> 7.5 -> 11 -> 22`). The same
  comparison also discarded crossings *tied* with the incumbent, leaving a row
  sitting exactly on a bound it was not in the working set for — which the
  next direction pushes it across.
- Both ratio tests now feed one `RatioTest`, which compares crossings exactly
  and fires the whole coincident set. It is a separate type because the rule
  is pure arithmetic on two numbers while the loop around it needs a KKT
  factorization per step to reach; `pounce-qp/src/tests/homotopy_unit.rs` pins
  it directly, including the measured `2.9e-16`-vs-`1.1e-14` case.
- The consequence was larger than a numerical detail. `AUG2DC`'s path used to
  reach `t = 0.5` within 50 steps and then stop, spending thousands of KKT
  factorizations without moving the parameter at all, until it hit the time
  cap. It now completes in **104 steps** and the solve returns the published
  optimum. Re-measured across all 138 Maros-Mészáros convex QPs at a fixed
  cap, homotopy-on against the same run's homotopy-off: `AUG2D` and `AUG2DC`
  recover with nothing regressing, cold paths reaching `t = 1` go 92 → 98,
  paths killed mid-flight 37 → 31, and the median completed path halves
  (216 → 102 steps). `QSHARE2B`, the seventh loss recorded on
  `sqp_qp_use_homotopy`, also recovers.
- Scope, because it is easy to overclaim: this fixes paths that *wedge*, not
  the `O(|A|)` pivot cost in #434's title. The `benchmarks/warmstart` `-hom`
  arms reproduce **identically** after it (727 → 1692 inner active-set changes,
  0.43×, tracking the active-set fraction 82% → 0.63×, 5% → 1.00%,
  99% → 0.32×), because the defect never fires on those small non-degenerate
  QPs. That cost is real and remains. It is bounded, though: all arms return
  correct answers with identical outer iteration counts, so it is overhead
  rather than damage, and `use_homotopy` is `false` by default in `pounce-qp`,
  so the SQP inner-QP path does not take it unless asked.
- **No runtime guard is added**, which is what #434 was filed to ask for. The
  losses that remain after this fix cannot be separated from the gains by any
  threshold on (path steps, `t`): the only rule that catches all the reachable
  ones sits 3% above `KSIP`'s completed path length, and firing it also
  abandons `LASER`, which the homotopy solves in 16.3 s against 41.4 s on the
  conventional route. Per the issue's own instruction, the measurement is the
  deliverable rather than a fitted threshold. Recorded in
  `dev-notes/issue-434-homotopy-cost.md`, with the harness kept as
  `crates/pounce-convex/examples/homotopy_sweep.rs` and per-path telemetry
  (steps, final `t`, longest run of steps that did not advance `t`) on the
  existing `POUNCE_HOMOTOPY_DEBUG` trace.

### Fixed — active-set SQP: a warm start was discarded whenever the active set moved (#428)

- `solve_with_working_set` pins the hinted active rows to their new
  boundary values and hands the resulting primal to `solve`. Once the true
  active set has moved — by even a single entry — the hint still pins a row
  that should have been released, so that primal overshoots some *other*
  row by roughly the distance the problem moved. `solve`'s warm-start
  admission pre-check then rejected it and threw the whole hint away for a
  cold l1-elastic phase-1, whose recovery re-solve starts from
  `WorkingSet::cold`. The warm start was therefore either perfect (0
  working-set changes) or catastrophic (≈ one change per constraint row),
  with nothing in between, and the catastrophic branch is the one taken on
  essentially every step of a parameter sweep. Past `m > sqp_qp_max_iter`
  the elastic re-solve could not finish at all, so the warm arm stopped
  returning an answer where the cold arm solved cleanly.
- The pre-check itself was doing something legitimate (a crossover hint
  that violates hundreds of inactive rows really does stall the zero-RHS
  warm inner loop), so the fix is not to relax it — and it cannot be:
  `feas_tol` also gates whether a converged point is *accepted*, and the
  setting that admits the hint reliably also stops rejecting genuinely
  infeasible answers. Instead the hint is now **repaired**: the rows the
  pinned point violates are known, so they are pinned too and the solve
  re-factored, keeping the |A| − 1 entries the hint got right. The
  pre-check keeps its exact meaning and is simply handed a feasible point.
- The repair declines — leaving the old elastic recovery untouched — when
  the hint is not one it can help: an already-active row is violated, the
  violated rows exceed a quarter of the hint's active set (the
  badly-wrong-hint case the pre-check exists for), the repaired pin set
  would exceed `n` rows and so be necessarily rank-deficient, or three
  re-pin rounds do not reach feasibility.
- Measured on a parametric linear-quadratic MPC sweep (n = 32 … 302,
  m = 22 … 202), one θ step per solve: the warm arm previously spent
  exactly as many working-set changes as a cold solve at every horizon
  (10/15/19/20), i.e. the hint bought nothing. It now reaches the same
  optimum to ~1e-13 in 0 changes. Regression tests in
  `crates/pounce-qp/tests/warm_start_pin_repair.rs` cover the analytic
  one-entry-wrong case, the MPC sweep at four horizons, and a hopeless
  hint that must still fall through to elastic.

### Fixed — pyomo-pounce: `covariance(n_data=)` read the SSR from the live objective (#426)

- The `n_data=` branch estimated the noise variance with the SSR taken
  from `pyo.value(objective)` on the current model, which evaluates at
  the model's current variable and Param values: anything written after
  the solve (a measurement, a warm start for the next horizon) silently
  rescaled the reported covariance. Same staleness class as #420, found
  in #421's review. The `declare_residual` path was unaffected.
- The session now stores the objective value at the solve and the
  `n_data=` branch reads that, so post-solve writes to the model cannot
  move the answer. The stored number is the engine's `obj_val`, which is
  `eval_f` on this model's own bridge at the final iterate — unscaled,
  in the model's objective units, i.e. exactly what `pyo.value` returns
  an instant after the solve. Regression: solve, take the covariance,
  overwrite the model's values, take it again; the two must be
  identical.
- The unusable-objective guard tests `isfinite`, not `is None`: the
  engine always reports `obj_val` and signals "never computed" with NaN
  (`0.0` is an ordinary objective value), so a `None` check would have
  been dead code guarding a condition the producer cannot emit.
  Documented in the `covariance` docstring and the sensitivity chapter
  alongside `estimate()`'s matching baseline guarantee.

### Added — a large-scale sparse tier for the warm-start benchmark, and the defect it found

- **`mpc_horizon_200/400/800`** (`--tier large`) carry the horizon sweep's
  linear MPC to n = 602, 1202 and 2402 with a block-banded Jacobian. Nothing
  dense is materialized: families may now declare `sparse_structure()` with
  packed `jacobian_values()` / `hessian_values()`, and the convex-QP arm
  receives sparse matrices — at N = 800 a dense Hessian alone would be 46 MB
  rebuilt every iteration, and dense data costs the QP solver 60–80× by its
  own diagnostic. The self-test finite-differences the declared structure
  column by column and cross-checks it against the dense path wherever both
  exist.
- **The large tier found [#428](https://github.com/jkitchin/pounce/issues/428)
  on its first run** (fixed above). The warm-started SQP was returning
  `Maximum_Iterations_Exceeded` with zero outer iterations on 7 of 8 steps at
  every one of N = 200/400/800, while every other arm solved cleanly; inner
  working-set changes ran 0 → 43 → 164 → 403 → 795 → 1589 against a cold arm
  flat at 66. On the fixed solver the warm arm is flat at **3** across a 75×
  range of m, and its wall time is 0.02–0.03× its cold twin on the large
  tier — the fastest arm on the board where it previously did not produce an
  answer.
- **This retracts the horizon sweep's conclusion, not its measurements.** The
  reported crossover — warm/cold wall time 0.84 → 1.29 → 1.95 → 2.57 turning
  harmful above N = 20 — was this defect. Re-measured on the fixed solver the
  same sweep runs 0.24 → 0.12 → 0.11 → 0.10 at `large` and 0.17 → 0.08 →
  0.04 → 0.02 at `tiny`: the ratio *improves* with horizon, because cold cost
  grows with the problem while warm cost is set by how far the active set
  moved. The churn numbers behind the original claim were correct and are
  unchanged; the "absolute churn inflates with size" mechanism built on them
  was not, and is withdrawn.
- All the suite's other headline numbers moved with the fix and have been
  re-measured: `warm-sqp` total solve time over the 42-row sweep drops from
  21.82 s to 3.46 s on identical iteration counts, and per-family inner-work
  ratios rise across the board (`mpc_horizon_80` @ `tiny` 54.75×,
  `nmpc_vanderpol` @ `tiny` 18.80×).

### Fixed — `sqp_qp_use_homotopy` was registered but never read

- **Setting it on the active-set SQP path did nothing.** The option was
  registered with the parametric-homotopy work and documented in detail, but
  `apply_qp_subproblem_options` — the function that maps the `sqp_qp_*` family
  onto `pounce_qp::QpOptions` — never consulted it, so `pounce-qp`'s own
  default (`false`) always stood. Only `pounce_convex::active_set` got the
  homotopy, and it sets the field directly in Rust rather than through options.
  A registered knob that no code reads is worse than a missing one: it
  validates, accepts a value, and ships working documentation for behavior the
  user never gets.
- **The inverse of #360, and invisible to its guard.** That issue fixed
  read-but-unregistered and left a test walking the keys the *reader* consults,
  asserting each is registered. Nothing checked the other direction. The new
  `application_every_registered_sqp_qp_option_is_read_by_the_subproblem_reader`
  enumerates the registry and fails if the two sets diverge either way; both
  guards were checked for falsifiability rather than assumed.
- **Found by the benchmark, not by reading.** The warm-start suite's new
  `-hom` arms measured bit-identical results to their twins — an arm that
  cannot differ from its control is either a perfect null or a broken
  experiment.

### Added — an MPC horizon sweep, and the crossover it locates

- **`mpc_horizon_10/20/40/80`** — one linear-quadratic MPC at four horizons
  (n = 32 → 242, block-banded Jacobian), so reading down them isolates problem
  size from every other property. This closes the suite's last uncovered axis.
- **The active-set SQP's warm-start advantage is eroded by absolute working-set
  churn, and problem size inflates it.** Warm/cold wall time goes 0.84 → 1.29 →
  1.95 → **2.57** across the four horizons at large parameter steps — at N = 80
  a warm-started SQP solve takes 2.6× longer than solving cold. At small steps
  it stays excellent at every horizon, including the suite's best single row
  (0.08 at N = 40, twelve times faster than cold). The interior-point arms stay
  between 0.25 and 1.00 across the whole grid.
- **Mechanism, from the working sets:** the *fraction* of the active set that
  changes per step is horizon-independent (~3% at large steps for every N), but
  the *absolute* count grows with the problem (1.05 → 5.58 changes/step), and
  each change costs more as the active set grows. An active-set method pays for
  the absolute count, so two factors multiply.
- This puts a number on the qualitative caveat in the active-set SQP docs
  ("prefer the IPM for large-scale problems with thousands of active
  inequalities"): the measured crossover is tens to low hundreds of active
  constraints, not thousands. The docs now say so.

### Added — degeneracy families and a parametric-homotopy arm in the warm-start suite

- **Two degeneracy families**, completing the three distinct ways an active-set
  QP meets degeneracy. `degenerate_corner` already covered dual degeneracy (a
  multiplier through zero); `redundant_rows` adds rank deficiency (duplicated
  equality rows, so LICQ fails everywhere, plus a duplicated inequality pair
  that activates together mid-path), and `degenerate_vertex` adds primal
  degeneracy (12 rows tight at a 4-variable vertex — the ratio-test ties Harris
  and GMSW EXPAND exist for). Both are convex QPs, so all three solvers take
  them; both converge on all eight arms with zero correctness failures, and the
  reported working set confirms the LICQ-violating vertex is pruned to its
  maximal independent subset.
- **`cold-sqp-hom` / `warm-sqp-hom` arms**, differing from their twins in the
  single option above, which makes the §4.2 parametric homotopy measurable for
  the first time. It is a sharply mixed trade: 4.2–12.3× less inner active-set
  work on `redundant_rows`, 2.0–2.9× on `degenerate_corner`, unchanged on four
  families, and ~2× *more* on `nmpc_vanderpol` and ~1.4× more on
  `simplex_proj`, for 0.70× over all 30 rows. It wins on the degenerate,
  netlib-like geometry it was designed for and loses on well-conditioned
  MPC-shaped QPs — a mechanism-level account of #412's 20-gained/7-lost
  Maros-Mészáros result, and an argument for the default staying off while the
  knob is reachable.

### Fixed — pyomo-pounce: `estimate()` measured its perturbation from the Param's current value (#420)

- `estimate(model, perturb)` computed each step as `new value` minus the
  Param's current value on the model. The factorization the step runs
  through describes exactly one point, the solve point, so the two agree
  only while the Param is untouched. In the receding-horizon pattern the
  caller solves at a prediction, writes the arriving measurement into the
  Param, then asks for the estimate at that value: the delta came out
  zero and `estimate()` silently returned the unperturbed solution. No
  error, no warning, and the output is a valid solution (at the
  prediction), so nothing looked wrong downstream.
- The baseline is now the pin constraint's stored right-hand side: the
  value the perturbation actually shifts, holding the Param's solve-time
  value exactly, already retained. A caller that has not touched the
  Param sees the same numbers as before; no new state is stored. The
  docstring and `docs/src/sensitivity.md` state the baseline semantics.
- Regression: solve, write the new value into the Param, ask at that
  value, compare against a re-solve there. Fails on the old baseline
  (zero delta), passes now.

### Added — a benchmark for warm starting, and the counter that makes it measurable

- **`benchmarks/warmstart/` — the first suite in the tree that is not a
  cold-solve set.** Every existing suite measures one problem, one solve,
  from scratch; warm starting only means anything over a *sequence* of
  related solves. The unit of work here is a parametric family plus a
  scripted path through its parameter space, solved end to end by four arms
  (`cold-ipm`, `cold-sqp`, `warm-ipm`, `warm-sqp`) so the warm-start effect
  is separated from the algorithm change, plus two more (`cold-qp-ipm` /
  `warm-qp-ipm`) that route the QP-shaped families through the dedicated
  convex solver for a three-way comparison. Eight families cover the active-set
  regimes that decide whether warm starting pays — stable, flipping,
  degenerate (a path that passes exactly through a zero multiplier), a clean
  activation switch, re-activation from an empty working set, an entirely
  unconstrained path (`m = 0`, the zero mark the other speedups are read
  against), and a closed-loop NMPC sequence — each at three step sizes,
  because payoff is a function of how far the problem moved. Nothing
  outside `adapters/` imports a solver. See `benchmarks/warmstart/README.md`
  and `dev-notes/warm-start-benchmark.md`; a survey of the existing
  literature is in the latter (nothing public covers this case).
- **`info["n_qp_solves"]` and `info["n_qp_ws_changes"]` on the Python
  path**, backed by `SqpResult::n_qp_working_set_changes` and
  `SolveStatistics::sqp_qp_solves` / `sqp_qp_working_set_changes`. The inner
  active-set work an SQP warm start exists to avoid was previously not
  observable from any user-facing surface: `pounce-qp` counted per-QP
  working-set changes, nothing accumulated them. Outer `iter_count` is not a
  substitute — on a QP-shaped NLP the outer loop terminates in one iteration
  warm or cold, so a cold/warm comparison reads exactly 1.00× while the
  inner work differs by an order of magnitude (`simplex_proj`: 313 → 4
  active-set changes; `nmpc_vanderpol`: 931 → 66). Both keys are 0 on the
  interior-point path, which solves no QP subproblems.
### Fixed — the active-set QP driver lost its cold-start fallback when the homotopy's feasibility invariant broke (#413)

- **#412's parametric homotopy does not keep `x(t)` feasible, and its own
  documentation said it did.** The module header asserted that `x(t)` is
  feasible for the `t`-problem "at every point on the path by construction".
  Measured on Maros-Mészáros `QSHARE2B`, 14 rows are crossed uncapped on a
  single path and the worst grows `8e-2 → 0.4 → 7.5 → 11 → 22` on the way to
  `t = 1`. The path's primal ratio test only ever *prevents* a violation — a row
  whose gap has gone negative yields `dt < 0`, which the test discards — so a
  crossing is never repaired and compounds. Two mechanisms, both measured: the
  rank-repair tabu hides a row from the ratio test rather than only from the add
  decision (10 of 14), and events coincident in `t` or below the `T_EPS` floor
  lose the strict comparison that would cap the step (4 of 14).
- **The damage was not a slow solve, it was a *seedless* one.** That false claim
  was load-bearing: `pounce-convex`'s driver switched off its simplex phase-1
  vertex seed whenever the homotopy is on, "because the homotopy is itself the
  cold-start mechanism". When the path's prediction turned out unusable, the
  engine therefore had neither — and cold-started the l1-elastic phase-1 that
  `pounce-qp` documents as not terminating on the degenerate netlib-derived QPs
  in this set. On `QSHARE2B` that spent the whole iteration budget to return
  `4854` against a published `11703.7`, still carrying a constraint violation of
  `20`; the seeded route solves it in 52 iterations.
- **The seed is restored as a last-resort retry**, after the existing
  Ruiz-equilibration retry rather than before it, so both earlier attempts stay
  bit-identical and nothing that already solved can be displaced — including by
  the clock, which is how an added stage regresses a benchmark even when its
  logic cannot. (Ordering was measured: placed first, it took `QPCBOEI1` from
  `0.68 s` to a 60 s timeout.)
- **The path reports its feasibility loss but does not abandon the path.**
  Abandoning was implemented and measured *worse*: the corrector is a genuine
  corrector and often recovers, so losing feasibility degrades the prediction
  rather than invalidating it (`QSHIP04S` reaches a violation of `7.1e4` at
  `t = 0.5` and still solves to the published optimum). Repairing the crossings
  properly needs an exchange pivot at the degenerate vertex and is left to
  follow-up.
- **Also fixes a pre-existing infeasibility-reporting bug this exposed.**
  `cold_general_initial` skipped every `bl == bu` row in its feasibility sweep,
  on the reasoning that a pinned equality is satisfied by construction — true
  only for the rows actually *kept*. An equality its own rank guard pruned as
  linearly dependent but *inconsistent* (`x₀+x₁ = 1` with `x₀+x₁ = 3`) was never
  re-checked, so the model came back `NumericalFailure` instead of routing to
  the elastic phase-1 that certifies it `PrimalInfeasible` — a `500` where
  callers needed a `200`.
- Maros-Mészáros active-set column, 60 s cap: **58 → 59 solved, zero lost, zero
  solved-but-wrong**. This addresses the mechanism behind #413's timeouts and
  one of the instances; the bulk of the 49 remain and are **not** corrector-bound
  as the issue hypothesised.
- **What the measurement rules out, for whoever picks this up next.** 43 of 138
  paths lose feasibility, and they solve 19% of the time against 59% for the 95
  that stay clean — so the path really is the discriminator. But the obvious
  repair is a dead end: a degenerate-vertex exchange pivot (active set split from
  the KKT working set, promotion into the working set, plus a post-step capture
  giving the ratio test the repair it lacks) was implemented and measured on
  exactly those 43 at a 120 s cap, and moved **9/43 solved to 9/43**, repairing
  none of the broken paths. It buys one large win (`STADAT2`, 10979 corrector
  iterations → 22) and one real regression (`QBEACONF`, 0.43 s → 8.23 s). Kept on
  the `experiment/413-exchange-pivot` branch rather than merged.
- **And the other half of #413's timeouts is not a defect at all — it is the
  method's complexity class.** The 39 remaining unsolved instances have
  perfectly *clean* paths, including all six #412 regressed (`AUG2D`, `AUG2DC`,
  `CONT-050`, `CONT-100`, `DTOC3`, `STADAT3`); they are all large, and they time
  out because the **optimal active set is a constant fraction of `n`**:

  | instance | `n` | active at the optimum |
  |---|---|---|
  | LISWET1 (and 11 siblings) | 10002 | 10000 (100%) |
  | POWELL20 | 10000 | 10000 (100%) |
  | QSHIP08L | 4283 | 4200 (98%) |
  | CONT-050 | 2597 | 2402 (92%) |
  | CVXQP1_M | 1000 | 887 (89%) |
  | DTOC3 | 14999 | 7901 (53%) |
  | AUG2D | 20200 | 10000 (50%) |

  An active-set method changes the working set by one row per pivot, so from a
  cold start these need 10³–10⁴ pivots, each at minimum a back-solve on an
  O(`n`) KKT. The interior-point engine takes **21 iterations on LISWET1
  regardless of size** — which is why `auto` routes cold convex QPs there and
  why pounce solves 137/138 overall. No amount of work on the active-set path
  closes an Ω(`n`)-versus-O(1) iteration-count gap.

  #413's headline — that the timeouts are "corrector-bound, not path-bound" —
  does not hold for these either: at a 60 s cap `LISWET1` reaches only
  `t = 0.976` after 7050 path steps, `AUG2D` only `t = 0.50`, `QSHIP08L`
  `t = 0.655`. The path *is* the runtime. The 4% figure in the issue was
  profiled on `QSCTAP1` (n=480) and `QSCSD6` (n=1350) and does not generalise.
  The Schur-update remedy that observation rules out was nonetheless tried here
  (rank-2 updates of a cached K_max, the machinery `solve_general_schur`
  already uses) and it made the path **slower** — 4850 steps in 60 s against
  7050 — because K_max is fixed at dimension `n + m + n` and the SMW solve plus
  refinement costs several back-solves per step against one factorization of a
  smaller active-set KKT.

### Fixed — exact-Hessian SQP gave up on unconstrained nonconvex NLPs (#423)

- **`algorithm = active-set-sqp` with `sqp_hessian = exact` no longer stops at
  the first indefinite iterate of a problem with nothing to block a
  negative-curvature direction.** The reported case — a chain of coupled
  double wells, `n = 12`, `m = 0`, no finite bounds, started inside the
  negative-curvature region — solved to `f = 0.027424` in 24 outer iterations
  before #419 and afterwards exited `Search_Direction_Becomes_Too_Small` at
  outer iteration **1**, at `f = 26.0257`: ~950× the optimum, and the value at
  a point one step from the start. It converges to `f = 0.027424` in 24
  iterations again.
- **The regression is #419's own mechanism reaching a case it has no answer
  for.** #419 made an unblocked non-positive-curvature direction a recession
  certificate, which is right for a standalone QP and right for #416's
  Rosenbrock, where the direction runs to a bound and pivots. But a *step* QP
  is unbounded below at every indefinite iterate that has nothing to block
  that direction — and with `m = 0` and no finite bounds, nothing can ever
  block, so every indefinite iterate produced one. The driver then re-verified
  the ray against the true NLP (#388), correctly found the quartic objective
  bounded below, and had no third branch: not-unbounded meant `QpStepFailed`.
  The two issues are mirror images — #416 was "the δ-shifted step is capped at
  α = 1, so the solver spins without pivoting"; this was "the δ-shifted step is
  gone, so a solver with nothing to pivot *to* has no step at all".
- **An unbounded model on a bounded NLP is a signal to regularize, not to
  stop** (Nocedal-Wright §18.4) — and δ from §4.5 inertia control already *is*
  that regularization. When the ray fails NLP re-verification the driver now
  re-solves the same subproblem with the certificate declined
  (`QpOptions::certify_recession_ray = false`, new, default `true`): the shift
  stays, the unblocked direction takes the δ-shifted proximal step (`α = 1`),
  and the QP returns a point. `QpStepFailed` is reported only if even that
  produces nothing. #419's ratio-test cap is untouched wherever a bound exists,
  so #416 and every Rosenbrock row stay exactly where #419 left them, and a
  genuinely unbounded NLP still reports `Diverging_Iterates` — the re-solve
  happens only *after* the ray has failed re-verification.
- Standalone `solve_qp` is unaffected: `certify_recession_ray` defaults to
  `true`, because the point of solving a QP is to learn whether it has a
  minimizer.

### Fixed — the active-set QP crawled instead of stepping on an indefinite Hessian (#416)

- **`algorithm = active-set-sqp` with `sqp_hessian = exact` now solves plain
  Rosenbrock at the default QP budget.** The reported case — extended
  Rosenbrock in a ball, `n = 10`, from the canonical `(−1.2, 1, …)` start —
  gave up after 4 outer iterations with
  `Search_Direction_Becomes_Too_Small`/`Maximum_Iterations_Exceeded` at
  `f = 9.62214`, `KKT = 2.62`, against `f = 3.9866` from the interior-point
  path. It now converges to `f = 3.9866` in 20 outer iterations, and does so
  ~2.5× faster than the documented `sqp_qp_max_iter = 250` workaround.
- **The budget was never the problem.** The tell in the report was that the
  smallest cap that worked was 250 whether `n` was 10 or 40, and that the QP
  made **zero** working-set changes while spending 200 iterations. Both follow
  from a single line: after §4.5 inertia control factors `H + δI`, the inner
  loop still capped its step at `α = 1`. The unit step is the model minimizer
  only for an *unshifted* Newton direction — writing `r = Hx + g`, the shifted
  system gives `pᵀr = −(pᵀHp + δ‖p‖²)`, so the model's own minimizer along `p`
  is `α* = 1 + δ‖p‖²/pᵀHp`, and `+∞` when the true curvature along `p` is
  non-positive. Capping at 1 turns the loop into proximal-point iteration with
  parameter δ, whose per-eigenvalue contraction is `δ/(λ + δ)`. Since δ is
  reached by multiplying `QpOptions::inertia_shift_initial` (1e-8) by
  `inertia_shift_factor` (100) until the system is PD, it overshoots the
  spectrum badly: on the reported QP `λ_min = −1.4` and δ = 100, putting
  every factor within 3 % of 1. The trace shows it exactly — δ = 100,
  `‖p‖∞ ≈ 2e−3`, `α = 1`, `pᵀHp < 0`, empty working set, 200 times over. The
  crawl is dimension-independent because δ and the spectrum are, which is why
  raising the cap looked like it fixed something.
- **The fix is to take the step the model asks for.** The ratio test is now
  capped at `α*` instead of at 1, so a negative-curvature direction runs to
  its blocking bound and *changes the working set* — what an active-set method
  is supposed to do with negative curvature (Nocedal-Wright §16.5). `δ = 0`
  returns 1.0 without touching the data, so every non-shifted solve is
  bit-identical. Applied on all four inner loops (box, equality-plus-bounds,
  general, and the Schur-update variant, which now tracks the δ in its cached
  base factor).
- **A negative-curvature direction with nothing to block it is unboundedness**,
  and is now reported as such — the F2 recession certificate with `pᵀHp < 0`
  in place of `Hp = 0`. Two fixtures that the SQP path previously ran out of
  iterations on, `unbounded_exp.nl` (`min −exp(x)`) and `unbounded_cubic.nl`,
  now exit `Diverging_Iterates`, agreeing with the IPM path. The SQP driver
  still re-verifies the ray against the true NLP before reporting it.
- **Also fixed, found by the above:** l1-elastic phase-1 labelled its result
  `Optimal` whenever the slacks reached zero, even when the phase-1 solve had
  hit its own iteration limit on the way. Past the point where the slacks
  vanish the augmented objective *is* the original one, so an unconverged
  phase-1 leaves an ordinary suboptimal iterate — `afiro` with
  `sqp_qp_max_iter = 3` returned `Optimal` carrying a KKT error of 10 at
  objective 440 against a −464.75 optimum. The inner verdict is now carried
  out; the point is still returned, just not dressed up.

### Fixed — the convex QP IPM certified `Optimal` at a non-KKT point when the *variables* spanned many decades (#414)

- **`solve_qp(method="ipm")` returned `status="optimal"`, `success=True`,
  objective `67.13` — with `kkt_error = 8282.5` on the very same result
  object**, against a true optimum of `-3.9585018079`. Through the CLI the same
  engine reported `SolveSucceeded` / `solve_result_num=0` / exit `0`, so the
  AMPL, Pyomo, and GAMS drivers all accepted the wrong point as a solution.
  `solver_selection=socp` and `auto` route to the same engine and inherited it.
  pounce's own `qp-active-set` and `nlp` engines, clarabel, and scipy
  `trust-constr` all solve the identical model correctly.
- **The instance is not hard — it is badly *stated*.** Variables scaled
  `10^-6‥10^6` give `cond(P) ~ 1e24`, but one diagonal rescaling `z = x/s`
  takes it to `cond = 10`. The failure threshold tracked exactly that spread:
  correct at `±3` decades, wrong from `±4` up.
- **Root cause: the convergence test was measured in a metric that does not
  bound the error.** HSDE certifies on *scale-relative* residuals once the
  problem's natural scale puts absolute `tol` accuracy below the
  finite-precision floor. Those normalizers are **global** ∞-norms, so once the
  variable scales spread, the worst-scaled column dominates `‖Px̂‖` and dividing
  every component's residual by it hands a blanket relaxation to the components
  where the real violation lives. `±3` decades stays under the gate that opens
  the relative arm, which is why it was correct.
- **The check now runs in the Ruiz-equilibrated metric**, where every variable
  and row carries an `O(1)` scale and no column can mask another. The reported
  point reads `1.2e2` there and the true optimum of the same problem `2.9e-10`
  — against `2e-4` for *both* in the unscaled metric, which is why the existing
  #324 relative re-check could not see it. A rejected `Optimal` is repaired by
  an equilibrated re-solve, the same repair #293 already applies to a
  *non-converged* HSDE solve; on the reported instance it returns the oracle's
  `-3.958501808`.
- **Never a false success.** If the re-solve cannot certify a genuine optimum
  either, the verdict is *demoted*, not upgraded — `OptimalInaccurate` would
  still report `ok` / exit `0` through the CLI, which a relative residual of
  `1e-3` or worse is not.
- **Complementarity is normalized by the objective, not the gradient.** Its
  terms `ŝᵢẑᵢ` are the duality gap's and survive the diagonal congruence
  unchanged while the gradient scale does not, so normalizing it by a gradient
  scale Ruiz has pulled to `O(1)` rejects the genuine #286 huge-magnitude
  optima. Measured: the #414 false optima land in `2e-2‥1.2e2`, their repaired
  counterparts in `1e-12‥1e-9`, and the #286 solves — the genuine optima most
  at risk of being rejected here — at `4e-10` and `1.5e-8`.
- Costs nothing on a solve that reaches absolute `tol` accuracy, which
  short-circuits before any of this runs.

### Changed — warm-started convex QPs converge in 35–60% fewer iterations (#417)

- **The warm start was never the bottleneck; the *static* fraction-to-boundary
  parameter was.** With `τ = 0.95` fixed, every accepted step covers at most 95%
  of the distance to the cone boundary, so once the Newton direction stops being
  the limit, μ and the residuals fall by a fixed ~20× per iteration. The
  iteration count is then `log₂₀(μ₀/tol)` *regardless of the starting point* — a
  warm start can only lower μ₀, buying a logarithm of the perturbation instead
  of the one or two Newton steps a nearby problem deserves. Warm traces showed
  it plainly: `α_p = α_d = 0.950` exactly, from the second iteration to the last.
- **The direct driver's corrector now takes the standard Mehrotra tail**,
  `τ = clamp(1 − μ, tau, tau_max)`, so τ → 1 as the solve converges and a
  near-optimal iterate takes a near-full Newton step. Measured over 20-step
  warm sequences at three perturbation sizes, mean warm iterations per step:
  `simplex_proj` 5.0–7.2 → 2.4–4.7, `moving_bound_qp` 6.0 → 2.0–4.2,
  `degenerate_corner` 5.0–6.0 → 2.0–3.0. Every warm solve still returns
  `optimal` with the objective matching its cold solve to 1e-6 relative.
- **Scoped to nonnegative-orthant blocks, deliberately.** The same rule applied
  to every cone kind fails `second_order_cones_agree_across_drivers` — the
  direct driver loses ~60% of the SOC instances it solves, because an SOC/PSD
  block's boundary is curved and its Nesterov–Todd scaling blows up as the
  iterate approaches it. Second-order and PSD blocks therefore keep the static
  τ, as does the predictor step (whose step lengths feed Mehrotra's
  σ = (μ_aff/μ)³ heuristic) and the HSDE driver used by cold solves, where the
  step is also limited by the τ/κ ray and the idea needs its own study.
- **Cold solves are unchanged** — they run the HSDE driver, a different loop.
- New `QpOptions::tau_max` (CLI `qp_tau_max`, Python `solve_qp(tau_max=)`)
  caps the tail; setting it equal to `tau` restores the previous static
  behaviour exactly. `tau` itself is now reachable from Python too
  (`solve_qp(tau=)`), which it was not before.

### Fixed — the convex IPM reported some infeasible models as unbounded

- **Found while fixing #415**, by the randomized sweep written to check that
  fix: on 8 of 200 infeasible-by-construction models, `qp-ipm` returned
  `DivergingIterates` / `solve_result_num=300` where HiGHS and pounce's own
  active-set engine both said the model is infeasible. Wrong family, same harm
  as #415 in the other engine — `300` sends a caller looking for an unbounded
  objective on a model that has no feasible point at all.
- **Not a bug in the certificate.** `DualInfeasible` rests on a recession
  direction `d` with `Pd ≈ 0, Ad ≈ 0, −Gd ∈ K, cᵀd < 0`, and that certificate is
  about the **dual**. It stays valid when the primal is empty — the recession
  direction of an empty feasible set exists just the same — so a model can
  honestly earn *both* verdicts at once, and these did. Which one gets reported
  is then a choice, not a measurement, and it was being settled by whichever
  residual gate happened to clear first.
- **On the instance now pinned as a regression test**, the Farkas value held at
  `−1.72` with `z ∈ K*` while its residual fell `1.9e-3 → 9.5e-5 → 4.7e-6 →
  2.4e-7` toward its `8.6e-11` gate — and the recession gate opened with three
  orders still to go.
- **Deciding it inside the iteration means picking a tolerance, and that was
  tried and rejected.** A rule loose enough to catch this case also suppressed
  11 of 200 *genuine* unbounded verdicts, trading one wrong answer for more
  missing ones. So the question is now asked directly instead of inferred: on a
  `DualInfeasible` verdict the driver re-solves the objective-free twin
  (`P = 0, c = 0`), which has the same feasible set and, having no objective,
  cannot be unbounded. An infeasible twin corrects the verdict; anything else
  leaves it alone. Costs one extra solve, only on `DualInfeasible`, and the twin
  cannot recurse into the same branch (`c = 0` admits no `cᵀd < 0`).
- Measured after the change: all 200 infeasible models report
  `primal_infeasible` and agree with the active-set engine, and 200
  feasible-and-unbounded models still certify `dual_infeasible` at exactly the
  pre-change rate — the correction costs nothing.

### Fixed — `qp-active-set` reported an infeasible model as a solver crash (#415)

- **An infeasible QP now exits `InfeasibleProblemDetected` / `solve_result_num
  = 200`, not `InternalError` / `500`.** The trigger is as plain as it gets: an
  LP with `x₀ + x₁ ≤ 1` and `x₀ + x₁ ≥ 3`. Every other engine on the identical
  `.nl` — `lp-ipm`, `qp-ipm`, `socp`, `auto`, `nlp` — said `200`; only
  `qp-active-set` said `500`. The two codes are different AMPL families and
  callers branch on them: `200` means "the model is infeasible, fix the model",
  `500` means "the solver broke, retry or switch solvers". So a correctly
  diagnosed model was being reported to AMPL, Pyomo, and the GAMS links as a
  POUNCE failure. Same fix on the Python surface, where
  `solve_qp(method="active-set")` returned `numerical_failure` where
  `method="ipm"` returned `primal_infeasible`.
- **The engine had it right; the driver threw the answer away.** `pounce-qp`
  returned `Infeasible`, and the convex driver deliberately refuses to propagate
  that on the engine's word — an infeasibility verdict is a proof obligation
  (#282: `DUALC1` is feasible and the phase-1 elastic mode calls it infeasible).
  But the driver had no way to *check* the claim, so it downgraded every one of
  them to `NumericalFailure`. Correct claims and false ones got the same
  treatment.
- **Why the textbook Farkas test could not be used as-is.** It wants
  `Aᵀy + Gᵀz = 0` with `bᵀy + hᵀz < 0`. These multipliers come from an l1-elastic
  phase-1 that minimizes the *original objective plus* `γ·(violation)`, so its
  stationarity leaves `Aᵀy + Gᵀz = −(Px + c)` — a residual that never vanishes,
  only shrinks relative to `‖(y,z)‖ ∝ γ`. On the LP above that is `1e-6`
  relative against a `FARKAS_RESID_TOL` of `1e-10`: a real certificate, read as
  noise.
- **What replaced it.** For any feasible `x`, `qᵀx ≤ bᵀy + hᵀz` where
  `q := Aᵀy + Gᵀz`; a feasible `x` is also in the box, so
  `qᵀx ≥ L := Σᵢ min(qᵢ·lbᵢ, qᵢ·ubᵢ)`. `L > bᵀy + hᵀz` therefore proves the
  feasible set empty. This is the Farkas test generalized — with no finite
  bounds it collapses back to it — and on a boxed problem it accounts for the
  `−(Px + c)` residual *exactly* instead of tolerating it. When there is no box
  to work with, the driver spends one more solve on the objective-free twin
  (`P = 0, c = 0`), whose phase-1 minimizes violation alone and so returns a
  residual-free certificate (`q = (0,0)` exactly on the same LP).
- **Soundness is preserved, which is the point.** Every step is an inequality
  that holds at every feasible point, so a pass is a proof up to floating point;
  a claim that does not verify is still demoted exactly as before. A false
  `PrimalInfeasible` would be a wrong statement about the user's model — worse
  than any failure status — so the negative case is tested directly: a QP whose
  feasible set is the single point `{0}` (every row active, no interior, the
  #282 geometry) must never acquire an infeasibility verdict, and a
  minimizing-corner slip in the box bound is caught by its own test.
- This is the infeasible analogue of the same status-mapping gap #388 fixed for
  unbounded problems and #313 fixed for rank-deficient equalities.
- Two side effects worth knowing about: a certified `PrimalInfeasible` or
  `DualInfeasible` now ends the solve instead of burning a second, equilibrated
  attempt that could not have overturned a proof; and a `PrimalInfeasible`
  solution's `y` / `z` now hold the multipliers that actually certify it, so the
  verdict is re-checkable from the solution it is attached to.
- Measured on 200 infeasible-by-construction models (random `n`, `m`, Hessian,
  and bound pattern): `qp-active-set` went from agreeing with `qp-ipm` on almost
  none of them to `primal_infeasible` on all 200. On 200 feasible-by-construction
  models it produced no false infeasibility verdict and matched the IPM objective
  throughout.

### Fixed — the active-set QP's Schur-update path was unreachable from any user-facing surface

- **`sqp_qp_use_schur_updates` and `sqp_qp_max_schur_updates_before_refactor`
  are now registered options.** `pounce-qp` has implemented Schur-complement
  rank-2 working-set updates (Kirches 2011 / qpOASES-extended) for some time,
  but the only way to enable them was `SqpAlgorithm::with_qp_options` — a
  library-only entry point. No CLI user, no `.nl` run, and no benchmark could
  reach it. That is the same unreachable-knob defect #360 fixed for the rest of
  the `sqp_qp_*` family, and the guard test added there now covers these two.
- **What it costs to leave off.** With updates disabled, every working-set
  change assembles a fresh active-set KKT and factors it from scratch,
  repeating the full symbolic analysis (fill-reducing ordering + MC64
  matching). Profiling `Q25FV47` put **32% of total runtime** in symbolic
  factorization, across 1000+ factorizations. Enabling updates: 19.7s → 0.5s.
  Measured 28-88× on the instances tried.
- **Why the default nevertheless stays `false`.** The speedup above is measured
  on instances that fail either way — a biased sample. Re-running the 46
  Maros-Mészáros instances the default path solves *correctly*, with updates
  enabled: **9 regress** (4 `InternalError`, 2 stalls, 2 timeouts, 1 wrong
  objective) and total wall time rises 107s → 251s. So this is opt-in for
  warm-started workloads where the speedup dominates, not a general
  accelerator. The option's own documentation records those numbers, and the
  test asserts the default rather than leaving it to drift.
- Caching the *symbolic* analysis was tried first and does not work: the KKT
  pattern genuinely changes on every working-set change (instrumented:
  0 cache hits, 1000+ misses, dimension oscillating 1620/1720/6365). The
  reusable object is the factorization, which is exactly what the Schur path
  maintains.
- **Not fixed here.** Enabling updates does not make the failing instances
  solve; it changes `Search Direction is becoming Too Small` into
  `KKT matrix is singular (LICQ violation or rank-deficient Jacobian)`, faster.
  The underlying rank-deficient active-set KKT on degenerate Netlib-derived
  QPs is a separate defect, tracked on its own.

### Tests — two correctness ratchets silently measured whatever `pounce` was on `PATH`

- **`test_infeasibility_no_false_positives` and `test_scale_invariance` now
  refuse to run against a binary this checkout cannot vouch for.** Both resolved
  the solver with a bare `shutil.which("pounce")` and `subprocess.run`, so they
  measured whatever happened to be installed. Found while fixing #403: a
  pip-installed `pounce` from the previous day reported **49 of 200**
  feasible-by-construction models in the AMPL infeasible band, against a ratchet
  whose limit is `0`.
- **Neither existing guard could reach this.** The #366 `conftest` hook covers
  tests that go through pyomo's `SolverFactory` and fires only on
  `ApplicationError: ... did not exit normally` — the solver producing *no*
  result. These two use neither: no `SolverFactory`, so the plugin's
  bundled-binary resolution never runs; and a foreign binary that answers
  *incorrectly* exits cleanly and writes a valid `.sol`. That narrowness is
  deliberate and still correct — #366's docstring argues a guard broad enough to
  swallow wrong answers makes the suite meaningless — it just leaves this gap.
- **Resolution path is not evidence; the build is.** The first cut preferred the
  wheel-bundled binary, on the documented reasoning that it is "by construction
  the build under test". That holds in CI, which stages it fresh, and fails in a
  working tree, where `python/pounce/bin/pounce` is gitignored and survives
  across days and commits — the binary that produced the 49 false positives was
  sitting exactly there, one day and six commits stale. Both builds reported
  `0.9.0`; only the embedded commit distinguished them (`10a6fe0c+dirty` vs
  `e17b0279`), which is what `_build_id` exists for.
- The new `pounce_exe` fixture compares the resolved binary's build against the
  checkout's `HEAD` and skips with an actionable message naming both. Two
  deliberate asymmetries:
  - **It refuses only on a *proven* mismatch** — both ids readable and
    different. An unreadable id must never skip, because a skipped ratchet
    proves nothing and is the same silent loss by another route.
    `crates/pounce-cli/build.rs` embeds `unknown` when it builds outside a git
    checkout (a wheel built in a container without `.git`), and CI is where a
    silent skip would hurt most.
  - **The `+dirty` flag is ignored**, so a clean build against an edited tree
    still runs. That staleness is real but narrow, already covered by the
    source-mtime guard, and failing on it would fire during ordinary
    edit-build-test work — which trains people to bypass the guard.
- `POUNCE_TEST_EXE` names a binary explicitly and steps the guard aside.
  Explicit beats bypassing: it records *which* binary was meant, where `PATH`
  manipulation records nothing.
- The guard is load-bearing, so it has its own coverage:
  `pyomo-pounce/tests/test_exe_guard.py`, ten cases over both directions.

### Fixed — `verify` accepted a `.sol` that violates a bound past the opposite sentinel (#403)

- **`pounce verify` no longer reports ACCEPTED for a solution outside the
  model's declared box.** `verify` exists to be the *independent* check on a
  `.sol`, so a checker that under-reports is worth more than its blast radius
  suggests — this is the surface a user reaches for when they doubt the solver.
- **Cause:** `is_finite_bound(b) = b > NLP_LOWER_BOUND_INF && b < NLP_UPPER_BOUND_INF`
  — a *band* membership test, applied to `lo` and `hi` alike. A real upper bound
  of `-5e20` failed it, so `box_violation`'s `above` term became `-inf` and the
  violation read `0.0`. Mirror case for a real lower bound of `+5e20`. The same
  predicate also sized `row_magnitude`, so a row written at `5e20` scale
  reported a magnitude ignoring its own bound — which then fed the
  scale-relative feasibility comparison.
- The existing coverage pinned only the *sentinel* case, so it passed either
  way. That is why this survived.
- **`pounce check-x0` shared the helper** and used it for bound-activity
  counting and `clamp_to_interior`, so a real out-of-range bound was not clamped
  against and the reported starting point was not the one the solver would
  actually begin from. Note the irony: `check-x0` measuring a fixture at exactly
  zero violation is what established the #396/#398 model was feasible in the
  first place.
- Two further equality/fixed-variable tests in `verify.rs` were gated on
  presence at the same time, the same class #398 and #402 fixed: a row with
  `g_l = g_u = -5e20` is the one-sided `g <= -5e20`, not an equality, and
  skipping it dropped a real complementarity term.
- **The Python surfaces carried exact twins**, now directional too:
  `python/pounce/_preflight.py` (`_box_violation`, `_clamp_to_interior`,
  `at_lo`/`at_hi`), `python/pounce/_starts.py`,
  `pyomo-pounce/pyomo_pounce/preflight.py`, and
  `pyomo-pounce/pyomo_pounce/block_init.py` — where `_seed_var` on a variable
  with `ub = -5e20` and no `lb` fell through to seeding `0.0`, outside the
  variable's own declared box.
- Five regression tests (two Rust, three Python), all verified to fail against
  the old band predicate.
- **This closes the sentinel family opened by #396.** Four issues, one root
  cause: reading a directional convention as a magnitude. Every site now goes
  through the shared `lower_bound_present` / `upper_bound_present` helpers
  introduced in #401.
  - Checked and deliberately left alone: `python/pounce/_route.py` uses plain
    `np.isfinite`, which is *correct* there — its bounds come from
    `_normalize_bounds` (scipy `±inf`) and `_wrap_constraints`, neither of which
    uses the sentinel convention. `python/pounce/gams/link.py` and
    `gams/gams_pounce.c` were already directional by construction.

### Fixed — presolve could still certify infeasibility from an absent-bound sentinel (#402)

- **The last of the presolve machinery behind #396's witness gate now reads the
  sentinel directionally.** #396 fixed the gate that *guards* the infeasibility
  certificate; four sites behind it still asked whether a bound was real by
  magnitude, and one of them could manufacture the certificate in the first
  place. A verdict of "proved infeasible" is the strongest claim POUNCE makes,
  and it should not depend on a downstream gate to be withdrawn.
- **The crossed-box test had no presence check** (`bound_tighten.rs`). These
  arrays hold raw `±INF_BOUND` sentinels, not infinities, so a variable with no
  lower bound (`x_l = -1e19`) and a real upper bound (`x_u = -5e20`) satisfied
  `x_l > x_u` on the *first* pass, before any propagation. That is #398's
  failure mode with `infeasible` as the verdict instead of an error, and
  `crossing_is_certifiable` had the same hole — the gap is `5e20`, far too large
  for `is_negligible` to dismiss, so it certified.
- **`mul_bound` was magnitude-driven and could produce `NaN`.** It had no idea
  whether it was converting a lower or an upper bound: a real upper bound of
  `-5e20` was discarded as an infinity, and a real *lower* bound of `+5e20`
  became `+INFINITY`, which `row_activity` summed into `lo_finite` (it counts
  only `NEG_INFINITY`), after which `others_for` computed `inf - inf`. Presence
  is now decided in `contribution`, where the side is known, which makes the
  `NaN` **structurally unreachable**: `cj_lo` can only be finite or `-∞` and
  `cj_hi` only finite or `+∞`, so `row_activity`'s counters classify every term
  correctly. A zero coefficient now short-circuits to `(0, 0)` rather than
  reaching `0 * ∞`.
  - Worth recording precisely: in `tighten_bounds` the `NaN` was *masked* — the
    ungated crossed-box test fires on the same variable and returns `infeasible`
    before the poisoned bound is used. So the observable defect on that path was
    the false infeasibility, and the `NaN` was latent behind it.
- **The margin probe failed open.** `fbbt_infeasibility_survives_margin` sized
  each row's acceptance slack with a symmetric `|b| < INF_BOUND` test, so a row
  whose real bound is `-5e20` contributed `0.0` and the margin collapsed to
  `tol * 1.0` — a widening of `1e-8` on a row written at `5e20`. Any
  infeasibility survives that, which is exactly what the probe exists to rule
  out. Its own two `relaxed` maps, three lines below, already used the
  directional form; the halves of the function disagreed. The logic is now a
  named `row_margin_for` so it can be tested directly.
- **`trivial_elim` fixed variables from a sentinel.** `big_bound` is a parameter
  of `find_trivial_eliminations` and was already consulted directionally for
  rows eight lines away, but never for variables — so `x_l = x_u = -5e20`
  ("no lower bound, upper bound `-5e20`") was declared FIXED at `-5e20`, and
  `x_l = x_u = -1e19` fixed at the sentinel itself. Not test-only: the result
  feeds `diag.trivially_fixed_vars` and `excluded_vars_buf` via `auxiliary.rs`.
- All four use the shared `lower_bound_present` / `upper_bound_present` helpers
  introduced in #401, so this predicate now has one spelling across the tree.
- Ten regression tests, all verified to fail against the old predicates and pass
  with the fix. The genuine cases are preserved: `x in [5, 3]` still reports
  infeasible, a sub-tolerance crossing still refuses to certify (the #380 rule),
  and a variable with both bounds present and equal is still fixed.

### Fixed — the convex route silently dropped constraints bounded past the opposite sentinel (#401)

- **An LP/QP/QCQP no longer reports `Optimal` at a point its own bounds
  exclude.** This is the same absent-bound-sentinel confusion as #396 and #398,
  on the convex path — where it is worse, because it does not misclassify a
  constraint, it **deletes** it, and the solver then answers a different problem
  than the one that was asked. Unlike its siblings it fails *silently*: #396 was
  a false infeasibility proof and #398 a refusal to solve, both loud.
- **Cause:** two independent copies of a symmetric magnitude test,
  `fn is_finite_bound(v: f64) -> bool { v.abs() < NL_INF }`, in
  `pounce-cli/src/qp_extract.rs` and `pounce-cli/src/dispatch.rs`. Presence is
  directional — a lower bound is absent only at or below `-1e19`, an upper bound
  only at or above `+1e19` — so a real upper bound of `-5e20` is an ordinary
  bound and `|v| < 1e19` is simply the wrong question about it.
- Three distinct wrong answers came out of that:
  - **A real variable bound never entered `G`.** `x_u = -5e20` failed the test,
    so `x_i <= -5e20` was never built and the QP was solved over a strictly
    larger box. Mirror case for a real `x_l = +5e20`.
  - **A row with equal bounds past the sentinel vanished from the problem
    entirely.** `lo == hi && is_finite_bound(lo)` failed, so the row fell into
    the inequality branch — where both `is_finite_bound(hi)` and
    `is_finite_bound(lo)` were false too, leaving `upper` and `lower` both
    `None`. It contributed nothing to `A` and nothing to `G`.
  - **A real quadratic row was discarded as "vacuous".** `g(x) >= 5e20` arrives
    as `g_l = 5e20` (real), `g_u = 1e19` (sentinel); both tests were false, so
    `vacuous` was true and the row was skipped with "Free row: imposes
    nothing". The model then classified as a convex QCQP and went to the conic
    solver *as if the constraint were not there* — it is reverse-convex, and
    the honest answer is NLP.
- **Fix:** one shared directional pair, `lower_bound_present` /
  `upper_bound_present`, added next to `NLP_LOWER_BOUND_INF` /
  `NLP_UPPER_BOUND_INF` in `pounce-common::types` — so there is now one spelling
  of this predicate in the tree to fix, rather than the several that #396, #398
  and this issue each turned up independently. The equality test is additionally
  gated on *both* bounds being present, as #398 did in `classify_bounds`.
- `recover_bound_mults` walks the G-row layout with the same predicate and was
  *consistent* with the buggy builder, so it moved in lockstep; a regression
  test now pins the two together on a box only the directional reading admits.
- **Worth knowing: an equality row outside `±1e19` cannot be expressed under
  this convention at all.** An equality needs both bounds present, and the two
  presence tests only overlap inside the band — so `g_l = g_u = -5e20` is not an
  equality at `-5e20`, it is the one-sided `g <= -5e20` (no lower bound). The
  original issue report described this case as a lost *equality*; the row was
  indeed lost, but what it should become is a one-sided inequality.
- Covered by four regression tests, all verified to fail against the old
  symmetric predicate and pass with the fix:
  `variable_bound_past_the_opposite_sentinel_is_kept`,
  `a_row_with_equal_bounds_past_the_sentinel_does_not_vanish`,
  `bound_multipliers_stay_aligned_with_the_built_rows` (all in
  `qp_extract.rs`), and `a_quadratic_row_bounded_past_the_sentinel_is_not_vacuous`
  (`dispatch.rs`), plus `presence_is_directional_not_symmetric` on the new
  shared helpers.

### Fixed — a one-sided bound beyond `±1e19` was read as a crossed bound pair (#398)

- **A feasible model no longer comes back `Invalid_Problem_Definition`.**
  `TNLPAdapter` rejected any model carrying a legitimate one-sided bound whose
  value lies past the *opposite* absent-bound sentinel, reporting AMPL
  `solve_result_num` **504** / Pyomo `internalSolverError` before the first
  iteration — so there was no solver output to diagnose it from. The model in
  question (property-test seed 223, the same one #396 stopped presolve from
  certifying infeasible) has a starting point `pounce check-x0` measures at
  *exactly* zero violation, and the convex QP route solves it to optimality.
  It now solves on the NLP route too.
- **Cause: a symmetric reading of a directional convention.** A `<=`-only row
  reaches the adapter with its absent lower bound filled in by the `-1e19`
  sentinel. On row `-1e30·x[0] <= -5.0000000000000007e20` that gives
  `g_l = -1e19`, `g_u = -5e20`, and the adapter's `lo > hi` test fired.
  Nothing is inconsistent — `-5e20` is an ordinary finite upper bound, and the
  sentinel standing in for the absent lower bound is not a bound to compare it
  against at all. The bound is only "beyond infinity" if you read the sentinel
  as a magnitude.
- **Fix:** `classify_bounds` now derives presence directionally, the same
  convention `pounce_presolve::bound_tighten` and the #396 witness gate use — a
  lower bound is absent iff `lo <= nlp_lower_bound_inf`, an upper bound iff
  `hi >= nlp_upper_bound_inf` — and classifies equality / one-sided /
  two-sided / free from the *present* bounds only. This closes an asymmetry
  that had been live for some time: a *variable* bound past the sentinel was
  already handled directionally by `bound_tighten`, while a *constraint* bound
  past it was a hard error.
- `INCONSISTENT_BOUNDS` is now reserved for the case it was meant for — both
  bounds present and genuinely crossed, a modelling error. `x in [5, 3]` still
  returns 504 on both the `presolve=yes` and `presolve=no` routes
  (`user_declared_crossed_box_is_still_rejected`).
- The equality and fixed-variable tests are gated on presence for the same
  reason: `g_l == g_u == 1e20` describes `g >= 1e20`, a one-sided row, not an
  equality at `1e20`, and `x_l = 5e20` with no upper bound is a
  lower-bounded variable, not one fixed at `5e20`.
- **Scope.** Any model with a one-sided bound outside `±1e19` — unusual but not
  pathological. `nlp_lower_bound_inf` / `nlp_upper_bound_inf` are user-settable
  options, so a well-scaled model trips this the moment someone lowers the
  threshold, and an `.nl` may legitimately carry a bound of `1e20`.
- **One sibling of the same predicate fixed alongside it.**
  `starting_point_refutes_infeasibility`'s crossed-box guard
  (`pounce-algorithm/src/infeasibility_refutation.rs`) tested the raw pair too,
  and was the only bound test in that file not already directional — so a
  variable declared `x <= -5e20` with no lower bound made it decline to refute,
  withholding a witness the model plainly admits. Failing that way is safe (the
  gate can only *withdraw* an infeasibility verdict, never issue one), which is
  why it had gone unnoticed; it is now consistent with the two clamps
  immediately below it.
- **A deliberate divergence from upstream**, noted in the source so it is not
  "corrected" back by a future porter. `IpTNLPAdapter` runs the same raw
  `lower == upper` / `lower > upper` tests before consulting the sentinels, and
  inherits the same rejection. Models upstream accepts classify identically
  here; the divergence is confined to bounds outside the sentinels, which
  upstream cannot express at all.
- Covered by `feasible_sentinel_bound_model_solves` in
  `crates/pounce-cli/tests/presolve_certified_infeasibility.rs` (asserts
  `Solve_Succeeded`, on both the presolve and no-presolve routes) and four unit
  tests on `classify_bounds` in `crates/pounce-nlp/src/tnlp_adapter.rs`
  covering the row, the variable box, the still-rejected crossed pair, and the
  sentinel-valued equality. The existing property sweep does not see this class
  of defect: `test_infeasibility_no_false_positives.py` watches the `200..299`
  infeasible band, and `504` is outside it — a feasible model answered with
  `Invalid_Problem_Definition` is a wrong answer of a different shape.

### Fixed — equality rows had no scale-relative runtime feasibility measure (#390)

- **A nonlinear equality contradiction is no longer accepted as solved once its
  rows are written small enough.** `x*y == 1` with `x + y == 0.5` has no real
  solution (the roots would need discriminant `0.25 - 4 < 0`), and refuting it
  needs the discriminant, not intervals — so neither the DOF gate's linear
  bound propagation (#389) nor FBBT can reach it, and the verdict comes from
  the runtime feasibility test. Multiplying both rows by `1e-6` used to flip
  that verdict from `Infeasible_Problem_Detected` to `Solve_Succeeded`. Same
  empty solution set, opposite answers.
- **Cause:** POUNCE folds an equality's right-hand side into `c(x) = 0`, so
  `|c_i|` *is* the violation and carries no independent magnitude. The
  scale-relative measure added for inequality rows in #386
  (`curr_relative_primal_infeasibility_max`) therefore skipped the `c` block
  entirely, and every runtime feasibility decision on an equality row was
  against an absolute tolerance — which down-scaling walks straight under.
- **Fix:** the pre-fold RHS is plumbed back out as
  `IpoptNlp::declared_c_rhs`, the same "declared, not live" pattern
  `declared_d_bounds` established for inequality rows, and reported in the
  same internally-scaled space as `eval_c` so the solver's own row scaling
  cancels. The measure now covers the `c` block with `|c_i| / |b_i|`, which is
  identical at every writing of the row. Both consumers pick it up unchanged:
  the certificate/acceptable-point veto and the rapid-infeasibility
  pre-filter.
- **A homogeneous row (`b_i = 0`) keeps the absolute test and contributes
  nothing** — it has no declared magnitude, and needs none, since `s·g(x) == 0`
  is the same row at every `s`. Dividing by a fabricated floor there would turn
  float-noise residuals into 100% "violations" on the most common equality row
  there is. An NLP that does not track the RHS (the trait default — e.g. the
  restoration NLP, whose `c` block is not the user's rows) abstains wholesale
  rather than invent a magnitude.
- Covered by a new `inf_eq_nl` model in
  `pyomo-pounce/tests/test_scale_invariance.py`: 5 wrong cells out of 13 row
  scalings before (`SOLVED` at `k in {-10, -8, -6}`, no verdict at
  `k in {-12, -4}`), 0 after. No other model in the harness moved, and a
  differential sweep over the repository's `.nl` fixtures shows no verdict
  changes. `crates/pounce-algorithm/tests/issue_390_nonlinear_equality_scale.rs`
  pins the same property on the direct driver, in both directions.
- Trade-off worth knowing: refusing a certificate keeps the run going, so on a
  *nonconvex* model the continued trajectory can land in a different basin than
  the premature stop did. The measured instance — the same product/sum pair
  started at `(1, 1)`, where the feasible twin already converges to a
  locally-infeasible point at unit scale — traded a `Solve_Succeeded` at row
  scale `1e-8` for a local-infeasibility verdict. What was given up there was
  not, in general, a solution: at `1e-12` the same "success" returned
  `x = y = 2.3e-14`, which satisfies neither `x*y == 1` nor `x + y == 2.5`.
- This closes out #387; #389 and #391 handled the linear half at the DOF gate.

### Fixed — the last 3 `inf_eq` cells: the DOF-gate proof survives sub-tolerance row scales (#391)

- **A contradictory over-determined equality system is now reported infeasible
  at every row scale, including the sub-`1e-8` ones.** #387 took
  `test_scale_invariance.py`'s `inf_eq` from 13 wrong cells to 3; the residual
  three (`k ∈ {-12, -10, -8}`) still reported the 504 structural error. The
  `inf_eq` baseline goes **3 → 0** and the model is now scale-invariant across
  the harness's full 25 decades.
- **Root cause: an absolute floor in the witness gate, not in the proof.** The
  bound-propagation proof is already scale-free — `s*x == 0.2*s` with
  `s*x == 0.8*s` crosses by `0.6` at every `s`. What moved was the refutation:
  the witness test accepts a row when the residual is negligible through the
  *clamped* form `tol * max(scale, 1)`, and that clamp reinstates an absolute
  floor once a row's magnitude drops below 1. At `s <= ~3e-8` every point of
  `[0, 1]` therefore "satisfied" both rows, refuted the verdict, and the proof
  was withdrawn.
- **The fix is scoped to the one path where the solve cannot run.** The clamp
  exists to honor #380 — never certify what the solver itself would accept as
  feasible, or the same model reports "proved infeasible" with presolve on and
  `Solve_Succeeded` with it off. On the DOF-gate path that counterfactual never
  materializes: the gate fired *because* the solve cannot run (1 variable, 2
  equality rows), so the alternative to the proof is the structural error, not a
  solution. Only there does the witness switch to
  `pounce_presolve::WitnessRule::DeclaredRowRelative`, which measures the
  residual against the row's **declared** magnitude — `max(|g_l|, |g_u|)` over
  its finite bounds, the same "declared, not live" pattern
  `fbbt_infeasibility_survives_margin` already uses — with no clamp. Every
  wrapper that is actually solved through keeps the clamped form; the rule is a
  property of the call site (`PresolveTnlp::probing_without_a_solve`), not a
  user-settable option.
- **The `b = 0` hazard is handled explicitly and fails closed.** A homogeneous
  row (`g_l = g_u = 0`, or one with no finite bound at all) has no declared
  magnitude, which would make a pure relative test unsatisfiable by construction
  — the violation *is* the scale, so float noise on a genuinely feasible point
  would read as a full-magnitude violation and fail to refute. Such rows keep
  the absolute floor.
- Unchanged, and covered by tests: a *consistent* over-determined system still
  reports the DOF error at every scale, and at extreme row magnitude
  (declared `~1e30`) the witness still refutes on a residual of `1e5` — fifteen
  relative digits — which is the over-approximation class the gate exists for.
- The algorithm-level `sub_tolerance_scales_keep_the_dof_error` test is
  deliberately flipped to `sub_tolerance_scales_are_certified_too`.

### Fixed — provably infeasible over-determined systems reported a DOF failure (#387)

- **A contradictory over-determined equality system now reports
  `Infeasible_Problem_Detected` (AMPL 200 band) instead of
  `NotEnoughDegreesOfFreedom` (504).** `x == 0.2` with `x == 0.8` over
  `x in [0, 1]` has an empty feasible set — about as provable as infeasibility
  gets — yet it exited through the too-few-degrees-of-freedom gate (1 variable,
  2 equality rows) before any iteration ran, so nothing downstream ever got the
  chance to look at the constraints. On the scale-invariance harness the model
  was wrong at all 13 row scalings; it is now wrong at 3 (see below).
  - The DOF gate now consults presolve's bound-propagation certification on its
    failure path before reporting the structural error. The probe runs only when
    the gate fires, is independent of the `presolve` master switch (nothing is
    transformed — the wrapper is dropped without solving through it), and
    inherits the certification's fail-closed safety net wholesale: the crossing
    must exceed the solver's own acceptance margin at the crossed pair's scale,
    and a concrete witness point satisfying every constraint withdraws the
    verdict.
  - Consequences of fail-closed: a *consistent* over-determined system still
    reports the DOF error, and at row scales at or below ~`1e-8` — where every
    point in the box satisfies both rows within the solver's own acceptance
    tolerance — the proof is withheld and the DOF error stands. Those were the
    3 remaining wrong cells in `test_scale_invariance.py`'s `inf_eq` baseline
    (13 → 3); #391 above takes them to 0.
  - Also fixed on the way: the CLI's `CountingTnlp` / `SeededTnlp` wrappers did
    not forward `get_variables_linearity` / `get_objective_variables_linearity`
    / `get_constraints_linearity` (trait default `false`), so anything stacked
    above them — including this probe — saw every row as nonlinear and silently
    lost linear-row bound propagation.
  - What remained of #387 — equality rows having no scale-relative *runtime*
    feasibility measure, so a genuinely *nonlinear* equality contradiction (out
    of bound propagation's reach) still depended on absolute tolerances — was
    split out as #390 and is fixed above.

### Added — presolve can now certify infeasibility instead of discarding the proof

- **When presolve proves the feasible region empty, that proof is now the
  verdict.** With `presolve=yes`, a model whose emptiness bound propagation or
  FBBT can establish is reported immediately as
  `InfeasibleProblemDetected (detected by presolve: <method>)` with AMPL
  `solve_result_num` **201**, and the solve is skipped entirely.
  - Previously presolve detected these cases, logged a warning, threw the result
    away, and let the IPM re-derive a strictly *weaker* numerical verdict — a
    stationary point of the constraint violation, which on a nonconvex problem
    does not rule out a feasible point elsewhere. The code said why: *"Presolve
    has no channel to certify infeasibility."* This adds that channel
    (`TNLP::presolve_infeasibility_proof`).
  - The two verdicts are now distinguishable. `200` remains the numerical
    "converged to a point of local infeasibility"; `201` means *proved*. Both sit
    in AMPL's `200..299` infeasible band, so band-reading consumers are
    unaffected — Pyomo maps the whole range to
    `TerminationCondition.infeasible` in both of its SOL readers. Sub-coding
    within a band is the AMPL-native idiom (Ipopt does the same with 500/501/502
    in the failure band).
  - Both proof methods are sound. Propagating a linear row over a box is a
    decision procedure, and the crossing must exceed a `1e-12` margin before it
    counts; FBBT's interval arithmetic is outward-rounded (one ULP per side), so
    an empty computed interval means the true range is empty.
  - **Soundness guard:** a contradiction found while a Phase-0 auxiliary
    elimination is in force can be an artifact of that elimination — presolve
    breaking a *feasible* model. Those are re-checked on the rolled-back box and
    only certified if they survive. Pinned by a test on exactly that scenario.
  - A certified verdict also skips the MC64 second-opinion re-solve, which exists
    to second-guess a numerical local infeasibility and cannot overturn a proof.
  - Unchanged by default: presolve is off unless requested, so the default path
    still reports `200` via the numerical route.
  - Docs: [Solution output](docs/src/solution-output.md) gains a
    `solve_result_num` band table and the proved-vs-local distinction.

### Fixed — `qp-active-set` reported an unbounded model as a solver internal error (#388)

- **An unbounded model solved with `solver_selection=qp-active-set` now reports
  `Diverging_Iterates` (AMPL `solve_result_num` **300**, "your model is
  unbounded"), matching every other selector on the identical `.nl` file.** It
  previously reported `Internal_Error` (**500**, "the solver broke") — so a
  modeler with a genuinely unbounded model was told POUNCE had a bug, and
  drivers that branch on `solve_result_num` routed the result to the wrong
  handler. Minimal reproduction: `min −x s.t. x ≥ 0`, one variable, one
  constraint.
  - The unboundedness was already *detected* — the inner QP said so in as many
    words — but the SQP driver folded that correct diagnosis into
    `QpFailure(LinearSolverFailure("QP subproblem returned status unbounded"))`,
    which the application layer maps to `Internal_Error`. It was also mislabeled
    a linear-solver failure with no linear solver having failed. A
    status-mapping defect, not a numerical one.
  - The verdict is **not** blanket-mapped, because an unbounded *step* QP is on
    its own only a statement about the linearization: on a nonconvex NLP the
    constraints can curve back and the objective turn around (`min −x s.t.
    x² ≤ 1` has an unbounded step QP at `x = 0` and a bounded feasible set).
    `pounce-qp` now returns the certified recession ray alongside the verdict
    (`QpSolution::unbounded_ray`), and the SQP driver re-tests that ray against
    the **true** `f` and `c` — feasible probe points spanning twelve decades of
    step length, each required to sustain at least half the initial linear
    descent rate, the same non-decelerating-descent bar the IPM's divergence
    guard applies (#248/#252/#285). Only a ray that survives is reported
    unbounded.
  - A ray that does not survive now yields the honest non-committal
    `Search_Direction_Becomes_Too_Small` (`QpStepFailed`) instead of the old
    hard error — the QP could not produce a usable step, and no claim is made
    either way.
  - The unbounded verdict also travels the normal result path, so it reaches
    `finalize_solution`: the JSON report carries an objective and an `x` block
    where it previously emitted `"objective": null` and no `x`.
  - Unaffected: the infeasible and feasible cases on the same selector, which
    already answered `200` and `0` correctly.

### Fixed — presolve *certified* a feasible model infeasible, from an absent-bound sentinel (#396)

- **A model with a feasible point is no longer reported `solve_result_num` 201.**
  `201` claims a *proof*, which is the strongest thing POUNCE says; this one was
  false. The model (property-test seed 223) is feasible by construction, and
  `pounce check-x0` measures its own declared starting point at **exactly** zero
  violation — `rows violated: 0, max violation: 0.000e0` — while the convex QP
  route solves it to optimality.
  - Root cause is the absent-bound sentinel, mishandled in **both directions on
    the same row**. POUNCE stores an absent constraint bound as `±1e19`
    (`INF_BOUND`), not as an infinity, so every use has to ask whether a bound is
    real. `witness_refutes_infeasibility` asked twice, differently, and got both
    wrong for row `-1e30·x[0] <= -5.0000000000000007e20`:
    - The violation term was `(g_l - v).max(v - g_u)` with **no** presence test,
      so it scored `4.9e20` against a lower bound the row does not have — at a
      point whose true violation is zero.
    - The magnitude term used a **symmetric** test, `|b| < INF_BOUND`, which
      called the row's genuine upper bound `-5e20` absent and discarded a real
      constraint.
  - Either error alone sinks the witness, so no candidate point could ever
    refute the verdict, and the certification escaped.
  - Fix: one **directional** predicate, matching the convention
    `pounce_presolve::bound_tighten` already uses (`x_l <= -INF_BOUND`,
    `x_u >= INF_BOUND`) — a lower bound is absent below `-INF_BOUND`, an upper
    bound above `+INF_BOUND` — derived once and used for both the magnitude and
    the violation, so the two cannot drift apart again.
  - Strictly one-directional, like the gate it sits in: it can only ever
    *withdraw* a verdict. Every certified-infeasibility test still passes,
    including the must-still-detect cases.
  - **The property sweep is now at zero.** Over 400 feasible-by-construction
    instances, false positives in the AMPL infeasible band go `1 → 0` on the full
    presolve option set (and stay at 0 on the numerical path from #379).
    `MAX_FALSE_POSITIVES` ratchets `1 → 0` — a floor, not a target.
  - **Was still open on this model:** it did not solve. It returned
    `Invalid_Problem_Definition` (504), because `TNLPAdapter` read the same
    out-of-range bound as a crossed constraint pair — `g_l = -1e19` against
    `g_u = -5e20` tripped the `lo > hi` check. That is a separate defect in the
    sentinel convention for constraint bounds, affecting any model with a
    one-sided bound beyond `±1e19`; it was tracked and fixed as #398 (above).
    The regression test here asserts the *band*, not a specific code, for that
    reason, and did not have to be rewritten.

### Fixed — the NLP path reported `Infeasible_Problem_Detected` on models whose own starting point is feasible (#379)

- **No numerical path may now claim infeasibility while holding a point that
  satisfies every constraint.** The feasible-by-construction property sweep in
  `pyomo-pounce/tests/test_infeasibility_no_false_positives.py` found a model
  (seed 294) that POUNCE was *handed a feasible starting point for* — `(5e5, 5e5)`,
  satisfying both rows exactly — and reported as infeasible: AMPL
  `solve_result_num` 200, Pyomo `TerminationCondition.infeasible`. POUNCE's own
  convex QP route solves the same model to optimality, and `pounce verify`
  reports `0.000e0` constraint and bound violation on that solution.
  - Root cause is not a threshold. The gates that produce this verdict — the
    restoration gates, the outer cycle detector, the SQP infeasible-subproblem
    exit, the ℓ₁ wrapper's uncollapsed-slack certificate — all argue from a
    *stalled feasibility sub-problem*: the violation is bounded away from zero
    and no local move reduces it. That is evidence, not proof. On this model the
    first row carries `±1e30` coefficients; in the scaled space its slack
    initializes far from anything `x` can follow, the outer line search walks the
    violation *up* from an exactly feasible start, restoration burns its
    3000-iteration budget, and the cycle gate concluded infeasibility from the
    exhaustion.
  - Fix: a refutation. `pounce_algorithm::infeasibility_refutation` evaluates the
    model's own starting point, clamped into the variable box, and withdraws the
    verdict if it satisfies every row. One concrete feasible point settles the
    question outright, whatever a local argument concluded.
  - Applied as **one gate over all four claim sites**, not per-site. The two
    preceding safeguards in this area were each added to one path and not its
    twin, and a hole survived both times.
  - One-directional by construction: it can only ever *withdraw* a verdict, never
    create one. A model with no feasible point cannot produce a witness, so every
    correct verdict is untouched — and a failure to evaluate simply declines to
    refute.
  - The withdrawn verdict becomes `Error_In_Step_Computation` (AMPL 500), the
    status this codebase already uses for "the solve broke down and we are *not*
    claiming infeasibility". The solve still fails on the `±1e30` model — that row
    is genuinely hostile — but it fails visibly instead of returning a confident
    wrong answer.
  - The witness test is **pure relative** (`tol · scale`), not the clamped
    accepting form. Measured: the clamped form withdrew *correct* infeasibility
    verdicts on three models at row scalings `1e-12 … 1e-8`, because the clamp
    reinstates an absolute floor exactly in the down-scaled direction. Caught by
    `pyomo-pounce/tests/test_scale_invariance.py` before it could ship.
  - Presolve *certificates* are exempt: a proof is not a numerical inference, and
    it carries its own, tighter refutation.
  - Measured: false positives on the numerical path (`solver_selection=nlp
    presolve=no`) over 400 feasible-by-construction instances go **1 → 0**; over
    the property test's full option set, **3 → 1** (the remaining one is seed 223
    on the presolve path, unrelated to this fix). `MAX_FALSE_POSITIVES` ratchets
    from 4 to 1.
  - Not a bug: the CLI's MC64 second-opinion re-solve did not overturn these. That
    guard exists for *hypersensitivity* — two backward-stable scalings falling
    into different basins — and this failure reproduces under both scalings, so
    the second opinion correctly agreed. The refutation runs before it, so the
    wasted re-solve is now skipped too.
  - Regression: `crates/pounce-cli/tests/false_local_infeasibility.rs` gains both
    reported shapes and pins that the convex route still solves them, so the two
    routes cannot drift back into disagreeing about whether the feasible set is
    empty.

### Fixed — an infeasible model's verdict depended on the user's `tol` (follow-up to #372)

- **A genuinely infeasible model now reports `Infeasible_Problem_Detected` (AMPL
  200, Pyomo `TerminationCondition.infeasible`) at every tolerance.** Sweeping
  `tol` across the other infeasible fixtures after #372 found a second,
  independent instance of the same user-visible defect — pre-existing, and
  triggered from the opposite end. `infeasible_equalities.nl` reported
  `Error_In_Step_Computation` (AMPL 500, Pyomo `internalSolverError`) for
  `tol >= 3e-7` and `Infeasible_Problem_Detected` for `tol <= 1e-7`, a
  non-monotonic split on a model whose true constraint violation is `2.0`.
  - Root cause: a **units mismatch**. The constraint violation was measured
    *scaled* (`eval_c` returns `dc ⊙ c_user`; `curr_constraint_violation`
    likewise) but compared against *absolute* floors —
    `max(100·outer_tol, 1e-4)`. Those floors are user-facing magnitudes meaning
    "the violation is meaningfully nonzero", so the comparison mixed two unit
    systems. NLP scaling shrinks this fixture's rows by ~3e6, so a violation of
    `2.0` read as `6.67e-7` and could never clear a `1e-4` floor.
  - Two sites had to change, because the fixture is a *square* 2×2 system:
    `resto_inner_solver::eval_orig_inf_pr_at_inner_curr`, which feeds every
    restoration locally-infeasible gate; and `ipopt_alg`'s outer `cycle_exit`.
    Square problems are deliberately carved out of the `strict` restoration
    gate so the outer loop gets another attempt, which makes that cycle exit
    their *only* locally-infeasible safety net — and the unit mismatch had
    silently disabled it.
  - `ipopt_cq::unscaled_block_amax` is now public and shared by both call sites
    rather than duplicated, so the two cannot drift.
  - The `cycle_exit` test also moves from a 1-norm to a max-norm. Max-norm ≤
    1-norm, so it is marginally *stricter* about declaring infeasibility on an
    unscaled problem — the safe direction for a verdict this consequential.
  - Regression: `crates/pounce-cli/tests/infeasible_status_tol_invariance.rs`
    sweeps four infeasible fixtures over `tol` from `1e-3` to `1e-12`
    (including the `3e-7` value that failed while its neighbour `1e-7` passed)
    and pins the tolerance-invariance property rather than specific tolerances.
  - Verified no false-positive risk: across 477 real `.nl` problems the count of
    `Infeasible_Problem_Detected` verdicts is unchanged (1 before, 1 after) and
    no problem changed verdict.

### Fixed — a trivially infeasible NLP was reported as `Restoration_Failed` at tight `tol` (#372)

- **A visibly contradictory model now reports `Infeasible_Problem_Detected`
  (AMPL `solve_result_num` 200, Pyomo `TerminationCondition.infeasible`) at any
  user `tol`, not just the default.** The reporter's model is a one-variable
  contradiction — `min x³ + x²  s.t.  x ≥ 0.7`, `0 ≤ x ≤ 0.6` — which Ipopt
  3.14.16 diagnoses as `Converged to a locally infeasible point`. POUNCE
  returned `Restoration_Failed`, which lands in the AMPL 500 *failure* range
  and surfaces through Pyomo as `SolverStatus.error` /
  `TerminationCondition.internalSolverError`, leaving client code unable to
  distinguish a mathematically infeasible model from a solver bug.
  - Root cause: the trigger was the reporter's `tol=1e-10`, not the model — the
    identical model at the default `tol=1e-8` was already diagnosed correctly.
    The locally-infeasible gates in `resto_inner_solver.rs` all key off *how*
    the restoration inner sub-IPM terminated. At a tight `tol` the inner
    reaches the same stationary point of the feasibility problem
    (`inner_kkt_err ≈ 1.2e-10`, `orig_inf_pr = 1.0e-1`) but can no longer
    certify it against its own, equally tightened, convergence test — every
    remaining step is below the tiny-step threshold, so it exits
    `StopAtTinyStep` rather than `Success`. No gate admitted that status, so a
    correct infeasibility diagnosis was discarded and the solve fell through to
    `Restoration_Failed`.
  - Fix: a `tiny_step_locally_infeasible` gate. A step shrunk to machine
    precision *is* the numerical stationarity evidence, so unlike the `strict`
    gate it does not additionally demand `inner_kkt_err ≤ 10·outer_tol` — the
    threshold the inner cannot reach at tight `tol`, which is exactly how the
    case arises. It keeps the `alt` gate's looser `1e-2` KKT ceiling to reject
    inners that stalled without ever approaching stationarity, and mirrors
    `strict`'s `!is_square_problem` guard and `max(100·outer_tol, 1e-4)`
    violation floor, so a merely near-feasible kappa-guard exit still cannot
    manufacture a false infeasibility verdict.
  - The AMPL status mapping is unchanged: `Restoration_Failed` stays in the 500
    range, matching Ipopt's own `AmplTNLP` mapping. The defect was the
    classification, not the mapping.
  - Regression: `crates/pounce-cli/tests/issue_372_infeasible_bounds_status.rs`
    pins the reporter's exact option set, sweeps `tol` from `1e-6` to `1e-12`,
    and asserts the result is never advertised in the AMPL *solved* family.

### Added

- **Generic presolve for library TNLP solves.** `presolve=yes` now wraps every
  `IpoptApplication::optimize_tnlp` call once, before algorithm dispatch, so
  IPM, SQP, and retry paths all use the same reduced TNLP while
  `finalize_solution` continues to receive original-space values and
  multipliers. Library code no longer needs to call `wrap_with_presolve` for
  the ordinary callback-TNLP case.
  - `IpoptApplication::set_presolve_already_applied` declares an explicit
    caller-owned presolve wrapper; `optimize_tnlp_without_presolve` is the
    scoped bypass for consumers, such as sensitivity analysis, that require
    the original KKT coordinate system; and `TNLP::is_presolve_wrapper` lets
    generic TNLP decorators preserve the no-double-wrap marker.

### Changed

- **`pounce-nlp` warm starts now fetch one coherent TNLP starting-point
  snapshot.** With `warm_start_init_point=yes`, the adapter calls
  `TNLP::get_starting_point` once with the primal, bound-multiplier, and
  constraint-multiplier flags enabled, then projects that single payload into
  the IPM's `x`, `y`, and `z` blocks. This replaces three independent callback
  invocations, matching Ipopt's `GetStartingPoint` contract and avoiding
  inconsistent or side-effect-dependent warm starts in existing library,
  Python, and C-bridge users. A TNLP that refuses that requested warm-start
  payload now fails explicitly with `Invalid_Problem_Definition` rather than
  silently using default iterates.

### Tests — pyomo-pounce: the shadowing-binary test failed on any source checkout (#366)

- `test_check_binary_flags_a_shadowing_build` asserted that a different-build
  `pounce` prepended to `PATH` is reported as shadowing. That only holds when
  the resolved binary is PATH-independent, i.e. when a wheel-bundled binary
  exists. Without one, `_default_executable` falls back to `shutil.which`, so
  the fake *became* the resolved binary and shadowed nothing — the assertion
  inverted and the test failed on every source checkout with no
  `pounce-solver` wheel installed. Its sibling tests guard with
  `if _bundled_path() is None: skip`; this one only guarded on
  `resolved is None`.
- Fixed by standing the real binary in as the bundled one when none is
  installed, so the scenario is still exercised locally rather than skipped.
  CI, which stages the built CLI into the wheel, is unaffected either way.
- `pyomo-pounce/README.md` gains a **Running the tests locally** section. The
  staging step that makes a local run match CI (`cp target/release/pounce
  python/pounce/bin/pounce` *before* building the wheel) was discoverable only
  by reading `ci.yml`, so a source checkout silently exercised the PATH
  fallback rather than the bundled path — the root of the confusion behind
  this issue. It also points at `check_binary()` as the first thing to check
  when a dual/multiplier test fails, since a stale binary reports a plausible
  version string.
- Context for the wider report behind #366: the `pyomo-pounce` suite is green
  on `main` (82 passed, 2 skipped) when run against a binary built from the
  same commit. The other four failures reported there did not reproduce, and
  the two multiplier tests among them (`test_bound_multipliers_populate_ipopt_zL_zU`,
  `test_multiplier_gradient_matches_finite_difference`) guard #296 and
  #271/#272 — both of which landed *in* 0.9.0. Since builds from before and
  after the dual-sign fix both report `0.9.0`, a locally built binary from a
  slightly stale checkout fails exactly those two while looking current, which
  is the scenario `check_binary()` exists to detect.

### Performance — pyomo-pounce: `sens.py` resolved every row by linear scan, making `gradient(target=None).to_dataframe()` quadratic (#365)

- **Name-to-row lookups are now dict lookups instead of `list.index` scans.**
  Every query in `pyomo_pounce.sens` resolved a component name to its row by
  scanning `var_names` or `con_names`, which is O(n) per lookup. The full
  Jacobian was the worst case: `gradient(wrt=p)` with no target fans out over
  *every* variable, and `to_dataframe()` then re-scanned the name list once per
  cell — **O(n²·p)** string comparisons. At n = 2,000 that is unnoticeable; at
  n = 50,000, a size an ordinary DAE discretization reaches, it is ~2.5e9
  comparisons and the call effectively hangs.
- The per-solve paths are fixed too: the fitted-variable and residual loops in
  `sens_solve` were O(k·n) for k residuals, paid on *every* solve — which the
  repeated-solve NMPC workflow the module is built around pays each cycle.
- `_Session` now builds `{name: row}` maps once (`_row_index`), and
  `sens_solve` hands over the maps it already built rather than having them
  rebuilt. No API or behavior change: `var_entry`/`mult_entry` still raise
  `ValueError` for an unknown name rather than leaking the dict's `KeyError`,
  and the `con_alias` translation and inequality-multiplier errors are
  untouched.

### Fixed — pyomo-pounce: a declared Param in a `Var` bound reported exactly zero sensitivity (#356)

- **A limit written as a variable bound now moves with the Param that sets
  it.** `pyomo.contrib.sensitivity_toolbox`, which supplies the expression
  surgery behind `declare_sens_param`, substitutes declared Params in
  *constraint* expressions only. A Param left in a `Var` bound was written to
  the `.nl` file as a constant at its pre-perturbation value, so the bound
  never moved and `gradient(m.x, wrt=m.p)` read exactly `0.0` — a wrong answer
  indistinguishable from a legitimate insensitivity. `sens_solve` now rewrites
  such a bound as a constraint over the substituted variable, so the two
  spellings of the same limit agree. Expression bounds such as
  `bounds=(0, 2*p + 1)` are handled, not just a bare Param.
  - On a minimum-time racing problem with an acceleration cap, discretized at
    `nfe=100`, `d tf/d u_up` goes from `0.000000` to `-6.323665` — matching the
    constraint spelling of the same cap to every digit, and cutting the
    estimate error at a 10% perturbation from `5.81e-01` to `5.17e-02`.
- Fixed Vars and Vars on deactivated Blocks are deliberately skipped. A fixed
  Var's bounds are never enforced by the solver (Pyomo substitutes it out as a
  constant), so rewriting one would impose a restriction on the pinned Param
  that the original model never had; a deactivated Block's Var would be pulled
  into the `.nl` file as a free column restricted only by the new row.
- Two consequences are deliberate and worth knowing. The rewrite **drops the
  bound** on the clone that is solved: `m.x.ub` reads `None` there and the NL
  row carries the reader's no-bound sentinel `1e19` (finite, so an `isinf()`
  test would not catch it). The original model is untouched. And `estimate()`
  no longer clamps or warns for those variables — correct, because the bound
  now moves with the perturbation, so the linear step already respects it to
  first order. `covariance()`'s bound-active projection is unaffected: the
  pre-rewrite bound value is recorded at solve time and read back for the
  activity test.
- Cost: a simple bound is handled directly in the barrier, whereas a general
  inequality costs a slack and a Jacobian row, so a model with many
  Param-dependent bounds trades roughly one row per bound. Only models that
  write a bound in terms of a declared Param pay this.
- This makes pyomo-pounce answer **differently from `sensitivity_calculation`**
  on the same model, deliberately: writing a limit as a bound is the natural
  spelling, and requiring a manual rewrite to a constraint is precisely the
  usability problem `declare_sens_param` exists to avoid. Recorded in the
  `pyomo_pounce.sens` module docstring so it is discoverable without being
  opt-in-able.

### Fixed — `qp-active-set` facade returned `success=False` and a wrong `x` on convex QPs with an active inequality (#358)

- **The default `pounce.minimize(..., solver_selection="qp-active-set")` path
  now converges** on easy, well-conditioned convex QPs whose general inequality
  is active at the optimum. When the caller supplies no analytic Hessian the
  facade sets `hessian_approximation=limited-memory`, which on the
  active-set-SQP path selected the L-BFGS Lagrangian Hessian — it stalled with
  `Search_Direction_Becomes_Too_Small` (or reported the QP subproblem
  `unbounded`) and returned `success=False` together with a silently wrong `x`,
  even at `cond(P)=10`, `n=3`. On a 36-instance sweep 29/36 failed.
  - Fix: on the SQP path the automatic quasi-Newton approximation now maps to
    the dense Powell-damped BFGS (`crates/pounce-algorithm/src/application.rs`),
    which is far more robust and — because L-BFGS materializes the same dense
    `n×n` Hessian for the QP subproblem today — costs nothing extra. An explicit
    `sqp_hessian="lbfgs"` opt-in is still honored unchanged. The `#348`
    exact-without-Hessian downgrade now targets damped-BFGS for the same reason.
  - Independent oracles (closed-form KKT, `pounce.solve_qp` IPM, scipy SLSQP)
    and every other pounce path already solved these; only the facade's L-BFGS
    SQP default was affected.

### Fixed — active-set-SQP quasi-Newton overshoot on ill-conditioned QPs (#358 tail)

- **The damped-BFGS active-set-SQP now sizes its initial Hessian**, fixing the
  ill-conditioned tail of #358 (`cond(P) ≳ 1e3`). The identity seed `B₀ = I` is
  a catastrophic scale when `‖∇²L‖ ≫ 1`: the first QP step overshoots the Newton
  step by `~cond(∇²L)`, and the filter line search — with an empty filter at a
  near-feasible start (`θ_curr` tiny) — accepts the objective-blowing step
  because it drives the negligible constraint violation to zero. The working set
  is corrupted and the solve diverges to `‖x‖ ~ 1e4` before dying with
  `Search_Direction_Becomes_Too_Small`.
  - Fix (`crates/pounce-algorithm/src/sqp/bfgs.rs`): before the first rank-2
    update, rescale `B` from `I` to `γI` with the Rayleigh-quotient curvature
    estimate `γ = sᵀy / sᵀs ∈ [λ_min(∇²L), λ_max(∇²L)]` — applied once, so the
    persistent damped updates still accumulate on top. Halves the failure rate
    on a broad ill-conditioned-QP sweep (~21% → ~10%) and clears the #358
    36-instance tail sweep entirely.
  - Three further fixes below close the rest of this gap.

### Fixed — three more active-set-SQP robustness holes on ill-conditioned QPs (#358)

Continuing the above: on a 500-instance convex-QP sweep (`n ∈ {2,3,5,8,12}`,
`cond ∈ {1…1e4}`, 20 seeds) failures fell from **~10% to 0.8%**, and the
`Search_Direction_Becomes_Too_Small` failure mode was eliminated entirely
(46 → 0). The error distribution of the solves that already succeeded is
unchanged (median `6e-11`, max true constraint violation `4e-11`).

- **Iteration-0 curvature probe.** The one-time BFGS sizing above cannot fire
  until iteration 1 — but iteration **0** already solves a QP, against the raw
  identity seed. Before that first QP, the driver now differences the gradient
  across a short steepest-descent probe step and seeds `B = γI` with the
  resulting Rayleigh quotient (one extra gradient evaluation). This is applied
  **only when the constraint Jacobian is detected constant** (linear
  constraints, or none), because the probe measures `∇²f`, which equals the
  Lagrangian Hessian `∇²L = ∇²f + Σλᵢ∇²cᵢ` only when the constraint-curvature
  term vanishes. On the Maratos problem `∇²f = 4I` while `∇²L ≈ I`, so probing
  there would over-scale fourfold — the linearity gate keeps that path
  untouched.
- **Scale-relative inner-QP tolerances.** `QpOptions::{feas_tol, opt_tol}` are
  absolute `1e-9`. As an *inner* subproblem the QP inherits the NLP's scale, so
  with `‖∇f‖ ~ ‖B‖ ~ 1e3` that is `~1e-12` relative — the f64 noise floor. The
  active-set solver could not certify its own optimality, burned its iteration
  budget, returned `MaxIter`, and the driver aborted with `QpStepFailed`. Both
  tolerances are now scaled by the QP data magnitude (clamped at `1e6`).
  Correctness is unaffected: the outer loop still gates optimality on the true,
  unscaled NLP KKT residuals, so a sloppier inner step can cost an extra outer
  iteration but never a false `Optimal`.
- **Quasi-Newton reset-and-retry.** A QP subproblem that still fails is usually
  reporting a drifted quasi-Newton Hessian, not a bad linearization. Rather than
  abort the whole solve, the driver now discards the accumulated curvature
  (retaining the matrix's scale), rebuilds the subproblem, and re-solves once
  from cold.

### Fixed — quasi-Newton curvature pair used inconsistent multipliers (#361)

- **The active-set-SQP quasi-Newton update now differences `∇L` at a single
  fixed multiplier**, per Nocedal-Wright §18.3:
  `y = ∇L(x_k, λ_k) − ∇L(x_{k−1}, λ_k)`. It previously held
  `∇L(x_{k−1}, λ_{k−1})` inside the Hessian object and differenced against
  `∇L(x_k, λ_k)` — two *different* multipliers — which for linear constraints
  contributes a spurious `Aᵀ(λ_k − λ_{k−1})` term: pure multiplier difference,
  carrying no curvature at all (the true `∇²L` equals `∇²f` there).
  - That fed a divergent loop — a perturbed `B` yields a worse QP multiplier,
    which injects a larger error into the next `y`, which corrupts `B` further.
    On equality-constrained QPs (where `λ` is sign-free) the multiplier was
    observed oscillating and growing exponentially
    (`−13, 19, −69, 104, −145, 581, −1320, 3176, …`) while **`x` sat on the
    exact optimum**. The solve burned its whole iteration budget and exited
    `Maximum_Iterations_Exceeded` *at the right answer*, because the reported
    stationarity residual is formed from that multiplier.
  - Equality-constrained sweep (144 instances): **92 failures on 0.9.0 → 0**.
    Inequality sweep (500 instances): **4 → 0**. All constraint families
    (linear equality, linear inequality, nonlinear inequality) now solve clean.
  - Fixed for both `damped-bfgs` and `lbfgs`. The consistent-multiplier form
    telescopes to `Σλᵢ(∇cᵢ(x_k) − ∇cᵢ(x_{k−1}))`, which correctly vanishes for
    linear constraints and is retained for nonlinear ones.

- **`sqp_tol` is now honored.** It was registered and documented as the max-norm
  KKT stationarity tolerance (default `1e-8`) but never read, so the looser
  `sqp_dual_inf_tol` (`1e-4`) governed alone and silently capped attainable
  accuracy. The convergence test now requires the tighter of the two — the only
  reading under which neither option is inert. Worst-case error on the #358
  sweep improves from `7e-5` to `5e-9` for ~10% more iterations. Same
  registered-but-inert defect family as #360.

### Fixed — `sqp_qp_*` inner-QP options were unusable (#360)

- **The five `sqp_qp_*` options are now registered** and reach
  `pounce_qp::QpOptions`: `sqp_qp_max_iter`, `sqp_qp_feas_tol`,
  `sqp_qp_opt_tol`, `sqp_qp_elastic_gamma`, `sqp_qp_anti_cycling`.
  `apply_qp_subproblem_options` read all five, but none was registered, so the
  options registry rejected every one with `OPTION_INVALID` ("Unknown option")
  and the reader was unreachable — the whole documented family was dead code.
  Added a guard test asserting each key is settable *and* propagates, and that
  unset keys keep the `pounce-qp` defaults.

**Correction to an earlier note in this section:** a previous revision of this
entry stated that equality-constrained ill-conditioned QPs were "measured
unchanged" by the #358 work and that the path showed run-to-run
nondeterminism. Both claims were wrong — artifacts of a benchmark harness that
seeded `numpy` from `hash()` of a tuple **containing a string**, which Python
randomizes per process, so every run silently generated different problems. With
integer-only seeds the solver is bit-for-bit reproducible, and the #358 work in
fact improved that family substantially (92 → 38 failures/144) before #361 took
it to 0.


### Tests — pyomo-pounce: `test_binary_check.py` failed on any Windows checkout (#366)

- Three tests were POSIX-only: the shadowing test joined `PATH` with a
  hardcoded `":"` instead of `os.pathsep`, and the fake `pounce` binaries were
  extensionless shell scripts, which Windows can neither execute (so
  `_build_id` probed them to `None`) nor resolve (the scan looks for
  `pounce.exe`). They failed on every Windows machine and passed on Linux,
  which is why CI never saw them and why they were misread as pre-existing
  failures on `main`.
- Fakes are `.bat` files on Windows where the path is probed directly; where
  PATH resolution is exercised, the stand-in is a copy of the real binary,
  with the shadowing fake's build id injected at the seam `check_binary`
  reads through, since a fabricated `.exe` cannot print a chosen `--about`.


## [0.9.0] - 2026-07-24

### Fixed — active-set-SQP stalled on curved-constraint NLPs via the Maratos effect (#349)

- **The active-set-SQP driver now defeats the Maratos effect** with a
  second-order correction (SOC) step in both globalizations (filter and
  l1-elastic). Previously the filter/l1 line searches rejected good unit SQP
  steps whenever the linearized constraints under-predicted the true (curved)
  constraint violation, so the solver either stalled
  (`Search_Direction_Becomes_Too_Small`) or exhausted its iteration cap on
  standard problems that Ipopt solves to machine precision. On the Maratos
  problem (`min 2(x₁²+x₂²−1) − x₁ s.t. x₁²+x₂²=1`) the solver now converges from
  every tested start under the exact, damped-BFGS, and L-BFGS Hessians — in a
  handful of iterations rather than 17–200 (or failing outright) — restoring the
  superlinear rate the Maratos effect destroys. HS6 now converges from its
  canonical `(-1.2, 1)` start under the exact Hessian, and from `(0.5, 0.5)`
  under all Hessians.
  - Fix: when the full step (α = 1) is rejected because it *increased* the
    constraint violation, the line search re-solves the QP with its
    general-constraint RHS re-centered on the trial-point constraint values
    (`c(x_k) → c(x_k + p) − A p`, Nocedal-Wright §18.11) to obtain a corrected
    full step. The correction is accepted only if it genuinely reduces the
    violation (Wächter-Biegler 2006 §3.3 `κ_soc` guard) and does not overshoot
    the QP step; a taken SOC step carries its own consistent multipliers and
    working set so the quasi-Newton Hessian update and next warm start stay
    well-conditioned.
  - The SQP driver also gained a **cold-start fallback**: when a warm-started QP
    subproblem stalls at its iteration limit (or hits a numerical breakdown) on
    a QP that is solvable from a clean start, the driver re-solves once from
    cold instead of surrendering with `QpStepFailed` — which additionally
    rescues several of the issue's `Search_Direction_Becomes_Too_Small` cases.
  - Not yet addressed: from far starts under a quasi-Newton Hessian, the hardest
    curved case (HS6 from `(-1.2, 1)`) can still stall at a blocked line search;
    a full feasibility-restoration phase remains future work.

### Fixed — `sqp_hessian=exact` with no Hessian gave `Internal_Error` (#348)

- Explicitly requesting an exact Lagrangian Hessian (`sqp_hessian="exact"` or
  `hessian_approximation="exact"`) on a problem that supplies no second
  derivatives died with `Internal_Error` (the QP subproblem reported
  "unbounded"). The automatic `hessian_approximation=limited-memory` downgrade
  runs first, but an explicit exact request was applied *after* it, re-enabling
  the `Exact` source with no Hessian behind it — the driver then evaluated a
  zero Lagrangian Hessian, so the QP lost its curvature term and went unbounded.
  This hit the Python `minimize`/`Problem` path whenever a `hess` was absent, or
  present but not usable as the Lagrangian Hessian (dict constraints, which are
  nonlinear-by-policy, or any genuinely nonlinear constraint). It affected only
  those callers: it worked unconstrained and bounds-only, and the exact Hessian
  is still honored when it *is* available (all-linear `LinearConstraint` blocks,
  or unconstrained, with a `hess`). Now, when the problem exposes no
  `hessian`/`hessianstructure`, an explicit exact request is downgraded to the
  limited-memory (L-BFGS) approximation with a warning instead of failing. The
  CLI `.nl` path (which always carries an exact Hessian) is unaffected. A
  regression test pins the downgrade-and-solve behavior and the still-honored
  exact path.

### Fixed — `solve_bvp` could not reach tolerances tighter than ~1.5e-8, exhausting the node budget instead (#345)

- **`pounce.solve_bvp` with a tolerance below ~1.5e-8 now converges** (in the
  same node count as `scipy.integrate.solve_bvp`) instead of refining to the
  `max_nodes` cap and returning `success=False`. On a textbook linear BVP
  (`y'' = -y`) at `tol=1e-8`, the adaptive solver previously used the entire
  node budget (100 000+ nodes) with its estimated RMS residual stuck at
  `1.47e-8`, while SciPy converged in 161 nodes; it now also converges in 161.
  The returned *solution* was already accurate, so this was a
  tolerance/`success`-reporting defect, not a wrong answer.
  - Cause: the mesh-refinement residual estimate scores each interval by
    `1.5 · col_res / h`, but the inner Newton solve stopped at a *fixed*
    absolute residual. After each refinement the warm-started iterate already
    sat below that stop, so Newton took zero steps and left `col_res` frozen at
    the interpolation level instead of driving it to round-off — so the
    estimate could not fall below a floor as `h` shrank.
  - Fix: when driving adaptive refinement, the Newton stop now scales with the
    mesh (`|col_res| < 2/3 · h · 5e-2 · tol` per interval), reproducing SciPy's
    criterion, so the collocation system is solved proportionally tighter as
    the mesh refines. A standalone `adaptive=False` solve is unchanged (it
    still drives the given mesh to round-off).

### Fixed — `pyomo_pounce` silently solved models with Binary/Integer variables as a continuous relaxation (#341)

- **`SolverFactory('pounce').solve(model)` now raises a clear `ValueError`
  when the model has an active, non-fixed `Binary`/`Integer` variable**,
  instead of silently handing POUNCE (a continuous NLP solver with no
  branch-and-bound or SOS handling) a MINLP and reporting `optimal` with a
  fractional value for a variable declared discrete (e.g. `b = 0.3` for a
  `Var(domain=Binary)`). A *fixed* discrete variable is unaffected — its value
  is already pinned, not a live decision.
  - The plugin also corrected its declared solver capabilities
    (`integer`/`sos1`/`sos2` are now `False`, not the generic `ASL` base
    class's default `True`), so `has_capability('integer')` no longer
    misreports support Pyomo's own solver-selection logic could act on.
  - This gap is inherited AMPL/ASL-ecosystem behavior (Ipopt-via-ASL does the
    same continuous relaxation on the identical model), not a pounce-specific
    numerical regression — but `pyomo_pounce` already fails loudly for
    comparable silent-wrongness risks elsewhere (the ambiguous `curve_fit`
    bounds shape, #260/#265; the stale-binary check, #315), so it now does
    here too.

### Fixed — HSDE exp/power driver reported `numerical_failure` (with a wrong iterate) when one cone-triple coordinate was pinned extreme (#339)

- **The non-symmetric (exp/power) HSDE conic driver could stall on, and
  discard, an essentially trivial exponential-cone problem where one
  cone-triple coordinate is pinned to a large-magnitude value (by an equality
  constraint elsewhere in the problem) while its companion slack should land
  near `0`.** Unlike #336 (mislabeling an already-correct point), this stalled
  the *iteration itself*: `max_step`'s backtracking line search for the
  exp/power blocks (no closed-form fraction-to-boundary, so it backtracks on
  cone membership) tested membership with a **fixed absolute** tolerance
  (`1e-12`). A non-symmetric cone coordinate that legitimately tracks the
  barrier parameter `μ` down the central path — e.g. the dual coordinate
  conjugate to the pinned argument, driven to `0` as `μ → 0` because that
  triple's cone-membership slack is comfortably non-tight — shrinks *with*
  `μ`. Once its magnitude neared the fixed floor, the backtracking started
  rejecting any further legitimate shrinkage as if the point were leaving the
  cone, collapsing the step length geometrically, iteration over iteration,
  until it hit exactly `0` well short of convergence — stranding the run on a
  stalled, badly wrong iterate/objective (not merely an under-labelled correct
  one) and reporting `numerical_failure`.
  - Fix: the interior-membership floor used by `max_step`'s exp/power
    backtracking (`nscone_mem_tol` in `hsde_nonsym.rs`) is now scaled by the
    current barrier parameter `μ`, capped at the legacy `1e-12` constant — bit-
    identical to before while `μ` is not yet tiny (any well-scaled solve), and
    shrinking in lockstep with `μ` once the central path has driven it far
    below that, instead of running into a fixed wall. This is a general fix,
    not a repro-specific one: it applies to any of the three exp/power cone
    coordinates, in either the primal or the dual block.
  - Regression: `crates/pounce-convex/tests/issue339_pinned_cone_arg.rs` pins
    one cone-argument variable via an equality constraint (distinct from
    #336's construction, which scales a GP's overall objective) across a scale
    sweep and a pinned-value sweep, asserting `numerical_failure` never fires
    and the objective stays correct (`~0`, the underflowed-`exp()` optimum)
    once scale is large enough to underflow.

### Fixed — HSDE exp/power driver reported `numerical_failure` on correct answers under extreme scaling (#336)

- **The non-symmetric (exp/power) HSDE conic driver no longer discards a
  correct solution as `numerical_failure` when the data are extremely (but
  legitimately) scaled.** Its post-loop status test keyed the reduced-accuracy
  salvage off the *absolute* KKT residual, which carries the raw, unnormalized
  complementarity gap `s·z`. When the optimal cone variables are large (e.g. a
  geometric program with `K = 1e12`, whose optimal cone values are `~1e6`), that
  absolute floor scales with them and can never reach the absolute tolerance —
  so a point that is primal-feasible, dual-feasible, and objective-correct was
  labelled a failure and `success=False`, and the answer was thrown away.
  - Fix: the adjudication now scores the recovered point on the *scale-relative*
    conic KKT residual (each residual normalized by its own term magnitudes, as
    ECOS/Clarabel do) at whichever of the final and best-snapshot iterates
    certifies tighter. A genuinely tight certificate at a dual-feasible point is
    still promoted to `Optimal`; a primal/dual-feasible point whose *normalized*
    gap is only moderately above `tol` (the accuracy plateau of a
    boundary-riding non-symmetric cone) is reported `OptimalInaccurate` —
    matching the symmetric SOC driver and ECOS/Clarabel's `*_inacc` — instead of
    a spurious `NumericalFailure`. The prior absolute reduced-accuracy fallback
    is retained, so a well-scaled near-`tol` stall salvages exactly as before.
  - Extends the scale-relative adjudication of #329 to the reduced-accuracy
    salvage; the conic analogue of the scale-normalization work in #286/#293.
    Infeasible / unbounded conic solves are unchanged (they terminate with a
    certificate status before the adjudication and are never falsely promoted).
  - Regression: `crates/pounce-convex/tests/issue336_scale_status.rs` reproduces
    both cases — the `K`-swept exponential GP (`Optimal` while the certificate
    is tight, then `OptimalInaccurate` on the plateau, never `NumericalFailure`)
    and the large-budget power cone — and pins the well-scaled end at `Optimal`.

### Fixed — active-set path printed a `nan` scaled objective (#313)

- On the active-set (`qp-active-set` / SQP) path, `final_scaled_objective` was
  left at its `NaN` default — the result block populated the unscaled objective
  and mirrored the residuals but never the scaled objective — so even a clean
  optimal solve printed `Objective ...: nan  <unscaled>`. It now mirrors the
  unscaled value (the SQP path does not thread `nlp_scaling`, so the two are
  equal). The rank-deficient-but-consistent equality QP from #313 (`x0+x1==2`
  with a redundant `2x0+2x1==4`) solves correctly on the active-set path
  (objective −6.75, the same value `qp-ipm`/`nlp` find); the "INTERNAL ERROR:
  Unknown SolverReturn value" the issue reported was from a stale pre-fix
  binary and does not occur on a current build. A regression test pins both the
  correct solve and the finite objective.

### Fixed — unbounded NLP with an inequality-slack recession ray reported as unbounded (#314)

- **An unbounded-below NLP whose recession ray increases an inequality
  constraint's slack is now correctly reported as `DivergingIterates`
  (unbounded) instead of a generic `ErrorInStepComputation`.** Completes the
  #274 family for the inequality-slack recession shape:
  `min -(x1³ + x2³)  s.t.  x1 + x2 ≥ 1`, `x1, x2` free is unbounded below along
  the feasible ray `x1 = t, x2 = 0`. The core "reported as solved"
  defect (`solve_result_num = 100`, exit 0) the adversary bot filed against
  was already fixed on `main` (the bot ran a stale pre-fix binary); this makes
  the CLI's verdict *correct* and consistent with the library, which already
  reported `Diverging_Iterates`.
  - Root cause: the #285 checked recession-ray proof already handled a ray in
    `null(A_eq)`, but a variable-swap bug in its
    `recession_blocked_by_inequality` gate returned the lower/upper finite-bound
    indicator vectors transposed, inverting the bound semantics. A ray that
    *increases* an inequality's slack (moves deeper into the feasible region)
    was spuriously treated as blocked, so the proof never fired and the KKT step
    broke down first. The proof is otherwise unchanged: it still requires a
    feasible large-norm witness, an unbounded free-variable escape side, a
    `null(A_eq)` direction, strict objective descent, and no finitely-bounded
    inequality actually driven toward its bound — so a bounded feasible region
    can never manufacture a false unbounded verdict.
### Fixed — `qp-active-set` internal error / churn on rank-deficient equality blocks (#313)

- **`solver_selection=qp-active-set` now solves a QP whose equality block is
  exactly rank-deficient but consistent** (one row an exact scalar multiple of
  another — routine when a constraint is written twice or a generator emits a
  scaled duplicate). Such a model with finite variable bounds and no inequality
  row previously aborted with `INTERNAL ERROR: Unknown SolverReturn value.` and
  exit 1 (zero iterations, `objective = 0.0`), while `qp-ipm` and `nlp` solved
  it correctly. The issue's reproduction now returns the exact optimum
  `x* = (0.5, 1.5, 3.0)`, `objective = -6.75`.
  - **Cause.** The equality+bounds path pins every equality row in each KKT and
    cannot prune a redundant one itself. The shared rank-repair guard is keyed
    off a *reported* recoverable factorization failure, but the inertia-control
    δ·I shift grows until the backend stops flagging the singular constraint
    block (masking the null direction) and returns a garbage solution instead of
    failing — so the guard never fired.
  - **Fix.** `factor_pinned_primal` now treats a nonzero inertia shift on a
    pinned KKT as a red flag: it rank-reveals the pinned rows and, if any is
    redundant, reports the recoverable failure the existing callers already
    prune on. The equality+bounds path factors its initial point through that
    helper and, on such a failure, delegates to the rank-deficiency-aware
    general path. Full-rank problems (the `δ == 0` common case) are unaffected.

### Fixed — convex QP convergence and honesty on tiny-curvature objectives (#293)

- **A convex QP whose Hessian curvature is tiny relative to its linear/constraint
  data now converges to the true optimum instead of silently truncating or
  returning a wrong unboundedness verdict.** These are the shapes a
  portfolio/least-squares user hits when a regularizer or a near-linear
  objective makes the curvature small; the answer was previously silently wrong
  or silently short of the optimum, with nothing in the status to flag it.
  - **Uniformly tiny Hessian converges.** `min ½·1e-12·(x0²+x1²) − x1 s.t. x ≥ 0`
    (optimum `x1*=1e12`, `f*=−5e11`) returned `iteration_limit` at `≈−4.95e11`;
    it now returns `optimal` at `−5e11`. The default HSDE driver's per-cone NT
    scaling never sees a Hessian 12+ orders below O(1), so on any non-clean
    status the solver now retries once with Ruiz equilibration (which lifts the
    scaled Hessian to O(1)) and accepts the retry only when it reaches a clean
    `optimal` — a genuinely hard problem keeps its truthful status. Covers both
    the unconstrained (`iteration_limit`) and constrained (`optimal_inaccurate`)
    manifestations.
  - **Spurious unboundedness refuted (machine-epsilon tail).** At `P ≈ 1e-20`
    a bounded problem could be wrongly certified `dual_infeasible`; a
    direct-driver reverify on the equilibrated problem now refutes the bogus
    certificate and returns the verified finite optimum (gated to `P ≠ 0`, so
    genuinely unbounded problems are unaffected).
  - **Scaling warning for the unconvergeable residue.** When no driver can
    converge a tiny-curvature problem at the default budget (e.g. a uniformly
    tiny Hessian coupled through an equality), the honest non-`optimal` status
    now carries an actionable scaling diagnostic naming tiny curvature as the
    cause. Surfaced on the CLI (stderr) and in the Python `solve_qp` result as a
    `scaling_warning` key. A clean `optimal` or a well-scaled hard problem emits
    nothing.

### Added — pyomo-pounce: detect a stale/shadowing `pounce` binary (#315)

- `pyomo_pounce.check_binary()` reports which `pounce` executable
  `SolverFactory('pounce')` will run, its build, whether it matches the
  wheel-bundled binary, and — critically — whether a *different* `pounce`
  earlier on `PATH` would shadow it. The comparison is on the git **commit**
  embedded in `pounce --about`, not the version string, because two builds can
  share the same `X.Y.Z` while behaving differently (a binary from before and
  after the #271/#272 dual-sign fix both report `0.9.0`).
- The plugin now **warns** when it falls back to a `PATH` binary (only reachable
  on a source/dev install without the `pounce-solver` wheel; a normal install
  runs the bundled binary). Note: with `pyomo_pounce` un-imported,
  `SolverFactory('pounce')` raises a clear `UnknownSolver` error — it never
  silently runs an unrelated binary. Docs updated to state the import
  requirement and point at `check_binary()`.
- Follow-up hardening of the above: the PATH scan now resolves each entry
  through `shutil.which`, so it looks for `pounce.exe` on Windows — previously
  the bare-`pounce` filename test found nothing there, and `check_binary()`
  reported no shadowing on Windows even when a stale `pounce.exe` was earlier on
  `PATH`. The build id also keeps a `+dirty` marker (`96fc5890+dirty`), so a
  build with uncommitted changes is distinguished from the clean build at the
  same commit; a `commit unknown` build (made outside a git checkout) still
  reads as unqueryable so two independent such builds never compare equal.

### Fixed — `curve_fit` two-parameter bounds: the last silently-transposing spellings now raise (#260)

- **`pounce.curve_fit` no longer silently transposes a 2-parameter box when the
  bounds are written as a *list of two lists* (`[[l0, l1], [u0, u1]]`) or a
  `2 x 2` ndarray.** #263/#265 fixed the outer-*tuple* spellings — the scipy
  `(lower, upper)` tuple is read scipy's way and the ambiguous tuple-of-pairs
  raises — but the same `n == 2` collision reached through a non-tuple container
  slipped past the guard: it fell through to the per-parameter pair-list reading,
  applied the transposed box, and returned `Solve_Succeeded` with a badly wrong
  fit (e.g. the reported optimum sitting on the *misread* box while the requested
  upper bound was violated, with no warning). This was the remaining half of
  issue #260 (its first comment enumerated all four spellings).
  - **All three fit surfaces are covered** — `curve_fit`, `curve_fit_minima`, and
    `curve_fit_streaming` share `_normalize_bound_arg`, so the guard applies
    uniformly.
  - **The unambiguous spellings are unchanged.** scipy's tuple of lists/arrays
    `([l0, l1], [u0, u1])` still follows scipy; pounce's per-parameter pair list
    spelled with `(lo, hi)` **tuples**, `[(l0, u0), (l1, u1)]`, still fits its
    box. Only the genuinely ambiguous 2×2 shapes (matching-container list-of-lists
    or ndarray, and mixed tuple/list pairs) now raise with a message naming both
    unambiguous spellings — consistent with the #265 decision that the ambiguous
    `n == 2` shape must be an error rather than a silent guess.

### Added — bound-multiplier `.sol` suffixes for Ipopt-parity reduced costs (#296)

- **The `.sol` writer now emits `ipopt_zL_out` / `ipopt_zU_out` variable-suffix
  blocks — the reduced costs / bound sensitivities Ipopt writes and Pyomo
  surfaces as `model.ipopt_zL_out[var]` / `model.ipopt_zU_out[var]` (AMPL's
  variable `.rc`).** pounce emitted the constraint duals (correctly signed
  since #287) but *no* bound multipliers, so a user migrating from Ipopt read
  `None` back with no error — a silent parity gap surfaced in the #271/#272
  dual audit. The underlying multipliers already existed internally
  (`mult_x_L`/`mult_x_U`, confirmed correct against Ipopt in the #271 audit);
  they are now written out.
  - **Sign convention, verified numerically against Ipopt 3.14** on
    bound-active models: Ipopt writes `ipopt_zL_out = +z_l` (≥ 0 at an active
    lower bound) and `ipopt_zU_out = −z_u` (≤ 0 at an active upper bound) —
    both equal to the objective-gradient component at the bound. pounce now
    matches to solver tolerance (e.g. `min (x−3)² s.t. 0 ≤ x ≤ 1`: `x*=1`,
    `ipopt_zU_out[x] = −4`, matching Ipopt's `−3.99999998…`).
  - **All three `.sol`-producing paths covered.** The NLP interior-point path
    lifts the converged bound multipliers through
    `OrigIpoptNlp::finalize_solution_z_l`/`_z_u` (fixed-var + scaling maps
    unwound, `obj_scale_factor` divided out); the convex QP and SOCP paths fold
    variable bounds into `G` rows, so a new
    `qp_extract::recover_bound_mults` reads the multipliers back out of the
    inequality-multiplier vector and applies the maximize `sign`.
  - **Pyomo populated automatically.** Pyomo's ASL `.sol` reader maps the suffix
    blocks straight onto the model suffixes, so `model.ipopt_zL_out` /
    `model.ipopt_zU_out` now come back populated after
    `SolverFactory('pounce').solve(model)` — matching Ipopt exactly (both
    leave the derived `rc` suffix `None`; AMPL, not the reader, derives `.rc`).
    No `pyomo-pounce` changes were needed.

### Fixed — the impossible-bound guard now lives in the convex core (#295, completes #275)

- **A `QpProblem` with a box that admits no finite point — a *present* `+∞`
  lower bound (`lb ≥ BOUND_INF`) or a *present* `−∞` upper bound
  (`ub ≤ −BOUND_INF`) — was silently mishandled by the `pounce-convex` core and
  could be reported `Optimal` at a point violating the bound by an infinite
  margin.** #275 (fixed in #291) rejected these sign-inconsistent infinite
  bounds, but the fix lived *entirely* in the Python layer
  (`python/pounce/qp.py::_validate`, `python/pounce/_minimize.py`). The core's
  bound-presence test (`expand_bounds` in `crates/pounce-convex/src/ipm.rs`) is
  sign-agnostic — it keys only on magnitude (`ub < BOUND_INF`,
  `lb > -BOUND_INF`) — so any surface reaching the core without
  `_validate` (the raw `_pounce` PyO3 bindings, `pounce-convex` used directly as
  a Rust crate) got the pre-#275 wrong `optimal`.
  - **Fix (defense in depth).** The invariant now lives in the core, so every
    surface inherits it. A new `QpProblem::bounds_admit_no_point` screen maps an
    impossible box to `QpStatus::PrimalInfeasible` — the same class the finite
    reversed box (`lb > ub`) already produces — at the earliest point of every
    bound-accepting solve entry (`solve_qp_ipm`, `_warm`, `_debug`;
    `solve_socp_ipm`, `_debug`) *before* the sign-agnostic bound expansion, and
    at the top of `presolve` (`presolve.rs`, beside the existing reversed-bound
    detection) so a presolve-then-solve caller is covered even when the solver
    is otherwise reached directly.
  - **Absent vs. impossible preserved.** An *absent* one-sided bound
    (`lb ≤ −BOUND_INF` / `ub ≥ +BOUND_INF`, the normal `±∞` encoding for
    "unbounded on that side") is untouched and still solves; a fixed variable
    (`lb == ub`) still solves; a finite reversed box (`lb > ub`) still reports
    `PrimalInfeasible`.
  - The Python-layer `_validate` guard is retained as the first line of defense
    (it raises a `ValueError` with an index-named message earlier, before the
    solve). This change makes the core a certified second line.
  - Verified by reproducing the wrong `optimal` on the raw `_pounce` binding on
    `main` and confirming it now returns `primal_infeasible`
    (`python/tests/test_qp_host.py::test_raw_binding_rejects_impossible_bounds`,
    plus Rust unit tests in `crates/pounce-convex/tests/bounded_form.rs`).

### Tests — external/analytic dual-SIGN regression guard (#294, hardening after #271/#272/#287)

- **A durable guard so a constraint-dual sign inversion cannot ship silently
  again.** The #271/#272 flip inverted every AMPL/Pyomo/GAMS marginal for an
  unknown span of releases and no automated check caught it: the benchmark
  suite compares objectives/status/iterations/wall-time but never duals, and
  `pounce verify` keeps the better KKT residual of `+λ`/`−λ` so it certifies
  either sign. Agreement *between pounce surfaces* is not a guard (a uniform
  flip satisfies it); each new assertion pins a dual against an **external**
  reference (IPOPT via pyomo) or an **analytic** value with an explicit
  expected sign, on every dual-bearing surface:
  - `crates/pounce-cli/tests/issue_294_dual_sign_regression.rs` — the `.sol`
    marginal block (AMPL `d obj/d b`) and JSON `solution.lambda` (internal
    Lagrange convention), pinned to the equality QP `min x0²+x1² s.t. x0+x1=2`
    (marginal `+2`, lambda `−2`) and the Wyndor Glass LP (active-inequality
    marginals `[0,−1.5,−1]`, lambda `[0,1.5,1]`). New fixture
    `tests/fixtures/wyndor_min.nl`.
  - `python/tests/test_dual_sign_regression.py` — `pyomo-pounce` `model.dual`
    (cross-asserted equal to IPOPT's on the same model, both senses),
    `minimize(...).info["mult_g"]`, and `solve_qp` `y`/`z`, pinned to the
    Wyndor shadow prices `(0, 1.5, 1)` and the equality multiplier `y=−2`.
  - `python/tests/test_gams_link.py` — extends the `gams_pi` cases with the
    Wyndor shadow prices for both minimizing and maximizing senses.
  The guard was confirmed to have teeth by momentarily flipping the sign in the
  `.sol` writer and in `gams_pi` and observing the exact-value asserts fail
  (reverted).

### Fixed — convex QP IPM falsely certified a mixed-scale Hessian unbounded (#293, completes #273/#290)

- **A bounded QP with a mixed-scale Hessian was returned as `dual_infeasible`
  — a confident but wrong unboundedness certificate.** `pounce.solve_qp` on
  `min ½(1e6·x0² + 1e-12·x1²) − x1  s.t.  x ≥ 0` (`P = diag(1e6, 1e-12)`) is
  bounded with a unique optimum `x1* = 1e12`, `f* = −5e11`, yet it reported
  `status='dual_infeasible'`. Returning a wrong answer with a certificate
  attached is the worst outcome the solver can produce.
  - **Root cause.** The dual-infeasibility (unboundedness) certificate accepts a
    direction `d` as a recession ray of the quadratic when its curvature
    vanishes. #290 tested the *residual* `‖Pd‖ ≤ rtol·‖d‖·max|P|` — a global
    scale that cannot express `d ∈ null(P)` when `P`'s eigenvalues span many
    orders. The descent ray `x1` has genuine curvature `dᵀPd = 1e-12 > 0`
    (so the objective has a finite minimum along it), but `1e-12` is `18` orders
    below `max|P| = 1e6`, so it read as "null relative to `‖P‖`" and was falsely
    certified unbounded.
  - **Fix.** The certificate now tests the **normalized directional curvature**
    `dᵀPd/‖d‖²` — the curvature per unit length along `d`, an eigenvalue-scale
    quantity a diverging iterate cannot inflate — against an absolute floor
    (`RECESSION_CURV_TOL = 1e-20`, in `crates/pounce-convex/src/ipm.rs`). A
    convex QP recedes along `d` iff that curvature is zero *and* `cᵀd < 0`. A
    bounded problem floors the normalized curvature at its smallest real
    directional eigenvalue (`1e-12` here, `1e-16` for the #273 `P = 1e-16`
    case), which stays far above the floor and is correctly *not* certified; a
    genuine recession drives it to zero (exactly `0` for an LP or an axis-aligned
    null block; `~1e-140` and shrinking when a singular `P`'s curved variable is
    pinned to a bound as the null variable diverges), so genuine unboundedness —
    LPs, singular-`P` nullspace rays, exp/SOC-cone recessions — is still
    certified. The mixed-scale case now converges to `Optimal` at `x1* = 1e12`.
  - **Preserved.** The #273/#290 tiny-Hessian cases (`P = 1e-10…1e-16` strictly
    convex) stay `Optimal`; true recession rays stay `dual_infeasible`; the
    normal-problem benchmark corpus is unchanged. A *uniformly* tiny Hessian
    (`P = diag(1e-12, 1e-12)`, the #293 second symptom) still reaches its
    iteration limit rather than the optimum — an honest, non-wrong status,
    unchanged from before; the curvature fix removes only the wrong certificate,
    not that conditioning-limited convergence.

### Fixed — NaN gradient / constraint Jacobian silently reported `Solve_Succeeded` (#292)

- **An objective gradient or constraint Jacobian that returns `NaN` was laundered
  into a successful solve.** `pounce.minimize(lambda x: float(x[0]**2),
  np.array([0.5]), jac=lambda x: np.array([np.nan]))` returned `success=True`,
  `message='Solve_Succeeded'`, `fun=0.25`, `nit=0` — the most dangerous shape,
  because `fun` is finite (the value at `x0`) so the caller got no signal at all.
  Root cause: the max-norm behind the dual-infeasibility measure
  (`crates/pounce-linalg` `amax` / BLAS `iamax`) silently *drops* `NaN` — `NaN >
  m` is `false`, so a `NaN` component leaves the running max untouched and the
  KKT error read `0.0`. The existing finiteness guard
  (`ipopt_alg.rs`, `!nlp_err.is_finite()` → `Invalid_Number_Detected`) never
  fired because the value it checks had already been laundered to zero.
  - **Fix.** `curr_nlp_error` (`crates/pounce-algorithm/src/ipopt_cq.rs`) now
    detects any non-finite component of the Lagrangian gradients, constraint
    residuals, and complementarity blocks — via the `NaN`-propagating `asum`
    behind `has_valid_numbers`, *not* `amax` — and surfaces a non-finite KKT
    error, so the guard fires `Invalid_Number_Detected` honestly. This covers a
    `NaN` gradient, a `NaN` constraint Jacobian (through `∇_x L`'s `Jᵀy` term),
    and `NaN`/`Inf` residuals. The check is confined to the convergence/error
    measure; the general `amax` semantics that step-size selection, the line
    search, and the divergence detectors depend on are deliberately left
    unchanged. A `fun` that returns `NaN` already failed honestly and still
    does; normal all-finite solves are bit-for-bit unchanged (the finiteness
    check is a no-op when every component is finite).

### Fixed — convex QP IPM stalled on huge-magnitude objectives, never labeling the optimum `Optimal` (#286)

- **A badly-scaled convex QP (`cond(P) = 1e10` with objective coefficients of
  magnitude `O(1e22)`) exhausted the default iteration budget and returned
  `IterationLimit` at a point violating the box by ~0.88 — a problem
  CLARABEL/OSQP/ECOS/SCS all solve at their defaults.** Even at `max_iter = 5000`
  the status stayed `IterationLimit` at a tightened tolerance despite the iterate
  being accurate to ~5e-9. Root cause (`crates/pounce-convex/src/ipm.rs`,
  `solve_qp_core`): the default HSDE driver deliberately skips Ruiz
  equilibration — its per-cone NT scaling conditions the *constraint* system —
  but nothing normalized the sheer *magnitude* of the objective data `(P, c)`.
  With `‖P‖ ~ 1e22` the homogeneous embedding's `τ` collapsed toward the `τ → 0`
  certificate boundary: the dual residual scale swamped the `τ`-row, primal
  feasibility then crawled, and the solve ground to its cap even though the dual
  and gap had converged in a few dozen steps.
  - **Fix.** The HSDE QP/LP path now normalizes the objective by a scalar
    `σ = max(‖P‖∞, ‖c‖∞)` — argmin-invariant, so the minimizer is unchanged —
    before the solve, and maps the recovered dual multipliers and objective back
    (`y, z ← σ·y, σ·z`, `obj ← σ·obj`; the primal `x` needs no correction). With
    an `O(1)` objective the embedding's `τ` stays healthy and the badly-scaled QP
    now converges to `Optimal` **in 9 iterations** at the correct optimum
    (objective matched to CLARABEL / exact active-set enumeration to ~1e-7
    relative), the cost scaling Clarabel/OSQP apply as a matter of course.
  - **Normal problems are untouched.** The normalization is a no-op (`σ = 1`,
    rounded to a power of two so the round-trip is exact) unless the coefficient
    magnitude is large enough to genuinely destabilize the embedding
    (`σ·ε > tol`, the same crossover the scale-relative stop already uses). It
    keys on the objective *coefficient* magnitude, not the objective *value*, so
    the large-data QP cluster (POWELL20/BOYD, whose large objective comes from a
    large `‖x*‖` with modest coefficients) is left bit-for-bit unchanged. The
    full `.nl` benchmark corpus (44 problems) is identical before/after in both
    `solve_result_num` and objective, and the convex QP/LP suites are unchanged.

### Added — near-LICQ conditioning diagnostic for `QpSensitivity` (#284)

- **`QpSensitivity.parametric_step` no longer silently over-damps `dx/db` on a
  near-rank-deficient (near-LICQ) KKT.** When the active-constraint gradients are
  *nearly* — not exactly — rank-deficient (e.g. two almost-parallel equality
  rows), the static regularization `δ` floored the smallest KKT singular value
  and a single back-solve returned a smooth but badly wrong sensitivity (up to
  ~100% relative error), with `weakly_active_indices` empty, `kkt_dim` full, no
  status change, and no exception — so a caller had no way to know. Two changes
  close the gap (`crates/pounce-convex/src/sensitivity.rs`, the PyO3 getters in
  `crates/pounce-py/src/qp.rs`, and the `pounce.qp.QpSensitivity` wrapper):
  - **Diagnostic.** New `QpSensitivity.kkt_cond_estimate` (a cheap Hager 1-norm
    estimate of the KKT condition number `κ₁`), the boolean
    `QpSensitivity.ill_conditioned` (fires when `κ₁ > 1e14`), and
    `QpSensitivity.last_step_residual` (the relative KKT residual the most
    recent step achieved, measured against the *unregularized* system). On the
    near-LICQ sweep the flag fires and the residual is large; on the
    well-conditioned equality-only and active-set cases (κ₁ ≈ 3–8e9) it stays
    quiet — no false alarm. All three are new, backward-compatible attributes.
  - **Accuracy.** Each parametric step is now refined against the unregularized
    KKT, stripping the `O(δ)` regularization bias wherever the information
    survives in double precision. `dx/db` now tracks a plain float64 LU solve
    (e.g. the badly-scaled equality-only case improved from ~1.6e-7 to ~7e-14
    relative error); genuinely singular cases that refinement cannot recover are
    the ones the diagnostic flags. Well-conditioned solves are unaffected (their
    first residual is already at round-off, so refinement is a no-op).

### Fixed — NLP-path divergence detector missed an unbounded LP whose recession ray lies in `null(A_eq)` over free variables (#285)

- **On the forced-NLP path, an unbounded LP whose recession ray lives in the
  equality null space over *free* variables (e.g. `min −x0 s.t. x0 − x1 = 0`,
  ray `d = (1, 1)`) reported `Maximum_Iterations_Exceeded` instead of the
  unbounded verdict `Diverging_Iterates`** that Ipopt, scipy/HiGHS and Clarabel
  all return on the identical model. The existing `diverging_iterates_tol`
  (`1e20`) magnitude guard only fires once `|x|_∞` crosses `1e20` and only
  accumulates its streak on geometric growth; a recession ray in an equality
  null space is walked out by regularized zero-Hessian Newton steps whose
  growth decelerates so hard that `|x|` never reaches `1e20` within `max_iter`
  (it stalls around `5e17` even at `max_iter = 3000`), so the guard never
  fired. No wrong answer was ever certified — both statuses are non-success —
  but the conservative fallback hid a definite certificate. The fix adds a
  second, independent **checked recession-ray proof** to the running divergence
  guard (`crates/pounce-algorithm/src/ipopt_alg.rs`), active from a far lower
  magnitude floor (`1e10`): a genuinely *feasible* iterate of large norm
  already witnesses that the feasible region is unbounded, and the proof
  additionally certifies the escape direction lies in `null(A_eq)`
  (`‖J_c x‖∞ ≪ |x|∞`), is not blocked by any finitely-bounded inequality
  (`Pd_L`/`Pd_U` expansion), heads toward a variable side with no finite bound
  (the existing free-to-escape check), and strictly lowers the objective
  (`∇f·x ≤ −ε‖∇f‖‖x‖`) — for several consecutive *growing* iterations. Because
  a bounded feasible region cannot supply a growing sequence of feasible
  over-floor iterates, this remains a proof rather than a heuristic and cannot
  manufacture a false `Diverging_Iterates` on a bounded problem. The existing
  `1e20` magnitude guard is unchanged (new path is a pure `OR` branch), so the
  unbounded shapes that already worked — unconstrained rays,
  bounded-inequality rays — still report unbounded, and the #248 / #252
  bounded / finite-optimum controls stay green. Regression tests in
  `crates/pounce-algorithm/tests/repro_issue285.rs` pin the fixed shape as
  `DivergingIterates` alongside four bounded controls (variable-bounded,
  fully-pinned by equalities, and inequality-capped free variable) as optimal.

### Fixed — non-symmetric HSDE driver returned bare `numerical_failure` on infeasible/unbounded exp/power programs (#283)

- **The exponential/power-cone (non-symmetric HSDE) driver degraded genuinely
  infeasible and unbounded programs to `numerical_failure`** instead of the
  definite `primal_infeasible` / `dual_infeasible` its symmetric (LP/SOC)
  counterpart and the ECOS/SCS/CLARABEL oracles all report. No wrong answer was
  ever certified — the cost was diagnostic quality — but callers could not tell
  "the model is infeasible/unbounded" from "the solver hit numerical trouble".
  Two exact, certificate-backed detectors close the gap
  (`crates/pounce-convex/src/hsde_nonsym.rs`, `ipm.rs`):
  - **Recession / unboundedness (`dual_infeasible`).** A recession ray lands on
    the *boundary* of the cone (e.g. the exp cone's `y = 0` face), but the
    recession-membership test used the strict-**interior** oracle `in_primal_cone`
    (`y > tol`), which rejected the genuine ray and let the iterate diverge to
    `numerical_failure`. A new **closure**-membership test `in_primal_closure`
    accepts the boundary/recession faces of `cl(K_exp)` and `K_α`, so
    `min u s.t. (u,1,t) ∈ K_exp` now certifies `dual_infeasible`. It cannot
    false-positive: the certificate still requires a meaningfully negative
    directional cost with near-zero `A`/`P` residuals, i.e. a true recession ray.
  - **Cone-domain infeasibility (`primal_infeasible`).** A power/exp cone requires
    two of its coordinates `≥ 0` at every feasible point; when the data pin such a
    coordinate strictly negative (a constant `y = −1` slack, or `y = t−2` forced
    `≤ −1` by another row), the embedding stalls with a small-but-finite Farkas
    residual that never clears `FARKAS_RESID_TOL` (~1e-10). A new exact setup-time
    screen, `detect_cone_domain_infeasible`, propagates variable ranges through the
    `≥ 0` and equality rows by sound interval arithmetic (FBBT) and reports
    `primal_infeasible` when any cone-domain slack has a strictly negative upper
    bound (or a variable range goes empty). Every derived bound is a valid
    implication of feasibility, so a contradiction proves infeasibility — no
    feasible/bounded problem is ever flagged. The barely-feasible controls
    (`y`-slack reaching `+0.05` just off the boundary) still solve to `optimal`.
  All ten analytically-certified adversary instances now match the oracles
  (previously 3 mismatches), and the hunted false `primal_infeasible` on a
  barely-feasible control does not occur.
### Fixed — active-set path certified a feasible problem infeasible when m/n ≫ 1 (#282)

- **The active-set QP path (`solver_selection="qp-active-set"` and
  `algorithm="active-set-sqp"`) returned a confident `Infeasible_Problem_Detected`
  on a genuinely feasible problem** whenever constraints were grossly
  over-multiplied and there was no interior (Slater fails). The adversary repro is
  `min ½‖x−e‖² s.t. a_iᵀx ≤ 0, i=1..40` with `a_i` random unit vectors in R⁵: the
  positive hull of `{a_i}` spans R⁵, so the feasible set collapses to exactly `{0}`
  (all 40 rows active at a 5-D point, LICQ fails, multipliers non-unique). The true
  optimum is `x* = 0`, and `0` is trivially feasible (`G·0 = 0 ≤ 0` exactly), yet at
  `m/n ≥ 5` the solver reported infeasible — even when *started at the exact
  optimum* `x0 = 0`.
- **Root cause** (`crates/pounce-qp/src/solver.rs`, `solve_elastic`): the l1-elastic
  phase-1 minimizes the constraint violation via an augmented QP. On this extreme
  degeneracy the augmented active-set solve **stalls at `MaxIter`** (churning the
  working set from an already-optimal seed — many more active rows than variables,
  a rank-deficient active Jacobian), leaving sub-`feas_tol` residual elastic slacks.
  `solve_elastic` then declared `Infeasible` **purely from the residual slacks,
  ignoring whether phase-1 had actually converged** — a false certificate, since a
  feasible problem has no Farkas proof.
- **Fix.** Two parts. (1) *Recovery to the correct answer:* when residual slacks
  remain, re-solve the original QP with a warm-started **phase-2** active-set solve
  (which bypasses the elastic audit and cannot re-enter phase-1) from the
  near-feasible points phase-1 produced; phase-2 from a feasible seed of this
  geometry converges in a handful of pivots, recovering `x* = 0`. (2) *Honest
  status when recovery fails:* only emit `QpStatus::Infeasible` when phase-1 itself
  **converged** to its minimal-l1 optimum (a genuine certificate); if phase-1
  stalled (`MaxIter` / numerical breakdown), report that non-committal status
  instead. The SQP driver
  (`crates/pounce-algorithm/src/sqp/sqp_alg.rs`) now maps a QP `MaxIter` /
  `NumericalError` subproblem outcome to the new honest
  `SqpStatus::QpStepFailed` → `Search_Direction_Becomes_Too_Small` rather than a
  hard error or a false infeasible.
- **Result.** Across the reported m-sweep started at the exact optimum `x0 = 0`
  (`m = 12..40`), the path now returns `Solve_Succeeded` at `x* = 0` everywhere
  (previously `m = 30, 40` reported `Infeasible_Problem_Detected`). From an interior
  start `x0 = 0.1·e` where recovery cannot find the feasible point, it returns the
  honest `Search_Direction_Becomes_Too_Small` (`success=False`) — **never**
  `Infeasible_Problem_Detected`. The classic anti-cycling geometries that already
  worked are unchanged: the duplicated `[0,1]⁸` hypercube (128 rows, 64 active,
  1 iter), Beale's regularized cycling LP, the LICQ-degenerate vertex in R², and the
  everyday convex QPs all still solve. Regression:
  `collapsed_cone_no_interior_not_false_infeasible` in
  `crates/pounce-qp/src/tests/analytical.rs`.

### Fixed — `sos_minimize` certified a wrong minimizer as exact on Rosenbrock (#281)

- **`sos_minimize` reported `is_exact=True, num_minimizers=1` while returning a
  point that does not attain the certified bound.** On boxed Rosenbrock-2D
  (`f = (1-x)² + 100(y-x²)²`, unique global minimizer `(1,1)`, `f* = 0`) the
  moment relaxation is not flat at the true measure, so the SDP's first moments
  land in the flat "banana" valley at `≈(0.86, 0.74)` — `0.26` from `(1,1)` yet
  with `f ≈ 0.017–0.020`, close to the (correct) lower bound `≈0`. That point was
  handed back as an exact minimizer. The lower bound itself was always sound; the
  defect was the atom-objective consistency guard in `recover_from_moments`
  (`crates/pounce-convex/src/sos.rs`), whose tolerance `ATOM_OBJ_TOL = 1e-3` was
  too loose: Rosenbrock's flat valley makes a far-off point still read `f ≈ bound`,
  so the guard admitted it. The threshold is tightened to `1e-6` — measured to sit
  ~4× above the worst genuine extraction that asserts exactness (the rank-4
  `facial_reduction_four` case at `2.35e-7`) and ~10× below Rosenbrock's residual
  (`1.05e-5`), both deterministic. When it fires, `is_exact` is withdrawn and the
  still-valid lower bound is returned with **no** minimizers (the safe failure),
  rather than a confidently wrong point. Known-good extractions are unaffected:
  boxed Booth `(1,3)`, Dixon-Price n=2, the three-/six-hump camels, and the
  facial-reduction multi-atom cases all still certify and extract correctly. The
  enforced invariant is now `is_exact ⇒ f(extracted) ≈ lower_bound`.

### Fixed — best-acceptable fallback ranking degenerated to objective-only outside the feasibility band (#280)

- **The NLP best-acceptable fallback could prefer a *strictly more infeasible*
  point**, completing the #267/#270 fix. That fix ranked recorded acceptable
  points by a `(feasible_enough, objective)` key, but the key was a two-class
  *partition*, not an ordering: once **both** the incumbent and a recorded
  candidate sat outside `FEASIBLE_ENOUGH_CAP = 1e-2`, `a_ok == b_ok` and
  `ranks_better_within_band` fell through to a bare `a_obj < b_obj` — reading
  neither point's constraint violation, exactly the pre-#267 objective-only rule.
  Among two infeasible points it again picked the better *objective*, which can
  be *more* infeasible. With the opt-in dual-divergence guard on
  (`dual_diverging_streak=2`) and a widened `acceptable_constr_viol_tol=1e0`, the
  fallback on `deb7` discarded the incumbent at violation `5.292e-1` for a
  recorded point at `9.951e-1` (worse) to gain 36 % of objective, returning it
  under `solve_result_num=100`; `pounce verify` rejected that point. The ranking
  is now a **total order**: each violation is clamped *up* to the band before
  comparison (`viol.max(band)`), so points inside the band tie on feasibility and
  objective decides (unchanged), while outside the band the actual violation
  decides and a strictly-more-infeasible point can never win. Both the record and
  read sides (`record_best_acceptable`,
  `honour_best_acceptable_after_dual_guard`) route through the same
  `ranks_better_within_band` helper, so they cannot disagree. #270's headline
  (`autocorr_bern55-06` at obj `-2303.9999305`, viol `4.149e-5`) and stock-default
  behaviour (honest `MaximumIterationsExceeded` on non-convergence, no fallback
  under a success status) are unchanged. Config-gated: the guard is off by
  default, so stock-tolerance solves never hit this.

### Fixed — `solve_socp` panicked across the FFI boundary on a zero-dimension cone block (#278)

- **A zero-dimension cone block made `solve_socp` raise
  `pyo3_runtime.PanicException`** (a Rust panic crossing the FFI boundary,
  which Python cannot catch as a normal error) instead of a clean, catchable
  `ValueError`. The validator only checked that the cone dimensions *sum* to
  `rows(G)`, so a 0-dim block — contributing 0 rows — passed every documented
  check and then aborted inside a cone constructor: `SecondOrderCone::new(0)`
  hit an `assert!`, and `PsdCone` indexed `vals[0]` on an empty eigenvalue
  vector. It was reachable three ways, all ordinary user input: an explicit
  `("soc", 0)` / `("psd", 0)`; a **negative** dimension (silently saturated to
  `0` by `v.round() as usize`); and a **fractional** dimension below `0.5`
  (rounded to `0`). `parse_cones` now validates each cone's dimension at the
  Python boundary — rejecting non-finite, non-integer, negative, and
  below-minimum values (`soc`/`psd` need `≥ 1`; the empty-safe nonnegative
  orthant still permits `0`) — with a clear `ValueError` naming the offending
  cone's index, kind, and value. As defense in depth, the constructors no
  longer panic on a `0` reaching them from any path: `SecondOrderCone::new`
  drops its `assert!`, and `PsdCone::min_eig` / `max_step` short-circuit the
  degenerate `n = 0` block. A valid `("nonneg", 0)` block still solves
  unchanged, and every well-posed SOCP/SDP/exp/pow solve is unaffected.

### Fixed — convex `tol` / `max_iter` options were unvalidated (#277)

- **`solve_qp` / `solve_socp` (and the batch, multi-RHS, factorization, and
  sensitivity entry points) applied no validation to `tol`,** while every
  other pounce surface (`minimize`, the CLI, `sos_minimize`) rejects a
  non-positive or non-finite tolerance with `OPTION_INVALID`. Consequences: an
  unsatisfiable `tol` (`0`, `-1`, `NaN`, `Inf`) silently burned every
  iteration, and a huge finite `tol` (`1e300`) short-circuited at the
  interior-point *starting* iterate — the convex IPM tests `max KKT residual
  <= tol` at every iterate, so an O(1) tolerance "passes" immediately —
  returning `status="optimal"` after **0 iterations at a wrong point**
  (`x=(0,0)`, `kkt_error=1.0` on the issue's repro). That mislabel propagated
  through the facade: `minimize(solver_selection="qp-ipm", tol=1e300)` reported
  `success=True, nit=0`. Every convex entry point now rejects `tol <= 0`,
  non-finite `tol`, and `tol >= 1` with a clear `ValueError` naming the option
  and value. Capping at `1.0` (rather than accepting any positive `tol`)
  guarantees an accepted tolerance with an `"optimal"` result carries
  `kkt_error <= tol < 1`, i.e. a genuinely near-stationary point — a wrong
  point can never again be labeled `optimal`. A legitimate tight `tol` (e.g.
  `1e-8`) is untouched.
- **`solve_qp(max_iter=-5)` leaked a raw PyO3
  `OverflowError: can't convert negative int to unsigned`** from the `usize`
  binding. `max_iter` is now validated in Python — before it reaches the
  binding — on every convex entry point, so a negative, zero, or non-integer
  value raises a named `ValueError` instead.

### Fixed — integer options above `i32::MAX` silently truncated (#276)

- **`Problem.add_option` / `minimize(...)` wrapped out-of-range integer
  options instead of rejecting them.** The PyO3 binding converted an
  extracted `i64` with a bare `i as Index` (`Index = i32`) cast, which
  *wraps* rather than checks. So `max_iter = 2**32 + 3` silently truncated to
  `3` and ran exactly three iterations with no error or warning, while the CLI
  and Pyomo plugin rejected the same input — the surfaces disagreed. The cast
  is now a checked `i32::try_from`, so any integer option (`max_iter`,
  `acceptable_iter`, `print_level`, `max_soc`, … all share this one path)
  outside the signed-32-bit range raises a clear `ValueError` naming the
  option and quoting the value the user actually passed (not the truncated
  one). In-range values — including the `i32::MIN`/`i32::MAX` boundaries and
  legitimate negatives — still work unchanged.

### Fixed — non-finite inputs silently reported success (#275)

- **`lb = +inf` / `ub = -inf` were dropped as if the bound were absent.** The
  solver's presence test (`lb > -BOUND_INF`, `ub < BOUND_INF`) is
  sign-agnostic, so a bound no finite value can satisfy was discarded and
  `solve_qp` returned `status="optimal"` at a point violating it by an
  infinite margin. `solve_qp` / `solve_qp_batch` / `solve_qp_multi_rhs` and
  `minimize` now reject these spellings with a message naming the index.
  `±inf` on the *absent* side (lower `-inf`, upper `+inf`) remains the
  documented one-sided encoding, and a finite reversed box (`lb=1 > ub=0`) is
  still reported as `primal_infeasible` rather than raising.
- **A NaN or infinite `x0` reported `Solve_Succeeded` with `fun=nan` at
  iteration 0.** Every convergence test is a comparison against a tolerance
  and comparisons against NaN are False — including the ones that would have
  rejected the iterate — so the loop fell through to "converged". `minimize`
  now rejects a non-finite `x0`.

### Fixed — unbounded NLP reported in the AMPL "solved" family (#274)

- An unbounded NLP could be written to the `.sol` as
  `Solved_To_Acceptable_Level` with `solve_result_num = 100`, which lands in
  AMPL's *solved* range — so Pyomo reported `TerminationCondition.optimal` and
  **loaded the diverging iterate as an optimal solution**. `min -exp(x) s.t.
  x >= 0` returned `x ≈ 110.4`, `obj ≈ -8.8e47` under that label.
- Cause: the near-feasible restoration re-entry detector claimed acceptability
  from the *primal* residual alone. A feasible iterate can still be
  arbitrarily far from stationary, which is exactly what an unbounded
  objective looks like — the constraints stay satisfied (`inf_pr ≈ 1.7e-10`)
  while the iterates run off (`inf_du ≈ 8.8e+47`). The only guard was a
  non-finite check, and `-8.8e47` is finite.
- The detector now requires the point to pass the full acceptable-level
  triplet — including `acceptable_dual_inf_tol` — before reporting
  `Solved_To_Acceptable_Level`; otherwise it surfaces the same honest
  restoration-cycle status its sibling exits use. The CLI and the library API
  now agree on this model (`Error_In_Step_Computation`), where previously the
  library reported failure and the CLI reported success.
### Fixed — strictly convex QP falsely reported unbounded (#273)

- The convex IPM's dual-infeasibility (unboundedness) certificate tested
  `‖Pd‖ ≤ rtol·‖d‖` for the candidate recession direction `d`. Because `‖Pd‖`
  is itself proportional to `‖P‖·‖d‖`, the `‖d‖` cancelled and the test
  collapsed to `‖P‖ ≤ rtol` — a bare comparison of the Hessian's magnitude
  against the absolute constant `1e-10`, with no reference to `d` at all.
- Consequence: **any** strictly convex QP with a small enough Hessian was
  certified unbounded despite having a finite minimizer. `min -x + x²/(2M)
  s.t. x >= 0` (unique minimum `x* = M`) was reported unbounded for every
  `M >= 1e10`, terminating after 2 iterations twelve orders of magnitude short
  of the optimum, on a problem Ipopt and pounce's own NLP path both solve
  exactly.
- The residual bound is now scaled by `‖P‖`, restoring the intended meaning:
  a relative test for `d ∈ null(P)`. LPs (`P` empty, `Pd` exactly zero) and
  genuinely singular Hessians with `d` in the nullspace are unaffected, so
  real unboundedness is still detected.
### Fixed — `check_psd` validated a different matrix than the solver used (#279)

- With a `scipy.sparse` COO `P` containing **duplicate `(row, col)` entries**,
  the PSD guard reconstructed the Hessian by *assignment* (last duplicate
  wins) while the solver *sums* them, per the COO convention. The guard
  therefore validated a matrix that was never solved: an indefinite `P` passed
  `check_psd=True` and `solve_qp` returned `status="optimal"` at a saddle
  point. `coo_matrix(([2, 2, 1.5, 1.5], ([0, 1, 1, 1], [0, 1, 0, 0])))` is
  indefinite when summed (eigenvalues `[-1, 5]`) but positive definite under
  overwrite (`[0.5, 3.5]`); the identical **dense** matrix was always rejected
  correctly.
- Duplicate entries now accumulate, so sparse and dense inputs reach the same
  verdict. The mirror write is skipped on the diagonal so a duplicated
  diagonal entry is not counted twice.

### Fixed — constraint dual sign convention (#271, #272)

- **`.sol` / Pyomo `model.dual`** carried pounce's internal Lagrange
  multiplier instead of the AMPL *marginal* `d obj / d b`, so every dual came
  back negated relative to Ipopt, glpk, CBC and CONOPT (#271). Objectives and
  primal solutions were never affected. The `.sol` writer now performs the
  conversion. **This flips the sign of duals read through Pyomo and AMPL** —
  if you compensated for the old behavior in downstream code, remove that
  workaround.
- The Python API's `mult_g` and the JSON report's `solution.lambda` are
  unchanged: they keep the Lagrange-multiplier convention, which matches
  cyipopt. `marginal = −λ`; both conventions are now documented side by side
  in [Running Solves](docs/src/cli.md).
- **GAMS equation marginals on `maximizing` models** were inverted in both the
  pip link and the native C link (#272): the `obj_sign` factor already applied
  to the objective value and the variable marginals was missing from the
  multiplier conversion, which is now `pi = −obj_sign · λ`. `minimizing`
  models, objective values, variable marginals and status mapping were never
  affected, and the two links behaved identically, so install method made no
  difference.
- `pyomo_pounce.gradient()` on a `Constraint` target returned
  `d(λ)/d(param)`, which no longer matched `model.dual` once duals were
  converted; it now reports `d(dual)/d(param)`.

### Fixed — ambiguous 2-parameter tuple bounds and NaN bounds (#265)

- `curve_fit` / `curve_fit_minima` / `curve_fit_streaming`: at `n == 2`, a
  length-2 **tuple** of `(lo, hi)` pairs (e.g. `((0, 10), (0, 10))`) was
  silently read as scipy's `(lower, upper)` — the transposed box pinned both
  parameters and still reported `Solve_Succeeded` (#265, the mirror of #260).
  That shape is genuinely ambiguous and now **raises**, naming both unambiguous
  spellings; the list-of-pairs form `[(l0, u0), (l1, u1)]` and the
  tuple-of-arrays form `([l0, l1], [u0, u1])` are unchanged. A bare `None` side
  (`(None, 10.0)`) still means unbounded on that side.
- NaN bounds are now rejected in `minimize` and every `curve_fit` surface
  (previously they slipped past the reversed-bound check — `lb > ub` is `False`
  against `NaN` — and behaved as a silent "no bound"). `None` / ±inf remain the
  supported unbounded spellings.
- A degenerate covariance is no longer silent. A zero-width bound (`lo == hi`,
  pounce's "hold a parameter constant" idiom) fixes a parameter and reports its
  `perr` as `0`; a corner solution with every parameter on an active bound
  reports `pcov = 0` throughout. Both are intended, but previously came back
  with no signal — "infinite confidence in a wrong answer" in #265's words.
  `curve_fit` / `curve_fit_minima` / `curve_fit_streaming` now emit a
  `UserWarning` in each case (naming the pinned parameters for the zero-width
  case), so the perr of 0 reads as the constraint it is, not an estimated
  uncertainty. No numbers change.

### Changed — `dual_diverging_streak` is now **off by default** (#250 follow-up)

- **The dual-divergence guard is opt-in.** It shipped default-on at `15` to bound
  a reported emfl050 bad-warm-start grind. That justification did not survive
  being reproduced: the reported `11.7 s / iterations=0` measurement is
  caller-side JAX compilation (it follows call order, not the guard, in both
  orders and under both settings), and the build predating the guard solves both
  emfl050 instances to the same optimum in the same time.
- **What remained was basin luck, and knife-edge at that.** Across 1284 MINLPLib
  models the guard changes four outcomes, non-monotonically in its own threshold:
  `deb7`/`deb9` reach a better local optimum (104.95 -> 97.56) at *exactly* 15 and
  at no other value tried, while `pooling_rt2stp` turns from `Solve_Succeeded`
  into `Maximum_Iterations_Exceeded` at 10 and 15 only, solving cleanly at 0, 5,
  25 and 40.
- **The effect differs by host in _sign_.** Those `deb7`/`deb9` numbers are
  macOS/FERAL. On the Linux CI runner the same `dual_diverging_streak=15` makes
  `deb7` *worse*: 97.56 with the guard off, 127.87 with it on. It helps on one
  platform and hurts on the other, same source, same fixture. A heuristic whose
  sign depends on the host is not a property of the algorithm, which settles the
  question: it does not belong in the default path.
- **This also bounds what the fallback below can promise.** It guarantees a
  diverted run never returns worse than the best acceptable point *that run
  visited*; it cannot make the diversion no worse than not diverting, because
  that counterfactual solve never happened. On Linux the guard costs `deb7` 30 %
  of its objective and the fallback cannot recover it — 127.87 *is* the best
  acceptable point the diverted run reached.
- **The two sides are not commensurate.** The upside is a better local optimum on
  an already-solved nonconvex problem; the downside is a clean solve becoming a
  failure. A net-positive count on one corpus is not a reason to impose that on
  every user's problem, so the guard stays available and is no longer default.
- Set `dual_diverging_streak=15` to restore the previous behaviour. When enabled,
  the never-worse-off fallback below applies.
- Full-corpus effect of this release's changes, measured against the commit
  preceding them (`f5aea43`) over all 1284 models: **0 regressions**, 4
  improvements (`jit1`, `nvs04`, `heatexch_spec2`, `supplychainp1_030510` all
  move from a limit/failure status to solved), and one objective improvement
  (`st_e35` 21357.40 -> 21355.24, feasible to 1.8e-12).

### Fixed — the dual-divergence guard's diversion can no longer return a worse point (#250 follow-up)

- **The guard's bet is now non-destructive.** `dual_diverging_streak` (added for
  the emfl050 warm-start stall) routes a solve into restoration once the dual
  infeasibility has grown for that many consecutive iterations in an elevated
  regime. That is a bet, and on the MINLPLib corpus it is usually a good one — it
  rescues twice as many models as it harms — but nothing made *losing* it safe.
  POUNCE now records the best acceptable-quality iterate seen anywhere in the
  solve, and hands it back if a diverted run ends up worse. Applies whenever the
  guard is enabled; since the entry above it is no longer enabled by default.
- **Symptom it fixes.** On `autocorr_bern55-06` the guard fires at iteration 23;
  the diverted run reaches the true optimum (`-2304.0000278`, which Ipopt also
  finds) and holds it from iteration 57 to 86, but the dual residual sawtooths
  between `1e-8` and `2e-1` there, so it never strings together the
  `acceptable_iter` consecutive qualifying iterates that would stop the solve.
  It then entered restoration a second time, wandered into a worse basin, and
  returned `-2263.46` — 1.8 % worse, with an overall NLP error of **1.0**
  (feasible, but nowhere near a KKT point) under a "solved to acceptable level"
  status. The better point had already passed the acceptable test; it was
  overwritten only because `store_acceptable_point` keeps the latest rather than
  the best.
- **The firing threshold's meaning is unchanged** (only its default moved, per
  the entry above). Retuning it to spare this model was tried first and rejected:
  every setting that does so (>= 25) also loses the `deb7` / `deb9` / `deb8`
  rescues, which need exactly 15. No value separates the two classes, so the fix
  addresses the consequence rather than the trigger. Note that the rescues
  themselves are not regression-pinned and deliberately so — `deb7`'s response to
  the guard differs by host in sign (see above), so there is no cross-platform
  assertion to make about it.
- **Statuses that carry a fact of their own are preserved.** A restored point
  never relabels `MaxiterExceeded` / `CpuTimeExceeded` / `WallTimeExceeded` /
  `UserRequestedStop` — a caller polling for "did I run out of time" is not told
  "solved to acceptable level" merely because a better point was recoverable.
- The *use* of this bookkeeping is gated on the guard having actually fired (3 of
  500 corpus models), so every solve it never touches is bit-identical. The
  recording itself is deliberately not gated: the guard returns to the driver
  before the recording site on the iteration it fires, so gating the recording
  too would capture nothing at or before the diversion — exactly the case where a
  diversion wrecks a solve immediately. Recording costs one `f64` comparison per
  acceptable iterate, off already-computed quantities, and clones only on an
  improvement.

### Fixed — the best-acceptable fallback no longer trades feasibility for objective (#267)

- **The fallback now ranks by `(feasibility, objective)`, not objective alone.**
  The #250 fallback above chose among recorded acceptable points by scaled
  objective. Every candidate is *bounded* by `acceptable_constr_viol_tol`, but
  being bounded by the band is not the same as not trading feasibility *within*
  it, and the band is a user option. Widen it and a pure-objective argmax has no
  lower bound on the feasibility it will spend: it discards a nearly-feasible
  point for a lower-objective one that is grossly infeasible, then hands it back
  under a `Solved_To_Acceptable_Level` status.
- **Symptom it fixes.** On `autocorr_bern55-06` at
  `dual_diverging_streak=15 acceptable_constr_viol_tol=1e1 acceptable_tol=1e10
  acceptable_dual_inf_tol=1e30 acceptable_compl_inf_tol=1e10`, the guard-on solve
  returned objective `-2307.32` at a constraint violation of **9.94** — below the
  true optimum `-2304.0` precisely because the point is infeasible, and rejected
  by `pounce verify`. At the same tolerances the guard-*off* control returns a
  feasible point (`-2298.57`, violation `1.06e-4`), so the loose band alone is not
  the cause — it only widens what the fallback can exploit. The fix returns a
  feasible point in the guard-on case too.
- **How the ranking works.** Each recorded/returned point carries its unscaled
  max-norm constraint violation — the same quantity the `acceptable_constr_viol_tol`
  gate is defined against. The key is `(feasible_enough, objective)`: a point
  inside the feasibility band beats one outside it outright, and objective decides
  *only among points already inside the band*. `feasible_enough` uses the
  acceptable feasibility band **capped at its upstream default** (`1e-2`), so it
  never gets looser than a normal solve's — a grossly-infeasible low-objective
  iterate is simply not a candidate. At default or tighter tolerances this is
  behaviour-neutral: every recorded point already passed the
  `acceptable_constr_viol_tol` gate, so with that band at or below the cap every
  candidate is feasible-enough and objective alone decides, exactly as before. The
  cap only bites once the user loosens `acceptable_constr_viol_tol` past its
  default.
- **Symptom, resolved.** With the fix the guard-on solve above returns the
  diverted run's own near-optimal endpoint — objective `-2303.9999` at violation
  `1.13e-4`, which `pounce verify` accepts — instead of the infeasible `9.94`
  point a pure-objective ranking restored.
- **Scope.** Latent and config-gated, not a live default-path bug: the guard is
  off by default, and the trade needs `acceptable_constr_viol_tol` widened past
  its default. Every solve the guard never fires on is still bit-identical — the
  read stays gated on `dual_guard_fired`; only the recording cost grows by one
  already-computed norm per acceptable iterate.

### Fixed — false `Infeasible_Problem_Detected` on a feasible, aggressively scaled NLP

- **Rapid infeasibility detection now confirms its own claim before issuing it.**
  The detector fired on a violation gate plus a stationarity *surrogate*,
  `‖Jᵀc‖ / max(1, ‖c‖)`, against an absolute tolerance. That surrogate is not
  scale-invariant: under a row scaling `dc` the numerator carries `dc²` while the
  denominator clamps at 1, so an aggressive scaling drives it toward zero
  regardless of where the iterate actually is.
- **Symptom it fixes.** On Hock–Schittkowski 13 from `x₀ = (1e4, 1e4)` the
  starting Jacobian is ~3e8, gradient-based scaling picks `dc ≈ 3.3e-7`, and the
  surrogate read `5e-14` — far under the `1e-8` tolerance — at a point whose
  constraint violation was **0.51**, whose `‖∇θ‖` was 1.40, and where neither
  bound was active to block descent. POUNCE reported
  `Infeasible_Problem_Detected` on a *feasible* problem with `f* = 1`. It now
  converges to `0.98492872` in 29 iterations, matching Ipopt from the same start
  to 9 significant figures.
- **No tolerance fixes this, so the fix is not a retune.** Measured over 800
  corpus models plus targeted infeasible problems: the scaled surrogate is not
  separable on the targeted cases; measuring it *unscaled* needs a tolerance
  ≥ 1e-2 to fire at all, which introduces new false infeasibility on 3+ corpus
  models while still losing 2 correct detections; and a scale-invariant
  `‖Jᵀc‖ / ‖c‖²` is not separable either. A single absolute threshold on a
  surrogate cannot separate these cases.
- **What it does instead.** The surrogate is kept as a cheap pre-filter, and
  before the verdict is issued POUNCE probes for a materially less-violating
  point nearby — a few bound-clamped steps along `−∇θ`, comparing `θ` to `θ`.
  That is scale-free by construction and needs no calibration. The two regimes
  are far apart: near a genuine infeasible stationary point only ~0.07 % further
  descent exists, whereas at HS13's false verdict one step takes `θ` from 0.51 to
  zero.
- **Why it matters more than a wrong number.** A false *unbounded* verdict is
  loud and a driver can retry it; a false *infeasible* silently prunes a
  branch-and-bound node that may contain the optimum.
- Costs a handful of constraint evaluations, and only on solves that were about
  to terminate anyway. Genuine detection is preserved: across 1284 MINLPLib
  models the infeasible-verdict count is unchanged at 37, with **0** new false
  positives and **0** lost detections, and the sweep is otherwise bit-identical.

### Fixed — unreachable termination certificate on a strongly objective-scaled NLP (#257)

- **The dynamic barrier floor now expresses `compl_inf_tol` in the space μ
  actually lives in.** The floor is
  `min(tol, compl_inf_tol) / (barrier_tol_factor + 1)`, but its two terms are
  enforced in different spaces: `tol` is compared against the *scaled* NLP
  error, while `compl_inf_tol` is compared against the *unscaled*
  complementarity (pounce#173). Feeding the raw `compl_inf_tol` into a
  scaled-space floor put it `1/|obj_scaling_factor|` too high whenever the
  objective was scaled down, so on a strongly deflated objective μ bottomed out
  *above* the level the convergence test required and the strict certificate
  became unreachable — no matter how long the solve ran. The floor now uses
  `compl_inf_tol · |obj_scaling_factor|` (magnitude, so a maximization posed
  via `obj_scaling_factor = -1` is unaffected), falling back to the
  unconverted tolerance if the factor is absent or degenerate.
- **This removes a tolerance inversion**: because a smaller `tol` incidentally
  dragged the floor down, the failure appeared at *looser* tolerances and not
  at the default `1e-8`. Loosening `tol` could cost a user their certificate.
- **Symptom it fixes.** POUNCE would sit exactly on the optimum, with a scaled
  NLP error well under `tol`, unable to certify it; μ-at-floor plus the
  resulting vanishing step then exited
  `Search_Direction_Becomes_Too_Small`, which drivers commonly map onto
  unboundedness. On the branch-and-bound node subproblems discopt generates for
  MINLPLib `jit1` (`obj_scaling_factor = 1e-5`, `tol = 1e-7`) this hit **59 of
  59 node solves**, leaving the MINLP with no incumbent (`status=unknown`)
  unless the driver retried every node with Ipopt. Post-fix all nodes certify,
  POUNCE reproduces Ipopt's node optimum to the last digit
  (`173345.37683089852`), and `jit1` solves to `obj = 173982.61006345798`,
  `gap = 0` in 9 nodes — node-for-node identical to an Ipopt-driven search.
  Pinned by `crates/pounce-cli/tests/issue_257_jit1_node_certificate.rs`
  against `jit1_node.nl`, the failing node captured verbatim.

### Fixed — no spurious `Unbounded` on a bounded, ill-scaled NLP (#248)

- **The divergence guard no longer reports `DivergingIterates` (Ipopt's
  unboundedness verdict) purely on iterate magnitude.** `DivergingIterates`
  maps to the AMPL 300 "unbounded" range, but a large `max_i |x_i|` does not
  by itself prove unboundedness: under severe objective ill-scaling the
  normal-mode IPM can take a large excursion on a problem that is bounded
  below with a finite optimum (MINLPLib `jit1`), and if every variable is
  boxed the feasible region is bounded so unboundedness is structurally
  impossible. Both divergence guards (the running post-accept check and the
  restoration-failure fallback) now surface `DivergingIterates` only when the
  growth is consistent with an unbounded feasible region — some component
  past `diverging_iterates_tol` heading toward a side with no finite bound.
  When every large component is pinned by a finite bound (in particular, all
  variables boxed), the solve continues to its best iterate and returns a
  non-unbounded status (optimal / iteration limit) instead of a spurious
  `Unbounded`.
- **The verdict now also requires the divergence to *persist*.** For a
  genuinely free variable a low `diverging_iterates_tol` (the kind a branch-
  and-bound driver sets to abort runaway nodes) could still trip on `jit1`'s
  transient excursion — `max_i |x_i|` climbs to ~16 then recedes to the
  finite optimum near ~2.9. `DivergingIterates` is now surfaced only once a
  free, over-threshold iterate has kept *growing* for several consecutive
  iterations (a real recession ray grows geometrically; a settling iterate
  does not), or has blown past an absolute runaway backstop (`1e18`, at or
  below the default `1e20` threshold, so the default "fire the instant
  `|x|` crosses the threshold" behaviour is preserved). Verified end-to-end
  on the MINLPLib `jit1` `.nl`: at `diverging_iterates_tol=2` the published
  model (free variables) and a ±100-boxed variant both now reach the finite
  optimum (`obj ≈ 173345`), while a genuinely unbounded-below problem still
  reports `DivergingIterates`. Together these remove a non-rigorous fathom in
  discopt's branch-and-bound, which uses POUNCE as its per-node NLP backend.

### Fixed — no spurious `Unbounded` on an unbounded-box subproblem with a finite optimum (#252)

- **The divergence guard now also checks the objective trajectory, not just
  iterate growth.** #248's structural + growth-persistence gates cleared
  `jit1`'s boxed / free-variable *root* relaxation, but its spatial-B&B *node*
  subproblems carry variables with `ub = +∞` (integer-tightened boxes). Those
  pass the structural "heads toward an unbounded side" check, and under the
  linear tail's `1e7`-scale ill-scaling the transient excursion climbs past
  enough doublings to satisfy the growth streak too — so every one of jit1's
  59 nodes was reported `DivergingIterates` (an UNBOUNDED false negative;
  cyipopt/Ipopt find each node's finite optimum). A large, still-growing
  `max_i |x_i|` toward an open box side is *not* enough: the guard now requires
  the divergence to look like a genuine recession ray, whose per-step objective
  drop *keeps up* as `|x|` grows geometrically. An excursion converging to a
  finite optimum lowers `f` for a few steps but with a per-step drop that
  *decelerates* toward zero (settling onto the finite floor); it therefore no
  longer accumulates the streak and the solve converges to the optimum instead
  of reporting `Unbounded`. A genuinely unbounded-below objective
  (`f → −∞`), whose per-step drop grows without bound, still reports
  `DivergingIterates`, and the absolute runaway backstop (`1e18`) is unchanged.
  This lets discopt's branch-and-bound retire the load-bearing
  cyipopt-retry-on-UNBOUNDED guard it kept for jit1's nodes.

### Documentation — `PathFollower` gets a book page, figures, and torch docstrings

- **New book page `docs/src/path-following.md`**, linked from the Python API
  page and the TOC. `PathFollower` / `inverse_map_rhs` previously appeared in
  the book exactly once — a single cell in the JAX↔PyTorch parity table — so
  the whole path-following surface was effectively invisible to anyone
  reading the manual rather than the source. The page covers parameter
  continuation (`follow`), pseudo-arclength continuation through folds
  (`trace_arclength`), the inverse/uncertainty map, the `PathTrace` fields, a
  when-to-use-which table, and the equality-only / scalar-θ scope limits. Its
  numbers are measured, not asserted: the `monitor_tol` sweep table (10 → 3
  solves as the tolerance loosens from `1e-6` to `2e-2`, against 12 for
  re-solve-every-step) was generated by running the examples. The page
  carries the fold figure as `docs/src/images/path-following-fold.svg`,
  regenerated by the new `scripts/make-docs-figures.py` so it stays
  reproducible rather than being a pasted screenshot.
- **`docs/src/sensitivity.md` now points forward to it.** The sensitivity page
  ends at a single first-order perturbation off one factor, which is exactly
  where a reader starts wondering about repeated steps, active-set changes,
  and singular `∂x*/∂θ` — it said nothing about the machinery that handles
  them.
- **Notebook 14 is now illustrated** — four figures where it previously had
  only printed numbers: the traced parameter and solution paths coloured by
  path parameter; the `monitor_tol` accuracy-vs-solves trade-off curve
  against the re-solve-every-step baseline; the cubic fold S-curve with both
  turning points marked and `θ(s)` shown reversing (the reason parameter
  continuation stalls there); and the inverse map's recovered input path
  against its analytic value plus the loop-closure check.
- **`pounce.torch.PathFollower` is documented directly** rather than by
  reference. The class, `follow`, `trace_arclength`, and `PathTrace` had one-
  and two-line stubs pointing at the JAX versions, so `help()` on the torch
  class listed none of its nine constructor kwargs. Also restored
  `PathFollower` / `inverse_map_rhs` to the `pounce.jax` package docstring,
  which enumerated the frontend's layers but omitted them.

### Added — the last FERAL numerics env knob is now a registered option (#235)

- **`POUNCE_FERAL_MIN_PAR_FLOPS` is now the `feral_min_par_flops` option**,
  with the env var kept as a fallback — matching the pattern the other
  `feral_*` knobs already follow. FERAL's parallel-dispatch flop gate
  (feral#19) was previously reachable only through the environment: it was
  not a registered option, could not be set per solve, and left no trace in
  the solve report. It is registered as a lower-bounded
  **number** option (not integer) because the gate is a `u64` — an `i32`
  cannot hold large flop counts or the `u64::MAX` reject-all sentinel; the
  value is cast to `u64` with saturation. The four other numerics knobs the
  audit in #235 flagged (`feral_refine`, `feral_fma`, `feral_cascade_break`,
  `feral_singular_pivot_floor`) were already registered options.

### Added — the solve report records active environment overrides (#235)

- **The JSON solve report now captures solve-affecting environment
  variables** in `fair_metadata.environment`. A run that differs because
  `POUNCE_FERAL_PIVTOL` (or any `POUNCE_FERAL_*` knob, or the legacy
  `FERAL_PIVTOL` / `FERAL_PARALLEL`) was exported in a shell profile now
  says so, instead of differing silently between machines — closing the
  reproducibility gap the report is built to serve. Each entry is a
  `{ name, value }` pair; the block is omitted when nothing is set (the
  common case), and the `POUNCE_DBG_*` debug gates are deliberately not
  captured (they never change the result). Additive to
  `pounce.solve-report/v1` — older readers ignore the new field, and older
  reports deserialize unchanged.

### Changed — restoration debug gate reconciled onto one spelling (#235)

- **`POUNCE_DBG_RESTO` now also enables the augmented restoration-system
  stats** that previously answered only to `POUNCE_RESTO_DBG`. The two live
  spellings gated different output in different crates, so guessing the
  wrong one produced silence with no way to tell a spelling mistake from a
  code path that did not run. `POUNCE_DBG_RESTO` is the canonical name (the
  `POUNCE_DBG_*` convention every other gate follows); `POUNCE_RESTO_DBG` is
  retained as a deprecated alias.

### Docs — the previously source-only environment gates are documented (#235)

- **A new [Environment overrides](docs/src/options.md) section** tables the
  `POUNCE_FERAL_*` numerics fallbacks (each mapped to its option) and every
  `POUNCE_DBG_*` / diagnostic gate, noting which take a value, which tracing
  target each needs under `RUST_LOG`, and which print straight to stderr.
  `troubleshooting.md` points to it from the logging recipes.

### Fixed — failed initialization solves no longer poison variable values (Pyomo, #230)

- **A failed solve anywhere in the initialization pipeline now leaves
  variable values exactly as they were.** Previously `block_initialize`
  delegated its block loop to Pyomo's `solve_strongly_connected_components`,
  which loads whatever the subsystem solver returns: a diverged block with a
  warning-status result (infeasible, iteration limit) was written into
  `Var.value` and the loop continued, reporting success. On a 1525-equation
  air-separation model this turned a solvable steady state into a
  3000-iteration stall — the model's own build point solved fine, but the
  "initialized" point, containing a diverged 1500-variable block iterate,
  did not. The block loop is now in this module: each block's verdict is
  checked before its values are kept, a failed block restores its seed
  values, the loop stops there (later blocks typically feed on the failed
  one), and the failure names the block. `project_to_feasible` had
  the same defect — a diverged projection loaded its iterate, making the
  pipeline's "continuing with the unrepaired point" warning untrue — and now
  restores the pre-projection point on any non-optimal termination.

### Added — automatic specification repair: `block_repair_plan`, and repair inside `initialize`/`block_initialize` (Pyomo, #228)

- **A structurally broken specification now initializes anyway, and the report
  says what was repaired.** Some specifications are wrong by structure, not by
  starting point: hold every flow control of a distillation column at steady
  state and the drum levels are undetermined while the holdup balances turn
  redundant — square by count, singular in structure, unsolvable from any
  start. `pyomo_pounce.block_repair_plan(model, decision_candidates=...)`
  plans a valid specification from the variables you would like held: the
  subset a square system can hold (`decisions`), the ones the equalities
  claim and solve for instead (`pruned`, provably the minimum number), and
  the variables nothing can determine (`pinned`) — held at values of your
  choosing. Like `block_analyze` it is a plan, not an action: nothing fixed,
  read, or written, no values needed, and the defect lists
  (`loose_variables`, `redundant_constraints`) come back as uncapped
  component objects.
- **Pins are identified automatically, with no user input.** A variable is
  pinned when every one of its incidence edges is provably unusable: an
  equation `0 == f/g` cannot determine a variable appearing only in the
  denominator `g` (its sensitivity there vanishes at every solution), which
  is exactly the shape substituting `d/dt = 0` into a dynamic balance
  produces. Dropping those edges makes the identification canonical — on the
  569-equation double-column example the four drum levels come out
  identically under every matching order, where raw-incidence matching left
  an order-dependent (and numerically singular) choice.
- `initialize` and `block_initialize` run the same check on their `decisions`
  automatically (`repair="auto"`, the default). A square specification is
  used exactly as given (the shipped behavior); a broken one is repaired,
  with `report.repair` recording the plan (None when nothing was needed)
  and `n_pinned` counting pins separately from decisions. The repair is
  call-scoped like the decisions themselves, so it never alters the model's
  own specification. A pruned decision no longer needs a value (it gets
  solved for); a valueless pinned variable gets a bounds-aware seed that is
  never exactly zero (a pin lives in denominators). `repair="off"` is the
  strict path: decisions held exactly as given, non-square specifications
  reported instead of repaired.
- Pruning ties are deterministic and user-steerable: among candidates,
  earlier-listed ones are preferentially kept, so the `decision_candidates`
  listing order is an implicit priority. Fixing a variable removes it from
  the plan entirely.

### Added — `pyomo_pounce.block_analyze`, the analysis half of `block_initialize` on its own (Pyomo, #224)

- **The Dulmage-Mendelsohn partition is now available without solving
  anything.** `block_initialize` already computes the full partition of the
  equality system, but a caller could only see it through the initialization
  report, where the underconstrained/overconstrained lists are capped,
  name-only, and bundled behind the fill and block solves. `block_analyze(model,
  decisions=[...])` runs the same decision handling and the same decomposition
  and returns a `BlockAnalysisReport` carrying the **full** partition as the
  component objects themselves: the underconstrained and overconstrained
  subsystems (variables and constraints on both sides), the square part, and
  its block-triangular calculation order, with nothing capped. Convenience
  counts (`n_extra_degrees_of_freedom`, `n_extra_specifications`) say how far
  from square the system is.
- Analysis is purely structural, so decisions do not need values here (they do
  in `block_initialize`, whose solve must hold them at a concrete point), no
  values are read or written, and fixed flags are restored on the way out.
  This makes it a safe first pass on a badly specified model: diagnose, decide
  what to specify, then call `initialize`/`block_initialize` to do the work.
- `block_initialize` now delegates its partition step to `block_analyze`; its
  behavior and report are unchanged.

### Fixed — false `Solve_Succeeded` behind an extreme objective scale (#200)

- **POUNCE no longer certifies optimality at a point that is not a minimum.**
  Gradient-based objective scaling picks `df = nlp_scaling_max_gradient /
  max‖∇f‖`, floored at `nlp_scaling_min_value = 1e-8`. On a flat quartic the
  initial gradient is enormous, `df` pins at that floor, and — because a
  quartic's gradient vanishes *cubically* toward its minimum while `df` stays
  fixed — the scaled convergence test trips roughly 30% of the way in. The
  solver reported `Solve_Succeeded` at objective **248.88** (`quartc`) and
  **39.36** (`dqrtic`) when the true minimum of each is ~0, with an unscaled
  dual infeasibility of 0.84.

  | problem | before | after |
  |---|---|---|
  | `quartc` | `Solve_Succeeded`, obj 248.88 | `Solve_Succeeded`, obj **8.8e-07** |
  | `dqrtic` | `Solve_Succeeded`, obj 39.36 | `Solve_Succeeded`, obj **7.0e-07** |
  | `penalty1` | `Solve_Succeeded`, obj 6.44 | `Solve_Succeeded`, obj **0.0097** (true) |

- **This is a deliberate deviation from upstream Ipopt**, which was verified
  here to have the identical failure (`ipopt quartc.nl` → `Optimal Solution
  Found` at 248.88). The new `obj_scale_certificate_threshold` option (default
  `1e-4`, set `0` to disable) restores bit-for-bit upstream behaviour.

- **The mechanism tests the stop rather than predicting it.** When the objective
  scale is below the threshold and the unscaled KKT error is still above
  `acceptable_tol`, POUNCE refuses to terminate and keeps iterating: a constant
  objective scale cancels out of the Newton step and the line-search tests are
  scale-invariant, so the run follows the trajectory an unscaled run would take.
  If that reaches a better point, the stop was false. If it achieves nothing,
  the refused point is restored and reported with the status it would originally
  have had — so the mechanism is **never worse than not having it, by
  construction**, and no benchmark-fitted constant is needed to tell the two
  cases apart. Whether a stop is genuinely false cannot be read off the
  residuals: `meyer3` sits at the same 1e-8 scale floor as `quartc` while being
  genuinely converged. Measured across the 733-problem Vanderbei suite, all 16
  problems eligible for the veto keep their original status.

- **Fixed alongside: the console report hid the discrepancy.** The NLP summary
  passed the *scaled* residual to both the `(scaled)` and `(unscaled)` columns,
  so `quartc` printed dual infeasibility `8.38e-09` twice when the unscaled
  value is `8.38e-01` — a user auditing the suspicious certificate was shown a
  report that agreed with it. The unscaled statistics were already computed and
  already surfaced through the Python bindings; only the console dropped them.
  Upstream Ipopt prints these correctly, so this was a porting defect rather
  than a deviation.

### Fixed — a diverged conic solve could report `Optimal` with a `NaN` solution (#222)

- **`solve_socp_ipm` could return `QpStatus::Optimal` alongside `x = [NaN, NaN]`.**
  A caller checking the status — the documented way to know an answer is usable
  — was handed a garbage solution with no indication anything had gone wrong.
  Observed on the direct symmetric driver (`use_hsde: false`) on PSD programs;
  the underlying flaw was shared by both drivers.

- **Cause: `inf_norm` swallowed `NaN`.** `f64::max` is specified to *ignore*
  `NaN`, so the natural `fold(0.0, |m, x| m.max(x.abs()))` reports the ∞-norm of
  an all-`NaN` vector as a perfect `0.0`. Every convergence test in both drivers
  compares such a norm against `tol`, so a fully diverged iterate read as
  converged: on the reported instance the iterate went non-finite at iteration
  31 and the residuals computed from it came back `pinf = dinf = res = 0`, so
  `res < tol` passed and the solve declared success. `inf_norm` now
  short-circuits on `NaN`, making every such comparison correctly false.

- Two further guards, so the guarantee does not rest on one primitive: each
  driver now breaks with `NumericalFailure` as soon as its iterate goes
  non-finite, and a final pass demotes any `Optimal` / `OptimalInaccurate`
  verdict that is not backed by a finite solution and objective. Across a
  31,920-solve randomized sweep over both drivers, **no solve now reports
  success with a non-finite solution** (previously 32).

- The divergence itself is left as-is: the direct driver stalling or diverging
  on a degenerate face is known behaviour and precisely why the homogeneous
  self-dual embedding is the default (`use_hsde: true`, see `sos_opts`). HSDE
  solves the reported instance correctly. What was wrong was reporting that
  divergence as success, not the divergence.

- The randomized differential suite added in #221 asserted this property of the
  HSDE driver only, and had to skip the direct driver's cases as
  `reference-unusable`; the skip is gone and both drivers are now asserted.

### Added — weak-activity detection for QP sensitivity (#219)

- **`QpSensitivity` can now report whether its own precondition holds.** The
  first-order predictor `parametric_step` is exact only while the active set is
  unchanged, but nothing on the object let a caller check that. Two new
  properties expose what the object already knew: `active_indices` — which
  inequality rows and variable bounds are in the active set, by identity rather
  than the count implied by `kkt_dim` — and `weakly_active_indices`, the
  constraints at which **strict complementarity fails**. Both return an
  `ActiveSet` (`.inequalities` indexes rows of `G`, `.bounds` indexes
  variables), also exported at the package root.
- A weakly active constraint is binding in the primal while carrying a
  negligible multiplier. Classical post-optimal sensitivity (Fiacco) assumes
  this away; where it happens the perturbation changes the active set, so
  `dx/db` is a genuine *one-sided* derivative and the other direction has a
  different, equally valid value. Nothing previously returned was wrong — both
  branches are real derivatives — but a caller could not tell the situation
  apart from an ordinary one.
- The screen is deliberately tolerance-invariant, which `kkt_dim` is not. On the
  reported QP the two branches of `dx/db` are 33% apart and which one is
  reported turns on the solver's `tol`, an unrelated setting; `kkt_dim` flips
  4 → 3 across that sweep while the geometry never changes. The new flag stays
  on throughout, because it tests the multiplier and the slack *together* —
  at a degenerate optimum both collapse at ~`√tol`, while strict
  complementarity keeps one of them bounded away from zero.

### Fixed — the Lasserre/SOS hierarchy now converges on constrained programs (#218)

- **`sos_minimize` returns a real bound where it used to return `nan`.** On the
  reported benchmark (Lasserre, *SIAM J. Optim.* 11(3):796–817, Example 5) the
  hierarchy now tightens to the global optimum instead of stalling:

  | order | before | after |
  |---|---|---|
  | 2 | `optimal` `−7.000` (trivial box bound) | `optimal` `−7.000` |
  | 3 | `iteration_limit` `nan` | `optimal` `−6.667` |
  | 4 | `iteration_limit` `nan` | `optimal` `−5.5080139` (certified), **exact**, minimizer `(2.3295, 3.1785)` |

  True global minimum `−5.5080132716` at `(2.3295, 3.1783)`, verified
  independently here by a 400-start SLSQP sweep. Order 4 is certified exact by
  flat truncation, so the minimizer comes back with the bound, and the reported
  value is a *rigorous* lower bound (see below) rather than merely an accurate
  one.

- **Root cause: Mehrotra's centering parameter inverts on a degenerate face.**
  `σ = (μ_aff/μ)³` infers centering from how far the affine direction could
  travel. On a degenerate face that direction looks excellent while pointing
  almost straight out of the cone, so `σ` collapses toward zero — nearly no
  centering — exactly where centering is the only thing that helps. Order 4 of
  the reported problem pinned `σ` at 0.0218 while the step fell
  `4.0e-1 → 2.1e-2 → 9.7e-4 → … → 1e-281`, throttled by the PSD slack block
  alone, with `μ` frozen at 2.6e-3 and residuals at 4.2e-4 — nowhere near
  converged, simply stuck on the boundary. The corrector is now recomputed under
  an escalating `σ` when its step collapses, each retry one extra back-solve
  through the factorization already in hand. This is the PSD-cone counterpart of
  what the Gondzio correctors do on the orthant, where they are confined because
  a PSD block's complementarity product needs Jordan-algebra machinery.

  The same fix carries the **NETLIB GEN family** (`gen`, `gen1`, `gen4`) and
  `pilot87` from `Maximum_Iterations_Exceeded` to `Solved_To_Acceptable_Level`.

- **Regularization no longer ratchets on a conditioning symptom.** When
  iterative refinement could not reach its tolerance, the driver escalated both
  the `(z,z)` dynamic regularization *and* `δ_c`, the equality-block
  regularization — treating a cone-conditioning symptom as an
  equality-Jacobian rank defect. The KKT is inherently ill-conditioned in the
  `μ→0` endgame (the NT scaling's condition number blows up by design), so this
  fired there on healthy solves and drove `δ_c` to its `1e-1` ceiling, biasing
  the equality residual by `~δ_c·‖dy‖` and flooring `pres` permanently. Order 3
  of the reported problem stood at `pres` 8.6e-9 ✓, `dres` 1.2e-10 ✓, `gap`
  1.7e-8 — one step from converging — when the escalation pushed `pres` to
  2.7e-8, where it stayed. Inertia was correct at every try throughout, so no
  rank defect was ever present. Only the `(z,z)` regularization escalates on
  this signal now.

- **The bound is now certified, not merely converged.** A converged SDP reports
  a `γ` that is *accurate* but need not be a lower **bound**: on the reported
  problem at order 4 the raw value came back `2.2e-7` **above** the true
  minimum, which is the one genuinely unsound failure mode for this API and one
  no solver tolerance removes. `sos_minimize` now measures the miss instead of
  trusting the solve — it projects each Gram block onto the PSD cone, evaluates
  the residual `e` of the coefficient-matching system there, and reports
  `γ − Σ_α|e_α|`. Since every other term of the Putinar identity is nonnegative
  on the feasible set and `|u^α| ≤ 1` on the normalized box, that value is a
  true lower bound however the solve went. Order 4 moves from `−5.508013056`
  (invalid) to `−5.508013930` (valid, and within 1e-6 of the optimum). Costs one
  eigendecomposition per block — no extra solve.

  A new `certified` flag (Rust `SosSolution::certified`, Python
  `SosResult.certified`) reports whether this held. It requires the feasible set
  to lie in a box readable off the constraints — either `x ≥ l` / `x ≤ u` pairs
  or the `c − a·x² ≥ 0` idiom — and is `false` on an unbounded domain, where no
  finite correction exists (a residual with a negative leading coefficient is
  unbounded below). Adding explicit box constraints upgrades such a problem.

- **`tol` and `max_iter` are exposed** on `sos_minimize` (Python) and via
  `sos_minimize_opts` / the now-public `sos_opts` (Rust) — the escape hatch the
  report asked for, for a relaxation that will not converge. Loosening `tol`
  buys a weaker bound, never an invalid one: certification measures the actual
  residual rather than assuming convergence, so validity is preserved across
  the whole range.

- **A coarser bound is no longer discarded.** If the requested order does not
  converge, `sos_minimize` falls back through successively coarser orders and
  reports the first that does, via a new `order` field on the result (Rust
  `SosSolution::order`, Python `SosResult.order`) identifying which relaxation
  produced the bound. A lower-order bound is a valid bound on the same problem,
  so returning nothing in its place threw away a certificate already computed.

- **`PolyProblem::equilibrated` now normalizes the domain, not just the
  coefficients.** When box constraints pin a variable's range it is mapped onto
  `[−1, 1]` before the SDP is assembled. Coefficient equilibration (#124) left
  the domain alone, so a wide box made the moment matrix span decades by itself
  — over `x₁ ∈ [0,3]` the degree-8 moments span `3⁸ ≈ 6561` against 1, which no
  coefficient scaling touches. The change of variables is value- and
  minimizer-preserving, and recovered minimizers are mapped back to the caller's
  coordinates.

- Verified against the committed benchmark baselines: **371 NETLIB LPs** (4
  improvements above, no regressions — the three objective deltas are all on
  runs whose status is unchanged and non-converged or infeasible, where the
  objective is not a meaningful output) and **138 Maros–Mészáros QPs**
  (byte-identical statuses and objectives). The changed driver serves the
  symmetric cones only — exponential/power route to the separate non-symmetric
  driver, and the CBLIB tier is bit-identical to an unmodified tree — so SOC and
  PSD are covered by a new randomized differential suite
  (`tests/conic_hsde_vs_direct.rs`) that checks the HSDE driver against the
  untouched direct driver over **318 generated instances** with planted Slater
  points and compact feasible sets: no disagreement on any optimal value, no
  instance solved by the direct driver that HSDE failed, and 36 that only HSDE
  solved.

### Added — thread-scoped iteration capture from Rust (pounce-rs)

- **A Rust consumer embedding POUNCE can now record a solve's iteration
  trajectory with no direct `tracing`/`tracing-subscriber` dependencies and
  without touching the global subscriber.** `pounce-observability` gains
  thread-scoped helpers that bundle the collector-layer install:
  `with_iter_capture(|| nlp.solve())` runs a closure with capture active and
  returns its result alongside the recorded `IterRecord`s;
  `ScopedIterCapture::start()`/`.finish()` is the guard-shaped equivalent for
  solves that don't fit in a closure; and `collector_scope()` installs just
  the collector for the `IpoptApplication` path
  (`enable_iter_history()` + `statistics().iterations`), where the driver
  manages its own capture guard. All three activate the collector for the
  scope's lifetime and never touch the global subscriber: when
  `init_subscriber` already owns the global (which carries the collector)
  the scope is a no-op and capture composes with console/JSON logging;
  otherwise a collector-only thread-default is installed, which shadows the
  host's own subscriber on that thread for the duration — scope it tightly
  around the solve. Restoration sub-solve exclusion and nested /
  sequential capture semantics are unchanged, and the driver now propagates
  its iteration-history records to any enclosing capture on finish (new
  `extend_active_capture`), so wrapping a solve that has iteration history
  enabled in `with_iter_capture` yields the trajectory in both places
  instead of an empty outer buffer. `pounce-rs` re-exports the
  helpers plus `IterRecord`, `SolveStatistics`, `IterCaptureGuard`, and
  `init_subscriber` (and the whole `pounce_observability` crate), so the
  facade alone covers both iteration capture and console logging;
  `with_iter_capture`, `collector_scope`, `IterRecord`, and
  `SolveStatistics` join the prelude.

### Changed — richer `Nlp::solve()` result (pounce-rs)

- **`builder::Solution` now carries the full per-solve picture.** New fields:
  `g` (constraint values at the solution), `z_l`/`z_u` (bound multipliers),
  and `stats: SolveStatistics` — wall time
  (`stats.total_wallclock_time_secs`), `iteration_count`, evaluation counts,
  final scaled and unscaled infeasibilities, final barrier `mu`, and
  restoration counters, all filled on every solve with no new bookkeeping
  (the driver already computed them; the builder now reads
  `app.statistics()`). A new `.capture_iterations()` builder flag opts into
  the per-iteration trajectory (`stats.iterations`, one `IterRecord` per
  Newton iteration) by activating the thread-scoped collector around the
  solve — a no-op composing with the logs when `init_subscriber` owns the
  global subscriber, a host-subscriber-shadowing thread-default install
  otherwise. Only the interior-point engine emits the per-iteration event
  (an active-set SQP solve leaves `stats.iterations` empty while
  `iteration_count` still counts). The field
  additions are breaking only for code that exhaustively destructures
  `Solution` (pre-1.0).

### Changed — pyomo-pounce streams the engine's log under `tee=True`

- **`SolverFactory('pounce').solve(m, tee=True)` now streams the engine's own
  log — banner, problem statistics, iteration table, and end-of-run summary —
  live to `sys.stdout`, including in Jupyter.** The ~300-line Python
  reproduction of the CLI's blocks is gone: the solver core emits them (see
  below) and pyomo-pounce tails fd 1 to `sys.stdout` on a worker thread, so a
  long solve shows its iteration table as it runs rather than as one block at
  the end. The results object regained `solver.name` and the objective
  bounds, and `solver.time` now measures the solve alone (excluding stream and
  decode). Requires `pounce-solver >= 0.9.0`.

### Changed — the solver core emits its own console log

- **The problem-statistics and end-of-run summary blocks are now emitted by
  the solver core (`pounce-algorithm`), gated on `print_level`,** instead of
  by the CLI. Every frontend — the CLI, the Python bindings, and the C
  interface — gets the identical Ipopt-style log at `print_level >= 1`
  (`print_level 0` is silent). The console printers moved from `pounce-cli`
  to the shared `pounce-solve-report::console` crate as the single source of
  truth, and `IpoptNlp::eval_counts()` drains the NLP's per-evaluation
  counters into `SolveStatistics` so the summary's tallies come from the
  solver's own tracking. `pounce.print_banner()` is exposed for in-process
  frontends that print the up-front banner themselves.
- CLI stdout is byte-identical except: the summary's objective/gradient/
  constraint/Jacobian evaluation counts now report the solver's true count
  (one lower than before, which had included a frontend-side evaluation), and
  `print_level 0` now suppresses the statistics/summary blocks too (it
  previously silenced only the per-iteration table).

### Added — parameter covariance for estimation problems (Pyomo)

- **Parameter covariance for estimation problems, from one solve
  (Pyomo).** Declare the fitted variables (`declare_fitted(m.A, m.k)`,
  varargs, they stay free) and the residual container (`declare_residual(m.r)`,
  optional `group=` strings for heteroscedastic noise groups), solve
  ordinarily with `SolverFactory('pounce')`, then
  `pyomo_pounce.covariance(m)` returns the asymptotic covariance with
  no further information: `cov[m.A, m.k]`, `cov.std_err[m.A]`,
  `cov.correlation[m.A, m.k]`, `cov.matrix`, per-group `cov.sigma_sq`,
  and `cov.eigen()` for identifiability diagnosis. Multiple noise
  groups switch to the heteroscedastic sandwich covariance. Known
  variance (`sigma_sq=`, scalar or per-group) and a bare data count
  (`n_data=`) remain as alternatives to declared residuals. The
  objective must be a plain sum of squared residuals (the solve warns
  when the declared residuals do not reproduce it); the scaling
  (cov = 2 sigma^2 times the parameter block of the inverse KKT
  matrix) is pinned against the analytical linear-regression
  covariance in `tests/test_covariance.py`. All declarations also have
  explicit call-time forms on `solve()` (`sens_params=`, `fitted=`,
  `residuals=`). `covariance(..., hessian="gauss-newton")` reports the
  expected-information (Gauss-Newton) form instead of the default
  `hessian="lagrangian"` observed-information form (the exact reduced
  Hessian of the Lagrangian), from the same
  backsolves; the two agree for linear fits, and Gauss-Newton is
  structurally positive semidefinite and matches the scipy /
  `pounce.curve_fit` convention. A fitted parameter on an active bound
  is projected out (matching `curve_fit`): zero variance, covariance
  conditional on the bound, correlation entries reported as 0, plus the
  existing warning.

### Changed

- **Rust 2024 edition + resolver v3** (#204). The workspace now compiles on
  the Rust 2024 edition with Cargo's v3 (MSRV-aware) dependency resolver.
  This is a build-time change only: no public API, CLI, wheel, or numerical
  behavior changes. Building pounce from source now requires Rust ≥ 1.85 (the
  first toolchain to ship the 2024 edition); users installing the PyPI wheels
  or crates.io releases are unaffected. The migration is the mechanical
  `cargo fix --edition` output (`unsafe(no_mangle)` attributes, explicit
  `unsafe {}` blocks in `unsafe fn` bodies, closure captures) plus the 2024
  rustfmt style-edition reformatting; all `tail-expr-drop-order` sites were
  reviewed and are behavior-preserving (`RefCell`/`Mutex` guards dropped at
  end of scope with no observer of the relative order).

### Fixed — `solve_ivp(mass=M)` now projects inconsistent initial conditions (#215)

- **`pounce.ode.solve_ivp` with a singular mass matrix now projects `y0` onto
  the algebraic manifold before integrating, matching `solve_dae`.** Given a
  singular `M`, the mass path is an index-1 DAE, but it previously returned the
  user's `y0` verbatim in `res.y[:, 0]` even when its algebraic components
  violated `0 = f` — silently, with `res.success == True`. Since the whole
  point of the algebraic variables is that they are *determined* by the
  differential ones, a rough guess for them is the normal case, and the first
  column of the trajectory was left off the solution manifold. It now runs the
  same `consistent_initial_conditions` projection (the IDA `IDA_YA_YDP_INIT`
  computation) that `solve_dae(consistent="project")` already used, so both
  entry points to the index-1 DAE math agree. A new `consistent=` keyword
  (`"project"`, the default, or `"assume"`) opts out for callers relying on
  `res.y[:, 0]` echoing an already-consistent input. Non-singular (plain ODE)
  masses are unaffected.

### Added — opt-in on-manifold output projection for DAEs (#216)

- **`pounce.ode.solve_ivp` and `solve_dae` gain `project_output=False`.** Radau
  IIA is stiffly accurate, so a singular-mass DAE satisfies its algebraic
  constraints to round-off at every accepted step — but the dense-output
  polynomial only *interpolates* the constraint between steps, so intermediate
  `res.sol(t)` / `t_eval` points can sit slightly off the manifold. With
  `project_output=True` the algebraic components of each requested output point
  are Newton-polished back onto `0 = f_alg` (differential components held
  fixed), reusing the index-1 differential/algebraic split. It changes only
  what the caller reads — never the trajectory, step sequence, or error
  control. **Off by default, and skipped automatically when the algebraic rows
  are affine:** a linear conservation law (`sum(x) = 1`, atom / charge / site
  balance) is reproduced *exactly* by the degree-3 collocation output — the
  constraint cubic has four roots and is identically zero — so projection buys
  nothing there and the output is returned bit-for-bit unchanged. It matters
  only for a nonlinear algebraic constraint whose absolute interpolated
  residual is large enough to care about. New diagnostic
  `python/examples/dae_manifold_gap.py` (`manifold_gap()`) measures the gap on
  an arbitrary DAE, and notebook `27_dae_manifold_projection.ipynb` walks
  through both knobs.

## [0.8.0] - 2026-07-11

### Added — declared-parameter sensitivity for Pyomo (Python / Pyomo)

- **`pyomo_pounce` sensitivity interface.** Declare parameters while
  building the model (`declare_sens_param(m.p)` — a flag, no perturbed
  values), solve normally with `SolverFactory('pounce')`, then query:
  `gradient(m.x, wrt=m.p)` (exact dx*/dp; equality constraints give
  their multiplier's derivative), container/Jacobian access via the
  `Gradient` object, and `estimate(m, [(m.p, value)])` for
  first-order perturbed-solution estimates with bound clamping and an
  active-set warning. When declarations are present the solve runs
  in-process through the `pounce.Solver` session (`read_nl` + callback
  bridge) and keeps the converged KKT factorization; models without
  declarations use the CLI path unchanged.
- **Python `Solver` session additions**: `parametric_step_full`
  (full KKT-space step, exposing multiplier sensitivities) and
  `multiplier_rows` (map constraint indices to their `y_c` rows), with
  matching Rust methods `Solver::parametric_step_full` /
  `Solver::g_multiplier_rows` in `pounce-sensitivity`.

### Added — structure-aware KKT hooks (#180)

- **Caller-supplied KKT ordering** (item 1). A structure-aware presolve can
  now hand pounce a precomputed fill-reducing permutation for the KKT linear
  solver — a block-triangular / Schur ordering (Parker, Garcia & Bent,
  arXiv:2602.17968) or a tearing ordering from equation-oriented
  decomposition — that the built-in AMD/METIS pass cannot derive.
  Python: `Problem.set_ordering(perm)` / `get_ordering()` / `clear_ordering()`;
  Rust: `IpoptApplication::set_external_ordering(perm)`. `perm` is a 0-based
  new-to-old permutation whose length equals the augmented KKT dimension;
  FERAL validates it as a bijection and fails the factorization (never a wrong
  answer) on a bad permutation. Maps to FERAL's new `OrderingMethod::External`
  (feral#107); honored by the default FERAL backend only.
- **Per-solve linear-algebra / callback timing** (item 3). `Problem.solve`'s
  `info` dict now carries `info["wall_time"]` and an `info["timing"]`
  breakdown (overall total, the linear-algebra factorization-vs-back-solve
  split, and the per-callback objective / gradient / constraint / Jacobian /
  Lagrangian-Hessian eval time); `pounce.minimize` mirrors these as
  `res.wall_time` / `res.timing`. Lets a caller attribute a reduced-space
  solve's runtime (e.g. densified-Hessian eval cost) directly. The detailed
  breakdown is opt-in via `timing_statistics="yes"` (see #190 under Fixed);
  without it, `wall_time` / `overall_alg` are populated and the per-subsystem
  entries read `0.0`.
- **Block-triangular / Schur KKT solve** (item 2). A structure-aware presolve
  can hand pounce the reducible block of the KKT system; that block is
  Schur-complemented out and only the two diagonal blocks are factorized, with
  full-system inertia recovered a priori via Sylvester's law (Parker, Garcia &
  Bent, arXiv:2602.17968). Python: `Problem.set_kkt_schur_block(indices)` /
  `get_kkt_schur_block()` / `clear_kkt_schur_block()`; Rust:
  `IpoptApplication::set_kkt_schur_block(indices)`. `indices` are KKT-space
  (`x, slack, eq-dual, ineq-dual` block order). The Schur solver
  (`FeralSchurSolver` + `SchurAugSystemSolver`) uses only feral's stable
  factor/solve, and falls back to the standard full-space solver transparently
  when the partition is unsuitable (too large a fraction, malformed, or a
  singular diagonal block), so a stray hook never breaks a solve. Beneficial
  only when the Schur block is much smaller than the eliminated block; honored
  on the default feral + exact-Hessian path.

### Changed

- **feral 0.12.0 → 0.14.0.** 0.13.0 adds `OrderingMethod::External(Vec<usize>)`
  (feral#107), which backs the caller-supplied KKT ordering hook (#180); the
  enum is no longer `Copy` (the `External` arm carries a heap permutation), so
  pounce clones it where a copy was previously implicit. 0.14.0 is a pure-perf
  release targeting the IPM warm-refactor workload (one KKT pattern, thousands
  of factorizations): it splits the symbolic ordering races so only the winning
  candidate pays the expensive tail (feral#127), reuses the permute cache on the
  parallel numeric driver so a warm re-factor scatters in O(nnz) instead of
  rebuilding and re-sorting triplets every iteration (feral#124), and fuses the
  D-block solve into forward substitution for ~14–29% faster warm solves
  (feral#126), plus an opt-in tree-parallel sparse solve (feral#131) and
  analysis-time assembly maps (feral#125). Numerics are bit-identical; no
  breaking API change beyond the 0.13.0 `Copy` removal noted above.

### Fixed

- **A solved convex QCQP no longer reports `InternalError` / exit 1 (#209).**
  On the `.nl` → SOCP conic path (`solver_selection=socp`, and `auto` when it
  routes a QCQP there) POUNCE converged to the correct, feasible optimum and
  then reported failure, with an end-of-run summary showing a large
  `Constraint violation` for a point that is feasible to machine precision. Any
  driver that trusts the exit code or the status — a Pyomo/AMPL wrapper, a CI
  gate — read the solve as failed. Two independent defects:
  - The summary measured the second-order cone rows with the nonnegative
    orthant's per-row `Gx ≤ h` test. A converged SOC block legitimately has
    individual rows with `Gx > h` (only the cone membership `s₀ ≥ ‖s₁‖` must
    hold) and non-complementary rows (only the block product `⟨s, z⟩`
    vanishes), so the reported violation measured nothing to do with the
    quadratic constraint. Conic solves are now measured against their own
    cones (`QpSolution::kkt_residuals_conic`).
  - The conic driver's convergence test reads the *homogeneous* residuals,
    which carry the consistency of the internal slack `s` (`Gx + s − hτ`)
    alongside the real KKT quantities. That term is bookkeeping — `s` is never
    returned — and it floors out once μ reaches ~1e-16 and the Nesterov–Todd
    scaling's condition number explodes. On the reported QCQP it bottomed at
    1e-8, drifted back up to 1e-4, and the solve ground on to a factorization
    breakdown, all while the iterate itself was accurate to 1e-14. A solve
    that ends without a verdict of its own (breakdown or iteration limit) now
    takes one from the **true KKT error of the point it returns** — cone
    feasibility, stationarity, complementarity and `z ∈ K*`, measured on the
    un-homogenized iterate. Below `tol` that is `Optimal`; within `~1e3·tol`,
    `SolvedToAcceptableLevel`; beyond it the original verdict stands. The
    check runs only after the loop, so iterates and iteration counts are
    unchanged — only the verdict and the exit code differ.
- **`pounce.solve_socp` now reports final KKT `residuals`.** Conic solves
  previously returned `residuals=None` / `kkt_error=None` because only the
  orthant measure existed; they now carry the cone-aware residuals, so a conic
  solve's convergence is checkable from Python.
- **`pounce.minimize(..., args=...)` now works with convex routing.** The extra
  objective `args` were applied on the NLP path but not bound into the copies of
  `fun`/`jac`/`hess` handed to the LP/QP/SOCP routers, which probe them as bare
  `f(x)`. A parameterized convex objective therefore either silently never
  routed (`solver_selection=auto` fell back to NLP) or was wrongly rejected as
  "not convex" under a forced `solver_selection`. `args` are now bound into the
  router probes, so a parameterized convex QP/LP/QCQP routes to the specialized
  solver as expected.
- **Post-optimal requests are no longer silently dropped on the specialized
  solve paths (#196).** When an `.nl` declared the sIPOPT sensitivity suffixes
  (`sens_state_1` / `sens_state_value_1` / `sens_init_constr`) or the solve
  asked for a reduced Hessian (`--compute-red-hessian`), the request was
  honored only on the general NLP filter-IPM path. Three fast paths bypassed it
  without a word:
  - **Convex LP/QP/QCQP routing.** Under `solver_selection=auto`, a problem
    that classifies as LP / convex-QP was sent to the pounce-convex solver,
    which has no sensitivity / reduced-Hessian machinery, so no
    `sens_sol_state_1` was written. `auto` now declines the fast path and
    routes such a solve to the NLP path (which honors the request); an
    *explicit* convex `solver_selection` still runs convex but now warns that
    the request is skipped.
  - **`--minima` multistart.** The `--minima` early-return skipped the
    post-optimal step entirely; it now warns that sensitivity / reduced-Hessian
    is not available in a multistart search.
  - **`--minima` `.sol` duals.** The multistart `.sol` wrote a zero placeholder
    for the constraint multipliers; it now recovers the real base-problem duals
    at each reported minimum (via a clean re-solve, so a point accepted from an
    augmented penalty/tunnel solve still gets the base problem's multipliers).
  - **Python `pounce.minimize` convex route.** A user `callback` cannot fire on
    the convex/SOCP route (the solver consumes the extracted quadratic form and
    never calls back into Python); it is now surfaced in the dropped-options
    warning rather than silently ignored.
- **Registered-but-unread algorithmic tuning options are now honored (#191).**
  A range of options were registered but never read, so the solver always ran
  with the hard-coded defaults and any user-set value was silently dropped.
  Now wired through `AlgorithmBuilder`:
  - `kappa_sigma` (default `1e10`) — bounds how far the bound multipliers may
    deviate from their primal estimates via a clamp applied after every
    accepted step, including the documented `< 1` value that disables the
    correction.
  - `kappa_d` (default `1e-5`) — weight of the linear damping term for
    one-sided bounds in the barrier objective/gradient.
  - Filter switching / Armijo / margin constants for the filter line search:
    `eta_phi`, `theta_min_fact`, `theta_max_fact`, `gamma_phi`, `gamma_theta`,
    `s_phi`, `s_theta`, `alpha_min_frac`, `obj_max_inc`.
  - Second-order-correction constants: `max_soc` (incl. `0` to disable SOC),
    `kappa_soc`, `soc_method`.
  - Filter-reset heuristic: `max_filter_resets` (incl. `0` to disable),
    `filter_reset_trigger`.
  - Tiny-step and divergence guards on the algorithm: `tiny_step_tol`,
    `tiny_step_y_tol`, `diverging_iterates_tol`.
  - Inertia-correction / Jacobian-regularization constants on the
    perturbation handler: `max_hessian_perturbation`,
    `min_hessian_perturbation`, `first_hessian_perturbation`,
    `perturb_inc_fact_first`, `perturb_inc_fact`, `perturb_dec_fact`,
    `jacobian_regularization_value`, `jacobian_regularization_exponent`,
    `perturb_always_cd`.
  - Iterative-refinement constants on the KKT full-space solver:
    `min_refinement_steps`, `max_refinement_steps`, `residual_ratio_max`,
    `residual_ratio_singular`, `residual_improvement_factor`.
  - Restoration-phase constants: `bound_mult_reset_threshold`,
    `constr_mult_reset_threshold`, `resto_penalty_parameter`,
    `resto_proximity_weight`. The outer builder carries these (read from the
    options list) and propagates them into the restoration builder when the
    restoration factory is minted, so all frontends honor them without
    per-frontend plumbing.

  Every default equals the previously-hard-coded value, so runs that don't set
  these options are unchanged. The only options still not wired are ones whose
  underlying behavior is not yet implemented (e.g. `neg_curv_test_tol`'s
  non-zero branch, `expect_infeasible_problem`, `start_with_resto`,
  `alpha_for_y_tol`); wiring those would be misleading until the feature
  lands.
- **`timing_statistics=no` no longer runs the detailed timers every iteration
  (#190).** Every `TimedTask::start`/`end` pair calls `getrusage(RUSAGE_SELF)`
  (twice — once each), and the per-subsystem / per-callback timers wrap hot
  paths (each objective/gradient/constraint/Jacobian/Hessian evaluation, plus
  every solve phase). Upstream Ipopt gates these detailed timers on
  `timing_statistics` (default `no`), but pounce mirrored the timers without
  the gating — so the syscalls were paid unconditionally, measuring at 16–20%
  of busy CPU on fast-objective, high-iteration NLPs. The detailed timers are
  now disabled unless `timing_statistics yes` (or `print_timing_statistics
  yes`, which implies it) is set. `OverallAlgorithm` stays live regardless: it
  feeds the `max_cpu_time` convergence check and its total is always reported.
- **pip GAMS link honors `json_output` / `json_detail` (#187).** The pure-Python
  GAMS link parsed those option-file keys and then discarded them, so a
  `pounce.opt` requesting a `pounce.solve-report/v1` JSON was a silent no-op on
  the pip route even though `docs/src/gams.md` advertises it (only the native C
  link implemented it). `Problem.solve` now takes optional `report_path` /
  `report_detail` (`"summary"` | `"full"`) kwargs that emit the report through
  the **canonical Rust writer** (`pounce-solve-report`, the same schema/serializer
  as the CLI's `--json-output`) — no report format is reimplemented in Python.
  The GAMS link threads the two link options into that surface (Full detail
  enables the per-iteration trace the `pounce-studio`/MCP post-mortem tools
  consume), and the writer is now available to any Python caller.
- **Convex QP/LP path now honors `max_iter=0` (#186).** A Pyomo/AMPL solve
  auto-routed to the `pounce-convex` interior-point path reported *Optimal
  Solution Found* even with `max_iter=0`, because the routed problem was solved
  by presolve or a direct step that ignored the iteration cap — violating the
  AMPL/Ipopt contract that zero iterations cannot reach optimality (the NLP
  path already reported `MaximumIterationsExceeded`). The convex QP and SOCP
  dispatch now short-circuit to an iteration-limit result before any solve when
  `max_iter=0`, and `max_iter` is forwarded to the convex driver for the `=0`
  case (previously dropped). CI now runs `pytest pyomo-pounce/tests`
  end-to-end (not just an import smoke test), which is how this regressed
  silently.
- **`reaction_network` mode-aware dedup on flat eigenmodes (#183).** When the
  PES has a genuine zero (flat) Hessian eigenmode — rigid translation/rotation
  of any molecule, or an intrinsically flat coordinate — the minima dedup
  compared full-coordinate distance, so copies of the *same* basin displaced
  along the flat direction counted as distinct minima and exhausted the
  `n_states` budget before flooding reached other basins (a whole basin, and
  its connections, silently missed). `reaction_network` now deduplicates
  minima, saddles, and saddle→basin descent matches in the **non-null subspace**
  of the Hessian, quotienting out any eigenmode below `eig_tol`. Reduces
  exactly to the previous scaled-Euclidean metric when no null modes are
  present, so well-conditioned surfaces are unaffected. `find_saddles` gains an
  optional `distance` override for the same purpose.

## [0.7.0] - 2026-07-01

### Added — `pounce-rs` Rust facade crate (#168)

- **`pounce-rs`** is a single-crate facade for solving nonlinear programs from
  Rust. It re-exports the `TNLP` problem trait (`pounce-nlp`), the
  `IpoptApplication` driver (`pounce-algorithm`), and the supporting scalar
  types (`pounce-common`) in one place, plus a `prelude` — the Rust counterpart
  to the one-import `import pounce` Python API. Pins a single curated public
  surface. The 20th published crate.
- **Ergonomic builder API** in `pounce-rs` (argmin-style, per the #168
  discussion): implement the small `Problem` trait (only `objective` is
  required) and configure + solve with the `Nlp` builder
  (`Nlp::new(problem).var_bounds(..).constraint_bounds(..).solve()`).
  Unimplemented `gradient` / `jacobian` are finite-differenced and the Hessian
  defaults to limited-memory L-BFGS, so a simple problem stays small; the full
  `TNLP` trait remains for advanced use. Runnable HS071 + constrained-QP
  examples in the crate docs.

### Added — event detection (`pounce.ode.solve_ivp` / `solve_dae`)

- **SciPy-compatible `events=`** on both `solve_ivp` and `solve_dae`. Zero
  crossings of event functions `g(t, y)` are located during integration,
  root-found on each step's dense-output polynomial. Each event may carry
  `terminal` (`bool` or a positive `int` count — stops with `status=1`) and
  `direction` (`>0` rising, `<0` falling, `0` either); crossings are returned in
  `res.t_events` / `res.y_events`, and `args` are forwarded to events as in
  SciPy. Event times match `scipy.integrate.solve_ivp` to solver tolerance.
  (Resolves #165 item 4.)

### Added — state-dependent mass + higher-order differentiable DAE

- **`solve_ivp(mass=M(t, y))`** now accepts a callable mass (state/time-
  dependent `M(t, y) y' = f`), routed through the fully-implicit DAE engine as
  `F = M(t,y) y' − f`; the constant-array form is unchanged. (Resolves #165
  item 3.)
- **`pounce.jax.daeint` / `pounce.torch.daeint` default to BDF2** (`order=2`,
  L-stable, second-order) instead of backward Euler; pass `order=1` for BE.
  Same node-value collocation (one extra Jacobian subdiagonal), same IFT
  backward — validated as order-2 convergent with gradients matching finite
  differences.

### Added — fully-implicit DAEs (`pounce.ode.solve_dae`)

- **`pounce.ode.solve_dae(F, t_span, y0, yp0=None, ...)`** integrates a
  fully-implicit, index-1 DAE `F(t, y, y') = 0` with the same Radau IIA(5)
  collocation as `solve_ivp`, in residual form. A pounce extension —
  `scipy.integrate.solve_ivp` has no fully-implicit DAE solver. Reuses the
  whole stiff engine (sparse-LU stage solve + pattern reuse, stage predictor,
  adaptive control, dense output / `t_eval`). Index-1 only.
- **Consistent initial conditions** computed automatically
  (`consistent="project"`, the default): algebraic variables are detected from
  the sparsity of `∂F/∂y'`, then `(y0, y'0)` are Newton-projected onto
  `F(t0, y0, y'0) = 0` (the IDA `IDA_YA_YDP_INIT` computation), so an
  approximate `y0` and `yp0=None` are accepted. `consistent="assume"` uses a
  caller-supplied consistent `yp0` as-is.
- Optional analytic `jac(t, y, yp) -> (∂F/∂y, ∂F/∂y')`; both blocks are
  finite-differenced otherwise. Docs: `docs/src/dae.md`.
- **`pounce.jax.daeint` / `pounce.torch.daeint`** — differentiable fixed-mesh
  integration of `F(t, y, y', theta) = 0`, returning the trajectory
  differentiable w.r.t. `theta` and `y0` via the implicit-function theorem on a
  backward-Euler collocation (the FERAL sparse-LU back-solve mirrors
  `pounce.jax.odeint`). Gradients validated against finite differences.

### Changed

- **`pounce.ode.solve_ivp` no longer silently no-ops SciPy parameters**
  (gh #165): passing `vectorized=True` (ignored) or an unrecognized option now
  emits a `UserWarning` instead of vanishing. The fully-implicit `solve_dae`
  above also supersedes the "constant mass only" limitation for callers needing
  a general implicit form.
- **`pounce.minimize` routed LP/QP/SOCP results now report unbounded and
  infeasible outcomes in plain language** (gh #160). When the convex solver
  returns a dual- or primal-infeasibility certificate, the result `message` now
  reads "The problem appears unbounded …" / "… infeasible …" (status `3` /
  `2`, matching SciPy `linprog`) instead of the raw `dual_infeasible` /
  `primal_infeasible` string — so a downstream adapter can distinguish
  unboundedness from a generic iteration limit. The raw certificate is still
  available in `res.info["status"]`. (Note: the general NLP path —
  `solver_selection="nlp"`, the default — cannot certify LP unboundedness, the
  same as stock Ipopt; route linear/convex problems with
  `solver_selection="lp-ipm"` / `"auto"` to get the certificate.)


### Fixed — ODE/DAE Radau engine: dense LU, complex-split stage solve, exact Jacobian (#175)

- **`pounce.ode.solve_ivp` / `solve_dae` no longer crash with a `SingularBasis`
  error** on stiff/DAE problems whose stage matrices become ill-conditioned on
  the slow manifold (e.g. the Robertson index-1 DAE). The Radau IIA(5) stage and
  error operators now factor with a faer dense partial-pivoting LU (`DenseLU`)
  that — like LAPACK / SciPy's `Radau` — always completes (a singular matrix
  surfaces as `inf`/`nan` in the solve, which the step control already handles)
  instead of hard-failing. The stage solve is rewritten as the standard RADAU5
  **complex split** (one real + one complex shifted operator via the Butcher
  eigendecomposition), so it stays well-conditioned at a singular-Jacobian
  equilibrium.
- **The stage Jacobian now defaults to exact JAX forward-mode autodiff**
  (`jax.jacfwd`) when the RHS is JAX-traceable, falling back to an accurate
  central difference for opaque callables — replacing a noisy forward difference
  that inflated the step count ~45× near singular-Jacobian steady states.
  Robertson integrated to `t = 1e11` now matches SciPy's Radau step count.

### Changed — IPM status fidelity on ill-conditioned / scaled solves (#173)

- The interior-point solver no longer reports `Solve_Succeeded` when the
  **unscaled** KKT error remains large (untrustworthy duals) even though the
  scaled error looks converged. Convergence is now gated on the unscaled
  dual/primal/complementarity infeasibility, so a downstream consumer can trust
  an `Optimal` status. Adds unscaled-error accessors to the convergence check
  and extends the fidelity fix to the SQP and convex facades.

### Changed — feral linear-solver backend bumped to 0.12.0 (#177)

- Resolves the qap15 / mittelmann conic-KKT family end to end: (#91)
  `OrderingPreprocess::Auto` verifies fill instead of predicting, removing a
  misfiring MC64 `LdltCompress` trigger that inflated fill ~6× (qap15 factor
  15.4s → 0.77s); (#99) packed BLAS-3 dense trailing update (~8–10× on large
  dense fronts); (#102) fixes a latent re-entrant nested-rayon workspace-mutex
  deadlock the ordering change exposed; (#105) escalates the ordering to
  `LdltCompress` on pivot growth so factorization accuracy holds on late μ→0 IPM
  KKTs. qap15 now solves (was a 300s timeout) with no regressions across the
  mittelmann / LP / QP suites.

## [0.6.0] - 2026-06-20

### Performance — stiff ODE stepper (`pounce.ode`)

- **Stage predictor.** The adaptive Radau stepper now warm-starts each step's
  simplified-Newton stage solve by extrapolating the previous step's
  collocation polynomial (the standard RADAU5 predictor), instead of cold-
  starting from `K = 0`. This cuts the per-step Newton iterations: on Van der
  Pol (mu=1000) it drops `nfev` ~24% (≈23.7k → ≈18k) and wall-clock ~15%,
  bringing the stiff solve to near parity with `scipy.integrate.solve_ivp`.
  No change to accuracy or the public API.
- **Wider LU-reuse band.** The step-size controller now holds `h` (and so
  reuses the cached `(3n×3n)` factor) on growth up to 2× (was 1.2×). On large
  stiff systems where the dense factor dominates, this drops factorisations
  per step well below SciPy's and cuts wall-clock ~25–30% (e.g. a 100-state
  Brusselator), with no accuracy cost.
- **Reuse the LU pattern across refactors.** The stepper was rebuilding the
  `SparseLU` object — re-bucketing the `(3n)²` COO pattern and re-running
  FERAL's symbolic analysis — on *every* refactor, even though the sparsity
  pattern is fixed for the whole solve. The pattern object is now built once
  per solve and refactored in place (the binding already caches the symbolic),
  so each step pays only the numeric factorisation. This is a large-`n` win
  that grows with system size: **~4× faster on a 100-state Brusselator
  (318 → 79 ms) and ~7× on 300 states (2.83 s → 0.41 s)**, cutting the gap to
  `scipy.integrate.solve_ivp` from ~14–30× down to ~3–4×. Identical accuracy
  and step counts; no API change.

### Added — boundary value problems (`pounce.bvp`)

A `scipy.integrate.solve_bvp`-compatible boundary value problem solver, plus
differentiable JAX/PyTorch frontends:

- **`pounce.solve_bvp(fun, bc, x, y, p=None, ...)`** — drop-in for
  `scipy.integrate.solve_bvp`. Discretises the BVP with 4th-order
  Hermite–Simpson collocation on a fixed mesh and solves the square
  collocation root-find as a pounce feasibility NLP. Returns a SciPy-shaped
  bunch (`sol`, `x`, `y`, `yp`, `p`, `rms_residuals`, `niter`, `status`,
  `message`, `success`). Accuracy matches SciPy (same collocation scheme).
  The default `method="newton"` factors the exact **sparse** `N×N`
  collocation Jacobian (analytic per-node blocks from `fun_jac`/`bc_jac`,
  else a vectorised finite difference) with FERAL's unsymmetric sparse LU,
  using a **modified (frozen-Jacobian) Newton** that reuses the factor
  across steps and refactors only on stall — so it is **typically faster
  than `scipy.integrate.solve_bvp`** at equal mesh (≈0.6–1.0×), including
  large nonlinear problems. `method="ipm"` solves it as a pounce
  feasibility NLP.
  Adaptive mesh refinement is **on by default** (`adaptive=True`, like
  SciPy — a faithful port of SciPy's Lobatto residual estimator + refinement
  rule that reproduces its mesh sequence node-for-node); `adaptive=False`
  solves the given mesh as-is. The collocation system is solved to round-off
  independent of the mesh `tol` (the latter only gates refinement).
  `verbose` mirrors SciPy (1 = termination report, 2 = per-iteration
  progress). Result `status` codes: 0 converged, 1 max nodes, 2 singular
  Jacobian, 3 bc_tol unmet, 4 Newton non-convergence, 5 IPM acceptable-only.
- **`pounce._pounce.SparseLU`** — new PyO3 binding exposing FERAL's
  unsymmetric sparse LU (`factor` / `solve` / `solve_transpose`) for direct
  `A x = b` on general sparse matrices.
- **`pounce.solve_bvp_constrained`** — constrained / optimal-control BVPs
  (state & parameter bounds, inequality path constraints, optional
  objective over an under-determined system), solved with the interior-point
  method on the collocation NLP. This is unique to pounce —
  `scipy.integrate.solve_bvp` cannot express bounds, path constraints, or an
  objective.
- **`pounce.jax.solve_bvp` / `pounce.torch.solve_bvp`** — the same solve made
  differentiable w.r.t. a `theta` parameter threaded into `fun` / `bc`, via
  the implicit-function theorem on the collocation system. Supports
  gradients/Jacobians w.r.t. ODE/BC coefficients, boundary values, and the
  sensitivity of solved-for unknown parameters `p*`. The default
  `method="newton"` is the fast path (FERAL sparse-LU forward + sparse
  `R_zᵀ` backward, first-order). `method="ipm", second_order=True` wraps the
  solve in a `custom_jvp` that re-applies the implicit-function theorem,
  enabling `jax.grad(jax.grad(...))` / `jax.hessian` to arbitrary order.
- Docs: `docs/src/bvp.md`; worked accuracy/speed/differentiability comparison
  in `python/examples/bvp_scipy_compare.py`.

#### Scope and positioning

Honest framing of where this sits relative to other BVP solvers:

- **Algorithm class.** Fixed 4th-order Hermite–Simpson collocation — the same
  family as MATLAB `bvp4c` and `scipy.integrate.solve_bvp` (itself a
  bvp4c-style port). At equal mesh we match SciPy's accuracy and are
  typically a bit faster; this is "competitive with a widely-used production
  solver," **not** the numerical state of the art. Higher-order /
  variable-order collocation (COLNEW/COLSYS), 5th-order `bvp5c`, and
  deferred-correction / continuation codes (TWPBVP, ACDC) need fewer nodes
  per digit of accuracy and are more robust on stiff / singularly-perturbed
  boundary-layer problems.
- **Where it genuinely leads.** End-to-end **differentiability** of the
  solution (`∂y/∂θ`, Jacobians, second order) via implicit differentiation in
  JAX/PyTorch, and **integrated bound / path constraints and objectives**
  (optimal control) through the IPM — capabilities classical BVP solvers do
  not offer. (For heavy constrained optimal control, mature direct-collocation
  stacks such as CasADi and Pyomo.DAE + IPOPT remain more complete.)
- **Not yet covered:** variable/high-order collocation; continuation /
  deferred correction for stiff boundary layers; multipoint boundary
  conditions; DAEs; the singular term `S`; complex-valued problems. A
  credible "SOTA" claim would also require benchmarking against COLNEW /
  `bvp5c` / SciPy on a standard suite (e.g. the Cash–Mazzia test set) for
  accuracy-vs-nodes and robustness, not just speed-vs-SciPy.

### Added — stiff ODE / DAE initial value problems (`pounce.ode`)

A `scipy.integrate.solve_ivp`-compatible stiff solver, plus differentiable
JAX/PyTorch frontends:

- **`pounce.ode.solve_ivp(fun, t_span, y0, method="Radau", ...)`** — drop-in
  for `scipy.integrate.solve_ivp` with the implicit `Radau` method (3-stage
  Radau IIA, order 5, L-stable — the same RADAU5 of Hairer–Wanner that SciPy
  implements). Adaptive step control with the embedded order-3 error estimate
  and a simplified-Newton stage solve whose Jacobian is factored with FERAL's
  sparse LU. Tracks SciPy's `Radau` step-for-step on stiff problems (Van der
  Pol μ=1000: 1082 vs SciPy's 1188 steps, agreeing to ~7e-7) and returns a
  SciPy-shaped bunch (`t`, `y`, `sol`, `nfev`, `njev`, `nlu`, `status`,
  `message`, `success`). Supports `t_eval`, `dense_output`, `args`, `jac`,
  `first_step`, `max_step`, `rtol`/`atol`. Only `method="Radau"` is
  implemented (the stiff/DAE niche); other methods raise rather than silently
  substitute, and `events=` is not yet supported.
- **Index-1 DAEs** via a mass matrix: pass `mass=M` to integrate `M y' = f`.
  A **singular** `M` makes it an index-1 differential-algebraic equation —
  something `scipy.integrate.solve_ivp` cannot do. Validated on Robertson
  kinetics (conservation constraint held to round-off).
- **`pounce.jax.odeint` / `pounce.torch.odeint`** — differentiable
  fixed-mesh integration. An IVP on a fixed mesh is a BVP with
  `bc(ya, yb) = ya - y0`, so this reuses the Hermite–Simpson collocation and
  the same FERAL sparse-LU implicit-diff back-solve as `solve_bvp`. Returns
  the trajectory differentiably w.r.t. the ODE parameters `theta` **and** the
  initial condition `y0` (gradients exact for the discretisation; checked
  against analytic and finite differences).
- **Dict-subscriptable results.** `OdeResult` and `BVPResult` now support
  SciPy-`Bunch`-style item access (`res["y"]`, `"success" in res`,
  `res.keys()`, `res.get(...)`) alongside attribute access, for a tighter
  drop-in.
- Docs: `docs/src/ode.md`; worked stiff/DAE/differentiability comparison in
  `python/examples/ode_scipy_compare.py`.

### Added — GAMS solver link, now pip-installable (`pounce-solver[gams]`)

- **`pip install pounce-solver[gams]` + `pounce-gams register`** registers
  POUNCE as a GAMS NLP solver (`option nlp = pounce;`) with no compiler, no
  `sudo`, and nothing GAMS-owned redistributed — built on GAMS's own
  `gamsapi[core]` GMO/GEV bindings. The link wires GMO's numerical evaluators
  straight into the solver's cyipopt-style `Problem` callbacks (POUNCE is a
  local NLP solver, so no opcode translator is needed). Registration merges a
  per-user `gamsconfig.yaml` `solverConfig` entry, preserving other solvers and
  surviving GAMS upgrades. The native C link in `gams/` remains as the
  alternative route. Adds the `pounce-gams` / `pounce-gams-link` console scripts
  and the `[gams]` extra. Docs: `docs/src/gams.md`.

### Added — solver & `.nl` parser

- **`mu_strategy_fallback`** (opt-in, default off): on a
  `Solved_To_Acceptable_Level` or `Maximum_Iterations_Exceeded` exit, flip
  `mu_strategy` (adaptive↔monotone) once and re-solve, promoting the retry only
  if it reaches `Solve_Succeeded` (otherwise the original outcome is kept).
  Recovers genuine adaptive-μ stalls.
- **AMPL power opcodes** `o81` / `o82` / `o83` (`OP1POW` / `OP2POW` / `OPCPOW`)
  in the `.nl` reader. AMPL emits these as a hint that one operand is constant;
  they previously hit the unsupported-opcode fallthrough, so any `.nl` emitting
  them failed to parse. They lower to the existing negative-base-safe
  constant-power path.

### Fixed

- **`acceptable_iter=0` now disables acceptable-level termination** (restoring
  upstream Ipopt's `acceptable_iter_ > 0` guard) instead of firing on the very
  first acceptable iterate. The GAMS link also now defaults `acceptable_iter=0`,
  mirroring the GAMS–Ipopt link, which removes premature
  `Solved_To_Acceptable_Level` exits on several princetonlib models.
- **CLI:** honor `presolve` on the convex LP/QP path (#139); report reduced
  (post-fixed-variable-removal) dimensions in the solver banner (#140).


## [0.5.0] - 2026-06-14

### Added — broader `scipy.optimize.minimize` compatibility

`pounce.minimize` now covers much more of the SciPy surface, so it works as a
drop-in `method=` callable for `scipy.optimize.minimize` and ports existing
SciPy code with fewer changes:

- **`args=(...)`** — extra positional arguments forwarded to `fun` / `jac`.
- **`jac=True`** — `fun` returns `(value, gradient)` in one call; the pair is
  cached so the gradient is not recomputed.
- **`callback`** — invoked each iteration; both SciPy signatures are accepted
  (`callback(xk)` and `callback(intermediate_result)`).
- **scipy `Bounds` and `LinearConstraint` objects** — accepted alongside
  `(lo, hi)` pairs and constraint dicts; a `LinearConstraint` may carry a
  sparse `A`, which is honored. When all constraints are linear the objective
  Hessian is the Lagrangian Hessian, so an exact `hess` is used (no L-BFGS
  fallback).
- **scipy option spellings** as synonyms — `maxiter`→`max_iter`,
  `gtol`/`ftol`/`xtol`→`tol`, `disp`→`print_level`, `maxcor`→
  `limited_memory_max_history`; options may be passed as `**kwargs` (the legacy
  `options={…}` dict still works).
- The result is now a genuine `scipy.optimize.OptimizeResult` carrying the
  `nfev` / `njev` / `nhev` evaluation counters, with pounce extras under
  `res.info` and a back-compat shim so a key absent at the top level falls back
  to `res.info`.

**Changed:** the `solver_selection` default is now `"nlp"` (no structure
probing) — automatic LP/QP/QCQP routing is opt-in via `solver_selection="auto"`,
so a general NLP or an expensive `fun` pays no probe overhead. The `args`
argument is now the third positional parameter (matching SciPy), ahead of `jac`.

### Fixed — `obj_scaling_factor` was silently ignored (maximization diverged)

The `obj_scaling_factor` option was registered but never read: every solve
constructed the NLP with no-op scaling, so the documented behavior — a
constant multiplier on the objective, negative to **maximize** — was a silent
no-op and maximization problems diverged (the IPM minimized the unscaled
objective). The option value is now carried into `OrigIpoptNlp` on both the
IPM and SQP paths (`ConstObjScaling`), combining with gradient-based /
user scaling exactly as documented. Sensitivity analysis works under a
negative factor too: the natural-units correction from #128 below uses a
two-sided scaling with no square root, so `solve_with_sens` /
`Solver.reduced_hessian` return the declared problem's reduced Hessian for
maximization problems as well.

### Added — KKT regularization reported alongside sensitivity outputs

The IPM's inertia-correction perturbations are baked into the converged
factor in scaled space, so a regularized final factorization makes the
natural-units sensitivity outputs (covariance in particular) inexact and not
perfectly scaling-invariant. The final `(δ_x, δ_s, δ_c, δ_d)` are now
reported so workflows can check for the all-zero (exact) case:
`info["kkt_perturbations"]` and `Solver.kkt_perturbations` (Python),
`SensResult::kkt_perturbations` and `Solver::kkt_perturbations` (Rust).

### Fixed — sensitivity back-solves now return natural (unscaled) units (#128)

The reduced Hessian from `solve_with_sens(compute_reduced_hessian=True)` /
`Solver.reduced_hessian`, the parametric step `dx`, and the raw
`Solver.kkt_solve` were returned in the IPM's internally **scaled** space
whenever NLP scaling was active (the default
`nlp_scaling_method = "gradient-based"` fires when an objective gradient or a
constraint row exceeds 100 at the starting point). For a parameter-estimation
NLP this made `-inv(reduced_hessian)` differ from the true covariance by
`df / (dc_i·dc_j)` — the discretization-tracking "≈ nfe" fudge factor reported
in #128. The same scaled factor silently corrupted the factor-reuse VJP/JVP of
**both** differentiable frontends (`pounce.jax` `JaxProblem(factor_reuse=True)`
and `pounce.torch` `TorchProblem`) on badly-scaled problems.

The scaled primal-dual system is the two-sided diagonal scaling
`K_scaled = E·K_natural·F` (per-block: `E = (df, df/dd, dc, dd, df, df)` and
`F = (1, 1/dd, dc/df, dd/df, 1/df, dd/df)` over `x, s, y_c, y_d, z, v`), so
every held-factor back-solve now computes `K_natural⁻¹ = F·K_scaled⁻¹·E`: all
eight KKT blocks — including the bound-multiplier z/v rows in `dx_full` —
come back in the user's own units regardless of scaling method, and a
negative `obj_scaling_factor` is handled (no square root involved). The CLI
sIPOPT mode inherits the same correction: the `red_hessian` var-suffix output
is now natural-units where upstream sIPOPT prints a scaled value it warns
about. The pre-fix solver-space values and the factors stay accessible:
`info["reduced_hessian_scaled"]` / `info["obj_scaling_factor"]` /
`info["pin_g_scaling"]`, `Solver.reduced_hessian(..., scaled=True)`,
`Solver.kkt_solve(..., scaled=True)` / `kkt_solve_many(..., scaled=True)`,
the `Solver.nlp_scaling` dict, the C ABI's `IpoptSolverKktSolveScaled`, and
the matching Rust surfaces (`SensResult` fields,
`Solver::{compute_reduced_hessian_scaled, kkt_solve_scaled,
kkt_solve_many_scaled, nlp_scaling}`, `PdSensBacksolver::solve_scaled_space`).

Also fixed in the same change: `SensSolve` / `Solver` pin-constraint indices
are now mapped to KKT rows through the equality/inequality split
(`full_g_to_c_block`), so pins are selected correctly when inequality
constraints precede them in `g(x)` (previously the wrong row was used
silently; the CLI sIPOPT path already mapped correctly and now shares the
same helper). `pounce.curve_fit` no longer requires scaling to be off to
trust the converged factor for its covariance / data-sensitivity reads.

### Changed — C ABI: sensitivity entry points now return natural units (breaking)

Behavior change for C callers of the sensitivity ABI (`pounce-cinterface`).
`IpoptSolverReducedHessian`, `IpoptSolverParametricStep`, and
`IpoptSolverKktSolve` now return values in **natural (unscaled) units** as part
of the #128 fix above — previously, on a badly-scaled NLP (where the default
`gradient-based` method fires), they returned the IPM's internally scaled
values. A C caller that was compensating for the old behavior — e.g. passing a
non-`1.0` `obj_scal` to `IpoptSolverReducedHessian` to undo the `df / dc²`
factor by hand — will now get a doubly-corrected (wrong) result and must drop
that workaround; `obj_scal` is once again only the plain extra multiplier its
docs describe. Callers that want the old scaled values back have an escape
hatch **only** for the raw KKT solve: `IpoptSolverKktSolveScaled(..., scaled =
true)`. There is intentionally no scaled variant of `IpoptSolverReducedHessian`
or `IpoptSolverParametricStep` — the natural-units reduced Hessian and
parametric step are the only correct answers for a covariance / predictor read,
so the scaled forms are not re-exposed across the ABI (the Rust
`Solver::compute_reduced_hessian_scaled` remains for in-process calibrated
callers).

### Added — Batched NLP solving (`solve_nlp_batch`) (#126)

Solve N independent NLPs in parallel on a Rayon pool — the general-NLP
analog of `solve_qp_batch_parallel`, for parametric sweeps, multi-start,
MPC chains, and branch-and-bound node relaxations (each sibling node
differing only in tightened bounds).

- **Rust** — `pounce_algorithm::solve_nlp_batch` /
  `solve_nlp_batch_parallel`: one fully-equipped `IpoptApplication` per
  instance, built *inside* the worker via a `Sync` configure hook that
  receives the instance index (outer-parallel / inner-serial, like the
  QP batch; `install_serial_feral_backend` sets up the per-worker
  serial factor). Results return in input order with the captured
  final iterate, multipliers, and per-instance `SolveStatistics`.
- **`pounce-nl`** — CSE `Expr` sharing switched from `Rc` to `Arc`, so
  `NlProblem` / `NlTnlp` are `Send` and an owned evaluator can move to a
  worker. `NlTnlp` is now `Clone`, and `NlTnlp::variant` /
  `NlVariation` build per-instance bound / starting-point overrides on
  one parsed model (tapes are cheap to clone).
- **Python** — `pounce.solve_nlp_batch(problems, x0s=, options=,
  parallel=, warms=, share_structure=)`: native `NlProblem` inputs
  (from `read_nl` or the new `NlProblem.variant(...)`) solve in
  parallel with the GIL released. One `(x, info)` pair per input,
  `info` matching `Problem.solve`'s layout; `print_level` defaults to
  0 for the batch.
- **Phase 2: parallel callback batching** — callback-based
  `pounce.Problem` inputs also solve in parallel: each instance's
  bridge (Python callables + pre-resolved sparsity) moves to a rayon
  worker that owns the whole solve, re-acquiring the GIL transiently
  per `eval_*` callback. The GIL serializes only the Python share, so
  the speedup scales with the Rust/Python work ratio (~4x on 4 cores
  for an n=800 banded NLP with NumPy-vectorized callbacks; tiny
  callback-dominated problems won't speed up). Per-instance
  `add_option` settings are honored, with `options=` as a batch-level
  overlay; a raising callback degrades to that instance's failure
  without poisoning the batch.
- **Warm-started batches** — `solve_nlp_batch_parallel_warm` /
  `solve_nlp_batch_warm` (Rust) and `warms=` (Python, both input
  kinds): seed each instance from a previous result's iterate + duals
  and thread the converged barrier μ into `mu_init`
  (`warm_start_init_point=yes` forced; dimension mismatch falls back
  to a cold start). Re-solving a perturbed 24-instance `.nl` sweep
  warm cut total iterations 482 → 120.
- **Identical-sparsity structure sharing** — `FeralBackendPool` /
  `install_pooled_serial_feral_backend` (Rust) and
  `share_structure=True` (Python): opt-in per-worker backend pooling
  so FERAL's pattern-fingerprint symbolic cache (ordering + supernode
  structure) carries across batch instances instead of being rebuilt
  per instance. Always correct (pattern changes re-analyze); results
  are within solver tolerance of — not guaranteed bit-identical to —
  fresh-backend solves, which is why it is opt-in. Cross-*thread*
  symbolic sharing stays future work (needs the `BackendPool`
  ownership refactor documented in `dev-notes/backend-pool-resolve.md`
  or a feral-side symbolic export API).

### Added — PyTorch frontend for the differentiable solver (`pounce.torch`)

A PyTorch frontend mirroring `pounce.jax`: a solve is a
`torch.autograd.Function` you can drop inside a learned model and backprop
through, with the same constraint-satisfaction guarantee. This is a thin
adapter, not a second solver — the Rust IPM core and the
implicit-function-theorem backward are framework-agnostic; only the array
namespace differs. Because PyTorch is eager, the adapter is smaller than the
JAX one (no `pure_callback` / `ShapeDtypeStruct`, no host-callback registry or
single-thread executor pin), and float64 is requested per tensor rather than
via a global flag.

Surface (parity with `pounce.jax`):

- `from_torch` — build a `Problem` from `torch.func`-traced `f` / `g`
  (`grad` / `jacrev` / `jacfwd` / `hessian`; CPR colored AD for `sparse=`,
  via the shared detection/coloring helpers now in `pounce._ad_common`).
- `solve` / `solve_with_warm` — `autograd.Function` + KKT implicit-diff
  backward, with dual + barrier-μ warm-start threading.
- `vmap_solve` / `vmap_solve_parallel` — sequential / threadpool batches.
- `TorchProblem` — build-once handle with k_aug-style factor-reuse backward
  (`Solver.kkt_solve_many`), stacked block-diagonal batched solve, and the
  anchor / sensitivity / jvp_from_state / vjp_from_state / active_set_margin
  post-solve API.
- `solve_qp` / `solve_qp_batch` / `solve_socp` / `QpLayer` — OptNet-style
  differentiable conic layers (feasible-by-construction).
- `PathFollower` / `inverse_map_rhs` — predictor–corrector path following.

PyTorch is an optional dependency (`pip install pounce-solver[torch]`,
torch ≥ 2.2). See the [PyTorch integration guide](docs/src/python.md).

### Added — Convex QCQP auto-routes to the conic (SOCP) solver

The `auto` router now recognizes a convex **quadratically-constrained QP** and
sends it to the `pounce-convex` conic interior-point solver instead of the
general NLP path. Each convex-quadratic inequality `½xᵀHx + aᵀx + b ≤ 0`
(`H ⪰ 0`) is reformulated to one second-order cone (`H = FᵀF`, via a pivoted
rank-revealing Cholesky so a rank-deficient `H` yields the minimal cone), solved
alongside the QP objective and linear constraints, and its dual is mapped back
to a per-constraint multiplier. Works on both surfaces:

- **CLI** — a convex-QCQP `.nl`/Pyomo model routes automatically; force it with
  `solver_selection=socp` (errors if the problem is not a convex QCQP).
- **Python** — `minimize()` probes each constraint's Hessian at an anchor plus
  held-out points, validates the fitted quadratic before trusting it, and routes
  only when the feasible set is provably convex (a scipy `ineq` `g(x) ≥ 0` must
  be concave); otherwise it falls back to NLP. `options={"solver_selection":
  "socp"}` forces the conic path.

This closes the long-standing "conic solver: future" gap in the routing docs —
the conic solver shipped in 0.4.0 but was not reachable from either router for
quadratic *constraints*. See [LP / QP Solver Routing](docs/src/lp-qp-routing.md)
and [Choosing a Solver](docs/src/choosing-a-solver.md).

### Added — CLI knobs for the convex IPM and active-set QP solvers (#134)

Previously hard-coded solver defaults are now registered CLI options:

- **Convex IPM** (`solver_selection=lp-ipm` / `qp-ipm` / `socp`): `qp_tau`,
  `qp_reg`, `qp_infeas_tol`, `qp_hsde`, `qp_equilibrate`, `qp_crossover`.
  Each is forwarded only when explicitly set, so engine defaults are otherwise
  preserved.
- **Active-set QP** (`solver_selection=qp-active-set`): `sqp_qp_feas_tol`,
  `sqp_qp_opt_tol`, `sqp_qp_max_iter`, `sqp_qp_elastic_gamma`,
  `sqp_qp_anti_cycling` (`expand` / `bland` / `none`).

Both families are documented in `dev-notes/lp-qp-routing.md`. The LP crossover
default is now **off** (opt in with `qp_crossover=yes`): crossover-on regressed
the LP suite 3×–800× without reaching an exact vertex on the GEN family — the
machinery is unchanged, only the default flips.

### Fixed — `solver_selection=socp` was rejected; debugger silently no-op'd on `qp-active-set`

Two reachability gaps around the new convex solvers, found while confirming the
interactive debugger works on every backend:

- `solver_selection=socp` was a documented value (the conic IPM that a convex
  QCQP reaches under `auto`) but was missing from the option's registered
  allow-list, so forcing it failed with `Invalid value "socp"`. It is now
  accepted and routes a convex LP/QP/QCQP to the SOCP conic IPM.
- The interactive debugger (`--debug` / `--debug-script` / `--debug-json`) is a
  pdb-for-the-IPM. It engages correctly on `lp-ipm`, `qp-ipm`, and `socp`, but
  the active-set SQP engine (`qp-active-set`) has no such hook, so a debug
  request there used to run to completion without ever pausing — a silent
  no-op. It now prints an explicit note that the debugger is IPM-only and points
  at `qp-ipm` for interactive convex-QP debugging.

### Added — exact-vertex LP crossover (revised simplex)

`pounce-convex` gains a revised-simplex crossover that purifies a near-optimal
interior LP iterate to an exact optimal vertex, pivoting one variable at a time
on feral's unsymmetric sparse LU with Bland's anti-cycling rule (so it walks
through the highly degenerate NETLIB GEN vertices where the active-set bridge
stalls). It is tried first and accepted only when the KKT error does not
regress; on any breakdown it falls back to the legacy active-set bridge. Opt-in
via `qp_crossover=yes`.

### Fixed — convex LP/QP robustness on rank-deficient and large-scale data

Interior-point hardening in the dedicated convex (`lp-ipm` / `qp-ipm`) path:

- **Adaptive equality-block regularization (δ_c)** (#133). The
  equality-multiplier `(y,y)` block was frozen at a static value; on
  rank-deficient equality Jacobians that leaves a near-singular saddle and the
  solve plateaus to max_iter. δ_c now seeds from a μ-scaled base
  (`1e-8·μ^0.25`) and escalates on singular factorization / wrong KKT inertia /
  un-refinable direction probes, resetting each iteration so one hard iterate
  never inflates regularization for the rest of the solve. Regression-clean on
  NETLIB; drops the rank-deficient GEN constraint violation 9.3e-5 → 3.1e-8.
  (GEN still floors just above the 1e-8 tolerance, so #133 stays open.)
- **Scale-gated relative stopping + ratio-based infeasibility ray**. The
  absolute KKT stop is unreachable once the data scale pushes the
  finite-precision residual floor above tol — POWELL20 / BOYD1 / BOYD2 /
  QFORPLAN / QSHELL (scale 7e9–4e12) ran to max_iter despite being optimal to
  ~1e-9 relative. A Clarabel-style scale-relative residual now relaxes the
  absolute test, but only once roundoff sits below tol, so well-scaled problems
  are unaffected. Infeasibility now triggers on the ratio κ/τ→∞ rather than a
  bare τ floor, fixing a false `PrimalInfeasible` on feasible large-norm-x QPs.

### Fixed — active-set QP cycling on degenerate phase-1 (#133)

The active-set QP elastic phase-1 recovery (ℓ₁-infeasibility minimization,
γ=1e6) is inherently highly degenerate and could cycle under the default
anti-cycling rule (NETLIB `afiro` bailed at iteration 0). The phase-1 solve now
runs under Bland's rule (provably finite), and `solve_general` latches into
Bland after 50 consecutive non-improving iterations as a sticky,
scale-invariant anti-stall safety net.

### Changed — large sparse convex QPs are now recognized as convex (faster routing)

Problem classification certifies a coupled quadratic Hessian as PSD via feral's
sparse LDLᵀ inertia (~O(nnz·fill)) instead of a dense Jacobi eigensolve, so
large-but-sparse convex QPs (the CVXQP family, n≈1000) that previously fell back
to the general NLP solver are now sent to the dedicated convex path.

### Added — limited-memory update type & history honored on the IPM path (#131, #132)

`limited_memory_update_type` and `limited_memory_max_history` were registered
but read nowhere on the interior-point path (the updater was hard-wired to
Powell-damped BFGS). Both are now threaded through to the limited-memory
updater. The default is unchanged (`bfgs`, history 6 — bit-exact with Ipopt), so
there is no behavior change unless set; `sr1` (which can represent negative
curvature) is now selectable and rescues ill-conditioned nonconvex objectives
where damped BFGS hides indefiniteness from the inertia check.


## [0.4.0] — 2026-06-05

### Added — Convex / conic solver (`pounce-convex`; `solve_qp` / `solve_socp`)

POUNCE is no longer NLP-only: a new pure-Rust convex interior-point solver
(`pounce-convex`) handles **LP, convex QP, SOCP, and PSD / exp / power cones**,
solving each to a **global** optimum (a convex problem has no other kind). It
uses a homogeneous self-dual embedding (HSDE) — symmetric for the self-dual
cones and a non-symmetric driver for the exponential/power cones — over a
`Cone` abstraction (`nonneg`, `soc`, `psd`, `exp`, `power`, plus composite and
chordal decompositions for sparse SDPs). Convex solvers extract the constant
`P`, `A`, `c`, `b` data once at setup rather than re-evaluating per iteration,
and share the `pounce-linsol` / `pounce-linalg` factorization substrate with the
NLP path. Python entry points are typed (not SciPy-shaped, by necessity — a cone
program is *data*, not a callable): `solve_qp(P, c, A, b, G, h, lb, ub, …)`,
`solve_socp(…, cones=…)`, plus `solve_qp_batch` / `solve_qp_multi_rhs` for
batched factor reuse, and a reduced-Hessian sensitivity API. The CLI reads conic
instances from CBLIB / `.cbf` (including PSDCON / HCOORD / DCOORD SDP blocks).

### Fixed — Convex LP/QP reported objective dropped tree-folded constant

The convex LP/QP path (`solver_selection=lp-ipm` / `qp-ipm`) reported an
objective off by the objective's constant term whenever AMPL/Pyomo folded that
constant into the **nonlinear objective tree** (the `+9` of `(x-3)²`) rather
than the `.nl` linear-section constant. The quadratic-form extractor
(`analyze_quadratic_full`) discarded the degree-0 term — correct for the
*minimizer*, wrong for the *reported value* — so e.g. `HS21` reported `0.04`
instead of `−99.96` and `HS35` `−8.889` instead of `0.111`. The extractor now
returns that constant and the convex driver adds it to the reported objective
alongside `obj_constant`; the optimal point was always correct. Caught by a
head-to-head NLP-vs-convex run over the Maros-Mészáros QP and NETLIB LP suites
(`benchmarks/nl_compare_nlp_vs_convex.md`).

### Fixed — Convex LP/QP IPM stalled on badly-scaled NETLIB LPs

The static KKT regularization `δ` (added on the reduced KKT diagonal so the
LDLᵀ has a stable inertia) was `1e-8`, large enough to **floor the achievable
primal residual** at `δ·‖dy‖`: with a full Newton step `A·dx = −r_p + δ·dy`, so
on instances with large equality multipliers the primal infeasibility cannot
fall below `δ·‖dy‖`. On NETLIB `adlittle` (`‖dy‖ ≈ 4e8`) this froze `inf_pr`
near 4 and the LP IPM ran to its iteration cap, returning a wrong objective
(`439665` vs the published `225494.96`). Lowering the default `δ` to `1e-10` —
still strictly positive, so the system stays quasi-definite — clears the floor:
`adlittle` now converges in ~57 iterations to the optimum, `stocfor1` speeds up
(139 → 71 iters), and the rest of the LP/QP suites are unchanged (the QP suite
is bit-identical). The whole `1e-9‥1e-11` band converges the benchmark suites;
`1e-10` is centered in it.

Also: the convex IPM's opt-in iteration trace now records a **terminal record at
the converged iterate** (the NLP path's N+1 convention), so the trace always
ends at the optimum instead of at the last pre-step state — previously a solve
that converged in a single step left only the cold-start record in the trace.

### Added — SOS polynomial global optimization (`sos_minimize`)

`sos_minimize(objective, *, inequalities, equalities, …)` computes **certified
global** lower bounds for polynomial optimization via a sum-of-squares /
Lasserre relaxation (Putinar localizing multipliers for constraints), built on
the new PSD cone. When the relaxation is exact it extracts the global
minimizer(s) with an exactness certificate (multi-atom extraction without a
non-symmetric eig, plus facial reduction for degenerate solves).

### Added — Multi-backend interactive debugger (convex/conic IPM)

The interactive debugger was generalized over a `DebugState` trait so one REPL
drives the convex solver as well as the NLP loop. New backend: a
**convex/conic** debugger (`pounce_cblib --debug`, wired through the symmetric
and non-symmetric HSDE drivers), exposing the same checkpoints and commands as
the NLP path. This composes with the 0.4.0 debugger features below (quote-aware
tokenization, `ask` provider presets, `--debug-json` protocol, Ctrl-C escape
hatch).

### Added — `pounce.curve_fit` (Python)

A `scipy.optimize.curve_fit`-style nonlinear fitter on top of the
interior-point solver, returning much more than `(popt, pcov)`:

- parameter covariance, standard errors, and Student-t confidence intervals
  read pounce-natively from the converged factor's reduced Hessian
  (`pcov = 2·s²·inv(H_S) = s²·(JᵀJ)⁻¹`; matches scipy / `pycse.nlinfit`). The
  t-quantiles use scipy when present and an accurate scipy-free inverse-t
  (via the inverse regularized incomplete beta) otherwise, so the CIs are
  correct on a numpy-only install even for small samples;
- a smooth (C²) loss family — ordinary/weighted least squares plus robust
  Cauchy and a smooth pseudo-Huber, exposed under both `soft_l1` and `huber`
  (the same C² loss: a true piecewise Huber is only C¹, which the IPM can't
  use), with a sandwich covariance estimator (non-smooth L1/MAE is
  intentionally out of scope for the IPM);
- parameter constraints scipy can't express — positivity/negativity/ranges
  via `bounds`, and relations between parameters via `constraints=`; an active
  bound/constraint yields a covariance projected onto the free subspace;
- data sensitivity `dpopt/ddata` (∂params/∂data) from a single batched
  back-solve against the same factor (`Solver.kkt_solve_many`);
- a `CurveFitResult` with `predict()`, `confidence_band()` (both `confidence`
  and `prediction` kinds, heteroscedastic-aware), `correlation`, R²/χ²/dof,
  and `summary()`.

Derivatives resolve analytic `jac` → JAX autodiff (the default for
`jax.numpy` models) → a finite-difference fallback; exact derivatives let the
solve converge cleanly with scaling off, which is what makes the
factor-based covariance and sensitivity exact. Docs:
`docs/src/curve-fitting.md`; notebook `python/notebooks/18_curve_fit.ipynb`.

`p0` is now optional even without bounds: when omitted, the parameter count is
read from the model signature and the starting point is chosen data-drivenly
(a bound-aware, data-scale candidate sweep scored by the objective) instead of
defaulting to a flat vector of ones — so badly-scaled problems get a far better
seed, while `ones` (clipped into the bounds) is always among the scored
candidates so the choice is never worse than the old default.

### Added — `pounce.curve_fit_minima` (Python)

`curve_fit_minima` finds **multiple** parameter sets that each explain the
data, for the non-convex problems where one fit isn't the whole story
(peak-assignment ambiguity, frequency aliasing in sinusoids, amplitude/decay
trade-offs in sums of exponentials, sign/label symmetry, …).

- drives `pounce.find_minima` over the *very same* fitting objective as
  `curve_fit` — identical `sigma` weighting, robust `loss`, `f_scale`,
  `constraints`, and resolved Jacobian — so the enumerated minima are true
  optima of the actual fit, not a separate surrogate;
- reuses the model Jacobian as the search **gradient** and the Gauss-Newton
  matrix as the search **Hessian**, which sharpens the basin escapes and lets
  `find_minima` certify each point as a minimum (rejecting saddles);
- refines every distinct minimum into a full `CurveFitResult` (covariance,
  CIs, optional `dpopt/ddata`) and returns them ranked by SSE, best first;
- the `method`, `n_minima`, `max_solves`, `patience`, `dedup`, `seed`, and
  `find_minima_kw` arguments pass straight through to `find_minima`; finite
  `bounds` define the box it samples / repels within. Docs:
  `docs/src/curve-fitting.md`.

### Added — `pounce verify` subcommand + signed receipts

A `verify` subcommand that re-derives feasibility from the canonical `.nl`
rather than trusting a `.sol`'s status line or the solver/agent that produced
it — the trust anchor when pounce is a tool an agent calls: the agent
proposes a solution, a small deterministic checker disposes.

- `pounce verify <problem.nl> <claim.sol>` evaluates `g(x*)` and bounds
  against the canonical model, reporting the worst constraint/bound violation
  and (when the `.sol` carries duals) a bound-projected KKT stationarity
  residual. Exit 0 = VERIFIED, 20 = REJECTED, 2 = usage/IO. Feasibility
  gates; optimality is informational unless `--require-optimal`.
- The JSON receipt content-addresses both inputs by SHA-256 (zero new deps);
  with `POUNCE_VERIFY_KEY` set it signs the receipt with HMAC-SHA256 over a
  float-free preimage so any language can re-derive it.
- MCP `verify_solution` tool plus dependency-free `verify_sig` helpers and a
  stdlib reference signer service.

The check itself (recompute feasibility against the model + a content-addressed
receipt) is ready to use and needs no secrets; the signing / remote-authority
layer is an explicit proof of concept. Docs: `docs/src/verify.md`.

### Added — Debugger `load` / `sweep` / `multistart`

The interactive solver debugger gained three commands for seeding solves
from externally-computed points and for initialization-sensitivity
diagnostics:

- `load <file> [block]` — the inverse of `save`. Reads a block (default
  `x`) into the live iterate from either a `save` artifact (JSON; every
  block present is loaded) or a plain numeric file
  (comma/whitespace/newline-separated). The many-variable escape hatch:
  generate a start once (`numpy.savetxt`) and `load` it instead of typing
  it. A loaded `x` becomes the seed for the next step / `resolve`.
- `sweep <file>` — run one full solve per start in a file (one per line),
  then tabulate each terminal status / objective, count distinct minima
  (objectives clustered to a relative `1e-6`), and flag the best solve.
- `multistart <N> [rel]` — `N` solves from sampled restarts: each variable
  with a finite box `[x_Lᵢ, x_Uᵢ]` is drawn **uniformly in that box**;
  unbounded variables fall back to a relative jitter `±rel·(|xᵢ|+1)`
  around `x`. Start 0 is the unperturbed point; deterministic (fixed-seed
  PRNG), so runs reproduce. Backed by a new `DebugCtx::var_bounds()` that
  reconstructs full-length algorithm-space bounds (post-scaling, with `±∞`
  for absent bounds) from the NLP's reduced bound vectors + expansion
  matrices.

Tab completion now also covers **filesystem paths** (after
`load`/`sweep`/`save`/`source`, with a trailing `/` on directories) and
block names for `load`'s optional second argument — available both at the
REPL Tab key and via the programmatic `complete` command.

**Ctrl-C at the prompt** is now a working escape hatch: the first press
cancels the current input line (readline convention), a second in a row
stops the solve (a clean `UserRequestedStop`) — mirroring the running-mode
double-tap, so two Ctrl-Cs always exit whether running or paused.

And a little something for the 2am debugging sessions: an undocumented
`coffee` command at the prompt. ☕

Both sweep commands build on the existing re-solve machinery and keep each
solve's trajectory observable (breakpoints/events still fire inside a
sweep). JSON mode emits `sweep_result` per solve and a final
`sweep_summary`; `hello.capabilities` advertises `load` and `sweep`. For
automated global search with dedup and minimum certification, the Python
`find_minima` remains the production path. Docs: `docs/src/debugger.md`
(new "Multi-start and initialization sensitivity" section + scripting
examples).

### Added — Sparse (colored) AD for the JAX front-ends (`sparse=`)

`from_jax` and `JaxProblem` gained a `sparse=True` flag that computes the
constraint Jacobian and the Lagrangian Hessian with CPR-style colored AD
— one JVP/HVP per color (`k ≪ n` colors) scattered back to the detected
nonzeros — instead of materializing the dense matrix and slicing it
(pounce#83). Per-iteration derivative cost drops from `O(n)` to `O(k)`
AD passes on genuinely sparse problems; benchmarked on a banded family at
~560× (Jacobian) / ~200× (Hessian) per eval and 7.6× faster full solve
by `n=2000`. When the sparsity pattern is **value-independent** (any
composition of smooth pointwise ops) the reported structure, values, and
solutions are identical to the dense path; the differentiable backward is
unaffected. For **value-dependent** structure (`where` / `abs` / branches) a
random probe can miss a nonzero, and under compression a missed entry aliases
into a same-colored reported entry — silently wrong derivatives — so such
models should hand-specify the pattern via the `Problem` API or stay on the
dense path. Dense problems see a small bounded overhead, so the flag is opt-in.

- Forward/reverse mode selection (`jacfwd` when `n < m`, else `jacrev`)
  for the dense path / sparsity probe.
- Multi-probe sparsity detection (`n_probes=`, default 3 under
  `sparse=True`, 1 otherwise) unions several random probes to harden
  against value-dependent structure.
- Benchmark: `python/benchmarks/bench_sparse_ad_83.py`. Docs:
  `docs/src/python.md` (JAX integration → "Sparse Jacobian/Hessian
  compression").

### Added — Interactive solver debugger (`--debug` / `--debug-json`)

A "pdb for the interior-point loop." `pounce <problem> --debug` opens a
branded REPL that pauses the solve to inspect and **mutate** live state;
`--debug-json` speaks a newline-delimited JSON protocol so an LLM agent,
a script, or a visual debugger (VS Code DAP / webview) can drive it.
Full guide in `docs/src/debugger.md`. Zero effect on the solve when not
attached.

- **Checkpoints & stepping:** pauses at `iter_start`, the sub-iteration
  phases (`after_mu` / `after_search_dir` / `after_step`), `step_rejected`
  (line search gave up, before restoration), around restoration
  (`pre_/post_restoration_entry/exit`), and `terminated`.
  `step` / `stepi` / `continue` / `run N` / `stop-at <cp>` / `detach` /
  `quit`. The same debugger **steps into the restoration inner IPM**
  (pauses flagged `in_restoration`).
- **Breakpoints:** by iteration (`break N`, one-shot `tbreak N`),
  conditional with `&&`/`||`, on a solver **event** (`break on
  regularized|resto_entered|tiny_step|ls_rejected|mu_stalled|nan`), and
  **watchpoints** (`watchpoint x[3]`). `commands N …` auto-runs a list on
  hit.
- **Inspect:** `info`; `print` of blocks, search-direction blocks (`dx`),
  scalars (`mu obj inf_pr inf_du err compl iter`), `kkt` (inertia +
  regularization), and `active`; `watch`/`display`; `diff`.
- **Named-equation diagnostics:** `print residuals` labels primal/dual
  residuals with their original `.nl` constraint/variable names; `print
  equation <name|row>` renders the source algebra of a named constraint
  (by model name or `.nl` row index); `print rank` reports the SVD
  numerical rank of the equality Jacobian J_c and names the implicated
  rows. `diagnose` (alias `diag`) runs a panel of heuristics over the
  current iterate and emits a **named** health report — *"the worst
  constraint residual is c[mass_balance]"* rather than *"row 13 is
  infeasible"* — the live counterpart to the `pounce-studio` `diagnose`
  tool.
- **Mutate / what-if:** `set mu`, `set x[i]`, `set opt`; `goto`/`restart`
  (soft rewind) and `resolve` (re-solve from the current point).
- **Visualize:** `viz kkt`/`viz L`/`viz <block>` open via `pounce-dbg-viz`
  — an interactive Plotly viewer (spy/heatmap for the KKT matrix & LDLᵀ
  factor, bars for vectors); `save` dumps the iterate. `pip install
  'pounce-solver[viz]'`.
- **Attach & drive:** `--debug-on-error` (post-mortem), `--debug-on-
  interrupt` / Ctrl-C / in-band `{"cmd":"pause"}` (async pause),
  `--debug-script` / `source`, option discovery + Tab completion, `ask`
  (consult an LLM about the paused state; provider-selectable via
  `$POUNCE_DBG_LLM` = `claude` / `codex` / `gemini` / `llm` or a custom
  command template, default Claude Code), and a branded REPL banner
  reusing the project wordmark with a command cheat-sheet.
- **JSON protocol:** `hello` → `pause` → `result` (with `request_id`) →
  `progress` → `terminated`. Engine in `pounce-algorithm::debug`; front
  end in `pounce-cli::debug_repl`.
- **MCP live-debug proxy:** `pounce-studio` exposes the debugger over the
  Model Context Protocol (`debug_start` / `debug_command` / `debug_state`
  / `debug_sessions` / `debug_close`), proxying the `--debug-json`
  protocol so an MCP client can start, drive, and inspect a live solve.

### Added — `read_nl` / `NlProblem` (Python)

`pounce.read_nl(path)` loads an AMPL `.nl` file through pounce's own reader
and returns an `NlProblem` exposing the model's `objective`, `gradient`,
`hessian`, and constraint `jacobian` at any point — the same evaluation
pipeline the solver uses, available standalone for inspection, finite-
difference checks, or feeding another tool. Exported from `pounce`
(`read_nl`, `NlProblem` are in `__all__`).

### Added — Expanded `.nl` opcode coverage

The `.nl` reader now handles conditional/logical opcodes (`if-then-else`,
comparisons), the n-ary list reducers `o11` (MINLIST) / `o12` (MAXLIST), and
the remaining smooth transcendentals (inverse and hyperbolic trig). Models
that previously failed to load with an "unsupported opcode" error now parse,
with FD-verified first/second derivatives on the smooth interior.

> `min`/`max`/`if-then-else` are **non-smooth**: at a kink the gradient is a
> subgradient and the Hessian misses the kink curvature, so an iterate landing
> on or oscillating across the switch can stall the interior-point solve. The
> inverse-trig opcodes (`asin`/`acos`/`atanh`/`acosh`) have **bounded domains**
> whose derivatives blow up at the edge — bound such variables away from the
> boundary. The reader accepts these models; convergence is on you.

### Added — `pounce --cite` and `--minima`

- `pounce --cite [REPORT.json]` lists the citations to use for pounce (and,
  when a solve report is given, any method-specific references it triggered,
  e.g. the Byrd restoration paper). `--bibtex` emits ready-to-paste entries.
- `pounce <problem> --minima` runs the multistart global search from the CLI
  with full `find_minima` parity (method, `n_minima`, dedup, seed).

### Changed

- **Default solver trajectory** moved on several fronts as the interior-point
  method was brought closer to IPOPT. These change which iterates are visited
  (and, on a few problems, the iteration count) but not the math being solved:
  - the barrier parameter `μ` is now updated *inside* the monotone reduction
    loop, so the relaxed-complementarity error reflects the current `μ`. Net
    +2 problems reach Optimal on the internal `.nl` sweep, at a ~2.7% total
    iteration-count cost and a regression on `deconvb` / `gausselm`;
  - under the watchdog, the line search bypasses the acceptor's `alpha_min`
    floor (mirrors IPOPT) so the full-step watchdog trial actually runs;
  - the IPOPT safe-slack bound-adjustment mechanism (`slack_move`) is ported
    and active by default;
  - NLP gradient-based scaling now lifts fixed variables to their value before
    sampling, so the computed scale factors match the operating point.
- **Auto-retry on local infeasibility (default on).** New option
  `feral_infeasibility_scaling_retry` (default `yes`): when a solve ends in
  `Infeasible_Problem_Detected` under a non-MC64 effective scaling, pounce
  re-solves once with `feral_scaling=mc64` (main IPM and restoration sub-IPM).
  This rescues problems where a backward-stable scaling choice lands in a
  spurious infeasible basin under sensitive dependence (`discs.nl` is the
  canonical case); every individual solve along both trajectories is itself
  backward-stable, so an a-priori scaling router can't distinguish them. Set
  to `no` to restore the single-solve behavior.
- **New option `feral_scaling`** (default `auto`, mirrors `feral_ordering`):
  pins FERAL's diagonal KKT scaling strategy; also settable via the
  `POUNCE_FERAL_SCALING` env var.
- **Dependency:** `feral` pinned to crates.io `0.10.0` (was a git rev),
  bringing AMF ordering by default and MC64 inertia-guided scaling fallback.
- **Internal:** the `.nl` pipeline was extracted into a new leaf crate
  `pounce-nl` (re-exported from `pounce-cli`; no public API change).
- **`pounce-studio-mcp` → 0.1.0** (versioned independently of the `0.4.0`
  core): the MCP server graduated from its `0.0.1` spike to its first
  functional release — analyze / run / explain / citations tools, GAMS
  problem tools, a live debug-session proxy, and PyO3 backing via
  `pounce-studio-core`.

### Fixed

- **Windows build:** the debugger's `SIGINT`-to-break handler referenced
  `nix::sys` / `nix::libc`, which the (Unix-only) `nix` crate does not expose
  on Windows, breaking the `pounce-cli` build there. The POSIX handler is now
  `#[cfg(unix)]`-gated with a no-op `install()` stub elsewhere; the rustyline
  prompt's Ctrl-C double-tap remains the cross-platform escape hatch.
- **`.sol` banner no longer goes stale:** the `parse_sol` round-trip test
  fixture derived its `POUNCE <version>:` message from a hardcoded literal,
  which silently drifted on each release (it was still `0.3.1`). It now
  reads `CARGO_PKG_VERSION`, like the production writer always has, so the
  fixture self-updates and never needs a manual bump.
- **Restoration:** the limited-memory (L-BFGS) Hessian is now built in the
  iterates' native space, fixing a space mismatch on compound problems (#102);
  the cycle detector rolls back to the last acceptable point instead of
  erroring out when a usable iterate exists.
- **KKT:** the negative-eigenvalue cache is refreshed on `WrongInertia` /
  `Singular` outcomes (not only `Success`), matching IPOPT's inertia
  pass-through so δ_c regularization routing stays live near a singular KKT (#99).
- **`find_minima`:** the in-bounds test uses a bound-magnitude-relative
  tolerance so large-scale boxes aren't spuriously rejected (#101); MLSL is
  bounded by a sample budget so it always terminates instead of looping when
  its clustering filter rejects every sample (#103).
- **Bounds length is validated up front** across `minimize`, `find_minima`,
  `find_saddles`, `find_critical_points`, `reaction_network`, and `curve_fit`.
  A `bounds` list whose length didn't match the variable/parameter count used
  to fail silently — a too-short list left trailing variables unbounded, and in
  the sampling-based searches a length-1 box could *broadcast* across every
  dimension (sampling all of them from variable 0's interval). It now raises a
  clear `ValueError` immediately, like scipy; `curve_fit`'s scipy-style
  `(lo, hi)` tuple form is likewise checked so array sides must be scalar or
  length-`n_params`.
- **Input validation hardened** so imperfect-but-plausible arguments raise a
  clear `ValueError` up front instead of failing cryptically deep in the solve:
  - `minimize` / `find_minima` / `find_saddles` now **promote a scalar / 0-d
    `x0` to 1-D** (matching scipy), so `minimize(f, 1.5)` works instead of
    raising `iteration over a 0-d array`;
  - a **reversed bound** (`low > high`) is rejected instead of silently
    producing an infeasible box (a fixed `low == high` is still allowed);
  - **malformed constraint dicts** (not a dict, or missing `type` / `fun`, or a
    non-callable `fun`) raise a descriptive error instead of a bare `KeyError`;
  - `curve_fit` validates its data and weights: `xdata`/`ydata` length must
    match and be non-empty and finite, `sigma` must be positive and finite,
    `f_scale` must be positive and finite, and an explicit `p0` must have one
    start per model parameter — each previously surfaced as a `LinAlgError`,
    `ZeroDivisionError`, back-solve `RuntimeError`, broadcast error, or a
    silently wrong fit;
  - a model with **keyword-only parameters** (`f(x, *, a, b)`) — which
    `curve_fit` cannot call positionally as `f(x, *params)` — is rejected with
    a clear message instead of a downstream `TypeError`;
  - `CurveFitResult.confidence_band` checks that `x` has the same
    dimensionality as the fitted `xdata` and that a prediction-band `sigma` is
    scalar or matches `x`, replacing a cryptic einsum/broadcast error;
  - `find_minima` / `find_saddles` reject a sub-1 `n_minima` / `n_saddles` /
    `patience` / `max_solves`, and `find_saddles` rejects a Morse `index`
    outside `[1, n]` (which previously sliced the step vector wrong and found
    the wrong critical points).

## [0.3.0] — 2026-06-02

### Added — Multiple-minima & critical-point global search (PR #94)

`pounce.find_minima(fun, x0, n_minima=..., method=...)` returns several
distinct local minima from a single call. Methods: `flooding` /
`deflation` (add a repulsive Gaussian / pole "hump" — with analytic
gradient and Hessian — at each found minimum and re-solve), `multistart`,
`mlsl`, and `basinhopping`. Anisotropic auto bump widths and a
curvature-derived auto amplitude; Hessian-based saddle rejection; global
box restarts; bounds and constraints pass through `minimize` untouched.
The six-hump-camel demo recovers all six minima.

- `pounce.find_critical_points` / `pounce.find_saddles` — stationary
  points via the squared-gradient merit `½‖∇f‖²`, classified by Morse
  index from the Hessian eigenvalues (degenerate / non-Morse points are
  *flagged*, not mislabeled); eigenvector-following saddle search with
  box-clipped steps.
- `pounce.reaction_network` — minima, transition states, and barriers
  (Müller–Brown example).
- Robustness: non-finite candidates and objectives are rejected before
  acceptance; the de-duplication metric is the same per-dimension scaled
  distance across the minima and saddle routes.
- Examples (`gaussian_hump_minima.py`, `critical_points.py`,
  `reaction_barrier.py`), notebooks 15–17, docs (`find-minima.md`,
  `find-minima-choosing.md`), and 29 unit tests.

### Fixed

- **Acceptable-point termination now rejects a non-finite objective.** A
  near-feasible iterate whose objective evaluates to NaN/Inf (e.g. CUTE
  `himmelbj`) exits `Invalid_Number_Detected` instead of a spurious
  `Solved_To_Acceptable_Level` carrying a `nan` objective; the benchmark
  driver's objective scrape is null-safe to match.
- **No spurious `jacobian()` call on unconstrained problems**
  (`pounce-py`). `eval_jac_g` short-circuits when there are no constraint
  entries (mirroring the Hessian guard), so the unconstrained `minimize`
  facade — which legitimately omits `jacobian` — no longer logs an
  `AttributeError` at ERROR level on every iteration.

### Benchmarks & docs

- The benchmark suite now runs **single-threaded and sequential** by
  default (`OMP/VECLIB/RAYON=1`) so POUNCE and Ipopt solve times are
  directly comparable on one host; the ipopt-ma57 reference was
  regenerated and `BENCHMARK_REPORT.md` carries a threading/timing note
  (POUNCE's `faer`/`rayon` dense linear algebra is up to ~2× faster
  multi-threaded, so the single-threaded numbers are a controlled lower
  bound).
- Example notebooks re-executed against the current solver and corrected
  where the prose had drifted: warm-starting now demonstrates the
  `mu_init` + tight `warm_start_*_bound_push` tuning that actually cuts
  iterations (11 → 5 on HS071); the scaling example uses a `1e12`
  constraint where gradient scaling visibly helps (31 → 11 iters); the
  FBBT notebook shows the domain-safety and false-infeasibility wins.
- mdBook restyled in the POUNCE tiger / cream brand palette.

### Added — Inverse-map ODE recipe over a sensitivity RHS (pounce#91)

`pounce.jax.inverse_map_rhs(jp, dy_ds, *, output=None, x0=None)` builds
the right-hand side of the Alves–Kitchin–Lima inverse / uncertainty
mapping ODE (pounce#84, Eq. 3):

```
dθ/ds = (∂y/∂θ)^{-1} · dy/ds
```

where ``y = output(x*(θ), θ)`` is an output of the embedded optimizer.
POUNCE supplies the RHS; an off-the-shelf adaptive integrator (diffrax,
scipy) does the stepping — *no NLP inversion*.

- The inverse map is a **linear solve against** the total output
  sensitivity ``∂y/∂θ = (∂h/∂x) J + ∂h/∂θ`` (with ``J = ∂x*/∂θ`` from
  the held factor), not a Jacobian-vector product — so it wants the full
  ``J`` and ``jnp.linalg.solve``.
- The whole evaluation rides one `jax.pure_callback`, so the RHS is
  JAX-traceable and composes under `jax.jit` and diffrax (which
  jit-compiles the vector field).
- Worked example `python/examples/inverse_map_diffrax.py` integrates a
  closed output boundary with diffrax Dopri5 and round-trips back through
  the optimizer onto the boundary (~1e-7). `diffrax` is an optional
  extra (`pip install pounce[diffrax]`); the example falls back to RK4
  if it's absent.
- `inverse_map_rhs(..., warm=True)` warm-starts each inner solve from the
  previous evaluation's primal/duals/μ (pounce#86). Result-invariant (up
  to solver tolerance); a *modest* lever — ~1.4-1.7× fewer IPM iterations,
  ~1.3× wall-clock, roughly flat in problem size (interior-point
  warm-start ceiling + per-eval Jacobian-build overhead). Benchmark:
  `python/benchmarks/inverse_map_warm.py`. For a real speedup on a smooth
  map, prefer `PathFollower` (it skips solves, not just cheapens them).
- Switch to `PathFollower` when the path folds or the active set changes.
- Worked notebook `python/notebooks/14_path_following.ipynb` tours the
  whole family (sensitivity → margin → continuation → fold → inverse map).

Also fixes the build-once / stacked path for **unconstrained** problems
(``g=None``, ``m=0``): the constraint callbacks no longer dereference the
(``None``) constraint-Jacobian jit, so `JaxProblem.solve` /
`solve_with_jacobian` / the batched solves now work with no constraints.

### Added — Predictor–corrector path-following engine (pounce#90)

`pounce.jax.PathFollower` traces a solution path of a parametric NLP by
*composing* the post-solve sensitivity primitives instead of re-solving
at every step:

```python
from pounce.jax import PathFollower
pf = PathFollower(jp, monitor_tol=1e-6, ds0=0.05)
trace = pf.follow(theta_of_s, (0.0, 1.0), x0)   # parameter continuation
# trace.x, trace.theta, trace.s, trace.lam,
# trace.n_correctors, trace.n_accepts, trace.active_set_changes
```

- **predict** — extrapolate primal *and duals* along the held-factor
  sensitivity (`jvp_from_state(..., with_duals=True)`); **monitor**
  (no solve) — KKT residual + active-set margin (#89) at the predicted
  point; **correct** — only when the monitor trips, a warm-μ re-solve
  that also re-anchors the factor in one solve (`warm_anchor`, #86).
- Adaptive step size; detects and records active-set changes and
  re-anchors on the new active set.
- `PathFollower.trace_arclength(...)` — pseudo-arclength continuation for
  a scalar-parameter, equality/unconstrained family, tracing **past
  folds** where `∂x*/∂θ` is singular (parameter continuation cannot).
  Reports turning points. Bifurcation/branch-switching and
  inequality-active folds are out of scope for v1.
- On a linear-response NLP the predictor is exact, so the whole path is
  traced with **zero correctors** (one anchor solve vs one cold solve
  per step); nonlinear paths correct adaptively and still trace to
  tolerance.

New supporting public surface:

- `JaxProblem.warm_anchor(p, x0, *, duals=None, mu=None)` — a warm-started,
  μ-seeded re-solve that pins the converged factor and returns a `B=1`
  `AnchorState` (the corrector + anchor in one solve). Threads μ through
  the reusable build-once path (the #86 follow-up).
- `JaxProblem.jvp_from_state(..., with_duals=True)` /
  `batched_jvp_from_state(..., with_duals=True)` — also return the dual
  sensitivity `∂λ*/∂θ · dp` from the same held-factor back-solve.

### Added — Active-set-proximity monitor (pounce#89)

`JaxProblem.active_set_margin(state)` reports the distance to an
active-set change at the anchor point — the "predictor is about to
become invalid" signal for predictor–corrector path following. The
post-solve sensitivity is a derivative on a *fixed* active set; this
flags when a bound / inequality is about to cross its critical-region
boundary (where the sensitivity is discontinuous).

```python
r = jp.active_set_margin(state)
# r["margin"], r["min_mult"], r["min_slack"]  — each (B,)
```

- By complementarity: an **active** bound/inequality (multiplier `>
  active_tol`) is about to leave the set — its *multiplier* heads to
  zero; an **inactive** one is about to enter — its *slack* heads to
  zero. `min_mult` / `min_slack` track each; `margin = min(min_mult,
  min_slack)`.
- Equalities (`cl == cu`) are excluded (always active); `±inf` bounds
  and the slack side of a one-sided inequality drop out naturally.
  An unconstrained interior point returns `inf`.
- Pure-JAX reduction over state the `AnchorState` already holds — no
  solve, no back-solve. Pairs with the caller-side KKT-residual
  (smooth-drift) monitor: re-anchor when either trips.

### Added — Single-problem ergonomic sensitivity wrappers (pounce#88)

Thin un-batched wrappers over the `batched_*` post-solve sensitivity
methods, for the scalar / path-following user (one NLP at a time):

```python
x_star, (lam, zL, zU), J = jp.solve_with_jacobian(theta, x0)   # J: (n, p)
state = jp.anchor(theta, x0)             # un-batched point → B=1 state
J  = jp.sensitivity(state)               # (n, p) from the held factor
dx = jp.jvp_from_state(state, dtheta)    # J @ dtheta  -> (n,)
dp = jp.vjp_from_state(state, x_bar)     # J^T @ x_bar -> (p,)
```

- `solve_with_jacobian` / `sensitivity` / `jvp_from_state` /
  `vjp_from_state` accept and return un-batched shapes, delegating to
  the batched methods with `B=1` and squeezing — no new numerics.
- `anchor` now accepts a single un-batched point (`p_shape`) in addition
  to a batch (`(B,) + p_shape`); a single point yields a `B=1`
  `AnchorState`. The single-problem from-state wrappers reject a `B>1`
  state rather than silently mis-shaping.
- Implemented as `JaxProblem` methods (mirroring the `batched_*` names)
  rather than free functions, for consistency with the existing surface.

### Added — Exact post-solve sensitivity at a supplied point (pounce#87)

`JaxProblem.sensitivity_at(x_star, theta, duals, *, wrt_cols=None)`
returns the exact primal sensitivity `∂x*/∂θ` evaluated at a
caller-supplied primal-dual point, by re-assembling and factoring the
KKT system *there* — no IPM re-solve.

```python
J = jp.sensitivity_at(x_star, theta, (lam, zL, zU))   # (n, p_dim)
```

- **Re-factor, not reuse.** A held FERAL factor encodes the anchor
  point's `H` / `J`, so back-solving it at a moved `x_star` gives a
  first-order-stale sensitivity. `sensitivity_at` assembles the dense
  `(n+m)×(n+m)` KKT at the supplied point, which is exact there
  (assuming a KKT point for `theta`). The cheap-but-local reuse path
  stays as the predictor `batched_jvp_from_state`; this is its
  exact-refresh complement.
- Active set is read from the supplied bound multipliers `(zL, zU)`,
  exactly like the `custom_vjp` backward — the caller passes the duals
  the anchoring solve / `solve_with_warm` returned at this point.
- Pure-JAX, so itself differentiable (second-order sensitivities work);
  matches `jax.jacobian` over a fresh solve to ~1e-6 at every point
  along a swept path, including a binding bound.

This is the exact-refresh primitive for the inverse map, where `x*`
traces a known output boundary and the sensitivity must be evaluated at
the known point without paying a full re-solve per RK stage.

### Added — Barrier-μ warm start for predictor–corrector correctors (pounce#86)

The interior-point barrier parameter μ is now reported on every solve and
can be threaded into a warm-started re-solve, so a predictor–corrector
corrector resumes near the central path instead of re-walking the barrier
homotopy from the default initial μ.

- **`info["mu"]`** — every `Problem.solve` / `Solver.solve` /
  `solve_with_sens` info dict now carries the converged barrier parameter
  (`0.0` on the barrier-free SQP path).
- **`pounce.jax.solve_with_warm`** accepts a 4-element warm-state
  `(lam, zL, zU, mu)` that seeds `mu_init` / `warm_start_target_mu`, and
  returns the converged μ in a matching 4-tuple. The 3-tuple form is
  unchanged; passing `mu=None` inside a 4-tuple reports μ out without
  seeding it in. Differentiability w.r.t. `p` is preserved (the μ
  input/output are stop-gradient, like the duals).

On a small parametric NLP, seeding μ from the previous solve's converged
barrier cut a warm-started corrector from 5 interior-point iterations to
1 (same optimum). The `mu_init` / `warm_start_target_mu` algorithm
options already existed; this exposes the converged μ needed to drive
them along a path.

### Added — Post-solve Jacobian / sensitivity API from the held KKT factor (pounce#82)

`JaxProblem` now exposes a first-class post-solve sensitivity surface
that reuses the held FERAL stacked KKT factor instead of round-tripping
through `jax.vjp` / `jax.jacrev`:

```python
x_star, (lam, zL, zU), J, state = jp.batched_solve_with_jacobian(
    p_batch, x0,
    wrt_cols=slice(0, ny),   # optional parameter-column selection (1-D p)
    return_state=True,
)
dp_bar = jp.batched_vjp_from_state(state, x_bar)   # J^T @ x_bar
state.close()
```

- **`batched_solve_with_jacobian(...)`** returns the full per-block
  primal Jacobian `J` of shape `(B, n, p_dim)` (or `(B, n, len(wrt_cols))`)
  alongside `x_star` and the `(lam, zL, zU)` duals (same contract as
  `batched_solve_with_warm`). The Jacobian is assembled by evaluating the
  existing factor-reuse backward over the `n×n` identity output basis —
  one multi-RHS `Solver.kkt_solve_many` against the held LDLᵀ factor, no
  NLP re-solve.
- **`anchor(p_batch, x0, *, wrt_cols=None)`** solves once and pins the
  factor, returning an **`AnchorState`** handle for reuse across several
  post-solve sensitivity calls (linear-update pattern).
- **`batched_vjp_from_state(state, x_bar)`** is the public reverse-mode
  product `Jᵀ x̄` against a held factor.
- **`batched_jvp_from_state(state, dp)`** is the forward-mode product
  `J @ dp` — the cheap path for linear updates that never materialise the
  full `J`. It assembles the parameter-side RHS `[∂²L/∂x∂p · dp;
  ∂g/∂p · dp]` into the compound x- and constraint-blocks and back-solves
  once against the held factor. Accepts a reduced `dp` when the state was
  anchored with `wrt_cols`.
- **`AnchorState`** lifetime: works as a context manager
  (`with jp.anchor(...) as state:`) *and* supports explicit ownership
  (`state.close()`, `state.reanchor(...)`) for handles that outlive a
  lexical block. Pinned factors are exempt from the LRU but capped
  (`_pinned_capacity`, default 16) with a loud overflow error, and a
  `weakref` finalizer reclaims the factor if a handle is dropped without
  `close()`.

### Added — Structured logging + colored iteration table (pounce#71)

POUNCE now emits diagnostics through the
[`tracing`](https://docs.rs/tracing) ecosystem and renders the
per-iteration table in a tiger/rust branded color theme.

- **Colored iteration table.** Restoration lines take a background that
  varies by restoration kind (soft-stay → tan, soft-exit → amber, hard →
  deep rust); the row text shades smoothly from black toward red as the
  primal step length `alpha` shrinks (a visual stalling cue, shifted to
  cream → bright-yellow on the dark restoration backgrounds). Color is
  emitted only when stdout is a terminal — redirected output and
  `NO_COLOR` get plain text with identical column alignment.
- **Structured logs.** Solver-internal diagnostics, warnings, and
  developer instrumentation are now `tracing` events under namespaced
  targets (`pounce::algorithm`, `pounce::linsol`, `pounce::mu`,
  `pounce::sqp`, `pounce::linesearch`, `pounce::restoration`,
  `pounce::presolve`, `pounce::py`). Logs go to **stderr**; program
  output (iteration table, summary, `--dump`) stays on **stdout**.
- **Spans.** `solve`, `iteration`, `linear_solve`, and `restoration`
  spans tag nested events with context.
- **New environment variables:** `RUST_LOG` (verbosity / per-target
  filtering, default `info`), `POUNCE_LOG_FORMAT=text|json` (JSON sink on
  stderr, including the per-iteration `pounce::iteration` stream for
  Studio / CI), `NO_COLOR` / `CLICOLOR_FORCE` (color policy). Documented
  in `docs/src/options.md` and `docs/src/troubleshooting.md`.
- New `pounce-observability` crate (subscriber install + iteration
  collector) and a `pounce-common::style` palette module.
- A `log` → `tracing` bridge (`tracing_log::LogTracer`) so any remaining
  `log::*` call sites — chiefly transitive dependencies — surface through
  the subscriber and obey `RUST_LOG`.
- **Branded CLI header.** The `pounce` banner now renders a molten
  tiger/rust POUNCE logo (terminal-only; `NO_COLOR` / non-TTY get plain
  text).

### Changed

- Per-iteration JSON solve-report data is now sourced from the
  `pounce::iteration` tracing event (via an in-process collector layer)
  rather than an in-loop accumulation; the report contents are
  unchanged. Capturing iteration history requires the tracing subscriber
  installed by the CLI / Python / C frontends (or
  `pounce_observability::init_for_tests()` in tests).
- Bumped the `feral` linear-algebra dependency from 0.8.0 to 0.9.0.

### Removed

- Dropped the direct `log` crate dependency in favor of `tracing`.

### Added — Active-set SQP with working-set warm start (Phase 5b + 5c + 5d)

A new sequential-quadratic-programming driver sits alongside the
existing interior-point method, opt-in via a single option flip.
Designed for **warm-started NLP sequences** (MPC, parametric
continuation, homotopy sweeps), where the previous solve's active
set is a strong starting point.

**Tutorial:** `docs/src/active-set-sqp.md`.
**Python notebook:** `python/notebooks/06_sqp_parametric_continuation.ipynb`.
**C example:** `crates/pounce-cinterface/examples/sqp_warm_start.c`.
**GAMS example:** `gams/examples/parametric_sqp_warm_start.gms`.
**Design note:** `docs/src/active-set-sqp-warm-start.md`.

#### Algorithm selection (cross-cutting)

- New top-level option `algorithm`, values `interior-point`
  (default; existing IPM path) and `active-set-sqp` (new SQP driver).
  Settable through every interface — `add_option` in Rust /
  Python, `AddIpoptStrOption` in C, `pounce.opt` in GAMS — exactly
  like `linear_solver` already is.

#### SQP suboptions (`sqp_*` namespace)

`sqp_globalization` (`filter` | `l1-elastic`),
`sqp_hessian` (`exact` | `damped-bfgs` | `lbfgs`),
`sqp_max_iter`, `sqp_tol`, `sqp_constr_viol_tol`,
`sqp_dual_inf_tol`, `sqp_l1_penalty`, `sqp_l1_penalty_safety`,
`sqp_l1_penalty_max`, `sqp_bt_reduction`, `sqp_bt_min_alpha`,
`sqp_print_level`, `sqp_lbfgs_max_history`. Defaults mirror
`SqpOptions::default()`. Each is "only consulted when `algorithm`
is `active-set-sqp`"; the IPM path ignores them silently.

#### Python — `pounce.Problem`

New keyword argument and methods:

```python
prob.add_option("algorithm", "active-set-sqp")
x, info = prob.solve(x0, working_set=ws)
ws = info["working_set"]      # always present; None on the IPM path
ws = prob.get_working_set()
prob.set_working_set(ws)
prob.clear_working_set()
```

The `working_set` value is a 2-tuple `(bounds, constraints)` of
numpy int8 arrays with status codes 0..=3 (Inactive / AtLower /
AtUpper / Fixed-or-Equality). Module-level helper
`pounce.classify_working_set(x, x_l, x_u, g, g_l, g_u, lambda_g,
z_l, z_u, m_eq, ...)` classifies an IPM-converged iterate
into a WS suitable for `Problem.solve(working_set=…)`.

#### C ABI — four new entry points

```c
Bool IpoptGetWorkingSet(IpoptProblem, IpoptBoundStatus*, IpoptConsStatus*);
Bool IpoptSetWarmStartWorkingSet(IpoptProblem, const IpoptBoundStatus*, const IpoptConsStatus*);
Bool IpoptClearWarmStartWorkingSet(IpoptProblem);
enum ApplicationReturnStatus IpoptSolveWarmStart(
    IpoptProblem, ipnumber *x, *g, *obj_val, *mult_g, *mult_x_L, *mult_x_U,
    const IpoptBoundStatus *bound_in,
    const IpoptConsStatus  *cons_in,
    IpoptBoundStatus       *bound_out,
    IpoptConsStatus        *cons_out,
    UserDataPtr user_data);
```

Plus typedefs `IpoptBoundStatus`, `IpoptConsStatus` and the four
status constants `POUNCE_WS_INACTIVE` (= 0), `POUNCE_WS_AT_LOWER`
(= 1), `POUNCE_WS_AT_UPPER` (= 2), `POUNCE_WS_FIXED_OR_EQ` (= 3).
**No existing C entry-point signature changed** — cyipopt / JuMP /
AMPL clients link unchanged.

#### GAMS solver link

Two mechanisms ship in tandem:

- **§7.4(a) marginal-based reconstruction** (default, no
  configuration). The solver link reads variable and equation
  marginals (`x.m`, `con.m`) at the top of every `pouCallSolver`
  invocation and reconstructs the SQP working set automatically.
  Lossy at degenerate active sets — same idiom as CONOPT, IPOPT,
  KNITRO under GAMS.
- **§7.4(b) persistent state file** (opt-in via
  `sqp_state_file <path>` in `pounce.opt`). A small binary blob
  with FNV-1a checksum keyed by `(n, m, x_l, x_u, g_l, g_u)` so
  structural changes invalidate cleanly. Falls back to §7.4(a) on
  any read failure.

#### Sensitivity (`pounce-sensitivity`)

`SensResult` now carries the converged user-space multipliers
(`mult_g`, `mult_x_L`, `mult_x_U`) and constraint values (`g`),
so the parametric "predictor + SQP corrector" pattern is a single
`SensSolve::run` followed by one `classify_working_set` call.

#### Hessian sources

The `sqp_hessian` option selects between three implementations:

- `exact` — uses `eval_h`; pounce-qp's inertia control handles
  indefiniteness via diagonal-shift retry (§4.5).
- `damped-bfgs` — Powell-damped rank-2 BFGS, dense `n×n`,
  guaranteed PSD (Powell 1978).
- `lbfgs` — limited-memory BFGS with circular history, default
  6 pairs (matches IPOPT's `limited_memory_max_history`),
  materialized to dense Triplet at QP-solve time.

#### Globalizations

`sqp_globalization` selects the SQP outer-loop step-acceptance
test:

- `filter` (default) — Fletcher-Leyffer 2002 Pareto-frontier
  filter on `(constraint violation, objective)`. No penalty
  parameter; recommended general default.
- `l1-elastic` — Han-Powell merit `φ(x; ν) = f(x) + ν · violation(x)`
  with adaptive ν clamped by `sqp_l1_penalty_safety` /
  `sqp_l1_penalty_max`. SNOPT-style behaviour.

### Added — `feral_ordering` option (FERAL fill-reducing ordering)

User-facing knob for the FERAL backend's fill-reducing ordering. New
string option `feral_ordering` accepts `auto` (default; feral's
adaptive dispatcher — picks AMD / AMF / MetisND from cheap pattern
features), `auto_race` (runs symbolic factorization on AMD, MetisND,
ScotchND, KahipND and keeps the smallest factor_nnz; ~4× a single
symbolic pass, amortized across numeric refactorizations), and the
concrete methods `amd`, `amf`, `metis`, `scotch`, `kahip`. Settable
through every interface that consumes `pounce.opt` /
`OptionsList` — Rust, Python, C, GAMS, CLI — and also via the
`POUNCE_FERAL_ORDERING` environment variable for option-free
callers. Reuses the same explicit-set semantics as the other
`feral_*` options: leaving it unset keeps the `FeralConfig::from_env`
default (`Auto`).

The motivating case is `pinene_3200_0009`, where the cheap `Auto`
heuristic picks MetisND (88 s numeric) but AMD factors in 19.5 s on
the same matrix; `feral_ordering auto_race` measures both and lands
on the winner without per-problem manual tuning. See
`docs/src/options.md` "FERAL backend tuning" and
`docs/src/troubleshooting.md` for guidance.

### Added — AMPL imported (external) function support (issue #49)

`.nl` files that declare imported functions in their `F` segments
and call them via `f<id> <nargs>` tokens are now solved end-to-end.
Set `AMPLFUNC` to a newline-separated list of shared-library paths;
pounce loads each library via the standard AMPL `funcadd_ASL` ABI,
binds every referenced funcall id to a `(library, name)` pair, and
emits `TapeOp::Funcall` nodes that participate in full forward /
reverse / Hessian sweeps (first- and second-derivative requests
are issued back through the library on demand, with the packed
upper-triangular Hessian indexed as `hes[lo + hi*(hi+1)/2]`).

Tested against the IDAES `general_helmholtz_external.dylib`
fixture from the issue report — pounce reaches
`EXIT: Optimal Solution Found` on the 3-variable Helmholtz
problem. Without `AMPLFUNC` set, problems that need external
functions fail with a clear error naming the offending function
and pointing at `AMPLFUNC`.

Limitations: only the `Tape` (default) AD path supports external
functions. The `HybridTape` partial-separability path and the
JIT-style `HessianProgram` path panic on `TapeOp::Funcall` — both
are alternative routes not on `NlTnlp::new`'s critical path, so
the current production flow is unaffected.

### Added — Phase 5a `pounce-qp` crate

Standalone sparse parametric active-set QP solver. Drives the
SQP subproblem solves; also exposed as a standalone crate
(`pounce_qp::ParametricActiveSetSolver`). Implements
Gill-Murray-Saunders elastic mode (§4.3), full GMSW EXPAND
anti-cycling (§4.4), Bunch-Kaufman inertia control via
diagonal-shift retry (§4.5), iterative refinement (§4.7), and
Sherman-Morrison-Woodbury Schur-complement factor updates (§4.2,
opt-in via `QpOptions::use_schur_updates`).

### Added — In-repo regression fixtures

- `crates/pounce-algorithm/tests/hock_schittkowski_subset.rs` —
  10 HS problems with published closed-form optima.
- `crates/pounce-qp/tests/mm_published_optima.rs` —
  Maros-Mészáros-flavoured framework with 5 fixtures + reusable
  `compare_qps_to_published(text, x*, f*, …)` helper.
- `crates/pounce-algorithm/tests/parametric_sqp_corrector.rs` —
  IPM → classify_working_set → SQP corrector end-to-end.
- `crates/pounce-algorithm/tests/sqp_filter_vs_l1_elastic.rs` —
  parity between the two globalizations.

### Added — Auxiliary-equality preprocessing (Phase 0 presolve, issue #53)

A 14-PR series that scaffolds an opt-in *Phase 0* presolve pass:
detects block-triangular structure in the equality system, solves
the dependent blocks ahead of the IPM, and substitutes the
recovered variables back into the user TNLP. Targets gas-network,
power-flow, and process-design problems where a few hundred
algebraic state variables eliminate cleanly.

The algorithm and reference implementation are a port of
[ripopt PR #32](https://github.com/jkitchin/ripopt/pull/32) by
**David Bernal Neira** ([@bernalde](https://github.com/bernalde)).
The ripopt work also vendored the
`tutorial_flow_density{,_perturbed}.nl` and `gaslib11_steady.nl`
fixtures we now use for end-to-end testing.

- Hopcroft-Karp incidence matching, Dulmage-Mendelsohn decomposition,
  Tarjan SCC → block-triangular form.
- Coupling classification (linear / nonlinear / inequality-coupled)
  plus a damped-Newton block solver with large-block fallback.
- Trivial-elimination pre-pass; inequality-coupled blocks handled
  by projection.
- Reduction-frame bookkeeping with full multiplier recovery so
  `final_zL` / `final_zU` round-trip back to the user space.
- Orchestrator wired into `PresolveTnlp`, gated by
  `presolve_auxiliary` (default off). Diagnostics surfaced via
  `presolve_auxiliary_diagnostics`.
- Design note: `dev-notes/auxiliary-equality-preprocessing.md`;
  user docs in `docs/src/auxiliary-presolve.md`.

### Added — FBBT (Feasibility-Based Bound Tightening, #62)

Three-commit landing of FBBT inside `pounce-presolve`:

- `pounce-presolve::interval` — outward-rounded interval arithmetic
  on `f64`, with `Interval::div` reciprocal endpoints rounded
  outward (fixes a subtle near-zero straddle case discovered in
  review).
- `ExpressionProvider` trait + forward pass walks each constraint
  expression and tightens variable bounds from the constraint's
  `g_l`/`g_u` envelope.
- Reverse propagation + orchestrator wired through `PresolveTnlp`
  end-to-end. New options: `presolve_fbbt` (master switch,
  default off), `fbbt_tol`, `fbbt_max_iter`, `fbbt_max_constraints`.
- Docs: `docs/src/fbbt.md`; demo notebook
  `python/notebooks/08_fbbt.ipynb`.

### Added — Problem and KKT-system scaling (#61, f00c1f9)

End-to-end wiring of the upstream `nlp_scaling_*` and
`linear_system_scaling` option families:

- `nlp_scaling_method`: `none` / `user-scaling` (new — pulled from
  `set_problem_scaling` Python API or `SetIpoptProblemScaling`
  C API) / `gradient-based` (existing, now with target-gradient
  knobs `nlp_scaling_obj_target_gradient` and
  `nlp_scaling_constr_target_gradient`).
- `linear_system_scaling`: `none` / `mc19` / `ruiz` (iterative
  symmetric infinity-norm equilibration, new) / `slack-based`.
  Applied to the augmented system independent of NLP-level
  scaling.
- Python `Problem.set_problem_scaling(obj_scaling, x_scaling=None,
  g_scaling=None)` plus a worked example in
  `python/notebooks/07_scaling.ipynb`.
- Documentation: `docs/src/scaling.md`.

### Added — Mehrotra adaptive-μ defaults and init cascade (upstream parity)

- `mehrotra_algorithm` option routed through `PdSearchDirCalc`
  (previously parsed but inert).
- `adaptive_mu_globalization` cascade finished per upstream Ipopt;
  `bound_push` / `bound_frac` / `bound_mult_init_val` / `alpha_for_y`
  cascade from `mehrotra_algorithm yes`.
- `least_square_init_primal` implemented in
  `DefaultIterateInitializer`.
- `accept_every_trial_step` honored in the line search and
  cascaded from `mehrotra_algorithm` (matches upstream
  initialization behavior).

### Added — FERAL backend tunables and 0.8.0 bump

- `feral_pivtol` exposed as an `OptionsList` option with
  `FERAL_PIVTOL` environment-variable fallback.
- Tri-state `cascade_break` (#55): `auto` / `on` / `off`, inheriting
  the FERAL Phase B default unless explicitly set.
- Workspace bump to `feral 0.8.0`, which ships the SSIDS-aligned
  strict-zero-pivot inertia policy (feral gh#54 / pounce gh#52,
  *nuffield2_trap*). The temporary `[patch.crates-io]` block
  pointing at the local feral checkout has been removed.

### Added — `pounce-solve-report` crate + `IpoptWriteSolveReport` C API

- New publishable crate `pounce-solve-report` (first crates.io
  release) emits the machine-readable `pounce.solve-report/v1`
  JSON shared by the CLI, the C ABI, and the GAMS driver.
- C ABI: `IpoptWriteSolveReport(IpoptProblem, const char *path)`
  writes the report to disk after `IpoptSolve`.
- GAMS driver now emits `pounce.solve-report/v1` alongside the
  `.lst` so studio tooling can consume it directly.

### Added — Diagnostics dumps

- `--dump iterates:{summary,full}` (#68) — per-iteration trajectory
  artefacts the studio can replay. `summary` writes one JSON line
  per outer iteration; `full` adds the primal/dual vectors and
  KKT residuals.
- `--dump kkt:*+L` (#69) — augments the existing KKT-system dump
  with the LDLᵀ factor pattern (block structure, fill-in, pivot
  signs) for inertia post-mortems.
- `print_options_documentation yes` now actually walks the
  registered options and emits a categorized dump (previously a
  registered-but-inert toggle).

### Added — Studio Claude-skill and MCP GAMS tools

- `studio/skill/` — Claude-skill front-end as an alternative to the
  MCP server. Lighter-weight install path for users who just want
  the studio prompts and don't need an MCP runtime.
- `studio/mcp` — new GAMS problem tools (`run_gams_problem`,
  `analyze_gams_problem`, `parse_gams_listing`,
  `list_gams_examples`) plus an install script.

### Added — Parallel batched `pounce.jax.vmap_solve_parallel` + GIL release (pounce#74)

`pounce_py::Problem::solve` now releases the Python GIL across the
`optimize_tnlp` call (every TNLP callback was already
`Python::with_gil`-wrapped, so this is a localized
`py.allow_threads` block in `crates/pounce-py/src/problem.rs`).
That unlocks true concurrent IPM iteration across independent
`Problem` instances on different OS threads — Python-level
`f` / `g` callbacks still serialize on the GIL but the linear-algebra
heart of the solver runs in parallel.

`pounce.jax.vmap_solve_parallel` rides that change: a drop-in
replacement for `vmap_solve` that dispatches the batch over a
`ThreadPoolExecutor` of independent `Problem` instances. Forward
is parallel via the threadpool; backward is `jax.vmap` over the
per-element KKT solve (pure JAX, vectorizes naturally).

```python
from pounce.jax import vmap_solve_parallel

x_batch = vmap_solve_parallel(
    p_batch, f=f, g=g, x0=x0, n=n, m=m,
    lb=lb, ub=ub, cl=cl, cu=cu,
    workers=8,  # default: min(B, 8)
)
```

Microbench (`n=30`, `B=16`, nonlinear unconstrained, M1 8-core):
`vmap_solve` 1.00s → `vmap_solve_parallel(workers=8)` 0.37s
(~2.75×). Speedup grows with per-element solve cost. Numerically
identical to the sequential reference.

### Added — `pounce.jax.solve_with_warm` (pounce#74)

Companion to `pounce.jax.solve` that threads the previous solve's
dual triple `(mult_g, mult_x_L, mult_x_U)` into the next call via
IPOPT's `warm_start_init_point=yes` machinery.

```python
from pounce.jax import solve_with_warm

x_star, warm = solve_with_warm(
    p, f=f, g=g, x0=x0, n=n, m=m,
    lb=lb, ub=ub, cl=cl, cu=cu,
    warm_start=None,                # cold first call
)
for p_k in trajectory[1:]:
    x_star, warm = solve_with_warm(
        p_k, f=f, g=g, x0=x_star, n=n, m=m,
        lb=lb, ub=ub, cl=cl, cu=cu,
        warm_start=warm,            # threaded duals
    )
```

Differentiable w.r.t. `p` via the same implicit-function rule as
`solve`. Cotangents on the warm-state outputs and the warm-state
inputs are dropped (zero) — at the optimum the duals are a
function of `p` and the active set, not an independent input to
`dx*/dp`. `solve` itself is unchanged (non-breaking).

### Added — `pounce.jax.JaxProblem` build-once/solve-many handle (pounce#75)

Iterative outer loops (differentiable constrained layers in a
training step, parametric sweeps) were paying a ~45ms rebuild on
every call to `pounce.jax.solve` / `vmap_solve_parallel` /
`solve_with_warm` — re-JIT of `jax.grad`/`jacrev`/`hessian`, the
one-shot random sparsity probe, plus a fresh `pounce.Problem`
construction — versus a ~3ms underlying solve. On `n=5, m=6`
problems that's a ~14× wrapper overhead.

`JaxProblem` is a build-once handle: do the JIT and sparsity probe
in `__init__`, then expose `.solve(p, x0)`, `.solve_with_warm(p, x0,
warm)`, `.vmap_solve(p_batch, x0)`, and `.vmap_solve_parallel(p_batch,
x0, workers=)` as methods that reuse the prebuilt state across
calls. Each worker thread in `vmap_solve_parallel` keeps its own
cached `pounce.Problem` via `threading.local` so the build cost is
paid at most once per worker (typically `min(B, 8)` total) rather
than `B` times per batch.

```python
from pounce.jax import JaxProblem

jp = JaxProblem(
    f=f, g=g, n=n, m=m, p_example=p0,
    lb=lb, ub=ub, cl=cl, cu=cu,
    options={"tol": 1e-9, "print_level": 0},
)
for p_k in trajectory:
    x_star = jp.solve(p_k, x0=x_prev)
    x_prev = x_star
```

Microbench on the issue's `n=5, m=6` shape — 20 sequential solves at
different `p`:

```
top-level solve   (20 calls): 1.914s  → 95.7ms/solve
JaxProblem.solve  (20 calls): 0.136s  → 6.8ms/solve
speedup: 14.1x
```

Existing top-level `solve` / `vmap_solve` / `vmap_solve_parallel` /
`solve_with_warm` are unchanged (non-breaking) — `JaxProblem` is a
new surface for performance-sensitive iterative use.

### Added — `JaxProblem` factor-reuse backward (k_aug-style; pounce#76)

The `custom_vjp` backward of `JaxProblem.solve` /
`solve_with_warm` no longer assembles a dense
`(n+m) × (n+m)` KKT block in JAX and runs `jnp.linalg.solve` on it.
Instead it reuses the IPM's converged compound KKT factor through
`pounce.Solver.kkt_solve` — the same factor [k_aug] uses for
parametric sensitivity. Two wins:

* **Perf.** The dense back-solve is O((n+m)³) on every bwd call;
  reusing the held LDLᵀ factor makes it O(nnz(L)). For modest `n`
  the absolute savings are small; for `n+m` in the hundreds-to-
  thousands it dominates the bwd.
* **Correctness.** The compound block's bound-multiplier rows
  `(z_l, z_u)` already encode active-set behaviour — at convergence
  active bounds have unbounded `z` (forces `Δx_i = 0` in the
  back-solve), inactive bounds have `z ≈ 0` (leaves `Δx_i` free).
  Slack inequality rows in the user's `g` are handled the same way
  by `(v_l, v_u)`. The factor-reuse path therefore drops the
  explicit active-set masking the dense path does on `H` / `J` / `v`;
  accuracy is `O(μ)` at the IPM barrier parameter, well below `tol`
  after convergence.

Behaviour change: `JaxProblem(factor_reuse=True)` is the default. Set
`factor_reuse=False` for a verbatim fallback to the pre-#76 dense
backward (useful for higher-order differentiation, since the dense
backward stays JAX-traced and is itself differentiable).

Plumbing:

* `pounce.Solver` exposes a new `block_dims` getter returning the
  `(n_x, n_s, n_y_c, n_y_d, n_z_l, n_z_u, n_v_l, n_v_u)` layout of
  the compound KKT vector so the JAX bwd can pack a partial RHS
  (just the x-block) and unpack `u_x` / `u_y_c` / `u_y_d`.
* Each fwd registers its converged `Solver` in a bounded-LRU cache
  on the `JaxProblem` (default capacity 128, exposed as
  `clear_solver_cache()` for early eviction). LRU rather than
  pop-on-read because `jax.jacobian` calls the bwd N times per
  fwd; pop semantics would crash from the second direction onward.
* The back-solve `pure_callback` uses
  `vmap_method="sequential"` so `jax.jacobian` / `jax.vmap` of a
  loss-gradient correctly iterate one cotangent at a time across
  the impure host call.

The standalone `pounce.jax.solve` / `vmap_solve_parallel` /
`solve_with_warm` keep the dense backward for now.

[k_aug]: https://github.com/dthierry/k_aug

### Added — `JaxProblem.batched_solve` stacked block-diagonal solve (pounce#76 (A))

`JaxProblem.batched_solve(p_batch, x0)` runs one IPM solve over a
single NLP whose variables are `[x^(1); ...; x^(B)]`, constraints are
`concat(g(x^(k), p^(k)))`, and objective is `Σ_k f(x^(k), p^(k))`.
The Jacobian and Lagrangian Hessian are block-diagonal (no
cross-block coupling, since each block-`k` constraint touches only
the block-`k` slice of `X` and the objective is a pure sum), so the
IPM sees one big sparse problem but spends linear-system work
proportional to `B × (per-block factor cost)`.

Complementary to the existing batched surfaces:

* `vmap_solve` — sequential `jax.lax.map`, one solve per iterate.
* `vmap_solve_parallel` — B independent IPMs in a
  `ThreadPoolExecutor` (GIL released per solve). Wins when batch
  elements have very different convergence behaviour.
* `batched_solve` — one stacked IPM. Wins when blocks have similar
  convergence behaviour (shared barrier homotopy and shared
  symbolic factorisation amortise across the batch) and when B is
  large enough that the per-call Python overhead of B fwd
  dispatches becomes visible — one Rust crossing instead of B.

`custom_vjp`-wrapped: `jax.grad` / `jax.jacobian` through
`batched_solve` work end-to-end. The bwd vmaps the per-element
dense KKT back-solve, which is exact because the block-diagonal
coupling means `∂x^(k)*/∂p^(j) = 0` for `k ≠ j`.

Plumbing:

* `_StackedJaxNlp` lifts the per-block sparsity pattern (cached on
  the parent `JaxProblem` from the one-shot probe) to the stacked
  problem's block-diagonal pattern at construction time, so the
  per-solve `jacobianstructure` / `hessianstructure` callbacks are
  O(1).
* Stacked Problems are built per (thread, B) with a tiny LRU on
  the `JaxProblem` (cap 4) keyed by batch size — guards against
  cycling between a couple of sizes (e.g. eval batch ≠ train
  batch).
* Per-block bounds `lb`/`ub`/`cl`/`cu` are tiled across the batch;
  per-block bounds aren't exposed on this surface.

### Changed

- `pounce-qp::ParametricActiveSetSolver::solve_equality_plus_bounds`
  now falls through to `solve_elastic` when the equality-relaxed
  cold start violates a variable bound. Previously returned
  `UnsupportedFeature`.
- `optimize_sqp_tnlp` now populates `SolveStatistics`
  (`iteration_count`, `final_dual_inf`, `final_constr_viol`,
  `final_objective`) so `GetIpoptIterCount`, `info["iter_count"]`,
  etc. report SQP-side numbers on the SQP path.

### Fixed

- SQP `check_kkt` stationarity formula: was `∇f + Jᵀ λ_g + λ_x`,
  must be `∇f + Jᵀ λ_g − λ_x` (pounce-qp packs
  `λ_x = z_l − z_u = −λ_sat`). Latent — only triggered by problems
  with an active variable bound. Discovered on a 3-D simplex
  projection.
- `fix(mu): guard probing oracle against corrupted iterate (#58)`
  — the probing oracle no longer dereferences fields of an
  iterate that the line-search rejected mid-update.
- `fix(mu/probing)`: σ denominator uses `curr_avrg_compl`, not
  `data.curr_mu`, matching upstream.
- `fix(mu-oracle)`: allow inexact affine predictor solves to feed
  the quality-function oracle (upstream parity).
- `fix(l1-wrapper): use multi-pass restoration factory provider
  (#24)` — the ℓ₁ penalty wrapper now nests a restoration sub-IPM
  whose own restoration provider is the multi-pass factory,
  matching the outer IPM path.
- `fix(restoration)`: restoration sub-IPM inherits the outer
  `mu_strategy` rather than resetting to `monotone`.
- `fix(feral)`: zero-pivot factorizations on LP-shape KKT
  systems route to `Singular` instead of bubbling up as
  `Internal`.
- `fix(fbbt)`: outward-round reciprocal endpoints in
  `Interval::div` for the near-zero straddle case.
- `fix(presolve)`: auxiliary preprocessing + `presolve_bound_tightening`
  infeasibility paths (#60).
- `fix(init/ls)`: perturb `delta_c`/`delta_d` by 1e-8 in the
  least-squares-init augmented system to avoid exact rank
  deficiency.
- `fix(scaling)`: scale `d_l` / `d_u` in step with `d(x)` under
  gradient-based scaling.
- `fix(hsl)`: HSL build script is a no-op when `COINHSL_DIR` is
  unset, so `cargo build` works on machines without HSL
  installed even with the `ma57` feature off.
- `fix(benchmark-report)`: composite report now globs the newest
  `pounce_*.json` under `benchmarks/mittelmann/results/` instead
  of hard-coding `pounce_v0.1.0.json`.
- `fix(jax)`: `pounce.jax.solve` backward pass now respects the
  constraint active set, not just variable bounds. Slack inequality
  rows are dropped from the implicit-function-theorem KKT block via
  the same identity-augment trick used for active bounds; previously
  they were kept as equalities, silently returning the wrong
  `dx*/dp` whenever an inequality was inactive at the optimum
  (pounce#73).

### Docs

- `docs: adaptive-μ option tables, scaling worked example,
  troubleshooting guide` — `docs/src/options.md`,
  `docs/src/scaling.md`, `docs/src/troubleshooting.md` refreshed.
- FBBT reference page (`docs/src/fbbt.md`) and Pyomo demo
  notebook `python/notebooks/08_fbbt.ipynb` (#62).
- Scaling docs page (`docs/src/scaling.md`) + Python demo notebook
  `python/notebooks/07_scaling.ipynb` (#61).
- `studio/skill` README: corrected `POUNCE_BIN` claim,
  `inspect --json`, sibling-feral layout.
- README badges: PyPI version + downloads for `pounce-solver` and
  `pyomo-pounce`; Zenodo DOI
  `10.5281/zenodo.20387011` published.

### Compatibility

- All existing IPM users (`IpoptSolve`, `Problem.solve(x0=…)`,
  `option nlp = pounce` without `algorithm` set) continue to
  behave identically. Every Phase 5 addition is opt-in.
- The C ABI is strictly additive — four new symbols, no signature
  changes.
- The Python `Problem.solve` signature gained one optional kwarg
  (`working_set=None`); positional callers are unaffected.


### Algorithm-path isolation guarantees

The IPM and active-set SQP paths share the TNLP layer, options
registry, linear-solver backend, and `finalize_solution`, but are
otherwise isolated. Toggling `algorithm` is always safe:

- The default (`algorithm = interior-point`) runs zero Phase 5
  code. Users who never set `active-set-sqp` are unaffected.
- `sqp_*` options are silently ignored on the IPM path.
- IPM warm-start options (`warm_start_init_point`, `bound_push`,
  `bound_frac`, `slack_bound_push`, `mult_init_max`, `mu_init`,
  `mu_target`, …) are silently ignored on the SQP path.
- Warm-start payloads are path-local:
  `set_sqp_warm_start(SqpIterates)` /
  `Problem.solve(working_set=…)` / `IpoptSetWarmStartWorkingSet`
  feed the SQP loop only; `lagrange=` / `zl=` / `zu=` paired with
  `warm_start_init_point=yes` feed the IPM only.
- `info["working_set"]` is always present in the Python info
  dict but is `None` on the IPM path.
- Callers can flip between paths across solves on the same
  problem handle — the parametric corrector pattern in the
  tutorial uses this for cold IPM warm-up followed by an SQP
  corrector.

These guarantees are exercised by the test suite: see
`application_default_does_not_select_sqp`,
`application_sqp_warm_start_auto_clears_after_use`,
`application_sqp_warm_start_round_trip`, and
`test_get_working_set_returns_none_on_ipm_path` (Python).

## [0.2.0] — 2026-05-25

First tagged release. The `0.1.0` work-in-progress version was never
tagged; everything below summarizes the state of `main` as of this
release.

### Solver core

- **Full Ipopt-parity C ABI**: `CreateIpoptProblem`, `IpoptSolve`,
  `AddIpoptStrOption` / `AddIpoptNumOption` / `AddIpoptIntOption`,
  `OpenIpoptOutputFile`, `SetIpoptProblemScaling`,
  `SetIntermediateCallback`, `GetIpoptCurrentIterate`,
  `GetIpoptCurrentViolations`, plus a new `IpoptSolver` session
  handle (`IpoptSolverSolve`, `IpoptSolverResolve`,
  `IpoptSolverKktSolve`, `IpoptSolverParametricStep`).
- **Restoration phase** wired through `IpoptSolve` with the soft
  restoration line search; nested IPM honors the parent's
  `print_iter_output` gate.
- **Rapid infeasibility detection** in the main loop; convergence
  statuses certified against upstream Ipopt.
- **Option-parity (tier-A waves 1-4)**: convergence options
  (`tol`, `acceptable_tol`, etc.), mu/watchdog/output toggles,
  iteration-output flags, warm-start machinery,
  `fixed_variable_treatment`, `nlp_*_bound_inf`,
  `barrier_tol_factor`, `sigma_min` / `sigma_max` for the adaptive
  quality-function oracle.
- **Sensitivity (sIPOPT)**: Phase D landed — convenience API,
  eigendecomposition, fixed-variable lifting, boundcheck. New
  `Solver` session API on top: value-typed `Factorization` handle
  in `pounce-linsol` enables factor-once / solve-many; `Solver`
  exposes `kkt_solve`, `parametric_step`, and
  `compute_reduced_hessian` without callback shapes.
- **Presolve** crate (`pounce-presolve`) as an opt-in TNLP wrapper.

### Backends and bindings

- **Python** (`pounce-solver`): PyO3 bindings with a cyipopt-style
  `Problem` class and a scipy-style `minimize()` facade. The wheel
  bundles the `pounce` CLI executable.
- **Python session API** (`pounce.Solver`): pyclass that wraps the
  Rust `Solver`, enabling warm-start sequences (MPC / parametric /
  B&B) and many-RHS sensitivity workflows without the
  callback-based shape.
- **pyomo-pounce** (`pyomo-pounce`): Pyomo SolverFactory plugin
  that drives the `pounce` CLI on the user's PATH.
- **GAMS link**: native solver link (`libGamsPounce`) for GAMS;
  Jacobian eval skips dense memsets and pure-linear rows.
- **CLI**: bundled `pounce` binary writes AMPL `.sol` solution
  output; new `--about` prints version / build / features / paths;
  `--dump` writes per-iteration KKT artefacts; the sIPOPT
  sensitivity step is folded in.

### Linear-solver layer

- **Public `Factorization`** in `pounce-linsol`: factor once,
  back-solve many RHS, refactor with new values reusing the
  symbolic factor / AMD ordering.
- **MA57** backend (`pounce-hsl`) honors the `linear_solver`
  option default (`"ma57"`).
- **Feral** backend: cascade-break and FMA default off (opt-in via
  env); near-singular factorizations are flagged via an absolute
  pivot floor; explicit-zero stripping before KKT factor; skips
  refactor on same-matrix back-solve.

### Numerical robustness

- TNLP `eval_*` user-callback failures surface as NaN instead of
  panicking.
- Round-off-tolerant `Compare_le` in the Armijo line-search test.
- Unconstrained problems routed through the IPM (no degenerate
  paths).
- `push_x_into_interior` uses `dim()` (not `values().len()`),
  fixing a subtle off-by-one on partially-filled vectors.
- `OrigIpoptNlp::eval_h` always uses the `h_entry_in_full`
  mapping; closes the panic when an entire Hessian row sits on a
  fixed variable.

### Benchmarks

- **Composite report** (`make benchmark` →
  `benchmarks/BENCHMARK_REPORT.md`) covering 9 suites: CUTEst (727
  curated; 1542 full sweep), Mittelmann LP/QP, water-network
  design, gas-network, electrolyte, grid, CHO, large-scale, and
  the GAMS link.
- **Incremental per-suite targets**: `make benchmark-<suite>`
  skips when `results.json` is fresh; `make benchmark-<suite>-rerun`
  forces a rebuild.
- **MA57 baseline** integrated into the composite report.

### Studio & tooling

- **studio/mcp** MCP server (`pounce-studio-mcp`) with
  `analyze`, `run`, `explain`, `citations` tools and an embedded
  glossary; backed by `pounce-studio-core` via PyO3.
- **Linear-solver post-mortem** aggregated end-to-end and
  surfaced through the studio.

### Infrastructure

- CI workflow with format / clippy / build / test, plus
  wheel-smoke for `pounce-solver` and `pyomo-pounce`.
- mdbook documentation built and deployed to GitHub Pages via the
  new `docs.yml` workflow.
- Zenodo metadata (`.zenodo.json`) and `CITATION.cff` for
  archival on every GitHub Release.

[Unreleased]: https://github.com/jkitchin/pounce/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/jkitchin/pounce/releases/tag/v0.10.0
[0.9.0]: https://github.com/jkitchin/pounce/releases/tag/v0.9.0
[0.8.0]: https://github.com/jkitchin/pounce/releases/tag/v0.8.0
[0.7.0]: https://github.com/jkitchin/pounce/releases/tag/v0.7.0
[0.6.0]: https://github.com/jkitchin/pounce/releases/tag/v0.6.0
[0.5.0]: https://github.com/jkitchin/pounce/releases/tag/v0.5.0
[0.4.0]: https://github.com/jkitchin/pounce/releases/tag/v0.4.0
[0.3.0]: https://github.com/jkitchin/pounce/releases/tag/v0.3.0
[0.2.0]: https://github.com/jkitchin/pounce/releases/tag/v0.2.0
