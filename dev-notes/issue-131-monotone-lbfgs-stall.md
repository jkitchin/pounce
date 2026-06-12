# issue #131 — monotone L-BFGS stall: the two follow-ups

[#131](https://github.com/jkitchin/pounce/issues/131): a D-optimal design
problem on the **limited-memory (L-BFGS) + monotone** path runs to `max_iter`
(status `-1`) where the reporter expected convergence.

**Root cause (documented in the issue reply):** it is the *curvature model*,
not the barrier strategy. Under monotone, μ only decreases once the barrier
subproblem error drops below `barrier_tol_factor · μ` (κ_ε = 10). On this
problem μ freezes at `lg(mu) = −2.5` while `inf_du ≈ 0.24 ≫ κ_ε·μ ≈ 0.032`, so
the subproblem never clears its tolerance and μ is wedged. The true Hessian has
min eigenvalue ≈ −1.8e7 (genuinely nonconvex), but **Powell-damped BFGS forces
a PSD model**, so the IPM's inertia check never sees the indefiniteness, never
fires regularization, and never escapes. cyipopt with the same L-BFGS monotone
settings also fails in our environment (status `-3`); the exact Lagrangian
Hessian (= `obj_factor · ∇²f`, since all constraints are linear) fixes both
solvers (pounce: 108 iters, status 0).

The reply committed to two internal follow-ups. This note records both
outcomes.

---

## Follow-up A — auto stall-detector: **prototyped and discarded**

An opt-in, default-off `monotone_stall_iter` option that watched for μ frozen
for N consecutive iterations and terminated early with a diagnostic instead of
grinding to `max_iter`. Wired end-to-end (option → `application.rs` →
`ipopt_alg.rs` detection block → early return), default-off kept the
bit-exact-Ipopt path inert, all `mu::` tests passed. On #131 it bailed ~300–400
iters early at the same objective the baseline reaches.

**Why discarded.** Building it falsified the premise. The "μ freeze" looks like
a hard stall in a `print_level=5` trace, but it is actually a *uniformly
decelerating crawl*: μ keeps inching forward, E_μ and the objective keep
falling, just too slowly to clear `κ_ε·μ` within budget. There is **no clean
bimodality** — no fixed-window gate separates "slow but will clear" from
"doomed." An objective-progress gate (`Δf < 1e-6`) never fired (the objective
genuinely creeps ~`2e-5`/window the whole time); a barrier-error stagnation
gate (`E_μ` fell <50% over the window) worked but false-positived on the μ=0.1
initial plateau, so the only usable thresholds were problem-specific (~150–200).

So `monotone_stall_iter` is fundamentally a **patience limit** dressed up as a
stall detector — an unprincipled tuning knob whose only good value is
problem-specific, competing with the actual fix. Reverted; nothing landed.
If a "don't silently grind to max_iter" message is ever wanted, prefer a
*post-hoc* diagnostic keyed on the existing `max_iter`+frozen-μ exit state, not
a mid-solve heuristic that perturbs the trajectory.

---

## Follow-up B — honor `limited_memory_update_type`: **wired and kept**

While surveying the limited-memory machinery for "strengthen the inner solve,"
we found the strengthening lever already exists and is unit-tested — it was
just never reachable:

- `crates/pounce-algorithm/src/hess/lim_mem_quasi_newton.rs` implements **both**
  `UpdateType::Bfgs` *and* `UpdateType::Sr1`. SR1 is sign-indefinite: when the
  curvature denominator is negative it stores the pair as a `−UUᵀ`
  (negative-curvature) column instead of damping it away.
- That indefinite model is already wired into inertia correction: in
  `kkt/low_rank_aug_system_solver.rs` an SR1 negative-curvature column makes the
  SMW middle-matrix Cholesky fail → returns `WrongInertia` → triggers
  `PdPerturbationHandler` regularization (unit test
  `smw_reports_wrong_inertia_on_indefinite_negative_update`).
- But `limited_memory_update_type` (and `limited_memory_max_history`,
  `limited_memory_initialization`, `limited_memory_init_val*`,
  `limited_memory_special_for_resto`) were **registered but read nowhere on the
  IPM path** — the updater was hard-wired to BFGS via the struct `Default`. A
  user setting `limited_memory_update_type=sr1` was silently ignored. That's a
  genuine Ipopt-parity defect independent of #131.

**What was wired** (the clean-mapping subset):

- `application.rs` now reads `limited_memory_update_type` (`bfgs`/`sr1`) →
  `AlgorithmBuilder.limited_memory_update_type` (field already existed and was
  already consumed at `alg_builder.rs`), and `limited_memory_max_history` →
  new `AlgorithmBuilder.limited_memory_max_history` field → the updater's
  `max_history`.
- **Default unchanged: `bfgs`, history 6** — bit-exact with Ipopt's default, so
  no behavior change unless the option is explicitly set. All 259
  `pounce-algorithm` lib tests pass.

**Empirical result on #131** (`/tmp/dopt_sr1_test.py`, `max_iter=1000`,
`OMP_NUM_THREADS=1`):

```
limited-memory (no Hessian supplied):
L-BFGS bfgs / monotone (baseline)    iters=1000 status= -1 obj=123.6373   <- the #131 stall
L-BFGS sr1  / monotone               iters= 597 status=  1 obj=122.3683   <- converges
L-BFGS bfgs / adaptive               iters= 786 status=  1 obj=122.8050
L-BFGS sr1  / adaptive               iters= 402 status=  1 obj=123.9170
EXACT-H     / monotone (reference)   iters= 108 status=  0 obj=122.9828
```

`limited_memory_update_type=sr1` **breaks the monotone stall** (status `-1` →
`1`, feasible, `viol≈1e-8`) and lands on a better objective (122.37) than the
stalled BFGS reached, near the exact-Hessian reference (122.98). The mechanism
held: SR1's negative-curvature columns make the inertia check fire,
regularization engages, μ unfreezes.

### Honest caveat — do NOT make SR1 the default

Ipopt's own option registration labels SR1 **"(not working well)"** (see
`upstream_options.rs:602`), and that judgment is upstream's for good reason —
SR1 limited-memory is unreliable in general. The right framing:

- We now **honor** the option (parity fix) — that part is unconditionally
  correct.
- For **#131's problem class** specifically, `limited_memory_update_type=sr1`
  is an *effective opt-in workaround* when an exact Hessian isn't available.
- The **default stays `bfgs`**. The substantive recommendation for #131 remains
  the exact Hessian (status 0, fewest iters, matches Ipopt); SR1 and `adaptive`
  are the limited-memory fallbacks.

### Not wired (deliberate, documented)

`limited_memory_initialization` (only `scalar1`/`scalar2` map to the
`InitialApprox` enum; `scalar3`/`scalar4`/`constant` have no variant),
`limited_memory_init_val*`, `limited_memory_max_skipping`, and
`limited_memory_special_for_resto` remain registered-but-unread. Wiring them
partially would itself be a parity mismatch; left for a focused follow-up if
needed.

### Files touched (kept)

- `crates/pounce-algorithm/src/application.rs` — read `limited_memory_update_type`
  + `limited_memory_max_history`; `use ...::UpdateType`.
- `crates/pounce-algorithm/src/alg_builder.rs` — new
  `limited_memory_max_history` builder field (+ default, + consumed in the
  `LimMemQuasiNewtonUpdater` construction, + the test struct literal).
