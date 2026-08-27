# The sensitivity-layer review checklist

A companion to `/adversary`. That one hunts for wrong answers from the outside,
by solving problems whose answers are known. This one reads a **diff** to
`pounce-sensitivity` and asks the questions that the defects which actually
shipped from this crate would have been caught by — in a fixed order, so the
same questions get asked by whoever reviews next.

Every entry below carries a worked example from this repository rather than a
maxim, with the file and symbol you can go read. That is deliberate: a
checklist whose examples cannot be checked rots into a ritual, and entry 4 is
about exactly that failure.

**This checklist does not approve anything.** It produces a verdict per entry
with the evidence attached. A `RISK` verdict is a question for the author, not
a veto.

## Arguments

$ARGUMENTS

Parse for what to review:

- A **PR number** (e.g. `800`) — review that PR's diff.
- A **git ref or range** (e.g. `HEAD~1`, `main..HEAD`) — review that diff.
- A **path** (e.g. `crates/pounce-sensitivity/src/solver.rs`) — review the
  working-tree changes to it.
- **Empty** — review the current uncommitted diff, then `main..HEAD`.

Optional trailing text narrows the pass: `indices`, `scaling`, `thresholds`,
`docs`, `silent`, `branches`, `populations`, `binary` select single entries.
Default is all eight, in order.

## How to run it

1. Read the **whole** diff first, then the files it touches around the change.
   A cross-space read three lines outside the diff is still this change's
   problem if the diff is what made it reachable.
2. Walk entries 1–8 **in order**. They are ordered by how silently the class
   fails: entry 1's defects return a plausible neighbouring answer, entry 8's
   announce themselves.
3. For each, emit one of:
   - **N/A** — the diff cannot reach this class. Say why in one line.
   - **PASS** — reached, and discharged. Name the evidence.
   - **RISK** — reached, and not discharged. Name what is missing and the
     cheapest thing that would settle it.
4. Never mark PASS on the strength of a green test without checking **which
   branch that test's fixture takes** (entry 6). That inference is what shipped
   gh#756's defect through a green suite.

---

## 1. Index space

**Ask.** Every index in the diff: which space is it in — `var_x` (the
algorithm's free variables), `full_x` (the model's variables, fixed ones
included), a **KKT row**, or a **user-g** row? At every point where one is used
where another is expected, is there a conversion? Is the conversion in the
right direction?

**Why.** gh#450, then gh#672 finding 1 shipped it again. The read was
`report.var_status.get(var_row)` — a `var_x` row used to index a `full_x`
array. The two spaces coincide until the first `make_parameter`-removed
variable and diverge after it, so the fixture corpus of the day could not see
it, and the value returned is a *neighbouring variable's* answer: plausible,
in range, and wrong.

The site now reads, at `crates/pounce-sensitivity/src/solver.rs:1454`:

```rust
let Some(&st) = report.var_status.get(full_of[var_row]) else {
```

with `full_of` built from `var_x_to_full_x` immediately above it. The `full_of`
indirection *is* the fix.

**The sharp rule, from gh#764 item 3:** *table lookup fails loudly; direct
array indexing fails silently.* A released row resolved through
`rows.iter().find(|b| b.row == r)?`
(`crates/pounce-sensitivity/src/algorithm_backsolver.rs:645`) turns a
space-swap into `SensComputationFailed`, every time, immediately. A direct
index does not. So the scrutiny belongs at conversions **consumed by direct
indexing**, and that is a far smaller surface than every index in the crate.

The one to look at hardest is `crates/pounce-sensitivity/src/solver.rs:1176`:

```rust
report.var_sigma.get(full).copied().unwrap_or(0.0)
```

A direct index *and* an `unwrap_or(0.0)` — a miss does not error, it returns
zero, which zeroes that row's kappa and silently drops the row from the engaged
set.

**The typed path.** `crate::index` (gh#764 item 3) gives `VarX`, `FullX`,
`VarToFull` and `FullXSlice` for exactly the shape above. `FullX` has no
public constructor — it comes only from putting a `VarX` through the map — so
at the two sites that use it, a swap is `E0624` rather than a neighbouring
row's answer. If the diff adds a *third* such site, the question is why it is
not using them.

It covers the typed path and no more: `ActivityReport`'s `Vec` fields are
public and `report.var_status.get(row.get())` still compiles. So a green
compile is not a discharge on its own.

**Discharge.** Point at a leg whose fixture puts a fixed variable **ahead of**
the row under test, so the spaces actually diverge —
`leg_fixed_the_index_spaces_actually_diverge` in
`sens_invariance_legs.rs` asserts the divergence itself, and the three
`leg_fixed_*` legs after it assert the answers survive it. For a pin's g-index
versus its KKT row, that is `cd_split_pin_mapping.rs`, whose fixture puts an
inactive inequality ahead of three equalities on purpose.

---

## 2. Frame and scaling

**Ask.** Does the diff mix the algorithm's **scaled** frame with the model's
**units**? Any quantity added to, compared with, or divided by another: are
both in the same frame? Where a quantity is compared against `Sigma`, is it
`Sigma` or `Sigma/mu`?

**Why.** `205bb67`: the corrector added the scaled iterate to bounds held in
the model's units. In its own words, the two "coincide only at unit scaling,
which is every fixture it had." Under `x̃ = d ⊙ x` the barrier diagonal carries
`d⁻²`, so anything compared against a bare `Sigma` moves when `d` does.

`Sigma` is also proportional to the `mu` its run stopped at, so **the invariant
is `Sigma/mu`, never `Sigma`** — `variable_scaling_sensitivity.rs` says the
same of the classifier.

**Discharge.** A leg that solves the same model under two different
`user-scaling` factor sets and asserts the quantity is unmoved:
`leg_scaling_the_weak_set_is_unmoved_by_the_change_of_variables`,
`..._the_reduced_curvature_...`, `..._the_directional_derivative_...` in
`sens_invariance_legs.rs`. A new public accessor gets a row in each — the cost
of leaving one out is that the next defect in that dimension is invisible.

---

## 3. Absolute thresholds on scale-dependent quantities

**Ask.** Does the diff introduce or move a **constant**? What are the units of
the quantity it is compared against? If that quantity scales with anything —
the perturbation, `mu`, the model's units, the problem size — then a fixed
constant is a bug with a fixture-shaped blind spot.

**Why.** @devin-griff's statement of it, which is the sharpest we have:

> an absolute length compared against a homogeneous quantity breaks linearity
> even when the length is measured rather than chosen

gh#672 finding 4 put an absolute tolerance on a quantity that scales with the
perturbation. At `delta = 1e-10` the step cleared feasibility *everywhere*, so
the holding side's derivative read `-1` where it should have read `0`. The
defect is invisible at `1e-2` and total at `1e-10`, and the corpus of the day
only ran near `1e-2`.

The live one to weigh this against is `KAPPA_MIN`
(`crates/pounce-sensitivity/src/solver.rs:1082`, used at `:1243`):

```rust
if nat_sigma[k] * own < KAPPA_MIN {
```

`nat_sigma[k]` is the barrier diagonal `Σ = z/s` for that row — a curvature —
and `own` is the corresponding diagonal of the *reduced* compliance, an inverse
curvature. Their product is a **ratio of curvatures**, dimensionless by
construction, which is precisely why a bare constant is legitimate there. That
is the test to apply: not "is there a constant" but "is the quantity it is
compared against dimensionless".

Note the shape of the answer — the quantity was made dimensionless by
*dividing by the scale it carries*, not by picking a tolerance small enough to
look safe. `ActivityReport::var_sigma` documents the same move: the
classification runs on the solver's scaled quantities because the ratio is
scale-invariant, while the report itself is in natural units.

**This applies to test tolerances, not just to `src/` constants.** A budget
that does not scale with the thing it bounds makes a leg go quietly vacuous at
one end of its span, and it is the same defect wearing a different hat. Caught
in review on PR #800: `leg_oracle_the_step_reproduces_the_resolve_above_the_barrier_width`
budgets a flat `4 · max(floor, width) = 2.17e-4` while the step it predicts
shrinks with `delta`, so at `delta = ±1e-4` the entire exact displacement is
`1.0e-4` — inside the budget, meaning a predictor that returned the base point
unmoved would pass those two entries. It is only those two: at `±1e-3` a null
step misses by `1.0e-3`, five budgets out.

Note that the fix there was *not* to tighten the budget — that would have left
`1.58×` between the measured error and the threshold, trading a robust leg for
a flaky one. It was to record the limit in the file's not-evidence-about
section. Both are legitimate outcomes; what is not legitimate is leaving the
reader to assume the fine end bites.

**Discharge.** A leg that sweeps the scale-carrying quantity over orders and
asserts the decision does not move: the `leg_magnitude_*` legs sweep `delta`
over eight orders on both sides of the kink. For a test budget specifically,
ask what a **null step** — the predictor returning its input unchanged —
would score at each end of the span; if it passes anywhere, say so where the
reader will look. Compare **slopes, not
`dx/delta`** — the parametric step is affine in `delta`, not linear, and
carries a base-point term of order `mu` that dividing by `delta` inflates until
it is the whole answer at `1e-10`. `the_step_is_affine_in_delta` pins that
term's size so a leg never fails for a reason its name does not describe.

---

## 4. Doc drift

**Ask.** For every contract in a doc comment the diff touches — and every one
**above code the diff changed** — is it still true of the code below it? Do the
numbers in it still reproduce? Does it name a test that still exists, a
constant that still has that value, an argument that is still passed?

**Why.** The recurring failure is a comment that was accurate three commits
ago. A contract that outlives its code is worse than no contract: the next
reader budgets against it.

A worked example from this crate, caught in review on PR #800:
`the_oracle_out_resolves_the_base_offset` carried a comment claiming the oracle
was "three orders finer than that floor at every step size the leg uses", while
the assert below it enforced two (`floor / 100.0`). Measured, the margin runs
`6.0e5` at `delta = 1e-1` down to `5.4e2` at `1e-4` — so three orders was true
at the coarse end, false at the fine end, and the assert was the half that was
right. The fix was to the comment.

**Discharge.** Re-read each touched doc comment against the final diff, not the
intended one. Where the comment states a number, re-measure it rather than
trusting it; where it names a symbol, grep for the symbol. If a design changed
mid-review, the comments describing the first attempt are the ones that survive
into `main`.

---

## 5. Silently wrong while reporting success

**Ask.** Does the diff change what a **success signal** means — `improved()`, a
residual, a converged status, an `Ok`? Could the new code return that signal
while the answer is wrong? Is there any guard that would notice, and does that
guard read a number this layer did not produce?

**Why.** This is the class with the worst blast radius, and it is `205bb67`'s
own framing:

> All three turn an essentially exact step into a wrong one while reporting
> `improved()` and converged … the residual halved so nothing warned.

Every internal guard compares a number the sensitivity layer produced against
another number it produced. Those catch a rule that is not invariant. They
cannot catch a step that is **self-consistently** wrong.

The number that makes this concrete is pinned in
`the_corrector_reports_improved_without_crossing_a_release`
(`sens_resolve_oracle.rs`): across a strongly active release the corrector
reports `improved()` and drops its residual by `3e-8` while the point stays
`0.1333` from the truth — the entire distance the variable should have
travelled off its bound. Not a defect; the held barrier diagonal cannot
represent a bound leaving the active set. But it is the proof that
`improved()` plus a falling residual does **not** imply the answer is close.

**Discharge.** `sens_resolve_oracle.rs` — a warm re-solve at the perturbed
parameter, tolerance two orders tighter than the base solve. It is the only
guard in the crate that reads an outside number. A change that reroutes which
correction a mode reaches for gets a row in it.

Two constraints on using it, both load-bearing:

- **It owns steps above the barrier width and nothing below it.** Below
  `sqrt(mu)` the warm re-solve and the directional-derivative contract
  genuinely diverge, by the row's slack, and *both are correct*. An oracle arm
  placed there fails on correct code. The invariance legs own that region.
- **Verify the oracle before trusting it.** `the_resolve_is_an_independent_oracle`
  pins the re-solve against a closed form before any leg compares a step to it.
  An oracle sharing the machinery's defect is worse than none.

---

## 6. Which branch does the fixture actually take?

**Ask.** For every test offered as evidence: does the rule under test
**branch** — on an activity class, a status, which side of a threshold a
quantity falls? If so, which branch does this fixture reach? Is there a fixture
for the other one?

**Why.** A leg is only evidence about the branch its fixture reaches. It stays
green while the other branch is broken.

This is not hypothetical. The invariance legs passed on gh#756's head while
that PR's defect was live, because their kink certifies `WEAKLY_ACTIVE` and
that PR's rule gave certified rows an early return — so the code actually under
review was never executed. The leg that found the defect was the one whose
fixture lands in the *other* class.

**Corollary for the classifier, gh#763.** `AMBIGUOUS` is not "probably not a
kink". `classify` divides `Sigma` by the Hessian's **diagonal**, but a kink's
multiplier is generated by the curvature *reduced* along that coordinate, so
the ratio is `reduced/diagonal` and equals 1 only when the coordinate is
decoupled. Couple it — routine on a collocation model — and a genuine kink
lands in `AMBIGUOUS`. **Never use the activity class as a proxy for
kink-ness**; that inference is exactly what produced the gh#756 defect.

**Discharge.** A second fixture reaching the other branch. It is the test, not
a duplicate of the first: `the_coupled_fixture_carries_an_ambiguous_kink` and
`the_reduced_normalizer_certifies_a_coupled_kink_at_every_coupling` exist
because the decoupled one could not see gh#763. In `sens_resolve_oracle.rs`,
`the_fixtures_take_different_branches` asserts the split rather than assuming
it — worth copying, since a fixture can drift into the same class as its
partner and take the evidence with it silently.

---

## 7. Name the measured populations on each side

**Ask.** Does the diff add a **threshold or discriminator**? Have the
populations on each side of it been measured, or is the value argued for? What
is the deciding statistic on every counterexample already on the record?

**Why.** From @devin-griff on gh#764: three gh#756 designs were withdrawn, and
each rested on a claim nobody had measured. The one that survived was designed
*after* measuring the deciding statistic on every prior counterexample.

A threshold defended by argument is a threshold nobody can retune later,
because the next maintainer cannot tell which numbers it was chosen to
separate.

**Discharge.** The measurement, in the PR body or the comment above the
constant: what was measured, over which models, and what the two populations
look like. "Measured over N models, every threshold that fires also introduces
X" beats "this seemed better". A regression accepted as the cost of a fix needs
an issue and an owner — without one it is indistinguishable from noise to the
next reader.

---

## 8. Which binary did the harness load?

**Ask.** For any result offered as evidence from the Python surface: which
extension module did that run import — the in-tree `python/pounce/_pounce*.so`,
or the one in site-packages? Was it built from the code under review? Was the
staleness guard bypassed?

**Why.** From @devin-griff on gh#764: a day of gh#756 validation ran against a
stale in-tree `.pyd` while a freshly built module sat unloaded in
site-packages. The repo's own guard had flagged it, and was bypassed on the
assumption that the site-packages build was the loaded one.

The guard is `python/tests/conftest.py`, which fails fast when the in-repo
`.so` is older than the Rust binding sources. Its bypass is
`POUNCE_SKIP_EXT_STALE_CHECK=1`.

**Discharge.** State which artifact ran, and rebuild before trusting a
surprising Python-side result. A bypassed staleness guard invalidates the
run — treat `POUNCE_SKIP_EXT_STALE_CHECK=1` in the transcript as a `RISK` on
every Python-derived number in the review, not as a detail.

---

## Output

A table, then the RISK entries in full:

```
| # | class | verdict | evidence / what is missing |
|---|-------|---------|----------------------------|
| 1 | index space          | PASS | leg_fixed_* legs; the only new cross-space read is table-lookup |
| 2 | frame and scaling    | N/A  | no quantity compared against Sigma |
| 3 | absolute thresholds  | RISK | new constant at solver.rs:NNNN compared against a delta-scaled quantity |
...
```

For each RISK: the question, why the diff reaches that class, and the cheapest
thing that would settle it — a fixture, a measurement, a re-read. Prefer
naming the fixture that would decide it over asking the author to think again.

## What this checklist is not

Stated because the same discipline applies to it as to the legs it points at:

- **Not a substitute for running the tests.** It reads a diff. Every discharge
  above names a test because the test is the evidence; the checklist only says
  which one.
- **Not evidence about the arithmetic.** It asks whether a quantity is in the
  right frame, not whether the formula in that frame is correct. `/adversary`
  and the re-solve oracle own that.
- **Not scale coverage.** The resource paths that only appear at 62k (gh#672
  f2's `2^n` masks, gh#708 f4's full-length basis columns) were found by
  profiling on a real collocation model, not by review and not by a corpus.
  gh#764 puts them out of scope for the in-repo guards; they stay out of scope
  here.
- **Not closed.** Entries earn their place by having shipped a defect. When the
  next one ships from a class not listed here, it gets an entry with its own
  worked example, and the entry that failed to catch it gets re-read.
