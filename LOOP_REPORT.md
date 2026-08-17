# gh #621 — a reordered warm start is not refused unless the caller supplies `var_ids`

Branch: `claude/pounce-issue-621-loop`, merged up to `origin/main` at
`eeca47f9`. Two content commits plus a merge. **No Rust changed.**

> **On the commands below.** The measurement scripts named
> `scratchpad/*.py` were written for this investigation and run from a
> scratch directory; they are **not committed** — this branch adds only
> `LOOP_REPORT.md`, and the work branch adds only the source, test, doc
> and CHANGELOG changes. Their content is described in enough detail
> here to be rebuilt, and every behaviour they measure is also pinned by
> a committed test in `python/tests/test_warm_start_schema.py`. The
> `pytest`, `cargo`, `scripts/` and `git` commands are reproducible as
> written.

---

## 1. The reproduction, on unmodified `main`

Before changing anything I reproduced the failure on `main` at
`b4c4d32e`. Script: permuted HS071 (`perm = [2, 0, 3, 1]`, uniform box
`[1,5]^4`, dense jacobian), capture a *signed* warm start against the
canonical model with **no `var_ids`**, then replay it on the permuted
model.

```
python/.venv/bin/python scratchpad/repro_621.py        # on b4c4d32e
```

Every facet of the two signatures came out identical:

| facet | canonical | permuted | |
|---|---|---|---|
| n / m | 4 / 2 | 4 / 2 | SAME |
| var_ids / con_ids | None | None | SAME |
| bounds | `51e5c8cd33c97b92` | `51e5c8cd33c97b92` | SAME |
| sparsity | `77ecf28c2ff3692a` | `77ecf28c2ff3692a` | SAME |
| scaling | `12ac3016134fa70f` | `12ac3016134fa70f` | SAME |
| algorithm | `4a8839771719ec68` | `4a8839771719ec68` | SAME |
| model | `4a8839771719ec68` | `4a8839771719ec68` | SAME |

- `ws.check_compatible(permuted_problem)` returned `[]` — **no mismatch**.
- `ws.describe_compatibility(...)` returned `"warm start is compatible
  with this problem"`.
- The replay was attempted and **damaged the solve**:

| | value |
|---|---|
| true objective (cold, either ordering) | **17.0140171452** in 11 iterations, `Solve_Succeeded` |
| reordered warm replay | **16.0909032757** in 44 iterations, status `-3 Error_In_Step_Computation` |
| absolute objective error | 0.9231138695 |
| `‖x − x_true‖_∞` (un-permuted) | 0.2204702401 |

**Difference from the issue's numbers, stated plainly.** The issue
reports objective **16.3801074993** and `x` off by **0.257**, measured on
the pre-#607 parent commit. I measured **16.0909032757** and **0.2205**
on current `main`. Same status (`Error_In_Step_Computation`), same
iteration count (44), same character of failure — but these are *my*
numbers on *this* commit, not the issue's, and the drift is most likely
#606's residual recentering landing in between. Everything in this
report is measured on `b4c4d32e` unless it says otherwise.

This reproduction is the positive control: I watched the detector fire on
it after the fix (§3), not only on synthetic cases.

---

## 2. What I changed, and why this option

The issue lists three options. I took **the order-sensitive model probe**,
and here is the reasoning, not just the choice:

- **Auto-derived `var_ids` on Pyomo / `.nl`** cannot close the acceptance
  criterion, because the reproduction *is* a plain `pounce.Problem` built
  from a `problem_obj`. That frontend has no names to derive. It would fix
  the structured-model case and leave the measured failure standing.
- **Making the limitation louder** does not refuse anything.
- **The probe** needs no names, so it covers the raw-`Problem` path where
  the failure actually lives.

I folded the third option in as well — it is nearly free, and it is the
honest thing to say in the cases where the probe is unavailable (§6).

### The mechanism

`ProblemSignature` gains a `probe` facet: the model evaluated **once** at
a fixed point inside the bounds, recorded as a small vector of
order-weighted projections.

**The probe point** (`_probe_point`) is `centre + (t_i − ½)·min(span, 1)`
for a boxed variable, with `t_i = 0.25 + 0.5·frac((i+1)·φ)`:

- **Deterministic** — no RNG. `*` and `floor` are exact IEEE-754
  operations, so it is bit-identical on every platform; nothing routes
  through libm (`cos`, `exp`), which is *not* cross-platform stable.
- **Strictly inside the bounds**, because a model is entitled to be
  undefined outside its own box (`log(x)` with `lb=0`). Infinite bounds
  and fixed variables (`lb == ub`) are handled explicitly and tested.
- **Varies with the index**, deliberately. A point that did not (all-ones)
  would be a fixed point of every permutation, so a model symmetric at
  that point would slip through.
- The span cap keeps a `[-1e18, 1e18]` box from being probed at `1e17`,
  where a badly scaled model overflows and the probe is lost.

**What is digested, and at what tolerance.** Nothing is digested — and
that is the central design decision. Four blocks (`objective`,
`gradient`, `constraints`, `jacobian`) are each reduced to 4
order-weighted projections plus an L1 scale, and compared to a
**relative tolerance** `PROBE_RTOL = 1e-9`.

A hash cannot be compared approximately. Had I hashed the probe, a model
whose evaluation is not bitwise reproducible would be refused for
reproducing itself to 15 digits instead of 17 — **refusing a valid replay
is a worse failure than the one being fixed**, so the comparison must
have slack, and that rules a digest out. Each block is compared against
its own L1 scale with the largest block's scale as a floor, so a
gradient that is identically zero is judged against the magnitude of the
rest of the model rather than against nothing.

**Files:** `python/pounce/_warm_start_schema.py` (probe, comparison,
report), `python/pounce/_warm_start.py` (capture/replay plumbing),
`python/pounce/_starts.py` (one line, §5), plus tests and docs.

---

## 3. Does it fire?

Re-running the same reproduction on the branch:

```
python/.venv/bin/python scratchpad/repro_621.py
```

```
warm start is not compatible with this problem (1 mismatch,
exact-structure replay, schema v2):
  - probe: this problem's model does not evaluate to the same numbers as
    the one the warm start was captured against (a reordering of the
    variables looks exactly like this; so does a different model of the
    same shape)
```

Refused **before the solver is entered**, with no `var_ids` from anyone.

One measured detail worth recording, because it justifies probing four
blocks rather than one: on HS071 the **constraints and jacobian blocks
are permutation-invariant** (`prod`/`dot` are symmetric, and the jacobian
entries are per-variable so the permutation cancels). They agreed to the
last ulp. Only the **objective and gradient** blocks caught the
permutation. A single-block probe would have missed this fixture.

---

## 4. Reproducibility — measured, not assumed

The issue asks what happens to a model whose evaluation is not bitwise
reproducible, and says to test it.

```
python/.venv/bin/python scratchpad/probe_reproducibility.py
```

| test | result |
|---|---|
| **A.** Pure numpy model, 20 repeated captures | **bit-identical** across all 20 |
| **C.** Same maths, internal summation order reversed (what a different BLAS / thread count / SIMD width actually does) | not bit-identical; worst difference **4.441e-16** absolute, **4.764e-18** relative → **accepted** |
| **D.** jax 0.10.2, `jit`+`grad`, x64, two freshly compiled models | **bit-identical** → accepted |
| **D.** torch | **not installed — not measured.** Reported as a gap, not as a pass. |

**B. Where is the refusal threshold?** Injected relative jitter, 40 trials
each:

| jitter | accepted | verdict |
|---:|---:|---|
| 0 | 40/40 | accepted |
| 1e-16 | 40/40 | accepted |
| 1e-14 | 40/40 | accepted |
| 1e-12 | 40/40 | accepted |
| 1e-11 | 40/40 | accepted |
| 1e-10 | 40/40 | accepted |
| 1e-9 | 40/40 | accepted |
| 1e-8 | 23/40 | borderline |
| 1e-6 | 0/40 | REFUSED |

So real evaluation noise (≈5e-18 relative, measurement C) sits **nine
orders below** the tolerance, and a permutation (which moves a projection
by a fraction of the block's own magnitude) is far above it. The
tolerance sits in a wide gap, which is what makes `1e-9` defensible
rather than tuned.

---

## 5. Cost — measured on the fixture corpus and on the paths that changed

```
python/.venv/bin/python scratchpad/probe_cost.py
```

Capture side:

| n | nnz | signature, no probe | signature + probe | **probe cost** | cold solve | **% of solve** |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 0.0445 ms | 0.1942 ms | **0.1497 ms** | 12.78 ms | **1.171%** |
| 100 | 200 | 0.0556 ms | 0.2344 ms | **0.1788 ms** | 44.96 ms | **0.398%** |
| 1 000 | 2 000 | 0.1370 ms | 0.4748 ms | **0.3378 ms** | 400.4 ms | **0.084%** |
| 10 000 | 20 000 | 0.8794 ms | 2.6092 ms | **1.7299 ms** | 86 986 ms | **0.002%** |

Replay side — an artifact captured with `probe=False` costs nothing extra,
by construction (the target is probed only when the artifact carries a
probe to compare against):

| n | captured `probe=False` | captured `probe=True` | delta |
|---:|---:|---:|---:|
| 4 | 0.0476 ms | 0.2002 ms | 0.1526 ms |
| 100 | 0.0565 ms | 0.2252 ms | 0.1687 ms |
| 1 000 | 0.3975 ms | 0.5734 ms | 0.1760 ms |
| 10 000 | 0.9272 ms | 2.5580 ms | 1.6308 ms |

Artifact growth: **1 640 B**, and it is a **fixed 20 floats regardless of
problem size**.

### The regression the cost measurement caught

The full suite failed on
`test_starts_racing.py::test_the_ladder_beats_the_pre_610_cost`. It was
mine, and it was real, not a threshold artefact.

`_starts.py` re-checks its warm start once per multistart rung. The probe
evaluates the model — and on that path the model is the **caller's own
counted callable** — so probing per rung spent four of the user's
evaluations per resume:

```
python/.venv/bin/python scratchpad/racing_evals.py
```

| | user-callable evaluations |
|---|---:|
| `origin/main` baseline | **2592** |
| probe enabled on that path | **2920** |
| after the fix (`probe=False` there) | **2592** — and objectives bit-identical |

The fix is one line plus a comment: that path captures from, and resumes
on, the **same `Problem` object in the same process**, so no reordering is
possible and the facet cannot fire. Declining it at capture also makes the
per-rung check free.

I pinned this with a test that asserts the racing ladder never evaluates
the model **at the probe point**, and I confirmed the test fails when the
probe is reinstated (it reported 12 probe-point evaluations) — a guard I
never watched fire would not be a guard.

### Fixture sweep

Built a baseline binary from `origin/main` in a separate worktree and
swept all 57 CLI fixtures both ways:

```
scripts/sweep-fixtures.sh /tmp/p-base           /tmp/base.txt
scripts/sweep-fixtures.sh target/release/pounce /tmp/new.txt
diff /tmp/base.txt /tmp/new.txt
```

**Empty diff, 57/57, 0 NO_JSON.**

I want to be honest about what that does and does not prove. `git diff
origin/main -- crates/ Cargo.toml Cargo.lock` is **empty** — I changed no
Rust — and the sweep drives the Rust CLI, which never touches the Python
warm-start path. So the empty diff is *necessary but not sufficient*
evidence for this change. The load-bearing trajectory evidence is the
Python-side analog I wrote for exactly this reason:

```
python/.venv/bin/python scratchpad/warm_trajectory.py   # on main, then on the branch
diff /tmp/wt-base.txt /tmp/wt-new.txt
```

Cold solve, warm replay, warm replay **from file** (exercising the
signature round-trip), and the multistart ladder, at n = 4 / 10 / 50 /
200, recording status, objective **and iteration count**. **Empty diff.**
For the record:

```
hs071/cold    status=0 it=11 obj=17.0140171452     hs071/warm   status=0 it=1 obj=17.0140171420
quad10/cold   status=0 it=10 obj=0.0000000000      quad10/warm  status=0 it=1 obj=0.0000000000
quad50/cold   status=0 it=12 obj=0.0000000000      quad50/warm  status=0 it=1 obj=0.0000000000
quad200/cold  status=0 it=14 obj=0.0000000000      quad200/warm status=0 it=1 obj=0.0000000000
```

---

## 6. The #607 non-regression

#607 deliberately rejected any design where signing an artifact *with*
IDs becomes strictly worse than signing it without. The probe does not
reintroduce that asymmetry — it is symmetric, computed the same way on
both sides — and `test_signing_with_ids_is_never_worse_than_signing_without`
pins it: for five targets (same model, reordered, moved bounds, extra
nonzero, changed scaling), an ID-signed artifact replayed against a plain
`Problem` must be refused on exactly the same facets as one signed
without IDs.

I confirmed the test detects a break: making `var_ids` a required facet
(the exact design #607 rejected) fails it with

```
same model: IDs changed the verdict against a plain Problem
(['var_ids', 'con_ids'] vs []) — signing with IDs must not cost anything
```

**`var_ids` remain the rigorous answer, and the docs say so.** The probe
infers ordering from arithmetic, so a model genuinely symmetric under the
permutation is invisible to it —
`test_ids_still_beat_the_probe_where_the_probe_is_blind` builds exactly
such a model (`sum x²`, `sum x`, `dot(x,x)`: every function symmetric),
shows the probe accepts the permutation, and shows the IDs still catch
it. And only IDs let `reindex` **repair** a reordering rather than only
refuse it.

Where neither route was available on both sides, the report now says so
instead of returning a bare `"compatible"` (option 3 from the issue,
kept alongside the fix).

---

## 7. Tests run — counts I watched print

| command | result |
|---|---|
| `pytest python/tests -q -p no:randomly` | **1095 passed, 23 skipped** (517 s) |
| `pytest python/tests/test_warm_start_schema.py -q` | **47 passed** |
| `cargo fmt --all` | clean, no diff |
| `cargo clippy --workspace --all-targets` | 2 496 warnings — **all pre-existing**; `git diff origin/main -- crates/` is empty, so none can be mine |
| `bash scripts/check-release-consistency.sh` | **OK** (run although no version surface was touched) |
| `cargo build --release --bin pounce` | ok (needed for the sweep) |

**The 23 skips, honestly** — none are passes: 9 × no `torch`, 4 × no
`yaml`, 3 × missing `.nl` fixtures, 2 × no `pyomo`, 1 × no `sympy`, 1 ×
no `plotly`, 1 × needs `gamsapi[core]`. I installed **jax** specifically
so measurement D would be real rather than skipped; **torch is still not
installed, so the torch AD backend is unmeasured** (§8).

`cargo test --workspace` was **not** run — I changed no Rust, and `git
diff origin/main -- crates/ Cargo.toml Cargo.lock` is empty. Flagging it
rather than implying it passed.

---

## 8. What I did NOT do / what is still unmet

- **`torch` is not installed in this container, so measurement D covers
  jax only.** jax was bit-identical across fresh captures; I have **no
  measurement** for torch and am not going to claim one. A threaded torch
  backend is the most plausible candidate for evaluation noise above
  1e-9, and the jitter sweep (§4) says that would be refused. This is the
  weakest point in the change.
- **The probe cannot see a permutation that is a genuine symmetry of the
  model** at the probe point. Demonstrated, not hidden, by
  `test_ids_still_beat_the_probe_where_the_probe_is_blind`. For a fully
  symmetric model the reordered model *is* the same model and the replay
  is legitimate, so this is mostly benign — but a model symmetric only
  *at the probe point* is a real, if narrow, false-negative window. IDs
  close it; the docs say so.
- **Option 2 (auto-derived `var_ids` on Pyomo / the AMPL `.nl` path) was
  not implemented.** It is an alternative to the probe, not a gap in it,
  and the probe already covers the frontends it would have covered plus
  the raw-`Problem` path it could not. It would still be a real
  improvement — it enables `reindex` (repair) where the probe only
  refuses. I am leaving it for the parent to decide rather than filing
  anything, per the standing instruction.
- **`compat="strict"` gets stricter for artifacts captured on this
  build.** That is the point of the issue, but it is a behaviour change:
  a replay that used to be silently attempted now raises. `compat="warn"`
  / `"unsafe"` / `probe=False` / `migrate()` are the outs, all documented.
- **Forward-compatibility footgun, inherited from #607.** A signature
  written by this build carries a facet older builds do not know, and
  `ProblemSignature.from_json` **raises** on unknown facets by design
  rather than silently weakening the artifact. So a new-build artifact is
  unreadable by an old build. I did not change that policy — it is #607's
  deliberate choice — but I called it out in the CHANGELOG.
- **The probe also fires whenever the bounds move**, because the probe
  point is derived from the bounds. Harmless (the `bounds` facet already
  refuses those, and the report lists both) but it makes the report
  slightly redundant on that one case.
- **No benchmark-suite run.** The cost numbers in §5 are from my own
  timing harness at four sizes, not from `make benchmark`. I judged the
  fixture sweep plus the Python trajectory sweep sufficient given the
  change touches no solver input; a reviewer who disagrees should ask for
  the benchmark suite.
- **Process note:** partway through I ran `git checkout <file>` to undo a
  temporary test patch and reverted my own uncommitted work in
  `_warm_start_schema.py`. I reconstructed it and verified by re-running
  the full schema suite (47 passed) and the reproduction. Nothing is
  missing, but the incident is why the fix is committed in two steps
  rather than one.

---

## 9. Acceptance criterion, restated with a verdict

> *A reordered replay is refused, or accepted deliberately, without the
> user having to know to pass `var_ids` — and whatever it costs is
> measured on the fixture corpus, not asserted.*

**Met, with one qualification.**

- *Refused without `var_ids`* — **yes**, for the issue's own fixture and
  in general for any model that is not permutation-symmetric at the probe
  point. The reproduction that returned 16.0909 against 17.0140 now
  raises before the solver is entered.
- *Or accepted deliberately* — **yes**: `probe=False` at capture,
  `compat="warn"`/`"unsafe"` at replay, `migrate()` to re-sign.
- *Cost measured, not asserted* — **yes**: §5, including a real
  regression the measurement caught and fixed (2592 → 2920 → 2592), an
  empty 57-fixture sweep against a baseline built from `origin/main`, and
  an empty Python-side trajectory diff.
- **Qualification:** the guarantee is "refused unless the model is
  symmetric under the permutation", not "always refused". That residual
  is inherent to inferring ordering from arithmetic instead of from
  names, it is tested and documented rather than glossed, and `var_ids`
  remain the complete answer.
