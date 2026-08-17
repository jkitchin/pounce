VERDICT: improved

# pounce#611 — warm-start benchmark coverage, and the composite report it promised

Closes the coverage gaps pounce#611 lists, adds an external-solver
baseline, and publishes a composite report **plus the machine-readable
raw results it is computed from**. No Rust changed; `git diff
origin/main -- crates/ Cargo.toml Cargo.lock` is empty.

## Provenance — where every number came from

| | |
|---|---|
| `main` at | **`eeca47f922372834a7e8d925ce7fe7b6160c8c7b`** (`eeca47f9`) — unmoved for the whole session, so the merge-up was a clean no-op and no `CHANGELOG.md` conflict arose |
| branch | `claude/pounce-issue-611-loop` at **`fe6c658`**, 13 commits over `main` (11 non-merge) |
| tree measured | **`73eec883`**, as the report's own provenance table records |
| build | **release** — `cargo build -p pounce-cli --release` + `maturin develop --release` |
| external solver | cyipopt 1.7.0 against **Ipopt 3.11.9 on MUMPS** (no HSL) |
| coverage | the documented `--scales all` sweep at the default tier: 22 families × 3 scales × 15 arms = 66 cells, 1335 steps per arm, **all 66 completed** |

**Every number here was measured twice, by two independent sweeps.**
`73eec88` ("`resistive_network` was infeasible
at V ≥ 120") was pushed to this branch by another session while my
first sweep was running, and it changes a family the sweep measures. I
merged it and re-ran the entire pipeline rather than publishing numbers
taken on the superseded family. That session then independently
generated and pushed its own post-fix composite, and the two runs agree
**exactly** on every noise-free column. Details in § *The report was
regenerated mid-flight, then independently corroborated* below.

Since no Rust differs from `main`, the solver behaviour measured here
*is* `main@eeca47f9`. Sibling PRs **#638**
(`crates/pounce-algorithm/src/init/warm_start.rs`) and **#639**
(`python/pounce/_warm_start*.py`) will both move arms measured here;
because every table is generated rather than transcribed, re-measuring
after they land is a re-run of the pipeline below, not an editing pass.

The opt-in `--tier large` families (`mpc_horizon_200/400/800`,
`elliptic_control_600`, `resistive_network_800`) are **not** in this
report, exactly as the documented pipeline specifies. To add them:
`python -m warmstart.run --tier large --scales all --arms all`.

## What this branch delivers

- **Five new initialization arms**, completing the nine pounce#611
  lists: `warm-ipm-primal`, `warm-ipm-norecenter`, `cold-ipm-lsq`,
  `race-fixed`, `race-halving`. Racing is charged for its tournament via
  `StepResult.init_time`.
- **Families outside the ones warm-start work chose**:
  `rastrigin_drift`, `rastrigin_scatter`, `elliptic_control_*`,
  `resistive_network_*`, `badly_scaled_qp`.
- **An external solver arm** — `adapters/ipopt_adapter.py` drives Ipopt
  through the *same* callback object as the pounce adapter, so
  evaluation counts mean the same thing on both sides. Arms Ipopt has no
  counterpart for are skipped **with a recorded reason**, not dropped.
- **One correctness gate for both solvers** — `kkt.py` recomputes dual
  feasibility, primal feasibility and complementarity from the returned
  point, so neither solver's status line decides whether its own step
  counted.
- **Changed-structure arms** — `transfer.py`: horizon shift via
  `WarmStart.reindex`, mesh prolongation via `WarmStart.transfer`.
- **The composite report** — `composite.py`, wired into
  `dev-notes/warm-start-benchmark.md`.

## The generated report

Committed at **[`dev-notes/warm-start-611-composite.md`](dev-notes/warm-start-611-composite.md)**
with its machine-readable twin
**[`dev-notes/warm-start-611-composite.json`](dev-notes/warm-start-611-composite.json)**.
Regenerate both with:

    cd benchmarks
    python -m warmstart.run      --scales all --arms all --out warmstart/results.json
    python -m warmstart.run      --solver ipopt --scales all --arms all \
                                 --out warmstart/results-ipopt.json
    python -m warmstart.transfer --experiment all --out warmstart/transfer.json
    python -m warmstart.composite --results  warmstart/results.json \
        --ipopt warmstart/results-ipopt.json --transfer warmstart/transfer.json \
        -o ../dev-notes/warm-start-611-composite.md \
        --json-out ../dev-notes/warm-start-611-composite.json

Headline numbers, all from that report. `speedup` is the shifted
geometric mean of cold/warm iterations, each arm against its own cold
counterpart; **`bad` counts steps that failed to converge or landed on a
worse optimum, and a nonzero entry voids the speedup on its row**.

| arm | steps | iters | solve s | speedup | bad |
|---|--:|--:|--:|--:|--:|
| `cold-ipm` | 1335 | 15640 | 12.31 | — | 0 |
| `cold-sqp` | 1335 | 24052 | 233.36 | — | 186 |
| `warm-ipm` | 1335 | 5825 | 6.23 | **3.05** | 32 |
| `warm-sqp` | 1335 | 21653 | 173.15 | 1.32 | 207 |
| `warm-ipm-primal` | 1335 | 9056 | 8.49 | 1.90 | 30 |
| `warm-ipm-norecenter` | 1335 | 6054 | 6.48 | 2.90 | 34 |
| `cold-ipm-lsq` | 1335 | 15195 | 12.29 | 1.03 | 0 |
| `warm-qp-ipm` | 789 | 5007 | 5.13 | 1.76 | 1 |
| `race-fixed` | 1335 | 12513 | 10.69 | 1.27 | 35 |
| `race-halving` | 1335 | 13436 | 12.84 | 1.20 | 75 |
| `pred-ipm` | 540 | 2204 | 3.15 | 3.09 | 0 |
| `predcorr-ipm` | 540 | 2399 | 3.23 | 2.86 | 0 |

Carrying the **dual** state is worth about 1.6× on its own —
`warm-ipm` 3.05× against `warm-ipm-primal` 1.90×, from the same primal
point. Against the external baseline, on identical callbacks:

| arm | solver | iters | solve s | bad | speedup |
|---|---|--:|--:|--:|--:|
| `cold-ipm` | pounce | 15640 | 12.31 | 0 | — |
| `cold-ipm` | ipopt | 15739 | 10.20 | 0 | — |
| `warm-ipm` | pounce | 5825 | 6.23 | 32 | 3.05 |
| `warm-ipm` | ipopt | 5579 | 4.45 | 35 | **3.14** |
| `warm-ipm-primal` | pounce | 9056 | 8.49 | 30 | **1.90** |
| `warm-ipm-primal` | ipopt | 11918 | 8.13 | 38 | 1.30 |

**This is not a "state of the art" claim, and the report does not make
one.** Cold, the two are within 0.6% on iterations. Warm, **Ipopt warm
starts marginally better than pounce** (3.14× against 3.05×) and is
ahead on wall time. pounce's one clear win is `warm-ipm-primal`, 1.90×
against 1.30× — with only a primal point to work from it recovers more
than Ipopt does. The wall-time column compares against a MUMPS-linked
Ipopt 3.11.9, not a licensed HSL build, which the report labels in
place.

## Two findings that must not get lost

### 1. The horizon-shift arm is a negative result

The concern that the suite "understates what an MPC implementation would
do" is now measured, and it does not. Shifting the previous horizon by
one stage through `WarmStart.reindex` **loses** to simply carrying the
previous solution unshifted:

| experiment | cold | carry | shift |
|---|--:|--:|--:|
| `nmpc_vanderpol` (genuinely closed-loop) | 136 | **71** | 73 |
| `mpc_horizon_10` | 94 | **23** | 87 |
| `mpc_horizon_20` | 103 | **48** | 92 |
| `mpc_horizon_40` | 125 | **59** | 114 |

iterations over steps 1+. The unseeded final stage costs more than the
shift saves; and on `mpc_horizon_*` the path is a *rotation* of the
initial state rather than a receding horizon, so the shift is not even
the operation relating consecutive problems — hence the much wider gap
there. Mesh prolongation, by contrast, works: `prolong-dual` cuts
`elliptic_control_80->161` from 16 cold iterations to **3**.

### 2. `elliptic_control_*` exhausts the inner QP budget — this is `main`'s behaviour, not this branch's

`cold-sqp` does not solve `elliptic_control_*` at all under the
conventional phase-1/phase-2 inner QP. On `elliptic_control_40 @ small`
it returns `Maximum_Iterations_Exceeded` with **zero completed outer
iterations**, `n_qp_solves = 1`, `n_qp_ws_changes = 88`. I re-checked
the "budget is the inner QP's" diagnosis directly: raising
`sqp_max_iter` 10× (500 → 5000) changes **nothing** — same status, same
`iters = 0`, same `n_qp_solves = 1`, same `n_qp_ws_changes = 88`. The
homotopy twin solves the identical problem in **one** outer iteration to
a KKT residual of **3.1e-13**.

It scales with the mesh, and at the finest default-tier mesh the
homotopy stops rescuing it:

| family @ small | `cold-sqp` | `cold-sqp-hom` |
|---|---|---|
| `elliptic_control_40` | 20 steps, 10 exceed max-iter | **all 20 solve**, 1 iter, KKT 3.1e-13 |
| `elliptic_control_80` | 20 steps, 18 exceed max-iter | **all 20 solve**, 1 iter, KKT 1.7e-12 |
| `elliptic_control_160` | 20 steps, **20** exceed max-iter | **20 exceed max-iter** |

Reproduce with:

    cd benchmarks && python -m warmstart.run --families elliptic_control_40 \
        --scales small --arms cold-sqp,cold-sqp-hom

This is behaviour of `main`, not of anything pounce#611 changed — no
Rust was touched. Recorded here rather than filed, per the repo owner's
standing instruction.

## The report was regenerated mid-flight, then independently corroborated

While the first sweep was running, another session pushed `73eec88` to
this branch: `resistive_network` was **infeasible at V ≥ 120**. My
first sweep had already measured the broken family, and my own data
confirms the diagnosis exactly — `resistive_network_120` returned
`Infeasible_Problem_Detected` on **all 780** of its steps (20 per scale
× 3 scales × 13 arms), while `resistive_network_60` was clean on all
780 of its own.

My push was rejected as non-fast-forward. I merged (`b6408aa`, no
conflicts; **never force-pushed**) and re-ran the whole four-stage
pipeline. All 780 steps now solve, and the new families pass
`warmstart.selftest`'s finite-difference checks.

**The stale rows were not confined to their own family.** An infeasible
instance is a failed step for every arm that meets it, so those 60
per-arm failures were sitting in the sweep-wide `bad` column:

| arm | bad, first run | bad, after the fix |
|---|--:|--:|
| `cold-ipm` | 60 | **0** |
| `cold-ipm-lsq` | 60 | **0** |
| `pred-ipm` | 60 | **0** |
| `predcorr-ipm` | 60 | **0** |
| `warm-ipm` | 92 | 32 |
| `cold-sqp` | 246 | 186 |
| `warm-sqp` | 267 | 207 |

Four arms that looked like they failed 60 steps apiece were in fact
clean on every step they were given. The headline warm-start speedup
moved the *other* way — `warm-ipm` 3.23× → **3.05×** — because the
removed steps were ones cold could not solve either. Had I published
the first run, the report would have overstated the speedup and
libelled four arms as failing. The three claims in
`dev-notes/warm-start-benchmark.md` are unmoved, being deterministic
iteration counts on families the fix does not touch.

**Then the same session pushed its own post-fix composite** (`51a6881`)
while I was committing mine. Rather than a collision to paper over,
this is free corroboration: two sweeps run independently, in different
containers, agree **exactly** — arm for arm across all 15 arms and 1335
steps — on iterations, evaluation counts, speedups, `bad`, `better` and
max-KKT. The *only* divergence is wall time (`cold-ipm` 11.86 s against
12.31 s; `cold-sqp` 208.21 against 233.36), which is the 10–30%
between-run drift the suite already documents and precisely why it
tells the reader to read ratios rather than milliseconds.

The committed artefact is **their** version (`fe6c658` resolves the
add/add conflict to it): both are post-fix and equivalent, and its
provenance string is the plain `73eec883` rather than a merge commit.
`report.py`, which my fix below touches, is not on `composite.py`'s
code path, so it cannot account for any difference. The wall-time
figures quoted in this report are therefore the committed ones.

## A defect I found and fixed on the way

The reproduction command above — documented by this branch in
`dev-notes/warm-start-benchmark.md` — **had never worked**.
`warmstart/report.py` selected its homotopy section on `cold-sqp-hom`
being present, then read all four of `cold-sqp`, `cold-sqp-hom`,
`warm-sqp` and `warm-sqp-hom` unconditionally, so any narrowed `--arms`
without the warm twins died with `KeyError: 'warm-sqp'` — *after* every
solve had run, at the point of writing results out. Fixed in `ec861f6`:
the warm pair is looked up defensively and renders as `—` when absent.

`report.py` is `main`'s file, untouched by this branch, so the bug is
pre-existing; this branch is what documents a command that triggers it.
Verified byte-identical output on the full 66-cell sweep across the
change, so nothing that already worked moved.

## Acceptance criteria

> **1. Reproducible scripts and machine-readable raw results.**

**MET.** The scripts were already committed; as of this branch the
*results* are too — `dev-notes/warm-start-611-composite.json` (93 KB,
parses with `json.load`, top-level keys `arms`, `falsification`,
`ipopt_arms`, `ipopt_skipped`, `issue_arm_coverage`, `profiles`,
`schema`, `sources`, `transfer`) alongside the rendered `.md`. Both
links in `dev-notes/warm-start-benchmark.md` now resolve; before this
branch's final commit they were dead. The four-stage pipeline is
documented verbatim and was run end to end to produce exactly these
files.

> **2. Equal stopping criteria and clearly documented solver-specific settings.**

**MET.** The report opens with a settings table: `tol` 1e-8,
`constr_viol_tol` 1e-6, `max_iter` 500 for **both** solvers, and a
harness converged-gate (KKT ≤ 1e-4, viol ≤ 1e-5) applied identically to
both from the returned point via `kkt.py` — neither solver's own status
line decides whether its step counted. Solver-specific settings are
listed as such and marked where they are *not* equal: pounce's
`warm_start_recentering`, Ipopt's `warm_start_bound_push`/slack/mult at
1e-9 (the 1e-2 default discards most of a warm start), and the linear
solver — MUMPS versus pounce's own sparse LDLᵀ — flagged **"not equal —
see caveats"** rather than papered over.

> **3. Separate cold-start, repeated-solve, and changed-structure claims.**

**MET.** Three separate result sets, produced by different code paths:
cold arms (`cold-*`) and repeated-solve arms (`warm-*`) in
`warmstart.run`, and changed-structure in `warmstart.transfer`, which
gets its own report section and its own JSON. The changed-structure
claim is reported as a **loss** where it is one (§ Finding 1 above),
which is the point of keeping it separate.

> **4. Composite report wired into the existing benchmark documentation.**

**MET.** `dev-notes/warm-start-benchmark.md` links both artefacts by
name, states that every table is computed from the three raw JSON files,
and documents the regeneration pipeline. It deliberately stays out of
`BENCHMARK_REPORT.md`, which is per-problem cold-solve rows against an
Ipopt reference and has no shape for a per-sequence result.

> **5. Any "state of the art" claim is scoped to the tested workload and supported by the published data.**

**MET — by making no such claim.** The suite now contains families built
specifically to make warm starting *lose*, and they do: on
`rastrigin_scatter @ small`, `warm-sqp` is **bad on 20 of 20 steps** and
`warm-ipm` on 17 of 20, while converging cleanly — a wrong-basin step
never appears in a status code, only in the `bad` column. The report's
falsification section carries the instruction to read it *before quoting
any speedup from this suite*. And the external comparison is published
as it came out, which is not flattering: **Ipopt warm starts marginally
better than pounce on this workload** (3.14× against 3.05×) and is
ahead on wall time. That is stated plainly in the report alongside the
MUMPS-not-HSL caveat, rather than being scoped away.

## Checks run

| check | result |
|---|---|
| `bash scripts/check-docs-consistency.sh` | **OK** — "46 pages, all reachable from SUMMARY.md, no dead links" |
| `bash scripts/check-release-consistency.sh` | **OK** — versions agree at 0.10.0 across `Cargo.toml`/`python`/`pyomo-pounce`; publish list 20 crates, topological; docker ARG matches |
| `cargo fmt --all` | clean — exit 0, `git status --porcelain` empty afterwards |
| `git diff origin/main -- crates/ Cargo.toml Cargo.lock` | **empty**, as expected — no Rust touched |
| `python -m warmstart.selftest` | **All families pass** — finite-difference checks on all 23 registered families |
| `python -m pytest tests -q` (python/) | **1082 passed, 23 skipped, 0 failed** (479 s) |
| full 4-stage pipeline | all 66 cells completed, exit 0 at every stage |

The compiled extension was **rebuilt** in release rather than bypassing
the staleness guard.

One note on that pytest line, since the first run of it was red. With a
bare environment,
`test_continuation.py::test_step_controller_is_what_the_jax_follower_uses`
fails rather than skipping: it guards with
`pytest.importorskip("pounce.jax._path")`, but `pounce/jax/__init__.py`
re-raises the missing-JAX `ModuleNotFoundError` as a chained "useful
error", and `importorskip` does not treat that as a skip. Installing
JAX turns it green. This branch touches **no** file under `python/`
(`git diff --stat origin/main...HEAD -- python/` is empty), so it is an
environment artefact and not something this branch caused — but a guard
that fails instead of skipping when its optional dependency is absent
is arguably a real if minor defect in `main`. Not filed, per the
standing instruction; noted here for the parent.

## What I did NOT do

Filed as report content rather than issues, per the repo owner's
standing instruction. The parent decides.

- **The `--tier large` families were not run.** The documented pipeline
  does not include them and neither does this report. `mpc_horizon_800`
  is n = 2402; the default-tier sweep alone took ~53 min of the wall
  clock here, with `elliptic_control_160` at ~400 s per cell. Command to
  add them is in § Provenance.
- **`elliptic_control_160` is unsolved by both SQP arms** (20/20 steps
  exceed max-iter on `cold-sqp` *and* `cold-sqp-hom`). The doc's claim
  that the homotopy variant rescues `elliptic_control_*` holds at meshes
  40 and 80 and stops holding at 160. Evidence in the table above; not
  chased further.
- **`cold-sqp` still carries `bad = 186` and `warm-sqp` `bad = 207`**
  across the sweep after the `resistive_network` fix, now dominated by
  the `elliptic_control_*` and `rastrigin_*` rows. Not investigated
  beyond the two findings above.
- **Nothing from the "Still not covered" list below was closed.**

### Still not covered

Carried forward verbatim from `dev-notes/warm-start-benchmark.md` — it
is honest and it belongs here:

- **A second external solver.** nlopt 2.11.0 and casadi 3.7.2 (which
  bundles its own Ipopt) both install cleanly here, but neither was
  wired to an adapter. nlopt has no dual warm-start API at all, so it
  would only populate the "previous primal only" arm; casadi's Ipopt
  would duplicate the cyipopt arm through a different binding.
- **DAE discretizations.** `elliptic_control_*` is an elliptic PDE.
  Nothing in the suite is an index-1 DAE or a collocation
  transcription, which behave differently under a warm start because
  the algebraic variables have no dynamics to carry them.
- **Memory peak.** The issue lists it as a metric. It is not measured:
  a Python-level `tracemalloc` figure would report the harness's
  allocations rather than the solver's, and RSS is too coarse at these
  problem sizes to separate the arms.
- **HSL.** The Ipopt arm runs on MUMPS. Wall-time comparisons against
  it are comparisons against a MUMPS build, and are labelled that way
  in the report.

One addition from this run: the Ipopt available here is **3.11.9**
(Debian/Ubuntu packaging), which has **no sIPOPT**, so the `pred-ipm`
and `predcorr-ipm` sensitivity arms have no external counterpart and are
recorded as skipped with that reason rather than compared.
