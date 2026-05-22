# Development plan — closing capability gaps

**Status: development plan.** This is the execution half of the
research → plan → implement workflow. It sits below
`future-work-roadmap.md` (the *what and why*) and turns it into a
*data-gated order of work*. Nothing here is committed until the
Phase-0 failure inventory justifies it.

## 1. The benefit we are chasing

Two benefits were on the table:

1. **Slow problems solve faster.** Real, but *out of scope here* — it
   is a tuning/perf track, not a capability track.
2. **Problems that do not currently solve, solve.** This is the
   focus. Every item in this plan is justified only if it converts a
   genuine non-solve into a solve.

One nuance up front: a **timeout is a non-solve.** If a problem times
out because convergence is too slow, fixing it *looks* like benefit 1
but is really benefit 2 — the deliverable is a solve where there was
none. The dividing line is not "fast vs slow," it is "does an answer
come out." Plain speedups on already-solving problems are excluded;
turning a timeout into a solve is in scope.

## 2. Two tiers of "doesn't solve → solves"

A failure is only an opportunity if the problem is *actually
solvable*. That requires a reference solver. With pounce + IPOPT we
can already separate two tiers; a third reference (§5) is needed for
the rest.

- **Tier I — close the gap to IPOPT.** pounce fails, IPOPT solves.
  The problem is provably solvable by the *same method class* pounce
  implements, so the failure is a **port defect** — a bug in pounce,
  fixable in the current code with no new solver path. Cheapest,
  most certain wins. Do first.
- **Tier II — close the gap to KNITRO.** pounce *and* IPOPT both
  fail, but the problem is solvable (KNITRO or the literature solves
  it). This is a genuine **method blind spot** — one of the C1–C5
  classes — and needs the design-note work (composite-step,
  Interior/CG, penalty-IPM). Ambitious; gated on Phase 0.
- **Not an opportunity.** Both fail *and* the problem is genuinely
  infeasible/unsolvable. Detecting that correctly is already the
  right answer — not a gap to close.

The whole plan is: enumerate every failure, sort each into one of
these three, and only then schedule work.

## 3. Preliminary evidence — the curated 727-problem run

`benchmarks/cutest/results.json` (curated CUTEst subset, pounce vs
IPOPT+MA57, `n,m` up to ~4600 but mostly small) already gives a
first read.

Of **603 substantive problems** (excluding 124 that both solvers
flag `Not_Enough_Degrees_Of_Freedom` — a pre-solve classification,
not a failure):

- pounce **solves 567**.
- pounce **fails 36**.
- IPOPT succeeds on **only 3** of those 36.

**pounce is already at near-parity with IPOPT on this set.**

### 3.1 Tier I — the 3 port defects

| Problem | n | m | pounce | IPOPT |
|---|--:|--:|---|---|
| DMN15102LS | 66 | 0 | Timeout | Solve_Succeeded |
| PFIT4 | 3 | 3 | Infeasible_Problem_Detected | Solve_Succeeded |
| TAXR13322 | 72 | 1261 | Maximum_Iterations_Exceeded | Solved_To_Acceptable |

Three concrete, provably-solvable problems pounce gets wrong. PFIT4
is the most diagnostic: pounce reports the problem *infeasible* when
it has a solution — a false infeasibility, a real bug. These three
are fixable now, independent of every design note.

### 3.2 Tier II candidates — the ~34 shared failures

The 34 problems where pounce and IPOPT both fail, by status pair:

| Count | pounce / IPOPT status | Bucket |
|--:|---|---|
| 10 | Infeasible_Problem_Detected (both) | A — likely genuinely infeasible |
| 7 | Maximum_Iterations_Exceeded (both) | B — solvable, method stalls |
| 5 | Timeout (both) — all `n ≤ 99` | B — slow convergence, *not* linear algebra |
| 2 | Restoration_Failed (both) — S365, S365MOD | B — restoration-thrash |
| 2 | Diverging_Iterates (both) — MESH, TRO4X4 | B — globalization |
| ~8 | mixed (LHAIFAM, EQC, DMN/DIAMON LS, POLAK6, …) | B/A — triage individually |

Three buckets:

- **Bucket A — genuinely infeasible (≈10–12).** Both solvers detect
  infeasibility; `WACHBIEG` is literally the Wächter–Biegler
  hard-counterexample. Mostly *correct behavior*, not a gap — but a
  few need a false-positive check (`PFIT1`, given that its sibling
  `PFIT4` is a confirmed false infeasibility).
- **Bucket B — solvable, the IPOPT method stalls (≈18–22).** Max-iter,
  timeout, restoration-failure, divergence on problems that *have* a
  solution. **This is the Tier-II target population** and maps to
  composite-step (#4) and penalty-IPM (#5). The composite-step note's
  named targets `S365`, `S365MOD`, `PFIT*` are confirmed present here.
- **Bucket C — does not exist in this dataset.** The curated set has
  no large problems: every timeout above is `n ≤ 99` and times out
  from *iteration count*, not factorization cost. **The curated set
  cannot show Interior/CG value at all.** That evidence can only come
  from the full run.

### 3.3 What this already tells us

- Tier I is small and nearly done — 3 named bugs.
- Bucket B is the real prize and is **modest in size** (~20 problems
  on the curated set). Whether ~20 capability wins justify multi-month
  composite-step work is a genuine question — the full run must
  confirm the count is larger before that work is committed.
- **Interior/CG is entirely unproven by current data.** It must not
  be scheduled until the full run demonstrates a population of
  large-`n` failures whose cost is factorization-bound.

## 4. Phase 0 — the failure inventory (first deliverable)

The authoritative dataset, not the curated subset.

- **cutest-full** — all 1542 CUTEst problems, no size cap, pounce vs
  IPOPT+MA57. *Running now*; writes `benchmarks/cutest/full_results.json`.
  This is the run that surfaces Bucket C.
- **Mittelmann ampl-nlp** — 47 NLP instances at the 7200 s
  convention. Queued after cutest-full.

**Deliverable: a classification script + inventory.**
`benchmarks/cutest/classify_failures.py` reads `full_results.json`
(and the Mittelmann results), and produces `failure_inventory.md`:

1. Drop `Not_Enough_Degrees_Of_Freedom`-on-both (not failures).
2. Tier I — pounce fails / IPOPT solves → the port-defect list.
3. Tier II — both fail → bucket each as A (infeasible), B (solvable
   stall), or C (large-`n`, factorization-bound) using status pair,
   `n,m`, solve time, and `constraint_violation`.
4. Per-class problem lists — the named benchmark sets each design
   note will be validated against.

A failure is tagged Bucket C when `n` (or the KKT dimension `n+m`) is
large *and* the solve time is dominated by the linear-algebra phase —
that distinguishes a factorization-bound timeout (Interior/CG
territory) from an iteration-count timeout (globalization territory).

## 5. The third-solver control — decided: published data only

Buckets A and B cannot be separated with pounce + IPOPT alone: a
both-fail problem is either genuinely infeasible (A) or a solvable
method blind spot (B), and only a solver *outside* the IPOPT method
class can tell them apart.

A live KNITRO install is not available (licensing), so the control is
**published benchmark data** — there is no install option. Hans
Mittelmann's benchmark site publishes per-problem results for KNITRO,
LOQO, CONOPT and others, and the Hinder–Ye paper [hinder2018]
tabulates IPOPT-vs-others failure counts. This resolves the well-known
problems (`WACHBIEG`, the `*NE` family, `S365`) immediately and at no
setup cost.

The accepted limitation: coverage is partial and run conditions
differ. A both-fail problem that no published benchmark covers cannot
be definitively bucketed — it stays **provisionally A/B** and is
flagged for individual triage on problem structure (constraint count,
known infeasibility, literature). `failure_inventory.md` must carry a
**control column** — `published` vs `none` — so the confidence of
every Tier-II bucketing is explicit on its face.

## 6. Implementation phases — gated on Phase 0

Each design note's implementation is **gated**: it proceeds only if
the inventory shows enough Bucket-B/C problems, solvable by a
reference solver, in that class. Do not build a globalization spine
for three problems.

| Phase | Work | Gate to start |
|---|---|---|
| I | Fix the 3 Tier-I port defects (DMN15102LS, PFIT4, TAXR13322) | none — start now |
| II | composite-step #4, Phase 1 (normal step) | ≥ ~15 Bucket-B problems confirmed solvable, restoration/stall class |
| III | composite-step #4, Phases 2–3 (full TR) | Phase II shows real wins on its validation set |
| IV | penalty-IPM #5 | Bucket-B residual after composite-step still has restoration-thrash problems |
| V | Interior/CG / C5 | full run shows a Bucket-C population (large-`n`, factorization-bound) — **and** composite-step has landed |
| VI | MPCC / C4; active-set/SQP / C1 | a real MPEC / warm-start workload exists |

Phase I is unconditional and immediate. Everything else waits for the
inventory. This is the deliberate change from the roadmap's
provisional ordering: the roadmap ranks by *expected* impact; this
plan spends effort only on *measured* impact.

## 7. Per-move validation sets and success criteria

Once Phase 0 names the per-class problem sets, each phase gets a
concrete pass/fail bar. Provisional, from the curated data:

| Move | Validation set (provisional) | Success criterion |
|---|---|---|
| Tier-I fixes | DMN15102LS, PFIT4, TAXR13322 | all 3 solve, matching IPOPT's objective |
| composite-step #4 | S365, S365MOD, PFIT1, LHAIFAM, AVION2, HS87, PALMER5E/7A | ≥ half solve; **zero regressions** on the 567 already-solving |
| penalty-IPM #5 | restoration-thrash residual after Phase III | the residual shrinks; no false `LocalInfeasibility` |
| Interior/CG / C5 | the Bucket-C list from the full run | large-`n` timeouts become solves |

The non-negotiable on every phase: **zero regressions** on the
problems pounce already solves. Each new path is opt-in; the curated
suite is run both ways before any default changes.

## 8. Next actions

1. Let cutest-full and Mittelmann finish (in progress).
2. Cross-reference published KNITRO/Mittelmann data for every
   both-fail problem; record `published`/`none` in the inventory's
   control column (§5).
3. Write `classify_failures.py`; produce `failure_inventory.md`.
4. Triage the 3 Tier-I defects — these can be debugged now, in
   parallel with everything else, no dependency.
5. Read the Bucket-B/C counts; make the Phase-II and Phase-V go/no-go
   calls against the §6 gates.

## 9. Open questions / decision points

- **Both-fail problems with no published coverage.** §5's control is
  published data only; problems no benchmark covers stay provisionally
  A/B. How aggressively to triage that residual by hand — and whether
  an uncovered problem can ever justify Tier-II work — is open.
- **Bucket-B size threshold.** What confirmed count justifies the
  composite-step commitment? This plan suggests ~15; the full run
  may move it.
- **Interior/CG evidence bar.** How many factorization-bound large-`n`
  failures must the full run show before Phase V is committed? It must
  be a real population, not a handful.
- **The DMN/DIAMOND `*LS` family.** Six-plus problems, small `n`, that
  both solvers fail — a coherent hard-nonconvex least-squares cluster.
  Triage whether these are a fixable class or intrinsically
  multi-modal (in which case they belong to neither tier).
- **Regression budget.** Confirm "zero regressions" is the bar, or
  whether a small, named, reviewed regression set is acceptable in
  exchange for a larger capability gain.
