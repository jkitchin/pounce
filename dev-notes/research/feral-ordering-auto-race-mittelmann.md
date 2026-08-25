# `feral_ordering=auto_race` on the Mittelmann NLP suite: a net loss

**Status: measurement record + doc change.** This note records the sweep
filed as [gh#768](https://github.com/jkitchin/pounce/issues/768) and the
documentation change it forced. It is a record of what was measured and
what the user-facing pages now say; it proposes no code change to POUNCE,
because the selection criterion at fault lives upstream in `feral` (§5).

## 1. What the docs used to say

`docs/src/options.md` closed the `feral_ordering` section with:

> When in doubt: leave `feral_ordering` at the default. When a hard problem
> looks linear-solver-bound, try `feral_ordering auto_race` before
> per-variant manual sweeping — it's the safe choice when the per-problem
> winner is uncertain.

and `docs/src/troubleshooting.md` routed "seconds-per-iter dominated by the
linear solve" straight to `auto_race`, calling it "the safe recipe … to
*measure* the right ordering rather than guess". The registered option help
(`upstream_options.rs`) said the same: "the safest choice when the
per-problem winner is uncertain."

## 2. Setup

| | |
|---|---|
| POUNCE | `5dc835f5`, `--release` |
| suite | Mittelmann NLP, 47 instances |
| threads | pinned to 1 |
| arms | ten options screened; each arm capped at 1.05× the default's wall clock |
| comparable | 42 of 47 instances yielded a default-vs-arm verdict |

Data: the 470-row options × problems matrix attached to gh#768, with
`linear_solver.last_nnz_l` recorded per arm.

## 3. The verdict

| verdict | count |
|---|---:|
| faster than default beyond the noise band | **1** |
| within noise | 14 |
| **slower** | **27** |
| median speedup | **0.82×** |

`auto_race` was the worst of the ten options screened. This is not a
population that is indifferent to the linear solver, which is what makes it
the relevant one: `LinearSystemFactorization` is 44–98 % of
`OverallAlgorithm` on 36 of the 47 instances — exactly the symptom the
troubleshooting page pointed at `auto_race`.

## 4. Two mechanisms, both structural

**It frequently buys nothing.** The deterministic `last_nnz_l` counter in
the JSON report shows the race returning the ordering `auto` had already
picked. On `ex8_2_2`:

| arm | `last_nnz_l` | `max_fill_ratio` |
|---|---:|---:|
| default (`auto`) | 85 662 | 2.3810 |
| `auto_race` | **85 662** | **2.3810** |
| `amd` | 85 662 | 2.3810 |
| `metis` | 85 293 | 2.3707 |

Same ordering, ~4× the symbolic cost to reach it: 0.32 s default vs 0.56 s.
Note it does not even take the marginally sparser METIS ordering — the fill
difference is under whatever margin it needs.

**Smallest `factor_nnz` is not smallest wall clock.** This is the deeper
one, and it bites hardest where the ordering matters most. From
`robot-abc-theta-max-stall.md` §5 (`robot_a`, 50 iterations, median of 3):
METIS and KaHIP carry ~1.9 % **more** nonzeros in `L` than AMD and factor
25–31 % **faster** — squarer fronts on a banded, nearly-1D KKT. The race
optimizes the wrong quantity, so it keeps AMD, and pays 4.06 s of symbolic
(against AMD's 0.56 s) to arrive at a 16.8 % regression.

## 5. Where ordering does pay — and the race still loses

Only where the factorization share is *extreme* rather than merely dominant.
`qssp180` (96 % factorization, `nnz_l` 63.8 M, fill ratio 33.9):

| arm | wall | speedup |
|---|---:|---:|
| default | 45.06 s | — |
| `feral_ordering=amd` | 14.63 s | **3.08×** |
| `feral_ordering=metis` | 23.6 s | 1.91× |
| `feral_ordering=auto_race` | 28.7 s | 1.57× |

`auto_race` is the worst of the three explicit choices on the instance where
racing should pay best. That is the cleanest statement of the defect: the
race is not a cheap way to find what a sweep finds, it is a different and
worse answer.

## 6. What changed, and what did not

Changed — the recommendation, in all three places that carried it:

- `docs/src/options.md`: the `auto_race` variant row now carries the
  measured verdict and the `robot_a` / `qssp180` counterexamples; the
  closing paragraph recommends sweeping the **concrete** variants under a
  capped `max_iter` and pinning the winner, gated on two cheap checks
  (factorization share, and `max_fill_ratio` / `last_nnz_l` from
  `--json-output`).
- `docs/src/troubleshooting.md`: the symptom-table row and the
  "FERAL ordering" recipe now say the same, with an explicit "why the sweep
  and not `auto_race`" paragraph. The heading — and therefore the anchor
  other pages link — is unchanged.
- `crates/pounce-algorithm/src/upstream_options.rs` and
  `crates/pounce-feral/src/lib.rs`: the registered option help and the
  `FeralConfig::ordering` doc comment no longer call the race safe.

Not changed — POUNCE's behaviour. `auto_race` is opt-in and nothing routes
to it by default; `FeralConfig::ordering` defaults to `Auto` and POUNCE
reads the option only when set explicitly. There is no trajectory change
here and the fixture sweep does not apply.

Not fixable here — the criterion. `OrderingMethod::AutoRace` is implemented
in `feral`, which is pinned to a published crates.io release. The upstream
ask is that the race select on something that tracks wall clock —
estimated factor flops, or one timed numeric factorization — rather than on
`factor_nnz`. Until then the honest description is "a one-shot diagnostic of
how much the four methods disagree about fill".

## 7. The process failure worth recording

`robot-abc-theta-max-stall.md` (§5, and its closing §7 item 3) recorded both
halves of this in advance: that `auto_race` selects the loser on `robot_a`,
and that the whole ordering story "needs measuring before `feral_ordering`
gets recommended anywhere user-facing." The recommendation shipped anyway,
on a mechanism argument — symbolic factorization is cached, so the ~4× is
paid once — that is true and does not reach the conclusion. On this suite
that one payment exceeds the saving on the median instance, and where there
is a real saving the race does not capture it.

A cached one-time cost is still a cost. "Cheap relative to the run" is a
claim about the numerator only; it needs the denominator measured on the
population you are about to point users at.
