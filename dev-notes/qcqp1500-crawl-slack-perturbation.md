# qcqp1500-1nc crawl: slack-block perturbation and the centrality floor (gh#1098)

## The symptom

On the default exact-Hessian path, Mittelmann `qcqp1500-1nc` (1500
variables, 10 508 constraints) sat at μ = 1.8e-6 for about 1200 iterations
and hit the 300 s CPU limit. During that stretch `inf_pr` oscillated between
1e-3 and 1e-2, the step fractions were tiny, and `δ_w` was non-zero almost
every iteration. Ipopt 3.14 crawls the same way: it was still unsolved at
1800 s, after 2827 iterations. #1077 put pounce on Ipopt's trajectory, so
it inherited the crawl.

## What the crawl is

Instrumenting the crawl iterates gave this picture:

- It is a walk from vertex to vertex of a nonconvex polytope, at about 15
  iterations per net change of active set.
- **Fraction to the boundary.** When a blocking pair (slack, multiplier)
  sits on the central path, `s·v ≈ μ`, the τ rule
  (`τ = max(τ_min, 1 − μ)`) cuts it to about `μ²`. Recentering it then
  costs 5–10 iterations as soon as the next direction reverses.
- **Inertia correction.** It adds `δ_x` to the Hessian block and,
  following Ipopt, the same `δ_s = δ_x` to the slack block.
  - The slack block already carries `Σ_s = v/s`. That is small on inactive
    inequalities, so the added `δ_s` mostly stiffens the inactive slacks.
  - Across thousands of inactive rows, that term dominates the curvature
    along a direction.
  - Successive regularized directions then disagree, and the sawtooth in
    `δ_w` follows.

**Refuted.** A first explanation was that the crawl was `δ_w` on the slacks
alone, removable by a smaller `δ_s` applied only once a stall is detected.
A stall-triggered prototype did not get out of a crawl that had already
started. A small `δ_s` has to be in effect from the start, before the
trajectory reaches the slow basin.

## The two remedies

1. **`perturb_slack_fact`**, so that `δ_s = fact · δ_x`. Upstream is 1.
   Wired through `PDPerturbationHandler::delta_s_fact`. It changes only
   the slack block. `δ_x` stays on the Hessian, where the inertia defect
   is, and the inertia test still decides whether a perturbation is
   accepted.
2. **`ftb_centrality_floor` (γ)**: Mehrotra's step-length heuristic
   (Mehrotra 1992 §5; Wright, *Primal-Dual Interior-Point Methods*,
   eq. 10.9).
   - It keeps a `1 − γ` floor on the fraction, and steps beyond it only
     while the blocking pair's product stays `≥ γμ`.
   - Only pairs that are currently centered get the floor.
   - The floored step is capped at the τ rule's step, so it can only
     shorten.
   - It is not applied in restoration.
   - See `crates/pounce-algorithm/src/ftb_floor.rs`.

## Why the floor is capped and why it is off

The first version was not capped.
`issue_390_nonlinear_equality_scale` caught the problem.

- At iteration 0 (μ = 0.1) the blocking pair had `s·v = 9`.
- "Keep `s·v ≥ γμ`" then allowed a longer dual step than τ.
- That put the multiplier at `γμ/s` instead of about `μ/s`: less central
  than upstream leaves it, which is the opposite of what the rule is for.
- At row scale 1e-8 the infeasibility certificate became
  `Restoration_Failed`.

Capping fixed that test. Three others still change verdict with γ = 0.01:

- `issue_884_biactive_dual_divergence`: `ErrorInStepComputation`, where the
  fixture is supposed to reach a certificate without the retry.
- `masked_certificate_fuzz`: with the certificate veto off and
  `mu_strategy=adaptive`, it reports `SolveSucceeded` at f = 0.125 against
  a bound optimum of 0.1. Under the default veto the answer is still
  correct.
- `line_search_options_wiring`: the fixture's trajectory no longer reacts
  to `filter_max_margin`.

In the fixture sweep, γ = 0.01 also turns `discs` into a final
`InfeasibleProblemDetected`. So γ stays opt-in, at default 0.

## Measurements

Baseline: main `26991b7b`.

| config | γ | slack fact |
|---|---|---|
| main | 0 | 1 |
| A | 0 | 0.01 |
| B | 0.01 (capped) | 0.01 |
| C (best opt-in) | 0 | 0.03 |

The three lanes ran at the same time, so the times are approximate.

**Mittelmann:**

| config | solved | iterations | time |
|---|---|---|---|
| main | 44/47 | 5182 | 2216 s |
| A | 45/47 | 4324 | 1818 s |
| B | 46/47 | 4575 | 1814 s |
| C | 45/47 | 4355 | 1674 s |

| model | main | A | B | C |
|---|---|---|---|---|
| qcqp1500-1nc | CPU limit | 199 | 143 | 228 |
| qcqp1500-1c | CPU limit | CPU limit | 148 | CPU limit |
| henon120 | 267 | 67 | 69 | 80 |
| lane_emden120 | 84 | 21 | 21 | 25 |
| qcqp500-3c | 330 | 138 | 140 | 177 |
| qcqp750-2c | 75 | 218 | 307 | 198 |
| dirichlet120 | 56 | 66 | 154 | 79 |

**Water:** all configs solve 6/6.

| config | total iterations | waterx objective |
|---|---|---|
| main | 1401 | 1019.7 |
| A | 1248 | 927.1 |
| B | 1336 | 963.9 |
| C | 1331 | 1151.5 |

**Fixture sweep, 252 lines.** All lines that move are on the exact leg.

| config | lines moved | status changes and retries |
|---|---|---|
| A | 14 | `cresc4` and `discs`: false `Infeasible_Problem_Detected` on the first attempt, rescued by the second-opinion ladder (total 824 and 484, against 69 and 177 on main) |
| B | 47 | `discs`: false infeasible, final status |
| C | 14 | no status loss and no new retry; `ssebnln` acceptable → optimal |

Config C, every line that moves:

- **Different minimum** (nonconvex models):
  - `autocorr_bern55-06`: better, −2368 against −2304.
  - `hs13_bigstart`: 0.98168 against 0.98493.
- **Same answer, fewer iterations:**
  - `pooling_rt2stp` 201 → 130
  - `discs` 177 → 166
  - `haverly_multistart_t3_brf0` 37 → 31
- **Same answer, more iterations:**
  - `csfi2` 21 → 37
  - `deb7` 180 → 195, still acceptable
  - `mpcc_qpec_small_biactive` 29 → 31
  - `nonconvex_qcqp` 59 → 61
- **Last digit of the objective only:**
  - `cresc4`
  - `nonconvex_qp_ineq`
- **Unbounded models**, same status, diverging at a different rate:
  - `unbounded_cubic` 290 → 65
  - `unbounded_exp` 23 → 31
- **`ssebnln`**: acceptable → optimal. It still uses the `mu_strategy`
  retry.

**The value does not respond smoothly.** On `cresc4`, `discs` and
`qcqp1500-1nc`:

- 0.01: false first-attempt infeasibility on two fixtures.
- 0.03: clean.
- 0.1: `discs` locally infeasible.
- 0.3: `cresc4` needs a retry, and `qcqp1500-1nc` takes 446 iterations /
  275 s.

0.03 was picked from those runs and then confirmed on the full sweep,
Water and Mittelmann. It is a measured choice, not an optimum.

## Why both options ship opt-in

With 0.03 as the default, the fixture sweep was clean, but 22 `pounce-cli`
regression tests that pass on main failed (`cargo test --workspace`).

- **Most are premise tests.** `csfi2` now converges on the first attempt,
  so the retry, fallback and one-verdict tests built on it (gh#748, #757,
  #1042, #616, `one_verdict_per_run`) no longer reach their branch. Two
  witness tests (#534, #592) also lose their premise.
- **Some assert an answer:**
  - `hs13` from the remote start lands at 0.98168, where Ipopt gets 0.98493.
  - `issue_250` reads `autocorr_bern55-06` at −2368 as a worse basin.
  - `mpcc_qpec_small_biactive` fails under the options the #884 and #981
    tests use.

The sweep runs every fixture at stock options. These tests run a few
fixtures at non-default options and check specific branches, so they see
what the sweep cannot.

Both options therefore default to upstream's behaviour (`1` and `0`), and
the workspace tests pass unchanged. What is still needed to make 0.03 the
default is recorded in `dev-notes/known-limitations.md`.

## Lesson: diff the `2nd=` column too

My first reading of config A's sweep compared only status and `it=`, and
reported "nothing that converged stopped converging". That was true of the
final status. But two fixtures had started being declared infeasible on the
first attempt, and the second-opinion ladder had hidden it. A first attempt
that fails and is rescued only shows in the `2nd=` column, so diff it along
with the other columns.
