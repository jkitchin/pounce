# The convex/conic sensitivity arm

Companion to `sensitivity-design.md`, which covers the NLP arm and is
Pyomo-scoped. This one records the decisions behind `QpSensitivity` and the
shared core, and — more usefully — the **measurements** that decided them.
The user-facing description lives in `docs/src/convex-solver.md` and
`docs/src/sensitivity.md`; what is here is the reasoning a reader needs
before *changing* any of it.

## The finding the whole thing rests on

The NLP crate's machinery was already generic over a trait. `SensBacksolver`
has two required methods — `dim()` and `solve(rhs, lhs)` — and
`boundcheck`'s fix-relax / path / directional routines are generic over it,
derive `n_x` from slice lengths, and read bounds only through
`BoundRow { row, var_row, lower }`. They are layout-agnostic.

So the convex arm never needed a port. It needed to become an *implementor*.
That is why this work is a few hundred lines of adapter and several thousand
lines of test rather than the reverse.

`pounce-sens-core` is the extraction: the trait, `boundcheck`, the
`P = K⁻¹A` / Schur stack, the activity **rule**, and, behind the
`BarrierIterate` trait, everything the NLP arm computes over a converged
factor (`ConvergedSolve`), with the statistics and steps behind
`pounce.sensitivity`. Both arms depend on it, and neither depends on the
other. Its own dependencies are `pounce-common` and `pounce-linalg`,
`pounce-qp` and `pounce-feral` for the directional derivative's reduced QP,
and `faer` for the statistics' dense algebra.

## What deliberately did not unify

- **Activity classification's plumbing across the two arms.** The *rule*
  moved (`activity_kernel`), and so did the NLP arm's classification, which
  reads `Σ`, `q` and `μ` through the `BarrierIterate` trait. The convex arm
  does not implement that trait. It reconstructs the same quantities from
  `(problem, solution)` because `QpSolution` carries no iterate. Sharing the
  rule is what stops the arms disagreeing about what a kink is.
- **The two reduced Hessians.** sIPOPT's Schur route and a null-space
  projection are different computations behind one word. They stay separate,
  and the CLI routes `--compute-red-hessian` to the NLP arm for that reason.

## Cone faces: the part that is genuinely new

A cone's active object is not a set of rows. Its slack sits on a **face**,
and every family splits three ways — `Interior`, `Apex`, `Boundary`. The
boundary rows per family are in the book; two things are not.

**The curvature is part of the answer, not a refinement.** Every orthant row
and every variable bound is a hyperplane, so the sensitivity KKT's `(x,x)`
block had always been `P` alone. A conic boundary face is curved and
contributes its own Hessian. Written without it — the natural first draft,
by analogy — the step converges to the **wrong derivative**: `dx/db` reads
`(0.348, 0.652)` against a closed-form `(0.5, 0.5)`, at every `δ`, with every
internal residual clean, because the step solves exactly the KKT it was
handed and that KKT is not the problem's.

What caught it was the re-solve oracle in
`crates/pounce-convex/tests/convex_soc_sensitivity.rs` — the only guard in
the crate that reads a number the sensitivity layer did not produce. That is
the same thesis `sens_resolve_oracle.rs` was written for on the NLP arm, and
it earned its place here on the first day it existed.

**Measured, and worth keeping:** flipping the sign of the boundary normal
leaves the closed-form-derivative assertion **green** and reddens only the
oracle-backed tests. A hand-derived expectation is not a substitute for an
outside number.

## Thresholds, and the populations behind them

Two are calibrated against the **non-symmetric** HSDE driver, whose accuracy
is well short of the symmetric IPM's. Measured across four fixtures
(exponential ×2, `Power(0.6)`, `Power(0.3)`) at `tol` `1e-9` and `1e-11`:

| quantity | measured range | constant | margin |
|---|---|---|---|
| `\|φ\|/primal_scale` | `4.1e-10` … `2.1e-9` | `FACET_ACTIVE_REL = 1e-6` | ~500× |
| `‖z − ν∇φ‖∞ / max(‖z‖∞, dual_scale)` | `2.8e-8` … `3.4e-5` | `FACET_DUAL_REL = 1e-3` | ~30× above, ~1000× below a genuine mismatch |

`FACET_DUAL_REL`'s first value, `1e-6`, refused **two of the four correct
solutions**. That is the argument for measuring rather than picking a round
number that looks safe.

The apex/boundary decision is relative to the problem's `primal_scale`, the
same quantity the orthant guard uses, so the two cannot disagree about what
"zero" means on one solution.

## Where a fixture went blind, three times

This is the most transferable thing in the file. Each of these was a
**green mutation** — a deliberate defect no test caught — and each was found
by running the mutation rather than predicting it.

1. **`λ_k / a_l` → `λ_k · a_l`** in the PSD curvature. Green across the whole
   crate, because the fixtures' surviving eigenvalue was exactly `1.0` and
   the two expressions coincide there. Fixed by retuning the objective so it
   is `3`, with `the_psd_fixtures_have_a_nonunit_curvature_scale` as a
   precondition.
2. **`primal_scale` → `1.0`** in `build_conic`. Green, because every fixture
   was `O(1)` where the relative and absolute readings coincide. Fixed by an
   `O(1e6)` fixture where they disagree Apex-vs-Boundary.
3. **The CLI's presolve guard removed.** Green, because the guard's *stated*
   purpose (protecting the pin index space) turned out not to be what it
   does — see below.

The rule they instance is CLAUDE.md's: a corpus uniform in the dimension a
change acts on reports nothing however sharp its assertions are. Before
trusting a green suite on a threshold or a normalizer, ask what value the
fixture gives the quantity being divided by.

## The CLI presolve guard is an accuracy fix, not an index-space fix

Worth recording because the tidier explanation is wrong and would have been
believed.

A CLI run that serves a sensitivity request switches the convex presolve off.
The obvious reason — presolve's row space would invalidate the pins — is
**false**: `run_convex_qp` postsolves back to the extracted-QP space before
anything downstream runs, so the pin indices stay valid, and with presolve
left on the step is still within `1e-6` of the NLP path's.

What presolve actually costs, on the one fixture that exercises it: it fixes
the parameter the pin parametrizes and drops its row (`3 → 2 vars,
1 → 0 rows`), so the sensitivity reads a postsolve reconstruction rather than
the converged KKT. The step lands `5.0e-11` from the analytic answer instead
of `6.2e-15`.

**Unmeasured, and the reason the guard stays:** whether a reconstructed bound
multiplier can move the *active set* the sensitivity infers. That would be a
wrong derivative rather than a less accurate one, and answering it needs a
model whose active set is nontrivial — which the corpus does not have.

## What the corpus cannot tell you

The routing change moves **exactly one** fixture (`convex_qp_sens`,
`nlp → cvx-qp`, both legs). Three fixtures carry the sIPOPT suffixes, but
carrying them is necessary and not sufficient: `parametric.nl` and
`parametric_red_hessian.nl` both classify NLP, so `auto` never routed them
to the convex path.

That leaves **one** model exercising the change and none of any size. The
sweep shows containment; it says nothing about magnitude. A large LP with
sensitivity suffixes is the measurement nobody has made.

Re-measured after merging `origin/main`, which had landed
`8c38e55` (`bound_relax_factor` is opt-in on the convex arm) and `837e3c2`
(*"the reroute must solve the declared model, **and the sweep must see it**"*)
— both of which move what the sweep reports, and one of which says so in its
subject. The corpus had also grown from 79 fixtures to 93. Same result:
still three fixtures carrying the sIPOPT suffixes, still one classifying
convex, still **exactly one line moving**. Main's reroute decides what
happens when the convex solve *declines*; this gate decides whether the model
reaches the convex path at all, and the two do not interact on any fixture.

## Still open

- `sens_jacobian` / `Jacobian` live only in the Pyomo layer, so a bare-`.nl`
  caller has no Jacobian API. Moving them follows `2995a4c`'s pattern and is
  a real piece of work, not a move: they take Pyomo component data objects
  throughout.
- `python/pounce/qp.py` has no session/analysis layer, so the convex arm's
  new capability is Rust-only from Python.
- The CLI's conic route does not serve sensitivity: `build_conic` can answer
  for every family, but `extract_socp_with_map` has its own provenance map
  and mapping pins through it is unwritten.
