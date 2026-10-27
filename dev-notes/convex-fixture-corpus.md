# The convex half of the CLI fixture corpus

`scripts/sweep-fixtures.sh` is the tool CLAUDE.md requires before a trajectory
change merges, and it is only as good as the models it sweeps. This note is
about one population inside it: the fixtures that reach the **convex** driver
(`pounce-convex` — the HSDE loop and the direct driver behind it), as opposed
to the general filter-IPM.

## The gap, and how it was found

gh#690 measured an adaptive-τ tail for the HSDE corrector three times over
four months and declined it on the third pass. The number was not the problem:
−4.2% iterations on the exact leg, −1.7% on the L-BFGS leg, no status changes,
no objective regressions. The population was.

Every fixture that moved under the variant:

| moved | vars |
|---|---|
| `lp_afiro` | 32 |
| `scaled_feasible_a` / `_b` | 4 |
| `rankdef_eq_qp` | 3 |
| the other 13 | 1–2 |

The corpus's substantial models — `deb7` (813 vars), `eigena2`/`eigenb2` (110),
`autocorr_bern55-06` (56), `pooling_rt2stp` (46) — are all NLP class, and the
convex driver never sees them. `airport` (84 vars) was the only convex fixture
above 32 variables and did not move at all. Net of one pathological ill-scaled
stress fixture, the corpus-wide saving was **90 iterations across 59 problems,
almost entirely `8 → 5` on models of one to three variables.**

That is arithmetic over a population that cannot answer the question. #690's
closing note put the prerequisite plainly:

> The prerequisite is not more τ variants — it is convex fixtures large enough
> to measure. […] Until the convex path is exercised by something bigger than
> 32 variables, an HSDE step-rule change cannot be evaluated here at all, and
> that gap is worth more than this rule was.

The cause is worth naming, because it is not carelessness. Every convex fixture
in the corpus was added as a **routing or verdict witness** — to prove which
engine ran, or what it reported, or that a presolve line appeared exactly once.
Two variables do that perfectly well, and a bigger model would only have made
the test slower. Nothing was ever added as a **trajectory witness**, and the
sweep silently inherited the difference.

## What was added (gh#724 branch)

Four fixtures, chosen for distinct pathologies rather than for size alone:

| fixture | source | n | m | iters | qp_tau=0.99999 |
|---|---|---|---|---|---|
| `lp_degen2` | NETLIB `degen2` | 534 | 444 | 15 | 11 |
| `lp_share1b` | NETLIB `share1b` | 225 | 117 | 32 | 25 |
| `lp_israel` | NETLIB `israel` | 142 | 174 | 29 | 23 |
| `convex_qp_share1b` | Maros–Mészáros `QSHARE1B` | 225 | 117 | 28 | 22 |

* **`degen2`** — massively primal-degenerate, the gh#535 / gh#133 population:
  strict complementarity fails and a pure IPM struggles to certify the vertex.
  The largest of the four at 534 columns.
* **`share1b`** — the classic ill-conditioned NETLIB instance, and the longest
  trajectory of the four, so the most sensitive to how far a step may go.
* **`israel`** — dense columns, which stress the KKT factorization and its
  ordering rather than the barrier trajectory.
* **`QSHARE1B`** — `share1b` with a quadratic objective bolted on. Same
  sparsity, same bounds, non-zero `P`: it exercises the convex-QP branch of the
  same driver against an LP control differing in one term.

All four solve in under 0.11 s each, so both sweep legs stay cheap; together
they add ~115 KB to the repository.

The `qp_tau` column is the point. `qp_tau` is the fraction-to-boundary
parameter — the one knob that changes how far a step may go without changing
the model, the convergence test, or the engine — and it stands in here for "a
step-rule change". Before these four, the convex corpus answered that
perturbation with `8 → 5` on toy models. It now also answers with `32 → 25` on
an ill-conditioned 225-column LP whose optimum is published, and with `15 → 11`
on a 534-column degenerate one.

`lp_afiro` stays exactly where it is; it is the gh#535 and gh#588 witness and
several tests name it directly. These join it.

## Provenance

Nothing here is hand-built. All four are regenerated from cached upstream data
by benchmark harnesses already in the tree:

```sh
# NETLIB LP (netlib.org/lp/data, expanded with netlib's own `emps`)
cd benchmarks/lp && python3 generate_nl.py --netlib-only degen2 share1b israel
cp nl/degen2.nl   ../../crates/pounce-cli/tests/fixtures/lp_degen2.nl
cp nl/share1b.nl  ../../crates/pounce-cli/tests/fixtures/lp_share1b.nl
cp nl/israel.nl   ../../crates/pounce-cli/tests/fixtures/lp_israel.nl

# Maros–Mészáros convex QP (qpsolvers/maros_meszaros_qpbenchmark mirror)
cd benchmarks/qp && python3 generate_nl.py QSHARE1B
cp nl/QSHARE1B.nl ../../crates/pounce-cli/tests/fixtures/convex_qp_share1b.nl
```

Each carries a published optimum from its source collection, which is what
makes it a usable sentinel: a trajectory change that moves the answer can be
checked against a number nobody in this repository chose.

| fixture | published optimum | pounce (convex path) |
|---|---|---|
| `lp_degen2` | −1435.1780000 | −1435.17800002 |
| `lp_share1b` | −76589.318579 | −76589.31857895 |
| `lp_israel` | −896644.82186 | −896644.82186290 |
| `convex_qp_share1b` | 720078.3182 | 720078.31815401 |

`crates/pounce-cli/tests/issue_690_convex_corpus_scale.rs` pins all of it:
the published optimum, the convex routing, the dimensions (so a fixture cannot
be swapped for a smaller model of the same name), and the fact that each one
still *responds* to a step-rule perturbation. It deliberately asserts no
absolute iteration count — those are the most platform-sensitive numbers in the
repository. Measuring is the sweep's job.

## The second gap, one dimension over (gh#760)

Size was not the whole of it. Two months after the four fixtures above landed,
`4c02817d` applied `bound_relax_factor` on the convex arm — a correctness fix
— and cost **+515 iterations across the 138-problem Maros–Mészáros QP suite**,
4.4× on the QSCFXM family. The corpus, by then carrying a 534-column
degenerate LP, still did not predict it.

Not because the models were small. Because **not one of them was a model on
which relaxing the box costs anything.** Swept twice on `fdea82b5`, default
versus `bound_relax_factor=0`, every convex line that moves at all:

| fixture | relaxed | `bound_relax_factor=0` |
|---|---|---|
| `lp_degen2` (534 cols) | 18 | 15 |
| `feasible_x0_sentinel_bound` | 27 | 25 |
| `feasible_x0_extreme_row` | 32 | 33 |
| `scaled_feasible_b` | 44 | 47 |
| `qcqp_ball` | 12 | 17 |
| `lp_afiro`, `lp_israel`, `lp_share1b`, `convex_qp_share1b` | unmoved | unmoved |

Tens of percent, in both directions, plus two ill-scaled stress fixtures
(`scaled_feasible_a` 69 → 199, `feasible_x0_wide_scale` 32 → 80) whose numbers
are about their scaling. A reviewer reading that diff learns "small and mixed",
which is true of the corpus and false of the suite.

The four above were chosen for *pathology* — degeneracy, ill-conditioning,
dense columns, a quadratic term. None of those is the dimension a bound
relaxation acts on, which is how much slack the box has where the multipliers
are large. That is the same failure as gh#690's, one dimension over: a corpus
uniform in the dimension a change acts on reports "no effect" no matter what
else it varies.

| fixture | source | n | m | iters | `bound_relax_factor=0` |
|---|---|---|---|---|---|
| `convex_qp_qscfxm1` | Maros–Mészáros `QSCFXM1` | 457 | 330 | 131 | 30 |

`QSCFXM1` is the cheapest member of the family the benchmark measured — 457
columns and 0.40 s per leg, against `QSCFXM3`'s 1371 columns and 1.7 s — and
carries the same 4.4× signature. Published optimum `1.68826917D+07` (DOC 97/6,
BPMPD); POUNCE returns 1.6882691485e+07, 1.3e-8 relative. Provenance is the
same harness as the others:

```sh
cd benchmarks/qp && python3 generate_nl.py QSCFXM1
cp nl/QSCFXM1.nl ../../crates/pounce-cli/tests/fixtures/convex_qp_qscfxm1.nl
```

The sweep's wall time goes 10.9 s → 11.9 s for it (three runs each way, spread
under 0.3 s). `crates/pounce-cli/tests/issue_760_convex_bound_relax_magnitude.rs`
pins the routing, the dimensions, the published optimum and the *ratio* — no
absolute count, same reason as the gh#690 file. Mutation-checked: set the
convex arm's relax factor back to zero and two of its three tests go red at
1.00×.

Full record of the measurement it exists for:
`dev-notes/qp-bound-relax-iteration-cost.md`.

## What this does not fix

The models gh#535 was actually filed on — NETLIB/Mészáros `gen`/`gen1`, where
the convex path burns its whole budget in ~191 s and the NLP path finishes in
19 iterations — are still not fixtures, and should not be: `gen.nl` is n=2560 /
m=769 / 63085 Jacobian nonzeros and takes three minutes to *fail*. The four
above are big enough to move under a step rule; they are not big enough to
reach the failure modes that motivated the LP→NLP reroute. Reaching those in
the corpus at a realistic tolerance remains open, and gh#724 documents the
sweep (all fixtures × `qp_tau ∈ {0.99, 0.999, 0.9999, 0.99999}`) that produced
no `NumericalFailure` on the corpus as it stood.

## Related

* gh#690 — the τ study, closed measured-and-declined; its closing comment is
  the source of the table above.
* gh#724 — the LP→NLP reroute gate omitted `NumericalFailure`; found by the
  same study, fixed alongside this corpus work.
* gh#760 — the same population failure one dimension over, and the
  `convex_qp_qscfxm1` fixture that closes it:
  `dev-notes/qp-bound-relax-iteration-cost.md`.
* `dev-notes/trajectory-regressions-and-the-fixture-sweep.md` — why the sweep
  exists at all (gh#544, gh#592).
* gh#1076 — `convex_qp_qgfrdxpn` (Maros–Mészáros `QGFRDXPN`, 1092 columns,
  616 rows, copied unmodified from the benchmark's `nl/QGFRDXPN.nl`). It is
  here for the active-set arm, not the IPM: under the default
  `solver_selection=auto` both sweep legs route it to `cvx-qp` and it does
  not move. Under `solver_selection=qp-active-set` it is the witness for the
  counting prune's per-pass cap in `pounce-qp`
  (`crates/pounce-cli/tests/issue_1076_qgfrdxpn_active_set_prune_cap.rs`).
  `convex_qp_qforplan` (Maros–Mészáros `QFORPLAN`, 421 columns, 161 rows,
  copied the same way) bounds the cap from the other side in the same file.
  It takes 40–50 s per solve under `qp-active-set`, so it adds that much to
  the `qp-active-set` sweep leg.
