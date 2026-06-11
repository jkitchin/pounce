# pounce for education & research — the introspectable, LLM-explainable solver

> Discussion note. The claim: pounce's interactive debugger + LLM/MCP
> integration is a capability **no other optimization solver has**, and it is
> uniquely valuable for *teaching* and *research*. This note grounds that claim
> in the shipping surface and maps the value for both audiences.

## The shipping surface this rests on

From `pounce-studio` (CLI skill + MCP server) and the `--debug` solver mode:

- **Live debugger** — Ctrl-C breaks into a running solve at the next iteration;
  inspect the iterate (primals, duals, KKT residuals, μ, inertia); `sweep` a
  variable, `multistart` from jittered points, `load` a saved iterate and step
  forward.
- **`explain`** — a glossary of every per-iteration column (`inf_pr`, `inf_du`,
  `mu`, `alpha`, inertia, …) *and* the `diagnose` finding codes. The trace is
  self-documenting.
- **`citations`** — curated paper references keyed by subsystem / bib key, so
  observed behavior links straight to the literature.
- **`diagnose`** — Ipopt-failure heuristics with severity-tagged findings.
- **`convergence_trace` / `find_stalls` / `restoration_windows` / `get_iterate`**
  — the trajectory as queryable, structured data.
- **`verify`** — signed, content-addressed solve receipts.
- All of it **driven conversationally over MCP** by any LLM client.

## Why "no other solver has this" — the unoccupied quadrant

Two axes: **introspectable internals** × **LLM-grounded explanation**.

- **Ipopt / SNOPT / KNITRO** — print a log wall; no live debugger, no LLM,
  internals behind a C/Fortran ABI.
- **Gurobi / BARON / commercial** — black box by design, licensed, no internal
  introspection.
- **CVXPY / JuMP / Pyomo** — modeling layers; the solver underneath is opaque.
- **Toy teaching solvers** — introspectable but *not faithful* to a production
  algorithm, so nothing transfers.

pounce occupies the empty intersection: a **faithful production algorithm** (the
Ipopt port — same logs and option semantics, so skills transfer to the tool people
actually use) that is **fully introspectable** and **explained by an LLM grounded
in the real trace and the literature.** Nothing else lives there.

## Education value

- **E1 — Glass-box pedagogy.** Students watch the IPM *actually run* — μ shrinking,
  inertia corrections firing, the filter accepting/rejecting steps, restoration
  kicking in — instead of a black box returning `x*`. `explain` makes every column
  self-documenting; the trace *is* the textbook.
- **E2 — A TA that watches your solve.** LLM + MCP = a tutor that reads *your*
  trace, finds the stall window (`find_stalls`), explains it in algorithm terms,
  and cites the paper (`citations`). Socratic mode: "`inf_du` is rising while
  `inf_pr` falls — what does that say about dual feasibility?" Scales to every
  student, every solve, any hour.
- **E3 — Zero-setup classroom.** `pip install`, pure Rust, **no HSL, no licenses,
  no GAMS.** Identical on every student laptop and in CI. Removes the single
  biggest practical barrier to teaching real optimization.
- **E4 — Grade the process, not just the answer.** Signed `verify` receipts +
  solve-report JSON as artifacts → assignments where the student submits a
  *trace*, and autograding inspects *how* they got there (warm-start? why 200
  iters?). The solve becomes a gradeable, reproducible object.
- **E5 — Curriculum-as-code.** Builtin problems (HS suite, Rosenbrock), GAMS
  examples, and the report schema → ready-made problem sets with known, documented
  behavior the LLM can reference.
- **E6 — Teaching differentiable optimization / SciML.** pounce.jax + the debugger
  → a course where students *inspect the KKT system being differentiated* (the
  implicit function theorem made concrete), bridging classical optimization and
  modern ML in one tool.

## Research value

- **R1 — The trace as a dataset.** `convergence_trace` + the `.iterdump` binary
  format across whole suites = a reproducible corpus for studying restoration
  triggers, stall morphology, μ-strategy behavior. Pure-Rust determinism means
  results replicate exactly.
- **R2 — A hackable, faithful baseline.** Researchers fork the *readable Rust*
  algorithm — swap a barrier update, a filter rule, a step-acceptance test — and
  A/B it against the faithful-Ipopt baseline in one codebase, not Fortran behind an
  ABI. The faithfulness is the experimental control.
- **R3 — LLM-as-experimentalist.** An agent drives the MCP surface to run studies:
  "run these 5 μ-update strategies over the Mittelmann set, cluster failures by
  `diagnose` code, summarize which converged faster and hypothesize why." The
  solver becomes scriptable by an agent that also does the literature-grounded
  write-up.
- **R4 — A failure-mode taxonomy.** `diagnose` + `find_stalls` +
  `restoration_windows` systematized across suites → a catalog of *where and why*
  IPMs fail, as a publishable research artifact.
- **R5 — One lens across the whole family.** NLP, conic, global, and (via discopt)
  MINLP share the report/debug surface → study B&B node behavior, conic centrality,
  and global bounding *with the same instrument* — cross-solver-class research
  that's normally impossible because each solver has its own opaque format.

## What to lead with

```
HEADLINE (unique, defensible, demonstrable today):
  ★ "The first optimization solver you can debug interactively and ask an LLM to
     explain — grounded in the real trace and the literature."   = E1 + E2

HIGH-LEVERAGE EDUCATION:
  ✓ E3 zero-setup classroom   (removes the #1 adoption barrier; true now)
  ✓ E4 grade-the-process      (signed receipts already exist)

HIGH-LEVERAGE RESEARCH:
  ★ R2 hackable faithful baseline + R1 trace-as-dataset   (pure-Rust determinism enables it)
  ○ R3 LLM-as-experimentalist (the agent differentiator, longer horizon)
```

## The through-line

Every other solver treats the solve as a *transaction*: submit, wait, read the
answer. pounce treats it as an **observable, explainable, reproducible process** —
and the LLM/MCP layer turns that observability into *conversation*. For education
that's a tutor that scales; for research that's an instrument with a faithful
baseline and a deterministic trace. It is the "legible to agents" pillar pointed
at the two audiences where legibility *is* the value.

## Publishable angle

This is itself a paper: *"An LLM-drivable interactive debugger for interior-point
methods as a pedagogical and research instrument."* JOSS (software) or an
education-track venue; the Zenodo DOI + CITATION.cff infrastructure is already in
the README. R4 (failure-mode taxonomy) is a second, more methods-flavored paper.
