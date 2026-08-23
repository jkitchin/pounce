# Scoping a TNLP-based convex classifier

**Verdict: this needs its own design pass and its own PR. It is not a
relocation, and it must not ride along with one.**

Context: `crates/pounce-cli/src/qp_extract.rs` (2169 lines) and
`python/pounce/_route.py` (617 lines) both implement classify-and-extract —
decide whether a model is an LP / convex QP / convex QCQP, and if so marshal it
into `QpProblem` / `ConeSpec`. The Rust library path has neither, which is why
`Application::optimize_tnlp` refuses an explicit `solver_selection=lp-ipm` /
`qp-ipm` / `socp` and why the `qp_*` options raise "this entry point cannot
route a model to it".

## Why it is not tractable in a relocation pass

Everything else in this audit moved code between crates with the compiler
checking the result: imports rebased, behaviour byte-identical, tests
travelling with their files. This is the opposite kind of change — new
numerical code deciding which *engine* solves a model.

Per `CLAUDE.md`, a change that reroutes which correction the solver reaches for
is a trajectory change needing `scripts/sweep-fixtures.sh` on both legs before
merge. Changing which *solver* a model reaches is strictly larger than that.
Bundling it with pure moves would put the moves and the routing behind one
review, and the sweep diff would no longer have a quiet baseline to be read
against.

The failure modes are also asymmetric, and `_route.py` already says so in its
module docstring:

> a convex LP/QP/QCQP routed to the NLP solver is merely *slower* — the
> filter-IPM solves it correctly; a genuinely nonlinear or nonconvex problem
> routed to a convex solver returns a **silently wrong** answer.

A wrong answer that reports `Solve_Succeeded` is the expensive kind of bug —
gh#414, gh#324, gh#273, gh#744 and gh#745 are all that shape.

## The design the existing code already implies

### Why the CLI classifier does not port

`classify_problem` (`dispatch.rs:326`) and `extract_qp` work by walking the
`.nl` symbolic expression tree — `prob.obj_nonlinear.analyze_quadratic_full()`.
Its routing is therefore *certain*. A `TNLP` has no expression tree; it exposes
callbacks. So this is not a move of `qp_extract.rs` into a library crate. The
symbolic path cannot follow the TNLP, and none of those 2169 lines port.

`_route.py` is the real prior art: it faces the same opacity (Python callables)
and answers it by probing, fitting a linear/quadratic model, and validating
against held-out points before trusting the fit. A Rust TNLP classifier is
substantially a port of `_route.py`, not of `qp_extract.rs`.

### What a TNLP gives that a Python callable does not

The port is not a straight translation, because `TNLP` carries *declared*
structure that opaque callables lack. Three tiers of evidence, in decreasing
certainty:

1. **Declared — free and certain.** `get_constraints_linearity`,
   `get_objective_variables_linearity`, `get_variables_linearity`,
   `get_number_of_nonlinear_variables` / `get_list_of_nonlinear_variables`, and
   `derivative_proofs().hessian == Constant` (gh#588 Q6). A TNLP that declares
   all-linear constraints and a constant Lagrangian Hessian *is* a QP by its
   own contract — no probing, and no held-out validation needed, because the
   answer came from the model rather than from a fit. This tier does not exist
   on the Python path and should be tried first.
2. **Probed and validated** — `_route.py`'s algorithm, for TNLPs that declare
   nothing. `eval_h` gives an exact Hessian, and `eval_jac_g` gives one *with
   sparsity*, so the probe costs O(nnz) rather than `_route.py`'s O(n²)
   central differences. Convexity still needs a PSD test on the fitted `P`,
   and the held-out validation still gates the dangerous direction.
3. **Refuse** — fall through to the NLP path, which is the safe direction.

One trap: `eval_h` has a default implementation and is legitimately absent
under `hessian_approximation=limited-memory`, which per `CLAUDE.md` the Python
frontend and the CasADi plugin *select automatically* whenever no exact
Lagrangian Hessian is available. Tier 2 must therefore keep a
finite-difference fallback — i.e. exactly `_route.py`'s path — rather than
assuming `eval_h` is there. A classifier tested only against TNLPs with exact
Hessians would repeat the corpus gap that shipped gh#677.

### Where it goes

Neither obvious crate can see both types: `pounce-convex` does not depend on
`pounce-nlp`, and `pounce-algorithm` does not depend on `pounce-convex`.

It does **not** belong inside `Application::optimize_tnlp`. That would make the
NLP solver depend on the convex solver, and it misreads why `qp_extract.rs`
lives in `pounce-cli`: routing is a *frontend* concern, and the CLI is the
frontend that owns it. The library's equivalent of that frontend is
`pounce-rs`.

`pounce-rs` with the existing `convex` feature already depends on both
`pounce-nlp` and `pounce-convex`. So the classifier lands in
`crates/pounce-rs/`, feature-gated behind `convex`, with **no new crate, no new
dependency edge, and no cycle**. The lean default build is unaffected.

The refusal in `application.rs` then stays correct and stays put: that entry
point genuinely cannot route, and the routing frontend one level up is what
gains the ability.

## Scope sketch

1. Port `_route.py`'s probe/fit/validate core to Rust against `TNLP`, with the
   tier-1 declared-structure short circuit in front of it.
2. Extract to `QpProblem` (LP/QP) and `ConeSpec` (convex QCQP), reusing
   `pounce-convex`'s existing types. Return `Option`, `None` meaning "let the
   NLP path have it" — the same shape `extract_qp` and `_route.py` already use.
3. Dual recovery: map the convex solver's `(y, z)` back to TNLP-ordered
   multipliers. `qp_extract.rs`'s `recover_duals` / `recover_bound_mults` are
   the reference for the sign and row-mapping conventions even though the
   extraction does not port.
4. Tests: the three-way agreement that matters is NLP path vs. convex path vs.
   known answer, on the same model. Include TNLPs *without* `eval_h`. Mine
   gh#414 / gh#324 / gh#273 / gh#744 / gh#745 for the nonconvex-and-nearly-QP
   cases that must classify as `None`.
5. Only then consider honouring `solver_selection=lp-ipm|qp-ipm|socp` on a
   library entry point, and relaxing the refusal.
6. Fixture sweep, both legs, before merge.

## Open question for the owner

`_route.py` and a Rust classifier would be two independent implementations of
one conservative decision, free to disagree — a `minimize()` call and the
equivalent Rust TNLP could route differently on the same model. Whether that is
acceptable (they are separate frontends) or whether the Python path should
eventually delegate to the Rust one through `pounce._pounce` is a design
decision worth making deliberately, before there are two divergent
classifiers to reconcile.

No tracking issue exists for this gap. gh#561 covered *exposing*
`pounce-convex` through the facade (closed by PR #564, feature-gated modules)
but not classifying or routing a TNLP into it. The stale gh#604 pointer that
used to sit on the refusal message was about cold-start initialization options
and never covered this.
