> ## SUPERSEDED — historical design note
>
> This was the original brainstorm. It is kept for the design rationale (why a
> separate repo, why witnesses need not be trusted, why the tiers are what they
> are), all of which survived contact with the implementation. **Do not use it
> as a specification.** The implemented contract is
> [`docs/src/schema/lean-cert-v1.md`](../docs/src/schema/lean-cert-v1.md);
> the emitter is `crates/pounce-lean-cert/`.
>
> Known errors below, left in place rather than quietly rewritten:
>
> * The schema tag is **`pounce.lean-cert/v1`**, not `pounce-cert/v1` (used in
>   four places here, including the section titled "The contract").
> * The binding-fields list names `statement_sha256` as part of the certificate;
>   a later section of this same document correctly retracts that. It is not in
>   the cert — it belongs to the post-codegen receipt.
> * "Unforgeable two ways: SHA-256 content-addressing and an optional HMAC" is
>   **wrong**. A content hash is a *binding*, not authentication — anyone can
>   compute a correct hash over any bytes. `docs/src/verify.md` attributes
>   unforgeability to HMAC alone, and otherwise to recomputation.
> * The emitter cannot "reuse the SOS plumbing in `sos.py`" — it is Rust and
>   that is Python. The Rust SOS engine is `crates/pounce-convex/src/sos.rs`,
>   which this note never mentions, and it does not currently expose the Gram
>   matrices a certificate would need.
> * POUNCE emits `.sol` only; there is no `.nl` writer in the workspace.
> * The diagram gives `lake build` exit code 20. That is `pounce verify`'s
>   infeasible code; `lake build` has no such convention.
> * The problem encoding shipped as matrix/vector form, not expression trees.
>   Trees are deferred to a future `nlp-poly` slice.

# Lean-verified solution certificates

**Status: design note / brainstorm.** Nothing here is implemented yet. This
captures the architecture for emitting a certificate from a POUNCE solve that
the [Lean 4](https://lean-lang.org/) theorem prover can independently verify —
proving, with a kernel-checked proof, that a returned `x*` is **feasible** and
(for the tractable problem classes) **a minimum**, using exact rational
arithmetic so there is no floating-point trust gap.

It builds directly on the existing `pounce verify` trust model
(`crates/pounce-cli/src/verify.rs`, `docs/src/verify.md`). Read that first: it
is the keyless, content-addressed feasibility checker this extends.

## Why Lean, on top of `pounce verify`

`pounce verify` today re-evaluates `g(x*)` in **f64**, checks
`g_l ≤ g(x*) ≤ g_u` against the canonical `.nl`, and makes the receipt
unforgeable two ways: SHA-256 **content-addressing** (a receipt is meaningful
only for the exact `.nl`/`.sol` bytes it names) and an optional **HMAC** so a
keyholder can attest a receipt. `verify.md` is candid about two standing
non-goals:

> *"Feasibility is fully checkable; global optimality is not. The stationarity
> residual certifies a first-order/KKT point, not a global minimum."*

and the float-tolerance fuzz of `--feas-tol`. Lean attacks both.

It also changes the **nature of the trust anchor**. HMAC's guarantee — "a party
without the key cannot mint a receipt" — is conditional on key secrecy, and
`verify.md` admits an agent sharing the host defeats it. **A Lean proof has no
key.** Its unforgeability is intrinsic: a proof term either typechecks against
the kernel or it does not, and nobody can fabricate one that checks. That is
strictly stronger than HMAC and removes the entire "key isolation" chapter. The
SHA-256 hash does not go away — it does a *different* job (binding the proof to
the canonical problem; see [Trust boundaries](#trust-boundaries)).

## The lossless float→rational fact

The enabling observation: **every f64 is exactly a dyadic rational** `m·2^e`.
Converting `x*` and `λ` from the `.sol` into Lean `ℚ` is therefore **lossless
and canonical** — no rounding, no ambiguity, no float in the trusted path. The
SHA-256 POUNCE already computes over the `.sol` bytes commits to exactly those
rationals. Lean then reasons over ℚ exactly.

So the "rational approximation to mitigate float issues" is **not** an
approximation of `x*` — `x*` is represented exactly. The only approximation is
that `x*` ≠ the *true* optimum, handled explicitly per claim tier below.

## What "is a minimum" can mean — three claim tiers

Be precise about the verdict; overselling here would be the worst outcome. The
certificate names exactly one proven claim:

| Verdict | Means | Tractable for |
|---|---|---|
| `feasible` | `x*` satisfies all constraints/bounds (within a declared ε) | any algebraic (polynomial/rational) model |
| `local-min-strict` | KKT + second-order sufficient ⟹ strict local minimizer | smooth algebraic NLP |
| `global-min` | certified global minimizer | convex (LP/QP/convex NLP) **or** polynomial-via-SOS |

### Tier 1 — feasibility

Given exact-rational `x̃`, Lean proves `g_l ≤ g(x̃) ≤ g_u` and
`x_l ≤ x̃ ≤ x_u`.

* **Polynomial / rational-function constraints over ℚ:** closed by `norm_num` /
  `polyrith` / `ring` / `decide`. Fully exact.
* **The equality-constraint snag.** A rational `x̃` generally cannot satisfy a
  nonlinear *equality* exactly. Two honest treatments:
  * **(a) declared tolerance** — prove `|g(x̃) − rhs| ≤ ε` *exactly* over ℚ,
    with ε stated in the certificate. Shippable now, and still strictly better
    than f64 fuzz because the arithmetic is exact and the bound is a theorem.
  * **(b) interval-Newton / Kantorovich existence** — prove a *true* zero lives
    in a tiny box around `x̃`. The gold standard; a genuine Lean formalization
    effort. Deferred.
* **Transcendentals** (`exp`/`sin`/`log`): need verified interval bounds;
  Mathlib coverage is thin. Out of scope for v1; `dReal` (δ-complete nonlinear
  SMT) is a complementary checker for this fragment later.

### Tier 2 — strict local minimum (smooth algebraic NLP)

Second-order *sufficient* conditions, all Lean-checkable over ℚ:

* KKT stationarity `∇f + Jᵀλ + (bound multipliers) = 0` — exact, given `λ` as
  rationals.
* Dual feasibility (sign of `λ`) and complementarity — exact.
* **Reduced Hessian positive-definite** on the active-constraint null space.
  The Lean-friendly PSD certificate: POUNCE emits a rational `LDLᵀ`
  factorization of the reduced Lagrangian Hessian; Lean checks the matrix
  identity `M = L D Lᵀ` by `norm_num`/`ring` and `Dᵢ > 0` entrywise. Certifies
  a **strict local** minimizer. This is the honest ceiling for nonconvex NLP.

### Tier 3 — global minimum (two routes only)

* **Convex (LP/QP/convex NLP):** KKT ⟹ global. Global then reduces to
  *certifying convexity* in Lean — for a QP, prove the (constant) Hessian PSD
  once via the same `LDLᵀ` trick, plus the local KKT certificate. For an LP it
  collapses to an exact **Farkas/dual certificate** Lean checks by arithmetic.
* **Polynomial nonconvex → SOS duality.** POUNCE already has `python/pounce/sos.py`
  and an SOS global-optimization notebook. An SOS certificate writes
  `f(x) − γ = σ₀(x) + Σ λᵢ(x)·gᵢ(x)` with `σ` sums-of-squares. Lean verifies
  exactly the two things it is good at: a **polynomial identity** (`ring`/`norm_num`
  over ℚ coefficients) plus **PSD of the Gram matrices** (rational `LDLᵀ`). That
  certifies a global *lower bound* `γ`; pairing it with the feasible point that
  achieves `γ` yields **certified global optimality**, kernel-checked, no float.

## Repository topology

The pipeline has three pieces with three natural homes. **The Lean library
lives in a separate repo; the emitter stays in POUNCE; a versioned schema is the
contract between them.**

```
 .nl (canonical, hashed)
        │
   ┌────┴──────────────────────────┐  POUNCE's job: produce DATA, never a proof
   │ POUNCE (Rust, THIS repo)        │
   │  • lossless f64→ℚ of x*, λ       │
   │  • problem as ℚ expr-trees       │──▶ problem cert  ("the statement")
   │  • witnesses: LDLᵀ, SOS Gram     │──▶ witness data   ("the proof hints")
   └─────────────────────────────────┘
        │  versioned schema = wire contract (pounce-cert/v1)
        ▼
   ┌─────────────────────────────────┐  Lean's job: data → KERNEL-CHECKED proof
   │ pounce-lean (Lean4 + Mathlib,    │
   │  SEPARATE repo)                  │
   │  • cert → .lean statement         │
   │  • reusable lemmas/tactics:       │──▶ lake build → verdict (exit 0 / 20)
   │    PSD-via-LDLᵀ, SOS identity,    │
   │    convex-KKT ⟹ global            │
   └─────────────────────────────────┘
```

### Why the Lean library wants its own repo

* **Toolchain blast radius.** Lean/Mathlib is a multi-GB, `elan`/`lake`-built,
  revision-pinned dependency with slow CI. CLAUDE.md shows POUNCE already
  juggling 3 registries, 19 crates, and a pre-tag consistency guard. Bolting a
  Mathlib build into that couples every POUNCE PR and release tag to Mathlib's
  cadence.
* **Independent versioning.** The Lean lib versions against *Mathlib revs*, not
  POUNCE `X.Y.Z`. The only thing that must agree across the seam is the **cert
  schema version**.
* **Different contributor pool.** Lean+Mathlib formalization is a distinct
  community; a Rust-optimizer monorepo is a barrier to them and vice-versa.
* **Optional high-assurance lane.** Almost all users `pip install pounce-solver`
  and never touch Lean. The core repo should not carry that weight.

### Why the emitter stays in POUNCE

The converter from a solve to a certificate reuses `pounce-nl`'s `.nl` reader,
the `.sol` parser and `sha256` module in `verify.rs`, and the SOS plumbing in
`sos.py`. Reimplementing `.nl` parsing on the Lean side would create a *second*
TCB and duplicate the best reader. So: **POUNCE emits, pounce-lean verifies** —
mirroring how POUNCE already emits `.nl`/`.sol` as the contract to external
tools. The Lean certificate is one more emitted artifact format.

### Not a git submodule

Do **not** vendor pounce-lean as a submodule of POUNCE. Submodules worsen the
toolchain coupling; the schema-contract decoupling is what keeps the slow
Mathlib repo off POUNCE's critical path.

## Trust boundaries

The property that makes the repo seam *safe*: **the witnesses do not need to be
trusted.** If POUNCE emits a wrong `λ`, a bogus `LDLᵀ`, or a bad SOS Gram
matrix, the Lean proof simply **fails to typecheck** — bad witness data cannot
produce a passing proof. POUNCE can be fully adversarial and forge nothing. The
Rust→Lean boundary therefore carries only *untrusted hints + a statement*,
exactly the kind of boundary a repo seam can sit on.

The "is it even the right problem" gap closes the same way `verify.md` closes it
— *recompute, don't trust a receipt*. The consumer's acceptance test:

> accept **iff** `lake build` succeeds **∧** the proof's `nl_sha256` literal
> equals SHA-256 of *the consumer's own canonical* `.nl` **∧** `statement_sha256`
> equals the hash of the statement re-derived from that `.nl`.

So the trusted base shrinks to **{Lean kernel} + {the deterministic
nl→cert emitter}**, and the emitter is content-addressed so a suspicious
consumer re-runs it and matches the hash. **No key anywhere** — a smaller,
keyless TCB than today's HMAC + key-isolation story.

The Lean theorem statement **embeds `nl_sha256` and `sol_sha256` as literals**
(in module/def names or a documented header) so the artifact provably concerns
those exact bytes. The danger is never a forged proof; it is a proof of the
*wrong, easier theorem* — which the `statement_sha256` re-derivation catches.

## The contract: `pounce-cert/v1`

A versioned certificate schema is the linchpin that lets the two repos evolve
independently. Precedent already exists: `docs/src/schema/solve-report-v1.md`
(schema tag `pounce.solve-report/v1`). Mint `pounce-cert/v1` the same way and
keep the **schema doc in this repo** (it is the producer's contract). It pins:

* **exact-rational representation** — dyadic `m·2^e` (lossless from f64), or
  general `p/q` integers in ℚ;
* **problem encoding** — objective and constraints as expression trees over ℚ,
  plus bounds `x_l,x_u,g_l,g_u`;
* **witnesses per tier** — KKT duals `λ`; the reduced-Hessian `LDLᵀ` factors;
  SOS Gram matrices + multiplier polynomials;
* **binding fields** — `nl_sha256`, `sol_sha256`, `statement_sha256`, the
  claimed verdict ∈ `{feasible, local-min-strict, global-min}`, tolerance ε,
  and (for reproducibility) the intended Lean toolchain + Mathlib revision.

Versioning policy mirrors the solve-report schema: adding fields is
non-breaking; removing/renaming bumps the major; changing a field's semantics
without a rename is forbidden.

The concrete field-level schema is written up in
[`dev-notes/lean-cert-schema-v1.md`](lean-cert-schema-v1.md), and a
complete worked `qp-convex` / `global-min` certificate plus the `.lean` it
should generate lives in [`dev-notes/lean-cert-example/`](lean-cert-example/).
One refinement settled while writing those: `statement_sha256` (and any
signature) is **not** in the certificate POUNCE emits — it belongs to the
*verification receipt* produced after codegen, because the statement is derived
by `pounce-lean`, not by POUNCE. The certificate carries `nl_sha256` and
`sol_sha256` only.

**Drift guard** (mirrors `scripts/check-release-consistency.sh`): POUNCE CI
emits a golden `cert` fixture; pounce-lean CI checks committed golden fixtures
still verify. A schema break then fails *someone's* CI loudly instead of
silently rotting.

## Phasing

* **Phase 0 (this repo).** Define `pounce-cert/v1`; add a `pounce certify`
  path emitting the certificate (problem + witnesses) for the **convex-QP**
  slice. No Lean yet — just exact data + a golden fixture and the schema doc.
* **Phase 1 (new `pounce-lean` repo).** PSD-via-`LDLᵀ` lemma + convex-KKT⟹global
  theorem + `cert → .lean` codegen. End-to-end: QP → certified **global** min.
  The smallest thing that exercises the whole architecture; global result on day
  one; no SOS machinery; no equality-residual fuzz.
* **Phase 2.** SOS identity checker in pounce-lean + Gram-matrix witnesses from
  `sos.py` → certified global min for nonconvex polynomials.
* **Phase 3.** `local-min-strict` for general smooth algebraic NLP; later,
  transcendentals (where `dReal` may complement Mathlib's thin interval
  arithmetic).

## Recommended first slice

**Convex QP, `global-min`.** It exercises the full pipeline — lossless rational
`x*`, exact constraint evaluation, KKT, PSD Hessian ⟹ global — on the easiest
math, with no SOS and no equality-residual argument, and produces a genuinely
*global* certificate immediately.

## Resolved: emitter output form (was open question #2)

**Decision: POUNCE emits a neutral `cert.json`; `pounce-lean` owns the
`cert → .lean` codegen.** POUNCE never emits Lean source.

Rationale:

* **Keeps `.nl`→math translation in Rust**, where `pounce-nl`'s reader already
  lives — no second `.nl` parser, no duplicated TCB.
* **Keeps the Lean repo free of POUNCE-format knowledge** — it codes against a
  small, versioned, language-neutral schema, not against AMPL internals.
* **Makes the witness/statement split explicit**, which is what the whole trust
  argument rests on: the cert carries *data* (problem + untrusted witnesses);
  the Lean side derives the *statement* and the *proof*. Emitting `.lean`
  directly would blur that line and drag Lean syntax into POUNCE's release
  surface.
* **Decouples release cadence** — a Mathlib/Lean syntax change is a `pounce-lean`
  concern only; the cert schema (and POUNCE) are unaffected.

Schema: [`dev-notes/lean-cert-schema-v1.md`](lean-cert-schema-v1.md).
Worked example: [`dev-notes/lean-cert-example/`](lean-cert-example/).

## Open questions

1. **Equality constraints in tier 1:** ship declared-tolerance ε now, or invest
   in interval-Newton existence? Changes what "verified feasible" *means* to a
   consumer.
2. **Consumer ergonomics:** is requiring `lake build` acceptable, or do we also
   ship a prebuilt/attested verdict for consumers who will not run Lean? (The
   latter reintroduces a trust-the-attester problem the proof was meant to
   remove.)
3. **Naming:** `pounce-lean` vs `pounce-cert` for the verification repo.
