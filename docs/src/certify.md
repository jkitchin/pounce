# Certifying Solutions (Lean)

```sh
pounce certify <problem.nl> <claim.sol> [OPTIONS]
pounce cert-verify <problem.nl> <cert.json>
```

[`pounce verify`](verify.md) re-evaluates the constraints in f64 and answers
*"is this point feasible?"* — and is candid that **global optimality is not
checkable that way**. `pounce certify` answers a different question: it emits
an exact-rational certificate that an external Lean 4 development turns into a
**machine-checked proof**, verified by the Lean kernel over ℚ with no floating
point anywhere in the trusted path.

The two are complements, not alternatives. `verify` runs anywhere, on any
problem, in milliseconds. `certify` covers a narrow slice of problems and needs
a Lean toolchain, but what it produces is a proof rather than a check.

> **Status.** The producer half — everything in this repository — is complete
> and tested for the slices below. The consumer half (`pounce-lean`: codegen
> plus the reusable Lean lemmas) **is not yet public**, so today you can emit
> and bind certificates, but you cannot run the final `lake build` step
> yourself. Certificates are plain JSON and stable under the `pounce.lean-cert/v1`
> schema tag, so they remain useful — and checkable later — regardless.

## Why a certificate instead of a check

Verification is only as good as what it *can't* be talked out of. Three
properties do the work here:

**The witnesses are untrusted.** A certificate carries proof hints — duals, a
Gram matrix, an `LDLᵀ` factorization — but nothing about them is believed.
Wrong witness data makes the generated Lean fail to typecheck; it can never
make a false statement pass. POUNCE could be adversarial and forge nothing.
The worst failure mode is a certificate that does not verify.

**There is no key.** `pounce verify` can sign its receipt with an HMAC, which
is exactly as trustworthy as the secrecy of the key. A Lean proof has no such
dependency: the trust anchor is the kernel, which the consumer runs, using
*their* pinned Lean and Mathlib.

**Nothing is approximate.** Every optimality certificate carries
`tolerance = 0`. The f64 solve is treated as a *heuristic that proposes* — an
active set, a Gram matrix, a bound — and everything emitted is recomputed
exactly over ℚ. If the float guess was wrong, the emitter refuses rather than
shipping a certificate that cannot verify. (The one verdict with a nonzero
tolerance is [`feasible`](#certifying-the-float-itself), where ε is *the claim*
rather than slack in it — and even there ε is an exact rational the kernel
checks, not a comparison threshold.)

That last point is the one that surprises people, so it is worth being concrete.
On the `certify_infeasible` fixture the solver's Farkas ray has norm ≈ 2.3e10
and satisfies `Aᵀy = 0` to a relative residual of ~1.7e-11 — excellent for
floating point. Converted losslessly to ℚ, that residual is `−103801/262144`,
which is not zero, so Lean cannot discharge the hypothesis. Copying the
solver's numbers would produce a certificate that always fails. POUNCE instead
uses the float ray as a hint for the *support* of the exact one and solves for
`y = (1, 1, 1)`.

## What it can prove

Four verdicts, chosen automatically from the `.nl` — you do not select one:

| Verdict | Slice | The proven statement |
|---|---|---|
| `global-min` | convex QP | `x*` is feasible and no feasible point has a lower objective |
| `infeasible` | LP / QP | no feasible point exists (Farkas witness) |
| `unbounded` | LP or QP, any `Q` | the objective decreases without bound along a recession direction |
| `global-lower-bound` | polynomial, degree > 2 | `γ ≤ p(x)` for **every** real `x` — or, if the problem has constraints, for every *feasible* `x` |
| `global-min` | polynomial, degree > 2 | that bound is *attained* at an exhibited `x₀`, so `p(x₀) ≤ p(x)` (again, everywhere or on the feasible set) |

plus three you *do* select, with `--feasible` or `--local`:

| Verdict | Slice | The proven statement |
|---|---|---|
| `feasible` | any linearly-constrained QP, convex or not | `x*` violates no row by more than ε, **and** a genuinely feasible point exists within ε of it |
| `local-lower-bound` | polynomial, degree > 2, with `--local` | `γ ≤ p(x)` for every feasible `x` **within an exhibited ball** — and nothing outside it |
| `local-min` | polynomial, degree > 2, with `--local` | that bound is attained at an `x₀` proved to lie in the ball, so `p(x₀) ≤ p(x)` there |

The last two rows are the interesting ones, because they cover the case where a
solver's usual answer is worth very little. A nonconvex polynomial has many
basins; a local solve returns one and can say nothing about the others, and no
KKT argument fixes that. What *can* be established is a bound, via a
sum-of-squares identity — exhibit a monomial basis `m(x)`, a rational `γ`, and
a positive-semidefinite Gram matrix `G` with

```text
p(x) − γ = m(x)ᵀ G m(x)
```

The right-hand side is a nonnegative quadratic form for every `x`, so the
bound holds globally. Lean closes the identity with `ring` and discharges
`G ⪰ 0` from an exact `LDLᵀ` — two independent obligations, both required.

If some rational point *reaches* γ, the claim strengthens from a bound to a
minimum, and the extra obligation is a single equation: `p(x₀) = γ` over ℚ. For
`x⁴ − 2x² + 2` that point is `x = 1`, and `pounce certify` finds it by snapping
the local solve's iterate to a short ladder of rational grids and evaluating `p`
**exactly** — equality or nothing, never closeness.

Not every polynomial has one. `x⁴ − 3x² + 2` minimizes at `±√(3/2)`, which no
rational point reaches, so its certificate stops at `γ = −1/4` and says so. That
is the honest answer rather than a limitation to route around: a certificate
over ℚ cannot exhibit an irrational minimizer, and the bound it does prove still
holds for every real `x`.

### Constraints: the same idea, modulo the feasible set

`x³ − 3x` has no lower bound on ℝ at all, so no γ makes `p − γ` a sum of
squares and the identity above simply does not exist. On `0 ≤ x ≤ 3` it does —
not outright, but *modulo the constraints*. Write the feasible set as
`gₖ(x) ≥ 0` (POUNCE folds finite variable bounds into that list, so a box
becomes `x ≥ 0` and `3 − x ≥ 0`) and the certificate becomes a **Putinar**
identity:

```text
p(x) − γ = σ₀(x) + Σₖ σₖ(x)·gₖ(x),      every σ a sum of squares
```

At a feasible point every `gₖ(x) ≥ 0` and every `σₖ(x) ≥ 0`, so the right-hand
side is nonnegative and `γ ≤ p(x)` — on the feasible set, which is all that was
claimed. For `min x³ − 3x` on `[0, 3]` POUNCE finds `γ = −2` with

```text
x³ − 3x + 2 = ½(x−1)² + 3⁄2(x−1)²·x + ½(x−1)²·(3−x)
```

and `x = 1` attains it, so the verdict is `global-min`.

Mechanically this is the same machinery with more Gram blocks: one per σ, each
with its own `LDLᵀ`, jointly matching `p − γ` coefficient by coefficient. Three
things are worth knowing about the difference:

- **The claim is weaker, and the certificate says which.** The constraints
  travel with it in `problem.poly_constraints`, and the generated Lean theorem
  is guarded by `∀ i, 0 ≤ g i x`. A consumer that ignored that field would read
  a bound on a box as a bound on all of ℝⁿ.
- **Attainment now needs feasibility.** A point outside the feasible set can
  beat every point inside it — `x = −2` also gives `x³ − 3x = −2` — so the
  candidate carries a second obligation, `gₖ(x₀) ≥ 0` for every `k`, alongside
  `p(x₀) = γ`. Both are exact checks over ℚ.
- **Equality constraints are refused.** An equality needs a sign-unrestricted
  multiplier rather than a sum of squares, and splitting `h = 0` into `h ≥ 0`
  and `−h ≥ 0` leaves a feasible set with empty interior, where Putinar's
  theorem stops guaranteeing that a certificate exists at *any* degree. Refusing
  is better than searching forever.

### `--local`: a ball is just another constraint

A nonconvex problem usually has local minima that are not global, and at one of
those every claim above is simply **false**. `3x⁴ + 4x³ − 12x²` has a minimum at
`x = 1` worth `−5`, and another at `x = −2` worth `−32`. Started near the first,
the solver reports `x = 1` — and no certificate that `−5 ≤ p(x)` exists at any
relaxation order, because it is not true.

What *is* true is that nothing nearby beats it. So say that instead:

```console
$ pounce certify --local cubic.nl cubic.sol -o c.json    # verdict: local-min
```

The point is that this needs no new theory. A neighborhood is a polynomial
inequality like any other —

```text
r² − ‖x − c‖² ≥ 0
```

— so `--local` adjoins it to the `gₖ` family and runs the *same* Putinar
machinery with one extra multiplier `σ_B`:

```text
p(x) − γ = σ₀(x) + Σₖ σₖ(x)·gₖ(x) + σ_B(x)·(r² − ‖x − c‖²)
```

The proved statement is `γ ≤ p(x)` for every feasible `x` **within the ball**,
and with an attaining candidate inside it, `p(x₀) ≤ p(x)` there — which is
local minimality, stated directly. No second-order sufficient conditions, no
constraint qualification, no Taylor remainder, no implicit function theorem: the
Lean proof is the two-line sign argument the unconstrained case already used.

Three practical notes:

- **The radius is stored squared.** ℚ is not closed under square roots, so `c`
  and `r²` are what travel in `problem.neighborhood` and what the theorem
  mentions. `--radius <r>` takes an ordinary radius (default `1`) and squares
  it. A non-positive `r²` is refused: the claim would be vacuous.
- **The centre is not `x*`.** It is `x*` snapped to a coarse rational grid,
  chosen so the true solution stays well inside — the certificate is about the
  ball it names, so that ball has to be one a rational centre can describe
  exactly.
- **Localizing also helps numerically.** A ball is a strong localizer, so a
  low-order relaxation often closes where the global problem's gap will not —
  even when the global claim happens to be true.
- **The candidate must be inside.** `p(x₀) = γ` does not imply `x₀` is in the
  neighborhood: for `x⁴ − 2x² + 2` both `±1` attain the minimum. So the
  candidate carries `‖x₀ − c‖² ≤ r²` as a third exact obligation, next to
  feasibility and attainment.

`--local` is refused on the convex-QP path, where the KKT certificate already
proves a *global* minimum: narrowing that to a ball would trade a strong claim
for a weaker one and gain nothing. It is also refused together with
`--feasible`, which asks for a different claim entirely.

## Certifying the float itself

Every verdict above claims *optimality*, and each is available only where the
mathematics closes. On a nonconvex QP none of them do: the KKT argument needs
convexity, and the SOS route is unconstrained-only. The solver still returns a
point, and a weaker but genuine statement about that point is still worth
having.

```console
$ pounce certify nonconvex_qp.nl nonconvex_qp.sol -o c.json
pounce certify: cannot certify this solve: Ldl(Indefinite { col: 0 })
$ pounce certify --feasible nonconvex_qp.nl nonconvex_qp.sol -o c.json
```

`--feasible` proves two things about the returned `x*`, at an ε the emitter
computes rather than accepts:

1. **ε-feasibility** — every constraint row is violated by at most ε at `x*`.
2. **Existence** — there is a genuinely feasible `x̂` with `‖x̂ − x*‖∞ ≤ ε`.

The second claim is the load-bearing one. Claim 1 alone is satisfiable by a
point sitting just outside an *empty* feasible region, which would make the
certificate worthless; `x̂` rules that out by construction. It is found by
projecting `x*` onto the affine hull of the equalities and the active
inequalities — `x̂ = x₀ − Rᵀ(RRᵀ)⁻¹(Rx₀ − r)` — entirely over ℚ, with the float
solve proposing only *which* face to project onto. If the projection lands
outside some inactive row, or the problem has no feasible point at all, the
emitter refuses; there is no phase-1 search hiding behind the flag.

```json
"verdict": "feasible",
"tolerance": { "num": "1", "den": "25000000" },
"candidate": { "x": [ … ] },
"witnesses": { "feasible_witness": { "xhat": [ … ] } }
```

`candidate.x` is the solver's float converted losslessly to ℚ — the point being
certified is the one you were actually handed, not a cleaned-up neighbour. ε is
the true worst case over both claims, rounded *up* to one significant figure so
the number stays readable without ever understating the violation.

The flag is opt-in rather than a fallback. A failed optimality certificate is a
result worth seeing, not one to quietly replace with a weaker claim — so
`certify` never downgrades on its own, and `--feasible` on a problem where it
would prove nothing (an unconstrained polynomial, where every point is trivially
feasible) is refused too. It is also refused on the constrained polynomial
slice, where the claim would be meaningful but the emitter's exact projection
needs the constraints to be linear.

## What it refuses

Anything outside these slices **exits 2** rather than emit something unsound:

```console
$ pounce certify certify_maximize.nl certify_maximize.sol
pounce certify: certify supports minimize objectives only (v1)
$ echo $?
2
```

Maximize objectives, non-polynomial objectives, and equality constraints on a
higher-degree problem are all refused today. Indefinite `Q` is refused for
*optimality*, but is exactly the case `--feasible` exists to cover.

## Emitting a certificate

Solve first, then certify the result. The `.sol` is an input: `certify` does
not rerun the solver, it certifies the claim.

```console
$ pounce nonconvex.nl
$ pounce certify nonconvex.nl nonconvex.sol -o nonconvex.cert.json
```

For `x⁴ − 2x² + 2` — global minima at both `+1` and `−1`, so the local solve
lands in one basin arbitrarily — the certificate opens:

```json
{
  "schema": "pounce.lean-cert/v1",
  "verdict": "global-min",
  "problem_class": "sos-poly",
  "tolerance": { "num": "0", "den": "1" },
  "bound":     { "num": "1", "den": "1" },
  "binding": {
    "nl_sha256":  "69ee731038777d1c5b26b0ada4191a93d11dd91b10d427a0e7a16d2da01946d8",
    "sol_sha256": "ff379f9e3437895a5af918b6e01e71a6c7742795bc710689f10a05ee4bb9ff49",
    "solver": "pounce 0.9.0"
  },
  "toolchain": { "lean": "leanprover/lean4:v4.31.0", "mathlib": "fabf563a…" },
  "candidate": { "x": [ { "num": "1", "den": "1" } ],
                 "objective": { "num": "1", "den": "1" } }
}
```

Three details are load-bearing. The SDP that finds `G` returns a bound near
`1 − 1e-9`; the certificate says exactly `1/1`. The iterate is near `1 − 4e-10`;
the candidate is exactly `1/1`. And `candidate.objective` equals `bound` — that
equality is the entire difference between this and a `global-lower-bound`
certificate, which omits the `candidate` key rather than nulling it, so a
consumer cannot mistake a bound for a claimed minimizer.

The full field-by-field reference is the
[Lean Certificate Schema v1](schema/lean-cert-v1.md).

## Binding a certificate to your problem

A hash match alone is not enough. A certificate could carry the right
`nl_sha256` while its `problem` block describes something *easier* — drop a
constraint and the resulting proof is perfectly valid, just about a different
problem. `cert-verify` closes that hole by re-deriving the problem from **your**
`.nl` using the trusted, deterministic frontend and comparing:

```console
$ pounce cert-verify nonconvex.nl nonconvex.cert.json
cert-verify: OK — certificate matches this .nl
```

```console
$ pounce cert-verify other.nl nonconvex.cert.json
cert-verify: REJECT — binding.nl_sha256 does not match this .nl
         cert: 69ee731038777d1c5b26b0ada4191a93d11dd91b10d427a0e7a16d2da01946d8
         .nl : ba8f872e8e6881259b04c6c5999ee211bd2e75c2f977e6ff853b6f18e701f10b
```

Exit `0` on match, `2` otherwise. This step does not run Lean.

Producer and consumer share the same re-derivation code rather than
reimplementing it, so a mismatch always means real drift — never two
implementations disagreeing about what the `.nl` says.

## The acceptance rule

A certificate is accepted **iff all three** hold. Any one alone is insufficient:

1. **`pounce cert-verify`** passes — the proof is about *this* problem.
2. **`lake build`** of the generated Lean succeeds, under the consumer's own
   pinned toolchain, not the one the certificate suggests.
3. **The axiom audit** — `#print axioms` lists only `propext`,
   `Classical.choice`, `Quot.sound`, and no `sorryAx`. Without this, a proof
   could lean on an admitted lemma and still build.

Step 3 is why the toolchain pin is recorded but not trusted: a proof reproduces
only under a matching Lean/Mathlib, and the consumer must be the one to decide
which that is.

## How it is tested

`scripts/check-lean-cert.sh` runs three layers on every fixture, and the third
is the one that tests the actual soundness claim:

| Layer | Checks | Needs Lean? |
|---|---|---|
| 1 | certificates regenerate byte-identically; `cert-verify` binds each to its `.nl` | no |
| 2 | codegen reproduces the golden `.lean`; `lake build` succeeds; axioms audited | yes |
| 3 | **deliberately corrupted certificates must FAIL to build** | yes |

Layer 3 carries one forgery per obligation a verdict rests on — a corrupted KKT
dual, an indefinite Hessian, a broken Farkas ray, an inflated SOS bound, an
indefinite Gram matrix that still satisfies the SOS identity, a minimizer that
does not attain a bound which is itself genuine, and — for `feasible` — an `x̂`
that is close to `x*` but not feasible, alongside one that is feasible but too
far away. That last pair is the point of the technique: each forgery leaves the
*other* obligation perfectly intact, so a proof that checked only one of them
would sail through. The recession forgery works the same way: a direction that
is feasible (`A d = 2 ≥ 0`) and descending (`c·d = −1 < 0`), and so passes every
condition the LP slice ever had, but is not flat (`Q d ≠ 0`).

A test suite where everything passes proves only that valid inputs work; these
fixtures are the ones that would catch a proof that accepts too much.

## Limits

* **Feasibility is checkable; the model is not.** Like `verify`, this certifies
  a statement about *a given* model. Whether that model is the right one is
  owned outside POUNCE.
* **The slice is narrow by construction.** Every extension is real
  mathematics — the SOS route needed exact Gram rounding, not plumbing — so
  refusal is the honest default rather than a limitation to work around.
* **`unbounded` needs a direction the objective is *exactly* flat along.** For
  an LP every direction qualifies (`Q = 0`), which is why that case needed no
  refinement at all: its conditions `A d ≥ 0` and `c·d < 0` are inequalities,
  and an inequality satisfied with margin survives the f64→ℚ conversion. A
  nonzero `Q` adds the equality `Q d = 0`, which no float meets, so the
  direction is recomputed by projecting the solver's diverging iterate onto
  `ker Q` over ℚ. If `Q` is definite there is no such direction and the emitter
  refuses — correctly, since the objective is then bounded below along every
  line. Equality rows are still outside the slice.
* **The SOS route handles inequalities, not equalities.** Inequality
  constraints go in as Putinar multipliers; an equality would need a
  sign-unrestricted one, and splitting it into two inequalities empties the
  feasible set's interior, where Putinar stops guaranteeing that a certificate
  exists at any degree. Routing is decided by objective *degree*, never by a QP
  extraction failing, so a convex QP can never silently downgrade from
  `global-min` to a mere bound.
* **A `local-min` is only ever about the ball it names.** It is silent outside
  it, and at a local minimum that is not global, silent is the only honest
  thing to be — a wider claim would be false. The ball is part of the theorem
  statement, so a consumer that ignored `problem.neighborhood` would read a
  local result as a global one. What `--local` does *not* prove is *strict*
  local minimality: nothing rules out a tie elsewhere in the ball.
* **`feasible` proves nothing about optimality.** It is a statement about where
  a point *sits*, not about whether anything better exists. On a nonconvex
  problem that is genuinely all a local solve establishes — the verdict is
  narrow on purpose, and reaching for it says the optimality claim was not
  available.
* **`--feasible` needs an active set the float gets right.** The projection
  target comes from the solver's slacks. If the guess is wrong the exact
  projection lands outside an inactive row and the emitter refuses rather than
  widening ε to cover the miss.
