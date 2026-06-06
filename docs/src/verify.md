# Verifying Solutions

```sh
pounce verify <problem.nl> <claim.sol> [OPTIONS]
```

`pounce verify` independently checks that the solution in a `.sol` file
actually satisfies the constraints and bounds of a `.nl` problem. It
**re-derives feasibility from the model itself** — it does not trust the
`.sol`'s status line, and it does not rerun the solver. This makes it the
trust anchor when pounce is a *tool an agent calls*: the agent proposes a
solution, and a small, deterministic checker disposes.

Optimization is unusually well-suited to this because a solution is far
cheaper to *verify* than to *produce*: a claimed `x*` is just numbers, and
feasibility is a single constraint evaluation — `g_l ≤ g(x*) ≤ g_u`,
`x_l ≤ x* ≤ x_u` — `O(nnz)` work with no resolve and no dense linear
algebra.

> **Status.** The `verify` *check itself* — recompute feasibility against the
> canonical model, with a content-addressed receipt — is solid and ready to
> use; it needs no secrets and is the recommended default. The **signing and
> remote-service trust layer** layered on top of it (HMAC receipts, the
> `signer_service.py` reference, running the MCP server as a remote authority)
> is a **proof of concept**: it demonstrates the architecture but is not
> hardened for production. If you want to rely on the signed/remote path for
> real, see [Status and hardening](#status-and-hardening) at the end for the
> checklist of what that would take.

## What it defends against

In an agent workflow, three things can go wrong with "here is a solution":

| Failure mode | How `verify` catches it |
|---|---|
| **Fabrication** — a `.sol` that *looks* like a pounce result but wasn't solved | invented numbers fail the residual check against the real model |
| **Ignoring the solver** — claiming success without actually solving | a consumer gates on the receipt's `verified: true` + the problem hash, not on prose |
| **Solving the wrong problem** — dropping or relaxing a constraint to dodge infeasibility | the check runs against the **canonical** constraints/bounds, so a point that is only feasible for a *relaxed* model is rejected here |

The key design rule: **always verify against the canonical problem**, never
against whatever the agent claims it solved. If the agent loosened a bound
to manufacture feasibility, the returned `x*` still violates the canonical
bound, and `verify` reports it.

## Output and exit codes

```text
$ pounce verify gaslib40_steady.nl good.sol
pounce verify — independent solution check
  problem : gaslib40_steady.nl  (1694 vars, 1682 cons)
            sha256:4bb435a3…
  solution: good.sol
            sha256:b77d9e7b…
  claimed solve_result_num: 0

  feasibility (tol 1.0e-6):
    max constraint violation: 1.407e-12  at c[114] (value 1.4e-12, bounds [0, 0])
    max bound violation     : 9.775e-9   at x[24]  (value 1.05, bounds [1.05, 2.0])
  objective at x*: 1.2899875310e0

  optimality (tol 1.0e-6, duals supplied):
    KKT stationarity residual: 2.675e-3  (dual sign +1)
    complementarity residual : 0.000e0

  VERDICT: VERIFIED — solution is feasible for the canonical problem
```

| Exit code | Meaning |
|---|---|
| `0` | `VERIFIED` — every violation within tolerance |
| `20` | `REJECTED` — a constraint or bound violation exceeds tolerance |
| `2` | usage / I/O error (missing file, malformed `.sol`, dimension mismatch) |

A consumer (CI step, agent harness, Makefile) gates on the exit code.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--feas-tol <t>` | `1e-6` | feasibility tolerance for constraints and bounds |
| `--opt-tol <t>` | `1e-6` | stationarity tolerance for the optimality check |
| `--require-optimal` | off | also fail (exit 20) if the KKT stationarity residual exceeds `--opt-tol` |
| `--json-output <path>` | — | write a JSON verification receipt |

### Feasibility is the gate; optimality is reported

By default only **feasibility** gates the exit code. Feasibility is rigorous
and sign-convention-independent — it is the guarantee that matters when the
claim is "this solution meets the constraints."

When the `.sol` carries constraint duals, `verify` also reports a **KKT
stationarity residual** (the bound-projected "dual infeasibility": the part
of `∇f + Jᵀλ` that a valid sign-constrained bound multiplier cannot absorb)
and a complementarity residual. These are *informational* unless you pass
`--require-optimal`. The AMPL dual-sign convention can differ from pounce's,
so `verify` computes the residual for both signs and reports the better one
plus the sign it used. Bound multipliers `z_L, z_U` are not present in a
`.sol`, so they are inferred from which bounds are active.

## The JSON receipt

`--json-output` writes a machine-readable receipt that **content-addresses
both inputs by SHA-256** — so a downstream consumer can confirm exactly
*which* problem and *which* solution were checked:

```json
{
  "pounce_verify_version": 1,
  "solver": "pounce 0.4.0",
  "problem":  { "path": "…", "sha256": "4bb435a3…", "n_vars": 1694, "n_cons": 1682 },
  "solution": { "path": "…", "sha256": "b77d9e7b…", "duals_present": true },
  "tolerances": { "feasibility": 1e-6, "optimality": 1e-6 },
  "feasibility": {
    "max_constraint_violation": 1.4e-12,
    "worst_constraint": { "index": 114, "name": "c[114]", "value": 1.4e-12,
                          "lower": 0.0, "upper": 0.0, "violation": 1.4e-12 },
    "max_bound_violation": 9.77e-9,
    "worst_bound": { "index": 24, "name": "x[24]", … },
    "feasible": true
  },
  "optimality": { "available": true, "stationarity_residual": 2.6e-3, … },
  "verdict": "VERIFIED",
  "verified": true
}
```

A consumer should accept a solution **iff**:

1. `verified == true`, **and**
2. `problem.sha256` equals the SHA-256 of *its own* canonical `.nl`, **and**
3. (when signing is used) the signature validates — see below.

Checking the hash in step 2 is what closes the "solved the wrong problem"
gap at the receipt layer: the receipt is only meaningful for the exact
problem bytes it names.

## The default: recompute, don't trust a receipt

The strongest and simplest design uses **no key and no signature at all**:
the consumer runs `pounce verify` *itself*, against *its own* copy of the
canonical `.nl`.

```sh
# the consumer does this — not the agent
pounce verify ./canonical/problem.nl ./from-agent/claim.sol || reject
```

Because verification is keyless, deterministic, and cheap (`O(nnz)`, no
resolve), the consumer can afford to just *do it* rather than trust someone
else's word. In this design the agent is **never in the trust path**: it
hands over `x*`, and the consumer believes its own arithmetic. There is no
key to steal, so the question "what if the agent gets the key?" does not
arise. Forgery is impossible because nothing is being trusted on faith —
feasibility is decided by *evaluating* `g(x*)`, not by matching fields in a
document.

This is the recommended default. Prefer it whenever the consumer can run
`pounce verify` (or call a verifier it controls). Reach for signatures only
when it genuinely cannot — see below.

## Signed receipts — trust *transport*, conditional on key isolation

Signing addresses a *narrower* situation: the consumer **won't or can't
recompute** — a remote or expensive verifier, or an audit log you want to
trust later without re-solving — and instead wants to trust a receipt
produced elsewhere. A signature lets that receipt be checked without redoing
the work.

When the `POUNCE_VERIFY_KEY` environment variable is set (non-empty), the
receipt gains:

```json
"signature_alg": "HMAC-SHA256",
"signed_fields": ["verify_version","nl_sha256","sol_sha256",
                  "n_vars","n_cons","feasible","verified","verdict"],
"signature": "5bdcc146bf60754e…"
```

The signature is **HMAC-SHA256(key, preimage)**, where `preimage` is a
deliberately **float-free** byte string — only hex hashes, integer counts,
and the verdict — so any language reproduces it byte-for-byte without
float-formatting parity problems. The exact preimage is:

```text
pounce-verify-receipt/v1
verify_version=1
nl_sha256=<hex>
sol_sha256=<hex>
n_vars=<int>
n_cons=<int>
feasible=<true|false>
verified=<true|false>
verdict=<VERIFIED|REJECTED>
```

(eight lines, `\n`-joined, with a trailing newline; booleans lowercase.)
A holder of the key recomputes the HMAC over this preimage and compares it
to `signature`.

### What the signature does and does not guarantee

HMAC gives **existential unforgeability under chosen-message attack — but
only while the key stays secret.** That single condition carries the entire
weight of the scheme:

* **A party *without* the key cannot mint a receipt that validates.** It
  cannot produce a correct MAC for *any* message it has not already seen one
  for. This is the real guarantee.
* **A party *with* the key can sign anything** — including a hand-written
  receipt that falsely says `verified: true`. If the agent obtains the key,
  signing buys you **nothing**, and you must fall back to recomputation.

So the security reduces to a question that is *not cryptographic*: **can the
agent reach the key?** If the keyholder and the agent share a process, a
user account, an environment, or a filesystem, the agent simply reads it —
`echo "$POUNCE_VERIFY_KEY"`, `cat keyfile`, or introspecting shared memory —
and the signature becomes theater. An agent with a shell on the same
user/host as the signer is *not* kept out by HMAC.

> Setting `POUNCE_VERIFY_KEY` for a `pounce verify` that runs **in the
> agent's own environment is the weakest form** and provides no real
> protection against that agent. It is only meaningful when the key lives
> behind a privilege boundary the agent cannot cross. See *Out-of-process
> signing* below.

Two further limits HMAC does **not** address, even with a perfectly isolated
key:

* **Replay.** A previously, legitimately-signed receipt can be re-presented.
  Here that is benign: a receipt attests "this specific `x*` (by
  `sol_sha256`) is feasible for this specific problem (by `nl_sha256`)," and
  that statement stays true. Replay cannot manufacture a valid-looking
  attestation for a *new* (problem, solution) pair the signer never blessed.
  If you ever need freshness ("the work was done *this* time"), add a
  nonce/timestamp to the signed preimage — but for "does this solution meet
  the constraints," freshness is irrelevant.
* **Wrong model.** The signature says nothing about whether the `.nl`
  encodes the problem you *meant*. That spec-ownership question must live
  outside the agent regardless of any crypto. `verify` certifies feasibility
  against a given model; it does not certify that the model is the right one.
  Likewise, feasibility is fully checkable but **global optimality is not** —
  the stationarity residual certifies a first-order/KKT point, not a global
  minimum.

## Out-of-process signing

For a signature to mean anything against a capable agent, the key must live
in a **separate trust boundary** — a different process *and* user/container/
host that the agent cannot exec into or read. The signer exposes
"verify-and-sign" as an RPC; the key never enters the agent's address space.

```text
 ┌─ agent boundary (no key) ────────────┐      ┌─ signer boundary (holds key) ─────┐
 │ agent proposes x*                     │      │ POUNCE_VERIFY_KEY in THIS env only │
 │   │                                   │ RPC  │   1. run `pounce verify` on the    │
 │   └── POST /verify {nl, sol} ─────────┼─────▶│      CANONICAL .nl (+ the key)     │
 │                                       │      │   2. binary signs the receipt      │
 │   signed receipt ◀────────────────────┼──────┤   3. return receipt JSON           │
 └───────────────────────────────────────┘      └────────────────────────────────────┘
        │
        └── relays receipt to the consumer
consumer: accept iff  verified==true  ∧  problem.sha256==canonical  ∧  signature valid
```

What each party can do under this split:

| Party | Has key? | Can forge a verdict? |
|---|---|---|
| Agent (proposer) | no | no — it can only ask the signer to verify a real `x*` |
| Signer service | yes | yes, but it *is* the trusted authority — that's the point |
| Consumer | shares key *or* recomputes | detects any tampering / can verify independently |

The boundary is only real if the agent cannot run code as the signer's user
or on its host. Running the signer as a **separate user, container, or host**
(or behind a KMS/HSM that signs without exposing the key) is what turns
"signed" from theater into a guarantee. An MCP server is already a separate
process from the model, which helps — but only achieves isolation if the
agent also lacks a shell on the same user/host.

A minimal reference signer is in
[`studio/mcp/examples/signer_service.py`](https://github.com/jkitchin/pounce/blob/main/studio/mcp/examples/signer_service.py):
a stdlib HTTP service that holds the key in its own environment, shells out
to `pounce verify`, and returns the signed receipt. The agent calls it; the
agent's environment never contains the key.

## Use in an agent workflow

Putting it together — recompute by default, sign only to transport trust:

```text
agent ── proposes x* ──▶  consumer / verifier-it-controls
                            1. pin + hash the canonical .nl
                            2. pounce verify .nl .sol   (against the CANONICAL model)
                          ◀─ accept iff verified==true ∧ problem.sha256==canonical
```

When the verifier must be remote and the consumer won't recompute, insert an
out-of-process signer (above) and add `∧ signature valid` to the consumer's
acceptance test — remembering that the last clause is only as strong as the
signer's key isolation.

The `pounce-studio` MCP server exposes `verify_solution` so an agent can
*request* a check but cannot fake its result. Deploy that server as a
distinct boundary from the agent (separate user/container) for the signature
to carry weight; otherwise rely on the consumer recomputing.

## Status and hardening

What is ready to use as-is:

* **The feasibility check** (`pounce verify`, and the consumer-recomputes
  pattern). It is deterministic, keyless, content-addressed, and rigorous —
  this is the part to build on.

What is a **proof of concept** — demonstrates the shape, *not* hardened:

* HMAC signing via `POUNCE_VERIFY_KEY`, the `signer_service.py` reference,
  and treating a remotely-deployed MCP server as a signing authority.

If you ever want to depend on the signed/remote path in production, these are
the gaps to close. None are implemented here.

**Key management**

* Don't keep the key in a plain environment variable or file. Use a KMS/HSM
  (or sealed secret) that signs without exposing the key to the process —
  then even a compromised signer host can't exfiltrate it.
* Add key **rotation** and a key **id** in the receipt (`kid`) so a consumer
  knows which key to check against and old receipts stay verifiable across
  rotations.
* Consider an **asymmetric** scheme (e.g. Ed25519) instead of HMAC when more
  than one party must *verify* without also being able to *sign* — HMAC's
  symmetric key means every verifier is also a forger. Public-key signatures
  give public verifiability with a single private signer.

**Transport / service (the moment it leaves stdio)**

* **TLS** on the endpoint; never plaintext for a service that holds a key.
* **Authn/authz** — bearer token or OAuth on every request (MCP's HTTP
  transport supports this). An unauthenticated endpoint that runs solves and
  shells out is effectively remote code execution.
* **Resource limits** — request-size caps, solve **timeouts** (there is a
  `timeout_seconds`, but also wall/CPU/memory limits at the OS level),
  concurrency caps, and **rate limiting**.
* **Sandbox the solve** — treat every `.nl` as untrusted input. Parsing and
  evaluating an arbitrary model is attacker-controlled computation; run it in
  a locked-down container/user with no network and a constrained filesystem.

**Input handling**

* Over a network the path-based tools (`nl_file`/`sol_file`) assume a shared
  filesystem. Prefer **content upload** (the server receives and hashes the
  exact `.nl`/`.sol` bytes) so there's no path-traversal surface and the
  receipt binds what was actually sent. The reference signer's
  `POUNCE_SIGNER_ROOT` allowlist is a stopgap, not a substitute.

**Freshness / replay**

* The current preimage has no nonce or timestamp, so a signed receipt is
  replayable. That is benign for "is this `x*` feasible" (a timeless fact),
  but if a consumer needs "this was checked *recently*" or "in response to
  *my* request," add a **nonce/timestamp** (and a receipt **expiry**) to the
  signed preimage and bump the `pounce-verify-receipt` version.

**Auditability**

* Log every verification (problem hash, solution hash, verdict, key id,
  caller identity) to an append-only store, so a disputed result can be
  reconstructed. Keep the key out of the logs.

**Standing non-goals** (true regardless of hardening)

* `verify` certifies feasibility against *a given* model — it does **not**
  certify the model is the *right* one. Model/spec correctness must be owned
  outside the agent.
* Feasibility is fully checkable; **global optimality is not**. The
  stationarity residual certifies a first-order/KKT point, not a global
  minimum.
