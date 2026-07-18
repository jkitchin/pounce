> ## SUPERSEDED — hand-authored example
>
> This was written before the emitter existed. Real, machine-generated
> certificate/proof pairs now live in `crates/pounce-cli/tests/fixtures/`
> (`certify_qp`, `certify_box`, `certify_range`, `certify_eq`) and are
> regenerated and kernel-checked by `scripts/check-lean-cert.sh`.
>
> The maths here is correct — the SOS identity checks out — but `qp.lean` was
> never compiled, and its namespace differs from the shipped convention.

# Lean certificate — worked example

End-to-end reference for the `pounce-lean` codegen. Design-stage; see
`dev-notes/lean-certificate.md` for the architecture and
`../lean-cert-schema-v1.md` for the schema.

## The problem

A tiny **convex QP** chosen so every quantity is exact and the proof needs no
tolerance fuzz:

```
minimize    f(x) = x₁² + x₂²          (= ½ xᵀQx with Q = diag(2,2))
subject to  x₁ + x₂ ≥ 1
```

Solution: `x* = (1/2, 1/2)`, `f(x*) = 1/2`, dual `λ = 1`. Both coordinates are
exactly representable (`1/2 = 1·2⁻¹`), the active constraint holds *exactly*
(`1/2 + 1/2 = 1`), so `tolerance = 0` — a fully exact certificate.

Why it exercises the whole `global-min` pipeline: lossless rational `x*`, exact
constraint evaluation, KKT stationarity (`∇f(x*) = (1,1) = λ·∇g`), and a PSD
Hessian (`Q ⪰ 0`) which, with convex feasible set, upgrades the KKT point from
local to **global** minimizer.

## The two artifacts

| File | Role |
|---|---|
| `qp.cert.json` | What **POUNCE emits** — problem over ℚ + untrusted witnesses. Conforms to `pounce.lean-cert/v1`. |
| `qp.lean` | What **`pounce-lean` generates** from the cert and checks with `lake build`. |

The mapping cert → Lean, field by field:

| cert field | becomes in `qp.lean` |
|---|---|
| `problem.objective` (`½xᵀQx`, Q=diag 2,2) | `def f x₁ x₂ := x₁^2 + x₂^2` |
| `problem.constraints[0]` (`1 ≤ x₁+x₂`) | `def Feasible x₁ x₂ := 1 ≤ x₁ + x₂` |
| `candidate.x` (`1/2, 1/2`) | `def xs₁ := 1/2`, `def xs₂ := 1/2` |
| `candidate.objective` (`1/2`) | `theorem candidate_objective` |
| feasibility (Tier 1) | `theorem candidate_feasible` |
| `witnesses.hessian_psd` + `witnesses.duals` | the `nlinarith` hints in `theorem global_min` |
| `binding.{nl,sol}_sha256` | embedded `nlSha256` / `solSha256` literals |

## Status / open item

`qp.lean` is a **design sketch — not yet compiled**. The proof terms encode the
intended shape; validating that `nlinarith`/`norm_num` actually close them
against a pinned Lean+Mathlib is the first task when the `pounce-lean` repo is
created. The exact-arithmetic SOS identity behind `global_min` is worked out in
the docstring, so the combination exists even if a tactic name needs adjusting.
