/-
  POUNCE Lean certificate — worked example (DESIGN SKETCH, not yet compiled).

  Generated target for `qp.cert.json` (schema `pounce.lean-cert/v1`,
  verdict = global-min, problem_class = qp-convex).

  Problem (convex QP):
      minimize    f(x) = ½ xᵀ Q x ,   Q = diag(2, 2)   ⇒  f = x₁² + x₂²
      subject to  x₁ + x₂ ≥ 1
  Candidate from POUNCE:  x* = (1/2, 1/2),  f(x*) = 1/2,  dual λ = 1.

  This file is the END-TO-END REFERENCE the `pounce-lean` codegen targets. It
  has NOT been run through Lean/Mathlib yet — treat the proof terms as the
  intended shape, to be validated when the toolchain exists (open item below).

  Trust binding: the codegen embeds the canonical hashes as literals so the
  theorem provably concerns these exact bytes. A consumer accepts iff this
  builds AND `nl_sha256` matches the SHA-256 of its own canonical .nl.
-/
import Mathlib

namespace PounceCert.QPExample

-- binding (from cert.binding); placeholder zeros in this design sketch
def nlSha256  : String := "0000000000000000000000000000000000000000000000000000000000000000"
def solSha256 : String := "0000000000000000000000000000000000000000000000000000000000000000"

/-- Objective, expanded from ½·xᵀQx with Q = diag(2,2). All over `ℚ`. -/
def f (x₁ x₂ : ℚ) : ℚ := x₁ ^ 2 + x₂ ^ 2

/-- Feasible set: the single linear constraint `1 ≤ x₁ + x₂`. -/
def Feasible (x₁ x₂ : ℚ) : Prop := 1 ≤ x₁ + x₂

-- candidate x* (from cert.candidate.x), exact rationals
def xs₁ : ℚ := 1 / 2
def xs₂ : ℚ := 1 / 2

/-- Tier 1: the candidate is feasible (here exactly: 1/2 + 1/2 = 1 ≥ 1). -/
theorem candidate_feasible : Feasible xs₁ xs₂ := by
  unfold Feasible xs₁ xs₂; norm_num

/-- Sanity: the reported objective value matches (cert.candidate.objective). -/
theorem candidate_objective : f xs₁ xs₂ = 1 / 2 := by
  unfold f xs₁ xs₂; norm_num

/--
  Tier 3 (global): `x*` is a global minimizer.

  Proof is the exact convex-QP identity, which the codegen discharges with the
  witnesses from the cert:

    f(y) − f(x*) = ½ (y−x*)ᵀ Q (y−x*)         -- ≥ 0 by `hessian_psd` (Q ⪰ 0)
                 + ∇f(x*)·(y−x*)               -- ≥ 0 by KKT (`duals`) + feasibility

  For this instance it reduces to the SOS witness
    y₁²+y₂² − ½ = ½(y₁−y₂)² + ½(y₁+y₂−1)² + (y₁+y₂−1),
  every term ≥ 0 once `1 ≤ y₁+y₂`. `nlinarith` finds exactly this combination
  from the hints (which are the cert's PSD + dual data made concrete).
-/
theorem global_min :
    ∀ y₁ y₂ : ℚ, Feasible y₁ y₂ → f xs₁ xs₂ ≤ f y₁ y₂ := by
  intro y₁ y₂ hfeas
  unfold f xs₁ xs₂ Feasible at *
  nlinarith [sq_nonneg (y₁ - y₂), sq_nonneg (y₁ + y₂ - 1), hfeas]

end PounceCert.QPExample
