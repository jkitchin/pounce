# Julia native complementarity integration

Base/native source: `550b1d5e002198b6b0d1fd31727544d2ca837f2c` (fresh upstream).
This source contains the finalized pre-0.17 API from #1161, but still reports
native 0.16.0. Validation must not be described as testing a released 0.17 binary.

## Implementation and result contract

- Bind the finalized optional complementarity API, including `Cuint flags = 0`
  in the batch setter. Require its symbol group, including `PounceErrorString`,
  before advertising support; the required import-time symbols are unchanged.
- Preserve per-function success conventions. Negative extension errors become
  `NativeAPIError` with a copied message and one-based optional batch position.
  Bool-returning APIs keep their original convention.
- Initialize the two sized C output structs before every call; independently
  compare their sizes and every offset with a compiled current header.
- Retain original MOI vector functions, use dedicated free rows and a separate
  nonlinear evaluator block, and let `AddPounceComplementsConstraint` lower all
  five bound cases. Both sides of Hessian sparsity contribute to L-BFGS hints.
- Repeated paired variables are rejected atomically, including fixed/free
  declarations. Paired parameters are rejected. No silent product reformulation.
- Preserve ordinary QP, oracle and NLP block row/derivative offsets. Parameters
  share their current values across evaluators without rebuilding the handle.
  Bound edits rebuild from original declarations and immediately invalidate MOI
  results. General constraint deletion/function replacement remains unsupported.
- Complementarity `ConstraintDual` is `[function-row multipliers; zeros]` in the
  ordinary wrapper sense convention. Explicit variable bounds retain their own
  side's signed MPCC multiplier, without clamping or double counting. Fixed/free
  declarations use ordinary NLP bound handling. These stationarity multipliers
  can be nonunique and are not convex dual certificates; `DualStatus` is unknown.
- Primal status recomputes original row/bound feasibility and the mixed natural
  map even for successful native returns. Stationarity certificates, driver
  iterations and formulation outcomes remain separate diagnostics.

## Test scope and an upstream MOI test limitation

Package regressions cover all bound cases in direct/cached models; affine,
quadratic, nonlinear and variable-vector inputs; metadata and ownership;
invalid and duplicate declarations; parameter reuse and bound-class rebuilds;
signed lower/upper multipliers in both objective senses; biactivity; mixed
QP/oracle/legacy/new NLP blocks; exact/L-BFGS/finite-difference modes; registered
operators; callbacks; native batch atomicity/error indices; unavailable results,
owned diagnostics and JSON reports. Older libraries must still load, solve NLPs,
and reject unsupported complementarity declarations.

The newly enabled MOI `test_nonlinear_qp_complementarity_constraint` stores a
quadratic objective but never sets `ObjectiveSense`; MOI defaults to
`FEASIBILITY_SENSE`. Its expected minimizer/objective therefore do not follow
from the supplied model. Exclude that one upstream test and run the same model
with explicit minimization in `test/complementarity.jl`, followed by a separate
feasibility-sense control. Do not change the optimizer's objective semantics to
satisfy the missing test instruction. Other enabled MOI complementarity tests
remain in the full suite.

Application harness: `benchmarks/julia_mpcc_opf/run.jl`. It wraps the original
MPCCModels NLP callbacks in an MOI NLPBlock and declares original variable pairs
through `MOI.Complements`. Four original volt-var/volt-watt scenarios are run in
exact and limited-memory modes. Acceptance uses the upstream independent checks
for original constraints, bounds, min(G,H) and direct controller laws, not just
solver status. Source/license/citation are recorded in that directory and the
benchmark catalog. This is correctness evidence, not a speedup or global-optimum
claim.

## Final validation

To be recorded against the committed implementation, with native/library,
Julia/dependency revisions and log hashes. Development runs are not the final
release evidence.
