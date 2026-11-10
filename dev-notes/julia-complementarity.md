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

## Final validation — 2026-11-10

Implementation/test revision: `1a4bb7fba02134be7e3ac4e4cb837f97d4089b3c`.
All runs below used that clean implementation. This evidence note and the local
roadmap were updated afterward; executable code/tests were unchanged.

On macOS arm64, full-suite runs use `Pounce.jl/test/runtests.jl`, the same entry
point as `Pkg.test()`, including the threaded subprocess and MOI conformance
suite. Main environments: JuMP 1.32.1, MOI 1.54.0, Ipopt.jl 1.16.0 and JSON
1.13.4. Minimum environment: JuMP 1.29.0 and MOI 1.48.0 on Julia 1.10.13.

| Native source | Julia | Check | Passed assertions | Log SHA-256 |
|---|---|---|---:|---|
| upstream `550b1d5e` | 1.10.13 | Full suite | 4,586 | `c1bc5868942c6ad93ffef2afece751d87798f00e5d24232dd4bb92c4d28420ef` |
| upstream `550b1d5e` | 1.13.3 | Full suite | 4,586 | `ae80f7e850657ae10974064062cc7bfa70912199ce66c3ceddf8cc21a7b8376c` |
| v0.15.0 `13c38013` | 1.10.13 | Full suite | 4,116 | `495d1552cf6138ed8d279f945c25196bd4adb1e7877bc5401a69c5126171e87a` |
| v0.16.0 `9cbb5a71` | 1.13.3 | Full suite | 4,116 | `9cc3078d52091902e42a5d3208882ae9a3c9a1a3c00ba91a2c01a2434817a243` |
| upstream `550b1d5e` | 1.10.13 | Minimum dependencies, focused | 365 | `41a264f06417a904958e635ec192a737d7189877cefb5d887c1f234ae1f2ce2b` |

C header layout checks pass on the same implementation: **57** assertions with
the current header (both new sized structs plus the existing scalar/linear-stat
ABI), and **17** each with v0.15.0 and v0.16.0 headers. Parent control at
`550b1d5e` confirms the original wrapper rejects the scalar JuMP complementarity
example that the new adapter solves.

Native library identities (release C-interface builds):

| Native revision | Library SHA-256 |
|---|---|
| `550b1d5e002198b6b0d1fd31727544d2ca837f2c` | `2b29fb0ab0dd1d602143ad802fa4d0b729d337de97b2e5a4d8501944bbfedae8` |
| `13c38013498eecf54578d2c0b8405b0086dd62a6` | `575c336c8a0512c0b39ba8cdd9dfa25224ead87fd43a4a5e0c57ed04546ca64a` |
| `9cbb5a711661c07525e66c3ebb7d2b42bea56785` | `7bce9ff05910c002a4b31a56f1fd307ef1624338f9506a7791d9b24c83ce26cc` |

The current native CLI and library were built from the base above before wrapper
edits. No Rust source changed. Rust toolchain: rustc 1.96.0 (ac68faa20 2026-05-25).

### Original-model application checks

At the implementation revision above, all **48 assertions across eight solves**
pass on Julia 1.13.3, native `550b1d5e`, MPCCModels
`19a4b275c8ebb3fb5f7a051cfe36c45ad1a948de`, ADNLPModels 0.8.15 and NLPModels
0.21.12. Scenarios: `vv_lower_knee`, `vv_injection_20`, `vw_ramp_50` and
`curtailment`, each with exact and limited-memory Hessians.

Every scenario satisfies all five independently recomputed residuals at the
1e-6 acceptance threshold: original constraints, original bounds, min(G,H),
volt-var law error and volt-watt cap violation. `vv_lower_knee` reports a W
certificate in both modes; the other six runs report S. The W result is not
relabelled as strong stationarity or a convex dual certificate.

### Other checks and retained evidence

Julia formatting, `git diff --check`, `cargo fmt --all -- --check`, documentation
consistency and an mdBook build pass. The release-consistency guard passes with
the same previously documented host workaround: write the extracted CRATES
array to a temporary file and source that instead of Bash process substitution.
The unmodified script fails on this host with `CRATES[*]: unbound variable`;
its remaining logic and the checked array are unchanged in the successful run.
No repository guard script was modified. Whole-workspace Rust tests/Clippy and
Python tests were not rerun locally for this Julia-only implementation; hosted
CI remains separate evidence.

Local evidence filenames and SHA-256 values:

| File under `/private/tmp/` | SHA-256 |
|---|---|
| `pounce-complements-opf-final.log` | `1b5c73722daebff6edca3561220f44b1a9d2d91e079de908cb45d79342c14af7` |
| `pounce-complements-abi-current-final.log` | `3717f38d260a93eea2ee2a5c4c04e98c70b8b72d8263dc6406a7c8fcc6d4a23a` |
| `pounce-complements-abi-v015-final.log` | `185e7732efa410f7137a5f31e9819ac82c77fff73fa88b997aa1cbbdc718a338` |
| `pounce-complements-abi-v016-final.log` | `68caa4062cd1f68e7f9b92b8fc77d74695260bc56d47526073a434bce875dd35` |
| `pounce-complements-parent-check.log` | `c61f43d11ff70e82df3de7645230e8aaba3d0d7decfabe7ee9df71a3126dae5d` |
| `pounce-complements-docs-check.log` | `57ff859285322d713488d6a2e6c403db7440d340d61fcbeab197051e71ef16de` |
| `pounce-complements-release-workaround.log` | `61c54d7cd127ee2b577376b3b4b9fa12a329225f1367d9786d63db52a606f8be` |

Environment Manifest SHA-256 values:

- `/private/tmp/pounce-julia017-env110/Manifest.toml`: `547855826951de5b22b7b9fc6e7c36c700ff7383e44e0279b91fdc2409863be9`.
- `/private/tmp/pounce-julia017-env/Manifest.toml`: `daf7593396cb46cc6b2dbf1b429b86e429c73cd01f92dbb0b83e21bf0d665beb`.
- `/private/tmp/pounce-diagnostics-minimum/Manifest.toml`: `c695c4f78da44e564bc40cf8bfc3b4544b252d388e942523bfaf1bee41dd1136`.
- `/private/tmp/pounce-complements-opf-env/Manifest.toml`: `0a8c65f044d9d0f21599017b10438d98fe8418f62246a6b32bcceedc5a81bc50`.

At the user's request, completed Cargo build intermediates were removed from
the current and earlier development build directories. Runnable native libraries,
executables and evidence were retained. No test depends on those discarded
intermediates; subsequent validation used the retained, hashed libraries.


## Review correction: stable complementarity feasibility

Fix and permanent regressions: `64a9a4eb2605cffd729d47fdb0959866688aaaa6`.
The native library remains the build of
`550b1d5e002198b6b0d1fd31727544d2ca837f2c`, SHA-256
`2b29fb0ab0dd1d602143ad802fa4d0b729d337de97b2e5a4d8501944bbfedae8`.

The initial natural-map calculation could lose F when subtracting it from a
large x. Actual solves stopped with `max_iter=0` at x=1e16/F=±1 or
x=1e12/F=±1e-5 incorrectly returned `FEASIBLE_POINT`. Free, lower-only,
upper-only and boxed interior declarations all reproduced. The equivalent
`abs(clamp(F, x-upper, x-lower))` preserves these residuals.

The permanent tests first ran against the pre-fix implementation at `8bdb5c6`
with only the new tests in the worktree: 32 assertions failed across 16 actual
solves (the primal status and residual in each). The log records the parent
commit and worktree-diff hash. After correction, all 156 new assertions pass:
80 cover those actual solves, and 76 cover controlled returned points at bounds,
fixed/free cases, out-of-bound points and nonfinite values. The controlled-point
checks test result interpretation, not native convergence or diagnosis.

Validation on the clean fix commit above:

| Environment | Validation | Result |
|---|---|---|
| Julia 1.10.13, current native | Full suite | 4,742 passed |
| Julia 1.13.3, current native | Full suite | 4,742 passed |
| Julia 1.10.13, JuMP 1.29 / MOI 1.48, current native | Focused complementarity | 521 passed |

Julia/Rust formatting, diff checks, documentation consistency and mdBook passed.
The release guard passed using the same verified temporary-file workaround
noted above. No native rebuild was needed. The older-native, C ABI and OPF
results in the earlier sections retain their original implementation revision;
they were not rerun for this result-calculation change. Hosted CI on the updated
PR is separate from these local results.

| File under `/private/tmp/` | SHA-256 |
|---|---|
| `pounce-complements-cancellation-red.log` | `4a97784558382359969690786bc20aa0a435d43b8a1ad3972162c58a07f7f1b6` |
| `pounce-complements-cancellation-110.log` | `7f615345d0a14afc592f00208683e2f882ed7eb9416e9776479c71fcfb8f3f65` |
| `pounce-complements-cancellation-113.log` | `3a482e937dfe28feca35ac230bd9786d993ff67dc938c3e114c6e86b6860c2b6` |
| `pounce-complements-cancellation-minimum.log` | `00c5439bc5f6ae6ea44d4bbbe5a6f7fd254060a5db7e085920bbb1df3485050f` |
| `pounce-complements-cancellation-release.log` | `61c54d7cd127ee2b577376b3b4b9fa12a329225f1367d9786d63db52a606f8be` |
