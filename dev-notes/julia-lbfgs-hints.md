# Julia L-BFGS nonlinear-variable hints

This change binds `IpoptSetNonlinearVariables` and
`IpoptClearNonlinearVariables` and supplies a structural subset automatically
for Julia limited-memory solves. It starts from upstream
`d5fa98080b9bce9fa3f5a93eb06b435c802dc286`, independently of the native 0.17
compatibility PR. It does not widen the native version gate or change Rust.

## Declaration policy

The subset is the sorted union of both coordinates of the combined
Lagrangian Hessian sparsity. Existing assembly includes quadratic objective
and constraint terms, modern/legacy NLP evaluators, and vector nonlinear
oracles mapped into model columns. Duplicate and off-diagonal entries are
handled without evaluating any Hessian values. The existing pattern can
conservatively include inactive objective slots; extra columns are safe,
whereas omitting a genuinely nonlinear column is not.

A missing Hessian evaluator makes the combined pattern unknown, not empty.
No automatic declaration is made then. A known empty pattern declares a
zero Hessian. Parameters do not become native columns, and a parameter's
current numerical value is not used to decide whether a column is linear.
The existing revision caches retain the structural pattern through parameter
and bound edits and recompute it on structural changes. Each new native
handle receives the declaration before solving.

Automatic declarations apply only to `limited-memory`, after user options
have been installed. An explicit `num_linear_variables` option disables
automatic detection and retains the native prefix convention; setting it to
zero is the full-space control. Exact and finite-difference modes receive no
automatic declarations. Explicit low-level declarations can also affect native
finite-difference structure, as implemented by the native application.

The two native symbols are optional and detected on the loaded library at
runtime. They are not added to the required import-time symbol list. Automatic
use skips unsupported libraries; explicit low-level calls report unsupported
capability. Low-level lists use distinct, one-based indices, are copied into
`Cint` storage for the call, and are copied by the native library. Invalid
lists, freed handles, and mutation during a solve are rejected.

## Evidence

Implementation: `901c8a9c23db876e316e7b8cc9d54998ec647c85`. Initial validation
used test revision `ad06a4ebab4538f3984e71d66abdbab49f26e3f3`; subsequent
regression-test promotion is recorded below.

The integration fixture reuses the model from
`crates/pounce-cinterface/tests/nonlinear_variables_mask.rs`, with its nonlinear
columns permuted to Julia indices 2 and 5. Automatic and explicit declarations
produce identical native callback trajectories and final points. The
`num_linear_variables=0` control produces a different trajectory but agrees
on the solution within tolerance. Exact and finite-difference trajectories
are unchanged by enabling automatic detection. These are correctness controls,
not a benchmark or a claim of faster solves.

Additional checks cover clearing and replacing declarations, native ownership
of the index list, empty/all-variable subsets, invalid input, callback-time
mutation, freed handles, quadratic/off-diagonal support, legacy NLP structure,
vector-oracle index mapping, first-order-only evaluators/oracles, structural
rebuilds, parameter/bound updates, and direct/cached JuMP models. Missing
optional symbols are simulated by substituting the lookup handle while the
required bindings continue to use the real solver library; that path runs a
real full-space solve.

The permanent suite also checks fixed nonlinear columns (including an empty
reduced nonlinear space), switching Hessian modes, enabling the full-space
override after a hinted solve, and adding a first-order-only oracle after a
hinted solve. These cases verify that rebuilt handles discard stale declarations
and that fixed-column compression preserves the intended subset.

## Initial local validation

At wrapper/test revision `ad06a4ebab4538f3984e71d66abdbab49f26e3f3`, all four
full-suite runs below pass **4,055 assertions** each on macOS arm64, including
the parent check of the separate two-worker threading subprocess. The full
runs use JuMP 1.32.1, MathOptInterface 1.54.0, Ipopt.jl 1.16.0 and JSON 1.13.4
in temporary, Julia-version-specific environments. They invoke
`Pounce.jl/test/runtests.jl`, the same entry point used by `Pkg.test()`.
Native libraries were built with Rust 1.96.0 and the locked release build of
`pounce-cinterface`. Only this note and the local roadmap changed during
these final runs.

| Native source | Native commit | Julia | Full-suite log SHA-256 |
|---|---|---|---|
| Upstream development | `d5fa98080b9bce9fa3f5a93eb06b435c802dc286` | 1.10.13 | `5cb0089b1edcaa82c0af539b4598293cf3c1ea399d48f5de8d2c06dc87179f46` |
| Upstream development | `d5fa98080b9bce9fa3f5a93eb06b435c802dc286` | 1.13.3 | `51d5b208fa5e8f705c1965a2a6822029af6353b79a337396f92e8ad5f7056a2c` |
| v0.15.0 | `13c38013498eecf54578d2c0b8405b0086dd62a6` | 1.10.13 | `0a9ba7c381e6e9335460c6b3541366de190b27bc0dd1aacf009cfafca12e2d12` |
| v0.16.0 | `9cbb5a711661c07525e66c3ebb7d2b42bea56785` | 1.13.3 | `af365b72211bf2ca125484e948f1c1181320ae2d9b0064373911ba972d1b180d` |

All **80 focused assertions** also pass on Julia 1.10.13 with minimum-supported
JuMP **1.29.0** and MOI **1.48.0**, at the same wrapper/test revision and native
`d5fa9808`. Log SHA-256:
`602d6b51e26f4e760550aeb995c18d11dce72bda74649e50fbb333bb46dd19cc`.

Native library SHA-256 values:

- `d5fa9808`: `4a9dd2a85dc363778e3899b0ea090198720eb3f874b32e0d8bce29cb5f9e2418`.
- v0.15.0: `575c336c8a0512c0b39ba8cdd9dfa25224ead87fd43a4a5e0c57ed04546ca64a`.
- v0.16.0: `7bce9ff05910c002a4b31a56f1fd307ef1624338f9506a7791d9b24c83ce26cc`.

Julia formatting, documentation consistency, mdBook build, and
`git diff --check` pass. No Rust solver code is modified; the Julia controls
above exercise the changed approximation-space selection. Hosted CI is
separate from these local results.

## Promoted review regressions

Test revision `546fa66d83304a034fff707c16a8bb10b45465da` promotes the 47
additional review assertions into `test/nonlinear_variables.jl`, which is
already included by `test/runtests.jl` and therefore runs in normal package CI.
The full suite now passes **4,102 assertions** on each of Julia 1.10.13 and
1.13.3 against upstream native `d5fa9808`. The expanded focused suite passes
**127 assertions** on native v0.15.0 and v0.16.0, and with minimum-supported
JuMP 1.29.0/MOI 1.48.0. The native builds, hashes, platform and other dependency
versions are the same as in the initial validation above. Only validation
notes and the local roadmap changed during these runs.

| Native source | Julia | Check | Log SHA-256 |
|---|---|---|---|
| Upstream `d5fa9808` | 1.10.13 | Full suite | `adf423a3039a9448aefdf6c36687d2f546b194e5bb4b536f0cebb4fec8b33275` |
| Upstream `d5fa9808` | 1.13.3 | Full suite | `68db10ec31a27ae0782187bda824047aa08adebe18622559f7e76df78cca1a93` |
| v0.15.0 | 1.10.13 | Focused | `bbe134d87cf83a14da675e48ae10ebb1b2a4080f5f4364194ef1502103ebd402` |
| v0.16.0 | 1.13.3 | Focused | `168fdda08eb2622c4e0d780e4eb19bf3c99a807c405770001d959d71e42bf0ba` |
| Upstream `d5fa9808`, minimum JuMP/MOI | 1.10.13 | Focused | `2be7e0c113a67b7ca1c482856ad3d874277dcc95f390b21d60d9b0d32b33905e` |

The full suites include the threading subprocess. Julia formatting and
`git diff --check` pass. This promotion changes tests and documentation only.
