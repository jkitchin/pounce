# Julia native bound updates

## Source and dependency

- Upstream/native source: `550b1d5e002198b6b0d1fd31727544d2ca837f2c`.
- Reviewed complementarity dependency: PR #1163,
  `97cd704df1b005fbd949a81a7b95b2d53f62948c`.
- Bound-update implementation and tests:
  `1baee4edaef2702a0f0aec75a2f5f1dff9a88654`.

The branch was created from upstream and fast-forwarded to the reviewed #1163
head because that PR was still open. The bound-update commit is separate and
depends on its shared error handling and complementarity declarations. Rebase
the dependent commits onto upstream after #1163 merges; do not treat this as
an independent patch against the earlier main branch.

## Contract

`supports_bound_updates()` detects the optional finalized Set/GetPounceBounds
and PounceErrorString symbols. The required import symbol set is unchanged.
Both bindings pass zero reserved flags and use the exact current-header
signatures, including the trailing bad_index output. Julia indices and reported
batch positions are one-based. Reads return owned arrays; sparse getter
indices may repeat, including more entries than the block's dimension. Dense
updates use a null index pointer; omitted bound arrays use null pointers to
preserve that side. Invalid counts, indices and array lengths are checked
before crossing the C boundary; native negative outcomes use NativeAPIError.

MOI setters retain a supported native handle and record touched variable/row
indices in sets. They invalidate results immediately. At the next solve, after
refreshing the nonlinear bound snapshot, one sparse call applies the final
bounds. QP rows retain their indices; modern nonlinear rows use a cached index
map plus the QP/oracle offset. Oracle rows and dedicated complementarity USER
rows are not accidentally included. Parameter edits cannot clear pending bound
edits. A rejected batch remains pending so it can be corrected and retried.

A complementarity class refusal rebuilds from the original MOI declarations.
Other native errors propagate. Structural changes and changed options rebuild
the handle and incorporate the model's final bounds. Older libraries use the
previous bound-edit rebuild path. Changes to existing lifecycle/hint tests
reflect the capability-dependent handle lifetime; their solution and evaluator
checks remain intact.

This saves handle reconstruction and state replay. It does not retain native
solver state, factorizations, or implicitly reuse prior primals/multipliers.
No timing or application-speedup claim is made.

## Validation scope

The focused suite has 377 assertions on current native after the review-check
promotion below (198 originally), covering:

- Direct and cached variable/affine/quadratic/nonlinear edits, compared against
  fresh models for primal solutions, duals and termination status.
- Coalescing repeated edits, result invalidation before the native write,
  parameter edits, explicit options changes, structural rebuilds and L-BFGS
  structural hints.
- Failed batch atomicity and recovery, including NaN bounds; crossed ordinary
  bounds are compared with a fresh model and corrected afterward.
- Sparse/dense C calls, optional sides, USER/EFFECTIVE views, duplicate read
  indices, owned outputs, one-based error positions, callback and freed-handle
  guards.
- Mixed QP, oracle, modern NLP and complementarity row offsets in exact and
  L-BFGS modes; simultaneous variable, row and parameter updates.
- All 25 complementarity bound-class transitions against fresh declarations,
  including same-class reuse, transient edits coalescing back to the original
  class, class-change rebuilds, and fixed-to-fixed reuse.
- Explicit primal and lower/upper/row dual starts, including MOI sign conversion,
  observed at the C boundary after a handle-preserving bound update.

The same public bound-edit control was run on the archived parent and current
wrapper with the same native library. Both solve to x=2 after changing an upper
bound from 4 to 2; only the new wrapper retains the native handle. Both controls
check immediate result invalidation and the returned solution/status.

Development probe logs are not the final evidence: the first probe used JuMP's
`set_normalized_rhs` on a nonlinear constraint, which JuMP does not support.
The permanent test uses MOI.ConstraintSet and accounts for JuMP's nonlinear
constant normalization. No solver code was changed to accommodate that test.

All final test logs below identify the clean implementation commit, Julia
environment, native source, loaded library and its SHA-256. The native source
still reports version 0.16.0; it provides the forthcoming 0.17 APIs, and is not
a released 0.17 binary.

Documentation consistency, mdBook, Julia/Rust formatting and diff checks passed.
The release guard passed with the same host-only temporary-file workaround for
Bash process substitution described in `julia-complementarity.md`; the extracted
CRATES array was verified against the unmodified publish script. Full Rust and
Python tests are left to hosted CI for this Julia-only change. The existing
native libraries were reused; no Cargo build intermediates were regenerated.


## Initial local results

All feature validation below was run on implementation
`1baee4edaef2702a0f0aec75a2f5f1dff9a88654` with a clean tracked worktree.
The archived parent control uses `97cd704df1b005fbd949a81a7b95b2d53f62948c`.

| Native source | Julia | Scope | Result |
|---|---|---|---|
| `550b1d5e002198b6b0d1fd31727544d2ca837f2c` | 1.10.13 | Full suite | 4,940 passed |
| `550b1d5e002198b6b0d1fd31727544d2ca837f2c` | 1.13.3 | Full suite | 4,940 passed |
| v0.15.0 `13c38013498eecf54578d2c0b8405b0086dd62a6` | 1.10.13 | Full suite / rebuild fallback | 4,217 passed |
| v0.16.0 `9cbb5a711661c07525e66c3ebb7d2b42bea56785` | 1.13.3 | Full suite / rebuild fallback | 4,217 passed |
| `550b1d5e`, JuMP 1.29 / MOI 1.48 | 1.10.13 | Focused bounds | 198 passed |

Native library SHA-256 values:

- Current: `2b29fb0ab0dd1d602143ad802fa4d0b729d337de97b2e5a4d8501944bbfedae8`.
- v0.15.0: `575c336c8a0512c0b39ba8cdd9dfa25224ead87fd43a4a5e0c57ed04546ca64a`.
- v0.16.0: `7bce9ff05910c002a4b31a56f1fd307ef1624338f9506a7791d9b24c83ce26cc`.

| File under `/private/tmp/` | SHA-256 |
|---|---|
| `pounce-bounds-current110.log` | `b28313db0076bc06ec869ad542c4a4f84cf6912f7846e9b808d6d649d3af298f` |
| `pounce-bounds-current113.log` | `59f30ef1fe6c23b7a3dcf8cdd21b5133c28850a3fc5e446f718a49e1fa3154a4` |
| `pounce-bounds-v015.log` | `4ac454514f9a5c3e85608a5eaf6b0c7cd759271ff4b84cd61401dc94644162a0` |
| `pounce-bounds-v016.log` | `ab90e75b52f1e0495f241c51895e6f3bf5f09f848c439fbd271109e5ccb4db4a` |
| `pounce-bounds-minimum.log` | `21477ab34205108f2a2e853070365dd29605405d4e1539b40817851638f261a9` |
| `pounce-bounds-parent-control.log` | `00163d7525da01a042ef4c3cef744d3b5137720dfa4b15bc737208386d3b09e3` |
| `pounce-bounds-updated-control.log` | `87857e21828dcc715afc407cf0b1f25282e241f3ae7da1f11fa62cf8b5e2ea1e` |
| `pounce-bounds-parent-control.jl` | `518f2050eb97d422dc11195f51f48951660954884e7045e2e57e8c5b35bb40e3` |
| `pounce-bounds-release.log` | `61c54d7cd127ee2b577376b3b4b9fa12a329225f1367d9786d63db52a606f8be` |

Environment Manifest SHA-256 values:

- `/private/tmp/pounce-julia017-env110/Manifest.toml`: `547855826951de5b22b7b9fc6e7c36c700ff7383e44e0279b91fdc2409863be9`.
- `/private/tmp/pounce-julia017-env/Manifest.toml`: `daf7593396cb46cc6b2dbf1b429b86e429c73cd01f92dbb0b83e21bf0d665beb`.
- `/private/tmp/pounce-diagnostics-minimum/Manifest.toml`: `c695c4f78da44e564bc40cf8bfc3b4544b252d388e942523bfaf1bee41dd1136`.
- `/private/tmp/pounce-bounds-parent-env/Manifest.toml`: `1593871bbe60aed73686f93167cfaf79abc3a0153e80ddc841cf2c8b4c3a234b`.


## Permanent review regressions — 2026-11-10

Promoted test revision: `c12bd77efd5d067f95bc72ff4bf2d23450687169`.
The implementation is unchanged. The suite now includes the 179 selected review
assertions: 175 for all 25 complementarity class transitions, and four for
explicit primal/lower-bound/upper-bound/row dual starts after a bound update.
The latter captures the Julia input buffers during the native callback and
checks the MOI dual-sign conversion. Native capability guards keep the older
library path valid. The remaining exploratory review checks stay in the
original temporary review script; they were not part of this promotion.

Validation on the clean test revision, using the same retained native binaries
and dependency environments listed above:

| Native | Julia | Scope | Result |
|---|---|---|---|
| Current `550b1d5e` | 1.10.13 | Full suite | 5,119 passed |
| Current `550b1d5e` | 1.13.3 | Full suite | 5,119 passed |
| Current, JuMP 1.29 / MOI 1.48 | 1.10.13 | Focused bounds | 377 passed |
| v0.15.0 | 1.10.13 | Focused bounds / capability fallback | 101 passed |
| v0.16.0 | 1.13.3 | Focused bounds / capability fallback | 101 passed |

The earlier full older-native runs remain attributed to `1baee4ed`; they were
not repeated for these capability-guarded test additions. Julia formatting and
diff checks passed. No native code or user-facing behavior changed, and no new
native build artifacts were generated.

| File under `/private/tmp/` | SHA-256 |
|---|---|
| `pounce-bounds-promotion-current110.log` | `5268427eeea1dcd651c176069559e3b731f13ddc4cf1d4a2baebbd6c0b5a2f64` |
| `pounce-bounds-promotion-current113.log` | `86e41928a88142c82fbcc40cc1c7388a8dbb386cec0076a280f88c71dc41f1ab` |
| `pounce-bounds-promotion-minimum.log` | `68c669c60bc0d2a1f3deb1e33ffd3c0c1c786a07dd8c7df2be8f262a861bdfce` |
| `pounce-bounds-promotion-v015.log` | `da8c570902dfc7025c5d9d189f77f6d65dde581943c3fc6af0ab07d63c22e74f` |
| `pounce-bounds-promotion-v016.log` | `4db9c428a5e1578dc292923ab0b07b979b0932c630c91798949021717ac8bb65` |


## John review follow-up — 2026-11-10

Tested code revision: `326db9982a200ad7517c8712d27ee82b3c68385e`, clean worktree.
Dependency #1163 is updated to `2f9e1c571cafe019bad0754cd9f763166d390c7e`
(tested implementation `f4c66b45c8a4429bea911153ecaf6a769f36fef6`). The original
four bound-update commits were rebased above that dependency. A duplicated
CHANGELOG entry from Git's automatic merge was removed; range-diff showed no
other unexpected change to those commits. #1164 remains draft pending #1163.

The shared fix makes crossed paired bounds return INVALID_MODEL before any
native batch is applied, with no stale result. This PR adds 86 permanent
assertions for direct/cached invalid attempts with simultaneous ordinary
variable/row changes, preserved pending sets, unchanged USER/EFFECTIVE bounds,
recovery to the same or a different class, fresh-model agreement, evaluator
retention and transient crossed edits repaired before solving. The focused
bounds suite now has 463 assertions. Index and value buffers use owned dense
conversions without splatting; range/view inputs exercise the public C calls.

Validation used the same retained native binaries and environments listed above:

| Native | Julia | Scope | Passed |
|---|---|---|---:|
| Current `550b1d5e` | 1.10.13 | Full suite | 5,446 |
| Current `550b1d5e` | 1.13.3 | Full suite | 5,446 |
| Current, JuMP 1.29 / MOI 1.48 | 1.10.13 | Complementarity and bounds | 1,225 |
| v0.15.0 | 1.10.13 | Loading/extension/complementarity/bounds fallback | 114 |
| v0.16.0 | 1.13.3 | Loading/extension/complementarity/bounds fallback | 114 |

Julia formatting, diff checks, documentation consistency and mdBook passed.
The earlier full older-native measurements retain their original revision;
the new checks above are focused compatibility runs. No native solver code,
ABI signature or struct layout changed. No new native build artifacts were
generated; retained libraries were reused.

| Log under `/private/tmp/` | SHA-256 |
|---|---|
| `pounce-john1164-full110.log` | `387a178eb87f33451d9ef342c42dd872ba80d18578c1b1163d23a0dd4fd52128` |
| `pounce-john1164-full113.log` | `ee85d65366f29626e462d73714f952eb2d710aad133523713e0e72497f9e64b9` |
| `pounce-john1164-min.log` | `3675f8b4559f4b89f896fb41d3d7bbf0236e43c7d367e58b73f163cdda272554` |
| `pounce-john1164-v015.log` | `6c99ce78d261eb90ebda1e786cf63644e0365b78a94a6b0f043e97ca379e5827` |
| `pounce-john1164-v016.log` | `96235f2db2e4bfb2fb569d252f7533678bbf423e41348ebc8acc63828113b05f` |
| `pounce-john1164-docs.log` | `7a10d85d1c1afc07e9f8f4fab47b447db1488cc3c727c13b30b7a50586570db7` |
