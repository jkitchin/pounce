# gh#1147 — C API beyond the NLP path, in stages (complementarity first)

Implementation plan for the agent loop on branch `claude/hopeful-bardeen-1k3nx9`.
One work item per iteration. This file is also the status board: tick the
checklist and append to the per-iteration log (§9) at the end of every
iteration, **in the same commit as the code**.

Written against `2ac0ad9` (main after 0.16.0, 2026-11-07). Every `path:line`
below was read on that commit; re-grep before relying on a line number, the
files move. Nothing here was measured at benchmark scale.

---

## 0. Progress checklist

Tick an item only when every command in its **Gate** passed on the commit
that ticks it.

### Stage 1 — declared complementarity through the C API

- [x] **S1-I1** Shared pair classifier in `pounce-nlp` (`validate_pairs`), `MpccTnlp::with_lift` uses it; every shape branch has a unit test; cc: sweep lines diff empty.
- [x] **S1-I2** Export-surface guard: `tests/export_surface.rs` + `header_abi.rs` prototype check; stale `POUNCE_VERSION` fixed; new book page `docs/src/c-api.md`; README completed for the *existing* surface.
- [x] **S1-I3** C compile-and-run test: `tests/c_examples_build_and_run.rs` builds and runs every `examples/*.c` against `pounce.h` and the cdylib.
- [x] **S1-I4** Declaration API: `AddPounceComplementarityPair`, `SetPounceComplementarityPairs`, `ClearPounceComplementarityPairs`, `GetPounceComplementarityPairCount`, `GetPounceComplementarityPairs`; every error code tested.
- [x] **S1-I5** `IpoptSolve` routes through `solve_mpcc` when pairs are declared; sessions and SQP working-set warm starts refuse pairs; live inspectors pinned.
- [x] **S1-I6** Outcome getters: `GetPounceComplementarityStats`, `GetPounceComplementaritySolve`, `GetPounceComplementarityText`; "driver did not run" behaviour tested.
- [x] **S1-I7** `IpoptWriteSolveReport` carries the `complementarity` block.
- [x] **S1-I8** CLI-parity test over `crates/pounce-cli/tests/fixtures_mpcc/` through the C API (`crates/pounce-cli/tests/cinterface_mpcc_parity.rs`).
- [x] **S1-I9** MOI-semantics helper `AddPounceComplementsConstraint`: all five table rows + error rows tested.
- [x] **S1-I10** C example `examples/complementarity.c`; docs (`complementarity.md`, `c-api.md`, README, CHANGELOG); Pounce.jl gate/ABI-review checklist written.
- [x] **S1-I11** Stage-1 closing gate: full fixture sweep (both legs, cc: lines) diffs empty against the baseline binary; `make check clippy fmt-check`; `make python-test`.
- [x] **S1-I12** `IpoptWriteSolveReport` carries the `second_opinion` block (gh#850) after a pairs-free `IpoptSolve` on which the ladder ran, under the CLI's and Python's `ran()` test and through `SecondOpinionOutcome::report_info()`; no solve-behaviour change, no new symbol or struct. Approved by the human after S1-I11 (§8.5). **Gate:** `cargo test -p pounce-cinterface` (every binary green, new `tests/second_opinion_report.rs`: rescued block equals Python's, every-rung-fails, converged, pairs, no leak; the rescued test fails without the change); `cargo test -p pounce-cli --test cinterface_mpcc_parity`; export guard (`export_surface`, `header_abi`); CI clippy flags on this crate (`--all-targets --no-deps -D clippy::correctness -D clippy::suspicious`); `cargo fmt --all -- --check`; `make book`; `make ask-check`; `make c-api-examples`. No sweep (the CLI does not link this crate; confirmed in §9).

**Stop here and report to the human before starting any later stage.**

### Stage 2 — in-place bounds/data update and warm re-solve

- [ ] **S2-BLOCKED** Blocked on confirmation from the Julia side (§6). Do not start.

### Stage 3 — matrix-form convex/conic entry

- [x] **S3-I1** Lift `solve_presolved` / `demote_unverified_optimal` from `pounce-py/src/qp.rs` into `pounce-convex` (pure move).
- [x] **S3-I2** `PounceQp*` handle, builder, solve and getters.
- [x] **S3-I3** Tests for every cone kind and every error path; parity against Python `solve_qp` / `solve_socp` anchors.
- [x] **S3-I4** Docs, README, CHANGELOG, `examples/qp_socp.c`; export guard green.

### Stage 4 — multistart over an NLP handle

- [ ] **S4-I1** `PounceFindMinima` + `PounceMinima` result handle and getters.
- [ ] **S4-I2** Parity against `pounce --minima` for one seed across all six methods; docs, README, CHANGELOG, example; export guard green.

---

## 1. Loop instructions

1. **Pick** the first unchecked item in §0, top to bottom. Never skip ahead;
   S1-I2 and S1-I3 are the validation mechanisms every later item is gated
   on, so they come before the first new symbol.
2. **Read first** the files the item's "Read first" line names. Do not work
   from this plan's quotations alone; they are a map, not the territory.
3. **Tests first or alongside.** Write the test the item names before or
   with the code. A test that passes before the code exists is not a test
   of the code (CLAUDE.md, "A test is only evidence about the branch it
   reaches"). For every rule that branches — a shape class, an error code,
   a status — the second fixture is the test.
4. **Run the Gate.** Every command listed under **Gate** must pass. Then run
   `cargo test -p pounce-cinterface` in full; it must stay green at every
   iteration regardless of what the item touched.
5. **Record.** Tick the box, append an entry to §9 (commit hash, what moved,
   anything measured), commit. One iteration, one commit (plus fix-up
   commits if a gate fails on CI). Commit messages end with the attribution
   lines the session reminder gives.
6. **On failure** of a gate: fix and re-run, at most twice. If the third run
   still fails, do not loosen a tolerance, do not delete a test, do not
   grandfather a symbol: record the failure in §9 with the exact output and
   **stop and ask the human**.
7. **Stop and ask the human** also when:
   - any line of `scripts/sweep-fixtures.sh` output moves against the
     baseline binary (including an engine-only or `cc=`-only move);
   - a parity test disagrees with the CLI and the cause is not an option the
     C side forgot to set;
   - an item would need to change `pounce_restoration::mpcc_driver` or
     `pounce_algorithm::mpcc` behaviour (not just factor out a function);
   - an ABI decision in §3 turns out not to be implementable as written;
   - Stage 1 is complete (S1-I11 ticked).
8. **Never** change the layout of an existing `#[repr(C)]` struct
   (`PounceLinearSolverStats`: Pounce.jl mirrors it field for field,
   `Pounce.jl/src/diagnostics.jl:81-95`), an existing prototype, or an
   existing `#define` value. ABI additions only.
9. **Never** bump the workspace version in a feature commit. The release
   skill owns that (`.claude/skills/release/SKILL.md`). The CHANGELOG
   `[Unreleased]` section is where Stage 1 is recorded.

---

## 2. Where things are (read on `2ac0ad9`)

### The C interface today

- Crate: `crates/pounce-cinterface/` — `Cargo.toml` (cdylib + rlib,
  `crate-type = ["lib", "cdylib"]`), `build.rs` (MA57 rpath only),
  `src/lib.rs` (4022 lines), `src/solver.rs` (sessions, 848), `src/fortran.rs`
  (F77 shim, 763), `include/pounce.h` (811), `include/IpoptReturnCodes.h`,
  `README.md`, `examples/sqp_warm_start.c`, 12 integration tests in `tests/`.
- Handle state: `IpoptProblemInfo`, `lib.rs:134-182` — holds `app:
  IpoptApplication`, dims, bound copies `x_l/x_u/g_l/g_u`, callbacks,
  `user_scaling`, `last_solve: Option<LastSolve>`, `pending_working_set`,
  `nonlinear_vars: Option<Vec<Index>>` (stored in the problem's own index
  style, `lib.rs:173-181`).
- `LastSolve`, `lib.rs:199-206`: `stats, status, linear_solver, final_x,
  final_lambda, final_obj`. This is what the post-solve getters and
  `IpoptWriteSolveReport` read.
- The one solve path: `IpoptSolve`, `lib.rs:618-834`. Builds a
  `CCallbackTnlp` bridge (`lib.rs:707-740`), wires restoration
  (`lib.rs:751-765`), `defer_end_verdict`, `optimize_tnlp` (`lib.rs:775`),
  then the second-opinion ladder (`lib.rs:792-802`), then copies the bridge's
  `final_*` into the caller's buffers (`lib.rs:813-830`). The bridge's
  `finalize_solution` (`lib.rs:2436-2454`) is what fills `final_*`.
- Sessions: `IpoptCreateSolver` consumes the problem (`solver.rs:76`);
  `IpoptSolverSolve` (`solver.rs:135-300`) runs `RustSolver::solve`, no
  ladder, keeps the KKT factor.
- Panic guard: `ffi_guard`, `lib.rs:112-117`. Every new entry point's body
  goes through it.
- Return codes: `include/IpoptReturnCodes.h` (`Invalid_Problem_Definition =
  -11`, `Invalid_Option = -12`, `Internal_Error = -199`); Rust mirror
  `pounce_nlp::return_codes::ApplicationReturnStatus`.
- Index style precedent: `IpoptSetNonlinearVariables`, `lib.rs:1816-1849` —
  indices arrive in the problem's `index_style` (0 = C, 1 = Fortran),
  validated before stored, stored as given, all-or-nothing on error.
- Struct-returning getter precedent: `PounceLinearSolverStats` /
  `GetPounceLinearSolverStats`, `pounce.h:424-462`, `lib.rs:1400-1500`:
  `#[repr(C)]`, fixed `char solver_name[32]`, NaN / -1 sentinels, returns
  `FALSE` leaving the struct untouched when there is no data.
- Version: `GetIpoptVersion` returns `CARGO_PKG_VERSION` parsed
  (`lib.rs:1167-1190`), i.e. **0.16.0** (`Cargo.toml:63`). The header's
  `POUNCE_VERSION_*` defines (`pounce.h:40-43`) say **0.1.0** and nothing
  tests them. They are stale.
- Existing ABI test: `tests/header_abi.rs` (110 lines) parses `typedef`
  lines out of `pounce.h` and compares widths of `Bool/Index/Number`. It
  does **not** look at prototypes. Baseline: `cargo test -p pounce-cinterface
  --test header_abi` → 2 passed on `2ac0ad9`.
- Nothing compiles `examples/sqp_warm_start.c` in CI. The only CI consumer
  of `pounce.h` from C is the CasADi plugin (`.github/workflows/ci.yml:737-745`,
  `casadi/Makefile:121`).
- CI runs the crate's tests through `cargo test --workspace --exclude
  pounce-hsl` (`ci.yml:107`); `make test` is the same (`Makefile:95-96`).
- Pounce.jl: version gate `v"0.15.0" <= version < v"0.17.0-0"`
  (`Pounce.jl/src/Pounce.jl:16-21`), required-symbol list
  (`Pounce.jl/src/Pounce.jl:24-48`), documented in `docs/src/julia.md:27-29`.
  Julia CI builds the cdylib and runs `Pkg.test()` (`.github/workflows/julia.yml:43-48`).
  No `Complements` handling exists in `Pounce.jl/ext/` today (grep empty).
- Fortran shim (`src/fortran.rs:1-30`) exposes exactly the eight upstream
  `ip*_` symbols; none of pounce's extensions are mirrored there. Precedent:
  extensions are C-only.

### The complementarity machinery

- Pair type: `pounce_nlp::mpcc::ComplementarityPair { row: usize, var: usize }`
  (0-based, `crates/pounce-nlp/src/mpcc.rs:91-97`).
- Lowering wrapper and the **only** shape validator:
  `MpccTnlp::with_lift(inner, pairs, lowering, lift) -> Result<Self, String>`,
  `mpcc.rs:366-457`. Validates in this order per pair `k`: out-of-range
  (`mpcc.rs:401-406`), duplicate variable (`:407`), duplicate row (`:410`),
  then classifies on `(x_l[var], x_u[var], g_l[row], g_u[row])`:
  - **one-sided**: `one_sided(xl,xu)` and `one_sided(gl,gu)` both `Some`
    (`:414-420`; `one_sided` at `:330-336`, `INF = 1e19` at `:161`);
  - **one-sided plus a plain bound**: row one-sided, variable boxed — the
    row's finite side picks the variable bound (`:425-434`);
  - **mixed**: `is_mixed(xl,xu,gl,gu)` = variable boxed with `xl < xu`, row
    free (`:338-343`, `:441-447`) — lowered to two product rows;
  - otherwise refused with the long message at `:448-457` (contains the
    substring `one-sided`, which `python/tests/test_complementarity.py::
    test_a_pair_it_cannot_lower_raises_with_the_reason` and
    `crates/pounce-nlp/src/mpcc.rs:2507` assert on).
  Note that a **fixed variable with a free row** (`xl == xu`) fails
  `is_mixed` and is refused today; the MOI table (§3.1) says it means
  "F free, no pair".
- Driver: `pounce_restoration::mpcc_driver::solve_mpcc(app: &mut
  IpoptApplication, inner: Rc<RefCell<dyn TNLP>>, pairs:
  Vec<ComplementarityPair>, report: &mut dyn FnMut(&str)) -> Result<MpccOutcome,
  String>` (`crates/pounce-restoration/src/mpcc_driver.rs:332-337`). Reads
  `complementarity_lift`, `constr_viol_tol`, `max_iter`, `mu_strategy_fallback`,
  `tol`, `obj_scaling_factor` from `app.options()`; snapshots and restores
  `TOUCHED_OPTIONS`; defers the inner `finalize_solution` and forwards the
  reported point once through `MpccTnlp::forward_finalize` (`pounce-nlp
  mpcc.rs:850-880`), so the C bridge's `finalize_solution` runs exactly once,
  with the reported point, as the CLI relies on. It forwards the
  intermediate callback (`mpcc.rs:1385-1394`).
- Outcome: `MpccOutcome` (`mpcc_driver.rs:90-137`): `status, statistics,
  accepted_by: Option<MpccStage>, fallback: Option<MpccStage>, reported:
  Reported, strategy: Strategy, solves: Vec<MpccSolveRecord>,
  total_iterations, x, max_abs_product, certificate: Option<Certificate>,
  b_check: Option<BCheck>, escapes, lifted, natural_residual: Option<Number>,
  polish: Option<PolishRecord>, polish_threshold`.
  `MpccSolveRecord` (`:51-72`): `stage, tau: Option<Number>, status,
  iterations, budget_iterations, max_abs_product, seconds`.
  `PolishRecord` (`:77-86`): `taken, residual_before, residual_after, reason:
  Option<String>`.
- The one conversion to the report block: `MpccOutcome::report_info() ->
  pounce_solve_report::ComplementarityInfo` (`mpcc_driver.rs:143-230`),
  shared by the CLI and Python so the two cannot differ. The C API must use
  it too.
- Enums the getters encode (all in `crates/pounce-algorithm/src/mpcc.rs`):
  `MpccStage` (`:58-80`, labels at `:83-91`: `exact product`, `Scholtes
  continuation`, `l1 penalty`, `branch NLP`, `branch polish`), `Reported`
  (`:464-478`), `Strategy` (`:550-554`, `name()` at `:568`), `StationarityClass`
  (`:643-660`: `S, W, None, NotSearched`), `Certificate` (`:664-`), `BVerdict`
  (`:1445`), `BCheck` (`:1460`), `describe_outcome` (`:480`).
- Report block: `pounce_solve_report::ComplementarityInfo` and friends
  (`crates/pounce-solve-report/src/lib.rs:116-260`), attached with
  `ReportBuilder::set_complementarity` (`:1017`). Schema text:
  `docs/src/schema/solve-report-v1.md:371`.
- How the CLI routes: pairs read from the `.nl` (`crates/pounce-cli/src/main.rs:576-580`,
  `:685-697`, forcing `ProblemClass::Nlp`), driver call at `:2028-2065`
  (`Err` → exit 2 with "cannot solve this model's complementarity
  constraints"), the ladder and the debugger stand down, `solve_stats =
  out.statistics` (`:2080-2083`). Presolve is declined with pairs (`:1839`);
  `--minima` refuses pairs (`:854-858`).
- How Python routes: `crates/pounce-py/src/problem.rs:88-90` (field),
  `:405-440` (driver call replaces solve + ladder; `release_end_verdict` /
  `print_end_verdict`), `:476-481` (`Err` → `ValueError`), `:518-628`
  (`info["complementarity"]` dict — the field list the C getters must
  match), `:1657-1659` (report block). `pounce.Solver`, `find_minima`,
  `solve_nlp_batch`, `solve_with_sens` refuse pairs
  (`docs/src/complementarity.md:510-520`); the WASM build refuses them
  (`crates/pounce-wasm/src/lib.rs:277-283`).
- Fixtures: `crates/pounce-cli/tests/fixtures_mpcc/` (**not** `tests/fixtures_mpcc/`
  as the issue says): `qpec_small.nl` (+`.sol`), `ralph1.nl`, `mixed_pair.nl`,
  `design_cent_3.nl` (reaches the lifted path at the default, its
  `SOURCE.md`), `infeasible_pair.nl` (+`.py`, `.col`; the verdict path),
  `opf_vv_lower_knee.nl` and `opf_vv_lower_knee_held.nl` (+`.col`, `SOURCE.md`,
  two LICENSE files). CLI tests over them: `crates/pounce-cli/tests/mpcc_complementarity.rs`
  (test names at `:71-435`). Python tests: `python/tests/test_complementarity.py`
  (defines `QpecSmall` inline at `:22-`, points `FIXTURES` at the CLI dir).
- `.nl` loading from Rust, for a C-callback harness: `pounce_nl::nl_reader::read_nl_file(&Path)
  -> Result<NlProblem, String>` (`nl_reader.rs:1166`); `NlProblem.complementarity:
  Vec<NlComplementarity { row, var, flags }>` 0-based (`:765-778`);
  `NlTnlp::try_new(prob) -> Result<NlTnlp, String>` (`:4953`), a `TNLP`.
  TNLP methods: `tnlp.rs:169-200` (`eval_f(&[Number], bool) -> Option<Number>`,
  `eval_grad_f(.., &mut [Number]) -> bool`, `eval_g`, `eval_jac_g(Option<&[Number]>,
  bool, SparsityRequest)`, `eval_h(..)`, `get_starting_point(StartingPoint)`).
- The sweep: `scripts/sweep-fixtures.sh` runs the six declared-pair fixtures
  as `cc:<name>` lines with a `cc=` field (`:128-148`, `FX_MPCC` at `:187`,
  naming at `:201`, column at `:287-300`). Any refactor touching
  `MpccTnlp::with_lift` reaches those lines.

### Volt-var / volt-watt models

The volt-var case is in-repo: `opf_vv_lower_knee.nl` is the two-bus
`vv_lower_knee` from MPCCModels.jl, written by Pyomo's `mpec.nl`
(`fixtures_mpcc/opf_vv_lower_knee.SOURCE.md`). The **volt-watt** models and
the rest of the 37-model OPF suite live **only in the benchmark data root**
(`$(POUNCE_BENCH_DATA)/mpcc_opf/`, `benchmarks/Makefile:558-570`, target
`mpcc-opf-run`), which is not in this repository and is not available to the
loop. **Substitute:** the parity test (S1-I8) uses the two in-repo OPF
fixtures; the plan records the data-root suite as a manual check the human
runs before merge (§8). Do not fabricate a volt-watt model.

---

## 3. Design decisions (Stage 1)

### 3.1 Native shapes are the primitive; MOI semantics is a separate helper

**Decision:** `AddPounceComplementarityPair` / `SetPounceComplementarityPairs`
declare pairs in **pounce's native shapes** (exactly what Python's
`Problem(complementarity=)` and the `.nl` reader accept), validated at
declaration time by the **same classifier** `MpccTnlp::with_lift` uses. A
second function, `AddPounceComplementsConstraint`, takes MOI
`[F, x] ∈ Complements` semantics and maps the five-row table onto native
shapes by editing the handle's own copy of the row bounds, returning a code
that says what it did.

Why both, and why in this order:

- The primitive must match Python and the `.nl` reader byte for byte, or the
  three frontends answer different questions on the same model (the
  `test_complementarity.py` docstring states this as the invariant). Native
  shapes are that contract; MOI semantics are a transcription of it.
- Putting the MOI table in Rust means it exists **once**, is tested per row
  here (the branch rule), and every frontend that wants MOI semantics gets the
  same answer. Pounce.jl stays out of scope but inherits a tested mapping.
- Validation at declaration needs the classifier outside `MpccTnlp`; the
  refactor (S1-I1) is reached by the sweep's `cc:` lines, which is the proof
  it changed nothing.

The MOI table, as the C helper applies it, on a row that is **free**
(`g_L ≤ -1e19`, `g_U ≥ 1e19`) — any other row is refused with
`POUNCE_COMPLEMENTS_ERR_ROW_NOT_FREE` ("use `AddPounceComplementarityPair`,
the row already has a side"):

| `x` bounds | action on the handle | pair declared | outcome code |
|---|---|---|---|
| `l < u` both finite | none | yes (native **mixed**) | `POUNCE_COMPLEMENTS_PAIR_MIXED` |
| `l` finite, `u` infinite | `g_L[row] = 0` (F ≥ 0) | yes (native one-sided, lower) | `POUNCE_COMPLEMENTS_PAIR_LOWER` |
| `u` finite, `l` infinite | `g_U[row] = 0` (F ≤ 0) | yes (native one-sided, upper) | `POUNCE_COMPLEMENTS_PAIR_UPPER` |
| both infinite | `g_L[row] = g_U[row] = 0` (F = 0) | **no** | `POUNCE_COMPLEMENTS_ROW_FIXED_NO_PAIR` |
| `l == u` finite | none (F free) | **no** | `POUNCE_COMPLEMENTS_VAR_FIXED_NO_PAIR` |

Errors (negative): `_ERR_NULL`, `_ERR_OUT_OF_RANGE`, `_ERR_ROW_NOT_FREE`,
`_ERR_VAR_ALREADY_PAIRED`, `_ERR_ROW_ALREADY_PAIRED`. Editing the bounds is
honest: `F ≥ 0` is a constraint of the MPCC, not a solver trick, and the
handle already owns its bound copies (`lib.rs:141-144`). It is documented
loudly, the outcome code says it happened, and `GetPounceComplementarityPairs`
shows what was declared.

### 3.2 Declaration API shape and indexing

Both single and bulk, as the issue suggests; bulk is **all-or-nothing** and
validated against the pairs already declared. Indices are in the problem's
`index_style`, as `IpoptSetNonlinearVariables` does (`lib.rs:1832`); stored
0-based internally. Error reporting is an `int` code, not a bare `Bool`,
because "which pair and why" is the whole point of validating early:

```c
#define POUNCE_CC_OK                   0
#define POUNCE_CC_ERR_NULL            -1   /* NULL handle, or NULL array with n_pairs > 0 */
#define POUNCE_CC_ERR_OUT_OF_RANGE    -2   /* row or var outside [0,m) / [0,n) (in index style) */
#define POUNCE_CC_ERR_VAR_ALREADY     -3   /* variable already in a pair (declared or in this batch) */
#define POUNCE_CC_ERR_ROW_ALREADY     -4   /* row already in a pair */
#define POUNCE_CC_ERR_SHAPE           -5   /* neither one-sided, one-sided+bound, nor mixed */
#define POUNCE_CC_ERR_COUNT           -6   /* n_pairs < 0 */

int AddPounceComplementarityPair(IpoptProblem, ipindex row, ipindex var);
int SetPounceComplementarityPairs(IpoptProblem, ipindex n_pairs,
                                  const ipindex* rows, const ipindex* vars,
                                  ipindex* bad_index /* may be NULL; -1 on success */);
Bool ClearPounceComplementarityPairs(IpoptProblem);
ipindex GetPounceComplementarityPairCount(IpoptProblem);          /* -1 on NULL */
Bool GetPounceComplementarityPairs(IpoptProblem, ipindex* rows, ipindex* vars); /* index style */
```

`SetPounceComplementarityPairs` **replaces** the declared set (so a frontend
can rebuild it), `Add` appends. Pairs persist across solves on the handle,
like `nonlinear_vars`. Shape validation uses the handle's **current** bound
copies, so it must be re-run inside `IpoptSolve` as well (the driver does it
anyway through `with_lift`; a mismatch there is reported as
`Invalid_Problem_Definition`, never a panic).

### 3.3 Sessions refuse pairs

`IpoptCreateSolver` returns `NULL` when the problem has declared pairs (and,
at `print_level ≥ 1`, prints one line saying why). Rationale: the session
holds the KKT factor of the *lowered* model (extra product rows, possibly
lifted variables), so `IpoptSolverParametricStep` / `ReducedHessian` over
the caller's row indices would answer a different system. Python's
`pounce.Solver` refuses pairs for the same reason (`docs/src/complementarity.md:516`).
`IpoptSolverSolve` keeps a defensive guard too (unreachable, tested as
unreachable by construction: the only way to a session is through
`IpoptCreateSolver`). Stage 2 may revisit.

### 3.4 Getter design: one `#[repr(C)]` stats struct, a per-solve record getter, and a text getter

Strings cross the ABI two ways in this header already: fixed `char[32]`
inside a struct (`PounceLinearSolverStats.solver_name`) and NUL-terminated
`const char*` *inputs*. Free-form text (the driver's `summary`, the polish
`reason`, the B-check `reason`) must not be a fixed array. So:

```c
#define POUNCE_CC_CLASS_S           0
#define POUNCE_CC_CLASS_W           1
#define POUNCE_CC_CLASS_NONE        2
#define POUNCE_CC_CLASS_NOT_SEARCHED 3
#define POUNCE_CC_CLASS_ABSENT     -1   /* no certificate: model not evaluable there */

#define POUNCE_CC_REPORTED_ANSWER               0
#define POUNCE_CC_REPORTED_WEAK_FALLBACK        1
#define POUNCE_CC_REPORTED_FEASIBLE_UNCERTIFIED 2
#define POUNCE_CC_REPORTED_VERDICT              3

#define POUNCE_CC_B_VERDICT_B           0
#define POUNCE_CC_B_VERDICT_NOT_B       1
#define POUNCE_CC_B_VERDICT_NOT_CHECKED 2
#define POUNCE_CC_B_VERDICT_ABSENT     -1   /* the check did not run */

typedef struct {
    char     strategy[16];       /* "both" | "exact" | "continuation" */
    ipindex  reported;           /* POUNCE_CC_REPORTED_* */
    char     accepted_by[32];    /* stage label, "" when none */
    char     fallback[32];       /* stage label, "" when none */
    ipindex  n_solves;
    ipindex  total_iterations;
    ipnumber max_abs_product;    /* NaN when unmeasured */
    ipindex  certificate_class;  /* POUNCE_CC_CLASS_* */
    ipindex  n_pairs, n_biactive, n_neither;
    ipindex  multipliers_unique; /* 0/1 */
    ipnumber residual, resid_tol;/* NaN when unmeasured */
    ipindex  tight, feasible;    /* 0/1 */
    ipnumber primal_violation;   /* NaN when unmeasured */
    ipindex  b_verdict;          /* POUNCE_CC_B_VERDICT_* */
    ipindex  b_n_biactive, b_branches_solved, b_enumerated;
    ipindex  escapes, lifted;
    ipnumber natural_residual;   /* NaN when unmeasured */
    ipindex  polish_ran, polish_taken;          /* 0/1; polish_taken meaningful iff polish_ran */
    ipnumber polish_residual_before, polish_residual_after, polish_threshold; /* NaN when unmeasured */
} PounceComplementarityStats;

Bool GetPounceComplementarityStats(IpoptProblem, PounceComplementarityStats*);
/* FALSE, struct untouched: NULL, no solve yet, or the driver did not run (no pairs). */

typedef struct {
    char     stage[32];          /* MpccStage::label() */
    ipindex  relaxed;            /* 1 when tau is meaningful */
    ipnumber tau;
    ipindex  status;             /* ApplicationReturnStatus */
    ipindex  iterations;
    ipnumber max_abs_product;    /* NaN when unmeasured */
    ipnumber seconds;
} PounceComplementaritySolve;

Bool GetPounceComplementaritySolve(IpoptProblem, ipindex k, PounceComplementaritySolve*);
/* k in [0, n_solves); FALSE otherwise. */

#define POUNCE_CC_TEXT_SUMMARY       0
#define POUNCE_CC_TEXT_POLISH_REASON 1
#define POUNCE_CC_TEXT_B_REASON      2
ipindex GetPounceComplementarityText(IpoptProblem, int which, char* buf, ipindex buflen);
/* snprintf contract: writes at most buflen-1 bytes plus NUL, returns the full
   length without NUL (so buf=NULL, buflen=0 sizes the buffer); -1 when there
   is no such text (no solve, driver did not run, or that text is absent). */
```

Every field maps to one `MpccOutcome` field through the same encodings
`report_info()` uses (`mpcc_driver.rs:155-229`); the test in S1-I7 asserts
the getter values against the JSON block the same solve writes, so the two
cannot drift.

### 3.5 Live inspectors and other options under a driver solve

`GetIpoptCurrentIterate` returns `FALSE` unless `n == info.n && m == info.m`
(`lib.rs:921`) and the copy refuses a length mismatch (`lib.rs:944-949`). A
driver solve runs on the lowered model (`m + n_product_rows`, plus lifted
variables), so today the inspector already returns `FALSE` during such a
solve. **Pin it** (test) and document it in `pounce.h` next to the
restoration paragraph (`pounce.h:229-231`): "during a complementarity solve
the inspectors report no data". Slicing the first `n`/`m` entries is a
possible follow-up, not Stage 1.

Also refuse, with `Invalid_Problem_Definition` and a `print_level`-gated
line: a pending SQP working set (`pending_working_set.is_some()`) together
with pairs (the working set has the caller's dimensions; the lowered model
has more rows). `IpoptGetWorkingSet` returns `FALSE` after a driver solve.
For `SetIpoptProblemScaling` and `IpoptSetNonlinearVariables` under pairs:
read `pounce-nlp/src/mpcc.rs` for whether `MpccTnlp` forwards
`get_scaling_parameters` / `get_list_of_nonlinear_variables`; pin whichever
behaviour exists with a test and state it in the header. Do not change the
wrapper.

### 3.6 Fortran: deferred, explicitly

No pounce extension is mirrored in `src/fortran.rs` (only the eight upstream
`ip*_` symbols). Stage 1 adds none. `README.md` gets a sentence saying the
Fortran shim is the upstream surface only; the export guard excludes
`fortran.rs` symbols from the header check and requires them in the README's
Fortran paragraph instead.

### 3.7 Versioning

- Stage 1 ships in the next minor (**0.17.0**). The feature PR does not bump
  `Cargo.toml`; it adds to `CHANGELOG.md` `[Unreleased]` under `### Added`.
- Fix the header: `POUNCE_VERSION_*` must equal the crate version. S1-I2 adds
  a test in `header_abi.rs` comparing the defines to `env!("CARGO_PKG_VERSION")`,
  and a check in `scripts/check-release-consistency.sh` so the release guard
  catches drift (`.claude/skills/release/SKILL.md:13-17` describes that guard).
- Pounce.jl: widening `_check_native_version` to `< v"0.18.0-0"` is an
  ABI-review action (S1-I10 writes the checklist; the human performs the
  review). The Julia tests must still pass against the 0.17 cdylib without
  any Julia code change, which is what "ABI additions only" promises.

---

## 4. Stage 1 — iterations

Each iteration: **Touch** (files), **Internals** (Rust/C), **Tests**, **Gate**,
**Done when**.

### S1-I1 — Shared pair classifier in `pounce-nlp`

Read first: `crates/pounce-nlp/src/mpcc.rs:1-135, 325-470, 2000-2130, 2490-2510`;
`python/tests/test_complementarity.py:161-170`; `scripts/sweep-fixtures.sh:128-148`.

Touch: `crates/pounce-nlp/src/mpcc.rs` only.

Internals:
```rust
/// How a declared pair is lowered; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PairShape {
    /// `(g_side, h_side)` as `(sign, bound)`.
    OneSided { g: (Number, Number), h: (Number, Number) },
    /// Row one-sided, variable boxed: the row's side names the bound.
    OneSidedPlusBound { g: (Number, Number), h: (Number, Number) },
    /// Variable boxed `l < u`, row free: two product rows.
    Mixed { l: Number, u: Number },
}
#[derive(Debug, Clone, PartialEq)]
pub enum PairErrorKind { OutOfRange, VarAlreadyPaired, RowAlreadyPaired, UnsupportedShape }
#[derive(Debug, Clone, PartialEq)]
pub struct PairError { pub pair: usize, pub kind: PairErrorKind, pub message: String }
impl std::fmt::Display for PairError { /* prints `message`, byte for byte today's text */ }

pub fn classify_pair(xl: Number, xu: Number, gl: Number, gu: Number) -> Option<PairShape>;
pub fn validate_pairs(n: usize, m: usize, x_l: &[Number], x_u: &[Number],
                      g_l: &[Number], g_u: &[Number], pairs: &[ComplementarityPair])
    -> Result<Vec<PairShape>, PairError>;
```
`with_lift` calls `validate_pairs` and builds `lowered/g_side/h_side/signless`
from the returned shapes; its `Err(String)` is `e.to_string()`, so every
existing message (including the `one-sided` substring at `:448-457` and the
`pair {k}: ...` prefixes) is unchanged. `INF` stays `1e19`.

Tests (`mpcc.rs` `#[cfg(test)]`): one test per branch of `classify_pair`
(one-sided lower/lower, upper/upper, lower/upper, upper/lower; plus-bound
with row lower and with row upper; mixed) and per refusal (boxed row; both
free; fixed variable with free row — asserts it is refused *today*, S1-I9
relies on that; out of range var; out of range row; duplicate var; duplicate
row), each asserting the `PairErrorKind` **and** that `Display` equals the
pre-refactor text. Existing tests at `:2085-2130`, `:2433-2510` keep passing
unchanged.

Gate:
```
cargo test -p pounce-nlp
cargo test -p pounce-restoration
cargo test -p pounce-cli --test mpcc_complementarity
make python-ext && python -m pytest python/tests/test_complementarity.py -q
# baseline binary built from 2ac0ad9 in a separate worktree with its own CARGO_TARGET_DIR:
scripts/sweep-fixtures.sh <baseline-pounce> > /tmp/sweep-base.txt
scripts/sweep-fixtures.sh target/release/pounce > /tmp/sweep-head.txt
diff /tmp/sweep-base.txt /tmp/sweep-head.txt   # must be empty, both legs, cc: lines included
```
Done when: all green, sweep diff empty, §9 entry records the sweep command
and the two binaries' commits.

### S1-I2 — Export-surface guard and header prototype check

Read first: `tests/header_abi.rs`; `include/pounce.h` whole; `README.md`;
`docs/src/SUMMARY.md`; `scripts/check-release-consistency.sh:160-190`.

Touch: `crates/pounce-cinterface/tests/export_surface.rs` (new),
`tests/header_abi.rs`, `include/pounce.h` (version defines, nothing else),
`README.md`, `docs/src/c-api.md` (new), `docs/src/SUMMARY.md` (add
`- [C API](c-api.md)` under Integrations, after Julia),
`scripts/check-release-consistency.sh`.

Internals — `tests/export_surface.rs` is pure text analysis over the crate's
own files (`include_str!` / `std::fs` relative to `CARGO_MANIFEST_DIR`):

1. **Exports**: scan `src/lib.rs` and `src/solver.rs` for
   `#[unsafe(no_mangle)]` followed (within 3 lines, attributes allowed) by
   `pub unsafe extern "C" fn NAME(`. Collect `NAME` and the parameter list up
   to the matching `)` and the `-> RET`. `src/fortran.rs` symbols are
   collected separately.
2. **Header prototypes**: strip `/* */` and `//` comments from `pounce.h`,
   then match `RET NAME(ARGS);` at top level (not inside `typedef`). Assert
   the two name sets are **equal** (no undeclared export, no phantom
   prototype).
3. **Signature agreement** per name: same arity; each parameter and the
   return mapped to a width class `{ i1, i32, f64, ptr, fnptr, void }` on
   both sides and compared positionally. C map: `bool/Bool → i1`;
   `int/ipindex/Index/IpoptBoundStatus/IpoptConsStatus/enum X → i32`;
   `ipnumber/Number/double → f64`; anything with `*` or an opaque handle
   typedef (`IpoptProblem`, `IpoptSolver`, `UserDataPtr`) → `ptr`;
   `*_CB` typedef → `fnptr`; `void → void`. Rust map: `Bool → i1`;
   `Index/c_int/IpoptBoundStatus/IpoptConsStatus → i32`; `Number/f64 → f64`;
   `*const/*mut` and `IpoptProblem`/`IpoptSolver` → `ptr`;
   `Option<Eval_*_CB>/Option<Intermediate_CB> → fnptr`; no return → `void`.
   This is what `header_abi.rs` cannot see today.
4. **Docs**: every exported name appears in backticks in `README.md` and in
   `docs/src/c-api.md`. Fortran names must appear in README's Fortran
   paragraph.
5. **Tests**: every exported name appears as `NAME(` in at least one file
   under `tests/` or `examples/`.
6. **Error paths**: every exported name has a line `// error-path: NAME`
   somewhere under `tests/`, placed in the test that exercises its failure
   branch (NULL handle, bad index, no prior solve…). A frozen
   `GRANDFATHERED_NO_ERROR_PATH: &[&str]` list may name pre-existing
   symbols that have no such test yet; the test asserts every entry still
   exists as an export (so the list can only shrink) and the list is not
   added to after this commit (comment says so; reviewers enforce it).
   Prefer writing the missing markers now where a test already exists.
7. **Version**: `header_abi.rs` gains `header_version_defines_match_the_crate`
   parsing `POUNCE_VERSION_MAJOR/MINOR/PATCH` and `POUNCE_VERSION` against
   `env!("CARGO_PKG_VERSION")`. Fix `pounce.h:40-43` to `0 16 0 / "0.16.0"`.
   Add the same check to `scripts/check-release-consistency.sh` next to the
   Dockerfile check (`:175-182`).

`docs/src/c-api.md`: one page, sections *Build and link*, *Lifecycle and
options*, *Solving*, *Post-solve statistics* (every `Get*`), *Working-set warm
start*, *Sessions* (link `sessions.md`), *Solve report* (link `json-output.md`),
*Fortran shim*, *ABI and versioning* (the `GetIpoptVersion` contract, the
Pounce.jl gate, "additions only"). It must name every exported symbol; the
guard enforces that. Keep prose short; the header's own comments are the
reference.

Gate:
```
cargo test -p pounce-cinterface --test export_surface --test header_abi
cargo test -p pounce-cinterface
scripts/check-release-consistency.sh
make book     # mdbook builds; the new page is in SUMMARY
```
Done when: the guard passes on the *existing* surface with no new symbols,
and the grandfather list is as short as the existing tests allow.

### S1-I3 — C programs compile and run against the header

Read first: `examples/sqp_warm_start.c`; `casadi/Makefile:100-130`
(flags used by the one CI consumer today); `Makefile:60-70` (`CDYLIB_NAME`).

Touch: `crates/pounce-cinterface/tests/c_examples_build_and_run.rs` (new),
`Makefile` (new target `c-api-examples`), `README.md` (how to build the
examples), `.github/workflows/ci.yml` only if `cc` turns out to be missing
on the runner (it is present on ubuntu-latest; verify by reading the job's
image).

Internals: for every `examples/*.c`, run
`cc -std=c99 -Wall -Wextra -Werror -pedantic -I <manifest>/include <file> -o <tmp>
-L <libdir> -lpounce_cinterface -Wl,-rpath,<libdir>` where `<libdir>` is the
directory of the cdylib: `std::env::current_exe()` → `target/<profile>/deps/`,
so `<libdir>` = its parent. Assert the file `libpounce_cinterface.{so,dylib}`
exists there (cargo builds all crate types of the lib target for integration
tests; if it does not on some toolchain, the test must say so rather than
skip). Run the binary with `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` set to
`<libdir>`, assert exit 0, and assert a per-example stdout marker (for
`sqp_warm_start.c`: the four solves succeed; read the file for the exact
line). `-Werror -pedantic` is deliberate: the header must be clean C99.
If `cc` is not on `PATH`: fail unless `POUNCE_C_EXAMPLES_SKIP=1` is set, and
document that variable in the test's module docs.

Tests: the test itself, plus a negative unit test of the C-flag assembly
(pure function) if one is factored out.

Gate:
```
cargo test -p pounce-cinterface --test c_examples_build_and_run
make c-api-examples
cargo test -p pounce-cinterface
```
Done when: `sqp_warm_start.c` builds with `-Werror -pedantic` and runs green
from `cargo test`.

### S1-I4 — Declaration API

Read first: `lib.rs:134-182, 1816-1884`; §3.2; S1-I1's `validate_pairs`.

Touch: `src/lib.rs`, `include/pounce.h`, `README.md`, `docs/src/c-api.md`,
`docs/src/complementarity.md` (new subsection "C API"), `CHANGELOG.md`,
`tests/complementarity_declare.rs` (new).

Internals:
- `IpoptProblemInfo.complementarity: Vec<ComplementarityPair>` (0-based).
- `fn declare_pairs(info: &mut IpoptProblemInfo, new: &[(Index, Index)], replace: bool)
  -> Result<(), (usize, PairErrorKind)>` — translate by `index_style`, check
  range, build the candidate set (existing + new, or new alone when
  `replace`), call `pounce_nlp::mpcc::validate_pairs(n, m, &x_l, &x_u, &g_l,
  &g_u, &candidate)`, store only on `Ok`. Map `PairErrorKind` → the
  `POUNCE_CC_ERR_*` codes. Out-of-range is checked before translation
  overflow (`p - offset` on `i32::MIN` must not panic: use `checked_sub`).
- Five entry points as in §3.2, all through `ffi_guard`.
- Constants `POUNCE_CC_*` as `pub const` in Rust and `#define` in the
  header; `export_surface.rs` does not check defines, so add a small test in
  `header_abi.rs` that parses `#define POUNCE_CC_` lines and compares values
  with the Rust constants (same pattern as the typedef parser).

Tests (`tests/complementarity_declare.rs`, with inline C-style callbacks like
`tests/nonlinear_variables_mask.rs:17-70`): the `QpecSmall` model from
`python/tests/test_complementarity.py:22-` written as Rust `extern "C"`
callbacks (reuse it in later iterations: put it in `tests/common/qpec.rs`
with `#[path]` includes). Assert: single add OK; count and get-back in
index style 0 and 1; `Set` replaces; `Add` after `Set` appends; each error
code (NULL handle; `n_pairs < 0`; NULL arrays with `n_pairs > 0`; var out of
range; row out of range; duplicate var across two `Add`s; duplicate row
inside one `Set` batch with `bad_index` reporting the second; shape refusal
on a boxed row; shape refusal on both-free) **and that a failed call leaves
the previously declared set unchanged**; `Clear` empties. Each error test
carries `// error-path: <Name>` for the function it exercises. No solve yet.

Gate:
```
cargo test -p pounce-cinterface
cargo clippy -p pounce-cinterface --all-targets -- -D warnings
```
Done when: export guard green with the five new names (header, README, book,
tests, error-path markers), header define test green.

### S1-I5 — Route `IpoptSolve` through the driver

Read first: `lib.rs:618-834`; `solver.rs:76-135`; `mpcc_driver.rs:1-28,
332-470`; `pounce-py/src/problem.rs:405-440, 476-481`; `cli/main.rs:2028-2083`;
`pounce_algorithm::second_opinion::narration_is_wanted`.

Touch: `src/lib.rs`, `src/solver.rs`, `include/pounce.h` (doc text on
`IpoptSolve`, `IpoptCreateSolver`, `GetIpoptCurrentIterate`,
`IpoptSetWarmStartWorkingSet`), `tests/complementarity_solve.rs` (new),
`CHANGELOG.md`.

Internals:
- `LastSolve.mpcc: Option<pounce_restoration::mpcc_driver::MpccOutcome>`
  (`None` on every non-driver solve; `Default` stays `None`).
- In `IpoptSolve`, after restoration wiring and `defer_end_verdict`, branch:
  ```rust
  if !info.complementarity.is_empty() {
      if info.pending_working_set.is_some() { /* print_level-gated line; */ return InvalidProblemDefinition }
      let narrate = narration_is_wanted(info.app.options());
      match solve_mpcc(&mut info.app, bridge.clone() as Rc<RefCell<dyn TNLP>>,
                       info.complementarity.clone(), &mut |l| if narrate { eprintln!("{l}") }) {
          Ok(out) => { if narrate { eprintln!("{}", describe_outcome(..)) }
                       if info.app.release_end_verdict() { info.app.print_end_verdict(out.status) }
                       status = out.status; stats = out.statistics.clone(); mpcc = Some(out) }
          Err(e)  => { if narrate { eprintln!("pounce: cannot solve this model's complementarity constraints: {e}") }
                       return InvalidProblemDefinition as Index }
      }
  } else { /* today's optimize_tnlp + ladder */ }
  ```
  Verify by reading `IpoptApplication` that `release_end_verdict` /
  `print_end_verdict` exist with those names (`pounce-py/src/problem.rs:425-427`
  uses them). The ladder does **not** run after the driver, as in the CLI
  and Python. Copy-back of `x/g/obj/mult_*` is unchanged: the driver's
  `forward_finalize` has filled the bridge's `final_*`.
- `IpoptCreateSolver`: return `NULL` when `(*prob_handle).complementarity`
  is non-empty, after a `print_level`-gated line. Do **not** null the
  caller's handle in that case (they still own it). `IpoptSolverSolve`
  keeps a guard returning `Invalid_Problem_Definition`.
- `IpoptGetWorkingSet` returns `FALSE` when `last_solve.mpcc.is_some()`.

Tests (`tests/complementarity_solve.rs`):
- QpecSmall with pairs `(0,1),(1,2)`: `IpoptSolve` returns
  `Solve_Succeeded`, `x ≈ (1,1,0)` within `1e-6` (the CLI test's tolerance,
  `mpcc_complementarity.rs:81-84`), `g` buffer filled, `GetIpoptIterCount`
  equals the reported solve's count (not the total), and
  `IpoptWriteSolveReport` succeeds (block content is S1-I7's test).
- Same model, pairs **cleared** before the solve: plain NLP, the answer
  differs from `(1,1,0)` or the status differs — whichever the model gives;
  assert the driver did not run (`GetPounceComplementarityStats` is S1-I6;
  here assert via the report having no `complementarity` key).
- A refused shape that passes declaration but fails in `with_lift` is not
  constructible (same validator); instead cover the `Err` arm with a bridge
  whose `get_nlp_info` cannot fail… it cannot from C. Mark the arm
  `// unreachable from C: validation ran at declaration` and leave it
  guarded; the §9 entry says so.
- `IpoptCreateSolver` on a problem with pairs → `NULL`, handle still valid
  (then `FreeIpoptProblem` it). `// error-path: IpoptCreateSolver`.
- Pending working set + pairs → `Invalid_Problem_Definition`, no callback
  invoked (count calls in `user_data`).
- Intermediate callback fires during the driver's lowered solves
  (`alg_mod = 0`), and inside it `GetIpoptCurrentIterate(n, m)` returns
  `FALSE` (§3.5). Returning `false` from the callback ends the solve with
  `User_Requested_Stop` (the driver's `stopped` path,
  `mpcc_driver.rs:444-448`).
- `SetIpoptProblemScaling` / `IpoptSetNonlinearVariables` with pairs: pin
  the observed behaviour (§3.5) with one test each and a header sentence.

Gate:
```
cargo test -p pounce-cinterface
cargo clippy -p pounce-cinterface --all-targets -- -D warnings
```
Done when: the QpecSmall answer matches the CLI's within `1e-6` and every
refusal above is tested.

### S1-I6 — Outcome getters

Read first: §3.4; `mpcc_driver.rs:139-230`; `pounce-py/src/problem.rs:518-628`;
`lib.rs:1400-1500` (the stats-struct precedent); `Pounce.jl/src/diagnostics.jl:81-95`
(how a struct is mirrored downstream — a reason to never reorder fields).

Touch: `src/lib.rs`, `include/pounce.h`, `README.md`, `docs/src/c-api.md`,
`tests/complementarity_getters.rs` (new), `tests/header_abi.rs` (defines).

Internals: `#[repr(C)] pub struct PounceComplementarityStats { .. }` and
`PounceComplementaritySolve` exactly as §3.4, field order frozen by a test
that checks `size_of` and `offset_of!` of every field against the C layout
computed by the C compiler in S1-I3's harness (compile a tiny C file that
prints `sizeof` and `offsetof` for each field; compare). A helper
`fn fill_stats(out: &MpccOutcome) -> PounceComplementarityStats` with the
encodings copied from `report_info()`; NaN for every `Option::None` /
non-finite number; `copy_label(&mut [c_char; N], &str)` truncates and
NUL-terminates. `GetPounceComplementarityText` implements the snprintf
contract (§3.4), returning `-1` for absent text.

Tests: on QpecSmall: `GetPounceComplementarityStats` → `TRUE`;
`strategy == "both"`, `reported == ANSWER`, `certificate_class == S`,
`n_solves > 0`, `total_iterations > GetIpoptIterCount`, `polish_ran == 1 &&
polish_taken == 1` and `natural_residual <= 1e-7` (the facts
`test_the_solve_report_carries_the_complementarity_block` asserts,
`python/tests/test_complementarity.py:272-286`); every `k` in
`[0, n_solves)` yields a record with a known stage label, `k == n_solves` →
`FALSE`; text getter sizes then fills `SUMMARY`, equals
`describe_outcome(..)`; `POLISH_REASON` is `-1` when polish was taken (reason
absent). Before any solve, and after a pair-less solve: stats → `FALSE`
struct untouched (fill it with a sentinel pattern first), solve getter →
`FALSE`, text → `-1`. `// error-path:` markers for all three. Under
`complementarity_strategy=exact`, `strategy == "exact"` (the second branch
of that field).

Gate: `cargo test -p pounce-cinterface`; clippy; export guard.
Done when: every field of the struct is asserted by at least one test
against a value derived independently (the JSON block of S1-I7 closes the
loop).

### S1-I7 — `IpoptWriteSolveReport` carries the block

Read first: `lib.rs:2500-2562`; `pounce-solve-report/src/lib.rs:100-260,
1010-1020, 1352-1452`; `docs/src/schema/solve-report-v1.md:371-`.

Touch: `src/lib.rs` (one `if let Some(out) = &last.mpcc {
builder.set_complementarity(out.report_info()) }`), `tests/complementarity_report.rs`
(new; `serde_json` is already a dev-dependency), `docs/src/c-api.md`.

Tests: QpecSmall solve → report at `"full"`; assert `report["complementarity"]`
exists and that `strategy, reported, accepted_by, total_iterations,
len(solves), certificate.class, natural_residual, polish.taken,
polish_threshold, escapes, lifted` equal the values the S1-I6 getters return
on the same handle (encode the ints back to the strings the schema uses).
Pair-less solve → no `complementarity` key. Report at `"summary"` carries the
block too (it is not detail-gated in the builder; verify by reading
`ReportBuilder::finish`).

Gate: `cargo test -p pounce-cinterface`; export guard.
Done when: getters and JSON agree on one solve, and a no-pairs report is clean.

### S1-I8 — CLI parity over the fixture corpus

Read first: `crates/pounce-cli/tests/mpcc_complementarity.rs:1-70, 135-185`;
`pounce_nl::nl_reader::{read_nl_file, NlTnlp::try_new}`; `tnlp.rs:169-200`;
`crates/pounce-cli/Cargo.toml` `[dev-dependencies]`; root `Cargo.toml:77-78`.

Touch: root `Cargo.toml` (`[workspace.dependencies] pounce-cinterface = { path
= "crates/pounce-cinterface", version = "0.16.0" }`), `crates/pounce-cli/Cargo.toml`
(`[dev-dependencies] pounce-cinterface.workspace = true`), new
`crates/pounce-cli/tests/cinterface_mpcc_parity.rs`, new helper
`crates/pounce-cli/tests/common/nl_c_harness.rs`.

Why in `pounce-cli`: `CARGO_BIN_EXE_pounce` is only available to that
crate's integration tests, and the CLI binary is the oracle the issue names.
`pounce-cinterface` does not depend on `pounce-cli`, so there is no cycle.

Internals — the harness: `struct NlHarness { tnlp: NlTnlp, n, m, nnz_jac,
nnz_hess }` behind `user_data`; `extern "C"` trampolines `ev_f/ev_grad_f/ev_g/
ev_jac_g/ev_h` that build slices and call the `TNLP` methods
(`SparsityRequest::Structure` when `values.is_null()`, `::Values` otherwise;
Hessian likewise). `NlTnlp` reports its index style and bounds through
`get_nlp_info`/`get_bounds_info`; pass them to `CreateIpoptProblem`; take
`x0` from `get_starting_point`. Declare `prob.complementarity` through
`SetPounceComplementarityPairs`. `AddIpoptIntOption("print_level", 0)`.
Then `IpoptSolve` + `IpoptWriteSolveReport(path, "full")`.

Oracle: `Command::new(env!("CARGO_BIN_EXE_pounce")).arg(fixture).arg("--no-sol")
.arg("--json-output").arg(p)`, as the CLI test does (`mpcc_complementarity.rs:28-58`).

Compare, per fixture and per option set: `solution.status`,
`solution.objective` (`1e-8` relative), `solution.x` (`1e-6` abs),
`complementarity.{strategy, reported, accepted_by, fallback, total_iterations,
escapes, lifted, certificate.class}`, `len(solves)` and each solve's
`{stage, status, iterations}`. Iteration counts are **exact**: same driver,
same model, same options. If they differ, the C side is missing an option
the CLI sets (read `cli/main.rs` for every `read_from_str` and every option
the `.nl` path applies, e.g. maximize handling, `nl_scaling`) — fix the
harness, do not loosen. If they still differ, stop and ask (§1.7).

Fixtures × option sets:
- `qpec_small.nl`, `ralph1.nl` (W point: the B-check branch,
  `test_the_solve_report_carries_the_b_stationarity_check`), `mixed_pair.nl`
  (mixed shape), `design_cent_3.nl` (lift path, `lifted > 0`),
  `infeasible_pair.nl` (`reported == "verdict"`), `opf_vv_lower_knee.nl`
  (polish taken), `opf_vv_lower_knee_held.nl` (polish not taken, reason
  present) — at the default;
- `qpec_small.nl` under `complementarity_strategy=exact`,
  `=continuation`, `complementarity_lift=yes`, `complementarity_lift=no`,
  and `hessian_approximation=limited-memory` (the lbfgs leg; this is where
  `auto` does not lift — `mpcc_driver.rs:345-361`).
Each option set is one `#[test]` so a failure names its branch.

Gate:
```
cargo test -p pounce-cli --test cinterface_mpcc_parity
cargo test -p pounce-cli --test mpcc_complementarity
cargo test -p pounce-cinterface
```
Done when: every row above is exact on status, class and iteration counts,
and within tolerance on objective and point.

### S1-I9 — MOI-semantics helper

Read first: §3.1; the issue's MOI table; `MathOptInterface` docs for
`Complements` only to confirm the five cases (no code from there).

Touch: `src/lib.rs`, `include/pounce.h`, `docs/src/complementarity.md` (the
table under a "From MOI / JuMP `Complements`" heading), `docs/src/c-api.md`,
`README.md`, `CHANGELOG.md`, `tests/complementarity_moi.rs` (new),
`tests/header_abi.rs` (defines).

Internals: `int AddPounceComplementsConstraint(IpoptProblem, ipindex row,
ipindex var)` → codes `POUNCE_COMPLEMENTS_*` (§3.1). Steps: NULL/range
checks; row must be free (`g_l <= -1e19 && g_u >= 1e19`, the wrapper's `INF`)
else `_ERR_ROW_NOT_FREE`; var/row not already paired else the `_ERR_*_ALREADY_PAIRED`
codes; then the table: edit `info.g_l[row]` / `info.g_u[row]` on the handle's
copy and call `declare_pairs` for the rows that declare one (which must
succeed by construction: the edited bounds are exactly a native shape —
assert it with a `debug_assert!` **and** a test, never `unwrap`).

Tests — one `#[test]` per table row, each on a small hand-solvable model
built from C callbacks so the answer is known independently of pounce:
1. `l < u`, F free → `PAIR_MIXED`; count 1; solve reaches the known answer;
   `GetPounceComplementarityPairs` returns `(row, var)`.
2. `x ≥ l` only → `PAIR_LOWER`; afterwards the handle's `g_L[row] == 0`
   (observable: `GetIpoptCurrentViolations`? no — expose nothing new; assert
   via the solve answer and via the JSON report's problem block if it lists
   bounds, else via a second declaration of the same row now failing with
   `_ERR_ROW_NOT_FREE`, which proves the edit happened).
3. `x ≤ u` only → `PAIR_UPPER`; same proof pattern.
4. `x` free → `ROW_FIXED_NO_PAIR`; pair count unchanged; the solve honours
   `F = 0` (choose a model where `F = 0` moves the optimum).
5. `l == u` → `VAR_FIXED_NO_PAIR`; pair count unchanged; `F` stays free
   (model where a bound on `F` would move the optimum; assert it did not).
Plus the three error rows: row not free; var already paired; row already
paired; and NULL / out-of-range. `// error-path: AddPounceComplementsConstraint`.

Gate: `cargo test -p pounce-cinterface`; clippy; export guard; header defines.
Done when: five rows and all error rows are each a passing test with its own
model.

### S1-I10 — C example, docs, ABI-review checklist

Read first: `examples/sqp_warm_start.c`; `docs/src/julia.md:1-40`;
`Pounce.jl/src/Pounce.jl:14-48`; `Pounce.jl/README.md:160-170`.

Touch: `examples/complementarity.c` (new: QpecSmall in C, declares the two
pairs, solves, prints the certificate class and the summary text; exits
non-zero unless class is `S` and `x ≈ (1,1,0)`), `README.md` (surface list
complete, Fortran sentence §3.6), `docs/src/c-api.md` (complementarity
section with the header snippets), `docs/src/complementarity.md` ("Limits"
list: sessions and SQP working sets refuse pairs on the C API; inspectors
report no data), `docs/src/julia.md:27-29` (note that 0.17 adds symbols and
the gate needs widening after review), `CHANGELOG.md`, and a new section
§8 of this file filled in with the ABI-review checklist for the human.

Tests: S1-I3's harness picks the new example up automatically; add its
stdout marker there. `make -C casadi test` is not required (the plugin does
not use the new symbols), but `cargo build --release -p pounce-cinterface`
must still produce the cdylib the CasADi job links.

Gate:
```
cargo test -p pounce-cinterface
make book
scripts/check-release-consistency.sh
```
Done when: the example runs from `cargo test`, every doc target names every
symbol (guard), CHANGELOG lists every new function under `[Unreleased] ###
Added`.

### S1-I11 — Stage-1 closing gate

No new code. Run, on the final Stage-1 commit:
```
cargo build --release
scripts/sweep-fixtures.sh <baseline-pounce-from-2ac0ad9> > /tmp/sweep-base.txt
scripts/sweep-fixtures.sh target/release/pounce > /tmp/sweep-head.txt
diff /tmp/sweep-base.txt /tmp/sweep-head.txt        # empty: both legs, every cc: line
make check && make clippy && make fmt-check
cargo test --workspace --exclude pounce-hsl
make python-test                                    # pounce-nlp was touched
```
The only non-cinterface crate Stage 1 edits is `pounce-nlp` (S1-I1), so the
sweep diff must be **empty**, engine column and `cc=` column included; an
empty diff is the proof the NLP path and the driver did not move. Record the
two binaries' commits and the diff result in §9, then **stop and report** to
the human with: the ABI additions (symbol list), the parity table from
S1-I8, the §8 checklist, and the data-root suites that still need a manual
run.

---

## 5. Validation mechanism summary (what proves completeness and correctness)

| Claim | Mechanism | Where |
|---|---|---|
| Every exported symbol is declared in `pounce.h`, and vice versa | name-set equality over `#[unsafe(no_mangle)]` scan vs header prototypes | `tests/export_surface.rs` (S1-I2) |
| Header prototypes match Rust signatures | arity + positional width-class comparison | `tests/export_surface.rs` (S1-I2) |
| Scalar typedefs match | existing | `tests/header_abi.rs` |
| `#define` constants match Rust constants | parse `#define POUNCE_*` lines | `tests/header_abi.rs` (S1-I4, I6, I9) |
| Header version matches the crate | parse `POUNCE_VERSION_*` vs `CARGO_PKG_VERSION`; release guard | `tests/header_abi.rs`, `scripts/check-release-consistency.sh` (S1-I2) |
| Every symbol is documented | backtick presence in `README.md` and `docs/src/c-api.md` | `tests/export_surface.rs` |
| Every symbol has a test and an error-path test | `NAME(` under `tests/`+`examples/`; `// error-path: NAME` marker; frozen grandfather list | `tests/export_surface.rs` |
| The header is real, compilable C99 and the cdylib links and runs | compile every `examples/*.c` with `-Werror -pedantic`, link, run | `tests/c_examples_build_and_run.rs` (S1-I3) |
| New structs have the C layout the header says | C `sizeof`/`offsetof` program vs Rust `size_of`/`offset_of!` | S1-I6 |
| C API answers equal the CLI's | `.nl`→C-callback harness vs `CARGO_BIN_EXE_pounce --json-output`; exact iteration counts | `crates/pounce-cli/tests/cinterface_mpcc_parity.rs` (S1-I8) |
| Getters equal the JSON block | same solve, same handle | S1-I7 |
| Each MOI row works | one model per row | S1-I9 |
| Nothing else moved | fixture sweep, both legs, cc: lines, diff empty | S1-I1, S1-I11 |

---

## 6. Stage 2 — BLOCKED: in-place data update and warm re-solve

Do not start. Open questions for the Julia side, to be answered in the issue
before design:

1. Which data changes are needed per re-solve: bounds only (`x_L/x_U/g_L/g_U`),
   or also the starting point and multipliers (already in/out buffers of
   `IpoptSolve`), or objective/constraint *coefficients* (which live in the
   caller's callbacks and need no API)?
2. Is "without rebuilding" about avoiding `CreateIpoptProblem` (cheap: it
   copies four vectors) or about reusing solver state across solves (the
   `IpoptSolver` session, symbolic factorization, SQP working set —
   `IpoptSolveWarmStart` already carries the working set)?
3. Does MOI's `modify`/`set(ConstraintSet)` path hit this often enough to
   matter, and on which model sizes? A measurement from Pounce.jl's MOI tests
   would decide.
4. Should bound updates be allowed on a problem with declared pairs (the
   shape classification can change, so a pair may become invalid)? If so,
   the update must re-run `validate_pairs` and report.
5. Interaction with `IpoptCreateSolver`'s consume-the-handle semantics: a
   session cannot take new bounds today.

Candidate shape, **not to be implemented until confirmed**:
`Bool IpoptSetVariableBounds(IpoptProblem, const ipnumber* x_L, const ipnumber* x_U)`,
`Bool IpoptSetConstraintBounds(IpoptProblem, const ipnumber* g_L, const ipnumber* g_U)`,
both re-validating declared pairs, both refused on a consumed handle.

---

## 7. Stage 3 and Stage 4 — design and iteration list

### Stage 3 — matrix-form convex/conic entry

Read first: `crates/pounce-convex/src/qp.rs:33-120, 360-432` (`QpProblem`,
`QpStatus`, `QpSolution`); `crates/pounce-convex/src/ipm.rs:124-` (`QpOptions`),
`:339` (`solve_qp_ipm`), `:1426` (`_warm`); `cones/composite.rs:22-40`
(`ConeSpec`); `hsde_nonsym.rs:1905` (`solve_conic_hsde_nonsym`);
`active_set.rs:237` (`solve_qp_active_set_inertia`); `pounce-py/src/qp.rs:342-390`
(cone parsing and the refusals), `:611-700` (`solve_presolved`,
`demote_unverified_optimal`), `:775-895` (the `solve_qp` pipeline), `:895-`
(`solve_socp`); `python/pounce/qp.py:1091-1390` (`solve_qp`, `solve_socp`,
`QpResult` fields `:141-285`); `crates/pounce-rs/src/convex.rs` (the facade's
documented standard form); `docs/src/convex-solver.md`, `lp-qp-routing.md`;
`CLAUDE.md` "The convex arm is covered".

Design:
- Standard form is `QpProblem`: `min ½xᵀPx + cᵀx, Ax = b, Gx ≤ h, lb ≤ x ≤ ub`,
  `P` lower triangle in triplets, cones partition the `G` rows in order
  (`ConeSpec::{Nonneg(d), SecondOrder(d), Exponential, Power(α), Psd(d)}`).
- Not an Ipopt API, so names carry the `Pounce` prefix and do not pretend
  to be upstream:
  ```c
  typedef struct PounceQpInfo* PounceQp;
  PounceQp PounceCreateQp(ipindex n, ipindex m_eq, ipindex m_ineq, ipindex index_style);
  void     PounceFreeQp(PounceQp);
  Bool PounceQpSetObjective(PounceQp, ipindex nnz_p, const ipindex* p_row, const ipindex* p_col, const ipnumber* p_val, const ipnumber* c);
  Bool PounceQpSetEqualities(PounceQp, ipindex nnz_a, const ipindex* a_row, const ipindex* a_col, const ipnumber* a_val, const ipnumber* b);
  Bool PounceQpSetInequalities(PounceQp, ipindex nnz_g, const ipindex* g_row, const ipindex* g_col, const ipnumber* g_val, const ipnumber* h);
  Bool PounceQpSetBounds(PounceQp, const ipnumber* lb, const ipnumber* ub);   /* NULL = unbounded side */
  int  PounceQpAddCone(PounceQp, int kind /* POUNCE_CONE_* */, ipindex dim, ipnumber alpha);
  Bool PounceQpSetNumOption(PounceQp, const char* key, ipnumber v);   /* tol, max_iter, time_limit, tau, tau_max, qp_reg */
  Bool PounceQpSetStrOption(PounceQp, const char* key, const char* v); /* method=ipm|active-set, hessian_inertia, presolve, hsde, equilibrate, crossover: yes|no */
  int  PounceQpSolve(PounceQp);          /* POUNCE_QP_STATUS_* mirroring QpStatus; negative on setup errors */
  Bool PounceQpGetSolution(PounceQp, ipnumber* x, ipnumber* y, ipnumber* z, ipnumber* z_lb, ipnumber* z_ub, ipnumber* obj, ipindex* iters);
  Bool PounceQpGetResiduals(PounceQp, PounceQpResiduals*);   /* the QpResult residual/kkt fields, scaled and raw */
  Bool PounceQpGetCertificate(PounceQp, PounceQpCertificate*); /* Farkas / recession data when status is infeasible */
  Bool PounceQpWriteSolveReport(PounceQp, const char* path, const char* detail);
  ```
  Index style 0/1 for the triplets. Cone rows must cover exactly `m_ineq`
  (checked at solve; error code). PSD refusal of an indefinite `P` under
  `method=ipm` (the gh#112 rule) is **kept**: the C API calls the same check
  Python does (`_check_psd` is Python-side; the Rust side is
  `demote_unverified_optimal`/`hessian_inertia` — read both and reuse the
  Rust one; if only Python checks PSD, lift that check into `pounce-convex`
  in S3-I1 so C and Python share it).
- The pipeline must be **the Python one** (`presolve → engine →
  demote_unverified_optimal → crossover report`), lifted into `pounce-convex`
  so there is one implementation.

Iterations:
- **S3-I1** Move `solve_presolved`, `demote_unverified_optimal`,
  `EngineSwitches` and the PSD screen from `pounce-py/src/qp.rs` into
  `pounce_convex::pipeline`; Python calls the moved functions. Pure move.
  Gate: `cargo test -p pounce-convex -p pounce-py`, `make python-test`
  (`python/tests/test_qp_host.py`, `test_socp_jax.py`, `test_qp_sensitivity.py`),
  **fixture sweep both legs diff empty** (37 + 5 fixtures reach the convex
  arm; the CLI's dispatch is a separate pipeline — confirm it is untouched
  by reading `crates/pounce-cli/src/dispatch.rs`), `benchmarks/qp` not
  required for a pure move but run `make benchmark` on one Maros–Meszaros
  family if any convex line moved.
- **S3-I2** Handle, builders, `PounceQpSolve`, getters, report writer, in a
  new `src/qp.rs` module of `pounce-cinterface`; `pounce-convex` becomes a
  dependency. Header, defines, `header_abi` define test.
- **S3-I3** Tests: LP, convex QP, SOC, exp, power, PSD, each a tiny model
  with a hand-known optimum; indefinite `P` refused under `ipm` and solved
  under `active-set`; cone dims not covering `m_ineq`; `dim ≤ 0`; `alpha`
  outside `(0,1)`; NULL arrays; unknown option key; `Get*` before solve →
  `FALSE`. Parity: same models through Python `solve_qp`/`solve_socp`
  anchors copied into the test with their source line cited, **and** through
  `pounce_convex` directly from Rust (same iteration counts, since it is the
  same pipeline). `// error-path:` markers for every new symbol.
- **S3-I4** `examples/qp_socp.c`, `docs/src/c-api.md` section,
  `docs/src/convex-solver.md` cross-link, README, CHANGELOG; export guard
  green; `make c-api-examples`.

### Stage 4 — multistart over an NLP handle

Read first: `crates/pounce-minima/src/search.rs:160-200` (`Hooks`,
`TraceEntry`), `:1071-1130` (`Minimum`, `MinimaResult`, `MinimaError`),
`:1140-1160` (`find_minima`, `find_minima_with`); `options.rs:10-60`
(`Method`, `parse`, `as_str`), `:64-130` (`MinimaOptions`);
`local_solve.rs:82-106` (`LocalSolve`), `:143-170` (`IpoptLocalSolve::new`);
`crates/pounce-py/src/minima.rs:1-120` (the Python glue, including the
no-Hessian → `QuasiNewton` demotion); `crates/pounce-cli/src/minima/mod.rs:29-`;
`python/pounce/_minima.py`; `python/tests/test_minima_cli_parity.py`;
`scripts/sweep-minima.sh`; `docs/src/find-minima*.md`; `cli/main.rs:854-858`
(pairs refused).

Design:
```c
typedef struct {
    int      method;          /* POUNCE_MINIMA_METHOD_* from Method::as_str names */
    ipindex  n_minima;        /* ≤ 0 → default 10 */
    ipindex  max_solves;      /* ≤ 0 → default 8·n_minima */
    ipindex  patience;        /* ≤ 0 → default */
    ipnumber dedup, psd_tol;  /* NaN → default */
    ipnumber kkt_tol;         /* NaN → default; +inf → disabled (None) */
    int64_t  seed;            /* <0 → 0 */
    ipindex  sobol;           /* 0/1, -1 default */
    ipnumber time_limit;      /* seconds; NaN → unbounded */
    /* strategy knobs, NaN → per-method default: */
    ipnumber sigma, sigma_frac, amplitude, amp_margin, eta, power, soft, length, length_frac, gamma, step, temperature, restart_jitter;
    ipindex  samples_per_round; /* ≤ 0 → default */
} PounceMinimaOptions;
void PounceMinimaOptionsInit(PounceMinimaOptions*);   /* fills every default sentinel */

typedef struct PounceMinimaInfo* PounceMinima;
PounceMinima PounceFindMinima(IpoptProblem, const PounceMinimaOptions*, UserDataPtr); /* NULL on error; see PounceMinimaLastError */
void    PounceFreeMinima(PounceMinima);
ipindex GetPounceMinimaCount(PounceMinima);                 /* -1 on NULL */
Bool    GetPounceMinimum(PounceMinima, ipindex k, ipnumber* x, ipnumber* obj, ipnumber* lambda, ipnumber* z_L, ipnumber* z_U); /* any buffer NULL to skip */
int     GetPounceMinimaStop(PounceMinima);                   /* POUNCE_MINIMA_STOP_* from Stop::as_str */
Bool    GetPounceMinimaStats(PounceMinima, ipindex* n_solves, ipindex* n_kkt_rejected);
ipindex GetPounceMinimaWarning(PounceMinima, ipindex k, char* buf, ipindex buflen); /* snprintf contract; -1 past the end */
```
`PounceFindMinima` builds the `CCallbackTnlp` bridge exactly as `IpoptSolve`
does (restoration wiring included), wraps `IpoptLocalSolve::new(&mut info.app)`
in a `LocalSolve` that demotes to `HessianMode::QuasiNewton` when `eval_h`
is `None` (as `PyLocalSolve` does; also set `hessian_approximation
limited-memory` the way `PyProblem::prepare` does — read it), refuses
declared pairs (`NULL` + print-gated line, as the CLI), and calls
`find_minima_with(&bridge, &mut solver, &cfg, Hooks { trace: false, .. })`.
Results are copied into the opaque handle; the `IpoptProblem` is not
consumed. `last_solve` is **not** overwritten (the search is many solves;
the per-solve getters keep describing the last `IpoptSolve`).

Iterations:
- **S4-I1** Module `src/minima.rs` in `pounce-cinterface`; `pounce-minima`
  dependency; options struct + defaults; result handle + getters; header,
  defines, `header_abi` define test; tests: a two-well 1-D model through C
  callbacks finds both minima under each of the six methods (assert count
  and values), `kkt_tol = +inf` disables the veto, pairs → `NULL`, NULL
  options → defaults, `k` out of range → `FALSE`, warnings text contract.
  `// error-path:` markers.
- **S4-I2** Parity: in `crates/pounce-cli/tests/cinterface_minima_parity.rs`
  (same harness as S1-I8), one `.nl` fixture `scripts/sweep-minima.sh` uses
  (read the script for its list), one seed, all six methods, versus
  `pounce <fixture> --minima ... --json-output` — same status, solve count
  and minima values (the contract `python/tests/test_minima_cli_parity.py`
  holds Python to). `examples/find_minima.c`; `docs/src/find-minima.md` gets
  a C subsection; README; CHANGELOG; export guard. If `pounce-minima` itself
  was touched (it should not be), run `scripts/sweep-minima.sh` against a
  baseline binary and diff empty.

---

## 8. ABI-review checklist for the human (filled in by S1-I10)

Written at S1-I10 against the tree that ticks it. Base for "added" /
"changed" is `2ac0ad9` (0.16.0). Layouts measured on x86_64 Linux (gcc,
LP64) with a `sizeof`/`offsetof` program against `pounce.h`; the same
numbers are asserted against the Rust side on every `cargo test -p
pounce-cinterface` by `repr_c_structs_match_the_header_layout`
(`tests/c_examples_build_and_run.rs`).

### 8.1 Surface added (nothing removed, no existing prototype changed)

Nine functions, all `#[unsafe(no_mangle)] pub unsafe extern "C"`, bodies
through `ffi_guard`, prototypes in `pounce.h` (held to the Rust signatures by
`tests/export_surface.rs`):

- [ ] `int AddPounceComplementarityPair(IpoptProblem, ipindex row, ipindex var)`
- [ ] `int SetPounceComplementarityPairs(IpoptProblem, ipindex n_pairs, const ipindex *rows, const ipindex *vars, ipindex *bad_index)`
- [ ] `Bool ClearPounceComplementarityPairs(IpoptProblem)`
- [ ] `ipindex GetPounceComplementarityPairCount(IpoptProblem)` (`-1` on NULL)
- [ ] `Bool GetPounceComplementarityPairs(IpoptProblem, ipindex *rows, ipindex *vars)`
- [ ] `int AddPounceComplementsConstraint(IpoptProblem, ipindex row, ipindex var)`
- [ ] `Bool GetPounceComplementarityStats(IpoptProblem, PounceComplementarityStats *)`
- [ ] `Bool GetPounceComplementaritySolve(IpoptProblem, ipindex k, PounceComplementaritySolve *)`
- [ ] `ipindex GetPounceComplementarityText(IpoptProblem, int which, char *buf, ipindex buflen)`

No Fortran symbol added (`src/fortran.rs` is still the eight upstream
`ip*_`; the export guard pins that).

### 8.2 `#define`s added (values held to the Rust `pub const`s, both directions, by `tests/header_abi.rs`)

- [ ] Declaration codes: `POUNCE_CC_OK 0`, `_ERR_NULL -1`, `_ERR_OUT_OF_RANGE -2`,
      `_ERR_VAR_ALREADY -3`, `_ERR_ROW_ALREADY -4`, `_ERR_SHAPE -5`,
      `_ERR_COUNT -6`, `_ERR_INTERNAL -7`.
- [ ] MOI codes: `POUNCE_COMPLEMENTS_PAIR_MIXED 0`, `_PAIR_LOWER 1`,
      `_PAIR_UPPER 2`, `_ROW_FIXED_NO_PAIR 3`, `_VAR_FIXED_NO_PAIR 4`,
      `_ERR_NULL -1`, `_ERR_OUT_OF_RANGE -2`, `_ERR_VAR_ALREADY_PAIRED -3`,
      `_ERR_ROW_ALREADY_PAIRED -4`, `_ERR_ROW_NOT_FREE -5`,
      `_ERR_VAR_BOUNDS -6`, `_ERR_INTERNAL -7` (shared negatives equal the
      `POUNCE_CC_*` ones).
- [ ] Certificate: `POUNCE_CC_CLASS_S 0`, `_W 1`, `_NONE 2`,
      `_NOT_SEARCHED 3`, `_ABSENT -1`.
- [ ] Reported: `POUNCE_CC_REPORTED_ANSWER 0`, `_WEAK_FALLBACK 1`,
      `_FEASIBLE_UNCERTIFIED 2`, `_VERDICT 3`.
- [ ] B-check: `POUNCE_CC_B_VERDICT_B 0`, `_NOT_B 1`, `_NOT_CHECKED 2`,
      `_ABSENT -1`.
- [ ] Texts: `POUNCE_CC_TEXT_SUMMARY 0`, `_POLISH_REASON 1`, `_B_REASON 2`.

### 8.3 `#define`s changed

- [ ] `POUNCE_VERSION_MINOR 1 → 16`, `POUNCE_VERSION "0.1.0" → "0.16.0"`
      (S1-I2). Loop rule 8 forbids changing an existing `#define` value;
      S1-I2 ordered this fix because the old value was simply wrong (the
      library has reported 0.16.0 all along). No C code in the repo reads
      them. Now held to `CARGO_PKG_VERSION` by `header_abi.rs` and check 6
      of `scripts/check-release-consistency.sh`. Confirm this is acceptable
      as a fix rather than an ABI change.

### 8.4 Structs added (`PounceLinearSolverStats` untouched: 96 bytes, offsets guarded)

> Superseded before release: both structs now begin with `ipindex
> struct_size` (§8.5, and §9 "ABI hardening before release" for the new
> offsets). The tables below are the S1-I10 layout.

- [ ] `PounceComplementarityStats`, 224 bytes, align 8:

  | field | type | offset |
  |---|---|---|
  | `strategy` | `char[16]` | 0 |
  | `reported` | `ipindex` | 16 |
  | `accepted_by` | `char[32]` | 20 |
  | `fallback` | `char[32]` | 52 |
  | `n_solves` | `ipindex` | 84 |
  | `total_iterations` | `ipindex` | 88 |
  | `max_abs_product` | `ipnumber` | 96 |
  | `certificate_class` | `ipindex` | 104 |
  | `n_pairs` | `ipindex` | 108 |
  | `n_biactive` | `ipindex` | 112 |
  | `n_neither` | `ipindex` | 116 |
  | `multipliers_unique` | `ipindex` | 120 |
  | `residual` | `ipnumber` | 128 |
  | `resid_tol` | `ipnumber` | 136 |
  | `tight` | `ipindex` | 144 |
  | `feasible` | `ipindex` | 148 |
  | `primal_violation` | `ipnumber` | 152 |
  | `b_verdict` | `ipindex` | 160 |
  | `b_n_biactive` | `ipindex` | 164 |
  | `b_branches_solved` | `ipindex` | 168 |
  | `b_enumerated` | `ipindex` | 172 |
  | `escapes` | `ipindex` | 176 |
  | `lifted` | `ipindex` | 180 |
  | `natural_residual` | `ipnumber` | 184 |
  | `polish_ran` | `ipindex` | 192 |
  | `polish_taken` | `ipindex` | 196 |
  | `polish_residual_before` | `ipnumber` | 200 |
  | `polish_residual_after` | `ipnumber` | 208 |
  | `polish_threshold` | `ipnumber` | 216 |

  Padding: 4 bytes at 92, 124. A mirror in another language must
  reproduce it (Julia: a plain `struct` with these field types does).
- [ ] `PounceComplementaritySolve`, 72 bytes, align 8: `stage char[32]` 0,
      `relaxed ipindex` 32, `tau ipnumber` 40, `status ipindex` 48,
      `iterations ipindex` 52, `max_abs_product ipnumber` 56,
      `seconds ipnumber` 64 (padding 4 bytes at 36).

### 8.5 Decisions the plan did not settle, made conservatively, to confirm

- [ ] **`POUNCE_CC_ERR_INTERNAL = -7`** (S1-I4): the `ffi_guard` fallback
      for the `int`-returning declaration calls, a value that cannot be
      read as a diagnosis of the input. Not in §3.2.
- [ ] **`POUNCE_COMPLEMENTS_ERR_INTERNAL = -7`** (S1-I9): the same, plus the
      re-validation refusal after a tentative bound edit (unreachable from
      any public call today; Stage 2's in-place bound update would make it
      reachable).
- [ ] **`POUNCE_COMPLEMENTS_ERR_VAR_BOUNDS = -6`** (S1-I9): NaN or crossed
      (`l > u`) variable bounds match no row of the MOI table. Not in §3.1.
- [ ] **`-1` sentinels** (S1-I6): with no certificate its counts and 0/1
      flags read `-1` (reals `NaN`); with no B-check, `b_n_biactive`,
      `b_branches_solved`, `b_enumerated` read `-1`; `polish_taken` reads
      `0` when `polish_ran` is `0`. The `PounceLinearSolverStats` precedent,
      instead of per-field "present" flags.
- [ ] **The outcome record survives a clear** (S1-I6/I7): the getters and
      the report's `complementarity` block describe the most recent solve;
      `ClearPounceComplementarityPairs` after a driver solve does not erase
      them; the next solve (with or without pairs) replaces them, and a
      refused solve discards them.
- [ ] **`complements_used` semantics** (S1-I9): a row or variable accepted
      by `AddPounceComplementsConstraint` — pair or not — can never be used
      by that function again on the handle; Clear/Set empty neither the
      ledger nor the bound edits. The native `AddPounceComplementarityPair`
      does not read the ledger.
- [ ] **`ROW_NOT_FREE` for agreeing bounds** (S1-I9): a row whose bound
      already agrees with MOI's sign (`g_L = 0` with `x >= 0`) is refused as
      `ROW_NOT_FREE` (§3.1: "any other row"); a frontend must declare it
      natively. Because the free check comes first, a row in a one-sided
      pair reads `ROW_NOT_FREE`, not `ROW_ALREADY_PAIRED`.
- [ ] **The live inspectors are guarded** (S1-I5, deviation from §3.5): the
      plan thought they already returned `FALSE` during a lowered solve;
      an `x`-only `GetIpoptCurrentIterate` returned `TRUE`. A thread-local
      guard now makes both inspectors `FALSE` for every request shape.
- [ ] **Header version fix** (§8.3).
- [ ] **`pounce-cinterface` as a dev-dependency of `pounce-cli`** (S1-I8),
      via `[workspace.dependencies]` (`version = "0.16.0"`, path). **Not**
      stripped on publish (corrected in the second review round: cargo
      keeps a versioned dev-dependency in the published manifest), so
      `pounce-cli` can publish only after `pounce-cinterface` is live on
      crates.io; `scripts/publish-crates.sh` orders them so and
      `scripts/check-release-consistency.sh` counts dev-dependency edges. Its `version` moves with the
      other `[workspace.dependencies]` entries at a release bump.
- [ ] **`serde_json` `float_roundtrip` dev feature** in
      `pounce-cinterface` (S1-I7): tests only, so reals read back to the
      bit; `Cargo.lock` unchanged. `pounce-cli`'s parser deliberately not
      changed (its tests compare report text instead).
- [x] **No `second_opinion` block in the C report** (noticed in S1-I7):
      **resolved by S1-I12**, which the human approved. `IpoptWriteSolveReport`
      now writes the block under the CLI's and Python's `ran()` test, through
      the same `report_info()`; `docs/src/c-api.md`, `pounce.h`, the README,
      the schema page and the CHANGELOG say so.
- [ ] **Trajectory vs the CLI** (S1-I5/I8): a callback model and the `.nl`
      file of the same model agree on the answer but not always on the
      iteration count (row constant in `g` or in its bound; the `.nl`
      nonlinear-variable list under L-BFGS). Documented as frontend
      guidance in `docs/src/c-api.md` ("Matching the CLI"), not changed.
- [x] **The output structs could not grow** (ABI review before release):
      `PounceComplementarityStats` and `PounceComplementaritySolve` (and
      §8.7's two) would have frozen at release, and a field added later
      would overflow an old caller's buffer. **Resolved by `struct_size`**,
      which the human approved: each now begins with `ipindex struct_size`
      (caller sets `sizeof`); a smaller value is refused with nothing
      written; the getter writes this release's bytes only, never
      `struct_size` and never past its own size; later releases append only
      and write `min(struct_size, sizeof)`. Layouts in §8.4 are superseded
      by the §9 "ABI hardening before release" entry (224 and 72 bytes).
      `PounceLinearSolverStats` unchanged (it shipped).

### 8.6 Release and Pounce.jl

- [ ] `GetIpoptVersion` reports `0.17.0` once released; `pounce.h`
      `POUNCE_VERSION_*` agree (release guard check 6). Stage 1 ships in
      **0.17.0**; this branch bumps nothing.
- [ ] Pounce.jl gate, `Pounce.jl/src/Pounce.jl` `_check_native_version`:
      today `v"0.15.0" <= version < v"0.17.0-0"` (and its error text names
      "0.15.x and 0.16.x"). After this review, widen to
      `< v"0.18.0-0"` and the message to "0.15.x through 0.17.x"; update
      `Pounce.jl/README.md` and `docs/src/julia.md` ("accepts native POUNCE
      0.15.x and 0.16.x"; S1-I10 added a sentence there saying 0.17 needs
      the widened gate). Then `Pkg.test()` must pass against the 0.17
      cdylib **with no other Julia change** (additions only).
- [ ] Optional for Pounce.jl: add the new symbols to
      `_required_native_symbols` only when the MOI side starts using them
      (it would otherwise refuse a 0.15/0.16 library it can still drive).
- [ ] Manual runs not available to the loop: `make -C benchmarks
      mpcc-opf-run` (volt-var/volt-watt, data root) against a baseline, and
      the other declared-pair corpora named in `CLAUDE.md`, only if a review
      finds `pounce-nlp` changed beyond the pure classifier move (S1-I1's
      sweep and S1-I11's say it did not).

### 8.7 Stage 3 — matrix-form convex/conic entry (filled in by S3-I4)

Written at S3-I4 against the tree that ticks it. Base for "added" is the
Stage 1 surface above; nothing in §8.1–8.4 changed. Layouts measured as in
S1-I10 (x86_64 Linux, gcc 13.3, LP64, a `sizeof`/`offsetof`/`_Alignof`
program against `pounce.h`) and held to the Rust side on every `cargo test
-p pounce-cinterface` by `repr_c_structs_match_the_header_layout`.

Fourteen functions (`src/qp.rs`, all `#[unsafe(no_mangle)] pub unsafe extern
"C"`, bodies through `ffi_guard` except `PounceFreeQp` and the NULL check of
`PounceQpSolve`), and one opaque handle type
`typedef struct PounceQpInfo* PounceQp`:

- [ ] `PounceQp PounceCreateQp(ipindex n, ipindex m_eq, ipindex m_ineq, ipindex index_style)` (`NULL` unless `n ≥ 1`, `m_* ≥ 0`, style 0/1)
- [ ] `void PounceFreeQp(PounceQp)` (`NULL` no-op)
- [ ] `Bool PounceQpSetObjective(PounceQp, ipindex nnz_p, const ipindex *p_row, const ipindex *p_col, const ipnumber *p_val, const ipnumber *c)`
- [ ] `Bool PounceQpSetEqualities(PounceQp, ipindex nnz_a, const ipindex *a_row, const ipindex *a_col, const ipnumber *a_val, const ipnumber *b)`
- [ ] `Bool PounceQpSetInequalities(PounceQp, ipindex nnz_g, const ipindex *g_row, const ipindex *g_col, const ipnumber *g_val, const ipnumber *h)`
- [ ] `Bool PounceQpSetBounds(PounceQp, const ipnumber *lb, const ipnumber *ub)`
- [ ] `int PounceQpAddCone(PounceQp, int kind, ipindex dim, ipnumber alpha)`
- [ ] `Bool PounceQpSetNumOption(PounceQp, const char *key, ipnumber value)`
- [ ] `Bool PounceQpSetStrOption(PounceQp, const char *key, const char *value)`
- [ ] `int PounceQpSolve(PounceQp)`
- [ ] `Bool PounceQpGetSolution(PounceQp, ipnumber *x, ipnumber *y, ipnumber *z, ipnumber *z_lb, ipnumber *z_ub, ipnumber *obj, ipindex *iters)`
- [ ] `Bool PounceQpGetResiduals(PounceQp, PounceQpResiduals *)`
- [ ] `Bool PounceQpGetCertificate(PounceQp, PounceQpCertificate *)`
- [ ] `Bool PounceQpWriteSolveReport(PounceQp, const char *path, const char *detail)`

(S3-I2's §9 entry says "15 exports"; `grep -c 'pub unsafe extern "C" fn'
src/qp.rs` is 14, the list above. `export_surface.rs` holds the header and
the Rust signatures to each other.)

Twenty-two `#define`s (held both ways by
`header_qp_codes_match_the_implementation` in `tests/header_abi.rs`):

- [ ] `POUNCE_QP_STATUS_OPTIMAL 0`, `_OPTIMAL_INACCURATE 1`,
      `_PRIMAL_INFEASIBLE 2`, `_DUAL_INFEASIBLE 3`, `_ITERATION_LIMIT 4`,
      `_TIME_LIMIT 5`, `_NUMERICAL_FAILURE 6` (one per `QpStatus`).
- [ ] `POUNCE_QP_ERR_NULL -1`, `_ARGUMENT -2`, `_INCOMPLETE -3`,
      `_CONES -4`, `_OPTIONS -5`, `_INDEFINITE -6`, `_INTERNAL -7`.
- [ ] `POUNCE_CONE_NONNEG 0`, `_SOC 1`, `_EXP 2`, `_POW 3`, `_PSD 4`.
- [ ] `POUNCE_QP_CERT_NONE 0`, `_PRIMAL_INFEASIBLE 1`, `_DUAL_INFEASIBLE 2`.

Two structs:

- [ ] `PounceQpResiduals`, 64 bytes, align 8: `primal_infeasibility` 0,
      `dual_infeasibility` 8, `complementarity` 16, `kkt_error` 24,
      `primal_infeasibility_raw` 32, `dual_infeasibility_raw` 40,
      `complementarity_raw` 48, `kkt_error_raw` 56 (all `ipnumber`; no
      padding).
- [ ] `PounceQpCertificate`, 32 bytes, align 8: `kind ipindex` 0,
      `scale ipnumber` 8, `tau ipnumber` 16, `kappa ipnumber` 24 (padding
      4 bytes at 4).

Decisions made in S3-I1/S3-I2/S3-I3, to confirm:

- [ ] **PSD screen** (S3-I1/S3-I2): `hessian_inertia=auto` asks
      `pounce_convex::certify_psd_lower_triangle` at the CLI's band
      `1e-9·min(max|P_ij|, 1)`, and an undecided factorization counts as
      not certified (refused under `ipm`). Python's numpy verdict
      (`_psd_verdict`, relative `1e-8·max|P_ij|`, undecided → unknown with a
      warning) is not ported, so the C entry can refuse a borderline or badly
      scaled rank-deficient `P` that Python accepts. `hessian_inertia=psd`
      overrides it. S3-I3's parity models keep every eigenvalue in
      `{1, 2, 3, −1}`, away from the border, so no test measures the
      divergence.
- [ ] **Option spellings** (S3-I2): `qp_presolve`, `qp_hsde`,
      `qp_equilibrate`, `qp_crossover` (the CLI's and Python's names, like
      `qp_reg`), not §7's bare `presolve` etc.
- [ ] **Conic option refusals** (S3-I2): with any cone, `method=active-set`,
      `tau`, `tau_max` and every `qp_*` switch (even `qp_hsde=yes`, the
      default) are `POUNCE_QP_ERR_OPTIONS`, because `solve_socp` takes none
      of them; refused rather than silently dropped.
- [ ] **`cvx-qcqp` for exp/pow/PSD** (S3-I2): the report's
      `solution.engine` is `cvx-qcqp` for every conic solve, including the
      non-symmetric exp/power driver; there is no engine string for a
      matrix-form conic solve, and a new one would be a new value in the
      report schema's vocabulary.
- [ ] **Report input and `lambda`** (S3-I2): the input descriptor is
      `TnlpDirect` (no `.nl` file, no matrix-form descriptor exists), and
      `solution.lambda` is `[y; z]` (equality then inequality multipliers);
      `z_lb`, `z_ub` are not in the report.
- [x] **Small re-statements of private helpers** (S3-I2): three were
      copies — the status mapping (`qp_status_to_ars`,
      `pounce-cli/src/main.rs`), the PSD band (`dispatch.rs::PSD_TOL` /
      `psd_band`), and Python's residual block and certificate inf-norms
      (`pounce-py/src/qp.rs::solution_dict_hsde`). **Resolved** in the second
      review round (§9): all three moved, unchanged, into `pounce_convex`
      (`qp_status_to_ars`, `reported_kkt_residuals`, `certificate_scale` in
      `report.rs`; `PSD_TOL`, `psd_band` beside
      `certify_psd_lower_triangle`), and the CLI, Python and C call those
      definitions. Sweep empty both legs; Python output byte-identical.
      The `opts_tau` override rule (a bare `tau` lifts `tau_max`) followed
      as `pounce_convex::qp_options_from_overrides`, called by Python's
      `opts_tau` and by `PounceQpSolve`; unit-tested in `pipeline.rs` and
      pinned by the bit-exact parity rows with `tau`/`tau_max`; Python output
      byte-identical (180 lines). `pounce-cli/src/bin/pounce_cblib.rs` keeps
      its own status map, which differs (`NumericalFailure → InternalError`),
      so sharing it would change that binary; left as is.
- [x] **PSD cost per solve** (review): the C entry summed `P` into a map for
      the scale and the certificate summed it again, on every
      `PounceQpSolve`. **Resolved**: `certify_psd_lower_triangle_in_band`
      reads the band off the certificate's own sum, and the verdict is
      cached on the handle, dropped only by `PounceQpSetObjective` and any
      `hessian_inertia` setting.
- [ ] **Two unreachable overflow guards** (S3-I3): `PounceQpAddCone`'s PSD
      `d·(d+1)` `checked_mul` and `PounceQpSolve`'s cone-row `checked_add`
      cannot overflow on a 64-bit target with a 32-bit `ipindex`; kept as
      guards, not tested.
- [x] **Bounds with cones** (noticed in S3-I4): the handle accepts a box with
      cones (`solve_socp_ipm` appends it as a trailing nonnegative block),
      which Python's `solve_socp` does not offer, so that combination has no
      Python parity row. Checked by hand once in S3-I4 (SOC distance with
      `x1 ≤ 1`: `x = (1, 1, −1)`, `z_ub[1] = 1`), not by a test.
      **Resolved** (ABI hardening, §9): `bounds_with_a_cone_active_bound` and
      `bounds_with_a_cone_inactive_bounds` in `tests/qp_cones_and_errors.rs`,
      answers re-derived by hand; still no Python row (reason in the test).
- [ ] **Conic `hessian_inertia` refusal** (second review round): with any
      cone, `hessian_inertia` other than `auto` (`psd`, `indefinite`,
      `unknown`) is `POUNCE_QP_ERR_OPTIONS`. `solve_socp` takes no such
      argument and always screens; `unknown` had skipped the screen (an
      indefinite `P` reached the conic IPM), and `indefinite` had returned
      `INDEFINITE` where the docs said `OPTIONS`.
- [ ] **Bounds refusals** (second review round): `PounceQpSetBounds` refuses,
      storing nothing, `lb[i] ≥ 1e19`, `ub[i] ≤ -1e19` and `lb[i] > ub[i]`
      (as well as NaN); `lb == ub` is accepted.
- [ ] **Results go with the problem** (second review round): every builder
      that returns `TRUE` (`SetObjective`, `SetEqualities`,
      `SetInequalities`, `SetBounds`, `AddCone`) discards the last result, so
      the getters and the report return `FALSE` until the next solve; the
      option setters keep it (decision: the result is still a solve of the
      problem held, as `AddIpoptNumOption` keeps an `IpoptProblem`'s).
      `PounceQpGetSolution` returns `FALSE`, writing nothing, if a result
      vector is not of its declared length.
- [x] **`PounceQpResiduals` / `PounceQpCertificate` could not grow**:
      resolved by `struct_size`, as §8.5's item; layouts now 72/8 (padding 4
      at 4) and 32/8 (no padding), see §9.

## 9. Per-iteration log

Append one entry per iteration: `### <item> — <commit>` then what moved,
what was measured (with the command), and anything that surprised you.

### S1-I1 — shared pair classifier (commit that ticks S1-I1)

What moved: `crates/pounce-nlp/src/mpcc.rs` only. New public
`PairShape`, `PairErrorKind`, `PairError` (`Display` = `message`,
`std::error::Error`), `classify_pair(xl, xu, gl, gu)` and
`validate_pairs(n, m, x_l, x_u, g_l, g_u, pairs)`; `MpccTnlp::with_lift` now
calls `validate_pairs(..).map_err(|e| e.to_string())` and builds
`lowered/g_side/h_side/signless` from the shapes. Check order unchanged
(per pair: range, duplicate var, duplicate row, shape). `is_mixed` stays
public (used by `pounce-algorithm/src/mpcc.rs`). The four message format
strings were extracted from the base and head files and diffed: identical.

Tests added (all in `mpcc.rs` tests): one per `classify_pair` branch
(one-sided lower/lower, upper/upper, lower/upper, upper/lower; plus-bound
with row lower and row upper; plus-bound with a fixed variable; mixed) and
per refusal (boxed row; both free; fixed variable with free row; equality
row; var out of range; row out of range; duplicate var; duplicate row; first
refusal in declaration order), each asserting `PairErrorKind` and the
literal pre-refactor text; plus one that `MpccTnlp::new`'s `Err` equals
`validate_pairs`'s `Display`.

Gate (all on the ticking commit's tree):
- `cargo test -p pounce-nlp` — `test result: ok. 188 passed` (lib).
- `cargo test -p pounce-restoration` — all green (lib 157 passed).
- `cargo test -p pounce-cli --test mpcc_complementarity` — `ok. 13 passed`.
- `make python-ext && python -m pytest python/tests/test_complementarity.py -q`
  — `19 passed` (needed a virtualenv: `maturin develop` refuses without one).
- `cargo test -p pounce-cinterface` — all green (lib 52 passed).
- Sweep: baseline = `73a98e9` (2ac0ad9 + this plan, no code change) built
  `--release -p pounce-cli` in a `git worktree` with its own
  `CARGO_TARGET_DIR`; head = this commit's `target/release/pounce`.
  `scripts/sweep-fixtures.sh <bin> <out>` for each, `diff` — **empty**,
  282 lines each (141 exact, 141 lbfgs, 14 of them `cc:` lines).

Noticed: `cargo clippy -p pounce-nlp --all-targets -- -D warnings` fails on
five pre-existing lints in `pounce-common` (collapsible `if`, `expect`) with
this toolchain; nothing in `mpcc.rs` warns. Not touched (out of scope).

### S1-I2 — export-surface guard (commit that ticks S1-I2)

What moved:
- `crates/pounce-cinterface/tests/export_surface.rs` (new): scans
  `src/lib.rs` + `src/solver.rs` for `#[unsafe(no_mangle)] pub unsafe extern
  "C" fn` (a `no_mangle` not followed within three lines by one is an error)
  and `src/fortran.rs` separately; parses top-level prototypes out of
  `pounce.h` (comments, preprocessor lines and the `#ifdef __cplusplus`
  blocks stripped; `typedef`/`enum {}`/struct statements skipped). Asserts
  equal name sets, equal arity, positional width-class agreement. Function
  pointers carry their typedef name (`fnptr(Eval_G_CB)`), so swapped
  callbacks of the same width are caught too — stricter than the plan's bare
  `fnptr`. Also: every name in backticks in README and `docs/src/c-api.md`,
  Fortran names in README's Fortran section; every name called as `NAME(`;
  every name has a `// error-path: NAME` marker; markers naming no export
  fail; the Fortran shim is exactly the eight upstream symbols. Nine unit
  tests feed the comparison broken inputs (missing prototype, phantom
  prototype, arity, width on a parameter and on a return, swapped callbacks,
  unparseable export, doc/marker scanner branches).
- **Deviation from the plan text, deliberate:** items 5 and 6 (called /
  error-path marker) also look in the `#[cfg(test)]` part of `src/*.rs`
  (everything after its first `#[cfg(test)]`), not only `tests/` and
  `examples/`. Most existing error-path tests (NULL handle, no prior solve,
  bad status code) are unit tests in `src/lib.rs`/`src/solver.rs`; the
  markers were added there (comment lines only, 23) rather than duplicating
  those tests.
- `GRANDFATHERED_NO_ERROR_PATH` is **empty**. Exports with no test or no
  failure-branch test got one in `tests/entry_point_error_paths.rs` (new, 10
  tests): `GetPounceRestorationStats`, `GetPounceFdHessianStats` (did-not-run
  vs FD-ran branches), `IpoptSolverKktSolveScaled` (agrees with
  `IpoptSolverKktSolve` on an unscaled model), all eight Fortran symbols
  (NULL handles; options set/refused; callback fires, then not after unset),
  plus NULL-handle branches of `SetIntermediateCallback`,
  `IpoptClear{NonlinearVariables,WarmStartWorkingSet}`,
  `IpoptEnableIterHistory`, `IpoptSolveWarmStart`, `IpoptWriteSolveReport`
  (NULL handle / no solve / NULL path / bad detail / success),
  `IpoptSolverSolve`, `IpoptSolverParametricStep`,
  `IpoptSolverReducedHessian`. One marker added to
  `tests/nonlinear_variables_mask.rs`.
- `pounce.h`: `POUNCE_VERSION_MINOR 1 → 16`, `POUNCE_VERSION "0.1.0" →
  "0.16.0"` (nothing else). `header_abi.rs` gains
  `header_version_defines_match_the_crate` (checked to fail with the old
  value); `scripts/check-release-consistency.sh` gains check 6 (checked to
  exit 1 with `MINOR 1`). The release skill now says to bump the defines.
- `docs/src/c-api.md` (new, all nine sections), wired into `SUMMARY.md`
  after Julia. Crate `README.md` Surface section completed (was missing 16
  C symbols and all Fortran ones), plus Fortran and Versioning subsections.

Gate (all on the ticking commit's tree):
- `cargo test -p pounce-cinterface --test export_surface --test header_abi`
  — `ok. 13 passed` / `ok. 3 passed`.
- `cargo test -p pounce-cinterface` — all green (lib 52, the new
  `entry_point_error_paths` 10, every other binary ok).
- `scripts/check-release-consistency.sh` — `check-release-consistency: OK`,
  exit 0.
- `make book` (mdbook 0.5.2, the version `docs.yml` pins, via `cargo
  install`) — builds, no warnings; `docs/book/c-api.html` written.
- `make ask-check` — `top-1 17/20`, `top-5 20/20`, OK; identical to the
  tree without `c-api.md`.
- `cargo fmt --all -- --check` clean; `cargo clippy -p pounce-cinterface
  --no-deps --all-targets -- -D clippy::correctness -D clippy::suspicious`
  (the CI gate) clean. Under plain `-D warnings` the new tests carry only the
  workspace's restriction-group `unwrap_used`/`expect_used` warns, which every
  existing test in the crate also carries; `src/` has two pre-existing
  style lints (`fortran.rs:493` unnecessary cast, `lib.rs:3711`
  useless `vec!`), not touched.

Noticed: the plan's loop rule 8 forbids changing an existing `#define`
value; S1-I2 itself orders the `POUNCE_VERSION*` fix, which is done and
flagged here for the ABI review. No C code in the repo reads those defines.

### S1-I3 — C examples compile and run (commit that ticks S1-I3)

What moved:
- `crates/pounce-cinterface/tests/c_examples_build_and_run.rs` (new,
  `#![cfg(unix)]`, 4 tests). Finds the cdylib as
  `current_exe().parent().parent()/libpounce_cinterface.{so,dylib}` (follows
  `CARGO_TARGET_DIR`, profile and `--target` without reading them) and
  fails, with the path, if it is absent. Compiler is `$CC` if non-empty,
  else `cc`; if it cannot run, the test fails unless
  `POUNCE_C_EXAMPLES_SKIP=1` (documented in the module docs). Each
  `examples/*.c` is built with `-std=c99 -Wall -Wextra -Werror -pedantic
  -I include <src> -o <tmp> -L <libdir> -lpounce_cinterface -lm
  -Wl,-rpath,<libdir>` into `CARGO_TARGET_TMPDIR`, run with
  `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` = `<libdir>`, and must exit 0 and
  print its `EXPECTED_STDOUT` line (`sqp_warm_start.c`: `all 4 solves
  succeeded`). Other tests: every example has an `EXPECTED_STDOUT` entry and
  every entry a file; the flag assembly (`compile_args`, pure) carries the
  strict flags, puts `-l` after the source, keeps a spaced path one argv
  entry; and the negative branch — a header-including file with one unused
  parameter must *fail* the strict build while the same file without it
  builds, links and runs.
- `-lm` is added (the plan's command line omitted it): `sqp_warm_start.c`
  calls `fabs`.
- `examples/sqp_warm_start.c`: did **not** build under the strict flags —
  `unused parameter 'n'` in `eval_jac_g` and `eval_h` (gcc and clang). Fixed
  with `(void)n;`; its header comment now gives the strict build line and
  `make c-api-examples`. The header itself was already clean C99.
- `Makefile`: `c-api-examples` (builds the cdylib for `$(PROFILE)`, compiles
  every example with `C_EXAMPLES_CFLAGS` into
  `$(TARGET_DIR)/c-api-examples/`, runs each). Crate README gains an
  *Examples* subsection.
- `.github/workflows/ci.yml` unchanged: the `test` job is `ubuntu-latest`
  (gcc preinstalled, `CC` and `POUNCE_C_EXAMPLES_SKIP` unset) and its
  `cargo test --workspace --exclude pounce-hsl` step runs this test.

Checked that the test bites: with the original example restored it fails
(`unused parameter 'n'`, 1 failed); `CC=/nonexistent` fails both compiler
tests with the "no C compiler" message; `CC=/nonexistent
POUNCE_C_EXAMPLES_SKIP=1` passes (skips); `CC=clang` passes.

Gate (all on the ticking commit's tree):
- `cargo test -p pounce-cinterface --test c_examples_build_and_run` —
  `ok. 4 passed`.
- `make c-api-examples` — release build, four `status = 0` steps, `all 4
  solves succeeded; final sum(x) = 1.000000000`.
- `cargo test -p pounce-cinterface` — all green (lib 52, every binary ok).
- `cargo fmt --all -- --check` clean; CI clippy flags
  (`-D clippy::correctness -D clippy::suspicious`) clean; under plain
  warnings the new file carries only the workspace's restriction-group
  `expect_used` warns, as every test in the crate does.

### S1-I4 — declaration API (commit that ticks S1-I4)

What moved:
- `src/lib.rs`: `IpoptProblemInfo.complementarity: Vec<ComplementarityPair>`
  (0-based, empty at create); `pub const POUNCE_CC_OK` .. `POUNCE_CC_ERR_COUNT`
  as §3.2 plus **`POUNCE_CC_ERR_INTERNAL = -7`** (not in §3.2: the
  `ffi_guard` fallback for the two `int`-returning entry points needed a
  value that cannot be read as a diagnosis of the input; flagged for the
  ABI review). `declare_pairs(info, new, replace)` maps each index through
  `zero_based` (`checked_sub` of the style offset; anything outside
  `[0, len)` becomes `usize::MAX`), so `validate_pairs` reports the **first
  refusal in declaration order** with its own range check rather than after
  a range pre-pass; stores only on `Ok`; `bad_index` is the position in
  the batch. Five entry points, all through `ffi_guard`:
  `AddPounceComplementarityPair`, `SetPounceComplementarityPairs`,
  `ClearPounceComplementarityPairs`, `GetPounceComplementarityPairCount`,
  `GetPounceComplementarityPairs`. Precedence in `Set`: NULL handle, then
  `n_pairs < 0`, then NULL arrays with `n_pairs > 0`; `bad_index` is `-1`
  on success and on those three. `Get` with no pairs returns `TRUE` and
  accepts NULL arrays.
- `pounce.h`: the eight `#define POUNCE_CC_*` and five prototypes after
  `IpoptClearNonlinearVariables`, with the shape/index-style contract and an
  explicit **"in development: IpoptSolve does not solve with them yet"**
  note. The same note is in README, `c-api.md`, `complementarity.md` ("C
  API" subsection) and the CHANGELOG `[Unreleased] ### Added` entry —
  **S1-I5 must remove all five** when it routes the solve.
- `tests/header_abi.rs`: `header_complementarity_codes_match_the_implementation`
  parses every `#define POUNCE_CC_` and holds names and values equal to the
  Rust constants in both directions (S1-I6 must add its `POUNCE_CC_CLASS_*`
  etc. to the list there).
- `tests/common/qpec.rs` (new): `QpecSmall` as C callbacks, Jacobian/Hessian
  structure generic over the index offset (`ev_jac_g::<1>` for Fortran
  style), `create(bounds.., index_style)` / `create_qpec(style)`.
- `tests/complementarity_declare.rs` (new, 34 tests): success paths (fresh
  handle; add + read back; both index styles; Fortran last index accepted,
  0 refused, `i32::MIN` no overflow; `Set` replaces; `Add` after `Set`
  appends; `Set` with 0 clears and accepts NULL; NULL `bad_index`; `Clear`;
  every lowered shape accepted) and one test per refusal: NULL handle on all
  five, negative count, each NULL-array combination, var and row out of
  range (incl. `Index::MAX`/`MIN`), a model with `m = 0`, duplicate var and
  duplicate row across two `Add`s, duplicate var and row inside one `Set`
  (`bad_index` = the second), out of range in `Set`, first-refusal order,
  all-or-nothing against a prior set, Fortran `bad_index`, and six shape
  refusals (boxed row, equality row, both free, free var + one-sided row,
  one-sided var + free row, fixed var + free row) with an accepting control
  through the same helper. Every refusal test runs on a handle that already
  holds a pair and asserts it is unchanged. `// error-path:` markers for all
  five names.

Gate (all on the ticking commit's tree):
- `cargo test -p pounce-cinterface` — all green, 18 binaries (lib 52,
  `complementarity_declare` 34, `export_surface` 13, `header_abi` 4,
  `c_examples_build_and_run` 4 — `sqp_warm_start.c` still builds strict
  against the extended header).
- `cargo clippy -p pounce-cinterface --all-targets -- -D warnings` — **fails
  before reaching this crate**, on the five pre-existing `pounce-common`
  lints S1-I1 recorded. With `--no-deps` the only warnings in `src/` are the
  two pre-existing ones (`fortran.rs` cast, `lib.rs` useless `vec!`, now at
  `:3967`); the three files this item adds or extends
  (`complementarity_declare.rs`, `common/qpec.rs`, `header_abi.rs`) have
  **zero** warnings even under plain clippy. CI flags (`-D clippy::correctness
  -D clippy::suspicious`, `--no-deps`) clean.
- `cargo fmt --all -- --check` clean; `mdbook build docs` (0.5.2) clean, the
  `complementarity.html#c-api` anchor `c-api.md` links to exists.

### S1-I5 — `IpoptSolve` routes through the driver (commit that ticks S1-I5)

What moved:
- `src/lib.rs`: `LastSolve.mpcc: Option<MpccOutcome>` (`None` on every
  non-driver solve and every session solve). `IpoptSolve`: with pairs and a
  staged working set, refuse with `Invalid_Problem_Definition` (one
  `print_level`-gated line) **before** the working set is taken or the
  bridge built — no callback runs, and neither the set nor the pairs is
  consumed (so it refuses again until one is cleared; `IpoptSolveWarmStart`
  with an input set inherits this). After restoration wiring and
  `defer_end_verdict`, the solve is one `if`: pairs → `solve_mpcc(&mut app,
  bridge, pairs.clone(), narration)`, `describe_outcome` narrated, verdict
  released and printed with the driver's status, `(out.status,
  out.statistics, Some(out))`; else the old `optimize_tnlp` + ladder,
  moved into the `else` arm verbatim (re-indented; `ladder.statistics` is
  moved instead of cloned). The `LastSolve` fill and the copy-back are the
  shared tail. The driver's `Err` arm (narrated, verdict released, returns
  `Invalid_Problem_Definition`, `last_solve` stays `None`) is marked
  unreachable from C: its only source is `with_lift`'s `validate_pairs` on
  the bridge's bounds, which are the bounds the declaration was validated
  against.
- Live inspectors: a thread-local `IN_LOWERED_SOLVE`, set by a drop guard
  around `solve_mpcc` only; `GetIpoptCurrentIterate` and
  `GetIpoptCurrentViolations` return `FALSE` while it is set, before reading
  the handle. **Deviation from §3.5**, measured: the plan said the
  inspectors already return `FALSE` in a lowered solve; that holds only when
  `g`/`lambda` is requested. An `x`-only `GetIpoptCurrentIterate` returned
  `TRUE` on every fire of every lowered solve (the lowered `n` equals the
  caller's when nothing is lifted). The guard is load-bearing for that
  shape; the test checks x-only, x+g and violations.
- `IpoptGetWorkingSet` returns `FALSE` when `last_solve.mpcc.is_some()`.
  Measured **not** load-bearing on today's code: with the guard disabled
  the test still passes (the lowered set has `m + n_pairs` rows and fails
  the length check). Kept as stated in §3.5.
- `src/solver.rs`: `IpoptCreateSolver` returns `NULL` with pairs, after a
  `print_level`-gated line, **without** consuming or nulling the handle.
  `IpoptSolverSolve` guard (unreachable from C) returns
  `Invalid_Problem_Definition`; unit test `session_solve_refuses_pairs_planted_past_create`
  plants a pair on a session's problem, then clears it and solves.
- `MpccTnlp` forwarding, read and pinned (no wrapper change):
  `get_scaling_parameters` passes the bridge's user scaling for the
  caller's rows/variables and `1` for product rows/lifts;
  `get_list_of_nonlinear_variables` forwards the caller's mask plus every
  product variable. Stated in `pounce.h` (scaling) and pinned by tests.
- Docs: the five S1-I4 "in development" notes are gone (`pounce.h`,
  README, `c-api.md`, `complementarity.md`, CHANGELOG), replaced by the
  routing contract; `pounce.h` also documents it on `IpoptSolve`,
  `SetIntermediateCallback`, the inspectors block, `IpoptGetWorkingSet`,
  `IpoptSetWarmStartWorkingSet`, `IpoptCreateSolver`, and
  `SetIpoptProblemScaling`. No new prototype, no new `#define`.

Tests (`tests/complementarity_solve.rs`, new, 16; a `Probe` through
`user_data` counts eval_f calls, callback fires, iteration-0 starts and
records the inspectors' answers): pairs routed (status, `x` to the bit,
`g`, objective, `GetIpoptIterCount` = 12 = the reported solve's, 139 total
iterations, report writes with `iteration_count` 12, a second solve on the
same handle identical); **no pairs** and **pairs declared then cleared**
both pinned to the pre-change path, measured with the same test on
`5aa87107` before any code changed — `Solve_Succeeded`, 15 iterations,
objective `7.732176659546963e-9` and `x` to the bit, one start, 16 fires,
17 `eval_f` calls; session refused (handle intact, solves, then with pairs
cleared a session is created and consumes it); staged working set + pairs
refused with 0 `eval_f` calls and 0 fires, refused again, then solves once
cleared; the same working set without pairs solves on active-set SQP;
`IpoptSolveWarmStart` + pairs refused, output buffers untouched;
`IpoptGetWorkingSet` `TRUE` after a pair-less SQP solve, `FALSE` after a
driver solve on the same handle; inspectors `FALSE` on every fire of a
driver solve and `TRUE` on every fire of a plain one; a stop on fire 20
(inside the second lowered solve) ends the run with `User_Requested_Stop`,
two starts, no later fire; seven option rows (`complementarity_strategy`
both/exact/continuation, `complementarity_continuation` coupled/schedule,
`complementarity_lift` no/yes) each matching Python's reported `x` to the
bit and its total iterations (schedule: callback sees 145 against the
driver's 142 — three gh#983 scale-audit candidates inside lowered solves
fire the callback and are discarded); bad option values refused at set
time; `SetIpoptProblemScaling` + `user-scaling` under pairs matches
Python's `set_problem_scaling` to the bit (83 iterations against 139
unscaled); an L-BFGS nonlinear mask `{x}` under pairs gives the unmasked
run to the bit (product variables cover all three), while without pairs
the same mask changes the run (15 → 46 iterations). Before the code, 9 of
the 13 then-written tests failed on `5aa87107` (the stop test was then
rewritten because it passed there).

**Reference: Python, not the CLI, for the trajectory.** From the `.nl`
file's own start `(0, 0.5, 0.5)` the C API matches Python's
`Problem(complementarity=)` on the same callbacks exactly (point, `g`,
objective, `iter_count`, total iterations, every option row). The CLI on
`qpec_small.nl` agrees on the answer (`1e-6`) but not the trajectory: 191
iterations in all against 139, its first three lowered solves identical
(15, 24, 12) and the fourth 40 against 12. Both callback frontends agree,
so this is not an option the C side forgot; the CLI's `.nl` TNLP reports
model structure a callback bridge does not (linearity, derivative proofs),
the likely seat of the difference — not investigated further. **S1-I8
(CLI parity through the C API) will meet this**; flagged for the human.

Gate (all on the ticking commit's tree):
- `cargo test -p pounce-cinterface` — all green, 19 binaries (lib 53,
  `complementarity_solve` 16, `complementarity_declare` 34, `export_surface`
  13, `header_abi` 4, `c_examples_build_and_run` 4).
- `cargo clippy -p pounce-cinterface --all-targets -- -D warnings` — fails
  before this crate on the five pre-existing `pounce-common` lints (as in
  S1-I1/S1-I4). With `--no-deps -D warnings` the only `src/` hits are the
  two pre-existing ones (`fortran.rs:493`, `lib.rs` useless `vec!` now at
  `:4101`); `complementarity_solve.rs` has none. CI flags (`--no-deps -D
  clippy::correctness -D clippy::suspicious`) clean.
- `cargo fmt --all -- --check` clean; `mdbook build docs` (0.5.2) clean;
  `make ask-check` top-1 17/20, top-5 20/20, OK.
- Sweep: baseline `5aa87107` built `--release -p pounce-cli` in a `git
  worktree` with its own `CARGO_TARGET_DIR`; head = this tree's
  `target/release/pounce`. **Empty diff**, 282 lines each (141 exact, 141
  lbfgs, 14 `cc:`). Empty by construction: the CLI does not link
  `pounce-cinterface`; the no-pairs pins above are the evidence for the C
  path.

### S1-I6 — outcome getters (commit that ticks S1-I6)

What moved:
- `src/lib.rs`: `#[repr(C)] PounceComplementarityStats` and
  `PounceComplementaritySolve`, field for field as §3.4 (the header writes
  `n_pairs, n_biactive, n_neither` and the polish reals one per line; same
  layout). The 16 `POUNCE_CC_CLASS_*`, `_REPORTED_*`, `_B_VERDICT_*`,
  `_TEXT_*` constants. `complementarity_stats(&MpccOutcome)`,
  `complementarity_solve(&MpccSolveRecord)`, `complementarity_text(.., which)`
  with `report_info()`'s encodings, matched on the enums (exhaustive, so a
  new variant fails to compile) rather than on its strings; `copy_label`
  truncates and NUL-terminates. Three exports through `ffi_guard`:
  `GetPounceComplementarityStats`, `GetPounceComplementaritySolve`,
  `GetPounceComplementarityText` (snprintf: NULL buf or `buflen <= 0` is a
  pure size query; `-1` for NULL handle, unknown `which`, no driver record,
  absent text).
- **Sentinels, not in §3.4:** with no certificate, its counts and 0/1
  flags read `-1` (reals NaN); with no B-check, `b_n_biactive`,
  `b_branches_solved`, `b_enumerated` read `-1`; `polish_taken` reads `0`
  when `polish_ran` is `0`. The `PounceLinearSolverStats` precedent.
  Stated in `pounce.h`.
- **"Did not run" semantics, a choice to confirm at the ABI review:** the
  getters describe the most recent solve, like every post-solve accessor.
  Clearing the pairs after a driver solve does **not** erase the record;
  the next solve (without pairs) does, as does a refused solve (`IpoptSolve`
  clears `last_solve` up front). Both pinned.
- `pounce.h` (structs, defines, three prototypes, the sentinel and
  most-recent-solve contract), README, `c-api.md`, CHANGELOG `[Unreleased]`
  (one sentence; not in this item's Touch list).
- `tests/header_abi.rs`: the 16 new defines in the known-names list, held
  both ways.
- `tests/c_examples_build_and_run.rs`: `repr_c_structs_match_the_header_layout`
  compiles a C program against `pounce.h` that prints `sizeof` of each
  struct and `offsetof` **and** `sizeof` of every field; each is held to
  `size_of` / `offset_of!` / `size_of_val` on the Rust side, and the
  header's field list (parsed) to the Rust list, in order. Covers
  `PounceLinearSolverStats` too (unchanged, now guarded). Field widths were
  added after a mutation `ipindex relaxed` → `ipnumber relaxed` passed an
  offsets-only version (it widens into padding, no offset moves); with
  widths it fails, as does a renamed header field. Parser unit test added.

Tests (`tests/complementarity_getters.rs`, new, 13). Each certificate class
reached by a real solve through `IpoptSolve`, every value held **to the
bit** against Python's `info["complementarity"]` for the same callbacks,
start and options (generated from the venv's 0.16.0 extension; point,
`iter_count`, every struct field, every lowered solve's stage/tau/status/
iterations/product, the summary text):
- `S`: QpecSmall from `(0, 0.5, 0.5)` — answer, `both`, 6 solves, 139
  iterations (`GetIpoptIterCount` 12), no B-check, no polish.
- polish ran and taken: QpecSmall from the origin — 7 solves, last
  `branch polish`, `natural_residual` 0, no polish reason.
- `strategy == "exact"`: 4 solves, all `exact product`.
- `W`: ralph1 (callbacks) — weak fallback, `accepted_by` "", B verdict B
  (1 biactive, 2 branches, enumerated).
- `none`: infeasible_pair (callbacks; the pair's row is `y` itself rather
  than Pyomo's auxiliary) — verdict, 8 solves ending `l1 penalty`,
  residual NaN (Python `inf`, report `null`).
- `not searched`: new model, `min Σx`, 605 variables `x >= 0`, row 0 = `x_0`
  (duplicates a bound: multipliers a set), row 1 = `x_1 ⟂ x_2` — more than
  `MAX_SEARCHED_COLUMNS` sign-constrained columns; reported as an answer,
  `Solved_To_Acceptable_Level`, B verdict B.
- Not run: before any solve; after a pair-less solve; pairs cleared then
  re-solved; a refused solve (staged working set) after a driver solve —
  stats `FALSE` with the 0xA5-filled struct byte-identical, solve `FALSE`
  for `k = 0, 1, -1`, every text `-1`. NULL handle on all three, NULL
  output pointers; `k` = `n_solves`, `n_solves + 1`, `-1`, `MIN`, `MAX`.
  snprintf: size query, NULL buf with positive length, exact and roomy
  buffers (nothing past the NUL), truncation at `len`, 5, 2, a one-byte
  buffer (just the NUL), `buflen` 0 / -1 / `MIN` (nothing written),
  unknown `which` 3 / -1 / 99.
- **Unreached by any in-repo model, unit-tested on a synthetic
  `MpccOutcome` in `src/lib.rs`** (5 tests): `CLASS_ABSENT` (no
  certificate), `REPORTED_FEASIBLE_UNCERTIFIED`, `B_VERDICT_NOT_B`,
  `B_VERDICT_NOT_CHECKED` with its `B_REASON` text, a polish not taken with
  its `POLISH_REASON`, non-finite reals → NaN, an unrelaxed `branch NLP`
  record, label truncation; every class and reported code.
- Mutation check: mapping W to `CLASS_NONE` and an unrelaxed `tau` to 0
  fails 6 of the 13 integration tests.

Gate (all on the ticking commit's tree):
- `cargo test -p pounce-cinterface` — all green, 20 binaries (lib 58,
  `complementarity_getters` 13, `complementarity_solve` 16,
  `complementarity_declare` 34, `export_surface` 13, `header_abi` 4,
  `c_examples_build_and_run` 6).
- Export guard (`--test export_surface --test header_abi`) — `ok. 13
  passed` / `ok. 4 passed`.
- `cargo clippy -p pounce-cinterface --all-targets -- -D warnings` — fails
  before this crate on the five pre-existing `pounce-common` lints (as in
  S1-I1/I4/I5). CI flags (`--no-deps -D clippy::correctness -D
  clippy::suspicious`) clean. Under plain `--no-deps`, `src/` has only the
  pre-existing useless `vec!` (now `lib.rs:4485`) and `fortran.rs:493`; the
  new and extended tests carry only restriction-group `unwrap_used` /
  `expect_used` warns, as the crate's other tests do.
- `cargo fmt --all -- --check` clean; `mdbook build docs` (0.5.2) clean;
  `make ask-check` top-1 17/20, top-5 20/20, OK; `make c-api-examples`
  green.

### S1-I7 — `IpoptWriteSolveReport` carries the block (commit that ticks S1-I7)

What moved:
- `src/lib.rs`: `IpoptWriteSolveReport` adds `if let Some(out) =
  &last.mpcc { builder.set_complementarity(out.report_info()) }` — the
  conversion the CLI (`pounce-cli/src/main.rs`) and Python
  (`pounce-py/src/problem.rs::write_solve_report`) already share, so there
  is no second emitter. `ReportBuilder::finish` copies the block whatever
  the detail level, so `"summary"` carries it too. Doc comment, `pounce.h`,
  README, `c-api.md`, `schema/solve-report-v1.md` (the C API named as a
  producer) and one CHANGELOG sentence.
- Not changed, noticed: the C report still omits the `second_opinion`
  block the CLI and Python write when the ladder ran. Outside this item.
- Tests: the S1-I6 models and getter readers moved, unchanged, from
  `complementarity_getters.rs` to `tests/common/cc_models.rs`, shared with
  the new `tests/complementarity_report.rs` (10).
- `Cargo.toml`: dev-dependency `serde_json` gains `float_roundtrip`.
  Measured: without it serde_json reads `9.035978839861492e-24` (what the
  writer printed, Python's value) back as `...493e-24`, and 7 of the 10
  tests fail on 1-ulp reals. Parsing only; `Cargo.lock` unchanged.

Tests (`complementarity_report.rs`), on each class S1-I6 reached — `S`
(QpecSmall from `(0, 0.5, 0.5)`), polish taken (from the origin), `W`
(ralph1), `none` (infeasible_pair), `not searched` (605 variables) — plus
`complementarity_strategy=exact`:
- Getters vs block, same solve, **every** field: each
  `PounceComplementarityStats` field, each `PounceComplementaritySolve`
  record (stage, relaxed/tau, status by upstream name, iterations, product,
  `seconds` too, bitwise: both come from one record), and the three texts
  against `summary` / `polish.reason` / `b_stationarity.reason`; the getters'
  NaN / `-1` / `""` are the block's `null` or missing key. Every block key
  must be one of the compared ones, so a field added to `report_info()`
  without a getter fails here. **Agreed on all six runs; no disagreement
  found.** Also `solution.status_upstream` = the solve's status,
  `statistics.iteration_count` = `GetIpoptIterCount`.
- Python: the block minus `seconds` equals, as a JSON value, the block
  Python's `solve(report_path=, report_detail=)` wrote on the same callbacks
  and start (`tests/common/cc_python_report_blocks.json`, from the venv's
  editable 0.16.0 build of this branch; Python's `full` and `summary` blocks
  identical). **Equal on all five Python runs.** The CLI writes the block
  through the same `report_info()`; its qpec_small trajectory differs (191
  against 139 iterations, S1-I5) and is left to S1-I8.
- `summary` block == `full` block (seconds included).
- Absent: before any solve the write returns `FALSE` (no file); after a
  solve without pairs, no key at either level (what the CLI and Python do:
  Python checked on the same model); clearing the pairs keeps the block of
  the solve that used them (most-recent-solve, as the getters), the
  re-solve drops it; a refused solve (staged working set) after a driver
  solve leaves no report (`FALSE`, no file).
- With the `set_complementarity` line removed, 8 of 10 fail (the two that
  pass are the no-solve and no-pairs tests).

Gate (all on the ticking commit's tree):
- `cargo test -p pounce-cinterface` — all green, 21 binaries (lib 58,
  `complementarity_report` 10, `complementarity_getters` 13,
  `complementarity_solve` 16, `complementarity_declare` 34,
  `export_surface` 13, `header_abi` 4, `c_examples_build_and_run` 6).
- Export guard (`--test export_surface --test header_abi`) — `ok. 13
  passed` / `ok. 4 passed`.
- Clippy under the CI flags (`--all-targets --no-deps -D clippy::correctness
  -D clippy::suspicious`) clean; `cargo fmt --all -- --check` clean;
  `mdbook build docs` (0.5.2) clean; `make ask-check` top-1 17/20, top-5
  20/20, OK; `make c-api-examples` green.
- Sweep not run: no solver code moved, and the CLI does not link
  `pounce-cinterface`.

### S1-I8 — CLI parity over the fixture corpus (commit that ticks S1-I8)

**First: why the S1-I5 trajectory differed (191 vs 139 on `qpec_small`).**
Investigated before any test was written, as the human asked. Bisected in
Rust with `solve_mpcc` called directly on the `.nl` model's `NlTnlp` behind
a wrapper that forwards the required TNLP methods and switches each
optional one on or off (scaling suffixes, variable / objective /
constraint linearity, the nonlinear-variable list, derivative proofs,
metadata, `wants_intermediate_callback`), and that can substitute a
hand-written `f`, `g` or row bound:

| `NlTnlp` wrapped as | lowered solves | total |
|---|---|---|
| all optional methods on, or each one alone, or all off | 15, 36, 47, 32, 13, 46, 2 | **191** |
| hand-written `g = 2y1 - x` (the `.nl` form), `g_l = 1` | same | 191 |
| `.nl` `g` minus 1, `g_l = 0` | 15, 36, 87, 32, 12, 19, 2 | 203 |
| hand-written `g = 2y1 - 1 - x`, `g_l = 0` (QpecSmall's row) | 15, 36, 24, 32, 12, 20 | **139** |
| the same plus QpecSmall's `f = (x-1)^2 + ...` | 15, 36, 24, 32, 12, 20 | 139 |

**Root cause: the model encoding, not a frontend.** AMPL writes a row's
constant into its bound, so the `.nl` reader hands the solver `2 y1 - x >= 1`
and `x + 2 y2 >= 1` (`g(x0) = (1, 1)`, `g_l = (1, 1)`), where the
hand-written `QpecSmall` of `python/tests/test_complementarity.py` and
`pounce-cinterface/tests/common/qpec.rs` says `2 y1 - 1 - x >= 0`. The same
feasible set; `g`, and the products the driver forms from it, differ in the
last bits, and on this degenerate model the third Scholtes solve takes 47
iterations one way and 24 the other; a third rounding of the same row gives
203. No optional TNLP answer moves an exact-Hessian run, and no option the
CLI sets does either: with `NlTnlp` itself behind the C callbacks the C API
reproduces the CLI **bit for bit** at the default on every fixture. Not a
bug in the `.nl` frontend, the driver or the C API; a sensitivity of this
model's trajectory to rounding, which the plan's exact-iteration rule
assumed away by comparing "same model" without saying which encoding.

Two differences remain, both information the `.nl` carries and a callback
model expresses differently or not at all:
- **The nonlinear-variable list, under `hessian_approximation=limited-memory`.**
  `NlTnlp::get_list_of_nonlinear_variables` restricts the L-BFGS
  approximation to the variables the `.nl` header calls nonlinear; the C
  bridge treats every variable as nonlinear unless told. Passing the `.nl`
  list through `IpoptSetNonlinearVariables` makes the C API exact against
  the CLI on all three fixtures where it matters; Python's `Problem` has no
  such declaration, so Python matches the unmasked C run exactly.
  `qpec_small` is nonlinear in all three variables, so its limited-memory
  row is exact without the list.
- **The objective sense.** `NlTnlp` negates a maximize objective, so its
  callbacks minimize `-f`: the C API and Python report `-f*`, the CLI `f*`.
  `design_cent_3` (maximize) is exact up to that sign.

Comparison (`total` = the driver's total iterations; status `OK` =
`SolveSucceeded`, `Acc` = `SolvedToAcceptableLevel`, `Inf` =
`InfeasibleProblemDetected`; C / Python on `NlTnlp`'s callbacks from the
`.nl` start; Python = the venv's editable build of this tree):

| fixture, options | CLI | C (`.nl` list) | C (no list) | Python | class | objective (CLI) |
|---|---|---|---|---|---|---|
| qpec_small | OK 191 | OK 191 | = | OK 191 | S | 0 |
| ralph1 | OK 303 | OK 303 | = | OK 303 | W | 2.0e-9 |
| mixed_pair | OK 357 | OK 357 | = | OK 357 | W | -1.37e-8 |
| design_cent_3 | OK 69 | OK 69 | = | OK 69 | S | 3.7233690686801 (C, Python: its negation) |
| infeasible_pair | Inf 178 | Inf 178 | = | Inf 178 | none | 1.9841530 |
| opf_vv_lower_knee | OK 219 | OK 219 | = | OK 219 | W | -0.3954730634 |
| opf_vv_lower_knee_held | OK 182 | OK 182 | = | OK 182 | S | -0.3954779015 |
| qpec_small, strategy=exact | OK 171 | OK 171 | = | OK 171 | S | 0 |
| qpec_small, strategy=continuation | OK 100 | OK 100 | = | OK 100 | S | 0 |
| qpec_small, continuation=schedule | OK 215 | OK 215 | = | OK 215 | S | 0 |
| qpec_small, lift=yes | OK 181 | OK 181 | = | OK 181 | S | 2.2e-13 |
| qpec_small, lift=no | OK 191 | OK 191 | = | OK 191 | S | 0 |
| qpec_small, limited-memory | OK 128 | OK 128 | OK 128 | OK 128 | S | 0 |
| ralph1, limited-memory | Acc 584 | — | Acc 584 | — | W | (measured, not in the test) |
| mixed_pair, limited-memory | OK 373 | — | OK 373 | — | W | (measured, not in the test) |
| design_cent_3, limited-memory | OK 10967 | — | OK 10967 | — | S | (measured; too slow for the test) |
| infeasible_pair, limited-memory | Inf 3672 | Inf 3672 | Inf 3206 | Inf 3206 | none | verdict; last iterates differ by 6.2e-2 |
| opf_vv_lower_knee, limited-memory | Acc 748 | Acc 748 | OK 506 | OK 506 | W | -0.39547306 vs -0.39546824 (no list); `x` 4.9e-6; polish taken only without the list |
| opf_vv_lower_knee_held, limited-memory | OK 390 | OK 390 | OK 380 | OK 380 | S | `x` 5.3e-8; accepted by `exact product` (CLI) vs `Scholtes continuation` (no list) |

"=" : the list changes nothing under the exact Hessian (it is not read).
Every "C (`.nl` list)" cell equals the CLI to the bit: status, objective (in
the CLI's sense), point, `statistics.iteration_count` and the whole
`complementarity` block bar `seconds`. Every Python cell equals the
matching C cell the same way.

What moved:
- Root `Cargo.toml`: `pounce-cinterface` in `[workspace.dependencies]`
  (path + `version = "0.16.0"`). `crates/pounce-cli/Cargo.toml`: first
  `[dev-dependencies]` section, `pounce-cinterface.workspace = true`
  (kept in the published manifest — a versioned dev-dependency is not
  stripped — so `pounce-cinterface` must be live on crates.io before
  `pounce-cli` publishes; `scripts/publish-crates.sh` orders them so, and
  the release guard counts dev-dependency edges; corrected in the second
  review round). `Cargo.lock` gains
  the edge only. `scripts/check-release-consistency.sh` OK.
- `crates/pounce-cli/tests/common/nl_c_harness.rs` (new): `NlTnlp` behind
  five `extern "C"` trampolines (`user_data` = the harness), bounds and
  start from the TNLP, pairs through `SetPounceComplementarityPairs`,
  options typed by how the value parses, `print_level 0`, `IpoptSolve`,
  `GetIpoptIterCount`, `IpoptWriteSolveReport(.., "full")`;
  `solve_with(.., forward_nonlinear)` passes the `.nl` list through
  `IpoptSetNonlinearVariables`.
- `crates/pounce-cli/tests/cinterface_mpcc_parity.rs` (new, 16 tests, ~3 s):
  one `#[test]` per row of the table except the four "measured" rows. Each
  default and `qpec_small` option row: exact vs the CLI and exact vs
  Python, plus the branch it is listed for (class S/W/none; reported
  answer/weak_fallback/verdict; polish taken / declined with a reason / not
  run; B verdict `B`; `lifted > 0` at the default and on request; strategy
  `both`/`exact`/`continuation`; maximize). Limited-memory on the three
  list-sensitive fixtures: with the list, exact vs the CLI; without it,
  exact vs Python and the answer only vs the CLI (same class and reported
  kind, statuses equal or both converged; for a non-verdict, objective
  within `1e-4` relative and `x` within `1e-5`), **and** the block must
  differ from the CLI's — so the masked row stays evidence that the list is
  what restores parity. Mutation: with the list forwarding disabled the
  three masked rows fail (13 pass, 3 fail). `not searched` is reached by no
  fixture (stated in the test header; `complementarity_getters.rs` covers
  it through callbacks).
- `crates/pounce-cli/tests/common/cinterface_mpcc_python.json` +
  `gen_cinterface_mpcc_python.py` (new): the Python side, 16 rows, reals
  copied as the report writer printed them. **Deviation, noted:** the
  bitwise comparisons are between reports (C vs CLI vs Python), not between
  `IpoptSolve`'s buffers and a report: `pounce-cli`'s `serde_json` has no
  `float_roundtrip` (adding it would change the CLI binary's parser), so a
  printed real can read back one ulp off (`1.0000000019942585` read as
  `...583`). Equal text from the same writer is equal bits; the buffers are
  checked against the C report to `4 ulp`.
- Not done: the plan's per-solve `{stage, status, iterations}` comparison is
  subsumed by the whole-block equality.

Gate (all on the ticking commit's tree):
- `cargo test -p pounce-cli --test cinterface_mpcc_parity` — `ok. 16 passed`.
- `cargo test -p pounce-cli --test mpcc_complementarity` — `ok. 13 passed`.
- `cargo test -p pounce-cinterface` — all green, 20 binaries (lib,
  `complementarity_report` 10, `complementarity_getters` 13,
  `complementarity_solve` 16, `complementarity_declare` 34,
  `export_surface` 13, `header_abi` 4, `c_examples_build_and_run` 6, …).
- `cargo clippy -p pounce-cli --all-targets --no-deps -- -D
  clippy::correctness -D clippy::suspicious` clean; the new files carry
  only the restriction-group `unwrap_used` / `expect_used` warns other
  tests carry. `cargo fmt --all -- --check` clean.
- Sweep not run: no solver or CLI code moved (test-only change).

### S1-I9 — MOI-semantics helper (commit that ticks S1-I9)

What moved:
- `src/lib.rs`: `int AddPounceComplementsConstraint(IpoptProblem, ipindex
  row, ipindex var)` through `ffi_guard`, and 12 constants. Outcomes
  `POUNCE_COMPLEMENTS_PAIR_MIXED 0`, `_PAIR_LOWER 1`, `_PAIR_UPPER 2`,
  `_ROW_FIXED_NO_PAIR 3`, `_VAR_FIXED_NO_PAIR 4`; errors `_ERR_NULL -1`,
  `_ERR_OUT_OF_RANGE -2`, `_ERR_VAR_ALREADY_PAIRED -3`,
  `_ERR_ROW_ALREADY_PAIRED -4`, `_ERR_ROW_NOT_FREE -5`, **`_ERR_VAR_BOUNDS
  -6`** and **`_ERR_INTERNAL -7`** (the last two not in §3.1; see below).
  The shared negative values equal the `POUNCE_CC_*` ones (NULL, range,
  var, row, internal). Check order: NULL; range (`zero_based`, so
  `Index::MIN` in Fortran style cannot overflow); row free (`g_L <= -1e19
  && g_U >= 1e19`, so a NaN row bound is "not free"); variable, then row,
  not already used; `x`'s bounds neither NaN nor crossed; then the table.
  The table is a pure `complements_row(xl, xu)` (same `±1e19` thresholds
  as the classifier) plus `row_bounds`. The edit is written tentatively,
  then **every declared pair plus the new one** is re-validated through
  `declare_pairs` against the edited bounds (also for the no-pair
  `ROW_FIXED` edit, with an empty batch); on `Err` the two bounds are
  restored and `_ERR_INTERNAL` returned — carry-over (a). A `debug_assert!`
  checks the edited bounds classify as the native shape the row names.
- `IpoptProblemInfo.complements_used` (new, 0-based): every accepted
  `(row, var)`, pair or not, never cleared.
- `pounce.h`: table, check order, permanence, 12 defines, prototype; the
  §3.2 text "validated against the bounds given to `CreateIpoptProblem`"
  now says "the handle's bounds, as `AddPounceComplementsConstraint` may
  have edited them" (also in `lib.rs`, `complementarity.md`). README,
  `c-api.md`, `complementarity.md` ("From MOI / JuMP `Complements`", the
  five-row table and every refusal), CHANGELOG `[Unreleased]` (one sentence).
- `tests/header_abi.rs`: the define parser also reads `POUNCE_COMPLEMENTS_`;
  the 12 names in the known list, both directions (checked to fail with
  `_VAR_BOUNDS` set to `-8` in the header).

Design choices the plan did not settle, conservative, **for the ABI review**:
1. **Same row or variable in two calls** — refused whether or not the
   first call declared a pair: a `VAR_FIXED_NO_PAIR` row (still free) is
   `ROW_ALREADY_PAIRED` on reuse, a variable from a `ROW_FIXED` or
   `VAR_FIXED` call is `VAR_ALREADY_PAIRED`. That needs the
   `complements_used` ledger. Clear/Set do not empty it and do not undo
   edits, so the helper is a one-pass builder per handle. The native
   `AddPounceComplementarityPair` does not read the ledger (S1-I4's
   contract is unchanged): a free `VAR_FIXED` row can still be paired
   natively with a boxed variable.
2. **Row bounds that agree or conflict with MOI's implied sign** — both
   `ROW_NOT_FREE` (§3.1 says any non-free row), including `g_L = 0` with
   `x >= 0`. That is a native declaration. Because the free check comes
   first (plan order), a row in a **one-sided** pair reads
   `ROW_NOT_FREE`, not `ROW_ALREADY_PAIRED`; a row in a mixed pair (free)
   reads `ROW_ALREADY_PAIRED`.
3. **Crossed (`l > u`) or NaN variable bounds** match no row of the table
   → new `_ERR_VAR_BOUNDS`.
4. **`_ERR_INTERNAL`** covers a panic and the re-validation refusal. No
   public call reaches the second today: the edited row is in no pair, and
   no call edits a paired row or variable. Stage 2's in-place bound update
   would make it reachable.

Tests (`tests/complementarity_moi.rs`, new, 17). Model family `min (x-a)^2 +
(y-b)^2`, one row `F = cx x + cy y - s`, the pair `[F, x]`; answers worked
by branch in each test's comment and checked to `1e-6` (status, `x`, `F`,
objective) after `IpoptSolve`:
- mixed, `x ∈ [0,2]`, `F = x + y`: target `(1,1)` → `(0,1)` (lower end),
  target `(1.5,-5)` → `(2,-5)` (upper end); row stays free (reuse is
  `ROW_ALREADY_PAIRED`).
- lower, `x >= 1`, `F = x + y - 4`: target `(2,5)` → `(1,5)` (the pair
  decides; `F >= 0` alone keeps the target), target `(0,0)` → `(2,2)` (the
  sign decides; the signless pair would accept `(1,0)`, `F = -3`); reuse
  `ROW_NOT_FREE` proves the edit. **Also bit for bit against the native
  formulation** (`g_L = 0` at create + `AddPounceComplementarityPair`):
  `x`, `F`, objective, status, total iterations; and **Python's**
  `Problem(complementarity=[(0, 0)])` on the same model, start `(2,2)`:
  `x` and objective to the bit, total iterations 43 and 40 (the venv's
  editable build of this tree).
- upper, `x <= 2`, `F = x - y + 1`: the mirror, `(2,5)` and `(1,2)`.
- free `x`, `F = x + y - 3`: `ROW_FIXED_NO_PAIR`, no pair, driver not run;
  targets on both sides of `F = 0` → `(-0.5,3.5)`, `(1.5,1.5)`.
- fixed `x = 1`: `VAR_FIXED_NO_PAIR`; targets `(1,1)` → `F = -1` and
  `(1,5)` → `F = 3`, so neither sign was written.
- (a) QpecSmall with row 1 created free: native `(0, y1)`, then the helper
  on `(1, y2)` writes `g_L[1] = 0` (= QpecSmall): both pairs listed,
  `SetPounceComplementarityPairs` of the same set accepted (re-validation
  from outside against the edited bounds), solve equal **to the bit**
  (status, `x`, objective, `GetIpoptIterCount`) to native QpecSmall, at
  `(1,1,0)`. Unit test (`src/lib.rs`) plants a declared mixed pair whose
  box is then collapsed and checks `_ERR_INTERNAL` with bounds, pairs and
  ledger unchanged, then the same call succeeding once restored. Two more
  unit tests cover `complements_row` on every branch (including `±1e19`
  exactly) and that each pair row's edit classifies as its native shape.
- Errors, each on a handle that already holds a declaration, asserting the
  set unchanged and the probe row still free: NULL; out of range (−1, m/n,
  `MIN`/`MAX` on each index); Fortran style (0 and m+1 refused, 1-based
  accepted, read back 1-based); `ROW_NOT_FREE` for `[0,∞)`, `[1,∞)`,
  `(−∞,5]`, `(−∞,0]`, `[0,0]`, `[−1,1]`, `[NaN,∞)`; `VAR_ALREADY_PAIRED`
  (helper pair, native pair); `ROW_ALREADY_PAIRED` (helper mixed, native
  mixed); `VAR_BOUNDS` (crossed, NaN either side); check order; reuse after
  no-pair calls; Clear keeps the edit and the ledger.
- Mutations: `ROW_FIXED` writing only `g_L` fails the free-`x` test;
  `VAR_FIXED` writing `g_L = 0` fails the fixed-`x` test; no ledger push,
  or a check that ignores the ledger, fails 3.

Gate (all on the ticking commit's tree):
- `cargo test -p pounce-cinterface` — all green, 22 binaries (lib 61,
  `complementarity_moi` 17, `export_surface` 13, `header_abi` 4,
  `c_examples_build_and_run` 6: `sqp_warm_start.c` and the struct-layout
  program still build strict against the extended header).
- Export guard (`--test export_surface --test header_abi`): `ok. 13 passed` /
  `ok. 4 passed`. It failed first on the missing README and `c-api.md`
  mentions, as it should.
- Clippy, CI flags (`--all-targets --no-deps -D clippy::correctness -D
  clippy::suspicious`): clean, exit 0. Plain `-D warnings` still fails
  first in `pounce-common` (pre-existing, as before). Under plain
  `--no-deps`, the new test has only the restriction-group `unwrap_used`
  warns.
- `cargo fmt --all -- --check` clean; `mdbook build docs` (0.5.2) clean,
  anchor `complementarity.html#from-moi--jump-complements` exists and
  `c-api.md` links to it; `make ask-check` top-1 17/20, top-5 20/20, OK;
  `make c-api-examples` green; `scripts/check-release-consistency.sh` OK.
- Sweep not run: no solver code moved, and the CLI does not link
  `pounce-cinterface`.

### S1-I10 — C example, docs, ABI-review checklist (commit that ticks S1-I10)

What moved:
- `examples/complementarity.c` (new): QpecSmall in C with row 1 created
  **free**. Declares `(0, y1)` natively (`AddPounceComplementarityPair`) and
  `(1, y2)` through `AddPounceComplementsConstraint` (`PAIR_LOWER`, writes
  `g_L[1] = 0`, so the model is QpecSmall exactly); checks
  `GetPounceComplementarityStats` is `FALSE` before any solve, three
  refusals (`POUNCE_CC_ERR_VAR_ALREADY` for `y1` twice,
  `_ERR_OUT_OF_RANGE` for row `m`, `POUNCE_COMPLEMENTS_ERR_ROW_NOT_FREE` on
  the edited row), the read-back of both pairs, `IpoptCreateSolver`
  returning `NULL` without consuming the handle; solves from `(0, 0.5,
  0.5)`; requires `Solve_Succeeded`, `x = (1, 1, 0)` to `1e-6`,
  `REPORTED_ANSWER`, `CLASS_S`, 2 pairs; prints every lowered-solve record,
  checks `k = n_solves` is refused, reads the summary through a size query,
  checks `B_REASON` is `-1`; writes the `summary` report to
  `argv[0].report.json` (next to the binary, so under `target/`) and checks
  it has a `"complementarity"` key. Last line, registered in
  `EXPECTED_STDOUT`: `complementarity example: S-stationary answer (1, 1, 0)
  certified`. Measured output: 6 lowered solves, 139 iterations in all,
  reported solve 12 — S1-I5's numbers for this start. Builds clean under
  the strict flags with gcc and with clang (`make c-api-examples CC=clang`).
- Docs accuracy pass. Corrected or added:
  - `c-api.md`: the getters' "nothing" cases listed only "before any solve,
    or a solve without pairs" — now also a refused solve (which discards
    the record), and that clearing the pairs keeps the record until the
    next solve; the error list lacked `POUNCE_CC_ERR_INTERNAL`; the
    struct-layout sentence named only `PounceLinearSolverStats`. The
    complementarity material, scattered over the Lifecycle and Solving
    bullets, is now one *Complementarity* section: prototypes, both code
    tables with values, the MOI table, the check order, `ROW_NOT_FREE` for
    agreeing bounds, the permanent edits and ledger, the three refusals
    (sessions, staged working sets, inspectors), scaling / nonlinear-list
    forwarding, the getters and sentinels, and **Matching the CLI** (the
    row constant in the bound; the `.nl` nonlinear-variable list under
    L-BFGS; the maximize sign), from S1-I8's measurements. The report
    section now says the C report lacks the `second_opinion` block.
  - `complementarity.md`: `ERR_INTERNAL` in the C code list; the C getters
    and report after the solve paragraph, with a pointer to *Matching the
    CLI*; the `Complements` ledger "even if the pairs have since been
    cleared"; `GetPounceComplementarityStats` as a place to read the
    certificate; a *Limits* bullet for the C API (sessions, SQP working
    sets, inspectors, Fortran shim).
  - `pounce.h`: an over-long reflowed comment line; the declared-pairs block
    now says the Fortran shim declares none and gives the two trajectory
    notes with a pointer to `c-api.md`; `ClearPounceComplementarityPairs`
    says bound edits and the outcome record stay. Comments only.
  - Crate README: an example list, `ERR_INTERNAL`, `ROW_NOT_FREE` for
    agreeing bounds, edits and ledger surviving a clear, the getters'
    most-recent-solve contract, working sets not combined with pairs, the
    *Status* paragraph (driver path and the trajectory note), the three
    guarded structs under *Versioning*.
  - `julia.md`: 0.17 adds symbols and changes none, but Pounce.jl refuses it
    until the review widens the gate. No Julia code changed.
  - `CHANGELOG.md`: the one long "(in progress)" paragraph accumulated over
    I4–I9 is one entry with five sub-bullets (declaring, MOI, solving,
    reading back, example/docs), plus an *Added* entry for the C-examples
    build and the export guard and a *Fixed* entry for the header version
    defines.
- §8 written: nine functions, 36 defines, two structs with measured
  offsets, the changed version defines, 13 decisions to confirm, and the
  release / Pounce.jl steps (`_check_native_version` is `0.15.0 <= v <
  0.17.0-0` today; widen to `< 0.18.0-0` after review).

Gate (all on the ticking commit's tree):
- `cargo test -p pounce-cinterface` — all green, 22 binaries, 0 failures
  (lib 61, `c_examples_build_and_run` 6 — both examples build strict and
  print their lines —, `export_surface` 13, `header_abi` 4,
  `complementarity_*` 34/13/17/10/16).
- `make book` (mdbook 0.5.2) — builds; anchors `c-api.html#complementarity`,
  `#matching-the-cli`, `#moi-f-x--complements`,
  `complementarity.html#declaring-pairs` and `#what-you-get-back` exist.
- `scripts/check-release-consistency.sh` — `check-release-consistency: OK`,
  exit 0.
- `make ask-check` — top-1 17/20, top-5 20/20, OK (unchanged).
- `make c-api-examples` (gcc, and `CC=clang`) green;
  `cargo build --release -p pounce-cinterface` produces the cdylib.
- `cargo fmt --all -- --check` clean; clippy under the CI flags
  (`--all-targets --no-deps -D clippy::correctness -D clippy::suspicious`)
  clean.
- Sweep not run: no Rust source outside a test moved.

### S1-I11 — Stage-1 closing gate (commit that ticks S1-I11)

Binaries: baseline `pounce` built `--release -p pounce-cli` from `2ac0ad9`
(main before this branch) in its own `git worktree` and `CARGO_TARGET_DIR`;
head `pounce` from `cargo build --release` on `363c05d` (S1-I10's tip; this
commit changes test files and this plan only, so the binary is the same).
Both report `pounce 0.16.0`.

Gate results:
- **Sweep** (`scripts/sweep-fixtures.sh <bin> <out>`, both legs): `diff`
  **empty**, exit 0. 282 lines each: 141 `exact`, 141 `lbfgs`, of which 14
  are `cc:` lines (7 per leg: the six named in `CLAUDE.md` plus
  `opf_vv_lower_knee_held`); no `NO_JSON`. Engine, `mu=` and `cc=` columns
  included.
- **Minima sweep** (`scripts/sweep-minima.sh`; not in the gate, run because
  `pounce-nlp` moved): `diff` empty, 54 lines each.
- `make check` — exit 0, head and baseline.
- `make fmt-check` — clean, head and baseline.
- `make clippy` (`--workspace --all-targets -- -D warnings`) — **fails on
  head and on baseline, pre-existing.** Head aborts in `pounce-common`
  (5 errors: `expect_used` `diagnostics.rs:577`; `collapsible_if`
  `options_list.rs:303,335,367`, `pd_perturbation.rs:587`). Baseline aborts
  with the same 5 plus 8 `collapsible_if` in `pounce-studio-core`
  (`markdown.rs`, `preflight.rs`), which head simply did not reach before
  the abort (parallel job order). Head's error set is a subset of the
  baseline's. Because the abort hides every downstream crate, the real
  comparison is the full warning set: `cargo clippy --workspace
  --all-targets --message-format=json` (no `-D`) on both trees, keyed by
  (lint, file, message, span text). Baseline 3244 distinct diagnostics; head
  first 3352: **109 new**, all `unwrap_used` / `expect_used` (restriction
  group, `warn` in `Cargo.toml`) in seven test crates this branch added
  (`complementarity_report` 36, `cinterface_mpcc_parity` 29,
  `c_examples_build_and_run` 20, `complementarity_getters` 12,
  `export_surface` 9, `entry_point_error_paths` 7, `complementarity_moi` 2),
  and 1 gone (`collapsible_if` in `pounce-nlp/src/mpcc.rs`, removed by
  S1-I1). Ours, so fixed: `#![allow(clippy::unwrap_used,
  clippy::expect_used)]` at each of the seven crate roots (the spelling four
  existing test files use). After: head 3243, **0 new**, 1 gone — head's
  warnings are a strict subset of the baseline's. No lint in non-test code
  was ever new.
- CI's clippy steps (`ci.yml`: `cargo clippy --workspace --exclude
  pounce-hsl --all-targets -- -D clippy::correctness -D clippy::suspicious`,
  and the same flags on `-p pounce-rs --features full --all-targets`) —
  exit 0 on head, before and after the fix.
- `cargo test --workspace --exclude pounce-hsl` (from a clean
  `target/debug`, `CARGO_PROFILE_DEV_DEBUG=0` for disk): 5328 passed, 11
  ignored, **3 failed** — all of `c_examples_build_and_run` (S1-I3's), the
  cdylib not found. From a clean target under `--workspace`, cargo writes
  `libpounce_cinterface.so` to `deps/` only and does not uplift it to
  `target/debug/` (the lib is not a root unit once `pounce-cli`
  dev-depends on it, S1-I8); CI never saw it because its *Build* step runs
  first, and every earlier iteration ran `-p pounce-cinterface`, which
  uplifts. Not on baseline (the test is new). Fixed: `lib_dir()` looks in
  `deps/` first (where the cdylib always is, next to the test binary;
  the uplifted file is a hard link to it), then the profile directory, and
  still panics, never skips, when neither has it. Re-run in the failing
  configuration (`cargo test --workspace --exclude pounce-hsl --test
  c_examples_build_and_run`, no uplifted `.so` present): `ok. 6 passed`.
  `cargo test -p pounce-cinterface`: 22 binaries, 230 passed, 0 failed.
- `make python-test` (extension and CLI binary rebuilt from this tree by
  its `python-ext` prerequisite): first run stopped at collection on
  `jsonschema` and `jax` missing from the venv (environment, not code);
  with what `ci.yml` installs (`jax[cpu] diffrax jsonschema`) added, `cd
  python && python -m pytest tests -q`: **2016 passed, 35 skipped**, 0
  failed.
- `make ask-check` — top-1 17/20 (floor 14), top-5 20/20 (floor 18), OK.
- `make book` (mdbook 0.5.2) — builds, no warnings.
- `scripts/check-release-consistency.sh` — `check-release-consistency: OK`.
- `make c-api-examples` — both examples build strict and print their lines.

Branch review (`git diff 2ac0ad9..HEAD`, 38 files, +14851/-121 before this
commit): Stage 1 only — `pounce-cinterface` (src, header, README, tests,
examples), `pounce-nlp/src/mpcc.rs`, `pounce-cli` (dev-dependency and two
test files plus fixtures data), docs, CHANGELOG, `Makefile`
(`c-api-examples`), release guard and its skill note, this plan. No Stage
2-4 code. The `eprintln!`s added in `src/` are the deliberate refusal
diagnostics, gated on `narration_is_wanted`; no `dbg!`/`todo!`/`FIXME`. No
AI model name in code or commit text beyond the attribution trailers. The
export guard (`export_surface` 13, `header_abi` 4) is green over all nine
new functions.

Manual runs still owed (not available to the loop): §8.6's data-root
suites (`make -C benchmarks mpcc-opf-run`, `capacity-equilibria-run`,
`mpcc-compare`) and the other declared-pair corpora in `CLAUDE.md` — only
if review finds `pounce-nlp` changed beyond the classifier move; both
sweeps say it did not.

### S1-I12 — `IpoptWriteSolveReport` carries the `second_opinion` block (commit that ticks S1-I12)

Approved by the human after S1-I11 as an extra Stage-1 item (§8.5's open
`second_opinion` question).

What moved:
- `src/lib.rs`: `LastSolve` gains `second_opinion:
  Option<SecondOpinionOutcome>`. `IpoptSolve`'s path without pairs keeps the
  whole ladder outcome (it took `status` and `statistics` and dropped the
  rest); the driver path and the session solve (`src/solver.rs`) store
  `None`. `IpoptWriteSolveReport` adds `if let Some(so) =
  &last.second_opinion && so.ran() { builder.set_second_opinion(
  so.report_info()) }`: the CLI's (`pounce-cli/src/main.rs`, `if
  second_opinion.ran()`) and Python's (`pounce-py/src/problem.rs::
  write_solve_report`) test and conversion, so no second emitter.
  `ReportBuilder::finish` copies the block at every detail level, as it does
  for the CLI and Python. Solve behaviour unchanged: the ladder already ran,
  its status and statistics are still what ships and what the getters read.
- Other entry points: `IpoptSolveWarmStart` reaches the ladder only through
  `IpoptSolve`, so it is covered with no code of its own; `IpoptSolverSolve`
  never ran the ladder and stores `None`. No other call site of
  `run_second_opinion_ladder` in the crate.
- Pairs: the CLI skips the ladder when the driver ran (`mpcc_outcome.is_none()`
  in its gate) and Python returns `SecondOpinionOutcome::unchanged`
  (`ran()` false), so neither writes a block on a driver solve; the C path
  stores `None` there, the same behaviour.
- No new exported symbol, no struct or `#define` change; export guard green.
- Docs: `docs/src/c-api.md` ("Solve report": the "does not yet carry"
  sentence S1-I10 added is replaced by what the block is and when it is
  absent), `pounce.h` and the Rust doc comment on `IpoptWriteSolveReport`,
  the crate README, `docs/src/schema/solve-report-v1.md` (the C API named as
  a producer, and "never with declared pairs"), one CHANGELOG bullet under
  the gh#1147 entry; §8.5's item marked resolved. The S1-I7 and S1-I10 log
  entries above are left as written (history).

Tests (`tests/second_opinion_report.rs`, 5; Python reference in
`tests/common/second_opinion_python_reports.json`, from the scratchpad
venv's 0.16.0 extension built from this branch, whose Python path this item
does not touch; Python's `full` and `summary` blocks identical):
- (1) rescued: `degenerate_start_hs008` (the CLI's gh#850 model,
  `issue850_second_opinion_is_recorded.rs`) as callbacks with its exact
  Hessian, free variables, from the origin: base
  `Infeasible_Problem_Detected`/9, rungs `feral_scaling=mc64` 9,
  `mu_strategy=adaptive` 7, `start_point_perturbation=1e-2` 5 (promoted),
  total 30, reported `Solve_Succeeded`/5 — the same numbers as the CLI on
  the `.nl` file. The C block equals Python's as a JSON value at both
  levels (the block has no timing field, so nothing is stripped); the
  report's status and `iteration_count` equal Python's;
  `statistics.iteration_count` = `GetIpoptIterCount` = the promoted rung's
  entry of `rung_iteration_counts`; `total = base + Σ rungs`; `base_status`
  differs from the shipped status.
- (2) every rung fails: Python's `NoRealSolution` (`x² + 1 = 0`) with its
  exact Hessian, from 0.5: block present, no `promoted_by`, `base_status` =
  the shipped status, reported count = `base_iteration_count` (17), three
  rungs (17, 14, 26). Equal to Python's block on this platform; the test
  compares whole only when the rung lists agree, because rung 4 is
  platform arithmetic (`python/tests/test_second_opinion.py::
  _expected_rungs`), and always checks the three fixed rungs lead.
- (3) converged (HS008 from `(3, 2)`, 5 iterations): no key at either level.
- (4) pairs (`infeasible_pair`): the driver's verdict is
  `Infeasible_Problem_Detected`, a status that opens the ladder on the
  path without pairs, and the report has `complementarity` and no
  `second_opinion` at either level; Python's report on the same model has
  the same two facts.
- (5) no leak: HS008 rescued from the origin, then re-solved from `(3, 2)`
  on the same handle: the second report has no block, and its iteration
  count (5) is Python's for the same sequence on one `Problem`.
- Without the change (`src/` stashed, tests kept): `2 passed; 3 failed` —
  (1), (2) and (5) panic at `no second_opinion block in the C report`;
  (3) and (4) pass, being the absent branches.

Gate (all on the ticking commit's tree):
- `cargo test -p pounce-cinterface` — all green, 22 binaries, 235 passed
  (lib 61, `second_opinion_report` 5, `complementarity_report` 10,
  `export_surface` 13, `header_abi` 4, `c_examples_build_and_run` 6, ...).
- `cargo test -p pounce-cli --test cinterface_mpcc_parity` — `ok. 16 passed`.
- Export guard (`export_surface` 13, `header_abi` 4) green within the above.
- `cargo clippy -p pounce-cinterface --all-targets --no-deps -- -D
  clippy::correctness -D clippy::suspicious` — exit 0 (only pre-existing
  warnings in other test files; none in `second_opinion_report.rs`).
- `cargo fmt --all -- --check` clean; `make book` (mdbook 0.5.2) builds, no
  warnings; `make ask-check` top-1 17/20 (floor 14), top-5 20/20 (floor
  18), OK; `make c-api-examples` green.
- Sweep not run, and not needed: no solver code moved, and the CLI binary
  the sweep runs does not link `pounce-cinterface` (it is a dev-dependency
  of `pounce-cli`, S1-I8, i.e. test binaries only).

### S3-I1 — convex pipeline lifted into `pounce_convex::pipeline` (commit that ticks S3-I1)

Approved by the human after Stage 1, on this branch.

What moved (pure move; no solve-behaviour change):
- New `crates/pounce-convex/src/pipeline.rs`, re-exported at the crate
  root: `solve_presolved`, `demote_unverified_optimal`, `PresolveReport`
  (fields now `pub`), `EngineSwitches` (fields `pub`, `check`/`apply`), and
  the PSD screen as `check_ipm_inertia(HessianInertia)`. Bodies are the
  bindings' own.
- The two `PyResult` screens now return `Result<(), PipelineError>`. The
  enum's `Display` is the old message minus the `"solve_qp: "` prefix, and
  `pounce-py` adds that prefix back (`pipeline_err`). Before the edit, the
  five literals on both sides were compared after Rust's line-continuation
  rule: identical. `check` takes `active_set: bool` (Python passes
  `method == "active-set"`), not the method string. So the check order
  is unchanged, including `method="bogus", qp_reg=-2` reporting `qp_reg`
  first.
- `pounce-py/src/qp.rs` keeps the Python-only pieces: dict building
  (`attach_engine_reports`), `parse_hessian_inertia`, `check_tau`, option
  parsing.
- Not moved: Python's eigenvalue PSD verdict (`python/pounce/qp.py::
  _psd_verdict`, numpy/LDL). It is a Python implementation, not Rust code in
  `pounce-py`, so porting it would not be a move, and Python could not stay
  bit-identical. The Rust-side verdict that S3-I2 can reuse already exists:
  `pounce_convex::certify_psd_lower_triangle` (the CLI's dispatch uses it).
  §7's design bullet ("if only Python checks PSD") is therefore answered
  without a lift.
- CLI untouched: `crates/pounce-cli/src/dispatch.rs` imports only `Triplet`
  and `certify_psd_lower_triangle` from `pounce_convex`. Nothing in
  `crates/pounce-cli/src` names a `pipeline` item, and the diff does not
  touch `pounce-cli`. `run_convex_qp` stays a separate pipeline.

Tests (`pipeline::tests`, 10, one fixture per branch):
- `demote_unverified_optimal`: demoted (HSDE, `Optimal`, KKT above tol at
  `x = (5, 5)`); kept on each failed conjunct (point passes KKT; HSDE off;
  status not `Optimal`).
- `solve_presolved`: `reduced` (fixed column turns the row into a bound;
  the engine sees 0 rows; postsolved to the original size; `obj = −0.5`);
  `infeasible` and `unbounded` (the engine closure panics if it is called).
- `EngineSwitches::check`: every refusal and each pass. `apply`: unset
  leaves the defaults alone, set writes each field. `check_ipm_inertia`:
  `Psd` and `Unknown` pass, `Indefinite` is refused.

Python bit-identity, measured directly as well as through the suites. A
script (scratchpad, not committed) dumps `solve_qp` results as hex floats
over 7 models (reduced, infeasible, unbounded, 4 random convex QPs) × both
engines × `qp_presolve` {None, True} × `qp_hsde` {unset, True, False}, plus
every refusal message. I ran it once on a wheel built from the parent
commit and once on this tree: 147 lines, `cmp` identical.

Gate (all on the ticking commit's tree):
- Fixture sweep, release baseline from the parent commit in its own
  worktree and target dir against this tree: **diff empty**, engine column
  included. exact leg 141 lines (52 `cvx-qp`, 15 `cvx-qcqp`, 74 `nlp`),
  lbfgs leg 141 lines (the same split); `cc:` lines included.
- `cargo test -p pounce-convex -p pounce-py`: 81 binaries, 818 passed,
  0 failed. This includes `issue414_cost_normalized_false_optimal` and
  `illconditioned_huge_scale`, run although the move does not touch
  `normalized_optimum_is_genuine` / `hsde_cost_scale`.
- `test_qp_host.py`, `test_socp_jax.py`, `test_qp_sensitivity.py`: 129
  passed. `make python-test` (extension rebuilt from this tree): 2016 passed,
  36 skipped, 0 failed.
- `cargo clippy --workspace --exclude pounce-hsl --all-targets -- -D
  clippy::correctness -D clippy::suspicious` (CI's flags): exit 0, and no
  warning in `pipeline.rs`. `pounce-py/src/qp.rs` shows only the existing
  `field_reassign_with_default` in `opts_tau`, which this item does not
  touch.
- `cargo fmt --all -- --check` clean.
- `cargo test -p pounce-cinterface` (loop rule 4): 23 binaries, 235 passed.
- `benchmarks/qp` not run: no convex line moved.

### S3-I2 — `PounceQp*` handle, builders, solve, getters, report (commit that ticks S3-I2)

Approved by the human with Stage 3, on this branch.

What moved:
- New `crates/pounce-cinterface/src/qp.rs` (`pub use qp::*` from `lib.rs`);
  `pounce-convex` and `pounce-feral` become direct dependencies (both were
  already in the closure through `pounce-algorithm`; no new crate).
- 15 exports, all through `ffi_guard`: `PounceCreateQp(n, m_eq, m_ineq,
  index_style)` (NULL unless `n ≥ 1`, `m_* ≥ 0`, style 0/1),
  `PounceFreeQp` (NULL no-op), `PounceQpSetObjective` / `SetEqualities` /
  `SetInequalities` (triplets in the handle's index style; `P` lower
  triangle; validate-then-write, FALSE changes nothing),
  `PounceQpSetBounds` (NULL side = unbounded; `≤ -1e19` / `≥ 1e19` = no
  bound, the Ipopt convention; NaN refused), `PounceQpAddCone` (returns the
  cone's position or `POUNCE_QP_ERR_*`), `PounceQpSetNumOption` /
  `SetStrOption`, `PounceQpSolve`, `PounceQpGetSolution`,
  `PounceQpGetResiduals`, `PounceQpGetCertificate`,
  `PounceQpWriteSolveReport`. Exact prototypes in `pounce.h`.
- 2 `#[repr(C)]` structs: `PounceQpResiduals` (8 `ipnumber`: Python's
  `residuals` block, floor-read and `_raw`), `PounceQpCertificate`
  (`ipindex kind; ipnumber scale, tau, kappa`: Python's
  `certificate_scale`, `tau`, `kappa`). 22 defines: `POUNCE_QP_STATUS_*`
  (0–6, one per `QpStatus`), `POUNCE_QP_ERR_*` (−1…−7: NULL, ARGUMENT,
  INCOMPLETE, CONES, OPTIONS, INDEFINITE, INTERNAL), `POUNCE_CONE_*` (0–4),
  `POUNCE_QP_CERT_*` (0–2).
- Pipeline: no cones → `solve_qp`'s (`EngineSwitches::check` with
  `warm = false`, `check_ipm_inertia`, IPM or `solve_presolved` +
  `demote_unverified_optimal`, or the active-set engine with the inertia,
  optionally presolved); any cone → `solve_socp_ipm`, as `solve_socp`. All
  from `pounce_convex::pipeline`; the only re-stated code is Python's
  three-line `opts_tau` rule (bare `tau` lifts `tau_max`) and its residual
  block assembly, both private to `pounce-py`.
- **Option names, a deviation from §7's sketch:** the switches are spelled
  `qp_presolve`, `qp_hsde`, `qp_equilibrate`, `qp_crossover` (the CLI's and
  Python's names, like `qp_reg` in the sketch), not bare `presolve` etc.
- **Conic path refusals (choice):** with cones, `method=active-set`, `tau`,
  `tau_max` and every `qp_*` switch are `POUNCE_QP_ERR_OPTIONS`, because
  Python's `solve_socp` takes none of them; refused rather than accepted
  and possibly dropped.
- **Getters:** describe the most recent `PounceQpSolve` that returned a
  status. `PounceQpSolve` discards the previous result first, so before a
  solve and after a refused one every getter and the writer return FALSE and
  write nothing; a builder call after a solve does not erase it (pinned).
  `GetSolution` skips NULL outputs; the struct getters refuse a NULL `out`.
  `GetCertificate` is TRUE after every solve; `kind` says whether a ray is
  present.
- **Report:** `pounce.solve-report/v1` via `ReportBuilder`, input
  `TnlpDirect`, `n_constraints = m_eq + m_ineq`, `lambda = [y; z]`, status
  through the CLI's `qp_status_to_ars` mapping (re-stated: it is private to
  `pounce-cli/src/main.rs`; a unit test pins every variant and that the AMPL
  codes equal the CLI's `convex_status_report`). `solution.engine` is
  `cvx-qp`, `qp-active-set`, or **`cvx-qcqp` for any cone** — the conic IPM
  the CLI's QCQP arm also runs; there is no engine string for a
  matrix-form conic solve, and exp/power cones run its non-symmetric driver.
  A choice to confirm.

How PSD is decided (carry-over b): `hessian_inertia=auto` (default): empty
`P` is PSD; otherwise `pounce_convex::certify_psd_lower_triangle` at the
CLI's band `1e-9·min(max|P_ij|, 1)` (`dispatch.rs::psd_band`, re-stated:
private). Certified → `Psd`; not certified (including a factorization that
cannot decide) → `Indefinite`: refused under `ipm`, handed to the active-set
engine as indefinite. `psd` / `indefinite` / `unknown` are caller claims used
as given (Python's binding parameter). Python's numpy verdict (relative
`1e-8·max|P_ij|`, dense eigenvalues for `n ≤ 1500`, "undecided" → `unknown`)
is not ported, so on a borderline or badly scaled rank-deficient `P` the C
entry can refuse what Python accepts. Documented in `pounce.h`, README,
`c-api.md`. S3-I3's Python parity should pick models away from that border.

Tests (`tests/qp_handle.rs`, new, 14; `src/qp.rs` unit tests, 2):
- LP `min −x0 − 2x1`, `x0 − x1 = 0`, `x0 + x1 ≤ 1`, `x ≥ 0` → `x = (½, ½)`,
  `obj = −3/2`, `y = −½`, `z = 3/2`: on the HSDE IPM, presolve + crossover,
  the direct driver (`tau`, `kappa` NaN), the active-set engine.
- Convex QP, 1-based triplets, `min ½‖x‖² − x0 − x1`, `x0 + x1 ≤ 1`,
  `0 ≤ x ≤ 10` → `(½, ½)`, `−3/4`, `z = ½`: IPM; active-set presolved.
- SOC distance from `(2, −1)` to `x1 + x2 = 0` → `t = 1/√2`,
  `x = (3/2, −3/2)`; exp-cone GP from `solve_socp`'s docstring → 2.
- Certificates: primal infeasible (kind 1, finite scale), unbounded (kind 2,
  scale = |x|). Indefinite `P` refused under `ipm` (auto and claimed),
  solved under active-set to `−½`; a coupled PSD `P` certified.
- Every refusal code of `PounceQpSolve` (NULL, INCOMPLETE ×2, CONES ×2,
  OPTIONS ×3, INDEFINITE) and that a refusal discards the previous result;
  builder refusals change nothing; `AddCone` ARGUMENT ×9; option refusals;
  `max_iter=1` reaches the engine (`ITERATION_LIMIT`).
- Every getter: NULL handle, before any solve, after a refused solve →
  FALSE, outputs untouched.
- Report: NULL handle / no solve / NULL path / bad detail / bad directory →
  FALSE; at NULL, summary and full detail it deserializes as a
  `SolveReport` with the schema, dimensions, engine, `Solve_Succeeded`,
  `x`, `lambda`, objective, iterations and residuals equal to the getters'
  bitwise; engine `qp-active-set` and `cvx-qcqp` on those paths;
  `Diverging_Iterates` / 300 on an unbounded model.
- Mutations: `Ok(false) → Psd` fails the indefinite test; `ipindex kind` →
  `ipnumber` fails the layout check; `POUNCE_CONE_PSD 4 → 5` fails
  `header_abi`.
- Guards extended: `export_surface.rs` scans `src/qp.rs` (exports and its
  `#[cfg(test)]` part) and knows the `PounceQp` handle type;
  `header_abi.rs` gains `header_qp_codes_match_the_implementation` (22
  names, both ways); the C layout check covers the two new structs.

Gate (all on the ticking commit's tree):
- `cargo test -p pounce-cinterface`: 24 binaries, all green (lib 64,
  `qp_handle` 14, `export_surface` 13, `header_abi` 5,
  `c_examples_build_and_run` 6).
- `cargo clippy --workspace --exclude pounce-hsl --all-targets -- -D
  clippy::correctness -D clippy::suspicious`: exit 0. Plain `--no-deps`
  clippy on this crate: no warning in any file this item touched.
- `cargo fmt --all -- --check` clean; `make book` (mdbook 0.5.2) clean;
  `make ask-check` top-1 17/20, top-5 20/20, OK; `make c-api-examples`
  green (`all 4 solves succeeded`); `scripts/check-release-consistency.sh`
  OK.
- No sweep: `git diff` touches nothing under `crates/pounce-convex`,
  `crates/pounce-cli` or `crates/pounce-py`; the CLI does not link this
  crate. Existing solve paths unchanged.

### S3-I3 — cone, refusal and status tests; Python parity (commit that ticks S3-I3)

Approved by the human with Stage 3, on this branch. Tests only: nothing
under `src/`, `pounce-convex`, `pounce-cli` or `pounce-py` changed, and no
defect was found in S3-I2's handle (every new test passed on its first run;
the mutations below show they bite).

What was added (`crates/pounce-cinterface/tests/`):
- `common/qp_c.rs`: the shared driver (self-freeing handle, one wrapper per
  export).
- `qp_cones_and_errors.rs` (63 tests), the handle alone:
  - one model per cone kind with a hand-known optimum: nonneg (`(1, 2)`,
    `−3`, `z = (1, 1)`), SOC with `P = I` (projection onto `Q³`,
    `test_socp.py:15`), exp alone (`z = e`, `tau`/`kappa` NaN on the
    non-symmetric driver), power `α = ½` (`x0 = 1`, `test_socp.py:115`) and
    `α = 0.01`, PSD of order 3 (`λ_min = 1`); a zero-row nonneg block; `alpha`
    ignored off the power cone;
  - indefinite `P`: refused under the default and an explicit `ipm`, under a
    claimed `indefinite` on a PSD `P`, and with a cone; solved under
    `active-set`; `unknown`/`psd` claims pass the screen and `auto` restores
    the verdict;
  - cones vs `m_ineq`: short, long, an exp cone on 2 rows, a PSD cone counting
    `k(k+1)/2` rows, PSD mixed with pow and with exp;
  - `AddCone`: `dim ≤ 0` per kind (SOC, PSD, nonneg `< 0` and `Index::MIN`,
    exp/pow `≠ 3`), `alpha` in `{0, 1, −½, 3/2, −0, NaN, ±∞}`, unknown kind,
    NULL handle; each asserts nothing was added;
  - NULL arrays: each of row/col/val/vector for each builder at `nnz = 1`,
    each its own handle, then the `nnz = 0` side accepted; NULL `b`/`h` with
    no rows; NULL bounds as unbounded sides (and both NULL resetting a box);
    the `±1e19` convention on both sides and just inside it; NaN bound;
  - `INCOMPLETE` per missing piece (objective, `A`/`b`, `G`/`h`);
  - unknown option keys (numeric, string, wrong-type key, non-UTF-8 key);
  - each getter and the report writer before any solve → FALSE, nothing
    written;
  - each conic refusal its own fixture: `method=active-set`, `tau`,
    `tau_max`, `qp_presolve`, `qp_reg`, `qp_hsde` (at its default `yes`),
    `qp_equilibrate`, `qp_crossover`; and the passing side (`method=ipm`,
    `tol`, `max_iter`, `time_limit`);
  - active-set refusals `qp_reg`, `qp_hsde`, `qp_crossover` and the switches
    it reads (`qp_presolve` yes/no, `qp_equilibrate`); IPM `tau`/`tau_max`;
  - statuses: conic primal infeasible (Farkas ray checked: `Gᵀz = 0`,
    `hᵀz < 0`, `scale` = its inf-norm), conic dual infeasible (ray in the
    cone with `cᵀd < 0`), conic `ITERATION_LIMIT`, `time_limit = 0` →
    `TIME_LIMIT`, presolve proving infeasible/unbounded on the active-set
    engine (`iters = 0`).
- `qp_python_parity.rs` (26 tests: 25 rows + one that the reference holds
  exactly the listed rows), with `common/gen_qp_python.py` and its output
  `common/qp_python.json` (25 rows; regenerating is byte-identical). The
  generator records the exact inputs Python's `_build` hands the binding
  (`_lower_triangle_coo` / `_coo` triplets in their order) and the result;
  the test feeds those inputs to the handle. Rows: LP on the HSDE IPM,
  presolve + crossover (0 iterations), the direct driver, active-set; QP on
  the IPM, with `tau`/`tau_max`, with `qp_reg` + `qp_equilibrate=no` + `tol`,
  active-set presolved; a coupled `P`; indefinite `P` on active-set;
  `max_iter = 1`; QP primal and dual infeasible; nonneg, SOC (distance, with
  `tol`, projection with `P`), nonneg + SOC (`test_socp.py:22`), exp GP
  (`test_socp.py:43`), SOC + exp (`test_socp.py:96`), pow `α = 0.3`
  (`test_socp.py:115`), PSD (`test_socp.py:152`); conic primal and dual
  infeasible and `max_iter = 2`. Each row is checked three ways:
  1. against Python, **bit-exact on every compared field**: status, `obj`,
     `iters`, `x`, `y`, `z`, `z_lb`, `z_ub`, all eight residuals, `tau`,
     `kappa`, `certificate_scale` (null ↔ NaN) and the certificate kind. No
     field needed an exemption. The one input difference — Python passes an
     omitted `lb`/`ub` as an empty vector, the handle as `n` infinities — is
     read identically by `QpProblem::lb_of`/`ub_of` (and `FixedColumns`
     finds no pinned column in either), and the bitwise match on every row with an
     omitted side is the evidence. Gated on
     `cfg!(all(target_os = "linux", target_arch = "x86_64"))`; elsewhere the
     row must still reach Python's status and, on a solution, `x` and `obj`
     to `1e-6`.
  2. against `pounce_convex` called directly on Python's own `QpProblem`
     (empty box sides kept), bitwise, on every platform (same binary):
     status, `iters`, `obj`, `x`, `y`, `z`, `z_lb`, `z_ub`.
  3. against the row's hand-known answer to a tolerance, so a reference
     regenerated from a broken build cannot pass by agreeing with itself.
- PSD border: every parity `P` has eigenvalues in `{1, 2, 3, −1}`, far from
  where `certify_psd_lower_triangle` and Python's numpy verdict can differ.
- `// error-path:` markers added in the new file for the 12 exports whose
  refusals it exercises (all 15 already had one from S3-I2).
- Not reachable, recorded rather than tested: `AddCone`'s PSD
  `d·(d+1)` overflow guard and `Solve`'s cone-row `checked_add` cannot
  overflow on a 64-bit target with a 32-bit `ipindex`.

Mutations (each reverted): dropping `switches.apply` in `qp_options` fails
the parity rows `lp_direct` and `qp_ipm_reg_no_equilibrate`; dropping the
conic `tau` refusal fails `conic_refuses_tau`; dropping the PSD/exp-pow mix
check fails both mix tests; admitting `alpha = 0` fails the exponent test.

Gate (all on the ticking commit's tree):
- `cargo test -p pounce-cinterface`: 26 binaries, 342 passed, 0 failed
  (`qp_cones_and_errors` 63, `qp_python_parity` 26, `qp_handle` 14,
  `export_surface`, `header_abi`, `c_examples_build_and_run` green).
- `cargo clippy --workspace --exclude pounce-hsl --all-targets -- -D
  clippy::correctness -D clippy::suspicious`: exit 0; `--no-deps` clippy on
  this crate shows no warning in the new files.
- `cargo fmt --all -- --check` clean.
- `make python-test` (extension rebuilt from this tree): 2016 passed,
  36 skipped, 0 failed.
- No sweep: no solver source changed.

### S3-I4 — example, docs accuracy pass, CHANGELOG, §8.7 (commit that ticks S3-I4)

Approved by the human with Stage 3, on this branch. No solver or handle
behaviour changed: the only `src/` edit is one doc comment in `src/qp.rs`.

What moved:
- `examples/qp_socp.c` (new): (1) convex QP `min ½‖x‖² − x0 − 2x1`,
  `x0 − x1 = 0`, `x0 + x1 ≤ 1`, `x ≥ 0` (ub NULL), 0-based → checks
  `x = (½, ½)`, `obj = −5/4`, `y = −½`, `z = 1`, `z_lb = 0`, both KKT
  residuals `≤ 1e-6`, certificate `NONE` with `scale` NaN, `GetSolution`
  FALSE before the solve, and the summary report names `cvx-qp` and the
  schema; (2) SOC distance from `(2, −1)` to `x1 + x2 = 0` (`P` empty, NULL
  arrays at `nnz = 0`) → `t = obj = 1/√2`, `x = (3/2, −3/2)`, report engine
  `cvx-qcqp`; (3) `x ≤ −1`, `x ≥ 0` → `PRIMAL_INFEASIBLE`, certificate kind 1
  with a finite positive scale; (4) refusals: indefinite `diag(1, −1)` under
  `ipm` → `POUNCE_QP_ERR_INDEFINITE` and `GetSolution` FALSE after it, the
  same handle solves under `active-set`, one `SOC(1)` over 2 rows →
  `POUNCE_QP_ERR_CONES`, `POW` with `alpha = 1.5` → `POUNCE_QP_ERR_ARGUMENT`.
  Last line, registered in `EXPECTED_STDOUT`: `qp_socp example: QP (1/2,
  1/2), SOC distance 1/sqrt(2), infeasibility certified, 3 refusals`.
  Measured: QP 7 iterations, SOCP 7, Farkas scale 400. Strict flags, gcc and
  clang.
- Docs accuracy pass, against the code on this tree. Corrected:
  - `c-api.md` intro said the C API "does not route to the CLI's convex
    QP/QCQP engines" — true of `IpoptProblem` only; now points to `PounceQp`.
  - `c-api.md` *ABI and versioning* listed three guarded structs; now five.
  - `pounce.h` / `src/qp.rs`: `tau`/`kappa` NaN listed only `qp_hsde=no` and
    active-set; the exp/power-cone driver records none either (S3-I3's
    `exponential_cone` test).
  - S3-I2's §9 entry says "15 exports"; there are 14 (§8.7).
  Added (missing, not wrong): `c-api.md`'s Stage 3 section rewritten into
  subsections — standard form and index convention; the cone table with each
  cone's definition and row count, and the PSD `svec` order and `√2`
  scaling; every option with its range; the `OPTIONS` refusals (active-set
  and conic); the PSD screen (band, diagonal scan vs. `LDLᵀ` inertia,
  undecided = not certified, the divergence from `_psd_verdict` toward
  refusing, `hessian_inertia` claims overriding it, also with cones);
  status and error tables with the report status and AMPL codes; getters;
  certificates; the report (`cvx-qcqp` for every cone, `TnlpDirect`,
  `lambda = [y; z]`); Python parity bit-exact on Linux x86_64. `pounce.h`
  gains the cone definitions on `POUNCE_CONE_*`, the `svec` layout, "with
  or without cones" on the screen, and bounds-with-cones. Crate README: the
  example and the essentials. Root README crate table and
  `convex-solver.md` link the entry. `CHANGELOG.md`: one `[Unreleased]`
  *Added* entry beside Stage 1's.
- Bounds with cones: the handle accepts them (`solve_socp_ipm` appends a
  nonnegative block); Python's `solve_socp` has none, so there is no parity
  row. Checked once by hand (scratch C program, not committed): the SOC
  distance with `x1 ≤ 1` → `x = (1, 1, −1)`, `z_ub[1] ≈ 1`. Documented, and
  listed in §8.7.
- §8.7 written: 14 functions, the handle type, 22 defines, 2 structs with
  measured layouts (`PounceQpResiduals` 64/8, no padding;
  `PounceQpCertificate` 32/8, padding 4 at 4), and 8 decisions to confirm.

Gate (all on the ticking commit's tree):
- `cargo test -p pounce-cinterface`: 26 binaries, all green
  (`c_examples_build_and_run` 6 with the new example, `export_surface` 13,
  `header_abi` 5, `qp_cones_and_errors` 63, `qp_python_parity` 26,
  `qp_handle` 14, lib 64).
- `make c-api-examples` (gcc and `CC=clang`): all three examples print
  their lines.
- `make book` (mdbook 0.5.2): builds with no warning; anchors
  `c-api.html#matrix-form-convex-qp-and-conic-programs`, `#cones`,
  `#psd-screen`, `#python-parity`, `#json-report` exist. (A `|` inside a
  code span split the power-cone table row; escaped.)
- `make ask-check`: top-1 17/20, top-5 20/20, OK (unchanged).
- `scripts/check-release-consistency.sh`: OK.
- `cargo clippy --workspace --exclude pounce-hsl --all-targets -- -D
  clippy::correctness -D clippy::suspicious`: exit 0.
- `cargo fmt --all -- --check` clean.
- No sweep: no solver source changed.

### ABI hardening before release (commit that follows S3-I4)

Approved by the human on this branch (PR #1154), before 0.17.0 ships the
Stage 1 and Stage 3 surface. No solver code changed: `git diff` touches
`crates/pounce-cinterface` (`src/lib.rs`, `src/qp.rs`, `include/pounce.h`,
examples, tests, README), `docs/src/c-api.md`, `docs/src/complementarity.md`,
`CHANGELOG.md` and this file. `crates/pounce-convex`, `crates/pounce-cli/src`
and `crates/pounce-py` are untouched, so no fixture sweep.

What moved:
- **Extensible output structs.** `PounceComplementarityStats`,
  `PounceComplementaritySolve`, `PounceQpResiduals`, `PounceQpCertificate`
  gain `ipindex struct_size` as their first member (Rust and header). The
  getters go through one helper, `write_extensible` (`src/lib.rs`, behind an
  `unsafe trait ExtensibleOut`): `struct_size < size_of::<T>()` (0, negative,
  one short) → `FALSE`, nothing written; otherwise a bounded byte copy of
  bytes `4..size_of::<T>()` of the value, never `struct_size`, never past
  this release's size. Its comment states the growth rule for a maintainer
  appending a field. No `#define` for the sizes (`sizeof` is what callers
  pass; `header_abi.rs` unchanged). `PounceLinearSolverStats` untouched.
  Convention documented once in `pounce.h` (block before the complementarity
  outcome section), once in `c-api.md` (*Extensible output structs*, in the
  ABI section), the crate README, a pointer in `complementarity.md`, and the
  CHANGELOG.
- New layouts (x86_64 Linux, gcc and the Rust side, asserted by
  `repr_c_structs_match_the_header_layout`, which now also pins the four
  sizes and `struct_size` at offset 0 with `ipindex` width, and that
  `PounceLinearSolverStats` has none):
  - `PounceComplementarityStats` 224 bytes, align 8 (unchanged size: the
    new field fills old padding). `struct_size` 0, `strategy` 4,
    `reported` 20, `accepted_by` 24, `fallback` 56, `n_solves` 88,
    `total_iterations` 92, `max_abs_product` 96; from `certificate_class`
    (104) on, §8.4's offsets are unchanged. Padding: 4 bytes at 124 only.
  - `PounceComplementaritySolve` 72 bytes, align 8 (unchanged size):
    `struct_size` 0, `stage` 4, `relaxed` 36, `tau` 40, then §8.4's offsets.
    No padding.
  - `PounceQpResiduals` 72 bytes (was 64): `struct_size` 0, 4 bytes padding,
    then the eight reals at 8..64.
  - `PounceQpCertificate` 32 bytes (unchanged): `struct_size` 0, `kind` 4,
    `scale` 8, `tau` 16, `kappa` 24. No padding.
- Callers: `examples/complementarity.c` and `examples/qp_socp.c` set
  `struct_size = sizeof`; `qp_socp.c` also checks a zeroed `struct_size` is
  refused with the struct untouched (no new output line). Rust helpers
  (`common/cc_models.rs` `stats`/`solve_record`, `common/qp_c.rs`,
  `qp_handle.rs`), the NULL/before-solve tests and `complementarity_moi.rs`
  set it, so each still reaches the branch it was written for. The
  `pounce-cli` parity tests read the C report, not these getters: nothing to
  change there.
- Tests, one per branch (`tests/common/struct_size.rs`,
  `check_struct_size_contract`, on a handle with a result so the size check
  is what decides): (a) `struct_size` 0, −1 and `0xA5A5A5A5` → `FALSE`,
  buffer byte-identical; (b) `sizeof − 1` → same; (c) `sizeof` → filled,
  equal to the ordinary helper's fill, `struct_size` unchanged, nothing past
  `sizeof`; (d) `sizeof + 64` in a buffer that large, sentinels after
  `sizeof` → `TRUE`, the same values, every byte past `sizeof` untouched.
  Run as `stats_struct_size_contract`, `solve_struct_size_contract` (every
  lowered-solve `k`) in `complementarity_getters.rs`, and
  `residuals_struct_size_contract`, `certificate_struct_size_contract`
  (primal infeasible, so every field holds a value) in
  `qp_cones_and_errors.rs`, each with `// error-path:` markers.
- Shown to bite (then reverted, `diff` against a saved copy empty): with
  `write_extensible` also zeroing the caller's bytes `sizeof..struct_size`,
  all four tests fail at (d) ("struct_size = sizeof + 64 wrote bytes past
  sizeof at offsets [224, …]", `[72, …]`, `[72, …]`, `[32, …]`) while (c)
  passes; with the size check removed, all four fail at (a) ("struct_size = 0
  was accepted").
- **Bounds with cones**, `tests/qp_cones_and_errors.rs`, on the SOC distance
  model (`t ≥ ‖(x1 − 2, x2 + 1)‖`, `x1 + x2 = 0`). Re-derived: on the line
  `x = (a, −a)`, `f(a) = (a − 2)² + (1 − a)²`, `f' = 4a − 6`, free minimum
  `a = 3/2`, `t = 1/√2`. `bounds_with_a_cone_active_bound`: `x1 ≤ 1` cuts it
  off and `f` increases past `3/2`, so `a = 1`, `x = (1, 1, −1)`, `t = 1`;
  with `s = (1, −1, 0)` on the boundary, `z = μ(1, 1, 0)`, the `t` row gives
  `μ = 1`, the `x2` row `y = 0`, the `x1` row `z_ub1 = 1`. Measured `x` to
  1e-8, duals off by 7.5e-6 (asserted to 1e-4: `z2 = 0` is fixed only by
  `z` on the cone's boundary, approached like the root of the gap), 8
  iterations; and the same bound written as a trailing `nonneg(1)` G row
  gives the same `x`, `obj`, and row multiplier `= z_ub1`.
  `bounds_with_a_cone_inactive_bounds` (the other branch): `x1 ≤ 2`,
  `x2 ≥ −10`, both slack at `(3/2, −3/2)`: `t = 1/√2`, zero bound
  multipliers, equal to the box-free solve. No Python parity row:
  `solve_socp` takes no `lb`/`ub`, `solve_qp` no cones, and
  `qp_python_parity.rs` compares one `QpProblem` on both sides, so a
  reformulation is not a row; the test's comment says so.

Gate (on the committed tree):
- `cargo test -p pounce-cinterface`: 26 results (lib, 24 integration
  binaries, doc-tests), 348 passed, 0 failed (`complementarity_getters` 15,
  `qp_cones_and_errors` 67, `c_examples_build_and_run` 6, `export_surface`
  13, `header_abi` 5, `qp_handle` 14, `qp_python_parity` 26, lib 64).
- `cargo test -p pounce-cli --test cinterface_mpcc_parity`: 16 passed.
- `make c-api-examples`, gcc and `CC=clang`: all three examples print their
  last lines (`qp_socp example: QP (1/2, 1/2), SOC distance 1/sqrt(2),
  infeasibility certified, 3 refusals`; output unchanged).
- `make book` (mdbook 0.5.2): no warning; `c-api.html#extensible-output-structs`
  exists and the links to it from `c-api.html` and `complementarity.html`
  resolve. Build dir removed.
- `make ask-check`: top-1 17/20, top-5 20/20, OK (unchanged).
- `scripts/check-release-consistency.sh`: OK.
- `cargo clippy --workspace --exclude pounce-hsl --all-targets -- -D
  clippy::correctness -D clippy::suspicious`: exit 0; no new warning in
  `pounce-cinterface` (599 before and after).
- `cargo fmt --all -- --check`: clean.
- No sweep: no solver source changed.

### PR review, second round (on `67d84c71`)

Ten threads from the review against `84b073e7`. Each behaviour fix has a test
that fails with the fix reverted (reverted one fix at a time, test run, file
restored); a fixture per branch.

1. **Stale working set after the warm-start refusal** (`lib.rs`).
   `IpoptSolveWarmStart` copies a set out only when `last_solve` is set, and
   `IpoptGetWorkingSet` returns `FALSE` (nothing written) with no
   `last_solve` or with pairs declared. Tests
   (`tests/complementarity_solve.rs`):
   `a_refused_warm_start_does_not_copy_out_an_earlier_working_set` (SQP
   solve, pairs, warm start; sentinel buffers untouched) and
   `no_working_set_is_reported_once_pairs_are_declared_or_a_solve_is_refused`
   (branch 1: pairs declared, then cleared → readable again; branch 2: a
   pair-free solve refused for a NULL `x`). Both failed with the fix
   reverted.
2. **Lowered-solve flag per thread** (`lib.rs`). Every `IpoptSolve` now sets
   `IN_LOWERED_SOLVE` for its own duration (`LoweredSolveGuard::set(has
   pairs)`, restored on drop), so a pair-free solve nested in a lowered
   solve's callback clears it and the outer value returns after. Test
   `a_pair_free_solve_nested_in_a_lowered_solve_has_working_inspectors`: B's
   inspectors `[1, 1, 1]` on every fire, A's `[0, 0, 0]` before and after the
   nested solve. Failed reverted (B read `FALSE`).
3. **Conic `hessian_inertia`** (`qp.rs`): any value but `auto` with a cone is
   `POUNCE_QP_ERR_OPTIONS`. Tests (`tests/qp_cones_and_errors.rs`)
   `conic_refuses_hessian_inertia_{unknown,indefinite,psd}` on the
   indefinite-`P`-with-a-cone model and on `nonneg_model`, and the passing
   branch `conic_accepts_hessian_inertia_auto`. The three failed reverted
   (`unknown`: the indefinite `P` reached the conic IPM and came back
   `NUMERICAL_FAILURE` (6); `indefinite`: `INDEFINITE`; `psd`: solved).
4. **Results stale after the problem changed** (`qp.rs`): each builder that
   returns `TRUE` clears `info.last`. Decision: option setters keep it
   (documented in `qp.rs`, `pounce.h`, `c-api.md`). Tests
   (`tests/qp_handle.rs`): `every_builder_discards_the_previous_result` (one
   fixture per builder; getters, certificate and report `FALSE`, the next
   solve restores; a refused builder keeps the result) and
   `option_setters_keep_the_previous_result`. The builder test failed
   reverted, as did the updated `getters_answer_false_without_a_result`
   (which had pinned the old behaviour).
5. **`PounceQpGetSolution` length cap**: every source length is checked
   against `n, m_eq, m_ineq, n, n` before anything is written. Unit test
   `qp::tests::get_solution_refuses_a_result_of_the_wrong_shape` plants a
   one-long vector in each of the five in turn. Failed reverted.
6. **Unsatisfiable bounds**: `SetBounds` refuses `lb ≥ 1e19`, `ub ≤ -1e19`,
   `lb > ub`. Tests `bounds_refuse_a_lower_bound_at_plus_infinity`,
   `bounds_refuse_an_upper_bound_at_minus_infinity`,
   `bounds_refuse_crossed_bounds` (each also shows nothing was stored) and
   `bounds_accept_a_fixed_variable`. The three refusals failed reverted.
7. **PSD band** moved, unchanged, to `pounce_convex::{PSD_TOL, psd_band}`
   (`psd_certificate.rs`); `psd_band` takes the entries' values instead of
   the CLI's `QuadHessian`. The CLI's dispatch and the C crate call it; the C
   copy and its "must match" comment are gone. CLI tests
   `psd_band_does_not_widen_above_unit_scale` /
   `psd_threshold_is_psd_tol_at_unit_scale` pass unchanged;
   `psd_band_scales_down_only` added in `pounce-convex`.
8. **Status mapping, residuals, certificate scale** moved, unchanged, to
   `pounce_convex::report` (`qp_status_to_ars`, `reported_kkt_residuals`,
   `certificate_scale`); the CLI (`main.rs`), Python (`solution_dict_hsde`)
   and C call them. `pounce-convex` gains a direct `pounce-nlp` dependency
   (already transitive through `pounce-presolve`; acyclic). Nothing left as
   a copy, so no cross-check test was needed; the CLI's mapping tests and
   `status_mappings_cover_every_variant` (C) pass on the shared definition.
   `pounce_cblib`'s map differs and is untouched (§8.7).
9. **PSD work per solve**: `certify_psd_lower_triangle_in_band` reads the
   band off the certificate's own summed map (one map, not two), and the
   verdict is cached on the handle (`psd_verdict`), dropped by
   `SetObjective` and by every `hessian_inertia` setting. Unit tests (a
   `cfg(test)` thread-local counter):
   `the_psd_certificate_runs_once_per_objective` (`b`, bounds, an option: 1
   run; new `P`: 2; a claim: 2; back to `auto`: 3) and
   `a_cached_verdict_never_outlives_its_objective`; both failed with the
   cache disabled.
   `the_in_band_certificate_is_the_certificate_at_the_band_of_the_sum`
   checks the new entry against the old one on each branch.
10. **Publish-order comment**: `crates/pounce-cli/Cargo.toml` now says the
    versioned dev-dependency is kept on publish, so `pounce-cli` publishes
    only after `pounce-cinterface` is live (what
    `check-release-consistency.sh` enforces). §8.5's item and the S1-I8 log
    entry here carried the same wrong claim and are corrected;
    `CHANGELOG.md` did not have it.

Sweep (items 7 and 8 touch `pounce-convex` and `dispatch.rs`): release
baseline from `67d84c71` in its own worktree and target dir, head from this
tree; `scripts/sweep-fixtures.sh` on both: 282 fixture-legs each (141 exact,
141 lbfgs; engine column 15 `cvx-qcqp`, 52 `cvx-qp`, 74 `nlp` per leg), diff
**empty** on both legs, engine column included. Python: `solve_qp` /
`solve_socp` on ten models (QP, presolved QP, active-set, primal- and
dual-infeasible LP, SOC ×2, power, PSD, conic infeasible; every field of the
result, `repr`'d) byte-identical between the baseline and head extensions
(180 lines).

Gate:
- `cargo test -p pounce-cinterface -p pounce-convex -p pounce-py`: 1184
  passed, 0 failed.
- `cargo test -p pounce-cli --lib --bins --test cinterface_mpcc_parity`
  plus the dispatch/PSD-band and convex-report tests
  (`issue872_qp_second_order_scale`, `issue848_nonconvex_qp_verdict`,
  `issue871_nonconvex_qp_equality_row`,
  `issue_767_convex_path_reports_like_nlp`, `json_report`,
  `convex_time_limit`): 326 passed, 0 failed.
- `make c-api-examples`: the three examples build and run.
- Python suite with the extension rebuilt from this tree (`make python-ext`,
  then `pytest tests`): 2112 passed, 33 skipped.
- `make book`: OK. `make ask-check`: top-1 17/20, top-5 20/20, OK.
- `scripts/check-release-consistency.sh`: OK.
- `cargo clippy --workspace --exclude pounce-hsl --all-targets -- -D
  clippy::correctness -D clippy::suspicious`: exit 0, no warning in a file
  this round touched.
- `cargo fmt --check`: clean.

### Facts recorded while planning (2ac0ad9)

- `pounce.h:40-43` declares `POUNCE_VERSION "0.1.0"`; the crate is `0.16.0`
  and `GetIpoptVersion` reports `0.16.0`. Untested until S1-I2.
- `tests/header_abi.rs` checks typedef widths only; no prototype check,
  no define check, no C compilation anywhere in this crate's tests.
- `examples/sqp_warm_start.c` is built by nothing in CI.
- The declared-pair fixtures are under `crates/pounce-cli/tests/fixtures_mpcc/`,
  not `tests/fixtures_mpcc/`. Volt-watt models are data-root only
  (`benchmarks/Makefile:558-570`).
- A fixed variable with a free row is refused by `MpccTnlp::with_lift`
  today; the MOI table treats it as "no pair" — handled in the C helper,
  never by changing the wrapper.
- `GetIpoptCurrentIterate` already returns `FALSE` during a lowered solve
  whose dimensions differ (`lib.rs:921, 944-949`); S1-I5 pins that.
- Baseline: `cargo build -p pounce-cinterface` and `cargo test -p
  pounce-cinterface --test header_abi` (2 passed) both green on `2ac0ad9`.
- `cargo test -p pounce-cinterface --test header_abi` leaves
  `target/debug/libpounce_cinterface.so` (148 MB, debug) next to
  `libpounce_cinterface.rlib`, i.e. the lib target's cdylib is built for
  integration tests; S1-I3 can locate it as
  `current_exe().parent().parent()` without a separate `cargo build`.
