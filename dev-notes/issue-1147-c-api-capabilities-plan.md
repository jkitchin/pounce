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

- [ ] **S1-I1** Shared pair classifier in `pounce-nlp` (`validate_pairs`), `MpccTnlp::with_lift` uses it; every shape branch has a unit test; cc: sweep lines diff empty.
- [ ] **S1-I2** Export-surface guard: `tests/export_surface.rs` + `header_abi.rs` prototype check; stale `POUNCE_VERSION` fixed; new book page `docs/src/c-api.md`; README completed for the *existing* surface.
- [ ] **S1-I3** C compile-and-run test: `tests/c_examples_build_and_run.rs` builds and runs every `examples/*.c` against `pounce.h` and the cdylib.
- [ ] **S1-I4** Declaration API: `AddPounceComplementarityPair`, `SetPounceComplementarityPairs`, `ClearPounceComplementarityPairs`, `GetPounceComplementarityPairCount`, `GetPounceComplementarityPairs`; every error code tested.
- [ ] **S1-I5** `IpoptSolve` routes through `solve_mpcc` when pairs are declared; sessions and SQP working-set warm starts refuse pairs; live inspectors pinned.
- [ ] **S1-I6** Outcome getters: `GetPounceComplementarityStats`, `GetPounceComplementaritySolve`, `GetPounceComplementarityText`; "driver did not run" behaviour tested.
- [ ] **S1-I7** `IpoptWriteSolveReport` carries the `complementarity` block.
- [ ] **S1-I8** CLI-parity test over `crates/pounce-cli/tests/fixtures_mpcc/` through the C API (`crates/pounce-cli/tests/cinterface_mpcc_parity.rs`).
- [ ] **S1-I9** MOI-semantics helper `AddPounceComplementsConstraint`: all five table rows + error rows tested.
- [ ] **S1-I10** C example `examples/complementarity.c`; docs (`complementarity.md`, `c-api.md`, README, CHANGELOG); Pounce.jl gate/ABI-review checklist written.
- [ ] **S1-I11** Stage-1 closing gate: full fixture sweep (both legs, cc: lines) diffs empty against the baseline binary; `make check clippy fmt-check`; `make python-test`.

**Stop here and report to the human before starting any later stage.**

### Stage 2 — in-place bounds/data update and warm re-solve

- [ ] **S2-BLOCKED** Blocked on confirmation from the Julia side (§6). Do not start.

### Stage 3 — matrix-form convex/conic entry

- [ ] **S3-I1** Lift `solve_presolved` / `demote_unverified_optimal` from `pounce-py/src/qp.rs` into `pounce-convex` (pure move).
- [ ] **S3-I2** `PounceQp*` handle, builder, solve and getters.
- [ ] **S3-I3** Tests for every cone kind and every error path; parity against Python `solve_qp` / `solve_socp` anchors.
- [ ] **S3-I4** Docs, README, CHANGELOG, `examples/qp_socp.c`; export guard green.

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

To be completed at S1-I10 with the final symbol list. Template:

- [ ] Symbols added (names, prototypes) — none removed, none changed.
- [ ] `#define`s added — none changed.
- [ ] Structs added — `PounceLinearSolverStats` untouched (offsets verified by
      the S1-I6 layout test).
- [ ] `GetIpoptVersion` reports `0.17.0` once released; `pounce.h`
      `POUNCE_VERSION` agrees (release guard).
- [ ] Pounce.jl: `Pounce.jl/src/Pounce.jl:17` gate → `< v"0.18.0-0"`;
      `README.md`/`docs/src/julia.md` text updated; `Pkg.test()` green against
      the 0.17 cdylib **with no other Julia change** (additions only).
- [ ] Optional for Pounce.jl: add the new symbols to `_required_native_symbols`
      only when the MOI side starts using them.
- [ ] Manual runs not available to the loop: `make -C benchmarks mpcc-opf-run`
      (volt-var/volt-watt, data root) against a baseline, and the other
      declared-pair corpora named in `CLAUDE.md` if `pounce-nlp` was changed
      beyond the pure classifier move.

---

## 9. Per-iteration log

Append one entry per iteration: `### <item> — <commit>` then what moved,
what was measured (with the command), and anything that surprised you.

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
