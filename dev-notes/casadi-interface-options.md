# A CasADi interface for POUNCE — options, evidence, recommendation

Status: **decision pending**. Everything below was measured on this
machine against `casadi==3.7.2` (pip wheel, CPython 3.11, Linux x86-64)
and a `--release` build of this repo, unless marked otherwise.

Written while working gh#624 (nonlinear-variable masks), whose motivating
use case is "a POUNCE CasADi `Nlpsol` plugin".

---

## 1. What "fully support what CasADi users do" actually means

Not a wish list — this is the surface a CasADi user touches without
thinking about it, and anything that misses one of these reads as broken:

| Capability | Why it is table stakes |
| --- | --- |
| `nlpsol('S', '<name>', nlp, opts)` | The entry point. Everything else composes from it. |
| `Opti` (`opti.solver('<name>')`) | The modern front door. **`Opti` only accepts an `nlpsol` plugin name** — a Python-side solver object cannot be plugged into it at all. |
| MX graphs, not just SX | Anything built with `Opti`, `integrator`, or a `Function` call is MX. |
| Parameters `p` | Every MPC / estimation / parameter-sweep model has them. |
| Derivatives of the solution map (`jacobian(sol['x'], p)`, `nlpsol` inside another `Function`) | Bilevel, parameter estimation, learning-to-optimize. CasADi's `Nlpsol` base class implements `get_forward` / `get_reverse` generically off the KKT system, so *any* plugin inherits it. A hand-rolled Python wrapper does not. |
| `iteration_callback` | Live plots, early stopping, logging. |
| `stats()` — `success`, `return_status`, `iter_count`, `iterations` dict | Users branch on these; `Opti` surfaces them. |
| Warm start via `x0`/`lam_g0`/`lam_x0` | Every receding-horizon loop. |
| Option pass-through (`{'pounce': {...}}`) | The whole point of picking a solver. |

Two consequences fall straight out of the table: **it has to be a real
`nlpsol` plugin** (for `Opti` and for inherited sensitivities), and
**it has to accept MX with parameters**.

---

## 2. The options

### A. AMPL bridge — `nlpsol('S','ampl',nlp,{'solver':'pounce'})`

CasADi ships an `ampl` plugin (present in the pip wheel's plugin list)
that writes an `.nl` file, shells out to a solver binary, and reads the
`.sol` back. POUNCE already reads `.nl` and writes `.sol`.

Measured: **it does not work today, and cannot be made to fully work.**

* `AmplInterface::init` asserts `oracle().is_a("SXFunction")` → any MX
  model, i.e. everything built with `Opti`, is refused outright:
  `Only SX supported currently.`
* It then asserts `p.is_empty()` → `'p' currently not supported`.
* With both avoided (SX, no parameters), the invocation still fails
  because CasADi calls `<solver> -o<solfile> <nlfile>` and POUNCE's CLI
  does not accept ASL-style `-o<path>`; it prints help and exits
  non-zero, which CasADi reports as `Assertion "ret==0" failed`.
* Even repaired, the plugin sets **bound multipliers to NaN** (it does
  not parse them from `.sol`), one subprocess + temp files per solve, no
  warm start, and no iteration callback.

Verdict: **stopgap only.** The `-o<path>` gap is a small CLI fix worth
making regardless (it is the ASL convention, so it also helps anything
else that drives POUNCE as an AMPL solver), and POUNCE's existing
`pounce_options` environment variable already gives option pass-through.
But SX-only + no-parameters rules this out as *the* answer.

### B. Python-level wrapper (`casadi.Callback`)

Build the oracle functions from the `nlp` dict and drive POUNCE from
Python, exposing the result as a `casadi.Callback`.

Verdict: **rejected.** `Opti` cannot use it. Solution-map derivatives
would have to be re-implemented by hand instead of inherited from
`Nlpsol`. No codegen, no serialization, and the classic
keep-the-callback-alive footgun. This is precisely the "shim POC that is
limited" the request rules out.

### C. Out-of-tree `nlpsol` plugin — `libcasadi_nlpsol_pounce.so`

A C++ `Nlpsol` subclass compiled against CasADi's headers, registering
`casadi_register_nlpsol_pounce`, loaded by CasADi's plugin loader from
`CASADIPATH` / `LD_LIBRARY_PATH`, and talking to POUNCE through the
existing Ipopt-compatible C API (`pounce.h` / `libpounce_cinterface`).

**Built and working** — see §3. The interface lives in `casadi/`
(mirroring `gams/`): plugin source, Makefile, parity tests against
CasADi's own Ipopt, and six examples.

Verdict: **recommended.** It is the only option that satisfies the whole
table in §1 without waiting on anyone else.

### D. Upstream, in-tree CasADi interface

Contribute `casadi/interfaces/pounce/` to CasADi so `nlpsol('S','pounce')`
works out of a stock pip install.

There is a direct precedent: **`Nlpsol::madnlp`** is in the pip wheel's
plugin list, and MadNLP is a *Julia* solver bound through a C API
(`libmad_*` / `casadi_madnlp_*`) — structurally the same shape as POUNCE
(Rust solver, C API). `sleqp`, `alpaqa` and `fatrop` are further
recent third-party additions.

Verdict: **the right destination, not the right first step.** It costs a
build-system dependency on CasADi's side and a release cycle we do not
control. Route C is the same C++ source; graduating it upstream later is
a packaging change, not a rewrite. Do C first, propose D once it has
users.

### E. Drop-in `libipopt` replacement

Verdict: **rejected**, and the reason is worth stating precisely, because
"our C API is Ipopt-compatible" invites the question.

Nothing is *missing from the C API* for this job — the entire plugin in
§3 runs on it, and the parity table is what that buys. What a drop-in
would need is not more C API, it is a **C++ ABI**, which is a different
kind of artifact and not something a C API grows into.

Measured on the installed wheel:

```
$ objdump -p libcasadi_nlpsol_ipopt.so | grep -E 'NEEDED|RPATH'
  RPATH   $ORIGIN:$ORIGIN/.
  NEEDED  libcasadi.so.3.7
  NEEDED  libipopt.so.3          <- bundled *inside* the casadi wheel (3.14.11)

$ nm -D --undefined-only -C libcasadi_nlpsol_ipopt.so | grep Ipopt
  U Ipopt::IpoptApplication::IpoptApplication(bool, bool)
  U Ipopt::StreamJournal::StreamJournal(std::string const&, Ipopt::EJournalLevel)
  U Ipopt::StreamJournal::SetOutputStream(std::ostream*)
  U Ipopt::Journal::~Journal()
  U Ipopt::TNLP::get_curr_iterate(Ipopt::IpoptData const*, …) const
  U vtable for Ipopt::StreamJournal
  U vtable for Ipopt::RegisteredOption
  U vtable for Ipopt::RegisteredOptions
```

Eight symbols — but they are constructors, a non-virtual member with a
real implementation, and three **vtables**. Satisfying them means
reproducing Ipopt's object model: `ReferencedObject` refcounting and
`SmartPtr` semantics, the `TNLP` abstract base that the plugin
*subclasses* (so we own the callback side of its vtable too),
`RegisteredOptions` with its whole option registry, `OptionsList`,
`Journalist`, the opaque `IpoptData` / `IpoptCalculatedQuantities`
handed back to TNLP methods, `SolveStatistics`, and RTTI for the
`dynamic_cast`s. The library those come from exports 4770 symbols. Rust
cannot express C++ vtables, RTTI or mangling, so this would be a C++
layer re-implementing Ipopt's public classes — far larger than the
~420-line plugin, pinned to an Ipopt *ABI* version (3.14 differs from
3.13, and the soname moves), and carrying the same libstdc++ dual-ABI
trap, since `std::string` appears in the signatures above.

And then it would have to *displace* the `libipopt.so.3` sitting next to
`libcasadi_nlpsol_ipopt.so` under `$ORIGIN`, i.e. overwrite a file inside
someone else's wheel. That is hostile, it is not per-project, and it
destroys the property this integration is validated by: both solvers
loadable in one process, so `nlpsol('ipopt')` and `nlpsol('pounce')` can
be run against the same model object.

**Would it make a `pounce-casadi` wheel easier?** No — it makes it
harder. It trades "match CasADi's internal C++ ABI" for "match Ipopt's
C++ ABI *and* substitute a bundled shared library", and gives up
side-by-side operation. The plugin's coupling is the cheaper one.

### What a wheel actually takes

Worth writing down, since it is the open decision and it is smaller than
it looks. The plugin's only version-sensitive dependency is CasADi's
internal C++ headers, and CasADi minor releases are infrequent (3.6 in
2023, 3.7 in 2025). So:

* one plugin build per (CasADi minor × platform), each a `pip install
  casadi==X.Y.*` plus a `git clone --branch X.Y.Z` for headers — both
  scriptable, and already what `make fetch-src` does;
* ship the resulting `libcasadi_nlpsol_pounce.so` files plus
  `libpounce_cinterface.so` in one wheel, and select at import time on
  `casadi.__version__`;
* build inside manylinux with `-D_GLIBCXX_USE_CXX11_ABI=0` to match the
  CasADi wheels (§3), plus macOS x86_64/arm64 and Windows;
* installation is then a file copy into CasADi's package directory —
  already implemented as `make install`, no `sudo`, no env var.

The recurring cost is one matrix entry per CasADi minor version, which
is also exactly the maintenance that goes away if the interface is
upstreamed (route D).

---

## 3. Evidence from the prototype

~420 lines of C++ (`casadi/casadi_nlpsol_pounce.cpp`), built against
casadi 3.7.2 headers and linked against the wheel's `libcasadi.so` plus
`libpounce_cinterface.so`. `make test` runs all of this
(`casadi/test_parity.py`, 18 checks, all passing); the model is
Rosenbrock with a parametric circle constraint, MX with a parameter `p`:

| Check | Result |
| --- | --- |
| `nlpsol('S','pounce', nlp)` on MX + parameters | solves; `x = [0.907234, 0.822755]`, identical to `ipopt` to 6 digits |
| Bound multipliers with an active bound | `lam_x = [-80.953, 0]`, `lam_g = 16.7557` — **exactly** Ipopt's |
| `jacobian(S(...)['x'], p)` (inherited `get_forward`) | `[0.208074, 0.378275]` — matches Ipopt's |
| `Opti` (`opti.solver('pounce')`) | solves, `sol.stats()['return_status'] == 'Solve_Succeeded'` |
| `stats()` | `success`, `return_status`, `iter_count`, plus a full per-iteration `iterations` dict (`inf_pr`, `inf_du`, `mu`, `d_norm`, `regularization_size`, `obj`, `alpha_pr`, `alpha_du`, `ls_trials`) |
| `iteration_callback` with live `x`/`g`/`lam` | works — 13 fires, real iterates (**after** the fix in §4.1) |
| Warm start (`lam_g0`/`lam_x0` + `warm_start_init_point`) | re-solve at a perturbed `p` converges in 5 iterations |
| `hessian_approximation=limited-memory` + `pass_nonlinear_variables` | solves; same KKT point as unmasked and as Ipopt |
| `max_iter=2` through the option dict | `Maximum_Iterations_Exceeded`, `success == False` |
| Early termination from the callback | `User_Requested_Stop` at the iterate the callback saw |

Worth noting: CasADi's own Ipopt plugin **cannot** give you a full
`iteration_callback` unless Ipopt was built specially — it prints
*"intermediate_callback is disfunctional in your installation"* and hands
the callback nothing. POUNCE's `GetIpoptCurrentIterate` makes the full
callback work out of the box. That is a genuine advantage to advertise.

### Is the mask a performance win? Only after fixing the diagonal

The motivating claim in gh#624 is performance. The faithful port of
upstream's choice — `reduced_diag = true`, i.e. `W` *exactly zero* on the
variables that enter linearly — is markedly **slower**, on a synthetic
model with 2 nonlinear and 2000 linear variables:

| | unmasked | masked |
| --- | --- | --- |
| POUNCE, exact-zero diagonal (upstream's choice) | 0.85 s, 25 iters | 4.7 s, 28 iters |
| CasADi's Ipopt (`pass_nonlinear_variables`) | 0.40 s, 20 iters | **399 s**, 23 iters |

Ipopt reproducing the same effect two orders of magnitude more severely
is what said the flag, not the port, was at fault. Zeroing the
quasi-Newton diagonal leaves those rows of the augmented system's
`(1,1)` block carrying the barrier term `Σ_x` alone — ~0 for a variable
far from its bounds — and the symmetric factorization pays for a
near-singular diagonal on every one of them.

The fix that shipped: keep a **curvature floor**
(`limited_memory_init_val_min`, 1e-8 by default, registered with a strict
positive lower bound so it can never be zero) on the masked-out
coordinates, σ on the rest, and restrict only the `V`/`U` columns.

| n linear | unmasked | masked, exact zero | masked, floor |
| --- | --- | --- | --- |
| 2 000 | 0.89 s, 25 it | 4.7 s, 28 it | 0.93 s, 28 it |
| 10 000 | 6.0 s, 31 it | — | **5.1 s, 27 it** |

So the feature now does what the issue asked for, and the win grows with
the linear-to-nonlinear ratio.

The obvious cheaper alternative — fill the whole diagonal with σ — is
equally fast and *wrong in a way that shows up*: it injects a proximal
term of the problem's own curvature scale into coordinates whose
curvature is zero, and the masked update has no columns there to learn
it back down (the unmasked path does, which is why it never had this
problem). On the 6-variable fixture in
`crates/pounce-cinterface/tests/nonlinear_variables_mask.rs` that turns
`Solve_Succeeded` at `tol=1e-9` into a stall at
`Solved_To_Acceptable_Level`. 1e-8 is below anything the solver reasons
about and the tail converges.

This is a documented divergence from upstream, in
`hess/lim_mem_quasi_newton.rs`, with the measurements in the comment.

### Packaging constraints found the hard way

1. **The pip wheel does not ship CasADi's internal headers.** It ships
   `casadi/core/*.hpp` public headers but not `nlpsol_impl.hpp` /
   `plugin_interface.hpp`. A plugin build needs the matching source tree
   (`git checkout 3.7.2`) for headers, linking against the wheel's
   `libcasadi.so`.
2. **The wheel is built with the old libstdc++ ABI.** Its symbols mangle
   as `...ERKSs`, not `...ERKNSt7__cxx11...`, so the plugin must be
   compiled `-D_GLIBCXX_USE_CXX11_ABI=0` or it fails to load with an
   undefined-symbol error naming
   `casadi::FunctionInternal::generate_options`.
3. **The define set is part of the ABI.** `-DWITH_DL` is required to
   compile at all (`handle_t` is defined under it) and the rest of
   `casadi.CasADiMeta.compiler_flags()` (`CASADI_WITH_THREAD`,
   `WITH_DEPRECATED_FEATURES`, …) must match or struct layouts drift
   silently.
4. **No version handshake.** `PluginInterface::load_plugin` does *not*
   compare `plugin->version` against `CASADI_VERSION`; a mismatched
   plugin loads and misbehaves. We must pin: one plugin build per CasADi
   minor version, and refuse to load otherwise.

None of these is a blocker. All of them mean the eventual deliverable is
a **wheel** (`pounce-casadi`) with a build matrix, not a "here's a .so"
README.

Installation, once built, is genuinely easy: CasADi's loader searches its
own package directory *first*, so dropping the plugin there makes
`nlpsol(..., 'pounce', ...)` work with no environment variable and no
`sudo` (it is the user's site-packages). Verified from a bare Python
process with `CASADIPATH` unset. `make install` does exactly this; a
wheel would do it at install time, or ship a `pounce-casadi register`
command mirroring `pounce-gams register`.

---

## 4. POUNCE-side findings

### 4.1 `GetIpoptCurrentIterate` aborted the process — fixed in this branch

Calling it from inside an intermediate callback — its *documented* use —
with a non-NULL `g` panicked with `RefCell already borrowed` and, because
the panic crosses `extern "C"`, **aborted the process**. Root cause:
the function held a shared borrow of the algorithm-side `IpoptNlp` across
`IpoptCq::curr_c` / `curr_d`, which re-enter the NLP mutably.
`GetIpoptCurrentViolations` had the same defect on its `grad_lag_x`
branch. Both are fixed, with a regression test
(`crates/pounce-cinterface/tests/current_iterate_inspectors.rs`).

This is not a CasADi-only bug: it is on the path of any C consumer that
logs its iterates, including the GAMS C link.

### 4.2 Nonlinear-variable masks (gh#624) — implemented in this branch

The plugin's `pass_nonlinear_variables` needs somewhere to put the mask.
Landed: `TNLPAdapter::quasi_newton_nonlinear_vars` (port of upstream's
`GetQuasiNewtonApproxSpaces`), the reduced-space L-BFGS update, and
`IpoptSetNonlinearVariables` in the C API. See the CHANGELOG entry.

### 4.3 `Bool` is a C-ABI mismatch — open, not fixed here

`include/pounce.h` says `typedef bool Bool` (matching Ipopt 3.14, which
uses C99 `bool` throughout its C API); the Rust implementation says
`pub type Bool = c_int`. One byte versus four. In practice gcc/clang
zero-extend on both sides so it works, but the psABI leaves the upper
bits of a `bool` return unspecified — a C callback returning `false` to
signal a failed evaluation could be read as nonzero, i.e. success. Worth
its own issue; deliberately **not** folded into this branch, because
changing it touches every callback signature.

It is also why `IpoptSetNonlinearVariables` takes a count plus an index
list rather than the `const Bool*` mask the issue proposed: an *array* of
mismatched booleans is a data-layout bug, not a benign one.

### 4.4 Smaller gaps a full plugin will want

* **Option type introspection.** CasADi's Ipopt plugin asks
  `RegOptions()` for each option's type so a user dict can be dispatched
  to `SetNumericValue`/`SetIntegerValue`/`SetStringValue`. Over the C API
  the plugin has to guess from the Python/`GenericType` side. A
  `PounceOptionKind(const char*)` query would let us validate names and
  types the way the Ipopt plugin does.
* **`get_var_con_metadata`** (`var_string_md` & friends) has no C API
  surface. Low priority — Ipopt itself only echoes it back.
* **Thread safety.** CasADi can evaluate a `Function` from several
  threads (`map('thread')`). The plugin must document one
  `IpoptProblem` per thread; POUNCE's `Rc`-based core is not `Send`.

---

## 5. Recommendation

1. **Now (this branch):** the two POUNCE-side fixes above — done.
2. **Done in this branch:** a top-level `casadi/` directory mirroring
   `gams/` — plugin, Makefile (`fetch-src` / `install` / `test` /
   `examples`), parity tests, six examples, `casadi/README.md`, and user
   docs at `docs/src/casadi.md`.
3. **Next, if approved:**
   * a `pounce-casadi` wheel building the plugin per CasADi minor version
     against the matching source tree, with the ABI flags from §3 pinned
     in the build script, and a `register` command for the case where
     site-packages is not writable;
   * a CI job that installs a pip CasADi, builds the plugin, and runs
     `make test` — the parity checks are already written for it;
   * POUNCE-only extras once the basics have users: active-set-SQP
     working-set warm starts, the solve report, parametric sensitivity.
4. **Also cheap, independent:** teach the CLI ASL-style `-o<path>` so
   `nlpsol('S','ampl',...,{'solver':'pounce'})` works for the SX /
   no-parameter subset, and say plainly in the docs that it is a
   fallback.
5. **Later:** propose the same interface upstream to CasADi, citing the
   MadNLP precedent.

The thing left to decide is the wheel: whether POUNCE takes on a
per-CasADi-version build matrix in CI, or leaves the plugin as a
build-from-source integration like the native GAMS link. The C++ surface
itself is already here and tested.

---

## Reproducing §3

```bash
pip install casadi==3.7.2
cargo build --release -p pounce-cinterface
cd casadi && make fetch-src && make test && make examples
```
