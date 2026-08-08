# WebAssembly: POUNCE in the Browser

POUNCE's default build is pure Rust — no C, no Fortran, no BLAS to link —
so the entire solver compiles to WebAssembly and runs in a browser tab:
the AMPL `.nl` reader, the reverse-mode AD tape, the sparse LDL^T
factorization, and the interior-point algorithm. Nothing is sent to a
server.

Two pages ship with the docs, both published from `main` and both running
the solver locally in your tab:

- **[/demo](https://jkitchin.github.io/pounce/demo/)** — drop a `.nl` file
  on the page, see what is in the model, solve it, download the solution.
- **[/demo/python](https://jkitchin.github.io/pounce/demo/python/)** — write
  a **Pyomo** model in Python and solve it, via
  [Pyodide](https://pyodide.org).

To run them from a checkout:

```sh
rustup target add wasm32-wasip1               # once
crates/pounce-wasm/build.sh --serve           # the .nl page,   :8000
crates/pounce-wasm/build.sh --serve-python    # the Python page, :8000
```

Or `make wasm` to build the module without serving anything.

## Hosting it

Each page is a static directory (`crates/pounce-wasm/web/` and
`crates/pounce-wasm/web-python/`) — deploying either is a copy. Neither
needs a special server: no threads means no
`SharedArrayBuffer`, so none of the `Cross-Origin-Opener-Policy` /
`Cross-Origin-Embedder-Policy` headers that thread-enabled wasm requires,
and every URL the page fetches is relative, so it works under any base
path. If a host serves `.wasm` as something other than `application/wasm`,
the page falls back from streaming compilation to a buffered
`WebAssembly.instantiate` on its own.

GitHub Pages is what this repository uses: `.github/workflows/docs.yml`
builds the module and stages the two directories into the docs site at
`/demo/` and `/demo/python/`, so both ship with every docs deployment from
`main`. They are version-independent — one live build each, not one per
archived release tag.

## What you get

Dropping a model shows the problem summary POUNCE derives while building
its evaluator — sizes, degrees of freedom, how many rows are equalities,
how much of the model is nonlinear, Jacobian and Hessian sparsity, and how
the variable bounds break down. Solving streams the usual iteration table
into the page (that really is the solver's stdout) and reports the exit
status, KKT residuals, evaluation counts, and the solution vector next to
the `.col` / `.row` names when you drop those alongside the `.nl`.

Solve options are `ipopt.opt`-format text — the same option names the CLI
and the Python API take.

Three downloads come off a finished solve:

| Download | What it is |
| --- | --- |
| `.sol` | An AMPL solution file — byte-identical to what `pounce model.nl` writes, including the `ipopt_zL_out` / `ipopt_zU_out` reduced-cost suffixes. AMPL and Pyomo read it back. |
| CSV | One row per variable and per constraint: name, value, bounds, multiplier. |
| log | The solver output, as printed. |

The `.sol` and CSV are formatted inside wasm from the full solution, not
from the table on screen — the page truncates long vectors at 2,000 rows to
stay renderable, and a download that stopped there would be worse than none.

Dropping a new file resets everything: the page throws away its worker and
starts a fresh wasm instance, so no parsed model, solver state, or grown
heap carries from one file into the next.

## The Python page

Pyodide supplies CPython compiled to WebAssembly; `micropip` installs Pyomo
(a `py3-none-any` wheel — nothing to build). You write an ordinary Pyomo
model, and:

```python
from pyomo.environ import *
import pounce_browser

m = ConcreteModel()
m.x = Var([1, 2], initialize=0.5, bounds=(-10, 10))
m.circle = Constraint(expr=m.x[1]**2 + m.x[2]**2 == 1)
m.obj = Objective(expr=m.x[1])
m.dual = Suffix(direction=Suffix.IMPORT)

res = pounce_browser.solve(m, options="print_level 5")
print(res.status, value(m.x[1]), m.dual[m.circle])
```

`solve()` writes the model with Pyomo's own NL writer, hands the `.nl` text
to the POUNCE wasm module, and loads the returned `.sol` back onto the
model, so `x.value` and `model.dual[c]` read exactly as after a local solve.
Variables and rows are matched by the writer's own ordering
(`NLWriterInfo.variables` / `.constraints`), so the mapping cannot drift
from the file it just wrote — `crates/pounce-wasm/tests/pyomo_roundtrip.py`
pins that with a model whose optimum and multipliers are known in closed
form, and CI runs it on every PR with Node standing in for the browser.

The script box is a small editor — Python highlighting, line numbers,
Tab/Shift-Tab indent, indentation carried across Enter — built from a
highlighted `<pre>` behind a transparent `<textarea>` so the caret and undo
stay native. No editor library: a CDN dependency would be absent in exactly
the offline setup `?pyodide=` exists for.

Two wasm runtimes are in play — Pyodide's CPython and POUNCE — with separate
memories; all that crosses between them is `.nl` text one way and JSON plus
`.sol` text the other.

This is Pyomo's modelling layer, not POUNCE's own Python API: the model
reaches the solver as a file, so there are no Python callbacks mid-solve.
Running the real `pounce-solver` package in a browser would mean building
the compiled extension for Pyodide (emscripten), which this does not do.

The page needs the network for its first load — Pyodide from a CDN, Pyomo
from PyPI, about 15 MB, cached afterwards. Self-host both and pass
`?pyodide=…&pyomo=…` to avoid it entirely; see
`crates/pounce-wasm/web-python/README.md`. The solve itself is local either
way.

## Numerical parity with the native build

The wasm build runs the same code, so it produces the same answers. Over
all 37 `.nl` fixtures in `crates/pounce-cli/tests/fixtures`, driven through
the same entry points on both sides:

- exit status: identical on 37 of 37
- iteration count: identical on 37 of 37
- objective: bit-identical on 34 of the 36 that return one; the two
  exceptions (`scaled_feasible_a`, `feasible_x0_sentinel_bound`) differ by
  one ulp, at objectives of 4.5e-10 and 7.1e-11

The 37th (`presolve_overflow_feasible`) returns `InvalidNumberDetected`
with no objective on either side — that is the fixture's job.

Speed is what you would expect from wasm. Solver-internal wall time, same
build, same code path, native `x86_64` vs `wasm32-wasip1` under Node:

| model | n × m | native | wasm | ratio |
| --- | --- | --- | --- | --- |
| `pooling_rt2stp` | 46 × 72 | 9.9 ms | 40 ms | 4.0× |
| `jit1` | 25 × 32 | 8.1 ms | 43 ms | 5.3× |
| `airport` | 84 × 42 | 20 ms | 80 ms | 4.1× |
| `autocorr_bern55-06` | 56 × 1 | 50 ms | 101 ms | 2.0× |
| `deb7` | 813 × 897 | 461 ms | 556 ms | 1.2× |

The larger the model, the closer wasm gets: small solves are dominated by
per-call overhead, while big ones spend their time in the sparse
factorization, where the gap narrows. Nothing here is tuned — no SIMD, no
`wasm-opt`.

## How it is put together

| Piece | What it is |
| --- | --- |
| `crates/pounce-wasm` | C-ABI entry points (`pounce_load`, `pounce_solve`, the exporters), bytes in / JSON out |
| `crates/pounce-wasm/web` | the `.nl` page: `index.html`, `app.js`, `worker.js`, `wasi.js` |
| `crates/pounce-wasm/web-python` | the Pyodide page, plus `pounce_browser.py` — the Pyomo ↔ POUNCE shim |
| `crates/pounce-wasm/build.sh` | builds the module and stages it into both pages |

The target is `wasm32-wasip1`, not `wasm32-unknown-unknown`. WASI gives the
solver a clock (`std::time::Instant::now()` panics on
`wasm32-unknown-unknown`, and POUNCE times every solve) and a stdout to
write its iteration table to. Browsers do not implement WASI, so the page
carries a ~60-line shim, `wasi.js`, which answers `clock_time_get` from
`performance.now()` and turns each `fd_write` into a line in the log pane.
That shim is the entire cost of the approach: no `wasm-bindgen`, no npm, no
build step beyond `cargo build`.

A solve is one synchronous call into wasm that can run for seconds, so the
module lives in a web worker and the page stays responsive.

Payloads cross the boundary as a little-endian `u32` byte count followed by
that many UTF-8 bytes. Reading a length rather than scanning for a NUL
terminator keeps the reader's correctness independent of what is *in* the
payload, and lets a bad pointer or length be reported as exactly that
instead of surfacing later as an unrelated parse error.

## Limitations

- **Single-threaded.** No threads are spawned; rayon-parallel paths run
  serially. Results are unaffected.
- **No in-process iteration-history capture.** The Wasm build does not install
  the thread-scoped tracing collector, so `pounce_rs::Nlp::capture_iterations()`
  and `IpoptApplication::enable_iter_history()` (including the CLI/Python
  full-detail paths) return an empty `stats.iterations` vector. The scalar
  `iteration_count` remains available.
- **No live-iterate inspection.** `GetIpoptCurrentIterate` and
  `GetIpoptCurrentViolations` report “not available” because the callback
  context is not installed in the Wasm build.
- **No AMPL imported functions.** A model that calls compiled-C external
  functions (`funcadd_ASL` — IDAES property packages, for instance) needs a
  dynamic loader the browser sandbox does not provide. The summary flags
  such a model rather than failing mysteriously mid-solve.
- **No HSL.** The optional `ma57` backend links Fortran; the wasm build
  uses the default FERAL backend, like any stock `cargo build`.
- **2.4 MB module**, about 800 kB gzipped over the wire.

## Embedding it in your own page

`crates/pounce-wasm` is a thin shim you can copy or fork. The ABI is four
exports — allocate, load, solve, free — and every payload is JSON:

```js
const summary = fromWasm(wasm.pounce_load(nlPtr, nlLen, 0, 0, 0, 0));
const result  = fromWasm(wasm.pounce_solve(optsPtr, optsLen));
const solFile = fromWasm(wasm.pounce_solution_sol());   // AMPL .sol text
const csv     = fromWasm(wasm.pounce_solution_csv());   // every row
```

Both entry points catch panics and return `{"error": …}`, so a malformed
model cannot trap the instance. See `crates/pounce-wasm/web/README.md` for
the full walkthrough and `crates/pounce-wasm/tests/smoke.mjs` for a
headless (Node) driver of the same ABI.
