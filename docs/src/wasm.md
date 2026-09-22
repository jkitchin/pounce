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

Pyodide supplies CPython compiled to WebAssembly, and there are two ways to
reach the solver from it. A script picks one by what it imports; neither is
installed until a run asks for it.

`import pounce` is the `pounce-solver` package itself, built for emscripten
and installed by `micropip` — the same API as a local install, running inside
Pyodide's own wasm instance, so callbacks are called during the solve rather
than marshalled through a file:

```python
import numpy as np
import pounce

res = pounce.minimize(
    lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2,
    np.array([-1.2, 1.0]),
    constraints=[{"type": "ineq", "fun": lambda x: 1.0 - x @ x}],
)
print(res.message, res.x)
```

`import pounce_browser` takes the Pyomo route instead: `micropip` installs
Pyomo (a `py3-none-any` wheel — nothing to build), you write an ordinary
Pyomo model, and:

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

`import matplotlib` is a third on-demand install, orthogonal to both routes:
the worker sets `MPLBACKEND=AGG` before anything imports matplotlib, since
Pyodide's interactive backends draw into the DOM and a worker has none, and
replaces `plt.show()` with a function that saves every open figure to a PNG
and posts it to the page. Figures still open when a script ends are flushed
too, including when it ends by raising. They arrive as `<img>` elements below
the output, so they are pictures rather than interactive canvases.

The script box is a small editor — Python highlighting, line numbers,
Tab/Shift-Tab indent, indentation carried across Enter — built from a
highlighted `<pre>` behind a transparent `<textarea>` so the caret and undo
stay native. No editor library: a CDN dependency would be absent in exactly
the offline setup `?pyodide=` exists for.

Two wasm runtimes are in play — Pyodide's CPython and POUNCE — with separate
memories; all that crosses between them is `.nl` text one way and JSON plus
`.sol` text the other.

The Pyomo route is the one that stays available with no wheel deployed, and
it is how a Pyomo model reaches POUNCE without a translation layer — but it
is Pyomo's modelling surface, not POUNCE's own Python API, and the model
reaches the solver as a file, so there are no Python callbacks mid-solve.
That is what `import pounce` is for.

The emscripten wheel is built by `crates/pounce-wasm/build-wheel.sh` and
staged into `web-python/wheels/` beside a manifest naming it and recording
the Pyodide and emscripten versions it was compiled against. A wheel is valid
for exactly one Pyodide build: the worker compares the manifest against its
own pin and says so in those words, rather than letting `micropip` report the
skew as a missing package. The script pins Pyodide, pyodide-build,
emscripten, the Rust nightly, and the wasm-exception-handling sysroot
together, and refuses to run against an emsdk that is not the one
`pyodide xbuildenv install-emscripten` produces — Pyodide patches
emscripten's side-module export check, and without that patch emscripten
cannot link any Rust side module at all. Rebuild the wheel whenever the
Pyodide pin moves; `build-wheel.sh --check` verifies a staged one.

The page needs the network for its first load — Pyodide from a CDN, about
13 MB with a route installed, another ~9.5 MB if a script plots, cached
afterwards. Self-host everything and
pass `?pyodide=…&pyomo=…&pounce=…` to avoid it entirely; see
`crates/pounce-wasm/web-python/README.md`. The solve itself is local either
way.

## Numerical parity with the native build

The wasm module runs the same solver code as the native CLI, and since
gh#960 it runs the same *driver* too: the restoration phase and the
second-opinion ladder are wired in exactly as the CLI, the C interface and
the Python frontend wire them.

Before that fix the shim had no restoration phase, so any solve whose line
search asked for one stopped immediately with `RestorationFailed` and
`restoration_calls = 0` — PGLib `case6468_rte` at iteration 54, the exact
iteration where the CLI enters restoration and goes on to solve. The parity
table that used to stand here did not see it, because it compared the shim
compiled to wasm against *the same shim* compiled natively: both lacked
restoration, so they agreed. Parity is measured against `pounce model.nl`
now.

Over the 52 fixtures in `crates/pounce-cli/tests/fixtures` that the CLI
routes to the NLP arm, same default options, native `aarch64-apple-darwin`
CLI vs `wasm32-wasip1` under Node:

- before gh#960: status or iteration count differed on 15 of 52
- after: identical status and iteration count on 48 of 52

The four that still differ all reach the same optimum; only the iteration
count moves (`pooling_rt2stp` 184 vs 109, `cresc4` 68 vs 69,
`scaled_feasible_a` 20 vs 22, `deb7` 131 vs 147). `case6468_rte`
(49 734 × 75 002) matches the CLI exactly — 146 iterations, 3 restoration
calls, objective 2 069 730.14512.

**Why wasm32 is not bit-identical to native.** It is the same source, and
the arithmetic in it agrees: `pow`, `exp`, `sin`, `sqrt` and `mul_add` are
bit-identical between the two targets, as is scalar code including a plain
dot product. The factorization is not. A fixed 4 000 × 4 000 sparse
symmetric matrix, factored and solved through the same FERAL version on
both targets, gives solutions one ulp apart. It is not parallelism —
native is byte-identical with FERAL's internal threading forced on or off
— and not the wasm SIMD proposal, since `-C target-feature=+simd128` does
not change the result: `pulp`, which dispatches FERAL's kernels, has no
wasm backend, so wasm runs scalar lanes where aarch64 runs NEON `f64x2`.
Both results are backward-stable; an interior-point trajectory is simply
free to amplify the difference.

`deb7` is the worked example, and the reason it now solves. It follows the
native trajectory for 83 iterations, diverges by one ulp in `inf_du` at
iteration 84, and used to end `Error_In_Step_Computation` at 160. That
status opened no second-opinion ladder, although `mu_strategy=adaptive`
recovers this exact failure; it opens one now, and the same binary reaches
the native optimum in 131 iterations. So a browser solve whose trajectory
the target's rounding has walked into a bad region now gets the same
second opinion every other frontend gets.

The other 45 fixtures are convex LPs, QPs and QCQPs that the CLI hands to a
specialised engine (`solver_selection=auto`). The wasm shim has no such
routing and always solves with the NLP interior point, so those are not
compared here.

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
| `crates/pounce-wasm/web-python` | the Pyodide page: the editor, plus `pounce_browser.py` — the Pyomo ↔ POUNCE shim — and `wheels/`, the emscripten `pounce-solver` build |
| `crates/pounce-wasm/build-wheel.sh` | builds `pounce-solver` for emscripten and stages it into `web-python/wheels/` |
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
