# POUNCE in the browser

A static page where you write Python and solve it with POUNCE — both running
in the tab, nothing installed, nothing uploaded.

```sh
rustup target add wasm32-wasip1                # once
crates/pounce-wasm/build.sh --serve-python     # http://localhost:8000
```

## Three routes to the solver, chosen by what a script imports

```python
import pounce           # the pounce-solver package itself, built for emscripten
import pounce_browser   # Pyomo writes an .nl, a separate POUNCE wasm solves it
import pyomo_pounce     # Pyomo with pyomo-pounce's sensitivity, on the package
```

None is installed up front, because each is large and a script usually needs
one. The worker gets CPython running, then installs whichever route the first
run asks for and caches it. The example dropdown offers all three.

`import matplotlib` is another on-demand install, orthogonal to the routes. See
[Figures](#figures) below.

### `import pounce` — the real package

`crates/pounce-py` compiled for `wasm32-unknown-emscripten` and installed by
`micropip`. The extension module lives inside Pyodide's own wasm instance, so
`minimize`, `curve_fit`, the `Problem` class, `pounce.sensitivity` and numpy
arrays behave exactly as they do on a desktop install — callbacks are called
during the solve, not marshalled through a file.

### `import pounce_browser` — the Pyomo route

Two independent wasm instances, joined by two text formats:

```
  Pyodide (CPython + Pyomo, wasm)                POUNCE (wasm)
        │                                              │
        │  model.write() → .nl text  ────────────────►  parse + solve
        │                                              │
        │  ◄──────────────  JSON result + .sol text  ───┘
        │
        └─ pounce_browser.load_solution() → x.value, model.dual[c]
```

`pounce_browser.py` is the Python side. `solve(model, options=…)` writes the
`.nl` with Pyomo's own NL writer, calls the backend the worker installed
(which drives the POUNCE wasm exports), parses the `.sol`, and loads the
values back onto the model. Positions come from `NLWriterInfo.variables` /
`.constraints` — the writer's own column and row order — so the mapping
cannot drift from the file it just wrote.

Options are `ipopt.opt`-format text, the same names the CLI takes.

That round trip is tested off-browser, with Node standing in for the page:
`crates/pounce-wasm/tests/pyomo_roundtrip.py` solves a model whose optimum,
active set, and multipliers are known in closed form and checks the values
landed on the right components. CI runs it on every PR.

The route survives because it is not redundant: it is the one that keeps
working when no emscripten wheel is deployed, and it is how a Pyomo model
reaches POUNCE without a Pyomo-to-`Problem` translation layer.

### `import pyomo_pounce`: Pyomo with sensitivity

pyomo-pounce runs in Pyodide on the emscripten wheel. Its solves and its
sensitivity calls (`declare_sens_param`, `sens_jacobian`, `sens_solution`,
`sens_covariance` and the rest) write the model to `.nl` text, read it with
`pounce.read_nl` and answer from a `pounce.Solver` session in the same wasm
instance, as on a desktop. A plain `SolverFactory("pounce").solve(m)` takes
that in-process path too, because Pyodide cannot start the `pounce`
executable.

The route installs numpy, the pounce-solver wheel and pyomo-pounce without
their dependencies, and Pyomo from PyPI. pyomo-pounce uses none of
pounce-solver's scipy paths (`minimize`, `curve_fit`, the QP helpers) and
needs pandas only for `Jacobian.to_dataframe()`, so neither is loaded.
`build.sh` builds the pyomo-pounce wheel from the checkout into `wheels/`,
at the pounce-solver wheel's version.

`crates/pounce-wasm/tests/pyomo_pounce_pyodide.mjs` runs a declared-parameter
solve, `sens_jacobian`, `sens_solution` and a plain solve under Pyodide in
Node, on a model whose sensitivities are known in closed form.

## The emscripten wheel

`wheels/` holds the `pounce-solver` build for this page, plus
`wheels/pounce-wheel.json` naming it:

```json
{
  "wheel": "pounce_solver-0.11.0-cp39-abi3-emscripten_4_0_9_wasm32.whl",
  "version": "0.11.0",
  "pyodide_version": "0.28.3",
  "emscripten_version": "4.0.9"
}
```

The worker reads the manifest rather than hard-coding a file name: an
emscripten wheel's ABI tag carries the exact emscripten version Pyodide was
built with, so the name moves whenever either version does. It also compares
`pyodide_version` against its own pin and fails with that sentence, instead of
letting micropip report a version skew as "no matching distribution".

Build or rebuild it with:

```sh
crates/pounce-wasm/build-wheel.sh            # build + stage into wheels/
crates/pounce-wasm/build-wheel.sh --check    # verify the staged wheel matches the pins
```

The script pins Pyodide, pyodide-build, emscripten, the Rust nightly, and the
wasm-exception-handling sysroot together, and it explains in its own header why
each pin is load-bearing — in particular that the emsdk **must** be the one
`pyodide xbuildenv install-emscripten` produces, because Pyodide patches
emscripten's side-module export check and a stock emsdk cannot link *any* Rust
side module. The script hard-fails with that message if the patch is missing.

If no wheel is deployed, `import pounce` fails with a sentence saying so and
pointing at the script; the Pyomo route is unaffected.

## The editor

`editor.js` is a ~150-line Python editor: highlighting, line numbers,
Tab / Shift-Tab block indent, and indentation carried across Enter (a level
deeper after a colon). A highlighted `<pre>` sits under a transparent
`<textarea>`, so the caret, selection, undo, IME, and screen-reader
behaviour stay the browser's rather than being re-implemented on a
`contenteditable`.

It is written rather than imported on purpose. CodeMirror or Ace from a CDN
would add bracket matching and autocomplete, at the cost of a second
version-pinned network dependency — one that would be missing in exactly the
offline / self-hosted setup `?pyodide=` exists to serve. If the page ever
wants an IDE, this is the one file to replace.

The tokenizer has its own tests (`crates/pounce-wasm/tests/editor_tokens.mjs`,
run in CI): escaped quotes, triple-quoted strings, f-strings, `#` inside a
string versus a comment, and HTML in the source never reaching the DOM as
markup.

Ctrl/⌘-Enter runs. **Cancel** terminates the worker and starts a fresh one —
a solve runs synchronously inside it, so nothing short of killing the thread
can interrupt one; the replacement re-downloads Pyodide (from cache) and
reinstalls whichever route the next script imports. The theme picker is
auto / light / dark, resolved before first paint and remembered in
`localStorage`.

## Figures

`import matplotlib` (or `pylab`, or `mpl_toolkits`) installs matplotlib on the
first run that asks for it, and `plt.show()` renders the figure on the page.

The worker has no DOM, so Pyodide's interactive backends — which draw into a
canvas element — cannot run here at all: they fail at import, before a script
gets to say anything about what it wants. `MPLBACKEND=AGG` is set as soon as
CPython starts, so matplotlib renders to a buffer instead, and `plt.show()` is
replaced by a function that saves every open figure to a PNG and posts it to
the page. Agg's own `show()` is a no-op that warns, so without the
substitution a script written for a desktop would print a warning and produce
nothing.

The end of a run flushes whatever figures are still open, on the exception
path too: a script that plots its data and then fails should still show the
plot, which is usually the thing that explains the failure. Figures are
cleared when the next run starts, so a re-run that plots nothing does not
leave the previous one on screen looking current.

The transport is a base64 PNG on the worker's message channel and an `<img>`
on the other side — no canvas, no proxy object whose lifetime the page has to
manage. The cost is that figures are not interactive: no pan, no zoom, no
`%matplotlib widget`.

## What it loads, and from where

| Piece | Source | Size | When |
| --- | --- | --- | --- |
| Pyodide | jsDelivr CDN, version-pinned | ~10 MB | on load |
| `pounce-solver` | `./wheels/`, same origin | 3.4 MB | first `import pounce` |
| POUNCE (`.nl` route) | `./pounce.wasm`, same origin | 2.4 MB | first `import pounce_browser` |
| Pyomo | PyPI, via `micropip` (`py3-none-any` wheel) | ~4 MB | first `import pounce_browser` |
| matplotlib | Pyodide's own package repository | ~9.5 MB | first `import matplotlib` |

The solve itself is entirely local — the CDN and PyPI fetches are the Python
runtime arriving, not your model leaving. They are also the only reason this
app needs the network at all; the `.nl` app next door needs none.

To run without either, host them yourself and point the page at them:

```
index.html?pyodide=/vendor/pyodide/&pyomo=/vendor/pyomo-6.10.1-py3-none-any.whl,/vendor/ply-3.11-py2.py3-none-any.whl
```

`?pyodide=` takes a directory URL (with a trailing slash); `?pyomo=` takes a
comma-separated list of wheel URLs, installed in order; `?pounce=` takes a
single wheel URL, bypassing the manifest.

## Limitations

- **The wheel is pinned to one Pyodide release.** `wheels/` is valid for
  Pyodide 0.28.3 and nothing else. Bumping `PYODIDE_VERSION` in `worker.js`
  without rerunning `build-wheel.sh` makes `import pounce` fail — deliberately
  and by name, but it fails.
- **The log arrives at the end.** The solver writes to stdout while Python
  is blocked in the solve, so the whole iteration table appears when the
  solve returns rather than streaming line by line.
- **First load is slow** — ~13 MB of runtime before a script runs, cached by
  the browser afterwards. The solve is the fast part.
- Everything the `.nl` app cannot do, this cannot either: single-threaded,
  no AMPL imported functions, no HSL.
