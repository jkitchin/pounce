# POUNCE + Pyomo in the browser

A static page where you write a Pyomo model in Python and solve it with
POUNCE — both running in the tab, nothing installed, nothing uploaded.

```sh
rustup target add wasm32-wasip1                # once
crates/pounce-wasm/build.sh --serve-python     # http://localhost:8000
```

## How the pieces fit

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

## What it loads, and from where

| Piece | Source | Size |
| --- | --- | --- |
| POUNCE | `./pounce.wasm`, same origin | 2.4 MB |
| Pyodide | jsDelivr CDN, version-pinned | ~10 MB |
| Pyomo | PyPI, via `micropip` (`py3-none-any` wheel) | ~4 MB |

The solve itself is entirely local — the CDN and PyPI fetches are the Python
runtime arriving, not your model leaving. They are also the only reason this
app needs the network at all; the `.nl` app next door needs none.

To run without either, host them yourself and point the page at them:

```
index.html?pyodide=/vendor/pyodide/&pyomo=/vendor/pyomo-6.10.1-py3-none-any.whl,/vendor/ply-3.11-py2.py3-none-any.whl
```

`?pyodide=` takes a directory URL (with a trailing slash); `?pyomo=` takes a
comma-separated list of wheel URLs, installed in order.

## Limitations

- **Pyomo only.** The model reaches POUNCE as an `.nl` file, so this is
  Pyomo's modelling layer, not POUNCE's own Python API — no `Problem`
  callbacks, no numpy arrays crossing into the solver mid-iteration. The
  `pounce-solver` package is a compiled extension; running *that* in the
  browser would need a Pyodide-targeted (emscripten) build.
- **The log arrives at the end.** The solver writes to stdout while Python
  is blocked in the backend call, so the whole iteration table appears when
  the solve returns rather than streaming line by line.
- **No in-process iteration history or live-iterate inspection.** The WASI
  backend leaves `stats.iterations` empty and reports
  `GetIpoptCurrentIterate` / `GetIpoptCurrentViolations` as unavailable.
- **First load is slow** — ~15 MB of runtime, cached by the browser
  afterwards. The solve is the fast part.
- Everything the `.nl` app cannot do, this cannot either: single-threaded,
  no AMPL imported functions, no HSL.
