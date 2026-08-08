# POUNCE in the browser

A static page that reads an AMPL `.nl` file, summarizes it, and solves it —
entirely client-side, with the POUNCE solver compiled to WebAssembly. No
server, no upload, no install.

```sh
rustup target add wasm32-wasip1     # once
crates/pounce-wasm/build.sh --serve # build + serve on http://localhost:8000
```

Drag a `.nl` file onto the page (optionally with its sibling `.col` / `.row`
name files, which AMPL writes under `option auxfiles rc;`).

Deploying is a matter of copying this directory — after `build.sh` has
staged `pounce.wasm` into it — to any static host. No headers to configure:
no threads means no `SharedArrayBuffer`, hence no COOP/COEP requirement, and
every fetch is relative so any base path works. This repository publishes it
to GitHub Pages at [`/pounce/demo/`](https://jkitchin.github.io/pounce/demo/)
from `.github/workflows/docs.yml`.

## What is here

| File | Role |
| --- | --- |
| `index.html` | markup + styles |
| `app.js` | file intake and rendering, main thread |
| `worker.js` | owns the wasm instance; runs load/solve off the UI thread |
| `wasi.js` | ~60-line `wasi_snapshot_preview1` host |
| `pounce.wasm` | build artifact, staged by `build.sh` (not in git) |

## Why WASI and not `wasm-bindgen`

The module targets `wasm32-wasip1` rather than `wasm32-unknown-unknown`,
which buys two things and costs one.

It buys a working clock: `std::time::Instant::now()` panics on
`wasm32-unknown-unknown`, and POUNCE calls it from ~20 places (per-solve
wall time, `max_wall_time`, the restoration timer). Under WASI it is
`clock_time_get`, which `wasi.js` answers from `performance.now()`. It also
buys stdout: the solver's iteration table is written to fd 1 as usual, and
the shim turns each `fd_write` into a message the page appends to the log
pane — the live output you see during a solve is the real thing, not a
reconstruction.

It costs a host shim, since browsers do not implement WASI natively. That
shim is `wasi.js`, and it is the whole cost: no bindings generator, no npm
install, no build step beyond `cargo build`. Fourteen WASI functions are
imported; four do real work (`fd_write`, `clock_time_get`, `random_get`,
`environ_*`) and the rest return errno, because the solver never touches a
filesystem on this path.

## The interface

`crates/pounce-wasm/src/lib.rs` exports a small C ABI — bytes in, JSON out:

```js
const [ptr, len] = intoWasm(nlFileText);
const summary = fromWasm(wasm.pounce_load(ptr, len, 0, 0, 0, 0));
const result  = fromWasm(wasm.pounce_solve(...intoWasm('max_iter 500\n')));
const solFile = fromWasm(wasm.pounce_solution_sol());   // AMPL .sol text
const csv     = fromWasm(wasm.pounce_solution_csv());   // every row, named
```

Options are `ipopt.opt`-format text, exactly as the CLI and the Python API
take them. Every entry point catches panics and reports `{"error": …}`, so a
malformed model does not trap the instance.

Each returned payload is a little-endian `u32` byte count followed by that
many UTF-8 bytes — no terminator to scan for, so a bad pointer or length is
reported as itself rather than as a downstream parse error. Release it with
`pounce_free_payload`.

The page throws its worker away and builds a fresh instance on every file
drop, which is what makes "drag a new model in" a real reset rather than a
re-render.

## Limitations

- **Single-threaded.** The build never spawns a thread, so rayon-parallel
  paths run serially. Nothing else changes: results match the native build.
- **No in-process iteration history.** `capture_iterations()` and
  `enable_iter_history()` return an empty `stats.iterations` vector in the
  Wasm build; the scalar iteration count remains available.
- **No live-iterate inspection.** `GetIpoptCurrentIterate` and
  `GetIpoptCurrentViolations` report “not available” in Wasm.
- **No AMPL imported functions.** Models that call compiled-C external
  functions (`funcadd_ASL`, e.g. IDAES property packages) need a dynamic
  loader the browser sandbox has none of. The summary flags such a model.
- **No HSL.** The `ma57` feature links Fortran; the wasm build uses the
  default pure-Rust FERAL backend, same as a stock `cargo build`.
- **2.4 MB module** (~800 kB over the wire with gzip), compiled once per
  page load in ~100 ms.
