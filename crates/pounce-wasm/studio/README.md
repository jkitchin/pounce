# pounce studio modules

The wasm build for a front end that looks inside a solve: the live
debugger, the solve report and its analysis, the options catalog, and point
and grid evaluation of the model. The front end itself (menus, matrix
views, the assistant) lives in its consumer, the optimization book; this
directory holds what it loads.

```sh
crates/pounce-wasm/build.sh --studio          # optimized (a few minutes)
crates/pounce-wasm/build.sh --studio --fast   # skip the post-Asyncify optimizer (~40 s, ~2x size)
node crates/pounce-wasm/tests/debug_protocol.mjs
```

| File | What it is |
| --- | --- |
| `pounce-studio.wasm` | the `studio` feature build, `wasm-opt -Oz`; solve, report, analysis, catalog, evaluation |
| `pounce-studio-debug.wasm` | the same module after Binaryen's Asyncify pass; adds the live debugger |
| `debug-host.js` | instantiates either module and wraps the C ABI; `canDebug` says which one it has |
| `wasi.js` | copied from `../web/` by the build |

The two `.wasm` files and `wasi.js` are build output (gitignored).

## Why two modules

The debugger reads its commands with a blocking line read on stdin. Under
Asyncify that read (`fd_read`) can suspend into JavaScript and resume when
the page sends a command, with no threads and no `SharedArrayBuffer`, so the
page needs no cross-origin-isolation headers. But Asyncify instruments every
function that can reach the read, and the debugger hook is an indirect call
inside the iteration loop, so it slows every solve. Measured on `deb7`, five
interleaved solves per module: 1000 ms (sd 96) without the pass, 1617 ms (sd
253) with it. A front end solves with `pounce-studio.wasm` and switches to
`pounce-studio-debug.wasm` to debug.

## Driving it

```js
import { createStudio } from './debug-host.js';

const studio = await createStudio(await (await fetch('pounce-studio-debug.wasm')).arrayBuffer(), {
  onEvent(ev) {
    // the `--debug-json` protocol: hello, pause, progress, result, terminated.
    // The solve waits at each pause until a command resumes it.
    if (ev.event === 'pause') {
      if (ev.iter === 1) studio.send({ cmd: 'viz', args: ['kkt'], id: 1 }); // → result.data
      studio.send(ev.iter === 0 ? 'step' : 'continue');
    }
  },
});
studio.load(nlText);                       // → problem summary
const catalog = studio.optionsCatalog();   // every option, typed, with descriptions
const result = await studio.debugSolve('max_iter 200\n');
const report = studio.report();            // pounce.solve-report/v1
const analysis = studio.analyze();         // findings + glossary, stalls, trace
const grid = studio.slice({ i: 0, j: 1, range_i: [-2, 2], range_j: [-2, 2], nx: 101, ny: 101 });
```

`send` takes a command string or `{cmd, args, id}`; `close()` ends stdin,
which aborts a paused solve as it does for the CLI. In this build `viz` and
`save` return their payloads inline (`data` of the `result` event) and `ask
--prompt-only` returns the prompt for a model of the page's own; see
`docs/src/debugger.md`. An async `pause` while the solve runs is not
available (`hello.capabilities.pause_command` is `false`): set a breakpoint
or `stop-at` before continuing instead.

A debug solve runs on the thread that calls it, between awaits, so a page
runs the host in a web worker like the `.nl` page does.
