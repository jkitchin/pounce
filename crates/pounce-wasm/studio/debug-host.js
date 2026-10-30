// Host for the studio build's live debugger (`pounce_debug_solve`).
//
// The debugger reads its commands with a blocking line read on stdin, which
// under WASI is an `fd_read` on fd 0. A page cannot block, so the studio
// module is post-processed with Binaryen's Asyncify (`build.sh --studio`):
// when a command is not ready, `fd_read` *unwinds* the wasm stack into a
// buffer and returns to JavaScript; once the front end sends a command, the
// call is *rewound* to the same `fd_read`, which now has bytes to return. The
// solve stays suspended at the debugger's pause in between, with no threads
// and no SharedArrayBuffer (so no cross-origin-isolation headers, which would
// stop the page loading Pyodide or model weights from a CDN).
//
// The protocol on stdout is the one `pounce --debug-json` speaks: one JSON
// object per line (`hello`, `pause`, `progress`, `result`, `terminated`).
//
// Works in a browser worker and in Node (see tests/debug_protocol.mjs).

// `wasi.js` is the demo page's shim, copied here by `build.sh --studio`.
import { createWasi } from './wasi.js';

const ERRNO_SUCCESS = 0;
const ERRNO_BADF = 8;

// Asyncify's saved-stack buffer. The debugger pauses inside the solver's
// iteration loop, a few dozen frames deep; 1 MiB is far more than that
// needs, and an overflow traps loudly rather than corrupting anything.
const ASYNCIFY_STACK = 1 << 20;

// `asyncify_get_state()` values.
const STATE_NORMAL = 0;
const STATE_UNWINDING = 1;
const STATE_REWINDING = 2;

/**
 * Instantiate the studio module.
 *
 * @param {BufferSource | WebAssembly.Module} wasmSource
 * @param {object} handlers
 * @param {(event: object) => void} handlers.onEvent each protocol event
 * @param {(text: string) => void} [handlers.onText] stdout lines that are
 *   not JSON (none in the debugger's pure-protocol mode, but a plain
 *   `pounce_solve` on the same instance prints its iteration table here)
 */
export async function createStudio(wasmSource, { onEvent, onText = () => {} }) {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  // stdout → lines → events
  let partial = '';
  const onStdout = (chunk) => {
    partial += chunk;
    let nl;
    while ((nl = partial.indexOf('\n')) >= 0) {
      const line = partial.slice(0, nl);
      partial = partial.slice(nl + 1);
      if (line.startsWith('{')) {
        let ev = null;
        try {
          ev = JSON.parse(line);
        } catch {
          // fall through: not a protocol line after all
        }
        if (ev !== null) {
          onEvent(ev);
          continue;
        }
      }
      onText(line + '\n');
    }
  };

  const wasi = createWasi(onStdout);
  const imp = wasi.imports.wasi_snapshot_preview1;

  // stdin: bytes waiting to be read, and a promise for the next command
  // when there are none.
  let stdin = new Uint8Array(0);
  let stdinClosed = false;
  let wake = null; // resolves the wait for the next command
  let waiting = null; // the promise fd_read is suspended on

  let instance = null;
  let dataAddr = 0;
  const ex = () => instance.exports;

  function fillIovs(iovs, iovsLen, nread) {
    const mem = ex().memory.buffer;
    const view = new DataView(mem);
    let total = 0;
    for (let i = 0; i < iovsLen && stdin.length > 0; i++) {
      const ptr = view.getUint32(iovs + i * 8, true);
      const len = view.getUint32(iovs + i * 8 + 4, true);
      const n = Math.min(len, stdin.length);
      new Uint8Array(mem, ptr, n).set(stdin.subarray(0, n));
      stdin = stdin.subarray(n);
      total += n;
    }
    view.setUint32(nread, total, true);
  }

  imp.fd_read = (fd, iovs, iovsLen, nread) => {
    if (fd !== 0) return ERRNO_BADF;
    if (ex().asyncify_get_state() === STATE_REWINDING) {
      // Back from the wait: the command that ended it is in `stdin` (or the
      // session was closed, which reads as EOF).
      ex().asyncify_stop_rewind();
    }
    if (stdin.length > 0 || stdinClosed) {
      fillIovs(iovs, iovsLen, nread); // 0 bytes once closed: EOF
      return ERRNO_SUCCESS;
    }
    // Nothing to read: suspend until `send` or `close`.
    waiting = new Promise((resolve) => {
      wake = resolve;
    });
    const view = new DataView(ex().memory.buffer);
    view.setUint32(dataAddr, dataAddr + 8, true);
    view.setUint32(dataAddr + 4, dataAddr + 8 + ASYNCIFY_STACK, true);
    ex().asyncify_start_unwind(dataAddr);
    return ERRNO_SUCCESS; // ignored: the stack is unwinding
  };

  const module =
    wasmSource instanceof WebAssembly.Module ? wasmSource : await WebAssembly.compile(wasmSource);
  instance = await WebAssembly.instantiate(module, wasi.imports);
  wasi.bind(instance);
  // The plain studio module (pounce-studio.wasm) does everything but debug;
  // the asyncified one (pounce-studio-debug.wasm) adds the live debugger.
  const canDebug = typeof ex().asyncify_start_unwind === 'function';
  if (canDebug) {
    // `pounce_alloc` is byte-aligned; Asyncify's header is two i32s, so
    // over-allocate and round up. Held for the instance's lifetime.
    const raw = ex().pounce_alloc(8 + ASYNCIFY_STACK + 8);
    if (!raw) throw new Error('could not allocate the Asyncify stack');
    dataAddr = (raw + 7) & ~7;
  }

  function intoWasm(str) {
    if (!str) return [0, 0];
    const bytes = encoder.encode(str);
    const ptr = ex().pounce_alloc(bytes.length);
    if (!ptr) throw new Error('wasm allocation failed');
    new Uint8Array(ex().memory.buffer, ptr, bytes.length).set(bytes);
    return [ptr, bytes.length];
  }

  function fromWasm(ptr) {
    if (!ptr) return null;
    const len = new DataView(ex().memory.buffer).getUint32(ptr, true);
    const text = decoder.decode(new Uint8Array(ex().memory.buffer, ptr + 4, len));
    ex().pounce_free_payload(ptr);
    return text;
  }

  function freeInput([ptr, len]) {
    if (ptr) ex().pounce_dealloc(ptr, len);
  }

  function callJson(fn, request) {
    const arg = intoWasm(JSON.stringify(request));
    try {
      return JSON.parse(fromWasm(fn(...arg)));
    } finally {
      freeInput(arg);
    }
  }

  let running = false;

  return {
    exports: instance.exports,

    /** Whether this is the asyncified build, which can `debugSolve`. */
    canDebug,

    /** Parse an `.nl` (plus optional `.col` / `.row` text); returns the summary. */
    load(nl, col = '', row = '') {
      const args = [intoWasm(nl), intoWasm(col), intoWasm(row)];
      try {
        return JSON.parse(fromWasm(ex().pounce_load(...args.flat())));
      } finally {
        args.forEach(freeInput);
      }
    },

    /** Every registered option, for the options editor. */
    optionsCatalog() {
      return JSON.parse(fromWasm(ex().pounce_options_catalog()));
    },

    /** The model's functions at `x` (default: the starting point). */
    evaluate(x = null) {
      return callJson(ex().pounce_eval, x === null ? {} : { x });
    },

    /**
     * Objective and constraints on a grid through a point, for the slice
     * view: `{x, i, j, range_i, range_j, nx, ny, rows}` (see `pounce_eval_slice`).
     */
    slice(request) {
      return callJson(ex().pounce_eval_slice, request);
    },

    /** The last solve as an AMPL `.sol` file (null before a solve). */
    solutionSol() {
      return fromWasm(ex().pounce_solution_sol());
    },

    /** The last solve as CSV, one row per variable and constraint (null before a solve). */
    solutionCsv() {
      return fromWasm(ex().pounce_solution_csv());
    },

    /** The last solve's `pounce.solve-report/v1` document, or null. */
    report() {
      return JSON.parse(fromWasm(ex().pounce_solve_report()));
    },

    /** studio-core's analysis of the last solve (findings, stalls, trace), or null. */
    analyze() {
      return JSON.parse(fromWasm(ex().pounce_analyze()));
    },

    /** Log-column, finding and citation glossary, for tooltips. */
    glossary() {
      return JSON.parse(fromWasm(ex().pounce_glossary()));
    },

    /** Plain solve (no debugger); returns the result object. */
    solve(options = '') {
      const arg = intoWasm(options);
      try {
        return JSON.parse(fromWasm(ex().pounce_solve(...arg)));
      } finally {
        freeInput(arg);
      }
    },

    /**
     * Solve under the debugger. Resolves with the result object when the
     * solve ends; protocol events arrive through `onEvent` meanwhile, and
     * commands go in with `send`.
     */
    async debugSolve(options = '') {
      if (!canDebug) {
        throw new Error('this module has no live debugger: load pounce-studio-debug.wasm');
      }
      if (running) throw new Error('a debug solve is already running');
      running = true;
      stdin = new Uint8Array(0);
      stdinClosed = false;
      const arg = intoWasm(options);
      try {
        let ptr = ex().pounce_debug_solve(...arg);
        while (ex().asyncify_get_state() === STATE_UNWINDING) {
          ex().asyncify_stop_unwind();
          await waiting;
          ex().asyncify_start_rewind(dataAddr);
          ptr = ex().pounce_debug_solve(...arg);
        }
        if (ex().asyncify_get_state() !== STATE_NORMAL) {
          throw new Error('debug solve returned in the middle of an Asyncify unwind');
        }
        return JSON.parse(fromWasm(ptr));
      } finally {
        freeInput(arg);
        running = false;
      }
    },

    /**
     * Send one debugger command: a string (`"step"`) or an object
     * (`{cmd: "viz", args: ["kkt"], id: 7}`).
     */
    send(command) {
      if (stdinClosed) throw new Error('the debug session is closed');
      const line = (typeof command === 'string' ? command : JSON.stringify(command)) + '\n';
      const bytes = encoder.encode(line);
      const merged = new Uint8Array(stdin.length + bytes.length);
      merged.set(stdin);
      merged.set(bytes, stdin.length);
      stdin = merged;
      if (wake) {
        const w = wake;
        wake = null;
        w();
      }
    },

    /** Close stdin. At a pause this aborts the solve, as for the CLI. */
    close() {
      stdinClosed = true;
      if (wake) {
        const w = wake;
        wake = null;
        w();
      }
    },

    get running() {
      return running;
    },
  };
}
