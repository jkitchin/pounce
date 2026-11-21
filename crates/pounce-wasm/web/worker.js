// Worker that owns the POUNCE wasm instance.
//
// A solve is a single synchronous call into wasm that can run for seconds,
// so it lives off the main thread; the page stays responsive and the
// solver's console output is streamed back as it is produced.
//
// One worker holds one model. The page creates a fresh worker for every
// file it loads, which is also how it resets: a new instance starts with a
// new linear memory, so nothing — parsed model, solver state, heap grown by
// the last solve — can survive into the next file.

import { createWasi } from './wasi.js';

let exports = null;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

// With `wasm_iterate_callback yes` in the options, the module calls back once
// per iteration. Only the statistics are forwarded: the page plots the
// objective and never needs x, which can be large.
const wasi = createWasi((text) => self.postMessage({ type: 'log', text }), {
  onIterate: ({ iter, restoration, objective }) => {
    self.postMessage({ type: 'iterate', iter, restoration, objective });
  },
});

const ready = (async () => {
  let instance;
  try {
    ({ instance } = await WebAssembly.instantiateStreaming(fetch('./pounce.wasm'), wasi.imports));
  } catch {
    // Streaming compilation needs `Content-Type: application/wasm`; not
    // every static server sets it, so fall back to buffering the module.
    const bytes = await (await fetch('./pounce.wasm')).arrayBuffer();
    ({ instance } = await WebAssembly.instantiate(bytes, wasi.imports));
  }
  wasi.bind(instance);
  exports = instance.exports;
})();

/** Copy a JS string into wasm memory. Returns [ptr, len]; ptr 0 for empty. */
function intoWasm(str) {
  if (!str) return [0, 0];
  const bytes = encoder.encode(str);
  const ptr = exports.pounce_alloc(bytes.length);
  if (!ptr) throw new Error('wasm allocation failed');
  new Uint8Array(exports.memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}

/**
 * Read a returned payload: a little-endian u32 byte count followed by that
 * many UTF-8 bytes. Returns null for a null pointer, which entry points use
 * to mean "nothing to give you" (e.g. a .sol before any solve).
 *
 * The length is read rather than found by scanning for a terminator, so a
 * pointer or a length that does not describe live memory is reported as
 * exactly that instead of surfacing later as a mangled parse error.
 */
function fromWasm(ptr) {
  if (!ptr) return null;
  const memory = new Uint8Array(exports.memory.buffer);
  if (ptr < 0 || ptr + 4 > memory.length) {
    throw new Error(`wasm returned pointer ${ptr}, outside its ${memory.length}-byte memory`);
  }
  const len = new DataView(exports.memory.buffer).getUint32(ptr, true);
  if (ptr + 4 + len > memory.length) {
    throw new Error(
      `wasm payload at ${ptr} claims ${len} bytes, past the end of its ` +
        `${memory.length}-byte memory`,
    );
  }
  const text = decoder.decode(memory.subarray(ptr + 4, ptr + 4 + len));
  exports.pounce_free_payload(ptr);
  return text;
}

/** Same, parsed as JSON, with the raw text quoted if it will not parse. */
function fromWasmJson(ptr) {
  const text = fromWasm(ptr);
  if (text === null) throw new Error('wasm returned no payload');
  try {
    return JSON.parse(text);
  } catch (err) {
    const head = text.slice(0, 200);
    const tail = text.length > 200 ? ` … ${text.slice(-80)}` : '';
    throw new Error(
      `wasm returned ${text.length} bytes that are not JSON (${err.message}): ${head}${tail}`,
    );
  }
}

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    await ready;
    if (msg.type === 'load') {
      const args = [msg.nl, msg.col ?? '', msg.row ?? ''].map(intoWasm);
      try {
        const summary = fromWasmJson(exports.pounce_load(...args.flat()));
        self.postMessage({ type: 'summary', summary });
      } finally {
        for (const [ptr, len] of args) if (ptr) exports.pounce_dealloc(ptr, len);
      }
    } else if (msg.type === 'solve') {
      const [ptr, len] = intoWasm(msg.options ?? '');
      const started = performance.now();
      try {
        const result = fromWasmJson(exports.pounce_solve(ptr, len));
        result.browser_ms = performance.now() - started;
        self.postMessage({ type: 'result', result });
      } finally {
        if (ptr) exports.pounce_dealloc(ptr, len);
      }
    } else if (msg.type === 'export') {
      // Formatted on demand: a .sol or CSV for a large model is megabytes
      // that most solves never download.
      const text = fromWasm(
        msg.format === 'sol' ? exports.pounce_solution_sol() : exports.pounce_solution_csv(),
      );
      self.postMessage({ type: 'export', text, filename: msg.filename, mime: msg.mime });
    }
  } catch (err) {
    self.postMessage({
      type: 'fatal',
      request: msg.type,
      message: String(err && err.message ? err.message : err),
    });
  }
};
