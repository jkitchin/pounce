// Solve an AMPL .nl file with the POUNCE wasm module and print
// {"result": …, "sol": …, "log": …} on stdout.
//
//   node solve_nl.mjs model.nl "max_iter 200" [path/to/pounce_wasm.wasm]
//
// This is the same C ABI the browser worker drives, exercised from Node so
// the Pyodide app's Python layer can be tested without a browser (see
// pyomo_roundtrip.py). Solver output goes to stderr rather than stdout so
// the JSON on stdout stays parseable.

import { readFileSync } from 'node:fs';
import { WASI } from 'node:wasi';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const [nlPath, options = '', wasmPath] = process.argv.slice(2);
const modulePath =
  wasmPath ?? join(here, '../../../target/wasm32-wasip1/release/pounce_wasm.wasm');

const wasi = new WASI({ version: 'preview1', args: [], env: {} });
const instance = await WebAssembly.instantiate(
  await WebAssembly.compile(readFileSync(modulePath)),
  // The module's per-iteration call, unused here (iterate_callback.mjs).
  { ...wasi.getImportObject(), pounce: { iterate: () => 0 } },
);
wasi.initialize(instance);
const wasm = instance.exports;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function intoWasm(str) {
  if (!str) return [0, 0];
  const bytes = encoder.encode(str);
  const ptr = wasm.pounce_alloc(bytes.length);
  if (!ptr) throw new Error('wasm allocation failed');
  new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}

function fromWasm(ptr) {
  if (!ptr) return null;
  const len = new DataView(wasm.memory.buffer).getUint32(ptr, true);
  const text = decoder.decode(new Uint8Array(wasm.memory.buffer, ptr + 4, len));
  wasm.pounce_free_payload(ptr);
  return text;
}

const nl = readFileSync(nlPath, 'utf8');
const loaded = JSON.parse(fromWasm(wasm.pounce_load(...intoWasm(nl), 0, 0, 0, 0)));
if (loaded.error) {
  process.stdout.write(JSON.stringify({ result: loaded }));
  process.exit(0);
}
const result = JSON.parse(fromWasm(wasm.pounce_solve(...intoWasm(options))));
const sol = fromWasm(wasm.pounce_solution_sol());
process.stdout.write(JSON.stringify({ result, sol, summary: loaded }));
