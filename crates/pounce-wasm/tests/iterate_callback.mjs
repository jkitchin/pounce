// The per-iteration host call (`pounce.iterate`, docs/src/wasm.md), driven
// through the browser's own shim (web/wasi.js) rather than Node's WASI, so
// the copy of x and the stop convention the pages rely on are what is
// tested.
//
//   crates/pounce-wasm/build.sh
//   node crates/pounce-wasm/tests/iterate_callback.mjs

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';
import { createWasi } from '../web/wasi.js';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath =
  process.argv[2] ?? join(here, '../../../target/wasm32-wasip1/release/pounce_wasm.wasm');

// Each solve below installs its own recorder here.
let onIterate = () => {};
const wasi = createWasi(() => {}, { onIterate: (it) => onIterate(it) });
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), wasi.imports);
wasi.bind(instance);
const wasm = instance.exports;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function intoWasm(str) {
  if (!str) return [0, 0];
  const bytes = encoder.encode(str);
  const ptr = wasm.pounce_alloc(bytes.length);
  assert.notEqual(ptr, 0, 'pounce_alloc returned null');
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

/**
 * Solve with `options`, recording every per-iteration call. The host asks
 * to stop at iteration `stopAt`. Returns the result without its wall time,
 * which no two solves share, the .sol text, and the calls.
 */
function solve(options, stopAt = null) {
  const calls = [];
  onIterate = (it) => {
    calls.push(it);
    return it.iter === stopAt;
  };
  const [ptr, len] = intoWasm(options);
  try {
    const result = JSON.parse(fromWasm(wasm.pounce_solve(ptr, len)));
    assert.equal(result.error, undefined, `solve failed: ${result.error}`);
    delete result.wall_time_secs;
    return { result, sol: fromWasm(wasm.pounce_solution_sol()), calls };
  } finally {
    if (ptr) wasm.pounce_dealloc(ptr, len);
    onIterate = () => {};
  }
}

/** The primal block of a .sol: the n values after the m duals. */
function solPrimal(sol) {
  const lines = sol.split('\n');
  let i = lines.indexOf('Options') + 1;
  i += 1 + Number(lines[i]);
  const m = Number(lines[i]);
  const n = Number(lines[i + 2]);
  i += 4 + m;
  return lines.slice(i, i + n).map(Number);
}

const ON = 'print_level 0\nwasm_iterate_callback yes\n';

// min x0  s.t.  x0^2 + x1^2 == 1,  x0 + x1 >= 0
const summary = JSON.parse(
  fromWasm(wasm.pounce_load(...intoWasm(readFileSync(join(here, 'simple.nl'), 'utf8')), 0, 0, 0, 0)),
);
assert.equal(summary.error, undefined, `load failed: ${summary.error}`);

// Off, by default and explicitly: no calls, and the same solve.
const plain = solve('print_level 0\n');
assert.equal(plain.result.success, true, `unexpected status ${plain.result.status}`);
assert.equal(plain.calls.length, 0, 'no option, no calls');
const off = solve('print_level 0\nwasm_iterate_callback no\n');
assert.equal(off.calls.length, 0, 'option off, no calls');
assert.deepEqual(off.result, plain.result);
assert.equal(off.sol, plain.sol);

// On: one call per iteration from 0, ending at the .sol point, and the solve
// itself unchanged.
const on = solve(ON);
const n = on.result.iterations;
assert.deepEqual(
  on.calls.map((c) => c.iter),
  Array.from({ length: n + 1 }, (_, k) => k),
);
assert.ok(on.calls.every((c) => !c.restoration && c.x.length === 2));
const last = on.calls.at(-1);
assert.deepEqual(Array.from(last.x), solPrimal(on.sol), 'the last x is the .sol primal block');
assert.ok(Math.abs(last.objective - on.result.objective) < 1e-12, `${last.objective}`);
assert.deepEqual(on.result, plain.result);
assert.equal(on.sol, plain.sol);

// Every k: iterations 0, k, 2k, ...
const every = solve(`${ON}wasm_iterate_callback_every 3\n`);
assert.deepEqual(
  every.calls.map((c) => c.iter),
  on.calls.map((c) => c.iter).filter((k) => k % 3 === 0),
);

// Stop: returning true from the host ends the solve at that iteration.
const stopped = solve(ON, 2);
assert.equal(stopped.result.status, 'UserRequestedStop');
assert.equal(stopped.result.iterations, 2);
assert.equal(stopped.calls.at(-1).iter, 2);

console.log(
  `ok: ${on.calls.length} calls over ${n} iterations, ` +
    `${every.calls.length} with every 3, stop honored at iteration 2`,
);
