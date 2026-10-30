// The studio build's live debugger, driven end to end in Node through the
// same host the browser uses (studio/debug-host.js):
//
//   crates/pounce-wasm/build.sh --studio --fast
//   node crates/pounce-wasm/tests/debug_protocol.mjs
//
// What this pins that nothing on the Rust side can: the Asyncify round trip
// (a pause suspends the solve inside `fd_read` and a command resumes it),
// the protocol the front end reads (hello → pause → result … → terminated),
// the inline artifacts a page renders (`viz kkt` / `viz L` / `save`, which on
// a native build would write files and launch a viewer), and that the module
// stays usable for a plain solve after a debug session.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';
import { createStudio } from '../studio/debug-host.js';

const here = dirname(fileURLToPath(import.meta.url));
// The asyncified module; the plain one is expected beside it.
const debugWasm = process.argv[2] ?? join(here, '../studio/pounce-studio-debug.wasm');
const nl = readFileSync(join(here, 'simple.nl'), 'utf8');

// Every check goes through `check`, and the run fails if too few executed:
// a protocol that silently stops emitting must not read as a pass.
let checks = 0;
function check(fn) {
  fn();
  checks++;
}

const events = [];
let onEvent = () => {};
const studio = await createStudio(readFileSync(debugWasm), {
  onEvent: (ev) => {
    events.push(ev);
    onEvent(ev);
  },
});

const loaded = studio.load(nl);
check(() => assert.equal(loaded.n_vars, 2));
check(() => assert.equal(loaded.n_cons, 2));

// --- options catalog ------------------------------------------------------
const catalog = studio.optionsCatalog();
const byName = new Map(catalog.map((o) => [o.name, o]));
check(() => assert.ok(catalog.length > 100, `catalog has ${catalog.length} options`));
check(() => assert.equal(byName.get('max_iter').type, 'integer'));
check(() => assert.equal(byName.get('max_iter').default, 3000));
check(() => assert.equal(byName.get('tol').type, 'number'));
check(() => assert.ok(byName.get('tol').lower === 0 && byName.get('tol').lower_strict));
check(() =>
  assert.ok(byName.get('mu_strategy').valid.some((v) => v.value === 'adaptive' && v.description)),
);
check(() => assert.ok(catalog.every((o) => typeof o.short === 'string' && o.category !== undefined)));

// --- a scripted session ----------------------------------------------------
// Each pause pops the next step; each step is [command, assertion on its
// result]. `null` assertion: the command resumes the solve (no result check
// beyond `ok`).
const results = new Map();
let nextId = 1;
function send(cmd, args = []) {
  const id = nextId++;
  studio.send({ cmd, args, id });
  return id;
}

const pauses = [];
const plan = [
  // iter 0: nothing factored yet
  (p) => {
    check(() => assert.equal(p.iter, 0));
    check(() => assert.equal(p.final, false));
    send('step');
  },
  // iter 1: inspect everything, then run to the end
  (p) => {
    check(() => assert.equal(p.iter, 1));
    const ids = {
      kkt: send('viz', ['kkt']),
      L: send('viz', ['L']),
      x: send('viz', ['x']),
      save: send('save'),
      ask: send('ask', ['--prompt-only', 'why', 'is', 'mu', 'flat?']),
      opt: send('opt', ['max_iter']),
    };
    pending = ids;
    send('continue');
  },
  // terminal checkpoint
  (p) => {
    check(() => assert.equal(p.final, true));
    send('continue');
  },
];
let pending = null;

onEvent = (ev) => {
  if (ev.event === 'pause') {
    pauses.push(ev);
    const step = plan.shift();
    assert.ok(step, `unexpected pause at iter ${ev.iter}`);
    step(ev);
  } else if (ev.event === 'result' && ev.request_id != null) {
    results.set(ev.request_id, ev);
  }
};

const result = await studio.debugSolve('');
check(() => assert.equal(plan.length, 0, 'every scripted pause was reached'));

const hello = events[0];
check(() => assert.equal(hello.event, 'hello'));
check(() => assert.equal(hello.capabilities.inline_artifacts, true));
check(() => assert.equal(hello.capabilities.pause_command, false));
check(() => assert.equal(hello.capabilities.interruptible, false));
check(() => assert.ok(hello.commands.includes('artifacts')));

const kkt = results.get(pending.kkt);
check(() => assert.equal(kkt.ok, true));
check(() => {
  // simple.nl: 2 variables, 1 inequality slack, 2 constraints → a 5×5
  // quasidefinite system with exactly m = 2 negative eigenvalues.
  const d = kkt.data;
  assert.equal(d.dim, 5);
  assert.equal(d.n_pos + d.n_neg, d.dim);
  assert.equal(d.expected_neg, 2);
  assert.equal(d.n_neg, d.expected_neg);
  assert.equal(d.inertia_correct, true);
  const m = d.matrix;
  assert.equal(m.format, 'triplet_1based_lower');
  assert.equal(m.irn.length, m.vals.length);
  assert.equal(m.jcn.length, m.vals.length);
  for (let k = 0; k < m.irn.length; k++) {
    assert.ok(m.irn[k] >= 1 && m.irn[k] <= m.dim && m.jcn[k] >= 1 && m.jcn[k] <= m.irn[k]);
  }
});
check(() => {
  const d = results.get(pending.L).data;
  assert.equal(d.format, 'strict_lower_1based_permuted');
  assert.equal(d.n, 5);
  // a permutation of 0..n-1 or 1..n
  const base = Math.min(...d.perm);
  const sorted = [...d.perm].sort((a, b) => a - b);
  assert.deepEqual(sorted, Array.from({ length: d.n }, (_, i) => i + base));
  for (let k = 0; k < d.l_irn.length; k++) assert.ok(d.l_irn[k] > d.l_jcn[k]);
});
check(() => assert.equal(results.get(pending.x).data.values.length, 2));
check(() => {
  const d = results.get(pending.save).data;
  assert.equal(d.iterate.x.length, 2);
  assert.equal(d.delta.dx.length, 2);
});
check(() => {
  const d = results.get(pending.ask).data;
  assert.equal(d.question, 'why is mu flat?');
  assert.ok(d.prompt.includes('why is mu flat?'));
});
check(() => assert.equal(results.get(pending.opt).ok, true));

const terminated = events.at(-1);
check(() => assert.equal(terminated.event, 'terminated'));
check(() => assert.equal(terminated.status, 'SolveSucceeded'));
check(() => assert.ok(Math.abs(terminated.objective - -Math.SQRT1_2) < 1e-6));
check(() => assert.equal(result.status, 'SolveSucceeded'));
check(() => assert.equal(result.objective, terminated.objective));
check(() => assert.ok(events.some((e) => e.event === 'progress'), 'progress while running'));

// --- the report, its analysis, and point evaluation -----------------------
const report = studio.report();
check(() => assert.equal(report.schema, 'pounce.solve-report/v1'));
check(() => assert.equal(report.iterations.length, result.iterations + 1));
const analysis = studio.analyze();
check(() => assert.equal(analysis.summary.status, 'SolveSucceeded'));
check(() => assert.ok(analysis.findings.some((f) => f.code === 'converged' && f.glossary)));
check(() => assert.equal(analysis.trace.iter.length, result.iterations + 1));
const glossary = studio.glossary();
check(() => assert.ok(glossary.columns.inf_pr.definition));
const at = studio.evaluate([0.6, 0.8]);
check(() => assert.ok(Math.abs(at.f - 0.6) < 1e-12 && Math.abs(at.g[0] - 1) < 1e-12));
const grid = studio.slice({ i: 0, j: 1, range_i: [-1, 1], range_j: [-1, 1], nx: 21, ny: 21 });
check(() => assert.equal(grid.f.length, 21 * 21));
check(() => assert.ok(Math.abs(grid.g['0'][0] - 2) < 1e-12)); // (-1, -1): x0² + x1² = 2

// --- closing stdin at a pause aborts, and the instance stays usable ---------
events.length = 0;
onEvent = (ev) => {
  if (ev.event === 'pause') studio.close();
};
const aborted = await studio.debugSolve('');
check(() => assert.notEqual(aborted.status, 'SolveSucceeded'));
check(() => assert.equal(events.at(-1).event, 'terminated'));

const plain = studio.solve('');
check(() => assert.equal(plain.status, 'SolveSucceeded'));
check(() => assert.ok(studio.solutionSol().includes('objno')));
check(() => assert.ok(studio.solutionCsv().split('\n').length >= 5));
check(() => assert.ok(Math.abs(plain.objective - -Math.SQRT1_2) < 1e-6));

// --- the plain studio module: everything but the debugger ------------------
const plainPath = join(dirname(debugWasm), 'pounce-studio.wasm');
const fast = await createStudio(readFileSync(plainPath), { onEvent: () => {} });
check(() => assert.equal(fast.canDebug, false));
fast.load(nl);
check(() => assert.equal(fast.solve('').status, 'SolveSucceeded'));
check(() => assert.equal(fast.analyze().summary.status, 'SolveSucceeded'));
await assert.rejects(fast.debugSolve(''), /no live debugger/);
checks++;

if (checks < 40) {
  console.error(`only ${checks} checks ran`);
  process.exit(1);
}
console.log(`debug protocol: ${checks} checks passed (${pauses.length} pauses)`);
