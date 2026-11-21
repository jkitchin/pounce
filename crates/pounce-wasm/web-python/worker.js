// Worker for the Python app. Two independent ways to reach POUNCE from
// Python, and a script picks one by what it imports:
//
//   `import pounce`          — the real `pounce-solver` package, built for
//                              emscripten and installed by micropip. The
//                              extension module runs inside Pyodide's own
//                              wasm instance, so numpy arrays, callbacks and
//                              `pounce.sensitivity` all work as they do on a
//                              desktop.
//   `import pounce_browser`  — Pyomo writes an AMPL `.nl`, the standalone
//                              `pounce.wasm` (a separate wasm instance, its
//                              own memory) solves it, and the `.sol` is loaded
//                              back onto the Pyomo model. Nothing crosses but
//                              text.
//
// A third on-demand install, matplotlib, is orthogonal to both: whichever
// route a script takes, `plt.show()` sends a PNG to the page.
//
// None of the three is installed up front: they are large, and a script needs
// at most some of them. `ready` gets CPython running; `ensure()` installs the
// rest on the first run that asks for it, and caches the promise.


// Pinned so a Pyodide release cannot change what this page runs. Override
// with ?pyodide=<base-url> to serve Pyodide yourself (offline, or behind a
// network that does not allow the CDN).
const PYODIDE_VERSION = '0.28.3';
const params = new URLSearchParams(self.location.search);
const PYODIDE_URL = params.get('pyodide') || `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// The emscripten build of `pounce-solver` is named by a manifest that
// `crates/pounce-wasm/build-wheel.sh` writes beside it, rather than by a
// constant here: the wheel's ABI tag carries the exact emscripten version
// Pyodide was built with, so the file name changes whenever either version
// moves and a stale constant would fail as an unhelpful micropip resolution
// error. `?pounce=<url>` points at a wheel you host yourself.
async function pounceWheelUrl() {
  const override = params.get('pounce');
  if (override) return override;
  const res = await fetch('./wheels/pounce-wheel.json');
  if (!res.ok) {
    throw new Error(
      'no pounce-solver wheel is deployed here — run crates/pounce-wasm/build-wheel.sh, ' +
        'or pass ?pounce=<wheel url>. The Pyomo examples work without it.',
    );
  }
  const manifest = await res.json();
  // An emscripten wheel is valid for exactly the Pyodide build it was
  // compiled against. Say that plainly instead of letting micropip report it
  // as "no matching distribution", which reads like a missing package.
  if (manifest.pyodide_version !== PYODIDE_VERSION) {
    throw new Error(
      `the deployed wheel was built for Pyodide ${manifest.pyodide_version}, but this ` +
        `page runs ${PYODIDE_VERSION} — rebuild it with crates/pounce-wasm/build-wheel.sh`,
    );
  }
  return `./wheels/${manifest.wheel}`;
}

const say = (text) => self.postMessage({ type: 'status', text });
const out = (text) => self.postMessage({ type: 'stdout', text });

// --- the POUNCE module -----------------------------------------------------

let solver = null;
let solverLog = '';
const encoder = new TextEncoder();
const decoder = new TextDecoder();

// `wasi.js` is staged beside this file by `crates/pounce-wasm/build.sh`, next
// to the `pounce.wasm` it shims — both are build artifacts, neither is
// tracked. A static `import` of it would fail the whole worker module in a
// checkout that has not been built, and a worker that fails to load reports
// nothing: the page sits on "starting…" forever. Only the Pyomo route needs
// it, so it is imported where that route loads the module, and a missing
// build costs that one route an error message instead of costing the page
// everything.
let wasi = null;

async function loadSolver() {
  let createWasi;
  try {
    ({ createWasi } = await import('./wasi.js'));
  } catch {
    // Both this and `pounce.wasm` come from the same build step, so a missing
    // shim means a missing module: report the cause rather than the symptom.
    throw new Error(
      'the standalone POUNCE wasm module is not deployed here — run ' +
        'crates/pounce-wasm/build.sh. The `import pounce` examples work without it.',
    );
  }
  wasi = createWasi((text) => {
    solverLog += text;
    out(text);
  });
  let instance;
  try {
    ({ instance } = await WebAssembly.instantiateStreaming(fetch('./pounce.wasm'), wasi.imports));
  } catch {
    const bytes = await (await fetch('./pounce.wasm')).arrayBuffer();
    ({ instance } = await WebAssembly.instantiate(bytes, wasi.imports));
  }
  wasi.bind(instance);
  solver = instance.exports;
}

function intoWasm(str) {
  if (!str) return [0, 0];
  const bytes = encoder.encode(str);
  const ptr = solver.pounce_alloc(bytes.length);
  if (!ptr) throw new Error('wasm allocation failed');
  new Uint8Array(solver.memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}

function fromWasm(ptr) {
  if (!ptr) return null;
  const len = new DataView(solver.memory.buffer).getUint32(ptr, true);
  const text = decoder.decode(new Uint8Array(solver.memory.buffer, ptr + 4, len));
  solver.pounce_free_payload(ptr);
  return text;
}

/**
 * The backend `pounce_browser.solve()` calls. Synchronous, because the wasm
 * solve is: Python blocks here until POUNCE returns, exactly as it would
 * waiting on a subprocess.
 *
 * Returns a JSON string — the Python side accepts one, which keeps the
 * boundary free of Pyodide proxy objects and their lifetimes.
 */
function solveNl(nlText, options) {
  solverLog = '';
  const nlArgs = intoWasm(nlText);
  let loaded;
  try {
    loaded = JSON.parse(fromWasm(solver.pounce_load(...nlArgs, 0, 0, 0, 0)));
  } finally {
    if (nlArgs[0]) solver.pounce_dealloc(...nlArgs);
  }
  if (loaded.error) return JSON.stringify({ result: loaded, log: solverLog });

  const optArgs = intoWasm(options || '');
  let result;
  try {
    result = JSON.parse(fromWasm(solver.pounce_solve(...optArgs)));
  } finally {
    if (optArgs[0]) solver.pounce_dealloc(...optArgs);
  }
  const sol = fromWasm(solver.pounce_solution_sol());
  return JSON.stringify({ result, sol, summary: loaded, log: solverLog });
}

// --- Pyodide ---------------------------------------------------------------

let pyodide = null;
let micropip = null;

const ready = (async () => {
  say(`loading Pyodide ${PYODIDE_VERSION} (~10 MB, cached after the first run)…`);
  // `pyodide.mjs`, not `pyodide.js`: this is a module worker, where
  // `importScripts` does not exist.
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
  pyodide = await loadPyodide({
    indexURL: PYODIDE_URL,
    stdout: (line) => out(line + '\n'),
    stderr: (line) => out(line + '\n'),
  });
  await pyodide.loadPackage('micropip');
  micropip = pyodide.pyimport('micropip');
  // Pyodide's interactive matplotlib backends draw into the DOM, and a worker
  // has no DOM — they fail at import, before a script gets to say anything
  // about what it wants. Agg renders to a buffer instead, which is what
  // `installMatplotlib` below turns into a PNG. Set here rather than at
  // install time because matplotlib reads it once, when it is first imported,
  // and a script's own `import matplotlib` may well come first.
  await pyodide.runPythonAsync('import os\nos.environ["MPLBACKEND"] = "AGG"\n');
  say('ready');
  self.postMessage({ type: 'ready' });
})().catch((err) => {
  self.postMessage({ type: 'fatal', message: String(err && err.message ? err.message : err) });
});

// --- the two routes, installed on demand -----------------------------------

// What a script imports is what it gets. `\b` would match inside
// `pounce_browser`, since `_` is a word character — hence the explicit
// negative lookahead, which is the difference between the two routes here.
const WANTS_POUNCE = /^[ \t]*(?:import|from)[ \t]+pounce(?![\w])/m;
const WANTS_PYOMO = /^[ \t]*(?:import|from)[ \t]+(?:pyomo|pounce_browser)(?![\w])/m;
// pyomo-pounce solves in-process through the `pounce` package, so a script
// that imports it needs that wheel and not the standalone module. Its own
// `import pyomo.environ` matches WANTS_PYOMO, so the standalone module is
// loaded for it only when it names `pounce_browser` too.
const WANTS_PYOMO_POUNCE = /^[ \t]*(?:import|from)[ \t]+pyomo_pounce(?![\w])/m;
const WANTS_POUNCE_BROWSER = /^[ \t]*(?:import|from)[ \t]+pounce_browser(?![\w])/m;
// `pylab` and `mpl_toolkits` are matplotlib too, and a script that reaches for
// either without naming matplotlib is otherwise indistinguishable from one
// that does not plot at all.
const WANTS_MPL = /^[ \t]*(?:import|from)[ \t]+(?:matplotlib|pylab|mpl_toolkits)(?![\w])/m;

const installs = {};
// A failed install must not be cached as done: drop the rejected promise so
// the next Run retries rather than replaying the same network error forever.
const ensure = (name, install) =>
  (installs[name] ??= install().catch((err) => {
    delete installs[name];
    throw err;
  }));

async function installPounceSolver() {
  say('installing pounce-solver (numpy, scipy, then the wheel)…');
  // numpy and scipy come from Pyodide's own build, not PyPI — micropip would
  // otherwise try to resolve source distributions it cannot compile.
  const wheel = await pounceWheelUrl();
  await pyodide.loadPackage(['numpy', 'scipy']);
  await micropip.install(wheel);
  say('ready');
}

// Figures leave Python as base64 PNGs on the message channel. That is the
// whole transport: no canvas, no proxy object whose lifetime the page has to
// manage, and the main thread does nothing but set an <img> src.
async function installMatplotlib() {
  say('installing matplotlib (~9 MB)…');
  await pyodide.loadPackage('matplotlib');
  self.pounceEmitFigure = (png) => self.postMessage({ type: 'figure', png });
  await pyodide.runPythonAsync(`
import base64, io, js
import matplotlib.pyplot as plt


# What plt.show() becomes here: emit every open figure as a PNG, then close
# them. Agg's own show() is a no-op that warns, so a script written for a
# desktop would otherwise print a warning and produce nothing.
def _pounce_flush_figures():
    for num in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
        js.pounceEmitFigure(base64.b64encode(buf.getvalue()).decode())
    plt.close("all")


plt.show = lambda *args, **kwargs: _pounce_flush_figures()
`);
  say('ready');
}

// pyomo-pounce's solves and sensitivity calls use numpy and none of
// pounce-solver's scipy paths (`minimize`, `curve_fit`, the QP helpers), and
// pandas only for `Jacobian.to_dataframe()`. So this route installs both
// wheels without their dependencies and skips the ~20 MB of scipy, OpenBLAS
// and pandas. A script that also imports `pounce` for those paths gets scipy
// from `installPounceSolver`.
async function installPyomoPounce() {
  say('installing pounce-solver, Pyomo and pyomo-pounce…');
  await pyodide.loadPackage('numpy');
  await micropip.install.callKwargs(await pounceWheelUrl(), { deps: false });
  const wheels = params.get('pyomo');
  await micropip.install(wheels ? wheels.split(',') : 'pyomo');
  // `build.sh` builds pyomo-pounce beside the pounce-solver wheel, at the
  // same version.
  const manifest = await (await fetch('./wheels/pounce-wheel.json')).json();
  const wheel = new URL(
    `./wheels/pyomo_pounce-${manifest.version}-py3-none-any.whl`,
    self.location.href,
  ).href;
  if (!(await fetch(wheel, { method: 'HEAD' })).ok) {
    throw new Error('no pyomo-pounce wheel is deployed here. Run crates/pounce-wasm/build.sh.');
  }
  await micropip.install.callKwargs(wheel, { deps: false });
  say('ready');
}

async function installPyomoRoute() {
  say('loading the POUNCE solver module…');
  await loadSolver();

  say('installing Pyomo…');
  // Pyomo publishes a `py3-none-any` wheel, so micropip installs it as-is —
  // its compiled extensions are optional and unused here, and nothing has to
  // be built for wasm. `?pyomo=<url>[,<url>…]` points at self-hosted wheels
  // (offline, or where PyPI is unreachable); the default resolves from PyPI.
  const wheels = params.get('pyomo');
  await micropip.install(wheels ? wheels.split(',') : 'pyomo');

  say('wiring POUNCE into Python…');
  const shim = await (await fetch('./pounce_browser.py')).text();
  pyodide.FS.writeFile('/home/pyodide/pounce_browser.py', shim);
  self.pounceSolveNl = solveNl;
  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/home/pyodide")
import js, pounce_browser
pounce_browser.set_backend(lambda nl, options: js.pounceSolveNl(nl, options))
`);
  say('ready');
}

self.onmessage = async (event) => {
  if (event.data.type !== 'run') return;
  try {
    await ready;
    if (!pyodide) return;
    const code = event.data.code;
    // Installing before the timer starts: the download is not the model's
    // solve time, and reporting it as such would be a lie about the solver.
    const pyomoPounce = WANTS_PYOMO_POUNCE.test(code);
    if (WANTS_POUNCE.test(code)) await ensure('pounce', installPounceSolver);
    if (pyomoPounce) await ensure('pyomo-pounce', installPyomoPounce);
    if (WANTS_POUNCE_BROWSER.test(code) || (WANTS_PYOMO.test(code) && !pyomoPounce)) {
      await ensure('pyomo', installPyomoRoute);
    }
    const plots = WANTS_MPL.test(code);
    if (plots) await ensure('matplotlib', installMatplotlib);
    self.postMessage({ type: 'running' });
    const started = performance.now();
    try {
      await pyodide.runPythonAsync(code);
    } finally {
      // A script that builds a figure and never calls `plt.show()` is correct
      // in a notebook and silent here, so the end of the run flushes whatever
      // is still open. It runs on the exception path too: a script that
      // plotted its data and then failed should still show the plot, which is
      // usually the thing that explains the failure.
      if (plots) await pyodide.runPythonAsync('_pounce_flush_figures()');
    }
    self.postMessage({ type: 'done', ms: performance.now() - started });
  } catch (err) {
    // A Python exception arrives with its traceback in `message`; show it as
    // the script's own output rather than as a page-level failure.
    self.postMessage({ type: 'error', message: String(err && err.message ? err.message : err) });
  }
};
