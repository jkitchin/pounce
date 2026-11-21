// pyomo-pounce under Pyodide, as the Python page's `import pyomo_pounce`
// route runs it: the page's own pounce-solver wheel, Pyomo from PyPI, and the
// pyomo-pounce wheel build.sh stages beside it.
//
//   crates/pounce-wasm/build.sh
//   npm install --prefix <dir> pyodide@0.28.3
//   POUNCE_PYODIDE_DIR=<dir> node crates/pounce-wasm/tests/pyomo_pounce_pyodide.mjs
//
// Pyodide can start neither a thread nor a process. This is the test that
// pounce-py's `.nl` reader and pyomo-pounce's solve and sensitivity calls
// need neither.

import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const pyodideDir = process.env.POUNCE_PYODIDE_DIR;
if (!pyodideDir) throw new Error('set POUNCE_PYODIDE_DIR to a directory holding pyodide@0.28.3');
const { loadPyodide } = await import(
  pathToFileURL(join(pyodideDir, 'node_modules/pyodide/pyodide.mjs')).href
);

// Pinned, as the page pins it: keep in step with PYOMO_VERSION in
// web-python/worker.js.
const PYOMO_VERSION = '6.11.0';

const wheels = join(here, '../web-python/wheels');
const manifest = JSON.parse(readFileSync(join(wheels, 'pounce-wheel.json'), 'utf8'));
// build.sh names the pyomo-pounce wheel here: its version is pyomo-pounce's
// own, not the pounce-solver wheel's.
const pyomoPounce = JSON.parse(readFileSync(join(wheels, 'pyomo-pounce-wheel.json'), 'utf8'));

const py = await loadPyodide();
py.mountNodeFS('/wheels', wheels);
// numpy only, as the page installs this route: neither scipy nor pandas.
await py.loadPackage(['micropip', 'numpy']);
await py.runPythonAsync(`
import micropip
await micropip.install('emfs:/wheels/${manifest.wheel}', deps=False)
await micropip.install('pyomo==${PYOMO_VERSION}')
await micropip.install('emfs:/wheels/${pyomoPounce.wheel}', deps=False)
`);

const out = await py.runPythonAsync(`
import json
import pyomo.environ as pyo
from pyomo_pounce import declare_sens_param, sens_jacobian, sens_solution

# Two plants meet a demand D at least cost. Equal marginal costs give
# x1 = (1 + 0.1 D) / 0.3, so dx1/dD = 1/3 and dx2/dD = 2/3.
def plants():
    m = pyo.ConcreteModel()
    m.demand = pyo.Param(initialize=10.0, mutable=True)
    m.d = pyo.Var(initialize=10.0)
    m.set_d = pyo.Constraint(expr=m.d == m.demand)
    m.x = pyo.Var([1, 2], bounds=(0, None), initialize=5.0)
    m.meet = pyo.Constraint(expr=m.x[1] + m.x[2] == m.d)
    m.cost = pyo.Objective(
        expr=2 * m.x[1] + 0.1 * m.x[1] ** 2 + 3 * m.x[2] + 0.05 * m.x[2] ** 2
    )
    return m

declared = plants()
declare_sens_param(declared.demand)
pyo.SolverFactory("pounce").solve(declared, tee=True)
estimate = sens_solution(declared, [(declared.demand, 12.0)])

plain = plants()
plain_status = pyo.SolverFactory("pounce").solve(plain).solver.termination_condition

# The per-iteration callback: one call per iteration, the last at the
# point the solve loads.
watched = plants()
calls = []
pyo.SolverFactory("pounce").solve(watched, intermediate=calls.append)

import sys
assert "scipy" not in sys.modules and "pandas" not in sys.modules

json.dumps({
    "x1": pyo.value(declared.x[1]),
    "dx1": float(sens_jacobian(declared.x[1], wrt=declared.demand)),
    "dx2": float(sens_jacobian(declared.x[2], wrt=declared.demand)),
    "x1_at_12": float(estimate[declared.x[1]]),
    "plain_status": str(plain_status),
    "plain_x1": pyo.value(plain.x[1]),
    "calls": [it.iteration for it in calls],
    "last_x1": calls[-1].values[watched.x[1]],
    "watched_x1": pyo.value(watched.x[1]),
    "last_objective": calls[-1].objective,
    "watched_objective": pyo.value(watched.cost),
})
`);

const r = JSON.parse(out);
const close = (a, b, what) => {
  if (!(Math.abs(a - b) < 1e-6)) throw new Error(`${what}: ${a}, expected ${b}`);
};
close(r.x1, 2 / 0.3 * 1.0, 'x1 at D = 10');
close(r.dx1, 1 / 3, 'dx1/dD');
close(r.dx2, 2 / 3, 'dx2/dD');
close(r.x1_at_12, 2.2 / 0.3, 'x1 estimated at D = 12');
if (r.plain_status !== 'optimal') throw new Error(`undeclared solve ended ${r.plain_status}`);
close(r.plain_x1, 2 / 0.3, 'x1 from the undeclared solve');
if (r.calls.join() !== r.calls.map((_, k) => k).join()) {
  throw new Error(`intermediate called at iterations ${r.calls}`);
}
if (r.last_x1 !== r.watched_x1) throw new Error(`last call x1 ${r.last_x1} != ${r.watched_x1}`);
close(r.last_objective, r.watched_objective, 'objective at the last call');
console.log(
  `ok: pyomo-pounce under Pyodide, dx1/dD ${r.dx1.toFixed(6)}, dx2/dD ${r.dx2.toFixed(6)}, ` +
    `${r.calls.length} intermediate calls`,
);
