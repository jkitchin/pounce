// Page logic: pick an example, run it in the Pyodide worker, show what it
// printed and whatever it plotted. The worker owns both wasm instances; this
// side is a console and an <img> holder.
//
// Examples come in two groups because there are two ways to reach POUNCE from
// Python here — `import pounce`, the real package built for emscripten, and
// `import pounce_browser`, Pyomo writing an `.nl` for the standalone wasm
// module. The worker installs whichever one a script imports, so the groups
// are a labelling convention, not a mode switch.

import { attachEditor } from './editor.js';

const $ = (id) => document.getElementById(id);
const codeBox = $('code');
const outBox = $('out');
const runButton = $('run');
const cancelButton = $('cancel');
const statusLine = $('status');
const figuresPanel = $('figures-panel');
const figuresBox = $('figures');

const GROUPS = {
  pounce: 'pounce-solver (the package)',
  pyomo: 'Pyomo + .nl',
  'pyomo-pounce': 'Pyomo + pyomo-pounce (sensitivity)',
};

const EXAMPLES = {
  'Constrained Rosenbrock': { group: 'pounce', code: `# scipy-style: pounce.minimize, on the Rosenbrock function
# restricted to the unit disk. The constraint is active at the optimum.
import numpy as np
import pounce

def f(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

# scipy's dict form: 'ineq' means fun(x) >= 0, so this is |x| <= 1.
disk = {"type": "ineq", "fun": lambda x: 1.0 - x @ x}

res = pounce.minimize(f, np.array([-1.2, 1.0]), constraints=[disk], print_level=5)

print(res.message)
print("x* =", res.x)
print("f* =", res.fun)
print("|x*| =", np.linalg.norm(res.x), "(the disk boundary)")
` },

  'HS071 (pounce.minimize)': { group: 'pounce', code: `# The same model as the Pyomo HS071 example, written against the
# pounce-solver package instead — no Pyomo, no .nl file in between.
import numpy as np
import pounce

def f(x):
    return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

cons = [
    {"type": "ineq", "fun": lambda x: x[0] * x[1] * x[2] * x[3] - 25.0},
    {"type": "eq", "fun": lambda x: x @ x - 40.0},
]

res = pounce.minimize(
    f,
    np.array([1.0, 5.0, 5.0, 1.0]),
    bounds=[(1, 5)] * 4,
    constraints=cons,
    tol=1e-8,
    print_level=5,
)

print(res.message)
print("x* =", np.round(res.x, 9))
print("f* =", res.fun)
print("known optimum: 17.0140173")
` },

  'Curve fit + confidence band': { group: 'pounce', code: `# pounce.curve_fit is scipy's curve_fit plus what you actually wanted
# from it: standard errors, confidence intervals, goodness of fit.
import numpy as np
from pounce import curve_fit

A_true, k_true = 2.5, 0.8
t = np.linspace(0.0, 4.0, 30)
rng = np.random.default_rng(0)
y = A_true * np.exp(-k_true * t) + rng.normal(0.0, 0.08, t.size)

def model(t, A, k):
    return A * np.exp(-k * t)

fit = curve_fit(model, t, y, p0=[1.0, 0.1])

for name, true, p, se, (lo, hi) in zip(["A", "k"], [A_true, k_true],
                                       fit.popt, fit.perr, fit.ci):
    print(f"{name} = {p:.6f} +/- {se:.6f}   95% CI [{lo:.6f}, {hi:.6f}]   true {true}")

print(f"R^2 = {fit.r_squared:.6f}   RMSE = {fit.rmse:.3e}   dof = {fit.dof}")

# matplotlib works here too, on the Agg backend: plt.show() sends the figure
# to the page as a PNG. Both bands are delta-method; they answer different
# questions. The confidence band is "where is the true curve?" and is narrow
# — most data points fall outside it, and it shrinks as data is added. The
# prediction band adds the observation noise, so it is where the *next*
# measurement lands, and it stays about as wide however much data you take.
import matplotlib.pyplot as plt

grid = np.linspace(t[0], t[-1], 200)
yhat, c_lo, c_hi = fit.confidence_band(grid)
_, p_lo, p_hi = fit.confidence_band(grid, kind="prediction")

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.fill_between(grid, p_lo, p_hi, color="#f0913f", alpha=0.15, lw=0, label="95% prediction")
ax.fill_between(grid, c_lo, c_hi, color="#f0913f", alpha=0.55, lw=0, label="95% confidence")
ax.plot(grid, yhat, "-", lw=2, color="#c96a12", label="fit")
ax.plot(t, y, "o", ms=4, color="#333", label="data")
ax.set_xlabel("t")
ax.set_ylabel("y")
ax.set_title(f"A = {fit.popt[0]:.3f} +/- {fit.perr[0]:.3f},  "
             f"k = {fit.popt[1]:.3f} +/- {fit.perr[1]:.3f}")
ax.legend()
plt.show()
` },

  'Constrained NLP': { group: 'pyomo', code: `# Minimize x1 on the unit circle, above the line x1 + x2 = 0.
from pyomo.environ import *
import pounce_browser

m = ConcreteModel()
m.x1 = Var(initialize=0.5, bounds=(-10, 10))
m.x2 = Var(initialize=0.5, bounds=(-10, 10))
m.circle = Constraint(expr=m.x1**2 + m.x2**2 == 1)
m.halfspace = Constraint(expr=m.x1 + m.x2 >= 0)
m.obj = Objective(expr=m.x1)
m.dual = Suffix(direction=Suffix.IMPORT)

res = pounce_browser.solve(m, options="print_level 5")

print(res)
print(f"x1 = {value(m.x1):.9f}   (exact: {-2**-0.5:.9f})")
print(f"x2 = {value(m.x2):.9f}")
print(f"dual[circle]    = {m.dual[m.circle]:.9f}")
print(f"dual[halfspace] = {m.dual[m.halfspace]:.9f}")
` },


  'HS071 (Pyomo)': { group: 'pyomo', code: `# Hock-Schittkowski 71 — the classic Ipopt example.
from pyomo.environ import *
import pounce_browser

m = ConcreteModel()
m.I = RangeSet(1, 4)
m.x = Var(m.I, bounds=(1, 5), initialize={1: 1, 2: 5, 3: 5, 4: 1})
m.obj = Objective(expr=m.x[1] * m.x[4] * (m.x[1] + m.x[2] + m.x[3]) + m.x[3])
m.c1 = Constraint(expr=m.x[1] * m.x[2] * m.x[3] * m.x[4] >= 25)
m.c2 = Constraint(expr=sum(m.x[i] ** 2 for i in m.I) == 40)

res = pounce_browser.solve(m, options="print_level 5\\ntol 1e-8")

print(res)
print("x* =", [round(value(m.x[i]), 9) for i in m.I])
print("known optimum: 17.0140173")
` },


  'Parameter estimation': { group: 'pyomo', code: `# Least-squares fit of a kinetic rate law to noisy data.
from pyomo.environ import *
import math, pounce_browser

# Synthetic data: y = A*exp(-k*t), A = 2.5, k = 0.8
A_true, k_true = 2.5, 0.8
ts = [0.1 * i for i in range(40)]
noise = [0.02 * math.sin(37.0 * t) for t in ts]          # deterministic "noise"
ys = [A_true * math.exp(-k_true * t) + n for t, n in zip(ts, noise)]

m = ConcreteModel()
m.A = Var(initialize=1.0, bounds=(0, 10))
m.k = Var(initialize=0.1, bounds=(0, 10))
m.obj = Objective(
    expr=sum((m.A * exp(-m.k * t) - y) ** 2 for t, y in zip(ts, ys))
)

res = pounce_browser.solve(m, options="print_level 0")

print(res)
print(f"A = {value(m.A):.6f}  (true {A_true})")
print(f"k = {value(m.k):.6f}  (true {k_true})")
` },


  'Larger: 2,000 variables': { group: 'pyomo', code: `# A separable NLP big enough to feel the solver work.
from pyomo.environ import *
import pounce_browser

N = 1000
m = ConcreteModel()
m.I = RangeSet(1, N)
m.x = Var(m.I, initialize=0.5, bounds=(0.01, 10))
m.y = Var(m.I, initialize=0.5, bounds=(0.01, 10))
m.link = Constraint(m.I, rule=lambda m, i: m.x[i] * m.y[i] == 1.0)
m.obj = Objective(expr=sum(m.x[i] ** 2 + 0.5 * m.y[i] ** 2 for i in m.I))

res = pounce_browser.solve(m, options="print_level 5")

print(res)
print(f"x[1] = {value(m.x[1]):.6f}, y[1] = {value(m.y[1]):.6f}")
print("each pair solves x^2 + 0.5/x^2 -> x = (0.5)**0.25 =", 0.5 ** 0.25)
` },

  'Sensitivity to a parameter': { group: 'pyomo-pounce', code: `# Declare the parameter, solve once, then read dx*/dp and a perturbed
# solution from the stored KKT factorization instead of re-solving.
import pyomo.environ as pyo
from pyomo_pounce import declare_sens_param, sens_jacobian, sens_solution

# Two plants meet a demand at least cost. Marginal costs rise with output.
m = pyo.ConcreteModel()
m.demand = pyo.Param(initialize=10.0, mutable=True)
m.d = pyo.Var(initialize=10.0)
m.set_d = pyo.Constraint(expr=m.d == m.demand)   # the parameter's one defining equality
m.x = pyo.Var([1, 2], bounds=(0, None), initialize=5.0)
m.meet = pyo.Constraint(expr=m.x[1] + m.x[2] == m.d)
m.cost = pyo.Objective(
    expr=2 * m.x[1] + 0.1 * m.x[1] ** 2 + 3 * m.x[2] + 0.05 * m.x[2] ** 2
)
declare_sens_param(m.demand)

pyo.SolverFactory("pounce").solve(m, tee=True)

print(f"x1 = {pyo.value(m.x[1]):.4f}, x2 = {pyo.value(m.x[2]):.4f}")
print(f"dx1/d demand = {sens_jacobian(m.x[1], wrt=m.demand):.4f}  (exact 1/3)")
print(f"dx2/d demand = {sens_jacobian(m.x[2], wrt=m.demand):.4f}  (exact 2/3)")
est = sens_solution(m, [(m.demand, 12.0)])
print(f"at demand 12: x1 = {est[m.x[1]]:.4f}, x2 = {est[m.x[2]]:.4f}")
` },
};

const select = $('example');
for (const [key, label] of Object.entries(GROUPS)) {
  const group = document.createElement('optgroup');
  group.label = label;
  for (const [name, example] of Object.entries(EXAMPLES)) {
    if (example.group !== key) continue;
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    group.append(option);
  }
  select.append(group);
}
// The editor wraps the textarea rather than replacing it, so `codeBox.value`
// stays the single source of truth and the page still works if this module
// fails to load.
const editor = attachEditor(codeBox);
select.addEventListener('change', () => editor.set(EXAMPLES[select.value].code));
editor.set(EXAMPLES[select.value].code);

// --- worker ----------------------------------------------------------------

// A solve runs synchronously inside the worker, so nothing short of killing the
// thread can interrupt one — Cancel terminates the worker and starts a fresh
// one. That is a real reset: the replacement has to re-download Pyodide (from
// cache) and re-install whichever route the next script imports.
let worker = null;
let ready = false;

function spawnWorker() {
  if (worker) worker.terminate();
  ready = false;
  runButton.disabled = true;
  cancelButton.disabled = true;
  // The worker inherits this page's query string, so ?pyodide= / ?pyomo= /
  // ?pounce= overrides for a self-hosted deployment reach it.
  worker = new Worker(`./worker.js${self.location.search}`, { type: 'module' });
  worker.onmessage = onWorkerMessage;
}

function onWorkerMessage({ data }) {
  if (data.type === 'status') {
    setStatus(data.text);
  } else if (data.type === 'ready') {
    ready = true;
    runButton.disabled = false;
    // Neither solver route is installed yet — the first Run pulls whichever
    // one the script imports, and says so on this line while it does.
    setStatus('ready — Python is loaded');
  } else if (data.type === 'stdout') {
    append(data.text);
  } else if (data.type === 'figure') {
    // `plt.show()` in the worker sends a PNG per open figure. Appending
    // rather than replacing: a script may plot several.
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${data.png}`;
    img.alt = `Figure ${figuresBox.childElementCount + 1}`;
    figuresBox.append(img);
    figuresPanel.hidden = false;
  } else if (data.type === 'running') {
    outBox.textContent = '';
    // Figures belong to the run that produced them; a re-run that plots
    // nothing must not leave the last one on screen looking current.
    figuresBox.replaceChildren();
    figuresPanel.hidden = true;
    cancelButton.disabled = false;
    setStatus('running…');
  } else if (data.type === 'done') {
    runButton.disabled = false;
    cancelButton.disabled = true;
    setStatus(`done in ${(data.ms / 1000).toFixed(2)} s`);
  } else if (data.type === 'error') {
    runButton.disabled = false;
    cancelButton.disabled = true;
    append(`\n${data.message}\n`);
    setStatus('the script raised an exception', true);
  } else if (data.type === 'fatal') {
    runButton.disabled = true;
    cancelButton.disabled = true;
    setStatus(data.message, true);
  }
}

spawnWorker();

function setStatus(text, isError = false) {
  statusLine.textContent = text;
  statusLine.classList.toggle('err', isError);
}

function append(text) {
  outBox.textContent += text;
  outBox.scrollTop = outBox.scrollHeight;
}

function run() {
  if (!ready) return;
  runButton.disabled = true;
  worker.postMessage({ type: 'run', code: codeBox.value });
}

runButton.addEventListener('click', run);
cancelButton.addEventListener('click', () => {
  append('\n^C  cancelled — restarting Python\n');
  setStatus('cancelled — reloading Python…');
  spawnWorker();
});
codeBox.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    run();
  }
});

$('dl-code').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([codeBox.value], { type: 'text/x-python' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'pounce_model.py';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
});

// --- theme -----------------------------------------------------------------

// The inline script in index.html has already resolved "auto" into an explicit
// data-theme so the first paint is right; this re-resolves it on a change to
// the OS setting, which only matters while the preference *is* "auto".
const themeSelect = $('theme');
const systemDark = matchMedia('(prefers-color-scheme: dark)');

function readTheme() {
  try {
    return localStorage.getItem('pounce-theme') || 'auto';
  } catch {
    return 'auto';
  }
}

function applyTheme(pref) {
  const dark = pref === 'dark' || (pref === 'auto' && systemDark.matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
}

themeSelect.value = readTheme();
themeSelect.addEventListener('change', () => {
  try {
    localStorage.setItem('pounce-theme', themeSelect.value);
  } catch {
    // A blocked store is not a reason to refuse the change for this page view.
  }
  applyTheme(themeSelect.value);
});
systemDark.addEventListener('change', () => applyTheme(readTheme()));
