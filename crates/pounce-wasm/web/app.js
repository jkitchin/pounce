// Page logic: take dropped files, ask the worker to load and solve, render.

const $ = (id) => document.getElementById(id);
const drop = $('drop');
const fileInput = $('files');
const loadError = $('loaderr');
const solveButton = $('solve');
const solveStatus = $('solve-status');
const log = $('log');

let worker = null;
let summary = null;
let result = null;
let modelName = 'model';

// --- worker lifecycle ------------------------------------------------------

// Every load gets a brand-new worker, and with it a brand-new wasm instance.
// Reusing one would carry the previous model's parse, the previous solve, and
// a linear memory already grown to fit them into the next file — so dropping
// a second model would be solved by an instance whose state you cannot see.
// Recreating costs one module compile (~100 ms) and makes "drag a file in"
// mean exactly what it looks like: start over.
function restartWorker() {
  if (worker) worker.terminate();
  worker = new Worker('./worker.js', { type: 'module' });
  worker.onmessage = onWorkerMessage;
}

function resetUi() {
  summary = null;
  result = null;
  for (const id of [
    'summary-section',
    'solve-section',
    'plot-section',
    'result-section',
    'log-section',
  ]) {
    $(id).hidden = true;
  }
  log.textContent = '';
  solveStatus.textContent = '';
  solveButton.disabled = false;
  showLoadError(null);
}

// --- file intake -----------------------------------------------------------

['dragenter', 'dragover'].forEach((ev) =>
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add('over');
  }),
);
['dragleave', 'drop'].forEach((ev) =>
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.remove('over');
  }),
);
drop.addEventListener('drop', (e) => takeFiles(e.dataTransfer.files));
fileInput.addEventListener('change', () => takeFiles(fileInput.files));

async function takeFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  const pick = (ext) => files.find((f) => f.name.toLowerCase().endsWith(ext));
  // Anything that is not a .col/.row is treated as the model, so a file
  // named `stub` or `model.nl.txt` still works.
  const nl = pick('.nl') ?? files.find((f) => !/\.(col|row)$/i.test(f.name));

  resetUi();
  restartWorker();

  if (!nl) {
    showLoadError('Drop an .nl file (optionally with its .col / .row name files).');
    return;
  }
  modelName = nl.name.replace(/\.nl$/i, '') || 'model';
  $('filename').textContent = `— ${nl.name} (${formatBytes(nl.size)})`;
  const [nlText, colText, rowText] = await Promise.all([
    nl.text(),
    pick('.col')?.text() ?? Promise.resolve(''),
    pick('.row')?.text() ?? Promise.resolve(''),
  ]);
  worker.postMessage({ type: 'load', nl: nlText, col: colText, row: rowText });
}

function showLoadError(message) {
  loadError.hidden = !message;
  loadError.textContent = message ?? '';
}

// --- worker plumbing -------------------------------------------------------

function onWorkerMessage({ data }) {
  if (data.type === 'summary') {
    if (data.summary.error) {
      showLoadError(data.summary.error);
      return;
    }
    summary = data.summary;
    renderSummary(summary);
  } else if (data.type === 'iterate') {
    addPoint(data);
  } else if (data.type === 'log') {
    log.textContent += data.text;
    log.scrollTop = log.scrollHeight;
  } else if (data.type === 'result') {
    solveButton.disabled = false;
    if (data.result.error) {
      solveStatus.textContent = data.result.error;
      return;
    }
    solveStatus.textContent = '';
    result = data.result;
    renderResult(result);
  } else if (data.type === 'export') {
    if (data.text === null) {
      solveStatus.textContent = 'nothing to export yet — solve first';
      return;
    }
    download(data.filename, data.text, data.mime);
  } else if (data.type === 'fatal') {
    solveButton.disabled = false;
    if (data.request === 'load') showLoadError(data.message);
    else solveStatus.textContent = data.message;
  }
}

solveButton.addEventListener('click', () => {
  solveButton.disabled = true;
  solveStatus.textContent = 'solving…';
  log.textContent = '';
  $('log-section').hidden = false;
  $('result-section').hidden = true;
  result = null;
  resetPlot();
  worker.postMessage({ type: 'solve', options: $('options').value });
});

// --- live objective plot ---------------------------------------------------

// The worker forwards one message per `pounce.iterate` call, which arrive
// while the solve is still running. The section stays hidden until the
// first one, so a solve without `wasm_iterate_callback yes` shows no plot.
// Restoration iterations carry no objective (NaN) and leave a gap.
let points = [];
let frame = 0;

function resetPlot() {
  points = [];
  $('plot-section').hidden = true;
  $('trace').setAttribute('points', '');
  $('plot-note').textContent = '';
}

function addPoint({ iter, objective }) {
  points.push([iter, objective]);
  $('plot-section').hidden = false;
  // Redraw at most once per frame however fast the iterations come.
  if (!frame) {
    frame = requestAnimationFrame(() => {
      frame = 0;
      drawPlot();
    });
  }
}

function drawPlot() {
  const finitePoints = points.filter(([, f]) => Number.isFinite(f));
  if (!finitePoints.length) return;
  const lastIter = points.at(-1)[0];
  const values = finitePoints.map(([, f]) => f);
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (hi === lo) {
    lo -= 1;
    hi += 1;
  }
  const px = (k) => (lastIter ? (k / lastIter) * 600 : 300);
  const py = (f) => 196 - ((f - lo) / (hi - lo)) * 192;
  $('trace').setAttribute(
    'points',
    finitePoints.map(([k, f]) => `${px(k).toFixed(1)},${py(f).toFixed(1)}`).join(' '),
  );
  const [k, f] = finitePoints.at(-1);
  $('plot-note').textContent =
    `Iteration ${num(k)}: objective ${sci(f)}. The axis runs from ${sci(lo)} to ${sci(hi)}.`;
}

// --- downloads -------------------------------------------------------------

function download(filename, text, mime) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  // Revoke on the next turn: Safari needs the object URL to outlive the
  // synchronous click handler.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

// The solution downloads are formatted in wasm, not assembled here: the
// result JSON this page renders truncates long vectors so a large model stays
// displayable, and a download that silently stopped at 2,000 rows would be
// worse than no download at all.
$('dl-sol').addEventListener('click', () =>
  worker.postMessage({
    type: 'export',
    format: 'sol',
    filename: `${modelName}.sol`,
    mime: 'text/plain',
  }),
);

$('dl-csv').addEventListener('click', () =>
  worker.postMessage({
    type: 'export',
    format: 'csv',
    filename: `${modelName}-solution.csv`,
    mime: 'text/csv',
  }),
);

// The log is already whole on this side — it is what the solver printed.
$('dl-log').addEventListener('click', () =>
  download(`${modelName}-solve.log`, log.textContent, 'text/plain'),
);

// --- rendering -------------------------------------------------------------

const num = (v) => (v == null ? '—' : v.toLocaleString());
const sci = (v) => {
  if (v == null || Number.isNaN(v)) return '—';
  if (v === 0) return '0';
  return Math.abs(v) >= 1e-4 && Math.abs(v) < 1e6
    ? v.toPrecision(6).replace(/\.?0+$/, '')
    : v.toExponential(2);
};

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function stats(target, entries) {
  target.innerHTML = '';
  for (const [label, value] of entries) {
    const wrap = document.createElement('div');
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    if (value instanceof Node) dd.append(value);
    else dd.textContent = value;
    wrap.append(dt, dd);
    target.append(wrap);
  }
}

function renderSummary(s) {
  const b = s.var_bounds;
  stats($('stats'), [
    ['Variables', num(s.n_vars)],
    ['Constraints', num(s.n_cons)],
    ['Objective', s.sense],
    ['Degrees of freedom', num(s.degrees_of_freedom)],
    ['Equalities', num(s.n_equality_cons)],
    ['Inequalities', num(s.n_inequality_cons)],
    ['Nonlinear rows', `${num(s.n_nonlinear_cons)} / ${num(s.n_cons)}`],
    ['Nonlinear vars', `${num(s.n_nonlinear_vars)} / ${num(s.n_vars)}`],
    ['Jacobian nnz', `${num(s.nnz_jac)} (${(s.jac_density * 100).toFixed(2)}%)`],
    ['Hessian nnz', num(s.nnz_hess)],
    ['Bounded vars', num(b.boxed + b.lower_only + b.upper_only)],
    ['Fixed vars', num(b.fixed)],
    ['Free vars', num(b.free)],
  ]);

  const notes = [];
  if (s.external_funcs.length) {
    notes.push(
      `Model declares AMPL imported functions (${s.external_funcs.join(', ')}), ` +
        `which need a native shared library — this build cannot evaluate them.`,
    );
  }
  if (s.truncated) {
    notes.push(`Name / value lists are truncated to the first ${num(s.preview_limit)} entries.`);
  }
  $('summary-note').textContent = notes.join(' ');

  $('summary-section').hidden = false;
  $('solve-section').hidden = false;
}

function renderResult(r) {
  const badge = document.createElement('span');
  badge.className = `tag ${r.success ? 'ok' : 'bad'}`;
  badge.textContent = spaced(r.status);

  stats($('result-stats'), [
    ['Status', badge],
    ['Objective', sci(r.objective)],
    ['Iterations', num(r.iterations)],
    ['Solve time', `${(r.wall_time_secs * 1000).toFixed(0)} ms`],
    ['Constraint violation', sci(r.constraint_violation)],
    ['Dual infeasibility', sci(r.dual_infeasibility)],
    ['Complementarity', sci(r.complementarity)],
    ['KKT error', sci(r.kkt_error)],
    ['f / ∇f evals', `${num(r.evals.objective)} / ${num(r.evals.objective_grad)}`],
    ['g / ∇g evals', `${num(r.evals.constraints)} / ${num(r.evals.constraint_jac)}`],
    ['Hessian evals', num(r.evals.hessian)],
    ['Restorations', num(r.restoration_calls)],
  ]);

  const names = summary?.var_names ?? [];
  table($('vars'), ['Variable', 'Value'],
    r.x.map((v, i) => [names[i] ?? `x[${i}]`, sci(v)]));

  const conNames = summary?.con_names ?? [];
  table($('cons'), ['Constraint', 'Lower', 'Value', 'Upper', 'Multiplier'],
    r.g.map((v, i) => [
      conNames[i] ?? `c[${i}]`,
      finite(r.g_l[i]) ? sci(r.g_l[i]) : '−∞',
      sci(v),
      finite(r.g_u[i]) ? sci(r.g_u[i]) : '∞',
      sci(r.lambda?.[i]),
    ]));

  $('result-note').textContent = r.truncated
    ? `Tables show the first ${num(summary?.preview_limit ?? 0)} rows. The .sol and CSV ` +
      `downloads carry every row.`
    : '';
  $('result-section').hidden = false;
}

// AMPL writes ±1e19 for "no bound".
const finite = (v) => v != null && Math.abs(v) < 1e19;

// `SolvedToAcceptableLevel` → `Solved To Acceptable Level`.
const spaced = (s) => String(s).replace(/([a-z])([A-Z])/g, '$1 $2');

function table(el, headers, rows) {
  el.innerHTML = '';
  const thead = el.createTHead().insertRow();
  for (const h of headers) {
    const th = document.createElement('th');
    th.textContent = h;
    thead.append(th);
  }
  const body = el.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const cell of row) tr.insertCell().textContent = cell;
  }
}
