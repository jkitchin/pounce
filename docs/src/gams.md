# GAMS

POUNCE plugs into [GAMS](https://www.gams.com/) as an NLP solver, so a model
can hand its problem to POUNCE with:

```
option nlp = pounce;
solve mymodel using nlp minimizing obj;
```

There are two ways to make POUNCE available to GAMS. Pick one:

| Route | Install | What it is |
| --- | --- | --- |
| **pip (recommended)** | `pip install pounce-solver[gams]` then `pounce-gams register` | A pure-Python solver link built on GAMS's own `gamsapi` package. No compiler, no `sudo`, survives GAMS upgrades. |
| **native C link** | build + `sudo make -C gams install` | A C shared library installed into the GAMS system directory. Adds active-set-SQP working-set / state-file warm starts. See [`gams/README.md`](https://github.com/jkitchin/pounce/blob/main/gams/README.md). |

Both register POUNCE under the same name (`pounce`) for `NLP`, `DNLP`, and
`RMINLP` models — POUNCE is a continuous local NLP solver, so mixed-integer and
conic model types are not offered here.

## The pip route

### 1. Install

```
pip install pounce-solver[gams]
```

The `[gams]` extra pulls in
[`gamsapi[core]`](https://pypi.org/project/gamsapi/) — GAMS's own expert-level
GMO/GEV Python bindings — and PyYAML. The bindings `dlopen` the GAMS C
libraries from your local install, so **`gamsapi` must match your GAMS
version**. POUNCE itself redistributes nothing GAMS-owned. If your GAMS and
`gamsapi` versions disagree, install the matching one from your GAMS system
(GAMS ships a `gamsapi` wheel under `apifiles/Python/`), or:

```
pip install 'gamsapi[core]==<your GAMS X.Y.Z>'
```

### 2. Check the install

```
pounce-gams status
```

reports whether `gamsapi` imports, the config directory POUNCE will register
into, and whether POUNCE is already registered:

```
gamsapi:       available
               gamsapi 53.2.0 importable
config dir:    /Users/you/Library/Preferences/GAMS
gamsconfig:    /Users/you/Library/Preferences/GAMS/gamsconfig.yaml (missing)
POUNCE solver: not registered
```

### 3. Register

```
pounce-gams register
```

This writes a tiny launcher script and a `solverConfig` entry into your GAMS
per-user `gamsconfig.yaml`. It **merges** — any other solvers already in that
file (CONOPT overrides, discopt, …) are preserved — and is idempotent (re-running
just updates POUNCE in place). The per-user config directory GAMS searches is
OS-specific:

| OS | Directory |
| --- | --- |
| macOS | `~/Library/Preferences/GAMS` |
| Linux | `$XDG_CONFIG_HOME/GAMS` (else `~/.config/GAMS`) |
| Windows | `%LOCALAPPDATA%\GAMS` (else `…\Documents\GAMS`) |

Override the target with `--config-dir <path>` (e.g. to register into the GAMS
system directory instead). To undo, `pounce-gams unregister`.

No `sudo` is needed and nothing is written into the GAMS system directory, so a
GAMS upgrade does not wipe the registration.

### 4. Solve

```
option nlp = pounce;
solve mymodel using nlp minimizing obj;
```

GAMS invokes the launcher with a control file; the launcher runs the Python
link, which reads the model through GMO/GEV, solves it with POUNCE, and writes
the primal/dual solution and GAMS model/solve status back.

### Marginals

Equation marginals (`.M`) follow the usual GAMS convention: they are stated
against the objective **as you wrote it**, for both `minimizing` and
`maximizing` models. POUNCE always minimizes internally, so for a
`maximizing` model it solves `min(−f)` and converts its multipliers back
via `pi = −obj_sign · λ` (see `gams_pi` in `python/pounce/gams/link.py` and
the matching block in `gams/gams_pounce.c`).

Variable marginals (`.M` on variables, i.e. reduced costs) are
`z_L − z_U`, carrying the same `obj_sign` factor.

> Before v0.9.1 the conversion omitted the `obj_sign` factor, so equation
> marginals on `maximizing` models came back with the wrong sign
> ([#272](https://github.com/jkitchin/pounce/issues/272)). `minimizing`
> models, objective values, variable marginals and status mapping were
> never affected. Both the pip link and the native C link were affected
> identically, so the install method made no difference.

## Option files

If a model sets `mymodel.optfile = 1`, POUNCE reads `pounce.opt` (`.op2`,
`.op3`, … for higher `optfile` values). Each line is a `keyword value` pair
using POUNCE's [option names](options.md); lines starting with `*` or `#` are
comments. The GAMS `iterlim` and `reslim` are honored as `max_iter` and
`max_wall_time`.

```
* pounce.opt
tol        1e-10
max_iter   500
```

### Machine-readable solve report

Set `json_output` in `pounce.opt` to also emit a structured
`pounce.solve-report/v1` JSON report (identical to the CLI's `--json-output`,
consumable by `pounce-studio`):

```
json_output  my_solve.json
json_detail  full        * "summary" or "full"; default is "full"
```

See the [JSON Solve Report](json-output.md) schema for the format.

## Notes & limitations

- **Version match.** The single most common failure is a `gamsapi` ↔ GAMS
  version mismatch; `pounce-gams status` diagnoses it.
- **Warm starts.** The active-set-SQP working-set / state-file warm-start
  features (`algorithm active-set-sqp`, `sqp_state_file`) are currently only in
  the native C link, where each `solve` reuses an in-process state. The pip
  link runs each solve as a fresh process; full warm-start parity is a planned
  follow-up.
