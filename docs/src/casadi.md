# CasADi

POUNCE plugs into [CasADi](https://web.casadi.org/) as an `nlpsol`
plugin, so a model hands its problem to POUNCE the same way it would
hand it to Ipopt:

```python
import casadi as ca

x = ca.MX.sym("x", 2)
nlp = {"x": x, "f": (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2}

solver = ca.nlpsol("solver", "pounce", nlp, {"pounce": {"tol": 1e-9}})
sol = solver(x0=[0.5, 0.5])
```

and with CasADi's `Opti` front end:

```python
opti.solver("pounce", {"print_time": False}, {"tol": 1e-9, "print_level": 0})
```

Because POUNCE registers a genuine `nlpsol` plugin rather than a
Python-side wrapper, everything CasADi layers on top of a solver works
unchanged: `Opti`, MX graphs, parameters, embedding a solve inside
another `Function`, and **differentiating through the solve**.

## Install

The plugin is a small C++ shared library built against the CasADi you
have installed. Nothing is published to PyPI yet, so either route starts
from the repository.

**As a wheel.** `casadi/wheel/build.sh` builds the plugin for the CasADi
in your environment and packages it:

```bash
pip install casadi
git clone https://github.com/jkitchin/pounce && cd pounce/casadi/wheel
./build.sh && pip install dist/pounce_casadi-*.whl
```

```python
import casadi as ca
import pounce_casadi     # registers the plugin — no CASADIPATH, no file copying
```

Importing the package loads the plugin into the process and registers it
with CasADi directly, which leaves CasADi's own installation untouched
and its bundled plugins — Ipopt included — loadable alongside.

The wheel this produces is tagged `py3-none-<platform>`, and carries a
build of the plugin for each supported CasADi **minor** version under
`pounce_casadi/_plugins/<minor>/`, chosen on `casadi.__version__` at
import. The two axes are handled at different layers deliberately: the
platform is in the wheel tag, so `pip` refuses a wheel from the wrong
one, while the CasADi version cannot be expressed by any tag and is
resolved inside the package, with a plain `ImportError` when there is no
matching build. See [`casadi/wheel/README.md`](
https://github.com/jkitchin/pounce/blob/main/casadi/wheel/README.md) for
the release matrix.

**Or in place**, without packaging:

```bash
cargo build --release -p pounce-cinterface
cd casadi
make fetch-src     # CasADi source at your exact version, for its headers
make
make install       # copies the plugin next to the CasADi that loads it
```

`make install` writes into CasADi's own package directory, which its
plugin loader searches first — no environment variable and no `sudo`
(it is your `site-packages`). If CasADi lives somewhere you cannot
write, skip `make install` and set `CASADIPATH` to the `casadi/`
directory instead.

Verify with `make test`, which cross-checks POUNCE against CasADi's
bundled Ipopt on the same models (73 checks, also run in CI).

**Against a CasADi you built yourself.** The defaults above read the
installed *Python* CasADi, but nothing in the build requires Python —
every input is overridable, so a CI that builds CasADi from source can
build the plugin against it:

```bash
make -C casadi \
  CASADI_LIB=/opt/casadi/lib CASADI_INC=/opt/casadi/include \
  CASADI_SRC=/src/casadi CASADI_VER=3.7.2 CXX11_ABI=1
```

`CASADI_SRC` must be the directory *containing* `casadi/core/`, and
CasADi's `INSTALL_INTERNAL_HEADERS` defaults to **OFF**, so point it at
your CasADi source checkout — or build CasADi with
`-DINSTALL_INTERNAL_HEADERS=ON` and use `<prefix>/include` for both it
and `CASADI_INC`. `CXX11_ABI=1` matches a self-built CasADi's modern
libstdc++ string ABI. There is no CMake package for this yet; the full
recipe and its failure signatures are in [`casadi/README.md`](
https://github.com/jkitchin/pounce/blob/main/casadi/README.md).

Two constraints are worth knowing before you file a bug: the plugin must
be rebuilt for each CasADi **minor** version, and it must match CasADi's
libstdc++ ABI (the pip wheels use the pre-C++11 string ABI, which is the
Makefile's default). CasADi's loader does not check versions, so a
mismatch surfaces as an undefined-symbol error at load time.
[`casadi/README.md`](https://github.com/jkitchin/pounce/blob/main/casadi/README.md)
has the details and the failure signatures.

Building against a CasADi *nightly* works too. CasADi renamed a runtime
helper the plugin calls after 3.7.2 (gh#668), which broke the build for
anyone on master; the source now detects which name the installed CasADi
declares, so one tree compiles against 3.6, 3.7 and master unchanged,
with no build flags to pass.

## Options

| CasADi option | Meaning |
| --- | --- |
| `pounce` | Dict of POUNCE options, using Ipopt-compatible names — `tol`, `max_iter`, `print_level`, `mu_strategy`, `hessian_approximation`, `linear_solver`, … Anything you would put in an `ipopt.opt`. |
| `pass_nonlinear_variables` | Let CasADi work out which variables enter nonlinearly and tell POUNCE. Affects the limited-memory Hessian only — see below. |
| `nonlinear_variables` | The same subset, given explicitly as a list of booleans of length `nx`. |
| `clip_inactive_lam` | Zero the multipliers of demonstrably inactive bounds (default **true**). Needed for correct sensitivities — see below. `false` reproduces CasADi's ipopt-plugin default. |
| `inactive_lam_strategy`, `inactive_lam_value` | How that inactivity margin is sized: `reltol` (default) means `inactive_lam_value * constr_viol_tol`; `abstol` uses the value directly. Same meaning as in the ipopt plugin. |
| `warm_start_from_previous` | Carry the active-set-SQP working set from one call to the next (default **false**) — see below. |
| `grad_f`, `jac_g`, `hess_lag` | Supply your own derivative functions instead of the autogenerated ones. Signatures as in the ipopt plugin: `(x, p) -> (f, grad_f)`, `(x, p) -> (g, jac_g)`, `(x, p, lam_f, lam_g) -> triu(hess)`. A wrong shape is refused at construction with a message. |
| `convexify_strategy`, `convexify_margin`, `max_iter_eig` | Convexify the Lagrangian Hessian before it reaches the solver: `none` (default), `regularize`, `eigen-reflect`, `eigen-clip`. This is CasADi's own `Convexify`, the same code its ipopt plugin uses, so results match. Exact-Hessian path only. |
| `var_string_md`, `var_integer_md`, `var_numeric_md`, `con_string_md`, `con_integer_md`, `con_numeric_md` | Accepted so an ipopt script keeps working when it is swapped over, and echoed back through `stats()`. POUNCE has no metadata channel, so nothing is forwarded to the solver. |

Everything in CasADi's base `nlpsol` option set also applies:
`iteration_callback`, `iteration_callback_step`, `print_time`,
`bound_consistency`, `calc_lam_p`, `error_on_fail`, `discrete` (refused —
POUNCE is a continuous solver), and the rest.

Every one of these means what it means for CasADi's `ipopt` plugin, with
two deliberate differences, both noted where they appear below:
`clip_inactive_lam` defaults **on**, and `iteration_callback_step` does
not thin out `stats()["iterations"]`.

```python
solver = ca.nlpsol("solver", "pounce", nlp, {
    "print_time": False,                 # CasADi's own timing line
    "pounce": {
        "print_level": 5,                # POUNCE's iteration table
        "tol": 1e-9,
        "mu_strategy": "adaptive",
        "linear_solver": "ma57",
    },
})
```

An unknown option name is refused by POUNCE with a message naming it,
rather than being ignored.

An option's **type** comes from POUNCE's own registry, not from the
literal you wrote. This matters more than it sounds: `{"tol": 1}` is an
`int` in Python and a number to POUNCE, and forwarding it as an integer
gets it refused — leaving `tol` at its default while the script looks
like it set it. Write `1` or `1.0` for a numeric option and either
works. A `bool` reaches POUNCE's yes/no string options as `"yes"` /
`"no"`.

## Results and statistics

`solver.stats()` carries the usual CasADi keys plus POUNCE's
per-iteration trace:

```python
st = solver.stats()
st["success"]        # bool
st["return_status"]  # 'Solve_Succeeded', 'Maximum_Iterations_Exceeded', …
st["iter_count"]
st["t_solve_pounce"] # seconds inside POUNCE
st["iterations"]     # dict of per-iteration lists:
                     #   inf_pr, inf_du, mu, d_norm, regularization_size,
                     #   obj, alpha_pr, alpha_du, ls_trials, alg_mod

st["final_inf_pr"]      # final primal infeasibility
st["final_inf_du"]      # final dual infeasibility
st["final_compl_inf"]   # final complementarity error

st["restoration"]    # {'calls', 'inner_iters', 'outer_iters', 'wall_secs'}
st["linear_solver"]  # what the KKT backend did — see below
```

The `iterations` dict is the same data POUNCE prints in its iteration
table, so convergence plots need no stdout parsing. It describes the
most recent solve only: calling a solver in a loop does not concatenate
the traces.

### The linear solver

`stats()["linear_solver"]` reports what the KKT backend actually did:

```python
{'solver_name': 'feral',      # the backend that ran, not the one requested
 'n_factors': 17,             # factorizations over the solve
 'n_pattern_reuse': 16,       # …that reused the symbolic factorization
 'n_pattern_changes': 1,
 'max_fill_ratio': 1.0,       # nnz(L)/nnz(A); ≫10 means ordering trouble
 'min_abs_pivot': 1.0, 'max_abs_pivot': 2.0,
 'last_inertia': [2, 1, 0],   # (positive, negative, zero)
 'last_nnz_a': 6, 'last_nnz_l': 6}
```

`solver_name` is the direct answer to "did my `linear_solver` option take
effect?" — it names the backend that ran. Fields POUNCE did not measure
are **absent** rather than zero; in particular there are no phase
timings (symbolic analysis, numeric factorization, back-solve), because
POUNCE does not instrument those phases separately today.

### The structured solve report

`stats()` is the convenient view; POUNCE also writes a machine-readable
one. Set `solve_report` to a path and each solve leaves a
`pounce.solve-report/v1` JSON file — the same format the `pounce` CLI's
`--json-output` produces, so the tools that read those (`diagnose`,
`find_stalls`, `convergence_trace`) read this too.

```python
S = ca.nlpsol("S", "pounce", nlp, {
    "solve_report": "run.json",
    "solve_report_detail": "full",   # 'summary' (default) or 'full'
})
```

`full` embeds the per-iteration trajectory; `summary` omits it and
carries the problem, solution, statistics and linear-solver blocks.
The trajectory is not free — POUNCE has to retain each iterate as it
goes, which is why `summary` is the default and why the capture is
switched on before the solve rather than reconstructed after it. Asking
for `full` switches it on for you.

Two things worth knowing before you wire it into a loop:

- **The file is rewritten per solve.** A solver called repeatedly leaves
  only the last report. Give each call its own path if you want to keep
  them.
- **A write that fails does not fail the solve.** You get a warning and
  `stats()["solve_report_written"] == False`; the answer is still
  returned, because a diagnostic file is not worth an exception. Check
  that key rather than the log if a script depends on the file.

`solve_report_detail` is validated when the solver is constructed, so a
typo costs you the `nlpsol` call rather than a solve.

### Restoration

`stats()["restoration"]` counts restoration-phase entries, the
iterations its inner solver ran, and the seconds spent there — enough to
answer "did this solve struggle, and how much of it was restoration?"
without raising `print_level`.

Individual iterations are labelled too, by
`stats()["iterations"]["alg_mod"]`: `0` for an outer iteration, `1` for
one of the restoration subproblem. The solve-level dict above stays
useful alongside it — it is the only source for the inner iteration
count and the wall time, and it answers the question in one read.

**Read `alg_mod` before plotting anything else.** On a restoration row
every other column describes the min-‖c‖₁ *feasibility subproblem*, not
your NLP: its objective is the constraint-violation penalty, and its
`inf_pr` falls to zero as the subproblem converges while your problem's
violation is untouched. Plotted on one axis without splitting on
`alg_mod`, a restoration episode looks like the objective exploding and
the infeasibility being solved, and neither happened.

```python
it = st["iterations"]
outer = [(i, o) for i, (o, m)
         in enumerate(zip(it["obj"], it["alg_mod"])) if m == 0]
```

`iteration_callback` is not called for restoration iterations. CasADi
fixes its signature at `(x, f, g, lam_x, lam_g)` and a restoration
iterate supplies none of them — it is a point of a different problem, in
that problem's variable space. The trace still records those iterations,
so nothing is hidden; they simply are not handed to a callback that
would have to interpret them as a solution estimate.

`lam_p` deserves a note because its sign surprises people. CasADi's
`Nlpsol` base class computes it — no plugin is involved, which is why
POUNCE and Ipopt agree on it bit for bit — and it negates the result
(`nlpsol.cpp`: `casadi_scal(np_, -1., d_nlp->lam_p)`). So

```
lam_p = -df*/dp
```

not `+df*/dp`, where `f*` is the optimal objective. Both the Ipopt
agreement and the sign are pinned in the parity suite against a finite
difference of `f*`.

## Iteration callbacks

CasADi's `iteration_callback` is handed the live iterate — `x`, `f`,
`g`, `lam_x`, `lam_g` — once per iteration, and returning nonzero asks
the solver to stop (`User_Requested_Stop`):

```python
solver = ca.nlpsol("solver", "pounce", nlp, {"iteration_callback": watcher})
```

Worth noting if you are coming from the Ipopt plugin: a stock Ipopt build
cannot supply the iterate, and CasADi warns that *"intermediate_callback
is disfunctional in your installation"*. POUNCE serves live iterates
through its C API, so the callback receives real values with no special
build. See
[`casadi/examples/06_iteration_callback.py`](https://github.com/jkitchin/pounce/blob/main/casadi/examples/06_iteration_callback.py).

`iteration_callback_step` thins the callback out — `3` calls it every
third iteration — for a callback expensive enough that you would rather
not pay for it every time. One difference from the ipopt plugin: there,
the step also thins `stats()["iterations"]`, because the whole
intermediate callback returns early. Here the trace is always complete.
Throttling a plotting callback and losing the convergence history are
unrelated wishes, and only one of them was ever asked for.

A callback that raises does not take the process down (see [When your
model raises](#when-your-model-raises)); `iteration_callback_ignore_errors`
decides whether the solve continues or stops.

### Diagnostics from inside the callback

CasADi fixes the callback's inputs at `x`, `f`, `g`, `lam_x`, `lam_g`,
which leaves out most of what a progress display wants. The rest is
reachable without parsing the solver's log: **`solver.stats()` is
callable from inside the callback**, and mid-solve it describes the
iteration you are in.

```python
def eval(self, arg):
    st = solver.stats()
    it = st["iterations"]
    # the last entry of each trace is *this* iteration
    print(f"mu={it['mu'][-1]:.2e} inf_pr={it['inf_pr'][-1]:.2e} "
          f"step={it['d_norm'][-1]:.2e} ls_trials={it['ls_trials'][-1]}")

    v = st["current_violations"]      # present only while solving
    v["x_L_violation"], v["x_U_violation"]
    v["compl_x_L"], v["compl_x_U"]
    v["grad_lag_x"]
    v["nlp_constraint_violation"], v["compl_g"]
    return [0]
```

`current_violations` is Ipopt's `GetIpoptCurrentViolations` field set,
fetched on demand: it appears only while a solve is in flight and costs
nothing on the `stats()` call you make afterwards. Symmetrically,
`final_inf_pr` and friends appear only once the solve has ended — no key
is ever served with a stale or invented value.

## Warm starting

Pass the previous solution back in, and turn on the two options that make
POUNCE use the multipliers you supply (without them they are outputs
only — Ipopt's contract too):

```python
warm = ca.nlpsol("warm", "pounce", nlp, {"pounce": {
    "warm_start_init_point": "yes",
    "mu_init": 1e-6,
}})
sol2 = warm(x0=sol["x"], lam_g0=sol["lam_g"], lam_x0=sol["lam_x"], p=p_next, ...)
```

On the 20-step MPC in
[`casadi/examples/03_mpc_warm_start.py`](https://github.com/jkitchin/pounce/blob/main/casadi/examples/03_mpc_warm_start.py)
this takes the loop from a mean of 8.1 iterations per step to 5.3.

### Carrying the working set between calls

`x0` / `lam_g0` / `lam_x0` restart the *iterate*. The active-set SQP has a
second thing worth reusing: the **working set** — which bounds and
constraints it found active — and identifying that set is most of what a
QP solve does. There is no slot in `nlpsol`'s fixed input signature to
pass one, so the plugin can carry it for you, from one call of the same
solver object to the next:

```python
solver = ca.nlpsol("solver", "pounce", nlp, {"pounce": {
    "algorithm": "active-set-sqp",
    "warm_start_init_point": "yes", "mu_init": 1e-6,
}, "warm_start_from_previous": True})
```

**Turn this on if you select `active-set-sqp` for a receding-horizon
loop.** On a cart-pole MPC whose force limits saturate — so the active
set is large and genuinely has to be found — carrying it is the
difference between the SQP being the fastest option available and the
worst:

| 30-step loop, per solve | mean | max |
| --- | --- | --- |
| active-set SQP, no working set | 233 ms | 375 ms |
| active-set SQP, `warm_start_from_previous` | 18 ms | 23 ms |
| interior point, warm started (reference) | 27 ms | 38 ms |

An order of magnitude, and the control trajectory is identical throughout
(`max |Δu₀| ≈ 3e-11`): the working set is a starting guess for the QP,
not a constraint on the answer. The iteration counts barely move
(2.86 → 2.79) — the saving is *inside* each SQP iteration, in the QP that
no longer has to rediscover the active set from scratch. Those numbers
are the run saved in
[`python/notebooks/35_casadi.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/35_casadi.ipynb);
the ratio moves between runs, the order of magnitude does not.

On a loop whose bounds stay inactive there is much less to reuse; the
same measurement on the unsaturated version of that model gives 11.0 ms
against 10.1 ms, which is close to noise.

Two things to know before switching it on:

- **It makes the function stateful.** Call *k+1* starts from what call
  *k* found, so a solver object is no longer a pure map of its inputs.
  That is why it is off by default. Evaluate the same solver on
  unrelated problems in an interleaved way and each will hand the other
  a misleading guess — use separate solver objects.
- **A stale set is refused, not obeyed.** Bounds arrive as per-call
  inputs and may have moved under the stored set; POUNCE validates it
  against the model and rejects it, and that call cold-starts its
  working set. `stats()["warm_started_working_set"]` reports whether the
  call actually started from one.

Inert under the interior-point default, which produces no working set.

## Differentiating through a solve

A `nlpsol` object is a CasADi `Function`, so it composes into larger
graphs and can be differentiated — CasADi's `Nlpsol` base class builds
forward and adjoint derivatives of the solution map by linearizing the
KKT system, and every plugin inherits it:

```python
sol = solver(x0=x0, p=p, lbg=-ca.inf, ubg=0)
dx_dp = ca.Function("dx_dp", [p], [ca.jacobian(sol["x"], p)])
```

This is exact (implicit function theorem), not a finite difference, and
it makes bilevel problems and parameter estimation over a POUNCE solve
just work. It is also only as good as the multipliers, so tighten `tol`
if the sensitivities look off.
[`casadi/examples/04_parametric_sensitivity.py`](https://github.com/jkitchin/pounce/blob/main/casadi/examples/04_parametric_sensitivity.py)
checks it against a finite difference and then solves a bilevel problem.

### Bounded variables and silently-zero gains

There is a trap here that costs correctness, not speed, and it is worth
understanding before you trust a gain.

An interior-point method drives the multipliers of untouched bounds
toward zero without reaching it — POUNCE leaves ~1e-12 on them. CasADi's
solution-map derivative reads *any* nonzero bound multiplier as an active
constraint and holds that variable fixed, so one residual 1e-12 turns the
whole sensitivity row into zeros. On an NMPC model with bounded controls,
`jacobian(u0, x0)` — the feedback gain — then reads exactly zero where a
re-solve says −9.11.

The plugin therefore clips the multipliers of demonstrably inactive
bounds to zero by default, testing *primal distance to the bound* rather
than multiplier magnitude. It is the same rule, option name and margin as
CasADi's ipopt plugin `clip_inactive_lam` — with the default flipped,
because that plugin defaults it off and a silently-zero gain is a bad
default. `clip_inactive_lam=False` restores the Ipopt-identical
behaviour.

```python
# with the default, the analytic gain matches a re-solve;
# with clip_inactive_lam=False it comes back as 0.0
gain = ca.Function("gain", [x0_par], [ca.jacobian(sol["x"][iu0], x0_par)])
```

Pinned by `test_nmpc_feedback_gain_is_not_silently_zero` in
`casadi/test_parity.py`.

### POUNCE-specific algorithms

`algorithm=active-set-sqp` selects POUNCE's
[active-set SQP](active-set-sqp.md) driver through the ordinary option
dict — no plugin support needed, and it agrees with the interior-point
default (checked in the parity suite). Whether it is *faster* depends on
the problem; the notebook measures both on an MPC model.

What the plugin does **not** expose yet is the machinery that needs an
API beyond `nlpsol`'s: POUNCE's own
[parametric sensitivity](sensitivity.md) (the factor-once/solve-many
session, which would replace CasADi's generic KKT linearization) and
[working-set warm starts](active-set-sqp-warm-start.md) carried between
calls. Both are reachable from the C API the plugin already uses; say so
on the issue tracker if you want them.

## Limited-memory Hessians and nonlinear variables

**You may be on this path without having asked for it.** The plugin sets
`hessian_approximation=limited-memory` for you whenever an exact
Lagrangian Hessian is not available — the `!exact_hessian_` branch in
`casadi_nlpsol_pounce.cpp`. If you did not supply second derivatives,
every solve is an L-BFGS solve, and the options in this section apply to
it. That is worth knowing before comparing POUNCE against another
solver: you are comparing quasi-Newton runs, and quasi-Newton runs are
much more sensitive to the options below than exact-Hessian ones.

With `hessian_approximation=limited-memory`, POUNCE approximates
curvature over every variable by default. If your model is mostly linear
— slacks, balances, flows with constant coefficients — you can tell it
which variables actually enter nonlinearly:

```python
solver = ca.nlpsol("solver", "pounce", nlp, {
    "pass_nonlinear_variables": True,             # CasADi derives the set
    # or: "nonlinear_variables": [True, True] + [False] * n_lin,
    "pounce": {"hessian_approximation": "limited-memory"},
})
```

CasADi derives the set with `which_depends`; POUNCE then restricts the
L-BFGS update to that subspace, so no curvature is learned or stored for
the rest (they keep only a small diagonal floor — see below). It is an
approximation-space restriction, not a different problem — the KKT point
is unchanged, which
[`casadi/examples/05_limited_memory_mask.py`](https://github.com/jkitchin/pounce/blob/main/casadi/examples/05_limited_memory_mask.py)
demonstrates.

**Whether it is faster depends on the model, so measure.** The restriction is
a different approximation, so it takes a different path to the same KKT point.
Measured on a synthetic model with 2 nonlinear variables (saved outputs in the
notebook below):

| linear variables | full space | masked |
| --- | --- | --- |
| 2 000 | 0.86 s, 25 iterations | 1.13 s, 28 iterations |
| 10 000 | 8.3 s, 31 iterations | 7.1 s, 27 iterations |

The saving is in the curvature information and the stored columns, not in the
linear algebra, so it only pays once the linear block dominates — and a model
that is mostly nonlinear has nothing to gain.

For contrast, the same 2000-variable model through CasADi's Ipopt plugin goes
from 0.40 s unmasked to **399 s** masked. Ipopt zeroes the quasi-Newton
diagonal on the linear block, which leaves those rows of the KKT system
carrying only the barrier term and makes the symmetric factorization pay for a
near-singular diagonal. POUNCE keeps a small curvature floor there instead
(`limited_memory_init_val_min`, 1e-8 by default), which avoids the cliff
entirely — a deliberate divergence from upstream, documented at the code site
in `crates/pounce-algorithm/src/hess/lim_mem_quasi_newton.rs`.

The underlying entry point is `IpoptSetNonlinearVariables` in POUNCE's C
API, and `num_linear_variables` is the Ipopt-compatible
contiguous-prefix fallback.

### The initial Hessian scalar, and matching an Ipopt baseline

The L-BFGS model is `B = σI + VVᵀ − UUᵀ`. The rank-2 corrections come
from the curvature history; `σ` is the diagonal they sit on, and
`limited_memory_initialization` picks the formula for it. `scalar1`
(σ = sᵀy/sᵀs, the default, matching Ipopt) and `scalar2` (σ = yᵀy/sᵀy)
are related by

```
σ_scalar2 / σ_scalar1 = (yᵀy · sᵀs) / (sᵀy)²   ≥ 1
```

which is unbounded as the curvature pair becomes ill-conditioned. On a
well-scaled problem they are close; on a large collocation model they
need not be within six orders of magnitude. An over-large `σ` makes the
diagonal swamp the corrections, the model collapses toward a multiple of
the identity, and the primal step goes with it.

The symptom is recognisable without instrumenting anything:

- primal step sizes (`alpha_pr`) collapsing to `1e-3` or below and
  staying there,
- primal infeasibility (`inf_pr`) barely moving,
- dual infeasibility (`inf_du`) climbing by orders of magnitude,
- the barrier parameter `lg(mu)` stuck, because the barrier will not
  descend until the subproblem error comes down.

If you are comparing against an Ipopt run and the two agree for the first
iteration and then separate, suspect `σ`: it is hard-coded to
`limited_memory_init_val` (1.0) while the curvature history is empty, so
iteration 1 cannot differ. The first curvature pair lands at iteration 2,
and that is where a `σ` disagreement first shows.

`limited_memory_init_val_max` clamps `σ` however it was computed, and is
the blunt instrument if you suspect it but cannot change the formula:

```python
"pounce": {"hessian_approximation": "limited-memory",
           "limited_memory_init_val_max": 10.0}   # default 1e8
```

> **Changed in #677.** Every release before this used `scalar2` and
> *ignored* `limited_memory_initialization` entirely — it was registered
> but never read, so setting it had no effect and no warning. If you are
> reproducing older POUNCE results, set `limited_memory_initialization
> scalar2` explicitly.

### When the duals will not converge

A quasi-Newton dual step is computed from an approximate Hessian, so an
L-BFGS solve can settle a feasible primal, park the objective, and still
fail to drive dual infeasibility to tolerance — `inf_du` oscillating
inside a band instead of descending, for hundreds of iterations, while
`inf_pr` and the objective are already where they should be.

That shape is what `recalc_y` is for. It re-estimates the equality and
inequality multipliers by least squares on every iteration whose
constraint violation is under `recalc_y_feas_tol`, side-stepping the
Hessian approximation:

```python
"pounce": {"hessian_approximation": "limited-memory",
           "recalc_y": "yes",
           "recalc_y_feas_tol": 1e-6}   # the default gate
```

Each firing costs an extra augmented-system solve.

**POUNCE leaves this off by default and Ipopt does not.** Ipopt's option
text says it is used by default with a quasi-Newton Hessian; enabling it
by default here regressed 7 of 57 fixtures from solved to not solved,
because re-estimating `y` every iteration also discards Newton
multipliers that were converging perfectly well. So it is opt-in. If you
are chasing Ipopt parity on an L-BFGS model, this is one of the two
options — with `limited_memory_initialization` — most likely to explain
a difference.

## Examples

All runnable from
[`casadi/examples/`](https://github.com/jkitchin/pounce/tree/main/casadi/examples)
with `make examples`:

| Script | Shows |
| --- | --- |
| `01_rosenbrock.py` | The basics: `nlpsol`, options, results, stats |
| `02_opti_rocket.py` | `Opti` on a small optimal-control problem |
| `03_mpc_warm_start.py` | Receding-horizon loop with warm starts |
| `04_parametric_sensitivity.py` | `jacobian` through a solve; a bilevel problem |
| `05_limited_memory_mask.py` | `pass_nonlinear_variables` with L-BFGS |
| `06_iteration_callback.py` | Live iterates and early termination |
| `07_custom_derivatives_and_saving.py` | Your own `grad_f`/`jac_g`/`hess_lag`, `convexify_strategy`, and `save`/`load` |
| `08_codegen_embedded.py` | `generate()` the whole solve to C, compile it, check it matches |

## Notebook

[`python/notebooks/35_casadi.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/35_casadi.ipynb)
walks through all of the above end to end with saved outputs — first solve and
convergence plot, `Opti` optimal control, the warm-started MPC loop, the
sensitivity and bilevel example, the nonlinear-variable mask measured at two
problem sizes, and the Ipopt cross-check.

## When your model raises

POUNCE is Rust behind a C API, and an exception unwinding out of an oracle
callback into Rust frames aborts the process — `fatal runtime error:
Rust cannot catch foreign exceptions`. So the plugin converts at the
boundary rather than propagating through it. A model containing a
`casadi.Callback` that raises reports the error and fails that
evaluation, which the solver treats as an un-evaluable point and
responds to by cutting the step:

```
POUNCE: objective evaluation failed: boom: the user's model raised
...
return_status = 'Invalid_Number_Detected'
```

Identical to what CasADi's Ipopt plugin does with the same model. A
transient bad point is therefore recoverable rather than fatal, and a
genuinely broken model gives you a status and a message instead of a
dead interpreter.

A `KeyboardInterrupt` is treated differently from an evaluation error:
it is remembered, the solve is stopped at the next iteration
(`User_Requested_Stop`), and the interrupt is re-raised once control is
back on the C++ side — so Ctrl-C is responsive without crossing the
language boundary. `iteration_callback_ignore_errors` (CasADi's base
option) decides whether a *throwing iteration callback* stops the solve
or is shrugged off.

## Printing from a callback

If your own code prints from inside `iteration_callback` — or from a
model that logs during function evaluation — be aware that two writers
share stdout. POUNCE journals from Rust, where the stream goes out on
every newline; CasADi writes through `uout()` and leaves the buffering to
whatever sits behind it, which behind a pipe is a fully buffered stream.
A line long enough to straddle that buffer can therefore be split in two
by a POUNCE iteration row landing in the middle of it. A line-oriented
protocol reading its own stdout sees a line arrive without its
terminator, and the remainder show up several lines later.

For a **C++ host** the plugin handles this: it drains CasADi's streams on
every exit from a callback and once more before the solve starts, which
are the only moments it can know POUNCE is not writing (gh#667). Pinned
by `test_output_interleaving.cpp` in the parity suite.

For a **Python host** the plugin cannot help, and this is worth
understanding rather than working around blindly. CasADi's bindings point
`Logger::writeFun` at `PySys_WriteStdout` but leave `Logger::flush` at its
default, so output lands in Python's `sys.stdout` while a flush from the
plugin drains `std::cout` — a different buffer. Nothing the plugin does
from C++ reaches Python's. Until POUNCE's journal is routed through
`uout()` (gh#667 again — the general fix), make the buffer stop holding
partial lines:

```python
import sys
sys.stdout.reconfigure(line_buffering=True)     # or run python -u
```

That is sufficient, not just a mitigation: your callback runs while
POUNCE is blocked, so a line that is flushed by the time the callback
returns cannot be torn. The same applies to Ipopt — this is a property of
printing from callbacks, not something specific to POUNCE, and the
plugin's C++-side flushing brings the two to parity.

## Threads

`Function.map(N, "thread")` works. CasADi hands each worker its own
memory object, and the plugin keeps every piece of per-solve state there
— buffers, the iteration trace, the carried working set — so a batched
solve reproduces the serial answers exactly:

```python
batched = solver.map(24, "thread", 8)
out = batched(x0=X0, p=P, lbg=-ca.inf, ubg=0)     # bit-identical to a loop
```

Pinned in the parity suite (24 solves over 8 threads, `max |Δx| = 0`),
and stress-run at 48 solves over 8 threads. What is *not* safe is
driving one memory object from two threads at once, which CasADi does
not do. Note that `warm_start_from_previous` is per memory object, so
each worker carries its own working set.

## Saving and reloading a solver

`save` / `load` round-trip the solver, as they do for CasADi's own
plugins:

```python
solver.save("solver.casadi")
again = ca.Function.load("solver.casadi")     # solves identically
```

What crosses is configuration — the oracle, the sparsities, the option
dict, the metadata. What does not is per-solve state: a reloaded solver
is a cold one, and a working set carried under
`warm_start_from_previous` belongs to the memory object, which is never
serialized. Reading the file needs the plugin loadable in the reading
process (`import pounce_casadi`, or the plugin on the search path) — the
rule for every out-of-tree CasADi plugin. Without it the failure is a
clean *"Plugin 'pounce' is not found"*.

## Code generation

`generate()` on an `nlpsol` emits the model **and** the solve — the
oracle functions, the option calls, and the loop that drives them — as
one C file:

```python
solver = ca.nlpsol("mpc_step", "pounce", nlp, {"pounce": {"tol": 1e-9}})
solver.generate("mpc_step.c")
```

```bash
cc -O2 -shared -fPIC -o mpc_step.so mpc_step.c \
   -I .../crates/pounce-cinterface/include \
   -L .../target/release -lpounce_cinterface -lm
```

Neither CasADi nor Python is on that command line, and neither is needed
at run time — which is the point, for firmware, a ROS node, or a
real-time target. What *is* needed is `libpounce_cinterface`: the
generated file includes `pounce.h` and calls the solver through it, the
same way CasADi's generated Ipopt code includes
`<coin-or/IpStdCInterface.h>` and links libipopt. This is linked
codegen, not freestanding C, so it does not reach the smallest
microcontrollers.

The generated solve reaches the same point as the interpreted one — `x`,
`f`, `lam_x`, `lam_g` all bit-identical, pinned in the parity suite,
which compiles a generated file and runs it on every CI build. That
includes `clip_inactive_lam`, reproduced inside the emitted runtime
rather than skipped, and the L-BFGS nonlinear-variable subset, emitted
as a static index array.

Three options cannot be reproduced in generated code, and `generate()`
refuses them by name rather than quietly dropping them:

| Option | Why |
| --- | --- |
| `iteration_callback` | The callback is a CasADi `Function` living in this process; generated code runs without CasADi. |
| `warm_start_from_previous` | It carries an active set between calls of one solver *object*; the generated entry point has no such channel. Pass `x0` / `lam_g0` / `lam_x0` instead. |
| `convexify_strategy` | Not emitted yet. |
| `solve_report` | The generated code links the same C API and could call `IpoptWriteSolveReport`, but the emitted runtime does not. Refused rather than dropped, so you are not left waiting for a file that never appears. |

The runtime the plugin emits is `casadi/pounce_runtime.hpp`, the
counterpart of CasADi's `ipopt_runtime.hpp`. Worked example:
[`casadi/examples/08_codegen_embedded.py`](https://github.com/jkitchin/pounce/blob/main/casadi/examples/08_codegen_embedded.py).

## What is not supported

- **Integer variables.** POUNCE is a continuous local NLP solver, so
  `discrete` is refused by CasADi's base class rather than quietly
  relaxed.
- **Metadata forwarding.** `var_*_md` / `con_*_md` are accepted and
  echoed back through `stats()`, but POUNCE has no metadata channel to
  forward them to, so they do not reach the solver.
- **`convexify_strategy="regularize"` on a Hessian with off-diagonal
  entries.** CasADi's `Convexify` takes that strategy only for an input
  whose *pattern* is symmetric, and the Hessian both this plugin and the
  ipopt plugin build is upper triangular — so it works for a diagonal
  Hessian and is refused with *"Only truly symmetric matrices
  supported"* otherwise. Identical in both plugins; `eigen-clip` and
  `eigen-reflect` have no such restriction, and POUNCE's own
  inertia-correcting regularization is on by default regardless.
- **Linear-solver phase timings.** Reported at solve level only
  (`stats()["linear_solver"]`), because that is the granularity POUNCE
  measures — the per-phase numbers are absent rather than zero. (The
  per-iteration restoration flag that used to sit in this bullet now
  ships: see `iterations["alg_mod"]` above.) See
  [`dev-notes/casadi-diagnostics-and-native-builds.md`](https://github.com/jkitchin/pounce/blob/main/dev-notes/casadi-diagnostics-and-native-builds.md)
  for what each would take.
- **Native (non-Python) plugin builds and prebuilt plugin archives.**
  The plugin builds against a Python-installed CasADi; there is no CMake
  path taking a native CasADi SDK, and no per-platform archive is
  published. Tracked in the same note.

## AMPL fallback

CasADi also ships an `ampl` plugin that writes an `.nl` file and shells
out to a solver binary, and POUNCE reads `.nl`. It is a poor substitute
— CasADi's AMPL interface accepts only SX models with no parameters, and
returns no bound multipliers — so prefer the plugin above. It exists as
a fallback for a model that already fits those limits.
