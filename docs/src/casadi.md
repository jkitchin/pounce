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
bundled Ipopt on the same models.

Two constraints are worth knowing before you file a bug: the plugin must
be rebuilt for each CasADi **minor** version, and it must match CasADi's
libstdc++ ABI (the pip wheels use the pre-C++11 string ABI, which is the
Makefile's default). CasADi's loader does not check versions, so a
mismatch surfaces as an undefined-symbol error at load time.
[`casadi/README.md`](https://github.com/jkitchin/pounce/blob/main/casadi/README.md)
has the details and the failure signatures.

## Options

| CasADi option | Meaning |
| --- | --- |
| `pounce` | Dict of POUNCE options, using Ipopt-compatible names — `tol`, `max_iter`, `print_level`, `mu_strategy`, `hessian_approximation`, `linear_solver`, … Anything you would put in an `ipopt.opt`. |
| `pass_nonlinear_variables` | Let CasADi work out which variables enter nonlinearly and tell POUNCE. Affects the limited-memory Hessian only — see below. |
| `nonlinear_variables` | The same subset, given explicitly as a list of booleans of length `nx`. |
| `clip_inactive_lam` | Zero the multipliers of demonstrably inactive bounds (default **true**). Needed for correct sensitivities — see below. `false` reproduces CasADi's ipopt-plugin default. |
| `inactive_lam_strategy`, `inactive_lam_value` | How that inactivity margin is sized: `reltol` (default) means `inactive_lam_value * constr_viol_tol`; `abstol` uses the value directly. Same meaning as in the ipopt plugin. |

Everything in CasADi's base `nlpsol` option set also applies:
`iteration_callback`, `print_time`, `bound_consistency`, `calc_lam_p`,
`error_on_fail`, and the rest.

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

## Results and statistics

`solver.stats()` carries the usual CasADi keys plus POUNCE's
per-iteration trace:

```python
st = solver.stats()
st["success"]        # bool
st["return_status"]  # 'Solve_Succeeded', 'Maximum_Iterations_Exceeded', …
st["iter_count"]
st["iterations"]     # dict of per-iteration lists:
                     #   inf_pr, inf_du, mu, d_norm, regularization_size,
                     #   obj, alpha_pr, alpha_du, ls_trials
```

The `iterations` dict is the same data POUNCE prints in its iteration
table, so convergence plots need no stdout parsing.

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

## Notebook

[`python/notebooks/35_casadi.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/35_casadi.ipynb)
walks through all of the above end to end with saved outputs — first solve and
convergence plot, `Opti` optimal control, the warm-started MPC loop, the
sensitivity and bilevel example, the nonlinear-variable mask measured at two
problem sizes, and the Ipopt cross-check.

## What is not supported

- **Code generation.** `solver.generate()` is not available, the same as
  for CasADi's Ipopt plugin.
- **Multi-threaded evaluation.** Use one solver object per thread;
  POUNCE's core is single-threaded per problem instance.
- **`var_string_md` / `con_*_md` metadata** is accepted by CasADi and
  dropped rather than forwarded.
- **Integer variables.** POUNCE is a continuous local NLP solver;
  `discrete` is not honoured.

## AMPL fallback

CasADi also ships an `ampl` plugin that writes an `.nl` file and shells
out to a solver binary, and POUNCE reads `.nl`. It is a poor substitute
— CasADi's AMPL interface accepts only SX models with no parameters, and
returns no bound multipliers — so prefer the plugin above. It exists as
a fallback for a model that already fits those limits.
