"""`pounce --minima` and `pounce.find_minima` run the same search (gh#966).

They were separate implementations of the same six methods, coupled by
comments rather than by code, and they drifted: gh#983's `kkt_tol` veto
landed in Python only, so the CLI kept returning points the solver labelled
a success at a non-stationary point. Both now call
`pounce_minima::find_minima`. This file is the guard that they stay one
search, run through the real binary and the real extension module -- a
frontend that built the problem, the options or the seed differently would
show up here, which no test of the shared crate can see.

`test_same_search_on_both` is the strong form: same model, method and seed,
so the same starts, and the two must agree on status, solve count and every
value. The rest predate the shared core (when the two sides drew starts from
different Sobol' generators and only start-independent quantities could be
compared) and stay as named branch checks:

* ``camel`` -- exactly six minima, all of them found: the sorted values agree.
* ``bowl`` -- one minimum: both report ``converged`` with that one point.
* ``non_stationary`` -- the solver's acceptable-level test is opened so wide
  that every solve stops at iteration 0 and labels its own start
  ``Solved_To_Acceptable_Level``, with an unscaled residual ~1e4 times the
  ``kkt_tol`` bound. That is the gh#983 defect class made deterministic: every
  candidate on both sides is vetoed, whatever the start. Measured on the
  binary before the port (62166c0): the CLI accepted 4 such starting points
  as minima, where ``find_minima`` returned none.
* ``wavy_1e12`` -- gradients at 1e12: the bound is relative, so nothing is
  vetoed on either side (the absolute bound rejected 11 of 12 solves).
"""

import json
import os
import shutil
import subprocess
import warnings

import numpy as np
import pytest

import pounce

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _works(path):
    # `shutil.which("pounce")` can find the wheel's console-script shim with
    # no bundled binary behind it; only a binary that runs counts.
    if path is None:
        return False
    try:
        proc = subprocess.run([path, "--version"], capture_output=True,
                              text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _binary():
    # POUNCE_CLI selects a binary explicitly, e.g. a baseline build when
    # checking that this file goes red on a known divergence.
    override = os.environ.get("POUNCE_CLI")
    if override:
        return override
    for local in (os.path.join(_ROOT, "target", "release", "pounce"),
                  os.path.join(_ROOT, "python", "pounce", "bin", "pounce")):
        if os.path.exists(local):
            return local
    return shutil.which("pounce")


_CLI = _binary() if _works(_binary()) else None
if os.environ.get("POUNCE_CLI") and _CLI is None:
    # Named explicitly (CI's wheel-smoke job does): a binary that does not run
    # is a broken setup, and skipping would turn the guard off silently.
    raise RuntimeError(f"POUNCE_CLI={os.environ['POUNCE_CLI']!r} does not run")

pytestmark = pytest.mark.skipif(
    _CLI is None,
    reason="needs a *working* pounce binary (a wheel built without "
           "pounce/bin/ staged leaves only a shim that exits 1)",
)

_HEADER = ["g3 1 1 0", " 2 0 1 0 0", " 0 1", " 0 0", " 0 2 0",
           " 0 0 0 1", " 0 0 0 0 0", " 0 2", " 0 0", " 0 0 0 0 0", "O0 0"]


def _nl(expr, x0, box):
    tail = ["x2", f"0 {x0[0]!r}", f"1 {x0[1]!r}", "b"]
    tail += [f"0 {lo!r} {hi!r}" for lo, hi in box]
    tail += ["k1", "0", "G0 2", "0 0", "1 0"]
    return "\n".join(_HEADER + expr + tail) + "\n"


# ---- fixtures: the same model as `.nl` text and as Python callables --------

def _camel():
    # 4x^2 - 2.1x^4 + x^6/3 + xy - 4y^2 + 4y^4; the expression string of
    # crates/pounce-cli/tests/issue989_mlsl_cli.rs.
    expr = ("o54 6 o2 n4 o5 v0 n2 o2 n-2.1 o5 v0 n4 o3 o5 v0 n6 n3 "
            "o2 v0 v1 o2 n-4 o5 v1 n2 o2 n4 o5 v1 n4").split()

    def f(z):
        x, y = z
        return 4 * x**2 - 2.1 * x**4 + x**6 / 3 + x * y - 4 * y**2 + 4 * y**4

    def g(z):
        x, y = z
        return np.array([8 * x - 8.4 * x**3 + 2 * x**5 + y,
                         x - 8 * y + 16 * y**3])

    def h(z):
        x, y = z
        return np.array([[8 - 25.2 * x**2 + 10 * x**4, 1.0],
                         [1.0, -8 + 48 * y**2]])

    return expr, f, g, h, (0.5, 0.5), [(-3.0, 3.0), (-2.0, 2.0)]


def _bowl():
    # (x - 1)^2 + (y + 0.5)^2: a single minimum, interior.
    expr = "o0 o5 o0 v0 n-1 n2 o5 o0 v1 n0.5 n2".split()

    def f(z):
        return (z[0] - 1) ** 2 + (z[1] + 0.5) ** 2

    def g(z):
        return np.array([2 * (z[0] - 1), 2 * (z[1] + 0.5)])

    def h(z):
        return 2 * np.eye(2)

    return expr, f, g, h, (0.0, 0.0), [(-2.0, 2.0), (-2.0, 2.0)]


def _wavy(S):
    # S * sum(sin(3x) + 0.1 x^2), the model of test_issue_983_review.py.
    expr = ["o54", "4"]
    for i in range(2):
        expr += ["o2", f"n{S!r}", "o41", "o2", "n3", f"v{i}"]
        expr += ["o2", f"n{0.1 * S!r}", "o5", f"v{i}", "n2"]

    def f(x):
        return S * float(np.sum(np.sin(3 * x) + 0.1 * x**2))

    def g(x):
        return S * (3 * np.cos(3 * x) + 0.2 * x)

    def h(x):
        return np.diag(S * (-9 * np.sin(3 * x) + 0.2))

    return expr, f, g, h, (0.3, -0.2), [(-5.0, 5.0), (-5.0, 5.0)]


# The acceptable-level test opened wide enough that every solve stops at
# iteration 0 and returns its own start as Solved_To_Acceptable_Level.
_ACCEPT_ANYTHING = {
    "acceptable_tol": 100.0,
    "acceptable_compl_inf_tol": 100.0,
    "acceptable_obj_change_tol": 1e20,
    "acceptable_iter": 1,
}


def _both(model, tmp_path, method, *, n_minima, max_solves, patience=8,
          dedup=1e-4, seed=0, options=None):
    expr, f, g, h, x0, box = model
    options = options or {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        py = pounce.find_minima(
            f, np.array(x0, dtype=float), method=method, jac=g, hess=h,
            bounds=box, n_minima=n_minima, max_solves=max_solves,
            patience=patience, dedup=dedup, seed=seed, options=options,
        )
    py_warned = any("kkt_tol" in str(w.message) for w in caught)

    nl = tmp_path / "model.nl"
    nl.write_text(_nl(expr, x0, box))
    report = tmp_path / "report.json"
    proc = subprocess.run(
        [_CLI, str(nl), "--minima", method,
         "--n-minima", str(n_minima), "--max-solves", str(max_solves),
         "--patience", str(patience), "--dedup", str(dedup),
         "--seed", str(seed), "--json-output", str(report), "--no-sol"]
        + [f"{k}={v}" for k, v in options.items()],
        capture_output=True, text=True, timeout=600,
    )
    assert report.exists(), f"no report:\n{proc.stdout}\n{proc.stderr}"
    cli = json.loads(report.read_text())["minima"]
    cli["warned"] = "--kkt-tol" in proc.stderr
    return py, py_warned, cli, g


def _assert_all_six(py, cli):
    assert py.status == cli["status"] == "target_reached", (py.status, cli)
    np.testing.assert_allclose(sorted(py.values), sorted(cli["values"]),
                               atol=1e-6)
    # `.get`: a binary that predates the veto has no counter, and that is
    # not what this test is about.
    assert py.n_kkt_rejected == cli.get("n_kkt_rejected", 0) == 0


# deflation was a strict xfail here until the CLI's penalty solves took the
# exact augmented Hessian on an unconstrained problem, as find_minima(hess=)
# always did: measured over seeds 0-5 at this budget, all six minima in 6/6
# runs for Python with hess=, 3/6 without it, and 2/6 for the CLI (gh#966).
@pytest.mark.parametrize("method", ["multistart", "mlsl", "deflation"])
def test_camel_both_find_the_same_six_minima(tmp_path, method):
    py, _, cli, _ = _both(_camel(), tmp_path, method, n_minima=6,
                          max_solves=80, patience=20, dedup=1e-3)
    _assert_all_six(py, cli)


@pytest.mark.parametrize("method", ["multistart", "deflation"])
def test_bowl_both_converge_on_the_single_minimum(tmp_path, method):
    py, _, cli, _ = _both(_bowl(), tmp_path, method, n_minima=3,
                          max_solves=30, patience=4, dedup=1e-3)
    assert py.status == cli["status"] == "converged", (py.status, cli)
    assert len(py) == cli["n_minima"] == 1
    np.testing.assert_allclose(py.x, [1.0, -0.5], atol=1e-6)
    np.testing.assert_allclose(cli["minima"][0]["x"], [1.0, -0.5], atol=1e-6)


@pytest.mark.parametrize("method", ["multistart", "deflation", "mlsl"])
def test_a_solver_success_at_a_non_stationary_point_is_vetoed_on_both(
        tmp_path, method):
    py, py_warned, cli, g = _both(_wavy(1.0), tmp_path, method, n_minima=4,
                                  max_solves=12, seed=1,
                                  options=_ACCEPT_ANYTHING)
    # The premise: what the baseline CLI returned here were starting points,
    # with |grad f| ~ 2-3 against a bound of ~3e-4.
    leaked = [float(np.abs(g(np.array(m["x"]))).max()) for m in cli["minima"]]
    assert cli["n_minima"] == 0, f"CLI accepted non-stationary points: {leaked}"
    assert len(py) == 0
    assert py.status == cli["status"]
    assert py.n_kkt_rejected == py.n_solves > 0
    assert cli["n_kkt_rejected"] == cli["n_solves"] > 0
    # An empty result caused by the veto is never silent, on either side.
    assert py_warned and cli["warned"]


def test_the_bound_is_relative_on_both(tmp_path):
    py, _, cli, _ = _both(_wavy(1e12), tmp_path, "multistart", n_minima=4,
                          max_solves=12, seed=1)
    assert py.status == cli["status"] == "target_reached"
    assert len(py) == cli["n_minima"] == 4
    assert py.n_kkt_rejected == cli.get("n_kkt_rejected", 0) == 0


@pytest.mark.parametrize("model", ["camel", "bowl"])
@pytest.mark.parametrize("method", ["multistart", "mlsl", "basinhopping",
                                    "flooding", "deflation", "tunneling"])
def test_same_search_on_both(tmp_path, model, method):
    fixture = {"camel": _camel, "bowl": _bowl}[model]()
    py, _, cli, _ = _both(fixture, tmp_path, method, n_minima=6,
                          max_solves=80, patience=20, dedup=1e-3, seed=0)
    assert py.status == cli["status"]
    assert py.n_solves == cli["n_solves"]
    assert len(py) == cli["n_minima"]
    np.testing.assert_allclose(py.values, cli["values"], atol=1e-9, rtol=0)
