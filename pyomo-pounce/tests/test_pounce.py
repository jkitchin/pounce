"""Smoke tests for the pyomo-pounce solver plugin.

Run with `pytest`. The `pounce` binary must be on PATH (or bundled).
"""

import os
import stat

import pytest

import pyomo_pounce  # noqa: F401  (registers 'pounce' with SolverFactory)
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    SolverFactory,
    Var,
    value,
)


@pytest.fixture(scope="module")
def solver():
    s = SolverFactory("pounce")
    if not s.available(exception_flag=False):
        pytest.skip("pounce binary not found on PATH")
    return s


def test_registered():
    assert SolverFactory("pounce") is not None


def _make_executable(path):
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


# --------------------------------------------------------------------------
# issue M33 — the plugin must discover the binary bundled in the installed
# `pounce-solver` wheel (a deterministic path, independent of PATH), not only
# `shutil.which("pounce")`. Pre-fix, a non-activated-venv run (cron, IDE
# runner, Jupyter kernel) with the venv's bin off PATH reported the solver
# unavailable even though the bundled binary was present. (Code review M33.)
# --------------------------------------------------------------------------


def test_default_executable_prefers_bundled(monkeypatch, tmp_path):
    # A bundled binary exists at the deterministic wheel path, but PATH does
    # NOT contain `pounce`. The plugin must still resolve the bundled binary.
    # Needs the `pounce-solver` package (for `pounce._cli`); skip where only
    # the CLI binary is on PATH (e.g. CI's pyomo-pounce smoke job).
    cli = pytest.importorskip("pounce._cli")

    bundled = tmp_path / "bin" / "pounce"
    bundled.parent.mkdir()
    _make_executable(bundled)
    monkeypatch.setattr(cli, "_bundled_binary", lambda: bundled)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # no `pounce` here

    exe = SolverFactory("pounce")._default_executable()
    assert exe == str(bundled)


def test_default_executable_falls_back_to_path(monkeypatch, tmp_path):
    # No bundled binary (system install / local cargo dev build): fall back to
    # whatever `pounce` is on PATH.
    cli = pytest.importorskip("pounce._cli")

    monkeypatch.setattr(cli, "_bundled_binary", lambda: tmp_path / "absent" / "pounce")
    shim_dir = tmp_path / "pathbin"
    shim_dir.mkdir()
    shim = _make_executable(shim_dir / "pounce")
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}/usr/bin:/bin")

    exe = SolverFactory("pounce")._default_executable()
    assert exe == shim


def test_default_executable_none_when_nowhere(monkeypatch, tmp_path):
    # Neither bundled nor on PATH → None (the honest "unavailable" signal).
    cli = pytest.importorskip("pounce._cli")

    monkeypatch.setattr(cli, "_bundled_binary", lambda: tmp_path / "absent" / "pounce")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    assert SolverFactory("pounce")._default_executable() is None


def test_unconstrained(solver):
    """min (x - 2)^2  ->  x* = 2."""
    m = ConcreteModel()
    m.x = Var(initialize=0.5)
    m.obj = Objective(expr=(m.x - 2) ** 2)

    solver.solve(m)

    assert value(m.x) == pytest.approx(2.0, abs=1e-6)


def test_constrained(solver):
    """min (x-2)^2 + (y-3)^2  s.t. x + y <= 4  ->  (1.5, 2.5)."""
    m = ConcreteModel()
    m.x = Var(domain=NonNegativeReals, initialize=1.0)
    m.y = Var(domain=NonNegativeReals, initialize=1.0)
    m.obj = Objective(expr=(m.x - 2) ** 2 + (m.y - 3) ** 2)
    m.budget = Constraint(expr=m.x + m.y <= 4)

    solver.solve(m)

    assert value(m.x) == pytest.approx(1.5, abs=1e-5)
    assert value(m.y) == pytest.approx(2.5, abs=1e-5)


def test_options_forwarded(solver):
    """`max_iter` is forwarded; 0 iterations cannot reach optimality."""
    m = ConcreteModel()
    m.x = Var(initialize=0.5)
    m.obj = Objective(expr=(m.x - 2) ** 2)

    solver.options["max_iter"] = 0
    try:
        result = solver.solve(m)
    finally:
        del solver.options["max_iter"]

    assert str(result.solver.termination_condition) != "optimal"
