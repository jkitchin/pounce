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
    Suffix,
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


def test_bound_multipliers_populate_ipopt_zL_zU(solver):
    """`model.ipopt_zL_out` / `ipopt_zU_out` are populated with the reduced
    costs, matching Ipopt's convention (issue #296).

    min (x-3)^2 + (y+2)^2  s.t.  0<=x<=1, -1<=y<=1  ->  x*=1, y*=-1.
    The unconstrained minimum (3, -2) is outside the box, so both bounds bind:
      * x=1 upper-active:  d f/d x = 2(x-3) = -4  =>  ipopt_zU_out[x] = -4
      * y=-1 lower-active:  d f/d y = 2(y+2) = +2  =>  ipopt_zL_out[y] = +2
    Ipopt writes zL_out = +z_l (>= 0) and zU_out = -z_u (<= 0); before #296
    pounce wrote no suffix blocks at all and these came back as ``None``.
    """
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1), initialize=0.5)
    m.y = Var(bounds=(-1, 1), initialize=0.0)
    m.obj = Objective(expr=(m.x - 3) ** 2 + (m.y + 2) ** 2)
    m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)

    solver.solve(m)

    assert value(m.x) == pytest.approx(1.0, abs=1e-5)
    assert value(m.y) == pytest.approx(-1.0, abs=1e-5)

    # The blocks must actually be populated (the #296 gap was silent Nones).
    zU_x = m.ipopt_zU_out.get(m.x)
    zL_y = m.ipopt_zL_out.get(m.y)
    assert zU_x is not None, "ipopt_zU_out[x] must be populated"
    assert zL_y is not None, "ipopt_zL_out[y] must be populated"

    # Analytic reduced costs, with Ipopt's sign convention.
    assert zU_x == pytest.approx(-4.0, abs=1e-4)
    assert zL_y == pytest.approx(2.0, abs=1e-4)


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
