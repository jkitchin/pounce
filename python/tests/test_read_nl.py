"""Tests for ``pounce.read_nl`` — loading a `.nl` model through pounce's
own Rust reader and evaluating objective / gradient / Hessian and
constraints / Jacobian."""

from pathlib import Path

import numpy as np
import pytest

import pounce

# Benchmark `.nl` fixtures committed at the repo root.
_NL_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "large_scale" / "nl"
_ROSENBROCK = _NL_DIR / "rosenbrock.nl"
_BRATU = _NL_DIR / "bratu.nl"


@pytest.mark.skipif(not _ROSENBROCK.exists(), reason="rosenbrock.nl fixture missing")
def test_read_nl_unconstrained_objective_gradient_hessian():
    p = pounce.read_nl(str(_ROSENBROCK))

    assert p.n == 2000
    assert p.m == 0
    assert p.minimize is True

    x = np.asarray(p.x0, dtype=float)
    assert x.shape == (p.n,)

    # Objective is a finite scalar.
    f = p.objective(x)
    assert np.isfinite(f)

    # Gradient matches central finite differences on a few coordinates.
    g = p.gradient(x)
    assert g.shape == (p.n,)
    h = 1e-6
    for j in (0, 1, 137, p.n - 1):
        xp = x.copy(); xp[j] += h
        xm = x.copy(); xm[j] -= h
        fd = (p.objective(xp) - p.objective(xm)) / (2 * h)
        assert abs(g[j] - fd) <= 1e-4 * (1 + abs(g[j]))

    # Hessian structure and values align in length.
    hr, hc = p.hessian_structure()
    assert hr.shape == hc.shape == (p.nnz_hess,)
    hv = p.hessian(x)
    assert hv.shape == (p.nnz_hess,)
    assert np.all(np.isfinite(hv))
    # Lower triangle only.
    assert np.all(hr >= hc)

    # No constraints → empty constraint vector.
    assert p.constraints(x).shape == (0,)
    assert p.nnz_jac == 0


@pytest.mark.skipif(not _BRATU.exists(), reason="bratu.nl fixture missing")
def test_read_nl_constrained_jacobian():
    p = pounce.read_nl(str(_BRATU))

    assert p.n > 0
    assert p.m > 0

    x = np.asarray(p.x0, dtype=float)
    c = p.constraints(x)
    assert c.shape == (p.m,)
    assert np.all(np.isfinite(c))

    jr, jc = p.jacobian_structure()
    assert jr.shape == jc.shape == (p.nnz_jac,)
    # Indices are within bounds (0-based COO).
    assert jr.min() >= 0 and jr.max() < p.m
    assert jc.min() >= 0 and jc.max() < p.n

    jv = p.jacobian(x)
    assert jv.shape == (p.nnz_jac,)
    assert np.all(np.isfinite(jv))


@pytest.mark.skipif(not _ROSENBROCK.exists(), reason="rosenbrock.nl fixture missing")
def test_read_nl_accepts_lists_and_reports_bounds():
    p = pounce.read_nl(str(_ROSENBROCK))
    # Objective accepts a plain Python list, not just a NumPy array.
    f_list = p.objective(list(p.x0))
    f_arr = p.objective(np.asarray(p.x0, dtype=float))
    assert f_list == pytest.approx(f_arr)

    # Bounds getters return arrays of the right length.
    assert np.asarray(p.x_l).shape == (p.n,)
    assert np.asarray(p.x_u).shape == (p.n,)

    # Wrong-length input is rejected.
    with pytest.raises(ValueError):
        p.objective(np.zeros(p.n + 1))


def test_read_nl_missing_file_raises():
    with pytest.raises(ValueError):
        pounce.read_nl("/no/such/model.nl")


# A `.nl` that references AMPL imported (external) functions whose `$AMPLFUNC`
# library can't be resolved.
_EXTERNAL_NL = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "pounce-cli"
    / "tests"
    / "fixtures_issue_49"
    / "idaes_helmholtz.nl"
)


@pytest.mark.skipif(not _EXTERNAL_NL.exists(), reason="idaes_helmholtz.nl fixture missing")
def test_read_nl_unresolvable_external_raises_catchable_error(monkeypatch):
    """Regression: a model naming an AMPL external function with no resolvable
    ``$AMPLFUNC`` library must raise a *catchable* Python exception, not a
    ``pyo3_runtime.PanicException`` (a ``BaseException`` that ``except
    Exception`` would miss). Before the fix, ``NlTnlp::new`` panicked across
    the pyo3 boundary on this path."""
    monkeypatch.delenv("AMPLFUNC", raising=False)
    # `Exception` (not just `ValueError`) is the load-bearing assertion: a
    # PanicException is NOT an `Exception`, so this would not catch it.
    with pytest.raises(Exception) as exc_info:
        pounce.read_nl(str(_EXTERNAL_NL))
    assert isinstance(exc_info.value, Exception)
    assert "PanicException" not in type(exc_info.value).__name__
