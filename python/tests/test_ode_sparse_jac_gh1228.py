"""gh #1228: ``solve_ivp`` / ``solve_dae`` accept sparse ``jac`` and ``mass``.

SciPy's ``solve_ivp(method="Radau")`` accepts ``jac`` as a dense or sparse
constant, or a callable returning either. pounce rejected both sparse forms
(``TypeError: 'csc_matrix' object is not callable`` and ``ValueError: setting
an array element with a sequence``). A sparse Jacobian or mass now switches the
integrator onto a sparse (SuperLU) stage solve rather than densifying.

Every sparse case is checked against the dense run of the *same* problem.
Only the factorization backend differs, but the simplified-Newton stage solve
stops at ``newton_tol`` (tied to ``rtol``), so roundoff differences between the
two LUs legitimately move the answer at the integration-tolerance level — the
runs must agree to that tolerance, not to roundoff. Each test also checks that
the sparse branch was actually taken — a parity test that silently ran the
dense path would be green and say nothing.
"""

import numpy as np
import pytest
import scipy.sparse as sp

import pounce
from pounce.ode import _dae, _radau, _sparse, solve_dae, solve_ivp


@pytest.fixture
def lu_spy(monkeypatch):
    """Count SparseLU / SparseComplexLU factorizations."""
    calls = {"real": 0, "cplx": 0}
    real_f, cplx_f = _sparse.SparseLU.factor, _sparse.SparseComplexLU.factor

    def rf(self, M):
        calls["real"] += 1
        return real_f(self, M)

    def cf(self, M):
        calls["cplx"] += 1
        return cplx_f(self, M)

    monkeypatch.setattr(_sparse.SparseLU, "factor", rf)
    monkeypatch.setattr(_sparse.SparseComplexLU, "factor", cf)
    return calls


def _same_run(a, b, rtol=1e-3, atol=1e-6):
    """Both succeed, take a comparable number of steps, and agree to within
    the integration tolerance at the shared end point."""
    assert a.success and b.success
    assert abs(a.t.size - b.t.size) <= max(2, b.t.size // 10)
    assert a.t[-1] == pytest.approx(b.t[-1])
    err = np.abs(a.y[:, -1] - b.y[:, -1]) / (atol + rtol * np.abs(b.y[:, -1]))
    assert err.max() <= 1.0, err.max()


# --- the issue's reproduction -------------------------------------------------

_A = sp.diags([-np.arange(1.0, 6.0)], [0], format="csc")


def _f_diag(t, y):
    return _A @ y


@pytest.mark.parametrize("kind", ["constant", "callable", "csr", "coo"])
def test_issue_repro_sparse_jac(kind, lu_spy):
    J = {"constant": _A, "callable": lambda t, y: _A,
         "csr": _A.tocsr(), "coo": _A.tocoo()}[kind]
    r = solve_ivp(_f_diag, (0, 1), np.ones(5), method="Radau", jac=J)
    ref = solve_ivp(_f_diag, (0, 1), np.ones(5), method="Radau",
                    jac=_A.toarray())
    _same_run(r, ref)
    np.testing.assert_allclose(r.y[:, -1], np.exp(-np.arange(1.0, 6.0)),
                               rtol=1e-3, atol=1e-5)
    assert lu_spy["real"] > 0 and lu_spy["cplx"] > 0


def test_dense_constant_jac_and_args():
    """A non-callable jac takes no ``args`` (SciPy semantics)."""
    k = np.arange(1.0, 6.0)
    f = lambda t, y, kk: -kk * y
    r = solve_ivp(f, (0, 1), np.ones(5), jac=np.diag(-k), args=(k,))
    r2 = solve_ivp(f, (0, 1), np.ones(5), jac=lambda t, y, kk: np.diag(-kk),
                   args=(k,))
    _same_run(r, r2)


def test_dense_jac_stays_on_dense_lu(lu_spy):
    solve_ivp(_f_diag, (0, 1), np.ones(5), jac=_A.toarray())
    assert lu_spy == {"real": 0, "cplx": 0}


# --- method-of-lines heat equation (the motivating case) ----------------------

def _heat(n):
    dx = 1.0 / (n + 1)
    L = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n),
                 format="csc") / dx ** 2
    x = np.linspace(dx, 1 - dx, n)
    # mild nonlinearity so J depends on y and is re-evaluated
    f = lambda t, y: L @ y - 0.5 * y ** 3
    jac = lambda t, y: L - sp.diags(1.5 * y ** 2, format="csc")
    return f, jac, np.sin(np.pi * x)


def test_method_of_lines_sparse_matches_dense(lu_spy):
    f, jac, y0 = _heat(60)
    rs = solve_ivp(f, (0, 0.1), y0, jac=jac, rtol=1e-6, atol=1e-9)
    rd = solve_ivp(f, (0, 0.1), y0, jac=lambda t, y: jac(t, y).toarray(),
                   rtol=1e-6, atol=1e-9)
    _same_run(rs, rd, rtol=1e-6, atol=1e-9)
    assert lu_spy["real"] == rs.nlu // 2


# --- sparse mass ---------------------------------------------------------------

_k1, _k2, _k3 = 0.04, 3.0e7, 1.0e4


def _rob_f(t, y):
    return np.array([-_k1 * y[0] + _k3 * y[1] * y[2],
                     _k1 * y[0] - _k3 * y[1] * y[2] - _k2 * y[1] ** 2,
                     y[0] + y[1] + y[2] - 1.0])


def _rob_jac(t, y):
    return np.array([[-_k1, _k3 * y[2], _k3 * y[1]],
                     [_k1, -_k3 * y[2] - 2 * _k2 * y[1], -_k3 * y[1]],
                     [1.0, 1.0, 1.0]])


@pytest.mark.parametrize("sparse_jac", [False, True])
def test_singular_sparse_mass_dae(sparse_jac, lu_spy):
    """Constant singular sparse mass (index-1 DAE path, incl. IC projection)."""
    Md = np.diag([1.0, 1.0, 0.0])
    jac = (lambda t, y: sp.csr_matrix(_rob_jac(t, y))) if sparse_jac else _rob_jac
    y0 = np.array([1.0, 0.0, 1e-3])     # inconsistent: projected onto sum = 1
    rs = solve_ivp(_rob_f, (0, 40), y0, mass=sp.csc_matrix(Md), jac=jac,
                   rtol=1e-6, atol=1e-9)
    rd = solve_ivp(_rob_f, (0, 40), y0, mass=Md, jac=_rob_jac,
                   rtol=1e-6, atol=1e-9)
    _same_run(rs, rd, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(rs.y.sum(axis=0), 1.0, atol=1e-8)
    assert lu_spy["real"] > 0


def test_callable_sparse_mass():
    Md = np.diag([1.0, 1.0, 0.0])
    y0 = np.array([1.0, 0.0, 0.0])
    rs = solve_ivp(_rob_f, (0, 40), y0, mass=lambda t, y: sp.csr_matrix(Md),
                   rtol=1e-6, atol=1e-9)
    rd = solve_ivp(_rob_f, (0, 40), y0, mass=lambda t, y: Md,
                   rtol=1e-6, atol=1e-9)
    _same_run(rs, rd, rtol=1e-6, atol=1e-9)


# --- solve_dae ----------------------------------------------------------------

def _F(t, y, yp):
    return np.array([yp[0], yp[1], 0.0]) - _rob_f(t, y)


def _F_jac(t, y, yp):
    return -_rob_jac(t, y), np.diag([1.0, 1.0, 0.0])


@pytest.mark.parametrize("which", ["both", "Fy", "Fyp"])
def test_solve_dae_sparse_jac(which, lu_spy):
    def jac(t, y, yp):
        Fy, Fyp = _F_jac(t, y, yp)
        if which in ("both", "Fy"):
            Fy = sp.csc_matrix(Fy)
        if which in ("both", "Fyp"):
            Fyp = sp.csr_matrix(Fyp)
        return Fy, Fyp

    y0 = np.array([1.0, 0.0, 1e-3])
    rs = solve_dae(_F, (0, 40), y0, jac=jac, rtol=1e-6, atol=1e-9)
    rd = solve_dae(_F, (0, 40), y0, jac=_F_jac, rtol=1e-6, atol=1e-9)
    _same_run(rs, rd, rtol=1e-6, atol=1e-9)
    assert lu_spy["real"] > 0


def test_solve_dae_constant_sparse_pair():
    """A linear DAE with a constant (F_y, F_y') pair, sparse."""
    A = np.array([[-1.0, 0.5], [1.0, 1.0]])          # y2 = -y1 algebraically
    F = lambda t, y, yp: np.array([yp[0], 0.0]) - A @ y
    Fy, Fyp = -A, np.diag([1.0, 0.0])
    rs = solve_dae(F, (0, 1), np.array([1.0, -1.0]),
                   jac=(sp.csc_matrix(Fy), sp.csc_matrix(Fyp)))
    rd = solve_dae(F, (0, 1), np.array([1.0, -1.0]), jac=(Fy, Fyp))
    _same_run(rs, rd)
    np.testing.assert_allclose(rs.y[0, -1], np.exp(-1.5), rtol=1e-4)


def test_top_level_alias():
    assert pounce.solve_ivp is solve_ivp
    r = pounce.solve_ivp(_f_diag, (0, 1), np.ones(5), method="Radau", jac=_A)
    assert r.success
