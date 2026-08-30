"""gh #849: the PSD guard must not silently switch off above ``n = 1500``.

``_PSD_CHECK_AUTO_MAX_N = 1500`` made the default ``check_psd=None`` skip the
check entirely for larger problems, so the convex QP interior-point engine --
the *guarded* engine -- returned a silently-wrong ``optimal`` on an indefinite
``P`` at **default settings**, with no option named and no warning::

    P = I with P[0, 0] = -3, c = 0, box [-1, 1];  true infimum -1.5
    n = 1400, check_psd=None  ->  ValueError: P is not positive semidefinite
    n = 1600, check_psd=None  ->  status='optimal', obj = 0.0

The only thing that changed is ``n`` crossing a constant. The oracle is
arithmetic: ``P`` is diagonal, so ``x = (1, 0, ..., 0)`` is feasible and gives
``f = -1.5 < 0``, and no bound is active at the returned ``x = 0``, so the
reduced Hessian is ``P`` and the returned point is a strict saddle.

The trade-off the ceiling encoded was real -- the check was a dense ``O(n^3)``
``eigvalsh`` and a large sparse QP should not silently pay it -- but it paid for
it with a cliff. The check now answers the same question by an **inertia count**
on a sparse factorization (Sylvester's law), which is exact, needs no iteration,
and is *faster* than the dense path it replaces.

Why not Lanczos, which is the obvious alternative: measured on this exact
problem it is both slower and wrong in a way that is hard to notice. A
5000-variable 1-D Laplacian takes 9.3 s to reach its smallest eigenvalue
(a spectrum clustered near zero is Lanczos's worst case) against 0.6 ms for one
factorization, and under any bounded iteration budget it fails to refute
``Laplacian - 4I`` -- a matrix whose eigenvalues are *all* negative -- because
that matrix's extremes are clustered too. A guard that misses a negative
definite Hessian is not a guard.
"""

import numpy as np
import pytest

scipy_sparse = pytest.importorskip("scipy.sparse")
sp = scipy_sparse

from pounce.qp import (  # noqa: E402
    _PSD_CHECK_DENSE_MAX_N,
    _min_eig_lower_coo,
    solve_qp,
)

CAP = _PSD_CHECK_DENSE_MAX_N


def indefinite_box(n, bad=-3.0):
    """``min 1/2 x' P x`` over ``[-1, 1]^n`` with one negative curvature."""
    d = np.ones(n)
    d[0] = bad
    return dict(
        P=sp.diags(d).tocoo(),
        c=np.zeros(n),
        lb=-np.ones(n),
        ub=np.ones(n),
    )


def test_the_reported_pair_either_side_of_the_ceiling():
    """The heart of the report: the same family, one decade of ``n`` apart."""
    for n in (CAP - 100, CAP + 100):
        with pytest.raises(ValueError, match="not positive semidefinite"):
            solve_qp(**indefinite_box(n))


def test_the_default_no_longer_has_a_size_cliff():
    """Swept across the ceiling and well past it. Every one must refuse: the
    exhibited feasible point ``x = (1, 0, ..., 0)`` gives ``f = -1.5`` against
    the ``obj = 0.0`` the engine used to certify."""
    for n in (CAP - 1, CAP, CAP + 1, CAP + 2, 3 * CAP, 20 * CAP):
        with pytest.raises(ValueError, match="not positive semidefinite"):
            solve_qp(**indefinite_box(n))


def test_check_psd_false_still_opts_out_at_every_size():
    """The escape hatch is explicit and stays explicit -- it is the one way to
    get the old behaviour, and asking for it is not the same as being given it
    silently."""
    for n in (CAP - 100, CAP + 100):
        r = solve_qp(check_psd=False, **indefinite_box(n))
        assert r.status == "optimal"


def test_a_large_psd_problem_still_passes_quietly():
    """The other direction, which "always refuse" would break: a big PSD QP
    must solve, with no warning and no rejection. Two shapes -- a diagonal one
    and a 1-D Laplacian, whose spectrum is clustered against zero and is the
    hard case for any iterative method."""
    for n in (CAP + 100, 10 * CAP):
        with pytest.warns(None) if False else _no_warnings():
            r = solve_qp(
                P=sp.eye(n).tocoo(),
                c=np.zeros(n),
                lb=-np.ones(n),
                ub=np.ones(n),
            )
        assert r.status == "optimal"
    n = 4 * CAP
    lap = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)).tocoo()
    with _no_warnings():
        r = solve_qp(P=lap, c=np.zeros(n), lb=-np.ones(n), ub=np.ones(n))
    assert r.status == "optimal"


class _no_warnings:
    """`pytest.warns(None)` is an error on modern pytest; this is the intent."""

    def __enter__(self):
        import warnings

        self._cm = warnings.catch_warnings(record=True)
        self._log = self._cm.__enter__()
        import warnings as w

        w.simplefilter("always")
        return self._log

    def __exit__(self, *exc):
        log = [x for x in self._log if issubclass(x.category, RuntimeWarning)]
        self._cm.__exit__(*exc)
        assert not log, f"unexpected RuntimeWarning(s): {[str(x.message) for x in log]}"
        return False


@pytest.mark.parametrize(
    "name, build",
    [
        ("random symmetric (indefinite)", lambda r, n: _sym(r.standard_normal((n, n)))),
        ("PSD A^T A", lambda r, n: (lambda a: a.T @ a)(r.standard_normal((n, n)))),
        ("identity", lambda r, n: np.eye(n)),
        ("zero", lambda r, n: np.zeros((n, n))),
        ("negative definite", lambda r, n: -np.eye(n)),
        ("rank one", lambda r, n: np.outer(np.ones(n), np.ones(n))),
        ("1D Laplacian (PSD)", lambda r, n: _lap(n)),
        ("1D Laplacian - 4I (indefinite)", lambda r, n: _lap(n) - 4 * np.eye(n)),
        ("diag with one -3", lambda r, n: np.diag([-3.0] + [1.0] * (n - 1))),
        ("diag with one -1e-9 (inside tol)", lambda r, n: np.diag([-1e-9] + [1.0] * (n - 1))),
        ("lam_min = -1e-6 (outside tol)", lambda r, n: _spectrum(r, n, -1e-6)),
        ("lam_min = -1e-10 (inside tol)", lambda r, n: _spectrum(r, n, -1e-10)),
        ("lam_min = +1e-10", lambda r, n: _spectrum(r, n, 1e-10)),
        ("rank deficient (half zero)", lambda r, n: _spectrum_list(r, n, [0.0] * (n // 2) + [1.0] * (n - n // 2))),
        ("scale 1e12, PSD", lambda r, n: 1e12 * np.eye(n)),
        ("scale 1e12, indefinite", lambda r, n: np.diag([-3e12] + [1e12] * (n - 1))),
    ],
)
def test_the_sparse_verdict_agrees_with_the_dense_one(name, build):
    """The claim that makes the cliff removable: the cheap verdict is the same
    verdict. Both paths are run on the *same* matrix at a size where the dense
    one is affordable, so this compares the two answers rather than trusting
    either. The delicate rows are the point -- ``lam_min`` at ``+-1e-10``
    straddles the guard's own tolerance, and the rank-deficient and rank-one
    rows carry rounding-level negative eigenvalues that must **not** be read as
    indefinite."""
    rng = np.random.default_rng(849)
    n = 200
    m = build(rng, n)
    pr, pc, pv = _lower_coo(m)
    scale = max(abs(v) for v in pv) if pv else 0.0
    tol_abs = -1e-8 * max(scale, 1.0)

    dense = _min_eig_lower_coo(pr, pc, pv, n) if pv else 0.0
    want = dense >= tol_abs

    from pounce.qp import _psd_verdict_sparse

    got = _psd_verdict_sparse(pr, pc, pv, n, tol_abs) if pv else (True, 0.0)
    assert got is not None, f"{name}: the sparse path could not decide"
    assert got[0] == want, (
        f"{name}: sparse says psd={got[0]}, dense lam_min={dense:.6e} "
        f"against tol {tol_abs:.3e} says psd={want}"
    )
    if not want:
        # The reported eigenvalue is what the error message prints, to the
        # three digits it prints.
        assert abs(got[1] - dense) <= 1e-3 * max(abs(dense), 1e-300), (
            f"{name}: reported lam_min {got[1]:.6e} against dense {dense:.6e}"
        )


def test_an_undecided_verdict_is_not_a_pass():
    """``None`` from the factorization means *undecided*, and the guard turns it
    into a warning rather than silence. Reading "no check" as "check passed" is
    the whole of gh #849, so the two must stay distinguishable."""
    import warnings

    import pounce.qp as q

    n = CAP + 100
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        verdict = q._psd_verdict(
            sp.eye(n).tocoo(), np.zeros(n), None
        )
    assert verdict is not None and verdict[0], "a clean identity must decide"
    assert not log, "and must not warn"

    # Force the undecided branch and check it warns rather than passing.
    real = q._psd_verdict_coo
    try:
        q._psd_verdict_coo = lambda *a, **k: None
        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter("always")
            assert q._psd_verdict(sp.eye(n).tocoo(), np.zeros(n), None) is None
        assert log and issubclass(log[0].category, RuntimeWarning)
        assert "UNCHECKED" in str(log[0].message)
    finally:
        q._psd_verdict_coo = real


def _sym(a):
    return (a + a.T) / 2


def _lap(n):
    return sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)).toarray()


def _spectrum(rng, n, lo):
    ev = np.linspace(1.0, 50.0, n)
    ev[0] = lo
    return _spectrum_list(rng, n, ev)


def _spectrum_list(rng, n, ev):
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q @ np.diag(np.asarray(ev, dtype=float)) @ q.T


def _lower_coo(m):
    lower = sp.tril(sp.csr_matrix(m)).tocoo()
    return list(lower.row), list(lower.col), list(lower.data)
