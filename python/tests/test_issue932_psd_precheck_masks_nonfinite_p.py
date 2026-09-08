"""gh #932: the PSD pre-check must not mask the ``P`` non-finite guard.

Sibling of gh #862, same root cause — the PSD pre-check runs *before*
``_validate`` on every entry point that checks the Hessian before it builds —
on a different trigger. #862 was the *shape* case; this is the *non-finite*
case.

``_validate`` already carries the precise message for a ``P`` holding
``NaN``/``Inf``, and every sibling argument (``c``, ``A``, ``b``, ``h``,
``lb``, ``ub``) produces it on the default path. ``P`` did not: its values
reached ``np.linalg.eigvalsh`` inside ``_min_eig_lower_coo`` first, and what
came back depended on ``n``:

* ``n >= 3`` -> ``numpy.linalg.LinAlgError: Eigenvalues did not converge``,
  a raw exception out of numpy internals naming nothing the caller passed;
* ``n == 2`` -> the iteration converges to ``nan``, ``nan >= tol`` is False,
  and the *indefinite* error is raised instead: "P is not positive
  semidefinite (min eigenvalue nan) ... To solve an indefinite QP, pass
  method='active-set'". ``P`` is not indefinite, it is non-finite, and the
  remedy named does not help.

So one malformed input had three diagnoses, chosen by ``n`` and by
``check_psd``. The size split is why every case below is parametrized over
``n`` in ``{2, 3, 4, 5}``: pinning one size records the other's symptom as
absent.

The fix runs the finite check inside ``_psd_verdict`` before it reads ``P``,
in ``_validate``'s own order (finite, then shape), so the two spellings of
``check_psd`` agree even on an input that is malformed both ways. A second
check sits under it, inside ``_psd_verdict_coo`` itself, because reproducing
this turned up a symptom the report does not name: on a non-finite ``P`` the
verdict function did not only raise the wrong exception, it *returned* --
``(True, 0.0)`` for ``diag(1, 1, 1, nan)``, the guard affirming the PSD
precondition about a matrix that is not finite.

Mutation table (this file, 161 cases):

===================================  =========================================
removed                              fails
===================================  =========================================
``_reject_nonfinite`` in             4 -- only the doubly-malformed ordering
``_psd_verdict``                     case; the backstop catches the rest,
                                     with the same message
``_reject_nonfinite`` in             3 -- the verdict-level cases
``_psd_verdict_coo``
both (the parent's behaviour)        92
===================================  =========================================

Neither guard alone covers the site, which is why both are here: the frontend
check owns the *order* (and reads the dense ``P``, so it sees an entry the
lower-triangle filter drops), the backstop owns the *verdict*.
"""

import numpy as np
import pytest

from pounce import (
    QpFactorization,
    QpSensitivity,
    solve_qp,
    solve_qp_batch,
    solve_qp_multi_rhs,
    solve_socp,
)

EXPECTED = r"`P` contains NaN or Inf"

# Every way a non-finite entry can sit in `P`, including one the pre-check
# never reads: the guard takes the *lower* triangle, so `nan-upper` is a
# control -- it reached `_validate` intact before the fix and must keep the
# same message after it, the way gh #862's `np.eye(7, 5)` did on shape.
PLACEMENTS = ("inf-diag", "nan-diag", "inf-lower", "nan-upper")


def _bad_P(placement, n):
    if placement == "inf-diag":
        P = np.eye(n)
        P[1 % n, 1 % n] = np.inf
        return P
    if placement == "nan-diag":
        P = np.eye(n)
        P[n - 1, n - 1] = np.nan
        return P
    P = np.eye(n) * 2.0
    val = np.inf if placement == "inf-lower" else np.nan
    if placement == "inf-lower":
        P[n - 1, 0] = val
        P[0, n - 1] = val
    else:  # nan-upper: written to the upper triangle only
        P[0, n - 1] = val
    return P


SIZES = [2, 3, 4, 5]


def _entry_points(P, n):
    """Every public call that runs the PSD pre-check before ``_build``.

    Same seven as gh #862's test, and for the same reason: the pre-check is
    shared, so a fix that lands on one entry point and not the others is the
    inconsistency the issue is about, one level up.
    """
    c = np.ones(n)
    return {
        "solve_qp": lambda **kw: solve_qp(P=P, c=c, **kw),
        "solve_qp/active-set": lambda **kw: solve_qp(
            P=P, c=c, method="active-set", **kw
        ),
        "solve_qp_batch": lambda **kw: solve_qp_batch([dict(P=P, c=c)], **kw),
        "solve_qp_multi_rhs": lambda **kw: solve_qp_multi_rhs(
            P=P, cs=[c, 2.0 * c], **kw
        ),
        "solve_socp": lambda **kw: solve_socp(
            P=P,
            c=c,
            cones=[("nonneg", 1)],
            G=np.ones((1, n)),
            h=np.array([1.0]),
            **kw,
        ),
        "QpFactorization": lambda **kw: QpFactorization(P=P, c=c, **kw),
        "QpSensitivity": lambda **kw: QpSensitivity(P=P, c=c, **kw),
    }


@pytest.mark.parametrize("name", sorted(_entry_points(np.eye(2), 2)))
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("placement", sorted(PLACEMENTS))
def test_nonfinite_P_raises_the_written_ValueError(placement, n, name):
    call = _entry_points(_bad_P(placement, n), n)[name]
    with pytest.raises(ValueError, match=EXPECTED):
        call()
    # `LinAlgError` *is* a `ValueError` subclass, so the `raises` above does
    # not by itself reject the n >= 3 regression -- only the `match` does.
    # Assert the type as well so a failure says which symptom came back.
    try:
        call()
    except ValueError as exc:  # pragma: no branch - it always raises
        assert not isinstance(exc, np.linalg.LinAlgError), (
            f"{name}: PSD pre-check raised a raw LinAlgError: {exc}"
        )
        assert "positive semidefinite" not in str(exc), (
            f"{name}: non-finite P was diagnosed as indefinite: {exc}"
        )


@pytest.mark.parametrize("name", sorted(_entry_points(np.eye(2), 2)))
@pytest.mark.parametrize("n", SIZES)
def test_the_two_spellings_agree(n, name):
    """``check_psd=False`` skips the pre-check; it must reach the same error."""
    P = _bad_P("inf-diag", n)
    call = _entry_points(P, n)[name]
    with pytest.raises(ValueError, match=EXPECTED) as skipped:
        call(check_psd=False)
    with pytest.raises(ValueError, match=EXPECTED) as checked:
        call()
    assert str(skipped.value) == str(checked.value)


@pytest.mark.parametrize("n", SIZES)
def test_a_sparse_nonfinite_P_is_rejected_too(n):
    """The guard reads a sparse ``P`` through ``.tocoo().data``, not densely."""
    sp = pytest.importorskip("scipy.sparse")
    vals = np.ones(n)
    vals[0] = np.nan
    P = sp.diags(vals).tocsc()
    with pytest.raises(ValueError, match=EXPECTED):
        solve_qp(P=P, c=np.ones(n))


@pytest.mark.parametrize("n", SIZES)
def test_the_sibling_arguments_are_unchanged(n):
    """The siblings already produced their message; keep them pinned so a
    reordering cannot fix ``P`` while regressing them."""
    c = np.ones(n)
    c[0] = np.inf
    with pytest.raises(ValueError, match=r"`c` contains NaN or Inf"):
        solve_qp(P=np.eye(n), c=c)
    h = np.array([np.nan])
    with pytest.raises(ValueError, match=r"`h` contains NaN or Inf"):
        solve_qp(P=np.eye(n), c=np.ones(n), G=np.ones((1, n)), h=h)
    lb = np.full(n, -np.inf)
    lb[0] = np.nan
    with pytest.raises(ValueError, match=r"`lb` contains NaN"):
        solve_qp(P=np.eye(n), c=np.ones(n), lb=lb)


@pytest.mark.parametrize("n", SIZES)
def test_infinite_bounds_still_mean_absent(n):
    """The check that rejects ``Inf`` in ``P`` must not reject it in a bound:
    ``±inf`` is the idiomatic "no bound" and reaches the same code path."""
    res = solve_qp(
        P=np.eye(n),
        c=np.ones(n),
        lb=np.full(n, -np.inf),
        ub=np.full(n, np.inf),
    )
    assert res.status == "optimal"
    assert np.allclose(res.x, -np.ones(n), atol=1e-6)


def test_the_guard_was_placed_before_the_check_not_instead_of_it():
    """An indefinite but *finite* ``P`` is still refused, with the indefinite
    message -- the pre-check still runs, it just no longer speaks for inputs
    that are not about definiteness at all."""
    with pytest.raises(ValueError, match=r"not positive semidefinite"):
        solve_qp(P=np.diag([1.0, -1.0]), c=np.ones(2))
    # ...and a finite PSD `P` still solves.
    res = solve_qp(P=np.eye(3), c=np.ones(3))
    assert res.status == "optimal"


@pytest.mark.parametrize("n", SIZES)
def test_a_P_that_is_both_mis_shaped_and_nonfinite_agrees_with_validate(n):
    """Both defects at once: the two spellings must still name the same one.

    ``_validate`` checks finiteness before shape, so it reports the
    non-finite entry; the pre-check now uses that order rather than its own.
    A reordering here is invisible to every other case in this file, which is
    why the case is written down.
    """
    P = np.eye(n + 2) * 2.0
    P[0, 0] = np.nan
    c = np.ones(n)
    with pytest.raises(ValueError, match=EXPECTED) as skipped:
        solve_qp(P=P, c=c, check_psd=False)
    with pytest.raises(ValueError, match=EXPECTED) as checked:
        solve_qp(P=P, c=c)
    assert str(skipped.value) == str(checked.value)


@pytest.mark.parametrize(
    "vals,parent",
    [
        ([1.0, 1.0, 1.0, np.nan], "(True, 0.0)"),
        ([np.nan, 1.0, 1.0, 1.0], "(False, 0.0)"),
        ([1.0, np.inf, 1.0, 1.0], "LinAlgError"),
    ],
    ids=["nan-last", "nan-first", "inf"],
)
def test_the_verdict_itself_never_answers_about_a_nonfinite_P(vals, parent):
    """``_psd_verdict_coo`` is the backstop under the frontend checks.

    It is here because the issue's report understates the defect: the guard
    did not only raise the wrong exception, it *returned* on a non-finite
    ``P``, and what it returned depended on where the bad entry sat. The
    ``parent`` column is what each triplet produced on the parent commit --
    ``(True, 0.0)`` is the guard affirming the PSD precondition about a
    matrix that is not finite, and ``(False, 0.0)`` is an indefinite verdict
    whose own ``lam_min`` says the opposite. ``max(abs(v) for v in pv)`` is
    order-dependent under ``NaN``, which is how one kind of input reached
    three unrelated outcomes.

    Anything that is not a raise here is a regression, including a plausible
    one: ``(False, nan)`` would still be the guard answering a question it
    cannot answer.
    """
    from pounce.qp import _psd_verdict_coo

    n = 4
    idx = list(range(n))
    with pytest.raises(ValueError, match=EXPECTED):
        _psd_verdict_coo(idx, idx, vals, n)


def test_the_backstop_leaves_the_finite_verdicts_alone():
    """The two verdicts the guard exists to give are unchanged."""
    from pounce.qp import _psd_verdict_coo

    idx = [0, 1]
    ok, lam = _psd_verdict_coo(idx, idx, [2.0, 3.0], 2)
    assert ok and lam == pytest.approx(2.0)
    bad, lam = _psd_verdict_coo(idx, idx, [1.0, -1.0], 2)
    assert not bad and lam == pytest.approx(-1.0)
    # An LP (no Hessian entries) is trivially PSD and never reaches the check.
    assert _psd_verdict_coo([], [], [], 3) == (True, 0.0)
