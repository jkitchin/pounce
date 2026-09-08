"""gh #932 on the differentiable frontends: a non-finite ``P`` must be
rejected by :mod:`pounce.jax` and :mod:`pounce.torch` too.

The AD half of gh #932, in its own file for the reason gh #874 is in its own
file: the fix for gh #862 landed in ``qp.py`` alone, and the two layers that
run their own ``_guard_psd`` kept the defect. Sibling test file:
``test_issue874_ad_frontends_p_shape.py``, same two frontends, same shape of
argument.

# Why this is not covered by the qp.py file's mutation table

``test_issue932_psd_precheck_masks_nonfinite_p.py`` has two guards under it
and neither one reaches here. The ``_psd_verdict_coo`` backstop reads ``pv``
— the **lower triangle** — so it cannot see an entry ``_to_coo_lower`` drops,
and ``_psd_verdict`` is not on these layers' path at all. Neither is
``_validate``: both frontends build through ``_pounce`` directly, so there is
no later check to catch what the guard lets past. Measured with
``_reject_nonfinite`` deleted from ``_guard_psd``, on ``P = 2·I₄`` with
``P[0, 3] = nan`` (upper triangle only)::

    pounce.qp.solve_qp     -> ValueError: solve_qp: `P` contains NaN or Inf
    pounce.jax.solve_qp    -> NO RAISE, x = [-0.5 -0.5 -0.5 -0.5]
    pounce.torch.solve_qp  -> NO RAISE, x = [-0.5 -0.5 -0.5 -0.5]

That vector is the exact optimum of the matrix **without** the ``NaN`` — a
silently substituted model, not noise, and on a differentiable layer whose
``_kkt_backward`` then takes gradients through it. It is gh #874's shape
exactly: the wrong answer is consumed by an optimizer rather than read by a
person.

# The three branches, which are three different failures

Per CLAUDE.md's branch rule, a fixture is evidence only about the branch it
reaches, and the placement of the bad entry decides the branch:

* **upper triangle** — dropped by ``_to_coo_lower``, so the guard's own COO
  is finite and *nothing downstream can notice*. This is the silent-wrong-
  answer branch above, and the only guard on it is the dense check in
  ``_guard_psd``.
* **lower triangle, default ``check_psd``** — reaches ``eigvalsh``, which is
  gh #932's reported symptom (``LinAlgError``, or the indefinite error at
  ``n = 2``).
* **lower triangle, ``check_psd=False``** — the pre-check is skipped, the
  ``NaN`` reaches the solver, and the layer raises ``RuntimeError: convex
  solver returned status 'numerical_failure'``. A rejection, but one that
  names the solver rather than the argument.

``check_psd=False`` is the branch that fixes where the check has to *sit*.
It says whether the caller wants the definiteness precondition verified; it
is not permission to solve a different model, so — like the shape check, and
unlike ``qp.py``'s, which has ``_validate`` behind it — the finite check runs
above that early return. With it below, the first bullet is unguarded under
``check_psd=False`` and returns ``[-0.5, -0.5, -0.5, -0.5]``.

# Mutation table

Measured, 19 cases. Both changed lines are pinned, and separately:

=================================  ===========================================
removed                            fails
=================================  ===========================================
``_reject_nonfinite`` in           4, all jax: ``upper`` under both spellings
``pounce/jax/_qp.py``              of ``check_psd``, ``lower`` under
                                   ``check_psd=False``, and the batch/socp row
``_reject_nonfinite`` in           4, the same four, all torch
``pounce/torch/_qp.py``
moved back below the               4: the ``check_psd=False`` rows, both
``check_psd is False`` return      placements, both frontends
=================================  ===========================================

The ``qp`` rows never move under any of the three. They are the control that
says the message under test is the shared one, not a second implementation.

What the first two rows do **not** fail is as informative as what they do:
``[lower-jax]`` and ``[lower-torch]`` survive deleting the frontend check
outright, because a ``NaN`` in the lower triangle *is* in ``pv`` and
``_psd_verdict_coo``'s backstop raises the same message from underneath. So
the frontend check owns precisely the cases the backstop cannot see — the
dropped upper triangle, and everything under ``check_psd=False``, which does
not reach the verdict at all. Two guards, no overlap where it matters, which
is why deleting either one leaves this file red and the qp.py file green.
"""

import importlib

import numpy as np
import pytest

import pounce.qp

EXPECTED = "`P` contains NaN or Inf"
N = 4
C = np.ones(N)


def _frontend(name):
    """The module for ``name``, skipping if its dependency is absent.

    Per-test rather than at module scope on purpose: the ``python-test`` CI
    job installs jax and not torch, so a module-level
    ``importorskip("torch")`` would take the jax rows down with it — which is
    what leaves ``test_issue874_ad_frontends_p_shape.py`` skipped in that job
    today.
    """
    if name in ("jax", "torch"):
        pytest.importorskip(name)
    return importlib.import_module(f"pounce.{name}")


FRONTENDS = ("qp", "jax", "torch")


def _bad_P(placement):
    """``2·I₄`` with one non-finite entry, in the triangle named."""
    P = 2.0 * np.eye(N)
    if placement == "upper":  # dropped by `_to_coo_lower` -- the silent branch
        P[0, N - 1] = np.nan
    else:  # kept: reaches `eigvalsh`
        P[N - 1, 0] = np.nan
    return P


def _raises_the_shared_message(call):
    """Assert ``call`` rejects with the frontend's one message.

    jax rewraps whatever a host callback raises as ``JaxRuntimeError:
    INTERNAL: CpuCallback error``, so the *type* differs by frontend while
    the message does not. Unlike the shape check — which reads a static
    shape and so runs at trace time, where
    ``test_issue874_ad_frontends_p_shape.py`` can and does demand a bare
    ``ValueError`` — a value check cannot run before the values exist. The
    wrapping is therefore expected here, and matching on the message is the
    assertion that holds across all three frontends. The indefinite error
    (gh #112) is wrapped the same way on this same surface.
    """
    with pytest.raises(Exception) as exc:  # noqa: PT011 - see docstring
        call()
    assert EXPECTED in str(exc.value), exc.value
    # The misleading diagnosis from gh #932's `n = 2` symptom must not be
    # what came back instead.
    assert "positive semidefinite" not in str(exc.value), exc.value


@pytest.mark.parametrize("name", FRONTENDS)
@pytest.mark.parametrize("placement", ["upper", "lower"])
def test_a_nonfinite_P_is_rejected_on_every_frontend(placement, name):
    """The default path, both triangles, all three frontends."""
    mod = _frontend(name)
    _raises_the_shared_message(lambda: mod.solve_qp(P=_bad_P(placement), c=C))


@pytest.mark.parametrize("name", FRONTENDS)
@pytest.mark.parametrize("placement", ["upper", "lower"])
def test_check_psd_false_is_not_a_way_past_the_finite_check(placement, name):
    """gh #874's third branch, on gh #932's trigger.

    The ``upper``/``check_psd=False`` cell is the one that returned
    ``[-0.5, -0.5, -0.5, -0.5]`` with no exception while the finite check sat
    below the early return -- the documented escape hatch for a
    PSD-by-construction ``P`` doubling as an escape hatch for solving a
    different model.
    """
    mod = _frontend(name)
    _raises_the_shared_message(
        lambda: mod.solve_qp(P=_bad_P(placement), c=C, check_psd=False)
    )


def _batch_call(mod, name, P):
    """``solve_qp_batch`` spelled per frontend: ``qp`` takes a list of
    problem dicts, the two layers take one shared ``P`` and a stacked ``c``."""
    if name == "qp":
        return lambda: mod.solve_qp_batch([dict(P=P, c=C)])
    return lambda: mod.solve_qp_batch(P=P, c=np.stack([C, 2.0 * C]))


@pytest.mark.parametrize("name", FRONTENDS)
def test_the_batch_and_socp_entry_points_reject_it_too(name):
    """``_guard_psd`` is reached from three host forwards, not one -- and on
    the batch path the shared ``P`` is checked once for every row."""
    mod = _frontend(name)
    P = _bad_P("upper")
    _raises_the_shared_message(_batch_call(mod, name, P))
    _raises_the_shared_message(
        lambda: mod.solve_socp(
            P=P,
            c=C,
            cones=[("nonneg", 1)],
            G=np.ones((1, N)),
            h=np.array([1.0]),
        )
    )


@pytest.mark.parametrize("name", FRONTENDS)
def test_a_well_formed_model_still_solves(name):
    """The negative control. Without it, "raise on every P" passes everything
    above -- and the check runs on every forward now, including the
    ``check_psd=False`` one a training loop uses."""
    mod = _frontend(name)
    for kw in ({}, {"check_psd": False}):
        r = mod.solve_qp(P=2.0 * np.eye(N), c=C, **kw)
        x = np.asarray(getattr(r, "x", r)).ravel()
        assert np.allclose(x, -C / 2.0, atol=1e-6), (name, kw, x)


def test_the_gradient_path_is_unchanged_by_the_check():
    """The check is on every forward, so it is on every backward's forward.
    A finite `P` must still differentiate."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    import pounce.jax

    P = jnp.array([[3.0, 0.5], [0.5, 2.0]])
    c = jnp.array([-4.0, -1.0])

    def loss(Pm):
        return jnp.sum(pounce.jax.solve_qp(P=Pm, c=c) ** 2)

    assert np.all(np.isfinite(np.asarray(jax.grad(loss)(P))))
