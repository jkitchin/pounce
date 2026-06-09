"""Regression tests from the LP/QP/torch hands-on QA pass.

Companion to ``dev-notes/qa-lp-qp-torch.md``. Two kinds of test live here:

* **Green** — behavior that is correct today but was not covered by the existing
  fixtures (cross-checks vs scipy, routing transparency, warm-start optimality,
  edge cases, double-backward, JAX↔Torch parity, known optima). These guard
  against regressions in behavior we verified by hand.
* **xfail** — behavior the QA pass found wrong; each is written to assert the
  *desired* (post-fix) behavior and is marked ``xfail`` against its filed issue,
  so the suite stays green and the bug is pinned until fixed.

Filed issues: #112 (indefinite P), #113 (input validation), #114 (doc routing),
#115 (verbose default), #116 (dense-input cliff), #117 (sos UX).
"""
from __future__ import annotations

import numpy as np
import pytest

from pounce.qp import solve_qp, solve_socp


# ---------------------------------------------------------------------------
# Green: correctness cross-checks (untested before this pass)
# ---------------------------------------------------------------------------
def test_convex_qp_matches_scipy():
    scipy_opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(7)
    for _ in range(5):
        n = int(rng.integers(2, 6))
        M = rng.standard_normal((n, n))
        P = M @ M.T + 0.1 * np.eye(n)
        c = rng.standard_normal(n)
        G = rng.standard_normal((1, n))
        h = np.array([rng.uniform(1.0, 3.0)])
        lb, ub = -2 * np.ones(n), 2 * np.ones(n)
        r = solve_qp(P, c, G=G, h=h, lb=lb, ub=ub, tol=1e-10)
        assert r.status == "optimal"
        fun = lambda x: 0.5 * x @ P @ x + c @ x
        s = scipy_opt.minimize(
            fun, np.zeros(n), method="SLSQP",
            bounds=list(zip(lb, ub)),
            constraints=[{"type": "ineq", "fun": lambda x: h - G @ x}],
            options={"ftol": 1e-12, "maxiter": 500},
        )
        assert abs(fun(r.x) - fun(s.x)) < 1e-5


def test_lp_matches_scipy_highs():
    scipy_opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(11)
    for _ in range(5):
        n = int(rng.integers(2, 6))
        c = rng.standard_normal(n)
        G = rng.standard_normal((n + 1, n))
        h = np.abs(rng.standard_normal(n + 1)) + 5
        lb, ub = -3 * np.ones(n), 3 * np.ones(n)
        r = solve_qp(None, c, G=G, h=h, lb=lb, ub=ub, tol=1e-10)
        res = scipy_opt.linprog(c, A_ub=G, b_ub=h,
                                bounds=list(zip(lb, ub)), method="highs")
        assert r.status == "optimal" and res.success
        assert abs(c @ r.x - res.fun) < 1e-5


def test_routing_transparency_qp():
    """A convex QP with analytic jac+hess auto-routes to qp-ipm AND its x*
    matches the forced-NLP solve (routing never changes the answer)."""
    from pounce import minimize
    rng = np.random.default_rng(3)
    n = 4
    M = rng.standard_normal((n, n))
    P = M @ M.T + 0.2 * np.eye(n)
    c = rng.standard_normal(n)
    fun, jac, hess = (lambda x: 0.5 * x @ P @ x + c @ x,
                      lambda x: P @ x + c, lambda x: P)
    bounds = list(zip(-1.5 * np.ones(n), 1.5 * np.ones(n)))
    auto = minimize(fun, np.zeros(n), jac=jac, hess=hess, bounds=bounds,
                    options={"print_level": 0})
    nlp = minimize(fun, np.zeros(n), jac=jac, hess=hess, bounds=bounds,
                   options={"print_level": 0, "solver_selection": "nlp"})
    assert auto.info.get("solver") == "qp-ipm"
    assert np.max(np.abs(auto.x - nlp.x)) < 1e-6


def test_warm_start_fewer_iters_same_optimum():
    rng = np.random.default_rng(1)
    n = 50
    P = np.diag(rng.uniform(1, 3, n))
    c = rng.standard_normal(n)
    G = rng.standard_normal((5, n))
    h = np.abs(rng.standard_normal(5)) + n
    lb, ub = -3 * np.ones(n), 3 * np.ones(n)
    base = solve_qp(P, c, G=G, h=h, lb=lb, ub=ub, tol=1e-9)
    c2 = c + 0.01 * rng.standard_normal(n)
    cold = solve_qp(P, c2, G=G, h=h, lb=lb, ub=ub, tol=1e-9)
    warm = solve_qp(P, c2, G=G, h=h, lb=lb, ub=ub, tol=1e-9, warm_start=base)
    assert cold.status == warm.status == "optimal"
    assert warm.iters < cold.iters           # warm-start pays off
    assert np.max(np.abs(cold.x - warm.x)) < 1e-8   # same optimum


# ---------------------------------------------------------------------------
# Green: edge cases that are handled correctly
# ---------------------------------------------------------------------------
def test_zero_size_problem():
    r = solve_qp(np.zeros((0, 0)), np.zeros(0))
    assert r.status == "optimal" and r.x.shape == (0,)


def test_duplicate_equality_rows_ok():
    r = solve_qp(np.eye(2), np.array([-1.0, -1.0]),
                 A=np.array([[1.0, 1.0], [1.0, 1.0]]), b=np.array([1.0, 1.0]))
    assert r.status == "optimal"
    assert abs(r.x.sum() - 1.0) < 1e-6


def test_infeasible_and_unbounded_status():
    inf = solve_qp(np.eye(1), np.array([0.0]),
                   G=np.array([[1.0], [-1.0]]), h=np.array([1.0, -2.0]))
    assert inf.status == "primal_infeasible"
    unb = solve_qp(None, np.array([-1.0]), lb=[0.0], ub=[np.inf])
    assert unb.status == "dual_infeasible"


def test_known_socp_optimum():
    # min cᵀx s.t. ‖x‖ ≤ 1  ->  x* = -c/‖c‖,  obj = -‖c‖
    n = 2
    c = np.array([3.0, 4.0])
    G = np.vstack([np.zeros((1, n)), -np.eye(n)])
    h = np.concatenate([[1.0], np.zeros(n)])
    r = solve_socp(P=None, c=c, G=G, h=h, cones=[("soc", n + 1)])
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, [-0.6, -0.8], atol=1e-5)
    assert abs(c @ r.x + 5.0) < 1e-5


def test_known_sos_optima():
    from pounce.sos import sos_minimize
    # (x-1)² + 2  ->  2.0
    r1 = sos_minimize({(2,): 1.0, (1,): -2.0, (0,): 3.0})
    assert abs(r1.lower_bound - 2.0) < 1e-5
    # min x s.t. 4 - x² ≥ 0  ->  -2.0
    r2 = sos_minimize({(1,): 1.0}, inequalities=[{(0,): 4.0, (2,): -1.0}], order=2)
    assert abs(r2.lower_bound + 2.0) < 1e-5


def test_native_solve_qp_upcasts_float32():
    """Documents the (intentional, O2/O5) float32->float64 coercion of the
    native QP layer, so a change to this behavior is noticed."""
    P = np.diag([2.0, 2.0]).astype(np.float32)
    c = np.array([-2.0, -2.0], dtype=np.float32)
    r = solve_qp(P, c)
    assert r.status == "optimal"
    assert r.x.dtype == np.float64
    np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-6)


def test_import_guard_message_without_torch(monkeypatch):
    """pounce.torch raises a clear, actionable ImportError when torch is absent."""
    import sys
    import builtins
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return real_import(name, *a, **k)

    for m in list(sys.modules):
        if m == "torch" or m.startswith("torch.") or m.startswith("pounce.torch"):
            monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"pip install pounce\[torch\]"):
        import pounce.torch  # noqa: F401


# ---------------------------------------------------------------------------
# Torch: double-backward + JAX↔Torch parity (skipped if extras missing)
# ---------------------------------------------------------------------------
def test_torch_gradgradcheck_solve_qp():
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    from torch.autograd import gradcheck, gradgradcheck
    from pounce.torch import solve_qp as tqp
    P = torch.eye(3)
    G = torch.tensor([[1.0, 1.0, 1.0]])
    h = torch.tensor([1.0])
    f = lambda c: tqp(P=P, c=c, G=G, h=h)
    c0 = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
    assert gradcheck(f, (c0,), atol=1e-6, rtol=1e-4)
    assert gradgradcheck(f, (c0,), atol=1e-5, rtol=1e-3)


def test_jax_torch_parity_solve_qp():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    from pounce.torch import solve_qp as tqp
    from pounce.jax import solve_qp as jqp

    cases = [
        dict(P=np.eye(3), c=[-1., -2., -3.], A=None, b=None, G=None, h=None),
        dict(P=np.eye(3), c=[-1., -2., -3.], A=[[1., 1., 1.]], b=[1.0],
             G=None, h=None),
        dict(P=np.eye(2), c=[-4., -4.], G=[[1., 1.]], h=[0.5], A=None, b=None),
    ]
    t = lambda a: None if a is None else torch.tensor(np.asarray(a, float))
    j = lambda a: None if a is None else jnp.asarray(np.asarray(a, float))
    for d in cases:
        ct = torch.tensor(np.asarray(d["c"], float), requires_grad=True)
        xt = tqp(P=t(d["P"]), c=ct, G=t(d["G"]), h=t(d["h"]),
                 A=t(d["A"]), b=t(d["b"]))
        xt.sum().backward()
        gt, xtn = ct.grad.numpy(), xt.detach().numpy()
        gj = jax.grad(lambda c: jqp(P=j(d["P"]), c=c, G=j(d["G"]), h=j(d["h"]),
                                    A=j(d["A"]), b=j(d["b"])).sum())(j(d["c"]))
        xj = np.asarray(jqp(P=j(d["P"]), c=j(d["c"]), G=j(d["G"]), h=j(d["h"]),
                            A=j(d["A"]), b=j(d["b"])))
        np.testing.assert_allclose(xtn, xj, atol=1e-9)
        np.testing.assert_allclose(gt, np.asarray(gj), atol=1e-9)


# ---------------------------------------------------------------------------
# xfail: filed bugs (assert the desired post-fix behavior; pinned to issues)
# ---------------------------------------------------------------------------
@pytest.mark.xfail(reason="#112: no PSD guard; indefinite P returns 'optimal'",
                   strict=False)
def test_solve_qp_rejects_indefinite_P():
    P = np.array([[1.0, 0.0], [0.0, -1.0]])   # indefinite -> unbounded below
    r = solve_qp(P, np.zeros(2))
    # Desired: a non-optimal status (or a raised error). Today: 'optimal'.
    assert r.status != "optimal"


@pytest.mark.xfail(reason="#113: no shape validation; mismatch -> infeasible",
                   strict=False)
def test_solve_qp_validates_A_b_shape():
    with pytest.raises((ValueError, TypeError)):
        solve_qp(np.diag([2.0, 2.0]), np.zeros(2),
                 A=np.array([[1.0, 1.0]]), b=np.array([1.0, 2.0]))


@pytest.mark.xfail(reason="#113: no NaN/Inf validation; nan -> iteration_limit",
                   strict=False)
def test_solve_qp_validates_nan_inputs():
    with pytest.raises((ValueError, TypeError)):
        solve_qp(np.eye(2), np.array([np.nan, 1.0]))


@pytest.mark.xfail(reason="#115: minimize is verbose by default (unlike scipy)",
                   strict=False)
def test_minimize_silent_by_default(tmp_path):
    # The IPM log is written from Rust to OS fd 1, so it must be captured at the
    # file-descriptor level (Python's redirect_stdout cannot see it — that is
    # itself part of #115). Desired: a default minimize() emits nothing.
    import os
    from pounce import minimize
    cap = tmp_path / "fd1.txt"
    saved = os.dup(1)
    fd = os.open(str(cap), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        os.dup2(fd, 1)
        minimize(lambda x: (x - 1) @ (x - 1) + 1, x0=np.zeros(3))
    finally:
        os.dup2(saved, 1)
        os.close(fd)
        os.close(saved)
    assert cap.read_text() == ""


@pytest.mark.xfail(reason="#117: cryptic TypeError on SymPy input", strict=False)
def test_sos_clear_error_on_sympy_input():
    sp = pytest.importorskip("sympy")
    from pounce.sos import sos_minimize
    x = sp.symbols("x")
    with pytest.raises((TypeError, ValueError)) as ei:
        sos_minimize((x - 1) ** 2 + 2)
    # Desired: a message naming the dict format, not "'Add' object is not iterable".
    assert "iterable" not in str(ei.value)

# Note: F7 (garbage `lower_bound` on a `numerical_failure` SOS relaxation) is
# tracked in #117 and the QA report but is NOT unit-tested here — reproducing it
# requires the 5-var/order-3 relaxation, a ~66 s solve too slow and too
# nondeterministic in its failure trigger to belong in the test suite.
