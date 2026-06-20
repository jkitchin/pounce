"""Tests for pounce's stiff ODE / DAE solver and its differentiable layers.

Covers (1) the SciPy-signature adaptive Radau ``pounce.ode.solve_ivp`` —
accuracy vs analytic solutions and vs ``scipy.integrate.solve_ivp``, stiff
problems, index-1 DAEs (mass matrix), ``t_eval`` / ``dense_output`` /
``args`` plumbing, and the dict-subscriptable result — and (2) the
differentiable ``pounce.jax.odeint`` / ``pounce.torch.odeint`` gradients vs
analytic and finite differences.
"""

import numpy as np
import pytest

import pounce.ode as po


# --- adaptive Radau: accuracy ------------------------------------------------

def test_exponential_decay_accuracy():
    """y' = -k y -> y0 exp(-k t); match analytic at the nodes."""
    r = po.solve_ivp(lambda t, y: [-0.7 * y[0]], (0.0, 3.0), [2.0],
                     rtol=1e-9, atol=1e-12, dense_output=True)
    assert r.success and r.status == 0
    tt = np.linspace(0, 3, 50)
    assert np.max(np.abs(r.sol(tt)[0] - 2.0 * np.exp(-0.7 * tt))) < 1e-7


def test_linear_oscillator_accuracy():
    """y'' = -y -> [cos t, -sin t]."""
    r = po.solve_ivp(lambda t, y: [y[1], -y[0]], (0.0, 10.0), [1.0, 0.0],
                     rtol=1e-10, atol=1e-12, dense_output=True)
    tt = np.linspace(0, 10, 100)
    y = r.sol(tt)
    assert np.max(np.abs(y[0] - np.cos(tt))) < 1e-6
    assert np.max(np.abs(y[1] + np.sin(tt))) < 1e-6


def test_backward_integration():
    """Integration from a larger t0 to a smaller tf."""
    r = po.solve_ivp(lambda t, y: [y[0]], (2.0, 0.0), [np.e ** 2],
                     rtol=1e-9, atol=1e-12)
    assert abs(r.y[0, -1] - 1.0) < 1e-6


def test_backward_dense_output():
    """Dense output must be correct for a decreasing (backward) mesh."""
    r = po.solve_ivp(lambda t, y: [y[0]], (2.0, 0.0), [np.e ** 2],
                     rtol=1e-9, atol=1e-12, dense_output=True)
    tt = np.linspace(0, 2, 40)
    assert np.max(np.abs(r.sol(tt)[0] - np.exp(tt))) < 1e-7


def test_full_rank_nonidentity_mass():
    """A non-singular, non-identity M must be handled by the error estimate.

    M = diag(2, 1), f = (-2 y0, -y1)  =>  y0' = -y0, y1' = -y1  =>  both e^-t.
    """
    M = np.diag([2.0, 1.0])
    r = po.solve_ivp(lambda t, y: [-2 * y[0], -y[1]], (0.0, 3.0), [1.0, 1.0],
                     mass=M, rtol=1e-9, atol=1e-12, dense_output=True)
    tt = np.linspace(0, 3, 30)
    assert np.max(np.abs(r.sol(tt) - np.exp(-tt))) < 1e-6


# --- stiffness: agreement with SciPy -----------------------------------------

def test_stiff_vs_scipy():
    """A stiff scalar problem: match SciPy's Radau closely."""
    sp = pytest.importorskip("scipy.integrate")

    def f(t, y):
        return [-2000.0 * (y[0] - np.cos(t))]

    kw = dict(method="Radau", rtol=1e-8, atol=1e-10, dense_output=True)
    ref = sp.solve_ivp(f, (0, 1.5), [0.0], **kw)
    got = po.solve_ivp(f, (0, 1.5), [0.0], **kw)
    tt = np.linspace(0, 1.5, 60)
    assert np.max(np.abs(ref.sol(tt) - got.sol(tt))) < 1e-5


def test_van_der_pol_stiff_vs_scipy():
    """Van der Pol mu=1000 over a full relaxation cycle vs SciPy."""
    sp = pytest.importorskip("scipy.integrate")
    mu = 1000.0

    def f(t, y):
        return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]

    def jac(t, y):
        return [[0.0, 1.0],
                [-2 * mu * y[0] * y[1] - 1.0, mu * (1 - y[0] ** 2)]]

    kw = dict(method="Radau", jac=jac, rtol=1e-6, atol=1e-8, dense_output=True)
    ref = sp.solve_ivp(f, (0, 3000.0), [2.0, 0.0], **kw)
    got = po.solve_ivp(f, (0, 3000.0), [2.0, 0.0], **kw)
    tt = np.linspace(0, 3000, 400)
    assert np.max(np.abs(ref.sol(tt) - got.sol(tt))) < 1e-4
    # comparable work to SciPy (within ~3x on step count)
    assert got.nstep < 3 * ref.t.size


def test_stage_predictor_warm_start_reduces_nfev():
    """The collocation stage predictor warm-starts the simplified-Newton stage
    solve from the previous step instead of a cold ``K = 0``, cutting RHS
    evaluations. On Van der Pol mu=1000 with a finite-difference Jacobian a
    cold start needs ~23.7k evals; the predictor needs ~18k. ``nfev`` is
    deterministic, so this guards the warm start against silent regression."""
    def f(t, y):
        return [y[1], 1000.0 * (1 - y[0] ** 2) * y[1] - y[0]]

    r = po.solve_ivp(f, (0.0, 3000.0), [2.0, 0.0], rtol=1e-6, atol=1e-9)
    assert r.success
    assert r.nfev < 20000, f"stage predictor regressed: nfev={r.nfev}"


def test_lu_pattern_built_once_per_solve(monkeypatch):
    """The (3n×3n) stage and (n×n) error LU patterns — and FERAL's cached
    symbolic analysis — must be built once per solve and refactored in place,
    not re-created on every step. Re-analysing the pattern per refactor was the
    large-n bottleneck. Guard the invariant: exactly two patterns are built,
    regardless of how many times the step refactors."""
    from pounce.ode import _radau

    n_built = [0]
    orig = _radau._dense_lu_pattern

    def counting(N):
        n_built[0] += 1
        return orig(N)

    monkeypatch.setattr(_radau, "_dense_lu_pattern", counting)

    def f(t, y):
        return [y[1], 100.0 * (1 - y[0] ** 2) * y[1] - y[0]]

    r = po.solve_ivp(f, (0.0, 200.0), [2.0, 0.0], rtol=1e-6, atol=1e-9)
    assert r.success
    assert r.nlu >= 4, "test needs several refactors to be meaningful"
    assert n_built[0] == 2, f"LU pattern rebuilt per refactor ({n_built[0]} builds)"


# --- index-1 DAE (mass matrix) -----------------------------------------------

def test_robertson_dae():
    """Robertson kinetics as an index-1 DAE: M y' = f with singular M."""
    k1, k2, k3 = 0.04, 3e7, 1e4

    def f(t, y):
        return [-k1 * y[0] + k3 * y[1] * y[2],
                k1 * y[0] - k3 * y[1] * y[2] - k2 * y[1] ** 2,
                y[0] + y[1] + y[2] - 1.0]

    M = np.diag([1.0, 1.0, 0.0])
    r = po.solve_ivp(f, (0.0, 1e4), [1.0, 0.0, 0.0], mass=M,
                     rtol=1e-6, atol=1e-8)
    yend = r.y[:, -1]
    # algebraic constraint held exactly; known Robertson value at t=1e4
    assert abs(yend.sum() - 1.0) < 1e-8
    assert abs(yend[0] - 0.1083) < 5e-3


def test_simple_dae():
    """y0' = -y0, 0 = y1 - y0**2: algebraic state tracks the differential one."""
    def f(t, y):
        return [-y[0], y[1] - y[0] ** 2]

    M = np.diag([1.0, 0.0])
    r = po.solve_ivp(f, (0.0, 2.0), [1.0, 1.0], mass=M, rtol=1e-8, atol=1e-10,
                     dense_output=True)
    tt = np.linspace(0, 2, 30)
    y = r.sol(tt)
    assert np.max(np.abs(y[0] - np.exp(-tt))) < 1e-6
    assert np.max(np.abs(y[1] - np.exp(-2 * tt))) < 1e-6


# --- SciPy-signature plumbing ------------------------------------------------

def test_t_eval_and_args():
    """args threads parameters; t_eval picks output points."""
    te = np.linspace(0, 2, 6)
    r = po.solve_ivp(lambda t, y, a: [a * y[0]], (0, 2), [1.0],
                     args=(-1.0,), t_eval=te, rtol=1e-9, atol=1e-12)
    assert np.allclose(r.t, te)
    assert np.max(np.abs(r.y[0] - np.exp(-te))) < 1e-7


def test_dense_output_callable():
    r = po.solve_ivp(lambda t, y: [-y[0]], (0, 1), [1.0],
                     dense_output=True, rtol=1e-9, atol=1e-12)
    assert callable(r.sol)
    assert abs(r.sol(0.5)[0] - np.exp(-0.5)) < 1e-6


def test_result_is_dict_subscriptable():
    """OdeResult supports both attribute and SciPy-Bunch dict access."""
    r = po.solve_ivp(lambda t, y: [-y[0]], (0, 1), [1.0])
    assert r["y"][0, -1] == r.y[0, -1]
    assert "success" in r and r.get("status") == 0
    assert set(["t", "y", "status", "success"]).issubset(set(r.keys()))


def test_non_radau_method_raises():
    with pytest.raises(NotImplementedError):
        po.solve_ivp(lambda t, y: [-y[0]], (0, 1), [1.0], method="RK45")


def test_step_cap_reports_failure_not_success():
    """Hitting the internal step cap returns status<0 / success=False (SciPy
    behaviour), with the partial trajectory — never raises."""
    from pounce.ode import _radau
    res = _radau.integrate(lambda t, y: [-y[0]], 0.0, 100.0, np.array([1.0]),
                           rtol=1e-9, atol=1e-12, max_steps=3)
    assert res["success"] is False and res["status"] < 0
    assert "maximum number" in res["message"].lower()
    assert res["t"][-1] < 100.0          # partial trajectory returned


def test_failure_with_t_eval_returns_nans_not_garbage():
    """No recorded steps + t_eval requested must yield NaNs, not garbage."""
    from pounce.ode import _radau
    te = np.linspace(0, 1, 5)
    res = _radau.integrate(lambda t, y: [-y[0]], 0.0, 1.0, np.array([1.0]),
                           t_eval=te, max_steps=0)
    assert res["success"] is False
    assert res["y"].shape == (1, te.size)
    assert np.all(np.isnan(res["y"]))


def test_info_excluded_from_dict_surface():
    """The internal `info` field must not leak through the Bunch interface."""
    r = po.solve_ivp(lambda t, y: [-y[0]], (0, 1), [1.0])
    assert "info" not in r.keys()
    assert "info" not in r
    with pytest.raises(KeyError):
        r["info"]
    # consistency: a method name is neither a key nor membership-true
    assert "get" not in r and "keys" not in r


# --- differentiable JAX layer ------------------------------------------------

def test_jax_odeint_gradients():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import pounce.jax as pj
    from jax import config
    config.update("jax_enable_x64", True)

    t = jnp.linspace(0.0, 2.0, 81)
    T, k0 = 2.0, 0.7

    def f(tt, y, th):
        return jnp.array([-th[0] * y[0]])

    def yT_of_k(k):
        return pj.odeint(f, jnp.array([1.0]), t, jnp.array([k])).y[0, -1]

    def yT_of_y0(y0v):
        return pj.odeint(f, jnp.array([y0v]), t, jnp.array([k0])).y[0, -1]

    assert abs(float(yT_of_k(k0)) - np.exp(-k0 * T)) < 1e-6
    assert abs(float(jax.grad(yT_of_k)(k0)) + T * np.exp(-k0 * T)) < 1e-5
    assert abs(float(jax.grad(yT_of_y0)(1.0)) - np.exp(-k0 * T)) < 1e-5


def test_jax_odeint_jacobian_vs_fd():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import pounce.jax as pj
    from jax import config
    config.update("jax_enable_x64", True)

    t = jnp.linspace(0.0, 5.0, 201)
    y0 = jnp.array([1.0, 0.0])

    def f(tt, y, th):
        w, c = th[0], th[1]
        return jnp.array([y[1], -w * w * y[0] - 2 * c * w * y[1]])

    def end(th):
        return pj.odeint(f, y0, t, th).y[:, -1]

    th0 = jnp.array([1.3, 0.1])
    J = np.asarray(jax.jacobian(end)(th0))
    h = 1e-6
    Jfd = np.stack([
        (np.asarray(end(th0.at[i].add(h))) - np.asarray(end(th0.at[i].add(-h)))) / (2 * h)
        for i in range(2)
    ], axis=1)
    assert np.max(np.abs(J - Jfd)) < 1e-6


# --- differentiable Torch layer ----------------------------------------------

def test_torch_odeint_gradients():
    torch = pytest.importorskip("torch")
    import pounce.torch as pt
    torch.set_default_dtype(torch.float64)

    t = torch.linspace(0.0, 2.0, 81, dtype=torch.float64)
    T, k0 = 2.0, 0.7

    def f(tt, y, th):
        return torch.stack([-th[0] * y[0]])

    k = torch.tensor([k0], dtype=torch.float64, requires_grad=True)
    y0 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    yT = pt.odeint(f, y0, t, k).y[0, -1]
    yT.backward()
    assert abs(yT.item() - np.exp(-k0 * T)) < 1e-6
    assert abs(k.grad.item() + T * np.exp(-k0 * T)) < 1e-5
    assert abs(y0.grad.item() - np.exp(-k0 * T)) < 1e-5
