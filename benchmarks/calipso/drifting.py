"""Drifting parallel-park, ported from CALIPSO.jl
examples/contact_implicit/drifting.jl and models/cybertruck.jl
(thowell/CALIPSO.jl @ 0e3bdcf, 2022-12-14).

State x_t = (q_t, q_{t+1}) in R6, q = (px, py, theta); t = 0..T-1.
Action u_t in R14: [v_cmd, omega_cmd, beta1(3), eta1(3), beta2(3), eta2(3)].
beta_k = (friction budget, impulse_xy) must lie in SOC(3); eta_k = (slip
speed bound, contact-point velocity) must lie in SOC(3); and the
max-dissipation principle is the cone complementarity beta_k o eta_k = 0.

CALIPSO keeps beta, eta in its native second-order cones. An NLP solver
cannot, so the cone membership is lowered one of two ways (`soc_form`):
  "norm"     t - ||x|| >= 0        what the authors' own Ipopt script
                                   (examples/experimental/cyberdrift_ipopt.jl)
                                   does; nondifferentiable at x = 0
  "squared"  t^2 - ||x||^2 >= 0,   smooth, nonconvex
             t >= 0 (bound)
For two vectors in the (self-dual) second-order cone, beta.eta = 0 is
equivalent to the full Jordan product beta o eta = 0, so the scalar
beta.eta is the complementarity function the routes lower; the paper's
3-row Jordan form is kept as the `jordan_eq` route for comparison.

RoboDojo.lagrangian_derivatives (RoboDojo.jl src/robots/integrator.jl) is
D1L = -bias, D2L = M v; with zero dynamics bias (true here) that is
D1L = 0, D2L = M v, which is what dynamics_discrete below uses.
"""

import jax.numpy as jnp
import numpy as np

from mpcc_nlp import Model

T = 15
dt = 0.1
nx, nu = 6, 14
mass, inertia = 1.0, 0.1
mu_fric = (0.5, 0.25)
k_front = np.array([0.1, 0.0])
k_rear = np.array([-0.1, 0.0])
M = jnp.array([mass, mass, inertia])

x_init = np.array([0.0, 1.5, -0.5 * np.pi, 0.0, 1.5, -0.5 * np.pi])
x_goal = np.array([3.0, 0.0, 0.5 * np.pi, 3.0, 0.0, 0.5 * np.pi])
u_min = np.array([0.0, -0.5])
u_max = np.array([25.0, 0.5])
p_car1 = np.array([3.0, 0.65])
p_car2 = np.array([3.0, -0.65])

ix = [np.arange(nx) + nx * t for t in range(T)]
iu = [T * nx + np.arange(nu) + nu * t for t in range(T - 1)]
nz = T * nx + (T - 1) * nu


def contact_jacobian(q):
    c, s = jnp.cos(q[2]), jnp.sin(q[2])
    rf = jnp.array([c * k_front[0] - s * k_front[1], s * k_front[0] + c * k_front[1]])
    rr = jnp.array([c * k_rear[0] - s * k_rear[1], s * k_rear[0] + c * k_rear[1]])
    # [I2 0] * -hat(r)^T * e3 = hat(r) e3 = (r_y, -r_x)
    return jnp.array([[1.0, 0.0, rf[1]], [0.0, 1.0, -rf[0]],
                      [1.0, 0.0, rr[1]], [0.0, 1.0, -rr[0]]])


def dynamics(y, x, u):
    q1m, q2m = x[0:3], x[3:6]
    q2p, q3p = y[0:3], y[3:6]
    b = jnp.concatenate([u[3:5], u[9:11]])
    lam = contact_jacobian(q2p).T @ b
    vm1 = (q2p - q1m) / dt
    vm2 = (q3p - q2p) / dt
    qm2 = 0.5 * (q2p + q3p)
    Bu = jnp.array([jnp.cos(qm2[2]) * u[0], jnp.sin(qm2[2]) * u[0], u[1]])
    d = M * vm1 - M * vm2 + Bu + lam
    return jnp.concatenate([q2p - q2m, d])


def cones(u):
    return [u[2:5], u[5:8], u[8:11], u[11:14]]


def contact_eq(x, u):
    q2, q3 = x[0:3], x[3:6]
    b1, e1, b2, e2 = cones(u)
    v = contact_jacobian(q3) @ (q3 - q2) / dt
    return jnp.concatenate([
        jnp.array([b1[0] - mu_fric[0] * mass * 9.81 * dt, b2[0] - mu_fric[1] * mass * 9.81 * dt]),
        v[0:2] - e1[1:3], v[2:4] - e2[1:3]])


def soc_product(a, b):
    return jnp.concatenate([jnp.atleast_1d(jnp.dot(a, b)), a[0] * b[1:] + b[0] * a[1:]])


def circle(x, p, r):
    return (x[0] - p[0]) ** 2 + (x[1] - p[1]) ** 2 - r**2


def eq(z):
    rows = []
    for t in range(T - 1):
        x, u, y = z[ix[t]], z[iu[t]], z[ix[t + 1]]
        rows.append(dynamics(y, x, u))
        rows.append(contact_eq(x, u))
    rows.append(z[ix[0]] - x_init)
    rows.append((z[ix[T - 1]] - x_goal)[0:3])
    return jnp.concatenate(rows)


def make_ineq(soc_form):
    def ineq(z):
        rows = []
        for t in range(T - 1):
            x, u = z[ix[t]], z[iu[t]]
            rows.append(jnp.array([circle(x, p_car1, 0.1), circle(x, p_car2, 0.1)]))
            for c in cones(u):
                if soc_form == "native":
                    # declared to POUNCE as cones (`cone_vars`), no row here
                    continue
                if soc_form == "squared":
                    rows.append(jnp.atleast_1d(c[0] ** 2 - jnp.dot(c[1:], c[1:])))
                else:
                    rows.append(jnp.atleast_1d(c[0] - jnp.sqrt(jnp.dot(c[1:], c[1:]))))
        return jnp.concatenate(rows)
    return ineq


def comp(z):
    out = []
    for t in range(T - 1):
        b1, e1, b2, e2 = cones(z[iu[t]])
        out.append(jnp.array([jnp.dot(b1, e1), jnp.dot(b2, e2)]))
    return jnp.concatenate(out)


def jordan(z):
    out = []
    for t in range(T - 1):
        b1, e1, b2, e2 = cones(z[iu[t]])
        out += [soc_product(b1, e1), soc_product(b2, e2)]
    return jnp.concatenate(out)


def f(z):
    J = 0.0
    for t in range(T):
        x = z[ix[t]]
        v = (x[3:6] - x[0:3]) / dt
        w = 5.0 if t == T - 1 else 1e-3
        J += 0.5 * w * jnp.dot(v, v)
        J += 0.5 * 1e-3 * jnp.dot(x - x_goal, x - x_goal)
        if t < T - 1:
            u = z[iu[t]]
            J += 0.5 * 1e-3 * (jnp.dot(u[0:2], u[0:2]) + 1e-5 * jnp.dot(u[2:], u[2:]))
    return J


def bounds(soc_form):
    lb = np.full(nz, -np.inf)
    ub = np.full(nz, np.inf)
    for t in range(T - 1):
        lb[iu[t][0:2]] = u_min
        ub[iu[t][0:2]] = u_max
        if soc_form == "squared":
            for k in (2, 5, 8, 11):
                lb[iu[t][k]] = 0.0
    return lb, ub


def report(z):
    z = jnp.asarray(z)
    e = np.asarray(eq(z))
    jr = np.asarray(jordan(z))
    cone_v, obs_v, box_v = 0.0, 0.0, 0.0
    zn = np.asarray(z)
    for t in range(T - 1):
        x, u = zn[ix[t]], zn[iu[t]]
        for c in cones(u):
            cone_v = max(cone_v, np.linalg.norm(c[1:]) - c[0])
        obs_v = max(obs_v, -float(circle(x, p_car1, 0.1)), -float(circle(x, p_car2, 0.1)))
        box_v = max(box_v, *(u_min - u[0:2]), *(u[0:2] - u_max))
    viol = max(float(np.max(np.abs(e))), float(np.max(np.abs(jr))), cone_v, obs_v, box_v, 0.0)
    return {"obj": float(f(z)), "eq_viol": float(np.max(np.abs(e))),
            "compl": float(np.max(np.abs(jr))), "cone_viol": max(cone_v, 0.0),
            "obstacle_viol": max(obs_v, 0.0), "violation": viol}


def make_model(soc_form):
    lb, ub = bounds(soc_form)
    m = Model(f"drifting[{soc_form}]", nz, f, eq, make_ineq(soc_form), comp, lb, ub,
              extra={"jordan": jordan}, report=report)
    m.soc_complementarity = True
    if soc_form == "native":
        # Variable blocks POUNCE carries as second-order cones natively
        # (workstream C); `mpcc_nlp` adds one free row per entry and declares
        # them with `Problem(..., cones=...)`.
        m.cone_vars = [iu[t][k:k + 3] for t in range(T - 1) for k in (2, 5, 8, 11)]
    return m


def initial_point(seed):
    rng = np.random.default_rng(seed)
    z = np.zeros(nz)
    for t in range(T):
        z[ix[t]] = x_init + (x_goal - x_init) * t / (T - 1)
    for t in range(T - 1):
        z[iu[t]] = np.concatenate([1e-3 * rng.standard_normal(2), np.tile([1.0, 0.1, 0.1], 4)])
    return z
