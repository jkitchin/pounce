"""Ball-in-cup, ported from CALIPSO.jl examples/contact_implicit/ball_in_cup.jl
and models/ball_in_cup.jl (thowell/CALIPSO.jl @ 0e3bdcf, 2022-12-14).

q = (cup_x, cup_z, ball_x, ball_z). The string is a unilateral constraint
phi(q) = L^2 - |cup - ball|^2 >= 0 with tension impulse gamma >= 0, and
gamma ⟂ phi: the string pulls only when taut.

State x_1 = (q_1, q_2) in R8; x_t = (q_t, q_{t+1}, gamma_{t-1}) in R9 for
t >= 2 (the previous impulse is carried in the state so the complementarity
row can pair it with the next configuration). Action u_t = (cup force,
gamma_t) in R3. CALIPSO writes gamma_{t-1} * phi(q_{t+1}) = 0 as an
equality row at every t >= 2; here it is the complementarity block.
"""

import jax.numpy as jnp
import numpy as np

from mpcc_nlp import Model, generic_report

T = 21
h = 0.075
m_cup, m_ball = 1.0, 0.01
L = 1.0
g_cup, g_ball = 0.0, 10.0
M = jnp.array([m_cup, m_cup, m_ball, m_ball])
BIAS = jnp.array([0.0, m_cup * g_cup, 0.0, m_ball * g_ball])

x1 = np.array([0.0, 0.0, 0.0, -0.99, 0.0, 0.0, 0.0, -0.99])
xT = np.array([0.0, 0.0, 0.0, 0.125, 0.0, 0.0, 0.0, 0.125])
xM1 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
dr = np.sqrt(0.5 * L**2)
xM2 = np.array([0.0, 0.0, dr, dr, 0.0, 0.0, dr, dr])
tM1, tM2 = 11, 16  # 1-based, as in the source

nxs = [8] + [9] * (T - 1)
nu = 3
_off = np.cumsum([0] + nxs)
ix = [np.arange(nxs[t]) + _off[t] for t in range(T)]
iu = [_off[-1] + np.arange(nu) + nu * t for t in range(T - 1)]
nz = int(_off[-1] + nu * (T - 1))


def phi(q):
    dq = q[0:2] - q[2:4]
    return L**2 - jnp.dot(dq, dq)


def impact_jacobian_T(q):
    dq = q[0:2] - q[2:4]
    return -2.0 * jnp.concatenate([dq, -dq])


def implicit_dynamics(q0, q1, u1, lam, q2):
    vm1 = (q1 - q0) / h
    vm2 = (q2 - q1) / h
    D1L = -BIAS
    d = 0.5 * h * D1L + M * vm1 + 0.5 * h * D1L - M * vm2
    return d + jnp.concatenate([u1, jnp.zeros(2)]) + lam


def dynamics(y, x, u):
    q1m, q2m = x[0:4], x[4:8]
    q2p, q3p = y[0:4], y[4:8]
    gamma = u[2]
    lam = impact_jacobian_T(q2p) * gamma
    return jnp.concatenate([q2p - q2m, implicit_dynamics(q1m, q2p, u[0:2], lam, q3p),
                            jnp.atleast_1d(y[8] - u[2])])


def eq(z):
    rows = []
    for t in range(T - 1):
        rows.append(dynamics(z[ix[t + 1]], z[ix[t]], z[iu[t]]))
    rows.append(z[ix[0]] - x1)
    for t in range(1, T):  # 0-based t; source t + 1
        x = z[ix[t]]
        if t + 1 == tM1:
            rows.append(x[6:8] - xM1[6:8])
        if t + 1 == tM2:
            rows.append(x[6:8] - xM2[6:8])
    xl = z[ix[T - 1]]
    rows += [xl[0:2] - xT[0:2], xl[4:6] - xT[4:6], xl[6:8] - xT[6:8]]
    return jnp.concatenate(rows)


def ineq(z):
    return jnp.stack([phi(z[ix[t]][4:8]) for t in range(T)])


def comp(z):
    return jnp.stack([z[ix[t]][8] * phi(z[ix[t]][4:8]) for t in range(1, T)])


def f(z):
    J = 0.0
    for t in range(T):
        x = z[ix[t]]
        v = (x[4:8] - x[0:4]) / h
        J += 0.5 * 1e-1 * jnp.dot(v, v)
        dc = x[4:6] - xT[4:6]
        J += 0.5 * jnp.dot(dc, dc)
        if t < T - 1:
            u = z[iu[t]]
            J += 0.5 * 0.1 * jnp.dot(u, u)
        else:
            db = x[4:6] - x[6:8]
            J += 0.5 * jnp.dot(db, db)
    return J


lb = np.full(nz, -np.inf)
ub = np.full(nz, np.inf)
for t in range(T - 1):
    lb[iu[t][2]] = 0.0  # gamma >= 0 (the source's `γ` cone row)

model = Model("ballincup", nz, f, eq, ineq, comp, lb, ub)
model.report = generic_report(model)
#: (G, H) with comp = G * H: previous string impulse against string slack.
model.pairs = lambda z: (jnp.stack([z[ix[t]][8] for t in range(1, T)]),
                         jnp.stack([phi(z[ix[t]][4:8]) for t in range(1, T)]))


def _interp(a, b, n):
    return [a + (b - a) * k / (n - 1) for k in range(n)]


def initial_point(seed):
    rng = np.random.default_rng(seed)
    xi = _interp(x1, xM1, 11) + _interp(xM1, xM2, 6)[1:] + _interp(xM2, xT, 6)[1:]
    z = np.zeros(nz)
    z[ix[0]] = xi[0]
    for t in range(1, T):
        z[ix[t]] = np.concatenate([xi[t], [0.0]])
    for t in range(T - 1):
        z[iu[t]] = np.concatenate([1e-3 * rng.standard_normal(2), [1e-3]])
    return z


def from_calipso(rec):
    z = np.zeros(nz)
    for t in range(T):
        z[ix[t]] = rec["states"][t]
    for t in range(T - 1):
        z[iu[t]] = rec["actions"][t]
    return z
