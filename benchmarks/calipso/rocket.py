"""Rocket soft landing with state-triggered keep-out zones, ported from
CALIPSO.jl examples/state_triggered/rocket_landing.jl
(thowell/CALIPSO.jl @ 0e3bdcf, 2022-12-14).

x_t = (position, velocity) in R6, t = 0..T-1; u_t in R11, t = 0..T-2:
  u[0:3]   thrust
  u[3:7]   g1+, g1-, c1+, c1-   split of g1 = a - p_x and c1 = p_z - b
  u[7:11]  g2+, g2-, c2+, c2-   split of g2 = p_x - c and c2 = p_z - d

The trigger logic "g > 0 => c >= 0" (left of x = a, or right of x = c, the
rocket must stay above the elevated terrain) is CALIPSO's complementarity
formulation of the paper's eq. (47): g = g+ - g-, c = c+ - c-, all four
>= 0, and g+ * c- = 0. CALIPSO writes the two products as equality rows of
`stc_con`; here they are the complementarity block the routes lower.
`stc_con` applies at t = 2..T-1 (1-based), so u_1's split variables are
free apart from their bounds, exactly as in the source.
"""

import jax.numpy as jnp
import numpy as np

from mpcc_nlp import Model, generic_report

T = 51
nx, nu = 6, 11
h = 0.05
mass, gravity = 1.0, -9.81
x1 = np.array([-5.0, 0.0, 5.0, 0.0, 0.0, 0.0])
xT = np.zeros(6)
F_min = np.array([-10.0, -10.0, 0.0])
F_max = np.array([10.0, 10.0, 20.0])
a, b, c, d = -0.5, 3.0, 0.3, 3.0

ix = [np.arange(nx) + nx * t for t in range(T)]
iu = [T * nx + np.arange(nu) + nu * t for t in range(T - 1)]
nz = T * nx + (T - 1) * nu


def rocket(x, u):
    return jnp.concatenate([x[3:6], jnp.array([0.0, 0.0, gravity]) + u[0:3] / mass])


def midpoint_implicit(y, x, u):
    return y - (x + h * rocket(0.5 * (x + y), u))


def stc_rows(x, u):
    g1p, g1m, c1p, c1m, g2p, g2m, c2p, c2m = (u[k] for k in range(3, 11))
    g1 = -x[0] + a
    c1 = x[2] - b
    g2 = x[0] - c
    c2 = x[2] - d
    return jnp.array([g1p - g1m - g1, c1p - c1m - c1, g2p - g2m - g2, c2p - c2m - c2])


def eq(z):
    rows = []
    for t in range(T - 1):
        rows.append(midpoint_implicit(z[ix[t + 1]], z[ix[t]], z[iu[t]]))
    rows.append(z[ix[0]] - x1)
    for t in range(1, T - 1):
        rows.append(stc_rows(z[ix[t]], z[iu[t]]))
    rows.append(z[ix[T - 1]] - xT)
    return jnp.concatenate(rows)


def comp(z):
    out = []
    for t in range(1, T - 1):
        u = z[iu[t]]
        out.append(jnp.array([u[3] * u[6], u[7] * u[10]]))
    return jnp.concatenate(out)


def f(z):
    J = 0.0
    for t in range(T):
        x = z[ix[t]]
        J += jnp.dot(x[0:3] - xT[0:3], x[0:3] - xT[0:3]) + 0.1 * jnp.dot(x[3:6], x[3:6])
        if t < T - 1:
            u = z[iu[t]]
            J += 0.1 * jnp.dot(u[0:3], u[0:3])
    return J


lb = np.full(nz, -np.inf)
ub = np.full(nz, np.inf)
for t in range(T - 1):
    lb[iu[t][0:3]] = F_min
    ub[iu[t][0:3]] = F_max
    lb[iu[t][3:11]] = 0.0

model = Model("rocket", nz, f, eq, lambda z: jnp.zeros(0), comp, lb, ub)
#: (G, H) with comp = G * H: trigger positive part g+ against the
#: constraint's negative part c-, for both zones at every step t = 2..T-1.
model.pairs = lambda z: (
    jnp.concatenate([jnp.array([z[iu[t]][3], z[iu[t]][7]]) for t in range(1, T - 1)]),
    jnp.concatenate([jnp.array([z[iu[t]][6], z[iu[t]][10]]) for t in range(1, T - 1)]))
_generic = generic_report(model)


def report(z):
    """Generic violation plus the quantity the paper's figure is about: how
    far the trajectory enters a keep-out zone (left of a or right of c and
    below the terrain), read off the positions alone."""
    r = _generic(z)
    zn = np.asarray(z, dtype=float)
    pen = 0.0
    for t in range(T):
        px, pz = zn[ix[t][0]], zn[ix[t][2]]
        if px < a:
            pen = max(pen, b - pz)
        if px > c:
            pen = max(pen, d - pz)
    r["keepout_viol"] = max(pen, 0.0)
    r["violation"] = max(r["violation"], r["keepout_viol"])
    return r


model.report = report


def initial_point(seed):
    """The source's guess: linear interpolation of the states with a constant
    velocity, hover thrust, and the split variables at their exact values
    for that interpolation. It has no random part, so every seed is the
    same start."""
    z = np.zeros(nz)
    xs = [x1 + (xT - x1) * t / (T - 1) for t in range(T)]
    for t in range(T):
        xs[t] = xs[t].copy()
        xs[t][3:6] = (xT[0:3] - x1[0:3]) / ((0.05 / 2) * T)
        z[ix[t]] = xs[t]
    for t in range(T - 1):
        u = np.zeros(nu)
        u[0:3] = [0, 0, 9.8]
        x = xs[t]
        for base, val in ((3, -x[0] + a), (5, x[2] - b), (7, x[0] - c), (9, x[2] - d)):
            u[base], u[base + 1] = (val, 0.0) if val >= 0 else (0.0, -val)
        z[iu[t]] = u
    return z


def from_calipso(rec):
    z = np.zeros(nz)
    for t in range(T):
        z[ix[t]] = rec["states"][t]
    for t in range(T - 1):
        z[iu[t]] = rec["actions"][t]
    return z
