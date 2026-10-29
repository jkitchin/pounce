"""Bunny-hop, ported from CALIPSO.jl examples/contact_implicit/bunnyhop.jl
and models/mountain_bike.jl (thowell/CALIPSO.jl @ 0e3bdcf, 2022-12-14).

Variables (0-based here; Julia is 1-based):
  q_i  in R6, i=0..N-1   (wheel1 xy, wheel2 xy, rider xy)
  u_i  in R2, i=0..N-2   (rider-wheel prismatic forces)
  lam_i in R1, i=0..N-3  (wheel-base link multiplier)
  eta_i in R3, i=0..N-3  (contact impulses: wheel1, wheel2, rider-height)

CALIPSO's source writes the contact complementarity eta .* d(q3) = 0 as
equality rows inside the dynamics block, with eta >= 0 and d(q) >= 0 as
cone (nonnegative) constraints. Here eta >= 0 and d(q) >= 0 are variable
bounds (d(q) is just q[1], q[3], q[5]-0.3), and the products are the
complementarity block that the route lowers.
"""

import jax.numpy as jnp
import numpy as np

from mpcc_nlp import Model

m_b, m1, m2 = 10.0, 1.0, 1.0
h = 0.2
gravity = jnp.array([0.0, -9.8])
r_wheel_base = 2.0
N = 10

iq = [np.arange(6) + 6 * i for i in range(N)]
iu = [60 + np.arange(2) + 2 * i for i in range(N - 1)]
il = [78 + i for i in range(N - 2)]
ie = [86 + np.arange(3) + 3 * i for i in range(N - 2)]
nz = 110

q0 = np.array([-r_wheel_base / 2, 0, r_wheel_base / 2, 0, 0, np.sqrt(1.5**2 - (r_wheel_base / 2) ** 2)])
q1 = q0 + 11 * h * np.array([1, 0, 1, 0, 1, 0.0])
usref = [-0.95 * np.sqrt(2) * 0.5 * m_b * 9.8 * np.ones(2) for _ in range(N - 1)]
qsref = [q0 + 11 * h * i * np.array([1, 0, 1, 0, 1, 0.0]) for i in range(N)]
Q1 = jnp.array([0, 1, 0, 1, 0, 0.1])
Q2 = jnp.array([0, 1, 0, 1, 0, 1e4])
R = 0.01


def control_forces(r1, r2, u):
    d = -r1 + r2
    r12 = d / jnp.sqrt(jnp.dot(d, d))
    return u * r12, -u * r12


def trans_del(m, r1, r2, r3):
    return (1 / h) * m * (r2 - r1) - (1 / h) * m * (r3 - r2)


def link(q):
    r = q[0:2] - q[2:4]
    return jnp.dot(r, r) - r_wheel_base**2


def Dc_T(q):
    t0 = q[0:2] - q[2:4]
    return 2 * jnp.concatenate([t0, -t0, jnp.zeros(2)])


def d(q):
    return jnp.array([q[1], q[3], q[5] - 0.3])


def Dd_T_eta(eta):
    return jnp.array([0, eta[0], 0, eta[1], 0, eta[2]])


def DEL(qa, qb, qc, um, up):
    r1m, r2m, r3m = qa[0:2], qa[2:4], qa[4:6]
    r1, r2, r3 = qb[0:2], qb[2:4], qb[4:6]
    r1p, r2p, r3p = qc[0:2], qc[2:4], qc[4:6]
    Fg = jnp.concatenate([m1 * gravity, m2 * gravity, m_b * gravity])
    f1m, f31m = control_forces(0.5 * (r1m + r1), 0.5 * (r3m + r3), um[0])
    f2m, f32m = control_forces(0.5 * (r2m + r2), 0.5 * (r3m + r3), um[1])
    f1p, f31p = control_forces(0.5 * (r1p + r1), 0.5 * (r3p + r3), up[0])
    f2p, f32p = control_forces(0.5 * (r2p + r2), 0.5 * (r3p + r3), up[1])
    Fm = Fg + jnp.concatenate([f1m, f2m, f31m + f32m])
    Fp = Fg + jnp.concatenate([f1p, f2p, f31p + f32p])
    return jnp.concatenate([trans_del(m1, r1m, r1, r1p), trans_del(m2, r2m, r2, r2p),
                            trans_del(m_b, r3m, r3, r3p)]) + (h / 2) * Fm + (h / 2) * Fp


def eq(z):
    rows = []
    for i in range(N - 2):
        qa, qb, qc = z[iq[i]], z[iq[i + 1]], z[iq[i + 2]]
        lam, eta = z[il[i]], z[ie[i]]
        rows.append(DEL(qa, qb, qc, z[iu[i]], z[iu[i + 1]]) + h * Dc_T(qb) * lam + h * Dd_T_eta(eta))
        rows.append(jnp.atleast_1d(link(qc)))
    rows.append(z[iq[0]] - q0)
    rows.append(z[iq[1]] - q1)
    rows.append(z[iq[4]][jnp.array([1, 3])] - jnp.array([1.1, 1.15]))
    return jnp.concatenate(rows)


def ineq(z):
    return jnp.zeros(0)


def comp(z):
    return jnp.concatenate([z[ie[i]] * d(z[iq[i + 2]]) for i in range(N - 2)])


def f(z):
    J = 0.0
    for i in range(N):
        dq = z[iq[i]] - qsref[i]
        Q = Q2 if i + 1 > 6 else Q1
        J += 0.5 * jnp.dot(dq * Q, dq)
        qq = z[iq[i]]
        J += 10 * (qq[4] - 0.5 * (qq[0] + qq[2])) ** 2
    for i in range(N - 1):
        du = z[iu[i]] - usref[i]
        J += 0.5 * R * jnp.dot(du, du)
        dq = z[iq[i]] - z[iq[i + 1]]
        J += 0.005 * jnp.dot(dq, dq)
    for i in range(N - 2):
        a = z[iq[i + 2]][::2] - 2 * z[iq[i + 1]][::2] + z[iq[i]][::2]
        J += 10 * jnp.dot(a, a)
    return J


lb = np.full(nz, -np.inf)
ub = np.full(nz, np.inf)
for i in range(N):
    lb[iq[i][1]] = 0.0
    lb[iq[i][3]] = 0.0
    lb[iq[i][5]] = 0.3
for i in range(N - 2):
    lb[ie[i]] = 0.0


def report(z):
    z = jnp.asarray(z)
    e = np.asarray(eq(z))
    c = np.asarray(comp(z))
    bnd = float(np.max(np.maximum(lb - np.asarray(z), 0)))
    return {"obj": float(f(z)), "eq_viol": float(np.max(np.abs(e))),
            "compl": float(np.max(np.abs(c))), "bound_viol": bnd,
            "violation": max(float(np.max(np.abs(e))), float(np.max(np.abs(c))), bnd)}


model = Model("bunnyhop", nz, f, eq, ineq, comp, lb, ub, report=report)
#: (G, H) with comp = G * H: contact impulse against gap, per step.
model.pairs = lambda z: (jnp.concatenate([z[ie[i]] for i in range(N - 2)]),
                         jnp.concatenate([d(z[iq[i + 2]]) for i in range(N - 2)]))


def initial_point(seed):
    rng = np.random.default_rng(seed)
    z = 10 * np.ones(nz)
    for i in range(N):
        z[iq[i]] = qsref[i] + 0.001 * np.abs(rng.standard_normal(6))
    for i in range(N - 1):
        z[iu[i]] = usref[i] + 0.001 * rng.standard_normal(2)
    for i in range(N - 2):
        z[ie[i]] = 9.8 * np.array([m1, m2, m_b])
    return z + 0.001 * rng.standard_normal(nz)
