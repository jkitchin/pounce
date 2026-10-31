"""Planar quadruped gait, ported from CALIPSO.jl
examples/contact_implicit/quadruped_gait.jl (thowell/CALIPSO.jl @ 0e3bdcf,
2022-12-14) and RoboDojo's planar quadruped (thowell/RoboDojo.jl @ b8af4d8,
src/robots/quadruped/model.jl, src/robots/integrator.jl,
src/robots/codegen.jl).

One step of a gait with a loop constraint: legs 1 and 3 (rear foot 1, front
foot 1) stay pinned to the ground; legs 2 and 4 follow an elliptical swing
reference after T_fix steps; the final state must equal the mirrored first
state shifted by half a stride.

Model (RoboDojo): q in R11 = (x, z, torso angle, 8 absolute joint angles).
RoboDojo's `codegen_dynamics` derives M = d2L/dqdot2 and the bias
C = (d2L/dqdot dq) qdot - dL/dq from its Lagrangian symbolically. Here both
are written in closed form from RoboDojo's own kinematic Jacobians: M = sum
of m J^T J per link plus rotational inertias, C = sum m J^T (Jdot qdot) +
gravity gradient. Deriving them by nested JAX autodiff instead is
equivalent, but the NLP Hessian then differentiates the Lagrangian four
times, batched over all 2302 variables, which ran out of memory (8.5 GB).
Both forms were checked against RoboDojo's codegen to 1e-13. The
integrator is RoboDojo's generic midpoint variational integrator with joint
friction 0.1 on the eight joints (`floating_base_dim = 3`). The example sets
friction_foot_world = 0.2.

Layout. State x_1 = (q_1, q_2) in R22; x_t = (q_t, q_{t+1}, gamma_{t-1},
x_1) in R48 for t >= 2 -- the previous normal impulse rides in the state for
the gamma- * phi row, and x_1 is carried forward for the loop constraint.
Action u_t in R28 = (u_ctrl[8], gamma[4], beta[8], eta[8]); beta_i =
(beta0, b) is foot i's friction budget and tangential impulse, eta_i =
(eta0, v) its slip bound and tangential velocity. T = 31.

Second-order cones -- what the source actually constrains. The source
declares eight 2-D cones per stage as u[8 + (i-1)*2 .+ (1:2)], i = 1..8,
i.e. on u[9:24] (1-based): (gamma1, gamma2), (gamma3, gamma4), beta_1..4,
eta_1, eta_2. eta_3 and eta_4 (u[25:28]) are in no cone, and the two gamma
pairs are. That looks like an index offset of 4 in the source (the friction
pairs start at u[13]), but it is what CALIPSO solves, so `cones="source"`
(the registered model) reproduces it; `cones="intended"` puts the cones on
beta_1..4 and eta_1..4 instead. A 2-D cone t >= |x| is lowered as
t^2 - x^2 >= 0 with a bound t >= 0 (drifting.py's "squared" lowering).

Complementarity. The source writes every product as an equality row of the
stage's contact constraints. Here:
  * gamma-_i * phi_i(q3) (t = 2..T-1, 4 per stage; both factors >= 0) are
    scalars in `comp`. The source does NOT add them at the last stage
    (`contact_constraints_equality_T` is defined but unused), and neither
    does this port.
  * For a friction pair whose beta and eta are both in a cone, beta.eta = 0
    is equivalent to the 2-row Jordan product beta o eta = 0 (self-dual
    cone), so the scalar beta.eta goes in `comp`.
  * For a pair whose eta is in no cone (pairs 3 and 4 under
    `cones="source"`), that equivalence does not hold, so its two Jordan
    rows stay equality rows in `eq`, exactly as the source has them.
  * model.extra["jordan"] is the source's full complementarity transcription
    for the `jordan_eq` route: the gamma- * phi rows plus the Jordan rows of
    every pair that `comp` carries as a scalar.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mpcc_nlp import Model

# ---------------------------------------------------------------- model data
nq, nc_feet = 11, 4
g = 9.81
m_torso = 4.713 + 4 * 0.696
m_thigh, m_calf = 1.013, 0.166
J_torso = 0.01683 + 4 * 0.696 * 0.183**2.0
J_thigh, J_calf = 0.00552, 0.00299
l_torso1 = l_torso2 = 0.183
l_thigh = l_calf = 0.2
lc_thigh = 0.5 * l_thigh - 0.00323
lc_calf = 0.5 * l_calf - 0.006435
friction_joint = 0.1
mu_foot = 0.2  # the example's override of friction_foot_world

# leg k (0-based): shoulder angle index, hip offset along the torso
LEG_IDX = (3, 5, 7, 9)
LEG_HIP = (l_torso1, l_torso1, -l_torso2, -l_torso2)


def hip(q, k):
    l = LEG_HIP[k]
    return jnp.array([q[0] + l * jnp.cos(q[2]), q[1] + l * jnp.sin(q[2])])


def thigh(q, k, length=l_thigh):
    t = q[LEG_IDX[k]]
    return hip(q, k) + jnp.array([length * jnp.sin(t), -length * jnp.cos(t)])


def calf(q, k, length=l_calf):
    t = q[LEG_IDX[k] + 1]
    return thigh(q, k) + jnp.array([length * jnp.sin(t), -length * jnp.cos(t)])


def foot(q, k):
    return calf(q, k)


def hip1(q):
    return hip(q, 0)


def hip2(q):
    return hip(q, 2)


def _hip_terms(q, k):
    """Hip position, Jacobian (2 x 11), and d2p/ds2 along qd per unit qd2^2."""
    l = LEG_HIP[k]
    c, s = jnp.cos(q[2]), jnp.sin(q[2])
    J = jnp.zeros((2, nq)).at[0, 0].set(1.0).at[1, 1].set(1.0)
    J = J.at[0, 2].set(-l * s).at[1, 2].set(l * c)
    return J, jnp.array([-l * c, -l * s])


def link_jacobian_and_bias(q, qd, k, which, length):
    """Jacobian J (2 x 11) and the velocity-product term Jdot(q, qd) qd of the
    thigh ('thigh', point at `length` along it) or calf ('calf', thigh end
    plus `length` along the calf) of leg k. RoboDojo's hand-written
    kinematic Jacobians, plus their time derivative along qd."""
    J, a_hip = _hip_terms(q, k)
    a = a_hip * qd[2] ** 2
    i = LEG_IDX[k]
    lt = length if which == "thigh" else l_thigh
    ct, st = jnp.cos(q[i]), jnp.sin(q[i])
    J = J.at[0, i].add(lt * ct).at[1, i].add(lt * st)
    a = a + jnp.array([-lt * st, lt * ct]) * qd[i] ** 2
    if which == "calf":
        cc, sc = jnp.cos(q[i + 1]), jnp.sin(q[i + 1])
        J = J.at[0, i + 1].add(length * cc).at[1, i + 1].add(length * sc)
        a = a + jnp.array([-length * sc, length * cc]) * qd[i + 1] ** 2
    return J, a


def _links(q, qd):
    out = []
    for k in range(4):
        i = LEG_IDX[k]
        out.append((m_thigh, J_thigh, i) + link_jacobian_and_bias(q, qd, k, "thigh", lc_thigh))
        out.append((m_calf, J_calf, i + 1) + link_jacobian_and_bias(q, qd, k, "calf", lc_calf))
    return out


def mass_matrix(q):
    """d2L/dqdot2 of RoboDojo's Lagrangian, in closed form: point-mass
    m J^T J per link plus each link's rotational inertia on its own
    (absolute) angle."""
    M = jnp.diag(jnp.zeros(nq).at[0].set(m_torso).at[1].set(m_torso).at[2].set(J_torso))
    for m, Ir, i, J, _ in _links(q, jnp.zeros(nq)):
        M = M + m * J.T @ J
        M = M.at[i, i].add(Ir)
    return M


def dynamics_bias(q, qd):
    """RoboDojo's C = (d2L/dqdot dq) qdot - dL/dq, in closed form: for point
    masses with constant rotational inertias this is sum m J^T (Jdot qdot)
    plus the gravity gradient sum m g J_z^T."""
    C = jnp.zeros(nq).at[1].set(m_torso * g)
    for m, _, _, J, a in _links(q, qd):
        C = C + m * J.T @ a + m * g * J[1]
    return C


# B^T in RoboDojo's input_jacobian (8 x 11), constant
_B = np.zeros((8, nq))
for r, (a, b) in enumerate([(2, 3), (3, 4), (2, 5), (5, 6), (2, 7), (7, 8), (2, 9), (9, 10)]):
    _B[r, a] = -1.0
    _B[r, b] = 1.0
B = jnp.asarray(_B)


def lagrangian_derivatives(q, v):
    return -dynamics_bias(q, v), mass_matrix(q) @ v


def robodojo_dynamics(h, q0, q1, u1, lam, q2):
    qm1, vm1 = 0.5 * (q0 + q1), (q1 - q0) / h
    qm2, vm2 = 0.5 * (q1 + q2), (q2 - q1) / h
    D1L1, D2L1 = lagrangian_derivatives(qm1, vm1)
    D1L2, D2L2 = lagrangian_derivatives(qm2, vm2)
    d = 0.5 * h * D1L1 + D2L1 + 0.5 * h * D1L2 - D2L2
    d = d + B.T @ u1 + lam
    return d.at[3:].add(-friction_joint * vm2[3:])


def foot_jacobian(q, k):
    return link_jacobian_and_bias(q, jnp.zeros(nq), k, "calf", l_calf)[0]


def signed_distance(q):
    return jnp.stack([foot(q, k)[1] for k in range(4)])


# -------------------------------------------------------------- problem data
T = 31
T_fix = 5
h = 0.015625
nx1, nxt, nu = 2 * nq, 2 * nq + nc_feet + 2 * nq, 8 + 4 + 8 + 8

PERM = np.zeros((nq, nq))
for r, c in enumerate([0, 1, 2, 5, 6, 3, 4, 9, 10, 7, 8]):
    PERM[r, c] = 1.0


def initial_configuration(t1, t2, t3):
    q = np.zeros(nq)
    q[3], q[4], q[7], q[8] = -t1, t2, -t1, t2
    q[1] = l_thigh * np.cos(q[3]) + l_calf * np.cos(q[4])
    q[9] = -t3
    q[10] = np.arccos((q[1] - l_thigh * np.cos(q[9])) / l_calf)
    q[5] = -t3
    q[6] = np.arccos((q[1] - l_thigh * np.cos(q[5])) / l_calf)
    return q


def ellipse_trajectory(x_start, x_goal, zh, n):
    a = 0.5 * (x_goal - x_start)
    x = np.linspace(x_start, x_goal, n)
    z = np.sqrt(np.maximum(0.0, zh**2 * (1.0 - (x - (x_start + a)) ** 2 / a**2)))
    return x, z


q1 = initial_configuration(np.pi / 4, np.pi / 4, np.pi / 3)
_f = jax.jit(lambda q: jnp.stack([foot(q, k) for k in range(4)]))
pr1, pr2, pf1, pf2 = (np.asarray(p) for p in _f(jnp.asarray(q1)))
stride = 2 * (pr1 - pr2)[0]
qT = PERM @ q1
qT[0] += 0.5 * stride
zh = 0.05


def _swing(p):
    xe, ze = ellipse_trajectory(p[0], p[0] + stride, zh, T - T_fix)
    xs = np.concatenate([np.full(T_fix, xe[0]), xe])
    zs = np.concatenate([np.full(T_fix, ze[0]), ze])
    return np.stack([xs, zs], axis=1)


pr1_ref = np.tile(pr1, (T, 1))
pf1_ref = np.tile(pf1, (T, 1))
pr2_ref = _swing(pr2)
pf2_ref = _swing(pf2)

_nxs = [nx1] + [nxt] * (T - 1)
_off = np.cumsum([0] + _nxs)
ix = [np.arange(_nxs[t]) + _off[t] for t in range(T)]
iu = [_off[-1] + np.arange(nu) + nu * t for t in range(T - 1)]
nz = int(_off[-1] + nu * (T - 1))

# action slices (0-based)
U_CTRL, U_GAMMA, U_BETA, U_ETA = slice(0, 8), slice(8, 12), slice(12, 20), slice(20, 28)


def pairs(u):
    """(beta_i, eta_i), i = 0..3, each a 2-vector."""
    return [(u[12 + 2 * i: 14 + 2 * i], u[20 + 2 * i: 22 + 2 * i]) for i in range(4)]


def cone_blocks(cones):
    """Start index (within u) of each 2-D cone block."""
    if cones == "source":
        return [8 + 2 * i for i in range(8)]  # u[9:24] in 1-based terms
    if cones == "intended":
        return [12 + 2 * i for i in range(8)]  # beta_1..4, eta_1..4
    raise ValueError(cones)


def pair_in_cones(cones):
    """Per friction pair: are both beta_i and eta_i cone-constrained?"""
    starts = set(cone_blocks(cones))
    return [(12 + 2 * i in starts) and (20 + 2 * i in starts) for i in range(4)]


def soc_product(a, b):
    return jnp.array([jnp.dot(a, b), a[0] * b[1] + b[0] * a[1]])


# ------------------------------------------------------------ stage functions
def quadruped_dyn(y, x, u):
    q1m, q2m = x[0:nq], x[nq:2 * nq]
    q2p, q3p = y[0:nq], y[nq:2 * nq]
    gamma = u[U_GAMMA]
    beta = u[U_BETA]
    lam = jnp.zeros(nq)
    for k in range(4):
        lam = lam + foot_jacobian(q2p, k).T @ jnp.array([beta[2 * k + 1], gamma[k]])
    return jnp.concatenate([q2p - q2m, robodojo_dynamics(h, q1m, q2p, u[U_CTRL], lam, q3p)])


def quadruped_dyn1(y, x, u):
    return jnp.concatenate([quadruped_dyn(y, x, u), y[22:26] - u[U_GAMMA], y[26:48] - x[0:22]])


def quadruped_dynt(y, x, u):
    return jnp.concatenate([quadruped_dyn(y, x, u), y[22:26] - u[U_GAMMA], y[26:48] - x[26:48]])


def contact_fc_vc(x, u):
    """The source's fc and vc rows: mu*gamma_i - beta_i0, and the foot's
    tangential velocity minus eta_i's second entry."""
    q2, q3 = x[0:nq], x[nq:2 * nq]
    v = (q3 - q2) / h
    gamma, beta, eta = u[U_GAMMA], u[U_BETA], u[U_ETA]
    vc = jnp.stack([(foot_jacobian(q3, k) @ v)[0] - eta[2 * k + 1] for k in range(4)])
    fc = mu_foot * gamma - beta[0::2]
    return jnp.concatenate([fc, vc])


def gamma_phi(x):
    return signed_distance(x[nq:2 * nq]) * x[22:26]


def kin_z(q, fn):
    return fn(q)[1]


def stage_cost(x, u, t):
    """t is 0-based. u is None at the last stage."""
    q = x[nq:2 * nq]
    J = 1e-3 * jnp.dot(q - qT, q - qT)
    J += (kin_z(q, hip1) - qT[1]) ** 2 + (kin_z(q, hip2) - qT[1]) ** 2
    if t > 0:
        J += jnp.sum((pr2_ref[t] - foot(q, 1)) ** 2) + jnp.sum((pf2_ref[t] - foot(q, 3)) ** 2)
    v = (q - x[0:nq]) / h
    J += 1e-3 * jnp.dot(v, v)
    if u is not None:
        J += 1e-2 * jnp.dot(u[U_CTRL], u[U_CTRL])
    return J


def pinned1(x, t):
    q = x[0:nq]
    return jnp.concatenate([pr1_ref[t] - foot(q, 0), pf1_ref[t] - foot(q, 2)])


def pinned2(x, t):
    q = x[0:nq]
    return jnp.concatenate([pr2_ref[t] - foot(q, 1), pf2_ref[t] - foot(q, 3)])


_P2 = np.kron(np.eye(2), PERM)


def loop(x):
    e = x[26:48] - _P2 @ x[0:22]
    return jnp.concatenate([e[1:11], e[12:22]])


# --------------------------------------------------------------- whole model
def make_model(cones="source"):
    starts = cone_blocks(cones)
    in_cone = pair_in_cones(cones)

    def eq(z):
        # The Lagrangian-derived dynamics are traced once and vmapped over
        # the stages; tracing them per stage made one evaluation take minutes.
        X = jnp.stack([z[ix[t]][0:2 * nq] for t in range(T - 1)])
        Y = jnp.stack([z[ix[t + 1]][0:2 * nq] for t in range(T - 1)])
        U = jnp.stack([z[iu[t]] for t in range(T - 1)])
        rows = [jax.vmap(quadruped_dyn)(Y, X, U).ravel(),
                jax.vmap(contact_fc_vc)(X, U).ravel()]
        for t in range(T - 1):
            x, u, y = z[ix[t]], z[iu[t]], z[ix[t + 1]]
            rows.append(y[22:26] - u[U_GAMMA])
            rows.append(y[26:48] - (x[0:22] if t == 0 else x[26:48]))
            rows.append(pinned1(x, t))
            if t < T_fix:  # source: pinned2 for t = 1..T_fix (1-based)
                rows.append(pinned2(x, t))
            if t == 0:
                rows.append(x[nq:2 * nq] - q1)
            for (b, e), ok in zip(pairs(u), in_cone):
                if not ok:
                    rows.append(soc_product(b, e))
        xl = z[ix[T - 1]]
        rows.append(loop(xl))
        rows.append(jnp.atleast_1d(xl[nq] - qT[0]))
        return jnp.concatenate(rows)

    def ineq(z):
        rows = []
        for t in range(T):
            rows.append(signed_distance(z[ix[t]][nq:2 * nq]))
        for t in range(T - 1):
            u = z[iu[t]]
            rows.append(jnp.stack([u[s] ** 2 - u[s + 1] ** 2 for s in starts]))
        return jnp.concatenate(rows)

    def comp(z):
        rows = []
        for t in range(T - 1):
            x, u = z[ix[t]], z[iu[t]]
            if t > 0:
                rows.append(gamma_phi(x))
            sc = [jnp.dot(b, e) for (b, e), ok in zip(pairs(u), in_cone) if ok]
            if sc:
                rows.append(jnp.stack(sc))
        return jnp.concatenate(rows)

    def jordan(z):
        rows = []
        for t in range(T - 1):
            x, u = z[ix[t]], z[iu[t]]
            if t > 0:
                rows.append(gamma_phi(x))
            for (b, e), ok in zip(pairs(u), in_cone):
                if ok:
                    rows.append(soc_product(b, e))
        return jnp.concatenate(rows)

    def f(z):
        J = 0.0
        for t in range(T):
            J += stage_cost(z[ix[t]], z[iu[t]] if t < T - 1 else None, t)
        return J

    lb = np.full(nz, -np.inf)
    ub = np.full(nz, np.inf)
    for t in range(T - 1):
        lb[iu[t][U_GAMMA]] = 0.0  # the source's nonnegative gamma rows
        for s in starts:
            lb[iu[t][s]] = 0.0  # cone head t >= 0 for the squared lowering

    m = Model(f"quadruped[{cones}]", nz, f, eq, ineq, comp, lb, ub, extra={"jordan": jordan})

    def report(zv):
        zj = jnp.asarray(zv)
        zn = np.asarray(zv, dtype=float)
        e = np.asarray(eq(zj))
        jr = np.asarray(jordan(zj))
        phi = np.concatenate([np.asarray(signed_distance(zj[ix[t]][nq:2 * nq])) for t in range(T)])
        cone_v = 0.0
        for t in range(T - 1):
            u = zn[iu[t]]
            for s in starts:
                cone_v = max(cone_v, abs(u[s + 1]) - u[s])
        bnd = float(np.max(np.maximum(lb - zn, 0)))
        eqv = float(np.max(np.abs(e)))
        cv = float(np.max(np.abs(jr))) if jr.size else 0.0
        pv = float(np.max(np.maximum(-phi, 0)))
        out = {"obj": float(f(zj)), "eq_viol": eqv, "compl": cv, "cone_viol": max(cone_v, 0.0),
               "phi_viol": pv, "bound_viol": bnd}
        out["violation"] = max(eqv, cv, out["cone_viol"], pv, bnd)
        return out

    m.report = report
    # Its complementarity is mostly between second-order cones; the NCP-pair
    # stationarity classes in stationarity.py do not apply (reported n/a).
    m.soc_complementarity = True
    return m


model = make_model("source")


def initial_point(seed):
    """The source's guess: configurations linearly interpolated from q1 to qT
    over T+1 points, x_t = (q_t, q_{t+1}); actions and the carried gamma at
    max(0, 1e-3 randn); the carried x_1 slot set to x_{t-1} as in the source
    (not to x_1). Julia's RNG stream is not reproduced."""
    rng = np.random.default_rng(seed)
    qi = [q1 + (qT - q1) * k / T for k in range(T + 1)]
    xi = [np.concatenate([qi[t], qi[t + 1]]) for t in range(T)]
    us = [np.maximum(0.0, 1e-3 * rng.standard_normal(nu)) for _ in range(T - 1)]
    z = np.zeros(nz)
    z[ix[0]] = xi[0]
    for t in range(1, T):
        z[ix[t]] = np.concatenate([xi[t], np.maximum(0.0, 1e-3 * rng.standard_normal(nc_feet)), xi[t - 1]])
    for t in range(T - 1):
        z[iu[t]] = us[t]
    return z


def from_calipso(rec):
    z = np.zeros(nz)
    for t in range(T):
        z[ix[t]] = rec["states"][t]
    for t in range(T - 1):
        z[iu[t]] = rec["actions"][t]
    return z
