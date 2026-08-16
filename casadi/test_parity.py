#!/usr/bin/env python3
"""Parity checks for the POUNCE CasADi plugin, against the bundled ipopt.

Run with the plugin's directory on CasADi's search path::

    make test          # or: CASADIPATH=$PWD python3 test_parity.py

Every check compares POUNCE against `nlpsol(..., 'ipopt', ...)` on the
same model, so a failure says "the two solvers disagree", not "the
number moved".
"""

import sys

import casadi as ca
import numpy as np

QUIET_POUNCE = {"pounce": {"print_level": 0}, "print_time": False}
QUIET_IPOPT = {"ipopt": {"print_level": 0}, "print_time": False}

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def close(a, b, tol=1e-6):
    return float(ca.norm_inf(ca.DM(a) - ca.DM(b))) < tol


def rosenbrock_nlp():
    """MX Rosenbrock with a parametric circle constraint."""
    x = ca.MX.sym("x", 2)
    p = ca.MX.sym("p")
    f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    g = x[0] ** 2 + x[1] ** 2 - p
    return {"x": x, "p": p, "f": f, "g": g}


def test_mx_with_parameters():
    nlp = rosenbrock_nlp()
    kw = dict(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    a = ca.nlpsol("a", "pounce", nlp, QUIET_POUNCE)(**kw)
    b = ca.nlpsol("b", "ipopt", nlp, QUIET_IPOPT)(**kw)
    check("MX + parameters: primal", close(a["x"], b["x"], 1e-6), f"x={a['x'].T}")
    check("MX + parameters: objective", close(a["f"], b["f"], 1e-8))


def test_multipliers_with_active_bound():
    nlp = rosenbrock_nlp()
    kw = dict(
        x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0, lbx=[0.95, -ca.inf], ubx=ca.inf
    )
    a = ca.nlpsol("a", "pounce", nlp, QUIET_POUNCE)(**kw)
    b = ca.nlpsol("b", "ipopt", nlp, QUIET_IPOPT)(**kw)
    check("bound multipliers", close(a["lam_x"], b["lam_x"], 1e-5), f"lam_x={a['lam_x'].T}")
    check("constraint multipliers", close(a["lam_g"], b["lam_g"], 1e-5))


def test_solution_map_derivative():
    """dx*/dp, inherited from `Nlpsol` — the base class differentiates
    any plugin through the KKT system, so this is a real check that the
    solution and its multipliers are consistent."""
    nlp = rosenbrock_nlp()
    p = ca.MX.sym("p")

    def jac(plugin, opts):
        S = ca.nlpsol("S", plugin, nlp, opts)
        r = S(x0=[0.5, 0.5], p=p, lbg=-ca.inf, ubg=0)
        return ca.Function("J", [p], [ca.jacobian(r["x"], p)])(1.5)

    a = jac("pounce", QUIET_POUNCE)
    b = jac("ipopt", QUIET_IPOPT)
    check("dx*/dp", close(a, b, 1e-5), f"{a.T}")


def test_opti():
    opti = ca.Opti()
    y = opti.variable(2)
    par = opti.parameter()
    opti.minimize((1 - y[0]) ** 2 + 100 * (y[1] - y[0] ** 2) ** 2)
    opti.subject_to(y[0] ** 2 + y[1] ** 2 <= par)
    opti.set_value(par, 1.5)
    opti.set_initial(y, [0.5, 0.5])
    opti.solver("pounce", {"print_time": False}, {"print_level": 0})
    sol = opti.solve()
    check(
        "Opti",
        sol.stats()["return_status"] == "Solve_Succeeded",
        f"x={sol.value(y)}",
    )


def test_stats():
    nlp = rosenbrock_nlp()
    S = ca.nlpsol("S", "pounce", nlp, QUIET_POUNCE)
    S(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    st = S.stats()
    wanted = {"inf_pr", "inf_du", "mu", "d_norm", "regularization_size",
              "obj", "alpha_pr", "alpha_du", "ls_trials"}
    check("stats: success flag", st["success"] is True)
    check("stats: iter_count", st["iter_count"] > 0, f"{st['iter_count']} iterations")
    check("stats: iterations dict", wanted <= set(st["iterations"]))
    check(
        "stats: per-iteration trace is populated",
        len(st["iterations"]["inf_pr"]) > 1,
    )


def test_iteration_callback():
    """CasADi's `iteration_callback` needs live iterates from the solver.
    Stock Ipopt only provides them in a specially built binary; POUNCE
    serves them through `GetIpoptCurrentIterate`."""
    nlp = rosenbrock_nlp()

    class Recorder(ca.Callback):
        def __init__(self):
            ca.Callback.__init__(self)
            self.xs = []
            self.construct("Recorder", {})

        def get_n_in(self):
            return ca.nlpsol_n_out()

        def get_n_out(self):
            return 1

        def get_name_in(self, i):
            return ca.nlpsol_out(i)

        def get_sparsity_in(self, i):
            name = ca.nlpsol_out(i)
            sizes = {"f": 1, "x": 2, "g": 1, "lam_x": 2, "lam_g": 1, "lam_p": 1}
            return ca.Sparsity.dense(sizes[name], 1) if name in sizes else ca.Sparsity(0, 0)

        def eval(self, arg):
            self.xs.append(float(arg[ca.nlpsol_out().index("x")][0]))
            return [0]

    cb = Recorder()
    opts = dict(QUIET_POUNCE)
    opts["iteration_callback"] = cb
    S = ca.nlpsol("S", "pounce", nlp, opts)
    r = S(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    moved = len(set(cb.xs)) > 1
    check("iteration_callback fires", len(cb.xs) > 0, f"{len(cb.xs)} iterations")
    check("iteration_callback sees live iterates", moved)
    check("callback run still converges", close(r["x"][0], 0.907234, 1e-4))


def test_warm_start():
    nlp = rosenbrock_nlp()
    cold = ca.nlpsol("cold", "pounce", nlp, QUIET_POUNCE)
    r1 = cold(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    n_cold = cold.stats()["iter_count"]

    warm_opts = {
        "print_time": False,
        "pounce": {
            "print_level": 0,
            "warm_start_init_point": "yes",
            "mu_init": 1e-6,
        },
    }
    warm = ca.nlpsol("warm", "pounce", nlp, warm_opts)
    r2 = warm(
        x0=r1["x"], lam_g0=r1["lam_g"], lam_x0=r1["lam_x"],
        p=1.55, lbg=-ca.inf, ubg=0,
    )
    n_warm = warm.stats()["iter_count"]
    ref = ca.nlpsol("ref", "ipopt", nlp, QUIET_IPOPT)(
        x0=[0.5, 0.5], p=1.55, lbg=-ca.inf, ubg=0
    )
    check("warm start: same answer", close(r2["x"], ref["x"], 1e-5))
    check(
        "warm start: fewer iterations than cold",
        n_warm <= n_cold,
        f"{n_warm} warm vs {n_cold} cold",
    )


def test_limited_memory_and_nonlinear_variables():
    """A model whose variables mostly enter linearly: `pass_nonlinear_variables`
    hands POUNCE the nonlinear subset (gh#624) so the L-BFGS approximation
    spans only those."""
    n_lin = 20
    x = ca.MX.sym("x", 2 + n_lin)
    f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2 + ca.sum1(x[2:])
    g = ca.vertcat(x[0] ** 2 + x[1] ** 2 - 1.5, ca.sum1(x[2:]) - 1)
    nlp = {"x": x, "f": f, "g": g}
    x0 = [0.5, 0.5] + [0.1] * n_lin
    kw = dict(x0=x0, lbx=-5, ubx=5, lbg=[-ca.inf, 0], ubg=[0, 0])

    base = {"print_time": False,
            "pounce": {"print_level": 0, "hessian_approximation": "limited-memory"}}
    masked = dict(base)
    masked["pass_nonlinear_variables"] = True

    a = ca.nlpsol("a", "pounce", nlp, base)(**kw)
    b = ca.nlpsol("b", "pounce", nlp, masked)(**kw)
    c = ca.nlpsol("c", "ipopt", nlp, {
        "print_time": False,
        "ipopt": {"print_level": 0, "hessian_approximation": "limited-memory"},
    })(**kw)
    check("L-BFGS masked == unmasked", close(a["x"], b["x"], 1e-4))
    check("L-BFGS masked == ipopt", close(b["x"], c["x"], 1e-4), f"f={float(b['f']):.6f}")


def test_nmpc_feedback_gain_is_not_silently_zero():
    """The sensitivity of a *bounded* variable, which is where CasADi's
    solution-map derivative has a trap: an interior-point solve leaves a
    residual ~1e-12 multiplier on bounds it never touched, and the
    derivative reads any nonzero bound multiplier as an active constraint,
    zeroing that variable's whole row. The plugin clips demonstrably
    inactive multipliers by default, so the gain is right; the check is
    against a re-solve, which cannot be fooled the same way."""
    Nh, dt = 20, 0.05
    X, U = ca.MX.sym("X", 2, Nh + 1), ca.MX.sym("U", 1, Nh)
    x0p = ca.MX.sym("x0p", 2)
    cost, cons = 0, [X[:, 0] - x0p]
    for k in range(Nh):
        cons.append(X[:, k + 1] - ca.vertcat(
            X[0, k] + dt * X[1, k],
            X[1, k] + dt * (U[0, k] - 0.1 * X[1, k] * ca.fabs(X[1, k]))))
        cost += X[0, k]**2 + 0.1 * X[1, k]**2 + 0.01 * U[0, k]**2
    cost += 10 * (X[0, Nh]**2 + X[1, Nh]**2)
    nlp = {"x": ca.vertcat(ca.vec(X), ca.vec(U)), "p": x0p,
           "f": cost, "g": ca.vertcat(*cons)}
    nx, iu0 = 2 * (Nh + 1) + Nh, 2 * (Nh + 1)
    args = dict(lbg=0, ubg=0,
                lbx=[-ca.inf] * (2 * (Nh + 1)) + [-2.0] * Nh,
                ubx=[ca.inf] * (2 * (Nh + 1)) + [2.0] * Nh)
    opts = {"print_time": False, "pounce": {"print_level": 0, "tol": 1e-11}}

    S = ca.nlpsol("S", "pounce", nlp, opts)
    p0, eps = ca.DM([0.05, 0.0]), 1e-4
    u = lambda pv: float(S(x0=ca.DM.zeros(nx), p=pv, **args)["x"][iu0])
    truth = (u(p0 + ca.DM([eps, 0])) - u(p0 - ca.DM([eps, 0]))) / (2 * eps)

    ps = ca.MX.sym("p", 2)
    sol = S(x0=ca.DM.zeros(nx), p=ps, **args)
    analytic = float(ca.Function("J", [ps], [ca.jacobian(sol["x"][iu0], ps)])(p0)[0])
    check("NMPC feedback gain vs re-solve",
          abs(analytic - truth) < 1e-3 * max(1.0, abs(truth)),
          f"analytic {analytic:.6f} vs re-solve {truth:.6f}")

    # And the escape hatch reproduces the Ipopt-plugin default.
    unclipped = ca.nlpsol("U", "pounce", nlp, dict(opts, clip_inactive_lam=False))
    sol_u = unclipped(x0=ca.DM.zeros(nx), p=ps, **args)
    zeroed = float(ca.Function("J", [ps], [ca.jacobian(sol_u["x"][iu0], ps)])(p0)[0])
    check("clip_inactive_lam=False restores ipopt-plugin behaviour",
          abs(zeroed) < 1e-9, f"{zeroed:.3e}")


def test_active_set_sqp_algorithm():
    """`algorithm=active-set-sqp` is POUNCE-specific and reachable through
    the option dict; it must agree with the interior-point default."""
    nlp = rosenbrock_nlp()
    kw = dict(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    ipm = ca.nlpsol("ipm", "pounce", nlp, QUIET_POUNCE)(**kw)
    sqp_opts = {"print_time": False,
                "pounce": {"print_level": 0, "algorithm": "active-set-sqp"}}
    sqp = ca.nlpsol("sqp", "pounce", nlp, sqp_opts)(**kw)
    check("active-set-sqp agrees with the IPM", close(ipm["x"], sqp["x"], 1e-6),
          f"x={sqp['x'].T}")


def test_working_set_carries_between_calls():
    """`warm_start_from_previous` hands the active-set SQP the working set its
    last call ended on. The check is that it engages, and that engaging it
    changes nothing about the answer — the working set is a starting guess for
    the QP, not a constraint on the solution."""
    mc, mp_, L_, g_ = 1.0, 0.2, 0.5, 9.81

    def cartpole(s, u):
        th, dx, dth = s[1], s[2], s[3]
        sth, cth = ca.sin(th), ca.cos(th)
        den = mc + mp_ * sth**2
        return ca.vertcat(dx, dth,
                          (u + mp_ * sth * (L_ * dth**2 + g_ * cth)) / den,
                          (-u * cth - mp_ * L_ * dth**2 * cth * sth
                           - (mc + mp_) * g_ * sth) / (L_ * den))

    def rk4(s, u, h):
        k1 = cartpole(s, u); k2 = cartpole(s + h/2*k1, u)
        k3 = cartpole(s + h/2*k2, u); k4 = cartpole(s + h*k3, u)
        return s + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    N, h = 25, 0.04
    S, U = ca.MX.sym("S", 4, N + 1), ca.MX.sym("U", 1, N)
    s0 = ca.MX.sym("s0", 4)
    cost, cons = 0, [S[:, 0] - s0]
    for k in range(N):
        cons.append(S[:, k+1] - rk4(S[:, k], U[0, k], h))
        cost += (10*S[1, k]**2 + S[0, k]**2
                 + 0.1*(S[2, k]**2 + S[3, k]**2) + 0.01*U[0, k]**2)
    cost += 100 * (S[1, N]**2 + S[3, N]**2)
    nlp = {"x": ca.vertcat(ca.vec(S), ca.vec(U)), "p": s0,
           "f": cost, "g": ca.vertcat(*cons)}
    nx = 4 * (N + 1) + N
    # Tight force limits, so the control saturates and the active set is
    # something the QP has to work for.
    args = dict(lbg=0, ubg=0,
                lbx=[-ca.inf] * (4 * (N + 1)) + [-2.5] * N,
                ubx=[ca.inf] * (4 * (N + 1)) + [2.5] * N)

    def run(carry):
        opts = {"print_time": False, "pounce": {
            "print_level": 0, "tol": 1e-6, "algorithm": "active-set-sqp",
            "warm_start_init_point": "yes", "mu_init": 1e-6}}
        if carry:
            opts["warm_start_from_previous"] = True
        S_ = ca.nlpsol("S", "pounce", nlp, opts)
        state, prev, us, reused = ca.DM([0.0, 0.8, 0.0, 0.0]), None, [], 0
        for _ in range(12):
            prev = (S_(x0=ca.DM.zeros(nx), p=state, **args) if prev is None else
                    S_(x0=prev["x"], lam_g0=prev["lam_g"], lam_x0=prev["lam_x"],
                       p=state, **args))
            reused += bool(S_.stats().get("warm_started_working_set"))
            u0 = float(prev["x"][4 * (N + 1)])
            us.append(u0)
            state = ca.DM(np.array(rk4(state, u0, h)).ravel())
        return np.array(us), reused

    plain, reused_off = run(False)
    carried, reused_on = run(True)
    check("working set is not carried by default", reused_off == 0, f"{reused_off} reuses")
    check("working set carries between calls", reused_on >= 10, f"{reused_on}/12 reuses")
    check("carrying it does not change the trajectory",
          float(np.abs(plain - carried).max()) < 1e-6,
          f"max|Δu0| = {np.abs(plain - carried).max():.2e}")


def test_a_raising_model_fails_the_solve_not_the_process():
    """POUNCE is Rust behind a C API, and an exception unwinding out of an
    oracle callback into Rust frames aborts the process outright. A model with
    a `casadi.Callback` that raises — or a Ctrl-C mid-solve — must therefore be
    converted at the boundary, not propagated through it. Ipopt's plugin
    reports `Invalid_Number_Detected` here; so should this one, and the process
    has to still be alive to say so."""

    class Boom(ca.Callback):
        def __init__(self, trip):
            ca.Callback.__init__(self)
            self.n, self.trip = 0, trip
            self.construct("boom", {"enable_fd": True})

        def get_n_in(self): return 1
        def get_n_out(self): return 1
        def get_sparsity_in(self, i): return ca.Sparsity.dense(2, 1)
        def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)

        def eval(self, arg):
            self.n += 1
            if self.n >= self.trip:
                raise RuntimeError("boom: the user's model raised")
            x = arg[0]
            return [(1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2]

    cb = Boom(trip=25)
    x = ca.MX.sym("x", 2)
    S = ca.nlpsol("S", "pounce", {"x": x, "f": cb(x)},
                  {"print_time": False, "pounce": {"print_level": 0}})
    try:
        S(x0=[0.5, 0.5])
        survived, status = True, S.stats()["return_status"]
    except Exception as exc:                     # a clean exception is fine too
        survived, status = True, type(exc).__name__
    check("a raising oracle does not abort the process", survived, status)


def test_iteration_callback_can_interrupt():
    """A KeyboardInterrupt raised inside `iteration_callback` has to stop the
    solve rather than unwind through POUNCE."""
    nx = 2

    class Stopper(ca.Callback):
        def __init__(self):
            ca.Callback.__init__(self)
            self.n = 0
            self.construct("stopper", {})

        def get_n_in(self): return ca.nlpsol_n_out()
        def get_n_out(self): return 1
        def get_name_in(self, i): return ca.nlpsol_out(i)

        def get_sparsity_in(self, i):
            d = {"f": 1, "x": nx, "g": 0, "lam_x": nx,
                 "lam_g": 0, "lam_p": 0}.get(ca.nlpsol_out(i), 0)
            return ca.Sparsity.dense(d, 1) if d else ca.Sparsity(0, 0)

        def eval(self, arg):
            self.n += 1
            if self.n >= 3:
                raise KeyboardInterrupt("user pressed Ctrl-C")
            return [0]

    x = ca.MX.sym("x", nx)
    S = ca.nlpsol("S", "pounce", {"x": x, "f": (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2},
                  {"print_time": False, "iteration_callback": Stopper(),
                   "pounce": {"print_level": 0}})
    try:
        S(x0=[-1.2, 1.0])
        outcome = S.stats()["return_status"]
    except KeyboardInterrupt:
        outcome = "KeyboardInterrupt"
    check("an interrupting callback stops the solve",
          outcome in ("User_Requested_Stop", "KeyboardInterrupt"), outcome)


def test_lam_p_matches_ipopt_and_the_envelope_theorem():
    """`lam_p` is computed by CasADi's `Nlpsol` base class, not by the plugin,
    but it is a promised output and worth pinning: it must match Ipopt, and it
    must match a finite difference of the optimal objective. Note the sign —
    CasADi negates it (`nlpsol.cpp`: `casadi_scal(np_, -1., d_nlp->lam_p)`), so
    `lam_p = -df*/dp`, not `+`."""
    x, p = ca.MX.sym("x", 2), ca.MX.sym("p", 2)
    nlp = {"x": x, "p": p,
           "f": (x[0] - p[0])**2 + (x[1] - p[1])**2 + 0.1 * x[0] * x[1],
           "g": x[0]**2 + x[1]**2 - 1}
    kw = dict(x0=[0.1, 0.1], lbg=-ca.inf, ubg=0)
    pv, eps = ca.DM([2.0, 1.0]), 1e-6

    def lam_p_of(plugin, key):
        S = ca.nlpsol("S", plugin, nlp,
                      {"print_time": False, key: {"print_level": 0, "tol": 1e-12}})
        r = S(p=pv, **kw)
        fd = []
        for j in range(2):
            d = ca.DM.zeros(2)
            d[j] = eps
            fd.append((float(S(p=pv + d, **kw)["f"]) - float(S(p=pv - d, **kw)["f"]))
                      / (2 * eps))
        return np.array(r["lam_p"]).ravel(), np.array(fd)

    lam_pounce, fd = lam_p_of("pounce", "pounce")
    lam_ipopt, _ = lam_p_of("ipopt", "ipopt")
    check("lam_p matches ipopt", np.abs(lam_pounce - lam_ipopt).max() < 1e-7,
          f"{lam_pounce}")
    check("lam_p is -df*/dp (CasADi's sign)",
          np.abs(lam_pounce + fd).max() < 1e-5,
          f"lam_p {lam_pounce} vs -df*/dp {-fd}")


def test_threaded_map_matches_serial():
    """CasADi batches solves with `Function.map(N, "thread")`, giving each
    worker its own memory object. The plugin keeps every piece of per-solve
    state there — buffers, the iteration trace, the carried working set — so
    the batch must reproduce the serial answers exactly."""
    x, p = ca.MX.sym("x", 2), ca.MX.sym("p")
    nlp = {"x": x, "p": p,
           "f": (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2,
           "g": x[0]**2 + x[1]**2 - p}
    S = ca.nlpsol("S", "pounce", nlp,
                  {"print_time": False, "pounce": {"print_level": 0, "tol": 1e-9}})
    n = 24
    P = ca.DM(np.linspace(1.2, 2.0, n)).T
    X0 = ca.repmat(ca.DM([0.5, 0.5]), 1, n)
    serial = ca.hcat([S(x0=X0[:, i], p=P[0, i], lbg=-ca.inf, ubg=0)["x"]
                      for i in range(n)])
    try:
        batched = S.map(n, "thread", 8)(x0=X0, p=P, lbg=-ca.inf, ubg=0)["x"]
    except Exception as exc:                 # no thread support in this build
        check("threaded map", False, f"{type(exc).__name__}: {exc}")
        return
    err = float(ca.norm_inf(batched - serial))
    check("threaded map matches serial", err == 0.0, f"max|Δx| = {err:.2e}")


def test_option_pass_through():
    nlp = rosenbrock_nlp()
    S = ca.nlpsol("S", "pounce", nlp, {
        "print_time": False,
        "pounce": {"print_level": 0, "max_iter": 2},
    })
    S(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    st = S.stats()
    check(
        "options reach the solver (max_iter=2)",
        st["return_status"] == "Maximum_Iterations_Exceeded" and not st["success"],
        st["return_status"],
    )


def main():
    probe_x = ca.MX.sym("x")
    try:
        ca.nlpsol("probe", "pounce", {"x": probe_x, "f": probe_x**2})
    except RuntimeError as exc:
        print("pounce plugin not loadable — is CASADIPATH set to this directory?")
        print(exc)
        return 1
    for t in (
        test_mx_with_parameters,
        test_multipliers_with_active_bound,
        test_solution_map_derivative,
        test_opti,
        test_stats,
        test_iteration_callback,
        test_warm_start,
        test_limited_memory_and_nonlinear_variables,
        test_nmpc_feedback_gain_is_not_silently_zero,
        test_active_set_sqp_algorithm,
        test_working_set_carries_between_calls,
        test_a_raising_model_fails_the_solve_not_the_process,
        test_iteration_callback_can_interrupt,
        test_lam_p_matches_ipopt_and_the_envelope_theorem,
        test_threaded_map_matches_serial,
        test_option_pass_through,
    ):
        t()
    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
