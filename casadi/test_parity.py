#!/usr/bin/env python3
"""Parity checks for the POUNCE CasADi plugin, against the bundled ipopt.

Run with the plugin's directory on CasADi's search path::

    make test          # or: CASADIPATH=$PWD python3 test_parity.py

Every check compares POUNCE against `nlpsol(..., 'ipopt', ...)` on the
same model, so a failure says "the two solvers disagree", not "the
number moved".
"""

import os
import shutil
import subprocess
import sys
import tempfile

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


def test_custom_derivative_functions():
    """`grad_f` / `jac_g` / `hess_lag` replace the autogenerated ones."""
    x = ca.MX.sym("x", 2)
    p = ca.MX.sym("p")
    lam_f = ca.MX.sym("lam_f")
    lam_g = ca.MX.sym("lam_g")
    f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    g = x[0] ** 2 + x[1] ** 2 - p
    nlp = {"x": x, "p": p, "f": f, "g": g}
    custom = {
        "grad_f": ca.Function("my_grad_f", [x, p], [f, ca.gradient(f, x)]),
        "jac_g": ca.Function("my_jac_g", [x, p], [g, ca.jacobian(g, x)]),
        "hess_lag": ca.Function("my_hess_l", [x, p, lam_f, lam_g],
                                [ca.triu(ca.hessian(lam_f * f + lam_g * g, x)[0])]),
    }
    kw = dict(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    auto = ca.nlpsol("auto", "pounce", nlp, QUIET_POUNCE)(**kw)
    mine = ca.nlpsol("mine", "pounce", nlp, dict(QUIET_POUNCE, **custom))(**kw)
    ipopt = ca.nlpsol("ip", "ipopt", nlp, dict(QUIET_IPOPT, **custom))(**kw)
    check("custom grad_f/jac_g/hess_lag == autogenerated",
          close(mine["x"], auto["x"], 1e-9), f"x={mine['x'].T}")
    check("custom derivatives agree with ipopt", close(mine["x"], ipopt["x"], 1e-6))

    # A wrong signature is refused with a message, not a segfault later on.
    bad = ca.Function("bad_grad_f", [x], [ca.gradient(f, x)])
    try:
        ca.nlpsol("bad", "pounce", nlp, dict(QUIET_POUNCE, grad_f=bad))
        refused = False
    except RuntimeError as exc:
        refused = "grad_f must take 2 inputs" in str(exc)
    check("a mis-shaped custom derivative is refused", refused)


def test_convexify_matches_ipopt():
    """`convexify_strategy` is CasADi's own `Convexify`, so it must agree.

    The model is deliberately nonconvex (`sin`), which is where the strategies
    differ from each other: `eigen-reflect` walks to a different — here better
    — local minimum than the unconvexified run, in both plugins alike.
    """
    x = ca.MX.sym("x", 3)
    nlp = {"x": x, "f": ca.sum1(ca.sin(3 * x)) + 0.5 * ca.sumsqr(x - 0.3),
           "g": ca.sum1(x)}
    kw = dict(x0=[0.4, 0.1, -0.2], lbg=-1, ubg=1)
    for strategy in ("eigen-clip", "eigen-reflect"):
        a = ca.nlpsol("a", "pounce", nlp,
                      dict(QUIET_POUNCE, convexify_strategy=strategy))(**kw)
        b = ca.nlpsol("b", "ipopt", nlp,
                      dict(QUIET_IPOPT, convexify_strategy=strategy))(**kw)
        check(f"convexify_strategy={strategy} matches ipopt",
              close(a["x"], b["x"], 1e-5), f"f={float(a['f']):.9f}")

    plain = ca.nlpsol("p", "pounce", nlp, QUIET_POUNCE)(**kw)
    reflected = ca.nlpsol("r", "pounce", nlp,
                          dict(QUIET_POUNCE, convexify_strategy="eigen-reflect"))(**kw)
    check("convexify actually changes the trajectory",
          not close(plain["x"], reflected["x"], 1e-3),
          f"f {float(plain['f']):.6f} -> {float(reflected['f']):.6f}")


def test_serialization_round_trip():
    """`S.save()` / `Function.load()`, as CasADi's own plugins support."""
    nlp = rosenbrock_nlp()
    opts = dict(QUIET_POUNCE, clip_inactive_lam=False,
                var_string_md={"names": ["a", "b"]})
    opts["pounce"] = {"print_level": 0, "tol": 1e-10}
    S = ca.nlpsol("S", "pounce", nlp, opts)
    kw = dict(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    before = S(**kw)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.casadi")
        S.save(path)
        T = ca.Function.load(path)
    after = T(**kw)
    identical = (float(ca.norm_inf(before["x"] - after["x"])) == 0.0
                 and float(ca.norm_inf(before["lam_g"] - after["lam_g"])) == 0.0)
    check("serialized solver reloads and solves bit-identically",
          identical, f"x={after['x'].T}")
    check("serialized options survive the round trip",
          T.stats()["var_string_md"] == {"names": ["a", "b"]})


def test_metadata_options_are_accepted():
    """An ipopt script that sets metadata keeps working when swapped over."""
    nlp = rosenbrock_nlp()
    md = {
        "var_string_md": {"name": ["x0", "x1"]},
        "var_integer_md": {"prio": [1, 2]},
        "con_numeric_md": {"scale": [2.0]},
    }
    S = ca.nlpsol("S", "pounce", nlp, dict(QUIET_POUNCE, **md))
    S(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    st = S.stats()
    check("metadata options are accepted, not rejected",
          all(st[k] == v for k, v in md.items()))


def test_iteration_callback_step():
    """`iteration_callback_step` throttles the callback; the trace stays whole."""
    nlp = rosenbrock_nlp()

    class Counter(ca.Callback):
        def __init__(self):
            ca.Callback.__init__(self)
            self.n = 0
            self.construct("counter", {})

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
            self.n += 1
            return [0]

    kw = dict(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    every = Counter()
    third = Counter()
    a = ca.nlpsol("a", "pounce", nlp, dict(QUIET_POUNCE, iteration_callback=every))
    a(**kw)
    b = ca.nlpsol("b", "pounce", nlp, dict(QUIET_POUNCE, iteration_callback=third,
                                           iteration_callback_step=3))
    b(**kw)
    iters = len(b.stats()["iterations"]["inf_pr"])
    check("iteration_callback_step throttles the callback",
          0 < third.n < every.n, f"{third.n} calls at step 3 vs {every.n} at step 1")
    check("iteration_callback_step leaves stats()['iterations'] complete",
          iters == len(a.stats()["iterations"]["inf_pr"]) and iters > third.n,
          f"{iters} recorded iterations")


HERE = os.path.dirname(os.path.abspath(__file__))
POUNCE_INC = os.path.join(HERE, "..", "crates", "pounce-cinterface", "include")
POUNCE_LIB = os.path.join(HERE, "..", "target", "release")


def _compile_generated(solver, workdir, stem):
    """`solver.generate()` → a shared object → a callable `ca.external`.

    The C compiler sees only `pounce.h` and links `libpounce_cinterface`:
    no CasADi, no Python, no plugin. That is the whole point of the
    exercise, so the command line is deliberately spelled out.
    """
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        solver.generate(f"{stem}.c")
        so = os.path.join(workdir, f"{stem}.so")
        cc = shutil.which("cc") or shutil.which("gcc")
        subprocess.run(
            [cc, "-O2", "-Wall", "-shared", "-fPIC", "-o", so, f"{stem}.c",
             "-I", POUNCE_INC, "-L", POUNCE_LIB, "-lpounce_cinterface",
             "-Wl,-rpath," + os.path.abspath(POUNCE_LIB), "-lm"],
            check=True, capture_output=True, text=True)
    finally:
        os.chdir(cwd)
    return ca.external(solver.name(), so)


def test_codegen_matches_the_interpreted_solve():
    """`solver.generate()` — the model *and* the solve, as compiled C.

    CasADi's ipopt plugin generates C that talks to Ipopt's C API; POUNCE
    generates C that talks to the same API through `pounce.h`. The
    generated solve has to reach the same point as the interpreted one,
    multipliers included — including `clip_inactive_lam`, which lives in
    the plugin and so has to be reproduced in the emitted runtime.
    """
    if not (shutil.which("cc") or shutil.which("gcc")):
        print("SKIP  codegen (no C compiler)")
        return

    with tempfile.TemporaryDirectory() as d:
        # 1. A plain solve, exact Hessian.
        nlp = rosenbrock_nlp()
        kw = dict(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
        S = ca.nlpsol("cg_plain", "pounce", nlp,
                      {"print_time": False, "pounce": {"print_level": 0, "tol": 1e-10}})
        want = S(**kw)
        try:
            G = _compile_generated(S, d, "cg_plain")
        except subprocess.CalledProcessError as exc:
            check("generated C compiles", False, exc.stderr.strip().splitlines()[-1][:120])
            return
        check("generated C compiles and links against libpounce_cinterface", True)
        got = G(x0=[0.5, 0.5], p=1.5, lbx=-ca.inf, ubx=ca.inf, lbg=-ca.inf, ubg=0,
                lam_x0=0, lam_g0=0)
        for key in ("x", "f", "lam_x", "lam_g"):
            check(f"codegen == interpreted: {key}",
                  float(ca.norm_inf(ca.DM(got[key]) - ca.DM(want[key]))) == 0.0,
                  f"{np.array(got[key]).ravel()}" if key == "x" else "")

        # 2. A bounded model, where the plugin's default clipping is what
        #    keeps the bound multipliers usable. If the runtime skipped it,
        #    lam_x would differ here and nowhere else.
        y = ca.MX.sym("y", 2)
        bounded = {"x": y, "f": (y[0] - 3) ** 2 + (y[1] - 0.5) ** 2,
                   "g": y[0] + y[1]}
        bkw = dict(x0=[0.0, 0.0], lbx=[-10, -10], ubx=[1, 10], lbg=-10, ubg=10)
        B = ca.nlpsol("cg_bounded", "pounce", bounded,
                      {"print_time": False, "pounce": {"print_level": 0}})
        bwant = B(**bkw)
        BG = _compile_generated(B, d, "cg_bounded")
        bgot = BG(lam_x0=0, lam_g0=0, **bkw)
        check("codegen reproduces clip_inactive_lam",
              float(ca.norm_inf(ca.DM(bgot["lam_x"]) - ca.DM(bwant["lam_x"]))) == 0.0,
              f"lam_x={np.array(bgot['lam_x']).ravel()} (one active, one clipped to 0)")

        # 3. Limited memory plus a nonlinear-variable subset: the mask has to
        #    reach the generated solver too, or it silently approximates over
        #    every variable.
        n = 12
        z = ca.MX.sym("z", n)
        masked = {"x": z,
                  "f": (1 - z[0]) ** 2 + 100 * (z[1] - z[0] ** 2) ** 2 + ca.sum1(z[2:]),
                  "g": ca.sum1(z)}
        mkw = dict(x0=[0.5] * n, lbx=-5, ubx=5, lbg=-10, ubg=10)
        M = ca.nlpsol("cg_masked", "pounce", masked, {
            "print_time": False, "pass_nonlinear_variables": True,
            "pounce": {"print_level": 0, "hessian_approximation": "limited-memory"}})
        mwant = M(**mkw)
        MG = _compile_generated(M, d, "cg_masked")
        mgot = MG(lam_x0=0, lam_g0=0, **mkw)
        check("codegen carries the L-BFGS nonlinear-variable subset",
              float(ca.norm_inf(ca.DM(mgot["x"]) - ca.DM(mwant["x"]))) == 0.0,
              f"f={float(mgot['f']):.9f}")


def test_codegen_refuses_what_it_cannot_reproduce():
    """Options the generated code cannot honour fail loudly at generate()."""
    nlp = rosenbrock_nlp()

    class Noop(ca.Callback):
        def __init__(self):
            ca.Callback.__init__(self)
            self.construct("noop", {})

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
            return [0]

    cases = [
        ("iteration_callback", {"iteration_callback": Noop()}),
        ("warm_start_from_previous", {"warm_start_from_previous": True}),
        ("convexify_strategy", {"convexify_strategy": "eigen-clip"}),
    ]
    with tempfile.TemporaryDirectory() as d:
        for label, opts in cases:
            S = ca.nlpsol("cg_" + label, "pounce", nlp, dict(QUIET_POUNCE, **opts))
            cwd = os.getcwd()
            try:
                os.chdir(d)
                S.generate("bad.c")
                refused = False
                detail = "generated anyway"
            except RuntimeError as exc:
                refused = label in str(exc)
                detail = str(exc).strip().splitlines()[-1][:70]
            finally:
                os.chdir(cwd)
            check(f"codegen refuses {label} by name", refused, detail)


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
        test_custom_derivative_functions,
        test_convexify_matches_ipopt,
        test_serialization_round_trip,
        test_metadata_options_are_accepted,
        test_iteration_callback_step,
        test_codegen_matches_the_interpreted_solve,
        test_codegen_refuses_what_it_cannot_reproduce,
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
