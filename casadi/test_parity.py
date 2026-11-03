#!/usr/bin/env python3
"""Parity checks for the POUNCE CasADi plugin, against the bundled ipopt.

Run with the plugin's directory on CasADi's search path::

    make test          # or: CASADIPATH=$PWD python3 test_parity.py

Every check compares POUNCE against `nlpsol(..., 'ipopt', ...)` on the
same model, so a failure says "the two solvers disagree", not "the
number moved".
"""

import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import casadi as ca
import numpy as np

QUIET_POUNCE = {"pounce": {"print_level": 0}, "print_time": False}
QUIET_IPOPT = {"ipopt": {"print_level": 0}, "print_time": False}

failures = []


def check(name, ok, detail="", log=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    # Verbatim, indented, and never truncated. A check that carries a log
    # carries it because the one-line summary cannot be trusted to contain
    # the answer, and this output's only reader is a CI log.
    for line in log.strip().splitlines():
        print(f"      | {line}")
    if not ok:
        failures.append(name)


def first_error(stderr):
    """The line a compiler failure is actually about.

    Deliberately not `splitlines()[-1]`, which is what this used to be:
    clang finishes with `1 error generated.` and MSVC with a summary, so
    the last line reports the *count* and discards the diagnostic. gh#782
    is what that cost — a code generation break against CasADi 3.8 whose
    CI log read, in full, `FAIL  generated C compiles — 1 error
    generated.` The defect had to be reproduced locally to learn anything
    at all about it, and the log it left behind was evidence of nothing
    beyond its own existence.

    Callers pass the whole of `stderr` to `check(log=...)` as well; this
    is the summary that sits on the FAIL line, not a substitute for it.
    """
    lines = stderr.strip().splitlines()
    for line in lines:
        # `error:` covers gcc and clang; `error C2065:` is MSVC, which also
        # writes `fatal error C1083:`.
        if "error:" in line.lower() or re.search(r"\berror [A-Z]?\d+\b", line):
            return line.strip()[:200]
    return lines[0].strip()[:200] if lines else "(the compiler said nothing)"


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


def test_repeated_solves_do_not_concatenate_the_trace():
    """One memory object, many solves: `iterations` describes the last one.

    A receding-horizon loop calls the same solver object over and over.
    The trace vectors used to accumulate across those calls while
    `iter_count` beside them described only the latest solve, so the two
    disagreed by a factor of however many solves had run (gh#634).
    """
    nlp = rosenbrock_nlp()
    S = ca.nlpsol("S", "pounce", nlp, QUIET_POUNCE)
    lengths = []
    for p in (1.5, 1.6, 1.7):
        S(x0=[0.5, 0.5], p=p, lbg=-ca.inf, ubg=0)
        st = S.stats()
        lengths.append((st["iter_count"], len(st["iterations"]["inf_pr"])))
    # ipopt records the initial point too, hence trace == iter_count + 1.
    consistent = all(trace == iters + 1 for iters, trace in lengths)
    check("repeated solves: trace describes only the last solve", consistent,
          f"(iter_count, trace) = {lengths}")

    # The same property, checked against the plugin this one is modelled
    # on: ipopt clears its trace at the top of every solve too.
    I = ca.nlpsol("I", "ipopt", nlp, QUIET_IPOPT)
    ip = []
    for p in (1.5, 1.6, 1.7):
        I(x0=[0.5, 0.5], p=p, lbg=-ca.inf, ubg=0)
        st = I.stats()
        ip.append((st["iter_count"], len(st["iterations"]["inf_pr"])))
    check("repeated solves: same trace behaviour as ipopt",
          all(trace == iters + 1 for iters, trace in ip),
          f"pounce={lengths} ipopt={ip}")


def test_restoration_iterations_are_labelled():
    """`iterations['alg_mod']` separates outer rows from restoration ones.

    POUNCE used to fire the intermediate callback only from its outer
    loop, so this column would have been constant zero and #637 declined
    to publish it. gh#645 made the restoration inner solver fire too,
    which is what gives the column something to say — and what makes it
    load-bearing: on a restoration row the other vectors describe the
    min-||c||_1 feasibility subproblem, not this NLP.
    """
    clean = rosenbrock_nlp()
    S = ca.nlpsol("S", "pounce", clean, QUIET_POUNCE)
    S(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    it = S.stats()["iterations"]
    check("stats: iterations carries alg_mod", "alg_mod" in it,
          f"keys = {sorted(it)}")
    if "alg_mod" not in it:
        return
    check("stats: alg_mod is as long as the other traces",
          len(it["alg_mod"]) == len(it["inf_pr"]),
          f"{len(it['alg_mod'])} vs {len(it['inf_pr'])}")
    check("stats: a clean solve is all regular iterations",
          set(it["alg_mod"]) <= {0}, f"{sorted(set(it['alg_mod']))}")

    # The infeasible equality from `test_restoration_stats`: this one
    # actually restores.
    x = ca.MX.sym("x")
    hard = {"x": x, "f": x**2, "g": x**2 + 1}
    R = ca.nlpsol("R", "pounce", hard, dict(QUIET_POUNCE))
    try:
        R(x0=0.5, lbg=0, ubg=0)
    except RuntimeError:
        pass
    st = R.stats()
    modes = st["iterations"]["alg_mod"]
    check("stats: restoration iterations are labelled 1",
          any(m == 1 for m in modes),
          f"{st['return_status']}, alg_mod = {modes}")

    # The column has to stay usable as an index into the others.
    check("stats: alg_mod still aligns with the traces under restoration",
          len(modes) == len(st["iterations"]["inf_pr"]),
          f"{len(modes)} vs {len(st['iterations']['inf_pr'])}")

    # `iter_count` must keep describing the outer solve. The inner
    # solver restarts its own counter from zero on every restoration
    # entry, so recording those would leave `iter_count` reporting
    # whatever the last episode happened to reach — the same class of
    # disagreement #637 fixed for the accumulating trace, one level down.
    #
    # Necessary rather than sufficient, but it is the part an outside
    # caller can see: a clobbered `iter_count` could not exceed the
    # longest run of restoration rows, because that run is what would
    # have written it.
    longest_resto_run = max((len(list(g)) for m, g in
                             itertools.groupby(modes) if m == 1), default=0)
    check("stats: iter_count is the outer count, not the inner one",
          st["iter_count"] > longest_resto_run,
          f"iter_count {st['iter_count']} vs longest restoration run "
          f"{longest_resto_run}")

    # Note for anyone reading the two side by side: `iter_count` counts
    # more than the regular rows here (the outer counter advances across
    # a restoration episode, as upstream's does — its `r`-suffixed rows
    # share the same counter). That predates gh#645 and is unchanged by
    # it; `test_repeated_solves_do_not_concatenate_the_trace` is where
    # the exact trace/`iter_count` agreement is pinned, on solves that
    # never restore.


def test_final_kkt_errors_in_stats():
    """The final infeasibilities POUNCE already computes, in `stats()`."""
    nlp = rosenbrock_nlp()
    S = ca.nlpsol("S", "pounce", nlp, QUIET_POUNCE)
    S(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    st = S.stats()
    present = {"final_inf_pr", "final_inf_du", "final_compl_inf"} <= set(st)
    check("stats: final KKT errors present", present)
    if present:
        tol = 1e-6
        converged = (st["final_inf_pr"] < tol and st["final_inf_du"] < tol
                     and st["final_compl_inf"] < tol)
        check("stats: final KKT errors agree with the success flag", converged,
              f"inf_pr={st['final_inf_pr']:.2e} inf_du={st['final_inf_du']:.2e} "
              f"compl={st['final_compl_inf']:.2e}")

    # The final numbers and the end of the trace are the same quantities,
    # and must not come from two different places. Checked on a solve cut
    # short by `max_iter`, where they are O(1e-2) — on a converged solve
    # both are ~1e-17 and any tolerance passes for the wrong reason.
    T = ca.nlpsol("T", "pounce", nlp,
                  {"print_time": False, "pounce": {"print_level": 0, "max_iter": 3}})
    T(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    cut = T.stats()
    big = cut["final_inf_pr"] > 1e-4 and cut["final_inf_du"] > 1e-4
    agrees = (abs(cut["final_inf_pr"] - cut["iterations"]["inf_pr"][-1]) < 1e-12
              and abs(cut["final_inf_du"] - cut["iterations"]["inf_du"][-1]) < 1e-12)
    check("stats: final KKT errors match the end of the trace", big and agrees,
          f"final=({cut['final_inf_pr']:.3e}, {cut['final_inf_du']:.3e}) "
          f"trace=({cut['iterations']['inf_pr'][-1]:.3e}, "
          f"{cut['iterations']['inf_du'][-1]:.3e})")


def test_linear_solver_stats():
    """Which KKT backend ran, and what it did — `linear_solver=feral`."""
    nlp = rosenbrock_nlp()
    S = ca.nlpsol("S", "pounce", nlp,
                  {"print_time": False,
                   "pounce": {"print_level": 0, "linear_solver": "feral"}})
    S(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    st = S.stats()
    ls = st.get("linear_solver")
    check("stats: linear_solver post-mortem present", ls is not None)
    if ls is None:
        return
    check("stats: linear_solver names the backend that ran",
          ls["solver_name"] == "feral", ls["solver_name"])
    check("stats: linear_solver counts factorizations",
          ls["n_factors"] > 0
          and ls["n_pattern_reuse"] + ls["n_pattern_changes"] == ls["n_factors"],
          f"{ls['n_factors']} factors, {ls['n_pattern_reuse']} pattern reuses")
    # Absent is absent: nothing pounce does not measure is reported as 0.
    # Phase timings are the ones deliberately missing (gh#634).
    check("stats: linear_solver omits what pounce does not measure",
          not {"analyze_time", "factor_time", "solve_time"} & set(ls),
          f"keys = {sorted(ls)}")


def test_restoration_stats():
    """Restoration is reported per solve — zero when it never ran, and
    non-zero when it did."""
    clean = rosenbrock_nlp()
    S = ca.nlpsol("S", "pounce", clean, QUIET_POUNCE)
    S(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    quiet = S.stats().get("restoration")
    check("stats: restoration present", quiet is not None)
    if quiet is None:
        return
    check("stats: restoration is zero on a clean solve",
          quiet["calls"] == 0 and quiet["wall_secs"] == 0.0, str(quiet))

    # An equality that cannot be satisfied: the solver goes to
    # restoration, fails to find a feasible point, and says so.
    x = ca.MX.sym("x")
    hard = {"x": x, "f": x**2, "g": x**2 + 1}
    R = ca.nlpsol("R", "pounce", hard, dict(QUIET_POUNCE))
    try:
        R(x0=0.5, lbg=0, ubg=0)
    except RuntimeError:
        pass
    st = R.stats()
    resto = st.get("restoration", {})
    check("stats: restoration counts a real restoration entry",
          resto.get("calls", 0) > 0 and resto.get("inner_iters", 0) > 0
          and resto.get("wall_secs", 0.0) > 0.0,
          f"{st['return_status']}, {resto}")


def test_live_diagnostics_during_the_callback():
    """Everything an `iteration_callback` needs, without parsing the log.

    CasADi fixes the callback signature at (x, f, g, lam_x, lam_g), so
    the extra per-iteration diagnostics come through `stats()`, which is
    callable from inside the callback: the trace's last entry is the
    current iteration, and the current violation vectors are fetched on
    demand while the solve is in flight (gh#634).
    """
    nlp = rosenbrock_nlp()
    seen = []

    class Probe(ca.Callback):
        def __init__(self):
            ca.Callback.__init__(self)
            self.construct("probe", {})

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
            seen.append(S.stats())
            return [0]

    probe = Probe()
    S = ca.nlpsol("S", "pounce", nlp, dict(QUIET_POUNCE, iteration_callback=probe))
    S(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    check("live: callback can read stats()", len(seen) > 1, f"{len(seen)} snapshots")
    if not seen:
        return

    scalars = {"inf_pr", "inf_du", "mu", "d_norm", "regularization_size",
               "obj", "alpha_pr", "alpha_du", "ls_trials"}
    check("live: every per-iteration scalar is readable mid-solve",
          all(scalars <= set(s["iterations"]) for s in seen))
    # The trace grows by one per callback — its last entry is *this*
    # iteration, not a stale one.
    grows = [len(s["iterations"]["inf_pr"]) for s in seen]
    check("live: the trace ends on the current iteration",
          grows == list(range(1, len(seen) + 1)), f"lengths {grows[:5]}…")

    viol = seen[-1].get("current_violations")
    check("live: current violations are available mid-solve", viol is not None)
    if viol is not None:
        wanted = {"x_L_violation", "x_U_violation", "compl_x_L", "compl_x_U",
                  "grad_lag_x", "nlp_constraint_violation", "compl_g"}
        check("live: violations carry the full Ipopt field set", wanted <= set(viol),
              f"keys = {sorted(viol)}")
        check("live: violation vectors are the right shape and finite",
              len(viol["grad_lag_x"]) == 2 and len(viol["compl_g"]) == 1
              and all(np.isfinite(viol["grad_lag_x"])))

    # Mid-solve, the final numbers do not exist yet and are not faked;
    # once the solve ends, they do and the live ones are gone.
    check("live: no final KKT errors mid-solve",
          all("final_inf_pr" not in s for s in seen))
    after = S.stats()
    check("live: no stale violations after the solve",
          "current_violations" not in after and "final_inf_pr" in after)


def test_option_types_come_from_pounce_not_the_literal():
    """`tol: 1` is an int in Python and a number to POUNCE.

    Dispatching on the value's own type sent it to AddIpoptIntOption,
    which refuses it — the option silently kept its default. The type
    now comes from POUNCE's registry (gh#634).
    """
    nlp = rosenbrock_nlp()
    kw = dict(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)

    def iters(tol):
        S = ca.nlpsol("S", "pounce", nlp,
                      {"print_time": False,
                       "pounce": {"print_level": 0, "tol": tol, "max_iter": 200}})
        S(**kw)
        return S.stats()["iter_count"]

    # `1` and `1.0` are the same tolerance and must give the same solve.
    # This is the check that bites: an int-valued `tol` was refused and
    # left the default in place, so the two spellings disagreed. Note
    # "int takes fewer iterations than 1e-12" does *not* bite — the
    # default tolerance also takes fewer, so it passes pre-fix.
    as_int, as_float, tight = iters(1), iters(1.0), iters(1e-12)
    check("options: int and float spellings of a number option agree",
          as_int == as_float,
          f"tol=1 took {as_int} iters, tol=1.0 took {as_float}")
    check("options: a loose tolerance really is looser",
          as_int < tight, f"tol=1 took {as_int} iters, tol=1e-12 took {tight}")
    # An integer option given as an int stays an integer option.
    S = ca.nlpsol("S", "pounce", nlp,
                  {"print_time": False, "pounce": {"print_level": 0, "max_iter": 2}})
    S(**kw)
    check("options: integer options are unaffected",
          S.stats()["return_status"] == "Maximum_Iterations_Exceeded")


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


def test_output_does_not_tear_embedder_lines():
    """gh#667: POUNCE's log must not split a line the embedder is printing
    from inside a callback.

    Driven by a C++ host (`test_output_interleaving.cpp`), not from here.
    CasADi's Python bindings point `Logger::writeFun` at `PySys_WriteStdout`
    but leave `Logger::flush` at `flushDefault`, so the plugin's flush drains
    `std::cout` while the bytes are sitting in Python's `sys.stdout`. Run
    from Python this would report on CasADi's buffering, not on the plugin's
    flushing, and would keep passing however broken the plugin got.

    The driver prints a long line in chunks from `iteration_callback` while
    POUNCE writes its iteration rows to the same descriptor. Pre-fix every
    such line arrives without its terminator.
    """
    exe = os.path.join(HERE, "test_output_interleaving")
    if not os.path.exists(exe):
        print("SKIP  output interleaving (driver not built; run `make`)")
        return
    env = dict(os.environ, CASADIPATH=HERE)
    # stdout must be a pipe: on a tty the competing buffer is line-buffered
    # and the tear cannot happen in the first place.
    out = subprocess.run([exe], env=env, capture_output=True, text=True,
                         timeout=300).stdout
    host = [ln for ln in out.splitlines() if ln.startswith("HOST ")]
    torn = [ln for ln in host if not ln.endswith(" END")]
    check("output: the embedder actually printed", len(host) > 1,
          f"{len(host)} lines from the callback")
    check("output: POUNCE does not tear embedder lines (gh#667)",
          not torn, f"{len(torn)}/{len(host)} lines torn")


def test_solve_report_option():
    """`solve_report` writes POUNCE's structured report (gh#644).

    Both entry points (`IpoptEnableIterHistory`, `IpoptWriteSolveReport`)
    already existed in the C interface; what was missing was any way for
    a CasADi caller to reach them, so the report was available to a
    `pounce` CLI user and not to this one.
    """
    nlp = rosenbrock_nlp()

    # Off by default: no keys claiming a report, and nothing written.
    S = ca.nlpsol("S", "pounce", nlp, QUIET_POUNCE)
    S(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
    st = S.stats()
    check("solve_report: absent from stats when not requested",
          "solve_report" not in st and "solve_report_written" not in st,
          f"keys = {sorted(k for k in st if 'report' in k)}")

    with tempfile.TemporaryDirectory() as d:
        full = os.path.join(d, "full.json")
        F = ca.nlpsol("F", "pounce", nlp,
                      dict(QUIET_POUNCE, solve_report=full,
                           solve_report_detail="full"))
        F(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
        st = F.stats()
        check("solve_report: stats reports the write", 
              st.get("solve_report_written") is True and st.get("solve_report") == full,
              f"{st.get('solve_report_written')}, {st.get('solve_report')}")
        check("solve_report: file exists", os.path.exists(full))
        if not os.path.exists(full):
            return
        with open(full) as fh:
            report = json.load(fh)
        check("solve_report: schema is pounce.solve-report/v1",
              report.get("schema") == "pounce.solve-report/v1",
              str(report.get("schema")))

        # `detail=full` is the whole reason `IpoptEnableIterHistory` has
        # to be called before the solve; if that ordering were wrong the
        # report would arrive with no trajectory and nothing saying why.
        traj = report.get("iterations")
        check("solve_report: full embeds the trajectory",
              isinstance(traj, list) and len(traj) > 0,
              f"{type(traj).__name__}, {len(traj) if isinstance(traj, list) else 0} entries")
        # Same convention as `stats()['iterations']`: the initial point
        # is recorded too, so the trajectory is one longer than the
        # iteration count. A disagreement here means the two views of one
        # solve are describing different things — the defect class #637
        # fixed for the trace.
        if isinstance(traj, list):
            check("solve_report: trajectory agrees with iter_count",
                  len(traj) == st["iter_count"] + 1,
                  f"{len(traj)} entries vs iter_count {st['iter_count']}")

        # Default detail is a summary: same report, no trajectory. Worth
        # pinning because the cost of `full` is a retained iterate per
        # iteration, and a default that quietly paid it would be a
        # surprise on a long solve.
        summary = os.path.join(d, "summary.json")
        M = ca.nlpsol("M", "pounce", nlp, dict(QUIET_POUNCE, solve_report=summary))
        M(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
        with open(summary) as fh:
            sm = json.load(fh)
        check("solve_report: summary is the default and omits the trajectory",
              sm.get("schema") == "pounce.solve-report/v1" and not sm.get("iterations"),
              f"iterations = {sm.get('iterations')}")

        # A typo costs the construction, not a solve.
        try:
            ca.nlpsol("B", "pounce", nlp,
                      dict(QUIET_POUNCE, solve_report=os.path.join(d, "b.json"),
                           solve_report_detail="verbose"))
            refused, detail = False, "accepted"
        except RuntimeError as exc:
            refused = "solve_report_detail" in str(exc)
            detail = str(exc).strip().splitlines()[-1][:70]
        check("solve_report: an invalid detail is refused at construction",
              refused, detail)

        # An unwritable path must not cost the answer. The solve
        # succeeded; a diagnostic file that could not be written is a
        # warning and a False in stats, not a failed solve.
        bad = os.path.join(d, "no-such-dir", "r.json")
        B = ca.nlpsol("B2", "pounce", nlp, dict(QUIET_POUNCE, solve_report=bad))
        r = B(x0=[0.5, 0.5], p=1.5, lbg=-ca.inf, ubg=0)
        st = B.stats()
        check("solve_report: an unwritable path does not fail the solve",
              st["success"] and st.get("solve_report_written") is False,
              f"success={st['success']}, written={st.get('solve_report_written')}, "
              f"x={r['x']}")


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
            check("generated C compiles", False, first_error(exc.stderr),
                  log=exc.stderr)
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
        ("solve_report", {"solve_report": "report.json"}),
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


def _hessian_free_objective():
    """An objective whose Jacobian is an opaque callback declaring no
    derivative of its own, so CasADi genuinely cannot form a Hessian.

    This is the model class `hessian_approximation='finite-difference'`
    exists for -- an FMU or `DaeBuilder` transcription with analytic first
    derivatives and nothing above them -- and it is the one the plugin
    could not serve while a single `exact_hessian_` flag stood for both
    "may call cb_h for values" and "can declare a sparsity pattern".
    """
    class JacCB(ca.Callback):
        def __init__(self, name):
            ca.Callback.__init__(self)
            self.construct(name, {})

        def get_n_in(self):
            return 2

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, i):
            return ca.Sparsity.dense(3, 1) if i == 0 else ca.Sparsity.dense(1, 1)

        def get_sparsity_out(self, i):
            return ca.Sparsity.dense(1, 3)

        def eval(self, arg):
            x = np.array(arg[0]).flatten()
            e = np.exp(x[0] * x[1])
            return [ca.DM([[e * x[1] + 2 * (x[0] - 1), e * x[0], 4 * x[2] ** 3]])]

        def has_jacobian(self):
            return False

    class FCB(ca.Callback):
        def __init__(self, name):
            ca.Callback.__init__(self)
            self.jc = JacCB(name + "_jac")
            self.construct(name, {})

        def get_n_in(self):
            return 1

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, i):
            return ca.Sparsity.dense(3, 1)

        def get_sparsity_out(self, i):
            return ca.Sparsity.dense(1, 1)

        def eval(self, arg):
            x = np.array(arg[0]).flatten()
            return [ca.DM(np.exp(x[0] * x[1]) + x[2] ** 4 + (x[0] - 1) ** 2)]

        def has_jacobian(self):
            return True

        def get_jacobian(self, name, inames, onames, opts):
            return self.jc

    return FCB


def test_finite_difference_does_not_need_second_derivatives():
    """`finite-difference` must not require what it exists to replace.

    Fails on the parent commit: `exact_hessian_` was cleared only for
    `limited-memory`, so this mode still ran `create_function('nlp_hess_l')`
    and died with CasADi's `Derivatives cannot be calculated for ...` --
    the same error `exact` correctly gives -- on the only model class the
    mode is for. Found in review by @srikanth-gm (gh#823).
    """
    FCB = _hessian_free_objective()
    fcb = FCB("fcb_nohess")
    x = ca.MX.sym("x", 3)
    nlp = {"x": x, "f": fcb(x),
           "g": ca.vertcat(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1.0)}
    kw = dict(x0=[0.5, 0.5, 0.5], lbg=0, ubg=0)

    # The reference: the same model written so CasADi *can* differentiate it.
    y = ca.MX.sym("y", 3)
    ref_nlp = {"x": y,
               "f": ca.exp(y[0] * y[1]) + y[2] ** 4 + (y[0] - 1) ** 2,
               "g": ca.vertcat(y[0] ** 2 + y[1] ** 2 + y[2] ** 2 - 1.0)}
    ref = ca.nlpsol("ref", "pounce", ref_nlp,
                    {"print_time": False, "pounce": {"print_level": 0}})(**kw)

    # `exact` SHOULD still fail here -- there is no Hessian to evaluate.
    exact_failed = False
    try:
        ca.nlpsol("e", "pounce", nlp,
                  {"print_time": False,
                   "pounce": {"print_level": 0, "hessian_approximation": "exact"}})
    except Exception:
        exact_failed = True
    check("no-Hessian model: exact is still refused", exact_failed)

    for pattern in ("declared", "jacobian"):
        opts = {"print_time": False,
                "pounce": {"print_level": 0,
                           "hessian_approximation": "finite-difference",
                           "fd_hessian_pattern": pattern}}
        try:
            r = ca.nlpsol("fd", "pounce", nlp, opts)(**kw)
            ok, detail = close(r["f"], ref["f"], 1e-8), f"f={float(r['f']):.12g}"
        except Exception as exc:
            ok, detail = False, str(exc).strip().splitlines()[-1][:90]
        check(f"no-Hessian model: finite-difference/{pattern} solves", ok, detail)


def test_finite_difference_takes_structure_without_values():
    """The capability split, on a model that HAS a Hessian.

    `finite-difference` may read CasADi's Hessian *sparsity* -- it is worth
    real probe groups -- but must never ask for a value, because the values
    are what it recovers by probing. `cb_h` enforces that by refusing a
    values request outright, so a solve that completes is itself the proof
    that no value was ever requested.
    """
    x = ca.MX.sym("x", 3)
    nlp = {"x": x,
           "f": ca.exp(x[0] * x[1]) + x[2] ** 4 + (x[0] - 1) ** 2,
           "g": ca.vertcat(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1.0)}
    kw = dict(x0=[0.5, 0.5, 0.5], lbg=0, ubg=0)
    ref = ca.nlpsol("ref", "pounce", nlp,
                    {"print_time": False, "pounce": {"print_level": 0}})(**kw)

    def built_hess(sol):
        try:
            sol.get_function("nlp_hess_l")
            return True
        except Exception:
            return False

    made = {}
    for pattern in ("declared", "jacobian"):
        sol = ca.nlpsol("fd", "pounce", nlp,
                        {"print_time": False,
                         "pounce": {"print_level": 0,
                                    "hessian_approximation": "finite-difference",
                                    "fd_hessian_pattern": pattern}})
        made[pattern] = built_hess(sol)
        r = sol(**kw)
        check(f"fd/{pattern} reaches the exact objective",
              close(r["f"], ref["f"], 1e-8), f"f={float(r['f']):.12g}")

    # `declared` wants the pattern, so it builds the symbolic Hessian and
    # uses it for STRUCTURE only. `jacobian` says the pattern comes from
    # the Jacobian, so building it at all would be pure cost.
    check("fd/declared obtains CasADi's Hessian sparsity", made["declared"])
    check("fd/jacobian does not build a symbolic Hessian", not made["jacobian"])

    lm = ca.nlpsol("lm", "pounce", nlp,
                   {"print_time": False,
                    "pounce": {"print_level": 0,
                               "hessian_approximation": "limited-memory"}})
    check("limited-memory builds no symbolic Hessian", not built_hess(lm))

    # The split added a serialized field, so the stream version moved 1 -> 2.
    # A v1 stream has no `hessian_structure` to read and is restored with
    # `exact_hessian`, which is what the flag meant when the two were one.
    with tempfile.TemporaryDirectory() as d:
        for pattern in ("declared", "jacobian"):
            S = ca.nlpsol("ser", "pounce", nlp,
                          {"print_time": False,
                           "pounce": {"print_level": 0,
                                      "hessian_approximation": "finite-difference",
                                      "fd_hessian_pattern": pattern}})
            before = S(**kw)
            path = os.path.join(d, f"fd_{pattern}.casadi")
            S.save(path)
            after = ca.Function.load(path)(**kw)
            check(f"fd/{pattern} survives a serialization round trip",
                  float(ca.norm_inf(ca.DM(before["x"]) - ca.DM(after["x"]))) == 0.0)


def test_finite_difference_survives_restoration():
    """A nonconvex pair with bounds that block the Newton direction, so the
    solve enters feasibility restoration.

    Fails on the parent commit with `Restoration_Failed` at iteration 6,
    where every other Hessian mode converges: the FD updater ran inside the
    restoration sub-NLP, whose primal is the 5-block compound, carrying a
    pattern and an objective clique built for the original NLP's space.
    Restoration runs limited-memory for it now, as it already did for the
    partitioned Hessian and for the same stated reason.
    """
    x = ca.MX.sym("x", 3)
    nlp = {"x": x, "f": x[0] + x[1] + x[2],
           "g": ca.vertcat(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1.0,
                           ca.sin(5 * x[0]) + x[1] ** 3 - 0.9)}
    kw = dict(x0=[0.99, 0.99, 0.99], lbx=[-1, -1, -1], ubx=[1, 1, 1],
              lbg=0, ubg=0)
    gf = ca.Function("gf", [x], [nlp["g"]])

    for mode in ("exact", "limited-memory", "finite-difference"):
        sol = ca.nlpsol("s", "pounce", nlp,
                        {"print_time": False,
                         "pounce": {"print_level": 0, "max_iter": 500,
                                    "hessian_approximation": mode}})
        r = sol(**kw)
        st = sol.stats()
        entered = st.get("restoration", {}).get("calls", 0)
        viol = float(np.max(np.abs(np.array(gf(r["x"])).flatten())))
        xs = np.array(r["x"]).flatten()
        boxed = bool(np.all(xs >= -1 - 1e-8) and np.all(xs <= 1 + 1e-8))
        check(f"restoration/{mode}: converges",
              st.get("return_status") == "Solve_Succeeded",
              f'{st.get("return_status")}, resto_calls={entered}')
        # The problem is nonconvex and the modes legitimately land on
        # different local solutions, so the assertion is feasibility, not
        # a shared objective.
        check(f"restoration/{mode}: answer is feasible and in bounds",
              viol < 1e-8 and boxed, f"|g|inf={viol:.2e}")
    check("restoration: the fixture really does enter restoration",
          entered > 0, f"resto_calls={entered}")


def test_codegen_reproduces_the_finite_difference_hessian():
    """The generated path must carry the same capability split as the
    interpreted one.

    Under `finite-difference` the emitted C declares the Hessian *pattern*
    -- baked in as a literal, so the block needs no `nlp_hess_l`
    dependency -- and wires an `eval_h` that serves the structure request
    and refuses a values one. Getting this wrong is silent: the generated
    solve would simply use CasADi's exact Hessian and quietly stop being a
    finite-difference solve.
    """
    if not (shutil.which("cc") or shutil.which("gcc")):
        print("SKIP  codegen finite-difference (no C compiler)")
        return

    x = ca.MX.sym("x", 3)
    nlp = {"x": x,
           "f": ca.exp(x[0] * x[1]) + x[2] ** 4 + (x[0] - 1) ** 2,
           "g": ca.vertcat(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1.0)}
    kw = dict(x0=[0.5, 0.5, 0.5], lbg=0, ubg=0)

    with tempfile.TemporaryDirectory() as d:
        for pattern in ("declared", "jacobian"):
            S = ca.nlpsol(f"cg_fd_{pattern}", "pounce", nlp,
                          {"print_time": False,
                           "pounce": {"print_level": 0, "tol": 1e-10,
                                      "hessian_approximation": "finite-difference",
                                      "fd_hessian_pattern": pattern}})
            want = S(**kw)
            try:
                G = _compile_generated(S, d, f"cg_fd_{pattern}")
            except subprocess.CalledProcessError as exc:
                check(f"codegen fd/{pattern} compiles", False,
                      first_error(exc.stderr), log=exc.stderr)
                continue
            check(f"codegen fd/{pattern} compiles", True)
            # What makes it a finite-difference solve, read off the emitted
            # C: no `nlp_hess_l`, the option itself, and the Hessian callback
            # the pattern calls for -- the declared structure, or for
            # `jacobian` none at all (POUNCE derives the pattern itself). This used to be inferred from the answer
            # matching the interpreted solve bit for bit, which tells an
            # exact-Hessian solve apart (it differs in the last bits) only
            # while both engines round identically; GitHub's ubuntu-26.04
            # runners broke that (2026-11-03) though an ubuntu:26.04 container
            # with the same compiler and glibc did not.
            src = open(os.path.join(d, f"cg_fd_{pattern}.c")).read()
            callback = {"declared": "hess_l_struct",
                        "jacobian": "casadi_pounce_hess_l_empty"}[pattern]
            check(f"codegen fd/{pattern} emits a finite-difference Hessian",
                  "nlp_hess_l" not in src and callback in src
                  and '"hessian_approximation", "finite-difference"' in src,
                  f"expects {callback}")
            got = G(x0=[0.5, 0.5, 0.5], lbx=-ca.inf, ubx=ca.inf,
                    lbg=0, ubg=0, lam_x0=0, lam_g0=0)
            dx = float(ca.norm_inf(ca.DM(got["x"]) - ca.DM(want["x"])))
            # The two engines may round differently; the solve is to 1e-10.
            check(f"codegen fd/{pattern} == interpreted", dx <= 1e-8,
                  f"|dx|={dx:.3e} f={float(got['f']):.12g}")


def test_fd_hessian_stats_are_reported():
    """`stats()["fd_hessian"]` — which pattern the run ended up with and
    what it cost.

    Before this the two numbers that decide whether the mode is
    affordable on a model — the pattern source and the probe-group count —
    were reachable only through the `POUNCE_FD_HESSIAN_DEBUG` environment
    variable, i.e. not from an embedder at all. Asked for by
    @srikanth-gm (gh#823).

    The key property is that `pattern` reports the source actually
    **used**, not the one requested: `declared` falls back to `jacobian`
    whenever the model declares no Hessian structure, and that fallback
    is the whole reason to look.
    """
    x = ca.MX.sym("x", 3)
    nlp = {"x": x,
           "f": ca.exp(x[0] * x[1]) + x[2] ** 4 + (x[0] - 1) ** 2,
           "g": ca.vertcat(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1.0)}
    kw = dict(x0=[0.5, 0.5, 0.5], lbg=0, ubg=0)

    def run(popts):
        s = ca.nlpsol("s", "pounce", nlp,
                      {"print_time": False, "pounce": dict(popts, print_level=0)})
        s(**kw)
        return s.stats().get("fd_hessian")

    # Absent on every mode that is not finite-difference: a zero probe
    # count would read as "free" rather than "not this mode".
    check("fd_hessian absent under exact",
          run({"hessian_approximation": "exact"}) is None)
    check("fd_hessian absent under limited-memory",
          run({"hessian_approximation": "limited-memory"}) is None)

    dec = run({"hessian_approximation": "finite-difference",
               "fd_hessian_pattern": "declared"})
    jac = run({"hessian_approximation": "finite-difference",
               "fd_hessian_pattern": "jacobian"})
    check("fd_hessian present under finite-difference", dec is not None)
    check("fd_hessian names the declared pattern", dec["pattern"] == "declared",
          str(dec))
    check("fd_hessian names the jacobian pattern", jac["pattern"] == "jacobian",
          str(jac))
    # The declared pattern is the true one; the Jacobian derivation is a
    # superset, so it can only be wider and cost at least as many probes.
    check("the declared pattern is tighter than the Jacobian superset",
          dec["nnz"] < jac["nnz"] and dec["groups"] <= jac["groups"],
          f'declared {dec["nnz"]}nnz/{dec["groups"]}groups vs '
          f'jacobian {jac["nnz"]}nnz/{jac["groups"]}groups')
    check("fd_hessian carries n, so compression is derivable",
          dec["n"] == 3 and jac["n"] == 3, f'n={dec["n"]}')
    for k in ("nnz", "groups", "rho_max", "n"):
        check(f"fd_hessian.{k} is a positive count", dec[k] > 0, f"{k}={dec[k]}")
    check("fd_hessian reports the colouring fallback",
          dec["coloring_fell_back"] is False)
    # The declared pattern needs no objective clique at all; the Jacobian
    # derivation does, and this model states no objective linearity
    # through the C interface, so it is the widened one. That distinction
    # is what makes a surprising `groups` diagnosable.
    check("fd_hessian reports the objective-clique widening",
          dec["objective_clique_widened"] is False
          and jac["objective_clique_widened"] is True,
          f'declared={dec["objective_clique_widened"]} '
          f'jacobian={jac["objective_clique_widened"]}')

    # The property that makes the field worth reading: asking for
    # `declared` on a model with no Hessian at all reports what actually
    # ran, not what was asked for.
    FCB = _hessian_free_objective()
    fcb = FCB("fcb_stats")
    y = ca.MX.sym("y", 3)
    nohess = {"x": y, "f": fcb(y),
              "g": ca.vertcat(y[0] ** 2 + y[1] ** 2 + y[2] ** 2 - 1.0)}
    s = ca.nlpsol("s", "pounce", nohess,
                  {"print_time": False,
                   "pounce": {"print_level": 0,
                              "hessian_approximation": "finite-difference",
                              "fd_hessian_pattern": "declared"}})
    s(x0=[0.5, 0.5, 0.5], lbg=0, ubg=0)
    got = s.stats().get("fd_hessian")
    check("a declared request that fell back reports 'jacobian'",
          got is not None and got["pattern"] == "jacobian", str(got))


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
        test_repeated_solves_do_not_concatenate_the_trace,
        test_final_kkt_errors_in_stats,
        test_linear_solver_stats,
        test_restoration_stats,
        test_restoration_iterations_are_labelled,
        test_live_diagnostics_during_the_callback,
        test_option_types_come_from_pounce_not_the_literal,
        test_solve_report_option,
        test_codegen_matches_the_interpreted_solve,
        test_codegen_refuses_what_it_cannot_reproduce,
        test_output_does_not_tear_embedder_lines,
        test_finite_difference_does_not_need_second_derivatives,
        test_finite_difference_takes_structure_without_values,
        test_finite_difference_survives_restoration,
        test_codegen_reproduces_the_finite_difference_hessian,
        test_fd_hessian_stats_are_reported,
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
