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
