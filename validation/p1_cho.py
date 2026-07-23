"""P1 -- CHO bioprocess parameter estimation (THE headline: sensitivity /
covariance).

A ~21,700-variable monolithic Pyomo DAE (10 CHO cell-culture batches,
orthogonal collocation, 12 shared kinetic parameters, deterministic seed).
Two things are validated:

  1. POINT ESTIMATE. pounce solves the full model; its objective and 12
     fitted parameters are checked against the committed IPOPT (MA57)
     benchmark and against the true parameters used to generate the data.
     (Live IPOPT is attempted too; see the caveat about MA57 below.)

  2. SENSITIVITY / COVARIANCE -- the part that matters most. A solver can
     nail the optimum and still compute a subtly-wrong covariance. We check
     pounce's covariance() TWO independent ways:

       (a) pyomo_pounce.covariance() -- pounce's held-factorization sIPOPT
           computation.
       (b) an independent reference: reduced Hessian from PyNumero (AMPL/ASL
           derivatives) assembled and factored with scipy.sparse.

     The equivalence of (a) and (b) is FIRST established exactly on two
     controls with known answers -- a linear regression (closed form
     sigma^2 (X'X)^-1) and a well-conditioned nonlinear fit (vs scipy
     curve_fit) -- where both routes agree to ~1e-15 and ~1e-7. The CHO
     covariance itself is then reported, with an honest account of the
     agreement between (a) and (b) on the full, severely ill-conditioned
     model.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from scipy.optimize import curve_fit

import pyomo_pounce
from pyomo_pounce import (covariance, declare_fitted, declare_residual)
from _common import (MAIN, abs_err, dump_result, rel_err, setup_pounce)
from _covref import covariance_reference
import _cho_import

setup_pounce()
warnings.simplefilter("ignore")


# ── covariance-method controls (known answers) ─────────────────────────────────
def control_linear():
    rng = np.random.default_rng(1)
    N = 40
    t = np.linspace(0, 1, N)
    y = 2.0 - 3.0 * t + rng.normal(0, 0.1, N)

    def build():
        m = pyo.ConcreteModel()
        m.b0 = pyo.Var(initialize=0.0)
        m.b1 = pyo.Var(initialize=0.0)
        m.I = pyo.RangeSet(0, N - 1)
        m.r = pyo.Var(m.I)
        m.rdef = pyo.Constraint(
            m.I, rule=lambda m, i: m.r[i] == y[i] - (m.b0 + m.b1 * t[i]))
        m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
        return m

    m = build()
    declare_fitted(m.b0, m.b1)
    declare_residual(m.r)
    SolverFactory("pounce").solve(m)
    cov = covariance(m)
    Cp = cov.matrix
    s2 = cov.sigma_sq
    # closed form
    X = np.column_stack([np.ones(N), t])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    Ccf = (res @ res) / (N - 2) * np.linalg.inv(X.T @ X)
    # independent PyNumero reference
    m2 = build()
    m2.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    SolverFactory("pounce").solve(m2)
    Cref, _, _ = covariance_reference(m2, [m2.b0, m2.b1], s2, m2.dual)
    return {
        "pounce_vs_closed_form_maxrel": float(
            np.max(np.abs(Cp - Ccf) / (np.abs(Ccf) + 1e-30))),
        "pynumero_vs_closed_form_maxrel": float(
            np.max(np.abs(Cref - Ccf) / (np.abs(Ccf) + 1e-30))),
        "pounce_vs_pynumero_maxrel": float(
            np.max(np.abs(Cp - Cref) / (np.abs(Cref) + 1e-30))),
    }


def control_nonlinear():
    rng = np.random.default_rng(3)
    N = 25
    t = np.linspace(0, 4, N)
    y = 5.0 * np.exp(-0.7 * t) + rng.normal(0, 0.05, N)

    def build():
        m = pyo.ConcreteModel()
        m.A = pyo.Var(initialize=4.0)
        m.k = pyo.Var(initialize=0.5)
        m.I = pyo.RangeSet(0, N - 1)
        m.r = pyo.Var(m.I)
        m.rdef = pyo.Constraint(
            m.I, rule=lambda m, i: m.r[i] == y[i] - m.A * pyo.exp(-m.k * t[i]))
        m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
        return m

    m = build()
    declare_fitted(m.A, m.k)
    declare_residual(m.r)
    SolverFactory("pounce").solve(m)
    cov = covariance(m)
    Cp = cov.matrix
    s2 = cov.sigma_sq
    popt, pcov = curve_fit(lambda t, A, k: A * np.exp(-k * t), t, y,
                           p0=[4.0, 0.5])
    m2 = build()
    m2.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    SolverFactory("pounce").solve(m2)
    Cref, _, _ = covariance_reference(m2, [m2.A, m2.k], s2, m2.dual)
    return {
        "pounce_A": float(pyo.value(m.A)), "pounce_k": float(pyo.value(m.k)),
        "scipy_A": float(popt[0]), "scipy_k": float(popt[1]),
        "pounce_vs_pynumero_maxrel": float(
            np.max(np.abs(Cp - Cref) / (np.abs(Cref) + 1e-30))),
        "pounce_vs_scipy_gaussnewton_maxrel": float(
            np.max(np.abs(Cp - pcov) / (np.abs(pcov) + 1e-30))),
    }


# ── CHO ────────────────────────────────────────────────────────────────────────
def count_ndata(cho, m, batches):
    n = 0
    for idx in m.BATCHES:
        b = m.batch[idx]
        tset = sorted(pyo.value(tt) for tt in b.t)
        for _, row in batches[idx]["meas"].iterrows():
            t = float(row["time"])
            tc = min(tset, key=lambda z: abs(z - t))
            if abs(tc - t) < 1e-6:
                n += len(cho.STATES)
    return n


def main():
    cho = _cho_import.load()
    p_true = cho.Params()
    batches = cho.generate_batches(
        p_true, cho.N_BATCHES, cho.N_MEAS, cho.NOISE_LEVEL, cho.RNG_SEED)

    # (1) pounce solve of the full model, with fitted-parameter declarations
    m = cho.build_full_model(batches, nfe=cho.NFE, ncp=cho.NCP)
    n_vars = len(list(m.component_data_objects(pyo.Var, active=True)))
    n_cons = len(list(m.component_data_objects(pyo.Constraint, active=True)))
    for nm in cho.THETA_NAMES:
        declare_fitted(getattr(m, nm))
    res = SolverFactory("pounce").solve(m)
    obj_p = pyo.value(m.obj)
    theta_p = {nm: float(pyo.value(getattr(m, nm))) for nm in cho.THETA_NAMES}
    theta_true = {nm: float(getattr(p_true, nm)) for nm in cho.THETA_NAMES}

    # committed IPOPT (MA57) benchmark for the same model
    bench = json.loads((MAIN / "benchmarks/cho/ipopt_ma57.json").read_text())[0]

    # live IPOPT attempt (this box's IPOPT lacks MA57; MUMPS fails on CHO)
    live_ipopt = {}
    try:
        mi = cho.build_full_model(batches, nfe=cho.NFE, ncp=cho.NCP)
        ri = SolverFactory("ipopt").solve(mi, load_solutions=False,
                                          options={"tol": 1e-8})
        live_ipopt = {"termination": str(ri.solver.termination_condition),
                      "message": str(ri.solver.message)}
    except Exception as e:  # noqa: BLE001
        live_ipopt = {"error": str(e)[:200]}

    # (2) covariance via pounce
    ndata = count_ndata(cho, m, batches)
    sigma_sq = obj_p / (ndata - len(cho.THETA_NAMES))
    cov = covariance(m, sigma_sq=sigma_sq, hessian="lagrangian")
    se_p = {nm: float(cov.std_err[getattr(m, nm)]) for nm in cho.THETA_NAMES}
    corr_p = np.array([[cov.correlation[getattr(m, a), getattr(m, b)]
                        for b in cho.THETA_NAMES] for a in cho.THETA_NAMES])
    on_bound = [nm for nm in cho.THETA_NAMES
                if se_p[nm] == 0.0]

    # independent CHO reference (PyNumero + scipy) -- needs duals, so a twin
    # plain solve to populate the dual suffix
    m2 = cho.build_full_model(batches, nfe=cho.NFE, ncp=cho.NCP)
    m2.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    SolverFactory("pounce").solve(m2)
    fitted2 = [getattr(m2, nm) for nm in cho.THETA_NAMES]
    Cref, active_ref, rh_eigs = covariance_reference(
        m2, fitted2, sigma_sq, m2.dual)
    se_ref = {nm: float(np.sqrt(Cref[i, i]))
              for i, nm in enumerate(cho.THETA_NAMES)}
    Cp = cov.matrix
    free = [i for i, nm in enumerate(cho.THETA_NAMES) if se_p[nm] > 0
            and se_ref[nm] > 0]
    se_reldiffs = {cho.THETA_NAMES[i]:
                   abs(se_p[cho.THETA_NAMES[i]] - se_ref[cho.THETA_NAMES[i]])
                   / abs(se_ref[cho.THETA_NAMES[i]]) for i in free}
    cond = (float(rh_eigs.max() / rh_eigs.min())
            if rh_eigs.size and rh_eigs.min() > 0 else float("inf"))

    payload = {
        "problem": "CHO_parmest",
        "size": {"n_vars": n_vars, "n_cons": n_cons,
                 "nfe": cho.NFE, "ncp": cho.NCP, "n_batches": cho.N_BATCHES,
                 "n_data": ndata},
        "point_estimate": {
            "pounce_status": str(res.solver.termination_condition),
            "pounce_objective": obj_p,
            "benchmark_ipopt_ma57_objective": bench["objective"],
            "benchmark_ipopt_ma57_iters": bench["iterations"],
            "obj_absdiff_pounce_vs_benchmark":
                abs_err(obj_p, bench["objective"]),
            "obj_relerr_pounce_vs_benchmark":
                rel_err(obj_p, bench["objective"]),
            "live_ipopt": live_ipopt,
            "theta_pounce": theta_p,
            "theta_true": theta_true,
            "theta_max_relerr_vs_true": max(
                rel_err(theta_p[nm], theta_true[nm])
                for nm in cho.THETA_NAMES if nm not in on_bound),
        },
        "covariance_method_controls": {
            "linear_regression": control_linear(),
            "nonlinear_expfit": control_nonlinear(),
        },
        "cho_covariance": {
            "sigma_sq": float(sigma_sq),
            "std_err_pounce": se_p,
            "std_err_independent_ref": se_ref,
            "parameters_on_bound_unidentifiable": on_bound,
            "correlation_pounce": corr_p.tolist(),
            "reduced_hessian_condition_number": cond,
            "reduced_hessian_eig_min": (float(rh_eigs.min())
                                        if rh_eigs.size else None),
            "reduced_hessian_eig_max": (float(rh_eigs.max())
                                        if rh_eigs.size else None),
            "std_err_reldiff_pounce_vs_ref_freeparams": se_reldiffs,
            "std_err_reldiff_max": (max(se_reldiffs.values())
                                    if se_reldiffs else None),
        },
    }
    print(json.dumps(payload, indent=2, default=float))
    dump_result("p1_cho", payload)


if __name__ == "__main__":
    main()
