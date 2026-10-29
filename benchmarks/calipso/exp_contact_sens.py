"""Experiment 2: sensitivities at a contact switch (POUNCE side).

One implicit contact step of a falling particle: the next height is

    z+ = argmin_z  1/2 (z - zhat)^2   s.t.  z >= 0,    zhat = z0 + h v - h^2 g

so z+ = max(0, zhat), and with the initial velocity v as the parameter the
exact one-sided derivatives are known: dz+/dv = h where the particle is in
flight (zhat > 0), 0 where it rests on the floor (zhat < 0), and at the
switch v* (zhat = 0) the derivative is h for an increase of v and 0 for a
decrease. This is the shape a differentiable contact simulator, and
CALIPSO's policy tuning through rollouts, differentiates through.

Two transcriptions of the same step:

  bound  the contact is the bound z >= 0; the contact force is its
         multiplier. POUNCE's kink machinery (`degeneracy=`) acts here.
  comp   the force is a variable gamma >= 0 with z - zhat - gamma = 0 and
         z * gamma = 0 written as a constraint row -- how CALIPSO's examples
         write contact.

For each transcription and base point v in {v*, v* +- 1e-6, v* +- 1e-3},
every POUNCE estimate of z+(v + d), d in {+-1e-2, +-1e-3}, is scored two
ways: `val_err = |z_est - z_exact(v + d)| / |d|`, the error in what a user
gets, which is the primary number; and the slope (z_est - z_base) / d
against the exact one-sided slope. The slope inherits the interior-point
base offset (z_base sits ~sqrt(mu) above the floor at the kink), which is
why it is secondary -- the CLAUDE.md note on comparing slopes, not dx/d.
One JSON line per (form, base, method, d).
"""
import json
import warnings

import numpy as np
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers the "pounce" solver)
from pyomo_pounce import declare_sens_param, sens_jacobian, sens_solution

Z0, H, G = 1.0, 0.1, 9.81
V_STAR = (H * H * G - Z0) / H  # zhat = 0
D = 1e-3


def zhat(v):
    return Z0 + H * v - H * H * G


def exact_z(v):
    return max(0.0, zhat(v))


KINK_TOL = 1e-12  # zhat(V_STAR) evaluates to ~6e-17, not 0


def exact_slope(v, d):
    """One-sided derivative of z+ at v in the direction sign(d)."""
    zh = zhat(v)
    if zh > KINK_TOL or (abs(zh) <= KINK_TOL and d > 0):
        return H
    return 0.0


def build(form, v):
    m = pyo.ConcreteModel()
    m.v_hat = pyo.Param(initialize=v, mutable=True)
    m.v = pyo.Var(initialize=v)
    m.pin = pyo.Constraint(expr=m.v == m.v_hat)
    m.z = pyo.Var(bounds=(0, None), initialize=max(exact_z(v), 0.1))
    zh = Z0 + H * m.v - H * H * G
    if form == "bound":
        m.obj = pyo.Objective(expr=0.5 * (m.z - zh) ** 2)
    else:
        m.gamma = pyo.Var(bounds=(0, None), initialize=0.1)
        m.balance = pyo.Constraint(expr=m.z - zh - m.gamma == 0)
        m.comp = pyo.Constraint(expr=m.z * m.gamma == 0)
        m.obj = pyo.Objective(expr=0.0 * m.z)
    declare_sens_param(m.v_hat)
    return m


def main():
    for form in ("bound", "comp"):
        for off in (0.0, 1e-6, -1e-6, 1e-3, -1e-3):
            v = V_STAR + off
            m = build(form, v)
            res = pyo.SolverFactory("pounce").solve(m, options={"tol": 1e-10, "print_level": 0})
            z = pyo.value(m.z)
            base = {"form": form, "base_offset": off, "z": z, "z_exact": exact_z(v),
                    "status": str(res.solver.termination_condition)}
            try:
                base["dzdv_jacobian"] = float(sens_jacobian(m.z, wrt=m.v_hat))
            except Exception as e:
                base["dzdv_jacobian_error"] = repr(e)[:200]
            for d in (1e-2, -1e-2, D, -D):
                ex = exact_slope(v, d)
                for method, kw in (
                    ("linear/directional", {}),
                    ("linear/one_sided", {"degeneracy": "one_sided"}),
                    ("linear/release_all", {"degeneracy": "release_all"}),
                    ("fix_relax", {"mode": "fix_relax"}),
                ):
                    rec = dict(base, method=method, d=d, exact_slope=ex)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            est = sens_solution(m, [(m.v_hat, v + d)], **kw)
                        zv = est[m.z] if not hasattr(est, "get") else est.get(m.z, est.get("z"))
                        slope = (float(zv) - z) / d
                        rec.update(z_est=float(zv), slope=slope, err=abs(slope - ex),
                                   val_err=abs(float(zv) - exact_z(v + d)) / abs(d))
                    except Exception as e:
                        rec["error"] = repr(e)[:200]
                    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
