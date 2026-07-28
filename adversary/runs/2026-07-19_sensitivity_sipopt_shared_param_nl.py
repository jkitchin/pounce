"""Adversary cross-check: sIPOPT CLI suffixes (sens_sol_state_1) on a .nl where
the PARAMETER APPEARS IN BOTH THE OBJECTIVE AND THE CONSTRAINT, nonlinearly.

Family: sensitivity   Class: NLP parametric sensitivity via the sIPOPT CLI path,
                             parameter shared between objective and constraints
Source: sIPOPT / NLP sensitivity theory — Pirnay, Lopez-Negrete & Biegler,
        "Optimal sensitivity based on IPOPT", Math. Prog. Comp. 4 (2012) 307-331
        (the sens_state_1 / sens_state_value_1 / sens_init_constr suffix
        protocol); Fiacco (1983) Thm 2.1 for dx/dp = -(KKT)^-1 d(grad L)/dp.

Distinct from the existing 2026-06-09 sIPOPT run, which reused the upstream
`parametric.nl` fixture where p enters ONLY the constraints and x*(p) is AFFINE
(so the first-order predictor is exact and cannot distinguish a correct
sensitivity from a lucky re-solve).  Here:
  * p enters the OBJECTIVE twice — once in (x0-p)^2 and once nonlinearly as
    exp(p)*x1^2 — AND the CONSTRAINT twice (bilinear 0.5*p*x1, and the rhs).
    A sensitivity implementation that forgets the objective's d(grad_x f)/dp
    term, or the constraint Jacobian's dp term, gets a WRONG answer here.
  * x*(p) is genuinely NONLINEAR in p, so the first-order predictor is NOT the
    re-solve; we check pounce against the FD predictor and separately confirm
    the predictor/re-solve gap is O(Delta_p^2) (quadratic shrinkage), which is
    what proves the returned quantity is a true first derivative.

Problem (p is a variable pinned by the sens_init_constr, sIPOPT style):
    min  (x0 - p)^2 + (x1 - 1)^2 + exp(p) * x1^2
    s.t. x0 + 2*x1 + 0.5*p*x1 = 1 + p
         p = 1                          (nominal, pinned)
    perturbed parameter value: p = 1 + Delta_p

Oracle: an independent CLOSED-FORM re-solve (never pounce).  For fixed p the
objective is a strictly convex quadratic in x and the constraint is linear in x,
so x*(p) is the exact solution of a 3x3 KKT linear system solved in float64 —
machine-accurate, unlike an SLSQP re-solve whose ~1e-9 solution noise would
pollute the finite difference.  Central FD dx/dp = (x*(p+d) - x*(p-d)) / 2d,
with a delta plateau sweep over 1e-2 ... 1e-9.  A scipy SLSQP re-solve is run
alongside purely to confirm the closed form is the true optimum.
"""
import os
import re
import subprocess
import time

import numpy as np
import pyomo.environ as pyo
from scipy.optimize import minimize

CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
WORK = "/tmp/adv_sens_shared_param"
os.makedirs(WORK, exist_ok=True)

P_NOM = 1.0
DELTAS_P = [0.2, 0.02]          # two perturbation sizes -> quadratic-gap check
FD_DELTAS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]


# ------------------------------------------------------------------ oracle
def obj(x, p):
    return (x[0] - p) ** 2 + (x[1] - 1.0) ** 2 + np.exp(p) * x[1] ** 2


def resolve(p):
    """EXACT re-solve at parameter p: for fixed p this is an equality-constrained
    strictly convex QP in x, so the KKT system is 3x3 linear and solved exactly.

        min  x' Q x / 2 - g' x   s.t.  a' x = r
        Q = diag(2, 2 + 2 e^p),  g = (2p, 2),  a = (1, 2 + 0.5 p),  r = 1 + p
    """
    Q = np.diag([2.0, 2.0 + 2.0 * np.exp(p)])
    gv = np.array([2.0 * p, 2.0])
    a = np.array([1.0, 2.0 + 0.5 * p])
    r = 1.0 + p
    K = np.block([[Q, a.reshape(2, 1)], [a.reshape(1, 2), np.zeros((1, 1))]])
    rhs = np.concatenate([gv, [r]])
    return np.linalg.solve(K, rhs)[:2]


def resolve_slsqp(p):
    """Independent numerical re-solve, only to validate the closed form."""
    cons = [{"type": "eq",
             "fun": lambda x, p=p: np.array([x[0] + 2 * x[1] + 0.5 * p * x[1]
                                             - (1.0 + p)])}]
    rr = minimize(lambda x: obj(x, p), np.array([0.5, 0.2]),
                  constraints=cons, method="SLSQP",
                  options={"maxiter": 4000, "ftol": 1e-16})
    return rr.x


# ------------------------------------------------------------------ model
def build_nl(dp, tag):
    """Write a .nl carrying the three sIPOPT suffixes; return (nl, col)."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var([0, 1], initialize=0.5)
    m.p = pyo.Var(initialize=P_NOM)
    m.obj = pyo.Objective(expr=(m.x[0] - m.p) ** 2 + (m.x[1] - 1.0) ** 2
                          + pyo.exp(m.p) * m.x[1] ** 2)
    m.c1 = pyo.Constraint(expr=m.x[0] + 2 * m.x[1] + 0.5 * m.p * m.x[1]
                          == 1.0 + m.p)
    m.cpin = pyo.Constraint(expr=m.p == P_NOM)
    m.sens_state_1 = pyo.Suffix(direction=pyo.Suffix.EXPORT,
                                datatype=pyo.Suffix.INT)
    m.sens_state_value_1 = pyo.Suffix(direction=pyo.Suffix.EXPORT,
                                      datatype=pyo.Suffix.FLOAT)
    m.sens_init_constr = pyo.Suffix(direction=pyo.Suffix.EXPORT,
                                    datatype=pyo.Suffix.INT)
    m.sens_state_1[m.p] = 1
    m.sens_state_value_1[m.p] = P_NOM + dp
    m.sens_init_constr[m.cpin] = 1
    nl = os.path.join(WORK, f"shared_{tag}.nl")
    m.write(nl, io_options={"symbolic_solver_labels": True})
    return nl, nl[:-3] + ".col"


def var_index_map(col):
    """.col gives the .nl variable order; map name -> nl index."""
    names = [ln.strip() for ln in open(col) if ln.strip()]
    return {n: i for i, n in enumerate(names)}


def parse_suffix(text, want):
    """Pull a real-var suffix block out of a .sol."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^suffix\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", lines[i])
        if m:
            count, tabline = int(m.group(2)), int(m.group(5))
            name = lines[i + 1].strip()
            body = i + 2 + tabline
            if name == want:
                out = {}
                for j in range(count):
                    idx, val = lines[body + j].split()
                    out[int(idx)] = float(val)
                return out
            i = body + count
            continue
        i += 1
    return None


def ninf(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), np.inf))


# --------------------------------------------------- oracle: nominal + FD sweep
t0 = time.perf_counter()
x_nom = resolve(P_NOM)
sweep = []
for d in FD_DELTAS:
    sweep.append((d, (resolve(P_NOM + d) - resolve(P_NOM - d)) / (2 * d)))
t_oracle = time.perf_counter() - t0

# plateau: the mid-range deltas must agree with each other; take the median-ish
plateau_rows = [s[1] for s in sweep if 1e-7 <= s[0] <= 1e-3]
plateau_spread = max(ninf(a, plateau_rows[0]) for a in plateau_rows)
DXDP_FD = plateau_rows[len(plateau_rows) // 2]     # middle of the plateau
x_nom_slsqp = resolve_slsqp(P_NOM)
closed_form_err = ninf(x_nom, x_nom_slsqp)

# --------------------------------------------------- pounce CLI, each Delta_p
results = []
for dp in DELTAS_P:
    tag = f"{dp:g}".replace(".", "p")
    nl, col = build_nl(dp, tag)
    vmap = var_index_map(col)
    sol = os.path.join(WORK, f"shared_{tag}.sol")
    t0 = time.perf_counter()
    proc = subprocess.run([CLI, nl, sol], capture_output=True, text=True)
    t_p = time.perf_counter() - t0
    assert proc.returncode == 0, f"pounce CLI failed ({proc.returncode})\n{proc.stderr}"
    txt = open(sol).read()
    sens = parse_suffix(txt, "sens_sol_state_1")
    assert sens is not None, "sens_sol_state_1 suffix missing from .sol"
    ix0, ix1, ip = vmap["x[0]"], vmap["x[1]"], vmap["p"]
    x_pert_pounce = np.array([sens[ix0], sens[ix1]])
    p_pert_pounce = sens[ip]

    # primal (nominal) solution from the .sol body, for a forward-solve check.
    # AMPL .sol layout: "Options", n_opt, n_opt ints, then
    # (n_con, n_dual, n_var, n_prim), then n_dual duals, then n_prim primals.
    body = [ln.strip() for ln in txt.splitlines()]
    k = body.index("Options")
    nopt = int(body[k + 1])
    q = k + 2 + nopt
    n_dual, n_prim = int(body[q + 1]), int(body[q + 3])
    pstart = q + 4 + n_dual
    prim = [float(body[pstart + j]) for j in range(n_prim)]
    x_nom_pounce = np.array([prim[ix0], prim[ix1]])

    x_pred = x_nom + DXDP_FD * dp             # first-order predictor (oracle)
    x_true = resolve(P_NOM + dp)              # full nonlinear re-solve
    results.append(dict(dp=dp, t=t_p, x_pounce=x_pert_pounce,
                        p_pounce=p_pert_pounce, x_nom_pounce=x_nom_pounce,
                        x_pred=x_pred, x_true=x_true,
                        err_pred=ninf(x_pert_pounce, x_pred),
                        err_true=ninf(x_pert_pounce, x_true),
                        gap_true=ninf(x_pred, x_true),
                        dxdp_implied=(x_pert_pounce - x_nom) / dp))

# quadratic-shrinkage check on the predictor/re-solve gap
ratio = results[0]["gap_true"] / max(results[1]["gap_true"], 1e-300)
expected_ratio = (DELTAS_P[0] / DELTAS_P[1]) ** 2      # 100 for 0.2 vs 0.02

# ---------------------------------------------------------------- report
print("=== problem: p in BOTH objective (x0-p)^2, exp(p)x1^2 AND constraint 0.5*p*x1 = 1+p ===")
print(f"x*(p=1) closed-form oracle = {x_nom}")
print(f"  cross-checked vs scipy SLSQP: inf_err = {closed_form_err:.2e}")
print("=== FD delta plateau sweep (oracle: exact closed-form re-solve, float64) ===")
print(f"{'delta':>8}  {'central FD dx/dp':>30}")
for d, v in sweep:
    print(f"{d:8.0e}  {str(np.round(v, 10)):>30}")
print(f"plateau (1e-7..1e-3) max spread = {plateau_spread:.2e}   -> dx/dp = {DXDP_FD}")
print("=== pounce CLI sens_sol_state_1 ===")
for r in results:
    print(f"-- Delta_p = {r['dp']}  (t={r['t']:.4f}s)")
    print(f"   x* nominal (pounce .sol)  = {r['x_nom_pounce']}  "
          f"err_vs_scipy={ninf(r['x_nom_pounce'], x_nom):.2e}")
    print(f"   sens_sol_state_1 x        = {r['x_pounce']}   (p slot = {r['p_pounce']})")
    print(f"   FD first-order predictor  = {r['x_pred']}")
    print(f"   full re-solve x*(p+Dp)    = {r['x_true']}")
    print(f"   implied dx/dp             = {r['dxdp_implied']}  "
          f"(FD dx/dp = {DXDP_FD})")
    print(f"   |pounce - FD predictor|   = {r['err_pred']:.2e}")
    print(f"   |pounce - full re-solve|  = {r['err_true']:.2e}")
    print(f"   |predictor - re-solve|    = {r['gap_true']:.2e}  (2nd-order truncation)")
print("=== quadratic shrinkage of the predictor/re-solve gap ===")
print(f"gap(0.2)/gap(0.02) = {ratio:.2f}   expected ~{expected_ratio:.0f} "
      f"if sens_sol_state_1 is a true FIRST derivative")

nom_ok = all(ninf(r["x_nom_pounce"], x_nom) < 1e-6 for r in results)
# tolerance scales with Delta_p: the first-order predictor is what sIPOPT returns
pred_ok = all(r["err_pred"] / r["dp"] < 1e-6 for r in results)   # error in dx/dp
quad_ok = 0.5 * expected_ratio < ratio < 2.0 * expected_ratio
plateau_ok = plateau_spread < 1e-6 and closed_form_err < 1e-7
print(f"nom_ok={nom_ok} pred_ok={pred_ok} quad_ok={quad_ok} plateau_ok={plateau_ok}")
print(f"t_oracle={t_oracle:.3f}s")
print("VERDICT: PASS" if (nom_ok and pred_ok and quad_ok and plateau_ok) else
      "VERDICT: FAIL (" + f"nom={nom_ok} pred={pred_ok} quad={quad_ok} plateau={plateau_ok})")
