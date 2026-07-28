"""Adversary cross-check: INTERNAL CONSISTENCY of pounce's reported DUALS and
its reported SENSITIVITIES.

Family: sensitivity   Class: duals / multipliers / envelope theorem /
                             second-order predictor / CLI-vs-library parity

The premise: the multiplier y and the sensitivity dx/db are two views of the
same KKT factorization.  They must agree.  Concretely, for

    min  1/2 x'P x + c'x   s.t.  A x = b        (pounce sign convention,
                                                 verified below:
                                                 P x + c + A' y = 0)

the value function phi(b) = obj(x*(b)) satisfies

    dphi/db_i = -y_i                                     (Lagrange/duality)
              = (P x* + c)' (dx/db_i)                    (chain rule)
              = -(A' y)' (dx/db_i) = -y' (A dx/db_i)
              = -y' e_i = -y_i                           (consistency loop)

so a mismatch anywhere in that loop means one of {y, dx/db, obj} is wrong.

Sources
  * Fiacco, "Introduction to Sensitivity and Stability Analysis in Nonlinear
    Programming", Academic Press 1983, Thm 2.1 / Sec. 3.2 (basic sensitivity
    theorem; dz*/de = -(KKT)^{-1} d(grad L)/de).
  * Nocedal & Wright, "Numerical Optimization" 2e, Thm 12.9 and Sec. 16.5
    (multiplier = rate of change of the optimal objective w.r.t. the
    constraint right-hand side).
  * Envelope theorem: Milgrom & Segal, Econometrica 70 (2002) 583, Thm 1;
    textbook statement dphi/dp = dL/dp evaluated at (x*(p), y*(p)).
  * Pirnay, Lopez-Negrete & Biegler, "Optimal sensitivity based on IPOPT",
    Math. Prog. Comp. 4 (2012) 307-331 (the sens_state_1 /
    sens_state_value_1 / sens_init_constr / sens_sol_state_1 protocol).

Oracles (all independent of pounce)
  * closed-form KKT solves in float64 / exact rational sympy where available;
  * central finite differences with a STEP-SIZE SWEEP, requiring a stable
    plateau before the FD number is trusted.

Cases
  (a) multiplier y_i  ==  -dobj/db_i  ==  (P x + c)'(dx/db_i), on a QP with
      equalities AND a strictly-active inequality.
  (b) envelope theorem for a parameter in the objective only, in a constraint
      only, and in both.
  (c) second-order consistency: x + dx/db*delta vs x*(b+delta).
  (d) sIPOPT CLI (sens_sol_state_1) vs the library QpSensitivity on an
      EQUIVALENT problem; plus the .sol dual sign question.
  (e) sensitivity w.r.t. a parameter with NO effect must be EXACTLY zero.
"""

import os
import re
import subprocess
import time

import numpy as np

from pounce import solve_qp
from pounce.qp import QpSensitivity

np.set_printoptions(precision=12, suppress=False, linewidth=160)

CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
WORK = "/private/tmp/claude-501/-Users-jkitchin-projects-pounce/671a5f76-82be-4f1a-bac6-59f0cb187d8b/scratchpad/sens_consistency"
os.makedirs(WORK, exist_ok=True)

TIGHT = dict(tol=1e-12, max_iter=500)
FAILURES = []


def note(ok, label, detail):
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        FAILURES.append(f"{label}: {detail}")


def ninf(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), np.inf))


def plateau(sweep, lo, hi):
    """Return (value, spread) over the deltas in [lo, hi] of an FD sweep."""
    rows = [v for d, v in sweep if lo <= d <= hi]
    ref = np.asarray(rows[0], dtype=float)
    spread = max(float(np.max(np.abs(np.asarray(r, dtype=float) - ref)))
                 for r in rows)
    return np.asarray(rows[len(rows) // 2], dtype=float), spread


# =====================================================================
# Shared oracle: exact equality-constrained QP KKT solve (closed form).
# =====================================================================
def kkt_solve(P, c, A, b):
    n, m = P.shape[0], A.shape[0]
    K = np.block([[P, A.T], [A, np.zeros((m, m))]])
    rhs = np.concatenate([-c, b])
    s = np.linalg.solve(K, rhs)
    return s[:n], s[n:]


# =====================================================================
# CASE 0: pin down the documented sign convention.
# =====================================================================
print("=" * 74)
print("CASE 0 -- sign convention of the reported equality multiplier y")
print("=" * 74)
P0 = np.eye(2)
c0 = np.zeros(2)
A0 = np.array([[1.0, 1.0]])
b0 = np.array([2.0])
r0 = solve_qp(P=P0, c=c0, A=A0, b=b0, **TIGHT)
stat = P0 @ r0.x + c0 + A0.T @ r0.y
print(f"  x*={r0.x}  y={r0.y}")
print(f"  ||P x + c + A' y||_inf = {np.max(np.abs(stat)):.3e}   "
      f"(convention: L = f + y'(Ax-b), so dobj/db = -y)")
note(np.max(np.abs(stat)) < 1e-8, "stationarity with +A'y",
     f"{np.max(np.abs(stat)):.3e}")
SIGN = -1.0   # dobj/db_i = SIGN * y_i


# =====================================================================
# CASE (a): y_i == -dobj/db_i == (Px+c)'(dx/db_i), with an ACTIVE inequality.
# =====================================================================
print()
print("=" * 74)
print("CASE (a) -- multiplier vs dobj/db vs QpSensitivity dx/db")
print("=" * 74)

n = 5
rng = np.random.default_rng(20260722)
M = rng.standard_normal((n, n))
Pa = M @ M.T + 2.0 * np.eye(n)          # SPD, well conditioned
ca = np.array([1.0, -2.0, 0.5, 3.0, -1.5])
Aa = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
               [2.0, -1.0, 0.0, 1.0, 0.5],
               [0.0, 0.0, 1.0, -1.0, 2.0]])
ba = np.array([1.0, 0.5, -0.25])
# an inequality chosen to be STRICTLY active at the optimum
Ga = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
xe, _ = kkt_solve(Pa, ca, Aa, ba)
ha = np.array([xe[0] - 0.35])            # forces x0 <= xe0-0.35, strictly active

ra = solve_qp(P=Pa, c=ca, A=Aa, b=ba, G=Ga, h=ha, **TIGHT)
print(f"  status={ra.status}  obj={ra.obj:.15e}")
print(f"  y={ra.y}   z={ra.z}")
note(ra.status == "optimal", "solve status", ra.status)
note(float(ra.z[0]) > 1e-6, "inequality strictly active",
     f"z0={float(ra.z[0]):.6e}, slack={float((ha - Ga @ ra.x)[0]):.2e}")

# exact oracle: with the inequality active it is an equality-constrained QP
Aact = np.vstack([Aa, Ga])
bact = np.concatenate([ba, ha])
x_act, mu_act = kkt_solve(Pa, ca, Aact, bact)
note(ninf(x_act, ra.x) < 1e-8, "primal vs closed-form active-set KKT",
     f"{ninf(x_act, ra.x):.2e}")
note(ninf(mu_act[:3], ra.y) < 1e-7, "y vs closed-form active-set KKT",
     f"{ninf(mu_act[:3], ra.y):.2e}")

# --- oracle A: FD sweep of the value function w.r.t. each b_i (closed form) --
def phi_a(bvec):
    xx, _ = kkt_solve(Pa, ca, np.vstack([Aa, Ga]),
                      np.concatenate([bvec, ha]))
    return 0.5 * xx @ Pa @ xx + ca @ xx


FD = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
dphi_db = np.zeros(3)
print("  FD sweep of dobj/db (closed-form re-solve oracle):")
for i in range(3):
    sweep = []
    for d in FD:
        bp, bm = ba.copy(), ba.copy()
        bp[i] += d
        bm[i] -= d
        sweep.append((d, (phi_a(bp) - phi_a(bm)) / (2 * d)))
    val, spr = plateau(sweep, 1e-6, 1e-3)
    dphi_db[i] = val
    print(f"    b[{i}]: plateau value {val:+.12f}  spread {spr:.2e}")
    note(spr < 1e-6, f"FD plateau b[{i}]", f"spread={spr:.2e}")

print(f"  dobj/db (FD)   = {dphi_db}")
print(f"  -y (pounce)    = {SIGN * np.asarray(ra.y)}")
note(ninf(dphi_db, SIGN * np.asarray(ra.y)) < 1e-6,
     "(a1) y_i == -dobj/db_i", f"{ninf(dphi_db, SIGN * np.asarray(ra.y)):.2e}")

# --- QpSensitivity dx/db, one pin at a time ---------------------------------
sa = QpSensitivity(P=Pa, c=ca, A=Aa, b=ba, G=Ga, h=ha, **TIGHT)
note(ninf(sa.x, ra.x) < 1e-8, "QpSensitivity x == solve_qp x",
     f"{ninf(sa.x, ra.x):.2e}")
note(sa.active_indices.inequalities == (0,), "sensitivity active set",
     str(sa.active_indices))
note(sa.weakly_active_indices.inequalities == (), "strict complementarity",
     str(sa.weakly_active_indices))

dxdb = np.column_stack([sa.parametric_step([i], [1.0]) for i in range(3)])
print(f"  dx/db (QpSensitivity) =\n{dxdb}")

# exact dx/db from the active-set KKT (rhs = e_i in the equality block)
Kact = np.block([[Pa, Aact.T], [Aact, np.zeros((4, 4))]])
dxdb_exact = np.zeros((n, 3))
for i in range(3):
    rhs = np.zeros(n + 4)
    rhs[n + i] = 1.0
    dxdb_exact[:, i] = np.linalg.solve(Kact, rhs)[:n]
note(ninf(dxdb, dxdb_exact) < 1e-8, "(a2) dx/db vs exact KKT back-solve",
     f"{ninf(dxdb, dxdb_exact):.2e}")

# chain rule: (Px+c)' dx/db_i must equal dobj/db_i must equal -y_i
grad = Pa @ np.asarray(ra.x) + ca
chain = grad @ dxdb
print(f"  (Px+c)'dx/db   = {chain}")
note(ninf(chain, dphi_db) < 1e-6, "(a3) chain rule == FD dobj/db",
     f"{ninf(chain, dphi_db):.2e}")
note(ninf(chain, SIGN * np.asarray(ra.y)) < 1e-6, "(a4) chain rule == -y",
     f"{ninf(chain, SIGN * np.asarray(ra.y)):.2e}")

# A dx/db_i must equal e_i (feasibility of the sensitivity direction)
note(ninf(Aa @ dxdb, np.eye(3)) < 1e-9, "(a5) A dx/db == I",
     f"{ninf(Aa @ dxdb, np.eye(3)):.2e}")
note(float(np.max(np.abs(Ga @ dxdb))) < 1e-9,
     "(a6) active inequality stays active along dx/db",
     f"{float(np.max(np.abs(Ga @ dxdb))):.2e}")


# =====================================================================
# CASE (b): envelope theorem, three placements of the parameter.
# =====================================================================
print()
print("=" * 74)
print("CASE (b) -- envelope theorem dphi/dp = dL/dp at the optimum")
print("=" * 74)

Pb = np.array([[4.0, 1.0, 0.0],
               [1.0, 3.0, 0.5],
               [0.0, 0.5, 2.0]])
cb0 = np.array([-1.0, 2.0, 0.5])
Ab = np.array([[1.0, 1.0, 1.0],
               [1.0, -1.0, 2.0]])
bb0 = np.array([1.0, 0.5])
dvec = np.array([0.75, -0.25, 1.5])     # dc/dp
evec = np.array([1.0, -2.0])            # db/dp
P_NOM = 0.7


def solve_b(p, in_obj, in_con):
    c = cb0 + (p * dvec if in_obj else 0.0)
    b = bb0 + (p * evec if in_con else 0.0)
    x, y = kkt_solve(Pb, c, Ab, b)
    return x, y, 0.5 * x @ Pb @ x + c @ x


for label, in_obj, in_con in [("objective only", True, False),
                              ("constraint only", False, True),
                              ("BOTH", True, True)]:
    c = cb0 + (P_NOM * dvec if in_obj else 0.0)
    b = bb0 + (P_NOM * evec if in_con else 0.0)
    r = solve_qp(P=Pb, c=c, A=Ab, b=b, **TIGHT)
    # envelope prediction from POUNCE's OWN reported quantities:
    #   L = f(x,p) + y'(A x - b(p));  dL/dp = d'x  -  y'e
    pred = (dvec @ np.asarray(r.x) if in_obj else 0.0) \
        - (np.asarray(r.y) @ evec if in_con else 0.0)
    sweep = []
    for d in FD:
        _, _, up = solve_b(P_NOM + d, in_obj, in_con)
        _, _, um = solve_b(P_NOM - d, in_obj, in_con)
        sweep.append((d, (up - um) / (2 * d)))
    val, spr = plateau(sweep, 1e-6, 1e-3)
    print(f"  p in {label:16s}: dphi/dp(FD)={val:+.12f} (spread {spr:.1e})  "
          f"envelope(pounce y,x)={pred:+.12f}")
    note(spr < 1e-6, f"(b) FD plateau [{label}]", f"{spr:.2e}")
    note(abs(val - pred) < 1e-7, f"(b) envelope theorem [{label}]",
         f"|FD - dL/dp| = {abs(val - pred):.2e}")

    # cross-check via QpSensitivity for the constraint-carried part
    if in_con:
        s = QpSensitivity(P=Pb, c=c, A=Ab, b=b, **TIGHT)
        dxdp = s.parametric_step([0, 1], list(evec))
        gradp = Pb @ np.asarray(r.x) + c
        # total derivative: partial_p f + grad_x f . dx/dp
        tot = (dvec @ np.asarray(r.x) if in_obj else 0.0) + gradp @ dxdp
        note(abs(tot - val) < 1e-7,
             f"(b) total-derivative via QpSensitivity [{label}]",
             f"|FD - (df/dp + grad.dx/dp)| = {abs(tot - val):.2e}")


# =====================================================================
# CASE (c): second-order consistency of the predictor.
# =====================================================================
print()
print("=" * 74)
print("CASE (c) -- predictor error scaling")
print("=" * 74)
print("  For a QP with a FIXED active set x*(b) is AFFINE in b, so the")
print("  first-order predictor is EXACT (error ~ machine eps, i.e. better")
print("  than O(delta^2)).  We check that, then check the genuinely")
print("  nonlinear NLP path in case (d) for true O(delta^2) shrinkage.")
dirn = np.array([1.0, -0.5, 0.25])
prev = None
for delta in [1e-1, 1e-2, 1e-3]:
    step = sa.parametric_step([0, 1, 2], list(delta * dirn))
    x_pred = np.asarray(sa.x) + step
    x_true, _ = kkt_solve(Pa, ca, Aact,
                          np.concatenate([ba + delta * dirn, ha]))
    err = ninf(x_pred, x_true)
    ratio = "" if prev is None else f"  ratio={prev / max(err, 1e-300):.2e}"
    print(f"    delta={delta:8.0e}   ||pred-true||_inf = {err:.3e}{ratio}")
    note(err < 1e-9, f"(c) affine predictor exact at delta={delta:g}",
         f"{err:.2e}")
    prev = err
# and confirm the active set really did not move
for delta in [1e-1, 1e-2, 1e-3]:
    rr = solve_qp(P=Pa, c=ca, A=Aa, b=ba + delta * dirn, G=Ga, h=ha, **TIGHT)
    note(float(rr.z[0]) > 1e-8, f"(c) active set unchanged at delta={delta:g}",
         f"z0={float(rr.z[0]):.3e}")


# =====================================================================
# CASE (e): a parameter with NO effect must give EXACTLY zero.
# =====================================================================
print()
print("=" * 74)
print("CASE (e) -- dummy parameter must give exactly zero sensitivity")
print("=" * 74)
# variables (x0..x4, w).  w appears only in its own equality and its own
# separable objective term, so d x_{0..4} / d b_dummy is IDENTICALLY zero.
Pe = np.zeros((n + 1, n + 1))
Pe[:n, :n] = Pa
Pe[n, n] = 1.0
ce = np.concatenate([ca, [0.0]])
Ae = np.zeros((4, n + 1))
Ae[:3, :n] = Aa
Ae[3, n] = 1.0                 # w = b3   <- the dummy parameter
be = np.concatenate([ba, [0.3]])
Ge = np.zeros((1, n + 1))
Ge[0, :n] = Ga
se = QpSensitivity(P=Pe, c=ce, A=Ae, b=be, G=Ge, h=ha, **TIGHT)
note(ninf(np.asarray(se.x)[:n], ra.x) < 1e-8,
     "(e) augmented problem reproduces x*", f"{ninf(np.asarray(se.x)[:n], ra.x):.2e}")
step_dummy = se.parametric_step([3], [1.0])
print(f"  dx/db_dummy = {step_dummy}")
xpart = np.max(np.abs(step_dummy[:n]))
note(xpart == 0.0, "(e) dummy sensitivity on x is EXACTLY zero",
     f"max|dx/db_dummy| = {xpart:.3e} (exact-zero required)")
note(abs(step_dummy[n] - 1.0) < 1e-8, "(e) dw/db_dummy == 1 (to IPM tol)",
     f"{step_dummy[n]:.15f}")


# =====================================================================
# CASE (d): sIPOPT CLI (sens_sol_state_1) vs library QpSensitivity.
# =====================================================================
print()
print("=" * 74)
print("CASE (d) -- CLI sens_sol_state_1 vs library QpSensitivity")
print("=" * 74)

try:
    import pyomo.environ as pyo
    HAVE_PYOMO = True
except Exception as exc:                      # pragma: no cover
    HAVE_PYOMO = False
    print(f"  pyomo unavailable ({exc}); skipping the CLI leg")


def parse_suffix(text, want):
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


def sol_primal_dual(text, n_expect):
    body = [ln.strip() for ln in text.splitlines()]
    k = body.index("Options")
    nopt = int(body[k + 1])
    q = k + 2 + nopt
    n_dual, n_prim = int(body[q + 1]), int(body[q + 3])
    duals = [float(body[q + 4 + j]) for j in range(n_dual)]
    prim = [float(body[q + 4 + n_dual + j]) for j in range(n_prim)]
    return np.array(prim), np.array(duals)


if HAVE_PYOMO:
    # --- The EQUIVALENT problem, small and fully closed-form -----------------
    #     min 1/2 x'Pd x + cd'x   s.t.  Ad x = bd(p),  bd = bd0 + p*ed
    Pd = np.array([[3.0, 0.5], [0.5, 2.0]])
    cd = np.array([-1.0, 0.5])
    Ad = np.array([[1.0, 2.0]])
    bd0 = np.array([1.0])
    ed = np.array([1.0])
    PD_NOM = 0.25
    DP = 0.1

    bd = bd0 + PD_NOM * ed
    x_lib_ref, y_lib_ref = kkt_solve(Pd, cd, Ad, bd)
    r_lib = solve_qp(P=Pd, c=cd, A=Ad, b=bd, **TIGHT)
    s_lib = QpSensitivity(P=Pd, c=cd, A=Ad, b=bd, **TIGHT)
    dx_lib = s_lib.parametric_step([0], [DP * float(ed[0])])
    x_lib_pert = np.asarray(s_lib.x) + dx_lib
    x_true_pert, _ = kkt_solve(Pd, cd, Ad, bd0 + (PD_NOM + DP) * ed)
    print(f"  library: x*={np.asarray(s_lib.x)}  y={np.asarray(r_lib.y)}")
    print(f"  library: x*(p+dp) predicted = {x_lib_pert}")
    print(f"  exact  : x*(p+dp)           = {x_true_pert}")
    note(ninf(x_lib_pert, x_true_pert) < 1e-10,
         "(d) library predictor exact (affine in b)",
         f"{ninf(x_lib_pert, x_true_pert):.2e}")

    # --- same problem as a .nl with a pinned parameter variable p ------------
    m = pyo.ConcreteModel()
    m.x = pyo.Var([0, 1], initialize=0.0)
    m.p = pyo.Var(initialize=PD_NOM)
    m.obj = pyo.Objective(
        expr=0.5 * (Pd[0, 0] * m.x[0] ** 2 + 2 * Pd[0, 1] * m.x[0] * m.x[1]
                    + Pd[1, 1] * m.x[1] ** 2)
        + cd[0] * m.x[0] + cd[1] * m.x[1])
    m.c1 = pyo.Constraint(
        expr=Ad[0, 0] * m.x[0] + Ad[0, 1] * m.x[1] == bd0[0] + ed[0] * m.p)
    m.cpin = pyo.Constraint(expr=m.p == PD_NOM)
    m.sens_state_1 = pyo.Suffix(direction=pyo.Suffix.EXPORT,
                                datatype=pyo.Suffix.INT)
    m.sens_state_value_1 = pyo.Suffix(direction=pyo.Suffix.EXPORT,
                                      datatype=pyo.Suffix.FLOAT)
    m.sens_init_constr = pyo.Suffix(direction=pyo.Suffix.EXPORT,
                                    datatype=pyo.Suffix.INT)
    m.sens_state_1[m.p] = 1
    m.sens_state_value_1[m.p] = PD_NOM + DP
    m.sens_init_constr[m.cpin] = 1
    nl = os.path.join(WORK, "equiv.nl")
    m.write(nl, io_options={"symbolic_solver_labels": True})
    col = nl[:-3] + ".col"
    row = nl[:-3] + ".row"
    vmap = {s.strip(): i for i, s in enumerate(open(col)) if s.strip()}
    rmap = {s.strip(): i for i, s in enumerate(open(row)) if s.strip()}
    sol = os.path.join(WORK, "equiv.sol")
    t0 = time.perf_counter()
    proc = subprocess.run([CLI, nl, sol], capture_output=True, text=True)
    t_cli = time.perf_counter() - t0
    note(proc.returncode == 0, "(d) CLI exit code",
         f"{proc.returncode} ({t_cli:.3f}s)\n{proc.stderr[:400]}")
    txt = open(sol).read()
    prim, duals = sol_primal_dual(txt, 3)
    ix0, ix1, ip = vmap["x[0]"], vmap["x[1]"], vmap["p"]
    x_cli_nom = np.array([prim[ix0], prim[ix1]])
    note(ninf(x_cli_nom, x_lib_ref) < 1e-8, "(d) CLI nominal x == library x",
         f"{ninf(x_cli_nom, x_lib_ref):.2e}")

    sens = parse_suffix(txt, "sens_sol_state_1")
    note(sens is not None, "(d) sens_sol_state_1 present", str(sens is not None))
    if sens is not None:
        x_cli_pert = np.array([sens[ix0], sens[ix1]])
        print(f"  CLI sens_sol_state_1 x*(p+dp) = {x_cli_pert}  "
              f"(p slot = {sens.get(ip)})")
        note(ninf(x_cli_pert, x_lib_pert) < 1e-7,
             "(d) CLI sens_sol_state_1 == library QpSensitivity predictor",
             f"{ninf(x_cli_pert, x_lib_pert):.2e}")
        note(ninf(x_cli_pert, x_true_pert) < 1e-7,
             "(d) CLI sens_sol_state_1 == exact re-solve",
             f"{ninf(x_cli_pert, x_true_pert):.2e}")
        note(abs(sens.get(ip, PD_NOM) - (PD_NOM + DP)) < 1e-7,
             "(d) parameter slot carries the perturbed value",
             f"{sens.get(ip)}")

    # --- the .sol DUAL SIGN question ----------------------------------------
    # AMPL convention: for `min`, duals reported by ipopt satisfy
    # grad f = sum lambda_j grad c_j, i.e. lambda = -y in pounce's convention.
    ic1 = rmap["c1"]
    y_pounce = float(y_lib_ref[0])
    print(f"  library y (pounce convention, P x + c + A'y = 0) = {y_pounce:+.12f}")
    print(f"  .sol dual for c1                                = {duals[ic1]:+.12f}")
    dphi_dp_fd = []
    for d in FD:
        xp, _ = kkt_solve(Pd, cd, Ad, bd0 + (PD_NOM + d) * ed)
        xm, _ = kkt_solve(Pd, cd, Ad, bd0 + (PD_NOM - d) * ed)
        up = 0.5 * xp @ Pd @ xp + cd @ xp
        um = 0.5 * xm @ Pd @ xm + cd @ xm
        dphi_dp_fd.append((d, (up - um) / (2 * d)))
    v, spr = plateau(dphi_dp_fd, 1e-6, 1e-3)
    print(f"  dobj/db (FD, closed form) = {v:+.12f}  (spread {spr:.1e})")
    note(abs(v - (-y_pounce)) < 1e-7, "(d) library y == -dobj/db",
         f"{abs(v - (-y_pounce)):.2e}")
    # is the .sol dual the negation of the library dual?
    negated = abs(duals[ic1] + y_pounce) < 1e-7
    same = abs(duals[ic1] - y_pounce) < 1e-7
    print(f"  .sol dual is {'the NEGATION of' if negated else ('EQUAL to' if same else 'UNRELATED to')}"
          f" the library dual")

    # INDEPENDENT ORACLE for the .sol sign: run Ipopt on the SAME .nl.  The
    # AMPL convention is that the .sol dual equals dobj/db_i (the "shadow
    # price"), which the FD above computes without reference to any solver.
    ip_nl = os.path.join(WORK, "ip.nl")
    with open(ip_nl, "wb") as fh:
        fh.write(open(nl, "rb").read())
    ipr = subprocess.run(["/opt/homebrew/bin/ipopt", ip_nl, "-AMPL"],
                         capture_output=True, text=True, cwd=WORK)
    ip_sol = os.path.join(WORK, "ip.sol")
    if ipr.returncode == 0 and os.path.exists(ip_sol):
        ip_prim, ip_duals = sol_primal_dual(open(ip_sol).read(), 3)
        print(f"  ipopt .sol dual for c1                          = "
              f"{ip_duals[ic1]:+.12f}")
        note(abs(ip_duals[ic1] - v) < 1e-6,
             "(d) ipopt .sol dual == +dobj/db (AMPL shadow-price convention)",
             f"{abs(ip_duals[ic1] - v):.2e}")
        # NOT counted as a failure of THIS run: the .sol dual negation is a
        # separate, already-established finding.  Recorded here because this
        # run corroborates it from an independent direction (an FD value
        # function rather than a code read).
        sol_ok = abs(duals[ic1] - v) < 1e-6
        print(f"  [{'ok ' if sol_ok else 'KNOWN'}] (d) pounce .sol dual vs "
              f"+dobj/db: pounce={duals[ic1]:+.9f} vs dobj/db={v:+.9f} "
              f"(sum={duals[ic1] + v:.2e})")
        if not sol_ok:
            print("         -> pounce .sol dual is NEGATED relative to "
                  "ipopt / the AMPL shadow-price convention.")
            print("         -> SEPARATE known finding; the library dual is "
                  "self-consistent (checked above).")
            print("         -> sensitivity SUFFIXES are unaffected (primal "
                  "values; checked next).")
    else:
        print(f"  (ipopt leg skipped: rc={ipr.returncode})")
    # Does the sign question touch the SENSITIVITY suffix?  The suffix holds
    # PRIMAL variable values, which have no sign convention at all, so it
    # cannot.  Assert that explicitly by comparing to the exact re-solve.
    if sens is not None:
        note(ninf(x_cli_pert, x_true_pert) < 1e-7,
             "(d) sensitivity suffix UNAFFECTED by the dual sign question",
             f"primal values match exact re-solve to {ninf(x_cli_pert, x_true_pert):.2e}")

    # --- true O(delta^2) shrinkage on a genuinely NONLINEAR parametric NLP ---
    # min (x0-p)^2 + (x1-1)^2 + exp(p)*x1^2  s.t. x0 + 2x1 + 0.5*p*x1 = 1+p
    # (x*(p) is NOT affine in p, so the predictor error must shrink like dp^2)
    def nl_resolve(p):
        Q = np.diag([2.0, 2.0 + 2.0 * np.exp(p)])
        g = np.array([2.0 * p, 2.0])
        a = np.array([1.0, 2.0 + 0.5 * p])
        rr = 1.0 + p
        K = np.block([[Q, a.reshape(2, 1)], [a.reshape(1, 2), np.zeros((1, 1))]])
        return np.linalg.solve(K, np.concatenate([g, [rr]]))[:2]

    gaps = []
    for dp in [0.2, 0.02]:
        mm = pyo.ConcreteModel()
        mm.x = pyo.Var([0, 1], initialize=0.5)
        mm.p = pyo.Var(initialize=1.0)
        mm.obj = pyo.Objective(expr=(mm.x[0] - mm.p) ** 2 + (mm.x[1] - 1.0) ** 2
                               + pyo.exp(mm.p) * mm.x[1] ** 2)
        mm.c1 = pyo.Constraint(expr=mm.x[0] + 2 * mm.x[1] + 0.5 * mm.p * mm.x[1]
                               == 1.0 + mm.p)
        mm.cpin = pyo.Constraint(expr=mm.p == 1.0)
        mm.sens_state_1 = pyo.Suffix(direction=pyo.Suffix.EXPORT,
                                     datatype=pyo.Suffix.INT)
        mm.sens_state_value_1 = pyo.Suffix(direction=pyo.Suffix.EXPORT,
                                           datatype=pyo.Suffix.FLOAT)
        mm.sens_init_constr = pyo.Suffix(direction=pyo.Suffix.EXPORT,
                                         datatype=pyo.Suffix.INT)
        mm.sens_state_1[mm.p] = 1
        mm.sens_state_value_1[mm.p] = 1.0 + dp
        mm.sens_init_constr[mm.cpin] = 1
        tag = f"{dp:g}".replace(".", "p")
        nl2 = os.path.join(WORK, f"nlin_{tag}.nl")
        mm.write(nl2, io_options={"symbolic_solver_labels": True})
        v2 = {s.strip(): i for i, s in enumerate(open(nl2[:-3] + ".col"))
              if s.strip()}
        sol2 = os.path.join(WORK, f"nlin_{tag}.sol")
        pr = subprocess.run([CLI, nl2, sol2], capture_output=True, text=True)
        assert pr.returncode == 0, pr.stderr
        t2 = open(sol2).read()
        sx = parse_suffix(t2, "sens_sol_state_1")
        xp = np.array([sx[v2["x[0]"]], sx[v2["x[1]"]]])
        gaps.append((dp, ninf(xp, nl_resolve(1.0 + dp))))
        print(f"    nonlinear NLP dp={dp}: ||sens - exact re-solve|| = "
              f"{gaps[-1][1]:.3e}")
    ratio = gaps[0][1] / max(gaps[1][1], 1e-300)
    print(f"    gap ratio (dp 0.2 vs 0.02) = {ratio:.1f}   "
          f"(O(dp^2) predicts 100, O(dp) predicts 10)")
    note(30.0 < ratio < 400.0, "(d/c) predictor error is O(dp^2), not O(dp)",
         f"ratio={ratio:.1f}")


# =====================================================================
print()
print("=" * 74)
if FAILURES:
    print(f"VERDICT: FAIL ({len(FAILURES)} checks)")
    for f in FAILURES:
        print(f"  - {f}")
else:
    print("VERDICT: PASS")
