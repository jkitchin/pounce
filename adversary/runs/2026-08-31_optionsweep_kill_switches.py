"""Adversary probe: kill-switch ablation over the options merged this week.
Family: option-space (the arm that found gh#505 and gh#508)
Targets: PR #867 (feral_increase_quality / _retry), #822 (alpha_red_factor_min,
         limited_memory_initialization=history-max, limited_memory_ls_failure_restarts),
         #802 (neg_curv_escapes), #860 (sqp_qp_certify_second_order),
         #784 (start_point_conditioner), #836 (ma57_batched_backsolve),
         #823 (hessian_approximation=partitioned / fd_hessian)

No oracle: the corpus is the repo's own adjudicated .nl fixtures, and the
assertion is INTERNAL CONSISTENCY. A heuristic switch may cost iterations or
reroute a trajectory; what it must never do is take a model this same binary
solves at stock settings and turn it into an infeasibility or an error verdict
(the gh#505 class), because that is a contradiction in POUNCE's own semantics
rather than a disagreement with anyone else.

Objective moves are reported separately: on a nonconvex model a trajectory
change may legitimately land on a different local optimum, so those are for
review, not automatic failures.
"""
import subprocess, re, sys, os, json
from pathlib import Path

CLI = "./target/release/pounce"
FIXTURES = sorted(Path("crates/pounce-cli/tests/fixtures").glob("*.nl"))
FIXTURES = [f for f in FIXTURES if (f.with_suffix(".sol")).exists()][:28]

SETTINGS = {
    "stock":                      [],
    "feral_quality=no":           ["feral_increase_quality=no"],
    "feral_retry=no":             ["feral_increase_quality_retry=no"],
    "neg_curv_escapes=0":         ["neg_curv_escapes=0"],
    "qp_certify2nd=no":           ["sqp_qp_certify_second_order=no"],
    "lbfgs":                      ["hessian_approximation=limited-memory"],
    "lbfgs+historymax":           ["hessian_approximation=limited-memory",
                                   "limited_memory_initialization=history-max"],
    "lbfgs+upstream_alpha":       ["hessian_approximation=limited-memory",
                                   "alpha_red_factor_min=0.5"],
    "lbfgs+ls_restarts=1":        ["hessian_approximation=limited-memory",
                                   "limited_memory_ls_failure_restarts=1"],
    "start_cond=adam":            ["start_point_conditioner=adam"],
}

# AMPL solve_result bands: <100 solved, 200s infeasible, 400 limit, 500 error.
BAD_EXIT = re.compile(r"(Infeasible_Problem_Detected|Invalid_Number|Error_In_Step|"
                      r"Restoration_Failed|Converged to a point of local infeasibility|"
                      r"Iterates diverging)", re.I)
OBJ = re.compile(r"Objective\.+:\s+(\S+)")

def run(nl, opts, timeout=60):
    try:
        r = subprocess.run([CLI, str(nl), "-AMPL", *opts],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", None
    out = r.stdout + r.stderr
    m = OBJ.search(out)
    obj = float(m.group(1)) if m else None
    if "Optimal Solution Found" in out:            st = "optimal"
    elif "Solved To Acceptable Level" in out:      st = "acceptable"
    elif BAD_EXIT.search(out):                     st = "BAD:" + BAD_EXIT.search(out).group(1)[:28]
    elif "Maximum Number of Iterations" in out:    st = "maxiter"
    else:                                          st = "other"
    return st, obj

# Per-family baselines. `hessian_approximation=limited-memory` is a MODE
# change, not a kill switch: the L-BFGS leg has its own known verdicts (the
# option docs record deb7 and eigena2 ending Error_In_Step_Computation there on
# `main`), so comparing an lbfgs-family setting against the exact-Hessian leg
# manufactures regressions that predate this week. Each setting is diffed
# against the baseline in its OWN family.
BASE_OF = {lbl: ("lbfgs" if lbl.startswith("lbfgs") else "stock") for lbl in SETTINGS}

base = {}
print(f"running {len(FIXTURES)} fixtures x {len(SETTINGS)} settings ...", file=sys.stderr)
for nl in FIXTURES:
    base[("stock", nl.name)] = run(nl, [])
    base[("lbfgs", nl.name)] = run(nl, SETTINGS["lbfgs"])

regressions, obj_moves = [], []
for label, opts in SETTINGS.items():
    if label in ("stock", "lbfgs"):
        continue
    for nl in FIXTURES:
        b_st, b_obj = base[(BASE_OF[label], nl.name)]
        st, obj = run(nl, opts)
        if b_st in ("optimal", "acceptable") and (st.startswith("BAD") or st == "TIMEOUT"):
            regressions.append((nl.name, label, BASE_OF[label], b_st, st))
        if (b_obj is not None and obj is not None and b_st in ("optimal", "acceptable")
                and st in ("optimal", "acceptable")):
            denom = max(1.0, abs(b_obj))
            if abs(obj - b_obj) / denom > 1e-6:
                obj_moves.append((nl.name, label, b_obj, obj, (obj - b_obj) / denom))

print(f"\n=== verdict regressions (solved at stock -> infeasible/error) : {len(regressions)} ===")
for r in regressions:
    print(f"   {r[0]:<34} {r[1]:<22} (vs {r[2]}) {r[3]} -> {r[4]}")
print(f"\n=== objective moves > 1e-6 relative (review, not automatic failures) : {len(obj_moves)} ===")
for r in sorted(obj_moves, key=lambda t: -abs(t[4]))[:25]:
    print(f"   {r[0]:<34} {r[1]:<22} {r[2]:>14.7e} -> {r[3]:>14.7e}  rel {r[4]:+.2e}")

print(f"\nVERDICT: {'PASS' if not regressions else f'FAIL ({len(regressions)} verdict regressions)'}")
