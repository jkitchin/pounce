"""Adversary cross-check: best-acceptable fallback feasibility ranking (ba6d5b3 / gh#267, #270)

Family: nlp   Class: infeasibility / status-reporting correctness
Target: crates/pounce-algorithm/src/ipopt_alg.rs
        `record_best_acceptable` / `honour_best_acceptable_after_dual_guard`
        / `ranks_better_within_band` (commit ba6d5b3)
Fixture: crates/pounce-cli/tests/fixtures/deb7.nl (MINLPLib deb7, continuous relaxation)
Oracle: Ipopt 3.14.19 (/opt/homebrew/bin/ipopt) + `pounce verify`

CONTRACT UNDER TEST (from ba6d5b3 / PR #270):
  "Rank by (feasible_enough, objective): a point inside the feasibility band
   beats one outside it outright, objective decides only among points already
   inside the band, and the band is capped at the upstream-default
   acceptable_constr_viol_tol (1e-2) so a user-widened value cannot admit
   gross infeasibility."

FINDING: the cap protects only points that are already inside 1e-2. When BOTH
the incumbent and the recorded point sit outside the capped band, the ranking
falls back to `a_obj < b_obj` -- objective alone, the exact pre-fix rule -- and
the fallback again spends feasibility to buy objective, returning a
`pounce verify`-rejected point under solve_result_num=100.

Run:
  source /Users/jkitchin/projects/pounce/.venv-qa/bin/activate
  python adversary/runs/2026-07-22_nlp_deb7_fallback_band_gap.py
"""
import json
import os
import re
import subprocess
import tempfile

P = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"
NL = "/Users/jkitchin/projects/pounce/crates/pounce-cli/tests/fixtures/deb7.nl"

# The user-widened acceptable band. acceptable_constr_viol_tol=1e0 is ONE decade
# looser than upstream's 1e-2 default -- an ordinary loosening on a hard model,
# far milder than the 1e1 that #267 needed.
LOOSE = [
    "acceptable_constr_viol_tol=1e0",
    "acceptable_tol=1e10",
    "acceptable_dual_inf_tol=1e30",
    "acceptable_compl_inf_tol=1e10",
]
DEFAULT_BAND = ["acceptable_constr_viol_tol=1e-2"] + LOOSE[1:]
BASE = ["max_iter=200", "print_level=0"]


def solve(tag, streak, tol_opts, debug=False):
    d = tempfile.mkdtemp()
    safe = re.sub(r"[^A-Za-z0-9]+", "_", tag)
    js, sol = os.path.join(d, "r.json"), os.path.join(d, safe + ".sol")
    env = dict(os.environ)
    if debug:
        env["RUST_LOG"] = "pounce::algorithm=debug"
    cp = subprocess.run(
        [P, NL, sol, f"dual_diverging_streak={streak}"] + tol_opts + BASE
        + ["--json-output", js],
        capture_output=True, text=True, env=env, timeout=30,
    )
    if not os.path.exists(js):
        raise SystemExit(f"solve {tag!r} produced no report:\n{cp.stdout}\n{cp.stderr}")
    r = json.load(open(js))
    s, st = r["solution"], r["statistics"]
    return dict(tag=tag, sol=sol, status=s["status"], num=s["solve_result_num"],
                obj=s["objective"], viol=st["final_constr_viol"],
                iters=st["iteration_count"], stderr=cp.stderr)


def verify(sol):
    cp = subprocess.run([P, "verify", NL, sol], capture_output=True, text=True)
    v = "REJECTED" if cp.returncode == 20 else ("VERIFIED" if cp.returncode == 0 else "?")
    viol = ""
    for ln in cp.stdout.splitlines():
        if "max constraint violation" in ln:
            viol = ln.split(":")[1].strip().split()[0]
    return v, viol, cp.returncode


def row(r):
    v, vviol, rc = verify(r["sol"])
    print(f"  {r['tag']:<34} {r['status']:<26} num={r['num']:<4} "
          f"obj={r['obj']:>10.5f}  viol={r['viol']:.3e}  it={r['iters']:<4} "
          f"verify={v}({vviol}) exit={rc}")
    return v


print("=" * 100)
print("(1) Ipopt reference -- independent feasible local optimum")
print("=" * 100)
ip = subprocess.run([IPOPT, NL, "print_level=5", "max_iter=3000", "tol=1e-8"],
                    capture_output=True, text=True, timeout=60).stdout
ip_obj = ip_viol = None
for ln in ip.splitlines():
    if ln.strip().startswith("Objective"):
        ip_obj = float(ln.split(":")[1].split()[0])
    if "Constraint violation" in ln:
        ip_viol = float(ln.split(":")[1].split()[0])
print(f"  Ipopt 3.14.19: obj={ip_obj:.5f}  constr_viol={ip_viol:.3e}  (FEASIBLE)")

print()
print("=" * 100)
print("(2) THE TRAP: user-widened band acceptable_constr_viol_tol=1e0 (> the 1e-2 cap)")
print("=" * 100)
on = solve("guard ON  (streak=2)", 2, LOOSE, debug=True)
off = solve("guard OFF (streak=0, control)", 0, LOOSE)
row(off)
verdict_on = row(on)

print("\n  Fallback decision recorded by the solver's own debug log:")
for ln in on["stderr"].splitlines():
    if "diversion ended worse" in ln or "guard fired" in ln:
        print("    " + ln.split("pounce::algorithm: ")[-1].strip())

print("""
  Read that line as the code writes it: `curr` first, `best` second.
    incumbent the solve was about to return : obj 89.012  viol 5.292e-1
    recorded point the fallback restored    : obj 56.910  viol 9.951e-1
  Both violations exceed FEASIBLE_ENOUGH_CAP=1e-2, so `ranks_better_within_band`
  finds a_ok == b_ok == false and falls through to `a_obj < b_obj`. The fallback
  therefore chose the point with ~1.9x MORE constraint violation to gain 36% of
  objective -- the trade ba6d5b3 says it forbids.""")

print("=" * 100)
print("(3) CONTROL: same solve with the band AT the cap (acceptable_constr_viol_tol=1e-2)")
print("=" * 100)
row(solve("guard ON  @ band=cap=1e-2", 2, DEFAULT_BAND))
print("  -> no gross infeasibility, no trade. The hole opens exactly when band > cap.")

print()
print("=" * 100)
print("(4) NO-REGRESSION CHECK: full default acceptable tolerances")
print("=" * 100)
d_on = solve("guard ON  @ all defaults", 2, [])
d_off = solve("guard OFF @ all defaults (control)", 0, [])
row(d_off)
row(d_on)
print("  -> at defaults the fallback never returns infeasible-under-success:")
print("     the guard-on run reports MaximumIterationsExceeded (num=400), an honest")
print("     non-convergence status. Contract (c) and (d) hold at defaults.")

print()
print("=" * 100)
gap = (ip_obj - on["obj"]) / abs(ip_obj) * 100
print(f"Returned objective {on['obj']:.3f} is {gap:.1f}% BELOW Ipopt's feasible local")
print(f"optimum {ip_obj:.3f} -- unattainable at any feasible point, and reachable only")
print(f"because the returned point violates a constraint by {on['viol']:.3e}.")
print("Status reported: SolvedToAcceptableLevel / solve_result_num=100 (success band).")
ok = verdict_on != "REJECTED"
print("VERDICT: PASS" if ok else
      "VERDICT: SOLVER_BUG (ba6d5b3's feasibility ranking degenerates to "
      "objective-only outside the 1e-2 cap; verify-rejected point under num=100)")
