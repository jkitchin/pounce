"""Adversary option-space probe: tolerance monotonicity (feasible direction) across
both NLP arms -- the interior point (default) and algorithm=active-set-sqp, whose
step-QP path PR #1074/#1082 (gh#1061) just reworked -- plus the
sqp_qp_use_schur_updates kill-switch.
Family: option-space   Class: tolerance monotonicity + kill-switch ablation
Source: dev-notes/termination-status-invariants.md: "tightening any tolerance must
        never produce an infeasible or error verdict" -- on a model the same arm
        solves at defaults, a tolerance sweep may cost iterations or downgrade
        Solved -> Acceptable -> MaxIter, but must never land in the AMPL 200
        (infeasible) or 500 (error) bands. No oracle needed: the model and the arm
        are fixed; only an option moves.
Corpus: crates/pounce-cli/tests/fixtures/*.nl whose default solve on that arm is
        SolveSucceeded on engine=nlp/sqp-active-set in < 3 s (re-selected at run time).
"""
import glob, json, os, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
BIN = "target/release/pounce"; TMO = 20
ARMS = {"ipm": [], "sqp": ["algorithm=active-set-sqp"]}
TOLS = ["tol", "constr_viol_tol", "dual_inf_tol", "compl_inf_tol", "acceptable_tol", "acceptable_constr_viol_tol"]
VALS = ["1e-2", "1e-4", "1e-6", "1e-8", "1e-10", "1e-12"]
tmp = tempfile.mkdtemp(prefix="advopt_", dir="adversary/runs/scratch_1027")
def run(nl, opts):
    js = os.path.join(tmp, f"{abs(hash((nl, tuple(opts))))}.json")
    t0 = time.perf_counter()
    try:
        subprocess.run([BIN, nl, "--no-sol", "--json-output", js, "print_level=0", *opts], capture_output=True, timeout=TMO)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", None, None, None, TMO)
    dt = time.perf_counter() - t0
    try:
        s = json.load(open(js))["solution"]; st = json.load(open(js))["statistics"]
        return (s["status"], s.get("solve_result_num"), s.get("engine"), st.get("iteration_count"), dt)
    except Exception:
        return ("NO_JSON", None, None, None, dt)
fixtures = sorted(glob.glob("crates/pounce-cli/tests/fixtures/*.nl"))
with ThreadPoolExecutor(8) as ex:
    base = {(f, a): r for (f, a), r in zip([(f, a) for f in fixtures for a in ARMS],
            ex.map(lambda fa: run(fa[0], ARMS[fa[1]]), [(f, a) for f in fixtures for a in ARMS]))}
corpus = [(f, a) for (f, a), r in base.items() if r[0] == "SolveSucceeded" and r[2] in ("nlp", "sqp-active-set") and r[4] < 3]
print(f"fixtures={len(fixtures)} corpus (fixture,arm) pairs={len(corpus)}: ipm={sum(a=='ipm' for _,a in corpus)} sqp={sum(a=='sqp' for _,a in corpus)}")
jobs = []
for f, a in corpus:
    for k in TOLS:
        for v in VALS:
            jobs.append((f, a, f"{k}={v}"))
    if a == "sqp":
        for v in ("yes", "no"):
            jobs.append((f, a, f"sqp_qp_use_schur_updates={v}"))
t0 = time.perf_counter()
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(lambda j: run(j[0], ARMS[j[1]] + [j[2]]), jobs))
print(f"cells={len(jobs)} wall={time.perf_counter()-t0:.1f}s")
bad = []; tally = {}
for (f, a, o), r in zip(jobs, res):
    tally[r[0]] = tally.get(r[0], 0) + 1
    srn = r[1]
    if r[0] in ("TIMEOUT", "NO_JSON") or (srn is not None and (200 <= srn < 300 or srn >= 500)):
        bad.append((os.path.basename(f), a, o, r[0], srn, r[3], base[(f, a)][3]))
print("status tally:", dict(sorted(tally.items(), key=lambda kv: -kv[1])))
for b in bad:
    print("BAND-CROSS fixture=%s arm=%s opt=%s -> %s (srn=%s, it=%s; base it=%s)" % b)
print("VERDICT: PASS" if not bad else f"VERDICT: REVIEW ({len(bad)} cells crossed into the 200/500 bands or timed out; 400-band downgrades are allowed)")
