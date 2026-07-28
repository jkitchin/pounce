"""Adversary cross-check: determinism, reentrancy and resource contracts.

Family: api (contracts) / qp
Class:  bit-reproducibility, thread safety, parallel-answer-invariance,
        memory growth, post-exception usability, same-instance repeatability
Source: contract test (no published optimum); oracle = self-consistency
        (sequential single-threaded result is ground truth) + cvxpy CLARABEL
        for absolute correctness of the base problem.

Subtests
  (a) 50 solves in one process -> bit identical?
  (b) 5 separate processes     -> bit identical?
  (c) concurrent solve_qp from N Python threads on DIFFERENT problems
      -> match sequential answers bit-for-bit? (threads + ThreadPoolExecutor)
  (d) parallelism knob: solve_qp_batch (Rayon-parallel) and
      solve_qp_multi_rhs vs per-item sequential solve_qp -> same ANSWER
  (e) 2000 solves of a small QP -> RSS plateau or monotone climb?
  (f) exception mid-solve (invalid input) -> library still usable?
  (g) repeated solves of the SAME problem dict/instance -> same answer?

Run:
  source /Users/jkitchin/projects/pounce/.venv-qa/bin/activate
  python adversary/runs/2026-07-22_api_determinism_reentrancy.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pounce import solve_qp, solve_qp_batch, solve_qp_multi_rhs

# --------------------------------------------------------------------------
# problem builders (deterministic given a seed)
# --------------------------------------------------------------------------


def make_qp(seed: int, n: int = 12, m_eq: int = 3, m_in: int = 8):
    """A strictly convex, feasible, bounded random QP."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    P = M @ M.T + n * np.eye(n)  # SPD
    c = rng.standard_normal(n)
    A = rng.standard_normal((m_eq, n))
    x_feas = rng.standard_normal(n) * 0.1
    b = A @ x_feas
    G = rng.standard_normal((m_in, n))
    h = G @ x_feas + 1.0  # strictly feasible at x_feas
    lb = -5.0 * np.ones(n)
    ub = 5.0 * np.ones(n)
    return dict(P=P, c=c, A=A, b=b, G=G, h=h, lb=lb, ub=ub)


def make_small_qp(seed: int = 0, n: int = 6):
    return make_qp(seed, n=n, m_eq=1, m_in=3)


def fingerprint(r) -> str:
    """Bit-exact fingerprint of a QpResult's numeric payload."""
    hsh = hashlib.sha256()
    hsh.update(str(r.status).encode())
    for name in ("x", "y", "z", "z_lb", "z_ub"):
        v = getattr(r, name, None)
        if v is None:
            hsh.update(b"None")
        else:
            hsh.update(np.ascontiguousarray(np.asarray(v, dtype=np.float64)).tobytes())
    hsh.update(np.float64(r.obj).tobytes())
    hsh.update(str(int(getattr(r, "iters", -1))).encode())
    return hsh.hexdigest()


# --------------------------------------------------------------------------
# child-process mode for subtest (b)
# --------------------------------------------------------------------------

if len(sys.argv) > 1 and sys.argv[1] == "--child":
    prob = make_qp(1234)
    res = solve_qp(**prob)
    print(fingerprint(res))
    print(repr(float(res.obj)))
    sys.exit(0)


FAILURES: list[str] = []
NOTES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


print("=" * 74)
print("pounce adversary — determinism / reentrancy / resource contracts")
print("=" * 74)

# --------------------------------------------------------------------------
# absolute-correctness anchor: cvxpy on the base problem
# --------------------------------------------------------------------------
print("\n[0] absolute correctness anchor (cvxpy CLARABEL)")
base = make_qp(1234)
r_base = solve_qp(**base)
print(f"  pounce: status={r_base.status} obj={float(r_base.obj):.12e} "
      f"iters={r_base.iters}")
try:
    import cvxpy as cp

    xv = cp.Variable(base["P"].shape[0])
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(base["P"])) + base["c"] @ xv),
        [
            base["A"] @ xv == base["b"],
            base["G"] @ xv <= base["h"],
            xv >= base["lb"],
            xv <= base["ub"],
        ],
    )
    prob.solve(solver=cp.CLARABEL)
    obj_err = abs(float(r_base.obj) - prob.value) / max(1.0, abs(prob.value))
    x_err = float(np.max(np.abs(np.asarray(r_base.x) - xv.value)))
    print(f"  cvxpy : obj={prob.value:.12e}")
    print(f"  obj_rel_err={obj_err:.3e}  x_inf_err={x_err:.3e}")
    check(obj_err < 1e-7, "base QP matches cvxpy (objective)")
    check(x_err < 1e-6, "base QP matches cvxpy (solution)")
except ImportError:
    NOTES.append("cvxpy unavailable; correctness anchor skipped")
    print("  (cvxpy unavailable — skipped)")

# --------------------------------------------------------------------------
# (a) bit-reproducibility, 50 solves in one process
# --------------------------------------------------------------------------
print("\n[a] 50 in-process solves of the same problem")
fps = []
for _ in range(50):
    fps.append(fingerprint(solve_qp(**make_qp(1234))))
uniq = sorted(set(fps))
print(f"  distinct fingerprints: {len(uniq)}  ({uniq[0][:16]}...)")
check(len(uniq) == 1, "50 in-process solves are bit-identical")

# same, reusing the SAME numpy arrays (no rebuild) -> subtest (g) too
print("\n[g] 50 repeated solves of the SAME problem arrays (state carryover)")
same_fps = [fingerprint(solve_qp(**base)) for _ in range(50)]
print(f"  distinct fingerprints: {len(set(same_fps))}")
check(len(set(same_fps)) == 1, "repeated solves of same instance are bit-identical")
check(
    same_fps[0] == fps[0],
    "same-instance result equals fresh-arrays result (bit-identical)",
)

# --------------------------------------------------------------------------
# (b) bit-reproducibility across 5 separate processes
# --------------------------------------------------------------------------
print("\n[b] 5 separate processes")
child_fps = []
env = dict(os.environ)
for i in range(5):
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout.split()
    child_fps.append(out[0])
    print(f"  proc {i}: {out[0][:16]}...  obj={out[1]}")
check(len(set(child_fps)) == 1, "5 separate processes are bit-identical")
check(
    child_fps[0] == fps[0],
    "cross-process fingerprint equals in-process fingerprint",
)

# --------------------------------------------------------------------------
# (c) thread safety: concurrent solve_qp on DIFFERENT problems
# --------------------------------------------------------------------------
NPROB = 24
probs = [make_qp(1000 + i, n=10 + (i % 5)) for i in range(NPROB)]

print(f"\n[c] thread safety on {NPROB} DIFFERENT problems")
t0 = time.perf_counter()
seq = [fingerprint(solve_qp(**p)) for p in probs]
t_seq = time.perf_counter() - t0
seq_obj = [float(solve_qp(**p).obj) for p in probs]
print(f"  sequential baseline: {t_seq:.3f}s")

# raw threading.Thread, all launched at once
results: dict[int, str] = {}
errors: dict[int, str] = {}
barrier = threading.Barrier(NPROB)


def worker(i: int) -> None:
    try:
        barrier.wait(timeout=30)
        results[i] = fingerprint(solve_qp(**probs[i]))
    except Exception as exc:  # noqa: BLE001
        errors[i] = f"{type(exc).__name__}: {exc}"


t0 = time.perf_counter()
threads = [threading.Thread(target=worker, args=(i,)) for i in range(NPROB)]
for th in threads:
    th.start()
for th in threads:
    th.join()
t_thr = time.perf_counter() - t0
print(f"  threaded (barrier-synced): {t_thr:.3f}s  errors={len(errors)}")
if errors:
    for k, v in list(errors.items())[:3]:
        print(f"    thread {k}: {v}")
check(not errors, "no exceptions raised in concurrent threads")
mism = [i for i in range(NPROB) if results.get(i) != seq[i]]
check(not mism, f"threaded results bit-identical to sequential (mismatch={mism})")

# ThreadPoolExecutor, repeated rounds to shake out races
print("\n[c2] ThreadPoolExecutor, 6 rounds x problems (8 workers)")
pool_bad = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    for _round in range(6):
        got = list(ex.map(lambda p: fingerprint(solve_qp(**p)), probs))
        pool_bad += sum(1 for a, b in zip(got, seq) if a != b)
print(f"  mismatches across {6 * NPROB} pooled solves: {pool_bad}")
check(pool_bad == 0, "ThreadPoolExecutor results bit-identical to sequential")

# interleave a *different* problem repeatedly against a hot loop (shared state)
print("\n[c3] two threads hammering different problems, interleaved")
hot_bad = [0, 0]


def hammer(idx: int, slot: int) -> None:
    want = seq[idx]
    for _ in range(60):
        if fingerprint(solve_qp(**probs[idx])) != want:
            hot_bad[slot] += 1


ta = threading.Thread(target=hammer, args=(0, 0))
tb = threading.Thread(target=hammer, args=(NPROB - 1, 1))
ta.start(), tb.start()
ta.join(), tb.join()
print(f"  mismatches: {hot_bad}")
check(sum(hot_bad) == 0, "interleaved hot-loop threads stay bit-identical")

# --------------------------------------------------------------------------
# (d) parallelism must not change the ANSWER
# --------------------------------------------------------------------------
print("\n[d] parallel entry points vs sequential answers")
batch_problems = [
    dict(P=p["P"], c=p["c"], A=p["A"], b=p["b"], G=p["G"], h=p["h"],
         lb=p["lb"], ub=p["ub"])
    for p in probs
]
t0 = time.perf_counter()
batch = solve_qp_batch(batch_problems)
t_batch = time.perf_counter() - t0
print(f"  solve_qp_batch: {t_batch:.3f}s for {len(batch)} items "
      f"(sequential {t_seq:.3f}s)")
bad_obj = []
bad_bits = []
for i, (rb, want_obj) in enumerate(zip(batch, seq_obj)):
    if abs(float(rb.obj) - want_obj) > 1e-9 * max(1.0, abs(want_obj)):
        bad_obj.append((i, float(rb.obj), want_obj))
    if fingerprint(rb) != seq[i]:
        bad_bits.append(i)
check(not bad_obj, f"solve_qp_batch objectives match sequential (bad={bad_obj[:3]})")
if bad_bits:
    NOTES.append(
        f"solve_qp_batch differs bit-wise from solve_qp on {len(bad_bits)}/"
        f"{NPROB} items while objectives agree to 1e-9 — different code path, "
        "not a race (deterministic across repeats, checked below)"
    )
# determinism *of* the parallel path itself
batch2 = solve_qp_batch(batch_problems)
batch3 = solve_qp_batch(batch_problems)
same_batch = all(
    fingerprint(a) == fingerprint(b) == fingerprint(c)
    for a, b, c in zip(batch, batch2, batch3)
)
check(same_batch, "solve_qp_batch is bit-deterministic across repeated calls")

# multi-rhs path
base_s = make_small_qp(7, n=8)
rhs = np.stack([base_s["c"] + 0.1 * k for k in range(6)])
multi = solve_qp_multi_rhs(
    P=base_s["P"], cs=rhs, A=base_s["A"], b=base_s["b"],
    G=base_s["G"], h=base_s["h"], lb=base_s["lb"], ub=base_s["ub"],
)
mr_bad = []
for k, rm in enumerate(multi):
    single = solve_qp(
        P=base_s["P"], c=rhs[k], A=base_s["A"], b=base_s["b"],
        G=base_s["G"], h=base_s["h"], lb=base_s["lb"], ub=base_s["ub"],
    )
    if abs(float(rm.obj) - float(single.obj)) > 1e-9 * max(1.0, abs(float(single.obj))):
        mr_bad.append((k, float(rm.obj), float(single.obj)))
    elif float(np.max(np.abs(np.asarray(rm.x) - np.asarray(single.x)))) > 1e-8:
        mr_bad.append((k, "x", None))
check(not mr_bad, f"solve_qp_multi_rhs matches per-item solve_qp (bad={mr_bad[:3]})")

# --------------------------------------------------------------------------
# (e) memory: 2000 solves of a small QP
# --------------------------------------------------------------------------
print("\n[e] memory growth over 2000 solves of a small QP")


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / 1024 / 1024 if sys.platform == "darwin" else raw / 1024


small = make_small_qp(7)
for _ in range(200):  # warm-up: let allocators/caches settle
    solve_qp(**small)

samples = []
N_LOOP = 2000
t0 = time.perf_counter()
for i in range(N_LOOP):
    solve_qp(**small)
    if (i + 1) % 200 == 0:
        samples.append((i + 1, rss_mb()))
t_loop = time.perf_counter() - t0
for n, mb in samples:
    print(f"    after {n:5d} solves: RSS = {mb:8.2f} MB")
print(f"  {N_LOOP} solves in {t_loop:.2f}s ({1e3 * t_loop / N_LOOP:.2f} ms/solve)")

first_half = samples[len(samples) // 2 - 1][1]
last = samples[-1][1]
growth_second_half = last - first_half
total_growth = last - samples[0][1]
print(f"  growth over 2nd half of the loop: {growth_second_half:+.2f} MB")
print(f"  growth over whole loop:           {total_growth:+.2f} MB")
# a real leak keeps climbing; a plateau shows ~0 growth in the second half
check(
    growth_second_half < 2.0,
    f"RSS plateaus (2nd-half growth {growth_second_half:+.2f} MB < 2 MB)",
)
check(
    total_growth < 10.0,
    f"total RSS growth over 2000 solves is bounded ({total_growth:+.2f} MB)",
)

# --------------------------------------------------------------------------
# (f) exception mid-solve leaves the library usable
# --------------------------------------------------------------------------
print("\n[f] usability after exceptions from invalid input")
n = base["P"].shape[0]
bad_inputs = [
    ("shape mismatch A/b", dict(P=base["P"], c=base["c"],
                                A=base["A"], b=base["b"][:-1])),
    ("P wrong dim", dict(P=np.eye(n + 3), c=base["c"])),
    ("c wrong length", dict(P=base["P"], c=base["c"][:-2])),
    ("NaN in c", dict(P=base["P"], c=np.r_[np.nan, base["c"][1:]])),
    ("inf in P", dict(P=base["P"] + np.diag(np.r_[np.inf, np.zeros(n - 1)]),
                      c=base["c"])),
    ("non-square P", dict(P=np.ones((n, n + 1)), c=base["c"])),
    ("lb > ub", dict(P=base["P"], c=base["c"],
                     lb=np.ones(n), ub=-np.ones(n))),
    ("ragged G", dict(P=base["P"], c=base["c"],
                      G=np.ones((3, n)), h=np.ones(5))),
    ("negative max_iter", dict(P=base["P"], c=base["c"], max_iter=-5)),
    ("empty problem", dict(P=np.zeros((0, 0)), c=np.zeros(0))),
]
for label, kwargs in bad_inputs:
    try:
        out = solve_qp(**kwargs)
        outcome = f"returned status={out.status} (no raise)"
    except Exception as exc:  # noqa: BLE001
        outcome = f"raised {type(exc).__name__}"
    after = solve_qp(**base)
    ok = fingerprint(after) == fps[0]
    print(f"  {label:22s} -> {outcome:38s} recovery={'OK' if ok else 'CORRUPT'}")
    check(ok, f"library usable & bit-identical after '{label}'")

# also: exception from a worker thread must not poison other threads
print("\n[f2] exception in one thread does not corrupt a concurrent solver")
poison_bad = [0]


def poisoner() -> None:
    for _ in range(40):
        try:
            solve_qp(P=np.eye(4), c=np.ones(7))
        except Exception:  # noqa: BLE001, S110
            pass


def victim() -> None:
    for _ in range(40):
        if fingerprint(solve_qp(**base)) != fps[0]:
            poison_bad[0] += 1


tp = threading.Thread(target=poisoner)
tv = threading.Thread(target=victim)
tp.start(), tv.start()
tp.join(), tv.join()
print(f"  victim mismatches: {poison_bad[0]}")
check(poison_bad[0] == 0, "concurrent exceptions do not corrupt other threads")

# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
print("\n" + "=" * 74)
for note in NOTES:
    print(f"NOTE: {note}")
if FAILURES:
    print(f"{len(FAILURES)} failing contract(s):")
    for f in FAILURES:
        print(f"  - {f}")
    print("VERDICT: FAIL")
else:
    print("all determinism / reentrancy / resource contracts hold")
    print("VERDICT: PASS")
