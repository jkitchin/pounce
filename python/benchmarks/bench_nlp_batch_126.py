"""Benchmark parallel vs sequential batched NLP solving (issue #126).

`pounce.solve_nlp_batch(..., parallel=True)` runs N independent native
(`.nl`-loaded) instances on a Rayon pool, each worker with its own
IpoptApplication and an inner-serial FERAL factor (outer-parallel /
inner-serial — the same model as the QP batch). This script measures
the end-to-end wall time of the parallel path against the sequential
one (`parallel=False`, where each instance's factor may parallelize
internally) on a multi-start sweep of one model: K clones of the same
structure, each from a randomly perturbed starting point — the cheap
per-instance `NlProblem.variant(...)` path.

A correctness gate runs first: every instance must report
Solve_Succeeded under both modes, and the per-instance solutions must
agree across modes, so a timing win can't come from solving the wrong
thing (or nothing).

Expected shape of the result: speedup ~ min(K, cores), modulo
per-instance iteration-count imbalance (NLP instances converge in
ragged iteration counts; a finished instance frees its worker). On a
4-core container with K=24 gaslib11 instances this measured ~4.7x —
slightly super-linear because the sequential baseline pays the parallel
factor's coordination overhead on factors this small.

Run (from the repo root, with pounce installed):

    python python/benchmarks/bench_nlp_batch_126.py [path/to/model.nl]
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

import pounce

DEFAULT_NL = (
    Path(__file__).resolve().parents[2]
    / "crates" / "pounce-cli" / "tests" / "fixtures"
    / "aux_presolve" / "gaslib11_steady.nl"
)
K = 24
SEED = 0
X0_NOISE = 0.01


def main() -> int:
    nl = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NL
    base = pounce.read_nl(str(nl))
    print(f"model: {nl.name}  n={base.n} m={base.m} "
          f"nnz_jac={base.nnz_jac} nnz_hess={base.nnz_hess}")

    rng = np.random.default_rng(SEED)
    x0 = np.asarray(base.x0)
    batch = [
        base.variant(x0=x0 + rng.normal(0.0, X0_NOISE, base.n))
        for _ in range(K)
    ]

    t0 = time.perf_counter()
    seq = pounce.solve_nlp_batch(batch, parallel=False)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    par = pounce.solve_nlp_batch(batch, parallel=True)
    t_par = time.perf_counter() - t0

    # Correctness gate before any timing claim.
    bad = [
        i for i, ((_, si), (_, pi)) in enumerate(zip(seq, par))
        if si["status_msg"] != "Solve_Succeeded"
        or pi["status_msg"] != "Solve_Succeeded"
    ]
    if bad:
        print(f"FAIL: instances {bad} did not converge")
        return 1
    worst = max(
        float(np.max(np.abs(xs - xp)))
        for (xs, _), (xp, _) in zip(seq, par)
    )
    iters = [info["iter_count"] for _, info in par]

    print(f"instances: {K}   cores: {os.cpu_count()}")
    print(f"iters/instance: min={min(iters)} max={max(iters)} "
          f"(ragged is expected)")
    print(f"max |x_seq - x_par| over batch: {worst:.3e}")
    print(f"sequential: {t_seq:.3f} s   parallel: {t_par:.3f} s   "
          f"speedup: {t_seq / t_par:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
