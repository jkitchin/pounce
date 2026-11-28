"""gh#1271: an ``NlProblem`` evaluated from Python must never wait on the Rayon pool.

``NlProblem.hessian`` (and ``constraints`` / ``jacobian``) evaluate with the GIL
held. Above their size gates the tape walks run a ``par_iter`` on the global
Rayon pool, which is the same pool a callback batch (``solve_nlp_batch`` over
``pounce.Problem``) runs its instances on, and every callback waits for the GIL.
So a GIL holder waiting on the pool deadlocked against a pool worker waiting on
the GIL. The fix pins the Python-facing evaluators to the serial walk.

Each case runs in a subprocess because ``RAYON_NUM_THREADS`` is read once, when
the global pool starts, and because the failure mode is a hang: the timeout turns
it into a test failure instead of a stalled CI job.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("numpy")

TIMEOUT_S = 120

# Deterministic form (the issue's repro2.py): one pool thread, held inside a
# callback, while the main thread evaluates an unrelated tape whose Hessian is
# above the parallel gate (2 colors x 1024 rows).
CROSS_THREAD = textwrap.dedent(
    """
    import threading, time
    import numpy as np
    import pounce

    a, b = (pounce.NlExpr.var(i) for i in range(2))
    tape = pounce.build_nl_problem(2, a, constraints=[a * a * b] * 1024)
    ready = threading.Event()

    class Quadratic:  # never touches `tape`
        def objective(self, x):
            if not ready.is_set():
                ready.set()
                time.sleep(0.25)  # keep the pool's worker inside a callback
            return x[0] ** 2
        def gradient(self, x):
            return 2 * x
        def constraints(self, x):
            return np.empty(0)
        def jacobian(self, x):
            return np.empty(0)

    out = {}
    def solve_batch():
        problems = [pounce.Problem(n=1, m=0, problem_obj=Quadratic(), lb=[-2.0],
                                   ub=[2.0], cl=[], cu=[]) for _ in range(2)]
        res = pounce.solve_nlp_batch(problems, x0s=[[1.0], [1.0]],
                                     options={"print_level": 0})
        out["statuses"] = [info["status"] for _, info in res]

    worker = threading.Thread(target=solve_batch)
    worker.start()
    assert ready.wait(30)
    h = tape.hessian([1.0, 1.0], lam=np.ones(1024))
    g = tape.constraints([1.0, 1.0])
    j = tape.jacobian([1.0, 1.0])
    worker.join()
    assert list(h) == [2048.0, 2048.0], list(h)
    assert len(g) == 1024 and len(j) == 2048
    assert out["statuses"] == [0, 0], out
    print("OK")
    """
)

# Nested form (the issue's repro.py, fewer rounds): callbacks running on pool
# workers evaluate their own tape, whose Hessian is above the parallel gate.
NESTED = textwrap.dedent(
    """
    import numpy as np
    import pounce

    N, M, BATCH, ROUNDS = 8, 256, 16, 25

    def make_tape():
        x = [pounce.NlExpr.var(i) for i in range(N)]
        obj = x[0] * x[0]
        for i in range(1, N):
            obj = obj + (x[i] - 0.1 * i) * (x[i] - 0.1 * i)
        def cube_of_sum():
            s = x[0]
            for v in x[1:]:
                s = s + v
            return s * s * s
        return pounce.build_nl_problem(N, obj, constraints=[cube_of_sum() for _ in range(M)])

    class Callbacks:
        def __init__(self, t):
            self.t = t
        def objective(self, x):
            return self.t.objective(x)
        def gradient(self, x):
            return self.t.gradient(x)
        def constraints(self, x):
            return self.t.constraints(x)
        def jacobian(self, x):
            return self.t.jacobian(x)
        def jacobianstructure(self):
            return self.t.jacobian_structure()
        def hessianstructure(self):
            return self.t.hessian_structure()
        def hessian(self, x, lagrange, obj_factor):
            return self.t.hessian(x, lam=lagrange, obj_factor=obj_factor)

    tapes = [make_tape() for _ in range(BATCH)]
    rng = np.random.default_rng(0)
    for r in range(ROUNDS):
        problems = [pounce.Problem(n=N, m=M, problem_obj=Callbacks(t), lb=[-3.0] * N,
                                   ub=[3.0] * N, cl=[-1e20] * M, cu=[100.0] * M)
                    for t in tapes]
        x0s = [rng.uniform(-2.0, 2.0, size=N) for _ in tapes]
        res = pounce.solve_nlp_batch(problems, x0s=x0s, options={"print_level": 0})
        statuses = {info["status"] for _, info in res}
        assert statuses == {0}, (r, statuses)
    print("OK")
    """
)


def _run(script, **env):
    full_env = dict(os.environ)
    full_env.pop("POUNCE_NL_PARALLEL_EVAL", None)
    full_env.update(env)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=full_env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"deadlocked (no exit in {TIMEOUT_S} s) with env {env}")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().endswith("OK"), proc.stdout + proc.stderr


@pytest.mark.parametrize(
    "env",
    [
        {"RAYON_NUM_THREADS": "1"},
        # The override that forces the parallel walks must not reopen the
        # deadlock: the serial pin outranks it.
        {"RAYON_NUM_THREADS": "1", "POUNCE_NL_PARALLEL_EVAL": "1"},
    ],
    ids=["default-gate", "forced-parallel"],
)
def test_python_eval_while_callback_batch_holds_pool(env):
    _run(CROSS_THREAD, **env)


@pytest.mark.parametrize("threads", ["3", "4"])
def test_callback_batch_evaluating_parallel_sized_tapes(threads):
    _run(NESTED, RAYON_NUM_THREADS=threads)
