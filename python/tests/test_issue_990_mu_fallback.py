"""gh#990 (remaining item h): the Python path for a problem without
``hessian`` pins ``mu_strategy=monotone`` (so gh#746's limited-memory ->
adaptive substitution does not fire on a Hessian choice POUNCE made itself).
That pin used to count as the caller naming a strategy, which switched off the
default-on ``mu_strategy_fallback`` stall retry (gh#748) on exactly this path,
contradicting docs/src/options.md. The pin is now POUNCE's choice: the retry
runs, unless the caller names ``mu_strategy`` or ``mu_strategy_fallback=no``.

Observable: a capped (``max_iter=5``) solve that triggers the retry runs the
model a second time, so it evaluates the objective about twice as often.
"""

import numpy as np

import pounce


class _CountedRosen:
    def __init__(self):
        self.n_obj = 0

    def objective(self, x):
        self.n_obj += 1
        return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    def gradient(self, x):
        g = np.zeros_like(x)
        g[:-1] += -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])
        g[1:] += 200 * (x[1:] - x[:-1] ** 2)
        return g

    def constraints(self, x):
        return np.array([x.sum()])

    def jacobianstructure(self):
        return np.zeros(10, int), np.arange(10)

    def jacobian(self, x):
        return np.ones(10)


def _evals(**opts):
    cb = _CountedRosen()
    p = pounce.Problem(n=10, m=1, problem_obj=cb, lb=[-5] * 10, ub=[5] * 10, cl=[-100.0], cu=[100.0])
    p.add_option("print_level", 0)
    p.add_option("max_iter", 5)
    for k, v in opts.items():
        p.add_option(k, v)
    _x, info = p.solve(x0=np.full(10, -1.2))
    assert info["status"] == -1  # Maximum_Iterations_Exceeded either way
    return cb.n_obj


def test_the_stall_retry_runs_on_the_no_hessian_path():
    default = _evals()
    named = _evals(mu_strategy="monotone")
    off = _evals(mu_strategy_fallback="no")
    assert named == off, (named, off)
    assert default > 1.5 * off, (
        f"the default-on mu_strategy_fallback did not retry: {default} objective "
        f"evaluations against {off} with the retry off"
    )
