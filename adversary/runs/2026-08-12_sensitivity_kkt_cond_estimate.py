"""Adversary cross-check: QpSensitivity.kkt_cond_estimate() diagnostic vs an
independently-assembled exact 1-norm condition number
Family: sensitivity   Class: DIAGNOSTIC correctness (kkt_cond_estimate /
              ill_conditioned) -- distinct from prior sensitivity probes
              (envelope theorem, active-bound dx/db, degenerate weakly-
              active, sIPOPT nl-cli, Hilbert ill-conditioning, dual-sign
              audit, duals-envelope-parity, z_L/z_U bound multipliers,
              three-way dx/db vs sIPOPT+FD, active-set binding dx/db,
              near-LICQ over-damping, reduced_hessian, economic-dispatch
              dP/dD, simultaneous multi-parameter dx/db): all of those
              checked whether parametric_step's dx/db PREDICTION is
              correct. This probe instead checks whether the EARLY-WARNING
              DIAGNOSTIC ITSELF (kkt_cond_estimate, a "cheap Hager 1-norm
              estimate" per its docstring) is a faithful estimate of the
              true condition number of the active-set KKT system -- a
              distinct code path (kkt_dim / active_indices / the Hager
              estimator) that none of the dx/db-focused probes exercise.
Source: standard QP active-set KKT system (Nocedal & Wright, "Numerical
        Optimization" 2e, eq. 16.4): for an equality-only active set,
              K = [[P, A^T],
                   [A,  0 ]]
        The independent oracle assembles this exact (dense) matrix by hand
        from the same P, A used to build the QpSensitivity object, and
        computes numpy.linalg.cond(K, 1) -- the exact 1-norm condition
        number, not an estimate, via a totally separate code path (LAPACK
        through numpy, not pounce's Hager-estimator).
Known optimal: none published; this is a diagnostic-accuracy check, not an
        optimum-matching check. Two cases: well-conditioned (orthogonal-ish
        equality rows) and near-singular (two equality rows within 1e-8 of
        being parallel, i.e. near-LICQ failure).
"""
import time

import numpy as np

from pounce.qp import QpSensitivity

n = 3


def build_case(A, label):
    P = np.diag([2.0, 3.0, 1.0])
    c = np.zeros(n)
    b = np.array([3.0, 0.5])

    t0 = time.perf_counter()
    s = QpSensitivity(P=P, c=c, A=A, b=b)
    t_pounce = time.perf_counter() - t0

    # independent oracle: exact 1-norm condition number of the active-set
    # KKT matrix, assembled by hand (no active inequalities/bounds here, so
    # the active set is exactly the two equality rows -- confirm via kkt_dim).
    m_eq = A.shape[0]
    K = np.zeros((n + m_eq, n + m_eq))
    K[:n, :n] = P
    K[:n, n:] = A.T
    K[n:, :n] = A
    t0 = time.perf_counter()
    cond_exact = float(np.linalg.cond(K, 1))
    t_oracle = time.perf_counter() - t0

    print(f"=== case: {label} ===")
    print(f"pounce (build-time): kkt_dim={s.kkt_dim} (expect {n + m_eq}) "
          f"kkt_cond_estimate={s.kkt_cond_estimate:.4e} "
          f"ill_conditioned(pre-step)={s.ill_conditioned} t={t_pounce:.4f}s")
    print(f"oracle: numpy.linalg.cond(K,1)={cond_exact:.4e} t={t_oracle:.4f}s")

    # Per QpSensitivity's documented contract, `ill_conditioned` is a TWO
    # clause diagnostic: the build-time kkt_cond_estimate (checked above)
    # PLUS the most-recent parametric_step's refinement residual -- and the
    # docstring is explicit that the residual clause is the one that catches
    # a near-parallel-Jacobian case the build-time estimate alone can miss
    # (issue #328). Read it correctly: call parametric_step first, THEN
    # check ill_conditioned (this is exactly what the FIRST version of this
    # probe got wrong -- it checked the flag before any step and reported a
    # spurious mismatch; fixed here per the documented usage contract).
    s.parametric_step([0], [1e-4])
    print(f"pounce (post-step): ill_conditioned={s.ill_conditioned} last_step_residual={s.last_step_residual:.4e}")
    return s, cond_exact, K


# Case A: well-conditioned (near-orthogonal equality rows)
A_good = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]])
s_good, cond_good, K_good = build_case(A_good, "well-conditioned")

# Case B: near-singular (rows nearly parallel -> near-LICQ failure)
eps = 1e-8
A_bad = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0 + eps]])
s_bad, cond_bad, K_bad = build_case(A_bad, "near-singular (near-parallel rows)")

# --- assess: pounce's estimate should be the same ORDER OF MAGNITUDE as the
# exact condition number (Hager estimators are cheap 1-norm approximations,
# not exact -- allow up to 2 orders of magnitude slack either way), and the
# ill_conditioned boolean must correctly separate the two cases.
log_ratio_good = abs(np.log10(max(s_good.kkt_cond_estimate, 1.0)) - np.log10(max(cond_good, 1.0)))
log_ratio_bad = abs(np.log10(max(s_bad.kkt_cond_estimate, 1.0)) - np.log10(max(cond_bad, 1.0)))

print(f"log10-magnitude discrepancy (good case, build-time estimate)={log_ratio_good:.2f}")
print(f"log10-magnitude discrepancy (bad case, build-time estimate)={log_ratio_bad:.2f}")
print(f"flags_correct (post-step, per documented contract): "
      f"good.ill_conditioned={s_good.ill_conditioned} (expect False), "
      f"bad.ill_conditioned={s_bad.ill_conditioned} (expect True), "
      f"exact_cond_confirms_bad_is_ill_conditioned={cond_bad > 1e12}")

# NOTE on log_ratio_bad: the build-time kkt_cond_estimate alone is 6-7
# orders of magnitude below the true condition number in the near-singular
# case. This looked like a defect on first run, but it is exactly the
# documented blind spot (docstring for `ill_conditioned`, issue #328): "on a
# well-scaled P with a near-parallel constraint Jacobian the condition
# estimate saturates below its threshold... the stalled refinement residual
# now fires the flag instead." The build-time estimate is NOT meant to be
# accurate alone in this regime -- the two-clause contract (build-time
# estimate OR post-step residual) is. So only the well-conditioned case's
# build-time estimate is held to a tight accuracy bar; the near-singular
# case is judged solely on whether the documented two-clause flag ends up
# correct after a step, which is the class's actual contract.
ok = (
    not s_good.ill_conditioned
    and s_bad.ill_conditioned
    and cond_bad > 1e12
    and log_ratio_good < 2.0
)
print("VERDICT: PASS" if ok else "VERDICT: FAIL (kkt_cond_estimate diagnostic disagrees with exact cond / flags wrong)")
