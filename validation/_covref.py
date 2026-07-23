"""Independent parameter-covariance oracle: PyNumero (AMPL/ASL derivatives)
+ scipy sparse linear algebra.

Reproduces the asymptotic least-squares covariance
    cov = 2 * sigma^2 * (K^{-1})_pp,   K = [[H, J^T],[J, 0]]
by a toolchain entirely separate from pounce's converged factorization:
Hessian of the Lagrangian H and constraint Jacobian J come from PyNumero's
AMPL Solver Library interface (the same evaluators IPOPT uses); the KKT
matrix is assembled and factored with scipy.sparse. Constraint multipliers
are taken from the model's dual suffix (a solution property).

This oracle matches pounce's covariance() to ~5e-16 on a linear regression
and ~1e-7 on a well-conditioned nonlinear fit (see p1_cho.py). It is exact
in the mu->0 / well-conditioned limit; on a severely ill-conditioned reduced
Hessian it loses precision because the only multipliers available to it are
solver-reported (finite accuracy) and get amplified by the conditioning.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP


def covariance_reference(model, fitted_vars, sigma_sq, dual_suffix):
    """Independent covariance of `fitted_vars` at the model's current
    (converged) point. `dual_suffix` is a populated Pyomo dual Suffix on
    `model`. Returns (cov 12x12, active_mask, reduced_hessian_eigs)."""
    nlp = PyomoNLP(model)
    xnames = [v.name for v in nlp.get_pyomo_variables()]
    name2i = {nm: i for i, nm in enumerate(xnames)}
    p_rows = [name2i[v.name] for v in fitted_vars]
    npar = len(p_rows)

    cons = nlp.get_pyomo_constraints()
    lam = np.array([dual_suffix[c] for c in cons])

    J = nlp.evaluate_jacobian().tocsc()
    gf = nlp.evaluate_grad_objective()
    # pounce/AMPL stationarity is gf - J^T lam = z (bound mult); the Lagrangian
    # Hessian is d2(f - lam^T c) = d2f - sum lam_i d2 c_i, so PyNumero (which
    # forms d2(f + duals^T c)) needs duals = -lam.
    nlp.set_duals(-lam)
    H = nlp.evaluate_hessian_lag().tocsc()

    x = nlp.get_primals()
    lo, hi = nlp.primals_lb(), nlp.primals_ub()
    tol = 1e-6 * (1.0 + np.abs(x))
    active_var = (x - lo < tol) | (hi - x < tol)
    active_par = np.array([active_var[r] for r in p_rows])

    # reduced KKT over the interior (off-bound) variables; active-bound
    # variables are held fixed (the mu->0 interior-point limit).
    freev = np.where(~active_var)[0]
    pos = {r: k for k, r in enumerate(freev)}
    Jf = J[:, freev]
    Hff = H[np.ix_(freev, freev)]
    mm = Jf.shape[0]
    K = sp.bmat([[Hff, Jf.transpose()],
                 [Jf, sp.csc_matrix((mm, mm))]]).tocsc()
    lu = spla.splu(K)

    fp = [i for i in range(npar) if not active_par[i]]
    ncol = K.shape[0]
    cols = {}
    for i in fp:
        e = np.zeros(ncol)
        e[pos[p_rows[i]]] = 1.0
        cols[i] = lu.solve(e)
    M = np.zeros((npar, npar))
    for i in fp:
        for j in fp:
            M[i, j] = cols[j][pos[p_rows[i]]]
    M = 0.5 * (M + M.T)

    cov = np.zeros((npar, npar))
    fpa = np.array(fp)
    if len(fp):
        cov[np.ix_(fpa, fpa)] = 2.0 * sigma_sq * M[np.ix_(fpa, fpa)]

    # reduced-Hessian eigenvalues on the free block (conditioning diagnostic)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            rh = np.linalg.inv(M[np.ix_(fpa, fpa)]) if len(fp) else np.zeros((0, 0))
            eigs = np.linalg.eigvalsh(rh) if rh.size else np.array([])
        except np.linalg.LinAlgError:
            eigs = np.array([])
    return cov, active_par, eigs
