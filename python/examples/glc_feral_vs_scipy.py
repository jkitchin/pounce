"""GLC bubble-column tritium-extraction BVP: pounce (FERAL) vs SciPy.

The model is the gas-liquid contactor BVP from pathsim-chem
(``src/pathsim_chem/tritium/glc.py``): a 4-state counter-current bubble
column for tritium extraction from Pb-Li. Its ``ode_system(xi, S)`` and
``boundary_conditions(Sa, Sb)`` signatures are *already* exactly what both
``scipy.integrate.solve_bvp`` and ``pounce.solve_bvp`` expect, so switching
solver is a one-line change.

We compute the dimensionless groups exactly as glc.py does, then solve the
same BVP two ways and compare speed and accuracy:

* SciPy (adaptive mesh refinement, as in the original).
* pounce ``method="newton"`` (FERAL unsymmetric sparse-LU Newton, fixed
  mesh) on (a) the initial mesh and (b) a mesh matched to SciPy's refined
  node count.

Standalone — only needs numpy, scipy, pounce (no pathsim).
"""

import time

import numpy as np
import scipy.constants as const
from scipy.integrate import solve_bvp as scipy_solve_bvp
from scipy.optimize import root_scalar

import pounce

g = const.g
R = const.R
N_A = const.N_A
M_LiPb = 2.875e-25


# --- verbatim from glc.py: properties + dimensionless groups ---------------

def _calculate_properties(params):
    T, D, flow_l, flow_g, P_in = (
        params["T"], params["D"], params["flow_l"], params["flow_g"], params["P_in"]
    )
    rho_l = 10.45e3 * (1 - 1.61e-4 * T)
    sigma_l = 0.52 - 0.11e-3 * T
    mu_l = 1.87e-4 * np.exp(11640 / (R * T))
    nu_l = mu_l / rho_l
    D_T = 2.5e-7 * np.exp(-27000 / (R * T))
    K_s_at = 2.32e-8 * np.exp(-1350 / (R * T))
    K_s = K_s_at * (rho_l / (M_LiPb * N_A))
    A = np.pi * (D / 2) ** 2
    Q_l = flow_l / rho_l
    Q_g = (flow_g * R * T) / P_in
    u_l = Q_l / A
    u_g0 = Q_g / A
    Bn = (g * D**2 * rho_l) / sigma_l
    Ga = (g * D**3) / nu_l**2
    Sc = nu_l / D_T
    Fr = u_g0 / (g * D) ** 0.5
    C = 0.2 * (Bn ** (1 / 8)) * (Ga ** (1 / 12)) * Fr
    sol = root_scalar(lambda e, c: e / (1 - e) ** 4 - c, args=(C,),
                      bracket=[1e-12, 1 - 1e-12])
    epsilon_g = sol.root
    epsilon_l = 1 - epsilon_g
    E_l = (D * u_g0) / ((13 * Fr) / (1 + 6.5 * (Fr**0.8)))
    E_g = (0.2 * D**2) * u_g0
    d_b = (26 * (Bn**-0.5) * (Ga**-0.12) * (Fr**-0.12)) * D
    a = 6 * epsilon_g / d_b
    h_l_a = D_T * (0.6 * Sc**0.5 * Bn**0.62 * Ga**0.31 * epsilon_g**1.1) / (D**2)
    h_l = h_l_a / a
    return dict(rho_l=rho_l, K_s=K_s, Q_l=Q_l, Q_g=Q_g, u_l=u_l, u_g0=u_g0,
                epsilon_g=epsilon_g, epsilon_l=epsilon_l, E_l=E_l, E_g=E_g, a=a, h_l=h_l)


def _calculate_dimensionless_groups(params, p):
    L, T, P_in, c_T_in = params["L"], params["T"], params["P_in"], params["c_T_in"]
    psi = (p["rho_l"] * g * p["epsilon_l"] * L) / P_in
    nu = ((c_T_in / p["K_s"]) ** 2) / P_in
    Bo_l = p["u_l"] * L / (p["epsilon_l"] * p["E_l"])
    phi_l = p["a"] * p["h_l"] * L / p["u_l"]
    Bo_g = p["u_g0"] * L / (p["epsilon_g"] * p["E_g"])
    phi_g = 0.5 * (R * T * c_T_in / P_in) * (p["a"] * p["h_l"] * L / p["u_g0"])
    return dict(Bo_l=Bo_l, phi_l=phi_l, Bo_g=Bo_g, phi_g=phi_g, psi=psi, nu=nu)


def make_bvp(dim, y_T2_in, BCs):
    """Return (ode_system, boundary_conditions) — verbatim glc.py logic."""
    Bo_l, phi_l, Bo_g, phi_g, psi, nu = (
        dim["Bo_l"], dim["phi_l"], dim["Bo_g"], dim["phi_g"], dim["psi"], dim["nu"]
    )

    def ode_system(xi, S):
        x_T, dx_T_dxi, y_T2, dy_T2_dxi = S
        theta = x_T - np.sqrt(np.maximum(0, (1 - psi * xi) * y_T2 / nu))
        dS0 = dx_T_dxi
        dS1 = Bo_l * (phi_l * theta - dx_T_dxi)
        dS2 = dy_T2_dxi
        term1 = (1 + 2 * psi / Bo_g) * dy_T2_dxi
        term2 = phi_g * theta
        dS3 = (Bo_g / (1 - psi * xi)) * (term1 - term2)
        return np.vstack((dS0, dS1, dS2, dS3))

    def boundary_conditions(Sa, Sb):
        if BCs == "C-C":
            return np.array([
                Sa[1],
                Sb[0] - (1 - (1 / Bo_l) * Sb[1]),
                Sa[2] - y_T2_in - (1 / Bo_g) * Sa[3],
                Sb[3],
            ])
        return np.array([Sa[1], Sb[0] - 1.0, Sa[2] - y_T2_in, Sb[3]])

    return ode_system, boundary_conditions


def _time(fn, repeats=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def main():
    # Physically reasonable Pb-Li GLC parameters.
    params = dict(
        T=600.0, D=0.1, L=2.0, P_in=3.0e5,
        flow_l=1.0, flow_g=0.02, c_T_in=1.0, y_T2_in=1e-20,
        BCs="C-C", elements=20,
    )
    phys = _calculate_properties(params)
    dim = _calculate_dimensionless_groups(params, phys)
    print("Dimensionless groups:", {k: round(v, 4) for k, v in dim.items()})

    ode, bc = make_bvp(dim, max(params["y_T2_in"], 1e-20), params["BCs"])
    m0 = params["elements"] + 1
    xi0 = np.linspace(0, 1, m0)
    y0 = np.zeros((4, m0))

    # --- SciPy (adaptive) ---
    ref = scipy_solve_bvp(ode, bc, xi0, y0, tol=1e-5, max_nodes=10000)
    m_scipy = ref.x.size
    t_scipy = _time(lambda: scipy_solve_bvp(ode, bc, xi0, y0, tol=1e-5, max_nodes=10000))
    print(f"\nSciPy:  success={ref.success}  nodes {m0} -> {m_scipy} (adaptive)  "
          f"{t_scipy*1e3:.2f} ms")

    # --- pounce FERAL Newton, fixed mesh matched to SciPy's refined node count ---
    xi_f = np.linspace(0, 1, m_scipy)
    yf = np.zeros((4, m_scipy))
    rp = pounce.solve_bvp(ode, bc, xi_f, yf, tol=1e-5, method="newton")
    t_pounce = _time(lambda: pounce.solve_bvp(ode, bc, xi_f, yf, tol=1e-5, method="newton"))
    print(f"pounce: success={rp.success}  fixed mesh {m_scipy} nodes  niter={rp.niter}  "
          f"{t_pounce*1e3:.2f} ms   ({t_scipy/t_pounce:.2f}x scipy)")

    # --- accuracy: compare the two solutions on a common grid ---
    xq = np.linspace(0, 1, 401)
    Ys = ref.sol(xq)
    Yp = rp.sol(xq)
    state_names = ["x_T", "dx_T/dxi", "y_T2", "dy_T2/dxi"]
    print("\nMax |pounce - scipy| per state (over xi in [0,1]):")
    for i, nm in enumerate(state_names):
        scale = max(1.0, np.max(np.abs(Ys[i])))
        print(f"  {nm:12s}: abs {np.max(np.abs(Yp[i]-Ys[i])):.2e}   "
              f"rel {np.max(np.abs(Yp[i]-Ys[i]))/scale:.2e}")

    # Headline physical output: extraction efficiency = 1 - x_T(outlet=xi 0).
    eff_scipy = 1 - ref.sol(0.0)[0]
    eff_pounce = 1 - rp.sol(0.0)[0]
    print(f"\nExtraction efficiency:  scipy {eff_scipy:.6f}   pounce {eff_pounce:.6f}")

    # --- speed scan on common fixed meshes (both solvers, same nodes) ---
    print("\nFixed-mesh speed scan (both on the same mesh):")
    print(f"{'nodes':>7} | {'scipy ms':>9} {'pounce ms':>10} {'ratio':>7} {'max|Δ|':>9}")
    for m in (21, 51, 101, 201, 401):
        xm = np.linspace(0, 1, m)
        ym = np.zeros((4, m))
        ts = _time(lambda: scipy_solve_bvp(ode, bc, xm, ym, tol=1e-5, max_nodes=m))
        rpm = pounce.solve_bvp(ode, bc, xm, ym, tol=1e-5, method="newton")
        tp = _time(lambda: pounce.solve_bvp(ode, bc, xm, ym, tol=1e-5, method="newton"))
        d = np.max(np.abs(rpm.sol(xq) - (scipy_solve_bvp(ode, bc, xm, ym, tol=1e-5,
                                                         max_nodes=m).sol(xq))))
        print(f"{m:>7} | {ts*1e3:>9.2f} {tp*1e3:>10.2f} {ts/tp:>6.2f}x {d:>9.1e}")


if __name__ == "__main__":
    main()
