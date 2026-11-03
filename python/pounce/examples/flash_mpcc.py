"""Phase-changing flash as an MPCC: the Gate 1 model, and its oracle.

A single equilibrium stage whose temperature path crosses single-liquid,
two-phase and single-vapor, formulated as a mathematical program with
complementarity constraints and checked against an independent flash.
This is the model behind notebook 38 and behind the evidence harness in
`benchmarks/flash/`; it deliberately stops short of being a property
package, exactly as `phase_envelope` does.

Why a flash is an MPCC at all
-----------------------------

Phase appearance is awkward in an equation-oriented model. A formulation
that only instantiates equations for the phases that are present changes
dimension at a switching event; explicit on/off variables make it a
mixed-integer problem; smoothing or clipping converges to false phase
states. Complementarity says "the phase amount is zero **or** its
stability condition is active" while keeping the dimension fixed.

That convenience has a mathematical price: an MPCC violates the standard
constraint qualifications at every feasible point, so ordinary NLP
convergence theory cannot be quoted about it without care. This module
is the smallest honest instance of the trade.

The model
---------

For ``nc`` components at fixed ``(T, P)`` with feed ``z``, the unknowns
are the vapor fraction ``beta`` and the two phases' mole numbers per
unit feed, ``x`` (liquid) and ``y`` (vapor)::

    n = 2 nc + 1        variables   [beta, x_1..x_nc, y_1..y_nc]

    z_i = (1 - beta) x_i + beta y_i                           (balance)
    ln x_i + ln phi_i^L(x/Sx)
        = ln y_i + ln phi_i^V(y/Sy)                       (isofugacity)

    0 <= beta      _|_  1 - Sy >= 0                            (pair V)
    0 <= 1 - beta  _|_  1 - Sx >= 0                            (pair L)

with ``Sx = sum_i x_i`` and ``Sy = sum_i y_i``.

Why *these* pairs, and not ``L _|_ V``
--------------------------------------

The pair is "nonnegative phase amount complementary to nonnegative
stability slack", and **not** liquid flow complementary to vapor flow:
the two phases coexist on a two-phase tray, so ``L _|_ V`` would encode
the wrong physics. What is written above is that rule with the slack
identified:

* ``beta`` is the vapor amount and ``1 - Sy`` is the vapor's stability
  slack. Where no vapor is present the vapor variables are not junk --
  they are Michelsen's *trial phase*. Setting ``beta = 0`` makes the
  balance give ``x = z``, hence ``Sx = 1``, and the isofugacity rows
  collapse to ``ln y_i + ln phi_i(y/Sy) = ln z_i + ln phi_i(z)``, which
  is exactly the stationarity condition of the tangent-plane distance.
  ``Sy <= 1`` is then exactly ``TPD >= 0``. The complementarity is not
  an encoding trick: it *is* the stability test.
* ``1 - beta`` is the liquid amount and ``1 - Sx`` its slack, by the
  mirror argument at ``beta = 1``.
* In the two-phase regime both amounts are positive, so both slacks
  vanish, ``Sx = Sy = 1``, and the rows are the ordinary isofugacity
  conditions.

At the bubble point ``beta = 0`` **and** ``Sy = 1`` together: pair V is
biactive. At the dew point pair L is. The two regime switches are
therefore exactly the two biactive points of this MPCC, which is what
makes it a phase-change test rather than a decorated NLP -- the
degeneracy is the physics, not an artifact of writing it this way.

Where the normalization goes
----------------------------

``phi_i`` is a function of a *composition*, so it is evaluated at
``x/Sx`` and ``y/Sy``. The logarithm outside it is **not** normalized:
the row is ``ln x_i + ln phi_i(x/Sx)``, not ``ln(x_i/Sx) + ln
phi_i(x/Sx)``. That asymmetry is Michelsen's tangent plane, not a typo,
and normalizing the log term as well adds ``ln(Sy/Sx)`` to every row.

It is worth stating at length because the first implementation of this
model got it wrong in exactly that way, and the shape of the error is
the shape the cross-check exists to catch: ``ln(Sy/Sx)`` **vanishes
identically in the two-phase region**, where ``Sx = Sy = 1``, so the
model solved, converged, and agreed with the oracle at every two-phase
temperature. It was wrong only in the single-phase regimes -- the ones
the fixture exists to reach.

Layout
------

``ln_phi`` ... ``supercritical_components``
    the Peng--Robinson layer over `phase_envelope`, and the
    cubic-root / trivial-solution / supercritical guards.
`FlashCase`, `PAIRS`, `GATE1_FLASH`
    the source model, its complementarity pairs and what each branch
    means, and the pinned case.
`tangent_plane`, `flash`, `bubble_and_dew`
    the independent oracle: Michelsen stability with a multistart per
    phase label, Rachford--Rice with a Newton polish, and the
    switch-point bisection.
`lower`, `LoweredFlash`, `fd_check`
    MPCC -> smooth NLP by hand (product rows, Scholtes relaxation) with
    exact JAX derivatives.
`declare`, `DeclaredFlash`
    the same MPCC with its pairs *declared* to POUNCE, which picks the
    formulation, runs the finish and certifies the point.

All temperatures are in K, pressures in Pa, and phase amounts are moles
per mole of feed unless a docstring says otherwise.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, root

from pounce.examples.phase_envelope import (
    PengRobinsonMixture,
    compressibility,
    log_fugacity_coefficients,
)

jax.config.update("jax_enable_x64", True)


# ====================================================================
# Peng-Robinson layer, and the guards
# ====================================================================
#
# Everything here is a thin layer over
# `phase_envelope.log_fugacity_coefficients`, which landed with gh#777
# and is validated there against the published Deiters--Bell
# methane/propane envelope. **Reusing it rather than reimplementing it
# is deliberate**: this model and the oracle below are supposed to be
# independent calculations of the same flash, and the honest way to say
# that is to name the one primitive they share and check everything
# above it. A second copy of the cubic would make them look more
# independent than they are while adding a place for the two to
# disagree for a reason that has nothing to do with the MPCC.
#
# Three of the guards gh#776 asks for are computed here, as
# measurements rather than assumptions -- see `root_diagnostics`,
# `is_trivial` and `supercritical_components`.

#: Below this, ``max_i |ln K_i|`` is the trivial solution rather than a
#: narrow two-phase region. It is loose on purpose: a genuine flash at
#: the pinned condition runs ``|ln K|`` of order 1, and the trivial
#: attractor arrives at 1e-8 or below, so anything in between is a
#: measurement that should be looked at rather than classified.
TRIVIAL_LN_K = 1e-4

#: Two cubic roots closer than this are one root numerically, and the
#: root-selection question does not arise.
ONE_ROOT_GAP = 1e-9


def ln_phi(composition, temperature_k, pressure_pa, mixture, *, largest: bool):
    """``ln phi_i`` for a *normalized* composition. [-]"""
    return log_fugacity_coefficients(
        composition, temperature_k, pressure_pa, mixture, largest=largest
    )


def reduced_gibbs(composition, temperature_k, pressure_pa, mixture, *, largest: bool) -> float:
    """``sum_i w_i (ln w_i + ln phi_i)``, the reduced molar Gibbs energy. [-]

    Only differences of this at fixed composition are used, so the
    ``w ln w`` term cancels and could be dropped; it is kept because a
    quantity named "Gibbs" that is missing its entropy term is a trap
    for whoever reads it next.
    """
    w = np.asarray(composition, dtype=float)
    lnp = np.asarray(ln_phi(w, temperature_k, pressure_pa, mixture, largest=largest))
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = np.where(w > 0.0, w * np.log(np.where(w > 0.0, w, 1.0)), 0.0)
    return float(np.sum(entropy + w * lnp))


def root_diagnostics(
    composition, temperature_k, pressure_pa, mixture, *, largest: bool
) -> Dict[str, object]:
    """Did the label-selected cubic root agree with the lower-Gibbs one?

    Returns ``z_selected``, ``z_other``, the signed reduced-Gibbs
    difference ``selected - other``, and ``root_is_gibbs_optimal``
    (``None`` where the cubic has a single real root).
    """
    w = np.asarray(composition, dtype=float)
    z_hi = float(compressibility(w, temperature_k, pressure_pa, mixture, largest=True))
    z_lo = float(compressibility(w, temperature_k, pressure_pa, mixture, largest=False))
    z_sel, z_oth = (z_hi, z_lo) if largest else (z_lo, z_hi)
    if abs(z_hi - z_lo) <= ONE_ROOT_GAP:
        return {
            "z_selected": z_sel,
            "z_other": z_oth,
            "one_real_root": True,
            "gibbs_gap": 0.0,
            "root_is_gibbs_optimal": None,
        }
    g_sel = reduced_gibbs(w, temperature_k, pressure_pa, mixture, largest=largest)
    g_oth = reduced_gibbs(w, temperature_k, pressure_pa, mixture, largest=not largest)
    return {
        "z_selected": z_sel,
        "z_other": z_oth,
        "one_real_root": False,
        "gibbs_gap": float(g_sel - g_oth),
        "root_is_gibbs_optimal": bool(g_sel <= g_oth + 1e-12),
    }


def wilson_k(temperature_k, pressure_pa, mixture: PengRobinsonMixture) -> np.ndarray:
    """Wilson correlation ``K_i``. [-] The standard cold start."""
    return np.asarray(
        mixture.critical_pressure
        / pressure_pa
        * np.exp(
            5.373
            * (1.0 + mixture.acentric_factor)
            * (1.0 - mixture.critical_temperature / temperature_k)
        ),
        dtype=float,
    )


def is_trivial(k_values, tol: float = TRIVIAL_LN_K) -> bool:
    """``K_i = 1`` for every component: the classic false flash answer."""
    k = np.asarray(k_values, dtype=float)
    return bool(np.max(np.abs(np.log(k))) < tol)


def reduced_temperatures(temperature_k, mixture: PengRobinsonMixture) -> Dict[str, float]:
    """``T/Tc_i`` per pure component, for the supercritical record."""
    return {
        name: float(temperature_k / tc)
        for name, tc in zip(mixture.names, mixture.critical_temperature)
    }


def supercritical_components(temperature_k, mixture: PengRobinsonMixture) -> Optional[list]:
    """Which pure components are above their own ``Tc`` at this ``T``."""
    return [
        name
        for name, tc in zip(mixture.names, mixture.critical_temperature)
        if temperature_k > tc
    ]

# ====================================================================
# The source model
# ====================================================================
#
# Nothing below this point that belongs to the *model* reads a solver
# status, an NLP residual, or a lowering. That split is load-bearing:
# an MPCC lowering's NLP residuals are residuals of a different
# problem, and a converged NLP is not a solved flash.

#: Solver tolerance every route in this harness is pinned to, and the
#: corner tolerance derived from it. Both carry the meaning gh#794's
#: report established: `G*H` is quadratically flat at the corner, so a
#: solve converged to `tol` pins each side only to `sqrt(tol)`, and a
#: membership threshold at solver tolerance misreads a converged MPCC
#: point. `sqrt(tol)` is the "complementarity-accuracy floor" the Gate 0
#: report names, and it is numerical resolution, not phase physics.

#: Solver tolerance every route in this harness is pinned to, and the
#: corner tolerance derived from it. Both carry the meaning gh#794's
#: report established: `G*H` is quadratically flat at the corner, so a
#: solve converged to `tol` pins each side only to `sqrt(tol)`, and a
#: membership threshold at solver tolerance misreads a converged MPCC
#: point. `sqrt(tol)` is the "complementarity-accuracy floor" the Gate 0
#: report names, and it is numerical resolution, not phase physics.
SOLVE_TOL = 1e-8
CORNER_TOL = SOLVE_TOL**0.5

#: Floor on the membership tolerance, as in `mpcc/spec.py`.
ACTIVE_TOL = 1e-6

#: Lower bound on a phase mole number, and the reason it is not zero:
#: the isofugacity rows carry ``ln x_i`` and ``ln y_i``. See
#: `FlashCase.lb`.
MOLE_FLOOR = 1e-12

#: Regime labels. `two_phase` is the only one with both phases present;
#: `bubble` and `dew` are the biactive switch points, which are named
#: rather than folded into a neighbour because the whole fixture is
#: about them.
REGIMES = ("liquid", "bubble", "two_phase", "dew", "vapor", "undetermined")


#: The pinned mixture. Constants are the ethane and n-butane rows of
#: `phase_envelope.NATURAL_GAS`, which carries them for notebook 34;
#: taking them from the existing in-repo table rather than retyping a
#: handbook keeps one source of truth for the numbers. `k_ij = 0` is the
#: classical one-fluid choice that table already documents, and it is a
#: modelling choice, not a measurement: nothing here is a claim about
#: the real ethane/n-butane system beyond what Peng--Robinson with zero
#: binary interaction gives.
ETHANE_N_BUTANE = PengRobinsonMixture(
    names=("ethane", "n-butane"),
    critical_temperature=np.array([305.3, 425.1]),  # [K]
    critical_pressure=np.array([48.72, 37.96]) * 1e5,  # [Pa]
    acentric_factor=np.array([0.100, 0.200]),  # [-]
    composition=np.array([0.50, 0.50]),  # [-] feed
    binary_interaction=np.zeros((2, 2)),  # [-]
    source=(
        "Ethane and n-butane rows of phase_envelope.NATURAL_GAS "
        "(notebook 34 natural-gas demonstration); k_ij = 0."
    ),
)


@dataclasses.dataclass(frozen=True)
class Pair:
    """One complementarity pair ``0 <= G _|_ H >= 0``, with its physics."""

    name: str
    #: Which branch means what, in the model's own words. Read by no
    #: code; read by every reviewer, which is the point.
    branch_G_zero: str
    branch_H_zero: str
    units: str = "mol per mol feed"


PAIRS: Tuple[Pair, ...] = (
    Pair(
        name="vapor",
        branch_G_zero="beta = 0: no vapor present; y is Michelsen's trial phase",
        branch_H_zero="Sy = 1: the vapor is a real phase in equilibrium",
    ),
    Pair(
        name="liquid",
        branch_G_zero="beta = 1: no liquid present; x is Michelsen's trial phase",
        branch_H_zero="Sx = 1: the liquid is a real phase in equilibrium",
    ),
)


@dataclasses.dataclass(frozen=True)
class FlashCase:
    """One flash: a mixture, a feed, a pressure, and the temperature path."""

    name: str
    mixture: PengRobinsonMixture
    pressure_pa: float
    #: Temperature path, low to high. The harness walks it in both
    #: directions; `path.py` owns that, not this.
    temperatures_k: np.ndarray
    provenance: str = ""

    @property
    def nc(self) -> int:
        return self.mixture.n_components

    @property
    def n(self) -> int:
        return 2 * self.nc + 1

    @property
    def z(self) -> np.ndarray:
        return np.asarray(self.mixture.composition, dtype=float)

    # -- unpacking -------------------------------------------------

    def unpack(self, v) -> Tuple[float, np.ndarray, np.ndarray]:
        """``[beta, x, y]`` -> ``(beta, x, y)``."""
        v = np.asarray(v, dtype=float)
        nc = self.nc
        return float(v[0]), v[1 : 1 + nc], v[1 + nc : 1 + 2 * nc]

    def pack(self, beta: float, x, y) -> np.ndarray:
        return np.concatenate([[float(beta)], np.asarray(x, float), np.asarray(y, float)])

    @property
    def variable_names(self) -> List[str]:
        return (
            ["beta"]
            + [f"x_{c}" for c in self.mixture.names]
            + [f"y_{c}" for c in self.mixture.names]
        )

    # -- bounds ----------------------------------------------------

    @property
    def lb(self) -> np.ndarray:
        """``beta in [0,1]``; phase mole numbers nonnegative.

        The lower bound on ``x`` and ``y`` is a small positive number,
        not zero, and that is a modelling decision worth stating: the
        isofugacity rows carry ``ln x_i`` and ``ln y_i``, which are not
        defined at zero, and a component's amount in a *present* phase
        is bounded away from zero at any finite ``K_i``. It bounds what
        the fixture is evidence about -- it says nothing about a
        vanishing *component*, only about a vanishing *phase*, which is
        the transition gh#776 asks for.
        """
        return np.concatenate([[0.0], np.full(2 * self.nc, MOLE_FLOOR)])

    @property
    def ub(self) -> np.ndarray:
        return np.concatenate([[1.0], np.full(2 * self.nc, 1.0 / MOLE_FLOOR)])

    # -- the pairs, evaluated --------------------------------------

    def pair_values(self, v) -> Tuple[np.ndarray, np.ndarray]:
        """``(G, H)`` for the two pairs, in path order ``(vapor, liquid)``."""
        beta, x, y = self.unpack(v)
        g = np.array([beta, 1.0 - beta])
        h = np.array([1.0 - float(np.sum(y)), 1.0 - float(np.sum(x))])
        return g, h

    def pair_activity(self, v) -> List[Tuple[bool, bool]]:
        """Which side of each pair counts as zero, on the `sqrt(tol)` rule.

        The threshold is ``max(ACTIVE_TOL, CORNER_TOL)`` rather than
        ``ACTIVE_TOL``, for the reason `mpcc/spec.pair_activity`
        measured: a converged solve is entitled to leave the pair
        ``sqrt(tol)`` from the corner, and a fixed ``1e-6`` reads such a
        point as lying on neither branch. Every quantity here is a mole
        fraction or a phase fraction, so all of them live at O(1) and
        the relative-scale term that function needs does not arise.
        """
        g, h = self.pair_values(v)
        thresh = max(ACTIVE_TOL, CORNER_TOL)
        return [
            (bool(abs(gi) <= thresh), bool(abs(hi) <= thresh)) for gi, hi in zip(g, h)
        ]

    def regime(self, v) -> str:
        """The phase regime at ``v``, from the pair activity alone.

        Deliberately *not* from ``beta`` alone: ``beta = 0`` with the
        vapor slack also at zero is the bubble point, which is a
        different state from a subcooled liquid and is the state the
        fixture exists to reach.
        """
        (beta_zero, sy_one), (beta_one, sx_one) = self.pair_activity(v)
        if beta_zero and sy_one:
            return "bubble"
        if beta_one and sx_one:
            return "dew"
        if beta_zero:
            return "liquid"
        if beta_one:
            return "vapor"
        if sx_one and sy_one:
            return "two_phase"
        return "undetermined"

    # -- source residuals ------------------------------------------

    def balance_residual(self, v) -> np.ndarray:
        """``(1-beta) x_i + beta y_i - z_i``. [mol per mol feed]"""
        beta, x, y = self.unpack(v)
        return (1.0 - beta) * x + beta * y - self.z

    def isofugacity_residual(self, v, temperature_k: float) -> np.ndarray:
        """``ln x_i + ln phi_i^L(x/Sx) - ln y_i - ln phi_i^V(y/Sy)``. [-]

        Dimensionless by construction, so the tolerance it is judged
        against is a pure number and does not move with the pressure.
        """
        beta, x, y = self.unpack(v)
        sx, sy = float(np.sum(x)), float(np.sum(y))
        xn, yn = x / sx, y / sy
        lx = np.asarray(
            ln_phi(xn, temperature_k, self.pressure_pa, self.mixture, largest=False)
        )
        ly = np.asarray(
            ln_phi(yn, temperature_k, self.pressure_pa, self.mixture, largest=True)
        )
        return np.log(x) + lx - np.log(y) - ly

    def k_values(self, v) -> np.ndarray:
        """``K_i = (y_i/Sy) / (x_i/Sx)``, from the point itself. [-]"""
        _, x, y = self.unpack(v)
        return (y / np.sum(y)) / (x / np.sum(x))

    def source_feasibility(self, v, temperature_k: float) -> Dict[str, float]:
        """Original-space feasibility, with complementarity kept apart.

        Five quantities, never merged into one headline number, and the
        complementarity product is not among the ones that may be
        compared against ``tol``: it is a source quantity whose floor is
        ``sqrt(tol)`` (the Gate 0 report's accuracy floor), and judging
        it at ``tol`` would report every correct answer as a failure.
        """
        v = np.asarray(v, dtype=float)
        bal = float(np.max(np.abs(self.balance_residual(v))))
        iso = float(np.max(np.abs(self.isofugacity_residual(v, temperature_k))))
        bnd = float(np.max(np.maximum(np.maximum(self.lb - v, v - self.ub), 0.0)))
        g, h = self.pair_values(v)
        sign = float(np.max(np.maximum(np.maximum(-g, -h), 0.0)))
        prod = np.abs(g * h)
        return {
            "balance_viol": bal,
            "isofugacity_viol": iso,
            "bound_viol": bnd,
            "sign_viol": sign,
            "compl_max": float(prod.max()),
            "compl_sum": float(prod.sum()),
        }


#: The pinned Gate 1 case.
#:
#: 10 bar is chosen so that a single temperature path crosses all three
#: regimes with both switches interior to it and far from the mixture
#: critical point: liquid below ~267 K, two-phase, vapor above ~327 K.
#: The path is deliberately not symmetric about the two-phase region --
#: it carries more points near the switches, because that is where a
#: route either follows the branch or does not.
def _path() -> np.ndarray:
    coarse = np.arange(230.0, 361.0, 10.0)
    near_bubble = np.arange(262.0, 273.0, 1.0)
    near_dew = np.arange(322.0, 333.0, 1.0)
    return np.unique(np.concatenate([coarse, near_bubble, near_dew]))


GATE1_FLASH = FlashCase(
    name="ethane_n_butane_10bar",
    mixture=ETHANE_N_BUTANE,
    pressure_pa=10.0e5,
    temperatures_k=_path(),
    provenance=(
        "gh#776 Gate 1. Peng--Robinson, classical mixing, k_ij = 0; constants "
        "from phase_envelope.NATURAL_GAS. The bubble and dew temperatures are "
        "not asserted from a handbook -- they are located by the independent "
        "oracle in oracle.py and recorded in the manifest."
    ),
)

# ====================================================================
# The independent oracle
# ====================================================================
#
# The reason the fixture can claim to be *validated* rather than merely
# converged. What is shared with the model above: `ln_phi`, and the
# phase-label root convention. Everything else is a different algorithm
# reaching the same answer by a different route --
#
#   phase detection      a complementarity pair | Michelsen TPD, multistart
#   the two-phase solve  an interior-point NLP  | successive substitution +
#                                                 Rachford-Rice, then a Newton
#                                                 polish on (lnK, beta)
#   the certificate      one stationary point   | every stationary point the
#                                                 multistart reaches
#
# The last row is an asymmetry and it is the right way round. The
# MPCC's single-phase branch asserts that the *one* trial phase its
# solve landed on has `sum Y <= 1`; Michelsen's test is a statement
# about the global minimum of the tangent-plane distance, which no
# single stationary point establishes. Agreement therefore means the
# MPCC found the stationary point that matters, and a disagreement is a
# finding about the MPCC rather than about the oracle.
#
# Why the cubic root is chosen by phase label
# -------------------------------------------
#
# The first draft chose each root by comparing reduced Gibbs energies,
# on the reasoning that an energy criterion is more principled than a
# label. Measured, it is wrong here in two ways a green two-phase
# region would have hidden:
#
# * It collapses the trial phase onto the trivial solution wherever the
#   feed's cubic has one real root. At 230 K and 350 K on the pinned
#   path the feed has a single root, so "the other root" does not exist
#   to compare against, and every start converged to `Y = z`. The
#   oracle reported "stable, no incipient phase", which reads as a
#   well-posed answer and is not one: iterating with the *opposite
#   label* to the feed finds the incipient phase at `sum Y = 0.394`
#   (230 K) and `0.622` (350 K).
# * It makes the two-phase system symmetric. With both phases at
#   whichever root is lower in Gibbs energy, `(x, y, beta)` and
#   `(y, x, 1-beta)` are both solutions, and at 310 K the Newton polish
#   returned the mirror one -- `beta = 0.339` where the answer is
#   `0.661`. Labelling the roots breaks the symmetry, because the
#   liquid *is* the small root and the vapor the large one.
#
# So both calculations use the label convention. That costs a dimension
# of independence, and the honest replacement is to check the
# convention rather than to vary it: `root_diagnostics` compares the
# label-selected root against the lower-Gibbs one wherever two roots
# exist. The guard is that check, not the choice.

#: Successive substitution stops here; the Newton polish takes it the
#: rest of the way. SS alone converges linearly with a ratio that
#: approaches one near a switch point, and an oracle only as tight as
#: the thing it judges is not an oracle.

#: Successive substitution stops here; the Newton polish takes it the
#: rest of the way. SS alone converges linearly with a ratio that
#: approaches one near a switch point, and an oracle only as tight as
#: the thing it judges is not an oracle.
SS_TOL = 1e-11
SS_MAX_ITER = 1000

#: A trial phase whose ``sum Y`` exceeds one by more than this is a
#: genuine instability rather than a stationary point on the boundary.
#: At a switch point the true value *is* one, so this is the width of
#: the band in which the oracle reports the switch itself.
STABILITY_EPS = 1e-9

#: A trial phase this close to the feed composition is the trivial
#: stationary point ``Y = z``, which is always present and says nothing.
TRIVIAL_W = 1e-7


@dataclasses.dataclass
class TrialPhase:
    """One tangent-plane test: is the feed stable against this phase label?"""

    #: ``"vapor"`` tests a vapor-like trial phase against a liquid feed;
    #: ``"liquid"`` the other way round.
    trial_label: str
    #: ``sum_i Y_i`` at the most unstable stationary point found. This is
    #: the MPCC's ``Sy`` (vapor trial) or ``Sx`` (liquid trial).
    sum_y: float
    #: Tangent-plane distance there; negative means unstable.
    tpd: float
    stable: bool
    composition: Optional[np.ndarray]
    #: ``(start label, sum Y, tpd)`` for every start that converged, so
    #: a record can show the verdict came from more than one.
    stationary_points: List[Tuple[str, float, float]] = dataclasses.field(
        default_factory=list
    )


def _starts(z, temperature_k, pressure_pa, mixture) -> List[Tuple[str, np.ndarray]]:
    """Michelsen's recommended starts: vapor-like, liquid-like, and pure.

    The pure-component starts are what make this a multistart rather
    than a two-sided guess, and they are the ones that find the
    incipient phase at the ends of the path where the Wilson estimate is
    poor. Wilson's correlation appears here and nowhere else in this
    module: it supplies *starts*, never values, so nothing the oracle
    reports depends on its accuracy.
    """
    z = np.asarray(z, dtype=float)
    nc = z.size
    k = wilson_k(temperature_k, pressure_pa, mixture)
    out = [
        ("wilson_vapor", k * z),
        ("wilson_liquid", z / k),
        # Michelsen's cube-root pair. Not decoration: at 266 K on the
        # pinned path the four starts above it reach only the trivial
        # stationary point for the liquid trial phase, while the true
        # one sits at ``sum Y = 4.6``. A multistart that misses a
        # stationary point reports a weaker stability claim than it
        # could, and the whole value of this module is that its claim is
        # the stronger of the two.
        ("wilson_vapor_cbrt", z * np.cbrt(k)),
        ("wilson_liquid_cbrt", z / np.cbrt(k)),
    ]
    for i in range(nc):
        w = np.full(nc, 0.01 / max(nc - 1, 1))
        w[i] = 0.99
        out.append((f"pure_{mixture.names[i]}", w))
    return out


def tangent_plane(
    z,
    temperature_k: float,
    pressure_pa: float,
    mixture: PengRobinsonMixture,
    *,
    trial_is_vapor: bool,
) -> TrialPhase:
    """Michelsen's test on feed ``z`` against a trial phase of one label.

    Iterates ``ln Y_i = d_i - ln phi_i(Y/sum Y)`` from every start, with
    ``d_i = ln z_i + ln phi_i(z)`` the feed's tangent plane. The feed
    takes the root *opposite* the trial phase's, which is what makes
    this the test for "is the feed, as that phase, stable?".
    """
    z = np.asarray(z, dtype=float)
    d = np.log(z) + np.asarray(
        ln_phi(z, temperature_k, pressure_pa, mixture, largest=not trial_is_vapor)
    )

    best: Optional[Tuple[float, float, np.ndarray]] = None
    points: List[Tuple[str, float, float]] = []
    for label, y0 in _starts(z, temperature_k, pressure_pa, mixture):
        y = np.array(y0, dtype=float)
        converged = False
        for _ in range(SS_MAX_ITER):
            w = y / np.sum(y)
            lnp = np.asarray(
                ln_phi(w, temperature_k, pressure_pa, mixture, largest=trial_is_vapor)
            )
            y_new = np.exp(d - lnp)
            if not np.all(np.isfinite(y_new)) or np.any(y_new <= 0.0):
                break
            if np.max(np.abs(np.log(y_new / y))) < SS_TOL:
                y = y_new
                converged = True
                break
            y = y_new
        if not converged:
            continue
        s = float(np.sum(y))
        w = y / s
        if np.max(np.abs(np.log(w / z))) < TRIVIAL_W:
            points.append((f"{label} (trivial)", s, 0.0))
            continue
        lnp = np.asarray(
            ln_phi(w, temperature_k, pressure_pa, mixture, largest=trial_is_vapor)
        )
        tpd = float(1.0 + np.sum(y * (np.log(y) + lnp - d - 1.0)))
        points.append((label, s, tpd))
        if best is None or s > best[0]:
            best = (s, tpd, w)

    if best is None:
        # Only the trivial stationary point. The feed is stable against
        # this phase label and there is no incipient phase to report --
        # a different claim from "sum Y = 0", so the sum is reported as
        # the trivial point's own value of 1 and the composition as None.
        return TrialPhase(
            trial_label="vapor" if trial_is_vapor else "liquid",
            sum_y=1.0,
            tpd=0.0,
            stable=True,
            composition=None,
            stationary_points=points,
        )
    s, tpd, w = best
    return TrialPhase(
        trial_label="vapor" if trial_is_vapor else "liquid",
        sum_y=s,
        tpd=tpd,
        stable=bool(s <= 1.0 + STABILITY_EPS),
        composition=w,
        stationary_points=points,
    )


# --------------------------------------------------------------------
# the two-phase solve
# --------------------------------------------------------------------


def rachford_rice(z, k) -> float:
    """The vapor-fraction root of ``sum_i z_i (K_i-1)/(1 + b(K_i-1))``.

    Monotone decreasing in ``b``, so the root is unique where one
    exists, and the endpoints decide the single-phase cases without a
    bracket search.
    """
    z = np.asarray(z, dtype=float)
    k = np.asarray(k, dtype=float)

    def f(b):
        return float(np.sum(z * (k - 1.0) / (1.0 + b * (k - 1.0))))

    if f(0.0) <= 0.0:
        return 0.0
    if f(1.0) >= 0.0:
        return 1.0
    return float(brentq(f, 0.0, 1.0, xtol=1e-15, rtol=8.9e-16, maxiter=200))


@dataclasses.dataclass
class FlashResult:
    """The oracle's answer at one ``(z, T, P)``."""

    regime: str
    beta: float
    x: np.ndarray
    y: np.ndarray
    k: np.ndarray
    #: Directly comparable with the MPCC's ``Sx`` and ``Sy``: in a
    #: single-phase regime these are the tangent-plane tests' ``sum Y``,
    #: and in the two-phase regime both are one.
    sum_x: float
    sum_y: float
    #: Both tests, at every point, whichever regime won. Reporting the
    #: losing one is not padding: "stable as a vapor" and "unstable as a
    #: liquid" are separate facts, and a point where both tests come
    #: back stable is a metastable ambiguity that a single verdict would
    #: hide.
    vapor_trial: TrialPhase
    liquid_trial: TrialPhase
    converged: bool
    residual: float
    trivial: bool = False
    note: str = ""

    @property
    def ambiguous(self) -> bool:
        """Both labels stable *and* both incipient phases real.

        A metastable point, where the tangent-plane test alone does not
        name the phase. Distinct from `no_incipient_phase` below, which
        is the far more common reason for two "stable" verdicts and is
        not an ambiguity at all -- one of the tests simply had nothing
        to test against. Conflating the two is what made the first draft
        of this module flag five ordinary points on the pinned path.
        """
        return (
            self.regime != "two_phase"
            and self.vapor_trial.stable
            and self.liquid_trial.stable
            and self.vapor_trial.composition is not None
            and self.liquid_trial.composition is not None
        )

    @property
    def no_incipient_phase(self) -> bool:
        """Neither label has a non-trivial stationary point.

        A formulation boundary rather than a numerical one, and worth a
        field of its own because of what it does to the MPCC in
        `spec.py`. With no incipient phase, ``x = y = z`` is the only
        solution of the isofugacity rows, so ``Sx = Sy = 1``, *both*
        pair slacks vanish, and ``beta`` is left completely
        undetermined -- a one-parameter family of solutions rather than
        a degenerate point. The pinned path does not reach it (every
        point has an incipient phase for at least one label), and a
        record that ever reports it is reporting that the fixture has
        left the region where it means anything.
        """
        return self.vapor_trial.composition is None and self.liquid_trial.composition is None


def _two_phase(z, temperature_k, pressure_pa, mixture, k0) -> Tuple[np.ndarray, float, float]:
    """Successive substitution, then a Newton polish on ``(ln K, beta)``.

    The liquid is always the small cubic root and the vapor the large
    one, which is what keeps ``(x, y, beta)`` and ``(y, x, 1-beta)``
    from both being solutions. Returns ``(K, beta, residual)``.
    """
    z = np.asarray(z, dtype=float)
    k = np.array(k0, dtype=float)
    for _ in range(SS_MAX_ITER):
        beta = rachford_rice(z, k)
        x = z / (1.0 + beta * (k - 1.0))
        y = k * x
        xn, yn = x / np.sum(x), y / np.sum(y)
        lx = np.asarray(ln_phi(xn, temperature_k, pressure_pa, mixture, largest=False))
        ly = np.asarray(ln_phi(yn, temperature_k, pressure_pa, mixture, largest=True))
        k_new = np.exp(lx - ly)
        if np.max(np.abs(np.log(k_new / k))) < SS_TOL:
            k = k_new
            break
        k = k_new

    def residual(u):
        kk = np.exp(u[:-1])
        b = u[-1]
        xx = z / (1.0 + b * (kk - 1.0))
        yy = kk * xx
        lx = np.asarray(
            ln_phi(xx / np.sum(xx), temperature_k, pressure_pa, mixture, largest=False)
        )
        ly = np.asarray(
            ln_phi(yy / np.sum(yy), temperature_k, pressure_pa, mixture, largest=True)
        )
        return np.concatenate([u[:-1] - (lx - ly), [np.sum(yy - xx)]])

    beta = rachford_rice(z, k)
    u0 = np.concatenate([np.log(k), [beta]])
    before = float(np.max(np.abs(residual(u0))))
    sol = root(residual, u0, method="hybr", tol=1e-14)
    # The polish is accepted only if it helps and stays in the box. A
    # Newton step that leaves [0,1] has left the model, and silently
    # keeping it is how an oracle starts reporting the mirror solution.
    if sol.success and 0.0 <= sol.x[-1] <= 1.0:
        after = float(np.max(np.abs(residual(sol.x))))
        if after <= before:
            return np.exp(sol.x[:-1]), float(sol.x[-1]), after
    return k, float(beta), before


def flash(
    temperature_k: float,
    pressure_pa: float,
    mixture: PengRobinsonMixture,
    z=None,
) -> FlashResult:
    """Both stability tests first, then a two-phase solve only if warranted.

    This order is the whole point. A flash that goes straight to
    successive substitution from a Wilson start collapses onto the
    trivial solution wherever the feed is comfortably single-phase and
    then reports a converged two-phase answer with ``K_i = 1``. The
    stability tests are what make "single phase" a *result* rather than
    a failure to converge.
    """
    z = np.asarray(mixture.composition if z is None else z, dtype=float)
    vap = tangent_plane(z, temperature_k, pressure_pa, mixture, trial_is_vapor=True)
    liq = tangent_plane(z, temperature_k, pressure_pa, mixture, trial_is_vapor=False)

    if vap.stable or liq.stable:
        # Which single phase? Not a heuristic on ``Z``, and not the
        # tie-break between two stability numbers that the first draft
        # used. The feed is whichever phase the *incipient* one is not:
        # a non-trivial vapor-like trial phase can only condense out of
        # a liquid.
        #
        # This matters at the ends of the path, where the feed's cubic
        # has a single real root and one of the two tests therefore has
        # nothing to say -- it finds only the trivial stationary point
        # and reports ``sum Y = 1``, stable. Reading that as a second
        # stability verdict is what made the first draft call 230 K a
        # vapor. It is not a verdict; it is the absence of one.
        has_vap, has_liq = vap.composition is not None, liq.composition is not None
        if has_vap != has_liq:
            as_liquid = has_vap
        else:
            as_liquid = vap.stable and (not liq.stable or vap.sum_y <= liq.sum_y)
        trial = vap if as_liquid else liq
        amount = trial.sum_y
        w = trial.composition
        trial_vec = z.copy() if w is None else w * amount
        if as_liquid:
            regime, beta, x, y = "liquid", 0.0, z.copy(), trial_vec
        else:
            regime, beta, x, y = "vapor", 1.0, trial_vec, z.copy()
        k = (y / np.sum(y)) / (x / np.sum(x))
        return FlashResult(
            regime=regime,
            beta=beta,
            x=x,
            y=y,
            k=k,
            sum_x=float(np.sum(x)),
            sum_y=float(np.sum(y)),
            vapor_trial=vap,
            liquid_trial=liq,
            converged=True,
            residual=0.0,
            trivial=bool(w is None),
            note=(
                "single phase; no non-trivial incipient phase"
                if w is None
                else "single phase; incipient phase from the tangent-plane test"
            ),
        )

    # Both labels unstable: two phases. Seed from the incipient phase
    # the vapor test found, which near a switch point is a far better
    # start than Wilson and is free -- the test has already done it.
    if vap.composition is not None:
        k0 = vap.composition / z
        if np.max(np.abs(np.log(k0))) < 1e-3:
            k0 = wilson_k(temperature_k, pressure_pa, mixture)
    else:  # pragma: no cover - unstable with no trial phase cannot happen
        k0 = wilson_k(temperature_k, pressure_pa, mixture)

    k, beta, res = _two_phase(z, temperature_k, pressure_pa, mixture, k0)
    x = z / (1.0 + beta * (k - 1.0))
    y = k * x
    trivial = is_trivial(k)
    return FlashResult(
        regime="two_phase",
        beta=float(beta),
        x=x,
        y=y,
        k=k,
        sum_x=float(np.sum(x)),
        sum_y=float(np.sum(y)),
        vapor_trial=vap,
        liquid_trial=liq,
        converged=bool(res < 1e-9 and not trivial),
        residual=res,
        trivial=trivial,
        note="trivial solution" if trivial else "",
    )


# --------------------------------------------------------------------
# the switch points
# --------------------------------------------------------------------


def switch_temperature(
    lo_k: float,
    hi_k: float,
    pressure_pa: float,
    mixture: PengRobinsonMixture,
    z=None,
    *,
    trial_is_vapor: bool,
    xtol: float = 1e-10,
) -> float:
    """Bisect one stability boundary between ``lo_k`` and ``hi_k``.

    The function bisected is ``sum Y - 1`` from the tangent-plane test,
    which is continuous across the boundary and changes sign there.
    ``beta`` is not -- it is pinned at 0 or 1 on one side -- so
    bisecting on ``beta`` would find nothing, which is why the switch
    points are located from the stability test rather than from the
    flash.
    """
    z = np.asarray(mixture.composition if z is None else z, dtype=float)

    def f(t):
        return (
            tangent_plane(z, t, pressure_pa, mixture, trial_is_vapor=trial_is_vapor).sum_y - 1.0
        )

    f_lo, f_hi = f(lo_k), f(hi_k)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"bracket [{lo_k}, {hi_k}] does not straddle the "
            f"{'bubble' if trial_is_vapor else 'dew'} boundary "
            f"(sum Y - 1 = {f_lo:.3e} and {f_hi:.3e})"
        )
    return float(brentq(f, lo_k, hi_k, xtol=xtol, rtol=8.9e-16, maxiter=200))


def bubble_and_dew(case, xtol: float = 1e-10) -> Dict[str, float]:
    """The two switch temperatures on the case's own path, located.

    Brackets come from the path itself by finding the regime changes, so
    this cannot return a number for a path that does not cross a
    boundary -- it raises instead.
    """
    z = case.z
    temps = np.asarray(case.temperatures_k, dtype=float)
    regimes = [flash(float(t), case.pressure_pa, case.mixture, z).regime for t in temps]
    out: Dict[str, float] = {}
    for i in range(len(temps) - 1):
        a, b = regimes[i], regimes[i + 1]
        if a == "liquid" and b == "two_phase":
            out["bubble_k"] = switch_temperature(
                float(temps[i]),
                float(temps[i + 1]),
                case.pressure_pa,
                case.mixture,
                z,
                trial_is_vapor=True,
                xtol=xtol,
            )
        elif a == "two_phase" and b == "vapor":
            out["dew_k"] = switch_temperature(
                float(temps[i]),
                float(temps[i + 1]),
                case.pressure_pa,
                case.mixture,
                z,
                trial_is_vapor=False,
                xtol=xtol,
            )
    return out

# ====================================================================
# MPCC -> smooth NLP
# ====================================================================
#
# The three lowerings are the ones gh#794 compared, unchanged, because
# comparing this model against Gate 0's supported route only means
# something if the route is the same object:
#
#   prod_ineq   H >= 0, G*H <= 0. The direct formulation; its feasible
#               points are exactly the MPCC's, since G, H >= 0 makes
#               G*H <= 0 equivalent to G*H = 0.
#   prod_eq     the same with G*H = 0: exact-product / NCP equality.
#   scholtes    G*H <= tau, feasible for the MPCC only as tau -> 0.
#
# Two things differ from Gate 0's algebraic corpus, and both are
# properties of the model rather than choices:
#
# * **The `G` sides are bounds, not rows.** `G_V = beta` and
#   `G_L = 1 - beta` are already enforced exactly by `beta in [0, 1]`,
#   and adding them again as rows would put two linearly dependent rows
#   into every active set for no gain. An MPCC's active set is
#   degenerate enough by construction; manufacturing more of it makes a
#   solver failure harder to attribute.
# * **The rows are nonlinear.** Gate 0's corpus is quadratic with
#   affine pairs so that every derivative is closed-form. Here the
#   isofugacity rows carry a cubic root and a logarithm, so the
#   derivatives come from JAX -- exact to round-off, not finite
#   differences -- and `fd_check` verifies them against a central
#   difference anyway, because "the derivative is the first suspect"
#   survives the change of technique.
#
# Row order is fixed and is part of the contract; a caller reads
# multipliers back out of `info["mult_g"]` positionally:
#
#     [ balance_1..nc ] [ isofug_1..nc ] [ H_V ] [ H_L ] [ prod_V ] [ prod_L ]

LOWERINGS = ("prod_ineq", "prod_eq", "scholtes")


def _residuals(v, temperature_k, case: FlashCase):
    """The MPCC's own rows at ``v``, in the order documented above.

    Written once, in JAX, and used for both the value and every
    derivative -- so a row and its gradient cannot drift apart, which is
    the single most common way a benchmark ends up measuring its own
    harness.
    """
    nc = case.nc
    beta = v[0]
    x = v[1 : 1 + nc]
    y = v[1 + nc : 1 + 2 * nc]
    sx = jnp.sum(x)
    sy = jnp.sum(y)
    xn = x / sx
    yn = y / sy
    z = jnp.asarray(case.z)

    balance = (1.0 - beta) * x + beta * y - z
    ln_phi_l = log_fugacity_coefficients(
        xn, temperature_k, case.pressure_pa, case.mixture, largest=False
    )
    ln_phi_v = log_fugacity_coefficients(
        yn, temperature_k, case.pressure_pa, case.mixture, largest=True
    )
    # Normalized *inside* phi, un-normalized in the log. See the
    # "Where the normalization goes" section of `spec.py`: carrying the
    # normalization into the log term too is a defect that is invisible
    # in the two-phase region and wrong in both single-phase ones.
    isofug = jnp.log(x) + ln_phi_l - jnp.log(y) - ln_phi_v

    h_v = 1.0 - sy
    h_l = 1.0 - sx
    prod_v = beta * h_v
    prod_l = (1.0 - beta) * h_l
    return jnp.concatenate([balance, isofug, jnp.array([h_v, h_l, prod_v, prod_l])])


#: Compiled callbacks, one set per case rather than per solve.
#:
#: **The Hessian is forward-over-forward on purpose.** `jax.hessian` is
#: `jacfwd(jacrev(.))`, and under `jax.jit` its reverse-mode half is
#: catastrophically inaccurate on this model near the cubic's
#: discriminant boundary -- where the equation of state has a double
#: root and Cardano's trigonometric branch runs `arccos` into its
#: endpoint singularity. Measured against a value-only second
#: difference, at the oracle's own point:
#:
#: =========================  ===============  ==============
#: composition                268 K and 270 K  everywhere else
#: =========================  ===============  ==============
#: `jit(jax.hessian)`         **2.1e+01**      3e-14
#: `jit(jacfwd(grad))`        **2.1e+01**      3e-14
#: `jit(jacrev(jacfwd))`      **5.8e+00**      2e-14
#: `jit(jacfwd(jacfwd))`      2.3e-14          3e-14
#: =========================  ===============  ==============
#:
#: 268 K and 270 K are the two path points straddling the bubble point
#: at 268.89 K, and they are the *only* two where it happens -- which is
#: exactly what makes it dangerous. The Jacobian is unaffected in every
#: mode, so gradients are right, KKT residuals are right, and the
#: converged answers were right: the full traversal agreed with the
#: oracle to 1e-11 at all 34 temperatures *with the wrong Hessian*. It
#: costs iterations and robustness at the one place the fixture exists
#: to test, and nothing reports it.
#:
#: Unjitting the Hessian also fixes it and costs 240 ms per call against
#: 0.15 ms, which is not available to a fixture gh#776 asks to be fast.
#: Forward-over-forward is exact and free.
#:
#: ``temperature_k`` is a *traced* argument, not baked in, so the whole
#: temperature path shares one compilation. Closing over it instead --
#: which the first draft did, via `functools.partial` -- costs a JAX
#: compilation per stage: 34 temperatures times ten continuation stages
#: is 340 of them, and the traversal spent essentially all of its wall
#: clock compiling the same function again. Compilation time is not
#: measurement, and a fixture gh#776 asks to be *fast* cannot pay it per
#: stage.
_COMPILED: Dict[int, tuple] = {}


def _compiled(case: FlashCase) -> tuple:
    key = id(case)
    hit = _COMPILED.get(key)
    if hit is not None:
        return hit

    def fn(v, t):
        return _residuals(v, t, case)

    built = (
        jax.jit(fn),
        jax.jit(jax.jacfwd(fn, argnums=0)),
        # Forward-over-forward, and NOT `jax.hessian`. See the note
        # below: reverse mode loses this model's Hessian entirely in a
        # ~1 K band around the bubble point, and costs nothing to avoid.
        jax.jit(
            jax.jacfwd(
                jax.jacfwd(lambda v, lam, t: jnp.dot(lam, fn(v, t)), argnums=0),
                argnums=0,
            )
        ),
    )
    _COMPILED[key] = built
    return built


@dataclasses.dataclass
class LoweredFlash:
    """A cyipopt-style callback object plus the bookkeeping to read it back."""

    case: FlashCase
    temperature_k: float
    lowering: str
    tau: Optional[float]
    n: int
    m: int
    lb: np.ndarray
    ub: np.ndarray
    cl: np.ndarray
    cu: np.ndarray
    balance_row0: int
    isofug_row0: int
    h_row0: int
    prod_row0: int
    _c: object = None
    _j: object = None
    _h: object = None
    _t: object = None

    # -- cyipopt-style callbacks ----------------------------------
    #
    # The objective is identically zero: this is a *square flash*, a
    # feasibility problem, and gh#776's Gate 1 asks for a flash rather
    # than a design. Giving it an objective would change which point in
    # a degenerate solution set the solver returns, and the fixture's
    # whole claim is that the point is the one the oracle computes.

    def objective(self, x):
        return 0.0

    def gradient(self, x):
        return np.zeros(self.n)

    def constraints(self, x):
        return np.asarray(
            self._c(jnp.asarray(x, dtype=jnp.float64), self._t), dtype=float
        )

    def jacobian(self, x):
        return np.asarray(
            self._j(jnp.asarray(x, dtype=jnp.float64), self._t), dtype=float
        ).reshape(-1)

    def jacobianstructure(self):
        rows = np.repeat(np.arange(self.m), self.n)
        cols = np.tile(np.arange(self.n), self.m)
        return rows, cols

    def hessianstructure(self):
        return np.tril_indices(self.n)

    def hessian(self, x, lagrange, obj_factor):
        full = np.asarray(
            self._h(
                jnp.asarray(x, dtype=jnp.float64),
                jnp.asarray(lagrange, dtype=jnp.float64),
                self._t,
            ),
            dtype=float,
        )
        r, c = np.tril_indices(self.n)
        return full[r, c]

    # -- reading a solve back -------------------------------------

    def pair_multipliers(self, mult_g) -> Tuple[np.ndarray, np.ndarray]:
        """``(mult_H, mult_prod)`` sliced out of ``info['mult_g']``.

        The *NLP* multipliers of the lowered rows. They are reported and
        they are not the MPCC's multipliers; nothing in this harness
        presents them as such.
        """
        mg = np.asarray(mult_g, dtype=float)
        return (
            mg[self.h_row0 : self.h_row0 + 2],
            mg[self.prod_row0 : self.prod_row0 + 2],
        )

    @property
    def row_names(self) -> List[str]:
        names = [f"balance_{c}" for c in self.case.mixture.names]
        names += [f"isofugacity_{c}" for c in self.case.mixture.names]
        return names + ["H_vapor", "H_liquid", "prod_vapor", "prod_liquid"]


def lower(
    case: FlashCase, temperature_k: float, lowering: str, tau: Optional[float] = None
) -> LoweredFlash:
    """Build the smooth NLP for ``case`` at ``temperature_k``.

    ``tau`` is required by ``scholtes`` and rejected by the others: a
    relaxation parameter on a non-relaxed lowering would be silently
    ignored, and a record whose ``tau`` says ``1e-8`` when nothing read
    it is worse than no field at all.
    """
    if lowering not in LOWERINGS:
        raise ValueError(f"unknown lowering {lowering!r}")
    if lowering == "scholtes":
        if tau is None:
            raise ValueError("scholtes lowering needs tau")
    elif tau is not None:
        raise ValueError(f"{lowering} lowering takes no tau")

    nc = case.nc
    n = case.n
    m = 2 * nc + 4
    balance_row0, isofug_row0, h_row0, prod_row0 = 0, nc, 2 * nc, 2 * nc + 2

    cl = np.zeros(m)
    cu = np.zeros(m)
    cl[h_row0 : h_row0 + 2] = 0.0
    cu[h_row0 : h_row0 + 2] = np.inf
    if lowering == "prod_eq":
        cl[prod_row0 : prod_row0 + 2] = 0.0
        cu[prod_row0 : prod_row0 + 2] = 0.0
    elif lowering == "prod_ineq":
        cl[prod_row0 : prod_row0 + 2] = -np.inf
        cu[prod_row0 : prod_row0 + 2] = 0.0
    else:
        cl[prod_row0 : prod_row0 + 2] = -np.inf
        cu[prod_row0 : prod_row0 + 2] = float(tau)

    c, j, h = _compiled(case)

    return LoweredFlash(
        case=case,
        temperature_k=float(temperature_k),
        lowering=lowering,
        tau=tau,
        n=n,
        m=m,
        lb=case.lb.copy(),
        ub=case.ub.copy(),
        cl=cl,
        cu=cu,
        balance_row0=balance_row0,
        isofug_row0=isofug_row0,
        h_row0=h_row0,
        prod_row0=prod_row0,
        _c=c,
        _j=j,
        _h=h,
        _t=jnp.asarray(float(temperature_k), dtype=jnp.float64),
    )


# ====================================================================
# The same MPCC, declared
# ====================================================================
#
# Everything above writes the complementarity by hand: the product rows
# are ordinary constraints, and which lowering to use, the tau schedule
# and the finish are the caller's. `declare` hands POUNCE the *pairs*
# instead (`pounce.Problem(..., complementarity=...)`) and lets its
# driver choose the formulation, run the finish, and certify the point.
#
# One change of variables is needed, and it is forced by the declaration
# rule rather than chosen: a variable may sit in only one pair, and the
# hand-written model pairs `beta` on *both* of its bounds (`beta ⟂ H_V`
# and `1 - beta ⟂ H_L`). So the declared model carries `gamma = 1 - beta`
# as its own variable, tied by one linear row, and each pair names its
# own variable's lower bound:
#
#     0 <= beta   ⟂  H_V = 1 - Sy >= 0
#     0 <= gamma  ⟂  H_L = 1 - Sx >= 0,      beta + gamma = 1
#
# Variables are `[beta, x, y, gamma]`, so `v[:case.n]` is the
# hand-written model's point and every `FlashCase` method reads it
# unchanged. Rows are the hand-written ones minus the two products, plus
# the link:
#
#     [ balance_1..nc ] [ isofug_1..nc ] [ H_V ] [ H_L ] [ link ]

DECLARED = "declared"


def _declared_residuals(v, temperature_k, case: FlashCase):
    """The declared model's rows: the hand-written rows without the products."""
    n = case.n
    rows = _residuals(v[:n], temperature_k, case)[: 2 * case.nc + 2]
    return jnp.concatenate([rows, jnp.array([v[0] + v[n] - 1.0])])


_COMPILED_DECLARED: Dict[int, tuple] = {}


def _compiled_declared(case: FlashCase) -> tuple:
    key = id(case)
    hit = _COMPILED_DECLARED.get(key)
    if hit is not None:
        return hit

    def fn(v, t):
        return _declared_residuals(v, t, case)

    # Forward-over-forward for the Hessian, for the reason `_compiled`
    # documents.
    built = (
        jax.jit(fn),
        jax.jit(jax.jacfwd(fn, argnums=0)),
        jax.jit(
            jax.jacfwd(
                jax.jacfwd(lambda v, lam, t: jnp.dot(lam, fn(v, t)), argnums=0),
                argnums=0,
            )
        ),
    )
    _COMPILED_DECLARED[key] = built
    return built


@dataclasses.dataclass
class DeclaredFlash(LoweredFlash):
    """`LoweredFlash`'s callbacks for the declared model, plus its pairs.

    Pass ``complementarity`` to `pounce.Problem` with the rest; nothing
    else differs. ``prod_row0`` is ``-1``: there are no product rows,
    because writing them is the driver's job.
    """

    link_row: int = -1
    complementarity: List[Tuple[int, int]] = dataclasses.field(default_factory=list)

    def flash_point(self, v) -> np.ndarray:
        """The hand-written model's ``[beta, x, y]`` from a declared point."""
        return np.asarray(v, dtype=float)[: self.case.n]

    def start(self, v_flash) -> np.ndarray:
        """A declared start from a hand-written one: ``gamma = 1 - beta``."""
        v_flash = np.asarray(v_flash, dtype=float)[: self.case.n]
        return np.concatenate([v_flash, [1.0 - v_flash[0]]])

    def pair_multipliers(self, mult_g):
        raise TypeError(
            "the declared model has no product rows; read the driver's "
            "certificate from info['complementarity'] instead"
        )

    @property
    def row_names(self) -> List[str]:
        names = [f"balance_{c}" for c in self.case.mixture.names]
        names += [f"isofugacity_{c}" for c in self.case.mixture.names]
        return names + ["H_vapor", "H_liquid", "link_beta_gamma"]


def declare(case: FlashCase, temperature_k: float) -> DeclaredFlash:
    """Build the flash with its complementarity *declared*, not lowered.

    Solve it with ``pounce.Problem(..., complementarity=nlp.complementarity)``.
    """
    nc = case.nc
    n = case.n + 1
    m = 2 * nc + 3
    balance_row0, isofug_row0, h_row0, link_row = 0, nc, 2 * nc, 2 * nc + 2

    cl = np.zeros(m)
    cu = np.zeros(m)
    cu[h_row0 : h_row0 + 2] = np.inf

    c, j, h = _compiled_declared(case)
    return DeclaredFlash(
        case=case,
        temperature_k=float(temperature_k),
        lowering=DECLARED,
        tau=None,
        n=n,
        m=m,
        lb=np.concatenate([case.lb, [0.0]]),
        ub=np.concatenate([case.ub, [1.0]]),
        cl=cl,
        cu=cu,
        balance_row0=balance_row0,
        isofug_row0=isofug_row0,
        h_row0=h_row0,
        prod_row0=-1,
        _c=c,
        _j=j,
        _h=h,
        _t=jnp.asarray(float(temperature_k), dtype=jnp.float64),
        link_row=link_row,
        # (row, var): H_V with beta's lower bound, H_L with gamma's.
        complementarity=[(h_row0, 0), (h_row0 + 1, n - 1)],
    )


def fd_check(nlp: LoweredFlash, v: np.ndarray, h: float = 1e-6) -> dict:
    """Central-difference check of the declared derivatives.

    Gate 0's corpus is quadratic, so its equivalent check is exact to
    round-off and any tolerance above that hides something. Here the
    rows carry a cubic root and a logarithm, so the second differences
    carry genuine truncation error and the thresholds `selftest` applies
    are set to that, not tighter -- an assertion tighter than the method
    it uses would fail for a reason that has nothing to do with the
    model.
    """
    v = np.asarray(v, dtype=float)
    n = nlp.n

    def num_jac():
        out = np.zeros((nlp.m, n))
        for jx in range(n):
            e = np.zeros(n)
            e[jx] = h
            out[:, jx] = (nlp.constraints(v + e) - nlp.constraints(v - e)) / (2 * h)
        return out

    jac = nlp.jacobian(v).reshape(nlp.m, n)
    jerr = float(np.max(np.abs(num_jac() - jac)))

    rng = np.random.default_rng(776)
    lam = rng.normal(size=nlp.m)

    def lag_grad(w):
        return lam @ nlp.jacobian(w).reshape(nlp.m, n)

    num_h = np.zeros((n, n))
    for jx in range(n):
        e = np.zeros(n)
        e[jx] = h
        num_h[:, jx] = (lag_grad(v + e) - lag_grad(v - e)) / (2 * h)
    tri = nlp.hessian(v, lam, 0.0)
    full = np.zeros((n, n))
    r, c = np.tril_indices(n)
    full[r, c] = tri
    full = full + np.tril(full, -1).T
    herr = float(np.max(np.abs(full - 0.5 * (num_h + num_h.T))))
    return {"jac": jerr, "hess": herr}
