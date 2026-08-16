"""The solver plug-in interface.

An adapter's job is to solve one instance of a family at its current
parameter, optionally seeded with a :class:`~..spec.WarmState`, and
report back in the suite's own vocabulary. Everything
solver-specific — option names, warm-start plumbing, status codes,
working-set encodings — is confined to the adapter, which is what
makes it possible to add Ipopt (or anything else) to this benchmark
without touching a family.

Arms
----

An *arm* is a (algorithm, warm-start) pairing. The four the suite
defines:

``cold-ipm``
    Interior point, every step from the family's cold starting point.
    This is the baseline: it is what a solver without warm-start
    support does, and it is also the correctness reference.
``cold-sqp``
    Active-set SQP, every step cold. Present so that the warm-start
    effect can be separated from the algorithm change — without it,
    ``warm-sqp`` beating ``cold-ipm`` would confound the two.
``warm-ipm``
    Interior point seeded with the previous step's primal-dual point
    and barrier parameter. The fair comparison for ``warm-sqp``.
``warm-sqp``
    Active-set SQP seeded with the previous step's working set and
    primal point. The capability under test.
``cold-sqp-hom`` / ``warm-sqp-hom``
    The same active-set SQP, but with the inner QP's **cold** solves
    tracing the §4.2 parametric homotopy (`sqp_qp_use_homotopy`) rather
    than the conventional phase-1/phase-2 scheme. The homotopy is the
    algorithm ``pounce-qp`` is named for; it is off by default in the
    crate and on only in the convex QP driver, so without these arms
    the suite exercises working-set hot starts but never the parametric
    path. Paired against ``cold-sqp`` / ``warm-sqp``, which differ in
    that one option and nothing else.
``pred-ipm`` / ``predcorr-ipm``
    Interior point seeded with a **tangent predictor** rather than with
    the previous solution alone: the first-order parametric step
    ``Δx ≈ ∂x*/∂θ · Δθ`` read off the previous solve's held KKT factor
    (``pounce.Solver.parametric_step``). ``predcorr-ipm`` additionally
    steps the multipliers (``parametric_step_full``) and re-anchors on
    an active-set event. These are the third and fourth arms pounce#608
    asks for; paired against ``warm-ipm``, which differs only in the
    seed. Defined only for families that route θ through pin-equality
    rows (:attr:`ParametricFamily.pin_rows`), because that is what the
    sensitivity step's ``deltas`` argument means -- on any other family
    they are skipped with a reason rather than silently omitted.
``cold-qp-ipm`` / ``warm-qp-ipm``
    The dedicated convex QP interior-point solver, handed the problem
    in matrix form rather than through callbacks, cold and seeded with
    the previous step's primal-dual point. Only defined for families
    whose instances are literally QPs (``ParametricFamily.quadratic``);
    on any other family these arms are skipped with a reason rather
    than silently omitted. Note the asymmetry when reading wall time:
    this arm receives the problem data once per step, where the
    callback-driven arms re-evaluate every iteration.

An adapter declares which arms it supports; unsupported arms are
skipped and reported as such rather than silently omitted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from ..spec import ParametricFamily, StepResult, WarmState
from ..sparsity import SparseCallbacks

ARMS: List[str] = [
    "cold-ipm",
    "cold-sqp",
    "warm-ipm",
    "warm-sqp",
    "cold-sqp-hom",
    "warm-sqp-hom",
    "pred-ipm",
    "predcorr-ipm",
    "cold-qp-ipm",
    "warm-qp-ipm",
]

#: Arms that run the active-set SQP driver (either inner-QP variant).
SQP_ARMS = ("cold-sqp", "warm-sqp", "cold-sqp-hom", "warm-sqp-hom")

#: Arms whose inner QP traces the §4.2 parametric homotopy on a cold
#: solve (`sqp_qp_use_homotopy`) instead of the conventional
#: phase-1/phase-2 scheme. Paired against their non-`-hom` twins.
HOMOTOPY_ARMS = ("cold-sqp-hom", "warm-sqp-hom")

#: Arms that need the family's instances to be QPs.
QP_ARMS = ("cold-qp-ipm", "warm-qp-ipm")

#: Arms whose primal seed is a held-factor tangent predictor rather than
#: the previous solution (pounce#608). They need the family to declare
#: which constraint rows carry θ.
PREDICTOR_ARMS = ("pred-ipm", "predcorr-ipm")

#: The arm whose per-step solutions every other arm is checked
#: against, and which generates the parameter path for adaptive
#: families.
REFERENCE_ARM = "cold-ipm"


def is_warm(arm: str) -> bool:
    """Whether the arm carries state forward from the previous step.

    The predictor arms do: they seed from the previous solve and then
    step that seed along the sensitivity, so they are warm arms with a
    better seed, not a separate category.
    """
    return arm.startswith("warm") or arm in PREDICTOR_ARMS


def uses_predictor(arm: str) -> bool:
    return arm in PREDICTOR_ARMS


def predicts_duals(arm: str) -> bool:
    return arm == "predcorr-ipm"


def is_sqp(arm: str) -> bool:
    """Whether the arm drives the active-set SQP (either inner-QP variant)."""
    return arm in SQP_ARMS


def uses_homotopy(arm: str) -> bool:
    return arm in HOMOTOPY_ARMS


def arm_applies(arm: str, family: ParametricFamily) -> Optional[str]:
    """``None`` if the arm is defined for this family, else why not.

    Separate from :meth:`SolverAdapter.supports`, which is about what
    the *solver* can do. This is about what the *problem* admits: a
    convex QP solver has nothing to say about a family with nonlinear
    constraints, and running it there would be a category error rather
    than a slow result.
    """
    if arm in QP_ARMS and not family.quadratic:
        return "family is not a QP (nonlinear objective or constraints)"
    if arm in PREDICTOR_ARMS and not family.pin_rows:
        return (
            "family does not route theta through pin-equality rows, so a "
            "sensitivity step has no deltas to take"
        )
    return None


class SolverAdapter(ABC):
    """Solve one parametric instance, cold or warm."""

    #: Name used in output files and the report.
    name: str = "unnamed"

    @abstractmethod
    def supports(self, arm: str) -> bool:
        """Whether this adapter can run the given arm."""

    @abstractmethod
    def solve(
        self,
        family: ParametricFamily,
        callbacks: SparseCallbacks,
        arm: str,
        x0: np.ndarray,
        warm: Optional[WarmState],
        step: int,
        tol: float,
    ) -> Tuple[StepResult, Optional[WarmState]]:
        """Solve at the family's current θ.

        Returns the step's measurements and the state to hand to the
        next step (``None`` if this arm carries nothing forward). The
        adapter fills every :class:`~..spec.StepResult` field except
        the correctness ones, which the runner supplies once the
        reference solution is known.
        """
