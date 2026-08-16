"""Solver-agnostic core types for the warm-start benchmark.

Nothing in this module imports pounce. A *family* here is a
parameterized NLP plus a scripted path through its parameter space;
an *arm* is one way of solving that path (cold every step, or warm
from the previous step); an *adapter* (see ``adapters/``) is the only
place a concrete solver appears. Keeping the split strict is what
lets this suite be lifted out of the pounce tree later, or pointed at
Ipopt/knitro/whatever for a cross-solver comparison.

Why families-and-paths rather than a bag of problems: warm starting
has no meaning for a single isolated NLP. The quantity under test is
the cost of solving a *sequence* of related problems, and the thing
that predicts whether warm starting pays is how the active set moves
along that sequence. So the benchmark's unit of work is the whole
path, and every family carries tags describing the active-set regime
it was built to exercise.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import numpy as np

# Working-set status codes. These match pounce's int8 encoding
# (see docs/src/active-set-sqp.md §3) but are restated here so the
# core stays free of solver imports; an adapter for a solver with a
# different encoding translates into these.
WS_INACTIVE = 0
WS_AT_LOWER = 1
WS_AT_UPPER = 2
WS_FIXED = 3  # Fixed (variable) / Equality (constraint)

# Step-size multipliers applied to a family's natural per-step
# parameter increment. Warm-start payoff is a function of how far the
# problem moved, so every family is run at all three: `tiny` is the
# continuation regime (active set almost surely unchanged), `large`
# is where the active set churns and warm starting can *hurt*.
SCALES: Dict[str, float] = {"tiny": 0.1, "small": 1.0, "large": 4.0}


@dataclasses.dataclass(frozen=True)
class Bounds:
    """Variable and constraint bounds, cyipopt naming."""

    lb: np.ndarray
    ub: np.ndarray
    cl: np.ndarray
    cu: np.ndarray


@dataclasses.dataclass
class WarmState:
    """What one solve hands to the next.

    Deliberately a plain data record: primal point, the three
    multiplier blocks, the barrier parameter, and the discrete working
    set. A solver adapter consumes whichever fields its warm-start
    path understands (an active-set method wants ``working_set``; an
    interior-point method wants the multipliers and ``mu``) and
    ignores the rest.

    ``working_set`` is a ``(bound_status, constraint_status)`` pair of
    int8 arrays using the ``WS_*`` codes above.
    """

    x: np.ndarray
    mult_g: Optional[np.ndarray] = None
    mult_x_L: Optional[np.ndarray] = None
    mult_x_U: Optional[np.ndarray] = None
    mu: Optional[float] = None
    working_set: Optional[Sequence[np.ndarray]] = None
    #: Adapter-private payload. An adapter whose warm start does not fit
    #: the fields above (a conic solver's own primal-dual state, say) can
    #: stash it here; nothing in the core inspects it, and no other
    #: adapter should read it.
    extra: Optional[dict] = None


@dataclasses.dataclass
class StepResult:
    """One solve of one step of one path."""

    step: int
    theta: List[float]
    success: bool
    status: int
    status_msg: str
    iters: int
    solve_time: float
    obj: float
    kkt_error: float
    constr_viol: float
    # Evaluation counts, measured by the harness rather than reported
    # by the solver, so they mean the same thing for every solver.
    n_obj: int
    n_grad: int
    n_cons: int
    n_jac: int
    n_hess: int
    # Inner subproblem work on an active-set path: QP subproblems
    # solved and active-set changes (adds + drops) inside them. `None`
    # for a solver/path with no QP subproblems. On a QP-shaped family
    # the outer iteration count is 1 by construction and this is the
    # only measurement that responds to a warm start at all.
    n_qp_solves: Optional[int] = None
    n_qp_ws_changes: Optional[int] = None
    # Active-set descriptors, present only when the solver returned a
    # working set (i.e. the active-set path).
    n_active: Optional[int] = None
    ws_changed: Optional[int] = None  # Hamming distance from previous step
    # Correctness, filled in by the runner. `correct` requires the step
    # to have converged on its own terms (status, KKT residual, feasibility)
    # *and* to not have landed on a worse optimum than the reference arm.
    # `better` marks a step that found a strictly better objective than the
    # reference — on a nonconvex family that happens, and it is not an error.
    x_err: Optional[float] = None
    obj_err: Optional[float] = None
    converged: Optional[bool] = None
    better: Optional[bool] = None
    correct: Optional[bool] = None


class ParametricFamily(ABC):
    """A parameterized NLP together with a path through parameter space.

    Subclasses supply the usual cyipopt-shaped callbacks, but in
    *dense* form — :mod:`sparsity` derives the sparsity patterns and
    the packed value vectors from them, so a family author cannot get
    the structure/value correspondence out of sync. Sizes here are
    small enough (n ≲ 300) that dense assembly costs nothing next to
    the solve.

    The parameter enters through :meth:`set_theta`; everything else
    about the problem — shape, bounds, sparsity — must stay fixed
    along the path, since that is precisely the situation a warm start
    exploits.
    """

    #: Short identifier used on the command line and in the report.
    name: str = "unnamed"

    #: Free-form descriptors, rendered in the report so a reader can
    #: connect a result to the property that produced it. Conventional
    #: keys: ``regime`` (stable / flipping / degenerate / none),
    #: ``channel`` (objective / rhs / bounds / mixed),
    #: ``curvature`` (convex / nonconvex).
    tags: Dict[str, str] = {}

    #: Number of parameter steps in the path (the k=0 solve included).
    n_steps: int = 20

    #: True when every instance along this family's path is literally a
    #: convex QP — quadratic objective, linear constraints. Families that
    #: set this can additionally be routed to a dedicated convex QP
    #: solver (see :mod:`..qpform`); the suite's self-test verifies the
    #: claim by re-deriving the family from the extracted QP data rather
    #: than trusting the flag.
    quadratic: bool = False

    #: Size tier. ``"default"`` families are small and fast and make up
    #: the standard sweep; ``"large"`` families exist to exercise the
    #: sparse path at size and are opt-in (`--tier large`), because a
    #: single active-set solve on them can take seconds.
    tier: str = "default"

    #: Constraint rows through which θ enters as a pin equality
    #: ``g_i(x) = θ_i`` (``cl[i] == cu[i]``), in θ's own component order.
    #: Empty (the default) means θ does not enter that way, which is what
    #: a held-factor sensitivity step needs: `deltas` is a perturbation of
    #: exactly these rows' right-hand sides. Declaring it enables the
    #: predictor arms (pounce#608); the suite's self-test checks the claim
    #: by comparing the tangent step against a finite difference of two
    #: solves.
    pin_rows: tuple = ()

    #: True when the next parameter depends on the previous solution
    #: (closed-loop MPC / moving horizon). The runner then records the
    #: path produced by the reference arm and *replays* it for the
    #: others, so every arm sees an identical sequence.
    adaptive: bool = False

    # -- shape -----------------------------------------------------

    @property
    @abstractmethod
    def n(self) -> int:
        """Number of variables."""

    @property
    @abstractmethod
    def m(self) -> int:
        """Number of constraints."""

    @abstractmethod
    def bounds(self) -> Bounds:
        """Bounds at the *current* parameter (bounds may move with θ)."""

    @abstractmethod
    def cold_x0(self) -> np.ndarray:
        """The starting point a cold solve gets, independent of θ."""

    # -- parameter path --------------------------------------------

    @abstractmethod
    def set_theta(self, theta: np.ndarray) -> None:
        """Install parameter θ. Must not change n, m, or sparsity."""

    @abstractmethod
    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        """The scripted parameter path, or ``None`` when adaptive.

        ``scale`` multiplies the family's natural per-step increment.
        """

    def current_theta(self) -> Optional[np.ndarray]:
        """The parameter currently installed, or ``None`` if unknown.

        Families that keep it in ``self._theta`` (all of them today) get
        this for free. Needed by the predictor arms, which form Δθ from
        the previous step's value rather than from the runner's path --
        an adapter never sees the path.
        """
        theta = getattr(self, "_theta", None)
        return None if theta is None else np.asarray(theta, float).ravel().copy()

    def initial_theta(self, scale: float) -> np.ndarray:
        """First parameter of an adaptive path (adaptive families only)."""
        raise NotImplementedError

    def next_theta(self, x_solution: np.ndarray) -> np.ndarray:
        """Advance an adaptive path using the step's solution."""
        raise NotImplementedError

    # -- problem functions (dense) ---------------------------------

    @abstractmethod
    def objective(self, x: np.ndarray) -> float: ...

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def constraints(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def jacobian_dense(self, x: np.ndarray) -> np.ndarray:
        """Constraint Jacobian, shape (m, n).

        Families large enough that an ``(m, n)`` array is impractical
        override :meth:`sparse_structure` and the ``*_values`` methods
        instead; this one is then only called by the self-test, at a
        reduced size.
        """

    @abstractmethod
    def hessian_dense(
        self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float
    ) -> np.ndarray:
        """Hessian of the Lagrangian, full symmetric, shape (n, n).

        Sign convention matches cyipopt/Ipopt:
        ``obj_factor·∇²f + Σ_i lagrange_i·∇²g_i``.
        """

    # -- optional sparse path --------------------------------------
    #
    # A family whose dense Jacobian or Hessian would not fit — anything
    # past a few hundred variables — implements these three instead.
    # The structure is fixed for the whole path (the suite requires
    # that of every family anyway), so it is computed once.

    def sparse_structure(self):
        """``(jac_rows, jac_cols, hess_rows, hess_cols)`` or ``None``.

        ``None`` (the default) means "derive the pattern from the dense
        callbacks", which is right for every small family. Returning a
        structure switches :class:`~.sparsity.SparseCallbacks` onto the
        sparse path, where the dense methods are never called during a
        solve. The Hessian entries must be lower-triangular.
        """
        return None

    def jacobian_values(self, x: np.ndarray) -> np.ndarray:
        """Jacobian entries at ``sparse_structure()``'s ``jac`` indices."""
        raise NotImplementedError

    def hessian_values(
        self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float
    ) -> np.ndarray:
        """Hessian entries at ``sparse_structure()``'s ``hess`` indices."""
        raise NotImplementedError
