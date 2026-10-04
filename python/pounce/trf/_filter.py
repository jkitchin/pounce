"""The (theta, f) filter used to accept or reject trust-region steps.

A filter replaces the merit function of a classical trust-region method. Rather
than collapsing infeasibility and objective into one scalar with a penalty
parameter that has to be tuned, it keeps a Pareto front of ``(theta, f)`` pairs
and accepts a trial step when it is not dominated by anything already on the
front. This is Fletcher & Leyffer's idea (Math. Program. 91:239, 2002), adapted
to the trust-region-filter setting by Eason & Biegler.

Here ``theta`` is *not* general constraint violation. Every trust-region
subproblem is solved subject to the glass-box constraints, so those hold at
every iterate by construction. The only infeasibility that can survive is the
mismatch between the surrogate's prediction and the truth model::

    theta(x) = || y - d(w) ||

which is exactly the quantity the surrogate is responsible for. Driving it to
zero is what makes the method converge to a KKT point of the *truth* model
rather than of the surrogate.

References
----------
Fletcher, R. & Leyffer, S. Nonlinear programming without a penalty function.
    Math. Program. 91, 239-269 (2002).
Eason, J.P. & Biegler, L.T. A trust region filter method for glass box/black
    box optimization. AIChE J. 62, 3124-3136 (2016).
Eason, J.P. & Biegler, L.T. Advanced trust region optimization strategies for
    glass box/black box models. AIChE J. 64, 3934-3943 (2018).  [Eq. 12]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class FilterPoint:
    """One ``(infeasibility, objective)`` pair on the filter front.

    Parameters
    ----------
    theta:
        Infeasibility measure ``||y - d(w)||`` at the point. Non-negative.
    f:
        Objective value at the point.
    """

    theta: float
    f: float

    def dominates(self, other: "FilterPoint") -> bool:
        """True if ``self`` is at least as good as ``other`` in both measures."""
        return self.theta <= other.theta and self.f <= other.f


class TRFilter:
    """A trust-region filter over ``(theta, f)`` pairs.

    Parameters
    ----------
    gamma_theta, gamma_f:
        Envelope parameters in ``(0, 1)``. A trial point must beat every filter
        entry by a margin proportional to that entry's infeasibility, which is
        what stops the iterates converging to a feasible but suboptimal point.
        Eason & Biegler (2018) Eq. 12: the step is acceptable if, for all
        ``(theta_j, f_j)`` in the filter,

            ``theta_trial <= (1 - gamma_theta) * theta_j``  or
            ``f_trial     <= f_j - gamma_f * theta_j``

        Setting both to ``0.0`` reduces the test to plain Pareto dominance,
        which is what ``pyomo.contrib.trustregion`` implements; that is useful
        for cross-checking but is not the algorithm as published.
    theta_max:
        Hard cap on infeasibility. A trial point with ``theta > theta_max`` is
        rejected outright regardless of its objective.

    Notes
    -----
    The filter is a *front*, not a log: adding a point prunes every entry it
    dominates, and a point that is itself dominated is not stored. So the size
    stays proportional to the number of genuinely non-dominated trade-offs seen,
    not to the iteration count.
    """

    def __init__(
        self,
        gamma_theta: float = 1e-5,
        gamma_f: float = 1e-5,
        theta_max: float = math.inf,
    ) -> None:
        if not (0.0 <= gamma_theta < 1.0):
            raise ValueError(f"gamma_theta must be in [0, 1), got {gamma_theta}")
        if not (0.0 <= gamma_f < 1.0):
            raise ValueError(f"gamma_f must be in [0, 1), got {gamma_f}")
        if theta_max <= 0.0:
            raise ValueError(f"theta_max must be positive, got {theta_max}")

        self.gamma_theta = gamma_theta
        self.gamma_f = gamma_f
        self.theta_max = theta_max
        self._points: list[FilterPoint] = []

    def is_acceptable(
        self, point: FilterPoint, current: FilterPoint | None = None
    ) -> bool:
        """Whether ``point`` may be accepted as the next iterate.

        ``current`` is the iterate the step starts from. Fletcher-Leyffer and
        Eason-Biegler test the trial against the filter *and* the current
        point; f-steps never store the current point, and the filter starts
        empty, so without it a trial that raises theta tenfold is accepted.
        """
        if point.theta > self.theta_max:
            return False
        entries = self._points if current is None else [*self._points, current]
        for entry in entries:
            beats_theta = point.theta <= (1.0 - self.gamma_theta) * entry.theta
            beats_f = point.f <= entry.f - self.gamma_f * entry.theta
            if not (beats_theta or beats_f):
                return False
        return True

    def add(self, point: FilterPoint) -> bool:
        """Add ``point`` to the front, pruning entries it dominates.

        Returns ``True`` if the point was stored, ``False`` if an existing entry
        already dominated it (in which case the filter is unchanged).
        """
        for entry in self._points:
            if entry.dominates(point):
                return False
        self._points = [e for e in self._points if not point.dominates(e)]
        self._points.append(point)
        return True

    def clear(self) -> None:
        self._points.clear()

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self) -> Iterator[FilterPoint]:
        return iter(self._points)

    def __repr__(self) -> str:
        return f"TRFilter({len(self._points)} points, theta_max={self.theta_max:g})"
