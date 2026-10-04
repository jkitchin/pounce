"""gh #589: a restoration failure must not raise on one route and return on
the other.

`Restoration_Failed` was missing from both status tables —
`sens._STATUS_RESULT` and `v2._V2_STATUS` — so it fell to their defaults. On
the legacy route that default is `(error, error)`: a results object the caller
can inspect. On the v2 route it was `(error, noSolution)`, and `noSolution` is
the one value that decides whether `solve()` raises — it makes `has_solution`
False, which turns the loader off and raises `NoSolutionError` under the
default `load_solutions=True`. So one failed solve returned under
`SolverFactory("pounce")` and raised under `SolverFactory("pounce_v2")`, for
the same model and options.

The exit was never *decided* to be a no-solution case — it was simply never
added, and eleven of the engine's twenty exits were in the same position. That
is what makes the bug worth a coverage test rather than one more entry: the
failure mode is silent. An unlisted exit produces no warning, no log line, and
no test failure; it just quietly changes what one route does.

So the tests below are in two layers:

* **Coverage** — every `ApplicationReturnStatus` in
  `crates/pounce-nlp/src/return_codes.rs` appears in both tables. The Rust enum
  is the source of truth (its `upstream_name()` is literally the `status_msg`
  string these tables key on), and it is read from the checkout when there is
  one, so adding a status there fails this test until both tables are updated.
* **Semantics** — no exit maps to `noSolution`, on any route, including the
  fallback for a status name that does not exist yet. The in-process route
  always has a primal vector (the engine returns `x` regardless of status), so
  `noSolution` is never the accurate answer, and route agreement does not have
  to be re-argued exit by exit.
"""

import os
import re

import pytest

import pyomo_pounce  # noqa: F401  (registers both interfaces)

from pyomo.opt.results.solver import (
    SolverStatus,
    TerminationCondition as LegacyTerminationCondition,
)

from pyomo_pounce.sens import _STATUS_RESULT


#: Every exit of the engine's `ApplicationReturnStatus`, spelled as
#: `upstream_name()` spells it — which is what `pounce.Solver.solve` reports
#: in `info["status_msg"]`, and therefore what both tables are keyed by.
#:
#: Pinned here rather than only derived from the Rust source so the test still
#: means something from an installed sdist, where `crates/` is not shipped.
#: `test_pinned_status_names_match_the_rust_enum` keeps the pin honest in a
#: checkout.
ENGINE_STATUSES = (
    "Solve_Succeeded",
    "Solved_To_Acceptable_Level",
    "Infeasible_Problem_Detected",
    "Search_Direction_Becomes_Too_Small",
    "Diverging_Iterates",
    "User_Requested_Stop",
    "Feasible_Point_Found",
    "Maximum_Iterations_Exceeded",
    "Restoration_Failed",
    "Error_In_Step_Computation",
    "Maximum_CpuTime_Exceeded",
    "Maximum_WallTime_Exceeded",
    "Not_Enough_Degrees_Of_Freedom",
    "Invalid_Problem_Definition",
    "Invalid_Option",
    "Invalid_Number_Detected",
    "Unrecoverable_Exception",
    "NonIpopt_Exception_Thrown",
    "Insufficient_Memory",
    "Internal_Error",
)

#: Statuses the Python front end reports that are not engine exits (so not in
#: `return_codes.rs`): `Callback_Error` (-198) is a raising `intermediate`
#: callback, which the engine sees only as a stop request (gh#986). They must
#: be in both tables too -- falling to the defaults is the gh #589 failure.
FRONTEND_STATUSES = ("Callback_Error",)

_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
_RETURN_CODES_RS = os.path.join(
    _REPO_ROOT, "crates", "pounce-nlp", "src", "return_codes.rs")

#: The `upstream_name` match arms, e.g.
#: `Self::RestorationFailed => "Restoration_Failed",`
_UPSTREAM_ARM = re.compile(r'Self::\w+\s*=>\s*"([A-Za-z_]+)"')


def _rust_status_names():
    """The status names `upstream_name()` returns, read from the Rust
    source, or None when this is not a checkout (an installed sdist ships
    no `crates/`)."""
    try:
        with open(_RETURN_CODES_RS, encoding="utf-8") as fh:
            src = fh.read()
    except OSError:
        return None
    # Only the `upstream_name` arm bodies; the file's tests repeat the names in
    # assertions, and `fn upstream_name` is the one place they are defined.
    start = src.find("fn upstream_name")
    if start < 0:
        return None
    end = src.find("#[cfg(test)]", start)
    body = src[start:end if end > 0 else len(src)]
    return tuple(_UPSTREAM_ARM.findall(body))


def test_pinned_status_names_match_the_rust_enum():
    """`ENGINE_STATUSES` is the contract the two tables are checked against, so
    it must not drift from the enum it claims to mirror. A status added to
    `return_codes.rs` fails here first, and the fix is to add it in all three
    places — which is the point."""
    names = _rust_status_names()
    if names is None:
        pytest.skip(f"not a checkout: {_RETURN_CODES_RS} is not readable")
    assert sorted(names) == sorted(ENGINE_STATUSES)


def test_sens_table_covers_every_engine_exit():
    missing = [s for s in ENGINE_STATUSES if s not in _STATUS_RESULT]
    assert not missing, (
        f"{missing} fall to the `_STATUS_RESULT` default, which reports a "
        f"less specific outcome than the `.sol` route gives the same solve "
        f"(gh #589)")


def test_sens_table_covers_every_frontend_status():
    missing = [s for s in FRONTEND_STATUSES if s not in _STATUS_RESULT]
    assert not missing


def test_a_raising_callback_is_an_error_not_an_interrupt_on_the_legacy_route():
    """gh#986: `Callback_Error` is the caller's code failing, not a deliberate
    `return False` -- it must not read as `userInterrupt` / `aborted`."""
    tc, status = _STATUS_RESULT["Callback_Error"]
    assert tc is LegacyTerminationCondition.error
    assert status is SolverStatus.error


def test_restoration_failure_is_a_solver_error_on_the_legacy_route():
    """The reported exit, on the route that already behaved acceptably: a
    results object, with the severity and condition the `.sol` route reports
    for AMPL's 500 failure band."""
    tc, status = _STATUS_RESULT["Restoration_Failed"]
    assert tc is LegacyTerminationCondition.internalSolverError
    assert status is SolverStatus.error


@pytest.mark.skipif(not pyomo_pounce.HAVE_V2_INTERFACE,
                    reason="the v2 interface needs Pyomo >= 6.10.1")
class TestV2Table:
    """The route that raised. Grouped so the whole class skips on an older
    Pyomo, where `pyomo_pounce.v2` cannot even be imported."""

    def test_v2_table_covers_every_engine_exit(self):
        from pyomo_pounce.v2 import _V2_STATUS

        missing = [s for s in ENGINE_STATUSES if s not in _V2_STATUS]
        assert not missing, (
            f"{missing} fall to the `_V2_STATUS` default; an exit that "
            f"defaults to `noSolution` raises `NoSolutionError` where the "
            f"legacy route returns a results object (gh #589)")

    def test_no_engine_exit_maps_to_no_solution(self):
        """The invariant that makes the two routes agree by construction.

        `noSolution` is not merely inaccurate here — it is the switch that
        decides whether `solve()` raises. The engine returns a primal vector
        for every exit, and `sens_solve` captures it before its non-converged
        early return, so there is always something to hand back.
        """
        from pyomo.contrib.solver.common.results import SolutionStatus
        from pyomo_pounce.v2 import _V2_STATUS

        offenders = [name for name, (_tc, ss) in _V2_STATUS.items()
                     if ss is SolutionStatus.noSolution]
        assert not offenders

    def test_v2_table_covers_every_frontend_status(self):
        """gh#986: `Callback_Error` (-198) reaches the table by name and
        reports an error with the iterate still loadable."""
        from pyomo.contrib.solver.common.results import (
            SolutionStatus,
            TerminationCondition,
        )
        from pyomo_pounce.v2 import _V2_STATUS

        missing = [s for s in FRONTEND_STATUSES if s not in _V2_STATUS]
        assert not missing
        tc, ss = _V2_STATUS["Callback_Error"]
        assert tc is TerminationCondition.error
        assert ss is SolutionStatus.unknown

    def test_restoration_failure_reports_a_loadable_iterate(self):
        """The fix for the reported symptom: a failed restoration is an error,
        and the point it stopped at is still handed over."""
        from pyomo.contrib.solver.common.results import (
            SolutionStatus,
            TerminationCondition,
        )
        from pyomo_pounce.v2 import _V2_STATUS

        tc, ss = _V2_STATUS["Restoration_Failed"]
        # `error` is what the ordinary `.sol` route reports for the 500 band.
        assert tc is TerminationCondition.error
        # ...and this is what stops `NoSolutionError`.
        assert ss is SolutionStatus.unknown

    def test_an_unrecognized_status_still_returns_its_iterate(self):
        """The default is the part of #589 a coverage test cannot fix on its
        own: a status name POUNCE does not have yet must not be able to flip
        the raise/return decision the moment it appears."""
        from pyomo.contrib.solver.common.results import (
            SolutionStatus,
            TerminationCondition,
        )
        from pyomo_pounce.v2 import _v2_status

        tc, ss = _v2_status("Some_Status_From_A_Later_POUNCE")
        assert tc is TerminationCondition.error
        assert ss is SolutionStatus.unknown

    def test_the_lookup_is_what_the_solve_path_uses(self):
        """Guard the indirection: `_v2_status` is only meaningful as a test
        seam if the solve path goes through it."""
        import inspect

        from pyomo_pounce.v2 import Pounce

        assert "_v2_status(" in inspect.getsource(Pounce._sens_solve)
