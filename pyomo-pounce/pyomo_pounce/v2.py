"""POUNCE on Pyomo's modern solver interface (``pyomo.contrib.solver``).

`pyomo_pounce.pounce_solver.POUNCE` drives POUNCE through Pyomo's
*legacy* interface (`pyomo.solvers.plugins.solvers.ASL.ASL`). This
module registers a second, independent interface for the same solver
against `pyomo.contrib.solver` -- Pyomo's newer solver API, the one
`ipopt_v2` uses -- so that the modern route is a supported way to reach
POUNCE rather than a generic `ipopt_v2`-pointed-at-our-binary hack
(gh #558).

Both interfaces stay registered, under separate names::

    import pyomo_pounce
    from pyomo.environ import SolverFactory
    SolverFactory('pounce')                     # legacy (unchanged)
    SolverFactory('pounce_v2')                  # v2 engine, legacy API

    from pyomo.contrib.solver.common.factory import SolverFactory as SF2
    SF2('pounce')                               # v2 interface

Why this exists rather than "just use `ipopt_v2`": pointing `ipopt_v2`
at the POUNCE binary does solve the model, but it silently drops
everything `pyomo-pounce` adds. This class carries all of it onto the v2
lifecycle:

* bundled-binary resolution, so a stale `pounce` on `PATH` is not picked
  up silently (gh #315);
* the guard that refuses a model with live integer variables rather than
  solving the continuous relaxation and calling a fractional value
  optimal (gh #341);
* `scaling_factor` Suffix handling (gh #483 / #486);
* the sensitivity path (`declare_sens_param` -> in-process
  `pounce.Solver`), translated onto the v2 `Results`/solution-loader
  contract rather than copied -- see :class:`PounceSensSolutionLoader`.

What is inherited from Pyomo's own `Ipopt` v2 class, deliberately: the
`.nl` write, the `.sol` read, option splitting between the command line
and the `.opt` file, and the solver-log parse. POUNCE is
ASL/Ipopt-compatible on every one of those surfaces -- it accepts
`key=value` options and `option_file_name=<path>`, and its log carries
the same ``Number of Iterations....:`` and ``Total seconds in POUNCE =``
lines the Ipopt parser looks for. What is *not* compatible, and is
overridden here, is the version banner: `pounce --version` prints
``pounce X.Y.Z``, which Pyomo's Ipopt parser rejects by design (it
requires a leading ``ipopt`` precisely so that other ASL executables are
not mistaken for Ipopt).
"""

from __future__ import annotations

import datetime
import logging
import re
import subprocess
import warnings
from timeit import default_timer
from typing import Any, Mapping, Sequence

from pyomo.common.collections import ComponentMap
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var

try:
    from pyomo.contrib.solver.common.base import LegacySolverWrapper
    from pyomo.contrib.solver.common.factory import (
        SolverFactory as ContribSolverFactory,
    )
    from pyomo.contrib.solver.common.results import (
        Results,
        SolutionStatus,
        TerminationCondition,
    )
    from pyomo.contrib.solver.common.solution_loader import SolutionLoader
    from pyomo.contrib.solver.common.util import (
        NoOptimalSolutionError,
        NoSolutionError,
    )
    from pyomo.contrib.solver.solvers.ipopt import Ipopt, IpoptConfig
except ImportError as exc:  # pragma: no cover - depends on the Pyomo version
    # The floor is **6.10.1**, and the discriminator is `SolutionLoader`.
    #
    # `pyomo.contrib.solver.common` as a package landed in 6.9.2, which
    # makes it a tempting thing to probe -- and the wrong one. Through
    # 6.10.0 that package still shipped the older loader API
    # (`SolutionLoaderBase`, `load_vars`, `get_primals`); `SolutionLoader`
    # with `load_solution`/`get_vars`, which this module subclasses and
    # calls, is 6.10.1. So 6.9.2 through 6.10.0 have the package and not
    # the API, and a package-level probe passes there and then explodes
    # here.
    #
    # `pyomo_pounce` supports pyomo>=6.0 through the legacy plugin, so
    # this is raised with a clear message rather than papered over, and
    # `pyomo_pounce/__init__.py` guards on the same name so that an old
    # Pyomo means "v2 unavailable" and never a failed `import
    # pyomo_pounce`.
    raise ImportError(
        "pyomo_pounce.v2 needs Pyomo's `pyomo.contrib.solver` interface "
        "as it exists in Pyomo 6.10.1 or newer (6.9.2-6.10.0 ship the "
        "package but the older SolutionLoaderBase/get_primals API). The "
        "legacy SolverFactory('pounce') plugin works on any supported "
        "Pyomo; upgrade Pyomo to use the v2 interface."
    ) from exc

from pyomo_pounce.pounce_solver import (
    _bundled_path,
    _checkout_path,
    _warn_checkout_fallback,
    _warn_path_fallback,
    reject_discrete_vars,
)

logger = logging.getLogger(__name__)

__all__ = ["Pounce", "PounceConfig", "LegacyPounceSolver",
           "PounceSensSolutionLoader"]


def _default_executable():
    """Default for the `executable` config: the wheel-bundled binary when one
    is installed, else the surrounding source checkout's cargo build, else the
    bare name for a `PATH` lookup.

    Same precedence as the legacy plugin's `_default_executable`, and for the
    same reason -- the bundled path is deterministic while `PATH` can hand
    back a stale build that reports an identical version string (gh #315).
    The checkout rung is there for the same reason it is there on the legacy
    route: a `maturin develop` install bundles nothing, and the `pounce` this
    would otherwise find on PATH is that install's own console-script shim
    (gh #816). Keeping the two routes in step is not cosmetic -- gh #558 is a
    guard that covered the legacy interface only, and left the modern one with
    exactly the silent wrongness it existed to prevent.

    Resolved once, when the CONFIG is built at import: the bundled binary's
    location is fixed at install time, and so is the checkout's.
    """
    bundled = _bundled_path()
    if bundled is not None:
        return bundled
    checkout = _checkout_path()
    if checkout is not None:
        return checkout
    return "pounce"


class PounceConfig(IpoptConfig):
    """`IpoptConfig` with the executable defaulted to POUNCE's binary.

    Everything else -- `writer_config`, `solver_options`, `tee`,
    `time_limit`, ... -- is Pyomo's, unchanged, because POUNCE takes the
    same `.nl` input and the same `key=value` / `.opt` option forms.
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        exe = self.get("executable")
        exe.set_default_value(_default_executable())
        exe.reset()
        exe._description = (
            "Preferred executable for pounce. Defaults to the `pounce` "
            "binary bundled in the installed `pounce-solver` wheel, "
            "falling back to searching the ``PATH`` for the first "
            "available ``pounce``."
        )


#: `pounce --version` prints ``pounce X.Y.Z``. Anchored, and requiring the
#: program name, for the same reason Pyomo's Ipopt parser requires
#: ``ipopt``: so that some *other* ASL executable handed to `executable=`
#: is reported as "not found" rather than silently driven as POUNCE.
_VERSION_RE = re.compile(r"^\s*pounce\s+(\d+(?:\.\d+)*)", re.IGNORECASE)


#: POUNCE exit status (the engine's `status_msg`) -> the v2 pair
#: (TerminationCondition, SolutionStatus). The legacy sensitivity path
#: maps the same statuses onto the legacy `TerminationCondition` /
#: `SolverStatus` pair in `sens._STATUS_RESULT`; the two enums are
#: different sets with different members, so this is a translation of
#: the same table, not an alias of it.
#:
#: The **solution status** column deliberately agrees with what the
#: ordinary (`.sol`) route produces, because disagreeing changes whether
#: a solve raises. Pyomo derives it in
#: `asl_solve_code_to_solution_status` as
#:
#:     status = SolutionStatus.unknown if sol_data.primals else noSolution
#:
#: overridden only for the optimal / feasible / infeasible code bands.
#: So "a primal vector came back" is the rule, and the limit and
#: divergence cases keep `unknown`. The in-process route always has a
#: primal iterate -- the engine returns `x` regardless of status -- so
#: they map to `unknown` here too. Mapping them to `noSolution` (as this
#: table first did) made `solve(m, solver_options={'max_iter': 1},
#: raise_exception_on_nonoptimal_result=False)` load the final iterate on
#: the ordinary route and raise `NoSolutionError` on this one, for the
#: same model and options; `noSolution` was also simply the less accurate
#: answer, since POUNCE does return a usable iterate in both cases.
#:
#: The **termination condition** column does NOT follow the ordinary
#: route, on purpose: Pyomo's table reads the AMPL solve-code *band* and
#: cannot tell a time limit from an iteration limit (its source carries
#: `# this is not always correct` on that line), while POUNCE names the
#: status exactly. Reporting `maxTimeLimit` for a time limit is strictly
#: more informative, so that difference is kept.
#:
#: The table is **exhaustive** over the engine's exits -- all twenty of
#: `ApplicationReturnStatus` (`crates/pounce-nlp/src/return_codes.rs`),
#: whose `upstream_name()` is exactly the `status_msg` this route reads.
#: It was not, and the gap was invisible: an unlisted exit fell to the
#: default, and a `noSolution` default is the one value that changes
#: whether `solve` raises. `Restoration_Failed` -- an ordinary numerical
#: exit that stops at an iterate -- raised `NoSolutionError` here while
#: the legacy route returned a results object for the very same solve
#: (gh #589). `test_issue_589_status_table_coverage.py` holds both
#: tables to the full enum so a new engine exit cannot repeat it.
_V2_STATUS = {
    "Solve_Succeeded": (
        TerminationCondition.convergenceCriteriaSatisfied,
        SolutionStatus.optimal,
    ),
    "Solved_To_Acceptable_Level": (
        TerminationCondition.convergenceCriteriaSatisfied,
        SolutionStatus.optimal,
    ),
    # A square problem solved to feasibility. This is a success, and the
    # row must say so: POUNCE emits this status only when
    # `is_square_problem` holds (`resto_inner_solver.rs`, the status's one
    # gate), which is Ipopt's own condition, and on a square problem the
    # objective is constant -- a feasible point is the solution. The row
    # used to read `(unknown, feasible)` on the theory that POUNCE used the
    # status more loosely than Ipopt does; it does not.
    #
    # These values are exactly what the `.sol` route now produces for the
    # same solve: POUNCE writes AMPL code 2 (Ipopt's own), and Pyomo's v2
    # reader maps the 0..99 band to
    # `(convergenceCriteriaSatisfied, optimal)`. Keeping the two routes in
    # agreement is the same rule that governs `Solved_To_Acceptable_Level`
    # above (gh #591); disagreeing here is what gh #815 was -- an IDAES
    # square flowsheet solved to a 2.2e-06 constraint violation and
    # reported to the caller as a failure.
    "Feasible_Point_Found": (
        TerminationCondition.convergenceCriteriaSatisfied,
        SolutionStatus.optimal,
    ),
    "Infeasible_Problem_Detected": (
        TerminationCondition.locallyInfeasible,
        SolutionStatus.infeasible,
    ),
    "Diverging_Iterates": (
        TerminationCondition.unbounded,
        SolutionStatus.unknown,
    ),
    "Maximum_Iterations_Exceeded": (
        TerminationCondition.iterationLimit,
        SolutionStatus.unknown,
    ),
    "Maximum_CpuTime_Exceeded": (
        TerminationCondition.maxTimeLimit,
        SolutionStatus.unknown,
    ),
    "Maximum_WallTime_Exceeded": (
        TerminationCondition.maxTimeLimit,
        SolutionStatus.unknown,
    ),
    "User_Requested_Stop": (
        TerminationCondition.interrupted,
        SolutionStatus.unknown,
    ),
    # The step-length exit AMPL puts in the same 400 "limit" band as the
    # iteration and time limits; the v2 enum names it exactly.
    "Search_Direction_Becomes_Too_Small": (
        TerminationCondition.minStepLength,
        SolutionStatus.unknown,
    ),
    # ---- the failure band (AMPL 500..599) --------------------------------
    #
    # `error` is what the ordinary route reports for every one of these
    # (`asl_solve_code_to_solution_status` maps the whole 500 band to
    # `TerminationCondition.error`), and the v2 enum has no finer member
    # to reach for -- unlike the legacy table, which distinguishes
    # `internalSolverError` from `invalidProblem`.
    #
    # `unknown`, not `noSolution`, for the same reason the limit cases
    # take it: a primal vector came back. These exits stop the algorithm
    # at an iterate and hand it to `finalize_solution`, so the solve has
    # a point to report even though it failed -- `sens_solve` captures it
    # before its non-converged early return, and the legacy route loads
    # it onto the model. `Restoration_Failed` is the exit that surfaced
    # this (gh #589); the others are its neighbours on the same path.
    "Restoration_Failed": (
        TerminationCondition.error,
        SolutionStatus.unknown,
    ),
    "Error_In_Step_Computation": (
        TerminationCondition.error,
        SolutionStatus.unknown,
    ),
    "Invalid_Number_Detected": (
        TerminationCondition.error,
        SolutionStatus.unknown,
    ),
    "Insufficient_Memory": (
        TerminationCondition.error,
        SolutionStatus.unknown,
    ),
    # ---- refused before the algorithm ran --------------------------------
    #
    # These return from `IpoptApplication::optimize_tnlp` ahead of the
    # iteration loop, so nothing ever reached `finalize_solution` and the
    # engine's `x` is the zero vector it was initialized with, not an
    # iterate. `unknown` all the same, and deliberately: the ordinary
    # route reports exactly that, because the CLI writes a `.sol` for a
    # refused solve too -- `crates/pounce-cli/src/main.rs` writes it
    # "unconditionally once a target path is resolved, even on a failed
    # solve", falling back to zero blocks sized from the pre-solve NLP
    # dimensions so the primal block still aligns with the `.nl` -- and
    # Pyomo's rule is "a primal vector came back".
    # Claiming `noSolution` here would buy nothing -- `sens_solve` has
    # already written those values onto the model by the time this route
    # inspects the status, so raising `NoSolutionError` would not spare
    # the caller the zeros, only the results object describing them --
    # and would cost the route agreement this table exists to keep. The
    # default `raise_exception_on_nonoptimal_result=True` still raises
    # `NoOptimalSolutionError` on every status in this block.
    "Not_Enough_Degrees_Of_Freedom": (
        TerminationCondition.error,
        SolutionStatus.unknown,
    ),
    "Invalid_Problem_Definition": (
        TerminationCondition.error,
        SolutionStatus.unknown,
    ),
    "Invalid_Option": (
        TerminationCondition.error,
        SolutionStatus.unknown,
    ),
    "Internal_Error": (
        TerminationCondition.error,
        SolutionStatus.unknown,
    ),
    # Present for ABI parity with upstream Ipopt's enum; POUNCE itself
    # never returns either (nothing constructs them outside
    # `return_codes.rs`). Mapped so the table stays exhaustive over the
    # enum rather than over what today's engine happens to emit.
    "Unrecoverable_Exception": (
        TerminationCondition.error,
        SolutionStatus.unknown,
    ),
    "NonIpopt_Exception_Thrown": (
        TerminationCondition.error,
        SolutionStatus.unknown,
    ),
}


def _v2_status(status_msg):
    """`(TerminationCondition, SolutionStatus)` for an engine exit.

    The fallback is `unknown`, not `noSolution`. `_V2_STATUS` covers every
    exit the engine has, so it fires only for a status name POUNCE does not
    have yet -- and a *new* exit will come back with a primal vector like
    every existing one does, since the engine returns `x` unconditionally.
    `noSolution` here would make the next added status raise
    `NoSolutionError` on this route while the legacy one returned the
    iterate, which is exactly the asymmetry gh #589 reported. The coverage
    test keeps the *termination condition* honest; this keeps the
    raise-or-return decision from ever turning on an oversight.
    """
    return _V2_STATUS.get(
        status_msg, (TerminationCondition.error, SolutionStatus.unknown))


class PounceSensSolutionLoader(SolutionLoader):
    """v2 solution loader over an in-process sensitivity solve.

    The sensitivity path never writes a `.sol`: it hands POUNCE
    evaluator callbacks built from `pounce.read_nl` and reads the
    converged primal/dual vectors straight out of the engine. So the
    ASL-backed loader the ordinary v2 route uses has nothing to read,
    and this stands in for it -- the "deliberate translation" the v2
    lifecycle needs, since v2 returns values *through* the loader rather
    than loading them as a side effect of `solve` the way the legacy
    `load_from` path does.

    Two sign conventions are crossed on the way out, mirroring
    `sens._warm_start_from_suffixes` on the way in:

    * `dual` is the AMPL marginal ``d obj / d b = -lambda`` (gh #271),
      while the engine reports the internal ``+lambda`` in
      ``info['mult_g']`` -- so duals negate.
    * `ipopt_zU_out` is negative at an active upper bound (Ipopt's
      convention, gh #296) while the engine's ``mult_x_U`` is the
      internal non-negative ``z_u`` -- so it negates too; ``zL`` is
      positive in both. The reduced cost is then combined from the two
      bound multipliers exactly as `IpoptSolutionLoader` combines the
      `.sol` suffixes, so `rc` means the same thing on both routes.

    A third crosses on a `maximize` model and multiplies both. A
    multiplier is a coefficient of the objective it was generated
    against, and `pounce.read_nl` negates a maximization before the
    engine ever sees it, so ``mult_g`` and ``mult_x_*`` are stated
    against ``-f``. ``capture['obj_sign']`` is the conversion back to
    the objective the caller wrote; it is +1 on every minimization,
    which is why its absence was invisible until a maximization asked.
    """

    def __init__(self, model, capture, has_solution=True):
        self._pyomo_model = model
        self._capture = capture
        self._has_solution = has_solution
        self._var_row = {n: i for i, n in enumerate(capture["var_names"])}
        self._con_row = {n: i for i, n in enumerate(capture["con_names"])}
        self._con_alias = capture.get("con_alias") or {}
        # +1 minimize / -1 maximize; a capture from before this key
        # existed can only have come from a minimization
        self._obj_sign = float(capture.get("obj_sign", 1.0))
        self._warned_unresolved = False

    def get_number_of_solutions(self) -> int:
        return 1 if self._has_solution else 0

    def get_solution_ids(self) -> list:
        # The base implementation would do this too, but only by way of
        # `get_number_of_solutions`; spelling it out keeps `solution()`
        # from raising the base class's NotImplementedError.
        return [None] if self._has_solution else []

    def _warn_unresolved(self, kind, names):
        """Warn about solve-space names that do not resolve on the model.

        Not a plain count check against `capture['n']`: on a model with
        `declare_sens_param` declarations the solve runs on a
        *surgery clone*, whose `.col`/`.row` files legitimately carry
        components that exist only there -- the pinned parameter becomes
        `_SENSITIVITY_TOOLBOX_DATA.p`, and the pin itself becomes a
        constraint row. Those are expected to be unresolvable on the
        original model (`sens_solve`'s own load-back skips them the same
        way), so counting would warn on every sensitivity model, which is
        precisely the case this route exists for. Only names outside the
        surgery block indicate a real mismatch -- one that would
        otherwise leave a variable silently stale after `load_vars`.
        """
        if self._warned_unresolved:
            return
        try:
            from pyomo.contrib.sensitivity_toolbox.sens import (
                SensitivityInterface,
            )
            prefix = SensitivityInterface.get_default_block_name() + "."
        except Exception:  # noqa: BLE001 - a broken probe must not mislead
            prefix = "_SENSITIVITY_TOOLBOX_DATA."
        real = [n for n in names if not n.startswith(prefix)]
        if not real:
            return
        self._warned_unresolved = True
        shown = ", ".join(real[:5])
        more = f" (+{len(real) - 5} more)" if len(real) > 5 else ""
        warnings.warn(
            f"pounce: {len(real)} {kind} from the solve could not be "
            f"resolved on the model (e.g. {shown}{more}); their values "
            f"are missing from this solution and will not be loaded.",
            UserWarning, stacklevel=3)

    def _require_solution(self):
        if not self._has_solution:
            raise NoSolutionError()

    def _vector(self, key):
        vec = self._capture.get("info", {}).get(key)
        return None if vec is None else vec

    def get_vars(
        self, vars_to_load: Sequence[Any] | None = None
    ) -> Mapping[Any, float]:
        self._require_solution()
        x = self._capture["x"]
        out = ComponentMap()
        if vars_to_load is None:
            missing = []
            for name, val in zip(self._capture["var_names"], x):
                vd = self._pyomo_model.find_component(name)
                if vd is None:
                    missing.append(name)
                else:
                    out[vd] = float(val)
            if missing:
                self._warn_unresolved("variables", missing)
            return out
        for vd in vars_to_load:
            row = self._var_row.get(vd.name)
            if row is not None:
                out[vd] = float(x[row])
        return out

    def _row_of(self, con_data):
        """The solve's row index for a constraint of the *original* model.

        A constraint the declared-parameter surgery replaced lives in the
        solve under its clone's name, so it is reached through the alias
        map -- the same indirection the warm-start reader applies.
        """
        name = con_data.name
        return self._con_row.get(self._con_alias.get(name, name))

    def get_duals(
        self, cons_to_load: Sequence[Any] | None = None
    ) -> dict:
        self._require_solution()
        lam = self._vector("mult_g")
        if lam is None:
            raise NoSolutionError(
                "pounce: this solve returned no constraint multipliers, so "
                "duals are not available")
        report_missing = cons_to_load is None
        if cons_to_load is None:
            cons_to_load = self._pyomo_model.component_data_objects(
                Constraint, active=True, descend_into=True)
        out, missing = {}, []
        for cd in cons_to_load:
            row = self._row_of(cd)
            if row is None:
                # A model constraint with no row in the solve. This
                # direction is the reverse of the variable one -- we are
                # walking the model, not the clone -- so a surgery
                # artifact cannot explain it away and it is always worth
                # reporting.
                missing.append(cd.name)
            else:
                # engine's internal +lambda, against the objective it
                # minimized -> the AMPL marginal Pyomo's `dual` suffix
                # carries, against the objective the model states
                out[cd] = -self._obj_sign * float(lam[row])
        if missing and report_missing:
            self._warn_unresolved("constraints", missing)
        return out

    def get_reduced_costs(
        self, vars_to_load: Sequence[Any] | None = None
    ) -> Mapping[Any, float]:
        self._require_solution()
        zl = self._vector("mult_x_L")
        zu = self._vector("mult_x_U")
        if zl is None or zu is None:
            raise NoSolutionError(
                "pounce: this solve returned no bound multipliers, so "
                "reduced costs are not available")
        if vars_to_load is None:
            vars_to_load, missing = [], []
            for n in self._capture["var_names"]:
                vd = self._pyomo_model.find_component(n)
                (vars_to_load if vd is not None else missing).append(
                    vd if vd is not None else n)
            if missing:
                self._warn_unresolved("variables", missing)
        out = ComponentMap()
        for vd in vars_to_load:
            row = self._var_row.get(vd.name)
            if row is None:
                continue
            lo = self._obj_sign * float(zl[row])
            # Ipopt's `ipopt_zU_out` convention, so that the combination
            # below is the same arithmetic IpoptSolutionLoader does
            hi = -self._obj_sign * float(zu[row])
            out[vd] = hi if abs(hi) > abs(lo) else lo
        return out


class Pounce(Ipopt):
    """Interface to the POUNCE NLP solver (NL file based)."""

    CONFIG = PounceConfig()

    #: Availability/version cache. Redeclared rather than inherited: the
    #: cache is keyed by executable path only, and `Pounce` and `Ipopt`
    #: parse the version banner differently -- sharing one dict would let
    #: an `Ipopt` probe of some path poison the answer here (and vice
    #: versa) for the same path.
    _exe_cache: dict = {}

    def _get_version(self, exe):
        """POUNCE's version, from ``pounce --version`` (``pounce X.Y.Z``).

        Pyomo's `Ipopt._get_version` demands a banner starting `ipopt`,
        so POUNCE reads as "not found" through it -- which is why this
        override exists, and why it is just as strict about the program
        name: an executable that does not announce itself as `pounce`
        must not be driven as POUNCE.
        """
        try:
            return self._exe_cache[exe]
        except KeyError:
            pass
        if exe is None:
            self._exe_cache[None] = None
            return None
        try:
            results = subprocess.run(
                [str(exe), "--version"],
                timeout=self._version_timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            self._exe_cache[exe] = None
            return None
        ver = None
        if not results.returncode:
            m = _VERSION_RE.match(results.stdout)
            if m:
                try:
                    ver = tuple(int(i) for i in m.group(1).split("."))
                except ValueError:
                    ver = None
        if ver is None:
            logger.warning(
                f"Failed parsing POUNCE version: '{exe} --version':"
                f"\n\n{results.stdout}")
        self._exe_cache[exe] = ver
        return ver

    def _check_executable(self, config):
        """Warn once if this solve will run a `PATH` binary rather than the
        wheel-bundled one -- the case that silently ran a stale,
        dual-sign-flipped build before gh #315.

        Only for the *default* executable: an explicitly passed
        `executable=` is the user naming a binary on purpose, and warning
        about it would be noise. Read off the per-solve `config`, not
        `self.config`, so that a call-time `solve(model, executable=...)`
        counts as explicit too -- the flag propagates into the derived
        config, an instance-level one does not propagate back out.
        """
        if _bundled_path() is not None:
            return
        # `_userSet` is Pyomo-internal; via getattr so that a rename
        # downgrades this to a spurious warning rather than breaking
        # every solve.
        if getattr(config.get("executable"), "_userSet", False):
            return
        path = config.executable.path()
        if path is None:
            return
        # Two different fallbacks, two different warnings: the checkout's own
        # cargo build is at least provably this tree's, a PATH binary is not.
        checkout = _checkout_path()
        if checkout is not None and path == checkout:
            _warn_checkout_fallback(path)
        else:
            _warn_path_fallback(path)

    def solve(self, model, **kwds) -> Results:
        """Solve `model` with POUNCE, returning a v2 `Results`.

        Beyond `Ipopt.solve`: the integer-variable guard, the
        `scaling_factor` Suffix check, and the in-process sensitivity
        route when the model carries `declare_sens_param` (or the
        equivalent call-time keywords) declarations.
        """
        from pyomo_pounce.scaling import (
            user_scaling_requested,
            warn_if_no_suffix,
        )
        from pyomo_pounce.sens import has_declarations

        explicit = {k: kwds.pop(k) for k in
                    ("sens_params", "fitted", "residuals") if k in kwds}

        reject_discrete_vars(model)

        # Derived here to read the executable and the merged solver
        # options before deciding anything. On the ordinary route
        # `super().solve()` derives its own from the same `kwds` and this
        # one is discarded -- so the `_check_executable` warning below is
        # computed off an object that is then thrown away. That is safe
        # because deriving does not mutate `self.config` and the two
        # derivations are identical, and it is cheap next to a solve; the
        # alternative (reaching into `Ipopt.solve` to reuse ours) would
        # mean copying its body.
        config: PounceConfig = self.config(value=kwds, preserve_implicit=True)
        self._check_executable(config)

        # Solver options as this solve will actually see them. Deriving
        # `config` from `kwds` above already layered them the way
        # `Ipopt.solve` does -- instance-level `solver.config.
        # solver_options[...]` merged with per-call `solver_options=`,
        # the per-call value winning -- so this reads the merged view
        # rather than re-implementing the precedence. Reading only the
        # per-call half here is what silently un-tuned every model the
        # day it gained a declaration on the legacy side (gh #432).
        opts = dict(config.solver_options.value() or {})
        if user_scaling_requested(opts):
            warn_if_no_suffix(model)

        if has_declarations(model) or explicit:
            return self._sens_solve(model, config, opts, explicit)
        return super().solve(model, **kwds)

    def _sens_solve(self, model, config, opts, explicit) -> Results:
        """The sensitivity route, translated onto the v2 contract.

        `sens.sens_solve` solves in-process so that the converged KKT
        factorization stays available for `sens_jacobian()` /
        `sens_solution()` / `sens_covariance()`, and returns a *legacy*
        SolverResults. The v2
        contract is a different object with a different status enum, and
        -- the substantive difference -- it hands the solution back
        through a solution loader that the caller may decline to load
        (`load_solutions=False`) or read without loading. So the raw
        solve is captured and re-expressed here rather than the legacy
        results object being adapted.
        """
        from pyomo_pounce.sens import sens_solve

        start_time = default_timer()
        results = Results()
        results.timing_info.start_timestamp = datetime.datetime.now(
            datetime.timezone.utc)
        results.solver_name = self.name
        results.solver_version = self._get_version(config.executable.path())

        opts = dict(opts)
        if config.time_limit is not None:
            # same mapping `_run_ipopt` applies for the ordinary route
            opts.setdefault("max_cpu_time", config.time_limit)

        # `sens_solve` writes the converged iterate onto the model's
        # variables itself -- the legacy contract, where loading IS the
        # solve's side effect. v2 promises the opposite: with
        # `load_solutions=False` the model must come back untouched and
        # the values are read through the loader. So snapshot first and
        # roll back after; the loader serves the values either way, and
        # the retained KKT session is unaffected because it answers from
        # its own `base_x`, not from the model.
        restore = None
        if not config.load_solutions:
            restore = [(vd, vd.value, vd.stale) for vd in
                       model.component_data_objects(
                           Var, active=True, descend_into=True)]

        capture = {}
        try:
            sens_solve(model, tee=bool(config.tee), options=opts,
                       capture=capture, **explicit)
        finally:
            if restore is not None:
                for vd, val, stale in restore:
                    vd.set_value(val, skip_validation=True)
                    vd.stale = stale

        tc, ss = _v2_status(capture.get("status_msg", ""))
        results.termination_condition = tc
        results.solution_status = ss

        info = capture.get("info", {})
        iters = info.get("iter_count")
        results.extra_info.iteration_count = (
            None if iters is None else int(iters))
        results.extra_info.solver_message = capture.get("status_msg")
        # Same key the ordinary route ends up with: Pyomo's log parser
        # turns POUNCE's `Total seconds in POUNCE = …` summary line into
        # `timing_info.POUNCE`, so a caller reading the solver's own time
        # finds it under one name on both routes. The quantity differs
        # slightly -- there it is POUNCE's self-reported time, here it is
        # measured around the in-process call (the tee stream/decode
        # excluded) because no log is parsed.
        results.timing_info.POUNCE = capture.get("solve_secs")

        # Kept as a derived test rather than a literal `True`, even though
        # `_v2_status` no longer returns `noSolution` for anything (gh #589):
        # this is the same shape `Ipopt.solve` has, `has_solution` is part of
        # the loader's constructor contract, and reading it off the status
        # keeps the two in step if that ever changes. The `NoSolutionError`
        # branch below is unreachable today for the same reason.
        has_solution = ss is not SolutionStatus.noSolution
        results.solution_loader = PounceSensSolutionLoader(
            model, capture, has_solution=has_solution)

        if (config.raise_exception_on_nonoptimal_result
                and ss is not SolutionStatus.optimal):
            raise NoOptimalSolutionError()

        if config.load_solutions:
            if not has_solution:
                raise NoSolutionError()
            results.solution_loader.load_solution()

        # Gated exactly as `Ipopt.solve` gates it -- on {feasible,
        # optimal}, not on "a solution exists". The looser test reported
        # an objective for an infeasible or limit-stopped solve where the
        # ordinary route leaves `incumbent_objective` as None, which is a
        # route disagreement in the direction of over-claiming: the
        # number is the objective at whatever iterate the solve stopped
        # on, not at a solution.
        if ss in (SolutionStatus.feasible, SolutionStatus.optimal):
            obj = info.get("obj_val")
            if obj is not None:
                results.incumbent_objective = float(obj)

        results.solver_config = config
        results.timing_info.wall_time = default_timer() - start_time
        return results


class LegacyPounceSolver(LegacySolverWrapper, Pounce):
    """`SolverFactory('pounce_v2')`: the v2 engine behind the legacy API."""


# Registered explicitly (not as a decorator) so the legacy wrapper above
# is the one installed, rather than the anonymous subclass the factory
# would synthesize. `legacy_name` is what keeps this additive: the v2
# factory gets `pounce`, the legacy factory gets `pounce_v2`, and the
# legacy factory's existing `pounce` -> `pounce_solver.POUNCE`
# registration is untouched. Same split Pyomo itself uses for
# `ipopt` / `ipopt_v2`.
ContribSolverFactory.register(
    name="pounce",
    legacy_name="pounce_v2",
    doc="The POUNCE interior-point NLP solver",
)(Pounce, LegacyPounceSolver)
