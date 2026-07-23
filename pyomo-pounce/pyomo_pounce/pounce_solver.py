"""POUNCE solver plugin for Pyomo.

Registers 'pounce' with Pyomo's SolverFactory. POUNCE speaks the AMPL
NL/SOL protocol, so Pyomo drives it through the AMPL Solver Library
interface exactly as it drives IPOPT.

The `pounce` binary is provided by the `pounce-solver` dependency,
which ships a per-platform wheel that drops the executable into the
active environment under `<venv>/bin/pounce`. The plugin resolves that
**bundled** binary deterministically (independent of PATH). Only when no
bundled binary is present (a source/dev checkout without the wheel) does
it fall back to whatever `pounce` is first on `PATH`.

**You must** ``import pyomo_pounce`` before ``SolverFactory('pounce')``:
without it Pyomo does not know the solver and raises a clear
``UnknownSolver`` / "plugin not registered" error (it does **not** silently
run some other `pounce`). If the plugin has to fall back to a PATH binary,
it warns — and :func:`check_binary` reports exactly which executable will
run, its build, and whether a different `pounce` earlier on PATH would
shadow it (version strings alone cannot tell two builds apart — e.g. a
binary from before and after a fix can both report the same ``X.Y.Z`` — so
the check compares the git *commit* embedded in ``pounce --about``).

Usage:
    import pyomo_pounce
    from pyomo.environ import *
    solver = SolverFactory('pounce')
    result = solver.solve(model)

Diagnose which binary will run:
    import pyomo_pounce
    pyomo_pounce.check_binary()          # prints a report; returns a dict
"""

import re
import shutil
import subprocess
import warnings

from pyomo.opt import SolverFactory
from pyomo.solvers.plugins.solvers.ASL import ASL


def _bundled_path():
    """Path to the wheel-bundled `pounce` binary, or None if not present."""
    try:
        from pounce._cli import _bundled_binary

        b = _bundled_binary()
        return str(b) if b.is_file() else None
    except Exception:
        return None


def _build_id(exe):
    """Best-effort build identifier of a `pounce` executable: the short git
    commit embedded in ``pounce --about``. Returns None if the executable is
    missing, too old to support ``--about``, or otherwise unqueryable.

    This is the discriminator a version string cannot provide: two builds
    straddling a bug fix can share the same ``X.Y.Z`` while differing in
    commit — see the pounce dual-sign fix (gh #271/#272), where a stale
    0.9.0 binary returned flipped duals that a version check would miss.

    A dirty working tree is kept in the identifier (``96fc5890+dirty``), so a
    build with uncommitted changes is distinguished from the clean build at
    the same commit — that is exactly a "same commit, different bits" case.
    ``unknown`` (a build made outside a git checkout) is still treated as
    unqueryable so two independent such builds never compare equal.
    """
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--about"], capture_output=True, text=True, timeout=15
        ).stdout
    except Exception:
        return None
    m = re.search(r"commit\s+([0-9a-f]+(?:\+dirty)?)", out)
    return m.group(1) if m else None


def _discrete_vars(model, max_list=5):
    """Names of active, non-fixed Binary/Integer variables in ``model``
    (capped at ``max_list``), plus the true total count.

    A *fixed* discrete variable is not a decision POUNCE has to relax — its
    value is already pinned by the user, so it is excluded.
    """
    from pyomo.core.base.var import Var

    names = []
    total = 0
    for v in model.component_data_objects(Var, active=True, descend_into=True):
        if v.fixed:
            continue
        if v.is_binary() or v.is_integer():
            total += 1
            if len(names) < max_list:
                names.append(v.name)
    return names, total


_fallback_warned = False


def _warn_path_fallback(resolved):
    """One-time warning that the plugin fell back to a PATH binary rather than
    the wheel-bundled one, naming the resolved executable and its build."""
    global _fallback_warned
    if _fallback_warned:
        return
    _fallback_warned = True
    bid = _build_id(resolved)
    warnings.warn(
        f"pyomo-pounce: no wheel-bundled `pounce` binary found; using the "
        f"PATH executable {resolved!r} (build {bid or 'unknown'}). This may "
        f"be a stale or unrelated `pounce` — version strings do not "
        f"distinguish builds. Run `pyomo_pounce.check_binary()` to verify, "
        f"or `pip install -U pounce-solver` to get the bundled binary.",
        UserWarning,
        stacklevel=3,
    )


@SolverFactory.register("pounce", doc="The POUNCE interior-point NLP solver")
class POUNCE(ASL):
    """Pyomo solver interface for POUNCE via the AMPL Solver Library protocol."""

    def __init__(self, **kwds):
        kwds["type"] = "pounce"
        super().__init__(**kwds)
        self._metasolver = False
        self.options.solver = "pounce"
        # POUNCE is a continuous NLP solver: no branch-and-bound, no SOS
        # handling. The generic `ASL` base class defaults these capabilities
        # to True (valid for many .nl-driven solvers), which is simply wrong
        # here and would let Pyomo's own solver-selection logic recommend
        # `pounce` for a MINLP (gh #341).
        self._capabilities.integer = False
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False

    def solve(self, *args, **kwds):
        # When the model declares sensitivity parameters
        # (pyomo_pounce.declare_sens_param), solve in-process through the
        # pounce.Solver session so the converged KKT factorization stays
        # available for gradient()/estimate(). Otherwise the ordinary
        # ASL/CLI path runs. The model may arrive positionally or as the
        # `model` keyword.
        from pyomo_pounce.sens import has_declarations, sens_solve

        model = args[0] if args else kwds.get("model")
        explicit = {k: kwds.pop(k) for k in
                    ("sens_params", "fitted", "residuals") if k in kwds}
        if explicit and model is None:
            # The sensitivity declarations were given with no model to solve;
            # surface the mistake rather than silently dropping them (they
            # have already been popped and would otherwise vanish).
            raise ValueError(
                "pounce solve: "
                f"{', '.join(sorted(explicit))}= given but no model was "
                "passed; call solve(model, ...) with the model positionally "
                "or as the `model` keyword")
        if model is not None:
            # POUNCE has no branch-and-bound: handing it a model with a live
            # (non-fixed) Binary/Integer variable does not fail -- it silently
            # solves the continuous relaxation and reports a fractional value
            # as `optimal` for a variable you declared discrete (gh #341).
            # Ipopt-via-ASL has this same gap, but pyomo_pounce already fails
            # loudly elsewhere for comparable silent-wrongness risks (the
            # ambiguous curve_fit bounds shape, gh #260/#265; the stale-binary
            # check, gh #315), so it does here too rather than matching the
            # generic ASL/Ipopt behavior.
            names, total = _discrete_vars(model)
            if names:
                shown = ", ".join(names)
                more = f" (+{total - len(names)} more)" if total > len(names) else ""
                raise ValueError(
                    "pounce solve: model has "
                    f"{total} active, non-fixed integer/binary variable(s) "
                    f"(e.g. {shown}{more}), but POUNCE is a continuous NLP "
                    "solver with no branch-and-bound or SOS handling -- it "
                    "would silently solve the continuous relaxation and "
                    "report a fractional value as 'optimal' for a variable "
                    "declared discrete. Fix the variable's domain (or set "
                    "var.domain = pyomo.environ.Reals / relax it explicitly) "
                    "if a continuous relaxation is what you intend, or use a "
                    "MINLP-capable solver.")
        if model is not None and (has_declarations(model) or explicit):
            return sens_solve(model, tee=kwds.get("tee", False), **explicit)
        return super().solve(*args, **kwds)

    def _default_executable(self):
        # Prefer the binary bundled in the installed ``pounce-solver`` wheel,
        # whose location is deterministic (``pounce/bin/pounce`` inside the
        # package) and independent of PATH. ``shutil.which("pounce")`` alone
        # finds only the ``<venv>/bin/pounce`` console-script shim, which is
        # invisible to non-activated-environment runs (cron, IDE runners,
        # Jupyter kernels) and can be shadowed by a stale system binary. Fall
        # back to PATH for system installs and local cargo dev builds where
        # ``pounce-solver`` is not installed — and WARN when we do, since a
        # PATH binary may be stale/unrelated and version strings cannot tell
        # builds apart (gh #315).
        bundled = _bundled_path()
        if bundled is not None:
            return bundled
        resolved = shutil.which("pounce")
        if resolved is not None:
            _warn_path_fallback(resolved)
        return resolved


def _all_path_pounce():
    """Every `pounce` executable resolvable via PATH, in PATH order — the
    candidates Pyomo's ASL fallback *would* pick from if the bundled binary
    were absent. Used to flag a binary that shadows the intended one.

    The executable is named ``pounce.exe`` on Windows; resolving it through
    ``shutil.which`` (rather than a bare ``pounce`` filename test) applies the
    platform's own name/extension and executable-bit rules, so the shadowing
    scan works on Windows too."""
    import os

    name = "pounce.exe" if os.name == "nt" else "pounce"
    seen, found = set(), []
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d:
            continue
        cand = shutil.which(name, path=d)
        if cand is None:
            continue
        real = os.path.realpath(cand)
        if real not in seen:
            seen.add(real)
            found.append(cand)
    return found


def check_binary(verbose=True):
    """Report which `pounce` executable ``SolverFactory('pounce')`` will run,
    its build, and whether it matches the wheel-bundled binary — and flag any
    *different* `pounce` earlier on PATH that would shadow it under Pyomo's
    ASL fallback (the case that silently ran a stale, dual-sign-flipped binary
    before gh #315).

    Returns a dict; when ``verbose`` also prints a human-readable report.
    Because two builds can share a version string, the comparison is on the
    git *commit* embedded in ``pounce --about``, not the version.
    """
    resolved = SolverFactory("pounce")._default_executable()
    bundled = _bundled_path()
    resolved_id = _build_id(resolved)
    bundled_id = _build_id(bundled)

    path_bins = []
    for p in _all_path_pounce():
        path_bins.append({"path": p, "build_id": _build_id(p)})

    # A PATH binary shadows the intended one if it is NOT the resolved binary
    # and its build differs — i.e. a bare SolverFactory ASL fallback (or a
    # forgotten `import pyomo_pounce`) could run a different build.
    import os

    resolved_real = os.path.realpath(resolved) if resolved else None
    shadowing = [
        b
        for b in path_bins
        if os.path.realpath(b["path"]) != resolved_real
        and b["build_id"] != resolved_id
    ]

    info = {
        "resolved_executable": resolved,
        "resolved_build_id": resolved_id,
        "bundled_executable": bundled,
        "bundled_build_id": bundled_id,
        "using_bundled": bool(bundled) and resolved == bundled,
        "matches_bundled": (
            bundled_id is not None and resolved_id == bundled_id
        ),
        "path_pounce_binaries": path_bins,
        "shadowing_path_binaries": shadowing,
    }

    if verbose:
        print("pyomo-pounce binary check")
        print(f"  will run : {resolved}  (build {resolved_id or 'unknown'})")
        if bundled:
            match = "MATCHES" if info["matches_bundled"] else "DIFFERS from"
            print(f"  bundled  : {bundled}  (build {bundled_id or 'unknown'})"
                  f"  [{match} the binary that will run]")
        else:
            print("  bundled  : none found (source/dev install without the "
                  "pounce-solver wheel)")
        if shadowing:
            print("  WARNING  : a different `pounce` is earlier on PATH; "
                  "without `import pyomo_pounce` (or on the ASL fallback "
                  "path) it could be run instead:")
            for b in shadowing:
                bid = b["build_id"] or "unknown"
                print(f"             {b['path']}  (build {bid})")
        else:
            print("  PATH     : no shadowing `pounce` detected")
    return info
