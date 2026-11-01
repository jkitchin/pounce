"""POUNCE solver plugin for Pyomo.

Registers 'pounce' with Pyomo's SolverFactory. POUNCE speaks the AMPL
NL/SOL protocol, so Pyomo drives it through the AMPL Solver Library
interface exactly as it drives IPOPT.

The `pounce` binary is provided by the `pounce-solver` dependency,
which ships a per-platform wheel that drops the executable into the
active environment under `<venv>/bin/pounce`. The plugin resolves that
**bundled** binary deterministically (independent of PATH). Failing that
it takes the cargo build of the source checkout the `pounce` package is
imported out of (`target/release/pounce` or `target/debug/pounce`), and
only if there is no checkout either does it fall back to whatever
`pounce` is first on `PATH`. The middle rung exists because
`maturin develop` builds the extension module and nothing else: without
it a source install fell through to PATH and found its own
console-script shim there, pointing at a binary nobody had built
(gh #816).

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


def _checkout_path():
    """Path to the cargo-built `pounce` of the surrounding source checkout,
    or None when this is not a checkout / nothing has been built.

    `maturin develop` builds the extension module and nothing else, so a
    source install has no bundled binary at all (gh #816). Without this the
    plugin fell through to PATH, where the *console-script shim* of that same
    install is what it found -- a shim that then failed to find the binary it
    was supposed to exec. Resolving cargo's own output directly skips both
    the failure and the per-solve Python interpreter the shim costs.
    """
    try:
        from pounce._cli import _checkout_binary

        b = _checkout_binary()
        return str(b) if b is not None else None
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


def reject_discrete_vars(model):
    """Raise if `model` has a live (active, non-fixed) Binary/Integer
    variable, which POUNCE cannot honor.

    POUNCE has no branch-and-bound: handing it such a model does not
    fail -- it silently solves the continuous relaxation and reports a
    fractional value as `optimal` for a variable you declared discrete
    (gh #341). Ipopt-via-ASL has this same gap, but pyomo_pounce already
    fails loudly elsewhere for comparable silent-wrongness risks (the
    ambiguous curve_fit bounds shape, gh #260/#265; the stale-binary
    check, gh #315), so it does here too rather than matching the
    generic ASL/Ipopt behavior.

    Shared by both interfaces: the legacy `POUNCE` plugin and the
    `pyomo.contrib.solver` one in `pyomo_pounce.v2`. A guard that
    protected only the legacy route would leave the modern route with
    exactly the silent wrongness this exists to prevent (gh #558).
    """
    if model is None:
        return
    names, total = _discrete_vars(model)
    if not names:
        return
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


_checkout_warned = False


def _warn_checkout_fallback(resolved):
    """One-time warning that the plugin resolved the source checkout's cargo
    build because the install carries no bundled binary (gh #816).

    Quieter than the PATH warning by one notch and for one reason: the binary
    is at least provably this checkout's own build, not an arbitrary `pounce`
    that happened to be first on PATH. It is still not the wheel's binary, and
    it can still be a stale `target/release` from before the change under
    test, so it is still named out loud."""
    global _checkout_warned
    if _checkout_warned:
        return
    _checkout_warned = True
    bid = _build_id(resolved)
    warnings.warn(
        f"pyomo-pounce: no wheel-bundled `pounce` binary in this install; "
        f"using the source checkout's cargo build {resolved!r} "
        f"(build {bid or 'unknown'}). `make dev` from the repo root stages it "
        f"into the package the way a wheel ships it; "
        f"`pyomo_pounce.check_binary()` reports what will run.",
        UserWarning,
        stacklevel=3,
    )


_unavailable_warned = set()


def _warn_unavailable(reason):
    """Warn once per distinct reason. ``available(exception_flag=False)`` is
    called in loops (test skips, preflight helpers); the reason is worth
    exactly one line however many times it is asked."""
    if reason in _unavailable_warned:
        return
    _unavailable_warned.add(reason)
    warnings.warn(reason, UserWarning, stacklevel=3)


def _version_probe(exe):
    """`(returncode, merged output)` from ``<exe> -v`` -- the exact call
    Pyomo's ASL layer makes to decide whether a solver is available
    (`ASL._get_version`). Returns `(None, str(exc))` if it cannot be run."""
    try:
        r = subprocess.run(
            [exe, "-v"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            errors="ignore",
        )
    except Exception as exc:  # OSError, TimeoutExpired, ...
        return None, f"{type(exc).__name__}: {exc}"
    return r.returncode, r.stdout.strip()


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
        # Complementarity pairs lowered by pyomo_pounce.mpec for the solve
        # in flight (None outside a solve, or with no Complementarity).
        self._cc_lowering = None

    def solve(self, *args, **kwds):
        # When the model declares sensitivity parameters
        # (pyomo_pounce.declare_sens_param), solve in-process through the
        # pounce.Solver session so the converged KKT factorization stays
        # available for sens_jacobian()/sens_solution(). Otherwise the ordinary
        # ASL/CLI path runs. The model may arrive positionally or as the
        # `model` keyword.
        from pyomo_pounce.scaling import (
            user_scaling_requested,
            warn_if_no_suffix,
        )
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
        reject_discrete_vars(model)
        # Solver options as the solve will actually see them: factory-level
        # options (SolverFactory("pounce", options=...) or
        # solver.options[...]) first, per-call options={...} on top.
        # Dropping them here silently un-tuned every model the day it gained
        # a declaration (gh #432).
        # the ASL layer keeps its executable option prefix under the
        # `solver` key; it is bookkeeping, not a solver option
        opts = {k: v for k, v in self.options.items() if k != "solver"}
        opts.update(kwds.get("options") or {})
        if model is not None and user_scaling_requested(opts):
            # gh #483: the `scaling_factor` Suffix is the standard Pyomo
            # channel for user scaling, and it used to reach pyomo-pounce
            # and stop -- the option was accepted and meant "none". All
            # three kinds of factor now flow through: objective and
            # constraint via the writer's `.nl` suffix segments, and
            # variable factors as a change of variables inside the core
            # (gh #486 stage 2). A request with nothing to apply still
            # says so.
            warn_if_no_suffix(model)
        if model is not None and (has_declarations(model) or explicit):
            return sens_solve(model, tee=kwds.get("tee", False),
                              options=opts, **explicit)
        from pyomo.common.errors import ApplicationError
        from pyomo_pounce.mpec import lower_complementarity

        # Pyomo's mpec.nl route loses two-sided complementarity pairs (and
        # one-sided pairs on a variable with a different bound of its own)
        # on the way to `.nl`, and the solve then reports "optimal" at a
        # non-complementary point. Lower those pairs ourselves first; the
        # written file is checked in `_presolve` (see pyomo_pounce.mpec).
        self._cc_lowering = (lower_complementarity(model)
                             if model is not None else None)
        try:
            return super().solve(*args, **kwds)
        except ApplicationError as e:
            # Pyomo's message is only "did not exit normally"; the reason the
            # CLI refused the model (a sensitivity request with complementarity pairs, say) is
            # in the solver log, which a caller catching the exception never
            # sees. Carry the CLI's own `pounce: ...` lines into it.
            reasons = [ln.strip() for ln in str(getattr(self, "_log", "") or "").splitlines()
                       if ln.strip().startswith("pounce: ")]
            if not reasons:
                raise
            raise ApplicationError(f"{e}: {' '.join(reasons)}") from e
        finally:
            # After the solution load: the auxiliary pair components go, the
            # user's Complementarity data come back, and so do the bounds
            # mpec.nl wrote onto the user's variables.
            if self._cc_lowering is not None:
                self._cc_lowering.restore()
            self._cc_lowering = None

    def _presolve(self, *args, **kwds):
        # The ASL layer applies mpec.nl and writes the `.nl` here. Fail
        # closed if the written file does not carry every pair.
        super()._presolve(*args, **kwds)
        lowering = getattr(self, "_cc_lowering", None)
        files = getattr(self, "_problem_files", None) or ()
        if lowering is not None and files and str(files[0]).endswith(".nl"):
            from pyomo_pounce.mpec import check_written_nl

            check_written_nl(files[0], lowering)

    def _default_executable(self):
        # Prefer the binary bundled in the installed ``pounce-solver`` wheel,
        # whose location is deterministic (``pounce/bin/pounce`` inside the
        # package) and independent of PATH. ``shutil.which("pounce")`` alone
        # finds only the ``<venv>/bin/pounce`` console-script shim, which is
        # invisible to non-activated-environment runs (cron, IDE runners,
        # Jupyter kernels) and can be shadowed by a stale system binary.
        #
        # Second choice is the cargo build of the source checkout the
        # ``pounce`` package is being imported out of. A ``maturin develop``
        # install has no bundled binary, so this used to fall straight to
        # PATH — and what it found there was that install's own console-script
        # shim, pointing at the binary nobody built (gh #816). Resolving
        # cargo's output directly is both correct and cheaper than routing
        # every ASL call through a Python shim.
        #
        # PATH is the last resort, for system installs where ``pounce-solver``
        # is not installed at all — and WARN when we get there, since a PATH
        # binary may be stale/unrelated and version strings cannot tell builds
        # apart (gh #315).
        bundled = _bundled_path()
        if bundled is not None:
            return bundled
        checkout = _checkout_path()
        if checkout is not None:
            _warn_checkout_fallback(checkout)
            return checkout
        resolved = shutil.which("pounce")
        if resolved is not None:
            _warn_path_fallback(resolved)
        return resolved

    def available(self, exception_flag=True):
        """True when a `pounce` executable is resolvable *and* runs.

        Same verdict as ``ASL.available`` — an executable plus a parseable
        ``pounce -v`` — but never a bare ``False``. Pyomo's version reports
        no reason, and the reason is the whole content of the answer: gh #816
        is a report of ``available()`` returning False while ``solve()``
        returned ``optimal``, which is what it looks like when the two take
        different routes to a binary (``solve()`` through the bundled path,
        ``available()`` through a PATH console-script shim that could not
        find one). A False that names the executable and quotes what running
        it printed is diagnosable in one line; a bare False sent that report
        looking at the model.

        With ``exception_flag`` (Pyomo's default) the reason is raised as
        ``ApplicationError``; without it, it is warned once and False
        returned, so ``available(exception_flag=False)`` stays a predicate.
        """
        if super().available(exception_flag=False):
            return True
        reason = self._unavailable_reason()
        if exception_flag:
            from pyomo.common.errors import ApplicationError

            raise ApplicationError(reason)
        _warn_unavailable(reason)
        return False

    def _unavailable_reason(self):
        """Why :meth:`available` said no, as a sentence a user can act on."""
        try:
            exe = self.executable()
        except NotImplementedError:
            exe = None
        if exe is None:
            return (
                "pyomo-pounce: no `pounce` executable found. Looked for the "
                "binary bundled in an installed `pounce-solver` wheel, then "
                "for a cargo build in a surrounding source checkout "
                "(target/release or target/debug), then on PATH. Install the "
                "wheel (`pip install -U pounce-solver`), or in a checkout run "
                "`make dev` from the repo root."
            )
        rc, out = _version_probe(exe)
        shown = out.splitlines()[0] if out else "(no output)"
        return (
            f"pyomo-pounce: the `pounce` executable {exe!r} did not report a "
            f"version, so Pyomo treats the solver as unavailable. "
            f"`pounce -v` exited {rc} and printed: {shown!r}. If that is the "
            f"console-script shim of a `maturin develop` install, no CLI "
            f"binary was ever built — run `make dev` from the repo root "
            f"(gh #816). `pyomo_pounce.check_binary()` reports the full "
            f"picture."
        )


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
    checkout = _checkout_path()
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
        "checkout_executable": checkout,
        "using_bundled": bool(bundled) and resolved == bundled,
        "using_checkout": bool(checkout) and resolved == checkout,
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
        if checkout:
            note = ("  [this is what will run]" if info["using_checkout"]
                    else "")
            print(f"  checkout : {checkout}{note}")
        elif bundled is None:
            print("  checkout : none built (run `make dev` from the repo "
                  "root, or `cargo build --release --bin pounce`)")
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
