"""Register POUNCE as a CasADi ``nlpsol`` plugin.

    import pounce_casadi   # noqa: F401
    solver = casadi.nlpsol("solver", "pounce", nlp, {"pounce": {"tol": 1e-9}})

Importing this package loads the plugin into the running process and
registers it with CasADi directly, so nothing is written into CasADi's
installation, no environment variable is needed, and CasADi's own
plugins — Ipopt included — stay loadable side by side.

The plugin is a C++ extension of CasADi, so it is built against a
specific CasADi **minor** version; CasADi performs no version handshake
of its own, and a mismatched plugin would load and then misbehave. This
package therefore ships one build per supported minor version and
selects on the CasADi actually installed, refusing to guess.
"""

from __future__ import annotations

import ctypes
import os
import sys

__all__ = ["plugin_path", "supported_casadi_versions", "register"]

_PLUGIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_plugins")

if sys.platform == "darwin":
    _PLUGIN_LIB = "libcasadi_nlpsol_pounce.dylib"
    _SOLVER_LIB = "libpounce_cinterface.dylib"
elif sys.platform == "win32":
    _PLUGIN_LIB = "casadi_nlpsol_pounce.dll"
    _SOLVER_LIB = "pounce_cinterface.dll"
else:
    _PLUGIN_LIB = "libcasadi_nlpsol_pounce.so"
    _SOLVER_LIB = "libpounce_cinterface.so"

_registered = False


def supported_casadi_versions() -> list[str]:
    """CasADi minor versions this install carries a plugin for, e.g. ``["3.7"]``."""
    if not os.path.isdir(_PLUGIN_DIR):
        return []
    return sorted(
        d
        for d in os.listdir(_PLUGIN_DIR)
        if os.path.isfile(os.path.join(_PLUGIN_DIR, d, _PLUGIN_LIB))
    )


def plugin_path(casadi_version: str | None = None) -> str:
    """Absolute path to the plugin matching `casadi_version` (default: installed)."""
    import casadi

    version = casadi_version or casadi.__version__
    minor = ".".join(version.split(".")[:2])
    candidate = os.path.join(_PLUGIN_DIR, minor, _PLUGIN_LIB)
    if not os.path.isfile(candidate):
        available = supported_casadi_versions()
        raise ImportError(
            f"pounce-casadi has no plugin for casadi {version}. "
            f"Builds available here: {', '.join(available) or 'none'}. "
            "Install a matching casadi, upgrade pounce-casadi, or build the "
            "plugin from the pounce repository (see casadi/README.md)."
        )
    return candidate


def register() -> None:
    """Load the plugin and register it with CasADi. Idempotent."""
    global _registered
    if _registered:
        return

    # Import first so libcasadi is already in the process: the plugin's
    # DT_NEEDED on it then resolves against the loaded copy, which is by
    # construction the one it was built for.
    import casadi  # noqa: F401

    path = plugin_path()

    # The solver itself ships next to the plugin. Load it explicitly
    # rather than relying on an rpath, so the wheel is relocatable.
    solver_lib = os.path.join(os.path.dirname(path), _SOLVER_LIB)
    if os.path.isfile(solver_lib):
        ctypes.CDLL(solver_lib, mode=ctypes.RTLD_GLOBAL)

    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    # `casadi_load_nlpsol_pounce` calls Nlpsol::registerPlugin — the same
    # entry point CasADi's own loader would call after finding the file
    # on its search path.
    lib.casadi_load_nlpsol_pounce()
    _registered = True


register()
