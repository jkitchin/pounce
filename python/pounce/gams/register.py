"""Register POUNCE as a solver with a GAMS system.

A third-party solver is made known to GAMS through the ``solverConfig`` section
of ``gamsconfig.yaml`` (the modern mechanism; the legacy equivalent is an entry
in ``gmscmpun.txt``).  GAMS invokes the configured ``scriptName`` with the path
to a control file; that launcher just runs the POUNCE link
(:mod:`pounce.gams.link`).  No sudo, no system-directory writes, and the
registration survives GAMS upgrades.

* :func:`gamsconfig_snippet` returns the YAML block for POUNCE alone.
* :func:`render_gamsconfig` merges that block into an existing
  ``gamsconfig.yaml``, preserving any other solvers and top-level keys.
* :func:`run_script` returns the launcher (POSIX ``sh`` or Windows ``.cmd``).
* :func:`write_registration` writes the launcher + merged config into the GAMS
  user config directory.
* :func:`gams_user_config_dir` locates that directory.
* :func:`check_gamsapi` diagnoses gamsapi importability.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

SOLVER_NAME = "pounce"
# GAMS model types POUNCE can solve. POUNCE is a continuous local NLP solver,
# so this matches the C link's gmscmpun.txt registration -- NOT MINLP/MIP.
MODEL_TYPES = ("NLP", "DNLP", "RMINLP")


def gamsconfig_snippet(script_path: str | Path = "pounce-gams-link") -> str:
    """Return the ``gamsconfig.yaml`` ``solverConfig`` block for POUNCE.

    Follows the GAMS ``gamsconfig_schema.json`` solver schema: each entry maps
    the *solver name* to its config object with ``scriptName`` + ``modelTypes``.
    POUNCE is a control-file *script* solver, so no ``library`` block is emitted
    (GAMS invokes ``scriptName`` with the control file).
    """
    model_types = "\n".join(f"        - {t}" for t in MODEL_TYPES)
    return f"""\
solverConfig:
  - {SOLVER_NAME}:
      scriptName: {script_path}
      modelTypes:
{model_types}
"""


def _pounce_solver_entry(script_path: str) -> dict:
    """The POUNCE ``solverConfig`` entry as a Python object (for YAML merge)."""
    return {
        SOLVER_NAME: {
            "scriptName": script_path,
            "modelTypes": list(MODEL_TYPES),
        }
    }


def render_gamsconfig(existing: str | None, script_path: str | Path) -> tuple[str, str]:
    """Merge a POUNCE ``solverConfig`` entry into an existing config.

    ``existing`` is the current ``gamsconfig.yaml`` text (or ``None`` / empty for
    a fresh file).  Returns ``(text, action)`` where ``action`` is one of
    ``"created"`` (no prior file), ``"merged"`` (POUNCE added alongside other
    solvers), or ``"replaced"`` (a prior POUNCE entry was updated in place).

    Other solvers and unrelated top-level keys are preserved.  Falls back to a
    plain appended snippet when PyYAML is unavailable.
    """
    script_path = str(script_path)
    entry = _pounce_solver_entry(script_path)

    if not existing or not existing.strip():
        return gamsconfig_snippet(script_path), "created"

    try:
        import yaml
    except Exception:
        # No PyYAML: append the snippet and report it as a merge. The user may
        # need to hand-resolve a duplicate solverConfig key.
        return existing.rstrip() + "\n\n" + gamsconfig_snippet(script_path), "merged"

    data = yaml.safe_load(existing)
    if not isinstance(data, dict):
        data = {}

    solver_cfg = data.get("solverConfig")
    if not isinstance(solver_cfg, list):
        solver_cfg = []

    action = "merged"
    replaced = False
    for i, item in enumerate(solver_cfg):
        if isinstance(item, dict) and SOLVER_NAME in item:
            solver_cfg[i] = entry
            replaced = True
            action = "replaced"
            break
    if not replaced:
        solver_cfg.append(entry)
    data["solverConfig"] = solver_cfg

    text = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
    return text, action


def run_script(python_executable: str | None = None, *, windows: bool | None = None) -> str:
    """Return the launcher that runs the POUNCE GAMS link.

    GAMS calls it with the control-file path as ``$1`` (POSIX) or ``%*``
    (Windows).  ``windows`` defaults to the current platform.
    """
    py = python_executable or sys.executable
    if windows is None:
        windows = os.name == "nt"
    # Invoke via `-c` import rather than `-m pounce.gams.link`: running the
    # module as __main__ after the package has already imported it triggers a
    # benign-but-noisy runpy RuntimeWarning, which would otherwise show up in
    # the GAMS log on every solve.
    run = "from pounce.gams.link import main; import sys; sys.exit(main())"
    if windows:
        return f'@echo off\r\n"{py}" -c "{run}" %*\r\n'
    return f"""\
#!/bin/sh
# POUNCE GAMS solver link -- invoked by GAMS as: <script> <control-file>
exec "{py}" -c "{run}" "$@"
"""


def gams_user_config_dir() -> Path:
    """Return the GAMS per-user config directory (created on demand by callers).

    These are the per-user locations GAMS searches for ``gamsconfig.yaml``
    (verified with ``gamsinst -listdirs``; see the GAMS "Standard Locations"
    docs).  The path is OS-specific and NOT the XDG default on macOS:

    * Windows: ``%LOCALAPPDATA%/GAMS`` (else ``%USERPROFILE%/Documents/GAMS``)
    * macOS:   ``$HOME/Library/Preferences/GAMS``
    * Linux:   ``$XDG_CONFIG_HOME/GAMS`` (else ``~/.config/GAMS``)
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "GAMS"
        home = os.environ.get("USERPROFILE") or str(Path.home())
        return Path(home) / "Documents" / "GAMS"
    if sys.platform == "darwin":
        # macOS does NOT use XDG; GAMS searches ~/Library/Preferences/GAMS.
        return Path.home() / "Library" / "Preferences" / "GAMS"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "GAMS"


def write_registration(directory: str | Path | None = None) -> dict[str, object]:
    """Write the launcher and a merged ``gamsconfig.yaml``.

    Writes into ``directory`` (defaults to :func:`gams_user_config_dir`).  An
    existing ``gamsconfig.yaml`` is merged in place (other solvers preserved).
    Returns ``{"config": Path, "script": Path, "action": str, "config_dir": Path}``.
    """
    out = Path(directory) if directory is not None else gams_user_config_dir()
    out.mkdir(parents=True, exist_ok=True)

    windows = os.name == "nt"
    script = out / ("pounce-gams-link.cmd" if windows else "pounce-gams-link")
    script.write_text(run_script(windows=windows))
    if not windows:
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    config = out / "gamsconfig.yaml"
    existing = config.read_text() if config.exists() else None
    text, action = render_gamsconfig(existing, script_path=str(script))
    config.write_text(text)

    return {"config": config, "script": script, "action": action, "config_dir": out}


def unregister(directory: str | Path | None = None) -> dict[str, object]:
    """Remove POUNCE from ``gamsconfig.yaml`` and delete the launcher.

    Returns ``{"config": Path, "removed": bool, "config_dir": Path}``.  Other
    solvers in the config are preserved.  Requires PyYAML for a clean removal;
    without it, only the launcher is deleted and ``removed`` is ``False``.
    """
    out = Path(directory) if directory is not None else gams_user_config_dir()
    config = out / "gamsconfig.yaml"
    removed = False

    if config.exists():
        try:
            import yaml

            data = yaml.safe_load(config.read_text())
            if isinstance(data, dict) and isinstance(data.get("solverConfig"), list):
                kept = [
                    item
                    for item in data["solverConfig"]
                    if not (isinstance(item, dict) and SOLVER_NAME in item)
                ]
                removed = len(kept) != len(data["solverConfig"])
                if kept:
                    data["solverConfig"] = kept
                else:
                    data.pop("solverConfig", None)
                config.write_text(
                    yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
                )
        except Exception:
            removed = False

    for name in ("pounce-gams-link", "pounce-gams-link.cmd"):
        launcher = out / name
        if launcher.exists():
            launcher.unlink()

    return {"config": config, "removed": removed, "config_dir": out}


def check_gamsapi() -> dict[str, object]:
    """Diagnose whether the GAMS expert-level Python API (gamsapi) is usable.

    Returns ``{"available": bool, "detail": str}``.  ``available`` is true only
    when both ``gams.core.gmo`` and ``gams.core.gev`` import (the bindings
    ``dlopen`` the user's GAMS libraries, so an import failure usually means a
    missing or version-mismatched ``gamsapi``/GAMS install).
    """
    try:
        import gams  # noqa: F401
        import gams.core.gev  # noqa: F401
        import gams.core.gmo  # noqa: F401
    except Exception as exc:
        return {
            "available": False,
            "detail": (
                f"gamsapi not importable ({exc}). Install it from your GAMS "
                "system, matched to your GAMS version: pip install gamsapi[core]"
            ),
        }
    version = getattr(__import__("gams"), "__version__", "unknown")
    return {"available": True, "detail": f"gamsapi {version} importable"}
