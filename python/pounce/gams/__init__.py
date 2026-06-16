"""POUNCE as a GAMS solver (pure-Python, pip-installable).

This package lets a GAMS user solve a model with POUNCE::

    option nlp = pounce;
    solve m using nlp minimizing z;

After ``pip install pounce-solver[gams]`` and a one-shot ``pounce-gams
register``, GAMS launches the POUNCE *solver link* with a control file; the link
reads the model through the GAMS Modeling Object (GMO), feeds GMO's numerical
function / gradient / Hessian evaluators straight into POUNCE's
cyipopt-compatible :class:`pounce.Problem`, solves it, and writes the solution
back.

It is the pip-installable counterpart to the native C link in ``gams/`` and
relies on GAMS's own ``gamsapi`` package (the ``[gams]`` extra); nothing
GAMS-owned is redistributed.
"""

from __future__ import annotations

from .gmo_translate import GmoProblem, GmoView, problem_from_gmo
from .link import (
    is_available,
    parse_option_file,
    solve_from_control_file,
    solve_view,
    status_to_gams,
)
from .register import (
    check_gamsapi,
    gams_user_config_dir,
    gamsconfig_snippet,
    render_gamsconfig,
    run_script,
    unregister,
    write_registration,
)

__all__ = [
    "GmoProblem",
    "GmoView",
    "problem_from_gmo",
    "is_available",
    "parse_option_file",
    "solve_from_control_file",
    "solve_view",
    "status_to_gams",
    "check_gamsapi",
    "gams_user_config_dir",
    "gamsconfig_snippet",
    "render_gamsconfig",
    "run_script",
    "unregister",
    "write_registration",
]
