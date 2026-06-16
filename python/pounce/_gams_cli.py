"""``pounce-gams`` -- register POUNCE as a GAMS solver.

The main ``pounce`` executable is the Rust CLI binary, so GAMS registration is
its own pure-Python console script.  Subcommands:

* ``register``   -- write/merge ``gamsconfig.yaml`` + the launcher script
* ``unregister`` -- remove POUNCE from ``gamsconfig.yaml`` + delete the launcher
* ``status``     -- report gamsapi importability and whether POUNCE is registered
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_register(args) -> int:
    from pounce.gams import register

    written = register.write_registration(args.config_dir)
    action = written["action"]
    verb = {
        "created": "Created",
        "merged": "Merged POUNCE into",
        "replaced": "Updated POUNCE entry in",
    }.get(action, "Wrote")
    print(f"{verb} {written['config']}")
    print(f"Launcher: {written['script']}")
    print(f"\nPOUNCE is now registered for GAMS in {written['config_dir']}.")
    print("Solve a model with:  option nlp = pounce;")

    check = register.check_gamsapi()
    if not check["available"]:
        print(f"\nNote: {check['detail']}", file=sys.stderr)
    return 0


def _cmd_unregister(args) -> int:
    from pounce.gams import register

    result = register.unregister(args.config_dir)
    if result["removed"]:
        print(f"Removed POUNCE from {result['config']}")
    else:
        print(f"No POUNCE entry found in {result['config']} (launcher removed if present)")
    return 0


def _cmd_status(args) -> int:
    from pounce.gams import register

    cfg_dir = Path(args.config_dir) if args.config_dir else register.gams_user_config_dir()
    config = cfg_dir / "gamsconfig.yaml"

    check = register.check_gamsapi()
    print(f"gamsapi:       {'available' if check['available'] else 'NOT available'}")
    print(f"               {check['detail']}")
    print(f"config dir:    {cfg_dir}")

    registered = False
    if config.exists():
        text = config.read_text()
        registered = f"{register.SOLVER_NAME}:" in text and "scriptName" in text
    print(f"gamsconfig:    {config} ({'exists' if config.exists() else 'missing'})")
    print(f"POUNCE solver: {'registered' if registered else 'not registered'}")
    if not registered:
        print("\nRun `pounce-gams register` to register POUNCE with GAMS.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pounce-gams",
        description="Register POUNCE as a GAMS solver (pure-Python, gamsapi-based).",
    )
    sub = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config-dir",
        default=None,
        help="GAMS user config directory (default: auto-detected).",
    )

    p_reg = sub.add_parser(
        "register", parents=[common], help="Register POUNCE with GAMS."
    )
    p_reg.set_defaults(func=_cmd_register)

    p_unreg = sub.add_parser(
        "unregister", parents=[common], help="Remove POUNCE from GAMS."
    )
    p_unreg.set_defaults(func=_cmd_unregister)

    p_stat = sub.add_parser(
        "status", parents=[common], help="Show gamsapi + registration status."
    )
    p_stat.set_defaults(func=_cmd_status)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
