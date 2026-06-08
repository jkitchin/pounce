#!/usr/bin/env python3
"""Drive `pounce --debug` like a human typing, for an asciinema screencast.

Reads a scenario file (a list of debugger commands plus a couple of header
directives) and replays it against the interactive solver debugger, typing each
command one character at a time so the recording looks hand-driven rather than
pasted. Used by ``scripts/demo/record.sh`` and ``make screencast``.

Scenario file format (see scripts/demo/scenarios/*.dbg)::

    # problem: circle                     # which built-in to solve (required)
    # title:   Catch a bad step, perturb  # banner printed before the session
    b 3                                   # one debugger command per line
    @1.6 set x[0] 1.2                     # @<secs> prefix overrides the pause
    # a plain comment line is ignored

Lines beginning with ``#`` are comments/directives; blank lines are skipped.
"""
import argparse
import os
import sys
import time

import pexpect

PROMPT = "pounce-dbg>"
# Which pounce to drive. Defaults to whatever is on PATH; record.sh points this
# at the freshly built target/release/pounce so the banner shows this repo's
# version rather than a stale globally-installed one.
POUNCE_BIN = os.environ.get("POUNCE_BIN", "pounce")


def parse_scenario(path):
    """Return (problem, title, [(pause, command), ...]) from a .dbg file."""
    problem = title = None
    commands = []
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                body = stripped[1:].strip()
                if body.lower().startswith("problem:"):
                    problem = body.split(":", 1)[1].strip()
                elif body.lower().startswith("title:"):
                    title = body.split(":", 1)[1].strip()
                continue  # ordinary comment
            pause = None
            if stripped.startswith("@"):
                tok, _, rest = stripped[1:].partition(" ")
                pause = float(tok)
                stripped = rest.strip()
            commands.append((pause, stripped))
    if not problem:
        sys.exit(f"{path}: missing `# problem:` directive")
    return problem, title, commands


def type_out(child, text, cps):
    """Send ``text`` one character at a time so the recording shows typing."""
    for ch in text:
        child.send(ch)
        time.sleep(1.0 / cps)
    child.send("\r")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario", help="path to a .dbg scenario file")
    ap.add_argument("--cps", type=float, default=22.0,
                    help="characters typed per second (default: 22)")
    ap.add_argument("--pause", type=float, default=1.0,
                    help="default seconds to wait at each prompt (default: 1.0)")
    ap.add_argument("--rows", type=int, default=38)
    ap.add_argument("--cols", type=int, default=124)
    args = ap.parse_args()

    problem, title, commands = parse_scenario(args.scenario)

    if title:
        # A leading banner so a viewer knows what they're about to watch.
        print(f"$ # {title}")
        time.sleep(0.8)
    # Always display `pounce …` regardless of which binary we actually run.
    print(f"$ pounce --problem {problem} --debug")
    time.sleep(0.4)

    child = pexpect.spawn(
        POUNCE_BIN, args=["--problem", problem, "--debug"],
        encoding="utf-8", timeout=60,
        dimensions=(args.rows, args.cols),
    )
    child.logfile_read = sys.stdout

    for pause, cmd in commands:
        try:
            child.expect_exact(PROMPT)
        except pexpect.EOF:
            break  # solve ended before we ran every command — fine
        time.sleep(args.pause if pause is None else pause)
        type_out(child, cmd, args.cps)

    # When a solve TERMINATES it returns to the prompt rather than exiting, so
    # quit cleanly if we're still at one. Scenarios therefore don't need a
    # trailing `q`, and a stray one is harmless (the process is already gone).
    try:
        child.expect_exact(PROMPT, timeout=5)
        time.sleep(0.6)
        type_out(child, "q", args.cps)
    except (pexpect.EOF, pexpect.TIMEOUT):
        pass
    try:
        child.expect(pexpect.EOF, timeout=10)
    except (pexpect.EOF, pexpect.TIMEOUT):
        pass
    time.sleep(0.6)


if __name__ == "__main__":
    main()
