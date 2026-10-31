---
name: gams-link
description: How POUNCE registers as a GAMS solver - the two independent routes (the pip/pure-Python link built on gamsapi, and the native C link in gams/), where each lives, the per-user gamsconfig.yaml registration and its OS-specific config dir (not XDG on macOS), and which surface is CI-testable. Use when working on GAMS integration, gams_pounce.c, python/pounce/gams/, or GAMS registration problems.
---

# GAMS solver link

## GAMS solver link — two routes

POUNCE registers with GAMS (`option nlp = pounce;`) two independent ways:

1. **pip (pure-Python, recommended for users)** — `pip install
   pounce-solver[gams]` + `pounce-gams register`. Lives in
   `python/pounce/gams/` (`gmo_translate.py`, `link.py`, `register.py`) with the
   `pounce-gams` CLI in `python/pounce/_gams_cli.py`. Built on GAMS's own
   `gamsapi[core]` PyPI bindings (which `dlopen` the user's GAMS libs) — **we
   redistribute nothing GAMS-owned**. POUNCE is a local NLP solver, so the link
   wires GMO's numerical evaluators straight into the cyipopt-style `Problem`
   callbacks (no opcode translator, unlike discopt's global solver). Registers a
   script solver via a `gamsconfig.yaml` `solverConfig` entry — no `sudo`, no
   system-dir writes, survives GAMS upgrades. The per-user config dir is
   OS-specific and **NOT XDG on macOS**: macOS `~/Library/Preferences/GAMS`,
   Linux `~/.config/GAMS`, Windows `%LOCALAPPDATA%\GAMS` (verify with `gamsinst
   -listdirs`). License-free unit tests in `python/tests/test_gams_link.py`
   drive a fake `GmoView`; the live `gamsapi` adapter is the only
   CI-untestable surface.
2. **native C link** — `gams/gams_pounce.c` + `make -C gams && sudo make -C gams
   install`. The authoritative reference for GMO call sequence, sign
   conventions, option keywords, and status mapping. Adds active-set-SQP
   working-set / state-file warm starts the pip link does not yet reproduce.

Docs: `docs/src/gams.md` (user-facing), `gams/README.md` (C link).
