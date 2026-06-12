#!/usr/bin/env bash
# Docs consistency guard — run in CI on every PR. Area-4 tech debt (#108) is
# "the feature exists but isn't documented where a user looks." A page that is
# never linked from the table of contents is invisible in the rendered book,
# and a TOC entry pointing at a missing file breaks the build. This guard
# fails loudly on both:
#
#   1. NO ORPHAN PAGES. Every docs/src/**/*.md (except SUMMARY.md itself) must
#      be reachable from docs/src/SUMMARY.md — otherwise a freshly written
#      page ships in the repo but not in the book.
#
#   2. NO DEAD TOC LINKS. Every *.md target referenced in SUMMARY.md must
#      exist on disk.
#
# This is the mechanizable slice of the "one feature -> CHANGELOG entry + book
# section" definition-of-done (see CONTRIBUTING.md): it can't verify that a
# section is *good*, but it guarantees a new page is actually wired into the
# navigation a reader follows.
#
# Usage:
#   scripts/check-docs-consistency.sh        # exit 0 if consistent, 1 otherwise

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Run the checker as the condition of an `if` so `set -e` does NOT abort the
# script when python exits non-zero — that lets the friendlier failure message
# below actually run. (Previously the `rc=$?`/message block sat after a bare
# `python3 … <<PY` under `set -e`, so a failing check killed the script before
# the message was ever reached.)
if python3 - <<'PY'
import re, os, glob, sys

root = "docs/src"
summary_path = os.path.join(root, "SUMMARY.md")
summary = open(summary_path).read()

# Markdown link targets in SUMMARY that point at a local .md file (drop any
# #anchor). External links (http...) and non-md targets are ignored.
linked = {m.split('#')[0]
          for m in re.findall(r'\]\(([^)]+\.md)\)', summary)
          if not m.startswith(('http://', 'https://'))}

present = {os.path.relpath(p, root)
           for p in glob.glob(os.path.join(root, "**/*.md"), recursive=True)}
present.discard("SUMMARY.md")

orphans = sorted(present - linked)
dead = sorted(l for l in linked if not os.path.exists(os.path.join(root, l)))

rc = 0
if orphans:
    print("  ORPHAN pages (exist under docs/src but not linked from SUMMARY.md):")
    for o in orphans:
        print("    - " + o)
    rc = 1
if dead:
    print("  DEAD links (referenced in SUMMARY.md but no such file):")
    for d in dead:
        print("    - " + d)
    rc = 1

if rc == 0:
    print("  OK — %d pages, all reachable from SUMMARY.md, no dead links" % len(present))
sys.exit(rc)
PY
then
  echo "check-docs-consistency: OK"
else
  echo "check-docs-consistency: FAILED — wire new pages into docs/src/SUMMARY.md." >&2
  exit 1
fi
