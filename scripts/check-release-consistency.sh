#!/usr/bin/env bash
# Release consistency guard — run in CI before tagging, and locally before a
# release. POUNCE ships to three registries (PyPI pounce-solver, PyPI
# pyomo-pounce, and 19 crates.io workspace crates); this script fails loudly
# if any of the facts a release depends on have drifted apart:
#
#   1. VERSIONS AGREE. The Rust [workspace.package] version, the
#      pounce-solver wheel version, and the pyomo-pounce version must be the
#      same X.Y.Z. A tag publishes all three, so a mismatch ships a split
#      release.
#
#   2. THE PUBLISH LIST MATCHES THE WORKSPACE. scripts/publish-crates.sh
#      carries an explicit, reviewable CRATES=(...) list. The ground truth for
#      "what crates.io should receive" is `cargo metadata` (every workspace
#      member not marked `publish = false`). This check fails if the script
#      lists a crate that no longer exists (the bug that shipped
#      pounce-simplex / pounce-global long after they were removed) or omits a
#      newly added publishable crate.
#
#   3. THE PUBLISH ORDER IS TOPOLOGICAL. cargo refuses to publish a crate
#      before the crates it depends on are live, so the list must be in
#      dependency order. This check fails if any crate appears before one of
#      its own (publishable) workspace dependencies.
#
#   4. EVERY DEPENDENCY IS PUBLISHABLE. cargo publish rewrites path/git deps to
#      a crates.io version requirement and refuses to upload a crate whose
#      dependency lacks one (or pins a git rev / wildcard). Without this check a
#      tag would publish the leading crates and hard-fail mid-batch at the first
#      crate carrying such a dep — an irreversible partial release. See
#      scripts/check_dep_publishability.py (e.g. the `feral` git pin).
#
# `cargo metadata` is the single source of truth: it is the real workspace and
# cannot drift. The explicit list in publish-crates.sh stays because it makes
# the publish set reviewable in a PR diff — this guard is what keeps it honest.
#
# Usage:
#   scripts/check-release-consistency.sh        # exit 0 if consistent, 1 otherwise

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

fail=0
note() { printf '  %s\n' "$1"; }

# --- 1. Versions agree across the three release surfaces -------------------
cargo_ver="$(grep -m1 -E '^version[[:space:]]*=' Cargo.toml \
  | sed -E 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/')"
pysolver_ver="$(grep -m1 -E '^version[[:space:]]*=' python/pyproject.toml \
  | sed -E 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/')"
pyomo_ver="$(grep -m1 -E '^version[[:space:]]*=' pyomo-pounce/pyproject.toml \
  | sed -E 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/')"

echo "== version agreement =="
note "Cargo.toml [workspace.package] : ${cargo_ver}"
note "python/pyproject.toml          : ${pysolver_ver}"
note "pyomo-pounce/pyproject.toml    : ${pyomo_ver}"
if [[ "$cargo_ver" == "$pysolver_ver" && "$cargo_ver" == "$pyomo_ver" ]]; then
  note "OK — all three at ${cargo_ver}"
else
  note "DRIFT — the three release surfaces disagree; align them before tagging."
  fail=1
fi
echo

# --- 2 & 3. Publish list matches workspace, in topological order -----------
# Source the CRATES=(...) array out of publish-crates.sh without running it.
# shellcheck disable=SC1090
source <(sed -n '/^CRATES=(/,/^)/p' scripts/publish-crates.sh)
script_list="${CRATES[*]}"

echo "== crates.io publish list (scripts/publish-crates.sh) =="
# The single-quoted python below reads the list via os.environ, not bash
# expansion — SC2016 ($-in-single-quotes) is intentional here.
# shellcheck disable=SC2016
cargo metadata --format-version 1 2>/dev/null | SCRIPT_LIST="$script_list" python3 -c '
import sys, json, os

m = json.load(sys.stdin)
ws = set(m["workspace_members"])
id2pkg = {p["id"]: p for p in m["packages"]}

# Publishable = workspace member without `publish = false` (which cargo
# metadata reports as an empty publish list).
pub = {}
for pid in ws:
    p = id2pkg[pid]
    if p.get("publish") == []:
        continue
    pub[p["name"]] = p
names = set(pub)

# Dependency edges restricted to publishable normal/build deps.
edges = {n: set() for n in names}
for n, p in pub.items():
    for d in p["dependencies"]:
        if d.get("kind") in (None, "build") and d["name"] in names:
            edges[n].add(d["name"])

script = os.environ["SCRIPT_LIST"].split()
script_set = set(script)

rc = 0

# (2) Set equality between the script list and the workspace publishable set.
missing = sorted(names - script_set)        # publishable but not in script
extra = sorted(script_set - names)          # in script but not publishable
dups = sorted({c for c in script if script.count(c) > 1})
if missing:
    print("  MISSING from publish-crates.sh (publishable but not listed): " + ", ".join(missing)); rc = 1
if extra:
    print("  STALE in publish-crates.sh (listed but not a publishable workspace crate): " + ", ".join(extra)); rc = 1
if dups:
    print("  DUPLICATED in publish-crates.sh: " + ", ".join(dups)); rc = 1

# (3) The listed order must be topological: every dep of crate c that is
# itself published must appear before c. Only meaningful when the set matches.
if not missing and not extra and not dups:
    pos = {c: i for i, c in enumerate(script)}
    bad = []
    for c in script:
        for d in sorted(edges.get(c, ())):
            if pos[d] > pos[c]:
                bad.append("%s depends on %s but is listed before it" % (c, d))
    if bad:
        print("  NON-TOPOLOGICAL publish order:")
        for b in bad:
            print("    - " + b)
        rc = 1

if rc == 0:
    print("  OK — %d crates, set matches cargo metadata, order is topological" % len(script))

sys.exit(rc)
' || fail=1
echo

# --- 4. Every dependency of a publishable crate is itself publishable -------
# cargo publish drops git/path specs and needs a crates.io version for each
# dependency; a git pin or version-less dep makes the publish hard-fail
# mid-batch. Check the publish list specifically (the crates a tag will upload).
echo "== dependency publishability =="
if cargo metadata --format-version 1 2>/dev/null \
  | python3 scripts/check_dep_publishability.py $script_list; then
  :
else
  fail=1
fi
echo

if [[ $fail -ne 0 ]]; then
  echo "check-release-consistency: FAILED — fix the drift above before releasing." >&2
  exit 1
fi
echo "check-release-consistency: OK"
