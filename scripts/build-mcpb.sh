#!/usr/bin/env bash
# Build the POUNCE Studio Claude Desktop extension (dist/pounce-studio-<v>.mcpb).
#
# The bundle carries no Python code of its own: studio/mcpb/pyproject.toml
# pins pounce-solver[mcp]==<v>, and Claude Desktop installs it with uv. The
# version is stamped here from python/pyproject.toml into both the manifest
# and that pin, so a bundle never asks for a solver release other than its
# own. The pin only resolves once <v> is on PyPI with pounce.studio in it.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
ver="$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/python/pyproject.toml" | head -1)"
[ -n "$ver" ] || { echo "could not read version from python/pyproject.toml" >&2; exit 1; }

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp -R "$root/studio/mcpb/." "$stage/"
rm -rf "$stage/.venv" "$stage/uv.lock" "$stage/src/__pycache__"
python3 - "$stage" "$ver" <<'PY'
import json, re, sys
stage, ver = sys.argv[1], sys.argv[2]
mp = f"{stage}/manifest.json"
m = json.load(open(mp)); m["version"] = ver
json.dump(m, open(mp, "w"), indent=2)
pp = f"{stage}/pyproject.toml"
s = open(pp).read()
s = re.sub(r'^version = ".*"$', f'version = "{ver}"', s, flags=re.M)
s = re.sub(r'pounce-solver\[mcp\]==[^"]+', f'pounce-solver[mcp]=={ver}', s)
open(pp, "w").write(s)
PY
mkdir -p "$root/dist"
out="$root/dist/pounce-studio-$ver.mcpb"
npx -y @anthropic-ai/mcpb validate "$stage/manifest.json"
npx -y @anthropic-ai/mcpb pack "$stage" "$out"
echo "wrote $out"
