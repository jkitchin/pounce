#!/usr/bin/env bash
# Record an asciinema screencast for every scenario in scripts/demo/scenarios/
# and (if `agg` is installed) convert each to a GIF. Outputs land in docs/demo/.
#
#   scripts/demo/record.sh              # record all scenarios
#   scripts/demo/record.sh circle       # record only matching scenario(s)
#
# Requires: asciinema, python3 + pexpect, pounce on PATH. agg is optional
# (cast -> gif); without it you still get the .cast files.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
scen_dir="$here/scenarios"
out_dir="$repo/docs/demo"
mkdir -p "$out_dir"

filter="${1:-}"

# Prefer this repo's freshly built binary so the recorded banner shows the
# workspace version, not a stale globally-installed pounce. Fall back to PATH.
THEME="${POUNCE_DEMO_THEME:-github-dark}"
PTY_ROWS=40 PTY_COLS=128   # the pty asciinema records in (wider than the 124
                           # cols pounce renders at, so nothing re-wraps)
if [ -x "$repo/target/release/pounce" ]; then
  export POUNCE_BIN="$repo/target/release/pounce"
elif [ -x "$repo/target/debug/pounce" ]; then
  export POUNCE_BIN="$repo/target/debug/pounce"
else
  command -v pounce >/dev/null || { echo "error: no pounce binary (build it or put it on PATH)" >&2; exit 1; }
  export POUNCE_BIN="pounce"
fi
echo "using $POUNCE_BIN ($("$POUNCE_BIN" --version 2>/dev/null || echo '?'))"

command -v asciinema >/dev/null || { echo "error: asciinema not found" >&2; exit 1; }
python3 -c 'import pexpect' 2>/dev/null || { echo "error: python pexpect not installed" >&2; exit 1; }
have_agg=0; command -v agg >/dev/null && have_agg=1 || echo "note: agg not found — skipping GIF conversion" >&2

for scenario in "$scen_dir"/*.dbg; do
  name="$(basename "$scenario" .dbg)"
  [ -n "$filter" ] && [[ "$name" != *"$filter"* ]] && continue
  cast="$out_dir/$name.cast"
  title="$(sed -n 's/^#[[:space:]]*title:[[:space:]]*//Ip' "$scenario" | head -1)"

  echo ">> recording $name  ($title)"
  # Record asciinema inside a wide pty (_rec_pty.py) so the iteration table
  # isn't re-wrapped at the default 80 columns.
  python3 "$here/_rec_pty.py" "$PTY_ROWS" "$PTY_COLS" \
    asciinema rec --overwrite --idle-time-limit 1.5 \
      -t "pounce --debug · $name" \
      -c "python3 '$here/drive_debug.py' '$scenario'" \
      "$cast"

  if [ "$have_agg" = 1 ]; then
    echo ">> rendering $name.gif (theme=$THEME)"
    agg --theme "$THEME" --font-size 18 "$cast" "$out_dir/$name.gif"
  fi
done

echo "done -> $out_dir"
