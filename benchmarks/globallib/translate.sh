#!/usr/bin/env bash
# Regenerate the GLOBALLib `.nl` benchmark files from their AMPL `.mod` sources.
#
# The benchmark set is the GLOBALLib subset that has a *proven* global optimum
# (MINLPLib `=opt=`). The `.mod` files come from ampl/global-optimization; the
# `.nl` files are produced by AMPL's `write` and dropped into the bench-data
# tree (Dropbox), the same place every other supplied benchmark tier lives.
#
# Requirements: an `ampl` on PATH (or set $AMPL), and the optima reference
# (`optima.txt`) that ships next to this script.
#
# Usage:  benchmarks/globallib/translate.sh [out_nl_dir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMPL="${AMPL:-ampl}"
OUT="${1:-${POUNCE_BENCH_DATA:-$HOME/Dropbox/projects/pounce-bench-data}/globallib/nl}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

command -v "$AMPL" >/dev/null || { echo "error: no '$AMPL' on PATH (set \$AMPL)"; exit 1; }
mkdir -p "$OUT"

echo "cloning ampl/global-optimization (.mod sources)..."
git clone --depth 1 https://github.com/ampl/global-optimization.git "$WORK/go" >/dev/null 2>&1
MOD="$WORK/go/global"

echo "translating $(wc -l < "$HERE/optima.txt") models -> $OUT"
 n=0; fail=0
while read -r stem _val; do
  [ -n "$stem" ] || continue
  src="$MOD/$stem.mod"
  if [ ! -f "$src" ]; then echo "  MISSING .mod: $stem"; fail=$((fail+1)); continue; fi
  ( cd "$OUT" && printf 'model %s;\noption auxfiles rc;\nwrite g%s;\n' "$src" "$stem" \
      | "$AMPL" >/dev/null 2>&1 )
  if [ -f "$OUT/$stem.nl" ]; then n=$((n+1)); else echo "  FAIL: $stem"; fail=$((fail+1)); fi
done < "$HERE/optima.txt"
echo "done: $n translated, $fail failed"
