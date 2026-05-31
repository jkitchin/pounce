#!/usr/bin/env bash
# Publish POUNCE crates to crates.io in dependency order.
#
# The first publish of all 13 crates will hit the crates.io rate limit
# for *new* crate names (5 burst then 1 per ~10 min). Before the initial
# release email help@crates.io and ask for a temporary exemption for
# this batch — they typically grant within a day. See
# dev-notes/cargo-release.md for the full procedure.
#
# Usage:
#   scripts/publish-crates.sh                 # publish all, real upload
#   scripts/publish-crates.sh --dry-run       # cargo publish --dry-run on each
#   scripts/publish-crates.sh --start-from pounce-algorithm
#                                             # resume after a mid-batch failure
#   SLEEP=600 scripts/publish-crates.sh       # override per-crate sleep (default 0)
#                                             # set to 600 (10 min) if no rate-limit exemption
#
# Each crate is published with `cargo publish -p <name>`; the
# workspace-deps refactor (Cargo.toml: [workspace.dependencies] with
# version= entries) means cargo accepts the path deps for registry
# publication. Crates already marked `publish = false` (pounce-py,
# iter-diff) are not in this list. (The benchmark crates pounce-cutest
# and pounce-large-scale were retired when those suites moved to .nl.)

set -euo pipefail

# Topologically sorted: each crate appears only after every crate it
# depends on. Verified against `cargo metadata` — see
# dev-notes/cargo-release.md for the dependency graph.
CRATES=(
  pounce-common
  pounce-linalg
  pounce-linsol
  pounce-nlp
  pounce-feral
  pounce-hsl
  pounce-l1penalty
  pounce-presolve
  pounce-algorithm
  pounce-restoration
  pounce-sensitivity
  pounce-cinterface
  pounce-cli
)

DRY_RUN=""
START_FROM=""
SLEEP="${SLEEP:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN="--dry-run"; shift ;;
    --start-from)
      START_FROM="$2"; shift 2 ;;
    --start-from=*) START_FROM="${1#*=}"; shift ;;
    -h|--help)
      sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# If --start-from given, skip until we hit it.
if [[ -n "$START_FROM" ]]; then
  found=0
  filtered=()
  for c in "${CRATES[@]}"; do
    if [[ "$c" == "$START_FROM" ]]; then found=1; fi
    [[ $found -eq 1 ]] && filtered+=("$c")
  done
  if [[ $found -eq 0 ]]; then
    echo "error: --start-from '$START_FROM' is not in the publish list" >&2
    exit 2
  fi
  CRATES=("${filtered[@]}")
fi

cd "$(git rev-parse --show-toplevel)"

echo "publish-crates.sh: ${#CRATES[@]} crate(s) to publish ${DRY_RUN:+(dry run)}"
echo "  inter-crate sleep: ${SLEEP}s"
printf "  order: %s\n" "${CRATES[*]}"
echo

for i in "${!CRATES[@]}"; do
  c="${CRATES[$i]}"
  n=$((i+1))
  total="${#CRATES[@]}"
  echo "[${n}/${total}] cargo publish -p $c $DRY_RUN"
  if ! cargo publish -p "$c" $DRY_RUN; then
    echo
    echo "FAILED at $c. To resume after fixing the issue:" >&2
    echo "  scripts/publish-crates.sh --start-from $c ${DRY_RUN}" >&2
    exit 1
  fi
  # Sleep after every publish except the last so the next iteration
  # respects the rate limit. crates.io needs a few seconds anyway to
  # make the just-published crate visible to its dependents.
  if [[ $n -lt $total && "$SLEEP" -gt 0 ]]; then
    echo "  sleeping ${SLEEP}s before next publish..."
    sleep "$SLEEP"
  fi
done

echo
echo "All ${#CRATES[@]} crate(s) published successfully ${DRY_RUN:+(dry run)}."
