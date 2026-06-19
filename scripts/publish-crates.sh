#!/usr/bin/env bash
# Publish POUNCE crates to crates.io in dependency order.
#
# The first publish of all 19 crates will hit the crates.io rate limit
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
# publication. Crates marked `publish = false` (pounce-py, iter-diff,
# pounce-studio-pyo3) are not in this list. NOTE: pounce-studio-core IS
# published as of 0.4.0 — the published pounce-cli took a hard dependency
# on it, so the old "nothing published depends on studio" exclusion no
# longer holds. pounce-nl is likewise required by pounce-cli and published.
# (The benchmark crates pounce-cutest and pounce-large-scale were retired
# when those suites moved to .nl.)

set -euo pipefail

# Topologically sorted: each crate appears only after every crate it
# depends on. Verified against `cargo metadata` — see
# dev-notes/cargo-release.md for the dependency graph. This list is guarded
# by scripts/check-release-consistency.sh (run in CI): it fails the build if
# the set drifts from the workspace's publishable crates or the order stops
# being topological, so keep edits here in sync with the actual members.
CRATES=(
  pounce-common
  pounce-linalg
  pounce-linsol
  pounce-feral
  pounce-hsl
  pounce-nlp
  pounce-l1penalty
  pounce-observability
  pounce-presolve
  pounce-qp
  pounce-solve-report
  pounce-algorithm
  pounce-restoration
  pounce-sensitivity
  pounce-cinterface
  pounce-convex
  pounce-nl
  pounce-lean-cert
  pounce-studio-core
  pounce-cli
  pounce-rs
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

# Version being published. Every publishable crate inherits the workspace
# version (`version.workspace = true`), so the single [workspace.package]
# version in the root Cargo.toml is what each crate will publish as.
TARGET_VERSION="$(grep -m1 -E '^version[[:space:]]*=' Cargo.toml \
  | sed -E 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/')"

# True if <crate>@<version> already exists on crates.io. Used to make real
# uploads idempotent: a version can never be re-published, so a CI run (or a
# resumed run after a mid-batch failure) must skip what is already up rather
# than erroring. NB: crates.io rejects requests without a User-Agent, so a
# missing UA looks like "not published" — always send one.
UA="pounce-crates-publish (https://github.com/jkitchin/pounce)"
crate_version_published() {
  local c="$1" v="$2" code
  code="$(curl -fsS -o /dev/null -w '%{http_code}' \
    -H "User-Agent: $UA" \
    "https://crates.io/api/v1/crates/$c/$v" 2>/dev/null || true)"
  [[ "$code" == "200" ]]
}

echo "publish-crates.sh: ${#CRATES[@]} crate(s) to publish ${DRY_RUN:+(dry run)} @ ${TARGET_VERSION}"
echo "  inter-crate sleep: ${SLEEP}s"
printf "  order: %s\n" "${CRATES[*]}"
echo

# Pre-flight: refuse to start an irreversible batch when any crate carries a
# dependency cargo publish cannot upload (a git pin or version-less dep). Once
# the first crate is live its version can never be re-published, so a mid-batch
# hard-fail leaves a split, un-rollback-able release. Catch it before crate 1.
# (dev-only check; the publish itself is what enforces this per-crate, but by
# then it is too late for the crates already uploaded.)
echo "pre-flight: checking that every crate's dependencies are publishable..."
if ! cargo metadata --format-version 1 2>/dev/null \
  | python3 "$(dirname "$0")/check_dep_publishability.py" "${CRATES[@]}"; then
  echo
  echo "ABORTING before any upload: the crates above have dependencies that" >&2
  echo "cargo publish cannot satisfy. Publishing now would hard-fail mid-batch" >&2
  echo "and leave an irreversible partial release on crates.io. Resolve the" >&2
  echo "dependency (e.g. release & pin a crates.io version) before publishing." >&2
  exit 1
fi
echo

for i in "${!CRATES[@]}"; do
  c="${CRATES[$i]}"
  n=$((i+1))
  total="${#CRATES[@]}"
  # Idempotency: on a real upload, skip a crate already live at this version
  # (dry runs still package every crate so packaging stays fully validated).
  if [[ -z "$DRY_RUN" ]] && crate_version_published "$c" "$TARGET_VERSION"; then
    echo "[${n}/${total}] $c ${TARGET_VERSION} already on crates.io — skipping"
    continue
  fi
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
