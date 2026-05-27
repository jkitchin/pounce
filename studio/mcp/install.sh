#!/usr/bin/env bash
# Install pounce-studio-mcp into a self-contained venv and (optionally) wire
# it into Claude Code. Idempotent: re-running upgrades in place.
#
# Usage:
#   ./install.sh                    # build only
#   ./install.sh --register         # build + `claude mcp add pounce-studio ...`
#   ./install.sh --register --user  # register at user scope (default is local)
#   POUNCE_VENV=/custom/path ./install.sh

set -euo pipefail

# --- Paths -----------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV="${POUNCE_VENV:-$HERE/.venv}"

REGISTER=0
SCOPE=local
SKIP_POUNCE_BUILD=0

for arg in "$@"; do
    case "$arg" in
        --register)            REGISTER=1 ;;
        --user)                SCOPE=user ;;
        --skip-pounce-build)   SKIP_POUNCE_BUILD=1 ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!! \033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; exit 1; }

# --- Prereq checks ---------------------------------------------------------

command -v python3 >/dev/null || die "python3 not found on PATH"
command -v cargo   >/dev/null || die "cargo not found on PATH. Install rustup from https://rustup.rs"

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
case "$PY_VER" in
    3.10|3.11|3.12|3.13) ;;
    *) die "python $PY_VER too old/new — need 3.10+" ;;
esac

# --- Build the `pounce` CLI (used by run_problem) -------------------------

if [[ "$SKIP_POUNCE_BUILD" -eq 0 ]]; then
    if [[ ! -x "$REPO_ROOT/target/release/pounce" ]]; then
        say "Building pounce CLI (cargo build --release -p pounce-cli)"
        (cd "$REPO_ROOT" && cargo build --release -p pounce-cli)
    else
        say "pounce CLI already built at $REPO_ROOT/target/release/pounce"
    fi
fi

# --- Venv setup ------------------------------------------------------------

if [[ ! -d "$VENV" ]]; then
    say "Creating venv at $VENV"
    python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
say "Upgrading pip + installing maturin / mcp / pytest"
pip install --quiet --upgrade pip
pip install --quiet --upgrade 'maturin>=1.4,<2.0' 'mcp>=1.0' 'pytest>=7'

# --- Build native extension + editable install ----------------------------

say "Building native extension and installing pounce-studio-mcp (editable)"
(cd "$HERE" && maturin develop --release)

BIN="$VENV/bin/pounce-studio-mcp"
[[ -x "$BIN" ]] || die "post-install: $BIN missing — check maturin output above"

# --- Smoke test ------------------------------------------------------------

say "Smoke test: import + run tests"
(cd "$HERE" && python -m pytest tests/ -q)

# --- Claude Code registration ---------------------------------------------

if [[ "$REGISTER" -eq 1 ]]; then
    if ! command -v claude >/dev/null; then
        warn "claude CLI not found — skipping registration."
        warn "Install Claude Code (https://claude.com/claude-code), then re-run with --register"
    else
        if claude mcp list 2>/dev/null | grep -q '^pounce-studio'; then
            say "Removing existing pounce-studio MCP registration (scope=$SCOPE)"
            claude mcp remove pounce-studio --scope "$SCOPE" 2>/dev/null || true
        fi
        say "Registering pounce-studio with claude mcp add (scope=$SCOPE)"
        claude mcp add pounce-studio \
            --scope "$SCOPE" \
            --env "POUNCE_BIN=$REPO_ROOT/target/release/pounce" \
            -- "$BIN"
        claude mcp list | sed 's/^/    /'
    fi
fi

# --- Done ------------------------------------------------------------------

cat <<EOF

$(printf '\033[1;32m==> Installed.\033[0m')

  Binary:   $BIN
  Venv:     $VENV
  POUNCE_BIN: $REPO_ROOT/target/release/pounce

Wire into an MCP client manually:

  Claude Code (user scope, recommended):
    claude mcp add pounce-studio --scope user \\
        --env "POUNCE_BIN=$REPO_ROOT/target/release/pounce" \\
        -- "$BIN"

  Claude Desktop / Cursor / Zed — add to the client's MCP config:
    {
      "mcpServers": {
        "pounce-studio": {
          "command": "$BIN",
          "env": { "POUNCE_BIN": "$REPO_ROOT/target/release/pounce" }
        }
      }
    }

To run the standalone server (debug / stdio):
  $BIN
EOF
