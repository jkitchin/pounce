#!/usr/bin/env bash
# Compile CUTEst SIF problems into shared libraries for the benchmark suite.
#
# Usage:
#   bash prepare.sh                           # all problems in problem_list.txt
#   bash prepare.sh ROSENBR HS35 HS71        # specific problems
#   bash prepare.sh --from-file mylist.txt   # from a custom file
#
# Prerequisites:
#   1. CUTEst toolchain installed (~/.local/cutest/install/bin/sifdecoder, gfortran)
#   2. MASTSIF: the full SIF problem library (~/.local/cutest/mastsif/)
#      Install with: git clone https://github.com/ralna/mastsif ~/.local/cutest/mastsif
#   3. Source the environment: source ~/.local/cutest/env.sh
#
# On Linux:  produces lib<NAME>.so
# On macOS:  produces lib<NAME>.dylib

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBLEMS_DIR="${SCRIPT_DIR}/problems"

# Detect platform shared library extension
case "$(uname -s)" in
    Darwin) SOEXT="dylib" ;;
    *)      SOEXT="so"    ;;
esac

# ---------------------------------------------------------------------------
# Source CUTEst environment
# ---------------------------------------------------------------------------
if [[ -f ~/.local/cutest/env.sh ]]; then
    source ~/.local/cutest/env.sh 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Auto-detect MASTSIF if not set or empty
# ---------------------------------------------------------------------------
if [[ -z "${MASTSIF:-}" ]] || ! ls "${MASTSIF}"/*.SIF &>/dev/null 2>&1; then
    for candidate in \
        "${HOME}/.local/cutest/mastsif" \
        "${HOME}/.local/cutest/sifdecode/sif" \
        "${HOME}/.local/cutest/cutest/sif" \
        "/usr/share/cutest/sif" \
        "/opt/cutest/mastsif"
    do
        if [[ -d "$candidate" ]] && ls "$candidate"/*.SIF &>/dev/null 2>&1; then
            # Only use if it has a meaningful number of SIF files (>20)
            count=$(ls "$candidate"/*.SIF 2>/dev/null | wc -l | tr -d ' ')
            if [[ $count -gt 20 ]]; then
                export MASTSIF="$candidate"
                break
            fi
        fi
    done
fi

# Report MASTSIF status
if [[ -n "${MASTSIF:-}" ]] && ls "${MASTSIF}"/*.SIF &>/dev/null 2>&1; then
    SIF_COUNT=$(ls "${MASTSIF}"/*.SIF 2>/dev/null | wc -l | tr -d ' ')
    echo "MASTSIF: $MASTSIF ($SIF_COUNT SIF files)"
else
    echo "WARNING: MASTSIF not found or empty."
    echo "  To install the full SIF library:"
    echo "    git clone https://github.com/ralna/mastsif ~/.local/cutest/mastsif"
    echo "  Problems without a SIF file will be skipped."
    MASTSIF=""
fi

# ---------------------------------------------------------------------------
# Verify required tools
# ---------------------------------------------------------------------------
MISSING_TOOLS=()
command -v sifdecoder &>/dev/null || MISSING_TOOLS+=("sifdecoder")
command -v gfortran   &>/dev/null || MISSING_TOOLS+=("gfortran")

if [[ ${#MISSING_TOOLS[@]} -gt 0 ]]; then
    echo "ERROR: Required tools not found: ${MISSING_TOOLS[*]}"
    echo "  Install CUTEst toolchain: see install_cutest.sh"
    echo "  Install gfortran: brew install gcc  (macOS) or apt install gfortran  (Linux)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
PROBLEM_NAMES=()
if [[ $# -eq 0 ]]; then
    LIST_FILE="${SCRIPT_DIR}/problem_list.txt"
    if [[ ! -f "$LIST_FILE" ]]; then
        echo "ERROR: No arguments and problem_list.txt not found"
        exit 1
    fi
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(echo "$line" | xargs)"
        [[ -n "$line" ]] && PROBLEM_NAMES+=("$line")
    done < "$LIST_FILE"
elif [[ "$1" == "--from-file" ]]; then
    LIST_FILE="$2"
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(echo "$line" | xargs)"
        [[ -n "$line" ]] && PROBLEM_NAMES+=("$line")
    done < "$LIST_FILE"
else
    PROBLEM_NAMES=("$@")
fi

mkdir -p "$PROBLEMS_DIR"

# Record mastsif commit for reproducibility
MASTSIF_COMMIT="unknown"
if [[ -n "${MASTSIF:-}" && -d "${MASTSIF}/.git" ]]; then
    MASTSIF_COMMIT=$(git -C "$MASTSIF" rev-parse --short HEAD 2>/dev/null || echo "unknown")
fi
echo "mastsif commit: $MASTSIF_COMMIT" > "${PROBLEMS_DIR}/build_info.txt"
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${PROBLEMS_DIR}/build_info.txt"
echo "platform: $(uname -sm)" >> "${PROBLEMS_DIR}/build_info.txt"
echo "gfortran: $(gfortran --version 2>&1 | head -1)" >> "${PROBLEMS_DIR}/build_info.txt"

echo "Preparing ${#PROBLEM_NAMES[@]} CUTEst problems..."

SUCCESS=0
FAIL=0
SKIP=0

for NAME in "${PROBLEM_NAMES[@]}"; do
    DYLIB="${PROBLEMS_DIR}/lib${NAME}.${SOEXT}"
    OUTSDIF="${PROBLEMS_DIR}/${NAME}_OUTSDIF.d"

    # Already prepared
    if [[ -f "$DYLIB" && -f "$OUTSDIF" ]]; then
        echo "  OK   $NAME (already prepared)"
        ((SUCCESS++)) || true
        continue
    fi

    # Find SIF file
    SIF_FILE=""
    if [[ -n "${MASTSIF:-}" && -f "${MASTSIF}/${NAME}.SIF" ]]; then
        SIF_FILE="${MASTSIF}/${NAME}.SIF"
    fi

    if [[ -z "$SIF_FILE" ]]; then
        echo "  SKIP $NAME (${NAME}.SIF not found — install MASTSIF to enable compilation)"
        ((SKIP++)) || true
        continue
    fi

    # Compile from SIF
    WORK_TMPDIR="$(mktemp -d)"
    echo -n "  BUILD $NAME ... "

    if ! (cd "$WORK_TMPDIR" && sifdecoder "$SIF_FILE" > /dev/null 2>&1); then
        echo "FAILED (sifdecoder)"
        rm -rf "$WORK_TMPDIR"
        ((FAIL++)) || true
        continue
    fi

    if [[ ! -f "$WORK_TMPDIR/ELFUN.f" || ! -f "$WORK_TMPDIR/GROUP.f" || ! -f "$WORK_TMPDIR/RANGE.f" ]]; then
        echo "FAILED (missing Fortran files after sifdecoder)"
        rm -rf "$WORK_TMPDIR"
        ((FAIL++)) || true
        continue
    fi

    FORTRAN_FILES=("$WORK_TMPDIR"/*.f)
    if ! gfortran -shared -fPIC -O2 -o "$WORK_TMPDIR/lib${NAME}.${SOEXT}" "${FORTRAN_FILES[@]}" 2>/dev/null; then
        echo "FAILED (gfortran)"
        rm -rf "$WORK_TMPDIR"
        ((FAIL++)) || true
        continue
    fi

    cp "$WORK_TMPDIR/lib${NAME}.${SOEXT}" "$DYLIB"
    cp "$WORK_TMPDIR/OUTSDIF.d" "$OUTSDIF"
    rm -rf "$WORK_TMPDIR"

    echo "OK"
    ((SUCCESS++)) || true
done

echo ""
echo "Done: $SUCCESS prepared, $FAIL failed, $SKIP skipped"
if [[ $SKIP -gt 0 && -z "${MASTSIF:-}" ]]; then
    echo ""
    echo "To prepare all problems, install MASTSIF:"
    echo "  git clone https://github.com/ralna/mastsif ~/.local/cutest/mastsif"
fi
