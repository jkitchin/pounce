#!/bin/bash
# install.sh — Register POUNCE as a GAMS solver.
#
# Usage:
#   ./install.sh                       # uses default GAMS path
#   GAMS_PATH=/path/to/gams ./install.sh
#
# What it does:
#   1. Copies libGamsPounce and libpounce_cinterface into the GAMS system dir
#   2. Adds a POUNCE entry to gmscmpun.txt (the solver registration file)

set -euo pipefail

GAMS_PATH="${GAMS_PATH:-/Library/Frameworks/GAMS.framework/Versions/Current/Resources}"
POUNCE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

UNAME="$(uname -s)"
if [ "$UNAME" = "Darwin" ]; then
    EXT="dylib"
else
    EXT="so"
fi

LINK_LIB="$POUNCE_ROOT/gams/libGamsPounce.$EXT"
SOLVER_LIB="$POUNCE_ROOT/target/release/libpounce_cinterface.$EXT"
CMPTXT="$GAMS_PATH/gmscmpun.txt"

# --- Checks ---

if [ ! -f "$LINK_LIB" ]; then
    echo "Error: $LINK_LIB not found. Run 'make -C gams' first." >&2
    exit 1
fi

if [ ! -f "$SOLVER_LIB" ]; then
    echo "Error: $SOLVER_LIB not found." >&2
    echo "Run 'cargo build --release -p pounce-cinterface' first." >&2
    exit 1
fi

if [ ! -d "$GAMS_PATH" ]; then
    echo "Error: GAMS directory $GAMS_PATH not found." >&2
    exit 1
fi

# --- Copy libraries ---

echo "Copying libGamsPounce.$EXT → $GAMS_PATH/"
cp "$LINK_LIB" "$GAMS_PATH/"

echo "Copying libpounce_cinterface.$EXT → $GAMS_PATH/"
cp "$SOLVER_LIB" "$GAMS_PATH/"

# Fix library paths on macOS so libGamsPounce finds libpounce_cinterface
# in the same directory.
if [ "$UNAME" = "Darwin" ]; then
    OLD_PATH=$(otool -L "$GAMS_PATH/libGamsPounce.dylib" \
        | grep libpounce_cinterface | awk '{print $1}')
    if [ -n "$OLD_PATH" ]; then
        echo "Rewriting libpounce_cinterface path: $OLD_PATH → @loader_path/libpounce_cinterface.dylib"
        install_name_tool -change \
            "$OLD_PATH" \
            "@loader_path/libpounce_cinterface.dylib" \
            "$GAMS_PATH/libGamsPounce.dylib"
    fi

    install_name_tool -id \
        "@loader_path/libGamsPounce.dylib" \
        "$GAMS_PATH/libGamsPounce.dylib" 2>/dev/null || true
fi

# --- Register solver in gmscmpun.txt ---

if grep -q "^POUNCE " "$CMPTXT" 2>/dev/null; then
    echo "POUNCE already registered in $CMPTXT"
else
    echo "Adding POUNCE to $CMPTXT (before DEFAULTS section)"
    sed -i.bak "/^DEFAULTS\$/i\\
POUNCE 11 5 00010203040506070809 1 0 2 NLP DNLP RMINLP\\
gmsgenus.run\\
gmsgenux.out\\
libGamsPounce.$EXT pou 1 0\\
" "$CMPTXT"
    echo "Done."
fi

echo ""
echo "Installation complete. Test with:"
echo "  gams gams/test_hs071.gms"
